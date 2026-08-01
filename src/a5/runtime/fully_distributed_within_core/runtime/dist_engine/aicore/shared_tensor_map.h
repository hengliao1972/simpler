/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include "dist_engine/aicore/primitive.h"
#include "dist_engine/common/atomic.h"
#include "dist_engine/common/state.h"

namespace {

struct DistSharedPaAicoreOps {
    PTO_DEVICE_FUNC static int64_t Load(__gm__ volatile int64_t *address) { return atomic_load(*address); }

    PTO_DEVICE_FUNC static int64_t Exchange(__gm__ volatile int64_t *address, int64_t desired) {
        return atomic_exchange(*address, desired);
    }

    PTO_DEVICE_FUNC static int64_t
    CompareExchange(__gm__ volatile int64_t *address, int64_t expected, int64_t desired) {
        return atomic_compare_exchange(*address, expected, desired);
    }

    PTO_DEVICE_FUNC static int64_t FetchAdd(__gm__ volatile int64_t *address, int64_t delta) {
        return atomic_fetch_add(*address, delta);
    }

    PTO_DEVICE_FUNC static int64_t FetchMax(__gm__ volatile int64_t *address, int64_t desired) {
        return atomic_fetch_max(*address, desired);
    }

    PTO_DEVICE_FUNC static void InvalidateRegion(__gm__ const void *address, uint64_t bytes) {
        dist_aicore_invalidate_region(address, bytes);
    }

    PTO_DEVICE_FUNC static void FlushRegion(__gm__ void *address, uint64_t bytes) {
        dist_aicore_flush_region(address, bytes);
    }

    PTO_DEVICE_FUNC static void PreloadDataCache(__gm__ void *address) {
#if defined(__CCE_AICORE__)
        // Performance hint only. Publication ordering remains the history
        // clean-out + store barrier + return-ready group CAS protocol.
        dc_preload(reinterpret_cast<__gm__ uint64_t *>(address), static_cast<int64_t>(0));
#else
        (void)address;
#endif
    }

    PTO_DEVICE_FUNC static void StoreBarrier() { store_barrier(); }
};

struct DistSharedPaOutputPublishTrace {
    uint64_t copy_begin;
    uint64_t copy_end;
    uint64_t flush_begin;
    uint64_t flush_end;
};

struct DistSharedPaMetadataTrace {
    uint64_t history_dcci_begin;
    uint64_t history_dcci_end;
    uint32_t history_dcci_lines;
    uint64_t group_cas_begin;
    uint64_t group_cas_end;
    bool group_cas_captured;
};

template <typename Ops, bool Observe>
PTO_DEVICE_FUNC int64_t dist_shared_pa_trace_load(
    __gm__ volatile int64_t *address, int32_t task_id, FdwicAtomicSite site
) {
    if constexpr (Observe) {
        return fdwic_trace_atomic_load(task_id, site, *address, /*result_used=*/true);
    }
    return Ops::Load(address);
}

template <typename Ops, bool Observe>
PTO_DEVICE_FUNC int64_t dist_shared_pa_trace_fetch_add(
    __gm__ volatile int64_t *address, int64_t delta, int32_t task_id, FdwicAtomicSite site
) {
    if constexpr (Observe) {
        return fdwic_trace_atomic_fetch_add(
            task_id, site, *address, delta, /*result_used=*/true
        );
    }
    return Ops::FetchAdd(address, delta);
}

template <typename Ops, bool Observe>
PTO_DEVICE_FUNC int64_t dist_shared_pa_trace_fetch_max(
    __gm__ volatile int64_t *address, int64_t desired, int32_t task_id,
    FdwicAtomicSite site
) {
    if constexpr (Observe) {
        return fdwic_trace_atomic_fetch_max(
            task_id, site, *address, desired, /*result_used=*/true
        );
    }
    return Ops::FetchMax(address, desired);
}

template <typename Ops, bool Observe>
PTO_DEVICE_FUNC int64_t dist_shared_pa_trace_exchange(
    __gm__ volatile int64_t *address, int64_t desired, int32_t task_id,
    FdwicAtomicSite site
) {
    if constexpr (Observe) {
        return fdwic_trace_atomic_exchange(
            task_id, site, *address, desired, /*result_used=*/true
        );
    }
    return Ops::Exchange(address, desired);
}

template <typename Ops, bool Observe>
PTO_DEVICE_FUNC void dist_shared_pa_trace_flush(
    __gm__ void *address, uint64_t bytes, int32_t task_id, FdwicDcciSite site,
    bool defer_record, uint64_t *captured_begin = nullptr,
    uint64_t *captured_end = nullptr, uint32_t *captured_lines = nullptr,
    uint64_t known_begin = 0
) {
#if DIST_TRACE_ENABLED
    const bool tracing = Observe && fdwic_atomic_swimlane_enabled();
    const uint64_t begin =
        tracing ? (known_begin != 0 ? known_begin : fdwic_swimlane_detail_now()) : 0;
#else
    constexpr bool tracing = false;
#endif
    Ops::FlushRegion(address, bytes);
#if DIST_TRACE_ENABLED
    if (!tracing) return;
    const uint64_t end = fdwic_swimlane_detail_now();
    const uint32_t lines = fdwic_dcci_region_cache_line_count(address, bytes);
    if (captured_begin != nullptr) *captured_begin = begin;
    if (captured_end != nullptr) *captured_end = end;
    if (captured_lines != nullptr) *captured_lines = lines;
    if (!defer_record) {
        (void)fdwic_swimlane_record_dcci(
            g_self, task_id, -1, site, /*trailing_dsb=*/true, lines, begin, end
        );
    }
#else
    (void)task_id;
    (void)site;
    (void)defer_record;
    (void)captured_begin;
    (void)captured_end;
    (void)captured_lines;
    (void)known_begin;
#endif
}

template <typename Ops, bool Observe>
PTO_DEVICE_FUNC void dist_shared_pa_trace_invalidate(
    __gm__ const void *address, uint64_t bytes, int32_t task_id,
    FdwicDcciSite site
) {
#if DIST_TRACE_ENABLED
    const bool tracing = Observe && fdwic_atomic_swimlane_enabled();
    const uint64_t begin = tracing ? fdwic_swimlane_detail_now() : 0;
#else
    constexpr bool tracing = false;
#endif
    Ops::InvalidateRegion(address, bytes);
#if DIST_TRACE_ENABLED
    if (tracing) {
        const uint64_t end = fdwic_swimlane_detail_now();
        (void)fdwic_swimlane_record_dcci(
            g_self, task_id, -1, site, /*trailing_dsb=*/true,
            fdwic_dcci_region_cache_line_count(address, bytes), begin, end
        );
    }
#else
    (void)task_id;
    (void)site;
#endif
}

PTO_DEVICE_FUNC inline bool dist_shared_pa_output_ref_valid(const FdwicOutputRef &ref) {
    return fdwic_plain_output_ref(ref) &&
           static_cast<uint32_t>(ref.producer_task_id) < kFdwicSharedPaTaskCapacity &&
           static_cast<uint32_t>(ref.output_slot) < kFdwicSharedOutputMaxPerTask;
}

PTO_DEVICE_FUNC inline bool dist_shared_pa_output_key(const FdwicOutputRef &ref, uint32_t &key) {
    if (!dist_shared_pa_output_ref_valid(ref)) return false;
    key = static_cast<uint32_t>(ref.producer_task_id) * kFdwicSharedOutputMaxPerTask +
          static_cast<uint32_t>(ref.output_slot) + 1U;
    return true;
}

PTO_DEVICE_FUNC inline FdwicOutputRef dist_shared_pa_output_ref_from_key(uint32_t key) {
    if (key == 0) return fdwic_invalid_output_ref();
    --key;
    const uint32_t producer = key / kFdwicSharedOutputMaxPerTask;
    const uint32_t slot = key % kFdwicSharedOutputMaxPerTask;
    if (producer >= kFdwicSharedPaTaskCapacity) return fdwic_invalid_output_ref();
    return FdwicOutputRef{static_cast<int32_t>(producer), static_cast<int16_t>(slot), 0, 0, 0, 0};
}

template <typename Ops, bool Observe = false>
PTO_DEVICE_FUNC bool dist_shared_pa_publish_outputs_impl(
    __gm__ SharedPaTensorMapState &state, int32_t task_id, const TaskOutputTensors &outputs,
    uint32_t expected_output_count, DistSharedPaOutputPublishTrace *trace = nullptr
) {
    if (task_id < 0 || static_cast<uint32_t>(task_id) >= kFdwicSharedPaTaskCapacity ||
        outputs.task_id().raw != PTO2TaskId::make(0, static_cast<uint32_t>(task_id)).raw ||
        outputs.size() != expected_output_count || expected_output_count > kFdwicSharedOutputMaxPerTask) {
        return false;
    }
    __gm__ SharedOutputCell &cell = state.shared_outputs[static_cast<uint32_t>(task_id)];

    // One Claim winner owns the whole task cell. Do not pre-read reset state
    // with ordinary cached GM loads: AICPU may reuse and reset the arena
    // between runs while resident AICores still retain an old line. FetchMax
    // below is the authoritative reservation, and all slots are reserved
    // before any descriptor is exposed.
    for (uint32_t slot = 0; slot < expected_output_count; ++slot) {
        if (dist_shared_pa_trace_fetch_max<Ops, Observe>(
                &cell.last_writer[slot].v, static_cast<int64_t>(task_id), task_id,
                FdwicAtomicSite::SharedOutputWriterReserve
            ) != -1) {
            return false;
        }
    }

#if DIST_TRACE_ENABLED
    if (trace != nullptr && fdwic_swimlane_enabled()) {
        trace->copy_begin = fdwic_swimlane_detail_now();
    }
#else
    (void)trace;
#endif
    for (uint32_t slot = 0; slot < expected_output_count; ++slot) {
        Tensor::copy(cell.tensors[slot], outputs.get_ref(slot));
    }
#if DIST_TRACE_ENABLED
    uint64_t copy_boundary = 0;
    if (trace != nullptr && fdwic_swimlane_enabled()) {
        copy_boundary = fdwic_swimlane_detail_now();
        trace->copy_end = copy_boundary;
        trace->flush_begin = copy_boundary;
    }
#endif
    if (expected_output_count != 0) {
        dist_shared_pa_trace_flush<Ops, Observe>(
            &cell.tensors[0], static_cast<uint64_t>(expected_output_count) * sizeof(Tensor),
            task_id, FdwicDcciSite::SharedOutputDescriptorFlush,
            /*defer_record=*/false, nullptr,
            trace == nullptr ? nullptr : &trace->flush_end, nullptr
#if DIST_TRACE_ENABLED
            , copy_boundary
#endif
        );
#if DIST_TRACE_ENABLED
        if (trace != nullptr && fdwic_swimlane_enabled() &&
            !fdwic_atomic_swimlane_enabled()) {
            trace->flush_end = fdwic_swimlane_detail_now();
        }
#endif
#if DIST_TRACE_ENABLED
    } else if (trace != nullptr && fdwic_swimlane_enabled()) {
        trace->flush_end = copy_boundary;
#endif
    }
    Ops::StoreBarrier();
    for (uint32_t slot = 0; slot < expected_output_count; ++slot) {
        if (dist_shared_pa_trace_exchange<Ops, Observe>(
                &cell.published[slot].v, static_cast<int64_t>(task_id), task_id,
                FdwicAtomicSite::SharedOutputPublishedExchange
            ) != -1) {
            return false;
        }
    }
    return true;
}

PTO_DEVICE_FUNC bool dist_shared_pa_publish_outputs(
    __gm__ SharedPaTensorMapState &state, int32_t task_id, const TaskOutputTensors &outputs,
    uint32_t expected_output_count, DistSharedPaOutputPublishTrace *trace = nullptr
) {
    return dist_shared_pa_publish_outputs_impl<DistSharedPaAicoreOps, true>(
        state, task_id, outputs, expected_output_count, trace
    );
}

template <typename Ops>
PTO_DEVICE_FUNC bool dist_shared_pa_copy_output_descriptor_impl(
    __gm__ SharedPaTensorMapState &state, const FdwicOutputRef &ref, __gm__ Tensor &destination
) {
    if (!dist_shared_pa_output_ref_valid(ref)) return false;
    __gm__ SharedOutputCell &cell = state.shared_outputs[static_cast<uint32_t>(ref.producer_task_id)];
    if (Ops::Load(&cell.published[static_cast<uint32_t>(ref.output_slot)].v) != ref.producer_task_id) {
        return false;
    }
    __gm__ Tensor &source = cell.tensors[static_cast<uint32_t>(ref.output_slot)];
    Ops::InvalidateRegion(&source, sizeof(Tensor));
    Tensor::copy(destination, source);
    return destination.owner_task_id.raw ==
           PTO2TaskId::make(0, static_cast<uint32_t>(ref.producer_task_id)).raw;
}

PTO_DEVICE_FUNC inline bool dist_shared_pa_copy_output_descriptor(
    __gm__ SharedPaTensorMapState &state, const FdwicOutputRef &ref, __gm__ Tensor &destination
) {
    return dist_shared_pa_copy_output_descriptor_impl<DistSharedPaAicoreOps>(state, ref, destination);
}

template <typename Ops, bool Observe = false>
PTO_DEVICE_FUNC bool dist_shared_pa_copy_acquired_output_descriptor_impl(
    __gm__ SharedPaTensorMapState &state, const FdwicOutputRef &ref,
    __gm__ Tensor &destination, int32_t reader_task = -1
) {
    if (!dist_shared_pa_output_ref_valid(ref)) return false;
    // CollectFanin has already acquire-validated published and last_writer for
    // every active SharedOutputRef. Descriptors are immutable after publish, so
    // Build only needs to invalidate and copy the descriptor itself.
    __gm__ Tensor &source =
        state.shared_outputs[static_cast<uint32_t>(ref.producer_task_id)]
            .tensors[static_cast<uint32_t>(ref.output_slot)];
    dist_shared_pa_trace_invalidate<Ops, Observe>(
        &source, sizeof(Tensor), reader_task,
        FdwicDcciSite::SharedWinnerBuildDescriptorInvalidate
    );
    Tensor::copy(destination, source);
    return destination.owner_task_id.raw ==
           PTO2TaskId::make(0, static_cast<uint32_t>(ref.producer_task_id)).raw;
}

PTO_DEVICE_FUNC bool dist_shared_pa_copy_acquired_output_descriptor(
    __gm__ SharedPaTensorMapState &state, const FdwicOutputRef &ref,
    __gm__ Tensor &destination, int32_t reader_task
) {
    return dist_shared_pa_copy_acquired_output_descriptor_impl<DistSharedPaAicoreOps, true>(
        state, ref, destination, reader_task
    );
}

struct DistSharedPaHeapReservation {
    uint64_t task_base;
    uint64_t aggregate_vend;
    uint64_t reserved_bytes;
};

PTO_DEVICE_FUNC inline bool dist_shared_pa_aligned(uint64_t value) {
    return (value & (static_cast<uint64_t>(PTO2_PACKED_OUTPUT_ALIGN) - 1U)) == 0;
}

/**
 * Reserve output memory from eight fixed 32-MiB shards.
 *
 * The cursors are absolute within one run and never wrap. A failed post-atomic
 * check is terminal; rolling a cursor back would race a later winner and turn
 * a capacity failure into overlapping live allocations.
 */
template <typename Ops, bool Observe = false>
PTO_DEVICE_FUNC bool dist_shared_pa_reserve_heap_impl(
    __gm__ SharedPaTensorMapState &state, int32_t task_id, uint64_t output_bytes, uint64_t heap_size,
    DistSharedPaHeapReservation &reservation
) {
    reservation = DistSharedPaHeapReservation{0, 0, 0};
    if (task_id < 0 || static_cast<uint32_t>(task_id) >= kFdwicSharedPaTaskCapacity ||
        heap_size != kFdwicSharedHeapBytes) {
        return false;
    }
    const int64_t current_vend = dist_shared_pa_trace_load<Ops, Observe>(
        &state.shared_heap_vend.v, task_id, FdwicAtomicSite::SharedHeapVendLoad
    );
    if (current_vend < 0 || !dist_shared_pa_aligned(static_cast<uint64_t>(current_vend)) ||
        static_cast<uint64_t>(current_vend) > kFdwicSharedHeapBytes) {
        return false;
    }
    if (output_bytes == 0) {
        reservation.aggregate_vend = static_cast<uint64_t>(current_vend);
        return true;
    }
    if (output_bytes > UINT64_MAX - (PTO2_PACKED_OUTPUT_ALIGN - 1U)) return false;
    const uint64_t reserve = PTO2_ALIGN_UP(output_bytes, PTO2_PACKED_OUTPUT_ALIGN);
    if (reserve == 0 || reserve > kFdwicSharedHeapShardBytes ||
        static_cast<uint64_t>(current_vend) > kFdwicSharedHeapBytes - reserve) {
        return false;
    }

    const uint32_t shard = static_cast<uint32_t>(task_id) & (kFdwicSharedHeapShards - 1U);
    __gm__ volatile int64_t *cursor_address = &state.shared_heap_cursor[shard].v;
    const int64_t cursor_snapshot = dist_shared_pa_trace_load<Ops, Observe>(
        cursor_address, task_id, FdwicAtomicSite::SharedHeapCursorLoad
    );
    if (cursor_snapshot < 0 || !dist_shared_pa_aligned(static_cast<uint64_t>(cursor_snapshot)) ||
        static_cast<uint64_t>(cursor_snapshot) > kFdwicSharedHeapShardBytes - reserve) {
        return false;
    }
    const int64_t cursor_observed = dist_shared_pa_trace_fetch_add<Ops, Observe>(
        cursor_address, static_cast<int64_t>(reserve), task_id,
        FdwicAtomicSite::SharedHeapCursorReserve
    );
    if (cursor_observed < 0 || !dist_shared_pa_aligned(static_cast<uint64_t>(cursor_observed)) ||
        static_cast<uint64_t>(cursor_observed) > kFdwicSharedHeapShardBytes - reserve) {
        return false;
    }

    const int64_t vend_observed = dist_shared_pa_trace_fetch_add<Ops, Observe>(
        &state.shared_heap_vend.v, static_cast<int64_t>(reserve), task_id,
        FdwicAtomicSite::SharedHeapVendAdvance
    );
    if (vend_observed < 0 || !dist_shared_pa_aligned(static_cast<uint64_t>(vend_observed)) ||
        static_cast<uint64_t>(vend_observed) > kFdwicSharedHeapBytes - reserve) {
        return false;
    }
    reservation.task_base =
        static_cast<uint64_t>(shard) * kFdwicSharedHeapShardBytes + static_cast<uint64_t>(cursor_observed);
    reservation.aggregate_vend = static_cast<uint64_t>(vend_observed) + reserve;
    reservation.reserved_bytes = reserve;
    return true;
}

PTO_DEVICE_FUNC bool dist_shared_pa_reserve_heap(
    __gm__ SharedPaTensorMapState &state, int32_t task_id, uint64_t output_bytes, uint64_t heap_size,
    DistSharedPaHeapReservation &reservation
) {
    return dist_shared_pa_reserve_heap_impl<DistSharedPaAicoreOps, true>(
        state, task_id, output_bytes, heap_size, reservation
    );
}

template <typename Ops, bool Observe = false>
PTO_DEVICE_FUNC bool dist_shared_pa_prepare_up_history_impl(
    __gm__ SharedPaTensorMapState &state, int32_t task_id, int32_t batch_start,
    const uint32_t symbol_keys[3], DistSharedPaMetadataTrace *trace = nullptr
) {
    if (task_id < 0 || static_cast<uint32_t>(task_id) >= kFdwicSharedPaTaskCapacity || batch_start < 0 ||
        batch_start >= task_id || task_id != batch_start + 4 || symbol_keys == nullptr) {
        return false;
    }
    const uint32_t key_base = static_cast<uint32_t>(batch_start) * kFdwicSharedOutputMaxPerTask + 1U;
    if (symbol_keys[0] != key_base + 2U || symbol_keys[1] != key_base + 1U || symbol_keys[2] != key_base) {
        return false;
    }

    __gm__ SharedWriterHistoryCell &history = state.writer_history[static_cast<uint32_t>(task_id)];
    for (uint32_t index = 0; index < 3; ++index) {
        history.entries[index].symbol_key = symbol_keys[index];
        history.entries[index].previous_writer = batch_start;
    }
    history.magic = kFdwicSharedWriterHistoryMagic;
    history.writer_task = task_id;
    history.count = 3;
    history.reserved = 0;
    constexpr uint64_t kPublishedHistoryBytes =
        offsetof(SharedWriterHistoryCell, entries) + 3U * sizeof(SharedWriterHistoryRecord);
    dist_shared_pa_trace_flush<Ops, Observe>(
        &history, kPublishedHistoryBytes, task_id,
        FdwicDcciSite::SharedWriterHistoryFlush, /*defer_record=*/true,
        trace == nullptr ? nullptr : &trace->history_dcci_begin,
        trace == nullptr ? nullptr : &trace->history_dcci_end,
        trace == nullptr ? nullptr : &trace->history_dcci_lines
    );
    Ops::StoreBarrier();
    return true;
}

PTO_DEVICE_FUNC bool dist_shared_pa_prepare_up_history(
    __gm__ SharedPaTensorMapState &state, int32_t task_id, int32_t batch_start,
    const uint32_t symbol_keys[3], DistSharedPaMetadataTrace *trace = nullptr
) {
    return dist_shared_pa_prepare_up_history_impl<DistSharedPaAicoreOps, true>(
        state, task_id, batch_start, symbol_keys, trace
    );
}

template <typename Ops, bool Capture = false>
PTO_DEVICE_FUNC bool dist_shared_pa_commit_up_group_writer_impl(
    __gm__ SharedPaTensorMapState &state, int32_t task_id, int32_t batch_start,
    DistSharedPaMetadataTrace *trace = nullptr
) {
    if (task_id != batch_start + 4 || batch_start < 0 ||
        static_cast<uint32_t>(task_id) >= kFdwicSharedPaTaskCapacity) {
        return false;
    }
    __gm__ volatile int64_t *last_writer =
        &state.shared_outputs[static_cast<uint32_t>(batch_start)].last_writer[0].v;
#if DIST_TRACE_ENABLED
    if constexpr (Capture) {
        if (trace != nullptr && fdwic_atomic_swimlane_enabled()) {
            trace->group_cas_begin = fdwic_swimlane_detail_now();
            const int64_t observed = Ops::CompareExchange(
                last_writer, static_cast<int64_t>(batch_start),
                static_cast<int64_t>(task_id)
            );
            trace->group_cas_end = fdwic_atomic_result_ready_tick(observed);
            trace->group_cas_captured = true;
            return observed == batch_start;
        }
    }
#else
    (void)trace;
#endif
    if constexpr (Capture) {
        return fdwic_trace_atomic_compare_exchange(
                   task_id, FdwicAtomicSite::SharedMetadataLastWriterCommit,
                   *last_writer, static_cast<int64_t>(batch_start),
                   static_cast<int64_t>(task_id), /*result_used=*/true
               ) == batch_start;
    }
    return Ops::CompareExchange(
               last_writer, static_cast<int64_t>(batch_start),
               static_cast<int64_t>(task_id)
           ) == batch_start;
}

PTO_DEVICE_FUNC bool dist_shared_pa_commit_up_group_writer(
    __gm__ SharedPaTensorMapState &state, int32_t task_id, int32_t batch_start,
    DistSharedPaMetadataTrace *trace = nullptr
) {
    return dist_shared_pa_commit_up_group_writer_impl<DistSharedPaAicoreOps, true>(
        state, task_id, batch_start, trace
    );
}

/**
 * Resolve the newest writer strictly earlier than reader_task.
 *
 * PA's three accumulator symbols share Alloc.slot0 as their latest-writer
 * cache. If the cache already points at this UP task, the immutable history
 * cell supplies each symbol's previous writer.
 */
template <typename Ops, bool Observe = false>
PTO_DEVICE_FUNC bool dist_shared_pa_resolve_writer_impl(
    __gm__ SharedPaTensorMapState &state, const FdwicOutputRef &ref, int32_t reader_task,
    int32_t history_window, int32_t batch_start, int32_t &writer
) {
    writer = -1;
    if (!dist_shared_pa_output_ref_valid(ref) || reader_task <= ref.producer_task_id || history_window < 0 ||
        batch_start < 0 || batch_start > ref.producer_task_id) {
        return false;
    }
    __gm__ SharedOutputCell &origin =
        state.shared_outputs[static_cast<uint32_t>(ref.producer_task_id)];
    if (dist_shared_pa_trace_load<Ops, Observe>(
            &origin.published[static_cast<uint32_t>(ref.output_slot)].v, reader_task,
            FdwicAtomicSite::SharedFaninOutputPublishedLoad
        ) != ref.producer_task_id) {
        return false;
    }
    const uint32_t latest_slot =
        ref.producer_task_id == batch_start && ref.output_slot < 3 ? 0U : static_cast<uint32_t>(ref.output_slot);
    int64_t latest = dist_shared_pa_trace_load<Ops, Observe>(
        &origin.last_writer[latest_slot].v, reader_task,
        FdwicAtomicSite::SharedFaninLastWriterLoad
    );
    uint32_t steps = 0;
    uint32_t symbol_key = 0;
    if (!dist_shared_pa_output_key(ref, symbol_key)) return false;
    while (latest >= reader_task) {
        if (latest < 0 || static_cast<uint32_t>(latest) >= kFdwicSharedPaTaskCapacity ||
            steps++ >= kFdwicSharedPaTaskCapacity) {
            return false;
        }
        __gm__ SharedWriterHistoryCell &cell = state.writer_history[static_cast<uint32_t>(latest)];
        // The retained PA fast path publishes three records, all in the first
        // cache line. Reject any other shape instead of reading an unproven
        // continuation line.
        dist_shared_pa_trace_invalidate<Ops, Observe>(
            &cell, kCacheLine, reader_task,
            FdwicDcciSite::SharedFaninHistoryInvalidate
        );
        if (cell.magic != kFdwicSharedWriterHistoryMagic || cell.writer_task != latest || cell.count != 3 ||
            cell.reserved != 0) {
            return false;
        }
        bool found = false;
        int32_t previous = -1;
        for (uint32_t index = 0; index < cell.count; ++index) {
            if (cell.entries[index].symbol_key != symbol_key) continue;
            if (found) return false;
            found = true;
            previous = cell.entries[index].previous_writer;
        }
        if (!found || previous < ref.producer_task_id || previous >= latest) return false;
        latest = previous;
    }
    if (latest < ref.producer_task_id || latest >= reader_task) return false;
    const int32_t lower = reader_task > history_window ? reader_task - history_window : 0;
    writer = latest < lower ? -1 : static_cast<int32_t>(latest);
    return true;
}

PTO_DEVICE_FUNC bool dist_shared_pa_resolve_writer(
    __gm__ SharedPaTensorMapState &state, const FdwicOutputRef &ref, int32_t reader_task,
    int32_t history_window, int32_t batch_start, int32_t &writer
) {
    return dist_shared_pa_resolve_writer_impl<DistSharedPaAicoreOps, true>(
        state, ref, reader_task, history_window, batch_start, writer
    );
}

}  // namespace
