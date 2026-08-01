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

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "data_type.h"
#include "intrinsic.h"
#include "inner_kernel.h"
#include "pto_runtime2.h"
#include "runtime.h"
#include "dist_engine/common/state.h"
#include "dist_engine/common/trace.h"
#include "dist_engine/common/swimlane.h"
#include "dist_engine/aicore/shared_tensor_map.h"
#include "dist_engine/aicpu/shared_tensor_map_init.h"

[[noreturn]] void assert_impl(const char *condition, const char *, int) { throw std::logic_error(condition); }

namespace {

static_assert(PTO_FDWIC_SHARED_MAP == 1, "shared PA heap tests require the shared artifact identity");

struct HostHeapOps {
    static inline uint32_t compare_exchange_calls = 0;
    static inline uint32_t flush_calls = 0;
    static inline uint32_t invalidate_calls = 0;
    static inline uint32_t store_barrier_calls = 0;

    static void ResetObservations() {
        compare_exchange_calls = 0;
        flush_calls = 0;
        invalidate_calls = 0;
        store_barrier_calls = 0;
    }

    static int64_t Load(volatile int64_t *address) { return __atomic_load_n(address, __ATOMIC_SEQ_CST); }

    static int64_t Exchange(volatile int64_t *address, int64_t desired) {
        return __atomic_exchange_n(address, desired, __ATOMIC_SEQ_CST);
    }

    static int64_t CompareExchange(volatile int64_t *address, int64_t expected, int64_t desired) {
        ++compare_exchange_calls;
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(
            address, &observed, desired, /*weak=*/false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
        );
        return observed;
    }

    static int64_t FetchAdd(volatile int64_t *address, int64_t delta) {
        return __atomic_fetch_add(address, delta, __ATOMIC_SEQ_CST);
    }

    static int64_t FetchMax(volatile int64_t *address, int64_t desired) {
        int64_t observed = Load(address);
        while (desired > observed) {
            if (__atomic_compare_exchange_n(
                    address, &observed, desired, /*weak=*/true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
                )) {
                return observed;
            }
        }
        return observed;
    }

    static void InvalidateRegion(const void *, uint64_t) { ++invalidate_calls; }

    static void FlushRegion(void *, uint64_t) { ++flush_calls; }

    static void StoreBarrier() {
        ++store_barrier_calls;
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }
};

std::unique_ptr<SharedPaTensorMapState> make_empty_state() {
    auto state = std::make_unique<SharedPaTensorMapState>();
    dist_shared_pa_tensor_map_reset(*state);
    return state;
}

bool reserve(
    SharedPaTensorMapState &state, int32_t task_id, uint64_t bytes, DistSharedPaHeapReservation &reservation,
    uint64_t heap_size = kFdwicSharedHeapBytes
) {
    return dist_shared_pa_reserve_heap_impl<HostHeapOps>(state, task_id, bytes, heap_size, reservation);
}

Tensor make_owned_tensor(int32_t task_id, uint32_t slot, uintptr_t address_base = 0x100000) {
    const uint32_t shape[2] = {slot + 2U, 4U};
    Tensor tensor = make_tensor_external(
        reinterpret_cast<void *>(address_base + slot * 0x1000U), shape, 2, DataType::FLOAT32,
        /*manual_dep=*/false, /*version=*/static_cast<int32_t>(100U + slot)
    );
    tensor.owner_task_id = PTO2TaskId::make(0, static_cast<uint32_t>(task_id));
    return tensor;
}

void make_task_outputs(
    int32_t task_id, uint32_t output_count, Tensor tensors[kFdwicSharedOutputMaxPerTask],
    TaskOutputTensors &outputs, uintptr_t address_base = 0x100000
) {
    outputs.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(task_id)));
    for (uint32_t slot = 0; slot < output_count; ++slot) {
        tensors[slot] = make_owned_tensor(task_id, slot, address_base);
        outputs.materialize_output(tensors[slot]);
    }
}

void expect_same_descriptor(const Tensor &actual, const Tensor &expected) {
    EXPECT_EQ(actual.buffer.addr, expected.buffer.addr);
    EXPECT_EQ(actual.buffer.size, expected.buffer.size);
    EXPECT_EQ(actual.owner_task_id.raw, expected.owner_task_id.raw);
    EXPECT_EQ(actual.start_offset, expected.start_offset);
    EXPECT_EQ(actual.version, expected.version);
    EXPECT_EQ(actual.ndims, expected.ndims);
    EXPECT_EQ(actual.dtype, expected.dtype);
    EXPECT_EQ(actual.manual_dep, expected.manual_dep);
    EXPECT_EQ(actual.is_contiguous, expected.is_contiguous);
    EXPECT_EQ(actual.extent_elem_cache, expected.extent_elem_cache);
    for (uint32_t dimension = 0; dimension < expected.ndims; ++dimension) {
        EXPECT_EQ(actual.shapes[dimension], expected.shapes[dimension]) << "dimension=" << dimension;
        EXPECT_EQ(actual.strides[dimension], expected.strides[dimension]) << "dimension=" << dimension;
    }
}

TEST(FdwicSharedPaOutputRef, ProductionKeyConversionChecksPlainFormAndBounds) {
    uint32_t key = 0;
    const FdwicOutputRef first{0, 0, 0, 0, 0, 0};
    ASSERT_TRUE(dist_shared_pa_output_key(first, key));
    EXPECT_EQ(key, 1U);
    EXPECT_EQ(dist_shared_pa_output_ref_from_key(key).producer_task_id, 0);
    EXPECT_EQ(dist_shared_pa_output_ref_from_key(key).output_slot, 0);

    const FdwicOutputRef last{
        static_cast<int32_t>(kFdwicSharedPaTaskCapacity - 1),
        static_cast<int16_t>(kFdwicSharedOutputMaxPerTask - 1),
        0,
        0,
        0,
        0,
    };
    ASSERT_TRUE(dist_shared_pa_output_key(last, key));
    EXPECT_EQ(key, kFdwicSharedPaTaskCapacity * kFdwicSharedOutputMaxPerTask);
    const FdwicOutputRef round_trip = dist_shared_pa_output_ref_from_key(key);
    EXPECT_EQ(round_trip.producer_task_id, last.producer_task_id);
    EXPECT_EQ(round_trip.output_slot, last.output_slot);

    FdwicOutputRef bad = last;
    bad.output_slot = static_cast<int16_t>(kFdwicSharedOutputMaxPerTask);
    EXPECT_FALSE(dist_shared_pa_output_key(bad, key));
    bad = last;
    bad.producer_task_id = static_cast<int32_t>(kFdwicSharedPaTaskCapacity);
    EXPECT_FALSE(dist_shared_pa_output_key(bad, key));
    bad = last;
    bad.view_ndims = 1;
    EXPECT_FALSE(dist_shared_pa_output_key(bad, key));

    EXPECT_FALSE(fdwic_plain_output_ref(dist_shared_pa_output_ref_from_key(0)));
    EXPECT_FALSE(fdwic_plain_output_ref(
        dist_shared_pa_output_ref_from_key(kFdwicSharedPaTaskCapacity * kFdwicSharedOutputMaxPerTask + 1)
    ));
}

TEST(FdwicSharedPaOutputDescriptor, FreshPublicationCanBeCopiedAndDuplicatePublisherIsRejected) {
    auto state = make_empty_state();
    constexpr int32_t kTask = 7;
    constexpr uint32_t kOutputCount = 3;
    Tensor tensors[kFdwicSharedOutputMaxPerTask]{};
    TaskOutputTensors outputs;
    make_task_outputs(kTask, kOutputCount, tensors, outputs);

    HostHeapOps::ResetObservations();
    ASSERT_TRUE(dist_shared_pa_publish_outputs_impl<HostHeapOps>(*state, kTask, outputs, kOutputCount));
    EXPECT_EQ(HostHeapOps::flush_calls, 1U);
    EXPECT_EQ(HostHeapOps::store_barrier_calls, 1U);
    for (uint32_t slot = 0; slot < kOutputCount; ++slot) {
        EXPECT_EQ(state->shared_outputs[kTask].last_writer[slot].v, kTask) << "slot=" << slot;
        EXPECT_EQ(state->shared_outputs[kTask].published[slot].v, kTask) << "slot=" << slot;

        Tensor copied{};
        ASSERT_TRUE(dist_shared_pa_copy_output_descriptor_impl<HostHeapOps>(
            *state, FdwicOutputRef{kTask, static_cast<int16_t>(slot), 0, 0, 0, 0}, copied
        ));
        expect_same_descriptor(copied, tensors[slot]);
    }
    EXPECT_EQ(HostHeapOps::invalidate_calls, kOutputCount);

    Tensor replacements[kFdwicSharedOutputMaxPerTask]{};
    TaskOutputTensors duplicate;
    make_task_outputs(kTask, kOutputCount, replacements, duplicate, 0x900000);
    EXPECT_FALSE(dist_shared_pa_publish_outputs_impl<HostHeapOps>(
        *state, kTask, duplicate, kOutputCount
    ));

    Tensor copied_after_duplicate{};
    ASSERT_TRUE(dist_shared_pa_copy_output_descriptor_impl<HostHeapOps>(
        *state, FdwicOutputRef{kTask, 0, 0, 0, 0, 0}, copied_after_duplicate
    ));
    expect_same_descriptor(copied_after_duplicate, tensors[0]);
}

TEST(FdwicSharedPaOutputDescriptor, UnpublishedAndCorruptInputsFailClosed) {
    auto state = make_empty_state();
    constexpr int32_t kTask = 9;
    Tensor destination{};
    EXPECT_FALSE(dist_shared_pa_copy_output_descriptor_impl<HostHeapOps>(
        *state, FdwicOutputRef{kTask, 0, 0, 0, 0, 0}, destination
    ));
    EXPECT_FALSE(dist_shared_pa_copy_output_descriptor_impl<HostHeapOps>(
        *state, FdwicOutputRef{kTask, 0, 0, 1, 1, 0}, destination
    ));

    Tensor tensors[kFdwicSharedOutputMaxPerTask]{};
    TaskOutputTensors outputs;
    make_task_outputs(kTask, 1, tensors, outputs);
    EXPECT_FALSE(dist_shared_pa_publish_outputs_impl<HostHeapOps>(*state, kTask, outputs, 2));

    outputs.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(kTask + 1)));
    EXPECT_FALSE(dist_shared_pa_publish_outputs_impl<HostHeapOps>(*state, kTask, outputs, 1));
    EXPECT_EQ(state->shared_outputs[kTask].published[0].v, -1);
    EXPECT_EQ(state->shared_outputs[kTask].last_writer[0].v, -1);

    outputs.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(kTask)));
    ASSERT_TRUE(dist_shared_pa_publish_outputs_impl<HostHeapOps>(*state, kTask, outputs, 1));
    state->shared_outputs[kTask].tensors[0].owner_task_id = PTO2TaskId::invalid();
    EXPECT_FALSE(dist_shared_pa_copy_output_descriptor_impl<HostHeapOps>(
        *state, FdwicOutputRef{kTask, 0, 0, 0, 0, 0}, destination
    ));
}

TEST(FdwicSharedPaWriterHistory, UpPublishesThreeRecordsAndCommitsOneGroupWriterCas) {
    auto state = make_empty_state();
    constexpr int32_t kAlloc = 0;
    constexpr int32_t kUp = 4;
    constexpr uint32_t kOutputCount = 3;
    Tensor tensors[kFdwicSharedOutputMaxPerTask]{};
    TaskOutputTensors alloc_outputs;
    make_task_outputs(kAlloc, kOutputCount, tensors, alloc_outputs);
    ASSERT_TRUE(dist_shared_pa_publish_outputs_impl<HostHeapOps>(
        *state, kAlloc, alloc_outputs, kOutputCount
    ));

    uint32_t keys[kOutputCount] = {};
    ASSERT_TRUE(dist_shared_pa_output_key(FdwicOutputRef{kAlloc, 2, 0, 0, 0, 0}, keys[0]));
    ASSERT_TRUE(dist_shared_pa_output_key(FdwicOutputRef{kAlloc, 1, 0, 0, 0, 0}, keys[1]));
    ASSERT_TRUE(dist_shared_pa_output_key(FdwicOutputRef{kAlloc, 0, 0, 0, 0, 0}, keys[2]));

    HostHeapOps::ResetObservations();
    ASSERT_TRUE(dist_shared_pa_prepare_up_history_impl<HostHeapOps>(*state, kUp, kAlloc, keys));
    EXPECT_EQ(HostHeapOps::flush_calls, 1U);
    EXPECT_EQ(HostHeapOps::store_barrier_calls, 1U);
    const SharedWriterHistoryCell &history = state->writer_history[kUp];
    EXPECT_EQ(history.magic, kFdwicSharedWriterHistoryMagic);
    EXPECT_EQ(history.writer_task, kUp);
    ASSERT_EQ(history.count, kOutputCount);
    EXPECT_EQ(history.reserved, 0U);
    for (uint32_t index = 0; index < kOutputCount; ++index) {
        EXPECT_EQ(history.entries[index].symbol_key, keys[index]) << "index=" << index;
        EXPECT_EQ(history.entries[index].previous_writer, kAlloc) << "index=" << index;
    }

    HostHeapOps::ResetObservations();
    ASSERT_TRUE(dist_shared_pa_commit_up_group_writer_impl<HostHeapOps>(*state, kUp, kAlloc));
    EXPECT_EQ(HostHeapOps::compare_exchange_calls, 1U);
    EXPECT_EQ(state->shared_outputs[kAlloc].last_writer[0].v, kUp);
    EXPECT_EQ(state->shared_outputs[kAlloc].last_writer[1].v, kAlloc);
    EXPECT_EQ(state->shared_outputs[kAlloc].last_writer[2].v, kAlloc);

    HostHeapOps::ResetObservations();
    EXPECT_FALSE(dist_shared_pa_commit_up_group_writer_impl<HostHeapOps>(*state, kUp, kAlloc));
    EXPECT_EQ(HostHeapOps::compare_exchange_calls, 1U);
    EXPECT_EQ(state->shared_outputs[kAlloc].last_writer[0].v, kUp);

    for (int16_t slot = 0; slot < static_cast<int16_t>(kOutputCount); ++slot) {
        const FdwicOutputRef ref{kAlloc, slot, 0, 0, 0, 0};
        int32_t writer = -2;
        ASSERT_TRUE(dist_shared_pa_resolve_writer_impl<HostHeapOps>(
            *state, ref, kUp, /*history_window=*/4, kAlloc, writer
        )) << "slot=" << slot;
        EXPECT_EQ(writer, kAlloc) << "slot=" << slot;

        writer = -2;
        ASSERT_TRUE(dist_shared_pa_resolve_writer_impl<HostHeapOps>(
            *state, ref, kUp, /*history_window=*/3, kAlloc, writer
        )) << "slot=" << slot;
        EXPECT_EQ(writer, -1) << "slot=" << slot;
    }
}

TEST(FdwicSharedPaWriterHistory, UnpublishedOrCorruptHistoryFailsClosed) {
    auto state = make_empty_state();
    constexpr int32_t kAlloc = 0;
    constexpr int32_t kUp = 4;
    const FdwicOutputRef slot_zero{kAlloc, 0, 0, 0, 0, 0};
    int32_t writer = 123;
    EXPECT_FALSE(dist_shared_pa_resolve_writer_impl<HostHeapOps>(
        *state, slot_zero, kUp, /*history_window=*/4, kAlloc, writer
    ));
    EXPECT_EQ(writer, -1);

    Tensor tensors[kFdwicSharedOutputMaxPerTask]{};
    TaskOutputTensors alloc_outputs;
    make_task_outputs(kAlloc, 3, tensors, alloc_outputs);
    ASSERT_TRUE(dist_shared_pa_publish_outputs_impl<HostHeapOps>(*state, kAlloc, alloc_outputs, 3));

    uint32_t keys[3] = {};
    ASSERT_TRUE(dist_shared_pa_output_key(FdwicOutputRef{kAlloc, 2, 0, 0, 0, 0}, keys[0]));
    ASSERT_TRUE(dist_shared_pa_output_key(FdwicOutputRef{kAlloc, 1, 0, 0, 0, 0}, keys[1]));
    ASSERT_TRUE(dist_shared_pa_output_key(slot_zero, keys[2]));
    const uint32_t corrupt_keys[3] = {keys[0], keys[1], keys[1]};
    EXPECT_FALSE(dist_shared_pa_prepare_up_history_impl<HostHeapOps>(
        *state, kUp, kAlloc, corrupt_keys
    ));
    EXPECT_EQ(state->writer_history[kUp].magic, 0U);

    ASSERT_TRUE(dist_shared_pa_prepare_up_history_impl<HostHeapOps>(*state, kUp, kAlloc, keys));
    ASSERT_TRUE(dist_shared_pa_commit_up_group_writer_impl<HostHeapOps>(*state, kUp, kAlloc));

    state->writer_history[kUp].magic = 0;
    EXPECT_FALSE(dist_shared_pa_resolve_writer_impl<HostHeapOps>(
        *state, slot_zero, kUp, /*history_window=*/4, kAlloc, writer
    ));
    EXPECT_EQ(writer, -1);

    state->writer_history[kUp].magic = kFdwicSharedWriterHistoryMagic;
    state->writer_history[kUp].entries[2].symbol_key = keys[1];
    EXPECT_FALSE(dist_shared_pa_resolve_writer_impl<HostHeapOps>(
        *state, slot_zero, kUp, /*history_window=*/4, kAlloc, writer
    ));
    EXPECT_EQ(writer, -1);
}

TEST(FdwicSharedPaHeap, EightTaskModuloShardsHaveDisjointFixedRanges) {
    auto state = make_empty_state();

    for (uint32_t task = 0; task < kFdwicSharedHeapShards; ++task) {
        DistSharedPaHeapReservation reservation{};
        ASSERT_TRUE(reserve(*state, static_cast<int32_t>(task), 1, reservation)) << "task=" << task;
        EXPECT_EQ(reservation.task_base, task * kFdwicSharedHeapShardBytes);
        EXPECT_EQ(reservation.aggregate_vend, (task + 1U) * PTO2_PACKED_OUTPUT_ALIGN);
        EXPECT_EQ(reservation.reserved_bytes, PTO2_PACKED_OUTPUT_ALIGN);
        EXPECT_EQ(state->shared_heap_cursor[task].v, PTO2_PACKED_OUTPUT_ALIGN);
    }

    DistSharedPaHeapReservation second_on_shard_zero{};
    ASSERT_TRUE(reserve(*state, 8, PTO2_PACKED_OUTPUT_ALIGN + 1, second_on_shard_zero));
    EXPECT_EQ(second_on_shard_zero.task_base, PTO2_PACKED_OUTPUT_ALIGN);
    EXPECT_EQ(second_on_shard_zero.reserved_bytes, 2U * PTO2_PACKED_OUTPUT_ALIGN);
    EXPECT_EQ(state->shared_heap_cursor[0].v, 3U * PTO2_PACKED_OUTPUT_ALIGN);
    for (uint32_t shard = 1; shard < kFdwicSharedHeapShards; ++shard) {
        EXPECT_EQ(state->shared_heap_cursor[shard].v, PTO2_PACKED_OUTPUT_ALIGN);
    }
}

TEST(FdwicSharedPaHeap, FullFixedHeapDoesNotWrapOrReuseAnyShard) {
    auto state = make_empty_state();

    for (uint32_t task = 0; task < kFdwicSharedHeapShards; ++task) {
        DistSharedPaHeapReservation reservation{};
        ASSERT_TRUE(reserve(*state, static_cast<int32_t>(task), kFdwicSharedHeapShardBytes, reservation))
            << "task=" << task;
        EXPECT_EQ(reservation.task_base, task * kFdwicSharedHeapShardBytes);
        EXPECT_EQ(reservation.aggregate_vend, (task + 1U) * kFdwicSharedHeapShardBytes);
        EXPECT_EQ(reservation.reserved_bytes, kFdwicSharedHeapShardBytes);
    }
    EXPECT_EQ(state->shared_heap_vend.v, static_cast<int64_t>(kFdwicSharedHeapBytes));

    DistSharedPaHeapReservation zero_bytes{};
    ASSERT_TRUE(reserve(*state, 8, 0, zero_bytes));
    EXPECT_EQ(zero_bytes.task_base, 0U);
    EXPECT_EQ(zero_bytes.aggregate_vend, kFdwicSharedHeapBytes);
    EXPECT_EQ(zero_bytes.reserved_bytes, 0U);

    DistSharedPaHeapReservation overflow{1, 2, 3};
    EXPECT_FALSE(reserve(*state, 8, 1, overflow));
    EXPECT_EQ(overflow.task_base, 0U);
    EXPECT_EQ(overflow.aggregate_vend, 0U);
    EXPECT_EQ(overflow.reserved_bytes, 0U);
    EXPECT_EQ(state->shared_heap_cursor[0].v, static_cast<int64_t>(kFdwicSharedHeapShardBytes));
    EXPECT_EQ(state->shared_heap_vend.v, static_cast<int64_t>(kFdwicSharedHeapBytes));
}

TEST(FdwicSharedPaHeap, InvalidShapeFailsWithoutMutatingControls) {
    auto state = make_empty_state();

    const auto expect_unchanged = [&] {
        EXPECT_EQ(state->shared_heap_vend.v, 0);
        for (uint32_t shard = 0; shard < kFdwicSharedHeapShards; ++shard) {
            EXPECT_EQ(state->shared_heap_cursor[shard].v, 0) << "shard=" << shard;
        }
    };

    DistSharedPaHeapReservation reservation{1, 2, 3};
    EXPECT_FALSE(reserve(*state, -1, 1, reservation));
    expect_unchanged();
    EXPECT_FALSE(reserve(*state, static_cast<int32_t>(kFdwicSharedPaTaskCapacity), 1, reservation));
    expect_unchanged();
    EXPECT_FALSE(reserve(*state, 0, 1, reservation, kFdwicSharedHeapBytes - 1));
    expect_unchanged();
    EXPECT_FALSE(reserve(*state, 0, kFdwicSharedHeapShardBytes + 1, reservation));
    expect_unchanged();
}

}  // namespace
