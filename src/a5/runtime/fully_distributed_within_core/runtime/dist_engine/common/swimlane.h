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

#include "dist_engine/common/state.h"
#include "dist_engine/common/swimlane_types.h"
#include "dist_engine/common/worker_state.h"
#include "dist_engine/common/atomic.h"
#include "dist_engine/aicore/primitive.h"

#if defined(__CCE_AICORE__) || defined(__CPU_SIM)
#include "inner_kernel.h"
#endif

namespace {

PTO_DEVICE_FUNC inline uint64_t fdwic_swimlane_detail_now() {
#if defined(__CCE_AICORE__) || defined(__CPU_SIM)
    return get_sys_cnt_aicore();
#else
    return 0;
#endif
}

PTO_DEVICE_FUNC inline bool fdwic_swimlane_enabled() { return g_fdwic_swimlane_level != 0; }

PTO_DEVICE_FUNC inline bool fdwic_atomic_swimlane_enabled() {
    return g_fdwic_swimlane_level >= kFdwicAtomicSwimlaneLevel;
}

PTO_DEVICE_FUNC inline bool fdwic_atomic_return_ready_observed() {
#if defined(__CCE_AICORE__)
    return true;
#else
    // CPU simulation validates the scheduler and raw schema, not A5 atomic
    // completion timing. Do not claim a hardware return-ready boundary there.
    return false;
#endif
}

template <typename T>
PTO_DEVICE_FUNC inline uint64_t fdwic_swimlane_detail_now_after_atomic_result(T value) {
#if defined(__CCE_AICORE__)
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "atomic dependency expects a scalar result");
    uint64_t cycle = 0;
    // Consume the returned scalar in the same asm block as SYS_CNT. This is a
    // local return-value-ready boundary, not a cross-core visibility fence.
    asm volatile("MOV %0, %0\n"
                 "MOV %1, SYS_CNT\n"
                 : "+l"(value), "=&l"(cycle));
    return cycle;
#else
    (void)value;
    return fdwic_swimlane_detail_now();
#endif
}

PTO_DEVICE_FUNC inline void fdwic_swimlane_attach(__gm__ Runtime *runtime) {
    g_fdwic_swimlane_level = 0;
    g_fdwic_swimlane_header = nullptr;
    g_fdwic_swimlane_core = nullptr;
    g_fdwic_swimlane_records = nullptr;
    g_fdwic_swimlane_records_per_core = 0;
    g_fdwic_atomic_poll_burst.active_mask = 0;
    g_fdwic_atomic_poll_burst.enabled_mask = 0;
    g_fdwic_atomic_calls = 0;
    g_fdwic_poll_calls = 0;
    g_fdwic_poll_batch_records = 0;
    g_fdwic_atomic_counter_overflow = false;
    if (runtime == nullptr) return;
#if defined(__CCE_AICORE__)
    dist_aicore_invalidate_region(const_cast<__gm__ uint64_t *>(&runtime->dist.swimlane_base), 64);
#endif
    const uint64_t base = runtime->dist.swimlane_base;
    const uint32_t level = runtime->dist.swimlane_level;
    const uint32_t records_per_core = runtime->dist.swimlane_records_per_core;
    if (level == 0 || base == 0 || records_per_core == 0) return;
    g_fdwic_swimlane_header = reinterpret_cast<__gm__ FdwicSwimlaneHeader *>(base);
#if defined(__CCE_AICORE__)
    // The host initializes this cache line before launching the kernel. Drop a
    // possibly stale line left by a previous allocation at the same GM address.
    // Do not invalidate the whole header: other cores update their own states.
    dist_aicore_invalidate_region(g_fdwic_swimlane_header, 64);
#endif
    g_fdwic_swimlane_records_per_core = records_per_core;
    g_fdwic_swimlane_level = level;
}

PTO_DEVICE_FUNC inline __gm__ FdwicSwimlaneRecord *fdwic_swimlane_detail_records(__gm__ FdwicSwimlaneHeader *header) {
    return reinterpret_cast<__gm__ FdwicSwimlaneRecord *>(
        reinterpret_cast<__gm__ uint8_t *>(header) + sizeof(FdwicSwimlaneHeader)
    );
}

PTO_DEVICE_FUNC inline void fdwic_swimlane_reset_core(__gm__ DistCore *self) {
    if (!fdwic_swimlane_enabled() || self == nullptr) return;
    __gm__ FdwicSwimlaneHeader *header = g_fdwic_swimlane_header;
    if (header == nullptr) return;
    if (self->core_idx < 0 || self->core_idx >= static_cast<int32_t>(header->num_cores)) return;
    g_fdwic_swimlane_core = &header->cores[self->core_idx];
    __gm__ FdwicSwimlaneRecord *records = fdwic_swimlane_detail_records(header);
    const uint64_t record_offset = static_cast<uint64_t>(self->core_idx) * g_fdwic_swimlane_records_per_core;
    g_fdwic_swimlane_records = records + record_offset;
    g_fdwic_swimlane_core->count = 0;
    g_fdwic_swimlane_core->dropped = 0;
    g_fdwic_swimlane_core->atomic_calls = 0;
    g_fdwic_swimlane_core->poll_calls = 0;
    g_fdwic_swimlane_core->poll_batch_records = 0;
    g_fdwic_swimlane_core->core_idx = self->core_idx;
    g_fdwic_swimlane_core->block_id = self->block_id;
    g_fdwic_swimlane_core->lane = self->lane;
    g_fdwic_atomic_poll_burst.active_mask = 0;
    g_fdwic_atomic_poll_burst.enabled_mask = 0;
    g_fdwic_atomic_calls = 0;
    g_fdwic_poll_calls = 0;
    g_fdwic_poll_batch_records = 0;
    g_fdwic_atomic_counter_overflow = false;
}

PTO_DEVICE_FUNC inline bool fdwic_swimlane_detail_write_record(
    __gm__ DistCore *self, int32_t task_id, int32_t func_id, FdwicSwimlanePhase phase, uint64_t start_cycle,
    uint64_t end_cycle, uint32_t flags, uint32_t aux
) {
    if (!fdwic_swimlane_enabled() || self == nullptr) return false;
    const uint32_t records_per_core = g_fdwic_swimlane_records_per_core;
    __gm__ FdwicSwimlaneCoreState *core = g_fdwic_swimlane_core;
    if (core == nullptr || g_fdwic_swimlane_records == nullptr || records_per_core == 0) return false;
    const uint32_t slot = core->count;
    if (slot >= records_per_core) {
        if (core->dropped != UINT32_MAX) core->dropped = core->dropped + 1;
        return false;
    }
    __gm__ FdwicSwimlaneRecord *record = &g_fdwic_swimlane_records[slot];
    record->start_cycle = start_cycle;
    record->end_cycle = end_cycle;
    record->task_id = task_id;
    record->func_id = func_id;
    record->flags = flags;
    record->phase = static_cast<uint16_t>(phase);
    record->aux = static_cast<uint16_t>(aux);
    core->count = slot + 1;
    return true;
}

PTO_DEVICE_FUNC inline void fdwic_swimlane_detail_record(
    __gm__ DistCore *self, int32_t task_id, int32_t func_id, FdwicSwimlanePhase phase, uint64_t start_cycle,
    uint64_t end_cycle, uint32_t flags = 0, uint32_t aux = 0
) {
    (void)fdwic_swimlane_detail_write_record(self, task_id, func_id, phase, start_cycle, end_cycle, flags, aux);
}

PTO_DEVICE_FUNC inline uint32_t fdwic_atomic_trace_flags(
    FdwicAtomicOp op, bool result_used, bool return_ready, bool value_zero = false, uint64_t retries = 0
) {
    constexpr uint64_t kMaxRetries = (1ULL << (32 - kFdwicAtomicRetriesShift)) - 1;
    const uint32_t encoded_retries = static_cast<uint32_t>(retries > kMaxRetries ? kMaxRetries : retries);
    return static_cast<uint32_t>(op) | (result_used ? kFdwicAtomicResultUsed : 0U) |
           (value_zero ? kFdwicAtomicValueZero : 0U) | (return_ready ? kFdwicAtomicReturnReady : 0U) |
           (encoded_retries << kFdwicAtomicRetriesShift);
}

PTO_DEVICE_FUNC inline uint32_t fdwic_atomic_poll_trace_flags(FdwicAtomicSite site, uint32_t call_count) {
    return static_cast<uint32_t>(fdwic_atomic_site_op(site)) | kFdwicAtomicResultUsed | kFdwicAtomicPollBatch |
           (call_count << kFdwicAtomicPollCountShift);
}

PTO_DEVICE_FUNC inline uint32_t fdwic_atomic_site_mask(FdwicAtomicSite site) {
    return 1U << static_cast<uint32_t>(site);
}

PTO_DEVICE_FUNC inline uint32_t fdwic_atomic_block_won_poll_mask() {
    return fdwic_atomic_site_mask(FdwicAtomicSite::WonAnyLoad) | fdwic_atomic_site_mask(FdwicAtomicSite::WonStateLoad) |
           fdwic_atomic_site_mask(FdwicAtomicSite::WonLaneClaimExchange) |
           fdwic_atomic_site_mask(FdwicAtomicSite::WonDrainedLoad);
}

PTO_DEVICE_FUNC inline void fdwic_swimlane_count_atomic_call(bool poll_batch) {
    if (g_fdwic_atomic_calls == UINT32_MAX) {
        g_fdwic_atomic_counter_overflow = true;
        return;
    }
    g_fdwic_atomic_calls++;
    if (!poll_batch) return;
    if (g_fdwic_poll_calls == UINT32_MAX) {
        g_fdwic_atomic_counter_overflow = true;
        return;
    }
    g_fdwic_poll_calls++;
}

PTO_DEVICE_FUNC inline void fdwic_atomic_poll_boundary_at(uint64_t end_cycle) {
    if (!fdwic_atomic_swimlane_enabled() || g_fdwic_atomic_poll_burst.active_mask == 0) return;
    __gm__ DistCore *self = g_self;
    if (self == nullptr || g_fdwic_swimlane_core == nullptr) return;
    const uint32_t active_mask = g_fdwic_atomic_poll_burst.active_mask;
    for (uint32_t batch_index = 0; batch_index < kFdwicAtomicPollBatchSiteCount; ++batch_index) {
        const uint32_t bit = 1U << batch_index;
        if ((active_mask & bit) == 0) continue;
        const uint32_t call_count = g_fdwic_atomic_poll_burst.call_count[batch_index];
        if (call_count == 0 || call_count > kFdwicAtomicPollCountMax) {
            g_fdwic_atomic_counter_overflow = true;
            continue;
        }
        const FdwicAtomicSite site = fdwic_atomic_poll_batch_site(batch_index);
        const uint32_t site_index = static_cast<uint32_t>(site);
        const bool written = fdwic_swimlane_detail_write_record(
            self, -1, -1, FdwicSwimlanePhase::Atomic, g_fdwic_atomic_poll_burst.start_cycle[batch_index], end_cycle,
            fdwic_atomic_poll_trace_flags(site, call_count), site_index
        );
        if (written) {
            if (g_fdwic_poll_batch_records == UINT32_MAX) {
                g_fdwic_atomic_counter_overflow = true;
            } else {
                g_fdwic_poll_batch_records++;
            }
        }
        g_fdwic_atomic_poll_burst.call_count[batch_index] = 0;
    }
    g_fdwic_atomic_poll_burst.active_mask = 0;
}

PTO_DEVICE_FUNC inline void fdwic_atomic_poll_boundary() {
    if (g_fdwic_atomic_poll_burst.active_mask == 0) return;
    fdwic_atomic_poll_boundary_at(fdwic_swimlane_detail_now());
}

PTO_DEVICE_FUNC inline uint32_t fdwic_atomic_poll_region_begin(uint32_t site_mask) {
    const uint32_t previous_mask = g_fdwic_atomic_poll_burst.enabled_mask;
    if (!fdwic_atomic_swimlane_enabled()) return previous_mask;
    fdwic_atomic_poll_boundary();
    g_fdwic_atomic_poll_burst.enabled_mask = previous_mask | site_mask;
    return previous_mask;
}

PTO_DEVICE_FUNC inline void fdwic_atomic_poll_region_end(uint32_t previous_mask) {
    if (!fdwic_atomic_swimlane_enabled()) return;
    fdwic_atomic_poll_boundary();
    g_fdwic_atomic_poll_burst.enabled_mask = previous_mask;
}

PTO_DEVICE_FUNC inline bool fdwic_atomic_poll_batch_enabled(FdwicAtomicSite site, FdwicAtomicOp actual_op) {
    if (!fdwic_atomic_site_is_poll_batchable(site) || fdwic_atomic_site_op(site) != actual_op) return false;
    return (g_fdwic_atomic_poll_burst.enabled_mask & fdwic_atomic_site_mask(site)) != 0;
}

PTO_DEVICE_FUNC inline void fdwic_swimlane_accumulate_poll_call(FdwicAtomicSite site, uint64_t start_cycle) {
    const int32_t batch_index = fdwic_atomic_poll_batch_index(site);
    if (batch_index < 0) {
        g_fdwic_atomic_counter_overflow = true;
        return;
    }
    const uint32_t bit = 1U << static_cast<uint32_t>(batch_index);
    if ((g_fdwic_atomic_poll_burst.active_mask & bit) == 0) {
        g_fdwic_atomic_poll_burst.start_cycle[batch_index] = start_cycle;
        g_fdwic_atomic_poll_burst.call_count[batch_index] = 0;
        g_fdwic_atomic_poll_burst.active_mask |= bit;
    }
    uint32_t &call_count = g_fdwic_atomic_poll_burst.call_count[batch_index];
    call_count++;
    if (call_count == kFdwicAtomicPollCountMax) fdwic_atomic_poll_boundary();
}

PTO_DEVICE_FUNC inline void fdwic_swimlane_flush_core(__gm__ DistCore *self) {
    if (!fdwic_swimlane_enabled() || self == nullptr) return;
    fdwic_atomic_poll_boundary();
    const uint32_t records_per_core = g_fdwic_swimlane_records_per_core;
    __gm__ FdwicSwimlaneCoreState *core = g_fdwic_swimlane_core;
    if (core == nullptr || g_fdwic_swimlane_records == nullptr || records_per_core == 0) return;
    core->atomic_calls = g_fdwic_atomic_calls;
    core->poll_calls = g_fdwic_poll_calls;
    core->poll_batch_records = g_fdwic_poll_batch_records;
    if (g_fdwic_atomic_counter_overflow && core->dropped != UINT32_MAX) core->dropped = core->dropped + 1;
    const uint32_t count = core->count < records_per_core ? core->count : records_per_core;
    if (count > 0) {
        dist_aicore_flush_region(g_fdwic_swimlane_records, static_cast<uint64_t>(count) * sizeof(FdwicSwimlaneRecord));
    }
    dist_aicore_flush_region(core, sizeof(FdwicSwimlaneCoreState));
}

PTO_DEVICE_FUNC inline void fdwic_swimlane_detail_record_atomic(
    int32_t task_id, FdwicAtomicSite site, FdwicAtomicOp op, uint64_t start_cycle, uint64_t end_cycle, bool result_used,
    bool return_ready, bool value_zero = false, uint64_t retries = 0
) {
    __gm__ DistCore *self = g_self;
    if (!fdwic_atomic_swimlane_enabled() || self == nullptr || g_fdwic_swimlane_core == nullptr) return;
    fdwic_swimlane_detail_record(
        self, task_id, -1, FdwicSwimlanePhase::Atomic, start_cycle, end_cycle,
        fdwic_atomic_trace_flags(op, result_used, return_ready, value_zero, retries), static_cast<uint32_t>(site)
    );
}

// Direct atomics remain one row per source call but do not split an active
// PollBatch. A batch is a logical wait-region window and may contain these
// interleaved rows; only its call_count, not its duration, represents atomic
// work. Region/phase/lap/final boundaries still close every active batch.

template <typename T>
PTO_DEVICE_FUNC inline T fdwic_trace_atomic_load(
    int32_t task_id, FdwicAtomicSite site, __gm__ volatile T &value, bool result_used = true,
    int memorder = __ATOMIC_ACQUIRE
) {
    if (!fdwic_atomic_swimlane_enabled()) return atomic_load(value, memorder);
    const bool poll_batch = result_used && fdwic_atomic_poll_batch_enabled(site, FdwicAtomicOp::Load);
    const int32_t batch_index = poll_batch ? fdwic_atomic_poll_batch_index(site) : -1;
    const bool first_in_batch =
        poll_batch && (g_fdwic_atomic_poll_burst.active_mask & (1U << static_cast<uint32_t>(batch_index))) == 0;
    const uint64_t begin = !poll_batch || first_in_batch ? fdwic_swimlane_detail_now() : 0;
    const T old = atomic_load(value, memorder);
    if (poll_batch) {
        fdwic_swimlane_count_atomic_call(true);
        fdwic_swimlane_accumulate_poll_call(site, begin);
        return old;
    }
    const bool return_ready = result_used && fdwic_atomic_return_ready_observed();
    const uint64_t end = result_used ? fdwic_swimlane_detail_now_after_atomic_result(old) : fdwic_swimlane_detail_now();
    // Keep tracing bookkeeping outside the measured direct-atomic boundary.
    fdwic_swimlane_count_atomic_call(false);
    fdwic_swimlane_detail_record_atomic(
        task_id, site, FdwicAtomicOp::Load, begin, end, result_used, return_ready, old == static_cast<T>(0)
    );
    return old;
}

template <typename T, typename V>
PTO_DEVICE_FUNC inline T fdwic_trace_atomic_exchange(
    int32_t task_id, FdwicAtomicSite site, __gm__ volatile T &value, V desired, bool result_used = false,
    int memorder = __ATOMIC_ACQ_REL
) {
    if (!fdwic_atomic_swimlane_enabled()) return atomic_exchange(value, desired, memorder);
    const T desired_value = static_cast<T>(desired);
    const bool failed_claim_batch_enabled = result_used && site == FdwicAtomicSite::WonLaneClaimExchange &&
                                            desired_value == static_cast<T>(kDrainedClaimed) &&
                                            fdwic_atomic_poll_batch_enabled(site, FdwicAtomicOp::Exchange);
    const uint64_t begin = fdwic_swimlane_detail_now();
    const T old = atomic_exchange(value, desired, memorder);
    const bool failed_claim_batch = failed_claim_batch_enabled && old == desired_value;
    if (failed_claim_batch) {
        fdwic_swimlane_count_atomic_call(true);
        fdwic_swimlane_accumulate_poll_call(site, begin);
        return old;
    }
    const bool return_ready = result_used && fdwic_atomic_return_ready_observed();
    const uint64_t end = result_used ? fdwic_swimlane_detail_now_after_atomic_result(old) : fdwic_swimlane_detail_now();
    fdwic_swimlane_count_atomic_call(false);
    // Close failed retries at the successful transition's issue boundary. The
    // successful claim itself remains an exact direct row. Capture its end
    // first so batch-record writes are not charged to the direct span.
    if (failed_claim_batch_enabled) fdwic_atomic_poll_boundary_at(begin);
    fdwic_swimlane_detail_record_atomic(task_id, site, FdwicAtomicOp::Exchange, begin, end, result_used, return_ready);
    return old;
}

template <typename T>
PTO_DEVICE_FUNC inline T fdwic_trace_atomic_fetch_add(
    int32_t task_id, FdwicAtomicSite site, __gm__ volatile T &value, T delta, bool result_used = false,
    int memorder = __ATOMIC_ACQ_REL
) {
    if (!fdwic_atomic_swimlane_enabled()) return atomic_fetch_add(value, delta, memorder);
    const uint64_t begin = fdwic_swimlane_detail_now();
    const T old = atomic_fetch_add(value, delta, memorder);
    const bool return_ready = result_used && fdwic_atomic_return_ready_observed();
    const uint64_t end = result_used ? fdwic_swimlane_detail_now_after_atomic_result(old) : fdwic_swimlane_detail_now();
    fdwic_swimlane_count_atomic_call(false);
    fdwic_swimlane_detail_record_atomic(task_id, site, FdwicAtomicOp::FetchAdd, begin, end, result_used, return_ready);
    return old;
}

template <typename T>
PTO_DEVICE_FUNC inline T fdwic_trace_atomic_fetch_sub(
    int32_t task_id, FdwicAtomicSite site, __gm__ volatile T &value, T delta, bool result_used = false,
    int memorder = __ATOMIC_ACQ_REL
) {
    if (!fdwic_atomic_swimlane_enabled()) return atomic_fetch_sub(value, delta, memorder);
    const uint64_t begin = fdwic_swimlane_detail_now();
    const T old = atomic_fetch_sub(value, delta, memorder);
    const bool return_ready = result_used && fdwic_atomic_return_ready_observed();
    const uint64_t end = result_used ? fdwic_swimlane_detail_now_after_atomic_result(old) : fdwic_swimlane_detail_now();
    fdwic_swimlane_count_atomic_call(false);
    fdwic_swimlane_detail_record_atomic(task_id, site, FdwicAtomicOp::FetchSub, begin, end, result_used, return_ready);
    return old;
}

template <typename T>
PTO_DEVICE_FUNC inline T fdwic_trace_atomic_fetch_max(
    int32_t task_id, FdwicAtomicSite site, __gm__ volatile T &value, T desired, bool result_used = true,
    int memorder = __ATOMIC_ACQ_REL
) {
    if (!fdwic_atomic_swimlane_enabled()) return atomic_fetch_max(value, desired, memorder);
    const uint64_t begin = fdwic_swimlane_detail_now();
    const T old = atomic_fetch_max(value, desired, memorder);
    const bool return_ready = result_used && fdwic_atomic_return_ready_observed();
    const uint64_t end = result_used ? fdwic_swimlane_detail_now_after_atomic_result(old) : fdwic_swimlane_detail_now();
    fdwic_swimlane_count_atomic_call(false);
    fdwic_swimlane_detail_record_atomic(task_id, site, FdwicAtomicOp::FetchMax, begin, end, result_used, return_ready);
    return old;
}

PTO_DEVICE_FUNC inline bool fdwic_trace_is_fatal(int32_t task_id = -1) {
    return fdwic_trace_atomic_load(task_id, FdwicAtomicSite::FatalPoll, g_dist.fatal) != 0;
}

PTO_DEVICE_FUNC inline void fdwic_trace_set_fatal(int32_t task_id = -1) {
    const int32_t previous = fdwic_trace_atomic_exchange(
        task_id, FdwicAtomicSite::FatalSet, g_dist.fatal, int32_t{1}, /*result_used=*/false
    );
    (void)previous;
}

PTO_DEVICE_FUNC inline void fdwic_swimlane_record_clock_baselines(__gm__ DistCore *self, int32_t dependency_value) {
    if (!fdwic_atomic_swimlane_enabled() || self == nullptr) return;
    fdwic_atomic_poll_boundary();
    const uint64_t clock_begin = fdwic_swimlane_detail_now();
    const uint64_t clock_end = fdwic_swimlane_detail_now();
    fdwic_swimlane_detail_record(self, -1, -1, FdwicSwimlanePhase::ClockBaseline, clock_begin, clock_end);
    const uint64_t dependency_begin = fdwic_swimlane_detail_now();
    const uint64_t dependency_end = fdwic_swimlane_detail_now_after_atomic_result(dependency_value);
    const uint32_t flags =
        kFdwicClockAtomicDependency | (fdwic_atomic_return_ready_observed() ? kFdwicClockAtomicDependencyApplied : 0U);
    fdwic_swimlane_detail_record(
        self, -1, -1, FdwicSwimlanePhase::ClockBaseline, dependency_begin, dependency_end, flags
    );
}

PTO_DEVICE_FUNC inline void fdwic_swimlane_lap_reset(__gm__ DistCore *self) {
    if (self == nullptr) return;
    const uint64_t cycle = fdwic_swimlane_detail_now();
    fdwic_atomic_poll_boundary_at(cycle);
    self->swimlane_last_cycle = cycle;
}

PTO_DEVICE_FUNC inline void
fdwic_swimlane_lap(__gm__ DistCore *self, int32_t task_id, int32_t func_id, FdwicSwimlanePhase phase) {
    if (self == nullptr) return;
    const uint64_t end_cycle = fdwic_swimlane_detail_now();
    fdwic_atomic_poll_boundary_at(end_cycle);
    const uint64_t start_cycle = self->swimlane_last_cycle;
    fdwic_swimlane_detail_record(self, task_id, func_id, phase, start_cycle, end_cycle);
    self->swimlane_last_cycle = end_cycle;
}

}  // namespace
