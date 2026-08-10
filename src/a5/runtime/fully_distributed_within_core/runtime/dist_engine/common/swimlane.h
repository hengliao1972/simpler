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
#include "dist_engine/common/submit_pmu.h"

#if defined(__CCE_AICORE__) || defined(__CPU_SIM)
#include "inner_kernel.h"
#endif

namespace {

template <typename T>
PTO_DEVICE_FUNC inline uint64_t fdwic_scalar_result_ready_tick(T value) {
#if defined(__CCE_AICORE__)
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "result dependency expects a scalar value");
    uint64_t cycle = 0;
    // Consume a scalar result in the same asm block as SYS_CNT. This creates
    // a local result-ready boundary; it is not a cross-core visibility barrier.
    asm volatile("MOV %0, %0\n"
                 "MOV %1, SYS_CNT\n"
                 : "+l"(value), "=&l"(cycle));
    return cycle;
#elif defined(__CPU_SIM)
    (void)value;
    return get_sys_cnt_aicore();
#else
    (void)value;
    return 0;
#endif
}

template <typename T>
PTO_DEVICE_FUNC inline uint64_t fdwic_atomic_result_ready_tick(T value) {
    return fdwic_scalar_result_ready_tick(value);
}

#if DIST_TRACE_ENABLED

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

PTO_DEVICE_FUNC inline void fdwic_swimlane_attach(__gm__ Runtime *runtime) {
    g_fdwic_swimlane_level = 0;
    g_fdwic_swimlane_header = nullptr;
    g_fdwic_swimlane_core = nullptr;
    g_fdwic_swimlane_records = nullptr;
    g_fdwic_swimlane_records_per_core = 0;
#if PTO_FDWIC_SHARED_MAP
    g_fdwic_swimlane_record_count = 0;
    g_fdwic_swimlane_dropped_records = 0;
    g_fdwic_swimlane_shared_submit_count = 0;
    g_fdwic_dcci_calls = 0;
    g_fdwic_dcci_lines = 0;
    g_fdwic_dcci_records = 0;
    g_fdwic_dcci_counter_overflow = false;
#endif
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
#if PTO_FDWIC_SHARED_MAP
    if (g_fdwic_swimlane_header->magic != kFdwicSwimlaneMagic ||
        g_fdwic_swimlane_header->version != kFdwicSwimlaneVersion ||
        g_fdwic_swimlane_header->record_size_bytes != kFdwicSwimlaneRecordSizeBytes ||
        g_fdwic_swimlane_header->records_per_core != records_per_core) {
        g_fdwic_swimlane_header = nullptr;
        return;
    }
#endif
    g_fdwic_swimlane_records_per_core = records_per_core;
    g_fdwic_swimlane_level = level;
}

PTO_DEVICE_FUNC inline __gm__ FdwicSwimlaneStorageRecord *
fdwic_swimlane_detail_records(__gm__ FdwicSwimlaneHeader *header, uint32_t core_idx) {
#if PTO_FDWIC_SHARED_MAP
    return reinterpret_cast<__gm__ FdwicSwimlaneStorageRecord *>(
        reinterpret_cast<__gm__ uint8_t *>(header) + sizeof(FdwicSwimlaneHeader) +
        static_cast<uint64_t>(core_idx) * kFdwicSwimlaneWorkerBytes + kFdwicSharedSubmitClaimBytesPerCore
    );
#else
    return reinterpret_cast<__gm__ FdwicSwimlaneStorageRecord *>(
        reinterpret_cast<__gm__ uint8_t *>(header) + sizeof(FdwicSwimlaneHeader) +
        static_cast<uint64_t>(core_idx) * kFdwicSwimlaneWorkerBytes
    );
#endif
}

#if PTO_FDWIC_SHARED_MAP
PTO_DEVICE_FUNC inline __gm__ FdwicSharedSubmitClaimRecord *
fdwic_swimlane_shared_submit_claim_records(__gm__ FdwicSwimlaneStorageRecord *records) {
    return reinterpret_cast<__gm__ FdwicSharedSubmitClaimRecord *>(
        reinterpret_cast<__gm__ uint8_t *>(records) - kFdwicSharedSubmitClaimBytesPerCore
    );
}
#endif

PTO_DEVICE_FUNC inline void fdwic_swimlane_reset_core(__gm__ DistCore *self) {
    if (!fdwic_swimlane_enabled() || self == nullptr) return;
    __gm__ FdwicSwimlaneHeader *header = g_fdwic_swimlane_header;
    if (header == nullptr) return;
    if (self->core_idx < 0 || self->core_idx >= static_cast<int32_t>(header->num_cores)) return;
    g_fdwic_swimlane_core = &header->cores[self->core_idx];
    g_fdwic_swimlane_records = fdwic_swimlane_detail_records(header, static_cast<uint32_t>(self->core_idx));
#if PTO_FDWIC_SHARED_MAP
    g_fdwic_swimlane_record_count = 0;
    g_fdwic_swimlane_dropped_records = 0;
    g_fdwic_swimlane_shared_submit_count = 0;
    g_fdwic_dcci_calls = 0;
    g_fdwic_dcci_lines = 0;
    g_fdwic_dcci_records = 0;
    g_fdwic_dcci_counter_overflow = false;
#else
    g_fdwic_swimlane_core->count = 0;
    g_fdwic_swimlane_core->dropped = 0;
    g_fdwic_swimlane_core->atomic_calls = 0;
    g_fdwic_swimlane_core->poll_calls = 0;
    g_fdwic_swimlane_core->poll_batch_records = 0;
    g_fdwic_swimlane_core->core_idx = self->core_idx;
    g_fdwic_swimlane_core->block_id = self->block_id;
    g_fdwic_swimlane_core->lane = self->lane;
#endif
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
#if PTO_FDWIC_SHARED_MAP
    if (end_cycle < start_cycle ||
        end_cycle - start_cycle > UINT32_MAX ||
        !fdwic_compact_trace_fields_fit(task_id, func_id, phase, aux)) {
        if (g_fdwic_swimlane_dropped_records != UINT32_MAX) ++g_fdwic_swimlane_dropped_records;
        return false;
    }
    const uint32_t slot = g_fdwic_swimlane_record_count;
    if (slot >= records_per_core) {
        if (g_fdwic_swimlane_dropped_records != UINT32_MAX) ++g_fdwic_swimlane_dropped_records;
        return false;
    }
    __gm__ FdwicSwimlaneStorageRecord *record = &g_fdwic_swimlane_records[slot];
    record->start_cycle_low = static_cast<uint32_t>(start_cycle);
    record->end_cycle_low = static_cast<uint32_t>(end_cycle);
    record->flags = flags;
    record->packed = fdwic_pack_compact_trace_fields(task_id, func_id, phase, aux);
    g_fdwic_swimlane_record_count = slot + 1;
#else
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
#endif
    return true;
}

#if PTO_FDWIC_SHARED_MAP
PTO_DEVICE_FUNC inline bool fdwic_swimlane_record_shared_claim(
    __gm__ DistCore *self, int32_t task_id, uint64_t start_cycle, uint64_t end_cycle, bool winner
) {
    if (!fdwic_swimlane_enabled() || self == nullptr || g_fdwic_swimlane_records == nullptr ||
        task_id < 0 || task_id >= static_cast<int32_t>(kFdwicSharedTraceTaskCapacity) ||
        start_cycle == 0 || end_cycle < start_cycle || (end_cycle & kFdwicSharedClaimWinnerBit) != 0) {
        if (fdwic_swimlane_enabled() && g_fdwic_swimlane_dropped_records != UINT32_MAX) {
            ++g_fdwic_swimlane_dropped_records;
        }
        return false;
    }
    __gm__ FdwicSharedSubmitClaimRecord *records =
        fdwic_swimlane_shared_submit_claim_records(g_fdwic_swimlane_records);
    records[task_id].claim_begin = start_cycle;
    records[task_id].claim_end_and_winner = end_cycle | (winner ? kFdwicSharedClaimWinnerBit : 0U);
    return true;
}

PTO_DEVICE_FUNC inline bool fdwic_swimlane_record_shared_submit(
    __gm__ DistCore *self, int32_t task_id, uint64_t start_cycle, uint64_t end_cycle
) {
    if (!fdwic_swimlane_enabled() || self == nullptr || g_fdwic_swimlane_records == nullptr ||
        task_id < 0 || task_id >= static_cast<int32_t>(kFdwicSharedTraceTaskCapacity) ||
        start_cycle == 0 || end_cycle < start_cycle || (end_cycle & kFdwicSharedClaimWinnerBit) != 0) {
        if (fdwic_swimlane_enabled() && g_fdwic_swimlane_dropped_records != UINT32_MAX) {
            ++g_fdwic_swimlane_dropped_records;
        }
        return false;
    }
    __gm__ FdwicSharedSubmitClaimRecord *records =
        fdwic_swimlane_shared_submit_claim_records(g_fdwic_swimlane_records);
    records[task_id].submit_begin = start_cycle;
    records[task_id].submit_end = end_cycle;
    const uint32_t next = static_cast<uint32_t>(task_id) + 1U;
    if (next > g_fdwic_swimlane_shared_submit_count) g_fdwic_swimlane_shared_submit_count = next;
    return true;
}
#endif

PTO_DEVICE_FUNC inline void fdwic_swimlane_detail_record(
    __gm__ DistCore *self, int32_t task_id, int32_t func_id, FdwicSwimlanePhase phase, uint64_t start_cycle,
    uint64_t end_cycle, uint32_t flags = 0, uint32_t aux = 0
) {
#if PTO_FDWIC_SHARED_MAP
    if (phase == FdwicSwimlanePhase::Claim) {
        (void)fdwic_swimlane_record_shared_claim(
            self, task_id, start_cycle, end_cycle, (flags & kFdwicClaimWon) != 0
        );
        return;
    }
    if (phase == FdwicSwimlanePhase::Submit) {
        (void)fdwic_swimlane_record_shared_submit(self, task_id, start_cycle, end_cycle);
        return;
    }
#endif
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
#if PTO_FDWIC_SHARED_MAP
    const int32_t index = fdwic_atomic_poll_batch_index(site);
    return index >= 0 ? 1U << static_cast<uint32_t>(index) : 0U;
#else
    return 1U << static_cast<uint32_t>(site);
#endif
}

PTO_DEVICE_FUNC inline uint32_t fdwic_atomic_block_won_poll_mask() {
#if PTO_FDWIC_SHARED_MAP
    return 0;
#else
    return fdwic_atomic_site_mask(FdwicAtomicSite::WonAnyLoad) | fdwic_atomic_site_mask(FdwicAtomicSite::WonStateLoad) |
           fdwic_atomic_site_mask(FdwicAtomicSite::WonLaneClaimExchange) |
           fdwic_atomic_site_mask(FdwicAtomicSite::WonDrainedLoad);
#endif
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

// 调用方已经在边界处取得 end_cycle；十类 PollBatch 的遍历与落盘只在
// level-4 且确有活动批次时需要。把慢体共享起来，避免它被每个 phase/lap
// 边界重复内联，同时不让 level-1 快路径承担函数调用。
PTO_DEVICE_FUNC __attribute__((noinline)) void fdwic_atomic_poll_boundary_slow(uint64_t end_cycle) {
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

PTO_DEVICE_FUNC inline void fdwic_atomic_poll_boundary_at(uint64_t end_cycle) {
    if (!fdwic_atomic_swimlane_enabled() || g_fdwic_atomic_poll_burst.active_mask == 0) return;
    fdwic_atomic_poll_boundary_slow(end_cycle);
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

#if PTO_FDWIC_SHARED_MAP
PTO_DEVICE_FUNC inline uint32_t fdwic_dcci_trace_flags(
    FdwicDcciOp op, bool trailing_dsb, uint32_t call_count, uint32_t line_count
) {
    return static_cast<uint32_t>(op) | (trailing_dsb ? kFdwicDcciTrailingDsb : 0U) |
           (call_count << kFdwicDcciCallCountShift) | (line_count << kFdwicDcciLineCountShift);
}

template <typename Pointer>
PTO_DEVICE_FUNC inline uint32_t fdwic_dcci_region_cache_line_count(Pointer address, uint64_t bytes) {
    if (address == nullptr || bytes == 0) return 0;
    const uint64_t begin = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(address));
    if (bytes > UINT64_MAX - begin || begin + bytes > UINT64_MAX - 63U) return 0;
    const uint64_t aligned_begin = begin & ~UINT64_C(63);
    const uint64_t aligned_end = (begin + bytes + 63U) & ~UINT64_C(63);
    const uint64_t lines = (aligned_end - aligned_begin) / 64U;
    return lines == 0 || lines > kFdwicDcciLineCountMax ? 0 : static_cast<uint32_t>(lines);
}

PTO_DEVICE_FUNC inline bool fdwic_swimlane_record_dcci(
    __gm__ DistCore *self, int32_t task_id, int32_t func_id, FdwicDcciSite site, FdwicDcciOp op,
    bool trailing_dsb, uint32_t call_count, uint32_t line_count, uint64_t start_cycle, uint64_t end_cycle
) {
    if (!fdwic_atomic_swimlane_enabled()) return false;
    const bool shape_valid =
        static_cast<uint32_t>(site) < static_cast<uint32_t>(FdwicDcciSite::Count) &&
        static_cast<uint32_t>(op) < static_cast<uint32_t>(FdwicDcciOp::Count) &&
        op == fdwic_dcci_site_op(site) && call_count > 0 && call_count <= kFdwicDcciCallCountMask &&
        line_count >= call_count && line_count <= kFdwicDcciLineCountMax && trailing_dsb &&
        end_cycle >= start_cycle;
    const bool counters_fit =
        g_fdwic_dcci_calls <= UINT32_MAX - call_count && g_fdwic_dcci_lines <= UINT32_MAX - line_count &&
        g_fdwic_dcci_records != UINT32_MAX;
    if (!shape_valid || !counters_fit) {
        g_fdwic_dcci_counter_overflow = true;
        return false;
    }
    const bool written = fdwic_swimlane_detail_write_record(
        self, task_id, func_id, FdwicSwimlanePhase::Dcci, start_cycle, end_cycle,
        fdwic_dcci_trace_flags(op, trailing_dsb, call_count, line_count), static_cast<uint32_t>(site)
    );
    if (!written) return false;
    g_fdwic_dcci_calls += call_count;
    g_fdwic_dcci_lines += line_count;
    ++g_fdwic_dcci_records;
    return true;
}

PTO_DEVICE_FUNC inline bool fdwic_swimlane_record_dcci(
    __gm__ DistCore *self, int32_t task_id, int32_t func_id, FdwicDcciSite site, bool trailing_dsb,
    uint32_t line_count, uint64_t start_cycle, uint64_t end_cycle
) {
    return fdwic_swimlane_record_dcci(
        self, task_id, func_id, site, fdwic_dcci_site_op(site), trailing_dsb, 1, line_count, start_cycle, end_cycle
    );
}
#endif

PTO_DEVICE_FUNC inline void fdwic_swimlane_flush_core(__gm__ DistCore *self) {
    if (!fdwic_swimlane_enabled() || self == nullptr) return;
    fdwic_atomic_poll_boundary();
    const uint32_t records_per_core = g_fdwic_swimlane_records_per_core;
    __gm__ FdwicSwimlaneCoreState *core = g_fdwic_swimlane_core;
    if (core == nullptr || g_fdwic_swimlane_records == nullptr || records_per_core == 0) return;
#if PTO_FDWIC_SHARED_MAP
    uint32_t observer_slot = UINT32_MAX;
#if PTO_FDWIC_SCHEDULER_MODE == 0
    if (g_fdwic_swimlane_shared_submit_count != kFdwicSharedTracePhase1TaskCount &&
        g_fdwic_swimlane_dropped_records != UINT32_MAX) {
        ++g_fdwic_swimlane_dropped_records;
    }
#else
    // Cross-core 调度不再要求每个 Scalar 回放完整 Submit 流，因此固定的
    // Submit/Claim 区域没有逐 task 端点。复用首个端点槽保存一个 64-bit
    // SYS_CNT 锚点，供 host 展开 16B compact record 的 low32 时钟；它不是
    // 伪造的 Submit 记录，也不进入业务事件计数。
    __gm__ FdwicSharedSubmitClaimRecord *anchor_record =
        fdwic_swimlane_shared_submit_claim_records(g_fdwic_swimlane_records);
    anchor_record[0].claim_begin = 0;
    anchor_record[0].claim_end_and_winner = 0;
    anchor_record[0].submit_begin = fdwic_swimlane_detail_now();
    anchor_record[0].submit_end = 0;
    g_fdwic_swimlane_shared_submit_count = 1;
#endif
    if (fdwic_atomic_swimlane_enabled()) {
        const uint64_t generic_bytes =
            static_cast<uint64_t>(g_fdwic_swimlane_record_count + 1U) * sizeof(FdwicSwimlaneStorageRecord);
        const uint64_t submit_bytes =
            static_cast<uint64_t>(g_fdwic_swimlane_shared_submit_count) * sizeof(FdwicSharedSubmitClaimRecord);
        const uint64_t generic_lines = (generic_bytes + 63U) / 64U;
        const uint64_t submit_lines = (submit_bytes + 63U) / 64U;
        const uint64_t total_lines = generic_lines + submit_lines + 1U;
        const uint64_t observer_cycle = fdwic_swimlane_detail_now();
        const uint32_t candidate_slot = g_fdwic_swimlane_record_count;
        if (total_lines > kFdwicDcciLineCountMax ||
            !fdwic_swimlane_record_dcci(
                self, -1, -1, FdwicDcciSite::ObserverTraceExport, FdwicDcciOp::CleanOut,
                /*trailing_dsb=*/true, /*call_count=*/3, static_cast<uint32_t>(total_lines),
                observer_cycle, observer_cycle
            )) {
            g_fdwic_dcci_counter_overflow = true;
        } else {
            observer_slot = candidate_slot;
        }
    }
    if ((g_fdwic_atomic_counter_overflow || g_fdwic_dcci_counter_overflow) &&
        g_fdwic_swimlane_dropped_records != UINT32_MAX) {
        ++g_fdwic_swimlane_dropped_records;
    }
    core->count = g_fdwic_swimlane_record_count;
    core->dropped = g_fdwic_swimlane_dropped_records;
    core->atomic_calls = g_fdwic_atomic_calls;
    core->poll_calls = g_fdwic_poll_calls;
    core->poll_batch_records = g_fdwic_poll_batch_records;
    core->core_idx = self->core_idx;
    core->block_id = self->block_id;
    core->lane = self->lane;
    core->dcci_calls = g_fdwic_dcci_calls;
    core->dcci_lines = g_fdwic_dcci_lines;
    core->dcci_records = g_fdwic_dcci_records;
    const uint32_t count =
        g_fdwic_swimlane_record_count < records_per_core ? g_fdwic_swimlane_record_count : records_per_core;
    if (count > 0) {
        dist_aicore_flush_region(
            g_fdwic_swimlane_records, static_cast<uint64_t>(count) * sizeof(FdwicSwimlaneStorageRecord)
        );
    }
    if (g_fdwic_swimlane_shared_submit_count > 0) {
        dist_aicore_flush_region(
            fdwic_swimlane_shared_submit_claim_records(g_fdwic_swimlane_records),
            static_cast<uint64_t>(g_fdwic_swimlane_shared_submit_count) * sizeof(FdwicSharedSubmitClaimRecord)
        );
    }
#else
    core->atomic_calls = g_fdwic_atomic_calls;
    core->poll_calls = g_fdwic_poll_calls;
    core->poll_batch_records = g_fdwic_poll_batch_records;
    if (g_fdwic_atomic_counter_overflow && core->dropped != UINT32_MAX) core->dropped = core->dropped + 1;
    const uint32_t count = core->count < records_per_core ? core->count : records_per_core;
    if (count > 0) {
        dist_aicore_flush_region(g_fdwic_swimlane_records, static_cast<uint64_t>(count) * sizeof(FdwicSwimlaneRecord));
    }
#endif
    dist_aicore_flush_region(core, sizeof(FdwicSwimlaneCoreState));
#if PTO_FDWIC_SHARED_MAP
    if (observer_slot != UINT32_MAX && observer_slot < g_fdwic_swimlane_record_count) {
        __gm__ FdwicSwimlaneStorageRecord *observer = &g_fdwic_swimlane_records[observer_slot];
        const uint32_t observer_end = static_cast<uint32_t>(fdwic_swimlane_detail_now());
#if defined(__CCE_AICORE__)
        // The terminal row was already included in the generic-region clean.
        // Publish only its low32 end point through the diagnostic bypass path;
        // a cached read/modify/write here could overwrite neighboring rows.
        __builtin_cce_st_dev(observer_end, &observer->end_cycle_low, 0);
#else
        observer->end_cycle_low = observer_end;
#endif
    }
#endif
}

// Atomic 记录落盘是 level-4 诊断冷路径。保持它为单一设备函数，避免完整的
// GM 边界检查与记录写入被复制到每一个 atomic 调用点；level-1 快路径在各
// wrapper 的首个分支已经返回，不会承担这里的 call/ret。
PTO_DEVICE_FUNC __attribute__((noinline)) void fdwic_swimlane_detail_record_atomic(
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

#if PTO_FDWIC_SHARED_MAP
// Some shared-map call sites must keep the measured atomic boundary free of
// trace-record stores. Capture start/end next to the atomic, then invoke this
// API after the surrounding scheduler state has been published. The call
// writes one raw row immediately; it does not defer or retain the endpoints.
PTO_DEVICE_FUNC inline bool fdwic_swimlane_record_captured_atomic(
    int32_t task_id, FdwicAtomicSite site, FdwicAtomicOp op, uint64_t start_cycle, uint64_t end_cycle,
    bool result_used, bool return_ready, bool value_zero = false, uint64_t retries = 0
) {
    if (!fdwic_atomic_swimlane_enabled()) return false;
    const bool shape_valid =
        static_cast<uint32_t>(site) < static_cast<uint32_t>(FdwicAtomicSite::Count) &&
        static_cast<uint32_t>(op) <= static_cast<uint32_t>(FdwicAtomicOp::CompareExchange) &&
        site != FdwicAtomicSite::SharedInsertTurnPoll && op == fdwic_atomic_site_op(site) &&
        result_used == fdwic_atomic_site_result_used(site) &&
        return_ready == (result_used && fdwic_atomic_return_ready_observed()) &&
        (!value_zero || op == FdwicAtomicOp::Load) &&
        (retries == 0 || op == FdwicAtomicOp::FetchMax) && end_cycle >= start_cycle;
    if (!shape_valid) {
        g_fdwic_atomic_counter_overflow = true;
        return false;
    }
    fdwic_swimlane_count_atomic_call(false);
    return fdwic_swimlane_detail_write_record(
        g_self, task_id, -1, FdwicSwimlanePhase::Atomic, start_cycle, end_cycle,
        fdwic_atomic_trace_flags(op, result_used, return_ready, value_zero, retries), static_cast<uint32_t>(site)
    );
}

PTO_DEVICE_FUNC inline bool fdwic_swimlane_record_aggregate_atomic_poll(
    FdwicAtomicSite site, uint64_t start_cycle, uint64_t end_cycle, uint32_t call_count, bool return_ready_end
) {
    if (!fdwic_atomic_swimlane_enabled()) return false;
    if (site != FdwicAtomicSite::SharedInsertTurnPoll || call_count == 0 ||
        call_count > kFdwicAtomicPollCountMax || end_cycle < start_cycle ||
        return_ready_end != fdwic_atomic_return_ready_observed() ||
        g_fdwic_atomic_calls > UINT32_MAX - call_count || g_fdwic_poll_calls > UINT32_MAX - call_count) {
        g_fdwic_atomic_counter_overflow = true;
        return false;
    }
    const uint32_t flags =
        static_cast<uint32_t>(FdwicAtomicOp::Load) | kFdwicAtomicResultUsed | kFdwicAtomicPollBatch |
        (return_ready_end ? kFdwicAtomicReturnReady : 0U) | (call_count << kFdwicAtomicPollCountShift);
    const bool written = fdwic_swimlane_detail_write_record(
        g_self, -1, -1, FdwicSwimlanePhase::Atomic, start_cycle, end_cycle, flags, static_cast<uint32_t>(site)
    );
    if (!written) return false;
    g_fdwic_atomic_calls += call_count;
    g_fdwic_poll_calls += call_count;
    if (g_fdwic_poll_batch_records == UINT32_MAX) {
        g_fdwic_atomic_counter_overflow = true;
        return false;
    }
    ++g_fdwic_poll_batch_records;
    return true;
}
#endif

// Direct atomics remain one row per source call but do not split an active
// PollBatch. A batch is a logical wait-region window and may contain these
// interleaved rows; only its call_count, not its duration, represents atomic
// work. Explicit region end, lap and final flush close every active batch;
// ordinary phase timestamps deliberately do not inspect or split it.

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
    const uint64_t end = result_used ? fdwic_atomic_result_ready_tick(old) : fdwic_swimlane_detail_now();
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
    const uint64_t end = result_used ? fdwic_atomic_result_ready_tick(old) : fdwic_swimlane_detail_now();
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
    const uint64_t end = result_used ? fdwic_atomic_result_ready_tick(old) : fdwic_swimlane_detail_now();
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
    const uint64_t end = result_used ? fdwic_atomic_result_ready_tick(old) : fdwic_swimlane_detail_now();
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
    const uint64_t end = result_used ? fdwic_atomic_result_ready_tick(old) : fdwic_swimlane_detail_now();
    fdwic_swimlane_count_atomic_call(false);
    fdwic_swimlane_detail_record_atomic(task_id, site, FdwicAtomicOp::FetchMax, begin, end, result_used, return_ready);
    return old;
}

#if PTO_FDWIC_SHARED_MAP
template <typename T>
PTO_DEVICE_FUNC inline T fdwic_trace_atomic_compare_exchange(
    int32_t task_id, FdwicAtomicSite site, __gm__ volatile T &value, T expected, T desired,
    bool result_used = true, int success_memorder = __ATOMIC_ACQ_REL, int failure_memorder = __ATOMIC_ACQUIRE
) {
    if (!fdwic_atomic_swimlane_enabled()) {
        return atomic_compare_exchange(value, expected, desired, success_memorder, failure_memorder);
    }
    const uint64_t begin = fdwic_swimlane_detail_now();
    const T old = atomic_compare_exchange(value, expected, desired, success_memorder, failure_memorder);
    const bool return_ready = result_used && fdwic_atomic_return_ready_observed();
    const uint64_t end = result_used ? fdwic_atomic_result_ready_tick(old) : fdwic_swimlane_detail_now();
    fdwic_swimlane_count_atomic_call(false);
    fdwic_swimlane_detail_record_atomic(
        task_id, site, FdwicAtomicOp::CompareExchange, begin, end, result_used, return_ready
    );
    return old;
}
#endif

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
    const uint64_t dependency_end = fdwic_atomic_result_ready_tick(dependency_value);
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

#else

// 无泳道构建必须在编译期回到原始 atomic，而不是只让运行时 level=0。
// 这样 perf-clock / submit-pmu ELF 不携带 record 分支、轮询聚合状态或
// atomic 观察慢体，避免诊断代码布局反过来污染权威性能基线。
PTO_DEVICE_FUNC inline void fdwic_swimlane_attach(__gm__ Runtime *) {}

PTO_DEVICE_FUNC constexpr uint32_t fdwic_atomic_site_mask(FdwicAtomicSite site) {
#if PTO_FDWIC_SHARED_MAP
    (void)site;
    return 0;
#else
    return 1U << static_cast<uint32_t>(site);
#endif
}

PTO_DEVICE_FUNC inline uint32_t fdwic_atomic_block_won_poll_mask() { return 0; }
PTO_DEVICE_FUNC inline uint32_t fdwic_atomic_poll_region_begin(uint32_t) { return 0; }
PTO_DEVICE_FUNC inline void fdwic_atomic_poll_region_end(uint32_t) {}
PTO_DEVICE_FUNC inline void fdwic_atomic_poll_boundary() {}

template <typename T>
PTO_DEVICE_FUNC inline T fdwic_trace_atomic_load(
    int32_t, FdwicAtomicSite site, __gm__ volatile T &value, bool result_used = true, int memorder = __ATOMIC_ACQUIRE
) {
    const bool observe_return_ready = result_used && fdwic_atomic_site_result_used(site);
    const uint32_t token = observe_return_ready ? fdwic_submit_pmu_return_ready_atomic_begin() : 0;
    const T old = atomic_load(value, memorder);
    if (token != 0) fdwic_submit_pmu_return_ready_atomic_end(token, fdwic_atomic_result_ready_tick(old));
    return old;
}

template <typename T, typename V>
PTO_DEVICE_FUNC inline T fdwic_trace_atomic_exchange(
    int32_t, FdwicAtomicSite site, __gm__ volatile T &value, V desired, bool result_used = false,
    int memorder = __ATOMIC_ACQ_REL
) {
    const bool observe_return_ready = result_used && fdwic_atomic_site_result_used(site);
    const uint32_t token = observe_return_ready ? fdwic_submit_pmu_return_ready_atomic_begin() : 0;
    const T old = atomic_exchange(value, static_cast<T>(desired), memorder);
    if (token != 0) fdwic_submit_pmu_return_ready_atomic_end(token, fdwic_atomic_result_ready_tick(old));
    return old;
}

template <typename T>
PTO_DEVICE_FUNC inline T fdwic_trace_atomic_fetch_add(
    int32_t, FdwicAtomicSite site, __gm__ volatile T &value, T delta, bool result_used = false,
    int memorder = __ATOMIC_ACQ_REL
) {
    const bool observe_return_ready = result_used && fdwic_atomic_site_result_used(site);
    const uint32_t token = observe_return_ready ? fdwic_submit_pmu_return_ready_atomic_begin() : 0;
    const T old = atomic_fetch_add(value, delta, memorder);
    if (token != 0) fdwic_submit_pmu_return_ready_atomic_end(token, fdwic_atomic_result_ready_tick(old));
    return old;
}

template <typename T>
PTO_DEVICE_FUNC inline T fdwic_trace_atomic_fetch_sub(
    int32_t, FdwicAtomicSite site, __gm__ volatile T &value, T delta, bool result_used = false,
    int memorder = __ATOMIC_ACQ_REL
) {
    const bool observe_return_ready = result_used && fdwic_atomic_site_result_used(site);
    const uint32_t token = observe_return_ready ? fdwic_submit_pmu_return_ready_atomic_begin() : 0;
    const T old = atomic_fetch_sub(value, delta, memorder);
    if (token != 0) fdwic_submit_pmu_return_ready_atomic_end(token, fdwic_atomic_result_ready_tick(old));
    return old;
}

template <typename T>
PTO_DEVICE_FUNC inline T fdwic_trace_atomic_fetch_max(
    int32_t, FdwicAtomicSite site, __gm__ volatile T &value, T desired, bool result_used = true,
    int memorder = __ATOMIC_ACQ_REL
) {
    const bool observe_return_ready = result_used && fdwic_atomic_site_result_used(site);
    const uint32_t token = observe_return_ready ? fdwic_submit_pmu_return_ready_atomic_begin() : 0;
    const T old = atomic_fetch_max(value, desired, memorder);
    if (token != 0) fdwic_submit_pmu_return_ready_atomic_end(token, fdwic_atomic_result_ready_tick(old));
    return old;
}

#if PTO_FDWIC_SHARED_MAP
template <typename T>
PTO_DEVICE_FUNC inline T fdwic_trace_atomic_compare_exchange(
    int32_t, FdwicAtomicSite site, __gm__ volatile T &value, T expected, T desired, bool result_used = true,
    int success_memorder = __ATOMIC_ACQ_REL, int failure_memorder = __ATOMIC_ACQUIRE
) {
    const bool observe_return_ready = result_used && fdwic_atomic_site_result_used(site);
    const uint32_t token = observe_return_ready ? fdwic_submit_pmu_return_ready_atomic_begin() : 0;
    const T old = atomic_compare_exchange(value, expected, desired, success_memorder, failure_memorder);
    if (token != 0) fdwic_submit_pmu_return_ready_atomic_end(token, fdwic_atomic_result_ready_tick(old));
    return old;
}

PTO_DEVICE_FUNC inline bool fdwic_swimlane_record_shared_claim(
    __gm__ DistCore *, int32_t, uint64_t, uint64_t, bool
) {
    return false;
}

PTO_DEVICE_FUNC inline bool fdwic_swimlane_record_shared_submit(
    __gm__ DistCore *, int32_t, uint64_t, uint64_t
) {
    return false;
}

PTO_DEVICE_FUNC inline bool fdwic_swimlane_record_captured_atomic(
    int32_t, FdwicAtomicSite, FdwicAtomicOp, uint64_t, uint64_t, bool, bool, bool = false, uint64_t = 0
) {
    return false;
}

PTO_DEVICE_FUNC inline bool fdwic_swimlane_record_aggregate_atomic_poll(
    FdwicAtomicSite, uint64_t, uint64_t, uint32_t, bool
) {
    return false;
}

PTO_DEVICE_FUNC inline bool fdwic_swimlane_record_dcci(
    __gm__ DistCore *, int32_t, int32_t, FdwicDcciSite, FdwicDcciOp, bool, uint32_t, uint32_t, uint64_t, uint64_t
) {
    return false;
}

PTO_DEVICE_FUNC inline bool fdwic_swimlane_record_dcci(
    __gm__ DistCore *, int32_t, int32_t, FdwicDcciSite, bool, uint32_t, uint64_t, uint64_t
) {
    return false;
}
#endif

PTO_DEVICE_FUNC inline bool fdwic_trace_is_fatal(int32_t = -1) {
    return fdwic_trace_atomic_load(-1, FdwicAtomicSite::FatalPoll, g_dist.fatal) != 0;
}

PTO_DEVICE_FUNC inline void fdwic_trace_set_fatal(int32_t = -1) {
    (void)fdwic_trace_atomic_exchange(
        -1, FdwicAtomicSite::FatalSet, g_dist.fatal, int32_t{1}, /*result_used=*/false, __ATOMIC_ACQ_REL
    );
}

PTO_DEVICE_FUNC inline void fdwic_swimlane_record_clock_baselines(__gm__ DistCore *, int32_t) {}

#endif

}  // namespace
