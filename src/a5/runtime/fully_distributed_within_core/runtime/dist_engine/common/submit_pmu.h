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

#include "dist_engine/common/target.h"

#if PTO_FDWIC_SUBMIT_PMU

#include "aicore/aicore.h"
#include "aicore/fdwic_submit_pmu_state.h"
#include "common/platform_config.h"
#include "dist_engine/aicore/primitive.h"
#include "dist_engine/common/submit_pmu_types.h"
#include "dist_engine/common/worker_state.h"

namespace {

#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 0
constexpr FdwicSubmitPmuPhase kFdwicSubmitPmuCompiledPhase = FdwicSubmitPmuPhase::None;
constexpr uint16_t kFdwicSubmitPmuCompiledMode = kFdwicSubmitPmuModeNone;
constexpr size_t kFdwicSubmitPmuCompiledBytes = kFdwicSubmitPmuNoneBytes;
#elif PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 1
constexpr FdwicSubmitPmuPhase kFdwicSubmitPmuCompiledPhase = FdwicSubmitPmuPhase::ArgBuild;
constexpr uint16_t kFdwicSubmitPmuCompiledMode = kFdwicSubmitPmuModeArgBuild;
constexpr size_t kFdwicSubmitPmuCompiledBytes = kFdwicSubmitPmuPhaseBytes;
#elif PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 2
constexpr FdwicSubmitPmuPhase kFdwicSubmitPmuCompiledPhase = FdwicSubmitPmuPhase::EmptyBracket;
constexpr uint16_t kFdwicSubmitPmuCompiledMode = kFdwicSubmitPmuModeEmptyBracket;
constexpr size_t kFdwicSubmitPmuCompiledBytes = kFdwicSubmitPmuPhaseBytes;
#elif PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 3
constexpr FdwicSubmitPmuPhase kFdwicSubmitPmuCompiledPhase = FdwicSubmitPmuPhase::Materialize;
constexpr uint16_t kFdwicSubmitPmuCompiledMode = kFdwicSubmitPmuModeMaterialize;
constexpr size_t kFdwicSubmitPmuCompiledBytes = kFdwicSubmitPmuPhaseBytes;
#elif PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 4
constexpr FdwicSubmitPmuPhase kFdwicSubmitPmuCompiledPhase = FdwicSubmitPmuPhase::Claim;
constexpr uint16_t kFdwicSubmitPmuCompiledMode = kFdwicSubmitPmuModeClaim;
constexpr size_t kFdwicSubmitPmuCompiledBytes = kFdwicSubmitPmuPhaseBytes;
#else
#error "invalid real FDWIC submit-PMU phase"
#endif

template <uint32_t BlockOffset, uint32_t RegisterOffset>
PTO_DEVICE_FUNC inline uint32_t fdwic_submit_pmu_ld(uint64_t reg_base) {
    int32_t *block = reinterpret_cast<int32_t *>(reg_base + BlockOffset);
    return static_cast<uint32_t>(ld_dev(block, static_cast<int16_t>(RegisterOffset - BlockOffset)));
}

PTO_DEVICE_FUNC inline void fdwic_submit_pmu_clear_counters(uint64_t reg_base) {
    constexpr uint32_t kCounterBlock = REG_MMIO_PMU_CTRL_0_OFFSET;
    (void)fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT0_OFFSET>(reg_base);
    (void)fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT1_OFFSET>(reg_base);
    (void)fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT2_OFFSET>(reg_base);
    (void)fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT3_OFFSET>(reg_base);
    (void)fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT4_OFFSET>(reg_base);
    (void)fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT5_OFFSET>(reg_base);
    (void)fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT6_OFFSET>(reg_base);
    (void)fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT7_OFFSET>(reg_base);
    (void)fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT8_OFFSET>(reg_base);
    (void)fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT9_OFFSET>(reg_base);
    (void)fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT_TOTAL0_OFFSET>(reg_base);
    (void)fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT_TOTAL1_OFFSET>(reg_base);
}

PTO_DEVICE_FUNC inline uint32_t fdwic_submit_pmu_selector_status(uint64_t reg_base) {
    constexpr uint32_t kSelectorBlock = REG_MMIO_PMU_CTRL_1_OFFSET;
    uint32_t status = 0;
    if (fdwic_submit_pmu_ld<kSelectorBlock, REG_MMIO_PMU_CNT2_IDX_OFFSET>(reg_base) == kFdwicSubmitPmuCnt2ScalarBusy) {
        status |= kFdwicSubmitPmuCnt2SelectorValid;
    }
    if (fdwic_submit_pmu_ld<kSelectorBlock, REG_MMIO_PMU_CNT5_IDX_OFFSET>(reg_base) ==
        kFdwicSubmitPmuCnt5ShadowIcacheMiss) {
        status |= kFdwicSubmitPmuCnt5SelectorValid;
    }
    if (fdwic_submit_pmu_ld<kSelectorBlock, REG_MMIO_PMU_CNT6_IDX_OFFSET>(reg_base) ==
        kFdwicSubmitPmuCnt6IcacheRequest) {
        status |= kFdwicSubmitPmuCnt6SelectorValid;
    }
    if (fdwic_submit_pmu_ld<kSelectorBlock, REG_MMIO_PMU_CNT7_IDX_OFFSET>(reg_base) == kFdwicSubmitPmuCnt7IcacheMiss) {
        status |= kFdwicSubmitPmuCnt7SelectorValid;
    }
    if (fdwic_submit_pmu_ld<kSelectorBlock, REG_MMIO_PMU_CNT8_IDX_OFFSET>(reg_base) ==
        kFdwicSubmitPmuCnt8ShadowIcacheRequest) {
        status |= kFdwicSubmitPmuCnt8SelectorValid;
    }
    return status;
}

#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
struct FdwicSubmitPmuShadowSnapshot {
    uint32_t requests;
    uint32_t misses;
};

// noinline 同时是 phase ELF 门禁：none 中不得出现该符号。CNT8/CNT5 是
// 顺序 read-to-clear，不是同一时刻的原子快照。
PTO_DEVICE_FUNC __attribute__((noinline)) FdwicSubmitPmuShadowSnapshot fdwic_submit_pmu_phase_read_shadow_counters() {
    constexpr uint32_t kCounterBlock = REG_MMIO_PMU_CTRL_0_OFFSET;
    return FdwicSubmitPmuShadowSnapshot{
        fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT8_OFFSET>(g_fdwic_submit_pmu_reg_base),
        fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT5_OFFSET>(g_fdwic_submit_pmu_reg_base),
    };
}

PTO_DEVICE_FUNC inline bool
fdwic_submit_pmu_add_shadow(const FdwicSubmitPmuShadowSnapshot &sample, bool include_in_phase) {
    FdwicSubmitPmuPhaseAccumulator &phase = g_fdwic_submit_pmu_phase;
    if (phase.shadow_requests > UINT64_MAX - sample.requests || phase.shadow_misses > UINT64_MAX - sample.misses) {
        return false;
    }
    phase.shadow_requests += sample.requests;
    phase.shadow_misses += sample.misses;
    if (include_in_phase) {
        if (phase.phase_requests > UINT64_MAX - sample.requests || phase.phase_misses > UINT64_MAX - sample.misses) {
            return false;
        }
        phase.phase_requests += sample.requests;
        phase.phase_misses += sample.misses;
    }
    if (sample.requests > phase.max_shadow_request_chunk) phase.max_shadow_request_chunk = sample.requests;
    if (sample.misses > phase.max_shadow_miss_chunk) phase.max_shadow_miss_chunk = sample.misses;
    return true;
}
#endif

// noinline 是构建门禁的一部分：最终 ELF 必须能证明完整 Submit 计数读取
// 存在，同时普通泳道/perf-clock ELF 必须不含该符号。
PTO_DEVICE_FUNC __attribute__((noinline)) void fdwic_submit_pmu_read_counters() {
    const uint64_t reg_base = g_fdwic_submit_pmu_reg_base;
    if (reg_base == 0) return;
    constexpr uint32_t kCounterBlock = REG_MMIO_PMU_CTRL_0_OFFSET;
    g_fdwic_submit_pmu_scalar_busy = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT2_OFFSET>(reg_base);
    g_fdwic_submit_pmu_icache_requests = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT6_OFFSET>(reg_base);
    g_fdwic_submit_pmu_icache_misses = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT7_OFFSET>(reg_base);
    // primary 必须先读且窗口中从不 read-clear。none 随后直接读取一次
    // shadow；phase 则在这里补最后一个 tail segment，软件重建完整 shadow。
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 0
    g_fdwic_submit_pmu_shadow_requests = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT8_OFFSET>(reg_base);
    g_fdwic_submit_pmu_shadow_misses = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT5_OFFSET>(reg_base);
#else
    const FdwicSubmitPmuShadowSnapshot tail = fdwic_submit_pmu_phase_read_shadow_counters();
    if (fdwic_submit_pmu_add_shadow(tail, /*include_in_phase=*/false)) {
        g_fdwic_submit_pmu_phase.status |= kFdwicSubmitPmuPhaseTailRead;
    } else {
        g_fdwic_submit_pmu_phase.boundary_error = true;
    }
    g_fdwic_submit_pmu_shadow_requests = static_cast<uint32_t>(g_fdwic_submit_pmu_phase.shadow_requests);
    g_fdwic_submit_pmu_shadow_misses = static_cast<uint32_t>(g_fdwic_submit_pmu_phase.shadow_misses);
#endif
    const uint64_t low = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT_TOTAL0_OFFSET>(reg_base);
    const uint64_t high = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT_TOTAL1_OFFSET>(reg_base);
    g_fdwic_submit_pmu_total_cycles = low | (high << 32);
}

PTO_DEVICE_FUNC inline void fdwic_submit_pmu_reset_local() {
    g_fdwic_submit_pmu_core = nullptr;
    g_fdwic_submit_pmu_phase_core = nullptr;
    g_fdwic_submit_pmu_reg_base = 0;
    g_fdwic_submit_pmu_start_tick = 0;
    g_fdwic_submit_pmu_end_tick = 0;
    g_fdwic_submit_pmu_total_cycles = 0;
    g_fdwic_submit_pmu_scalar_busy = 0;
    g_fdwic_submit_pmu_icache_requests = 0;
    g_fdwic_submit_pmu_icache_misses = 0;
    g_fdwic_submit_pmu_shadow_requests = 0;
    g_fdwic_submit_pmu_shadow_misses = 0;
    g_fdwic_submit_pmu_expected_submits = 0;
    g_fdwic_submit_pmu_status = 0;
    g_fdwic_submit_pmu_started = false;
    g_fdwic_submit_pmu_stopped = false;
    g_fdwic_submit_pmu_phase.shadow_requests = 0;
    g_fdwic_submit_pmu_phase.shadow_misses = 0;
    g_fdwic_submit_pmu_phase.phase_requests = 0;
    g_fdwic_submit_pmu_phase.phase_misses = 0;
    g_fdwic_submit_pmu_phase.phase_elapsed_ticks = 0;
    g_fdwic_submit_pmu_phase.phase_begin_tick = 0;
    g_fdwic_submit_pmu_phase.begin_reads = 0;
    g_fdwic_submit_pmu_phase.end_reads = 0;
    g_fdwic_submit_pmu_phase.max_shadow_request_chunk = 0;
    g_fdwic_submit_pmu_phase.max_shadow_miss_chunk = 0;
    g_fdwic_submit_pmu_phase.status = 0;
    g_fdwic_submit_pmu_phase.armed = false;
    g_fdwic_submit_pmu_phase.boundary_error = false;
}

PTO_DEVICE_FUNC inline void fdwic_submit_pmu_attach(__gm__ Runtime *runtime, __gm__ DistCore *self) {
    fdwic_submit_pmu_reset_local();
    if (runtime == nullptr || self == nullptr) return;
    g_fdwic_submit_pmu_status = kFdwicSubmitPmuRequested;
    dist_aicore_invalidate_region(const_cast<__gm__ uint64_t *>(&runtime->dist.swimlane_base), 64);
    const uint64_t base = runtime->dist.swimlane_base;
    if (base == 0 || self->core_idx < 0 || self->core_idx >= static_cast<int32_t>(kFdwicSubmitPmuExpectedCores)) {
        return;
    }
    __gm__ auto *header = reinterpret_cast<__gm__ FdwicSubmitPmuHeader *>(base);
    dist_aicore_invalidate_region(header, 128);
    if (header->magic != kFdwicSubmitPmuMagic || header->version != kFdwicSubmitPmuVersion ||
        header->mode != kFdwicSubmitPmuCompiledMode || header->header_bytes != kFdwicSubmitPmuCompiledBytes ||
        header->record_bytes != sizeof(FdwicSubmitPmuCoreData) || header->num_cores != kFdwicSubmitPmuExpectedCores) {
        return;
    }
    g_fdwic_submit_pmu_core = &header->cores[self->core_idx];
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
    g_fdwic_submit_pmu_phase_core =
        &reinterpret_cast<__gm__ FdwicSubmitPmuPhaseCoreData *>(base + sizeof(FdwicSubmitPmuHeader))[self->core_idx];
    g_fdwic_submit_pmu_phase.status = kFdwicSubmitPmuPhaseRequested;
#endif
    g_fdwic_submit_pmu_reg_base = get_fdwic_submit_pmu_reg_base();
    if (g_fdwic_submit_pmu_reg_base == 0) return;
    g_fdwic_submit_pmu_status |= kFdwicSubmitPmuRegMapped;
    const uint32_t physical_core_id = get_physical_core_id();
    if (physical_core_id >= kFdwicSubmitPmuPhysicalSubcores) return;
    g_fdwic_submit_pmu_status |= kFdwicSubmitPmuPhysicalIdValid;

    // Owner 在唤醒 worker 前已经完成 MMIO 配置。先冻结本核 AICore CTRL gate，
    // 再验证 selector 并读清 owner 配置到首个 Submit 之间的冷路径计数。
    bisheng::cce::metrics_prof_stop();
    g_fdwic_submit_pmu_status |= fdwic_submit_pmu_selector_status(g_fdwic_submit_pmu_reg_base);
    fdwic_submit_pmu_clear_counters(g_fdwic_submit_pmu_reg_base);
}

PTO_DEVICE_FUNC inline void fdwic_submit_pmu_expect_submits(uint32_t expected_submits) {
    g_fdwic_submit_pmu_expected_submits = expected_submits;
}

template <FdwicSubmitPmuPhase Phase>
PTO_DEVICE_FUNC inline void fdwic_submit_pmu_phase_begin() {
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
    if constexpr (Phase == kFdwicSubmitPmuCompiledPhase) {
        FdwicSubmitPmuPhaseAccumulator &phase = g_fdwic_submit_pmu_phase;
        if (!g_fdwic_submit_pmu_started || g_fdwic_submit_pmu_stopped || g_fdwic_submit_pmu_phase_core == nullptr ||
            g_fdwic_submit_pmu_reg_base == 0 || phase.armed) {
            phase.boundary_error = true;
            return;
        }
        const FdwicSubmitPmuShadowSnapshot sample = fdwic_submit_pmu_phase_read_shadow_counters();
        if (!fdwic_submit_pmu_add_shadow(sample, /*include_in_phase=*/false)) {
            phase.boundary_error = true;
            return;
        }
        ++phase.begin_reads;
        phase.armed = true;
        // 起点位于 begin read-clear/bookkeeping 之后。empty-bracket wrapper
        // 会在外层另行覆盖为完整 begin/end 对的经验耗时。
        phase.phase_begin_tick = get_sys_cnt_aicore();
    }
#endif
}

template <FdwicSubmitPmuPhase Phase>
PTO_DEVICE_FUNC inline void fdwic_submit_pmu_phase_end() {
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
    if constexpr (Phase == kFdwicSubmitPmuCompiledPhase) {
        const uint64_t phase_end_tick = get_sys_cnt_aicore();
        FdwicSubmitPmuPhaseAccumulator &phase = g_fdwic_submit_pmu_phase;
        if (!g_fdwic_submit_pmu_started || g_fdwic_submit_pmu_stopped || g_fdwic_submit_pmu_phase_core == nullptr ||
            g_fdwic_submit_pmu_reg_base == 0 || !phase.armed || phase_end_tick < phase.phase_begin_tick ||
            phase.phase_elapsed_ticks > UINT64_MAX - (phase_end_tick - phase.phase_begin_tick)) {
            phase.boundary_error = true;
            return;
        }
        phase.phase_elapsed_ticks += phase_end_tick - phase.phase_begin_tick;
        // 终点位于两次 read-clear 之前，阶段时间不包含 end 读取开销；end
        // segment 同时进入完整 shadow 软件重建与局部阶段观测值。begin/end
        // 两侧还存在少量观测 bookkeeping，不能把它解释成原业务事件数的
        // 数学下界。
        const FdwicSubmitPmuShadowSnapshot sample = fdwic_submit_pmu_phase_read_shadow_counters();
        if (!fdwic_submit_pmu_add_shadow(sample, /*include_in_phase=*/true)) {
            phase.boundary_error = true;
            return;
        }
        ++phase.end_reads;
        phase.armed = false;
        phase.phase_begin_tick = 0;
    }
#endif
}

// 在 Claim.end 调用点量化一对原样 running begin/end observer。外层 SYS_CNT
// 位于两次 shadow read-clear 之外，因此不会主动进入局部 request/miss
// observed；它自身仍是计时边界底噪，结果只能作经验量尺。
PTO_DEVICE_FUNC inline void fdwic_submit_pmu_empty_bracket_calibrate() {
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 2
    FdwicSubmitPmuPhaseAccumulator &phase = g_fdwic_submit_pmu_phase;
    const uint64_t old_elapsed = phase.phase_elapsed_ticks;
    const uint32_t old_begin_reads = phase.begin_reads;
    const uint32_t old_end_reads = phase.end_reads;
    const uint64_t outer_begin = get_sys_cnt_aicore();
    fdwic_submit_pmu_phase_begin<FdwicSubmitPmuPhase::EmptyBracket>();
    fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::EmptyBracket>();
    const uint64_t outer_end = get_sys_cnt_aicore();
    if (phase.boundary_error || phase.armed || phase.begin_reads != old_begin_reads + 1U ||
        phase.end_reads != old_end_reads + 1U || outer_end < outer_begin ||
        old_elapsed > UINT64_MAX - (outer_end - outer_begin)) {
        phase.boundary_error = true;
        return;
    }
    phase.phase_elapsed_ticks = old_elapsed + outer_end - outer_begin;
#endif
}

PTO_DEVICE_FUNC inline void fdwic_submit_pmu_submit_begin(int32_t task_id) {
    constexpr uint32_t kReadyMask = (1U << 8) - 1U;
    if (task_id != 0 || g_fdwic_submit_pmu_core == nullptr || g_fdwic_submit_pmu_expected_submits == 0 ||
        (g_fdwic_submit_pmu_status & kReadyMask) != kReadyMask) {
        return;
    }
    // SYS_CNT 包围的是同一业务挂点；PMU gate 的 PIPE_ALL 成本位于 PMU
    // window 外，不能把该 tick 区间与另一 ELF 的绝对时间直接相减。
    g_fdwic_submit_pmu_start_tick = get_sys_cnt_aicore();
    bisheng::cce::metrics_prof_start();
    g_fdwic_submit_pmu_started = true;
    g_fdwic_submit_pmu_status |= kFdwicSubmitPmuWindowStarted;
}

PTO_DEVICE_FUNC inline void fdwic_submit_pmu_submit_end(int32_t task_id) {
    if (!g_fdwic_submit_pmu_started || g_fdwic_submit_pmu_stopped || task_id < 0 ||
        static_cast<uint32_t>(task_id + 1) != g_fdwic_submit_pmu_expected_submits) {
        return;
    }
    bisheng::cce::metrics_prof_stop();
    g_fdwic_submit_pmu_end_tick = get_sys_cnt_aicore();
    fdwic_submit_pmu_read_counters();
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
    FdwicSubmitPmuPhaseAccumulator &phase = g_fdwic_submit_pmu_phase;
    if (!phase.boundary_error && !phase.armed && phase.begin_reads == phase.end_reads) {
        phase.status |= kFdwicSubmitPmuPhaseBoundaryBalanced;
    }
    if (phase.begin_reads == g_fdwic_submit_pmu_expected_submits &&
        phase.end_reads == g_fdwic_submit_pmu_expected_submits) {
        phase.status |= kFdwicSubmitPmuPhaseShapeValid;
    }
    if (phase.phase_requests <= phase.shadow_requests && phase.phase_misses <= phase.shadow_misses &&
        phase.shadow_misses <= phase.shadow_requests && phase.shadow_requests <= g_fdwic_submit_pmu_icache_requests &&
        phase.shadow_misses <= g_fdwic_submit_pmu_icache_misses) {
        phase.status |= kFdwicSubmitPmuPhaseValuesOrdered;
    }
    if (phase.phase_elapsed_ticks != 0 &&
        phase.phase_elapsed_ticks <= g_fdwic_submit_pmu_end_tick - g_fdwic_submit_pmu_start_tick) {
        phase.status |= kFdwicSubmitPmuPhaseTimeValid;
    }
#endif
    g_fdwic_submit_pmu_stopped = true;
    g_fdwic_submit_pmu_status |= kFdwicSubmitPmuWindowStopped;
    if (g_fdwic_submit_pmu_total_cycles != 0) {
        g_fdwic_submit_pmu_status |= kFdwicSubmitPmuTotalNonzero;
    }
}

PTO_DEVICE_FUNC inline void fdwic_submit_pmu_flush(__gm__ DistCore *self) {
    if (g_fdwic_submit_pmu_started && !g_fdwic_submit_pmu_stopped) {
        // 只负责关闭遗留 gate，故意不伪造 WindowStopped。host 会因闭合失败
        // 拒绝正式 raw；FinalDrain 也不会被包装成有效 Submit 样本。
        bisheng::cce::metrics_prof_stop();
    }
    __gm__ FdwicSubmitPmuCoreData *core = g_fdwic_submit_pmu_core;
    if (core == nullptr || self == nullptr) return;
    core->first_submit_start_tick = g_fdwic_submit_pmu_start_tick;
    core->last_submit_end_tick = g_fdwic_submit_pmu_end_tick;
    core->total_cycles = g_fdwic_submit_pmu_total_cycles;
    core->scalar_busy = g_fdwic_submit_pmu_scalar_busy;
    core->icache_requests = g_fdwic_submit_pmu_icache_requests;
    core->icache_misses = g_fdwic_submit_pmu_icache_misses;
    core->shadow_icache_requests = g_fdwic_submit_pmu_shadow_requests;
    core->shadow_icache_misses = g_fdwic_submit_pmu_shadow_misses;
    core->submit_count = static_cast<uint32_t>(self->local_index);
    core->expected_submit_count = g_fdwic_submit_pmu_expected_submits;
    core->logical_core_id = static_cast<uint16_t>(self->core_idx);
    core->physical_core_id = static_cast<uint16_t>(get_physical_core_id());
    core->block_id = static_cast<uint16_t>(self->block_id);
    core->lane = static_cast<uint16_t>(self->lane);
    core->status = g_fdwic_submit_pmu_status;
    dist_aicore_flush_region(core, sizeof(FdwicSubmitPmuCoreData));
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
    __gm__ FdwicSubmitPmuPhaseCoreData *phase_core = g_fdwic_submit_pmu_phase_core;
    if (phase_core == nullptr) return;
    const FdwicSubmitPmuPhaseAccumulator &phase = g_fdwic_submit_pmu_phase;
    phase_core->phase_elapsed_ticks = phase.phase_elapsed_ticks;
    phase_core->phase_icache_requests_observed = phase.phase_requests;
    phase_core->phase_icache_misses_observed = phase.phase_misses;
    phase_core->phase_id = static_cast<uint32_t>(kFdwicSubmitPmuCompiledPhase);
    phase_core->phase_begin_reads = phase.begin_reads;
    phase_core->phase_end_reads = phase.end_reads;
    phase_core->max_shadow_request_chunk = phase.max_shadow_request_chunk;
    phase_core->max_shadow_miss_chunk = phase.max_shadow_miss_chunk;
    phase_core->status = phase.status;
    for (uint32_t index = 0; index < 4U; ++index)
        phase_core->reserved[index] = 0;
    dist_aicore_flush_region(phase_core, sizeof(FdwicSubmitPmuPhaseCoreData));
#endif
}

}  // namespace

#else

namespace {
PTO_DEVICE_FUNC inline void fdwic_submit_pmu_attach(__gm__ Runtime *, __gm__ DistCore *) {}
PTO_DEVICE_FUNC inline void fdwic_submit_pmu_expect_submits(uint32_t) {}
PTO_DEVICE_FUNC inline void fdwic_submit_pmu_submit_begin(int32_t) {}
PTO_DEVICE_FUNC inline void fdwic_submit_pmu_submit_end(int32_t) {}
PTO_DEVICE_FUNC inline void fdwic_submit_pmu_flush(__gm__ DistCore *) {}
template <FdwicSubmitPmuPhase Phase>
PTO_DEVICE_FUNC inline void fdwic_submit_pmu_phase_begin() {}
template <FdwicSubmitPmuPhase Phase>
PTO_DEVICE_FUNC inline void fdwic_submit_pmu_phase_end() {}
PTO_DEVICE_FUNC inline void fdwic_submit_pmu_empty_bracket_calibrate() {}
}  // namespace

#endif
