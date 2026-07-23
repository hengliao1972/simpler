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
#elif PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 5
constexpr FdwicSubmitPmuPhase kFdwicSubmitPmuCompiledPhase = FdwicSubmitPmuPhase::Register;
constexpr uint16_t kFdwicSubmitPmuCompiledMode = kFdwicSubmitPmuModeRegister;
constexpr size_t kFdwicSubmitPmuCompiledBytes = kFdwicSubmitPmuPhaseBytes;
#elif PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 6
constexpr FdwicSubmitPmuPhase kFdwicSubmitPmuCompiledPhase = FdwicSubmitPmuPhase::SubmitTransition;
constexpr uint16_t kFdwicSubmitPmuCompiledMode = kFdwicSubmitPmuModeSubmitTransition;
constexpr size_t kFdwicSubmitPmuCompiledBytes = kFdwicSubmitPmuPhaseBytes;
#elif PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 7
constexpr FdwicSubmitPmuPhase kFdwicSubmitPmuCompiledPhase = FdwicSubmitPmuPhase::EfDrainControl;
constexpr uint16_t kFdwicSubmitPmuCompiledMode = kFdwicSubmitPmuModeEfDrainControl;
constexpr size_t kFdwicSubmitPmuCompiledBytes = kFdwicSubmitPmuPhaseBytes;
#elif PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 8
constexpr FdwicSubmitPmuPhase kFdwicSubmitPmuCompiledPhase = FdwicSubmitPmuPhase::PrepareMap;
constexpr uint16_t kFdwicSubmitPmuCompiledMode = kFdwicSubmitPmuModePrepareMap;
constexpr size_t kFdwicSubmitPmuCompiledBytes = kFdwicSubmitPmuPhaseBytes;
#elif PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 9
constexpr FdwicSubmitPmuPhase kFdwicSubmitPmuCompiledPhase = FdwicSubmitPmuPhase::Fanin;
constexpr uint16_t kFdwicSubmitPmuCompiledMode = kFdwicSubmitPmuModeFanin;
constexpr size_t kFdwicSubmitPmuCompiledBytes = kFdwicSubmitPmuPhaseBytes;
#elif PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 10
constexpr FdwicSubmitPmuPhase kFdwicSubmitPmuCompiledPhase = FdwicSubmitPmuPhase::WinnerBuild;
constexpr uint16_t kFdwicSubmitPmuCompiledMode = kFdwicSubmitPmuModeWinnerBuild;
constexpr size_t kFdwicSubmitPmuCompiledBytes = kFdwicSubmitPmuPhaseBytes;
#elif PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 11
constexpr FdwicSubmitPmuPhase kFdwicSubmitPmuCompiledPhase = FdwicSubmitPmuPhase::AllocComplete;
constexpr uint16_t kFdwicSubmitPmuCompiledMode = kFdwicSubmitPmuModeAllocComplete;
constexpr size_t kFdwicSubmitPmuCompiledBytes = kFdwicSubmitPmuPhaseBytes;
#elif PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 12
constexpr FdwicSubmitPmuPhase kFdwicSubmitPmuCompiledPhase = FdwicSubmitPmuPhase::LoserReplay;
constexpr uint16_t kFdwicSubmitPmuCompiledMode = kFdwicSubmitPmuModeLoserReplay;
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
    if (fdwic_submit_pmu_ld<kSelectorBlock, REG_MMIO_PMU_CNT0_IDX_OFFSET>(reg_base) == kFdwicSubmitPmuCnt0VectorBusy) {
        status |= kFdwicSubmitPmuCnt0SelectorValid;
    }
    if (fdwic_submit_pmu_ld<kSelectorBlock, REG_MMIO_PMU_CNT1_IDX_OFFSET>(reg_base) == kFdwicSubmitPmuCnt1CubeBusy) {
        status |= kFdwicSubmitPmuCnt1SelectorValid;
    }
    if (fdwic_submit_pmu_ld<kSelectorBlock, REG_MMIO_PMU_CNT2_IDX_OFFSET>(reg_base) == kFdwicSubmitPmuCnt2ScalarBusy) {
        status |= kFdwicSubmitPmuCnt2SelectorValid;
    }
    if (fdwic_submit_pmu_ld<kSelectorBlock, REG_MMIO_PMU_CNT3_IDX_OFFSET>(reg_base) ==
        kFdwicSubmitPmuCnt3ShadowScalarBusy) {
        status |= kFdwicSubmitPmuCnt3SelectorValid;
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
struct FdwicSubmitPmuIcacheShadowSnapshot {
    uint32_t requests;
    uint32_t misses;
};

struct FdwicSubmitPmuTotalShadowSnapshot {
    uint32_t low;
    uint32_t high;
};

// 三类 shadow 都在 PMU gate 运行时 read-to-clear。没有 DSB/PIPE_ALL，
// 相邻寄存器也不是同一时刻的原子快照；因此 raw 明确使用 observed 命名。
// 这些 noinline 符号同时是 phase ELF 门禁，none 中不得出现。
PTO_DEVICE_FUNC __attribute__((noinline)) FdwicSubmitPmuIcacheShadowSnapshot
fdwic_submit_pmu_phase_read_shadow_counters() {
    constexpr uint32_t kCounterBlock = REG_MMIO_PMU_CTRL_0_OFFSET;
    return FdwicSubmitPmuIcacheShadowSnapshot{
        fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT8_OFFSET>(g_fdwic_submit_pmu_reg_base),
        fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT5_OFFSET>(g_fdwic_submit_pmu_reg_base),
    };
}

PTO_DEVICE_FUNC __attribute__((noinline)) uint32_t fdwic_submit_pmu_phase_read_scalar_shadow() {
    constexpr uint32_t kCounterBlock = REG_MMIO_PMU_CTRL_0_OFFSET;
    return fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT3_OFFSET>(g_fdwic_submit_pmu_reg_base);
}

PTO_DEVICE_FUNC __attribute__((noinline)) FdwicSubmitPmuTotalShadowSnapshot fdwic_submit_pmu_phase_read_total_shadow() {
    constexpr uint32_t kCounterBlock = REG_MMIO_PMU_CTRL_0_OFFSET;
    return FdwicSubmitPmuTotalShadowSnapshot{
        fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT_TOTAL0_OFFSET>(g_fdwic_submit_pmu_reg_base),
        fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT_TOTAL1_OFFSET>(g_fdwic_submit_pmu_reg_base),
    };
}

PTO_DEVICE_FUNC inline bool
fdwic_submit_pmu_add_total_shadow(const FdwicSubmitPmuTotalShadowSnapshot &sample, bool include_in_phase) {
    FdwicSubmitPmuPhaseAccumulator &phase = g_fdwic_submit_pmu_phase;
    // 当前约 5 ms 窗和更短的 phase chunk 不应跨越 32-bit。要求 high==0
    // 既规避 low/high 顺序读取的 rollover 歧义，也让运行中重建失败可见。
    if (sample.high != 0 || phase.shadow_total_cycles > UINT64_MAX - sample.low ||
        (include_in_phase && phase.phase_total_cycles > UINT64_MAX - sample.low)) {
        phase.counter_error = true;
        return false;
    }
    phase.shadow_total_cycles += sample.low;
    if (include_in_phase) phase.phase_total_cycles += sample.low;
    return true;
}

PTO_DEVICE_FUNC inline bool fdwic_submit_pmu_add_scalar_shadow(uint32_t sample, bool include_in_phase) {
    FdwicSubmitPmuPhaseAccumulator &phase = g_fdwic_submit_pmu_phase;
    if (phase.shadow_scalar_busy > UINT64_MAX - sample ||
        (include_in_phase && phase.phase_scalar_busy > UINT64_MAX - sample)) {
        phase.counter_error = true;
        return false;
    }
    phase.shadow_scalar_busy += sample;
    if (include_in_phase) phase.phase_scalar_busy += sample;
    return true;
}

PTO_DEVICE_FUNC inline bool
fdwic_submit_pmu_add_icache_shadow(const FdwicSubmitPmuIcacheShadowSnapshot &sample, bool include_in_phase) {
    FdwicSubmitPmuPhaseAccumulator &phase = g_fdwic_submit_pmu_phase;
    if (phase.shadow_requests > UINT64_MAX - sample.requests || phase.shadow_misses > UINT64_MAX - sample.misses) {
        phase.counter_error = true;
        return false;
    }
    phase.shadow_requests += sample.requests;
    phase.shadow_misses += sample.misses;
    if (include_in_phase) {
        if (phase.phase_requests > UINT64_MAX - sample.requests || phase.phase_misses > UINT64_MAX - sample.misses) {
            phase.counter_error = true;
            return false;
        }
        phase.phase_requests += sample.requests;
        phase.phase_misses += sample.misses;
    }
    return true;
}
#endif

// noinline 是构建门禁的一部分：最终 ELF 必须能证明完整 Submit 计数读取
// 存在，同时普通泳道/perf-clock ELF 必须不含该符号。
PTO_DEVICE_FUNC __attribute__((noinline)) void fdwic_submit_pmu_read_counters() {
    const uint64_t reg_base = g_fdwic_submit_pmu_reg_base;
    if (reg_base == 0) return;
    constexpr uint32_t kCounterBlock = REG_MMIO_PMU_CTRL_0_OFFSET;
    const uint32_t vector_busy = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT0_OFFSET>(reg_base);
    const uint32_t cube_busy = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT1_OFFSET>(reg_base);
    if (vector_busy == 0) g_fdwic_submit_pmu_status |= kFdwicSubmitPmuVectorBusyZero;
    if (cube_busy == 0) g_fdwic_submit_pmu_status |= kFdwicSubmitPmuCubeBusyZero;
    g_fdwic_submit_pmu_scalar_busy = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT2_OFFSET>(reg_base);
    g_fdwic_submit_pmu_icache_requests = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT6_OFFSET>(reg_base);
    g_fdwic_submit_pmu_icache_misses = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT7_OFFSET>(reg_base);
    // primary 必须先读且窗口中从不 read-clear。none 随后直接读取一次
    // shadow；phase 则在这里补最后一个 tail segment，软件重建完整 shadow。
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 0
    const uint32_t shadow_scalar_busy = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT3_OFFSET>(reg_base);
    const uint32_t shadow_requests = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT8_OFFSET>(reg_base);
    const uint32_t shadow_misses = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT5_OFFSET>(reg_base);
    if (shadow_requests == g_fdwic_submit_pmu_icache_requests && shadow_misses == g_fdwic_submit_pmu_icache_misses) {
        g_fdwic_submit_pmu_status |= kFdwicSubmitPmuNoneIcacheShadowPrimaryMatch;
    }
    if (shadow_scalar_busy == g_fdwic_submit_pmu_scalar_busy) {
        g_fdwic_submit_pmu_status |= kFdwicSubmitPmuNoneScalarShadowPrimaryMatch;
    }
    const uint64_t low = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT_TOTAL0_OFFSET>(reg_base);
    const uint64_t high = fdwic_submit_pmu_ld<kCounterBlock, REG_MMIO_PMU_CNT_TOTAL1_OFFSET>(reg_base);
    g_fdwic_submit_pmu_total_cycles = low | (high << 32);
#else
    // gate 已停止；仍沿 phase end 的 I-cache -> scalar -> TOTAL 反序补尾。
    const FdwicSubmitPmuIcacheShadowSnapshot icache_tail = fdwic_submit_pmu_phase_read_shadow_counters();
    const uint32_t scalar_tail = fdwic_submit_pmu_phase_read_scalar_shadow();
    const FdwicSubmitPmuTotalShadowSnapshot total_tail = fdwic_submit_pmu_phase_read_total_shadow();
    FdwicSubmitPmuPhaseAccumulator &phase = g_fdwic_submit_pmu_phase;
    if (fdwic_submit_pmu_add_icache_shadow(icache_tail, /*include_in_phase=*/false)) {
        phase.status |= kFdwicSubmitPmuPhaseIcacheTailRead;
    } else {
        phase.boundary_error = true;
    }
    if (fdwic_submit_pmu_add_scalar_shadow(scalar_tail, /*include_in_phase=*/false)) {
        phase.status |= kFdwicSubmitPmuPhaseScalarTailRead;
    } else {
        phase.boundary_error = true;
    }
    if (fdwic_submit_pmu_add_total_shadow(total_tail, /*include_in_phase=*/false)) {
        phase.status |= kFdwicSubmitPmuPhaseTotalTailRead;
    } else {
        phase.boundary_error = true;
    }
    g_fdwic_submit_pmu_total_cycles = phase.shadow_total_cycles;
#endif
}

PTO_DEVICE_FUNC inline void fdwic_submit_pmu_reset_local() {
    g_fdwic_submit_pmu_core = nullptr;
    g_fdwic_submit_pmu_phase_core = nullptr;
    g_fdwic_submit_pmu_reg_base = 0;
    g_fdwic_submit_pmu_start_tick = 0;
    g_fdwic_submit_pmu_end_tick = 0;
    g_fdwic_submit_pmu_scalar_elapsed_ticks = 0;
    g_fdwic_submit_pmu_scalar_segment_begin_tick = 0;
    g_fdwic_submit_pmu_scalar_segment_excluded_atomic_ticks = 0;
    g_fdwic_submit_pmu_return_ready_atomic_begin_tick = 0;
    g_fdwic_submit_pmu_total_cycles = 0;
    g_fdwic_submit_pmu_scalar_busy = 0;
    g_fdwic_submit_pmu_icache_requests = 0;
    g_fdwic_submit_pmu_icache_misses = 0;
    g_fdwic_submit_pmu_expected_submits = 0;
    g_fdwic_submit_pmu_status = 0;
    g_fdwic_submit_pmu_started = false;
    g_fdwic_submit_pmu_stopped = false;
    g_fdwic_submit_pmu_gate_running = false;
    g_fdwic_submit_pmu_gate_error = false;
    g_fdwic_submit_pmu_return_ready_atomic_active = false;
    g_fdwic_submit_pmu_return_ready_atomic_phase_armed = false;
    g_fdwic_submit_pmu_return_ready_atomic_seen = false;
    g_fdwic_submit_pmu_return_ready_atomic_time_error = false;
    g_fdwic_submit_pmu_phase.shadow_total_cycles = 0;
    g_fdwic_submit_pmu_phase.phase_total_cycles = 0;
    g_fdwic_submit_pmu_phase.shadow_scalar_busy = 0;
    g_fdwic_submit_pmu_phase.phase_scalar_busy = 0;
    g_fdwic_submit_pmu_phase.shadow_requests = 0;
    g_fdwic_submit_pmu_phase.shadow_misses = 0;
    g_fdwic_submit_pmu_phase.phase_requests = 0;
    g_fdwic_submit_pmu_phase.phase_misses = 0;
    g_fdwic_submit_pmu_phase.phase_elapsed_ticks = 0;
    g_fdwic_submit_pmu_phase.phase_begin_tick = 0;
    g_fdwic_submit_pmu_phase.phase_excluded_atomic_ticks = 0;
    g_fdwic_submit_pmu_phase.begin_reads = 0;
    g_fdwic_submit_pmu_phase.end_reads = 0;
    g_fdwic_submit_pmu_phase.status = 0;
    g_fdwic_submit_pmu_phase.armed = false;
    g_fdwic_submit_pmu_phase.boundary_error = false;
    g_fdwic_submit_pmu_phase.counter_error = false;
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
    g_fdwic_submit_pmu_excluded_kernel_calls = 0;
#endif
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
        if (!g_fdwic_submit_pmu_started || g_fdwic_submit_pmu_stopped || !g_fdwic_submit_pmu_gate_running ||
            g_fdwic_submit_pmu_phase_core == nullptr || g_fdwic_submit_pmu_reg_base == 0 || phase.armed ||
            g_fdwic_submit_pmu_return_ready_atomic_active) {
            phase.boundary_error = true;
            if (g_fdwic_submit_pmu_return_ready_atomic_active) {
                g_fdwic_submit_pmu_return_ready_atomic_time_error = true;
            }
            return;
        }
        // begin 使用 TOTAL -> scalar -> I-cache。三者均为运行中 read-clear；
        // 由外向内的顺序使随后观测到的 scalar 窗包含在 total 窗内。
        const FdwicSubmitPmuTotalShadowSnapshot total = fdwic_submit_pmu_phase_read_total_shadow();
        const uint32_t scalar = fdwic_submit_pmu_phase_read_scalar_shadow();
        const FdwicSubmitPmuIcacheShadowSnapshot icache = fdwic_submit_pmu_phase_read_shadow_counters();
        if (!fdwic_submit_pmu_add_total_shadow(total, /*include_in_phase=*/false) ||
            !fdwic_submit_pmu_add_scalar_shadow(scalar, /*include_in_phase=*/false) ||
            !fdwic_submit_pmu_add_icache_shadow(icache, /*include_in_phase=*/false)) {
            phase.boundary_error = true;
            return;
        }
        ++phase.begin_reads;
        phase.armed = true;
        phase.phase_excluded_atomic_ticks = 0;
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
        if (!g_fdwic_submit_pmu_started || g_fdwic_submit_pmu_stopped || !g_fdwic_submit_pmu_gate_running ||
            g_fdwic_submit_pmu_phase_core == nullptr || g_fdwic_submit_pmu_reg_base == 0 || !phase.armed ||
            phase_end_tick < phase.phase_begin_tick || g_fdwic_submit_pmu_return_ready_atomic_active) {
            phase.boundary_error = true;
            if (g_fdwic_submit_pmu_return_ready_atomic_active) {
                g_fdwic_submit_pmu_return_ready_atomic_time_error = true;
            }
            return;
        }
        const uint64_t raw_phase_ticks = phase_end_tick - phase.phase_begin_tick;
        if (phase.phase_excluded_atomic_ticks > raw_phase_ticks ||
            phase.phase_elapsed_ticks > UINT64_MAX - (raw_phase_ticks - phase.phase_excluded_atomic_ticks)) {
            phase.boundary_error = true;
            g_fdwic_submit_pmu_return_ready_atomic_time_error = true;
            return;
        }
        phase.phase_elapsed_ticks += raw_phase_ticks - phase.phase_excluded_atomic_ticks;
        // 终点位于 read-clear 之前，SYS 时间不包含 end 读取开销。PMU end
        // 使用 I-cache -> scalar -> TOTAL，与 begin 反序；scalar 观测窗因而
        // 包含在 total 观测窗内。边界读取及少量 bookkeeping 会进入 PMU
        // observed，不能把它解释成原业务计数的数学下界。
        const FdwicSubmitPmuIcacheShadowSnapshot icache = fdwic_submit_pmu_phase_read_shadow_counters();
        const uint32_t scalar = fdwic_submit_pmu_phase_read_scalar_shadow();
        const FdwicSubmitPmuTotalShadowSnapshot total = fdwic_submit_pmu_phase_read_total_shadow();
        if (!fdwic_submit_pmu_add_icache_shadow(icache, /*include_in_phase=*/true) ||
            !fdwic_submit_pmu_add_scalar_shadow(scalar, /*include_in_phase=*/true) ||
            !fdwic_submit_pmu_add_total_shadow(total, /*include_in_phase=*/true)) {
            phase.boundary_error = true;
            return;
        }
        ++phase.end_reads;
        phase.armed = false;
        phase.phase_begin_tick = 0;
        phase.phase_excluded_atomic_ticks = 0;
    }
#endif
}

constexpr uint32_t kFdwicSubmitPmuKernelTokenWhole = 1U << 0;
constexpr uint32_t kFdwicSubmitPmuKernelTokenPhase = 1U << 1;

// scalar 分母只累计 gate 真正运行的离散片段。调用点位于 stop 之前、start
// 之后，因此 linked Kernel 和两侧 PIPE_ALL 门控成本都不进入该 SYS_CNT 值。
PTO_DEVICE_FUNC inline bool fdwic_submit_pmu_close_scalar_segment() {
    const uint64_t segment_end = get_sys_cnt_aicore();
    if (!g_fdwic_submit_pmu_gate_running || g_fdwic_submit_pmu_scalar_segment_begin_tick == 0 ||
        segment_end < g_fdwic_submit_pmu_scalar_segment_begin_tick) {
        g_fdwic_submit_pmu_gate_error = true;
        g_fdwic_submit_pmu_scalar_segment_begin_tick = 0;
        g_fdwic_submit_pmu_scalar_segment_excluded_atomic_ticks = 0;
        return false;
    }
    const uint64_t raw_segment_ticks = segment_end - g_fdwic_submit_pmu_scalar_segment_begin_tick;
    if (g_fdwic_submit_pmu_return_ready_atomic_active ||
        g_fdwic_submit_pmu_scalar_segment_excluded_atomic_ticks > raw_segment_ticks ||
        g_fdwic_submit_pmu_scalar_elapsed_ticks >
            UINT64_MAX - (raw_segment_ticks - g_fdwic_submit_pmu_scalar_segment_excluded_atomic_ticks)) {
        g_fdwic_submit_pmu_return_ready_atomic_time_error = true;
        g_fdwic_submit_pmu_scalar_segment_begin_tick = 0;
        g_fdwic_submit_pmu_scalar_segment_excluded_atomic_ticks = 0;
        return false;
    }
    g_fdwic_submit_pmu_scalar_elapsed_ticks +=
        raw_segment_ticks - g_fdwic_submit_pmu_scalar_segment_excluded_atomic_ticks;
    g_fdwic_submit_pmu_scalar_segment_begin_tick = 0;
    g_fdwic_submit_pmu_scalar_segment_excluded_atomic_ticks = 0;
    return true;
}

// return-ready atomic 不停 PMU gate：begin 只在有效 scalar segment 内保存
// SYS_CNT，end 由 atomic wrapper 在消费返回值的数据依赖 SYS_CNT 后回填。
// 这样只从时间分母扣除本地完成等待，不引入 PIPE_ALL，也不声称跨核可见。
constexpr uint32_t kFdwicSubmitPmuReturnReadyAtomicToken = 1U;

PTO_DEVICE_FUNC inline uint32_t fdwic_submit_pmu_return_ready_atomic_begin() {
    if (!g_fdwic_submit_pmu_started || g_fdwic_submit_pmu_stopped || !g_fdwic_submit_pmu_gate_running) return 0;
    // 嵌套 wrapper 复用外层 bracket：内层拿到 token=0，end(0) 也完全
    // no-op。只有没有外层可覆盖却缺失有效 scalar segment 才算异常。
    if (g_fdwic_submit_pmu_return_ready_atomic_active) return 0;
    if (g_fdwic_submit_pmu_scalar_segment_begin_tick == 0) {
        g_fdwic_submit_pmu_return_ready_atomic_time_error = true;
        return 0;
    }
    g_fdwic_submit_pmu_return_ready_atomic_begin_tick = get_sys_cnt_aicore();
    g_fdwic_submit_pmu_return_ready_atomic_active = true;
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
    g_fdwic_submit_pmu_return_ready_atomic_phase_armed = g_fdwic_submit_pmu_phase.armed;
#else
    g_fdwic_submit_pmu_return_ready_atomic_phase_armed = false;
#endif
    return kFdwicSubmitPmuReturnReadyAtomicToken;
}

PTO_DEVICE_FUNC inline void fdwic_submit_pmu_return_ready_atomic_end(uint32_t token, uint64_t dependency_end_tick) {
    if (token == 0) return;
    if (token != kFdwicSubmitPmuReturnReadyAtomicToken || !g_fdwic_submit_pmu_started || g_fdwic_submit_pmu_stopped ||
        !g_fdwic_submit_pmu_gate_running || !g_fdwic_submit_pmu_return_ready_atomic_active ||
        dependency_end_tick < g_fdwic_submit_pmu_return_ready_atomic_begin_tick) {
        g_fdwic_submit_pmu_return_ready_atomic_time_error = true;
        g_fdwic_submit_pmu_return_ready_atomic_active = false;
        g_fdwic_submit_pmu_return_ready_atomic_begin_tick = 0;
        g_fdwic_submit_pmu_return_ready_atomic_phase_armed = false;
        return;
    }

    const uint64_t elapsed = dependency_end_tick - g_fdwic_submit_pmu_return_ready_atomic_begin_tick;
    bool valid = g_fdwic_submit_pmu_scalar_segment_excluded_atomic_ticks <= UINT64_MAX - elapsed;
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
    FdwicSubmitPmuPhaseAccumulator &phase = g_fdwic_submit_pmu_phase;
    valid = valid && phase.armed == g_fdwic_submit_pmu_return_ready_atomic_phase_armed;
    if (valid && phase.armed) {
        valid = phase.phase_excluded_atomic_ticks <= UINT64_MAX - elapsed;
    }
#endif
    if (!valid) {
        g_fdwic_submit_pmu_return_ready_atomic_time_error = true;
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
        phase.boundary_error = true;
#endif
    } else {
        g_fdwic_submit_pmu_scalar_segment_excluded_atomic_ticks += elapsed;
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
        if (phase.armed) phase.phase_excluded_atomic_ticks += elapsed;
#endif
        g_fdwic_submit_pmu_return_ready_atomic_seen = true;
    }
    g_fdwic_submit_pmu_return_ready_atomic_active = false;
    g_fdwic_submit_pmu_return_ready_atomic_begin_tick = 0;
    g_fdwic_submit_pmu_return_ready_atomic_phase_armed = false;
}

// execute_slot() 也会被背压等待和 FinalDrain 调用。窗口外返回 0，不触碰
// PMU；窗口内每个真实 linked Kernel 都先暂停 whole gate。若选中 phase 正在
// armed，则先将其闭合并把额外边界数编码进 token，供紧邻调用后的 resume 恢复。
PTO_DEVICE_FUNC inline uint32_t fdwic_submit_pmu_linked_kernel_pause() {
    if (!g_fdwic_submit_pmu_started || g_fdwic_submit_pmu_stopped) return 0;
    if (!g_fdwic_submit_pmu_gate_running) {
        // 不允许嵌套 pause 或丢失 resume 后继续操作 gate。外层若仍持有 token，
        // 保持关窗可确保 Kernel 不污染计数；本轮最终由 core status 拒绝。
        g_fdwic_submit_pmu_gate_error = true;
        return 0;
    }

    uint32_t token = kFdwicSubmitPmuKernelTokenWhole;
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
    FdwicSubmitPmuPhaseAccumulator &phase = g_fdwic_submit_pmu_phase;
    if (phase.armed) {
        const uint32_t old_end_reads = phase.end_reads;
        fdwic_submit_pmu_phase_end<kFdwicSubmitPmuCompiledPhase>();
        if (phase.boundary_error || phase.armed || phase.end_reads != old_end_reads + 1U ||
            g_fdwic_submit_pmu_excluded_kernel_calls == UINT32_MAX) {
            phase.boundary_error = true;
        } else {
            ++g_fdwic_submit_pmu_excluded_kernel_calls;
            token |= kFdwicSubmitPmuKernelTokenPhase;
        }
    }
#endif

    (void)fdwic_submit_pmu_close_scalar_segment();
    bisheng::cce::metrics_prof_stop();
    g_fdwic_submit_pmu_gate_running = false;
    return token;
}

PTO_DEVICE_FUNC inline void fdwic_submit_pmu_linked_kernel_resume(uint32_t token) {
    if (token == 0) return;
    if ((token & kFdwicSubmitPmuKernelTokenWhole) == 0 || !g_fdwic_submit_pmu_started || g_fdwic_submit_pmu_stopped ||
        g_fdwic_submit_pmu_gate_running) {
        g_fdwic_submit_pmu_gate_error = true;
        return;
    }

    bisheng::cce::metrics_prof_start();
    g_fdwic_submit_pmu_gate_running = true;
    g_fdwic_submit_pmu_scalar_segment_begin_tick = get_sys_cnt_aicore();
    g_fdwic_submit_pmu_scalar_segment_excluded_atomic_ticks = 0;

#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
    if ((token & kFdwicSubmitPmuKernelTokenPhase) != 0) {
        FdwicSubmitPmuPhaseAccumulator &phase = g_fdwic_submit_pmu_phase;
        if (phase.armed) {
            phase.boundary_error = true;
            return;
        }
        const uint32_t old_begin_reads = phase.begin_reads;
        fdwic_submit_pmu_phase_begin<kFdwicSubmitPmuCompiledPhase>();
        if (phase.boundary_error || !phase.armed || phase.begin_reads != old_begin_reads + 1U) {
            phase.boundary_error = true;
        }
    }
#else
    if ((token & kFdwicSubmitPmuKernelTokenPhase) != 0) g_fdwic_submit_pmu_gate_error = true;
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
    constexpr uint32_t kReadyMask = ((1U << 8) - 1U) | kFdwicSubmitPmuCnt0SelectorValid |
                                    kFdwicSubmitPmuCnt1SelectorValid | kFdwicSubmitPmuCnt3SelectorValid;
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 6
    // 非首个 Submit 已完成 dist_submit_begin()，在统一 begin hook 关闭
    // 上一次 tail 打开的跨 Submit 区间。
    if (task_id > 0 && static_cast<uint32_t>(task_id) < g_fdwic_submit_pmu_expected_submits) {
        fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::SubmitTransition>();
    }
#endif
    if (task_id != 0 || g_fdwic_submit_pmu_core == nullptr || g_fdwic_submit_pmu_expected_submits == 0 ||
        (g_fdwic_submit_pmu_status & kReadyMask) != kReadyMask) {
        return;
    }
    // SYS_CNT 包围的是同一业务挂点；PMU gate 的 PIPE_ALL 成本位于 PMU
    // window 外，不能把该 tick 区间与另一 ELF 的绝对时间直接相减。
    g_fdwic_submit_pmu_start_tick = get_sys_cnt_aicore();
    bisheng::cce::metrics_prof_start();
    g_fdwic_submit_pmu_gate_running = true;
    g_fdwic_submit_pmu_scalar_segment_begin_tick = get_sys_cnt_aicore();
    g_fdwic_submit_pmu_scalar_segment_excluded_atomic_ticks = 0;
    g_fdwic_submit_pmu_started = true;
    g_fdwic_submit_pmu_status |= kFdwicSubmitPmuWindowStarted;
}

PTO_DEVICE_FUNC inline void fdwic_submit_pmu_submit_end(int32_t task_id) {
    if (!g_fdwic_submit_pmu_started || g_fdwic_submit_pmu_stopped || task_id < 0) {
        return;
    }
    const uint32_t submit_ordinal = static_cast<uint32_t>(task_id) + 1U;
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID == 6
    // 每个非末次 Submit 在统一 end hook 打开 transition；末次只负责关闭
    // 整窗，避免制造一个没有后继 Submit 的悬空区间。
    if (submit_ordinal < g_fdwic_submit_pmu_expected_submits) {
        fdwic_submit_pmu_phase_begin<FdwicSubmitPmuPhase::SubmitTransition>();
        return;
    }
#endif
    if (submit_ordinal != g_fdwic_submit_pmu_expected_submits) return;
    const bool gate_was_running = g_fdwic_submit_pmu_gate_running;
    const bool scalar_segment_closed = gate_was_running && fdwic_submit_pmu_close_scalar_segment();
    if (gate_was_running) {
        bisheng::cce::metrics_prof_stop();
    } else {
        g_fdwic_submit_pmu_gate_error = true;
    }
    g_fdwic_submit_pmu_gate_running = false;
    g_fdwic_submit_pmu_end_tick = get_sys_cnt_aicore();
    fdwic_submit_pmu_read_counters();
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
    FdwicSubmitPmuPhaseAccumulator &phase = g_fdwic_submit_pmu_phase;
    if (!phase.boundary_error && !phase.armed && phase.begin_reads == phase.end_reads) {
        phase.status |= kFdwicSubmitPmuPhaseBoundaryBalanced;
    }
    const uint64_t expected_boundary_reads = fdwic_submit_pmu_expected_phase_boundary_reads(
        kFdwicSubmitPmuCompiledPhase, g_fdwic_submit_pmu_expected_submits, g_fdwic_submit_pmu_excluded_kernel_calls
    );
    const bool dynamic_calls = fdwic_submit_pmu_phase_has_dynamic_calls(kFdwicSubmitPmuCompiledPhase);
    const bool outer_reads_valid = phase.begin_reads >= g_fdwic_submit_pmu_excluded_kernel_calls &&
                                   phase.end_reads >= g_fdwic_submit_pmu_excluded_kernel_calls;
    const uint64_t outer_begin_calls =
        outer_reads_valid ? phase.begin_reads - g_fdwic_submit_pmu_excluded_kernel_calls : UINT64_MAX;
    const uint64_t outer_end_calls =
        outer_reads_valid ? phase.end_reads - g_fdwic_submit_pmu_excluded_kernel_calls : UINT64_MAX;
    const bool shape_valid =
        dynamic_calls ? outer_reads_valid && outer_begin_calls == outer_end_calls &&
                            outer_begin_calls <= fdwic_submit_pmu_dynamic_calls_max_per_core(
                                                     kFdwicSubmitPmuCompiledPhase, g_fdwic_submit_pmu_expected_submits
                                                 ) :
                        phase.begin_reads == expected_boundary_reads && phase.end_reads == expected_boundary_reads;
    if (shape_valid) {
        phase.status |= kFdwicSubmitPmuPhaseShapeValid;
    }
    const uint32_t excluded_kernel_calls = g_fdwic_submit_pmu_excluded_kernel_calls;
    const bool zero_call_dynamic = fdwic_submit_pmu_phase_has_dynamic_calls(kFdwicSubmitPmuCompiledPhase) &&
                                   phase.begin_reads == excluded_kernel_calls &&
                                   phase.end_reads == excluded_kernel_calls;
    if (phase.phase_requests <= phase.shadow_requests && phase.phase_misses <= phase.shadow_misses &&
        phase.shadow_misses <= phase.shadow_requests && phase.shadow_requests <= g_fdwic_submit_pmu_icache_requests &&
        phase.shadow_misses <= g_fdwic_submit_pmu_icache_misses) {
        phase.status |= kFdwicSubmitPmuPhaseIcacheValuesOrdered;
    }
    if (phase.phase_scalar_busy <= phase.shadow_scalar_busy &&
        phase.shadow_scalar_busy <= g_fdwic_submit_pmu_scalar_busy &&
        phase.phase_total_cycles <= phase.shadow_total_cycles && phase.phase_scalar_busy <= phase.phase_total_cycles &&
        phase.shadow_scalar_busy <= phase.shadow_total_cycles &&
        g_fdwic_submit_pmu_scalar_busy <= g_fdwic_submit_pmu_total_cycles) {
        phase.status |= kFdwicSubmitPmuPhasePmuValuesOrdered;
    }
    const bool phase_activity_valid = zero_call_dynamic ?
                                          (phase.phase_total_cycles == 0 && phase.phase_scalar_busy == 0 &&
                                           phase.phase_requests == 0 && phase.phase_misses == 0) :
                                          phase.phase_total_cycles != 0;
    if (!phase.counter_error && phase_activity_valid && phase.shadow_total_cycles == g_fdwic_submit_pmu_total_cycles &&
        phase.shadow_scalar_busy != 0 && phase.shadow_scalar_busy <= UINT32_MAX &&
        phase.shadow_requests <= UINT32_MAX && phase.shadow_misses <= UINT32_MAX &&
        g_fdwic_submit_pmu_scalar_busy < kFdwicSubmitPmuCounterRiskThreshold &&
        g_fdwic_submit_pmu_icache_requests < kFdwicSubmitPmuCounterRiskThreshold &&
        g_fdwic_submit_pmu_icache_misses < kFdwicSubmitPmuCounterRiskThreshold) {
        phase.status |= kFdwicSubmitPmuPhaseCounterReconstructionValid;
    }
    if (!g_fdwic_submit_pmu_return_ready_atomic_time_error &&
        ((zero_call_dynamic && phase.phase_elapsed_ticks == 0) ||
         (!zero_call_dynamic && phase.phase_elapsed_ticks != 0 &&
          phase.phase_elapsed_ticks <= g_fdwic_submit_pmu_scalar_elapsed_ticks))) {
        phase.status |= kFdwicSubmitPmuPhaseTimeValid;
    }
#endif
    g_fdwic_submit_pmu_stopped = true;
    g_fdwic_submit_pmu_status |= kFdwicSubmitPmuWindowStopped;
    if (!g_fdwic_submit_pmu_gate_error && gate_was_running) {
        g_fdwic_submit_pmu_status |= kFdwicSubmitPmuLinkedKernelGateBalanced;
    }
    if (!g_fdwic_submit_pmu_gate_error && !g_fdwic_submit_pmu_return_ready_atomic_time_error && scalar_segment_closed &&
        g_fdwic_submit_pmu_scalar_elapsed_ticks != 0 && g_fdwic_submit_pmu_end_tick >= g_fdwic_submit_pmu_start_tick &&
        g_fdwic_submit_pmu_scalar_elapsed_ticks <= g_fdwic_submit_pmu_end_tick - g_fdwic_submit_pmu_start_tick) {
        g_fdwic_submit_pmu_status |= kFdwicSubmitPmuScalarElapsedValid;
    }
    bool return_ready_atomic_time_closed =
        g_fdwic_submit_pmu_return_ready_atomic_seen && !g_fdwic_submit_pmu_return_ready_atomic_time_error &&
        !g_fdwic_submit_pmu_return_ready_atomic_active && g_fdwic_submit_pmu_return_ready_atomic_begin_tick == 0 &&
        g_fdwic_submit_pmu_scalar_segment_excluded_atomic_ticks == 0;
#if PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
    return_ready_atomic_time_closed =
        return_ready_atomic_time_closed && g_fdwic_submit_pmu_phase.phase_excluded_atomic_ticks == 0;
#endif
    if (return_ready_atomic_time_closed) {
        g_fdwic_submit_pmu_status |= kFdwicSubmitPmuReturnReadyAtomicTimeValid;
    }
    if (g_fdwic_submit_pmu_total_cycles != 0) {
        g_fdwic_submit_pmu_status |= kFdwicSubmitPmuTotalNonzero;
    }
}

PTO_DEVICE_FUNC inline void fdwic_submit_pmu_flush(__gm__ DistCore *self) {
    if (g_fdwic_submit_pmu_started && !g_fdwic_submit_pmu_stopped && g_fdwic_submit_pmu_gate_running) {
        // 只负责关闭遗留 gate，故意不伪造 WindowStopped。host 会因闭合失败
        // 拒绝正式 raw；FinalDrain 也不会被包装成有效 Submit 样本。
        bisheng::cce::metrics_prof_stop();
        g_fdwic_submit_pmu_gate_running = false;
    }
    __gm__ FdwicSubmitPmuCoreData *core = g_fdwic_submit_pmu_core;
    if (core == nullptr || self == nullptr) return;
    core->first_submit_start_tick = g_fdwic_submit_pmu_start_tick;
    core->last_submit_end_tick = g_fdwic_submit_pmu_end_tick;
    core->total_cycles = g_fdwic_submit_pmu_total_cycles;
    core->scalar_submit_elapsed_ticks = g_fdwic_submit_pmu_scalar_elapsed_ticks;
    core->scalar_busy = g_fdwic_submit_pmu_scalar_busy;
    core->icache_requests = g_fdwic_submit_pmu_icache_requests;
    core->icache_misses = g_fdwic_submit_pmu_icache_misses;
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
    phase_core->phase_total_cycles_observed = phase.phase_total_cycles;
    phase_core->phase_icache_requests_observed = phase.phase_requests;
    phase_core->phase_icache_misses_observed = phase.phase_misses;
    phase_core->phase_scalar_busy_observed = static_cast<uint32_t>(phase.phase_scalar_busy);
    phase_core->shadow_scalar_busy = static_cast<uint32_t>(phase.shadow_scalar_busy);
    phase_core->shadow_icache_requests = static_cast<uint32_t>(phase.shadow_requests);
    phase_core->shadow_icache_misses = static_cast<uint32_t>(phase.shadow_misses);
    phase_core->phase_id = static_cast<uint16_t>(kFdwicSubmitPmuCompiledPhase);
    phase_core->status = static_cast<uint16_t>(phase.status);
    phase_core->phase_begin_reads = phase.begin_reads;
    phase_core->phase_end_reads = phase.end_reads;
    phase_core->excluded_kernel_calls = g_fdwic_submit_pmu_excluded_kernel_calls;
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
PTO_DEVICE_FUNC inline uint32_t fdwic_submit_pmu_linked_kernel_pause() { return 0; }
PTO_DEVICE_FUNC inline void fdwic_submit_pmu_linked_kernel_resume(uint32_t) {}
PTO_DEVICE_FUNC inline uint32_t fdwic_submit_pmu_return_ready_atomic_begin() { return 0; }
PTO_DEVICE_FUNC inline void fdwic_submit_pmu_return_ready_atomic_end(uint32_t, uint64_t) {}
}  // namespace

#endif
