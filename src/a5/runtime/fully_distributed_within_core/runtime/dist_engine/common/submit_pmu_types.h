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

#include <cstddef>
#include <cstdint>

#include "data_type.h"

// 真实 A5 PA 的 submit-PMU 不复用普通泳道的逐事件 record，也不复用通用
// PMU 的逐 kernel task ring。none 每核只发布一个 64B 整窗结果；局部阶段
// 在相同整窗结果之后追加一个 64B sidecar，仍然没有逐事件记录。
constexpr uint32_t kFdwicSubmitPmuMagic = 0x554d5053U;  // little-endian "SPMU"
constexpr uint16_t kFdwicSubmitPmuVersion = 1;
constexpr uint16_t kFdwicSubmitPmuModeNone = 1;
constexpr uint16_t kFdwicSubmitPmuModeArgBuild = 2;
constexpr uint16_t kFdwicSubmitPmuModeEmptyBracket = 3;
constexpr uint16_t kFdwicSubmitPmuModeMaterialize = 4;
constexpr uint16_t kFdwicSubmitPmuModeClaim = 5;
constexpr uint16_t kFdwicSubmitPmuModeRegister = 6;
constexpr uint16_t kFdwicSubmitPmuModeSubmitTransition = 7;
constexpr uint16_t kFdwicSubmitPmuModeEfDrainControl = 8;
constexpr uint16_t kFdwicSubmitPmuModePrepareMap = 9;
constexpr uint16_t kFdwicSubmitPmuModeFanin = 10;
constexpr uint32_t kFdwicSubmitPmuExpectedAic = 32;
constexpr uint32_t kFdwicSubmitPmuExpectedAiv = 64;
constexpr uint32_t kFdwicSubmitPmuExpectedCores = kFdwicSubmitPmuExpectedAic + kFdwicSubmitPmuExpectedAiv;
constexpr uint32_t kFdwicSubmitPmuPhysicalSubcores = 108;
constexpr uint32_t kFdwicSubmitPmuBitmapWords = 4;
constexpr uint32_t kFdwicSubmitPmuCounterRiskThreshold = 0x3fffffffU;

enum class FdwicSubmitPmuPhase : uint16_t {
    None = 0,
    // compete-first Claim 完成到匹配 Finish 的 Materialize 入口；包含同步
    // eager callback 构参、Begin 返回和 Finish 重入。
    ArgBuild = 1,
    // Claim.end 同一调用点的紧邻 begin/end；只提供 running bracket 自身
    // 引入的空区间经验观察指纹，不代表任何业务 phase 或数学最小值。
    EmptyBracket = 2,
    // 当前泳道 Materialize.begin 到 Materialize.end：task-cap 检查与
    // dist_submit_materialize_args 主体；每个 Submit 固定调用一次。
    Materialize = 3,
    // 当前泳道 Claim.begin 到 Claim.end：compete-first 还包含 Claim 前的
    // task-cap 检查；每个 Submit 固定调用一次。
    Claim = 4,
    // dist_submit_register_outputs() 调用入口到返回。刻意排除前一阶段
    // record 发布和 caller 衔接；每个 Submit 固定调用一次。
    Register = 5,
    // 上一次 Submit 返回前到下一次 dist_submit_begin() 完成。首个 Submit
    // 没有前驱、末个 Submit 没有后继，因此每核固定 expected_submits - 1 次。
    SubmitTransition = 6,
    // 每次 Submit 开头的 EfDrain 控制段。execute_slot() 中的真实 linked-kernel
    // 调用通过成对 pause/resume 排除，barrier、完成发布和 frontier 等 scalar
    // 后处理仍保留在控制段内。
    EfDrainControl = 7,
    // dist_submit_prepare_map() 调用入口到返回；与泳道 PrepareMap 的业务
    // 主体边界一致，每个 Submit 固定调用一次。
    PrepareMap = 8,
    // Kernel winner 的 dist_submit_collect_fanin() 调用体。winner 分布由
    // 多核竞争决定，因此只要求逐核边界平衡与全局调用数闭合。
    Fanin = 9,
    Count = 10,
};

constexpr uint16_t fdwic_submit_pmu_mode_for_phase(FdwicSubmitPmuPhase phase) {
    switch (phase) {
    case FdwicSubmitPmuPhase::None:
        return kFdwicSubmitPmuModeNone;
    case FdwicSubmitPmuPhase::ArgBuild:
        return kFdwicSubmitPmuModeArgBuild;
    case FdwicSubmitPmuPhase::EmptyBracket:
        return kFdwicSubmitPmuModeEmptyBracket;
    case FdwicSubmitPmuPhase::Materialize:
        return kFdwicSubmitPmuModeMaterialize;
    case FdwicSubmitPmuPhase::Claim:
        return kFdwicSubmitPmuModeClaim;
    case FdwicSubmitPmuPhase::Register:
        return kFdwicSubmitPmuModeRegister;
    case FdwicSubmitPmuPhase::SubmitTransition:
        return kFdwicSubmitPmuModeSubmitTransition;
    case FdwicSubmitPmuPhase::EfDrainControl:
        return kFdwicSubmitPmuModeEfDrainControl;
    case FdwicSubmitPmuPhase::PrepareMap:
        return kFdwicSubmitPmuModePrepareMap;
    case FdwicSubmitPmuPhase::Fanin:
        return kFdwicSubmitPmuModeFanin;
    case FdwicSubmitPmuPhase::Count:
        break;
    }
    return 0;
}

PTO_DEVICE_FUNC constexpr bool fdwic_submit_pmu_phase_has_dynamic_calls(FdwicSubmitPmuPhase phase) {
    return phase == FdwicSubmitPmuPhase::Fanin;
}

// 当前 PA 每个 batch 固定提交四个 Kernel task 和一个 Alloc task。动态
// winner/loser phase 不伪造“每核固定次数”，只在 96 核聚合后按业务形状闭合。
constexpr uint32_t fdwic_submit_pmu_batch_count(uint32_t expected_submits) {
    return expected_submits != 0 && expected_submits % 5U == 0 ? expected_submits / 5U : 0U;
}

constexpr uint64_t fdwic_submit_pmu_expected_dynamic_calls_all(
    FdwicSubmitPmuPhase phase, uint32_t expected_submits
) {
    const uint64_t batches = fdwic_submit_pmu_batch_count(expected_submits);
    return phase == FdwicSubmitPmuPhase::Fanin ? 4U * batches : 0U;
}

constexpr uint64_t fdwic_submit_pmu_expected_dynamic_calls_aic(
    FdwicSubmitPmuPhase phase, uint32_t expected_submits
) {
    const uint64_t batches = fdwic_submit_pmu_batch_count(expected_submits);
    return phase == FdwicSubmitPmuPhase::Fanin ? 2U * batches : 0U;
}

constexpr uint64_t fdwic_submit_pmu_expected_dynamic_calls_aiv(
    FdwicSubmitPmuPhase phase, uint32_t expected_submits
) {
    const uint64_t batches = fdwic_submit_pmu_batch_count(expected_submits);
    return phase == FdwicSubmitPmuPhase::Fanin ? 2U * batches : 0U;
}

PTO_DEVICE_FUNC constexpr uint32_t fdwic_submit_pmu_dynamic_calls_max_per_core(
    FdwicSubmitPmuPhase phase, uint32_t expected_submits
) {
    const uint32_t batches =
        expected_submits != 0 && expected_submits % 5U == 0 ? expected_submits / 5U : 0U;
    return phase == FdwicSubmitPmuPhase::Fanin ? 2U * batches : 0U;
}

PTO_DEVICE_FUNC constexpr uint32_t
fdwic_submit_pmu_expected_phase_calls(FdwicSubmitPmuPhase phase, uint32_t expected_submits) {
    if (phase == FdwicSubmitPmuPhase::None) return 0;
    if (fdwic_submit_pmu_phase_has_dynamic_calls(phase)) return 0;
    if (phase == FdwicSubmitPmuPhase::SubmitTransition) {
        return expected_submits == 0 ? 0 : expected_submits - 1U;
    }
    return expected_submits;
}

PTO_DEVICE_FUNC constexpr uint64_t fdwic_submit_pmu_expected_phase_boundary_reads(
    FdwicSubmitPmuPhase phase, uint32_t expected_submits, uint32_t excluded_kernel_calls
) {
    const uint64_t outer_calls = fdwic_submit_pmu_expected_phase_calls(phase, expected_submits);
    return outer_calls + (phase == FdwicSubmitPmuPhase::EfDrainControl ? excluded_kernel_calls : 0U);
}

constexpr bool fdwic_submit_pmu_mode_has_phase(uint16_t mode) {
    return mode == kFdwicSubmitPmuModeArgBuild || mode == kFdwicSubmitPmuModeEmptyBracket ||
           mode == kFdwicSubmitPmuModeMaterialize || mode == kFdwicSubmitPmuModeClaim ||
           mode == kFdwicSubmitPmuModeRegister || mode == kFdwicSubmitPmuModeSubmitTransition ||
           mode == kFdwicSubmitPmuModeEfDrainControl || mode == kFdwicSubmitPmuModePrepareMap ||
           mode == kFdwicSubmitPmuModeFanin;
}

// A5 PIPE_UTIL 事件布局。CNT6/CNT7 是权威值；CNT8/CNT5 使用相同事件作
// 同窗副本。none 构建没有中途 read-clear，故两组必须逐核精确相等。
constexpr uint32_t kFdwicSubmitPmuCnt2ScalarBusy = 0x001U;
constexpr uint32_t kFdwicSubmitPmuCnt5ShadowIcacheMiss = 0x035U;
constexpr uint32_t kFdwicSubmitPmuCnt6IcacheRequest = 0x034U;
constexpr uint32_t kFdwicSubmitPmuCnt7IcacheMiss = 0x035U;
constexpr uint32_t kFdwicSubmitPmuCnt8ShadowIcacheRequest = 0x034U;

enum FdwicSubmitPmuCoreStatus : uint32_t {
    kFdwicSubmitPmuRequested = 1U << 0,
    kFdwicSubmitPmuRegMapped = 1U << 1,
    kFdwicSubmitPmuPhysicalIdValid = 1U << 2,
    kFdwicSubmitPmuCnt2SelectorValid = 1U << 3,
    kFdwicSubmitPmuCnt5SelectorValid = 1U << 4,
    kFdwicSubmitPmuCnt6SelectorValid = 1U << 5,
    kFdwicSubmitPmuCnt7SelectorValid = 1U << 6,
    kFdwicSubmitPmuCnt8SelectorValid = 1U << 7,
    kFdwicSubmitPmuWindowStarted = 1U << 8,
    kFdwicSubmitPmuWindowStopped = 1U << 9,
    kFdwicSubmitPmuTotalNonzero = 1U << 10,
};
constexpr uint32_t kFdwicSubmitPmuRequiredCoreStatus = (1U << 11) - 1U;

enum FdwicSubmitPmuPhaseStatus : uint32_t {
    kFdwicSubmitPmuPhaseRequested = 1U << 0,
    kFdwicSubmitPmuPhaseBoundaryBalanced = 1U << 1,
    kFdwicSubmitPmuPhaseShapeValid = 1U << 2,
    kFdwicSubmitPmuPhaseValuesOrdered = 1U << 3,
    kFdwicSubmitPmuPhaseTimeValid = 1U << 4,
    kFdwicSubmitPmuPhaseTailRead = 1U << 5,
};
constexpr uint32_t kFdwicSubmitPmuRequiredPhaseStatus = (1U << 6) - 1U;

enum FdwicSubmitPmuOwnerStatus : uint32_t {
    kFdwicSubmitPmuOwnerRequested = 1U << 0,
    kFdwicSubmitPmuOwnerTopologyValid = 1U << 1,
    kFdwicSubmitPmuOwnerConfigured = 1U << 2,
    kFdwicSubmitPmuOwnerConfigReadbackValid = 1U << 3,
    kFdwicSubmitPmuOwnerRestoreAttempted = 1U << 4,
    kFdwicSubmitPmuOwnerRestored = 1U << 5,
    kFdwicSubmitPmuOwnerAborted = 1U << 31,
};
constexpr uint32_t kFdwicSubmitPmuRequiredOwnerStatus =
    kFdwicSubmitPmuOwnerRequested | kFdwicSubmitPmuOwnerTopologyValid | kFdwicSubmitPmuOwnerConfigured |
    kFdwicSubmitPmuOwnerConfigReadbackValid | kFdwicSubmitPmuOwnerRestoreAttempted | kFdwicSubmitPmuOwnerRestored;

// 失败字段只用于 owner 冷路径诊断；正式 raw 只有全部字段闭合后才发布。
enum class FdwicSubmitPmuOwnerField : uint32_t {
    None = 0,
    State,
    Topology,
    RegisterBase,
    Ctrl0,
    Ctrl1,
    Selector0,
    StartLow = 16,
    StartHigh,
    StopLow,
    StopHigh,
};

struct FdwicSubmitPmuCoreData {
    uint64_t first_submit_start_tick;
    uint64_t last_submit_end_tick;
    uint64_t total_cycles;
    uint32_t scalar_busy;
    uint32_t icache_requests;
    uint32_t icache_misses;
    uint32_t shadow_icache_requests;
    uint32_t shadow_icache_misses;
    uint32_t submit_count;
    uint32_t expected_submit_count;
    uint16_t logical_core_id;
    uint16_t physical_core_id;
    uint16_t block_id;
    uint16_t lane;
    uint32_t status;
} __attribute__((aligned(64)));

static_assert(sizeof(FdwicSubmitPmuCoreData) == 64, "submit-PMU core record must occupy one cacheline");
static_assert(offsetof(FdwicSubmitPmuCoreData, status) == 60, "submit-PMU status offset changed");

// phase sidecar 只在局部阶段 mode 中分配。每个 worker 独占一条 cacheline，
// 避免相邻 worker 发布结果时产生伪共享。CNT6/7 的整窗 primary 仍保存在
// FdwicSubmitPmuCoreData；这里的 request/miss 是运行中 read-clear 观测值。
// begin/end 两侧的少量 bookkeeping 取指也会进入该样本，因此它不是原始
// 业务区间事件数的数学下界。
struct FdwicSubmitPmuPhaseCoreData {
    uint64_t phase_elapsed_ticks;
    uint64_t phase_icache_requests_observed;
    uint64_t phase_icache_misses_observed;
    uint32_t phase_id;
    uint32_t phase_begin_reads;
    uint32_t phase_end_reads;
    uint32_t max_shadow_request_chunk;
    uint32_t max_shadow_miss_chunk;
    uint32_t status;
    // EfDrainControl 专属：reserved[0] 保存被 pause/resume 排除的 linked-kernel
    // 调用数；其他 phase 以及 reserved[1..3] 必须保持 0。复用保留字可维持
    // 每核 64B 与总 GM 容量不变。
    uint32_t reserved[4];
} __attribute__((aligned(64)));

static_assert(sizeof(FdwicSubmitPmuPhaseCoreData) == 64, "submit-PMU phase record must occupy one cacheline");
static_assert(offsetof(FdwicSubmitPmuPhaseCoreData, status) == 44, "submit-PMU phase status offset changed");

struct FdwicSubmitPmuPhaseAccumulator {
    uint64_t shadow_requests;
    uint64_t shadow_misses;
    uint64_t phase_requests;
    uint64_t phase_misses;
    uint64_t phase_elapsed_ticks;
    uint64_t phase_begin_tick;
    uint32_t begin_reads;
    uint32_t end_reads;
    uint32_t max_shadow_request_chunk;
    uint32_t max_shadow_miss_chunk;
    uint32_t status;
    bool armed;
    bool boundary_error;
};

struct FdwicSubmitPmuHeader {
    // Host 初始化的只读配置 cacheline。
    uint32_t magic;
    uint16_t version;
    uint16_t mode;
    uint32_t header_bytes;
    uint32_t record_bytes;
    uint32_t num_cores;
    uint32_t expected_aic;
    uint32_t expected_aiv;
    uint64_t sys_cnt_freq_hz;
    uint32_t selectors[5];
    uint32_t reserved0;

    // AICPU owner 独占写入的状态 cacheline。
    volatile uint32_t owner_status;
    volatile uint32_t configured_count;
    volatile uint32_t restored_count;
    volatile uint32_t configured_aic;
    volatile uint32_t configured_aiv;
    volatile uint32_t complete_mixed_triplets;
    volatile uint32_t restore_failures;
    volatile uint32_t active_after_restore;
    volatile uint32_t configured_bitmap_words[kFdwicSubmitPmuBitmapWords];
    volatile uint32_t first_failure_core;
    volatile uint32_t first_failure_field;
    volatile uint32_t first_failure_observed;
    volatile uint32_t first_failure_expected;

    FdwicSubmitPmuCoreData cores[kFdwicSubmitPmuExpectedCores];
} __attribute__((aligned(64)));

static_assert(offsetof(FdwicSubmitPmuHeader, owner_status) == 64, "owner state must occupy cacheline two");
static_assert(offsetof(FdwicSubmitPmuHeader, cores) == 128, "per-core records must start after two cachelines");
static_assert(sizeof(FdwicSubmitPmuHeader) == 128 + 64 * kFdwicSubmitPmuExpectedCores, "submit-PMU ABI size changed");

constexpr size_t kFdwicSubmitPmuNoneBytes = sizeof(FdwicSubmitPmuHeader);
constexpr size_t kFdwicSubmitPmuPhaseBytes =
    sizeof(FdwicSubmitPmuHeader) + sizeof(FdwicSubmitPmuPhaseCoreData) * kFdwicSubmitPmuExpectedCores;

constexpr size_t fdwic_submit_pmu_bytes_for_mode(uint16_t mode) {
    return fdwic_submit_pmu_mode_has_phase(mode) ? kFdwicSubmitPmuPhaseBytes : kFdwicSubmitPmuNoneBytes;
}
