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

// submit-pmu-none 是真实 A5 PA 的独立诊断 ABI。它不复用普通泳道的逐事件
// record，也不复用通用 PMU 的逐 kernel task ring；每个 worker 只在完整
// Submit 窗口结束后发布一个固定 64B 结果。
constexpr uint32_t kFdwicSubmitPmuMagic = 0x554d5053U;  // little-endian "SPMU"
constexpr uint16_t kFdwicSubmitPmuVersion = 1;
constexpr uint16_t kFdwicSubmitPmuModeNone = 1;
constexpr uint32_t kFdwicSubmitPmuExpectedAic = 32;
constexpr uint32_t kFdwicSubmitPmuExpectedAiv = 64;
constexpr uint32_t kFdwicSubmitPmuExpectedCores = kFdwicSubmitPmuExpectedAic + kFdwicSubmitPmuExpectedAiv;
constexpr uint32_t kFdwicSubmitPmuPhysicalSubcores = 108;
constexpr uint32_t kFdwicSubmitPmuBitmapWords = 4;
constexpr uint32_t kFdwicSubmitPmuCounterRiskThreshold = 0x3fffffffU;

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
