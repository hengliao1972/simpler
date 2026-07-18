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

#ifndef PA_SCHEDULER_CCEC_PMU_PROBE_H
#define PA_SCHEDULER_CCEC_PMU_PROBE_H

#include <stdint.h>

namespace pa_scheduler::ccec_pmu {

// Empty/Scalar/ScalarDouble 在调度结束后校准门控底噪和 scalar 正向响应；
// SubmitAll 则在公共调度器 hook 内覆盖本 worker 的完整 Submit 回放窗口。
enum class WindowMode : uint32_t {
    Off = 0,
    Empty = 1,
    Scalar = 2,
    ScalarDouble = 3,
    // SubmitAll 从本 worker 的 orchestration/Submit 回放前开始，到最后一次
    // Submit 返回后停止。
    SubmitAll = 4,
};

inline bool IsSubmitWindow(WindowMode mode) {
    return mode == WindowMode::SubmitAll;
}

// RunConfig::reserved 保持既有 64B ABI；CCEC 独占解释以下五个槽位，其他后端仍看到全零。
constexpr uint32_t kConfigMode = 0;
constexpr uint32_t kConfigScalarNops = 1;
constexpr uint32_t kConfigRegTableLow = 2;
constexpr uint32_t kConfigRegTableHigh = 3;
constexpr uint32_t kConfigMagic = 4;
constexpr uint32_t kConfigMagicValue = 0x504d5531U;  // "PMU1"

// DAV_3510 有 36 个物理 AICore，每个 AICore 展开为 1 AIC + 2 AIV，共 108 个物理子核编号。
constexpr uint32_t kPhysicalAicoreCount = 36;
constexpr uint32_t kPhysicalSubcoreCount = 108;
constexpr uint32_t kAicorePerDie = 18;
constexpr uint32_t kSubcoresPerDie = 54;
constexpr uint64_t kAivFirstOffset = 0x100000ULL;
constexpr uint64_t kAivSecondOffset = 0x200000ULL;
constexpr uint32_t kAicoreMapBytes = 0x300000U;

// PIPE_UTILIZATION 事件由 standalone Main AICPU owner 配置，kernel 逐核读回并核对 selector。
constexpr uint32_t kScalarBusyEvent = 0x1U;
constexpr uint32_t kIcacheRequestEvent = 0x34U;
constexpr uint32_t kIcacheMissEvent = 0x35U;
constexpr uint32_t kVectorBusyEvent = 0x501U;
constexpr uint32_t kCubeBusyEvent = 0x301U;
constexpr uint32_t kMte1BusyEvent = 0x701U;
constexpr uint32_t kMte2BusyEvent = 0x202U;
constexpr uint32_t kMte3BusyEvent = 0x203U;
constexpr uint32_t kFixBusyEvent = 0x714U;

// DAV_3510 PMU MMIO offset。ld_dev 的立即数只有 12 bit，因此 kernel 会分别重基址到 0x2400/0x4200。
constexpr uint32_t kSelectorBlockOffset = 0x2400U;
constexpr uint32_t kCounterBlockOffset = 0x4200U;
constexpr uint32_t kCnt2Offset = 0x4220U;
constexpr uint32_t kCnt0Offset = 0x4210U;
constexpr uint32_t kCnt1Offset = 0x4218U;
constexpr uint32_t kCnt3Offset = 0x4228U;
constexpr uint32_t kCnt4Offset = 0x4230U;
constexpr uint32_t kCnt5Offset = 0x4238U;
constexpr uint32_t kCnt6Offset = 0x4240U;
constexpr uint32_t kCnt7Offset = 0x4248U;
constexpr uint32_t kCnt8Offset = 0x4250U;
constexpr uint32_t kTotalLowOffset = 0x4260U;
constexpr uint32_t kTotalHighOffset = 0x4264U;
constexpr uint32_t kCnt2SelectorOffset = 0x2508U;
constexpr uint32_t kCnt0SelectorOffset = 0x2500U;
constexpr uint32_t kCnt1SelectorOffset = 0x2504U;
constexpr uint32_t kCnt3SelectorOffset = 0x250cU;
constexpr uint32_t kCnt4SelectorOffset = 0x2510U;
constexpr uint32_t kCnt5SelectorOffset = 0x2514U;
constexpr uint32_t kCnt6SelectorOffset = 0x2518U;
constexpr uint32_t kCnt7SelectorOffset = 0x251cU;
constexpr uint32_t kCnt8SelectorOffset = 0x2520U;

// pmu_status 的低位描述本条记录是否可信，高 16 bit 保存 get_coreid()，便于 host 检查 96 核唯一性。
constexpr uint32_t kStatusRequested = 1U << 0;
constexpr uint32_t kStatusRegMapped = 1U << 1;
constexpr uint32_t kStatusCoreIdValid = 1U << 2;
constexpr uint32_t kStatusCnt2Selector = 1U << 3;
constexpr uint32_t kStatusCnt6Selector = 1U << 4;
constexpr uint32_t kStatusCnt7Selector = 1U << 5;
constexpr uint32_t kStatusWindowStarted = 1U << 6;
constexpr uint32_t kStatusTotalNonzero = 1U << 7;
constexpr uint32_t kStatusPriorSnapshotLarger = 1U << 8;
constexpr uint32_t kStatusCnt0Selector = 1U << 9;
constexpr uint32_t kStatusCnt1Selector = 1U << 10;
constexpr uint32_t kStatusCnt3Selector = 1U << 11;
constexpr uint32_t kStatusCnt4Selector = 1U << 12;
constexpr uint32_t kStatusCnt5Selector = 1U << 13;
constexpr uint32_t kStatusCnt8Selector = 1U << 14;
constexpr uint32_t kStatusWindowStopped = 1U << 15;
constexpr uint32_t kStatusRequired = kStatusRequested | kStatusRegMapped | kStatusCoreIdValid |
                                     kStatusCnt2Selector | kStatusCnt6Selector | kStatusCnt7Selector |
                                     kStatusCnt0Selector | kStatusCnt1Selector | kStatusCnt3Selector |
                                     kStatusCnt4Selector | kStatusCnt5Selector | kStatusCnt8Selector |
                                     kStatusWindowStarted | kStatusWindowStopped | kStatusTotalNonzero;
constexpr uint32_t kStatusCoreIdShift = 16;
constexpr uint32_t kStatusCoreIdMask = 0x0fffU;

inline uint64_t PackPointer(const uint32_t *words) {
    return static_cast<uint64_t>(words[kConfigRegTableLow]) |
           (static_cast<uint64_t>(words[kConfigRegTableHigh]) << 32);
}

inline void StorePointer(uint32_t *words, const void *pointer) {
    const uint64_t raw = reinterpret_cast<uint64_t>(pointer);
    words[kConfigRegTableLow] = static_cast<uint32_t>(raw);
    words[kConfigRegTableHigh] = static_cast<uint32_t>(raw >> 32);
}

inline uint32_t StatusCoreId(uint32_t status) {
    return (status >> kStatusCoreIdShift) & kStatusCoreIdMask;
}

}  // namespace pa_scheduler::ccec_pmu

#endif  // PA_SCHEDULER_CCEC_PMU_PROBE_H
