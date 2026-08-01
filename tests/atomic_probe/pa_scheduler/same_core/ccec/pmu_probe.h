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

#include "../common/pa_model.h"

namespace pa_scheduler::ccec_pmu {

// Empty/Scalar/ScalarDouble 在调度结束后校准门控底噪和 scalar 正向响应；
// IcacheSingle 在每核上成对累计隔离的 cold/warm 目标调用；SubmitAll 则在
// 公共调度器 hook 内覆盖本 worker 的完整 Submit 回放窗口。
enum class WindowMode : uint32_t {
    Off = 0,
    Empty = 1,
    Scalar = 2,
    ScalarDouble = 3,
    // 保留已落远端的 I-cache 校准模式值，避免 standalone host/kernel 混用旧产物时
    // 把校准请求误解释成 Submit 窗口；新增模式只在枚举尾部扩展。
    IcacheSingle = 4,
    // SubmitAll 从本 worker 的 orchestration/Submit 回放前开始，到最后一次
    // Submit 返回后停止。
    SubmitAll = 5,
};

inline bool IsSubmitWindow(WindowMode mode) {
    return mode == WindowMode::SubmitAll;
}

// PMU 控制已从 RunConfig 尾部拆到独立 PmuProbeConfig cache line。mode、
// work_amount、64 位寄存器表地址和 magic 都有具名字段，避免用数组下标
// 跨过 RunConfig 边界覆盖 winner workload。
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
constexpr uint32_t kCnt9Offset = 0x4254U;
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
constexpr uint32_t kCnt9SelectorOffset = 0x2524U;

// pmu_status 的 bits16..27 保存 get_coreid()；bits28..31 留给不参与 core id
// 解码的模式诊断。其余低位描述本条记录是否可信。
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
// I-cache 配对标志不能复用 Submit start bit；StatusCoreId 只取 12 bit，故将
// 它放在 core-id 区间之上的独立诊断位。
constexpr uint32_t kStatusIcachePairObserved = 1U << 28;
constexpr uint32_t kStatusRequired = kStatusRequested | kStatusRegMapped | kStatusCoreIdValid |
                                     kStatusCnt2Selector | kStatusCnt6Selector | kStatusCnt7Selector |
                                     kStatusCnt0Selector | kStatusCnt1Selector | kStatusCnt3Selector |
                                     kStatusCnt4Selector | kStatusCnt5Selector | kStatusCnt8Selector |
                                     kStatusWindowStarted | kStatusWindowStopped | kStatusTotalNonzero;
constexpr uint32_t kStatusCoreIdShift = 16;
constexpr uint32_t kStatusCoreIdMask = 0x0fffU;

// pmu_phase_status 独立于旧 pmu_status，避免与其中的物理 core-id 位域
// 冲突。bits4/5 只记录 shadow 是否恰好等于 primary：phase=none 没有
// 运行中 read-to-clear，host 会要求两位都成立；局部 phase 会在计数仍开启时
// 读取 shadow，A5 实测存在同周期递增与读清竞争，因此不能把“逐次严格相等”
// 作为可信记录的共同必选位。局部 phase 的方向和误差包络由 host/raw 独立校验。
constexpr uint32_t kPhaseStatusRequested = 1U << 0;
constexpr uint32_t kPhaseStatusShadowSelectors = 1U << 1;
constexpr uint32_t kPhaseStatusWindowStarted = 1U << 2;
constexpr uint32_t kPhaseStatusWindowStopped = 1U << 3;
constexpr uint32_t kPhaseStatusShadowRequestsMatch = 1U << 4;
constexpr uint32_t kPhaseStatusShadowMissesMatch = 1U << 5;
constexpr uint32_t kPhaseStatusBoundariesBalanced = 1U << 6;
constexpr uint32_t kPhaseStatusValuesOrdered = 1U << 7;
constexpr uint32_t kPhaseStatusUint32Fit = 1U << 8;
constexpr uint32_t kPhaseStatusPhaseShape = 1U << 9;
// none 必须保持 0 tick；运行阶段则必须确实累计到非零 SYS_CNT。阶段时间是否
// 不超过同核首 Submit 到末 Submit 的完整区间，由拿到两端结果的 host 再校验。
constexpr uint32_t kPhaseStatusTimeValid = 1U << 10;
constexpr uint32_t kPhaseStatusRequired =
    kPhaseStatusRequested | kPhaseStatusShadowSelectors |
    kPhaseStatusWindowStarted | kPhaseStatusWindowStopped |
    kPhaseStatusBoundariesBalanced | kPhaseStatusValuesOrdered |
    kPhaseStatusUint32Fit | kPhaseStatusPhaseShape | kPhaseStatusTimeValid;

inline const char *SubmitPmuPhaseName(SubmitPmuPhase phase) {
    switch (phase) {
    case SubmitPmuPhase::None:
        return "none";
    case SubmitPmuPhase::Claim:
        return "claim";
    case SubmitPmuPhase::EfDrain:
        return "efdrain";
    case SubmitPmuPhase::Materialize:
        return "materialize";
    case SubmitPmuPhase::Register:
        return "register";
    case SubmitPmuPhase::Count:
        break;
    }
    return "invalid";
}

inline uint32_t StatusCoreId(uint32_t status) {
    return (status >> kStatusCoreIdShift) & kStatusCoreIdMask;
}

}  // namespace pa_scheduler::ccec_pmu

#endif  // PA_SCHEDULER_CCEC_PMU_PROBE_H
