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

#ifndef PA_SCHEDULER_CCEC_PMU_OWNER_CONTROL_H_
#define PA_SCHEDULER_CCEC_PMU_OWNER_CONTROL_H_

#include <cstddef>
#include <cstdint>

namespace pa_scheduler::pmu_owner {

// 这份头文件同时供 x86 host 与 AArch64 AICPU helper 使用。所有跨端字段都采用
// 固定宽度整数；禁止在 ABI 中放 host 指针、bool、STL 容器或编译器相关位域。
constexpr uint32_t kPmuOwnerControlMagic = 0x504d554fU;  // "PMUO"
constexpr uint32_t kPmuOwnerControlVersion = 1U;

// DAV_3510 一共有 2 die；每个 die 依次放 18 个 AIC 和 36 个 AIV 物理槽。
// 当前 A5 stream 实际开放 32 个 AIC 与 64 个 AIV，其余 12 个槽的 MMIO
// 读回不会匹配配置值，因此 owner 必须扫描 108 槽，最终取得 96 个可用槽。
constexpr uint32_t kPhysicalSubcoreCount = 108U;
constexpr uint32_t kExpectedSubcoreCount = 96U;
constexpr uint32_t kExpectedAicCount = 32U;
constexpr uint32_t kExpectedAivCount = 64U;
constexpr uint32_t kAicPerDie = 18U;
constexpr uint32_t kSubcoresPerDie = 54U;
constexpr uint32_t kConfiguredBitmapWords = 4U;
constexpr uint32_t kDiagnosticIndexUnset = 0xffffffffU;

static_assert(kExpectedAicCount + kExpectedAivCount == kExpectedSubcoreCount, "active topology count mismatch");
static_assert(kAicPerDie * 2U == 36U, "physical AIC topology changed");
static_assert(kSubcoresPerDie * 2U == kPhysicalSubcoreCount, "physical subcore topology changed");

// A5 PIPE_UTILIZATION 的正式 counter 槽位布局。submit-pmu 用 CNT8/CNT5
// 重复配置 I-cache request/miss，作为允许中途 read-to-clear 的 shadow；
// CNT6/7 始终不在阶段边界读取，保留为完整 Submit 的权威对照。
//
// 不能把 miss 放进 CNT9：A5 b1 实测表明 CNT9 selector 虽能回读 0x35，
// 但计数恒为 0；正式 PIPE_UTIL 表也把 CNT9 标成 unused。0x35 在独立
// I-cache 微基准的低位 counter 已验证可计数，因此诊断构建让 CNT5 承担
// shadow miss，并明确放弃该构建中的 MTE3 busy。
constexpr uint32_t kPmuCounterCount = 10U;
constexpr uint32_t kConfiguredSelectors[kPmuCounterCount] = {
    0x501U,  // CNT0: vector busy
    0x301U,  // CNT1: cube busy
    0x001U,  // CNT2: scalar busy
    0x701U,  // CNT3: MTE1 busy
    0x202U,  // CNT4: MTE2 busy
#if PA_BUILD_SUBMIT_PMU
    0x035U,  // CNT5: shadow I-cache miss
    0x034U,  // CNT6: I-cache request（完整 Submit）
    0x035U,  // CNT7: I-cache miss（完整 Submit）
    0x034U,  // CNT8: shadow I-cache request
    0x000U,  // CNT9: A5 PIPE_UTIL 正式未使用
#else
    0x203U,  // CNT5: MTE3 busy
    0x034U,  // CNT6: I-cache request
    0x035U,  // CNT7: I-cache miss
    0x714U,  // CNT8: fix-pipe busy
    0x000U,  // CNT9: 未使用
#endif
};

constexpr int32_t kStatusPending = 0x7fffffff;

// AICPU entry 始终向 runtime 返回 0；协议结果只通过 control.status 回传，
// 从而避免一次可诊断的配置失败被 runtime 升格成整条 stream 异常。
enum class PmuOwnerStatus : int32_t {
    Success = 0,
    InvalidArguments = -1,
    InvalidControl = -2,
    UnexpectedTopology = -3,
    AlreadyConfigured = -4,
    ConfigureCountMismatch = -5,
    ConfigureRollbackFailed = -6,
    ConfigureSlotRestoreFailed = -7,
    RestoreFailed = -8,
};

// 首个异常寄存器使用稳定的枚举编号，host 不需要解析 AICPU 日志即可定位
// 是基址、selector、计数范围还是 enable 控制读回不一致。
enum class PmuOwnerField : uint32_t {
    None = 0,
    Arguments,
    ControlMagic,
    ControlVersion,
    ControlSize,
    State,
    RegisterBase,
    Ctrl0,
    Ctrl1,
    Selector0,
    Selector1,
    Selector2,
    Selector3,
    Selector4,
    Selector5,
    Selector6,
    Selector7,
    Selector8,
    Selector9,
    StartCycleLow,
    StartCycleHigh,
    StopCycleLow,
    StopCycleHigh,
    BitmapCount,
    TotalCount,
    AicCount,
    AivCount,
};

// 单个物理子核被 owner 改动的完整可恢复状态恰好占一条 cache line。
// PMU counter 是 read-to-clear，旧 counter 值无法恢复；owner 会话必须独占。
struct alignas(64) PmuSavedRegisters {
    uint32_t ctrl0;
    uint32_t ctrl1;
    uint32_t selectors[kPmuCounterCount];
    uint32_t start_cycle_low;
    uint32_t start_cycle_high;
    uint32_t stop_cycle_low;
    uint32_t stop_cycle_high;
};

// Host 与 AICPU 共享的 owner 状态。前 128B 是命令结果和诊断，随后内嵌
// 108 个 MMIO 基址、4-word 所有权 bitmap，以及每槽 64B 的 Configure 快照。
// bitmap 的严格语义是“原值已保存、且 owner 可能已经改写 MMIO、但尚未
// 完整恢复”的槽；它在 Configure 写第一项 MMIO 前置位，仅在恢复读回完整
// 一致后清位。Restore 期间不得清零或重建 saved[]，只能按 bitmap 逆序消费。
struct alignas(64) PmuOwnerControl {
    uint32_t magic;
    uint32_t version;
    uint32_t struct_bytes;
    volatile int32_t status;

    uint32_t configured;
    uint32_t expected_total;
    uint32_t expected_aic;
    uint32_t expected_aiv;

    // active_* 与 bitmap 表示仍由本 owner 持有、尚未恢复的物理槽。
    uint32_t active_total;
    uint32_t active_aic;
    uint32_t active_aiv;
    // discovered_* 保留本次 Configure 扫描结果；即使计数不匹配后回滚，
    // host 仍能看到回滚前究竟探测到了多少 AIC/AIV。
    uint32_t discovered_total;
    uint32_t discovered_aic;
    uint32_t discovered_aiv;
    uint32_t skipped_total;

    uint32_t first_failed_index;
    uint32_t first_failed_field;
    uint32_t first_failed_observed;
    uint32_t first_failed_expected;

    uint32_t restore_failures;
    uint32_t first_restore_failed_index;
    uint32_t first_restore_failed_field;
    uint32_t first_restore_failed_observed;
    uint32_t first_restore_failed_expected;
    uint32_t reserved_header[8];

    uint64_t register_bases[kPhysicalSubcoreCount];
    // 字段名保留 configured_bitmap 以稳定 host/device ABI；失败路径中它还会
    // 临时包含“配置未通过但恢复仍待重试”的 owned 槽。
    uint32_t configured_bitmap[kConfiguredBitmapWords];
    // 让 saved[] 从新的 64B cache line 开始；该 padding 不承载协议含义。
    uint32_t reserved_bitmap[4];
    PmuSavedRegisters saved[kPhysicalSubcoreCount];
};

static_assert(sizeof(PmuSavedRegisters) == 64U, "one saved PMU slot must occupy one cache line");
static_assert(alignof(PmuSavedRegisters) == 64U, "saved PMU slot alignment changed");
static_assert(offsetof(PmuOwnerControl, status) == 12U, "PMU owner status offset changed");
static_assert(offsetof(PmuOwnerControl, register_bases) == 128U, "PMU owner header must occupy two cache lines");
static_assert(offsetof(PmuOwnerControl, configured_bitmap) == 992U, "PMU owner bitmap offset changed");
static_assert(offsetof(PmuOwnerControl, saved) == 1024U, "PMU owner saved area must be cache-line aligned");
static_assert(sizeof(PmuOwnerControl) == 7936U, "PMU owner control ABI changed");
static_assert(sizeof(PmuOwnerControl) % 64U == 0U, "PMU owner control must use complete cache lines");
static_assert(alignof(PmuOwnerControl) == 64U, "PMU owner control alignment changed");

inline bool IsAicPhysicalSlot(uint32_t index)
{
    return index < kPhysicalSubcoreCount && (index % kSubcoresPerDie) < kAicPerDie;
}

inline bool IsConfigured(const PmuOwnerControl &control, uint32_t index)
{
    return index < kPhysicalSubcoreCount &&
        (control.configured_bitmap[index / 32U] & (1U << (index % 32U))) != 0U;
}

inline void SetConfigured(PmuOwnerControl *control, uint32_t index)
{
    if (index < kPhysicalSubcoreCount) {
        control->configured_bitmap[index / 32U] |= 1U << (index % 32U);
    }
}

inline void ClearConfigured(PmuOwnerControl *control, uint32_t index)
{
    if (index < kPhysicalSubcoreCount) {
        control->configured_bitmap[index / 32U] &= ~(1U << (index % 32U));
    }
}

inline uint32_t CountConfigured(const PmuOwnerControl &control)
{
    uint32_t count = 0U;
    for (uint32_t index = 0U; index < kPhysicalSubcoreCount; ++index) {
        count += IsConfigured(control, index) ? 1U : 0U;
    }
    return count;
}

}  // namespace pa_scheduler::pmu_owner

#endif  // PA_SCHEDULER_CCEC_PMU_OWNER_CONTROL_H_
