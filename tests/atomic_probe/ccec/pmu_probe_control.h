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

#ifndef TESTS_ATOMIC_PROBE_CCEC_PMU_PROBE_CONTROL_H_
#define TESTS_ATOMIC_PROBE_CCEC_PMU_PROBE_CONTROL_H_

#include <cstddef>
#include <cstdint>

namespace atomic_probe::pmu {

constexpr uint32_t kPmuControlMagic = 0x504d5551U;  // "PMUQ"
constexpr uint32_t kPmuControlVersion = 2;
constexpr uint32_t kPmuPhysicalSubcores = 108;
constexpr uint32_t kPmuBitmapWords = (kPmuPhysicalSubcores + 31U) / 32U;
constexpr int32_t kPmuStatusPending = 0x7fffffff;
constexpr uint32_t kPmuDiagnosticUnset = 0xffffffffU;

enum class PmuRegisterField : uint32_t {
    None = 0,
    RegisterBase,
    Ctrl0,
    Ctrl1,
    Selector0,
    Selector1,
    Selector2,
    StartCycleLow,
    StartCycleHigh,
    StopCycleLow,
    StopCycleHigh,
    ConfiguredCount,
};

enum class PmuCommand : uint32_t {
    Configure = 1,
    Restore = 2,
};

// Host 与单线程 AICPU helper 共享的 PMU 所有权记录。helper 在 Configure
// 阶段保存被改寄存器，在 Restore 阶段先关 PMU、恢复 selector/range，最后恢复 CTRL。
// counter 本身是 read-to-clear，旧计数内容无法恢复，因此 PMU session 必须独占。
struct alignas(64) PmuControl {
    uint32_t magic;
    uint32_t version;
    uint32_t command;
    volatile int32_t status;
    uint32_t configured;
    // v2 中这是 bitmap 内成功配置的数量，不再表示 0..N 连续前缀。
    uint32_t processed_subcores;
    // 由 host 从同一 stream 的 cube/vector resource limit 求和后填入。
    uint32_t expected_subcores;
    uint32_t configured_bitmap[kPmuBitmapWords];
    uint32_t first_failed_index;
    uint32_t first_failed_field;
    uint32_t first_failed_observed;
    uint32_t first_failed_expected;
    uint32_t skipped_subcores;

    uint32_t saved_ctrl0[kPmuPhysicalSubcores];
    uint32_t saved_ctrl1[kPmuPhysicalSubcores];
    uint32_t saved_selector0[kPmuPhysicalSubcores];
    uint32_t saved_selector1[kPmuPhysicalSubcores];
    uint32_t saved_selector2[kPmuPhysicalSubcores];
    uint32_t saved_start_cycle_low[kPmuPhysicalSubcores];
    uint32_t saved_start_cycle_high[kPmuPhysicalSubcores];
    uint32_t saved_stop_cycle_low[kPmuPhysicalSubcores];
    uint32_t saved_stop_cycle_high[kPmuPhysicalSubcores];
};

static_assert(offsetof(PmuControl, saved_ctrl0) == 64, "PMU control header must occupy one cache line");
static_assert(sizeof(PmuControl) % 64 == 0, "PMU control must use complete cache lines");

inline bool IsPmuSubcoreConfigured(const PmuControl &control, uint32_t index)
{
    return index < kPmuPhysicalSubcores &&
        (control.configured_bitmap[index / 32U] & (1U << (index % 32U))) != 0;
}

inline void SetPmuSubcoreConfigured(PmuControl *control, uint32_t index)
{
    if (index < kPmuPhysicalSubcores) {
        control->configured_bitmap[index / 32U] |= 1U << (index % 32U);
    }
}

inline void ClearPmuSubcoreConfigured(PmuControl *control, uint32_t index)
{
    if (index < kPmuPhysicalSubcores) {
        control->configured_bitmap[index / 32U] &= ~(1U << (index % 32U));
    }
}

inline uint32_t CountPmuConfiguredSubcores(const PmuControl &control)
{
    uint32_t count = 0;
    for (uint32_t index = 0; index < kPmuPhysicalSubcores; ++index) {
        count += IsPmuSubcoreConfigured(control, index) ? 1U : 0U;
    }
    return count;
}

inline const char *PmuRegisterFieldName(PmuRegisterField field)
{
    switch (field) {
        case PmuRegisterField::None: return "none";
        case PmuRegisterField::RegisterBase: return "register-base";
        case PmuRegisterField::Ctrl0: return "ctrl0";
        case PmuRegisterField::Ctrl1: return "ctrl1";
        case PmuRegisterField::Selector0: return "selector0";
        case PmuRegisterField::Selector1: return "selector1";
        case PmuRegisterField::Selector2: return "selector2";
        case PmuRegisterField::StartCycleLow: return "start-cycle-low";
        case PmuRegisterField::StartCycleHigh: return "start-cycle-high";
        case PmuRegisterField::StopCycleLow: return "stop-cycle-low";
        case PmuRegisterField::StopCycleHigh: return "stop-cycle-high";
        case PmuRegisterField::ConfiguredCount: return "configured-count";
        default: return "unknown";
    }
}

}  // namespace atomic_probe::pmu

#endif  // TESTS_ATOMIC_PROBE_CCEC_PMU_PROBE_CONTROL_H_
