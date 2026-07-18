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
constexpr uint32_t kPmuControlVersion = 1;
constexpr uint32_t kPmuPhysicalSubcores = 108;
constexpr int32_t kPmuStatusPending = 0x7fffffff;

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
    uint32_t processed_subcores;
    uint32_t expected_subcores;
    uint32_t reserved[9];

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

}  // namespace atomic_probe::pmu

#endif  // TESTS_ATOMIC_PROBE_CCEC_PMU_PROBE_CONTROL_H_
