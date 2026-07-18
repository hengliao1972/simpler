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

#include "pmu_probe_control.h"

#include "aicpu/platform_regs.h"
#include "common/kernel_args.h"

#include <cstddef>
#include <cstdint>

namespace {

using atomic_probe::pmu::PmuControl;

void FlushControl(const PmuControl *control)
{
    const uintptr_t begin = reinterpret_cast<uintptr_t>(control);
    const uintptr_t end = begin + sizeof(*control);
    for (uintptr_t address = begin; address < end; address += 64) {
        __asm__ volatile("dc cvac, %0" : : "r"(address) : "memory");
    }
    __asm__ volatile("dsb sy" ::: "memory");
    __asm__ volatile("isb" ::: "memory");
}

bool SavedConfigurationMatches(const PmuControl *control, uint64_t base, uint32_t index)
{
    return read_reg(base, RegId::PMU_CTRL_0) == control->saved_ctrl0[index] &&
        read_reg(base, RegId::PMU_CTRL_1) == control->saved_ctrl1[index] &&
        read_reg(base, RegId::PMU_CNT0_IDX) == control->saved_selector0[index] &&
        read_reg(base, RegId::PMU_CNT1_IDX) == control->saved_selector1[index] &&
        read_reg(base, RegId::PMU_CNT2_IDX) == control->saved_selector2[index] &&
        read_reg(base, RegId::PMU_START_CYC0) == control->saved_start_cycle_low[index] &&
        read_reg(base, RegId::PMU_START_CYC1) == control->saved_start_cycle_high[index] &&
        read_reg(base, RegId::PMU_STOP_CYC0) == control->saved_stop_cycle_low[index] &&
        read_reg(base, RegId::PMU_STOP_CYC1) == control->saved_stop_cycle_high[index];
}

bool ProbeConfigurationMatches(uint64_t base)
{
    return read_reg(base, RegId::PMU_CTRL_0) == REG_MMIO_PMU_CTRL_0_ENABLE_VAL &&
        read_reg(base, RegId::PMU_CTRL_1) == REG_MMIO_PMU_CTRL_1_ENABLE_VAL &&
        read_reg(base, RegId::PMU_CNT0_IDX) == 0x1U && read_reg(base, RegId::PMU_CNT1_IDX) == 0x34U &&
        read_reg(base, RegId::PMU_CNT2_IDX) == 0x35U && read_reg(base, RegId::PMU_START_CYC0) == 0 &&
        read_reg(base, RegId::PMU_START_CYC1) == 0 && read_reg(base, RegId::PMU_STOP_CYC0) == 0xffffffffU &&
        read_reg(base, RegId::PMU_STOP_CYC1) == 0xffffffffU;
}

bool RestoreOne(PmuControl *control, uint64_t base, uint32_t index)
{
    write_reg(base, RegId::PMU_CTRL_0, 0);
    write_reg(base, RegId::PMU_CTRL_1, 0);
    write_reg(base, RegId::PMU_CNT0_IDX, control->saved_selector0[index]);
    write_reg(base, RegId::PMU_CNT1_IDX, control->saved_selector1[index]);
    write_reg(base, RegId::PMU_CNT2_IDX, control->saved_selector2[index]);
    write_reg(base, RegId::PMU_START_CYC0, control->saved_start_cycle_low[index]);
    write_reg(base, RegId::PMU_START_CYC1, control->saved_start_cycle_high[index]);
    write_reg(base, RegId::PMU_STOP_CYC0, control->saved_stop_cycle_low[index]);
    write_reg(base, RegId::PMU_STOP_CYC1, control->saved_stop_cycle_high[index]);
    write_reg(base, RegId::PMU_CTRL_0, control->saved_ctrl0[index]);
    write_reg(base, RegId::PMU_CTRL_1, control->saved_ctrl1[index]);
    return SavedConfigurationMatches(control, base, index);
}

int Configure(PmuControl *control, const uint64_t *register_bases)
{
    if (control->configured != 0) return -10;
    control->processed_subcores = 0;
    for (uint32_t index = 0; index < atomic_probe::pmu::kPmuPhysicalSubcores; ++index) {
        const uint64_t base = register_bases[index];
        if (base == 0) {
            while (control->processed_subcores != 0) {
                --control->processed_subcores;
                const uint32_t rollback = control->processed_subcores;
                (void)RestoreOne(control, register_bases[rollback], rollback);
            }
            return -11;
        }

        control->saved_ctrl0[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_CTRL_0));
        control->saved_ctrl1[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_CTRL_1));
        control->saved_selector0[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_CNT0_IDX));
        control->saved_selector1[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_CNT1_IDX));
        control->saved_selector2[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_CNT2_IDX));
        control->saved_start_cycle_low[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_START_CYC0));
        control->saved_start_cycle_high[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_START_CYC1));
        control->saved_stop_cycle_low[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_STOP_CYC0));
        control->saved_stop_cycle_high[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_STOP_CYC1));

        // 先冻结框架，再配置 Custom 三事件及完整计数周期；最后启用
        // GLB_PMU_EN | USER_PMU_MODE_EN | SAMPLE_PMU_MODE_EN。
        write_reg(base, RegId::PMU_CTRL_0, 0);
        write_reg(base, RegId::PMU_CTRL_1, 0);
        write_reg(base, RegId::PMU_CNT0_IDX, 0x1U);
        write_reg(base, RegId::PMU_CNT1_IDX, 0x34U);
        write_reg(base, RegId::PMU_CNT2_IDX, 0x35U);
        for (int counter = 0; counter < 10; ++counter) {
            (void)read_reg(base, reg_index(RegId::PMU_CNT0, counter));
        }
        (void)read_reg(base, RegId::PMU_CNT_TOTAL0);
        (void)read_reg(base, RegId::PMU_CNT_TOTAL1);
        write_reg(base, RegId::PMU_START_CYC0, 0);
        write_reg(base, RegId::PMU_START_CYC1, 0);
        write_reg(base, RegId::PMU_STOP_CYC0, 0xffffffffU);
        write_reg(base, RegId::PMU_STOP_CYC1, 0xffffffffU);
        write_reg(base, RegId::PMU_CTRL_0, REG_MMIO_PMU_CTRL_0_ENABLE_VAL);
        write_reg(base, RegId::PMU_CTRL_1, REG_MMIO_PMU_CTRL_1_ENABLE_VAL);
        if (!ProbeConfigurationMatches(base)) {
            (void)RestoreOne(control, base, index);
            while (control->processed_subcores != 0) {
                --control->processed_subcores;
                const uint32_t rollback = control->processed_subcores;
                (void)RestoreOne(control, register_bases[rollback], rollback);
            }
            return -12;
        }
        control->processed_subcores = index + 1;
    }
    control->configured = 1;
    return 0;
}

int Restore(PmuControl *control, const uint64_t *register_bases)
{
    if (control->configured == 0 || control->processed_subcores != atomic_probe::pmu::kPmuPhysicalSubcores) {
        return -20;
    }
    bool all_restored = true;
    while (control->processed_subcores != 0) {
        --control->processed_subcores;
        const uint32_t index = control->processed_subcores;
        all_restored &= RestoreOne(control, register_bases[index], index);
    }
    control->configured = 0;
    return all_restored ? 0 : -21;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_exec(void *argument)
{
    if (argument == nullptr) return -1;
    auto *kernel_args = reinterpret_cast<KernelArgs *>(argument);
    auto *control = reinterpret_cast<atomic_probe::pmu::PmuControl *>(kernel_args->runtime_args);
    auto *register_bases = reinterpret_cast<const uint64_t *>(kernel_args->regs);
    if (control == nullptr || register_bases == nullptr) return -2;

    // command 位于每次 launch 都由 CANN 重新复制的 inline KernelArgs 中；
    // PmuControl 初始化后只由 AICPU 写，因此这里不依赖 EL0 cache invalidate。
    control->command = kernel_args->enable_profiling_flag;
    control->status = atomic_probe::pmu::kPmuStatusPending;
    if (control->magic != atomic_probe::pmu::kPmuControlMagic ||
        control->version != atomic_probe::pmu::kPmuControlVersion ||
        control->expected_subcores != atomic_probe::pmu::kPmuPhysicalSubcores) {
        control->status = -3;
        FlushControl(control);
        return -3;
    }

    int status = -4;
    const auto command = static_cast<atomic_probe::pmu::PmuCommand>(kernel_args->enable_profiling_flag);
    if (command == atomic_probe::pmu::PmuCommand::Configure) {
        status = Configure(control, register_bases);
    } else if (command == atomic_probe::pmu::PmuCommand::Restore) {
        status = Restore(control, register_bases);
    }
    control->status = status;
    FlushControl(control);
    return status;
}
