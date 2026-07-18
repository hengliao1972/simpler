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
using atomic_probe::pmu::PmuRegisterField;

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

void ResetFailureDiagnostic(PmuControl *control)
{
    control->first_failed_index = atomic_probe::pmu::kPmuDiagnosticUnset;
    control->first_failed_field = static_cast<uint32_t>(PmuRegisterField::None);
    control->first_failed_observed = 0;
    control->first_failed_expected = 0;
}

void RecordFirstFailure(
    PmuControl *control, uint32_t index, PmuRegisterField field, uint32_t observed, uint32_t expected
)
{
    if (control->first_failed_index != atomic_probe::pmu::kPmuDiagnosticUnset) return;
    control->first_failed_index = index;
    control->first_failed_field = static_cast<uint32_t>(field);
    control->first_failed_observed = observed;
    control->first_failed_expected = expected;
}

bool CheckProbeRegister(
    uint64_t base, RegId reg, uint32_t expected, PmuRegisterField field,
    PmuRegisterField *failed_field, uint32_t *observed
)
{
    const uint32_t actual = static_cast<uint32_t>(read_reg(base, reg));
    if (actual == expected) return true;
    *failed_field = field;
    *observed = actual;
    return false;
}

bool ProbeConfigurationMatches(
    uint64_t base, PmuRegisterField *failed_field, uint32_t *observed, uint32_t *expected
)
{
    struct RegisterExpectation {
        RegId reg;
        uint32_t value;
        PmuRegisterField field;
    };
    const RegisterExpectation expectations[] = {
        {RegId::PMU_CTRL_0, REG_MMIO_PMU_CTRL_0_ENABLE_VAL, PmuRegisterField::Ctrl0},
        {RegId::PMU_CTRL_1, REG_MMIO_PMU_CTRL_1_ENABLE_VAL, PmuRegisterField::Ctrl1},
        {RegId::PMU_CNT0_IDX, 0x1U, PmuRegisterField::Selector0},
        {RegId::PMU_CNT1_IDX, 0x34U, PmuRegisterField::Selector1},
        {RegId::PMU_CNT2_IDX, 0x35U, PmuRegisterField::Selector2},
        {RegId::PMU_START_CYC0, 0U, PmuRegisterField::StartCycleLow},
        {RegId::PMU_START_CYC1, 0U, PmuRegisterField::StartCycleHigh},
        {RegId::PMU_STOP_CYC0, 0xffffffffU, PmuRegisterField::StopCycleLow},
        {RegId::PMU_STOP_CYC1, 0xffffffffU, PmuRegisterField::StopCycleHigh},
    };
    for (const RegisterExpectation &expectation : expectations) {
        if (!CheckProbeRegister(
                base, expectation.reg, expectation.value, expectation.field, failed_field, observed
            )) {
            *expected = expectation.value;
            return false;
        }
    }
    return true;
}

bool SavedConfigurationMatches(
    const PmuControl *control, uint64_t base, uint32_t index,
    PmuRegisterField *failed_field, uint32_t *observed, uint32_t *expected
)
{
    struct RegisterExpectation {
        RegId reg;
        uint32_t value;
        PmuRegisterField field;
    };
    const RegisterExpectation expectations[] = {
        {RegId::PMU_CTRL_0, control->saved_ctrl0[index], PmuRegisterField::Ctrl0},
        {RegId::PMU_CTRL_1, control->saved_ctrl1[index], PmuRegisterField::Ctrl1},
        {RegId::PMU_CNT0_IDX, control->saved_selector0[index], PmuRegisterField::Selector0},
        {RegId::PMU_CNT1_IDX, control->saved_selector1[index], PmuRegisterField::Selector1},
        {RegId::PMU_CNT2_IDX, control->saved_selector2[index], PmuRegisterField::Selector2},
        {RegId::PMU_START_CYC0, control->saved_start_cycle_low[index], PmuRegisterField::StartCycleLow},
        {RegId::PMU_START_CYC1, control->saved_start_cycle_high[index], PmuRegisterField::StartCycleHigh},
        {RegId::PMU_STOP_CYC0, control->saved_stop_cycle_low[index], PmuRegisterField::StopCycleLow},
        {RegId::PMU_STOP_CYC1, control->saved_stop_cycle_high[index], PmuRegisterField::StopCycleHigh},
    };
    for (const RegisterExpectation &expectation : expectations) {
        if (!CheckProbeRegister(
                base, expectation.reg, expectation.value, expectation.field, failed_field, observed
            )) {
            *expected = expectation.value;
            return false;
        }
    }
    return true;
}

bool RestoreOne(
    PmuControl *control, uint64_t base, uint32_t index,
    PmuRegisterField *failed_field, uint32_t *observed, uint32_t *expected
)
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
    return SavedConfigurationMatches(control, base, index, failed_field, observed, expected);
}

void SaveOne(PmuControl *control, uint64_t base, uint32_t index)
{
    control->saved_ctrl0[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_CTRL_0));
    control->saved_ctrl1[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_CTRL_1));
    control->saved_selector0[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_CNT0_IDX));
    control->saved_selector1[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_CNT1_IDX));
    control->saved_selector2[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_CNT2_IDX));
    control->saved_start_cycle_low[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_START_CYC0));
    control->saved_start_cycle_high[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_START_CYC1));
    control->saved_stop_cycle_low[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_STOP_CYC0));
    control->saved_stop_cycle_high[index] = static_cast<uint32_t>(read_reg(base, RegId::PMU_STOP_CYC1));
}

void ConfigureOne(uint64_t base)
{
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
}

bool RestoreConfiguredBitmap(PmuControl *control, const uint64_t *register_bases)
{
    bool all_restored = true;
    for (uint32_t next = atomic_probe::pmu::kPmuPhysicalSubcores; next != 0; --next) {
        const uint32_t index = next - 1;
        if (!atomic_probe::pmu::IsPmuSubcoreConfigured(*control, index)) continue;
        PmuRegisterField failed_field = PmuRegisterField::None;
        uint32_t observed = 0;
        uint32_t expected = 0;
        if (RestoreOne(
                control, register_bases[index], index, &failed_field, &observed, &expected
            )) {
            atomic_probe::pmu::ClearPmuSubcoreConfigured(control, index);
            if (control->processed_subcores != 0) --control->processed_subcores;
        } else {
            RecordFirstFailure(control, index, failed_field, observed, expected);
            all_restored = false;
        }
    }
    return all_restored && control->processed_subcores == 0 &&
        atomic_probe::pmu::CountPmuConfiguredSubcores(*control) == 0;
}

int Configure(PmuControl *control, const uint64_t *register_bases)
{
    if (control->configured != 0) return -10;
    control->processed_subcores = 0;
    control->skipped_subcores = 0;
    for (uint32_t word = 0; word < atomic_probe::pmu::kPmuBitmapWords; ++word) {
        control->configured_bitmap[word] = 0;
    }
    ResetFailureDiagnostic(control);
    for (uint32_t index = 0; index < atomic_probe::pmu::kPmuPhysicalSubcores; ++index) {
        const uint64_t base = register_bases[index];
        if (base == 0) {
            RecordFirstFailure(control, index, PmuRegisterField::RegisterBase, 0, 1);
            ++control->skipped_subcores;
            continue;
        }

        SaveOne(control, base, index);
        ConfigureOne(base);
        PmuRegisterField failed_field = PmuRegisterField::None;
        uint32_t observed = 0;
        uint32_t expected = 0;
        if (!ProbeConfigurationMatches(base, &failed_field, &observed, &expected)) {
            RecordFirstFailure(control, index, failed_field, observed, expected);
            ++control->skipped_subcores;
            PmuRegisterField restore_failed_field = PmuRegisterField::None;
            uint32_t restore_observed = 0;
            uint32_t restore_expected = 0;
            if (!RestoreOne(
                    control, base, index, &restore_failed_field, &restore_observed, &restore_expected
                )) {
                // 该项没有进入成功 bitmap，无法在后续 Restore 命令中重试；
                // 先回滚此前成功项，再用独立状态区分“探测失败且现场恢复失败”。
                const bool rollback_ok = RestoreConfiguredBitmap(control, register_bases);
                control->configured = control->processed_subcores == 0 ? 0U : 1U;
                return rollback_ok ? -13 : -14;
            }
            continue;
        }
        atomic_probe::pmu::SetPmuSubcoreConfigured(control, index);
        ++control->processed_subcores;
    }

    if (control->processed_subcores != control->expected_subcores ||
        atomic_probe::pmu::CountPmuConfiguredSubcores(*control) != control->expected_subcores) {
        RecordFirstFailure(
            control, atomic_probe::pmu::kPmuPhysicalSubcores, PmuRegisterField::ConfiguredCount,
            control->processed_subcores, control->expected_subcores
        );
        const bool rollback_ok = RestoreConfiguredBitmap(control, register_bases);
        control->configured = control->processed_subcores == 0 ? 0U : 1U;
        return rollback_ok ? -12 : -14;
    }
    control->configured = 1;
    return 0;
}

int Restore(PmuControl *control, const uint64_t *register_bases)
{
    if (control->configured == 0 || control->processed_subcores == 0 ||
        control->processed_subcores != atomic_probe::pmu::CountPmuConfiguredSubcores(*control)) {
        return -20;
    }
    ResetFailureDiagnostic(control);
    const bool all_restored = RestoreConfiguredBitmap(control, register_bases);
    control->configured = all_restored ? 0U : 1U;
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
        control->expected_subcores == 0 ||
        control->expected_subcores > atomic_probe::pmu::kPmuPhysicalSubcores) {
        control->status = -3;
        FlushControl(control);
        return 0;
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
    // AICPU entry 的非零返回会被 runtime 升格为 stream 异常，host 从而无法
    // D2H 读取上面的精确状态。协议级成败统一由 control->status 传递，
    // entry 只报告“命令已执行并已发布状态”。
    return 0;
}
