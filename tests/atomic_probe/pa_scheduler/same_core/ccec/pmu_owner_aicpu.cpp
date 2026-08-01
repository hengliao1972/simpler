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

#include "pmu_owner_control.h"
#include "pmu_owner_main_abi.h"

#include <cstddef>
#include <cstdint>

namespace {

using pa_scheduler::pmu_owner::PmuOwnerControl;
using pa_scheduler::pmu_owner::PmuOwnerField;
using pa_scheduler::pmu_owner::PmuOwnerMainCommand;
using pa_scheduler::pmu_owner::PmuOwnerMainKernelArgs;
using pa_scheduler::pmu_owner::PmuOwnerStatus;
using pa_scheduler::pmu_owner::PmuSavedRegisters;

constexpr size_t kCacheLineBytes = 64U;

// 以下 offset 逐项核对自当前 A5 的 platform_config.h 与 onboard
// inner_platform_regs.cpp。standalone helper 在本目录重写这些常量，绝不
// include pa_scheduler 目录外的 Simpler platform 实现。
constexpr uint32_t kCtrl0Offset = 0x4200U;
constexpr uint32_t kCtrl1Offset = 0x2400U;
constexpr uint32_t kCounterOffsets[pa_scheduler::pmu_owner::kPmuCounterCount] = {
    0x4210U, 0x4218U, 0x4220U, 0x4228U, 0x4230U,
    0x4238U, 0x4240U, 0x4248U, 0x4250U, 0x4254U,
};
constexpr uint32_t kSelectorOffsets[pa_scheduler::pmu_owner::kPmuCounterCount] = {
    0x2500U, 0x2504U, 0x2508U, 0x250cU, 0x2510U,
    0x2514U, 0x2518U, 0x251cU, 0x2520U, 0x2524U,
};
constexpr uint32_t kTotalLowOffset = 0x4260U;
constexpr uint32_t kTotalHighOffset = 0x4264U;
constexpr uint32_t kStartCycleLowOffset = 0x42a0U;
constexpr uint32_t kStartCycleHighOffset = 0x42a4U;
constexpr uint32_t kStopCycleLowOffset = 0x42a8U;
constexpr uint32_t kStopCycleHighOffset = 0x42acU;
constexpr uint32_t kCtrl0Enabled = 0x7U;
constexpr uint32_t kCtrl1Enabled = 0x1U;

static_assert(
    sizeof(kCounterOffsets) / sizeof(kCounterOffsets[0]) == pa_scheduler::pmu_owner::kPmuCounterCount,
    "PMU counter offset count changed"
);
static_assert(
    sizeof(kSelectorOffsets) / sizeof(kSelectorOffsets[0]) == pa_scheduler::pmu_owner::kPmuCounterCount,
    "PMU selector offset count changed"
);

inline void InstructionBarrier()
{
    __asm__ volatile("isb" ::: "memory");
}

inline void FullSystemBarrier()
{
    __asm__ volatile("dsb sy" ::: "memory");
}

// Host 的 H2D/D2H 不会维护 AICPU L1。一条 Configure/Restore 命令开始前
// 对 control 全区间执行 CIVAC，避免复用同一 GM 地址时读到上一轮 cache 内容。
void InvalidateControl(const PmuOwnerControl *control)
{
    const uintptr_t begin = reinterpret_cast<uintptr_t>(control) & ~(kCacheLineBytes - 1U);
    const uintptr_t end =
        (reinterpret_cast<uintptr_t>(control) + sizeof(*control) + kCacheLineBytes - 1U) &
        ~(kCacheLineBytes - 1U);
    for (uintptr_t address = begin; address < end; address += kCacheLineBytes) {
        __asm__ volatile("dc civac, %0" : : "r"(address) : "memory");
    }
    FullSystemBarrier();
    InstructionBarrier();
}

// 命令完成后 clean 整个 control：不仅发布 status，也发布 bitmap、诊断以及
// Configure 保存区。最后的 DSB/ISB 保证 runtime 同步返回后 host D2H 可见。
void CleanControl(const PmuOwnerControl *control)
{
    const uintptr_t begin = reinterpret_cast<uintptr_t>(control) & ~(kCacheLineBytes - 1U);
    const uintptr_t end =
        (reinterpret_cast<uintptr_t>(control) + sizeof(*control) + kCacheLineBytes - 1U) &
        ~(kCacheLineBytes - 1U);
    for (uintptr_t address = begin; address < end; address += kCacheLineBytes) {
        __asm__ volatile("dc cvac, %0" : : "r"(address) : "memory");
    }
    FullSystemBarrier();
    InstructionBarrier();
}

inline volatile uint32_t *MmioPointer(uint64_t base, uint32_t offset)
{
    return reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(base + offset));
}

inline uint32_t ReadMmio(uint64_t base, uint32_t offset)
{
    return *MmioPointer(base, offset);
}

inline void WriteMmio(uint64_t base, uint32_t offset, uint32_t value)
{
    *MmioPointer(base, offset) = value;
}

constexpr PmuOwnerField SelectorField(uint32_t counter)
{
    return static_cast<PmuOwnerField>(
        static_cast<uint32_t>(PmuOwnerField::Selector0) + counter
    );
}

void ResetDiagnostics(PmuOwnerControl *control)
{
    control->first_failed_index = pa_scheduler::pmu_owner::kDiagnosticIndexUnset;
    control->first_failed_field = static_cast<uint32_t>(PmuOwnerField::None);
    control->first_failed_observed = 0U;
    control->first_failed_expected = 0U;
    control->restore_failures = 0U;
    control->first_restore_failed_index = pa_scheduler::pmu_owner::kDiagnosticIndexUnset;
    control->first_restore_failed_field = static_cast<uint32_t>(PmuOwnerField::None);
    control->first_restore_failed_observed = 0U;
    control->first_restore_failed_expected = 0U;
}

void RecordFirstFailure(
    PmuOwnerControl *control, uint32_t index, PmuOwnerField field,
    uint32_t observed, uint32_t expected
)
{
    if (control->first_failed_index != pa_scheduler::pmu_owner::kDiagnosticIndexUnset) return;
    control->first_failed_index = index;
    control->first_failed_field = static_cast<uint32_t>(field);
    control->first_failed_observed = observed;
    control->first_failed_expected = expected;
}

void RecordRestoreFailure(
    PmuOwnerControl *control, uint32_t index, PmuOwnerField field,
    uint32_t observed, uint32_t expected
)
{
    ++control->restore_failures;
    if (control->first_restore_failed_index != pa_scheduler::pmu_owner::kDiagnosticIndexUnset) return;
    control->first_restore_failed_index = index;
    control->first_restore_failed_field = static_cast<uint32_t>(field);
    control->first_restore_failed_observed = observed;
    control->first_restore_failed_expected = expected;
}

bool CheckRegister(
    uint64_t base, uint32_t offset, uint32_t expected, PmuOwnerField field,
    PmuOwnerField *failed_field, uint32_t *observed
)
{
    const uint32_t actual = ReadMmio(base, offset);
    if (actual == expected) return true;
    *failed_field = field;
    *observed = actual;
    return false;
}

void SaveOne(PmuOwnerControl *control, uint64_t base, uint32_t index)
{
    PmuSavedRegisters &saved = control->saved[index];
    saved.ctrl0 = ReadMmio(base, kCtrl0Offset);
    saved.ctrl1 = ReadMmio(base, kCtrl1Offset);
    for (uint32_t counter = 0U; counter < pa_scheduler::pmu_owner::kPmuCounterCount; ++counter) {
        saved.selectors[counter] = ReadMmio(base, kSelectorOffsets[counter]);
    }
    saved.start_cycle_low = ReadMmio(base, kStartCycleLowOffset);
    saved.start_cycle_high = ReadMmio(base, kStartCycleHighOffset);
    saved.stop_cycle_low = ReadMmio(base, kStopCycleLowOffset);
    saved.stop_cycle_high = ReadMmio(base, kStopCycleHighOffset);
}

void ConfigureOne(uint64_t base)
{
    // 先冻结 PMU，再写完整 10 槽 selector。submit-pmu 的 CNT8/CNT5 是
    // request/miss shadow；包括 unused CNT9 在内的所有槽都会在 Restore 恢复。
    WriteMmio(base, kCtrl0Offset, 0U);
    WriteMmio(base, kCtrl1Offset, 0U);
    for (uint32_t counter = 0U; counter < pa_scheduler::pmu_owner::kPmuCounterCount; ++counter) {
        WriteMmio(base, kSelectorOffsets[counter], pa_scheduler::pmu_owner::kConfiguredSelectors[counter]);
    }

    // A5 counter 是 read-to-clear。旧计数无法保存，所以先清十个事件 counter
    // 和 64-bit total，之后整个 configure→restore 区间由本 owner 独占。
    for (uint32_t counter = 0U; counter < pa_scheduler::pmu_owner::kPmuCounterCount; ++counter) {
        (void)ReadMmio(base, kCounterOffsets[counter]);
    }
    (void)ReadMmio(base, kTotalLowOffset);
    (void)ReadMmio(base, kTotalHighOffset);

    WriteMmio(base, kStartCycleLowOffset, 0U);
    WriteMmio(base, kStartCycleHighOffset, 0U);
    WriteMmio(base, kStopCycleLowOffset, 0xffffffffU);
    WriteMmio(base, kStopCycleHighOffset, 0xffffffffU);
    WriteMmio(base, kCtrl0Offset, kCtrl0Enabled);
    WriteMmio(base, kCtrl1Offset, kCtrl1Enabled);
    // Device-nGnRnE 会维持同一区域顺序；额外 DSB 只位于 owner 冷路径，用于
    // 保证下面的逐寄存器读回发生在所有配置写真正完成之后。
    FullSystemBarrier();
}

bool ConfigurationMatches(
    uint64_t base, PmuOwnerField *failed_field, uint32_t *observed, uint32_t *expected
)
{
    if (!CheckRegister(base, kCtrl0Offset, kCtrl0Enabled, PmuOwnerField::Ctrl0, failed_field, observed)) {
        *expected = kCtrl0Enabled;
        return false;
    }
    if (!CheckRegister(base, kCtrl1Offset, kCtrl1Enabled, PmuOwnerField::Ctrl1, failed_field, observed)) {
        *expected = kCtrl1Enabled;
        return false;
    }
    for (uint32_t counter = 0U; counter < pa_scheduler::pmu_owner::kPmuCounterCount; ++counter) {
        const uint32_t selector = pa_scheduler::pmu_owner::kConfiguredSelectors[counter];
        if (!CheckRegister(
                base, kSelectorOffsets[counter], selector, SelectorField(counter), failed_field, observed
            )) {
            *expected = selector;
            return false;
        }
    }
    if (!CheckRegister(base, kStartCycleLowOffset, 0U, PmuOwnerField::StartCycleLow, failed_field, observed)) {
        *expected = 0U;
        return false;
    }
    if (!CheckRegister(base, kStartCycleHighOffset, 0U, PmuOwnerField::StartCycleHigh, failed_field, observed)) {
        *expected = 0U;
        return false;
    }
    if (!CheckRegister(
            base, kStopCycleLowOffset, 0xffffffffU, PmuOwnerField::StopCycleLow, failed_field, observed
        )) {
        *expected = 0xffffffffU;
        return false;
    }
    if (!CheckRegister(
            base, kStopCycleHighOffset, 0xffffffffU, PmuOwnerField::StopCycleHigh, failed_field, observed
        )) {
        *expected = 0xffffffffU;
        return false;
    }
    return true;
}

void WriteSavedConfiguration(const PmuOwnerControl &control, uint64_t base, uint32_t index)
{
    const PmuSavedRegisters &saved = control.saved[index];
    WriteMmio(base, kCtrl0Offset, 0U);
    WriteMmio(base, kCtrl1Offset, 0U);
    for (uint32_t counter = 0U; counter < pa_scheduler::pmu_owner::kPmuCounterCount; ++counter) {
        WriteMmio(base, kSelectorOffsets[counter], saved.selectors[counter]);
    }
    WriteMmio(base, kStartCycleLowOffset, saved.start_cycle_low);
    WriteMmio(base, kStartCycleHighOffset, saved.start_cycle_high);
    WriteMmio(base, kStopCycleLowOffset, saved.stop_cycle_low);
    WriteMmio(base, kStopCycleHighOffset, saved.stop_cycle_high);
    // CTRL 最后恢复，避免在 selector/range 尚未回到原值时短暂恢复旧计数状态。
    WriteMmio(base, kCtrl0Offset, saved.ctrl0);
    WriteMmio(base, kCtrl1Offset, saved.ctrl1);
    FullSystemBarrier();
}

bool SavedConfigurationMatches(
    const PmuOwnerControl &control, uint64_t base, uint32_t index,
    PmuOwnerField *failed_field, uint32_t *observed, uint32_t *expected
)
{
    const PmuSavedRegisters &saved = control.saved[index];
    if (!CheckRegister(base, kCtrl0Offset, saved.ctrl0, PmuOwnerField::Ctrl0, failed_field, observed)) {
        *expected = saved.ctrl0;
        return false;
    }
    if (!CheckRegister(base, kCtrl1Offset, saved.ctrl1, PmuOwnerField::Ctrl1, failed_field, observed)) {
        *expected = saved.ctrl1;
        return false;
    }
    for (uint32_t counter = 0U; counter < pa_scheduler::pmu_owner::kPmuCounterCount; ++counter) {
        if (!CheckRegister(
                base, kSelectorOffsets[counter], saved.selectors[counter], SelectorField(counter),
                failed_field, observed
            )) {
            *expected = saved.selectors[counter];
            return false;
        }
    }
    if (!CheckRegister(
            base, kStartCycleLowOffset, saved.start_cycle_low, PmuOwnerField::StartCycleLow,
            failed_field, observed
        )) {
        *expected = saved.start_cycle_low;
        return false;
    }
    if (!CheckRegister(
            base, kStartCycleHighOffset, saved.start_cycle_high, PmuOwnerField::StartCycleHigh,
            failed_field, observed
        )) {
        *expected = saved.start_cycle_high;
        return false;
    }
    if (!CheckRegister(
            base, kStopCycleLowOffset, saved.stop_cycle_low, PmuOwnerField::StopCycleLow,
            failed_field, observed
        )) {
        *expected = saved.stop_cycle_low;
        return false;
    }
    if (!CheckRegister(
            base, kStopCycleHighOffset, saved.stop_cycle_high, PmuOwnerField::StopCycleHigh,
            failed_field, observed
        )) {
        *expected = saved.stop_cycle_high;
        return false;
    }
    return true;
}

bool RestoreOne(
    PmuOwnerControl *control, uint32_t index,
    PmuOwnerField *failed_field, uint32_t *observed, uint32_t *expected
)
{
    const uint64_t base = control->register_bases[index];
    if (base == 0U) {
        *failed_field = PmuOwnerField::RegisterBase;
        *observed = 0U;
        *expected = 1U;
        return false;
    }
    WriteSavedConfiguration(*control, base, index);
    return SavedConfigurationMatches(*control, base, index, failed_field, observed, expected);
}

void IncrementRoleCount(uint32_t index, uint32_t *aic, uint32_t *aiv)
{
    if (pa_scheduler::pmu_owner::IsAicPhysicalSlot(index)) {
        ++(*aic);
    } else {
        ++(*aiv);
    }
}

void DecrementActiveRole(PmuOwnerControl *control, uint32_t index)
{
    if (control->active_total != 0U) --control->active_total;
    if (pa_scheduler::pmu_owner::IsAicPhysicalSlot(index)) {
        if (control->active_aic != 0U) --control->active_aic;
    } else if (control->active_aiv != 0U) {
        --control->active_aiv;
    }
}

// 恢复除 skip_index 外的 owned 槽。Configure 的“当前失败槽”若当场恢复失败，
// rollback 会跳过它，确保该 bit 留给 host 随后的幂等 Restore 再次重试。
bool RestoreOwnedBitmapExcept(PmuOwnerControl *control, uint32_t skip_index)
{
    bool all_restored = true;
    for (uint32_t next = pa_scheduler::pmu_owner::kPhysicalSubcoreCount; next != 0U; --next) {
        const uint32_t index = next - 1U;
        if (!pa_scheduler::pmu_owner::IsConfigured(*control, index)) continue;
        if (index == skip_index) continue;

        PmuOwnerField failed_field = PmuOwnerField::None;
        uint32_t observed = 0U;
        uint32_t expected = 0U;
        if (RestoreOne(control, index, &failed_field, &observed, &expected)) {
            pa_scheduler::pmu_owner::ClearConfigured(control, index);
            DecrementActiveRole(control, index);
        } else {
            RecordRestoreFailure(control, index, failed_field, observed, expected);
            all_restored = false;
        }
    }
    return all_restored;
}

// 只消费 owner bitmap，严格按 107→0 恢复。某槽只有在完整读回一致后才清 bit；
// 因而 Restore 失败后可再次调用，下一次只重试仍由 owner 持有的槽。
bool RestoreConfiguredBitmap(PmuOwnerControl *control)
{
    const bool all_restored = RestoreOwnedBitmapExcept(
        control, pa_scheduler::pmu_owner::kDiagnosticIndexUnset
    );
    const uint32_t bitmap_count = pa_scheduler::pmu_owner::CountConfigured(*control);
    return all_restored && bitmap_count == 0U && control->active_total == 0U &&
        control->active_aic == 0U && control->active_aiv == 0U;
}

bool ValidateControlHeader(PmuOwnerControl *control)
{
    if (control->magic != pa_scheduler::pmu_owner::kPmuOwnerControlMagic) {
        RecordFirstFailure(
            control, pa_scheduler::pmu_owner::kPhysicalSubcoreCount, PmuOwnerField::ControlMagic,
            control->magic, pa_scheduler::pmu_owner::kPmuOwnerControlMagic
        );
        return false;
    }
    if (control->version != pa_scheduler::pmu_owner::kPmuOwnerControlVersion) {
        RecordFirstFailure(
            control, pa_scheduler::pmu_owner::kPhysicalSubcoreCount, PmuOwnerField::ControlVersion,
            control->version, pa_scheduler::pmu_owner::kPmuOwnerControlVersion
        );
        return false;
    }
    if (control->struct_bytes != sizeof(PmuOwnerControl)) {
        RecordFirstFailure(
            control, pa_scheduler::pmu_owner::kPhysicalSubcoreCount, PmuOwnerField::ControlSize,
            control->struct_bytes, static_cast<uint32_t>(sizeof(PmuOwnerControl))
        );
        return false;
    }
    return true;
}

bool ValidateExpectedTopology(PmuOwnerControl *control)
{
    if (control->expected_total != pa_scheduler::pmu_owner::kExpectedSubcoreCount) {
        RecordFirstFailure(
            control, pa_scheduler::pmu_owner::kPhysicalSubcoreCount, PmuOwnerField::TotalCount,
            control->expected_total, pa_scheduler::pmu_owner::kExpectedSubcoreCount
        );
        return false;
    }
    if (control->expected_aic != pa_scheduler::pmu_owner::kExpectedAicCount) {
        RecordFirstFailure(
            control, pa_scheduler::pmu_owner::kPhysicalSubcoreCount, PmuOwnerField::AicCount,
            control->expected_aic, pa_scheduler::pmu_owner::kExpectedAicCount
        );
        return false;
    }
    if (control->expected_aiv != pa_scheduler::pmu_owner::kExpectedAivCount) {
        RecordFirstFailure(
            control, pa_scheduler::pmu_owner::kPhysicalSubcoreCount, PmuOwnerField::AivCount,
            control->expected_aiv, pa_scheduler::pmu_owner::kExpectedAivCount
        );
        return false;
    }
    return true;
}

PmuOwnerStatus Configure(PmuOwnerControl *control)
{
    if (control->configured != 0U || control->active_total != 0U ||
        pa_scheduler::pmu_owner::CountConfigured(*control) != 0U) {
        RecordFirstFailure(
            control, pa_scheduler::pmu_owner::kPhysicalSubcoreCount, PmuOwnerField::State,
            control->active_total, 0U
        );
        return PmuOwnerStatus::AlreadyConfigured;
    }

    control->active_total = 0U;
    control->active_aic = 0U;
    control->active_aiv = 0U;
    control->discovered_total = 0U;
    control->discovered_aic = 0U;
    control->discovered_aiv = 0U;
    control->skipped_total = 0U;
    for (uint32_t word = 0U; word < pa_scheduler::pmu_owner::kConfiguredBitmapWords; ++word) {
        control->configured_bitmap[word] = 0U;
    }
    ResetDiagnostics(control);

    for (uint32_t index = 0U; index < pa_scheduler::pmu_owner::kPhysicalSubcoreCount; ++index) {
        const uint64_t base = control->register_bases[index];
        if (base == 0U) {
            RecordFirstFailure(control, index, PmuOwnerField::RegisterBase, 0U, 1U);
            ++control->skipped_total;
            continue;
        }

        SaveOne(control, base, index);
        // 从保存完成到首次 MMIO 改写之间先取得所有权。即使后面的配置读回
        // 失败且当场恢复也失败，该槽仍留在 bitmap 中供后续 Restore 重试。
        pa_scheduler::pmu_owner::SetConfigured(control, index);
        ++control->active_total;
        IncrementRoleCount(index, &control->active_aic, &control->active_aiv);
        ConfigureOne(base);
        PmuOwnerField failed_field = PmuOwnerField::None;
        uint32_t observed = 0U;
        uint32_t expected = 0U;
        if (!ConfigurationMatches(base, &failed_field, &observed, &expected)) {
            RecordFirstFailure(control, index, failed_field, observed, expected);
            ++control->skipped_total;

            // 配置未通过时必须当场恢复；只有原值完整读回一致才能释放该槽
            // 的所有权。若恢复失败，保留它的 bit 并回滚其余 owned 槽。
            PmuOwnerField restore_field = PmuOwnerField::None;
            uint32_t restore_observed = 0U;
            uint32_t restore_expected = 0U;
            if (!RestoreOne(control, index, &restore_field, &restore_observed, &restore_expected)) {
                RecordRestoreFailure(control, index, restore_field, restore_observed, restore_expected);
                const bool rollback_ok = RestoreOwnedBitmapExcept(control, index);
                control->configured = control->active_total == 0U ? 0U : 1U;
                return rollback_ok ? PmuOwnerStatus::ConfigureSlotRestoreFailed :
                                     PmuOwnerStatus::ConfigureRollbackFailed;
            }
            pa_scheduler::pmu_owner::ClearConfigured(control, index);
            DecrementActiveRole(control, index);
            continue;
        }

        // discovered 只统计配置值全部读回一致的物理槽；bitmap/active 则表达
        // 更严格的“仍持有原值快照、尚未恢复”所有权，两者不可混用。
        ++control->discovered_total;
        IncrementRoleCount(index, &control->discovered_aic, &control->discovered_aiv);
    }

    const uint32_t bitmap_count = pa_scheduler::pmu_owner::CountConfigured(*control);
    const bool topology_matches =
        bitmap_count == pa_scheduler::pmu_owner::kExpectedSubcoreCount &&
        control->active_total == pa_scheduler::pmu_owner::kExpectedSubcoreCount &&
        control->active_aic == pa_scheduler::pmu_owner::kExpectedAicCount &&
        control->active_aiv == pa_scheduler::pmu_owner::kExpectedAivCount &&
        control->discovered_total == pa_scheduler::pmu_owner::kExpectedSubcoreCount &&
        control->discovered_aic == pa_scheduler::pmu_owner::kExpectedAicCount &&
        control->discovered_aiv == pa_scheduler::pmu_owner::kExpectedAivCount;
    if (!topology_matches) {
        if (bitmap_count != control->active_total) {
            RecordFirstFailure(
                control, pa_scheduler::pmu_owner::kPhysicalSubcoreCount, PmuOwnerField::BitmapCount,
                bitmap_count, control->active_total
            );
        } else if (control->active_total != pa_scheduler::pmu_owner::kExpectedSubcoreCount) {
            RecordFirstFailure(
                control, pa_scheduler::pmu_owner::kPhysicalSubcoreCount, PmuOwnerField::TotalCount,
                control->active_total, pa_scheduler::pmu_owner::kExpectedSubcoreCount
            );
        } else if (control->active_aic != pa_scheduler::pmu_owner::kExpectedAicCount) {
            RecordFirstFailure(
                control, pa_scheduler::pmu_owner::kPhysicalSubcoreCount, PmuOwnerField::AicCount,
                control->active_aic, pa_scheduler::pmu_owner::kExpectedAicCount
            );
        } else if (control->active_aiv != pa_scheduler::pmu_owner::kExpectedAivCount) {
            RecordFirstFailure(
                control, pa_scheduler::pmu_owner::kPhysicalSubcoreCount, PmuOwnerField::AivCount,
                control->active_aiv, pa_scheduler::pmu_owner::kExpectedAivCount
            );
        } else if (control->discovered_total != pa_scheduler::pmu_owner::kExpectedSubcoreCount) {
            RecordFirstFailure(
                control, pa_scheduler::pmu_owner::kPhysicalSubcoreCount, PmuOwnerField::TotalCount,
                control->discovered_total, pa_scheduler::pmu_owner::kExpectedSubcoreCount
            );
        } else if (control->discovered_aic != pa_scheduler::pmu_owner::kExpectedAicCount) {
            RecordFirstFailure(
                control, pa_scheduler::pmu_owner::kPhysicalSubcoreCount, PmuOwnerField::AicCount,
                control->discovered_aic, pa_scheduler::pmu_owner::kExpectedAicCount
            );
        } else {
            RecordFirstFailure(
                control, pa_scheduler::pmu_owner::kPhysicalSubcoreCount, PmuOwnerField::AivCount,
                control->discovered_aiv, pa_scheduler::pmu_owner::kExpectedAivCount
            );
        }
        const bool rollback_ok = RestoreConfiguredBitmap(control);
        control->configured = control->active_total == 0U ? 0U : 1U;
        return rollback_ok ? PmuOwnerStatus::ConfigureCountMismatch :
                             PmuOwnerStatus::ConfigureRollbackFailed;
    }

    control->configured = 1U;
    return PmuOwnerStatus::Success;
}

PmuOwnerStatus Restore(PmuOwnerControl *control)
{
    // 幂等恢复：bitmap 已空时，无论是一次正常 Restore 后的重入，还是 host
    // 失败路径的兜底调用，都统一收口为“未持有任何 PMU 槽”。
    const uint32_t bitmap_count = pa_scheduler::pmu_owner::CountConfigured(*control);
    if (bitmap_count == 0U) {
        control->configured = 0U;
        control->active_total = 0U;
        control->active_aic = 0U;
        control->active_aiv = 0U;
        ResetDiagnostics(control);
        return PmuOwnerStatus::Success;
    }

    // 若上一次命令在更新计数字段后异常退出，bitmap 才是唯一权威所有权源。
    // Restore 先由 bitmap 重建 active 计数，再逆序恢复，避免陈旧计数阻塞清理。
    control->active_total = bitmap_count;
    control->active_aic = 0U;
    control->active_aiv = 0U;
    for (uint32_t index = 0U; index < pa_scheduler::pmu_owner::kPhysicalSubcoreCount; ++index) {
        if (pa_scheduler::pmu_owner::IsConfigured(*control, index)) {
            IncrementRoleCount(index, &control->active_aic, &control->active_aiv);
        }
    }
    ResetDiagnostics(control);
    const bool restored = RestoreConfiguredBitmap(control);
    control->configured = restored ? 0U : 1U;
    return restored ? PmuOwnerStatus::Success : PmuOwnerStatus::RestoreFailed;
}

// 主 aicpu_scheduler 会复制完整 152B KernelArgs；其中 runtime_args_device
// 指向跨 Configure/Restore 持续存在的 GM control。参数副本本身不需要 cache 维护。
PmuOwnerControl *ResolveControl(const PmuOwnerMainKernelArgs *arguments)
{
    if (arguments == nullptr || arguments->runtime_args_device == 0U ||
        (arguments->runtime_args_device % alignof(PmuOwnerControl)) != 0U) {
        return nullptr;
    }
    return reinterpret_cast<PmuOwnerControl *>(
        static_cast<uintptr_t>(arguments->runtime_args_device)
    );
}

void ExecuteOwnerCommand(void *argument)
{
    const auto *arguments = reinterpret_cast<const PmuOwnerMainKernelArgs *>(argument);
    PmuOwnerControl *control = ResolveControl(arguments);
    if (control == nullptr) return;

    InvalidateControl(control);
    // 从这一行起，即使协议校验失败也把精确业务状态 clean 回 GM。
    control->status = pa_scheduler::pmu_owner::kStatusPending;
    ResetDiagnostics(control);
    if (!ValidateControlHeader(control)) {
        control->status = static_cast<int32_t>(PmuOwnerStatus::InvalidControl);
        CleanControl(control);
        return;
    }
    const auto command = static_cast<PmuOwnerMainCommand>(arguments->command);
    if (command == PmuOwnerMainCommand::Configure && !ValidateExpectedTopology(control)) {
        control->status = static_cast<int32_t>(PmuOwnerStatus::UnexpectedTopology);
        CleanControl(control);
        return;
    }

    PmuOwnerStatus status = PmuOwnerStatus::InvalidArguments;
    if (command == PmuOwnerMainCommand::Configure) {
        status = Configure(control);
    } else if (command == PmuOwnerMainCommand::Restore) {
        // Restore 以 bitmap 为唯一所有权依据。即使 expected_* 诊断字段被局部
        // 覆盖，也优先尝试恢复已经保存的寄存器，避免清理被无关字段阻塞。
        status = Restore(control);
    }
    control->status = static_cast<int32_t>(status);
    CleanControl(control);
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_exec(void *argument)
{
    ExecuteOwnerCommand(argument);
    return 0;
}
