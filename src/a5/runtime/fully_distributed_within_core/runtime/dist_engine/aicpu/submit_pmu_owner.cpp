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

#include "dist_engine/aicpu/submit_pmu_owner.h"

#include <cstddef>
#include <cstdint>

#include "aicpu/platform_regs.h"
#include "common/core_type.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "runtime.h"

namespace {

constexpr uint32_t kCounterCount = 10U;
constexpr uint32_t kPhysicalSubcoresPerDie = 54U;
constexpr uint32_t kAicPerDie = 18U;
constexpr uint32_t kDies = kFdwicSubmitPmuPhysicalSubcores / kPhysicalSubcoresPerDie;
constexpr uint32_t kUnsetCore = UINT32_MAX;

// submit-PMU 保留 CNT6/CNT7 作为不在窗口中途读取的权威计数，并用
// CNT8/CNT5 复制同一 request/miss 事件。none 要求两组逐核精确相等；
// running phase 则重建 shadow whole 并要求不超过 primary。
constexpr uint32_t kConfiguredSelectors[kCounterCount] = {
    0x501U,  // CNT0: vector busy
    0x301U,  // CNT1: cube busy
    0x001U,  // CNT2: scalar busy
    0x701U,  // CNT3: MTE1 busy
    0x202U,  // CNT4: MTE2 busy
    0x035U,  // CNT5: shadow I-cache miss
    0x034U,  // CNT6: primary I-cache request
    0x035U,  // CNT7: primary I-cache miss
    0x034U,  // CNT8: shadow I-cache request
    0x000U,  // CNT9: DAV3510 正式未使用；本机已反证不能承载 miss
};

static_assert(kConfiguredSelectors[2] == kFdwicSubmitPmuCnt2ScalarBusy, "CNT2 selector contract changed");
static_assert(kConfiguredSelectors[5] == kFdwicSubmitPmuCnt5ShadowIcacheMiss, "CNT5 selector contract changed");
static_assert(kConfiguredSelectors[6] == kFdwicSubmitPmuCnt6IcacheRequest, "CNT6 selector contract changed");
static_assert(kConfiguredSelectors[7] == kFdwicSubmitPmuCnt7IcacheMiss, "CNT7 selector contract changed");
static_assert(kConfiguredSelectors[8] == kFdwicSubmitPmuCnt8ShadowIcacheRequest, "CNT8 selector contract changed");
static_assert(kFdwicSubmitPmuPhysicalSubcores == kDies * kPhysicalSubcoresPerDie, "physical topology changed");

// owner 改写的寄存器原值恰好占一条 cacheline。事件 counter 和 TOTAL 是
// read-to-clear，旧计数无法保存，也不属于可恢复配置。
struct alignas(64) SavedRegisters {
    uint32_t ctrl0;
    uint32_t ctrl1;
    uint32_t selectors[kCounterCount];
    uint32_t start_low;
    uint32_t start_high;
    uint32_t stop_low;
    uint32_t stop_high;
};

static_assert(sizeof(SavedRegisters) == 64U, "saved PMU configuration must occupy one cacheline");

// 该状态只存在于本次 AICPU runtime SO 内，不属于 host/device ABI。saved[]
// 按物理子核编号索引；owned_bitmap 是唯一所有权真相，配置或恢复失败后
// 不得覆盖对应 saved 槽。
struct OwnerState {
    SavedRegisters saved[kFdwicSubmitPmuPhysicalSubcores];
    uint32_t owned_bitmap[kFdwicSubmitPmuBitmapWords];
    uint32_t configured_bitmap[kFdwicSubmitPmuBitmapWords];
    uint32_t configured_count;
    uint32_t configured_aic;
    uint32_t configured_aiv;
    uint32_t restored_count;
    uint32_t restore_failures;
    bool configured;
};

OwnerState g_owner;

inline void FullSystemBarrier() {
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    // submit-pmu 只允许真实 A5 使用；该分支仅保证同一源码可参与 a5sim 构建。
    __asm__ volatile("" ::: "memory");
#endif
}

inline bool IsAicPhysicalId(uint32_t physical_id) {
    return physical_id < kFdwicSubmitPmuPhysicalSubcores && (physical_id % kPhysicalSubcoresPerDie) < kAicPerDie;
}

inline bool BitmapContains(const uint32_t *bitmap, uint32_t physical_id) {
    return physical_id < kFdwicSubmitPmuPhysicalSubcores &&
           (bitmap[physical_id / 32U] & (1U << (physical_id % 32U))) != 0U;
}

inline void BitmapSet(uint32_t *bitmap, uint32_t physical_id) {
    bitmap[physical_id / 32U] |= 1U << (physical_id % 32U);
}

inline void BitmapClear(uint32_t *bitmap, uint32_t physical_id) {
    bitmap[physical_id / 32U] &= ~(1U << (physical_id % 32U));
}

uint32_t BitmapCount(const uint32_t *bitmap) {
    uint32_t count = 0U;
    for (uint32_t physical_id = 0; physical_id < kFdwicSubmitPmuPhysicalSubcores; ++physical_id) {
        count += BitmapContains(bitmap, physical_id) ? 1U : 0U;
    }
    return count;
}

constexpr FdwicSubmitPmuOwnerField SelectorField(uint32_t counter) {
    return static_cast<FdwicSubmitPmuOwnerField>(static_cast<uint32_t>(FdwicSubmitPmuOwnerField::Selector0) + counter);
}

void FlushOwnerState(FdwicSubmitPmuHeader *header) {
    if (header == nullptr) return;
    cache_flush_range(
        const_cast<const uint32_t *>(&header->owner_status),
        offsetof(FdwicSubmitPmuHeader, cores) - offsetof(FdwicSubmitPmuHeader, owner_status)
    );
}

void ResetHeaderOwnerState(FdwicSubmitPmuHeader *header) {
    header->owner_status = kFdwicSubmitPmuOwnerRequested;
    header->configured_count = 0U;
    header->restored_count = 0U;
    header->configured_aic = 0U;
    header->configured_aiv = 0U;
    header->complete_mixed_triplets = 0U;
    header->restore_failures = 0U;
    header->active_after_restore = 0U;
    for (uint32_t word = 0; word < kFdwicSubmitPmuBitmapWords; ++word) {
        header->configured_bitmap_words[word] = 0U;
    }
    header->first_failure_core = kUnsetCore;
    header->first_failure_field = static_cast<uint32_t>(FdwicSubmitPmuOwnerField::None);
    header->first_failure_observed = 0U;
    header->first_failure_expected = 0U;
    FlushOwnerState(header);
}

void RecordFirstFailure(
    FdwicSubmitPmuHeader *header, uint32_t physical_id, FdwicSubmitPmuOwnerField field, uint32_t observed,
    uint32_t expected
) {
    if (header == nullptr || header->first_failure_core != kUnsetCore) return;
    header->first_failure_core = physical_id;
    header->first_failure_field = static_cast<uint32_t>(field);
    header->first_failure_observed = observed;
    header->first_failure_expected = expected;
}

void PublishOwnerProgress(FdwicSubmitPmuHeader *header) {
    if (header == nullptr) return;
    header->configured_count = g_owner.configured_count;
    header->configured_aic = g_owner.configured_aic;
    header->configured_aiv = g_owner.configured_aiv;
    header->restored_count = g_owner.restored_count;
    header->restore_failures = g_owner.restore_failures;
    header->active_after_restore = BitmapCount(g_owner.owned_bitmap);
    for (uint32_t word = 0; word < kFdwicSubmitPmuBitmapWords; ++word) {
        // 这是本次成功配置集合的永久快照，不是恢复过程中逐步清零的
        // ownership bitmap；host 在 Restore 后仍需据此核对96条worker记录。
        header->configured_bitmap_words[word] = g_owner.configured_bitmap[word];
    }
    FlushOwnerState(header);
}

void ResetOwnerState() {
    for (uint32_t word = 0; word < kFdwicSubmitPmuBitmapWords; ++word) {
        g_owner.owned_bitmap[word] = 0U;
        g_owner.configured_bitmap[word] = 0U;
    }
    g_owner.configured_count = 0U;
    g_owner.configured_aic = 0U;
    g_owner.configured_aiv = 0U;
    g_owner.restored_count = 0U;
    g_owner.restore_failures = 0U;
    g_owner.configured = false;
}

void SaveOne(uint64_t reg_base, uint32_t physical_id) {
    SavedRegisters &saved = g_owner.saved[physical_id];
    saved.ctrl0 = static_cast<uint32_t>(read_reg(reg_base, RegId::PMU_CTRL_0));
    saved.ctrl1 = static_cast<uint32_t>(read_reg(reg_base, RegId::PMU_CTRL_1));
    for (uint32_t counter = 0; counter < kCounterCount; ++counter) {
        saved.selectors[counter] = static_cast<uint32_t>(read_reg(reg_base, reg_index(RegId::PMU_CNT0_IDX, counter)));
    }
    saved.start_low = static_cast<uint32_t>(read_reg(reg_base, RegId::PMU_START_CYC0));
    saved.start_high = static_cast<uint32_t>(read_reg(reg_base, RegId::PMU_START_CYC1));
    saved.stop_low = static_cast<uint32_t>(read_reg(reg_base, RegId::PMU_STOP_CYC0));
    saved.stop_high = static_cast<uint32_t>(read_reg(reg_base, RegId::PMU_STOP_CYC1));
}

void ConfigureOne(uint64_t reg_base) {
    // 所有写之前，调用方已经保存原值并置 ownership bit。DSB 只存在于
    // AICPU owner 冷路径，不进入任何 Submit 或 AICore scalar 观察窗口。
    write_reg(reg_base, RegId::PMU_CTRL_0, 0U);
    write_reg(reg_base, RegId::PMU_CTRL_1, 0U);
    for (uint32_t counter = 0; counter < kCounterCount; ++counter) {
        write_reg(reg_base, reg_index(RegId::PMU_CNT0_IDX, counter), kConfiguredSelectors[counter]);
    }

    // DAV3510 的事件 counter 和 64-bit TOTAL 都是 read-to-clear。此处清除
    // owner 配置前的残值；旧计数无法也不应伪装成可恢复状态。
    for (uint32_t counter = 0; counter < kCounterCount; ++counter) {
        (void)read_reg(reg_base, reg_index(RegId::PMU_CNT0, counter));
    }
    (void)read_reg(reg_base, RegId::PMU_CNT_TOTAL0);
    (void)read_reg(reg_base, RegId::PMU_CNT_TOTAL1);

    write_reg(reg_base, RegId::PMU_START_CYC0, 0U);
    write_reg(reg_base, RegId::PMU_START_CYC1, 0U);
    write_reg(reg_base, RegId::PMU_STOP_CYC0, UINT32_MAX);
    write_reg(reg_base, RegId::PMU_STOP_CYC1, UINT32_MAX);
    write_reg(reg_base, RegId::PMU_CTRL_0, REG_MMIO_PMU_CTRL_0_ENABLE_VAL);
    write_reg(reg_base, RegId::PMU_CTRL_1, REG_MMIO_PMU_CTRL_1_ENABLE_VAL);
    FullSystemBarrier();
}

bool CheckRegister(
    uint64_t reg_base, RegId reg, uint32_t expected, FdwicSubmitPmuOwnerField field,
    FdwicSubmitPmuOwnerField *failed_field, uint32_t *observed
) {
    const uint32_t actual = static_cast<uint32_t>(read_reg(reg_base, reg));
    if (actual == expected) return true;
    *failed_field = field;
    *observed = actual;
    return false;
}

bool ConfigurationMatches(
    uint64_t reg_base, FdwicSubmitPmuOwnerField *failed_field, uint32_t *observed, uint32_t *expected
) {
    if (!CheckRegister(
            reg_base, RegId::PMU_CTRL_0, REG_MMIO_PMU_CTRL_0_ENABLE_VAL, FdwicSubmitPmuOwnerField::Ctrl0, failed_field,
            observed
        )) {
        *expected = REG_MMIO_PMU_CTRL_0_ENABLE_VAL;
        return false;
    }
    if (!CheckRegister(
            reg_base, RegId::PMU_CTRL_1, REG_MMIO_PMU_CTRL_1_ENABLE_VAL, FdwicSubmitPmuOwnerField::Ctrl1, failed_field,
            observed
        )) {
        *expected = REG_MMIO_PMU_CTRL_1_ENABLE_VAL;
        return false;
    }
    for (uint32_t counter = 0; counter < kCounterCount; ++counter) {
        if (!CheckRegister(
                reg_base, reg_index(RegId::PMU_CNT0_IDX, counter), kConfiguredSelectors[counter],
                SelectorField(counter), failed_field, observed
            )) {
            *expected = kConfiguredSelectors[counter];
            return false;
        }
    }
    if (!CheckRegister(
            reg_base, RegId::PMU_START_CYC0, 0U, FdwicSubmitPmuOwnerField::StartLow, failed_field, observed
        )) {
        *expected = 0U;
        return false;
    }
    if (!CheckRegister(
            reg_base, RegId::PMU_START_CYC1, 0U, FdwicSubmitPmuOwnerField::StartHigh, failed_field, observed
        )) {
        *expected = 0U;
        return false;
    }
    if (!CheckRegister(
            reg_base, RegId::PMU_STOP_CYC0, UINT32_MAX, FdwicSubmitPmuOwnerField::StopLow, failed_field, observed
        )) {
        *expected = UINT32_MAX;
        return false;
    }
    if (!CheckRegister(
            reg_base, RegId::PMU_STOP_CYC1, UINT32_MAX, FdwicSubmitPmuOwnerField::StopHigh, failed_field, observed
        )) {
        *expected = UINT32_MAX;
        return false;
    }
    return true;
}

void WriteSavedConfiguration(uint64_t reg_base, uint32_t physical_id) {
    const SavedRegisters &saved = g_owner.saved[physical_id];
    write_reg(reg_base, RegId::PMU_CTRL_0, 0U);
    write_reg(reg_base, RegId::PMU_CTRL_1, 0U);
    for (uint32_t counter = 0; counter < kCounterCount; ++counter) {
        write_reg(reg_base, reg_index(RegId::PMU_CNT0_IDX, counter), saved.selectors[counter]);
    }
    write_reg(reg_base, RegId::PMU_START_CYC0, saved.start_low);
    write_reg(reg_base, RegId::PMU_START_CYC1, saved.start_high);
    write_reg(reg_base, RegId::PMU_STOP_CYC0, saved.stop_low);
    write_reg(reg_base, RegId::PMU_STOP_CYC1, saved.stop_high);
    // CTRL 最后恢复，避免 selector/range 尚未恢复时短暂重新开启旧配置。
    write_reg(reg_base, RegId::PMU_CTRL_0, saved.ctrl0);
    write_reg(reg_base, RegId::PMU_CTRL_1, saved.ctrl1);
    FullSystemBarrier();
}

bool SavedConfigurationMatches(
    uint64_t reg_base, uint32_t physical_id, FdwicSubmitPmuOwnerField *failed_field, uint32_t *observed,
    uint32_t *expected
) {
    const SavedRegisters &saved = g_owner.saved[physical_id];
    if (!CheckRegister(
            reg_base, RegId::PMU_CTRL_0, saved.ctrl0, FdwicSubmitPmuOwnerField::Ctrl0, failed_field, observed
        )) {
        *expected = saved.ctrl0;
        return false;
    }
    if (!CheckRegister(
            reg_base, RegId::PMU_CTRL_1, saved.ctrl1, FdwicSubmitPmuOwnerField::Ctrl1, failed_field, observed
        )) {
        *expected = saved.ctrl1;
        return false;
    }
    for (uint32_t counter = 0; counter < kCounterCount; ++counter) {
        if (!CheckRegister(
                reg_base, reg_index(RegId::PMU_CNT0_IDX, counter), saved.selectors[counter], SelectorField(counter),
                failed_field, observed
            )) {
            *expected = saved.selectors[counter];
            return false;
        }
    }
    if (!CheckRegister(
            reg_base, RegId::PMU_START_CYC0, saved.start_low, FdwicSubmitPmuOwnerField::StartLow, failed_field, observed
        )) {
        *expected = saved.start_low;
        return false;
    }
    if (!CheckRegister(
            reg_base, RegId::PMU_START_CYC1, saved.start_high, FdwicSubmitPmuOwnerField::StartHigh, failed_field,
            observed
        )) {
        *expected = saved.start_high;
        return false;
    }
    if (!CheckRegister(
            reg_base, RegId::PMU_STOP_CYC0, saved.stop_low, FdwicSubmitPmuOwnerField::StopLow, failed_field, observed
        )) {
        *expected = saved.stop_low;
        return false;
    }
    if (!CheckRegister(
            reg_base, RegId::PMU_STOP_CYC1, saved.stop_high, FdwicSubmitPmuOwnerField::StopHigh, failed_field, observed
        )) {
        *expected = saved.stop_high;
        return false;
    }
    return true;
}

bool RestoreOne(uint64_t *register_bases, uint32_t physical_id, FdwicSubmitPmuHeader *header) {
    const uint64_t reg_base = register_bases == nullptr ? 0U : register_bases[physical_id];
    if (reg_base == 0U) {
        RecordFirstFailure(header, physical_id, FdwicSubmitPmuOwnerField::RegisterBase, 0U, 1U);
        ++g_owner.restore_failures;
        return false;
    }

    WriteSavedConfiguration(reg_base, physical_id);
    FdwicSubmitPmuOwnerField failed_field = FdwicSubmitPmuOwnerField::None;
    uint32_t observed = 0U;
    uint32_t expected = 0U;
    if (!SavedConfigurationMatches(reg_base, physical_id, &failed_field, &observed, &expected)) {
        RecordFirstFailure(header, physical_id, failed_field, observed, expected);
        ++g_owner.restore_failures;
        return false;
    }
    return true;
}

bool RestoreOwned(FdwicSubmitPmuHeader *header) {
    uint64_t *register_bases = reinterpret_cast<uint64_t *>(get_platform_regs());
    bool all_restored = true;
    for (uint32_t next = kFdwicSubmitPmuPhysicalSubcores; next != 0U; --next) {
        const uint32_t physical_id = next - 1U;
        if (!BitmapContains(g_owner.owned_bitmap, physical_id)) continue;
        if (RestoreOne(register_bases, physical_id, header)) {
            BitmapClear(g_owner.owned_bitmap, physical_id);
            ++g_owner.restored_count;
        } else {
            all_restored = false;
        }
    }
    return all_restored && BitmapCount(g_owner.owned_bitmap) == 0U;
}

bool HeaderConfigurationMatches(const FdwicSubmitPmuHeader &header) {
    const uint32_t expected_selectors[5] = {
        kFdwicSubmitPmuCnt2ScalarBusy, kFdwicSubmitPmuCnt5ShadowIcacheMiss,    kFdwicSubmitPmuCnt6IcacheRequest,
        kFdwicSubmitPmuCnt7IcacheMiss, kFdwicSubmitPmuCnt8ShadowIcacheRequest,
    };
    const bool mode_valid = header.mode == kFdwicSubmitPmuModeNone || header.mode == kFdwicSubmitPmuModeArgBuild;
    if (header.magic != kFdwicSubmitPmuMagic || header.version != kFdwicSubmitPmuVersion || !mode_valid ||
        header.header_bytes != fdwic_submit_pmu_bytes_for_mode(header.mode) ||
        header.record_bytes != sizeof(FdwicSubmitPmuCoreData) || header.num_cores != kFdwicSubmitPmuExpectedCores ||
        header.expected_aic != kFdwicSubmitPmuExpectedAic || header.expected_aiv != kFdwicSubmitPmuExpectedAiv ||
        header.sys_cnt_freq_hz != PLATFORM_PROF_SYS_CNT_FREQ) {
        return false;
    }
    for (uint32_t index = 0; index < 5U; ++index) {
        if (header.selectors[index] != expected_selectors[index]) return false;
    }
    return true;
}

bool ValidateActiveTopology(
    Runtime *runtime, uint32_t *active_bitmap, uint32_t *aic_count, uint32_t *aiv_count, uint32_t *triplet_count,
    uint32_t *failed_core, uint32_t *observed, uint32_t *expected
) {
    if (runtime == nullptr || runtime->worker_count != static_cast<int>(kFdwicSubmitPmuExpectedCores)) {
        *failed_core = kUnsetCore;
        *observed = runtime == nullptr ? 0U : static_cast<uint32_t>(runtime->worker_count);
        *expected = kFdwicSubmitPmuExpectedCores;
        return false;
    }

    uint64_t *register_bases = reinterpret_cast<uint64_t *>(get_platform_regs());
    if (register_bases == nullptr) {
        *failed_core = kUnsetCore;
        *observed = 0U;
        *expected = 1U;
        return false;
    }

    *aic_count = 0U;
    *aiv_count = 0U;
    *triplet_count = 0U;
    for (uint32_t word = 0; word < kFdwicSubmitPmuBitmapWords; ++word)
        active_bitmap[word] = 0U;

    for (uint32_t logical_id = 0; logical_id < kFdwicSubmitPmuExpectedCores; ++logical_id) {
        const Handshake &worker = runtime->workers[logical_id];
        const uint32_t physical_id = worker.physical_core_id;
        const CoreType role = worker.core_type;
        if (physical_id >= kFdwicSubmitPmuPhysicalSubcores || BitmapContains(active_bitmap, physical_id)) {
            *failed_core = physical_id;
            *observed = physical_id >= kFdwicSubmitPmuPhysicalSubcores ? physical_id : 2U;
            *expected = physical_id >= kFdwicSubmitPmuPhysicalSubcores ? kFdwicSubmitPmuPhysicalSubcores - 1U : 1U;
            return false;
        }
        if (register_bases[physical_id] == 0U) {
            *failed_core = physical_id;
            *observed = 0U;
            *expected = 1U;
            return false;
        }

        const bool physical_aic = IsAicPhysicalId(physical_id);
        const bool reported_aic = role == CoreType::AIC;
        const bool role_valid = reported_aic || role == CoreType::AIV;
        if (!role_valid || physical_aic != reported_aic) {
            *failed_core = physical_id;
            *observed = static_cast<uint32_t>(role);
            *expected = static_cast<uint32_t>(physical_aic ? CoreType::AIC : CoreType::AIV);
            return false;
        }
        BitmapSet(active_bitmap, physical_id);
        *aic_count += reported_aic ? 1U : 0U;
        *aiv_count += reported_aic ? 0U : 1U;
    }

    if (*aic_count != kFdwicSubmitPmuExpectedAic || *aiv_count != kFdwicSubmitPmuExpectedAiv) {
        *failed_core = kUnsetCore;
        *observed = (*aic_count << 16U) | *aiv_count;
        *expected = (kFdwicSubmitPmuExpectedAic << 16U) | kFdwicSubmitPmuExpectedAiv;
        return false;
    }

    uint32_t broken_triplets = 0U;
    for (uint32_t die = 0; die < kDies; ++die) {
        const uint32_t die_base = die * kPhysicalSubcoresPerDie;
        for (uint32_t local = 0; local < kAicPerDie; ++local) {
            const bool aic = BitmapContains(active_bitmap, die_base + local);
            const bool aiv0 = BitmapContains(active_bitmap, die_base + kAicPerDie + local * 2U);
            const bool aiv1 = BitmapContains(active_bitmap, die_base + kAicPerDie + local * 2U + 1U);
            if (aic == aiv0 && aic == aiv1) {
                *triplet_count += aic ? 1U : 0U;
            } else {
                ++broken_triplets;
            }
        }
    }
    if (*triplet_count != kFdwicSubmitPmuExpectedAic || broken_triplets != 0U) {
        *failed_core = kUnsetCore;
        *observed = (*triplet_count << 16U) | broken_triplets;
        *expected = kFdwicSubmitPmuExpectedAic << 16U;
        return false;
    }
    return true;
}

}  // namespace

bool fdwic_submit_pmu_requested(Runtime *runtime, FdwicSubmitPmuHeader **header_out) {
    if (header_out != nullptr) *header_out = nullptr;
    if (runtime == nullptr || runtime->dist.swimlane_base == 0U || runtime->dist.swimlane_level != 0U ||
        runtime->dist.swimlane_records_per_core != 0U) {
        return false;
    }

    auto *header = reinterpret_cast<FdwicSubmitPmuHeader *>(runtime->dist.swimlane_base);
    if ((reinterpret_cast<uintptr_t>(header) & (alignof(FdwicSubmitPmuHeader) - 1U)) != 0U) return false;
    // Host H2D 不维护 AICPU L1。这里只失效 host 配置与 owner 两条 cacheline，
    // 不触碰随后由96个AICore独占写入的结果区。
    cache_invalidate_range(header, offsetof(FdwicSubmitPmuHeader, cores));
    if (!HeaderConfigurationMatches(*header)) return false;
    if (header_out != nullptr) *header_out = header;
    return true;
}

int fdwic_submit_pmu_owner_configure(Runtime *runtime, FdwicSubmitPmuHeader *header) {
    if (runtime == nullptr || header == nullptr) return -1;

    ResetHeaderOwnerState(header);

    // 上一次 Restore 若留下 bit，saved[] 仍是唯一原值，绝不能被新一轮
    // Configure 覆盖。先按当前完整 MMIO 表幂等重试；失败则拒绝新会话。
    const bool retried_stale_owner = BitmapCount(g_owner.owned_bitmap) != 0U;
    if (retried_stale_owner) {
        if (!RestoreOwned(header)) {
            header->owner_status =
                kFdwicSubmitPmuOwnerRequested | kFdwicSubmitPmuOwnerRestoreAttempted | kFdwicSubmitPmuOwnerAborted;
            PublishOwnerProgress(header);
            return -1;
        }
    }

    ResetOwnerState();
    // stale owner 已恢复时，上一会话的恢复计数与配置历史不能混进本轮。
    if (retried_stale_owner) ResetHeaderOwnerState(header);

    uint32_t active_bitmap[kFdwicSubmitPmuBitmapWords] = {};
    uint32_t aic_count = 0U;
    uint32_t aiv_count = 0U;
    uint32_t triplet_count = 0U;
    uint32_t failed_core = kUnsetCore;
    uint32_t observed = 0U;
    uint32_t expected = 0U;
    if (!ValidateActiveTopology(
            runtime, active_bitmap, &aic_count, &aiv_count, &triplet_count, &failed_core, &observed, &expected
        )) {
        RecordFirstFailure(header, failed_core, FdwicSubmitPmuOwnerField::Topology, observed, expected);
        header->owner_status |= kFdwicSubmitPmuOwnerAborted;
        FlushOwnerState(header);
        return -1;
    }
    header->owner_status |= kFdwicSubmitPmuOwnerTopologyValid;
    header->complete_mixed_triplets = triplet_count;
    FlushOwnerState(header);

    uint64_t *register_bases = reinterpret_cast<uint64_t *>(get_platform_regs());
    for (uint32_t physical_id = 0; physical_id < kFdwicSubmitPmuPhysicalSubcores; ++physical_id) {
        if (!BitmapContains(active_bitmap, physical_id)) continue;
        const uint64_t reg_base = register_bases[physical_id];

        SaveOne(reg_base, physical_id);
        // ownership bit 必须先于本槽第一条 MMIO 写。即使配置和当场恢复都
        // 失败，该 bit 仍保留，使后续 Restore 可以继续使用未被覆盖的 saved[]。
        BitmapSet(g_owner.owned_bitmap, physical_id);
        ConfigureOne(reg_base);

        FdwicSubmitPmuOwnerField failed_field = FdwicSubmitPmuOwnerField::None;
        observed = 0U;
        expected = 0U;
        if (!ConfigurationMatches(reg_base, &failed_field, &observed, &expected)) {
            RecordFirstFailure(header, physical_id, failed_field, observed, expected);
            const bool rollback_ok = RestoreOwned(header);
            header->owner_status |= kFdwicSubmitPmuOwnerRestoreAttempted | kFdwicSubmitPmuOwnerAborted;
            if (rollback_ok) header->owner_status |= kFdwicSubmitPmuOwnerRestored;
            PublishOwnerProgress(header);
            return -1;
        }

        BitmapSet(g_owner.configured_bitmap, physical_id);
        ++g_owner.configured_count;
        if (IsAicPhysicalId(physical_id)) {
            ++g_owner.configured_aic;
        } else {
            ++g_owner.configured_aiv;
        }
    }

    const bool counts_match = g_owner.configured_count == kFdwicSubmitPmuExpectedCores &&
                              g_owner.configured_aic == kFdwicSubmitPmuExpectedAic &&
                              g_owner.configured_aiv == kFdwicSubmitPmuExpectedAiv &&
                              BitmapCount(g_owner.owned_bitmap) == kFdwicSubmitPmuExpectedCores;
    if (!counts_match) {
        RecordFirstFailure(
            header, kUnsetCore, FdwicSubmitPmuOwnerField::State, g_owner.configured_count, kFdwicSubmitPmuExpectedCores
        );
        const bool rollback_ok = RestoreOwned(header);
        header->owner_status |= kFdwicSubmitPmuOwnerRestoreAttempted | kFdwicSubmitPmuOwnerAborted;
        if (rollback_ok) header->owner_status |= kFdwicSubmitPmuOwnerRestored;
        PublishOwnerProgress(header);
        return -1;
    }

    g_owner.configured = true;
    header->owner_status |= kFdwicSubmitPmuOwnerConfigured | kFdwicSubmitPmuOwnerConfigReadbackValid;
    PublishOwnerProgress(header);
    LOG_INFO_V0(
        "FDWIC submit-PMU owner configured %u physical subcores (%u AIC + %u AIV)", g_owner.configured_count,
        g_owner.configured_aic, g_owner.configured_aiv
    );
    return 0;
}

int fdwic_submit_pmu_owner_restore(FdwicSubmitPmuHeader *header) {
    if (header != nullptr) header->owner_status |= kFdwicSubmitPmuOwnerRestoreAttempted;

    // bitmap 为空时仍把 Restore 视为幂等成功；configured_count/bitmap 快照
    // 保留给 host 做 worker membership 校验。
    const bool restored = RestoreOwned(header);
    if (restored) {
        g_owner.configured = false;
        if (header != nullptr) header->owner_status |= kFdwicSubmitPmuOwnerRestored;
    } else if (header != nullptr) {
        header->owner_status |= kFdwicSubmitPmuOwnerAborted;
    }
    PublishOwnerProgress(header);

    if (!restored) {
        LOG_ERROR(
            "FDWIC submit-PMU owner restore incomplete: active=%u failures=%u", BitmapCount(g_owner.owned_bitmap),
            g_owner.restore_failures
        );
        return -1;
    }
    LOG_INFO_V0("FDWIC submit-PMU owner restored %u physical subcores", g_owner.restored_count);
    return 0;
}
