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

#ifndef PA_SCHEDULER_CCEC_PMU_OWNER_HOST_H_
#define PA_SCHEDULER_CCEC_PMU_OWNER_HOST_H_

#include "pmu_owner_control.h"
#include "pmu_owner_main_abi.h"
#include "pmu_owner_main_loader.h"

#include "acl/acl.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace pa_scheduler::pmu_owner {

inline bool OwnerCheckAcl(aclError error, const char *label)
{
    if (error == ACL_SUCCESS) return true;
    std::fprintf(stderr, "ACL error %d: %s\n", static_cast<int>(error), label);
    return false;
}

inline std::string ArtifactBesideKernel(const std::string &kernel_path, const char *name)
{
    const size_t slash = kernel_path.find_last_of('/');
    return slash == std::string::npos ? std::string(name) : kernel_path.substr(0, slash + 1U) + name;
}

inline const char *OwnerFieldName(PmuOwnerField field)
{
    switch (field) {
    case PmuOwnerField::None: return "none";
    case PmuOwnerField::Arguments: return "arguments";
    case PmuOwnerField::ControlMagic: return "control-magic";
    case PmuOwnerField::ControlVersion: return "control-version";
    case PmuOwnerField::ControlSize: return "control-size";
    case PmuOwnerField::State: return "state";
    case PmuOwnerField::RegisterBase: return "register-base";
    case PmuOwnerField::Ctrl0: return "ctrl0";
    case PmuOwnerField::Ctrl1: return "ctrl1";
    case PmuOwnerField::Selector0: return "selector0";
    case PmuOwnerField::Selector1: return "selector1";
    case PmuOwnerField::Selector2: return "selector2";
    case PmuOwnerField::Selector3: return "selector3";
    case PmuOwnerField::Selector4: return "selector4";
    case PmuOwnerField::Selector5: return "selector5";
    case PmuOwnerField::Selector6: return "selector6";
    case PmuOwnerField::Selector7: return "selector7";
    case PmuOwnerField::Selector8: return "selector8";
    case PmuOwnerField::Selector9: return "selector9";
    case PmuOwnerField::StartCycleLow: return "start-cycle-low";
    case PmuOwnerField::StartCycleHigh: return "start-cycle-high";
    case PmuOwnerField::StopCycleLow: return "stop-cycle-low";
    case PmuOwnerField::StopCycleHigh: return "stop-cycle-high";
    case PmuOwnerField::BitmapCount: return "bitmap-count";
    case PmuOwnerField::TotalCount: return "total-count";
    case PmuOwnerField::AicCount: return "aic-count";
    case PmuOwnerField::AivCount: return "aiv-count";
    }
    return "unknown";
}

struct ActiveSubcoreLimits {
    uint32_t aic = 0U;
    uint32_t aiv = 0U;
    uint32_t total = 0U;
};

inline bool QueryActiveSubcoreLimits(aclrtStream scheduling_stream, ActiveSubcoreLimits *limits)
{
    if (scheduling_stream == nullptr || limits == nullptr) {
        std::fprintf(stderr, "Cannot query active PMU subcores with a null stream/result.\n");
        return false;
    }
    uint32_t aic = 0U;
    uint32_t aiv = 0U;
    const aclError aic_error =
        aclrtGetStreamResLimit(scheduling_stream, ACL_RT_DEV_RES_CUBE_CORE, &aic);
    const aclError aiv_error =
        aclrtGetStreamResLimit(scheduling_stream, ACL_RT_DEV_RES_VECTOR_CORE, &aiv);
    const uint64_t total = static_cast<uint64_t>(aic) + aiv;
    if (aic_error != ACL_SUCCESS || aiv_error != ACL_SUCCESS ||
        aic != kExpectedAicCount || aiv != kExpectedAivCount || total != kExpectedSubcoreCount) {
        std::fprintf(
            stderr,
            "Unexpected stream PMU topology: aic_error=%d aiv_error=%d "
            "aic=%u/%u aiv=%u/%u total=%llu/%u\n",
            static_cast<int>(aic_error), static_cast<int>(aiv_error), aic, kExpectedAicCount,
            aiv, kExpectedAivCount, static_cast<unsigned long long>(total), kExpectedSubcoreCount
        );
        return false;
    }
    limits->aic = aic;
    limits->aiv = aiv;
    limits->total = static_cast<uint32_t>(total);
    std::printf(
        "[PMU_OWNER] stream_active aic=%u aiv=%u total=%u physical_slots=%u\n",
        limits->aic, limits->aiv, limits->total, kPhysicalSubcoreCount
    );
    return true;
}

// mixed launch 的一个物理 block 必须同时拥有 1 个 AIC 和相邻的 2 个 AIV。
// 只检查 32/64 总数仍可能放过孤立 AIV；这里直接按两 die 的真实编号布局验闭包。
inline bool ValidateConfiguredTripletTopology(const PmuOwnerControl &control)
{
    uint32_t complete_triplets = 0U;
    uint32_t broken_triplets = 0U;
    const uint32_t dies = kPhysicalSubcoreCount / kSubcoresPerDie;
    for (uint32_t die = 0U; die < dies; ++die) {
        const uint32_t die_base = die * kSubcoresPerDie;
        for (uint32_t local = 0U; local < kAicPerDie; ++local) {
            const bool aic = IsConfigured(control, die_base + local);
            const bool aiv0 = IsConfigured(control, die_base + kAicPerDie + local * 2U);
            const bool aiv1 = IsConfigured(control, die_base + kAicPerDie + local * 2U + 1U);
            if (aic == aiv0 && aic == aiv1) {
                complete_triplets += aic ? 1U : 0U;
            } else {
                ++broken_triplets;
            }
        }
    }
    const bool passed = complete_triplets == kExpectedAicCount && broken_triplets == 0U;
    std::printf(
        "[ASSERT] %-48s %s (complete=%u broken=%u)\n",
        "PMU owner bitmap is complete 1-AIC + 2-AIV triplets",
        passed ? "PASS" : "FAIL", complete_triplets, broken_triplets
    );
    return passed;
}

// owner 命令使用独立 stream，但通过 mode=0 JSON 在主 aicpu_scheduler 中执行。
// Configure 同步完成后才允许启动 AICore；AICore 正常或异常退出后，Restore
// 都不会依赖业务 stream。MMIO 映射必须保持到 Finalize 完成之后。
class PmuOwnerSession {
public:
    PmuOwnerSession() = default;
    PmuOwnerSession(const PmuOwnerSession &) = delete;
    PmuOwnerSession &operator=(const PmuOwnerSession &) = delete;

    ~PmuOwnerSession()
    {
        if (HasResources()) (void)Finalize();
    }

    bool Initialize(
        uint32_t device, aclrtStream scheduling_stream, const std::string &dispatcher_path,
        const std::string &owner_path, const std::vector<uint64_t> &register_bases
    )
    {
        if (HasResources() || register_bases.size() != kPhysicalSubcoreCount) {
            std::fprintf(
                stderr, "Invalid PMU owner initialization state or register table size: %zu\n",
                register_bases.size()
            );
            return false;
        }
        device_ = device;
        if (!QueryActiveSubcoreLimits(scheduling_stream, &limits_)) return false;
        if (!OwnerCheckAcl(aclrtCreateStream(&owner_stream_), "aclrtCreateStream(PMU owner)")) return false;
        if (loader_.Initialize(
                dispatcher_path, owner_path, owner_stream_, static_cast<int32_t>(device_)
            ) != 0) {
            return false;
        }
        if (!OwnerCheckAcl(
                aclrtMalloc(&control_device_, sizeof(PmuOwnerControl), ACL_MEM_MALLOC_NORMAL_ONLY),
                "aclrtMalloc(PMU owner control)"
            )) {
            return false;
        }
        if ((reinterpret_cast<uintptr_t>(control_device_) & (alignof(PmuOwnerControl) - 1U)) != 0U) {
            std::fprintf(stderr, "PMU owner control is not 64-byte aligned: %p\n", control_device_);
            return false;
        }

        control_ = PmuOwnerControl{};
        control_.magic = kPmuOwnerControlMagic;
        control_.version = kPmuOwnerControlVersion;
        control_.struct_bytes = sizeof(PmuOwnerControl);
        control_.status = kStatusPending;
        control_.expected_total = limits_.total;
        control_.expected_aic = limits_.aic;
        control_.expected_aiv = limits_.aiv;
        std::memcpy(control_.register_bases, register_bases.data(), sizeof(control_.register_bases));
        if (!OwnerCheckAcl(
                aclrtMemcpy(
                    control_device_, sizeof(control_), &control_, sizeof(control_), ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "aclrtMemcpy(H2D initial PMU owner control)"
            )) {
            return false;
        }
        ready_ = true;
        return true;
    }

    bool Configure()
    {
        if (!ready_) return false;
        const bool command_ok = RunCommand(PmuOwnerMainCommand::Configure, "Configure");
        configured_ = CountConfigured(control_) != 0U;
        const uint32_t bitmap_count = CountConfigured(control_);
        const bool triplets_ok = ValidateConfiguredTripletTopology(control_);
        const bool state_ok = control_.status == static_cast<int32_t>(PmuOwnerStatus::Success) &&
            control_.configured == 1U && control_.active_total == limits_.total &&
            control_.active_aic == limits_.aic && control_.active_aiv == limits_.aiv &&
            control_.discovered_total == limits_.total && control_.discovered_aic == limits_.aic &&
            control_.discovered_aiv == limits_.aiv && bitmap_count == limits_.total &&
            control_.skipped_total + bitmap_count == kPhysicalSubcoreCount && triplets_ok;
        PrintControl("Configure", bitmap_count);
        return command_ok && state_ok;
    }

    bool Restore()
    {
        if (control_device_ == nullptr || owner_stream_ == nullptr || !loader_.IsInitialized()) {
            return !configured_;
        }
        const bool command_ok = RunCommand(PmuOwnerMainCommand::Restore, "Restore");
        const uint32_t bitmap_count = CountConfigured(control_);
        const bool state_ok = control_.status == static_cast<int32_t>(PmuOwnerStatus::Success) &&
            control_.configured == 0U && control_.active_total == 0U &&
            control_.active_aic == 0U && control_.active_aiv == 0U && bitmap_count == 0U;
        configured_ = bitmap_count != 0U;
        PrintControl("Restore", bitmap_count);
        return command_ok && state_ok;
    }

    bool Finalize()
    {
        bool ok = true;
        if (control_device_ != nullptr && owner_stream_ != nullptr && loader_.IsInitialized()) {
            bool restored = Restore();
            if (!restored) restored = Restore();
            ok &= restored;
        } else {
            ok &= !configured_;
        }
        ok &= loader_.Finalize() == 0;
        if (control_device_ != nullptr) {
            ok &= OwnerCheckAcl(aclrtFree(control_device_), "aclrtFree(PMU owner control)");
            control_device_ = nullptr;
        }
        if (owner_stream_ != nullptr) {
            ok &= OwnerCheckAcl(aclrtDestroyStream(owner_stream_), "aclrtDestroyStream(PMU owner)");
            owner_stream_ = nullptr;
        }
        ready_ = false;
        configured_ = false;
        std::printf("[PMU_OWNER] restore_and_cleanup=%s\n", ok ? "PASS" : "FAIL");
        return ok;
    }

    uint64_t RegisterTableDeviceAddress() const
    {
        if (control_device_ == nullptr) return 0U;
        return reinterpret_cast<uint64_t>(control_device_) + offsetof(PmuOwnerControl, register_bases);
    }

    const PmuOwnerControl &Control() const { return control_; }

    bool IsConfiguredSubcore(uint32_t index) const
    {
        return ready_ && IsConfigured(control_, index);
    }

private:
    bool HasResources() const
    {
        return owner_stream_ != nullptr || control_device_ != nullptr || loader_.IsInitialized();
    }

    bool RunCommand(PmuOwnerMainCommand command, const char *label)
    {
        const PmuOwnerMainKernelArgs arguments = MakePmuOwnerMainKernelArgs(
            reinterpret_cast<uint64_t>(control_device_), command, device_
        );
        const std::string sync_label = std::string("aclrtSynchronizeStream(PMU ") + label + ")";
        const std::string copy_label = std::string("aclrtMemcpy(D2H PMU ") + label + ")";
        if (loader_.Launch(owner_stream_, const_cast<PmuOwnerMainKernelArgs *>(&arguments), sizeof(arguments)) != 0 ||
            !OwnerCheckAcl(aclrtSynchronizeStream(owner_stream_), sync_label.c_str()) ||
            !OwnerCheckAcl(
                aclrtMemcpy(
                    &control_, sizeof(control_), control_device_, sizeof(control_), ACL_MEMCPY_DEVICE_TO_HOST
                ),
                copy_label.c_str()
            )) {
            return false;
        }
        return true;
    }

    void PrintControl(const char *command, uint32_t bitmap_count) const
    {
        const auto failed_field = static_cast<PmuOwnerField>(control_.first_failed_field);
        const auto restore_field = static_cast<PmuOwnerField>(control_.first_restore_failed_field);
        std::printf(
            "[PMU_OWNER] command=%s status=%d configured=%u active=%u/%u/%u "
            "discovered=%u/%u/%u bitmap=%u skipped=%u first_failed=%u:%s(%u):0x%x/0x%x "
            "restore_failures=%u first_restore=%u:%s(%u):0x%x/0x%x\n",
            command, static_cast<int>(control_.status), control_.configured,
            control_.active_total, control_.active_aic, control_.active_aiv,
            control_.discovered_total, control_.discovered_aic, control_.discovered_aiv,
            bitmap_count, control_.skipped_total, control_.first_failed_index,
            OwnerFieldName(failed_field), control_.first_failed_field,
            control_.first_failed_observed, control_.first_failed_expected,
            control_.restore_failures, control_.first_restore_failed_index,
            OwnerFieldName(restore_field), control_.first_restore_failed_field,
            control_.first_restore_failed_observed, control_.first_restore_failed_expected
        );
        std::printf(
            "[PMU_OWNER] configured_bitmap=%08x:%08x:%08x:%08x\n",
            control_.configured_bitmap[3], control_.configured_bitmap[2],
            control_.configured_bitmap[1], control_.configured_bitmap[0]
        );
    }

    uint32_t device_ = 0U;
    aclrtStream owner_stream_ = nullptr;
    MainAicpuLoader loader_;
    void *control_device_ = nullptr;
    PmuOwnerControl control_{};
    ActiveSubcoreLimits limits_{};
    bool ready_ = false;
    bool configured_ = false;
};

}  // namespace pa_scheduler::pmu_owner

#endif  // PA_SCHEDULER_CCEC_PMU_OWNER_HOST_H_
