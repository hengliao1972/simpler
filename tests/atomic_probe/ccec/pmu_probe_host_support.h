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

#ifndef TESTS_ATOMIC_PROBE_CCEC_PMU_PROBE_HOST_SUPPORT_H_
#define TESTS_ATOMIC_PROBE_CCEC_PMU_PROBE_HOST_SUPPORT_H_

#include "pmu_probe_control.h"

#include "acl/acl.h"
#include "aicpu_loader/host/load_aicpu_op.h"
#include "common/kernel_args.h"
#include "driver/ascend_hal.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include <vector>

namespace atomic_probe::pmu {

inline bool CheckAcl(aclError error, const char *label)
{
    if (error == ACL_SUCCESS) return true;
    std::fprintf(stderr, "ACL error %d: %s\n", static_cast<int>(error), label);
    return false;
}

inline std::vector<char> ReadBinary(const std::string &path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0) return {};
    std::vector<char> bytes(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(bytes.data(), size)) return {};
    return bytes;
}

inline std::string ArtifactBesideKernel(const std::string &kernel_path, const char *name)
{
    const size_t slash = kernel_path.find_last_of('/');
    return slash == std::string::npos ? std::string(name) : kernel_path.substr(0, slash + 1) + name;
}

struct ActiveSubcoreLimits {
    uint32_t cube = 0;
    uint32_t vector = 0;
    uint32_t total = 0;
};

// PMU MMIO 表覆盖芯片的 108 个物理槽，但当前 stream 只会调度 ACL
// resource limit 报告的活跃 cube/vector 子核。A5 当前典型值为 32+64=96；
// 不把 96 写死，以免把 stream 分区变化误报成寄存器配置失败。
inline bool QueryActiveSubcoreLimits(aclrtStream stream, ActiveSubcoreLimits *limits)
{
    if (stream == nullptr || limits == nullptr) {
        std::fprintf(stderr, "Cannot query PMU active subcores with a null stream/result.\n");
        return false;
    }
    uint32_t cube = 0;
    uint32_t vector = 0;
    const aclError cube_error = aclrtGetStreamResLimit(stream, ACL_RT_DEV_RES_CUBE_CORE, &cube);
    const aclError vector_error = aclrtGetStreamResLimit(stream, ACL_RT_DEV_RES_VECTOR_CORE, &vector);
    const uint64_t total = static_cast<uint64_t>(cube) + vector;
    constexpr uint32_t kMaximumCubeSubcores = kPmuPhysicalSubcores / 3U;
    constexpr uint32_t kMaximumVectorSubcores = kPmuPhysicalSubcores - kMaximumCubeSubcores;
    if (cube_error != ACL_SUCCESS || vector_error != ACL_SUCCESS || cube == 0 || vector == 0 ||
        cube > kMaximumCubeSubcores || vector > kMaximumVectorSubcores || total > kPmuPhysicalSubcores) {
        std::fprintf(
            stderr,
            "Cannot determine stream PMU subcores: cube_error=%d vector_error=%d "
            "cube=%u vector=%u total=%llu physical_slots=%u\n",
            static_cast<int>(cube_error), static_cast<int>(vector_error), cube, vector,
            static_cast<unsigned long long>(total), kPmuPhysicalSubcores
        );
        return false;
    }
    limits->cube = cube;
    limits->vector = vector;
    limits->total = static_cast<uint32_t>(total);
    std::printf(
        "[PMU_OWNER] stream_active cube=%u vector=%u total=%u physical_slots=%u\n",
        limits->cube, limits->vector, limits->total, kPmuPhysicalSubcores
    );
    return true;
}

constexpr uint32_t kPhysicalAicoreCount = 36;
constexpr uint32_t kSubcoresPerAicore = 3;
constexpr uint32_t kPhysicalSubcoreCount = kPhysicalAicoreCount * kSubcoresPerAicore;
constexpr uint32_t kAicorePerDie = 18;
constexpr uint32_t kSubcoresPerDie = kAicorePerDie * kSubcoresPerAicore;
constexpr uint32_t kAivBaseInDie = kAicorePerDie;
constexpr uint64_t kSubcoreStride = 0x100000ULL;
constexpr uint32_t kAicoreMapBytes = 0x300000U;

static_assert(kPhysicalSubcoreCount == kPmuPhysicalSubcores, "PMU table size mismatch");

// halResMap 只负责把 36 个物理 AICore 展开成 AIC/AIV 共 108 项 MMIO base。
// 它不负责恢复 PMU selector/CTRL；恢复必须先由 PmuSession::Restore 完成。
class RegisterMappings {
public:
    using MapFn = drvError_t (*)(unsigned int, struct res_map_info *, unsigned long *, unsigned int *);
    using UnmapFn = drvError_t (*)(unsigned int, struct res_map_info *);

    ~RegisterMappings() { Release(); }

    bool Initialize(uint32_t device)
    {
        device_ = device;
        map_ = reinterpret_cast<MapFn>(dlsym(RTLD_DEFAULT, "halResMap"));
        unmap_ = reinterpret_cast<UnmapFn>(dlsym(RTLD_DEFAULT, "halResUnmap"));
        if (map_ == nullptr || unmap_ == nullptr) {
            hal_handle_ = dlopen("libascend_hal.so", RTLD_NOW | RTLD_GLOBAL);
            if (hal_handle_ != nullptr) {
                map_ = reinterpret_cast<MapFn>(dlsym(hal_handle_, "halResMap"));
                unmap_ = reinterpret_cast<UnmapFn>(dlsym(hal_handle_, "halResUnmap"));
            }
        }
        if (map_ == nullptr || unmap_ == nullptr) {
            std::fprintf(stderr, "Cannot resolve halResMap/halResUnmap.\n");
            return false;
        }

        for (uint32_t physical = 0; physical < kPhysicalAicoreCount; ++physical) {
            res_map_info &info = map_info_[physical];
            std::memset(&info, 0, sizeof(info));
            info.target_proc_type = PROCESS_CP1;
            info.res_type = RES_AICORE;
            info.res_id = physical;
            unsigned long map_address = 0;
            unsigned int map_length = kAicoreMapBytes;
            const drvError_t error = map_(device_, &info, &map_address, &map_length);
            if (error != 0 || map_address == 0 || map_length < kAicoreMapBytes) {
                std::fprintf(
                    stderr, "halResMap failed: physical=%u error=%d address=0x%lx length=%u\n", physical,
                    static_cast<int>(error), map_address, map_length
                );
                Release();
                return false;
            }
            ++mapped_count_;
            const uint64_t base = static_cast<uint64_t>(map_address);
            const uint32_t die = physical / kAicorePerDie;
            const uint32_t local = physical % kAicorePerDie;
            const uint32_t die_base = die * kSubcoresPerDie;
            register_bases_[die_base + local] = base;
            const uint32_t aiv0 = die_base + kAivBaseInDie + local * 2;
            register_bases_[aiv0] = base + kSubcoreStride;
            register_bases_[aiv0 + 1] = base + 2 * kSubcoreStride;
        }
        return true;
    }

    void Release()
    {
        while (mapped_count_ != 0) {
            --mapped_count_;
            const drvError_t error = unmap_(device_, &map_info_[mapped_count_]);
            if (error != 0) {
                std::fprintf(
                    stderr, "halResUnmap failed: physical=%u error=%d\n", mapped_count_,
                    static_cast<int>(error)
                );
            }
        }
        if (hal_handle_ != nullptr) {
            dlclose(hal_handle_);
            hal_handle_ = nullptr;
        }
    }

    const std::array<uint64_t, kPhysicalSubcoreCount> &RegisterBases() const { return register_bases_; }

private:
    uint32_t device_ = 0;
    uint32_t mapped_count_ = 0;
    void *hal_handle_ = nullptr;
    MapFn map_ = nullptr;
    UnmapFn unmap_ = nullptr;
    std::array<res_map_info, kPhysicalAicoreCount> map_info_{};
    std::array<uint64_t, kPhysicalSubcoreCount> register_bases_{};
};

// 两个 scalar PMU probe 共用唯一配置会话：selector/range 只配置一次，所有
// 单 AIV 样本完成后统一恢复。command 始终经 inline KernelArgs 传给 AICPU；
// Configure/Restore 之间绝不由 host 再 H2D 覆盖 PmuControl cache line。
class PmuSession {
public:
    bool Initialize(
        uint32_t device, aclrtStream stream, const std::string &kernel_path, const char *helper_name
    )
    {
        stream_ = stream;
        device_ = device;
        if (!QueryActiveSubcoreLimits(stream, &active_limits_)) return false;
        if (!mappings_.Initialize(device)) return false;

        const size_t register_bytes = sizeof(mappings_.RegisterBases());
        if (!CheckAcl(
                aclrtMalloc(&register_bases_device_, register_bytes, ACL_MEM_MALLOC_NORMAL_ONLY),
                "aclrtMalloc(PMU regs)"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    register_bases_device_, register_bytes, mappings_.RegisterBases().data(), register_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "aclrtMemcpy(H2D PMU regs)"
            ) ||
            !CheckAcl(
                aclrtMalloc(&control_device_, sizeof(PmuControl), ACL_MEM_MALLOC_NORMAL_ONLY),
                "aclrtMalloc(PMU control)"
            )) {
            return false;
        }

        control_.magic = kPmuControlMagic;
        control_.version = kPmuControlVersion;
        control_.expected_subcores = active_limits_.total;
        if (!CheckAcl(
                aclrtMemcpy(
                    control_device_, sizeof(control_), &control_, sizeof(control_), ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "aclrtMemcpy(H2D initial PMU control)"
            )) {
            return false;
        }

        const std::string dispatcher_path = ArtifactBesideKernel(kernel_path, "libsimpler_aicpu_dispatcher.so");
        const std::string helper_path = ArtifactBesideKernel(kernel_path, helper_name);
        const std::vector<char> dispatcher_data = ReadBinary(dispatcher_path);
        const std::vector<char> helper_data = ReadBinary(helper_path);
        if (dispatcher_data.empty() || helper_data.empty()) {
            std::fprintf(stderr, "Cannot read PMU artifacts: %s %s\n", dispatcher_path.c_str(), helper_path.c_str());
            return false;
        }
        if (loader_.BootstrapDispatcher(
                dispatcher_data.data(), dispatcher_data.size(), helper_data.data(), helper_data.size(), stream,
                static_cast<int>(device)
            ) != 0 ||
            loader_.Init() != 0) {
            std::fprintf(stderr, "Cannot initialize PMU AICPU helper.\n");
            return false;
        }
        kernel_args_.runtime_args = reinterpret_cast<Runtime *>(control_device_);
        kernel_args_.regs = reinterpret_cast<uint64_t>(register_bases_device_);
        kernel_args_.device_id = device;
        initialized_ = true;
        return true;
    }

    bool Configure()
    {
        if (!initialized_ || configured_) return false;
        const bool command_ok = RunCommand(PmuCommand::Configure);
        // 若 Configure 报错且设备侧回滚也失败，bitmap 仍保存待恢复项；
        // 不能因为业务状态失败就丢失 Restore 所有权。
        configured_ = control_.configured != 0;
        return command_ok;
    }

    bool Restore()
    {
        if (!configured_) return true;
        if (!RunCommand(PmuCommand::Restore)) return false;
        configured_ = false;
        return true;
    }

    bool Finalize()
    {
        bool ok = Restore();
        loader_.Finalize();
        if (control_device_ != nullptr) {
            ok &= CheckAcl(aclrtFree(control_device_), "aclrtFree(PMU control)");
            control_device_ = nullptr;
        }
        if (register_bases_device_ != nullptr) {
            ok &= CheckAcl(aclrtFree(register_bases_device_), "aclrtFree(PMU regs)");
            register_bases_device_ = nullptr;
        }
        mappings_.Release();
        initialized_ = false;
        return ok;
    }

    uint64_t RegisterBasesDeviceAddress() const
    {
        return reinterpret_cast<uint64_t>(register_bases_device_);
    }

    bool IsConfiguredSubcore(uint32_t index) const
    {
        return configured_ && IsPmuSubcoreConfigured(control_, index);
    }

private:
    bool RunCommand(PmuCommand command)
    {
        control_.command = static_cast<uint32_t>(command);
        control_.status = kPmuStatusPending;
        kernel_args_.enable_profiling_flag = static_cast<uint32_t>(command);
        const int launch_error = loader_.LaunchBuiltInOp(
            stream_, &kernel_args_, 1, host::KernelNames::RunName
        );
        if (launch_error != 0) {
            std::fprintf(stderr, "AICPU PMU helper launch failed: %d\n", launch_error);
            return false;
        }
        if (!CheckAcl(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream(PMU helper)") ||
            !CheckAcl(
                aclrtMemcpy(
                    &control_, sizeof(control_), control_device_, sizeof(control_), ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "aclrtMemcpy(D2H PMU control)"
            )) {
            return false;
        }
        const uint32_t bitmap_count = CountPmuConfiguredSubcores(control_);
        const bool expected_state = command == PmuCommand::Configure
            ? control_.configured == 1 && control_.processed_subcores == control_.expected_subcores &&
                bitmap_count == control_.expected_subcores &&
                control_.skipped_subcores + bitmap_count == kPmuPhysicalSubcores
            : control_.configured == 0 && control_.processed_subcores == 0 && bitmap_count == 0;
        if (control_.status != 0 || !expected_state) {
            const auto failed_field = static_cast<PmuRegisterField>(control_.first_failed_field);
            std::fprintf(
                stderr,
                "PMU helper failed: command=%u status=%d configured=%u processed=%u bitmap_count=%u "
                "expected_subcores=%u skipped=%u failed_index=%u failed_field=%s(%u) "
                "observed=0x%x expected=0x%x\n",
                control_.command, static_cast<int>(control_.status), control_.configured,
                control_.processed_subcores, bitmap_count, control_.expected_subcores,
                control_.skipped_subcores, control_.first_failed_index,
                PmuRegisterFieldName(failed_field), control_.first_failed_field,
                control_.first_failed_observed, control_.first_failed_expected
            );
            return false;
        }
        if (command == PmuCommand::Configure) {
            const auto failed_field = static_cast<PmuRegisterField>(control_.first_failed_field);
            std::printf(
                "[PMU_OWNER] configured=%u bitmap_count=%u expected=%u skipped=%u "
                "first_skipped_index=%u first_skipped_field=%s(%u) observed=0x%x expected_value=0x%x\n",
                control_.processed_subcores, bitmap_count, control_.expected_subcores,
                control_.skipped_subcores, control_.first_failed_index,
                PmuRegisterFieldName(failed_field), control_.first_failed_field,
                control_.first_failed_observed, control_.first_failed_expected
            );
            std::printf(
                "[PMU_OWNER] configured_bitmap=%08x:%08x:%08x:%08x\n",
                control_.configured_bitmap[3], control_.configured_bitmap[2],
                control_.configured_bitmap[1], control_.configured_bitmap[0]
            );
        }
        return true;
    }

    uint32_t device_ = 0;
    aclrtStream stream_ = nullptr;
    RegisterMappings mappings_;
    host::LoadAicpuOp loader_;
    void *register_bases_device_ = nullptr;
    void *control_device_ = nullptr;
    PmuControl control_{};
    KernelArgs kernel_args_{};
    ActiveSubcoreLimits active_limits_{};
    bool initialized_ = false;
    bool configured_ = false;
};

}  // namespace atomic_probe::pmu

#endif  // TESTS_ATOMIC_PROBE_CCEC_PMU_PROBE_HOST_SUPPORT_H_
