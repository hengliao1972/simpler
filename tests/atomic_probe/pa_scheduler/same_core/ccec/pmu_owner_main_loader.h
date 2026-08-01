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

#ifndef PA_SCHEDULER_CCEC_PMU_OWNER_MAIN_LOADER_H_
#define PA_SCHEDULER_CCEC_PMU_OWNER_MAIN_LOADER_H_

// PMU owner 的自包含 host 装载器：
//   1. 通过 libaicpu_extend_kernels bootstrap 临时 dispatcher；
//   2. dispatcher 将 owner SO 落到主 aicpu_scheduler 的预安装目录；
//   3. 用 cpuKernelMode=0 JSON 注册 owner 的 simpler_aicpu_exec；
//   4. 后续 Configure/Restore 都用缓存的 rtFuncHandle 直接下发。
//
// 本头文件故意不定义 owner 命令字段。Launch 接受调用方构造的完整参数块，
// 从而让装载 ABI 与 PMU 状态机 ABI 解耦，也便于先独立验证 Path-A。

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "runtime/rt.h"
#include "runtime/runtime/rts/rts_kernel.h"

namespace pa_scheduler::pmu_owner {

class MainAicpuLoader {
public:
    MainAicpuLoader() = default;
    MainAicpuLoader(const MainAicpuLoader &) = delete;
    MainAicpuLoader &operator=(const MainAicpuLoader &) = delete;
    MainAicpuLoader(MainAicpuLoader &&) = delete;
    MainAicpuLoader &operator=(MainAicpuLoader &&) = delete;

    ~MainAicpuLoader() { (void)Finalize(); }

    // stream 必须属于当前 device，并且调用期间当前 ACL device 不能切换。
    // 成功后 owner SO 已注册到主 aicpu_scheduler，但尚未执行任何 PMU 命令。
    int Initialize(
        const std::string &dispatcher_so_path, const std::string &owner_so_path,
        aclrtStream stream, int32_t device_id
    )
    {
        if (IsInitialized() || stream == nullptr || device_id < 0) {
            return Fail("Initialize received invalid state, stream, or device id", kInvalidArgument);
        }

        const std::vector<uint8_t> dispatcher = ReadBinary(dispatcher_so_path);
        const std::vector<uint8_t> owner = ReadBinary(owner_so_path);
        if (dispatcher.empty() || owner.empty()) {
            std::fprintf(
                stderr, "[PMU_OWNER_LOADER] cannot read dispatcher/owner: %s (%zu B), %s (%zu B)\n",
                dispatcher_so_path.c_str(), dispatcher.size(), owner_so_path.c_str(), owner.size()
            );
            return kFileError;
        }

        device_id_ = device_id;
        owner_fingerprint_ = FingerprintBytes(owner.data(), owner.size());
        owner_so_basename_ = MakeOwnerSoBasename(owner_fingerprint_, device_id_);
        op_type_ = MakeOpType(owner_fingerprint_, device_id_);

        int result = Bootstrap(dispatcher, owner, stream);
        if (result == 0) result = RegisterOwner();
        if (result != 0) {
            (void)Finalize();
            return result;
        }
        return 0;
    }

    // 参数块由 runtime 在 launch 时复制；调用方只需保证本函数返回前 host
    // buffer 有效。参数中的 GM 指针仍必须在设备命令同步结束前保持有效。
    int Launch(
        aclrtStream stream, void *kernel_arguments, size_t argument_bytes,
        uint32_t aicpu_blocks = 1U
    ) const
    {
        if (!IsInitialized() || stream == nullptr || kernel_arguments == nullptr ||
            argument_bytes == 0U || argument_bytes > std::numeric_limits<uint32_t>::max() ||
            aicpu_blocks == 0U) {
            return Fail("Launch received invalid state or arguments", kInvalidArgument);
        }

        rtCpuKernelArgs_t cpu_arguments = {};
        cpu_arguments.baseArgs.args = kernel_arguments;
        cpu_arguments.baseArgs.argsSize = static_cast<uint32_t>(argument_bytes);
        rtKernelLaunchCfg_t launch_config = {};
        rtLaunchKernelAttr_t launch_attribute = {};
        launch_config.attrs = &launch_attribute;
        launch_config.numAttrs = 0U;

        const rtError_t result = rtsLaunchCpuKernel(
            function_handle_, aicpu_blocks, static_cast<rtStream_t>(stream),
            &launch_config, &cpu_arguments
        );
        if (result != RT_ERROR_NONE) {
            std::fprintf(stderr, "[PMU_OWNER_LOADER] rtsLaunchCpuKernel failed: %d\n", result);
        }
        return static_cast<int>(result);
    }

    // Finalize 只释放 host/runtime 注册资源，不删除设备侧预安装 SO；后者按内容
    // 指纹命名，可由同一设备上的后续进程原子覆盖。
    int Finalize()
    {
        int result = 0;
        function_handle_ = nullptr;
        if (binary_handle_ != nullptr) {
            const rtError_t unload_result = rtsBinaryUnload(binary_handle_);
            if (unload_result != RT_ERROR_NONE) {
                std::fprintf(stderr, "[PMU_OWNER_LOADER] rtsBinaryUnload failed: %d\n", unload_result);
                result = static_cast<int>(unload_result);
            }
            binary_handle_ = nullptr;
        }
        if (!json_path_.empty()) {
            if (std::remove(json_path_.c_str()) != 0 && result == 0) {
                std::fprintf(stderr, "[PMU_OWNER_LOADER] remove JSON failed: %s\n", json_path_.c_str());
                result = kFileError;
            }
            json_path_.clear();
        }
        device_id_ = -1;
        owner_fingerprint_ = 0U;
        owner_so_basename_.clear();
        op_type_.clear();
        return result;
    }

    bool IsInitialized() const { return binary_handle_ != nullptr && function_handle_ != nullptr; }
    uint64_t OwnerFingerprint() const { return owner_fingerprint_; }
    const std::string &OwnerSoBasename() const { return owner_so_basename_; }
    const std::string &OpType() const { return op_type_; }

private:
    static constexpr int kInvalidArgument = -1;
    static constexpr int kFileError = -2;
    static constexpr int kBootstrapError = -3;
    static constexpr uint64_t kFnvOffsetBasis = UINT64_C(14695981039346656037);
    static constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);
    static constexpr const char *kOwnerFunction = "simpler_aicpu_exec";

    struct DeviceBuffer {
        void *address = nullptr;
        DeviceBuffer() = default;
        DeviceBuffer(const DeviceBuffer &) = delete;
        DeviceBuffer &operator=(const DeviceBuffer &) = delete;
        ~DeviceBuffer()
        {
            if (address != nullptr) (void)aclrtFree(address);
        }
        aclError Allocate(size_t bytes)
        {
            return aclrtMalloc(&address, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        }
    };

    static int Fail(const char *message, int code)
    {
        std::fprintf(stderr, "[PMU_OWNER_LOADER] %s\n", message);
        return code;
    }

    static std::vector<uint8_t> ReadBinary(const std::string &path)
    {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input.is_open()) return {};
        const std::streampos end = input.tellg();
        if (end <= std::streampos(0) ||
            static_cast<uint64_t>(end) > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return {};
        }
        std::vector<uint8_t> bytes(static_cast<size_t>(end));
        input.seekg(0, std::ios::beg);
        if (!input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
            return {};
        }
        return bytes;
    }

    static uint64_t FingerprintBytes(const void *data, size_t bytes)
    {
        const auto *input = static_cast<const uint8_t *>(data);
        uint64_t hash = kFnvOffsetBasis;
        for (size_t index = 0U; index < bytes; ++index) {
            hash ^= input[index];
            hash *= kFnvPrime;
        }
        return hash;
    }

    static std::string MakeOwnerSoBasename(uint64_t fingerprint, int32_t device_id)
    {
        char name[128] = {};
        (void)snprintf(
            name, sizeof(name), "pa_scheduler_pmu_owner_%016llx_d%d.so",
            static_cast<unsigned long long>(fingerprint), device_id
        );
        return name;
    }

    static std::string MakeOpType(uint64_t fingerprint, int32_t device_id)
    {
        char name[160] = {};
        (void)snprintf(
            name, sizeof(name), "pa_scheduler_pmu_owner_%016llx_d%d",
            static_cast<unsigned long long>(fingerprint), device_id
        );
        return name;
    }

    int Bootstrap(
        const std::vector<uint8_t> &dispatcher, const std::vector<uint8_t> &owner,
        aclrtStream stream
    ) const
    {
        DeviceBuffer dispatcher_device;
        DeviceBuffer owner_device;
        DeviceBuffer device_args;
        aclError acl_result = dispatcher_device.Allocate(dispatcher.size());
        if (acl_result != ACL_SUCCESS) return ReportAcl("aclrtMalloc(dispatcher)", acl_result);
        acl_result = aclrtMemcpy(
            dispatcher_device.address, dispatcher.size(), dispatcher.data(), dispatcher.size(),
            ACL_MEMCPY_HOST_TO_DEVICE
        );
        if (acl_result != ACL_SUCCESS) return ReportAcl("aclrtMemcpy(dispatcher H2D)", acl_result);

        acl_result = owner_device.Allocate(owner.size());
        if (acl_result != ACL_SUCCESS) return ReportAcl("aclrtMalloc(owner)", acl_result);
        acl_result = aclrtMemcpy(
            owner_device.address, owner.size(), owner.data(), owner.size(), ACL_MEMCPY_HOST_TO_DEVICE
        );
        if (acl_result != ACL_SUCCESS) return ReportAcl("aclrtMemcpy(owner H2D)", acl_result);

        constexpr size_t kDeviceArgsBytes = 160U;
        uint8_t host_device_args[kDeviceArgsBytes] = {};
        const auto write_qword = [&](size_t offset, uint64_t value) {
            std::memcpy(host_device_args + offset, &value, sizeof(value));
        };
        write_qword(96U, reinterpret_cast<uint64_t>(dispatcher_device.address));
        write_qword(104U, static_cast<uint64_t>(dispatcher.size()));
        write_qword(112U, static_cast<uint64_t>(device_id_));
        write_qword(120U, reinterpret_cast<uint64_t>(owner_device.address));
        write_qword(128U, static_cast<uint64_t>(owner.size()));

        acl_result = device_args.Allocate(kDeviceArgsBytes);
        if (acl_result != ACL_SUCCESS) return ReportAcl("aclrtMalloc(bootstrap args)", acl_result);
        acl_result = aclrtMemcpy(
            device_args.address, kDeviceArgsBytes, host_device_args, kDeviceArgsBytes,
            ACL_MEMCPY_HOST_TO_DEVICE
        );
        if (acl_result != ACL_SUCCESS) return ReportAcl("aclrtMemcpy(bootstrap args H2D)", acl_result);

        // k_args 总长和三个字符串 offset 与仓内已上板的 Path-A 完全一致。
        struct BootstrapArguments {
            struct {
                uint64_t unused[5];
                uint64_t device_args_address;
                uint64_t padding[20];
            } kernel_args;
            char kernel_name[32];
            char so_name[32];
            char op_name[32];
        } arguments = {};
        static_assert(offsetof(BootstrapArguments, kernel_args.device_args_address) == 40U, "bootstrap ABI changed");
        arguments.kernel_args.device_args_address = reinterpret_cast<uint64_t>(device_args.address);
        constexpr char kBootstrapKernel[] = "DynTileFwkKernelServerInit";
        constexpr char kBootstrapSo[] = "libaicpu_extend_kernels.so";
        static_assert(sizeof(kBootstrapKernel) <= sizeof(arguments.kernel_name), "bootstrap kernel name too long");
        static_assert(sizeof(kBootstrapSo) <= sizeof(arguments.so_name), "bootstrap SO name too long");
        std::memcpy(arguments.kernel_name, kBootstrapKernel, sizeof(kBootstrapKernel));
        std::memcpy(arguments.so_name, kBootstrapSo, sizeof(kBootstrapSo));

        rtAicpuArgsEx_t runtime_arguments = {};
        runtime_arguments.args = &arguments;
        runtime_arguments.argsSize = sizeof(arguments);
        runtime_arguments.kernelNameAddrOffset = offsetof(BootstrapArguments, kernel_name);
        runtime_arguments.soNameAddrOffset = offsetof(BootstrapArguments, so_name);

        const rtError_t launch_result = rtAicpuKernelLaunchExWithArgs(
            rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", 1U,
            &runtime_arguments, nullptr, static_cast<rtStream_t>(stream), 0U
        );
        if (launch_result != RT_ERROR_NONE) {
            std::fprintf(
                stderr, "[PMU_OWNER_LOADER] rtAicpuKernelLaunchExWithArgs failed: %d\n",
                launch_result
            );
            return static_cast<int>(launch_result);
        }
        acl_result = aclrtSynchronizeStream(stream);
        if (acl_result != ACL_SUCCESS) return ReportAcl("aclrtSynchronizeStream(bootstrap)", acl_result);
        return 0;
    }

    int RegisterOwner()
    {
        char path[256] = {};
        (void)snprintf(
            path, sizeof(path), "/tmp/pa_scheduler_pmu_owner_%016llx_d%d_p%d_i%016llx.json",
            static_cast<unsigned long long>(owner_fingerprint_), device_id_, static_cast<int>(getpid()),
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(this))
        );
        json_path_ = path;
        if (!WriteJson()) return kFileError;

        rtLoadBinaryOption_t option = {};
        option.optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
        option.value.cpuKernelMode = 0;
        rtLoadBinaryConfig_t configuration = {};
        configuration.options = &option;
        configuration.numOpt = 1U;

        rtError_t result = rtsBinaryLoadFromFile(json_path_.c_str(), &configuration, &binary_handle_);
        if (result != RT_ERROR_NONE) {
            std::fprintf(stderr, "[PMU_OWNER_LOADER] rtsBinaryLoadFromFile failed: %d\n", result);
            return static_cast<int>(result);
        }
        result = rtsFuncGetByName(binary_handle_, op_type_.c_str(), &function_handle_);
        if (result != RT_ERROR_NONE) {
            std::fprintf(stderr, "[PMU_OWNER_LOADER] rtsFuncGetByName(%s) failed: %d\n", op_type_.c_str(), result);
            return static_cast<int>(result);
        }
        return 0;
    }

    bool WriteJson() const
    {
        std::ofstream json(json_path_, std::ios::out | std::ios::trunc);
        if (!json.is_open()) {
            std::fprintf(stderr, "[PMU_OWNER_LOADER] cannot create JSON: %s\n", json_path_.c_str());
            return false;
        }
        // 所有动态字段仅含固定前缀、十六进制、十进制和下划线，不需要 JSON 转义。
        json << "{\n"
             << "  \"" << op_type_ << "\": {\n"
             << "    \"opInfo\": {\n"
             << "      \"functionName\": \"" << kOwnerFunction << "\",\n"
             << "      \"kernelSo\": \"" << owner_so_basename_ << "\",\n"
             << "      \"opKernelLib\": \"AICPUKernel\",\n"
             << "      \"computeCost\": \"100\",\n"
             << "      \"engine\": \"DNN_VM_AICPU\",\n"
             << "      \"flagAsync\": \"False\",\n"
             << "      \"flagPartial\": \"False\",\n"
             << "      \"userDefined\": \"False\"\n"
             << "    }\n"
             << "  }\n"
             << "}\n";
        json.close();
        if (!json) {
            std::fprintf(stderr, "[PMU_OWNER_LOADER] writing JSON failed: %s\n", json_path_.c_str());
            return false;
        }
        return true;
    }

    static int ReportAcl(const char *operation, aclError result)
    {
        std::fprintf(stderr, "[PMU_OWNER_LOADER] %s failed: %d\n", operation, static_cast<int>(result));
        return static_cast<int>(result == ACL_SUCCESS ? kBootstrapError : result);
    }

    int32_t device_id_ = -1;
    uint64_t owner_fingerprint_ = 0U;
    std::string owner_so_basename_;
    std::string op_type_;
    std::string json_path_;
    rtBinHandle binary_handle_ = nullptr;
    rtFuncHandle function_handle_ = nullptr;
};

}  // namespace pa_scheduler::pmu_owner

#endif  // PA_SCHEDULER_CCEC_PMU_OWNER_MAIN_LOADER_H_
