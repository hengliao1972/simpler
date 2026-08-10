/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */
#ifndef PA_SCHEDULER_AICPU_PLAN_PROTOCOL_PROBE_LOADER_H_
#define PA_SCHEDULER_AICPU_PLAN_PROTOCOL_PROBE_LOADER_H_

// Self-contained copy of the repository's verified Path-A mechanism:
// extend-kernel bootstrap -> temporary dispatcher installs a fingerprinted SO
// -> cpuKernelMode=0 registration -> direct main-aicpu_scheduler launch.

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

namespace plan_protocol_probe {

class PlanAicpuLoader {
public:
    PlanAicpuLoader() = default;
    PlanAicpuLoader(const PlanAicpuLoader &) = delete;
    PlanAicpuLoader &operator=(const PlanAicpuLoader &) = delete;
    ~PlanAicpuLoader() { (void)Finalize(); }

    int Initialize(
        const std::string &dispatcher_path, const std::string &owner_path,
        aclrtStream stream, int32_t device_id)
    {
        if (IsInitialized() || stream == nullptr || device_id < 0) {
            return Fail("invalid Initialize arguments", kInvalidArgument);
        }
        const std::vector<uint8_t> dispatcher = ReadBinary(dispatcher_path);
        const std::vector<uint8_t> owner = ReadBinary(owner_path);
        if (dispatcher.empty() || owner.empty()) {
            std::fprintf(
                stderr, "[PLAN_PROTOCOL_LOADER] cannot read %s (%zu B) or %s (%zu B)\n",
                dispatcher_path.c_str(), dispatcher.size(), owner_path.c_str(), owner.size());
            return kFileError;
        }

        device_id_ = device_id;
        fingerprint_ = Fingerprint(owner.data(), owner.size());
        so_basename_ = OwnerBasename(fingerprint_, device_id_);
        op_type_ = OpType(fingerprint_, device_id_);
        int result = Bootstrap(dispatcher, owner, stream);
        if (result == 0) result = Register();
        if (result != 0) (void)Finalize();
        return result;
    }

    int Launch(
        aclrtStream stream, void *arguments, size_t argument_bytes,
        uint32_t blocks = 1U) const
    {
        if (!IsInitialized() || stream == nullptr || arguments == nullptr ||
            argument_bytes == 0U || argument_bytes > std::numeric_limits<uint32_t>::max() ||
            blocks == 0U) {
            return Fail("invalid Launch arguments", kInvalidArgument);
        }
        rtCpuKernelArgs_t cpu_arguments{};
        cpu_arguments.baseArgs.args = arguments;
        cpu_arguments.baseArgs.argsSize = static_cast<uint32_t>(argument_bytes);
        rtLaunchKernelAttr_t attribute{};
        rtKernelLaunchCfg_t configuration{};
        configuration.attrs = &attribute;
        configuration.numAttrs = 0U;
        const rtError_t result = rtsLaunchCpuKernel(
            function_, blocks, static_cast<rtStream_t>(stream),
            &configuration, &cpu_arguments);
        if (result != RT_ERROR_NONE) {
            std::fprintf(stderr, "[PLAN_PROTOCOL_LOADER] rtsLaunchCpuKernel failed: %d\n", result);
        }
        return static_cast<int>(result);
    }

    int Finalize()
    {
        int status = 0;
        function_ = nullptr;
        if (binary_ != nullptr) {
            const rtError_t result = rtsBinaryUnload(binary_);
            if (result != RT_ERROR_NONE) status = static_cast<int>(result);
            binary_ = nullptr;
        }
        if (!json_path_.empty()) {
            if (std::remove(json_path_.c_str()) != 0 && status == 0) status = kFileError;
            json_path_.clear();
        }
        device_id_ = -1;
        fingerprint_ = 0U;
        so_basename_.clear();
        op_type_.clear();
        return status;
    }

    bool IsInitialized() const { return binary_ != nullptr && function_ != nullptr; }

private:
    static constexpr int kInvalidArgument = -1;
    static constexpr int kFileError = -2;
    static constexpr uint64_t kFnvOffset = UINT64_C(14695981039346656037);
    static constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);
    static constexpr const char *kOwnerFunction = "plan_protocol_aicpu_exec";

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

    static int Fail(const char *message, int status)
    {
        std::fprintf(stderr, "[PLAN_PROTOCOL_LOADER] %s\n", message);
        return status;
    }

    static std::vector<uint8_t> ReadBinary(const std::string &path)
    {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) return {};
        const std::streampos end = input.tellg();
        if (end <= std::streampos(0) ||
            static_cast<uint64_t>(end) > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return {};
        }
        std::vector<uint8_t> bytes(static_cast<size_t>(end));
        input.seekg(0, std::ios::beg);
        if (!input.read(reinterpret_cast<char *>(bytes.data()),
                        static_cast<std::streamsize>(bytes.size()))) return {};
        return bytes;
    }

    static uint64_t Fingerprint(const void *data, size_t bytes)
    {
        const auto *input = static_cast<const uint8_t *>(data);
        uint64_t hash = kFnvOffset;
        for (size_t index = 0U; index < bytes; ++index) {
            hash = (hash ^ input[index]) * kFnvPrime;
        }
        return hash;
    }

    static std::string OwnerBasename(uint64_t fingerprint, int32_t device)
    {
        char name[128]{};
        (void)snprintf(
            name, sizeof(name), "pa_plan_protocol_owner_%016llx_d%d.so",
            static_cast<unsigned long long>(fingerprint), device);
        return name;
    }

    static std::string OpType(uint64_t fingerprint, int32_t device)
    {
        char name[128]{};
        (void)snprintf(
            name, sizeof(name), "pa_plan_protocol_%016llx_d%d",
            static_cast<unsigned long long>(fingerprint), device);
        return name;
    }

    int Bootstrap(
        const std::vector<uint8_t> &dispatcher, const std::vector<uint8_t> &owner,
        aclrtStream stream) const
    {
        DeviceBuffer dispatcher_device;
        DeviceBuffer owner_device;
        DeviceBuffer device_args;
        aclError acl_result = dispatcher_device.Allocate(dispatcher.size());
        if (acl_result != ACL_SUCCESS) return ReportAcl("allocate dispatcher", acl_result);
        acl_result = aclrtMemcpy(
            dispatcher_device.address, dispatcher.size(), dispatcher.data(), dispatcher.size(),
            ACL_MEMCPY_HOST_TO_DEVICE);
        if (acl_result != ACL_SUCCESS) return ReportAcl("copy dispatcher", acl_result);
        acl_result = owner_device.Allocate(owner.size());
        if (acl_result != ACL_SUCCESS) return ReportAcl("allocate owner", acl_result);
        acl_result = aclrtMemcpy(
            owner_device.address, owner.size(), owner.data(), owner.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        if (acl_result != ACL_SUCCESS) return ReportAcl("copy owner", acl_result);

        constexpr size_t kDeviceArgsBytes = 160U;
        uint8_t host_args[kDeviceArgsBytes]{};
        const auto put64 = [&](size_t offset, uint64_t value) {
            std::memcpy(host_args + offset, &value, sizeof(value));
        };
        put64(96U, reinterpret_cast<uint64_t>(dispatcher_device.address));
        put64(104U, static_cast<uint64_t>(dispatcher.size()));
        put64(112U, static_cast<uint64_t>(device_id_));
        put64(120U, reinterpret_cast<uint64_t>(owner_device.address));
        put64(128U, static_cast<uint64_t>(owner.size()));
        acl_result = device_args.Allocate(kDeviceArgsBytes);
        if (acl_result != ACL_SUCCESS) return ReportAcl("allocate bootstrap args", acl_result);
        acl_result = aclrtMemcpy(
            device_args.address, kDeviceArgsBytes, host_args, kDeviceArgsBytes,
            ACL_MEMCPY_HOST_TO_DEVICE);
        if (acl_result != ACL_SUCCESS) return ReportAcl("copy bootstrap args", acl_result);

        struct BootstrapArguments {
            struct {
                uint64_t unused[5];
                uint64_t device_args_address;
                uint64_t padding[20];
            } kernel_args;
            char kernel_name[32];
            char so_name[32];
            char op_name[32];
        } arguments{};
        static_assert(
            offsetof(BootstrapArguments, kernel_args.device_args_address) == 40U,
            "bootstrap ABI changed");
        arguments.kernel_args.device_args_address = reinterpret_cast<uint64_t>(device_args.address);
        constexpr char kBootstrapKernel[] = "DynTileFwkKernelServerInit";
        constexpr char kBootstrapSo[] = "libaicpu_extend_kernels.so";
        std::memcpy(arguments.kernel_name, kBootstrapKernel, sizeof(kBootstrapKernel));
        std::memcpy(arguments.so_name, kBootstrapSo, sizeof(kBootstrapSo));

        rtAicpuArgsEx_t runtime_arguments{};
        runtime_arguments.args = &arguments;
        runtime_arguments.argsSize = sizeof(arguments);
        runtime_arguments.kernelNameAddrOffset = offsetof(BootstrapArguments, kernel_name);
        runtime_arguments.soNameAddrOffset = offsetof(BootstrapArguments, so_name);
        const rtError_t result = rtAicpuKernelLaunchExWithArgs(
            rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", 1U,
            &runtime_arguments, nullptr, static_cast<rtStream_t>(stream), 0U);
        if (result != RT_ERROR_NONE) {
            std::fprintf(
                stderr, "[PLAN_PROTOCOL_LOADER] bootstrap launch failed: %d\n", result);
            return static_cast<int>(result);
        }
        acl_result = aclrtSynchronizeStream(stream);
        return acl_result == ACL_SUCCESS ? 0 : ReportAcl("synchronize bootstrap", acl_result);
    }

    int Register()
    {
        char path[256]{};
        (void)snprintf(
            path, sizeof(path), "/tmp/pa_plan_protocol_%016llx_d%d_p%d_i%016llx.json",
            static_cast<unsigned long long>(fingerprint_), device_id_, static_cast<int>(getpid()),
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(this)));
        json_path_ = path;
        if (!WriteJson()) return kFileError;

        rtLoadBinaryOption_t option{};
        option.optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
        option.value.cpuKernelMode = 0;
        rtLoadBinaryConfig_t configuration{};
        configuration.options = &option;
        configuration.numOpt = 1U;
        rtError_t result = rtsBinaryLoadFromFile(json_path_.c_str(), &configuration, &binary_);
        if (result != RT_ERROR_NONE) {
            std::fprintf(stderr, "[PLAN_PROTOCOL_LOADER] binary registration failed: %d\n", result);
            return static_cast<int>(result);
        }
        result = rtsFuncGetByName(binary_, op_type_.c_str(), &function_);
        if (result != RT_ERROR_NONE) {
            std::fprintf(stderr, "[PLAN_PROTOCOL_LOADER] function lookup failed: %d\n", result);
            return static_cast<int>(result);
        }
        return 0;
    }

    bool WriteJson() const
    {
        std::ofstream json(json_path_, std::ios::out | std::ios::trunc);
        if (!json) return false;
        json << "{\n"
             << "  \"" << op_type_ << "\": {\n"
             << "    \"opInfo\": {\n"
             << "      \"functionName\": \"" << kOwnerFunction << "\",\n"
             << "      \"kernelSo\": \"" << so_basename_ << "\",\n"
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
        return static_cast<bool>(json);
    }

    static int ReportAcl(const char *operation, aclError result)
    {
        std::fprintf(
            stderr, "[PLAN_PROTOCOL_LOADER] %s failed: %d\n",
            operation, static_cast<int>(result));
        return static_cast<int>(result == ACL_SUCCESS ? kFileError : result);
    }

    int32_t device_id_ = -1;
    uint64_t fingerprint_ = 0U;
    std::string so_basename_;
    std::string op_type_;
    std::string json_path_;
    rtBinHandle binary_ = nullptr;
    rtFuncHandle function_ = nullptr;
};

}  // namespace plan_protocol_probe

#endif  // PA_SCHEDULER_AICPU_PLAN_PROTOCOL_PROBE_LOADER_H_
