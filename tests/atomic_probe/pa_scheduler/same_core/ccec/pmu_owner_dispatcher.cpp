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

// 这个 SO 只在初始化阶段由 libaicpu_extend_kernels.so 临时加载。它在主
// aicpu_scheduler 有权限访问的预安装目录中落盘真正的 PMU owner SO；随后
// host 通过 mode=0 JSON 注册直接调用 owner，不会在每次命令中再经过本文件。

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>

extern "C" void DlogRecord(int module_id, int level, const char *format, ...);

namespace {

constexpr int kDlogModuleCcecpu = 3;
constexpr int kDlogLevelError = 3;
constexpr uint64_t kFnvOffsetBasis = UINT64_C(14695981039346656037);
constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);

void Log(const char *format, ...)
{
    char buffer[1024] = {};
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(buffer, sizeof(buffer), format, arguments);
    va_end(arguments);
    DlogRecord(kDlogModuleCcecpu, kDlogLevelError, "[pa-pmu-dispatcher] %s", buffer);
}

// libaicpu_extend_kernels 固定从 KernelArgs::device_args（offset 40）获取
// DeviceArgs。后五个 qword 是本地 bootstrap 协议，offset 必须保持不变。
struct BootstrapKernelArgs {
    uint64_t unused[5];
    void *device_args;
    void *runtime_args;
    uint64_t regs;
};

struct BootstrapDeviceArgs {
    uint64_t unused[12];
    uint64_t dispatcher_so_device;  // offset 96，extend kernel 消费
    uint64_t dispatcher_so_bytes;   // offset 104
    uint64_t device_id;             // offset 112
    uint64_t owner_so_device;       // offset 120，本 dispatcher 消费
    uint64_t owner_so_bytes;        // offset 128
};

static_assert(offsetof(BootstrapKernelArgs, device_args) == 40U, "bootstrap KernelArgs ABI changed");
static_assert(offsetof(BootstrapDeviceArgs, dispatcher_so_device) == 96U, "dispatcher address offset changed");
static_assert(offsetof(BootstrapDeviceArgs, dispatcher_so_bytes) == 104U, "dispatcher size offset changed");
static_assert(offsetof(BootstrapDeviceArgs, device_id) == 112U, "device id offset changed");
static_assert(offsetof(BootstrapDeviceArgs, owner_so_device) == 120U, "owner address offset changed");
static_assert(offsetof(BootstrapDeviceArgs, owner_so_bytes) == 128U, "owner size offset changed");

// host 与 device 都对完整 SO 字节做 FNV-1a；不只散列 ELF header，避免同一
// toolchain 产出的等长 SO 发生名字碰撞并误加载旧代码。
uint64_t FingerprintBytes(const void *data, uint64_t bytes)
{
    const auto *input = static_cast<const uint8_t *>(data);
    uint64_t hash = kFnvOffsetBasis;
    for (uint64_t index = 0U; index < bytes; ++index) {
        hash ^= input[index];
        hash *= kFnvPrime;
    }
    return hash;
}

std::string OwnerSoPath(uint64_t fingerprint, uint64_t device_id)
{
    char path[256] = {};
    (void)snprintf(
        path, sizeof(path),
        "/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/pa_scheduler_pmu_owner_%016llx_d%llu.so",
        static_cast<unsigned long long>(fingerprint), static_cast<unsigned long long>(device_id)
    );
    return path;
}

// 先写同目录临时文件，再原子 rename。目标名由内容、device 共同确定；临时
// 名再加入进程和源地址，避免同进程并发 bootstrap 写同一个临时 inode。
bool WriteOwnerSo(const std::string &target, const void *data, uint64_t bytes)
{
    if (data == nullptr || bytes == 0U ||
        bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        Log("invalid owner SO buffer: data=%p bytes=%llu", data, static_cast<unsigned long long>(bytes));
        return false;
    }

    char temporary[384] = {};
    (void)snprintf(
        temporary, sizeof(temporary), "%s.tmp.%d.%016llx", target.c_str(), static_cast<int>(getpid()),
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(data))
    );
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output.is_open()) {
            Log("open %s failed: %s", temporary, strerror(errno));
            return false;
        }
        output.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
        output.close();
        if (!output) {
            Log("write %s failed", temporary);
            (void)unlink(temporary);
            return false;
        }
    }
    if (chmod(temporary, 0755) != 0) {
        Log("chmod %s failed: %s", temporary, strerror(errno));
        (void)unlink(temporary);
        return false;
    }
    if (rename(temporary, target.c_str()) != 0) {
        Log("rename %s -> %s failed: %s", temporary, target.c_str(), strerror(errno));
        (void)unlink(temporary);
        return false;
    }
    return true;
}

}  // namespace

extern "C" {

// extend kernel 在 dlopen 后要求三个符号同时存在；本 dispatcher 只使用 Init，
// 另外两个入口保持无副作用成功返回，避免未来 runtime 的预探测变成故障。
__attribute__((visibility("default"))) int StaticTileFwkBackendKernelServer(void *arguments)
{
    (void)arguments;
    return 0;
}

__attribute__((visibility("default"))) uint32_t DynTileFwkBackendKernelServer(void *arguments)
{
    (void)arguments;
    return 0U;
}

__attribute__((visibility("default"))) uint32_t DynTileFwkBackendKernelServerInit(void *arguments)
{
    if (arguments == nullptr) {
        Log("Init received null KernelArgs");
        return 1U;
    }
    auto *kernel_args = static_cast<BootstrapKernelArgs *>(arguments);
    auto *device_args = static_cast<BootstrapDeviceArgs *>(kernel_args->device_args);
    if (device_args == nullptr || device_args->owner_so_device == 0U || device_args->owner_so_bytes == 0U) {
        Log(
            "Init received invalid DeviceArgs: args=%p owner=%016llx bytes=%llu", device_args,
            static_cast<unsigned long long>(device_args == nullptr ? 0U : device_args->owner_so_device),
            static_cast<unsigned long long>(device_args == nullptr ? 0U : device_args->owner_so_bytes)
        );
        return 2U;
    }

    const void *owner_so = reinterpret_cast<const void *>(static_cast<uintptr_t>(device_args->owner_so_device));
    const uint64_t fingerprint = FingerprintBytes(owner_so, device_args->owner_so_bytes);
    const std::string target = OwnerSoPath(fingerprint, device_args->device_id);
    if (!WriteOwnerSo(target, owner_so, device_args->owner_so_bytes)) return 3U;

    Log(
        "installed %s (%llu bytes)", target.c_str(),
        static_cast<unsigned long long>(device_args->owner_so_bytes)
    );
    return 0U;
}

}  // extern "C"
