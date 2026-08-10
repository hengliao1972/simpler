/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */
// Initialization-only Path-A dispatcher.  The three exported names are fixed
// by libaicpu_extend_kernels; the installed artifact and owner entry are probe
// specific and cannot collide with cross_core_ordinary.

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
constexpr uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);

void Log(const char *format, ...)
{
    char buffer[1024]{};
    va_list arguments;
    va_start(arguments, format);
    (void)vsnprintf(buffer, sizeof(buffer), format, arguments);
    va_end(arguments);
    DlogRecord(kDlogModuleCcecpu, kDlogLevelError, "[plan-protocol-dispatcher] %s", buffer);
}

struct BootstrapKernelArgs {
    uint64_t unused[5];
    void *device_args;
    void *runtime_args;
    uint64_t regs;
};

struct BootstrapDeviceArgs {
    uint64_t unused[12];
    uint64_t dispatcher_so_device;
    uint64_t dispatcher_so_bytes;
    uint64_t device_id;
    uint64_t owner_so_device;
    uint64_t owner_so_bytes;
};

static_assert(offsetof(BootstrapKernelArgs, device_args) == 40U, "bootstrap ABI changed");
static_assert(offsetof(BootstrapDeviceArgs, dispatcher_so_device) == 96U, "dispatcher offset changed");
static_assert(offsetof(BootstrapDeviceArgs, owner_so_device) == 120U, "owner offset changed");

uint64_t Fingerprint(const void *data, uint64_t bytes)
{
    const auto *input = static_cast<const uint8_t *>(data);
    uint64_t hash = kFnvOffset;
    for (uint64_t index = 0U; index < bytes; ++index) hash = (hash ^ input[index]) * kFnvPrime;
    return hash;
}

std::string OwnerPath(uint64_t fingerprint, uint64_t device)
{
    char path[256]{};
    (void)snprintf(
        path, sizeof(path),
        "/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/pa_plan_protocol_owner_%016llx_d%llu.so",
        static_cast<unsigned long long>(fingerprint), static_cast<unsigned long long>(device));
    return path;
}

bool WriteOwner(const std::string &target, const void *data, uint64_t bytes)
{
    if (data == nullptr || bytes == 0U ||
        bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) return false;
    char temporary[384]{};
    (void)snprintf(
        temporary, sizeof(temporary), "%s.tmp.%d.%016llx", target.c_str(),
        static_cast<int>(getpid()),
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(data)));
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            Log("open %s failed: %s", temporary, strerror(errno));
            return false;
        }
        output.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
        output.close();
        if (!output) {
            (void)unlink(temporary);
            return false;
        }
    }
    if (chmod(temporary, 0755) != 0 || rename(temporary, target.c_str()) != 0) {
        Log("install %s failed: %s", target.c_str(), strerror(errno));
        (void)unlink(temporary);
        return false;
    }
    return true;
}

}  // namespace

extern "C" {

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
    if (arguments == nullptr) return 1U;
    auto *kernel_args = static_cast<BootstrapKernelArgs *>(arguments);
    auto *device_args = static_cast<BootstrapDeviceArgs *>(kernel_args->device_args);
    if (device_args == nullptr || device_args->owner_so_device == 0U ||
        device_args->owner_so_bytes == 0U) return 2U;
    const void *owner = reinterpret_cast<const void *>(
        static_cast<uintptr_t>(device_args->owner_so_device));
    const uint64_t fingerprint = Fingerprint(owner, device_args->owner_so_bytes);
    const std::string target = OwnerPath(fingerprint, device_args->device_id);
    if (!WriteOwner(target, owner, device_args->owner_so_bytes)) return 3U;
    Log("installed %s (%llu bytes)", target.c_str(),
        static_cast<unsigned long long>(device_args->owner_so_bytes));
    return 0U;
}

}  // extern "C"
