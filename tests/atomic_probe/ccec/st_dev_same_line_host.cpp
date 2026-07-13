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
#include "../probe_host.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

constexpr size_t STORAGE_WORDS = 128;
constexpr uint32_t RESULT_BASE = 80;
constexpr uint32_t LAUNCHES = 20;
constexpr uint32_t CHECKS_PER_LAUNCH = 200;

static void Check(aclError error, const char *label)
{
    if (!atomic_probe::CheckAcl(error, label, __FILE__, __LINE__)) std::exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <kernel.o>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int32_t device = atomic_probe::DeviceId();
    if (device < 0) return EXIT_FAILURE;
    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file) return EXIT_FAILURE;
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> binary(size);
    file.read(binary.data(), static_cast<std::streamsize>(size));
    if (!file) return EXIT_FAILURE;

    aclrtBinHandle binary_handle;
    Check(atomic_probe::LoadAicoreBinaryFromData(binary.data(), size, &binary_handle),
          "LoadAicoreBinaryFromData");
    aclrtFuncHandle function_handle;
    Check(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function_handle),
          "aclrtBinaryGetFunctionByEntry");

    void *device_storage = nullptr;
    Check(aclrtMalloc(&device_storage, STORAGE_WORDS * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
          "aclrtMalloc");

    uint32_t total_same_errors = 0;
    uint32_t total_separate_errors = 0;
    uint32_t total_ordered_errors = 0;
    atomic_probe::Result result;
    for (uint32_t launch = 0; launch < LAUNCHES; launch++) {
        uint32_t host[STORAGE_WORDS] = {0};
        Check(aclrtMemcpy(device_storage, sizeof(host), host, sizeof(host), ACL_MEMCPY_HOST_TO_DEVICE),
              "aclrtMemcpy init");
        uint64_t args = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device_storage));
        Check(aclrtLaunchKernelWithHostArgs(function_handle, 2, stream, nullptr,
                                            &args, sizeof(args), nullptr, 0),
              "aclrtLaunchKernelWithHostArgs");
        Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
        Check(aclrtMemcpy(host, sizeof(host), device_storage, sizeof(host), ACL_MEMCPY_DEVICE_TO_HOST),
              "aclrtMemcpy result");

        bool control_ok = host[RESULT_BASE + 5] == 100u &&
                          host[RESULT_BASE + 6] == 0x3510C1A5u &&
                          host[RESULT_BASE + 7] == 2u &&
                          host[RESULT_BASE + 8] == 0xA1A00000u &&
                          host[RESULT_BASE + 9] == 0xA1A00001u;
        char label[96];
        std::snprintf(label, sizeof(label), "same-line participation launch=%u", launch);
        result.Expect(control_ok, label);

        total_same_errors += host[RESULT_BASE + 0];
        total_separate_errors += host[RESULT_BASE + 1];
        total_ordered_errors += host[RESULT_BASE + 2];
        std::printf("launch=%u same=%u/%u separate=%u/%u ordered=%u/%u first=0x%08x expected=0x%08x\n",
                    launch, host[RESULT_BASE + 0], CHECKS_PER_LAUNCH,
                    host[RESULT_BASE + 1], CHECKS_PER_LAUNCH,
                    host[RESULT_BASE + 2], CHECKS_PER_LAUNCH,
                    host[RESULT_BASE + 3], host[RESULT_BASE + 4]);
    }

    std::printf("aggregate same=%u/%u separate=%u/%u ordered=%u/%u\n",
                total_same_errors, LAUNCHES * CHECKS_PER_LAUNCH,
                total_separate_errors, LAUNCHES * CHECKS_PER_LAUNCH,
                total_ordered_errors, LAUNCHES * CHECKS_PER_LAUNCH);
    result.Expect(total_same_errors == 0, "two AIVs same line keep exact final values");
    result.Expect(total_separate_errors == 0, "two AIVs on separate cache lines exact");
    result.Expect(total_ordered_errors == 0, "two AIVs same line with per-round DSB exact");

    Check(aclrtFree(device_storage), "aclrtFree");
    Check(aclrtBinaryUnLoad(binary_handle), "aclrtBinaryUnLoad");
    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    Check(aclrtResetDevice(device), "aclrtResetDevice");
    Check(aclFinalize(), "aclFinalize");
    return result.ExitCode();
}
