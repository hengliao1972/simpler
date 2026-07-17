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
#include "../nested_lambda_probe.h"
#include "../probe_host.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {

using nested_lambda_probe::Field;

struct KernelArgs {
    uint64_t storage_pointer;
    uint32_t seed;
    uint32_t padding;
};

static_assert(sizeof(KernelArgs) == 16, "unexpected CCEC kernel argument ABI");

void Validate(
    const std::vector<uint32_t> &storage, uint32_t seed, atomic_probe::Result &result)
{
    bool exact = true;
    for (uint32_t raw_field = 0; raw_field < static_cast<uint32_t>(Field::Count); raw_field++) {
        const Field field = static_cast<Field>(raw_field);
        const uint32_t actual = storage[nested_lambda_probe::FieldIndex(field)];
        const uint32_t expected = nested_lambda_probe::Expected(field, seed);
        exact &= actual == expected;
    }
    result.Expect(exact, "CCEC nested capture/template semantics");
}

} // namespace

int main(int argc, char *argv[])
{
    const char *kernel_path = argc > 1 ? argv[1] : "./nested_lambda_kernel.o";
    const int32_t device_id = atomic_probe::DeviceId();
    if (device_id < 0) return EXIT_FAILURE;

    PROBE_ACL_CHECK(aclInit(nullptr));
    PROBE_ACL_CHECK(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    PROBE_ACL_CHECK(aclrtCreateStream(&stream));

    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "Cannot open %s\n", kernel_path);
        return EXIT_FAILURE;
    }
    const size_t binary_size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> binary(binary_size);
    file.read(binary.data(), static_cast<std::streamsize>(binary_size));

    aclrtBinHandle binary_handle = nullptr;
    PROBE_ACL_CHECK(atomic_probe::LoadAicoreBinaryFromData(
        binary.data(), binary.size(), &binary_handle));
    aclrtFuncHandle function_handle = nullptr;
    PROBE_ACL_CHECK(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function_handle));

    const size_t storage_bytes = nested_lambda_probe::kStorageWords * sizeof(uint32_t);
    void *storage_device = nullptr;
    PROBE_ACL_CHECK(aclrtMalloc(&storage_device, storage_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    std::vector<uint32_t> storage(nested_lambda_probe::kStorageWords, 0);
    PROBE_ACL_CHECK(aclrtMemcpy(
        storage_device, storage_bytes, storage.data(), storage_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    constexpr uint32_t seed = nested_lambda_probe::kDefaultSeed;
    KernelArgs args{reinterpret_cast<uint64_t>(storage_device), seed, 0};
    PROBE_ACL_CHECK(aclrtLaunchKernelWithHostArgs(
        function_handle, 1, stream, nullptr, &args, sizeof(args), nullptr, 0));
    PROBE_ACL_CHECK(aclrtSynchronizeStream(stream));
    PROBE_ACL_CHECK(aclrtMemcpy(
        storage.data(), storage_bytes, storage_device, storage_bytes, ACL_MEMCPY_DEVICE_TO_HOST));

    std::printf("=== Pure CCEC Nested-Lambda/Template A5 Compiler Probe ===\n");
    atomic_probe::Result result;
    Validate(storage, seed, result);

    bool cleanup_ok = true;
    cleanup_ok &= atomic_probe::CheckAcl(
        aclrtFree(storage_device), "aclrtFree(storage_device)", __FILE__, __LINE__);
    cleanup_ok &= atomic_probe::CheckAcl(
        aclrtBinaryUnLoad(binary_handle), "aclrtBinaryUnLoad(binary_handle)", __FILE__, __LINE__);
    cleanup_ok &= atomic_probe::CheckAcl(
        aclrtDestroyStream(stream), "aclrtDestroyStream(stream)", __FILE__, __LINE__);
    cleanup_ok &= atomic_probe::CheckAcl(
        aclrtResetDevice(device_id), "aclrtResetDevice(device_id)", __FILE__, __LINE__);
    cleanup_ok &= atomic_probe::CheckAcl(aclFinalize(), "aclFinalize()", __FILE__, __LINE__);
    result.Expect(cleanup_ok, "CCEC ACL cleanup");
    return result.ExitCode();
}
