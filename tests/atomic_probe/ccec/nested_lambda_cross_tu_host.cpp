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
#include "nested_lambda_cross_tu_layout.h"
#include "../probe_host.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

using nested_lambda_cross_tu_probe::Field;
using nested_lambda_cross_tu_probe::Variant;

struct KernelArgs {
    uint64_t storage_pointer;
};

uint32_t ReadField(const std::vector<uint32_t> &storage, Field field) {
    return storage[nested_lambda_cross_tu_probe::FieldIndex(field)];
}

bool ParseVariant(const char *text, Variant *variant) {
    for (uint32_t raw = 0; raw < static_cast<uint32_t>(Variant::Count); raw++) {
        const Variant candidate = static_cast<Variant>(raw);
        if (std::strcmp(text, nested_lambda_cross_tu_probe::VariantName(candidate)) == 0) {
            *variant = candidate;
            return true;
        }
    }
    return false;
}

void Validate(const std::vector<uint32_t> &storage, Variant variant, atomic_probe::Result &result) {
    const uint64_t checksum = static_cast<uint64_t>(ReadField(storage, Field::ChecksumLow)) |
                              (static_cast<uint64_t>(ReadField(storage, Field::ChecksumHigh)) << 32);
    const uint32_t args_bytes = ReadField(storage, Field::L0TaskArgsBytes);
    bool exact = true;
    exact &= ReadField(storage, Field::CompletedRounds) == nested_lambda_cross_tu_probe::kRounds;
    exact &= ReadField(storage, Field::MismatchCount) == 0;
    exact &=
        ReadField(storage, Field::DispatcherCalls) == nested_lambda_cross_tu_probe::ExpectedDispatcherCalls(variant);
    exact &= ReadField(storage, Field::AddressMaterializations) ==
             nested_lambda_cross_tu_probe::ExpectedMaterializations(variant);
    exact &= checksum == nested_lambda_cross_tu_probe::ExpectedTotalChecksum();
    exact &= args_bytes == nested_lambda_cross_tu_probe::kExpectedL0TaskArgsBytes;
    exact &= ReadField(storage, Field::VariantEcho) == static_cast<uint32_t>(variant);

    char label[120];
    std::snprintf(
        label, sizeof(label), "CCEC AIC caller-capture variant=%s", nested_lambda_cross_tu_probe::VariantName(variant)
    );
    result.Expect(exact, label);
    std::printf(
        "[VALUES] rounds=%u mismatches=%u dispatches=%u materializations=%u "
        "checksum=0x%016llx L0TaskArgs=%uB\n",
        ReadField(storage, Field::CompletedRounds), ReadField(storage, Field::MismatchCount),
        ReadField(storage, Field::DispatcherCalls), ReadField(storage, Field::AddressMaterializations),
        static_cast<unsigned long long>(checksum), args_bytes
    );
}

}  // namespace

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <kernel.o> <variant>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *kernel_path = argv[1];
    Variant variant = Variant::Count;
    if (!ParseVariant(argv[2], &variant)) {
        std::fprintf(stderr, "Unknown caller-capture variant: %s\n", argv[2]);
        return EXIT_FAILURE;
    }

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
    PROBE_ACL_CHECK(atomic_probe::LoadAicoreBinaryFromData(binary.data(), binary.size(), &binary_handle));
    const size_t storage_bytes = nested_lambda_cross_tu_probe::kStorageWords * sizeof(uint32_t);
    void *storage_device = nullptr;
    PROBE_ACL_CHECK(aclrtMalloc(&storage_device, storage_bytes, ACL_MEM_MALLOC_HUGE_FIRST));

    std::printf("=== Pure CCEC AIC Caller-Capture Call-Boundary Probe ===\n");
    atomic_probe::Result result;
    aclrtFuncHandle function_handle = nullptr;
    PROBE_ACL_CHECK(aclrtBinaryGetFunctionByEntry(
        binary_handle, nested_lambda_cross_tu_probe::KernelEntry(variant), &function_handle
    ));
    std::vector<uint32_t> storage(nested_lambda_cross_tu_probe::kStorageWords, 0);
    PROBE_ACL_CHECK(
        aclrtMemcpy(storage_device, storage_bytes, storage.data(), storage_bytes, ACL_MEMCPY_HOST_TO_DEVICE)
    );
    KernelArgs args{reinterpret_cast<uint64_t>(storage_device)};
    PROBE_ACL_CHECK(
        aclrtLaunchKernelWithHostArgs(function_handle, 1, stream, nullptr, &args, sizeof(args), nullptr, 0)
    );
    PROBE_ACL_CHECK(aclrtSynchronizeStream(stream));
    PROBE_ACL_CHECK(
        aclrtMemcpy(storage.data(), storage_bytes, storage_device, storage_bytes, ACL_MEMCPY_DEVICE_TO_HOST)
    );
    Validate(storage, variant, result);

    bool cleanup_ok = true;
    cleanup_ok &= atomic_probe::CheckAcl(aclrtFree(storage_device), "aclrtFree(storage_device)", __FILE__, __LINE__);
    cleanup_ok &= atomic_probe::CheckAcl(
        aclrtBinaryUnLoad(binary_handle), "aclrtBinaryUnLoad(binary_handle)", __FILE__, __LINE__
    );
    cleanup_ok &= atomic_probe::CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream(stream)", __FILE__, __LINE__);
    cleanup_ok &=
        atomic_probe::CheckAcl(aclrtResetDevice(device_id), "aclrtResetDevice(device_id)", __FILE__, __LINE__);
    cleanup_ok &= atomic_probe::CheckAcl(aclFinalize(), "aclFinalize()", __FILE__, __LINE__);
    result.Expect(cleanup_ok, "CCEC caller-capture ACL cleanup");
    return result.ExitCode();
}
