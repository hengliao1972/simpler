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
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

constexpr uint32_t STORAGE_WORDS = 512;
constexpr uint32_t CONTROL_WORD = 256;
constexpr uint32_t MARKER_WORD = CONTROL_WORD + 16;
constexpr uint32_t MATRIX_ROUNDS = 257;
constexpr uint32_t DONE_MAGIC = 0xCCEC3510u;

static void check(aclError error, const char *label)
{
    if (!atomic_probe::CheckAcl(error, label, __FILE__, __LINE__)) std::exit(EXIT_FAILURE);
}

template <typename T>
static T matrix_value(uint32_t participant, uint32_t round)
{
    uint64_t value = 0xA5A5A5A5A5A5A5A5ULL;
    value += (uint64_t)(participant + 1) * 0x0101010101010101ULL;
    value += round;
    return static_cast<T>(value);
}

template <typename T>
static uint32_t host_data_errors(const uint8_t *storage, uint32_t mode, uint32_t participants)
{
    uint32_t errors = 0;
    for (uint32_t participant = 0; participant < participants; participant++) {
        uint32_t offset = mode >= 4 ? participant * 64 : participant * sizeof(T);
        T actual;
        std::memcpy(&actual, storage + offset, sizeof(T));
        if (actual != matrix_value<T>(participant, MATRIX_ROUNDS - 1)) errors++;
    }
    return errors;
}

template <typename T>
static void print_host_values(const uint8_t *storage, uint32_t mode, uint32_t participants)
{
    std::printf("  values:");
    for (uint32_t participant = 0; participant < participants; participant++) {
        uint32_t offset = mode >= 4 ? participant * 64 : participant * sizeof(T);
        T actual;
        std::memcpy(&actual, storage + offset, sizeof(T));
        T expected = matrix_value<T>(participant, MATRIX_ROUNDS - 1);
        std::printf(" [%u]=0x%016lx/0x%016lx", participant,
                    (unsigned long)(uint64_t)actual, (unsigned long)(uint64_t)expected);
    }
    std::printf("\n");
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <kernel.o> <aiv|mix>\n", argv[0]);
        return EXIT_FAILURE;
    }
    std::string variant = argv[2];
    if (variant != "aiv" && variant != "mix") {
        std::fprintf(stderr, "Unknown matrix variant: %s\n", argv[2]);
        return EXIT_FAILURE;
    }
    bool aiv_only = variant == "aiv";

    int32_t device_id = atomic_probe::DeviceId();
    if (device_id < 0) return EXIT_FAILURE;
    check(aclInit(nullptr), "aclInit");
    check(aclrtSetDevice(device_id), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    check(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "Cannot open %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    size_t binary_size = file.tellg();
    file.seekg(0);
    std::vector<char> binary(binary_size);
    file.read(binary.data(), binary_size);
    if (!file) {
        std::fprintf(stderr, "Cannot read %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    aclrtBinHandle binary_handle;
    check(atomic_probe::LoadAicoreBinaryFromData(binary.data(), binary_size, &binary_handle),
          "LoadAicoreBinaryFromData");
    aclrtFuncHandle function_handle;
    check(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function_handle),
          "aclrtBinaryGetFunctionByEntry");
    void *storage_device = nullptr;
    check(aclrtMalloc(&storage_device, STORAGE_WORDS * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
          "aclrtMalloc");

    struct KernelArgs {
        uint64_t storage_pointer;
        uint32_t mode;
        uint32_t num_blocks;
    } args = {(uint64_t)(uintptr_t)storage_device, 0, 0};
    static_assert(sizeof(KernelArgs) == 16, "unexpected CCEC kernel argument ABI");

    const char *width_names[] = {"1B", "2B", "4B", "8B"};
    const uint32_t block_counts[] = {2, 4};
    atomic_probe::Result result;
    std::printf("=== CCEC Exact Cache-Line Matrix: %s ===\n", aiv_only ? "AIV" : "AIC+AIV");

    uint32_t first_mode = 0;
    uint32_t last_mode = 8;
    const char *mode_filter = std::getenv("ATOMIC_PROBE_MATRIX_MODE");
    if (mode_filter == nullptr || mode_filter[0] == '\0') {
        std::fprintf(stderr, "ATOMIC_PROBE_MATRIX_MODE=0..7 is required; run_all.sh enumerates all modes\n");
        return EXIT_FAILURE;
    } else {
        char *end = nullptr;
        unsigned long selected = std::strtoul(mode_filter, &end, 10);
        if (end == mode_filter || *end != '\0' || selected >= 8) {
            std::fprintf(stderr, "Invalid ATOMIC_PROBE_MATRIX_MODE: %s\n", mode_filter);
            return EXIT_FAILURE;
        }
        first_mode = (uint32_t)selected;
        last_mode = first_mode + 1;
    }

    for (uint32_t blocks : block_counts) {
        uint32_t participants = blocks * (aiv_only ? 1 : 2);
        uint32_t expected_aic = aiv_only ? 0 : blocks;
        uint32_t expected_aiv = blocks;

        for (uint32_t mode = first_mode; mode < last_mode; mode++) {
            uint32_t host[STORAGE_WORDS] = {0};
            check(aclrtMemcpy(storage_device, sizeof(host), host, sizeof(host), ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy init");

            args.mode = mode;
            args.num_blocks = blocks;
            check(aclrtLaunchKernelWithHostArgs(function_handle, blocks, stream, nullptr,
                                                &args, sizeof(args), nullptr, 0),
                  "aclrtLaunchKernelWithHostArgs");
            check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
            check(aclrtMemcpy(host, sizeof(host), storage_device, sizeof(host), ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy result");

            uint32_t direct_errors = 0;
            const uint8_t *bytes = reinterpret_cast<const uint8_t *>(host);
            switch (mode & 3u) {
                case 0: direct_errors = host_data_errors<uint8_t>(bytes, mode, participants); break;
                case 1: direct_errors = host_data_errors<uint16_t>(bytes, mode, participants); break;
                case 2: direct_errors = host_data_errors<uint32_t>(bytes, mode, participants); break;
                default: direct_errors = host_data_errors<uint64_t>(bytes, mode, participants); break;
            }

            uint32_t *control = &host[CONTROL_WORD];
            bool pass = control[1] == participants && control[2] == expected_aic &&
                        control[3] == expected_aiv && control[4] == 0 &&
                        control[5] == participants && control[6] == DONE_MAGIC &&
                        control[7] == mode && control[8] == 0 && direct_errors == 0;
            std::printf("[%s %-9s] blocks=%u participants=%u (AIC=%u AIV=%u) data_err=%u marker_err=%u host_err=%u %s\n",
                        width_names[mode & 3u], mode < 4 ? "same-line" : "separate", blocks,
                        control[1], control[2], control[3], control[4], control[8], direct_errors,
                        pass ? "PASS" : "FAIL");
            if (!pass) {
                std::printf("  control: expected participants=%u AIC=%u AIV=%u mode=%u; got tail=%u magic=0x%08x mode=%u\n",
                            participants, expected_aic, expected_aiv, mode, control[5], control[6], control[7]);
                std::printf("  markers:");
                for (uint32_t p = 0; p < participants; p++) {
                    std::printf(" [%u]=0x%08x", p, host[MARKER_WORD + p]);
                }
                std::printf("\n");
                switch (mode & 3u) {
                    case 0: print_host_values<uint8_t>(bytes, mode, participants); break;
                    case 1: print_host_values<uint16_t>(bytes, mode, participants); break;
                    case 2: print_host_values<uint32_t>(bytes, mode, participants); break;
                    default: print_host_values<uint64_t>(bytes, mode, participants); break;
                }
            }
            char assertion[112];
            std::snprintf(assertion, sizeof(assertion), "CCEC %s %s %s blocks=%u exact",
                          aiv_only ? "AIV" : "MIX", width_names[mode & 3u],
                          mode < 4 ? "same-line" : "separate", blocks);
            result.Expect(pass, assertion);
        }
    }

    check(aclrtFree(storage_device), "aclrtFree");
    check(aclrtBinaryUnLoad(binary_handle), "aclrtBinaryUnLoad");
    check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    check(aclrtResetDevice(device_id), "aclrtResetDevice");
    check(aclFinalize(), "aclFinalize");
    return result.ExitCode();
}
