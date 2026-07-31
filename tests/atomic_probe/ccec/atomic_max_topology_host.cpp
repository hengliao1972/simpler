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

// A5 atomicMax 地址拓扑探针 host。
//
// 固定覆盖 Claim 的三个实际竞争人口 24/32/64，并比较：
//   flat     ：所有 AIV 访问同一根节点；
//   grouped  ：只做分组第一层，用于确认地址分流是否真实生效；
//   two-level：每组一个局部 winner，再由组 winner 竞争根节点。
//
// 不把 CPU 原子性能当作 A5 代理；本文件只负责启动 AIV kernel、核对每轮
// winner 数量，并输出多次运行的中位数。
#include "../probe_host.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kModeFlat = 0;
constexpr uint32_t kModeGrouped = 1;
constexpr uint32_t kModeTwoLevel = 2;
constexpr uint32_t kOpFetchMax = 0;
constexpr uint32_t kOpCompareExchange = 1;
constexpr uint32_t kOpExchange = 2;
constexpr uint32_t kConfigMagic = 0x41544d58U;
constexpr uint32_t kRounds = 128;
constexpr uint32_t kRepeats = 7;
constexpr size_t kStorageBytes = 65536;

struct alignas(64) ProbeResult {
    uint64_t candidate_ticks;
    uint64_t candidate_max_ticks;
    uint64_t loop_ticks;
    uint64_t checksum;
    uint32_t local_wins;
    uint32_t root_wins;
    uint32_t participant;
    uint32_t completed_rounds;
    uint8_t padding[16];
};
static_assert(sizeof(ProbeResult) == 64, "one result must occupy one cache line");

struct KernelArgs {
    uint64_t storage_pointer;
    uint64_t results_pointer;
};
static_assert(sizeof(KernelArgs) == 16, "unexpected CCEC kernel argument ABI");

struct alignas(64) ProbeConfig {
    uint32_t magic;
    uint32_t mode;
    uint32_t group_count;
    uint32_t stride_bytes;
    uint32_t rounds;
    uint32_t operation;
    uint8_t padding[40];
};
static_assert(sizeof(ProbeConfig) == 64, "config must occupy one cache line");

struct Variant {
    uint32_t participants;
    uint32_t mode;
    uint32_t groups;
    uint32_t stride;
    uint32_t operation;
};

struct Sample {
    double candidate_mean_ticks;
    double loop_ticks_per_round;
    uint64_t candidate_max_ticks;
    bool exact;
    bool per_core_valid;
    uint64_t local_wins;
    uint64_t root_wins;
    uint64_t expected_local;
    uint64_t expected_root;
};

void Check(aclError error, const char *message) {
    if (error != ACL_SUCCESS) {
        std::fprintf(stderr, "ACL error %d: %s\n", error, message);
        std::exit(EXIT_FAILURE);
    }
}

double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

uint64_t Median(std::vector<uint64_t> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

const char *ModeName(uint32_t mode) {
    if (mode == kModeFlat) return "flat";
    if (mode == kModeGrouped) return "grouped";
    return "two-level";
}

const char *OperationName(uint32_t operation) {
    if (operation == kOpCompareExchange) return "cas";
    if (operation == kOpExchange) return "exchange";
    return "fetch-max";
}

Sample RunOnce(
    aclrtFuncHandle function, aclrtStream stream, void *storage_device, void *results_device, const Variant &variant
) {
    std::vector<uint8_t> storage(kStorageBytes, 0);
    std::vector<ProbeResult> host_results(variant.participants);
    const ProbeConfig config{
        kConfigMagic, variant.mode, variant.groups, variant.stride, kRounds, variant.operation, {},
    };
    std::memcpy(storage.data(), &config, sizeof(config));
    Check(
        aclrtMemcpy(storage_device, kStorageBytes, storage.data(), storage.size(), ACL_MEMCPY_HOST_TO_DEVICE),
        "initialize atomic topology storage"
    );
    Check(
        aclrtMemset(results_device, sizeof(ProbeResult) * 64U, 0, sizeof(ProbeResult) * 64U),
        "initialize atomic topology results"
    );

    KernelArgs args{
        reinterpret_cast<uint64_t>(storage_device),
        reinterpret_cast<uint64_t>(results_device),
    };
    Check(
        aclrtLaunchKernelWithHostArgs(function, variant.participants, stream, nullptr, &args, sizeof(args), nullptr, 0),
        "launch atomic topology kernel"
    );
    Check(aclrtSynchronizeStream(stream), "synchronize atomic topology kernel");
    Check(
        aclrtMemcpy(
            host_results.data(), sizeof(ProbeResult) * variant.participants, results_device,
            sizeof(ProbeResult) * variant.participants, ACL_MEMCPY_DEVICE_TO_HOST
        ),
        "copy atomic topology results"
    );

    uint64_t candidate_ticks = 0;
    uint64_t candidate_max_ticks = 0;
    uint64_t loop_max_ticks = 0;
    uint64_t local_wins = 0;
    uint64_t root_wins = 0;
    uint64_t checksum = 0;
    bool per_core_valid = true;
    for (uint32_t participant = 0; participant < variant.participants; ++participant) {
        const ProbeResult &core = host_results[participant];
        candidate_ticks += core.candidate_ticks;
        candidate_max_ticks = std::max(candidate_max_ticks, core.candidate_max_ticks);
        loop_max_ticks = std::max(loop_max_ticks, core.loop_ticks);
        local_wins += core.local_wins;
        root_wins += core.root_wins;
        checksum ^= core.checksum + participant;
        per_core_valid &= core.participant == participant;
        per_core_valid &= core.completed_rounds == kRounds;
    }

    const uint64_t expected_local = variant.mode == kModeFlat ? 0ULL : static_cast<uint64_t>(variant.groups) * kRounds;
    const uint64_t expected_root = variant.mode == kModeGrouped ? 0ULL : kRounds;
    // checksum 只用于强制消费 atomic 返回值；不同核异或后允许自然抵消为
    // 零，不能把聚合 checksum 非零误当成协议正确性条件。
    const bool exact = per_core_valid && local_wins == expected_local && root_wins == expected_root;
    return Sample{
        static_cast<double>(candidate_ticks) / (static_cast<double>(variant.participants) * kRounds),
        static_cast<double>(loop_max_ticks) / kRounds,
        candidate_max_ticks,
        exact,
        per_core_valid,
        local_wins,
        root_wins,
        expected_local,
        expected_root,
    };
}

void RunVariant(
    aclrtFuncHandle function, aclrtStream stream, void *storage_device, void *results_device, const Variant &variant,
    atomic_probe::Result &result
) {
    std::vector<double> candidate_means;
    std::vector<double> loop_per_round;
    std::vector<uint64_t> candidate_maxima;
    bool all_exact = true;
    Sample first_failure{};
    bool has_failure = false;
    for (uint32_t repeat = 0; repeat < kRepeats; ++repeat) {
        const Sample sample = RunOnce(function, stream, storage_device, results_device, variant);
        candidate_means.push_back(sample.candidate_mean_ticks);
        loop_per_round.push_back(sample.loop_ticks_per_round);
        candidate_maxima.push_back(sample.candidate_max_ticks);
        all_exact &= sample.exact;
        if (!sample.exact && !has_failure) {
            first_failure = sample;
            has_failure = true;
        }
    }
    if (has_failure) {
        std::fprintf(
            stderr,
            "[DETAIL] N=%u op=%s mode=%s G=%u stride=%u "
            "per_core_valid=%u local=%llu/%llu root=%llu/%llu\n",
            variant.participants, OperationName(variant.operation), ModeName(variant.mode), variant.groups,
            variant.stride, first_failure.per_core_valid ? 1U : 0U,
            static_cast<unsigned long long>(first_failure.local_wins),
            static_cast<unsigned long long>(first_failure.expected_local),
            static_cast<unsigned long long>(first_failure.root_wins),
            static_cast<unsigned long long>(first_failure.expected_root)
        );
    }
    char assertion[192];
    std::snprintf(
        assertion, sizeof(assertion), "atomic topology exact N=%u op=%s mode=%s G=%u stride=%u", variant.participants,
        OperationName(variant.operation), ModeName(variant.mode), variant.groups, variant.stride
    );
    result.Expect(all_exact, assertion);

    std::printf(
        "%2u %-8s %-9s %2u %4u %18.1f %19.1f %19llu\n", variant.participants, OperationName(variant.operation),
        ModeName(variant.mode), variant.groups, variant.stride, Median(candidate_means), Median(loop_per_round),
        static_cast<unsigned long long>(Median(candidate_maxima))
    );
}

}  // namespace

int main(int argc, char **argv) {
    const char *kernel_path = argc > 1 ? argv[1] : "./atomic_max_topology_kernel.o";

    const int32_t device = atomic_probe::DeviceId();
    if (device < 0) return EXIT_FAILURE;
    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");

    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::ifstream binary_file(kernel_path, std::ios::binary | std::ios::ate);
    if (!binary_file) {
        std::fprintf(stderr, "Cannot open %s\n", kernel_path);
        return EXIT_FAILURE;
    }
    const size_t binary_size = static_cast<size_t>(binary_file.tellg());
    binary_file.seekg(0);
    std::vector<char> binary(binary_size);
    binary_file.read(binary.data(), static_cast<std::streamsize>(binary_size));

    aclrtBinHandle binary_handle = nullptr;
    Check(
        atomic_probe::LoadAicoreBinaryFromData(binary.data(), binary.size(), &binary_handle),
        "load atomic topology binary"
    );
    aclrtFuncHandle function = nullptr;
    Check(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function), "get atomic topology entry");

    void *storage_device = nullptr;
    void *results_device = nullptr;
    Check(aclrtMalloc(&storage_device, kStorageBytes, ACL_MEM_MALLOC_HUGE_FIRST), "allocate atomic topology storage");
    Check(
        aclrtMalloc(&results_device, sizeof(ProbeResult) * 64U, ACL_MEM_MALLOC_HUGE_FIRST),
        "allocate atomic topology results"
    );

    std::printf("=== A5 AIV atomicMax address-topology probe ===\n");
    std::printf("rounds=%u repeats=%u sys-counter values are reported as raw ticks\n", kRounds, kRepeats);
    std::printf(
        "%2s %-8s %-9s %2s %4s %18s %19s %19s\n", "N", "op", "mode", "G", "gap", "candidate median",
        "loop/round median", "candidate max median"
    );

    atomic_probe::Result result;
    const uint32_t populations[] = {24, 32, 64};
    for (uint32_t population : populations) {
        RunVariant(
            function, stream, storage_device, results_device, Variant{population, kModeFlat, 1, 64, kOpFetchMax}, result
        );

        const uint32_t groups[] = {
            population == 24 ? 3U : 4U,
            population == 24 ? 4U : (population == 32 ? 6U : 8U),
            population == 24 ? 6U : (population == 32 ? 8U : 16U),
        };
        const uint32_t strides[] = {64, 128, 256, 512};
        for (uint32_t group : groups) {
            for (uint32_t stride : strides) {
                RunVariant(
                    function, stream, storage_device, results_device,
                    Variant{population, kModeGrouped, group, stride, kOpFetchMax}, result
                );
                RunVariant(
                    function, stream, storage_device, results_device,
                    Variant{population, kModeTwoLevel, group, stride, kOpFetchMax}, result
                );
            }
        }

        const uint32_t selected_groups = population == 24 ? 4U : (population == 32 ? 6U : 8U);
        const uint32_t operations[] = {
            kOpCompareExchange,
            kOpExchange,
        };
        for (uint32_t operation : operations) {
            RunVariant(
                function, stream, storage_device, results_device, Variant{population, kModeFlat, 1, 512, operation},
                result
            );
            RunVariant(
                function, stream, storage_device, results_device,
                Variant{population, kModeTwoLevel, selected_groups, 512, operation}, result
            );
        }
    }

    Check(aclrtFree(results_device), "free atomic topology results");
    Check(aclrtFree(storage_device), "free atomic topology storage");
    Check(aclrtBinaryUnLoad(binary_handle), "unload atomic topology binary");
    Check(aclrtDestroyStream(stream), "destroy atomic topology stream");
    Check(aclrtResetDevice(device), "reset atomic topology device");
    Check(aclFinalize(), "aclFinalize");
    return result.ExitCode();
}
