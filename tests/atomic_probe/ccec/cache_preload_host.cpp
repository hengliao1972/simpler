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
#include "../cache_preload_shared.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct KernelArgs {
    uint64_t state_pointer;
};

struct Sample {
    cache_preload::ProbeResult result{};
};

static_assert(sizeof(KernelArgs) == sizeof(uint64_t), "unexpected CCEC kernel argument ABI");

bool Check(aclError error, const char *expression) {
    return atomic_probe::CheckAcl(error, expression, __FILE__, __LINE__);
}

std::vector<char> ReadBinary(const std::string &path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return {};
    const std::streamsize size = stream.tellg();
    if (size <= 0) return {};
    stream.seekg(0);
    std::vector<char> data(static_cast<size_t>(size));
    if (!stream.read(data.data(), size)) return {};
    return data;
}

const char *ModeName(cache_preload::Mode mode) {
    switch (mode) {
    case cache_preload::Mode::DCacheBaseline:
        return "dcache-baseline";
    case cache_preload::Mode::DCachePreload:
        return "dcache-preload";
    case cache_preload::Mode::ICacheColdBaseline:
        return "icache-cold";
    case cache_preload::Mode::ICacheCurrentPcAsync:
        return "icache-current-pc";
    case cache_preload::Mode::ICacheCurrentPcWait:
        return "icache-wait";
    case cache_preload::Mode::Count:
        break;
    }
    return "unknown";
}

uint64_t Median(std::vector<uint64_t> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

bool RunOne(
    aclrtFuncHandle function, aclrtStream stream, void *state_device, cache_preload::Mode mode, uint32_t target_word,
    uint64_t seed, Sample *sample
) {
    cache_preload::ProbeControl control{};
    control.mode = static_cast<uint32_t>(mode);
    control.target_word = target_word;
    control.gap_rounds = cache_preload::kGapRounds;
    control.seed = seed;
    cache_preload::ProbeResult zero{};

    auto *state_bytes = reinterpret_cast<uint8_t *>(state_device);
    void *control_device = state_bytes + offsetof(cache_preload::ProbeState, control);
    void *result_device = state_bytes + offsetof(cache_preload::ProbeState, result);
    if (!Check(
            aclrtMemcpy(control_device, sizeof(control), &control, sizeof(control), ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy(H2D control)"
        ) ||
        !Check(
            aclrtMemcpy(result_device, sizeof(zero), &zero, sizeof(zero), ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy(H2D result reset)"
        )) {
        return false;
    }

    KernelArgs args{reinterpret_cast<uint64_t>(state_device)};
    if (!Check(
            aclrtLaunchKernelWithHostArgs(function, 1, stream, nullptr, &args, sizeof(args), nullptr, 0),
            "aclrtLaunchKernelWithHostArgs"
        ) ||
        !Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream") ||
        !Check(
            aclrtMemcpy(
                &sample->result, sizeof(sample->result), result_device, sizeof(sample->result),
                ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "aclrtMemcpy(D2H result)"
        )) {
        return false;
    }
    return true;
}

bool ValidateSample(
    const Sample &sample, cache_preload::Mode mode, uint32_t target_word, uint64_t seed, std::string *reason
) {
    const cache_preload::ProbeResult &result = sample.result;
    if (result.mode_echo != static_cast<uint32_t>(mode)) {
        *reason = "mode-echo";
        return false;
    }
    if (result.target_word_echo != target_word) {
        *reason = "target-word-echo";
        return false;
    }
    if (result.gap_rounds_echo != cache_preload::kGapRounds) {
        *reason = "gap-rounds-echo";
        return false;
    }
    const uint64_t expected_gap = cache_preload::GapOracle(seed, cache_preload::kGapRounds);
    if (result.gap_checksum != expected_gap) {
        *reason = "gap-checksum";
        return false;
    }

    if (mode == cache_preload::Mode::DCacheBaseline || mode == cache_preload::Mode::DCachePreload) {
        if (result.value != cache_preload::DataValue(target_word)) {
            *reason = "dcache-load-value";
            return false;
        }
        if (result.preparation_checksum != 0 || result.icache_immediate_status != 0 ||
            result.icache_final_status != 0 || result.icache_polls != 0) {
            *reason = "dcache-unexpected-icache-fields";
            return false;
        }
    } else {
        if (result.preparation_checksum != cache_preload::ICacheEvictorOracle(seed)) {
            *reason = "icache-evictor-checksum";
            return false;
        }
        if (result.value != cache_preload::ICacheTargetOracle(seed ^ expected_gap)) {
            *reason = "icache-target-checksum";
            return false;
        }
        if (mode == cache_preload::Mode::ICacheCurrentPcWait &&
            (result.icache_final_status != 0 || result.icache_polls >= cache_preload::kICachePollLimit)) {
            *reason = "icache-wait-did-not-reach-idle";
            return false;
        }
    }

    if (result.access_or_work_ticks == 0 || result.total_ticks == 0) {
        *reason = "zero-timing-window";
        return false;
    }
    return true;
}

void PrintSummary(cache_preload::Mode mode, const std::vector<Sample> &samples) {
    std::vector<uint64_t> issue;
    std::vector<uint64_t> access_or_work;
    std::vector<uint64_t> total;
    std::vector<uint64_t> polls;
    uint32_t immediate_busy = 0;
    uint32_t final_busy = 0;
    for (const Sample &sample : samples) {
        issue.push_back(sample.result.issue_ticks);
        access_or_work.push_back(sample.result.access_or_work_ticks);
        total.push_back(sample.result.total_ticks);
        polls.push_back(sample.result.icache_polls);
        immediate_busy += sample.result.icache_immediate_status != 0 ? 1U : 0U;
        final_busy += sample.result.icache_final_status != 0 ? 1U : 0U;
    }
    std::printf(
        "%-19s issue=%6llu access/work=%6llu total=%6llu polls=%5llu "
        "immediate_busy=%u/%zu final_busy=%u/%zu\n",
        ModeName(mode), static_cast<unsigned long long>(Median(std::move(issue))),
        static_cast<unsigned long long>(Median(std::move(access_or_work))),
        static_cast<unsigned long long>(Median(std::move(total))),
        static_cast<unsigned long long>(Median(std::move(polls))), immediate_busy, samples.size(), final_busy,
        samples.size()
    );
}

}  // namespace

int main(int argc, char **argv) {
    const std::string kernel_path = argc > 1 ? argv[1] : "./cache_preload_kernel.o";
    if (argc > 2) {
        std::fprintf(stderr, "Usage: %s [cache_preload_kernel.o]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const std::vector<char> kernel_data = ReadBinary(kernel_path);
    if (kernel_data.empty()) {
        std::fprintf(stderr, "Cannot read kernel binary: %s\n", kernel_path.c_str());
        return EXIT_FAILURE;
    }
    const int32_t device = atomic_probe::DeviceId();
    if (device < 0) return EXIT_FAILURE;

    if (!Check(aclInit(nullptr), "aclInit") || !Check(aclrtSetDevice(device), "aclrtSetDevice")) {
        return EXIT_FAILURE;
    }
    aclrtStream stream = nullptr;
    if (!Check(aclrtCreateStream(&stream), "aclrtCreateStream")) {
        return EXIT_FAILURE;
    }

    aclrtBinHandle binary_handle = nullptr;
    if (!Check(
            atomic_probe::LoadAicoreBinaryFromData(kernel_data.data(), kernel_data.size(), &binary_handle),
            "LoadAicoreBinaryFromData"
        )) {
        return EXIT_FAILURE;
    }
    aclrtFuncHandle function = nullptr;
    if (!Check(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function), "aclrtBinaryGetFunctionByEntry")) {
        return EXIT_FAILURE;
    }

    void *state_device = nullptr;
    if (!Check(
            aclrtMalloc(&state_device, sizeof(cache_preload::ProbeState), ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc(probe state)"
        )) {
        return EXIT_FAILURE;
    }

    auto initial = std::make_unique<cache_preload::ProbeState>();
    for (uint32_t index = 0; index < cache_preload::kDataWords; ++index) {
        initial->data[index] = cache_preload::DataValue(index);
    }
    if (!Check(
            aclrtMemcpy(state_device, sizeof(*initial), initial.get(), sizeof(*initial), ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy(H2D initial state)"
        )) {
        return EXIT_FAILURE;
    }

    constexpr std::array<cache_preload::Mode, 5> kModes = {
        cache_preload::Mode::DCacheBaseline,      cache_preload::Mode::DCachePreload,
        cache_preload::Mode::ICacheColdBaseline,  cache_preload::Mode::ICacheCurrentPcAsync,
        cache_preload::Mode::ICacheCurrentPcWait,
    };
    std::array<std::vector<Sample>, 5> samples;
    bool launch_ok = true;
    bool semantic_ok = true;
    for (uint32_t sample_index = 0; sample_index < cache_preload::kSamples; ++sample_index) {
        const uint32_t target_word = cache_preload::kTargetStartWord + sample_index * cache_preload::kTargetStrideWords;
        const uint64_t seed = 0x123456789abcdef0ULL ^ (static_cast<uint64_t>(sample_index) * 0x9e3779b97f4a7c15ULL);

        for (const cache_preload::Mode mode : kModes) {
            Sample sample;
            launch_ok = RunOne(function, stream, state_device, mode, target_word, seed, &sample);
            if (!launch_ok) break;

            std::string reason;
            if (!ValidateSample(sample, mode, target_word, seed, &reason)) {
                std::fprintf(
                    stderr,
                    "[MISMATCH] sample=%u mode=%s reason=%s value=0x%llx "
                    "issue=%llu access/work=%llu total=%llu immediate=%llu "
                    "final=%llu polls=%llu\n",
                    sample_index, ModeName(mode), reason.c_str(), static_cast<unsigned long long>(sample.result.value),
                    static_cast<unsigned long long>(sample.result.issue_ticks),
                    static_cast<unsigned long long>(sample.result.access_or_work_ticks),
                    static_cast<unsigned long long>(sample.result.total_ticks),
                    static_cast<unsigned long long>(sample.result.icache_immediate_status),
                    static_cast<unsigned long long>(sample.result.icache_final_status),
                    static_cast<unsigned long long>(sample.result.icache_polls)
                );
                semantic_ok = false;
            }
            samples[static_cast<uint32_t>(mode)].push_back(sample);
        }
        if (!launch_ok) break;
    }

    atomic_probe::Result result;
    result.Expect(launch_ok, "CCEC cache preload launches and result copies");
    result.Expect(semantic_ok, "CCEC cache preload values and status contract");

    std::printf(
        "=== A5 pure CCEC Cache Preload Usage Probe ===\n"
        "kernel=%s bytes=%zu samples_per_mode=%u gap_rounds=%u "
        "icache_preload_units=%u\n"
        "All timing values are raw SYS_CNT deltas; they are observations, "
        "not API guarantees.\n",
        kernel_path.c_str(), kernel_data.size(), cache_preload::kSamples, cache_preload::kGapRounds,
        cache_preload::kICachePreloadUnits
    );
    for (const cache_preload::Mode mode : kModes) {
        const auto &mode_samples = samples[static_cast<uint32_t>(mode)];
        if (!mode_samples.empty()) PrintSummary(mode, mode_samples);
    }

    bool cleanup_ok = true;
    cleanup_ok &= Check(aclrtFree(state_device), "aclrtFree(probe state)");
    cleanup_ok &= Check(aclrtBinaryUnLoad(binary_handle), "aclrtBinaryUnLoad");
    cleanup_ok &= Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    cleanup_ok &= Check(aclrtResetDevice(device), "aclrtResetDevice");
    cleanup_ok &= Check(aclFinalize(), "aclFinalize");
    result.Expect(cleanup_ok, "CCEC cache preload cleanup");
    return result.ExitCode();
}
