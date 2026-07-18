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

// 单 AIV、同一 target 调用点的 WARM/COLD I-cache PMU 配对 host。
// 每一对样本使用同一个 seed；WARM 在窗外预热 target，COLD 在窗外执行 32 KiB
// evictor。Host 逐样本复算两种 checksum，并要求配对样本落在同一物理 AIV。
// 最终只报告 COLD-WARM 的原始差值和比例，不在代码中预设“miss 是否计入
// scalar busy”的结论。

#include "icache_scalar_pmu_shared.h"
#include "pmu_probe_host_support.h"
#include "../probe_host.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using atomic_probe::pmu::CheckAcl;
using atomic_probe::pmu::ReadBinary;

constexpr uint32_t kMinimumRepeats = 11;
constexpr uint32_t kMaximumRepeats = 101;
constexpr uint64_t kRepeatSeedStride = 0x9e3779b97f4a7c15ULL;

const char *ModeName(icache_scalar_pmu::Mode mode)
{
    switch (mode) {
        case icache_scalar_pmu::Mode::WarmTarget: return "WARM_TARGET";
        case icache_scalar_pmu::Mode::ColdTarget: return "COLD_TARGET";
        default: return "UNKNOWN";
    }
}

bool ParseUint64(const char *text, uint64_t maximum, uint64_t *value)
{
    if (text == nullptr || text[0] == '\0') return false;
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || parsed > maximum) return false;
    *value = static_cast<uint64_t>(parsed);
    return true;
}

uint32_t RepeatsFromEnv()
{
    const char *raw = std::getenv("ICACHE_SCALAR_PMU_REPEATS");
    if (raw == nullptr || raw[0] == '\0') return kMinimumRepeats;
    uint64_t value = 0;
    if (!ParseUint64(raw, kMaximumRepeats, &value) || value < kMinimumRepeats) {
        std::fprintf(
            stderr, "ICACHE_SCALAR_PMU_REPEATS must be in %u..%u: %s\n", kMinimumRepeats,
            kMaximumRepeats, raw
        );
        return 0;
    }
    return static_cast<uint32_t>(value);
}

uint64_t SeedFromEnv(bool *ok)
{
    const char *raw = std::getenv("ICACHE_SCALAR_PMU_SEED");
    if (raw == nullptr || raw[0] == '\0') return 0x123456789abcdef0ULL;
    uint64_t value = 0;
    const bool parsed = ParseUint64(raw, std::numeric_limits<uint64_t>::max(), &value);
    *ok &= parsed;
    if (!parsed) std::fprintf(stderr, "Invalid ICACHE_SCALAR_PMU_SEED: %s\n", raw);
    return value;
}

struct Sample {
    icache_scalar_pmu::ProbeResult result{};
};

bool ValidateSample(
    const Sample &sample, icache_scalar_pmu::Mode mode, uint64_t seed, std::string *reason
)
{
    const uint64_t expected_preparation = mode == icache_scalar_pmu::Mode::WarmTarget
        ? icache_scalar_pmu::TargetOracle(seed)
        : icache_scalar_pmu::EvictorOracle(seed);
    if (sample.result.target_checksum != icache_scalar_pmu::TargetOracle(seed)) {
        *reason = "target-checksum";
        return false;
    }
    if (sample.result.preparation_checksum != expected_preparation) {
        *reason = "preparation-checksum";
        return false;
    }
    if (sample.result.mode_echo != static_cast<uint32_t>(mode)) {
        *reason = "mode-echo";
        return false;
    }
    if (sample.result.physical_core_id >= atomic_probe::pmu::kPhysicalSubcoreCount) {
        *reason = "physical-core-id";
        return false;
    }
    if ((sample.result.pmu_ctrl_after_stop & 1ULL) != 0) {
        *reason = "pmu-gate-still-enabled";
        return false;
    }
    if (sample.result.sys_cycles == 0 || sample.result.pmu_total_cycles == 0) {
        *reason = "zero-cycle-window";
        return false;
    }
    if (sample.result.pmu_icache_miss > sample.result.pmu_icache_request) {
        *reason = "icache-miss-exceeds-request";
        return false;
    }
    return true;
}

bool RunOne(
    aclrtFuncHandle function, aclrtStream stream, void *state_device, uint64_t pmu_register_bases,
    icache_scalar_pmu::Mode mode, uint32_t repeat, uint64_t seed, Sample *sample
)
{
    icache_scalar_pmu::ProbeState state{};
    state.control.pmu_register_bases = pmu_register_bases;
    state.control.mode = static_cast<uint32_t>(mode);
    state.control.seed = seed;
    if (!CheckAcl(
            aclrtMemcpy(state_device, sizeof(state), &state, sizeof(state), ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy(H2D probe state)"
        )) {
        return false;
    }

    struct KernelArgs {
        uint64_t state_pointer;
    } args{reinterpret_cast<uint64_t>(state_device)};
    static_assert(sizeof(KernelArgs) == sizeof(uint64_t), "unexpected CCEC kernel argument ABI");
    if (!CheckAcl(
            aclrtLaunchKernelWithHostArgs(function, 1, stream, nullptr, &args, sizeof(args), nullptr, 0),
            "aclrtLaunchKernelWithHostArgs"
        ) ||
        !CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(AIV probe)") ||
        !CheckAcl(
            aclrtMemcpy(&state, sizeof(state), state_device, sizeof(state), ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy(D2H probe state)"
        )) {
        return false;
    }

    sample->result = state.result;
    std::string reason;
    const bool semantic_ok = ValidateSample(*sample, mode, seed, &reason);
    std::printf(
        "[RAW] repeat=%u mode=%s seed=0x%llx sys_cycles=%llu total=%llu scalar=%llu "
        "icache_req=%llu icache_miss=%llu target=0x%llx preparation=0x%llx "
        "physical=%llu ctrl=0x%llx status=%s%s%s\n",
        repeat, ModeName(mode), static_cast<unsigned long long>(seed),
        static_cast<unsigned long long>(sample->result.sys_cycles),
        static_cast<unsigned long long>(sample->result.pmu_total_cycles),
        static_cast<unsigned long long>(sample->result.pmu_scalar_busy),
        static_cast<unsigned long long>(sample->result.pmu_icache_request),
        static_cast<unsigned long long>(sample->result.pmu_icache_miss),
        static_cast<unsigned long long>(sample->result.target_checksum),
        static_cast<unsigned long long>(sample->result.preparation_checksum),
        static_cast<unsigned long long>(sample->result.physical_core_id),
        static_cast<unsigned long long>(sample->result.pmu_ctrl_after_stop), semantic_ok ? "PASS" : "FAIL",
        semantic_ok ? "" : " reason=", semantic_ok ? "" : reason.c_str()
    );
    return semantic_ok;
}

uint64_t Median(std::vector<uint64_t> values)
{
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if ((values.size() & 1U) != 0) return values[middle];
    return values[middle - 1] + (values[middle] - values[middle - 1]) / 2;
}

double Median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return (values.size() & 1U) != 0 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
}

using CounterMember = uint64_t icache_scalar_pmu::ProbeResult::*;

double PairedDelta(
    const std::vector<Sample> &cold, const std::vector<Sample> &warm, CounterMember member
)
{
    std::vector<double> deltas;
    deltas.reserve(cold.size());
    for (size_t index = 0; index < cold.size(); ++index) {
        deltas.push_back(
            static_cast<double>(cold[index].result.*member) -
            static_cast<double>(warm[index].result.*member)
        );
    }
    return Median(std::move(deltas));
}

struct Metric {
    const char *name;
    CounterMember member;
};

constexpr Metric kMetrics[] = {
    {"sys_cycles", &icache_scalar_pmu::ProbeResult::sys_cycles},
    {"total", &icache_scalar_pmu::ProbeResult::pmu_total_cycles},
    {"scalar", &icache_scalar_pmu::ProbeResult::pmu_scalar_busy},
    {"icache_req", &icache_scalar_pmu::ProbeResult::pmu_icache_request},
    {"icache_miss", &icache_scalar_pmu::ProbeResult::pmu_icache_miss},
};

void PrintSummary(const std::array<std::vector<Sample>, 2> &samples)
{
    for (uint32_t mode_index = 0; mode_index < static_cast<uint32_t>(icache_scalar_pmu::Mode::Count);
         ++mode_index) {
        const auto mode = static_cast<icache_scalar_pmu::Mode>(mode_index);
        std::printf("[MEDIAN] mode=%s", ModeName(mode));
        for (const Metric &metric : kMetrics) {
            std::vector<uint64_t> values;
            values.reserve(samples[mode_index].size());
            for (const Sample &sample : samples[mode_index]) values.push_back(sample.result.*(metric.member));
            std::printf(" %s=%llu", metric.name, static_cast<unsigned long long>(Median(std::move(values))));
        }
        std::printf("\n");
    }

    const auto &warm = samples[static_cast<uint32_t>(icache_scalar_pmu::Mode::WarmTarget)];
    const auto &cold = samples[static_cast<uint32_t>(icache_scalar_pmu::Mode::ColdTarget)];
    for (const Metric &metric : kMetrics) {
        std::printf(
            "[PAIRED_DELTA] metric=%s cold_minus_warm=%.6f\n", metric.name,
            PairedDelta(cold, warm, metric.member)
        );
    }

    const double sys_delta = PairedDelta(cold, warm, &icache_scalar_pmu::ProbeResult::sys_cycles);
    const double total_delta = PairedDelta(cold, warm, &icache_scalar_pmu::ProbeResult::pmu_total_cycles);
    const double scalar_delta = PairedDelta(cold, warm, &icache_scalar_pmu::ProbeResult::pmu_scalar_busy);
    const double request_delta = PairedDelta(cold, warm, &icache_scalar_pmu::ProbeResult::pmu_icache_request);
    const double miss_delta = PairedDelta(cold, warm, &icache_scalar_pmu::ProbeResult::pmu_icache_miss);
    const double scalar_share = total_delta == 0.0 ? 0.0 : scalar_delta / total_delta;
    const double scalar_gap = total_delta - scalar_delta;
    const double total_per_miss = miss_delta == 0.0 ? 0.0 : total_delta / miss_delta;
    const double scalar_per_miss = miss_delta == 0.0 ? 0.0 : scalar_delta / miss_delta;
    const double gap_per_miss = miss_delta == 0.0 ? 0.0 : scalar_gap / miss_delta;
    std::printf(
        "[ICACHE_CLASSIFICATION] completion_delta_sys_cycles=%.6f pmu_total_delta_cycles=%.6f "
        "scalar_busy_delta_cycles=%.6f request_delta=%.6f miss_delta=%.6f "
        "scalar_share_of_total_delta=%.9f scalar_gap_cycles=%.6f "
        "total_cycles_per_extra_miss=%.6f scalar_cycles_per_extra_miss=%.6f "
        "gap_cycles_per_extra_miss=%.6f\n",
        sys_delta, total_delta, scalar_delta, request_delta, miss_delta, scalar_share, scalar_gap,
        total_per_miss, scalar_per_miss, gap_per_miss
    );
}

}  // namespace

int main(int argc, char **argv)
{
    const std::string kernel_path = argc > 1 ? argv[1] : "./icache_scalar_pmu_kernel.o";
    if (argc > 2) {
        std::fprintf(stderr, "Usage: %s [icache_scalar_pmu_kernel.o]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const uint32_t repeats = RepeatsFromEnv();
    bool options_ok = repeats != 0;
    const uint64_t base_seed = SeedFromEnv(&options_ok);
    if (!options_ok) return EXIT_FAILURE;

    const int32_t device = atomic_probe::DeviceId();
    if (device < 0) return EXIT_FAILURE;
    const std::vector<char> kernel_data = ReadBinary(kernel_path);
    if (kernel_data.empty()) {
        std::fprintf(stderr, "Cannot read kernel binary: %s\n", kernel_path.c_str());
        return EXIT_FAILURE;
    }

    if (!CheckAcl(aclInit(nullptr), "aclInit") || !CheckAcl(aclrtSetDevice(device), "aclrtSetDevice")) {
        return EXIT_FAILURE;
    }
    aclrtStream stream = nullptr;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) return EXIT_FAILURE;

    aclrtBinHandle binary_handle = nullptr;
    if (!CheckAcl(
            atomic_probe::LoadAicoreBinaryFromData(kernel_data.data(), kernel_data.size(), &binary_handle),
            "LoadAicoreBinaryFromData"
        )) {
        return EXIT_FAILURE;
    }
    aclrtFuncHandle function = nullptr;
    if (!CheckAcl(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function), "aclrtBinaryGetFunctionByEntry")) {
        return EXIT_FAILURE;
    }

    void *state_device = nullptr;
    if (!CheckAcl(
            aclrtMalloc(&state_device, sizeof(icache_scalar_pmu::ProbeState), ACL_MEM_MALLOC_NORMAL_ONLY),
            "aclrtMalloc(probe state)"
        )) {
        return EXIT_FAILURE;
    }

    atomic_probe::pmu::PmuSession pmu_session;
    if (!pmu_session.Initialize(
            static_cast<uint32_t>(device), stream, kernel_path, "libicache_scalar_pmu_aicpu.so"
        ) ||
        !pmu_session.Configure()) {
        (void)pmu_session.Finalize();
        return EXIT_FAILURE;
    }

    std::printf(
        "=== Single-AIV paired WARM/COLD I-cache scalar-busy PMU probe ===\n"
        "device=%d pairs=%u base_seed=0x%llx "
        "events=total,scalar_busy(0x1),icache_req(0x34),icache_miss(0x35)\n",
        device, repeats, static_cast<unsigned long long>(base_seed)
    );

    std::array<std::vector<Sample>, 2> samples;
    for (auto &mode_samples : samples) mode_samples.reserve(repeats);
    bool all_passed = true;
    for (uint32_t repeat = 1; repeat <= repeats; ++repeat) {
        const uint64_t seed = base_seed + static_cast<uint64_t>(repeat - 1) * kRepeatSeedStride;
        const bool warm_first = (repeat & 1U) != 0;
        const std::array<icache_scalar_pmu::Mode, 2> order = warm_first
            ? std::array<icache_scalar_pmu::Mode, 2>{
                  icache_scalar_pmu::Mode::WarmTarget, icache_scalar_pmu::Mode::ColdTarget}
            : std::array<icache_scalar_pmu::Mode, 2>{
                  icache_scalar_pmu::Mode::ColdTarget, icache_scalar_pmu::Mode::WarmTarget};

        std::array<Sample, 2> pair;
        bool pair_samples_ok = true;
        for (const icache_scalar_pmu::Mode mode : order) {
            const uint32_t mode_index = static_cast<uint32_t>(mode);
            pair_samples_ok &= RunOne(
                function, stream, state_device, pmu_session.RegisterBasesDeviceAddress(), mode, repeat, seed,
                &pair[mode_index]
            );
            if (!pair_samples_ok) break;
        }
        if (!pair_samples_ok) {
            all_passed = false;
            break;
        }

        const Sample &warm = pair[static_cast<uint32_t>(icache_scalar_pmu::Mode::WarmTarget)];
        const Sample &cold = pair[static_cast<uint32_t>(icache_scalar_pmu::Mode::ColdTarget)];
        const bool same_physical_core = warm.result.physical_core_id == cold.result.physical_core_id;
        const bool cold_has_more_misses = cold.result.pmu_icache_miss > warm.result.pmu_icache_miss;
        std::printf(
            "[PAIR] repeat=%u order=%s physical_warm=%llu physical_cold=%llu "
            "miss_warm=%llu miss_cold=%llu same_physical=%s cold_gt_warm_miss=%s status=%s\n",
            repeat, warm_first ? "WARM,COLD" : "COLD,WARM",
            static_cast<unsigned long long>(warm.result.physical_core_id),
            static_cast<unsigned long long>(cold.result.physical_core_id),
            static_cast<unsigned long long>(warm.result.pmu_icache_miss),
            static_cast<unsigned long long>(cold.result.pmu_icache_miss), same_physical_core ? "PASS" : "FAIL",
            cold_has_more_misses ? "PASS" : "FAIL",
            same_physical_core && cold_has_more_misses ? "PASS" : "FAIL"
        );
        if (!same_physical_core || !cold_has_more_misses) {
            all_passed = false;
            break;
        }
        samples[static_cast<uint32_t>(icache_scalar_pmu::Mode::WarmTarget)].push_back(warm);
        samples[static_cast<uint32_t>(icache_scalar_pmu::Mode::ColdTarget)].push_back(cold);
    }

    if (all_passed) PrintSummary(samples);

    bool cleanup_ok = pmu_session.Finalize();
    cleanup_ok &= CheckAcl(aclrtFree(state_device), "aclrtFree(probe state)");
    cleanup_ok &= CheckAcl(aclrtBinaryUnLoad(binary_handle), "aclrtBinaryUnLoad");
    cleanup_ok &= CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream");
    cleanup_ok &= CheckAcl(aclrtResetDevice(device), "aclrtResetDevice");
    cleanup_ok &= CheckAcl(aclFinalize(), "aclFinalize");
    std::printf(
        "[SUMMARY] completed_pairs=%zu requested_pairs=%u semantic_status=%s "
        "pmu_restore_and_cleanup=%s\n",
        samples[0].size(), repeats, all_passed ? "PASS" : "FAIL", cleanup_ok ? "PASS" : "FAIL"
    );
    return all_passed && cleanup_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
