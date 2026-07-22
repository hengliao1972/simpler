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

#include "vector_scalar_pmu_shared.h"
#include "pmu_probe_host_support.h"
#include "../probe_host.h"
#include "../pa_scheduler/ccec/pmu_owner_host.h"

#include "acl/acl.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

using atomic_probe::pmu::CheckAcl;

constexpr double kAicoreCyclesPerNanosecond = 1.65;
constexpr uint32_t kMeasuredRepeats = 5U;
constexpr std::array<uint32_t, 2U> kRoundValues = {16U, 128U};

const char *ModeName(vector_scalar_pmu::Mode mode) {
    switch (mode) {
    case vector_scalar_pmu::Mode::Empty:
        return "EMPTY";
    case vector_scalar_pmu::Mode::LoopControl:
        return "LOOP_CONTROL";
    case vector_scalar_pmu::Mode::VectorAdd:
        return "VECTOR_ADD";
    default:
        return "UNKNOWN";
    }
}

struct Sample {
    vector_scalar_pmu::ProbeResult result{};
    bool output_ok = false;
};

bool ValidateSample(
    const Sample &sample, vector_scalar_pmu::Mode mode, uint32_t rounds,
    const pa_scheduler::pmu_owner::PmuOwnerControl &owner, std::string *reason
) {
    const auto &result = sample.result;
    if (result.observed_mode != static_cast<uint32_t>(mode) || result.observed_rounds != rounds) {
        *reason = "stale-or-mismatched-control";
        return false;
    }
    if (result.physical_core_id >= pa_scheduler::pmu_owner::kPhysicalSubcoreCount ||
        !pa_scheduler::pmu_owner::IsConfigured(owner, static_cast<uint32_t>(result.physical_core_id))) {
        *reason = "physical-core-not-owned";
        return false;
    }
    if (result.selector_status != vector_scalar_pmu::kRequiredSelectorStatus) {
        *reason = "selector-map";
        return false;
    }
    if ((result.pmu_ctrl_after_stop & 1ULL) != 0U) {
        *reason = "pmu-gate-still-enabled";
        return false;
    }
    if (result.sys_ticks == 0U || result.pmu_total_cycles == 0U) {
        *reason = "zero-cycle-window";
        return false;
    }
    if (result.pmu_scalar_busy > result.pmu_total_cycles || result.pmu_vector_busy > result.pmu_total_cycles ||
        result.pmu_mte2_busy > result.pmu_total_cycles || result.pmu_mte3_busy > result.pmu_total_cycles) {
        *reason = "busy-counter-exceeds-total";
        return false;
    }
    if (result.pmu_icache_miss > result.pmu_icache_request) {
        *reason = "icache-miss-exceeds-request";
        return false;
    }
    if (mode == vector_scalar_pmu::Mode::VectorAdd && (result.pmu_vector_busy == 0U || result.pmu_scalar_busy == 0U ||
                                                       result.pmu_mte2_busy == 0U || result.pmu_mte3_busy == 0U)) {
        *reason = "vector-pipeline-counter-zero";
        return false;
    }
    if (mode == vector_scalar_pmu::Mode::VectorAdd && !sample.output_ok) {
        *reason = "vector-output";
        return false;
    }
    return true;
}

bool RunOne(
    aclrtFuncHandle function, aclrtStream stream, void *state_device, void *input_a_device, void *input_b_device,
    void *output_device, uint64_t pmu_register_bases, vector_scalar_pmu::Mode mode, uint32_t rounds, uint32_t repeat,
    const pa_scheduler::pmu_owner::PmuOwnerControl &owner, Sample *sample, bool print_raw
) {
    vector_scalar_pmu::ProbeState state{};
    state.control.pmu_register_bases = pmu_register_bases;
    state.control.input_a = reinterpret_cast<uint64_t>(input_a_device);
    state.control.input_b = reinterpret_cast<uint64_t>(input_b_device);
    state.control.output = reinterpret_cast<uint64_t>(output_device);
    state.control.mode = static_cast<uint32_t>(mode);
    state.control.rounds = rounds;
    if (!CheckAcl(
            aclrtMemcpy(state_device, sizeof(state), &state, sizeof(state), ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy(H2D vector PMU state)"
        )) {
        return false;
    }
    if (mode == vector_scalar_pmu::Mode::VectorAdd &&
        !CheckAcl(
            aclrtMemset(output_device, vector_scalar_pmu::kTileBytes, 0xff, vector_scalar_pmu::kTileBytes),
            "aclrtMemset(vector output sentinel)"
        )) {
        return false;
    }

    struct KernelArgs {
        uint64_t state_pointer;
    } args{reinterpret_cast<uint64_t>(state_device)};
    static_assert(sizeof(KernelArgs) == sizeof(uint64_t), "unexpected CCEC kernel argument ABI");
    if (!CheckAcl(
            aclrtLaunchKernelWithHostArgs(function, 1U, stream, nullptr, &args, sizeof(args), nullptr, 0U),
            "aclrtLaunchKernelWithHostArgs(vector scalar PMU)"
        ) ||
        !CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(vector scalar PMU)") ||
        !CheckAcl(
            aclrtMemcpy(&state, sizeof(state), state_device, sizeof(state), ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy(D2H vector PMU state)"
        )) {
        return false;
    }

    sample->result = state.result;
    sample->output_ok = mode != vector_scalar_pmu::Mode::VectorAdd;
    if (mode == vector_scalar_pmu::Mode::VectorAdd) {
        std::vector<float> output(vector_scalar_pmu::kTileElements);
        if (!CheckAcl(
                aclrtMemcpy(
                    output.data(), vector_scalar_pmu::kTileBytes, output_device, vector_scalar_pmu::kTileBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "aclrtMemcpy(D2H vector output)"
            )) {
            return false;
        }
        sample->output_ok = std::all_of(output.begin(), output.end(), [](float value) {
            return value == 5.0F;
        });
    }

    std::string reason;
    const bool passed = ValidateSample(*sample, mode, rounds, owner, &reason);
    if (print_raw) {
        const uint64_t residual = sample->result.pmu_total_cycles - sample->result.pmu_scalar_busy;
        const double scalar_ratio =
            static_cast<double>(sample->result.pmu_scalar_busy) / static_cast<double>(sample->result.pmu_total_cycles);
        std::printf(
            "[RAW] repeat=%u rounds=%u mode=%s sys_ticks=%llu total=%llu scalar=%llu "
            "non_scalar_residual=%llu scalar_ratio=%.9f vector=%llu mte2=%llu mte3=%llu "
            "icache_req=%llu icache_miss=%llu physical=%llu selectors=0x%llx output=%s status=%s%s%s\n",
            repeat, rounds, ModeName(mode), static_cast<unsigned long long>(sample->result.sys_ticks),
            static_cast<unsigned long long>(sample->result.pmu_total_cycles),
            static_cast<unsigned long long>(sample->result.pmu_scalar_busy), static_cast<unsigned long long>(residual),
            scalar_ratio, static_cast<unsigned long long>(sample->result.pmu_vector_busy),
            static_cast<unsigned long long>(sample->result.pmu_mte2_busy),
            static_cast<unsigned long long>(sample->result.pmu_mte3_busy),
            static_cast<unsigned long long>(sample->result.pmu_icache_request),
            static_cast<unsigned long long>(sample->result.pmu_icache_miss),
            static_cast<unsigned long long>(sample->result.physical_core_id),
            static_cast<unsigned long long>(sample->result.selector_status), sample->output_ok ? "PASS" : "FAIL",
            passed ? "PASS" : "FAIL", passed ? "" : " reason=", passed ? "" : reason.c_str()
        );
    }
    return passed;
}

uint64_t Median(std::vector<uint64_t> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2U;
    if ((values.size() & 1U) != 0U) return values[middle];
    return values[middle - 1U] + (values[middle] - values[middle - 1U]) / 2U;
}

double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2U;
    return (values.size() & 1U) != 0U ? values[middle] : (values[middle - 1U] + values[middle]) / 2.0;
}

using CounterMember = uint64_t vector_scalar_pmu::ProbeResult::*;

uint64_t MedianCounter(const std::vector<Sample> &samples, CounterMember member) {
    std::vector<uint64_t> values;
    values.reserve(samples.size());
    for (const Sample &sample : samples)
        values.push_back(sample.result.*member);
    return Median(std::move(values));
}

double PairedDeltaPerIteration(
    const std::vector<Sample> &minuend, const std::vector<Sample> &subtrahend, CounterMember member, uint32_t rounds
) {
    std::vector<double> deltas;
    deltas.reserve(minuend.size());
    for (size_t index = 0U; index < minuend.size(); ++index) {
        deltas.push_back(
            (static_cast<double>(minuend[index].result.*member) - static_cast<double>(subtrahend[index].result.*member)
            ) /
            static_cast<double>(rounds)
        );
    }
    return Median(std::move(deltas));
}

void PrintRoundSummary(uint32_t rounds, const std::array<std::vector<Sample>, 3> &samples) {
    struct Metric {
        const char *name;
        CounterMember member;
    };
    constexpr Metric metrics[] = {
        {"sys_ticks", &vector_scalar_pmu::ProbeResult::sys_ticks},
        {"total", &vector_scalar_pmu::ProbeResult::pmu_total_cycles},
        {"scalar", &vector_scalar_pmu::ProbeResult::pmu_scalar_busy},
        {"vector", &vector_scalar_pmu::ProbeResult::pmu_vector_busy},
        {"mte2", &vector_scalar_pmu::ProbeResult::pmu_mte2_busy},
        {"mte3", &vector_scalar_pmu::ProbeResult::pmu_mte3_busy},
        {"icache_req", &vector_scalar_pmu::ProbeResult::pmu_icache_request},
        {"icache_miss", &vector_scalar_pmu::ProbeResult::pmu_icache_miss},
    };

    for (uint32_t mode_index = 0U; mode_index < static_cast<uint32_t>(vector_scalar_pmu::Mode::Count); ++mode_index) {
        const auto mode = static_cast<vector_scalar_pmu::Mode>(mode_index);
        std::printf("[MEDIAN] rounds=%u mode=%s", rounds, ModeName(mode));
        for (const Metric &metric : metrics) {
            std::printf(
                " %s=%llu", metric.name,
                static_cast<unsigned long long>(MedianCounter(samples[mode_index], metric.member))
            );
        }
        std::printf("\n");
    }

    const auto &loop = samples[static_cast<uint32_t>(vector_scalar_pmu::Mode::LoopControl)];
    const auto &vector = samples[static_cast<uint32_t>(vector_scalar_pmu::Mode::VectorAdd)];
    for (const Metric &metric : metrics) {
        std::printf(
            "[VECTOR_MINUS_LOOP_PER_ITER] rounds=%u metric=%s value=%.6f\n", rounds, metric.name,
            PairedDeltaPerIteration(vector, loop, metric.member, rounds)
        );
    }

    std::vector<double> scalar_ratios;
    std::vector<double> residual_ratios;
    scalar_ratios.reserve(vector.size());
    residual_ratios.reserve(vector.size());
    for (const Sample &sample : vector) {
        const double total = static_cast<double>(sample.result.pmu_total_cycles);
        const double scalar = static_cast<double>(sample.result.pmu_scalar_busy);
        scalar_ratios.push_back(scalar / total);
        residual_ratios.push_back((total - scalar) / total);
    }
    const uint64_t median_total = MedianCounter(vector, &vector_scalar_pmu::ProbeResult::pmu_total_cycles);
    const uint64_t median_scalar = MedianCounter(vector, &vector_scalar_pmu::ProbeResult::pmu_scalar_busy);
    const double scalar_ratio = Median(std::move(scalar_ratios));
    const double residual_ratio = Median(std::move(residual_ratios));
    std::printf(
        "[VECTOR_CLASSIFICATION] rounds=%u median_total=%llu median_total_ns_at_1p65ghz=%.3f "
        "median_scalar=%llu median_non_scalar_residual=%llu scalar_share=%.9f residual_share=%.9f "
        "dominant=%s\n",
        rounds, static_cast<unsigned long long>(median_total),
        static_cast<double>(median_total) / kAicoreCyclesPerNanosecond, static_cast<unsigned long long>(median_scalar),
        static_cast<unsigned long long>(median_total - median_scalar), scalar_ratio, residual_ratio,
        scalar_ratio > residual_ratio ? "scalar_busy" : "non_scalar_residual"
    );
}

}  // namespace

int main(int argc, char **argv) {
    const std::string kernel_path = argc > 1 ? argv[1] : "./vector_scalar_pmu_kernel.o";
    if (argc > 2) {
        std::fprintf(stderr, "Usage: %s [vector_scalar_pmu_kernel.o]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const int32_t device = atomic_probe::DeviceId();
    if (device < 0) return EXIT_FAILURE;
    const std::vector<char> kernel_data = atomic_probe::pmu::ReadBinary(kernel_path);
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
    if (!CheckAcl(aclrtBinaryGetFunctionByEntry(binary_handle, 0U, &function), "aclrtBinaryGetFunctionByEntry")) {
        return EXIT_FAILURE;
    }

    void *state_device = nullptr;
    void *input_a_device = nullptr;
    void *input_b_device = nullptr;
    void *output_device = nullptr;
    if (!CheckAcl(
            aclrtMalloc(&state_device, sizeof(vector_scalar_pmu::ProbeState), ACL_MEM_MALLOC_NORMAL_ONLY),
            "aclrtMalloc(vector PMU state)"
        ) ||
        !CheckAcl(
            aclrtMalloc(&input_a_device, vector_scalar_pmu::kTileBytes, ACL_MEM_MALLOC_NORMAL_ONLY),
            "aclrtMalloc(vector input A)"
        ) ||
        !CheckAcl(
            aclrtMalloc(&input_b_device, vector_scalar_pmu::kTileBytes, ACL_MEM_MALLOC_NORMAL_ONLY),
            "aclrtMalloc(vector input B)"
        ) ||
        !CheckAcl(
            aclrtMalloc(&output_device, vector_scalar_pmu::kTileBytes, ACL_MEM_MALLOC_NORMAL_ONLY),
            "aclrtMalloc(vector output)"
        )) {
        return EXIT_FAILURE;
    }

    const std::vector<float> input_a(vector_scalar_pmu::kTileElements, 2.0F);
    const std::vector<float> input_b(vector_scalar_pmu::kTileElements, 3.0F);
    if (!CheckAcl(
            aclrtMemcpy(
                input_a_device, vector_scalar_pmu::kTileBytes, input_a.data(), vector_scalar_pmu::kTileBytes,
                ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy(H2D vector input A)"
        ) ||
        !CheckAcl(
            aclrtMemcpy(
                input_b_device, vector_scalar_pmu::kTileBytes, input_b.data(), vector_scalar_pmu::kTileBytes,
                ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy(H2D vector input B)"
        )) {
        return EXIT_FAILURE;
    }

    atomic_probe::pmu::RegisterMappings mappings;
    if (!mappings.Initialize(static_cast<uint32_t>(device))) return EXIT_FAILURE;
    const auto &mapped_array = mappings.RegisterBases();
    const std::vector<uint64_t> mapped_bases(mapped_array.begin(), mapped_array.end());

    pa_scheduler::pmu_owner::PmuOwnerSession owner;
    const std::string dispatcher_path =
        atomic_probe::pmu::ArtifactBesideKernel(kernel_path, "libvector_scalar_pmu_owner_dispatcher.so");
    const std::string owner_path =
        atomic_probe::pmu::ArtifactBesideKernel(kernel_path, "libvector_scalar_pmu_owner_aicpu.so");
    if (!owner.Initialize(static_cast<uint32_t>(device), stream, dispatcher_path, owner_path, mapped_bases) ||
        !owner.Configure()) {
        std::fprintf(stderr, "Cannot establish the ten-slot PMU owner session.\n");
        return EXIT_FAILURE;
    }

    std::printf(
        "=== Single-AIV PA vector-loop scalar-busy PMU probe ===\n"
        "device=%d repeats=%u tile=128x128-f32 bytes_per_iteration=196608 "
        "aicore_hz=1.65GHz sys_counter_hz=1GHz\n"
        "events=CNT0:vector(0x501),CNT2:scalar(0x001),CNT4:MTE2(0x202),"
        "CNT5:MTE3(0x203),CNT6:icache_req(0x034),CNT7:icache_miss(0x035)\n"
        "note=pipe busy counters may overlap; only total-scalar is reported as the non-scalar residual\n",
        device, kMeasuredRepeats
    );

    bool all_passed = true;
    // One unreported warm-up per mode removes first-launch/runtime setup from
    // the measured samples. It remains outside every reported paired repeat.
    for (const uint32_t rounds : kRoundValues) {
        for (uint32_t mode_index = 0U; mode_index < static_cast<uint32_t>(vector_scalar_pmu::Mode::Count);
             ++mode_index) {
            Sample warmup;
            all_passed &= RunOne(
                function, stream, state_device, input_a_device, input_b_device, output_device,
                owner.RegisterTableDeviceAddress(), static_cast<vector_scalar_pmu::Mode>(mode_index), rounds, 0U,
                owner.Control(), &warmup, false
            );
        }
        if (!all_passed) break;

        std::array<std::vector<Sample>, 3> samples;
        for (uint32_t repeat = 1U; repeat <= kMeasuredRepeats && all_passed; ++repeat) {
            for (uint32_t mode_index = 0U; mode_index < static_cast<uint32_t>(vector_scalar_pmu::Mode::Count);
                 ++mode_index) {
                Sample sample;
                const bool passed = RunOne(
                    function, stream, state_device, input_a_device, input_b_device, output_device,
                    owner.RegisterTableDeviceAddress(), static_cast<vector_scalar_pmu::Mode>(mode_index), rounds,
                    repeat, owner.Control(), &sample, true
                );
                all_passed &= passed;
                samples[mode_index].push_back(sample);
                if (!passed) break;
            }
        }
        if (all_passed) PrintRoundSummary(rounds, samples);
    }

    const bool owner_cleanup_ok = owner.Finalize();
    mappings.Release();
    bool cleanup_ok = owner_cleanup_ok;
    cleanup_ok &= CheckAcl(aclrtFree(output_device), "aclrtFree(vector output)");
    cleanup_ok &= CheckAcl(aclrtFree(input_b_device), "aclrtFree(vector input B)");
    cleanup_ok &= CheckAcl(aclrtFree(input_a_device), "aclrtFree(vector input A)");
    cleanup_ok &= CheckAcl(aclrtFree(state_device), "aclrtFree(vector PMU state)");
    cleanup_ok &= CheckAcl(aclrtBinaryUnLoad(binary_handle), "aclrtBinaryUnLoad");
    cleanup_ok &= CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream");
    cleanup_ok &= CheckAcl(aclrtResetDevice(device), "aclrtResetDevice");
    cleanup_ok &= CheckAcl(aclFinalize(), "aclFinalize");
    std::printf(
        "[SUMMARY] semantic_status=%s pmu_restore_and_cleanup=%s\n", all_passed ? "PASS" : "FAIL",
        cleanup_ok ? "PASS" : "FAIL"
    );
    return all_passed && cleanup_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
