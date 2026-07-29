/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the LICENSE file for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the license.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "../probe_host.h"
#include "../shared_preload_model_shared.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct KernelArgs {
    uint64_t control;
    uint64_t results;
    uint64_t data;
};

struct DeviceResources {
    aclrtStream stream = nullptr;
    aclrtBinHandle binary = nullptr;
    aclrtFuncHandle function = nullptr;
    void *control = nullptr;
    void *results = nullptr;
    void *data = nullptr;
};

struct TimingAggregate {
    std::vector<uint64_t> setup;
    std::vector<uint64_t> issue;
    std::vector<uint64_t> gap;
    std::vector<uint64_t> access;
    std::vector<uint64_t> publish;
    std::vector<uint64_t> total;
    std::vector<uint64_t> critical_access;
    std::vector<uint64_t> critical_total;
    uint64_t immediate_busy = 0;
    uint64_t final_busy = 0;
    uint64_t result_count = 0;
};

struct DCacheCase {
    const char *name;
    uint32_t bytes;
};

constexpr std::array<DCacheCase, 3> kDCacheCases = {{
    {"writer-history-3-symbols", shared_preload_model::kWriterHistoryBytes},
    {"shared-output-1-desc", shared_preload_model::kOneDescriptorBytes},
    {"shared-output-3-desc", shared_preload_model::kThreeDescriptorBytes},
}};

static_assert(
    sizeof(KernelArgs) == 3U * sizeof(uint64_t),
    "unexpected CCEC kernel argument ABI"
);

bool Check(aclError error, const char *expression) {
    return atomic_probe::CheckAcl(
        error, expression, __FILE__, __LINE__
    );
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

uint64_t Quantile(std::vector<uint64_t> values, double quantile) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(
        std::llround(
            quantile * static_cast<double>(values.size() - 1U)
        )
    );
    return values[index];
}

uint64_t Median(const std::vector<uint64_t> &values) {
    return Quantile(values, 0.5);
}

bool IsZero(const shared_preload_model::ProbeResult &result) {
    const auto *bytes = reinterpret_cast<const uint8_t *>(&result);
    for (size_t byte = 0; byte < sizeof(result); ++byte) {
        if (bytes[byte] != 0U) return false;
    }
    return true;
}

uint64_t RotateLeftOne(uint64_t value) {
    return (value << 1U) | (value >> 63U);
}

std::array<shared_preload_model::WorkerData,
           shared_preload_model::kWorkers>
MakeInitialData() {
    std::array<shared_preload_model::WorkerData,
               shared_preload_model::kWorkers>
        data{};
    for (uint32_t worker = 0;
         worker < shared_preload_model::kWorkers; ++worker) {
        for (uint32_t byte = 0;
             byte < shared_preload_model::kMaxPayloadBytes; ++byte) {
            data[worker].source[byte] =
                shared_preload_model::PayloadByte(worker, byte);
        }
    }
    return data;
}

bool ValidateResult(
    const shared_preload_model::ProbeResult &result,
    const shared_preload_model::ProbeControl &control, uint32_t worker,
    std::string *reason
) {
    const bool active =
        worker >= control.first_worker &&
        worker < control.first_worker + control.worker_count;
    if (!active) {
        if (!IsZero(result)) {
            *reason = "inactive-worker-wrote-result-" +
                      std::to_string(worker);
            return false;
        }
        return true;
    }
    if (result.status != shared_preload_model::kStatusComplete ||
        result.worker_id != worker ||
        result.experiment != control.experiment ||
        result.mode != control.mode ||
        result.active_bytes != control.active_bytes ||
        result.gap_rounds != control.gap_rounds ||
        result.sample_id != control.sample_id ||
        result.access_ticks == 0U || result.total_ticks == 0U) {
        *reason = "result-header-or-timing-" + std::to_string(worker);
        return false;
    }
    if (control.gap_rounds >=
            shared_preload_model::kOverlapGapRounds &&
        result.gap_ticks < control.gap_rounds) {
        *reason = "independent-gap-was-not-bracketed-" +
                  std::to_string(worker);
        return false;
    }

    const uint64_t gap_checksum = cache_preload::GapOracle(
        control.seed ^ static_cast<uint64_t>(worker),
        control.gap_rounds
    );
    if (control.experiment ==
        static_cast<uint32_t>(
            shared_preload_model::Experiment::ICachePlacement
        )) {
        const uint64_t expected_preparation =
            cache_preload::ICacheEvictorOracle(control.seed);
        const uint64_t expected_result =
            cache_preload::ICacheTargetOracle(
                control.seed ^ gap_checksum
            );
        if (result.preparation_checksum != expected_preparation ||
            result.result_checksum != expected_result ||
            result.setup_ticks != 0U ||
            result.publish_ticks != 0U) {
            *reason = "icache-oracle-" + std::to_string(worker);
            return false;
        }
        if (control.mode ==
                static_cast<uint32_t>(
                    shared_preload_model::Mode::ICacheBaseline
                ) &&
            (result.icache_immediate_status != 0U ||
             result.icache_final_status != 0U)) {
            *reason = "icache-baseline-status-" +
                      std::to_string(worker);
            return false;
        }
        return true;
    }

    const uint64_t expected_payload =
        shared_preload_model::PayloadChecksum(
            worker, control.active_bytes
        );
    const uint64_t expected_result =
        RotateLeftOne(expected_payload) ^ gap_checksum;
    if (result.preparation_checksum != expected_payload ||
        result.result_checksum != expected_result ||
        result.icache_immediate_status != 0U ||
        result.icache_final_status != 0U) {
        *reason = "dcache-oracle-" + std::to_string(worker);
        return false;
    }
    if (control.experiment ==
            static_cast<uint32_t>(
                shared_preload_model::Experiment::Publish
            ) &&
        (result.setup_ticks != 0U || result.publish_ticks == 0U)) {
        *reason = "publish-phase-shape-" + std::to_string(worker);
        return false;
    }
    if (control.experiment ==
            static_cast<uint32_t>(
                shared_preload_model::Experiment::Consume
            ) &&
        (result.setup_ticks == 0U || result.publish_ticks != 0U)) {
        *reason = "consume-phase-shape-" + std::to_string(worker);
        return false;
    }
    return true;
}

bool ValidatePayload(
    const std::array<shared_preload_model::WorkerData,
                     shared_preload_model::kWorkers> &data,
    const shared_preload_model::ProbeControl &control,
    std::string *reason
) {
    if (control.experiment ==
        static_cast<uint32_t>(
            shared_preload_model::Experiment::ICachePlacement
        )) {
        return true;
    }
    for (uint32_t worker = control.first_worker;
         worker < control.first_worker + control.worker_count; ++worker) {
        for (uint32_t byte = 0; byte < control.active_bytes; ++byte) {
            const uint8_t expected =
                shared_preload_model::PayloadByte(worker, byte);
            if (data[worker].source[byte] != expected ||
                data[worker].destination[byte] != expected) {
                *reason = "payload-" + std::to_string(worker) + "-" +
                          std::to_string(byte);
                return false;
            }
        }
    }
    return true;
}

void AppendTimings(
    const std::array<shared_preload_model::ProbeResult,
                     shared_preload_model::kWorkers> &results,
    const shared_preload_model::ProbeControl &control,
    TimingAggregate *aggregate
) {
    uint64_t critical_access = 0;
    uint64_t critical_total = 0;
    for (uint32_t worker = control.first_worker;
         worker < control.first_worker + control.worker_count; ++worker) {
        const auto &result = results[worker];
        aggregate->setup.push_back(result.setup_ticks);
        aggregate->issue.push_back(result.issue_ticks);
        aggregate->gap.push_back(result.gap_ticks);
        aggregate->access.push_back(result.access_ticks);
        aggregate->publish.push_back(result.publish_ticks);
        aggregate->total.push_back(result.total_ticks);
        critical_access =
            std::max(critical_access, result.access_ticks);
        critical_total =
            std::max(critical_total, result.total_ticks);
        aggregate->immediate_busy +=
            result.icache_immediate_status != 0U ? 1U : 0U;
        aggregate->final_busy +=
            result.icache_final_status != 0U ? 1U : 0U;
        ++aggregate->result_count;
    }
    aggregate->critical_access.push_back(critical_access);
    aggregate->critical_total.push_back(critical_total);
}

bool RunOne(
    const DeviceResources &device,
    const std::array<shared_preload_model::WorkerData,
                     shared_preload_model::kWorkers> &initial_data,
    const shared_preload_model::ProbeControl &control,
    TimingAggregate *aggregate, std::string *reason
) {
    std::array<shared_preload_model::ProbeResult,
               shared_preload_model::kWorkers>
        zero_results{};
    if (!Check(
            aclrtMemcpy(
                device.control, sizeof(control), &control,
                sizeof(control), ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy(H2D shared preload control)"
        ) ||
        !Check(
            aclrtMemcpy(
                device.results, sizeof(zero_results),
                zero_results.data(), sizeof(zero_results),
                ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy(H2D shared preload result reset)"
        ) ||
        !Check(
            aclrtMemcpy(
                device.data, sizeof(initial_data),
                initial_data.data(), sizeof(initial_data),
                ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy(H2D shared preload data reset)"
        )) {
        *reason = "setup-copy";
        return false;
    }

    KernelArgs args{
        reinterpret_cast<uint64_t>(device.control),
        reinterpret_cast<uint64_t>(device.results),
        reinterpret_cast<uint64_t>(device.data),
    };
    if (!Check(
            aclrtLaunchKernelWithHostArgs(
                device.function, shared_preload_model::kAicWorkers,
                device.stream, nullptr, &args, sizeof(args), nullptr, 0U
            ),
            "aclrtLaunchKernelWithHostArgs(shared preload model)"
        ) ||
        !Check(
            aclrtSynchronizeStream(device.stream),
            "aclrtSynchronizeStream(shared preload model)"
        )) {
        *reason = "launch-or-sync";
        return false;
    }

    std::array<shared_preload_model::ProbeResult,
               shared_preload_model::kWorkers>
        results{};
    std::array<shared_preload_model::WorkerData,
               shared_preload_model::kWorkers>
        data{};
    if (!Check(
            aclrtMemcpy(
                results.data(), sizeof(results), device.results,
                sizeof(results), ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "aclrtMemcpy(D2H shared preload results)"
        ) ||
        !Check(
            aclrtMemcpy(
                data.data(), sizeof(data), device.data, sizeof(data),
                ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "aclrtMemcpy(D2H shared preload data)"
        )) {
        *reason = "result-copy";
        return false;
    }
    for (uint32_t worker = 0;
         worker < shared_preload_model::kWorkers; ++worker) {
        if (!ValidateResult(results[worker], control, worker, reason)) {
            return false;
        }
    }
    if (!ValidatePayload(data, control, reason)) return false;
    AppendTimings(results, control, aggregate);
    return true;
}

double DeltaPercent(uint64_t baseline, uint64_t candidate) {
    if (baseline == 0U) return 0.0;
    return 100.0 *
           (static_cast<double>(candidate) -
            static_cast<double>(baseline)) /
           static_cast<double>(baseline);
}

void PrintMetric(
    const char *name, const std::vector<uint64_t> &baseline,
    const std::vector<uint64_t> &candidate
) {
    const uint64_t base = Median(baseline);
    const uint64_t preload = Median(candidate);
    std::printf(
        "  %-22s %8llu -> %8llu  %+8.3f%%\n", name,
        static_cast<unsigned long long>(base),
        static_cast<unsigned long long>(preload),
        DeltaPercent(base, preload)
    );
}

void PrintComparison(
    const char *label, uint32_t bytes, uint32_t gap_rounds,
    const TimingAggregate &baseline,
    const TimingAggregate &candidate
) {
    std::printf(
        "[COMPARE] %-31s bytes=%3u lines=%u gap_rounds=%u "
        "samples=%zu workers/sample=%llu\n",
        label, bytes,
        shared_preload_model::CacheLinesForBytes(bytes), gap_rounds,
        baseline.critical_total.size(),
        static_cast<unsigned long long>(
            baseline.result_count /
            std::max<size_t>(baseline.critical_total.size(), 1U)
        )
    );
    PrintMetric("core-median setup", baseline.setup, candidate.setup);
    PrintMetric("core-median issue", baseline.issue, candidate.issue);
    PrintMetric("core-median gap", baseline.gap, candidate.gap);
    PrintMetric("core-median copy/work", baseline.access, candidate.access);
    PrintMetric(
        "core-median publish", baseline.publish, candidate.publish
    );
    PrintMetric("core-median total", baseline.total, candidate.total);
    PrintMetric(
        "critical copy/work", baseline.critical_access,
        candidate.critical_access
    );
    PrintMetric(
        "critical total", baseline.critical_total,
        candidate.critical_total
    );
}

bool RunDCachePair(
    const DeviceResources &device,
    const std::array<shared_preload_model::WorkerData,
                     shared_preload_model::kWorkers> &initial_data,
    shared_preload_model::Experiment experiment,
    const DCacheCase &test_case, uint32_t gap_rounds
) {
    TimingAggregate aggregates[2];
    for (uint32_t sample = 0;
         sample < shared_preload_model::kDCacheSamples; ++sample) {
        const uint32_t first = sample & 1U;
        for (uint32_t order = 0; order < 2U; ++order) {
            const uint32_t policy = first ^ order;
            shared_preload_model::ProbeControl control{};
            control.magic = shared_preload_model::kControlMagic;
            control.experiment = static_cast<uint32_t>(experiment);
            control.mode = policy == 0U
                               ? static_cast<uint32_t>(
                                     shared_preload_model::Mode::
                                         DCacheBaseline
                                 )
                               : static_cast<uint32_t>(
                                     shared_preload_model::Mode::
                                         DCachePreload
                                 );
            control.active_bytes = test_case.bytes;
            control.gap_rounds = gap_rounds;
            control.sample_id = sample;
            control.first_worker = 0;
            control.worker_count = shared_preload_model::kWorkers;
            control.seed =
                0x1020304050607080ULL +
                static_cast<uint64_t>(sample) * 0x10001ULL +
                static_cast<uint64_t>(test_case.bytes) * 17ULL +
                static_cast<uint64_t>(control.experiment) * 0x100000ULL;
            std::string reason;
            if (!RunOne(
                    device, initial_data, control,
                    &aggregates[policy], &reason
                )) {
                std::fprintf(
                    stderr,
                    "[FAIL] dcache experiment=%u case=%s gap=%u "
                    "sample=%u mode=%u reason=%s\n",
                    control.experiment, test_case.name, gap_rounds,
                    sample, control.mode, reason.c_str()
                );
                return false;
            }
        }
    }
    const char *experiment_name =
        experiment == shared_preload_model::Experiment::Publish
            ? "publish"
            : "consume";
    const std::string label =
        std::string(experiment_name) + "/" + test_case.name;
    PrintComparison(
        label.c_str(), test_case.bytes, gap_rounds,
        aggregates[0], aggregates[1]
    );
    return true;
}

bool RunICacheMatrix(
    const DeviceResources &device,
    const std::array<shared_preload_model::WorkerData,
                     shared_preload_model::kWorkers> &initial_data
) {
    constexpr std::array<shared_preload_model::Mode, 3> modes = {{
        shared_preload_model::Mode::ICacheBaseline,
        shared_preload_model::Mode::ICacheCallerPreload,
        shared_preload_model::Mode::ICacheTargetPreload,
    }};
    std::array<TimingAggregate, modes.size()> aggregates;
    for (uint32_t sample = 0;
         sample < shared_preload_model::kICacheSamples; ++sample) {
        const uint32_t rotation = sample % modes.size();
        for (uint32_t order = 0; order < modes.size(); ++order) {
            const uint32_t index = (rotation + order) % modes.size();
            shared_preload_model::ProbeControl control{};
            control.magic = shared_preload_model::kControlMagic;
            control.experiment = static_cast<uint32_t>(
                shared_preload_model::Experiment::ICachePlacement
            );
            control.mode = static_cast<uint32_t>(modes[index]);
            control.active_bytes = 0;
            control.gap_rounds =
                shared_preload_model::kOverlapGapRounds;
            control.sample_id = sample;
            control.first_worker =
                shared_preload_model::kAicWorkers;
            control.worker_count =
                shared_preload_model::kAivWorkers;
            control.seed =
                0xa5a5000011110000ULL +
                static_cast<uint64_t>(sample) * 0x10001ULL;
            std::string reason;
            if (!RunOne(
                    device, initial_data, control,
                    &aggregates[index], &reason
                )) {
                std::fprintf(
                    stderr,
                    "[FAIL] icache sample=%u mode=%u reason=%s\n",
                    sample, control.mode, reason.c_str()
                );
                return false;
            }
        }
    }

    std::printf(
        "[ICACHE] AIV-only same target region; baseline/caller/target "
        "samples=%u workers/sample=%u gap_rounds=%u\n",
        shared_preload_model::kICacheSamples,
        shared_preload_model::kAivWorkers,
        shared_preload_model::kOverlapGapRounds
    );
    PrintComparison(
        "icache caller-current-PC", 0,
        shared_preload_model::kOverlapGapRounds,
        aggregates[0], aggregates[1]
    );
    PrintComparison(
        "icache target-current-PC", 0,
        shared_preload_model::kOverlapGapRounds,
        aggregates[0], aggregates[2]
    );
    std::printf(
        "  status caller immediate/final busy=%llu/%llu of %llu; "
        "target=%llu/%llu of %llu\n",
        static_cast<unsigned long long>(aggregates[1].immediate_busy),
        static_cast<unsigned long long>(aggregates[1].final_busy),
        static_cast<unsigned long long>(aggregates[1].result_count),
        static_cast<unsigned long long>(aggregates[2].immediate_busy),
        static_cast<unsigned long long>(aggregates[2].final_busy),
        static_cast<unsigned long long>(aggregates[2].result_count)
    );
    return true;
}

bool Cleanup(DeviceResources *device, int32_t device_id) {
    bool ok = true;
    if (device->data != nullptr) {
        ok &= Check(aclrtFree(device->data), "aclrtFree(shared data)");
        device->data = nullptr;
    }
    if (device->results != nullptr) {
        ok &= Check(
            aclrtFree(device->results), "aclrtFree(shared results)"
        );
        device->results = nullptr;
    }
    if (device->control != nullptr) {
        ok &= Check(
            aclrtFree(device->control), "aclrtFree(shared control)"
        );
        device->control = nullptr;
    }
    if (device->binary != nullptr) {
        ok &= Check(
            aclrtBinaryUnLoad(device->binary), "aclrtBinaryUnLoad"
        );
        device->binary = nullptr;
    }
    if (device->stream != nullptr) {
        ok &= Check(
            aclrtDestroyStream(device->stream), "aclrtDestroyStream"
        );
        device->stream = nullptr;
    }
    ok &= Check(aclrtResetDevice(device_id), "aclrtResetDevice");
    ok &= Check(aclFinalize(), "aclFinalize");
    return ok;
}

}  // namespace

int main(int argc, char **argv) {
    const std::string kernel_path =
        argc > 1 ? argv[1] : "./shared_preload_model_kernel.o";
    if (argc > 2) {
        std::fprintf(
            stderr, "Usage: %s [shared_preload_model_kernel.o]\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }
    const std::vector<char> kernel_data = ReadBinary(kernel_path);
    if (kernel_data.empty()) {
        std::fprintf(
            stderr, "Cannot read kernel binary: %s\n",
            kernel_path.c_str()
        );
        return EXIT_FAILURE;
    }
    const int32_t device_id = atomic_probe::DeviceId();
    if (device_id < 0) return EXIT_FAILURE;

    DeviceResources device;
    if (!Check(aclInit(nullptr), "aclInit") ||
        !Check(aclrtSetDevice(device_id), "aclrtSetDevice") ||
        !Check(
            aclrtCreateStream(&device.stream), "aclrtCreateStream"
        ) ||
        !Check(
            atomic_probe::LoadAicoreBinaryFromData(
                kernel_data.data(), kernel_data.size(), &device.binary
            ),
            "LoadAicoreBinaryFromData"
        ) ||
        !Check(
            aclrtBinaryGetFunctionByEntry(
                device.binary, 0U, &device.function
            ),
            "aclrtBinaryGetFunctionByEntry"
        ) ||
        !Check(
            aclrtMalloc(
                &device.control,
                sizeof(shared_preload_model::ProbeControl),
                ACL_MEM_MALLOC_HUGE_FIRST
            ),
            "aclrtMalloc(shared control)"
        ) ||
        !Check(
            aclrtMalloc(
                &device.results,
                shared_preload_model::kWorkers *
                    sizeof(shared_preload_model::ProbeResult),
                ACL_MEM_MALLOC_HUGE_FIRST
            ),
            "aclrtMalloc(shared results)"
        ) ||
        !Check(
            aclrtMalloc(
                &device.data,
                shared_preload_model::kWorkers *
                    sizeof(shared_preload_model::WorkerData),
                ACL_MEM_MALLOC_HUGE_FIRST
            ),
            "aclrtMalloc(shared data)"
        )) {
        Cleanup(&device, device_id);
        return EXIT_FAILURE;
    }

    std::printf(
        "=== A5 CCEC PA-shared cache preload model ===\n"
        "kernel=%s bytes=%zu topology=32_AIC+64_AIV "
        "dcache_samples=%u icache_samples=%u\n"
        "All timing values are raw SYS_CNT deltas. DCache publish keeps "
        "DCCI(CACHELINE_OUT)+DSB; consume keeps invalidate+DSB.\n",
        kernel_path.c_str(), kernel_data.size(),
        shared_preload_model::kDCacheSamples,
        shared_preload_model::kICacheSamples
    );

    const auto initial_data = MakeInitialData();
    bool semantic_ok = true;
    constexpr std::array<uint32_t, 2> gaps = {{
        shared_preload_model::kShortGapRounds,
        shared_preload_model::kOverlapGapRounds,
    }};
    for (const auto experiment : {
             shared_preload_model::Experiment::Publish,
             shared_preload_model::Experiment::Consume,
         }) {
        for (const auto &test_case : kDCacheCases) {
            for (const uint32_t gap : gaps) {
                if (!semantic_ok) break;
                semantic_ok = RunDCachePair(
                    device, initial_data, experiment, test_case, gap
                );
            }
        }
    }
    if (semantic_ok) {
        semantic_ok = RunICacheMatrix(device, initial_data);
    }

    atomic_probe::Result result;
    result.Expect(
        semantic_ok,
        "shared publish/consume payload, topology, and ICache oracles"
    );
    const bool cleanup_ok = Cleanup(&device, device_id);
    result.Expect(cleanup_ok, "shared preload model cleanup");
    return result.ExitCode();
}
