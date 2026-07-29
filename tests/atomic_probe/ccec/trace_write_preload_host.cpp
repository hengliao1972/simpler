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
#include "../trace_write_preload_shared.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

struct KernelArgs {
    uint64_t control;
    uint64_t results;
    uint64_t records;
};

struct Policy {
    const char *name;
    uint32_t distance;
    uint32_t cadence;
};

struct RunObservation {
    std::vector<trace_write_preload::ProbeResult> results;
    double wall_us = 0.0;
};

struct PolicyAggregate {
    Policy policy{};
    std::vector<uint64_t> issue_ticks;
    std::vector<uint64_t> flush_ticks;
    std::vector<uint64_t> total_ticks;
    std::vector<uint64_t> aic_total_ticks;
    std::vector<uint64_t> aiv_total_ticks;
    std::vector<uint64_t> critical_issue_ticks;
    std::vector<uint64_t> critical_total_ticks;
    std::vector<uint64_t> phase_issue_span_ticks;
    std::vector<uint64_t> phase_total_span_ticks;
    std::vector<uint64_t> start_skew_ticks;
    std::vector<double> wall_us;
};

struct DeviceResources {
    aclrtStream stream = nullptr;
    aclrtBinHandle binary = nullptr;
    aclrtFuncHandle function = nullptr;
    void *control = nullptr;
    void *results = nullptr;
    void *records = nullptr;
};

constexpr std::array<Policy, 9> kTunePolicies = {{
    {"baseline", 0U, 1U},
    {"d1-c1", 1U, 1U},
    {"d2-c1", 2U, 1U},
    {"d4-c1", 4U, 1U},
    {"d8-c1", 8U, 1U},
    {"d16-c1", 16U, 1U},
    {"d4-c4", 4U, 4U},
    {"d8-c4", 8U, 4U},
    {"d16-c4", 16U, 4U},
}};

constexpr uint32_t kTuneSamples = 9;
constexpr uint32_t kConfirmSamples = 21;
constexpr uint32_t kFollowupSamples = 13;
constexpr uint32_t kCapacitySamples = 9;
constexpr uint32_t kPrimaryRecords = (120U * 1024U) / trace_write_preload::kRecordBytes;
constexpr uint32_t kSecondaryRecords = (96U * 1024U) / trace_write_preload::kRecordBytes;

static_assert(sizeof(KernelArgs) == 3U * sizeof(uint64_t), "unexpected CCEC kernel argument ABI");
static_assert(kPrimaryRecords <= trace_write_preload::kMaxRecordsPerWorker, "primary write set exceeds stride");

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

uint64_t Quantile(std::vector<uint64_t> values, double quantile) {
    if (values.empty()) return 0U;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(
        std::llround(quantile * static_cast<double>(values.size() - 1U))
    );
    return values[index];
}

double QuantileDouble(std::vector<double> values, double quantile) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(
        std::llround(quantile * static_cast<double>(values.size() - 1U))
    );
    return values[index];
}

uint64_t Median(const std::vector<uint64_t> &values) {
    return Quantile(values, 0.5);
}

double MedianDouble(const std::vector<double> &values) {
    return QuantileDouble(values, 0.5);
}

bool ResultIsZero(const trace_write_preload::ProbeResult &result) {
    const auto *bytes = reinterpret_cast<const uint8_t *>(&result);
    for (size_t index = 0; index < sizeof(result); ++index) {
        if (bytes[index] != 0U) return false;
    }
    return true;
}

bool IsActiveWorker(
    uint32_t worker, const trace_write_preload::ProbeControl &control
) {
    return worker >= control.first_worker &&
           worker < control.first_worker + control.worker_count;
}

bool ValidateResultSet(
    const std::vector<trace_write_preload::ProbeResult> &results,
    const trace_write_preload::ProbeControl &control, std::string *reason
) {
    if (results.size() != trace_write_preload::kWorkers) {
        *reason = "result-count";
        return false;
    }
    for (uint32_t worker = 0; worker < trace_write_preload::kWorkers; ++worker) {
        const auto &result = results[worker];
        if (!IsActiveWorker(worker, control)) {
            if (!ResultIsZero(result)) {
                *reason = "inactive-worker-wrote-result";
                return false;
            }
            continue;
        }
        if (result.status != trace_write_preload::kStatusComplete ||
            result.worker_id != worker ||
            result.experiment != control.experiment ||
            result.sample_id != control.sample_id ||
            result.phase_begin == 0U ||
            result.phase_split <= result.phase_begin ||
            result.phase_end <= result.phase_split) {
            *reason = "active-result-header-or-timing";
            return false;
        }
        if (control.experiment ==
            static_cast<uint32_t>(trace_write_preload::Experiment::TraceWrite)) {
            if (result.records_or_lines != control.records_per_worker ||
                result.preload_distance_lines != control.preload_distance_lines ||
                result.preload_cadence_lines != control.preload_cadence_lines ||
                result.terminal_value != trace_write_preload::RecordStamp(
                    control.seed, worker, control.records_per_worker - 1U
                )) {
                *reason = "trace-write-result-echo";
                return false;
            }
        } else {
            if (result.records_or_lines != control.capacity_lines ||
                result.preload_distance_lines != 0U ||
                result.preload_cadence_lines != 0U ||
                result.terminal_value != 0U) {
                *reason = "capacity-result-echo";
                return false;
            }
        }
    }
    return true;
}

bool ValidateTracePayload(
    void *records_device, const trace_write_preload::ProbeControl &control,
    std::string *reason
) {
    const size_t span_bytes =
        static_cast<size_t>(control.worker_count) *
        trace_write_preload::kWorkerStrideBytes;
    std::vector<uint8_t> host(span_bytes);
    const auto *records_bytes = reinterpret_cast<const uint8_t *>(records_device);
    const void *source =
        records_bytes + static_cast<size_t>(control.first_worker) *
                            trace_write_preload::kWorkerStrideBytes;
    if (!Check(
            aclrtMemcpy(
                host.data(), host.size(), source, host.size(),
                ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "aclrtMemcpy(D2H trace payload verification)"
        )) {
        *reason = "payload-copy";
        return false;
    }

    for (uint32_t local = 0; local < control.worker_count; ++local) {
        const uint32_t worker = control.first_worker + local;
        const auto *records = reinterpret_cast<const trace_write_preload::TraceRecord *>(
            host.data() + static_cast<size_t>(local) *
                              trace_write_preload::kWorkerStrideBytes
        );
        for (uint32_t record_id = 0;
             record_id < control.records_per_worker; ++record_id) {
            const auto &record = records[record_id];
            const uint64_t stamp = trace_write_preload::RecordStamp(
                control.seed, worker, record_id
            );
            if (record.start_cycle != stamp ||
                record.end_cycle != stamp + 1U ||
                record.task_id != static_cast<int32_t>(record_id) ||
                record.function_id != static_cast<int32_t>(worker) ||
                record.flags !=
                    trace_write_preload::RecordFlags(worker, record_id) ||
                record.phase != static_cast<uint16_t>(record_id & 0xffffU) ||
                record.auxiliary != static_cast<uint16_t>(worker)) {
                *reason = "payload-record-" + std::to_string(worker) + "-" +
                          std::to_string(record_id);
                return false;
            }
        }
    }
    return true;
}

bool RunOne(
    const DeviceResources &device,
    const trace_write_preload::ProbeControl &control,
    bool validate_payload, RunObservation *observation, std::string *reason
) {
    std::array<trace_write_preload::ProbeResult, trace_write_preload::kWorkers>
        zero_results{};
    if (!Check(
            aclrtMemcpy(
                device.control, sizeof(control), &control, sizeof(control),
                ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy(H2D trace-write control)"
        ) ||
        !Check(
            aclrtMemcpy(
                device.results, sizeof(zero_results), zero_results.data(),
                sizeof(zero_results), ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy(H2D trace-write result reset)"
        )) {
        *reason = "setup-copy";
        return false;
    }

    KernelArgs args{
        reinterpret_cast<uint64_t>(device.control),
        reinterpret_cast<uint64_t>(device.results),
        reinterpret_cast<uint64_t>(device.records),
    };
    const auto wall_begin = std::chrono::steady_clock::now();
    if (!Check(
            aclrtLaunchKernelWithHostArgs(
                device.function, trace_write_preload::kAicWorkers,
                device.stream, nullptr, &args, sizeof(args), nullptr, 0U
            ),
            "aclrtLaunchKernelWithHostArgs(trace write preload)"
        ) ||
        !Check(
            aclrtSynchronizeStream(device.stream),
            "aclrtSynchronizeStream(trace write preload)"
        )) {
        *reason = "launch-or-sync";
        return false;
    }
    const auto wall_end = std::chrono::steady_clock::now();
    observation->wall_us =
        std::chrono::duration<double, std::micro>(wall_end - wall_begin).count();
    observation->results.resize(trace_write_preload::kWorkers);
    if (!Check(
            aclrtMemcpy(
                observation->results.data(),
                observation->results.size() *
                    sizeof(trace_write_preload::ProbeResult),
                device.results,
                observation->results.size() *
                    sizeof(trace_write_preload::ProbeResult),
                ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "aclrtMemcpy(D2H trace-write results)"
        )) {
        *reason = "result-copy";
        return false;
    }
    if (!ValidateResultSet(observation->results, control, reason)) return false;
    if (validate_payload &&
        control.experiment ==
            static_cast<uint32_t>(
                trace_write_preload::Experiment::TraceWrite
            ) &&
        !ValidateTracePayload(device.records, control, reason)) {
        return false;
    }
    return true;
}

void AccumulateWrite(
    const RunObservation &observation,
    const trace_write_preload::ProbeControl &control,
    PolicyAggregate *aggregate
) {
    uint64_t min_begin = std::numeric_limits<uint64_t>::max();
    uint64_t max_begin = 0U;
    uint64_t max_split = 0U;
    uint64_t max_end = 0U;
    uint64_t max_issue = 0U;
    uint64_t max_total = 0U;
    for (uint32_t worker = control.first_worker;
         worker < control.first_worker + control.worker_count; ++worker) {
        const auto &result = observation.results[worker];
        const uint64_t issue = result.phase_split - result.phase_begin;
        const uint64_t flush = result.phase_end - result.phase_split;
        const uint64_t total = result.phase_end - result.phase_begin;
        aggregate->issue_ticks.push_back(issue);
        aggregate->flush_ticks.push_back(flush);
        aggregate->total_ticks.push_back(total);
        if (worker < trace_write_preload::kAicWorkers) {
            aggregate->aic_total_ticks.push_back(total);
        } else {
            aggregate->aiv_total_ticks.push_back(total);
        }
        min_begin = std::min(min_begin, result.phase_begin);
        max_begin = std::max(max_begin, result.phase_begin);
        max_split = std::max(max_split, result.phase_split);
        max_end = std::max(max_end, result.phase_end);
        max_issue = std::max(max_issue, issue);
        max_total = std::max(max_total, total);
    }
    aggregate->critical_issue_ticks.push_back(max_issue);
    aggregate->critical_total_ticks.push_back(max_total);
    aggregate->phase_issue_span_ticks.push_back(max_split - min_begin);
    aggregate->phase_total_span_ticks.push_back(max_end - min_begin);
    aggregate->start_skew_ticks.push_back(max_begin - min_begin);
    aggregate->wall_us.push_back(observation.wall_us);
}

void PrintWriteAggregate(
    const char *scope, uint32_t records, uint32_t samples,
    const PolicyAggregate &aggregate
) {
    const uint64_t issue_median = Median(aggregate.issue_ticks);
    const uint64_t flush_median = Median(aggregate.flush_ticks);
    const uint64_t total_median = Median(aggregate.total_ticks);
    const double issue_per_record =
        static_cast<double>(issue_median) / static_cast<double>(records);
    const double total_per_record =
        static_cast<double>(total_median) / static_cast<double>(records);
    std::printf(
        "[WRITE] scope=%-10s bytes_per_worker=%6u samples=%2u "
        "policy=%-8s(d=%u,c=%u) "
        "core_issue=%llu[%llu,%llu] core_flush=%llu core_total=%llu "
        "critical_issue=%llu critical_total=%llu phase_total=%llu "
        "start_skew=%llu wall_us=%.3f issue/record=%.3f total/record=%.3f "
        "aic_total=%llu aiv_total=%llu\n",
        scope, records * trace_write_preload::kRecordBytes, samples,
        aggregate.policy.name, aggregate.policy.distance,
        aggregate.policy.cadence,
        static_cast<unsigned long long>(issue_median),
        static_cast<unsigned long long>(
            Quantile(aggregate.issue_ticks, 0.1)
        ),
        static_cast<unsigned long long>(
            Quantile(aggregate.issue_ticks, 0.9)
        ),
        static_cast<unsigned long long>(flush_median),
        static_cast<unsigned long long>(total_median),
        static_cast<unsigned long long>(
            Median(aggregate.critical_issue_ticks)
        ),
        static_cast<unsigned long long>(
            Median(aggregate.critical_total_ticks)
        ),
        static_cast<unsigned long long>(
            Median(aggregate.phase_total_span_ticks)
        ),
        static_cast<unsigned long long>(Median(aggregate.start_skew_ticks)),
        MedianDouble(aggregate.wall_us), issue_per_record, total_per_record,
        static_cast<unsigned long long>(Median(aggregate.aic_total_ticks)),
        static_cast<unsigned long long>(Median(aggregate.aiv_total_ticks))
    );
}

bool RunWriteMatrix(
    const DeviceResources &device, const char *scope,
    uint32_t first_worker, uint32_t worker_count, uint32_t records,
    const std::vector<Policy> &policies, uint32_t samples,
    std::vector<PolicyAggregate> *aggregates
) {
    aggregates->clear();
    for (const Policy &policy : policies) {
        PolicyAggregate aggregate;
        aggregate.policy = policy;
        aggregates->push_back(aggregate);
    }

    for (uint32_t sample = 0; sample < samples; ++sample) {
        for (uint32_t order = 0; order < policies.size(); ++order) {
            const uint32_t policy_index =
                (order + sample) % static_cast<uint32_t>(policies.size());
            const Policy &policy = policies[policy_index];
            trace_write_preload::ProbeControl control{};
            control.magic = trace_write_preload::kControlMagic;
            control.experiment = static_cast<uint32_t>(
                trace_write_preload::Experiment::TraceWrite
            );
            control.first_worker = first_worker;
            control.worker_count = worker_count;
            control.records_per_worker = records;
            control.preload_distance_lines = policy.distance;
            control.preload_cadence_lines = policy.cadence;
            control.sample_id = sample;
            control.seed =
                0x123456789abcdef0ULL ^
                (static_cast<uint64_t>(sample) * 0x9e3779b97f4a7c15ULL);
            control.worker_stride_bytes =
                trace_write_preload::kWorkerStrideBytes;

            RunObservation observation;
            std::string reason;
            if (!RunOne(
                    device, control, sample == 0U, &observation, &reason
                )) {
                std::fprintf(
                    stderr,
                    "[MISMATCH] scope=%s sample=%u policy=%s reason=%s\n",
                    scope, sample, policy.name, reason.c_str()
                );
                return false;
            }
            AccumulateWrite(
                observation, control, &(*aggregates)[policy_index]
            );
        }
    }
    for (const auto &aggregate : *aggregates) {
        PrintWriteAggregate(scope, records, samples, aggregate);
    }
    return true;
}

double PercentDelta(uint64_t candidate, uint64_t baseline) {
    if (baseline == 0U) return 0.0;
    return 100.0 *
           (static_cast<double>(candidate) - static_cast<double>(baseline)) /
           static_cast<double>(baseline);
}

size_t BestNonBaseline(const std::vector<PolicyAggregate> &aggregates) {
    size_t best = 1U;
    uint64_t best_value =
        Median(aggregates[best].critical_total_ticks);
    for (size_t index = 2U; index < aggregates.size(); ++index) {
        const uint64_t candidate =
            Median(aggregates[index].critical_total_ticks);
        if (candidate < best_value) {
            best = index;
            best_value = candidate;
        }
    }
    return best;
}

void PrintComparison(
    const char *label, const PolicyAggregate &baseline,
    const PolicyAggregate &candidate
) {
    const uint64_t baseline_issue =
        Median(baseline.critical_issue_ticks);
    const uint64_t candidate_issue =
        Median(candidate.critical_issue_ticks);
    const uint64_t baseline_total =
        Median(baseline.critical_total_ticks);
    const uint64_t candidate_total =
        Median(candidate.critical_total_ticks);
    std::printf(
        "[DELTA] %-18s candidate=%s critical_issue=%llu->%llu (%+.3f%%) "
        "critical_total=%llu->%llu (%+.3f%%)\n",
        label, candidate.policy.name,
        static_cast<unsigned long long>(baseline_issue),
        static_cast<unsigned long long>(candidate_issue),
        PercentDelta(candidate_issue, baseline_issue),
        static_cast<unsigned long long>(baseline_total),
        static_cast<unsigned long long>(candidate_total),
        PercentDelta(candidate_total, baseline_total)
    );
}

std::vector<uint8_t> BuildPointerCycle(
    uint32_t lines, uint32_t seed, uint32_t *start_line
) {
    std::vector<uint32_t> order(lines);
    for (uint32_t line = 0; line < lines; ++line) order[line] = line;
    std::mt19937 generator(seed);
    std::shuffle(order.begin(), order.end(), generator);
    std::vector<uint8_t> bytes(trace_write_preload::kWorkerStrideBytes, 0U);
    for (uint32_t index = 0; index < lines; ++index) {
        const uint32_t line = order[index];
        const uint32_t next = order[(index + 1U) % lines];
        std::memcpy(
            bytes.data() +
                static_cast<size_t>(line) *
                    trace_write_preload::kCacheLineBytes,
            &next, sizeof(next)
        );
    }
    *start_line = order[0];
    return bytes;
}

bool RunCapacitySweep(
    const DeviceResources &device, const char *scope, uint32_t worker
) {
    constexpr std::array<uint32_t, 9> kWorkingSetKiB = {
        4U, 8U, 12U, 16U, 20U, 24U, 32U, 48U, 64U,
    };
    for (const uint32_t kib : kWorkingSetKiB) {
        const uint32_t lines =
            kib * 1024U / trace_write_preload::kCacheLineBytes;
        uint32_t start_line = 0U;
        std::vector<uint8_t> cycle = BuildPointerCycle(
            lines, 0x95000000U ^ worker ^ kib, &start_line
        );
        auto *device_bytes = reinterpret_cast<uint8_t *>(device.records);
        void *worker_records =
            device_bytes + static_cast<size_t>(worker) *
                               trace_write_preload::kWorkerStrideBytes;
        if (!Check(
                aclrtMemcpy(
                    worker_records, cycle.size(), cycle.data(), cycle.size(),
                    ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "aclrtMemcpy(H2D capacity pointer cycle)"
            )) {
            return false;
        }

        std::vector<double> cold_per_load;
        std::vector<double> reuse_per_load;
        for (uint32_t sample = 0; sample < kCapacitySamples; ++sample) {
            trace_write_preload::ProbeControl control{};
            control.magic = trace_write_preload::kControlMagic;
            control.experiment = static_cast<uint32_t>(
                trace_write_preload::Experiment::CapacitySweep
            );
            control.first_worker = worker;
            control.worker_count = 1U;
            control.sample_id = sample;
            control.capacity_lines = lines;
            control.capacity_passes = trace_write_preload::kCapacityPasses;
            control.capacity_start_line = start_line;
            control.seed = 0x9500000000000000ULL ^ sample;
            control.worker_stride_bytes =
                trace_write_preload::kWorkerStrideBytes;

            RunObservation observation;
            std::string reason;
            if (!RunOne(device, control, false, &observation, &reason)) {
                std::fprintf(
                    stderr,
                    "[MISMATCH] capacity scope=%s kib=%u sample=%u "
                    "reason=%s\n",
                    scope, kib, sample, reason.c_str()
                );
                return false;
            }
            const auto &result = observation.results[worker];
            cold_per_load.push_back(
                static_cast<double>(result.phase_split - result.phase_begin) /
                static_cast<double>(lines)
            );
            reuse_per_load.push_back(
                static_cast<double>(result.phase_end - result.phase_split) /
                static_cast<double>(
                    lines * trace_write_preload::kCapacityPasses
                )
            );
        }
        std::printf(
            "[CAPACITY] scope=%-10s working_set=%2uKiB lines=%4u "
            "samples=%u cold_ticks/load=%.3f "
            "reuse_ticks/load=%.3f[%.3f,%.3f]\n",
            scope, kib, lines, kCapacitySamples,
            MedianDouble(cold_per_load), MedianDouble(reuse_per_load),
            QuantileDouble(reuse_per_load, 0.1),
            QuantileDouble(reuse_per_load, 0.9)
        );
    }
    return true;
}

bool Cleanup(DeviceResources *device, int32_t device_id) {
    bool ok = true;
    if (device->records != nullptr) {
        ok &= Check(aclrtFree(device->records), "aclrtFree(trace records)");
        device->records = nullptr;
    }
    if (device->results != nullptr) {
        ok &= Check(aclrtFree(device->results), "aclrtFree(trace results)");
        device->results = nullptr;
    }
    if (device->control != nullptr) {
        ok &= Check(aclrtFree(device->control), "aclrtFree(trace control)");
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
        argc > 1 ? argv[1] : "./trace_write_preload_kernel.o";
    if (argc > 2) {
        std::fprintf(
            stderr, "Usage: %s [trace_write_preload_kernel.o]\n", argv[0]
        );
        return EXIT_FAILURE;
    }
    const std::vector<char> kernel_data = ReadBinary(kernel_path);
    if (kernel_data.empty()) {
        std::fprintf(
            stderr, "Cannot read kernel binary: %s\n", kernel_path.c_str()
        );
        return EXIT_FAILURE;
    }
    const int32_t device_id = atomic_probe::DeviceId();
    if (device_id < 0) return EXIT_FAILURE;

    DeviceResources device;
    if (!Check(aclInit(nullptr), "aclInit") ||
        !Check(aclrtSetDevice(device_id), "aclrtSetDevice") ||
        !Check(aclrtCreateStream(&device.stream), "aclrtCreateStream") ||
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
                &device.control, sizeof(trace_write_preload::ProbeControl),
                ACL_MEM_MALLOC_HUGE_FIRST
            ),
            "aclrtMalloc(trace control)"
        ) ||
        !Check(
            aclrtMalloc(
                &device.results,
                trace_write_preload::kWorkers *
                    sizeof(trace_write_preload::ProbeResult),
                ACL_MEM_MALLOC_HUGE_FIRST
            ),
            "aclrtMalloc(trace results)"
        ) ||
        !Check(
            aclrtMalloc(
                &device.records,
                static_cast<size_t>(trace_write_preload::kWorkers) *
                    trace_write_preload::kWorkerStrideBytes,
                ACL_MEM_MALLOC_HUGE_FIRST
            ),
            "aclrtMalloc(trace records)"
        )) {
        Cleanup(&device, device_id);
        return EXIT_FAILURE;
    }

    std::printf(
        "=== A5 CCEC trace-write DCache preload probe ===\n"
        "kernel=%s bytes=%zu mixed_topology=32_AIC+64_AIV "
        "record=32B fields=7 records_per_cacheline=2 "
        "worker_stride=%uKiB\n"
        "Timing values are raw SYS_CNT deltas. issue ends after the last "
        "ordinary scalar store; total ends after per-line "
        "DCCI(CACHELINE_OUT)+DSB.\n",
        kernel_path.c_str(), kernel_data.size(),
        trace_write_preload::kWorkerStrideBytes / 1024U
    );

    std::vector<Policy> tune_policies(
        kTunePolicies.begin(), kTunePolicies.end()
    );
    std::vector<PolicyAggregate> tune;
    bool semantic_ok = RunWriteMatrix(
        device, "all96-tune", 0U, trace_write_preload::kWorkers,
        kPrimaryRecords, tune_policies, kTuneSamples, &tune
    );
    size_t best_index = 1U;
    if (semantic_ok) {
        best_index = BestNonBaseline(tune);
        PrintComparison("all96-tune", tune[0], tune[best_index]);
    }

    const std::vector<Policy> confirm_policies = {
        kTunePolicies[0], kTunePolicies[best_index],
    };
    std::vector<PolicyAggregate> confirm;
    if (semantic_ok) {
        semantic_ok = RunWriteMatrix(
            device, "all96-confirm", 0U, trace_write_preload::kWorkers,
            kPrimaryRecords, confirm_policies, kConfirmSamples, &confirm
        );
    }
    if (semantic_ok) {
        PrintComparison("all96-confirm", confirm[0], confirm[1]);
    }

    std::vector<PolicyAggregate> secondary;
    if (semantic_ok) {
        semantic_ok = RunWriteMatrix(
            device, "all96-96K", 0U, trace_write_preload::kWorkers,
            kSecondaryRecords, confirm_policies, kFollowupSamples, &secondary
        );
    }
    if (semantic_ok) {
        PrintComparison("all96-96K", secondary[0], secondary[1]);
    }

    std::vector<PolicyAggregate> single_aic;
    if (semantic_ok) {
        semantic_ok = RunWriteMatrix(
            device, "single-AIC", 0U, 1U, kPrimaryRecords,
            confirm_policies, kFollowupSamples, &single_aic
        );
    }
    if (semantic_ok) {
        PrintComparison("single-AIC", single_aic[0], single_aic[1]);
    }

    std::vector<PolicyAggregate> single_aiv;
    if (semantic_ok) {
        semantic_ok = RunWriteMatrix(
            device, "single-AIV", trace_write_preload::kAicWorkers, 1U,
            kPrimaryRecords, confirm_policies, kFollowupSamples, &single_aiv
        );
    }
    if (semantic_ok) {
        PrintComparison("single-AIV", single_aiv[0], single_aiv[1]);
    }

    if (semantic_ok) {
        semantic_ok = RunCapacitySweep(device, "single-AIC", 0U);
    }
    if (semantic_ok) {
        semantic_ok = RunCapacitySweep(
            device, "single-AIV", trace_write_preload::kAicWorkers
        );
    }

    atomic_probe::Result result;
    result.Expect(
        semantic_ok,
        "mixed trace writes, GM publication, topology, and pointer cycles"
    );
    const bool cleanup_ok = Cleanup(&device, device_id);
    result.Expect(cleanup_ok, "trace-write preload probe cleanup");
    return result.ExitCode();
}
