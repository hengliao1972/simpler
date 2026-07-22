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
#include "st_dev_ld_dev_sync_shared.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

namespace {

using st_dev_ld_dev_sync::ProbeStorage;
using st_dev_ld_dev_sync::SignalLine;
using st_dev_ld_dev_sync::WorkerResult;

constexpr uint32_t kWarmupLaunches = 10;
constexpr uint32_t kMeasuredLaunches = 200;
constexpr double kSysCntFrequencyHz = 1000000000.0;

struct KernelArgs {
    uint64_t storage_pointer;
};

struct Sample {
    uint64_t overall_ticks;
    uint64_t start_skew_ticks;
    uint64_t writer_st_dev_ticks;
    uint64_t last_reader_observe_ticks;
    uint64_t final_arrival_skew_ticks;
    uint64_t final_release_ticks;
    uint64_t end_skew_ticks;
    uint64_t maximum_reader_polls;
};

struct Distribution {
    uint64_t minimum;
    uint64_t p50;
    uint64_t p95;
    uint64_t maximum;
    double mean;
};

static_assert(sizeof(KernelArgs) == 8, "unexpected CCEC kernel argument ABI");

void Check(aclError error, const char *label)
{
    if (!atomic_probe::CheckAcl(error, label, __FILE__, __LINE__)) {
        std::exit(EXIT_FAILURE);
    }
}

std::vector<char> ReadBinary(const char *path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return {};
    }
    const std::streamsize size = stream.tellg();
    if (size <= 0) {
        return {};
    }
    stream.seekg(0);
    std::vector<char> data(static_cast<size_t>(size));
    if (!stream.read(data.data(), size)) {
        return {};
    }
    return data;
}

bool ReadWorkerCount(uint32_t *workers)
{
    const char *raw = std::getenv("ATOMIC_PROBE_AIVS");
    if (raw == nullptr || raw[0] == '\0') {
        *workers = st_dev_ld_dev_sync::kDefaultAivWorkers;
        return true;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' ||
        parsed < st_dev_ld_dev_sync::kMinAivWorkers ||
        parsed > st_dev_ld_dev_sync::kMaxAivWorkers) {
        std::fprintf(
            stderr, "ATOMIC_PROBE_AIVS must be in [%u, %u], got: %s\n",
            st_dev_ld_dev_sync::kMinAivWorkers,
            st_dev_ld_dev_sync::kMaxAivWorkers, raw);
        return false;
    }
    *workers = static_cast<uint32_t>(parsed);
    return true;
}

bool SignalPaddingIsZero(const SignalLine &line)
{
    return std::all_of(std::begin(line.padding), std::end(line.padding),
                       [](uint8_t value) { return value == 0U; });
}

bool WorkerResultIsZero(const WorkerResult &worker)
{
    const auto *bytes = reinterpret_cast<const uint8_t *>(&worker);
    return std::all_of(bytes, bytes + sizeof(worker),
                       [](uint8_t value) { return value == 0U; });
}

Distribution Summarize(std::vector<uint64_t> values)
{
    std::sort(values.begin(), values.end());
    const uint64_t sum = std::accumulate(values.begin(), values.end(), uint64_t{0});
    return Distribution{
        values.front(),
        values[(values.size() - 1U) * 50U / 100U],
        values[(values.size() - 1U) * 95U / 100U],
        values.back(),
        static_cast<double>(sum) / static_cast<double>(values.size()),
    };
}

void PrintDistribution(const char *label, const std::vector<uint64_t> &values)
{
    const Distribution distribution = Summarize(values);
    std::printf(
        "[METRIC] %-24s n=%zu min=%" PRIu64 " p50=%" PRIu64
        " p95=%" PRIu64 " max=%" PRIu64 " mean=%.1f SYS_CNT ticks"
        " (p50=%.3f us @1GHz SYS_CNT)\n",
        label, values.size(), distribution.minimum, distribution.p50,
        distribution.p95, distribution.maximum, distribution.mean,
        static_cast<double>(distribution.p50) /
            (kSysCntFrequencyHz / 1000000.0));
}

void PrintCountDistribution(const char *label,
                            const std::vector<uint64_t> &values)
{
    const Distribution distribution = Summarize(values);
    std::printf(
        "[METRIC] %-24s n=%zu min=%" PRIu64 " p50=%" PRIu64
        " p95=%" PRIu64 " max=%" PRIu64 " mean=%.1f polls\n",
        label, values.size(), distribution.minimum, distribution.p50,
        distribution.p95, distribution.maximum, distribution.mean);
}

bool ValidateAndMeasure(const ProbeStorage &storage, uint32_t launch,
                        uint32_t workers, bool print_topology, Sample *sample)
{
    bool valid = true;
    auto Mismatch = [&](uint32_t block, const char *field, uint64_t actual,
                        uint64_t expected) {
        std::fprintf(stderr,
            "[MISMATCH] launch=%u block=%u field=%s actual=%" PRIu64
            " expected=%" PRIu64 "\n",
            launch, block, field, actual, expected);
        valid = false;
    };

    if (storage.signal.value != st_dev_ld_dev_sync::kExpectedValue) {
        Mismatch(0U, "signal.value", storage.signal.value,
                 st_dev_ld_dev_sync::kExpectedValue);
    }
    if (!SignalPaddingIsZero(storage.signal)) {
        Mismatch(0U, "signal.padding", 1U, 0U);
    }
    if (storage.guard.value != 0U || !SignalPaddingIsZero(storage.guard)) {
        Mismatch(0U, "guard", 1U, 0U);
    }

    uint64_t minimum_begin = UINT64_MAX;
    uint64_t maximum_begin = 0;
    uint64_t maximum_reader_observe = 0;
    uint64_t minimum_final_arrive = UINT64_MAX;
    uint64_t maximum_final_arrive = 0;
    uint64_t minimum_end = UINT64_MAX;
    uint64_t maximum_end = 0;
    uint64_t maximum_reader_polls = 0;
    std::set<std::pair<uint32_t, uint32_t>> physical_aivs;

    if (print_topology) {
        std::printf("[TOPOLOGY]");
    }
    for (uint32_t block = 0; block < workers; ++block) {
        const WorkerResult &worker = storage.workers[block];
        if (print_topology) {
            std::printf(" block%u=(core%u,sub%u)", block, worker.core_id,
                        worker.subblock_id);
        }
        physical_aivs.emplace(worker.core_id, worker.subblock_id);
        if (worker.magic != st_dev_ld_dev_sync::kResultMagic) {
            Mismatch(block, "magic", worker.magic,
                     st_dev_ld_dev_sync::kResultMagic);
        }
        if (worker.block_id != block) {
            Mismatch(block, "block_id", worker.block_id, block);
        }
        if (worker.block_count != workers) {
            Mismatch(block, "block_count", worker.block_count, workers);
        }
        if (worker.observed_value != st_dev_ld_dev_sync::kExpectedValue) {
            Mismatch(block, "observed_value", worker.observed_value,
                     st_dev_ld_dev_sync::kExpectedValue);
        }
        if (worker.flags != 0U) {
            Mismatch(block, "flags", worker.flags, 0U);
        }
        if (block == 0U && worker.poll_count != 0U) {
            Mismatch(block, "writer.poll_count", worker.poll_count, 0U);
        }
        if (block != 0U && worker.poll_count == 0U) {
            Mismatch(block, "reader.poll_count", worker.poll_count, 1U);
        }
        if (worker.begin_tick == 0U ||
            worker.observe_tick < worker.begin_tick ||
            worker.final_barrier_arrive_tick < worker.observe_tick ||
            worker.end_tick < worker.final_barrier_arrive_tick) {
            std::fprintf(stderr,
                "[MISMATCH] launch=%u block=%u invalid timing order: begin=%" PRIu64
                " observe=%" PRIu64 " final_arrive=%" PRIu64 " end=%" PRIu64 "\n",
                launch, block, worker.begin_tick, worker.observe_tick,
                worker.final_barrier_arrive_tick, worker.end_tick);
            valid = false;
        }
        minimum_begin = std::min(minimum_begin, worker.begin_tick);
        maximum_begin = std::max(maximum_begin, worker.begin_tick);
        minimum_final_arrive = std::min(
            minimum_final_arrive, worker.final_barrier_arrive_tick);
        maximum_final_arrive = std::max(
            maximum_final_arrive, worker.final_barrier_arrive_tick);
        minimum_end = std::min(minimum_end, worker.end_tick);
        maximum_end = std::max(maximum_end, worker.end_tick);
        if (block != 0U) {
            maximum_reader_observe = std::max(
                maximum_reader_observe, worker.observe_tick);
            maximum_reader_polls = std::max<uint64_t>(
                maximum_reader_polls, worker.poll_count);
        }
    }
    if (print_topology) {
        std::printf("\n");
    }
    for (uint32_t block = workers;
         block < st_dev_ld_dev_sync::kMaxAivWorkers; ++block) {
        if (!WorkerResultIsZero(storage.workers[block])) {
            Mismatch(block, "unused_worker_result", 1U, 0U);
        }
    }
    if (physical_aivs.size() != workers) {
        Mismatch(0U, "unique_physical_aivs", physical_aivs.size(),
                 workers);
    }

    const WorkerResult &writer = storage.workers[0];
    if (maximum_end < minimum_begin || maximum_begin < minimum_begin ||
        maximum_reader_observe < writer.begin_tick ||
        maximum_final_arrive < minimum_final_arrive ||
        maximum_end < maximum_final_arrive || minimum_end > maximum_end) {
        std::fprintf(stderr, "[MISMATCH] launch=%u cross-core timing order invalid\n",
                     launch);
        valid = false;
    }

    sample->overall_ticks = maximum_end - minimum_begin;
    sample->start_skew_ticks = maximum_begin - minimum_begin;
    sample->writer_st_dev_ticks = writer.observe_tick - writer.begin_tick;
    sample->last_reader_observe_ticks =
        maximum_reader_observe - writer.begin_tick;
    sample->final_arrival_skew_ticks =
        maximum_final_arrive - minimum_final_arrive;
    sample->final_release_ticks = maximum_end - maximum_final_arrive;
    sample->end_skew_ticks = maximum_end - minimum_end;
    sample->maximum_reader_polls = maximum_reader_polls;
    return valid;
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <kernel.o>\n", argv[0]);
        return EXIT_FAILURE;
    }

    uint32_t workers = 0;
    if (!ReadWorkerCount(&workers)) {
        return EXIT_FAILURE;
    }

    const std::vector<char> binary = ReadBinary(argv[1]);
    if (binary.empty()) {
        std::fprintf(stderr, "Cannot read kernel binary: %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    const int32_t device = atomic_probe::DeviceId();
    if (device < 0) {
        return EXIT_FAILURE;
    }
    Check(aclInit(nullptr), "initialize ACL");
    Check(aclrtSetDevice(device), "set probe device");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "create probe stream");

    aclrtBinHandle binary_handle;
    Check(atomic_probe::LoadAicoreBinaryFromData(
              binary.data(), binary.size(), &binary_handle),
          "load st_dev/ld_dev sync binary");
    aclrtFuncHandle function_handle;
    Check(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function_handle),
          "get st_dev/ld_dev sync entry");

    void *device_storage = nullptr;
    Check(aclrtMalloc(&device_storage, sizeof(ProbeStorage),
                      ACL_MEM_MALLOC_HUGE_FIRST),
          "allocate st_dev/ld_dev sync storage");
    const uintptr_t storage_address =
        reinterpret_cast<uintptr_t>(device_storage);
    if ((storage_address & 63U) != 0U) {
        std::fprintf(stderr,
            "Probe storage is not 64-byte aligned: 0x%zx\n",
            static_cast<size_t>(storage_address));
        return EXIT_FAILURE;
    }

    std::printf(
        "=== %u-AIV st_dev/ld_dev one-shot synchronization ===\n"
        "writer=block0 readers=blocks1..%u publish='raw st_dev, no explicit DSB' "
        "poll='ld_dev until exact value'\n"
        "layout=signal owns one exclusive 64B cache line; results and guard use separate lines\n"
        "measurement=after initial SyncAll -> publish/poll -> after final "
        "SyncAll; result stores excluded\n"
        "warmup_launches=%u measured_launches=%u timeout=%" PRIu64
        " SYS_CNT ticks SYS_CNT_frequency=1GHz device=%d\n",
        workers, workers - 1U, kWarmupLaunches, kMeasuredLaunches,
        st_dev_ld_dev_sync::kWaitTimeoutTicks, device);

    std::array<std::vector<uint64_t>, 8> metrics;
    bool all_valid = true;
    const uint32_t total_launches = kWarmupLaunches + kMeasuredLaunches;
    for (uint32_t launch = 0; launch < total_launches; ++launch) {
        ProbeStorage host_storage{};
        Check(aclrtMemcpy(device_storage, sizeof(host_storage), &host_storage,
                          sizeof(host_storage), ACL_MEMCPY_HOST_TO_DEVICE),
              "initialize st_dev/ld_dev sync storage");
        KernelArgs args{static_cast<uint64_t>(storage_address)};
        Check(aclrtLaunchKernelWithHostArgs(
                  function_handle, workers, stream,
                  nullptr, &args, sizeof(args), nullptr, 0),
              "launch st_dev/ld_dev sync kernel");
        Check(aclrtSynchronizeStream(stream),
              "wait for st_dev/ld_dev sync kernel");
        Check(aclrtMemcpy(&host_storage, sizeof(host_storage), device_storage,
                          sizeof(host_storage), ACL_MEMCPY_DEVICE_TO_HOST),
              "read st_dev/ld_dev sync result");

        Sample sample{};
        const bool valid = ValidateAndMeasure(
            host_storage, launch, workers, launch == 0U, &sample);
        all_valid = all_valid && valid;
        if (launch >= kWarmupLaunches && valid) {
            metrics[0].push_back(sample.overall_ticks);
            metrics[1].push_back(sample.start_skew_ticks);
            metrics[2].push_back(sample.writer_st_dev_ticks);
            metrics[3].push_back(sample.last_reader_observe_ticks);
            metrics[4].push_back(sample.final_arrival_skew_ticks);
            metrics[5].push_back(sample.final_release_ticks);
            metrics[6].push_back(sample.end_skew_ticks);
            metrics[7].push_back(sample.maximum_reader_polls);
        }
    }

    atomic_probe::Result result;
    char semantics_label[96];
    std::snprintf(
        semantics_label, sizeof(semantics_label),
        "all launches preserve one-writer/%u-reader semantics", workers - 1U);
    result.Expect(all_valid, semantics_label);
    result.Expect(metrics[0].size() == kMeasuredLaunches,
                  "all measured launches produced valid samples");
    if (metrics[0].size() == kMeasuredLaunches) {
        PrintDistribution("overall", metrics[0]);
        PrintDistribution("initial_start_skew", metrics[1]);
        PrintDistribution("writer_st_dev_span", metrics[2]);
        PrintDistribution("last_reader_observe", metrics[3]);
        PrintDistribution("final_arrival_skew", metrics[4]);
        PrintDistribution("final_sync_release", metrics[5]);
        PrintDistribution("final_end_skew", metrics[6]);
        PrintCountDistribution("maximum_reader_polls", metrics[7]);
    }

    Check(aclrtFree(device_storage), "free st_dev/ld_dev sync storage");
    Check(aclrtBinaryUnLoad(binary_handle), "unload st_dev/ld_dev sync binary");
    Check(aclrtDestroyStream(stream), "destroy st_dev/ld_dev sync stream");
    Check(aclrtResetDevice(device), "reset probe device");
    Check(aclFinalize(), "finalize ACL");
    return result.ExitCode();
}
