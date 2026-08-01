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
#include "atomic_poll_exchange_contention_shared.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

namespace {

using namespace atomic_poll_exchange_contention;

constexpr uint32_t kLaunches = 3;

struct KernelArgs {
    uint64_t storage_pointer;
    uint32_t mode;
    uint32_t num_blocks;
};

static_assert(sizeof(KernelArgs) == 16, "unexpected CCEC kernel argument ABI");

void Check(aclError error, const char *label) {
    if (!atomic_probe::CheckAcl(error, label, __FILE__, __LINE__)) {
        std::exit(EXIT_FAILURE);
    }
}

bool OptionalUintEnv(const char *name, uint32_t default_value, uint32_t minimum, uint32_t maximum, uint32_t *value) {
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        *value = default_value;
        return true;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || parsed < minimum || parsed > maximum) {
        std::fprintf(stderr, "%s must be in [%u, %u], got: %s\n", name, minimum, maximum, raw);
        return false;
    }
    *value = static_cast<uint32_t>(parsed);
    return true;
}

const char *ModeName(ProbeMode mode) {
    switch (mode) {
    case ProbeMode::AddDifferentLine:
        return "atomicAdd(0), different 64B line";
    case ProbeMode::AddSameLineDifferentWord:
        return "atomicAdd(0), same line different word";
    case ProbeMode::AddTargetThenAway:
        return "atomicAdd(0), target x31 then away";
    case ProbeMode::AddSameAddress:
        return "atomicAdd(0), same address continuous";
    case ProbeMode::MaxDifferentLine:
        return "atomicMax(INT64_MIN), different 64B line";
    case ProbeMode::MaxSameLineDifferentWord:
        return "atomicMax(INT64_MIN), same line different word";
    case ProbeMode::MaxTargetThenAway:
        return "atomicMax(INT64_MIN), target x31 then away";
    case ProbeMode::MaxSameAddress:
        return "atomicMax(INT64_MIN), same address continuous";
    default:
        return "unknown";
    }
}

bool LineMatches(const AtomicLine &line, int64_t expected_value, int64_t expected_neighbor) {
    if (line.value != expected_value || line.neighbor != expected_neighbor) return false;
    for (int64_t value : line.reserved) {
        if (value != 0) return false;
    }
    return true;
}

bool IsZeroReader(const ReaderResult &reader) {
    return reader.total_pre_polls == 0 && reader.total_followup_polls == 0 && reader.max_observe_ticks == 0 &&
           reader.last_observed == 0 && reader.magic == 0 && reader.block == 0 && reader.core == 0 &&
           reader.subblock == 0 && reader.seen_rounds == 0 && reader.pre_poll_mismatches == 0 &&
           reader.epoch_timeouts == 0 && reader.observe_timeouts == 0;
}

struct TimingStats {
    std::vector<uint64_t> samples;

    void Add(uint64_t value) { samples.push_back(value); }

    uint64_t Percentile(uint32_t numerator, uint32_t denominator) const {
        std::vector<uint64_t> ordered = samples;
        std::sort(ordered.begin(), ordered.end());
        return ordered[(ordered.size() - 1U) * numerator / denominator];
    }

    double Mean() const {
        const long double sum = std::accumulate(samples.begin(), samples.end(), static_cast<long double>(0));
        return static_cast<double>(sum / samples.size());
    }

    void Print(const char *label) const {
        if (samples.empty()) {
            std::printf("[TIMING] %-28s no samples\n", label);
            return;
        }
        const auto bounds = std::minmax_element(samples.begin(), samples.end());
        std::printf(
            "[TIMING] %-28s n=%zu min=%llu p10=%llu p25=%llu p50=%llu "
            "p75=%llu p90=%llu p95=%llu p99=%llu max=%llu "
            "mean=%.3f SYS_CNT ticks (A5: 1 tick = 1 ns)\n",
            label, samples.size(), static_cast<unsigned long long>(*bounds.first),
            static_cast<unsigned long long>(Percentile(10, 100)), static_cast<unsigned long long>(Percentile(25, 100)),
            static_cast<unsigned long long>(Percentile(50, 100)), static_cast<unsigned long long>(Percentile(75, 100)),
            static_cast<unsigned long long>(Percentile(90, 100)), static_cast<unsigned long long>(Percentile(95, 100)),
            static_cast<unsigned long long>(Percentile(99, 100)), static_cast<unsigned long long>(*bounds.second),
            Mean()
        );
    }

    void PrintDistribution(const char *label) const {
        if (samples.empty()) return;
        constexpr std::array<uint64_t, 10> upper_bounds = {224, 319, 383, 447, 639, 959, 1279, 1599, 1999, 2999};
        std::array<size_t, upper_bounds.size() + 1U> counts{};
        for (uint64_t sample : samples) {
            size_t bucket = 0;
            while (bucket < upper_bounds.size() && sample > upper_bounds[bucket])
                ++bucket;
            ++counts[bucket];
        }

        std::printf("[DISTRIBUTION] %s", label);
        uint64_t lower = 0;
        for (size_t bucket = 0; bucket < upper_bounds.size(); ++bucket) {
            const double percent = 100.0 * static_cast<double>(counts[bucket]) / static_cast<double>(samples.size());
            if (bucket == 0) {
                std::printf(
                    " <=%llu:%zu(%.1f%%)", static_cast<unsigned long long>(upper_bounds[bucket]), counts[bucket],
                    percent
                );
            } else {
                std::printf(
                    " %llu-%llu:%zu(%.1f%%)", static_cast<unsigned long long>(lower),
                    static_cast<unsigned long long>(upper_bounds[bucket]), counts[bucket], percent
                );
            }
            lower = upper_bounds[bucket] + 1U;
        }
        const double overflow_percent =
            100.0 * static_cast<double>(counts.back()) / static_cast<double>(samples.size());
        std::printf(" >=%llu:%zu(%.1f%%)\n", static_cast<unsigned long long>(lower), counts.back(), overflow_percent);
    }
};

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <kernel.o>\n", argv[0]);
        return EXIT_FAILURE;
    }

    uint32_t mode_value = 0;
    uint32_t blocks = 0;
    if (!atomic_probe::RequiredUintEnv("ATOMIC_PROBE_MODE", ModeIndex(ProbeMode::Count), &mode_value) ||
        !OptionalUintEnv("ATOMIC_PROBE_AIVS", kDefaultAivs, kMinAivs, kMaxAivs, &blocks)) {
        return EXIT_FAILURE;
    }
    const ProbeMode mode = static_cast<ProbeMode>(mode_value);

    const int32_t device = atomic_probe::DeviceId();
    if (device < 0) return EXIT_FAILURE;
    Check(aclInit(nullptr), "initialize ACL");
    Check(aclrtSetDevice(device), "set atomic poll contention device");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "create atomic poll contention stream");

    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "cannot open kernel file: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    const size_t binary_size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> binary(binary_size);
    file.read(binary.data(), static_cast<std::streamsize>(binary_size));
    if (!file) return EXIT_FAILURE;

    aclrtBinHandle binary_handle;
    Check(
        atomic_probe::LoadAicoreBinaryFromData(binary.data(), binary_size, &binary_handle),
        "load atomic poll contention AICore binary"
    );
    aclrtFuncHandle function_handle;
    Check(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function_handle), "get atomic poll contention kernel entry");

    void *device_storage = nullptr;
    Check(
        aclrtMalloc(&device_storage, sizeof(ProbeStorage), ACL_MEM_MALLOC_HUGE_FIRST),
        "allocate atomic poll contention storage"
    );
    const uintptr_t storage_address = reinterpret_cast<uintptr_t>(device_storage);
    const bool aligned = (storage_address & 63U) == 0U;
    std::printf(
        "=== atomic poll -> completion exchange contention ===\n"
        "mode=%u '%s' launches=%u AIVs=%u readers=%u rounds=%u "
        "pre_polls=%u arm=%llu ticks timeout=%llu ticks\n"
        "storage=0x%zx size=%zu mod64=%zu\n",
        mode_value, ModeName(mode), kLaunches, blocks, blocks - 1U, kRounds, kPrePolls,
        static_cast<unsigned long long>(kArmTicks), static_cast<unsigned long long>(kWaitTimeoutTicks),
        static_cast<size_t>(storage_address), sizeof(ProbeStorage), static_cast<size_t>(storage_address & 63U)
    );
    if (!aligned) {
        std::fprintf(stderr, "probe storage is not 64-byte aligned\n");
        Check(aclrtFree(device_storage), "free unaligned atomic poll contention storage");
        Check(aclrtBinaryUnLoad(binary_handle), "unload atomic poll contention binary");
        Check(aclrtDestroyStream(stream), "destroy atomic poll contention stream");
        Check(aclrtResetDevice(device), "reset atomic poll contention device");
        Check(aclFinalize(), "finalize ACL");
        return EXIT_FAILURE;
    }

    const uint32_t readers = blocks - 1U;
    uint64_t protocol_failures = 0;
    uint64_t total_pre_polls = 0;
    uint64_t total_followup_polls = 0;
    uint64_t total_seen = 0;
    uint64_t total_reader_timeouts = 0;
    TimingStats writer_exchange;
    TimingStats notification_exchange;
    TimingStats write_to_all_ack;

    for (uint32_t launch = 0; launch < kLaunches; ++launch) {
        ProbeStorage host{};
        Check(
            aclrtMemcpy(device_storage, sizeof(host), &host, sizeof(host), ACL_MEMCPY_HOST_TO_DEVICE),
            "initialize atomic poll contention storage"
        );
        KernelArgs args{static_cast<uint64_t>(storage_address), mode_value, blocks};
        Check(
            aclrtLaunchKernelWithHostArgs(function_handle, blocks, stream, nullptr, &args, sizeof(args), nullptr, 0),
            "launch atomic poll contention kernel"
        );
        Check(aclrtSynchronizeStream(stream), "wait for atomic poll contention kernel");
        Check(
            aclrtMemcpy(&host, sizeof(host), device_storage, sizeof(host), ACL_MEMCPY_DEVICE_TO_HOST),
            "read atomic poll contention result"
        );

        const int64_t expected_target = Token(mode, kRounds - 1U);
        const int64_t expected_neighbor = PollsTargetNeighbor(mode) ? expected_target : 0;
        const int64_t expected_signal =
            !PollsTargetUntilPublish(mode) && !PollsTargetNeighbor(mode) ? expected_target : 0;
        const int64_t expected_count = static_cast<int64_t>(readers) * kRounds;
        bool launch_ok = host.header.magic == kHeaderMagic && host.header.mode == mode_value &&
                         host.header.blocks == blocks && host.header.readers == readers &&
                         host.header.rounds == kRounds && host.header.pre_polls == kPrePolls &&
                         host.header.ready_timeouts == 0 && host.header.ack_timeouts == 0 &&
                         host.header.old_value_mismatches == 0 && host.header.timeout_ticks == kWaitTimeoutTicks &&
                         host.header.arm_ticks == kArmTicks && host.header.finish == kFinishMagic &&
                         host.header.final_target_low == static_cast<uint32_t>(expected_target) &&
                         host.header.final_signal_low == static_cast<uint32_t>(expected_signal) &&
                         LineMatches(host.target, expected_target, expected_neighbor) &&
                         LineMatches(host.signal, expected_signal, 0) && LineMatches(host.epoch, kRounds, 0) &&
                         LineMatches(host.ready, expected_count, 0) && LineMatches(host.ack, expected_count, 0) &&
                         LineMatches(host.guard, 0, 0) && IsZeroReader(host.readers[0]);

        for (uint32_t round = 0; round < kRounds; ++round) {
            const TimingRecord &record = host.timing[round];
            const uint32_t expected_progress = readers * (round + 1U);
            const bool timing_ok = record.magic_round == (kTimingMagic | round) && record.flags == 0 &&
                                   record.old_value == 0 && record.ready_actual == expected_progress &&
                                   record.ack_actual == expected_progress && record.writer_end >= record.writer_begin &&
                                   record.all_ack >= record.writer_end;
            launch_ok = launch_ok && timing_ok;
            writer_exchange.Add(record.writer_end - record.writer_begin);
            write_to_all_ack.Add(record.all_ack - record.writer_begin);
            if (PollsTargetUntilPublish(mode)) {
                launch_ok =
                    launch_ok && record.notify_begin == record.writer_end && record.notify_end == record.writer_end;
            } else {
                launch_ok =
                    launch_ok && record.notify_begin >= record.writer_end && record.notify_end >= record.notify_begin;
                notification_exchange.Add(record.notify_end - record.notify_begin);
            }
        }

        std::set<std::pair<uint32_t, uint32_t>> topology;
        topology.emplace(host.header.writer_core, host.header.writer_subblock);
        for (uint32_t block = 1; block < blocks; ++block) {
            const ReaderResult &reader = host.readers[block];
            topology.emplace(reader.core, reader.subblock);
            const bool reader_ok = reader.magic == kReaderMagic && reader.block == block &&
                                   reader.total_pre_polls == static_cast<uint64_t>(kRounds) * kPrePolls &&
                                   reader.total_followup_polls >= kRounds && reader.last_observed == expected_target &&
                                   reader.seen_rounds == kRounds && reader.pre_poll_mismatches == 0 &&
                                   reader.epoch_timeouts == 0 && reader.observe_timeouts == 0;
            launch_ok = launch_ok && reader_ok;
            total_pre_polls += reader.total_pre_polls;
            total_followup_polls += reader.total_followup_polls;
            total_seen += reader.seen_rounds;
            total_reader_timeouts += reader.epoch_timeouts + reader.observe_timeouts;
        }
        launch_ok = launch_ok && topology.size() == blocks;
        for (uint32_t block = blocks; block < kMaxAivs; ++block) {
            launch_ok = launch_ok && IsZeroReader(host.readers[block]);
        }
        if (!launch_ok) ++protocol_failures;
        std::printf(
            "[LAUNCH] index=%u writer=(core=%u sub=%u) topology=%zu/%u "
            "ready=%lld ack=%lld target=0x%016llx neighbor=0x%016llx signal=0x%016llx "
            "status=%s\n",
            launch, host.header.writer_core, host.header.writer_subblock, topology.size(), blocks,
            static_cast<long long>(host.ready.value), static_cast<long long>(host.ack.value),
            static_cast<unsigned long long>(host.target.value), static_cast<unsigned long long>(host.target.neighbor),
            static_cast<unsigned long long>(host.signal.value), launch_ok ? "PASS" : "FAIL"
        );
    }

    const uint64_t expected_pre = static_cast<uint64_t>(kLaunches) * readers * kRounds * kPrePolls;
    const uint64_t expected_seen = static_cast<uint64_t>(kLaunches) * readers * kRounds;
    std::printf(
        "[COUNTS] pre_polls=%llu/%llu followup_polls=%llu "
        "seen=%llu/%llu reader_timeouts=%llu\n",
        static_cast<unsigned long long>(total_pre_polls), static_cast<unsigned long long>(expected_pre),
        static_cast<unsigned long long>(total_followup_polls), static_cast<unsigned long long>(total_seen),
        static_cast<unsigned long long>(expected_seen), static_cast<unsigned long long>(total_reader_timeouts)
    );
    writer_exchange.Print("target AtomicExch");
    writer_exchange.PrintDistribution("target AtomicExch");
    if (!PollsTargetUntilPublish(mode)) {
        notification_exchange.Print("active-poll notify AtomicExch");
    }
    write_to_all_ack.Print("target write -> all ack");
    std::printf(
        "[METRIC] mode=%u aivs=%u readers=%u writer_exchange_mean_ns=%.3f "
        "writer_exchange_p50_ns=%llu writer_exchange_p95_ns=%llu "
        "write_to_all_ack_mean_ns=%.3f followup_polls_per_reader_round=%.3f\n",
        mode_value, blocks, readers, writer_exchange.Mean(),
        static_cast<unsigned long long>(writer_exchange.Percentile(50, 100)),
        static_cast<unsigned long long>(writer_exchange.Percentile(95, 100)), write_to_all_ack.Mean(),
        static_cast<double>(total_followup_polls) / static_cast<double>(expected_seen)
    );

    atomic_probe::Result result;
    result.Expect(protocol_failures == 0, "布局、轮次、地址终值、旧值、控制计数与参与拓扑必须精确");
    result.Expect(total_pre_polls == expected_pre, "每个 reader 每轮必须先完成精确 31 次 identity RMW load");
    result.Expect(
        total_seen == expected_seen && total_reader_timeouts == 0,
        "每个 reader 必须逐轮看到 completion token 且不得超时"
    );

    Check(aclrtFree(device_storage), "free atomic poll contention storage");
    Check(aclrtBinaryUnLoad(binary_handle), "unload atomic poll contention binary");
    Check(aclrtDestroyStream(stream), "destroy atomic poll contention stream");
    Check(aclrtResetDevice(device), "reset atomic poll contention device");
    Check(aclFinalize(), "finalize ACL");
    return result.ExitCode();
}
