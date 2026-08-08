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

#include "../common/warp_concurrency_probe.h"

#include "acl/acl.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace pa_scheduler::simt_cross_core::warp_concurrency;

constexpr uint32_t kGuardCount = 5U;
constexpr uint32_t kAtomicCellCount = 2U;
constexpr std::array<ProbeMode, 4U> kModes{
    ProbeMode::kAOnly,
    ProbeMode::kBOnly,
    ProbeMode::kSameWarp,
    ProbeMode::kCrossWarp,
};

struct Options {
    std::string kernel_path;
    int32_t device = 0;
    uint32_t runs = 100U;
};

bool ParseUnsigned(const char *raw, uint64_t maximum, uint64_t *value) {
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || parsed > maximum) {
        return false;
    }
    *value = static_cast<uint64_t>(parsed);
    return true;
}

bool ParseOptions(int argc, char **argv, Options *options) {
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--kernel") == 0 && index + 1 < argc) {
            options->kernel_path = argv[++index];
        } else if (std::strcmp(argv[index], "--device") == 0 && index + 1 < argc) {
            uint64_t value = 0U;
            if (!ParseUnsigned(argv[++index], static_cast<uint64_t>(std::numeric_limits<int32_t>::max()), &value)) {
                std::fprintf(stderr, "invalid --device value: %s\n", argv[index]);
                return false;
            }
            options->device = static_cast<int32_t>(value);
        } else if (std::strcmp(argv[index], "--runs") == 0 && index + 1 < argc) {
            uint64_t value = 0U;
            if (!ParseUnsigned(argv[++index], 100000U, &value) || value == 0U) {
                std::fprintf(stderr, "invalid --runs value: %s\n", argv[index]);
                return false;
            }
            options->runs = static_cast<uint32_t>(value);
        } else {
            std::fprintf(stderr, "usage: %s --kernel FILE [--device N] [--runs N]\n", argv[0]);
            return false;
        }
    }
    if (options->kernel_path.empty()) {
        std::fprintf(stderr, "--kernel FILE is required\n");
        return false;
    }
    return true;
}

bool CheckAcl(aclError error, const char *operation) {
    if (error == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "ACL error %d: %s\n", static_cast<int>(error), operation);
    return false;
}

bool ReadBinary(const std::string &path, std::vector<char> *data) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        std::fprintf(stderr, "cannot open kernel binary: %s\n", path.c_str());
        return false;
    }
    const std::streamsize size = input.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "kernel binary is empty: %s\n", path.c_str());
        return false;
    }
    input.seekg(0, std::ios::beg);
    data->resize(static_cast<size_t>(size));
    if (!input.read(data->data(), size)) {
        std::fprintf(stderr, "cannot read kernel binary: %s\n", path.c_str());
        return false;
    }
    return true;
}

class AclSession {
public:
    ~AclSession() {
        if (device_state_ != nullptr) {
            (void)aclrtFree(device_state_);
        }
        if (binary_ != nullptr) {
            (void)aclrtBinaryUnLoad(binary_);
        }
        if (stream_ != nullptr) {
            (void)aclrtDestroyStream(stream_);
        }
        if (device_set_) {
            (void)aclrtResetDevice(device_);
        }
        if (acl_initialized_) {
            (void)aclFinalize();
        }
    }

    bool Initialize(int32_t device, const std::vector<char> &binary_data) {
        device_ = device;
        if (!CheckAcl(aclInit(nullptr), "aclInit")) {
            return false;
        }
        acl_initialized_ = true;
        if (!CheckAcl(aclrtSetDevice(device_), "aclrtSetDevice")) {
            return false;
        }
        device_set_ = true;
        const char *soc_name = aclrtGetSocName();
        if (soc_name == nullptr || soc_name[0] == '\0') {
            std::fprintf(stderr, "aclrtGetSocName returned an empty SoC name\n");
            return false;
        }
        soc_name_ = soc_name;
        if (!CheckAcl(aclrtCreateStream(&stream_), "aclrtCreateStream")) {
            return false;
        }

        aclrtBinaryLoadOption option{};
        option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
        option.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
        aclrtBinaryLoadOptions options{&option, 1U};
        return CheckAcl(
                   aclrtBinaryLoadFromData(binary_data.data(), binary_data.size(), &options, &binary_),
                   "aclrtBinaryLoadFromData(AIV-only warp concurrency ELF)"
               ) &&
               CheckAcl(aclrtBinaryGetFunctionByEntry(binary_, 0U, &function_), "aclrtBinaryGetFunctionByEntry") &&
               CheckAcl(
                   aclrtMalloc(&device_state_, sizeof(ProbeState), ACL_MEM_MALLOC_HUGE_FIRST),
                   "aclrtMalloc(warp concurrency ProbeState)"
               );
    }

    const std::string &SocName() const { return soc_name_; }

    bool Run(ProbeState *state) {
        if (!CheckAcl(
                aclrtMemcpy(device_state_, sizeof(*state), state, sizeof(*state), ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy(H2D warp concurrency ProbeState)"
            )) {
            return false;
        }
        struct KernelArgs {
            uint64_t state_pointer;
        } args{reinterpret_cast<uint64_t>(device_state_)};
        static_assert(sizeof(KernelArgs) == sizeof(uint64_t), "unexpected CCEC kernel argument ABI");
        return CheckAcl(
                   aclrtLaunchKernelWithHostArgs(function_, 1U, stream_, nullptr, &args, sizeof(args), nullptr, 0U),
                   "aclrtLaunchKernelWithHostArgs(warp concurrency)"
               ) &&
               CheckAcl(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream(warp concurrency)") &&
               CheckAcl(
                   aclrtMemcpy(state, sizeof(*state), device_state_, sizeof(*state), ACL_MEMCPY_DEVICE_TO_HOST),
                   "aclrtMemcpy(D2H warp concurrency ProbeState)"
               );
    }

private:
    bool acl_initialized_ = false;
    bool device_set_ = false;
    int32_t device_ = 0;
    std::string soc_name_;
    aclrtStream stream_ = nullptr;
    aclrtBinHandle binary_ = nullptr;
    aclrtFuncHandle function_ = nullptr;
    void *device_state_ = nullptr;
};

const char *ModeName(ProbeMode mode) {
    switch (mode) {
    case ProbeMode::kAOnly:
        return "AOnly";
    case ProbeMode::kBOnly:
        return "BOnly";
    case ProbeMode::kSameWarp:
        return "SameWarp";
    case ProbeMode::kCrossWarp:
        return "CrossWarp";
    }
    return "Invalid";
}

char HandshakeCode(uint64_t raw_status) {
    if (raw_status == static_cast<uint64_t>(HandshakeStatus::kSuccess)) {
        return 'S';
    }
    if (raw_status == static_cast<uint64_t>(HandshakeStatus::kTimeout)) {
        return 'T';
    }
    if (raw_status == static_cast<uint64_t>(HandshakeStatus::kNotApplicable)) {
        return 'N';
    }
    return '?';
}

uint64_t ExpectedGuardWord(uint64_t nonce, uint32_t guard, uint32_t word) {
    return 0xD15EA5E000000000ULL ^ RotateLeft(nonce, (guard * 7U + word) % 31U + 1U) ^
           (static_cast<uint64_t>(guard) << 24U) ^ static_cast<uint64_t>(word);
}

uint64_t ExpectedAtomicPadding(uint64_t nonce, uint32_t cell, uint32_t word) {
    return 0xC311C31100000000ULL ^ RotateLeft(nonce, (cell * 9U + word) % 29U + 1U) ^
           (static_cast<uint64_t>(cell) << 20U) ^ static_cast<uint64_t>(word);
}

uint64_t ExpectedControlReserved(uint64_t nonce) { return 0xC047A01C00000000ULL ^ RotateLeft(nonce, 23U); }

void InitializeGuard(ProbeGuard *guard, uint64_t nonce, uint32_t guard_index) {
    for (uint32_t word = 0U; word < 8U; ++word) {
        guard->words[word] = ExpectedGuardWord(nonce, guard_index, word);
    }
}

bool GuardValid(const ProbeGuard &guard, uint64_t nonce, uint32_t guard_index) {
    for (uint32_t word = 0U; word < 8U; ++word) {
        if (guard.words[word] != ExpectedGuardWord(nonce, guard_index, word)) {
            return false;
        }
    }
    return true;
}

void InitializeAtomicCell(AtomicCell *cell, uint64_t nonce, uint32_t cell_index) {
    cell->value = 0U;
    for (uint32_t word = 0U; word < 7U; ++word) {
        cell->padding[word] = ExpectedAtomicPadding(nonce, cell_index, word);
    }
}

bool AtomicPaddingValid(const AtomicCell &cell, uint64_t nonce, uint32_t cell_index) {
    for (uint32_t word = 0U; word < 7U; ++word) {
        if (cell.padding[word] != ExpectedAtomicPadding(nonce, cell_index, word)) {
            return false;
        }
    }
    return true;
}

template <typename T>
void FillWords(T *object, uint64_t value) {
    static_assert(sizeof(T) % sizeof(uint64_t) == 0U, "sentinel object must contain full uint64 words");
    uint64_t *words = reinterpret_cast<uint64_t *>(object);
    std::fill(words, words + sizeof(T) / sizeof(uint64_t), value);
}

bool AllWordsEqual(const ParticipantReport &report, uint64_t value) {
    const uint64_t *words = reinterpret_cast<const uint64_t *>(&report);
    return std::all_of(words, words + sizeof(report) / sizeof(uint64_t), [value](uint64_t word) {
        return word == value;
    });
}

void InitializeState(ProbeState *state, uint64_t nonce, ProbeMode mode) {
    *state = ProbeState{};
    state->control.magic = kProbeMagic;
    state->control.version = kProbeVersion;
    state->control.mode = static_cast<uint64_t>(mode);
    state->control.launch_nonce = nonce;
    state->control.poll_limit = kDefaultPollLimit;
    state->control.poll_clock_budget = kDefaultPollClockBudget;
    state->control.work_iterations = kDefaultWorkIterations;
    state->control.reserved = ExpectedControlReserved(nonce);

    InitializeAtomicCell(&state->ready_a, nonce, 0U);
    InitializeAtomicCell(&state->ready_b, nonce, 1U);
    InitializeGuard(&state->guard_before_ready_a, nonce, 0U);
    InitializeGuard(&state->guard_before_ready_b, nonce, 1U);
    InitializeGuard(&state->guard_before_reports, nonce, 2U);
    InitializeGuard(&state->guard_after_reports, nonce, 3U);
    InitializeGuard(&state->guard_after_result, nonce, 4U);
    for (ParticipantReport &report : state->reports) {
        FillWords(&report, kOutputSentinel);
    }
    FillWords(&state->result, kOutputSentinel);
}

struct Validation {
    bool passed = false;
    std::string reason;
    uint64_t span = 0U;
    std::array<uint64_t, kParticipantCount> participant_duration{};
};

Validation Fail(const char *reason) { return Validation{false, reason, 0U, {0U, 0U}}; }

Validation Validate(const ProbeState &state, uint64_t nonce, ProbeMode mode) {
    const uint64_t raw_mode = static_cast<uint64_t>(mode);
    if (state.control.magic != kProbeMagic || state.control.version != kProbeVersion ||
        state.control.mode != raw_mode || state.control.launch_nonce != nonce ||
        state.control.poll_limit != kDefaultPollLimit || state.control.poll_clock_budget != kDefaultPollClockBudget ||
        state.control.work_iterations != kDefaultWorkIterations ||
        state.control.reserved != ExpectedControlReserved(nonce)) {
        return Fail("control mismatch");
    }
    const std::array<const ProbeGuard *, kGuardCount> guards{
        &state.guard_before_ready_a, &state.guard_before_ready_b, &state.guard_before_reports,
        &state.guard_after_reports,  &state.guard_after_result,
    };
    for (uint32_t guard = 0U; guard < guards.size(); ++guard) {
        if (!GuardValid(*guards[guard], nonce, guard)) {
            return Fail("guard corruption");
        }
    }
    const std::array<const AtomicCell *, kAtomicCellCount> cells{&state.ready_a, &state.ready_b};
    for (uint32_t cell = 0U; cell < cells.size(); ++cell) {
        if (!AtomicPaddingValid(*cells[cell], nonce, cell)) {
            return Fail("atomic-cell padding corruption");
        }
    }

    const ProbeResult &result = state.result;
    if (result.magic != kResultMagic || result.status != kExpectedStatus || result.physical_core_id > 0x0FFFU ||
        result.subblock_id > 1U || result.launch_nonce != nonce || result.mode != raw_mode || result.reserved != 0U) {
        return Fail("result identity/status mismatch");
    }

    const bool handshake_mode = mode == ProbeMode::kSameWarp || mode == ProbeMode::kCrossWarp;
    const uint64_t expected_ready_a = handshake_mode ? ExpectedReady(nonce, 0U) : 0U;
    const uint64_t expected_ready_b = handshake_mode ? ExpectedReady(nonce, 1U) : 0U;
    if (state.ready_a.value != expected_ready_a || state.ready_b.value != expected_ready_b ||
        result.ready_a != expected_ready_a || result.ready_b != expected_ready_b) {
        return Fail("ready value mismatch");
    }

    std::array<uint64_t, kParticipantCount> starts{};
    std::array<uint64_t, kParticipantCount> ends{};
    std::array<uint64_t, kParticipantCount> durations{};
    uint64_t active_reports = 0U;
    uint64_t successes = 0U;
    uint64_t timeouts = 0U;
    uint64_t not_applicable = 0U;
    for (uint32_t participant = 0U; participant < kParticipantCount; ++participant) {
        const bool active = ParticipantIsActive(mode, participant);
        const ParticipantReport &report = state.reports[participant];
        if (!active) {
            if (!AllWordsEqual(report, kOutputSentinel)) {
                return Fail("inactive report was written");
            }
            continue;
        }

        ++active_reports;
        const uint32_t expected_thread = ExpectedThread(mode, participant);
        if (report.marker != ExpectedReportMarker(nonce, mode, participant) || report.launch_nonce != nonce ||
            report.mode != raw_mode || report.participant != participant || report.thread_id != expected_thread ||
            report.warp_id != expected_thread / kWarpSize || report.lane_id != expected_thread % kWarpSize ||
            report.reserved[0] != 0U || report.reserved[1] != 0U) {
            return Fail("participant identity/report mismatch");
        }
        if (report.end_clock <= report.start_clock) {
            return Fail("non-positive CLOCK64 interval");
        }
        starts[participant] = report.start_clock;
        ends[participant] = report.end_clock;
        durations[participant] = report.end_clock - report.start_clock;

        const uint64_t expected_checksum =
            participant == 0U ? WorkA(nonce, kDefaultWorkIterations) : WorkB(nonce, kDefaultWorkIterations);
        if (report.checksum != expected_checksum) {
            return Fail("work checksum mismatch");
        }

        const uint64_t handshake = report.handshake_status;
        successes += handshake == static_cast<uint64_t>(HandshakeStatus::kSuccess) ? 1U : 0U;
        timeouts += handshake == static_cast<uint64_t>(HandshakeStatus::kTimeout) ? 1U : 0U;
        not_applicable += handshake == static_cast<uint64_t>(HandshakeStatus::kNotApplicable) ? 1U : 0U;
        if (!handshake_mode) {
            if (handshake != static_cast<uint64_t>(HandshakeStatus::kNotApplicable) || report.poll_count != 0U ||
                report.observed_peer != kOutputSentinel || report.published_ready != 0U) {
                return Fail("single-participant handshake fields mismatch");
            }
            continue;
        }

        const uint64_t peer_ready = ExpectedReady(nonce, 1U - participant);
        if (report.published_ready != ExpectedReady(nonce, participant) || report.poll_count == 0U ||
            report.poll_count > kDefaultPollLimit) {
            return Fail("handshake publish/poll range mismatch");
        }
        if (handshake == static_cast<uint64_t>(HandshakeStatus::kSuccess)) {
            if (report.observed_peer != peer_ready) {
                return Fail("successful handshake did not observe peer ready");
            }
        } else if (handshake == static_cast<uint64_t>(HandshakeStatus::kTimeout)) {
            if (mode != ProbeMode::kSameWarp || report.observed_peer != 0U) {
                return Fail("unexpected handshake timeout observation");
            }
        } else {
            return Fail("invalid handshake status");
        }
    }

    const uint64_t expected_active = ExpectedActiveParticipants(mode);
    const uint64_t expected_successes = mode == ProbeMode::kSameWarp ? 1U : mode == ProbeMode::kCrossWarp ? 2U : 0U;
    const uint64_t expected_timeouts = mode == ProbeMode::kSameWarp ? 1U : 0U;
    const uint64_t expected_not_applicable = handshake_mode ? 0U : 1U;
    if (active_reports != expected_active || successes != expected_successes || timeouts != expected_timeouts ||
        not_applicable != expected_not_applicable || result.active_reports != active_reports ||
        result.handshake_successes != successes || result.handshake_timeouts != timeouts ||
        result.handshake_not_applicable != not_applicable) {
        return Fail("handshake/result aggregate mismatch");
    }

    uint64_t expected_first = 0U;
    uint64_t expected_last = 0U;
    IntervalRelation expected_relation = IntervalRelation::kUnknown;
    if (!handshake_mode) {
        const uint32_t participant = mode == ProbeMode::kAOnly ? 0U : 1U;
        expected_first = starts[participant];
        expected_last = ends[participant];
        expected_relation = IntervalRelation::kSingle;
    } else {
        const bool overlap = starts[0] < ends[1] && starts[1] < ends[0];
        expected_first = std::min(starts[0], starts[1]);
        expected_last = std::max(ends[0], ends[1]);
        expected_relation = overlap ? IntervalRelation::kOverlap : IntervalRelation::kDisjoint;
        if ((mode == ProbeMode::kSameWarp && overlap) || (mode == ProbeMode::kCrossWarp && !overlap)) {
            return Fail("CLOCK64 interval relation contradicts mode");
        }
    }
    if (result.interval_relation != static_cast<uint64_t>(expected_relation) || result.first_start != expected_first ||
        result.last_end != expected_last || expected_last <= expected_first) {
        return Fail("CLOCK64 result interval mismatch");
    }

    return Validation{true, {}, expected_last - expected_first, durations};
}

uint64_t MakeNonce(uint32_t run, ProbeMode mode) {
    return 0xA150000000000000ULL | (static_cast<uint64_t>(run + 1U) << 8U) | static_cast<uint64_t>(mode);
}

struct ClockStatistics {
    uint64_t count = 0U;
    uint64_t minimum = std::numeric_limits<uint64_t>::max();
    uint64_t maximum = 0U;
    long double total = 0.0L;

    void Add(uint64_t value) {
        ++count;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        total += static_cast<long double>(value);
    }

    long double Average() const { return count == 0U ? 0.0L : total / static_cast<long double>(count); }
    uint64_t Minimum() const { return count == 0U ? 0U : minimum; }
};

struct ModeStatistics {
    uint32_t passes = 0U;
    ClockStatistics span;
    std::array<ClockStatistics, kParticipantCount> participant;
};

struct ModeObservation {
    Validation validation;
    std::array<uint64_t, kParticipantCount> polls{};
    std::array<uint64_t, kParticipantCount> handshakes{};
};

void PrintRound(uint32_t run, const std::array<ModeObservation, kModes.size()> &observations) {
    std::printf("[ROUND] run=%u", run + 1U);
    for (uint32_t index = 0U; index < kModes.size(); ++index) {
        const ProbeMode mode = kModes[index];
        const ModeObservation &observation = observations[index];
        const Validation &validation = observation.validation;
        std::printf(
            " %s{%s span=%llu A=%llu B=%llu poll=%llu/%llu hs=%c/%c}", ModeName(mode),
            validation.passed ? "PASS" : "FAIL", static_cast<unsigned long long>(validation.span),
            static_cast<unsigned long long>(validation.participant_duration[0]),
            static_cast<unsigned long long>(validation.participant_duration[1]),
            static_cast<unsigned long long>(observation.polls[0]),
            static_cast<unsigned long long>(observation.polls[1]), HandshakeCode(observation.handshakes[0]),
            HandshakeCode(observation.handshakes[1])
        );
    }
    std::printf("\n");
}

void PrintSummary(ProbeMode mode, const ModeStatistics &statistics, uint32_t runs) {
    std::printf(
        "[SUMMARY] mode=%-9s pass=%u/%u CLOCK64_span(avg=%.1Lf min=%llu max=%llu) ", ModeName(mode), statistics.passes,
        runs, statistics.span.Average(), static_cast<unsigned long long>(statistics.span.Minimum()),
        static_cast<unsigned long long>(statistics.span.maximum)
    );
    for (uint32_t participant = 0U; participant < kParticipantCount; ++participant) {
        const ClockStatistics &clock = statistics.participant[participant];
        std::printf(
            "%c(avg=%.1Lf min=%llu max=%llu count=%llu)%s", participant == 0U ? 'A' : 'B', clock.Average(),
            static_cast<unsigned long long>(clock.Minimum()), static_cast<unsigned long long>(clock.maximum),
            static_cast<unsigned long long>(clock.count), participant + 1U == kParticipantCount ? "\n" : " "
        );
    }
}

}  // namespace

int main(int argc, char **argv) {
    Options options{};
    if (!ParseOptions(argc, argv, &options)) {
        return EXIT_FAILURE;
    }
    std::vector<char> binary_data;
    if (!ReadBinary(options.kernel_path, &binary_data)) {
        return EXIT_FAILURE;
    }
    AclSession session;
    if (!session.Initialize(options.device, binary_data)) {
        return EXIT_FAILURE;
    }
    if (session.SocName().rfind("Ascend950", 0U) != 0U) {
        std::fprintf(
            stderr, "warp concurrency probe requires A5/Ascend950, but ACL reports: %s\n", session.SocName().c_str()
        );
        return EXIT_FAILURE;
    }
    std::printf(
        "[DEVICE] id=%d soc=%s state_bytes=%zu launch_threads=%u warp_size=%u reused_address=yes\n", options.device,
        session.SocName().c_str(), sizeof(ProbeState), kLaunchThreads, kWarpSize
    );

    bool passed = true;
    uint32_t printed_failures = 0U;
    std::array<ModeStatistics, kModes.size()> statistics{};
    for (uint32_t run = 0U; run < options.runs; ++run) {
        std::array<ModeObservation, kModes.size()> observations{};
        for (uint32_t index = 0U; index < kModes.size(); ++index) {
            const ProbeMode mode = kModes[index];
            const uint64_t nonce = MakeNonce(run, mode);
            ProbeState state{};
            InitializeState(&state, nonce, mode);
            if (!session.Run(&state)) {
                return EXIT_FAILURE;
            }
            ModeObservation &observation = observations[index];
            observation.validation = Validate(state, nonce, mode);
            for (uint32_t participant = 0U; participant < kParticipantCount; ++participant) {
                if (ParticipantIsActive(mode, participant)) {
                    observation.polls[participant] = state.reports[participant].poll_count;
                    observation.handshakes[participant] = state.reports[participant].handshake_status;
                } else {
                    observation.polls[participant] = 0U;
                    observation.handshakes[participant] = static_cast<uint64_t>(HandshakeStatus::kUnset);
                }
            }

            if (observation.validation.passed) {
                ModeStatistics &mode_statistics = statistics[index];
                ++mode_statistics.passes;
                mode_statistics.span.Add(observation.validation.span);
                for (uint32_t participant = 0U; participant < kParticipantCount; ++participant) {
                    if (ParticipantIsActive(mode, participant)) {
                        mode_statistics.participant[participant].Add(
                            observation.validation.participant_duration[participant]
                        );
                    }
                }
            } else {
                passed = false;
                if (printed_failures < 12U) {
                    std::fprintf(
                        stderr,
                        "[RAW-FAIL] run=%u mode=%s nonce=0x%llx reason=%s status=0x%llx "
                        "ready=0x%llx/0x%llx reports=%llu hs=%llu/%llu/%llu relation=%llu clocks=%llu..%llu\n",
                        run + 1U, ModeName(mode), static_cast<unsigned long long>(nonce),
                        observation.validation.reason.c_str(), static_cast<unsigned long long>(state.result.status),
                        static_cast<unsigned long long>(state.ready_a.value),
                        static_cast<unsigned long long>(state.ready_b.value),
                        static_cast<unsigned long long>(state.result.active_reports),
                        static_cast<unsigned long long>(state.result.handshake_successes),
                        static_cast<unsigned long long>(state.result.handshake_timeouts),
                        static_cast<unsigned long long>(state.result.handshake_not_applicable),
                        static_cast<unsigned long long>(state.result.interval_relation),
                        static_cast<unsigned long long>(state.result.first_start),
                        static_cast<unsigned long long>(state.result.last_end)
                    );
                    ++printed_failures;
                }
            }
        }
        PrintRound(run, observations);
    }

    for (uint32_t index = 0U; index < kModes.size(); ++index) {
        PrintSummary(kModes[index], statistics[index], options.runs);
        passed = passed && statistics[index].passes == options.runs;
    }
    std::printf(
        "[%s] A5 warp concurrency runs=%u modes=AOnly/BOnly/SameWarp/CrossWarp reused_address=yes\n",
        passed ? "PASS" : "FAIL", options.runs
    );
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
