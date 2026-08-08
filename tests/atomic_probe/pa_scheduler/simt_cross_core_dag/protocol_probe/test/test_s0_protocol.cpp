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

#include "../common/s0_probe.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using namespace pa_scheduler::simt_cross_core;
using namespace pa_scheduler::simt_cross_core::s0;

constexpr uint32_t kClaimActors = 16U;
constexpr uint32_t kFatalActors = 8U;

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] S0 CPU protocol: %s\n", message);
    ++g_failures;
}

class PausePoint {
public:
    void Stop() {
        std::unique_lock<std::mutex> lock(mutex_);
        reached_ = true;
        condition_.notify_all();
        condition_.wait(lock, [this] {
            return released_;
        });
    }

    bool WaitUntilReached() {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, std::chrono::seconds(5), [this] {
            return reached_;
        });
    }

    void Release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool reached_ = false;
    bool released_ = false;
};

struct alignas(kCacheLineBytes) CpuCell {
    std::atomic<uint64_t> state{kEmptyState};
    std::array<uint8_t, kCacheLineBytes - sizeof(std::atomic<uint64_t>)> control_padding{};
    alignas(kCacheLineBytes) std::array<uint64_t, kPayloadWords> payload{};
};

static_assert(offsetof(CpuCell, payload) == kCacheLineBytes, "CPU payload must not share the atomic control line");

struct Counters {
    std::atomic<uint64_t> payload_stores{0U};
    std::atomic<uint64_t> payload_reads{0U};
    std::atomic<uint64_t> builder_wins{0U};
    std::atomic<uint64_t> claim_wins{0U};
};

uint64_t ExpectedPayloadChecksum(uint64_t nonce) {
    uint64_t checksum = 0x243F6A8885A308D3ULL;
    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        checksum = FoldChecksum(checksum, ExpectedPayloadWord(nonce, word));
    }
    return checksum;
}

bool TryBuild(CpuCell &cell, uint64_t nonce, Counters &counters, PausePoint *mid_pack_pause) {
    uint64_t expected = kEmptyState;
    if (!cell.state.compare_exchange_strong(
            expected, kBuildingState, std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        return false;
    }
    counters.builder_wins.fetch_add(1U, std::memory_order_relaxed);

    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        cell.payload[word] = ExpectedPayloadWord(nonce, word);
        counters.payload_stores.fetch_add(1U, std::memory_order_relaxed);
        if (mid_pack_pause != nullptr && word + 1U == kPayloadWords / 2U) {
            mid_pack_pause->Stop();
        }
    }

    expected = kBuildingState;
    return cell.state.compare_exchange_strong(
        expected, kBuiltState, std::memory_order_release, std::memory_order_acquire
    );
}

bool TryClaimAndComplete(CpuCell &cell, Counters &counters, uint64_t *observed_checksum) {
    uint64_t expected = kBuiltState;
    if (!cell.state.compare_exchange_strong(
            expected, kClaimedState, std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        return false;
    }
    counters.claim_wins.fetch_add(1U, std::memory_order_relaxed);

    uint64_t checksum = 0x243F6A8885A308D3ULL;
    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        checksum = FoldChecksum(checksum, cell.payload[word]);
        counters.payload_reads.fetch_add(1U, std::memory_order_relaxed);
    }
    *observed_checksum = checksum;

    expected = kClaimedState;
    return cell.state.compare_exchange_strong(
        expected, kDoneState, std::memory_order_release, std::memory_order_acquire
    );
}

bool WaitForDoneBounded(const CpuCell &cell, uint32_t polls) {
    for (uint32_t poll = 0U; poll < polls; ++poll) {
        if (cell.state.load(std::memory_order_acquire) == kDoneState) {
            return true;
        }
    }
    return false;
}

bool RunProtocolRound(uint32_t round) {
    const uint64_t nonce = 0x600D000000000000ULL ^ static_cast<uint64_t>(round + 1U);
    CpuCell cell{};
    Counters counters{};
    PausePoint pause;
    bool builder_published = false;

    std::thread builder([&] {
        builder_published = TryBuild(cell, nonce, counters, &pause);
    });

    const bool reached = pause.WaitUntilReached();
    if (!reached) {
        pause.Release();
        builder.join();
        return false;
    }

    bool passed = true;
    passed = passed && cell.state.load(std::memory_order_acquire) == kBuildingState;
    passed = passed && counters.payload_stores.load(std::memory_order_relaxed) == kPayloadWords / 2U;

    uint64_t premature_checksum = 0U;
    passed = passed && !TryClaimAndComplete(cell, counters, &premature_checksum);
    passed = passed && counters.payload_reads.load(std::memory_order_relaxed) == 0U;

    const auto payload_before_duplicate = cell.payload;
    passed = passed && !TryBuild(cell, nonce ^ 0xFFFFU, counters, nullptr);
    passed = passed && cell.payload == payload_before_duplicate;
    passed = passed && counters.builder_wins.load(std::memory_order_relaxed) == 1U;

    pause.Release();
    builder.join();
    passed = passed && builder_published;
    passed = passed && cell.state.load(std::memory_order_acquire) == kBuiltState;

    std::atomic<uint32_t> ready{0U};
    std::atomic<bool> start{false};
    std::array<uint64_t, kClaimActors> checksums{};
    std::array<bool, kClaimActors> completed{};
    std::vector<std::thread> claimers;
    claimers.reserve(kClaimActors);
    for (uint32_t actor = 0U; actor < kClaimActors; ++actor) {
        claimers.emplace_back([&, actor] {
            ready.fetch_add(1U, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            completed[actor] = TryClaimAndComplete(cell, counters, &checksums[actor]);
        });
    }
    while (ready.load(std::memory_order_acquire) != kClaimActors) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (std::thread &claimer : claimers) {
        claimer.join();
    }

    uint32_t completed_count = 0U;
    uint64_t winner_checksum = 0U;
    for (uint32_t actor = 0U; actor < kClaimActors; ++actor) {
        if (completed[actor]) {
            ++completed_count;
            winner_checksum = checksums[actor];
        }
    }
    passed = passed && completed_count == 1U;
    passed = passed && counters.claim_wins.load(std::memory_order_relaxed) == 1U;
    passed = passed && counters.payload_reads.load(std::memory_order_relaxed) == kPayloadWords;
    passed = passed && winner_checksum == ExpectedPayloadChecksum(nonce);
    passed = passed && WaitForDoneBounded(cell, 1U);

    const DecodedExecState decoded = DecodeExecState(cell.state.load(std::memory_order_acquire));
    passed = passed && decoded.valid && decoded.phase == ExecPhase::Done && decoded.build_owner == kBuilderOwner &&
             decoded.execute_owner == kExecutorOwner && decoded.engine_class == ExecEngineClass::Aiv &&
             decoded.payload_lines == 1U && decoded.task_id == kTaskId;
    return passed;
}

bool TestBoundedTimeout() {
    CpuCell cell{};
    cell.state.store(kBuildingState, std::memory_order_release);
    return !WaitForDoneBounded(cell, 64U) && cell.state.load(std::memory_order_acquire) == kBuildingState;
}

bool TestPackedStateValidation() {
    const auto valid_empty = DecodeExecState(kEmptyState);
    const auto valid_building = DecodeExecState(kBuildingState);
    const auto valid_built = DecodeExecState(kBuiltState);
    const auto valid_claimed = DecodeExecState(kClaimedState);
    const auto valid_done = DecodeExecState(kDoneState);

    const uint64_t building_with_published_metadata =
        EncodeExecState(ExecPhase::Building, kBuilderOwner, kUnboundOwner, ExecEngineClass::Aiv, 1U, kTaskId);
    const uint64_t built_without_engine =
        EncodeExecState(ExecPhase::Built, kBuilderOwner, kUnboundOwner, ExecEngineClass::None, 1U, kTaskId);
    const uint64_t built_with_bound_executor =
        EncodeExecState(ExecPhase::Built, kBuilderOwner, kExecutorOwner, ExecEngineClass::Aiv, 1U, kTaskId);
    const uint64_t done_without_executor =
        EncodeExecState(ExecPhase::Done, kBuilderOwner, kUnboundOwner, ExecEngineClass::Aiv, 1U, kTaskId);
    const uint64_t oversized_payload = EncodeExecState(
        ExecPhase::Claimed, kBuilderOwner, kExecutorOwner, ExecEngineClass::Aiv, kMaxPayloadLines + 1U, kTaskId
    );

    return valid_empty.valid && valid_building.valid && valid_built.valid && valid_claimed.valid && valid_done.valid &&
           !DecodeExecState(building_with_published_metadata).valid && !DecodeExecState(built_without_engine).valid &&
           !DecodeExecState(built_with_bound_executor).valid && !DecodeExecState(done_without_executor).valid &&
           !DecodeExecState(oversized_payload).valid && !DecodeExecState(kDoneState | (1ULL << 61U)).valid;
}

bool TestFirstFatalWins() {
    std::atomic<uint64_t> fatal{0U};
    std::atomic<uint32_t> ready{0U};
    std::atomic<bool> start{false};
    std::atomic<uint32_t> winners{0U};
    std::vector<std::thread> reporters;
    reporters.reserve(kFatalActors);

    for (uint32_t actor = 0U; actor < kFatalActors; ++actor) {
        reporters.emplace_back([&, actor] {
            ready.fetch_add(1U, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            uint64_t expected = 0U;
            const uint64_t report = EncodeExecFatal(ExecFatalReason::Timeout, 40U + actor, kTaskId + actor);
            if (fatal.compare_exchange_strong(expected, report, std::memory_order_acq_rel, std::memory_order_acquire)) {
                winners.fetch_add(1U, std::memory_order_relaxed);
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != kFatalActors) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (std::thread &reporter : reporters) {
        reporter.join();
    }

    const DecodedExecFatal decoded = DecodeExecFatal(fatal.load(std::memory_order_acquire));
    return winners.load(std::memory_order_relaxed) == 1U && decoded.valid &&
           decoded.reason == ExecFatalReason::Timeout && decoded.reporter_owner >= 40U &&
           decoded.reporter_owner < 40U + kFatalActors && decoded.task_id == kTaskId + decoded.reporter_owner - 40U;
}

bool ParseRounds(int argc, char **argv, uint32_t *rounds) {
    *rounds = 128U;
    if (argc == 1) {
        return true;
    }
    if (argc != 3 || std::strcmp(argv[1], "--rounds") != 0) {
        std::fprintf(stderr, "usage: %s [--rounds N]\n", argv[0]);
        return false;
    }
    char *end = nullptr;
    const unsigned long value = std::strtoul(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || value == 0U || value > 100000U) {
        std::fprintf(stderr, "invalid --rounds value: %s\n", argv[2]);
        return false;
    }
    *rounds = static_cast<uint32_t>(value);
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    uint32_t rounds = 0U;
    if (!ParseRounds(argc, argv, &rounds)) {
        return EXIT_FAILURE;
    }

    Check(TestBoundedTimeout(), "bounded polling must report timeout without changing state");
    Check(TestPackedStateValidation(), "packed states must reject malformed phase-specific fields");
    Check(TestFirstFatalWins(), "fatal publication must preserve exactly one valid first error");
    for (uint32_t round = 0U; round < rounds; ++round) {
        if (!RunProtocolRound(round)) {
            std::fprintf(stderr, "[FAIL] S0 CPU protocol round=%u\n", round);
            ++g_failures;
            break;
        }
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "[FAIL] S0 CPU protocol failures=%d\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf(
        "[PASS] S0 CPU protocol rounds=%u: half-packet hidden, unique claim, "
        "first fatal and bounded timeout\n",
        rounds
    );
    return EXIT_SUCCESS;
}
