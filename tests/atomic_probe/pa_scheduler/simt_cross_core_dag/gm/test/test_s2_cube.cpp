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

#include "../common/s2_cube.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

namespace {

using namespace pa_scheduler::simt_cross_core;
using namespace pa_scheduler::simt_cross_core::s2;

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] S2 CPU cube: %s\n", message);
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

struct CpuBuffers {
    std::array<float, kElementCount> input_a{};
    std::array<float, kElementCount> input_b{};
    std::array<float, kElementCount> output{};
};

struct Counters {
    std::atomic<uint64_t> builder_attempts{0U};
    std::atomic<uint64_t> builder_wins{0U};
    std::atomic<uint64_t> claim_attempts{0U};
    std::atomic<uint64_t> claim_wins{0U};
    std::atomic<uint64_t> payload_reads{0U};
    std::atomic<uint64_t> cube_executes{0U};
};

enum class ExecuteResult {
    Ineligible,
    NotBuilt,
    PayloadInvalid,
    Executed,
};

void InitializeBuffers(CpuBuffers *buffers, uint64_t nonce) {
    for (uint32_t row = 0U; row < kTileRows; ++row) {
        for (uint32_t column = 0U; column < kTileColumns; ++column) {
            const uint32_t index = row * kTileColumns + column;
            buffers->input_a[index] = ExpectedInputA(row, column);
            buffers->input_b[index] = ExpectedInputB(nonce, row, column);
            buffers->output[index] = kOutputSentinel;
        }
    }
}

std::array<uint64_t, kPayloadWords> MakePayload(CpuBuffers *buffers, uint64_t nonce) {
    const uint64_t input_a = reinterpret_cast<uint64_t>(buffers->input_a.data());
    const uint64_t input_b = reinterpret_cast<uint64_t>(buffers->input_b.data());
    const uint64_t output = reinterpret_cast<uint64_t>(buffers->output.data());
    return {
        kPayloadMagic,
        kPayloadVersion,
        nonce,
        input_a,
        input_b,
        output,
        PackTaskShape(kTaskId, kElementCount),
        ComputePayloadChecksum(nonce, input_a, input_b, output),
    };
}

bool TryBuild(
    ProbeRole role, CpuCell &cell, const std::array<uint64_t, kPayloadWords> &payload, Counters &counters,
    PausePoint *mid_payload_pause
) {
    if (role != ProbeRole::Aiv0Builder) {
        return false;
    }
    counters.builder_attempts.fetch_add(1U, std::memory_order_relaxed);
    uint64_t expected = kEmptyState;
    if (!cell.state.compare_exchange_strong(
            expected, kBuildingState, std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        return false;
    }
    counters.builder_wins.fetch_add(1U, std::memory_order_relaxed);
    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        cell.payload[word] = payload[word];
        if (mid_payload_pause != nullptr && word + 1U == kPayloadWords / 2U) {
            mid_payload_pause->Stop();
        }
    }
    expected = kBuildingState;
    return cell.state.compare_exchange_strong(
        expected, kBuiltState, std::memory_order_release, std::memory_order_acquire
    );
}

void RunCpuMatmul(const float *input_a, const float *input_b, float *output) {
    for (uint32_t row = 0U; row < kTileRows; ++row) {
        for (uint32_t column = 0U; column < kTileColumns; ++column) {
            float sum = 0.0F;
            for (uint32_t inner = 0U; inner < kTileColumns; ++inner) {
                sum += input_a[row * kTileColumns + inner] * input_b[inner * kTileColumns + column];
            }
            output[row * kTileColumns + column] = sum;
        }
    }
}

ExecuteResult TryExecute(ProbeRole role, CpuCell &cell, CpuBuffers &buffers, uint64_t nonce, Counters &counters) {
    if (role != ProbeRole::AicExecutor) {
        return ExecuteResult::Ineligible;
    }
    counters.claim_attempts.fetch_add(1U, std::memory_order_relaxed);
    uint64_t expected = kBuiltState;
    if (!cell.state.compare_exchange_strong(
            expected, kClaimedState, std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        return ExecuteResult::NotBuilt;
    }
    counters.claim_wins.fetch_add(1U, std::memory_order_relaxed);

    std::array<uint64_t, kPayloadWords> payload{};
    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        payload[word] = cell.payload[word];
        counters.payload_reads.fetch_add(1U, std::memory_order_relaxed);
    }
    const uint64_t checksum = ComputePayloadChecksum(
        payload[kPayloadNonceWord], payload[kPayloadInputAWord], payload[kPayloadInputBWord],
        payload[kPayloadOutputWord]
    );
    const uint64_t expected_input_a = reinterpret_cast<uint64_t>(buffers.input_a.data());
    const uint64_t expected_input_b = reinterpret_cast<uint64_t>(buffers.input_b.data());
    const uint64_t expected_output = reinterpret_cast<uint64_t>(buffers.output.data());
    const bool valid =
        payload[kPayloadMagicWord] == kPayloadMagic && payload[kPayloadVersionWord] == kPayloadVersion &&
        payload[kPayloadNonceWord] == nonce && payload[kPayloadInputAWord] == expected_input_a &&
        payload[kPayloadInputBWord] == expected_input_b && payload[kPayloadOutputWord] == expected_output &&
        PayloadTaskId(payload[kPayloadShapeWord]) == kTaskId &&
        PayloadElementCount(payload[kPayloadShapeWord]) == kElementCount && payload[kPayloadChecksumWord] == checksum;
    if (valid) {
        RunCpuMatmul(
            reinterpret_cast<const float *>(payload[kPayloadInputAWord]),
            reinterpret_cast<const float *>(payload[kPayloadInputBWord]),
            reinterpret_cast<float *>(payload[kPayloadOutputWord])
        );
        counters.cube_executes.fetch_add(1U, std::memory_order_relaxed);
    }

    expected = kClaimedState;
    const bool completed =
        cell.state.compare_exchange_strong(expected, kDoneState, std::memory_order_release, std::memory_order_acquire);
    if (!completed) {
        return ExecuteResult::NotBuilt;
    }
    return valid ? ExecuteResult::Executed : ExecuteResult::PayloadInvalid;
}

bool OutputMatches(const CpuBuffers &buffers, uint64_t nonce) {
    for (uint32_t row = 0U; row < kTileRows; ++row) {
        for (uint32_t column = 0U; column < kTileColumns; ++column) {
            const uint32_t index = row * kTileColumns + column;
            if (buffers.output[index] != ExpectedOutput(nonce, row, column)) {
                return false;
            }
        }
    }
    return true;
}

bool OutputIsSentinel(const CpuBuffers &buffers) {
    for (float value : buffers.output) {
        if (value != kOutputSentinel) {
            return false;
        }
    }
    return true;
}

bool RunRound(uint32_t round) {
    const uint64_t nonce = 0x5200000000000000ULL ^ static_cast<uint64_t>(round + 1U);
    auto buffers = std::make_unique<CpuBuffers>();
    InitializeBuffers(buffers.get(), nonce);
    const auto payload = MakePayload(buffers.get(), nonce);
    CpuCell cell{};
    Counters counters{};
    PausePoint pause;
    bool published = false;

    std::thread builder([&] {
        published = TryBuild(ProbeRole::Aiv0Builder, cell, payload, counters, &pause);
    });
    if (!pause.WaitUntilReached()) {
        pause.Release();
        builder.join();
        return false;
    }

    bool passed = cell.state.load(std::memory_order_acquire) == kBuildingState;
    passed = passed && TryExecute(ProbeRole::AicExecutor, cell, *buffers, nonce, counters) == ExecuteResult::NotBuilt;
    passed = passed && counters.payload_reads.load(std::memory_order_relaxed) == 0U;
    passed = passed && !TryBuild(ProbeRole::AicExecutor, cell, payload, counters, nullptr);
    passed = passed && TryExecute(ProbeRole::Aiv0Builder, cell, *buffers, nonce, counters) == ExecuteResult::Ineligible;
    passed = passed && !TryBuild(ProbeRole::Aiv1Observer, cell, payload, counters, nullptr);
    passed =
        passed && TryExecute(ProbeRole::Aiv1Observer, cell, *buffers, nonce, counters) == ExecuteResult::Ineligible;

    pause.Release();
    builder.join();
    passed = passed && published && cell.state.load(std::memory_order_acquire) == kBuiltState;
    passed = passed && TryExecute(ProbeRole::AicExecutor, cell, *buffers, nonce, counters) == ExecuteResult::Executed;
    passed = passed && cell.state.load(std::memory_order_acquire) == kDoneState && OutputMatches(*buffers, nonce);
    passed = passed && counters.builder_attempts.load(std::memory_order_relaxed) == 1U;
    passed = passed && counters.builder_wins.load(std::memory_order_relaxed) == 1U;
    passed = passed && counters.claim_attempts.load(std::memory_order_relaxed) == 2U;
    passed = passed && counters.claim_wins.load(std::memory_order_relaxed) == 1U;
    passed = passed && counters.payload_reads.load(std::memory_order_relaxed) == kPayloadWords;
    passed = passed && counters.cube_executes.load(std::memory_order_relaxed) == 1U;
    return passed;
}

bool TestInvalidPayloadFailsClosed(bool rewrite_output_address) {
    const uint64_t nonce = 0x52000000BAD00001ULL;
    auto buffers = std::make_unique<CpuBuffers>();
    InitializeBuffers(buffers.get(), nonce);
    CpuCell cell{};
    Counters counters{};
    auto payload = MakePayload(buffers.get(), nonce);
    if (rewrite_output_address) {
        payload[kPayloadOutputWord] += kCacheLineBytes;
        payload[kPayloadChecksumWord] = ComputePayloadChecksum(
            payload[kPayloadNonceWord], payload[kPayloadInputAWord], payload[kPayloadInputBWord],
            payload[kPayloadOutputWord]
        );
    } else {
        payload[kPayloadChecksumWord] ^= 1U;
    }
    cell.payload = payload;
    cell.state.store(kBuiltState, std::memory_order_release);

    const ExecuteResult result = TryExecute(ProbeRole::AicExecutor, cell, *buffers, nonce, counters);
    return result == ExecuteResult::PayloadInvalid && cell.state.load(std::memory_order_acquire) == kDoneState &&
           counters.claim_wins.load(std::memory_order_relaxed) == 1U &&
           counters.cube_executes.load(std::memory_order_relaxed) == 0U && OutputIsSentinel(*buffers);
}

bool ParseRounds(int argc, char **argv, uint32_t *rounds) {
    *rounds = 16U;
    if (argc == 1) {
        return true;
    }
    if (argc != 3 || std::strcmp(argv[1], "--rounds") != 0) {
        std::fprintf(stderr, "usage: %s [--rounds N]\n", argv[0]);
        return false;
    }
    char *end = nullptr;
    const unsigned long value = std::strtoul(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || value == 0U || value > 1000U) {
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
    Check(TestInvalidPayloadFailsClosed(false), "a bad checksum must not execute Cube work");
    Check(TestInvalidPayloadFailsClosed(true), "a checksummed address substitution must not execute Cube work");
    for (uint32_t round = 0U; round < rounds; ++round) {
        if (!RunRound(round)) {
            std::fprintf(stderr, "[FAIL] S2 CPU cube round=%u\n", round);
            ++g_failures;
            break;
        }
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "[FAIL] S2 CPU cube failures=%d\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf(
        "[PASS] S2 CPU cube rounds=%u: AIV0 build-only, AIC execute-only, half-packet hidden, golden exact\n", rounds
    );
    return EXIT_SUCCESS;
}
