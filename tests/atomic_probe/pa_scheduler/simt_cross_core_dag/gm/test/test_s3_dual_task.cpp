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

#include "../common/s3_dual_task.h"

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
using namespace pa_scheduler::simt_cross_core::s3;

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] S3 CPU dual-task: %s\n", message);
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

struct alignas(kCacheLineBytes) CpuTaskCell {
    std::atomic<uint64_t> state{kEmptyState};
    std::array<uint8_t, kCacheLineBytes - sizeof(std::atomic<uint64_t>)> control_padding{};
    alignas(kCacheLineBytes) std::array<uint64_t, kPayloadWords> payload{};
};

struct alignas(kCacheLineBytes) CpuDrain {
    std::atomic<uint64_t> builder_finished{0U};
    std::atomic<uint64_t> done_count{0U};
};

struct CpuBuffers {
    std::array<float, kElementCount> vector_input_a{};
    std::array<float, kElementCount> vector_input_b{};
    std::array<float, kElementCount> vector_output{};
    std::array<float, kElementCount> cube_input_a{};
    std::array<float, kElementCount> cube_input_b{};
    std::array<float, kElementCount> cube_output{};
};

struct Counters {
    std::atomic<uint64_t> builder_attempts{0U};
    std::atomic<uint64_t> builder_wins{0U};
    std::atomic<uint64_t> claim_attempts{0U};
    std::atomic<uint64_t> claim_wins{0U};
    std::atomic<uint64_t> payload_reads{0U};
    std::atomic<uint64_t> vector_executes{0U};
    std::atomic<uint64_t> cube_executes{0U};
};

enum class ExecuteResult {
    Ineligible,
    NotBuilt,
    PayloadInvalid,
    Executed,
};

constexpr uint64_t BuildingState(uint32_t task_index) {
    return task_index == kVectorTaskIndex ? kVectorBuildingState : kCubeBuildingState;
}

constexpr uint64_t BuiltState(uint32_t task_index) {
    return task_index == kVectorTaskIndex ? kVectorBuiltState : kCubeBuiltState;
}

constexpr uint64_t ClaimedState(uint32_t task_index) {
    return task_index == kVectorTaskIndex ? kVectorClaimedState : kCubeClaimedState;
}

constexpr uint64_t DoneState(uint32_t task_index) {
    return task_index == kVectorTaskIndex ? kVectorDoneState : kCubeDoneState;
}

bool RoleCanExecute(ProbeRole role, uint32_t task_index) {
    return (task_index == kVectorTaskIndex && role == ProbeRole::Aiv1VectorExecutor) ||
           (task_index == kCubeTaskIndex && role == ProbeRole::AicCubeExecutor);
}

void InitializeBuffers(CpuBuffers *buffers, uint64_t nonce) {
    for (uint32_t row = 0U; row < kTileRows; ++row) {
        for (uint32_t column = 0U; column < kTileColumns; ++column) {
            const uint32_t index = row * kTileColumns + column;
            buffers->vector_input_a[index] = ExpectedVectorInputA(nonce, index);
            buffers->vector_input_b[index] = ExpectedVectorInputB(nonce, index);
            buffers->vector_output[index] = kOutputSentinel;
            buffers->cube_input_a[index] = ExpectedCubeInputA(row, column);
            buffers->cube_input_b[index] = ExpectedCubeInputB(nonce, row, column);
            buffers->cube_output[index] = kOutputSentinel;
        }
    }
}

std::array<uint64_t, kPayloadWords> MakePayload(uint32_t task_index, CpuBuffers *buffers, uint64_t nonce) {
    const bool vector = task_index == kVectorTaskIndex;
    const uint64_t input_a =
        reinterpret_cast<uint64_t>(vector ? buffers->vector_input_a.data() : buffers->cube_input_a.data());
    const uint64_t input_b =
        reinterpret_cast<uint64_t>(vector ? buffers->vector_input_b.data() : buffers->cube_input_b.data());
    const uint64_t output =
        reinterpret_cast<uint64_t>(vector ? buffers->vector_output.data() : buffers->cube_output.data());
    return {
        TaskPayloadMagic(task_index),
        kPayloadVersion,
        nonce,
        input_a,
        input_b,
        output,
        PackTaskShape(TaskId(task_index), kElementCount),
        ComputePayloadChecksum(task_index, nonce, input_a, input_b, output),
    };
}

bool TryBuild(
    ProbeRole role, uint32_t task_index, CpuTaskCell &cell, const std::array<uint64_t, kPayloadWords> &payload,
    Counters &counters, PausePoint *mid_payload_pause
) {
    if (role != ProbeRole::Aiv0Builder || task_index >= kTaskCount) {
        return false;
    }
    counters.builder_attempts.fetch_add(1U, std::memory_order_relaxed);
    uint64_t expected = kEmptyState;
    if (!cell.state.compare_exchange_strong(
            expected, BuildingState(task_index), std::memory_order_acq_rel, std::memory_order_acquire
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
    expected = BuildingState(task_index);
    return cell.state.compare_exchange_strong(
        expected, BuiltState(task_index), std::memory_order_release, std::memory_order_acquire
    );
}

void RunCpuVector(const float *input_a, const float *input_b, float *output) {
    for (uint32_t index = 0U; index < kElementCount; ++index) {
        output[index] = input_a[index] + input_b[index];
    }
}

void RunCpuCube(const float *input_a, const float *input_b, float *output) {
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

ExecuteResult TryExecute(
    ProbeRole role, uint32_t task_index, CpuTaskCell &cell, CpuBuffers &buffers, uint64_t nonce, Counters &counters,
    CpuDrain &drain
) {
    if (task_index >= kTaskCount || !RoleCanExecute(role, task_index)) {
        return ExecuteResult::Ineligible;
    }
    counters.claim_attempts.fetch_add(1U, std::memory_order_relaxed);
    uint64_t expected = BuiltState(task_index);
    if (!cell.state.compare_exchange_strong(
            expected, ClaimedState(task_index), std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        return ExecuteResult::NotBuilt;
    }
    counters.claim_wins.fetch_add(1U, std::memory_order_relaxed);

    std::array<uint64_t, kPayloadWords> payload{};
    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        payload[word] = cell.payload[word];
        counters.payload_reads.fetch_add(1U, std::memory_order_relaxed);
    }
    const auto expected_payload = MakePayload(task_index, &buffers, nonce);
    const uint64_t checksum = ComputePayloadChecksum(
        task_index, payload[kPayloadNonceWord], payload[kPayloadInputAWord], payload[kPayloadInputBWord],
        payload[kPayloadOutputWord]
    );
    bool valid = true;
    for (uint32_t word = 0U; word < kPayloadWords - 1U; ++word) {
        valid = valid && payload[word] == expected_payload[word];
    }
    valid = valid && payload[kPayloadChecksumWord] == checksum && checksum == expected_payload[kPayloadChecksumWord];
    if (valid && task_index == kVectorTaskIndex) {
        RunCpuVector(
            reinterpret_cast<const float *>(payload[kPayloadInputAWord]),
            reinterpret_cast<const float *>(payload[kPayloadInputBWord]),
            reinterpret_cast<float *>(payload[kPayloadOutputWord])
        );
        counters.vector_executes.fetch_add(1U, std::memory_order_relaxed);
    } else if (valid) {
        RunCpuCube(
            reinterpret_cast<const float *>(payload[kPayloadInputAWord]),
            reinterpret_cast<const float *>(payload[kPayloadInputBWord]),
            reinterpret_cast<float *>(payload[kPayloadOutputWord])
        );
        counters.cube_executes.fetch_add(1U, std::memory_order_relaxed);
    }

    expected = ClaimedState(task_index);
    if (!cell.state.compare_exchange_strong(
            expected, DoneState(task_index), std::memory_order_release, std::memory_order_acquire
        )) {
        return ExecuteResult::NotBuilt;
    }
    drain.done_count.fetch_add(1U, std::memory_order_release);
    return valid ? ExecuteResult::Executed : ExecuteResult::PayloadInvalid;
}

bool DrainReady(const CpuDrain &drain) {
    return drain.builder_finished.load(std::memory_order_acquire) == 1U &&
           drain.done_count.load(std::memory_order_acquire) == kTaskCount;
}

bool BuffersValid(const CpuBuffers &buffers, uint64_t nonce) {
    for (uint32_t row = 0U; row < kTileRows; ++row) {
        for (uint32_t column = 0U; column < kTileColumns; ++column) {
            const uint32_t index = row * kTileColumns + column;
            if (buffers.vector_input_a[index] != ExpectedVectorInputA(nonce, index) ||
                buffers.vector_input_b[index] != ExpectedVectorInputB(nonce, index) ||
                buffers.vector_output[index] != ExpectedVectorOutput(nonce, index) ||
                buffers.cube_input_a[index] != ExpectedCubeInputA(row, column) ||
                buffers.cube_input_b[index] != ExpectedCubeInputB(nonce, row, column) ||
                buffers.cube_output[index] != ExpectedCubeOutput(nonce, row, column)) {
                return false;
            }
        }
    }
    return true;
}

bool OutputIsSentinel(const std::array<float, kElementCount> &output) {
    for (float value : output) {
        if (value != kOutputSentinel) {
            return false;
        }
    }
    return true;
}

bool RunRound(uint32_t round) {
    const uint64_t nonce = 0x5300000000000000ULL ^ static_cast<uint64_t>(round + 1U);
    auto buffers = std::make_unique<CpuBuffers>();
    InitializeBuffers(buffers.get(), nonce);
    std::array<CpuTaskCell, kTaskCount> cells{};
    CpuDrain drain{};
    Counters counters{};
    PausePoint pause;
    bool vector_published = false;

    const auto vector_payload = MakePayload(kVectorTaskIndex, buffers.get(), nonce);
    const auto cube_payload = MakePayload(kCubeTaskIndex, buffers.get(), nonce);
    std::thread vector_builder([&] {
        vector_published = TryBuild(
            ProbeRole::Aiv0Builder, kVectorTaskIndex, cells[kVectorTaskIndex], vector_payload, counters, &pause
        );
    });
    if (!pause.WaitUntilReached()) {
        pause.Release();
        vector_builder.join();
        return false;
    }

    bool passed = cells[kVectorTaskIndex].state.load(std::memory_order_acquire) == kVectorBuildingState;
    passed = passed &&
             TryBuild(ProbeRole::Aiv0Builder, kCubeTaskIndex, cells[kCubeTaskIndex], cube_payload, counters, nullptr);
    passed = passed && cells[kCubeTaskIndex].state.load(std::memory_order_acquire) == kCubeBuiltState;
    passed = passed &&
             TryExecute(
                 ProbeRole::Aiv1VectorExecutor, kCubeTaskIndex, cells[kCubeTaskIndex], *buffers, nonce, counters, drain
             ) == ExecuteResult::Ineligible;
    passed = passed &&
             TryExecute(
                 ProbeRole::AicCubeExecutor, kVectorTaskIndex, cells[kVectorTaskIndex], *buffers, nonce, counters, drain
             ) == ExecuteResult::Ineligible;
    passed = passed && TryExecute(
                           ProbeRole::Aiv1VectorExecutor, kVectorTaskIndex, cells[kVectorTaskIndex], *buffers, nonce,
                           counters, drain
                       ) == ExecuteResult::NotBuilt;
    passed = passed &&
             TryExecute(
                 ProbeRole::AicCubeExecutor, kCubeTaskIndex, cells[kCubeTaskIndex], *buffers, nonce, counters, drain
             ) == ExecuteResult::Executed;
    passed = passed && drain.done_count.load(std::memory_order_acquire) == 1U && !DrainReady(drain);

    pause.Release();
    vector_builder.join();
    passed = passed && vector_published;
    drain.builder_finished.store(1U, std::memory_order_release);
    passed = passed && !DrainReady(drain);
    passed = passed && TryExecute(
                           ProbeRole::Aiv1VectorExecutor, kVectorTaskIndex, cells[kVectorTaskIndex], *buffers, nonce,
                           counters, drain
                       ) == ExecuteResult::Executed;
    passed = passed && DrainReady(drain) && BuffersValid(*buffers, nonce);
    passed = passed && cells[kVectorTaskIndex].state.load(std::memory_order_acquire) == kVectorDoneState;
    passed = passed && cells[kCubeTaskIndex].state.load(std::memory_order_acquire) == kCubeDoneState;
    passed = passed && counters.builder_attempts.load(std::memory_order_relaxed) == kTaskCount;
    passed = passed && counters.builder_wins.load(std::memory_order_relaxed) == kTaskCount;
    passed = passed && counters.claim_attempts.load(std::memory_order_relaxed) == 3U;
    passed = passed && counters.claim_wins.load(std::memory_order_relaxed) == kTaskCount;
    passed = passed && counters.payload_reads.load(std::memory_order_relaxed) == kTaskCount * kPayloadWords;
    passed = passed && counters.vector_executes.load(std::memory_order_relaxed) == 1U;
    passed = passed && counters.cube_executes.load(std::memory_order_relaxed) == 1U;
    return passed;
}

bool TestAddressSubstitutionFailsClosed(uint32_t task_index) {
    const uint64_t nonce = 0x53000000BAD00000ULL ^ task_index;
    auto buffers = std::make_unique<CpuBuffers>();
    InitializeBuffers(buffers.get(), nonce);
    CpuTaskCell cell{};
    CpuDrain drain{};
    Counters counters{};
    auto payload = MakePayload(task_index, buffers.get(), nonce);
    payload[kPayloadOutputWord] += kCacheLineBytes;
    payload[kPayloadChecksumWord] = ComputePayloadChecksum(
        task_index, payload[kPayloadNonceWord], payload[kPayloadInputAWord], payload[kPayloadInputBWord],
        payload[kPayloadOutputWord]
    );
    cell.payload = payload;
    cell.state.store(BuiltState(task_index), std::memory_order_release);
    const ProbeRole role = task_index == kVectorTaskIndex ? ProbeRole::Aiv1VectorExecutor : ProbeRole::AicCubeExecutor;
    const ExecuteResult result = TryExecute(role, task_index, cell, *buffers, nonce, counters, drain);
    const auto &output = task_index == kVectorTaskIndex ? buffers->vector_output : buffers->cube_output;
    return result == ExecuteResult::PayloadInvalid &&
           cell.state.load(std::memory_order_acquire) == DoneState(task_index) &&
           drain.done_count.load(std::memory_order_acquire) == 1U &&
           counters.vector_executes.load(std::memory_order_relaxed) == 0U &&
           counters.cube_executes.load(std::memory_order_relaxed) == 0U && OutputIsSentinel(output);
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
    Check(TestAddressSubstitutionFailsClosed(kVectorTaskIndex), "Vector output address substitution must fail");
    Check(TestAddressSubstitutionFailsClosed(kCubeTaskIndex), "Cube output address substitution must fail");
    for (uint32_t round = 0U; round < rounds; ++round) {
        if (!RunRound(round)) {
            std::fprintf(stderr, "[FAIL] S3 CPU dual-task round=%u\n", round);
            ++g_failures;
            break;
        }
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "[FAIL] S3 CPU dual-task failures=%d\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf("[PASS] S3 CPU dual-task rounds=%u: two SIMT builds, engine routes, exact drain and goldens\n", rounds);
    return EXIT_SUCCESS;
}
