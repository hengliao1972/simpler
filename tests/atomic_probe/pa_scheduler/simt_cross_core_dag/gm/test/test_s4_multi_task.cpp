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

#include "../common/s4_multi_task.h"

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
using namespace pa_scheduler::simt_cross_core::s4;

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

struct alignas(kCacheLineBytes) CpuTask {
    std::atomic<uint64_t> state{0U};
    std::array<uint8_t, kCacheLineBytes - sizeof(std::atomic<uint64_t>)> padding{};
    alignas(kCacheLineBytes) std::array<uint64_t, kPayloadWords> payload{};
};

static_assert(offsetof(CpuTask, payload) == kCacheLineBytes, "S4 CPU payload must not share the control line");

struct CpuData {
    std::array<std::array<float, kElementCount>, kVectorTaskCount> vector_input_a{};
    std::array<std::array<float, kElementCount>, kVectorTaskCount> vector_input_b{};
    std::array<std::array<float, kElementCount>, kVectorTaskCount> vector_output{};
    std::array<std::array<float, kElementCount>, kCubeTaskCount> cube_input_a{};
    std::array<std::array<float, kElementCount>, kCubeTaskCount> cube_input_b{};
    std::array<std::array<float, kElementCount>, kCubeTaskCount> cube_output{};
};

uint64_t Address(const float *pointer) { return reinterpret_cast<uint64_t>(pointer); }

void InitializeData(CpuData *data, uint64_t nonce) {
    for (uint32_t ordinal = 0U; ordinal < kVectorTaskCount; ++ordinal) {
        for (uint32_t index = 0U; index < kElementCount; ++index) {
            data->vector_input_a[ordinal][index] = ExpectedVectorInputA(nonce, ordinal, index);
            data->vector_input_b[ordinal][index] = ExpectedVectorInputB(nonce, ordinal, index);
            data->vector_output[ordinal][index] = kOutputSentinel;
        }
    }
    for (uint32_t ordinal = 0U; ordinal < kCubeTaskCount; ++ordinal) {
        for (uint32_t row = 0U; row < kTileRows; ++row) {
            for (uint32_t column = 0U; column < kTileColumns; ++column) {
                const uint32_t index = row * kTileColumns + column;
                data->cube_input_a[ordinal][index] = ExpectedCubeInputA(ordinal, row, column);
                data->cube_input_b[ordinal][index] = ExpectedCubeInputB(nonce, ordinal, row, column);
                data->cube_output[ordinal][index] = kOutputSentinel;
            }
        }
    }
}

void TaskAddresses(const CpuData &data, uint32_t task_index, uint64_t *input_a, uint64_t *input_b, uint64_t *output) {
    const uint32_t ordinal = TaskOrdinal(task_index);
    if (TaskIsVector(task_index)) {
        *input_a = Address(data.vector_input_a[ordinal].data());
        *input_b = Address(data.vector_input_b[ordinal].data());
        *output = Address(data.vector_output[ordinal].data());
    } else {
        *input_a = Address(data.cube_input_a[ordinal].data());
        *input_b = Address(data.cube_input_b[ordinal].data());
        *output = Address(data.cube_output[ordinal].data());
    }
}

bool BuildTask(CpuTask *task, const CpuData &data, uint64_t nonce, uint32_t task_index, PausePoint *pause) {
    uint64_t expected = 0U;
    if (!task->state.compare_exchange_strong(
            expected, BuildingState(task_index), std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        return false;
    }
    uint64_t input_a = 0U;
    uint64_t input_b = 0U;
    uint64_t output = 0U;
    TaskAddresses(data, task_index, &input_a, &input_b, &output);
    const std::array<uint64_t, kPayloadWords> payload{
        TaskPayloadMagic(task_index),
        kPayloadVersion,
        nonce,
        input_a,
        input_b,
        output,
        PackTaskShape(TaskId(task_index), kElementCount),
        ComputePayloadChecksum(task_index, nonce, input_a, input_b, output),
    };
    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        task->payload[word] = payload[word];
        if (pause != nullptr && word + 1U == kPayloadWords / 2U) {
            pause->Stop();
        }
    }
    expected = BuildingState(task_index);
    return task->state.compare_exchange_strong(
        expected, BuiltState(task_index), std::memory_order_release, std::memory_order_acquire
    );
}

bool PayloadValid(const CpuTask &task, const CpuData &data, uint64_t nonce, uint32_t task_index) {
    uint64_t input_a = 0U;
    uint64_t input_b = 0U;
    uint64_t output = 0U;
    TaskAddresses(data, task_index, &input_a, &input_b, &output);
    return task.payload[kPayloadMagicWord] == TaskPayloadMagic(task_index) &&
           task.payload[kPayloadVersionWord] == kPayloadVersion && task.payload[kPayloadNonceWord] == nonce &&
           task.payload[kPayloadInputAWord] == input_a && task.payload[kPayloadInputBWord] == input_b &&
           task.payload[kPayloadOutputWord] == output &&
           task.payload[kPayloadShapeWord] == PackTaskShape(TaskId(task_index), kElementCount) &&
           task.payload[kPayloadChecksumWord] == ComputePayloadChecksum(task_index, nonce, input_a, input_b, output);
}

void RunWorkload(CpuData *data, uint64_t nonce, uint32_t task_index) {
    const uint32_t ordinal = TaskOrdinal(task_index);
    if (TaskIsVector(task_index)) {
        for (uint32_t index = 0U; index < kElementCount; ++index) {
            data->vector_output[ordinal][index] =
                data->vector_input_a[ordinal][index] + data->vector_input_b[ordinal][index];
        }
        return;
    }
    for (uint32_t row = 0U; row < kTileRows; ++row) {
        for (uint32_t column = 0U; column < kTileColumns; ++column) {
            float value = 0.0F;
            for (uint32_t inner = 0U; inner < kTileColumns; ++inner) {
                value += data->cube_input_a[ordinal][row * kTileColumns + inner] *
                         data->cube_input_b[ordinal][inner * kTileColumns + column];
            }
            data->cube_output[ordinal][row * kTileColumns + column] = value;
        }
    }
    (void)nonce;
}

class CpuExecutor {
public:
    CpuExecutor(bool vector, std::atomic<uint64_t> *engine_done, std::atomic<uint64_t> *done_count) :
        vector_(vector),
        engine_done_(engine_done),
        done_count_(done_count) {}

    bool TryClaim(CpuTask *task, uint32_t task_index) {
        if (busy_) {
            ++busy_blocked_;
            return false;
        }
        if (TaskIsVector(task_index) != vector_) {
            return false;
        }
        ++claim_attempts_;
        uint64_t expected = BuiltState(task_index);
        if (!task->state.compare_exchange_strong(
                expected, ClaimedState(task_index), std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            return false;
        }
        ++claim_wins_;
        busy_ = true;
        active_task_ = task_index;
        max_busy_ = 1U;
        return true;
    }

    bool ExecuteAndComplete(CpuTask *task, CpuData *data, uint64_t nonce, uint32_t task_index) {
        if (!busy_ || active_task_ != task_index || !PayloadValid(*task, *data, nonce, task_index)) {
            return false;
        }
        RunWorkload(data, nonce, task_index);
        uint64_t expected = ClaimedState(task_index);
        if (!task->state.compare_exchange_strong(
                expected, DoneState(task_index), std::memory_order_release, std::memory_order_acquire
            )) {
            return false;
        }
        engine_done_->fetch_add(1U, std::memory_order_acq_rel);
        done_count_->fetch_add(1U, std::memory_order_acq_rel);
        ++executed_;
        busy_ = false;
        active_task_ = kTaskCount;
        return true;
    }

    uint32_t ClaimAttempts() const { return claim_attempts_; }
    uint32_t ClaimWins() const { return claim_wins_; }
    uint32_t Executed() const { return executed_; }
    uint32_t MaxBusy() const { return max_busy_; }
    uint32_t BusyBlocked() const { return busy_blocked_; }
    bool Busy() const { return busy_; }

private:
    bool vector_;
    std::atomic<uint64_t> *engine_done_;
    std::atomic<uint64_t> *done_count_;
    bool busy_ = false;
    uint32_t active_task_ = kTaskCount;
    uint32_t claim_attempts_ = 0U;
    uint32_t claim_wins_ = 0U;
    uint32_t executed_ = 0U;
    uint32_t max_busy_ = 0U;
    uint32_t busy_blocked_ = 0U;
};

bool OutputsValid(const CpuData &data, uint64_t nonce) {
    for (uint32_t ordinal = 0U; ordinal < kVectorTaskCount; ++ordinal) {
        for (uint32_t index = 0U; index < kElementCount; ++index) {
            if (data.vector_output[ordinal][index] != ExpectedVectorOutput(nonce, ordinal, index)) {
                return false;
            }
        }
    }
    for (uint32_t ordinal = 0U; ordinal < kCubeTaskCount; ++ordinal) {
        for (uint32_t row = 0U; row < kTileRows; ++row) {
            for (uint32_t column = 0U; column < kTileColumns; ++column) {
                if (data.cube_output[ordinal][row * kTileColumns + column] !=
                    ExpectedCubeOutput(nonce, ordinal, row, column)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool TestHalfPacketHidden() {
    const uint64_t nonce = 0xA540000000000001ULL;
    CpuData data{};
    InitializeData(&data, nonce);
    CpuTask task{};
    PausePoint pause;
    bool published = false;
    std::thread builder([&] {
        published = BuildTask(&task, data, nonce, 0U, &pause);
    });
    const bool reached = pause.WaitUntilReached();
    if (!reached) {
        pause.Release();
        builder.join();
        return false;
    }
    uint64_t expected = BuiltState(0U);
    const bool premature_claim = task.state.compare_exchange_strong(
        expected, ClaimedState(0U), std::memory_order_acq_rel, std::memory_order_acquire
    );
    const bool remained_building = task.state.load(std::memory_order_acquire) == BuildingState(0U);
    pause.Release();
    builder.join();
    return !premature_claim && remained_building && published &&
           task.state.load(std::memory_order_acquire) == BuiltState(0U) && PayloadValid(task, data, nonce, 0U);
}

bool RunRound(uint32_t round) {
    const uint64_t nonce = 0xA540000000000000ULL ^ (static_cast<uint64_t>(round + 1U) << 8U);
    CpuData data{};
    InitializeData(&data, nonce);
    std::array<CpuTask, kTaskCount> tasks{};
    std::array<uint32_t, kTaskCount> builder_thread{};
    builder_thread.fill(kBuilderThreadCount);
    std::atomic<uint32_t> ready{0U};
    std::atomic<bool> start{false};
    std::atomic<uint32_t> build_wins{0U};
    std::vector<std::thread> builders;
    builders.reserve(kBuilderThreadCount);
    for (uint32_t thread = 0U; thread < kBuilderThreadCount; ++thread) {
        builders.emplace_back([&, thread] {
            ready.fetch_add(1U, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (uint32_t task_index = BuilderFirstTask(thread); task_index < kTaskCount;
                 task_index += kBuilderThreadCount) {
                if (BuildTask(&tasks[task_index], data, nonce, task_index, nullptr)) {
                    builder_thread[task_index] = thread;
                    build_wins.fetch_add(1U, std::memory_order_relaxed);
                }
            }
        });
    }
    while (ready.load(std::memory_order_acquire) != kBuilderThreadCount) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (std::thread &builder : builders) {
        builder.join();
    }
    if (build_wins.load(std::memory_order_relaxed) != kTaskCount) {
        return false;
    }
    for (uint32_t task_index = 0U; task_index < kTaskCount; ++task_index) {
        if (tasks[task_index].state.load(std::memory_order_acquire) != BuiltState(task_index) ||
            builder_thread[task_index] != BuilderThreadForTask(task_index) ||
            !PayloadValid(tasks[task_index], data, nonce, task_index)) {
            return false;
        }
    }

    std::atomic<uint64_t> vector_done{0U};
    std::atomic<uint64_t> cube_done{0U};
    std::atomic<uint64_t> done_count{0U};
    CpuExecutor vector_executor(true, &vector_done, &done_count);
    CpuExecutor cube_executor(false, &cube_done, &done_count);

    if (!vector_executor.TryClaim(&tasks[0], 0U) || vector_executor.TryClaim(&tasks[2], 2U) ||
        tasks[2].state.load(std::memory_order_acquire) != BuiltState(2U) || !cube_executor.TryClaim(&tasks[1], 1U) ||
        cube_executor.TryClaim(&tasks[3], 3U) || tasks[3].state.load(std::memory_order_acquire) != BuiltState(3U) ||
        !vector_executor.ExecuteAndComplete(&tasks[0], &data, nonce, 0U) ||
        !cube_executor.ExecuteAndComplete(&tasks[1], &data, nonce, 1U)) {
        return false;
    }
    if (vector_executor.TryClaim(&tasks[3], 3U) || cube_executor.TryClaim(&tasks[2], 2U)) {
        return false;
    }

    bool vector_ok = true;
    bool cube_ok = true;
    std::thread vector_worker([&] {
        for (uint32_t task_index = 2U; task_index < kTaskCount; task_index += 2U) {
            vector_ok = vector_ok && vector_executor.TryClaim(&tasks[task_index], task_index) &&
                        vector_executor.ExecuteAndComplete(&tasks[task_index], &data, nonce, task_index);
        }
    });
    std::thread cube_worker([&] {
        for (uint32_t task_index = 3U; task_index < kTaskCount; task_index += 2U) {
            cube_ok = cube_ok && cube_executor.TryClaim(&tasks[task_index], task_index) &&
                      cube_executor.ExecuteAndComplete(&tasks[task_index], &data, nonce, task_index);
        }
    });
    vector_worker.join();
    cube_worker.join();

    if (!vector_ok || !cube_ok || vector_executor.Busy() || cube_executor.Busy() ||
        vector_executor.ClaimAttempts() != kVectorTaskCount || vector_executor.ClaimWins() != kVectorTaskCount ||
        vector_executor.Executed() != kVectorTaskCount || vector_executor.MaxBusy() != 1U ||
        vector_executor.BusyBlocked() != 1U || cube_executor.ClaimAttempts() != kCubeTaskCount ||
        cube_executor.ClaimWins() != kCubeTaskCount || cube_executor.Executed() != kCubeTaskCount ||
        cube_executor.MaxBusy() != 1U || cube_executor.BusyBlocked() != 1U ||
        vector_done.load(std::memory_order_acquire) != kVectorTaskCount ||
        cube_done.load(std::memory_order_acquire) != kCubeTaskCount ||
        done_count.load(std::memory_order_acquire) != kTaskCount || !OutputsValid(data, nonce)) {
        return false;
    }
    for (uint32_t task_index = 0U; task_index < kTaskCount; ++task_index) {
        const uint64_t final_state = tasks[task_index].state.load(std::memory_order_acquire);
        const DecodedExecState decoded = DecodeExecState(final_state);
        if (final_state != DoneState(task_index) || !decoded.valid || decoded.phase != ExecPhase::Done ||
            decoded.task_id != TaskId(task_index) || decoded.engine_class != TaskEngine(task_index) ||
            decoded.execute_owner != TaskExecutorOwner(task_index)) {
            return false;
        }
    }
    return true;
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
    if (!TestHalfPacketHidden()) {
        std::fprintf(stderr, "[FAIL] S4 CPU multi-task half-packet visibility\n");
        return EXIT_FAILURE;
    }
    for (uint32_t round = 0U; round < rounds; ++round) {
        if (!RunRound(round)) {
            std::fprintf(stderr, "[FAIL] S4 CPU multi-task round=%u\n", round);
            return EXIT_FAILURE;
        }
    }
    std::printf(
        "[PASS] S4 CPU multi-task rounds=%u: 4-warp/128-thread mapping builds 16 tasks, busy depth 1, fan-in and "
        "goldens\n",
        rounds
    );
    return EXIT_SUCCESS;
}
