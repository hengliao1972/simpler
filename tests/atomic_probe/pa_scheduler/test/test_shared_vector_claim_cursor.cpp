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

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <new>
#include <sys/mman.h>
#include <thread>
#include <vector>

#define PA_DEVICE inline
#define PA_GM
#include "pa_scheduler_core.h"

namespace {

using namespace pa_scheduler;

constexpr std::array<TaskKind, 8> kTaskKinds = {
    TaskKind::Alloc, TaskKind::Qk, TaskKind::Sf, TaskKind::Pv,
    TaskKind::Up, TaskKind::Pv, TaskKind::Alloc, TaskKind::Up,
};
constexpr std::array<uint32_t, 8> kTaskIds = {
    100, 101, 102, 103, 104, 105, 108, 110,
};

int g_failures = 0;

struct ClaimTestOps {
    static constexpr bool kAtomicReturnReadyObserved = false;
    static inline thread_local uint64_t fetch_max_calls = 0;
    static inline thread_local volatile int64_t *last_fetch_max_address = nullptr;

    static int32_t Exchange(volatile int32_t *address, int32_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t FetchMax(volatile int64_t *address, int64_t value, uint64_t &retries) {
        ++fetch_max_calls;
        last_fetch_max_address = address;
        int64_t current = __atomic_load_n(address, __ATOMIC_ACQUIRE);
        retries = 0;
        while (value > current) {
            if (__atomic_compare_exchange_n(address, &current, value, true, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
                break;
            }
            ++retries;
        }
        return current;
    }

    static uint64_t Now() { return 0; }

    template <typename T>
    static uint64_t NowAfterAtomicResult(T) {
        return 0;
    }

    static void ResetThreadTrace() {
        fetch_max_calls = 0;
        last_fetch_max_address = nullptr;
    }
};

struct ClaimEvidence {
    ClaimOutcome outcome{};
    uint64_t fetch_max_calls = 0;
    volatile int64_t *fetch_max_address = nullptr;
};

struct CursorValues {
    std::array<int64_t, kCursorShards> cube{};
    std::array<int64_t, kCursorShards> vector{};
    std::array<int64_t, kCursorShards> alloc{};
    std::array<int64_t, kSharedVectorCursorCapacity> shared_vector{};
};

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] %s\n", message);
    ++g_failures;
}

template <typename T>
T *MapSparseObject() {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void *memory = mmap(nullptr, sizeof(T), PROT_READ | PROT_WRITE, flags, -1, 0);
    if (memory == MAP_FAILED) {
        return nullptr;
    }
    return ::new (memory) T;
}

void InitializeClaimCursors(SchedulerState &state, CursorValues &expected) {
    expected.cube.fill(-1);
    expected.vector.fill(-1);
    expected.alloc.fill(-1);
    expected.shared_vector.fill(-1);
    for (uint32_t shard = 0; shard < kCursorShards; ++shard) {
        state.cube_cursor[shard].value = -1;
        state.vector_cursor[shard].value = -1;
        state.alloc_cursor[shard].value = -1;
    }
    for (uint32_t shard = 0; shard < kSharedVectorCursorCapacity; ++shard) {
        state.shared_map.shared_vector_cursor[shard].value = -1;
    }
}

volatile int64_t *ExpectedClaimAddress(SchedulerState &state, uint32_t task_id, TaskKind kind) {
    if (kind == TaskKind::Alloc) {
        return &state.alloc_cursor[task_id % kCursorShards].value;
    }
    if (kind == TaskKind::Qk || kind == TaskKind::Pv) {
        return &state.cube_cursor[task_id % kCursorShards].value;
    }
    return &state.shared_map.shared_vector_cursor[task_id % kSharedVectorCursorShards].value;
}

void RecordExpectedCursor(CursorValues &expected, uint32_t task_id, TaskKind kind) {
    if (kind == TaskKind::Alloc) {
        expected.alloc[task_id % kCursorShards] = task_id;
    } else if (kind == TaskKind::Qk || kind == TaskKind::Pv) {
        expected.cube[task_id % kCursorShards] = task_id;
    } else {
        expected.shared_vector[task_id % kSharedVectorCursorShards] = task_id;
    }
}

bool CursorsMatch(const SchedulerState &state, const CursorValues &expected) {
    for (uint32_t shard = 0; shard < kCursorShards; ++shard) {
        if (state.cube_cursor[shard].value != expected.cube[shard] ||
            state.vector_cursor[shard].value != expected.vector[shard] ||
            state.alloc_cursor[shard].value != expected.alloc[shard]) {
            return false;
        }
    }
    for (uint32_t shard = 0; shard < kSharedVectorCursorCapacity; ++shard) {
        if (state.shared_map.shared_vector_cursor[shard].value != expected.shared_vector[shard]) {
            return false;
        }
    }
    return true;
}

uint32_t ExpectedCandidates(TaskKind kind) {
    switch (kind) {
    case TaskKind::Alloc:
        return kWorkers;
    case TaskKind::Qk:
    case TaskKind::Pv:
        return kAicWorkers;
    case TaskKind::Sf:
    case TaskKind::Up:
        return kAivWorkers;
    case TaskKind::Count:
        return 0;
    }
    return 0;
}

bool IsCandidate(
    TaskKind kind, uint32_t worker_id, uint32_t
) {
    if (kind == TaskKind::Alloc) {
        return true;
    }
    const bool is_aic = worker_id < kAicWorkers;
    return kind == TaskKind::Qk || kind == TaskKind::Pv
        ? is_aic
        : !is_aic;
}

bool RunConcurrentClaim(SchedulerState &state, uint32_t task_id, TaskKind kind) {
    state.tasks[task_id].deps_prepared = -1;
    state.fatal.value = 0;
    std::array<ClaimEvidence, kWorkers> evidence{};
    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    threads.reserve(kWorkers);

    for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
        WorkerState &worker = state.workers[worker_id];
        worker.role = worker_id < kAicWorkers ? CoreRole::Aic : CoreRole::Aiv;
        worker.core_idx = static_cast<int32_t>(worker_id);
        threads.emplace_back([&state, &worker, &evidence, &ready, &start, worker_id, task_id, kind]() {
            ClaimTestOps::ResetThreadTrace();
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {}
            LocalStats stats{};
            evidence[worker_id].outcome = Claim<ClaimTestOps>(&state, worker, task_id, kind, stats);
            evidence[worker_id].fetch_max_calls = ClaimTestOps::fetch_max_calls;
            evidence[worker_id].fetch_max_address = ClaimTestOps::last_fetch_max_address;
        });
    }

    while (ready.load(std::memory_order_acquire) != kWorkers) {}
    start.store(true, std::memory_order_release);
    for (std::thread &thread : threads) {
        thread.join();
    }

    uint32_t attempts = 0;
    uint32_t winners = 0;
    bool exact = true;
    volatile int64_t *expected_address = ExpectedClaimAddress(state, task_id, kind);
    for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
        const bool candidate =
            IsCandidate(kind, worker_id, task_id);
        const ClaimEvidence &entry = evidence[worker_id];
        attempts += entry.outcome.attempted ? 1U : 0U;
        winners += entry.outcome.won ? 1U : 0U;
        exact &= entry.outcome.attempted == candidate && entry.fetch_max_calls == (candidate ? 1U : 0U) &&
                 entry.fetch_max_address == (candidate ? expected_address : nullptr);
        if (entry.outcome.won) {
            exact &= kind == TaskKind::Alloc ? entry.outcome.function_id == -1 :
                                               entry.outcome.function_id == FunctionId(kind);
        } else {
            exact &= entry.outcome.function_id == -1;
        }
    }
    return exact && attempts == ExpectedCandidates(kind) && winners == 1 &&
           *expected_address == static_cast<int64_t>(task_id) && state.tasks[task_id].deps_prepared == -1 &&
           state.fatal.value == 0;
}

void TestAllTaskKindsUseCursorClaim() {
    SchedulerState *state = MapSparseObject<SchedulerState>();
    Check(state != nullptr, "sparse shared cursor Claim fixture maps successfully");
    if (state == nullptr) {
        return;
    }

    CursorValues expected{};
    InitializeClaimCursors(*state, expected);
    bool exact = true;
    for (uint32_t index = 0; index < kTaskKinds.size(); ++index) {
        exact &= RunConcurrentClaim(*state, kTaskIds[index], kTaskKinds[index]);
        RecordExpectedCursor(expected, kTaskIds[index], kTaskKinds[index]);
        exact &= CursorsMatch(*state, expected);
        for (uint32_t observed = 0; observed <= index; ++observed) {
            exact &= state->tasks[kTaskIds[observed]].deps_prepared == -1;
        }
    }

    // 后三项分别复用 QK101 的 cube shard、Alloc100 的 alloc
    // shard 和 SF102 的 shared-vector shard。它们仍各自产生唯一
    // winner，证明完整 role 候选集合能推进同一高水位链。
    // 再回放较旧的 QK101 时，cube cursor 已被 PV105 推进到 105，
    // 本 shard 候选必须正常发 atomic 并判输。
    WorkerState &replay_worker = state->workers[1];
    replay_worker.role = CoreRole::Aic;
    replay_worker.core_idx = 1;
    ClaimTestOps::ResetThreadTrace();
    LocalStats replay_stats{};
    const ClaimOutcome replay = Claim<ClaimTestOps>(state, replay_worker, kTaskIds[1], TaskKind::Qk, replay_stats);
    exact &= replay.attempted && !replay.won && replay.function_id == -1 && ClaimTestOps::fetch_max_calls == 1 &&
             ClaimTestOps::last_fetch_max_address == ExpectedClaimAddress(*state, kTaskIds[1], TaskKind::Qk) &&
             state->fatal.value == 0 && CursorsMatch(*state, expected);

    Check(
        exact, "all task kinds keep cursor routing, exact candidates, "
               "role participants, one winner, legal replay, "
               "and untouched deps_prepared"
    );
    (void)munmap(state, sizeof(*state));
}

}  // namespace

int main() {
    TestAllTaskKindsUseCursorClaim();
    if (g_failures != 0) {
        std::fprintf(stderr, "[FAIL] shared cursor Claim tests: %d\n", g_failures);
        return 1;
    }
    std::printf("[PASS] shared cursor Claim tests\n");
    return 0;
}
