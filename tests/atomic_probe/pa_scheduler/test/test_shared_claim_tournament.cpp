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
#include <cstring>
#include <new>
#include <sys/mman.h>
#include <thread>
#include <vector>

#define PA_DEVICE inline
#define PA_GM
#include "pa_scheduler_core.h"

namespace {

using namespace pa_scheduler;

constexpr std::array<TaskKind, 5> kTaskKinds = {
    TaskKind::Alloc, TaskKind::Qk, TaskKind::Sf,
    TaskKind::Pv, TaskKind::Up,
};
constexpr std::array<uint32_t, 5> kTaskIds = {
    100, 101, 102, 103, 104,
};

int g_failures = 0;

struct ClaimTestOps {
    static constexpr bool kAtomicReturnReadyObserved = false;
    static inline thread_local uint32_t cas_calls = 0;
    static inline thread_local std::array<volatile int64_t *, 2>
        cas_addresses{};

    static int32_t Exchange(
        volatile int32_t *address, int32_t value
    ) {
        return __atomic_exchange_n(
            address, value, __ATOMIC_ACQ_REL
        );
    }

    static int64_t CompareExchange(
        volatile int64_t *address, int64_t expected,
        int64_t desired
    ) {
        if (cas_calls < cas_addresses.size()) {
            cas_addresses[cas_calls] = address;
        }
        ++cas_calls;
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(
            address, &observed, desired, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        return observed;
    }

    static uint64_t Now() { return 0; }

    template <typename T>
    static uint64_t NowAfterAtomicResult(T) {
        return 0;
    }

    static void ResetThreadTrace() {
        cas_calls = 0;
        cas_addresses.fill(nullptr);
    }
};

struct ClaimEvidence {
    ClaimOutcome outcome{};
    uint32_t cas_calls = 0;
    std::array<volatile int64_t *, 2> cas_addresses{};
};

struct LegacyCursorSnapshot {
    std::array<int64_t, kCursorShards> cube{};
    std::array<int64_t, kCursorShards> vector{};
    std::array<int64_t, kCursorShards> alloc{};
    std::array<int64_t, kSharedVectorCursorCapacity>
        shared_vector{};
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
    void *memory = mmap(
        nullptr, sizeof(T), PROT_READ | PROT_WRITE,
        flags, -1, 0
    );
    if (memory == MAP_FAILED) {
        return nullptr;
    }
    return ::new (memory) T;
}

LegacyCursorSnapshot SeedLegacyCursors(SchedulerState &state) {
    LegacyCursorSnapshot snapshot{};
    for (uint32_t shard = 0; shard < kCursorShards; ++shard) {
        snapshot.cube[shard] = 1000 + shard;
        snapshot.vector[shard] = 2000 + shard;
        snapshot.alloc[shard] = 3000 + shard;
        state.cube_cursor[shard].value = snapshot.cube[shard];
        state.vector_cursor[shard].value =
            snapshot.vector[shard];
        state.alloc_cursor[shard].value = snapshot.alloc[shard];
    }
    for (uint32_t shard = 0;
         shard < kSharedVectorCursorCapacity; ++shard) {
        snapshot.shared_vector[shard] = 4000 + shard;
        state.shared_map.shared_vector_cursor[shard].value =
            snapshot.shared_vector[shard];
    }
    return snapshot;
}

bool LegacyCursorsMatch(
    const SchedulerState &state,
    const LegacyCursorSnapshot &snapshot
) {
    for (uint32_t shard = 0; shard < kCursorShards; ++shard) {
        if (state.cube_cursor[shard].value !=
                snapshot.cube[shard] ||
            state.vector_cursor[shard].value !=
                snapshot.vector[shard] ||
            state.alloc_cursor[shard].value !=
                snapshot.alloc[shard]) {
            return false;
        }
    }
    for (uint32_t shard = 0;
         shard < kSharedVectorCursorCapacity; ++shard) {
        if (state.shared_map.shared_vector_cursor[shard].value !=
            snapshot.shared_vector[shard]) {
            return false;
        }
    }
    return true;
}

uint32_t ExpectedCandidates(TaskKind kind) {
    switch (kind) {
        case TaskKind::Alloc:
            return kSharedAllocClaimParticipants;
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

uint32_t ExpectedGroups(TaskKind kind) {
    switch (kind) {
        case TaskKind::Alloc:
            return kSharedAllocClaimTournamentGroups;
        case TaskKind::Qk:
        case TaskKind::Pv:
            return kSharedAicClaimTournamentGroups;
        case TaskKind::Sf:
        case TaskKind::Up:
            return kSharedAivClaimTournamentGroups;
        case TaskKind::Count:
            return 0;
    }
    return 0;
}

bool IsCandidate(
    TaskKind kind, uint32_t worker_id, uint32_t task_id
) {
    if (kind == TaskKind::Alloc) {
        return worker_id % kCursorShards ==
               task_id % kCursorShards;
    }
    const bool is_aic = worker_id < kAicWorkers;
    return kind == TaskKind::Qk || kind == TaskKind::Pv
        ? is_aic
        : !is_aic;
}

uint32_t CandidateRank(
    TaskKind kind, uint32_t worker_id
) {
    if (kind == TaskKind::Alloc) {
        return worker_id / kCursorShards;
    }
    return kind == TaskKind::Qk || kind == TaskKind::Pv
        ? worker_id
        : worker_id - kAicWorkers;
}

void ResetTournamentTask(
    SchedulerState &state, uint32_t task_id
) {
    std::memset(
        &state.claim_tournament[task_id], 0xff,
        sizeof(state.claim_tournament[task_id])
    );
    state.tasks[task_id].deps_prepared = -1;
}

bool TournamentStateMatches(
    const SchedulerState &state, uint32_t task_id,
    TaskKind kind
) {
    const int64_t expected = static_cast<int64_t>(task_id);
    const uint32_t active_groups = ExpectedGroups(kind);
    const SharedClaimTournamentTask &tournament =
        state.claim_tournament[task_id];
    bool exact = tournament.root.owner.value == expected;
    for (uint32_t group = 0;
         group < kSharedClaimTournamentMaxGroups; ++group) {
        exact &= tournament.local[group].owner.value ==
            (group < active_groups ? expected : -1);
    }
    return exact;
}

bool RunConcurrentClaim(
    SchedulerState &state, uint32_t task_id, TaskKind kind,
    bool reset_state, bool expect_winner
) {
    if (reset_state) {
        ResetTournamentTask(state, task_id);
    }
    state.fatal.value = 0;
    std::array<ClaimEvidence, kWorkers> evidence{};
    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    threads.reserve(kWorkers);

    for (uint32_t worker_id = 0;
         worker_id < kWorkers; ++worker_id) {
        WorkerState &worker = state.workers[worker_id];
        worker.role = worker_id < kAicWorkers
            ? CoreRole::Aic
            : CoreRole::Aiv;
        worker.core_idx = static_cast<int32_t>(worker_id);
        threads.emplace_back([
            &state, &worker, &evidence, &ready, &start,
            worker_id, task_id, kind
        ]() {
            ClaimTestOps::ResetThreadTrace();
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            LocalStats stats{};
            evidence[worker_id].outcome =
                Claim<ClaimTestOps>(
                    &state, worker_id, worker.role,
                    task_id, kind, stats
                );
            evidence[worker_id].cas_calls =
                ClaimTestOps::cas_calls;
            evidence[worker_id].cas_addresses =
                ClaimTestOps::cas_addresses;
        });
    }

    while (ready.load(std::memory_order_acquire) != kWorkers) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (std::thread &thread : threads) {
        thread.join();
    }

    const uint32_t groups = ExpectedGroups(kind);
    const SharedClaimTournamentTask &tournament =
        state.claim_tournament[task_id];
    uint32_t attempts = 0;
    uint32_t winners = 0;
    uint32_t root_attempts = 0;
    uint32_t physical_cas = 0;
    bool exact = true;
    for (uint32_t worker_id = 0;
         worker_id < kWorkers; ++worker_id) {
        const bool candidate =
            IsCandidate(kind, worker_id, task_id);
        const ClaimEvidence &entry = evidence[worker_id];
        attempts += entry.outcome.attempted ? 1U : 0U;
        winners += entry.outcome.won ? 1U : 0U;
        root_attempts += entry.cas_calls == 2 ? 1U : 0U;
        physical_cas += entry.cas_calls;
        exact &= entry.outcome.attempted == candidate;
        if (!candidate) {
            exact &= entry.cas_calls == 0 &&
                entry.cas_addresses[0] == nullptr &&
                entry.outcome.function_id == -1;
            continue;
        }
        const uint32_t group =
            CandidateRank(kind, worker_id) % groups;
        exact &= entry.cas_calls >= 1 && entry.cas_calls <= 2;
        exact &= entry.cas_addresses[0] ==
            &tournament.local[group].owner.value;
        if (entry.cas_calls == 2) {
            exact &= entry.cas_addresses[1] ==
                &tournament.root.owner.value;
        }
        if (entry.outcome.won) {
            exact &= kind == TaskKind::Alloc
                ? entry.outcome.function_id == -1
                : entry.outcome.function_id == FunctionId(kind);
        } else {
            exact &= entry.outcome.function_id == -1;
        }
    }

    const uint32_t expected_root_attempts =
        expect_winner ? groups : 0;
    const uint32_t expected_physical_cas =
        ExpectedCandidates(kind) + expected_root_attempts;
    return exact &&
        attempts == ExpectedCandidates(kind) &&
        winners == (expect_winner ? 1U : 0U) &&
        root_attempts == expected_root_attempts &&
        physical_cas == expected_physical_cas &&
        TournamentStateMatches(state, task_id, kind) &&
        state.tasks[task_id].deps_prepared == -1 &&
        state.fatal.value == 0;
}

void TestAllTaskKindsAndReplay() {
    SchedulerState *state = MapSparseObject<SchedulerState>();
    Check(
        state != nullptr,
        "sparse shared Claim Tournament state maps successfully"
    );
    if (state == nullptr) {
        return;
    }

    const LegacyCursorSnapshot legacy = SeedLegacyCursors(*state);
    bool exact = true;
    for (uint32_t index = 0; index < kTaskKinds.size(); ++index) {
        exact &= RunConcurrentClaim(
            *state, kTaskIds[index], kTaskKinds[index],
            true, true
        );
        // 同一个 task 的完整 replay 只能再次做 local CAS 并全部失败，
        // 不能重新进入 root，也不能产生第二个 owner。
        exact &= RunConcurrentClaim(
            *state, kTaskIds[index], kTaskKinds[index],
            false, false
        );
    }
    exact &= LegacyCursorsMatch(*state, legacy);

    Check(
        exact,
        "all task kinds preserve 24/32/64 candidates, exact-one "
        "owner, immediate replay losers, untouched deps_prepared, "
        "and untouched legacy cursors"
    );
    (void)munmap(state, sizeof(*state));
}

void TestFutureTaskCannotOverwriteDelayedTask() {
    SchedulerState *state = MapSparseObject<SchedulerState>();
    Check(
        state != nullptr,
        "sparse out-of-order Tournament state maps successfully"
    );
    if (state == nullptr) {
        return;
    }

    constexpr uint32_t kEarlierTask = 200;
    constexpr uint32_t kFutureTask = 208;
    const LegacyCursorSnapshot legacy = SeedLegacyCursors(*state);
    // 先完成 future task 的仲裁，再让 earlier task 开始。两者使用独立
    // 节点，因此 future root 不会把 earlier 的线性化机会覆盖掉。
    const bool future_ok = RunConcurrentClaim(
        *state, kFutureTask, TaskKind::Up, true, true
    );
    const bool earlier_ok = RunConcurrentClaim(
        *state, kEarlierTask, TaskKind::Up, true, true
    );
    Check(
        future_ok && earlier_ok &&
            TournamentStateMatches(
                *state, kFutureTask, TaskKind::Up
            ) &&
            TournamentStateMatches(
                *state, kEarlierTask, TaskKind::Up
            ) &&
            state->tasks[kEarlierTask].deps_prepared == -1 &&
            state->tasks[kFutureTask].deps_prepared == -1 &&
            LegacyCursorsMatch(*state, legacy),
        "future task election cannot overwrite a delayed earlier task; "
        "TensorMap order remains delegated to deps_prepared"
    );
    (void)munmap(state, sizeof(*state));
}

}  // namespace

int main() {
    TestAllTaskKindsAndReplay();
    TestFutureTaskCannotOverwriteDelayedTask();
    if (g_failures != 0) {
        std::fprintf(
            stderr,
            "[FAIL] shared Claim Tournament tests: %d\n",
            g_failures
        );
        return 1;
    }
    std::printf("[PASS] shared Claim Tournament tests\n");
    return 0;
}
