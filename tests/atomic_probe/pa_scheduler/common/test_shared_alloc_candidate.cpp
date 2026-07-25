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

#include <cstdint>
#include <cstdio>
#include <new>
#include <sys/mman.h>

#define PA_DEVICE inline
#define PA_GM
#include "pa_scheduler_core.h"

namespace {

using namespace pa_scheduler;

int g_failures = 0;

struct ClaimTestOps {
    static constexpr bool kAtomicReturnReadyObserved = false;
    static inline uint64_t tick = 0;

    static int64_t FetchMax(
        volatile int64_t *address, int64_t value, uint64_t &retries
    ) {
        int64_t current = __atomic_load_n(address, __ATOMIC_ACQUIRE);
        retries = 0;
        while (value > current) {
            if (__atomic_compare_exchange_n(
                    address, &current, value, true, __ATOMIC_ACQ_REL,
                    __ATOMIC_ACQUIRE
                )) {
                break;
            }
            ++retries;
        }
        return current;
    }

    static uint64_t Now() {
        return ++tick;
    }

    template <typename T>
    static uint64_t NowAfterAtomicResult(T value) {
        asm volatile("" : "+r"(value));
        return Now();
    }
};

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] %s\n", message);
    ++g_failures;
}

void WorkerTopology(
    uint32_t worker_id, int32_t *block_id, int32_t *lane
) {
    if (worker_id < kAicWorkers) {
        *block_id = static_cast<int32_t>(worker_id);
        *lane = 0;
        return;
    }
    const uint32_t vector_id = worker_id - kAicWorkers;
    *block_id = static_cast<int32_t>(vector_id / 2U);
    *lane = static_cast<int32_t>(1U + vector_id % 2U);
}

template <typename T>
T *MapSparseObject() {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void *memory = mmap(
        nullptr, sizeof(T), PROT_READ | PROT_WRITE, flags, -1, 0
    );
    if (memory == MAP_FAILED) {
        return nullptr;
    }
    return ::new (memory) T;
}

void TestFullPaTopology() {
    int32_t owner_by_shard[kCursorShards] = {-1, -1, -1, -1};
    for (uint32_t task_id = 0; task_id < kMaxTasks; ++task_id) {
        uint32_t candidates = 0;
        int32_t owner = -1;
        for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
            int32_t block_id = -1;
            int32_t lane = -1;
            WorkerTopology(worker_id, &block_id, &lane);
            if (IsSharedAllocCandidate(
                    block_id, lane, task_id
                )) {
                ++candidates;
                owner = static_cast<int32_t>(worker_id);
            }
        }
        Check(candidates == 1, "each task has exactly one Alloc candidate");

        const uint32_t shard = task_id % kCursorShards;
        if (owner_by_shard[shard] < 0) {
            owner_by_shard[shard] = owner;
        }
        Check(
            owner_by_shard[shard] == owner,
            "all tasks on one Alloc cursor shard keep the same worker"
        );
    }
}

void TestExactShardOwners() {
    constexpr int32_t expected_workers[kCursorShards] = {0, 34, 37, 3};
    for (uint32_t shard = 0; shard < kCursorShards; ++shard) {
        int32_t owner = -1;
        for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
            int32_t block_id = -1;
            int32_t lane = -1;
            WorkerTopology(worker_id, &block_id, &lane);
            if (IsSharedAllocCandidate(block_id, lane, shard)) {
                owner = static_cast<int32_t>(worker_id);
            }
        }
        Check(
            owner == expected_workers[shard],
            "each Alloc cursor shard keeps its documented physical owner"
        );
    }
}

void TestOwnedTaskCountAndSum() {
    constexpr uint32_t batch_cases[] = {1, 2, 4, 5, 17, kMaxBatches};
    for (uint32_t batches : batch_cases) {
        uint64_t total_count = 0;
        uint64_t total_task_id_sum = 0;
        for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
            int32_t block_id = -1;
            int32_t lane = -1;
            WorkerTopology(worker_id, &block_id, &lane);
            total_count +=
                SharedAllocOwnedTaskCount(block_id, lane, batches);
            total_task_id_sum +=
                SharedAllocOwnedTaskIdSum(block_id, lane, batches);
        }
        Check(
            total_count == batches,
            "fixed owners cover each batch Alloc exactly once"
        );
        Check(
            total_task_id_sum ==
                static_cast<uint64_t>(kTasksPerBatch) * batches *
                    (batches - 1U) / 2U,
            "fixed-owner Alloc task-id sums close exactly"
        );
    }

    Check(
        SharedAllocOwnedTaskCount(0, 0, kMaxBatches) == 64 &&
            SharedAllocOwnedTaskCount(1, 1, kMaxBatches) == 64 &&
            SharedAllocOwnedTaskCount(2, 2, kMaxBatches) == 64 &&
            SharedAllocOwnedTaskCount(3, 0, kMaxBatches) == 64,
        "b256 distributes the four cursor shards evenly across owners"
    );
    Check(
        SharedAllocOwnedTaskCount(4, 0, kMaxBatches) == 0 &&
            SharedAllocOwnedTaskIdSum(4, 0, kMaxBatches) == 0,
        "nonowner workers have no full-path shared Alloc"
    );
}

void TestClaimOrderingContract() {
    SchedulerState *state = MapSparseObject<SchedulerState>();
    WorkerState *candidate = MapSparseObject<WorkerState>();
    WorkerState *noncandidate = MapSparseObject<WorkerState>();
    Check(
        state != nullptr && candidate != nullptr && noncandidate != nullptr,
        "sparse Claim fixtures map successfully"
    );
    if (state == nullptr || candidate == nullptr || noncandidate == nullptr) {
        if (state != nullptr) munmap(state, sizeof(*state));
        if (candidate != nullptr) munmap(candidate, sizeof(*candidate));
        if (noncandidate != nullptr) munmap(noncandidate, sizeof(*noncandidate));
        return;
    }

    candidate->block_id = 0;
    candidate->lane = 0;
    noncandidate->block_id = 1;
    noncandidate->lane = 0;
    LocalStats candidate_stats{};
    LocalStats noncandidate_stats{};
    state->alloc_cursor[0].value = -1;

    const ClaimOutcome first = Claim<ClaimTestOps>(
        state, *candidate, 0, TaskKind::Alloc, candidate_stats
    );
    Check(
        first.attempted && first.won && state->alloc_cursor[0].value == 0,
        "fixed owner wins the first task on its Alloc cursor"
    );

    const ClaimOutcome skipped = Claim<ClaimTestOps>(
        state, *noncandidate, 0, TaskKind::Alloc, noncandidate_stats
    );
    Check(
        !skipped.attempted && !skipped.won &&
            state->alloc_cursor[0].value == 0 &&
            noncandidate_stats.result.atomic_trace_calls == 0,
        "noncandidate neither issues atomicMax nor changes its cursor"
    );

    const ClaimOutcome second = Claim<ClaimTestOps>(
        state, *candidate, 20, TaskKind::Alloc, candidate_stats
    );
    const ClaimOutcome third = Claim<ClaimTestOps>(
        state, *candidate, 40, TaskKind::Alloc, candidate_stats
    );
    Check(
        second.attempted && second.won && third.attempted && third.won &&
            state->alloc_cursor[0].value == 40,
        "one fixed owner advances same-shard tasks in increasing order"
    );

    state->alloc_cursor[0].value = -1;
    const ClaimOutcome later_first = Claim<ClaimTestOps>(
        state, *candidate, 20, TaskKind::Alloc, candidate_stats
    );
    const ClaimOutcome stale_after = Claim<ClaimTestOps>(
        state, *candidate, 0, TaskKind::Alloc, candidate_stats
    );
    Check(
        later_first.won && stale_after.attempted && !stale_after.won &&
            state->alloc_cursor[0].value == 20,
        "later-first counterexample proves why candidate Claim loss is fatal"
    );

    munmap(noncandidate, sizeof(*noncandidate));
    munmap(candidate, sizeof(*candidate));
    munmap(state, sizeof(*state));
}

void TestRejectedInputs() {
    Check(
        !IsSharedAllocCandidate(0, 0, kTaskCellCapacity),
        "task outside the completion table is rejected"
    );
}

}  // namespace

int main() {
    TestFullPaTopology();
    TestExactShardOwners();
    TestOwnedTaskCountAndSum();
    TestClaimOrderingContract();
    TestRejectedInputs();
    if (g_failures != 0) {
        std::fprintf(
            stderr, "[FAIL] shared Alloc candidate tests: %d\n", g_failures
        );
        return 1;
    }
    std::printf("[PASS] shared Alloc candidate tests\n");
    return 0;
}
