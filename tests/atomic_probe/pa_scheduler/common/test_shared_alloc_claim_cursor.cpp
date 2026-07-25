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
    static inline uint64_t fetch_max_calls = 0;
    static inline volatile int64_t *last_fetch_max_address = nullptr;

    static int64_t FetchMax(
        volatile int64_t *address, int64_t value, uint64_t &retries
    ) {
        ++fetch_max_calls;
        last_fetch_max_address = address;
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

    static int32_t Exchange(volatile int32_t *address, int32_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static uint64_t Now() {
        return ++tick;
    }

    template <typename T>
    static uint64_t NowAfterAtomicResult(T value) {
        asm volatile("" : "+r"(value));
        return Now();
    }

    static void ResetFetchMaxTrace() {
        fetch_max_calls = 0;
        last_fetch_max_address = nullptr;
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
    uint32_t worker_id, int32_t *block_id, int32_t *lane,
    CoreRole *role
) {
    if (worker_id < kAicWorkers) {
        *block_id = static_cast<int32_t>(worker_id);
        *lane = 0;
        *role = CoreRole::Aic;
        return;
    }
    const uint32_t vector_id = worker_id - kAicWorkers;
    *block_id = static_cast<int32_t>(vector_id / 2U);
    *lane = static_cast<int32_t>(1U + vector_id % 2U);
    *role = CoreRole::Aiv;
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

void TestPaAllocOwnerTopology() {
    uint32_t wins_by_worker[kWorkers] = {};
    int32_t owner_by_cursor[kSharedAllocCursorLanes]
                           [kSharedAllocCursorShards];
    for (uint32_t lane = 0; lane < kSharedAllocCursorLanes; ++lane) {
        for (uint32_t shard = 0; shard < kSharedAllocCursorShards; ++shard) {
            owner_by_cursor[lane][shard] = -1;
        }
    }

    for (uint32_t batch = 0; batch < kMaxBatches; ++batch) {
        const uint32_t task_id = batch * kTasksPerBatch;
        uint32_t candidates = 0;
        int32_t owner = -1;
        for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
            int32_t block_id = -1;
            int32_t lane = -1;
            CoreRole role = CoreRole::Aic;
            WorkerTopology(worker_id, &block_id, &lane, &role);
            if (IsSharedAllocCandidate(block_id, lane, task_id)) {
                ++candidates;
                owner = static_cast<int32_t>(worker_id);
            }
        }
        Check(candidates == 1, "each PA Alloc task has exactly one owner");
        if (owner < 0) {
            continue;
        }
        ++wins_by_worker[static_cast<uint32_t>(owner)];

        const uint32_t lane = task_id % kSharedAllocCursorLanes;
        const uint32_t shard = task_id & kSharedAllocCursorShardMask;
        if (owner_by_cursor[lane][shard] < 0) {
            owner_by_cursor[lane][shard] = owner;
        }
        Check(
            owner_by_cursor[lane][shard] == owner,
            "one shared Alloc cursor keeps one physical owner"
        );
    }

    uint32_t active_owners = 0;
    uint32_t owners_with_ten = 0;
    uint32_t owners_with_eleven = 0;
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        if (wins_by_worker[worker] == 0) {
            continue;
        }
        ++active_owners;
        owners_with_ten += wins_by_worker[worker] == 10;
        owners_with_eleven += wins_by_worker[worker] == 11;
        Check(
            wins_by_worker[worker] == 10 || wins_by_worker[worker] == 11,
            "b256 Alloc owners receive only ten or eleven tasks"
        );
    }
    Check(active_owners == 24, "b256 covers all 24 shared Alloc owners");
    Check(
        owners_with_ten == 8 && owners_with_eleven == 16,
        "b256 shared Alloc owner distribution is 8x10 plus 16x11"
    );

    for (uint32_t lane = 0; lane < kSharedAllocCursorLanes; ++lane) {
        for (uint32_t shard = 0; shard < kSharedAllocCursorShards; ++shard) {
            const int32_t expected =
                lane == 0
                    ? static_cast<int32_t>(shard)
                    : static_cast<int32_t>(
                          kAicWorkers + 2U * shard + (lane - 1U)
                      );
            Check(
                owner_by_cursor[lane][shard] == expected,
                "each (lane, shard) maps to its documented worker"
            );
        }
    }
}

void InitializeClaimCursors(SchedulerState *state) {
    for (uint32_t shard = 0; shard < kCursorShards; ++shard) {
        state->cube_cursor[shard].value = -1;
        state->vector_cursor[shard].value = -1;
        state->alloc_cursor[shard].value = -1;
    }
    for (uint32_t lane = 0; lane < kSharedAllocCursorLanes; ++lane) {
        for (uint32_t shard = 0; shard < kSharedAllocCursorShards; ++shard) {
            state->shared_map.shared_alloc_cursor[lane][shard].value = -1;
        }
    }
    state->fatal.value = 0;
}

void TestClaimRoutingAndOrdering() {
    SchedulerState *state = MapSparseObject<SchedulerState>();
    WorkerState *worker = MapSparseObject<WorkerState>();
    Check(
        state != nullptr && worker != nullptr,
        "sparse shared Claim fixtures map successfully"
    );
    if (state == nullptr || worker == nullptr) {
        if (state != nullptr) munmap(state, sizeof(*state));
        if (worker != nullptr) munmap(worker, sizeof(*worker));
        return;
    }

    InitializeClaimCursors(state);
    worker->block_id = 0;
    worker->lane = 0;
    worker->role = CoreRole::Aic;
    LocalStats owner_stats{};
    ClaimTestOps::ResetFetchMaxTrace();
    const ClaimOutcome first = Claim<ClaimTestOps>(
        state, *worker, 0, TaskKind::Alloc, owner_stats
    );
    Check(
        first.attempted && first.won &&
            state->shared_map.shared_alloc_cursor[0][0].value == 0,
        "fixed owner wins the first task on its shared Alloc cursor"
    );
    Check(
        ClaimTestOps::fetch_max_calls == 1 &&
            ClaimTestOps::last_fetch_max_address ==
                &state->shared_map.shared_alloc_cursor[0][0].value,
        "owner issues exactly one FetchMax to the expected cursor address"
    );

    worker->block_id = 1;
    LocalStats nonowner_stats{};
    const uint64_t calls_before_rejected = ClaimTestOps::fetch_max_calls;
    const ClaimOutcome rejected = Claim<ClaimTestOps>(
        state, *worker, 0, TaskKind::Alloc, nonowner_stats
    );
    Check(
        !rejected.attempted && !rejected.won &&
            state->shared_map.shared_alloc_cursor[0][0].value == 0 &&
            ClaimTestOps::fetch_max_calls == calls_before_rejected,
        "non-owner neither issues atomicMax nor changes shared Alloc cursor"
    );

    worker->block_id = 0;
    const ClaimOutcome second = Claim<ClaimTestOps>(
        state, *worker, 120, TaskKind::Alloc, owner_stats
    );
    const ClaimOutcome third = Claim<ClaimTestOps>(
        state, *worker, 240, TaskKind::Alloc, owner_stats
    );
    Check(
        second.attempted && second.won && third.attempted && third.won &&
            state->shared_map.shared_alloc_cursor[0][0].value == 240,
        "one owner advances same-cursor Alloc tasks in steps of 120"
    );

    state->shared_map.shared_alloc_cursor[0][0].value = -1;
    const ClaimOutcome later_first = Claim<ClaimTestOps>(
        state, *worker, 120, TaskKind::Alloc, owner_stats
    );
    const ClaimOutcome stale_after = Claim<ClaimTestOps>(
        state, *worker, 0, TaskKind::Alloc, owner_stats
    );
    const bool stale_accepted = ValidateSharedAllocOwnerClaim<ClaimTestOps>(
        state, owner_stats, 0, stale_after
    );
    Check(
        later_first.won && stale_after.attempted && !stale_after.won &&
            !stale_accepted && state->fatal.value == 1 &&
            state->shared_map.shared_alloc_cursor[0][0].value == 120,
        "later-first owner-order violation publishes fatal through production helper"
    );

    InitializeClaimCursors(state);
    worker->block_id = 0;
    worker->lane = 0;
    worker->role = CoreRole::Aic;
    LocalStats kernel_stats{};
    ClaimTestOps::ResetFetchMaxTrace();
    const ClaimOutcome qk = Claim<ClaimTestOps>(
        state, *worker, 1, TaskKind::Qk, kernel_stats
    );
    worker->lane = 1;
    worker->role = CoreRole::Aiv;
    const ClaimOutcome sf = Claim<ClaimTestOps>(
        state, *worker, 2, TaskKind::Sf, kernel_stats
    );
    Check(
        qk.attempted && qk.won && sf.attempted && sf.won &&
            state->cube_cursor[1].value == 1 &&
            state->vector_cursor[2].value == 2,
        "kernel Claim keeps the original cube/vector four-shard routing"
    );
    bool shared_alloc_untouched = true;
    for (uint32_t lane = 0; lane < kSharedAllocCursorLanes; ++lane) {
        for (uint32_t shard = 0; shard < kSharedAllocCursorShards; ++shard) {
            shared_alloc_untouched &=
                state->shared_map.shared_alloc_cursor[lane][shard].value == -1;
        }
    }
    Check(
        shared_alloc_untouched,
        "kernel Claim never touches the shared Alloc cursor domain"
    );

    munmap(worker, sizeof(*worker));
    munmap(state, sizeof(*state));
}

void TestRejectedTaskId() {
    Check(
        !IsSharedAllocCandidate(0, 0, kTaskCellCapacity),
        "task outside the completion table has no shared Alloc owner"
    );
}

}  // namespace

int main() {
    TestPaAllocOwnerTopology();
    TestClaimRoutingAndOrdering();
    TestRejectedTaskId();
    if (g_failures != 0) {
        std::fprintf(
            stderr, "[FAIL] shared Alloc Claim cursor tests: %d\n", g_failures
        );
        return 1;
    }
    std::printf("[PASS] shared Alloc Claim cursor tests\n");
    return 0;
}
