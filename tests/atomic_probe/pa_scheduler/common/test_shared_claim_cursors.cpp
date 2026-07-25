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

// 只实现 Claim/atomic trace 会触及的最小 Ops 接口，同时记录 FetchMax
// 地址和次数，避免定向测试仅凭最终 cursor 猜测实际走过的路径。
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

void InitializeClaimCursors(SchedulerState *state) {
    for (uint32_t shard = 0; shard < kCursorShards; ++shard) {
        state->cube_cursor[shard].value = -1;
        state->vector_cursor[shard].value = -1;
        state->alloc_cursor[shard].value = -1;
    }
    for (uint32_t shard = 0; shard < kSharedVectorCursorCapacity; ++shard) {
        state->shared_map.shared_vector_cursor[shard].value = -1;
    }
    for (uint32_t shard = 0; shard < kSharedCubeCursorCapacity; ++shard) {
        state->shared_map.shared_cube_cursor[shard].value = -1;
    }
}

bool PrefixClaimCursorsUntouched(const SchedulerState &state) {
    for (uint32_t shard = 0; shard < kCursorShards; ++shard) {
        if (state.cube_cursor[shard].value != -1 ||
            state.vector_cursor[shard].value != -1) {
            return false;
        }
    }
    return true;
}

void TestSharedClaimRouting() {
    SchedulerState *state = MapSparseObject<SchedulerState>();
    WorkerState *worker = MapSparseObject<WorkerState>();
    Check(
        state != nullptr && worker != nullptr,
        "sparse shared Claim cursor fixtures map successfully"
    );
    if (state == nullptr || worker == nullptr) {
        if (state != nullptr) munmap(state, sizeof(*state));
        if (worker != nullptr) munmap(worker, sizeof(*worker));
        return;
    }

    InitializeClaimCursors(state);
    worker->role = CoreRole::Aiv;
    LocalStats stats{};
    ClaimTestOps::ResetFetchMaxTrace();

    const ClaimOutcome sf = Claim<ClaimTestOps>(
        state, *worker, 2, TaskKind::Sf, stats
    );
    Check(
        sf.attempted && sf.won &&
            state->shared_map.shared_vector_cursor[2].value == 2,
        "SF Claim advances the active-shard modulo sidecar Vector cursor"
    );
    Check(
        ClaimTestOps::fetch_max_calls == 1 &&
            ClaimTestOps::last_fetch_max_address ==
                &state->shared_map.shared_vector_cursor[2].value,
        "SF Claim issues one FetchMax to the expected sidecar address"
    );

    // S4.14b 在相同 sidecar 上把 active 4→8：task 14 改落 shard 6，
    // task 2 仍落 shard 2，正好锁定新增的取模范围。
    const ClaimOutcome up = Claim<ClaimTestOps>(
        state, *worker, 14, TaskKind::Up, stats
    );
    Check(
        up.attempted && up.won &&
            state->shared_map.shared_vector_cursor[2].value == 2 &&
            state->shared_map.shared_vector_cursor[6].value == 14,
        "active eight-shard routing separates task 2 and task 14"
    );

    const ClaimOutcome same_shard = Claim<ClaimTestOps>(
        state, *worker, 34, TaskKind::Up, stats
    );
    const ClaimOutcome replay = Claim<ClaimTestOps>(
        state, *worker, 34, TaskKind::Up, stats
    );
    Check(
        same_shard.attempted && same_shard.won &&
            replay.attempted && !replay.won &&
            state->shared_map.shared_vector_cursor[2].value == 34,
        "same-shard task advances once and a repeated candidate becomes loser"
    );
    Check(
        PrefixClaimCursorsUntouched(*state),
        "shared Cube/Vector Claim never touches production-prefix cursors"
    );

    worker->role = CoreRole::Aic;
    const uint64_t calls_before_wrong_role = ClaimTestOps::fetch_max_calls;
    const ClaimOutcome wrong_role = Claim<ClaimTestOps>(
        state, *worker, 42, TaskKind::Sf, stats
    );
    Check(
        !wrong_role.attempted && !wrong_role.won &&
            ClaimTestOps::fetch_max_calls == calls_before_wrong_role,
        "AIC worker neither issues atomicMax nor changes a Vector cursor"
    );

    const ClaimOutcome qk = Claim<ClaimTestOps>(
        state, *worker, 1, TaskKind::Qk, stats
    );
    const ClaimOutcome pv = Claim<ClaimTestOps>(
        state, *worker, 13, TaskKind::Pv, stats
    );
    const ClaimOutcome alloc = Claim<ClaimTestOps>(
        state, *worker, 0, TaskKind::Alloc, stats
    );
    Check(
        qk.attempted && qk.won && pv.attempted && pv.won &&
            state->shared_map.shared_cube_cursor[1].value == 13 &&
            state->shared_map.shared_cube_cursor[5].value == -1,
        "shared Cube uses active four-shard routing inside an eight-line sidecar"
    );
    Check(
        alloc.attempted && alloc.won && state->alloc_cursor[0].value == 0,
        "shared Alloc Claim retains the production-prefix four-shard cursor"
    );
    Check(
        PrefixClaimCursorsUntouched(*state),
        "shared Cube/Vector routing leaves both production-prefix cursors untouched"
    );

    worker->role = CoreRole::Aiv;
    const uint64_t calls_before_wrong_cube_role =
        ClaimTestOps::fetch_max_calls;
    const ClaimOutcome wrong_cube_role = Claim<ClaimTestOps>(
        state, *worker, 21, TaskKind::Qk, stats
    );
    Check(
        !wrong_cube_role.attempted && !wrong_cube_role.won &&
            ClaimTestOps::fetch_max_calls == calls_before_wrong_cube_role,
        "AIV worker neither issues atomicMax nor changes a Cube cursor"
    );

    munmap(worker, sizeof(*worker));
    munmap(state, sizeof(*state));
}

void TestB256ClaimFamily(bool vector_family) {
    static_assert(
        kSharedVectorCursorCapacity == kSharedCubeCursorCapacity,
        "the generic b256 Claim test expects equal physical capacities"
    );
    SchedulerState *state = MapSparseObject<SchedulerState>();
    WorkerState *worker = MapSparseObject<WorkerState>();
    Check(
        state != nullptr && worker != nullptr,
        vector_family
            ? "b256 shared Vector Claim fixtures map successfully"
            : "b256 shared Cube Claim fixtures map successfully"
    );
    if (state == nullptr || worker == nullptr) {
        if (state != nullptr) munmap(state, sizeof(*state));
        if (worker != nullptr) munmap(worker, sizeof(*worker));
        return;
    }

    InitializeClaimCursors(state);
    LocalStats stats{};
    ClaimTestOps::ResetFetchMaxTrace();
    int64_t expected[kSharedVectorCursorCapacity] = {
        -1, -1, -1, -1, -1, -1, -1, -1
    };
    uint32_t attempts_by_worker[kWorkers] = {};
    uint32_t attempts_by_shard[kSharedVectorCursorCapacity] = {};
    const uint32_t active_shards =
        vector_family ? kSharedVectorCursorShards :
                        kSharedCubeCursorShards;
    const uint32_t candidate_workers =
        vector_family ? kAivWorkers : kAicWorkers;
    uint32_t winners = 0;
    bool topology_exact = true;
    for (uint32_t task_id = 0;
         task_id < kMaxBatches * kTasksPerBatch; ++task_id) {
        const TaskKind kind =
            static_cast<TaskKind>(task_id % kTasksPerBatch);
        const bool selected =
            vector_family
                ? (kind == TaskKind::Sf || kind == TaskKind::Up)
                : (kind == TaskKind::Qk || kind == TaskKind::Pv);
        if (!selected) {
            continue;
        }
        uint32_t task_attempts = 0;
        uint32_t task_winners = 0;
        for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
            worker->role =
                worker_id < kAicWorkers ? CoreRole::Aic : CoreRole::Aiv;
            const ClaimOutcome claim =
                Claim<ClaimTestOps>(state, *worker, task_id, kind, stats);
            if (claim.attempted) {
                ++task_attempts;
                ++attempts_by_worker[worker_id];
                ++attempts_by_shard[task_id % active_shards];
                volatile int64_t *expected_address =
                    vector_family
                        ? &state->shared_map.shared_vector_cursor[
                              task_id % active_shards
                          ].value
                        : &state->shared_map.shared_cube_cursor[
                              task_id % active_shards
                          ].value;
                topology_exact &=
                    ClaimTestOps::last_fetch_max_address ==
                        expected_address;
            }
            task_winners += claim.won;
        }
        topology_exact &=
            task_attempts == candidate_workers && task_winners == 1;
        winners += task_winners;
        expected[task_id % active_shards] = task_id;
    }

    for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
        const bool eligible =
            vector_family ? worker_id >= kAicWorkers :
                            worker_id < kAicWorkers;
        topology_exact &=
            attempts_by_worker[worker_id] ==
                (eligible ? kMaxBatches * 2U : 0U);
    }
    bool exact =
        topology_exact && winners == kMaxBatches * 2U &&
        PrefixClaimCursorsUntouched(*state);
    for (uint32_t shard = 0; shard < kSharedVectorCursorCapacity; ++shard) {
        const uint32_t expected_attempts =
            shard < active_shards
                ? kMaxBatches * 2U * candidate_workers / active_shards
                : 0U;
        const int64_t actual =
            vector_family
                ? state->shared_map.shared_vector_cursor[shard].value
                : state->shared_map.shared_cube_cursor[shard].value;
        exact &=
            actual == expected[shard] &&
            attempts_by_shard[shard] == expected_attempts;

        // 本次未选择的另一族 cursor 必须保持初值，避免测试只验证
        // 高水位，却漏掉错误地同时发射第二路 atomic。
        exact &=
            vector_family
                ? state->shared_map.shared_cube_cursor[shard].value == -1
                : state->shared_map.shared_vector_cursor[shard].value == -1;
    }
    Check(
        ClaimTestOps::fetch_max_calls ==
            static_cast<uint64_t>(kMaxBatches) * 2U * candidate_workers,
        vector_family
            ? "b256 issues exactly 32768 Vector Claim atomics"
            : "b256 issues exactly 16384 Cube Claim atomics"
    );
    Check(
        exact,
        vector_family
            ? "b256 Vector preserves 64 AIV candidates, one winner, 4096 attempts per active shard, and exact high watermarks"
            : "b256 Cube preserves 32 AIC candidates, one winner, 4096 attempts per active shard, four inactive lines, and exact high watermarks"
    );

    munmap(worker, sizeof(*worker));
    munmap(state, sizeof(*state));
}

}  // namespace

int main() {
    TestSharedClaimRouting();
    TestB256ClaimFamily(true);
    TestB256ClaimFamily(false);
    if (g_failures != 0) {
        std::fprintf(
            stderr, "[FAIL] shared Claim cursor tests: %d\n",
            g_failures
        );
        return 1;
    }
    std::printf("[PASS] shared Claim cursor tests\n");
    return 0;
}
