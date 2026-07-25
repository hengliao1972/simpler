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
}

bool PrefixVectorUntouched(const SchedulerState &state) {
    for (uint32_t shard = 0; shard < kCursorShards; ++shard) {
        if (state.vector_cursor[shard].value != -1) {
            return false;
        }
    }
    return true;
}

void TestSharedVectorRouting() {
    SchedulerState *state = MapSparseObject<SchedulerState>();
    WorkerState *worker = MapSparseObject<WorkerState>();
    Check(
        state != nullptr && worker != nullptr,
        "sparse shared Vector Claim fixtures map successfully"
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

    // S4.14a 是迁址对照，不改变四分片映射：task 14 与 task 2 仍同属
    // shard 2，只是地址从 production prefix 搬到固定 sidecar。
    const ClaimOutcome up = Claim<ClaimTestOps>(
        state, *worker, 14, TaskKind::Up, stats
    );
    Check(
        up.attempted && up.won &&
            state->shared_map.shared_vector_cursor[2].value == 14 &&
            state->shared_map.shared_vector_cursor[6].value == -1,
        "sidecar relocation preserves the original four-shard task mapping"
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
        PrefixVectorUntouched(*state),
        "shared Vector Claim never touches the production-prefix Vector cursor"
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
    const ClaimOutcome alloc = Claim<ClaimTestOps>(
        state, *worker, 0, TaskKind::Alloc, stats
    );
    Check(
        qk.attempted && qk.won && state->cube_cursor[1].value == 1,
        "shared Cube Claim retains the production-prefix four-shard cursor"
    );
    Check(
        alloc.attempted && alloc.won && state->alloc_cursor[0].value == 0,
        "shared Alloc Claim retains the production-prefix four-shard cursor"
    );

    munmap(worker, sizeof(*worker));
    munmap(state, sizeof(*state));
}

void TestB256VectorCursorFinalState() {
    SchedulerState *state = MapSparseObject<SchedulerState>();
    WorkerState *worker = MapSparseObject<WorkerState>();
    Check(
        state != nullptr && worker != nullptr,
        "b256 shared Vector Claim fixtures map successfully"
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
    uint32_t winners = 0;
    bool topology_exact = true;
    for (uint32_t task_id = 0;
         task_id < kMaxBatches * kTasksPerBatch; ++task_id) {
        const TaskKind kind =
            static_cast<TaskKind>(task_id % kTasksPerBatch);
        if (kind != TaskKind::Sf && kind != TaskKind::Up) {
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
                topology_exact &=
                    ClaimTestOps::last_fetch_max_address ==
                        &state->shared_map.shared_vector_cursor[
                            task_id % kSharedVectorCursorShards
                        ].value;
            }
            task_winners += claim.won;
        }
        topology_exact &=
            task_attempts == kAivWorkers && task_winners == 1;
        winners += task_winners;
        expected[task_id % kSharedVectorCursorShards] = task_id;
    }

    for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
        topology_exact &=
            attempts_by_worker[worker_id] ==
                (worker_id < kAicWorkers ? 0U : kMaxBatches * 2U);
    }
    bool exact =
        topology_exact && winners == kMaxBatches * 2U &&
        PrefixVectorUntouched(*state);
    for (uint32_t shard = 0; shard < kSharedVectorCursorCapacity; ++shard) {
        exact &=
            state->shared_map.shared_vector_cursor[shard].value ==
                expected[shard];
    }
    Check(
        ClaimTestOps::fetch_max_calls ==
            static_cast<uint64_t>(kMaxBatches) * 2U * kAivWorkers,
        "b256 issues exactly 32768 Vector Claim atomics"
    );
    Check(
        exact,
        "b256 preserves 64 AIV candidates, one winner, and exact sidecar high watermarks"
    );

    munmap(worker, sizeof(*worker));
    munmap(state, sizeof(*state));
}

}  // namespace

int main() {
    TestSharedVectorRouting();
    TestB256VectorCursorFinalState();
    if (g_failures != 0) {
        std::fprintf(
            stderr, "[FAIL] shared Vector Claim cursor tests: %d\n",
            g_failures
        );
        return 1;
    }
    std::printf("[PASS] shared Vector Claim cursor tests\n");
    return 0;
}
