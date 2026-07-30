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

#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <new>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define PA_COMPETE_FIRST_SPLIT_FINISH 1
#include "host_support.h"

#define PA_DEVICE inline
#define PA_GM
#define PA_BUILD_SWIMLANE 1
#define PA_TEST_SHARED_SUBMIT_HOOKS 1
#include "pa_scheduler_core.h"

namespace {

using namespace pa_scheduler;

enum class OrderedSubmitHookMode : uint32_t {
    None = 0,
    BuildOverlap = 1,
    ExecutionOverlap = 2,
};

struct OrderedSubmitTestOps {
    static constexpr bool kAtomicReturnReadyObserved = false;
    static inline thread_local CompeteFirstSplitRuntimeState runtime{};

    static inline SchedulerState *observed_state = nullptr;
    static inline OrderedSubmitHookMode hook_mode =
        OrderedSubmitHookMode::None;
    static inline std::atomic<uint64_t> shared_map_accesses{0};
    static inline std::atomic<uint32_t> task4_insert_hook_calls{0};
    static inline std::atomic<uint32_t> task8_build_hook_calls{0};
    static inline std::atomic<uint32_t> independent_insert_hook_calls{0};
    static inline std::atomic<bool> task4_waiting_after_insert{false};
    static inline std::atomic<bool> task8_built{false};
    static inline std::atomic<bool> task8_built_before_task4_completion{false};
    static inline std::atomic<bool> task6_executed_before_task4_build{false};
    static inline std::atomic<bool> hook_timed_out{false};
    static inline std::atomic<uint32_t>
        claim_attempts_by_task[kMaxTasks]{};
    static inline std::atomic<uint32_t>
        claim_wins_by_task[kMaxTasks]{};
    static inline std::atomic<uint32_t>
        completion_loads_by_cell[kMaxTasks]{};
    static inline std::atomic<uint32_t>
        completion_cas_by_task[kMaxTasks]{};
    static inline std::atomic<uint32_t>
        completion_publish_by_task[kMaxTasks]{};
    static inline std::atomic<uint32_t>
        bad_completion_cas{0};
    static inline std::atomic<uint32_t>
        legacy_turn_atomic_accesses{0};

    static void ResetHooks() {
        observed_state = nullptr;
        hook_mode = OrderedSubmitHookMode::None;
        shared_map_accesses.store(0, std::memory_order_relaxed);
        task4_insert_hook_calls.store(0, std::memory_order_relaxed);
        task8_build_hook_calls.store(0, std::memory_order_relaxed);
        independent_insert_hook_calls.store(0, std::memory_order_relaxed);
        task4_waiting_after_insert.store(false, std::memory_order_relaxed);
        task8_built.store(false, std::memory_order_relaxed);
        task8_built_before_task4_completion.store(false, std::memory_order_relaxed);
        task6_executed_before_task4_build.store(false, std::memory_order_relaxed);
        hook_timed_out.store(false, std::memory_order_relaxed);
        for (uint32_t task = 0; task < kMaxTasks; ++task) {
            claim_attempts_by_task[task].store(
                0, std::memory_order_relaxed
            );
            claim_wins_by_task[task].store(
                0, std::memory_order_relaxed
            );
            completion_loads_by_cell[task].store(
                0, std::memory_order_relaxed
            );
            completion_cas_by_task[task].store(
                0, std::memory_order_relaxed
            );
            completion_publish_by_task[task].store(
                0, std::memory_order_relaxed
            );
        }
        bad_completion_cas.store(
            0, std::memory_order_relaxed
        );
        legacy_turn_atomic_accesses.store(
            0, std::memory_order_relaxed
        );
    }

    static CompeteFirstSplitRuntimeState &
    CompeteFirstSplitState() {
        return runtime;
    }

    static bool FinishCompeteFirstCallback(
        const CallbackSubmitTicket *ticket,
        const TaskArgs *args
    ) {
        return FinishSplitCallbackSubmitFromRuntime<
                   OrderedSubmitTestOps
               >(ticket, args) == 1U;
    }

    static void CountSharedMapAccess(
        const volatile void *address, uint64_t bytes
    ) {
        if (observed_state == nullptr) {
            return;
        }
        const uintptr_t current =
            reinterpret_cast<uintptr_t>(address);
        const uintptr_t access_end = current + bytes;
        const uintptr_t begin = reinterpret_cast<uintptr_t>(
            &observed_state->shared_map
        );
        const uintptr_t end =
            begin + sizeof(SharedTensorMapSidecar);
        if (current < end && access_end > begin) {
            shared_map_accesses.fetch_add(
                1, std::memory_order_relaxed
            );
        }
    }

    static int32_t Load(volatile int32_t *address) {
        CountSharedMapAccess(address, sizeof(*address));
        return __atomic_fetch_add(address, static_cast<int32_t>(0), __ATOMIC_ACQUIRE);
    }

    static int64_t Load(volatile int64_t *address) {
        CountSharedMapAccess(address, sizeof(*address));
        const int32_t task_cell =
            CompletionTaskCell(address);
        if (task_cell >= 0) {
            completion_loads_by_cell[
                static_cast<uint32_t>(task_cell)
            ].fetch_add(1, std::memory_order_relaxed);
        }
        if (IsLegacyTurnAddress(address)) {
            legacy_turn_atomic_accesses.fetch_add(
                1, std::memory_order_relaxed
            );
        }
        return __atomic_fetch_add(address, static_cast<int64_t>(0), __ATOMIC_ACQUIRE);
    }

    static uint64_t Load(volatile uint64_t *address) {
        CountSharedMapAccess(address, sizeof(*address));
        return __atomic_fetch_add(address, static_cast<uint64_t>(0), __ATOMIC_ACQUIRE);
    }

    static int32_t Exchange(volatile int32_t *address, int32_t value) {
        CountSharedMapAccess(address, sizeof(*address));
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t Exchange(volatile int64_t *address, int64_t value) {
        CountSharedMapAccess(address, sizeof(*address));
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static uint64_t Exchange(volatile uint64_t *address, uint64_t value) {
        CountSharedMapAccess(address, sizeof(*address));
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t CompareExchange(
        volatile int64_t *address, int64_t expected, int64_t desired
    ) {
        CountSharedMapAccess(address, sizeof(*address));
        const int32_t completion_task =
            CompletionTaskCell(address);
        if (completion_task >= 0) {
            const uint32_t task =
                static_cast<uint32_t>(completion_task);
            completion_cas_by_task[task].fetch_add(
                1, std::memory_order_relaxed
            );
            if (expected != -1 ||
                desired != completion_task) {
                bad_completion_cas.fetch_add(
                    1, std::memory_order_relaxed
                );
            }
        }
        if (IsLegacyTurnAddress(address)) {
            legacy_turn_atomic_accesses.fetch_add(
                1, std::memory_order_relaxed
            );
        }
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(
            address, &observed, desired, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        if (completion_task >= 0 &&
            observed == expected && expected == -1 &&
            desired == completion_task) {
            completion_publish_by_task[
                static_cast<uint32_t>(completion_task)
            ].fetch_add(1, std::memory_order_relaxed);
        }
        return observed;
    }

    static int64_t FetchAdd(volatile int64_t *address, int64_t value) {
        CountSharedMapAccess(address, sizeof(*address));
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t FetchMax(
        volatile int64_t *address, int64_t value, uint64_t &retries
    ) {
        CountSharedMapAccess(address, sizeof(*address));
        const bool valid_task =
            value >= 0 &&
            value < static_cast<int64_t>(kMaxTasks);
        const bool claim_address =
            observed_state != nullptr && valid_task &&
            IsClaimCursorAddress(address);
        if (claim_address) {
            claim_attempts_by_task[
                static_cast<uint32_t>(value)
            ].fetch_add(1, std::memory_order_relaxed);
        }
        int64_t current = __atomic_load_n(address, __ATOMIC_ACQUIRE);
        retries = 0;
        bool won = false;
        while (value > current) {
            if (__atomic_compare_exchange_n(
                    address, &current, value, true,
                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
                )) {
                won = true;
                break;
            }
            ++retries;
        }
        if (claim_address && won) {
            claim_wins_by_task[
                static_cast<uint32_t>(value)
            ].fetch_add(1, std::memory_order_relaxed);
        }
        return current;
    }

    static bool IsLegacyTurnAddress(
        const volatile int64_t *address
    ) {
        if (observed_state == nullptr) {
            return false;
        }
        for (uint32_t lane = 0;
             lane < kSharedInsertTurnCapacity; ++lane) {
            if (address ==
                &SharedInsertTurnLine(
                    observed_state->shared_map, lane
                ).value) {
                return true;
            }
        }
        return false;
    }

    static int32_t CompletionTaskCell(
        const volatile int64_t *address
    ) {
        if (observed_state == nullptr) {
            return -1;
        }
        const uintptr_t current =
            reinterpret_cast<uintptr_t>(address);
        const uintptr_t begin =
            reinterpret_cast<uintptr_t>(
                &observed_state->tasks[0].deps_prepared
            );
        if (current < begin) {
            return -1;
        }
        const uintptr_t delta = current - begin;
        if (delta % sizeof(TaskCell) != 0) {
            return -1;
        }
        const uintptr_t task = delta / sizeof(TaskCell);
        return task < kMaxTasks
            ? static_cast<int32_t>(task)
            : -1;
    }

    static bool IsClaimCursorAddress(
        const volatile int64_t *address
    ) {
        if (observed_state == nullptr) {
            return false;
        }
        for (uint32_t shard = 0; shard < kCursorShards;
             ++shard) {
            if (address ==
                    &observed_state
                         ->cube_cursor[shard].value ||
                address ==
                    &observed_state
                         ->alloc_cursor[shard].value) {
                return true;
            }
        }
        for (uint32_t shard = 0;
             shard < kSharedVectorCursorShards; ++shard) {
            if (address ==
                &observed_state->shared_map
                     .shared_vector_cursor[shard].value) {
                return true;
            }
        }
        return false;
    }

    static void StoreBarrier() {
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }

    static uint64_t Now() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count()
        );
    }

    template <typename T>
    static uint64_t NowAfterAtomicResult(T value) {
        asm volatile("" : "+r"(value));
        return Now();
    }

    static void SpinHint() {
        std::this_thread::yield();
    }

    static void PreloadDataCache(void *) {
        // CPU 定向测试不模拟 A5 DCache hint。
    }

    static void InvalidateRegion(
        const void *address, uint64_t bytes
    ) {
        CountSharedMapAccess(address, bytes);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }

    static void FlushRegion(void *address, uint64_t bytes) {
        CountSharedMapAccess(address, bytes);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }

    static void Publish(uint64_t *address, uint64_t value) {
        CountSharedMapAccess(address, sizeof(*address));
        __atomic_store_n(address, value, __ATOMIC_RELEASE);
    }

    static bool PmuWindowStart(SchedulerState *, uint32_t) {
        return false;
    }

    static void PmuWindowStop(SchedulerState *, uint32_t, bool) {}

    static void ExecuteKernel(
        SchedulerState *, WorkerState &, TaskKind, uint32_t
    ) {}

    static void AfterSharedTaskInsert(
        SchedulerState *state, WorkerState &, uint32_t task_id
    ) {
        constexpr uint32_t kPausedUp = 4;
        if (hook_mode == OrderedSubmitHookMode::BuildOverlap) {
            if (task_id != kPausedUp) {
                return;
            }
            task4_insert_hook_calls.fetch_add(1, std::memory_order_relaxed);
            task4_waiting_after_insert.store(true, std::memory_order_release);
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!task8_built.load(std::memory_order_acquire)) {
                if (state->fatal.value != 0 ||
                    std::chrono::steady_clock::now() >= deadline) {
                    hook_timed_out.store(true, std::memory_order_relaxed);
                    break;
                }
                std::this_thread::yield();
            }
            return;
        }

        // B2/G1 中 batch0 UP=task4，batch1 QK=task6。task4 发布插入
        // 前沿后暂停在 fanin/Build 之前；task6 只依赖 batch1 Alloc，
        // 因此它的 completion flag 必须能在 task4 仍为 0 时变成 1。
        constexpr uint32_t kPausedBatch0Up = 4;
        constexpr uint32_t kIndependentBatch1Qk = 6;
        if (hook_mode != OrderedSubmitHookMode::ExecutionOverlap ||
            task_id != kPausedBatch0Up) {
            return;
        }
        independent_insert_hook_calls.fetch_add(
            1, std::memory_order_relaxed
        );
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (Load(&state->tasks[kIndependentBatch1Qk].flag) == 0) {
            if (state->fatal.value != 0 ||
                std::chrono::steady_clock::now() >= deadline) {
                hook_timed_out.store(true, std::memory_order_relaxed);
                return;
            }
            std::this_thread::yield();
        }
        task6_executed_before_task4_build.store(
            Load(&state->tasks[kPausedBatch0Up].flag) == 0,
            std::memory_order_release
        );
    }

    static void AfterSharedTaskBuild(
        SchedulerState *state, WorkerState &, uint32_t task_id, TaskKind kind
    ) {
        constexpr uint32_t kPausedUp = 4;
        constexpr uint32_t kFollowingUp = 8;
        if (task_id != kFollowingUp || kind != TaskKind::Up) {
            return;
        }
        task8_build_hook_calls.fetch_add(1, std::memory_order_relaxed);
        const bool overlap =
            task4_waiting_after_insert.load(std::memory_order_acquire) &&
            Load(&state->tasks[kPausedUp].flag) == 0;
        task8_built_before_task4_completion.store(overlap, std::memory_order_release);
        task8_built.store(true, std::memory_order_release);
    }
};

SchedulerState *MapSchedulerState() {
    void *memory = mmap(
        nullptr, sizeof(SchedulerState), PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0
    );
    if (memory == MAP_FAILED) {
        return nullptr;
    }
    return new (memory) SchedulerState{};
}

using LegacyTurnSnapshot =
    std::array<int64_t, kSharedInsertTurnCapacity>;

LegacyTurnSnapshot SeedLegacyTurns(
    SchedulerState &state
) {
    LegacyTurnSnapshot snapshot{};
    for (uint32_t lane = 0;
         lane < kSharedInsertTurnCapacity; ++lane) {
        snapshot[lane] =
            -10000 - static_cast<int64_t>(lane);
        __atomic_store_n(
            &SharedInsertTurnLine(
                state.shared_map, lane
            ).value,
            snapshot[lane], __ATOMIC_RELEASE
        );
    }
    return snapshot;
}

bool LegacyTurnsMatch(
    const SchedulerState &state,
    const LegacyTurnSnapshot &snapshot
) {
    for (uint32_t lane = 0;
         lane < kSharedInsertTurnCapacity; ++lane) {
        if (__atomic_load_n(
                &SharedInsertTurnLine(
                    const_cast<SharedTensorMapSidecar &>(
                        state.shared_map
                    ),
                    lane
                ).value,
                __ATOMIC_ACQUIRE
            ) != snapshot[lane]) {
            return false;
        }
    }
    return true;
}

uint32_t ExpectedClaimAttempts(TaskKind kind) {
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

bool RunLocalClaimAttemptAccountingTest() {
    LocalStats zero_attempts{};
    FinalizeSharedClaimAttempts(zero_attempts, 0);
    LocalStats stats{};
    uint64_t local_attempts = 0;
    const ClaimOutcome skipped{false, false, 0, -1};
    const ClaimOutcome loser{true, false, 3, -1};
    const ClaimOutcome winner{
        true, true, 2, FunctionId(TaskKind::Qk)
    };

    RecordClaimOutcome(
        stats, TaskKind::Alloc, skipped, local_attempts
    );
    const bool skipped_not_counted =
        local_attempts == 0 &&
        stats.result.claim_attempts == 0 &&
        stats.result.cas_retries == 0 &&
        stats.result.claim_wins == 0;
    RecordClaimOutcome(
        stats, TaskKind::Qk, loser, local_attempts
    );
    const bool loser_counted =
        local_attempts == 1 &&
        stats.result.claim_attempts == 0 &&
        stats.result.cas_retries == 3 &&
        stats.result.claim_wins == 0;
    RecordClaimOutcome(
        stats, TaskKind::Qk, winner, local_attempts
    );

    const bool before_publish =
        skipped_not_counted && loser_counted &&
        local_attempts == 2 &&
        stats.result.claim_attempts == 0 &&
        stats.result.cas_retries == 5 &&
        stats.result.claim_wins == 1 &&
        stats.result.wins[KindIndex(TaskKind::Qk)] == 1;
    FinalizeSharedClaimAttempts(stats, local_attempts);
    const bool ok =
        zero_attempts.result.claim_attempts == 0 &&
        before_publish && stats.result.claim_attempts == 2;
    std::printf(
        "[ORDERED_SUBMIT] local_claim_attempt_accounting=%s "
        "attempts=%llu retries=%llu wins=%llu\n",
        ok ? "PASS" : "FAIL",
        static_cast<unsigned long long>(
            stats.result.claim_attempts
        ),
        static_cast<unsigned long long>(
            stats.result.cas_retries
        ),
        static_cast<unsigned long long>(
            stats.result.claim_wins
        )
    );
    return ok;
}

bool RunSplitReplayTaskIdPrefixTest() {
    CompeteFirstSplitRuntimeState runtime{};
    runtime.task_id_sum = 99;

    runtime.stats.result.submits = 0;
    const bool task0_ok =
        RecordSharedSplitReplayTask(runtime, 0) &&
        runtime.task_id_sum == 0;

    runtime.stats.result.submits = 1;
    const bool task1_ok =
        RecordSharedSplitReplayTask(runtime, 1) &&
        runtime.task_id_sum == 1;

    runtime.stats.result.submits = 2;
    const bool task2_ok =
        RecordSharedSplitReplayTask(runtime, 2) &&
        runtime.task_id_sum == 3;

    runtime.stats.result.submits = 3;
    runtime.reserved = 7;
    const bool reserved_rejected =
        !RecordSharedSplitReplayTask(runtime, 3) &&
        runtime.task_id_sum == 3;

    runtime.reserved = 0;
    runtime.stats.result.submits = 4;
    const bool skipped_task_rejected =
        !RecordSharedSplitReplayTask(runtime, 3) &&
        runtime.task_id_sum == 3;

    runtime.stats.result.submits = 3;
    const bool task3_ok =
        RecordSharedSplitReplayTask(runtime, 3) &&
        runtime.task_id_sum == 6;

    runtime.stats.result.submits = kMaxTasks - 1U;
    const bool max_task_ok =
        RecordSharedSplitReplayTask(runtime, kMaxTasks - 1U) &&
        runtime.task_id_sum ==
            static_cast<uint64_t>(kMaxTasks - 1U) *
                kMaxTasks / 2U;

    const bool ok =
        task0_ok && task1_ok && task2_ok &&
        reserved_rejected && skipped_task_rejected &&
        task3_ok && max_task_ok;
    std::printf(
        "[ORDERED_SUBMIT] split_task_id_prefix=%s sum=%llu\n",
        ok ? "PASS" : "FAIL",
        static_cast<unsigned long long>(runtime.task_id_sum)
    );
    return ok;
}

bool ClaimAndInsertEvidenceMatches(
    const SchedulerState &state, uint32_t task_count
) {
    uint32_t planned_tasks = 0;
    bool exact = true;
    for (uint32_t batch = 0;
         batch < state.config.batches; ++batch) {
        SharedPaBatchPlan plan{};
        exact &= BuildSharedPaBatchPlan(
            static_cast<uint64_t>(
                state.context_lens[batch]
            ),
            planned_tasks, plan
        );
        if (!exact) {
            return false;
        }
        for (uint32_t offset = 0;
             offset < plan.task_count; ++offset) {
            SharedPaPlannedTask task{};
            exact &= SharedPaPlannedTaskAt(
                plan, offset, task
            );
            const uint32_t task_id =
                plan.batch_start + offset;
            exact &=
                OrderedSubmitTestOps::
                    claim_attempts_by_task[task_id].load(
                        std::memory_order_relaxed
                    ) == ExpectedClaimAttempts(task.kind);
            exact &=
                OrderedSubmitTestOps::
                    claim_wins_by_task[task_id].load(
                        std::memory_order_relaxed
                    ) == 1;
            exact &=
                state.tasks[task_id].deps_prepared ==
                static_cast<int64_t>(task_id);
            exact &=
                OrderedSubmitTestOps::
                    completion_cas_by_task[task_id]
                        .load(std::memory_order_relaxed) == 1;
            exact &=
                OrderedSubmitTestOps::
                    completion_publish_by_task[task_id]
                        .load(std::memory_order_relaxed) == 1;
            const uint32_t completion_loads =
                OrderedSubmitTestOps::
                    completion_loads_by_cell[task_id]
                        .load(std::memory_order_relaxed);
            exact &= task_id + 1U < task_count
                ? completion_loads != 0
                : completion_loads == 0;
        }
        planned_tasks += plan.task_count;
    }
    exact &= planned_tasks == task_count;
    for (uint32_t task = task_count;
         task < kMaxTasks; ++task) {
        exact &=
            OrderedSubmitTestOps::
                completion_cas_by_task[task].load(
                    std::memory_order_relaxed
                ) == 0 &&
            OrderedSubmitTestOps::
                completion_publish_by_task[task].load(
                    std::memory_order_relaxed
                ) == 0 &&
            OrderedSubmitTestOps::
                completion_loads_by_cell[task].load(
                    std::memory_order_relaxed
                ) == 0;
    }
    return exact &&
           OrderedSubmitTestOps::bad_completion_cas.load(
               std::memory_order_relaxed
           ) == 0 &&
           OrderedSubmitTestOps::
                   legacy_turn_atomic_accesses.load(
                       std::memory_order_relaxed
                   ) == 0;
}

bool RunLoserZeroTensorMapAccessTest() {
    SchedulerState *state = MapSchedulerState();
    if (state == nullptr) {
        return false;
    }
    pa_scheduler::host::Options options;
    options.batches = 1;
    options.shared_context_lens = {16384};
    options.trace_enabled = false;
    pa_scheduler::host::InitializeState(state, options);
    pa_scheduler::host::ConfigureTrace(state, options, nullptr);

    constexpr uint32_t kTask = 2;
    WorkerState &worker = state->workers[0];
    worker.local_index = kTask;
    SubmitContext context{};
    // 模拟同一 worker 上一 Submit 曾经获胜：Begin 必须把所有 actor
    // 的 task_id 更新为本次值；loser 不会重置或消费上一 winner
    // 留下的 won/kernel_id。
    context.task_id = 91;
    context.won = true;
    context.kernel_id = 17;
    BeginSharedCallbackSubmit(worker, context);
    const uint32_t task_id =
        static_cast<uint32_t>(context.task_id);
    const bool outputs_prepared =
        PrepareSharedTaskOutputs(
            context.shared_result,
            static_cast<int32_t>(task_id),
            TaskKind::Sf
        );
    // Begin 仍按所有 actor 的原合同写入当前 task id；在调用 loser
    // 收尾前重新放入毒值，证明该 helper 只依赖显式 task_id 和
    // 前序 Prepare 成功合同，不会再次从 context 取身份或数量。
    context.task_id = 91;
    LocalStats stats{};
    constexpr uint64_t kSubmitBegin = 1;

    int64_t turns_before[kSharedInsertTurnCapacity] = {};
    for (uint32_t lane = 0;
         lane < kSharedInsertTurnCapacity; ++lane) {
        volatile int64_t *address =
            &SharedInsertTurnLine(
                state->shared_map, lane
            ).value;
        turns_before[lane] =
            static_cast<int64_t>(77U + lane);
        __atomic_store_n(
            address, turns_before[lane],
            __ATOMIC_RELEASE
        );
    }
    OrderedSubmitTestOps::ResetHooks();
    OrderedSubmitTestOps::observed_state = state;
    const bool finished =
        FinishSharedLoserSubmit<
            TaskKind::Sf, OrderedSubmitTestOps, false
        >(
            state, context, stats, task_id, false,
            kSubmitBegin
        );
    bool turns_unchanged = true;
    for (uint32_t lane = 0;
         lane < kSharedInsertTurnCapacity; ++lane) {
        turns_unchanged &=
            __atomic_load_n(
                &SharedInsertTurnLine(
                    state->shared_map, lane
                ).value,
                __ATOMIC_ACQUIRE
            ) == turns_before[lane];
    }
    const FdwicOutputRef output0 =
        context.shared_result.OutputRef(0);
    const FdwicOutputRef output1 =
        context.shared_result.OutputRef(1);
    const FdwicOutputRef output2 =
        context.shared_result.OutputRef(2);

    SharedTaskOutputs wrong_task{};
    wrong_task.Reset(static_cast<int32_t>(kTask + 1U));
    const bool wrong_task_rejected =
        !PrepareSharedTaskOutputs(
            wrong_task, static_cast<int32_t>(kTask),
            TaskKind::Sf
        );
    SharedTaskOutputs nonempty{};
    nonempty.Reset(static_cast<int32_t>(kTask));
    const bool seeded =
        nonempty.AddOutputRef(
            static_cast<int32_t>(kTask), 0
        );
    const bool nonempty_rejected =
        !PrepareSharedTaskOutputs(
            nonempty, static_cast<int32_t>(kTask),
            TaskKind::Sf
        );

    WorkerState up_worker{};
    up_worker.local_index = kTask;
    SubmitContext up_context{};
    up_context.shared_result.Reset(91);
    BeginSharedCallbackSubmit(up_worker, up_context);
    const bool up_not_attempted_prepared =
        PrepareSharedTaskOutputsAfterClaim<TaskKind::Up>(
            up_context.shared_result,
            static_cast<int32_t>(kTask), false
        );
    const bool up_not_attempted_keeps_reset_state =
        up_not_attempted_prepared &&
        up_context.shared_result.TaskId() ==
            static_cast<int32_t>(kTask) &&
        up_context.shared_result.Empty();

    SharedTaskOutputs poisoned_up{};
    poisoned_up.Reset(static_cast<int32_t>(kTask + 1U));
    const bool up_participant_rejects_wrong_task =
        !PrepareSharedTaskOutputsAfterClaim<TaskKind::Up>(
            poisoned_up, static_cast<int32_t>(kTask), true
        );
    poisoned_up.Reset(static_cast<int32_t>(kTask));
    const bool poisoned_up_seeded =
        poisoned_up.AddOutputRef(
            static_cast<int32_t>(kTask), 0
        );
    const bool up_participant_rejects_nonempty =
        !PrepareSharedTaskOutputsAfterClaim<TaskKind::Up>(
            poisoned_up, static_cast<int32_t>(kTask), true
        );

    const bool ok =
        outputs_prepared && finished &&
        state->fatal.value == 0 &&
        task_id == kTask &&
        worker.local_index == kTask + 1U &&
        context.task_id == 91 &&
        context.shared_result.TaskId() ==
            static_cast<int32_t>(kTask) &&
        context.shared_result.Size() == 3 &&
        output0.producer_task_id ==
            static_cast<int32_t>(kTask) &&
        output0.output_slot == 0 &&
        output1.producer_task_id ==
            static_cast<int32_t>(kTask) &&
        output1.output_slot == 1 &&
        output2.producer_task_id ==
            static_cast<int32_t>(kTask) &&
        output2.output_slot == 2 &&
        wrong_task_rejected && seeded &&
        nonempty_rejected &&
        up_not_attempted_keeps_reset_state &&
        up_participant_rejects_wrong_task &&
        poisoned_up_seeded &&
        up_participant_rejects_nonempty &&
        stats.result.submits == 1 &&
        stats.declared_task_count == 0 &&
        turns_unchanged &&
        state->tasks[kTask].deps_prepared == -1 &&
        OrderedSubmitTestOps::shared_map_accesses.load(
            std::memory_order_relaxed
        ) == 0;
    std::printf(
        "[ORDERED_SUBMIT] loser_zero_map_access=%s accesses=%llu\n",
        ok ? "PASS" : "FAIL",
        static_cast<unsigned long long>(
            OrderedSubmitTestOps::shared_map_accesses.load(
                std::memory_order_relaxed
            )
        )
    );
    OrderedSubmitTestOps::observed_state = nullptr;
    (void)munmap(state, sizeof(SchedulerState));
    return ok;
}

bool RunReadyFaninPrefixCompactionTest() {
    SchedulerState *state = MapSchedulerState();
    if (state == nullptr) {
        return false;
    }
    pa_scheduler::host::Options options;
    options.batches = 1;
    options.shared_context_lens = {8192};
    options.trace_enabled = false;
    pa_scheduler::host::InitializeState(state, options);
    pa_scheduler::host::ConfigureTrace(state, options, nullptr);

    LocalSlot blocked_at_front{};
    blocked_at_front.fanin_count = 2;
    blocked_at_front.fanin[0] = 4;
    blocked_at_front.fanin[1] = 5;
    state->tasks[4].flag = 0;
    state->tasks[5].flag = 1;
    LocalStats front_stats{};
    const bool front_ready = SlotReady<OrderedSubmitTestOps>(
        state, blocked_at_front, front_stats
    );
    const bool front_unchanged =
        !front_ready && blocked_at_front.fanin_count == 2 &&
        blocked_at_front.fanin[0] == 4 &&
        blocked_at_front.fanin[1] == 5 &&
        front_stats.result.fanin_ready_loads == 0 &&
        front_stats.result.fanin_not_ready_loads == 1;

    LocalSlot slot{};
    slot.fanin_count = 4;
    slot.fanin[0] = 0;
    slot.fanin[1] = 1;
    slot.fanin[2] = 2;
    slot.fanin[3] = 3;
    state->tasks[0].flag = 1;
    state->tasks[1].flag = 0;
    state->tasks[2].flag = 0;
    state->tasks[3].flag = 0;
    LocalStats stats{};
    const bool stage1 =
        !SlotReady<OrderedSubmitTestOps>(state, slot, stats) &&
        slot.fanin_count == 3 &&
        slot.fanin[0] == 1 && slot.fanin[1] == 2 &&
        slot.fanin[2] == 3;
    state->tasks[1].flag = 1;
    const bool stage2 =
        !SlotReady<OrderedSubmitTestOps>(state, slot, stats) &&
        slot.fanin_count == 2 &&
        slot.fanin[0] == 2 && slot.fanin[1] == 3;
    state->tasks[2].flag = 1;
    const bool stage3 =
        !SlotReady<OrderedSubmitTestOps>(state, slot, stats) &&
        slot.fanin_count == 1 && slot.fanin[0] == 3;
    state->tasks[3].flag = 1;
    const bool final_ready =
        SlotReady<OrderedSubmitTestOps>(state, slot, stats);
    const bool ok =
        front_unchanged && stage1 && stage2 && stage3 &&
        final_ready && slot.fanin_count == 0 &&
        stats.result.fanin_ready_loads == 4 &&
        stats.result.fanin_not_ready_loads == 3;
    std::printf(
        "[ORDERED_SUBMIT] ready_fanin_prefix_compaction=%s\n",
        ok ? "PASS" : "FAIL"
    );
    (void)munmap(state, sizeof(SchedulerState));
    return ok;
}

bool RunPaUpWriterShapeContractTest() {
    SchedulerState *state = MapSchedulerState();
    if (state == nullptr) {
        return false;
    }
    pa_scheduler::host::Options options;
    options.batches = 1;
    options.shared_context_lens = {8192};
    options.trace_enabled = false;
    pa_scheduler::host::InitializeState(state, options);
    pa_scheduler::host::ConfigureTrace(state, options, nullptr);

    constexpr int32_t kTask = 4;
    SubmitContext context{};
    context.task_id = kTask;
    context.won = true;
    SharedTaskWriterDelta empty_delta{};
    empty_delta.prepared_task_id = kTask;
    LocalStats stats{};

    // 非 UP 的空 writer 集合合法；同一空集合若由非负 previous 标明
    // 当前 task 是 UP，则必须在发布 history/deps_prepared 前拒绝。
    const bool non_up_empty_ok =
        PublishSharedTaskWriterMetadata<
            OrderedSubmitTestOps, false, false, true, true
        >(
            state, context, empty_delta, stats,
            /*expected_previous=*/-1,
            /*expected_producer=*/0
        );
    const bool up_empty_rejected =
        !PublishSharedTaskWriterMetadata<
            OrderedSubmitTestOps, false, false, true, true
        >(
            state, context, empty_delta, stats,
            /*expected_previous=*/0,
            /*expected_producer=*/0
        );
    const bool ok =
        non_up_empty_ok && up_empty_rejected &&
        state->fatal.value == 1 &&
        state->shared_map.writer_history[kTask].magic == 0 &&
        state->tasks[kTask].deps_prepared == -1;
    std::printf(
        "[ORDERED_SUBMIT] pa_up_writer_shape_contract=%s\n",
        ok ? "PASS" : "FAIL"
    );
    (void)munmap(state, sizeof(SchedulerState));
    return ok;
}

bool RunInsertReleaseBeforeBuildTest() {
    SchedulerState *state = MapSchedulerState();
    if (state == nullptr) {
        return false;
    }
    pa_scheduler::host::Options options;
    options.batches = 1;
    options.runs = 1;
    options.trace_enabled = false;
    options.shared_context_lens = {kSharedPaMaxContextLength};
    options.final_barrier_shape = FinalBarrierShape::TwoLevel16;
    pa_scheduler::host::InitializeState(state, options);
    pa_scheduler::host::ConfigureTrace(state, options, nullptr);
    const LegacyTurnSnapshot legacy =
        SeedLegacyTurns(*state);
    OrderedSubmitTestOps::ResetHooks();
    OrderedSubmitTestOps::hook_mode =
        OrderedSubmitHookMode::BuildOverlap;
    OrderedSubmitTestOps::observed_state = state;

    std::vector<std::thread> workers;
    workers.reserve(kWorkers);
    for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
        const CoreRole role =
            worker_id < kAicWorkers ? CoreRole::Aic : CoreRole::Aiv;
        workers.emplace_back([state, worker_id, role]() {
            RunScheduler<OrderedSubmitTestOps>(state, worker_id, role);
        });
    }
    for (std::thread &worker : workers) {
        worker.join();
    }

    constexpr uint32_t kTaskCount = 17;
    uint64_t kernel_counts[4] = {};
    bool worker_results_ok = true;
    for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
        const WorkerResult &result = state->results[worker_id];
        worker_results_ok &=
            result.worker_id == worker_id &&
            result.submits == kTaskCount &&
            result.finish_cycle != 0 &&
            result.final_occupied == 0 &&
            result.completion_duplicates == 0;
        for (uint32_t kind = 0; kind < 4; ++kind) {
            kernel_counts[kind] += result.kernel_counts[kind];
        }
    }

    bool all_tasks_ready = true;
    bool claim_cells_match = true;
    for (uint32_t task = 0; task < kTaskCount; ++task) {
        all_tasks_ready &= state->tasks[task].flag == 1;
        claim_cells_match &=
            state->tasks[task].deps_prepared ==
            static_cast<int64_t>(task);
    }
    // 正式 PA 将三个 lockstep accumulator 的 latest 收敛为 slot0 的
    // group word；slot1/2 保持 Alloc producer，供 generic slot-specific
    // resolver 继续遵守原合同。
    const bool final_group_writer_ok =
        state->shared_map.shared_outputs[0]
            .last_writer[0].value == 16 &&
        state->shared_map.shared_outputs[0]
            .last_writer[1].value == 0 &&
        state->shared_map.shared_outputs[0]
            .last_writer[2].value == 0;

    const bool overlap =
        OrderedSubmitTestOps::task8_built_before_task4_completion.load(
            std::memory_order_acquire
        );
    const bool ok =
        state->fatal.value == 0 &&
        LegacyTurnsMatch(*state, legacy) &&
        ClaimAndInsertEvidenceMatches(
            *state, kTaskCount
        ) &&
        OrderedSubmitTestOps::task4_insert_hook_calls.load(
            std::memory_order_relaxed
        ) == 1 &&
        OrderedSubmitTestOps::task8_build_hook_calls.load(
            std::memory_order_relaxed
        ) == 1 &&
        !OrderedSubmitTestOps::hook_timed_out.load(
            std::memory_order_relaxed
        ) &&
        overlap && worker_results_ok && all_tasks_ready &&
        claim_cells_match && final_group_writer_ok &&
        kernel_counts[0] == 4 && kernel_counts[1] == 4 &&
        kernel_counts[2] == 4 && kernel_counts[3] == 4 &&
        pa_scheduler::host::FinalBarrierStateMatches(
            state->final_barrier, options.final_barrier_shape
        );
    std::printf(
        "[ORDERED_SUBMIT] release_before_build=%s "
        "completed=%u legacy_turn0=%lld overlap=%u "
        "kernels=%llu,%llu,%llu,%llu\n",
        ok ? "PASS" : "FAIL",
        kTaskCount,
        static_cast<long long>(
            state->shared_map.committed_tasks.value
        ),
        overlap ? 1U : 0U,
        static_cast<unsigned long long>(kernel_counts[0]),
        static_cast<unsigned long long>(kernel_counts[1]),
        static_cast<unsigned long long>(kernel_counts[2]),
        static_cast<unsigned long long>(kernel_counts[3])
    );
    OrderedSubmitTestOps::observed_state = nullptr;
    (void)munmap(state, sizeof(SchedulerState));
    return ok;
}

bool RunIndependentKernelExecutionTest() {
    SchedulerState *state = MapSchedulerState();
    if (state == nullptr) {
        return false;
    }
    pa_scheduler::host::Options options;
    options.batches = 2;
    options.runs = 1;
    options.trace_enabled = false;
    options.shared_context_lens = {8192, 8192};
    options.final_barrier_shape = FinalBarrierShape::TwoLevel16;
    pa_scheduler::host::InitializeState(state, options);
    pa_scheduler::host::ConfigureTrace(state, options, nullptr);
    const LegacyTurnSnapshot legacy =
        SeedLegacyTurns(*state);
    OrderedSubmitTestOps::ResetHooks();
    OrderedSubmitTestOps::hook_mode =
        OrderedSubmitHookMode::ExecutionOverlap;
    OrderedSubmitTestOps::observed_state = state;

    std::vector<std::thread> workers;
    workers.reserve(kWorkers);
    for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
        const CoreRole role =
            worker_id < kAicWorkers ? CoreRole::Aic : CoreRole::Aiv;
        workers.emplace_back([state, worker_id, role]() {
            RunScheduler<OrderedSubmitTestOps>(
                state, worker_id, role
            );
        });
    }
    for (std::thread &worker : workers) {
        worker.join();
    }

    constexpr uint32_t kTaskCount = 10;
    uint64_t kernel_counts[4] = {};
    bool worker_results_ok = true;
    for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
        const WorkerResult &result = state->results[worker_id];
        worker_results_ok &=
            result.worker_id == worker_id &&
            result.submits == kTaskCount &&
            result.finish_cycle != 0 &&
            result.final_occupied == 0 &&
            result.completion_duplicates == 0;
        for (uint32_t kind = 0; kind < 4; ++kind) {
            kernel_counts[kind] += result.kernel_counts[kind];
        }
    }
    bool all_tasks_ready = true;
    for (uint32_t task = 0; task < kTaskCount; ++task) {
        all_tasks_ready &= state->tasks[task].flag == 1;
    }
    const bool execution_overlap =
        OrderedSubmitTestOps::task6_executed_before_task4_build.load(
            std::memory_order_acquire
        );
    const bool ok =
        state->fatal.value == 0 &&
        LegacyTurnsMatch(*state, legacy) &&
        ClaimAndInsertEvidenceMatches(
            *state, kTaskCount
        ) &&
        OrderedSubmitTestOps::independent_insert_hook_calls.load(
            std::memory_order_relaxed
        ) == 1 &&
        !OrderedSubmitTestOps::hook_timed_out.load(
            std::memory_order_relaxed
        ) &&
        execution_overlap && worker_results_ok &&
        all_tasks_ready &&
        kernel_counts[0] == 2 && kernel_counts[1] == 2 &&
        kernel_counts[2] == 2 && kernel_counts[3] == 2 &&
        pa_scheduler::host::FinalBarrierStateMatches(
            state->final_barrier, options.final_barrier_shape
        );
    std::printf(
        "[ORDERED_SUBMIT] independent_kernel_overlap=%s "
        "completed=%u legacy_turn0=%lld "
        "kernels=%llu,%llu,%llu,%llu\n",
        ok ? "PASS" : "FAIL",
        kTaskCount,
        static_cast<long long>(
            state->shared_map.committed_tasks.value
        ),
        static_cast<unsigned long long>(kernel_counts[0]),
        static_cast<unsigned long long>(kernel_counts[1]),
        static_cast<unsigned long long>(kernel_counts[2]),
        static_cast<unsigned long long>(kernel_counts[3])
    );
    OrderedSubmitTestOps::observed_state = nullptr;
    (void)munmap(state, sizeof(SchedulerState));
    return ok;
}

}  // namespace

int main() {
    const bool claim_accounting_ok =
        RunLocalClaimAttemptAccountingTest();
    const bool task_id_prefix_ok =
        RunSplitReplayTaskIdPrefixTest();
    const bool loser_ok = RunLoserZeroTensorMapAccessTest();
    const bool fanin_compaction_ok =
        RunReadyFaninPrefixCompactionTest();
    const bool pa_up_shape_ok =
        RunPaUpWriterShapeContractTest();
    const bool overlap_ok = RunInsertReleaseBeforeBuildTest();
    const bool execution_ok =
        RunIndependentKernelExecutionTest();
    if (!claim_accounting_ok || !task_id_prefix_ok || !loser_ok ||
        !fanin_compaction_ok || !pa_up_shape_ok ||
        !overlap_ok || !execution_ok) {
        std::fprintf(
            stderr, "[FAIL] shared ordered-insert Submit tests\n"
        );
        return 1;
    }
    std::printf(
        "[PASS] shared loser skips TensorMap; lookup/Build and "
        "independent kernel execution cross prior owner Build\n"
    );
    return 0;
}
