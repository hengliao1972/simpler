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
        insert_publish_by_token[kMaxTasks + 1U]{};
    static inline std::atomic<uint32_t>
        insert_publish_by_lane[kSharedInsertTurnCapacity]{};

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
            insert_publish_by_token[task].store(
                0, std::memory_order_relaxed
            );
        }
        insert_publish_by_token[kMaxTasks].store(
            0, std::memory_order_relaxed
        );
        for (uint32_t lane = 0;
             lane < kSharedInsertTurnCapacity; ++lane) {
            insert_publish_by_lane[lane].store(
                0, std::memory_order_relaxed
            );
        }
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
        int32_t insert_lane = -1;
        if (observed_state != nullptr) {
            for (uint32_t lane = 0;
                 lane < kSharedInsertTurnCapacity; ++lane) {
                if (address ==
                    &SharedInsertTurnLine(
                        observed_state->shared_map, lane
                    ).value) {
                    insert_lane =
                        static_cast<int32_t>(lane);
                    break;
                }
            }
        }
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(
            address, &observed, desired, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        if (insert_lane >= 0 && observed == expected &&
            desired > 0 &&
            desired <= static_cast<int64_t>(kMaxTasks)) {
            insert_publish_by_token[
                static_cast<uint32_t>(desired)
            ].fetch_add(1, std::memory_order_relaxed);
            insert_publish_by_lane[
                static_cast<uint32_t>(insert_lane)
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
        bool claim_cursor = false;
        if (observed_state != nullptr) {
            const uintptr_t current =
                reinterpret_cast<uintptr_t>(address);
            const auto in_lines =
                [current](
                    const AtomicLine *lines,
                    uint32_t count
                ) {
                    const uintptr_t begin =
                        reinterpret_cast<uintptr_t>(lines);
                    const uintptr_t end =
                        begin +
                        static_cast<uintptr_t>(count) *
                            sizeof(AtomicLine);
                    return current >= begin && current < end &&
                           (current - begin) %
                                   sizeof(AtomicLine) ==
                               0;
                };
            claim_cursor =
                in_lines(
                    observed_state->cube_cursor,
                    kCursorShards
                ) ||
                in_lines(
                    observed_state->alloc_cursor,
                    kCursorShards
                ) ||
                in_lines(
                    observed_state->shared_map
                        .shared_vector_cursor,
                    kSharedVectorCursorShards
                );
        }
        if (claim_cursor && value >= 0 &&
            value < static_cast<int64_t>(kMaxTasks)) {
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
        if (claim_cursor && won && value >= 0 &&
            value < static_cast<int64_t>(kMaxTasks)) {
            claim_wins_by_task[
                static_cast<uint32_t>(value)
            ].fetch_add(1, std::memory_order_relaxed);
        }
        return current;
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

bool InsertTurnsMatch(
    const SchedulerState &state, uint32_t completed_tasks
) {
    for (uint32_t lane = 0;
         lane < kSharedInsertTurnCapacity; ++lane) {
        const volatile int64_t *address =
            lane == 0
                ? &state.shared_map.committed_tasks.value
                : &state.shared_map
                       .insert_turn_extra[lane - 1U].value;
        if (__atomic_load_n(address, __ATOMIC_ACQUIRE) !=
            SharedInsertTurnTokenAfterTasks(
                completed_tasks, lane
            )) {
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
                OrderedSubmitTestOps::
                    insert_publish_by_token[task_id + 1U]
                        .load(std::memory_order_relaxed) == 1;
        }
        planned_tasks += plan.task_count;
    }
    exact &= planned_tasks == task_count;

    uint32_t expected_lane_publishes[
        kSharedInsertTurnCapacity
    ] = {};
    for (uint32_t token = 1;
         token <= task_count; ++token) {
        ++expected_lane_publishes[
            token & kSharedInsertTurnMask
        ];
    }
    for (uint32_t lane = 0;
         lane < kSharedInsertTurnCapacity; ++lane) {
        exact &=
            OrderedSubmitTestOps::
                insert_publish_by_lane[lane].load(
                    std::memory_order_relaxed
                ) == expected_lane_publishes[lane];
    }
    return exact;
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

    constexpr uint32_t kTask = 4;
    SubmitContext context{};
    context.task_id = static_cast<int32_t>(kTask);
    context.won = false;
    context.kernel_id = -1;
    context.shared_result.Reset(static_cast<int32_t>(kTask));
    LocalStats stats{};
    const CallbackSubmitTicket ticket{
        1, kTask, -1, 0,
        EncodeSharedPaTaskMeta(TaskKind::Up, 0, true, false)
    };

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
        FinishSharedLoserSubmit<OrderedSubmitTestOps, false>(
            state, context, stats, ticket
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
    const bool ok =
        finished && state->fatal.value == 0 &&
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
    bool loser_gates_untouched = true;
    for (uint32_t task = 0; task < kTaskCount; ++task) {
        all_tasks_ready &= state->tasks[task].flag == 1;
        loser_gates_untouched &= state->tasks[task].deps_prepared == -1;
    }
    bool final_writers_ok = true;
    for (uint32_t slot = 0; slot < 3; ++slot) {
        final_writers_ok &=
            state->shared_map.shared_outputs[0]
                .last_writer[slot].value == 16;
    }

    const bool overlap =
        OrderedSubmitTestOps::task8_built_before_task4_completion.load(
            std::memory_order_acquire
        );
    const bool ok =
        state->fatal.value == 0 &&
        InsertTurnsMatch(*state, kTaskCount) &&
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
        loser_gates_untouched && final_writers_ok &&
        kernel_counts[0] == 4 && kernel_counts[1] == 4 &&
        kernel_counts[2] == 4 && kernel_counts[3] == 4 &&
        pa_scheduler::host::FinalBarrierStateMatches(
            state->final_barrier, options.final_barrier_shape
        );
    std::printf(
        "[ORDERED_SUBMIT] release_before_build=%s "
        "completed=%u turn0=%lld overlap=%u "
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
        InsertTurnsMatch(*state, kTaskCount) &&
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
        "completed=%u turn0=%lld kernels=%llu,%llu,%llu,%llu\n",
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
    const bool loser_ok = RunLoserZeroTensorMapAccessTest();
    const bool overlap_ok = RunInsertReleaseBeforeBuildTest();
    const bool execution_ok =
        RunIndependentKernelExecutionTest();
    if (!loser_ok || !overlap_ok || !execution_ok) {
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
