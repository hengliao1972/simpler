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
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <new>
#include <sys/mman.h>
#include <utility>

#ifndef PTO_FDWIC_SHARED_MAP
#define PTO_FDWIC_SHARED_MAP 1
#endif
#ifndef PA_BUILD_PERF_CLOCK
#define PA_BUILD_PERF_CLOCK 1
#endif
#define PA_DEVICE inline
#define PA_GM
#include "pa_scheduler_core.h"

namespace {

using namespace pa_scheduler;
using namespace pa_scheduler::cross_core;

int g_failures = 0;

constexpr uint32_t kAivBuildOwner = 34;

void Check(bool condition, const char *test, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] %s: %s\n", test, message);
    ++g_failures;
}

// 本测试只验证 CPU 上的共享状态机顺序，不模拟 A5 cache。原子返回值、
// payload 搬运和完成发布仍经过生产 helper 所要求的真实 Ops 边界。
struct ExecScanTestOps {
    static constexpr bool kAtomicReturnReadyObserved = false;
    static inline uint32_t execute_calls = 0;
    static inline std::array<uint32_t, 8> executed_tasks{};
    static inline volatile int64_t *watched_control = nullptr;
    static inline uint32_t watched_control_loads = 0;
    static inline uint32_t watched_control_cas_calls = 0;
    static inline uint32_t payload_loads = 0;
    static inline uint32_t invalidate_calls = 0;
    static inline volatile int64_t *claim_loss_address = nullptr;
    static inline int64_t injected_claimed_state = 0;
    static inline volatile int64_t *fatal_after_load_address = nullptr;
    static inline volatile int32_t *injected_global_fatal = nullptr;
    static inline volatile int32_t *fatal_after_execute = nullptr;

    static void ResetObservations() {
        execute_calls = 0;
        executed_tasks.fill(UINT32_MAX);
        watched_control = nullptr;
        watched_control_loads = 0;
        watched_control_cas_calls = 0;
        payload_loads = 0;
        invalidate_calls = 0;
        claim_loss_address = nullptr;
        injected_claimed_state = 0;
        fatal_after_load_address = nullptr;
        injected_global_fatal = nullptr;
        fatal_after_execute = nullptr;
    }

    static void WatchControl(volatile int64_t *control) {
        watched_control = control;
        watched_control_loads = 0;
        watched_control_cas_calls = 0;
    }

    static void InjectGlobalFatalAfterLoad(
        volatile int64_t *address, volatile int32_t *fatal
    ) {
        fatal_after_load_address = address;
        injected_global_fatal = fatal;
    }

    static void InjectGlobalFatalAfterExecute(
        volatile int32_t *fatal
    ) {
        fatal_after_execute = fatal;
    }

    static void InjectClaimLoss(
        volatile int64_t *address, int64_t claimed_state
    ) {
        claim_loss_address = address;
        injected_claimed_state = claimed_state;
    }

    static int32_t Load(volatile int32_t *address) {
        return __atomic_fetch_add(
            address, int32_t{0}, __ATOMIC_ACQUIRE
        );
    }

    static int64_t Load(volatile int64_t *address) {
        if (address == watched_control) {
            ++watched_control_loads;
        }
        const int64_t value = __atomic_fetch_add(
            address, int64_t{0}, __ATOMIC_ACQUIRE
        );
        if (address == fatal_after_load_address &&
            injected_global_fatal != nullptr) {
            volatile int32_t *fatal = injected_global_fatal;
            // 单次注入必须先撤销触发器，避免被后续同地址 Load 重复解释
            // 为多个并发故障窗口。
            fatal_after_load_address = nullptr;
            injected_global_fatal = nullptr;
            __atomic_store_n(fatal, int32_t{1}, __ATOMIC_RELEASE);
        }
        return value;
    }

    static uint64_t Load(volatile uint64_t *address) {
        return __atomic_fetch_add(
            address, uint64_t{0}, __ATOMIC_ACQUIRE
        );
    }

    static int32_t Exchange(
        volatile int32_t *address, int32_t value
    ) {
        return __atomic_exchange_n(
            address, value, __ATOMIC_ACQ_REL
        );
    }

    static int64_t Exchange(
        volatile int64_t *address, int64_t value
    ) {
        return __atomic_exchange_n(
            address, value, __ATOMIC_ACQ_REL
        );
    }

    static uint64_t Exchange(
        volatile uint64_t *address, uint64_t value
    ) {
        return __atomic_exchange_n(
            address, value, __ATOMIC_ACQ_REL
        );
    }

    static int64_t CompareExchange(
        volatile int64_t *address, int64_t expected,
        int64_t desired
    ) {
        if (address == watched_control) {
            ++watched_control_cas_calls;
        }
        if (address == claim_loss_address) {
            // 确定性模拟另一个合法候选恰好先完成 BUILT -> CLAIMED。
            // 撤销触发器后再执行被测 CAS，使本 observer 稳定取得 Lost，
            // 不依赖 host 线程竞争时序。
            const int64_t claimed_state = injected_claimed_state;
            claim_loss_address = nullptr;
            injected_claimed_state = 0;
            __atomic_store_n(
                address, claimed_state, __ATOMIC_RELEASE
            );
        }
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(
            address, &observed, desired, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        return observed;
    }

    static int64_t FetchAdd(
        volatile int64_t *address, int64_t value
    ) {
        return __atomic_fetch_add(
            address, value, __ATOMIC_ACQ_REL
        );
    }

    static void StoreBarrier() {
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }

    static uint64_t PerfClockNow() {
        return 0;
    }

    static uint64_t Now() {
        return 0;
    }

    static void StorePayloadWord(
        volatile uint64_t *address, uint64_t value
    ) {
        *address = value;
    }

    static void StoreTokenPayloadWord(
        volatile uint64_t *address, uint64_t value
    ) {
        *address = value;
    }

    static uint64_t LoadPayloadWord(
        const volatile uint64_t *address
    ) {
        ++payload_loads;
        return *address;
    }

    static void PreloadBuildDestination(void *, uint64_t) {}

    static void PreloadPayloadSource(const void *, uint64_t) {}

    static void PreloadTokenDestination(void *, uint64_t) {}

    static void BeforeBuiltPublish(uint32_t) {}

    static void BeforePayloadAcquire(uint32_t) {}

    static void FlushRegion(void *, uint64_t) {
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }

    static void InvalidateRegion(const void *, uint64_t) {
        ++invalidate_calls;
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }

    static bool ExecuteBoundKernel(
        SchedulerState *, WorkerState &,
        ExecutionToken &token, TaskKind, uint32_t
    ) {
        if (execute_calls >= executed_tasks.size()) {
            return false;
        }
        executed_tasks[execute_calls++] =
            ExecutionTokenHeader(token).task_id;
        if (fatal_after_execute != nullptr) {
            volatile int32_t *fatal = fatal_after_execute;
            fatal_after_execute = nullptr;
            __atomic_store_n(fatal, int32_t{1}, __ATOMIC_RELEASE);
        }
        return true;
    }
};

// 扫描测试不验证 descriptor 数值，但必须构造每种 PA task 的真实参数
// 数量，避免用通用 portable payload 绕过 adapter 的精确 dispatch ABI。
struct ScanPayloadSource {
    int32_t fanin[kExecMaxFanin]{};

    uint64_t TensorWord(uint32_t, uint32_t) const {
        return 0;
    }


    uint64_t TensorReference(uint32_t) const {
        return 0;
    }

    uint64_t Scalar(uint32_t) const {
        return 0;
    }

    int32_t Fanin(uint32_t edge) const {
        return fanin[edge];
    }
};

class MappedSchedulerState {
public:
    MappedSchedulerState() {
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
        flags |= MAP_NORESERVE;
#endif
        void *memory = mmap(
            nullptr, sizeof(SchedulerState),
            PROT_READ | PROT_WRITE, flags, -1, 0
        );
        if (memory == MAP_FAILED) {
            std::perror("mmap SchedulerState");
            return;
        }
        // fresh anonymous pages 已经全零；无花括号 placement-new 只建立
        // trivial SchedulerState 的对象生命周期，不主动触碰近 1 GiB。
        state_ = ::new (memory) SchedulerState;
    }

    ~MappedSchedulerState() {
        if (state_ != nullptr) {
            (void)munmap(state_, sizeof(SchedulerState));
        }
    }

    MappedSchedulerState(const MappedSchedulerState &) = delete;
    MappedSchedulerState &operator=(
        const MappedSchedulerState &
    ) = delete;

    SchedulerState *Get() const {
        return state_;
    }

private:
    SchedulerState *state_ = nullptr;
};

uint8_t ExecRouteForTaskKind(TaskKind kind) {
    switch (kind) {
        case TaskKind::Alloc:
            return EncodeExecDispatchRoute(
                false, ExecEngineClass::None
            );
        case TaskKind::Qk:
        case TaskKind::Pv:
            return EncodeExecDispatchRoute(
                true, ExecEngineClass::Aic
            );
        case TaskKind::Sf:
        case TaskKind::Up:
            return EncodeExecDispatchRoute(
                true, ExecEngineClass::Aiv
            );
        case TaskKind::Count:
            return 0;
    }
    return 0;
}

void EnsureDefaultDispatchPlan(SchedulerState &state) {
    if (state.build_dispatch.task_count != 0) {
        return;
    }
    constexpr uint32_t kPlanTasks =
        kMaxBatches * kTasksPerBatch;
    state.build_dispatch.task_count = kPlanTasks;
    state.build_dispatch.batch_count = kMaxBatches;
    uint32_t executable_tasks = 0;
    for (uint32_t task_id = 0;
         task_id < kPlanTasks; ++task_id) {
        const uint32_t task_offset = task_id % kTasksPerBatch;
        const TaskKind kind = task_offset == 0
            ? TaskKind::Alloc
            : static_cast<TaskKind>(task_offset);
        SharedBuildDispatchTaskIdentity &identity =
            state.build_dispatch.tasks[task_id];
        identity.batch = static_cast<uint16_t>(
            task_id / kTasksPerBatch
        );
        identity.encoded_meta = EncodeSharedPaTaskMeta(
            kind, 0, false,
            task_id + 1U == kPlanTasks
        );
        identity.exec_route = ExecRouteForTaskKind(kind);
        executable_tasks += kind == TaskKind::Alloc ? 0U : 1U;
        if (kind == TaskKind::Qk || kind == TaskKind::Pv) {
            state.exec_dispatch.aic_task_ids[
                state.exec_dispatch.aic_task_count++
            ] = task_id;
        } else if (kind == TaskKind::Sf ||
                   kind == TaskKind::Up) {
            state.exec_dispatch.aiv_task_ids[
                state.exec_dispatch.aiv_task_count++
            ] = task_id;
        }
    }
    state.build_dispatch.executable_task_count = executable_tasks;
}

bool SetDispatchTaskKind(
    SchedulerState &state, uint32_t task_id, TaskKind kind
) {
    EnsureDefaultDispatchPlan(state);
    if (task_id >= state.build_dispatch.task_count ||
        kind >= TaskKind::Count) {
        return false;
    }
    const uint32_t task_offset =
        SharedPaTaskOffset(kind, 0);
    if (task_id < task_offset) {
        return false;
    }
    SharedBuildDispatchTaskIdentity &identity =
        state.build_dispatch.tasks[task_id];
    identity.encoded_meta = EncodeSharedPaTaskMeta(
        kind, 0, false,
        task_id + 1U == state.build_dispatch.task_count
    );
    identity.exec_route = ExecRouteForTaskKind(kind);
    return identity.encoded_meta != 0 &&
           identity.exec_route != 0;
}

WorkerState &PrepareWorker(
    SchedulerState &state, uint32_t worker_id, CoreRole role
) {
    EnsureDefaultDispatchPlan(state);
    WorkerState &worker = state.workers[worker_id];
    worker.core_idx = static_cast<int32_t>(worker_id);
    worker.role = role;
    if (role == CoreRole::Aic) {
        worker.block_id = static_cast<int32_t>(worker_id);
        worker.lane = 0;
    } else {
        const uint32_t vector_id = worker_id - kAicWorkers;
        worker.block_id = static_cast<int32_t>(vector_id / 2U);
        worker.lane = static_cast<int32_t>(1U + vector_id % 2U);
    }
    worker.sub_block_id = worker.lane == 2 ? 1 : 0;
    for (uint32_t token_slot = 0;
         token_slot < kExecTokensPerWorker; ++token_slot) {
        ResetExecutionToken(
            state.exec_tokens[worker_id][token_slot]
        );
    }
    return worker;
}

bool PublishKernelCell(
    SchedulerState &state, uint32_t task_id,
    uint32_t build_owner, TaskKind kind,
    std::initializer_list<int32_t> fanin = {}
) {
    if (!SetDispatchTaskKind(state, task_id, kind)) {
        return false;
    }
    PaExecRoute route{};
    PaExecShape shape{};
    if (!ResolvePaExecRoute(kind, FunctionId(kind), route) ||
        !ResolvePaExecShape(kind, shape) ||
        fanin.size() != shape.fanin_count) {
        return false;
    }
    ScanPayloadSource source{};
    uint32_t edge = 0;
    for (int32_t producer : fanin) {
        source.fanin[edge++] = producer;
    }
    const ExecPayloadSpec spec{
        task_id,
        /*function_address=*/0,
        static_cast<uint64_t>(task_id + 1U) * kOutputAlignment,
        route.function_id,
        shape.tensor_count,
        shape.scalar_count,
        shape.fanin_count,
        route.engine_class,
        /*flags=*/0,
        /*multicore_group_id=*/0,
        /*multicore_rank=*/0,
        /*multicore_size=*/1,
        /*tensor_reference_mask=*/0,
    };
    return BuildAndPublishExecPayload<ExecScanTestOps>(
               state.exec_cells[task_id], build_owner,
               spec, source, state.exec_fatal
           ) == ExecBuildResult::Published;
}

void SetCellState(
    SchedulerState &state, uint32_t task_id, ExecPhase phase,
    uint32_t build_owner, uint32_t execute_owner,
    ExecEngineClass engine, uint32_t payload_lines
) {
    state.exec_cells[task_id].control.state =
        static_cast<int64_t>(EncodeExecState(
            phase, build_owner, execute_owner,
            engine, payload_lines, task_id
        ));
}

bool NoFatal(const SchedulerState &state) {
    return state.fatal.value == 0 &&
           state.exec_fatal.state == 0;
}

bool FatalMatches(
    const SchedulerState &state, ExecFatalReason reason,
    uint32_t task_id, uint32_t reporter
) {
    const DecodedExecFatal fatal =
        DecodeExecFatal(state.exec_fatal.state);
    return state.fatal.value == 1 && fatal.valid &&
           fatal.reason == reason && fatal.task_id == task_id &&
           fatal.reporter_owner == reporter;
}

void InitLocalStats(
    LocalStats &stats, uint32_t worker_id, CoreRole role
) {
    stats = {};
    stats.result.worker_id = worker_id;
    stats.result.role = static_cast<uint32_t>(role);
}

void TestDrainCompletionCountMismatchFailsClosed() {
    constexpr const char *kTest =
        "drain-completion-count-mismatch-fail-closed";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;

    state->config.batches = 1;
    state->context_lens[0] = 1;
    // B1 的五个 task 中只有四个 kernel。这里伪造 96 个 worker 已经全部
    // 到达，但分组完成数合计只有 3；root 必须在 device 内拒绝收口，
    // 不能依赖 kernel 返回后的 host 扫描才发现缺失。
    for (uint32_t group = 0;
         group < cross_core::kExecDrainArrivalGroups;
         ++group) {
        const uint64_t completions = group == 0 ? 3U : 0U;
        state->exec_drain.arrivals[group].state =
            static_cast<int64_t>(
                6U +
                (completions <<
                 cross_core::kExecDrainArrivalCountBits)
            );
    }
    WorkerState &root = PrepareWorker(*state, 0, CoreRole::Aic);
    state->build_dispatch.task_count = 5;
    state->build_dispatch.batch_count = 1;
    state->build_dispatch.executable_task_count = 4;
    LocalStats stats{};
    InitLocalStats(stats, 0, CoreRole::Aic);
    bool arrived = true;
    bool closed = false;
    const bool closure_ok =
        ProgressCrossCoreExecDrainClosure<ExecScanTestOps>(
            state, root, 5, stats, arrived, closed
        );
    const DecodedExecFatal fatal =
        DecodeExecFatal(state->exec_fatal.state);
    Check(
        !closure_ok && arrived && !closed &&
            state->fatal.value == 1 && fatal.valid &&
            fatal.reason ==
                ExecFatalReason::CompletionStateConflict,
        kTest,
        "root rejects an all-arrived drain with one missing completion"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestDrainUsesPublishedExecutableTaskCount() {
    constexpr const char *kTest =
        "drain-uses-published-executable-task-count";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;

    // 模拟一个与 PA 不同的计划：五个逻辑 task 中只有两个需要 engine
    // 执行。config.batches 仍为 1；旧 task_count-batches 推导会错误期待
    // 四个 completion，新协议必须只接受计划发布的精确值 2。
    state->config.batches = 1;
    for (uint32_t group = 0;
         group < cross_core::kExecDrainArrivalGroups;
         ++group) {
        const uint64_t completions = group == 0 ? 2U : 0U;
        state->exec_drain.arrivals[group].state =
            static_cast<int64_t>(
                6U +
                (completions <<
                 cross_core::kExecDrainArrivalCountBits)
            );
    }
    WorkerState &root = PrepareWorker(*state, 0, CoreRole::Aic);
    state->build_dispatch.task_count = 5;
    state->build_dispatch.batch_count = 1;
    state->build_dispatch.executable_task_count = 2;
    LocalStats stats{};
    InitLocalStats(stats, 0, CoreRole::Aic);
    bool arrived = true;
    bool closed = false;
    const bool closure_ok =
        ProgressCrossCoreExecDrainClosure<ExecScanTestOps>(
            state, root, 5, stats, arrived, closed
        );
    Check(
        closure_ok && arrived && closed && NoFatal(*state),
        kTest,
        "root closes against the explicit operator-independent count"
    );
    std::printf("[PASS] %s\n", kTest);
}

void LimitDefaultPlanToBatches(
    SchedulerState &state, uint32_t batches
) {
    EnsureDefaultDispatchPlan(state);
    const uint32_t task_count = batches * kTasksPerBatch;
    state.build_dispatch.task_count = task_count;
    state.build_dispatch.batch_count = batches;
    state.build_dispatch.executable_task_count = batches * 4U;
    state.build_dispatch.next_task.value = 0;
    state.exec_dispatch.aic_next.value = 0;
    state.exec_dispatch.aiv_next.value = 0;
    state.exec_dispatch.aic_task_count = batches * 2U;
    state.exec_dispatch.aiv_task_count = batches * 2U;
}

void TestDualTicketRoleDistribution() {
    constexpr const char *kTest = "dual-ticket-role-distribution";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;
    LimitDefaultPlanToBatches(*state, 2);

    const std::array<uint32_t, 4> aic_workers{7, 1, 31, 12};
    const std::array<uint32_t, 4> aiv_workers{63, 32, 95, 47};
    const std::array<uint32_t, 4> expected_aic{1, 3, 6, 8};
    const std::array<uint32_t, 4> expected_aiv{2, 4, 7, 9};
    std::array<LocalStats, 8> stats{};
    bool exact = true;
    for (uint32_t index = 0; index < 4; ++index) {
        InitLocalStats(
            stats[index], aic_workers[index], CoreRole::Aic
        );
        const SharedExecTicketResult ticket =
            TakeSharedExecTicket<ExecScanTestOps>(
                state, aic_workers[index], CoreRole::Aic,
                10, stats[index]
            );
        exact &= ticket.status == SharedExecTicketStatus::Acquired &&
            ticket.task_id == expected_aic[index] &&
            stats[index].exec_dispatch_exhausted == 0;

        InitLocalStats(
            stats[4 + index], aiv_workers[index], CoreRole::Aiv
        );
        const SharedExecTicketResult aiv_ticket =
            TakeSharedExecTicket<ExecScanTestOps>(
                state, aiv_workers[index], CoreRole::Aiv,
                10, stats[4 + index]
            );
        exact &=
            aiv_ticket.status == SharedExecTicketStatus::Acquired &&
            aiv_ticket.task_id == expected_aiv[index] &&
            stats[4 + index].exec_dispatch_exhausted == 0;
    }
    for (uint32_t index = 0; index < 4; ++index) {
        const SharedExecTicketResult aic_end =
            TakeSharedExecTicket<ExecScanTestOps>(
                state, aic_workers[index], CoreRole::Aic,
                10, stats[index]
            );
        const SharedExecTicketResult aiv_end =
            TakeSharedExecTicket<ExecScanTestOps>(
                state, aiv_workers[index], CoreRole::Aiv,
                10, stats[4 + index]
            );
        exact &= aic_end.status == SharedExecTicketStatus::Exhausted &&
            aiv_end.status == SharedExecTicketStatus::Exhausted &&
            stats[index].exec_dispatch_exhausted == 1 &&
            stats[4 + index].exec_dispatch_exhausted == 1;
    }
    exact &= state->exec_dispatch.aic_next.value == 8 &&
        state->exec_dispatch.aiv_next.value == 8 && NoFatal(*state);
    Check(
        exact, kTest,
        "all same-role workers consume one global ordered task stream"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestDualTicketWaitingBuiltAndOwnerIndependence() {
    constexpr const char *kTest =
        "dual-ticket-waiting-built-owner-independence";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;
    LimitDefaultPlanToBatches(*state, 1);
    state->tasks[0].flag = 1;

    constexpr uint32_t kExecutor = 0;
    WorkerState &worker = PrepareWorker(
        *state, kExecutor, CoreRole::Aic
    );
    LocalStats stats{};
    InitLocalStats(stats, kExecutor, CoreRole::Aic);
    ExecScanTestOps::ResetObservations();

    const uint32_t empty_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 5, false,
            DrainPlace::EfDrain, stats
        );
    Check(
        empty_progress == 0 &&
            state->exec_dispatch.aic_next.value == 1 &&
            state->exec_tokens[kExecutor][0].control.phase ==
                ExecTokenPhase::WaitingBuilt &&
            state->exec_tokens[kExecutor][0].control.task_id == 1 &&
            state->exec_tokens[kExecutor][1].control.phase ==
                ExecTokenPhase::Idle &&
            stats.max_occupied == 1 &&
            stats.exec_dispatch_exhausted == 0 && NoFatal(*state),
        kTest,
        "one boundary retains only its first unpublished ticket"
    );

    const uint32_t second_empty_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 5, false,
            DrainPlace::EfDrain, stats
        );
    const uint32_t exhaust_probe_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 5, false,
            DrainPlace::EfDrain, stats
        );
    const ExecutionTokenControl &first_wait =
        state->exec_tokens[kExecutor][0].control;
    const ExecutionTokenControl &second_wait =
        state->exec_tokens[kExecutor][1].control;
    Check(
        second_empty_progress == 0 &&
            exhaust_probe_progress == 0 &&
            state->exec_dispatch.aic_next.value == 3 &&
            first_wait.phase == ExecTokenPhase::WaitingBuilt &&
            first_wait.task_id == 1 &&
            second_wait.phase == ExecTokenPhase::WaitingBuilt &&
            second_wait.task_id == 3 &&
            stats.max_occupied == 2 &&
            stats.exec_dispatch_exhausted == 1 && NoFatal(*state),
        kTest,
        "later boundaries retain distinct tickets and eventually observe exhaustion"
    );

    Check(
        PublishKernelCell(
            *state, 1, kExecutor, TaskKind::Qk
        ),
        kTest, "publish same-owner QK payload"
    );
    const uint32_t same_owner_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 5, false,
            DrainPlace::EfDrain, stats
        );
    const DecodedExecState qk_done = DecodeExecState(
        state->exec_cells[1].control.state
    );
    Check(
        same_owner_progress == 1 && qk_done.valid &&
            qk_done.phase == ExecPhase::Done &&
            qk_done.build_owner == kExecutor &&
            qk_done.execute_owner == kExecutor &&
            state->tasks[1].flag == 1 &&
            stats.exec_dispatch_exhausted == 1 &&
            state->exec_dispatch.aic_next.value == 3 &&
            NoFatal(*state),
        kTest,
        "same Scalar may independently own Build and Execute"
    );

    Check(
        PublishKernelCell(
            *state, 3, kAivBuildOwner,
            TaskKind::Pv, {1}
        ),
        kTest, "publish cross-role-built PV payload"
    );
    const uint32_t cross_owner_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 5, true,
            DrainPlace::FinalDrain, stats
        );
    const DecodedExecState pv_done = DecodeExecState(
        state->exec_cells[3].control.state
    );
    Check(
        cross_owner_progress == 1 && pv_done.valid &&
            pv_done.phase == ExecPhase::Done &&
            pv_done.build_owner == kAivBuildOwner &&
            pv_done.execute_owner == kExecutor &&
            CrossCoreExecWorkerDrained(
                state, worker, 5, stats
            ) &&
            CrossCoreExecAllTokensFullyReset(
                state, kExecutor
            ) && NoFatal(*state),
        kTest,
        "cross-owner payload uses the same ticket and completion protocol"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestBlockedTokenStopsSameBoundaryLookahead() {
    constexpr const char *kTest =
        "blocked-token-stops-same-boundary-lookahead";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;
    LimitDefaultPlanToBatches(*state, 2);

    constexpr uint32_t kExecutor = 0;
    WorkerState &worker = PrepareWorker(
        *state, kExecutor, CoreRole::Aic
    );
    LocalStats stats{};
    InitLocalStats(stats, kExecutor, CoreRole::Aic);
    ExecScanTestOps::ResetObservations();

    const uint32_t first_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 10, false,
            DrainPlace::EfDrain, stats
        );
    Check(
        first_progress == 0 &&
            state->exec_dispatch.aic_next.value == 1 &&
            stats.max_occupied == 1 &&
            stats.exec_dispatch_exhausted == 0 &&
            state->exec_tokens[kExecutor][0].control.phase ==
                ExecTokenPhase::WaitingBuilt &&
            state->exec_tokens[kExecutor][0].control.task_id == 1 &&
            state->exec_tokens[kExecutor][1].control.phase ==
                ExecTokenPhase::Idle &&
            state->exec_tokens[kExecutor][2].control.phase ==
                ExecTokenPhase::Idle &&
            state->exec_tokens[kExecutor][3].control.phase ==
                ExecTokenPhase::Idle &&
            NoFatal(*state),
        kTest,
        "one unpublished task stops further tickets in the same boundary"
    );

    uint32_t later_progress = 0;
    for (uint32_t boundary = 0; boundary < 3; ++boundary) {
        later_progress += ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 10, false,
            DrainPlace::EfDrain, stats
        );
    }
    Check(
        later_progress == 0 &&
            state->exec_dispatch.aic_next.value == 4 &&
            stats.max_occupied == 4 &&
            stats.exec_dispatch_exhausted == 0 &&
            state->exec_tokens[kExecutor][0].control.phase ==
                ExecTokenPhase::WaitingBuilt &&
            state->exec_tokens[kExecutor][0].control.task_id == 1 &&
            state->exec_tokens[kExecutor][1].control.phase ==
                ExecTokenPhase::WaitingBuilt &&
            state->exec_tokens[kExecutor][1].control.task_id == 3 &&
            state->exec_tokens[kExecutor][2].control.phase ==
                ExecTokenPhase::WaitingBuilt &&
            state->exec_tokens[kExecutor][2].control.task_id == 6 &&
            state->exec_tokens[kExecutor][3].control.phase ==
                ExecTokenPhase::WaitingBuilt &&
            state->exec_tokens[kExecutor][3].control.task_id == 8 &&
            !CrossCoreExecWorkerDrained(
                state, worker, 10, stats
            ) && NoFatal(*state),
        kTest,
        "separate progress boundaries can still fill the four-token capacity"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestFaninWaitStillAllowsBoundedLookahead() {
    constexpr const char *kTest =
        "fanin-wait-allows-bounded-lookahead";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;
    LimitDefaultPlanToBatches(*state, 2);

    constexpr uint32_t kExecutor = 0;
    WorkerState &worker = PrepareWorker(
        *state, kExecutor, CoreRole::Aic
    );
    LocalStats stats{};
    InitLocalStats(stats, kExecutor, CoreRole::Aic);
    ExecScanTestOps::ResetObservations();
    // 跳过无 fanin 的 QK#1，从真实依赖 QK#1 的 PV#3 开始领取。
    state->exec_dispatch.aic_next.value = 1;
    Check(
        PublishKernelCell(
            *state, 3, kAivBuildOwner,
            TaskKind::Pv, {1}
        ),
        kTest, "publish PV payload with an unready QK fanin"
    );

    const uint32_t progressed =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 10, false,
            DrainPlace::EfDrain, stats
        );
    Check(
        progressed == 0 &&
            state->exec_dispatch.aic_next.value == 3 &&
            state->exec_tokens[kExecutor][0].control.phase ==
                ExecTokenPhase::WaitingFanin &&
            state->exec_tokens[kExecutor][0].control.task_id == 3 &&
            state->exec_tokens[kExecutor][1].control.phase ==
                ExecTokenPhase::WaitingBuilt &&
            state->exec_tokens[kExecutor][1].control.task_id == 6 &&
            stats.max_occupied == 2 && NoFatal(*state),
        kTest,
        "claimed fanin wait permits one more ticket before an unpublished task stops the boundary"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestDualTicketDuplicateConsumerFailsClosed() {
    constexpr const char *kTest =
        "dual-ticket-duplicate-consumer-fails-closed";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;
    LimitDefaultPlanToBatches(*state, 1);

    constexpr uint32_t kTicketOwner = 0;
    WorkerState &worker = PrepareWorker(
        *state, kTicketOwner, CoreRole::Aic
    );
    LocalStats stats{};
    InitLocalStats(stats, kTicketOwner, CoreRole::Aic);
    ExecScanTestOps::ResetObservations();
    (void)ProgressCrossCoreExec<ExecScanTestOps>(
        state, worker, 5, false,
        DrainPlace::EfDrain, stats
    );
    Check(
        PublishKernelCell(*state, 1, 7, TaskKind::Qk),
        kTest, "publish claimed-conflict source"
    );
    const DecodedExecState built = DecodeExecState(
        state->exec_cells[1].control.state
    );
    SetCellState(
        *state, 1, ExecPhase::Claimed,
        built.build_owner, 1,
        built.engine_class, built.payload_lines
    );
    const uint32_t progressed =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 5, false,
            DrainPlace::EfDrain, stats
        );
    Check(
        progressed == 0 &&
            FatalMatches(
                *state, ExecFatalReason::InvalidBuiltControl,
                1, kTicketOwner
            ) &&
            state->exec_tokens[kTicketOwner][0].control.phase ==
                ExecTokenPhase::Faulted,
        kTest,
        "a unique ticket never treats an existing CLAIMED owner as a loser"
    );
    std::printf("[PASS] %s\n", kTest);
}

}  // namespace

int main() {
    TestDualTicketRoleDistribution();
    TestDualTicketWaitingBuiltAndOwnerIndependence();
    TestBlockedTokenStopsSameBoundaryLookahead();
    TestFaninWaitStillAllowsBoundedLookahead();
    TestDualTicketDuplicateConsumerFailsClosed();
    TestDrainCompletionCountMismatchFailsClosed();
    TestDrainUsesPublishedExecutableTaskCount();

    if (g_failures != 0) {
        std::fprintf(
            stderr, "[FAIL] cross-core Execute ticket: %d checks failed\n",
            g_failures
        );
        return 1;
    }
    std::printf("[PASS] cross-core Execute ticket and drain closure\n");
    return 0;
}
