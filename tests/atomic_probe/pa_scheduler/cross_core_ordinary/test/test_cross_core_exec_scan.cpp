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

    static void ResetObservations() {
        execute_calls = 0;
        executed_tasks.fill(UINT32_MAX);
    }

    static int32_t Load(volatile int32_t *address) {
        return __atomic_fetch_add(
            address, int32_t{0}, __ATOMIC_ACQUIRE
        );
    }

    static int64_t Load(volatile int64_t *address) {
        return __atomic_fetch_add(
            address, int64_t{0}, __ATOMIC_ACQUIRE
        );
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

WorkerState &PrepareWorker(
    SchedulerState &state, uint32_t worker_id, CoreRole role
) {
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
    if (task_id >= kMaxTasks || kind >= TaskKind::Count) {
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

void SetTerminalCells(
    SchedulerState &state, uint32_t task_count
) {
    for (uint32_t task_id = 0;
         task_id < task_count; ++task_id) {
        state.tasks[task_id].flag = 1;
    }
    // task 0/2/4 保持 raw-zero EMPTY，代表已经封口的 metadata-only
    // task；task 1/3 分别代表 AIC/AIV 已完成的 runtime payload。
    SetCellState(
        state, 1, ExecPhase::Done,
        /*build_owner=*/34, /*execute_owner=*/0,
        ExecEngineClass::Aic, /*payload_lines=*/1
    );
    SetCellState(
        state, 3, ExecPhase::Done,
        /*build_owner=*/0, /*execute_owner=*/32,
        ExecEngineClass::Aiv, /*payload_lines=*/1
    );
}

void SetDrainArrivals(
    SchedulerState &state, uint64_t completed_tasks
) {
    for (uint32_t group = 0;
         group < kExecDrainArrivalGroups; ++group) {
        const uint64_t group_completions =
            group == 0 ? completed_tasks : 0U;
        state.exec_drain.arrivals[group].state =
            static_cast<int64_t>(
                6U +
                (group_completions <<
                 kExecDrainArrivalCountBits)
            );
    }
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

void TestBothRolesScanWholeTaskRange() {
    constexpr const char *kTest =
        "both-roles-scan-whole-runtime-range";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;

    // 故意发布错误的 Host 表头和 task-id 表：新 ticket 合同只接受
    // runtime task_count 和两类中央游标，不能再读取这些旧答案。
    state->build_dispatch.task_count = 1;
    state->build_dispatch.executable_task_count = 1;
    state->exec_dispatch.aic_task_count = 1;
    state->exec_dispatch.aiv_task_count = 1;
    state->exec_dispatch.aic_task_ids[0] = 99;
    state->exec_dispatch.aiv_task_ids[0] = 77;

    std::array<LocalStats, 6> stats{};
    const std::array<uint32_t, 3> aic_workers{0, 7, 31};
    const std::array<uint32_t, 3> aiv_workers{32, 47, 95};
    const std::array<std::array<uint32_t, 2>, 3> expected{
        std::array<uint32_t, 2>{0, 1},
        std::array<uint32_t, 2>{2, 3},
        std::array<uint32_t, 2>{4, UINT32_MAX},
    };

    bool exact = true;
    for (uint32_t index = 0; index < 3; ++index) {
        InitLocalStats(
            stats[index], aic_workers[index], CoreRole::Aic
        );
        const SharedExecTicketResult aic =
            TakeSharedExecTicket<ExecScanTestOps>(
                state, aic_workers[index], CoreRole::Aic,
                /*task_count=*/5, stats[index]
            );
        exact &=
            aic.status == SharedExecTicketStatus::Acquired &&
            aic.task_count == (index == 2 ? 1U : 2U) &&
            aic.task_ids[0] == expected[index][0] &&
            aic.task_ids[1] == expected[index][1] &&
            stats[index].exec_dispatch_exhausted ==
                (index == 2 ? 1U : 0U);

        InitLocalStats(
            stats[3 + index], aiv_workers[index], CoreRole::Aiv
        );
        const SharedExecTicketResult aiv =
            TakeSharedExecTicket<ExecScanTestOps>(
                state, aiv_workers[index], CoreRole::Aiv,
                /*task_count=*/5, stats[3 + index]
            );
        exact &=
            aiv.status == SharedExecTicketStatus::Acquired &&
            aiv.task_count == (index == 2 ? 1U : 2U) &&
            aiv.task_ids[0] == expected[index][0] &&
            aiv.task_ids[1] == expected[index][1] &&
            stats[3 + index].exec_dispatch_exhausted ==
                (index == 2 ? 1U : 0U);
    }
    exact &= state->exec_dispatch.aic_next.value == 6 &&
        state->exec_dispatch.aiv_next.value == 6 &&
        NoFatal(*state);
    Check(
        exact, kTest,
        "AIC/AIV independently cover [0,5) with exact disjoint tails"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestZeroTaskRange() {
    constexpr const char *kTest = "zero-task-runtime-range";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;

    LocalStats aic_stats{};
    LocalStats aiv_stats{};
    InitLocalStats(aic_stats, 0, CoreRole::Aic);
    InitLocalStats(aiv_stats, 32, CoreRole::Aiv);
    const SharedExecTicketResult aic =
        TakeSharedExecTicket<ExecScanTestOps>(
            state, 0, CoreRole::Aic, 0, aic_stats
        );
    const SharedExecTicketResult aiv =
        TakeSharedExecTicket<ExecScanTestOps>(
            state, 32, CoreRole::Aiv, 0, aiv_stats
        );
    Check(
        aic.status == SharedExecTicketStatus::Exhausted &&
            aiv.status == SharedExecTicketStatus::Exhausted &&
            aic.task_count == 0 && aiv.task_count == 0 &&
            aic_stats.exec_dispatch_exhausted == 1 &&
            aiv_stats.exec_dispatch_exhausted == 1 &&
            state->exec_dispatch.aic_next.value == 2 &&
            state->exec_dispatch.aiv_next.value == 2 &&
            NoFatal(*state),
        kTest,
        "zero task closes each role with one terminal ticket"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestRuntimeCellRoutesAndConsumesOnce() {
    constexpr const char *kTest =
        "runtime-cell-routes-and-consumes-once";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;

    // task 0 是已经封口的 metadata-only task。其余 task 的 engine 只由
    // Build winner 发布的 runtime cell 决定；没有 Host role task-id 表。
    state->tasks[0].flag = 1;
    Check(
        PublishKernelCell(
            *state, 1, /*build_owner=*/34, TaskKind::Qk
        ),
        kTest, "publish AIC QK runtime cell"
    );
    Check(
        PublishKernelCell(
            *state, 2, /*build_owner=*/0, TaskKind::Sf, {0}
        ),
        kTest, "publish AIV SF runtime cell"
    );
    Check(
        PublishKernelCell(
            *state, 3, /*build_owner=*/34, TaskKind::Pv, {1}
        ),
        kTest, "publish AIC PV runtime cell"
    );
    Check(
        PublishKernelCell(
            *state, 4, /*build_owner=*/0,
            TaskKind::Up, {1, 2, 3}
        ),
        kTest, "publish AIV UP runtime cell"
    );
    if (!NoFatal(*state)) return;

    WorkerState &aic_worker = PrepareWorker(
        *state, 0, CoreRole::Aic
    );
    WorkerState &aiv_worker = PrepareWorker(
        *state, 32, CoreRole::Aiv
    );
    LocalStats aic_stats{};
    LocalStats aiv_stats{};
    InitLocalStats(aic_stats, 0, CoreRole::Aic);
    InitLocalStats(aiv_stats, 32, CoreRole::Aiv);
    ExecScanTestOps::ResetObservations();

    const uint32_t aic_completed =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, aic_worker, 5,
            /*production_closed=*/true,
            DrainPlace::FinalDrain, aic_stats
        );
    const DecodedExecState sf_after_aic =
        DecodeExecState(state->exec_cells[2].control.state);
    const DecodedExecState up_after_aic =
        DecodeExecState(state->exec_cells[4].control.state);
    Check(
        aic_completed == 2 &&
            ExecScanTestOps::execute_calls == 2 &&
            ExecScanTestOps::executed_tasks[0] == 1 &&
            ExecScanTestOps::executed_tasks[1] == 3 &&
            sf_after_aic.valid &&
            sf_after_aic.phase == ExecPhase::Built &&
            up_after_aic.valid &&
            up_after_aic.phase == ExecPhase::Built &&
            aic_stats.exec_dispatch_exhausted == 1 &&
            state->exec_dispatch.aic_next.value == 6 &&
            NoFatal(*state),
        kTest,
        "AIC consumes matching cells and skips Empty/AIV cells"
    );

    const uint32_t aiv_completed =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, aiv_worker, 5,
            /*production_closed=*/true,
            DrainPlace::FinalDrain, aiv_stats
        );
    Check(
        aiv_completed == 2 &&
            ExecScanTestOps::execute_calls == 4 &&
            ExecScanTestOps::executed_tasks[2] == 2 &&
            ExecScanTestOps::executed_tasks[3] == 4 &&
            aiv_stats.exec_dispatch_exhausted == 1 &&
            state->exec_dispatch.aiv_next.value == 6 &&
            NoFatal(*state),
        kTest,
        "AIV consumes matching cells and skips Empty/AIC cells"
    );

    const uint32_t aic_repeat =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, aic_worker, 5, true,
            DrainPlace::FinalDrain, aic_stats
        );
    const uint32_t aiv_repeat =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, aiv_worker, 5, true,
            DrainPlace::FinalDrain, aiv_stats
        );
    uint32_t first_bad_task = 5;
    Check(
        aic_repeat == 0 && aiv_repeat == 0 &&
            ExecScanTestOps::execute_calls == 4 &&
            CrossCoreExecWorkerDrained(
                state, aic_worker, 5, aic_stats
            ) &&
            CrossCoreExecWorkerDrained(
                state, aiv_worker, 5, aiv_stats
            ) &&
            CrossCoreExecAllTokensFullyReset(state, 0) &&
            CrossCoreExecAllTokensFullyReset(state, 32) &&
            ValidateCrossCoreExecTerminalCells<ExecScanTestOps>(
                state, 5, first_bad_task
            ) &&
            first_bad_task == 5 && NoFatal(*state),
        kTest,
        "each matching runtime cell executes exactly once and reaches DONE"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestClosedBuildingCellFailsClosed() {
    constexpr const char *kTest =
        "closed-building-cell-fails-closed";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;

    SetCellState(
        *state, 0, ExecPhase::Building,
        /*build_owner=*/0,
        /*execute_owner=*/kExecUnboundOwner,
        ExecEngineClass::None, /*payload_lines=*/0
    );
    WorkerState &worker = PrepareWorker(
        *state, 0, CoreRole::Aic
    );
    LocalStats stats{};
    InitLocalStats(stats, 0, CoreRole::Aic);
    ExecScanTestOps::ResetObservations();

    const uint32_t completed =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 1,
            /*production_closed=*/true,
            DrainPlace::FinalDrain, stats
        );
    Check(
        completed == 0 && ExecScanTestOps::execute_calls == 0 &&
            FatalMatches(
                *state, ExecFatalReason::InvalidBuiltControl,
                0, 0
            ) &&
            CrossCoreExecAllTokensFullyReset(state, 0),
        kTest,
        "BUILDING is impossible after replay_done and must be fatal"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestDrainDerivesExecutableCountFromTerminalCells() {
    constexpr const char *kTest =
        "drain-derives-count-from-terminal-cells";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;

    SetTerminalCells(*state, /*task_count=*/5);
    SetDrainArrivals(*state, /*completed_tasks=*/2);
    // 故意写入矛盾的旧 Host 摘要；root 必须从 runtime terminal cells
    // 动态得到两个 executable task，而不是相信该字段。
    state->build_dispatch.executable_task_count = 99;
    WorkerState &root = PrepareWorker(
        *state, 0, CoreRole::Aic
    );
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
        "two DONE cells close despite contradictory Host executable count"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestDrainCompletionMismatchFailsClosed() {
    constexpr const char *kTest =
        "drain-runtime-completion-mismatch-fails-closed";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;

    SetTerminalCells(*state, /*task_count=*/5);
    // terminal cells 动态证明应有两个 kernel completion，但 arrival
    // 汇总只有一个；root 必须在 device 内拒绝收口。
    SetDrainArrivals(*state, /*completed_tasks=*/1);
    WorkerState &root = PrepareWorker(
        *state, 0, CoreRole::Aic
    );
    LocalStats stats{};
    InitLocalStats(stats, 0, CoreRole::Aic);
    bool arrived = true;
    bool closed = false;
    const bool closure_ok =
        ProgressCrossCoreExecDrainClosure<ExecScanTestOps>(
            state, root, 5, stats, arrived, closed
        );
    Check(
        !closure_ok && arrived && !closed &&
            FatalMatches(
                *state, ExecFatalReason::CompletionStateConflict,
                5, 0
            ),
        kTest,
        "arrival count must equal executable DONE cells derived at runtime"
    );
    std::printf("[PASS] %s\n", kTest);
}

}  // namespace

int main() {
    TestBothRolesScanWholeTaskRange();
    TestZeroTaskRange();
    TestRuntimeCellRoutesAndConsumesOnce();
    TestClosedBuildingCellFailsClosed();
    TestDrainDerivesExecutableCountFromTerminalCells();
    TestDrainCompletionMismatchFailsClosed();

    if (g_failures != 0) {
        std::fprintf(
            stderr,
            "[FAIL] cross-core runtime Execute scan: %d checks failed\n",
            g_failures
        );
        return 1;
    }
    std::printf(
        "[PASS] cross-core runtime Execute scan and drain closure\n"
    );
    return 0;
}
