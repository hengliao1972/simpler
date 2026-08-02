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

constexpr uint32_t kAicWorker = 3;
constexpr uint32_t kAivWorker = 40;
constexpr uint32_t kForeignWorker = 17;

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
    worker.sub_block_id = static_cast<int32_t>(worker_id & 1U);
    ResetExecutionToken(state.exec_tokens[worker_id]);
    return worker;
}

bool PublishKernelCell(
    SchedulerState &state, uint32_t task_id,
    uint32_t build_owner, TaskKind kind,
    std::initializer_list<int32_t> fanin = {}
) {
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

void TestOpenSubmitIsNotScanned() {
    constexpr const char *kTest = "open-submit-not-scanned";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) {
        return;
    }
    WorkerState &worker = PrepareWorker(
        *state, kAicWorker, CoreRole::Aic
    );
    LocalStats stats{};
    ExecScanTestOps::ResetObservations();
    Check(
        PublishKernelCell(
            *state, 1, kAicWorker, TaskKind::Qk
        ),
        kTest, "publish open task 1"
    );

    // local_index 已经前移也不能授权扫描；唯一上界必须是成功 Close 的
    // replay_closed_exclusive。task 0 是已闭合 Alloc EMPTY，task 1 已经
    // BUILT 但仍属于当前尚未 Close 的 Submit。
    worker.local_index = 2;
    const uint32_t open_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 1, DrainPlace::EfDrain, stats
        );
    const DecodedExecState still_built = DecodeExecState(
        state->exec_cells[1].control.state
    );
    Check(
        open_progress == 0 && stats.exec_scan_task == 1 &&
            ExecScanTestOps::execute_calls == 0 &&
            still_built.valid &&
            still_built.phase == ExecPhase::Built &&
            state->tasks[1].flag == 0 && NoFatal(*state),
        kTest, "open Submit stays outside the scan prefix"
    );

    const uint32_t closed_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 2, DrainPlace::EfDrain, stats
        );
    Check(
        closed_progress == 1 && stats.exec_scan_task == 2 &&
            ExecScanTestOps::execute_calls == 1 &&
            ExecScanTestOps::executed_tasks[0] == 1 &&
            DecodeExecState(
                state->exec_cells[1].control.state
            ).phase == ExecPhase::Done &&
            state->tasks[1].flag == 1 && NoFatal(*state),
        kTest, "the same task becomes visible only after Close"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestAllocEmptyHoleIsSkipped() {
    constexpr const char *kTest = "alloc-empty-hole-skipped";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) {
        return;
    }
    WorkerState &worker = PrepareWorker(
        *state, kAicWorker, CoreRole::Aic
    );
    LocalStats stats{};
    ExecScanTestOps::ResetObservations();

    const uint32_t progressed =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 1, DrainPlace::EfDrain, stats
        );
    Check(
        progressed == 0 && stats.exec_scan_task == 1 &&
            ExecScanTestOps::execute_calls == 0 &&
            DecodeExecState(
                state->exec_cells[0].control.state
            ).phase == ExecPhase::Empty &&
            NoFatal(*state),
        kTest, "EMPTY Alloc does not block the closed prefix"
    );
    std::printf("[PASS] %s\n", kTest);
}

void RunForeignHoleFollowerCase(
    ExecPhase hole_phase, const char *test
) {
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, test, "state mapping");
    if (state == nullptr) {
        return;
    }
    WorkerState &worker = PrepareWorker(
        *state, kAicWorker, CoreRole::Aic
    );
    LocalStats stats{};
    ExecScanTestOps::ResetObservations();
    if (hole_phase == ExecPhase::Building) {
        SetCellState(
            *state, 0, ExecPhase::Building,
            kForeignWorker, kExecUnboundOwner,
            ExecEngineClass::None, 0
        );
    }
    Check(
        PublishKernelCell(
            *state, 1, kAicWorker, TaskKind::Qk
        ),
        test, "publish the local follower"
    );

    const uint32_t progressed =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 2, DrainPlace::EfDrain, stats
        );
    Check(
        progressed == 1 && stats.exec_scan_task == 2 &&
            ExecScanTestOps::execute_calls == 1 &&
            ExecScanTestOps::executed_tasks[0] == 1 &&
            DecodeExecState(
                state->exec_cells[1].control.state
            ).phase == ExecPhase::Done &&
            NoFatal(*state),
        test,
        "foreign hole cannot permanently hide a later local task"
    );
    std::printf("[PASS] %s\n", test);
}

void TestForeignHolesDoNotHideFollowers() {
    RunForeignHoleFollowerCase(
        ExecPhase::Empty, "foreign-empty-follower-discovered"
    );
    RunForeignHoleFollowerCase(
        ExecPhase::Building,
        "foreign-building-follower-discovered"
    );
}

void RunSelfTransitionalFailClosedCase(
    ExecPhase phase, const char *test
) {
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, test, "state mapping");
    if (state == nullptr) {
        return;
    }
    WorkerState &worker = PrepareWorker(
        *state, kAicWorker, CoreRole::Aic
    );
    LocalStats stats{};
    ExecScanTestOps::ResetObservations();
    if (phase == ExecPhase::Building) {
        SetCellState(
            *state, 0, phase, kAicWorker,
            kExecUnboundOwner, ExecEngineClass::None, 0
        );
    } else {
        SetCellState(
            *state, 0, phase, kAicWorker,
            kAicWorker, ExecEngineClass::Aic, 1
        );
    }

    const uint32_t progressed =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 1, DrainPlace::EfDrain, stats
        );
    Check(
        progressed == 0 && stats.exec_scan_task == 0 &&
            ExecScanTestOps::execute_calls == 0 &&
            FatalMatches(
                *state, ExecFatalReason::InvalidBuiltControl,
                0, kAicWorker
            ),
        test, "self transitional state fails closed"
    );
    std::printf("[PASS] %s\n", test);
}

void TestSelfTransitionalStatesFailClosed() {
    RunSelfTransitionalFailClosedCase(
        ExecPhase::Building, "self-building-fails-closed"
    );
    RunSelfTransitionalFailClosedCase(
        ExecPhase::Claimed, "self-claimed-fails-closed"
    );
}

void TestBusyTokenResumesScanning() {
    constexpr const char *kTest = "busy-token-resumes-scanning";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) {
        return;
    }
    WorkerState &worker = PrepareWorker(
        *state, kAivWorker, CoreRole::Aiv
    );
    LocalStats stats{};
    ExecScanTestOps::ResetObservations();
    Check(
        PublishKernelCell(
            *state, 1, kAivWorker, TaskKind::Sf, {0}
        ) &&
            PublishKernelCell(
                *state, 2, kAivWorker, TaskKind::Sf, {0}
            ),
        kTest, "publish blocked task and its follower"
    );

    const uint32_t first =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 3, DrainPlace::EfDrain, stats
        );
    Check(
        first == 0 && stats.exec_scan_task == 2 &&
            state->exec_tokens[kAivWorker].control.phase ==
                ExecTokenPhase::WaitingFanin &&
            DecodeExecState(
                state->exec_cells[1].control.state
            ).phase == ExecPhase::Claimed &&
            DecodeExecState(
                state->exec_cells[2].control.state
            ).phase == ExecPhase::Built &&
            ExecScanTestOps::execute_calls == 0 && NoFatal(*state),
        kTest, "busy token pauses before scanning the follower"
    );

    const uint32_t still_blocked =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 3, DrainPlace::EfDrain, stats
        );
    Check(
        still_blocked == 0 && stats.exec_scan_task == 2 &&
            ExecScanTestOps::execute_calls == 0 && NoFatal(*state),
        kTest, "repeated poll retains the discovery cursor"
    );

    __atomic_store_n(
        &state->tasks[0].flag, int64_t{1}, __ATOMIC_RELEASE
    );
    const uint32_t resumed =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 3, DrainPlace::EfDrain, stats
        );
    Check(
        resumed == 2 && stats.exec_scan_task == 3 &&
            state->exec_tokens[kAivWorker].control.phase ==
                ExecTokenPhase::Idle &&
            ExecScanTestOps::execute_calls == 2 &&
            ExecScanTestOps::executed_tasks[0] == 1 &&
            ExecScanTestOps::executed_tasks[1] == 2 &&
            DecodeExecState(
                state->exec_cells[1].control.state
            ).phase == ExecPhase::Done &&
            DecodeExecState(
                state->exec_cells[2].control.state
            ).phase == ExecPhase::Done &&
            NoFatal(*state),
        kTest, "ready token completes before the follower is rediscovered"
    );
    std::printf("[PASS] %s\n", kTest);
}

void SetTerminalKernelCell(
    SchedulerState &state, uint32_t task_id,
    ExecEngineClass engine
) {
    SetCellState(
        state, task_id, ExecPhase::Done,
        kForeignWorker, kForeignWorker, engine, 1
    );
    state.tasks[task_id].flag = 1;
}

void TestFinalDrainClosesLastTask() {
    constexpr const char *kTest = "final-drain-closes-last-task";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) {
        return;
    }
    WorkerState &worker = PrepareWorker(
        *state, kAivWorker, CoreRole::Aiv
    );
    LocalStats stats{};
    ExecScanTestOps::ResetObservations();

    // context_length=1 对应真实一组 PA 计划：Alloc/QK/SF/PV/UP。
    state->config.batches = 1;
    state->context_lens[0] = 1;
    state->tasks[0].flag = 1;
    SetTerminalKernelCell(*state, 1, ExecEngineClass::Aic);
    SetTerminalKernelCell(*state, 2, ExecEngineClass::Aiv);
    SetTerminalKernelCell(*state, 3, ExecEngineClass::Aic);
    Check(
        PublishKernelCell(
            *state, 4, kAivWorker, TaskKind::Up, {2, 3, 0}
        ),
        kTest, "publish the last UP task"
    );

    const uint32_t replay_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 4, DrainPlace::EfDrain, stats
        );
    uint32_t first_bad_task = UINT32_MAX;
    Check(
        replay_progress == 0 && stats.exec_scan_task == 4 &&
            ExecScanTestOps::execute_calls == 0 &&
            DecodeExecState(
                state->exec_cells[4].control.state
            ).phase == ExecPhase::Built &&
            !ValidateCrossCoreExecTerminalCells<ExecScanTestOps>(
                state, 5, first_bad_task
            ) &&
            first_bad_task == 4 && NoFatal(*state),
        kTest, "replay prefix leaves the open last task for FinalDrain"
    );

    const uint32_t final_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 5, DrainPlace::FinalDrain, stats
        );
    first_bad_task = UINT32_MAX;
    Check(
        final_progress == 1 && stats.exec_scan_task == 5 &&
            ExecScanTestOps::execute_calls == 1 &&
            ExecScanTestOps::executed_tasks[0] == 4 &&
            stats.result.placement[
                static_cast<uint32_t>(DrainPlace::FinalDrain)
            ] == 1 &&
            DecodeExecState(
                state->exec_cells[4].control.state
            ).phase == ExecPhase::Done &&
            state->tasks[4].flag == 1 &&
            CrossCoreExecWorkerDrained(
                state, worker, 5, stats
            ) &&
            CrossCoreExecTokenFullyReset(
                state->exec_tokens[kAivWorker]
            ) &&
            ValidateCrossCoreExecTerminalCells<ExecScanTestOps>(
                state, 5, first_bad_task
            ) &&
            first_bad_task == 5 && NoFatal(*state),
        kTest, "FinalDrain executes and validates the last task"
    );

    // 以确定性单线程顺序让 96 个真实 worker 各到达一次；前 95 个不能
    // 发布 release，最后到达者必须先走 terminal validator。
    std::array<bool, kWorkers> arrived{};
    bool all_arrivals_ok = true;
    for (uint32_t worker_id = 0;
         worker_id < kWorkers; ++worker_id) {
        WorkerState &arrival_worker = PrepareWorker(
            *state, worker_id,
            worker_id < kAicWorkers
                ? CoreRole::Aic : CoreRole::Aiv
        );
        LocalStats arrival_stats{};
        arrival_stats.exec_scan_task = 5;
        bool released = false;
        const bool closure_ok =
            ProgressCrossCoreExecDrainClosure<ExecScanTestOps>(
                state, arrival_worker, 5, arrival_stats,
                arrived[worker_id], released
            );
        all_arrivals_ok &= closure_ok && arrived[worker_id] &&
            released == (worker_id + 1U == kWorkers) &&
            state->exec_drain.arrived ==
                static_cast<int64_t>(worker_id + 1U);
    }
    Check(
        all_arrivals_ok &&
            state->exec_drain.arrived ==
                static_cast<int64_t>(kWorkers) &&
            state->exec_drain.release == 1 && NoFatal(*state),
        kTest, "last arrival publishes drain closure"
    );

    LocalStats repeat_stats{};
    repeat_stats.exec_scan_task = 5;
    bool repeat_released = false;
    const bool repeat_ok =
        ProgressCrossCoreExecDrainClosure<ExecScanTestOps>(
            state, state->workers[0], 5, repeat_stats,
            arrived[0], repeat_released
        );
    Check(
        repeat_ok && repeat_released &&
            state->exec_drain.arrived ==
                static_cast<int64_t>(kWorkers),
        kTest, "an arrived worker cannot increment arrival twice"
    );
    std::printf("[PASS] %s\n", kTest);
}

}  // namespace

int main() {
    TestOpenSubmitIsNotScanned();
    TestAllocEmptyHoleIsSkipped();
    TestForeignHolesDoNotHideFollowers();
    TestSelfTransitionalStatesFailClosed();
    TestBusyTokenResumesScanning();
    TestFinalDrainClosesLastTask();

    if (g_failures != 0) {
        std::fprintf(
            stderr, "[FAIL] cross-core exec scan: %d checks failed\n",
            g_failures
        );
        return 1;
    }
    std::printf("[PASS] cross-core exec scan and drain closure\n");
    return 0;
}
