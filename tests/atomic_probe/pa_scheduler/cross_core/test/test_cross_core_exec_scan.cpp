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

CoreRole RoleForWorker(uint32_t worker_id) {
    return worker_id < kAicWorkers ? CoreRole::Aic : CoreRole::Aiv;
}

void InitLocalStats(
    LocalStats &stats, uint32_t worker_id, CoreRole role
) {
    stats = {};
    stats.result.worker_id = worker_id;
    stats.result.role = static_cast<uint32_t>(role);
}

bool CloseSyntheticSubmit(
    SchedulerState &state, LocalStats &stats,
    uint32_t task_id, TaskKind kind
) {
    // 走生产 Close 边界来登记候选，避免测试直接预填 bitmap 后
    // 错把测试搭建方式当成真实 Submit 合同。
    return CloseSharedCallbackSubmit<ExecScanTestOps, false>(
        &state, stats, task_id, kind,
        /*submit_begin=*/0, /*is_last_submit=*/false
    );
}

bool RegisterLocalCandidate(
    LocalStats &stats, uint32_t task_id, TaskKind kind
) {
    // 状态机局部用例没有重放完整 Submit 流，只验证给定 closed prefix
    // 内的扫描行为。这里直接走生产登记原语；调用方仍从 slot 0 开始，
    // 或显式证明当前 task 的 slot 没有落在游标之后。
    return RegisterCrossCoreExecCandidate(stats, task_id, kind);
}

bool SelectTestExecutor(
    uint32_t task_id, TaskKind kind, uint32_t build_owner,
    uint32_t &primary, uint32_t &secondary, uint32_t &execute_owner
) {
    PaExecRoute route{};
    execute_owner = kExecUnboundOwner;
    if (!ResolvePaExecRoute(kind, FunctionId(kind), route) ||
        !FixedPaExecuteCandidates(
            task_id, route.engine_class, primary, secondary
        )) {
        return false;
    }
    // 只为需要一个确定 executor 的状态机构造选择合法候选；这不是
    // 生产 owner 策略。build 在候选外时优先 primary，在候选内时选择
    // 另一个候选，最终仍由生产 eligibility helper 验证。
    execute_owner = build_owner == primary ? secondary : primary;
    return PaExecOwnerMatchesEngine(build_owner, route.engine_class) &&
           PaExecuteOwnerEligible(
               task_id, build_owner, route.engine_class, execute_owner
           );
}

bool CandidateBitForTask(
    const LocalStats &stats, uint32_t worker_id,
    CoreRole role, uint32_t task_id
) {
    uint32_t candidate_slot = kCrossCoreExecMaxCandidateSlots;
    return CrossCoreExecPotentialSlotForTask(
               worker_id, role, task_id, candidate_slot
           ) &&
           CrossCoreExecCandidateRegistered(stats, candidate_slot);
}

bool FindSameEngineNonCandidate(
    ExecEngineClass engine, uint32_t primary, uint32_t secondary,
    uint32_t &owner
) {
    owner = kExecUnboundOwner;
    for (uint32_t worker_id = 0; worker_id < kWorkers; ++worker_id) {
        if (worker_id != primary && worker_id != secondary &&
            PaExecOwnerMatchesEngine(worker_id, engine)) {
            owner = worker_id;
            return true;
        }
    }
    return false;
}

bool SetTerminalKernelCell(
    SchedulerState &state, uint32_t task_id,
    TaskKind kind, uint32_t build_owner,
    uint32_t requested_execute_owner = kExecUnboundOwner
) {
    PaExecRoute route{};
    uint32_t primary = kExecUnboundOwner;
    uint32_t secondary = kExecUnboundOwner;
    uint32_t selected_execute_owner = kExecUnboundOwner;
    if (!ResolvePaExecRoute(kind, FunctionId(kind), route) ||
        !SelectTestExecutor(
            task_id, kind, build_owner,
            primary, secondary, selected_execute_owner
        )) {
        return false;
    }
    const uint32_t execute_owner =
        requested_execute_owner == kExecUnboundOwner
            ? selected_execute_owner : requested_execute_owner;
    if (!PaExecuteOwnerEligible(
            task_id, build_owner, route.engine_class, execute_owner
        )) {
        return false;
    }
    SetCellState(
        state, task_id, ExecPhase::Done,
        build_owner, execute_owner, route.engine_class, 1
    );
    state.tasks[task_id].flag = 1;
    return true;
}

void TestPotentialSlotRoundTrip() {
    constexpr const char *kTest = "candidate-slot-round-trip";
    bool all_ok = true;

    for (uint32_t worker_id = 0;
         worker_id < kWorkers; ++worker_id) {
        const CoreRole role = RoleForWorker(worker_id);
        uint32_t previous_task = 0;
        bool has_previous = false;
        uint32_t mapped_slots = 0;
        for (uint32_t slot = 0;
             slot < kCrossCoreExecMaxCandidateSlots; ++slot) {
            uint32_t task_id = kMaxTasks;
            if (!CrossCoreExecPotentialTaskAt(
                    worker_id, role, slot, task_id
                )) {
                break;
            }
            uint32_t reverse_slot =
                kCrossCoreExecMaxCandidateSlots;
            all_ok &= task_id < kMaxTasks &&
                (!has_previous || task_id > previous_task) &&
                CrossCoreExecPotentialSlotForTask(
                    worker_id, role, task_id, reverse_slot
                ) &&
                reverse_slot == slot;
            previous_task = task_id;
            has_previous = true;
            ++mapped_slots;
        }
        all_ok &= mapped_slots != 0 &&
            mapped_slots <= kCrossCoreExecMaxCandidateSlots;
    }

    constexpr std::array<ExecEngineClass, 2> kEngines{
        ExecEngineClass::Aic, ExecEngineClass::Aiv
    };
    for (uint32_t task_id = 0; task_id < kMaxTasks; ++task_id) {
        for (ExecEngineClass engine : kEngines) {
            uint32_t primary = kExecUnboundOwner;
            uint32_t secondary = kExecUnboundOwner;
            all_ok &= FixedPaExecuteCandidates(
                task_id, engine, primary, secondary
            );
            for (uint32_t worker_id :
                 std::array<uint32_t, 2>{primary, secondary}) {
                const CoreRole role = RoleForWorker(worker_id);
                uint32_t slot = kCrossCoreExecMaxCandidateSlots;
                uint32_t round_trip_task = kMaxTasks;
                all_ok &= CrossCoreExecPotentialSlotForTask(
                              worker_id, role, task_id, slot
                          ) &&
                    CrossCoreExecPotentialTaskAt(
                        worker_id, role, slot, round_trip_task
                    ) &&
                    round_trip_task == task_id;
            }
        }
    }

    Check(
        all_ok, kTest,
        "all compact candidate slots are ordered and reversible"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestDynamicExecutorEligibility() {
    constexpr const char *kTest = "dynamic-executor-eligibility";
    bool all_ok = true;
    constexpr std::array<std::pair<uint32_t, TaskKind>, 2> kCases{{
        {1, TaskKind::Qk}, {2, TaskKind::Sf}
    }};

    for (const auto &test_case : kCases) {
        const uint32_t task_id = test_case.first;
        const TaskKind kind = test_case.second;
        PaExecRoute route{};
        uint32_t primary = kExecUnboundOwner;
        uint32_t secondary = kExecUnboundOwner;
        uint32_t outside = kExecUnboundOwner;
        all_ok &= ResolvePaExecRoute(
                      kind, FunctionId(kind), route
                  ) &&
            FixedPaExecuteCandidates(
                task_id, route.engine_class, primary, secondary
            ) &&
            FindSameEngineNonCandidate(
                route.engine_class, primary, secondary, outside
            );
        if (outside == kExecUnboundOwner) {
            continue;
        }

        // Build owner 是 primary/secondary 时，只剩另一个候选合法；
        // Build owner 在双候选外时，两个候选都可参加动态 CAS。
        all_ok &= PaExecOwnerMatchesEngine(
                      primary, route.engine_class
                  ) &&
            PaExecOwnerMatchesEngine(secondary, route.engine_class) &&
            PaExecOwnerMatchesEngine(outside, route.engine_class) &&
            !PaExecuteOwnerEligible(
                task_id, primary, route.engine_class, primary
            ) &&
            PaExecuteOwnerEligible(
                task_id, primary, route.engine_class, secondary
            ) &&
            PaExecuteOwnerEligible(
                task_id, secondary, route.engine_class, primary
            ) &&
            !PaExecuteOwnerEligible(
                task_id, secondary, route.engine_class, secondary
            ) &&
            PaExecuteOwnerEligible(
                task_id, outside, route.engine_class, primary
            ) &&
            PaExecuteOwnerEligible(
                task_id, outside, route.engine_class, secondary
            ) &&
            !PaExecuteOwnerEligible(
                task_id, outside, route.engine_class, outside
            );
    }

    Check(
        all_ok, kTest,
        "build placement yields the exact one-or-two executor set"
    );
    std::printf("[PASS] %s\n", kTest);
}

TaskKind SyntheticKind(uint32_t task_id) {
    constexpr std::array<TaskKind, 5> kKinds{
        TaskKind::Alloc, TaskKind::Qk, TaskKind::Sf,
        TaskKind::Pv, TaskKind::Up
    };
    return kKinds[task_id % kKinds.size()];
}

void TestSubmitCloseRegistersOnlyCandidates() {
    constexpr const char *kTest =
        "submit-close-candidate-registration";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;

    std::array<LocalStats, kWorkers> stats{};
    for (uint32_t worker_id = 0;
         worker_id < kWorkers; ++worker_id) {
        InitLocalStats(
            stats[worker_id], worker_id, RoleForWorker(worker_id)
        );
    }

    constexpr uint32_t kSyntheticTasks = 137;
    bool all_ok = true;
    for (uint32_t task_id = 0;
         task_id < kSyntheticTasks; ++task_id) {
        const TaskKind kind = SyntheticKind(task_id);
        PaExecRoute route{};
        uint32_t primary = kExecUnboundOwner;
        uint32_t secondary = kExecUnboundOwner;
        const bool kernel = kind != TaskKind::Alloc;
        if (kernel) {
            all_ok &= ResolvePaExecRoute(
                          kind, FunctionId(kind), route
                      ) &&
                FixedPaExecuteCandidates(
                    task_id, route.engine_class, primary, secondary
                );
        }
        uint32_t registered_workers = 0;
        for (uint32_t worker_id = 0;
             worker_id < kWorkers; ++worker_id) {
            all_ok &= CloseSyntheticSubmit(
                *state, stats[worker_id], task_id, kind
            );
            const bool registered = CandidateBitForTask(
                stats[worker_id], worker_id,
                RoleForWorker(worker_id), task_id
            );
            const bool expected = kernel &&
                (worker_id == primary || worker_id == secondary);
            all_ok &= registered == expected &&
                stats[worker_id].result.submits == task_id + 1U;
            registered_workers += registered ? 1U : 0U;
        }
        all_ok &= registered_workers == (kernel ? 2U : 0U);
    }

    Check(
        all_ok && NoFatal(*state), kTest,
        "real Close path registers two kernel candidates and no Alloc"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestCloseAndRegistrationRejectHistoryRewrite() {
    constexpr const char *kTest =
        "close-and-registration-reject-history-rewrite";
    bool all_ok = true;

    // Close 的 task id 必须恰好等于已经成功闭合的 Submit 数；不能从
    // task 1 起步，也不能把已经闭合的 task 0 再闭合一次。
    {
        MappedSchedulerState mapping;
        SchedulerState *state = mapping.Get();
        all_ok &= state != nullptr;
        if (state != nullptr) {
            LocalStats stats{};
            InitLocalStats(stats, 1, CoreRole::Aic);
            const bool out_of_order = CloseSyntheticSubmit(
                *state, stats, 1, TaskKind::Qk
            );
            all_ok &= !out_of_order &&
                stats.result.submits == 0 &&
                state->fatal.value == 1;
        }
    }
    {
        MappedSchedulerState mapping;
        SchedulerState *state = mapping.Get();
        all_ok &= state != nullptr;
        if (state != nullptr) {
            LocalStats stats{};
            InitLocalStats(stats, 1, CoreRole::Aic);
            const bool first = CloseSyntheticSubmit(
                *state, stats, 0, TaskKind::Alloc
            );
            const bool duplicate = CloseSyntheticSubmit(
                *state, stats, 0, TaskKind::Alloc
            );
            all_ok &= first && !duplicate &&
                stats.result.submits == 1 &&
                state->fatal.value == 1;
        }
    }

    // worker 2 的第一个潜在 AIC task 是 task 1。scanner 已将这个
    // 未登记槽永久越过后，登记原语必须拒绝把历史 bit 写回游标后方。
    {
        MappedSchedulerState mapping;
        SchedulerState *state = mapping.Get();
        all_ok &= state != nullptr;
        if (state != nullptr) {
            constexpr uint32_t kWorker = 2;
            WorkerState &worker = PrepareWorker(
                *state, kWorker, CoreRole::Aic
            );
            LocalStats stats{};
            InitLocalStats(stats, kWorker, CoreRole::Aic);
            const uint32_t progressed =
                ProgressCrossCoreExec<ExecScanTestOps>(
                    state, worker, 2,
                    /*production_closed=*/false,
                    DrainPlace::EfDrain, stats
                );
            const bool late_registration =
                RegisterLocalCandidate(stats, 1, TaskKind::Qk);
            all_ok &= progressed == 0 &&
                stats.exec_candidate_slot == 1 &&
                !late_registration &&
                CrossCoreExecCandidateBitmapEmpty(stats) &&
                NoFatal(*state);
        }
    }

    // 尚未越过的同一候选也不能重复置位，防止一次 Submit 被消费两次。
    {
        LocalStats stats{};
        InitLocalStats(stats, 2, CoreRole::Aic);
        const bool first =
            RegisterLocalCandidate(stats, 1, TaskKind::Qk);
        const bool duplicate =
            RegisterLocalCandidate(stats, 1, TaskKind::Qk);
        all_ok &= first && !duplicate &&
            CandidateBitForTask(
                stats, 2, CoreRole::Aic, 1
            );
    }

    Check(
        all_ok, kTest,
        "out-of-order/duplicate Close and stale candidate bit fail"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestOnlyTwoCandidatesObserveControl() {
    constexpr const char *kTest = "two-candidates-only";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;

    constexpr std::array<TaskKind, 4> kKernelKinds{
        TaskKind::Qk, TaskKind::Sf, TaskKind::Pv, TaskKind::Up
    };
    bool all_workers_ok = true;
    for (uint32_t kind_index = 0;
         kind_index < kKernelKinds.size(); ++kind_index) {
        const uint32_t task_id = kind_index + 1U;
        uint32_t primary = kExecUnboundOwner;
        uint32_t secondary = kExecUnboundOwner;
        uint32_t execute_owner = kExecUnboundOwner;
        const bool aic_task = kKernelKinds[kind_index] == TaskKind::Qk ||
            kKernelKinds[kind_index] == TaskKind::Pv;
        const uint32_t build_owner =
            aic_task ? task_id : kAicWorkers + task_id;
        all_workers_ok &= SelectTestExecutor(
            task_id, kKernelKinds[kind_index], build_owner,
            primary, secondary, execute_owner
        );
        uint32_t observer_workers = 0;
        uint32_t total_control_loads = 0;
        for (uint32_t worker_id = 0;
             worker_id < kWorkers; ++worker_id) {
            WorkerState &worker = PrepareWorker(
                *state, worker_id, RoleForWorker(worker_id)
            );
            LocalStats stats{};
            InitLocalStats(
                stats, worker_id, RoleForWorker(worker_id)
            );
            all_workers_ok &= RegisterLocalCandidate(
                stats, task_id, kKernelKinds[kind_index]
            );
            ExecScanTestOps::ResetObservations();
            ExecScanTestOps::WatchControl(
                &state->exec_cells[task_id].control.state
            );
            const uint32_t progressed =
                ProgressCrossCoreExec<ExecScanTestOps>(
                    state, worker, task_id + 1U,
                    /*production_closed=*/false,
                    DrainPlace::EfDrain, stats
                );
            const bool candidate =
                worker_id == primary || worker_id == secondary;
            observer_workers +=
                ExecScanTestOps::watched_control_loads != 0 ? 1U : 0U;
            total_control_loads +=
                ExecScanTestOps::watched_control_loads;
            all_workers_ok &= progressed == 0 &&
                ExecScanTestOps::execute_calls == 0 &&
                ExecScanTestOps::watched_control_loads ==
                    (candidate ? 1U : 0U);
            if (candidate) {
                all_workers_ok &= CandidateBitForTask(
                    stats, worker_id, RoleForWorker(worker_id),
                    task_id
                );
            }
        }
        all_workers_ok &= observer_workers == 2 &&
            total_control_loads == 2;
    }
    Check(
        all_workers_ok && NoFatal(*state),
        kTest,
        "each kernel task has two observers and 94 zero-load workers"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestBuildOwnerCandidateExclusion() {
    constexpr const char *kTest = "build-owner-candidate-exclusion";
    constexpr uint32_t kTask = 1;
    constexpr TaskKind kKind = TaskKind::Qk;
    PaExecRoute route{};
    uint32_t primary = kExecUnboundOwner;
    uint32_t secondary = kExecUnboundOwner;
    uint32_t outside = kExecUnboundOwner;
    bool all_ok = ResolvePaExecRoute(
                      kKind, FunctionId(kKind), route
                  ) &&
        FixedPaExecuteCandidates(
            kTask, route.engine_class, primary, secondary
        ) &&
        FindSameEngineNonCandidate(
            route.engine_class, primary, secondary, outside
        );

    for (uint32_t build_owner :
         std::array<uint32_t, 2>{primary, secondary}) {
        for (ExecPhase phase :
             std::array<ExecPhase, 2>{
                 ExecPhase::Building, ExecPhase::Built
             }) {
            MappedSchedulerState mapping;
            SchedulerState *state = mapping.Get();
            all_ok &= state != nullptr;
            if (state == nullptr) {
                continue;
            }
            if (phase == ExecPhase::Built) {
                all_ok &= PublishKernelCell(
                    *state, kTask, build_owner, kKind
                );
            } else {
                SetCellState(
                    *state, kTask, ExecPhase::Building,
                    build_owner, kExecUnboundOwner,
                    ExecEngineClass::None, 0
                );
            }

            WorkerState &builder = PrepareWorker(
                *state, build_owner, RoleForWorker(build_owner)
            );
            LocalStats builder_stats{};
            InitLocalStats(
                builder_stats, build_owner,
                RoleForWorker(build_owner)
            );
            all_ok &= RegisterLocalCandidate(
                builder_stats, kTask, kKind
            );
            ExecScanTestOps::ResetObservations();
            ExecScanTestOps::WatchControl(
                &state->exec_cells[kTask].control.state
            );
            const uint32_t builder_progress =
                ProgressCrossCoreExec<ExecScanTestOps>(
                    state, builder, kTask + 1U,
                    /*production_closed=*/false,
                    DrainPlace::EfDrain, builder_stats
                );
            all_ok &= builder_progress == 0 &&
                !CandidateBitForTask(
                    builder_stats, build_owner,
                    RoleForWorker(build_owner), kTask
                ) &&
                ExecScanTestOps::watched_control_loads == 1 &&
                ExecScanTestOps::watched_control_cas_calls == 0 &&
                ExecScanTestOps::execute_calls == 0 && NoFatal(*state);

            const uint32_t other =
                build_owner == primary ? secondary : primary;
            WorkerState &other_worker = PrepareWorker(
                *state, other, RoleForWorker(other)
            );
            LocalStats other_stats{};
            InitLocalStats(
                other_stats, other, RoleForWorker(other)
            );
            all_ok &= RegisterLocalCandidate(
                other_stats, kTask, kKind
            );
            ExecScanTestOps::ResetObservations();
            const uint32_t other_progress =
                ProgressCrossCoreExec<ExecScanTestOps>(
                    state, other_worker, kTask + 1U,
                    /*production_closed=*/false,
                    DrainPlace::EfDrain, other_stats
                );
            if (phase == ExecPhase::Building) {
                all_ok &= other_progress == 0 &&
                    CandidateBitForTask(
                        other_stats, other,
                        RoleForWorker(other), kTask
                    ) &&
                    ExecScanTestOps::execute_calls == 0;
            } else {
                const DecodedExecState done = DecodeExecState(
                    state->exec_cells[kTask].control.state
                );
                all_ok &= other_progress == 1 &&
                    !CandidateBitForTask(
                        other_stats, other,
                        RoleForWorker(other), kTask
                    ) &&
                    ExecScanTestOps::execute_calls == 1 &&
                    done.valid && done.phase == ExecPhase::Done &&
                    done.build_owner == build_owner &&
                    done.execute_owner == other;
            }
            all_ok &= NoFatal(*state);
        }
    }

    // Build owner 不在执行双候选内时，BUILDING 尚未产生可领取包；两个
    // 候选都必须保留各自 bit，不能提前替对方决定最终 winner。
    {
        MappedSchedulerState mapping;
        SchedulerState *state = mapping.Get();
        all_ok &= state != nullptr;
        if (state != nullptr) {
            SetCellState(
                *state, kTask, ExecPhase::Building,
                outside, kExecUnboundOwner,
                ExecEngineClass::None, 0
            );
            for (uint32_t candidate :
                 std::array<uint32_t, 2>{primary, secondary}) {
                WorkerState &worker = PrepareWorker(
                    *state, candidate, RoleForWorker(candidate)
                );
                LocalStats stats{};
                InitLocalStats(
                    stats, candidate, RoleForWorker(candidate)
                );
                all_ok &= RegisterLocalCandidate(
                    stats, kTask, kKind
                );
                ExecScanTestOps::ResetObservations();
                const uint32_t progressed =
                    ProgressCrossCoreExec<ExecScanTestOps>(
                        state, worker, kTask + 1U,
                        /*production_closed=*/false,
                        DrainPlace::EfDrain, stats
                    );
                all_ok &= progressed == 0 &&
                    CandidateBitForTask(
                        stats, candidate,
                        RoleForWorker(candidate), kTask
                    ) &&
                    ExecScanTestOps::execute_calls == 0 && NoFatal(*state);
            }
        }
    }

    Check(
        all_ok, kTest,
        "builder candidate skips; every remaining eligible observer stays live"
    );
    std::printf("[PASS] %s\n", kTest);
}

void RunDynamicCandidateFirstCase(
    bool primary_first, const char *test
) {
    constexpr uint32_t kTask = 1;
    constexpr TaskKind kKind = TaskKind::Qk;
    PaExecRoute route{};
    uint32_t primary = kExecUnboundOwner;
    uint32_t secondary = kExecUnboundOwner;
    uint32_t build_owner = kExecUnboundOwner;
    bool all_ok = ResolvePaExecRoute(
                      kKind, FunctionId(kKind), route
                  ) &&
        FixedPaExecuteCandidates(
            kTask, route.engine_class, primary, secondary
        ) &&
        FindSameEngineNonCandidate(
            route.engine_class, primary, secondary, build_owner
        );
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    all_ok &= state != nullptr;
    if (state == nullptr) {
        Check(false, test, "state mapping");
        return;
    }
    all_ok &= PublishKernelCell(
        *state, kTask, build_owner, kKind
    );

    const uint32_t first = primary_first ? primary : secondary;
    const uint32_t second = primary_first ? secondary : primary;
    WorkerState &first_worker = PrepareWorker(
        *state, first, RoleForWorker(first)
    );
    WorkerState &second_worker = PrepareWorker(
        *state, second, RoleForWorker(second)
    );
    LocalStats first_stats{};
    LocalStats second_stats{};
    InitLocalStats(first_stats, first, RoleForWorker(first));
    InitLocalStats(second_stats, second, RoleForWorker(second));
    all_ok &= RegisterLocalCandidate(first_stats, kTask, kKind) &&
        RegisterLocalCandidate(second_stats, kTask, kKind);

    ExecScanTestOps::ResetObservations();
    const uint32_t won = ProgressCrossCoreExec<ExecScanTestOps>(
        state, first_worker, kTask + 1U,
        /*production_closed=*/false,
        DrainPlace::EfDrain, first_stats
    );
    const uint32_t observed = ProgressCrossCoreExec<ExecScanTestOps>(
        state, second_worker, kTask + 1U,
        /*production_closed=*/false,
        DrainPlace::EfDrain, second_stats
    );
    const DecodedExecState done = DecodeExecState(
        state->exec_cells[kTask].control.state
    );
    all_ok &= won == 1 && observed == 0 &&
        CrossCoreExecCandidateBitmapEmpty(first_stats) &&
        CrossCoreExecCandidateBitmapEmpty(second_stats) &&
        ExecScanTestOps::execute_calls == 1 &&
        ExecScanTestOps::executed_tasks[0] == kTask &&
        done.valid && done.phase == ExecPhase::Done &&
        done.build_owner == build_owner &&
        done.execute_owner == first && state->tasks[kTask].flag == 1 &&
        NoFatal(*state);
    Check(
        all_ok, test,
        "first legal observer wins once and the other advances normally"
    );
    std::printf("[PASS] %s\n", test);
}

void TestEitherCandidateMayWin() {
    RunDynamicCandidateFirstCase(
        /*primary_first=*/true, "dynamic-primary-wins"
    );
    RunDynamicCandidateFirstCase(
        /*primary_first=*/false, "dynamic-secondary-wins"
    );
}

void TestMappedEmptyDelayedPublication() {
    constexpr const char *kTest = "mapped-empty-delayed-publish";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;
    uint32_t primary = 0;
    uint32_t secondary = 0;
    uint32_t target = 0;
    Check(
        SelectTestExecutor(
            1, TaskKind::Qk, 1, primary, secondary, target
        ) && target == secondary,
        kTest, "resolve delayed QK target"
    );
    WorkerState &worker = PrepareWorker(
        *state, target, RoleForWorker(target)
    );
    LocalStats stats{};
    InitLocalStats(stats, target, RoleForWorker(target));
    Check(
        RegisterLocalCandidate(stats, 1, TaskKind::Qk),
        kTest, "register delayed QK candidate"
    );
    ExecScanTestOps::ResetObservations();
    const uint32_t empty_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 2, /*production_closed=*/false,
            DrainPlace::EfDrain, stats
        );
    Check(
        empty_progress == 0 &&
            CandidateBitForTask(
                stats, target, RoleForWorker(target), 1
            ) &&
            ExecScanTestOps::execute_calls == 0 && NoFatal(*state),
        kTest, "mapped candidate waits at EMPTY"
    );

    Check(
        PublishKernelCell(*state, 1, 1, TaskKind::Qk),
        kTest, "builder publishes delayed QK"
    );
    const uint32_t built_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 2, /*production_closed=*/false,
            DrainPlace::EfDrain, stats
        );
    const DecodedExecState done = DecodeExecState(
        state->exec_cells[1].control.state
    );
    Check(
        built_progress == 1 &&
            !CandidateBitForTask(
                stats, target, RoleForWorker(target), 1
            ) &&
            ExecScanTestOps::execute_calls == 1 &&
            done.valid && done.phase == ExecPhase::Done &&
            done.build_owner == 1 && done.execute_owner == target &&
            state->tasks[1].flag == 1 &&
            CrossCoreExecWorkerDrained(
                state, worker, 2, stats
            ) && NoFatal(*state),
        kTest, "delayed BUILT is claimed exactly once"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestMappedBuildingDelayedPublication() {
    constexpr const char *kTest = "mapped-building-delayed-publish";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;
    state->tasks[1].flag = 1;

    uint32_t primary = 0;
    uint32_t secondary = 0;
    uint32_t target = 0;
    Check(
        SelectTestExecutor(
            2, TaskKind::Sf, 34,
            primary, secondary, target
        ) && target == secondary,
        kTest, "resolve delayed SF target"
    );
    Check(
        PublishKernelCell(
            *state, 2, 34, TaskKind::Sf, {1}
        ),
        kTest, "prepare delayed SF payload"
    );
    const int64_t built_state =
        state->exec_cells[2].control.state;
    SetCellState(
        *state, 2, ExecPhase::Building, 34,
        kExecUnboundOwner, ExecEngineClass::None, 0
    );

    WorkerState &target_worker = PrepareWorker(
        *state, target, RoleForWorker(target)
    );
    LocalStats target_stats{};
    InitLocalStats(
        target_stats, target, RoleForWorker(target)
    );
    Check(
        RegisterLocalCandidate(
            target_stats, 2, TaskKind::Sf
        ),
        kTest, "register target SF candidate"
    );
    ExecScanTestOps::ResetObservations();
    const uint32_t waiting =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, target_worker, 3,
            /*production_closed=*/false,
            DrainPlace::EfDrain, target_stats
        );
    Check(
        waiting == 0 &&
            CandidateBitForTask(
                target_stats, target, RoleForWorker(target), 2
            ) &&
            ExecScanTestOps::execute_calls == 0 && NoFatal(*state),
        kTest, "actual target waits at BUILDING"
    );

    WorkerState &other_worker = PrepareWorker(
        *state, primary, RoleForWorker(primary)
    );
    LocalStats other_stats{};
    InitLocalStats(
        other_stats, primary, RoleForWorker(primary)
    );
    Check(
        RegisterLocalCandidate(
            other_stats, 2, TaskKind::Sf
        ),
        kTest, "register alternate SF candidate"
    );
    const uint32_t skipped =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, other_worker, 3,
            /*production_closed=*/false,
            DrainPlace::EfDrain, other_stats
        );
    Check(
        skipped == 0 &&
            !CandidateBitForTask(
                other_stats, primary, RoleForWorker(primary), 2
            ) &&
            NoFatal(*state),
        kTest, "other candidate skips known-target BUILDING"
    );

    state->exec_cells[2].control.state = built_state;
    const uint32_t published =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, target_worker, 3,
            /*production_closed=*/false,
            DrainPlace::EfDrain, target_stats
        );
    const DecodedExecState done = DecodeExecState(
        state->exec_cells[2].control.state
    );
    Check(
        published == 1 &&
            !CandidateBitForTask(
                target_stats, target, RoleForWorker(target), 2
            ) &&
            ExecScanTestOps::execute_calls == 1 &&
            done.valid && done.phase == ExecPhase::Done &&
            done.build_owner == 34 &&
            done.execute_owner == target && NoFatal(*state),
        kTest, "target observes later BUILT without losing the task"
    );
    std::printf("[PASS] %s\n", kTest);
}

void RunOtherCandidateTerminalStateCase(
    ExecPhase phase, bool primary_won, const char *test
) {
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, test, "state mapping");
    if (state == nullptr) return;
    uint32_t primary = 0;
    uint32_t secondary = 0;
    uint32_t selected = 0;
    Check(
        SelectTestExecutor(
            1, TaskKind::Qk, 3,
            primary, secondary, selected
        ) && selected == primary,
        test, "resolve QK candidates"
    );
    const uint32_t winner = primary_won ? primary : secondary;
    const uint32_t observer = primary_won ? secondary : primary;
    SetCellState(
        *state, 1, phase, /*build_owner=*/3, winner,
        ExecEngineClass::Aic, /*payload_lines=*/1
    );
    WorkerState &worker = PrepareWorker(
        *state, observer, CoreRole::Aic
    );
    LocalStats stats{};
    InitLocalStats(stats, observer, CoreRole::Aic);
    Check(
        RegisterLocalCandidate(stats, 1, TaskKind::Qk),
        test, "register alternate QK candidate"
    );
    ExecScanTestOps::ResetObservations();
    ExecScanTestOps::WatchControl(
        &state->exec_cells[1].control.state
    );
    const uint32_t progressed =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 2, /*production_closed=*/false,
            DrainPlace::EfDrain, stats
        );
    Check(
        progressed == 0 &&
            !CandidateBitForTask(
                stats, observer, CoreRole::Aic, 1
            ) &&
            ExecScanTestOps::watched_control_loads == 1 &&
            ExecScanTestOps::execute_calls == 0 && NoFatal(*state),
        test, "other legal candidate validates the dynamic owner and advances"
    );
    std::printf("[PASS] %s\n", test);
}

void TestOtherCandidateSkipsClaimedAndDone() {
    RunOtherCandidateTerminalStateCase(
        ExecPhase::Claimed, true, "claimed-accepts-primary-owner"
    );
    RunOtherCandidateTerminalStateCase(
        ExecPhase::Claimed, false, "claimed-accepts-secondary-owner"
    );
    RunOtherCandidateTerminalStateCase(
        ExecPhase::Done, true, "done-accepts-primary-owner"
    );
    RunOtherCandidateTerminalStateCase(
        ExecPhase::Done, false, "done-accepts-secondary-owner"
    );
}

void TestDeterministicClaimLossIsNormal() {
    constexpr const char *kTest = "deterministic-claim-loss";
    constexpr uint32_t kTask = 1;
    uint32_t primary = kExecUnboundOwner;
    uint32_t secondary = kExecUnboundOwner;
    uint32_t selected = kExecUnboundOwner;
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;
    bool all_ok = SelectTestExecutor(
                      kTask, TaskKind::Qk, 3,
                      primary, secondary, selected
                  ) && selected == primary &&
        PublishKernelCell(*state, kTask, 3, TaskKind::Qk);
    const DecodedExecState built = DecodeExecState(
        state->exec_cells[kTask].control.state
    );
    WorkerState &worker = PrepareWorker(
        *state, primary, CoreRole::Aic
    );
    LocalStats stats{};
    InitLocalStats(stats, primary, CoreRole::Aic);
    all_ok &= RegisterLocalCandidate(stats, kTask, TaskKind::Qk);
    ExecScanTestOps::ResetObservations();
    ExecScanTestOps::WatchControl(
        &state->exec_cells[kTask].control.state
    );
    ExecScanTestOps::InjectClaimLoss(
        &state->exec_cells[kTask].control.state,
        static_cast<int64_t>(EncodeExecState(
            ExecPhase::Claimed, 3, secondary,
            ExecEngineClass::Aic, built.payload_lines, kTask
        ))
    );
    const uint32_t progressed =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, kTask + 1U,
            /*production_closed=*/false,
            DrainPlace::EfDrain, stats
        );
    const DecodedExecState claimed = DecodeExecState(
        state->exec_cells[kTask].control.state
    );
    all_ok &= progressed == 0 &&
        CrossCoreExecCandidateBitmapEmpty(stats) &&
        CrossCoreExecTokenFullyReset(state->exec_tokens[primary]) &&
        ExecScanTestOps::watched_control_cas_calls == 1 &&
        ExecScanTestOps::invalidate_calls == 0 &&
        ExecScanTestOps::payload_loads == 0 &&
        ExecScanTestOps::execute_calls == 0 &&
        claimed.valid && claimed.phase == ExecPhase::Claimed &&
        claimed.execute_owner == secondary && NoFatal(*state);
    Check(
        all_ok, kTest,
        "CAS Lost clears the loser candidate without touching payload"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestInvalidStatesFailClosed() {
    constexpr const char *kTest = "invalid-state-fail-closed";
    bool all_cases_ok = true;

    {
        MappedSchedulerState mapping;
        SchedulerState *state = mapping.Get();
        all_cases_ok &= state != nullptr;
        if (state != nullptr) {
            SetCellState(
                *state, 1, ExecPhase::Built, 1,
                kExecUnboundOwner, ExecEngineClass::Aiv, 1
            );
            WorkerState &worker = PrepareWorker(
                *state, 1, CoreRole::Aic
            );
            LocalStats stats{};
            InitLocalStats(stats, 1, CoreRole::Aic);
            all_cases_ok &= RegisterLocalCandidate(
                stats, 1, TaskKind::Qk
            );
            (void)ProgressCrossCoreExec<ExecScanTestOps>(
                state, worker, 2, false,
                DrainPlace::EfDrain, stats
            );
            all_cases_ok &= FatalMatches(
                *state, ExecFatalReason::InvalidBuiltControl, 1, 1
            );
        }
    }
    {
        MappedSchedulerState mapping;
        SchedulerState *state = mapping.Get();
        all_cases_ok &= state != nullptr;
        if (state != nullptr) {
            SetCellState(
                *state, 1, ExecPhase::Done, 1, 1,
                ExecEngineClass::Aic, 1
            );
            WorkerState &worker = PrepareWorker(
                *state, 1, CoreRole::Aic
            );
            LocalStats stats{};
            InitLocalStats(stats, 1, CoreRole::Aic);
            all_cases_ok &= RegisterLocalCandidate(
                stats, 1, TaskKind::Qk
            );
            (void)ProgressCrossCoreExec<ExecScanTestOps>(
                state, worker, 2, false,
                DrainPlace::EfDrain, stats
            );
            all_cases_ok &= FatalMatches(
                *state, ExecFatalReason::InvalidBuiltControl, 1, 1
            );
        }
    }
    {
        MappedSchedulerState mapping;
        SchedulerState *state = mapping.Get();
        all_cases_ok &= state != nullptr;
        if (state != nullptr) {
            SetCellState(
                *state, 1, ExecPhase::Done, 1, 2,
                ExecEngineClass::Aic, 1
            );
            WorkerState &worker = PrepareWorker(
                *state, 2, CoreRole::Aic
            );
            LocalStats stats{};
            InitLocalStats(stats, 2, CoreRole::Aic);
            all_cases_ok &= RegisterLocalCandidate(
                stats, 1, TaskKind::Qk
            );
            (void)ProgressCrossCoreExec<ExecScanTestOps>(
                state, worker, 2, false,
                DrainPlace::EfDrain, stats
            );
            all_cases_ok &= FatalMatches(
                *state, ExecFatalReason::InvalidBuiltControl, 1, 2
            );
        }
    }
    {
        MappedSchedulerState mapping;
        SchedulerState *state = mapping.Get();
        all_cases_ok &= state != nullptr;
        if (state != nullptr) {
            SetCellState(
                *state, 1, ExecPhase::Claimed, 1, 2,
                ExecEngineClass::Aic, 1
            );
            WorkerState &worker = PrepareWorker(
                *state, 2, CoreRole::Aic
            );
            LocalStats stats{};
            InitLocalStats(stats, 2, CoreRole::Aic);
            all_cases_ok &= RegisterLocalCandidate(
                stats, 1, TaskKind::Qk
            );
            (void)ProgressCrossCoreExec<ExecScanTestOps>(
                state, worker, 2, false,
                DrainPlace::EfDrain, stats
            );
            all_cases_ok &= FatalMatches(
                *state, ExecFatalReason::InvalidBuiltControl, 1, 2
            );
        }
    }
    {
        MappedSchedulerState mapping;
        SchedulerState *state = mapping.Get();
        all_cases_ok &= state != nullptr;
        if (state != nullptr) {
            WorkerState &worker = PrepareWorker(
                *state, 1, CoreRole::Aic
            );
            LocalStats stats{};
            InitLocalStats(stats, 1, CoreRole::Aic);
            all_cases_ok &= RegisterLocalCandidate(
                stats, 1, TaskKind::Qk
            );
            (void)ProgressCrossCoreExec<ExecScanTestOps>(
                state, worker, 2, true,
                DrainPlace::FinalDrain, stats
            );
            all_cases_ok &= FatalMatches(
                *state, ExecFatalReason::InvalidBuiltControl, 1, 1
            );
        }
    }
    {
        MappedSchedulerState mapping;
        SchedulerState *state = mapping.Get();
        all_cases_ok &= state != nullptr;
        if (state != nullptr) {
            SetCellState(
                *state, 1, ExecPhase::Building, 1,
                kExecUnboundOwner, ExecEngineClass::None, 0
            );
            WorkerState &worker = PrepareWorker(
                *state, 2, CoreRole::Aic
            );
            LocalStats stats{};
            InitLocalStats(stats, 2, CoreRole::Aic);
            all_cases_ok &= RegisterLocalCandidate(
                stats, 1, TaskKind::Qk
            );
            (void)ProgressCrossCoreExec<ExecScanTestOps>(
                state, worker, 2, true,
                DrainPlace::FinalDrain, stats
            );
            all_cases_ok &= FatalMatches(
                *state, ExecFatalReason::InvalidBuiltControl, 1, 2
            );
        }
    }
    Check(
        all_cases_ok, kTest,
        "wrong engine/mapping, lost target cursor and closed holes fail"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestExistingGlobalFatalDoesNotFabricateExecFatal() {
    constexpr const char *kTest =
        "existing-global-fatal-preserves-first-failure";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;

    constexpr uint32_t kTask = 1;
    constexpr uint32_t kBuilder = 1;
    WorkerState &worker = PrepareWorker(
        *state, kBuilder, CoreRole::Aic
    );
    LocalStats stats{};
    InitLocalStats(stats, kBuilder, CoreRole::Aic);
    TaskArgs args{};
    SubmitContext context{};
    __atomic_store_n(
        &state->fatal.value, int32_t{1}, __ATOMIC_RELEASE
    );
    ExecScanTestOps::ResetObservations();
    ExecScanTestOps::WatchControl(
        &state->exec_cells[kTask].control.state
    );

    const bool published =
        PublishCrossCoreExecTask<ExecScanTestOps>(
            state, worker, kTask, TaskKind::Qk,
            FunctionId(TaskKind::Qk), args, context, stats
        );
    const DecodedExecState cell = DecodeExecState(
        state->exec_cells[kTask].control.state
    );
    Check(
        !published && state->fatal.value == 1 &&
            state->exec_fatal.state == 0 &&
            cell.valid && cell.phase == ExecPhase::Empty &&
            ExecScanTestOps::watched_control_loads == 0 &&
            ExecScanTestOps::watched_control_cas_calls == 0 &&
            stats.result.slot_tensor_copies == 0 &&
            stats.result.slot_scalar_copies == 0 &&
            stats.result.fanin_edges == 0,
        kTest,
        "pre-existing scheduler fatal stops Build without fake exec fatal"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestGlobalFatalFaultsActiveTokens() {
    constexpr const char *kTest =
        "global-fatal-faults-active-token";
    bool all_cases_ok = true;

    // WAITING_FANIN 必须由真实 Claim/Bind 路径产生；producer 1 尚未
    // ready，使第一次 progress 确定停在该阶段。
    {
        MappedSchedulerState mapping;
        SchedulerState *state = mapping.Get();
        all_cases_ok &= state != nullptr;
        if (state != nullptr) {
            constexpr uint32_t kTask = 2;
            constexpr uint32_t kBuilder = 34;
            uint32_t primary = 0;
            uint32_t secondary = 0;
            uint32_t executor = 0;
            all_cases_ok &= SelectTestExecutor(
                kTask, TaskKind::Sf, kBuilder,
                primary, secondary, executor
            ) && PublishKernelCell(
                *state, kTask, kBuilder, TaskKind::Sf, {1}
            );
            WorkerState &worker = PrepareWorker(
                *state, executor, CoreRole::Aiv
            );
            LocalStats stats{};
            InitLocalStats(stats, executor, CoreRole::Aiv);
            all_cases_ok &= RegisterLocalCandidate(
                stats, kTask, TaskKind::Sf
            );
            ExecScanTestOps::ResetObservations();
            (void)ProgressCrossCoreExec<ExecScanTestOps>(
                state, worker, kTask + 1U, false,
                DrainPlace::EfDrain, stats
            );
            all_cases_ok &=
                state->exec_tokens[executor].control.phase ==
                    ExecTokenPhase::WaitingFanin &&
                DecodeExecState(
                    state->exec_cells[kTask].control.state
                ).phase == ExecPhase::Claimed &&
                ExecScanTestOps::execute_calls == 0 && NoFatal(*state);

            __atomic_store_n(
                &state->fatal.value, int32_t{1}, __ATOMIC_RELEASE
            );
            const uint32_t progressed =
                ProgressCrossCoreExec<ExecScanTestOps>(
                    state, worker, kTask + 1U, false,
                    DrainPlace::EfDrain, stats
                );
            all_cases_ok &= progressed == 0 &&
                state->exec_tokens[executor].control.phase ==
                    ExecTokenPhase::Faulted &&
                DecodeExecState(
                    state->exec_cells[kTask].control.state
                ).phase == ExecPhase::Claimed &&
                state->tasks[kTask].vend == 0 &&
                state->tasks[kTask].flag == 0 &&
                state->exec_fatal.state == 0 &&
                ExecScanTestOps::execute_calls == 0;
        }
    }

    // COMPLETING 由完整的 Claim/Bind/ready/kernel/engine-complete helper
    // 链构造；测试只把 global fatal 放在下一次 completion progress 前。
    {
        MappedSchedulerState mapping;
        SchedulerState *state = mapping.Get();
        all_cases_ok &= state != nullptr;
        if (state != nullptr) {
            constexpr uint32_t kTask = 1;
            constexpr uint32_t kBuilder = 1;
            uint32_t primary = 0;
            uint32_t secondary = 0;
            uint32_t executor = 0;
            all_cases_ok &= SelectTestExecutor(
                kTask, TaskKind::Qk, kBuilder,
                primary, secondary, executor
            ) && PublishKernelCell(
                *state, kTask, kBuilder, TaskKind::Qk
            );
            WorkerState &worker = PrepareWorker(
                *state, executor, CoreRole::Aic
            );
            LocalStats stats{};
            InitLocalStats(stats, executor, CoreRole::Aic);
            ExecutionToken &token = state->exec_tokens[executor];
            const ExecClaimResult claim =
                ClaimAndBindExecPayload<ExecScanTestOps>(
                    state->exec_cells[kTask], kTask, executor,
                    ExecEngineClass::Aic, token, state->exec_fatal
                );
            PaExecReadySource<ExecScanTestOps> ready{
                state, &stats.result
            };
            const PaExecSynchronousEngineCompletion engine_complete{};
            all_cases_ok &= claim == ExecClaimResult::Claimed &&
                BindPaExecutionTokenDispatchAfterClaim(token, worker) &&
                TryMarkExecutionTokenEngineInflight<ExecScanTestOps>(
                    token, ready, state->exec_fatal
                ) &&
                ExecutePaBoundKernel<ExecScanTestOps>(
                    state, worker, token, TaskKind::Qk, 0
                ) &&
                TryMarkExecutionTokenCompleting<ExecScanTestOps>(
                    token, engine_complete, state->exec_fatal
                ) &&
                token.control.phase == ExecTokenPhase::Completing;

            ExecScanTestOps::ResetObservations();
            __atomic_store_n(
                &state->fatal.value, int32_t{1}, __ATOMIC_RELEASE
            );
            const uint32_t progressed =
                ProgressCrossCoreExec<ExecScanTestOps>(
                    state, worker, kTask + 1U, false,
                    DrainPlace::EfDrain, stats
                );
            all_cases_ok &= progressed == 0 &&
                token.control.phase == ExecTokenPhase::Faulted &&
                DecodeExecState(
                    state->exec_cells[kTask].control.state
                ).phase == ExecPhase::Claimed &&
                state->tasks[kTask].vend == 0 &&
                state->tasks[kTask].flag == 0 &&
                state->exec_fatal.state == 0 &&
                ExecScanTestOps::execute_calls == 0 &&
                stats.result.placement[
                    static_cast<uint32_t>(DrainPlace::EfDrain)
                ] == 0;
        }
    }

    Check(
        all_cases_ok, kTest,
        "WAITING_FANIN/COMPLETING converge to Faulted without completion"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestGlobalFatalStopsIrreversibleBoundaries() {
    constexpr const char *kTest =
        "global-fatal-stops-irreversible-boundaries";
    bool all_cases_ok = true;

    // scanner 已取得 BUILT 快照后立刻注入 global fatal。下一步必须在
    // Claim CAS 前停住；保留 BUILT 和候选 bit 是可重复观察的直接证据。
    {
        MappedSchedulerState mapping;
        SchedulerState *state = mapping.Get();
        all_cases_ok &= state != nullptr;
        if (state != nullptr) {
            constexpr uint32_t kTask = 1;
            constexpr uint32_t kBuilder = 1;
            uint32_t primary = 0;
            uint32_t secondary = 0;
            uint32_t executor = 0;
            all_cases_ok &= SelectTestExecutor(
                kTask, TaskKind::Qk, kBuilder,
                primary, secondary, executor
            ) && PublishKernelCell(
                *state, kTask, kBuilder, TaskKind::Qk
            );
            WorkerState &worker = PrepareWorker(
                *state, executor, CoreRole::Aic
            );
            LocalStats stats{};
            InitLocalStats(stats, executor, CoreRole::Aic);
            all_cases_ok &= RegisterLocalCandidate(
                stats, kTask, TaskKind::Qk
            );
            ExecScanTestOps::ResetObservations();
            ExecScanTestOps::WatchControl(
                &state->exec_cells[kTask].control.state
            );
            ExecScanTestOps::InjectGlobalFatalAfterLoad(
                &state->exec_cells[kTask].control.state,
                &state->fatal.value
            );
            const uint32_t progressed =
                ProgressCrossCoreExec<ExecScanTestOps>(
                    state, worker, kTask + 1U, false,
                    DrainPlace::EfDrain, stats
                );
            all_cases_ok &= progressed == 0 &&
                state->fatal.value == 1 &&
                state->exec_fatal.state == 0 &&
                DecodeExecState(
                    state->exec_cells[kTask].control.state
                ).phase == ExecPhase::Built &&
                state->exec_tokens[executor].control.phase ==
                    ExecTokenPhase::Idle &&
                CandidateBitForTask(
                    stats, executor, CoreRole::Aic, kTask
                ) &&
                ExecScanTestOps::watched_control_loads == 1 &&
                ExecScanTestOps::watched_control_cas_calls == 0 &&
                ExecScanTestOps::execute_calls == 0 &&
                state->tasks[kTask].vend == 0 &&
                state->tasks[kTask].flag == 0;
        }
    }

    // fanin 的最后一次 ready Load 返回后注入 fatal，精确覆盖
    // WAITING_FANIN -> ENGINE_INFLIGHT 与真实 kernel 发射之间的窗口。
    {
        MappedSchedulerState mapping;
        SchedulerState *state = mapping.Get();
        all_cases_ok &= state != nullptr;
        if (state != nullptr) {
            constexpr uint32_t kTask = 2;
            constexpr uint32_t kBuilder = 34;
            uint32_t primary = 0;
            uint32_t secondary = 0;
            uint32_t executor = 0;
            all_cases_ok &= SelectTestExecutor(
                kTask, TaskKind::Sf, kBuilder,
                primary, secondary, executor
            ) && PublishKernelCell(
                *state, kTask, kBuilder, TaskKind::Sf, {1}
            );
            __atomic_store_n(
                &state->tasks[1].flag, int64_t{1}, __ATOMIC_RELEASE
            );
            WorkerState &worker = PrepareWorker(
                *state, executor, CoreRole::Aiv
            );
            LocalStats stats{};
            InitLocalStats(stats, executor, CoreRole::Aiv);
            all_cases_ok &= RegisterLocalCandidate(
                stats, kTask, TaskKind::Sf
            );
            ExecScanTestOps::ResetObservations();
            ExecScanTestOps::WatchControl(
                &state->exec_cells[kTask].control.state
            );
            ExecScanTestOps::InjectGlobalFatalAfterLoad(
                &state->tasks[1].flag, &state->fatal.value
            );
            const uint32_t progressed =
                ProgressCrossCoreExec<ExecScanTestOps>(
                    state, worker, kTask + 1U, false,
                    DrainPlace::EfDrain, stats
                );
            all_cases_ok &= progressed == 0 &&
                state->fatal.value == 1 &&
                state->exec_fatal.state == 0 &&
                DecodeExecState(
                    state->exec_cells[kTask].control.state
                ).phase == ExecPhase::Claimed &&
                state->exec_tokens[executor].control.phase ==
                    ExecTokenPhase::Faulted &&
                ExecScanTestOps::watched_control_cas_calls == 1 &&
                ExecScanTestOps::execute_calls == 0 &&
                state->tasks[kTask].vend == 0 &&
                state->tasks[kTask].flag == 0 &&
                stats.result.placement[
                    static_cast<uint32_t>(DrainPlace::EfDrain)
                ] == 0;
        }
    }

    // ExecuteBoundKernel 是当前同步 engine 边界；在它返回时注入 fatal，
    // 必须保留 CLAIMED 且不发布 vend、flag 或 DONE。
    {
        MappedSchedulerState mapping;
        SchedulerState *state = mapping.Get();
        all_cases_ok &= state != nullptr;
        if (state != nullptr) {
            constexpr uint32_t kTask = 1;
            constexpr uint32_t kBuilder = 1;
            uint32_t primary = 0;
            uint32_t secondary = 0;
            uint32_t executor = 0;
            all_cases_ok &= SelectTestExecutor(
                kTask, TaskKind::Qk, kBuilder,
                primary, secondary, executor
            ) && PublishKernelCell(
                *state, kTask, kBuilder, TaskKind::Qk
            );
            WorkerState &worker = PrepareWorker(
                *state, executor, CoreRole::Aic
            );
            LocalStats stats{};
            InitLocalStats(stats, executor, CoreRole::Aic);
            all_cases_ok &= RegisterLocalCandidate(
                stats, kTask, TaskKind::Qk
            );
            ExecScanTestOps::ResetObservations();
            ExecScanTestOps::WatchControl(
                &state->exec_cells[kTask].control.state
            );
            ExecScanTestOps::InjectGlobalFatalAfterExecute(
                &state->fatal.value
            );
            const uint32_t progressed =
                ProgressCrossCoreExec<ExecScanTestOps>(
                    state, worker, kTask + 1U, false,
                    DrainPlace::EfDrain, stats
                );
            all_cases_ok &= progressed == 0 &&
                state->fatal.value == 1 &&
                state->exec_fatal.state == 0 &&
                DecodeExecState(
                    state->exec_cells[kTask].control.state
                ).phase == ExecPhase::Claimed &&
                state->exec_tokens[executor].control.phase ==
                    ExecTokenPhase::Faulted &&
                ExecScanTestOps::watched_control_cas_calls == 1 &&
                ExecScanTestOps::execute_calls == 1 &&
                ExecScanTestOps::executed_tasks[0] == kTask &&
                state->tasks[kTask].vend == 0 &&
                state->tasks[kTask].flag == 0 &&
                stats.result.placement[
                    static_cast<uint32_t>(DrainPlace::EfDrain)
                ] == 0;
        }
    }

    Check(
        all_cases_ok, kTest,
        "fatal at Claim/kernel/completion gates leaves no later side effect"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestBusyTokenResumesScanning() {
    constexpr const char *kTest = "busy-token-resumes-scanning";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;
    state->tasks[0].flag = 1;
    Check(
        PublishKernelCell(
            *state, 2, 34, TaskKind::Sf, {1}
        ) &&
            SetTerminalKernelCell(
                *state, 3, TaskKind::Pv, 3
            ) &&
            PublishKernelCell(
                *state, 4, 40, TaskKind::Up, {2, 3, 0}
            ),
        kTest, "publish blocked SF and later UP"
    );
    const uint32_t executor = 36;
    WorkerState &worker = PrepareWorker(
        *state, executor, CoreRole::Aiv
    );
    LocalStats stats{};
    InitLocalStats(stats, executor, CoreRole::Aiv);
    Check(
        RegisterLocalCandidate(stats, 2, TaskKind::Sf) &&
            RegisterLocalCandidate(stats, 4, TaskKind::Up),
        kTest, "register both executor candidates"
    );
    ExecScanTestOps::ResetObservations();

    const uint32_t first =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 5, false,
            DrainPlace::EfDrain, stats
        );
    Check(
        first == 0 && stats.exec_candidate_slot == 1 &&
            CandidateBitForTask(
                stats, executor, CoreRole::Aiv, 4
            ) &&
            state->exec_tokens[executor].control.phase ==
                ExecTokenPhase::WaitingFanin &&
            DecodeExecState(
                state->exec_cells[2].control.state
            ).phase == ExecPhase::Claimed &&
            DecodeExecState(
                state->exec_cells[4].control.state
            ).phase == ExecPhase::Built &&
            ExecScanTestOps::execute_calls == 0 && NoFatal(*state),
        kTest, "busy token pauses before later candidate task"
    );
    const uint32_t still_blocked =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 5, false,
            DrainPlace::EfDrain, stats
        );
    Check(
        still_blocked == 0 && stats.exec_candidate_slot == 1 &&
            ExecScanTestOps::execute_calls == 0 && NoFatal(*state),
        kTest, "repeated fanin poll keeps candidate cursor"
    );

    __atomic_store_n(
        &state->tasks[1].flag, int64_t{1}, __ATOMIC_RELEASE
    );
    const uint32_t resumed =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 5, false,
            DrainPlace::EfDrain, stats
        );
    Check(
        resumed == 2 && stats.exec_candidate_slot == 2 &&
            CrossCoreExecCandidateBitmapEmpty(stats) &&
            state->exec_tokens[executor].control.phase ==
                ExecTokenPhase::Idle &&
            ExecScanTestOps::execute_calls == 2 &&
            ExecScanTestOps::executed_tasks[0] == 2 &&
            ExecScanTestOps::executed_tasks[1] == 4 &&
            DecodeExecState(
                state->exec_cells[2].control.state
            ).phase == ExecPhase::Done &&
            DecodeExecState(
                state->exec_cells[4].control.state
            ).phase == ExecPhase::Done &&
            NoFatal(*state),
        kTest, "ready token completes before later task is claimed"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestBusyCandidateDoesNotBlockPeerClaim() {
    constexpr const char *kTest = "busy-candidate-peer-claim";
    constexpr uint32_t kBusyTask = 3;
    constexpr uint32_t kNewTask = 35;
    uint32_t primary = kExecUnboundOwner;
    uint32_t secondary = kExecUnboundOwner;
    uint32_t selected = kExecUnboundOwner;
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;
    bool all_ok = SelectTestExecutor(
                      kNewTask, TaskKind::Qk, 5,
                      primary, secondary, selected
                  ) && selected == primary &&
        PublishKernelCell(
            *state, kBusyTask, 5, TaskKind::Pv, {1}
        ) &&
        PublishKernelCell(
            *state, kNewTask, 5, TaskKind::Qk
        );
    WorkerState &busy_worker = PrepareWorker(
        *state, primary, CoreRole::Aic
    );
    WorkerState &peer_worker = PrepareWorker(
        *state, secondary, CoreRole::Aic
    );
    ExecutionToken &busy_token = state->exec_tokens[primary];
    const ExecClaimResult busy_claim =
        ClaimAndBindExecPayload<ExecScanTestOps>(
            state->exec_cells[kBusyTask], kBusyTask, primary,
            ExecEngineClass::Aic, busy_token, state->exec_fatal
        );
    all_ok &= busy_claim == ExecClaimResult::Claimed &&
        BindPaExecutionTokenDispatchAfterClaim(
            busy_token, busy_worker
        ) &&
        busy_token.control.phase == ExecTokenPhase::WaitingFanin;

    LocalStats busy_stats{};
    LocalStats peer_stats{};
    InitLocalStats(busy_stats, primary, CoreRole::Aic);
    InitLocalStats(peer_stats, secondary, CoreRole::Aic);
    all_ok &= RegisterLocalCandidate(
                  busy_stats, kNewTask, TaskKind::Qk
              ) &&
        RegisterLocalCandidate(
            peer_stats, kNewTask, TaskKind::Qk
        );
    ExecScanTestOps::ResetObservations();
    ExecScanTestOps::WatchControl(
        &state->exec_cells[kNewTask].control.state
    );
    const uint32_t busy_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, busy_worker, kNewTask + 1U,
            /*production_closed=*/false,
            DrainPlace::EfDrain, busy_stats
        );
    const bool busy_did_not_touch_new_cell =
        busy_progress == 0 &&
        ExecScanTestOps::watched_control_loads == 0 &&
        ExecScanTestOps::watched_control_cas_calls == 0 &&
        CandidateBitForTask(
            busy_stats, primary, CoreRole::Aic, kNewTask
        );
    const uint32_t peer_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, peer_worker, kNewTask + 1U,
            /*production_closed=*/false,
            DrainPlace::EfDrain, peer_stats
        );
    const DecodedExecState done = DecodeExecState(
        state->exec_cells[kNewTask].control.state
    );
    all_ok &= busy_did_not_touch_new_cell && peer_progress == 1 &&
        CrossCoreExecCandidateBitmapEmpty(peer_stats) &&
        busy_token.control.phase == ExecTokenPhase::WaitingFanin &&
        ExecScanTestOps::watched_control_loads != 0 &&
        ExecScanTestOps::watched_control_cas_calls != 0 &&
        ExecScanTestOps::execute_calls == 1 &&
        done.valid && done.phase == ExecPhase::Done &&
        done.execute_owner == secondary && NoFatal(*state);
    Check(
        all_ok, kTest,
        "busy candidate performs zero new-cell access while peer executes"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestFinalDrainClosesLastTask() {
    constexpr const char *kTest = "final-drain-closes-last-task";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;
    state->config.batches = 1;
    state->context_lens[0] = 1;
    state->tasks[0].flag = 1;
    Check(
        SetTerminalKernelCell(
            *state, 1, TaskKind::Qk, 3,
            /*requested_execute_owner=*/2
        ) &&
            SetTerminalKernelCell(
                *state, 2, TaskKind::Sf, 38,
                /*requested_execute_owner=*/34
            ) &&
            SetTerminalKernelCell(
                *state, 3, TaskKind::Pv, 5,
                /*requested_execute_owner=*/4
            ) &&
            PublishKernelCell(
                *state, 4, 40, TaskKind::Up, {2, 3, 0}
            ),
        kTest, "prepare terminal prefix with either dynamic candidate"
    );

    const uint32_t executor = 36;
    WorkerState &worker = PrepareWorker(
        *state, executor, CoreRole::Aiv
    );
    LocalStats stats{};
    InitLocalStats(stats, executor, CoreRole::Aiv);
    Check(
        RegisterLocalCandidate(stats, 4, TaskKind::Up),
        kTest, "register the pending last task"
    );
    ExecScanTestOps::ResetObservations();
    uint32_t first_bad_task = UINT32_MAX;
    Check(
        !ValidateCrossCoreExecTerminalCells<ExecScanTestOps>(
            state, 5, first_bad_task
        ) && first_bad_task == 4 && NoFatal(*state),
        kTest, "open last task is not yet terminal"
    );

    const uint32_t final_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 5, /*production_closed=*/true,
            DrainPlace::FinalDrain, stats
        );
    first_bad_task = UINT32_MAX;
    Check(
        final_progress == 1 &&
            CrossCoreExecCandidateBitmapEmpty(stats) &&
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
                state->exec_tokens[executor]
            ) &&
            ValidateCrossCoreExecTerminalCells<ExecScanTestOps>(
                state, 5, first_bad_task
            ) &&
            first_bad_task == 5 && NoFatal(*state),
        kTest, "FinalDrain executes one dynamically eligible last-task owner"
    );

    std::array<bool, kWorkers> arrived{};
    std::array<LocalStats, kWorkers> arrival_stats{};
    bool all_arrivals_ok = true;
    for (uint32_t worker_id = 0;
         worker_id < kWorkers; ++worker_id) {
        WorkerState &arrival_worker = PrepareWorker(
            *state, worker_id, RoleForWorker(worker_id)
        );
        InitLocalStats(
            arrival_stats[worker_id], worker_id,
            RoleForWorker(worker_id)
        );
        // 汇合测试只需要把本核所有未登记的潜在槽交给生产 scanner
        // 越过；不能再伪造一个全局 task cursor。
        (void)ProgressCrossCoreExec<ExecScanTestOps>(
            state, arrival_worker, 5,
            /*production_closed=*/true,
            DrainPlace::FinalDrain, arrival_stats[worker_id]
        );
        bool released = false;
        const bool closure_ok =
            ProgressCrossCoreExecDrainClosure<ExecScanTestOps>(
                state, arrival_worker, 5,
                arrival_stats[worker_id],
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
        kTest, "last arrival validates dynamic legal terminal owners"
    );

    bool repeat_released = false;
    const bool repeat_ok =
        ProgressCrossCoreExecDrainClosure<ExecScanTestOps>(
            state, state->workers[0], 5, arrival_stats[0],
            arrived[0], repeat_released
        );
    Check(
        repeat_ok && repeat_released &&
            state->exec_drain.arrived ==
                static_cast<int64_t>(kWorkers),
        kTest, "one worker cannot increment drain twice"
    );
    std::printf("[PASS] %s\n", kTest);
}

}  // namespace

int main() {
    TestPotentialSlotRoundTrip();
    TestDynamicExecutorEligibility();
    TestSubmitCloseRegistersOnlyCandidates();
    TestCloseAndRegistrationRejectHistoryRewrite();
    TestOnlyTwoCandidatesObserveControl();
    TestBuildOwnerCandidateExclusion();
    TestEitherCandidateMayWin();
    TestMappedEmptyDelayedPublication();
    TestMappedBuildingDelayedPublication();
    TestOtherCandidateSkipsClaimedAndDone();
    TestDeterministicClaimLossIsNormal();
    TestInvalidStatesFailClosed();
    TestExistingGlobalFatalDoesNotFabricateExecFatal();
    TestGlobalFatalFaultsActiveTokens();
    TestGlobalFatalStopsIrreversibleBoundaries();
    TestBusyTokenResumesScanning();
    TestBusyCandidateDoesNotBlockPeerClaim();
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
