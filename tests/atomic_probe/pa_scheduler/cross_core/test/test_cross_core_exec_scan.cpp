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

constexpr uint32_t kAicBuildOwner = 3;
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

void EnsureDefaultDispatchPlan(SchedulerState &state) {
    if (state.build_dispatch.task_count != 0) {
        return;
    }
    constexpr uint32_t kPlanTasks =
        kMaxBatches * kTasksPerBatch;
    state.build_dispatch.task_count = kPlanTasks;
    state.build_dispatch.batch_count = kMaxBatches;
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
        identity.reserved = 0;
    }
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
    identity.reserved = 0;
    return identity.encoded_meta != 0;
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
    if (!SetDispatchTaskKind(state, task_id, kind)) {
        return false;
    }
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
    // 生产 owner 仲裁策略。builder 在 K2 外时选 primary，
    // builder 占用 primary 时选 secondary。
    execute_owner = build_owner == primary ? secondary : primary;
    return PaBuildOwnerEligible(build_owner, route.engine_class) &&
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
    if (!SetDispatchTaskKind(state, task_id, kind)) {
        return false;
    }
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

void TestOpportunisticProgressLocalGate() {
    constexpr const char *kTest =
        "opportunistic-progress-local-gate";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;

    constexpr uint32_t kWorker = 0;
    constexpr CoreRole kRole = CoreRole::Aic;
    WorkerState &worker = PrepareWorker(*state, kWorker, kRole);
    LocalStats stats{};
    InitLocalStats(stats, kWorker, kRole);

    // worker0 的首个潜在槽是 task0。前缀尚未 Close 时无事可做；
    // Close task0 后即使它是 Alloc、没有登记 candidate bit，也必须进入
    // scanner 把这个已知非候选槽永久越过。
    bool all_ok = !CrossCoreExecHasLocalProgressWork(
        state, kWorker, kRole, /*replay_closed_exclusive=*/0, stats
    );
    all_ok &= CloseSyntheticSubmit(
        *state, stats, /*task_id=*/0, TaskKind::Alloc
    );
    all_ok &= CrossCoreExecHasLocalProgressWork(
        state, kWorker, kRole, /*replay_closed_exclusive=*/1, stats
    );
    const uint32_t progressed =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, /*replay_closed_exclusive=*/1,
            /*production_closed=*/false,
            DrainPlace::EfDrain, stats
        );
    all_ok &= progressed == 0 && stats.exec_candidate_slot == 1 &&
        !CrossCoreExecHasLocalProgressWork(
            state, kWorker, kRole,
            /*replay_closed_exclusive=*/1, stats
        );

    // token 活跃时，即使下一个候选尚未进入已 Close 前缀也不得跳过。
    state->exec_tokens[kWorker][0].control.phase =
        ExecTokenPhase::WaitingFanin;
    all_ok &= CrossCoreExecHasLocalProgressWork(
        state, kWorker, kRole, /*replay_closed_exclusive=*/1, stats
    );

    // 非法输入必须返回 true，交给完整协议保存精确错误证据。
    all_ok &= CrossCoreExecHasLocalProgressWork(
        state, kWorkers, kRole, /*replay_closed_exclusive=*/1, stats
    );
    Check(
        all_ok, kTest,
        "gate skips only an Idle executor with no closed candidate"
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

        // S5b 允许任意有效 Scalar 构建。Build owner 是
        // primary/secondary 时只剩另一候选合法；在 K2 外
        // 时两候选都可参加 CAS。
        all_ok &= PaBuildOwnerEligible(primary, route.engine_class) &&
            PaBuildOwnerEligible(secondary, route.engine_class) &&
            PaBuildOwnerEligible(outside, route.engine_class) &&
            PaExecOwnerMatchesEngine(primary, route.engine_class) &&
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

        const uint32_t cross_role =
            route.engine_class == ExecEngineClass::Aic
                ? kAivBuildOwner : kAicBuildOwner;
        all_ok &= PaBuildOwnerEligible(
                      cross_role, route.engine_class
                  ) &&
            PaExecuteOwnerEligible(
                task_id, cross_role, route.engine_class, primary
            ) &&
            PaExecuteOwnerEligible(
                task_id, cross_role, route.engine_class, secondary
            );
    }

    Check(
        all_ok, kTest,
        "arbitrary builder placement yields the exact one-or-two K2 set"
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

    // worker 2 的第一个潜在 AIC task 是 task 1。immutable plan 已声明
    // 它是本核 K2 候选，因此 EMPTY 时必须保留队头；不能再因本地 bit
    // 尚未登记而永久越过。
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
            const bool transitional_registration =
                RegisterLocalCandidate(stats, 1, TaskKind::Qk);
            all_ok &= progressed == 0 &&
                stats.exec_candidate_slot == 0 &&
                transitional_registration &&
                CandidateBitForTask(
                    stats, kWorker, CoreRole::Aic, 1
                ) &&
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
        "Close history fails closed while planned EMPTY keeps its cursor"
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
            aic_task ? kAivBuildOwner : kAicBuildOwner;
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
            uint32_t target_slot =
                kCrossCoreExecMaxCandidateSlots;
            if (CrossCoreExecPotentialSlotForTask(
                    worker_id, RoleForWorker(worker_id),
                    task_id, target_slot
                )) {
                stats.exec_candidate_slot = target_slot;
            }
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

void TestArbitraryBuildOwnerCandidateBehavior() {
    constexpr const char *kTest =
        "arbitrary-build-owner-candidate-behavior";
    constexpr std::array<std::pair<uint32_t, TaskKind>, 2> kCases{{
        {1, TaskKind::Qk}, {2, TaskKind::Sf}
    }};
    bool all_ok = true;

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
        const uint32_t opposite_role =
            route.engine_class == ExecEngineClass::Aic
                ? kAivBuildOwner : kAicBuildOwner;

        // primary/secondary 验证 builder 占用 K2 时只剩另一
        // executor；outside 与 opposite_role 验证 K2 外同角色和
        // 跨角色 builder 都保留两个合法执行候选。
        for (uint32_t build_owner : std::array<uint32_t, 4>{
                 primary, secondary, outside, opposite_role
             }) {
            all_ok &= PaBuildOwnerEligible(
                build_owner, route.engine_class
            );
            for (ExecPhase phase : std::array<ExecPhase, 2>{
                     ExecPhase::Building, ExecPhase::Built
                 }) {
                MappedSchedulerState mapping;
                SchedulerState *state = mapping.Get();
                all_ok &= state != nullptr;
                if (state == nullptr) {
                    continue;
                }
                if (phase == ExecPhase::Built) {
                    if (kind == TaskKind::Sf) {
                        state->tasks[task_id - 1U].flag = 1;
                        all_ok &= PublishKernelCell(
                            *state, task_id, build_owner, kind,
                            {static_cast<int32_t>(task_id - 1U)}
                        );
                    } else {
                        all_ok &= PublishKernelCell(
                            *state, task_id, build_owner, kind
                        );
                    }
                } else {
                    SetCellState(
                        *state, task_id, ExecPhase::Building,
                        build_owner, kExecUnboundOwner,
                        ExecEngineClass::None, 0
                    );
                }

                uint32_t total_progress = 0;
                uint32_t total_execute = 0;
                uint32_t total_claim_cas = 0;
                uint32_t total_control_loads = 0;
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
                        stats, task_id, kind
                    );
                    ExecScanTestOps::ResetObservations();
                    ExecScanTestOps::WatchControl(
                        &state->exec_cells[task_id].control.state
                    );
                    total_progress +=
                        ProgressCrossCoreExec<ExecScanTestOps>(
                            state, worker, task_id + 1U,
                            /*production_closed=*/false,
                            DrainPlace::EfDrain, stats
                        );
                    total_execute += ExecScanTestOps::execute_calls;
                    total_claim_cas +=
                        ExecScanTestOps::watched_control_cas_calls;
                    total_control_loads +=
                        ExecScanTestOps::watched_control_loads;
                    const bool should_keep =
                        phase == ExecPhase::Building &&
                        candidate != build_owner;
                    const bool candidate_ok =
                        CandidateBitForTask(
                            stats, candidate,
                            RoleForWorker(candidate), task_id
                        ) == should_keep &&
                        NoFatal(*state);
                    all_ok &= candidate_ok;
                }

                if (phase == ExecPhase::Building) {
                    all_ok &= total_progress == 0 &&
                        total_execute == 0 && total_claim_cas == 0 &&
                        total_control_loads == 2;
                } else {
                    const DecodedExecState done = DecodeExecState(
                        state->exec_cells[task_id].control.state
                    );
                    const uint32_t expected_executor =
                        build_owner == primary ? secondary : primary;
                    // 唯一 winner 复用 scanner 已经返回的 BUILT
                    // 快照完成 BUILT->CLAIMED CAS；稍后到达的 loser
                    // 只需再读一次 DONE。CLAIMED->DONE 仍是独立 CAS。
                    all_ok &= total_progress == 1 &&
                        total_execute == 1 && total_claim_cas == 2 &&
                        total_control_loads == 2 &&
                        done.valid && done.phase == ExecPhase::Done &&
                        done.build_owner == build_owner &&
                        done.execute_owner == expected_executor &&
                        state->tasks[task_id].flag == 1 &&
                        NoFatal(*state);
                }
            }
        }
    }

    Check(
        all_ok, kTest,
        "primary/secondary/outside/cross-role builders preserve exact K2 liveness"
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
    constexpr uint32_t build_owner = kAivBuildOwner;
    bool all_ok = ResolvePaExecRoute(
                      kKind, FunctionId(kKind), route
                  ) &&
        FixedPaExecuteCandidates(
            kTask, route.engine_class, primary, secondary
        ) &&
        PaBuildOwnerEligible(
            build_owner, route.engine_class
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
    const uint32_t first_progress = ProgressCrossCoreExec<ExecScanTestOps>(
        state, first_worker, kTask + 1U,
        /*production_closed=*/false,
        DrainPlace::EfDrain, first_stats
    );
    const uint32_t second_progress = ProgressCrossCoreExec<ExecScanTestOps>(
        state, second_worker, kTask + 1U,
        /*production_closed=*/false,
        DrainPlace::EfDrain, second_stats
    );
    const uint32_t first_terminal_progress = primary_first
        ? 0U
        : ProgressCrossCoreExec<ExecScanTestOps>(
              state, first_worker, kTask + 1U,
              /*production_closed=*/false,
              DrainPlace::EfDrain, first_stats
          );
    const DecodedExecState done = DecodeExecState(
        state->exec_cells[kTask].control.state
    );
    all_ok &= first_progress == (primary_first ? 1U : 0U) &&
        second_progress == (primary_first ? 0U : 1U) &&
        first_terminal_progress == 0 &&
        CrossCoreExecCandidateBitmapEmpty(first_stats) &&
        CrossCoreExecCandidateBitmapEmpty(second_stats) &&
        ExecScanTestOps::execute_calls == 1 &&
        ExecScanTestOps::executed_tasks[0] == kTask &&
        done.valid && done.phase == ExecPhase::Done &&
        done.build_owner == build_owner &&
        done.execute_owner == primary &&
        state->tasks[kTask].flag == 1 &&
        NoFatal(*state);
    Check(
        all_ok, test,
        "primary claims first or fallback yields one opportunity to primary"
    );
    std::printf("[PASS] %s\n", test);
}

void TestEitherCandidateMayWin() {
    RunDynamicCandidateFirstCase(
        /*primary_first=*/true, "preferred-primary-wins"
    );
    RunDynamicCandidateFirstCase(
        /*primary_first=*/false, "fallback-yields-once"
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
            1, TaskKind::Qk, 1,
            primary, secondary, target
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
            done.build_owner == 1 &&
            done.execute_owner == target &&
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
        kTest, "builder candidate skips known-target BUILDING"
    );

    state->exec_cells[2].control.state = built_state;
    const uint32_t published =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, target_worker, 3,
            /*production_closed=*/false,
            DrainPlace::EfDrain, target_stats
        );
    const uint32_t terminal_observed =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, other_worker, 3,
            /*production_closed=*/false,
            DrainPlace::EfDrain, other_stats
        );
    const DecodedExecState done = DecodeExecState(
        state->exec_cells[2].control.state
    );
    Check(
        published == 1 && terminal_observed == 0 &&
            !CandidateBitForTask(
                target_stats, target, RoleForWorker(target), 2
            ) &&
            !CandidateBitForTask(
                other_stats, primary, RoleForWorker(primary), 2
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
        CrossCoreExecAllTokensFullyReset(state, primary) &&
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

void TestFinalDrainGlobalFatalDoesNotFabricateExecFatal() {
    constexpr const char *kTest =
        "final-drain-global-fatal-preserves-first-failure";
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;

    constexpr uint32_t kBuilder = 1;
    (void)PrepareWorker(*state, kBuilder, CoreRole::Aic);
    LocalStats stats{};
    InitLocalStats(stats, kBuilder, CoreRole::Aic);
    __atomic_store_n(
        &state->fatal.value, int32_t{1}, __ATOMIC_RELEASE
    );
    ExecScanTestOps::ResetObservations();
    const bool fatal_observed =
        ObserveCrossCoreFinalDrainFatal<ExecScanTestOps>(
            state, kBuilder, stats
        );
    const DecodedExecState cell0 = DecodeExecState(
        state->exec_cells[0].control.state
    );
    Check(
        fatal_observed && state->fatal.value == 1 &&
            state->exec_fatal.state == 0 &&
            cell0.valid && cell0.phase == ExecPhase::Empty &&
            state->exec_tokens[kBuilder][0].control.phase ==
                ExecTokenPhase::Idle &&
            state->exec_tokens[kBuilder][1].control.phase ==
                ExecTokenPhase::Idle &&
            ExecScanTestOps::execute_calls == 0,
        kTest,
        "FinalDrain observes scheduler fatal without fake exec fatal"
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
                state->exec_tokens[executor][0].control.phase ==
                    ExecTokenPhase::WaitingFanin &&
                DecodeExecState(
                    state->exec_cells[kTask].control.state
                ).phase == ExecPhase::Claimed &&
                ExecScanTestOps::execute_calls == 0 && NoFatal(*state);

            __atomic_store_n(
                &state->fatal.value, int32_t{1}, __ATOMIC_RELEASE
            );
            const uint32_t opportunistic_progress =
                ProgressCrossCoreExec<ExecScanTestOps>(
                    state, worker, kTask + 1U, false,
                    DrainPlace::EfDrain, stats
                );
            all_cases_ok &= opportunistic_progress == 0 &&
                state->exec_tokens[executor][0].control.phase ==
                    ExecTokenPhase::WaitingFanin &&
                DecodeExecState(
                    state->exec_cells[kTask].control.state
                ).phase == ExecPhase::Claimed &&
                state->tasks[kTask].vend == 0 &&
                state->tasks[kTask].flag == 0 &&
                state->exec_fatal.state == 0 &&
                ExecScanTestOps::execute_calls == 0;

            const bool final_fatal_observed =
                ObserveCrossCoreFinalDrainFatal<ExecScanTestOps>(
                    state, executor, stats
                );
            all_cases_ok &= final_fatal_observed &&
                state->exec_tokens[executor][0].control.phase ==
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
    // 链构造。global fatal 与已开始的合法工作单元并发时，当前 task 完整
    // 发布 completion，避免人为留下 CLAIMED 半状态；外层调度边界随后停止。
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
            ExecutionToken &token = state->exec_tokens[executor][0];
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
            bool completed = false;
            const bool progress_ok =
                ProgressCrossCoreActiveToken<ExecScanTestOps>(
                    state, worker, /*token_slot=*/0,
                    DrainPlace::EfDrain, stats, completed
                );
            const DecodedExecState completed_cell =
                DecodeExecState(
                    state->exec_cells[kTask].control.state
                );
            const bool completing_case_ok = progress_ok && completed &&
                token.control.phase == ExecTokenPhase::Idle &&
                completed_cell.phase == ExecPhase::Done &&
                state->tasks[kTask].vend != 0 &&
                state->tasks[kTask].flag == 1 &&
                state->exec_fatal.state == 0 &&
                ExecScanTestOps::execute_calls == 0 &&
                stats.result.placement[
                    static_cast<uint32_t>(DrainPlace::EfDrain)
                ] == 1;
            if (!completing_case_ok) {
                std::fprintf(
                    stderr,
                    "[DETAIL] completing progress_ok=%u completed=%u "
                    "token=%u cell=%u "
                    "vend=%lld flag=%lld exec_fatal=%lld execute=%u "
                    "placement=%llu\n",
                    progress_ok ? 1U : 0U,
                    completed ? 1U : 0U,
                    static_cast<uint32_t>(token.control.phase),
                    static_cast<uint32_t>(completed_cell.phase),
                    static_cast<long long>(state->tasks[kTask].vend),
                    static_cast<long long>(state->tasks[kTask].flag),
                    static_cast<long long>(state->exec_fatal.state),
                    ExecScanTestOps::execute_calls,
                    static_cast<unsigned long long>(
                        stats.result.placement[
                            static_cast<uint32_t>(DrainPlace::EfDrain)
                        ]
                    )
                );
            }
            all_cases_ok &= completing_case_ok;
        }
    }

    Check(
        all_cases_ok, kTest,
        "FinalDrain faults blocked token; valid completing token closes atomically"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestConcurrentGlobalFatalCompletesCurrentValidUnit() {
    constexpr const char *kTest =
        "concurrent-global-fatal-completes-current-valid-unit";
    bool all_cases_ok = true;

    // scanner 已取得 BUILT 快照后立刻注入 global fatal。当前合法 task
    // 允许继续 Claim、执行和完成；这证明正常路径不再逐边界轮询停止线。
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
            all_cases_ok &= progressed == 1 &&
                state->fatal.value == 1 &&
                state->exec_fatal.state == 0 &&
                DecodeExecState(
                    state->exec_cells[kTask].control.state
                ).phase == ExecPhase::Done &&
                state->exec_tokens[executor][0].control.phase ==
                    ExecTokenPhase::Idle &&
                ExecScanTestOps::execute_calls == 1 &&
                state->tasks[kTask].vend != 0 &&
                state->tasks[kTask].flag == 1;
        }
    }

    // fanin 的最后一次 ready Load 返回后注入 fatal，当前已经验证有效的
    // task 同样完整执行并发布 DONE。
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
            all_cases_ok &= progressed == 1 &&
                state->fatal.value == 1 &&
                state->exec_fatal.state == 0 &&
                DecodeExecState(
                    state->exec_cells[kTask].control.state
                ).phase == ExecPhase::Done &&
                state->exec_tokens[executor][0].control.phase ==
                    ExecTokenPhase::Idle &&
                ExecScanTestOps::execute_calls == 1 &&
                state->tasks[kTask].vend != 0 &&
                state->tasks[kTask].flag == 1 &&
                stats.result.placement[
                    static_cast<uint32_t>(DrainPlace::EfDrain)
                ] == 1;
        }
    }

    // ExecuteBoundKernel 是当前同步 engine 边界；在它返回时注入 fatal，
    // completion 仍必须闭合，不能把已经执行的 task 留在 CLAIMED。
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
            all_cases_ok &= progressed == 1 &&
                state->fatal.value == 1 &&
                state->exec_fatal.state == 0 &&
                DecodeExecState(
                    state->exec_cells[kTask].control.state
                ).phase == ExecPhase::Done &&
                state->exec_tokens[executor][0].control.phase ==
                    ExecTokenPhase::Idle &&
                ExecScanTestOps::execute_calls == 1 &&
                ExecScanTestOps::executed_tasks[0] == kTask &&
                state->tasks[kTask].vend != 0 &&
                state->tasks[kTask].flag == 1 &&
                stats.result.placement[
                    static_cast<uint32_t>(DrainPlace::EfDrain)
                ] == 1;
        }
    }

    Check(
        all_cases_ok, kTest,
        "fatal races never leave a valid current task half-completed"
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
                *state, 4, 40, TaskKind::Up, {0, 3, 0}
            ),
        kTest, "publish blocked SF and independent ready UP"
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
        first == 1 && stats.exec_candidate_slot == 2 &&
            CrossCoreExecCandidateBitmapEmpty(stats) &&
            state->exec_tokens[executor][0].control.phase ==
                ExecTokenPhase::WaitingFanin &&
            state->exec_tokens[executor][1].control.phase ==
                ExecTokenPhase::Idle &&
            DecodeExecState(
                state->exec_cells[2].control.state
            ).phase == ExecPhase::Claimed &&
            DecodeExecState(
                state->exec_cells[4].control.state
            ).phase == ExecPhase::Done &&
            ExecScanTestOps::execute_calls == 1 &&
            ExecScanTestOps::executed_tasks[0] == 4 &&
            NoFatal(*state),
        kTest,
        "slot0 blocked while slot1 claims and immediately executes ready work"
    );
    const uint32_t still_blocked =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, 5, false,
            DrainPlace::EfDrain, stats
        );
    Check(
        still_blocked == 0 && stats.exec_candidate_slot == 2 &&
            ExecScanTestOps::execute_calls == 1 && NoFatal(*state),
        kTest, "repeated fanin poll preserves the blocked first token"
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
        resumed == 1 && stats.exec_candidate_slot == 2 &&
            CrossCoreExecCandidateBitmapEmpty(stats) &&
            CrossCoreExecAllTokensFullyReset(state, executor) &&
            ExecScanTestOps::execute_calls == 2 &&
            ExecScanTestOps::executed_tasks[0] == 4 &&
            ExecScanTestOps::executed_tasks[1] == 2 &&
            DecodeExecState(
                state->exec_cells[2].control.state
            ).phase == ExecPhase::Done &&
            DecodeExecState(
                state->exec_cells[4].control.state
            ).phase == ExecPhase::Done &&
            NoFatal(*state),
        kTest, "the formerly blocked first token completes after its fanin"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestBusyCandidateUsesSecondToken() {
    constexpr const char *kTest = "busy-candidate-second-token";
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
    ExecutionToken &busy_token = state->exec_tokens[primary][0];
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
    InitLocalStats(busy_stats, primary, CoreRole::Aic);
    uint32_t new_candidate_slot = kCrossCoreExecMaxCandidateSlots;
    all_ok &= CrossCoreExecPotentialSlotForTask(
                  primary, CoreRole::Aic, kNewTask,
                  new_candidate_slot
              ) &&
        RegisterLocalCandidate(
            busy_stats, kNewTask, TaskKind::Qk
        );
    // kBusyTask 由本测试直接绑定到 token0，没有经过 scanner；因此把
    // owner-local cursor 定位到随后登记的 kNewTask，避免重访自己的
    // CLAIMED cell 冒充正常调度交错。
    busy_stats.exec_candidate_slot = new_candidate_slot;
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
    const DecodedExecState done = DecodeExecState(
        state->exec_cells[kNewTask].control.state
    );
    all_ok &= busy_progress == 1 &&
        CrossCoreExecCandidateBitmapEmpty(busy_stats) &&
        busy_token.control.phase == ExecTokenPhase::WaitingFanin &&
        state->exec_tokens[primary][1].control.phase ==
            ExecTokenPhase::Idle &&
        busy_stats.max_occupied == 2 &&
        ExecScanTestOps::watched_control_loads != 0 &&
        ExecScanTestOps::watched_control_cas_calls != 0 &&
        ExecScanTestOps::execute_calls == 1 &&
        done.valid && done.phase == ExecPhase::Done &&
        done.execute_owner == primary && NoFatal(*state);
    Check(
        all_ok, kTest,
        "one occupied token does not prevent Claim into the second token"
    );
    std::printf("[PASS] %s\n", kTest);
}

void TestTwoBlockedTokensStopClaimButPermitBuild() {
    constexpr const char *kTest =
        "two-blocked-tokens-stop-claim-permit-build";
    constexpr uint32_t kExecutor = 36;
    constexpr uint32_t kFirstTask = 2;
    constexpr uint32_t kSecondTask = 4;
    constexpr uint32_t kThirdTask = 66;
    MappedSchedulerState mapping;
    SchedulerState *state = mapping.Get();
    Check(state != nullptr, kTest, "state mapping");
    if (state == nullptr) return;

    state->tasks[0].flag = 1;
    state->tasks[3].flag = 1;
    bool all_ok =
        PublishKernelCell(
            *state, kFirstTask, 34, TaskKind::Sf, {1}
        ) &&
        PublishKernelCell(
            *state, kSecondTask, 40, TaskKind::Up,
            {2, 3, 0}
        ) &&
        PublishKernelCell(
            *state, kThirdTask, 34, TaskKind::Sf, {0}
        );
    WorkerState &worker = PrepareWorker(
        *state, kExecutor, CoreRole::Aiv
    );
    LocalStats stats{};
    InitLocalStats(stats, kExecutor, CoreRole::Aiv);
    all_ok &= RegisterLocalCandidate(
                  stats, kFirstTask, TaskKind::Sf
              ) &&
        RegisterLocalCandidate(
            stats, kSecondTask, TaskKind::Up
        ) &&
        RegisterLocalCandidate(
            stats, kThirdTask, TaskKind::Sf
        );
    ExecScanTestOps::ResetObservations();

    const uint32_t first_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, kThirdTask + 1U,
            /*production_closed=*/false,
            DrainPlace::EfDrain, stats
        );
    const uint32_t not_ready_after_claim =
        stats.result.fanin_not_ready_loads;
    ExecScanTestOps::WatchControl(
        &state->exec_cells[kThirdTask].control.state
    );
    const uint32_t second_progress =
        ProgressCrossCoreExec<ExecScanTestOps>(
            state, worker, kThirdTask + 1U,
            /*production_closed=*/false,
            DrainPlace::EfDrain, stats
        );
    const SharedBuildTicketResult build_ticket =
        TakeSharedBuildTicket<ExecScanTestOps>(
            state, kThirdTask + 1U, stats
        );

    all_ok &= first_progress == 0 && second_progress == 0 &&
        stats.exec_candidate_slot == 2 &&
        CandidateBitForTask(
            stats, kExecutor, CoreRole::Aiv, kThirdTask
        ) &&
        state->exec_tokens[kExecutor][0].control.phase ==
            ExecTokenPhase::WaitingFanin &&
        state->exec_tokens[kExecutor][1].control.phase ==
            ExecTokenPhase::WaitingFanin &&
        DecodeExecState(
            state->exec_cells[kFirstTask].control.state
        ).phase == ExecPhase::Claimed &&
        DecodeExecState(
            state->exec_cells[kSecondTask].control.state
        ).phase == ExecPhase::Claimed &&
        DecodeExecState(
            state->exec_cells[kThirdTask].control.state
        ).phase == ExecPhase::Built &&
        not_ready_after_claim >= 2 &&
        ExecScanTestOps::watched_control_loads == 0 &&
        ExecScanTestOps::watched_control_cas_calls == 0 &&
        stats.max_occupied == 2 &&
        build_ticket.status == SharedBuildTicketStatus::Acquired &&
        build_ticket.task_id == 0 && NoFatal(*state);
    Check(
        all_ok, kTest,
        "full token capacity blocks a third Claim but not the outer Build ticket"
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
            CrossCoreExecAllTokensFullyReset(state, executor) &&
            ValidateCrossCoreExecTerminalCells<ExecScanTestOps>(
                state, 5, first_bad_task
            ) &&
            first_bad_task == 5 && NoFatal(*state),
        kTest, "FinalDrain executes one dynamically eligible last-task owner"
    );

    std::array<bool, kWorkers> arrived{};
    std::array<LocalStats, kWorkers> arrival_stats{};
    std::array<int64_t,
               cross_core::kExecDrainArrivalGroups>
        expected_group_arrivals{};
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
        // 该子用例从“所有 terminal task 已由其原 LocalStats 消费”之后
        // 开始，只验证单向 drain arrival；因此把新建的测试上下文
        // 定位到第一个计划外候选，不能让 execute owner 用丢失的旧游标
        // 重新观察自己的 DONE task。
        uint32_t next_task = 0;
        while (CrossCoreExecPotentialTaskAt(
                   worker_id, RoleForWorker(worker_id),
                   arrival_stats[worker_id].exec_candidate_slot,
                   next_task
               ) &&
               next_task < 5) {
            ++arrival_stats[worker_id].exec_candidate_slot;
        }
        bool closed = false;
        const bool closure_ok =
            ProgressCrossCoreExecDrainClosure<ExecScanTestOps>(
                state, arrival_worker, 5,
                arrival_stats[worker_id],
                arrived[worker_id], closed
            );
        const uint32_t arrival_group =
            static_cast<uint32_t>(arrival_worker.block_id) %
            cross_core::kExecDrainArrivalGroups;
        ++expected_group_arrivals[arrival_group];
        all_arrivals_ok &= closure_ok && arrived[worker_id] &&
            closed == (worker_id != 0) &&
            state->exec_drain.arrivals[arrival_group].state ==
                expected_group_arrivals[arrival_group];
    }
    for (uint32_t group = 0;
         group < cross_core::kExecDrainArrivalGroups;
         ++group) {
        all_arrivals_ok &=
            expected_group_arrivals[group] == 6 &&
            state->exec_drain.arrivals[group].state == 6;
    }
    Check(
        all_arrivals_ok && NoFatal(*state),
        kTest, "six workers arrive in each of sixteen drain groups"
    );

    // worker 0 是唯一 root observer；它在自己的首次到达时尚未看到
    // 其余分组闭合。全部 96 个 worker 到达后再次推进，才校验所有
    // terminal cell 并完成 device 级收口。
    bool root_closed = false;
    const bool root_ok =
        ProgressCrossCoreExecDrainClosure<ExecScanTestOps>(
            state, state->workers[0], 5, arrival_stats[0],
            arrived[0], root_closed
        );
    Check(
        root_ok && root_closed && NoFatal(*state),
        kTest,
        "root validates terminal cells after every group arrives"
    );

    bool repeat_closed = root_closed;
    const bool repeat_ok =
        ProgressCrossCoreExecDrainClosure<ExecScanTestOps>(
            state, state->workers[0], 5, arrival_stats[0],
            arrived[0], repeat_closed
        );
    Check(
        repeat_ok && repeat_closed &&
            state->exec_drain.arrivals[0].state == 6,
        kTest, "one worker cannot increment drain twice"
    );
    std::printf("[PASS] %s\n", kTest);
}

}  // namespace

int main() {
    TestPotentialSlotRoundTrip();
    TestOpportunisticProgressLocalGate();
    TestDynamicExecutorEligibility();
    TestSubmitCloseRegistersOnlyCandidates();
    TestCloseAndRegistrationRejectHistoryRewrite();
    TestOnlyTwoCandidatesObserveControl();
    TestArbitraryBuildOwnerCandidateBehavior();
    TestEitherCandidateMayWin();
    TestMappedEmptyDelayedPublication();
    TestMappedBuildingDelayedPublication();
    TestOtherCandidateSkipsClaimedAndDone();
    TestDeterministicClaimLossIsNormal();
    TestInvalidStatesFailClosed();
    TestFinalDrainGlobalFatalDoesNotFabricateExecFatal();
    TestGlobalFatalFaultsActiveTokens();
    TestConcurrentGlobalFatalCompletesCurrentValidUnit();
    TestBusyTokenResumesScanning();
    TestBusyCandidateUsesSecondToken();
    TestTwoBlockedTokensStopClaimButPermitBuild();
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
