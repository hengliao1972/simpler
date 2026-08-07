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

#ifndef PA_SCHEDULER_COMMON_PA_SCHEDULER_CORE_H
#define PA_SCHEDULER_COMMON_PA_SCHEDULER_CORE_H

#ifndef PA_DEVICE
#define PA_DEVICE inline
#endif

#ifndef PA_GM
#define PA_GM
#endif

#include "pa_frontend.h"
#if PTO_FDWIC_SHARED_MAP
#include "pa_shared_tensormap.h"
#include "pa_exec_adapter.h"
#include "a5_exec_policy.h"
#endif
#include "pa_trace.h"

namespace pa_scheduler {

#if PTO_FDWIC_SHARED_MAP
static_assert(
    kAicWorkers == cross_core_dag::kA5AicScalarWorkers &&
        kAivWorkers == cross_core_dag::kA5AivScalarWorkers &&
        kWorkers == cross_core_dag::kA5ScalarWorkers,
    "standalone worker model must match the selected A5 backend policy"
);
#endif

struct LocalStats {
    WorkerResult result;
#if PTO_FDWIC_SHARED_MAP
    // shared 的物理 slot 固定为极小有界数组；16bit 足以保存精确高水位，
    // 并与下面的本地退避位合计复用原先 max_occupied 的 4B，不扩大
    // CCEC [[block_local]] split runtime。
    uint8_t max_occupied;
    // 双中央 Execute ticket 中，每个 worker 第一次取得越界 ordinal 后
    // 永久置一，避免 FinalDrain 反复触碰角色热点 cursor。
    uint8_t exec_dispatch_exhausted;
    // opportunistic EfDrain 的单槽轮询状态只属于当前 worker，既不进入
    // GM WorkerState，也不改变设备/host ABI。短等待每次只跳过一个
    // Submit；连续长等待才增加到两个。两槽背压和 FinalDrain 始终强制轮询。
    uint8_t efdrain_skip_budget;
    uint8_t efdrain_no_progress_polls;
    // 只在末个 shared Submit 成功收尾时写入 task_id+1；回放结束后与
    // local_index 对照，证明 ticket 的 last bit 没有提前或遗漏。
    uint32_t declared_task_count;
#else
    uint32_t max_occupied;
#endif
    TraceContext trace;
};
#if PTO_FDWIC_SHARED_MAP
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
static_assert(
    sizeof(LocalStats) == 1216,
    "cross-core split LocalStats block-local ABI changed"
);
#else
static_assert(
    sizeof(LocalStats) == 1152,
    "cross-core LocalStats block-local ABI changed"
);
#endif
static_assert(
    kPrivateSlots <= 0xFFU,
    "shared local occupancy must fit the packed local counter"
);
// 单槽无进展后允许跳过的 Submit 次数。该常量只调节非必需的
// opportunistic EfDrain 采样频率，不改变 WaitForSlot/FinalDrain 的推进协议。
constexpr uint8_t kSharedEfDrainNoProgressSkipSubmits = 1;
constexpr uint8_t kSharedEfDrainLongWaitSkipSubmits = 2;
constexpr uint8_t kSharedEfDrainLongWaitPollThreshold = 24;
static_assert(
    kSharedEfDrainNoProgressSkipSubmits != 0,
    "shared EfDrain backoff must retain a finite retry interval"
);
static_assert(
    kSharedEfDrainLongWaitSkipSubmits >=
        kSharedEfDrainNoProgressSkipSubmits,
    "long-wait EfDrain interval must not poll more frequently"
);
static_assert(
    kSharedEfDrainLongWaitPollThreshold != 0,
    "long-wait EfDrain threshold must require observed stalls"
);

// cross-core execution package 的统一观测适配器。shared_exec_protocol.h
// 只描述业务原语，不能依赖 standalone 的 raw ABI；这里把每个真实原语
// 映射到稳定 AtomicSite/DcciSite。PA_BUILD_TRACE_FREE 时下列封装均在
// 编译期退化为一次原始 Ops 调用，不改变 perf/PMU 构建的协议形态。
template <typename Ops>
struct SharedExecTraceObserver {
    LocalStats *stats;

    PA_DEVICE int64_t LoadFatal(
        PA_GM volatile int64_t *address, uint32_t task_id
    ) const {
        return TraceAtomicLoad<Ops>(
            stats->trace, stats->result,
            static_cast<int32_t>(task_id),
            AtomicSite::SharedExecFatalLoad, address,
            /*result_used=*/true
        );
    }

    PA_DEVICE int64_t PublishFatal(
        PA_GM volatile int64_t *address, int64_t expected,
        int64_t desired, uint32_t task_id
    ) const {
        return TraceAtomicCompareExchange<Ops>(
            stats->trace, stats->result,
            static_cast<int32_t>(task_id),
            AtomicSite::SharedExecFatalSet,
            address, expected, desired,
            /*result_used=*/true
        );
    }

    PA_DEVICE int64_t LoadCellState(
        PA_GM volatile int64_t *address, uint32_t task_id
    ) const {
        return TraceAtomicLoad<Ops>(
            stats->trace, stats->result,
            static_cast<int32_t>(task_id),
            AtomicSite::SharedExecCellStateLoad, address,
            /*result_used=*/true
        );
    }

    PA_DEVICE int64_t ReserveBuild(
        PA_GM volatile int64_t *address, int64_t expected,
        int64_t desired, uint32_t task_id
    ) const {
        return TraceAtomicCompareExchange<Ops>(
            stats->trace, stats->result,
            static_cast<int32_t>(task_id),
            AtomicSite::SharedExecBuildReserve,
            address, expected, desired,
            /*result_used=*/true
        );
    }

    PA_DEVICE int64_t PublishBuilt(
        PA_GM volatile int64_t *address, int64_t expected,
        int64_t desired, uint32_t task_id
    ) const {
        return TraceAtomicCompareExchange<Ops>(
            stats->trace, stats->result,
            static_cast<int32_t>(task_id),
            AtomicSite::SharedExecBuiltPublish,
            address, expected, desired,
            /*result_used=*/true
        );
    }

    PA_DEVICE int64_t ClaimCell(
        PA_GM volatile int64_t *address, int64_t expected,
        int64_t desired, uint32_t task_id
    ) const {
        return TraceAtomicCompareExchange<Ops>(
            stats->trace, stats->result,
            static_cast<int32_t>(task_id),
            AtomicSite::SharedExecClaim,
            address, expected, desired,
            /*result_used=*/true
        );
    }

    PA_DEVICE int64_t PublishDone(
        PA_GM volatile int64_t *address, int64_t expected,
        int64_t desired, uint32_t task_id
    ) const {
        return TraceAtomicCompareExchange<Ops>(
            stats->trace, stats->result,
            static_cast<int32_t>(task_id),
            AtomicSite::SharedExecDonePublish,
            address, expected, desired,
            /*result_used=*/true
        );
    }

    template <typename Pointer>
    PA_DEVICE void FlushBuildPayload(
        Pointer address, uint64_t bytes, uint32_t task_id
    ) const {
        (void)TraceConfiguredDcciFlush<
            Ops, PA_BUILD_ATOMIC_SWIMLANE
        >(
            &stats->trace, static_cast<int32_t>(task_id), -1,
            DcciSite::SharedExecPayloadFlush, address, bytes
        );
    }

    template <typename Pointer>
    PA_DEVICE void InvalidateClaimPayload(
        Pointer address, uint64_t bytes, uint32_t task_id
    ) const {
        (void)TraceConfiguredDcciInvalidate<
            Ops, PA_BUILD_ATOMIC_SWIMLANE
        >(
            &stats->trace, static_cast<int32_t>(task_id), -1,
            DcciSite::SharedExecPayloadInvalidate,
            address, bytes
        );
    }

    template <typename Pointer>
    PA_DEVICE void InvalidateTokenDescriptor(
        Pointer address, uint64_t bytes, uint32_t task_id
    ) const {
        (void)TraceConfiguredDcciInvalidate<
            Ops, PA_BUILD_ATOMIC_SWIMLANE
        >(
            &stats->trace, static_cast<int32_t>(task_id), -1,
            DcciSite::SharedExecTokenDescriptorInvalidate,
            address, bytes
        );
    }

    template <typename Pointer>
    PA_DEVICE void InvalidateBuildDescriptor(
        Pointer address, uint64_t bytes, uint32_t task_id
    ) const {
        (void)TraceConfiguredDcciInvalidate<
            Ops, PA_BUILD_ATOMIC_SWIMLANE
        >(
            &stats->trace, static_cast<int32_t>(task_id), -1,
            DcciSite::SharedExecBuildSourceDescriptorInvalidate,
            address, bytes
        );
    }

    PA_DEVICE int64_t LoadFaninFlag(
        PA_GM volatile int64_t *address, uint32_t producer
    ) const {
        return TraceAtomicLoad<Ops>(
            stats->trace, stats->result,
            static_cast<int32_t>(producer),
            AtomicSite::FaninFlagLoad, address,
            /*result_used=*/true
        );
    }

    template <typename T>
    PA_DEVICE T PublishCompletionVend(
        PA_GM volatile T *address, T value,
        uint32_t task_id
    ) const {
        return TraceAtomicExchange<Ops>(
            stats->trace, stats->result,
            static_cast<int32_t>(task_id),
            AtomicSite::SharedExecCompletionVendPublish,
            address, value, /*result_used=*/false
        );
    }

    PA_DEVICE int64_t PublishCompletionFlag(
        PA_GM volatile int64_t *address, int64_t value,
        uint32_t task_id
    ) const {
        return TraceAtomicExchange<Ops>(
            stats->trace, stats->result,
            static_cast<int32_t>(task_id),
            AtomicSite::SharedExecCompletionFlagPublish,
            address, value, /*result_used=*/true
        );
    }
};
#endif

#if PTO_FDWIC_SHARED_MAP && !PA_BUILD_TRACE_FREE
// 正式 PA-UP 的 history DCCI 与 group-writer CAS 先捕获端点，raw 写入
// 要等 task-level handoff 之后。数组保留三项是为了复用 generic helper
// 的本地承载形状；generation 12 正式 PA 只使用下标 0 且 count=1。
// 该对象只存在于 full-swimlane 的 owner 本地栈，不进入 SchedulerState、
// trace raw 或 host/device ABI。
struct DeferredSharedWriterMetadataTrace {
    uint64_t history_dcci_begin;
    uint64_t history_dcci_end;
    uint64_t writer_cas_begin[3];
    uint64_t writer_cas_end[3];
    uint32_t history_dcci_lines;
    uint32_t writer_cas_count;
};
static_assert(
    __is_trivially_constructible(
        DeferredSharedWriterMetadataTrace
    ),
    "deferred writer metadata trace must remain owner-local"
);
#endif

#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
// runtime-entry TU 按核型各自拥有一份 external [[block_local]] 实例。
// orchestration caller 与 noinline finish 只交换固定 ticket/TaskArgs，Submit
// 内部的 context、统计与状态指针全部留在这份每核状态里。central ticket
// 只让有效 owner 跨 TU：finish_calls 是本核取得的 task 数，task_id_sum
// 证明本核稀疏且单调的 task 所有权集合。
struct alignas(64) CompeteFirstSplitRuntimeState {
    PA_GM SchedulerState *scheduler;
    PA_GM WorkerState *worker;
    uint32_t task_count;
    uint32_t worker_id;
    SubmitContext context;
    LocalStats stats;
    uint64_t caller_state_address;
    uint64_t finish_state_address;
    uint64_t finish_calls;
    uint64_t protocol_errors;
    uint64_t state_cookie;
    uint64_t task_id_sum;
    uint64_t owner_worker_id;
    uint64_t reserved;
};
static_assert(sizeof(CompeteFirstSplitRuntimeState) == 1728,
              "split runtime logical ABI changed");
static_assert(
    __is_trivially_constructible(CompeteFirstSplitRuntimeState) &&
        __has_trivial_destructor(CompeteFirstSplitRuntimeState),
    "CCEC block-local split state must not require ctor/dtor"
);

PA_DEVICE uint64_t CompeteFirstSplitStateCookie(uint32_t worker_id, CoreRole role) {
    return kCompeteFirstSplitStateCookieBase ^ static_cast<uint64_t>(worker_id) ^
           (static_cast<uint64_t>(static_cast<uint32_t>(role)) << 32U);
}
#endif

// submit-pmu 的 phase 在编译期固定；非诊断构建完全不引用 Ops 的 phase
// 接口。这样公共调度代码保持一份，swimlane/CPU/AscendC 也不会多出运行时分支。
template <SubmitPmuPhase Phase, typename Ops, typename PmuContext>
PA_DEVICE void BeginSubmitPmuPhase(PmuContext &context) {
#if PA_BUILD_SUBMIT_PMU
    if constexpr (kCompiledSubmitPmuPhase == Phase) {
        Ops::PmuPhaseBegin(context);
    }
#else
    (void)context;
#endif
}

template <SubmitPmuPhase Phase, typename Ops, typename PmuContext>
PA_DEVICE void EndSubmitPmuPhase(PmuContext &context) {
#if PA_BUILD_SUBMIT_PMU
    if constexpr (kCompiledSubmitPmuPhase == Phase) {
        Ops::PmuPhaseEnd(context);
    }
#else
    (void)context;
#endif
}

template <typename Ops>
PA_DEVICE uint64_t TraceTimestamp(TraceContext &trace, WorkerResult &result) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)result;
    return 0;
#else
    (void)trace;
    (void)result;
    // PollBatch 只允许存在于显式 AtomicPollRegionBegin/End 区间，region
    // end 已用自己的结束时钟完整收口。普通阶段边界只负责取时钟，避免
    // 每个 Submit 的四个主端点都重复读取 active_mask。
    return Ops::Now();
#endif
}

#if PA_BUILD_PERF_CLOCK
template <typename Ops>
PA_DEVICE_NOINLINE uint64_t ReadPerfFinalDrainBoundary() {
    // 隔离到冷的单调用 helper，避免新增的尾部取时参与巨大调度函数的
    // 内联与尾合并决策。它位于全部 execution drain 之后，不进入 Submit。
    return Ops::PerfClockNow();
}
#endif

template <typename Ops, typename T>
PA_DEVICE uint64_t TraceTimestampAfterAtomicResult(
    TraceContext &trace, WorkerResult &result, T value
) {
#if PA_BUILD_TRACE_FREE
    // submit-pmu/perf-clock/纯性能构建必须在预处理后完全没有额外时钟读取；
    // value 也只用于保持模板调用形态，不在这些构建中制造返回依赖指令。
    (void)trace;
    (void)result;
    (void)value;
    return 0;
#else
    (void)trace;
    (void)result;
    // 与 TraceTimestamp 一样不触碰已经由显式 region 管理的 PollBatch；
    // 这里只让 SYS_CNT 真正依赖 atomic 返回值。该时间表示本核 scalar
    // 已能消费返回值，不宣称跨核可见。
    return Ops::NowAfterAtomicResult(value);
#endif
}

PA_DEVICE uint32_t KindIndex(TaskKind kind) { return static_cast<uint32_t>(kind); }

#if PTO_FDWIC_SHARED_MAP
struct SharedPaTaskMeta {
    TaskKind kind;
    uint32_t group_index;
    uint32_t batch_start;
    bool has_following_group;
    bool is_last_submit;
    bool chained_writer;
};

PA_DEVICE uint8_t EncodeSharedPaTaskMeta(
    TaskKind kind, uint32_t group_index, bool has_following_group,
    bool is_last_submit = false
) {
    if (kind >= TaskKind::Count ||
        group_index >= kSharedPaMaxBlockGroups ||
        (kind == TaskKind::Alloc &&
         (group_index != 0 || has_following_group)) ||
        (kind != TaskKind::Up && has_following_group) ||
        (has_following_group &&
         group_index + 1U >= kSharedPaMaxBlockGroups) ||
        (is_last_submit &&
         (has_following_group ||
          (kind != TaskKind::Alloc && kind != TaskKind::Up)))) {
        return 0;
    }
    return static_cast<uint8_t>(
        kSharedPaTicketMetaPresent |
        (is_last_submit ? kSharedPaTicketLastSubmit : 0U) |
        (has_following_group ? kSharedPaTicketHasFollowing : 0U) |
        (group_index << kSharedPaTicketGroupShift) |
        static_cast<uint32_t>(kind)
    );
}

PA_DEVICE bool DecodeSharedPaTaskMeta(
    uint8_t encoded, uint32_t task_id, SharedPaTaskMeta &meta
) {
    if ((encoded & kSharedPaTicketMetaPresent) == 0 ||
        task_id >= kMaxTasks) {
        return false;
    }
    const TaskKind kind =
        static_cast<TaskKind>(encoded & kSharedPaTicketKindMask);
    const uint32_t group_index =
        (encoded >> kSharedPaTicketGroupShift) &
        kSharedPaTicketGroupMask;
    const bool has_following_group =
        (encoded & kSharedPaTicketHasFollowing) != 0;
    const bool is_last_submit =
        (encoded & kSharedPaTicketLastSubmit) != 0;
    if (kind >= TaskKind::Count ||
        group_index >= kSharedPaMaxBlockGroups ||
        (kind == TaskKind::Alloc &&
         (group_index != 0 || has_following_group)) ||
        (kind != TaskKind::Up && has_following_group) ||
        (has_following_group &&
         group_index + 1U >= kSharedPaMaxBlockGroups) ||
        (is_last_submit &&
         (has_following_group ||
          (kind != TaskKind::Alloc && kind != TaskKind::Up)))) {
        return false;
    }
    const uint32_t task_offset =
        SharedPaTaskOffset(kind, group_index);
    if (task_id < task_offset) {
        return false;
    }
    meta.kind = kind;
    meta.group_index = group_index;
    meta.batch_start = task_id - task_offset;
    meta.has_following_group = has_following_group;
    meta.is_last_submit = is_last_submit;
    meta.chained_writer =
        kind == TaskKind::Up && group_index != 0;
    return true;
}

struct SharedBuildDispatchTask {
    uint32_t task_id;
    uint32_t batch;
    SharedPaTaskMeta meta;
    bool executable;
    cross_core_dag::ExecEngineClass engine_class;
};

struct SharedExecDispatchRoute {
    uint32_t task_id;
    bool executable;
    cross_core_dag::ExecEngineClass engine_class;
};

// 公共执行器只解码算子无关路由，不读取 batch、TaskKind 或 group。
// Build adapter 另行解析业务 identity；两者共用同一条 4B 计划项，但
// execution scanner 不得因 PA 元数据格式变化而改变资格判断。
PA_DEVICE bool DecodeSharedExecDispatchRoute(
    PA_GM const SharedBuildDispatchState &dispatch,
    uint32_t task_id, SharedExecDispatchRoute &route
) {
    if (dispatch.task_count == 0 ||
        dispatch.task_count > kMaxTasks ||
        dispatch.executable_task_count > dispatch.task_count ||
        task_id >= dispatch.task_count) {
        return false;
    }
    bool executable = false;
    cross_core_dag::ExecEngineClass engine_class =
        cross_core_dag::ExecEngineClass::None;
    if (!cross_core_dag::DecodeExecDispatchRoute(
            dispatch.tasks[task_id].exec_route,
            executable, engine_class
        )) {
        return false;
    }
    route.task_id = task_id;
    route.executable = executable;
    route.engine_class = engine_class;
    return true;
}

PA_DEVICE bool DecodeSharedBuildDispatchTask(
    PA_GM const SharedBuildDispatchState &dispatch,
    uint32_t task_id, SharedBuildDispatchTask &task
) {
    SharedExecDispatchRoute exec_route{};
    if (!DecodeSharedExecDispatchRoute(
            dispatch, task_id, exec_route
        ) ||
        dispatch.batch_count == 0 ||
        dispatch.batch_count > kMaxBatches ||
        task_id >= dispatch.task_count) {
        return false;
    }
    PA_GM const SharedBuildDispatchTaskIdentity &identity =
        dispatch.tasks[task_id];
    SharedPaTaskMeta meta{};
    if (identity.batch >= dispatch.batch_count ||
        !DecodeSharedPaTaskMeta(
            identity.encoded_meta, task_id, meta
        ) ||
        meta.is_last_submit !=
            (task_id + 1U == dispatch.task_count)) {
        return false;
    }
    task.task_id = task_id;
    task.batch = identity.batch;
    task.meta = meta;
    task.executable = exec_route.executable;
    task.engine_class = exec_route.engine_class;
    return true;
}

// 这一步属于 PA adapter 校验，不属于公共 dispatch 解码：公共计划只说明
// task 是否执行以及 engine；PA Build owner 在发布任何 execution cell 前，
// 另行证明该通用路由与本算子的 kind/function 映射一致。
PA_DEVICE bool SharedPaDispatchRouteMatches(
    const SharedBuildDispatchTask &task
) {
    if (task.meta.kind == TaskKind::Alloc) {
        return !task.executable &&
               task.engine_class ==
                   cross_core_dag::ExecEngineClass::None;
    }
    if (!task.executable) {
        return false;
    }
    cross_core_dag::PaExecRoute expected{};
    return cross_core_dag::ResolvePaExecRoute(
               task.meta.kind,
               static_cast<int32_t>(KindIndex(task.meta.kind) - 1U),
               expected
           ) &&
           task.engine_class == expected.engine_class;
}
#endif

PA_DEVICE TaskKind GetTaskKind(uint32_t task_id) {
    return static_cast<TaskKind>(task_id % kTasksPerBatch);
}

PA_DEVICE int32_t FunctionId(TaskKind kind) {
    return kind == TaskKind::Alloc ? -1 : static_cast<int32_t>(KindIndex(kind) - 1);
}

#if PTO_FDWIC_SHARED_MAP
PA_DEVICE bool SharedPaFunctionIdMatches(
    TaskKind kind, bool winner, int32_t function_id
) {
    // Claim loser 不执行 kernel，真实 function_id 固定为 -1；winner 则
    // 必须与 ticket 中显式 kind 一致。QK/PV 的 output count 同为 1，
    // 不能只靠 shared_result.Size() 间接校验。
    return function_id == (winner ? FunctionId(kind) : -1);
}
#endif

PA_DEVICE uint64_t DependencyEdgeSignature(
    uint32_t consumer, uint32_t producer
) {
    // SplitMix64 finalizer 只作用于稳定的 (consumer,producer) 编码；各
    // winner 将边哈希 XOR 到本核结果，host 再跨核 XOR，因此签名与
    // winner 分布和完成顺序无关。
    uint64_t value =
        (static_cast<uint64_t>(consumer) << 32U) | producer;
    value ^= value >> 30U;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D049BB133111EBULL;
    value ^= value >> 31U;
    return value;
}

PA_DEVICE uint32_t NopCountForKind(PA_GM const NopCounts &nops, TaskKind kind) {
    switch (kind) {
        case TaskKind::Qk:
            return nops.qk;
        case TaskKind::Sf:
            return nops.sf;
        case TaskKind::Pv:
            return nops.pv;
        case TaskKind::Up:
            return nops.up;
        default:
            return 0;
    }
}

PA_DEVICE uint32_t WorkloadCountForKind(PA_GM const WorkloadCounts &counts, TaskKind kind) {
    switch (kind) {
        case TaskKind::Qk:
            return counts.qk;
        case TaskKind::Sf:
            return counts.sf;
        case TaskKind::Pv:
            return counts.pv;
        case TaskKind::Up:
            return counts.up;
        default:
            return 0;
    }
}

PA_DEVICE uint32_t CountBits(uint32_t value) {
    uint32_t count = 0;
    while (value != 0) {
        count += value & 1U;
        value >>= 1;
    }
    return count;
}

template <typename Ops>
PA_DEVICE int64_t LoadLine(
    PA_GM AtomicLine &line, LocalStats &stats, AtomicSite site, int32_t task_id = -1
) {
    // Ops::Load 在 A5 后端是 atomicAdd(0)；返回值是该 RMW 线性化时观察到的共享值。
    return TraceAtomicLoad<Ops>(stats.trace, stats.result, task_id, site, &line.value);
}

template <typename Ops>
PA_DEVICE int32_t LoadLine(
    PA_GM AtomicFlagLine &line, LocalStats &stats, AtomicSite site, int32_t task_id = -1
) {
    return TraceAtomicLoad<Ops>(stats.trace, stats.result, task_id, site, &line.value);
}

template <typename Ops>
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH) && PA_BUILD_SWIMLANE
PA_DEVICE_NOINLINE __attribute__((cold))
#else
PA_DEVICE
#endif
void SetFatal(
    PA_GM SchedulerState *state, LocalStats &stats,
    int32_t task_id = -1
) {
    // fatal 只从 0 单调置 1，重复 Exchange 不会把其他 worker 已观察到的失败状态清除。
    TraceAtomicExchange<Ops>(
        stats.trace, stats.result, task_id, AtomicSite::FatalSet, &state->fatal.value,
        static_cast<int32_t>(1)
    );
}

template <typename Ops>
PA_DEVICE bool IsFatal(PA_GM SchedulerState *state, LocalStats &stats, int32_t task_id = -1) {
    return LoadLine<Ops>(state->fatal, stats, AtomicSite::FatalPoll, task_id) != 0;
}

template <typename Ops>
PA_DEVICE bool WatchdogExpired(
    PA_GM SchedulerState *state, LocalStats &stats, uint64_t begin, uint32_t &polls
) {
    // 每 1024 次自旋才读取系统计数器，降低正常启动屏障上的计时开销；超时后向所有 worker 广播 fatal。
    ++polls;
    if ((polls & 1023U) != 0 || Ops::Now() - begin <= kWatchdogTicks) {
        return false;
    }
    SetFatal<Ops>(state, stats);
    return true;
}

PA_DEVICE uint32_t TwoLevelFinalBarrierGroupCount(FinalBarrierShape shape) {
    switch (shape) {
    case FinalBarrierShape::TwoLevel4:
        return 4;
    case FinalBarrierShape::TwoLevel8:
        return 8;
    case FinalBarrierShape::TwoLevel16:
        return 16;
    default:
        return 0;
    }
}

PA_DEVICE uint32_t FinalBarrierLeafGroup(FinalBarrierShape shape, PA_GM const WorkerState &worker) {
    const uint32_t block = static_cast<uint32_t>(worker.block_id);
    if (shape == FinalBarrierShape::ThreeLevel6x4x4) {
        return block % kFinalBarrierMaxLeafGroups;
    }
    return block % TwoLevelFinalBarrierGroupCount(shape);
}

template <typename Ops>
PA_DEVICE void PublishFinalBarrierLine(PA_GM AtomicLine &line, LocalStats &stats, AtomicSite increment_site) {
    (void)TraceAtomicFetchAdd<Ops>(
        stats.trace, stats.result, -1, increment_site, &line.value, 1,
        /*result_used=*/false
    );
}

template <typename Ops>
PA_DEVICE void ArriveHierarchicalFinalBarrier(
    PA_GM FinalBarrierState &barrier, FinalBarrierShape shape, PA_GM const WorkerState &worker,
    LocalStats &stats, AtomicSite increment_site
) {
    const uint32_t leaf = FinalBarrierLeafGroup(shape, worker);
    PublishFinalBarrierLine<Ops>(barrier.leaf_arrivals[leaf], stats, increment_site);
}

template <typename Ops>
PA_DEVICE bool ProgressHierarchicalFinalBarrier(
    PA_GM FinalBarrierState &barrier, FinalBarrierShape shape, PA_GM const WorkerState &worker,
    LocalStats &stats, AtomicSite increment_site, AtomicSite poll_site, bool &leaf_forwarded, bool &middle_forwarded,
    bool &root_released, bool &middle_released, bool &leaf_released
) {
    const uint32_t block = static_cast<uint32_t>(worker.block_id);
    const uint32_t leaf = FinalBarrierLeafGroup(shape, worker);
    const bool leaf_leader = worker.lane == 0 && block == leaf;
    if (shape != FinalBarrierShape::ThreeLevel6x4x4) {
        const uint32_t groups = TwoLevelFinalBarrierGroupCount(shape);
        const int64_t workers_per_group = static_cast<int64_t>(kWorkers / groups);
        if (leaf_leader && !leaf_forwarded &&
            LoadLine<Ops>(barrier.leaf_arrivals[leaf], stats, poll_site) >= workers_per_group) {
            PublishFinalBarrierLine<Ops>(barrier.root_arrival, stats, increment_site);
            leaf_forwarded = true;
        }
        const bool root_leader = leaf_leader && leaf == 0;
        if (root_leader && !root_released &&
            LoadLine<Ops>(barrier.root_arrival, stats, poll_site) >= static_cast<int64_t>(groups)) {
            PublishFinalBarrierLine<Ops>(barrier.root_release, stats, increment_site);
            root_released = true;
        }
        if (leaf_leader && leaf_forwarded && !leaf_released &&
            LoadLine<Ops>(barrier.root_release, stats, poll_site) >= 1) {
            PublishFinalBarrierLine<Ops>(barrier.leaf_releases[leaf], stats, increment_site);
            leaf_released = true;
        }
        return LoadLine<Ops>(barrier.leaf_releases[leaf], stats, poll_site) >= 1;
    }

    constexpr int64_t kWorkersPerLeaf = 6;
    constexpr int64_t kLeavesPerMiddle = 4;
    constexpr int64_t kMiddleGroups = 4;
    const uint32_t middle = leaf % kFinalBarrierMaxMiddleGroups;
    const bool middle_leader = leaf_leader && leaf == middle;
    if (leaf_leader && !leaf_forwarded &&
        LoadLine<Ops>(barrier.leaf_arrivals[leaf], stats, poll_site) >= kWorkersPerLeaf) {
        PublishFinalBarrierLine<Ops>(barrier.middle_arrivals[middle], stats, increment_site);
        leaf_forwarded = true;
    }
    if (middle_leader && leaf_forwarded && !middle_forwarded &&
        LoadLine<Ops>(barrier.middle_arrivals[middle], stats, poll_site) >= kLeavesPerMiddle) {
        PublishFinalBarrierLine<Ops>(barrier.root_arrival, stats, increment_site);
        middle_forwarded = true;
    }
    const bool global_leader = middle_leader && middle == 0;
    if (global_leader && middle_forwarded && !root_released &&
        LoadLine<Ops>(barrier.root_arrival, stats, poll_site) >= kMiddleGroups) {
        PublishFinalBarrierLine<Ops>(barrier.root_release, stats, increment_site);
        root_released = true;
    }
    if (middle_leader && middle_forwarded && !middle_released &&
        LoadLine<Ops>(barrier.root_release, stats, poll_site) >= 1) {
        PublishFinalBarrierLine<Ops>(barrier.middle_releases[middle], stats, increment_site);
        middle_released = true;
    }
    if (leaf_leader && leaf_forwarded && !leaf_released &&
        LoadLine<Ops>(barrier.middle_releases[middle], stats, poll_site) >= 1) {
        PublishFinalBarrierLine<Ops>(barrier.leaf_releases[leaf], stats, increment_site);
        leaf_released = true;
    }
    return LoadLine<Ops>(barrier.leaf_releases[leaf], stats, poll_site) >= 1;
}

template <typename Ops>
PA_DEVICE void AdvanceFrontier(PA_GM SchedulerState *state, LocalStats &stats) {
    // frontier 只表示“从 task 0 开始已经连续完成”的最高 task id，不能越过尚未发布 flag 的空洞。
    // 多个完成者可以同时扫描同一段连续区间，FetchMax 保证共享 frontier 只前进、不回退。
    ++stats.result.frontier_initial_loads;
    int64_t frontier = LoadLine<Ops>(
        state->frontier, stats, AtomicSite::FrontierInitialLoad
    );
    while (true) {
        const int64_t next = frontier + 1;
        if (next < 0 || next >= static_cast<int64_t>(kTaskCellCapacity)) {
            break;
        }
        if (TraceAtomicLoad<Ops>(
                stats.trace, stats.result, static_cast<int32_t>(next), AtomicSite::FrontierFlagLoad,
                &state->tasks[next].flag
            ) == 0) {
            ++stats.result.frontier_terminal_loads;
            break;
        }
        uint64_t retries = 0;
        // FetchMax 返回更新前的值；若其他核已经走得更远，就从其 old 值继续扫描，避免重复从 next 起步。
        ++stats.result.frontier_updates;
        const int64_t old = TraceAtomicFetchMax<Ops>(
            stats.trace, stats.result, static_cast<int32_t>(next), AtomicSite::FrontierMax,
            &state->frontier.value, next, retries
        );
        stats.result.cas_retries += retries;
        frontier = old > next ? old : next;
    }
}

template <typename Ops>
PA_DEVICE void CompleteTask(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_id, LocalStats &stats
) {
    // 两种模式都先发布 vend，再经 store barrier 发布 ready flag：fanin 和
    // slot 执行以 flag 为可见性条件，不能交换顺序。private ring 还需要
    // 连续 frontier 做 heap reclaim；shared PA 使用有界 no-wrap shard，
    // 依赖逐 task flag，正常完成路径不维护无消费者的全局前沿。
    TraceAtomicExchange<Ops>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), AtomicSite::CompletionVendExchange,
        &state->tasks[task_id].vend, worker.heap_next
    );
    Ops::StoreBarrier();
    TraceAtomicExchange<Ops>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), AtomicSite::CompletionFlagExchange,
        &state->tasks[task_id].flag, static_cast<int64_t>(1)
    );
#if !PTO_FDWIC_SHARED_MAP
    AdvanceFrontier<Ops>(state, stats);
#endif
}

template <typename Ops>
PA_DEVICE bool SlotReady(PA_GM SchedulerState *state, PA_GM LocalSlot &slot, LocalStats &stats) {
    // 每个 fanin flag 都是跨核共享的完成条件；在单轮 kernel 内，
    // 完成值会单调保持 ready。shared 模式遇到第一个未就绪依赖时，
    // 把此前已确认 ready 的前缀从本核私有 slot 中移除，避免后续
    // 每次 EfDrain 都重复 atomic-load 同一前缀；private 模式保持原逻辑。
    for (uint32_t index = 0; index < slot.fanin_count; ++index) {
        const int32_t dependency = slot.fanin[index];
        if (TraceAtomicLoad<Ops>(
                stats.trace, stats.result, dependency, AtomicSite::FaninFlagLoad,
                &state->tasks[dependency].flag
            ) == 0) {
            ++stats.result.fanin_not_ready_loads;
#if PTO_FDWIC_SHARED_MAP
            if (index != 0) {
                const uint32_t remaining =
                    slot.fanin_count - index;
                for (uint32_t pending = 0;
                     pending < remaining; ++pending) {
                    slot.fanin[pending] =
                        slot.fanin[index + pending];
                }
                slot.fanin_count = remaining;
            }
#endif
            return false;
        }
        ++stats.result.fanin_ready_loads;
    }
#if PTO_FDWIC_SHARED_MAP
    slot.fanin_count = 0;
#endif
    return true;
}

PA_DEVICE void RecordKernelCycles(LocalStats &stats, TaskKind kind, uint64_t cycles) {
    const uint32_t index = KindIndex(kind) - 1;
    ++stats.result.kernel_counts[index];
#if PA_BUILD_TRACE_FREE
    // 无 trace 构建仍保留四类 kernel 的正确性计数，但不把恒为零的
    // 观察时长写进热路径，更不会在 host 侧把 0 冒充 kernel 性能。
    (void)cycles;
#else
    stats.result.kernel_cycles[index] += cycles;
    if (stats.result.kernel_min_cycles[index] == 0 || cycles < stats.result.kernel_min_cycles[index]) {
        stats.result.kernel_min_cycles[index] = cycles;
    }
    if (cycles > stats.result.kernel_max_cycles[index]) {
        stats.result.kernel_max_cycles[index] = cycles;
    }
#endif
}

template <typename Ops>
PA_DEVICE uint32_t DrainReady(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, DrainPlace place, LocalStats &stats
) {
    // 同一套 drain 被三个位置复用：每次 Submit 开头的 EfDrain、ring 背压等待和所有 Submit 后的最终 drain。
    // slot 属于当前 worker；只有其全部跨核 fanin 已 ready 时才执行所选 winner 负载、发布完成并释放 slot。
#if PTO_FDWIC_SHARED_MAP
    // shared 路径后续还要用这份入口值约束 header 扫描，因此只读取一次；
    // 不依赖编译器把两次 GM 读取自动合并。
    const uint32_t occupied_at_entry = worker.occupied_count;
    if (occupied_at_entry == 0) {
        return 0;
    }
#else
    if (worker.occupied_count == 0) {
        return 0;
    }
#endif
    uint32_t freed = 0;
#if PTO_FDWIC_SHARED_MAP
    // occupied_count 是进入本次 drain 时全部物理 slot 的精确快照。
    // 找齐这些 occupied header 后即可提前退出；仍遍历 kPrivateSlots，
    // 因此普通自领任务和 BlockWon/follower 使用任意物理槽都不会漏扫。
    // 使用入口快照而不是执行中递减的 worker.occupied_count，避免释放
    // slot 后改变本轮扫描边界。
    uint32_t remaining_occupied = occupied_at_entry;
#endif
    for (uint32_t index = 0; index < kPrivateSlots; ++index) {
#if PTO_FDWIC_SHARED_MAP
        if (remaining_occupied == 0) {
            break;
        }
#endif
        PA_GM LocalSlot &slot = worker.slots[index];
        if (!slot.occupied) {
            continue;
        }
#if PTO_FDWIC_SHARED_MAP
        --remaining_occupied;
#endif
        if (!slot.built || !SlotReady<Ops>(state, slot, stats)) {
            continue;
        }
        const TaskKind kind = static_cast<TaskKind>(slot.kind + 1);
        const uint64_t kernel_begin = TraceTimestamp<Ops>(stats.trace, stats.result);
        Ops::ExecuteKernel(state, worker, kind, NopCountForKind(state->config.nops, kind));
        const uint64_t kernel_end = TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<false>(
            stats.trace, stats.result, static_cast<int32_t>(slot.task_id), static_cast<int32_t>(slot.kind),
            TracePhase::Kernel, ProfilePhase::ReplayTail, kernel_begin, kernel_end
        );
        RecordKernelCycles(stats, kind, kernel_end - kernel_begin);
        CompleteTask<Ops>(state, worker, slot.task_id, stats);
        const uint64_t commit_cycle = TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<false>(
            stats.trace, stats.result, static_cast<int32_t>(slot.task_id), static_cast<int32_t>(slot.kind),
            TracePhase::Commit, ProfilePhase::ReplayTail, commit_cycle, commit_cycle
        );
        slot.built = false;
        slot.occupied = false;
        --worker.occupied_count;
        ++stats.result.placement[static_cast<uint32_t>(place)];
        ++freed;
    }
    return freed;
}

#if PTO_FDWIC_SHARED_MAP
template <typename Ops>
PA_DEVICE uint32_t OpportunisticDrainReady(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker,
    LocalStats &stats
) {
    // 单槽长期等待时，相邻 Submit 往往反复读取同一个尚未完成的 fanin。
    // 这里仅降低“不承担活性责任”的 EfDrain 轮询频率：短等待采用
    // poll/skip1，连续长等待才采用 poll/skip2。
    // 两槽状态可能马上进入 WaitForSlot，必须立即检查；显式背压和最终
    // drain 仍直接调用 DrainReady，因此不会丢失完成通知或改变依赖语义。
    if (worker.occupied_count == 0) {
        stats.efdrain_skip_budget = 0;
        stats.efdrain_no_progress_polls = 0;
        return 0;
    }
    const bool single_slot = worker.occupied_count == 1;
    if (single_slot && stats.efdrain_skip_budget != 0) {
        --stats.efdrain_skip_budget;
        return 0;
    }
    // 保持唯一的 DrainReady 调用点，避免 CCEC 把完整 slot 扫描体复制到
    // single/full 两个 successor 中而放大每种 Submit 实例的指令体积。
    stats.efdrain_skip_budget = 0;
    if (!single_slot) {
        stats.efdrain_no_progress_polls = 0;
    }
    const uint32_t freed = DrainReady<Ops>(
        state, worker, DrainPlace::EfDrain, stats
    );
    if (single_slot && freed == 0 &&
        worker.occupied_count == 1) {
        if (stats.efdrain_no_progress_polls <
            kSharedEfDrainLongWaitPollThreshold) {
            ++stats.efdrain_no_progress_polls;
        }
        const bool long_wait =
            stats.efdrain_no_progress_polls ==
                kSharedEfDrainLongWaitPollThreshold;
        stats.efdrain_skip_budget = long_wait
            ? kSharedEfDrainLongWaitSkipSubmits
            : kSharedEfDrainNoProgressSkipSubmits;
    } else {
        stats.efdrain_no_progress_polls = 0;
    }
    return freed;
}
#endif

PA_DEVICE int32_t FindFreeSlot(PA_GM WorkerState &worker) {
    for (uint32_t index = 0; index < kPrivateSlots; ++index) {
        if (!worker.slots[index].occupied) {
            return static_cast<int32_t>(index);
        }
    }
    return -1;
}

template <typename Ops, bool Profile>
PA_DEVICE void WaitForSlot(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker,
    uint32_t task_id, LocalStats &stats
#if PTO_FDWIC_SHARED_MAP
    , bool &fatal_exit
#endif
) {
    // 四个物理 slot 中预留两个 won slot 语义位，仅有 kUsableSlots 个可供本图使用；满时靠 drain 取得进展。
    const uint64_t wait_begin = TraceTimestamp<Ops>(stats.trace, stats.result);
    bool waited = false;
    // 只聚合这个显式背压等待区中的 fanin 观察；每次 Submit 开头的
    // opportunistic EfDrain 仍保留逐条 Atomic，不能仅凭 site 名称全局聚合。
    const uint32_t poll_region = AtomicPollRegionBegin<Ops>(
        stats.trace, stats.result,
        TraceAtomicPollBatchMask(AtomicSite::FaninFlagLoad) |
            TraceAtomicPollBatchMask(AtomicSite::FatalPoll)
    );
    // 退出条件只有 occupied_count 重新低于可用容量；依赖尚未 ready 时 SpinHint 后继续重试。
    while (worker.occupied_count >= kUsableSlots) {
        waited = true;
        ++stats.result.wait_iterations[0];
        if (DrainReady<Ops>(
                state, worker, DrainPlace::RingBackpressure, stats
            ) == 0) {
#if PTO_FDWIC_SHARED_MAP
            // gate 放行后若其他 worker 报错，本核可能正被两个永远无法
            // ready 的后继 slot 顶满。每 1024 次无进展轮询一次 fatal，
            // 只影响真正的背压慢路，不给正常 winner 热路增加原子读取。
            if ((stats.result.wait_iterations[0] & 1023ULL) == 0 &&
                IsFatal<Ops>(
                    state, stats, static_cast<int32_t>(task_id)
                )) {
                fatal_exit = true;
                break;
            }
#endif
            Ops::SpinHint();
        }
    }
    AtomicPollRegionEnd<Ops>(stats.trace, stats.result, poll_region);
    if (waited) {
        ++stats.result.wait_events[0];
        const uint64_t wait_end = TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<Profile>(
            stats.trace, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::RingBp,
            ProfilePhase::WaitForSlot, wait_begin, wait_end, 0, 0
        );
    } else if constexpr (Profile) {
        AccumulatePhase<true>(
            stats.result, ProfilePhase::WaitForSlot, wait_begin,
            TraceTimestamp<Ops>(stats.trace, stats.result)
        );
    }
}

template <typename Ops, bool Profile>
PA_DEVICE bool HeapGuard(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_id, uint64_t output_bytes,
    LocalStats &stats
) {
    // 只有产生新输出的 winner 需要保护环形 heap；retire=frontier-H 对应已经允许复用的最老任务 vend。
    // 等待期间也主动 drain 本核已就绪 slot，避免只自旋而阻塞能够推动 frontier 的 kernel。
    if (output_bytes == 0 || state->heap_base == 0) {
        return true;
    }
    const uint64_t ring = state->heap_size;
    ++stats.result.heap_guards;
    const uint64_t wait_begin = TraceTimestamp<Ops>(stats.trace, stats.result);
    bool waited = false;
    bool poll_region_active = false;
    uint32_t poll_region = 0;
    // 正常出口是 heap_next-vend 落入一个 ring；检测到不可能释放的覆盖或其他核 fatal 时返回失败。
    while (!IsFatal<Ops>(state, stats, static_cast<int32_t>(task_id))) {
        // 逻辑 heap 尚未走完第一圈时，所有物理输出区间都位于 [0, heap_next)，
        // 不可能覆盖此前分配；保留上面的 fatal 原子检查后，可直接跳过 frontier/vend 读取。
        if (worker.heap_next <= ring) {
            if (poll_region_active) {
                AtomicPollRegionEnd<Ops>(stats.trace, stats.result, poll_region);
            }
            if constexpr (Profile) {
                AccumulatePhase<true>(
                    stats.result, ProfilePhase::HeapGuard, wait_begin,
                    TraceTimestamp<Ops>(stats.trace, stats.result)
                );
            }
            return true;
        }
        // 与真实 PA 一样，首圈 fast path 上方的 FatalPoll 仍是直接记录；只有
        // 确认进入 heap wrap 慢路径后，才开启本等待 episode 的四类观察聚合。
        if (!poll_region_active) {
            poll_region = AtomicPollRegionBegin<Ops>(
                stats.trace, stats.result,
                TraceAtomicPollBatchMask(AtomicSite::FatalPoll) |
                    TraceAtomicPollBatchMask(AtomicSite::HeapFrontierLoad) |
                    TraceAtomicPollBatchMask(AtomicSite::HeapVendLoad) |
                    TraceAtomicPollBatchMask(AtomicSite::FaninFlagLoad)
            );
            poll_region_active = true;
        }
        const int64_t frontier = LoadLine<Ops>(
            state->frontier, stats, AtomicSite::HeapFrontierLoad, static_cast<int32_t>(task_id)
        );
        const int64_t retire = frontier - static_cast<int64_t>(state->heap_window);
        const uint64_t vend = retire < 0
            ? 0
            : TraceAtomicLoad<Ops>(
                  stats.trace, stats.result, static_cast<int32_t>(task_id), AtomicSite::HeapVendLoad,
                  &state->tasks[retire].vend
              );
        if (worker.heap_next - vend <= ring) {
            AtomicPollRegionEnd<Ops>(stats.trace, stats.result, poll_region);
            if (waited) {
                ++stats.result.wait_events[1];
                const uint64_t wait_end = TraceTimestamp<Ops>(stats.trace, stats.result);
                WriteTrace<Profile>(
                    stats.trace, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::RingBp,
                    ProfilePhase::HeapGuard, wait_begin, wait_end, 0, 1
                );
            } else if constexpr (Profile) {
                AccumulatePhase<true>(
                    stats.result, ProfilePhase::HeapGuard, wait_begin,
                    TraceTimestamp<Ops>(stats.trace, stats.result)
                );
            }
            return true;
        }
        if (frontier >= static_cast<int64_t>(task_id) - 1) {
            SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
            AtomicPollRegionEnd<Ops>(stats.trace, stats.result, poll_region);
            return false;
        }
        waited = true;
        ++stats.result.wait_iterations[1];
        if (DrainReady<Ops>(
                state, worker, DrainPlace::RingBackpressure, stats
            ) == 0) {
            Ops::SpinHint();
        }
    }
    if (poll_region_active) {
        AtomicPollRegionEnd<Ops>(stats.trace, stats.result, poll_region);
    }
    if (waited) {
        ++stats.result.wait_events[1];
        const uint64_t wait_end = TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<Profile>(
            stats.trace, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::RingBp,
            ProfilePhase::HeapGuard, wait_begin, wait_end, 0, 1
        );
    }
    return false;
}

struct ClaimOutcome {
    bool attempted;
    bool won;
    uint64_t retries;
    int32_t function_id;
};

template <typename Ops>
PA_DEVICE ClaimOutcome Claim(
    PA_GM SchedulerState *state, uint32_t worker_id, CoreRole role,
    uint32_t task_id, TaskKind kind, LocalStats &stats
) {
    // private 继续在单调 cursor 上执行 atomicMax；shared 则用每 task
    // 独立的两级 CAS Tournament 选出唯一 owner。private 保持原有
    // engine 同角色候选；shared S5b 则让每个 kernel 都由全部 96 个
    // Scalar 竞争 Build。kernel 类型只决定 function/Execute engine，
    // 不再限制 builder 角色。shared TensorMap 的严格插入顺序不由 Claim
    // 承担，而由 winner 后续的 per-task insert-completion chain 保证。
    // role 来自 RunSchedulerImpl 的入口 SSA 值；不能在每个 task 中再从
    // WorkerState GM 回读同一字段，否则 B256 会产生 98,304 次冗余读取。
    ClaimOutcome outcome{false, false, 0, -1};
#if PTO_FDWIC_SHARED_MAP
    const bool worker_role_matches =
        (role == CoreRole::Aic && worker_id < kAicWorkers) ||
        (role == CoreRole::Aiv && worker_id >= kAicWorkers &&
         worker_id < kWorkers);
    if (task_id >= kMaxTasks || !worker_role_matches) {
        return outcome;
    }
    uint32_t candidate_rank = 0;
    uint32_t tournament_groups = 0;
#else
    if (task_id >= kTaskCellCapacity) {
        return outcome;
    }
    PA_GM AtomicLine *cursor = nullptr;
#endif
    if (kind == TaskKind::Alloc) {
#if PTO_FDWIC_SHARED_MAP
        if (worker_id >= kWorkers) {
            return outcome;
        }
        // Alloc 的 96 个 worker 全部可候选；worker_id 本身就是
        // 稳定的 [0,96) rank，取模后八个 local 节点各承担
        // 12 个候选。这只改变 owner 仲裁人口，不改变后续
        // per-task insert-completion 严格插入链。
        candidate_rank = worker_id;
        tournament_groups =
            kSharedAllocClaimTournamentGroups;
#else
        (void)worker_id;
        cursor = &state->alloc_cursor[task_id % kCursorShards];
#endif
    } else {
        // Mirror MixedKernels::to_active_mask(), core_mask(), popcount(),
        // lane_active(), and self->role routing inside the real Claim span.
        const int32_t aic_kernel = kind == TaskKind::Qk || kind == TaskKind::Pv ? FunctionId(kind) : -1;
        const int32_t aiv0_kernel = kind == TaskKind::Sf || kind == TaskKind::Up ? FunctionId(kind) : -1;
        const int32_t aiv1_kernel = -1;
        uint8_t active_mask = 0;
        if (aic_kernel >= 0) active_mask |= 1U;
        if (aiv0_kernel >= 0) active_mask |= 2U;
        if (aiv1_kernel >= 0) active_mask |= 4U;
        const uint8_t core_mask = active_mask & 0x07U;
        const int32_t active_count = __builtin_popcount(static_cast<uint32_t>(core_mask));
        // 这里保留生产 Claim 的 lane-mask 路由边界。当前固定 PA 图按构造只生成单 lane
        // 的 QK/PV 或 SF/UP；需要两个及以上 lane 协作的 joint task 本应进入 BlockWon
        // 协议，本独立用例没有实现该动态路径，因此显式拒绝而不把它误当成单 lane task。
        if (active_count >= 2) {
            return outcome;
        }
        if ((core_mask & 1U) != 0) {
#if PTO_FDWIC_SHARED_MAP
            // QK/PV 最终仍在 AIC engine 执行；S5b 允许任意 Scalar
            // 构建 portable payload，worker_id 直接作为 96 核连续 rank。
            candidate_rank = worker_id;
            tournament_groups =
                kSharedKernelClaimTournamentGroups;
#else
            if (role != CoreRole::Aic) return outcome;
            cursor = &state->cube_cursor[task_id % kCursorShards];
#endif
            outcome.function_id = aic_kernel;
        } else if ((core_mask & 6U) != 0) {
#if PTO_FDWIC_SHARED_MAP
            // SF/UP 最终仍在 AIV engine 执行；Build 同样覆盖全部
            // 96 个 Scalar，不把 engine 路由反向写进 Claim 资格。
            candidate_rank = worker_id;
            tournament_groups =
                kSharedKernelClaimTournamentGroups;
#else
            if (role != CoreRole::Aiv) return outcome;
            cursor = &state->vector_cursor[task_id % kCursorShards];
#endif
            outcome.function_id = (core_mask & 2U) != 0 ? aiv0_kernel : aiv1_kernel;
        } else {
            return outcome;
        }
    }
    outcome.attempted = true;
#if PTO_FDWIC_SHARED_MAP
    PA_GM SharedClaimTournamentTask *tournament =
        &state->claim_tournament[task_id];
    const uint32_t group = candidate_rank % tournament_groups;
    const int64_t expected = -1;
    const int64_t desired = static_cast<int64_t>(task_id);

    // 每个候选只等待自己唯一一次 local CAS 的返回。失败者立即返回；
    // 每组唯一 local owner 才继续访问 root，因此 root 同地址竞争者从
    // 每个 task 的 96 个候选收敛为 8 个 root 竞争者。
    const int64_t local_observed =
        TraceAtomicCompareExchange<Ops>(
            stats.trace, stats.result,
            static_cast<int32_t>(task_id),
            AtomicSite::SharedClaimTournamentLocal,
            &tournament->local[group].owner.value,
            expected, desired
        );
    if (local_observed != expected) {
        if (local_observed != desired) {
            SetFatal<Ops>(
                state, stats, static_cast<int32_t>(task_id)
            );
        }
        outcome.function_id = -1;
        return outcome;
    }

    const int64_t root_observed =
        TraceAtomicCompareExchange<Ops>(
            stats.trace, stats.result,
            static_cast<int32_t>(task_id),
            AtomicSite::SharedClaimTournamentRoot,
            &tournament->root.owner.value,
            expected, desired
        );
    outcome.won = root_observed == expected;
    if (!outcome.won) {
        if (root_observed != desired) {
            SetFatal<Ops>(
                state, stats, static_cast<int32_t>(task_id)
            );
        }
        outcome.function_id = -1;
    }
#else
    // atomicMax 返回写入前的 cursor：old<task_id 表示本核完成首次推进
    // 并获胜，old>=task_id 则必须 Replay。
    const int64_t old = TraceAtomicFetchMax<Ops>(
        stats.trace, stats.result, static_cast<int32_t>(task_id),
        AtomicSite::ClaimMax, &cursor->value,
        static_cast<int64_t>(task_id), outcome.retries
    );
    outcome.won = old < static_cast<int64_t>(task_id);
    if (!outcome.won) outcome.function_id = -1;
#endif
    return outcome;
}

PA_DEVICE void RecordClaimOutcome(
    LocalStats &stats, TaskKind kind, const ClaimOutcome &outcome
#if PTO_FDWIC_SHARED_MAP
    , uint64_t &shared_claim_attempts
#endif
) {
    if (outcome.attempted) {
#if PTO_FDWIC_SHARED_MAP
        ++shared_claim_attempts;
#else
        ++stats.result.claim_attempts;
#endif
    }
    stats.result.cas_retries += outcome.retries;
    if (outcome.won) {
        ++stats.result.claim_wins;
        ++stats.result.wins[KindIndex(kind)];
    }
}

#if PTO_FDWIC_SHARED_MAP
PA_DEVICE void FinalizeSharedClaimAttempts(
    LocalStats &stats, uint64_t shared_claim_attempts
) {
    stats.result.claim_attempts = shared_claim_attempts;
}
#endif

template <typename Ops, bool Profile>
PA_DEVICE bool BuildWinner(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_id, TaskKind kind,
    const TaskArgs &args, const SubmitContext &context,
    const int32_t fanin[kMaxFanin], uint32_t fanin_count, LocalStats &stats
) {
    // kernel winner 不在 Submit 内立即执行计算，而是把完整 payload 和 fanin 存入自己的私有 ring slot。
    // 后续 EfDrain/背压 drain/最终 drain 在依赖满足后执行它，这正是 PA 的 Submit 与执行解耦点。
#if PTO_FDWIC_SHARED_MAP
    bool slot_wait_failed = false;
    WaitForSlot<Ops, Profile>(
        state, worker, task_id, stats, slot_wait_failed
    );
    if (slot_wait_failed) {
        return false;
    }
#else
    WaitForSlot<Ops, Profile>(state, worker, task_id, stats);
#endif
#if !PTO_FDWIC_SHARED_MAP
    // private heap_next 是单调 ring 坐标，必须通过 frontier/vend 防止覆盖。
    // shared S3.2 使用有界 shard cursor 且首版禁止回绕，两种坐标不能混用。
    if (!HeapGuard<Ops, Profile>(
            state, worker, task_id, context.output_bytes, stats
        )) {
        return false;
    }
#endif
    const int32_t slot_index = FindFreeSlot(worker);
    if (slot_index < 0) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    PA_GM LocalSlot &slot = worker.slots[slot_index];
    // Match dist_submit_alloc_slot(): reserve and account the private slot
    // before build_ring_slot_from_submit publishes its completed payload.
    // 状态按“occupied 占位 -> built 清零 -> 计入占用 -> BuildSlotPayload”推进；后者会先
    // 置 built，再填充 payload。slot 为 worker 私有、没有跨核发布竞争，所以此处的 built
    // 只是复刻生产状态机与构建成本，不承担对其他核发布完整 payload 的同步语义。
    slot.occupied = true;
    slot.built = 0;
    ++worker.occupied_count;
    if (worker.occupied_count > stats.max_occupied) {
        stats.max_occupied = worker.occupied_count;
    }
    const int32_t sub_block_id = worker.lane == 2 ? 1 : 0;
#if PTO_FDWIC_SHARED_MAP
    BuildSlotPayload<Ops, true>(
        slot, task_id, static_cast<uint32_t>(FunctionId(kind)), 0, args, context, fanin, fanin_count,
        state->shared_map, &stats.trace, sub_block_id
    );
#else
    BuildSlotPayload(
        slot, task_id, static_cast<uint32_t>(FunctionId(kind)), 0, args,
        context, fanin, fanin_count, sub_block_id
    );
#endif
    stats.result.slot_tensor_copies += static_cast<uint32_t>(context.tensor_count);
    stats.result.slot_scalar_copies += static_cast<uint32_t>(context.scalar_count);
    stats.result.fanin_edges += fanin_count;
    return true;
}

#if PTO_FDWIC_SHARED_MAP
PA_DEVICE bool DiscardBuiltTask(
    PA_GM WorkerState &worker, uint32_t task_id
) {
    // 只供“本 task 已 BuildWinner、随后 shared 封口失败”的终止路径使用。
    // slot 为本 worker 私有，可直接撤销；否则 FinalDrain 仍可能执行一个
    // 未完成 shared 最终封口的失败任务。
    for (uint32_t index = 0; index < kPrivateSlots; ++index) {
        PA_GM LocalSlot &slot = worker.slots[index];
        if (!slot.occupied || slot.task_id != task_id) {
            continue;
        }
        const bool accounting_valid = worker.occupied_count != 0;
        slot.built = false;
        slot.occupied = false;
        if (accounting_valid) {
            --worker.occupied_count;
        }
        // 即使 occupied_count 本身已经损坏，终止路径也必须先清掉 slot，
        // 避免 FinalDrain 执行未封口任务；返回 false 保留计数异常证据。
        return accounting_valid;
    }
    return false;
}

template <typename Ops>
PA_DEVICE bool DiscardSharedSlotsAfterReplayFatal(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker,
    LocalStats &stats
) {
    // 只有所有 worker 都已退出 replay、不会再生产 slot 后才能调用。
    // fatal 表示本轮结果已经整体无效；此时未完成 fanin 不可能再获得
    // completion，继续 FinalDrain 只会永久自旋。保留 slot 的 task 与
    // 尚未就绪 fanin 后缀供诊断；已确认 ready 的前缀可能已被 SlotReady
    // 移除。这里只清除本地执行资格与占用计数。
    if (state == nullptr ||
        !IsFatal<Ops>(state, stats, /*task_id=*/-1)) {
        return false;
    }
    uint32_t occupied_slots = 0;
    for (uint32_t index = 0; index < kPrivateSlots; ++index) {
        occupied_slots += worker.slots[index].occupied ? 1U : 0U;
        worker.slots[index].built = false;
        worker.slots[index].occupied = false;
    }
    const bool accounting_valid =
        occupied_slots == worker.occupied_count;
    worker.occupied_count = 0;
    return accounting_valid;
}

// register_mask 只指向已经存在的 Local/GM descriptor。两个地址空间分支
// 必须保持分离，避免 CCEC 把它们合并成不支持的 pointer phi。
PA_DEVICE bool ValidateEmptySharedRegistration(
    const TaskArgs &args, const SubmitContext &context
) {
    uint32_t register_mask = context.register_mask;
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const uint32_t bit = 1U << static_cast<uint32_t>(index);
        if ((register_mask & bit) == 0) {
            continue;
        }
        const TaskTensorRef &reference = args.tensors[index];
        // shared fresh Output 由 (producer,slot) 直接寻址，不能退回
        // region map。manual_dep 的 output_view 同样不是 TensorMap 的
        // 自动 hazard，保留在 task args 但不登记。
        if (reference.kind == TensorRefKind::SharedOutputRef) {
            register_mask &= ~bit;
            continue;
        }
        if (reference.kind == TensorRefKind::GmTensor) {
            PA_GM const TensorDesc &tensor = *reference.pointer.gm_tensor;
            if (!tensor.manual_dep) {
                return false;
            }
        } else if (reference.kind == TensorRefKind::LocalTensor) {
            const TensorDesc &tensor = *reference.pointer.local_tensor;
            if (!tensor.manual_dep) {
                return false;
            }
        } else {
            return false;
        }
        register_mask &= ~bit;
    }
    return register_mask == 0;
}

enum class SharedWriterIntentResult : uint32_t {
    NotRequired = 0,
    Published = 1,
    Failed = 2,
};

PA_DEVICE bool IsSharedWriterIntentTag(TensorArgType tag) {
    return tag == TensorArgType::Inout ||
           tag == TensorArgType::OutputExisting;
}

// 该扫描只回答“参数中是否存在需要自动登记的复写意图”，不依赖 PA
// TaskKind、group 或后继业务形状。manual_dep writer 由调用方显式管理，
// 不进入 shared TensorMap，也不要求 loser 等 writer-ready。
PA_DEVICE bool InspectSharedWriterIntent(
    const TaskArgs &args, bool &required
) {
    required = false;
    if (args.has_error || args.tensor_count < 0 ||
        args.tensor_count > static_cast<int32_t>(kMaxTaskTensors)) {
        return false;
    }
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const TensorArgType tag =
            TaskTag(args, static_cast<uint32_t>(index));
        if (!IsSharedWriterIntentTag(tag)) {
            continue;
        }
        const TaskTensorRef &reference = args.tensors[index];
        if (reference.kind == TensorRefKind::SharedOutputRef) {
            if (!IsPlainSharedOutputRef(
                    SharedOutputReference(reference)
                )) {
                return false;
            }
            required = true;
            continue;
        }
        if (reference.kind == TensorRefKind::GmTensor) {
            if (reference.pointer.gm_tensor == nullptr) {
                return false;
            }
            required |= !reference.pointer.gm_tensor->manual_dep;
            continue;
        }
        if (reference.kind == TensorRefKind::LocalTensor) {
            if (reference.pointer.local_tensor == nullptr) {
                return false;
            }
            required |= !reference.pointer.local_tensor->manual_dep;
            continue;
        }
        return false;
    }
    return true;
}

// AddFanin 的既有接口为固定 PA Case1 静默截断到 16 条；通用 writer
// intent 不能丢失依赖，因此单独使用有返回值的严格版本。负 producer
// 表示 external input，不占 fanin。
PA_DEVICE bool AddSharedWriterIntentFanin(
    int32_t fanin[kMaxFanin], uint32_t &count, int32_t producer
) {
    if (producer < 0) {
        return true;
    }
    for (uint32_t index = 0; index < count; ++index) {
        if (fanin[index] == producer) {
            return true;
        }
    }
    if (count >= kMaxFanin) {
        return false;
    }
    fanin[count++] = producer;
    return true;
}

template <bool Strict>
PA_DEVICE bool AddCollectedSharedFanin(
    int32_t fanin[kMaxFanin], uint32_t &count, int32_t producer
) {
    if constexpr (Strict) {
        return AddSharedWriterIntentFanin(
            fanin, count, producer
        );
    }
    AddFanin(fanin, count, producer);
    return true;
}

template <bool Strict>
PA_DEVICE bool AddCollectedSharedOwner(
    int32_t fanin[kMaxFanin], uint32_t &count, uint64_t owner,
    int32_t reader_task, int32_t reader_lower_bound
) {
    if (owner == kInvalidTaskId) {
        return true;
    }
    const int32_t producer =
        static_cast<int32_t>(owner & 0xFFFFFFFFU);
    if constexpr (Strict) {
        // 新 shared 路径只接受 [N-H,N) 内的真实前任。高 32 位非零、
        // self/future owner 都是协议错误；已经落到窗口左侧的旧 owner
        // 不再形成依赖，但后续 ordinary lookup 仍可找到窗口内的新 writer。
        if (owner > static_cast<uint64_t>(INT32_MAX) ||
            producer >= reader_task) {
            return false;
        }
        if (producer < reader_lower_bound) {
            return true;
        }
    }
    return AddCollectedSharedFanin<Strict>(
        fanin, count, producer
    );
}

template <typename Ops, bool ObserveAtomics = false>
PA_DEVICE bool WaitForSharedOutputPublished(
    PA_GM SharedTensorMapSidecar &map, const FdwicOutputRef &output_ref,
    PA_GM volatile int32_t *fatal, LocalStats *stats = nullptr,
    int32_t task_id = -1,
    AtomicSite site = AtomicSite::SharedFaninOutputPublishedLoad
) {
    // 前置条件：调用者已经用 IsPlainSharedOutputRef 校验 producer/slot/view
    // 范围，并确认 producer_task_id 严格早于当前 consumer task。
    PA_GM volatile int64_t *published =
        &map.shared_outputs[
             static_cast<uint32_t>(output_ref.producer_task_id)
         ].published[output_ref.output_slot].value;
    const int64_t expected =
        static_cast<int64_t>(output_ref.producer_task_id);
    if constexpr (ObserveAtomics) {
        if (stats == nullptr ||
            (site != AtomicSite::SharedFaninOutputPublishedLoad &&
             site != AtomicSite::SharedMetadataOutputPublishedLoad)) {
            return false;
        }
    }
#if !PA_BUILD_TRACE_FREE
    uint64_t trace_begin = 0;
    if constexpr (ObserveAtomics) {
        if (stats != nullptr && AtomicSwimlaneEnabled(stats->trace)) {
            trace_begin = Ops::Now();
        }
    }
#else
    (void)stats;
#endif
#if !PA_BUILD_TRACE_FREE
    uint64_t load_count = 1;
#endif
    // 观察构建以调用者指定的 output-published site 聚合整个等待 episode。
    // PA_ATOMIC_DCCI_SOURCE_EXEMPT: aggregate - 等待结束后统一记录精确 load 数
    int64_t observed = Ops::Load(published);
    if (observed == expected) {
#if !PA_BUILD_TRACE_FREE
        if constexpr (ObserveAtomics) {
            if (trace_begin != 0 && stats != nullptr) {
                const uint64_t trace_end =
                    Ops::NowAfterAtomicResult(observed);
                (void)WriteAggregateAtomicPollBatch(
                    stats->trace, stats->result,
                    site,
                    trace_begin, trace_end, load_count,
                    Ops::kAtomicReturnReadyObserved
                );
            }
        }
#endif
        return true;
    }
    if (observed != -1) {
        if (fatal != nullptr) {
            (void)TraceConfiguredAtomicExchange<
                Ops, ObserveAtomics
            >(
                stats == nullptr ? nullptr : &stats->trace,
                stats == nullptr ? nullptr : &stats->result,
                task_id, AtomicSite::FatalSet, fatal,
                static_cast<int32_t>(1), false
            );
        }
        return false;
    }

    // 只在 producer 尚未发布时建立超时窗口；正常已就绪路径不增加
    // SYS_CNT。轮询对象按 (producer,slot) 分散，不再让所有依赖消费者
    // 争用同一条全局发布前沿。
    const uint64_t begin = Ops::Now();
    uint32_t polls = 0;
    while (true) {
        Ops::SpinHint();
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: aggregate - 循环结束后以一条 PollBatch 精确保留 load_count
        observed = Ops::Load(published);
#if !PA_BUILD_TRACE_FREE
        ++load_count;
#endif
        if (observed == expected) {
#if !PA_BUILD_TRACE_FREE
            if constexpr (ObserveAtomics) {
                if (trace_begin != 0 && stats != nullptr) {
                    const uint64_t trace_end =
                        Ops::NowAfterAtomicResult(observed);
                    (void)WriteAggregateAtomicPollBatch(
                        stats->trace, stats->result,
                        site,
                        trace_begin, trace_end, load_count,
                        Ops::kAtomicReturnReadyObserved
                    );
                }
            }
#endif
            return true;
        }
        if (observed != -1) {
            if (fatal != nullptr) {
                (void)TraceConfiguredAtomicExchange<
                    Ops, ObserveAtomics
                >(
                    stats == nullptr ? nullptr : &stats->trace,
                    stats == nullptr ? nullptr : &stats->result,
                    task_id, AtomicSite::FatalSet, fatal,
                    static_cast<int32_t>(1), false
                );
            }
            return false;
        }
        ++polls;
        if ((polls & 1023U) != 0) {
            continue;
        }
        if (fatal != nullptr &&
            TraceConfiguredAtomicLoad<Ops, ObserveAtomics>(
                stats == nullptr ? nullptr : &stats->trace,
                stats == nullptr ? nullptr : &stats->result,
                task_id, AtomicSite::FatalPoll, fatal
            ) != 0) {
            return false;
        }
        // 这与 metadata predecessor 同属 cross-core owner 进度：
        // 不能用启动屏障的 2 s 门槛把尚未获得运行机会的
        // producer 误判为故障，因此与稀疏 writer 链共用 60 s
        // terminal watchdog。
        if (Ops::Now() - begin > kSharedInsertWatchdogTicks) {
            if (fatal != nullptr) {
                (void)TraceConfiguredAtomicExchange<
                    Ops, ObserveAtomics
                >(
                    stats == nullptr ? nullptr : &stats->trace,
                    stats == nullptr ? nullptr : &stats->result,
                    task_id, AtomicSite::FatalSet, fatal,
                    static_cast<int32_t>(1), false
                );
            }
            return false;
        }
    }
}

// 只供已经取得本 task 有序 insert turn 的调用者使用。对任意合法引用
// producer P < task N，P 的 fresh descriptor 发布先于 P 的 insert
// completion；逐 task predecessor completion 链又先于 N 取得 turn。因此
// 此处若仍未观察到精确的 P，不存在继续轮询后可恢复的正常时序，只能把它
// 视为协议错误。未取得该前提的通用路径必须继续使用上面的 Wait helper。
template <typename Ops, bool ObserveAtomics = false>
PA_DEVICE bool CheckSharedOutputPublishedAfterInsertTurn(
    PA_GM SharedTensorMapSidecar &map,
    const FdwicOutputRef &output_ref, int32_t task_id,
    AtomicSite site, LocalStats *stats
) {
    if (!IsPlainSharedOutputRef(output_ref)) {
        return false;
    }
    PA_GM volatile int64_t *published =
        &map.shared_outputs[
             static_cast<uint32_t>(output_ref.producer_task_id)
         ].published[
             static_cast<uint32_t>(output_ref.output_slot)
         ].value;
    return TraceConfiguredAtomicLoad<Ops, ObserveAtomics>(
               stats == nullptr ? nullptr : &stats->trace,
               stats == nullptr ? nullptr : &stats->result,
               task_id, site, published
           ) ==
           static_cast<int64_t>(output_ref.producer_task_id);
}

PA_DEVICE bool SharedSymbolHistoryKey(
    const FdwicOutputRef &output_ref, uint32_t &key
) {
    if (!IsPlainSharedOutputRef(output_ref)) {
        return false;
    }
    key =
        static_cast<uint32_t>(output_ref.producer_task_id) *
            kSharedOutputMaxPerTask +
        static_cast<uint32_t>(output_ref.output_slot) + 1U;
    return true;
}

PA_DEVICE FdwicOutputRef SharedSymbolHistoryReference(uint32_t key) {
    if (key == 0) {
        return InvalidSharedOutputRef();
    }
    --key;
    const uint32_t producer =
        key / kSharedOutputMaxPerTask;
    const uint32_t slot =
        key % kSharedOutputMaxPerTask;
    if (producer >= kMaxTasks ||
        slot >= kSharedOutputMaxPerTask) {
        return InvalidSharedOutputRef();
    }
    return FdwicOutputRef{
        static_cast<int32_t>(producer),
        static_cast<int16_t>(slot),
        0, 0, 0, 0
    };
}

// writer task 的 history cell 在对应 last_writer CAS 前完整写回，之后
// 保持不可变。latest cache 若指向 reader 的未来 task，就按该 task 的
// 精确 symbol key 取前驱，直到回到 reader 的过去；正常 latest<reader
// 快路完全不读取 history。
template <
    typename Ops,
    bool UsePaAccumulatorGroupWriter = false
>
PA_DEVICE bool ResolveSharedSymbolWriterBefore(
    PA_GM SharedTensorMapSidecar &map,
    const FdwicOutputRef &output_ref, int32_t reader_task,
    int32_t reader_lower_bound, int32_t &resolved_writer,
    LocalStats &stats,
    int32_t pa_accumulator_producer = -1
) {
    uint32_t symbol_key = 0;
    if (!SharedSymbolHistoryKey(output_ref, symbol_key) ||
        reader_task <= output_ref.producer_task_id ||
        reader_task < 0 || reader_lower_bound < 0 ||
        reader_lower_bound > reader_task) {
        return false;
    }
    if constexpr (UsePaAccumulatorGroupWriter) {
        // 正式 PA 的三个 accumulator 由同一 Alloc 产生，并由每个 UP
        // 同步推进。该 specialization 只把这三个 symbol 的 latest 快取
        // 收敛到 slot0；history 查询仍使用原始 slot key，不能把三条依赖
        // 记录合并成一条。
        if (pa_accumulator_producer < 0 ||
            pa_accumulator_producer >= reader_task) {
            return false;
        }
    }
    PA_GM SharedOutputCell &origin =
        map.shared_outputs[
            static_cast<uint32_t>(
                output_ref.producer_task_id
            )
        ];
    uint32_t last_writer_slot =
        static_cast<uint32_t>(output_ref.output_slot);
    if constexpr (UsePaAccumulatorGroupWriter) {
        if (output_ref.producer_task_id ==
                pa_accumulator_producer &&
            last_writer_slot < 3U) {
            last_writer_slot = 0;
        }
    }
    int64_t latest =
        TraceAtomicLoad<Ops>(
            stats.trace, stats.result, reader_task,
            AtomicSite::SharedFaninLastWriterLoad,
            &origin.last_writer[last_writer_slot].value
        );
    uint32_t steps = 0;
    while (latest >= reader_task) {
        if (latest < 0 ||
            latest >= static_cast<int64_t>(kMaxTasks) ||
            steps++ >= kMaxTasks) {
            return false;
        }
        PA_GM SharedWriterHistoryCell &history =
            map.writer_history[static_cast<uint32_t>(latest)];
        // latest CAS 是这份 immutable history 的发布边界。先失效首行
        // 取得 header；若该 task 有超过六个 symbol writer，再只失效
        // 余下实际使用的连续 record 行。
        (void)TraceConfiguredDcciInvalidate<Ops, true>(
            &stats.trace, reader_task, -1,
            DcciSite::SharedFaninHistoryInvalidate,
            &history, 64
        );
        const uint32_t count = history.count;
        if (history.magic != kSharedWriterHistoryMagic ||
            history.writer_task != latest ||
            history.reserved != 0 ||
            count == 0 ||
            count > kSharedWriterHistoryMaxPerTask) {
            return false;
        }
        const uint64_t used_bytes =
            offsetof(SharedWriterHistoryCell, entries) +
            static_cast<uint64_t>(count) *
                sizeof(SharedWriterHistoryRecord);
        if (used_bytes > 64) {
            (void)TraceConfiguredDcciInvalidate<Ops, true>(
                &stats.trace, reader_task, -1,
                DcciSite::SharedFaninHistoryInvalidate,
                &history.entries[6], used_bytes - 64
            );
        }
        bool found = false;
        int32_t previous = -1;
        for (uint32_t index = 0; index < count; ++index) {
            PA_GM const SharedWriterHistoryRecord &record =
                history.entries[index];
            if (record.symbol_key != symbol_key) {
                continue;
            }
            if (found) {
                return false;
            }
            found = true;
            previous = record.previous_writer;
        }
        if (!found ||
            previous < output_ref.producer_task_id ||
            previous >= latest) {
            return false;
        }
        latest = previous;
    }
    if (latest < output_ref.producer_task_id ||
        latest >= reader_task) {
        return false;
    }
    // 与 ordinary ring 使用同一半开窗口：[N-H,N)。history 仍需走到
    // 第一个 <N 的 writer 才能验证链完整；若它已经早于左边界，则
    // 返回 external/no-dependency，而不是把过期 producer 塞进 fanin。
    resolved_writer =
        latest < reader_lower_bound
            ? -1
            : static_cast<int32_t>(latest);
    return true;
}

template <
    typename Ops, bool ChainedWriter = false,
    bool AcceptLatestWriter = false,
    bool UsePaAccumulatorGroupWriter = false,
    bool NoOrdinaryReclaim = false,
    bool TrustOutputPublishedFromInsertChain = false,
    bool WaitOutputPublished = false
>
PA_DEVICE uint32_t CollectSharedFanin(
    PA_GM SharedTensorMapSidecar &map, const TaskArgs &args,
    int32_t task_id, int32_t heap_window, LocalStats &stats,
    int32_t fanin[kMaxFanin], bool &protocol_ok,
    uint32_t &ordinary_lookup_count,
    PA_GM volatile int32_t *fatal = nullptr,
    int32_t chained_producer_task_id = -1,
    int32_t expected_shared_writer = -1,
    int32_t pa_accumulator_producer = -1
) {
    static_assert(
        !(ChainedWriter && AcceptLatestWriter),
        "generic latest-writer lookup must not use the PA chained selector"
    );
    static_assert(
        !UsePaAccumulatorGroupWriter || AcceptLatestWriter,
        "PA accumulator group writer requires latest-writer history lookup"
    );
    static_assert(
        !TrustOutputPublishedFromInsertChain || AcceptLatestWriter,
        "output-publication proof requires the ordered latest-writer path"
    );
    static_assert(
        !WaitOutputPublished || !TrustOutputPublishedFromInsertChain,
        "direct output wait and insert-chain publication proof are exclusive"
    );
    protocol_ok = true;
    ordinary_lookup_count = 0;
    if (task_id < 0 || heap_window < 0 ||
        args.tensor_count < 0 ||
        args.tensor_count > static_cast<int32_t>(kMaxTaskTensors)) {
        protocol_ok = false;
        return 0;
    }
    if constexpr (UsePaAccumulatorGroupWriter) {
        if (pa_accumulator_producer < 0 ||
            pa_accumulator_producer >= task_id) {
            protocol_ok = false;
            return 0;
        }
    }
    const int32_t reader_lower_bound =
        task_id > heap_window ? task_id - heap_window : 0;
    if constexpr (ChainedWriter) {
        // PA 的三个 accumulator 共用同一个 Alloc producer，后续每个 UP
        // 同步推进这三个 slot。显式的 (producer,writer) 对只选择这一
        // symbol cell；本组 SF/PV 等 fresh refs 仍按自己的 producer
        // 校验。selector 必须至少命中一个消费引用，不能传错后静默退化。
        if (chained_producer_task_id < 0 ||
            chained_producer_task_id >= expected_shared_writer ||
            expected_shared_writer >= task_id) {
            protocol_ok = false;
            return 0;
        }
        bool matched_chain_ref = false;
        for (int32_t index = 0; index < args.tensor_count; ++index) {
            const TensorArgType tag =
                TaskTag(args, static_cast<uint32_t>(index));
            if (tag == TensorArgType::Output ||
                (tag != TensorArgType::Input &&
                 tag != TensorArgType::Inout &&
                 tag != TensorArgType::OutputExisting)) {
                continue;
            }
            const TaskTensorRef &reference = args.tensors[index];
            if (reference.kind != TensorRefKind::SharedOutputRef) {
                continue;
            }
            matched_chain_ref |=
                SharedOutputReference(reference).producer_task_id ==
                    chained_producer_task_id;
        }
        if (!matched_chain_ref) {
            protocol_ok = false;
            return 0;
        }
    }

    int32_t validated_fanin[kMaxFanin] = {};
    uint32_t validated_count = 0;
    uint32_t validated_ordinary_lookups = 0;
    uint32_t validated_input_loads = 0;

    // 只读取并校验，不修改 last_writer、统计或输出 fanin。旧 PA
    // chained-writer 路径仍按自己的 registration/Build 边界提交；
    // 独立 shared ordered-insert 路径则在进入这里前已经发布本 task
    // writer history，所以 AcceptLatestWriter 必须沿 history 回退到 <N。
    // 两种路径都先完成本轮只读校验，后续引用非法时不会再留下 writer 更新。
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const TensorArgType tag =
            TaskTag(args, static_cast<uint32_t>(index));
        if (tag == TensorArgType::Output) {
            continue;
        }
        const TaskTensorRef &reference = args.tensors[index];
        if (reference.kind == TensorRefKind::SharedOutputRef) {
            // 当前只接收普通 fresh Output；view ABI 已占位但尚未接入，
            // 不能静默把带 view 的符号当成 plain descriptor 使用。
            const FdwicOutputRef output_ref = SharedOutputReference(reference);
            if (!IsPlainSharedOutputRef(output_ref) ||
                output_ref.producer_task_id < 0 ||
                output_ref.producer_task_id >= task_id) {
                protocol_ok = false;
                return 0;
            }
            if constexpr (!TrustOutputPublishedFromInsertChain) {
                bool output_published = false;
                if constexpr (WaitOutputPublished) {
                    // 稀疏 metadata-writer 链不再证明所有前序
                    // fresh output 都已发布；每个 consumer 直接观察
                    // 自己实际使用的 (producer, slot) 独占完成字。
                    output_published =
                        WaitForSharedOutputPublished<Ops, true>(
                            map, output_ref, fatal,
                            &stats, task_id
                        );
                } else if constexpr (AcceptLatestWriter) {
                    // 旧 ordered 调用者只证明“不必等待”，仍保留一次
                    // published 权威校验；完整 completion-chain 证明由
                    // 独立 template 身份在编译期删除这枚 load。
                    output_published =
                        CheckSharedOutputPublishedAfterInsertTurn<
                            Ops, true
                        >(
                            map, output_ref, task_id,
                            AtomicSite::SharedFaninOutputPublishedLoad,
                            &stats
                        );
                } else {
                    output_published =
                        WaitForSharedOutputPublished<Ops>(
                            map, output_ref, fatal
                        );
                }
                if (!output_published) {
                    protocol_ok = false;
                    return 0;
                }
            }
            if (tag != TensorArgType::Input &&
                tag != TensorArgType::Inout &&
                tag != TensorArgType::OutputExisting) {
                protocol_ok = false;
                return 0;
            }
            // 同一 task 对同一 symbol 最多只能有一个写引用，否则后面的
            // writer 提交会把本 task 自己误当成预期 producer。
            if (tag == TensorArgType::Inout ||
                tag == TensorArgType::OutputExisting) {
                for (int32_t previous = 0; previous < index; ++previous) {
                    const TensorArgType previous_tag =
                        TaskTag(args, static_cast<uint32_t>(previous));
                    if (previous_tag != TensorArgType::Inout &&
                        previous_tag != TensorArgType::OutputExisting) {
                        continue;
                    }
                    const TaskTensorRef &previous_ref = args.tensors[previous];
                    if (previous_ref.kind != TensorRefKind::SharedOutputRef) {
                        continue;
                    }
                    const FdwicOutputRef previous_output =
                        SharedOutputReference(previous_ref);
                    if (previous_output.producer_task_id ==
                            output_ref.producer_task_id &&
                        previous_output.output_slot ==
                            output_ref.output_slot) {
                        protocol_ok = false;
                        return 0;
                    }
                }
            }
            int32_t writer = -1;
            if constexpr (AcceptLatestWriter) {
                // latest cell 是零开销快取；若 future writer 已经覆盖它，
                // 只在这一慢路沿不可变前驱链回到 max(writer<task_id)，
                // 再与 ordinary lookup 一样过滤到 [N-H,N)。
                if (!ResolveSharedSymbolWriterBefore<
                        Ops, UsePaAccumulatorGroupWriter
                    >(
                        map, output_ref, task_id,
                        reader_lower_bound, writer, stats,
                        pa_accumulator_producer
                    )) {
                    protocol_ok = false;
                    return 0;
                }
            } else {
                // PA 迁移完成前保留原来的精确 oracle：默认单组要求
                // writer==descriptor producer，ChainedWriter 只允许调用方
                // 指定的 accumulator 链。两种口径不能静默混用。
                PA_GM SharedOutputCell &cell =
                    map.shared_outputs[
                        static_cast<uint32_t>(
                            output_ref.producer_task_id
                        )
                    ];
                const bool chained_ref =
                    ChainedWriter &&
                    output_ref.producer_task_id ==
                        chained_producer_task_id;
                const int32_t expected_writer =
                    chained_ref
                        ? expected_shared_writer
                        : output_ref.producer_task_id;
                writer = static_cast<int32_t>(
                    // PA_ATOMIC_DCCI_SOURCE_EXEMPT: test-only - AcceptLatestWriter=false 的旧 PA oracle 分支不进入当前 shared scheduler
                    Ops::Load(
                        &cell.last_writer[
                            output_ref.output_slot
                        ].value
                    )
                );
                if (expected_writer < output_ref.producer_task_id ||
                    expected_writer >= task_id ||
                    writer != expected_writer) {
                    protocol_ok = false;
                    return 0;
                }
            }
            if (!AddCollectedSharedFanin<AcceptLatestWriter>(
                    validated_fanin, validated_count,
                    writer
                )) {
                protocol_ok = false;
                return 0;
            }
            if (tag == TensorArgType::Input) {
                ++validated_input_loads;
            }
            continue;
        }
        if (reference.kind == TensorRefKind::GmTensor) {
            PA_GM const TensorDesc &tensor = *reference.pointer.gm_tensor;
            if (tensor.manual_dep) {
                continue;
            }
            const uint64_t owner = tensor.owner_task_id;
            if (!AddCollectedSharedOwner<
                    AcceptLatestWriter
                >(
                    validated_fanin, validated_count, owner,
                    task_id, reader_lower_bound
                )) {
                protocol_ok = false;
                return 0;
            }
            if (tag == TensorArgType::Inout ||
                tag == TensorArgType::OutputExisting ||
                (tag == TensorArgType::Input &&
                 (owner != kInvalidTaskId || AcceptLatestWriter))) {
                bool lookup_ok = false;
                const int32_t producer =
                    SharedLookupTensor<
                        Ops, true, NoOrdinaryReclaim
                    >(
                    map, tensor, task_id, heap_window, lookup_ok,
                    &stats.trace, &stats.result
                );
                if (!lookup_ok) {
                    protocol_ok = false;
                    return 0;
                }
                ++validated_ordinary_lookups;
                if (!AddCollectedSharedFanin<
                        AcceptLatestWriter
                    >(
                        validated_fanin, validated_count, producer
                    )) {
                    protocol_ok = false;
                    return 0;
                }
            }
        } else if (reference.kind == TensorRefKind::LocalTensor) {
            const TensorDesc &tensor = *reference.pointer.local_tensor;
            if (tensor.manual_dep) {
                continue;
            }
            const uint64_t owner = tensor.owner_task_id;
            if (!AddCollectedSharedOwner<
                    AcceptLatestWriter
                >(
                    validated_fanin, validated_count, owner,
                    task_id, reader_lower_bound
                )) {
                protocol_ok = false;
                return 0;
            }
            if (tag == TensorArgType::Inout ||
                tag == TensorArgType::OutputExisting ||
                (tag == TensorArgType::Input &&
                 (owner != kInvalidTaskId || AcceptLatestWriter))) {
                bool lookup_ok = false;
                const int32_t producer =
                    SharedLookupTensor<
                        Ops, true, NoOrdinaryReclaim
                    >(
                    map, tensor, task_id, heap_window, lookup_ok,
                    &stats.trace, &stats.result
                );
                if (!lookup_ok) {
                    protocol_ok = false;
                    return 0;
                }
                ++validated_ordinary_lookups;
                if (!AddCollectedSharedFanin<
                        AcceptLatestWriter
                    >(
                        validated_fanin, validated_count, producer
                    )) {
                    protocol_ok = false;
                    return 0;
                }
            }
        } else {
            protocol_ok = false;
            return 0;
        }
    }

    // 全部引用只读校验通过后，才一次性发布统计与 fanin 结果。INPUT 次数
    // 在验证扫描中先落局部量，保留 late-failure 的 all-or-nothing 口径。
    stats.result.shared_symbol_input_loads += validated_input_loads;
    ordinary_lookup_count = validated_ordinary_lookups;
    for (uint32_t edge = 0; edge < validated_count; ++edge) {
        fanin[edge] = validated_fanin[edge];
    }
    return validated_count;
}

template <typename Ops>
PA_DEVICE bool PublishSharedWriterReady(
    PA_GM SchedulerState *state, int32_t task_id
) {
    if (state == nullptr || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks)) {
        return false;
    }
    // writer 登记必须先于门值对 loser 可见；本 task 是否已经执行完成
    // 仍由它自己的 completion flag 表达，不能把 deps_prepared 冒充成
    // 可执行/已完成。CAS 只允许初始化 sentinel -> task_id：重复 winner
    // 或错误 task-cell 复用不会先写入一个合法门值再报告失败。
    Ops::StoreBarrier();
    // PA_ATOMIC_DCCI_SOURCE_EXEMPT: test-only - 旧 writer-ready 协议只供隔离门槛，正式路径直接发布 task completion
    return Ops::CompareExchange(
               &state->tasks[static_cast<uint32_t>(task_id)].deps_prepared,
               static_cast<int64_t>(-1),
               static_cast<int64_t>(task_id)
           ) == -1;
}

template <typename Ops>
PA_DEVICE bool WaitForSharedWriterReady(
    PA_GM SchedulerState *state, int32_t task_id, LocalStats &stats
) {
    if (state == nullptr || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks)) {
        return false;
    }
    PA_GM volatile int64_t *prepared =
        &state->tasks[static_cast<uint32_t>(task_id)].deps_prepared;
    // PA_ATOMIC_DCCI_SOURCE_EXEMPT: test-only - 旧 writer-ready 等待只供隔离门槛
    int64_t observed = Ops::Load(prepared);
    if (observed == task_id) {
        return true;
    }
    if (observed != -1) {
        SetFatal<Ops>(state, stats, task_id);
        return false;
    }

    const uint64_t begin = Ops::Now();
    uint32_t polls = 0;
    while (true) {
        Ops::SpinHint();
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: test-only - 旧 writer-ready 等待只供隔离门槛
        observed = Ops::Load(prepared);
        if (observed == task_id) {
            return true;
        }
        if (observed != -1) {
            SetFatal<Ops>(state, stats, task_id);
            return false;
        }
        ++polls;
        if ((polls & 1023U) != 0) {
            continue;
        }
        if (IsFatal<Ops>(state, stats, task_id)) {
            return false;
        }
        if (Ops::Now() - begin > kWatchdogTicks) {
            SetFatal<Ops>(state, stats, task_id);
            return false;
        }
    }
}

PA_DEVICE bool CheckedMultiplyU64ByU32(
    uint64_t left, uint32_t right, uint64_t &product
) {
    // CCEC 9.1 会把“UINT64_MAX/right 预检 + 64-bit 乘法”融合成
    // AICore 环境没有运行库实现的 __multi3。按 32-bit limb 展开后只需
    // 32x32->64 乘法，同时保留 generic TensorDesc 原有的 uint64 extent
    // 取值域，不能为迁就编译器把合法的大 ordinary region 收窄到 32 bit。
    const uint64_t low_product =
        static_cast<uint64_t>(
            static_cast<uint32_t>(left)
        ) * right;
    const uint64_t high_product =
        static_cast<uint64_t>(
            static_cast<uint32_t>(left >> 32)
        ) * right;
    const uint64_t carry = low_product >> 32;
    if (high_product > UINT32_MAX - carry) {
        return false;
    }
    product =
        ((high_product + carry) << 32) |
        static_cast<uint32_t>(low_product);
    return true;
}

template <typename TensorReference>
PA_DEVICE bool MakeValidatedSharedWriterRegion(
    const TensorReference &tensor, int32_t task_id,
    SharedRegionValue &region
) {
    if (task_id < 0 || tensor.dtype >= DataType::Count ||
        tensor.ndims == 0 || tensor.ndims > kMaxTensorDims) {
        return false;
    }
    const uint64_t element_size = ElementSize(tensor.dtype);
    if (element_size == 0) {
        return false;
    }
    uint64_t extent = tensor.extent_elem_cache;
    if (tensor.is_contiguous) {
        extent = 1;
        for (uint32_t dimension = 0;
             dimension < tensor.ndims; ++dimension) {
            const uint32_t shape = tensor.shapes[dimension];
            uint64_t next_extent = 0;
            if (shape == 0 ||
                !CheckedMultiplyU64ByU32(
                    extent, shape, next_extent
                )) {
                return false;
            }
            extent = next_extent;
        }
    }
    uint32_t element_shift = 0;
    if (element_size == 8) {
        element_shift = 3;
    } else if (element_size == 4) {
        element_shift = 2;
    } else if (element_size == 2) {
        element_shift = 1;
    } else if (element_size != 1) {
        return false;
    }
    if (extent == 0 ||
        tensor.start_offset > UINT64_MAX - extent) {
        return false;
    }
    const uint64_t end_offset = tensor.start_offset + extent;
    const uint64_t max_element_offset =
        UINT64_MAX >> element_shift;
    if (tensor.start_offset > max_element_offset ||
        end_offset > max_element_offset) {
        return false;
    }
    region.buffer_addr = tensor.buffer_addr;
    region.lo = tensor.start_offset << element_shift;
    region.hi = end_offset << element_shift;
    region.producer = task_id;
    region.reserved = 0;
    return region.lo < region.hi;
}

template <typename TensorReference>
PA_DEVICE bool ValidateOrdinarySharedWriterOwner(
    const TensorReference &tensor, int32_t task_id
) {
    if (tensor.owner_task_id == kInvalidTaskId) {
        return true;
    }
    if (tensor.owner_task_id >
        static_cast<uint64_t>(INT32_MAX)) {
        return false;
    }
    const int32_t owner = static_cast<int32_t>(
        tensor.owner_task_id
    );
    return owner >= 0 && owner < task_id;
}

template <typename TensorReference>
PA_DEVICE bool ValidateOrdinarySharedWriterReference(
    const TensorReference &tensor, int32_t task_id
) {
    if (tensor.manual_dep) {
        return true;
    }
    SharedRegionValue unused{};
    if (!MakeValidatedSharedWriterRegion(tensor, task_id, unused)) {
        return false;
    }
    return ValidateOrdinarySharedWriterOwner(tensor, task_id);
}

// 在执行任一 atomic/region append 前先完成所有 writer 引用的结构校验。
// symbol 重复 writer 会让第二次 CAS 把本 task 自己当成旧 writer，因此
// 必须在第一项改写之前拒绝。ordinary 多 view 可以合法重叠，不在这里
// 按地址去重。
PA_DEVICE bool ValidateSharedWriterIntentSet(
    const TaskArgs &args, int32_t task_id
) {
    bool required = false;
    if (task_id < 0 ||
        !InspectSharedWriterIntent(args, required) ||
        !required) {
        return false;
    }
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const TensorArgType tag =
            TaskTag(args, static_cast<uint32_t>(index));
        if (!IsSharedWriterIntentTag(tag)) {
            continue;
        }
        const TaskTensorRef &reference = args.tensors[index];
        if (reference.kind == TensorRefKind::SharedOutputRef) {
            const FdwicOutputRef output_ref =
                SharedOutputReference(reference);
            if (output_ref.producer_task_id < 0 ||
                output_ref.producer_task_id >= task_id) {
                return false;
            }
            for (int32_t previous = 0; previous < index; ++previous) {
                if (!IsSharedWriterIntentTag(
                        TaskTag(
                            args, static_cast<uint32_t>(previous)
                        )
                    )) {
                    continue;
                }
                const TaskTensorRef &previous_ref =
                    args.tensors[previous];
                if (previous_ref.kind !=
                    TensorRefKind::SharedOutputRef) {
                    continue;
                }
                const FdwicOutputRef previous_output =
                    SharedOutputReference(previous_ref);
                if (previous_output.producer_task_id ==
                        output_ref.producer_task_id &&
                    previous_output.output_slot ==
                        output_ref.output_slot) {
                    return false;
                }
            }
            continue;
        }
        if (reference.kind == TensorRefKind::GmTensor) {
            if (!ValidateOrdinarySharedWriterReference(
                    *reference.pointer.gm_tensor, task_id
                )) {
                return false;
            }
            continue;
        }
        if (reference.kind == TensorRefKind::LocalTensor) {
            if (!ValidateOrdinarySharedWriterReference(
                    *reference.pointer.local_tensor, task_id
                )) {
                return false;
            }
            continue;
        }
        return false;
    }
    return true;
}

template <typename Ops, typename TensorReference>
PA_DEVICE bool CommitOrdinarySharedWriterIntent(
    PA_GM SharedTensorMapSidecar &map,
    const TensorReference &tensor, int32_t task_id,
    int32_t heap_window, int32_t fanin[kMaxFanin],
    uint32_t &fanin_count, LocalStats &stats
) {
    if (tensor.manual_dep) {
        return true;
    }
    if (tensor.owner_task_id != kInvalidTaskId) {
        if (tensor.owner_task_id >
            static_cast<uint64_t>(INT32_MAX)) {
            return false;
        }
        const int32_t owner = static_cast<int32_t>(
            tensor.owner_task_id
        );
        if (owner < 0 || owner >= task_id ||
            !AddSharedWriterIntentFanin(
                fanin, fanin_count, owner
            )) {
            return false;
        }
    }

    bool lookup_ok = false;
    const int32_t previous = SharedLookupTensor<Ops, true>(
        map, tensor, task_id, heap_window, lookup_ok,
        &stats.trace, &stats.result
    );
    if (!lookup_ok ||
        !AddSharedWriterIntentFanin(
            fanin, fanin_count, previous
        )) {
        return false;
    }
    ++stats.result.map_lookups;

    SharedRegionValue entry{};
    if (!MakeValidatedSharedWriterRegion(
            tensor, task_id, entry
        )) {
        return false;
    }
    // 通用 writer-ready 目前只证明 writer publication 的先后，尚未
    // 证明所有更早 reader 已结束。这里保持 append-only，不按 task_id
    // 推进 head；容量耗尽走 terminal failure，不能用可能仍被慢 reader
    // 扫描的槽换取表面上的无限回绕。
    if (SharedCheckTaskAppend<Ops>(
            map, &entry, 1, -1
        ) != SharedAppendCheck::Ready ||
        !SharedAppendPreparedEntry<Ops>(map, entry)) {
        return false;
    }
    ++stats.result.map_inserts;
    return true;
}

template <typename Ops>
PA_DEVICE bool CommitSymbolSharedWriterIntentSet(
    PA_GM SharedTensorMapSidecar &map, const TaskArgs &args,
    int32_t task_id,
    int32_t fanin[kMaxFanin], uint32_t &fanin_count,
    LocalStats &stats, PA_GM volatile int32_t *fatal
) {
    // 调用方必须保证同一 symbol 的 writer 按 task_id 单调进入本函数。
    // 独立 shared Submit 由全局 insert turn 建立这一顺序；仍保留的隔离
    // driver 则必须提供等价的唯一 ordered writer 合同。CAS 负责发现
    // 乱序或重复 owner，但不会替调用方补回已被跨越的 writer。
    if (task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks)) {
        return false;
    }
    PA_GM SharedWriterHistoryCell &history =
        map.writer_history[static_cast<uint32_t>(task_id)];

    uint32_t count = 0;
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        if (!IsSharedWriterIntentTag(
                TaskTag(args, static_cast<uint32_t>(index))
            )) {
            continue;
        }
        const TaskTensorRef &reference = args.tensors[index];
        if (reference.kind != TensorRefKind::SharedOutputRef) {
            continue;
        }
        if (count >= kSharedWriterHistoryMaxPerTask) {
            return false;
        }
        const FdwicOutputRef output_ref =
            SharedOutputReference(reference);
        uint32_t symbol_key = 0;
        if (!SharedSymbolHistoryKey(output_ref, symbol_key) ||
            !WaitForSharedOutputPublished<Ops>(
                map, output_ref, fatal
            )) {
            return false;
        }
        PA_GM volatile int64_t *last_writer =
            &map.shared_outputs[
                 static_cast<uint32_t>(
                     output_ref.producer_task_id
                 )
             ].last_writer[
                 static_cast<uint32_t>(output_ref.output_slot)
             ].value;
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: test-only - generic writer-intent helper 不进入当前 prepared ordered Submit
        const int64_t previous = Ops::Load(last_writer);
        if (previous < output_ref.producer_task_id ||
            previous >= task_id ||
            !AddSharedWriterIntentFanin(
                fanin, fanin_count,
                static_cast<int32_t>(previous)
            )) {
            return false;
        }
        history.entries[count].symbol_key = symbol_key;
        history.entries[count].previous_writer =
            static_cast<int32_t>(previous);
        ++count;
    }
    if (count == 0) {
        return true;
    }

    history.magic = kSharedWriterHistoryMagic;
    history.writer_task = task_id;
    history.count = count;
    history.reserved = 0;
    const uint64_t history_bytes =
        offsetof(SharedWriterHistoryCell, entries) +
        static_cast<uint64_t>(count) *
            sizeof(SharedWriterHistoryRecord);
    (void)TraceConfiguredDcciFlush<Ops, false>(
        nullptr, task_id, -1,
        DcciSite::SharedWriterHistoryFlush,
        &history, history_bytes
    );
    Ops::StoreBarrier();

    // last_writer CAS 是每条前驱记录的发布边界。history 已整体写回，
    // 因而 reader 观察到任一 current task 后都能按 key 取到其前驱。
    for (uint32_t index = 0; index < count; ++index) {
        PA_GM const SharedWriterHistoryRecord &record =
            history.entries[index];
        const FdwicOutputRef output_ref =
            SharedSymbolHistoryReference(record.symbol_key);
        if (!IsPlainSharedOutputRef(output_ref)) {
            return false;
        }
        PA_GM volatile int64_t *last_writer =
            &map.shared_outputs[
                 static_cast<uint32_t>(
                     output_ref.producer_task_id
                 )
             ].last_writer[
                 static_cast<uint32_t>(output_ref.output_slot)
             ].value;
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: test-only - generic writer-intent helper 不进入当前 prepared ordered Submit
        if (Ops::CompareExchange(
                last_writer,
                static_cast<int64_t>(record.previous_writer),
                static_cast<int64_t>(task_id)
            ) != record.previous_writer) {
            return false;
        }
        // 多 symbol 发布不是事务；若后项冲突，已经线性化的前缀不回滚。
        // 逐项计数保留故障现场，外层随后设置 fatal 且不发布 ready gate。
        ++stats.result.shared_symbol_inout_commits;
    }
    return true;
}

// 正式 PA-UP 专路的第一段：writer shape 已在 Materialize 尾部验证，
// 本 task 又独占 writer_history[task_id]，所以可以在等待 predecessor
// 之前发布 immutable payload。它此时只是不可达的准备态；随后对 Alloc
// slot0 的 group-writer CAS 才统一发布三条 symbol history。
template <typename Ops, bool ObserveDcci = false>
PA_DEVICE bool PublishTrustedPaUpWriterHistoryPayload(
    PA_GM SharedTensorMapSidecar &map,
    const uint32_t *symbol_keys, int32_t task_id,
    int32_t expected_previous, int32_t expected_producer,
    LocalStats *stats = nullptr
#if !PA_BUILD_TRACE_FREE
    , DeferredSharedWriterMetadataTrace *deferred_trace = nullptr
#endif
) {
    if (symbol_keys == nullptr || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks) ||
        expected_producer < 0 ||
        expected_producer >= task_id ||
        expected_previous < expected_producer ||
        expected_previous >= task_id) {
        return false;
    }
    const uint32_t key_base =
        static_cast<uint32_t>(expected_producer) *
            kSharedOutputMaxPerTask +
        1U;
    if (symbol_keys[0] != key_base + 2U ||
        symbol_keys[1] != key_base + 1U ||
        symbol_keys[2] != key_base) {
        return false;
    }

    PA_GM SharedWriterHistoryCell &history =
        map.writer_history[static_cast<uint32_t>(task_id)];
    PA_LOOP_NOUNROLL
    for (uint32_t index = 0; index < 3; ++index) {
        history.entries[index].symbol_key = symbol_keys[index];
        history.entries[index].previous_writer =
            expected_previous;
    }
    history.magic = kSharedWriterHistoryMagic;
    history.writer_task = task_id;
    history.count = 3;
    history.reserved = 0;
    constexpr uint64_t kHistoryBytes =
        offsetof(SharedWriterHistoryCell, entries) +
        3U * sizeof(SharedWriterHistoryRecord);
#if PA_BUILD_TRACE_FREE
    (void)TraceConfiguredDcciFlush<Ops, ObserveDcci>(
        stats == nullptr ? nullptr : &stats->trace,
        task_id, -1, DcciSite::SharedWriterHistoryFlush,
        &history, kHistoryBytes
    );
#else
    if (deferred_trace != nullptr && stats != nullptr) {
        deferred_trace->history_dcci_lines =
            CaptureDcciFlush<Ops>(
                stats->trace, &history, kHistoryBytes,
                deferred_trace->history_dcci_begin,
                deferred_trace->history_dcci_end
            );
    } else {
        (void)TraceConfiguredDcciFlush<Ops, ObserveDcci>(
            stats == nullptr ? nullptr : &stats->trace,
            task_id, -1, DcciSite::SharedWriterHistoryFlush,
            &history, kHistoryBytes
        );
    }
#endif
    Ops::StoreBarrier();
    return true;
}

// 正式 PA-UP 专路的第二段：上面的 payload 已经 DCCI+DSB 完成，且
// predecessor turn 已由调用方取得。三个 accumulator 在 PA 中始终同步
// 推进，因此只用 Alloc cell.last_writer[0] 作为 group latest，并执行
// 一次 return-ready CAS；三条原始 symbol history 仍完整保留。
template <
    typename Ops, bool ObserveAtomics = false,
    bool TrustDeferredTrace = false
>
PA_DEVICE bool CommitTrustedPaUpGroupWriter(
    PA_GM SharedTensorMapSidecar &map, int32_t task_id,
    int32_t expected_previous, int32_t expected_producer,
    LocalStats *stats = nullptr
#if !PA_BUILD_TRACE_FREE
    , DeferredSharedWriterMetadataTrace *deferred_trace = nullptr
#endif
) {
    PA_GM volatile int64_t *last_writer =
        &map.shared_outputs[
             static_cast<uint32_t>(expected_producer)
         ].last_writer[0].value;
#if PA_BUILD_TRACE_FREE
    return TraceConfiguredAtomicCompareExchange<
            Ops, ObserveAtomics
        >(
            stats == nullptr ? nullptr : &stats->trace,
            stats == nullptr ? nullptr : &stats->result,
            task_id,
            AtomicSite::SharedMetadataLastWriterCommit,
            last_writer,
            static_cast<int64_t>(expected_previous),
            static_cast<int64_t>(task_id)
        ) == expected_previous;
#else
    int64_t observed = INT64_MIN;
    if constexpr (TrustDeferredTrace) {
        // 正式 full-swimlane 调用提供 owner-local capture；只保存一次
        // group CAS 的发射与返回就绪端点，随后在串行区外写 raw。
        observed = CaptureAtomicCompareExchange<Ops>(
            stats->trace, last_writer,
            static_cast<int64_t>(expected_previous),
            static_cast<int64_t>(task_id),
            deferred_trace->writer_cas_begin[0],
            deferred_trace->writer_cas_end[0]
        );
        deferred_trace->writer_cas_count = 1U;
    } else if (deferred_trace != nullptr && stats != nullptr &&
               deferred_trace->writer_cas_count == 0) {
        observed = CaptureAtomicCompareExchange<Ops>(
            stats->trace, last_writer,
            static_cast<int64_t>(expected_previous),
            static_cast<int64_t>(task_id),
            deferred_trace->writer_cas_begin[0],
            deferred_trace->writer_cas_end[0]
        );
        deferred_trace->writer_cas_count = 1U;
    } else {
        observed =
            TraceConfiguredAtomicCompareExchange<
                Ops, ObserveAtomics
            >(
                stats == nullptr ? nullptr : &stats->trace,
                stats == nullptr ? nullptr : &stats->result,
                task_id,
                AtomicSite::SharedMetadataLastWriterCommit,
                last_writer,
                static_cast<int64_t>(expected_previous),
                static_cast<int64_t>(task_id)
            );
    }
    return observed == expected_previous;
#endif
}

// ordered Submit 专用入口：调用方已经在 insert turn 外完成 symbol ref
// 校验、去重和 packed-key 生成。这里不再扫描 args，也不构造随后会被
// 丢弃的 fanin；previous writer 仍必须在取得 turn 后读取，才能写入当前
// task 的不可变 history。通用 CommitSymbolSharedWriterIntentSet 继续保留
// 原有等待 publication、收集 fanin 和逐项统计的合同，二者不能互换。
template <
    typename Ops, bool ObserveAtomics = false,
    bool CheckOutputPublished = true,
    bool UseExpectedPrevious = false,
    bool UsePaUpShape = false,
    bool TrustPreparedPaShape = false
>
PA_DEVICE bool CommitPreparedSymbolSharedWriterIntentSet(
    PA_GM SharedTensorMapSidecar &map,
    const uint32_t *symbol_keys, uint32_t symbol_count,
    int32_t task_id, PA_GM volatile int32_t *fatal,
    LocalStats *stats = nullptr,
    int32_t expected_previous = -1,
    int32_t expected_producer = -1
#if !PA_BUILD_TRACE_FREE
    , DeferredSharedWriterMetadataTrace *deferred_trace = nullptr
#endif
) {
    // 正式 ordered Submit 在 task-level completion 成功后统一记录完整
    // transaction；本 helper 固定不产生逐项成功统计，避免部分 CAS 前缀
    // 与 task-level 计数混成两种口径。
    if (task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks) ||
        symbol_count > kSharedWriterHistoryMaxPerTask ||
        (symbol_count != 0 && symbol_keys == nullptr)) {
        return false;
    }
    static_assert(
        !UsePaUpShape || UseExpectedPrevious,
        "PA UP shape requires an expected previous writer"
    );
    static_assert(
        !TrustPreparedPaShape || UsePaUpShape,
        "only the PA UP path can trust a prepared writer shape"
    );
    if (symbol_count == 0) {
        if constexpr (!UsePaUpShape) {
            return true;
        }
        // PA UP 专路不能把“应有三个 writer、实际一个都没有”误认成
        // 合法的空 writer task；继续进入形状校验并在首次 GM 写前失败。
    }
    if constexpr (UseExpectedPrevious) {
        if (expected_previous < 0 ||
            expected_previous >= task_id) {
            return false;
        }
    }

    // 正式 PA UP 的 callback 按 accumulated_max/sum/output 构造参数，
    // Prepare 又按参数 index 原序收集，所以 writer key 必须精确对应同一
    // Alloc producer 的 slot 2/1/0。收紧为精确顺序后无需在有序区做
    // 三次除法/取模、去重或 packed-slot 搬运；任何漂移仍在首次 GM
    // history 写之前失败。
    if constexpr (UsePaUpShape && !TrustPreparedPaShape) {
        const bool producer_valid =
            expected_producer >= 0 &&
            expected_producer < task_id &&
            expected_producer <
                static_cast<int32_t>(kMaxTasks);
        const uint32_t key_base = producer_valid
            ? static_cast<uint32_t>(expected_producer) *
                  kSharedOutputMaxPerTask +
                  1U
            : 0U;
        const bool shape_valid =
            symbol_count == 3 &&
            producer_valid &&
            expected_previous >= expected_producer &&
            symbol_keys[0] == key_base + 2U &&
            symbol_keys[1] == key_base + 1U &&
            symbol_keys[2] == key_base;
        if (!shape_valid) {
            if (fatal != nullptr) {
                (void)TraceConfiguredAtomicExchange<
                    Ops, ObserveAtomics
                >(
                    stats == nullptr ? nullptr : &stats->trace,
                    stats == nullptr ? nullptr : &stats->result,
                    task_id, AtomicSite::FatalSet,
                    fatal, static_cast<int32_t>(1),
                    /*result_used=*/false
                );
            }
            return false;
        }
    }

    PA_GM SharedWriterHistoryCell &history =
        map.writer_history[static_cast<uint32_t>(task_id)];
    for (uint32_t index = 0; index < symbol_count; ++index) {
        const uint32_t symbol_key = symbol_keys[index];
        int64_t previous =
            static_cast<int64_t>(expected_previous);
        if constexpr (!UsePaUpShape) {
            const FdwicOutputRef output_ref =
                SharedSymbolHistoryReference(symbol_key);
            const bool valid_ref =
                IsPlainSharedOutputRef(output_ref) &&
                output_ref.producer_task_id < task_id;
            bool published = true;
            if constexpr (CheckOutputPublished) {
                published =
                    valid_ref &&
                    CheckSharedOutputPublishedAfterInsertTurn<
                        Ops, ObserveAtomics
                    >(
                        map, output_ref, task_id,
                        AtomicSite::SharedMetadataOutputPublishedLoad,
                        stats
                    );
            }
            if (!valid_ref || !published) {
                if (fatal != nullptr) {
                    (void)TraceConfiguredAtomicExchange<
                        Ops, ObserveAtomics
                    >(
                        stats == nullptr ? nullptr : &stats->trace,
                        stats == nullptr ? nullptr : &stats->result,
                        task_id, AtomicSite::FatalSet,
                        fatal, static_cast<int32_t>(1),
                        /*result_used=*/false
                    );
                }
                return false;
            }
            if constexpr (!UseExpectedPrevious) {
                PA_GM volatile int64_t *last_writer =
                    &map.shared_outputs[
                         static_cast<uint32_t>(
                             output_ref.producer_task_id
                         )
                     ].last_writer[
                         static_cast<uint32_t>(
                             output_ref.output_slot
                         )
                     ].value;
                previous =
                    TraceConfiguredAtomicLoad<
                        Ops, ObserveAtomics
                    >(
                        stats == nullptr
                            ? nullptr : &stats->trace,
                        stats == nullptr
                            ? nullptr : &stats->result,
                        task_id,
                        AtomicSite::SharedMetadataLastWriterLoad,
                        last_writer
                    );
            }
            if (previous < output_ref.producer_task_id ||
                previous >= task_id) {
                return false;
            }
        }
        history.entries[index].symbol_key = symbol_key;
        history.entries[index].previous_writer =
            static_cast<int32_t>(previous);
    }

    history.magic = kSharedWriterHistoryMagic;
    history.writer_task = task_id;
    history.count = symbol_count;
    history.reserved = 0;
    const uint64_t history_bytes =
        offsetof(SharedWriterHistoryCell, entries) +
        static_cast<uint64_t>(symbol_count) *
            sizeof(SharedWriterHistoryRecord);
#if PA_BUILD_TRACE_FREE
    (void)TraceConfiguredDcciFlush<Ops, ObserveAtomics>(
        stats == nullptr ? nullptr : &stats->trace,
        task_id, -1, DcciSite::SharedWriterHistoryFlush,
        &history, history_bytes
    );
#else
    if (deferred_trace != nullptr && stats != nullptr) {
        deferred_trace->history_dcci_lines =
            CaptureDcciFlush<Ops>(
                stats->trace, &history, history_bytes,
                deferred_trace->history_dcci_begin,
                deferred_trace->history_dcci_end
            );
    } else {
        (void)TraceConfiguredDcciFlush<Ops, ObserveAtomics>(
            stats == nullptr ? nullptr : &stats->trace,
            task_id, -1, DcciSite::SharedWriterHistoryFlush,
            &history, history_bytes
        );
    }
#endif
    Ops::StoreBarrier();

    for (uint32_t index = 0; index < symbol_count; ++index) {
        uint32_t symbol_key = 0;
        int64_t previous = -1;
        uint32_t producer = 0;
        uint32_t slot = 0;
        if constexpr (UsePaUpShape) {
            producer =
                static_cast<uint32_t>(expected_producer);
            slot = 2U - index;
            previous =
                static_cast<int64_t>(expected_previous);
        } else if constexpr (UseExpectedPrevious) {
            // 正式 PA 已在 flush 前验证并保留 owner-local key/previous。
            // clean-out 后不再从刚发布的 GM history 回读同一份记录；
            // history 仍完整写回，供跨核 reader 沿 writer 链读取。
            symbol_key = symbol_keys[index];
            previous = static_cast<int64_t>(expected_previous);
        } else {
            PA_GM const SharedWriterHistoryRecord &record =
                history.entries[index];
            symbol_key = record.symbol_key;
            previous = static_cast<int64_t>(
                record.previous_writer
            );
        }
        if constexpr (!UsePaUpShape) {
            const FdwicOutputRef output_ref =
                SharedSymbolHistoryReference(symbol_key);
            if (!IsPlainSharedOutputRef(output_ref)) {
                return false;
            }
            producer = static_cast<uint32_t>(
                output_ref.producer_task_id
            );
            slot = static_cast<uint32_t>(
                output_ref.output_slot
            );
        }
        PA_GM volatile int64_t *last_writer =
            &map.shared_outputs[producer]
                 .last_writer[slot].value;
#if PA_BUILD_TRACE_FREE
        if (TraceConfiguredAtomicCompareExchange<
                Ops, ObserveAtomics
            >(
                stats == nullptr ? nullptr : &stats->trace,
                stats == nullptr ? nullptr : &stats->result,
                task_id,
                AtomicSite::SharedMetadataLastWriterCommit,
                last_writer,
                previous,
                static_cast<int64_t>(task_id)
            ) != previous) {
            return false;
        }
#else
        int64_t observed = INT64_MIN;
        if (deferred_trace != nullptr && stats != nullptr &&
            deferred_trace->writer_cas_count < 3) {
            const uint32_t capture_index =
                deferred_trace->writer_cas_count++;
            observed = CaptureAtomicCompareExchange<Ops>(
                stats->trace, last_writer, previous,
                static_cast<int64_t>(task_id),
                deferred_trace->writer_cas_begin[capture_index],
                deferred_trace->writer_cas_end[capture_index]
            );
        } else {
            observed =
                TraceConfiguredAtomicCompareExchange<
                    Ops, ObserveAtomics
                >(
                    stats == nullptr ? nullptr : &stats->trace,
                    stats == nullptr ? nullptr : &stats->result,
                    task_id,
                    AtomicSite::SharedMetadataLastWriterCommit,
                    last_writer,
                    previous,
                    static_cast<int64_t>(task_id)
                );
        }
        if (observed != previous) {
            return false;
        }
#endif
    }
    return true;
}

// 该公共原语只处理写集合本身：读取旧 writer、发布当前 writer，并在
// 全部 symbol/ordinary 元数据完成后放行同 task loser。它不收集纯 INPUT，
// 不做 Materialize/Build，也不发布 completion；后两者必须继续使用
// task.flag。当前阶段锁定 A->B->慢 C->D->E 的慢 reader：
// - symbol 以 last_writer 为快取、task-indexed immutable history 为慢路；
// - 同一 symbol writer 必须按 task id 发布，乱序/部分 CAS 失败均终止整轮；
// - ordinary ring 只允许有序单追加且不回收，容量耗尽时 terminal fail。
// 因此本函数尚未接入 PA runtime，也不能被描述成通用多版本 backend 已闭合。
template <typename Ops>
PA_DEVICE SharedWriterIntentResult PrepareSharedWriterIntentSet(
    PA_GM SchedulerState *state, const TaskArgs &args,
    SubmitContext &context, LocalStats &stats
) {
    bool required = false;
    if (state == nullptr ||
        !InspectSharedWriterIntent(args, required)) {
        if (state != nullptr) {
            SetFatal<Ops>(state, stats, context.task_id);
        }
        return SharedWriterIntentResult::Failed;
    }
    if (!required) {
        return SharedWriterIntentResult::NotRequired;
    }
    const int32_t task_id = context.task_id;
    if (!context.won ||
        task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks) ||
        context.fanin_count < 0 ||
        context.fanin_count > static_cast<int32_t>(kMaxFanin) ||
        !ValidateSharedWriterIntentSet(args, task_id) ||
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: test-only - generic writer-intent helper 不进入当前 prepared ordered Submit
        Ops::Load(&state->fatal.value) != 0) {
        SetFatal<Ops>(state, stats, task_id);
        return SharedWriterIntentResult::Failed;
    }

    int32_t intent_fanin[kMaxFanin] = {};
    uint32_t intent_fanin_count = 0;
    // Writer intent 可以接在调用方已经完成的只读 fanin 解析之后。先把
    // 既有边复制进本地去重集合，全部 writer metadata 成功后再一次性
    // 回写 context；这样 PA 迁移无需保留另一套“先 Collect 再 Commit”
    // 专用协议，失败路径也不会留下半更新的 context。
    for (int32_t edge = 0; edge < context.fanin_count; ++edge) {
        const int32_t producer =
            context.fanin[static_cast<uint32_t>(edge)];
        if (producer < 0 || producer >= task_id ||
            !AddSharedWriterIntentFanin(
                intent_fanin, intent_fanin_count, producer
            )) {
            SetFatal<Ops>(state, stats, task_id);
            return SharedWriterIntentResult::Failed;
        }
    }
    if (!CommitSymbolSharedWriterIntentSet<Ops>(
            state->shared_map, args, task_id,
            intent_fanin, intent_fanin_count, stats,
            &state->fatal.value
        )) {
        SetFatal<Ops>(state, stats, task_id);
        return SharedWriterIntentResult::Failed;
    }
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const TensorArgType tag =
            TaskTag(args, static_cast<uint32_t>(index));
        if (!IsSharedWriterIntentTag(tag)) {
            continue;
        }
        const TaskTensorRef &reference = args.tensors[index];
        bool committed = false;
        if (reference.kind == TensorRefKind::SharedOutputRef) {
            // 全部 symbol history 与 latest CAS 已由上面的 batch 完成；
            // 这里仅保留 ordinary 参数的原顺序提交。
            continue;
        } else if (reference.kind == TensorRefKind::GmTensor) {
            committed = CommitOrdinarySharedWriterIntent<Ops>(
                state->shared_map, *reference.pointer.gm_tensor,
                task_id, static_cast<int32_t>(state->heap_window),
                intent_fanin, intent_fanin_count, stats
            );
        } else if (reference.kind == TensorRefKind::LocalTensor) {
            committed = CommitOrdinarySharedWriterIntent<Ops>(
                state->shared_map, *reference.pointer.local_tensor,
                task_id, static_cast<int32_t>(state->heap_window),
                intent_fanin, intent_fanin_count, stats
            );
        }
        if (!committed) {
            SetFatal<Ops>(state, stats, task_id);
            return SharedWriterIntentResult::Failed;
        }
    }
    for (uint32_t edge = 0; edge < intent_fanin_count; ++edge) {
        context.fanin[edge] = intent_fanin[edge];
    }
    context.fanin_count =
        static_cast<int32_t>(intent_fanin_count);
    if (!PublishSharedWriterReady<Ops>(state, task_id)) {
        SetFatal<Ops>(state, stats, task_id);
        return SharedWriterIntentResult::Failed;
    }
    return SharedWriterIntentResult::Published;
}

// 默认路径在本地执行状态建立后提交 INOUT writer；PA non-final UP 的
// intent 路径允许在 fanin/registration 已验证、winner Build 前提交，以
// 便 loser 构造下一组。FetchMax 返回旧 writer；默认实例要求精确等于 descriptor producer，
// 显式 ChainedWriter 实例按原 producer identity 选择链式 symbol，并要求
// 其旧值精确等于调用方给出的前一 writer；其他 fresh symbol 仍匹配各自
// producer。
// 异常旧值即使被 FetchMax 推进也不回滚：该 RMW 已经线性化，多 symbol
// 提交不是事务，伪造负向 RMW 会抹掉故障现场。调用者随后广播 fatal，
// 整个调度不再继续消费该状态。
template <typename Ops, bool ChainedWriter = false>
PA_DEVICE bool CommitSharedFaninWriters(
    PA_GM SharedTensorMapSidecar &map, const TaskArgs &args,
    int32_t task_id, LocalStats &stats,
    int32_t chained_producer_task_id = -1,
    int32_t expected_shared_writer = -1
) {
    if (args.tensor_count < 0 ||
        args.tensor_count > static_cast<int32_t>(kMaxTaskTensors)) {
        return false;
    }
    if constexpr (ChainedWriter) {
        if (chained_producer_task_id < 0 ||
            chained_producer_task_id >= expected_shared_writer ||
            expected_shared_writer >= task_id) {
            return false;
        }
        // 先验证 selector 确实命中至少一个合法 shared 写引用，再执行任何
        // FetchMax。调用参数错误不属于并发失败，不能留下半次 writer 推进。
        bool matched_chain_writer = false;
        for (int32_t index = 0; index < args.tensor_count; ++index) {
            const TensorArgType tag =
                TaskTag(args, static_cast<uint32_t>(index));
            if (tag != TensorArgType::Inout &&
                tag != TensorArgType::OutputExisting) {
                continue;
            }
            const TaskTensorRef &reference = args.tensors[index];
            if (reference.kind != TensorRefKind::SharedOutputRef) {
                continue;
            }
            const FdwicOutputRef output_ref =
                SharedOutputReference(reference);
            if (!IsPlainSharedOutputRef(output_ref) ||
                output_ref.producer_task_id < 0 ||
                output_ref.producer_task_id >= task_id) {
                return false;
            }
            matched_chain_writer |=
                output_ref.producer_task_id ==
                    chained_producer_task_id;
        }
        if (!matched_chain_writer) {
            return false;
        }
    }
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const TensorArgType tag =
            TaskTag(args, static_cast<uint32_t>(index));
        if (tag != TensorArgType::Inout &&
            tag != TensorArgType::OutputExisting) {
            continue;
        }
        const TaskTensorRef &reference = args.tensors[index];
        if (reference.kind != TensorRefKind::SharedOutputRef) {
            continue;
        }
        const FdwicOutputRef output_ref =
            SharedOutputReference(reference);
        if (!IsPlainSharedOutputRef(output_ref) ||
            output_ref.producer_task_id < 0 ||
            output_ref.producer_task_id >= task_id) {
            return false;
        }
        const bool chained_ref =
            ChainedWriter &&
            output_ref.producer_task_id == chained_producer_task_id;
        const int32_t expected_writer =
            chained_ref
                ? expected_shared_writer
                : output_ref.producer_task_id;
        if (expected_writer < output_ref.producer_task_id ||
            expected_writer >= task_id) {
            return false;
        }
        uint64_t retries = 0;
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: test-only - 旧 PA writer-intent 路径只由隔离测试覆盖
        const int64_t observed = Ops::FetchMax(
            &map.shared_outputs[
                 static_cast<uint32_t>(output_ref.producer_task_id)
             ].last_writer[output_ref.output_slot].value,
            static_cast<int64_t>(task_id), retries
        );
        stats.result.cas_retries += retries;
        if (observed != expected_writer) {
            return false;
        }
        ++stats.result.shared_symbol_inout_commits;
    }
    return true;
}

template <typename Ops, bool ChainedWriter = false>
PA_DEVICE bool ValidatePaSharedWriterIntentShape(
    PA_GM SchedulerState *state, const TaskArgs &args,
    const SubmitContext &context, LocalStats &stats,
    int32_t chained_producer_task_id = -1
) {
    const int32_t task_id = context.task_id;
    if (state == nullptr || !context.won || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks) ||
        args.has_error ||
        args.tensor_count < 0 ||
        args.tensor_count > static_cast<int32_t>(kMaxTaskTensors) ||
        args.scalar_count < 0 ||
        args.scalar_count > static_cast<int32_t>(kMaxTaskScalars)) {
        if (state != nullptr) {
            SetFatal<Ops>(state, stats, task_id);
        }
        return false;
    }

    const int32_t accumulator_producer =
        ChainedWriter
            ? chained_producer_task_id
            : task_id - 4;
    bool accumulator_slots[3] = {false, false, false};
    uint32_t shared_writer_refs = 0;
    uint32_t manual_dep_writer_refs = 0;
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const TensorArgType tag =
            TaskTag(args, static_cast<uint32_t>(index));
        if (tag != TensorArgType::Inout &&
            tag != TensorArgType::OutputExisting) {
            continue;
        }
        const TaskTensorRef &reference = args.tensors[index];
        if (reference.kind == TensorRefKind::SharedOutputRef) {
            const FdwicOutputRef output_ref =
                SharedOutputReference(reference);
            if (!IsPlainSharedOutputRef(output_ref) ||
                accumulator_producer < 0 ||
                output_ref.producer_task_id != accumulator_producer ||
                output_ref.output_slot < 0 ||
                output_ref.output_slot >= 3 ||
                accumulator_slots[
                    static_cast<uint32_t>(output_ref.output_slot)
                ]) {
                SetFatal<Ops>(state, stats, task_id);
                return false;
            }
            accumulator_slots[
                static_cast<uint32_t>(output_ref.output_slot)
            ] = true;
            ++shared_writer_refs;
            continue;
        }
        // PA UP 的真实参数还包含一个非 symbol、manual_dep 的 output
        // view。这里不凭地址猜测具体 view 身份，但要求这类 writer 恰好
        // 一条；任何普通 region writer 或重复 manual-dependency writer
        // 都不能借这个 PA 专用快路越过登记。
        bool manual_dep = false;
        if (reference.kind == TensorRefKind::GmTensor &&
            reference.pointer.gm_tensor != nullptr) {
            manual_dep = reference.pointer.gm_tensor->manual_dep;
        } else if (reference.kind == TensorRefKind::LocalTensor &&
                   reference.pointer.local_tensor != nullptr) {
            manual_dep = reference.pointer.local_tensor->manual_dep;
        }
        if (!manual_dep) {
            SetFatal<Ops>(state, stats, task_id);
            return false;
        }
        ++manual_dep_writer_refs;
    }
    // 这是 PA UP 专用快路，不是 ordinary-region writer 的通用替代品。
    // 三个 shared writer 必须正好对应 output/sum/max accumulator；缺失或
    // 多出任意一个都拒绝发布门，避免后继在 writer 状态不完整时前进。
    if (shared_writer_refs != 3 ||
        manual_dep_writer_refs != 1 ||
        !accumulator_slots[0] ||
        !accumulator_slots[1] ||
        !accumulator_slots[2]) {
        SetFatal<Ops>(state, stats, task_id);
        return false;
    }
    return true;
}

// 调用者已经完成只读 fanin 解析和 ordinary registration 校验后，登记
// 三个 accumulator writer，再发布本 task 的 insert-completion。它位于
// winner Build 之前；后续 Finish 必须复用 context.fanin，并在 Build 后跳过第二次
// Commit。默认实例处理首组，ChainedWriter 处理中间组。
template <typename Ops, bool ChainedWriter = false>
PA_DEVICE bool CommitPaSharedWriterIntentAfterFanin(
    PA_GM SchedulerState *state, const TaskArgs &args,
    SubmitContext &context, LocalStats &stats,
    int32_t chained_producer_task_id = -1,
    int32_t expected_shared_writer = -1
) {
    const int32_t task_id = context.task_id;
    if (!ValidatePaSharedWriterIntentShape<Ops, ChainedWriter>(
            state, args, context, stats,
            chained_producer_task_id
        )) {
        return false;
    }
    if (!CommitSharedFaninWriters<Ops, ChainedWriter>(
            state->shared_map, args, task_id, stats,
            chained_producer_task_id, expected_shared_writer
        ) ||
        !PublishSharedWriterReady<Ops>(state, task_id)) {
        SetFatal<Ops>(state, stats, task_id);
        return false;
    }
    return true;
}

// 隔离测试和无独立 Finish 阶段的调用点可一次完成 Collect + Commit +
// gate。真实 shared Finish 先自行 Collect/计入依赖签名，完成 registration
// 校验后只调用 CommitPaSharedWriterIntentAfterFanin，避免重复读 writer。
template <typename Ops, bool ChainedWriter = false>
PA_DEVICE bool PreparePaSharedWriterIntent(
    PA_GM SchedulerState *state, const TaskArgs &args,
    SubmitContext &context, LocalStats &stats,
    int32_t chained_producer_task_id = -1,
    int32_t expected_shared_writer = -1
) {
    const int32_t task_id = context.task_id;
    if (!ValidatePaSharedWriterIntentShape<Ops, ChainedWriter>(
            state, args, context, stats,
            chained_producer_task_id
        )) {
        return false;
    }

    bool protocol_ok = false;
    uint32_t ordinary_lookup_count = 0;
    context.fanin_count =
        static_cast<int32_t>(CollectSharedFanin<Ops, ChainedWriter>(
            state->shared_map, args, task_id,
            static_cast<int32_t>(state->heap_window), stats,
            context.fanin, protocol_ok, ordinary_lookup_count,
            &state->fatal.value, chained_producer_task_id,
            expected_shared_writer
        ));
    if (!protocol_ok || ordinary_lookup_count != 0) {
        SetFatal<Ops>(state, stats, task_id);
        return false;
    }
    return CommitPaSharedWriterIntentAfterFanin<Ops, ChainedWriter>(
            state, args, context, stats,
            chained_producer_task_id, expected_shared_writer
        );
}

// fresh descriptor 的内容写入每 task 独占的 shared-output cell，并通过
// FlushRegion 让 descriptor 与 writer 起点先于 published 可见。published
// 只表示后继可以读取 descriptor，不表示 producer 已 Build 或执行完成；
// kernel completion 仍由独立 completion flag 表达。
template <typename Ops>
PA_DEVICE_NOINLINE void RollbackSharedTaskOutputs(
    PA_GM SharedOutputCell &cell, uint32_t output_count,
    int32_t task_id = -1, LocalStats *stats = nullptr
) {
    // 此入口只处理唯一 producer cell 出现非法竞争后的冷失败路径。先撤销发布位，
    // 使任何非法越界 reader 都不能继续消费，再恢复 writer 与 descriptor
    // 的未发布状态；正常 Submit 不执行这些额外 atomic/DCCI。
    for (uint32_t output = 0; output < output_count; ++output) {
        (void)TraceOptionalAtomicExchange<Ops>(
            stats == nullptr ? nullptr : &stats->trace,
            stats == nullptr ? nullptr : &stats->result,
            task_id, AtomicSite::SharedOutputRollbackExchange,
            &cell.published[output].value, static_cast<int64_t>(-1),
            /*result_used=*/false
        );
    }
    for (uint32_t output = 0; output < output_count; ++output) {
        (void)TraceOptionalAtomicExchange<Ops>(
            stats == nullptr ? nullptr : &stats->trace,
            stats == nullptr ? nullptr : &stats->result,
            task_id, AtomicSite::SharedOutputRollbackExchange,
            &cell.last_writer[output].value, static_cast<int64_t>(-1),
            /*result_used=*/false
        );
    }
    for (uint32_t output = 0; output < output_count; ++output) {
        PA_GM volatile uint8_t *descriptor =
            reinterpret_cast<PA_GM volatile uint8_t *>(
                &cell.tensors[output]
            );
        for (uint32_t byte = 0; byte < sizeof(TensorDesc); ++byte) {
            descriptor[byte] = 0;
        }
    }
    if (output_count != 0) {
        (void)TraceConfiguredDcciFlush<Ops, true>(
            stats == nullptr ? nullptr : &stats->trace,
            task_id, -1, DcciSite::SharedOutputRollbackFlush,
            &cell.tensors[0],
            static_cast<uint64_t>(output_count) * sizeof(TensorDesc)
        );
    }
}

// 唯一 output publisher 在写 descriptor 前预留全部 writer 控制字。
// 默认 Publish 仍在内部调用；cross-core 原位构造实例则先显式预留，保证
// descriptor 尚未覆盖时就能拒绝旧轮残留或非法重复 publisher。
template <typename Ops, bool ObserveAtomics = false>
PA_DEVICE bool ReserveSharedTaskOutputWriters(
    PA_GM SharedTensorMapSidecar &map, uint32_t task_id,
    uint32_t output_count, LocalStats *stats = nullptr
) {
    if (task_id >= kMaxTasks ||
        output_count > kSharedOutputMaxPerTask) {
        return false;
    }
    PA_GM SharedOutputCell &cell = map.shared_outputs[task_id];
    // task-cell 唯一 winner 使预检到写入之间不存在合法竞争。仍用
    // FetchMax 预留全部 writer 控制字，并在异常旧值时撤回本次已预留项。
    for (uint32_t output = 0; output < output_count; ++output) {
        uint64_t retries = 0;
        const int64_t observed =
            TraceConfiguredAtomicFetchMax<Ops, ObserveAtomics>(
            stats == nullptr ? nullptr : &stats->trace,
            stats == nullptr ? nullptr : &stats->result,
            static_cast<int32_t>(task_id),
            AtomicSite::SharedOutputWriterReserve,
            &cell.last_writer[output].value,
            static_cast<int64_t>(task_id), retries
        );
        if (observed != -1) {
            // atomicMax 在 observed<task_id 时已经改写当前 slot；无论旧值
            // 大小都显式恢复，前面已成功预留的 slot 则回到 -1。
            (void)TraceConfiguredAtomicExchange<
                Ops, ObserveAtomics
            >(
                stats == nullptr ? nullptr : &stats->trace,
                stats == nullptr ? nullptr : &stats->result,
                static_cast<int32_t>(task_id),
                AtomicSite::SharedOutputRollbackExchange,
                &cell.last_writer[output].value, observed,
                /*result_used=*/false
            );
            for (uint32_t previous = 0; previous < output; ++previous) {
                (void)TraceConfiguredAtomicExchange<
                    Ops, ObserveAtomics
                >(
                    stats == nullptr ? nullptr : &stats->trace,
                    stats == nullptr ? nullptr : &stats->result,
                    static_cast<int32_t>(task_id),
                    AtomicSite::SharedOutputRollbackExchange,
                    &cell.last_writer[previous].value,
                    static_cast<int64_t>(-1),
                    /*result_used=*/false
                );
            }
            return false;
        }
    }
    return true;
}

// 可选时间戳 out-param 只在正式 Materialize 泳道路径传入；单元测试与
// 其它 helper 继续走默认空指针，不强制携带 LocalStats。预留后原位构造
// 实例在编译期跳过 writer reserve 与 descriptor copy，发布/DCCI 不变。
template <typename Ops, bool ObserveAtomics = false,
          bool WritersAlreadyReserved = false,
          bool DescriptorsAlreadyInCell = false>
PA_DEVICE bool PublishSharedTaskOutputs(
    PA_GM SharedTensorMapSidecar &map, const SubmitContext &context,
    uint32_t task_id, LocalStats *stats = nullptr,
    uint64_t *copy_begin = nullptr, uint64_t *copy_end = nullptr,
    uint64_t *flush_begin = nullptr, uint64_t *flush_end = nullptr
) {
    if (task_id >= kMaxTasks || context.shared_result.TaskId() != static_cast<int32_t>(task_id) ||
        context.shared_result.Size() != context.result.count ||
        context.result.count > kSharedOutputMaxPerTask) {
        return false;
    }
    PA_GM SharedOutputCell &cell = map.shared_outputs[task_id];
    // A5 Scalar 间没有 cache coherence，published/last_writer 即使独占
    // cache line，也不能用普通 GM load 判断跨轮初始化值。返回型 FetchMax
    // 仍是预留的权威线性化边界；直写实例只把该边界提前到 descriptor 写入前。
    for (uint32_t output = 0; output < context.result.count; ++output) {
        PA_GM TensorDesc *source = context.result.tensors[output];
        if (source == nullptr) {
            return false;
        }
        if constexpr (DescriptorsAlreadyInCell) {
            if (source != &cell.tensors[output]) {
                return false;
            }
        }
    }
    if constexpr (!WritersAlreadyReserved) {
        if (!ReserveSharedTaskOutputWriters<Ops, ObserveAtomics>(
                map, task_id, context.result.count, stats
            )) {
            return false;
        }
    }
    // 把原先“每 slot copy 后立刻 flush”拆成两段整批动作，便于泳道单独
    // 展示 copy 与 flush；语义不变：全部 desc 写完后再统一 flush，再
    // barrier + published。零输出 task 也保留零时长边界，保证每个
    // winner 的 raw 子层数量固定。
#if !PA_BUILD_TRACE_FREE
    if (copy_begin != nullptr) {
        *copy_begin = stats != nullptr
            ? TraceTimestamp<Ops>(stats->trace, stats->result)
            : 0;
    }
#else
    (void)stats;
    if (copy_begin != nullptr) {
        *copy_begin = 0;
    }
#endif
    if constexpr (!DescriptorsAlreadyInCell) {
        for (uint32_t output = 0; output < context.result.count; ++output) {
            PA_GM TensorDesc *source = context.result.tensors[output];
            CopyGmTensor(cell.tensors[output], *source);
        }
    }
#if !PA_BUILD_TRACE_FREE
    if (copy_end != nullptr || flush_begin != nullptr) {
        const uint64_t boundary = stats != nullptr
            ? TraceTimestamp<Ops>(stats->trace, stats->result)
            : 0;
        if (copy_end != nullptr) {
            *copy_end = boundary;
        }
        if (flush_begin != nullptr) {
            *flush_begin = boundary;
        }
    }
#else
    if (copy_end != nullptr) {
        *copy_end = 0;
    }
    if (flush_begin != nullptr) {
        *flush_begin = 0;
    }
#endif
    if (context.result.count != 0) {
        const uint64_t known_begin =
            flush_begin == nullptr ? 0 : *flush_begin;
        const uint64_t dcci_end =
            TraceConfiguredDcciFlush<Ops, ObserveAtomics>(
            stats == nullptr ? nullptr : &stats->trace,
            static_cast<int32_t>(task_id), -1,
            DcciSite::SharedOutputDescriptorFlush,
            &cell.tensors[0],
            static_cast<uint64_t>(context.result.count) *
                sizeof(TensorDesc),
            nullptr, known_begin
        );
        if (flush_end != nullptr) {
            *flush_end = dcci_end;
        }
    }
#if !PA_BUILD_TRACE_FREE
    if (flush_end != nullptr && context.result.count == 0) {
        *flush_end = stats != nullptr
            ? TraceTimestamp<Ops>(stats->trace, stats->result)
            : 0;
    }
#else
    if (flush_end != nullptr) {
        *flush_end = 0;
    }
#endif
    Ops::StoreBarrier();
    for (uint32_t output = 0; output < context.result.count; ++output) {
        if (TraceConfiguredAtomicExchange<Ops, ObserveAtomics>(
                stats == nullptr ? nullptr : &stats->trace,
                stats == nullptr ? nullptr : &stats->result,
                static_cast<int32_t>(task_id),
                AtomicSite::SharedOutputPublishedExchange,
                &cell.published[output].value,
                static_cast<int64_t>(task_id),
                /*result_used=*/true
            ) != -1) {
            RollbackSharedTaskOutputs<Ops>(
                cell, context.result.count,
                static_cast<int32_t>(task_id), stats
            );
            return false;
        }
    }
    return true;
}

template <typename Ops>
PA_DEVICE bool PublishSharedWinnerAfterBuild(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker,
    const TaskArgs &args, const SubmitContext &context,
    uint32_t task_id, TaskKind kind, LocalStats &stats,
    bool writers_prepared = false,
    bool chained_writer = false,
    int32_t chained_producer_task_id = -1,
    int32_t expected_shared_writer = -1
) {
    bool writers_committed = writers_prepared;
    if (!writers_prepared) {
        writers_committed = chained_writer
            ? CommitSharedFaninWriters<Ops, true>(
                  state->shared_map, args,
                  static_cast<int32_t>(task_id), stats,
                  chained_producer_task_id,
                  expected_shared_writer
              )
            : CommitSharedFaninWriters<Ops>(
                  state->shared_map, args,
                  static_cast<int32_t>(task_id), stats
              );
    }
    const bool outputs_published =
        writers_committed &&
        PublishSharedTaskOutputs<Ops>(
            state->shared_map, context, task_id
        );
    if (outputs_published) {
        return true;
    }
    if (kind != TaskKind::Alloc) {
        // BuildWinner 已经占用本 worker slot；封口失败后必须撤销，
        // 防止错误路径进入 FinalDrain 并执行未完成 shared 封口的任务。
        (void)DiscardBuiltTask(worker, task_id);
    }
    // Alloc 的 CompleteTask 已经发布 ready flag，无法事务性撤回。该路径
    // 只可能来自 shared invariant 损坏；fatal 使整轮结果无效，不能局部
    // 回滚后继续调度。
    SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
    return false;
}
#endif

// compete-first callback 跨 split finish 边界只传递这个固定 16B POD。
// callback closure 与内部 thunk 都在 caller 中同步结束，绝不跨 TU 保存。
struct CallbackSubmitTicket {
    uint64_t submit_begin;
    uint32_t task_id;
    int16_t function_id;
    uint8_t won;
    uint8_t reserved;
};
static_assert(sizeof(CallbackSubmitTicket) == 16, "callback ticket must remain a 16-byte POD");
static_assert(offsetof(CallbackSubmitTicket, submit_begin) == 0, "callback ticket timestamp offset mismatch");
static_assert(offsetof(CallbackSubmitTicket, task_id) == 8, "callback ticket task offset mismatch");
static_assert(offsetof(CallbackSubmitTicket, function_id) == 12, "callback ticket function offset mismatch");
static_assert(offsetof(CallbackSubmitTicket, won) == 14, "callback ticket winner offset mismatch");

#if PTO_FDWIC_SHARED_MAP && defined(PA_COMPETE_FIRST_SPLIT_FINISH)
constexpr uint64_t kSharedSplitTicketBindingPresent = 1ULL << 63U;
constexpr uint64_t kSharedSplitTicketDispatched = 1ULL << 62U;

PA_DEVICE uint64_t SharedSplitTicketBinding(
    const CallbackSubmitTicket &ticket, bool dispatched = false
) {
    return kSharedSplitTicketBindingPresent |
           (dispatched ? kSharedSplitTicketDispatched : 0U) |
           (static_cast<uint64_t>(ticket.task_id) << 8U) |
           static_cast<uint64_t>(ticket.reserved);
}

PA_DEVICE bool ArmSharedSplitTicket(
    CompeteFirstSplitRuntimeState &runtime,
    const CallbackSubmitTicket &ticket, bool dispatched = false
) {
    // caller 在跨 TU 前保存由动态 plan 推导出的唯一 task/meta 身份；
    // finish 必须逐字匹配并消费。旧 replay 继续要求逐核 0..N-1；中央
    // 发放模式允许每核取得不连续 task，但仍要求 task 位于权威计划内。
    if (runtime.reserved != 0 || ticket.task_id >= runtime.task_count ||
        (!dispatched &&
         static_cast<uint64_t>(ticket.task_id) !=
             runtime.stats.result.submits)) {
        return false;
    }
    runtime.reserved = SharedSplitTicketBinding(ticket, dispatched);
    if (dispatched) {
        runtime.task_id_sum += ticket.task_id;
    }
    return true;
}

PA_DEVICE bool RecordSharedSplitReplayTask(
    CompeteFirstSplitRuntimeState &runtime,
    uint32_t task_id
) {
    // task_id_sum 描述 caller 确实按 0..N-1 重放了完整前端序列，
    // 与只有 winner 才跨 TU 的 finish_calls 是两条不同的协议证据。
    // loser 不再构造或 Arm 跨 TU ticket，因此这里只消费已经存在的
    // task_id SSA 值，仍在 caller 中逐 task 校验。
    if (runtime.reserved != 0 ||
        static_cast<uint64_t>(task_id) !=
            runtime.stats.result.submits) {
        return false;
    }
    // 顺序检查已经证明此前成功记录的是 0..task_id-1。直接覆盖当前
    // 三角前缀，保留每个成功点的故障现场，同时省掉 block-local 旧值读取。
    const uint64_t task_id_u64 = task_id;
    runtime.task_id_sum =
        task_id_u64 * (task_id_u64 + 1U) / 2U;
    return true;
}
#endif

PA_DEVICE void BeginCallbackSubmit(PA_GM WorkerState &worker, SubmitContext &context) {
    // Claim 必须先于 TaskArgs 构造，因此这里只建立与参数无关的 Submit 上下文；
    // tensor/scalar 数量由 callback 完成后在 MaterializeTask 内写入。
    const uint32_t task_id = static_cast<uint32_t>(worker.local_index++);
    context.self = &worker;
    context.payload = &worker.payloads[task_id & kPayloadMask];
    context.task_id = static_cast<int32_t>(task_id);
    context.tensor_count = 0;
    context.scalar_count = 0;
    context.result.task_id = task_id;
    context.result.count = 0;
#if PTO_FDWIC_SHARED_MAP
    context.shared_result.Reset(static_cast<int32_t>(task_id));
#endif
    context.register_mask = 0;
    context.output_bytes = 0;
    context.fanin_count = 0;
    context.kernel_id = -1;
    context.won = false;
    context.joint = false;
    context.joint_init = false;
    context.joint_block = -1;
    context.joint_slot = -1;
    context.joint_count = 0;
}

#if PTO_FDWIC_SHARED_MAP
PA_DEVICE void BeginSharedCallbackSubmit(
    PA_GM WorkerState &worker, SubmitContext &context
) {
    // shared replay 的 96 个 actor 都要先取得同一个逻辑 task_id，但只有
    // Claim owner 会进入 Materialize/Build。loser 在 Claim 后只需要
    // task 身份与稳定 output symbol，因此这里不再为每个 replay actor
    // 清零整份 408-byte SubmitContext 的 winner-only 字段。
    const uint32_t task_id =
        static_cast<uint32_t>(worker.local_index++);
    context.task_id = static_cast<int32_t>(task_id);
    context.shared_result.Reset(static_cast<int32_t>(task_id));
}

PA_DEVICE void PrepareSharedWinnerContext(
    PA_GM WorkerState &worker, uint32_t task_id,
    SubmitContext &context
) {
    // 这些字段都只会被 shared winner 的 Materialize/Fanin/Build 消费。
    // tensor/scalar/register/output_bytes 由 MaterializeTask 在读取前覆盖。
    // cross-core payload adapter 会显式拒绝 Joint，因此这里必须把
    // winner 复用的 SubmitContext 恢复成普通单 lane task；不能依赖
    // CPU 栈或 CCEC block-local 旧值恰好为零。
    context.self = &worker;
    context.payload =
        &worker.payloads[task_id & kPayloadMask];
    context.result.task_id = task_id;
    context.result.count = 0;
    context.fanin_count = 0;
    context.joint = false;
    context.joint_init = false;
    context.joint_block = -1;
    context.joint_slot = -1;
    context.joint_count = 0;
}
#endif

#if defined(__CCE_AICORE__) || defined(__NPU_ARCH__)
#define PA_CALLBACK_LAMBDA_DEVICE __aicore__
#else
#define PA_CALLBACK_LAMBDA_DEVICE
#endif

template <TaskKind Kind>
PA_DEVICE bool BuildCallbackSubmitArgs(
    PaOrchestrationState &orch, TaskArgs &args, uint32_t batch, LocalStats &stats
) {
    CallbackSubmitArgsBuilder builder(args, Kind);
    // 外层 callback 和所有参数 thunk 都只在这一调用点同步执行。调用者决定
    // 是否构参：private 仍全员 eager；shared 的五类 task 都只由 Claim
    // owner 进入这里。
    auto callback = [&](CallbackSubmitArgsBuilder &out) PA_CALLBACK_LAMBDA_DEVICE {
        out.Begin();
        if constexpr (Kind == TaskKind::Alloc) {
            out.AddOutput([&]() PA_CALLBACK_LAMBDA_DEVICE -> const TensorCreateInfo & {
                return orch.tile_create_info;
            });
            out.AddOutput([&]() PA_CALLBACK_LAMBDA_DEVICE -> const TensorCreateInfo & {
                return orch.scalar_create_info;
            });
            out.AddOutput([&]() PA_CALLBACK_LAMBDA_DEVICE -> const TensorCreateInfo & {
                return orch.scalar_create_info;
            });
        } else if constexpr (Kind == TaskKind::Qk) {
            out.AddLocalInput([&]() PA_CALLBACK_LAMBDA_DEVICE -> const TensorDesc & {
                MakeCallbackQueryView(orch, batch);
                out.RecordView();
                return orch.query_view;
            });
            out.AddLocalInput([&]() PA_CALLBACK_LAMBDA_DEVICE -> const TensorDesc & {
                return orch.key_cache;
            });
            out.AddLocalInput([&]() PA_CALLBACK_LAMBDA_DEVICE -> const TensorDesc & {
                return orch.block_table;
            });
            out.AddOutput([&]() PA_CALLBACK_LAMBDA_DEVICE -> const TensorCreateInfo & {
                const uint32_t score_shape[kMaxTensorDims] = {
                    kPaHeads,
                    static_cast<uint32_t>(orch.current_nblocks * kPaBlockSize),
                    0, 0, 0
                };
                InitCreateInfo(orch.qk_create_info, score_shape, 2, DataType::Float32);
                out.RecordDynamicCreateInfo();
                return orch.qk_create_info;
            });
            out.AddScalar([&]() PA_CALLBACK_LAMBDA_DEVICE -> uint64_t {
                return orch.current_nblocks;
            });
            out.AddScalar([&]() PA_CALLBACK_LAMBDA_DEVICE -> uint64_t {
                return static_cast<uint64_t>(orch.current_batch) * kPaMaxBlocksPerRequest +
                       orch.current_block_offset;
            });
        } else if constexpr (Kind == TaskKind::Sf) {
            out.AddOutputHandleInput([&]() PA_CALLBACK_LAMBDA_DEVICE -> PaOutputHandle {
                return orch.qk_scores;
            });
            out.AddOutput([&]() PA_CALLBACK_LAMBDA_DEVICE -> const TensorCreateInfo & {
                const uint32_t probability_shape[kMaxTensorDims] = {
                    kPaHeads,
                    static_cast<uint32_t>(orch.current_nblocks * kPaBlockSize),
                    0, 0, 0
                };
                InitCreateInfo(orch.sf_create_info, probability_shape, 2, DataType::Bfloat16);
                out.RecordDynamicCreateInfo();
                return orch.sf_create_info;
            });
            out.AddOutput([&]() PA_CALLBACK_LAMBDA_DEVICE -> const TensorCreateInfo & {
                return orch.scalar_create_info;
            });
            out.AddOutput([&]() PA_CALLBACK_LAMBDA_DEVICE -> const TensorCreateInfo & {
                return orch.scalar_create_info;
            });
            out.AddScalar([&]() PA_CALLBACK_LAMBDA_DEVICE -> uint64_t { return orch.scale_bits; });
            out.AddScalar([&]() PA_CALLBACK_LAMBDA_DEVICE -> uint64_t {
                return orch.current_nblocks;
            });
            out.AddScalar([&]() PA_CALLBACK_LAMBDA_DEVICE -> uint64_t {
                return orch.current_valid_len;
            });
        } else if constexpr (Kind == TaskKind::Pv) {
            out.AddOutputHandleInput([&]() PA_CALLBACK_LAMBDA_DEVICE -> PaOutputHandle {
                return orch.sf_probs;
            });
            out.AddLocalInput([&]() PA_CALLBACK_LAMBDA_DEVICE -> const TensorDesc & {
                return orch.value_cache;
            });
            out.AddLocalInput([&]() PA_CALLBACK_LAMBDA_DEVICE -> const TensorDesc & {
                return orch.block_table;
            });
            out.AddOutput([&]() PA_CALLBACK_LAMBDA_DEVICE -> const TensorCreateInfo & {
                return orch.tile_create_info;
            });
            out.AddScalar([&]() PA_CALLBACK_LAMBDA_DEVICE -> uint64_t {
                return orch.current_nblocks;
            });
            out.AddScalar([&]() PA_CALLBACK_LAMBDA_DEVICE -> uint64_t {
                return static_cast<uint64_t>(orch.current_batch) * kPaMaxBlocksPerRequest +
                       orch.current_block_offset;
            });
        } else {
            static_assert(Kind == TaskKind::Up, "unsupported PA task kind");
            out.AddOutputHandleInput([&]() PA_CALLBACK_LAMBDA_DEVICE -> PaOutputHandle {
                return orch.sf_max;
            });
            out.AddOutputHandleInput([&]() PA_CALLBACK_LAMBDA_DEVICE -> PaOutputHandle {
                return orch.sf_sum;
            });
            out.AddOutputHandleInput([&]() PA_CALLBACK_LAMBDA_DEVICE -> PaOutputHandle {
                return orch.pv_output;
            });
            out.AddOutputHandleInout([&]() PA_CALLBACK_LAMBDA_DEVICE -> PaOutputHandle {
                return orch.accumulated_max;
            });
            out.AddOutputHandleInout([&]() PA_CALLBACK_LAMBDA_DEVICE -> PaOutputHandle {
                return orch.accumulated_sum;
            });
            out.AddOutputHandleInout([&]() PA_CALLBACK_LAMBDA_DEVICE -> PaOutputHandle {
                return orch.accumulated_output;
            });
            out.AddLocalInout([&]() PA_CALLBACK_LAMBDA_DEVICE -> const TensorDesc & {
                MakeCallbackOutputView(orch, batch);
                out.RecordView();
                return orch.output_view;
            });
            out.AddScalar([&]() PA_CALLBACK_LAMBDA_DEVICE -> uint64_t {
                return orch.current_block_offset == 0 ? 1 : 0;
            });
            out.AddScalar([&]() PA_CALLBACK_LAMBDA_DEVICE -> uint64_t {
                return orch.current_block_offset + orch.current_nblocks >= orch.current_blocks ? 1 : 0;
            });
        }
    };

    callback(builder);
    if (!builder.Valid()) return false;
    const CallbackSubmitBuildCounts &counts = builder.Counts();
    stats.result.arg_resets += counts.reset_calls;
    stats.result.views_created += counts.views_created;
    stats.result.dynamic_create_infos += counts.dynamic_create_infos;
    stats.result.tensor_args_added += counts.tensor_args_added;
    stats.result.scalar_args_added += counts.scalar_args_added;
    return true;
}

#undef PA_CALLBACK_LAMBDA_DEVICE

#if PTO_FDWIC_SHARED_MAP
template <typename Ops, bool Profile>
PA_DEVICE bool CloseSharedCallbackSubmit(
    PA_GM SchedulerState *state, LocalStats &stats,
    uint32_t task_id, TaskKind kind, uint64_t submit_begin,
    bool is_last_submit
) {
    // 该接口服务于旧的逐核顺序 replay，仍要求每核 Close 的 task id
    // 连续。Execute 任务发现已经完全由 host 发布的 AIC/AIV 角色计划和
    // 中央 ticket 负责，Close 不再维护 K2 候选位图。
    if (static_cast<uint64_t>(task_id) != stats.result.submits ||
        kind >= TaskKind::Count) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    ++stats.result.submits;

    // shared 的总 task 数取决于每批 context_len。winner 从 ticket 恢复
    // 末次身份，同 TU loser 直接使用已经校验的 batch plan 结果，避免
    // 96 个 worker 为了一个计时边界额外预扫整份 context_lens。
#if PA_BUILD_PERF_CLOCK
    // 无泳道性能构建的唯一终点是 FinalDrain 结束；这里不再为末个
    // Submit 单独读取时钟。
    const uint64_t submit_end = 0;
#elif PA_BUILD_SUBMIT_PMU
    const uint64_t submit_end = is_last_submit ? Ops::Now() : 0;
#else
    // 参考前端的 rt_submit_loser 仍是一次真实轻量 Submit；保留既有父
    // 区间用于覆盖稳定符号返回和轻量收尾；loser 不等待 TensorMap，
    // 也不再写任何 winner-only child。
    const uint64_t submit_end =
        TraceTimestamp<Ops>(stats.trace, stats.result);
#endif
    WriteSharedSubmitTrace<Profile>(
        stats.trace, stats.result, task_id,
        submit_begin, submit_end
    );
    if (!is_last_submit) {
        return true;
    }
    if (stats.declared_task_count != 0) {
        SetFatal<Ops>(
            state, stats, static_cast<int32_t>(task_id)
        );
        return false;
    }
    stats.declared_task_count = task_id + 1U;
    stats.result.submit_end = submit_end;
    return true;
}

template <typename Ops, bool Profile>
PA_DEVICE bool CloseSharedCallbackSubmit(
    PA_GM SchedulerState *state, LocalStats &stats,
    const CallbackSubmitTicket &ticket,
    const SharedPaTaskMeta &shared_task_meta, TaskKind kind
) {
    // winner 的跨 TU ABI 继续传递完整 ticket；公共收尾只消费实际需要的
    // task_id/submit_begin，避免迫使同 TU loser 也物化该 POD。
    return CloseSharedCallbackSubmit<Ops, Profile>(
        state, stats, ticket.task_id, kind, ticket.submit_begin,
        shared_task_meta.is_last_submit
    );
}

template <typename Ops, bool Profile>
PA_DEVICE bool FinishSharedLoserSubmit(
    PA_GM SchedulerState *state, LocalStats &stats, uint32_t task_id,
    TaskKind kind,
    bool is_last_submit,
    uint64_t submit_begin
) {
    // 同一调用链已用当前 task_id/Kind 成功执行
    // PrepareSharedTaskOutputs；失败会在进入本 helper 前终止，期间也没有
    // shared_result 写者。loser 不再重复读取它的 TaskId/Size；完整
    // symbol 仍保留给 Submit 返回后的 orchestration 消费。
    // loser 只完成本次 Submit 的轻量收尾。TensorMap 插入、前沿等待、
    // fanin lookup 与 Build 全部只属于 Claim owner；loser 不读取任何
    // TensorMap 控制字，也不再等待 writer-ready 门。
    return CloseSharedCallbackSubmit<Ops, Profile>(
        state, stats, task_id, kind, submit_begin, is_last_submit
    );
}

template <typename Ops, bool Profile>
PA_DEVICE bool CloseSharedDispatchedSubmit(
    PA_GM SchedulerState *state, LocalStats &stats,
    uint32_t task_id, TaskKind kind, uint64_t submit_begin
) {
    // 中央 ticket 已经给每个 task 唯一 Build owner，逐核 submits 因此只是
    // 本核完成数，不再是 0..N 的 replay 前缀。task 终点仍写入按真实
    // task_id 索引的独占泳道槽；整个调度窗口由外层循环统一封口。
    if (task_id >= state->build_dispatch.task_count ||
        kind >= TaskKind::Count) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    ++stats.result.submits;
#if PA_BUILD_TRACE_FREE
    const uint64_t submit_end = 0;
#else
    const uint64_t submit_end =
        TraceTimestamp<Ops>(stats.trace, stats.result);
#endif
    WriteSharedSubmitTrace<Profile>(
        stats.trace, stats.result, task_id,
        submit_begin, submit_end
    );
    return true;
}

PA_DEVICE bool CrossCoreExecWorkerMatchesRole(
    uint32_t worker_id, CoreRole role
) {
    return
        (role == CoreRole::Aic && worker_id < kAicWorkers) ||
        (role == CoreRole::Aiv && worker_id >= kAicWorkers &&
         worker_id < kWorkers);
}

PA_DEVICE cross_core_dag::ExecEngineClass CrossCoreEngineForRole(
    CoreRole role
) {
    if (role == CoreRole::Aic) {
        return cross_core_dag::ExecEngineClass::Aic;
    }
    if (role == CoreRole::Aiv) {
        return cross_core_dag::ExecEngineClass::Aiv;
    }
    return cross_core_dag::ExecEngineClass::None;
}

enum class SharedExecTicketStatus : uint32_t {
    Acquired = 0,
    Exhausted = 1,
    Invalid = 2,
};

struct SharedExecTicketResult {
    SharedExecTicketStatus status;
    uint32_t task_ids[cross_core_dag::kExecTicketBatchSize];
    uint32_t task_count;
};

// 默认入口继续逐次校验 immutable plan header，供独立协议调用保持
// fail-closed。正式中央调度只有在 RunSchedulerImpl 已于首张 ticket 前
// 校验同一 header 后才实例化 PlanHeaderValidated=true；task-id、角色路由、
// cursor 上界和返回型 Atomic 结果仍逐批检查。每批 ordinal 区间由一次
// FetchAdd 线性化，区间内每个 task 仍分别校验 route 并绑定独立 token。
template <typename Ops, bool PlanHeaderValidated = false>
PA_DEVICE SharedExecTicketResult TakeSharedExecTicket(
    PA_GM SchedulerState *state, uint32_t worker_id,
    CoreRole role, uint32_t task_count, LocalStats &stats
) {
    if (state == nullptr || worker_id >= kWorkers ||
        task_count == 0 || task_count > kMaxTasks ||
        !CrossCoreExecWorkerMatchesRole(worker_id, role) ||
        stats.exec_dispatch_exhausted != 0) {
        return SharedExecTicketResult{
            SharedExecTicketStatus::Invalid,
            {UINT32_MAX, UINT32_MAX}, 0
        };
    }

    PA_GM volatile int64_t *cursor = nullptr;
    PA_GM const uint32_t *task_ids = nullptr;
    uint32_t role_task_count = 0;
    const cross_core_dag::ExecEngineClass engine =
        CrossCoreEngineForRole(role);
    if (engine == cross_core_dag::ExecEngineClass::Aic) {
        cursor = &state->exec_dispatch.aic_next.value;
        task_ids = &state->exec_dispatch.aic_task_ids[0];
        role_task_count = state->exec_dispatch.aic_task_count;
    } else if (engine == cross_core_dag::ExecEngineClass::Aiv) {
        cursor = &state->exec_dispatch.aiv_next.value;
        task_ids = &state->exec_dispatch.aiv_task_ids[0];
        role_task_count = state->exec_dispatch.aiv_task_count;
    } else {
        return SharedExecTicketResult{
            SharedExecTicketStatus::Invalid,
            {UINT32_MAX, UINT32_MAX}, 0
        };
    }
    if constexpr (!PlanHeaderValidated) {
        if (role_task_count > task_count ||
            state->exec_dispatch.aic_task_count > task_count ||
            state->exec_dispatch.aiv_task_count > task_count ||
            state->exec_dispatch.aic_task_count +
                    state->exec_dispatch.aiv_task_count !=
                state->build_dispatch.executable_task_count) {
            return SharedExecTicketResult{
                SharedExecTicketStatus::Invalid,
                {UINT32_MAX, UINT32_MAX}, 0
            };
        }
    }

    const int64_t observed = TraceAtomicFetchAdd<Ops>(
        stats.trace, stats.result, -1,
        AtomicSite::SharedExecDispatchTicket,
        cursor, cross_core_dag::kExecTicketBatchSize,
        /*result_used=*/true
    );
    if (observed < 0) {
        return SharedExecTicketResult{
            SharedExecTicketStatus::Invalid,
            {UINT32_MAX, UINT32_MAX}, 0
        };
    }
    if (observed >= static_cast<int64_t>(role_task_count)) {
        stats.exec_dispatch_exhausted = 1;
        return SharedExecTicketResult{
            SharedExecTicketStatus::Exhausted,
            {UINT32_MAX, UINT32_MAX}, 0
        };
    }
    const uint32_t first = static_cast<uint32_t>(observed);
    const uint32_t acquired =
        role_task_count - first < cross_core_dag::kExecTicketBatchSize
            ? role_task_count - first
            : cross_core_dag::kExecTicketBatchSize;
    SharedExecTicketResult result{
        SharedExecTicketStatus::Acquired,
        {UINT32_MAX, UINT32_MAX}, acquired
    };
    for (uint32_t index = 0; index < acquired; ++index) {
        const uint32_t task_id = task_ids[first + index];
        if (task_id >= task_count) {
            return SharedExecTicketResult{
                SharedExecTicketStatus::Invalid,
                {UINT32_MAX, UINT32_MAX}, 0
            };
        }
        result.task_ids[index] = task_id;
    }
    if (first + cross_core_dag::kExecTicketBatchSize >=
        role_task_count) {
        // 尾批次既取得最后一项，也完成本 worker 的 exhaustion 证明；
        // 不需要再向同一 cursor 发射一次纯越界 RMW。
        stats.exec_dispatch_exhausted = 1;
    }
    return result;
}

PA_DEVICE bool BindSharedExecTicketToToken(
    PA_GM cross_core_dag::ExecutionToken &token,
    uint32_t task_id, uint32_t execute_owner,
    cross_core_dag::ExecEngineClass engine
) {
    if (token.control.phase != cross_core_dag::ExecTokenPhase::Idle ||
        task_id >= kMaxTasks || execute_owner >= kWorkers ||
        !cross_core_dag::A5SingleLaneOwnerMatchesEngine(
            execute_owner, engine
        )) {
        return false;
    }
    token.control.task_id = task_id;
    token.control.build_owner = UINT32_MAX;
    token.control.execute_owner = execute_owner;
    token.control.engine_class = engine;
    token.control.payload_lines = 0;
    token.control.payload_bytes = 0;
    token.control.fanin_ready_prefix = 0;
    token.control.payload_address = 0;
    token.control.completion_vend = 0;
    token.control.function_and_reference = 0;
    token.control.shape_and_scalar_offset = 0;
    token.control.phase = cross_core_dag::ExecTokenPhase::WaitingBuilt;
    return true;
}

PA_DEVICE uint32_t CrossCoreExecOccupiedTokenCount(
    PA_GM const SchedulerState *state, uint32_t worker_id
) {
    uint32_t occupied = 0;
    for (uint32_t token_slot = 0;
         token_slot < cross_core_dag::kExecTokensPerWorker;
         ++token_slot) {
        occupied +=
            state->exec_tokens[worker_id][token_slot].control.phase ==
                    cross_core_dag::ExecTokenPhase::Idle
                ? 0U
                : 1U;
    }
    return occupied;
}

PA_DEVICE uint32_t CrossCoreExecFirstIdleTokenSlot(
    PA_GM const SchedulerState *state, uint32_t worker_id
) {
    for (uint32_t token_slot = 0;
         token_slot < cross_core_dag::kExecTokensPerWorker;
         ++token_slot) {
        if (state->exec_tokens[worker_id][token_slot].control.phase ==
            cross_core_dag::ExecTokenPhase::Idle) {
            return token_slot;
        }
    }
    return cross_core_dag::kExecTokensPerWorker;
}

PA_DEVICE bool CrossCoreExecAllTokensIdle(
    PA_GM const SchedulerState *state, uint32_t worker_id
) {
    return CrossCoreExecOccupiedTokenCount(state, worker_id) == 0;
}

// opportunistic EfDrain 只依据 owner-local 状态决定是否进入完整执行推进：
// 已占用 token 需要继续观察 BUILT/fanin/completion；尚未取得本角色的越界
// ticket 时仍可能领取新 task。两者都不成立时，本核已经永久排空，不再读取
// AIC/AIV 中央 cursor。FinalDrain 仍会进入完整收口协议。
PA_DEVICE bool CrossCoreExecHasLocalProgressWork(
    PA_GM const SchedulerState *state,
    uint32_t worker_id, CoreRole role,
    uint32_t replay_closed_exclusive,
    const LocalStats &stats
) {
    if (state == nullptr || worker_id >= kWorkers ||
        replay_closed_exclusive > kMaxTasks ||
        !CrossCoreExecWorkerMatchesRole(worker_id, role)) {
        // 非法入口必须交给完整协议发布精确 fatal，不能被快路径吞掉。
        return true;
    }
    if (!CrossCoreExecAllTokensIdle(state, worker_id)) {
        return true;
    }
    return stats.exec_dispatch_exhausted == 0;
}

template <typename Ops>
PA_DEVICE void PublishCrossCoreRuntimeFailure(
    PA_GM SchedulerState *state, LocalStats &stats,
    cross_core_dag::ExecFatalReason reason, uint32_t task_id,
    uint32_t worker_id
) {
    if (state == nullptr || worker_id >= kWorkers) {
        return;
    }
    SharedExecTraceObserver<Ops> observer{&stats};
    (void)cross_core_dag::PublishExecFatal<Ops>(
        state->exec_fatal, reason, task_id, worker_id,
        observer
    );
    SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
}

template <typename Ops>
PA_DEVICE_NOINLINE bool PublishCrossCoreExecTask(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker,
    uint32_t task_id, TaskKind kind, int32_t function_id,
    const TaskArgs &args, const SubmitContext &context,
    LocalStats &stats
) {
    const uint32_t worker_id =
        static_cast<uint32_t>(worker.core_idx);
    SharedExecTraceObserver<Ops> observer{&stats};
    cross_core_dag::PaExecRoute route{};
    if (state == nullptr || worker.core_idx < 0 ||
        worker_id >= kWorkers || task_id >= kMaxTasks ||
        kind == TaskKind::Alloc || kind == TaskKind::Count ||
        function_id != FunctionId(kind) ||
        !cross_core_dag::ResolvePaExecRoute(kind, function_id, route) ||
        !cross_core_dag::A5SingleLaneBuildOwnerEligible(
            worker_id, route.engine_class
        )) {
        PublishCrossCoreRuntimeFailure<Ops>(
            state, stats,
            cross_core_dag::ExecFatalReason::InvalidBuildInput,
            task_id, worker_id
        );
        return false;
    }
    // scheduler fatal 由领取新 Build ticket 前的调度边界统一检查。
    // 本 helper 接收已经取得的合法 task，并只对本 task 的 Build 结果负责；
    // 不在 WinnerBuild 内再次读取同一全局停止线。

    cross_core_dag::ExecPayloadSpec spec{};
    cross_core_dag::PaExecPayloadSource source{};
    if (!cross_core_dag::MakePaExecPayloadSpec(
            task_id, kind, function_id,
            /*function_address=*/0, worker, args, context, spec
        )) {
        PublishCrossCoreRuntimeFailure<Ops>(
            state, stats,
            cross_core_dag::ExecFatalReason::InvalidBuildInput,
            task_id, worker_id
        );
        return false;
    }
    if (!cross_core_dag::ResolvePaExecPayloadSourceAfterFanin<Ops, true>(
            *state, args, context, task_id, source,
            observer
        )) {
        PublishCrossCoreRuntimeFailure<Ops>(
            state, stats,
            cross_core_dag::ExecFatalReason::BuildPackFailed,
            task_id, worker_id
        );
        return false;
    }

    // helper 在接触 cell 前只检查一次 exec_fatal。source 解析期间并发
    // 出现的通用 fatal 不要求中断当前 Build；即使发布 BUILT，后续 Claim
    // 也会拒绝执行，FinalDrain 仍能按首错收敛。
    const cross_core_dag::ExecBuildResult build_result =
        cross_core_dag::BuildAndPublishExecPayload<Ops>(
            state->exec_cells[task_id], worker_id, spec, source,
            state->exec_fatal, observer
        );
    if (build_result != cross_core_dag::ExecBuildResult::Published) {
        if (build_result ==
            cross_core_dag::ExecBuildResult::CellUnavailable) {
            if (IsFatal<Ops>(
                    state, stats, static_cast<int32_t>(task_id)
                )) {
                return false;
            }
            if (cross_core_dag::ExecFatalPublished<Ops>(
                    state->exec_fatal, task_id, observer
                )) {
                SetFatal<Ops>(
                    state, stats, static_cast<int32_t>(task_id)
                );
                return false;
            }
            // CellUnavailable 自身没有发布 execution reason；fresh
            // task-indexed cell 出现占用即为真实 control 冲突。
            PublishCrossCoreRuntimeFailure<Ops>(
                state, stats,
                cross_core_dag::ExecFatalReason::ControlPublishConflict,
                task_id, worker_id
            );
        } else {
            // InvalidInput/PublishConflict 已由协议 helper 保存精确首错；
            // FatalObserved 更必须保留先前 reason。这里只镜像通用 fatal。
            SetFatal<Ops>(
                state, stats, static_cast<int32_t>(task_id)
            );
        }
        return false;
    }

    // 沿用既有 host 统计字段的业务含义：它们现在表示
    // portable payload 中实际发布的 tensor/scalar/fanin 数，
    // 不再表示 private LocalSlot 的物理拷贝。
    stats.result.slot_tensor_copies +=
        static_cast<uint32_t>(context.tensor_count);
    stats.result.slot_scalar_copies +=
        static_cast<uint32_t>(context.scalar_count);
    stats.result.fanin_edges +=
        static_cast<uint32_t>(context.fanin_count);
    return true;
}

template <typename Ops>
PA_DEVICE bool ProgressCrossCoreActiveToken(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker,
    uint32_t token_slot, DrainPlace place,
    LocalStats &stats, bool &completed
) {
    completed = false;
    const uint32_t worker_id =
        static_cast<uint32_t>(worker.core_idx);
    SharedExecTraceObserver<Ops> observer{&stats};
    if (state == nullptr || worker.core_idx < 0 ||
        worker_id >= kWorkers ||
        token_slot >= cross_core_dag::kExecTokensPerWorker) {
        return false;
    }
    PA_GM cross_core_dag::ExecutionToken &token =
        state->exec_tokens[worker_id][token_slot];
    if (token.control.phase == cross_core_dag::ExecTokenPhase::Idle) {
        return true;
    }
    if (token.control.phase ==
        cross_core_dag::ExecTokenPhase::WaitingBuilt) {
        const uint32_t waiting_task_id = token.control.task_id;
        const cross_core_dag::ExecEngineClass waiting_engine =
            token.control.engine_class;
        SharedExecDispatchRoute planned{};
        if (waiting_task_id >= state->build_dispatch.task_count ||
            token.control.build_owner != UINT32_MAX ||
            token.control.execute_owner != worker_id ||
            waiting_engine != CrossCoreEngineForRole(worker.role) ||
            !DecodeSharedExecDispatchRoute(
                state->build_dispatch, waiting_task_id, planned
            ) ||
            !planned.executable ||
            planned.engine_class != waiting_engine) {
            PublishCrossCoreRuntimeFailure<Ops>(
                state, stats,
                cross_core_dag::ExecFatalReason::InvalidTokenPayload,
                waiting_task_id, worker_id
            );
            token.control.phase =
                cross_core_dag::ExecTokenPhase::Faulted;
            return false;
        }

        PA_GM cross_core_dag::SharedExecCell &cell =
            state->exec_cells[waiting_task_id];
        const int64_t observed_raw = observer.LoadCellState(
            &cell.control.state, waiting_task_id
        );
        const cross_core_dag::DecodedExecState observed =
            cross_core_dag::DecodeExecState(observed_raw);
        if (!observed.valid) {
            PublishCrossCoreRuntimeFailure<Ops>(
                state, stats,
                cross_core_dag::ExecFatalReason::InvalidBuiltControl,
                waiting_task_id, worker_id
            );
            token.control.phase =
                cross_core_dag::ExecTokenPhase::Faulted;
            return false;
        }
        if (observed.phase == cross_core_dag::ExecPhase::Empty) {
            return true;
        }
        if (observed.phase == cross_core_dag::ExecPhase::Building) {
            if (observed.task_id == waiting_task_id &&
                observed.engine_class ==
                    cross_core_dag::ExecEngineClass::None &&
                observed.execute_owner ==
                    cross_core_dag::kExecUnboundOwner &&
                observed.payload_lines == 0 &&
                cross_core_dag::A5SingleLaneBuildOwnerEligible(
                    observed.build_owner, waiting_engine
                )) {
                return true;
            }
            PublishCrossCoreRuntimeFailure<Ops>(
                state, stats,
                cross_core_dag::ExecFatalReason::InvalidBuiltControl,
                waiting_task_id, worker_id
            );
            token.control.phase =
                cross_core_dag::ExecTokenPhase::Faulted;
            return false;
        }
        if (observed.phase != cross_core_dag::ExecPhase::Built ||
            observed.task_id != waiting_task_id ||
            observed.engine_class != waiting_engine ||
            !cross_core_dag::A5SingleLaneBuildOwnerEligible(
                observed.build_owner, waiting_engine
            )) {
            // Execute ticket 按 task 唯一发放；在本核完成 BUILT->CLAIMED
            // 之前看到 CLAIMED/DONE 不是合法 loser，而是重复消费。
            PublishCrossCoreRuntimeFailure<Ops>(
                state, stats,
                cross_core_dag::ExecFatalReason::InvalidBuiltControl,
                waiting_task_id, worker_id
            );
            token.control.phase =
                cross_core_dag::ExecTokenPhase::Faulted;
            return false;
        }

        // Claim helper 的通用入口要求 IDLE。ticket 身份已经在上面完整
        // 校验；这里仅在 owner-local control 上临时切换，不发布共享状态。
        token.control.phase = cross_core_dag::ExecTokenPhase::Idle;
        const cross_core_dag::ExecClaimResult claim =
            cross_core_dag::ClaimAndBindObservedExecPayload<Ops>(
                cell, observed_raw, waiting_task_id,
                worker_id, waiting_engine, token,
                state->exec_fatal, observer
            );
        if (claim != cross_core_dag::ExecClaimResult::Claimed ||
            !cross_core_dag::BindPaExecutionTokenDispatchAfterClaim(
                token, worker
            )) {
            PublishCrossCoreRuntimeFailure<Ops>(
                state, stats,
                cross_core_dag::ExecFatalReason::InvalidTokenPayload,
                waiting_task_id, worker_id
            );
            token.control.phase =
                cross_core_dag::ExecTokenPhase::Faulted;
            return false;
        }
    }
    // global fatal 在外层调度边界观察。已经取得的合法 token 作为一个
    // 不可拆分工作单元推进到完成；不在 fanin、kernel 和 completion 之间
    // 反复读取共享停止线。
    const uint32_t token_task_id = token.control.task_id;
    const uint32_t token_function_id =
        cross_core_dag::ExecutionTokenFunctionId(token);
    TaskKind kind = TaskKind::Count;
    cross_core_dag::PaExecRoute route{};
    if (!cross_core_dag::PaTaskKindFromExecFunction(
            token_function_id, kind
        ) ||
        !cross_core_dag::ResolvePaExecRoute(
            kind, static_cast<int32_t>(token_function_id), route
        ) ||
        token_task_id >= kMaxTasks ||
        route.engine_class != token.control.engine_class ||
        route.engine_class != CrossCoreEngineForRole(worker.role) ||
        !cross_core_dag::A5SingleLaneBuildOwnerEligible(
            token.control.build_owner, route.engine_class
        ) ||
        !cross_core_dag::A5SingleLaneExecuteOwnerEligible(
            token_task_id, token.control.build_owner,
            route.engine_class, worker_id
        ) ||
        token.control.execute_owner != worker_id) {
        PublishCrossCoreRuntimeFailure<Ops>(
            state, stats,
            cross_core_dag::ExecFatalReason::InvalidTokenPayload,
            token.control.task_id, worker_id
        );
        token.control.phase = cross_core_dag::ExecTokenPhase::Faulted;
        return false;
    }
    if (token.control.phase ==
            cross_core_dag::ExecTokenPhase::WaitingFanin) {
        cross_core_dag::ObservedPaExecReadySource<
            Ops, SharedExecTraceObserver<Ops>
        > ready{
            state, &stats.result, &observer
        };
        // 未 ready 只是一次纯观察，不推进共享状态，也不发射 kernel。
        // 先在 owner-local token 上压缩 ready 前缀；只有全部依赖 ready，
        // 准备进入 ENGINE_INFLIGHT 时才调用带 exec-fatal 门禁的 helper。
        // 这样不会让每次 opportunistic poll 都汇聚读取同一 fatal line。
        // replay 失败后的强制收敛由 production_closed 的 Progress 入口
        // 负责，不能把这里的性能快路径外推到 FinalDrain。
        if (!cross_core_dag::ExecutionTokenFaninReady(token, ready)) {
            return true;
        }
        if (!cross_core_dag::TryMarkExecutionTokenEngineInflight<Ops>(
                token, ready, state->exec_fatal, observer
            )) {
            if (token.control.phase ==
                cross_core_dag::ExecTokenPhase::Faulted) {
                SetFatal<Ops>(
                    state, stats,
                    static_cast<int32_t>(token.control.task_id)
                );
                return false;
            }
            return true;
        }

        const uint64_t kernel_begin =
            TraceTimestamp<Ops>(stats.trace, stats.result);
        if (!cross_core_dag::ExecutePaBoundKernel<Ops>(
                state, worker, token, kind,
                NopCountForKind(state->config.nops, kind)
            )) {
            PublishCrossCoreRuntimeFailure<Ops>(
                state, stats,
                cross_core_dag::ExecFatalReason::InvalidTokenPayload,
                token_task_id, worker_id
            );
            token.control.phase =
                cross_core_dag::ExecTokenPhase::Faulted;
            return false;
        }
        const uint64_t kernel_end =
            TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<false>(
            stats.trace, stats.result,
            static_cast<int32_t>(token_task_id),
            static_cast<int32_t>(token_function_id),
            TracePhase::Kernel, ProfilePhase::ReplayTail,
            kernel_begin, kernel_end
        );
        RecordKernelCycles(
            stats, kind, kernel_end - kernel_begin
        );

        const cross_core_dag::PaExecSynchronousEngineCompletion
            engine_complete{};
        if (!cross_core_dag::TryMarkExecutionTokenCompleting<Ops>(
                token, engine_complete, state->exec_fatal,
                observer
            )) {
            PublishCrossCoreRuntimeFailure<Ops>(
                state, stats,
                cross_core_dag::ExecFatalReason::InvalidTokenPayload,
                token_task_id, worker_id
            );
            token.control.phase =
                cross_core_dag::ExecTokenPhase::Faulted;
            return false;
        }
    } else if (
        token.control.phase ==
            cross_core_dag::ExecTokenPhase::EngineInflight ||
        token.control.phase ==
            cross_core_dag::ExecTokenPhase::Binding ||
        token.control.phase ==
            cross_core_dag::ExecTokenPhase::Faulted
    ) {
        // 当前 ExecuteKernel 是同步边界，ENGINE_INFLIGHT/BINDING 不应
        // 跨越一次 progress 返回。若观察到这些状态，必须
        // fail-closed，否则可能在没有真正执行 kernel 时发布完成。
        PublishCrossCoreRuntimeFailure<Ops>(
            state, stats,
            cross_core_dag::ExecFatalReason::InvalidTokenPayload,
            token.control.task_id, worker_id
        );
        token.control.phase = cross_core_dag::ExecTokenPhase::Faulted;
        return false;
    }

    if (token.control.phase ==
            cross_core_dag::ExecTokenPhase::Completing ||
        token.control.phase ==
            cross_core_dag::ExecTokenPhase::VendPublished ||
        token.control.phase ==
            cross_core_dag::ExecTokenPhase::CompletionPublished) {
        const uint32_t completed_task = token.control.task_id;
        const uint32_t completed_function =
            cross_core_dag::ExecutionTokenFunctionId(token);
        cross_core_dag::ObservedPaExecCompletionSink<
            Ops, SharedExecTraceObserver<Ops>
        > completion{state, &observer};
        if (cross_core_dag::PublishExecDoneAfterCompletion<Ops>(
                state->exec_cells[completed_task], token,
                completion, state->exec_fatal, observer
            ) != cross_core_dag::ExecDoneResult::Done) {
            SetFatal<Ops>(
                state, stats,
                static_cast<int32_t>(completed_task)
            );
            return false;
        }
        const uint64_t commit_cycle =
            TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<false>(
            stats.trace, stats.result,
            static_cast<int32_t>(completed_task),
            static_cast<int32_t>(completed_function),
            TracePhase::Commit, ProfilePhase::ReplayTail,
            commit_cycle, commit_cycle
        );
        ++stats.result.placement[static_cast<uint32_t>(place)];
        completed = true;
    }
    return token.control.phase == cross_core_dag::ExecTokenPhase::Idle;
}

template <typename Ops>
PA_DEVICE bool ObserveCrossCoreFinalDrainFatal(
    PA_GM SchedulerState *state, uint32_t worker_id,
    LocalStats &stats
) {
    if (!IsFatal<Ops>(state, stats)) {
        return false;
    }
    // 只在 FinalDrain 的节流观察点命中 terminal 后收敛本核 token。
    // token 是 owner-local 普通状态，不需要额外 Atomic 或 DCCI。
    for (uint32_t token_slot = 0;
         token_slot < cross_core_dag::kExecTokensPerWorker;
         ++token_slot) {
        PA_GM cross_core_dag::ExecutionToken &token =
            state->exec_tokens[worker_id][token_slot];
        if (token.control.phase != cross_core_dag::ExecTokenPhase::Idle) {
            token.control.phase = cross_core_dag::ExecTokenPhase::Faulted;
        }
    }
    return true;
}

template <typename Ops>
PA_DEVICE bool ProgressCrossCoreOwnedTokens(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker,
    DrainPlace place, LocalStats &stats,
    uint32_t &completed_count
) {
    // 固定按 token 号检查，但只要后一个 token 的完成可能解开前一个
    // token，就再扫一轮。每个同步 token 在本次调用中最多完成一次，
    // 循环上界由 kExecTokensPerWorker 给出。
    bool completed_in_round = false;
    do {
        completed_in_round = false;
        for (uint32_t token_slot = 0;
             token_slot < cross_core_dag::kExecTokensPerWorker;
             ++token_slot) {
            bool completed = false;
            if (!ProgressCrossCoreActiveToken<Ops>(
                    state, worker, token_slot, place,
                    stats, completed
                )) {
                return false;
            }
            if (completed) {
                ++completed_count;
                completed_in_round = true;
            }
        }
    } while (completed_in_round);
    return true;
}

// PlanHeaderValidated 只消除 launch 前一次发布、运行期只读的摘要复核，
// 不放宽 token/cell/payload/fanin/completion 的动态协议检查。
template <typename Ops, bool PlanHeaderValidated = false>
PA_DEVICE uint32_t ProgressCrossCoreExec(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker,
    uint32_t task_count, bool production_closed,
    DrainPlace place, LocalStats &stats
) {
    (void)production_closed;
    const bool basic_input_valid =
        state != nullptr && worker.core_idx >= 0 &&
        static_cast<uint32_t>(worker.core_idx) < kWorkers &&
        task_count != 0 && task_count <= kMaxTasks &&
        CrossCoreExecWorkerMatchesRole(
            static_cast<uint32_t>(worker.core_idx), worker.role
        );
    bool plan_header_valid = true;
    if constexpr (!PlanHeaderValidated) {
        plan_header_valid = basic_input_valid &&
            state->build_dispatch.task_count == task_count &&
            state->exec_dispatch.aic_task_count <= task_count &&
            state->exec_dispatch.aiv_task_count <= task_count &&
            state->exec_dispatch.aic_task_count +
                    state->exec_dispatch.aiv_task_count ==
                state->build_dispatch.executable_task_count;
    }
    if (!basic_input_valid || !plan_header_valid) {
        if (state != nullptr && worker.core_idx >= 0 &&
            static_cast<uint32_t>(worker.core_idx) < kWorkers) {
            PublishCrossCoreRuntimeFailure<Ops>(
                state, stats,
                cross_core_dag::ExecFatalReason::InvalidTokenPayload,
                task_count,
                static_cast<uint32_t>(worker.core_idx)
            );
        }
        return 0;
    }
    const uint32_t worker_id =
        static_cast<uint32_t>(worker.core_idx);
    const cross_core_dag::ExecEngineClass role_engine =
        CrossCoreEngineForRole(worker.role);

    uint32_t completed_count = 0;
    if (!ProgressCrossCoreOwnedTokens<Ops>(
            state, worker, place, stats, completed_count
        )) {
        return 0;
    }
    if (CrossCoreExecOccupiedTokenCount(state, worker_id) >
        cross_core_dag::kExecTokensPerWorker -
            cross_core_dag::kExecTicketBatchSize) {
        return completed_count;
    }

    while (stats.exec_dispatch_exhausted == 0) {
        if (CrossCoreExecOccupiedTokenCount(state, worker_id) >
            cross_core_dag::kExecTokensPerWorker -
                cross_core_dag::kExecTicketBatchSize) {
            break;
        }
        const SharedExecTicketResult ticket =
            TakeSharedExecTicket<Ops, PlanHeaderValidated>(
                state, worker_id, worker.role,
                task_count, stats
            );
        if (ticket.status ==
            SharedExecTicketStatus::Exhausted) {
            break;
        }
        if (ticket.status != SharedExecTicketStatus::Acquired) {
            PublishCrossCoreRuntimeFailure<Ops>(
                state, stats,
                cross_core_dag::ExecFatalReason::InvalidBuiltControl,
                ticket.task_ids[0], worker_id
            );
            return completed_count;
        }
        if (ticket.task_count == 0 ||
            ticket.task_count > cross_core_dag::kExecTicketBatchSize) {
            PublishCrossCoreRuntimeFailure<Ops>(
                state, stats,
                cross_core_dag::ExecFatalReason::InvalidBuiltControl,
                ticket.task_ids[0], worker_id
            );
            return completed_count;
        }

        uint32_t bound_slots[cross_core_dag::kExecTicketBatchSize] = {
            cross_core_dag::kExecTokensPerWorker,
            cross_core_dag::kExecTokensPerWorker,
        };
        for (uint32_t index = 0;
             index < ticket.task_count; ++index) {
            const uint32_t task_id = ticket.task_ids[index];
            SharedExecDispatchRoute planned{};
            if (!DecodeSharedExecDispatchRoute(
                    state->build_dispatch, task_id, planned
                ) ||
                !planned.executable ||
                planned.engine_class != role_engine) {
                PublishCrossCoreRuntimeFailure<Ops>(
                    state, stats,
                    cross_core_dag::ExecFatalReason::InvalidBuiltControl,
                    task_id, worker_id
                );
                return completed_count;
            }
            const uint32_t idle_token_slot =
                CrossCoreExecFirstIdleTokenSlot(
                    state, worker_id
                );
            if (idle_token_slot >=
                cross_core_dag::kExecTokensPerWorker) {
                PublishCrossCoreRuntimeFailure<Ops>(
                    state, stats,
                    cross_core_dag::ExecFatalReason::InvalidTokenPayload,
                    task_id, worker_id
                );
                return completed_count;
            }
            PA_GM cross_core_dag::ExecutionToken &token =
                state->exec_tokens[worker_id][idle_token_slot];
            if (!BindSharedExecTicketToToken(
                    token, task_id, worker_id, role_engine
                )) {
                PublishCrossCoreRuntimeFailure<Ops>(
                    state, stats,
                    cross_core_dag::ExecFatalReason::InvalidTokenPayload,
                    task_id, worker_id
                );
                token.control.phase =
                    cross_core_dag::ExecTokenPhase::Faulted;
                return completed_count;
            }
            bound_slots[index] = idle_token_slot;
        }
        const uint32_t occupied =
            CrossCoreExecOccupiedTokenCount(state, worker_id);
        if (occupied > stats.max_occupied) {
            stats.max_occupied = static_cast<uint8_t>(occupied);
        }

        if (!ProgressCrossCoreOwnedTokens<Ops>(
                state, worker, place, stats, completed_count
            )) {
            return completed_count;
        }

        // 一个批次只预留两个 ordinal。首次观察后只要其中任一项尚未
        // BUILT，本调度边界就停止继续取下一批；已经绑定的另一项仍由
        // 上面的完整 token 扫描及时推进。这样既保留函数条带，又不恢复
        // 历史上“一次占满四槽未来任务”的长前视。
        bool batch_waiting_built = false;
        for (uint32_t index = 0;
             index < ticket.task_count; ++index) {
            batch_waiting_built |=
                state->exec_tokens[worker_id]
                    [bound_slots[index]].control.phase ==
                cross_core_dag::ExecTokenPhase::WaitingBuilt;
        }
        if (batch_waiting_built) {
            break;
        }
    }
    return completed_count;
}

template <bool PlanHeaderValidated = false>
PA_DEVICE bool CrossCoreExecWorkerDrained(
    PA_GM const SchedulerState *state,
    PA_GM const WorkerState &worker, uint32_t task_count,
    const LocalStats &stats
) {
    const bool basic_input_valid =
        state != nullptr && worker.core_idx >= 0 &&
        static_cast<uint32_t>(worker.core_idx) < kWorkers &&
        task_count <= kMaxTasks &&
        CrossCoreExecWorkerMatchesRole(
            static_cast<uint32_t>(worker.core_idx), worker.role
        );
    bool plan_header_valid = true;
    if constexpr (!PlanHeaderValidated) {
        plan_header_valid = basic_input_valid &&
            state->build_dispatch.task_count == task_count &&
            state->exec_dispatch.aic_task_count <= task_count &&
            state->exec_dispatch.aiv_task_count <= task_count &&
            state->exec_dispatch.aic_task_count +
                    state->exec_dispatch.aiv_task_count ==
                state->build_dispatch.executable_task_count;
    }
    if (!basic_input_valid || !plan_header_valid) {
        return false;
    }
    const uint32_t worker_id =
        static_cast<uint32_t>(worker.core_idx);
    return stats.exec_dispatch_exhausted != 0 &&
           CrossCoreExecAllTokensIdle(state, worker_id);
}

PA_DEVICE bool CrossCoreExecTokenFullyReset(
    PA_GM const cross_core_dag::ExecutionToken &token
) {
    PA_GM const cross_core_dag::ExecutionTokenControl &control =
        token.control;
    return control.phase == cross_core_dag::ExecTokenPhase::Idle &&
           control.task_id == UINT32_MAX &&
           control.build_owner == UINT32_MAX &&
           control.execute_owner == UINT32_MAX &&
           control.engine_class ==
               cross_core_dag::ExecEngineClass::None &&
           control.payload_lines == 0 &&
           control.payload_bytes == 0 &&
           control.fanin_ready_prefix == 0;
}

PA_DEVICE bool CrossCoreExecAllTokensFullyReset(
    PA_GM const SchedulerState *state, uint32_t worker_id
) {
    for (uint32_t token_slot = 0;
         token_slot < cross_core_dag::kExecTokensPerWorker;
         ++token_slot) {
        if (!CrossCoreExecTokenFullyReset(
                state->exec_tokens[worker_id][token_slot]
            )) {
            return false;
        }
    }
    return true;
}

template <typename Ops, typename Observer>
PA_DEVICE bool ValidateCrossCoreExecTerminalCellsObserved(
    PA_GM SchedulerState *state, uint32_t task_count,
    uint32_t &first_bad_task, Observer &observer
) {
    first_bad_task = task_count;
    if (state == nullptr || task_count > kMaxTasks ||
        state->build_dispatch.task_count != task_count ||
        state->build_dispatch.executable_task_count > task_count) {
        return false;
    }

    uint32_t executable_tasks = 0;
    for (uint32_t task_id = 0;
         task_id < task_count; ++task_id) {
        SharedExecDispatchRoute planned{};
        if (!DecodeSharedExecDispatchRoute(
                state->build_dispatch, task_id, planned
            )) {
            first_bad_task = task_id;
            return false;
        }
        const cross_core_dag::DecodedExecState decoded =
            cross_core_dag::DecodeExecState(
                observer.LoadCellState(
                    &state->exec_cells[task_id]
                         .control.state,
                    task_id
                )
            );
        bool valid = decoded.valid &&
            observer.LoadFaninFlag(
                &state->tasks[task_id].flag,
                task_id
            ) == 1;
        if (valid && !planned.executable) {
            valid =
                planned.engine_class ==
                    cross_core_dag::ExecEngineClass::None &&
                decoded.phase == cross_core_dag::ExecPhase::Empty;
        } else if (valid) {
            ++executable_tasks;
            valid =
                decoded.phase == cross_core_dag::ExecPhase::Done &&
                decoded.task_id == task_id &&
                decoded.engine_class == planned.engine_class &&
                cross_core_dag::A5SingleLaneBuildOwnerEligible(
                    decoded.build_owner, planned.engine_class
                ) &&
                cross_core_dag::A5SingleLaneExecuteOwnerEligible(
                    task_id, decoded.build_owner,
                    planned.engine_class, decoded.execute_owner
                );
        }
        if (!valid) {
            first_bad_task = task_id;
            return false;
        }
    }
    return executable_tasks ==
        state->build_dispatch.executable_task_count;
}

template <typename Ops>
PA_DEVICE bool ValidateCrossCoreExecTerminalCells(
    PA_GM SchedulerState *state, uint32_t task_count,
    uint32_t &first_bad_task, LocalStats &stats
) {
    SharedExecTraceObserver<Ops> observer{&stats};
    return ValidateCrossCoreExecTerminalCellsObserved<Ops>(
        state, task_count, first_bad_task, observer
    );
}

// 无 TraceContext 的协议门槛仍复用同一终态判定；正式 scheduler 只走
// 上面的观察版本，不会因保留测试入口而多执行共享原语。
template <typename Ops>
PA_DEVICE bool ValidateCrossCoreExecTerminalCells(
    PA_GM SchedulerState *state, uint32_t task_count,
    uint32_t &first_bad_task
) {
    cross_core_dag::DirectExecObserver<Ops> observer{};
    return ValidateCrossCoreExecTerminalCellsObserved<Ops>(
        state, task_count, first_bad_task, observer
    );
}

template <typename Ops, bool PlanHeaderValidated = false>
PA_DEVICE bool ProgressCrossCoreExecDrainClosure(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker,
    uint32_t task_count, LocalStats &stats,
    bool &arrived, bool &closed
) {
    if (closed) {
        return true;
    }
    closed = false;
    if (state == nullptr || worker.core_idx < 0 ||
        static_cast<uint32_t>(worker.core_idx) >= kWorkers ||
        worker.block_id < 0 ||
        static_cast<uint32_t>(worker.block_id) >= kAicWorkers) {
        return false;
    }
    const uint32_t worker_id =
        static_cast<uint32_t>(worker.core_idx);
    static_assert(
        kWorkers % cross_core_dag::kExecDrainArrivalGroups == 0,
        "execution drain groups must evenly partition workers"
    );
    constexpr int64_t kWorkersPerArrivalGroup =
        static_cast<int64_t>(
            kWorkers / cross_core_dag::kExecDrainArrivalGroups
        );
    const uint32_t arrival_group =
        static_cast<uint32_t>(worker.block_id) %
        cross_core_dag::kExecDrainArrivalGroups;
    if (!arrived) {
        if (!CrossCoreExecWorkerDrained<PlanHeaderValidated>(
                state, worker, task_count, stats
            ) ||
            !CrossCoreExecAllTokensFullyReset(state, worker_id)) {
            return true;
        }
        uint64_t local_completed = 0;
        for (uint32_t place = 0;
             place < static_cast<uint32_t>(DrainPlace::Count);
             ++place) {
            local_completed += stats.result.placement[place];
        }
        const int64_t contribution =
            cross_core_dag::EncodeExecDrainArrivalContribution(
                local_completed
            );
        if (contribution < 0 || local_completed > task_count) {
            PublishCrossCoreRuntimeFailure<Ops>(
                state, stats,
                cross_core_dag::ExecFatalReason::CompletionStateConflict,
                task_count, worker_id
            );
            return false;
        }
        const int64_t prior = TraceAtomicFetchAdd<Ops>(
            stats.trace, stats.result, -1,
            AtomicSite::SharedExecDrainArrive,
            &state->exec_drain.arrivals[arrival_group].state,
            contribution,
            /*result_used=*/true
        );
        const uint32_t prior_arrivals =
            cross_core_dag::DecodeExecDrainArrivalCount(prior);
        const uint64_t prior_completed =
            cross_core_dag::DecodeExecDrainCompletionCount(prior);
        if (prior_arrivals == UINT32_MAX ||
            prior_arrivals >= kWorkersPerArrivalGroup ||
            prior_completed > task_count) {
            PublishCrossCoreRuntimeFailure<Ops>(
                state, stats,
                cross_core_dag::ExecFatalReason::CompletionStateConflict,
                task_count, worker_id
            );
            return false;
        }
        arrived = true;
    }

    // 中央 ticket 循环保证本 worker 不会再生产 BUILT；arrival 又只在
    // 本核 scanner/token/engine 全部排空后发布。非 root 此后没有新的
    // 可执行工作，可以直接结束本核；固定 root 等待全部分组到齐并核对
    // 精确完成数。所有 worker 的到达合在一起同时证明 Build 与 Execute
    // 都已收口，不再依赖一层独立 replay barrier。
    if (worker_id != 0) {
        closed = true;
        return true;
    }
    uint64_t completed_tasks = 0;
    for (uint32_t group = 0;
         group < cross_core_dag::kExecDrainArrivalGroups;
         ++group) {
        const int64_t group_state =
            TraceAtomicLoad<Ops>(
                stats.trace, stats.result, -1,
                AtomicSite::SharedExecDrainArrivalPoll,
                &state->exec_drain.arrivals[group].state,
                /*result_used=*/true
            );
        const uint32_t group_arrivals =
            cross_core_dag::DecodeExecDrainArrivalCount(group_state);
        const uint64_t group_completed =
            cross_core_dag::DecodeExecDrainCompletionCount(group_state);
        if (group_arrivals == UINT32_MAX ||
            group_arrivals > kWorkersPerArrivalGroup ||
            group_completed > task_count ||
            completed_tasks > task_count - group_completed) {
            PublishCrossCoreRuntimeFailure<Ops>(
                state, stats,
                cross_core_dag::ExecFatalReason::CompletionStateConflict,
                task_count, worker_id
            );
            return false;
        }
        if (group_arrivals != kWorkersPerArrivalGroup) {
            return true;
        }
        completed_tasks += group_completed;
    }

    const uint32_t expected_executable_tasks =
        state->build_dispatch.executable_task_count;
    if (expected_executable_tasks > task_count ||
        completed_tasks != expected_executable_tasks) {
        PublishCrossCoreRuntimeFailure<Ops>(
            state, stats,
            cross_core_dag::ExecFatalReason::CompletionStateConflict,
            task_count, worker_id
        );
        return false;
    }
    closed = true;
    return true;
}

#include "pa_shared_submit_path.h"
#endif

template <typename Ops, bool Profile, bool SharedDispatched,
          typename PmuContext>
PA_DEVICE bool FinishCallbackSubmitBody(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_count,
    const TaskArgs &args, SubmitContext &context, LocalStats &stats,
    PmuContext &pmu_context, const CallbackSubmitTicket &ticket
) {
#if PTO_FDWIC_SHARED_MAP
    (void)task_count;
    return FinishSharedWinnerSubmitBody<
        Ops, Profile, SharedDispatched
    >(
        state, worker, args, context, stats, pmu_context, ticket
    );
#else
    const uint32_t task_id = ticket.task_id;
#if PTO_FDWIC_SHARED_MAP
    SharedPaTaskMeta shared_task_meta{};
    if (!DecodeSharedPaTaskMeta(
            ticket.reserved, task_id, shared_task_meta
        )) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    const TaskKind kind = shared_task_meta.kind;
#else
    const TaskKind kind = GetTaskKind(task_id);
#endif
    const int32_t function_id = static_cast<int32_t>(ticket.function_id);
    const bool winner = ticket.won != 0;
#if PTO_FDWIC_SHARED_MAP
    if (!SharedPaFunctionIdMatches(kind, winner, function_id)) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    // shared loser 必须在 caller 的轻路径返回；跨 TU / 完整 Finish 只允许
    // winner 进入。这样 Materialize、Fanin、Register 与 Build 的边界才与
    // 实际 shared TensorMap 协议一致。
    if (!winner) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
#endif

#if PTO_FDWIC_SHARED_MAP
    // callback 已经返回；只有 Claim winner 才把 CreateInfo 物化为 descriptor
    // 并预留 shared heap。loser 已在 caller 轻路径返回，这里只处理 winner。
    // PA Case1 的普通 region 恒为空，winner 不再等待全局 exact turn；
    // 跨 task 顺序只由实际消费的 (producer,slot).published 建立。
    // 删除 exact-turn 不能连带删除它成功出口的终止态检查：若其他核已经
    // 广播 fatal，本 winner 不得继续预留 heap、构建 slot 或发布 symbol。
    // 这里直接使用 Ops，不扩张 atomic 泳道记录，也不恢复任何全局前沿。
    // PA_ATOMIC_DCCI_SOURCE_EXEMPT: test-only - 该分支位于 private 外层中的不可达 shared 旧实现
    if (Ops::Load(&state->fatal.value) != 0) {
        return false;
    }
    const uint64_t materialize_begin =
        TraceTimestamp<Ops>(stats.trace, stats.result);
    BeginSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(pmu_context);
    const bool materialized = MaterializeTask<Ops, true>(
        worker, task_id, args, context, state->shared_map,
        state->heap_base, state->heap_size,
        kind, shared_task_meta.batch_start,
        shared_task_meta.group_index,
        &stats.trace, &stats.result
    );
    if (materialized) {
        stats.result.materialized_outputs += context.result.count;
    }
    if (!materialized) {
        EndSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(pmu_context);
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    // Materialize 只负责 shared heap reserve 与 descriptor 构造。fresh
    // symbol 必须等本任务 CompleteTask/BuildWinner 成功后再封口，因此
    // 这里不能提前写 published。
    EndSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(pmu_context);
    const uint64_t materialize_end =
        TraceTimestamp<Ops>(stats.trace, stats.result);
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), function_id,
        TracePhase::Materialize, ProfilePhase::Materialize,
        materialize_begin, materialize_end, 0,
        kind == TaskKind::Alloc ? 1U : 0U
    );
    // shared PA Case1 没有 ordinary-region PrepareMap；不再为兼容旧矩形
    // 泳道写零时长 marker。host/analyzer 直接校验 shared 稀疏真实边界。
#else
    // private 模式保持 S3.1 的 eager Materialize 与每核 heap 路径不变。
    const uint64_t materialize_begin =
        TraceTimestamp<Ops>(stats.trace, stats.result);
    BeginSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(pmu_context);
    const bool materialized =
        MaterializeTask(
            worker, task_id, args, context,
            state->heap_base, state->heap_size
        );
    if (!materialized) {
        EndSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(pmu_context);
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    stats.result.materialized_outputs += context.result.count;
    EndSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(pmu_context);
    const uint64_t materialize_end =
        TraceTimestamp<Ops>(stats.trace, stats.result);
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), function_id,
        TracePhase::Materialize, ProfilePhase::Materialize,
        materialize_begin, materialize_end, 0,
        kind == TaskKind::Alloc ? 1U : 0U
    );

    const uint64_t prepare_begin = materialize_end;
    AdvanceTensorMap(worker.map, task_id, static_cast<int32_t>(state->heap_window));
    const uint64_t prepare_end = TraceTimestamp<Ops>(stats.trace, stats.result);
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), function_id,
        TracePhase::PrepareMap, ProfilePhase::PrepareMap,
        prepare_begin, prepare_end, 0, kind == TaskKind::Alloc ? 1U : 0U
    );
#endif

#if PTO_FDWIC_SHARED_MAP
    bool shared_writers_prepared = false;
#endif
    uint64_t register_begin =
#if PTO_FDWIC_SHARED_MAP
        materialize_end;
#else
        prepare_end;
#endif
#if PTO_FDWIC_SHARED_MAP
    if (kind != TaskKind::Alloc) {
#else
    if (kind != TaskKind::Alloc &&
        __builtin_expect(winner, 0)) {
#endif
        const uint64_t fanin_begin = register_begin;
#if PTO_FDWIC_SHARED_MAP
        bool lookup_protocol_ok = false;
        uint32_t ordinary_lookup_count = 0;
        if (shared_task_meta.chained_writer) {
            context.fanin_count = static_cast<int32_t>(
                CollectSharedFanin<Ops, true>(
                    state->shared_map, args,
                    static_cast<int32_t>(task_id),
                    static_cast<int32_t>(state->heap_window), stats,
                    context.fanin, lookup_protocol_ok,
                    ordinary_lookup_count, &state->fatal.value,
                    static_cast<int32_t>(
                        shared_task_meta.batch_start
                    ),
                    static_cast<int32_t>(task_id) - 4
                )
            );
        } else {
            context.fanin_count = static_cast<int32_t>(
                CollectSharedFanin<Ops>(
                    state->shared_map, args,
                    static_cast<int32_t>(task_id),
                    static_cast<int32_t>(state->heap_window), stats,
                    context.fanin, lookup_protocol_ok,
                    ordinary_lookup_count, &state->fatal.value
                )
            );
        }
        if (!lookup_protocol_ok) {
            SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
            return false;
        }
        stats.result.map_lookups += ordinary_lookup_count;
#else
        context.fanin_count = static_cast<int32_t>(CollectFanin(worker.map, args, context.fanin));
        stats.result.map_lookups += static_cast<uint32_t>(args.tensor_count) - context.result.count;
#endif
        for (int32_t edge = 0; edge < context.fanin_count; ++edge) {
            stats.result.dependency_signature ^=
                DependencyEdgeSignature(
                    task_id,
                    static_cast<uint32_t>(context.fanin[edge])
                );
        }
        const uint64_t fanin_end = TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<Profile>(
            stats.trace, stats.result, static_cast<int32_t>(task_id), function_id,
            TracePhase::Fanin, ProfilePhase::Fanin,
            fanin_begin, fanin_end, 0, static_cast<uint32_t>(context.fanin_count)
        );
        register_begin = fanin_end;
    }

    BeginSubmitPmuPhase<SubmitPmuPhase::Register, Ops>(pmu_context);
#if PTO_FDWIC_SHARED_MAP
    // 当前 standalone 只模拟 PA Case1：fresh symbol 直接寻址，
    // output_view 又是 manual_dep，ordinary region 必须严格为空。
    // 这里只读验证，不构造空 delta，也不触碰 region sequencer。
    const bool registered =
        ValidateEmptySharedRegistration(args, context);
#else
    const bool registered = RegisterOutputs(context, args, kind != TaskKind::Alloc);
    if (registered && kind != TaskKind::Alloc) {
        stats.result.map_inserts += CountBits(context.register_mask);
    }
#endif
    EndSubmitPmuPhase<SubmitPmuPhase::Register, Ops>(pmu_context);
    const uint64_t register_end = TraceTimestamp<Ops>(stats.trace, stats.result);
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), function_id,
        TracePhase::Register, ProfilePhase::Register,
        register_begin, register_end, 0, kind == TaskKind::Alloc ? 0U : 1U
    );
    if (!registered) {
#if PTO_FDWIC_SHARED_MAP
        // PA Case1 不接入 ordinary-region backend。非 manual-dep 的普通
        // writer 或非法 register mask 会在 region append 前失败并广播
        // fatal；此前 Materialize 和 fanin 仍可能读取 shared sidecar。
#else
        // 固定桶容量不足时，InsertTensor 没有覆写任何 live 槽。沿用现有
        // fatal 广播终止所有 worker，禁止像旧 linked map 一样静默漏登记
        // hazard、随后带着不完整 fanin 继续执行。
#endif
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }

#if PTO_FDWIC_SHARED_MAP
    if (shared_task_meta.has_following_group) {
        // PA 的 non-final UP 固定有 SF/PV/accumulator 三条 fanin。先完成
        // registration，再登记 writer intent并放行 loser；Build 后只做
        // fresh-output 封口，绝不能重复 Collect/Commit。
        if (kind != TaskKind::Up || context.fanin_count != 3) {
            SetFatal<Ops>(
                state, stats, static_cast<int32_t>(task_id)
            );
            return false;
        }
        const bool prepared = shared_task_meta.chained_writer
            ? CommitPaSharedWriterIntentAfterFanin<Ops, true>(
                  state, args, context, stats,
                  static_cast<int32_t>(
                      shared_task_meta.batch_start
                  ),
                  static_cast<int32_t>(task_id) - 4
              )
            : CommitPaSharedWriterIntentAfterFanin<Ops>(
                  state, args, context, stats
              );
        if (!prepared) {
            return false;
        }
        shared_writers_prepared = true;
    }
#endif

#if PTO_FDWIC_SHARED_MAP
    {
#else
    if (__builtin_expect(winner, 0)) {
#endif
        const uint64_t winner_build_begin = register_end;
#if PTO_FDWIC_SHARED_MAP && \
    defined(PA_TEST_SHARED_POST_GATE_BUILD_FAILURE)
        // 只供 host 96-worker 故障门槛使用：non-final UP 已完成 writer
        // intent 与 insert-completion 发布后、建立可执行 slot 前注入失败。
        // 普通 CPU/CCEC 不定义该宏，预处理后不保留调用或分支。
        if (shared_writers_prepared &&
            Ops::InjectSharedPostGateBuildFailure(
                state, worker, task_id, kind
            )) {
            SetFatal<Ops>(
                state, stats, static_cast<int32_t>(task_id)
            );
            return false;
        }
#endif
        if (kind == TaskKind::Alloc) {
#if !PTO_FDWIC_SHARED_MAP
            if (!HeapGuard<Ops, Profile>(state, worker, task_id, context.output_bytes, stats)) {
                return false;
            }
#endif
            CompleteTask<Ops>(state, worker, task_id, stats);
        } else {
            if (!BuildWinner<Ops, Profile>(
                    state, worker, task_id, kind, args, context, context.fanin,
                    static_cast<uint32_t>(context.fanin_count), stats
                )) {
                return false;
            }
        }
        // 先建立可执行状态。普通/final task 随后提交本任务的 INOUT
        // writer；non-final UP 已在 Build 前登记 writer intent，这里只
        // 跳过重复 Commit。fresh outputs 最后封口；后继只等待自己实际
        // 依赖的 published cell，不再经过全局 committed_tasks。
        // published 成功之后只剩观察记录与 Submit 收尾。
#if PTO_FDWIC_SHARED_MAP
        if (!PublishSharedWinnerAfterBuild<Ops>(
                state, worker, args, context, task_id, kind, stats,
                shared_writers_prepared,
                shared_task_meta.chained_writer,
                static_cast<int32_t>(
                    shared_task_meta.batch_start
                ),
                static_cast<int32_t>(task_id) - 4
            )) {
            return false;
        }
#endif
        const uint64_t winner_build_end = TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<false>(
            stats.trace, stats.result, static_cast<int32_t>(task_id), function_id,
            kind == TaskKind::Alloc ? TracePhase::AllocComplete : TracePhase::WinnerBuild,
            ProfilePhase::ReplayTail, winner_build_begin, winner_build_end
        );
    }

#if PTO_FDWIC_SHARED_MAP
    (void)task_count;
    return CloseSharedCallbackSubmit<Ops, Profile>(
        state, stats, ticket, shared_task_meta, kind
    );
#else
    ++stats.result.submits;
#if PA_BUILD_PERF_CLOCK
    const uint64_t submit_end = 0;
#elif PA_BUILD_SUBMIT_PMU
    const uint64_t submit_end = task_id + 1 == task_count ? Ops::Now() : 0;
#else
    const uint64_t submit_end = TraceTimestamp<Ops>(stats.trace, stats.result);
#endif
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), function_id,
        TracePhase::Submit, ProfilePhase::Submit,
        ticket.submit_begin, submit_end, winner ? 1U : 0U, kind == TaskKind::Alloc ? 1U : 0U
    );
    if (task_id + 1 == task_count) stats.result.submit_end = submit_end;
    return true;
#endif
#endif
}

#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
template <typename Ops>
PA_DEVICE uint32_t FinishSplitCallbackSubmitFromRuntime(
    const CallbackSubmitTicket *ticket, const TaskArgs *args
) {
    CompeteFirstSplitRuntimeState &runtime = Ops::CompeteFirstSplitState();
    const uint64_t state_address = reinterpret_cast<uint64_t>(&runtime);
    runtime.finish_state_address = state_address;
#if PTO_FDWIC_SHARED_MAP
    SharedPaTaskMeta ticket_meta{};
    const uint64_t ticket_binding = runtime.reserved;
    const bool dispatched =
        (ticket_binding & kSharedSplitTicketDispatched) != 0;
    const bool expected_ticket_bound =
        ticket != nullptr &&
        ticket_binding ==
            SharedSplitTicketBinding(*ticket, dispatched);
    // binding 是一次性的 caller→finish 交接状态。无论 ticket 是否
    // 合法都在读取后清零，使 fatal 路也能收敛，并让最终协议检查继续
    // 要求 runtime.reserved==0。
    runtime.reserved = 0;
    bool valid =
        ticket != nullptr && args != nullptr &&
        runtime.scheduler != nullptr &&
        runtime.worker != nullptr && runtime.task_count != 0 &&
        runtime.worker_id < kWorkers &&
        runtime.owner_worker_id == runtime.worker_id &&
        runtime.worker->core_idx ==
            static_cast<int32_t>(runtime.worker_id) &&
        runtime.caller_state_address == state_address &&
        runtime.state_cookie == CompeteFirstSplitStateCookie(
            runtime.worker_id, runtime.worker->role
        ) &&
        expected_ticket_bound && ticket->won != 0 &&
        (dispatched ||
         static_cast<uint64_t>(ticket->task_id) ==
             runtime.stats.result.submits);
#else
    bool valid = ticket != nullptr && args != nullptr && runtime.scheduler != nullptr &&
                 runtime.worker != nullptr && runtime.task_count != 0 &&
                 runtime.worker_id < kWorkers && runtime.owner_worker_id == runtime.worker_id &&
                 runtime.worker->core_idx == static_cast<int32_t>(runtime.worker_id) &&
                 runtime.caller_state_address == state_address &&
                 runtime.state_cookie == CompeteFirstSplitStateCookie(
                     runtime.worker_id, runtime.worker->role
                 ) && runtime.reserved == 0;
#endif
    if (valid) {
#if PTO_FDWIC_SHARED_MAP
        valid = ticket->task_id < runtime.task_count &&
                runtime.context.task_id == static_cast<int32_t>(ticket->task_id) &&
                runtime.context.kernel_id == static_cast<int32_t>(ticket->function_id) &&
                runtime.context.won == (ticket->won != 0);
        if (valid) {
            valid =
                DecodeSharedPaTaskMeta(
                    ticket->reserved, ticket->task_id, ticket_meta
                ) &&
                SharedPaFunctionIdMatches(
                    ticket_meta.kind, ticket->won != 0,
                    static_cast<int32_t>(ticket->function_id)
                ) &&
                runtime.context.shared_result.TaskId() ==
                    static_cast<int32_t>(ticket->task_id) &&
                runtime.context.shared_result.Size() ==
                    FrontendTaskOutputCount(ticket_meta.kind);
        }
#else
        valid = ticket->reserved == 0 && ticket->task_id < runtime.task_count &&
                runtime.context.task_id == static_cast<int32_t>(ticket->task_id) &&
                runtime.context.kernel_id == static_cast<int32_t>(ticket->function_id) &&
                runtime.context.won == (ticket->won != 0);
#endif
    }
    ++runtime.finish_calls;
#if !PTO_FDWIC_SHARED_MAP
    if (ticket != nullptr) runtime.task_id_sum += ticket->task_id;
#endif
    if (!valid) {
        ++runtime.protocol_errors;
        if (runtime.scheduler != nullptr) {
            SetFatal<Ops>(
                runtime.scheduler, runtime.stats,
                ticket == nullptr ? -1 : static_cast<int32_t>(ticket->task_id)
            );
        }
        return 0;
    }

#if PA_BUILD_SUBMIT_PMU
    // none 不需要 finish 内的局部 PMU context；Claim/EfDrain 的起止点都在
    // callback 与 finish 之前，因此也能保持 split 形状。Materialize/Register
    // 边界在 finish 内，仍由构建脚本选择 inline finish 以复用同一 PmuContext。
    static_assert(
        kCompiledSubmitPmuPhase == SubmitPmuPhase::None ||
            kCompiledSubmitPmuPhase == SubmitPmuPhase::Claim ||
            kCompiledSubmitPmuPhase == SubmitPmuPhase::EfDrain,
        "split callback submit-PMU supports none/claim/efdrain only"
    );
#endif
    bool pmu_context = false;
#if PTO_FDWIC_SHARED_MAP
    if (dispatched) {
        return FinishCallbackSubmitBody<Ops, false, true>(
            runtime.scheduler, *runtime.worker,
            runtime.task_count, *args, runtime.context,
            runtime.stats, pmu_context, *ticket
        ) ? 1U : 0U;
    }
#endif
    return FinishCallbackSubmitBody<Ops, false, false>(
        runtime.scheduler, *runtime.worker, runtime.task_count,
        *args, runtime.context, runtime.stats,
        pmu_context, *ticket
    ) ? 1U : 0U;
}
#endif

template <TaskKind Kind, typename Ops, bool Profile, typename PmuContext>
PA_DEVICE bool SubmitCallbackTask(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, CoreRole role,
    uint32_t task_count,
    PaOrchestrationState &orch, TaskArgs &args, uint32_t batch,
    SubmitContext &context, LocalStats &stats, PmuContext &pmu_context
#if PTO_FDWIC_SHARED_MAP
    , uint64_t &shared_claim_attempts,
    const SharedPaBatchPlan &shared_batch_plan,
    uint32_t shared_task_offset,
    uint32_t shared_batch_count
#endif
) {
#if PTO_FDWIC_SHARED_MAP
    // 每个 task 都从通用 replay 前沿取得独立身份。不能利用当前 PA 的
    // SF→PV 邻接关系推导 task-id，否则动态 task 图或其他算子会失去
    // 自己的顺序锚点。
    BeginSharedCallbackSubmit(worker, context);
#else
    BeginCallbackSubmit(worker, context);
#endif
    const uint32_t task_id = static_cast<uint32_t>(context.task_id);
#if PTO_FDWIC_SHARED_MAP
    // RunSchedulerImpl 已在 replay 外读取同一份 batch 数，并把它作为
    // 本轮 replay 上界；沿用该 SSA 值，使每次 Submit 不再从只读
    // RunConfig 回取相同字段。
    SharedPaPlannedTask shared_planned_task{};
    if (batch >= shared_batch_count ||
        !SharedPaPlannedTaskAt(
            shared_batch_plan, shared_task_offset,
            shared_planned_task
        ) ||
        shared_planned_task.kind != Kind ||
        task_id !=
            shared_batch_plan.batch_start +
                shared_task_offset) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    const bool shared_is_last_submit =
        shared_planned_task.is_last_in_batch &&
        batch + 1U == shared_batch_count;
    const uint8_t shared_task_meta = EncodeSharedPaTaskMeta(
        Kind, shared_planned_task.group_index,
        shared_planned_task.has_following_group,
        shared_is_last_submit
    );
    if (shared_task_meta == 0) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
#endif
#if PA_BUILD_PERF_CLOCK
    // dist_submit_begin/BeginCallbackSubmit 已建立本次 task_id；首个
    // Submit 的 EfDrain 前只读一次性能时钟，不为其余 Submit 递增
    // 另一份观察计数。
    const uint64_t submit_begin = task_id == 0 ? Ops::PerfClockNow() : 0;
#elif PA_BUILD_SUBMIT_PMU
    const uint64_t submit_begin = task_id == 0 ? Ops::Now() : 0;
#else
    const uint64_t submit_begin = TraceTimestamp<Ops>(stats.trace, stats.result);
#endif
    if (task_id == 0) stats.result.submit_begin = submit_begin;

    const uint64_t efdrain_begin = submit_begin;
    BeginSubmitPmuPhase<SubmitPmuPhase::EfDrain, Ops>(pmu_context);
#if PTO_FDWIC_SHARED_MAP
    // BeginSharedCallbackSubmit 已经让 local_index 包含当前 task，
    // 而 stats.result.submits 只包含已经 Close 的前缀。执行发现
    // 必须使用后者，避免把尚未 Build 的当前 task 当成
    // EMPTY 并永久跳过。
    const uint32_t replay_closed_exclusive =
        static_cast<uint32_t>(stats.result.submits);
    const uint32_t worker_id =
        static_cast<uint32_t>(worker.core_idx);
    if (CrossCoreExecHasLocalProgressWork(
            state, worker_id, role,
            replay_closed_exclusive, stats
        )) {
        (void)ProgressCrossCoreExec<Ops>(
            state, worker, replay_closed_exclusive,
            /*production_closed=*/false,
            DrainPlace::EfDrain, stats
        );
    }
#else
    DrainReady<Ops>(state, worker, DrainPlace::EfDrain, stats);
#endif
    EndSubmitPmuPhase<SubmitPmuPhase::EfDrain, Ops>(pmu_context);
    const uint64_t efdrain_end = TraceTimestamp<Ops>(stats.trace, stats.result);
#if PTO_FDWIC_SHARED_MAP
    // shared 的 EfDrain 边界与现有父子记录严格重合：
    //   begin = Submit.start，end = Claim.start。
    // 设备不再为它单写一条 32B raw；converter/analyzer 用这两个既有
    // 端点精确还原泳道。Profile 聚合与 submit-pmu 的原始边界保持不变。
    AccumulatePhase<Profile>(
        stats.result, ProfilePhase::EfDrain,
        efdrain_begin, efdrain_end
    );
#else
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), -1,
        TracePhase::EfDrain, ProfilePhase::EfDrain, efdrain_begin, efdrain_end
    );
#endif

    const uint64_t claim_begin = efdrain_end;
    BeginSubmitPmuPhase<SubmitPmuPhase::Claim, Ops>(pmu_context);
    const ClaimOutcome claim =
        Claim<Ops>(
            state, static_cast<uint32_t>(worker.core_idx),
            role, task_id, Kind, stats
        );
#if !PTO_FDWIC_SHARED_MAP
    context.won = claim.won;
    context.kernel_id = claim.function_id;
#endif
    RecordClaimOutcome(
        stats, Kind, claim
#if PTO_FDWIC_SHARED_MAP
        , shared_claim_attempts
#endif
    );
    EndSubmitPmuPhase<SubmitPmuPhase::Claim, Ops>(pmu_context);
    const uint64_t claim_end = TraceTimestamp<Ops>(stats.trace, stats.result);
#if PTO_FDWIC_SHARED_MAP
    WriteSharedClaimTrace<Profile>(
        stats.trace, stats.result, task_id,
        claim_begin, claim_end, claim.won
    );
#else
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id),
        claim.function_id, TracePhase::Claim,
        ProfilePhase::Claim, claim_begin, claim_end,
        (claim.won ? kClaimWon : 0U) |
            (claim.attempted ? kClaimAttempted : 0U),
        Kind == TaskKind::Alloc ? 1U : 0U
    );
#endif

#if PTO_FDWIC_SHARED_MAP
    // fresh Output 的返回值是 task/slot 符号，不依赖哪个 worker 获胜。
    // 在跨 TU finish 前为所有 replay actor 建立同一句柄集，保证 loser
    // 返回后也能继续构造本核后续 task 的输入引用。
    if (!PrepareSharedTaskOutputs(
            context.shared_result, static_cast<int32_t>(task_id), Kind
        )) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
#endif
#if PTO_FDWIC_SHARED_MAP
    if (__builtin_expect(claim.won, 0)) {
        // shared loser 已在上方声明稳定 output symbol；它不需要构造本 task
        // 的 descriptor/scalar 参数，Alloc 也不例外。finish 的 loser
        // 分支只闭合边界，不读这里留下的上一 task args/won/function。
        // 这两个字段只由跨 TU winner finish 校验，因此也只由 winner 写。
        context.won = true;
        context.kernel_id = claim.function_id;
        PrepareSharedWinnerContext(
            worker, task_id, context
        );
        if (!BuildCallbackSubmitArgs<Kind>(orch, args, batch, stats)) {
            SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
            return false;
        }
    }
#else
    // private 保持所有 worker 对五个 task 的 eager 构参语义。
    if (!BuildCallbackSubmitArgs<Kind>(orch, args, batch, stats)) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
#endif
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
#if PTO_FDWIC_SHARED_MAP
    CompeteFirstSplitRuntimeState &split_runtime =
        Ops::CompeteFirstSplitState();
    if (!RecordSharedSplitReplayTask(split_runtime, task_id)) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    if (!claim.won) {
        return FinishSharedLoserSubmit<Ops, Profile>(
            state, stats, task_id, Kind,
            shared_is_last_submit, submit_begin
        );
    }
    // 只有 winner 才跨 TU；把 16B ticket 的物化下沉到本分支，loser
    // 继续用现有 SSA 标量完成同 TU 收尾。
    const CallbackSubmitTicket ticket{
        submit_begin,
        task_id,
        static_cast<int16_t>(claim.function_id),
        1,
        shared_task_meta
    };
    if (!ArmSharedSplitTicket(split_runtime, ticket)) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
#else
    const CallbackSubmitTicket ticket{
        submit_begin,
        task_id,
        static_cast<int16_t>(claim.function_id),
        static_cast<uint8_t>(claim.won ? 1 : 0),
        0
    };
#endif
    (void)state;
    (void)worker;
    (void)task_count;
    (void)context;
    (void)stats;
    (void)pmu_context;
    return Ops::FinishCompeteFirstCallback(&ticket, &args);
#else
#if PTO_FDWIC_SHARED_MAP
    if (!claim.won) {
        return FinishSharedLoserSubmit<Ops, Profile>(
            state, stats, task_id, Kind,
            shared_is_last_submit, submit_begin
        );
    }
    const CallbackSubmitTicket ticket{
        submit_begin,
        task_id,
        static_cast<int16_t>(claim.function_id),
        1,
        shared_task_meta
    };
#else
    const CallbackSubmitTicket ticket{
        submit_begin,
        task_id,
        static_cast<int16_t>(claim.function_id),
        static_cast<uint8_t>(claim.won ? 1 : 0),
        0
    };
#endif
    return FinishCallbackSubmitBody<Ops, Profile, false>(
        state, worker, task_count, args, context, stats, pmu_context, ticket
    );
#endif
}

#if PTO_FDWIC_SHARED_MAP
enum class SharedBuildTicketStatus : uint32_t {
    Acquired = 0,
    Exhausted = 1,
    Invalid = 2,
};

struct SharedBuildTicketResult {
    SharedBuildTicketStatus status;
    uint32_t task_id;
};

constexpr uint64_t kSharedBuildFatalPollPeriod = 16U;
static_assert(
    (kSharedBuildFatalPollPeriod &
     (kSharedBuildFatalPollPeriod - 1U)) == 0U,
    "shared Build fatal poll period must be a power of two"
);

PA_DEVICE bool SharedBuildFatalPollDue(uint64_t completed_builds) {
    return completed_builds != 0U &&
           (completed_builds &
            (kSharedBuildFatalPollPeriod - 1U)) == 0U;
}

template <typename Ops>
PA_DEVICE SharedBuildTicketResult TakeSharedBuildTicket(
    PA_GM SchedulerState *state, uint32_t task_count,
    LocalStats &stats
) {
    // 每个 worker 最终还会执行一次越界 FetchAdd 以独立确认发放结束，
    // 因而正常物理调用数是 task_count + kWorkers。返回旧值直接决定
    // task 身份，必须按 return-ready 语义消费。
    const int64_t observed = TraceAtomicFetchAdd<Ops>(
        stats.trace, stats.result, -1,
        AtomicSite::SharedBuildDispatchTicket,
        &state->build_dispatch.next_task.value, 1,
        /*result_used=*/true
    );
    ++stats.result.claim_attempts;
    if (task_count == 0 || task_count > kMaxTasks || observed < 0) {
        return SharedBuildTicketResult{
            SharedBuildTicketStatus::Invalid, UINT32_MAX
        };
    }
    if (observed >= static_cast<int64_t>(task_count)) {
        return SharedBuildTicketResult{
            SharedBuildTicketStatus::Exhausted, UINT32_MAX
        };
    }
    return SharedBuildTicketResult{
        SharedBuildTicketStatus::Acquired,
        static_cast<uint32_t>(observed)
    };
}

template <TaskKind Kind, typename Ops, bool Profile,
          typename PmuContext>
PA_DEVICE bool BuildDispatchedSharedTask(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker,
    uint32_t task_count, const SharedBuildDispatchTask &planned,
    PaOrchestrationState &orch, TaskArgs &args,
    SubmitContext &context, LocalStats &stats,
    PmuContext &pmu_context, uint64_t submit_begin,
    uint64_t claim_begin, uint64_t claim_end
) {
    const uint32_t task_id = planned.task_id;
    if (planned.meta.kind != Kind || task_id >= task_count ||
        planned.batch >= state->build_dispatch.batch_count ||
        !SharedPaDispatchRouteMatches(planned)) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }

    // ticket 已经唯一决定 Build owner；这里沿用 Claim/win 统计字段承载
    // “有效发放/成功取得”的既有 host 口径，但不再访问 Claim Tournament。
    ++stats.result.claim_wins;
    ++stats.result.wins[KindIndex(Kind)];
    ++worker.local_index;
    WriteSharedClaimTrace<Profile>(
        stats.trace, stats.result, task_id,
        claim_begin, claim_end, /*winner=*/true
    );
    AccumulatePhase<Profile>(
        stats.result, ProfilePhase::EfDrain,
        submit_begin, claim_begin
    );

    context.task_id = static_cast<int32_t>(task_id);
    context.shared_result.Reset(static_cast<int32_t>(task_id));
    if (!PrepareSharedTaskOutputs(
            context.shared_result, static_cast<int32_t>(task_id), Kind
        )) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    context.won = true;
    context.kernel_id = FunctionId(Kind);
    PrepareSharedWinnerContext(worker, task_id, context);

    // 每个 ticket 只恢复当前 task 所需的最小 PA 状态。输出句柄由
    // (producer task, slot) 确定，不要求本 worker 顺序经历前驱。
    ++stats.result.context_reads;
    if (!BindSharedPaTaskForRandomAccess(
            orch, planned.batch, planned.meta.batch_start,
            task_id, Kind, planned.meta.group_index
        ) ||
        !BuildCallbackSubmitArgs<Kind>(
            orch, args, planned.batch, stats
        )) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }

    const CallbackSubmitTicket ticket{
        submit_begin,
        task_id,
        static_cast<int16_t>(FunctionId(Kind)),
        1,
        state->build_dispatch.tasks[task_id].encoded_meta
    };
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
    CompeteFirstSplitRuntimeState &split_runtime =
        Ops::CompeteFirstSplitState();
    if (!ArmSharedSplitTicket(
            split_runtime, ticket, /*dispatched=*/true
        )) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    (void)pmu_context;
    return Ops::FinishCompeteFirstCallback(&ticket, &args);
#else
    return FinishCallbackSubmitBody<Ops, Profile, true>(
        state, worker, task_count, args, context, stats,
        pmu_context, ticket
    );
#endif
}

template <typename Ops, bool Profile, typename PmuContext>
PA_DEVICE bool DispatchOneSharedBuildTask(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker,
    CoreRole role, uint32_t task_count,
    PaOrchestrationState &orch, TaskArgs &args,
    SubmitContext &context, LocalStats &stats,
    PmuContext &pmu_context, bool execute_admission_open,
    bool &exhausted
) {
    exhausted = false;
#if PA_BUILD_PERF_CLOCK || PA_BUILD_SUBMIT_PMU
    const uint64_t submit_begin = 0;
#else
    const uint64_t submit_begin =
        TraceTimestamp<Ops>(stats.trace, stats.result);
#endif
    BeginSubmitPmuPhase<SubmitPmuPhase::EfDrain, Ops>(pmu_context);
    const uint32_t worker_id =
        static_cast<uint32_t>(worker.core_idx);
    if (execute_admission_open &&
        CrossCoreExecHasLocalProgressWork(
            state, worker_id, role, task_count, stats
        )) {
        (void)ProgressCrossCoreExec<Ops, true>(
            state, worker, task_count,
            /*production_closed=*/false,
            DrainPlace::EfDrain, stats
        );
    }
    EndSubmitPmuPhase<SubmitPmuPhase::EfDrain, Ops>(pmu_context);
    // central ticket 总量有界，任一 worker 最多再领取有限个 task 后就会
    // 进入 FinalDrain；strict-insert 等真正可能阻塞的慢路也保留自己的
    // fatal/watchdog。因此正常成功路径不需要在每张 ticket 前都让 96 核
    // 读取同一条 global-fatal cache line。启动后已经有一次立即观察，
    // 此处每完成 16 个本地 Build 再观察一次，把远端错误传播的额外工作量
    // 限制在每核最多 15 个 task，同时继续降低返回型 Atomic 竞争。
    const uint64_t completed_builds = stats.result.claim_wins;
    if (SharedBuildFatalPollDue(completed_builds) &&
        IsFatal<Ops>(state, stats)) {
        return false;
    }

    const uint64_t claim_begin =
        TraceTimestamp<Ops>(stats.trace, stats.result);
    BeginSubmitPmuPhase<SubmitPmuPhase::Claim, Ops>(pmu_context);
    const SharedBuildTicketResult ticket =
        TakeSharedBuildTicket<Ops>(state, task_count, stats);
    EndSubmitPmuPhase<SubmitPmuPhase::Claim, Ops>(pmu_context);
    const uint64_t claim_end =
        TraceTimestamp<Ops>(stats.trace, stats.result);
    if (ticket.status == SharedBuildTicketStatus::Exhausted) {
        exhausted = true;
        return true;
    }
    if (ticket.status != SharedBuildTicketStatus::Acquired) {
        SetFatal<Ops>(state, stats);
        return false;
    }

    SharedBuildDispatchTask planned{};
    if (!DecodeSharedBuildDispatchTask(
            state->build_dispatch, ticket.task_id, planned
        )) {
        SetFatal<Ops>(
            state, stats, static_cast<int32_t>(ticket.task_id)
        );
        return false;
    }
    switch (planned.meta.kind) {
        case TaskKind::Alloc:
            return BuildDispatchedSharedTask<
                TaskKind::Alloc, Ops, Profile
            >(
                state, worker, task_count, planned, orch, args,
                context, stats, pmu_context, submit_begin,
                claim_begin, claim_end
            );
        case TaskKind::Qk:
            return BuildDispatchedSharedTask<
                TaskKind::Qk, Ops, Profile
            >(
                state, worker, task_count, planned, orch, args,
                context, stats, pmu_context, submit_begin,
                claim_begin, claim_end
            );
        case TaskKind::Sf:
            return BuildDispatchedSharedTask<
                TaskKind::Sf, Ops, Profile
            >(
                state, worker, task_count, planned, orch, args,
                context, stats, pmu_context, submit_begin,
                claim_begin, claim_end
            );
        case TaskKind::Pv:
            return BuildDispatchedSharedTask<
                TaskKind::Pv, Ops, Profile
            >(
                state, worker, task_count, planned, orch, args,
                context, stats, pmu_context, submit_begin,
                claim_begin, claim_end
            );
        case TaskKind::Up:
            return BuildDispatchedSharedTask<
                TaskKind::Up, Ops, Profile
            >(
                state, worker, task_count, planned, orch, args,
                context, stats, pmu_context, submit_begin,
                claim_begin, claim_end
            );
        case TaskKind::Count:
            break;
    }
    SetFatal<Ops>(
        state, stats, static_cast<int32_t>(ticket.task_id)
    );
    return false;
}
#endif

PA_DEVICE uint32_t CountLiveMapEntries(PA_GM const TensorMap &map) {
    // AdvanceTensorMap 按 producer 精确维护 logical live_count；各桶 head
    // 允许惰性落后，不能再通过遍历物理槽推导逻辑存活数。
    return map.live_count;
}

#if PTO_FDWIC_SHARED_MAP
template <typename Ops>
PA_DEVICE uint32_t FinalizeSharedReplayTaskCount(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker,
    LocalStats &stats
) {
    uint32_t task_count = 0;
    if (worker.local_index < 0 ||
        static_cast<uint32_t>(worker.local_index) > kMaxTasks) {
        SetFatal<Ops>(state, stats, worker.local_index);
    } else {
        task_count = static_cast<uint32_t>(worker.local_index);
    }
    if (stats.declared_task_count != task_count) {
        SetFatal<Ops>(
            state, stats, static_cast<int32_t>(task_count)
        );
    }
    return task_count;
}
#endif

PA_DEVICE void LogicalTensorMapHashWord(
    uint64_t &hash, uint64_t value
) {
    for (uint32_t byte = 0; byte < 8; ++byte) {
        hash ^= (value >> (byte * 8U)) & 0xFFU;
        hash *= 1099511628211ULL;
    }
}

PA_DEVICE uint64_t PrivateLogicalTensorMapSignature(
    PA_GM const TensorMap &map
) {
    uint64_t hash = 1469598103934665603ULL;
    for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
        const uint64_t head = TensorMapBucketHead(map, bucket);
        const uint64_t tail = TensorMapBucketTail(map, bucket);
        for (uint64_t cursor = head; cursor < tail; ++cursor) {
            PA_GM const MapEntry &entry =
                map.entries[TensorMapSlotIndex(bucket, cursor)];
            if (entry.producer < map.alive_floor) {
                continue;
            }
            LogicalTensorMapHashWord(hash, bucket);
            LogicalTensorMapHashWord(hash, entry.buffer_addr);
            LogicalTensorMapHashWord(hash, entry.lo);
            LogicalTensorMapHashWord(hash, entry.hi);
            LogicalTensorMapHashWord(
                hash, static_cast<uint32_t>(entry.producer)
            );
        }
    }
    return hash;
}

template <typename Ops>
PA_DEVICE void PublishResult(PA_GM WorkerResult &destination, const WorkerResult &source) {
    // 每个 worker 只写自己独占、覆盖多条 cache line 的 WorkerResult 分区；逐字段
    // bypass 保证结果对 host 可见，而独立 sidecar 允许 D2H 只搬结果、不搬约 9 MiB WorkerState。
#define PA_PUBLISH_FIELD(field) Ops::Publish(&destination.field, source.field)
    PA_PUBLISH_FIELD(submit_begin);
    PA_PUBLISH_FIELD(submit_end);
    PA_PUBLISH_FIELD(finish_cycle);
    PA_PUBLISH_FIELD(checksum);
    PA_PUBLISH_FIELD(submits);
    PA_PUBLISH_FIELD(claim_attempts);
    PA_PUBLISH_FIELD(claim_wins);
    PA_PUBLISH_FIELD(heap_guards);
    PA_PUBLISH_FIELD(fanin_ready_loads);
    PA_PUBLISH_FIELD(completion_duplicates);
    PA_PUBLISH_FIELD(cas_retries);
    PA_PUBLISH_FIELD(joint_polls);
    for (uint32_t index = 0; index < static_cast<uint32_t>(TaskKind::Count); ++index) {
        Ops::Publish(&destination.wins[index], source.wins[index]);
    }
    for (uint32_t index = 0; index < 4; ++index) {
        Ops::Publish(&destination.kernel_counts[index], source.kernel_counts[index]);
        Ops::Publish(&destination.kernel_cycles[index], source.kernel_cycles[index]);
        Ops::Publish(&destination.kernel_min_cycles[index], source.kernel_min_cycles[index]);
        Ops::Publish(&destination.kernel_max_cycles[index], source.kernel_max_cycles[index]);
    }
    for (uint32_t index = 0; index < static_cast<uint32_t>(DrainPlace::Count); ++index) {
        Ops::Publish(&destination.placement[index], source.placement[index]);
    }
    for (uint32_t index = 0; index < static_cast<uint32_t>(ProfilePhase::Count); ++index) {
        Ops::Publish(&destination.phase_cycles[index], source.phase_cycles[index]);
        Ops::Publish(&destination.phase_calls[index], source.phase_calls[index]);
    }
    for (uint32_t index = 0; index < 2; ++index) {
        Ops::Publish(&destination.wait_events[index], source.wait_events[index]);
        Ops::Publish(&destination.wait_iterations[index], source.wait_iterations[index]);
    }
    PA_PUBLISH_FIELD(context_reads);
    PA_PUBLISH_FIELD(views_created);
    PA_PUBLISH_FIELD(dynamic_create_infos);
    PA_PUBLISH_FIELD(arg_resets);
    PA_PUBLISH_FIELD(tensor_args_added);
    PA_PUBLISH_FIELD(scalar_args_added);
    PA_PUBLISH_FIELD(materialized_outputs);
    PA_PUBLISH_FIELD(map_inserts);
    PA_PUBLISH_FIELD(map_lookups);
    PA_PUBLISH_FIELD(slot_tensor_copies);
    PA_PUBLISH_FIELD(slot_scalar_copies);
    PA_PUBLISH_FIELD(fanin_edges);
    PA_PUBLISH_FIELD(final_heap_next);
    PA_PUBLISH_FIELD(map_high_water);
    PA_PUBLISH_FIELD(map_alive_floor);
    PA_PUBLISH_FIELD(map_cleaned_upto);
    PA_PUBLISH_FIELD(map_live_entries);
    PA_PUBLISH_FIELD(worker_id);
    PA_PUBLISH_FIELD(role);
    PA_PUBLISH_FIELD(max_occupied);
    PA_PUBLISH_FIELD(final_occupied);
    PA_PUBLISH_FIELD(fanin_not_ready_loads);
    PA_PUBLISH_FIELD(frontier_initial_loads);
    PA_PUBLISH_FIELD(frontier_updates);
    PA_PUBLISH_FIELD(frontier_terminal_loads);
    PA_PUBLISH_FIELD(atomic_trace_calls);
    PA_PUBLISH_FIELD(startup_barrier_begin);
    PA_PUBLISH_FIELD(startup_barrier_end);
    PA_PUBLISH_FIELD(final_barrier_begin);
    PA_PUBLISH_FIELD(final_barrier_release);
    PA_PUBLISH_FIELD(final_barrier_end);
    PA_PUBLISH_FIELD(dependency_signature);
    PA_PUBLISH_FIELD(shared_symbol_input_loads);
    PA_PUBLISH_FIELD(shared_symbol_inout_commits);
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
    PA_PUBLISH_FIELD(compete_first_split_caller_state_address);
    PA_PUBLISH_FIELD(compete_first_split_finish_state_address);
    PA_PUBLISH_FIELD(compete_first_split_finish_calls);
    PA_PUBLISH_FIELD(compete_first_split_protocol_errors);
    PA_PUBLISH_FIELD(compete_first_split_state_cookie);
    PA_PUBLISH_FIELD(compete_first_split_task_id_sum);
    PA_PUBLISH_FIELD(compete_first_split_owner_worker_id);
    PA_PUBLISH_FIELD(compete_first_split_reserved);
#endif
#undef PA_PUBLISH_FIELD
    Ops::StoreBarrier();
}

template <typename Ops, bool Profile>
PA_DEVICE void RunSchedulerImpl(PA_GM SchedulerState *state, uint32_t worker_id, CoreRole role) {
    // 一个入口实例只拥有 state->workers[worker_id] 的私有 map/ring/payload；cursor、task cell 和屏障为跨核共享区。
    if (worker_id >= kWorkers) {
        return;
    }
    // RunConfig、PMU 配置与 winner workload 连续占据三条独立 cache line。
    // 在解释任何可能随 TensorMap 模式变化的 WorkerState 之前，先失效 host
    // 写入的控制区并核对稳定构建身份；混合 host/kernel 会置 fatal 后退出，
    // 不允许继续用错误 sizeof 或模式解释 GM。
    constexpr uint64_t kStartupConfigBytes =
        sizeof(state->config) + sizeof(state->pmu_probe) +
        sizeof(state->winner_workload);
    uint64_t startup_dcci_begin = 0;
    uint64_t startup_dcci_end = 0;
    CapturePreAttachDcciInvalidate<Ops>(
        &state->config,
        kStartupConfigBytes,
        startup_dcci_begin, startup_dcci_end
    );
    const bool build_identity_matches =
        state->config.build_identity_magic == kBuildIdentityMagic &&
        state->config.build_identity_abi_version == kBuildIdentityAbiVersion &&
        state->config.tensor_map_mode == static_cast<uint32_t>(kCompiledTensorMapMode) &&
        state->config.scheduler_state_size == static_cast<uint32_t>(sizeof(SchedulerState)) &&
        state->pmu_probe.build_variant == kCompiledBuildVariant;
    if (!build_identity_matches) {
        (void)PreAttachAtomicExchange<Ops>(
            &state->fatal.value, static_cast<int32_t>(1)
        );
        return;
    }
    PA_GM WorkerState &worker = state->workers[worker_id];
    worker.role = role;
    worker.core_idx = static_cast<int32_t>(worker_id);
    // standalone 使用连续 worker 编号：AIC 为 0..31；AIV 为 32..95。
    // 每个物理 block b 对应 AIC(b, lane0)、AIV(32+2b, lane1)、AIV(33+2b, lane2)。
    if (role == CoreRole::Aic) {
        worker.block_id = static_cast<int32_t>(worker_id);
        worker.lane = 0;
    } else {
        const uint32_t vector_id = worker_id - kAicWorkers;
        worker.block_id = static_cast<int32_t>(vector_id / 2);
        worker.lane = static_cast<int32_t>(1 + vector_id % 2);
    }
    worker.sub_block_id = worker.lane == 2 ? 1 : 0;
    worker.local_index = 0;
    worker.heap_next = 0;
#if !PTO_FDWIC_SHARED_MAP
    ResetTensorMap(worker.map);
#endif
    worker.occupied_count = 0;
    worker.owned_total = 0;
    worker.swimlane_last_cycle = 0;
#if PTO_FDWIC_SHARED_MAP
    // 全部 ExecutionToken 都是 worker 私有的普通 GM 状态，不由其他
    // Scalar 读取。
    // A5 Scalar 间没有 cache coherence，host H2D 写入的 IDLE 不能替代
    // 本核初始化：同一物理 Scalar 仍可能缓存上一 kernel 的 Waiting/Faulted
    // 控制字。每轮由 owner Scalar 覆盖自己的完整控制字段，后续也只由
    // 该 Scalar 消费；跨核可见性只通过 exec cell 的 atomic + DCCI 协议。
    for (uint32_t token_slot = 0;
         token_slot < cross_core_dag::kExecTokensPerWorker;
         ++token_slot) {
        cross_core_dag::ResetExecutionToken(
            state->exec_tokens[worker_id][token_slot]
        );
    }
#endif
    for (uint32_t index = 0; index < kPrivateSlots; ++index) {
        worker.slots[index].occupied = false;
        worker.slots[index].built = false;
    }

#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
    CompeteFirstSplitRuntimeState &split_runtime = Ops::CompeteFirstSplitState();
    split_runtime.context = SubmitContext{};
    split_runtime.stats = LocalStats{};
    split_runtime.scheduler = state;
    split_runtime.worker = &worker;
    split_runtime.task_count = 0;
    split_runtime.worker_id = worker_id;
    split_runtime.caller_state_address = reinterpret_cast<uint64_t>(&split_runtime);
    split_runtime.finish_state_address = 0;
    split_runtime.finish_calls = 0;
    split_runtime.protocol_errors = 0;
    split_runtime.state_cookie = CompeteFirstSplitStateCookie(worker_id, role);
    split_runtime.task_id_sum = 0;
    split_runtime.owner_worker_id = worker_id;
    split_runtime.reserved = 0;
    LocalStats &stats = split_runtime.stats;
#else
    LocalStats stats{};
#endif
    stats.result.worker_id = worker_id;
    stats.result.role = static_cast<uint32_t>(role);
    stats.result.checksum = 0;
    stats.trace = AttachTrace<Ops>(state, worker, worker_id);
#if PA_BUILD_ATOMIC_SWIMLANE
    // 完整泳道产物把阶段、Atomic 与 DCCI 视为同一构建合同。入口一次性
    // 验证 host 配置与 raw header；成功后各条记录可省掉重复附着判断。
    if (stats.trace.core == nullptr ||
        stats.trace.records == nullptr ||
        stats.trace.capacity != kTraceRecordsPerCore ||
        !stats.trace.atomics_enabled) {
        (void)PreAttachAtomicExchange<Ops>(
            &state->fatal.value, static_cast<int32_t>(1)
        );
        return;
    }
#endif
#if !PA_BUILD_TRACE_FREE
    if (TraceStorageAttached(stats.trace)) {
        const uint32_t startup_dcci_lines =
            DcciRegionCacheLineCount(
                &state->config, kStartupConfigBytes
            );
        if (startup_dcci_lines != 0) {
            (void)WriteDcciTrace(
                stats.trace, -1, -1,
                DcciSite::StartupConfigInvalidate,
                DcciOp::Invalidate,
                /*trailing_dsb=*/true,
                startup_dcci_lines,
                startup_dcci_begin, startup_dcci_end
            );
        } else {
            stats.trace.dcci_counter_overflow = true;
        }
    }
#else
    (void)startup_dcci_begin;
    (void)startup_dcci_end;
#endif

    // 每个 worker 都先发布一次启动到达标记。private replay 继续等待全员到齐；
    // shared cross-core 依靠唯一 Build/Execute ticket、严格 metadata writer 插入链
    // 和最终 execution drain 闭合，不把启动对齐当成调度正确性的前置条件。
#if PA_BUILD_PERF_CLOCK
    // perf-clock 的第一次权威读数从 startup increment 前开始，
    // 与 FinalDrain 后的第二次读数组成完整端到端边界。复用
    // submit_begin 字段避免增加 WorkerResult ABI；该构建不输出
    // Submit-only 边界。启动 watchdog 仍用独立时钟建立超时起点。
    stats.result.submit_begin = Ops::PerfClockNow();
    stats.result.startup_barrier_begin = 0;
#else
    stats.result.startup_barrier_begin = Ops::Now();
#endif
    TraceAtomicFetchAdd<Ops>(
        stats.trace, stats.result, -1, AtomicSite::StartupIncrement,
        &state->started_count.value, 1
    );
#if !PTO_FDWIC_SHARED_MAP
    const uint64_t start_wait = Ops::Now();
    uint32_t start_polls = 0;
    const uint32_t startup_poll_region = AtomicPollRegionBegin<Ops>(
        stats.trace, stats.result,
        TraceAtomicPollBatchMask(AtomicSite::StartupPoll) |
            TraceAtomicPollBatchMask(AtomicSite::FatalPoll)
    );
    // 全员到齐或任一核发布 fatal 即退出启动等待；watchdog 防止缺失参与者造成永久挂死。
    while (LoadLine<Ops>(state->started_count, stats, AtomicSite::StartupPoll) <
               static_cast<int64_t>(state->config.workers) &&
           !IsFatal<Ops>(state, stats)) {
        Ops::SpinHint();
        if (WatchdogExpired<Ops>(state, stats, start_wait, start_polls)) {
            break;
        }
    }
    AtomicPollRegionEnd<Ops>(stats.trace, stats.result, startup_poll_region);
#else
    // shared cross-core 的 Build/Execute 均由全局唯一 ticket 动态发放，
    // metadata side effect 又由严格 writer 插入链排序；最终 execution drain
    // 仍要求全部配置 worker 到达。因此 started_count 继续记录真实参与者，
    // 早到 worker 不必原地等待，可先领取 Build；Execute ticket 会在后面的
    // 调度循环确认全员到达后开放，避免少数早到核囤积执行所有权。
    // 这里不读取任何算子 task kind、DAG、batch 或 tensor shape。
#endif
#if PA_BUILD_PERF_CLOCK
    stats.result.startup_barrier_end = 0;
#else
    stats.result.startup_barrier_end = Ops::Now();
#endif

    const uint32_t batches = state->config.batches;
#if PTO_FDWIC_SHARED_MAP
    // host 已从 context_lens 独立构建并发布不可变计划。device 只接受与
    // 本次 RunConfig 一致的 header；Build 与 AIC/AIV Execute 三条中央
    // ticket 流共用这一 task_count，不再由任一 worker 的本地 replay
    // 前缀反推。
    const uint32_t task_count = state->build_dispatch.task_count;
    // header/plan 在 launch 前一次发布、运行期只读。摘要完整性只需每个
    // worker 在领取首张 Build ticket 前校验一次，不能在每个逻辑 task 的
    // Finish 中重复读取同一条 GM cache line。
    if (task_count == 0 || task_count > kMaxTasks ||
        !ValidateSharedMetadataWriterSummary(
            state->build_dispatch
        ) ||
        state->build_dispatch.executable_task_count > task_count ||
        state->exec_dispatch.aic_task_count > task_count ||
        state->exec_dispatch.aiv_task_count > task_count ||
        state->exec_dispatch.aic_task_count +
                state->exec_dispatch.aiv_task_count !=
            state->build_dispatch.executable_task_count ||
        state->build_dispatch.batch_count != batches ||
        batches == 0 || batches > kMaxBatches) {
        SetFatal<Ops>(state, stats);
    }
#else
    const uint32_t task_count = batches * kTasksPerBatch;
#endif
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
    split_runtime.task_count = task_count;
#endif
    PaOrchestrationState orchestration;
    TaskArgs args;
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
    SubmitContext &context = split_runtime.context;
#else
    SubmitContext context;
#endif
    uint64_t orchestration_begin = 0;
    uint64_t orchestration_end = 0;
#if PTO_FDWIC_SHARED_MAP
    // 启动偏斜期间允许中央 Build ticket 立即推进，但在全部配置 worker
    // 发布到达前不发放 owner-local Execute token，避免少数早到核囤积
    // 尚未 BUILT 的执行所有权。该门控只依赖通用参与者合同。
    bool execute_admission_open = false;
#endif
    // 复用入口已有的 terminal 读取结果。若本核已经发现不可变计划或启动
    // 协议错误，FinalDrain 不得在错相 fatal 观察到来前发放 Execute ticket。
    const bool scheduler_entry_ok = !IsFatal<Ops>(state, stats);
    if (scheduler_entry_ok) {
        // private 每批固定回放 Alloc/QK/SF/PV/UP；shared 直接消费 host
        // 按 context_len 发布的 1+4N immutable plan，由中央 ticket 选择
        // Build owner；Execute owner 由独立的 AIC/AIV 角色 ticket 决定。
        // CCEC 可在这里开启本 worker 私有 PMU 窗口；CPU/AscendC 适配层是空实现。
        // 窗口覆盖本 worker 的全部调度期：从 orchestration 初始化前开始，
        // 依次包含 EfDrain、Claim、当前模式实际执行的参数构造与后续 Submit
        // 阶段，到末次 Submit 返回。private 为全员 eager；shared 五类
        // task 都只由 ticket owner 构参。
        // 它与全局“首 Submit.begin～末 Submit.end”口径接近但不相同，host sidecar
        // 必须按 per-worker 累计解释。泳道父边界在 PMU-only 构建中会被编译为空，
        // 不应污染 Submit 取数。
        auto pmu_context = Ops::PmuWindowStart(state, worker_id);
        orchestration_begin = TraceTimestamp<Ops>(stats.trace, stats.result);
        InitPaOrchestration(orchestration, batches, &state->context_lens[0]);
#if PTO_FDWIC_SHARED_MAP
        // Build 中央单调 ticket 只让每个逻辑 task 构建一次。每个 Scalar
        // 在领取前先推进自己已经持有的 Execute token；取得 Build ticket
        // 后可乱序构参，但 Finish 内仍由 insert-completion[N-1] 串行提交
        // TensorMap 元数据。
#if PA_BUILD_PERF_CLOCK
        // 起点已在 startup increment 之前读取；这里不得覆盖。
#elif PA_BUILD_SUBMIT_PMU
        stats.result.submit_begin = Ops::Now();
#else
        stats.result.submit_begin = orchestration_begin;
#endif
        bool dispatch_exhausted = false;
        bool crossed_first_build_dispatch = false;
        while (!dispatch_exhausted) {
            // 不在所有 worker 刚完成 startup increment 后立刻同址读取
            // started_count。先经过一次通用 Build dispatch 边界，再借实际
            // dispatch 时延自然打散首次观察；Execute 仍只有在确证全员到达
            // 后才开放。这里不借用 PA/观测统计字段，终端 ticket 也构成边界。
            if (!execute_admission_open && crossed_first_build_dispatch) {
                execute_admission_open =
                    LoadLine<Ops>(
                        state->started_count, stats,
                        AtomicSite::StartupPoll
                    ) >= static_cast<int64_t>(state->config.workers);
            }
            const bool dispatch_ok =
                DispatchOneSharedBuildTask<Ops, Profile>(
                    state, worker, role, task_count,
                    orchestration, args, context, stats,
                    pmu_context, execute_admission_open,
                    dispatch_exhausted
                );
            if (!dispatch_ok) {
                break;
            }
            crossed_first_build_dispatch = true;
        }
#if PA_BUILD_PERF_CLOCK
        // 唯一性能终点在 FinalDrain 排空后读取；不保留 Submit-only
        // 性能边界。
        stats.result.submit_end = 0;
#elif PA_BUILD_SUBMIT_PMU
        stats.result.submit_end = Ops::Now();
#else
        stats.result.submit_end =
            TraceTimestamp<Ops>(stats.trace, stats.result);
#endif
#else
        for (uint32_t batch = 0; batch < batches; ++batch) {
            BeginPaBatchForCallback(orchestration, batch);
            ++stats.result.context_reads;
            if (!SubmitCallbackTask<TaskKind::Alloc, Ops, Profile>(
                    state, worker, role, task_count, orchestration, args,
                    batch, context, stats,
                    pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(
                orchestration, TaskKind::Alloc, OrchestrationOutputs(context)
            );

            PreparePaBlockGroup(orchestration, 0);
            if (!SubmitCallbackTask<TaskKind::Qk, Ops, Profile>(
                    state, worker, role, task_count, orchestration, args,
                    batch, context, stats,
                    pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(
                orchestration, TaskKind::Qk, OrchestrationOutputs(context)
            );

            if (!SubmitCallbackTask<TaskKind::Sf, Ops, Profile>(
                    state, worker, role, task_count, orchestration, args,
                    batch, context, stats,
                    pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(
                orchestration, TaskKind::Sf, OrchestrationOutputs(context)
            );

            if (!SubmitCallbackTask<TaskKind::Pv, Ops, Profile>(
                    state, worker, role, task_count, orchestration, args,
                    batch, context, stats,
                    pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(
                orchestration, TaskKind::Pv, OrchestrationOutputs(context)
            );

            if (!SubmitCallbackTask<TaskKind::Up, Ops, Profile>(
                    state, worker, role, task_count, orchestration, args,
                    batch, context, stats,
                    pmu_context
                )) {
                break;
            }
        }
#endif
        Ops::PmuWindowStop(state, worker_id, pmu_context);
        orchestration_end = TraceTimestamp<Ops>(stats.trace, stats.result);
    }

    // shared 的中央 ticket 循环保证每个 worker 只有在拿到越界 ticket
    // 后才进入这里；执行排空到达又要求本核取完本角色全部 ticket、所有 token
    // 完整复位。因而 shared 可以直接用更强的执行排空汇合收口，不再先做
    // 一遍 ReplayDone 树。private 仍保留原 replay barrier 协议。
    // 成功路径复用 orchestration end 作为 final drain start，使两个业务父区间
    // 首尾相接；父记录延后到 final drain 结束再写，避免记录自身落进任一业务 span。
    const uint64_t final_drain_begin = orchestration_end != 0
        ? orchestration_end
        : TraceTimestamp<Ops>(stats.trace, stats.result);
#if PA_BUILD_PERF_CLOCK
    stats.result.final_barrier_begin = 0;
#else
    stats.result.final_barrier_begin = Ops::Now();
#endif
#if PTO_FDWIC_SHARED_MAP
    // WorkerResult 的既有字段继续承载 FinalDrain 边界，但 shared 已没有
    // 独立 replay barrier；release 与 begin 相等，host 必须把跨核差值
    // 解释为 dispatch 退出偏斜，不能再称为 barrier 等待。
#if !PA_BUILD_PERF_CLOCK
    stats.result.final_barrier_release =
        stats.result.final_barrier_begin;
#endif
    const uint32_t final_poll_region = AtomicPollRegionBegin<Ops>(
        stats.trace, stats.result,
        TraceAtomicPollBatchMask(AtomicSite::FaninFlagLoad) |
            TraceAtomicPollBatchMask(AtomicSite::FatalPoll) |
            TraceAtomicPollBatchMask(
                AtomicSite::SharedExecDrainArrivalPoll
            )
    );
    bool cross_core_exec_ok = scheduler_entry_ok;
    bool cross_core_drain_arrived = false;
    bool cross_core_drain_closed = false;
    // 各 worker 使用不同的轮询相位，避免正常路径进入 FinalDrain 时同时
    // 对同一 global-fatal 地址发起返回型 Atomic。任一仍滞留在 drain 中的
    // worker 最迟在 255 轮内完成首次观察，之后仍保持每 256 轮一次的既有
    // 有界错误收敛；这里不依赖 task kind、DAG 或具体 worker 总数。
    uint32_t final_fatal_poll_iterations = worker_id & 255U;
    uint32_t final_watchdog_polls = 0;
    const uint64_t final_watchdog_begin = Ops::Now();
    while (true) {
        if (!execute_admission_open) {
            execute_admission_open =
                LoadLine<Ops>(
                    state->started_count, stats,
                    AtomicSite::StartupPoll
                ) >= static_cast<int64_t>(state->config.workers);
        }
        // 错误只需最终可见，不为成功路径的每次 progress 付出同地址返回型
        // Atomic。各 worker 的首次检查按 owner-local 相位错开，随后每 256
        // 轮检查一次；命中后本核 token 进入 Faulted 并结束。正常成功路径
        // 由执行排空到达证明完整性。
        if (cross_core_exec_ok &&
            ((final_fatal_poll_iterations++ & 255U) == 0U) &&
            ObserveCrossCoreFinalDrainFatal<Ops>(
                state, worker_id, stats
            )) {
            cross_core_exec_ok = false;
        }
        const uint32_t freed =
            cross_core_exec_ok && execute_admission_open
            ? ProgressCrossCoreExec<Ops, true>(
                  state, worker, task_count,
                  /*production_closed=*/false,
                  DrainPlace::FinalDrain, stats
              )
            : 0;

        // 不依赖 replay barrier 宣告 production_closed：尚未发布的
        // EMPTY/BUILDING 候选会保留 scanner 队头；对应 builder 自己也要
        // 完成越界 ticket 后才能到达。只有 scanner/token 全部排空的核
        // 才能发布一次携带 owner-local completion 数的到达。
        if (cross_core_exec_ok && execute_admission_open &&
            CrossCoreExecWorkerDrained<true>(
                state, worker, task_count, stats
            )) {
            cross_core_exec_ok =
                ProgressCrossCoreExecDrainClosure<Ops, true>(
                    state, worker, task_count, stats,
                    cross_core_drain_arrived,
                    cross_core_drain_closed
                );
        }
        if (!cross_core_exec_ok || cross_core_drain_closed) {
            break;
        }
        if (freed == 0) {
            // 删除 replay barrier 后，缺失 BUILT/到达不能退化成永久自旋。
            // 以进入 FinalDrain 的独立时钟为起点，每 1024 次无进展才读
            // 一次计数器；正常毫秒级运行只付出低频检查，异常在既有
            // 2 秒门限内终止，也不会误伤前面合法的长 Build/Execute。
            if (WatchdogExpired<Ops>(
                    state, stats, final_watchdog_begin,
                    final_watchdog_polls
                )) {
                cross_core_exec_ok = false;
                break;
            }
            Ops::SpinHint();
        }
    }
#else
    const auto final_barrier_shape = static_cast<FinalBarrierShape>(state->config.final_barrier_shape);
    const uint32_t final_two_level_groups = TwoLevelFinalBarrierGroupCount(final_barrier_shape);
    const bool hierarchical_final_barrier =
        final_barrier_shape == FinalBarrierShape::ThreeLevel6x4x4 || final_two_level_groups != 0;
    if (hierarchical_final_barrier) {
        ArriveHierarchicalFinalBarrier<Ops>(
            state->final_barrier, final_barrier_shape, worker, stats, AtomicSite::ReplayDoneIncrement
        );
    } else {
        PublishFinalBarrierLine<Ops>(state->replay_done, stats, AtomicSite::ReplayDoneIncrement);
    }
    const uint32_t final_poll_region = AtomicPollRegionBegin<Ops>(
        stats.trace, stats.result,
        TraceAtomicPollBatchMask(AtomicSite::ReplayDonePoll) |
            TraceAtomicPollBatchMask(AtomicSite::FaninFlagLoad) |
            TraceAtomicPollBatchMask(AtomicSite::FatalPoll)
    );
    bool leaf_forwarded = false;
    bool middle_forwarded = false;
    bool root_released = false;
    bool middle_released = false;
    bool leaf_released = false;
    bool global_release_observed = false;
    while (true) {
        const uint32_t freed =
            DrainReady<Ops>(state, worker, DrainPlace::FinalDrain, stats);
        const bool all_replayed = hierarchical_final_barrier ?
                                      ProgressHierarchicalFinalBarrier<Ops>(
                                          state->final_barrier, final_barrier_shape, worker, stats,
                                          AtomicSite::ReplayDoneIncrement, AtomicSite::ReplayDonePoll, leaf_forwarded,
                                          middle_forwarded, root_released, middle_released, leaf_released
                                      ) :
                                      LoadLine<Ops>(state->replay_done, stats, AtomicSite::ReplayDonePoll) >=
                                          static_cast<int64_t>(state->config.workers);
        if (all_replayed && !global_release_observed) {
#if !PA_BUILD_PERF_CLOCK
            stats.result.final_barrier_release = Ops::Now();
#endif
            global_release_observed = true;
        }
        // 必须同时满足“无人再生产新 slot”和“本核旧 slot 全部完成”，否则继续帮助系统推进 completion。
        if (global_release_observed && worker.occupied_count == 0) {
            break;
        }
        if (freed == 0) {
            Ops::SpinHint();
        }
    }
#endif
    AtomicPollRegionEnd<Ops>(stats.trace, stats.result, final_poll_region);
#if PA_BUILD_PERF_CLOCK
    // 权威性能终点在全部 execution cell 排空、全部 owner-local token
    // 复位且全局 drain 发布后读取。复用 WorkerResult::finish_cycle，
    // 不增加状态字段或逐 task 记录。
    const uint64_t final_drain_end_clock =
        ReadPerfFinalDrainBoundary<Ops>();
    stats.result.final_barrier_end = 0;
#else
    stats.result.final_barrier_end = Ops::Now();
#endif
    const uint64_t final_drain_end = TraceTimestamp<Ops>(stats.trace, stats.result);
    if (orchestration_begin != 0 && orchestration_end >= orchestration_begin) {
        WriteTrace<false>(
            stats.trace, stats.result, -1, -1, TracePhase::OrchestrationReplay,
            ProfilePhase::Orchestration, orchestration_begin, orchestration_end
        );
    }
    WriteTrace<false>(
        stats.trace, stats.result, -1, -1, TracePhase::FinalDrain,
        ProfilePhase::ReplayTail, final_drain_begin, final_drain_end
    );

#if !PA_BUILD_TRACE_FREE
    if (AtomicSwimlaneEnabled(stats.trace)) {
        // 两条基线都放在最终 drain 之后。第一条量连续
        // SYS_CNT，第二条量返回依赖钩子的固定成本；它们只描述计时底噪，不能
        // 从每条 atomic 中机械相减后宣称得到跨核全局可见性延迟。
        const uint64_t clock_begin = Ops::Now();
        const uint64_t clock_end = Ops::Now();
        WriteTrace<false>(
            stats.trace, stats.result, -1, -1, TracePhase::ClockBaseline,
            ProfilePhase::ReplayTail, clock_begin, clock_end
        );
        const uint64_t dependency_begin = Ops::Now();
        const uint64_t dependency_end = Ops::NowAfterAtomicResult(
            static_cast<uint64_t>(worker_id)
        );
        WriteTrace<false>(
            stats.trace, stats.result, -1, -1, TracePhase::ClockBaseline,
            ProfilePhase::ReplayTail, dependency_begin, dependency_end,
            kClockAtomicDependency |
                (Ops::kAtomicReturnReadyObserved ? kClockAtomicDependencyApplied : 0U)
        );
    }
#endif

#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
#if !PTO_FDWIC_SHARED_MAP
    const uint64_t expected_task_id_sum =
        static_cast<uint64_t>(task_count) * (task_count - 1U) / 2U;
#endif
    // terminal fatal 可能在某个 worker 进入首个 Submit 之前已经可见。
    // 该 worker 合法地没有 finish 调用，finish TU 地址也尚未回写；不能
    // 把这种零回放收敛误报成 split 状态错配。只要开始过任一 Submit，
    // 仍严格要求 caller/finish 是同一个 TLS runtime。
#if PTO_FDWIC_SHARED_MAP
    // shared 中每个有效 ticket 都跨 TU 完成一次 winner Finish；某个
    // worker 可以没有取得任何 task，此时 finish 地址合法地保持 0。
    const bool finish_state_matches =
        split_runtime.finish_calls == 0
            ? split_runtime.finish_state_address == 0
            : split_runtime.finish_state_address ==
                  split_runtime.caller_state_address;
    const uint64_t expected_finish_calls =
        stats.result.claim_wins;
    const bool split_task_accounting_ok =
        split_runtime.finish_calls == stats.result.submits &&
        stats.result.submits ==
            static_cast<uint64_t>(worker.local_index);
#else
    const bool finish_state_matches =
        task_count == 0
            // PA_ATOMIC_DCCI_SOURCE_EXEMPT: test-only - private split-finish 诊断不属于 standalone shared scheduler
            ? Ops::Load(&state->fatal.value) != 0 &&
                  split_runtime.finish_calls == 0 &&
                  split_runtime.finish_state_address == 0
            : split_runtime.finish_state_address ==
                  split_runtime.caller_state_address;
    const uint64_t expected_finish_calls = task_count;
    const bool split_task_accounting_ok =
        split_runtime.task_id_sum == expected_task_id_sum;
#endif
    const bool split_protocol_ok =
        split_runtime.scheduler == state && split_runtime.worker == &worker &&
        split_runtime.task_count == task_count && split_runtime.worker_id == worker_id &&
        split_runtime.owner_worker_id == worker_id && split_runtime.caller_state_address != 0 &&
        finish_state_matches &&
        split_runtime.finish_calls == expected_finish_calls &&
        split_task_accounting_ok &&
        split_runtime.state_cookie == CompeteFirstSplitStateCookie(worker_id, role) &&
        split_runtime.reserved == 0;
    if (!split_protocol_ok) {
        ++split_runtime.protocol_errors;
        SetFatal<Ops>(state, stats);
    }
    stats.result.compete_first_split_caller_state_address = split_runtime.caller_state_address;
    stats.result.compete_first_split_finish_state_address = split_runtime.finish_state_address;
    stats.result.compete_first_split_finish_calls = split_runtime.finish_calls;
    stats.result.compete_first_split_protocol_errors = split_runtime.protocol_errors;
    stats.result.compete_first_split_state_cookie = split_runtime.state_cookie;
    stats.result.compete_first_split_task_id_sum = split_runtime.task_id_sum;
    stats.result.compete_first_split_owner_worker_id = split_runtime.owner_worker_id;
    stats.result.compete_first_split_reserved = split_runtime.reserved;
#endif

    // PA writes swimlane records through the ordinary GM cache and explicitly
    // cleans each worker's record range before the kernel finishes.
    FlushTraceCore<Ops>(
        stats.trace, stats.result, task_count
    );
#if PA_BUILD_PERF_CLOCK
    stats.result.finish_cycle = final_drain_end_clock;
#else
    stats.result.finish_cycle = Ops::Now();
#endif
    stats.result.max_occupied = stats.max_occupied;
#if PTO_FDWIC_SHARED_MAP
    // CCEC 关闭 kernel-end 自动 DCCI，host 直接读取 token.control 可能只
    // 看到初始化时的 IDLE。final_occupied 在本核最终检查后经 bypass
    // PublishResult 导出：0 表示 scanner 已封口、token 全字段复位且本核
    // 已发布携带本核完成数的 arrival；root 另外通过 16 组到达与完成数
    // 汇总完成 device 级收口。非零表示该轮不能宣称 executor 排空。
    stats.result.final_occupied =
        cross_core_drain_closed &&
                CrossCoreExecWorkerDrained<true>(
                    state, worker, task_count, stats
                ) &&
                CrossCoreExecAllTokensFullyReset(state, worker_id)
            ? 0
            : 1;
#else
    stats.result.final_occupied = worker.occupied_count;
#endif
    stats.result.final_heap_next = worker.heap_next;
#if PTO_FDWIC_SHARED_MAP
    // shared 后端没有每核 map 控制字；这里发布统一逻辑窗口摘要，实际
    // bucket/head/tail/seq/payload 由 host 回读唯一 sidecar 后逐槽校验。
    // shared fresh Output 都由 shared-output table 直接寻址，manual_dep
    // 的 output_view 也不进入自动 region hazard；因此 Case1 region ring
    // 严格为空；四个摘要都保持零，不能再用跨模式签名比较所需的逻辑
    // floor 冒充 sidecar 实际发生过 ordered reclaim。
    stats.result.map_high_water = 0;
    stats.result.map_alive_floor = 0;
    stats.result.map_cleaned_upto = 0;
    stats.result.map_live_entries = 0;
#else
    stats.result.map_high_water = static_cast<uint32_t>(worker.map.high_water);
    stats.result.map_alive_floor = static_cast<uint32_t>(worker.map.alive_floor);
    stats.result.map_cleaned_upto = static_cast<uint32_t>(worker.map.cleaned_upto);
    stats.result.map_live_entries = CountLiveMapEntries(worker.map);
    stats.result.checksum =
        PrivateLogicalTensorMapSignature(worker.map);
#endif
    PublishResult<Ops>(state->results[worker_id], stats.result);
}

template <typename Ops>
PA_DEVICE void RunScheduler(PA_GM SchedulerState *state, uint32_t worker_id, CoreRole role) {
    // 两个正式 CCEC 构建都不再携带旧 phase-profile 模板副本：swimlane 用
    // records 表达阶段，submit-pmu 使用独立 PMU 边界。其他后端暂时保留原
    // 运行时入口，保证公共 standalone 的 CPU/AscendC 回归不被 CCEC 构建切分影响。
#if PA_BUILD_SWIMLANE || PA_BUILD_SUBMIT_PMU || PA_BUILD_PERF_CLOCK
    RunSchedulerImpl<Ops, false>(state, worker_id, role);
#else
    // Profile 作为编译期模板参数，只在显式开启时保留阶段累计代码，关闭时不在热路径增加运行时分支。
    if (state->config.profile_phases != 0) {
        RunSchedulerImpl<Ops, true>(state, worker_id, role);
    } else {
        RunSchedulerImpl<Ops, false>(state, worker_id, role);
    }
#endif
}

}  // namespace pa_scheduler

#endif  // PA_SCHEDULER_COMMON_PA_SCHEDULER_CORE_H
