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
#endif
#include "pa_trace.h"

namespace pa_scheduler {

struct LocalStats {
    WorkerResult result;
    uint32_t max_occupied;
#if PTO_FDWIC_SHARED_MAP
    // 只在末个 shared Submit 成功收尾时写入 task_id+1；回放结束后与
    // local_index 对照，证明 ticket 的 last bit 没有提前或遗漏。
    uint32_t declared_task_count;
#endif
    TraceContext trace;
};

#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
// runtime-entry TU 按核型各自拥有一份 external [[block_local]] 实例。
// orchestration caller 与 noinline finish 只交换固定 ticket/TaskArgs，Submit
// 内部的 context、统计与状态指针全部留在这份每核状态里。shared loser
// 不跨 TU：finish_calls 是 winner 数，task_id_sum 仍证明全量 caller replay。
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
static_assert(sizeof(CompeteFirstSplitRuntimeState) % 64 == 0,
              "split runtime state must occupy whole cache lines");
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
    (void)result;
    // 对齐真实 FDWIC 的 TRACE_SPAN_BEGIN/END：取一次时间后立即以同一
    // cycle 关闭活跃 PollBatch。不能在 WriteTrace 中统一关闭，否则直接
    // Atomic 记录也会错误切断等待 episode。
    const uint64_t cycle = Ops::Now();
    AtomicPollBoundaryAt<Ops>(trace, cycle);
    return cycle;
#endif
}

PA_DEVICE uint32_t KindIndex(TaskKind kind) { return static_cast<uint32_t>(kind); }

#if PTO_FDWIC_SHARED_MAP
constexpr uint8_t kSharedPaTicketMetaPresent = 1U << 7;
constexpr uint8_t kSharedPaTicketLastSubmit = 1U << 6;
constexpr uint8_t kSharedPaTicketHasFollowing = 1U << 5;
constexpr uint8_t kSharedPaTicketKindMask = 0x07U;
constexpr uint8_t kSharedPaTicketGroupShift = 3;
constexpr uint8_t kSharedPaTicketGroupMask = 0x03U;

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
PA_DEVICE void SetFatal(PA_GM SchedulerState *state, LocalStats &stats, int32_t task_id = -1) {
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
    // 每个 fanin flag 都是跨核共享的完成条件；遇到第一个未就绪依赖即返回，后续 drain 会再次轮询。
    for (uint32_t index = 0; index < slot.fanin_count; ++index) {
        const int32_t dependency = slot.fanin[index];
        if (TraceAtomicLoad<Ops>(
                stats.trace, stats.result, dependency, AtomicSite::FaninFlagLoad,
                &state->tasks[dependency].flag
            ) == 0) {
            ++stats.result.fanin_not_ready_loads;
            return false;
        }
        ++stats.result.fanin_ready_loads;
    }
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
    if (worker.occupied_count == 0) {
        return 0;
    }
    uint32_t freed = 0;
    // 一次调用遍历完本核全部私有 slot；未就绪项保留 occupied/built，已完成项立即清槽并继续扫描。
    for (uint32_t index = 0; index < kPrivateSlots; ++index) {
        PA_GM LocalSlot &slot = worker.slots[index];
        if (!slot.occupied || !slot.built || !SlotReady<Ops>(state, slot, stats)) {
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
        stats.trace, stats.result, TraceAtomicPollBatchMask(AtomicSite::FaninFlagLoad)
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
                Ops::Load(&state->fatal.value) != 0) {
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
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_id, TaskKind kind,
    LocalStats &stats
) {
    // Claim 在单调 cursor 上执行 atomicMax。private/Cube/Alloc 使用
    // production-prefix 四分片；S4.14b shared Vector 使用 sidecar 中
    // 的八分片 cursor。同一 task 只有观察到旧值更小的竞争者获胜。
    // Alloc 由全部 96 个 worker 竞争；QK/PV 仅 32 个 AIC、
    // SF/UP 仅 64 个 AIV 发 atomicMax。
    ClaimOutcome outcome{false, false, 0, -1};
    if (task_id >= kTaskCellCapacity) {
        return outcome;
    }
    PA_GM AtomicLine *cursor = nullptr;
    if (kind == TaskKind::Alloc) {
        cursor = &state->alloc_cursor[task_id % kCursorShards];
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
            if (worker.role != CoreRole::Aic) return outcome;
            cursor = &state->cube_cursor[task_id % kCursorShards];
            outcome.function_id = aic_kernel;
        } else if ((core_mask & 6U) != 0) {
            if (worker.role != CoreRole::Aiv) return outcome;
#if PTO_FDWIC_SHARED_MAP
            cursor = &state->shared_map.shared_vector_cursor[
                task_id % kSharedVectorCursorShards
            ];
#else
            cursor = &state->vector_cursor[task_id % kCursorShards];
#endif
            outcome.function_id = (core_mask & 2U) != 0 ? aiv0_kernel : aiv1_kernel;
        } else {
            return outcome;
        }
    }
    outcome.attempted = true;
    // atomicMax 返回写入前的 cursor：old<task_id 表示本核完成首次推进并获胜，old>=task_id 则必须 Replay。
    const int64_t old = TraceAtomicFetchMax<Ops>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), AtomicSite::ClaimMax,
        &cursor->value, static_cast<int64_t>(task_id), outcome.retries
    );
    outcome.won = old < static_cast<int64_t>(task_id);
    if (!outcome.won) outcome.function_id = -1;
    return outcome;
}

PA_DEVICE void RecordClaimOutcome(LocalStats &stats, TaskKind kind, const ClaimOutcome &outcome) {
    if (outcome.attempted) ++stats.result.claim_attempts;
    stats.result.cas_retries += outcome.retries;
    if (outcome.won) {
        ++stats.result.claim_wins;
        ++stats.result.wins[KindIndex(kind)];
    }
}

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
        state->shared_map, sub_block_id
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
    PA_GM SchedulerState *state, PA_GM WorkerState &worker
) {
    // 只有所有 worker 都已退出 replay、不会再生产 slot 后才能调用。
    // fatal 表示本轮结果已经整体无效；此时未完成 fanin 不可能再获得
    // completion，继续 FinalDrain 只会永久自旋。保留 slot 的 task/fanin
    // 内容供诊断，只清除本地执行资格与占用计数。
    if (state == nullptr || Ops::Load(&state->fatal.value) == 0) {
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

template <typename Ops>
PA_DEVICE bool WaitForSharedOutputPublished(
    PA_GM SharedTensorMapSidecar &map, const FdwicOutputRef &output_ref,
    PA_GM volatile int32_t *fatal
) {
    // 前置条件：调用者已经用 IsPlainSharedOutputRef 校验 producer/slot/view
    // 范围，并确认 producer_task_id 严格早于当前 consumer task。
    PA_GM volatile int64_t *published =
        &map.shared_outputs[
             static_cast<uint32_t>(output_ref.producer_task_id)
         ].published[output_ref.output_slot].value;
    const int64_t expected =
        static_cast<int64_t>(output_ref.producer_task_id);
    int64_t observed = Ops::Load(published);
    if (observed == expected) {
        return true;
    }
    if (observed != -1) {
        if (fatal != nullptr) {
            (void)Ops::Exchange(fatal, static_cast<int32_t>(1));
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
        observed = Ops::Load(published);
        if (observed == expected) {
            return true;
        }
        if (observed != -1) {
            if (fatal != nullptr) {
                (void)Ops::Exchange(fatal, static_cast<int32_t>(1));
            }
            return false;
        }
        ++polls;
        if ((polls & 1023U) != 0) {
            continue;
        }
        if (fatal != nullptr && Ops::Load(fatal) != 0) {
            return false;
        }
        if (Ops::Now() - begin > kWatchdogTicks) {
            if (fatal != nullptr) {
                (void)Ops::Exchange(fatal, static_cast<int32_t>(1));
            }
            return false;
        }
    }
}

template <
    typename Ops, bool ChainedWriter = false,
    bool AcceptLatestWriter = false
>
PA_DEVICE uint32_t CollectSharedFanin(
    PA_GM SharedTensorMapSidecar &map, const TaskArgs &args,
    int32_t task_id, int32_t heap_window, LocalStats &stats,
    int32_t fanin[kMaxFanin], bool &protocol_ok,
    uint32_t &ordinary_lookup_count,
    PA_GM volatile int32_t *fatal = nullptr,
    int32_t chained_producer_task_id = -1,
    int32_t expected_shared_writer = -1
) {
    static_assert(
        !(ChainedWriter && AcceptLatestWriter),
        "generic latest-writer lookup must not use the PA chained selector"
    );
    protocol_ok = true;
    ordinary_lookup_count = 0;
    if (args.tensor_count < 0 ||
        args.tensor_count > static_cast<int32_t>(kMaxTaskTensors)) {
        protocol_ok = false;
        return 0;
    }
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

    // 只读取并校验，不修改 last_writer、统计或输出 fanin。final UP 与
    // 普通任务在 Build 成功后提交 writer；non-final UP 则在 registration
    // 校验后、Build 前提交 intent。两条路径都先完成本轮只读校验，后续
    // 引用非法时不会留下半次 writer 更新。
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
            PA_GM SharedOutputCell &cell =
                map.shared_outputs[static_cast<uint32_t>(output_ref.producer_task_id)];
            if (!WaitForSharedOutputPublished<Ops>(
                    map, output_ref, fatal
                )) {
                protocol_ok = false;
                return 0;
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
            const int64_t writer =
                Ops::Load(&cell.last_writer[output_ref.output_slot].value);
            if constexpr (AcceptLatestWriter) {
                // A->B->C 门槛在 B 的 writer-ready 之后接受 cell 当前
                // last_writer；descriptor identity 仍固定为最初 producer。
                // 单 cell 尚未保存历史，因此 future writer>=current_task
                // 必须拒绝，不能谎称已经解析到“最新过去版本”。
                if (writer < output_ref.producer_task_id ||
                    writer >= task_id) {
                    protocol_ok = false;
                    return 0;
                }
            } else {
                // PA 迁移完成前保留原来的精确 oracle：默认单组要求
                // writer==descriptor producer，ChainedWriter 只允许调用方
                // 指定的 accumulator 链。两种口径不能静默混用。
                const bool chained_ref =
                    ChainedWriter &&
                    output_ref.producer_task_id ==
                        chained_producer_task_id;
                const int32_t expected_writer =
                    chained_ref
                        ? expected_shared_writer
                        : output_ref.producer_task_id;
                if (expected_writer < output_ref.producer_task_id ||
                    expected_writer >= task_id ||
                    writer != expected_writer) {
                    protocol_ok = false;
                    return 0;
                }
            }
            if (!AddCollectedSharedFanin<AcceptLatestWriter>(
                    validated_fanin, validated_count,
                    static_cast<int32_t>(writer)
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
            if (owner != kInvalidTaskId) {
                if (!AddCollectedSharedFanin<
                        AcceptLatestWriter
                    >(
                    validated_fanin, validated_count,
                    static_cast<int32_t>(owner & 0xFFFFFFFFU)
                )) {
                    protocol_ok = false;
                    return 0;
                }
            }
            if (tag == TensorArgType::Inout ||
                tag == TensorArgType::OutputExisting ||
                (tag == TensorArgType::Input &&
                 (owner != kInvalidTaskId || AcceptLatestWriter))) {
                bool lookup_ok = false;
                const int32_t producer = SharedLookupTensor<Ops>(
                    map, tensor, task_id, heap_window, lookup_ok
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
            if (owner != kInvalidTaskId) {
                if (!AddCollectedSharedFanin<
                        AcceptLatestWriter
                    >(
                    validated_fanin, validated_count,
                    static_cast<int32_t>(owner & 0xFFFFFFFFU)
                )) {
                    protocol_ok = false;
                    return 0;
                }
            }
            if (tag == TensorArgType::Inout ||
                tag == TensorArgType::OutputExisting ||
                (tag == TensorArgType::Input &&
                 (owner != kInvalidTaskId || AcceptLatestWriter))) {
                bool lookup_ok = false;
                const int32_t producer = SharedLookupTensor<Ops>(
                    map, tensor, task_id, heap_window, lookup_ok
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
            if (shape == 0 || extent > UINT64_MAX / shape) {
                return false;
            }
            extent *= shape;
        }
    }
    if (extent == 0 ||
        tensor.start_offset > UINT64_MAX - extent ||
        tensor.start_offset > UINT64_MAX / element_size ||
        tensor.start_offset + extent > UINT64_MAX / element_size) {
        return false;
    }
    region.buffer_addr = tensor.buffer_addr;
    region.lo = tensor.start_offset * element_size;
    region.hi = (tensor.start_offset + extent) * element_size;
    region.producer = task_id;
    region.reserved = 0;
    return region.lo < region.hi;
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
    if (tensor.owner_task_id == kInvalidTaskId) {
        return true;
    }
    const int32_t owner = static_cast<int32_t>(
        tensor.owner_task_id & 0xFFFFFFFFU
    );
    return owner >= 0 && owner < task_id;
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
        const int32_t owner = static_cast<int32_t>(
            tensor.owner_task_id & 0xFFFFFFFFU
        );
        if (!AddSharedWriterIntentFanin(
                fanin, fanin_count, owner
            )) {
            return false;
        }
    }

    bool lookup_ok = false;
    const int32_t previous = SharedLookupTensor<Ops>(
        map, tensor, task_id, heap_window, lookup_ok
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
PA_DEVICE bool CommitSymbolSharedWriterIntent(
    PA_GM SharedTensorMapSidecar &map,
    const FdwicOutputRef &output_ref, int32_t task_id,
    int32_t fanin[kMaxFanin], uint32_t &fanin_count,
    LocalStats &stats, PA_GM volatile int32_t *fatal
) {
    if (!WaitForSharedOutputPublished<Ops>(
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
    const int64_t previous = Ops::Load(last_writer);
    if (previous < output_ref.producer_task_id ||
        previous >= task_id ||
        !AddSharedWriterIntentFanin(
            fanin, fanin_count,
            static_cast<int32_t>(previous)
        ) ||
        Ops::CompareExchange(
            last_writer, previous,
            static_cast<int64_t>(task_id)
        ) != previous) {
        return false;
    }
    ++stats.result.shared_symbol_inout_commits;
    return true;
}

// 该公共原语只处理写集合本身：读取旧 writer、发布当前 writer，并在
// 全部 symbol/ordinary 元数据完成后放行同 task loser。它不收集纯 INPUT，
// 不做 Materialize/Build，也不发布 completion；后两者必须继续使用
// task.flag。当前阶段只锁定单条 A->B->C：
// - symbol 仍是单 last_writer cell，尚无 future-writer 历史查询；
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
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const TensorArgType tag =
            TaskTag(args, static_cast<uint32_t>(index));
        if (!IsSharedWriterIntentTag(tag)) {
            continue;
        }
        const TaskTensorRef &reference = args.tensors[index];
        bool committed = false;
        if (reference.kind == TensorRefKind::SharedOutputRef) {
            committed = CommitSymbolSharedWriterIntent<Ops>(
                state->shared_map,
                SharedOutputReference(reference), task_id,
                intent_fanin, intent_fanin_count, stats,
                &state->fatal.value
            );
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
// 三个 accumulator writer，再发布 deps_prepared。它位于 winner Build
// 之前；后续 Finish 必须复用 context.fanin，并在 Build 后跳过第二次
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
// FlushRegion 让 descriptor 与 writer 起点先于 published 可见。调用者
// 必须已经成功 CompleteTask/BuildWinner；因此 published 表示 producer
// Submit 已封口，而不再只是“descriptor 已经构造”。
template <typename Ops>
PA_DEVICE_NOINLINE void RollbackSharedTaskOutputs(
    PA_GM SharedOutputCell &cell, uint32_t output_count
) {
    // 此入口只处理唯一 producer cell 出现非法竞争后的冷失败路径。先撤销发布位，
    // 使任何非法越界 reader 都不能继续消费，再恢复 writer 与 descriptor
    // 的未发布状态；正常 Submit 不执行这些额外 atomic/DCCI。
    for (uint32_t output = 0; output < output_count; ++output) {
        (void)Ops::Exchange(&cell.published[output].value, -1);
    }
    for (uint32_t output = 0; output < output_count; ++output) {
        (void)Ops::Exchange(&cell.last_writer[output].value, -1);
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
        Ops::FlushRegion(
            &cell.tensors[0],
            static_cast<uint64_t>(output_count) * sizeof(TensorDesc)
        );
    }
}

template <typename Ops>
PA_DEVICE bool PublishSharedTaskOutputs(
    PA_GM SharedTensorMapSidecar &map, const SubmitContext &context,
    uint32_t task_id
) {
    if (task_id >= kMaxTasks || context.shared_result.TaskId() != static_cast<int32_t>(task_id) ||
        context.shared_result.Size() != context.result.count ||
        context.result.count > kSharedOutputMaxPerTask) {
        return false;
    }
    PA_GM SharedOutputCell &cell = map.shared_outputs[task_id];
    // 先完整预检所有 slot；异常重复发布不能覆盖已对 consumer 可见的
    // descriptor，也不能让多输出 task 留下前半段控制字。
    for (uint32_t output = 0; output < context.result.count; ++output) {
        PA_GM TensorDesc *source = context.result.tensors[output];
        // cell 由该 task 的唯一 Claim winner 独占；其他 task 只能在
        // published 就绪后读取，因此预检用普通 volatile GM load 即可。
        if (source == nullptr ||
            cell.published[output].value != -1 ||
            cell.last_writer[output].value != -1) {
            return false;
        }
    }
    // task-cell 唯一 winner 使预检到写入之间不存在合法竞争。仍用
    // FetchMax 预留全部 writer 控制字，并在异常旧值时撤回本次已预留项。
    for (uint32_t output = 0; output < context.result.count; ++output) {
        uint64_t retries = 0;
        const int64_t observed = Ops::FetchMax(
            &cell.last_writer[output].value,
            static_cast<int64_t>(task_id), retries
        );
        if (observed != -1) {
            // atomicMax 在 observed<task_id 时已经改写当前 slot；无论旧值
            // 大小都显式恢复，前面已成功预留的 slot 则回到 -1。
            (void)Ops::Exchange(
                &cell.last_writer[output].value, observed
            );
            for (uint32_t previous = 0; previous < output; ++previous) {
                (void)Ops::Exchange(&cell.last_writer[previous].value, -1);
            }
            return false;
        }
    }
    for (uint32_t output = 0; output < context.result.count; ++output) {
        PA_GM TensorDesc *source = context.result.tensors[output];
        CopyGmTensor(cell.tensors[output], *source);
        Ops::FlushRegion(&cell.tensors[output], sizeof(TensorDesc));
    }
    Ops::StoreBarrier();
    for (uint32_t output = 0; output < context.result.count; ++output) {
        if (Ops::Exchange(
                &cell.published[output].value, static_cast<int64_t>(task_id)
            ) != -1) {
            RollbackSharedTaskOutputs<Ops>(cell, context.result.count);
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

PA_DEVICE uint64_t SharedSplitTicketBinding(
    const CallbackSubmitTicket &ticket
) {
    return kSharedSplitTicketBindingPresent |
           (static_cast<uint64_t>(ticket.task_id) << 8U) |
           static_cast<uint64_t>(ticket.reserved);
}

PA_DEVICE bool ArmSharedSplitTicket(
    CompeteFirstSplitRuntimeState &runtime,
    const CallbackSubmitTicket &ticket
) {
    // caller 在跨 TU 前保存由动态 plan 推导出的唯一 task/meta 身份；
    // finish 必须逐字匹配并消费。成功 Submit 数同时充当 next task id，
    // 因而重复、跳号和乱序 ticket 都不能进入 Finish body。
    if (runtime.reserved != 0 ||
        static_cast<uint64_t>(ticket.task_id) !=
            runtime.stats.result.submits) {
        return false;
    }
    runtime.reserved = SharedSplitTicketBinding(ticket);
    return true;
}

PA_DEVICE bool RecordSharedSplitReplayTask(
    CompeteFirstSplitRuntimeState &runtime,
    const CallbackSubmitTicket &ticket
) {
    // task_id_sum 描述 caller 确实按 0..N-1 重放了完整前端序列，
    // 与只有 winner 才跨 TU 的 finish_calls 是两条不同的协议证据。
    // loser 不再 Arm ticket，因此这一步必须留在 caller。
    if (runtime.reserved != 0 ||
        static_cast<uint64_t>(ticket.task_id) !=
            runtime.stats.result.submits) {
        return false;
    }
    runtime.task_id_sum += ticket.task_id;
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
    // 是否构参：private 仍全员 eager；shared 的 Alloc 保留全员轻构参，
    // QK/SF/PV/UP 则只由 Claim winner 进入这里。
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
    const CallbackSubmitTicket &ticket,
    const SharedPaTaskMeta &shared_task_meta
) {
    const uint32_t task_id = ticket.task_id;
    const int32_t function_id =
        static_cast<int32_t>(ticket.function_id);
    const bool winner = ticket.won != 0;
    ++stats.result.submits;

    // shared 的总 task 数取决于每批 context_len。末次身份由调用者在
    // 既有 ticket bit 中显式携带，避免 96 个 worker 为了一个计时边界
    // 额外预扫整份 context_lens。
    const bool is_last_submit = shared_task_meta.is_last_submit;
#if PA_BUILD_PERF_CLOCK
    // 与真实 FDWIC perf-clock 相同：只有末个逻辑 Submit 完成后才读取
    // 一次专用性能边界；loser 也必须闭合自己的全量重放窗口。
    const uint64_t submit_end =
        is_last_submit ? Ops::PerfClockNow() : 0;
#elif PA_BUILD_SUBMIT_PMU
    const uint64_t submit_end = is_last_submit ? Ops::Now() : 0;
#else
    // 参考前端的 rt_submit_loser 仍是一次真实轻量 Submit；保留既有父
    // 区间用于覆盖稳定符号返回及 non-final UP writer-ready 等待，但
    // loser 不再写任何 winner-only child。
    const uint64_t submit_end =
        TraceTimestamp<Ops>(stats.trace, stats.result);
#endif
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id),
        function_id, TracePhase::Submit, ProfilePhase::Submit,
        ticket.submit_begin, submit_end, winner ? 1U : 0U,
        shared_task_meta.kind == TaskKind::Alloc ? 1U : 0U
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
PA_DEVICE bool FinishSharedLoserSubmit(
    PA_GM SchedulerState *state, SubmitContext &context,
    LocalStats &stats, const CallbackSubmitTicket &ticket
) {
    SharedPaTaskMeta shared_task_meta{};
    const uint32_t task_id = ticket.task_id;
    const bool valid =
        ticket.won == 0 &&
        DecodeSharedPaTaskMeta(
            ticket.reserved, task_id, shared_task_meta
        ) &&
        SharedPaFunctionIdMatches(
            shared_task_meta.kind, false,
            static_cast<int32_t>(ticket.function_id)
        ) &&
        context.task_id == static_cast<int32_t>(task_id) &&
        !context.won &&
        context.kernel_id ==
            static_cast<int32_t>(ticket.function_id) &&
        context.shared_result.TaskId() ==
            static_cast<int32_t>(task_id) &&
        context.shared_result.Size() ==
            FrontendTaskOutputCount(shared_task_meta.kind);
    if (!valid) {
        SetFatal<Ops>(
            state, stats, static_cast<int32_t>(task_id)
        );
        return false;
    }

    // non-final UP 的 loser 只保留这一条业务必需的等待：winner 先登记
    // INOUT writer intent，loser 才能返回前端并构造下一组。它不读取
    // TaskArgs，也不进入 Materialize/Fanin/Register/BuildWinner。
    if (shared_task_meta.has_following_group &&
        !WaitForSharedWriterReady<Ops>(
            state, static_cast<int32_t>(task_id), stats
        )) {
        return false;
    }
    return CloseSharedCallbackSubmit<Ops, Profile>(
        state, stats, ticket, shared_task_meta
    );
}
#endif

template <typename Ops, bool Profile, typename PmuContext>
PA_DEVICE bool FinishCallbackSubmitBody(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_count,
    const TaskArgs &args, SubmitContext &context, LocalStats &stats,
    PmuContext &pmu_context, const CallbackSubmitTicket &ticket
) {
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
        // intent 与 deps_prepared 发布后、建立可执行 slot 前注入失败。
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
        state, stats, ticket, shared_task_meta
    );
#else
    ++stats.result.submits;
#if PA_BUILD_PERF_CLOCK
    // 与真实 FDWIC perf-clock 相同：只有末个 Submit 完成全部尾动作后
    // 才采一次专用性能边界。协议 watchdog 继续使用 Ops::Now()，两者
    // 在源码上保持可审计的不同接口。
    const uint64_t submit_end =
        task_id + 1 == task_count ? Ops::PerfClockNow() : 0;
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
    const bool expected_ticket_bound =
        ticket != nullptr &&
        runtime.reserved == SharedSplitTicketBinding(*ticket);
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
        static_cast<uint64_t>(ticket->task_id) ==
            runtime.stats.result.submits;
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
    return FinishCallbackSubmitBody<Ops, false>(
        runtime.scheduler, *runtime.worker, runtime.task_count, *args,
        runtime.context, runtime.stats, pmu_context, *ticket
    ) ? 1U : 0U;
}
#endif

template <TaskKind Kind, typename Ops, bool Profile, typename PmuContext>
PA_DEVICE bool SubmitCallbackTask(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_count,
    PaOrchestrationState &orch, TaskArgs &args, uint32_t batch,
    SubmitContext &context, LocalStats &stats, PmuContext &pmu_context
#if PTO_FDWIC_SHARED_MAP
    , const SharedPaBatchPlan &shared_batch_plan,
    uint32_t shared_task_offset
#endif
) {
    BeginCallbackSubmit(worker, context);
    const uint32_t task_id = static_cast<uint32_t>(context.task_id);
#if PTO_FDWIC_SHARED_MAP
    SharedPaPlannedTask shared_planned_task{};
    if (batch >= state->config.batches ||
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
        batch + 1U == state->config.batches;
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
    DrainReady<Ops>(state, worker, DrainPlace::EfDrain, stats);
    EndSubmitPmuPhase<SubmitPmuPhase::EfDrain, Ops>(pmu_context);
    const uint64_t efdrain_end = TraceTimestamp<Ops>(stats.trace, stats.result);
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), -1,
        TracePhase::EfDrain, ProfilePhase::EfDrain, efdrain_begin, efdrain_end
    );

    const uint64_t claim_begin = efdrain_end;
    BeginSubmitPmuPhase<SubmitPmuPhase::Claim, Ops>(pmu_context);
    const ClaimOutcome claim = Claim<Ops>(state, worker, task_id, Kind, stats);
    context.won = claim.won;
    context.kernel_id = claim.function_id;
    RecordClaimOutcome(stats, Kind, claim);
    EndSubmitPmuPhase<SubmitPmuPhase::Claim, Ops>(pmu_context);
    const uint64_t claim_end = TraceTimestamp<Ops>(stats.trace, stats.result);
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), claim.function_id,
        TracePhase::Claim, ProfilePhase::Claim, claim_begin, claim_end,
        (claim.won ? kClaimWon : 0U) | (claim.attempted ? kClaimAttempted : 0U),
        Kind == TaskKind::Alloc ? 1U : 0U
    );

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
    if constexpr (Kind == TaskKind::Alloc) {
        // 为对齐参考 alloc_tensors(args) 的调用形状，并把本小步变量限定
        // 在四个重构参 task，Alloc 暂保留全员三个静态 Output 参数。
        // standalone 的 output symbol 已由上方独立声明，并不依赖这次构参。
        if (!BuildCallbackSubmitArgs<Kind>(orch, args, batch, stats)) {
            SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
            return false;
        }
    } else if (__builtin_expect(claim.won, 0)) {
        // shared loser 已在上方声明稳定 output symbol；它不需要构造本 task
        // 的 descriptor/scalar 参数。finish 的 loser 分支只闭合边界，不读
        // 这里留下的上一 task args。
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
    const CallbackSubmitTicket ticket{
        submit_begin,
        task_id,
        static_cast<int16_t>(claim.function_id),
        static_cast<uint8_t>(claim.won ? 1 : 0),
#if PTO_FDWIC_SHARED_MAP
        shared_task_meta,
#else
        0,
#endif
    };
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
#if PTO_FDWIC_SHARED_MAP
    CompeteFirstSplitRuntimeState &split_runtime =
        Ops::CompeteFirstSplitState();
    if (!RecordSharedSplitReplayTask(split_runtime, ticket)) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    if (!claim.won) {
        return FinishSharedLoserSubmit<Ops, Profile>(
            state, context, stats, ticket
        );
    }
    if (!ArmSharedSplitTicket(split_runtime, ticket)) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
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
            state, context, stats, ticket
        );
    }
#endif
    return FinishCallbackSubmitBody<Ops, Profile>(
        state, worker, task_count, args, context, stats, pmu_context, ticket
    );
#endif
}

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
    Ops::InvalidateRegion(
        &state->config,
        sizeof(state->config) + sizeof(state->pmu_probe) + sizeof(state->winner_workload)
    );
    const bool build_identity_matches =
        state->config.build_identity_magic == kBuildIdentityMagic &&
        state->config.build_identity_abi_version == kBuildIdentityAbiVersion &&
        state->config.tensor_map_mode == static_cast<uint32_t>(kCompiledTensorMapMode) &&
        state->config.scheduler_state_size == static_cast<uint32_t>(sizeof(SchedulerState)) &&
        state->pmu_probe.build_variant == kCompiledBuildVariant;
    if (!build_identity_matches) {
        (void)Ops::Exchange(&state->fatal.value, static_cast<int32_t>(1));
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

    // startup 严格保持生产 flat 语义；本实验只改变 replay 尾部的 final 汇合。
    // 96 个参与者全部完成本地状态初始化后再进入 task 0，主要用于压低启动偏斜对
    // winner 分布和 Submit 时序的干扰；atomicMax 的唯一 winner 正确性本身不依赖该屏障。
#if PA_BUILD_PERF_CLOCK
    // perf-clock 不采集生命周期诊断。启动 watchdog 仍在下一行建立自己的
    // 正确性超时起点，不能为了追求字面上的“两次 SYS_CNT”删除防挂死机制。
    stats.result.startup_barrier_begin = 0;
#else
    stats.result.startup_barrier_begin = Ops::Now();
#endif
    TraceAtomicFetchAdd<Ops>(
        stats.trace, stats.result, -1, AtomicSite::StartupIncrement,
        &state->started_count.value, 1
    );
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
#if PA_BUILD_PERF_CLOCK
    stats.result.startup_barrier_end = 0;
#else
    stats.result.startup_barrier_end = Ops::Now();
#endif

    const uint32_t batches = state->config.batches;
#if PTO_FDWIC_SHARED_MAP
    // split Finish 在 replay 过程中只需容量上限；真实 task_count 随每批
    // context_len 变化，回放结束后再由 local_index 封口。末次 Submit
    // 的精确计时身份由 ticket 携带，不预扫 context_lens。
    uint32_t task_count = kMaxTasks;
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
    bool shared_replay_window_started = false;
#endif
    if (!IsFatal<Ops>(state, stats)) {
        // private 每批固定回放 Alloc/QK/SF/PV/UP；shared 则先按
        // context_len 建立 1+4N task plan。所有 worker 的计划相同，执行
        // lane 仍由 Claim 筛选。
        // CCEC 可在这里开启本 worker 私有 PMU 窗口；CPU/AscendC 适配层是空实现。
        // 窗口覆盖本 worker 的全部调度期：从 orchestration 初始化前开始，
        // 依次包含 EfDrain、Claim、当前模式实际执行的参数构造与后续 Submit
        // 阶段，到末次 Submit 返回。private 为全员 eager；shared 仅 Alloc
        // 全员构参，其余 task 由 winner 构参。
        // 它与全局“首 Submit.begin～末 Submit.end”口径接近但不相同，host sidecar
        // 必须按 per-worker 累计解释。泳道父边界在 PMU-only 构建中会被编译为空，
        // 不应污染 Submit 取数。
        auto pmu_context = Ops::PmuWindowStart(state, worker_id);
#if PTO_FDWIC_SHARED_MAP
        shared_replay_window_started = true;
#endif
        orchestration_begin = TraceTimestamp<Ops>(stats.trace, stats.result);
        InitPaOrchestration(orchestration, batches, &state->context_lens[0]);
#if PTO_FDWIC_SHARED_MAP
        bool replay_ok = true;
        for (uint32_t batch = 0;
             batch < batches && replay_ok; ++batch) {
            BeginPaBatchForCallback(orchestration, batch);
            ++stats.result.context_reads;
            SharedPaBatchPlan batch_plan{};
            if (!BuildSharedPaBatchPlan(
                    orchestration.current_sequence,
                    worker.local_index, batch_plan
                )) {
                SetFatal<Ops>(
                    state, stats,
                    static_cast<int32_t>(worker.local_index)
                );
                break;
            }
            if (!SubmitCallbackTask<TaskKind::Alloc, Ops, Profile>(
                    state, worker, task_count, orchestration, args,
                    batch, context, stats, pmu_context,
                    batch_plan, 0
                )) {
                break;
            }
            AcceptTaskOutputs(
                orchestration, TaskKind::Alloc,
                OrchestrationOutputs(context)
            );

            for (uint32_t group = 0;
                 group < batch_plan.group_count; ++group) {
                const uint64_t block_offset =
                    static_cast<uint64_t>(group) *
                    kPaBlocksPerRequest;
                PreparePaBlockGroup(orchestration, block_offset);
                if (!SubmitCallbackTask<
                        TaskKind::Qk, Ops, Profile
                    >(
                        state, worker, task_count, orchestration,
                        args, batch, context, stats, pmu_context,
                        batch_plan,
                        SharedPaTaskOffset(TaskKind::Qk, group)
                    )) {
                    replay_ok = false;
                    break;
                }
                AcceptTaskOutputs(
                    orchestration, TaskKind::Qk,
                    OrchestrationOutputs(context)
                );

                if (!SubmitCallbackTask<
                        TaskKind::Sf, Ops, Profile
                    >(
                        state, worker, task_count, orchestration,
                        args, batch, context, stats, pmu_context,
                        batch_plan,
                        SharedPaTaskOffset(TaskKind::Sf, group)
                    )) {
                    replay_ok = false;
                    break;
                }
                AcceptTaskOutputs(
                    orchestration, TaskKind::Sf,
                    OrchestrationOutputs(context)
                );

                if (!SubmitCallbackTask<
                        TaskKind::Pv, Ops, Profile
                    >(
                        state, worker, task_count, orchestration,
                        args, batch, context, stats, pmu_context,
                        batch_plan,
                        SharedPaTaskOffset(TaskKind::Pv, group)
                    )) {
                    replay_ok = false;
                    break;
                }
                AcceptTaskOutputs(
                    orchestration, TaskKind::Pv,
                    OrchestrationOutputs(context)
                );

                if (!SubmitCallbackTask<
                        TaskKind::Up, Ops, Profile
                    >(
                        state, worker, task_count, orchestration,
                        args, batch, context, stats, pmu_context,
                        batch_plan,
                        SharedPaTaskOffset(TaskKind::Up, group)
                    )) {
                    replay_ok = false;
                    break;
                }
            }
            const uint32_t expected_batch_end =
                batch_plan.batch_start + batch_plan.task_count;
            if (replay_ok &&
                (worker.local_index < 0 ||
                 static_cast<uint32_t>(worker.local_index) !=
                     expected_batch_end)) {
                SetFatal<Ops>(
                    state, stats,
                    static_cast<int32_t>(worker.local_index)
                );
                replay_ok = false;
            }
        }
#else
        for (uint32_t batch = 0; batch < batches; ++batch) {
            BeginPaBatchForCallback(orchestration, batch);
            ++stats.result.context_reads;
            if (!SubmitCallbackTask<TaskKind::Alloc, Ops, Profile>(
                    state, worker, task_count, orchestration, args, batch, context, stats,
                    pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(
                orchestration, TaskKind::Alloc, OrchestrationOutputs(context)
            );

            PreparePaBlockGroup(orchestration, 0);
            if (!SubmitCallbackTask<TaskKind::Qk, Ops, Profile>(
                    state, worker, task_count, orchestration, args, batch, context, stats,
                    pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(
                orchestration, TaskKind::Qk, OrchestrationOutputs(context)
            );

            if (!SubmitCallbackTask<TaskKind::Sf, Ops, Profile>(
                    state, worker, task_count, orchestration, args, batch, context, stats,
                    pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(
                orchestration, TaskKind::Sf, OrchestrationOutputs(context)
            );

            if (!SubmitCallbackTask<TaskKind::Pv, Ops, Profile>(
                    state, worker, task_count, orchestration, args, batch, context, stats,
                    pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(
                orchestration, TaskKind::Pv, OrchestrationOutputs(context)
            );

            if (!SubmitCallbackTask<TaskKind::Up, Ops, Profile>(
                    state, worker, task_count, orchestration, args, batch, context, stats,
                    pmu_context
                )) {
                break;
            }
        }
#endif
#if PTO_FDWIC_SHARED_MAP
        // 先封口实际 task 数和唯一 last 身份，再停止 PMU。这样
        // missing/early/duplicate-last 不会先产出看似合法的 phase shape；
        // Stop 仍在 fatal 路无条件执行，保证本核计数器完成收口。
        task_count = FinalizeSharedReplayTaskCount<Ops>(
            state, worker, stats
        );
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
        split_runtime.task_count = task_count;
#endif
#endif
        Ops::PmuWindowStop(state, worker_id, pmu_context);
        orchestration_end = TraceTimestamp<Ops>(stats.trace, stats.result);
    }
#if PTO_FDWIC_SHARED_MAP
    // 构建身份或启动阶段已经 fatal 时 PMU 从未开启，但后续 split
    // 协议仍必须使用实际的零 task 数，不能保留容量上限 kMaxTasks。
    if (!shared_replay_window_started) {
        task_count = FinalizeSharedReplayTaskCount<Ops>(
            state, worker, stats
        );
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
        split_runtime.task_count = task_count;
#endif
    }
#endif

    // replay_done 表示所有 worker 已退出回放循环（成功路径即完整提交）；之后仍需 drain 到本核 slot 为空。
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
            TraceAtomicPollBatchMask(AtomicSite::FaninFlagLoad)
    );
    bool leaf_forwarded = false;
    bool middle_forwarded = false;
    bool root_released = false;
    bool middle_released = false;
    bool leaf_released = false;
    bool global_release_observed = false;
#if PTO_FDWIC_SHARED_MAP
    uint32_t final_stall_polls = 0;
#endif
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
#if PTO_FDWIC_SHARED_MAP
        // final barrier 证明所有 replay actor 已停止生产。只有本轮没有
        // 释放任何 slot、且本核仍被未完成 fanin 阻塞时才按 1024 次一批
        // 探测 fatal；正常 count==0 出口不新增原子，正常跨核推进也不会
        // 被误清。fatal 时撤销本核执行资格，保证所有 worker 收敛退出。
        if (global_release_observed && freed == 0 &&
            worker.occupied_count != 0) {
            ++final_stall_polls;
            if ((final_stall_polls & 1023U) == 0) {
                (void)DiscardSharedSlotsAfterReplayFatal<Ops>(
                    state, worker
                );
            }
        } else {
            final_stall_polls = 0;
        }
#endif
        // 必须同时满足“无人再生产新 slot”和“本核旧 slot 全部完成”，否则继续帮助系统推进 completion。
        if (global_release_observed && worker.occupied_count == 0) {
            break;
        }
        if (freed == 0) {
            Ops::SpinHint();
        }
    }
    AtomicPollRegionEnd<Ops>(stats.trace, stats.result, final_poll_region);
#if PA_BUILD_PERF_CLOCK
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
    if (stats.trace.atomics_enabled) {
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
    const uint64_t expected_task_id_sum =
        static_cast<uint64_t>(task_count) * (task_count - 1U) / 2U;
    // terminal fatal 可能在某个 worker 进入首个 Submit 之前已经可见。
    // 该 worker 合法地没有 finish 调用，finish TU 地址也尚未回写；不能
    // 把这种零回放收敛误报成 split 状态错配。只要开始过任一 Submit，
    // 仍严格要求 caller/finish 是同一个 TLS runtime。
#if PTO_FDWIC_SHARED_MAP
    // shared 的 loser 不跨 TU；某个 worker 即使完整重放了 N 个任务，也
    // 可能一个都没赢。finish 地址是否出现只取决于本核 winner 数。
    const bool finish_state_matches =
        split_runtime.finish_calls == 0
            ? split_runtime.finish_state_address == 0
            : split_runtime.finish_state_address ==
                  split_runtime.caller_state_address;
    const uint64_t expected_finish_calls =
        stats.result.claim_wins;
#else
    const bool finish_state_matches =
        task_count == 0
            ? Ops::Load(&state->fatal.value) != 0 &&
                  split_runtime.finish_calls == 0 &&
                  split_runtime.finish_state_address == 0
            : split_runtime.finish_state_address ==
                  split_runtime.caller_state_address;
    const uint64_t expected_finish_calls = task_count;
#endif
    const bool split_protocol_ok =
        split_runtime.scheduler == state && split_runtime.worker == &worker &&
        split_runtime.task_count == task_count && split_runtime.worker_id == worker_id &&
        split_runtime.owner_worker_id == worker_id && split_runtime.caller_state_address != 0 &&
        finish_state_matches &&
        split_runtime.finish_calls == expected_finish_calls &&
        split_runtime.task_id_sum == expected_task_id_sum &&
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
    FlushTraceCore<Ops>(stats.trace, stats.result);
#if PA_BUILD_PERF_CLOCK
    // 复用末个 Submit 的已保存边界满足结果有序性，不再为 worker 尾部
    // 增加一次纯诊断 SYS_CNT。
    stats.result.finish_cycle = stats.result.submit_end;
#else
    stats.result.finish_cycle = Ops::Now();
#endif
    stats.result.max_occupied = stats.max_occupied;
    stats.result.final_occupied = worker.occupied_count;
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
