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
    TraceContext trace;
};

#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
// runtime-entry TU 按核型各自拥有一份 external [[block_local]] 实例。
// orchestration caller 与 noinline finish 只交换固定 ticket/TaskArgs，Submit
// 内部的 context、统计与状态指针全部留在这份每核状态里。
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
#if PA_BUILD_SUBMIT_PMU
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

PA_DEVICE TaskKind GetTaskKind(uint32_t task_id) { return static_cast<TaskKind>(task_id % kTasksPerBatch); }

PA_DEVICE int32_t FunctionId(TaskKind kind) {
    return kind == TaskKind::Alloc ? -1 : static_cast<int32_t>(KindIndex(kind) - 1);
}

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
    // 完成发布顺序与 PA 一致：先公布该 worker 的 heap 游标，再发布 ready flag，最后推进连续 frontier。
    // fanin 和 heap 回收方以 flag/frontier 为可见性条件，因此不能交换 vend 与 flag 的先后关系。
    TraceAtomicExchange<Ops>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), AtomicSite::CompletionVendExchange,
        &state->tasks[task_id].vend, worker.heap_next
    );
    Ops::StoreBarrier();
    TraceAtomicExchange<Ops>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), AtomicSite::CompletionFlagExchange,
        &state->tasks[task_id].flag, static_cast<int64_t>(1)
    );
    AdvanceFrontier<Ops>(state, stats);
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
    stats.result.kernel_cycles[index] += cycles;
    if (stats.result.kernel_min_cycles[index] == 0 || cycles < stats.result.kernel_min_cycles[index]) {
        stats.result.kernel_min_cycles[index] = cycles;
    }
    if (cycles > stats.result.kernel_max_cycles[index]) {
        stats.result.kernel_max_cycles[index] = cycles;
    }
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
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_id, LocalStats &stats
) {
    // 四个物理 slot 中预留两个 won slot 语义位，仅有 kUsableSlots 个可供本图使用；满时靠 drain 取得进展。
    const uint64_t wait_begin = TraceTimestamp<Ops>(stats.trace, stats.result);
    bool waited = false;
    // 只聚合这个显式背压等待区中的 fanin 观察；每次 Submit 开头的
    // opportunistic EfDrain 仍保留逐条 Atomic，不能仅凭 site 名称全局聚合。
    const uint32_t poll_region = AtomicPollRegionBegin<Ops>(
        stats.trace, stats.result, TraceAtomicSiteMask(AtomicSite::FaninFlagLoad)
    );
    // 退出条件只有 occupied_count 重新低于可用容量；依赖尚未 ready 时 SpinHint 后继续重试。
    while (worker.occupied_count >= kUsableSlots) {
        waited = true;
        ++stats.result.wait_iterations[0];
        if (DrainReady<Ops>(
                state, worker, DrainPlace::RingBackpressure, stats
            ) == 0) {
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
                TraceAtomicSiteMask(AtomicSite::FatalPoll) |
                    TraceAtomicSiteMask(AtomicSite::HeapFrontierLoad) |
                    TraceAtomicSiteMask(AtomicSite::HeapVendLoad) |
                    TraceAtomicSiteMask(AtomicSite::FaninFlagLoad)
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
    // Claim 在四个 shard 的单调 cursor 上执行 atomicMax：同一 task 只有观察到旧值更小的竞争者获胜。
    // Alloc 由全部 96 个 worker 竞争；QK/PV 仅 32 个 AIC，SF/UP 仅 64 个 AIV 进入真正的 atomicMax。
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
            cursor = &state->vector_cursor[task_id % kCursorShards];
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
    WaitForSlot<Ops, Profile>(state, worker, task_id, stats);
    if (!HeapGuard<Ops, Profile>(
            state, worker, task_id, context.output_bytes, stats
        )) {
        return false;
    }
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
    BuildSlotPayload(
        slot, task_id, static_cast<uint32_t>(FunctionId(kind)), 0, args, context, fanin, fanin_count,
        sub_block_id
    );
    stats.result.slot_tensor_copies += static_cast<uint32_t>(context.tensor_count);
    stats.result.slot_scalar_copies += static_cast<uint32_t>(context.scalar_count);
    stats.result.fanin_edges += fanin_count;
    return true;
}

#if PTO_FDWIC_SHARED_MAP
// shared TensorMap 的顺序前沿不借用 Claim cursor：后者按 role 分片且会被
// 快核提前推进，不能证明更早 task 的 region 已经发布。只有当前 task 的
// winner 等待 committed_tasks 精确等于 task_id；若已经越过该 task，说明
// ordered append 协议被破坏，不能把“>=”误当成本 task 仍持有 turn。
template <typename Ops>
PA_DEVICE bool WaitForSharedTaskTurn(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker,
    uint32_t task_count, uint32_t task_id,
    LocalStats &stats
) {
    const uint64_t begin = Ops::Now();
    uint32_t polls = 0;
    while (true) {
        const int64_t committed =
            Ops::Load(&state->shared_map.committed_tasks.value);
        if (committed < 0 ||
            committed > static_cast<int64_t>(task_count)) {
            SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
            return false;
        }
        if (committed == static_cast<int64_t>(task_id)) {
            return true;
        }
        if (committed > static_cast<int64_t>(task_id)) {
            SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
            return false;
        }
        if (IsFatal<Ops>(state, stats, static_cast<int32_t>(task_id))) {
            return false;
        }
        if (DrainReady<Ops>(
                state, worker, DrainPlace::RingBackpressure, stats
            ) == 0) {
            Ops::SpinHint();
        }
        if (WatchdogExpired<Ops>(state, stats, begin, polls)) {
            return false;
        }
    }
}

// register_mask 只指向已经存在的 Local/GM descriptor。两个地址空间分支
// 必须保持分离，避免 CCEC 把它们合并成不支持的 pointer phi。
PA_DEVICE bool BuildSharedRegistrationEntries(
    const TaskArgs &args, const SubmitContext &context,
    SharedRegionValue entries[kMaxTaskTensors], uint32_t &count
) {
    count = 0;
    uint32_t register_mask = context.register_mask;
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const uint32_t bit = 1U << static_cast<uint32_t>(index);
        if ((register_mask & bit) == 0) {
            continue;
        }
        const TaskTensorRef &reference = args.tensors[index];
        // S3.1 的 fresh Output 由 (producer,slot) 直接寻址，不能退回
        // region map。manual_dep 的 output_view 同样不是 TensorMap 的
        // 自动 hazard，保留在 task args 但不登记。
        if (reference.kind == TensorRefKind::SharedOutputRef) {
            register_mask &= ~bit;
            continue;
        }
        if (reference.kind == TensorRefKind::GmTensor) {
            PA_GM const TensorDesc &tensor = *reference.pointer.gm_tensor;
            if (!tensor.manual_dep) {
                entries[count++] = MakeSharedRegionValue(tensor, context.task_id);
            }
        } else if (reference.kind == TensorRefKind::LocalTensor) {
            const TensorDesc &tensor = *reference.pointer.local_tensor;
            if (!tensor.manual_dep) {
                entries[count++] = MakeSharedRegionValue(tensor, context.task_id);
            }
        } else {
            return false;
        }
        register_mask &= ~bit;
    }
    return register_mask == 0 && count <= kMaxTaskTensors;
}

template <typename Ops>
PA_DEVICE uint32_t CollectSharedFanin(
    PA_GM SharedTensorMapSidecar &map, const TaskArgs &args, SubmitContext &context,
    int32_t task_id, int32_t heap_window, LocalStats &stats,
    int32_t fanin[kMaxFanin], bool &protocol_ok, uint32_t &ordinary_lookup_count
) {
    protocol_ok = true;
    ordinary_lookup_count = 0;
    if (args.tensor_count < 0 ||
        args.tensor_count > static_cast<int32_t>(kMaxTaskTensors)) {
        protocol_ok = false;
        return 0;
    }

    int32_t validated_fanin[kMaxFanin] = {};
    int64_t symbol_writers[kMaxTaskTensors] = {};
    uint32_t validated_count = 0;
    uint32_t validated_ordinary_lookups = 0;

    // 第一遍只读取并校验，不修改 last_writer、payload、统计或输出 fanin。
    // 后续引用即使非法，也不会让前面的合法 INOUT 留下半次提交。
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const TensorArgType tag =
            TaskTag(args, static_cast<uint32_t>(index));
        if (tag == TensorArgType::Output) {
            continue;
        }
        const TaskTensorRef &reference = args.tensors[index];
        if (reference.kind == TensorRefKind::SharedOutputRef) {
            // S3.1 暂只接收普通 fresh Output；view ABI 已占位但尚未接入，
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
            if (Ops::Load(&cell.published[output_ref.output_slot].value) !=
                static_cast<int64_t>(output_ref.producer_task_id)) {
                protocol_ok = false;
                return 0;
            }
            if (tag != TensorArgType::Input &&
                tag != TensorArgType::Inout &&
                tag != TensorArgType::OutputExisting) {
                protocol_ok = false;
                return 0;
            }
            // 同一 task 对同一 symbol 最多只能有一个写引用，否则第二次
            // exchange 会把本 task 自己误当成旧 writer。
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
            } else {
                const int64_t writer =
                    Ops::Load(&cell.last_writer[output_ref.output_slot].value);
                // INPUT 读取的 writer 必须严格属于已经提交的较早 task。
                if (writer < 0 || writer >= task_id) {
                    protocol_ok = false;
                    return 0;
                }
                symbol_writers[index] = writer;
                AddFanin(
                    validated_fanin, validated_count,
                    static_cast<int32_t>(writer)
                );
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
                AddFanin(
                    validated_fanin, validated_count,
                    static_cast<int32_t>(owner & 0xFFFFFFFFU)
                );
            }
            if (tag == TensorArgType::Inout ||
                tag == TensorArgType::OutputExisting ||
                (tag == TensorArgType::Input && owner != kInvalidTaskId)) {
                bool lookup_ok = false;
                const int32_t producer = SharedLookupTensor<Ops>(
                    map, tensor, task_id, heap_window, lookup_ok
                );
                if (!lookup_ok) {
                    protocol_ok = false;
                    return 0;
                }
                ++validated_ordinary_lookups;
                AddFanin(validated_fanin, validated_count, producer);
            }
        } else if (reference.kind == TensorRefKind::LocalTensor) {
            const TensorDesc &tensor = *reference.pointer.local_tensor;
            if (tensor.manual_dep) {
                continue;
            }
            const uint64_t owner = tensor.owner_task_id;
            if (owner != kInvalidTaskId) {
                AddFanin(
                    validated_fanin, validated_count,
                    static_cast<int32_t>(owner & 0xFFFFFFFFU)
                );
            }
            if (tag == TensorArgType::Inout ||
                tag == TensorArgType::OutputExisting ||
                (tag == TensorArgType::Input && owner != kInvalidTaskId)) {
                bool lookup_ok = false;
                const int32_t producer = SharedLookupTensor<Ops>(
                    map, tensor, task_id, heap_window, lookup_ok
                );
                if (!lookup_ok) {
                    protocol_ok = false;
                    return 0;
                }
                ++validated_ordinary_lookups;
                AddFanin(validated_fanin, validated_count, producer);
            }
        } else {
            protocol_ok = false;
            return 0;
        }
    }

    // 第二遍先发布全部写符号的 writer。exact-turn 保证本 task 解析期间
    // 没有其他合法 consumer；若 Exchange 返回非法旧值，则回滚已经完成
    // 的 exchange，并保持 payload、统计和 fanin 都未发布。
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const TaskTensorRef &reference = args.tensors[index];
        if (reference.kind != TensorRefKind::SharedOutputRef) {
            continue;
        }
        const TensorArgType tag =
            TaskTag(args, static_cast<uint32_t>(index));
        if (tag != TensorArgType::Inout &&
            tag != TensorArgType::OutputExisting) {
            continue;
        }
        const FdwicOutputRef output_ref = SharedOutputReference(reference);
        PA_GM volatile int64_t *writer_address =
            &map.shared_outputs[
                 static_cast<uint32_t>(output_ref.producer_task_id)
             ].last_writer[output_ref.output_slot].value;
        const int64_t observed =
            Ops::Exchange(writer_address, static_cast<int64_t>(task_id));
        // INOUT 在第二遍用 Exchange 同时取得旧 writer 和发布当前 writer；
        // 旧值必须来自更早的已提交 task。
        if (observed >= 0 && observed < task_id) {
            symbol_writers[index] = observed;
            AddFanin(
                validated_fanin, validated_count,
                static_cast<int32_t>(observed)
            );
            continue;
        }
        (void)Ops::Exchange(writer_address, observed);
        for (int32_t previous = index - 1; previous >= 0; --previous) {
            const TaskTensorRef &previous_ref = args.tensors[previous];
            if (previous_ref.kind != TensorRefKind::SharedOutputRef) {
                continue;
            }
            const TensorArgType previous_tag =
                TaskTag(args, static_cast<uint32_t>(previous));
            if (previous_tag != TensorArgType::Inout &&
                previous_tag != TensorArgType::OutputExisting) {
                continue;
            }
            const FdwicOutputRef previous_output =
                SharedOutputReference(previous_ref);
            (void)Ops::Exchange(
                &map.shared_outputs[
                     static_cast<uint32_t>(previous_output.producer_task_id)
                 ].last_writer[previous_output.output_slot].value,
                symbol_writers[previous]
            );
        }
        protocol_ok = false;
        return 0;
    }

    // writer 链完整更新后，才把 descriptor 复制到当前 task 的 payload
    // scratch，并一次性发布统计与 fanin 结果。RingSlot 永远不保存符号引用。
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const TaskTensorRef &reference = args.tensors[index];
        if (reference.kind != TensorRefKind::SharedOutputRef) {
            continue;
        }
        const FdwicOutputRef output_ref = SharedOutputReference(reference);
        PA_GM const TensorDesc &shared_tensor =
            map.shared_outputs[
                static_cast<uint32_t>(output_ref.producer_task_id)
            ].tensors[output_ref.output_slot];
        Ops::InvalidateRegion(&shared_tensor, sizeof(shared_tensor));
        CopyGmTensor(context.payload->tensors[index], shared_tensor);
        const TensorArgType tag =
            TaskTag(args, static_cast<uint32_t>(index));
        if (tag == TensorArgType::Input) {
            ++stats.result.shared_symbol_input_loads;
        } else if (tag == TensorArgType::Inout ||
            tag == TensorArgType::OutputExisting) {
            ++stats.result.shared_symbol_inout_exchanges;
        }
    }
    ordinary_lookup_count = validated_ordinary_lookups;
    for (uint32_t edge = 0; edge < validated_count; ++edge) {
        fanin[edge] = validated_fanin[edge];
    }
    return validated_count;
}

// fresh descriptor 的内容先写入每 task 独占的 shared-output cell，并通过
// FlushRegion 让 descriptor 先于 published 可见。last_writer 是该符号的
// writer 链起点；published 则只证明 descriptor 已完成发布。两者不可合并。
template <typename Ops>
PA_DEVICE_NOINLINE void RollbackSharedTaskOutputs(
    PA_GM SharedOutputCell &cell, uint32_t output_count
) {
    // 此入口只处理 exact-turn 已被判定破坏后的冷失败路径。先撤销发布位，
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
        // 当前 winner 已持有 exact turn；这些 slot 尚无合法并发访问者，
        // 因此预检用普通 volatile GM load 即可，不额外制造 atomic RMW。
        if (source == nullptr ||
            cell.published[output].value != -1 ||
            cell.last_writer[output].value != -1) {
            return false;
        }
    }
    // exact-turn 唯一 winner 使预检到写入之间不存在合法竞争。仍用
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

// 当前 task 的唯一 winner 持有 append turn。Append 前必须再次核对 exact
// turn，并按 task_id/H 精确推进 reclaim；容量不足与 seq/cursor/单调状态
// 破坏都直接 fatal，不在 ordered writer 内等待其他 worker 的 replay 进度。
// 写入全部 entry 后才发布空/非空 commit。
template <typename Ops>
PA_DEVICE bool AppendSharedTaskOrdered(
    PA_GM SchedulerState *state,
    const SharedRegionValue *entries, uint32_t count,
    const SubmitContext &context, uint32_t task_id, LocalStats &stats
) {
    int64_t reclaim_upto = -1;
    // Refresh 的第一步就是 committed_tasks==task_id 的 exact-turn 检查；
    // 不在热路径重复一次相同 atomic load。
    if (!SharedRefreshReclaimForTask<Ops>(
            state->shared_map, static_cast<int32_t>(task_id),
            static_cast<int32_t>(state->heap_window), reclaim_upto
        )) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    if (!SharedPreflightTaskAppend<Ops>(
            state->shared_map, entries, count, reclaim_upto
        )) {
        // CapacityBlocked 在精确 task-order reclaim 后仍无空间，已没有可等待
        // 的慢核进度；与 ProtocolError 一样立即广播 fatal。
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }

    // preflight 已经对当前 exact-turn 的全部 bucket/slot 做完容量与 seq
    // 核验。只有通过这一步，fresh descriptor 才允许写入可发布表；若
    // preflight 失败，则不会留下任何 published。此时 task 尚未 commit，
    // consumer 仍不能越过 ordered 前沿观察它。
    if (!PublishSharedTaskOutputs<Ops>(
            state->shared_map, context, task_id
        )) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    if (!SharedAppendPreparedTask<Ops>(
            state->shared_map, entries, count
        ) ||
        !SharedPublishTaskCommit<Ops>(
            state->shared_map, static_cast<int32_t>(task_id)
        )) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    return true;
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
    // 外层 callback 和所有参数 thunk 都只在这一调用点同步执行。eager
    // 语义没有 winner 条件：96 个 worker 的前端构参次数保持完全一致。
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

template <typename Ops, bool Profile, typename PmuContext>
PA_DEVICE bool FinishCallbackSubmitBody(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_count,
    const TaskArgs &args, SubmitContext &context, LocalStats &stats,
    PmuContext &pmu_context, const CallbackSubmitTicket &ticket
) {
    const uint32_t task_id = ticket.task_id;
    const TaskKind kind = GetTaskKind(task_id);
    const int32_t function_id = static_cast<int32_t>(ticket.function_id);
    const bool winner = ticket.won != 0;

    // callback 已经返回，finish 只消费稳定的 TaskArgs。该函数既可被
    // CPU/AscendC/局部 PMU 内联，也可在正式 CCEC 路径由独立 TU 实例化。
    const uint64_t materialize_begin = TraceTimestamp<Ops>(stats.trace, stats.result);
    BeginSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(pmu_context);
    const bool materialized =
        MaterializeTask(worker, task_id, args, context, state->heap_base, state->heap_size);
    if (!materialized) {
        EndSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(pmu_context);
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    stats.result.materialized_outputs += context.result.count;
    EndSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(pmu_context);
    const uint64_t materialize_end = TraceTimestamp<Ops>(stats.trace, stats.result);
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), function_id,
        TracePhase::Materialize, ProfilePhase::Materialize,
        materialize_begin, materialize_end, 0, kind == TaskKind::Alloc ? 1U : 0U
    );

    const uint64_t prepare_begin = materialize_end;
#if PTO_FDWIC_SHARED_MAP
    // 只有 winner 会查询 shared map 并追加 task N；因此也只有 winner 在
    // PrepareMap 等待 committed_tasks 精确等于 N。loser 不触碰 map，
    // 无需被全局 sequencer 串行化。
    if (__builtin_expect(winner, 0) &&
        !WaitForSharedTaskTurn<Ops>(
            state, worker, task_count, task_id, stats
        )) {
        return false;
    }
#else
    AdvanceTensorMap(worker.map, task_id, static_cast<int32_t>(state->heap_window));
#endif
    const uint64_t prepare_end = TraceTimestamp<Ops>(stats.trace, stats.result);
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), function_id,
        TracePhase::PrepareMap, ProfilePhase::PrepareMap,
        prepare_begin, prepare_end, 0, kind == TaskKind::Alloc ? 1U : 0U
    );

    uint64_t register_begin = prepare_end;
    if (kind != TaskKind::Alloc && __builtin_expect(winner, 0)) {
        const uint64_t fanin_begin = prepare_end;
#if PTO_FDWIC_SHARED_MAP
        bool lookup_protocol_ok = false;
        uint32_t ordinary_lookup_count = 0;
        context.fanin_count = static_cast<int32_t>(CollectSharedFanin<Ops>(
            state->shared_map, args, context, static_cast<int32_t>(task_id),
            static_cast<int32_t>(state->heap_window), stats, context.fanin,
            lookup_protocol_ok, ordinary_lookup_count
        ));
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
    bool registered = true;
    if (__builtin_expect(winner, 0)) {
        uint32_t shared_entry_count = 0;
        SharedRegionValue shared_entries[kMaxTaskTensors];
        registered = BuildSharedRegistrationEntries(
            args, context, shared_entries, shared_entry_count
        );
        if (registered) {
            registered = AppendSharedTaskOrdered<Ops>(
                state, shared_entries, shared_entry_count, context, task_id, stats
            );
        }
        if (registered) {
            stats.result.map_inserts += shared_entry_count;
        }
    }
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
        // 固定桶容量不足时，InsertTensor 没有覆写任何 live 槽。沿用现有
        // fatal 广播终止所有 worker，禁止像旧 linked map 一样静默漏登记
        // hazard、随后带着不完整 fanin 继续执行。
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }

    if (__builtin_expect(winner, 0)) {
        const uint64_t winner_build_begin = register_end;
        if (kind == TaskKind::Alloc) {
            if (!HeapGuard<Ops, Profile>(state, worker, task_id, context.output_bytes, stats)) {
                return false;
            }
            CompleteTask<Ops>(state, worker, task_id, stats);
        } else {
            if (!BuildWinner<Ops, Profile>(
                    state, worker, task_id, kind, args, context, context.fanin,
                    static_cast<uint32_t>(context.fanin_count), stats
                )) {
                return false;
            }
        }
        const uint64_t winner_build_end = TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<false>(
            stats.trace, stats.result, static_cast<int32_t>(task_id), function_id,
            kind == TaskKind::Alloc ? TracePhase::AllocComplete : TracePhase::WinnerBuild,
            ProfilePhase::ReplayTail, winner_build_begin, winner_build_end
        );
    }

    ++stats.result.submits;
#if PA_BUILD_SUBMIT_PMU
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
}

#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
template <typename Ops>
PA_DEVICE uint32_t FinishSplitCallbackSubmitFromRuntime(
    const CallbackSubmitTicket *ticket, const TaskArgs *args
) {
    CompeteFirstSplitRuntimeState &runtime = Ops::CompeteFirstSplitState();
    const uint64_t state_address = reinterpret_cast<uint64_t>(&runtime);
    runtime.finish_state_address = state_address;

    bool valid = ticket != nullptr && args != nullptr && runtime.scheduler != nullptr &&
                 runtime.worker != nullptr && runtime.task_count != 0 &&
                 runtime.worker_id < kWorkers && runtime.owner_worker_id == runtime.worker_id &&
                 runtime.worker->core_idx == static_cast<int32_t>(runtime.worker_id) &&
                 runtime.caller_state_address == state_address &&
                 runtime.state_cookie == CompeteFirstSplitStateCookie(
                     runtime.worker_id, runtime.worker->role
                 ) && runtime.reserved == 0;
    if (valid) {
        valid = ticket->reserved == 0 && ticket->task_id < runtime.task_count &&
                runtime.context.task_id == static_cast<int32_t>(ticket->task_id) &&
                runtime.context.kernel_id == static_cast<int32_t>(ticket->function_id) &&
                runtime.context.won == (ticket->won != 0);
    }
    ++runtime.finish_calls;
    if (ticket != nullptr) runtime.task_id_sum += ticket->task_id;
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
) {
    BeginCallbackSubmit(worker, context);
    const uint32_t task_id = static_cast<uint32_t>(context.task_id);
#if PA_BUILD_SUBMIT_PMU
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

    if (!BuildCallbackSubmitArgs<Kind>(orch, args, batch, stats)) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    const CallbackSubmitTicket ticket{
        submit_begin,
        task_id,
        static_cast<int16_t>(claim.function_id),
        static_cast<uint8_t>(claim.won ? 1 : 0),
        0,
    };
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
    (void)state;
    (void)worker;
    (void)task_count;
    (void)context;
    (void)stats;
    (void)pmu_context;
    return Ops::FinishCompeteFirstCallback(&ticket, &args);
#else
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
        const uint64_t head = map.bucket_heads[bucket];
        const uint64_t tail = map.bucket_tails[bucket];
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
    PA_PUBLISH_FIELD(shared_symbol_inout_exchanges);
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
        state->config.scheduler_state_size == static_cast<uint32_t>(sizeof(SchedulerState));
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
    stats.result.startup_barrier_begin = Ops::Now();
    TraceAtomicFetchAdd<Ops>(
        stats.trace, stats.result, -1, AtomicSite::StartupIncrement,
        &state->started_count.value, 1
    );
    const uint64_t start_wait = Ops::Now();
    uint32_t start_polls = 0;
    const uint32_t startup_poll_region = AtomicPollRegionBegin<Ops>(
        stats.trace, stats.result,
        TraceAtomicSiteMask(AtomicSite::StartupPoll) | TraceAtomicSiteMask(AtomicSite::FatalPoll)
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
    stats.result.startup_barrier_end = Ops::Now();

    const uint32_t batches = state->config.batches;
    const uint32_t task_count = batches * kTasksPerBatch;
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
    if (!IsFatal<Ops>(state, stats)) {
        // Case1 每个 batch 固定回放 Alloc/QK/SF/PV/UP 五个 task；所有 worker 顺序相同，执行 lane 由 Claim 筛选。
        // CCEC 可在这里开启本 worker 私有 PMU 窗口；CPU/AscendC 适配层是空实现。
        // 窗口覆盖本 worker 的全部调度期：从 orchestration 初始化前开始，
        // 依次包含 EfDrain、Claim、同步 eager 参数构造与后续 Submit 阶段，到末次 Submit 返回。
        // 它与全局“首 Submit.begin～末 Submit.end”口径接近但不相同，host sidecar
        // 必须按 per-worker 累计解释。泳道父边界在 PMU-only 构建中会被编译为空，
        // 不应污染 Submit 取数。
        auto pmu_context = Ops::PmuWindowStart(state, worker_id);
        orchestration_begin = TraceTimestamp<Ops>(stats.trace, stats.result);
        InitPaOrchestration(orchestration, batches, &state->context_lens[0]);
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
        Ops::PmuWindowStop(state, worker_id, pmu_context);
        orchestration_end = TraceTimestamp<Ops>(stats.trace, stats.result);
    }

    // replay_done 表示所有 worker 已退出回放循环（成功路径即完整提交）；之后仍需 drain 到本核 slot 为空。
    // 成功路径复用 orchestration end 作为 final drain start，使两个业务父区间
    // 首尾相接；父记录延后到 final drain 结束再写，避免记录自身落进任一业务 span。
    const uint64_t final_drain_begin = orchestration_end != 0
        ? orchestration_end
        : TraceTimestamp<Ops>(stats.trace, stats.result);
    stats.result.final_barrier_begin = Ops::Now();
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
        TraceAtomicSiteMask(AtomicSite::ReplayDonePoll) | TraceAtomicSiteMask(AtomicSite::FaninFlagLoad)
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
            stats.result.final_barrier_release = Ops::Now();
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
    AtomicPollRegionEnd<Ops>(stats.trace, stats.result, final_poll_region);
    stats.result.final_barrier_end = Ops::Now();
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

#if !PA_BUILD_SUBMIT_PMU
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
    const bool split_protocol_ok =
        split_runtime.scheduler == state && split_runtime.worker == &worker &&
        split_runtime.task_count == task_count && split_runtime.worker_id == worker_id &&
        split_runtime.owner_worker_id == worker_id && split_runtime.caller_state_address != 0 &&
        split_runtime.finish_state_address == split_runtime.caller_state_address &&
        split_runtime.finish_calls == task_count && split_runtime.task_id_sum == expected_task_id_sum &&
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
    stats.result.finish_cycle = Ops::Now();
    stats.result.max_occupied = stats.max_occupied;
    stats.result.final_occupied = worker.occupied_count;
    stats.result.final_heap_next = worker.heap_next;
#if PTO_FDWIC_SHARED_MAP
    // shared 后端没有每核 map 控制字；这里发布统一逻辑窗口摘要，实际
    // bucket/head/tail/seq/payload 由 host 回读唯一 sidecar 后逐槽校验。
    // S3.1 的 fresh Output 都由 shared-output table 直接寻址，manual_dep
    // 的 output_view 也不进入自动 region hazard；因此 Case1 region ring
    // 严格为空。ordered reclaim 前沿仍照 task/H 计算，供 host 闭合回收状态。
    const uint32_t shared_map_floor =
        task_count > kHeapWindow + 1
            ? task_count - kHeapWindow - 1
            : 0;
    stats.result.map_high_water = 0;
    stats.result.map_alive_floor = shared_map_floor;
    stats.result.map_cleaned_upto = shared_map_floor;
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
#if PA_BUILD_SWIMLANE || PA_BUILD_SUBMIT_PMU
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
