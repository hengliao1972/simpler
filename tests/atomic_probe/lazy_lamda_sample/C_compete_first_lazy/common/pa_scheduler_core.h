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
#include "pa_trace.h"

namespace pa_scheduler {

struct LocalStats {
    WorkerResult result;
    uint32_t max_occupied;
    TraceContext trace;
};

#if defined(PA_LAZY_SAMPLE_SPLIT_FINISH)
// runtime TU owns one external [[block_local]] instance per architecture.
// The caller imports that same object and keeps all Submit-internal context in
// it, so the only cross-TU function arguments are a POD ticket and built args.
struct alignas(64) LazySampleSplitRuntimeState {
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
static_assert(sizeof(LazySampleSplitRuntimeState) % 64 == 0,
              "split runtime state must occupy whole cache lines");

PA_DEVICE uint64_t LazySampleSplitStateCookie(uint32_t worker_id, CoreRole role) {
    return kLazySampleSplitStateCookieBase ^ static_cast<uint64_t>(worker_id) ^
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

template <typename Ops, bool Profile, typename PmuContext>
PA_DEVICE bool SubmitTask(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_count, TaskKind kind,
    const TaskArgs &args, SubmitContext &context, LocalStats &stats, PmuContext &pmu_context
) {
    // 每个 worker 都完整回放相同 task stream。主流程为：EfDrain -> materialize -> TensorMap retire
    // -> Claim -> winner 收集 fanin -> 全员 register -> winner Build / loser Replay。Alloc 在 Claim 前 register，
    // 且 winner 不入 kernel slot，而是在 heap guard 后直接发布完成。
    BeginSubmit(worker, args, context);
    const uint32_t task_id = static_cast<uint32_t>(context.task_id);
#if PA_BUILD_SUBMIT_PMU
    // PMU-only ELF 只保留首/末 Submit 的全局时间边界，不再为 1280 次调用
    // 各执行两条 trace-only SYS_CNT。
    const uint64_t submit_begin = task_id == 0 ? Ops::Now() : 0;
#else
    const uint64_t submit_begin = TraceTimestamp<Ops>(stats.trace, stats.result);
#endif
    if (task_id == 0) {
        stats.result.submit_begin = submit_begin;
    }

    // EfDrain 在当前 Submit 的参数物化前执行上一批已就绪 slot，是绝大多数 kernel 的正常落点。
    // 只在这个唯一 call-site 划 PMU 边界；DrainReady 还被 ring 背压和最终 drain
    // 复用，不能把 phase 插入函数体后按 place 混合累计。
    // EfDrain 是 Submit 的第一个真实阶段，直接复用父区间起点；这样每次
    // Submit 少一次 trace-only SYS_CNT，也不会留下人为的 prefix residual。
    const uint64_t efdrain_begin = submit_begin;
    BeginSubmitPmuPhase<SubmitPmuPhase::EfDrain, Ops>(pmu_context);
    DrainReady<Ops>(state, worker, DrainPlace::EfDrain, stats);
    EndSubmitPmuPhase<SubmitPmuPhase::EfDrain, Ops>(pmu_context);
    const uint64_t efdrain_end = TraceTimestamp<Ops>(stats.trace, stats.result);
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::EfDrain,
        ProfilePhase::EfDrain, efdrain_begin, efdrain_end
    );

    // schema-v4 的所有 Submit 子阶段都使用显式 start/end；不再通过共同 lap
    // 起点生成相互覆盖的 Build/Replay/Alloc 区间。
    // 后继 segment 复用前一阶段 end：既少一次 SYS_CNT，也把前一条 trace
    // 发布和阶段间胶水明确归入 Materialize，而不是留成无名 residual。
    const uint64_t materialize_begin = efdrain_end;
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
        stats.trace, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::Materialize,
        ProfilePhase::Materialize, materialize_begin, materialize_end, 0,
        kind == TaskKind::Alloc ? 1U : 0U
    );
    const uint64_t prepare_begin = materialize_end;
    AdvanceTensorMap(worker.map, task_id, static_cast<int32_t>(state->heap_window));
    const uint64_t prepare_end = TraceTimestamp<Ops>(stats.trace, stats.result);
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::PrepareMap,
        ProfilePhase::PrepareMap, prepare_begin, prepare_end, 0, kind == TaskKind::Alloc ? 1U : 0U
    );

    bool winner = false;
    int32_t function_id = -1;

    if (kind == TaskKind::Alloc) {
        // Alloc 没有 kernel lane，96 个 worker 都维护本地物化/heap 状态，但只有 Claim winner 发布全局完成。
        const uint64_t register_begin = prepare_end;
        BeginSubmitPmuPhase<SubmitPmuPhase::Register, Ops>(pmu_context);
        RegisterOutputs(context, args, false);
        EndSubmitPmuPhase<SubmitPmuPhase::Register, Ops>(pmu_context);
        const uint64_t register_end = TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<Profile>(
            stats.trace, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::Register,
            ProfilePhase::Register, register_begin, register_end, 0, 0
        );

        const uint64_t claim_begin = register_end;
        BeginSubmitPmuPhase<SubmitPmuPhase::Claim, Ops>(pmu_context);
        const ClaimOutcome claim = Claim<Ops>(state, worker, task_id, kind, stats);
        winner = claim.won;
        context.won = winner;
        context.kernel_id = claim.function_id;
        // Claim 的本地结果归档属于同一阶段；放在共同 end 边界内，避免把
        // winner/context/stat bookkeeping 留成无法归因的 Submit residual。
        RecordClaimOutcome(stats, kind, claim);
        EndSubmitPmuPhase<SubmitPmuPhase::Claim, Ops>(pmu_context);
        const uint64_t claim_end = TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<Profile>(
            stats.trace, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::Claim,
            ProfilePhase::Claim, claim_begin, claim_end,
            (winner ? kClaimWon : 0U) | (claim.attempted ? kClaimAttempted : 0U), 1
        );
        // 每个 task 只有 1/96 worker 进入 winner 重型路径。把该分支标成冷路
        // 只影响基本块布局，使占绝大多数的 loser 尽量顺序进入 Submit 公共
        // 尾部；不改变 Claim 结果、完成发布或泳道边界。
        if (__builtin_expect(winner, 0)) {
            const uint64_t alloc_complete_begin = claim_end;
            if (!HeapGuard<Ops, Profile>(
                    state, worker, task_id, context.output_bytes, stats
                )) {
                return false;
            }
            CompleteTask<Ops>(state, worker, task_id, stats);
            const uint64_t alloc_complete_end = TraceTimestamp<Ops>(stats.trace, stats.result);
            WriteTrace<false>(
                stats.trace, stats.result, static_cast<int32_t>(task_id), -1,
                TracePhase::AllocComplete, ProfilePhase::ReplayTail,
                alloc_complete_begin, alloc_complete_end
            );
        } else {
            // standalone 的 Alloc loser 没有真实 GM/Replay 动作，不再为业务
            // 路径名字写一条零时长记录。Claim 后到 Submit.end 的真实
            // scalar 时间由离线 submit_tail_gap 补集展示；排他报告将其汇总为
            // submit_tail_residual，避免伪装成 Alloc loser 业务阶段。
        }
    } else {
        const uint64_t claim_begin = prepare_end;
        BeginSubmitPmuPhase<SubmitPmuPhase::Claim, Ops>(pmu_context);
        const ClaimOutcome claim = Claim<Ops>(state, worker, task_id, kind, stats);
        winner = claim.won;
        function_id = claim.function_id;
        context.won = winner;
        context.kernel_id = function_id;
        RecordClaimOutcome(stats, kind, claim);
        EndSubmitPmuPhase<SubmitPmuPhase::Claim, Ops>(pmu_context);
        const uint64_t claim_end = TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<Profile>(
            stats.trace, stats.result, static_cast<int32_t>(task_id), function_id, TracePhase::Claim,
            ProfilePhase::Claim, claim_begin, claim_end,
            (winner ? kClaimWon : 0U) | (claim.attempted ? kClaimAttempted : 0U), 0
        );
        uint64_t register_begin = claim_end;
        if (winner) {
            const uint64_t fanin_begin = claim_end;
            context.fanin_count = static_cast<int32_t>(CollectFanin(worker.map, args, context.fanin));
            stats.result.map_lookups += static_cast<uint32_t>(args.tensor_count) - context.result.count;
            const uint64_t fanin_end = TraceTimestamp<Ops>(stats.trace, stats.result);
            WriteTrace<Profile>(
                stats.trace, stats.result, static_cast<int32_t>(task_id), function_id, TracePhase::Fanin,
                ProfilePhase::Fanin, fanin_begin, fanin_end, 0, static_cast<uint32_t>(context.fanin_count)
            );
            register_begin = fanin_end;
        }

        // loser 直接承接 Claim.end；winner 则承接 Fanin.end。两条路径都
        // 复用已有边界，不再为 Register 单独读取 SYS_CNT。
        BeginSubmitPmuPhase<SubmitPmuPhase::Register, Ops>(pmu_context);
        RegisterOutputs(context, args, true);
        stats.result.map_inserts += CountBits(context.register_mask);
        EndSubmitPmuPhase<SubmitPmuPhase::Register, Ops>(pmu_context);
        const uint64_t register_end = TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<Profile>(
            stats.trace, stats.result, static_cast<int32_t>(task_id), function_id, TracePhase::Register,
            ProfilePhase::Register, register_begin, register_end, 0, 1
        );
        // BuildWinner 会内联 ring、heap 和真计算提交逻辑；提示其为 1/96
        // 冷路，避免 loser 为跳过大块代码付出额外取指代价。
        if (__builtin_expect(winner, 0)) {
            const uint64_t winner_build_begin = register_end;
            if (!BuildWinner<Ops, Profile>(
                    state, worker, task_id, kind, args, context, context.fanin,
                    static_cast<uint32_t>(context.fanin_count), stats
                )) {
                return false;
            }
            const uint64_t winner_build_end = TraceTimestamp<Ops>(stats.trace, stats.result);
            WriteTrace<false>(
                stats.trace, stats.result, static_cast<int32_t>(task_id), function_id,
                TracePhase::WinnerBuild, ProfilePhase::ReplayTail,
                winner_build_begin, winner_build_end
            );
        } else {
            // 非 winner 不占用私有 ring slot，也没有可单列的 Replay 计算。
            // Register 后到 Submit.end 由离线 submit_tail_gap 补集展示；排他
            // 报告汇总为 submit_tail_residual，避免零时长 marker 增加 raw
            // 体积和 trace-buffer 写开销。
        }
    }

    ++stats.result.submits;
#if PA_BUILD_SUBMIT_PMU
    const uint64_t submit_end = task_id + 1 == task_count ? Ops::Now() : 0;
#else
    const uint64_t submit_end = TraceTimestamp<Ops>(stats.trace, stats.result);
#endif
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), function_id, TracePhase::Submit,
        ProfilePhase::Submit, submit_begin, submit_end, winner ? 1U : 0U, kind == TaskKind::Alloc ? 1U : 0U
    );
    if (task_id + 1 == task_count) {
        stats.result.submit_end = submit_end;
    }
    return true;
}

#if defined(PA_LAZY_SAMPLE_SHAPE_ID)
// The all-task callback path crosses the split-TU finish boundary through this
// fixed 16-byte POD.  No callback closure or nested thunk is stored in it.
struct LazySampleCallbackTicket {
    uint64_t submit_begin;
    uint32_t task_id;
    int16_t function_id;
    uint8_t won;
    uint8_t reserved;
};
static_assert(sizeof(LazySampleCallbackTicket) == 16, "lazy sample callback ticket must remain a 16-byte POD");
static_assert(offsetof(LazySampleCallbackTicket, submit_begin) == 0, "lazy sample callback ticket timestamp offset mismatch");
static_assert(offsetof(LazySampleCallbackTicket, task_id) == 8, "lazy sample callback ticket task offset mismatch");
static_assert(offsetof(LazySampleCallbackTicket, function_id) == 12, "lazy sample callback ticket function offset mismatch");
static_assert(offsetof(LazySampleCallbackTicket, won) == 14, "lazy sample callback ticket winner offset mismatch");

PA_DEVICE void BeginLazySampleCallbackSubmit(PA_GM WorkerState &worker, SubmitContext &context) {
    // This is BeginSubmit without an already-materialized TaskArgs.  The same
    // fields are completed synchronously after the single callback builds args.
    const uint32_t task_id = static_cast<uint32_t>(worker.local_index++);
    context.self = &worker;
    context.payload = &worker.payloads[task_id & kPayloadMask];
    context.task_id = static_cast<int32_t>(task_id);
    context.tensor_count = 0;
    context.scalar_count = 0;
    context.result.task_id = task_id;
    context.result.count = 0;
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
#define PA_LAZY_LAMBDA_DEVICE __aicore__
#else
#define PA_LAZY_LAMBDA_DEVICE
#endif

template <TaskKind Kind, bool Lazy>
PA_DEVICE bool BuildLazySampleCallbackArgs(
    PaOrchestrationState &orch, TaskArgs &args, uint32_t batch, bool won,
    LocalStats &stats
) {
    LazySampleCallbackArgsBuilder<Lazy> callback_builder(args, Kind, won);
    auto callback = [&](LazySampleCallbackArgsBuilder<Lazy> &builder) PA_LAZY_LAMBDA_DEVICE {
        builder.Begin();
        if constexpr (Kind == TaskKind::Alloc) {
            builder.AddOutput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorCreateInfo & {
                return orch.tile_create_info;
            });
            builder.AddOutput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorCreateInfo & {
                return orch.scalar_create_info;
            });
            builder.AddOutput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorCreateInfo & {
                return orch.scalar_create_info;
            });
        } else if constexpr (Kind == TaskKind::Qk) {
            builder.AddLocalInput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorDesc & {
                MakeLazySampleCallbackQueryView(orch, batch);
                builder.RecordView();
                return orch.query_view;
            });
            builder.AddLocalInput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorDesc & {
                return orch.key_cache;
            });
            builder.AddLocalInput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorDesc & {
                return orch.block_table;
            });
            builder.AddOutput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorCreateInfo & {
                const uint32_t score_shape[kMaxTensorDims] = {
                    kPaHeads,
                    static_cast<uint32_t>(orch.current_nblocks * kPaBlockSize),
                    0, 0, 0
                };
                InitCreateInfo(orch.qk_create_info, score_shape, 2, DataType::Float32);
                builder.RecordDynamicCreateInfo();
                return orch.qk_create_info;
            });
            builder.AddScalar([&]() PA_LAZY_LAMBDA_DEVICE -> uint64_t {
                return orch.current_nblocks;
            });
            builder.AddScalar([&]() PA_LAZY_LAMBDA_DEVICE -> uint64_t {
                return static_cast<uint64_t>(orch.current_batch) * kPaMaxBlocksPerRequest +
                       orch.current_block_offset;
            });
        } else if constexpr (Kind == TaskKind::Sf) {
            builder.AddGmInput([&]() PA_LAZY_LAMBDA_DEVICE -> PA_GM const TensorDesc & {
                return *orch.qk_scores;
            });
            builder.AddOutput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorCreateInfo & {
                const uint32_t probability_shape[kMaxTensorDims] = {
                    kPaHeads,
                    static_cast<uint32_t>(orch.current_nblocks * kPaBlockSize),
                    0, 0, 0
                };
                InitCreateInfo(orch.sf_create_info, probability_shape, 2, DataType::Bfloat16);
                builder.RecordDynamicCreateInfo();
                return orch.sf_create_info;
            });
            builder.AddOutput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorCreateInfo & {
                return orch.scalar_create_info;
            });
            builder.AddOutput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorCreateInfo & {
                return orch.scalar_create_info;
            });
            builder.AddScalar([&]() PA_LAZY_LAMBDA_DEVICE -> uint64_t { return orch.scale_bits; });
            builder.AddScalar([&]() PA_LAZY_LAMBDA_DEVICE -> uint64_t {
                return orch.current_nblocks;
            });
            builder.AddScalar([&]() PA_LAZY_LAMBDA_DEVICE -> uint64_t {
                return orch.current_valid_len;
            });
        } else if constexpr (Kind == TaskKind::Pv) {
            builder.AddGmInput([&]() PA_LAZY_LAMBDA_DEVICE -> PA_GM const TensorDesc & {
                return *orch.sf_probs;
            });
            builder.AddLocalInput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorDesc & {
                return orch.value_cache;
            });
            builder.AddLocalInput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorDesc & {
                return orch.block_table;
            });
            builder.AddOutput([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorCreateInfo & {
                return orch.tile_create_info;
            });
            builder.AddScalar([&]() PA_LAZY_LAMBDA_DEVICE -> uint64_t {
                return orch.current_nblocks;
            });
            builder.AddScalar([&]() PA_LAZY_LAMBDA_DEVICE -> uint64_t {
                return static_cast<uint64_t>(orch.current_batch) * kPaMaxBlocksPerRequest +
                       orch.current_block_offset;
            });
        } else {
            static_assert(Kind == TaskKind::Up, "unsupported PA task kind");
            builder.AddGmInput([&]() PA_LAZY_LAMBDA_DEVICE -> PA_GM const TensorDesc & {
                return *orch.sf_max;
            });
            builder.AddGmInput([&]() PA_LAZY_LAMBDA_DEVICE -> PA_GM const TensorDesc & {
                return *orch.sf_sum;
            });
            builder.AddGmInput([&]() PA_LAZY_LAMBDA_DEVICE -> PA_GM const TensorDesc & {
                return *orch.pv_output;
            });
            builder.AddGmInout([&]() PA_LAZY_LAMBDA_DEVICE -> PA_GM const TensorDesc & {
                return *orch.accumulated_max;
            });
            builder.AddGmInout([&]() PA_LAZY_LAMBDA_DEVICE -> PA_GM const TensorDesc & {
                return *orch.accumulated_sum;
            });
            builder.AddGmInout([&]() PA_LAZY_LAMBDA_DEVICE -> PA_GM const TensorDesc & {
                return *orch.accumulated_output;
            });
            builder.AddLocalInout([&]() PA_LAZY_LAMBDA_DEVICE -> const TensorDesc & {
                MakeLazySampleCallbackOutputView(orch, batch);
                builder.RecordView();
                return orch.output_view;
            });
            builder.AddScalar([&]() PA_LAZY_LAMBDA_DEVICE -> uint64_t {
                return orch.current_block_offset == 0 ? 1 : 0;
            });
            builder.AddScalar([&]() PA_LAZY_LAMBDA_DEVICE -> uint64_t {
                return orch.current_block_offset + orch.current_nblocks >= orch.current_blocks ? 1 : 0;
            });
        }
    };

    callback(callback_builder);
    if (!callback_builder.Valid()) return false;
    const LazySampleCallbackBuildCounts &counts = callback_builder.Counts();
    stats.result.arg_resets += counts.reset_calls;
    stats.result.views_created += counts.views_created;
    stats.result.dynamic_create_infos += counts.dynamic_create_infos;
    stats.result.tensor_args_added += counts.tensor_args_added;
    stats.result.scalar_args_added += counts.scalar_args_added;
    return true;
}

#undef PA_LAZY_LAMBDA_DEVICE

template <typename Ops, bool Profile, typename PmuContext>
PA_DEVICE bool FinishLazySampleCallbackSubmitBody(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_count,
    const TaskArgs &args, SubmitContext &context, LocalStats &stats,
    PmuContext &pmu_context, const LazySampleCallbackTicket &ticket
) {
    const uint32_t task_id = ticket.task_id;
    const TaskKind kind = GetTaskKind(task_id);
    const int32_t function_id = static_cast<int32_t>(ticket.function_id);
    const bool winner = ticket.won != 0;

    // The callback has ended before this body consumes TaskArgs.  No closure
    // or nested thunk escapes its lifetime.  Inline shapes instantiate this in
    // the caller; split shapes instantiate it inside the runtime finish TU.
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
    AdvanceTensorMap(worker.map, task_id, static_cast<int32_t>(state->heap_window));
    const uint64_t prepare_end = TraceTimestamp<Ops>(stats.trace, stats.result);
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), function_id,
        TracePhase::PrepareMap, ProfilePhase::PrepareMap,
        prepare_begin, prepare_end, 0, kind == TaskKind::Alloc ? 1U : 0U
    );

    uint64_t register_begin = prepare_end;
    if (kind != TaskKind::Alloc && __builtin_expect(winner, 0)) {
        const uint64_t fanin_begin = prepare_end;
        context.fanin_count = static_cast<int32_t>(CollectFanin(worker.map, args, context.fanin));
        stats.result.map_lookups += static_cast<uint32_t>(args.tensor_count) - context.result.count;
        const uint64_t fanin_end = TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<Profile>(
            stats.trace, stats.result, static_cast<int32_t>(task_id), function_id,
            TracePhase::Fanin, ProfilePhase::Fanin,
            fanin_begin, fanin_end, 0, static_cast<uint32_t>(context.fanin_count)
        );
        register_begin = fanin_end;
    }

    BeginSubmitPmuPhase<SubmitPmuPhase::Register, Ops>(pmu_context);
    RegisterOutputs(context, args, kind != TaskKind::Alloc);
    if (kind != TaskKind::Alloc) stats.result.map_inserts += CountBits(context.register_mask);
    EndSubmitPmuPhase<SubmitPmuPhase::Register, Ops>(pmu_context);
    const uint64_t register_end = TraceTimestamp<Ops>(stats.trace, stats.result);
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), function_id,
        TracePhase::Register, ProfilePhase::Register,
        register_begin, register_end, 0, kind == TaskKind::Alloc ? 0U : 1U
    );

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
            ProfilePhase::ReplayTail,
            winner_build_begin, winner_build_end
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
        ticket.submit_begin, submit_end, winner ? 1U : 0U, 0
    );
    if (task_id + 1 == task_count) {
        stats.result.submit_end = submit_end;
    }
    return true;
}

#if defined(PA_LAZY_SAMPLE_SPLIT_FINISH)
template <typename Ops>
PA_DEVICE uint32_t FinishSplitLazySampleCallbackFromRuntime(
    const LazySampleCallbackTicket *ticket, const TaskArgs *args
) {
    LazySampleSplitRuntimeState &runtime = Ops::LazySampleSplitState();
    const uint64_t state_address = reinterpret_cast<uint64_t>(&runtime);
    runtime.finish_state_address = state_address;

    bool valid = ticket != nullptr && args != nullptr && runtime.scheduler != nullptr &&
                 runtime.worker != nullptr && runtime.task_count != 0 &&
                 runtime.worker_id < kWorkers && runtime.owner_worker_id == runtime.worker_id &&
                 runtime.worker->core_idx == static_cast<int32_t>(runtime.worker_id) &&
                 runtime.caller_state_address == state_address &&
                 runtime.state_cookie == LazySampleSplitStateCookie(
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

    // split finish 不让 caller 的 SubmitContext/LocalStats/PMU 对象跨过
    // noinline 边界。诊断构建只允许 phase=none：权威 CNT6/7 仍由 caller
    // 外层完整窗口读取，finish 内部不做 read-clear 局部快照。
#if PA_BUILD_SUBMIT_PMU
    static_assert(
        kCompiledSubmitPmuPhase == SubmitPmuPhase::None,
        "split callback submit-PMU supports only the whole-window none phase"
    );
#endif
    bool pmu_context = false;
    return FinishLazySampleCallbackSubmitBody<Ops, false>(
        runtime.scheduler, *runtime.worker, runtime.task_count, *args,
        runtime.context, runtime.stats, pmu_context, *ticket
    ) ? 1U : 0U;
}
#endif

template <TaskKind Kind, typename Ops, bool Profile, typename PmuContext, bool Lazy>
PA_DEVICE bool SubmitLazySampleCallback(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_count,
    PaOrchestrationState &orch, TaskArgs &args, uint32_t batch,
    SubmitContext &context, LocalStats &stats, PmuContext &pmu_context
) {
    BeginLazySampleCallbackSubmit(worker, context);
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
        TracePhase::EfDrain, ProfilePhase::EfDrain,
        efdrain_begin, efdrain_end
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
        TracePhase::Claim, ProfilePhase::Claim,
        claim_begin, claim_end,
        (claim.won ? kClaimWon : 0U) | (claim.attempted ? kClaimAttempted : 0U),
        Kind == TaskKind::Alloc ? 1U : 0U
    );

    if (!BuildLazySampleCallbackArgs<Kind, Lazy>(orch, args, batch, claim.won, stats)) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    const LazySampleCallbackTicket ticket{
        submit_begin,
        task_id,
        // All standalone function IDs are -1..3 and exactly fit this ABI field.
        static_cast<int16_t>(claim.function_id),
        static_cast<uint8_t>(claim.won ? 1 : 0),
        0,
    };
#if defined(PA_LAZY_SAMPLE_SPLIT_FINISH)
    (void)state;
    (void)worker;
    (void)task_count;
    (void)context;
    (void)stats;
    (void)pmu_context;
    return Ops::FinishLazySampleCallback(&ticket, &args);
#else
    return FinishLazySampleCallbackSubmitBody<Ops, Profile>(
        state, worker, task_count, args, context, stats, pmu_context, ticket
    );
#endif
}
#endif

PA_DEVICE uint32_t CountLiveMapEntries(PA_GM const TensorMap &map) {
    uint32_t free_entries = 0;
    for (int32_t current = map.free_head; current >= 0; current = map.entries[current].next_in_bucket) {
        ++free_entries;
    }
    return static_cast<uint32_t>(map.high_water) - free_entries;
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
#if defined(PA_LAZY_SAMPLE_SPLIT_FINISH)
    PA_PUBLISH_FIELD(lazy_sample_split_caller_state_address);
    PA_PUBLISH_FIELD(lazy_sample_split_finish_state_address);
    PA_PUBLISH_FIELD(lazy_sample_split_finish_calls);
    PA_PUBLISH_FIELD(lazy_sample_split_protocol_errors);
    PA_PUBLISH_FIELD(lazy_sample_split_state_cookie);
    PA_PUBLISH_FIELD(lazy_sample_split_task_id_sum);
    PA_PUBLISH_FIELD(lazy_sample_split_owner_worker_id);
    PA_PUBLISH_FIELD(lazy_sample_split_reserved);
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
    ResetTensorMap(worker.map);
    worker.occupied_count = 0;
    worker.owned_total = 0;
    worker.swimlane_last_cycle = 0;
    for (uint32_t index = 0; index < kPrivateSlots; ++index) {
        worker.slots[index].occupied = false;
        worker.slots[index].built = false;
    }

#if defined(PA_LAZY_SAMPLE_SPLIT_FINISH)
    LazySampleSplitRuntimeState &lazy_sample_split_runtime = Ops::LazySampleSplitState();
    lazy_sample_split_runtime.context = SubmitContext{};
    lazy_sample_split_runtime.stats = LocalStats{};
    lazy_sample_split_runtime.scheduler = state;
    lazy_sample_split_runtime.worker = &worker;
    lazy_sample_split_runtime.task_count = 0;
    lazy_sample_split_runtime.worker_id = worker_id;
    lazy_sample_split_runtime.caller_state_address = reinterpret_cast<uint64_t>(&lazy_sample_split_runtime);
    lazy_sample_split_runtime.finish_state_address = 0;
    lazy_sample_split_runtime.finish_calls = 0;
    lazy_sample_split_runtime.protocol_errors = 0;
    lazy_sample_split_runtime.state_cookie = LazySampleSplitStateCookie(worker_id, role);
    lazy_sample_split_runtime.task_id_sum = 0;
    lazy_sample_split_runtime.owner_worker_id = worker_id;
    lazy_sample_split_runtime.reserved = 0;
    LocalStats &stats = lazy_sample_split_runtime.stats;
#else
    LocalStats stats{};
#endif
    stats.result.worker_id = worker_id;
    stats.result.role = static_cast<uint32_t>(role);
    stats.result.checksum = 0xcbf29ce484222325ULL ^ worker_id;
    // 该 invalidate 原先藏在 AttachTrace 中；它同时保护 PMU mode/register
    // table 与 winner workload，必须在两个构建中都执行。
    Ops::InvalidateRegion(
        &state->config, sizeof(state->config) + sizeof(state->winner_workload)
    );
    stats.trace = AttachTrace<Ops>(state, worker, worker_id);

    // 96 个参与者全部完成本地状态初始化后再进入 task 0，主要用于压低启动偏斜对
    // winner 分布和 Submit 时序的干扰；atomicMax 的唯一 winner 正确性本身不依赖该屏障。
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

    const uint32_t batches = state->config.batches;
    const uint32_t task_count = batches * kTasksPerBatch;
#if defined(PA_LAZY_SAMPLE_SPLIT_FINISH)
    lazy_sample_split_runtime.task_count = task_count;
#endif
    PaOrchestrationState orchestration;
    TaskArgs args;
#if defined(PA_LAZY_SAMPLE_SPLIT_FINISH)
    SubmitContext &context = lazy_sample_split_runtime.context;
#else
    SubmitContext context;
#endif
    uint64_t orchestration_begin = 0;
    uint64_t orchestration_end = 0;
    if (!IsFatal<Ops>(state, stats)) {
        // Case1 每个 batch 固定回放 Alloc/QK/SF/PV/UP 五个 task；所有 worker 顺序相同，执行 lane 由 Claim 筛选。
        // CCEC 可在这里开启本 worker 私有 PMU 窗口；CPU/AscendC 适配层是空实现。
        // 窗口覆盖从首个参数构造到末次 Submit 返回，与全局“首 Submit.begin～末 Submit.end”
        // 口径接近但不相同，host sidecar 必须按 per-worker 累计解释。
        // 泳道父边界在 PMU-only 构建中会被编译为空，不应污染 Submit 取数；窗口从
        // orchestration 初始化（即首批参数构造）前一条边界开始。
        auto pmu_context = Ops::PmuWindowStart(state, worker_id);
        orchestration_begin = TraceTimestamp<Ops>(stats.trace, stats.result);
        InitPaOrchestration(orchestration, batches, &state->context_lens[0]);
        for (uint32_t batch = 0; batch < batches; ++batch) {
#if defined(PA_LAZY_SAMPLE_SHAPE_ID)
            BeginPaBatchForLazySampleCallback(orchestration, batch);
            ++stats.result.context_reads;
            if (!SubmitLazySampleCallback<
                    TaskKind::Alloc, Ops, Profile, decltype(pmu_context), kLazySampleLazy
                >(
                    state, worker, task_count, orchestration, args, batch, context, stats,
                    pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(orchestration, TaskKind::Alloc, context.result);

            PreparePaBlockGroup(orchestration, 0);
            if (!SubmitLazySampleCallback<
                    TaskKind::Qk, Ops, Profile, decltype(pmu_context), kLazySampleLazy
                >(
                    state, worker, task_count, orchestration, args, batch, context, stats,
                    pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(orchestration, TaskKind::Qk, context.result);

            if (!SubmitLazySampleCallback<
                    TaskKind::Sf, Ops, Profile, decltype(pmu_context), kLazySampleLazy
                >(
                    state, worker, task_count, orchestration, args, batch, context, stats,
                    pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(orchestration, TaskKind::Sf, context.result);

            if (!SubmitLazySampleCallback<
                    TaskKind::Pv, Ops, Profile, decltype(pmu_context), kLazySampleLazy
                >(
                    state, worker, task_count, orchestration, args, batch, context, stats,
                    pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(orchestration, TaskKind::Pv, context.result);

            if (!SubmitLazySampleCallback<
                    TaskKind::Up, Ops, Profile, decltype(pmu_context), kLazySampleLazy
                >(
                    state, worker, task_count, orchestration, args, batch, context, stats,
                    pmu_context
                )) {
                break;
            }
#else
            BuildAllocArgs(orchestration, args, batch);
            ++stats.result.context_reads;
            stats.result.views_created += 2;
            stats.result.tensor_args_added += 3;
            if (!SubmitTask<Ops, Profile>(
                    state, worker, task_count, TaskKind::Alloc, args, context, stats, pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(orchestration, TaskKind::Alloc, context.result);

            BuildQkArgs(orchestration, args, batch);
            ++stats.result.dynamic_create_infos;
            ++stats.result.arg_resets;
            stats.result.tensor_args_added += 4;
            stats.result.scalar_args_added += 2;
            if (!SubmitTask<Ops, Profile>(
                    state, worker, task_count, TaskKind::Qk, args, context, stats, pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(orchestration, TaskKind::Qk, context.result);

            BuildSfArgs(orchestration, args);
            ++stats.result.dynamic_create_infos;
            ++stats.result.arg_resets;
            stats.result.tensor_args_added += 4;
            stats.result.scalar_args_added += 3;
            if (!SubmitTask<Ops, Profile>(
                    state, worker, task_count, TaskKind::Sf, args, context, stats, pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(orchestration, TaskKind::Sf, context.result);

            BuildPvArgs(orchestration, args, batch);
            ++stats.result.arg_resets;
            stats.result.tensor_args_added += 4;
            stats.result.scalar_args_added += 2;
            if (!SubmitTask<Ops, Profile>(
                    state, worker, task_count, TaskKind::Pv, args, context, stats, pmu_context
                )) {
                break;
            }
            AcceptTaskOutputs(orchestration, TaskKind::Pv, context.result);

            BuildUpdateArgs(orchestration, args);
            ++stats.result.arg_resets;
            stats.result.tensor_args_added += 7;
            stats.result.scalar_args_added += 2;
            if (!SubmitTask<Ops, Profile>(
                    state, worker, task_count, TaskKind::Up, args, context, stats, pmu_context
                )) {
                break;
            }
#endif
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
    TraceAtomicFetchAdd<Ops>(
        stats.trace, stats.result, -1, AtomicSite::ReplayDoneIncrement,
        &state->replay_done.value, 1
    );
    const uint32_t final_poll_region = AtomicPollRegionBegin<Ops>(
        stats.trace, stats.result,
        TraceAtomicSiteMask(AtomicSite::ReplayDonePoll) | TraceAtomicSiteMask(AtomicSite::FaninFlagLoad)
    );
    while (true) {
        const uint32_t freed =
            DrainReady<Ops>(state, worker, DrainPlace::FinalDrain, stats);
        const bool all_replayed =
            LoadLine<Ops>(state->replay_done, stats, AtomicSite::ReplayDonePoll) >=
            static_cast<int64_t>(state->config.workers);
        // 必须同时满足“无人再生产新 slot”和“本核旧 slot 全部完成”，否则继续帮助系统推进 completion。
        if (all_replayed && worker.occupied_count == 0) {
            break;
        }
        if (freed == 0) {
            Ops::SpinHint();
        }
    }
    AtomicPollRegionEnd<Ops>(stats.trace, stats.result, final_poll_region);
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

#if defined(PA_LAZY_SAMPLE_SPLIT_FINISH)
    const uint64_t expected_callback_task_id_sum =
        static_cast<uint64_t>(task_count) * (task_count - 1U) / 2U;
    const bool split_protocol_ok =
        lazy_sample_split_runtime.scheduler == state && lazy_sample_split_runtime.worker == &worker &&
        lazy_sample_split_runtime.task_count == task_count && lazy_sample_split_runtime.worker_id == worker_id &&
        lazy_sample_split_runtime.owner_worker_id == worker_id &&
        lazy_sample_split_runtime.caller_state_address != 0 &&
        lazy_sample_split_runtime.finish_state_address == lazy_sample_split_runtime.caller_state_address &&
        lazy_sample_split_runtime.finish_calls == task_count &&
        lazy_sample_split_runtime.task_id_sum == expected_callback_task_id_sum &&
        lazy_sample_split_runtime.state_cookie == LazySampleSplitStateCookie(worker_id, role) &&
        lazy_sample_split_runtime.reserved == 0;
    if (!split_protocol_ok) {
        ++lazy_sample_split_runtime.protocol_errors;
        SetFatal<Ops>(state, stats);
    }
    stats.result.lazy_sample_split_caller_state_address = lazy_sample_split_runtime.caller_state_address;
    stats.result.lazy_sample_split_finish_state_address = lazy_sample_split_runtime.finish_state_address;
    stats.result.lazy_sample_split_finish_calls = lazy_sample_split_runtime.finish_calls;
    stats.result.lazy_sample_split_protocol_errors = lazy_sample_split_runtime.protocol_errors;
    stats.result.lazy_sample_split_state_cookie = lazy_sample_split_runtime.state_cookie;
    stats.result.lazy_sample_split_task_id_sum = lazy_sample_split_runtime.task_id_sum;
    stats.result.lazy_sample_split_owner_worker_id = lazy_sample_split_runtime.owner_worker_id;
    stats.result.lazy_sample_split_reserved = lazy_sample_split_runtime.reserved;
#endif

    // PA writes swimlane records through the ordinary GM cache and explicitly
    // cleans each worker's record range before the kernel finishes.
    FlushTraceCore<Ops>(stats.trace, stats.result);
    stats.result.finish_cycle = Ops::Now();
    stats.result.max_occupied = stats.max_occupied;
    stats.result.final_occupied = worker.occupied_count;
    stats.result.final_heap_next = worker.heap_next;
    stats.result.map_high_water = static_cast<uint32_t>(worker.map.high_water);
    stats.result.map_alive_floor = static_cast<uint32_t>(worker.map.alive_floor);
    stats.result.map_cleaned_upto = static_cast<uint32_t>(worker.map.cleaned_upto);
    stats.result.map_live_entries = CountLiveMapEntries(worker.map);
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
