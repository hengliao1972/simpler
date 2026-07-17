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

PA_DEVICE uint32_t CountBits(uint32_t value) {
    uint32_t count = 0;
    while (value != 0) {
        count += value & 1U;
        value >>= 1;
    }
    return count;
}

template <typename Ops>
PA_DEVICE int64_t LoadLine(PA_GM AtomicLine &line) {
    // Ops::Load 在 A5 后端是 atomicAdd(0)；返回值是该 RMW 线性化时观察到的共享值。
    return Ops::Load(&line.value);
}

template <typename Ops>
PA_DEVICE int32_t LoadLine(PA_GM AtomicFlagLine &line) {
    return Ops::Load(&line.value);
}

template <typename Ops>
PA_DEVICE void SetFatal(PA_GM SchedulerState *state) {
    // fatal 只从 0 单调置 1，重复 Exchange 不会把其他 worker 已观察到的失败状态清除。
    Ops::Exchange(&state->fatal.value, static_cast<int32_t>(1));
}

template <typename Ops>
PA_DEVICE bool IsFatal(PA_GM SchedulerState *state) {
    return LoadLine<Ops>(state->fatal) != 0;
}

template <typename Ops>
PA_DEVICE bool WatchdogExpired(PA_GM SchedulerState *state, uint64_t begin, uint32_t &polls) {
    // 每 1024 次自旋才读取系统计数器，降低正常启动屏障上的计时开销；超时后向所有 worker 广播 fatal。
    ++polls;
    if ((polls & 1023U) != 0 || Ops::Now() - begin <= kWatchdogTicks) {
        return false;
    }
    SetFatal<Ops>(state);
    return true;
}

template <typename Ops>
PA_DEVICE void AdvanceFrontier(PA_GM SchedulerState *state, LocalStats &stats) {
    // frontier 只表示“从 task 0 开始已经连续完成”的最高 task id，不能越过尚未发布 flag 的空洞。
    // 多个完成者可以同时扫描同一段连续区间，FetchMax 保证共享 frontier 只前进、不回退。
    int64_t frontier = LoadLine<Ops>(state->frontier);
    while (true) {
        const int64_t next = frontier + 1;
        if (next < 0 || next >= static_cast<int64_t>(kTaskCellCapacity)) {
            break;
        }
        if (Ops::Load(&state->tasks[next].flag) == 0) {
            break;
        }
        uint64_t retries = 0;
        // FetchMax 返回更新前的值；若其他核已经走得更远，就从其 old 值继续扫描，避免重复从 next 起步。
        const int64_t old = Ops::FetchMax(&state->frontier.value, next, retries);
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
    Ops::Exchange(&state->tasks[task_id].vend, worker.heap_next);
    Ops::StoreBarrier();
    Ops::Exchange(&state->tasks[task_id].flag, 1);
    AdvanceFrontier<Ops>(state, stats);
}

template <typename Ops>
PA_DEVICE bool SlotReady(PA_GM SchedulerState *state, PA_GM LocalSlot &slot, LocalStats &stats) {
    // 每个 fanin flag 都是跨核共享的完成条件；遇到第一个未就绪依赖即返回，后续 drain 会再次轮询。
    for (uint32_t index = 0; index < slot.fanin_count; ++index) {
        ++stats.result.fanin_loads;
        if (Ops::Load(&state->tasks[slot.fanin[index]].flag) == 0) {
            return false;
        }
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
    // slot 属于当前 worker；只有其全部跨核 fanin 已 ready 时才执行 NOP 模拟的 kernel、发布完成并释放 slot。
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
        const uint64_t kernel_begin = Ops::Now();
        Ops::Nop(NopCountForKind(state->config.nops, kind));
        const uint64_t kernel_end = Ops::Now();
        WriteTrace<false>(
            stats.trace, stats.result, static_cast<int32_t>(slot.task_id), static_cast<int32_t>(slot.kind),
            TracePhase::Kernel, ProfilePhase::ReplayTail, kernel_begin, kernel_end
        );
        RecordKernelCycles(stats, kind, kernel_end - kernel_begin);
        CompleteTask<Ops>(state, worker, slot.task_id, stats);
        const uint64_t commit_cycle = Ops::Now();
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
    const uint64_t wait_begin = Ops::Now();
    bool waited = false;
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
    if (waited) {
        ++stats.result.wait_events[0];
        const uint64_t wait_end = Ops::Now();
        WriteTrace<Profile>(
            stats.trace, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::RingBp,
            ProfilePhase::WaitForSlot, wait_begin, wait_end, 0, 0
        );
    } else if constexpr (Profile) {
        AccumulatePhase<true>(stats.result, ProfilePhase::WaitForSlot, wait_begin, Ops::Now());
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
    const uint64_t wait_begin = Ops::Now();
    bool waited = false;
    // 正常出口是 heap_next-vend 落入一个 ring；检测到不可能释放的覆盖或其他核 fatal 时返回失败。
    while (!IsFatal<Ops>(state)) {
        const int64_t frontier = LoadLine<Ops>(state->frontier);
        const int64_t retire = frontier - static_cast<int64_t>(state->heap_window);
        const uint64_t vend = retire < 0 ? 0 : Ops::Load(&state->tasks[retire].vend);
        if (worker.heap_next - vend <= ring) {
            if (waited) {
                ++stats.result.wait_events[1];
                const uint64_t wait_end = Ops::Now();
                WriteTrace<Profile>(
                    stats.trace, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::RingBp,
                    ProfilePhase::HeapGuard, wait_begin, wait_end, 0, 1
                );
            } else if constexpr (Profile) {
                AccumulatePhase<true>(stats.result, ProfilePhase::HeapGuard, wait_begin, Ops::Now());
            }
            return true;
        }
        if (frontier >= static_cast<int64_t>(task_id) - 1) {
            SetFatal<Ops>(state);
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
    if (waited) {
        ++stats.result.wait_events[1];
        const uint64_t wait_end = Ops::Now();
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
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_id, TaskKind kind
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
    const int64_t old = Ops::FetchMax(&cursor->value, static_cast<int64_t>(task_id), outcome.retries);
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
        SetFatal<Ops>(state);
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

template <typename Ops, bool Profile>
PA_DEVICE bool SubmitTask(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker, uint32_t task_count, TaskKind kind,
    const TaskArgs &args, SubmitContext &context, LocalStats &stats
) {
    // 每个 worker 都完整回放相同 task stream。主流程为：EfDrain -> materialize -> TensorMap retire
    // -> Claim -> winner 收集 fanin -> 全员 register -> winner Build / loser Replay。Alloc 在 Claim 前 register，
    // 且 winner 不入 kernel slot，而是在 heap guard 后直接发布完成。
    BeginSubmit(worker, args, context);
    const uint32_t task_id = static_cast<uint32_t>(context.task_id);
    const uint64_t submit_begin = Ops::Now();
    if (task_id == 0) {
        stats.result.submit_begin = submit_begin;
    }

    ResetTraceLap<Ops>(worker);
    // EfDrain 在当前 Submit 的参数物化前执行上一批已就绪 slot，是绝大多数 kernel 的正常落点。
    DrainReady<Ops>(state, worker, DrainPlace::EfDrain, stats);
    WriteTraceLap<Ops, Profile>(
        stats.trace, worker, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::EfDrain,
        ProfilePhase::EfDrain
    );

    // dist_submit_materialize_and_prepare_map resets the lap origin before its
    // two independently traced spans. Build/Replay later consumes this origin.
    // lap 起点在 materialize 前重置；Materialize/PrepareMap 各自取独立绝对区间，而后续
    // Build/Replay 会从这个起点形成覆盖式 span。因此泳道上的这些阶段不能直接相加。
    ResetTraceLap<Ops>(worker);
    const uint64_t materialize_begin = Ops::Now();
    if (!MaterializeTask(worker, task_id, args, context, state->heap_base, state->heap_size)) {
        SetFatal<Ops>(state);
        return false;
    }
    const uint64_t materialize_end = Ops::Now();
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::Materialize,
        ProfilePhase::Materialize, materialize_begin, materialize_end, 0,
        kind == TaskKind::Alloc ? 1U : 0U
    );
    stats.result.materialized_outputs += context.result.count;

    const uint64_t prepare_begin = Ops::Now();
    AdvanceTensorMap(worker.map, task_id, static_cast<int32_t>(state->heap_window));
    const uint64_t prepare_end = Ops::Now();
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::PrepareMap,
        ProfilePhase::PrepareMap, prepare_begin, prepare_end, 0, kind == TaskKind::Alloc ? 1U : 0U
    );

    bool winner = false;
    int32_t function_id = -1;

    if (kind == TaskKind::Alloc) {
        // Alloc 没有 kernel lane，96 个 worker 都维护本地物化/heap 状态，但只有 Claim winner 发布全局完成。
        const uint64_t register_begin = Ops::Now();
        RegisterOutputs(context, args, false);
        const uint64_t register_end = Ops::Now();
        WriteTrace<Profile>(
            stats.trace, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::Register,
            ProfilePhase::Register, register_begin, register_end, 0, 0
        );

        const uint64_t claim_begin = Ops::Now();
        const ClaimOutcome claim = Claim<Ops>(state, worker, task_id, kind);
        winner = claim.won;
        context.won = winner;
        context.kernel_id = claim.function_id;
        const uint64_t claim_end = Ops::Now();
        WriteTrace<Profile>(
            stats.trace, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::Claim,
            ProfilePhase::Claim, claim_begin, claim_end, winner ? 1U : 0U, 1
        );
        RecordClaimOutcome(stats, kind, claim);
        if (winner) {
            if (!HeapGuard<Ops, Profile>(
                    state, worker, task_id, context.output_bytes, stats
                )) {
                return false;
            }
            CompleteTask<Ops>(state, worker, task_id, stats);
            WriteTraceLap<Ops, false>(
                stats.trace, worker, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::Alloc,
                ProfilePhase::ReplayTail
            );
        } else {
            // Replay 表示该 worker 输掉 Claim；前面的物化、TensorMap 和 register 仍已执行，以保持本地状态同步。
            WriteTraceLap<Ops, false>(
                stats.trace, worker, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::Replay,
                ProfilePhase::ReplayTail
            );
        }
    } else {
        const uint64_t claim_begin = Ops::Now();
        const ClaimOutcome claim = Claim<Ops>(state, worker, task_id, kind);
        winner = claim.won;
        function_id = claim.function_id;
        context.won = winner;
        context.kernel_id = function_id;
        const uint64_t claim_end = Ops::Now();
        WriteTrace<Profile>(
            stats.trace, stats.result, static_cast<int32_t>(task_id), function_id, TracePhase::Claim,
            ProfilePhase::Claim, claim_begin, claim_end, winner ? 1U : 0U, 0
        );
        RecordClaimOutcome(stats, kind, claim);

        if (winner) {
            const uint64_t fanin_begin = Ops::Now();
            context.fanin_count = static_cast<int32_t>(CollectFanin(worker.map, args, context.fanin));
            const uint64_t fanin_end = Ops::Now();
            WriteTrace<Profile>(
                stats.trace, stats.result, static_cast<int32_t>(task_id), function_id, TracePhase::Fanin,
                ProfilePhase::Fanin, fanin_begin, fanin_end, 0, static_cast<uint32_t>(context.fanin_count)
            );
            stats.result.map_lookups += static_cast<uint32_t>(args.tensor_count) - context.result.count;
        }

        const uint64_t register_begin = Ops::Now();
        RegisterOutputs(context, args, true);
        const uint64_t register_end = Ops::Now();
        WriteTrace<Profile>(
            stats.trace, stats.result, static_cast<int32_t>(task_id), function_id, TracePhase::Register,
            ProfilePhase::Register, register_begin, register_end, 0, 1
        );
        stats.result.map_inserts += CountBits(context.register_mask);

        if (winner) {
            WriteTraceLap<Ops, false>(
                stats.trace, worker, stats.result, static_cast<int32_t>(task_id), function_id, TracePhase::Build,
                ProfilePhase::ReplayTail
            );
            if (!BuildWinner<Ops, Profile>(
                    state, worker, task_id, kind, args, context, context.fanin,
                    static_cast<uint32_t>(context.fanin_count), stats
                )) {
                return false;
            }
        } else {
            // 非 winner 不占用私有 ring slot，也不执行 kernel；Replay marker 覆盖本次前端回放的尾段。
            WriteTraceLap<Ops, false>(
                stats.trace, worker, stats.result, static_cast<int32_t>(task_id), -1, TracePhase::Replay,
                ProfilePhase::ReplayTail
            );
            // drain_block_won() is a local boolean early-return for this
            // single-lane PA graph, so it intentionally performs no GM access.
        }
    }

    ++stats.result.submits;
    const uint64_t submit_end = Ops::Now();
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id), function_id, TracePhase::Submit,
        ProfilePhase::Submit, submit_begin, submit_end, winner ? 1U : 0U, kind == TaskKind::Alloc ? 1U : 0U
    );
    if (task_id + 1 == task_count) {
        stats.result.submit_end = submit_end;
    }
    return true;
}

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
    PA_PUBLISH_FIELD(fanin_loads);
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

    LocalStats stats{};
    stats.result.worker_id = worker_id;
    stats.result.role = static_cast<uint32_t>(role);
    stats.result.checksum = 0xcbf29ce484222325ULL ^ worker_id;
    stats.trace = AttachTrace<Ops>(state, worker, worker_id);

    // 96 个参与者全部完成本地状态初始化后再进入 task 0，主要用于压低启动偏斜对
    // winner 分布和 Submit 时序的干扰；atomicMax 的唯一 winner 正确性本身不依赖该屏障。
    Ops::FetchAdd(&state->started_count.value, 1);
    const uint64_t start_wait = Ops::Now();
    uint32_t start_polls = 0;
    // 全员到齐或任一核发布 fatal 即退出启动等待；watchdog 防止缺失参与者造成永久挂死。
    while (LoadLine<Ops>(state->started_count) < static_cast<int64_t>(state->config.workers) &&
           !IsFatal<Ops>(state)) {
        Ops::SpinHint();
        if (WatchdogExpired<Ops>(state, start_wait, start_polls)) {
            break;
        }
    }

    const uint32_t batches = state->config.batches;
    const uint32_t task_count = batches * kTasksPerBatch;
    PaOrchestrationState orchestration;
    TaskArgs args;
    SubmitContext context;
    if (!IsFatal<Ops>(state)) {
        // Case1 每个 batch 固定回放 Alloc/QK/SF/PV/UP 五个 task；所有 worker 顺序相同，执行 lane 由 Claim 筛选。
        ResetTraceLap<Ops>(worker);
        InitPaOrchestration(orchestration, batches, &state->context_lens[0]);
        for (uint32_t batch = 0; batch < batches; ++batch) {
            BuildAllocArgs(orchestration, args, batch);
            ++stats.result.context_reads;
            stats.result.views_created += 2;
            stats.result.tensor_args_added += 3;
            if (!SubmitTask<Ops, Profile>(
                    state, worker, task_count, TaskKind::Alloc, args, context, stats
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
                    state, worker, task_count, TaskKind::Qk, args, context, stats
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
                    state, worker, task_count, TaskKind::Sf, args, context, stats
                )) {
                break;
            }
            AcceptTaskOutputs(orchestration, TaskKind::Sf, context.result);

            BuildPvArgs(orchestration, args, batch);
            ++stats.result.arg_resets;
            stats.result.tensor_args_added += 4;
            stats.result.scalar_args_added += 2;
            if (!SubmitTask<Ops, Profile>(
                    state, worker, task_count, TaskKind::Pv, args, context, stats
                )) {
                break;
            }
            AcceptTaskOutputs(orchestration, TaskKind::Pv, context.result);

            BuildUpdateArgs(orchestration, args);
            ++stats.result.arg_resets;
            stats.result.tensor_args_added += 7;
            stats.result.scalar_args_added += 2;
            if (!SubmitTask<Ops, Profile>(
                    state, worker, task_count, TaskKind::Up, args, context, stats
                )) {
                break;
            }
        }
    }

    // replay_done 表示所有 worker 已退出回放循环（成功路径即完整提交）；之后仍需 drain 到本核 slot 为空。
    Ops::FetchAdd(&state->replay_done.value, 1);
    while (true) {
        const uint32_t freed =
            DrainReady<Ops>(state, worker, DrainPlace::FinalDrain, stats);
        const bool all_replayed =
            LoadLine<Ops>(state->replay_done) >= static_cast<int64_t>(state->config.workers);
        // 必须同时满足“无人再生产新 slot”和“本核旧 slot 全部完成”，否则继续帮助系统推进 completion。
        if (all_replayed && worker.occupied_count == 0) {
            break;
        }
        if (freed == 0) {
            Ops::SpinHint();
        }
    }

    // PA writes swimlane records through the ordinary GM cache and explicitly
    // cleans each worker's record range before the kernel finishes.
    FlushTraceCore<Ops>(stats.trace);
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
    // Profile 作为编译期模板参数，只在显式开启时保留阶段累计代码，关闭时不在热路径增加运行时分支。
    if (state->config.profile_phases != 0) {
        RunSchedulerImpl<Ops, true>(state, worker_id, role);
    } else {
        RunSchedulerImpl<Ops, false>(state, worker_id, role);
    }
}

}  // namespace pa_scheduler

#endif  // PA_SCHEDULER_COMMON_PA_SCHEDULER_CORE_H
