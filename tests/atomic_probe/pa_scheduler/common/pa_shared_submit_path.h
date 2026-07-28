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

#ifndef PA_SCHEDULER_COMMON_PA_SHARED_SUBMIT_PATH_H
#define PA_SCHEDULER_COMMON_PA_SHARED_SUBMIT_PATH_H

// 本文件由 pa_scheduler_core.h 在 pa_scheduler 命名空间内引入。shared
// TensorMap 的 Submit 控制流独立放在这里，避免继续把两套协议塞进同一
// 个宏分支密集的热函数；CPU 与 CCEC 仍复用相同的 Ops 和底层原语。

struct SharedTaskWriterDelta {
    // ordinary entry 先在 owner 私有上下文中完整准备；只有拿到 task 的
    // exact insert turn 后才批量预检和发布。symbol writer 不复制
    // descriptor，只记录本 task 是否需要更新不可变 writer history。
    SharedRegionValue ordinary_entries[kMaxTaskTensors];
    uint32_t ordinary_count;
    bool writer_intent_required;
};
static_assert(
    __is_trivially_constructible(SharedTaskWriterDelta),
    "shared task writer delta must remain trivial for CCEC local state"
);

PA_DEVICE bool PrepareSharedTaskWriterDelta(
    const TaskArgs &args, const SubmitContext &context,
    SharedTaskWriterDelta &delta
) {
    delta.ordinary_count = 0;
    delta.writer_intent_required = false;
    const int32_t task_id = context.task_id;
    if (!context.won || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks) ||
        args.has_error || args.tensor_count < 0 ||
        args.tensor_count > static_cast<int32_t>(kMaxTaskTensors) ||
        context.result.task_id != static_cast<uint64_t>(task_id) ||
        context.result.count > kSharedOutputMaxPerTask) {
        return false;
    }

    bool writer_required = false;
    if (!InspectSharedWriterIntent(args, writer_required) ||
        (writer_required &&
         !ValidateSharedWriterIntentSet(args, task_id))) {
        return false;
    }
    uint32_t expected_register_mask = 0;
    for (int32_t index = 0;
         index < args.tensor_count; ++index) {
        if (IsSharedWriterIntentTag(
                TaskTag(args, static_cast<uint32_t>(index))
            )) {
            expected_register_mask |=
                1U << static_cast<uint32_t>(index);
        }
    }
    if (context.register_mask != expected_register_mask) {
        return false;
    }
    uint32_t register_mask = context.register_mask;
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const uint32_t bit =
            1U << static_cast<uint32_t>(index);
        if ((register_mask & bit) == 0) {
            continue;
        }
        const TensorArgType tag =
            TaskTag(args, static_cast<uint32_t>(index));
        if (!IsSharedWriterIntentTag(tag)) {
            return false;
        }
        const TaskTensorRef &reference = args.tensors[index];
        if (reference.kind == TensorRefKind::SharedOutputRef) {
            if (!IsPlainSharedOutputRef(
                    SharedOutputReference(reference)
                )) {
                return false;
            }
        } else if (reference.kind == TensorRefKind::GmTensor) {
            PA_GM const TensorDesc &tensor =
                *reference.pointer.gm_tensor;
            if (!tensor.manual_dep) {
                if (delta.ordinary_count >= kMaxTaskTensors ||
                    !MakeValidatedSharedWriterRegion(
                        tensor, task_id,
                        delta.ordinary_entries[
                            delta.ordinary_count
                        ]
                    )) {
                    return false;
                }
                ++delta.ordinary_count;
            }
        } else if (reference.kind ==
                   TensorRefKind::LocalTensor) {
            const TensorDesc &tensor =
                *reference.pointer.local_tensor;
            if (!tensor.manual_dep) {
                if (delta.ordinary_count >= kMaxTaskTensors ||
                    !MakeValidatedSharedWriterRegion(
                        tensor, task_id,
                        delta.ordinary_entries[
                            delta.ordinary_count
                        ]
                    )) {
                    return false;
                }
                ++delta.ordinary_count;
            }
        } else {
            return false;
        }
        register_mask &= ~bit;
    }
    if (register_mask != 0) {
        return false;
    }
    delta.writer_intent_required = writer_required;
    return true;
}

template <typename Ops>
PA_DEVICE bool PublishSharedTaskWriterMetadata(
    PA_GM SchedulerState *state, const TaskArgs &args,
    const SubmitContext &context,
    const SharedTaskWriterDelta &delta, LocalStats &stats
) {
    const int32_t task_id = context.task_id;
    if (state == nullptr || !context.won || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks) ||
        delta.ordinary_count > kMaxTaskTensors ||
        Ops::Load(&state->fatal.value) != 0 ||
        !SharedCanPublishTaskCommit<Ops>(
            state->shared_map, task_id
        )) {
        if (state != nullptr) {
            SetFatal<Ops>(state, stats, task_id);
        }
        return false;
    }
    bool writer_required = false;
    if (!InspectSharedWriterIntent(args, writer_required) ||
        writer_required != delta.writer_intent_required ||
        (writer_required &&
         !ValidateSharedWriterIntentSet(args, task_id))) {
        SetFatal<Ops>(state, stats, task_id);
        return false;
    }
    for (uint32_t index = 0;
         index < delta.ordinary_count; ++index) {
        if (delta.ordinary_entries[index].producer != task_id) {
            SetFatal<Ops>(state, stats, task_id);
            return false;
        }
    }

    // insert-before-lookup 不能用本 task 的 reader_done 回收自己仍可能
    // 消费的 N-H。首版只使用已经证明正确的 -1 前沿；容量不足明确
    // 终止，绝不覆盖 live producer 或错误推进 task turn。
    if (SharedCheckTaskAppend<Ops>(
            state->shared_map, delta.ordinary_entries,
            delta.ordinary_count, -1
        ) != SharedAppendCheck::Ready) {
        SetFatal<Ops>(state, stats, task_id);
        return false;
    }

    int32_t ignored_fanin[kMaxFanin] = {};
    uint32_t ignored_fanin_count = 0;
    if (writer_required &&
        !CommitSymbolSharedWriterIntentSet<Ops>(
            state->shared_map, args, task_id,
            ignored_fanin, ignored_fanin_count, stats,
            &state->fatal.value
        )) {
        SetFatal<Ops>(state, stats, task_id);
        return false;
    }
    if (!SharedAppendPreparedTask<Ops>(
            state->shared_map, delta.ordinary_entries,
            delta.ordinary_count
        )) {
        SetFatal<Ops>(state, stats, task_id);
        return false;
    }
    stats.result.map_inserts += delta.ordinary_count;

    if (!PublishSharedTaskOutputs<Ops>(
            state->shared_map, context,
            static_cast<uint32_t>(task_id)
        )) {
        SetFatal<Ops>(state, stats, task_id);
        return false;
    }
    return true;
}

template <typename Ops>
PA_DEVICE bool HandoffSharedTaskInsertTurn(
    PA_GM SchedulerState *state, int32_t task_id, LocalStats &stats,
    int64_t &cas_observed
) {
    if (state == nullptr || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks)) {
        cas_observed = INT64_MIN;
        return false;
    }
    // ordinary payload 的 DCCI、symbol history/latest 与 fresh descriptor
    // 都必须先于 N->N+1 对其他 owner 可见。
    Ops::StoreBarrier();
    if (!SharedPublishTaskCommitAfterPreflightObserved<Ops>(
            state->shared_map, task_id, cas_observed
        )) {
        SetFatal<Ops>(state, stats, task_id);
        return false;
    }
    return true;
}

template <typename Ops>
PA_DEVICE bool PublishSharedTaskWriterDelta(
    PA_GM SchedulerState *state, const TaskArgs &args,
    const SubmitContext &context,
    const SharedTaskWriterDelta &delta, LocalStats &stats
) {
    if (!PublishSharedTaskWriterMetadata<Ops>(
            state, args, context, delta, stats
        )) {
        return false;
    }
    int64_t ignored_cas_observed = INT64_MIN;
    return HandoffSharedTaskInsertTurn<Ops>(
        state, context.task_id, stats, ignored_cas_observed
    );
}

template <typename Ops>
PA_DEVICE bool WaitForSharedTaskInsertTurn(
    PA_GM SchedulerState *state, int32_t task_id, LocalStats &stats,
    int64_t &ready_observed
) {
    ready_observed = -1;
    if (state == nullptr || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks)) {
        return false;
    }

    const uint64_t begin = Ops::Now();
    uint32_t polls = 0;
    while (true) {
        int64_t observed = -1;
        const SharedInsertTurnState turn_state =
            SharedInspectTaskTurnObserved<Ops, true>(
                state->shared_map, task_id, observed
            );
        if (turn_state == SharedInsertTurnState::Ready) {
            ready_observed = observed;
            return true;
        }
        if (turn_state ==
            SharedInsertTurnState::ProtocolError) {
            SetFatal<Ops>(state, stats, task_id);
            return false;
        }

        Ops::SpinHint();
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

template <typename Ops>
PA_DEVICE bool WaitForSharedTaskInsertTurn(
    PA_GM SchedulerState *state, int32_t task_id, LocalStats &stats
) {
    int64_t ignored_ready_observed = -1;
    return WaitForSharedTaskInsertTurn<Ops>(
        state, task_id, stats, ignored_ready_observed
    );
}

template <typename Ops, bool Profile, typename PmuContext>
PA_DEVICE bool FinishSharedWinnerSubmitBody(
    PA_GM SchedulerState *state, PA_GM WorkerState &worker,
    const TaskArgs &args, SubmitContext &context, LocalStats &stats,
    PmuContext &pmu_context, const CallbackSubmitTicket &ticket
) {
    const uint32_t task_id = ticket.task_id;
    SharedPaTaskMeta task_meta{};
    if (ticket.won == 0 ||
        !DecodeSharedPaTaskMeta(ticket.reserved, task_id, task_meta) ||
        !SharedPaFunctionIdMatches(
            task_meta.kind, true,
            static_cast<int32_t>(ticket.function_id)
        ) ||
        context.task_id != static_cast<int32_t>(task_id) ||
        !context.won ||
        Ops::Load(&state->fatal.value) != 0) {
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }

    const TaskKind kind = task_meta.kind;
    const int32_t function_id =
        static_cast<int32_t>(ticket.function_id);

    // Claim owner 先构造 descriptor 和 writer delta；这一段不查询
    // TensorMap，也不占用有序插入通道。
    const uint64_t materialize_begin =
        TraceTimestamp<Ops>(stats.trace, stats.result);
    BeginSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(
        pmu_context
    );
    const bool materialized = MaterializeTask<Ops, true>(
        worker, task_id, args, context, state->shared_map,
        state->heap_base, state->heap_size,
        kind, task_meta.batch_start, task_meta.group_index,
        &stats.trace, &stats.result
    );
    if (materialized) {
        stats.result.materialized_outputs += context.result.count;
    }
    if (!materialized) {
        EndSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(
            pmu_context
        );
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    SharedTaskWriterDelta writer_delta{};
    if (!PrepareSharedTaskWriterDelta(
            args, context, writer_delta
        )) {
        EndSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(
            pmu_context
        );
        SetFatal<Ops>(state, stats, static_cast<int32_t>(task_id));
        return false;
    }
    EndSubmitPmuPhase<SubmitPmuPhase::Materialize, Ops>(
        pmu_context
    );
    const uint64_t materialize_end =
        TraceTimestamp<Ops>(stats.trace, stats.result);
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id),
        function_id, TracePhase::Materialize,
        ProfilePhase::Materialize, materialize_begin,
        materialize_end, 0,
        kind == TaskKind::Alloc ? 1U : 0U
    );

    // 仅这一段全局串行：等 N-1 的 writer 元数据发布完，插入 N 的
    // ordinary/symbol/fresh-output 元数据，再把前沿从 N 推到 N+1。
    // 空写集合也必须推进，loser 完全不参与。
    const uint64_t register_begin = materialize_end;
    BeginSubmitPmuPhase<SubmitPmuPhase::Register, Ops>(
        pmu_context
    );
    int64_t ready_observed = -1;
    const bool turn_ready = WaitForSharedTaskInsertTurn<Ops>(
        state, static_cast<int32_t>(task_id), stats,
        ready_observed
    );
    // wait_end 对最后一次返回 Ready 的 atomic Load 建立数据依赖。只在
    // swimlane 构建读取 SYS_CNT；trace-free 的 PMU/性能构建预处理后为 0。
    const uint64_t metadata_begin = turn_ready
        ? TraceTimestampAfterAtomicResult<Ops>(
              stats.trace, stats.result, ready_observed
          )
        : TraceTimestamp<Ops>(stats.trace, stats.result);
    const bool metadata_published =
        turn_ready &&
        PublishSharedTaskWriterMetadata<Ops>(
            state, args, context, writer_delta, stats
        );
    const uint64_t metadata_end = metadata_published
        ? TraceTimestamp<Ops>(stats.trace, stats.result)
        : metadata_begin;

    int64_t cas_observed = INT64_MIN;
    const bool inserted =
        metadata_published &&
        HandoffSharedTaskInsertTurn<Ops>(
            state, static_cast<int32_t>(task_id), stats,
            cas_observed
        );
    // 正常路径的父区间终点依赖 CAS 返回值，表示本核已经取得 N->N+1
    // handoff 的返回结果；不加 DSB，也不把它解释成跨核全局可见时刻。
    const uint64_t register_end = metadata_published
        ? TraceTimestampAfterAtomicResult<Ops>(
              stats.trace, stats.result, cas_observed
          )
        : metadata_end;
    EndSubmitPmuPhase<SubmitPmuPhase::Register, Ops>(
        pmu_context
    );
    // raw 顺序保持父记录在前。唯一 detail 的两个端点把 Register 离线
    // 拆成 wait / metadata publish / handoff 三段，不逐 poll 扩张记录。
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id),
        function_id, TracePhase::Register,
        ProfilePhase::Register, register_begin, register_end,
        0, writer_delta.ordinary_count
    );
    WriteTrace<false>(
        stats.trace, stats.result, static_cast<int32_t>(task_id),
        function_id, TracePhase::SharedRegisterPublishMetadata,
        ProfilePhase::Register, metadata_begin, metadata_end
    );
    if (!inserted) {
        return false;
    }
#if defined(PA_TEST_SHARED_SUBMIT_HOOKS)
    // 仅供 CPU 定向并发门槛暂停某个 owner；正式 CPU/CCEC 构建预处理后
    // 不保留调用。测试借此证明 N+1 的 lookup/Build 不被 N 的 Build
    // 阶段串行化。
    Ops::AfterSharedTaskInsert(
        state, worker, task_id
    );
#endif

    // 前沿已经离开 N，后继 owner 可以插入 N+1；当前 owner 的 fanin
    // lookup、Build 和 slot 执行不再占住全局有序通道。
    uint64_t build_begin = register_end;
    if (kind != TaskKind::Alloc) {
        const uint64_t fanin_begin = register_end;
        bool lookup_protocol_ok = false;
        uint32_t ordinary_lookup_count = 0;
        // 本 task 的 writer 已经进入 history/latest；lookup 必须从
        // latest 沿不可变 history 回退到 max(writer < task_id)。这同时
        // 覆盖首组与后续组，不再依赖 PA 专用 chained-writer 特判。
        context.fanin_count = static_cast<int32_t>(
            CollectSharedFanin<Ops, false, true>(
                state->shared_map, args,
                static_cast<int32_t>(task_id),
                static_cast<int32_t>(state->heap_window),
                stats, context.fanin, lookup_protocol_ok,
                ordinary_lookup_count, &state->fatal.value
            )
        );
        if (!lookup_protocol_ok) {
            SetFatal<Ops>(
                state, stats, static_cast<int32_t>(task_id)
            );
            return false;
        }
        stats.result.map_lookups += ordinary_lookup_count;
        for (int32_t edge = 0;
             edge < context.fanin_count; ++edge) {
            stats.result.dependency_signature ^=
                DependencyEdgeSignature(
                    task_id,
                    static_cast<uint32_t>(context.fanin[edge])
                );
        }
        const uint64_t fanin_end =
            TraceTimestamp<Ops>(stats.trace, stats.result);
        WriteTrace<Profile>(
            stats.trace, stats.result,
            static_cast<int32_t>(task_id),
            function_id, TracePhase::Fanin,
            ProfilePhase::Fanin, fanin_begin, fanin_end, 0,
            static_cast<uint32_t>(context.fanin_count)
        );
        build_begin = fanin_end;
    }
    if (kind == TaskKind::Alloc) {
        CompleteTask<Ops>(state, worker, task_id, stats);
    } else if (!BuildWinner<Ops, Profile>(
                   state, worker, task_id, kind, args, context,
                   context.fanin,
                   static_cast<uint32_t>(context.fanin_count),
                   stats
               )) {
        return false;
    }
#if defined(PA_TEST_SHARED_SUBMIT_HOOKS)
    Ops::AfterSharedTaskBuild(
        state, worker, task_id, kind
    );
#endif
    const uint64_t build_end =
        TraceTimestamp<Ops>(stats.trace, stats.result);
    WriteTrace<false>(
        stats.trace, stats.result, static_cast<int32_t>(task_id),
        function_id,
        kind == TaskKind::Alloc
            ? TracePhase::AllocComplete
            : TracePhase::WinnerBuild,
        ProfilePhase::ReplayTail, build_begin, build_end
    );

    return CloseSharedCallbackSubmit<Ops, Profile>(
        state, stats, ticket, task_meta
    );
}

#endif  // PA_SCHEDULER_COMMON_PA_SHARED_SUBMIT_PATH_H
