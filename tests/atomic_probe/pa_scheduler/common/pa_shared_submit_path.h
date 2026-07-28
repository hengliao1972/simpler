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
    // descriptor，只记录本 task 需要更新的不可变 writer-history 数量。
    // prepared_task_id 只在全部结构检查通过后写入；Publish 阶段据此确认
    // 这份 owner-local delta 属于当前 task，不再在串行区重复扫描 args。
    SharedRegionValue ordinary_entries[kMaxTaskTensors];
    int32_t prepared_task_id;
    uint32_t ordinary_count;
    uint32_t symbol_count;
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
    delta.prepared_task_id = -1;
    delta.ordinary_count = 0;
    delta.symbol_count = 0;
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
                ) ||
                delta.symbol_count >= kMaxTaskTensors) {
                return false;
            }
            ++delta.symbol_count;
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
    // Inspect/Validate 与 delta 构造都只读取同一个 const TaskArgs；两者对
    // “是否需要自动登记”必须给出相同结论。该同步 Finish 控制流在
    // Publish 前不会修改 args，delta 也只存在于 winner 的本地栈上。
    const bool delta_requires_intent =
        delta.ordinary_count != 0 || delta.symbol_count != 0;
    if (writer_required != delta_requires_intent ||
        delta.ordinary_count + delta.symbol_count >
            kMaxTaskTensors) {
        return false;
    }
    delta.writer_intent_required = writer_required;
    delta.prepared_task_id = task_id;
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
        delta.prepared_task_id != task_id ||
        delta.ordinary_count > kMaxTaskTensors ||
        delta.symbol_count > kMaxTaskTensors ||
        delta.ordinary_count + delta.symbol_count >
            kMaxTaskTensors ||
        delta.writer_intent_required !=
            (delta.ordinary_count != 0 ||
             delta.symbol_count != 0) ||
        Ops::Load(&state->fatal.value) != 0) {
        if (state != nullptr) {
            SetFatal<Ops>(state, stats, task_id);
        }
        return false;
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
    if (delta.symbol_count != 0 &&
        !CommitSymbolSharedWriterIntentSet<Ops, false>(
            state->shared_map, args, task_id,
            ignored_fanin, ignored_fanin_count, stats,
            &state->fatal.value, delta.symbol_count
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
    return true;
}

PA_DEVICE void RecordCommittedSharedTaskWriterStats(
    const SharedTaskWriterDelta &delta, LocalStats &stats
) {
    // 这两个字段统计已经越过 task-level completion CAS 的完整事务。
    // metadata 已写入但 CAS 失败的 terminal task 不计入成功统计；故障现场
    // 仍由 fatal、共享元数据与 CAS observed value 保留。
    stats.result.map_inserts += delta.ordinary_count;
    stats.result.shared_symbol_inout_commits += delta.symbol_count;
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
    // 都必须先于本 task 的插入完成字对 N+1 owner 可见。每个 task 使用
    // 自己的 TaskCell，不再把一枚 baton 在 G 条 sidecar 线上轮换。
    Ops::StoreBarrier();
    cas_observed = Ops::CompareExchange(
        &state->tasks[static_cast<uint32_t>(task_id)]
             .deps_prepared,
        static_cast<int64_t>(-1),
        static_cast<int64_t>(task_id)
    );
    if (cas_observed != -1) {
        SetFatal<Ops>(state, stats, task_id);
        return false;
    }
    return true;
}

#if !PA_BUILD_TRACE_FREE
template <typename Ops>
PA_DEVICE bool TraceHandoffSharedTaskInsertTurn(
    PA_GM SchedulerState *state, int32_t task_id, LocalStats &stats,
    int64_t &cas_observed, uint64_t &cas_trace_begin,
    uint64_t &cas_trace_end
) {
    cas_trace_begin = 0;
    cas_trace_end = 0;
    if (state == nullptr || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks)) {
        cas_observed = INT64_MIN;
        return false;
    }
    Ops::StoreBarrier();
    cas_observed = CaptureAtomicCompareExchange<Ops>(
        stats.trace,
        &state->tasks[static_cast<uint32_t>(task_id)]
             .deps_prepared,
        static_cast<int64_t>(-1),
        static_cast<int64_t>(task_id),
        cas_trace_begin, cas_trace_end
    );
    if (cas_observed != -1) {
        SetFatal<Ops>(state, stats, task_id);
        return false;
    }
    return true;
}
#endif

template <typename Ops>
PA_DEVICE bool WaitForSharedTaskInsertTurn(
    PA_GM SchedulerState *state, int32_t task_id,
    LocalStats &stats, int64_t &ready_observed
);

template <typename Ops>
PA_DEVICE bool PublishSharedTaskWriterDelta(
    PA_GM SchedulerState *state, const TaskArgs &args,
    const SubmitContext &context,
    const SharedTaskWriterDelta &delta, LocalStats &stats
) {
    // fresh output cell 由本 task 的唯一 Claim winner 独占，不参与
    // ordinary/symbol 的 task-ID 串行插入。先发布 descriptor，再等待
    // predecessor；最终 deps_prepared handoff 仍同时封口两类发布。
    if (state == nullptr || !context.won || context.task_id < 0 ||
        context.task_id >= static_cast<int32_t>(kMaxTasks) ||
        Ops::Load(&state->fatal.value) != 0 ||
        !PublishSharedTaskOutputs<Ops>(
            state->shared_map, context,
            static_cast<uint32_t>(context.task_id)
        )) {
        if (state != nullptr) {
            SetFatal<Ops>(state, stats, context.task_id);
        }
        return false;
    }
    // 隔离测试和复用调用必须走与正式 Submit 相同的资格门：task 0
    // 直接进入，其余 task 只等待 N-1。Metadata helper 本身不再重复
    // atomic load，避免正式热路径把一次前驱等待测成三次原子访问。
    int64_t ignored_ready_observed = -1;
    if (!WaitForSharedTaskInsertTurn<Ops>(
            state, context.task_id, stats,
            ignored_ready_observed
        )) {
        RollbackSharedTaskOutputs<Ops>(
            state->shared_map.shared_outputs[
                static_cast<uint32_t>(context.task_id)
            ],
            context.result.count
        );
        return false;
    }
    if (!PublishSharedTaskWriterMetadata<Ops>(
            state, args, context, delta, stats
        )) {
        RollbackSharedTaskOutputs<Ops>(
            state->shared_map.shared_outputs[
                static_cast<uint32_t>(context.task_id)
            ],
            context.result.count
        );
        return false;
    }
    int64_t ignored_cas_observed = INT64_MIN;
    const bool inserted = HandoffSharedTaskInsertTurn<Ops>(
        state, context.task_id, stats, ignored_cas_observed
    );
    if (!inserted) {
        RollbackSharedTaskOutputs<Ops>(
            state->shared_map.shared_outputs[
                static_cast<uint32_t>(context.task_id)
            ],
            context.result.count
        );
        return false;
    }
    RecordCommittedSharedTaskWriterStats(delta, stats);
    return true;
}

template <typename Ops>
PA_DEVICE bool WaitForSharedTaskInsertTurn(
    PA_GM SchedulerState *state, int32_t task_id, LocalStats &stats,
    int64_t &ready_observed, uint64_t &load_count
) {
    ready_observed = -1;
    load_count = 0;
    if (state == nullptr || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks)) {
        return false;
    }

    // task 0 没有前驱，直接进入有序插入段。它完成后仍必须把自己的
    // deps_prepared 从 -1 发布为 0，供 task 1 建立真实跨核依赖。
    if (task_id == 0) {
        ready_observed = -1;
        return true;
    }

    PA_GM volatile int64_t *predecessor =
        &state->tasks[static_cast<uint32_t>(task_id - 1)]
             .deps_prepared;
    const uint64_t begin = Ops::Now();
    uint32_t polls = 0;
    while (true) {
        const int64_t observed = Ops::Load(predecessor);
        int64_t compare_observed = observed;
        int64_t dependency_observed = observed;
#if PA_BUILD_SWIMLANE && \
    (defined(PA_BUILD_AIC) || defined(PA_BUILD_AIV))
        // 分支判定和 SYS_CNT 依赖边界从同一个 atomic 返回寄存器派生，
        // 防止 CCEC 在 ready 分支把 ready_observed 常量折叠掉。
        compare_observed = Ops::ForkAtomicResultForBranch(
            observed, dependency_observed
        );
#endif
        if (compare_observed ==
            static_cast<int64_t>(task_id - 1)) {
            ready_observed = dependency_observed;
            load_count = static_cast<uint64_t>(polls) + 1;
            return true;
        }
        if (compare_observed != -1) {
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
    PA_GM SchedulerState *state, int32_t task_id, LocalStats &stats,
    int64_t &ready_observed
) {
    uint64_t ignored_load_count = 0;
    return WaitForSharedTaskInsertTurn<Ops>(
        state, task_id, stats, ready_observed,
        ignored_load_count
    );
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
#if PA_BUILD_TRACE_FREE
    const bool task_outputs_published =
        PublishSharedTaskOutputs<Ops>(
            state->shared_map, context, task_id
        );
#else
    uint64_t task_outputs_begin =
        TraceTimestamp<Ops>(stats.trace, stats.result);
    uint64_t task_outputs_copy_begin = task_outputs_begin;
    uint64_t task_outputs_copy_end = task_outputs_begin;
    uint64_t task_outputs_flush_begin = task_outputs_begin;
    uint64_t task_outputs_flush_end = task_outputs_begin;
    const bool task_outputs_published =
        PublishSharedTaskOutputs<Ops>(
            state->shared_map, context, task_id, &stats,
            &task_outputs_copy_begin, &task_outputs_copy_end,
            &task_outputs_flush_begin, &task_outputs_flush_end
        );
    const uint64_t task_outputs_end =
        TraceTimestamp<Ops>(stats.trace, stats.result);
#endif
    if (!task_outputs_published) {
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

    // 仅这一段全局串行：N>0 只等待 task[N-1].deps_prepared，随后插入
    // N 的 ordinary/symbol writer 元数据，再发布 task[N].deps_prepared。
    // fresh output descriptor 已在 Materialize 尾部按 task-cell 独占发布；
    // 空 writer 集合也必须推进，loser 完全不参与。
    const uint64_t register_begin = materialize_end;
    BeginSubmitPmuPhase<SubmitPmuPhase::Register, Ops>(
        pmu_context
    );
    int64_t ready_observed = -1;
    uint64_t insert_turn_load_count = 0;
    const bool turn_ready = WaitForSharedTaskInsertTurn<Ops>(
        state, static_cast<int32_t>(task_id), stats,
        ready_observed, insert_turn_load_count
    );
    // wait_end 对最后一次返回 Ready 的 atomic Load 建立数据依赖。只在
    // swimlane 构建读取 SYS_CNT；trace-free 的 PMU/性能构建预处理后为 0。
    const uint64_t metadata_begin =
        turn_ready && insert_turn_load_count != 0
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
#if PA_BUILD_TRACE_FREE
    const bool inserted =
        metadata_published &&
        HandoffSharedTaskInsertTurn<Ops>(
            state, static_cast<int32_t>(task_id), stats,
            cas_observed
        );
#else
    uint64_t cas_trace_begin = 0;
    uint64_t cas_trace_end = 0;
    const bool inserted =
        metadata_published &&
        TraceHandoffSharedTaskInsertTurn<Ops>(
            state, static_cast<int32_t>(task_id), stats,
            cas_observed, cas_trace_begin, cas_trace_end
        );
#endif
    // 正常路径的父区间终点依赖 CAS 返回值，表示本核已经取得
    // task[N].deps_prepared 的发布结果；不加 DSB，也不把它解释成
    // N+1 已经完成读取的时刻。
    const uint64_t register_end = metadata_published
        ? TraceTimestampAfterAtomicResult<Ops>(
              stats.trace, stats.result, cas_observed
          )
        : metadata_end;
    EndSubmitPmuPhase<SubmitPmuPhase::Register, Ops>(
        pmu_context
    );
    // 所有端点完成后再按业务顺序写 raw，避免写 trace 本身落入
    // Materialize/Register 的测量区间。Materialize 的 output detail
    // 还原独占 cell 发布；Register 只闭合 wait、writer metadata 与
    // handoff，不逐 poll 扩张记录。
    WriteTrace<Profile>(
        stats.trace, stats.result, static_cast<int32_t>(task_id),
        function_id, TracePhase::Materialize,
        ProfilePhase::Materialize, materialize_begin,
        materialize_end, 0,
        kind == TaskKind::Alloc ? 1U : 0U
    );
#if !PA_BUILD_TRACE_FREE
    if (task_outputs_published) {
        WriteTrace<false>(
            stats.trace, stats.result,
            static_cast<int32_t>(task_id), function_id,
            TracePhase::SharedMaterializePublishTaskOutputs,
            ProfilePhase::Materialize, task_outputs_begin,
            task_outputs_end
        );
        WriteTrace<false>(
            stats.trace, stats.result,
            static_cast<int32_t>(task_id), function_id,
            TracePhase::SharedMaterializePublishTaskOutputsCopy,
            ProfilePhase::Materialize, task_outputs_copy_begin,
            task_outputs_copy_end
        );
        WriteTrace<false>(
            stats.trace, stats.result,
            static_cast<int32_t>(task_id), function_id,
            TracePhase::SharedMaterializePublishTaskOutputsFlush,
            ProfilePhase::Materialize, task_outputs_flush_begin,
            task_outputs_flush_end
        );
    }
#endif
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
#if !PA_BUILD_TRACE_FREE
    // 记录动作延后到 Register 的全部时间端点之后，避免 64B raw 写入
    // 被误计入 metadata publish 或 handoff。循环内只累计本地 polls，
    // 每个成功 winner 固定至多增加这一条物理 PollBatch。
    if (turn_ready && insert_turn_load_count != 0) {
        (void)WriteAggregateAtomicPollBatch(
            stats.trace, stats.result,
            AtomicSite::SharedInsertTurnPoll,
            register_begin, metadata_begin,
            insert_turn_load_count,
            Ops::kAtomicReturnReadyObserved
        );
    }
    if (metadata_published && stats.trace.atomics_enabled) {
        WriteAtomicTrace<Ops>(
            stats.trace, stats.result,
            static_cast<int32_t>(task_id),
            AtomicSite::SharedInsertTurnHandoff,
            AtomicOp::CompareExchange,
            cas_trace_begin, cas_trace_end,
            true, Ops::kAtomicReturnReadyObserved
        );
    }
#endif
    if (!inserted) {
        // output cell 虽已在串行等待前短暂可见，但完成字尚未发布；失败路径
        // 恢复本 task 独占 cell，保留原有 fail-closed 终态。正常路径无额外
        // rollback 分支开销。
        RollbackSharedTaskOutputs<Ops>(
            state->shared_map.shared_outputs[task_id],
            context.result.count
        );
        return false;
    }
    // Register 的业务终点已经取完，失败分支也已经退出；成功统计因此
    // 既不污染串行区泳道，也不会把 completion CAS 失败的 metadata
    // 前缀误算成已提交事务。
    RecordCommittedSharedTaskWriterStats(
        writer_delta, stats
    );
#if defined(PA_TEST_SHARED_SUBMIT_HOOKS)
    // 仅供 CPU 定向并发门槛暂停某个 owner；正式 CPU/CCEC 构建预处理后
    // 不保留调用。测试借此证明 N+1 的 lookup/Build 不被 N 的 Build
    // 阶段串行化。
    Ops::AfterSharedTaskInsert(
        state, worker, task_id
    );
#endif

    // task[N] 的插入完成字已经发布，N+1 owner 可以进入有序插入段；
    // 当前 owner 的 fanin lookup、Build 和 slot 执行不再占住该链。
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
