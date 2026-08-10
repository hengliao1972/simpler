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

#ifndef PA_SCHEDULER_AICPU_PLAN_CPU_PRODUCER_H
#define PA_SCHEDULER_AICPU_PLAN_CPU_PRODUCER_H

#include "../common/aicpu_plan_pa_adapter.h"

#include <cstdint>

namespace pa_scheduler::cpu_plan {

struct ProductionResult {
    bool ok = false;
    uint32_t task_count = 0U;
    uint32_t metadata_count = 0U;
    uint32_t aic_count = 0U;
    uint32_t aiv_count = 0U;
};

inline aicpu_plan::EngineClass EngineForKind(TaskKind kind) {
    if (kind == TaskKind::Alloc) {
        return aicpu_plan::EngineClass::MetadataOnly;
    }
    return kind == TaskKind::Qk || kind == TaskKind::Pv
        ? aicpu_plan::EngineClass::Aic
        : aicpu_plan::EngineClass::Aiv;
}

inline bool BuildCallbackArgs(
    TaskKind kind, PaOrchestrationState &orchestration,
    TaskArgs &args, uint32_t batch, LocalStats &stats
) {
    switch (kind) {
        case TaskKind::Alloc:
            return BuildCallbackSubmitArgs<TaskKind::Alloc>(
                orchestration, args, batch, stats
            );
        case TaskKind::Qk:
            return BuildCallbackSubmitArgs<TaskKind::Qk>(
                orchestration, args, batch, stats
            );
        case TaskKind::Sf:
            return BuildCallbackSubmitArgs<TaskKind::Sf>(
                orchestration, args, batch, stats
            );
        case TaskKind::Pv:
            return BuildCallbackSubmitArgs<TaskKind::Pv>(
                orchestration, args, batch, stats
            );
        case TaskKind::Up:
            return BuildCallbackSubmitArgs<TaskKind::Up>(
                orchestration, args, batch, stats
            );
        case TaskKind::Count:
            return false;
    }
    return false;
}

// CPU planner 与真实 AICPU 后端承担相同的唯一 producer 角色。它调用
// PA callback 产生 TaskArgs，再经 canonical adapter 写入 PlanCell；task_id
// 只由 callback 成功发布次数单调生成，绝不从 task_id%5 或 Host descriptor
// 表反推身份。
template <typename ProducerOps>
inline bool PublishCallbackTask(
    const aicpu_plan::RuntimePlanView &view,
    PaOrchestrationState &orchestration, TaskArgs &args,
    LocalStats &stats, uint32_t batch, uint32_t batch_start,
    uint32_t group_index,
    TaskKind kind, bool has_following_group, bool is_last,
    ProductionResult &result
) {
    using namespace aicpu_plan;
    using namespace aicpu_plan_adapter;

    const uint32_t task_id = result.task_count;
    if (kind >= TaskKind::Count || task_id >= view.capacity) {
        return false;
    }

    SharedTaskOutputs outputs{};
    outputs.Reset(static_cast<int32_t>(task_id));
    if (!PrepareSharedTaskOutputs(
            outputs, static_cast<int32_t>(task_id), kind
        ) ||
        !BuildCallbackArgs(
            kind, orchestration, args, batch, stats
        )) {
        return false;
    }

    const uint8_t flags = EncodeSharedPaTaskMeta(
        kind, group_index, has_following_group, is_last
    );
    RuntimeTaskPlanSpec spec{};
    const EngineClass engine = EngineForKind(kind);
    if (flags == 0U || !MakePaRuntimeTaskPlanSpec(
            args, task_id, FunctionId(kind), engine, flags,
            batch_start, spec
        )) {
        return false;
    }
    const PaRuntimeTaskPlanSource source{args};
    if (PublishRuntimeTaskPlan<ProducerOps>(view, spec, source) !=
            PlanPublishResult::Published ||
        !AdvancePlannedFrontier<ProducerOps>(view, task_id)) {
        return false;
    }

    switch (engine) {
        case EngineClass::MetadataOnly:
            ++result.metadata_count;
            break;
        case EngineClass::Aic:
            ++result.aic_count;
            break;
        case EngineClass::Aiv:
            ++result.aiv_count;
            break;
    }
    ++result.task_count;
    AcceptTaskOutputs(orchestration, kind, outputs);
    return true;
}

// 首版是明确的 Plan-ahead：独立 planner 线程完整走完真实 PA callback
// 顺序并 Close，随后 Scalar 才开始中央 ticket Build。这个 CPU helper 只
// 是正式 AICPU 生命周期的协议回归，不参与 A5 性能计时。
template <typename ProducerOps>
inline ProductionResult ProducePaRuntimePlan(SchedulerState *state) {
    ProductionResult result{};
    if (state == nullptr || state->config.batches == 0U ||
        state->config.batches > kMaxBatches) {
        return result;
    }

    aicpu_plan::RuntimePlanView view{};
    if (!aicpu_plan::MakeRuntimePlanView(
            &state->runtime_plan_control,
            state->runtime_plan_storage, view
        )) {
        return result;
    }

    PaOrchestrationState orchestration{};
    InitPaOrchestration(
        orchestration, state->config.batches,
        &state->context_lens[0]
    );
    TaskArgs args{};
    LocalStats stats{};

    for (uint32_t batch = 0U; batch < state->config.batches; ++batch) {
        BeginPaBatchForCallback(orchestration, batch);
        if (orchestration.current_sequence > kSharedPaMaxContextLength) {
            return result;
        }

        const uint32_t batch_start = result.task_count;
        // LastSubmit 标记的是当前 batch 的最后一次 Submit，而不是整份
        // Plan 的末 task。空 batch 只有 Alloc，因此 Alloc 自己就是边界。
        const bool alloc_is_last =
            orchestration.current_blocks == 0U;
        if (!PublishCallbackTask<ProducerOps>(
                view, orchestration, args, stats, batch, batch_start, 0U,
                TaskKind::Alloc, false, alloc_is_last, result
            )) {
            return result;
        }

        uint32_t group_index = 0U;
        uint64_t block_offset = 0U;
        while (block_offset < orchestration.current_blocks) {
            if (group_index >= kSharedPaMaxBlockGroups) {
                return result;
            }
            PreparePaBlockGroup(orchestration, block_offset);
            if (orchestration.current_nblocks == 0U) {
                return result;
            }
            const bool has_following =
                block_offset + orchestration.current_nblocks <
                orchestration.current_blocks;
            if (!PublishCallbackTask<ProducerOps>(
                    view, orchestration, args, stats, batch, batch_start,
                    group_index, TaskKind::Qk, false, false, result
                ) ||
                !PublishCallbackTask<ProducerOps>(
                    view, orchestration, args, stats, batch, batch_start,
                    group_index, TaskKind::Sf, false, false, result
                ) ||
                !PublishCallbackTask<ProducerOps>(
                    view, orchestration, args, stats, batch, batch_start,
                    group_index, TaskKind::Pv, false, false, result
                )) {
                return result;
            }
            const bool up_is_last = !has_following;
            if (!PublishCallbackTask<ProducerOps>(
                    view, orchestration, args, stats, batch, batch_start,
                    group_index, TaskKind::Up, has_following,
                    up_is_last, result
                )) {
                return result;
            }
            block_offset += orchestration.current_nblocks;
            ++group_index;
        }
    }

    result.ok = aicpu_plan::CloseRuntimePlan<ProducerOps>(
        view, result.task_count
    );
    return result;
}

}  // namespace pa_scheduler::cpu_plan

#endif  // PA_SCHEDULER_AICPU_PLAN_CPU_PRODUCER_H
