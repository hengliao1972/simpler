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

#include "pa_scheduler_core.h"

#include <cstdio>

namespace {

using namespace pa_scheduler;

uint32_t g_failures = 0;

void Check(bool condition, const char *label) {
    std::printf("[RANDOM_ARGS_TEST] %-58s %s\n", label, condition ? "PASS" : "FAIL");
    if (!condition) {
        ++g_failures;
    }
}

bool SameTensorDesc(const TensorDesc &lhs, const TensorDesc &rhs) {
    bool same = lhs.buffer_addr == rhs.buffer_addr && lhs.buffer_size == rhs.buffer_size &&
                lhs.owner_task_id == rhs.owner_task_id && lhs.start_offset == rhs.start_offset &&
                lhs.version == rhs.version && lhs.ndims == rhs.ndims && lhs.dtype == rhs.dtype &&
                lhs.manual_dep == rhs.manual_dep && lhs.is_contiguous == rhs.is_contiguous &&
                lhs.child_memory == rhs.child_memory && lhs.extent_elem_cache == rhs.extent_elem_cache;
    for (uint32_t dim = 0; dim < kMaxTensorDims; ++dim) {
        same &= lhs.shapes[dim] == rhs.shapes[dim];
        same &= lhs.strides[dim] == rhs.strides[dim];
    }
    return same;
}

bool SameCreateInfo(const TensorCreateInfo &lhs, const TensorCreateInfo &rhs) {
    bool same = lhs.initial_value == rhs.initial_value && lhs.has_initial_value == rhs.has_initial_value &&
                lhs.reserved0 == rhs.reserved0 && lhs.start_offset == rhs.start_offset && lhs.version == rhs.version &&
                lhs.ndims == rhs.ndims && lhs.dtype == rhs.dtype && lhs.manual_dep == rhs.manual_dep &&
                lhs.is_contiguous == rhs.is_contiguous && lhs.child_memory == rhs.child_memory;
    for (uint32_t dim = 0; dim < kMaxTensorDims; ++dim) {
        same &= lhs.shapes[dim] == rhs.shapes[dim];
    }
    return same;
}

bool SameOutputRef(FdwicOutputRef lhs, FdwicOutputRef rhs) {
    return lhs.producer_task_id == rhs.producer_task_id && lhs.output_slot == rhs.output_slot &&
           lhs.flags == rhs.flags && lhs.view_ndims == rhs.view_ndims && lhs.view_shape0 == rhs.view_shape0 &&
           lhs.view_offset0 == rhs.view_offset0;
}

bool SameTensorRef(const TaskTensorRef &lhs, const TaskTensorRef &rhs) {
    if (lhs.kind != rhs.kind) {
        return false;
    }
    switch (lhs.kind) {
    case TensorRefKind::LocalTensor:
        return lhs.pointer.local_tensor != nullptr && rhs.pointer.local_tensor != nullptr &&
               SameTensorDesc(*lhs.pointer.local_tensor, *rhs.pointer.local_tensor);
    case TensorRefKind::GmTensor:
        return lhs.pointer.gm_tensor != nullptr && rhs.pointer.gm_tensor != nullptr &&
               SameTensorDesc(*lhs.pointer.gm_tensor, *rhs.pointer.gm_tensor);
    case TensorRefKind::CreateInfo:
        return lhs.pointer.create_info != nullptr && rhs.pointer.create_info != nullptr &&
               SameCreateInfo(*lhs.pointer.create_info, *rhs.pointer.create_info);
    case TensorRefKind::SharedOutputRef:
        return SameOutputRef(SharedOutputReference(lhs), SharedOutputReference(rhs));
    }
    return false;
}

bool SameTaskArgs(const TaskArgs &lhs, const TaskArgs &rhs) {
    if (lhs.tensor_count != rhs.tensor_count || lhs.scalar_count != rhs.scalar_count ||
        lhs.has_error != rhs.has_error || lhs.error_msg != rhs.error_msg ||
        lhs.launch_spec.core_num != rhs.launch_spec.core_num ||
        lhs.launch_spec.require_sync_start != rhs.launch_spec.require_sync_start ||
        lhs.explicit_deps != rhs.explicit_deps || lhs.explicit_dep_count != rhs.explicit_dep_count) {
        return false;
    }
    for (int32_t tensor = 0; tensor < lhs.tensor_count; ++tensor) {
        const uint32_t index = static_cast<uint32_t>(tensor);
        if (lhs.tags[index] != rhs.tags[index] || !SameTensorRef(lhs.tensors[index], rhs.tensors[index])) {
            return false;
        }
    }
    for (int32_t scalar = 0; scalar < lhs.scalar_count; ++scalar) {
        const uint32_t index = static_cast<uint32_t>(scalar);
        if (lhs.scalars[index] != rhs.scalars[index]) {
            return false;
        }
    }
    return true;
}

bool BuildArgsForKind(TaskKind kind, PaOrchestrationState &orch, TaskArgs &args, uint32_t batch, LocalStats &stats) {
    switch (kind) {
    case TaskKind::Alloc:
        return BuildCallbackSubmitArgs<TaskKind::Alloc>(orch, args, batch, stats);
    case TaskKind::Qk:
        return BuildCallbackSubmitArgs<TaskKind::Qk>(orch, args, batch, stats);
    case TaskKind::Sf:
        return BuildCallbackSubmitArgs<TaskKind::Sf>(orch, args, batch, stats);
    case TaskKind::Pv:
        return BuildCallbackSubmitArgs<TaskKind::Pv>(orch, args, batch, stats);
    case TaskKind::Up:
        return BuildCallbackSubmitArgs<TaskKind::Up>(orch, args, batch, stats);
    case TaskKind::Count:
        return false;
    }
    return false;
}

void PoisonDynamicState(PaOrchestrationState &orch, uint32_t task_id) {
    orch.current_sequence = UINT64_MAX;
    orch.current_blocks = UINT64_MAX;
    orch.current_block_offset = UINT64_MAX;
    orch.current_nblocks = UINT64_MAX;
    orch.current_valid_len = UINT64_MAX;
    orch.current_batch = UINT32_MAX;
    const uint32_t poison_task = task_id + 1U < kMaxTasks ? task_id + 1U : 0U;
    const FdwicOutputRef poison = MakePlainSharedOutputRef(poison_task, 0);
    orch.accumulated_output = poison;
    orch.accumulated_sum = poison;
    orch.accumulated_max = poison;
    orch.qk_scores = poison;
    orch.sf_probs = poison;
    orch.sf_max = poison;
    orch.sf_sum = poison;
    orch.pv_output = poison;
}

bool ActiveSharedRefsPrecedeTask(const TaskArgs &args, uint32_t task_id) {
    for (int32_t tensor = 0; tensor < args.tensor_count; ++tensor) {
        const TaskTensorRef &reference = args.tensors[static_cast<uint32_t>(tensor)];
        if (reference.kind != TensorRefKind::SharedOutputRef) {
            continue;
        }
        const FdwicOutputRef output_ref = SharedOutputReference(reference);
        if (!IsPlainSharedOutputRef(output_ref) || output_ref.producer_task_id < 0 ||
            static_cast<uint32_t>(output_ref.producer_task_id) >= task_id) {
            return false;
        }
    }
    return true;
}

bool CheckSequentialEquivalence() {
    constexpr uint32_t kBatches = 5;
    int32_t contexts[kBatches] = {
        0, 8192, 8193, 16384, 32768,
    };
    PaOrchestrationState sequential{};
    PaOrchestrationState random_access{};
    InitPaOrchestration(sequential, kBatches, contexts);
    InitPaOrchestration(random_access, kBatches, contexts);
    TaskArgs sequential_args{};
    TaskArgs random_args{};
    LocalStats sequential_stats{};
    LocalStats random_stats{};
    uint32_t task_cursor = 0;
    bool ok = true;

    for (uint32_t batch = 0; batch < kBatches; ++batch) {
        BeginPaBatchForCallback(sequential, batch);
        SharedPaBatchPlan plan{};
        ok &= BuildSharedPaBatchPlan(sequential.current_sequence, task_cursor, plan);
        if (!ok) {
            return false;
        }
        for (uint32_t offset = 0; offset < plan.task_count; ++offset) {
            SharedPaPlannedTask planned{};
            ok &= SharedPaPlannedTaskAt(plan, offset, planned);
            if (!ok) {
                return false;
            }
            const uint32_t task_id = task_cursor + offset;
            if (planned.kind == TaskKind::Qk) {
                PreparePaBlockGroup(sequential, static_cast<uint64_t>(planned.group_index) * kPaBlocksPerRequest);
            }
            ok &= BuildArgsForKind(planned.kind, sequential, sequential_args, batch, sequential_stats);

            PoisonDynamicState(random_access, task_id);
            ok &= BindSharedPaTaskForRandomAccess(
                random_access, batch, plan.batch_start, task_id, planned.kind, planned.group_index
            );
            ok &= BuildArgsForKind(planned.kind, random_access, random_args, batch, random_stats);
            ok &= SameTaskArgs(sequential_args, random_args);
            ok &= ActiveSharedRefsPrecedeTask(random_args, task_id);

            SharedTaskOutputs outputs{};
            outputs.Reset(static_cast<int32_t>(task_id));
            ok &= PrepareSharedTaskOutputs(outputs, static_cast<int32_t>(task_id), planned.kind);
            AcceptTaskOutputs(sequential, planned.kind, outputs);
        }
        task_cursor += plan.task_count;
    }
    ok &= task_cursor == 41;
    ok &= sequential_stats.result.arg_resets == random_stats.result.arg_resets;
    ok &= sequential_stats.result.views_created == random_stats.result.views_created;
    ok &= sequential_stats.result.dynamic_create_infos == random_stats.result.dynamic_create_infos;
    ok &= sequential_stats.result.tensor_args_added == random_stats.result.tensor_args_added;
    ok &= sequential_stats.result.scalar_args_added == random_stats.result.scalar_args_added;
    return ok;
}

bool CheckInvalidIdentityRejection() {
    int32_t context = 8192;
    PaOrchestrationState orch{};
    InitPaOrchestration(orch, 1, &context);
    return !BindSharedPaTaskForRandomAccess(orch, 1, 0, 0, TaskKind::Alloc, 0) &&
           !BindSharedPaTaskForRandomAccess(orch, 0, 1, 0, TaskKind::Alloc, 0) &&
           !BindSharedPaTaskForRandomAccess(orch, 0, 0, 1, TaskKind::Sf, 0) &&
           !BindSharedPaTaskForRandomAccess(orch, 0, 0, 4, TaskKind::Up, 1) &&
           !BindSharedPaTaskForRandomAccess(orch, 0, 0, 5, TaskKind::Count, 0);
}

bool CheckCompactDispatchIdentity() {
    constexpr uint32_t kBatches = 5;
    constexpr uint32_t kTasks = 41;
    int32_t contexts[kBatches] = {
        0, 8192, 8193, 16384, 32768,
    };
    SharedBuildDispatchState dispatch{};
    dispatch.task_count = kTasks;
    dispatch.batch_count = kBatches;
    uint32_t task_cursor = 0;
    uint32_t executable_tasks = 0;
    for (uint32_t batch = 0; batch < kBatches; ++batch) {
        SharedPaBatchPlan plan{};
        if (!BuildSharedPaBatchPlan(
                static_cast<uint64_t>(contexts[batch]),
                task_cursor, plan
            )) {
            return false;
        }
        for (uint32_t offset = 0;
             offset < plan.task_count; ++offset) {
            SharedPaPlannedTask planned{};
            if (!SharedPaPlannedTaskAt(
                    plan, offset, planned
                )) {
                return false;
            }
            const uint32_t task_id = task_cursor + offset;
            SharedBuildDispatchTaskIdentity &identity =
                dispatch.tasks[task_id];
            identity.batch = static_cast<uint16_t>(batch);
            identity.encoded_meta = EncodeSharedPaTaskMeta(
                planned.kind, planned.group_index,
                planned.has_following_group,
                task_id + 1U == kTasks
            );
            const bool executable =
                planned.kind != TaskKind::Alloc;
            const cross_core_dag::ExecEngineClass engine_class =
                !executable
                    ? cross_core_dag::ExecEngineClass::None
                    : (planned.kind == TaskKind::Qk ||
                       planned.kind == TaskKind::Pv
                           ? cross_core_dag::ExecEngineClass::Aic
                           : cross_core_dag::ExecEngineClass::Aiv);
            identity.exec_route =
                cross_core_dag::EncodeExecDispatchRoute(
                    executable, engine_class
                );
            executable_tasks += executable ? 1U : 0U;
        }
        task_cursor += plan.task_count;
    }
    if (task_cursor != kTasks) {
        return false;
    }
    dispatch.executable_task_count = executable_tasks;

    PaOrchestrationState orch{};
    InitPaOrchestration(orch, kBatches, contexts);
    for (uint32_t task_id = 0; task_id < kTasks; ++task_id) {
        SharedBuildDispatchTask task{};
        SharedExecDispatchRoute route{};
        if (!DecodeSharedBuildDispatchTask(
                dispatch, task_id, task
            ) ||
            !DecodeSharedExecDispatchRoute(
                dispatch, task_id, route
            ) ||
            task.task_id != task_id ||
            route.task_id != task_id ||
            route.executable != task.executable ||
            route.engine_class != task.engine_class ||
            !SharedPaDispatchRouteMatches(task) ||
            !BindSharedPaTaskForRandomAccess(
                orch, task.batch, task.meta.batch_start,
                task.task_id, task.meta.kind,
                task.meta.group_index
            )) {
            return false;
        }
    }

    SharedBuildDispatchTask ignored{};
    SharedExecDispatchRoute ignored_route{};
    dispatch.tasks[1].exec_route = 0;
    const bool rejects_missing_route =
        !DecodeSharedExecDispatchRoute(
            dispatch, 1, ignored_route
        ) &&
        !DecodeSharedBuildDispatchTask(dispatch, 1, ignored);
    dispatch.tasks[1].exec_route =
        cross_core_dag::EncodeExecDispatchRoute(
            true, cross_core_dag::ExecEngineClass::Aic
        );
    dispatch.tasks[1].batch = kBatches;
    const bool rejects_batch =
        !DecodeSharedBuildDispatchTask(dispatch, 1, ignored);
    dispatch.tasks[1].batch = 1;
    dispatch.tasks[kTasks - 1U].encoded_meta &=
        static_cast<uint8_t>(~kSharedPaTicketLastSubmit);
    const bool rejects_missing_last =
        DecodeSharedExecDispatchRoute(
            dispatch, kTasks - 1U, ignored_route
        ) &&
        !DecodeSharedBuildDispatchTask(
            dispatch, kTasks - 1U, ignored
        );

    // route 自身合法但与 PA kind 不一致时，公共执行解码仍成功；PA
    // adapter 必须在发布 execution cell 前单独拒绝这份业务计划。
    const uint8_t alloc_route = dispatch.tasks[0].exec_route;
    dispatch.tasks[0].exec_route =
        cross_core_dag::EncodeExecDispatchRoute(
            true, cross_core_dag::ExecEngineClass::Aic
        );
    const bool adapter_rejects_route_mismatch =
        DecodeSharedExecDispatchRoute(
            dispatch, 0, ignored_route
        ) &&
        DecodeSharedBuildDispatchTask(dispatch, 0, ignored) &&
        !SharedPaDispatchRouteMatches(ignored);
    dispatch.tasks[0].exec_route = alloc_route;

    return rejects_missing_route && rejects_batch &&
           rejects_missing_last &&
           adapter_rejects_route_mismatch &&
           !DecodeSharedBuildDispatchTask(
               dispatch, kTasks, ignored
           );
}

bool CheckG1DynamicDagClassification() {
    constexpr uint32_t kBatches = 4;
    constexpr uint32_t kTasksPerBatch = 5;
    constexpr uint32_t kTasks = kBatches * kTasksPerBatch;
    int32_t contexts[kBatches] = {
        8192, 8192, 8192, 8192,
    };
    SharedBuildDispatchState dispatch{};
    dispatch.task_count = kTasks;
    dispatch.batch_count = kBatches;
    dispatch.executable_task_count = kBatches * 4U;

    PaOrchestrationState orch{};
    InitPaOrchestration(orch, kBatches, contexts);
    TaskArgs args{};
    LocalStats stats{};
    uint32_t writers_by_kind[
        static_cast<uint32_t>(TaskKind::Count)
    ] = {};
    uint32_t dependency_tasks = 0;

    for (uint32_t batch = 0; batch < kBatches; ++batch) {
        SharedPaBatchPlan plan{};
        const uint32_t batch_start =
            batch * kTasksPerBatch;
        if (!BuildSharedPaBatchPlan(
                static_cast<uint64_t>(contexts[batch]),
                batch_start, plan
            ) ||
            plan.task_count != kTasksPerBatch) {
            return false;
        }
        for (uint32_t offset = 0;
             offset < plan.task_count; ++offset) {
            SharedPaPlannedTask planned{};
            const uint32_t task_id = batch_start + offset;
            if (!SharedPaPlannedTaskAt(plan, offset, planned) ||
                !BindSharedPaTaskForRandomAccess(
                    orch, batch, batch_start, task_id,
                    planned.kind, planned.group_index
                ) ||
                !BuildArgsForKind(
                    planned.kind, orch, args, batch, stats
                )) {
                return false;
            }
            SharedBuildDispatchTaskIdentity &identity =
                dispatch.tasks[task_id];
            identity.batch = static_cast<uint16_t>(batch);
            identity.encoded_meta = EncodeSharedPaTaskMeta(
                planned.kind, planned.group_index,
                planned.has_following_group,
                task_id + 1U == kTasks
            );
            const bool executable =
                planned.kind != TaskKind::Alloc;
            const cross_core_dag::ExecEngineClass engine =
                !executable
                    ? cross_core_dag::ExecEngineClass::None
                    : (planned.kind == TaskKind::Qk ||
                       planned.kind == TaskKind::Pv
                           ? cross_core_dag::ExecEngineClass::Aic
                           : cross_core_dag::ExecEngineClass::Aiv);
            identity.exec_route =
                cross_core_dag::EncodeExecDispatchRoute(
                    executable, engine
                );

            const SharedPaDagSchema schema{&dispatch};
            SharedMetadataDag dag{};
            if (!BuildSharedMetadataDagFromTaskArgs(
                    schema, task_id, args, dag
                ) ||
                dag.writer_count !=
                    (planned.kind == TaskKind::Up ? 3U : 0U) ||
                dag.ordinary_writer ||
                dag.ordinary_previous_writer != -1) {
                return false;
            }
            if (dag.writer_count != 0) {
                ++writers_by_kind[
                    static_cast<uint32_t>(planned.kind)
                ];
                for (uint32_t writer = 0;
                     writer < dag.writer_count; ++writer) {
                    if (dag.writer_previous[writer] !=
                        static_cast<int32_t>(batch_start)) {
                        return false;
                    }
                }
            }
            dependency_tasks +=
                dag.dependency_count != 0 ? 1U : 0U;
        }
    }

    // TaskArgs 权威路径必须在构建 DAG 的同一次 tensor 枚举中
    // 拒绝非法的 access/reference 组合，不能因删除独立 Validate
    // 遍历而只保留形式检查。循环结束时 args 对应最后一个 UP。
    const int32_t saved_tag = args.tags[0];
    args.tags[0] = static_cast<int32_t>(TensorArgType::Output);
    const SharedPaDagSchema final_schema{&dispatch};
    SharedMetadataDag invalid_dag{};
    const bool rejects_invalid_access =
        !BuildSharedMetadataDagFromTaskArgs(
            final_schema, kTasks - 1U, args, invalid_dag
        );
    args.tags[0] = saved_tag;
    if (!rejects_invalid_access) {
        return false;
    }

    std::printf(
        "[RANDOM_ARGS_TEST] G1 dynamic-DAG writers="
        "Alloc:%u,QK:%u,SF:%u,PV:%u,UP:%u dependency_tasks=%u\n",
        writers_by_kind[static_cast<uint32_t>(TaskKind::Alloc)],
        writers_by_kind[static_cast<uint32_t>(TaskKind::Qk)],
        writers_by_kind[static_cast<uint32_t>(TaskKind::Sf)],
        writers_by_kind[static_cast<uint32_t>(TaskKind::Pv)],
        writers_by_kind[static_cast<uint32_t>(TaskKind::Up)],
        dependency_tasks
    );
    return writers_by_kind[
               static_cast<uint32_t>(TaskKind::Alloc)
           ] == 0 &&
        writers_by_kind[
            static_cast<uint32_t>(TaskKind::Qk)
        ] == 0 &&
        writers_by_kind[
            static_cast<uint32_t>(TaskKind::Sf)
        ] == 0 &&
        writers_by_kind[
            static_cast<uint32_t>(TaskKind::Pv)
        ] == 0 &&
        writers_by_kind[
            static_cast<uint32_t>(TaskKind::Up)
        ] == kBatches &&
        dependency_tasks == 0;
}

}  // namespace

int main() {
    Check(CheckSequentialEquivalence(), "random-access args equal complete sequential G0/G1/G2/G4 replay");
    Check(CheckInvalidIdentityRejection(), "invalid batch bounds and task-local identities fail closed");
    Check(CheckCompactDispatchIdentity(), "compact immutable dispatch identities decode and bind each task");
    Check(
        CheckG1DynamicDagClassification(),
        "G1 derives per-symbol writers without a cross-batch metadata chain"
    );
    std::printf("[RANDOM_ARGS_TEST] status=%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
