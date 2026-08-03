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

}  // namespace

int main() {
    Check(CheckSequentialEquivalence(), "random-access args equal complete sequential G0/G1/G2/G4 replay");
    Check(CheckInvalidIdentityRejection(), "invalid batch bounds and task-local identities fail closed");
    std::printf("[RANDOM_ARGS_TEST] status=%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
