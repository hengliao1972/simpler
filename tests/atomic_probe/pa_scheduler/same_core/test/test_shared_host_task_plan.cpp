/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License Agreement").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "winner_workload_host.h"

#include <cstdio>
#include <memory>
#include <string>

namespace {

using namespace pa_scheduler;
using namespace pa_scheduler::host;

bool Check(bool condition, const char *label) {
    std::printf(
        "[HOST_PLAN_TEST] %-56s %s\n",
        label, condition ? "PASS" : "FAIL"
    );
    return condition;
}

bool SetContextsAndBuild(
    SchedulerState *state, const int32_t *contexts,
    uint32_t batches, SharedHostTaskPlan *plan
) {
    state->config.batches = batches;
    for (uint32_t batch = 0; batch < batches; ++batch) {
        state->context_lens[batch] = contexts[batch];
    }
    std::string error;
    const bool ok =
        BuildSharedHostTaskPlan(*state, plan, &error);
    if (!ok) {
        std::fprintf(
            stderr, "BuildSharedHostTaskPlan failed: %s\n",
            error.c_str()
        );
    }
    return ok;
}

bool CheckSingleContext(
    SchedulerState *state, int32_t context,
    uint32_t expected_groups, uint32_t expected_tasks
) {
    SharedHostTaskPlan plan;
    bool ok = SetContextsAndBuild(
        state, &context, 1, &plan
    );
    ok &= plan.batch_count == 1;
    ok &= plan.total_groups == expected_groups;
    ok &= plan.total_tasks == expected_tasks;
    ok &= plan.tasks_by_kind[
        static_cast<uint32_t>(TaskKind::Alloc)
    ] == 1;
    for (uint32_t kind =
             static_cast<uint32_t>(TaskKind::Qk);
         kind <= static_cast<uint32_t>(TaskKind::Up);
         ++kind) {
        ok &= plan.tasks_by_kind[kind] == expected_groups;
    }
    const SharedHostBatchPlan *batch = plan.BatchAt(0);
    ok &= batch != nullptr;
    if (batch != nullptr) {
        ok &= batch->batch_start == 0;
        ok &= batch->group_count == expected_groups;
        ok &= batch->task_count == expected_tasks;
        ok &= batch->final_up_task_id ==
            (expected_groups == 0
                 ? UINT32_MAX
                 : expected_tasks - 1U);
    }
    const SharedHostPlannedTask *first = plan.TaskAt(0);
    const SharedHostPlannedTask *last =
        plan.TaskAt(expected_tasks - 1U);
    ok &= first != nullptr &&
          first->kind == TaskKind::Alloc;
    ok &= last != nullptr && last->is_last_in_batch;
    ok &= plan.TaskAt(expected_tasks) == nullptr;
    return ok;
}

bool CheckRealComputeActivityContract(SchedulerState *state) {
    bool ok = true;
    const int32_t g0_context = 0;
    SharedHostTaskPlan plan;
    ok &= SetContextsAndBuild(
        state, &g0_context, 1, &plan
    );
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        state->results[worker] = WorkerResult{};
        state->results[worker].role = static_cast<uint32_t>(
            worker < kAicWorkers ? CoreRole::Aic : CoreRole::Aiv
        );
    }

    WinnerWorkloadOptions workload;
    std::vector<float> workspace_image;
    std::vector<float> outputs;
    InitializeWinnerWorkloadBuffers(
        workload, &workspace_image, &outputs
    );
    std::fill(
        outputs.begin(), outputs.end(),
        winner_workload::kOutputSentinel
    );
    ok &= RealComputeActivityMatchesPlan(*state, 0);
    ok &= !RealComputeActivityMatchesPlan(*state, 1);
    ok &= ValidateRealComputeOutputs(
        *state, workload, outputs, 1
    );

    const int32_t g1_context = 8192;
    ok &= SetContextsAndBuild(
        state, &g1_context, 1, &plan
    );
    ok &= !RealComputeActivityMatchesPlan(*state, 0);
    ok &= RealComputeActivityMatchesPlan(*state, 1);
    return ok;
}

bool CheckTensorDescSemanticMatch() {
    TensorDesc expected{};
    expected.buffer_addr = kSyntheticHeapBase;
    expected.buffer_size = 64;
    expected.owner_task_id = 17;
    expected.ndims = 1;
    expected.dtype = DataType::Float32;
    expected.is_contiguous = true;
    expected.shapes[0] = 16;
    expected.strides[0] = 1;
    expected.extent_elem_cache = 16;

    TensorDesc actual = expected;
    actual.shapes[1] = 99;
    actual.strides[1] = 7;
    bool ok = TensorDescFieldsMatch(actual, expected);
    actual.strides[0] = 2;
    ok &= !TensorDescFieldsMatch(actual, expected);
    actual.strides[0] = 1;
    actual.ndims = kMaxTensorDims + 1U;
    ok &= !TensorDescFieldsMatch(actual, expected);
    return ok;
}

bool CheckHeapAdmission(SchedulerState *state) {
    bool ok = true;
    const auto check_repeated = [&](
        int32_t context, uint32_t batches,
        bool expected_admitted, uint64_t expected_total,
        uint64_t expected_max_shard, uint64_t heap_size
    ) {
        std::vector<int32_t> contexts(batches, context);
        SharedHostTaskPlan plan;
        bool case_ok = SetContextsAndBuild(
            state, contexts.data(), batches, &plan
        );
        uint64_t planned_by_shard[kSharedHeapShards] = {};
        uint64_t planned_total = 0;
        for (const SharedHostPlannedTask &task : plan.tasks) {
            if (task.output_bytes == 0) {
                continue;
            }
            const uint64_t reserve =
                (task.output_bytes + kOutputAlignment - 1U) /
                kOutputAlignment * kOutputAlignment;
            planned_by_shard[
                task.task_id % kSharedHeapShards
            ] += reserve;
            planned_total += reserve;
        }
        uint64_t planned_max_shard = 0;
        for (uint32_t shard = 0;
             shard < kSharedHeapShards; ++shard) {
            planned_max_shard = std::max(
                planned_max_shard, planned_by_shard[shard]
            );
        }
        case_ok &= plan.canonical_heap_bytes == expected_total;
        case_ok &= planned_total == expected_total;
        case_ok &= planned_max_shard == expected_max_shard;
        SharedHostHeapAdmission admission;
        std::string error;
        const bool admitted = case_ok &&
            ValidateSharedHostHeapAdmission(
                plan, heap_size, &admission, &error
            );
        case_ok &= admitted == expected_admitted;
        case_ok &= admission.admitted == expected_admitted;
        if (expected_admitted) {
            uint64_t maximum_shard = 0;
            for (uint32_t shard = 0;
                 shard < kSharedHeapShards; ++shard) {
                maximum_shard = std::max(
                    maximum_shard,
                    admission.reserved_bytes_by_shard[shard]
                );
            }
            case_ok &=
                admission.total_reserved_bytes ==
                expected_total;
            case_ok &= maximum_shard == expected_max_shard;
            case_ok &=
                admission.first_failed_task == UINT32_MAX;
            case_ok &=
                admission.first_failed_shard == UINT32_MAX;
        } else {
            case_ok &= !error.empty();
            case_ok &=
                admission.first_failed_task != UINT32_MAX;
            case_ok &=
                admission.first_failed_shard <
                kSharedHeapShards;
        }
        return case_ok;
    };

    ok &= check_repeated(
        0, 1, true, 10240, 10240, kHeapBytes
    );
    ok &= check_repeated(
        8192, 1, true, 806912, 524288, kHeapBytes
    );
    ok &= check_repeated(
        8193, 1, true, 829440, 524288, kHeapBytes
    );
    ok &= check_repeated(
        16384, 1, true, 1603584, 524288, kHeapBytes
    );
    ok &= check_repeated(
        32768, 1, true, 3196928, 1048576, kHeapBytes
    );
    ok &= check_repeated(
        0, kDefaultBatches, true,
        2621440, 327680, kHeapBytes
    );
    ok &= check_repeated(
        8192, kDefaultBatches, true,
        206569472, 25821184, kHeapBytes
    );
    ok &= check_repeated(
        8193, kDefaultBatches, true,
        212336640, 26542080, kHeapBytes
    );
    ok &= check_repeated(
        16384, kDefaultBatches, false,
        410517504, 51314688, kHeapBytes
    );
    ok &= check_repeated(
        32768, kDefaultBatches, false,
        818413568, 102301696, kHeapBytes
    );
    ok &= check_repeated(
        0, kMaxBatches, true,
        5242880, 655360, kExtendedBatchHeapBytes
    );
    ok &= check_repeated(
        8192, kMaxBatches, true,
        413138944, 51642368, kExtendedBatchHeapBytes
    );

    const int32_t mixed_contexts[] = {
        0, 8192, 8193, 32768,
    };
    SharedHostTaskPlan mixed;
    ok &= SetContextsAndBuild(
        state, mixed_contexts, 4, &mixed
    );
    SharedHostHeapAdmission mixed_admission;
    std::string mixed_error;
    ok &= ValidateSharedHostHeapAdmission(
        mixed, kHeapBytes, &mixed_admission, &mixed_error
    );
    const uint64_t expected_mixed_shards[
        kSharedHeapShards
    ] = {
        1323008, 546816, 540672, 272384,
        1062912, 536576, 26624, 534528,
    };
    ok &= mixed_admission.total_reserved_bytes == 4843520;
    for (uint32_t shard = 0;
         shard < kSharedHeapShards; ++shard) {
        ok &=
            mixed_admission.reserved_bytes_by_shard[shard] ==
            expected_mixed_shards[shard];
    }

    std::vector<int32_t> g1_contexts(
        kDefaultBatches, 8192
    );
    SharedHostTaskPlan g1;
    ok &= SetContextsAndBuild(
        state, g1_contexts.data(), kDefaultBatches, &g1
    );
    SharedHostHeapAdmission exact;
    std::string exact_error;
    ok &= ValidateSharedHostHeapAdmission(
        g1, 206569472, &exact, &exact_error
    );
    SharedHostHeapAdmission one_byte_short;
    std::string short_error;
    ok &= !ValidateSharedHostHeapAdmission(
        g1, 206569471, &one_byte_short, &short_error
    );
    ok &= !short_error.empty();

    std::vector<int32_t> extended_g1_contexts(
        kMaxBatches, 8192
    );
    SharedHostTaskPlan extended_g1;
    ok &= SetContextsAndBuild(
        state, extended_g1_contexts.data(),
        kMaxBatches, &extended_g1
    );
    ok &= extended_g1.total_tasks == 2560;
    SharedHostHeapAdmission extended_exact;
    std::string extended_exact_error;
    ok &= ValidateSharedHostHeapAdmission(
        extended_g1, 413138944,
        &extended_exact, &extended_exact_error
    );
    SharedHostHeapAdmission extended_short;
    std::string extended_short_error;
    ok &= !ValidateSharedHostHeapAdmission(
        extended_g1, 413138943,
        &extended_short, &extended_short_error
    );
    ok &= !extended_short_error.empty();

    // B512 只为默认 PA-G1 翻倍模型扩容；不能让 batch 上限绕过
    // 既有 4,352-task output/history 物理容量。
    state->config.batches = kMaxBatches;
    for (uint32_t batch = 0; batch < kMaxBatches; ++batch) {
        state->context_lens[batch] = 32768;
    }
    SharedHostTaskPlan extended_g4;
    std::string extended_g4_error;
    ok &= !BuildSharedHostTaskPlan(
        *state, &extended_g4, &extended_g4_error
    );
    ok &= extended_g4.total_tasks == 0;
    ok &= !extended_g4_error.empty();

    // 构造“总量仍放得下、但 task_id%8 的单个 shard 已溢出”的偏斜计划，
    // 防止准入实现退化成只比较 aggregate heap。
    std::vector<int32_t> skew_contexts(9, 0);
    SharedHostTaskPlan skew;
    ok &= SetContextsAndBuild(
        state, skew_contexts.data(), 9, &skew
    );
    for (SharedHostPlannedTask &task : skew.tasks) {
        task.output_bytes = 0;
    }
    const uint64_t shard_span =
        (kHeapBytes / kSharedHeapShards) /
        kOutputAlignment * kOutputAlignment;
    skew.tasks[0].output_bytes = shard_span;
    skew.tasks[8].output_bytes = kOutputAlignment;
    skew.canonical_heap_bytes =
        shard_span + kOutputAlignment;
    SharedHostHeapAdmission skew_result;
    std::string skew_error;
    ok &= !ValidateSharedHostHeapAdmission(
        skew, kHeapBytes, &skew_result, &skew_error
    );
    ok &= skew.canonical_heap_bytes <
        shard_span * kSharedHeapShards;
    ok &= skew_result.first_failed_task == 8;
    ok &= skew_result.first_failed_shard == 0;
    ok &= !skew_error.empty();

    SharedHostHeapAdmission signed_overflow;
    std::string signed_error;
    ok &= !ValidateSharedHostHeapAdmission(
        g1, static_cast<uint64_t>(INT64_MAX) + 1U,
        &signed_overflow, &signed_error
    );
    ok &= !signed_error.empty();

    SharedHostTaskPlan output_overflow = g1;
    output_overflow.tasks[0].output_bytes = UINT64_MAX;
    SharedHostHeapAdmission output_overflow_result;
    std::string output_overflow_error;
    ok &= !ValidateSharedHostHeapAdmission(
        output_overflow, kHeapBytes,
        &output_overflow_result, &output_overflow_error
    );
    ok &= output_overflow_result.first_failed_task == 0;
    ok &= !output_overflow_error.empty();

    SharedHostTaskPlan non_contiguous = g1;
    non_contiguous.tasks[3].task_id = 4;
    SharedHostHeapAdmission non_contiguous_result;
    std::string non_contiguous_error;
    ok &= !ValidateSharedHostHeapAdmission(
        non_contiguous, kHeapBytes,
        &non_contiguous_result, &non_contiguous_error
    );
    ok &= non_contiguous_result.first_failed_task == 4;
    ok &= !non_contiguous_error.empty();
    return ok;
}

bool CheckCli() {
    bool ok = true;
    {
        Options options;
        char program[] = "host-plan-test";
        char batches_name[] = "--batches";
        char batches_value[] = "4";
        char contexts_name[] = "--shared-context-lens";
        char contexts_value[] = "0,8192,8193,32768";
        char *argv[] = {
            program, batches_name, batches_value,
            contexts_name, contexts_value,
        };
        ok &= ParseOptions(5, argv, false, &options) ==
              ParseStatus::Ok;
        ok &= options.batches == 4;
        ok &= options.shared_context_lens.size() == 4;
        ok &= options.shared_context_lens[0] == 0;
        ok &= options.shared_context_lens[1] == 8192;
        ok &= options.shared_context_lens[2] == 8193;
        ok &= options.shared_context_lens[3] == 32768;
    }
    {
        Options options;
        char program[] = "host-plan-test";
        char batches_name[] = "--batches";
        char batches_value[] = "4";
        char contexts_name[] = "--shared-context-lens";
        char contexts_value[] = "16384";
        char *argv[] = {
            program, batches_name, batches_value,
            contexts_name, contexts_value,
        };
        ok &= ParseOptions(5, argv, false, &options) ==
              ParseStatus::Ok;
        ok &= options.shared_context_lens.size() == 1;
        ok &= options.shared_context_lens.front() == 16384;
    }
    {
        Options options;
        char program[] = "host-plan-test";
        char batches_name[] = "--batches";
        char batches_value[] = "4";
        char contexts_name[] = "--shared-context-lens";
        char contexts_value[] = "0,8192";
        char *argv[] = {
            program, batches_name, batches_value,
            contexts_name, contexts_value,
        };
        ok &= ParseOptions(5, argv, false, &options) ==
              ParseStatus::Error;
    }
    {
        Options options;
        char program[] = "host-plan-test";
        char contexts_name[] = "--shared-context-lens";
        char contexts_value[] = "32769";
        char *argv[] = {
            program, contexts_name, contexts_value,
        };
        ok &= ParseOptions(3, argv, false, &options) ==
              ParseStatus::Error;
    }
    return ok;
}

}  // namespace

int main() {
    std::unique_ptr<SchedulerState> state(new SchedulerState);
    bool ok = true;

    ok &= Check(
        CheckSingleContext(state.get(), 0, 0, 1),
        "G0 context=0 -> Alloc only"
    );
    ok &= Check(
        CheckSingleContext(state.get(), 8192, 1, 5),
        "G1 context=8192 -> Alloc + one group"
    );
    ok &= Check(
        CheckSingleContext(state.get(), 8193, 2, 9),
        "G2 context=8193 -> full group + one-block group"
    );
    ok &= Check(
        CheckSingleContext(state.get(), 16384, 2, 9),
        "G2 context=16384 -> two full groups"
    );
    ok &= Check(
        CheckSingleContext(state.get(), 32768, 4, 17),
        "G4 context=32768 -> four full groups"
    );
    ok &= Check(
        CheckRealComputeActivityContract(state.get()),
        "real-compute activity follows G0/nonzero shared plan"
    );
    ok &= Check(
        CheckTensorDescSemanticMatch(),
        "descriptor oracle ignores inactive payload bytes but rejects active corruption"
    );
    ok &= Check(
        CheckHeapAdmission(state.get()),
        "shared heap admission rejects over-capacity plans before workers"
    );

    const int32_t mixed_contexts[] = {
        0, 8192, 8193, 32768,
    };
    SharedHostTaskPlan mixed;
    bool mixed_ok = SetContextsAndBuild(
        state.get(), mixed_contexts, 4, &mixed
    );
    mixed_ok &= mixed.batch_count == 4;
    mixed_ok &= mixed.total_groups == 7;
    mixed_ok &= mixed.total_tasks == 32;
    mixed_ok &= mixed.canonical_heap_bytes == 4843520;
    const uint32_t expected_starts[] = {0, 1, 6, 15};
    const uint32_t expected_counts[] = {1, 5, 9, 17};
    const uint32_t expected_final_ups[] = {
        UINT32_MAX, 5, 14, 31,
    };
    for (uint32_t batch = 0; batch < 4; ++batch) {
        const SharedHostBatchPlan *entry =
            mixed.BatchAt(batch);
        mixed_ok &= entry != nullptr;
        if (entry != nullptr) {
            mixed_ok &=
                entry->batch_start == expected_starts[batch];
            mixed_ok &=
                entry->task_count == expected_counts[batch];
            mixed_ok &=
                entry->final_up_task_id ==
                    expected_final_ups[batch];
        }
    }
    const SharedHostPlannedTask *g0_alloc =
        mixed.TaskAt(0);
    const SharedHostPlannedTask *g2_first_up =
        mixed.TaskAt(10);
    const SharedHostPlannedTask *g2_partial_qk =
        mixed.TaskAt(11);
    const SharedHostPlannedTask *g2_partial_sf =
        mixed.TaskAt(12);
    const SharedHostPlannedTask *g2_final_up =
        mixed.TaskAt(14);
    const SharedHostPlannedTask *g4_final_up =
        mixed.TaskAt(31);
    mixed_ok &=
        g0_alloc != nullptr &&
        g0_alloc->batch == 0 &&
        !g0_alloc->in_group &&
        g0_alloc->is_last_in_batch;
    mixed_ok &=
        g2_first_up != nullptr &&
        g2_first_up->batch == 2 &&
        g2_first_up->group_index == 0 &&
        g2_first_up->has_following_group &&
        !g2_first_up->is_final_up;
    mixed_ok &=
        g2_partial_qk != nullptr &&
        g2_partial_qk->kind == TaskKind::Qk &&
        g2_partial_qk->group_index == 1 &&
        g2_partial_qk->group_block_count == 1 &&
        g2_partial_qk->output_bytes == 8192;
    mixed_ok &=
        g2_partial_sf != nullptr &&
        g2_partial_sf->kind == TaskKind::Sf &&
        g2_partial_sf->group_block_count == 1 &&
        g2_partial_sf->output_bytes == 6144;
    mixed_ok &=
        g2_final_up != nullptr &&
        g2_final_up->is_final_up &&
        !g2_final_up->has_following_group &&
        g2_final_up->is_last_in_batch;
    mixed_ok &=
        g4_final_up != nullptr &&
        g4_final_up->batch == 3 &&
        g4_final_up->group_index == 3 &&
        g4_final_up->is_final_up &&
        g4_final_up->is_last_in_batch;
    mixed_ok &= mixed.TaskAt(32) == nullptr;
    ok &= Check(
        mixed_ok,
        "mixed G0/G1/G2/G4 has cumulative batch starts and TaskAt metadata"
    );

    const uint64_t expected_mixed_dependency_signature =
        DependencyEdgeSignatureHost(3, 2) ^
        DependencyEdgeSignatureHost(4, 3) ^
        DependencyEdgeSignatureHost(5, 3) ^
        DependencyEdgeSignatureHost(5, 4) ^
        DependencyEdgeSignatureHost(5, 1) ^
        DependencyEdgeSignatureHost(8, 7) ^
        DependencyEdgeSignatureHost(9, 8) ^
        DependencyEdgeSignatureHost(10, 8) ^
        DependencyEdgeSignatureHost(10, 9) ^
        DependencyEdgeSignatureHost(10, 6) ^
        DependencyEdgeSignatureHost(12, 11) ^
        DependencyEdgeSignatureHost(13, 12) ^
        DependencyEdgeSignatureHost(14, 12) ^
        DependencyEdgeSignatureHost(14, 13) ^
        DependencyEdgeSignatureHost(14, 10) ^
        DependencyEdgeSignatureHost(17, 16) ^
        DependencyEdgeSignatureHost(18, 17) ^
        DependencyEdgeSignatureHost(19, 17) ^
        DependencyEdgeSignatureHost(19, 18) ^
        DependencyEdgeSignatureHost(19, 15) ^
        DependencyEdgeSignatureHost(21, 20) ^
        DependencyEdgeSignatureHost(22, 21) ^
        DependencyEdgeSignatureHost(23, 21) ^
        DependencyEdgeSignatureHost(23, 22) ^
        DependencyEdgeSignatureHost(23, 19) ^
        DependencyEdgeSignatureHost(25, 24) ^
        DependencyEdgeSignatureHost(26, 25) ^
        DependencyEdgeSignatureHost(27, 25) ^
        DependencyEdgeSignatureHost(27, 26) ^
        DependencyEdgeSignatureHost(27, 23) ^
        DependencyEdgeSignatureHost(29, 28) ^
        DependencyEdgeSignatureHost(30, 29) ^
        DependencyEdgeSignatureHost(31, 29) ^
        DependencyEdgeSignatureHost(31, 30) ^
        DependencyEdgeSignatureHost(31, 27);
    ok &= Check(
        ExpectedPaDependencySignature(mixed) ==
            expected_mixed_dependency_signature,
        "dependency oracle chains each later UP to the previous UP writer"
    );

    state->config.batches = 1;
    state->context_lens[0] = -1;
    SharedHostTaskPlan rejected;
    ok &= Check(
        !BuildSharedHostTaskPlan(*state, &rejected),
        "negative final context_len is rejected"
    );
    state->context_lens[0] = 32769;
    ok &= Check(
        !BuildSharedHostTaskPlan(*state, &rejected),
        "context_len above four groups is rejected"
    );
    state->config.batches = kMaxBatches + 1U;
    ok &= Check(
        !BuildSharedHostTaskPlan(*state, &rejected),
        "batch count above compiled capacity is rejected"
    );

    Options default_options;
    default_options.batches = 1;
    InitializeState(state.get(), default_options);
    ok &= Check(
        state->context_lens[0] == 8192 &&
            state->heap_size == kHeapBytes,
        "InitializeState keeps the shared default context_len and 256 MiB heap"
    );
    Options extended_options;
    extended_options.batches = kMaxBatches;
    InitializeState(state.get(), extended_options);
    ok &= Check(
        state->context_lens[kMaxBatches - 1U] == 8192 &&
            state->heap_size == kExtendedBatchHeapBytes,
        "B512 defaults to PA-G1 and the 512 MiB no-wrap heap"
    );
    Options mixed_options;
    mixed_options.batches = 4;
    mixed_options.shared_context_lens.assign(
        mixed_contexts, mixed_contexts + 4
    );
    InitializeState(state.get(), mixed_options);
    bool initialized_mixed_ok = true;
    for (uint32_t batch = 0; batch < 4; ++batch) {
        initialized_mixed_ok &=
            state->context_lens[batch] ==
            mixed_contexts[batch];
    }
    ok &= Check(
        initialized_mixed_ok,
        "InitializeState writes the exact shared CLI context vector"
    );

    ok &= Check(
        CheckCli(),
        "shared test CLI accepts broadcast/mixed and rejects invalid lists"
    );

    std::printf(
        "[HOST_PLAN_TEST] status=%s\n",
        ok ? "PASS" : "FAIL"
    );
    return ok ? 0 : 1;
}
