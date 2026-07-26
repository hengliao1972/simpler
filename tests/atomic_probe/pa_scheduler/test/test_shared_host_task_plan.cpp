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

#include "host_support.h"

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
        state->context_lens[0] == 8192,
        "InitializeState keeps the shared default context_len=8192"
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
