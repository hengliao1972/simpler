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

#include "host_support.h"

#include <cstdio>

namespace {

using namespace pa_scheduler;
using namespace pa_scheduler::host;

bool Check(bool condition, const char *label) {
    std::printf("[HOST_CONTRACT] %-56s %s\n", label,
                condition ? "PASS" : "FAIL");
    return condition;
}

bool CheckTerminalControl() {
    constexpr uint32_t kTasks = 37U;
    aicpu_plan::RuntimePlanControl control{};
    control.planned_frontier.value = kTasks;
    control.closed_task_count.value = kTasks;
    control.build_next.value =
        kTasks + kRuntimePlanBuildWorkers;
    control.build_workers_done.value =
        kRuntimePlanBuildWorkers;
    control.build_release.value = kTasks;
    control.fatal.value = 0;
    bool ok = Check(
        HostRuntimePlanControlIsTerminal(control, kTasks),
        "N, N+W, W arrivals, release N form one terminal Plan"
    );

    --control.build_next.value;
    ok &= Check(
        !HostRuntimePlanControlIsTerminal(control, kTasks),
        "a missing terminal Build ticket is rejected"
    );
    ++control.build_next.value;
    control.fatal.value = 1;
    ok &= Check(
        !HostRuntimePlanControlIsTerminal(control, kTasks),
        "a fatal terminal Plan is rejected"
    );
    return ok;
}

bool CheckBuildOnlyWorkerResult() {
    WorkerResult result{};
    bool ok = Check(
        HostScalarWorkerBuildOnlyCountersAreZero(result),
        "zero Scalar WorkerResult contains no Build attribution"
    );

    result.submits = 1;
    ok &= Check(
        !HostScalarWorkerBuildOnlyCountersAreZero(result),
        "counterfeit Scalar submit is rejected"
    );
    result = {};
    result.wins[static_cast<uint32_t>(TaskKind::Qk)] = 1;
    ok &= Check(
        !HostScalarWorkerBuildOnlyCountersAreZero(result),
        "counterfeit Scalar winner is rejected"
    );
    result = {};
    result.materialized_outputs = 1;
    ok &= Check(
        !HostScalarWorkerBuildOnlyCountersAreZero(result),
        "counterfeit Scalar materialization is rejected"
    );
    result = {};
    result.fanin_edges = 1;
    ok &= Check(
        !HostScalarWorkerBuildOnlyCountersAreZero(result),
        "counterfeit Scalar Build-fanin is rejected"
    );
    result = {};
    result.dependency_signature = 1;
    ok &= Check(
        !HostScalarWorkerBuildOnlyCountersAreZero(result),
        "counterfeit Scalar dependency signature is rejected"
    );
    result = {};
    result.phase_calls[
        static_cast<uint32_t>(ProfilePhase::Register)
    ] = 1;
    ok &= Check(
        !HostScalarWorkerBuildOnlyCountersAreZero(result),
        "counterfeit Scalar Build phase is rejected"
    );
    return ok;
}

bool CheckOwnerPolicy() {
    const uint32_t last_legal_builder =
        kHostUsesSimtRuntimePlanBuild
            ? kHostSimtRuntimePlanBuildLeaders - 1U
            : kWorkers - 1U;
    const uint32_t first_illegal_builder =
        kHostUsesSimtRuntimePlanBuild
            ? kHostSimtRuntimePlanBuildLeaders
            : kWorkers;
    bool ok = Check(
        HostBuildOwnerMatchesS5bPolicy(last_legal_builder),
        "last backend-specific Build owner is legal"
    );
    ok &= Check(
        !HostBuildOwnerMatchesS5bPolicy(first_illegal_builder),
        "first out-of-population Build owner is rejected"
    );
    ok &= Check(
        HostDynamicPaExecuteOwnerIsLegal(
            11U, last_legal_builder, 0U,
            cross_core::ExecEngineClass::Aic
        ),
        "Build and AIC Execute ownership remain independent"
    );
    ok &= Check(
        HostDynamicPaExecuteOwnerIsLegal(
            12U, last_legal_builder, kAicWorkers,
            cross_core::ExecEngineClass::Aiv
        ),
        "Build and AIV Execute ownership remain independent"
    );
    return ok;
}

bool CheckCoarseTraceContract() {
    SharedSparseTraceValidator validator(nullptr, 0U);
    TraceRecord parent{};
    parent.task_id = -1;
    parent.function_id = -1;
    parent.phase =
        static_cast<int32_t>(TracePhase::RuntimePlanBuild);
    parent.start_cycle = 100;
    parent.end_cycle = 200;
    bool ok = Check(
        validator.Observe(parent) && validator.Closed(),
        "coarse RuntimePlanBuild parent closes without child Build rows"
    );

    SharedSparseTraceValidator rejects_child(nullptr, 0U);
    TraceRecord child{};
    child.task_id = 0;
    child.function_id = -1;
    child.phase = static_cast<int32_t>(TracePhase::Materialize);
    child.start_cycle = 120;
    child.end_cycle = 130;
    ok &= Check(
        rejects_child.Observe(parent) &&
            !rejects_child.Observe(child),
        "unowned PlannedBuild child cannot be fabricated"
    );
    return ok;
}

}  // namespace

int main() {
    static_assert(
        PTO_FDWIC_SHARED_MAP == 1,
        "Runtime Plan backend Host contract is shared-only"
    );
    static_assert(
        !kHostUsesSimtRuntimePlanBuild ||
            kRuntimePlanBuildWorkers == 4U,
        "SIMT Host contract requires W=4"
    );
    static_assert(
        kHostUsesSimtRuntimePlanBuild ||
            kRuntimePlanBuildWorkers == kWorkers,
        "default Scalar Host contract requires W=96"
    );

    bool ok = true;
    ok &= Check(
        kHostUsesSimtRuntimePlanBuild
            ? kRuntimePlanBuildWorkers == 4U
            : kRuntimePlanBuildWorkers == 96U,
        "compiled backend selects its exact Build population"
    );
    ok &= CheckTerminalControl();
    ok &= CheckBuildOnlyWorkerResult();
    ok &= CheckOwnerPolicy();
    ok &= CheckCoarseTraceContract();
    std::printf(
        "[HOST_CONTRACT] backend=%s workers=%u result=%s\n",
        kHostUsesSimtRuntimePlanBuild ? "simt" : "scalar",
        kRuntimePlanBuildWorkers, ok ? "PASS" : "FAIL"
    );
    return ok ? 0 : 1;
}
