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

#ifndef PA_SCHEDULER_RUNTIME_PLAN_PIPELINE_POLICY_H
#define PA_SCHEDULER_RUNTIME_PLAN_PIPELINE_POLICY_H

#include <cstdint>

// Runtime Plan 的阶段关系是构建合同，而不是一次运行里的动态选择：
//   0: AICPU 完整 Close Plan 后，Build 才能领取 ticket；
//   1: producer 发布 Ready 后，Build 可以持有 future ticket 等待 cell。
// 默认长期保留严格串行路径，作为正确性基线和并发收益/代价的对照组。
#ifndef PA_RUNTIME_PLAN_PIPELINE_POLICY
#define PA_RUNTIME_PLAN_PIPELINE_POLICY 0
#endif

#if PA_RUNTIME_PLAN_PIPELINE_POLICY != 0 && \
    PA_RUNTIME_PLAN_PIPELINE_POLICY != 1
#error "PA_RUNTIME_PLAN_PIPELINE_POLICY must be 0 (PlanAheadClosed) or 1 (StreamingFuture)"
#endif

namespace pa_scheduler {

enum class RuntimePlanPipelinePolicy : uint32_t {
    PlanAheadClosed = 0,
    StreamingFuture = 1,
};

constexpr RuntimePlanPipelinePolicy
    kCompiledRuntimePlanPipelinePolicy =
        static_cast<RuntimePlanPipelinePolicy>(
            PA_RUNTIME_PLAN_PIPELINE_POLICY
        );
constexpr bool kRuntimePlanPipelineIsPlanAheadClosed =
    kCompiledRuntimePlanPipelinePolicy ==
    RuntimePlanPipelinePolicy::PlanAheadClosed;
constexpr bool kRuntimePlanPipelineIsStreamingFuture =
    kCompiledRuntimePlanPipelinePolicy ==
    RuntimePlanPipelinePolicy::StreamingFuture;

static_assert(
    kRuntimePlanPipelineIsPlanAheadClosed !=
        kRuntimePlanPipelineIsStreamingFuture,
    "exactly one Runtime Plan pipeline policy must be selected"
);

}  // namespace pa_scheduler

#endif  // PA_SCHEDULER_RUNTIME_PLAN_PIPELINE_POLICY_H
