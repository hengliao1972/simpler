/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */

#ifndef PA_SCHEDULER_AICPU_PLAN_BACKEND_ABI_H
#define PA_SCHEDULER_AICPU_PLAN_BACKEND_ABI_H

#include <cstdint>

// 这个头只描述独立 smoke 与 AICPU Plan backend 之间的窄 C ABI。
// RuntimePlanControl/Cell 的真实类型仍由公共 Plan 协议定义；这里故意只
// 传地址和容量，避免 orchestration SO 把 Host 生成的 task identity 混入。
struct AicpuPlanBackendConfig {
    void *control;
    void *cells;
    uint32_t capacity;
    uint32_t reserved;
};

enum class AicpuPlanBackendStatus : int32_t {
    Ok = 0,
    NotBound = 1,
    BadConfig = 2,
    BadSequence = 3,
    BadTicket = 4,
    BadTaskArgs = 5,
    PublishFailed = 6,
    CloseFailed = 7,
    FatalReportedByOrchestration = 8,
};

struct AicpuPlanBackendResult {
    int32_t status;
    uint32_t task_count;
    uint32_t begin_count;
    uint32_t finish_count;
    uint32_t alloc_count;
    uint32_t aic_count;
    uint32_t aiv_count;
    uint32_t published_count;
    int32_t fatal_code;
    uint32_t reserved[7];
};

static_assert(sizeof(AicpuPlanBackendConfig) == 24U, "AICPU Plan backend config ABI changed");
static_assert(sizeof(AicpuPlanBackendResult) == 64U, "AICPU Plan backend result ABI changed");

extern "C" {

// Bind 只接收 AICPU 自己将要填充的空 Plan 存储，不接收 task 数量、类型或
// 任何 PA 业务计划。所有身份都必须由随后真实 orchestration callback 产生。
__attribute__((visibility("default"))) int32_t
aicpu_plan_backend_bind(const AicpuPlanBackendConfig *config);

// orchestration entry 返回后由 AICPU executor 调用。它发布最后一个延迟
// cell、关闭 task_count，并给 Scalar 的 Plan-ahead 阶段提供唯一闭合边界。
__attribute__((visibility("default"))) int32_t aicpu_plan_backend_close();

__attribute__((visibility("default"))) AicpuPlanBackendResult aicpu_plan_backend_result();

}

#endif  // PA_SCHEDULER_AICPU_PLAN_BACKEND_ABI_H
