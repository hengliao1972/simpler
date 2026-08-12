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

#ifndef PA_SCHEDULER_AICPU_PLAN_BACKEND_ABI_H
#define PA_SCHEDULER_AICPU_PLAN_BACKEND_ABI_H

#include <cstdint>

#include "aicpu_plan_operation_trace.h"

// 这个头只描述独立 smoke 与 AICPU Plan backend 之间的窄 C ABI。
// RuntimePlanControl/Cell 的真实类型仍由公共 Plan 协议定义；这里故意只
// 传地址和容量，避免 orchestration SO 把 Host 生成的 task identity 混入。
struct AicpuPlanBackendConfig {
    void *control;
    void *cells;
    uint32_t capacity;
    uint32_t reserved;
    void *task_trace_records;
    uint32_t task_trace_capacity;
    uint32_t task_trace_record_bytes;
    pa_scheduler::aicpu_plan_trace::State *operation_trace_state;
    uint32_t operation_trace_initial_count;
    uint32_t operation_trace_reserved;
};

// 逐 task 记录真实 orchestration callback 的五段时间线。Build 从前一
// task 发布完成后开始；orchestration 区间是 Begin 返回到 Finish 进入，
// stage 是 Finish backend 打包 canonical payload，defer_publish 是等待
// 下一个 callback（或 backend close）确定 flags，publish 是最终 flags
// patch、wire 校验和 policy-specific cell-control 发布。整个结构恰好一条 cache line，
// AICPU 在 owner 结束前统一 clean，Host 只在 plan stream 闭合后回拷。
struct alignas(64) AicpuPlanTaskTraceRecord {
    uint64_t build_begin_ns;
    uint64_t begin_end_ns;
    uint64_t finish_begin_ns;
    uint64_t finish_end_ns;
    uint64_t publish_begin_ns;
    uint64_t publish_end_ns;
    uint32_t task_id;
    int16_t function_id;
    uint16_t output_count;
    uint16_t payload_lines;
    uint8_t task_kind;
    uint8_t engine_class;
    uint8_t group;
    uint8_t reserved[3];
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
    TraceFailed = 9,
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
    uint32_t trace_count;
    uint32_t trace_record_bytes;
    uint32_t operation_trace_count;
    uint32_t operation_trace_record_bytes;
    uint32_t operation_trace_dropped;
    uint32_t reserved[2];
};

static_assert(sizeof(AicpuPlanBackendConfig) == 56U, "AICPU Plan backend config ABI changed");
static_assert(sizeof(AicpuPlanTaskTraceRecord) == 64U, "AICPU Plan task trace ABI changed");
static_assert(alignof(AicpuPlanTaskTraceRecord) == 64U, "AICPU Plan task trace alignment changed");
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
