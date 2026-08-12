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

#ifndef PA_SCHEDULER_AICPU_PLAN_ADAPTER_BRIDGE_H
#define PA_SCHEDULER_AICPU_PLAN_ADAPTER_BRIDGE_H

#include <cstdint>

#include "aicpu_plan_operation_trace.h"

// 正常发布路径信任同一 producer 在 stage 阶段已经完成的 canonical Pack，
// 不再逐 task 重读并扫描整份 GM wire。只有专门的协议调试构建才打开
// 发布前全量复核；泳道图并不隐式打开它，避免观测路径改变正常性能。
#ifndef PA_RUNTIME_PLAN_DEBUG_FULL_VALIDATION
#define PA_RUNTIME_PLAN_DEBUG_FULL_VALIDATION 0
#endif

#if PA_RUNTIME_PLAN_DEBUG_FULL_VALIDATION != 0 && \
    PA_RUNTIME_PLAN_DEBUG_FULL_VALIDATION != 1
#error "PA_RUNTIME_PLAN_DEBUG_FULL_VALIDATION must be 0 or 1"
#endif

extern "C" {

int32_t aicpu_plan_adapter_initialize(
    void *control, void *cells, uint32_t capacity, pa_scheduler::aicpu_plan_trace::State *operation_trace_state
);

// actual runtime 的 L0TaskArgs 和 standalone 的 PA adapter 使用同一份
// 1280B ABI，但两个定义位于不同头文件且名字冲突。桥接 TU 通过 byte copy
// 消除 strict-aliasing 问题，并在 callback 及其 descriptor 仍存活时
// 直接向目标 GM cell 执行唯一一次 canonical Pack。cell control
// 仍保持 Empty，payload 也不 clean，因此 consumer 此时不可见。
int32_t aicpu_plan_adapter_stage(
    void *control, void *cells, uint32_t capacity,
    const void *l0_task_args, uint32_t task_id, int32_t function_id,
    uint8_t engine_class, uint8_t provisional_adapter_flags,
    uint32_t batch_start,
    void *staged_metadata, uint32_t *payload_lines,
    uint16_t *output_count
);

// 下一个真实 Begin 到来后，backend 才能知道前一个 UP 是否还有下一组，
// 或前一个 UP/Alloc 是否为 batch 尾。桥接层只 patch 最终
// flags，再按 policy 发布：PlanAheadClosed
// 使用 ordinary payload -> release atomic Published control；
// StreamingFuture 保留 payload/control exact clean 与逐 task barrier。
// 完整 wire 与串行 producer 状态复核只由
// PA_RUNTIME_PLAN_DEBUG_FULL_VALIDATION=1 的专门调试构建启用；AICore
// consumer 在 acquire 后仍执行权威校验。
int32_t aicpu_plan_adapter_publish_staged(
    void *control, void *cells, uint32_t capacity,
    const void *staged_metadata, uint32_t payload_lines,
    uint8_t final_adapter_flags
);

int32_t aicpu_plan_adapter_close(
    void *control, void *cells, uint32_t capacity,
    uint32_t final_task_count
);

void aicpu_plan_adapter_publish_fatal(
    void *control, void *cells, uint32_t capacity,
    int64_t error_code
);

}

#endif  // PA_SCHEDULER_AICPU_PLAN_ADAPTER_BRIDGE_H
