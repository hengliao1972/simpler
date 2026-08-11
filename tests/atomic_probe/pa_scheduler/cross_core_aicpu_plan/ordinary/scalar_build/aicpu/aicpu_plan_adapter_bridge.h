/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */

#ifndef PA_SCHEDULER_AICPU_PLAN_ADAPTER_BRIDGE_H
#define PA_SCHEDULER_AICPU_PLAN_ADAPTER_BRIDGE_H

#include <cstdint>

#include "aicpu_plan_operation_trace.h"

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
// flags，对同一份 GM wire 完整校验后再按
// payload clean -> barrier -> cell Published 的顺序发布。
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
