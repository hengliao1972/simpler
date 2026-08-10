/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */

#ifndef PA_SCHEDULER_AICPU_PLAN_ADAPTER_BRIDGE_H
#define PA_SCHEDULER_AICPU_PLAN_ADAPTER_BRIDGE_H

#include <cstdint>

extern "C" {

int32_t aicpu_plan_adapter_initialize(
    void *control, void *cells, uint32_t capacity
);

// actual runtime 的 L0TaskArgs 和 standalone 的 PA adapter 使用同一份
// 1280B ABI，但两个定义位于不同头文件且名字冲突。桥接 TU 通过 byte copy
// 消除 strict-aliasing 问题，并把真实 callback 立即固化成无指针 Plan payload。
int32_t aicpu_plan_adapter_stage(
    const void *l0_task_args, uint32_t task_id, int32_t function_id,
    uint8_t engine_class, uint8_t provisional_adapter_flags,
    uint32_t batch_start,
    void *staged_cell, uint32_t *payload_lines, uint16_t *output_count
);

// 下一个真实 Begin 到来后，backend 才能知道前一个 UP 是否还有下一组，
// 或前一个 UP/Alloc 是否为 batch 尾。桥接层用最终 flags 重新打包并走
// 公共 PublishRuntimeTaskPlan/AdvancePlannedFrontier，不能绕开 wire 校验。
int32_t aicpu_plan_adapter_publish_staged(
    void *control, void *cells, uint32_t capacity,
    const void *staged_cell, uint32_t payload_lines,
    uint8_t final_adapter_flags
);

int32_t aicpu_plan_adapter_close(
    void *control, void *cells, uint32_t capacity,
    uint32_t final_task_count
);

void aicpu_plan_adapter_publish_fatal(void *control, int64_t error_code);

}

#endif  // PA_SCHEDULER_AICPU_PLAN_ADAPTER_BRIDGE_H
