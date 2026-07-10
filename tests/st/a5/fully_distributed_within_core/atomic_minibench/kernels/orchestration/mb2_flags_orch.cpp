/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except the compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * MB-2 flags[] clobber orchestration.
 *
 * Submits N independent write-only tasks, each writing a unique value into
 * a distinct slot of a single output tensor. No cross-task dependencies —
 * every task is independent, so completion flags are set concurrently by
 * different cores on densely packed cache lines.
 *
 * If dist_set_flag uses the buggy path (store + dcci CACHELINE_OUT), adjacent
 * flags get clobbered and some consumers never see completion → golden fails
 * (missing writes) or hangs. With the fix (atomicMax), all flags survive.
 *
 * Arg layout: [out, config]
 *   config = int64_t[1] = {num_tasks}
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_WRITE_INDEX 0

extern "C" {

__attribute__((visibility("default"), weak)) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 2,
    };
}

__attribute__((visibility("default"), weak)) PTO_DEVICE_FUNC void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_out = orch_args.tensor(0).ref();
    int64_t *config = orch_args.tensor(1).ref().data_as<int64_t>();
    int32_t num_tasks = static_cast<int32_t>(config[0]);

    LOG_INFO_V0("[mb2_orch] num_tasks=%d", num_tasks);

    for (int32_t i = 0; i < num_tasks; i++) {
        PTO2_SCOPE_GUARD();

        uint32_t slot_shapes[1] = {1};
        uint32_t slot_offsets[1] = {static_cast<uint32_t>(i)};
        Tensor out_view = Tensor::view(ext_out, slot_shapes, slot_offsets);

        L0TaskArgs args;
        args.add_inout(out_view);
        args.add_scalar(static_cast<uint64_t>(i));
        rt_submit_aiv_task(FUNC_WRITE_INDEX, args);
    }

    LOG_INFO_V0("[mb2_orch] submitted %d independent write tasks", num_tasks);
}

}  // extern "C"
