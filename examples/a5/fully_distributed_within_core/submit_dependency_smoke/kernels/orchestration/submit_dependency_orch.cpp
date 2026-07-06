/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_FILL_ALLOC_AIC 0
#define FUNC_MAKE_LEFT_AIC 1
#define FUNC_MAKE_RIGHT_AIV 2
#define FUNC_FANIN_AIV 3

extern "C" {

__attribute__((visibility("default"), weak)) PTO2OrchestrationConfig aicpu_orchestration_config(
    const L2TaskArgs &orch_args
) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"), weak)) PTO_DEVICE_FUNC void aicpu_orchestration_entry(
    const L2TaskArgs &orch_args
) {
    const Tensor &input = orch_args.tensor(0).ref();
    const Tensor &output = orch_args.tensor(1).ref();
    const uint64_t n = orch_args.scalar(0);
    const uint32_t shape[1] = {static_cast<uint32_t>(n)};
    TensorCreateInfo scratch_ci(shape, 1, DataType::FLOAT32);

    TaskOutputTensors seed_out = alloc_tensors(scratch_ci);
    __gm__ const Tensor &seed = seed_out.get_ref(0);

    L0TaskArgs fill_args;
    fill_args.add_input(input);
    fill_args.add_inout(seed);
    fill_args.add_scalar(n);
    rt_submit_aic_task(FUNC_FILL_ALLOC_AIC, fill_args);

    L0TaskArgs left_args;
    left_args.add_input(input);
    left_args.add_output(scratch_ci);
    left_args.add_scalar(n);
    TaskOutputTensors left_out = rt_submit_aic_task(FUNC_MAKE_LEFT_AIC, left_args);
    __gm__ const Tensor &left = left_out.get_ref(0);

    L0TaskArgs right_args;
    right_args.add_input(input);
    right_args.add_output(scratch_ci);
    right_args.add_scalar(n);
    TaskOutputTensors right_out = rt_submit_aiv_task(FUNC_MAKE_RIGHT_AIV, right_args);
    __gm__ const Tensor &right = right_out.get_ref(0);

    L0TaskArgs fanin_args;
    fanin_args.add_input(seed);
    fanin_args.add_input(left);
    fanin_args.add_input(right);
    fanin_args.add_inout(output);
    fanin_args.add_scalar(n);
    rt_submit_aiv_task(FUNC_FANIN_AIV, fanin_args);
}

}  // extern "C"
