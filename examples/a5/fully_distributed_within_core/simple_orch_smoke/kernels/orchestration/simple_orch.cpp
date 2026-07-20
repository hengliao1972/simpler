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

#include <cstdint>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_MARK_AIC 0
#define FUNC_MARK_AIV 1

extern "C" {

__attribute__((visibility("default"), weak)) PTO2OrchestrationConfig aicpu_orchestration_config(
    const L2TaskArgs &orch_args
) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 5,
    };
}

__attribute__((visibility("default"), weak)) PTO_DEVICE_FUNC void aicpu_orchestration_entry(
    const L2TaskArgs &orch_args
) {
    const Tensor &input = orch_args.tensor(0).ref();
    const Tensor &output = orch_args.tensor(1).ref();
    uint64_t n = orch_args.scalar(0);
    uint64_t delta = orch_args.scalar(1);
    uint64_t mixed = orch_args.scalar(2);

    for (uint64_t i = 0; i < n; i++) {
        if (mixed == 3) {
            MixedKernels mk;
            mk.aic_kernel_id = FUNC_MARK_AIC;
            mk.aiv0_kernel_id = FUNC_MARK_AIV;
            L0TaskArgs eager_args;
            rt_submit_task_compete_first(
                mk, eager_args,
                [&](L0TaskArgs &submit_args) PTO_DEVICE_FUNC {
                    submit_args.add_input(input);
                    submit_args.add_inout(output);
                    submit_args.add_scalar(n);
                    submit_args.add_scalar(delta);
                    submit_args.add_scalar(i);
                }
            );
            continue;
        }
        L0TaskArgs args;
        args.add_input(input);
        args.add_inout(output);
        args.add_scalar(n);
        args.add_scalar(delta);
        args.add_scalar(i);
        if (mixed == 2) {
            MixedKernels mk;
            mk.aiv0_kernel_id = FUNC_MARK_AIV;
            mk.aiv1_kernel_id = FUNC_MARK_AIV;
            rt_submit_task(mk, args);
        } else if (mixed != 0) {
            MixedKernels mk;
            mk.aic_kernel_id = FUNC_MARK_AIC;
            mk.aiv0_kernel_id = FUNC_MARK_AIV;
            rt_submit_task(mk, args);
        } else if (i % 3 == 0) {
            rt_submit_aic_task(FUNC_MARK_AIC, args);
        } else {
            rt_submit_aiv_task(FUNC_MARK_AIV, args);
        }
    }
}

}  // extern "C"
