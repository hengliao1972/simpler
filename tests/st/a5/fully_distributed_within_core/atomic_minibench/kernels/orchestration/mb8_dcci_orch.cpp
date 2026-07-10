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
/**
 * MB-8 dcci seam orchestration.
 *
 * A minimal producer→consumer chain repeated N rounds. Each round:
 *   producer (AIV): writes pattern[i] = round * 100 + i into buf[i]
 *   consumer (AIV): reads buf, writes result[i] = buf[i] + 1
 *
 * If the dcci publish/observe seam is broken (producer doesn't flush, or
 * consumer doesn't invalidate), the consumer reads stale data and the golden
 * comparison fails.
 *
 * The dependency (consumer reads buf that producer writes) forces the runtime
 * to exercise the flag-based publish/observe path: producer sets flag via
 * dist_set_flag, consumer waits via coherent_load. On A5 onboard this is the
 * atomicMax + dcci seam; on sim it compiles to std::atomic (always passes).
 *
 * Arg layout: [buf, result, config]
 *   config = int64_t[1] = {num_rounds}
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_PRODUCER 0
#define FUNC_CONSUMER 1

extern "C" {

__attribute__((visibility("default"), weak)) PTO2OrchestrationConfig aicpu_orchestration_config(const L2TaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"), weak)) PTO_DEVICE_FUNC void aicpu_orchestration_entry(const L2TaskArgs &orch_args) {
    const Tensor &ext_buf = orch_args.tensor(0).ref();
    const Tensor &ext_result = orch_args.tensor(1).ref();
    int64_t *config = orch_args.tensor(2).ref().data_as<int64_t>();
    int32_t num_rounds = static_cast<int32_t>(config[0]);

    LOG_INFO_V0("[mb8_orch] num_rounds=%d", num_rounds);

    for (int32_t r = 0; r < num_rounds; r++) {
        PTO2_SCOPE_GUARD();

        uint32_t scalar_val = static_cast<uint32_t>(r);

        // Producer: write pattern into buf (INOUT registers producer in TensorMap)
        L0TaskArgs prod_args;
        prod_args.add_inout(ext_buf);
        prod_args.add_scalar(static_cast<uint64_t>(scalar_val));
        rt_submit_aiv_task(FUNC_PRODUCER, prod_args);

        // Consumer: read buf (INPUT creates fan-in on producer), write result
        L0TaskArgs cons_args;
        cons_args.add_input(ext_buf);
        cons_args.add_inout(ext_result);
        cons_args.add_scalar(static_cast<uint64_t>(scalar_val));
        rt_submit_aiv_task(FUNC_CONSUMER, cons_args);
    }

    LOG_INFO_V0("[mb8_orch] completed %d rounds", num_rounds);
}

}  // extern "C"
