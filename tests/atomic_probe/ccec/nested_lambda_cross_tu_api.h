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
#ifndef TESTS_ATOMIC_PROBE_CCEC_NESTED_LAMBDA_CROSS_TU_API_H
#define TESTS_ATOMIC_PROBE_CCEC_NESTED_LAMBDA_CROSS_TU_API_H

#include "pto_types.h"

#include <cstdint>

namespace nested_lambda_cross_tu_probe {

enum class DispatchPhase : int32_t {
    Prepare = 0,
    WinnerBind = 1,
};

struct CallerContext {
    const Tensor *first;
    const Tensor *second;
    const Tensor *third;
    uint64_t salt;
};

} // namespace nested_lambda_cross_tu_probe

PTO_DEVICE_FUNC TaskOutputTensors nested_probe_submit_weak_context(
    int32_t site_id, uint64_t caller_context, L0TaskArgs *args);
PTO_DEVICE_FUNC TaskOutputTensors nested_probe_submit_weak_args(
    int32_t site_id, L0TaskArgs *args);
PTO_DEVICE_FUNC TaskOutputTensors nested_probe_submit_strong_context(
    int32_t site_id, uint64_t caller_context, L0TaskArgs *args);
PTO_DEVICE_FUNC TaskOutputTensors nested_probe_submit_control(L0TaskArgs *args);
PTO_DEVICE_FUNC TaskOutputTensors nested_probe_submit_args_runtime_read(L0TaskArgs *args);

#endif // TESTS_ATOMIC_PROBE_CCEC_NESTED_LAMBDA_CROSS_TU_API_H
