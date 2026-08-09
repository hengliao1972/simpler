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

#pragma once

#include "dist_engine/aicore/cross_core_simt_topology.h"
#include "dist_engine/common/cross_core_exec_protocol.h"

namespace {

PTO_DEVICE_FUNC bool dist_cross_core_classify_single_lane_kernel(
    const MixedKernels &mixed, fdwic::cross_core::ExecEngineClass &engine_class, int32_t &kernel_id
) {
    const ActiveMask active = mixed.to_active_mask();
    if (__builtin_popcount(active.core_mask()) != 1) return false;
    if (lane_active(active, LANE_AIC)) {
        engine_class = fdwic::cross_core::ExecEngineClass::Aic;
        kernel_id = mixed.aic_kernel_id;
        return kernel_id >= 0 && kernel_id < RUNTIME_MAX_FUNC_ID;
    }
    if (lane_active(active, LANE_AIV0)) {
        engine_class = fdwic::cross_core::ExecEngineClass::Aiv;
        kernel_id = mixed.aiv0_kernel_id;
        return kernel_id >= 0 && kernel_id < RUNTIME_MAX_FUNC_ID;
    }
    if (lane_active(active, LANE_AIV1)) {
        engine_class = fdwic::cross_core::ExecEngineClass::Aiv;
        kernel_id = mixed.aiv1_kernel_id;
        return kernel_id >= 0 && kernel_id < RUNTIME_MAX_FUNC_ID;
    }
    return false;
}

}  // namespace
