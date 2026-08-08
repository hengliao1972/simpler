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

#if PTO_FDWIC_SCHEDULER_MODE == 3

#include "dist_engine/aicore/cross_core_kernel_classification.h"
#include "dist_engine/aicore/cross_core_simt_request_source.h"

namespace {

using fdwic::cross_core::ExecEngineClass;
using fdwic::cross_core::SimtBuildRequestSpec;
using fdwic::cross_core::SimtL0TaskArgsRequestSource;
using fdwic::cross_core::SimtRequestPublishResult;
using fdwic::cross_core::SimtRequestReserveResult;

PTO_DEVICE_FUNC __gm__ SimtCrossCoreOrdinaryState &dist_simt_cross_core_state() {
    return g_dist.simt_cross_core_ordinary;
}

PTO_DEVICE_FUNC bool dist_simt_cross_core_fail(int32_t task_id, int32_t error_code) {
    __gm__ DistCore *self = g_self;
    const uint32_t owner = self != nullptr && self->core_idx >= 0 ? static_cast<uint32_t>(self->core_idx) : 0U;
    (void)fdwic::cross_core::PublishExecFatal<DistCrossCoreAicoreOps>(
        dist_simt_cross_core_state().runtime.fatal, fdwic::cross_core::ExecFatalReason::InvalidBuildInput,
        task_id >= 0 ? static_cast<uint32_t>(task_id) : UINT32_MAX, owner
    );
    set_fatal_code(error_code);
    if (self != nullptr) self->local_index = kFlagCap;
    return false;
}

PTO_DEVICE_FUNC bool dist_simt_cross_core_task_valid(const DistSubmitCtx &ctx) {
    return ctx.self != nullptr && ctx.self->core_idx >= 0 &&
           static_cast<uint32_t>(ctx.self->core_idx) <= fdwic::cross_core::kExecMaxOwner && ctx.task_id >= 0 &&
           static_cast<uint32_t>(ctx.task_id) < kFdwicCrossCoreTaskCapacity;
}

PTO_DEVICE_FUNC bool dist_simt_cross_core_reserve_request(DistSubmitCtx &ctx, bool &publisher) {
    publisher = false;
    if (!dist_simt_cross_core_task_valid(ctx)) {
        return dist_simt_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_CAPACITY);
    }
    __gm__ fdwic::cross_core::SimtBuildRequestCell &request =
        dist_simt_cross_core_state().requests[static_cast<uint32_t>(ctx.task_id)];
    const SimtRequestReserveResult result = fdwic::cross_core::ReserveSimtBuildRequest<DistCrossCoreAicoreOps>(
        request, static_cast<uint32_t>(ctx.task_id), static_cast<uint32_t>(ctx.self->core_idx),
        dist_simt_cross_core_state().runtime.fatal
    );
    publisher = result == SimtRequestReserveResult::Reserved;
    if (!publisher && result != SimtRequestReserveResult::CellUnavailable) {
        return dist_simt_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    return true;
}

PTO_DEVICE_FUNC bool dist_simt_cross_core_publish_request(
    DistSubmitCtx &ctx, const L0TaskArgs &args, ExecEngineClass engine_class, int32_t kernel_id
) {
    if (!fdwic::cross_core::ValidateSimtL0TaskArgs(args, static_cast<uint32_t>(ctx.task_id))) {
        return dist_simt_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    const bool immediate = engine_class == ExecEngineClass::Immediate;
    const uint64_t function_address = immediate ? 0 : dist_aicore_slot_function_addr(g_dist.runtime, kernel_id);
    const SimtBuildRequestSpec spec{
        static_cast<uint32_t>(ctx.task_id),
        function_address,
        immediate ? fdwic::cross_core::kExecInvalidFunctionId : static_cast<uint32_t>(kernel_id),
        static_cast<uint16_t>(args.tensor_count()),
        static_cast<uint16_t>(args.scalar_count()),
        static_cast<uint16_t>(args.explicit_dep_count()),
        engine_class,
        0,
    };
    const SimtL0TaskArgsRequestSource source{args};
    __gm__ fdwic::cross_core::SimtBuildRequestCell &request =
        dist_simt_cross_core_state().requests[static_cast<uint32_t>(ctx.task_id)];
    if (fdwic::cross_core::PublishReservedSimtBuildRequest<DistCrossCoreAicoreOps>(
            request, static_cast<uint32_t>(ctx.self->core_idx), spec, source, dist_simt_cross_core_state().runtime.fatal
        ) != SimtRequestPublishResult::Published) {
        return dist_simt_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    return true;
}

PTO_DEVICE_FUNC bool
dist_simt_cross_core_classify_kernel(const MixedKernels &mixed, ExecEngineClass &engine_class, int32_t &kernel_id) {
    return dist_cross_core_classify_single_lane_kernel(mixed, engine_class, kernel_id);
}

}  // namespace

#endif  // PTO_FDWIC_SCHEDULER_MODE == 3
