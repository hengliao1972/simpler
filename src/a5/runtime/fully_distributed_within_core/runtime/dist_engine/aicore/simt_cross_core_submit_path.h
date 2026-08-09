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

#if PTO_FDWIC_SCHEDULER_MODE == 3 || PTO_FDWIC_SCHEDULER_MODE == 4

#include "dist_engine/aicore/cross_core_kernel_classification.h"
#include "dist_engine/aicore/cross_core_simt_request_source.h"

namespace {

using fdwic::cross_core::ExecEngineClass;
using fdwic::cross_core::SimtBuildRequestSpec;
using fdwic::cross_core::SimtL0TaskArgsRequestSource;
using fdwic::cross_core::SimtRequestPublishResult;
using fdwic::cross_core::SimtRequestReserveResult;

#if PTO_FDWIC_SCHEDULER_MODE == 3
using DistSimtCrossCoreState = SimtCrossCoreOrdinaryState;
#else
using DistSimtCrossCoreState = SimtCrossCoreDagState;
#endif

PTO_DEVICE_FUNC __gm__ DistSimtCrossCoreState &dist_simt_cross_core_state() {
#if PTO_FDWIC_SCHEDULER_MODE == 3
    return g_dist.simt_cross_core_ordinary;
#else
    return g_dist.simt_cross_core_dag;
#endif
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
           static_cast<uint32_t>(ctx.task_id) < kFdwicCrossCoreTaskCapacity && ctx.payload != nullptr;
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

PTO_DEVICE_FUNC DistCompeteFirstTicket dist_simt_cross_core_invalid_ticket() {
    DistCompeteFirstTicket ticket{};
    ticket.task_id = kFlagCap;
    ticket.kernel_id = static_cast<int16_t>(INVALID_KERNEL_ID);
    return ticket;
}

PTO_DEVICE_FUNC bool dist_simt_cross_core_finish_request(
    DistSubmitCtx &ctx, const L0TaskArgs *publisher_args, DistSubmitKind kind, ExecEngineClass engine_class,
    int32_t kernel_id, bool publisher
) {
    if (publisher) {
        if (publisher_args == nullptr ||
            !dist_simt_cross_core_publish_request(ctx, *publisher_args, engine_class, kernel_id)) {
            return false;
        }
#if !defined(__CCE_AICORE__)
        // CPU simulation has no persistent A5 SIMT vector function. Consume
        // the same immutable request at the publication point with the proven
        // Scalar ordinary builder, preserving all output/map/exec protocols.
        // Real A5 never enters this fallback.
        bool scalar_builder = false;
        if (!dist_cross_core_reserve_build(ctx, scalar_builder) || !scalar_builder ||
            !dist_cross_core_finish_builder(ctx, *publisher_args, engine_class, kernel_id)) {
            return dist_simt_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
        }
#endif
    }
    if (!dist_cross_core_acquire_outputs(ctx)) return false;
    return kind != DistSubmitKind::Kernel || dist_cross_core_bind_execution(ctx, engine_class);
}

PTO_DEVICE_FUNC TaskOutputTensors
dist_simt_cross_core_submit(const MixedKernels *mixed, const L0TaskArgs &args, DistSubmitKind kind) {
    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, args, ctx);
    if (!dist_simt_cross_core_task_valid(ctx)) {
        (void)dist_simt_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_CAPACITY);
        return ctx.result;
    }
    (void)drain_phase_b(ctx.self);

    ExecEngineClass engine_class = ExecEngineClass::Immediate;
    int32_t kernel_id = INVALID_KERNEL_ID;
    if (kind == DistSubmitKind::Kernel &&
        (mixed == nullptr || !dist_simt_cross_core_classify_kernel(*mixed, engine_class, kernel_id))) {
        (void)dist_simt_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
        return ctx.result;
    }
    bool publisher = false;
    if (!dist_simt_cross_core_reserve_request(ctx, publisher)) return ctx.result;
    (void)dist_simt_cross_core_finish_request(
        ctx, publisher ? &args : nullptr, kind, engine_class, kernel_id, publisher
    );
    return ctx.result;
}

PTO_DEVICE_FUNC DistCompeteFirstTicket
dist_simt_cross_core_compete_first_begin(const MixedKernels *mixed, DistSubmitKind kind) {
    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, ctx);
    if (!dist_simt_cross_core_task_valid(ctx)) {
        (void)dist_simt_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_CAPACITY);
        return dist_simt_cross_core_invalid_ticket();
    }
    (void)drain_phase_b(ctx.self);

    ExecEngineClass engine_class = ExecEngineClass::Immediate;
    int32_t kernel_id = INVALID_KERNEL_ID;
    if (kind == DistSubmitKind::Kernel &&
        (mixed == nullptr || !dist_simt_cross_core_classify_kernel(*mixed, engine_class, kernel_id))) {
        (void)dist_simt_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
        return dist_simt_cross_core_invalid_ticket();
    }
    bool publisher = false;
    if (!dist_simt_cross_core_reserve_request(ctx, publisher)) return dist_simt_cross_core_invalid_ticket();

    DistCompeteFirstTicket ticket{};
    ticket.task_id = ctx.task_id;
    ticket.kernel_id = static_cast<int16_t>(kernel_id);
    ticket.won = static_cast<uint8_t>(publisher);
    ticket.ready = static_cast<uint8_t>(publisher);
    return ticket;
}

PTO_DEVICE_FUNC bool dist_simt_cross_core_restore_ticket(
    const DistCompeteFirstTicket &ticket, const MixedKernels *mixed, DistSubmitKind kind, DistSubmitCtx &ctx,
    ExecEngineClass &engine_class, int32_t &kernel_id
) {
    ctx.self = g_self;
    ctx.task_id = ticket.task_id;
    ctx.payload = ctx.self != nullptr && ticket.task_id >= 0 &&
                          static_cast<uint32_t>(ticket.task_id) < kFdwicCrossCoreTaskCapacity ?
                      &ctx.self->task_payloads[ticket.task_id & kTaskPayloadMask] :
                      nullptr;
    ctx.result.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(ticket.task_id)));
    ctx.tensor_count = 0;
    ctx.scalar_count = 0;
    ctx.register_mask = 0;
    ctx.output_bytes = 0;
    ctx.fanin_count = 0;
    ctx.kernel_id = ticket.kernel_id;
    ctx.won = ticket.won != 0;
    ctx.joint = false;
    ctx.joint_init = false;
    ctx.joint_block = -1;
    ctx.joint_slot = -1;
    ctx.joint_count = 0;
    ctx.claim_attempted = true;

    engine_class = ExecEngineClass::Immediate;
    kernel_id = INVALID_KERNEL_ID;
    const bool kernel_ok =
        kind == DistSubmitKind::Alloc ?
            (mixed == nullptr && ticket.kernel_id == INVALID_KERNEL_ID) :
            (mixed != nullptr && dist_simt_cross_core_classify_kernel(*mixed, engine_class, kernel_id) &&
             kernel_id == ticket.kernel_id);
    const bool fields_ok = ticket.won <= 1 && ticket.ready <= 1 && ticket.won == ticket.ready;
    const bool sequence_ok = dist_simt_cross_core_task_valid(ctx) && ctx.self->local_index == ticket.task_id + 1;
    if (kernel_ok && fields_ok && sequence_ok) return true;
    return dist_simt_cross_core_fail(ticket.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
}

PTO_DEVICE_FUNC TaskOutputTensors dist_simt_cross_core_compete_first_finish(
    const MixedKernels *mixed, const DistCompeteFirstTicket &ticket, const L0TaskArgs &args, DistSubmitKind kind
) {
    DistSubmitCtx ctx;
    ExecEngineClass engine_class = ExecEngineClass::None;
    int32_t kernel_id = INVALID_KERNEL_ID;
    if (!dist_simt_cross_core_restore_ticket(ticket, mixed, kind, ctx, engine_class, kernel_id)) return ctx.result;
    const bool publisher = ticket.won != 0;
    (void)dist_simt_cross_core_finish_request(
        ctx, publisher ? &args : nullptr, kind, engine_class, kernel_id, publisher
    );
    return ctx.result;
}

PTO_DEVICE_FUNC TaskOutputTensors
dist_simt_cross_core_submit_kernel(const MixedKernels &mixed, const L0TaskArgs &args) {
    return dist_simt_cross_core_submit(&mixed, args, DistSubmitKind::Kernel);
}

PTO_DEVICE_FUNC TaskOutputTensors dist_simt_cross_core_alloc(const L0TaskArgs &args) {
    return dist_simt_cross_core_submit(nullptr, args, DistSubmitKind::Alloc);
}

PTO_DEVICE_FUNC DistCompeteFirstTicket dist_simt_cross_core_submit_compete_first_begin(const MixedKernels &mixed) {
    return dist_simt_cross_core_compete_first_begin(&mixed, DistSubmitKind::Kernel);
}

PTO_DEVICE_FUNC TaskOutputTensors dist_simt_cross_core_submit_compete_first_finish(
    const MixedKernels &mixed, const DistCompeteFirstTicket &ticket, const L0TaskArgs &args
) {
    return dist_simt_cross_core_compete_first_finish(&mixed, ticket, args, DistSubmitKind::Kernel);
}

PTO_DEVICE_FUNC DistCompeteFirstTicket dist_simt_cross_core_alloc_compete_first_begin() {
    return dist_simt_cross_core_compete_first_begin(nullptr, DistSubmitKind::Alloc);
}

PTO_DEVICE_FUNC TaskOutputTensors
dist_simt_cross_core_alloc_compete_first_finish(const DistCompeteFirstTicket &ticket, const L0TaskArgs &args) {
    return dist_simt_cross_core_compete_first_finish(nullptr, ticket, args, DistSubmitKind::Alloc);
}

}  // namespace

#endif  // PTO_FDWIC_SCHEDULER_MODE == 3 || PTO_FDWIC_SCHEDULER_MODE == 4
