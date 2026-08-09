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
using fdwic::cross_core::SimtL0TaskArgsReferenceMask;
using fdwic::cross_core::SimtL0TaskArgsRequestSource;
using fdwic::cross_core::SimtRequestPublishResult;
using fdwic::cross_core::SimtRequestReserveResult;

#if PTO_FDWIC_SCHEDULER_MODE == 4
// DAG 模式已有多路持久 Builder，Build 吞吐足以让 Execute 在 replay 后集中
// 领取；AIC/AIV 两条单调 ticket 流替代每 task 两级 tournament + owner CAS。
// Ordinary 只有一个 Builder，必须保留逐 Submit 的 Build/Execute 流水，不能
// 套用此策略。该分支只依赖调度拓扑，不检查 PA task 类型或任务形状。
constexpr uint32_t kDistSimtAicDispatchControl = 0;
constexpr uint32_t kDistSimtAivDispatchControl = 1;
static_assert(kFdwicCrossCoreTaskCapacity > kDistSimtAivDispatchControl);
#endif

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

#if PTO_FDWIC_SCHEDULER_MODE == 4
PTO_DEVICE_FUNC __gm__ volatile int64_t &dist_simt_cross_core_dispatch_cursor(ExecEngineClass engine) {
    const uint32_t control = engine == ExecEngineClass::Aic ? kDistSimtAicDispatchControl : kDistSimtAivDispatchControl;
    return dist_simt_cross_core_state().runtime.execute_owner[control].state;
}
#endif

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
    bool tournament_winner = false;
    if (!dist_cross_core_win_build_tournament(ctx, tournament_winner)) return false;
    if (!tournament_winner) return true;
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
    DistSubmitCtx &ctx, const L0TaskArgs &args, ExecEngineClass engine_class, int32_t kernel_id,
    uint32_t expected_output_count = UINT32_MAX
) {
    if (!fdwic::cross_core::ValidateSimtL0TaskArgs(args, static_cast<uint32_t>(ctx.task_id), expected_output_count)) {
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
        SimtL0TaskArgsReferenceMask(args),
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
    int32_t kernel_id, bool publisher, bool acquire_outputs = true, uint32_t expected_output_count = UINT32_MAX
) {
    if (publisher) {
        if (publisher_args == nullptr || !dist_simt_cross_core_publish_request(
                                             ctx, *publisher_args, engine_class, kernel_id, expected_output_count
                                         )) {
            return false;
        }
#if !defined(__CCE_AICORE__)
        // CPU simulation has no persistent A5 SIMT vector function. Consume
        // the same immutable request at the publication point with the proven
        // Scalar ordinary builder, preserving all output/map/exec protocols.
        // Real A5 never enters this fallback.
        bool scalar_builder = false;
        if (!dist_cross_core_reserve_build(ctx, scalar_builder) || !scalar_builder ||
            !dist_cross_core_finish_builder(ctx, *publisher_args, engine_class, kernel_id, expected_output_count)) {
            return dist_simt_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
        }
#endif
    }
    if (acquire_outputs && !dist_cross_core_acquire_outputs(ctx)) return false;
#if PTO_FDWIC_SCHEDULER_MODE == 4
    // Execute 由独立的 AIC/AIV ticket 流领取。Submit 热路径只负责发布
    // immutable request 和返回 orchestration 所需输出；不得再把最早到达
    // Submit 的 Scalar 固定成等待 Built 的 execute owner。
    (void)kind;
    (void)engine_class;
    return true;
#else
    return kind != DistSubmitKind::Kernel || dist_cross_core_bind_execution(ctx, engine_class);
#endif
}

#if PTO_FDWIC_SCHEDULER_MODE == 4
enum class DistSimtDispatchResult : uint8_t {
    Completed = 0,
    Exhausted = 1,
    Failed = 2,
};

PTO_DEVICE_FUNC DistSimtDispatchResult
dist_simt_cross_core_dispatch_one(__gm__ DistCore *self, uint32_t task_id, ExecEngineClass executor_engine) {
    if (self == nullptr || task_id >= kFdwicCrossCoreTaskCapacity ||
        (executor_engine != ExecEngineClass::Aic && executor_engine != ExecEngineClass::Aiv)) {
        (void)dist_simt_cross_core_fail(static_cast<int32_t>(task_id), PTO2_ERROR_TENSORMAP_PROTOCOL);
        return DistSimtDispatchResult::Failed;
    }
    __gm__ SharedExecCell &cell = dist_simt_cross_core_state().runtime.tasks[task_id];
    uint32_t polls = 0;
    while (true) {
        const int64_t raw_state = DistCrossCoreAicoreOps::Load(&cell.control.state);
        if (raw_state == 0) {
            // 最先完成 replay 的 Scalar 用 sealed_task_count 发布真实任务总数。
            // 尾部 executor 可能在封口前领取到第 N 个 ticket；若 N 已越界，
            // 这是正常穷尽，不能永久等待一个永远不会发布的 task cell。
            const int64_t sealed =
                DistCrossCoreAicoreOps::Load(&dist_simt_cross_core_state().lifecycle.sealed_task_count.state);
            if (sealed >= 0 && static_cast<int64_t>(task_id) >= sealed) {
                return DistSimtDispatchResult::Exhausted;
            }
            SPIN_WAIT_HINT();
            if ((++polls & 1023U) == 0 && fdwic_trace_is_fatal(static_cast<int32_t>(task_id))) {
                return DistSimtDispatchResult::Failed;
            }
            continue;
        }
        const fdwic::cross_core::DecodedExecState state = fdwic::cross_core::DecodeExecState(raw_state);
        if (!state.valid || state.task_id != task_id) {
            (void)dist_simt_cross_core_fail(static_cast<int32_t>(task_id), PTO2_ERROR_TENSORMAP_PROTOCOL);
            return DistSimtDispatchResult::Failed;
        }
        if (state.phase == ExecPhase::Building) {
            SPIN_WAIT_HINT();
            if ((++polls & 1023U) == 0 && fdwic_trace_is_fatal(static_cast<int32_t>(task_id))) {
                return DistSimtDispatchResult::Failed;
            }
            continue;
        }
        if (state.engine_class != executor_engine) {
            // AIC/AIV 各自拥有独立 ticket 流，并都会按 task id 扫描。只有
            // engine 匹配的一侧消费；另一侧在 Builder 发布真实 engine 后
            // 立即跳过，避免依赖预制的 PA task 表。
            return DistSimtDispatchResult::Completed;
        }
        if (state.phase != ExecPhase::Built) {
            // 单调 ticket 对每个 engine/task 只分配一次；匹配任务若已被
            // Claimed/Done，说明存在重复 consumer，不能静默吞掉。
            (void)dist_simt_cross_core_fail(static_cast<int32_t>(task_id), PTO2_ERROR_TENSORMAP_PROTOCOL);
            return DistSimtDispatchResult::Failed;
        }
        if (!dist_submit_wait_slot_capacity(self, static_cast<int32_t>(task_id))) {
            return DistSimtDispatchResult::Failed;
        }
        ExecToken token{};
        fdwic::cross_core::ResetExecToken(token);
        const ExecAcquireResult acquired = fdwic::cross_core::AcquireExecPayload<DistCrossCoreAicoreOps>(
            cell, task_id, static_cast<uint32_t>(self->core_idx), executor_engine, token,
            dist_simt_cross_core_state().runtime.fatal
        );
        if (acquired == ExecAcquireResult::NotBuilt || acquired == ExecAcquireResult::Lost) continue;
        if (acquired != ExecAcquireResult::Acquired) {
            (void)dist_simt_cross_core_fail(static_cast<int32_t>(task_id), PTO2_ERROR_TENSORMAP_PROTOCOL);
            return DistSimtDispatchResult::Failed;
        }
        __gm__ RingSlot *slot = dist_submit_alloc_slot(self);
        if (slot == nullptr || !dist_cross_core_build_ring_slot(*slot, token)) {
            if (slot != nullptr) {
                slot->occupied = false;
                slot->built = false;
                --self->occupied_count;
            }
            (void)dist_simt_cross_core_fail(static_cast<int32_t>(task_id), PTO2_ERROR_TENSORMAP_PROTOCOL);
            return DistSimtDispatchResult::Failed;
        }
        (void)drain_phase_b(self);
        return DistSimtDispatchResult::Completed;
    }
}

PTO_DEVICE_FUNC bool dist_simt_cross_core_run_executor(__gm__ DistCore *self) {
    const ExecEngineClass engine = dist_cross_core_executor_engine(self);
    if (engine != ExecEngineClass::Aic && engine != ExecEngineClass::Aiv) return false;
    if (dist_cross_core_is_simt_builder_worker(self)) return true;
    __gm__ volatile int64_t &cursor = dist_simt_cross_core_dispatch_cursor(engine);
    uint32_t sealed_polls = 0;
    while (!fdwic_trace_is_fatal()) {
        const int64_t raw_ticket = DistCrossCoreAicoreOps::FetchAdd(&cursor, int64_t{1});
        if (g_dist.num_workers <= 0) {
            return dist_simt_cross_core_fail(-1, PTO2_ERROR_TENSORMAP_PROTOCOL);
        }
        const uint64_t maximum_tail_ticket =
            static_cast<uint64_t>(kFdwicCrossCoreTaskCapacity) + static_cast<uint64_t>(g_dist.num_workers) - 1U;
        if (raw_ticket < 0 || static_cast<uint64_t>(raw_ticket) > maximum_tail_ticket) {
            return dist_simt_cross_core_fail(-1, PTO2_ERROR_TENSORMAP_PROTOCOL);
        }
        const uint32_t task_id = static_cast<uint32_t>(raw_ticket);
        if (task_id < kFdwicCrossCoreTaskCapacity) {
            const DistSimtDispatchResult result = dist_simt_cross_core_dispatch_one(self, task_id, engine);
            if (result == DistSimtDispatchResult::Failed) return false;
            if (result == DistSimtDispatchResult::Exhausted) return true;
            continue;
        }

        // 恰好发布满容量时，多个 executor 会并发取得 capacity、capacity+1
        // 等尾票。它们都不得越界访问 task cell；等待 seal 后统一正常退出。
        while (!fdwic_trace_is_fatal()) {
            const int64_t sealed =
                DistCrossCoreAicoreOps::Load(&dist_simt_cross_core_state().lifecycle.sealed_task_count.state);
            if (sealed >= 0) {
                if (sealed <= static_cast<int64_t>(kFdwicCrossCoreTaskCapacity)) return true;
                return dist_simt_cross_core_fail(-1, PTO2_ERROR_TENSORMAP_CAPACITY);
            }
            SPIN_WAIT_HINT();
            if ((++sealed_polls & 1023U) == 0 && fdwic_trace_is_fatal()) return false;
        }
    }
    return false;
}
#endif

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

PTO_DEVICE_FUNC bool dist_simt_cross_core_compete_first_finish_deferred(
    const MixedKernels *mixed, const DistCompeteFirstTicket &ticket, const L0TaskArgs &args, DistSubmitKind kind,
    uint32_t expected_output_count
) {
    DistSubmitCtx ctx;
    ExecEngineClass engine_class = ExecEngineClass::None;
    int32_t kernel_id = INVALID_KERNEL_ID;
    if (!dist_simt_cross_core_restore_ticket(ticket, mixed, kind, ctx, engine_class, kernel_id)) return false;
    const bool publisher = ticket.won != 0;
    return dist_simt_cross_core_finish_request(
        ctx, publisher ? &args : nullptr, kind, engine_class, kernel_id, publisher, false, expected_output_count
    );
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

PTO_DEVICE_FUNC bool dist_simt_cross_core_submit_deferred_compete_first_finish(
    const MixedKernels &mixed, const DistCompeteFirstTicket &ticket, const L0TaskArgs &args,
    uint32_t expected_output_count
) {
    return dist_simt_cross_core_compete_first_finish_deferred(
        &mixed, ticket, args, DistSubmitKind::Kernel, expected_output_count
    );
}

PTO_DEVICE_FUNC bool dist_simt_cross_core_alloc_deferred_compete_first_finish(
    const DistCompeteFirstTicket &ticket, const L0TaskArgs &args, uint32_t expected_output_count
) {
    return dist_simt_cross_core_compete_first_finish_deferred(
        nullptr, ticket, args, DistSubmitKind::Alloc, expected_output_count
    );
}

}  // namespace

#endif  // PTO_FDWIC_SCHEDULER_MODE == 3 || PTO_FDWIC_SCHEDULER_MODE == 4
