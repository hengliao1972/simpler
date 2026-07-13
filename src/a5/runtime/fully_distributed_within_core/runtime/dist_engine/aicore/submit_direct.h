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

namespace {

PTO_DEVICE_FUNC void direct_complete_task(DistSubmitCtx &ctx) {
    if (ctx.self == nullptr) return;
    complete_executed_task(ctx.self, ctx.task_id);
}

PTO_DEVICE_FUNC int32_t anchor_lane_for_mask(const ActiveMask &M) {
    if (lane_active(M, LANE_AIC)) return LANE_AIC;
    if (lane_active(M, LANE_AIV0)) return LANE_AIV0;
    if (lane_active(M, LANE_AIV1)) return LANE_AIV1;
    return LANE_NONE;
}

PTO_DEVICE_FUNC bool direct_self_is_lane(__gm__ DistCore *self, int32_t block, int32_t lane) {
    if (self == nullptr || self->block_id != block || self->lane != lane) return false;
    if (lane == LANE_AIC) return self->role == CoreType::AIC;
    if (lane == LANE_AIV0 || lane == LANE_AIV1) return self->role == CoreType::AIV;
    return false;
}

PTO_DEVICE_FUNC bool direct_claim_kernel_submit(const MixedKernels &mixed, DistSubmitCtx &ctx) {
    ctx.kernel_id = INVALID_KERNEL_ID;
    ctx.won = false;
    if (ctx.self == nullptr || ctx.task_id < 0 || ctx.task_id >= kFlagCap) return false;
    const ActiveMask M = mixed.to_active_mask();
    const uint8_t cmask = M.core_mask();
    const int32_t pc = __builtin_popcount(cmask);
    if (pc >= 2) {
        const int32_t block = ctx.self->block_id;
        if (block < 0 || block >= g_dist.num_blocks) return false;
        ctx.joint = true;
        ctx.joint_block = block;
        ctx.joint_slot = -1;
        ctx.joint_count = pc;
        const int32_t anchor_lane = anchor_lane_for_mask(M);
        if (!direct_self_is_lane(ctx.self, block, anchor_lane)) return false;
        __gm__ PaddedCursor *cursors = anchor_lane == LANE_AIC ? g_dist.cube_cursor : g_dist.vector_cursor;
        ctx.won = claim(cursors[ctx.task_id % kCursorShards].v, ctx.task_id);
        if (!ctx.won) return false;
        ctx.kernel_id = kernel_id_for_lane(mixed, anchor_lane);
        ctx.joint_init = true;
        return true;
    }
    if (lane_active(M, LANE_AIC)) {
        if (ctx.self->role != CoreType::AIC) return false;
        ctx.won = claim(g_dist.cube_cursor[ctx.task_id % kCursorShards].v, ctx.task_id);
        if (!ctx.won) return false;
        ctx.kernel_id = mixed.aic_kernel_id;
        return true;
    }
    if (lane_active(M, LANE_AIV0) || lane_active(M, LANE_AIV1)) {
        if (ctx.self->role != CoreType::AIV) return false;
        ctx.won = claim(g_dist.vector_cursor[ctx.task_id % kCursorShards].v, ctx.task_id);
        if (!ctx.won) return false;
        const int32_t own_lane = lane_active(M, LANE_AIV0) ? LANE_AIV0 : LANE_AIV1;
        ctx.kernel_id = kernel_id_for_lane(mixed, own_lane);
        return true;
    }
    return false;
}

PTO_DEVICE_FUNC bool direct_claim_alloc_submit(DistSubmitCtx &ctx) {
    ctx.kernel_id = INVALID_KERNEL_ID;
    if (ctx.self == nullptr || ctx.task_id < 0 || ctx.task_id >= kFlagCap) return false;
    ctx.won = claim(g_dist.alloc_cursor[ctx.task_id % kCursorShards].v, ctx.task_id);
    return ctx.won;
}

PTO_DEVICE_FUNC bool direct_claim_submit(DistSubmitKind kind, const MixedKernels *mixed, DistSubmitCtx &ctx) {
    if (kind == DistSubmitKind::Alloc) return direct_claim_alloc_submit(ctx);
    if (mixed == nullptr) return false;
    return direct_claim_kernel_submit(*mixed, ctx);
}

PTO_DEVICE_FUNC bool direct_drain_block_won(__gm__ DistCore *self);
PTO_DEVICE_FUNC int32_t direct_drain_ready_slots(__gm__ DistCore *self);

PTO_DEVICE_FUNC __gm__ RingSlot *direct_alloc_slot(__gm__ DistCore *self) {
    if (self == nullptr) return nullptr;
    const int32_t si = alloc_ring_slot(self);
    if (si < 0) return nullptr;
    __gm__ RingSlot &slot = self->slots[si];
    slot.occupied = true;
    slot.built = false;
    self->occupied_count++;
    dist_aicore_flush_region(&slot, sizeof(RingSlot));
    return &slot;
}

PTO_DEVICE_FUNC void direct_wait_slot_capacity(__gm__ DistCore *self, int32_t task_id) {
    if (self == nullptr) return;
    bool waited = false;
    TRACE_SPAN_BEGIN(ring_bp_trace);
    while (self->occupied_count >= kPrivateSlots - kWonReserve) {
        waited = true;
        direct_drain_block_won(self);
        if (direct_drain_ready_slots(self) == 0) SPIN_WAIT_HINT();
    }
    if (waited) {
        TRACE_SPAN_END(ring_bp_trace, self, task_id, -1, TracePhase::RingBp, 0, 0);
    }
}

PTO_DEVICE_FUNC bool direct_fatal_set() {
#if defined(__CCE_AICORE__)
    dist_aicore_invalidate_region(const_cast<__gm__ int32_t *>(&g_dist.fatal), sizeof(g_dist.fatal));
    return g_dist.fatal != 0;
#else
    return fatal_set();
#endif
}

PTO_DEVICE_FUNC void direct_set_fatal() {
#if defined(__CCE_AICORE__)
    g_dist.fatal = 1;
    dist_aicore_flush_region(const_cast<__gm__ int32_t *>(&g_dist.fatal), sizeof(g_dist.fatal));
#else
    set_fatal();
#endif
}

PTO_DEVICE_FUNC bool direct_wait_heap_capacity(DistSubmitCtx &ctx, DistSubmitKind kind) {
    if (ctx.self == nullptr || ctx.output_bytes == 0 || g_dist.heap_base == nullptr) return true;
    const size_t ring = g_dist.heap_size;
    bool waited = false;
    TRACE_SPAN_BEGIN(heap_bp_trace);
    while (!direct_fatal_set()) {
        __gm__ int64_t *frontier = const_cast<__gm__ int64_t *>(&g_dist.frontier);
        dist_aicore_invalidate_region(frontier, 64);
        const int32_t f = static_cast<int32_t>(g_dist.frontier);
        const int32_t R = f - g_dist.H;
        const uint64_t vstart_live = load_task_vend(R);
        if (ctx.self->heap_next - vstart_live <= ring) {
            if (waited) {
                TRACE_SPAN_END(heap_bp_trace, ctx.self, ctx.task_id, -1, TracePhase::RingBp, 0, 1);
            }
            return true;
        }
        if (f >= ctx.task_id - 1) {
            direct_set_fatal();
            if (kind == DistSubmitKind::Alloc) {
                DIST_ERRF(
                    "[dist_engine] heap ring %zu B too small for H=%d window at alloc %d (live=%llu B)\n", ring,
                    g_dist.H, ctx.task_id, (unsigned long long)(ctx.self->heap_next - vstart_live)
                );
            } else {
                DIST_ERRF(
                    "[dist_engine] heap ring %zu B too small for H=%d window at task %d (live=%llu B); "
                    "enlarge PTO_DIST_HEAP_MB or reduce PTO_DIST_H\n",
                    ring, g_dist.H, ctx.task_id, (unsigned long long)(ctx.self->heap_next - vstart_live)
                );
            }
            return false;
        }
        waited = true;
        direct_drain_block_won(ctx.self);
        if (direct_drain_ready_slots(ctx.self) == 0) SPIN_WAIT_HINT();
    }
    if (waited) {
        TRACE_SPAN_END(heap_bp_trace, ctx.self, ctx.task_id, -1, TracePhase::RingBp, 0, 1);
    }
    return false;
}

PTO_DEVICE_FUNC int32_t direct_alloc_won_slot(__gm__ DistCore *self, int32_t block) {
#if !defined(__CCE_AICORE__)
    (void)self;
    return alloc_won_slot(block);
#else
    __gm__ BlockWon &bw = g_dist.blocks[block];
    while (true) {
        dist_aicore_invalidate_region(&bw, sizeof(BlockWon));
        for (int32_t i = 0; i < kPrivateSlots; i++) {
            if (bw.slots[i].state != 0) continue;
            bw.slots[i].state = 2;
            dist_aicore_flush_region(&bw.slots[i], sizeof(WonSlot));
            return i;
        }
        direct_drain_block_won(self);
        direct_drain_ready_slots(self);
        SPIN_WAIT_HINT();
    }
#endif
}

PTO_DEVICE_FUNC void direct_publish_joint_deposits(DistSubmitCtx &ctx, const MixedKernels &mixed) {
    if (!ctx.joint) return;
    __gm__ WonSlot &w = g_dist.blocks[ctx.joint_block].slots[ctx.joint_slot];
    const ActiveMask M = mixed.to_active_mask();
    populate_won_slot(
        w, ctx.task_id, M, mixed, ctx.self->lane,
#if defined(__CCE_AICORE__)
        nullptr,
#else
        g_dist.runtime,
#endif
        ctx.payload->tensors, ctx.tensor_count, ctx.payload->scalars, ctx.payload->scalar_count, ctx.fanin,
        ctx.fanin_count
    );
    w.state = 1;
    g_dist.blocks[ctx.joint_block].any_pub = 1;
    dist_aicore_flush_region(&w, sizeof(WonSlot));
    dist_aicore_flush_region(
        const_cast<__gm__ int32_t *>(&g_dist.blocks[ctx.joint_block].any_pub),
        sizeof(g_dist.blocks[ctx.joint_block].any_pub)
    );
}

PTO_DEVICE_FUNC bool direct_drain_won_slot(__gm__ DistCore *self, int32_t won_slot) {
    if (self == nullptr || self->lane == LANE_AIC || self->lane == LANE_NONE) return false;
    __gm__ WonSlot &w = g_dist.blocks[self->block_id].slots[won_slot];
    dist_aicore_invalidate_region(&w, sizeof(WonSlot));
    if (w.state != 1 || !w.lane[self->lane].present || w.drained[self->lane].v != 0) return false;
    __gm__ RingSlot *slot = direct_alloc_slot(self);
    if (slot == nullptr) return false;
    w.drained[self->lane].v = 1;
    dist_aicore_flush_region(&w.drained[self->lane], sizeof(DrainedCell));
    __gm__ BuiltSubtask &b = w.lane[self->lane];
    TRACE_SPAN_BEGIN(drain_won_trace);
    build_ring_slot(
        *slot, w.task_id, b.func_id, b.function_bin_addr, b.tensors, b.tensor_count, b.scalars, b.scalar_count, b.fanin,
        b.fanin_count, b.sub_block_id, /*is_multicore=*/true, self->block_id, won_slot
    );
    TRACE_SPAN_END(drain_won_trace, self, w.task_id, b.func_id, TracePhase::DrainWon, /*flags=*/1u, won_slot);
    dist_aicore_flush_region(slot, sizeof(RingSlot));
    return true;
}

PTO_DEVICE_FUNC bool direct_drain_block_won(__gm__ DistCore *self) {
    if (self == nullptr || self->lane == LANE_AIC || self->lane == LANE_NONE) return false;
    __gm__ BlockWon &bw = g_dist.blocks[self->block_id];
    dist_aicore_invalidate_region(&bw, sizeof(BlockWon));
    if (bw.any_pub == 0) return false;
    bool drained = false;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        drained = direct_drain_won_slot(self, i) || drained;
    }
    return drained;
}

PTO_DEVICE_FUNC bool direct_build_winner_slot(DistSubmitCtx &ctx, __gm__ RingSlot *slot) {
    if (slot == nullptr || ctx.payload == nullptr) return false;
    const int32_t sub_block_id = ctx.self != nullptr && ctx.self->lane == LANE_AIV1 ? 1 : 0;
    const uint64_t fn_addr = dist_aicore_slot_function_addr(g_dist.runtime, ctx.kernel_id);
    build_ring_slot(
        *slot, ctx.task_id, ctx.kernel_id, fn_addr, ctx.payload->tensors, ctx.tensor_count, ctx.payload->scalars,
        ctx.payload->scalar_count, ctx.fanin, ctx.fanin_count, sub_block_id, ctx.joint, ctx.joint_block, ctx.joint_slot
    );
    dist_aicore_flush_region(slot, sizeof(RingSlot));
    return true;
}

PTO_DEVICE_FUNC bool direct_slot_fanin_ready(__gm__ const RingSlot &slot) {
    for (int32_t i = 0; i < slot.fanin_count; i++) {
        if (!task_flag_ready(slot.fanin[i], __ATOMIC_ACQUIRE)) return false;
    }
    return true;
}

PTO_DEVICE_FUNC bool direct_try_execute_slot(__gm__ RingSlot &slot, __gm__ DistCore *self) {
    if (!slot.occupied || !slot.built || !direct_slot_fanin_ready(slot)) return false;
    execute_slot(self, slot);
    if (self != nullptr && self->occupied_count > 0) self->occupied_count--;
    return true;
}

PTO_DEVICE_FUNC int32_t direct_drain_ready_slots(__gm__ DistCore *self) {
    if (self == nullptr || self->occupied_count == 0) return 0;
    int32_t freed = 0;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ RingSlot &slot = self->slots[i];
        dist_aicore_invalidate_region(&slot, sizeof(RingSlot));
        if (direct_try_execute_slot(slot, self)) freed++;
    }
    return freed;
}

PTO_DEVICE_FUNC void direct_build_won_submit(DistSubmitCtx &ctx, const MixedKernels &mixed) {
    if (ctx.self == nullptr) return;
    direct_wait_slot_capacity(ctx.self, ctx.task_id);
    if (!direct_wait_heap_capacity(ctx, DistSubmitKind::Kernel)) return;
    if (ctx.joint && ctx.joint_slot < 0) {
        ctx.joint_slot = direct_alloc_won_slot(ctx.self, ctx.joint_block);
    }
    __gm__ RingSlot *slot = direct_alloc_slot(ctx.self);
    if (slot == nullptr) return;

    if (ctx.joint) direct_publish_joint_deposits(ctx, mixed);
    if (!direct_build_winner_slot(ctx, slot)) return;
}

PTO_DEVICE_FUNC void direct_complete_alloc_submit(DistSubmitCtx &ctx) {
    if (ctx.won) {
        if (!direct_wait_heap_capacity(ctx, DistSubmitKind::Alloc)) return;
        direct_complete_task(ctx);
    }
}

#include "dist_engine/aicore/run_state.h"

PTO_DEVICE_FUNC bool direct_has_pending_won(__gm__ DistCore *self) {
    if (self == nullptr || self->lane == LANE_AIC || self->lane == LANE_NONE) return false;
    __gm__ BlockWon &bw = g_dist.blocks[self->block_id];
    dist_aicore_invalidate_region(&bw, sizeof(BlockWon));
    if (bw.any_pub == 0) return false;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ WonSlot &w = bw.slots[i];
        if (w.state != 1) continue;
        if (w.lane[self->lane].present && w.drained[self->lane].v == 0) return true;
    }
    return false;
}

PTO_DEVICE_FUNC void direct_drain_to_completion(__gm__ DistCore *self) {
    if (self == nullptr) return;
    __gm__ int64_t *replay_done = const_cast<__gm__ int64_t *>(&g_dist.replay_done);
    publish_replay_done(*replay_done);
    while (true) {
        direct_drain_block_won(self);
        const int32_t freed = direct_drain_ready_slots(self);
        dist_aicore_invalidate_region(replay_done, sizeof(g_dist.replay_done));
        const bool all_replayed = g_dist.replay_done >= g_dist.num_workers;
        const bool ring_empty = self->occupied_count == 0;
        const bool pending = direct_has_pending_won(self);
        if (all_replayed && ring_empty && !pending) break;
        if (freed == 0) SPIN_WAIT_HINT();
    }
}

PTO_DEVICE_FUNC void direct_replay_orch(__gm__ Runtime *runtime) {
    __gm__ DistCore *self = g_self;
    (void)self;
#if defined(__CCE_AICORE__)
    dist_aicore_invalidate_region(runtime->dist.ccec_orch_tensors, sizeof(runtime->dist.ccec_orch_tensors));
    dist_aicore_invalidate_region(runtime->dist.ccec_orch_scalars, sizeof(runtime->dist.ccec_orch_scalars));
    dist_aicore_invalidate_region(const_cast<__gm__ const int32_t *>(&runtime->dist.ccec_orch_tensor_count), 64);
    if (aicpu_orchestration_entry == nullptr || !ccec_is_valid_worker()) {
        return;
    }
    L2TaskArgs local_args;
    Tensor local_tensors[CHIP_MAX_TENSOR_ARGS];
    const int32_t tensor_count = runtime->dist.ccec_orch_tensor_count;
    const int32_t scalar_count = runtime->dist.ccec_orch_scalar_count;
    for (int32_t i = 0; i < tensor_count && i < CHIP_MAX_TENSOR_ARGS; i++) {
        Tensor::copy(local_tensors[i], runtime->dist.ccec_orch_tensors[i]);
        local_args.add_input(local_tensors[i]);
    }
    for (int32_t i = 0; i < scalar_count && i < CHIP_MAX_SCALAR_ARGS; i++) {
        const uint64_t scalar = runtime->dist.ccec_orch_scalars[i];
        local_args.add_scalar(scalar);
    }
    aicpu_orchestration_entry(local_args);
#else
    (void)runtime;
    if (g_dist.orch_args != nullptr && !fatal_set()) {
        aicpu_orchestration_entry(*g_dist.orch_args);
    }
#endif
}

#include "dist_engine/aicore/onboard_entry.h"

}  // namespace

DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors
dist_submit_impl(PTO2Runtime *, const MixedKernels &mixed, const L0TaskArgs &args) {
    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, args, ctx);
    TRACE_LAP_RESET(ctx.self);
    direct_drain_block_won(ctx.self);
    direct_drain_ready_slots(ctx.self);
    TRACE_LAP(ctx.self, ctx.task_id, -1, TracePhase::EfDrain);
    if (!dist_submit_materialize_and_prepare_map(ctx.self, args, ctx, DistSubmitKind::Kernel)) return ctx.result;
    const bool is_winner = direct_claim_submit(DistSubmitKind::Kernel, &mixed, ctx);
    if (is_winner) {
        ctx.fanin_count = dist_submit_collect_fanin(ctx, ctx.fanin);
    }
    dist_submit_register_outputs(ctx, /*include_existing=*/true);
    if (is_winner) {
        TRACE_LAP(ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Build);
        direct_build_won_submit(ctx, mixed);
    } else {
        TRACE_LAP(ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Replay);
        direct_drain_block_won(ctx.self);
    }
    return ctx.result;
}

DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors dist_alloc_tensors(PTO2Runtime *, const L0TaskArgs &args) {
    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, args, ctx);
    TRACE_LAP_RESET(ctx.self);
    direct_drain_block_won(ctx.self);
    direct_drain_ready_slots(ctx.self);
    TRACE_LAP(ctx.self, ctx.task_id, -1, TracePhase::EfDrain);
    if (!dist_submit_materialize_and_prepare_map(ctx.self, args, ctx, DistSubmitKind::Alloc)) return ctx.result;
    dist_submit_register_outputs(ctx, /*include_existing=*/false);
    const bool is_winner = direct_claim_submit(DistSubmitKind::Alloc, nullptr, ctx);
    if (is_winner) {
        direct_complete_alloc_submit(ctx);
        TRACE_LAP(ctx.self, ctx.task_id, -1, TracePhase::Alloc);
    } else {
        TRACE_LAP(ctx.self, ctx.task_id, -1, TracePhase::Replay);
    }
    return ctx.result;
}
