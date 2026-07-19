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

PTO_DEVICE_FUNC int32_t anchor_lane_for_mask(const ActiveMask &M) {
    if (lane_active(M, LANE_AIC)) return LANE_AIC;
    if (lane_active(M, LANE_AIV0)) return LANE_AIV0;
    if (lane_active(M, LANE_AIV1)) return LANE_AIV1;
    return LANE_NONE;
}

PTO_DEVICE_FUNC bool dist_submit_self_is_lane(__gm__ DistCore *self, int32_t block, int32_t lane) {
    if (self == nullptr || self->block_id != block || self->lane != lane) return false;
    if (lane == LANE_AIC) return self->role == CoreType::AIC;
    if (lane == LANE_AIV0 || lane == LANE_AIV1) return self->role == CoreType::AIV;
    return false;
}

PTO_DEVICE_FUNC bool dist_submit_claim_kernel(const MixedKernels &mixed, DistSubmitCtx &ctx) {
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
        if (!dist_submit_self_is_lane(ctx.self, block, anchor_lane)) return false;
        __gm__ PaddedCursor *cursors = anchor_lane == LANE_AIC ? g_dist.cube_cursor : g_dist.vector_cursor;
        ctx.claim_attempted = true;
        ctx.won = claim(cursors[ctx.task_id % kCursorShards].v, ctx.task_id);
        if (!ctx.won) return false;
        ctx.kernel_id = kernel_id_for_lane(mixed, anchor_lane);
        ctx.joint_init = true;
        return true;
    }
    if (lane_active(M, LANE_AIC)) {
        if (ctx.self->role != CoreType::AIC) return false;
        ctx.claim_attempted = true;
        ctx.won = claim(g_dist.cube_cursor[ctx.task_id % kCursorShards].v, ctx.task_id);
        if (!ctx.won) return false;
        ctx.kernel_id = mixed.aic_kernel_id;
        return true;
    }
    if (lane_active(M, LANE_AIV0) || lane_active(M, LANE_AIV1)) {
        if (ctx.self->role != CoreType::AIV) return false;
        ctx.claim_attempted = true;
        ctx.won = claim(g_dist.vector_cursor[ctx.task_id % kCursorShards].v, ctx.task_id);
        if (!ctx.won) return false;
        const int32_t own_lane = lane_active(M, LANE_AIV0) ? LANE_AIV0 : LANE_AIV1;
        ctx.kernel_id = kernel_id_for_lane(mixed, own_lane);
        return true;
    }
    return false;
}

PTO_DEVICE_FUNC bool dist_submit_claim_alloc(DistSubmitCtx &ctx) {
    ctx.kernel_id = INVALID_KERNEL_ID;
    if (ctx.self == nullptr || ctx.task_id < 0 || ctx.task_id >= kFlagCap) return false;
    ctx.claim_attempted = true;
    ctx.won = claim(g_dist.alloc_cursor[ctx.task_id % kCursorShards].v, ctx.task_id);
    return ctx.won;
}

PTO_DEVICE_FUNC bool dist_submit_claim(DistSubmitKind kind, const MixedKernels *mixed, DistSubmitCtx &ctx) {
    if (kind == DistSubmitKind::Alloc) return dist_submit_claim_alloc(ctx);
    if (mixed == nullptr) return false;
    return dist_submit_claim_kernel(*mixed, ctx);
}

PTO_DEVICE_FUNC __gm__ RingSlot *dist_submit_alloc_slot(__gm__ DistCore *self) {
    if (self == nullptr) return nullptr;
    const int32_t si = alloc_ring_slot(self);
    if (si < 0) return nullptr;
    __gm__ RingSlot &slot = self->slots[si];
    slot.occupied = true;
    slot.built = false;
    self->occupied_count++;
    return &slot;
}

PTO_DEVICE_FUNC void dist_submit_wait_slot_capacity(__gm__ DistCore *self, int32_t task_id) {
    if (self == nullptr) return;
    bool waited = false;
    TRACE_SPAN_BEGIN(ring_bp_trace);
    const uint32_t slot_poll_region = fdwic_atomic_poll_region_begin(
        fdwic_atomic_site_mask(FdwicAtomicSite::FaninFlagLoad) | fdwic_atomic_block_won_poll_mask()
    );
    while (self->occupied_count >= kPrivateSlots - kWonReserve) {
        waited = true;
        drain_block_won(self);
        if (drain_phase_b(self) == 0) SPIN_WAIT_HINT();
    }
    fdwic_atomic_poll_region_end(slot_poll_region);
    if (waited) {
        TRACE_SPAN_END(ring_bp_trace, self, task_id, -1, TracePhase::RingBp, 0, 0);
    }
}

PTO_DEVICE_FUNC bool dist_submit_wait_heap_capacity(DistSubmitCtx &ctx, DistSubmitKind kind) {
    if (ctx.self == nullptr || ctx.output_bytes == 0 || g_dist.heap_base == nullptr) return true;
    const size_t ring = g_dist.heap_size;
    bool waited = false;
    TRACE_SPAN_BEGIN(heap_bp_trace);
    bool heap_poll_region_active = false;
    uint32_t heap_poll_region = 0;
    while (!fdwic_trace_is_fatal(ctx.task_id)) {
        // 逻辑 heap 尚未走完第一圈时，物理地址还没有发生环形复用；保留 fatal
        // 原子检查后，可直接跳过 frontier/vend 原子读取。
        if (ctx.self->heap_next <= ring) {
            if (heap_poll_region_active) fdwic_atomic_poll_region_end(heap_poll_region);
            return true;
        }
        if (!heap_poll_region_active) {
            heap_poll_region = fdwic_atomic_poll_region_begin(
                fdwic_atomic_site_mask(FdwicAtomicSite::FatalPoll) |
                fdwic_atomic_site_mask(FdwicAtomicSite::HeapFrontierLoad) |
                fdwic_atomic_site_mask(FdwicAtomicSite::HeapVendLoad) |
                fdwic_atomic_site_mask(FdwicAtomicSite::FaninFlagLoad) | fdwic_atomic_block_won_poll_mask()
            );
            heap_poll_region_active = true;
        }
        const int32_t f = static_cast<int32_t>(fdwic_trace_atomic_load(
            ctx.task_id, FdwicAtomicSite::HeapFrontierLoad, g_dist.frontier, /*result_used=*/true
        ));
        const int32_t R = f - g_dist.H;
        const uint64_t vstart_live = load_task_vend(ctx.task_id, R);
        if (ctx.self->heap_next - vstart_live <= ring) {
            fdwic_atomic_poll_region_end(heap_poll_region);
            if (waited) {
                TRACE_SPAN_END(heap_bp_trace, ctx.self, ctx.task_id, -1, TracePhase::RingBp, 0, 1);
            }
            return true;
        }
        if (f >= ctx.task_id - 1) {
            fdwic_trace_set_fatal(ctx.task_id);
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
            fdwic_atomic_poll_region_end(heap_poll_region);
            return false;
        }
        waited = true;
        drain_block_won(ctx.self);
        if (drain_phase_b(ctx.self) == 0) SPIN_WAIT_HINT();
    }
    if (heap_poll_region_active) fdwic_atomic_poll_region_end(heap_poll_region);
    if (waited) {
        TRACE_SPAN_END(heap_bp_trace, ctx.self, ctx.task_id, -1, TracePhase::RingBp, 0, 1);
    }
    return false;
}

PTO_DEVICE_FUNC void publish_joint_deposits(DistSubmitCtx &ctx, const MixedKernels &mixed, const L0TaskArgs &args) {
    if (!ctx.joint) return;
    __gm__ WonSlot &w = g_dist.blocks[ctx.joint_block].slots[ctx.joint_slot];
    const ActiveMask M = mixed.to_active_mask();
    populate_won_slot_from_submit(
        w, ctx.task_id, M, mixed, ctx.self->lane,
#if defined(__CCE_AICORE__)
        nullptr,
#else
        g_dist.runtime,
#endif
        args, ctx, ctx.fanin, ctx.fanin_count
    );
#if defined(__CCE_AICORE__)
    dist_aicore_flush_region(&w.meta, sizeof(w.meta));
    dist_aicore_flush_region(w.lane, sizeof(w.lane));
#endif
    store_won_remaining(w, ctx.joint_count, ctx.task_id);
    publish_won_slot(w, ctx.task_id);
    (void)fdwic_trace_atomic_exchange(
        ctx.task_id, FdwicAtomicSite::WonAnyPublishExchange, g_dist.blocks[ctx.joint_block].any_pub, int32_t{1},
        /*result_used=*/false
    );
}

PTO_DEVICE_FUNC int32_t wait_alloc_won_slot(__gm__ DistCore *self, int32_t block, int32_t task_id) {
    int32_t won_slot = alloc_won_slot(block, task_id);
    const uint32_t won_slot_poll_region = fdwic_atomic_poll_region_begin(
        fdwic_atomic_site_mask(FdwicAtomicSite::FatalPoll) | fdwic_atomic_site_mask(FdwicAtomicSite::FaninFlagLoad) |
        fdwic_atomic_block_won_poll_mask()
    );
    while (won_slot < 0 && !fdwic_trace_is_fatal(task_id)) {
        drain_block_won(self);
        if (drain_phase_b(self) == 0) SPIN_WAIT_HINT();
        won_slot = alloc_won_slot(block, task_id);
    }
    fdwic_atomic_poll_region_end(won_slot_poll_region);
    return won_slot;
}

PTO_DEVICE_FUNC bool dist_submit_build_winner_slot(DistSubmitCtx &ctx, const L0TaskArgs &args, __gm__ RingSlot *slot) {
    if (slot == nullptr || ctx.payload == nullptr) return false;
    const int32_t sub_block_id = ctx.self != nullptr && ctx.self->lane == LANE_AIV1 ? 1 : 0;
    const uint64_t fn_addr = dist_aicore_slot_function_addr(g_dist.runtime, ctx.kernel_id);
    build_ring_slot_from_submit(
        *slot, ctx.task_id, ctx.kernel_id, fn_addr, args, ctx, ctx.fanin, ctx.fanin_count, sub_block_id, ctx.joint,
        ctx.joint_block, ctx.joint_slot
    );
    return true;
}

PTO_DEVICE_FUNC void
dist_submit_build_winner_task(DistSubmitCtx &ctx, const MixedKernels &mixed, const L0TaskArgs &args) {
    if (ctx.self == nullptr) return;
    dist_submit_wait_slot_capacity(ctx.self, ctx.task_id);
    if (!dist_submit_wait_heap_capacity(ctx, DistSubmitKind::Kernel)) return;
    if (ctx.joint && ctx.joint_slot < 0) {
        ctx.joint_slot = wait_alloc_won_slot(ctx.self, ctx.joint_block, ctx.task_id);
        if (ctx.joint_slot < 0) return;
    }
    __gm__ RingSlot *slot = dist_submit_alloc_slot(ctx.self);
    if (slot == nullptr) return;

    if (ctx.joint) publish_joint_deposits(ctx, mixed, args);
    if (!dist_submit_build_winner_slot(ctx, args, slot)) return;
}

PTO_DEVICE_FUNC void dist_submit_complete_alloc(DistSubmitCtx &ctx) {
    if (ctx.won) {
        if (!dist_submit_wait_heap_capacity(ctx, DistSubmitKind::Alloc)) return;
        if (ctx.self != nullptr) complete_executed_task(ctx.self, ctx.task_id);
    }
}

#include "dist_engine/aicore/run_state.h"

PTO_DEVICE_FUNC void dist_submit_drain_to_completion(__gm__ DistCore *self) {
    if (self == nullptr) return;
    (void)fdwic_trace_atomic_fetch_add<int64_t>(
        -1, FdwicAtomicSite::ReplayDoneIncrement, g_dist.replay_done, 1, /*result_used=*/false
    );
    const uint32_t final_poll_region = fdwic_atomic_poll_region_begin(
        fdwic_atomic_site_mask(FdwicAtomicSite::ReplayDonePoll) |
        fdwic_atomic_site_mask(FdwicAtomicSite::FaninFlagLoad) | fdwic_atomic_block_won_poll_mask()
    );
    while (true) {
        drain_block_won(self);
        const int32_t freed = drain_phase_b(self);
        const bool all_replayed =
            fdwic_trace_atomic_load(-1, FdwicAtomicSite::ReplayDonePoll, g_dist.replay_done) >= g_dist.num_workers;
        const bool ring_empty = self->occupied_count == 0;
        const bool pending = has_pending_won(self);
        if (all_replayed && ring_empty && !pending) break;
        if (freed == 0) SPIN_WAIT_HINT();
    }
    fdwic_atomic_poll_region_end(final_poll_region);
}

PTO_DEVICE_FUNC void dist_submit_replay_orch(__gm__ Runtime *runtime) {
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
    if (g_dist.orch_args != nullptr && !fdwic_trace_is_fatal()) {
        aicpu_orchestration_entry(*g_dist.orch_args);
    }
#endif
}

#include "dist_engine/aicore/onboard_entry.h"

}  // namespace

DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors
dist_submit_impl(PTO2Runtime *, const MixedKernels &mixed, const L0TaskArgs &args) {
    const ActiveMask active = mixed.to_active_mask();
    if (__builtin_popcount(active.core_mask()) >= 2) g_fdwic_joint_submit_seen = true;
    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, args, ctx);
    TRACE_SPAN_BEGIN(submit_trace);
    TRACE_LAP_RESET(ctx.self);
    drain_block_won(ctx.self);
    drain_phase_b(ctx.self);
    TRACE_LAP(ctx.self, ctx.task_id, -1, TracePhase::EfDrain);
    if (!dist_submit_materialize_and_prepare_map(ctx.self, args, ctx, DistSubmitKind::Kernel)) return ctx.result;
    TRACE_SPAN_BEGIN(claim_trace);
    const bool is_winner = dist_submit_claim(DistSubmitKind::Kernel, &mixed, ctx);
    const uint32_t claim_flags =
        fdwic_atomic_swimlane_enabled() ?
            (is_winner ? kFdwicClaimWon : 0U) | (ctx.claim_attempted ? kFdwicClaimAttempted : 0U) :
            static_cast<uint32_t>(is_winner);
    TRACE_SPAN_END(claim_trace, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Claim, claim_flags, 0);
    if (is_winner) {
        TRACE_SPAN_BEGIN(fanin_trace);
        ctx.fanin_count = dist_submit_collect_fanin(args, ctx, ctx.fanin);
        TRACE_SPAN_END(
            fanin_trace, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Fanin, 0,
            static_cast<uint32_t>(ctx.fanin_count)
        );
    }
    TRACE_SPAN_BEGIN(register_trace);
    dist_submit_register_outputs(ctx, args, /*include_existing=*/true);
    TRACE_SPAN_END(register_trace, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Register, 0, 1);
    if (__builtin_expect(is_winner, 0)) {
        TRACE_LAP(ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Build);
        dist_submit_build_winner_task(ctx, mixed, args);
    } else {
        TRACE_LAP(ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Replay);
        drain_block_won(ctx.self);
    }
    TRACE_SPAN_END(
        submit_trace, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Submit, static_cast<uint32_t>(is_winner), 0
    );
    return ctx.result;
}

DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors dist_alloc_tensors(PTO2Runtime *, const L0TaskArgs &args) {
    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, args, ctx);
    TRACE_SPAN_BEGIN(submit_trace);
    TRACE_LAP_RESET(ctx.self);
    drain_block_won(ctx.self);
    drain_phase_b(ctx.self);
    TRACE_LAP(ctx.self, ctx.task_id, -1, TracePhase::EfDrain);
    if (!dist_submit_materialize_and_prepare_map(ctx.self, args, ctx, DistSubmitKind::Alloc)) return ctx.result;
    TRACE_SPAN_BEGIN(register_trace);
    dist_submit_register_outputs(ctx, args, /*include_existing=*/false);
    TRACE_SPAN_END(register_trace, ctx.self, ctx.task_id, -1, TracePhase::Register, 0, 0);
    TRACE_SPAN_BEGIN(claim_trace);
    const bool is_winner = dist_submit_claim(DistSubmitKind::Alloc, nullptr, ctx);
    const uint32_t claim_flags =
        fdwic_atomic_swimlane_enabled() ?
            (is_winner ? kFdwicClaimWon : 0U) | (ctx.claim_attempted ? kFdwicClaimAttempted : 0U) :
            static_cast<uint32_t>(is_winner);
    TRACE_SPAN_END(claim_trace, ctx.self, ctx.task_id, -1, TracePhase::Claim, claim_flags, 1);
    if (__builtin_expect(is_winner, 0)) {
        dist_submit_complete_alloc(ctx);
        TRACE_LAP(ctx.self, ctx.task_id, -1, TracePhase::Alloc);
    } else {
        TRACE_LAP(ctx.self, ctx.task_id, -1, TracePhase::Replay);
    }
    TRACE_SPAN_END(submit_trace, ctx.self, ctx.task_id, -1, TracePhase::Submit, static_cast<uint32_t>(is_winner), 1);
    return ctx.result;
}
