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

PTO_DEVICE_FUNC DistCompeteFirstTicket
dist_submit_make_ticket(const DistSubmitCtx &ctx, DistSubmitKind kind, uint64_t submit_begin, bool ready) {
    DistCompeteFirstTicket ticket;
    ticket.submit_begin = submit_begin;
    ticket.task_id = ctx.task_id;
    ticket.kernel_id = ctx.kernel_id;
    ticket.joint_block = ctx.joint_block;
    ticket.joint_count = ctx.joint_count;
    ticket.won = static_cast<uint8_t>(ctx.won);
    ticket.joint = static_cast<uint8_t>(ctx.joint);
    ticket.joint_init = static_cast<uint8_t>(ctx.joint_init);
    ticket.claim_attempted = static_cast<uint8_t>(ctx.claim_attempted);
    ticket.ready = static_cast<uint8_t>(ready);
    ticket.kind = static_cast<uint8_t>(
        kind == DistSubmitKind::Alloc ? DistCompeteFirstKind::Alloc : DistCompeteFirstKind::Kernel
    );
    ticket.reserved = 0;
    return ticket;
}

PTO_DEVICE_FUNC void
dist_submit_restore_from_ticket(const DistCompeteFirstTicket &ticket, const L0TaskArgs &args, DistSubmitCtx &ctx) {
    ctx.self = g_self;
    ctx.task_id = ticket.task_id;
    ctx.payload = ctx.self != nullptr && ctx.task_id >= 0 && ctx.task_id < kFlagCap ?
                      &ctx.self->task_payloads[ctx.task_id & kTaskPayloadMask] :
                      nullptr;
    ctx.result.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(ctx.task_id)));
    ctx.tensor_count = args.tensor_count();
    ctx.scalar_count = args.scalar_count();
    ctx.register_mask = 0;
    ctx.output_bytes = 0;
    ctx.fanin_count = 0;
    ctx.kernel_id = ticket.kernel_id;
    ctx.won = ticket.won != 0;
    ctx.joint = ticket.joint != 0;
    ctx.joint_init = ticket.joint_init != 0;
    ctx.joint_block = ticket.joint_block;
    // Begin cannot reserve a WonSlot because that can block and requires the
    // materialized task. The winner allocates it later in the existing tail.
    ctx.joint_slot = -1;
    ctx.joint_count = ticket.joint_count;
    ctx.claim_attempted = ticket.claim_attempted != 0;
}

PTO_DEVICE_FUNC bool dist_submit_validate_ticket(
    const DistCompeteFirstTicket &ticket, DistSubmitKind expected_kind, const DistSubmitCtx &ctx
) {
    const uint8_t expected = static_cast<uint8_t>(
        expected_kind == DistSubmitKind::Alloc ? DistCompeteFirstKind::Alloc : DistCompeteFirstKind::Kernel
    );
    const bool sequence_ok = ctx.self != nullptr && ticket.task_id >= 0 && ticket.task_id < kFlagCap &&
                             ctx.self->local_index == ticket.task_id + 1;
    const bool fields_ok = ticket.ready != 0 && ticket.kind == expected && ticket.reserved == 0 && ticket.won <= 1 &&
                           ticket.joint <= 1 && ticket.joint_init <= 1 && ticket.claim_attempted <= 1;
    if (__builtin_expect(sequence_ok && fields_ok, 1)) return true;
    // A callback must be synchronous and may not submit another task before
    // its matching Finish. Treat a stale/malformed ticket as a protocol error
    // instead of reconstructing a context for the wrong per-core payload.
    fdwic_trace_set_fatal(ticket.task_id);
    return false;
}

PTO_DEVICE_FUNC TaskOutputTensors dist_submit_finish_kernel_tail(
    DistSubmitCtx &ctx, const MixedKernels &mixed, const L0TaskArgs &args, uint64_t tail_begin, uint64_t submit_begin
) {
    uint64_t register_begin = tail_begin;
    if (ctx.won) {
        ctx.fanin_count = dist_submit_collect_fanin(args, ctx, ctx.fanin);
        TRACE_TIMESTAMP(fanin_end);
        TRACE_SPAN_RECORD(
            tail_begin, fanin_end, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Fanin, 0,
            static_cast<uint32_t>(ctx.fanin_count)
        );
        register_begin = fanin_end;
    }
    dist_submit_register_outputs(ctx, args, /*include_existing=*/true);
    TRACE_TIMESTAMP(register_end);
    TRACE_SPAN_RECORD(register_begin, register_end, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Register, 0, 1);
    if (__builtin_expect(ctx.won, 0)) {
        dist_submit_build_winner_task(ctx, mixed, args);
        TRACE_TIMESTAMP(winner_build_end);
        TRACE_SPAN_RECORD(
            register_end, winner_build_end, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::WinnerBuild, 0, 0
        );
    } else {
        // Production losers perform real BlockWon progress. This is not the
        // empty loser path used by the standalone single-lane probe.
        drain_block_won(ctx.self);
        TRACE_TIMESTAMP(loser_replay_end);
        TRACE_SPAN_RECORD(
            register_end, loser_replay_end, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::LoserReplay, 0, 0
        );
    }
    TRACE_TIMESTAMP(submit_end);
    fdwic_perf_clock_submit_end(ctx.task_id);
    fdwic_submit_pmu_submit_end(ctx.task_id);
    TRACE_SPAN_RECORD(
        submit_begin, submit_end, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Submit,
        static_cast<uint32_t>(ctx.won), 0
    );
    return ctx.result;
}

PTO_DEVICE_FUNC TaskOutputTensors
dist_submit_finish_alloc_tail(DistSubmitCtx &ctx, uint64_t completion_begin, uint64_t submit_begin) {
    if (__builtin_expect(ctx.won, 0)) {
        dist_submit_complete_alloc(ctx);
        TRACE_TIMESTAMP(alloc_complete_end);
        TRACE_SPAN_RECORD(
            completion_begin, alloc_complete_end, ctx.self, ctx.task_id, -1, TracePhase::AllocComplete, 0, 0
        );
    }
    // Alloc losers have no corresponding replay action. Their final suffix is
    // intentionally left as an offline Submit residual, not a fake phase.
    TRACE_TIMESTAMP(submit_end);
    fdwic_perf_clock_submit_end(ctx.task_id);
    fdwic_submit_pmu_submit_end(ctx.task_id);
    TRACE_SPAN_RECORD(
        submit_begin, submit_end, ctx.self, ctx.task_id, -1, TracePhase::Submit, static_cast<uint32_t>(ctx.won), 1
    );
    return ctx.result;
}

#include "dist_engine/aicore/run_state.h"

PTO_DEVICE_FUNC inline void dist_final_barrier_publish(__gm__ volatile int64_t &value) {
    (void)fdwic_trace_atomic_fetch_add<int64_t>(
        -1, FdwicAtomicSite::ReplayDoneIncrement, value, 1, /*result_used=*/false
    );
}

PTO_DEVICE_FUNC inline bool dist_final_barrier_progress(
    __gm__ DistCore *self, bool &leaf_forwarded, bool &root_released, bool &leaf_released
) {
    const int32_t group = self->block_id % kFinalBarrierGroups;
    __gm__ FinalBarrierArrival &leaf_arrival = g_dist.final_barrier.leaf_arrivals[group];
    const bool leaf_leader = self->lane == LANE_AIC && self->block_id == group;
    if (leaf_leader && !leaf_forwarded &&
        fdwic_trace_atomic_load(-1, FdwicAtomicSite::ReplayDonePoll, leaf_arrival.v) >= leaf_arrival.expected) {
        dist_final_barrier_publish(g_dist.final_barrier.root_arrival.v);
        leaf_forwarded = true;
    }

    const bool root_leader = leaf_leader && group == 0;
    if (root_leader && !root_released &&
        fdwic_trace_atomic_load(-1, FdwicAtomicSite::ReplayDonePoll, g_dist.final_barrier.root_arrival.v) >=
            g_dist.final_barrier.root_arrival.expected) {
        dist_final_barrier_publish(g_dist.final_barrier.root_release.v);
        root_released = true;
    }

    if (leaf_leader && leaf_forwarded && !leaf_released &&
        fdwic_trace_atomic_load(-1, FdwicAtomicSite::ReplayDonePoll, g_dist.final_barrier.root_release.v) >= 1) {
        dist_final_barrier_publish(g_dist.final_barrier.leaf_releases[group].v);
        leaf_released = true;
    }
    return fdwic_trace_atomic_load(
               -1, FdwicAtomicSite::ReplayDonePoll, g_dist.final_barrier.leaf_releases[group].v
           ) >= 1;
}

PTO_DEVICE_FUNC void dist_submit_drain_to_completion(__gm__ DistCore *self) {
    if (self == nullptr) return;
    const int32_t final_group = self->block_id % kFinalBarrierGroups;
    dist_final_barrier_publish(g_dist.final_barrier.leaf_arrivals[final_group].v);
    const uint32_t final_poll_region = fdwic_atomic_poll_region_begin(
        fdwic_atomic_site_mask(FdwicAtomicSite::ReplayDonePoll) |
        fdwic_atomic_site_mask(FdwicAtomicSite::FaninFlagLoad) | fdwic_atomic_block_won_poll_mask()
    );
    bool leaf_forwarded = false;
    bool root_released = false;
    bool leaf_released = false;
    bool global_release_observed = false;
    while (true) {
        drain_block_won(self);
        const int32_t freed = drain_phase_b(self);
        if (!global_release_observed) {
            global_release_observed =
                dist_final_barrier_progress(self, leaf_forwarded, root_released, leaf_released);
        }
        const bool ring_empty = self->occupied_count == 0;
        const bool pending = has_pending_won(self);
        if (global_release_observed && ring_empty && !pending) break;
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
    TRACE_TIMESTAMP(submit_begin);
    fdwic_perf_clock_submit_begin(ctx.task_id);
    fdwic_submit_pmu_submit_begin(ctx.task_id);
    drain_block_won(ctx.self);
    drain_phase_b(ctx.self);
    TRACE_TIMESTAMP(efdrain_end);
    TRACE_SPAN_RECORD(submit_begin, efdrain_end, ctx.self, ctx.task_id, -1, TracePhase::EfDrain, 0, 0);
    uint64_t prepare_map_end = efdrain_end;
    if (!dist_submit_materialize_and_prepare_map(
            ctx.self, args, ctx, DistSubmitKind::Kernel, efdrain_end, prepare_map_end
        )) {
        return ctx.result;
    }
    const uint64_t claim_begin = prepare_map_end;
    const bool is_winner = dist_submit_claim(DistSubmitKind::Kernel, &mixed, ctx);
    const uint32_t claim_flags = (is_winner ? kFdwicClaimWon : 0U) | (ctx.claim_attempted ? kFdwicClaimAttempted : 0U);
    TRACE_TIMESTAMP(claim_end);
    TRACE_SPAN_RECORD(claim_begin, claim_end, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Claim, claim_flags, 0);
    return dist_submit_finish_kernel_tail(ctx, mixed, args, claim_end, submit_begin);
}

DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors dist_alloc_tensors(PTO2Runtime *, const L0TaskArgs &args) {
    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, args, ctx);
    TRACE_TIMESTAMP(submit_begin);
    fdwic_perf_clock_submit_begin(ctx.task_id);
    fdwic_submit_pmu_submit_begin(ctx.task_id);
    drain_block_won(ctx.self);
    drain_phase_b(ctx.self);
    TRACE_TIMESTAMP(efdrain_end);
    TRACE_SPAN_RECORD(submit_begin, efdrain_end, ctx.self, ctx.task_id, -1, TracePhase::EfDrain, 0, 0);
    uint64_t prepare_map_end = efdrain_end;
    if (!dist_submit_materialize_and_prepare_map(
            ctx.self, args, ctx, DistSubmitKind::Alloc, efdrain_end, prepare_map_end
        )) {
        return ctx.result;
    }
    const uint64_t register_begin = prepare_map_end;
    dist_submit_register_outputs(ctx, args, /*include_existing=*/false);
    TRACE_TIMESTAMP(register_end);
    TRACE_SPAN_RECORD(register_begin, register_end, ctx.self, ctx.task_id, -1, TracePhase::Register, 0, 0);
    const uint64_t claim_begin = register_end;
    const bool is_winner = dist_submit_claim(DistSubmitKind::Alloc, nullptr, ctx);
    const uint32_t claim_flags = (is_winner ? kFdwicClaimWon : 0U) | (ctx.claim_attempted ? kFdwicClaimAttempted : 0U);
    TRACE_TIMESTAMP(claim_end);
    TRACE_SPAN_RECORD(claim_begin, claim_end, ctx.self, ctx.task_id, -1, TracePhase::Claim, claim_flags, 1);
    return dist_submit_finish_alloc_tail(ctx, claim_end, submit_begin);
}

DIST_API_ATTR PTO_DEVICE_FUNC DistCompeteFirstTicket
dist_submit_compete_first_begin(PTO2Runtime *, const MixedKernels &mixed) {
    const ActiveMask active = mixed.to_active_mask();
    if (__builtin_popcount(active.core_mask()) >= 2) g_fdwic_joint_submit_seen = true;

    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, ctx);
    TRACE_TIMESTAMP(submit_begin);
    fdwic_perf_clock_submit_begin(ctx.task_id);
    fdwic_submit_pmu_submit_begin(ctx.task_id);
    drain_block_won(ctx.self);
    drain_phase_b(ctx.self);
    TRACE_TIMESTAMP(efdrain_end);
    TRACE_SPAN_RECORD(submit_begin, efdrain_end, ctx.self, ctx.task_id, -1, TracePhase::EfDrain, 0, 0);

    const bool ready = dist_submit_check_task_cap(ctx, DistSubmitKind::Kernel);
    const uint64_t claim_begin = efdrain_end;
    const bool is_winner = ready && dist_submit_claim(DistSubmitKind::Kernel, &mixed, ctx);
    const uint32_t claim_flags = (is_winner ? kFdwicClaimWon : 0U) | (ctx.claim_attempted ? kFdwicClaimAttempted : 0U);
    TRACE_TIMESTAMP(claim_end);
    // 与泳道 Claim.end 使用同一源码边界。局部 PMU 从这里跨过 Begin 返回、
    // 同步 eager callback 构参，直到匹配 Finish 的 Materialize 入口。
    fdwic_submit_pmu_phase_begin<FdwicSubmitPmuPhase::ArgBuild>();
    TRACE_SPAN_RECORD(claim_begin, claim_end, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Claim, claim_flags, 0);
    return dist_submit_make_ticket(ctx, DistSubmitKind::Kernel, submit_begin, ready);
}

DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors dist_submit_compete_first_finish(
    PTO2Runtime *, const MixedKernels &mixed, const DistCompeteFirstTicket &ticket, const L0TaskArgs &args
) {
    DistSubmitCtx ctx;
    dist_submit_restore_from_ticket(ticket, args, ctx);
    if (!dist_submit_validate_ticket(ticket, DistSubmitKind::Kernel, ctx)) return ctx.result;

    fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::ArgBuild>();
    TRACE_TIMESTAMP(materialize_begin);
    uint64_t prepare_map_end = materialize_begin;
    if (!dist_submit_materialize_and_prepare_map(
            ctx.self, args, ctx, DistSubmitKind::Kernel, materialize_begin, prepare_map_end
        )) {
        return ctx.result;
    }
    return dist_submit_finish_kernel_tail(ctx, mixed, args, prepare_map_end, ticket.submit_begin);
}

DIST_API_ATTR PTO_DEVICE_FUNC DistCompeteFirstTicket dist_alloc_compete_first_begin(PTO2Runtime *) {
    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, ctx);
    TRACE_TIMESTAMP(submit_begin);
    fdwic_perf_clock_submit_begin(ctx.task_id);
    fdwic_submit_pmu_submit_begin(ctx.task_id);
    drain_block_won(ctx.self);
    drain_phase_b(ctx.self);
    TRACE_TIMESTAMP(efdrain_end);
    TRACE_SPAN_RECORD(submit_begin, efdrain_end, ctx.self, ctx.task_id, -1, TracePhase::EfDrain, 0, 0);

    const bool ready = dist_submit_check_task_cap(ctx, DistSubmitKind::Alloc);
    const uint64_t claim_begin = efdrain_end;
    const bool is_winner = ready && dist_submit_claim(DistSubmitKind::Alloc, nullptr, ctx);
    const uint32_t claim_flags = (is_winner ? kFdwicClaimWon : 0U) | (ctx.claim_attempted ? kFdwicClaimAttempted : 0U);
    TRACE_TIMESTAMP(claim_end);
    fdwic_submit_pmu_phase_begin<FdwicSubmitPmuPhase::ArgBuild>();
    TRACE_SPAN_RECORD(claim_begin, claim_end, ctx.self, ctx.task_id, -1, TracePhase::Claim, claim_flags, 1);
    return dist_submit_make_ticket(ctx, DistSubmitKind::Alloc, submit_begin, ready);
}

DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors
dist_alloc_compete_first_finish(PTO2Runtime *, const DistCompeteFirstTicket &ticket, const L0TaskArgs &args) {
    DistSubmitCtx ctx;
    dist_submit_restore_from_ticket(ticket, args, ctx);
    if (!dist_submit_validate_ticket(ticket, DistSubmitKind::Alloc, ctx)) return ctx.result;

    fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::ArgBuild>();
    TRACE_TIMESTAMP(materialize_begin);
    uint64_t prepare_map_end = materialize_begin;
    if (!dist_submit_materialize_and_prepare_map(
            ctx.self, args, ctx, DistSubmitKind::Alloc, materialize_begin, prepare_map_end
        )) {
        return ctx.result;
    }
    dist_submit_register_outputs(ctx, args, /*include_existing=*/false);
    TRACE_TIMESTAMP(register_end);
    TRACE_SPAN_RECORD(prepare_map_end, register_end, ctx.self, ctx.task_id, -1, TracePhase::Register, 0, 0);
    return dist_submit_finish_alloc_tail(ctx, register_end, ticket.submit_begin);
}
