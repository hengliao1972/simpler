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

PTO_DEVICE_FUNC inline void dist_aicore_call_slot_kernel(__gm__ RingSlot &slot) {
#if defined(__CCE_AICORE__)
    const bool is_aic = g_ccec_core_type == static_cast<int32_t>(CoreType::AIC);
    if (is_aic) {
        if (pto_call_linked_kernel_aic != nullptr) {
            pto_call_linked_kernel_aic(slot.func_id, reinterpret_cast<__gm__ int64_t *>(slot.args));
        }
    } else if (pto_call_linked_kernel_aiv != nullptr) {
        pto_call_linked_kernel_aiv(slot.func_id, reinterpret_cast<__gm__ int64_t *>(slot.args));
    }
#else
    typedef void (*KernelFn)(__gm__ int64_t *);
    if (slot.function_bin_addr != 0) {
        KernelFn fn = reinterpret_cast<KernelFn>(slot.function_bin_addr);
        fn(reinterpret_cast<__gm__ int64_t *>(slot.args));
    }
#endif
}

namespace {

PTO_DEVICE_FUNC void publish_task_flag(int32_t task_id) {
    if (task_id < 0 || task_id >= kFlagCap) return;
    __gm__ DistTaskCell &cell = task_cell(task_id);
    (void)fdwic_trace_atomic_exchange(
        task_id, FdwicAtomicSite::CompletionFlagExchange, cell.flag, int64_t{1}, /*result_used=*/false, __ATOMIC_RELEASE
    );
}

PTO_DEVICE_FUNC bool task_flag_ready(int32_t task_id, int memorder, FdwicAtomicSite site) {
    if (task_id < 0 || task_id >= kFlagCap) return false;
    __gm__ DistTaskCell &cell = task_cell(task_id);
    return fdwic_trace_atomic_load(task_id, site, cell.flag, /*result_used=*/true, memorder) != 0;
}

PTO_DEVICE_FUNC void store_task_vend(int32_t task_id, uint64_t vend) {
    if (task_id < 0 || task_id >= kFlagCap) return;
    __gm__ DistTaskCell &cell = task_cell(task_id);
    (void)fdwic_trace_atomic_exchange(
        task_id, FdwicAtomicSite::CompletionVendExchange, cell.vend, vend, /*result_used=*/false, __ATOMIC_RELAXED
    );
}

PTO_DEVICE_FUNC void store_won_remaining(__gm__ WonSlot &w, int32_t count, int32_t task_id) {
    (void)fdwic_trace_atomic_exchange(
        task_id, FdwicAtomicSite::WonRemainingExchange, w.remaining.v, static_cast<int64_t>(count),
        /*result_used=*/false, __ATOMIC_RELAXED
    );
}

PTO_DEVICE_FUNC void reset_won_lane(__gm__ WonSlot &w, int32_t lane, int32_t task_id) {
    (void)fdwic_trace_atomic_exchange(
        task_id, FdwicAtomicSite::WonLaneResetExchange, w.drained[lane].v, kDrainedClaimed,
        /*result_used=*/false
    );
    w.lane[lane].present = false;
}

PTO_DEVICE_FUNC bool claim_won_lane(__gm__ WonSlot &w, int32_t lane) {
    return fdwic_trace_atomic_exchange(
               -1, FdwicAtomicSite::WonLaneClaimExchange, w.drained[lane].v, kDrainedClaimed,
               /*result_used=*/true
           ) == kDrainedFree;
}

PTO_DEVICE_FUNC void publish_won_slot(__gm__ WonSlot &w, int32_t task_id) {
    (void)fdwic_trace_atomic_exchange(
        task_id, FdwicAtomicSite::WonStatePublishExchange, w.state.v, kWonStatePublished,
        /*result_used=*/false
    );
}

PTO_DEVICE_FUNC bool decrement_won_remaining_is_last(__gm__ WonSlot &w, int32_t task_id) {
    return fdwic_trace_atomic_fetch_sub<int64_t>(
               task_id, FdwicAtomicSite::WonRemainingFetchSub, w.remaining.v, 1, /*result_used=*/true
           ) == 1;
}

PTO_DEVICE_FUNC void clear_won_slot_state(__gm__ WonSlot &w, int32_t task_id) {
    (void)fdwic_trace_atomic_exchange(
        task_id, FdwicAtomicSite::WonStateClearExchange, w.state.v, kWonStateFree, /*result_used=*/false
    );
}

PTO_DEVICE_FUNC int64_t load_frontier_for_advance() {
    // The first load reads global scan state, not a particular task cell.
    return fdwic_trace_atomic_load(-1, FdwicAtomicSite::FrontierInitialLoad, g_dist.frontier);
}

PTO_DEVICE_FUNC bool try_advance_frontier_to(int64_t &frontier, int64_t next) {
    // Pair this update with the immediately preceding next-task flag load.
    const int64_t old = fdwic_trace_atomic_fetch_max<int64_t>(
        static_cast<int32_t>(next), FdwicAtomicSite::FrontierMax, g_dist.frontier, next, /*result_used=*/true
    );
    frontier = old > next ? old : next;
    return next > old;
}

PTO_DEVICE_FUNC void advance_frontier() {
#if defined(__CCE_AICORE__)
    if (g_dist_ptr == nullptr) return;
#endif
    int64_t f = load_frontier_for_advance();
    while (true) {
        const int64_t next = f + 1;
        if (next >= kFlagCap) break;
        if (!task_flag_ready(static_cast<int32_t>(next), __ATOMIC_ACQUIRE, FdwicAtomicSite::FrontierFlagLoad)) break;
        try_advance_frontier_to(f, next);
    }
}

PTO_DEVICE_FUNC void complete_executed_task(__gm__ DistCore *self, int32_t task_id) {
    if (self != nullptr) {
        store_task_vend(task_id, self->heap_next);
    }
    store_barrier();
    publish_task_flag(task_id);
    advance_frontier();
}

PTO_DEVICE_FUNC void execute_slot([[maybe_unused]] __gm__ DistCore *self, __gm__ RingSlot &s) {
    typedef void (*KernelFn)(__gm__ int64_t *);
#if DIST_SIM_HOST_CLOCK
    const Runtime *rt = g_dist.runtime;
    const int32_t sim_ns =
        (rt != nullptr && rt->use_example_exec_time_ && s.func_id >= 0 && s.func_id < RUNTIME_MAX_FUNC_ID) ?
            rt->example_exec_time_ns_[s.func_id] :
            0;
    if (sim_ns > 0) {
        const uint64_t t0 = now_ns();
        const uint64_t target = t0 + static_cast<uint64_t>(sim_ns);
        TRACE_SPAN_BEGIN(kernel_trace);
        while (now_ns() < target) { /* spin: emulate kernel busy time */
        }
        TRACE_SPAN_END(
            kernel_trace, self, s.task_id, s.func_id, TracePhase::Kernel, static_cast<uint32_t>(s.is_multicore ? 1 : 0),
            0
        );
    } else if (s.function_bin_addr != 0 && !g_skip_exec) {
        KernelFn fn = reinterpret_cast<KernelFn>(s.function_bin_addr);
        TRACE_SPAN_BEGIN(kernel_trace);
        fn(reinterpret_cast<__gm__ int64_t *>(s.args));
        TRACE_SPAN_END(
            kernel_trace, self, s.task_id, s.func_id, TracePhase::Kernel, static_cast<uint32_t>(s.is_multicore ? 1 : 0),
            0
        );
    }
#else
#if PTO_FDWIC_PERF_CLOCK_KERNEL
    const uint64_t perf_clock_kernel_begin = fdwic_perf_clock_kernel_begin();
#endif
    TRACE_SPAN_BEGIN(kernel_trace);
    const uint32_t submit_pmu_kernel_token = fdwic_submit_pmu_linked_kernel_pause();
    dist_aicore_call_slot_kernel(s);
    fdwic_submit_pmu_linked_kernel_resume(submit_pmu_kernel_token);
    TRACE_SPAN_END(
        kernel_trace, self, s.task_id, s.func_id, TracePhase::Kernel, static_cast<uint32_t>(s.is_multicore ? 1 : 0), 0
    );
#if PTO_FDWIC_PERF_CLOCK_KERNEL
    fdwic_perf_clock_kernel_end(perf_clock_kernel_begin);
#endif
#endif
    store_barrier();
    if (s.is_multicore) {
        __gm__ WonSlot &w = g_dist.blocks[s.won_block].slots[s.won_slot];
        if (decrement_won_remaining_is_last(w, s.task_id)) {
            clear_won_slot_state(w, s.task_id);
            complete_executed_task(self, s.task_id);
        }
    } else {
        complete_executed_task(self, s.task_id);
    }
    TRACE_INSTANT(self, s.task_id, s.func_id, TracePhase::Commit, static_cast<uint32_t>(s.is_multicore ? 1 : 0));
    s.built = false;
    s.occupied = false;
}

PTO_DEVICE_FUNC int32_t drain_phase_b(__gm__ DistCore *self) {
    if (self == nullptr || self->occupied_count == 0) return 0;
    int32_t freed = 0;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ RingSlot &s = self->slots[i];
        if (!s.occupied || !s.built) continue;
        bool ready = true;
        for (int32_t f = 0; f < s.fanin_count; f++) {
            if (!task_flag_ready(s.fanin[f], __ATOMIC_ACQUIRE, FdwicAtomicSite::FaninFlagLoad)) {
                ready = false;
                break;
            }
        }
        if (!ready) continue;
        execute_slot(self, s);
        self->occupied_count--;
        freed++;
    }
    return freed;
}

PTO_DEVICE_FUNC int32_t alloc_ring_slot(__gm__ DistCore *self) {
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        if (!self->slots[i].occupied) return i;
    }
    return -1;
}

PTO_DEVICE_FUNC inline int32_t kernel_id_for_lane(const MixedKernels &mixed, int32_t lane) {
    switch (lane) {
    case LANE_AIC:
        return mixed.aic_kernel_id;
    case LANE_AIV0:
        return mixed.aiv0_kernel_id;
    case LANE_AIV1:
        return mixed.aiv1_kernel_id;
    default:
        return INVALID_KERNEL_ID;
    }
}

PTO_DEVICE_FUNC inline bool lane_active(const ActiveMask &M, int32_t lane) {
    return M.subtask_active(static_cast<PTO2SubtaskSlot>(lane));
}

template <typename TensorArrPtr, typename ScalarArrPtr, typename FaninArrPtr>
PTO_DEVICE_FUNC void build_ring_slot(
    __gm__ RingSlot &s, int32_t task_id, int32_t func_id, uint64_t fn_addr, TensorArrPtr tensors, int32_t tc,
    ScalarArrPtr scalars, int32_t sc, FaninArrPtr fanin, int32_t fc, int32_t sub_block_id, bool is_multicore,
    int32_t won_block, int32_t won_slot
) {
    s.occupied = true;
    s.task_id = task_id;
    s.func_id = func_id;
    s.function_bin_addr = fn_addr;
    s.built = true;
    s.tensor_count = tc;
    s.scalar_count = sc;
    for (int32_t i = 0; i < tc; i++)
        Tensor::copy(s.tensors[i], tensors[i]);
    for (int32_t j = 0; j < sc; j++)
        s.scalars[j] = scalars[j];
    int32_t n = 0;
    for (int32_t i = 0; i < tc; i++)
        s.args[n++] = reinterpret_cast<uint64_t>(&s.tensors[i]);
    for (int32_t j = 0; j < sc; j++)
        s.args[n++] = s.scalars[j];
    s.local_ctx.s_block_idx = 0;
    s.local_ctx.s_block_num = 1;
    s.local_ctx.async_ctx.completion_count = nullptr;
    s.local_ctx.async_ctx.completion_error_code = nullptr;
    s.local_ctx.async_ctx.completion_entries = nullptr;
    s.local_ctx.async_ctx.completion_capacity = 0;
    s.local_ctx.async_ctx.task_token.raw = UINT64_MAX;
    s.global_ctx.sub_block_id = sub_block_id;
    s.args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&s.local_ctx);
    s.args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&s.global_ctx);
    s.fanin_count = fc;
    for (int32_t k = 0; k < fc; k++)
        s.fanin[k] = fanin[k];
    s.is_multicore = is_multicore;
    s.won_block = won_block;
    s.won_slot = won_slot;
}

PTO_DEVICE_FUNC bool drain_block_won(__gm__ DistCore *self) {
    // Workers replay the same stream; polling is enabled before the initial drain of their first joint submit.
    if (!g_fdwic_joint_submit_seen) return false;
    if (self == nullptr || self->lane == LANE_AIC || self->lane == LANE_NONE) return false;
    __gm__ BlockWon &bw = g_dist.blocks[self->block_id];
    if (fdwic_trace_atomic_load(-1, FdwicAtomicSite::WonAnyLoad, bw.any_pub) == 0) return false;
    bool drained = false;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ WonSlot &w = bw.slots[i];
        if (fdwic_trace_atomic_load(-1, FdwicAtomicSite::WonStateLoad, w.state.v) != kWonStatePublished) continue;
#if defined(__CCE_AICORE__)
        dist_aicore_invalidate_region(&w.lane[self->lane].present, sizeof(w.lane[self->lane].present));
#endif
        if (!w.lane[self->lane].present) continue;
        if (!claim_won_lane(w, self->lane)) continue;
        int32_t si = alloc_ring_slot(self);
        if (si < 0) {
            (void)fdwic_trace_atomic_exchange(
                -1, FdwicAtomicSite::WonLaneReleaseExchange, w.drained[self->lane].v, kDrainedFree,
                /*result_used=*/false
            );
            return drained;
        }
#if defined(__CCE_AICORE__)
        dist_aicore_invalidate_region(&w.meta, sizeof(w.meta));
        dist_aicore_invalidate_region(&w.lane[self->lane], sizeof(w.lane[self->lane]));
#endif
        const int32_t task_id = w.meta.task_id;
        __gm__ const BuiltSubtask &b = w.lane[self->lane];
        TRACE_SPAN_BEGIN(drain_won_trace);
        build_ring_slot(
            self->slots[si], task_id, b.func_id, b.function_bin_addr, b.tensors, b.tensor_count, b.scalars,
            b.scalar_count, b.fanin, b.fanin_count, b.sub_block_id, /*is_multicore=*/true, self->block_id, i
        );
        TRACE_SPAN_END(
            drain_won_trace, self, task_id, b.func_id, TracePhase::DrainWon, /*flags=*/1u, static_cast<uint32_t>(i)
        );
        self->occupied_count++;
        self->owned_total++;
        drained = true;
    }
    return drained;
}

PTO_DEVICE_FUNC bool has_pending_won(__gm__ DistCore *self) {
    if (!g_fdwic_joint_submit_seen) return false;
    if (self == nullptr || self->lane == LANE_AIC || self->lane == LANE_NONE) return false;
    __gm__ BlockWon &bw = g_dist.blocks[self->block_id];
    if (fdwic_trace_atomic_load(-1, FdwicAtomicSite::WonAnyLoad, bw.any_pub) == 0) return false;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ WonSlot &w = bw.slots[i];
        if (fdwic_trace_atomic_load(-1, FdwicAtomicSite::WonStateLoad, w.state.v) != kWonStatePublished) continue;
#if defined(__CCE_AICORE__)
        dist_aicore_invalidate_region(&w.lane[self->lane].present, sizeof(w.lane[self->lane].present));
#endif
        if (!w.lane[self->lane].present) continue;
        if (fdwic_trace_atomic_load(-1, FdwicAtomicSite::WonDrainedLoad, w.drained[self->lane].v) == kDrainedFree)
            return true;
    }
    return false;
}

enum class DistSubmitKind : int32_t {
    Kernel = 0,
    Alloc = 1,
};

struct DistSubmitCtx {
    __gm__ DistCore *self;
    __gm__ DistTaskPayload *payload;
    int32_t task_id;
    int32_t tensor_count;
    int32_t scalar_count;
    uint32_t register_mask;
    uint64_t output_bytes;
    TaskOutputTensors result;
    int32_t fanin[kMaxFanin];
    int32_t fanin_count;
    int32_t kernel_id;
    bool won;
    bool joint;
    bool joint_init;
    int32_t joint_block;
    int32_t joint_slot;
    int32_t joint_count;
    bool claim_attempted;
};

PTO_DEVICE_FUNC void dist_submit_begin(__gm__ DistCore *self, DistSubmitCtx &ctx) {
    ctx.self = self != nullptr ? self : g_self;
    if (ctx.self == nullptr) {
        ctx.task_id = kFlagCap;
        ctx.payload = nullptr;
    } else {
        ctx.task_id = ctx.self->local_index++;
        ctx.payload = &ctx.self->task_payloads[ctx.task_id & kTaskPayloadMask];
    }
    ctx.result.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(ctx.task_id)));
    // The compete-first Begin deliberately has no L0TaskArgs.  Finish fills
    // these two counts after the synchronous caller-side argument callback.
    ctx.tensor_count = 0;
    ctx.scalar_count = 0;
    ctx.register_mask = 0;
    ctx.output_bytes = 0;
    ctx.fanin_count = 0;
    ctx.kernel_id = INVALID_KERNEL_ID;
    ctx.won = false;
    ctx.joint = false;
    ctx.joint_init = false;
    ctx.joint_block = -1;
    ctx.joint_slot = -1;
    ctx.joint_count = 0;
    ctx.claim_attempted = false;
}

PTO_DEVICE_FUNC void dist_submit_begin(__gm__ DistCore *self, const L0TaskArgs &args, DistSubmitCtx &ctx) {
    dist_submit_begin(self, ctx);
    ctx.tensor_count = args.tensor_count();
    ctx.scalar_count = args.scalar_count();
}

PTO_DEVICE_FUNC bool dist_submit_check_task_cap(const DistSubmitCtx &ctx, DistSubmitKind kind) {
    if (ctx.task_id < kFlagCap) return true;
    fdwic_trace_set_fatal(ctx.task_id);
    if (kind == DistSubmitKind::Alloc) {
        DIST_ERRF("[dist_engine] alloc task id %d exceeds kFlagCap %d\n", ctx.task_id, kFlagCap);
    } else {
        DIST_ERRF(
            "[dist_engine] task id %d exceeds kFlagCap %d (enlarge or window the flag/vend rings)\n", ctx.task_id,
            kFlagCap
        );
    }
    return false;
}

PTO_DEVICE_FUNC uint32_t
calculate_output_layout(const L0TaskArgs &args, DistOutputLayout &layout, uint32_t &register_mask) {
    layout.total_output_size = 0;
    register_mask = 0;
    uint32_t output_mask = 0;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        const TensorArgType tag = args.tag(i);
        if (tag == TensorArgType::INOUT || tag == TensorArgType::OUTPUT_EXISTING) register_mask |= 1u << i;
        if (tag != TensorArgType::OUTPUT) continue;
        output_mask |= 1u << i;
        layout.buffer_sizes[i] = TensorCreateInfo::buffer_size_bytes(args.tensor(i).create_info());
        layout.total_output_size += PTO2_ALIGN_UP(layout.buffer_sizes[i], PTO2_PACKED_OUTPUT_ALIGN);
    }
    return output_mask;
}

PTO_DEVICE_FUNC bool dist_submit_materialize_args(const L0TaskArgs &args, DistSubmitCtx &ctx, DistSubmitKind kind) {
    if (ctx.payload == nullptr) return false;
    ctx.tensor_count = args.tensor_count();
    ctx.scalar_count = args.scalar_count();

    const size_t ring = g_dist.heap_size;
    DistOutputLayout layout;
    uint32_t output_mask = calculate_output_layout(args, layout, ctx.register_mask);
    const uint64_t total = layout.total_output_size;
    uint64_t task_base = PTO2_ALIGN_UP(ctx.self->heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    if (total > 0 && g_dist.heap_base != nullptr) {
        if (total > ring) {
            fdwic_trace_set_fatal(ctx.task_id);
            if (kind == DistSubmitKind::Alloc) {
                DIST_ERRF(
                    "[dist_engine] alloc task %d outputs %llu B exceed heap ring %zu B\n", ctx.task_id,
                    (unsigned long long)total, ring
                );
            } else {
                DIST_ERRF(
                    "[dist_engine] task %d outputs %llu B exceed heap ring %zu B (enlarge PTO_DIST_HEAP_MB)\n",
                    ctx.task_id, (unsigned long long)total, ring
                );
            }
            return false;
        }
        if ((task_base % ring) + total > ring) {
            task_base = ((task_base / ring) + 1) * ring;
        }
    }

    uint64_t output_offset = 0;
    for (int32_t i = 0; output_mask != 0; i++, output_mask >>= 1) {
        if ((output_mask & 1u) == 0) continue;
        const auto &ci = args.tensor(i).create_info();
        if (g_dist.heap_base == nullptr) {
            fdwic_trace_set_fatal(ctx.task_id);
            if (kind == DistSubmitKind::Alloc) {
                DIST_ERRF("[dist_engine] GM output heap not allocated at alloc %d\n", ctx.task_id);
            } else {
                DIST_ERRF("[dist_engine] GM output heap not allocated at task %d\n", ctx.task_id);
            }
            return false;
        }
        const uint64_t buffer_size = layout.buffer_sizes[i];
        const uint64_t phys = (task_base + output_offset) % ring;
        __gm__ Tensor &slot_t = ctx.payload->tensors[i];
        init_tensor_from_create_info(slot_t, ci, g_dist.heap_base + phys, buffer_size);
        slot_t.owner_task_id.raw = ctx.result.task_id().raw;
        ctx.result.materialize_output(slot_t);
        output_offset += PTO2_ALIGN_UP(buffer_size, PTO2_PACKED_OUTPUT_ALIGN);
    }
    ctx.self->heap_next = task_base + layout.total_output_size;
    ctx.output_bytes = total;
    return true;
}

PTO_DEVICE_FUNC inline void
dist_submit_copy_arg_tensor(__gm__ Tensor &dst, const L0TaskArgs &args, const DistSubmitCtx &ctx, int32_t i) {
    if (args.tag(i) == TensorArgType::OUTPUT) {
        Tensor::copy(dst, ctx.payload->tensors[i]);
        return;
    }
#if defined(__CCE_AICORE__)
    if (args.tensor(i).tensor_from_gm()) {
        Tensor::copy(dst, args.tensor(i).gm_ref());
    } else {
        Tensor::copy(dst, args.tensor(i).ref());
    }
#else
    Tensor::copy(dst, args.tensor(i).ref());
#endif
}

template <typename FaninArrPtr>
PTO_DEVICE_FUNC void build_ring_slot_from_submit(
    __gm__ RingSlot &s, int32_t task_id, int32_t func_id, uint64_t fn_addr, const L0TaskArgs &args,
    const DistSubmitCtx &ctx, FaninArrPtr fanin, int32_t fc, int32_t sub_block_id, bool is_multicore, int32_t won_block,
    int32_t won_slot
) {
    s.occupied = true;
    s.task_id = task_id;
    s.func_id = func_id;
    s.function_bin_addr = fn_addr;
    s.built = true;
    s.tensor_count = ctx.tensor_count;
    s.scalar_count = ctx.scalar_count;
    for (int32_t i = 0; i < ctx.tensor_count; i++)
        dist_submit_copy_arg_tensor(s.tensors[i], args, ctx, i);
    for (int32_t j = 0; j < ctx.scalar_count; j++)
        s.scalars[j] = args.scalar(j);
    int32_t n = 0;
    for (int32_t i = 0; i < ctx.tensor_count; i++)
        s.args[n++] = reinterpret_cast<uint64_t>(&s.tensors[i]);
    for (int32_t j = 0; j < ctx.scalar_count; j++)
        s.args[n++] = s.scalars[j];
    s.local_ctx.s_block_idx = 0;
    s.local_ctx.s_block_num = 1;
    s.local_ctx.async_ctx.completion_count = nullptr;
    s.local_ctx.async_ctx.completion_error_code = nullptr;
    s.local_ctx.async_ctx.completion_entries = nullptr;
    s.local_ctx.async_ctx.completion_capacity = 0;
    s.local_ctx.async_ctx.task_token.raw = UINT64_MAX;
    s.global_ctx.sub_block_id = sub_block_id;
    s.args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&s.local_ctx);
    s.args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&s.global_ctx);
    s.fanin_count = fc;
    for (int32_t k = 0; k < fc; k++)
        s.fanin[k] = fanin[k];
    s.is_multicore = is_multicore;
    s.won_block = won_block;
    s.won_slot = won_slot;
}

PTO_DEVICE_FUNC void dist_submit_prepare_map(__gm__ DistCore *self, int32_t task_id) {
    if (self == nullptr) return;
    dist_tensor_map_advance_retire(self->map, task_id, g_dist.H);
}

PTO_DEVICE_FUNC void dist_submit_add_fanin(int32_t fanin[], int32_t &fanin_count, int32_t producer) {
    if (producer < 0) return;
    for (int32_t k = 0; k < fanin_count; k++)
        if (fanin[k] == producer) return;
    if (fanin_count < kMaxFanin) fanin[fanin_count++] = producer;
}

PTO_DEVICE_FUNC int32_t dist_submit_collect_fanin(const L0TaskArgs &args, const DistSubmitCtx &ctx, int32_t fanin[]) {
    int32_t fc = 0;
    for (int32_t i = 0; i < ctx.tensor_count; i++) {
        const TensorArgType tag = args.tag(i);
        if (tag == TensorArgType::OUTPUT) continue;
#if defined(__CCE_AICORE__)
        if (args.tensor(i).tensor_from_gm()) {
            const uint64_t owner_raw = args.tensor(i).gm_ref().owner_task_id.raw;
            if (owner_raw != UINT64_MAX) {
                dist_submit_add_fanin(fanin, fc, static_cast<int32_t>(owner_raw & 0xFFFFFFFFu));
            }
            if (tag != TensorArgType::INPUT && tag != TensorArgType::INOUT) continue;
            const int32_t p = dist_tensor_map_lookup(ctx.self->map, args.tensor(i).gm_ref());
            dist_submit_add_fanin(fanin, fc, p);
        } else {
            const Tensor &t = args.tensor(i).ref();
            const uint64_t owner_raw = t.owner_task_id.raw;
            if (owner_raw != UINT64_MAX) {
                dist_submit_add_fanin(fanin, fc, static_cast<int32_t>(owner_raw & 0xFFFFFFFFu));
            }
            if (tag != TensorArgType::INPUT && tag != TensorArgType::INOUT) continue;
            const int32_t p = dist_tensor_map_lookup(ctx.self->map, t);
            dist_submit_add_fanin(fanin, fc, p);
        }
#else
        const Tensor &t = args.tensor(i).ref();
        const uint64_t owner_raw = t.owner_task_id.raw;
        if (owner_raw != UINT64_MAX) dist_submit_add_fanin(fanin, fc, static_cast<int32_t>(owner_raw & 0xFFFFFFFFu));
        if (tag != TensorArgType::INPUT && tag != TensorArgType::INOUT) continue;
        if (t.manual_dep) continue;
        const int32_t p = dist_tensor_map_lookup(ctx.self->map, t);
        dist_submit_add_fanin(fanin, fc, p);
#endif
    }
    return fc;
}

PTO_DEVICE_FUNC void dist_submit_insert_existing_tensor(DistSubmitCtx &ctx, const L0TaskArgs &args, int32_t i) {
#if defined(__CCE_AICORE__)
    if (args.tensor(i).tensor_from_gm()) {
        dist_tensor_map_insert(ctx.self->map, args.tensor(i).gm_ref(), ctx.task_id);
    } else {
        dist_tensor_map_insert(ctx.self->map, args.tensor(i).ref(), ctx.task_id);
    }
#else
    dist_tensor_map_insert(ctx.self->map, args.tensor(i).ref(), ctx.task_id);
#endif
}

PTO_DEVICE_FUNC void dist_submit_register_outputs(DistSubmitCtx &ctx, const L0TaskArgs &args, bool include_existing) {
    if (!include_existing) return;
    uint32_t register_mask = ctx.register_mask;
    for (int32_t i = 0; register_mask != 0; i++, register_mask >>= 1) {
        if ((register_mask & 1u) != 0) dist_submit_insert_existing_tensor(ctx, args, i);
    }
}

PTO_DEVICE_FUNC bool dist_submit_materialize_and_prepare_map(
    __gm__ DistCore *self, const L0TaskArgs &args, DistSubmitCtx &ctx, DistSubmitKind kind, uint64_t materialize_begin,
    uint64_t &prepare_map_end
) {
    if (!dist_submit_check_task_cap(ctx, kind)) return false;
    if (!dist_submit_materialize_args(args, ctx, kind)) return false;
    // end 与泳道 materialize_end 取时前的业务边界一致。失败路径不伪造
    // end，最终由 phase shape/平衡门禁拒绝 raw。
    fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::Materialize>();
    TRACE_TIMESTAMP(materialize_end);
    TRACE_SPAN_RECORD(
        materialize_begin, materialize_end, self, ctx.task_id, -1, TracePhase::Materialize, 0,
        static_cast<uint32_t>(kind)
    );
#if !defined(__CCE_AICORE__)
    if (fdwic_trace_is_fatal(ctx.task_id)) return false;
#endif
    // 与泳道 PrepareMap 的业务主体共用同一调用边界；PMU 变体只累计
    // dist_submit_prepare_map()，不把前一阶段的 trace 发布算入该阶段。
    fdwic_submit_pmu_phase_begin<FdwicSubmitPmuPhase::PrepareMap>();
    dist_submit_prepare_map(self, ctx.task_id);
    fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::PrepareMap>();
    TRACE_TIMESTAMP(prepare_map_finish);
    // compete-first Kernel winner 的泳道 Fanin 从 PrepareMap 结束边界开始；
    // legacy 路径此时尚未 Claim，ctx.won 为 false，不会提前打开。
    if (kind == DistSubmitKind::Kernel && ctx.won) {
        fdwic_submit_pmu_phase_begin<FdwicSubmitPmuPhase::Fanin>();
    }
    TRACE_SPAN_RECORD(
        materialize_end, prepare_map_finish, self, ctx.task_id, -1, TracePhase::PrepareMap, 0,
        static_cast<uint32_t>(kind)
    );
    prepare_map_end = prepare_map_finish;
    return true;
}

}  // namespace
