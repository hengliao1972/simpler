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

namespace {

PTO_DEVICE_FUNC void publish_task_flag(int32_t task_id) {
    if (task_id < 0 || task_id >= kFlagCap) return;
    __gm__ DistTaskCell &cell = task_cell(task_id);
#if defined(__CCE_AICORE__)
    cell.flag = 1;
    ccec_flush_region(&cell, sizeof(DistTaskCell));
#else
    atom_store(cell.flag, 1, __ATOMIC_RELEASE);
#endif
}

PTO_DEVICE_FUNC bool task_flag_ready(int32_t task_id, int memorder) {
    if (task_id < 0 || task_id >= kFlagCap) return false;
    __gm__ DistTaskCell &cell = task_cell(task_id);
#if defined(__CCE_AICORE__)
    (void)memorder;
    ccec_invalidate_region(&cell, sizeof(DistTaskCell));
    return cell.flag != 0;
#else
    return atom_load(cell.flag, memorder) != 0;
#endif
}

PTO_DEVICE_FUNC void store_task_vend(int32_t task_id, uint64_t vend) {
    if (task_id < 0 || task_id >= kFlagCap) return;
    __gm__ DistTaskCell &cell = task_cell(task_id);
#if defined(__CCE_AICORE__)
    cell.vend = vend;
    ccec_flush_region(&cell, sizeof(DistTaskCell));
#else
    atom_store(cell.vend, vend, __ATOMIC_RELAXED);
#endif
}

PTO_DEVICE_FUNC void advance_frontier() {
    int64_t f = atom_load(g_dist.frontier, __ATOMIC_ACQUIRE);
    while (true) {
        const int64_t next = f + 1;
        if (next >= kFlagCap) break;
        if (!task_flag_ready(static_cast<int32_t>(next), __ATOMIC_ACQUIRE)) break;
        if (atom_cas_weak(g_dist.frontier, f, next, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            f = next;
        }
    }
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
        while (now_ns() < target) { /* spin: emulate kernel busy time */
        }
#if DIST_TRACE_ENABLED
        if (g_trace_on) {
            trace_state(self).trace.push_back(
                TraceEvent{
                    s.task_id, s.func_id, self->lane, static_cast<uint8_t>(s.is_multicore ? 1 : 0), TracePhase::Kernel,
                    t0 - g_trace_epoch_ns, static_cast<uint64_t>(sim_ns), static_cast<uint64_t>(sim_ns)
                }
            );
        }
#endif
    } else if (s.function_bin_addr != 0 && !g_skip_exec) {
        KernelFn fn = reinterpret_cast<KernelFn>(s.function_bin_addr);
#if DIST_TRACE_ENABLED
        if (g_trace_on) {
            const uint64_t t0 = now_ns();
            fn(reinterpret_cast<__gm__ int64_t *>(s.args));
            const uint64_t t1 = now_ns();
            trace_state(self).trace.push_back(
                TraceEvent{
                    s.task_id, s.func_id, self->lane, static_cast<uint8_t>(s.is_multicore ? 1 : 0), TracePhase::Kernel,
                    t0 - g_trace_epoch_ns, t1 - t0, t1 - t0
                }
            );
        } else {
            fn(reinterpret_cast<__gm__ int64_t *>(s.args));
        }
#else
        fn(reinterpret_cast<__gm__ int64_t *>(s.args));
#endif
    }
#else
    if (s.function_bin_addr != 0) {
        KernelFn fn = reinterpret_cast<KernelFn>(s.function_bin_addr);
        fn(reinterpret_cast<__gm__ int64_t *>(s.args));
    }
#endif
    if (s.is_multicore) {
        __gm__ WonSlot &w = g_dist.blocks[s.won_block].slots[s.won_slot];
        if (atom_fetch_sub<int64_t>(w.remaining, 1, __ATOMIC_ACQ_REL) == 1) {
            publish_task_flag(s.task_id);
            atom_store(w.state, 0, __ATOMIC_RELEASE);
            advance_frontier();
        }
    } else {
        publish_task_flag(s.task_id);
        advance_frontier();
    }
    s.built = false;
    s.occupied = false;
}

PTO_DEVICE_FUNC int32_t drain_phase_b(__gm__ DistCore *self) {
    if (self->occupied_count == 0) return 0;
    int32_t freed = 0;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ RingSlot &s = self->slots[i];
        if (!s.occupied || !s.built) continue;
        bool ready = true;
        for (int32_t f = 0; f < s.fanin_count; f++) {
            if (!task_flag_ready(s.fanin[f], __ATOMIC_ACQUIRE)) {
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

PTO_DEVICE_FUNC void drain_block_won(__gm__ DistCore *self) {
    if (self->lane == LANE_AIC || self->lane == LANE_NONE) return;
    __gm__ BlockWon &bw = g_dist.blocks[self->block_id];
    if (atom_load(bw.any_pub, __ATOMIC_ACQUIRE) == 0) return;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ WonSlot &w = bw.slots[i];
        if (atom_load(w.state, __ATOMIC_ACQUIRE) != 1) continue;
        if (!w.lane[self->lane].present) continue;
        int32_t exp = 0;
        if (!atom_cas_strong(w.drained[self->lane].v, exp, 1, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) continue;
        int32_t si = alloc_ring_slot(self);
        if (si < 0) {
            atom_store(w.drained[self->lane].v, 0, __ATOMIC_RELEASE);
            return;
        }
        __gm__ const BuiltSubtask &b = w.lane[self->lane];
#if DIST_TRACE_ENABLED
        const uint64_t t_won0 = trace_now();
        const uint64_t t_won0_cpu = trace_now_cpu();
#endif
        build_ring_slot(
            self->slots[si], w.task_id, b.func_id, b.function_bin_addr, b.tensors, b.tensor_count, b.scalars,
            b.scalar_count, b.fanin, b.fanin_count, b.sub_block_id, /*is_multicore=*/true, self->block_id, i
        );
        self->occupied_count++;
        self->owned_total++;
#if DIST_TRACE_ENABLED
        if (g_trace_on) {
            for (int32_t k = 0; k < b.fanin_count; k++)
                trace_state(self).dep_edges.push_back({w.task_id, b.fanin[k]});
        }
        trace_overhead_impl(self, w.task_id, b.func_id, TracePhase::DrainWon, t_won0, t_won0_cpu);
#endif
    }
}

PTO_DEVICE_FUNC void dist_submit_execute_first(__gm__ DistCore *self) {
    TRACE_LAP_RESET(self);
    if (!fatal_set()) {
        drain_block_won(self);
        drain_phase_b(self);
    }
    TRACE_LAP(self, self->local_index, -1, TracePhase::EfDrain);
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
};

PTO_DEVICE_FUNC void dist_submit_begin(__gm__ DistCore *self, const L0TaskArgs &args, DistSubmitCtx &ctx) {
    ctx.self = self != nullptr ? self : g_self;
    if (ctx.self == nullptr) {
        ctx.task_id = kFlagCap;
        ctx.payload = nullptr;
    } else {
        ctx.task_id = ctx.self->local_index++;
        ctx.payload = &ctx.self->task_payloads[ctx.task_id & kTaskPayloadMask];
    }
    ctx.result.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(ctx.task_id)));
    ctx.tensor_count = args.tensor_count();
    ctx.output_bytes = 0;
    ctx.fanin_count = 0;
    ctx.kernel_id = INVALID_KERNEL_ID;
    ctx.won = false;
    ctx.joint = false;
    ctx.joint_init = false;
    ctx.joint_block = -1;
    ctx.joint_slot = -1;
    ctx.joint_count = 0;
}

PTO_DEVICE_FUNC bool dist_submit_check_task_cap(const DistSubmitCtx &ctx, DistSubmitKind kind) {
    if (ctx.task_id < kFlagCap) return true;
    set_fatal();
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

PTO_DEVICE_FUNC void calculate_output_layout(const L0TaskArgs &args, DistOutputLayout &layout) {
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        layout.offsets[i] = layout.total_output_size;
        layout.buffer_sizes[i] = TensorCreateInfo::buffer_size_bytes(args.tensor(i).create_info());
        layout.total_output_size += PTO2_ALIGN_UP(layout.buffer_sizes[i], PTO2_PACKED_OUTPUT_ALIGN);
    }
}

PTO_DEVICE_FUNC bool dist_submit_materialize_args(const L0TaskArgs &args, DistSubmitCtx &ctx, DistSubmitKind kind) {
    if (ctx.payload == nullptr) return false;
    ctx.payload->tensor_count = args.tensor_count();
    ctx.payload->scalar_count = args.scalar_count();
    ctx.tensor_count = args.tensor_count();

    const size_t ring = g_dist.heap_size;
    DistOutputLayout layout;
    calculate_output_layout(args, layout);
    const uint64_t total = layout.total_output_size;
    uint64_t task_base = PTO2_ALIGN_UP(ctx.self->heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    if (total > 0 && g_dist.heap_base != nullptr) {
        if (total > ring) {
            set_fatal();
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

    for (int32_t i = 0; i < ctx.tensor_count; i++) {
        const TensorArgType tag = args.tag(i);
        ctx.payload->tags[i] = tag;
        if (tag != TensorArgType::OUTPUT) {
#if defined(__CCE_AICORE__)
            if (args.tensor(i).tensor_from_gm()) {
                Tensor::copy(ctx.payload->tensors[i], args.tensor(i).gm_ref());
            } else {
                Tensor::copy(ctx.payload->tensors[i], args.tensor(i).ref());
            }
#else
            Tensor::copy(ctx.payload->tensors[i], args.tensor(i).ref());
#endif
            continue;
        }
        const auto &ci = args.tensor(i).create_info();
        if (g_dist.heap_base == nullptr) {
            set_fatal();
            if (kind == DistSubmitKind::Alloc) {
                DIST_ERRF("[dist_engine] GM output heap not allocated at alloc %d\n", ctx.task_id);
            } else {
                DIST_ERRF("[dist_engine] GM output heap not allocated at task %d\n", ctx.task_id);
            }
            return false;
        }
        const uint64_t phys = (task_base + layout.offsets[i]) % ring;
        __gm__ Tensor &slot_t = ctx.payload->tensors[i];
        init_tensor_from_create_info(slot_t, ci, g_dist.heap_base + phys, layout.buffer_sizes[i]);
        slot_t.owner_task_id.raw = ctx.result.task_id().raw;
        ctx.result.materialize_output(slot_t);
    }
    for (int32_t i = 0; i < args.scalar_count(); i++) {
        ctx.payload->scalars[i] = args.scalar(i);
    }
#if defined(__CCE_AICORE__)
    ccec_flush_region(ctx.payload, sizeof(DistTaskPayload));
#endif
    ctx.self->heap_next = task_base + layout.total_output_size;
    ctx.output_bytes = total;
#if !defined(__CCE_AICORE__)
    store_task_vend(ctx.task_id, ctx.self->heap_next);
#endif
    return true;
}

PTO_DEVICE_FUNC inline TensorArgType payload_tag(const DistSubmitCtx &ctx, int32_t i) { return ctx.payload->tags[i]; }

PTO_DEVICE_FUNC inline const auto &payload_tensor(const DistSubmitCtx &ctx, int32_t i) {
    return ctx.payload->tensors[i];
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

PTO_DEVICE_FUNC int32_t dist_submit_collect_fanin(const DistSubmitCtx &ctx, int32_t fanin[]) {
    int32_t fc = 0;
    for (int32_t i = 0; i < ctx.tensor_count; i++) {
        const TensorArgType tag = payload_tag(ctx, i);
        if (tag == TensorArgType::OUTPUT) continue;
        const uint64_t owner_raw = payload_tensor(ctx, i).owner_task_id.raw;
        if (owner_raw != UINT64_MAX) dist_submit_add_fanin(fanin, fc, static_cast<int32_t>(owner_raw & 0xFFFFFFFFu));
        if (tag != TensorArgType::INPUT && tag != TensorArgType::INOUT) continue;
#if defined(__CCE_AICORE__)
        const int32_t p = dist_tensor_map_lookup(ctx.self->map, payload_tensor(ctx, i));
#else
        const Tensor &t = payload_tensor(ctx, i);
        if (t.manual_dep) continue;
        const int32_t p = dist_tensor_map_lookup(ctx.self->map, t);
#endif
        dist_submit_add_fanin(fanin, fc, p);
    }
    return fc;
}

PTO_DEVICE_FUNC void dist_submit_register_outputs(DistSubmitCtx &ctx, bool include_existing) {
    for (int32_t i = 0; i < ctx.tensor_count; i++) {
        const TensorArgType tag = payload_tag(ctx, i);
        const bool registers_producer =
            tag == TensorArgType::OUTPUT ||
            (include_existing && (tag == TensorArgType::INOUT || tag == TensorArgType::OUTPUT_EXISTING));
        if (registers_producer) dist_tensor_map_insert(ctx.self->map, payload_tensor(ctx, i), ctx.task_id);
    }
}

PTO_DEVICE_FUNC bool dist_submit_materialize_and_prepare_map(
    __gm__ DistCore *self, const L0TaskArgs &args, DistSubmitCtx &ctx, DistSubmitKind kind
) {
    if (!dist_submit_check_task_cap(ctx, kind)) return false;
    if (!dist_submit_materialize_args(args, ctx, kind)) return false;
#if !defined(__CCE_AICORE__)
    if (fatal_set()) return false;
#endif
    dist_submit_prepare_map(self, ctx.task_id);
    return true;
}

}  // namespace
