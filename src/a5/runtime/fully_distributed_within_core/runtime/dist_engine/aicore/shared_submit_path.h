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

#if PTO_FDWIC_SHARED_MAP

// Phase-1 shared PA is intentionally a separate Submit protocol. Keeping it in
// this file prevents the private region-map path from accumulating shared
// branches and makes every unsupported generic/shared shape fail closed.

PTO_DEVICE_FUNC bool dist_shared_pa_fail(DistSubmitCtx &ctx, int32_t error_code) {
    if (ctx.self != nullptr) ctx.self->local_index = kFlagCap;
    set_fatal_code(error_code);
    return false;
}

PTO_DEVICE_FUNC bool dist_shared_pa_fail(__gm__ DistCore *self, int32_t error_code) {
    if (self != nullptr) self->local_index = kFlagCap;
    set_fatal_code(error_code);
    return false;
}

struct DistSharedPaBeginState {
    __gm__ DistCore *self;
    int32_t task_id;
    int32_t kernel_id;
    bool won;
    bool claim_attempted;
};

struct DistSharedPaHandoffTrace {
    uint64_t begin;
    uint64_t end;
    int64_t observed;
    bool captured;
};

constexpr uint32_t kFdwicSharedEfDrainSkipBudgetByte = 0;
constexpr uint32_t kFdwicSharedEfDrainNoProgressByte = 1;
constexpr uint8_t kFdwicSharedEfDrainNoProgressSkipSubmits = 1;
constexpr uint8_t kFdwicSharedEfDrainLongWaitSkipSubmits = 2;
constexpr uint8_t kFdwicSharedEfDrainLongWaitPollThreshold = 24;

// Submit-entry EfDrain is opportunistic: explicit ring backpressure and
// FinalDrain still call drain_phase_b() directly and remain responsible for
// progress. A single long-waiting slot therefore need not reload the same
// not-ready fanin on every adjacent Submit. Two occupied slots always poll,
// because the next winner may immediately enter WaitForSlot.
PTO_DEVICE_FUNC int32_t dist_shared_pa_opportunistic_drain(__gm__ DistCore *self) {
    if (self == nullptr) return 0;
    __gm__ uint8_t &skip_budget = self->slots_pad[kFdwicSharedEfDrainSkipBudgetByte];
    __gm__ uint8_t &no_progress_polls = self->slots_pad[kFdwicSharedEfDrainNoProgressByte];
    if (self->occupied_count == 0) {
        skip_budget = 0;
        no_progress_polls = 0;
        return 0;
    }
    const bool single_slot = self->occupied_count == 1;
    if (single_slot && skip_budget != 0) {
        --skip_budget;
        return 0;
    }
    skip_budget = 0;
    if (!single_slot) no_progress_polls = 0;
    const int32_t freed = drain_phase_b(self);
    if (single_slot && freed == 0 && self->occupied_count == 1) {
        if (no_progress_polls < kFdwicSharedEfDrainLongWaitPollThreshold) {
            ++no_progress_polls;
        }
        skip_budget = no_progress_polls == kFdwicSharedEfDrainLongWaitPollThreshold ?
                          kFdwicSharedEfDrainLongWaitSkipSubmits :
                          kFdwicSharedEfDrainNoProgressSkipSubmits;
    } else {
        no_progress_polls = 0;
    }
    return freed;
}

PTO_DEVICE_FUNC int32_t dist_shared_pa_expected_kernel_id(const MixedKernels &mixed, DistSharedPaTaskKind kind) {
    const int32_t kernel_id = kind == DistSharedPaTaskKind::Qk || kind == DistSharedPaTaskKind::Pv ?
                                  mixed.aic_kernel_id :
                                  mixed.aiv0_kernel_id;
    return kernel_id >= 0 && kernel_id < RUNTIME_MAX_FUNC_ID && kernel_id <= INT16_MAX ? kernel_id : INVALID_KERNEL_ID;
}

PTO_DEVICE_FUNC inline __attribute__((always_inline)) bool dist_shared_pa_claim_tournament(
    uint32_t candidate_rank, uint32_t tournament_groups, int32_t kernel_id, DistSharedPaBeginState &state
) {
    state.claim_attempted = true;
    __gm__ SharedClaimTournamentTask &tournament =
        g_dist.shared_pa.claim_tournament[static_cast<uint32_t>(state.task_id)];
    const uint32_t group = candidate_rank % tournament_groups;
    const int64_t expected = -1;
    const int64_t desired = static_cast<int64_t>(state.task_id);
    const int64_t local_observed = fdwic_trace_atomic_compare_exchange<int64_t>(
        state.task_id, FdwicAtomicSite::SharedClaimTournamentLocal, tournament.local[group].owner.v, expected, desired,
        /*result_used=*/true
    );
    if (local_observed != expected) {
        if (local_observed != desired) {
            return dist_shared_pa_fail(state.self, PTO2_ERROR_TENSORMAP_PROTOCOL);
        }
        return false;
    }
    const int64_t root_observed = fdwic_trace_atomic_compare_exchange<int64_t>(
        state.task_id, FdwicAtomicSite::SharedClaimTournamentRoot, tournament.root.owner.v, expected, desired,
        /*result_used=*/true
    );
    if (root_observed != expected && root_observed != desired) {
        return dist_shared_pa_fail(state.self, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    state.won = root_observed == expected;
    if (state.won) state.kernel_id = kernel_id;
    return state.won;
}

PTO_DEVICE_FUNC bool dist_shared_pa_claim(
    CoreType replay_role, int32_t replay_block_id, DistSharedPaTaskKind kind, const MixedKernels *mixed,
    DistSharedPaBeginState &state
) {
    (void)replay_block_id;
    state.kernel_id = INVALID_KERNEL_ID;
    state.won = false;
    state.claim_attempted = false;
    if (state.self == nullptr || state.task_id < 0 ||
        static_cast<uint32_t>(state.task_id) >= kFdwicSharedPaTaskCapacity) {
        return false;
    }
    int32_t kernel_id = INVALID_KERNEL_ID;
    const int32_t core_idx = state.self->core_idx;
    uint32_t candidate_rank = 0;
    uint32_t tournament_groups = 0;
    if (kind == DistSharedPaTaskKind::Alloc) {
        // Alloc has no executable lane. All 96 workers remain legal takeover
        // candidates; the tournament reduces same-address contention without
        // removing idle workers from the ownership population.
        if (core_idx < 0 || static_cast<uint32_t>(core_idx) >= kFdwicSharedWorkers ||
            (core_idx < static_cast<int32_t>(kFdwicSharedAicWorkers) ? replay_role != CoreType::AIC :
                                                                       replay_role != CoreType::AIV)) {
            return false;
        }
        candidate_rank = static_cast<uint32_t>(core_idx);
        tournament_groups = kFdwicSharedAllocClaimTournamentGroups;
    } else {
        if (mixed == nullptr) return false;
        kernel_id = dist_shared_pa_expected_kernel_id(*mixed, kind);
        if (kernel_id == INVALID_KERNEL_ID) return false;
        if (kind == DistSharedPaTaskKind::Qk || kind == DistSharedPaTaskKind::Pv) {
            if (replay_role != CoreType::AIC || core_idx < 0 ||
                static_cast<uint32_t>(core_idx) >= kFdwicSharedAicWorkers) {
                return false;
            }
            candidate_rank = static_cast<uint32_t>(core_idx);
            tournament_groups = kFdwicSharedAicClaimTournamentGroups;
        } else if (kind == DistSharedPaTaskKind::Sf || kind == DistSharedPaTaskKind::Up) {
            if (replay_role != CoreType::AIV || core_idx < static_cast<int32_t>(kFdwicSharedAicWorkers) ||
                static_cast<uint32_t>(core_idx) >= kFdwicSharedWorkers) {
                return false;
            }
            candidate_rank = static_cast<uint32_t>(core_idx) - kFdwicSharedAicWorkers;
            tournament_groups = kFdwicSharedAivClaimTournamentGroups;
        } else {
            return false;
        }
    }
    return dist_shared_pa_claim_tournament(candidate_rank, tournament_groups, kernel_id, state);
}

#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
template <CoreType CompiledReplayRole, DistSharedPaTaskKind CompiledKind>
PTO_DEVICE_FUNC inline __attribute__((always_inline)) bool dist_shared_pa_claim_fixed(
    int32_t kernel_id, DistSharedPaBeginState &state
) {
    static_assert(
        CompiledReplayRole == CoreType::AIC || CompiledReplayRole == CoreType::AIV,
        "shared PA unity replay role must be AIC or AIV"
    );
    static_assert(CompiledKind != DistSharedPaTaskKind::Count, "shared PA unity task kind must be concrete");

    state.kernel_id = INVALID_KERNEL_ID;
    state.won = false;
    state.claim_attempted = false;
    if (state.self == nullptr || state.task_id < 0 ||
        static_cast<uint32_t>(state.task_id) >= kFdwicSharedPaTaskCapacity) {
        return false;
    }

    const int32_t core_idx = state.self->core_idx;
    uint32_t candidate_rank = 0;
    uint32_t tournament_groups = 0;
    if constexpr (CompiledKind == DistSharedPaTaskKind::Alloc) {
        // The role-specific orchestration image already carries the replay
        // role as a template argument. Keep the attach-owned core-index check,
        // but do not rebuild a runtime MixedKernels shape on all 96 actors.
        if constexpr (CompiledReplayRole == CoreType::AIC) {
            if (core_idx < 0 || static_cast<uint32_t>(core_idx) >= kFdwicSharedAicWorkers) return false;
        } else {
            if (core_idx < static_cast<int32_t>(kFdwicSharedAicWorkers) ||
                static_cast<uint32_t>(core_idx) >= kFdwicSharedWorkers) {
                return false;
            }
        }
        candidate_rank = static_cast<uint32_t>(core_idx);
        tournament_groups = kFdwicSharedAllocClaimTournamentGroups;
    } else if constexpr (CompiledKind == DistSharedPaTaskKind::Qk ||
                         CompiledKind == DistSharedPaTaskKind::Pv) {
        if constexpr (CompiledReplayRole != CoreType::AIC) {
            return false;
        } else {
            if (core_idx < 0 || static_cast<uint32_t>(core_idx) >= kFdwicSharedAicWorkers) return false;
            candidate_rank = static_cast<uint32_t>(core_idx);
            tournament_groups = kFdwicSharedAicClaimTournamentGroups;
        }
    } else {
        static_assert(
            CompiledKind == DistSharedPaTaskKind::Sf || CompiledKind == DistSharedPaTaskKind::Up,
            "unsupported shared PA unity task kind"
        );
        if constexpr (CompiledReplayRole != CoreType::AIV) {
            return false;
        } else {
            if (core_idx < static_cast<int32_t>(kFdwicSharedAicWorkers) ||
                static_cast<uint32_t>(core_idx) >= kFdwicSharedWorkers) {
                return false;
            }
            candidate_rank = static_cast<uint32_t>(core_idx) - kFdwicSharedAicWorkers;
            tournament_groups = kFdwicSharedAivClaimTournamentGroups;
        }
    }
    return dist_shared_pa_claim_tournament(candidate_rank, tournament_groups, kernel_id, state);
}
#endif

PTO_DEVICE_FUNC void dist_shared_pa_restore_winner_ticket(const DistCompeteFirstTicket &ticket, DistSubmitCtx &ctx) {
    ctx.self = g_self;
    ctx.task_id = ticket.task_id;
    ctx.payload =
        ctx.self != nullptr && ctx.task_id >= 0 && static_cast<uint32_t>(ctx.task_id) < kFdwicSharedPaTaskCapacity ?
            &ctx.self->task_payloads[ctx.task_id & kTaskPayloadMask] :
            nullptr;
    ctx.result.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(ctx.task_id)));
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
}

PTO_DEVICE_FUNC bool dist_shared_pa_kind_matches_task(int32_t task_id, DistSharedPaTaskKind kind) {
    return task_id >= 0 && static_cast<uint32_t>(task_id) < kFdwicSharedPaTaskCapacity &&
           static_cast<uint32_t>(task_id) % kFdwicSharedPaTasksPerBatch == static_cast<uint32_t>(kind) &&
           kind != DistSharedPaTaskKind::Count;
}

PTO_DEVICE_FUNC bool dist_shared_pa_kernel_shape(const MixedKernels &mixed, DistSharedPaTaskKind kind) {
    const ActiveMask active = mixed.to_active_mask();
    if (__builtin_popcount(active.core_mask()) != 1) return false;
    if (kind == DistSharedPaTaskKind::Qk || kind == DistSharedPaTaskKind::Pv) {
        return lane_active(active, LANE_AIC) && dist_shared_pa_expected_kernel_id(mixed, kind) != INVALID_KERNEL_ID;
    }
    if (kind == DistSharedPaTaskKind::Sf || kind == DistSharedPaTaskKind::Up) {
        return lane_active(active, LANE_AIV0) && dist_shared_pa_expected_kernel_id(mixed, kind) != INVALID_KERNEL_ID;
    }
    return false;
}

PTO_DEVICE_FUNC bool
dist_shared_pa_ref_is(const L0TaskArgs &args, int32_t index, TensorArgType tag, int32_t producer, int16_t slot) {
    if (index < 0 || index >= args.tensor_count() || args.tag(index) != tag ||
        !args.tensor(index).tensor_from_shared_output()) {
        return false;
    }
    const FdwicOutputRef ref = args.tensor(index).shared_output_ref();
    return dist_shared_pa_output_ref_valid(ref) && ref.producer_task_id == producer && ref.output_slot == slot;
}

// CCEC 15.0.5 vector codegen must not merge the LM TensorRef::ref() and GM
// TensorRef::gm_ref() paths into the caller: at -O2/-O3 that inlining creates
// an invalid pointer address-space cast. The cube backend does not need this
// boundary, and the check remains outside the serialized handoff.
PTO_DEVICE_FUNC
#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)
__attribute__((noinline))
#endif
bool
dist_shared_pa_ordinary_manual_dep(const L0TaskArgs &args, int32_t index) {
    if (index < 0 || index >= args.tensor_count() || !args.tensor(index).has_existing_tensor()) {
        return false;
    }
#if defined(__CCE_AICORE__)
    if (args.tensor(index).tensor_from_gm()) {
        return args.tensor(index).gm_ref().manual_dep;
    }
    return args.tensor(index).ref().manual_dep;
#else
    return args.tensor(index).ref().manual_dep;
#endif
}

// Phase-1 PA has five fixed argument schemas and at most three created
// outputs, always in one contiguous argument range. Decode that schema once
// in ArgBuild, then carry only the range required by Materialize. This is
// replay-local state, not a cross-image ABI.
struct DistSharedPaMaterializePlan {
    uint64_t output_bytes[3];
    uint64_t total_output_bytes;
    uint32_t output_start;
    uint32_t output_count;
    uint32_t register_mask;
};

PTO_DEVICE_FUNC void dist_shared_pa_reset_materialize_plan(DistSharedPaMaterializePlan &plan) {
    plan.total_output_bytes = 0;
    plan.output_start = 0;
    plan.output_count = 0;
    plan.register_mask = 0;
}

PTO_DEVICE_FUNC bool dist_shared_pa_validate_and_plan(
    const L0TaskArgs &args, int32_t task_id, DistSharedPaTaskKind kind, DistSharedPaMaterializePlan &plan
) {
    dist_shared_pa_reset_materialize_plan(plan);
    if (args.tensor_count() < 0 || args.tensor_count() > MAX_TENSOR_ARGS || args.scalar_count() < 0 ||
        args.scalar_count() > MAX_SCALAR_ARGS || args.has_error || args.explicit_dep_count() != 0 ||
        !dist_shared_pa_kind_matches_task(task_id, kind)) {
        return false;
    }
    const int32_t batch_start = task_id - (task_id % static_cast<int32_t>(kFdwicSharedPaTasksPerBatch));

    switch (kind) {
    case DistSharedPaTaskKind::Alloc: {
        const bool valid = args.tensor_count() == 3 && args.scalar_count() == 0 &&
                           args.tag(0) == TensorArgType::OUTPUT && args.tag(1) == TensorArgType::OUTPUT &&
                           args.tag(2) == TensorArgType::OUTPUT && args.tensor(0).has_create_info() &&
                           args.tensor(1).has_create_info() && args.tensor(2).has_create_info();
        if (!valid) return false;
        plan.output_start = 0;
        plan.output_count = 3;
        return true;
    }
    case DistSharedPaTaskKind::Qk: {
        const bool valid = args.tensor_count() == 4 && args.scalar_count() == 2 &&
                           args.tag(0) == TensorArgType::INPUT && args.tag(1) == TensorArgType::INPUT &&
                           args.tag(2) == TensorArgType::INPUT && args.tag(3) == TensorArgType::OUTPUT &&
                           args.tensor(0).has_existing_tensor() && args.tensor(1).has_existing_tensor() &&
                           args.tensor(2).has_existing_tensor() && args.tensor(3).has_create_info();
        if (!valid) return false;
        plan.output_start = 3;
        plan.output_count = 1;
        return true;
    }
    case DistSharedPaTaskKind::Sf: {
        const bool valid = args.tensor_count() == 4 && args.scalar_count() == 3 &&
                           dist_shared_pa_ref_is(args, 0, TensorArgType::INPUT, batch_start + 1, 0) &&
                           args.tag(1) == TensorArgType::OUTPUT && args.tag(2) == TensorArgType::OUTPUT &&
                           args.tag(3) == TensorArgType::OUTPUT && args.tensor(1).has_create_info() &&
                           args.tensor(2).has_create_info() && args.tensor(3).has_create_info();
        if (!valid) return false;
        plan.output_start = 1;
        plan.output_count = 3;
        return true;
    }
    case DistSharedPaTaskKind::Pv: {
        const bool valid = args.tensor_count() == 4 && args.scalar_count() == 2 &&
                           dist_shared_pa_ref_is(args, 0, TensorArgType::INPUT, batch_start + 2, 0) &&
                           args.tag(1) == TensorArgType::INPUT && args.tag(2) == TensorArgType::INPUT &&
                           args.tensor(1).has_existing_tensor() && args.tensor(2).has_existing_tensor() &&
                           args.tag(3) == TensorArgType::OUTPUT && args.tensor(3).has_create_info();
        if (!valid) return false;
        plan.output_start = 3;
        plan.output_count = 1;
        return true;
    }
    case DistSharedPaTaskKind::Up: {
        const bool valid = args.tensor_count() == 7 && args.scalar_count() == 2 &&
                           dist_shared_pa_ref_is(args, 0, TensorArgType::INPUT, batch_start + 2, 1) &&
                           dist_shared_pa_ref_is(args, 1, TensorArgType::INPUT, batch_start + 2, 2) &&
                           dist_shared_pa_ref_is(args, 2, TensorArgType::INPUT, batch_start + 3, 0) &&
                           dist_shared_pa_ref_is(args, 3, TensorArgType::INOUT, batch_start, 2) &&
                           dist_shared_pa_ref_is(args, 4, TensorArgType::INOUT, batch_start, 1) &&
                           dist_shared_pa_ref_is(args, 5, TensorArgType::INOUT, batch_start, 0) &&
                           args.tag(6) == TensorArgType::INOUT && args.tensor(6).has_existing_tensor() &&
                           dist_shared_pa_ordinary_manual_dep(args, 6);
        if (!valid) return false;
        plan.register_mask = (1U << 3) | (1U << 4) | (1U << 5);
        return true;
    }
    case DistSharedPaTaskKind::Count:
        return false;
    }
    return false;
}

PTO_DEVICE_FUNC bool
dist_shared_pa_materialize_args(const L0TaskArgs &args, DistSubmitCtx &ctx, DistSharedPaMaterializePlan &plan) {
    // Complete every deterministic descriptor/layout check before the
    // no-rollback heap cursors or payload are touched.
    if (args.tensor_count() < 0 || args.tensor_count() > MAX_TENSOR_ARGS || ctx.self == nullptr ||
        ctx.payload == nullptr || ctx.task_id < 0 || static_cast<uint32_t>(ctx.task_id) >= kFdwicSharedPaTaskCapacity ||
        ctx.result.size() != 0 || plan.output_count > 3 ||
        plan.output_start > static_cast<uint32_t>(args.tensor_count()) ||
        plan.output_count > static_cast<uint32_t>(args.tensor_count()) - plan.output_start) {
        return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    plan.total_output_bytes = 0;
    for (uint32_t output = 0; output < plan.output_count; ++output) {
        const uint32_t arg_index = plan.output_start + output;
        uint64_t bytes = 0;
        if (arg_index >= static_cast<uint32_t>(args.tensor_count()) ||
            args.tag(static_cast<int32_t>(arg_index)) != TensorArgType::OUTPUT ||
            !args.tensor(static_cast<int32_t>(arg_index)).has_create_info() ||
            !dist_shared_pa_create_info_bytes(args.tensor(static_cast<int32_t>(arg_index)).create_info(), bytes) ||
            bytes > UINT64_MAX - (PTO2_PACKED_OUTPUT_ALIGN - 1U)) {
            return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_PROTOCOL);
        }
        const uint64_t aligned = PTO2_ALIGN_UP(bytes, PTO2_PACKED_OUTPUT_ALIGN);
        if (aligned < bytes || plan.total_output_bytes > UINT64_MAX - aligned) {
            return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_PROTOCOL);
        }
        plan.output_bytes[output] = bytes;
        plan.total_output_bytes += aligned;
    }
    if (plan.total_output_bytes != 0 && g_dist.heap_base == nullptr) {
        return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }

    DistSharedPaHeapReservation reservation;
    if (!dist_shared_pa_reserve_heap(
            g_dist.shared_pa, ctx.task_id, plan.total_output_bytes, static_cast<uint64_t>(g_dist.heap_size), reservation
        )) {
        return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_CAPACITY);
    }

    uint64_t output_offset = 0;
    for (uint32_t output = 0; output < plan.output_count; ++output) {
        const uint32_t arg_index = plan.output_start + output;
        const TensorCreateInfo &create_info = args.tensor(static_cast<int32_t>(arg_index)).create_info();
        __gm__ Tensor &slot = ctx.payload->tensors[arg_index];
        init_tensor_from_create_info(
            slot, create_info, g_dist.heap_base + reservation.task_base + output_offset, plan.output_bytes[output]
        );
        slot.owner_task_id.raw = ctx.result.task_id().raw;
        ctx.result.materialize_output(slot);
        output_offset += PTO2_ALIGN_UP(plan.output_bytes[output], PTO2_PACKED_OUTPUT_ALIGN);
    }
    ctx.tensor_count = args.tensor_count();
    ctx.scalar_count = args.scalar_count();
    ctx.register_mask = plan.register_mask;
    ctx.self->heap_next = reservation.aggregate_vend;
    ctx.output_bytes = plan.total_output_bytes;
    return true;
}

PTO_DEVICE_FUNC bool dist_shared_pa_prepare_writer_history(
    const L0TaskArgs &args, DistSubmitCtx &ctx, DistSharedPaTaskKind kind, DistSharedPaMetadataTrace *trace = nullptr
) {
    if (kind != DistSharedPaTaskKind::Up) return ctx.register_mask == 0;
    if (ctx.register_mask != ((1U << 3) | (1U << 4) | (1U << 5))) return false;
    uint32_t symbol_keys[3] = {};
    for (uint32_t index = 0; index < 3; ++index) {
        if (!dist_shared_pa_output_key(
                args.tensor(static_cast<int32_t>(index + 3)).shared_output_ref(), symbol_keys[index]
            )) {
            return false;
        }
    }
    const int32_t batch_start = ctx.task_id - (ctx.task_id % static_cast<int32_t>(kFdwicSharedPaTasksPerBatch));
    return dist_shared_pa_prepare_up_history(g_dist.shared_pa, ctx.task_id, batch_start, symbol_keys, trace);
}

// The predecessor publishes only after StoreBarrier has sealed its metadata;
// later payload consumers perform their own descriptor/history invalidation.
// A bypass load can therefore observe the monotonic {-1, task_id} completion
// word directly, without queuing an identity RMW in front of the publisher.
PTO_DEVICE_FUNC inline __attribute__((always_inline)) int64_t
dist_shared_pa_load_insert_turn_bypass(__gm__ volatile int64_t &value) {
#if defined(__CCE_AICORE__)
    __gm__ int64_t *signed_address = const_cast<__gm__ int64_t *>(&value);
    __gm__ uint64_t *address = reinterpret_cast<__gm__ uint64_t *>(signed_address);
    return static_cast<int64_t>(static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0)));
#else
    return atomic_load(value, __ATOMIC_RELAXED);
#endif
}

PTO_DEVICE_FUNC bool
dist_shared_pa_wait_insert_turn(DistSubmitCtx &ctx, int64_t &ready_observed, uint32_t &bypass_load_count) {
    ready_observed = -1;
    bypass_load_count = 0;
    if (ctx.task_id == 0) return true;
    __gm__ volatile int64_t &predecessor = g_dist.tasks[static_cast<uint32_t>(ctx.task_id - 1)].deps_prepared;
    uint32_t polls = 0;
    while (true) {
        const int64_t observed = dist_shared_pa_load_insert_turn_bypass(predecessor);
#if DIST_TRACE_ENABLED
        if (bypass_load_count < kFdwicCompactTraceAuxMask) ++bypass_load_count;
#endif
        if (observed == ctx.task_id - 1) {
            ready_observed = observed;
            return true;
        }
        if (observed != -1) return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_PROTOCOL);
        // The predecessor handoff depends only on metadata publication, never
        // on task execution. Running drain_phase_b here would put arbitrary
        // kernel work on the global insert chain and delay every later winner.
        SPIN_WAIT_HINT();
        ++polls;
        if ((polls & 1023U) == 0 &&
            fdwic_trace_atomic_load(
                ctx.task_id, FdwicAtomicSite::SharedMetadataFatalGuardLoad, g_dist.fatal, /*result_used=*/true
            ) != 0) {
            if (ctx.self != nullptr) ctx.self->local_index = kFlagCap;
            return false;
        }
    }
}

PTO_DEVICE_FUNC bool dist_shared_pa_publish_metadata_and_handoff(
    DistSubmitCtx &ctx, DistSharedPaTaskKind kind, DistSharedPaMetadataTrace *metadata_trace, int64_t &ready_observed,
    uint32_t &bypass_load_count, uint64_t &metadata_begin, uint64_t &metadata_end,
    DistSharedPaHandoffTrace &handoff_trace
) {
    handoff_trace.begin = 0;
    handoff_trace.end = 0;
    handoff_trace.observed = INT64_MIN;
    handoff_trace.captured = false;
    if (!dist_shared_pa_wait_insert_turn(ctx, ready_observed, bypass_load_count)) return false;
#if DIST_TRACE_ENABLED
    if (fdwic_swimlane_enabled()) {
        metadata_begin = bypass_load_count != 0 ? fdwic_scalar_result_ready_tick(ready_observed)
                                                : fdwic_swimlane_detail_now();
    } else {
        metadata_begin = 0;
    }
#else
    metadata_begin = 0;
#endif
    const int32_t batch_start = ctx.task_id - (ctx.task_id % static_cast<int32_t>(kFdwicSharedPaTasksPerBatch));
    if (kind == DistSharedPaTaskKind::Up &&
        !dist_shared_pa_commit_up_group_writer(g_dist.shared_pa, ctx.task_id, batch_start, metadata_trace)) {
        return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
#if DIST_TRACE_ENABLED
    metadata_end = fdwic_swimlane_enabled() ? fdwic_swimlane_detail_now() : 0;
#else
    metadata_end = 0;
#endif
    store_barrier();
    __gm__ volatile int64_t &completion = g_dist.tasks[static_cast<uint32_t>(ctx.task_id)].deps_prepared;
#if DIST_TRACE_ENABLED
    if (fdwic_atomic_swimlane_enabled()) {
        handoff_trace.begin = fdwic_swimlane_detail_now();
        handoff_trace.observed = atomic_compare_exchange(completion, int64_t{-1}, static_cast<int64_t>(ctx.task_id));
        handoff_trace.end = fdwic_atomic_result_ready_tick(handoff_trace.observed);
        handoff_trace.captured = true;
    } else {
        handoff_trace.observed = atomic_compare_exchange(completion, int64_t{-1}, static_cast<int64_t>(ctx.task_id));
    }
#else
    handoff_trace.observed = atomic_compare_exchange(completion, int64_t{-1}, static_cast<int64_t>(ctx.task_id));
#endif
    if (handoff_trace.observed != -1) {
        return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    return true;
}

PTO_DEVICE_FUNC bool dist_shared_pa_close_loser(__gm__ DistCore *self, const DistCompeteFirstTicket &ticket) {
    fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::ArgBuild>();
    // Phase-1 shared PA accepts only single-lane tasks, so no BlockWon
    // deposit can exist for a loser to progress here.
    TRACE_TIMESTAMP(submit_end);
    fdwic_perf_clock_submit_end(ticket.task_id);
    fdwic_submit_pmu_submit_end(ticket.task_id);
    TRACE_SPAN_RECORD(
        ticket.submit_begin, submit_end, self, ticket.task_id, ticket.kernel_id, TracePhase::Submit, 0, 0
    );
    return true;
}

#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
template <CoreType CompiledReplayRole, DistSharedPaTaskKind CompiledKind>
PTO_DEVICE_FUNC inline __attribute__((always_inline)) DistCompeteFirstTicket dist_shared_pa_begin_ticket(
#else
PTO_DEVICE_FUNC DistCompeteFirstTicket dist_shared_pa_begin_ticket(
#endif
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
    DistSharedPaReplayContext replay, int32_t expected_task_id, int32_t kernel_id
#else
    DistSharedPaReplayContext replay, DistSharedPaTaskKind kind, const MixedKernels *mixed
#endif
) {
    DistSharedPaBeginState state{};
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
    constexpr DistSharedPaTaskKind kind = CompiledKind;
    const bool replay_ready = replay.ready() && replay.role() == CompiledReplayRole && g_self != nullptr;
#else
    const bool replay_ready = replay.ready() && g_self != nullptr;
#endif
    if (!replay_ready) {
        // Reject a default/stale runtime-owned token before opening Submit
        // tracing or PMU phases.
        (void)dist_shared_pa_fail(g_self, PTO2_ERROR_DIST_CONFIG_INVALID);
        DistCompeteFirstTicket invalid{};
        invalid.task_id = kFlagCap;
        invalid.kernel_id = static_cast<int16_t>(INVALID_KERNEL_ID);
        invalid.won = 0;
        invalid.ready = 0;
        return invalid;
    }
    state.self = g_self;
    state.task_id = state.self->local_index++;
    state.kernel_id = INVALID_KERNEL_ID;
    TRACE_TIMESTAMP(submit_begin);
    fdwic_perf_clock_submit_begin(state.task_id);
    fdwic_submit_pmu_submit_begin(state.task_id);
    fdwic_submit_pmu_phase_begin<FdwicSubmitPmuPhase::EfDrainControl>();
    // Generic shared submits are rejected and every supported PA task is
    // single-lane. EfDrain therefore only has local ring work to progress.
    dist_shared_pa_opportunistic_drain(state.self);
    fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::EfDrainControl>();
    TRACE_TIMESTAMP(efdrain_end);
    // Shared schema-v5 reconstructs EfDrain exactly from the fixed endpoint
    // pair Submit.begin -> Claim.begin. Do not spend a compact generic row on
    // the same interval; the host deliberately rejects such duplicate rows.

#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
    // Phase-1 orchestration computes the authoritative five-task batch
    // sequence once and passes the exact expected id. The single winner still
    // repeats the full kind/MixedKernels validation in Finish before publishing
    // shared state; the 95 losers no longer pay task_id%5, three kernel-field
    // loads, active-mask construction, and popcount on every Submit.
    bool ready = expected_task_id >= 0 &&
                 static_cast<uint32_t>(expected_task_id) < kFdwicSharedPaTaskCapacity &&
                 state.task_id == expected_task_id;
    if constexpr (CompiledKind == DistSharedPaTaskKind::Alloc) {
        ready = ready && kernel_id == INVALID_KERNEL_ID;
    } else {
        ready = ready && kernel_id >= 0 && kernel_id < RUNTIME_MAX_FUNC_ID && kernel_id <= INT16_MAX;
    }
#else
    bool ready = dist_shared_pa_kind_matches_task(state.task_id, kind);
    const DistSubmitKind submit_kind =
        kind == DistSharedPaTaskKind::Alloc ? DistSubmitKind::Alloc : DistSubmitKind::Kernel;
    if (ready && submit_kind == DistSubmitKind::Kernel) {
        ready = mixed != nullptr && dist_shared_pa_kernel_shape(*mixed, kind);
    } else if (ready) {
        ready = mixed == nullptr;
    }
#endif
    if (!ready) {
        (void)dist_shared_pa_fail(state.self, PTO2_ERROR_DIST_CONFIG_INVALID);
    }

    const uint64_t claim_begin = efdrain_end;
    fdwic_submit_pmu_phase_begin<FdwicSubmitPmuPhase::Claim>();
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
    const bool won = ready && dist_shared_pa_claim_fixed<CompiledReplayRole, CompiledKind>(kernel_id, state);
#else
    const bool won = ready && dist_shared_pa_claim(replay.role(), replay.block_id(), kind, mixed, state);
#endif
    const uint32_t claim_flags = (won ? kFdwicClaimWon : 0U) | (state.claim_attempted ? kFdwicClaimAttempted : 0U);
    fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::Claim>();
    TRACE_TIMESTAMP(claim_end);
    fdwic_submit_pmu_phase_begin<FdwicSubmitPmuPhase::ArgBuild>();
    fdwic_submit_pmu_empty_bracket_calibrate();
    TRACE_SPAN_RECORD(
        claim_begin, claim_end, state.self, state.task_id, state.kernel_id, TracePhase::Claim, claim_flags,
        static_cast<uint32_t>(kind)
    );
    DistCompeteFirstTicket ticket{};
    ticket.submit_begin = submit_begin;
    ticket.task_id = state.task_id;
    ticket.kernel_id = static_cast<int16_t>(state.kernel_id);
    ticket.won = static_cast<uint8_t>(won);
    ticket.ready = static_cast<uint8_t>(ready);
    // The nonwinner has no callback work between Begin and Finish. Close it
    // in this TU so only the 1,280 PA winners cross the Finish ABI; the ticket
    // remains available for direct callers to validate without closing twice.
    if (ready && !won && !dist_shared_pa_close_loser(state.self, ticket)) {
        ticket.ready = 0;
    }
    return ticket;
}

PTO_DEVICE_FUNC bool dist_shared_pa_finish_winner(
    DistSubmitCtx &ctx, const DistCompeteFirstTicket &ticket, const MixedKernels *mixed, DistSharedPaTaskKind kind,
    const L0TaskArgs &args
) {
    // One winner-side guard per task preserves fail-closed convergence without
    // adding a contended global atomic load to every one of the 95 losers.
    if (fdwic_trace_atomic_load(
            ctx.task_id, FdwicAtomicSite::SharedWinnerFatalGuardLoad, g_dist.fatal, /*result_used=*/true
        ) != 0) {
        if (ctx.self != nullptr) ctx.self->local_index = kFlagCap;
        return false;
    }
    DistSharedPaMaterializePlan materialize_plan;
    if (!dist_shared_pa_validate_and_plan(args, ctx.task_id, kind, materialize_plan)) {
        return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    ctx.tensor_count = args.tensor_count();
    ctx.scalar_count = args.scalar_count();

    fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::ArgBuild>();
    TRACE_TIMESTAMP(materialize_begin);
    fdwic_submit_pmu_phase_begin<FdwicSubmitPmuPhase::Materialize>();
    if (!dist_shared_pa_materialize_args(args, ctx, materialize_plan) ||
        ctx.result.size() != materialize_plan.output_count) {
        fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::Materialize>();
        return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
#if DIST_TRACE_ENABLED
    DistSharedPaOutputPublishTrace output_trace{};
    DistSharedPaOutputPublishTrace *output_trace_ptr = &output_trace;
#else
    DistSharedPaOutputPublishTrace *output_trace_ptr = nullptr;
#endif
    TRACE_TIMESTAMP(task_outputs_begin);
    const bool outputs_published = dist_shared_pa_publish_outputs(
        g_dist.shared_pa, ctx.task_id, ctx.result, materialize_plan.output_count, output_trace_ptr
    );
    TRACE_TIMESTAMP(task_outputs_end);
    if (!outputs_published) {
        fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::Materialize>();
        return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
#if DIST_TRACE_ENABLED
    DistSharedPaMetadataTrace metadata_trace;
    metadata_trace.history_dcci_lines = 0;
    metadata_trace.group_cas_captured = false;
    DistSharedPaMetadataTrace *metadata_trace_ptr = &metadata_trace;
#else
    DistSharedPaMetadataTrace *metadata_trace_ptr = nullptr;
#endif
    if (!dist_shared_pa_prepare_writer_history(args, ctx, kind, metadata_trace_ptr)) {
        fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::Materialize>();
        return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    if (kind == DistSharedPaTaskKind::Up) {
        const int32_t batch_start = ctx.task_id - (ctx.task_id % static_cast<int32_t>(kFdwicSharedPaTasksPerBatch));
        // The history clean-out above provides the predecessor-wait lead time
        // for this performance hint. Correctness still depends on the
        // return-ready group CAS in Register.
        DistSharedPaAicoreOps::PreloadDataCache(
            &g_dist.shared_pa.shared_outputs[static_cast<uint32_t>(batch_start)].last_writer[0]
        );
    }
    fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::Materialize>();
    TRACE_TIMESTAMP(materialize_end);

    fdwic_submit_pmu_phase_begin<FdwicSubmitPmuPhase::Register>();
    int64_t ready_observed = -1;
    uint32_t insert_turn_bypass_load_count = 0;
    uint64_t metadata_begin = 0;
    uint64_t metadata_end = 0;
    DistSharedPaHandoffTrace handoff_trace{};
    if (!dist_shared_pa_publish_metadata_and_handoff(
            ctx, kind, metadata_trace_ptr, ready_observed, insert_turn_bypass_load_count, metadata_begin, metadata_end,
            handoff_trace
        )) {
        fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::Register>();
        return false;
    }
    fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::Register>();
#if DIST_TRACE_ENABLED
    const uint64_t register_end = fdwic_swimlane_enabled() ? fdwic_atomic_result_ready_tick(handoff_trace.observed) : 0;
#else
    const uint64_t register_end = 0;
#endif

#if DIST_TRACE_ENABLED
    // The global insert turn has already been handed to task N+1. Only now
    // write diagnostic rows whose business endpoints were captured earlier,
    // so trace stores never extend the serialized metadata publication chain.
    if (metadata_trace.history_dcci_lines != 0) {
        (void)fdwic_swimlane_record_dcci(
            ctx.self, ctx.task_id, -1, FdwicDcciSite::SharedWriterHistoryFlush, FdwicDcciOp::CleanOut,
            /*trailing_dsb=*/true,
            /*call_count=*/1, metadata_trace.history_dcci_lines, metadata_trace.history_dcci_begin,
            metadata_trace.history_dcci_end
        );
    }
    if (metadata_trace.group_cas_captured) {
        (void)fdwic_swimlane_record_captured_atomic(
            ctx.task_id, FdwicAtomicSite::SharedMetadataLastWriterCommit, FdwicAtomicOp::CompareExchange,
            metadata_trace.group_cas_begin, metadata_trace.group_cas_end,
            /*result_used=*/true, fdwic_atomic_return_ready_observed()
        );
    }
#endif
    TRACE_SPAN_RECORD(
        materialize_begin, materialize_end, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Materialize, 0,
        kind == DistSharedPaTaskKind::Alloc ? 1U : 0U
    );
    TRACE_SPAN_RECORD(
        task_outputs_begin, task_outputs_end, ctx.self, ctx.task_id, ctx.kernel_id,
        TracePhase::SharedMaterializePublishTaskOutputs, 0, 0
    );
#if DIST_TRACE_ENABLED
    TRACE_SPAN_RECORD(
        output_trace.copy_begin, output_trace.copy_end, ctx.self, ctx.task_id, ctx.kernel_id,
        TracePhase::SharedMaterializePublishTaskOutputsCopy, 0, 0
    );
    TRACE_SPAN_RECORD(
        output_trace.flush_begin, output_trace.flush_end, ctx.self, ctx.task_id, ctx.kernel_id,
        TracePhase::SharedMaterializePublishTaskOutputsFlush, 0, 0
    );
#endif
    TRACE_SPAN_RECORD(materialize_end, register_end, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Register, 0, 0);
    TRACE_SPAN_RECORD(
        materialize_end, metadata_begin, ctx.self, ctx.task_id, ctx.kernel_id,
        TracePhase::SharedRegisterWaitInsertTurnBypassLoad, 0, insert_turn_bypass_load_count
    );
    TRACE_SPAN_RECORD(
        metadata_begin, metadata_end, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::SharedRegisterPublishMetadata,
        0, 0
    );
#if DIST_TRACE_ENABLED
    if (handoff_trace.captured) {
        (void)fdwic_swimlane_record_captured_atomic(
            ctx.task_id, FdwicAtomicSite::SharedInsertTurnHandoff, FdwicAtomicOp::CompareExchange, handoff_trace.begin,
            handoff_trace.end,
            /*result_used=*/true, fdwic_atomic_return_ready_observed()
        );
    }
#endif

    uint64_t build_begin = register_end;
    if (kind != DistSharedPaTaskKind::Alloc) {
        fdwic_submit_pmu_phase_begin<FdwicSubmitPmuPhase::Fanin>();
        const bool fanin_ok = dist_submit_collect_fanin(args, ctx, ctx.fanin, ctx.fanin_count);
        fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::Fanin>();
        TRACE_TIMESTAMP(fanin_end);
        TRACE_SPAN_RECORD(
            register_end, fanin_end, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Fanin, 0,
            static_cast<uint32_t>(ctx.fanin_count)
        );
        if (!fanin_ok) return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_PROTOCOL);
        build_begin = fanin_end;
    }

    if (kind == DistSharedPaTaskKind::Alloc) {
        fdwic_submit_pmu_phase_begin<FdwicSubmitPmuPhase::AllocComplete>();
        const bool completed = dist_submit_complete_alloc(ctx);
        fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::AllocComplete>();
        if (!completed) return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_PROTOCOL);
        TRACE_TIMESTAMP(build_end);
        TRACE_SPAN_RECORD(build_begin, build_end, ctx.self, ctx.task_id, -1, TracePhase::AllocComplete, 0, 0);
    } else {
        if (mixed == nullptr) return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_PROTOCOL);
        fdwic_submit_pmu_phase_begin<FdwicSubmitPmuPhase::WinnerBuild>();
        const bool built = dist_submit_build_winner_task(ctx, *mixed, args);
        fdwic_submit_pmu_phase_end<FdwicSubmitPmuPhase::WinnerBuild>();
        if (!built) return dist_shared_pa_fail(ctx, PTO2_ERROR_TENSORMAP_PROTOCOL);
        TRACE_TIMESTAMP(build_end);
        TRACE_SPAN_RECORD(build_begin, build_end, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::WinnerBuild, 0, 0);
    }
    TRACE_TIMESTAMP(submit_end);
    fdwic_perf_clock_submit_end(ctx.task_id);
    fdwic_submit_pmu_submit_end(ctx.task_id);
    TRACE_SPAN_RECORD(
        ticket.submit_begin, submit_end, ctx.self, ctx.task_id, ctx.kernel_id, TracePhase::Submit, 1,
        static_cast<uint32_t>(kind)
    );
    return true;
}

#endif  // PTO_FDWIC_SHARED_MAP
