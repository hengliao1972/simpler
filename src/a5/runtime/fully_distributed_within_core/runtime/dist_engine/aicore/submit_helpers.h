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

#include "dist_engine/common/target.h"

namespace {

PTO_DEVICE_FUNC bool claim(__gm__ volatile int64_t &cursor, int32_t N) {
    if (N < 0 || N >= kFlagCap) return false;
#if defined(__CCE_AICORE__)
    __gm__ int64_t *addr = const_cast<__gm__ int64_t *>(&cursor);
    const int64_t old = atomicMax(addr, static_cast<int64_t>(N));
    return N > old;
#else
    int64_t c = atom_load(cursor, __ATOMIC_ACQUIRE);
    while (N > c) {
        if (atom_cas_weak(cursor, c, static_cast<int64_t>(N), __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return true;
    }
    return false;
#endif
}

PTO_DEVICE_FUNC uint64_t load_task_vend(int32_t task_id) {
    if (task_id < 0) return 0;
    __gm__ DistTaskCell &cell = task_cell(task_id);
#if defined(__CCE_AICORE__)
    dist_aicore_invalidate_region(&cell, sizeof(DistTaskCell));
    return cell.vend;
#else
    return atom_load(cell.vend, __ATOMIC_RELAXED);
#endif
}

// Resolve a kernel id to its executable address (CoreCallable::resolved_addr()).
// Reads Runtime::func_id_to_addr_ directly (public POD array) rather than
// calling the get_function_bin_addr() member so this compiles into libaicore
// too: the AICore .so does not link against libaicpu, so a member-function
// dispatch would fail with an unresolved symbol at dlopen.
PTO_DEVICE_FUNC uint64_t resolve_kernel_addr(Runtime *runtime, int32_t kernel_id) {
    if (kernel_id == INVALID_KERNEL_ID) return 0;
    if (kernel_id < 0 || kernel_id >= RUNTIME_MAX_FUNC_ID) return 0;
    uint64_t callable_addr = runtime->func_id_to_addr_[kernel_id];
    if (callable_addr == 0) return 0;
    const CoreCallable *callable = reinterpret_cast<const CoreCallable *>(callable_addr);
    return callable->resolved_addr();
}

PTO_DEVICE_FUNC void store_won_remaining(__gm__ WonSlot &w, int32_t count) {
#if defined(__CCE_AICORE__)
    w.remaining = count;
#else
    atom_store<int64_t>(w.remaining, count, __ATOMIC_RELAXED);
#endif
}

PTO_DEVICE_FUNC void reset_won_lane(__gm__ WonSlot &w, int32_t lane) {
#if defined(__CCE_AICORE__)
    w.drained[lane].v = 0;
#else
    atom_store(w.drained[lane].v, 0, __ATOMIC_RELAXED);
#endif
    w.lane[lane].present = false;
}

PTO_DEVICE_FUNC bool decrement_won_remaining_is_last(__gm__ WonSlot &w) {
#if defined(__CCE_AICORE__)
    __gm__ int64_t *remaining = const_cast<__gm__ int64_t *>(&w.remaining);
    return atomicSub(remaining, static_cast<int64_t>(1)) == 1;
#else
    return atom_fetch_sub<int64_t>(w.remaining, 1, __ATOMIC_ACQ_REL) == 1;
#endif
}

PTO_DEVICE_FUNC void publish_replay_done(__gm__ int64_t &replay_done) {
#if defined(__CCE_AICORE__)
    __gm__ int64_t *addr = const_cast<__gm__ int64_t *>(&replay_done);
    atomicAdd(addr, static_cast<int64_t>(1));
#else
    atom_fetch_add<int64_t>(replay_done, 1, __ATOMIC_ACQ_REL);
#endif
}

template <typename TensorArrPtr, typename ScalarArrPtr, typename FaninArrPtr>
PTO_DEVICE_FUNC void populate_won_slot(
    __gm__ WonSlot &w, int32_t task_id, const ActiveMask &M, const MixedKernels &mixed, int32_t own_lane,
    Runtime *runtime, TensorArrPtr tensors, int32_t tc, ScalarArrPtr scalars, int32_t sc, FaninArrPtr fanin, int32_t fc
) {
    const int32_t pc = __builtin_popcount(M.core_mask());
    w.task_id = task_id;
    store_won_remaining(w, pc);
#define POPULATE_WON_LANE(L)                                                                    \
    do {                                                                                        \
        if ((L) == own_lane || !lane_active(M, (L))) break;                                     \
        __gm__ BuiltSubtask &b = w.lane[(L)];                                                   \
        b.present = true;                                                                       \
        b.func_id = kernel_id_for_lane(mixed, (L));                                             \
        b.function_bin_addr = runtime != nullptr ? resolve_kernel_addr(runtime, b.func_id) : 0; \
        b.tensor_count = tc;                                                                    \
        b.scalar_count = sc;                                                                    \
        for (int32_t i = 0; i < tc; i++)                                                        \
            Tensor::copy(b.tensors[i], tensors[i]);                                             \
        for (int32_t j = 0; j < sc; j++)                                                        \
            b.scalars[j] = scalars[j];                                                          \
        b.fanin_count = fc;                                                                     \
        for (int32_t k = 0; k < fc; k++)                                                        \
            b.fanin[k] = fanin[k];                                                              \
        b.sub_block_id = ((L) == LANE_AIV1) ? 1 : 0;                                            \
    } while (0)
    reset_won_lane(w, LANE_AIC);
    reset_won_lane(w, LANE_AIV0);
    reset_won_lane(w, LANE_AIV1);
    POPULATE_WON_LANE(LANE_AIC);
    POPULATE_WON_LANE(LANE_AIV0);
    POPULATE_WON_LANE(LANE_AIV1);
#undef POPULATE_WON_LANE
}

PTO_DEVICE_FUNC int32_t alloc_won_slot(int32_t block) {
    __gm__ BlockWon &bw = g_dist.blocks[block];
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        int32_t exp = 0;
        if (atom_cas_strong(bw.slots[i].state, exp, 2, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            return i;
        }
    }
    return -1;
}

}  // namespace
