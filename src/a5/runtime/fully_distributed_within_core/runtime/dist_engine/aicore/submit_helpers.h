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

PTO_DEVICE_FUNC void publish_replay_done(__gm__ int64_t &replay_done) {
#if defined(__CCE_AICORE__)
    __gm__ int64_t *addr = const_cast<__gm__ int64_t *>(&replay_done);
    atomicAdd(addr, static_cast<int64_t>(1));
#else
    atom_fetch_add<int64_t>(replay_done, 1, __ATOMIC_ACQ_REL);
#endif
}

PTO_DEVICE_FUNC void publish_worker_started() {
#if defined(__CCE_AICORE__)
    __gm__ int64_t *started = const_cast<__gm__ int64_t *>(&g_dist.started_count);
    atomicAdd(started, static_cast<int64_t>(1));
#else
    atom_fetch_add<int64_t>(g_dist.started_count, 1, __ATOMIC_ACQ_REL);
#endif
}

PTO_DEVICE_FUNC int64_t load_worker_started_count() {
#if defined(__CCE_AICORE__)
    dist_aicore_invalidate_region(const_cast<__gm__ int64_t *>(&g_dist.started_count), sizeof(g_dist.started_count));
    return g_dist.started_count;
#else
    return atom_load(g_dist.started_count, __ATOMIC_ACQUIRE);
#endif
}

template <typename TensorArrPtr, typename ScalarArrPtr, typename FaninArrPtr>
PTO_DEVICE_FUNC void populate_won_slot(
    __gm__ WonSlot &w, int32_t task_id, const ActiveMask &M, const MixedKernels &mixed, int32_t own_lane,
    Runtime *runtime, TensorArrPtr tensors, int32_t tc, ScalarArrPtr scalars, int32_t sc, FaninArrPtr fanin, int32_t fc
) {
    w.task_id = task_id;
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
        exchange_won_drained(w, (L), kDrainedFree);                                             \
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
#if defined(__CCE_AICORE__)
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ int64_t *state = const_cast<__gm__ int64_t *>(&bw.slots[i].state);
        if (atomicMax(state, kWonStateClaimed) == kWonStateFree) return i;
    }
    return -1;
#else
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        int64_t exp = kWonStateFree;
        if (atom_cas_strong(bw.slots[i].state, exp, kWonStateClaimed, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
            return i;
        }
    }
    return -1;
#endif
}

}  // namespace
