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
    const int64_t old = atomic_fetch_max<int64_t>(cursor, static_cast<int64_t>(N));
    return N > old;
}

PTO_DEVICE_FUNC uint64_t load_task_vend(int32_t task_id) {
    if (task_id < 0) return 0;
    __gm__ DistTaskCell &cell = task_cell(task_id);
    return atomic_load(cell.vend, __ATOMIC_RELAXED);
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

template <typename FaninArrPtr>
PTO_DEVICE_FUNC void populate_won_slot_from_submit(
    __gm__ WonSlot &w, int32_t task_id, const ActiveMask &M, const MixedKernels &mixed, int32_t own_lane,
    Runtime *runtime, const L0TaskArgs &args, const DistSubmitCtx &ctx, FaninArrPtr fanin, int32_t fc
) {
    w.meta.task_id = task_id;
#define POPULATE_WON_LANE_FROM_SUBMIT(L)                                                        \
    do {                                                                                        \
        if ((L) == own_lane || !lane_active(M, (L))) break;                                     \
        __gm__ BuiltSubtask &b = w.lane[(L)];                                                   \
        b.present = true;                                                                       \
        b.func_id = kernel_id_for_lane(mixed, (L));                                             \
        b.function_bin_addr = runtime != nullptr ? resolve_kernel_addr(runtime, b.func_id) : 0; \
        b.tensor_count = ctx.tensor_count;                                                      \
        b.scalar_count = ctx.scalar_count;                                                      \
        for (int32_t i = 0; i < ctx.tensor_count; i++)                                          \
            dist_submit_copy_arg_tensor(b.tensors[i], args, ctx, i);                            \
        for (int32_t j = 0; j < ctx.scalar_count; j++)                                          \
            b.scalars[j] = args.scalar(j);                                                      \
        b.fanin_count = fc;                                                                     \
        for (int32_t k = 0; k < fc; k++)                                                        \
            b.fanin[k] = fanin[k];                                                              \
        b.sub_block_id = ((L) == LANE_AIV1) ? 1 : 0;                                            \
        atomic_exchange(w.drained[(L)].v, kDrainedFree);                                        \
    } while (0)
    reset_won_lane(w, LANE_AIC);
    reset_won_lane(w, LANE_AIV0);
    reset_won_lane(w, LANE_AIV1);
    POPULATE_WON_LANE_FROM_SUBMIT(LANE_AIC);
    POPULATE_WON_LANE_FROM_SUBMIT(LANE_AIV0);
    POPULATE_WON_LANE_FROM_SUBMIT(LANE_AIV1);
#undef POPULATE_WON_LANE_FROM_SUBMIT
}

PTO_DEVICE_FUNC int32_t alloc_won_slot(int32_t block) {
    __gm__ BlockWon &bw = g_dist.blocks[block];
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        if (atomic_fetch_max<int64_t>(bw.slots[i].state.v, kWonStateClaimed) == kWonStateFree) return i;
    }
    return -1;
}

}  // namespace
