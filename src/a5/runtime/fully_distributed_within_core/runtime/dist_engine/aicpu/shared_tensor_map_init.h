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

#include "dist_engine/common/atomic.h"
#include "dist_engine/common/state.h"

namespace {

// Only the AICPU setup thread may reset the shared PA sidecar, before workers
// are released. Descriptor/history payload bytes need not be cleared: their
// publication controls return to -1, and every writer overwrites a complete
// payload before publishing it. Resetting every control word supports repeated
// runs in the same runtime arena without introducing a generation protocol.
inline void dist_shared_pa_tensor_map_reset(SharedPaTensorMapState &state) {
    for (uint32_t task = 0; task < kFdwicSharedPaTaskCapacity; ++task) {
        for (uint32_t output = 0; output < kFdwicSharedOutputMaxPerTask; ++output) {
            atomic_exchange(state.shared_outputs[task].published[output].v, int64_t{-1}, __ATOMIC_RELAXED);
            atomic_exchange(state.shared_outputs[task].last_writer[output].v, int64_t{-1}, __ATOMIC_RELAXED);
        }
        state.writer_history[task].magic = 0;
        state.writer_history[task].writer_task = -1;
        state.writer_history[task].count = 0;
        state.writer_history[task].reserved = 0;
        atomic_exchange(state.claim_tournament[task].root.owner.v, int64_t{-1}, __ATOMIC_RELAXED);
        for (uint32_t group = 0; group < kFdwicSharedClaimTournamentMaxGroups; ++group) {
            atomic_exchange(
                state.claim_tournament[task].local[group].owner.v, int64_t{-1}, __ATOMIC_RELAXED
            );
        }
    }
    for (uint32_t shard = 0; shard < kFdwicSharedHeapShards; ++shard) {
        atomic_exchange(state.shared_heap_cursor[shard].v, int64_t{0}, __ATOMIC_RELAXED);
    }
    atomic_exchange(state.shared_heap_vend.v, int64_t{0}, __ATOMIC_RELAXED);
    for (uint32_t shard = 0; shard < kFdwicSharedVectorCursorShards; ++shard) {
        atomic_exchange(state.shared_vector_cursor[shard].v, int64_t{-1}, __ATOMIC_RELAXED);
    }
}

}  // namespace
