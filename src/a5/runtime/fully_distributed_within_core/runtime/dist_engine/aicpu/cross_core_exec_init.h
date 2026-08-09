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

#if PTO_FDWIC_SCHEDULER_MODE == 1 || PTO_FDWIC_SCHEDULER_MODE == 2 || PTO_FDWIC_SCHEDULER_MODE == 3 || \
    PTO_FDWIC_SCHEDULER_MODE == 4
inline void dist_cross_core_runtime_reset(CrossCoreRuntimeState &state) {
    atomic_exchange(state.fatal.state, int64_t{0}, __ATOMIC_RELAXED);
    atomic_exchange(state.heap_cursor.state, int64_t{0}, __ATOMIC_RELAXED);
    for (uint32_t task = 0; task < kFdwicCrossCoreOrdinaryTaskCapacity; ++task) {
        atomic_exchange(state.execute_owner[task].state, int64_t{0}, __ATOMIC_RELAXED);
        atomic_exchange(state.tasks[task].control.state, int64_t{0}, __ATOMIC_RELAXED);
        atomic_exchange(state.outputs[task].control.state, int64_t{0}, __ATOMIC_RELAXED);
        atomic_exchange(state.build_tournament[task].root.owner.v, int64_t{-1}, __ATOMIC_RELAXED);
        for (uint32_t group = 0; group < kFdwicSharedClaimTournamentMaxGroups; ++group) {
            atomic_exchange(state.build_tournament[task].local[group].owner.v, int64_t{-1}, __ATOMIC_RELAXED);
        }
    }
    for (uint32_t bucket = 0; bucket < fdwic::cross_core::kCrossMapBuckets; ++bucket) {
        atomic_exchange(state.tensor_map.tails[bucket].state, int64_t{0}, __ATOMIC_RELAXED);
    }
    for (uint32_t slot = 0; slot < fdwic::cross_core::kCrossMapCapacity; ++slot) {
        atomic_exchange(state.tensor_map.slots[slot].sequence.state, int64_t{-1}, __ATOMIC_RELAXED);
    }
}

#if PTO_FDWIC_SCHEDULER_MODE == 1
inline void dist_cross_core_ordinary_reset(CrossCoreOrdinaryState &state) { dist_cross_core_runtime_reset(state); }
#elif PTO_FDWIC_SCHEDULER_MODE == 2
inline void dist_cross_core_dag_reset(CrossCoreDagState &state) {
    dist_cross_core_runtime_reset(state.runtime);
    for (uint32_t task = 0; task < kFdwicCrossCoreTaskCapacity; ++task) {
        atomic_exchange(state.metadata[task].control.state, int64_t{0}, __ATOMIC_RELAXED);
    }
}
#else
template <typename State>
inline void dist_simt_cross_core_common_reset(State &state) {
    dist_cross_core_runtime_reset(state.runtime);
    atomic_exchange(state.lifecycle.builder_started.state, int64_t{0}, __ATOMIC_RELAXED);
    atomic_exchange(state.lifecycle.sealed_task_count.state, int64_t{-1}, __ATOMIC_RELAXED);
    atomic_exchange(state.lifecycle.builder_finished.state, int64_t{0}, __ATOMIC_RELAXED);
    for (uint32_t task = 0; task < kFdwicCrossCoreTaskCapacity; ++task) {
        atomic_exchange(state.requests[task].control.state, int64_t{0}, __ATOMIC_RELAXED);
    }
}

#if PTO_FDWIC_SCHEDULER_MODE == 3
inline void dist_simt_cross_core_ordinary_reset(SimtCrossCoreOrdinaryState &state) {
    dist_simt_cross_core_common_reset(state);
}
#else
inline void dist_simt_cross_core_dag_reset(SimtCrossCoreDagState &state) {
    dist_simt_cross_core_common_reset(state);
    for (uint32_t task = 0; task < kFdwicCrossCoreTaskCapacity; ++task) {
        atomic_exchange(state.metadata[task].control.state, int64_t{0}, __ATOMIC_RELAXED);
    }
}
#endif
#endif
#endif

}  // namespace
