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

#if PTO_FDWIC_SCHEDULER_MODE == 1
inline void dist_cross_core_ordinary_reset(CrossCoreOrdinaryState &state) {
    atomic_exchange(state.fatal.state, int64_t{0}, __ATOMIC_RELAXED);
    atomic_exchange(state.heap_cursor.state, int64_t{0}, __ATOMIC_RELAXED);
    for (uint32_t task = 0; task < kFdwicCrossCoreOrdinaryTaskCapacity; ++task) {
        atomic_exchange(state.tasks[task].control.state, int64_t{0}, __ATOMIC_RELAXED);
        atomic_exchange(state.outputs[task].control.state, int64_t{0}, __ATOMIC_RELAXED);
    }
    for (uint32_t bucket = 0; bucket < fdwic::cross_core::kCrossMapBuckets; ++bucket) {
        atomic_exchange(state.tensor_map.tails[bucket].state, int64_t{0}, __ATOMIC_RELAXED);
    }
    for (uint32_t slot = 0; slot < fdwic::cross_core::kCrossMapCapacity; ++slot) {
        atomic_exchange(state.tensor_map.slots[slot].sequence.state, int64_t{-1}, __ATOMIC_RELAXED);
    }
}
#endif

}  // namespace
