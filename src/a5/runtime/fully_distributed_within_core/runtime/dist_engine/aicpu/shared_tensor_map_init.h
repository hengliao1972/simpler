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

// shared sidecar 只能由 AICPU setup thread 在唤醒 worker 前初始化一次。
// payload 无需清零：seq=-1 是唯一有效性判据，且随后 AICPU 会把整个
// DistGlobal flush 到 GM。显式重置所有控制字可以支持同一 arena 的重复 run，
// 也避免把 slot 0 的合法 seq=0 与零填充状态混为一谈。
inline void dist_shared_tensor_map_reset(SharedTensorMapState &state) {
    atomic_exchange(state.committed_tasks.v, int64_t{kSharedTensorMapInitialCommit}, __ATOMIC_RELAXED);
    atomic_exchange(state.reclaim_upto.v, int64_t{kSharedTensorMapInitialReclaim}, __ATOMIC_RELAXED);
    for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
        atomic_exchange(state.buckets[bucket].head.v, int64_t{0}, __ATOMIC_RELAXED);
        atomic_exchange(state.buckets[bucket].tail.v, int64_t{0}, __ATOMIC_RELAXED);
    }
    for (int32_t slot = 0; slot < kMapCap; ++slot) {
        atomic_exchange(state.slots[slot].sequence.v, int64_t{kSharedTensorMapInvalidSequence}, __ATOMIC_RELAXED);
    }
}

}  // namespace
