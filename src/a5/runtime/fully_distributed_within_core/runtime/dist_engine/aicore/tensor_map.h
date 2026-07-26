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

#include "dist_engine/aicore/private_tensor_map.h"
#if PTO_FDWIC_SHARED_MAP
#include "dist_engine/aicore/shared_tensor_map.h"
#endif

namespace {

/*
 * Submit uses this facade instead of reading DistCore::map directly. Both
 * modes share hash/range/slot semantics, while ownership and publication
 * discipline remain compile-time choices:
 *
 * - private resets and retires one map per worker; every worker publishes its
 *   task outputs into its own copy;
 * - shared is reset once by AICPU; only the exact-turn Submit winner reads or
 *   publishes the global map. CPU-sim scalar data access has no Claim/turn
 *   proof and remains explicitly unsupported under the shared identity.
 *
 * Keeping the split here prevents loser paths and worker reset from
 * accidentally touching the shared single copy.
 */
PTO_DEVICE_FUNC void dist_tensor_map_reset_worker(__gm__ DistCore &worker) {
#if PTO_FDWIC_SHARED_MAP
    (void)worker;
#else
    dist_private_tensor_map_reset(worker.map);
#endif
}

PTO_DEVICE_FUNC void dist_tensor_map_prepare_task(__gm__ DistCore &worker, int32_t task_id, int32_t history) {
#if PTO_FDWIC_SHARED_MAP
    (void)worker;
    (void)task_id;
    (void)history;
#else
    dist_private_tensor_map_advance_retire(worker.map, task_id, history);
#endif
}

#if PTO_FDWIC_SHARED_MAP
template <typename TensorRef>
PTO_DEVICE_FUNC bool dist_tensor_map_lookup_for_submit_winner(
    __gm__ DistCore &worker, const TensorRef &tensor, int32_t consumer_task_id, int32_t &producer
) {
    (void)worker;
    bool protocol_ok = false;
    producer =
        dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, tensor, consumer_task_id, g_dist.H, protocol_ok);
    return protocol_ok;
}
#endif

template <typename TensorRef>
PTO_DEVICE_FUNC int32_t
dist_tensor_map_lookup_for_task(__gm__ DistCore &worker, const TensorRef &tensor, int32_t consumer_task_id) {
#if PTO_FDWIC_SHARED_MAP
    (void)worker;
    (void)tensor;
    (void)consumer_task_id;
    set_fatal_code(PTO2_ERROR_TENSORMAP_PROTOCOL);
    return -1;
#else
    (void)consumer_task_id;
    return dist_private_tensor_map_lookup(worker.map, tensor);
#endif
}

#if PTO_FDWIC_SHARED_MAP
PTO_DEVICE_FUNC int64_t dist_tensor_map_next_publish_task() {
    return DistSharedTensorMapAicoreOps::Load(&g_dist.shared_tensor_map.committed_tasks.v);
}

PTO_DEVICE_FUNC DistSharedTensorMapTaskPublishResult
dist_tensor_map_publish_shared_task(const SharedTensorMapValue *entries, uint32_t count, int32_t producer_task_id) {
    return dist_shared_tensor_map_publish_task(g_dist.shared_tensor_map, entries, count, producer_task_id, g_dist.H);
}
#else
template <typename TensorRef>
PTO_DEVICE_FUNC bool dist_tensor_map_insert_for_task(
    __gm__ DistCore &worker, const TensorRef &tensor, int32_t producer_task_id, bool task_won
) {
    (void)task_won;
    return dist_private_tensor_map_insert(worker.map, tensor, producer_task_id);
}
#endif

}  // namespace
