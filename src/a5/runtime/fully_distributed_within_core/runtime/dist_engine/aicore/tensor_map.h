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
 * Submit 只依赖这组模式无关入口，不再直接读取 DistCore::map。当前 shared
 * 镜像仍在零 Submit 前 fail-closed，因此本阶段两种编译身份都复用既有
 * private 实现来保证完整编译；解除门禁前会在这里接入真正的 shared
 * backend，而不是让 shared 运行时沿用 private 语义。
 *
 * consumer_task_id 和 task_won 暂时不会改变 private 行为。它们是 shared
 * 后端必须显式取得的两条上下文：lookup 需要过滤未来 producer，insert
 * 只允许当前 task 的唯一 winner 发布。把上下文固定在 facade 参数中，
 * 可避免后续在 Submit 各处散布 PTO_FDWIC_SHARED_MAP 分支。
 */
PTO_DEVICE_FUNC void dist_tensor_map_reset_worker(__gm__ DistCore &worker) {
    dist_private_tensor_map_reset(worker.map);
}

PTO_DEVICE_FUNC void dist_tensor_map_prepare_task(__gm__ DistCore &worker, int32_t task_id, int32_t history) {
    dist_private_tensor_map_advance_retire(worker.map, task_id, history);
}

template <typename TensorRef>
PTO_DEVICE_FUNC int32_t
dist_tensor_map_lookup_for_task(__gm__ DistCore &worker, const TensorRef &tensor, int32_t consumer_task_id) {
    (void)consumer_task_id;
    return dist_private_tensor_map_lookup(worker.map, tensor);
}

template <typename TensorRef>
PTO_DEVICE_FUNC bool dist_tensor_map_insert_for_task(
    __gm__ DistCore &worker, const TensorRef &tensor, int32_t producer_task_id, bool task_won
) {
    (void)task_won;
    return dist_private_tensor_map_insert(worker.map, tensor, producer_task_id);
}

}  // namespace
