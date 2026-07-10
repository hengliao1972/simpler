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

#include "dist_engine/aicore/tensor_scalar_access.h"

namespace {

#if !defined(__CCE_AICORE__)
// AICore sim orchestration replay shares the direct submit path, so scalar
// reads/writes must drain the worker's own queue until the producer is complete.
PTO_DEVICE_FUNC void wait_producer_ready(DistCore *self, const Tensor &t) {
    const int32_t p = dist_tensor_map_lookup(self->map, t);
    if (p < 0) return;
    uint64_t wd = 0;
    while (!fatal_set()) {
        if (task_flag_ready(p, __ATOMIC_ACQUIRE)) break;
        drain_block_won(self);
        if (drain_phase_b(self) == 0) {
            SPIN_WAIT_HINT();
            watchdog(wd);
        }
    }
}
#endif

PTO_DEVICE_FUNC void wait_tensor_data_access_ready(const Tensor &tensor) {
#if !defined(__CCE_AICORE__)
    DistCore *self = g_self;
    if (self != nullptr) wait_producer_ready(self, tensor);
#else
    (void)tensor;
#endif
}

}  // namespace

DIST_API_ATTR PTO_DEVICE_FUNC uint64_t
dist_get_tensor_data_impl(PTO2Runtime *, const Tensor &tensor, uint32_t ndims, const uint32_t indices[]) {
    if (tensor.buffer.addr == 0) return 0;
    wait_tensor_data_access_ready(tensor);
    return dist_read_tensor_scalar_raw(tensor, ndims, indices);
}

DIST_API_ATTR PTO_DEVICE_FUNC void dist_set_tensor_data_impl(
    PTO2Runtime *, const Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
) {
    if (tensor.buffer.addr == 0) return;
    wait_tensor_data_access_ready(tensor);
    dist_write_tensor_scalar_raw(tensor, ndims, indices, value);
}
