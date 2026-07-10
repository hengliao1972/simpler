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

PTO_DEVICE_FUNC __gm__ DistCore *dist_aicore_attach_worker(__gm__ Runtime *runtime, int core_idx, int core_type_int) {
    if (runtime == nullptr || core_idx < 0 || core_idx >= RUNTIME_MAX_WORKER) return nullptr;

#if defined(__CCE_AICORE__)
    dist_aicore_invalidate_region(const_cast<__gm__ uint64_t *>(&runtime->dist.shared_addr), 64);
    g_dist_ptr = reinterpret_cast<__gm__ DistGlobal *>(runtime->dist.shared_addr);
#else
    g_dist_ptr = reinterpret_cast<DistGlobal *>(runtime->dist.shared_addr);
#endif
    if (g_dist_ptr == nullptr) return nullptr;

#if defined(__CCE_AICORE__)
    g_ccec_runtime = runtime;
    g_ccec_core_idx = core_idx;
    g_ccec_core_type = core_type_int;
#endif
    g_self = &g_dist.cores[core_idx];

    dist_aicore_invalidate_region(&g_dist.layout[core_idx], sizeof(CoreLayout));
    __gm__ const CoreLayout &layout_gm = g_dist.layout[core_idx];
    const CoreLayout lay = {layout_gm.block_id, layout_gm.lane};
    dist_core_reset(*g_self, static_cast<CoreType>(core_type_int), lay.block_id, lay.lane);
    g_self->core_idx = core_idx;

#if defined(__CCE_AICORE__)
    dist_aicore_invalidate_region(const_cast<__gm__ int32_t *>(&runtime->dist.num_workers), 64);
    ccec_init_worker_layout();
    ccec_attach_run_state();
#endif
    return g_self;
}

PTO_DEVICE_FUNC void dist_aicore_finish_worker(__gm__ Runtime *runtime) {
#if defined(__CCE_AICORE__)
    (void)runtime;
    ccec_finish_worker();
#else
    g_self = nullptr;
    ccec_publish_done();
    __atomic_add_fetch(&runtime->dist.done_count, 1, __ATOMIC_ACQ_REL);
#endif
}
