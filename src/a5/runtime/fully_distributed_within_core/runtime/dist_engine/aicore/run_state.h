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

PTO_DEVICE_FUNC void ccec_init_worker_layout() {
#if defined(__CCE_AICORE__)
    if (g_ccec_runtime == nullptr) return;
    const int32_t num_workers = g_ccec_runtime->dist.num_workers;
    const int32_t block_dim = num_workers / PLATFORM_CORES_PER_BLOCKDIM;
    g_ccec_aic_count = block_dim;
    g_ccec_aiv_count = block_dim * PLATFORM_AIV_CORES_PER_BLOCKDIM;
    g_ccec_ordinal = INVALID_KERNEL_ID;
    g_ccec_valid_worker = false;
    const CoreType self_type = static_cast<CoreType>(g_ccec_core_type);
    if (self_type == CoreType::AIC && g_ccec_core_idx >= 0 && g_ccec_core_idx < block_dim) {
        g_ccec_ordinal = g_ccec_core_idx;
        g_ccec_valid_worker = true;
    } else if (self_type == CoreType::AIV && g_ccec_core_idx >= block_dim && g_ccec_core_idx < num_workers) {
        g_ccec_ordinal = g_ccec_core_idx - block_dim;
        g_ccec_valid_worker = true;
    }
#endif
}

PTO_DEVICE_FUNC bool ccec_is_valid_worker() {
#if defined(__CCE_AICORE__)
    return g_ccec_valid_worker;
#else
    return true;
#endif
}

PTO_DEVICE_FUNC void ccec_attach_run_state() {
#if defined(__CCE_AICORE__)
    dist_aicore_invalidate_region(const_cast<__gm__ int32_t *>(&g_dist.num_blocks), sizeof(g_dist.num_blocks));
    dist_aicore_invalidate_region(g_dist.cube_cursor, sizeof(g_dist.cube_cursor));
    dist_aicore_invalidate_region(g_dist.vector_cursor, sizeof(g_dist.vector_cursor));
    dist_aicore_invalidate_region(g_dist.alloc_cursor, sizeof(g_dist.alloc_cursor));
    dist_aicore_invalidate_region(const_cast<__gm__ int64_t *>(&g_dist.frontier), sizeof(g_dist.frontier));
    dist_aicore_invalidate_region(const_cast<__gm__ int64_t *>(&g_dist.replay_done), sizeof(g_dist.replay_done));
    dist_aicore_invalidate_region(const_cast<__gm__ int64_t *>(&g_dist.started_count), sizeof(g_dist.started_count));
    dist_aicore_invalidate_region(const_cast<__gm__ int32_t *>(&g_dist.fatal), sizeof(g_dist.fatal));
    for (int32_t b = 0; b < g_dist.num_blocks; b++) {
        dist_aicore_invalidate_region(&g_dist.blocks[b], sizeof(BlockWon));
    }
#endif
}

PTO_DEVICE_FUNC void ccec_publish_done() { dist_aicore_publish_done(); }

PTO_DEVICE_FUNC void ccec_finish_worker() {
#if defined(__CCE_AICORE__)
    g_self = nullptr;
    g_ccec_runtime = nullptr;
    g_ccec_valid_worker = false;
    ccec_publish_done();
#endif
}
