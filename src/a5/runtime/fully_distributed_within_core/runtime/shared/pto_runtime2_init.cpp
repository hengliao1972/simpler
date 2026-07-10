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
/**
 * Host-side runtime-arena layout and init_data implementations.
 */

#include <string.h>

#include "dist_engine/dist_engine.h"
#include "pto_runtime2.h"

// =============================================================================
// Top-level runtime arena
// =============================================================================

PTO2RuntimeArenaLayout runtime_reserve_layout(DeviceArena &arena) {
    PTO2RuntimeArenaLayout layout{};
    layout.off_dist_global = arena.reserve(dist_engine_global_state_size(), dist_engine_global_state_align());
    layout.off_runtime = arena.reserve(sizeof(PTO2Runtime), PTO2_ALIGN_SIZE);

    layout.arena_size = arena.total_size();
    return layout;
}

PTO2Runtime *runtime_init_data_from_layout(
    DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2RuntimeMode mode, void *gm_heap_dev_base,
    uint64_t heap_size
) {
    PTO2Runtime *rt = static_cast<PTO2Runtime *>(arena.region_ptr(layout.off_runtime));
    *rt = PTO2Runtime{};
    memset(arena.region_ptr(layout.off_dist_global), 0, dist_engine_global_state_size());

    // CPU-sim AICore binds rt->ops to its local distributed ops table when
    // entering dist_core_main. CCEC wrappers call dist_engine symbols directly.
    rt->mode = mode;
    rt->dist_global = arena.region_ptr(layout.off_dist_global);
    rt->gm_heap = gm_heap_dev_base;
    rt->gm_heap_size = heap_size;
    rt->gm_heap_owned = false;
    rt->total_cycles = 0;

    return rt;
}

void runtime_destroy(PTO2Runtime *rt, DeviceArena & /*arena*/) {
    // Arena buffer is pooled across runs by DeviceRunner — never freed here.
    if (!rt) return;
    rt->dist_global = nullptr;
}
