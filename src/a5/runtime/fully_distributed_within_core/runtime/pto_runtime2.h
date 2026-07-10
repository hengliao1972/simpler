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
 * PTO Runtime2 - Main Interface
 *
 * Runtime2 data layout shared by the host prebuilt image and the direct
 * distributed AICore replay path.
 */

#pragma once

#include "utils/device_arena.h"
#include "pto_runtime2_types.h"
#include "pto_submit_types.h"
#include "pto_tensormap.h"
#include "aicore_completion_mailbox.h"

// =============================================================================
// Runtime Context
// =============================================================================

/**
 * Runtime execution mode
 */
enum PTO2RuntimeMode {
    PTO2_MODE_EXECUTE = 0,    // Execute tasks on workers
    PTO2_MODE_SIMULATE = 1,   // Simulate task execution with cycle counting
    PTO2_MODE_GRAPH_ONLY = 2  // Build graph only, no execution
};

typedef struct PTO2Runtime PTO2Runtime;  // forward declare for ops signatures

/**
 * Layout descriptor for the prebuilt dist-engine runtime arena. Holds the
 * runtime header and the distributed engine's global state. Produced once on
 * the host by runtime_reserve_layout(); consumed by runtime_init_data_from_layout.
 */
struct PTO2RuntimeArenaLayout {
    size_t off_dist_global{0};
    size_t off_runtime{0};

    // Total arena byte size post-commit. Used by host to size the prebuilt
    // image buffer and as the rtMemcpy length.
    size_t arena_size{0};
};

/**
 * PTO Runtime2 context
 *
 * Contains the direct distributed runtime header shared between host, AICPU,
 * and AICore replay.
 */
struct PTO2Runtime {
    PTO2ScopeMode pending_scope_mode;

    // Components
    void *dist_global;

    // GM Heap for output buffers
    void *gm_heap;
    uint64_t gm_heap_size;
    bool gm_heap_owned;  // True if we allocated it

    // Mode
    PTO2RuntimeMode mode;

    // Statistics
    int64_t total_cycles;

    // Prebuilt-arena fast path metadata. The host builds a complete runtime
    // header image and records the layout beside it. The direct AICPU path
    // reads the header from Runtime::prebuilt_arena_base_ and passes it to
    // dist_engine_register; it no longer wires the full arena at AICPU boot.
    PTO2RuntimeArenaLayout prebuilt_layout;
};

// =============================================================================
// Runtime Lifecycle API
// =============================================================================

/**
 * Phase 1 — declare the dist-engine runtime header and global-state regions on
 * the supplied arena. Pure arithmetic; does not touch device memory and may run
 * on host. Returns the layout descriptor; caller commits/attaches the arena
 * before Phase 2/3.
 */
PTO2RuntimeArenaLayout runtime_reserve_layout(DeviceArena &arena);

/**
 * Phase 2 — write the data half of the runtime arena: standalone fields,
 * memset'd arena regions, sub-structure initializers, and SM-side device
 * pointers. The arena must already be committed (or attached); writes go
 * into arena.base() + sub-region offsets.
 *
 * `gm_heap_dev_base` is a device address; we only store it (never dereference).
 * Safe to run on a host arena that owns a host mirror of the runtime image —
 * the resulting buffer is rtMemcpy-ready.
 *
 * Returns the PTO2Runtime* that sits at layout.off_runtime within the arena.
 */
PTO2Runtime *runtime_init_data_from_layout(
    DeviceArena &arena, const PTO2RuntimeArenaLayout &layout, PTO2RuntimeMode mode, void *gm_heap_dev_base,
    uint64_t heap_size
);

/**
 * Destroy runtime sub-structures without releasing the arena buffer.
 */
void runtime_destroy(PTO2Runtime *rt, DeviceArena &arena);

/**
 * Slim config struct exported by orchestration .so via aicpu_orchestration_config().
 * Shared definition with pto_orchestration_api.h (same layout, guarded).
 */
#ifndef PTO2_ORCHESTRATION_CONFIG_DEFINED
#define PTO2_ORCHESTRATION_CONFIG_DEFINED
struct PTO2OrchestrationConfig {
    int expected_arg_count;
};
#endif
