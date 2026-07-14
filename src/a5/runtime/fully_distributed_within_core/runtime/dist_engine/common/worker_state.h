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
#include "dist_engine/common/swimlane_types.h"

// g_dist / g_self storage. The AICPU build owns the BSS DistGlobal and
// initializes it in dist_engine_register. AICore builds attach to the published
// Runtime::dist.shared_addr at dist_core_main entry: CCEC uses GM pointers,
// while CPU sim uses an ordinary process pointer into the AICPU DSO's BSS.
// The `g_dist` macro keeps hot-path code identical.
#if defined(__CCE_AICORE__)
[[block_local]] static __gm__ DistGlobal *g_dist_ptr;
[[block_local]] static __gm__ DistCore *g_self;
[[block_local]] static __gm__ Runtime *g_ccec_runtime;
[[block_local]] static int32_t g_ccec_core_idx;
[[block_local]] static int32_t g_ccec_core_type;
[[block_local]] static int32_t g_ccec_aic_count;
[[block_local]] static int32_t g_ccec_aiv_count;
[[block_local]] static int32_t g_ccec_ordinal;
[[block_local]] static bool g_ccec_valid_worker;
[[block_local]] static bool g_fdwic_swimlane_enabled;
[[block_local]] static __gm__ FdwicSwimlaneHeader *g_fdwic_swimlane_header;
[[block_local]] static __gm__ FdwicSwimlaneCoreState *g_fdwic_swimlane_core;
[[block_local]] static __gm__ FdwicSwimlaneRecord *g_fdwic_swimlane_records;
[[block_local]] static uint32_t g_fdwic_swimlane_records_per_core;
#define g_dist (*g_dist_ptr)
#elif defined(__CPU_SIM)
static DistGlobal g_dist_fallback;
static DistGlobal *g_dist_ptr = nullptr;
thread_local DistCore *g_self = nullptr;
thread_local bool g_fdwic_swimlane_enabled = false;
thread_local FdwicSwimlaneHeader *g_fdwic_swimlane_header = nullptr;
thread_local FdwicSwimlaneCoreState *g_fdwic_swimlane_core = nullptr;
thread_local FdwicSwimlaneRecord *g_fdwic_swimlane_records = nullptr;
thread_local uint32_t g_fdwic_swimlane_records_per_core = 0;
#define g_dist (*g_dist_ptr)
#else
static DistGlobal g_dist_fallback;
static DistGlobal *g_dist_ptr = &g_dist_fallback;
thread_local DistCore *g_self = nullptr;
thread_local bool g_fdwic_swimlane_enabled = false;
thread_local FdwicSwimlaneHeader *g_fdwic_swimlane_header = nullptr;
thread_local FdwicSwimlaneCoreState *g_fdwic_swimlane_core = nullptr;
thread_local FdwicSwimlaneRecord *g_fdwic_swimlane_records = nullptr;
thread_local uint32_t g_fdwic_swimlane_records_per_core = 0;
#define g_dist (*g_dist_ptr)
#endif
