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
 * fully_distributed_within_core engine — public wiring entry.
 *
 * The distributed runtime moves orchestration + scheduling + execution onto the
 * AI cores in SPMD fashion (see docs/fully_distributed_within_core.md). The
 * per-core submit engine is linked into the AICore image; the AICPU side only
 * initializes the shared DistGlobal state before waking workers.
 *
 * The AICPU "stub" thread does arena setup, then calls dist_engine_register()
 * once before waking AICore workers. Each AICore worker invokes the
 * dist_core_main entry linked into its own image, directly calls the
 * orchestration entry, and executes the tasks it wins.
 */

#pragma once

#include <cstddef>
#include <cstdint>

struct PTO2Runtime;
struct L2TaskArgs;
class Runtime;

// Arena footprint of the distributed engine's private DistGlobal. Host runtime
// does not link dist_engine.cpp, so these are header constants; dist_engine.cpp
// static_asserts the private type against them.
inline constexpr size_t kDistEngineGlobalStateSize = 0x42000000;
inline constexpr size_t kDistEngineGlobalStateAlign = 64;

inline constexpr size_t dist_engine_global_state_size() { return kDistEngineGlobalStateSize; }

inline constexpr size_t dist_engine_global_state_align() { return kDistEngineGlobalStateAlign; }

/**
 * Wire the distributed engine for one run.
 *
 * Resets the global claim cursors + completion-flag ring, (re)acquires the GM
 * output heap, and stores the args / PTO2Runtime used by on-core orchestration
 * replay. Must be called once on the AICPU setup thread before waking workers
 * through their per-core handshake flags.
 *
 * AICore workers call dist_engine_api.h symbols directly on sim and onboard.
 */
int32_t dist_engine_register(PTO2Runtime *rt, const L2TaskArgs *orch_args, int num_workers, Runtime *runtime);

// AICPU 在所有 worker 完成后读取 AICore 发布的 fatal cache line，并将
// 首个运行时错误转换成公共负状态码。正常运行返回 0。
int32_t dist_engine_runtime_status(PTO2Runtime *rt);
