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
 * dist_engine device-callable API.
 *
 * The distributed engine (dist_engine.cpp) provides the SPMD replay + claim-race
 * submit path used by fully_distributed_within_core. Orchestration source (per
 * example, under examples/.../kernels/orchestration/) drives that engine
 * through the
 * inline wrappers in pto_orchestration_api.h.
 *
 * This header exposes the same set of engine operations as direct symbols that
 * the wrappers can call at compile time. Since orchestration is compiled into
 * the same translation-unit family as dist_engine on device (aicore_kernel.o),
 * a direct call resolves at link time and needs no runtime function-pointer
 * indirection. In sim, per-example orchestration is compiled into
 * libaicore_kernel.so and reaches these symbols from the AICore image.
 * On CCEC onboard builds, orchestration is still replayed through a direct
 * AICore entry, but submit/alloc share the same payload materialization,
 * producer-map, fan-in, and output registration stages as the sim path. The
 * remaining CCEC-specific surface is the claim/execution/completion backend.
 *
 * PTO_DEVICE_FUNC expands to `__aicore__` under CCEC and to nothing on host /
 * sim / AICPU builds, so a single declaration serves both worlds.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "intrinsic.h"         // __gm__ (empty macro on host)
#include "pto_submit_types.h"  // MixedKernels
#include "pto_types.h"         // L0TaskArgs, TaskOutputTensors, PTO_DEVICE_FUNC (via data_type.h)
#include "tensor.h"            // Tensor
#include "dist_engine/common/state.h"

struct PTO2Runtime;

/**
 * Fixed-size hand-off between the two synchronous halves of a compete-first
 * submit.  Begin never retains an argument-builder closure: orchestration
 * receives this POD, builds its caller-owned L0TaskArgs, then immediately
 * passes both objects to Finish.
 *
 * Byte fields are intentional.  Besides keeping the cross-TU ABI explicit,
 * they avoid relying on the target compiler's bool layout.  `reserved` must
 * remain zero and is checked by Finish together with `kind` and the per-core
 * task sequence.
 */
enum class DistCompeteFirstKind : uint8_t {
    Kernel = 0,
    Alloc = 1,
};

struct DistCompeteFirstTicket {
    uint64_t submit_begin;
    int32_t task_id;
    int32_t kernel_id;
    int32_t joint_block;
    int32_t joint_count;
    uint8_t won;
    uint8_t joint;
    uint8_t joint_init;
    uint8_t claim_attempted;
    uint8_t ready;
    uint8_t kind;
    uint16_t reserved;
};

static_assert(sizeof(DistCompeteFirstTicket) == 32, "compete-first ticket ABI must remain 32 bytes");
static_assert(offsetof(DistCompeteFirstTicket, submit_begin) == 0, "compete-first timestamp ABI mismatch");
static_assert(offsetof(DistCompeteFirstTicket, task_id) == 8, "compete-first task ABI mismatch");
static_assert(offsetof(DistCompeteFirstTicket, won) == 24, "compete-first state ABI mismatch");
static_assert(offsetof(DistCompeteFirstTicket, reserved) == 30, "compete-first reserved ABI mismatch");

// Task submission and allocation. Host/sim definitions use the per-core g_self
// stashed by dist_core_main / thread_local sim. CCEC definitions use the same
// materialize/map/fanin/register stages, then dispatch through the current
// direct AICore execution backend.
PTO_DEVICE_FUNC TaskOutputTensors dist_submit_impl(PTO2Runtime *rt, const MixedKernels &mixed, const L0TaskArgs &args);
PTO_DEVICE_FUNC TaskOutputTensors dist_alloc_tensors(PTO2Runtime *rt, const L0TaskArgs &args);

// Explicit compete-first eager path. Begin performs argument-independent
// progress and Claim; Finish consumes the synchronously built args. The same
// MixedKernels object must remain unchanged until Finish returns. The old
// one-shot APIs above remain available unchanged for all existing examples.
PTO_DEVICE_FUNC DistCompeteFirstTicket
dist_submit_compete_first_begin(PTO2Runtime *rt, const MixedKernels &mixed);
PTO_DEVICE_FUNC TaskOutputTensors dist_submit_compete_first_finish(
    PTO2Runtime *rt, const MixedKernels &mixed, const DistCompeteFirstTicket &ticket, const L0TaskArgs &args
);
PTO_DEVICE_FUNC DistCompeteFirstTicket dist_alloc_compete_first_begin(PTO2Runtime *rt);
PTO_DEVICE_FUNC TaskOutputTensors dist_alloc_compete_first_finish(
    PTO2Runtime *rt, const DistCompeteFirstTicket &ticket, const L0TaskArgs &args
);

// perf-clock 专用构建由具体 orchestration 显式声明本核应重放的 Submit
// 总数。普通构建中该接口编译为空操作，不改变公开 submit ABI。
PTO_DEVICE_FUNC void dist_perf_clock_expect_submits(uint32_t expected_submits);

// Fatal-state helpers. dist_engine.cpp already exposes fatal_set() /
// set_fatal(); these are the CCEC-safe wrappers orchestration reaches.
PTO_DEVICE_FUNC bool dist_is_fatal_query();
PTO_DEVICE_FUNC void dist_report_fatal_msg(int32_t code, __gm__ const char *func, __gm__ const char *msg);

// Log sinks. On host / sim these forward to unified_log_host (varargs); on
// AICore the current implementation is a no-op stub — a real unified_log_device
// pipeline (GM log-ring + AICPU flush) is a follow-up. Signatures are kept
// simple (const-string msg only) to avoid CCEC va_list constraints.
// `func` / `msg` are declared __gm__ because CCEC places string literals
// (__FUNCTION__, "..." format strings expanded at call sites) in GM; the
// qualifier is empty under host / sim builds so callers there compile
// unchanged.
PTO_DEVICE_FUNC void dist_log_error_msg(__gm__ const char *func, __gm__ const char *msg);
PTO_DEVICE_FUNC void dist_log_warn_msg(__gm__ const char *func, __gm__ const char *msg);
PTO_DEVICE_FUNC void dist_log_debug_msg(__gm__ const char *func, __gm__ const char *msg);
PTO_DEVICE_FUNC void dist_log_info_v_msg(__gm__ const char *func, int v, __gm__ const char *msg);

// Cross-layer tensor data access. Host/sim waits for producers through the
// engine; CCEC currently performs direct GM scalar access only.
PTO_DEVICE_FUNC uint64_t
dist_get_tensor_data_impl(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[]);
PTO_DEVICE_FUNC void dist_set_tensor_data_impl(
    PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
);

// Scope guard hooks. Currently no-op inside dist_engine (per-core replay does
// not need scope batching); kept for wrapper-level API compatibility.
PTO_DEVICE_FUNC void dist_scope_begin_impl(PTO2Runtime *rt);
PTO_DEVICE_FUNC void dist_scope_end_impl(PTO2Runtime *rt);
PTO_DEVICE_FUNC void dist_orchestration_done_impl(PTO2Runtime *rt);
PTO_DEVICE_FUNC void dist_scope_set_site_impl(const char *file, int line);

// Dependency-only task submit (kernel-less).
PTO_DEVICE_FUNC TaskOutputTensors dist_submit_dummy_impl(PTO2Runtime *rt, const L0TaskArgs &args);
