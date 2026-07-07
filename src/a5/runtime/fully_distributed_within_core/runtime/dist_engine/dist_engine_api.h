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
 * inline wrappers in pto_orchestration_api.h. Those wrappers used to route
 * through rt->ops (a cross-library function-pointer table), which tied the
 * orchestration path to runtime symbol binding instead of normal linking.
 *
 * This header exposes the same set of engine operations as direct symbols that
 * the wrappers can call at compile time. Since orchestration is compiled into
 * the same translation-unit family as dist_engine on device (aicore_kernel.o),
 * a direct call resolves at link time and needs no runtime function-pointer
 * indirection. In sim, per-example orchestration is compiled into
 * libaicore_kernel.so and reaches these symbols from the AICore image.
 * On CCEC onboard builds, submit/core-main currently provide a minimal
 * direct-call submit path for ordinary single-core tasks. The full GM replay
 * engine is still blocked on CCEC GM atomic lowering and shared DistGlobal
 * ownership.
 *
 * PTO_DEVICE_FUNC expands to `__aicore__` under CCEC and to nothing on host /
 * sim / AICPU builds, so a single declaration serves both worlds.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "intrinsic.h"          // __gm__ (empty macro on host)
#include "pto_submit_types.h"   // MixedKernels
#include "pto_types.h"          // L0TaskArgs, TaskOutputTensors, PTO_DEVICE_FUNC (via data_type.h)
#include "tensor.h"             // Tensor

struct PTO2Runtime;

#if defined(__CCE_AICORE__)
struct alignas(64) DistTaskPayload {
    int32_t tensor_count;
    int32_t scalar_count;
    TensorArgType tags[MAX_TENSOR_ARGS];
    alignas(64) Tensor tensors[MAX_TENSOR_ARGS];
    alignas(64) uint64_t scalars[MAX_SCALAR_ARGS];
};
static_assert(sizeof(DistTaskPayload) % 64 == 0, "DistTaskPayload must not share cachelines");
static_assert(alignof(DistTaskPayload) == 64, "DistTaskPayload must be cacheline-aligned");
static_assert(offsetof(DistTaskPayload, tensors) % 64 == 0, "payload tensors must be cacheline-aligned");
static_assert(offsetof(DistTaskPayload, scalars) % 64 == 0, "payload scalars must be cacheline-aligned");

#endif

// Task submission and allocation. Host/sim definitions use the per-core g_self
// stashed by dist_core_main / thread_local sim. CCEC definitions cover only the
// minimal direct-call submit path until the onboard replay path is enabled.
PTO_DEVICE_FUNC TaskOutputTensors dist_submit_impl(PTO2Runtime *rt, const MixedKernels &mixed, const L0TaskArgs &args);
PTO_DEVICE_FUNC TaskOutputTensors dist_alloc_tensors(PTO2Runtime *rt, const L0TaskArgs &args);

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
PTO_DEVICE_FUNC uint64_t dist_get_tensor_data_impl(
    PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[]
);
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
