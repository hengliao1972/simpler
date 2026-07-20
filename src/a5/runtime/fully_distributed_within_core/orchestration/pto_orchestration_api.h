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
 * PTO Orchestration API - direct distributed-engine header.
 *
 * fully_distributed_within_core replays orchestration on AICore. The submit
 * helpers therefore call dist_engine_api.h symbols directly on both CPU sim and
 * CCEC onboard; this runtime has no submit function-pointer indirection.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "dist_engine/dist_engine_api.h"  // NOLINT(build/include_subdir)
#include "pto_runtime2_types.h"           // PTO2_ERROR_*
#include "pto_submit_types.h"             // MixedKernels, INVALID_KERNEL_ID
#include "pto_types.h"                    // Arg, TaskOutputTensors, TensorArgType
#include "task_args.h"                    // ChipStorageTaskArgs, Tensor
#include "tensor.h"                       // Tensor, TensorCreateInfo

struct PTO2Runtime;

PTO_DEVICE_FUNC inline TaskOutputTensors alloc_tensors(const L0TaskArgs &args) {
    if (dist_is_fatal_query()) return TaskOutputTensors{};
    return dist_alloc_tensors(nullptr, args);
}

PTO_DEVICE_FUNC inline TaskOutputTensors rt_submit_task(const MixedKernels &mixed_kernels, const L0TaskArgs &args) {
    if (dist_is_fatal_query()) return TaskOutputTensors{};
    return dist_submit_impl(nullptr, mixed_kernels, args);
}

PTO_DEVICE_FUNC inline TaskOutputTensors rt_submit_aic_task(int32_t kernel_id, const L0TaskArgs &args) {
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    return rt_submit_task(mk, args);
}

PTO_DEVICE_FUNC inline TaskOutputTensors rt_submit_aiv_task(int32_t kernel_id, const L0TaskArgs &args) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt_submit_task(mk, args);
}

/**
 * Compete-first eager wrappers.
 *
 * `args` is owned by the caller and is deliberately not reset here.  The
 * callback runs synchronously after EfDrain/Claim and before Finish; neither
 * the closure nor an internal thunk is retained by the runtime.  Device
 * orchestration must therefore give its lambda an AICore-callable operator
 * (for example, append `__aicore__` in a CCEC source).  The callback must not
 * submit another task or mutate the `MixedKernels` object used by the matching
 * Begin/Finish pair.
 */
template <typename BuildArgs>
PTO_DEVICE_FUNC inline TaskOutputTensors alloc_tensors_compete_first(L0TaskArgs &args, BuildArgs &&build_args) {
    if (dist_is_fatal_query()) return TaskOutputTensors{};
    const DistCompeteFirstTicket ticket = dist_alloc_compete_first_begin(nullptr);
    if (ticket.ready != 0) build_args(args);
    return dist_alloc_compete_first_finish(nullptr, ticket, args);
}

template <typename BuildArgs>
PTO_DEVICE_FUNC inline TaskOutputTensors rt_submit_task_compete_first(
    const MixedKernels &mixed_kernels, L0TaskArgs &args, BuildArgs &&build_args
) {
    if (dist_is_fatal_query()) return TaskOutputTensors{};
    const DistCompeteFirstTicket ticket = dist_submit_compete_first_begin(nullptr, mixed_kernels);
    if (ticket.ready != 0) build_args(args);
    return dist_submit_compete_first_finish(nullptr, mixed_kernels, ticket, args);
}

template <typename BuildArgs>
PTO_DEVICE_FUNC inline TaskOutputTensors rt_submit_aic_task_compete_first(
    int32_t kernel_id, L0TaskArgs &args, BuildArgs &&build_args
) {
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    return rt_submit_task_compete_first(mk, args, static_cast<BuildArgs &&>(build_args));
}

template <typename BuildArgs>
PTO_DEVICE_FUNC inline TaskOutputTensors rt_submit_aiv_task_compete_first(
    int32_t kernel_id, L0TaskArgs &args, BuildArgs &&build_args
) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt_submit_task_compete_first(mk, args, static_cast<BuildArgs &&>(build_args));
}

PTO_DEVICE_FUNC inline TaskOutputTensors rt_submit_dummy_task(const L0TaskArgs &args) {
    if (dist_is_fatal_query()) return TaskOutputTensors{};
    return dist_submit_dummy_impl(nullptr, args);
}

PTO_DEVICE_FUNC inline void rt_scope_begin(PTO2ScopeMode /*mode*/ = PTO2ScopeMode::AUTO) {
    if (dist_is_fatal_query()) return;
    dist_scope_begin_impl(nullptr);
}

PTO_DEVICE_FUNC inline void rt_scope_end() {
    if (dist_is_fatal_query()) return;
    dist_scope_end_impl(nullptr);
}

PTO_DEVICE_FUNC inline void rt_orchestration_done() { dist_orchestration_done_impl(nullptr); }

PTO_DEVICE_FUNC inline bool rt_is_fatal() { return dist_is_fatal_query(); }

#define rt_report_fatal(code, fmt, ...)                     \
    do {                                                    \
        dist_report_fatal_msg((code), __FUNCTION__, (fmt)); \
    } while (0)

#define LOG_ERROR(fmt, ...) dist_log_error_msg(__FUNCTION__, (fmt))
#define LOG_WARN(fmt, ...) dist_log_warn_msg(__FUNCTION__, (fmt))
#define LOG_DEBUG(fmt, ...) dist_log_debug_msg(__FUNCTION__, (fmt))
#define LOG_INFO_V0(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 0, (fmt))
#define LOG_INFO_V1(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 1, (fmt))
#define LOG_INFO_V2(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 2, (fmt))
#define LOG_INFO_V3(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 3, (fmt))
#define LOG_INFO_V4(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 4, (fmt))
#define LOG_INFO_V5(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 5, (fmt))
#define LOG_INFO_V6(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 6, (fmt))
#define LOG_INFO_V7(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 7, (fmt))
#define LOG_INFO_V8(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 8, (fmt))
#define LOG_INFO_V9(fmt, ...) dist_log_info_v_msg(__FUNCTION__, 9, (fmt))

template <typename T = uint64_t>
PTO_DEVICE_FUNC inline T get_tensor_data(const Tensor &tensor, uint32_t ndims, const uint32_t indices[]) {
    if (dist_is_fatal_query()) return from_u64<T>(0);
    return from_u64<T>(dist_get_tensor_data_impl(nullptr, tensor, ndims, indices));
}

template <typename T = uint64_t>
PTO_DEVICE_FUNC inline void set_tensor_data(const Tensor &tensor, uint32_t ndims, const uint32_t indices[], T value) {
    if (dist_is_fatal_query()) return;
    dist_set_tensor_data_impl(nullptr, tensor, ndims, indices, to_u64(value));
}

class PTO2ScopeGuard {
public:
    PTO_DEVICE_FUNC explicit PTO2ScopeGuard(
        PTO2ScopeMode mode = PTO2ScopeMode::AUTO, const char *file = __builtin_FILE(), int line = __builtin_LINE()
    ) {
        (void)mode;
        if (dist_is_fatal_query()) return;
        dist_scope_set_site_impl(file, line);
        dist_scope_begin_impl(nullptr);
    }

    PTO_DEVICE_FUNC ~PTO2ScopeGuard() {
        if (dist_is_fatal_query()) return;
        dist_scope_end_impl(nullptr);
    }
};

#define _PTO2_CONCATENATE_IMPL(x, y) x##y
#define _PTO2_CONCATENATE(x, y) _PTO2_CONCATENATE_IMPL(x, y)

#define PTO2_SCOPE_GUARD(...) \
    [[maybe_unused]] PTO2ScopeGuard _PTO2_CONCATENATE(scope_guard_, __COUNTER__) { __VA_ARGS__ }

#define PTO2_SCOPE(...) if (PTO2ScopeGuard _PTO2_CONCATENATE(scope_guard_, __COUNTER__){__VA_ARGS__}; true)

#ifndef PTO2_ORCHESTRATION_CONFIG_DEFINED
#define PTO2_ORCHESTRATION_CONFIG_DEFINED
struct PTO2OrchestrationConfig {
    int expected_arg_count;
};
#endif

#include "pto_arg_with_deps.h"
