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
PTO_DEVICE_FUNC inline TaskOutputTensors
rt_submit_task_compete_first(const MixedKernels &mixed_kernels, L0TaskArgs &args, BuildArgs &&build_args) {
    if (dist_is_fatal_query()) return TaskOutputTensors{};
    const DistCompeteFirstTicket ticket = dist_submit_compete_first_begin(nullptr, mixed_kernels);
    if (ticket.ready != 0) build_args(args);
    return dist_submit_compete_first_finish(nullptr, mixed_kernels, ticket, args);
}

template <typename BuildArgs>
PTO_DEVICE_FUNC inline TaskOutputTensors
rt_submit_aic_task_compete_first(int32_t kernel_id, L0TaskArgs &args, BuildArgs &&build_args) {
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    return rt_submit_task_compete_first(mk, args, static_cast<BuildArgs &&>(build_args));
}

template <typename BuildArgs>
PTO_DEVICE_FUNC inline TaskOutputTensors
rt_submit_aiv_task_compete_first(int32_t kernel_id, L0TaskArgs &args, BuildArgs &&build_args) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt_submit_task_compete_first(mk, args, static_cast<BuildArgs &&>(build_args));
}

#if PTO_FDWIC_SHARED_MAP
PTO_DEVICE_FUNC inline SharedTaskOutputs rt_deferred_task_outputs(int32_t task_id, uint32_t output_count) {
    SharedTaskOutputs outputs;
    if (!outputs.reset_deferred(task_id, output_count)) return fdwic_invalid_shared_outputs();
    return outputs;
}

/**
 * Generic cross-core compete-first helpers with deferred output references.
 *
 * The caller supplies the task's output arity as part of its schema.  Every
 * replay actor receives stable FdwicOutputRef values immediately; only the
 * unique builder constructs Args, validates that arity, materializes outputs,
 * and resolves predecessor references.  These helpers deliberately coexist
 * with the original TaskOutputTensors API, whose get_ref() contract remains
 * synchronous and materialized.
 */
template <typename BuildArgs>
PTO_DEVICE_FUNC inline SharedTaskOutputs
alloc_tensors_deferred_compete_first(uint32_t expected_output_count, L0TaskArgs &args, BuildArgs &&build_args) {
    if (dist_is_fatal_query()) return fdwic_invalid_shared_outputs();
    const DistCompeteFirstTicket ticket = dist_alloc_compete_first_begin(nullptr);
    if (ticket.ready != 0) build_args(args);
    if (!dist_alloc_deferred_compete_first_finish(nullptr, ticket, args, expected_output_count)) {
        return fdwic_invalid_shared_outputs();
    }
    return rt_deferred_task_outputs(ticket.task_id, expected_output_count);
}

template <typename BuildArgs>
PTO_DEVICE_FUNC inline SharedTaskOutputs rt_submit_task_deferred_compete_first(
    const MixedKernels &mixed_kernels, uint32_t expected_output_count, L0TaskArgs &args, BuildArgs &&build_args
) {
    if (dist_is_fatal_query()) return fdwic_invalid_shared_outputs();
    const DistCompeteFirstTicket ticket = dist_submit_compete_first_begin(nullptr, mixed_kernels);
    if (ticket.ready != 0) build_args(args);
    if (!dist_submit_deferred_compete_first_finish(nullptr, mixed_kernels, ticket, args, expected_output_count)) {
        return fdwic_invalid_shared_outputs();
    }
    return rt_deferred_task_outputs(ticket.task_id, expected_output_count);
}

template <typename BuildArgs>
PTO_DEVICE_FUNC inline SharedTaskOutputs rt_submit_aic_task_deferred_compete_first(
    int32_t kernel_id, uint32_t expected_output_count, L0TaskArgs &args, BuildArgs &&build_args
) {
    MixedKernels mixed;
    mixed.aic_kernel_id = kernel_id;
    return rt_submit_task_deferred_compete_first(
        mixed, expected_output_count, args, static_cast<BuildArgs &&>(build_args)
    );
}

template <typename BuildArgs>
PTO_DEVICE_FUNC inline SharedTaskOutputs rt_submit_aiv_task_deferred_compete_first(
    int32_t kernel_id, uint32_t expected_output_count, L0TaskArgs &args, BuildArgs &&build_args
) {
    MixedKernels mixed;
    mixed.aiv0_kernel_id = kernel_id;
    return rt_submit_task_deferred_compete_first(
        mixed, expected_output_count, args, static_cast<BuildArgs &&>(build_args)
    );
}

PTO_DEVICE_FUNC inline SharedTaskOutputs rt_shared_pa_outputs(int32_t task_id, DistSharedPaTaskKind kind) {
    if (task_id < 0) return fdwic_invalid_shared_outputs();
    SharedTaskOutputs outputs;
    outputs.reset(task_id);
    const uint32_t count = dist_shared_pa_output_count(kind);
    for (uint32_t slot = 0; slot < count; ++slot) {
        if (!outputs.add_output_ref(task_id, static_cast<int16_t>(slot))) {
            return fdwic_invalid_shared_outputs();
        }
    }
    return outputs;
}

#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
template <DistSharedPaTaskKind Kind>
PTO_DEVICE_FUNC inline __attribute__((always_inline)) SharedTaskOutputs rt_shared_pa_outputs_fixed(int32_t task_id) {
    static_assert(Kind != DistSharedPaTaskKind::Count, "shared PA output kind must be concrete");
    if (task_id < 0) return fdwic_invalid_shared_outputs();
    constexpr uint32_t output_count = Kind == DistSharedPaTaskKind::Alloc || Kind == DistSharedPaTaskKind::Sf ? 3U :
                                      Kind == DistSharedPaTaskKind::Qk || Kind == DistSharedPaTaskKind::Pv    ? 1U :
                                                                                                                0U;
    // Stable shared symbols are a replay-local task/count pair. Their full
    // descriptor publication remains winner-only in Finish, so constructing
    // this POD directly does not make any shared state visible early.
    SharedTaskOutputs outputs;
    outputs.producer_task_id = task_id;
    outputs.output_count = output_count;
    return outputs;
}
#endif

/**
 * Explicit PA-G1 shared wrappers.
 *
 * Stable output symbols are reconstructed for every replay actor. Begin
 * closes a nonwinner in the runtime TU; only the Claim winner executes the
 * eager argument callback and crosses the Finish ABI. A nonwinner therefore
 * cannot read a stale/reused L0TaskArgs object.
 */
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
template <CoreType ReplayRole, typename BuildArgs>
#else
template <typename BuildArgs>
#endif
PTO_DEVICE_FUNC inline SharedTaskOutputs shared_pa_alloc_tensors_compete_first(
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
    DistSharedPaReplayContext replay, int32_t expected_task_id, L0TaskArgs &args, BuildArgs &&build_args
#else
    DistSharedPaReplayContext replay, L0TaskArgs &args, BuildArgs &&build_args
#endif
) {
    // Shared Begin is the authoritative task-cap/protocol gate. The CCEC
    // dist_is_fatal_query() implementation is deliberately always false, so
    // calling it here only adds an external no-op to every replay actor.
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
    const DistCompeteFirstTicket ticket = dist_shared_pa_begin_ticket<ReplayRole, DistSharedPaTaskKind::Alloc>(
        replay, expected_task_id, INVALID_KERNEL_ID
    );
#elif PTO_FDWIC_SHARED_PA_UNITY
    const DistCompeteFirstTicket ticket = dist_shared_pa_begin_ticket(replay, DistSharedPaTaskKind::Alloc, nullptr);
#else
    const DistCompeteFirstTicket ticket = dist_shared_pa_alloc_begin(nullptr, replay);
#endif
    if (ticket.ready == 0) return fdwic_invalid_shared_outputs();
    if (ticket.won != 0) {
        build_args(args);
    }
    if (ticket.won != 0 && !dist_shared_pa_alloc_finish(nullptr, replay, ticket, &args)) {
        return fdwic_invalid_shared_outputs();
    }
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
    return rt_shared_pa_outputs_fixed<DistSharedPaTaskKind::Alloc>(ticket.task_id);
#else
    return rt_shared_pa_outputs(ticket.task_id, DistSharedPaTaskKind::Alloc);
#endif
}

#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
template <CoreType ReplayRole, DistSharedPaTaskKind Kind, typename BuildArgs>
#else
template <typename BuildArgs>
#endif
PTO_DEVICE_FUNC inline SharedTaskOutputs shared_pa_submit_task_compete_first(
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
    DistSharedPaReplayContext replay, int32_t expected_task_id, int32_t kernel_id, L0TaskArgs &args,
    BuildArgs &&build_args
#else
    DistSharedPaReplayContext replay, const MixedKernels &mixed, DistSharedPaTaskKind kind, L0TaskArgs &args,
    BuildArgs &&build_args
#endif
) {
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
    static_assert(
        Kind != DistSharedPaTaskKind::Alloc && Kind != DistSharedPaTaskKind::Count,
        "shared PA kernel submit requires QK/SF/PV/UP"
    );
    const DistCompeteFirstTicket ticket =
        dist_shared_pa_begin_ticket<ReplayRole, Kind>(replay, expected_task_id, kernel_id);
#elif PTO_FDWIC_SHARED_PA_UNITY
    const DistCompeteFirstTicket ticket = dist_shared_pa_begin_ticket(replay, kind, &mixed);
#else
    const DistCompeteFirstTicket ticket = dist_shared_pa_submit_begin(nullptr, replay, mixed, kind);
#endif
    if (ticket.ready == 0) return fdwic_invalid_shared_outputs();
    if (ticket.won != 0) {
        build_args(args);
    }
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
    if (ticket.won != 0) {
        MixedKernels mixed;
        if constexpr (Kind == DistSharedPaTaskKind::Qk || Kind == DistSharedPaTaskKind::Pv) {
            mixed.aic_kernel_id = kernel_id;
        } else {
            mixed.aiv0_kernel_id = kernel_id;
        }
        if (!dist_shared_pa_submit_finish(nullptr, replay, mixed, Kind, ticket, &args)) {
            return fdwic_invalid_shared_outputs();
        }
    }
    return rt_shared_pa_outputs_fixed<Kind>(ticket.task_id);
#else
    if (ticket.won != 0 && !dist_shared_pa_submit_finish(nullptr, replay, mixed, kind, ticket, &args)) {
        return fdwic_invalid_shared_outputs();
    }
    return rt_shared_pa_outputs(ticket.task_id, kind);
#endif
}

#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
template <CoreType ReplayRole, DistSharedPaTaskKind Kind, typename BuildArgs>
#else
template <typename BuildArgs>
#endif
PTO_DEVICE_FUNC inline SharedTaskOutputs shared_pa_submit_aic_compete_first(
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
    DistSharedPaReplayContext replay, int32_t expected_task_id, int32_t kernel_id, L0TaskArgs &args,
    BuildArgs &&build_args
#else
    DistSharedPaReplayContext replay, DistSharedPaTaskKind kind, int32_t kernel_id, L0TaskArgs &args,
    BuildArgs &&build_args
#endif
) {
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
    static_assert(
        Kind == DistSharedPaTaskKind::Qk || Kind == DistSharedPaTaskKind::Pv, "shared PA AIC wrapper requires QK or PV"
    );
    return shared_pa_submit_task_compete_first<ReplayRole, Kind>(
        replay, expected_task_id, kernel_id, args, static_cast<BuildArgs &&>(build_args)
    );
#else
    MixedKernels mixed;
    mixed.aic_kernel_id = kernel_id;
    return shared_pa_submit_task_compete_first(replay, mixed, kind, args, static_cast<BuildArgs &&>(build_args));
#endif
}

#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
template <CoreType ReplayRole, DistSharedPaTaskKind Kind, typename BuildArgs>
#else
template <typename BuildArgs>
#endif
PTO_DEVICE_FUNC inline SharedTaskOutputs shared_pa_submit_aiv_compete_first(
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
    DistSharedPaReplayContext replay, int32_t expected_task_id, int32_t kernel_id, L0TaskArgs &args,
    BuildArgs &&build_args
#else
    DistSharedPaReplayContext replay, DistSharedPaTaskKind kind, int32_t kernel_id, L0TaskArgs &args,
    BuildArgs &&build_args
#endif
) {
#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__)
    static_assert(
        Kind == DistSharedPaTaskKind::Sf || Kind == DistSharedPaTaskKind::Up, "shared PA AIV wrapper requires SF or UP"
    );
    return shared_pa_submit_task_compete_first<ReplayRole, Kind>(
        replay, expected_task_id, kernel_id, args, static_cast<BuildArgs &&>(build_args)
    );
#else
    MixedKernels mixed;
    mixed.aiv0_kernel_id = kernel_id;
    return shared_pa_submit_task_compete_first(replay, mixed, kind, args, static_cast<BuildArgs &&>(build_args));
#endif
}
#endif

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

PTO_DEVICE_FUNC inline void rt_perf_clock_expect_submits(uint32_t expected_submits) {
#if PTO_FDWIC_PERF_CLOCK
    dist_perf_clock_expect_submits(expected_submits);
#elif PTO_FDWIC_SUBMIT_PMU
    dist_submit_pmu_expect_submits(expected_submits);
#else
    (void)expected_submits;
#endif
}

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
