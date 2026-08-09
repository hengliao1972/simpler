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

#if defined(__CPU_SIM)
#define DIST_SIM_HOST_CLOCK 1
#else
#define DIST_SIM_HOST_CLOCK 0
#endif

#ifndef PTO2_PROFILING
#define PTO2_PROFILING 1
#endif

#ifndef PTO_FDWIC_PERF_CLOCK
#define PTO_FDWIC_PERF_CLOCK 0
#endif

// perf-clock-kernel is a separate low-volume diagnostic ELF. It preserves the
// first/last Submit clocks and accumulates linked-kernel time and call count in
// the same per-core window.
#ifndef PTO_FDWIC_PERF_CLOCK_KERNEL
#define PTO_FDWIC_PERF_CLOCK_KERNEL 0
#endif

#ifndef PTO_FDWIC_SUBMIT_PMU
#define PTO_FDWIC_SUBMIT_PMU 0
#endif

// Each submit-PMU diagnostic ELF may compile exactly one local phase. Zero is
// the whole-window "none" phase; submit_pmu_types.h defines the other IDs.
#ifndef PTO_FDWIC_SUBMIT_PMU_PHASE_ID
#define PTO_FDWIC_SUBMIT_PMU_PHASE_ID 0
#endif

#ifndef PTO_FDWIC_TRACE_ENABLED
#define PTO_FDWIC_TRACE_ENABLED PTO2_PROFILING
#endif

#ifndef PTO_FDWIC_SHARED_PA_UNITY
#define PTO_FDWIC_SHARED_PA_UNITY 0
#endif

#ifndef PTO_FDWIC_SHARED_PA_BUILD_ROLE
#define PTO_FDWIC_SHARED_PA_BUILD_ROLE -1
#endif

#if PTO_FDWIC_SHARED_PA_UNITY && !PTO_FDWIC_SHARED_MAP
#error "PTO_FDWIC_SHARED_PA_UNITY requires PTO_FDWIC_SHARED_MAP=1"
#endif

#if PTO_FDWIC_SHARED_PA_UNITY && defined(__CCE_AICORE__) && PTO_FDWIC_SHARED_PA_BUILD_ROLE != 0 && \
    PTO_FDWIC_SHARED_PA_BUILD_ROLE != 1
#error "CCEC shared PA unity requires an explicit AIC(0) or AIV(1) build role"
#endif

// A SIMT scheduler must retain an AIV-compiled dist body for its AIV entry.
// Unlike the historical same-core image, the linker cannot fold equal-named
// AIC/AIV weak functions into one body. Each role owns a distinct
// [[block_local]] worker state, so both the orchestration entry and every
// runtime API it calls must resolve to the matching role object. Otherwise an
// AIV entry can call the retained AIC weak Submit and read g_self/g_dist_ptr
// that its AIV dist_core_main never attached.
//
// Operator sources and wrappers keep the generic API names. Only a CCEC SIMT
// mixed ELF rewrites the final symbols here; CPU and non-SIMT ABIs are intact.
#if PTO_FDWIC_SCHEDULER_MODE == 3 && defined(__CCE_AICORE__)
#if PTO_FDWIC_SHARED_PA_BUILD_ROLE == 0
#define aicpu_orchestration_entry aicpu_orchestration_entry_aic
#define dist_submit_impl dist_submit_impl_aic
#define dist_alloc_tensors dist_alloc_tensors_aic
#define dist_submit_compete_first_begin dist_submit_compete_first_begin_aic
#define dist_submit_compete_first_finish dist_submit_compete_first_finish_aic
#define dist_alloc_compete_first_begin dist_alloc_compete_first_begin_aic
#define dist_alloc_compete_first_finish dist_alloc_compete_first_finish_aic
#define dist_shared_pa_replay_context dist_shared_pa_replay_context_aic
#define dist_shared_pa_submit_begin dist_shared_pa_submit_begin_aic
#define dist_shared_pa_submit_finish dist_shared_pa_submit_finish_aic
#define dist_shared_pa_alloc_begin dist_shared_pa_alloc_begin_aic
#define dist_shared_pa_alloc_finish dist_shared_pa_alloc_finish_aic
#define dist_perf_clock_expect_submits dist_perf_clock_expect_submits_aic
#define dist_submit_pmu_expect_submits dist_submit_pmu_expect_submits_aic
#define dist_is_fatal_query dist_is_fatal_query_aic
#define dist_report_fatal_msg dist_report_fatal_msg_aic
#define dist_log_error_msg dist_log_error_msg_aic
#define dist_log_warn_msg dist_log_warn_msg_aic
#define dist_log_debug_msg dist_log_debug_msg_aic
#define dist_log_info_v_msg dist_log_info_v_msg_aic
#define dist_get_tensor_data_impl dist_get_tensor_data_impl_aic
#define dist_set_tensor_data_impl dist_set_tensor_data_impl_aic
#define dist_scope_begin_impl dist_scope_begin_impl_aic
#define dist_scope_end_impl dist_scope_end_impl_aic
#define dist_orchestration_done_impl dist_orchestration_done_impl_aic
#define dist_scope_set_site_impl dist_scope_set_site_impl_aic
#define dist_submit_dummy_impl dist_submit_dummy_impl_aic
#elif PTO_FDWIC_SHARED_PA_BUILD_ROLE == 1
#define aicpu_orchestration_entry aicpu_orchestration_entry_aiv
#define dist_submit_impl dist_submit_impl_aiv
#define dist_alloc_tensors dist_alloc_tensors_aiv
#define dist_submit_compete_first_begin dist_submit_compete_first_begin_aiv
#define dist_submit_compete_first_finish dist_submit_compete_first_finish_aiv
#define dist_alloc_compete_first_begin dist_alloc_compete_first_begin_aiv
#define dist_alloc_compete_first_finish dist_alloc_compete_first_finish_aiv
#define dist_shared_pa_replay_context dist_shared_pa_replay_context_aiv
#define dist_shared_pa_submit_begin dist_shared_pa_submit_begin_aiv
#define dist_shared_pa_submit_finish dist_shared_pa_submit_finish_aiv
#define dist_shared_pa_alloc_begin dist_shared_pa_alloc_begin_aiv
#define dist_shared_pa_alloc_finish dist_shared_pa_alloc_finish_aiv
#define dist_perf_clock_expect_submits dist_perf_clock_expect_submits_aiv
#define dist_submit_pmu_expect_submits dist_submit_pmu_expect_submits_aiv
#define dist_is_fatal_query dist_is_fatal_query_aiv
#define dist_report_fatal_msg dist_report_fatal_msg_aiv
#define dist_log_error_msg dist_log_error_msg_aiv
#define dist_log_warn_msg dist_log_warn_msg_aiv
#define dist_log_debug_msg dist_log_debug_msg_aiv
#define dist_log_info_v_msg dist_log_info_v_msg_aiv
#define dist_get_tensor_data_impl dist_get_tensor_data_impl_aiv
#define dist_set_tensor_data_impl dist_set_tensor_data_impl_aiv
#define dist_scope_begin_impl dist_scope_begin_impl_aiv
#define dist_scope_end_impl dist_scope_end_impl_aiv
#define dist_orchestration_done_impl dist_orchestration_done_impl_aiv
#define dist_scope_set_site_impl dist_scope_set_site_impl_aiv
#define dist_submit_dummy_impl dist_submit_dummy_impl_aiv
#else
#error "CCEC SIMT scheduler requires an explicit AIC(0) or AIV(1) build role"
#endif
#endif

#if PTO_FDWIC_PERF_CLOCK && PTO_FDWIC_TRACE_ENABLED
#error "PTO_FDWIC_PERF_CLOCK requires PTO_FDWIC_TRACE_ENABLED=0"
#endif

#if PTO_FDWIC_PERF_CLOCK_KERNEL && !PTO_FDWIC_PERF_CLOCK
#error "PTO_FDWIC_PERF_CLOCK_KERNEL requires PTO_FDWIC_PERF_CLOCK=1"
#endif

#if PTO_FDWIC_SUBMIT_PMU && PTO_FDWIC_TRACE_ENABLED
#error "PTO_FDWIC_SUBMIT_PMU requires PTO_FDWIC_TRACE_ENABLED=0"
#endif

#if PTO_FDWIC_SUBMIT_PMU && PTO_FDWIC_PERF_CLOCK
#error "PTO_FDWIC_SUBMIT_PMU and PTO_FDWIC_PERF_CLOCK are mutually exclusive"
#endif

#if !PTO_FDWIC_SUBMIT_PMU && PTO_FDWIC_SUBMIT_PMU_PHASE_ID != 0
#error "PTO_FDWIC_SUBMIT_PMU_PHASE_ID requires PTO_FDWIC_SUBMIT_PMU=1"
#endif

#if PTO_FDWIC_TRACE_ENABLED
#define DIST_TRACE_ENABLED 1
#else
#define DIST_TRACE_ENABLED 0
#endif

#if defined(__CCE_AICORE__)
#define DIST_API_ATTR __attribute__((weak))
#else
#define DIST_API_ATTR
#endif
