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
#include "dist_engine/common/submit_pmu_types.h"
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
[[block_local]] static bool g_fdwic_joint_submit_seen;
#if DIST_TRACE_ENABLED
[[block_local]] static uint32_t g_fdwic_swimlane_level;
[[block_local]] static __gm__ FdwicSwimlaneHeader *g_fdwic_swimlane_header;
[[block_local]] static __gm__ FdwicSwimlaneCoreState *g_fdwic_swimlane_core;
[[block_local]] static __gm__ FdwicSwimlaneRecord *g_fdwic_swimlane_records;
[[block_local]] static uint32_t g_fdwic_swimlane_records_per_core;
[[block_local]] static FdwicAtomicPollBurst g_fdwic_atomic_poll_burst;
[[block_local]] static uint32_t g_fdwic_atomic_calls;
[[block_local]] static uint32_t g_fdwic_poll_calls;
[[block_local]] static uint32_t g_fdwic_poll_batch_records;
[[block_local]] static bool g_fdwic_atomic_counter_overflow;
#endif
#if PTO_FDWIC_PERF_CLOCK
[[block_local]] static __gm__ FdwicSwimlaneCoreState *g_fdwic_perf_clock_core;
[[block_local]] static uint64_t g_fdwic_perf_clock_first_submit;
[[block_local]] static uint64_t g_fdwic_perf_clock_last_submit;
[[block_local]] static uint32_t g_fdwic_perf_clock_expected_submits;
#if PTO_FDWIC_PERF_CLOCK_KERNEL
[[block_local]] static uint64_t g_fdwic_perf_clock_kernel_ticks;
[[block_local]] static uint32_t g_fdwic_perf_clock_kernel_calls;
[[block_local]] static uint32_t g_fdwic_perf_clock_kernel_status;
#endif
#endif
#if PTO_FDWIC_SUBMIT_PMU
[[block_local]] static __gm__ FdwicSubmitPmuCoreData *g_fdwic_submit_pmu_core;
[[block_local]] static __gm__ FdwicSubmitPmuPhaseCoreData *g_fdwic_submit_pmu_phase_core;
[[block_local]] static uint64_t g_fdwic_submit_pmu_reg_base;
[[block_local]] static uint64_t g_fdwic_submit_pmu_start_tick;
[[block_local]] static uint64_t g_fdwic_submit_pmu_end_tick;
[[block_local]] static uint64_t g_fdwic_submit_pmu_total_cycles;
[[block_local]] static uint32_t g_fdwic_submit_pmu_scalar_busy;
[[block_local]] static uint32_t g_fdwic_submit_pmu_icache_requests;
[[block_local]] static uint32_t g_fdwic_submit_pmu_icache_misses;
[[block_local]] static uint32_t g_fdwic_submit_pmu_shadow_requests;
[[block_local]] static uint32_t g_fdwic_submit_pmu_shadow_misses;
[[block_local]] static uint32_t g_fdwic_submit_pmu_expected_submits;
[[block_local]] static uint32_t g_fdwic_submit_pmu_status;
[[block_local]] static bool g_fdwic_submit_pmu_started;
[[block_local]] static bool g_fdwic_submit_pmu_stopped;
[[block_local]] static FdwicSubmitPmuPhaseAccumulator g_fdwic_submit_pmu_phase;
#endif
#define g_dist (*g_dist_ptr)
#elif defined(__CPU_SIM)
static DistGlobal g_dist_fallback;
static DistGlobal *g_dist_ptr = nullptr;
thread_local DistCore *g_self = nullptr;
thread_local bool g_fdwic_joint_submit_seen = false;
#if DIST_TRACE_ENABLED
thread_local uint32_t g_fdwic_swimlane_level = 0;
thread_local FdwicSwimlaneHeader *g_fdwic_swimlane_header = nullptr;
thread_local FdwicSwimlaneCoreState *g_fdwic_swimlane_core = nullptr;
thread_local FdwicSwimlaneRecord *g_fdwic_swimlane_records = nullptr;
thread_local uint32_t g_fdwic_swimlane_records_per_core = 0;
thread_local FdwicAtomicPollBurst g_fdwic_atomic_poll_burst = {};
thread_local uint32_t g_fdwic_atomic_calls = 0;
thread_local uint32_t g_fdwic_poll_calls = 0;
thread_local uint32_t g_fdwic_poll_batch_records = 0;
thread_local bool g_fdwic_atomic_counter_overflow = false;
#endif
#if PTO_FDWIC_PERF_CLOCK
thread_local FdwicSwimlaneCoreState *g_fdwic_perf_clock_core = nullptr;
thread_local uint64_t g_fdwic_perf_clock_first_submit = 0;
thread_local uint64_t g_fdwic_perf_clock_last_submit = 0;
thread_local uint32_t g_fdwic_perf_clock_expected_submits = 0;
#if PTO_FDWIC_PERF_CLOCK_KERNEL
thread_local uint64_t g_fdwic_perf_clock_kernel_ticks = 0;
thread_local uint32_t g_fdwic_perf_clock_kernel_calls = 0;
thread_local uint32_t g_fdwic_perf_clock_kernel_status = 0;
#endif
#endif
#if PTO_FDWIC_SUBMIT_PMU
thread_local FdwicSubmitPmuCoreData *g_fdwic_submit_pmu_core = nullptr;
thread_local FdwicSubmitPmuPhaseCoreData *g_fdwic_submit_pmu_phase_core = nullptr;
thread_local uint64_t g_fdwic_submit_pmu_reg_base = 0;
thread_local uint64_t g_fdwic_submit_pmu_start_tick = 0;
thread_local uint64_t g_fdwic_submit_pmu_end_tick = 0;
thread_local uint64_t g_fdwic_submit_pmu_total_cycles = 0;
thread_local uint32_t g_fdwic_submit_pmu_scalar_busy = 0;
thread_local uint32_t g_fdwic_submit_pmu_icache_requests = 0;
thread_local uint32_t g_fdwic_submit_pmu_icache_misses = 0;
thread_local uint32_t g_fdwic_submit_pmu_shadow_requests = 0;
thread_local uint32_t g_fdwic_submit_pmu_shadow_misses = 0;
thread_local uint32_t g_fdwic_submit_pmu_expected_submits = 0;
thread_local uint32_t g_fdwic_submit_pmu_status = 0;
thread_local bool g_fdwic_submit_pmu_started = false;
thread_local bool g_fdwic_submit_pmu_stopped = false;
thread_local FdwicSubmitPmuPhaseAccumulator g_fdwic_submit_pmu_phase = {};
#endif
#define g_dist (*g_dist_ptr)
#else
static DistGlobal g_dist_fallback;
static DistGlobal *g_dist_ptr = &g_dist_fallback;
thread_local DistCore *g_self = nullptr;
thread_local bool g_fdwic_joint_submit_seen = false;
#if DIST_TRACE_ENABLED
thread_local uint32_t g_fdwic_swimlane_level = 0;
thread_local FdwicSwimlaneHeader *g_fdwic_swimlane_header = nullptr;
thread_local FdwicSwimlaneCoreState *g_fdwic_swimlane_core = nullptr;
thread_local FdwicSwimlaneRecord *g_fdwic_swimlane_records = nullptr;
thread_local uint32_t g_fdwic_swimlane_records_per_core = 0;
thread_local FdwicAtomicPollBurst g_fdwic_atomic_poll_burst = {};
thread_local uint32_t g_fdwic_atomic_calls = 0;
thread_local uint32_t g_fdwic_poll_calls = 0;
thread_local uint32_t g_fdwic_poll_batch_records = 0;
thread_local bool g_fdwic_atomic_counter_overflow = false;
#endif
#if PTO_FDWIC_PERF_CLOCK
thread_local FdwicSwimlaneCoreState *g_fdwic_perf_clock_core = nullptr;
thread_local uint64_t g_fdwic_perf_clock_first_submit = 0;
thread_local uint64_t g_fdwic_perf_clock_last_submit = 0;
thread_local uint32_t g_fdwic_perf_clock_expected_submits = 0;
#if PTO_FDWIC_PERF_CLOCK_KERNEL
thread_local uint64_t g_fdwic_perf_clock_kernel_ticks = 0;
thread_local uint32_t g_fdwic_perf_clock_kernel_calls = 0;
thread_local uint32_t g_fdwic_perf_clock_kernel_status = 0;
#endif
#endif
#if PTO_FDWIC_SUBMIT_PMU
thread_local FdwicSubmitPmuCoreData *g_fdwic_submit_pmu_core = nullptr;
thread_local FdwicSubmitPmuPhaseCoreData *g_fdwic_submit_pmu_phase_core = nullptr;
thread_local uint64_t g_fdwic_submit_pmu_reg_base = 0;
thread_local uint64_t g_fdwic_submit_pmu_start_tick = 0;
thread_local uint64_t g_fdwic_submit_pmu_end_tick = 0;
thread_local uint64_t g_fdwic_submit_pmu_total_cycles = 0;
thread_local uint32_t g_fdwic_submit_pmu_scalar_busy = 0;
thread_local uint32_t g_fdwic_submit_pmu_icache_requests = 0;
thread_local uint32_t g_fdwic_submit_pmu_icache_misses = 0;
thread_local uint32_t g_fdwic_submit_pmu_shadow_requests = 0;
thread_local uint32_t g_fdwic_submit_pmu_shadow_misses = 0;
thread_local uint32_t g_fdwic_submit_pmu_expected_submits = 0;
thread_local uint32_t g_fdwic_submit_pmu_status = 0;
thread_local bool g_fdwic_submit_pmu_started = false;
thread_local bool g_fdwic_submit_pmu_stopped = false;
thread_local FdwicSubmitPmuPhaseAccumulator g_fdwic_submit_pmu_phase = {};
#endif
#define g_dist (*g_dist_ptr)
#endif
