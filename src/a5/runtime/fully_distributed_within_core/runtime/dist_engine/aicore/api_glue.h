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

#include "dist_engine/common/runtime_state.h"
#include "dist_engine/dist_engine_api.h"

DIST_API_ATTR PTO_DEVICE_FUNC bool dist_is_fatal_query() {
#if defined(__CCE_AICORE__)
    // The orchestration wrapper is executed by every replay actor. A global
    // atomic load here would serialize all 96 actors before every Submit.
    // Shared PA instead checks fatal once on the Claim winner and inside real
    // blocking loops; deterministic orchestration shape gates are evaluated
    // by every actor locally.
    return false;
#else
    return fdwic_trace_is_fatal();
#endif
}

DIST_API_ATTR PTO_DEVICE_FUNC void
dist_report_fatal_msg(int32_t code, __gm__ const char *func, __gm__ const char *msg) {
    (void)func;
    (void)msg;
#if PTO_FDWIC_SHARED_MAP
    // Phase-1 shared PA has fail-closed shape and protocol gates in
    // orchestration. The device logging sink is intentionally separate, but
    // the stable runtime error code and global fatal bit must still publish.
    set_fatal_code(code);
    if (g_self != nullptr) g_self->local_index = kFlagCap;
#else
    (void)code;
#endif
}

DIST_API_ATTR PTO_DEVICE_FUNC void dist_log_error_msg(__gm__ const char *func, __gm__ const char *msg) {
    (void)func;
    (void)msg;
}

DIST_API_ATTR PTO_DEVICE_FUNC void dist_log_warn_msg(__gm__ const char *, __gm__ const char *) {}
DIST_API_ATTR PTO_DEVICE_FUNC void dist_log_debug_msg(__gm__ const char *, __gm__ const char *) {}
DIST_API_ATTR PTO_DEVICE_FUNC void dist_log_info_v_msg(__gm__ const char *, int, __gm__ const char *) {}

DIST_API_ATTR PTO_DEVICE_FUNC void dist_scope_begin_impl(PTO2Runtime *) {}
DIST_API_ATTR PTO_DEVICE_FUNC void dist_scope_end_impl(PTO2Runtime *) {}
DIST_API_ATTR PTO_DEVICE_FUNC void dist_orchestration_done_impl(PTO2Runtime *) {}
DIST_API_ATTR PTO_DEVICE_FUNC void dist_scope_set_site_impl(const char *, int) {}

#if PTO_FDWIC_SHARED_MAP
DIST_API_ATTR PTO_DEVICE_FUNC DistSharedPaReplayContext
dist_shared_pa_replay_context() {
    DistSharedPaReplayContext replay{};
    __gm__ DistCore *self = g_self;
    if (self == nullptr) {
        return replay;
    }
    const CoreType role = self->role;
    const int32_t core_idx = self->core_idx;
    const int32_t block_id = self->block_id;
    const int32_t lane = self->lane;
    if ((role != CoreType::AIC && role != CoreType::AIV) ||
        core_idx < 0 || block_id < 0 || lane < 0) {
        return replay;
    }
    // Snapshot the attach-owned identity once.  The 1,280 Claim calls consume
    // these scalar fields from caller state instead of reloading DistCore GM.
    replay.role_ = role;
    replay.block_id_ = block_id;
    return replay;
}
#endif

#if PTO_FDWIC_PERF_CLOCK
DIST_API_ATTR PTO_DEVICE_FUNC void dist_perf_clock_expect_submits(uint32_t expected_submits) {
    fdwic_perf_clock_expect_submits(expected_submits);
}
#if PTO_FDWIC_PERF_CLOCK_KERNEL
// This is only a final-ELF identity marker and never enters a hot path.
DIST_API_ATTR PTO_DEVICE_FUNC uint32_t dist_perf_clock_kernel_profile_marker() {
    return PTO_FDWIC_SCHEDULER_MODE == 0 ? kFdwicPerfClockKernelMode : kFdwicPerfClockKernelCrossCoreE2eMode;
}
#endif
#endif

#if PTO_FDWIC_SUBMIT_PMU
DIST_API_ATTR PTO_DEVICE_FUNC void dist_submit_pmu_expect_submits(uint32_t expected_submits) {
    fdwic_submit_pmu_expect_submits(expected_submits);
}
#endif

DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors dist_submit_dummy_impl(PTO2Runtime *, const L0TaskArgs &) {
#if PTO_FDWIC_SHARED_MAP
    set_fatal_code(PTO2_ERROR_TENSORMAP_PROTOCOL);
    if (g_self != nullptr) g_self->local_index = kFlagCap;
#endif
    return TaskOutputTensors{};
}
