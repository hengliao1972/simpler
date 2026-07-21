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
    return false;
#else
    return fdwic_trace_is_fatal();
#endif
}

DIST_API_ATTR PTO_DEVICE_FUNC void
dist_report_fatal_msg(int32_t code, __gm__ const char *func, __gm__ const char *msg) {
    (void)code;
    (void)func;
    (void)msg;
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

#if PTO_FDWIC_PERF_CLOCK
DIST_API_ATTR PTO_DEVICE_FUNC void dist_perf_clock_expect_submits(uint32_t expected_submits) {
    fdwic_perf_clock_expect_submits(expected_submits);
}
#if PTO_FDWIC_PERF_CLOCK_KERNEL
// 只作为最终 ELF 的构建身份标记，不进入热路径。
DIST_API_ATTR PTO_DEVICE_FUNC uint32_t dist_perf_clock_kernel_profile_marker() { return kFdwicPerfClockKernelMode; }
#endif
#endif

#if PTO_FDWIC_SUBMIT_PMU
DIST_API_ATTR PTO_DEVICE_FUNC void dist_submit_pmu_expect_submits(uint32_t expected_submits) {
    fdwic_submit_pmu_expect_submits(expected_submits);
}
#endif

DIST_API_ATTR PTO_DEVICE_FUNC TaskOutputTensors dist_submit_dummy_impl(PTO2Runtime *, const L0TaskArgs &) {
    return TaskOutputTensors{};
}
