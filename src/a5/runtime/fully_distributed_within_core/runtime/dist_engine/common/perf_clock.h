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
#include "dist_engine/common/worker_state.h"
#include "dist_engine/aicore/primitive.h"

namespace {

#if PTO_FDWIC_PERF_CLOCK

PTO_DEVICE_FUNC inline void fdwic_perf_clock_attach(__gm__ Runtime *runtime, __gm__ DistCore *self) {
    g_fdwic_perf_clock_core = nullptr;
    g_fdwic_perf_clock_first_submit = 0;
    g_fdwic_perf_clock_last_submit = 0;
    g_fdwic_perf_clock_expected_submits = 0;
#if PTO_FDWIC_PERF_CLOCK_KERNEL
    g_fdwic_perf_clock_kernel_ticks = 0;
    g_fdwic_perf_clock_kernel_calls = 0;
    g_fdwic_perf_clock_kernel_status = 0;
#endif
    if (runtime == nullptr || self == nullptr) return;
#if defined(__CCE_AICORE__)
    dist_aicore_invalidate_region(const_cast<__gm__ uint64_t *>(&runtime->dist.swimlane_base), 64);
#endif
    const uint64_t base = runtime->dist.swimlane_base;
    if (base == 0 || self->core_idx < 0 || self->core_idx >= runtime->dist.num_workers) return;
    __gm__ auto *header = reinterpret_cast<__gm__ FdwicSwimlaneHeader *>(base);
    g_fdwic_perf_clock_core = &header->cores[self->core_idx];
}

PTO_DEVICE_FUNC inline void fdwic_perf_clock_expect_submits(uint32_t expected_submits) {
    // Orchestration calls this once before its first Submit. The host closes
    // DistCore::local_index against expected/final_seen without repairing it.
    g_fdwic_perf_clock_expected_submits = expected_submits;
}

PTO_DEVICE_FUNC inline void fdwic_perf_clock_worker_begin() {
#if PTO_FDWIC_SCHEDULER_MODE != 0
    // Cross-core work includes the startup barrier, dynamic Build/Execute,
    // and FinalDrain. Starting before the startup increment gives the
    // dedicated SIMT builder the same boundary even though it replays no Submit.
    g_fdwic_perf_clock_first_submit = get_sys_cnt_aicore();
#endif
}

PTO_DEVICE_FUNC inline void fdwic_perf_clock_worker_end() {
#if PTO_FDWIC_SCHEDULER_MODE != 0
    // core_main calls this immediately after dist_submit_drain_to_completion().
    // Later trace/observer publication does not belong to the business window.
    g_fdwic_perf_clock_last_submit = get_sys_cnt_aicore();
#endif
}

PTO_DEVICE_FUNC inline void fdwic_perf_clock_submit_begin(int32_t task_id) {
#if PTO_FDWIC_SCHEDULER_MODE == 0
    // task_id is already the strictly increasing per-core Submit ordinal, so
    // reuse it instead of adding one block-local increment/store per Submit.
    if (task_id == 0) {
        g_fdwic_perf_clock_first_submit = get_sys_cnt_aicore();
    }
#else
    (void)task_id;
#endif
}

PTO_DEVICE_FUNC inline void fdwic_perf_clock_submit_end(int32_t task_id) {
#if PTO_FDWIC_SCHEDULER_MODE == 0
    if (task_id >= 0 && static_cast<uint32_t>(task_id + 1) == g_fdwic_perf_clock_expected_submits &&
        g_fdwic_perf_clock_expected_submits != 0) {
        g_fdwic_perf_clock_last_submit = get_sys_cnt_aicore();
    }
#else
    (void)task_id;
#endif
}

#if PTO_FDWIC_PERF_CLOCK_KERNEL

constexpr uint32_t kFdwicPerfClockKernelTickOrderError = 1U << 0;
constexpr uint32_t kFdwicPerfClockKernelTickOverflow = 1U << 1;
constexpr uint32_t kFdwicPerfClockKernelCallOverflow = 1U << 2;

// Same-core opens the Kernel subwindow from its first through last Submit.
// Cross-core covers startup through FinalDrain, including tail Kernels.
PTO_DEVICE_FUNC inline uint64_t fdwic_perf_clock_kernel_begin() {
    if (g_fdwic_perf_clock_first_submit == 0 || g_fdwic_perf_clock_last_submit != 0 ||
        g_fdwic_perf_clock_kernel_status != 0) {
        return 0;
    }
#if PTO_FDWIC_SCHEDULER_MODE == 0
    if (g_fdwic_perf_clock_expected_submits == 0) return 0;
#endif
    return get_sys_cnt_aicore();
}

PTO_DEVICE_FUNC inline void fdwic_perf_clock_kernel_end(uint64_t begin_tick) {
    if (begin_tick == 0) return;
    const uint64_t end_tick = get_sys_cnt_aicore();
    if (end_tick < begin_tick) {
        g_fdwic_perf_clock_kernel_status |= kFdwicPerfClockKernelTickOrderError;
        return;
    }
    const uint64_t delta = end_tick - begin_tick;
    if (g_fdwic_perf_clock_kernel_ticks > UINT64_MAX - delta) {
        g_fdwic_perf_clock_kernel_status |= kFdwicPerfClockKernelTickOverflow;
        return;
    }
    if (g_fdwic_perf_clock_kernel_calls == UINT32_MAX) {
        g_fdwic_perf_clock_kernel_status |= kFdwicPerfClockKernelCallOverflow;
        return;
    }
    g_fdwic_perf_clock_kernel_ticks += delta;
    ++g_fdwic_perf_clock_kernel_calls;
}

#endif

PTO_DEVICE_FUNC inline void fdwic_perf_clock_flush(__gm__ DistCore *self) {
    __gm__ FdwicSwimlaneCoreState *core = g_fdwic_perf_clock_core;
    if (core == nullptr || self == nullptr) return;
    core->core_idx = self->core_idx;
    core->block_id = self->block_id;
    core->lane = self->lane;
#if PTO_FDWIC_PERF_CLOCK_KERNEL
    core->count = g_fdwic_perf_clock_kernel_calls;
    core->dropped = g_fdwic_perf_clock_kernel_status;
    core->atomic_calls = 0;
    core->poll_calls = 0;
    core->poll_batch_records =
        PTO_FDWIC_SCHEDULER_MODE == 0 ? kFdwicPerfClockKernelMode : kFdwicPerfClockKernelCrossCoreE2eMode;
    core->perf_clock_kernel.first_submit_start = g_fdwic_perf_clock_first_submit;
    core->perf_clock_kernel.last_submit_end = g_fdwic_perf_clock_last_submit;
    core->perf_clock_kernel.submit_count = static_cast<uint32_t>(self->local_index);
    core->perf_clock_kernel.expected_submit_count = g_fdwic_perf_clock_expected_submits;
    core->perf_clock_kernel.kernel_elapsed_ticks = g_fdwic_perf_clock_kernel_ticks;
#else
    core->count = 0;
    core->dropped = 0;
    core->atomic_calls = 0;
    core->poll_calls = 0;
    core->poll_batch_records = 0;
    core->perf_clock.first_submit_start = g_fdwic_perf_clock_first_submit;
    core->perf_clock.last_submit_end = g_fdwic_perf_clock_last_submit;
    core->perf_clock.submit_count = static_cast<uint32_t>(self->local_index);
    core->perf_clock.expected_submit_count = g_fdwic_perf_clock_expected_submits;
    core->perf_clock.mode = PTO_FDWIC_SCHEDULER_MODE == 0 ? kFdwicPerfClockMode : kFdwicPerfClockCrossCoreE2eMode;
    core->perf_clock.final_seen = g_fdwic_perf_clock_last_submit != 0 ? 1U : 0U;
#endif
    dist_aicore_flush_region(core, sizeof(FdwicSwimlaneCoreState));
}

#else

PTO_DEVICE_FUNC inline void fdwic_perf_clock_attach(__gm__ Runtime *, __gm__ DistCore *) {}
PTO_DEVICE_FUNC inline void fdwic_perf_clock_expect_submits(uint32_t) {}
PTO_DEVICE_FUNC inline void fdwic_perf_clock_worker_begin() {}
PTO_DEVICE_FUNC inline void fdwic_perf_clock_worker_end() {}
PTO_DEVICE_FUNC inline void fdwic_perf_clock_submit_begin(int32_t) {}
PTO_DEVICE_FUNC inline void fdwic_perf_clock_submit_end(int32_t) {}
PTO_DEVICE_FUNC inline void fdwic_perf_clock_flush(__gm__ DistCore *) {}

#endif

}  // namespace
