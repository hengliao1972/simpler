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
    // 该接口在首个 Submit 之前由 PA orchestration 调用一次。最终 host 会用
    // DistCore::local_index 与 expected/final_seen 做闭合校验，不静默修正。
    g_fdwic_perf_clock_expected_submits = expected_submits;
}

PTO_DEVICE_FUNC inline void fdwic_perf_clock_submit_begin(int32_t task_id) {
    // task_id 本来就是本核严格递增的 Submit 序号；直接复用，避免另做一份
    // 1280 次 block-local increment/store。
    if (task_id == 0) {
        g_fdwic_perf_clock_first_submit = get_sys_cnt_aicore();
    }
}

PTO_DEVICE_FUNC inline void fdwic_perf_clock_submit_end(int32_t task_id) {
    if (task_id >= 0 && static_cast<uint32_t>(task_id + 1) == g_fdwic_perf_clock_expected_submits &&
        g_fdwic_perf_clock_expected_submits != 0) {
        g_fdwic_perf_clock_last_submit = get_sys_cnt_aicore();
    }
}

#if PTO_FDWIC_PERF_CLOCK_KERNEL

constexpr uint32_t kFdwicPerfClockKernelTickOrderError = 1U << 0;
constexpr uint32_t kFdwicPerfClockKernelTickOverflow = 1U << 1;
constexpr uint32_t kFdwicPerfClockKernelCallOverflow = 1U << 2;

// 只在本核首个 Submit 已进入、末个 Submit 尚未返回时打开 Kernel 子窗。
// 因此最后一个 Submit 之后的 FinalDrain Kernel 不会混入逐核整数闭合。
PTO_DEVICE_FUNC inline uint64_t fdwic_perf_clock_kernel_begin() {
    if (g_fdwic_perf_clock_first_submit == 0 || g_fdwic_perf_clock_last_submit != 0 ||
        g_fdwic_perf_clock_expected_submits == 0 || g_fdwic_perf_clock_kernel_status != 0) {
        return 0;
    }
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
    core->poll_batch_records = kFdwicPerfClockKernelMode;
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
    core->perf_clock.mode = kFdwicPerfClockMode;
    core->perf_clock.final_seen = g_fdwic_perf_clock_last_submit != 0 ? 1U : 0U;
#endif
    dist_aicore_flush_region(core, sizeof(FdwicSwimlaneCoreState));
}

#else

PTO_DEVICE_FUNC inline void fdwic_perf_clock_attach(__gm__ Runtime *, __gm__ DistCore *) {}
PTO_DEVICE_FUNC inline void fdwic_perf_clock_expect_submits(uint32_t) {}
PTO_DEVICE_FUNC inline void fdwic_perf_clock_submit_begin(int32_t) {}
PTO_DEVICE_FUNC inline void fdwic_perf_clock_submit_end(int32_t) {}
PTO_DEVICE_FUNC inline void fdwic_perf_clock_flush(__gm__ DistCore *) {}

#endif

}  // namespace
