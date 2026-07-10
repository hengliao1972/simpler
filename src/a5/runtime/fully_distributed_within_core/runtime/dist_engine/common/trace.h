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

#include "dist_engine/common/state.h"

#if DIST_SIM_HOST_CLOCK
#include <chrono>
#endif
#if DIST_TRACE_ENABLED
#include <cstdlib>
#include <ctime>
#endif

namespace {

#if DIST_SIM_HOST_CLOCK
inline uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
}
#endif

#if DIST_TRACE_ENABLED
bool g_trace_on = false;
uint64_t g_trace_epoch_ns = 0;
int32_t g_trace_reserve = 0;
DistCoreTraceState g_trace_cores[kDistRuntimeMaxWorker];

inline uint64_t thread_cpu_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

inline uint64_t trace_now() { return g_trace_on ? now_ns() : 0; }
inline uint64_t trace_now_cpu() { return g_trace_on ? thread_cpu_ns() : 0; }

inline DistCoreTraceState &trace_state(DistCore *self) { return g_trace_cores[self->core_idx]; }

inline void trace_reset_core(DistCore *self) {
    DistCoreTraceState &ts = trace_state(self);
    ts.trace_last_ns = 0;
    ts.trace_last_cpu = 0;
    ts.trace.clear();
    if (g_trace_reserve > 0) ts.trace.reserve(g_trace_reserve);
    ts.dep_edges.clear();
    if (g_trace_reserve > 0) ts.dep_edges.reserve(g_trace_reserve);
    ts.slot_edges.clear();
    if (g_trace_reserve > 0) ts.slot_edges.reserve(g_trace_reserve);
}

inline void trace_overhead_impl(
    DistCore *self, int32_t task_id, int32_t func_id, TracePhase phase, uint64_t t0_ns, uint64_t t0_cpu
) {
    if (!g_trace_on) return;
    const uint64_t t1 = now_ns();
    const uint64_t c1 = thread_cpu_ns();
    trace_state(self).trace.push_back(
        TraceEvent{
            task_id, func_id, self->lane, /*multicore=*/0, phase, t0_ns - g_trace_epoch_ns, t1 - t0_ns, c1 - t0_cpu
        }
    );
}

inline void trace_lap_reset_impl(DistCore *self) {
    if (!g_trace_on) return;
    DistCoreTraceState &ts = trace_state(self);
    ts.trace_last_ns = now_ns();
    ts.trace_last_cpu = thread_cpu_ns();
}

inline void trace_lap_impl(DistCore *self, int32_t task_id, int32_t func_id, TracePhase phase) {
    if (!g_trace_on) return;
    const uint64_t t1 = now_ns();
    const uint64_t c1 = thread_cpu_ns();
    DistCoreTraceState &ts = trace_state(self);
    ts.trace.push_back(
        TraceEvent{
            task_id, func_id, self->lane, /*multicore=*/0, phase, ts.trace_last_ns - g_trace_epoch_ns,
            t1 - ts.trace_last_ns, c1 - ts.trace_last_cpu
        }
    );
    ts.trace_last_ns = t1;
    ts.trace_last_cpu = c1;
}

#define TRACE_LAP(self, task_id, func_id, phase) trace_lap_impl((self), (task_id), (func_id), (phase))
#define TRACE_LAP_RESET(self) trace_lap_reset_impl((self))
#define TRACE_OVERHEAD(self, task_id, func_id, phase, t0_ns, t0_cpu) \
    trace_overhead_impl((self), (task_id), (func_id), (phase), (t0_ns), (t0_cpu))
#else
#define TRACE_LAP(self, task_id, func_id, phase) ((void)0)
#define TRACE_LAP_RESET(self) ((void)0)
#define TRACE_OVERHEAD(self, task_id, func_id, phase, t0_ns, t0_cpu) ((void)0)
#endif

#if DIST_TRACE_ENABLED
inline bool dist_trace() {
    static const bool on = (getenv("PTO_DIST_TRACE") != nullptr);
    return on;
}
#else
inline bool dist_trace() { return false; }
#endif

}  // namespace
