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
#include "dist_engine/common/swimlane.h"

#if DIST_SIM_HOST_CLOCK
#include <chrono>
#endif

namespace {

PTO_DEVICE_FUNC inline FdwicSwimlanePhase trace_phase_to_swimlane_phase(TracePhase phase) {
    switch (phase) {
    case TracePhase::Kernel:
        return FdwicSwimlanePhase::Kernel;
    case TracePhase::Alloc:
        return FdwicSwimlanePhase::Alloc;
    case TracePhase::Build:
        return FdwicSwimlanePhase::Build;
    case TracePhase::DrainWon:
        return FdwicSwimlanePhase::DrainWon;
    case TracePhase::Replay:
        return FdwicSwimlanePhase::Replay;
    case TracePhase::RingBp:
        return FdwicSwimlanePhase::RingBp;
    case TracePhase::EfDrain:
        return FdwicSwimlanePhase::EfDrain;
    case TracePhase::Commit:
        return FdwicSwimlanePhase::Commit;
    }
    return FdwicSwimlanePhase::Kernel;
}

#if DIST_SIM_HOST_CLOCK
inline uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
}
#endif

#if DIST_TRACE_ENABLED

PTO_DEVICE_FUNC inline void trace_reset_core(__gm__ DistCore *self) { fdwic_swimlane_reset_core(self); }

PTO_DEVICE_FUNC inline void trace_lap_reset_impl(__gm__ DistCore *self) { fdwic_swimlane_lap_reset(self); }

PTO_DEVICE_FUNC inline void trace_lap_impl(__gm__ DistCore *self, int32_t task_id, int32_t func_id, TracePhase phase) {
    fdwic_swimlane_lap(self, task_id, func_id, trace_phase_to_swimlane_phase(phase));
}

PTO_DEVICE_FUNC inline void trace_span_impl(
    __gm__ DistCore *self, int32_t task_id, int32_t func_id, TracePhase phase, uint64_t start_cycle, uint64_t end_cycle,
    uint32_t flags = 0, uint32_t aux = 0
) {
    fdwic_swimlane_detail_record(
        self, task_id, func_id, trace_phase_to_swimlane_phase(phase), start_cycle, end_cycle, flags, aux
    );
}

PTO_DEVICE_FUNC inline void
trace_instant_impl(__gm__ DistCore *self, int32_t task_id, int32_t func_id, TracePhase phase, uint32_t flags = 0) {
    const uint64_t cycle = fdwic_swimlane_detail_now();
    trace_span_impl(self, task_id, func_id, phase, cycle, cycle, flags, 0);
}

PTO_DEVICE_FUNC inline uint64_t trace_span_begin_impl() { return fdwic_swimlane_detail_now(); }

#define TRACE_LAP(self, task_id, func_id, phase) trace_lap_impl((self), (task_id), (func_id), (phase))
#define TRACE_LAP_RESET(self) trace_lap_reset_impl((self))
#define TRACE_SPAN_BEGIN(name) const uint64_t name = trace_span_begin_impl()
#define TRACE_SPAN_END(name, self, task_id, func_id, phase, flags, aux) \
    trace_span_impl((self), (task_id), (func_id), (phase), (name), trace_span_begin_impl(), (flags), (aux))
#define TRACE_INSTANT(self, task_id, func_id, phase, flags) \
    trace_instant_impl((self), (task_id), (func_id), (phase), (flags))

#else

PTO_DEVICE_FUNC inline void trace_reset_core(__gm__ DistCore *) {}

#define TRACE_LAP(self, task_id, func_id, phase) ((void)0)
#define TRACE_LAP_RESET(self) ((void)0)
#define TRACE_SPAN_BEGIN(name) ((void)0)
#define TRACE_SPAN_END(name, self, task_id, func_id, phase, flags, aux) ((void)0)
#define TRACE_INSTANT(self, task_id, func_id, phase, flags) ((void)0)

#endif

}  // namespace
