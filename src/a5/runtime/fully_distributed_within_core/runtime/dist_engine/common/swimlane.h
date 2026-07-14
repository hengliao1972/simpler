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
#include "dist_engine/common/swimlane_types.h"
#include "dist_engine/common/worker_state.h"
#include "dist_engine/common/atomic.h"
#include "dist_engine/aicore/primitive.h"

#if defined(__CCE_AICORE__) || defined(__CPU_SIM)
#include "inner_kernel.h"
#endif

namespace {

PTO_DEVICE_FUNC inline uint64_t fdwic_swimlane_detail_now() {
#if defined(__CCE_AICORE__) || defined(__CPU_SIM)
    return get_sys_cnt_aicore();
#else
    return 0;
#endif
}

PTO_DEVICE_FUNC inline bool fdwic_swimlane_enabled() { return g_fdwic_swimlane_enabled; }

PTO_DEVICE_FUNC inline void fdwic_swimlane_attach(__gm__ Runtime *runtime) {
    g_fdwic_swimlane_enabled = false;
    g_fdwic_swimlane_header = nullptr;
    g_fdwic_swimlane_core = nullptr;
    g_fdwic_swimlane_records = nullptr;
    g_fdwic_swimlane_records_per_core = 0;
    if (runtime == nullptr) return;
#if defined(__CCE_AICORE__)
    dist_aicore_invalidate_region(const_cast<__gm__ uint64_t *>(&runtime->dist.swimlane_base), 64);
#endif
    const uint64_t base = runtime->dist.swimlane_base;
    const uint32_t records_per_core = runtime->dist.swimlane_records_per_core;
    if (runtime->dist.swimlane_enabled == 0 || base == 0 || records_per_core == 0) return;
    g_fdwic_swimlane_header = reinterpret_cast<__gm__ FdwicSwimlaneHeader *>(base);
    g_fdwic_swimlane_records_per_core = records_per_core;
    g_fdwic_swimlane_enabled = true;
}

PTO_DEVICE_FUNC inline __gm__ FdwicSwimlaneRecord *fdwic_swimlane_detail_records(__gm__ FdwicSwimlaneHeader *header) {
    return reinterpret_cast<__gm__ FdwicSwimlaneRecord *>(
        reinterpret_cast<__gm__ uint8_t *>(header) + sizeof(FdwicSwimlaneHeader)
    );
}

PTO_DEVICE_FUNC inline void fdwic_swimlane_reset_core(__gm__ DistCore *self) {
    if (!fdwic_swimlane_enabled() || self == nullptr) return;
    __gm__ FdwicSwimlaneHeader *header = g_fdwic_swimlane_header;
    if (header == nullptr) return;
    if (self->core_idx < 0 || self->core_idx >= static_cast<int32_t>(header->num_cores)) return;
    g_fdwic_swimlane_core = &header->cores[self->core_idx];
    g_fdwic_swimlane_records = &fdwic_swimlane_detail_records(
        header
    )[static_cast<uint64_t>(self->core_idx) * g_fdwic_swimlane_records_per_core];
    g_fdwic_swimlane_core->count = 0;
    g_fdwic_swimlane_core->dropped = 0;
}

PTO_DEVICE_FUNC inline void fdwic_swimlane_flush_core(__gm__ DistCore *self) {
    if (!fdwic_swimlane_enabled() || self == nullptr) return;
    const uint32_t records_per_core = g_fdwic_swimlane_records_per_core;
    __gm__ FdwicSwimlaneCoreState *core = g_fdwic_swimlane_core;
    if (core == nullptr || g_fdwic_swimlane_records == nullptr || records_per_core == 0) return;
    const uint32_t count = core->count < records_per_core ? core->count : records_per_core;
    if (count > 0) {
        dist_aicore_flush_region(g_fdwic_swimlane_records, static_cast<uint64_t>(count) * sizeof(FdwicSwimlaneRecord));
    }
    dist_aicore_flush_region(core, sizeof(FdwicSwimlaneCoreState));
}

PTO_DEVICE_FUNC inline void fdwic_swimlane_detail_record(
    __gm__ DistCore *self, int32_t task_id, int32_t func_id, FdwicSwimlanePhase phase, uint64_t start_cycle,
    uint64_t end_cycle, uint32_t flags = 0, uint32_t aux = 0
) {
    if (!fdwic_swimlane_enabled() || self == nullptr) return;
    const int32_t core_idx = self->core_idx;
    const uint32_t records_per_core = g_fdwic_swimlane_records_per_core;
    __gm__ FdwicSwimlaneCoreState *core = g_fdwic_swimlane_core;
    if (core == nullptr || g_fdwic_swimlane_records == nullptr || records_per_core == 0) return;
    uint32_t slot = core->count;
    if (slot >= records_per_core) {
        core->dropped = core->dropped + 1;
        return;
    }
    __gm__ FdwicSwimlaneRecord *record = &g_fdwic_swimlane_records[slot];
    record->start_cycle = start_cycle;
    record->end_cycle = end_cycle;
    record->task_id = task_id;
    record->func_id = func_id;
    record->phase = static_cast<int32_t>(phase);
    record->lane = self->lane;
    record->block_id = self->block_id;
    record->core_idx = core_idx;
    record->flags = flags;
    record->aux = aux;
    core->count = slot + 1;
}

PTO_DEVICE_FUNC inline void fdwic_swimlane_lap_reset(__gm__ DistCore *self) {
    if (self == nullptr) return;
    self->swimlane_last_cycle = fdwic_swimlane_detail_now();
}

PTO_DEVICE_FUNC inline void
fdwic_swimlane_lap(__gm__ DistCore *self, int32_t task_id, int32_t func_id, FdwicSwimlanePhase phase) {
    if (self == nullptr) return;
    const uint64_t end_cycle = fdwic_swimlane_detail_now();
    const uint64_t start_cycle = self->swimlane_last_cycle;
    fdwic_swimlane_detail_record(self, task_id, func_id, phase, start_cycle, end_cycle);
    self->swimlane_last_cycle = end_cycle;
}

}  // namespace
