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

#include "dist_engine/common/atomic.h"
#include "dist_engine/common/state.h"
#include "dist_engine/common/worker_state.h"

#if DIST_SIM_HOST_CLOCK
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#endif

#if defined(__CPU_SIM)
void dist_dump_state(int);
#endif

namespace {

#if DIST_SIM_HOST_CLOCK
inline uint64_t dist_now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
}
#endif

PTO_DEVICE_FUNC inline bool fatal_set() { return atom_load(g_dist.fatal, __ATOMIC_ACQUIRE) != 0; }
PTO_DEVICE_FUNC inline void set_fatal() { atom_store(g_dist.fatal, 1, __ATOMIC_RELEASE); }

PTO_DEVICE_FUNC inline void watchdog([[maybe_unused]] uint64_t &start_ns) {
#if DIST_SIM_HOST_CLOCK
    static const long budget_s = []() -> long {
        const char *e = getenv("PTO_DIST_WATCHDOG");
        return e ? std::strtol(e, nullptr, 10) : 0;
    }();
    if (budget_s <= 0) return;
    const uint64_t now = dist_now_ns();
    if (start_ns == 0) {
        start_ns = now;
        return;
    }
    if (now - start_ns > static_cast<uint64_t>(budget_s) * 1000000000ull) {
        static std::atomic<int32_t> dumped{0};
        int32_t exp = 0;
        if (dumped.compare_exchange_strong(exp, 1, std::memory_order_acq_rel)) {
            fprintf(stderr, "[dist_engine] WATCHDOG fired after %lds; dumping state\n", budget_s);
            dist_dump_state(0);
        }
        set_fatal();
    }
#endif
}

PTO_DEVICE_FUNC inline __gm__ DistTaskCell &task_cell(int32_t task_id) {
    return g_dist.tasks[task_id & (kFlagCap - 1)];
}

PTO_DEVICE_FUNC void reset_task_cell(int32_t task_id) {
    __gm__ DistTaskCell &cell = task_cell(task_id);
    atom_store(cell.flag, 0, __ATOMIC_RELAXED);
    atom_store(cell.vend, 0, __ATOMIC_RELAXED);
}

}  // namespace
