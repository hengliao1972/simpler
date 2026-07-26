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
#include "pto_runtime_status.h"

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

PTO_DEVICE_FUNC inline bool fatal_set() { return atomic_load(g_dist.fatal) != 0; }
PTO_DEVICE_FUNC inline void set_fatal() { atomic_exchange(g_dist.fatal, 1); }

// 运行时错误使用“首个非零错误码获胜”的合同。先发布错误码，再发布 fatal，
// AICPU 在观察 fatal 后即可取得与本次失败对应的稳定 code。
PTO_DEVICE_FUNC inline void set_fatal_code(int32_t code) {
    if (code == PTO2_ERROR_NONE) code = PTO2_ERROR_EXPLICIT_ORCH_FATAL;
    (void)atomic_compare_exchange(
        g_dist.error_code, int32_t{PTO2_ERROR_NONE}, code, __ATOMIC_RELEASE, __ATOMIC_RELAXED
    );
    store_barrier();
    (void)atomic_exchange(g_dist.fatal, int32_t{1}, __ATOMIC_RELEASE);
}

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
    atomic_exchange(cell.flag, int64_t{0}, __ATOMIC_RELAXED);
    atomic_exchange(cell.vend, uint64_t{0}, __ATOMIC_RELAXED);
}

}  // namespace
