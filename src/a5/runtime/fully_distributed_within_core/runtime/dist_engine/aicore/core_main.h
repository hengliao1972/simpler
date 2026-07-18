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

// -----------------------------------------------------------------------------
// Per-core entry point invoked by each AICore worker thread.
// -----------------------------------------------------------------------------
DIST_API_ATTR PTO_DEVICE_FUNC void dist_core_main(__gm__ Runtime *runtime, int core_idx, int core_type_int) {
    __gm__ DistCore *self = dist_aicore_attach_worker(runtime, core_idx, core_type_int);
    if (self == nullptr) return;
    g_fdwic_joint_submit_seen = false;
    fdwic_swimlane_attach(runtime);
    trace_reset_core(self);

    if (!fdwic_trace_is_fatal()) {
        (void)fdwic_trace_atomic_fetch_add<int64_t>(
            -1, FdwicAtomicSite::StartupIncrement, g_dist.started_count, 1, /*result_used=*/false
        );
        uint64_t wd_start = 0;
        const uint32_t startup_poll_region = fdwic_atomic_poll_region_begin(
            fdwic_atomic_site_mask(FdwicAtomicSite::StartupPoll) | fdwic_atomic_site_mask(FdwicAtomicSite::FatalPoll)
        );
        while (fdwic_trace_atomic_load(-1, FdwicAtomicSite::StartupPoll, g_dist.started_count) < g_dist.num_workers &&
               !fdwic_trace_is_fatal()) {
            SPIN_WAIT_HINT();
            watchdog(wd_start);
        }
        fdwic_atomic_poll_region_end(startup_poll_region);
    }

    TRACE_LAP_RESET(self);  // origin for the first lap span (post-barrier, pre-replay)
    dist_submit_replay_orch(runtime);

    dist_submit_drain_to_completion(self);
    fdwic_swimlane_record_clock_baselines(self, core_idx);
    TRACE_FLUSH_CORE(self);
    dist_aicore_finish_worker(runtime);
}
