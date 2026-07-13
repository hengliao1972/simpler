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
    trace_reset_core(self);

    if (!direct_fatal_set()) {
        publish_worker_started();
        uint64_t wd_start = 0;
        while (load_worker_started_count() < g_dist.num_workers && !direct_fatal_set()) {
            SPIN_WAIT_HINT();
            watchdog(wd_start);
        }
    }

    TRACE_LAP_RESET(self);  // origin for the first lap span (post-barrier, pre-replay)
    direct_replay_orch(runtime);

    direct_drain_to_completion(self);
    dist_aicore_finish_worker(runtime);
}
