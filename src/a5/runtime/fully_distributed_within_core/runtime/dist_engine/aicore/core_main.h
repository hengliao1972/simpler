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
#if defined(__CCE_AICORE__)
    dist_aicore_onboard_main(runtime, core_idx, core_type_int);
    return;
#else
    if (core_idx < 0 || core_idx >= RUNTIME_MAX_WORKER) return;
    g_dist_ptr = reinterpret_cast<DistGlobal *>(runtime->dist.shared_addr);
    if (g_dist_ptr == nullptr) return;
    __gm__ DistCore *self = &g_dist.cores[core_idx];
    const CoreType role = static_cast<CoreType>(core_type_int);

    // sub_block lane: only meaningful for AIV in MIX tasks (M3). bgemm's 1V add
    // ignores it, so 0 is correct for the M2 single-core scope.
    // Copy field-by-field from the __gm__ layout entry into a stack local so
    // downstream code reads plain int32_t (CCEC forbids copy-initializing a
    // non-__gm__ struct value from a __gm__ struct value, and CoreLayout has
    // no user-defined ctor / operator= to overload for the cross-space copy).
    __gm__ const CoreLayout &layout_gm = g_dist.layout[core_idx];
    const CoreLayout lay = {layout_gm.block_id, layout_gm.lane};
    dist_core_reset(*self, role, lay.block_id, lay.lane);
    self->core_idx = core_idx;
#if DIST_TRACE_ENABLED
    trace_reset_core(self);
#endif
    g_self = self;
    // Startup barrier: wait until every worker thread has been scheduled in and
    // reached this point before anyone begins replay. In sim the OS brings the
    // host threads up one at a time, so without this the cores that start early
    // race ahead and the swimlane's first-task stagger reflects thread-wakeup
    // skew rather than engine scheduling. Bare spin (no yield) per the AICPU
    // spin-wait convention. Skipped under fatal so a failed run still tears down.
    if (!fatal_set()) {
        atom_fetch_add<int64_t>(g_dist.started_count, 1, __ATOMIC_ACQ_REL);
        uint64_t wd_start = 0;
        while (atom_load(g_dist.started_count, __ATOMIC_ACQUIRE) < g_dist.num_workers && !fatal_set()) {
            SPIN_WAIT_HINT();
            watchdog(wd_start);
        }
    }

    // Replay the full orchestration submit stream: build the per-core map and
    // claim/build owned tasks into the private ring (back-pressure inline). MIX
    // anchors deposit follower subtasks into block.won during this replay.
    TRACE_LAP_RESET(self);  // origin for the first lap span (post-barrier, pre-replay)
    if (g_dist.orch_args != nullptr && !fatal_set()) {
        aicpu_orchestration_entry(*g_dist.orch_args);
    }

    ccec_drain_to_completion(self);
    g_self = nullptr;
    ccec_publish_done();
    __atomic_add_fetch(&runtime->dist.done_count, 1, __ATOMIC_ACQ_REL);
#endif
}
