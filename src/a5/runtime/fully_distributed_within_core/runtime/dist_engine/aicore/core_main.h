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
#if PTO_FDWIC_SCHEDULER_MODE == 3 && defined(__CCE_AICORE__)
#if PTO_FDWIC_SHARED_PA_BUILD_ROLE == 0
#define DIST_CORE_MAIN_ENTRY dist_core_main_aic
#else
#define DIST_CORE_MAIN_ENTRY dist_core_main_aiv
#endif
#else
#define DIST_CORE_MAIN_ENTRY dist_core_main
#endif

DIST_API_ATTR PTO_DEVICE_FUNC void DIST_CORE_MAIN_ENTRY(__gm__ Runtime *runtime, int core_idx, int core_type_int) {
    uint64_t startup_config_invalidate_begin = 0;
    uint64_t startup_config_invalidate_end = 0;
    __gm__ DistCore *self = dist_aicore_attach_worker(
        runtime, core_idx, core_type_int, startup_config_invalidate_begin, startup_config_invalidate_end
    );
    if (self == nullptr) return;
    g_fdwic_joint_submit_seen = false;
    fdwic_perf_clock_attach(runtime, self);
    fdwic_submit_pmu_attach(runtime, self);
    fdwic_swimlane_attach(runtime);
    trace_reset_core(self);
#if DIST_TRACE_ENABLED && PTO_FDWIC_SHARED_MAP
    if (fdwic_atomic_swimlane_enabled()) {
        const uint32_t startup_config_lines = fdwic_dcci_region_cache_line_count(&runtime->dist.shared_addr, 64);
        if (startup_config_lines != 1 ||
            !fdwic_swimlane_record_dcci(
                self, -1, -1, FdwicDcciSite::StartupConfigInvalidate, FdwicDcciOp::Invalidate, /*trailing_dsb=*/true,
                /*call_count=*/1, startup_config_lines, startup_config_invalidate_begin, startup_config_invalidate_end
            )) {
            g_fdwic_dcci_counter_overflow = true;
        }
    }
#else
    (void)startup_config_invalidate_begin;
    (void)startup_config_invalidate_end;
#endif

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

#if PTO_FDWIC_SCHEDULER_MODE == 3
    // Match the role contract proven by the standalone scheduler: the first
    // AIV0 Main Scalar only hosts the persistent SIMT builder and does not
    // replay orchestration. Other Scalars publish dynamic requests. Replaying
    // on the host Scalar would contend with its resident VF, and the dedicated
    // host's local_index cannot participate in sealing.
    const bool simt_builder_worker = dist_simt_cross_core_is_builder_worker(self);
    if (!fdwic_trace_is_fatal()) (void)dist_simt_cross_core_launch_builder(self);
    // Request slots are reset before the run and each task is published once.
    // The publisher therefore has no requirement to start the consumer first.
    // Non-builder Scalars replay Submit immediately; a request may precede its
    // VF consumer, avoiding pointless returned-atomic contention on one
    // builder_started address.
#endif

    // Schema-v4 observes the complete worker business window as two adjacent
    // parents. Reuse the orchestration end as the final-drain start so their
    // aggregate closes exactly in integer SYS_CNT cycles.
    TRACE_TIMESTAMP(orchestration_begin);
#if PTO_FDWIC_SCHEDULER_MODE == 3
    if (!simt_builder_worker) {
        dist_submit_replay_orch(runtime);
    }
#else
    dist_submit_replay_orch(runtime);
#endif
    TRACE_TIMESTAMP(orchestration_end);
#if PTO_FDWIC_SCHEDULER_MODE == 3
    if (!simt_builder_worker && !fdwic_trace_is_fatal()) (void)dist_simt_cross_core_seal_requests(self);
    if (!fdwic_trace_is_fatal()) (void)dist_simt_cross_core_join_builder(self);
#endif
    // A failed run must not wait on a task ring whose dependency closure is no
    // longer valid. Every private replica deterministically finds a capacity
    // error at the same logical Submit, so workers finish and AICPU reports the
    // aggregated error_code to the host.
    if (!fdwic_trace_is_fatal()) dist_submit_drain_to_completion(self);
    TRACE_TIMESTAMP(final_drain_end);
    // Publish both parent records after the measured work. Their own GM writes
    // therefore belong to neither business interval.
    TRACE_SPAN_RECORD(orchestration_begin, orchestration_end, self, -1, -1, TracePhase::OrchestrationReplay, 0, 0);
    TRACE_SPAN_RECORD(orchestration_end, final_drain_end, self, -1, -1, TracePhase::FinalDrain, 0, 0);
    fdwic_swimlane_record_clock_baselines(self, core_idx);
    TRACE_FLUSH_CORE(self);
    fdwic_perf_clock_flush(self);
    fdwic_submit_pmu_flush(self);
    dist_aicore_finish_worker(runtime);
}

#undef DIST_CORE_MAIN_ENTRY
