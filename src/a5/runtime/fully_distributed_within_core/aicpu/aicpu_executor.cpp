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
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "aicpu/device_time.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "runtime.h"
#include "spin_hint.h"

// Runtime headers
#include "pto_runtime2.h"
#include "dist_engine/dist_engine.h"

// Performance profiling headers
#include "aicpu/l2_swimlane_collector_aicpu.h"
#include "common/l2_swimlane_profiling.h"
#include "common/unified_log.h"

// Register-based communication
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

// Scheduler context class
#include "scheduler/scheduler_context.h"

static PTO2Runtime *rt{nullptr};

struct AicpuExecutor {
    int32_t sched_thread_num_;

    // ===== Thread management state =====
    std::atomic<int32_t> thread_idx_{0};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};
    std::atomic<bool> finished_{false};

    int32_t aicpu_thread_num_{0};

    // ===== Per-run completion state =====

    std::atomic<int32_t> finished_count_{0};
    std::atomic<bool> runtime_done_{false};

    // Entry-arg L2TaskArgs built from get_orch_args() and consumed by the
    // AICore-side distributed engine.
    L2TaskArgs orch_args_cached_;

    // ===== Core lifecycle context =====
    // Still named SchedulerContext because it also owns legacy dispatch code,
    // but this executor only uses its handshake, core counting, and shutdown
    // responsibilities.
    SchedulerContext sched_ctx_;

    // ===== Methods =====
    int32_t init(Runtime *runtime);
    int32_t run(Runtime *runtime);
    void deinit(Runtime *runtime);

    ~AicpuExecutor() = default;
};

static AicpuExecutor g_aicpu_executor;

// ===== AicpuExecutor Method Implementations =====

int32_t AicpuExecutor::init(Runtime *runtime) {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return 0;
    }

    LOG_INFO_V0("AicpuExecutor: Initializing");

    if (runtime == nullptr) {
        LOG_ERROR("runtime is nullptr");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    // Read execution parameters from runtime. The 0 → 1 fixup runs before the
    // sched_thread_num_ derivation so a zero input doesn't leave the scheduler
    // count at -1.
    aicpu_thread_num_ = runtime->aicpu_thread_num;
    if (aicpu_thread_num_ == 0) aicpu_thread_num_ = 1;
    sched_thread_num_ = aicpu_thread_num_ - 1;

    if (aicpu_thread_num_ < 1 || aicpu_thread_num_ > MAX_AICPU_THREADS) {
        LOG_ERROR("Invalid aicpu_thread_num: %d", aicpu_thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    if (sched_ctx_.init(runtime, aicpu_thread_num_, sched_thread_num_, false, get_platform_regs()) != 0) {
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    finished_count_.store(0, std::memory_order_release);
    runtime_done_.store(false, std::memory_order_release);

    init_done_.store(true, std::memory_order_release);
    LOG_INFO_V0("AicpuExecutor: Init complete");
    return 0;
}

/**
 * Shutdown AICore - Send exit signal via registers to all AICore kernels
 */
int32_t AicpuExecutor::run(Runtime *runtime) {
    // Prefer the filter gate's deterministic exec_idx so role assignment
    // (sched 0..N-2 / orch N-1) is driven by host-computed ALLOWED_CPUS,
    // not arrival order. Fall back to the legacy fetch-add counter on
    // platforms where the filter gate is inactive (sim sets exec_idx via
    // its own stub; the fallback covers any path that bypassed the gate).
    int32_t affinity_exec_idx = platform_aicpu_affinity_thread_idx();
    int32_t thread_idx = (affinity_exec_idx >= 0) ? affinity_exec_idx : (thread_idx_++);
    int32_t run_rc = 0;
    LOG_INFO_V0("Thread %d: Start (exec_idx=%d)", thread_idx, affinity_exec_idx);

    // Orchestrator check
    if (thread_idx >= sched_thread_num_) {
#if PTO2_PROFILING
        uint64_t orch_cycle_start = 0;
#endif
        // AICPU setup thread: initialize the shared distributed state, start
        // AICore workers, and wait for their on-core orchestration replay.
        {
            LOG_INFO_V0("Thread %d: Preparing shared dist state", thread_idx);

            // Build the entry-arg once per run. The AICore-side dist engine
            // consumes it when replaying the linked orchestration entry.
            orch_args_cached_.create_from_chip_args(runtime->get_orch_args());

            // rt is bound to *this* run's memory and must be reattached every
            // run.
            const ChipStorageTaskArgs &args = runtime->get_orch_args();
            int32_t arg_count = args.tensor_count() + args.scalar_count();
            LOG_INFO_V0("Thread %d: arg_count=%d", thread_idx, arg_count);
            for (int32_t i = 0; i < args.tensor_count() && i < 20; i++) {
                const Tensor &t = args.tensor(i);
                LOG_INFO_V0(
                    "Thread %d: orch_args[%d] = TENSOR(data=0x%lx, ndims=%u, dtype=%u)", thread_idx, i,
                    static_cast<uint64_t>(t.buffer.addr), t.ndims, static_cast<unsigned>(t.dtype)
                );
            }
            for (int32_t i = 0; i < args.scalar_count() && (args.tensor_count() + i) < 20; i++) {
                LOG_INFO_V0(
                    "Thread %d: orch_args[%d] = SCALAR(0x%lx)", thread_idx, args.tensor_count() + i,
                    static_cast<uint64_t>(args.scalar(i))
                );
            }

            // Host has pre-populated the runtime header and uploaded it into
            // the pooled runtime_arena buffer. The distributed path only needs
            // the PTO2Runtime header itself: CPU-sim AICore wrappers bind
            // rt->ops inside dist_core_main(), and dist_engine reads the
            // example-exec-time fields from rt.
            void *prebuilt_arena = runtime->get_prebuilt_arena_base();
            size_t off_runtime = runtime->get_prebuilt_runtime_offset();
            if (prebuilt_arena == nullptr) {
                LOG_ERROR("Thread %d: prebuilt_arena_base is null", thread_idx);
                runtime_done_.store(true, std::memory_order_release);
                return -1;
            }
            rt = reinterpret_cast<PTO2Runtime *>(static_cast<char *>(prebuilt_arena) + off_runtime);

#if PTO2_PROFILING
            if (get_l2_swimlane_level() >= L2SwimlaneLevel::ORCH_PHASES) {
                l2_swimlane_aicpu_set_orch_thread_idx(thread_idx);
            }
#endif

#if PTO2_PROFILING
            orch_cycle_start = get_sys_cnt_aicpu();
#endif
            // ---- fully_distributed_within_core handoff ----
            // Instead of running orchestration here, wire the distributed engine
            // (resets cursors/flags/heap) and hand the per-core entry off to
            // the AICore worker threads, which replay orchestration in SPMD
            // fashion and execute the tasks they win. This AICPU thread then
            // waits for all workers.
            // See runtime/dist_engine.* and docs/fully_distributed_within_core.md.
            {
                const int32_t num_workers = runtime->worker_count;
                void *core_main = dist_engine_register(rt, &orch_args_cached_, num_workers, runtime);
                runtime->dist.core_main_fn = reinterpret_cast<uint64_t>(core_main);
                runtime->dist.num_workers = num_workers;
                __atomic_store_n(&runtime->dist.done_count, 0, __ATOMIC_RELEASE);
                __atomic_store_n(&runtime->dist.go, 1u, __ATOMIC_RELEASE);
                const bool dist_trace = (getenv("PTO_DIST_TRACE") != nullptr);
                if (dist_trace)
                    fprintf(stderr, "[dist] Thread %d: engine wired, %d workers launched\n", thread_idx, num_workers);
                while (__atomic_load_n(&runtime->dist.done_count, __ATOMIC_ACQUIRE) < num_workers) {
                    SPIN_WAIT_HINT();
                }
                // All workers done (single-threaded here): emit the per-core
                // execution swimlane if PTO_DIST_SWIMLANE is set (else no-op).
                dist_engine_dump_trace();
                if (dist_trace)
                    fprintf(stderr, "[dist] Thread %d: all %d distributed workers finished\n", thread_idx, num_workers);
            }
            runtime_done_.store(true, std::memory_order_release);
        }
#if PTO2_PROFILING
        uint64_t orch_end_ts = get_sys_cnt_aicpu();
        LOG_INFO_V9(
            "Thread %d: orch_start=%" PRIu64 " orch_end=%" PRIu64 " orch_cost=%.3fus", thread_idx,
            static_cast<uint64_t>(orch_cycle_start), static_cast<uint64_t>(orch_end_ts),
            cycles_to_us(orch_end_ts - orch_cycle_start)
        );
#endif
        LOG_INFO_V0("Thread %d: Orchestrator completed", thread_idx);
    } else {
        while (!runtime_done_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
    }

    // Shutdown is gated by runtime_done_: AICores only enter the EXIT wait
    // after completing the distributed replay, so sending EXIT earlier would
    // either race execution or time out while workers are still inside Phase 4.
    int32_t shutdown_rc = sched_ctx_.shutdown(thread_idx);
    if (shutdown_rc != 0 && run_rc == 0) {
        run_rc = shutdown_rc;
    }

    LOG_INFO_V0("Thread %d: Completed", thread_idx);

    // Check if this is the last thread to finish
    int32_t prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == aicpu_thread_num_) {
        finished_.store(true, std::memory_order_release);
        rt = nullptr;
    }

    return run_rc;
}

void AicpuExecutor::deinit(Runtime *runtime) {
    // 1. Invalidate AICPU cache for Runtime address range.
    //    Next round's Host DMA (rtMemcpy) writes fresh Runtime to HBM but
    //    bypasses this cache. Invalidating now ensures next round reads from HBM.
    cache_invalidate_range(runtime, sizeof(Runtime));

    // Reset all SchedulerContext-owned state in one place.
    sched_ctx_.deinit();

    finished_count_.store(0, std::memory_order_release);
    runtime_done_.store(false, std::memory_order_release);

    aicpu_thread_num_ = 0;
    sched_thread_num_ = 0;

    orch_args_cached_.reset();
    // Clear file-scope PTO2Runtime pointer (freed by orchestrator thread before deinit)
    rt = nullptr;

    LOG_INFO_V0("DeInit: Runtime execution state reset");

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    LOG_INFO_V0("DeInit: AicpuExecutor reset complete");
}

// ===== Public Entry Point =====

/**
 * aicpu_execute - Main AICPU kernel execution entry point
 *
 * This is called by DynTileFwkBackendKernelServer in kernel.cpp.
 * Orchestrates the complete task runtime execution:
 * 1. Initialize executor (thread-safe, first thread only)
 * 2. Wait for initialization to complete
 * 3. Execute tasks on managed cores
 * 4. Cleanup when last thread finishes
 *
 * @param runtime Pointer to Runtime structure
 * @return 0 on success, non-zero on error
 */
extern "C" int32_t aicpu_execute(Runtime *runtime) {
    if (runtime == nullptr) {
        LOG_ERROR("%s", "Invalid argument: null Runtime pointer");
        return -1;
    }

    LOG_INFO_V0("%s", "aicpu_execute: Starting AICPU kernel execution");

    g_aicpu_executor.init(runtime);

    while (!g_aicpu_executor.init_done_.load(std::memory_order_acquire)) {
        if (g_aicpu_executor.init_failed_.load(std::memory_order_acquire)) {
            LOG_ERROR("%s", "aicpu_execute: Initialization failed, aborting execution");
            return -1;
        }
    }

    int32_t rc = g_aicpu_executor.run(runtime);
    if (rc != 0) {
        LOG_ERROR("aicpu_execute: Thread execution failed with rc=%d", rc);
    }

    // Last thread cleans up
    if (g_aicpu_executor.finished_.load(std::memory_order_acquire)) {
        LOG_INFO_V0("aicpu_execute: Last thread finished, cleaning up");
        g_aicpu_executor.deinit(runtime);
    }

    if (rc != 0) {
        return rc;
    }

    LOG_INFO_V0("%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}
