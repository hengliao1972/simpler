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

#include "aicpu/device_time.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "runtime.h"
#include "spin_hint.h"

// Runtime headers
#include "pto_runtime2.h"
#include "dist_engine/dist_engine.h"
#include "dist_engine/aicpu/submit_pmu_owner.h"

// Performance profiling headers
#include "aicpu/l2_swimlane_collector_aicpu.h"
#include "common/l2_swimlane_profiling.h"
#include "common/unified_log.h"

// Register-based communication
#include "aicpu/platform_regs.h"
#include "common/platform_config.h"

#if defined(__aarch64__)
#define AICPU_STORE_BARRIER() __asm__ volatile("dmb ishst" ::: "memory")
#elif defined(__x86_64__)
#define AICPU_STORE_BARRIER() __asm__ volatile("" ::: "memory")
#else
#define AICPU_STORE_BARRIER() std::atomic_thread_fence(std::memory_order_release)
#endif

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
    // Orchestrator owns the distributed run, then publishes one common status
    // before runtime_done_. Every AICPU block returns the same runtime failure
    // instead of depending on launcher-specific multi-block aggregation.
    std::atomic<int32_t> run_status_{0};

    // ===== Core lifecycle context =====
    PTO2DispatchPayload payload_per_core_[RUNTIME_MAX_WORKER][2];
    uint64_t core_reg_addrs_[RUNTIME_MAX_WORKER]{};
    int32_t core_count_{0};

    // ===== Methods =====
    int32_t init(Runtime *runtime);
    int32_t run(Runtime *runtime);
    void deinit(Runtime *runtime);
    int32_t handshake_all_cores(Runtime *runtime);
    int32_t shutdown_cores(int32_t thread_idx);

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

    if (aicpu_thread_num_ < 1 || aicpu_thread_num_ > PLATFORM_MAX_AICPU_THREADS) {
        LOG_ERROR("Invalid aicpu_thread_num: %d", aicpu_thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    if (handshake_all_cores(runtime) != 0) {
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    finished_count_.store(0, std::memory_order_release);
    runtime_done_.store(false, std::memory_order_release);
    run_status_.store(0, std::memory_order_release);

    init_done_.store(true, std::memory_order_release);
    LOG_INFO_V0("AicpuExecutor: Init complete");
    return 0;
}

int32_t AicpuExecutor::handshake_all_cores(Runtime *runtime) {
    Handshake *all_handshakes = reinterpret_cast<Handshake *>(runtime->workers);
    core_count_ = runtime->worker_count;

    if (core_count_ <= 0 || core_count_ > RUNTIME_MAX_WORKER) {
        LOG_ERROR("Invalid worker_count %d (expected 1-%d)", core_count_, RUNTIME_MAX_WORKER);
        return -1;
    }

    uint64_t *regs = reinterpret_cast<uint64_t *>(get_platform_regs());
    if (regs == nullptr) {
        LOG_ERROR("platform regs unavailable during handshake");
        return -1;
    }
    const uint32_t max_physical_cores_count = platform_get_physical_cores_count();

    LOG_INFO_V0("Handshaking with %d cores", core_count_);
    for (int32_t i = 0; i < core_count_; i++) {
        all_handshakes[i].task = reinterpret_cast<uint64_t>(&payload_per_core_[i][0]);
        AICPU_STORE_BARRIER();
        all_handshakes[i].aicpu_ready = AICPU_READY_HANDSHAKE;
    }
    AICPU_STORE_BARRIER();

    bool failed = false;
    for (int32_t i = 0; i < core_count_; i++) {
        Handshake *hank = &all_handshakes[i];

        while (hank->aicore_regs_ready == 0) {
            SPIN_WAIT_HINT();
        }

        const uint32_t physical_core_id = hank->physical_core_id;
        if (physical_core_id >= max_physical_cores_count) {
            LOG_ERROR(
                "Core %d reported invalid physical_core_id=%u (platform max=%u)", i, physical_core_id,
                max_physical_cores_count
            );
            failed = true;
            continue;
        }

        const uint64_t reg_addr = regs[physical_core_id];
        platform_init_aicore_regs(reg_addr);
        core_reg_addrs_[i] = reg_addr;
        AICPU_STORE_BARRIER();
        hank->aicpu_regs_ready = 1;
        AICPU_STORE_BARRIER();

        while (hank->aicore_done == 0) {
            SPIN_WAIT_HINT();
        }

        LOG_INFO_V0(
            "Core %d: %s, physical_id=%u, reg_addr=0x%lx", i, hank->core_type == CoreType::AIC ? "AIC" : "AIV",
            physical_core_id, reg_addr
        );
    }

    if (failed) {
        for (int32_t i = 0; i < core_count_; i++) {
            if (core_reg_addrs_[i] != 0) {
                platform_deinit_aicore_regs(core_reg_addrs_[i]);
            }
        }
        return -1;
    }
    return 0;
}

int32_t AicpuExecutor::shutdown_cores(int32_t thread_idx) {
    if (thread_idx != 0) return 0;

    LOG_INFO_V0("Thread %d: Shutting down %d cores", thread_idx, core_count_);
    int32_t rc = 0;
    for (int32_t i = 0; i < core_count_; i++) {
        if (core_reg_addrs_[i] == 0) {
            LOG_ERROR("Thread %d: Core %d has invalid register address", thread_idx, i);
            rc = -1;
            continue;
        }
        if (platform_deinit_aicore_regs(core_reg_addrs_[i]) != 0) {
            LOG_ERROR("Thread %d: Core %d deinit timed out", thread_idx, i);
            rc = -1;
        }
        core_reg_addrs_[i] = 0;
    }
    LOG_INFO_V0("Thread %d: Shutdown complete", thread_idx);
    return rc;
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
        bool orch_cycle_started = false;
#endif
        // AICPU setup thread: initialize the shared distributed state, start
        // AICore workers, and wait for their on-core orchestration replay.
        {
            LOG_INFO_V0("Thread %d: Preparing shared dist state", thread_idx);
            const int32_t num_workers = runtime->worker_count;

            // Validate the stable Runtime prefix before reading DistHandoff or
            // PTO2Runtime state. A mismatch means the Host/AICPU images came
            // from different TensorMap artifact families. The shared backend
            // is intentionally fail-closed until its real implementation is
            // connected in the next stage.
            cache_invalidate_range(&runtime->fdwic_build_identity, sizeof(runtime->fdwic_build_identity));
            bool dispatch_dist_run =
                fdwic_build_identity_matches(runtime->fdwic_build_identity, static_cast<uint32_t>(sizeof(Runtime)));
            if (!dispatch_dist_run) {
                LOG_ERROR(
                    "Thread %d: FDWIC build identity mismatch "
                    "(magic=0x%" PRIx64 ", abi=%u, mode=%u, runtime_bytes=%u, dist_layout=%u; "
                    "expected abi=%u, mode=%u, runtime_bytes=%zu, dist_layout=%u)",
                    thread_idx, static_cast<uint64_t>(runtime->fdwic_build_identity.magic),
                    runtime->fdwic_build_identity.abi_version, runtime->fdwic_build_identity.tensor_map_mode,
                    runtime->fdwic_build_identity.runtime_bytes,
                    runtime->fdwic_build_identity.dist_global_layout_version, kFdwicBuildAbiVersion,
                    static_cast<uint32_t>(kFdwicCompiledTensorMapMode), sizeof(Runtime), kFdwicDistGlobalLayoutVersion
                );
                __atomic_fetch_or(
                    &runtime->fdwic_build_identity.error_bits, static_cast<uint32_t>(FdwicBuildErrorAicpuMismatch),
                    __ATOMIC_RELEASE
                );
                run_rc = -1;
            } else if (!kFdwicCompiledBackendReady) {
                LOG_ERROR(
                    "Thread %d: FDWIC shared TensorMap artifact is ABI-valid but its runtime backend "
                    "is not connected yet; aborting before Submit",
                    thread_idx
                );
                __atomic_fetch_or(
                    &runtime->fdwic_build_identity.error_bits, static_cast<uint32_t>(FdwicBuildErrorBackendUnavailable),
                    __ATOMIC_RELEASE
                );
                dispatch_dist_run = false;
                run_rc = -1;
            }
            cache_flush_range(&runtime->fdwic_build_identity, sizeof(runtime->fdwic_build_identity));

            if (!dispatch_dist_run) {
                for (int32_t i = 0; i < num_workers; i++) {
                    runtime->workers[i].aicpu_ready = AICPU_READY_DIST_ABORT;
                    cache_flush_range(const_cast<const uint32_t *>(&runtime->workers[i].aicpu_ready), sizeof(uint32_t));
                }
            } else {
                // Build the entry-arg once per run. The AICore-side dist engine
                // consumes it when replaying the linked orchestration entry.
                runtime->dist.orch_args.create_from_chip_args(runtime->get_orch_args());
                const ChipStorageTaskArgs &orch_args = runtime->get_orch_args();
                runtime->dist.ccec_orch_tensor_count = orch_args.tensor_count();
                runtime->dist.ccec_orch_scalar_count = orch_args.scalar_count();
                for (int32_t i = 0; i < orch_args.tensor_count() && i < CHIP_MAX_TENSOR_ARGS; i++) {
                    Tensor::copy(runtime->dist.ccec_orch_tensors[i], orch_args.tensor(i));
                }
                for (int32_t i = 0; i < orch_args.scalar_count() && i < CHIP_MAX_SCALAR_ARGS; i++) {
                    runtime->dist.ccec_orch_scalars[i] = orch_args.scalar(i);
                }
                cache_flush_range(runtime->dist.ccec_orch_tensors, sizeof(runtime->dist.ccec_orch_tensors));
                cache_flush_range(runtime->dist.ccec_orch_scalars, sizeof(runtime->dist.ccec_orch_scalars));
                cache_flush_range(
                    const_cast<const int32_t *>(&runtime->dist.ccec_orch_tensor_count), 2 * sizeof(int32_t)
                );

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
                // the pooled runtime_arena buffer. The distributed path reads the
                // PTO2Runtime header directly from that arena.
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
                orch_cycle_started = true;
#endif
                // ---- fully_distributed_within_core handoff ----
                // Instead of running orchestration here, wire the distributed engine
                // (resets cursors/flags/heap) and wake the AICore worker threads,
                // which call their own linked dist_core_main entry, replay
                // orchestration in SPMD fashion, and execute the tasks they win.
                // This AICPU thread then waits for all workers.
                // See runtime/dist_engine.* and docs/fully_distributed_within_core.md.
                {
                    const int32_t register_rc =
                        dist_engine_register(rt, &runtime->dist.orch_args, num_workers, runtime);
                    if (register_rc != 0) {
                        LOG_ERROR(
                            "Thread %d: distributed runtime configuration failed with status %d", thread_idx,
                            register_rc
                        );
                        dispatch_dist_run = false;
                        run_rc = register_rc;
                    }
                    runtime->dist.num_workers = num_workers;
                    __atomic_store_n(&runtime->dist.done_count, 0, __ATOMIC_RELEASE);
                    if (dispatch_dist_run) {
                        cache_flush_range(rt->dist_global, dist_engine_global_state_size());
                    }
                    cache_flush_range(&runtime->dist, sizeof(runtime->dist));
                    cache_flush_range(const_cast<const int32_t *>(&runtime->dist.num_workers), sizeof(int32_t));
                    cache_flush_range(const_cast<const int64_t *>(&runtime->dist.done_count), sizeof(int64_t));
                    FdwicSubmitPmuHeader *submit_pmu_header = nullptr;
                    const bool submit_pmu_requested =
                        dispatch_dist_run && fdwic_submit_pmu_requested(runtime, &submit_pmu_header);
                    if (submit_pmu_requested && fdwic_submit_pmu_owner_configure(runtime, submit_pmu_header) != 0) {
                        LOG_ERROR(
                            "Thread %d: FDWIC submit-PMU owner configure failed; aborting dist replay", thread_idx
                        );
                        // Configure 已经执行一次回滚；这里再做一次幂等恢复，优先
                        // 避免失败诊断给后续本地设备运行留下 PMU 配置。
                        (void)fdwic_submit_pmu_owner_restore(submit_pmu_header);
                        dispatch_dist_run = false;
                        run_rc = -1;
                    }
                    uint64_t *regs = reinterpret_cast<uint64_t *>(get_platform_regs());
                    if (regs == nullptr) {
                        LOG_ERROR("Thread %d: platform regs unavailable", thread_idx);
                        runtime_done_.store(true, std::memory_order_release);
                        return -1;
                    }
                    for (int32_t i = 0; i < num_workers; i++) {
                        const uint32_t physical_core_id = runtime->workers[i].physical_core_id;
                        write_reg(regs[physical_core_id], RegId::COND, AICORE_IDLE_VALUE);
                    }
                    for (int32_t i = 0; i < num_workers; i++) {
                        runtime->workers[i].aicpu_ready =
                            dispatch_dist_run ? AICPU_READY_DIST_RUN : AICPU_READY_DIST_ABORT;
                        cache_flush_range(
                            const_cast<const uint32_t *>(&runtime->workers[i].aicpu_ready), sizeof(uint32_t)
                        );
                    }
                    if (dispatch_dist_run) {
                        while (true) {
                            if (__atomic_load_n(&runtime->dist.done_count, __ATOMIC_ACQUIRE) >= num_workers) break;
                            bool all_cond_done = true;
                            for (int32_t i = 0; i < num_workers; i++) {
                                const uint32_t physical_core_id = runtime->workers[i].physical_core_id;
                                const uint64_t cond = read_reg(regs[physical_core_id], RegId::COND);
                                const bool done = cond == MAKE_FIN_VALUE(0);
                                all_cond_done = all_cond_done && done;
                            }
                            if (all_cond_done) {
                                __atomic_store_n(&runtime->dist.done_count, num_workers, __ATOMIC_RELEASE);
                                break;
                            }
                            SPIN_WAIT_HINT();
                        }
                        if (submit_pmu_requested) {
                            // 每个 worker 在发布 done 前已经 stop/read/flush 自己的
                            // 整窗记录及可选 phase sidecar。所有 worker 完成后才
                            // 恢复共享 PMU 配置。
                            const int restore_rc = fdwic_submit_pmu_owner_restore(submit_pmu_header);
                            if (restore_rc != 0) {
                                // ownership bitmap 保留失败槽，允许一次幂等重试；
                                // 即使重试恢复，首次失败也会留在 header，正式 raw
                                // 仍被 host 拒绝。
                                (void)fdwic_submit_pmu_owner_restore(submit_pmu_header);
                                run_rc = -1;
                            }
                        }
                        cache_invalidate_range(&runtime->fdwic_build_identity, sizeof(runtime->fdwic_build_identity));
                        if ((runtime->fdwic_build_identity.error_bits & FdwicBuildErrorAicoreMismatch) != 0) {
                            LOG_ERROR(
                                "Thread %d: at least one AICore rejected the FDWIC build identity; "
                                "failing the run after clean worker completion",
                                thread_idx
                            );
                            run_rc = -1;
                        }
                        const int32_t dist_status = dist_engine_runtime_status(rt);
                        if (dist_status != 0) {
                            LOG_ERROR(
                                "Thread %d: distributed AICore run failed with status %d", thread_idx, dist_status
                            );
                            run_rc = dist_status;
                        }
                    }
                }
            }
            run_status_.store(run_rc, std::memory_order_release);
            runtime_done_.store(true, std::memory_order_release);
        }
#if PTO2_PROFILING
        if (orch_cycle_started) {
            uint64_t orch_end_ts = get_sys_cnt_aicpu();
            LOG_INFO_V9(
                "Thread %d: orch_start=%" PRIu64 " orch_end=%" PRIu64 " orch_cost=%.3fus", thread_idx,
                static_cast<uint64_t>(orch_cycle_start), static_cast<uint64_t>(orch_end_ts),
                cycles_to_us(orch_end_ts - orch_cycle_start)
            );
        }
#endif
        LOG_INFO_V0("Thread %d: Orchestrator completed", thread_idx);
    } else {
        while (!runtime_done_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
    }

    const int32_t distributed_status = run_status_.load(std::memory_order_acquire);
    if (distributed_status != 0) run_rc = distributed_status;

    // AICPU launches multiple threads, but the orchestrator alone owns the
    // distributed handoff. Mirror its ABI/backend failure onto every AICPU
    // return value instead of relying on undocumented multi-block return-code
    // aggregation in the platform launcher.
    cache_invalidate_range(&runtime->fdwic_build_identity, sizeof(runtime->fdwic_build_identity));
    if (__atomic_load_n(&runtime->fdwic_build_identity.error_bits, __ATOMIC_ACQUIRE) != FdwicBuildErrorNone) {
        run_rc = -1;
    }

    // Shutdown is gated by runtime_done_: AICores only enter the EXIT wait
    // after completing the distributed replay, so sending EXIT earlier would
    // either race execution or time out while workers are still inside Phase 4.
    int32_t shutdown_rc = shutdown_cores(thread_idx);
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

    finished_count_.store(0, std::memory_order_release);
    runtime_done_.store(false, std::memory_order_release);
    run_status_.store(0, std::memory_order_release);

    aicpu_thread_num_ = 0;
    sched_thread_num_ = 0;
    core_count_ = 0;

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
