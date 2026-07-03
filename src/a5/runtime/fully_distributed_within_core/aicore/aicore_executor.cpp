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

#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "aicore/l2_swimlane_collector_aicore.h"
#include "aicore/pmu_collector_aicore.h"
#include "common/l2_swimlane_profiling.h"
#include "common/platform_config.h"  // Register-based communication
#include "common/pmu_profiling.h"
#include "pto2_dispatch_payload.h"
#include "runtime.h"

#if defined(__CPU_SIM)
extern "C" PTO_DEVICE_FUNC void aicore_dist_core_main(__gm__ Runtime *runtime, int core_idx, int core_type_int);
#endif

/**
 * Unified function pointer type for kernel dispatch
 *
 * All kernels follow the same signature: void kernel(__gm__ int64_t* args)
 * This enables simple, switch-free dispatch.
 */
typedef void (*UnifiedKernelFunc)(__gm__ int64_t *);

/**
 * Execute task from PTO2DispatchPayload.
 *
 * Reads function_bin_addr and args from the dispatch payload.
 *
 * @param payload Pointer to PTO2DispatchPayload in global memory
 */
__aicore__ __attribute__((always_inline)) static void execute_task(__gm__ PTO2DispatchPayload *payload) {
    if (payload == nullptr || payload->function_bin_addr == 0) {
        return;
    }

    UnifiedKernelFunc kernel = (UnifiedKernelFunc)payload->function_bin_addr;
    kernel(reinterpret_cast<__gm__ int64_t *>(payload->args));
    OUT_OF_ORDER_STORE_BARRIER();
}

/**
 * AICore main execution loop — fully_distributed_within_core variant.
 *
 * Instead of polling DATA_MAIN_BASE for AICPU-dispatched tasks, each AICore
 * worker invokes the distributed engine entry (compiled into the AICPU .so on
 * sim; onboard is the target of the subsequent CCEC migration, see
 * docs/fully_distributed_within_core.md). The engine replays the orchestration
 * submit stream, claims/builds the tasks it wins, and executes them; on return
 * it has set this worker's completion flags. The worker then honors the
 * existing teardown protocol (wait for EXIT, ack EXITED). AICPU sends EXIT
 * only after every worker has incremented Runtime::dist.done_count.
 *
 * Handshake phases 1-3 are preserved verbatim (register handshake, physical
 * core id publication, per-core dispatch payload cache). The trb-style
 * Phase 4 poll loop is replaced by a single dist_core_main invocation.
 *
 * Profiling state (get_aicore_profiling_flag etc.) is not touched here: on the
 * dist path, per-task profiling records are emitted from inside dist_core_main
 * / dist_engine_dump_trace on the AICPU side, so this loop does not need to
 * pre-resolve the L2 swimlane / PMU rings.
 *
 * @param runtime Pointer to Runtime in global memory
 * @param s_block_idx Block index (core ID)
 * @param core_type Core type (AIC or AIV)
 */
__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int s_block_idx, CoreType core_type) {
    __gm__ Handshake *my_hank = (__gm__ Handshake *)(&runtime->workers[s_block_idx]);

    // Phase 1: Wait for AICPU initialization signal
    while (my_hank->aicpu_ready == 0) {
        dcci(my_hank, SINGLE_CACHE_LINE);
        SPIN_WAIT_HINT();
    }

    // Phase 2: Report physical core ID, signal ready
    my_hank->physical_core_id = get_physical_core_id();
    OUT_OF_ORDER_STORE_BARRIER();
    my_hank->aicore_regs_ready = 1;
    dcci(&my_hank->aicore_regs_ready, SINGLE_CACHE_LINE, CACHELINE_OUT);
    while (my_hank->aicpu_regs_ready == 0) {
        dcci(&my_hank->aicpu_regs_ready, SINGLE_CACHE_LINE);
        SPIN_WAIT_HINT();
    }
    // Report initial idle status via register
    write_reg(RegId::COND, AICORE_IDLE_VALUE);

    // Phase 3: Report core type, signal ready
    my_hank->core_type = core_type;
    OUT_OF_ORDER_STORE_BARRIER();
    my_hank->aicore_done = s_block_idx + 1;  // Signal ready (use s_block_idx + 1 to avoid 0)

    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);

    // ===========================================================================
    // Phase 4 (dist): wait for the AICPU to publish the engine entry, invoke it
    // once, then hand off to the shutdown protocol below. The engine handles all
    // per-core orchestration replay, task claim/build/execute, and completion
    // flag publication internally; there is no per-task register handshake.
    // ===========================================================================
    while (runtime->dist.go == 0) {
        dcci(&runtime->dist, SINGLE_CACHE_LINE);
        SPIN_WAIT_HINT();
    }
#if defined(__CPU_SIM)
    aicore_dist_core_main(runtime, s_block_idx, static_cast<int>(core_type));
#else
    DistCoreMainFn core_main = reinterpret_cast<DistCoreMainFn>(runtime->dist.core_main_fn);
    if (core_main != nullptr) {
        core_main(runtime, s_block_idx, static_cast<int>(core_type));
    }
#endif

    // Teardown: wait for the AICPU EXIT signal on DATA_MAIN_BASE and ack.
    while (true) {
        uint32_t reg_val = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        if (reg_val == AICORE_EXIT_SIGNAL) {
            write_reg(RegId::COND, AICORE_EXITED_VALUE);
            break;
        }
        SPIN_WAIT_HINT();
    }
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);
}
