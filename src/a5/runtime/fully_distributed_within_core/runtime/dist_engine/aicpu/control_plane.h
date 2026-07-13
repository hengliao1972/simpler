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

// dist_engine_register is the AICPU control-plane entry point. It owns
// configuration reads and signal-handler installation; AICore images never
// include this file.

void dist_engine_register(PTO2Runtime *rt, const L2TaskArgs *orch_args, int num_workers, Runtime *runtime) {
    if (rt == nullptr || rt->dist_global == nullptr || rt->gm_heap == nullptr || rt->gm_heap_size == 0) {
        DIST_ERRF("[dist_engine] missing host-allocated runtime state\n");
        if (runtime != nullptr) runtime->dist.shared_addr = 0;
        return;
    }
    g_dist_ptr = reinterpret_cast<DistGlobal *>(rt->dist_global);
    g_dist.heap_base = static_cast<uint8_t *>(rt->gm_heap);
    g_dist.heap_size = rt->gm_heap_size;
    // Dependency-span bound H (R = F - H). Env override for graphs with longer
    // heap spans; default kHDefault.
    g_dist.H = kHDefault;
    if (const char *e = getenv("PTO_DIST_H")) {
        const long h = std::strtol(e, nullptr, 10);
        if (h >= 0) g_dist.H = static_cast<int32_t>(h);
    }
    // The producer map recycles a task's entry-head slot kTaskWindow tasks later;
    // cleanup retires a task once it leaves the H span, so H must stay below the
    // window (with margin) or a slot could be reused before its task is cleaned.
    always_assert(g_dist.H < kTaskWindow - 1);
#if DIST_SIM_HOST_CLOCK
    // Overhead-isolation gate (skip incore kernel calls, keep all bookkeeping).
    g_skip_exec = (getenv("PTO_DIST_SKIP_EXEC") != nullptr);
#endif

    for (int32_t s = 0; s < kCursorShards; s++) {
        atomic_exchange(g_dist.cube_cursor[s].v, int64_t{-1}, __ATOMIC_RELAXED);
        atomic_exchange(g_dist.vector_cursor[s].v, int64_t{-1}, __ATOMIC_RELAXED);
        atomic_exchange(g_dist.alloc_cursor[s].v, int64_t{-1}, __ATOMIC_RELAXED);
    }
    atomic_exchange(g_dist.frontier, int64_t{-1}, __ATOMIC_RELAXED);
    for (int32_t i = 0; i < kFlagCap; i++)
        reset_task_cell(i);
    atomic_exchange(g_dist.fatal, int32_t{0}, __ATOMIC_RELAXED);
    atomic_exchange(g_dist.replay_done, int64_t{0}, __ATOMIC_RELAXED);
    atomic_exchange(g_dist.started_count, int64_t{0}, __ATOMIC_RELAXED);
    g_dist.orch_args = orch_args;
    g_dist.rt = rt;
    g_dist.runtime = runtime;

    // Derive the physical-block topology (1 AIC + 2 AIV per block) the same way
    // the centralized scheduler discovers clusters: AIC/AIV cores in worker-index
    // order, AIC[b] paired with AIV[2b] (AIV0) and AIV[2b+1] (AIV1). Followers and
    // anchors use this to address block.won deposits. See §3.1.
    g_dist.num_workers = num_workers;
    int32_t aic_ids[RUNTIME_MAX_WORKER];
    int32_t aiv_ids[RUNTIME_MAX_WORKER];
    int32_t naic = 0, naiv = 0;
    for (int32_t i = 0; i < num_workers && i < RUNTIME_MAX_WORKER; i++) {
        g_dist.layout[i].block_id = -1;
        g_dist.layout[i].lane = LANE_NONE;
        if (runtime->workers[i].core_type == CoreType::AIC) {
            aic_ids[naic++] = i;
        } else {
            aiv_ids[naiv++] = i;
        }
    }
    g_dist.num_blocks = naic;
    for (int32_t b = 0; b < naic; b++) {
        g_dist.layout[aic_ids[b]] = CoreLayout{b, LANE_AIC};
        if (2 * b < naiv) g_dist.layout[aiv_ids[2 * b]] = CoreLayout{b, LANE_AIV0};
        if (2 * b + 1 < naiv) g_dist.layout[aiv_ids[2 * b + 1]] = CoreLayout{b, LANE_AIV1};
        atomic_exchange(g_dist.blocks[b].any_pub, int32_t{0}, __ATOMIC_RELAXED);
        for (int32_t s = 0; s < kPrivateSlots; s++) {
            atomic_exchange(g_dist.blocks[b].slots[s].state, int64_t{0}, __ATOMIC_RELAXED);
        }
    }

#if DIST_SIM_HOST_CLOCK
    fprintf(
        stderr, "[dist] register: num_workers=%d heap_base=%p heap_size=%zu\n", num_workers,
        reinterpret_cast<void *>(g_dist.heap_base), g_dist.heap_size
    );
#endif

#if DIST_SIM_HOST_CLOCK
    // Install the SIGUSR1 deadlock dumper once, but only when diagnostics are
    // opted in (PTO_DIST_WATCHDOG set) — default runs install no signal handler.
    static bool handler_installed = false;
    if (!handler_installed && getenv("PTO_DIST_WATCHDOG") != nullptr) {
        signal(SIGUSR1, dist_dump_state);
        handler_installed = true;
    }
#endif

    // Publish the DistGlobal struct address so AICore workers can wire up their
    // g_dist_ptr at dist_core_main entry. In sim the host BSS `g_dist` is shared
    // across every worker pthread, so its own address is what all AICore workers
    // see; onboard uses the host-allocated GM state address.
    runtime->dist.shared_addr = reinterpret_cast<uint64_t>(g_dist_ptr);

    // Publish all of the above before AICPU wakes workers through their
    // per-core handshake flags.
    store_barrier();
    return;
}
