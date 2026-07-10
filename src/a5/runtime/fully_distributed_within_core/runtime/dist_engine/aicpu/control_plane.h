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

// dist_engine_register / dist_engine_dump_trace are AICPU control-plane entry
// points. They own configuration reads, signal-handler installation, and the
// swimlane dumper; AICore images never include this file.

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
#if DIST_TRACE_ENABLED
    // Swimlane tracing gate. Capture the epoch now so every core's event ts is
    // relative to the same run start.
    g_trace_on = (getenv("PTO_DIST_SWIMLANE") != nullptr);
    g_trace_epoch_ns = now_ns();
    // Per-core span reserve: 0 when off (reset never reserves → zero overhead on a
    // normal run); a generous bound when on so push_back never reallocs for the
    // sizes we actually analyze (a realloc would perturb heap layout + add timing
    // noise to the very gaps we measure). Best-effort: a huge trace may still grow.
    g_trace_reserve = g_trace_on ? (1 << 16) : 0;
#endif
#if DIST_SIM_HOST_CLOCK
    // Overhead-isolation gate (skip incore kernel calls, keep all bookkeeping).
    g_skip_exec = (getenv("PTO_DIST_SKIP_EXEC") != nullptr);
#endif

    for (int32_t s = 0; s < kCursorShards; s++) {
        atom_store(g_dist.cube_cursor[s].v, -1, __ATOMIC_RELAXED);
        atom_store(g_dist.vector_cursor[s].v, -1, __ATOMIC_RELAXED);
        atom_store(g_dist.alloc_cursor[s].v, -1, __ATOMIC_RELAXED);
    }
    atom_store(g_dist.frontier, -1, __ATOMIC_RELAXED);
    for (int32_t i = 0; i < kFlagCap; i++)
        reset_task_cell(i);
    atom_store(g_dist.fatal, 0, __ATOMIC_RELAXED);
    atom_store<int64_t>(g_dist.replay_done, 0, __ATOMIC_RELAXED);
    atom_store<int64_t>(g_dist.started_count, 0, __ATOMIC_RELAXED);
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
        atom_store(g_dist.blocks[b].any_pub, 0, __ATOMIC_RELAXED);
        for (int32_t s = 0; s < kPrivateSlots; s++) {
            atom_store(g_dist.blocks[b].slots[s].state, 0, __ATOMIC_RELAXED);
        }
    }

    if (dist_trace()) {
        fprintf(
            stderr, "[dist] register: num_workers=%d heap_base=%p heap_size=%zu\n", num_workers,
            reinterpret_cast<void *>(g_dist.heap_base), g_dist.heap_size
        );
    }

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
    atom_thread_fence(__ATOMIC_RELEASE);
    return;
}

#if DIST_TRACE_ENABLED
void dist_engine_dump_trace() {
    if (!g_trace_on) return;
    const char *path = getenv("PTO_DIST_SWIMLANE");
    if (path == nullptr || path[0] == '\0') return;
    FILE *f = fopen(path, "w");
    if (f == nullptr) {
        fprintf(stderr, "[dist_engine] cannot open swimlane file %s for write\n", path);
        return;
    }

    auto lane_name = [](int32_t lane) -> const char * {
        switch (lane) {
        case LANE_AIC:
            return "AIC";
        case LANE_AIV0:
            return "AIV0";
        case LANE_AIV1:
            return "AIV1";
        default:
            return "?";
        }
    };

    // Chrome Trace Event Format (https://ui.perfetto.dev / chrome://tracing).
    // Two process groups: pid = block_id is the WALL-clock swimlane; pid =
    // block_id + kCpuPid is a parallel CPU-time swimlane (same spans, width =
    // cpu_us). process_sort_index forces all wall groups above all cpu groups.
    // Dependency arrows (flow events) are emitted only in the cpu group so they
    // stay within the cpu lanes instead of tangling across the wall lanes.
    constexpr int32_t kCpuPid = 1000;
    fprintf(f, "{\n  \"displayTimeUnit\": \"ns\",\n  \"traceEvents\": [\n");
    bool first = true;
    const int32_t nw = g_dist.num_workers;

    // Lane/process name + sort metadata first (so idle lanes still appear).
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        if (!first) fprintf(f, ",\n");
        first = false;
        fprintf(
            f, "    {\"ph\":\"M\",\"name\":\"process_name\",\"pid\":%d,\"args\":{\"name\":\"block%d (wall)\"}}",
            co.block_id, co.block_id
        );
        fprintf(
            f, ",\n    {\"ph\":\"M\",\"name\":\"process_sort_index\",\"pid\":%d,\"args\":{\"sort_index\":%d}}",
            co.block_id, co.block_id
        );
        fprintf(
            f,
            ",\n    {\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":%d,\"tid\":%d,"
            "\"args\":{\"name\":\"%s (core%d)\"}}",
            co.block_id, co.lane, lane_name(co.lane), c
        );
        fprintf(
            f, ",\n    {\"ph\":\"M\",\"name\":\"process_name\",\"pid\":%d,\"args\":{\"name\":\"block%d (cpu)\"}}",
            co.block_id + kCpuPid, co.block_id
        );
        fprintf(
            f, ",\n    {\"ph\":\"M\",\"name\":\"process_sort_index\",\"pid\":%d,\"args\":{\"sort_index\":%d}}",
            co.block_id + kCpuPid, co.block_id + kCpuPid
        );
        fprintf(
            f,
            ",\n    {\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":%d,\"tid\":%d,"
            "\"args\":{\"name\":\"%s (core%d)\"}}",
            co.block_id + kCpuPid, co.lane, lane_name(co.lane), c
        );
        // CPU-group kernel sub-lane (tid = lane + 3): kernel spans live here so a
        // ringbp bar that time-contains its releasing kernel does not nest+hide it.
        fprintf(
            f,
            ",\n    {\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":%d,\"tid\":%d,"
            "\"args\":{\"name\":\"%s·kernel (core%d)\"}}",
            co.block_id + kCpuPid, co.lane + 3, lane_name(co.lane), c
        );
    }

    auto phase_name = [](TracePhase p) -> const char * {
        switch (p) {
        case TracePhase::Kernel:
            return "kernel";
        case TracePhase::Alloc:
            return "alloc";
        case TracePhase::Build:
            return "build";
        case TracePhase::DrainWon:
            return "drain_won";
        case TracePhase::Replay:
            return "replay";
        case TracePhase::RingBp:
            return "ringbp";
        case TracePhase::EfDrain:
            return "efdrain";
        case TracePhase::Commit:
            return "commit";
        default:
            return "?";
        }
    };

    // Index: task_id -> its kernel span location in the CPU group, so a dep edge
    // can anchor an arrow at the producer's and consumer's actual spans.
    struct SpanLoc {
        int32_t pid;
        int32_t tid;
        double ts_us;
        double dur_us;
    };
    // In the CPU group, kernel spans go on a SEPARATE sub-lane (tid = lane +
    // kCpuKernelLane) from the build/ringbp/replay/alloc spans (tid = lane). A
    // ringbp span time-contains the kernel that ends its wait, so on one lane
    // perfetto would nest the kernel inside the ringbp bar and hide it; splitting
    // the kernel onto its own row keeps both visible.
    constexpr int32_t kCpuKernelLane = 3;
    std::vector<SpanLoc> kloc(static_cast<size_t>(kFlagCap), SpanLoc{-1, -1, 0.0, 0.0});
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        DistCoreTraceState &ts = g_trace_cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const TraceEvent &e : ts.trace) {
            if (e.phase != TracePhase::Kernel || e.task_id < 0 || e.task_id >= kFlagCap) continue;
            kloc[static_cast<size_t>(e.task_id)] =
                SpanLoc{co.block_id + kCpuPid, co.lane + kCpuKernelLane, e.ts_ns / 1000.0, e.cpu_ns / 1000.0};
        }
    }
    // Index: task_id -> its ringbp span in the CPU group (the arrow head for a
    // slot-release edge anchors at the ringbp's END = when the wait was satisfied).
    std::vector<SpanLoc> rbloc(static_cast<size_t>(kFlagCap), SpanLoc{-1, -1, 0.0, 0.0});
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        DistCoreTraceState &ts = g_trace_cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const TraceEvent &e : ts.trace) {
            if (e.phase != TracePhase::RingBp || e.task_id < 0 || e.task_id >= kFlagCap) continue;
            rbloc[static_cast<size_t>(e.task_id)] =
                SpanLoc{co.block_id + kCpuPid, co.lane, e.ts_ns / 1000.0, e.cpu_ns / 1000.0};
        }
    }

    // Duration events: kernel + non-kernel overhead spans, emitted once in the
    // wall group (pid=block) and once in the cpu group (pid=block+kCpuPid).
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        DistCoreTraceState &ts = g_trace_cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const TraceEvent &e : ts.trace) {
            const char *ph = phase_name(e.phase);
            char name[64];
            if (e.phase != TracePhase::Kernel) {
                snprintf(name, sizeof(name), "%s#%d", ph, e.task_id);
            } else if (e.func_id >= 0) {
                snprintf(name, sizeof(name), "f%d#%d", e.func_id, e.task_id);
            } else {
                snprintf(name, sizeof(name), "task#%d", e.task_id);
            }
            if (!first) fprintf(f, ",\n");
            first = false;
            // Convert raw ns -> us (swimlane unit) here, at dump time — never on the
            // hot path (see TraceEvent).
            const double ts_us = e.ts_ns / 1000.0;
            const double dur_us = e.dur_ns / 1000.0;
            const double cpu_us = e.cpu_ns / 1000.0;
            fprintf(
                f,
                "    {\"ph\":\"X\",\"name\":\"%s\",\"pid\":%d,\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f,"
                "\"args\":{\"phase\":\"%s\",\"task_id\":%d,\"func_id\":%d,\"core\":%d,\"mc\":%d,\"cpu_us\":%.3f}}",
                name, co.block_id, co.lane, ts_us, dur_us, ph, e.task_id, e.func_id, c, e.multicore, cpu_us
            );
            fprintf(
                f,
                ",\n    {\"ph\":\"X\",\"name\":\"%s\",\"pid\":%d,\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f,"
                "\"args\":{\"phase\":\"%s\",\"task_id\":%d,\"func_id\":%d,\"wall_us\":%.3f}}",
                name, co.block_id + kCpuPid, e.phase == TracePhase::Kernel ? co.lane + kCpuKernelLane : co.lane, ts_us,
                cpu_us, ph, e.task_id, e.func_id, dur_us
            );
        }
    }

    // Flow events: the full static dependency graph. One arrow per dep edge, in
    // the cpu group, from the PRODUCER kernel span's end to the CONSUMER kernel
    // span's start (time always forward: a producer completes before its consumer
    // runs). Click any task and follow arrows backward hop-by-hop to walk the
    // chain "what was this waiting on, and what was THAT waiting on".
    int32_t flow_id = 0;
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        DistCoreTraceState &ts = g_trace_cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const DepEdge &de : ts.dep_edges) {
            if (de.producer_task < 0 || de.producer_task >= kFlagCap) continue;
            if (de.consumer_task < 0 || de.consumer_task >= kFlagCap) continue;
            const SpanLoc &pr = kloc[static_cast<size_t>(de.producer_task)];
            const SpanLoc &cs = kloc[static_cast<size_t>(de.consumer_task)];
            if (pr.pid < 0 || cs.pid < 0) continue;  // need both kernel spans
            fprintf(
                f, ",\n    {\"ph\":\"s\",\"name\":\"dep\",\"cat\":\"dep\",\"id\":%d,\"pid\":%d,\"tid\":%d,\"ts\":%.3f}",
                flow_id, pr.pid, pr.tid, pr.ts_us + pr.dur_us
            );
            fprintf(
                f,
                ",\n    {\"ph\":\"f\",\"name\":\"dep\",\"cat\":\"dep\",\"id\":%d,\"bp\":\"e\",\"pid\":%d,\"tid\":%d,"
                "\"ts\":%.3f}",
                flow_id, cs.pid, cs.tid, cs.ts_us
            );
            flow_id++;
        }
    }

    // Flow events (cat="slot"): slot-release edges that explain a ringbp's stall.
    // From the END of the occupant kernel's span (the moment it frees the slot) to
    // the END of the waiting ringbp span. Chains with the dep arrows: ringbp
    // --slot--> occupant kernel --dep--> the occupant's fan-in kernels.
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        DistCoreTraceState &ts = g_trace_cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const DepEdge &se : ts.slot_edges) {
            if (se.producer_task < 0 || se.producer_task >= kFlagCap) continue;  // occupant
            if (se.consumer_task < 0 || se.consumer_task >= kFlagCap) continue;  // ringbp waiter
            const SpanLoc &occ = kloc[static_cast<size_t>(se.producer_task)];
            const SpanLoc &rb = rbloc[static_cast<size_t>(se.consumer_task)];
            if (occ.pid < 0 || rb.pid < 0) continue;
            double tail = occ.ts_us + occ.dur_us;      // occupant kernel end (slot freed)
            const double head = rb.ts_us + rb.dur_us;  // ringbp end (wait satisfied)
            if (tail > head) tail = head;              // keep forward in time
            fprintf(
                f,
                ",\n    {\"ph\":\"s\",\"name\":\"slot\",\"cat\":\"slot\",\"id\":%d,\"pid\":%d,\"tid\":%d,\"ts\":%.3f}",
                flow_id, occ.pid, occ.tid, tail
            );
            fprintf(
                f,
                ",\n    {\"ph\":\"f\",\"name\":\"slot\",\"cat\":\"slot\",\"id\":%d,\"bp\":\"e\",\"pid\":%d,\"tid\":%d,"
                "\"ts\":%.3f}",
                flow_id, rb.pid, rb.tid, head
            );
            flow_id++;
        }
    }

    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    fprintf(stderr, "[dist_engine] swimlane trace written to %s\n", path);
}
#else   // !DIST_TRACE_ENABLED
// Tracing compiled out: keep the public symbol so aicpu_executor.cpp still links.
void dist_engine_dump_trace() {}
#endif  // DIST_TRACE_ENABLED
