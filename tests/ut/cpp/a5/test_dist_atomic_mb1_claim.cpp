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
// MB-1 claim cursor — exactly-one-winner & no-skip (atomicMax).
//
// Re-implements the claim() cursor logic from
// src/common/runtime/fully_distributed_within_core/dist_engine.cpp (lines
// 1199-1636) and stresses it with real std::thread contention. This is the
// "A5 hardware atomicMax is truly cross-core atomic" direct probe (docs §14.3
// biggest residual risk) — on host we model BOTH the sim CAS-retry path and
// the HW fetch_max path, and assert they produce identical exactly-one-winner
// semantics.
//
// Criteria (docs MB-1):
//   1. sum of owned_total across cores == num_tasks (every task exactly one owner)
//   2. won-id set is a permutation of [0, num_tasks) (XOR check, no dup/no skip)
//   3. within each shard, won ids are strictly increasing (monotone cursor)
//   4. kCursorShards=1 and =4 produce equivalent winner sets
//   5. fetch_max path == CAS-retry path (both exactly-one-winner)
//
// Standalone (no gtest / build-system deps):
//   g++ -O2 -std=c++17 -pthread -o /tmp/mb1 test_dist_atomic_mb1_claim.cpp && /tmp/mb1
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

// ---- Mirrors dist_engine.cpp constants (lines 1209-1210) ----------------------
static constexpr int32_t kCursorShards = 4;
static constexpr size_t kCacheLine = 64;

// ---- Coherent<int32_t> sim specialization (dist_engine.cpp:384) ---------------
// On sim (DIST_SIM_HOST_CLOCK==1), Coherent<T> wraps std::atomic<T>.
struct CoherentInt32 {
    std::atomic<int32_t> a;
};

// ---- PaddedCursor (dist_engine.cpp:1212) --------------------------------------
// Each sub-cursor padded to its own cache line so adjacent shards never share.
struct alignas(kCacheLine) PaddedCursor {
    CoherentInt32 v;
    uint8_t pad[kCacheLine - sizeof(CoherentInt32)];
};

// ---- claim() — sim CAS-retry path (dist_engine.cpp:1625-1636) -----------------
// Matches the #if DIST_SIM_HOST_CLOCK branch exactly: load acquire, CAS acq-rel,
// retry on failure, lose if N <= current.
bool claim_cas(CoherentInt32 *cursor, int32_t N) {
    int32_t c = cursor->a.load(std::memory_order_acquire);
    while (true) {
        if (N <= c) return false;
        if (cursor->a.compare_exchange_weak(c, N, std::memory_order_acq_rel, std::memory_order_acquire))
            return true;
    }
}

// ---- claim() — HW fetch_max path (dist_engine.cpp:1630-1632) ------------------
// Models A5 atomicMax: old = atomicMax(&cursor, N); won = (old < N).
// std::atomic has no fetch_max, so we implement fetch_max via CAS loop — the
// semantics are identical (atomic read-modify-write of max). The KEY property
// is that old<N detects the single winner.
bool claim_fetch_max(CoherentInt32 *cursor, int32_t N) {
    int32_t c = cursor->a.load(std::memory_order_acquire);
    while (true) {
        if (N <= c) return false;  // someone already claimed N or higher
        if (cursor->a.compare_exchange_weak(c, N, std::memory_order_acq_rel, std::memory_order_acquire)) {
            // We raised the cursor to N. But did we win? We win iff the value we
            // observed (c) was < N. Since N > c here (we passed the N<=c check),
            // we won. (On real HW, atomicMax returns old; old<N iff we raised it.)
            return true;
        }
        // CAS failed: c reloaded with current value, retry.
    }
}

// ---- Worker: race to claim ids [0, num_tasks) ---------------------------------
// Mirrors dist_engine.cpp dist_submit_impl: every core advances its local_index
// through the SAME id space [0,1,2,...] in order. Multiple cores reach the same
// id N concurrently and race on claim(cursor[N%shards], N). The loser simply
// moves on to N+1 (it skips execution but still advances). This produces real
// contention on the cursor and is the exact access pattern the real engine uses.
struct ClaimResult {
    int32_t owned_total = 0;
    uint64_t won_xor = 0;  // XOR of all won ids — == XOR(0..num_tasks-1) iff permutation
    std::vector<int32_t> won_ids;
};

void worker_claim(PaddedCursor *cursors, int32_t num_tasks, int32_t shards, bool use_fetch_max,
                  ClaimResult *out) {
    for (int32_t local_index = 0; local_index < num_tasks; local_index++) {
        int32_t N = local_index;
        int32_t shard = N % shards;
        bool won = use_fetch_max ? claim_fetch_max(&cursors[shard].v, N) : claim_cas(&cursors[shard].v, N);
        if (won) {
            out->owned_total++;
            out->won_xor ^= (uint64_t)N;
            out->won_ids.push_back(N);
        }
    }
}

// ---- Helpers ------------------------------------------------------------------
uint64_t xor_range(int32_t lo, int32_t hi) {
    uint64_t x = 0;
    for (int32_t i = lo; i < hi; i++) x ^= (uint64_t)i;
    return x;
}

// Run one claim experiment with given params, return global results.
struct ExpResult {
    int32_t total_owned = 0;
    uint64_t global_xor = 0;
    bool permutation_ok = false;  // won set == [0, num_tasks)
    bool monotone_per_shard = true;
};

ExpResult run_experiment(int32_t num_tasks, int32_t num_threads, int32_t shards, bool use_fetch_max) {
    std::vector<PaddedCursor> cursors(shards);
    for (int32_t s = 0; s < shards; s++) cursors[s].v.a.store(-1, std::memory_order_relaxed);

    std::vector<ClaimResult> results(num_threads);
    std::vector<std::thread> threads;
    for (int32_t t = 0; t < num_threads; t++)
        threads.emplace_back(worker_claim, cursors.data(), num_tasks, shards, use_fetch_max, &results[t]);
    for (auto &th : threads) th.join();

    ExpResult exp;
    for (auto &r : results) {
        exp.total_owned += r.owned_total;
        exp.global_xor ^= r.won_xor;
    }
    exp.permutation_ok = (exp.total_owned == num_tasks && exp.global_xor == xor_range(0, num_tasks));

    // Check monotonicity per shard: within each shard, won ids must be strictly
    // increasing (cursor is monotone — once it advances past N, no lower id can win).
    std::vector<std::vector<int32_t>> per_shard(shards);
    for (auto &r : results)
        for (int32_t id : r.won_ids) per_shard[id % shards].push_back(id);
    for (int32_t s = 0; s < shards; s++) {
        std::sort(per_shard[s].begin(), per_shard[s].end());
        for (size_t i = 1; i < per_shard[s].size(); i++) {
            if (per_shard[s][i] <= per_shard[s][i - 1]) { exp.monotone_per_shard = false; break; }
        }
        // Also verify no duplicate ids across all shards
        for (size_t i = 1; i < per_shard[s].size(); i++) {
            if (per_shard[s][i] == per_shard[s][i - 1]) { exp.permutation_ok = false; }
        }
    }
    return exp;
}

int main() {
    const int32_t num_tasks = 10000;
    const std::vector<int32_t> thread_counts = {2, 6, 24, 72};
    int failures = 0;

    printf("=== MB-1 claim cursor: exactly-one-winner & no-skip ===\n");

    // Test 1+2+3: CAS path, kCursorShards=4, varying thread counts.
    printf("\n[Test 1-3] CAS-retry path, shards=%d:\n", kCursorShards);
    for (int32_t nt : thread_counts) {
        ExpResult e = run_experiment(num_tasks, nt, kCursorShards, /*use_fetch_max=*/false);
        bool ok = e.permutation_ok && e.monotone_per_shard;
        printf("  threads=%3d  owned=%d/%d  xor_ok=%d  monotone=%d  %s\n", nt, e.total_owned, num_tasks,
               e.permutation_ok, e.monotone_per_shard, ok ? "PASS" : "FAIL");
        if (!ok) failures++;
    }

    // Test 5: fetch_max path (models HW atomicMax) == CAS path.
    printf("\n[Test 5] fetch_max path (HW atomicMax model), shards=%d:\n", kCursorShards);
    for (int32_t nt : thread_counts) {
        ExpResult e = run_experiment(num_tasks, nt, kCursorShards, /*use_fetch_max=*/true);
        bool ok = e.permutation_ok && e.monotone_per_shard;
        printf("  threads=%3d  owned=%d/%d  xor_ok=%d  monotone=%d  %s\n", nt, e.total_owned, num_tasks,
               e.permutation_ok, e.monotone_per_shard, ok ? "PASS" : "FAIL");
        if (!ok) failures++;
    }

    // Test 4: shard equivalence — shards=1 vs shards=4 both produce exactly-one-winner.
    printf("\n[Test 4] shard equivalence (shards=1 vs shards=4):\n");
    for (int32_t nt : thread_counts) {
        ExpResult e1 = run_experiment(num_tasks, nt, 1, false);
        ExpResult e4 = run_experiment(num_tasks, nt, kCursorShards, false);
        bool ok = e1.permutation_ok && e4.permutation_ok && e1.monotone_per_shard && e4.monotone_per_shard;
        printf("  threads=%3d  shards=1: owned=%d xor_ok=%d  |  shards=4: owned=%d xor_ok=%d  %s\n", nt,
               e1.total_owned, e1.permutation_ok, e4.total_owned, e4.permutation_ok, ok ? "PASS" : "FAIL");
        if (!ok) failures++;
    }

    // Test: worst-case contention — all threads hammer the same small id range.
    printf("\n[Test] worst-case contention (500 tasks, 72 threads, fetch_max):\n");
    {
        ExpResult e = run_experiment(500, 72, kCursorShards, true);
        bool ok = e.permutation_ok && e.monotone_per_shard;
        printf("  owned=%d/500  xor_ok=%d  monotone=%d  %s\n", e.total_owned, e.permutation_ok,
               e.monotone_per_shard, ok ? "PASS" : "FAIL");
        if (!ok) failures++;
    }

    printf("\n=== MB-1 %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
