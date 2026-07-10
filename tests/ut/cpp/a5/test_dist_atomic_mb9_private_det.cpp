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
// MB-9 private replica determinism — every core's private map is bit-identical.
//
// Re-implements the private DistTensorMap (ring-per-bucket) from
// dist_engine.cpp:727+ and verifies that N threads replaying the SAME
// deterministic submit sequence produce byte-identical lookup results at every
// task id. This is the correctness foundation of private mode: correctness does
// NOT come from locks — it comes from (a) only the owner core reads/writes its
// own DistCore, and (b) every core replays the same deterministic submit
// sequence so all replicas converge to identical content (only progress differs).
//
// This test verifies invariant (b): even under randomized scheduling (different
// per-task delays per thread), all replicas produce identical results.
//
// Criteria (docs MB-9):
//   1. All cores' DEPSIG (XOR of fan-in edges) are identical and == single-core ref
//   2. Every lookup at every task N returns the same producer on all cores
//   3. 1-core result == N-core result (deterministic regardless of core count)
//   4. Ring-per-bucket lookup == linked-list reference semantics (differential)
//
// Standalone:
//   g++ -O2 -std=c++17 -pthread -o /tmp/mb9 test_dist_atomic_mb9_private_det.cpp && /tmp/mb9
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

// ---- Constants (mirror dist_engine.cpp:649-657) ------------------------------
static constexpr int32_t kRingBuckets = 128;
static constexpr int32_t kRingBucketShift = 7;
static constexpr int32_t kBucketCapMax = 512;

static uint32_t hash_addr(uint64_t a) {
    a *= 0x9E3779B97F4A7C15ULL;
    return (uint32_t)(a >> (64 - kRingBucketShift));
}

// ---- Reference: former linked-list DistTensorMap::lookup semantics -----------
struct Ref {
    struct E { uint64_t addr, lo, hi; int32_t producer; };
    std::vector<E> es;
    int32_t alive_floor = 0;
    void reset() { es.clear(); alive_floor = 0; }
    void insert(uint64_t addr, uint64_t lo, uint64_t hi, int32_t p) { es.push_back({addr, lo, hi, p}); }
    void advance_retire(int32_t N, int32_t H) { int32_t f = N - H; if (f > alive_floor) alive_floor = f; }
    int32_t lookup(uint64_t addr, uint64_t lo, uint64_t hi) const {
        int32_t best = -1;
        for (const auto &e : es) {
            if (e.producer < alive_floor) continue;
            if (e.addr == addr && lo < e.hi && e.lo < hi && e.producer > best) best = e.producer;
        }
        return best;
    }
};

// ---- Private ring-per-bucket (mirror dist_engine.cpp DistTensorMap) -----------
struct PrivRing {
    struct Slot { uint64_t addr, lo, hi; int32_t producer; int32_t pad_; };
    static constexpr int32_t kStride = kBucketCapMax;
    std::vector<Slot> slots;
    std::vector<uint64_t> head, tail;
    int32_t cap, cap_mask, alive_floor = 0;
    bool overflowed = false;
    explicit PrivRing(int32_t c) : cap(c), cap_mask(c - 1) { reset(); }
    void reset() {
        slots.assign((size_t)kRingBuckets * kStride, Slot{});
        head.assign(kRingBuckets, 0);
        tail.assign(kRingBuckets, 0);
        alive_floor = 0;
        overflowed = false;
    }
    void advance_retire(int32_t N, int32_t H) { int32_t f = N - H; if (f > alive_floor) alive_floor = f; }
    void insert(uint64_t addr, uint64_t lo, uint64_t hi, int32_t p) {
        int32_t b = (int32_t)hash_addr(addr);
        while (head[b] < tail[b] && slots[(size_t)b * kStride + (int32_t)(head[b] & cap_mask)].producer < alive_floor)
            head[b]++;
        if (tail[b] - head[b] >= (uint64_t)cap) { overflowed = true; return; }
        Slot &s = slots[(size_t)b * kStride + (int32_t)(tail[b] & cap_mask)];
        s.addr = addr; s.lo = lo; s.hi = hi; s.producer = p;
        tail[b]++;
    }
    int32_t lookup(uint64_t addr, uint64_t lo, uint64_t hi) {
        int32_t b = (int32_t)hash_addr(addr);
        for (uint64_t k = tail[b]; k > head[b]; k--) {
            const Slot &s = slots[(size_t)b * kStride + (int32_t)((k - 1) & cap_mask)];
            if (s.producer < alive_floor) continue;
            if (s.addr == addr && lo < s.hi && s.lo < hi) return s.producer;
        }
        return -1;
    }
};

// ---- Op stream (deterministic, same on every core) ---------------------------
struct Region { uint64_t base, lo, hi; };
struct TaskOps { std::vector<Region> lookups; std::vector<Region> inserts; };

// ---- Per-core replay: each thread runs the SAME op stream on its OWN ring -----
struct CoreResult {
    uint64_t depsig = 0;  // XOR of (consumer_N, producer) edges — immune to ordering
    std::vector<std::vector<int32_t>> lookup_results;  // [N][j] -> producer
    bool overflowed = false;
};

void core_replay(const std::vector<TaskOps> &ops, int32_t num_tasks, int32_t H, int32_t cap,
                 uint32_t seed, CoreResult *out) {
    PrivRing ring(cap);
    out->lookup_results.resize(num_tasks);
    out->depsig = 0;
    out->overflowed = false;
    for (int32_t N = 0; N < num_tasks; N++) {
        ring.advance_retire(N, H);
        // Randomized delay (models PTO_DIST_FAKE_EXEC_NS — different per core)
        std::mt19937 rng(seed + N);
        for (volatile int d = 0; d < rng() % 128; d++) {}
        for (const auto &q : ops[N].lookups) {
            int32_t p = ring.lookup(q.base, q.lo, q.hi);
            out->lookup_results[N].push_back(p);
            if (p >= 0) out->depsig ^= ((uint64_t)N << 32) ^ (uint64_t)p;
        }
        for (const auto &o : ops[N].inserts) ring.insert(o.base, o.lo, o.hi, N);
    }
    out->overflowed = ring.overflowed;
}

bool test_determinism(int32_t num_tasks, int32_t num_threads, int32_t H, int32_t cap) {
    std::mt19937_64 rng(0xC0FFEE + num_tasks);
    uint64_t bases[24];
    for (auto &b : bases) b = (rng() & 0xFFFFF) << 12;
    auto rand_region = [&]() {
        Region r;
        r.base = bases[rng() % 24];
        r.lo = (rng() % 64) * 128;
        r.hi = r.lo + (1 + rng() % 8) * 128;
        return r;
    };

    std::vector<TaskOps> ops(num_tasks);
    for (int32_t N = 0; N < num_tasks; N++) {
        int nq = rng() % 4;
        for (int q = 0; q < nq; q++) ops[N].lookups.push_back(rand_region());
        int no = 1 + rng() % 3;
        for (int o = 0; o < no; o++) ops[N].inserts.push_back(rand_region());
    }

    // Single-core reference (also differential: ring == linked-list Ref).
    Ref ref;
    CoreResult single;
    core_replay(ops, num_tasks, H, cap, 0, &single);
    // Differential: ref (linked-list) == ring, in the SAME operation order as
    // replay (advance_retire → lookups → inserts per task N).
    ref.reset();
    int ring_vs_ref_mism = 0;
    for (int32_t N = 0; N < num_tasks; N++) {
        ref.advance_retire(N, H);
        for (size_t j = 0; j < ops[N].lookups.size(); j++) {
            int32_t r = ref.lookup(ops[N].lookups[j].base, ops[N].lookups[j].lo, ops[N].lookups[j].hi);
            if (r != single.lookup_results[N][j]) ring_vs_ref_mism++;
        }
        for (const auto &o : ops[N].inserts) ref.insert(o.base, o.lo, o.hi, N);
    }

    if (single.overflowed) {
        printf("  tasks=%d  OVERFLOW (cap=%d too small)\n", num_tasks, cap);
        return false;
    }

    // Multi-core: all threads replay the same stream on their own rings.
    std::vector<CoreResult> multi(num_threads);
    std::vector<std::thread> threads;
    for (int32_t t = 0; t < num_threads; t++)
        threads.emplace_back(core_replay, std::cref(ops), num_tasks, H, cap, (uint32_t)(t * 1000 + 1), &multi[t]);
    for (auto &th : threads) th.join();

    // Verify: all cores have same DEPSIG and same lookup results as single-core.
    bool depsig_ok = true;
    bool lookup_ok = true;
    int mism_count = 0;
    for (int32_t t = 0; t < num_threads; t++) {
        if (multi[t].depsig != single.depsig) depsig_ok = false;
        if (multi[t].overflowed) depsig_ok = false;
        for (int32_t N = 0; N < num_tasks && mism_count < 5; N++) {
            if (multi[t].lookup_results[N] != single.lookup_results[N]) {
                lookup_ok = false;
                mism_count++;
            }
        }
    }

    bool ok = depsig_ok && lookup_ok && (ring_vs_ref_mism == 0);
    printf("  tasks=%d threads=%d  depsig_ok=%d  lookup_ok=%d  ring==ref=%d  %s\n", num_tasks, num_threads,
           depsig_ok, lookup_ok, ring_vs_ref_mism == 0, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    int failures = 0;
    printf("=== MB-9 private replica determinism ===\n");

    printf("\n[Test 1] multi-core == single-core (deterministic replay):\n");
    for (int32_t nt : {2, 6, 24, 72}) {
        if (!test_determinism(4000, nt, 64, kBucketCapMax)) failures++;
    }

    printf("\n[Test 2] ring == linked-list reference (differential):\n");
    if (!test_determinism(8000, 1, 64, kBucketCapMax)) failures++;

    printf("\n[Test 3] tight H (aggressive retire, stress reclaim):\n");
    for (int32_t H : {4, 8, 16}) {
        if (!test_determinism(4000, 6, H, kBucketCapMax)) failures++;
    }

    printf("\n=== MB-9 %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
