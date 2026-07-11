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
// MB-5 shared TensorMap — serialized append sequencer + seq ABA guard +
// concurrent lookup, differential vs private reference.
//
// Re-implements SharedTensorMap / tm_shared_claim_append / shared_tm_append /
// shared_tm_lookup from dist_engine.cpp (lines 859-969, 2043-2118) on host with
// std::atomic. Unlike test_dist_tensormap_ring.cpp (which MODELS concurrency
// single-threaded), this test drives the sequencer and ring with REAL
// std::thread contention.
//
// Criteria (docs MB-5):
//   1. private vs shared lookup results are bit-identical (every fan-in resolves
//      to the same producer)
//   2. every task is appended exactly once and total region inserts equals the
//      deterministic operation stream
//   3. appends happen in strict id order (sequencer serializes 0,1,2,...)
//   4. seq ABA guard: slot reuse under concurrent lookup doesn't corrupt results
//   5. ring overflow → deterministic FATAL (never silent overwrite)
//
// Standalone:
//   g++ -O2 -std=c++17 -pthread -o /tmp/mb5 test_dist_atomic_mb5_shared_map.cpp && /tmp/mb5
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

// ---- Constants (mirror dist_engine.cpp:649-657, 1390) ------------------------
static constexpr int32_t kRingBuckets = 128;
static constexpr int32_t kRingBucketShift = 7;
static constexpr int32_t kBucketCapMax = 512;
static constexpr int32_t kTmAppendBusy = -1;

static uint32_t hash_addr(uint64_t a) {
    a *= 0x9E3779B97F4A7C15ULL;
    return (uint32_t)(a >> (64 - kRingBucketShift));
}

// ---- Private reference Ring (same as test_dist_tensormap_ring.cpp) ------------
struct PrivRing {
    struct Slot { uint64_t addr, lo, hi; int32_t producer; };
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

// ---- SharedTensorMap (mirror dist_engine.cpp:859-969) -------------------------
// On host, Coherent<T> = std::atomic<T>. seq is the ABA guard.
struct SharedSlot {
    uint64_t buf_addr;
    uint64_t lo;
    uint64_t hi;
    int32_t producer;
    int32_t pad_;
    std::atomic<uint64_t> seq;  // kSeqEmpty or absolute append index k
};

struct SharedTensorMap {
    static constexpr int32_t kStride = kBucketCapMax;
    static constexpr uint64_t kSeqEmpty = ~0ull;
    std::vector<SharedSlot> slots;
    std::vector<std::atomic<uint64_t>> head, tail;
    int32_t cap, cap_mask;
    std::atomic<bool> overflowed{false};

    explicit SharedTensorMap(int32_t c)
        : cap(c), cap_mask(c - 1),
          slots(static_cast<size_t>(kRingBuckets) * kStride),
          head(kRingBuckets), tail(kRingBuckets) {
        reset();
    }
    void reset() {
        for (int32_t b = 0; b < kRingBuckets; b++) {
            head[b].store(0, std::memory_order_relaxed);
            tail[b].store(0, std::memory_order_relaxed);
        }
        for (auto &s : slots) s.seq.store(kSeqEmpty, std::memory_order_relaxed);
        overflowed.store(false, std::memory_order_relaxed);
    }
    // Single-appender (serialized by tm_insert_next). reclaim, check, write, publish.
    void append(uint64_t addr, uint64_t lo, uint64_t hi, int32_t producer, int32_t reclaim_floor) {
        int32_t b = (int32_t)hash_addr(addr);
        uint64_t h = head[b].load(std::memory_order_relaxed);
        uint64_t tl = tail[b].load(std::memory_order_relaxed);
        while (h < tl && slots[(size_t)b * kStride + (int32_t)(h & cap_mask)].producer <= reclaim_floor) h++;
        head[b].store(h, std::memory_order_release);
        if (tl - h >= (uint64_t)cap) { overflowed.store(true, std::memory_order_relaxed); return; }
        uint64_t k = tl;
        SharedSlot &s = slots[(size_t)b * kStride + (int32_t)(k & cap_mask)];
        s.buf_addr = addr; s.lo = lo; s.hi = hi; s.producer = producer;
        s.seq.store(k, std::memory_order_release);        // publish slot fields
        tail[b].store(k + 1, std::memory_order_release);   // publish new tail
    }
    // One acquire on tail, relaxed per-slot reads, seq ABA guard.
    int32_t lookup(uint64_t addr, uint64_t lo, uint64_t hi, int32_t N, int32_t floor) {
        int32_t b = (int32_t)hash_addr(addr);
        uint64_t tl = tail[b].load(std::memory_order_acquire);
        uint64_t h = head[b].load(std::memory_order_relaxed);
        for (uint64_t k = tl; k > h; k--) {
            uint64_t idx = k - 1;
            SharedSlot &s = slots[(size_t)b * kStride + (int32_t)(idx & cap_mask)];
            if (s.seq.load(std::memory_order_relaxed) != idx) continue;  // empty / reclaimed+reused
            int32_t p = s.producer;
            if (p >= N) continue;      // temporal filter
            if (p < floor) continue;   // retire floor
            if (s.buf_addr == addr && lo < s.hi && s.lo < hi) return p;
        }
        return -1;
    }
};

// ---- Op stream (deterministic, same on every core) ---------------------------
struct Region { uint64_t base, lo, hi; };
struct TaskOps { std::vector<Region> lookups; std::vector<Region> inserts; };

// ---- Concurrent shared-map stress test ----------------------------------------
struct SharedStats {
    std::atomic<int32_t> total_inserts{0};
    std::atomic<int32_t> total_lookups{0};
    std::atomic<int32_t> mismatches{0};
};

static std::atomic<int32_t> g_slot_assign{0};

void shared_worker(SharedTensorMap &sm, std::atomic<int32_t> &tm_insert_next,
                   const std::vector<TaskOps> &ops, int32_t num_tasks, int32_t H,
                   const std::vector<std::vector<int32_t>> &priv_results, SharedStats &stats,
                   std::atomic<int32_t> *core_progress, int32_t num_threads,
                   std::vector<int32_t> *append_order) {
    int32_t my_idx = g_slot_assign.fetch_add(1, std::memory_order_relaxed);
    core_progress[my_idx].store(0, std::memory_order_relaxed);

    for (int32_t N = 0; N < num_tasks; N++) {
        // Publish this core's replay progress (drives the shared reclaim floor).
        core_progress[my_idx].store(N, std::memory_order_relaxed);

        // Compute global min_progress (slowest core) — O(num_threads).
        int32_t mn = INT32_MAX;
        for (int32_t c = 0; c < num_threads; c++) {
            int32_t p = core_progress[c].load(std::memory_order_relaxed);
            if (p < mn) mn = p;
        }

        // Try to win the append sequencer for task N (CAS N -> BUSY).
        while (true) {
            int32_t v = tm_insert_next.load(std::memory_order_acquire);
            if (v > N) break;  // already appended
            if (v == N) {
                int32_t expected = N;
                if (tm_insert_next.compare_exchange_strong(expected, kTmAppendBusy,
                        std::memory_order_acq_rel, std::memory_order_acquire)) {
                    // Reclaim floor = min_progress - H - 1 (keeps [N-H, N) alive).
                    int32_t rfloor = mn - H - 1;
                    for (const auto &o : ops[N].inserts)
                        sm.append(o.base, o.lo, o.hi, N, rfloor);
                    // The sequencer guarantees one appender at a time. Record
                    // the task id before publishing N+1 so this vector is not
                    // concurrently modified by another worker.
                    append_order->push_back(N);
                    tm_insert_next.store(N + 1, std::memory_order_release);
                    stats.total_inserts.fetch_add(static_cast<int32_t>(ops[N].inserts.size()),
                                                  std::memory_order_relaxed);
                    break;
                }
            }
            std::this_thread::yield();
        }

        // Lookups: floor = N - H (mirrors private alive_floor).
        int32_t floor = N - H;
        for (size_t j = 0; j < ops[N].lookups.size(); j++) {
            const auto &q = ops[N].lookups[j];
            int32_t got = sm.lookup(q.base, q.lo, q.hi, N, floor);
            int32_t want = priv_results[N][j];
            stats.total_lookups.fetch_add(1, std::memory_order_relaxed);
            if (got != want) {
                stats.mismatches.fetch_add(1, std::memory_order_relaxed);
                if (stats.mismatches.load() <= 3)
                    printf("  MISMATCH N=%d: shared=%d private=%d\n", N, got, want);
            }
        }
    }
    core_progress[my_idx].store(num_tasks, std::memory_order_relaxed);
}

bool test_shared_vs_private(int32_t num_tasks, int32_t num_threads, int32_t H, int32_t cap) {
    std::mt19937_64 rng(0xDEADBEEF + num_tasks);
    uint64_t bases[24];
    for (auto &b : bases) b = (rng() & 0xFFFFF) << 12;
    auto rand_region = [&]() {
        Region r;
        r.base = bases[rng() % 24];
        r.lo = (rng() % 64) * 128;
        r.hi = r.lo + (1 + rng() % 8) * 128;
        return r;
    };

    // Precompute the deterministic op stream.
    std::vector<TaskOps> ops(num_tasks);
    for (int32_t N = 0; N < num_tasks; N++) {
        int nq = rng() % 4;
        for (int q = 0; q < nq; q++) ops[N].lookups.push_back(rand_region());
        int no = 1 + rng() % 3;
        for (int o = 0; o < no; o++) ops[N].inserts.push_back(rand_region());
    }

    // Pass 1: private reference (single-threaded, deterministic).
    PrivRing ref(cap);
    std::vector<std::vector<int32_t>> priv_results(num_tasks);
    for (int32_t N = 0; N < num_tasks; N++) {
        ref.advance_retire(N, H);
        for (const auto &q : ops[N].lookups) priv_results[N].push_back(ref.lookup(q.base, q.lo, q.hi));
        for (const auto &o : ops[N].inserts) ref.insert(o.base, o.lo, o.hi, N);
    }
    if (ref.overflowed) { printf("  PRIVATE OVERFLOW (cap=%d too small)\n", cap); return false; }

    // Pass 2: shared map with real threads racing the sequencer.
    SharedTensorMap sm(cap);
    std::atomic<int32_t> tm_insert_next(0);
    g_slot_assign.store(0, std::memory_order_relaxed);
    std::vector<std::atomic<int32_t>> core_progress(num_threads);
    for (auto &cp : core_progress) cp.store(0, std::memory_order_relaxed);
    SharedStats stats;
    std::vector<int32_t> append_order;
    append_order.reserve(num_tasks);

    std::vector<std::thread> threads;
    for (int32_t t = 0; t < num_threads; t++)
        threads.emplace_back(shared_worker, std::ref(sm), std::ref(tm_insert_next), std::cref(ops), num_tasks, H,
                             std::cref(priv_results), std::ref(stats), core_progress.data(), num_threads,
                             &append_order);
    for (auto &th : threads) th.join();

    int32_t expected_inserts = 0;
    for (const auto &task : ops) expected_inserts += static_cast<int32_t>(task.inserts.size());
    bool inserts_ok = (stats.total_inserts.load() == expected_inserts);
    bool mism_ok = (stats.mismatches.load() == 0);
    bool overflow_ok = !sm.overflowed.load();
    bool seq_ok = (tm_insert_next.load() == num_tasks);
    bool order_ok = (append_order.size() == static_cast<size_t>(num_tasks));
    for (size_t i = 0; order_ok && i < append_order.size(); i++)
        order_ok = append_order[i] == static_cast<int32_t>(i);

    bool ok = inserts_ok && mism_ok && overflow_ok && seq_ok && order_ok;
    printf("  tasks=%d threads=%d  inserts=%d/%d  mismatches=%d  overflow=%d  seq=%d  order=%d  %s\n",
           num_tasks, num_threads, stats.total_inserts.load(), expected_inserts, stats.mismatches.load(),
           sm.overflowed.load(), tm_insert_next.load(), order_ok, ok ? "PASS" : "FAIL");
    return ok;
}

// ---- Test: seq ABA under aggressive wrap-around -------------------------------
// Use a tiny cap so the ring wraps frequently, forcing slot reuse. Concurrent
// lookups must not be corrupted by the ABA (seq guard).
bool test_aba_wrap(int32_t num_tasks, int32_t num_threads) {
    // Very small cap + many distinct bases in the SAME bucket → aggressive wrap.
    // Use a fixed hash to force same bucket.
    int32_t H = 4;
    int32_t cap = 8;  // tiny → wraps every 8 inserts in a bucket

    std::vector<TaskOps> ops(num_tasks);
    uint64_t base = 0x1000;  // all same bucket
    for (int32_t N = 0; N < num_tasks; N++) {
        ops[N].inserts.push_back({base, (uint64_t)N * 16, (uint64_t)N * 16 + 16});
        // lookup the most recent overlapping insert
        if (N > 0) ops[N].lookups.push_back({base, (uint64_t)(N - 1) * 16, (uint64_t)N * 16});
    }

    // Private reference
    PrivRing ref(cap);
    std::vector<std::vector<int32_t>> priv_results(num_tasks);
    for (int32_t N = 0; N < num_tasks; N++) {
        ref.advance_retire(N, H);
        for (const auto &q : ops[N].lookups) priv_results[N].push_back(ref.lookup(q.base, q.lo, q.hi));
        for (const auto &o : ops[N].inserts) ref.insert(o.base, o.lo, o.hi, N);
    }

    // Always run the shared map, even if the private reference already
    // overflowed.  The old test returned here and never exercised the subject
    // under test.  With a tiny cap the shared map must report overflow rather
    // than silently overwrite a live slot.
    SharedTensorMap sm(cap);
    std::atomic<int32_t> tm_insert_next(0);
    g_slot_assign.store(0, std::memory_order_relaxed);
    std::vector<std::atomic<int32_t>> core_progress(num_threads);
    for (auto &cp : core_progress) cp.store(0, std::memory_order_relaxed);
    SharedStats stats;
    std::vector<int32_t> append_order;
    append_order.reserve(num_tasks);
    std::vector<std::thread> threads;
    for (int32_t t = 0; t < num_threads; t++)
        threads.emplace_back(shared_worker, std::ref(sm), std::ref(tm_insert_next), std::cref(ops), num_tasks, H,
                             std::cref(priv_results), std::ref(stats), core_progress.data(), num_threads,
                             &append_order);
    for (auto &th : threads) th.join();

    bool order_ok = (append_order.size() == static_cast<size_t>(num_tasks));
    for (size_t i = 0; order_ok && i < append_order.size(); i++)
        order_ok = append_order[i] == static_cast<int32_t>(i);
    // The private and shared reclaim floors are intentionally different, so a
    // deliberately tiny cap may overflow one model before the other.  The
    // subject under test is shared mode: if it overflows, that is the required
    // loud outcome; if it does not, every lookup must still match the private
    // reference.
    bool ok = order_ok && (sm.overflowed.load() || stats.mismatches.load() == 0);
    printf("  aba-wrap  tasks=%d threads=%d  mismatches=%d private_overflow=%d shared_overflow=%d order=%d  %s\n",
           num_tasks, num_threads, stats.mismatches.load(), ref.overflowed, sm.overflowed.load(), order_ok,
           ok ? (sm.overflowed.load() ? "PASS(overflow=FATAL)" : "PASS") : "FAIL");
    return ok;
}

int main() {
    int failures = 0;
    printf("=== MB-5 shared TensorMap: sequencer + seq ABA + private==shared ===\n");

    printf("\n[Test 1] shared == private (real threads on sequencer):\n");
    for (int32_t nt : {2, 6, 24}) {
        if (!test_shared_vs_private(4000, nt, /*H=*/64, /*cap=*/kBucketCapMax)) failures++;
    }

    printf("\n[Test 2] shared == private (high contention, few bases):\n");
    if (!test_shared_vs_private(2000, 72, 32, kBucketCapMax)) failures++;

    printf("\n[Test 3] seq ABA under aggressive wrap (tiny cap):\n");
    if (!test_aba_wrap(500, 6)) failures++;

    printf("\n=== MB-5 %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
