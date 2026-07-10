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
// MB-3 frontier — cooperative CAS advance (monotone, no-hole-crossing, no-retreat).
//
// Re-implements advance_frontier() from dist_engine.cpp:1638-1653 and the
// flags[] completion ring (dist_engine.cpp:1226-1230) on host with std::atomic.
// Stresses the cooperative CAS loop with real std::thread out-of-order completion.
//
// Criteria (docs MB-3):
//   1. After all flags set + advance calls quiesce: frontier == num_tasks - 1
//   2. Frontier is monotone non-decreasing (sampled over time)
//   3. No hole crossing: at any point, all id <= frontier have flags[id]==1
//   4. No retreat: frontier never decreases
//
// Standalone:
//   g++ -O2 -std=c++17 -pthread -o /tmp/mb3 test_dist_atomic_mb3_frontier.cpp && /tmp/mb3
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

static constexpr int32_t kFlagCap = 1 << 16;
static constexpr int32_t kFlagCapMask = kFlagCap - 1;

// ---- advance_frontier (dist_engine.cpp:1638-1653) -----------------------------
// After any core publishes flag(N), the contiguous-done prefix may have grown,
// so any core walks F forward while flag(F+1) is set. Lock-free CAS loop.
void advance_frontier(std::atomic<int32_t> &frontier, std::atomic<int32_t> *flags) {
    int32_t f = frontier.load(std::memory_order_acquire);
    while (true) {
        const int32_t next = f + 1;
        if (next >= kFlagCap) break;
        if (flags[next & kFlagCapMask].load(std::memory_order_acquire) == 0) break;
        if (frontier.compare_exchange_weak(f, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
            f = next;
        }
        // On CAS failure, compare_exchange_weak reloads f with the current value.
    }
}

// ---- dist_set_flag (dist_engine.cpp:573-591, sim path) ------------------------
void dist_set_flag(std::atomic<int32_t> *flags, int32_t task_id) {
    flags[task_id & kFlagCapMask].store(1, std::memory_order_release);
}

// ---- Test 1: in-order completion, frontier advances to end -------------------
bool test_in_order(int32_t num_tasks, int32_t num_threads) {
    std::atomic<int32_t> frontier(-1);
    std::vector<std::atomic<int32_t>> flags(kFlagCap);
    for (int32_t i = 0; i < kFlagCap; i++) flags[i].store(0, std::memory_order_relaxed);

    std::atomic<int32_t> next_id(0);
    auto worker = [&]() {
        while (true) {
            int32_t N = next_id.fetch_add(1, std::memory_order_relaxed);
            if (N >= num_tasks) break;
            dist_set_flag(flags.data(), N);
            advance_frontier(frontier, flags.data());
        }
    };
    std::vector<std::thread> threads;
    for (int32_t t = 0; t < num_threads; t++) threads.emplace_back(worker);
    for (auto &th : threads) th.join();

    bool ok = (frontier.load() == num_tasks - 1);
    printf("  in-order  tasks=%d threads=%d  frontier=%d (expect %d)  %s\n", num_tasks, num_threads,
           frontier.load(), num_tasks - 1, ok ? "PASS" : "FAIL");
    return ok;
}

// ---- Test 2: reverse-order completion — frontier stalls at hole ---------------
bool test_reverse_order(int32_t num_tasks) {
    std::atomic<int32_t> frontier(-1);
    std::vector<std::atomic<int32_t>> flags(kFlagCap);
    for (int32_t i = 0; i < kFlagCap; i++) flags[i].store(0, std::memory_order_relaxed);

    // Set flags in reverse: num_tasks-1, num_tasks-2, ..., 1, 0
    // Frontier should stay at -1 until id 0 is set, then jump to num_tasks-1.
    for (int32_t N = num_tasks - 1; N >= 1; N--) {
        dist_set_flag(flags.data(), N);
        advance_frontier(frontier, flags.data());
    }
    // Before id 0: frontier should still be -1 (hole at 0)
    bool hole_ok = (frontier.load() == -1);
    // Now set id 0: frontier should advance to num_tasks-1
    dist_set_flag(flags.data(), 0);
    advance_frontier(frontier, flags.data());
    bool advance_ok = (frontier.load() == num_tasks - 1);

    bool ok = hole_ok && advance_ok;
    printf("  reverse   tasks=%d  hole_stall=%d final=%d (expect %d)  %s\n", num_tasks, hole_ok,
           frontier.load(), num_tasks - 1, ok ? "PASS" : "FAIL");
    return ok;
}

// ---- Test 3: stress out-of-order with concurrent advance + monotone check -----
bool test_stress(int32_t num_tasks, int32_t num_threads) {
    std::atomic<int32_t> frontier(-1);
    std::vector<std::atomic<int32_t>> flags(kFlagCap);
    for (int32_t i = 0; i < kFlagCap; i++) flags[i].store(0, std::memory_order_relaxed);

    // Monotone sampler: periodically reads frontier, asserts non-decreasing.
    std::atomic<bool> stop(false);
    std::atomic<int32_t> max_sampled(-1);
    std::atomic<bool> monotone_violated(false);
    std::thread sampler([&]() {
        int32_t prev = -1;
        while (!stop.load(std::memory_order_relaxed)) {
            int32_t f = frontier.load(std::memory_order_acquire);
            if (f < prev) monotone_violated.store(true, std::memory_order_relaxed);
            if (f > prev) prev = f;
            max_sampled.store(prev, std::memory_order_relaxed);
        }
    });

    // Workers: each grabs ids from a shared counter (natural out-of-order), sets
    // flag, calls advance_frontier. Random tiny delay to spread completion order.
    std::atomic<int32_t> next_id(0);
    auto worker = [&]() {
        while (true) {
            int32_t N = next_id.fetch_add(1, std::memory_order_relaxed);
            if (N >= num_tasks) break;
            // Simulate variable exec time (PTO_DIST_FAKE_EXEC_NS)
            for (volatile int d = 0; d < (N * 7) % 64; d++) {}
            dist_set_flag(flags.data(), N);
            advance_frontier(frontier, flags.data());
        }
    };
    std::vector<std::thread> threads;
    for (int32_t t = 0; t < num_threads; t++) threads.emplace_back(worker);
    for (auto &th : threads) th.join();

    stop.store(true, std::memory_order_relaxed);
    sampler.join();

    int32_t final_f = frontier.load();
    bool final_ok = (final_f == num_tasks - 1);
    bool monotone_ok = !monotone_violated.load();

    // Verify no-hole invariant at quiescence: all id <= final_f have flag==1.
    bool no_hole = true;
    for (int32_t i = 0; i <= final_f; i++) {
        if (flags[i & kFlagCapMask].load(std::memory_order_acquire) == 0) { no_hole = false; break; }
    }

    bool ok = final_ok && monotone_ok && no_hole;
    printf("  stress    tasks=%d threads=%d  frontier=%d (expect %d)  monotone=%d  no_hole=%d  %s\n",
           num_tasks, num_threads, final_f, num_tasks - 1, monotone_ok, no_hole, ok ? "PASS" : "FAIL");
    return ok;
}

// ---- Test 4: deliberate hole — frontier stops, then fills --------------------
bool test_hole() {
    std::atomic<int32_t> frontier(-1);
    std::vector<std::atomic<int32_t>> flags(kFlagCap);
    for (int32_t i = 0; i < kFlagCap; i++) flags[i].store(0, std::memory_order_relaxed);

    // Set 0,1,2,4,5 but NOT 3 — frontier should stop at 2.
    for (int32_t N : {0, 1, 2, 4, 5}) dist_set_flag(flags.data(), N);
    advance_frontier(frontier, flags.data());
    bool hole_ok = (frontier.load() == 2);

    // Now fill the hole (set 3) — frontier should advance to 5.
    dist_set_flag(flags.data(), 3);
    advance_frontier(frontier, flags.data());
    bool fill_ok = (frontier.load() == 5);

    bool ok = hole_ok && fill_ok;
    printf("  hole      stop_at=2 got=%d  fill_to=5 got=%d  %s\n", hole_ok ? 2 : -99, frontier.load(),
           ok ? "PASS" : "FAIL");
    return ok;
}

// ---- Test 5: concurrent advance by multiple threads on same flags -------------
bool test_concurrent_advance(int32_t num_tasks, int32_t num_threads) {
    std::atomic<int32_t> frontier(-1);
    std::vector<std::atomic<int32_t>> flags(kFlagCap);
    for (int32_t i = 0; i < kFlagCap; i++) flags[i].store(0, std::memory_order_relaxed);

    // Phase 1: set ALL flags first (no advance).
    for (int32_t N = 0; N < num_tasks; N++) dist_set_flag(flags.data(), N);
    // Phase 2: all threads race to advance_frontier concurrently.
    auto advancer = [&]() {
        for (int i = 0; i < 100; i++) advance_frontier(frontier, flags.data());
    };
    std::vector<std::thread> threads;
    for (int32_t t = 0; t < num_threads; t++) threads.emplace_back(advancer);
    for (auto &th : threads) th.join();

    bool ok = (frontier.load() == num_tasks - 1);
    printf("  conc-adv  tasks=%d threads=%d  frontier=%d (expect %d)  %s\n", num_tasks, num_threads,
           frontier.load(), num_tasks - 1, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    int failures = 0;
    printf("=== MB-3 frontier: cooperative CAS advance ===\n");

    printf("\n[Test 1] in-order completion:\n");
    for (int32_t nt : {2, 6, 24, 72}) if (!test_in_order(5000, nt)) failures++;

    printf("\n[Test 2] reverse-order (hole stall):\n");
    if (!test_reverse_order(1000)) failures++;

    printf("\n[Test 3] stress out-of-order + monotone sampler:\n");
    for (int32_t nt : {6, 24, 72}) if (!test_stress(5000, nt)) failures++;

    printf("\n[Test 4] deliberate hole:\n");
    if (!test_hole()) failures++;

    printf("\n[Test 5] concurrent advance (all flags pre-set):\n");
    for (int32_t nt : {6, 24, 72}) if (!test_concurrent_advance(5000, nt)) failures++;

    printf("\n=== MB-3 %s (%d failures) ===\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
