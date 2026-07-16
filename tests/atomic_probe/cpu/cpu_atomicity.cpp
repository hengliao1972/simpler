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
// Race-free CPU control group for the A5 cache-line probes.
//
// The important isomorphic case is multiple threads updating distinct 4B
// words in one 64B cache line. A coherent CPU must preserve every exact final
// value; placing each word on a separate line changes performance, not values.
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

constexpr size_t CACHELINE_BYTES = 64;

class Result {
public:
    void Expect(bool condition, const char *label)
    {
        std::printf("[ASSERT] %-48s %s\n", label, condition ? "PASS" : "FAIL");
        if (!condition) failures_++;
    }

    int ExitCode() const
    {
        std::printf("[SUMMARY] semantic_failures=%d\n", failures_);
        return failures_ == 0 ? 0 : 1;
    }

private:
    int failures_ = 0;
};

struct alignas(CACHELINE_BYTES) SameLineWords {
    std::atomic<uint32_t> value[16];
};

struct alignas(CACHELINE_BYTES) SeparateLineWord {
    std::atomic<uint32_t> value;
    std::array<uint8_t, CACHELINE_BYTES - sizeof(std::atomic<uint32_t>)> padding{};
};

static_assert(sizeof(SameLineWords) == CACHELINE_BYTES, "control must occupy exactly one cache line");
static_assert(sizeof(SeparateLineWord) == CACHELINE_BYTES, "padded control must occupy one cache line");

uint32_t ExactValue(uint32_t thread, uint32_t round)
{
    return 0xA5000000u + thread * 0x00100000u + round;
}

void TestAtomicIncrement(Result &result)
{
    constexpr uint32_t THREADS = 8;
    constexpr uint32_t ROUNDS = 200000;
    std::atomic<uint32_t> counter32{0};
    std::atomic<uint64_t> counter64{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;

    for (uint32_t thread = 0; thread < THREADS; thread++) {
        threads.emplace_back([&]() {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            for (uint32_t round = 0; round < ROUNDS; round++) {
                counter32.fetch_add(1, std::memory_order_relaxed);
                counter64.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &thread : threads) thread.join();

    uint64_t expected = (uint64_t)THREADS * ROUNDS;
    std::printf("\n=== Shared Atomic Increment ===\n");
    std::printf("u32=%u u64=%" PRIu64 " expected=%" PRIu64 "\n",
                counter32.load(), counter64.load(), expected);
    result.Expect(counter32.load() == expected, "CPU atomic<uint32_t> exact increment");
    result.Expect(counter64.load() == expected, "CPU atomic<uint64_t> exact increment");
}

template <typename Store>
double RunDistinctWordWriters(Store store)
{
    constexpr uint32_t THREADS = 8;
    constexpr uint32_t ROUNDS = 500000;
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    auto before = std::chrono::steady_clock::now();
    for (uint32_t thread = 0; thread < THREADS; thread++) {
        threads.emplace_back([&, thread]() {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            for (uint32_t round = 0; round < ROUNDS; round++) store(thread, ExactValue(thread, round));
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &thread : threads) thread.join();
    auto after = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(after - before).count();
}

void TestDistinctWords(Result &result)
{
    constexpr uint32_t THREADS = 8;
    constexpr uint32_t ROUNDS = 500000;
    SameLineWords same_line{};
    SeparateLineWord separate_lines[THREADS]{};

    for (uint32_t i = 0; i < 16; i++) same_line.value[i].store(0, std::memory_order_relaxed);
    for (auto &word : separate_lines) word.value.store(0, std::memory_order_relaxed);

    double same_ms = RunDistinctWordWriters(
        [&](uint32_t thread, uint32_t value) { same_line.value[thread].store(value, std::memory_order_relaxed); });
    double separate_ms = RunDistinctWordWriters(
        [&](uint32_t thread, uint32_t value) { separate_lines[thread].value.store(value, std::memory_order_relaxed); });

    bool same_exact = true;
    bool separate_exact = true;
    for (uint32_t thread = 0; thread < THREADS; thread++) {
        uint32_t expected = ExactValue(thread, ROUNDS - 1);
        same_exact &= same_line.value[thread].load(std::memory_order_relaxed) == expected;
        separate_exact &= separate_lines[thread].value.load(std::memory_order_relaxed) == expected;
    }

    std::printf("\n=== Distinct-Word Cache-Line Control ===\n");
    std::printf("same-line=%.1fms separate-lines=%.1fms ratio=%.2f\n",
                same_ms, separate_ms, separate_ms > 0 ? same_ms / separate_ms : 0.0);
    result.Expect(same_exact, "CPU same-cacheline distinct-word exact values");
    result.Expect(separate_exact, "CPU separate-cacheline distinct-word exact values");
}

void TestMultiwordObservation(Result &result)
{
    constexpr uint32_t WORDS = 16;
    constexpr uint32_t ROUNDS = 300000;
    alignas(CACHELINE_BYTES) std::atomic<uint32_t> data[WORDS]{};
    std::atomic<bool> reader_ready{false};
    std::atomic<bool> stop{false};
    uint64_t reads = 0;
    uint64_t torn = 0;

    std::thread reader([&]() {
        reader_ready.store(true, std::memory_order_release);
        while (!stop.load(std::memory_order_acquire)) {
            uint32_t first = data[0].load(std::memory_order_relaxed);
            uint32_t last = data[WORDS - 1].load(std::memory_order_relaxed);
            reads++;
            if (first != last) torn++;
        }
    });
    while (!reader_ready.load(std::memory_order_acquire)) std::this_thread::yield();

    for (uint32_t round = 0; round < ROUNDS; round++) {
        uint32_t pattern = (round & 1u) == 0 ? 0xAAAAAAAAu : 0xBBBBBBBBu;
        for (auto &word : data) word.store(pattern, std::memory_order_relaxed);
    }
    stop.store(true, std::memory_order_release);
    reader.join();

    std::printf("\n=== Multiword Snapshot Observation ===\n");
    std::printf("reads=%" PRIu64 " torn=%" PRIu64 " (observational)\n", reads, torn);
    result.Expect(reads > 0, "CPU multiword observer executed");
}

void Lock(std::atomic_flag &lock)
{
    while (lock.test_and_set(std::memory_order_acquire)) std::this_thread::yield();
}

void TestSpinlockSnapshot(Result &result)
{
    constexpr uint32_t WORDS = 16;
    constexpr uint32_t WRITERS = 4;
    constexpr uint32_t ROUNDS = 100000;
    alignas(CACHELINE_BYTES) std::array<uint32_t, WORDS> data{};
    std::atomic_flag lock = ATOMIC_FLAG_INIT;
    std::atomic<bool> start{false};
    std::atomic<bool> stop{false};
    std::atomic<bool> reader_ready{false};
    uint64_t reads = 0;
    uint64_t torn = 0;

    std::thread reader([&]() {
        reader_ready.store(true, std::memory_order_release);
        while (!stop.load(std::memory_order_acquire)) {
            Lock(lock);
            uint32_t first = data.front();
            bool consistent = true;
            for (uint32_t value : data) consistent &= value == first;
            lock.clear(std::memory_order_release);
            reads++;
            if (!consistent) torn++;
        }
    });
    while (!reader_ready.load(std::memory_order_acquire)) std::this_thread::yield();

    std::vector<std::thread> writers;
    for (uint32_t writer = 0; writer < WRITERS; writer++) {
        writers.emplace_back([&, writer]() {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            for (uint32_t round = 0; round < ROUNDS; round++) {
                uint32_t pattern = ExactValue(writer, round);
                Lock(lock);
                data.fill(pattern);
                lock.clear(std::memory_order_release);
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &writer : writers) writer.join();
    stop.store(true, std::memory_order_release);
    reader.join();

    Lock(lock);
    bool final_consistent = true;
    for (uint32_t value : data) final_consistent &= value == data.front();
    lock.clear(std::memory_order_release);

    std::printf("\n=== Spinlock-Protected Snapshot ===\n");
    std::printf("reads=%" PRIu64 " torn=%" PRIu64 "\n", reads, torn);
    result.Expect(reads > 0, "CPU spinlock observer executed");
    result.Expect(torn == 0 && final_consistent, "CPU spinlock snapshot exact consistency");
}

int main()
{
    std::printf("=== CPU Cache-Line Control (coherent host) ===\n");
    std::printf("hardware_concurrency=%u cacheline_bytes=%zu\n",
                std::thread::hardware_concurrency(), CACHELINE_BYTES);
    Result result;
    TestAtomicIncrement(result);
    TestDistinctWords(result);
    TestMultiwordObservation(result);
    TestSpinlockSnapshot(result);
    return result.ExitCode();
}
