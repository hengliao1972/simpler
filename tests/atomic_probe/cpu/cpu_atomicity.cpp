// CPU atomicity probe — multi-threaded counterpart to A5 probes.
//
// x86-64 has HW cache coherence (MESIF): no cacheline clobber, no stale reads.
// This file tests what IS and ISN'T atomic on "normal" hardware, as a
// control group against A5 (no HW coherence).
//
// Tests:
//   1. Single-word atomic increment (32-bit and 64-bit)
//   2. Multi-word "torn read" (expected: NOT atomic)
//   3. Cache-line false sharing performance (x86 coherence overhead)
//   4. std::atomic vs plain — lost update demonstration
//
// Build: g++ -O2 -std=c++17 -pthread -o cpu_atomicity cpu_atomicity.cpp
#include <atomic>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <cstring>

constexpr int CACHELINE = 64;

// ============================================================
// Test 1: Single-word atomic increment (32-bit & 64-bit)
//
// N threads each increment a counter ROUNDS times.
// Plain uint32_t: lost updates (data race).
// std::atomic<uint32_t>: all updates counted.
// std::atomic<uint64_t>: same for 64-bit.
// ============================================================
void test_atomic_increment() {
    constexpr int THREADS = 8;
    constexpr int ROUNDS = 1000000;

    printf("\n=== Test 1: Single-Word Atomic Increment ===\n");
    printf("Threads=%d  Rounds=%d  Expected=%d\n\n", THREADS, ROUNDS, THREADS * ROUNDS);

    // --- 32-bit non-atomic (volatile to prevent compiler optimization) ---
    volatile uint32_t plain32 = 0;
    auto inc_plain32 = [&]() {
        for (int i = 0; i < ROUNDS; i++) plain32++;
    };
    std::vector<std::thread> t;
    for (int i = 0; i < THREADS; i++) t.emplace_back(inc_plain32);
    for (auto &th : t) th.join();
    printf("uint32_t  plain:   %u  (lost %u updates, %.1f%%)\n",
           plain32, THREADS * ROUNDS - plain32,
           100.0 * (THREADS * ROUNDS - plain32) / (THREADS * ROUNDS));

    // --- 32-bit atomic ---
    std::atomic<uint32_t> atom32{0};
    t.clear();
    for (int i = 0; i < THREADS; i++) t.emplace_back([&]() {
        for (int i = 0; i < ROUNDS; i++) atom32.fetch_add(1, std::memory_order_relaxed);
    });
    for (auto &th : t) th.join();
    printf("uint32_t  atomic:  %u  %s\n", atom32.load(),
           atom32.load() == (uint32_t)(THREADS * ROUNDS) ? "OK" : "FAIL");

    // --- 64-bit atomic ---
    std::atomic<uint64_t> atom64{0};
    t.clear();
    for (int i = 0; i < THREADS; i++) t.emplace_back([&]() {
        for (int i = 0; i < ROUNDS; i++) atom64.fetch_add(1, std::memory_order_relaxed);
    });
    for (auto &th : t) th.join();
    printf("uint64_t  atomic:  %lu  %s\n", atom64.load(),
           atom64.load() == (uint64_t)THREADS * ROUNDS ? "OK" : "FAIL");
}

// ============================================================
// Test 2: Multi-word torn read (expected: NOT atomic)
//
// Writer alternates: fills 16 words with PATTERN_A, then PATTERN_B.
// Reader checks if word[0] == word[15].
// Even with std::atomic per word, multi-word read is NOT atomic.
// ============================================================
void test_torn_read() {
    constexpr int WORDS = 16;
    constexpr int ROUNDS = 1000000;

    // Align to cache line to ensure all 16 words are in the same line
    alignas(CACHELINE) std::atomic<uint32_t> data[WORDS];
    for (int i = 0; i < WORDS; i++) data[i].store(0, std::memory_order_relaxed);

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> torn_count{0};
    std::atomic<uint64_t> read_count{0};

    printf("\n=== Test 2: Multi-Word Torn Read (16 words × 64B cache line) ===\n");

    // Writer: alternate patterns
    std::thread writer([&]() {
        for (int r = 0; r < ROUNDS; r++) {
            uint32_t pat = (r % 2 == 0) ? 0xAAAAAAAA : 0xBBBBBBBB;
            for (int i = 0; i < WORDS; i++)
                data[i].store(pat, std::memory_order_relaxed);
        }
        stop.store(true);
    });

    // Reader: check consistency
    std::thread reader([&]() {
        while (!stop.load()) {
            uint32_t v0 = data[0].load(std::memory_order_relaxed);
            uint32_t v15 = data[WORDS - 1].load(std::memory_order_relaxed);
            read_count.fetch_add(1);
            if (v0 != v15) torn_count.fetch_add(1);
        }
    });

    writer.join();
    reader.join();

    printf("Reads=%lu  Torn=%lu  (%.1f%%)\n",
           read_count.load(), torn_count.load(),
           read_count.load() ? 100.0 * torn_count.load() / read_count.load() : 0);
    printf("Conclusion: multi-word read is %s — expected on ALL architectures\n",
           torn_count.load() > 0 ? "NOT atomic" : "atomic (unlikely)");
    printf("Fix: use mutex/spinlock for multi-word atomicity\n");
}

// ============================================================
// Test 3: Cache-line false sharing performance
//
// Multiple threads increment adjacent counters in the same cache line.
// x86 coherence (MESIF) serializes the cache line → performance loss.
// Padding each counter to its own cache line eliminates false sharing.
// ============================================================
void test_false_sharing() {
    constexpr int THREADS = 4;
    constexpr int ROUNDS = 50000000;

    printf("\n=== Test 3: Cache-Line False Sharing Performance ===\n");
    printf("Threads=%d  Rounds=%d  Cache line=%d bytes\n\n", THREADS, ROUNDS, CACHELINE);

    // --- False sharing: 4 counters packed in one cache line ---
    {
        struct alignas(64) PackedCounters {
            std::atomic<uint32_t> val[4];
        } counters;
        for (int i = 0; i < 4; i++) counters.val[i].store(0);

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> t;
        for (int i = 0; i < THREADS; i++) t.emplace_back([&, i]() {
            for (int r = 0; r < ROUNDS; r++)
                counters.val[i].fetch_add(1, std::memory_order_relaxed);
        });
        for (auto &th : t) th.join();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("False sharing (4 counters/line): %.0f ms  values: %u %u %u %u\n",
               ms,
               counters.val[0].load(), counters.val[1].load(),
               counters.val[2].load(), counters.val[3].load());
    }

    // --- Padded: each counter in its own cache line ---
    {
        struct alignas(64) PaddedCounter {
            std::atomic<uint32_t> val{0};
            char pad[CACHELINE - sizeof(std::atomic<uint32_t>)];
        } counters[4];

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> t;
        for (int i = 0; i < THREADS; i++) t.emplace_back([&, i]() {
            for (int r = 0; r < ROUNDS; r++)
                counters[i].val.fetch_add(1, std::memory_order_relaxed);
        });
        for (auto &th : t) th.join();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("Padded (1 counter/line):        %.0f ms  values: %u %u %u %u\n",
               ms,
               counters[0].val.load(), counters[1].val.load(),
               counters[2].val.load(), counters[3].val.load());
    }

    printf("\nNote: x86 has HW coherence → no data corruption (values correct in both).\n");
    printf("But false sharing causes cache ping-pong → slower.\n");
    printf("On A5 (no coherence): false sharing → data CORRUPTION (clobber).\n");
}

// ============================================================
// Test 4: Mutex for multi-word atomicity
//
// Demonstrates the CORRECT way to do multi-word atomic operations:
// use a spinlock (std::atomic flag) to protect the critical section.
// ============================================================
void test_mutex_atomicity() {
    constexpr int WORDS = 16;
    constexpr int THREADS = 4;
    constexpr int ROUNDS = 500000;

    alignas(CACHELINE) uint32_t data[WORDS];
    std::atomic<uint32_t> lock{0};  // simple spinlock
    std::atomic<uint64_t> torn{0};
    std::atomic<uint64_t> reads{0};

    printf("\n=== Test 4: Spinlock for Multi-Word Atomicity ===\n");
    printf("Threads=%d  Rounds=%d  Words=%d\n", THREADS, ROUNDS, WORDS);

    // Writers: each writes a unique pattern under spinlock
    std::vector<std::thread> writers;
    for (int tid = 0; tid < THREADS; tid++) {
        writers.emplace_back([&, tid]() {
            uint32_t pat = 0x10000000u * (tid + 1);
            for (int r = 0; r < ROUNDS; r++) {
                pat++;
                // Spinlock acquire
                while (lock.exchange(1, std::memory_order_acquire)) {}
                // Critical section: write all words
                for (int i = 0; i < WORDS; i++) data[i] = pat;
                // Release
                lock.store(0, std::memory_order_release);
            }
        });
    }

    // Reader: check consistency under spinlock
    std::atomic<bool> stop{false};
    std::thread reader([&]() {
        while (!stop.load()) {
            while (lock.exchange(1, std::memory_order_acquire)) {}
            uint32_t v0 = data[0];
            uint32_t v15 = data[WORDS - 1];
            lock.store(0, std::memory_order_release);
            reads.fetch_add(1);
            if (v0 != v15) torn.fetch_add(1);
        }
    });

    for (auto &t : writers) t.join();
    stop.store(true);
    reader.join();

    printf("Reads=%lu  Torn=%lu  %s\n", reads.load(), torn.load(),
           torn.load() == 0 ? "OK (spinlock works)" : "FAIL");
    printf("Conclusion: mutex/spinlock is the ONLY way to achieve multi-word atomicity.\n");
    printf("On A5: use atomicMax as spinlock, ld_dev/st_dev inside critical section.\n");
}

// ============================================================
// Test 5: 64-bit atomicity — is atomic<uint64_t> truly atomic?
//
// Concurrent writers do atomic OR with different bits.
// If 64-bit atomic is broken, some bits would be lost.
// ============================================================
void test_64bit_atomicity() {
    constexpr int THREADS = 8;
    constexpr int ROUNDS = 1000000;
    constexpr uint64_t BIT_MASK = 0x5555555555555555ULL;

    printf("\n=== Test 5: 64-Bit Atomicity ===\n");

    // Each thread sets its own bit pattern via atomic OR
    std::atomic<uint64_t> val{0};
    std::vector<std::thread> t;
    for (int i = 0; i < THREADS; i++) {
        t.emplace_back([&, i]() {
            uint64_t my_bit = 1ULL << (i * 8);
            for (int r = 0; r < ROUNDS; r++)
                val.fetch_or(my_bit, std::memory_order_relaxed);
        });
    }
    for (auto &th : t) th.join();

    uint64_t expected = 0;
    for (int i = 0; i < THREADS; i++) expected |= (1ULL << (i * 8));
    printf("atomic<uint64_t> fetch_or: expected=0x%016lx  actual=0x%016lx  %s\n",
           expected, val.load(),
           val.load() == expected ? "OK (64-bit atomic)" : "FAIL");

    // CAS-based increment
    std::atomic<uint64_t> counter{0};
    t.clear();
    for (int i = 0; i < THREADS; i++) {
        t.emplace_back([&]() {
            for (int r = 0; r < ROUNDS; r++) {
                uint64_t old = counter.load(std::memory_order_relaxed);
                while (!counter.compare_exchange_weak(old, old + 1,
                           std::memory_order_relaxed)) {}
            }
        });
    }
    for (auto &th : t) th.join();
    printf("atomic<uint64_t> CAS increment: %lu (expected %d)  %s\n",
           counter.load(), THREADS * ROUNDS,
           counter.load() == (uint64_t)THREADS * ROUNDS ? "OK" : "FAIL");
}

int main() {
    printf("=== CPU Atomicity Probe (x86-64, HW cache coherence) ===\n");
    printf("Control group for A5 probes (no HW coherence).\n");
    printf("CPU cores: %u\n", std::thread::hardware_concurrency());

    test_atomic_increment();
    test_torn_read();
    test_false_sharing();
    test_64bit_atomicity();
    test_mutex_atomicity();

    printf("\n=== Summary ===\n");
    printf("32-bit atomic:  OK on x86 (std::atomic<uint32_t>)\n");
    printf("64-bit atomic:  OK on x86 (std::atomic<uint64_t>)\n");
    printf("Multi-word:     NOT atomic on ANY architecture — need mutex\n");
    printf("False sharing:  x86=perf loss only; A5=DATA CORRUPTION\n");
    printf("\n");
    printf("x86 vs A5 key difference:\n");
    printf("  x86: HW coherence → cache line never stale, no clobber\n");
    printf("  A5: NO coherence → must use st_dev/atomic, avoid store+dcci\n");
    return 0;
}
