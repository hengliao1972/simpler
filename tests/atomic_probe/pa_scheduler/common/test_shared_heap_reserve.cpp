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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

#define PA_DEVICE inline
#define PA_GM
#include "pa_shared_heap.h"

namespace {

using namespace pa_scheduler;

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] %s\n", message);
    ++g_failures;
}

struct HeapTestOps {
    static int64_t Load(volatile int64_t *address) {
        return __atomic_load_n(address, __ATOMIC_ACQUIRE);
    }

    static int64_t FetchAdd(volatile int64_t *address, int64_t value) {
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }
};

// 在 cursor 的前置 Load 与 FetchAdd 之间插入另一笔 reserve；它可以暂停在
// cursor/vend 两条原子之间，也可以完整推进。helper 必须消费 FetchAdd 的
// 真实 old value，不能要求它等于预检快照，更不能回滚并发进度。
struct HeapInterleaveOps : HeapTestOps {
    static volatile int64_t *race_cursor;
    static volatile int64_t *race_vend;
    static int64_t injected_delta;
    static bool advance_vend;

    static int64_t FetchAdd(volatile int64_t *address, int64_t value) {
        if (address == race_cursor) {
            (void)__atomic_fetch_add(
                address, injected_delta, __ATOMIC_ACQ_REL
            );
            if (advance_vend) {
                (void)__atomic_fetch_add(
                    race_vend, injected_delta, __ATOMIC_ACQ_REL
                );
            }
            race_cursor = nullptr;
        }
        return HeapTestOps::FetchAdd(address, value);
    }
};

volatile int64_t *HeapInterleaveOps::race_cursor = nullptr;
volatile int64_t *HeapInterleaveOps::race_vend = nullptr;
int64_t HeapInterleaveOps::injected_delta = 0;
bool HeapInterleaveOps::advance_vend = false;

void ResetHeapState(SharedTensorMapSidecar &map) {
    for (uint32_t shard = 0; shard < kSharedHeapShards; ++shard) {
        map.shared_heap_cursor[shard].value = 0;
    }
    map.shared_heap_vend.value = 0;
}

struct HeapSnapshot {
    int64_t cursor[kSharedHeapShards];
    int64_t vend;
};

HeapSnapshot Snapshot(const SharedTensorMapSidecar &map) {
    HeapSnapshot snapshot{};
    for (uint32_t shard = 0; shard < kSharedHeapShards; ++shard) {
        snapshot.cursor[shard] = map.shared_heap_cursor[shard].value;
    }
    snapshot.vend = map.shared_heap_vend.value;
    return snapshot;
}

bool SameSnapshot(
    const SharedTensorMapSidecar &map, const HeapSnapshot &expected
) {
    for (uint32_t shard = 0; shard < kSharedHeapShards; ++shard) {
        if (map.shared_heap_cursor[shard].value != expected.cursor[shard]) {
            return false;
        }
    }
    return map.shared_heap_vend.value == expected.vend;
}

constexpr uint64_t kOutputBytes[kTasksPerBatch] = {
    10240, 524288, 264192, 8192, 0,
};

// 串行 reference 只校验 PA Case1 的业务字节总量、默认 shard 分布和 no-wrap
// 容量；生产并发时同一 shard 内的 task 物理次序不由 task_id 决定。
void TestPaCase(uint32_t batches) {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetHeapState(*map);
    uint64_t expected_cursor[kSharedHeapShards] = {};
    uint64_t expected_vend = 0;
    const uint64_t shard_span =
        SharedHeapAlignDown(kHeapBytes / kSharedHeapShards);

    for (uint32_t task_id = 0; task_id < batches * kTasksPerBatch;
         ++task_id) {
        const uint32_t shard = task_id % kSharedHeapShards;
        const uint64_t total = kOutputBytes[task_id % kTasksPerBatch];
        SharedHeapReservation reservation{UINT64_MAX, UINT64_MAX};
        Check(
            ReserveSharedOutputHeap<HeapTestOps>(
                *map, task_id, total, kHeapBytes, reservation
            ),
            "PA Case1 reservation succeeds"
        );
        const uint64_t expected_base =
            total == 0 ? 0 : shard * shard_span + expected_cursor[shard];
        Check(
            reservation.task_base == expected_base,
            "serial PA reference follows its current shard cursor"
        );
        expected_cursor[shard] += total;
        expected_vend += total;
        Check(
            reservation.aggregate_vend == expected_vend,
            "serial PA reference aggregate vend is exact"
        );
    }

    uint64_t cursor_sum = 0;
    for (uint32_t shard = 0; shard < kSharedHeapShards; ++shard) {
        Check(
            static_cast<uint64_t>(map->shared_heap_cursor[shard].value) ==
                expected_cursor[shard],
            "PA Case1 final shard cursor is exact"
        );
        Check(
            expected_cursor[shard] <= shard_span,
            "PA Case1 shard remains inside no-wrap capacity"
        );
        cursor_sum += expected_cursor[shard];
    }
    Check(cursor_sum == expected_vend, "sum of shard cursors equals vend");
    Check(
        static_cast<uint64_t>(map->shared_heap_vend.value) == expected_vend,
        "PA Case1 final aggregate vend is exact"
    );

    if (batches == 1) {
        const uint64_t expected_b1[kSharedHeapShards] = {
            10240, 524288, 264192, 8192, 0, 0, 0, 0,
        };
        Check(
            std::memcmp(expected_cursor, expected_b1, sizeof(expected_b1)) == 0,
            "b1 shard distribution matches the PA topology"
        );
        Check(expected_vend == 806912, "b1 vend is 806912 bytes");
    } else if (batches == kDefaultBatches) {
        for (uint32_t shard = 0; shard < kSharedHeapShards; ++shard) {
            Check(
                expected_cursor[shard] == 25821184,
                "b256 distributes 25821184 bytes to every shard"
            );
        }
        Check(
            expected_vend == 206569472,
            "b256 vend is 206569472 bytes"
        );
    }
}

void TestAlignmentAndShardTail() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetHeapState(*map);
    const uint64_t heap_size =
        kSharedHeapShards * (4096 + 512);
    const uint64_t shard_span = 4096;

    SharedHeapReservation first{};
    Check(
        ReserveSharedOutputHeap<HeapTestOps>(
            *map, 7, 1, heap_size, first
        ),
        "unaligned request is rounded to one KiB"
    );
    Check(
        first.task_base == 7 * shard_span &&
            first.aggregate_vend == kOutputAlignment,
        "heap tail is excluded and shard base remains aligned"
    );
    Check(
        map->shared_heap_cursor[7].value ==
            static_cast<int64_t>(kOutputAlignment),
        "rounded reservation advances only its shard"
    );

    ResetHeapState(*map);
    map->shared_heap_vend.value =
        static_cast<int64_t>(kSharedHeapShards * shard_span + kOutputAlignment);
    const HeapSnapshot excluded_tail = Snapshot(*map);
    SharedHeapReservation zero{};
    Check(
        !ReserveSharedOutputHeap<HeapTestOps>(
            *map, 0, 0, heap_size, zero
        ),
        "aggregate vend inside the excluded heap tail is rejected"
    );
    Check(
        SameSnapshot(*map, excluded_tail),
        "excluded-tail rejection changes no heap state"
    );
}

void TestBoundaryZeroAndPreflightFailures() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetHeapState(*map);
    const uint64_t heap_size =
        kSharedHeapShards * 4096;

    SharedHeapReservation empty_zero{UINT64_MAX, UINT64_MAX};
    const HeapSnapshot empty_snapshot = Snapshot(*map);
    Check(
        ReserveSharedOutputHeap<HeapTestOps>(
            *map, 4, 0, heap_size, empty_zero
        ),
        "zero-output task may observe an empty aggregate vend"
    );
    Check(
        empty_zero.task_base == 0 && empty_zero.aggregate_vend == 0,
        "empty zero-output reservation returns a zero diagnostic prefix"
    );
    Check(
        SameSnapshot(*map, empty_snapshot),
        "empty zero-output reservation changes no heap state"
    );

    SharedHeapReservation full{};
    Check(
        ReserveSharedOutputHeap<HeapTestOps>(
            *map, 0, 4096, heap_size, full
        ),
        "reservation may exactly fill one shard"
    );
    const HeapSnapshot full_snapshot = Snapshot(*map);

    SharedHeapReservation zero{UINT64_MAX, UINT64_MAX};
    Check(
        ReserveSharedOutputHeap<HeapTestOps>(
            *map, 8, 0, heap_size, zero
        ),
        "zero-output task succeeds when its shard is full"
    );
    Check(
        zero.task_base == 0 && zero.aggregate_vend == 4096,
        "zero-output task returns current vend without an address"
    );
    Check(
        SameSnapshot(*map, full_snapshot),
        "zero-output task changes no heap state"
    );

    SharedHeapReservation failed{UINT64_MAX, UINT64_MAX};
    Check(
        !ReserveSharedOutputHeap<HeapTestOps>(
            *map, 8, 1, heap_size, failed
        ),
        "no-wrap helper rejects a full shard"
    );
    Check(
        failed.task_base == 0 && failed.aggregate_vend == 0,
        "failed reservation returns a cleared result"
    );
    Check(
        SameSnapshot(*map, full_snapshot),
        "capacity failure changes no heap state"
    );

    Check(
        !ReserveSharedOutputHeap<HeapTestOps>(
            *map, 1, 4097, heap_size, failed
        ),
        "single reservation larger than shard span is rejected"
    );
    Check(
        SameSnapshot(*map, full_snapshot),
        "oversized request changes no heap state"
    );

    Check(
        !ReserveSharedOutputHeap<HeapTestOps>(
            *map, 1, UINT64_MAX, heap_size, failed
        ),
        "alignment overflow is rejected"
    );
    Check(
        SameSnapshot(*map, full_snapshot),
        "alignment overflow changes no heap state"
    );

    Check(
        !ReserveSharedOutputHeap<HeapTestOps>(
            *map, kMaxTasks, 0, heap_size, failed
        ),
        "out-of-range task id is rejected even for zero output"
    );
    Check(
        SameSnapshot(*map, full_snapshot),
        "out-of-range task id changes no heap state"
    );

    map->shared_heap_cursor[1].value = 1;
    const HeapSnapshot unaligned_cursor = Snapshot(*map);
    Check(
        !ReserveSharedOutputHeap<HeapTestOps>(
            *map, 1, 1024, heap_size, failed
        ),
        "unaligned cursor is rejected"
    );
    Check(
        SameSnapshot(*map, unaligned_cursor),
        "invalid cursor preflight changes no heap state"
    );

    ResetHeapState(*map);
    map->shared_heap_vend.value = -1;
    const HeapSnapshot negative_vend = Snapshot(*map);
    Check(
        !ReserveSharedOutputHeap<HeapTestOps>(
            *map, 0, 1024, heap_size, failed
        ),
        "negative aggregate vend is rejected"
    );
    Check(
        SameSnapshot(*map, negative_vend),
        "negative aggregate vend changes no heap state"
    );

    ResetHeapState(*map);
    map->shared_heap_vend.value = 1;
    const HeapSnapshot unaligned_vend = Snapshot(*map);
    Check(
        !ReserveSharedOutputHeap<HeapTestOps>(
            *map, 0, 1024, heap_size, failed
        ),
        "unaligned aggregate vend is rejected"
    );
    Check(
        SameSnapshot(*map, unaligned_vend),
        "unaligned aggregate vend changes no heap state"
    );

    ResetHeapState(*map);
    map->shared_heap_vend.value =
        static_cast<int64_t>(heap_size + kOutputAlignment);
    const HeapSnapshot oversized_vend = Snapshot(*map);
    Check(
        !ReserveSharedOutputHeap<HeapTestOps>(
            *map, 0, 1024, heap_size, failed
        ),
        "aggregate vend beyond heap capacity is rejected"
    );
    Check(
        SameSnapshot(*map, oversized_vend),
        "oversized aggregate vend changes no heap state"
    );

    ResetHeapState(*map);
    map->shared_heap_vend.value = static_cast<int64_t>(heap_size);
    const HeapSnapshot exhausted_vend = Snapshot(*map);
    Check(
        !ReserveSharedOutputHeap<HeapTestOps>(
            *map, 1, 1024, heap_size, failed
        ),
        "aggregate capacity exhaustion is rejected despite free target shard"
    );
    Check(
        SameSnapshot(*map, exhausted_vend),
        "aggregate capacity failure changes no heap state"
    );

    ResetHeapState(*map);
    map->shared_heap_cursor[2].value = -1;
    const HeapSnapshot negative_cursor = Snapshot(*map);
    Check(
        !ReserveSharedOutputHeap<HeapTestOps>(
            *map, 2, 1024, heap_size, failed
        ),
        "negative shard cursor is rejected"
    );
    Check(
        SameSnapshot(*map, negative_cursor),
        "negative shard cursor changes no heap state"
    );

    ResetHeapState(*map);
    map->shared_heap_cursor[2].value = 5120;
    const HeapSnapshot oversized_cursor = Snapshot(*map);
    Check(
        !ReserveSharedOutputHeap<HeapTestOps>(
            *map, 2, 1024, heap_size, failed
        ),
        "shard cursor beyond its span is rejected"
    );
    Check(
        SameSnapshot(*map, oversized_cursor),
        "oversized shard cursor changes no heap state"
    );

    ResetHeapState(*map);
    const HeapSnapshot tiny_heap = Snapshot(*map);
    SharedHeapReservation tiny_zero{};
    Check(
        ReserveSharedOutputHeap<HeapTestOps>(
            *map, 4, 0, kOutputAlignment - 1, tiny_zero
        ),
        "zero-output task does not require one allocatable shard"
    );
    Check(
        tiny_zero.task_base == 0 && tiny_zero.aggregate_vend == 0 &&
            SameSnapshot(*map, tiny_heap),
        "tiny-heap zero-output reservation preserves empty state"
    );
    Check(
        !ReserveSharedOutputHeap<HeapTestOps>(
            *map, 0, 1024, kOutputAlignment - 1, failed
        ),
        "heap too small to contain one aligned shard is rejected"
    );
    Check(
        SameSnapshot(*map, tiny_heap),
        "zero-span heap failure changes no heap state"
    );
}

void TestConcurrentReservations() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetHeapState(*map);
    constexpr uint32_t kThreads = 64;
    std::vector<SharedHeapReservation> reservations(kThreads);
    std::vector<uint64_t> reserve_bytes(kThreads);
    std::vector<uint8_t> success(kThreads, 0);
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};

    for (uint32_t task_id = 0; task_id < kThreads; ++task_id) {
        reserve_bytes[task_id] =
            (task_id / kSharedHeapShards % 4 + 1) * kOutputAlignment;
        workers.emplace_back([&, task_id] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            success[task_id] = ReserveSharedOutputHeap<HeapTestOps>(
                *map, task_id, reserve_bytes[task_id], kHeapBytes,
                reservations[task_id]
            ) ? 1 : 0;
        });
    }
    while (ready.load(std::memory_order_acquire) != kThreads) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (std::thread &worker : workers) {
        worker.join();
    }

    std::vector<std::pair<uint64_t, uint64_t>>
        shard_intervals[kSharedHeapShards];
    std::vector<std::pair<uint64_t, uint64_t>> vend_intervals;
    vend_intervals.reserve(kThreads);
    const uint64_t shard_span =
        SharedHeapAlignDown(kHeapBytes / kSharedHeapShards);
    uint64_t expected_total = 0;
    for (uint32_t task_id = 0; task_id < kThreads; ++task_id) {
        Check(success[task_id] != 0, "concurrent reservation succeeds");
        const uint32_t shard = task_id % kSharedHeapShards;
        const uint64_t bytes = reserve_bytes[task_id];
        shard_intervals[shard].push_back(
            {
                reservations[task_id].task_base,
                reservations[task_id].task_base + bytes
            }
        );
        vend_intervals.push_back(
            {
                reservations[task_id].aggregate_vend - bytes,
                reservations[task_id].aggregate_vend
            }
        );
        expected_total += bytes;
        Check(
            reservations[task_id].task_base >= shard * shard_span &&
                reservations[task_id].task_base + bytes <=
                    (shard + 1) * shard_span,
            "concurrent reservation remains inside its assigned shard"
        );
    }
    for (uint32_t shard = 0; shard < kSharedHeapShards; ++shard) {
        std::sort(
            shard_intervals[shard].begin(),
            shard_intervals[shard].end()
        );
        uint64_t next = shard * shard_span;
        for (const auto &interval : shard_intervals[shard]) {
            Check(
                interval.first == next,
                "variable-size same-shard reservations are unique and gap-free"
            );
            next = interval.second;
        }
        Check(
            map->shared_heap_cursor[shard].value ==
                static_cast<int64_t>(next - shard * shard_span),
            "concurrent shard cursor equals its reserved byte sum"
        );
    }
    std::sort(vend_intervals.begin(), vend_intervals.end());
    uint64_t next_vend = 0;
    for (const auto &interval : vend_intervals) {
        Check(
            interval.first == next_vend,
            "variable-size aggregate vend prefixes are unique and gap-free"
        );
        next_vend = interval.second;
    }
    Check(
        map->shared_heap_vend.value ==
            static_cast<int64_t>(expected_total) &&
            next_vend == expected_total,
        "concurrent aggregate vend equals all successful reservations"
    );
}

void TestInterleavingAndTerminalCapacityRace() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetHeapState(*map);
    HeapInterleaveOps::race_cursor = &map->shared_heap_cursor[0].value;
    HeapInterleaveOps::race_vend = &map->shared_heap_vend.value;
    HeapInterleaveOps::injected_delta = kOutputAlignment;
    HeapInterleaveOps::advance_vend = false;
    SharedHeapReservation reservation{UINT64_MAX, UINT64_MAX};
    Check(
        ReserveSharedOutputHeap<HeapInterleaveOps>(
            *map, 0, 1024, kHeapBytes, reservation
        ),
        "stale preflight snapshot accepts a legal concurrent reservation"
    );
    Check(
        reservation.task_base == kOutputAlignment &&
            reservation.aggregate_vend == kOutputAlignment,
        "physical cursor and aggregate vend may linearize in different orders"
    );
    Check(
        map->shared_heap_cursor[0].value == 2 * kOutputAlignment &&
            map->shared_heap_vend.value == kOutputAlignment,
        "paused competitor may own a cursor interval before publishing vend"
    );
    (void)__atomic_fetch_add(
        &map->shared_heap_vend.value, static_cast<int64_t>(kOutputAlignment),
        __ATOMIC_ACQ_REL
    );
    Check(
        map->shared_heap_cursor[0].value == 2 * kOutputAlignment &&
            map->shared_heap_vend.value == 2 * kOutputAlignment,
        "resumed competitor closes final cursor and vend byte sums"
    );

    // 两个 caller 都从 shard 尾部看到一份余量；竞争者先占满，当前
    // FetchAdd 随后越过 no-wrap 边界。此时必须 terminal fail 并保留
    // 5KiB cursor 现场，绝不能 Exchange 回 3KiB 覆盖竞争者的合法 1KiB。
    ResetHeapState(*map);
    const uint64_t heap_size = kSharedHeapShards * 4096;
    map->shared_heap_cursor[0].value = 3 * kOutputAlignment;
    map->shared_heap_vend.value = 3 * kOutputAlignment;
    HeapInterleaveOps::race_cursor = &map->shared_heap_cursor[0].value;
    HeapInterleaveOps::race_vend = &map->shared_heap_vend.value;
    HeapInterleaveOps::injected_delta = kOutputAlignment;
    HeapInterleaveOps::advance_vend = true;
    reservation = SharedHeapReservation{UINT64_MAX, UINT64_MAX};
    Check(
        !ReserveSharedOutputHeap<HeapInterleaveOps>(
            *map, 0, kOutputAlignment, heap_size, reservation
        ),
        "capacity race fails after the competing reservation fills the shard"
    );
    Check(
        map->shared_heap_cursor[0].value == 5 * kOutputAlignment &&
            map->shared_heap_vend.value == 4 * kOutputAlignment,
        "terminal capacity race preserves competitor progress and overrun evidence"
    );
    Check(
        reservation.task_base == 0 && reservation.aggregate_vend == 0,
        "terminal capacity race returns no usable reservation"
    );
}

}  // namespace

int main() {
    TestPaCase(1);
    TestPaCase(kDefaultBatches);
    TestAlignmentAndShardTail();
    TestBoundaryZeroAndPreflightFailures();
    TestConcurrentReservations();
    TestInterleavingAndTerminalCapacityRace();
    if (g_failures != 0) {
        std::fprintf(
            stderr, "shared heap reserve self-test failed: %d assertion(s)\n",
            g_failures
        );
        return 1;
    }
    std::printf("shared heap reserve self-test passed\n");
    return 0;
}
