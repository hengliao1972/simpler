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

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

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

    static int64_t Exchange(volatile int64_t *address, int64_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }
};

// 只在定向测试中模拟“预检完成后、FetchAdd 发射前”出现的非法状态变化。
// 生产 exact-turn 不允许该交错；helper 仍须拒绝异常 old value 并恢复两项状态。
struct HeapFaultOps : HeapTestOps {
    static volatile int64_t *fault_address;
    static int64_t injected_delta;

    static int64_t FetchAdd(volatile int64_t *address, int64_t value) {
        if (address == fault_address) {
            (void)__atomic_fetch_add(
                address, injected_delta, __ATOMIC_ACQ_REL
            );
            fault_address = nullptr;
        }
        return HeapTestOps::FetchAdd(address, value);
    }
};

volatile int64_t *HeapFaultOps::fault_address = nullptr;
int64_t HeapFaultOps::injected_delta = 0;

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
            "PA Case1 task base follows task-id shard cursor"
        );
        expected_cursor[shard] += total;
        expected_vend += total;
        Check(
            reservation.aggregate_vend == expected_vend,
            "PA Case1 aggregate vend is exact"
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
}

void TestBoundaryZeroAndPreflightFailures() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetHeapState(*map);
    const uint64_t heap_size =
        kSharedHeapShards * 4096;

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

void TestAtomicFaultRollback() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetHeapState(*map);
    SharedHeapReservation reservation{};

    const HeapSnapshot initial = Snapshot(*map);
    HeapFaultOps::fault_address = &map->shared_heap_cursor[0].value;
    HeapFaultOps::injected_delta = 2048;
    Check(
        !ReserveSharedOutputHeap<HeapFaultOps>(
            *map, 0, 1024, kHeapBytes, reservation
        ),
        "unexpected shard FetchAdd old value is rejected"
    );
    Check(
        SameSnapshot(*map, initial),
        "shard FetchAdd anomaly is fully rolled back"
    );

    HeapFaultOps::fault_address = &map->shared_heap_vend.value;
    HeapFaultOps::injected_delta = 2048;
    Check(
        !ReserveSharedOutputHeap<HeapFaultOps>(
            *map, 0, 1024, kHeapBytes, reservation
        ),
        "unexpected vend FetchAdd old value is rejected"
    );
    Check(
        SameSnapshot(*map, initial),
        "vend anomaly rolls back both vend and shard cursor"
    );

    // 再从非零合法快照注入相同故障，证明 rollback 恢复的是预检值，
    // 而不是测试恰好从全零开始才看似成功。
    ResetHeapState(*map);
    map->shared_heap_cursor[0].value = 4096;
    map->shared_heap_cursor[1].value = 4096;
    map->shared_heap_vend.value = 8192;
    const HeapSnapshot nonzero = Snapshot(*map);

    HeapFaultOps::fault_address = &map->shared_heap_cursor[0].value;
    HeapFaultOps::injected_delta = 2048;
    reservation = SharedHeapReservation{UINT64_MAX, UINT64_MAX};
    Check(
        !ReserveSharedOutputHeap<HeapFaultOps>(
            *map, 0, 1024, kHeapBytes, reservation
        ),
        "nonzero cursor FetchAdd anomaly is rejected"
    );
    Check(
        SameSnapshot(*map, nonzero),
        "cursor anomaly restores the nonzero snapshot"
    );
    Check(
        reservation.task_base == 0 && reservation.aggregate_vend == 0,
        "cursor anomaly clears the returned reservation"
    );

    HeapFaultOps::fault_address = &map->shared_heap_vend.value;
    HeapFaultOps::injected_delta = 2048;
    reservation = SharedHeapReservation{UINT64_MAX, UINT64_MAX};
    Check(
        !ReserveSharedOutputHeap<HeapFaultOps>(
            *map, 0, 1024, kHeapBytes, reservation
        ),
        "nonzero vend FetchAdd anomaly is rejected"
    );
    Check(
        SameSnapshot(*map, nonzero),
        "vend anomaly restores both nonzero control words"
    );
    Check(
        reservation.task_base == 0 && reservation.aggregate_vend == 0,
        "vend anomaly clears the returned reservation"
    );
}

}  // namespace

int main() {
    TestPaCase(1);
    TestPaCase(kDefaultBatches);
    TestAlignmentAndShardTail();
    TestBoundaryZeroAndPreflightFailures();
    TestAtomicFaultRollback();
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
