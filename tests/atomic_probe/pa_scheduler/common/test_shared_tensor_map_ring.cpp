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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <tuple>
#include <vector>

// 直接实例化 shared 生产 primitive；测试只提供 CPU 原子适配和确定性事件
// ledger，不复制 append/lookup/reclaim 算法，也不把 x86 一致性冒充 A5 DCCI。
#define PA_DEVICE inline
#define PA_GM
#include "pa_shared_tensormap.h"

namespace {

using pa_scheduler::SharedAppendPreparedTask;
using pa_scheduler::SharedAppendCheck;
using pa_scheduler::SharedBucketState;
using pa_scheduler::SharedCheckTaskAppend;
using pa_scheduler::SharedComputeOrderedReclaimCandidate;
using pa_scheduler::SharedHasExactTaskTurn;
using pa_scheduler::SharedLookupRegion;
using pa_scheduler::SharedPreflightTaskAppend;
using pa_scheduler::SharedPublishTaskCommit;
using pa_scheduler::SharedReadRegionSlot;
using pa_scheduler::SharedRefreshReclaimForTask;
using pa_scheduler::SharedRegionPayload;
using pa_scheduler::SharedRegionSlot;
using pa_scheduler::SharedRegionValue;
using pa_scheduler::SharedRetireBucket;
using pa_scheduler::SharedTensorMapSidecar;
using pa_scheduler::SharedTensorMapSlotIndex;
using pa_scheduler::TensorMapHash;
using pa_scheduler::kMapBucketCapacity;
using pa_scheduler::kMapBuckets;
using pa_scheduler::kMapCapacity;
using pa_scheduler::kSharedMapEmptySeq;

static_assert(PTO_FDWIC_SHARED_MAP == 1, "this test must compile as the shared TensorMap mode");
static_assert(sizeof(SharedRegionValue) == 32, "shared logical value ABI changed");
static_assert(sizeof(SharedRegionPayload) == 64, "shared payload must occupy one cache line");
static_assert(alignof(SharedRegionPayload) == 64, "shared payload alignment changed");
static_assert(sizeof(SharedRegionSlot) == 128, "shared slot must occupy two cache lines");
static_assert(offsetof(SharedRegionSlot, seq) == 64, "shared seq cache line offset changed");
static_assert(sizeof(SharedBucketState) == 128, "shared bucket control ABI changed");
static_assert(offsetof(SharedBucketState, tail) == 64, "shared head/tail cache lines merged");
static_assert(sizeof(SharedTensorMapSidecar) == 4735680, "shared sidecar ABI changed");
static_assert(alignof(SharedTensorMapSidecar) == 64, "shared sidecar alignment changed");
static_assert(offsetof(SharedTensorMapSidecar, buckets) == 128, "shared bucket offset changed");
static_assert(offsetof(SharedTensorMapSidecar, slots) == 16512, "shared slot offset changed");
static_assert(
    offsetof(SharedTensorMapSidecar, shared_outputs) == 2113664,
    "shared output table offset changed"
);
static_assert(
    offsetof(SharedTensorMapSidecar, shared_heap_cursor) == 4735104,
    "shared heap cursor offset changed"
);
static_assert(
    offsetof(SharedTensorMapSidecar, shared_heap_vend) == 4735616,
    "shared heap vend offset changed"
);

enum class EventKind : uint8_t {
    Load,
    Exchange,
    Invalidate,
    Flush,
};

struct Event {
    EventKind kind;
    const void *address;
    int64_t argument;
    int64_t result;
};

struct RecordingOps {
    static std::vector<Event> events;
    static bool record;
    static const void *mutate_payload;
    static volatile int64_t *mutate_seq;
    static int64_t mutate_seq_value;
    static bool mutate_once;

    static int64_t Load(volatile int64_t *address) {
        // 与 CPU scheduler 相同，以 acquire atomic add-zero 表达协议观察；
        // 返回值只用于正确性，不解释为 A5 atomic 完成时延。
        const int64_t value =
            __atomic_fetch_add(address, static_cast<int64_t>(0), __ATOMIC_ACQUIRE);
        if (record) {
            events.push_back({EventKind::Load, ConstAddress(address), 0, value});
        }
        return value;
    }

    static int64_t Exchange(volatile int64_t *address, int64_t value) {
        const int64_t old = __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
        if (record) {
            events.push_back({EventKind::Exchange, ConstAddress(address), value, old});
        }
        return old;
    }

    static void InvalidateRegion(const void *address, uint64_t bytes) {
        if (record) {
            events.push_back(
                {EventKind::Invalidate, address, static_cast<int64_t>(bytes), 0}
            );
        }
        // 该注入点只制造“第一次 seq 检查后，槽被另一 lap 复用”的确定性交错，
        // 用来证明第二次 seq 检查有效；它不模拟 cache line 内容或 DCCI。
        if (mutate_once && address == mutate_payload && mutate_seq != nullptr) {
            __atomic_store_n(mutate_seq, mutate_seq_value, __ATOMIC_RELEASE);
            mutate_once = false;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
    }

    static void FlushRegion(void *address, uint64_t bytes) {
        if (record) {
            events.push_back(
                {EventKind::Flush, address, static_cast<int64_t>(bytes), 0}
            );
        }
        std::atomic_thread_fence(std::memory_order_release);
    }

    static void ResetEvents() {
        events.clear();
        record = true;
        mutate_payload = nullptr;
        mutate_seq = nullptr;
        mutate_seq_value = 0;
        mutate_once = false;
    }

    static void DisableEvents() {
        events.clear();
        record = false;
        mutate_payload = nullptr;
        mutate_seq = nullptr;
        mutate_seq_value = 0;
        mutate_once = false;
    }

private:
    static const void *ConstAddress(volatile int64_t *address) {
        return const_cast<const int64_t *>(address);
    }
};

std::vector<Event> RecordingOps::events;
bool RecordingOps::record = true;
const void *RecordingOps::mutate_payload = nullptr;
volatile int64_t *RecordingOps::mutate_seq = nullptr;
int64_t RecordingOps::mutate_seq_value = 0;
bool RecordingOps::mutate_once = false;

int g_failures = 0;

void Expect(bool condition, const char *test, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] shared TensorMap ring/%s: %s\n", test, message);
    ++g_failures;
}

void ExpectEqual(int64_t actual, int64_t expected, const char *test, const char *field) {
    if (actual == expected) {
        return;
    }
    std::fprintf(
        stderr, "[FAIL] shared TensorMap ring/%s: %s actual=%lld expected=%lld\n",
        test, field, static_cast<long long>(actual), static_cast<long long>(expected)
    );
    ++g_failures;
}

void StoreControl(volatile int64_t *address, int64_t value) {
    __atomic_store_n(address, value, __ATOMIC_RELAXED);
}

int64_t LoadControl(volatile int64_t *address) {
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

void ResetSharedTensorMap(SharedTensorMapSidecar &map) {
    // 与 scheduler 的正式 host reset 保持相同控制字初值。payload 保持
    // 惰性：seq=-1 时任何旧字节都不可见；正式 reset 额外清零整块
    // sidecar，只是为了让 host 诊断中的保留字节也确定。
    StoreControl(&map.committed_tasks.value, 0);
    StoreControl(&map.reclaim_upto.value, -1);
    for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
        StoreControl(&map.buckets[bucket].head.value, 0);
        StoreControl(&map.buckets[bucket].tail.value, 0);
    }
    for (uint32_t slot = 0; slot < kMapCapacity; ++slot) {
        StoreControl(&map.slots[slot].seq.value, kSharedMapEmptySeq);
    }
    // descriptor 的零值由正式 host 对整块 sidecar 的 memset 建立；独立 ring
    // 用例不读取 descriptor，但仍把两组发布控制字初始化成协议要求的 -1，
    // 防止后续 symbol 子测把 task 0 误判成已发布。
    for (uint32_t task = 0; task < pa_scheduler::kMaxTasks; ++task) {
        for (uint32_t output = 0; output < pa_scheduler::kSharedOutputMaxPerTask; ++output) {
            StoreControl(&map.shared_outputs[task].published[output].value, -1);
            StoreControl(&map.shared_outputs[task].last_writer[output].value, -1);
        }
    }
}

std::unique_ptr<SharedTensorMapSidecar> NewMap() {
    // sidecar 超过 2 MiB，必须 heap 分配，不能依赖 host 线程栈容量。
    auto map = std::unique_ptr<SharedTensorMapSidecar>(new SharedTensorMapSidecar);
    ResetSharedTensorMap(*map);
    RecordingOps::ResetEvents();
    return map;
}

SharedRegionValue MakeRegion(
    uint64_t buffer_addr, uint64_t lo, uint64_t hi, int32_t producer
) {
    return {buffer_addr, lo, hi, producer, 0};
}

uint64_t FindAddressOutsideBucket(uint64_t seed, uint32_t excluded_bucket) {
    uint64_t address = seed;
    while (TensorMapHash(address) == excluded_bucket) {
        address += 64;
    }
    return address;
}

enum class CommitResult : uint8_t {
    Pending,
    Committed,
    Failed,
};

CommitResult TryCommitTask(
    SharedTensorMapSidecar &map, int32_t task_id,
    const std::vector<SharedRegionValue> &entries,
    int32_t heap_window = INT32_MAX
) {
    // 这只是按公开 primitive 组合的一步式确定性 driver：未持有 task turn
    // 时不触碰 reclaim/head/tail；持有 exact turn 后按 N-H-1 推进回收，
    // 再做整任务预检、append 和 commit。
    if (RecordingOps::Load(&map.committed_tasks.value) != task_id) {
        return CommitResult::Pending;
    }
    int64_t reclaim_upto = -2;
    if (!SharedRefreshReclaimForTask<RecordingOps>(
            map, task_id, heap_window, reclaim_upto
        )) {
        return CommitResult::Failed;
    }
    if (!SharedPreflightTaskAppend<RecordingOps>(
            map, entries.data(), static_cast<uint32_t>(entries.size()),
            reclaim_upto
        )) {
        return CommitResult::Failed;
    }
    if (!SharedAppendPreparedTask<RecordingOps>(
            map, entries.data(), static_cast<uint32_t>(entries.size())
        )) {
        return CommitResult::Failed;
    }
    if (!SharedPublishTaskCommit<RecordingOps>(map, task_id)) {
        return CommitResult::Failed;
    }
    return CommitResult::Committed;
}

struct LogicalTuple {
    uint64_t buffer_addr;
    uint64_t lo;
    uint64_t hi;
    int32_t producer;

    bool operator<(const LogicalTuple &other) const {
        return std::tie(buffer_addr, lo, hi, producer) <
               std::tie(other.buffer_addr, other.lo, other.hi, other.producer);
    }

    bool operator==(const LogicalTuple &other) const {
        return buffer_addr == other.buffer_addr && lo == other.lo &&
               hi == other.hi && producer == other.producer;
    }
};

LogicalTuple ToTuple(const SharedRegionValue &value) {
    return {value.buffer_addr, value.lo, value.hi, value.producer};
}

bool SnapshotLogicalMap(
    SharedTensorMapSidecar &map, std::vector<LogicalTuple> &snapshot
) {
    snapshot.clear();
    for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
        const int64_t head = LoadControl(&map.buckets[bucket].head.value);
        const int64_t tail = LoadControl(&map.buckets[bucket].tail.value);
        if (head < 0 || tail < head ||
            static_cast<uint64_t>(tail - head) > kMapBucketCapacity) {
            return false;
        }
        for (int64_t cursor = head; cursor < tail; ++cursor) {
            SharedRegionValue value{};
            if (!SharedReadRegionSlot<RecordingOps>(
                    map, bucket, static_cast<uint64_t>(cursor), value
                )) {
                return false;
            }
            snapshot.push_back(ToTuple(value));
        }
    }
    std::sort(snapshot.begin(), snapshot.end());
    return true;
}

size_t FindEvent(
    EventKind kind, const void *address, size_t begin,
    bool check_argument = false, int64_t argument = 0
) {
    for (size_t index = begin; index < RecordingOps::events.size(); ++index) {
        const Event &event = RecordingOps::events[index];
        if (event.kind == kind && event.address == address &&
            (!check_argument || event.argument == argument)) {
            return index;
        }
    }
    return RecordingOps::events.size();
}

void TestAbiResetAndZeroEntryCommit() {
    constexpr const char *kTest = "abi-reset-zero-entry";
    auto map = NewMap();
    RecordingOps::DisableEvents();

    ExpectEqual(LoadControl(&map->committed_tasks.value), 0, kTest, "committed reset");
    ExpectEqual(LoadControl(&map->reclaim_upto.value), -1, kTest, "reclaim reset");
    for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
        if (LoadControl(&map->buckets[bucket].head.value) != 0 ||
            LoadControl(&map->buckets[bucket].tail.value) != 0) {
            Expect(false, kTest, "bucket cursor reset");
            break;
        }
    }
    for (uint32_t slot = 0; slot < kMapCapacity; ++slot) {
        if (LoadControl(&map->slots[slot].seq.value) != kSharedMapEmptySeq) {
            Expect(false, kTest, "slot seq reset");
            break;
        }
    }

    const std::vector<SharedRegionValue> empty;
    Expect(
        TryCommitTask(*map, 0, empty) == CommitResult::Committed,
        kTest, "task 0 empty delta commit"
    );
    Expect(
        TryCommitTask(*map, 1, empty) == CommitResult::Committed,
        kTest, "task 1 empty delta commit"
    );
    ExpectEqual(LoadControl(&map->committed_tasks.value), 2, kTest, "empty commit sequencer");
    ExpectEqual(LoadControl(&map->reclaim_upto.value), -1, kTest, "empty commit reclaim");
    Expect(
        !SharedPublishTaskCommit<RecordingOps>(*map, 1),
        kTest, "repeated commit is rejected"
    );
    ExpectEqual(
        LoadControl(&map->committed_tasks.value), 2, kTest,
        "repeated commit preserves the advanced frontier"
    );
    StoreControl(&map->committed_tasks.value, 0);
    Expect(
        !SharedPublishTaskCommit<RecordingOps>(*map, 1),
        kTest, "future commit is rejected"
    );
    ExpectEqual(
        LoadControl(&map->committed_tasks.value), 0, kTest,
        "future commit preserves the earlier frontier"
    );
    for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
        if (LoadControl(&map->buckets[bucket].tail.value) != 0) {
            Expect(false, kTest, "empty commit changed a bucket tail");
            break;
        }
    }
}

void TestPublicationOrderAndDoubleSeqCheck() {
    constexpr const char *kTest = "publication-order-double-seq";
    auto map = NewMap();
    const SharedRegionValue entry = MakeRegion(0x100000000ULL, 0, 64, 0);
    const uint32_t bucket = TensorMapHash(entry.buffer_addr);
    SharedRegionSlot &slot = map->slots[SharedTensorMapSlotIndex(bucket, 0)];

    RecordingOps::ResetEvents();
    Expect(
        TryCommitTask(*map, 0, {entry}) == CommitResult::Committed,
        kTest, "single entry commit"
    );

    const void *seq_address =
        const_cast<const int64_t *>(&slot.seq.value);
    const void *payload_address = &slot.payload;
    const void *tail_address =
        const_cast<const int64_t *>(&map->buckets[bucket].tail.value);
    const size_t seq_invalidate =
        FindEvent(EventKind::Exchange, seq_address, 0, true, kSharedMapEmptySeq);
    const size_t payload_invalidate =
        FindEvent(EventKind::Invalidate, payload_address, seq_invalidate + 1);
    const size_t payload_flush =
        FindEvent(EventKind::Flush, payload_address, payload_invalidate + 1);
    const size_t seq_publish =
        FindEvent(EventKind::Exchange, seq_address, payload_flush + 1, true, 0);
    const size_t tail_publish =
        FindEvent(EventKind::Exchange, tail_address, seq_publish + 1, true, 1);
    Expect(
        seq_invalidate < payload_invalidate &&
            payload_invalidate < payload_flush &&
            payload_flush < seq_publish && seq_publish < tail_publish,
        kTest, "writer event order"
    );

    RecordingOps::ResetEvents();
    SharedRegionValue snapshot{};
    Expect(
        SharedReadRegionSlot<RecordingOps>(*map, bucket, 0, snapshot),
        kTest, "published slot read"
    );
    Expect(ToTuple(snapshot) == ToTuple(entry), kTest, "published payload contents");
    const size_t first_load = FindEvent(EventKind::Load, seq_address, 0);
    const size_t read_invalidate =
        FindEvent(EventKind::Invalidate, payload_address, first_load + 1);
    const size_t second_load =
        FindEvent(EventKind::Load, seq_address, read_invalidate + 1);
    Expect(
        first_load < read_invalidate && read_invalidate < second_load,
        kTest, "reader acquire-invalidate-double-check order"
    );

    RecordingOps::ResetEvents();
    RecordingOps::mutate_payload = payload_address;
    RecordingOps::mutate_seq = &slot.seq.value;
    RecordingOps::mutate_seq_value =
        static_cast<int64_t>(kMapBucketCapacity);
    RecordingOps::mutate_once = true;
    SharedRegionValue raced_snapshot{};
    Expect(
        !SharedReadRegionSlot<RecordingOps>(
            *map, bucket, 0, raced_snapshot
        ),
        kTest, "second seq check rejects deterministic ABA"
    );
    Expect(!RecordingOps::mutate_once, kTest, "ABA injection point reached");
}

void TestVersionsWindowAndMultipleBuckets() {
    constexpr const char *kTest = "versions-window-buckets";
    auto map = NewMap();
    RecordingOps::DisableEvents();

    const uint64_t versioned_address = 0x200000000ULL;
    const uint32_t versioned_bucket = TensorMapHash(versioned_address);
    const uint64_t stale_address =
        FindAddressOutsideBucket(0x210000000ULL, versioned_bucket);
    const uint32_t stale_bucket = TensorMapHash(stale_address);
    const uint64_t lower_address =
        FindAddressOutsideBucket(0x220000000ULL, stale_bucket);
    const uint64_t future_address =
        FindAddressOutsideBucket(0x230000000ULL, TensorMapHash(lower_address));

    const std::vector<SharedRegionValue> task0 = {
        MakeRegion(versioned_address, 0, 64, 0),
        MakeRegion(versioned_address, 128, 192, 0),
        MakeRegion(stale_address, 0, 64, 0),
    };
    const std::vector<SharedRegionValue> task1 = {
        MakeRegion(versioned_address, 0, 64, 1),
        MakeRegion(lower_address, 0, 64, 1),
    };
    const std::vector<SharedRegionValue> task2 = {
        MakeRegion(versioned_address, 0, 64, 2),
        MakeRegion(future_address, 0, 64, 2),
    };
    Expect(TryCommitTask(*map, 0, task0) == CommitResult::Committed, kTest, "task 0");
    Expect(TryCommitTask(*map, 1, task1) == CommitResult::Committed, kTest, "task 1");
    Expect(TryCommitTask(*map, 2, task2) == CommitResult::Committed, kTest, "task 2");
    ExpectEqual(
        LoadControl(&map->buckets[versioned_bucket].tail.value), 4,
        kTest, "same bucket multi-entry tail"
    );

    bool protocol_ok = false;
    int32_t producer = SharedLookupRegion<RecordingOps>(
        *map, MakeRegion(versioned_address, 16, 32, -1), 3, 2, protocol_ok
    );
    Expect(protocol_ok, kTest, "versioned lookup protocol");
    ExpectEqual(producer, 2, kTest, "same address maximum producer");

    producer = SharedLookupRegion<RecordingOps>(
        *map, MakeRegion(versioned_address, 16, 32, -1), 2, 2, protocol_ok
    );
    Expect(protocol_ok, kTest, "historical lookup protocol");
    ExpectEqual(producer, 1, kTest, "future producer excluded");

    producer = SharedLookupRegion<RecordingOps>(
        *map, MakeRegion(stale_address, 0, 32, -1), 3, 2, protocol_ok
    );
    Expect(protocol_ok, kTest, "stale lookup protocol");
    ExpectEqual(producer, -1, kTest, "producer below N-H excluded");

    producer = SharedLookupRegion<RecordingOps>(
        *map, MakeRegion(lower_address, 0, 32, -1), 3, 2, protocol_ok
    );
    Expect(protocol_ok, kTest, "lower boundary protocol");
    ExpectEqual(producer, 1, kTest, "producer at N-H accepted");

    producer = SharedLookupRegion<RecordingOps>(
        *map, MakeRegion(future_address, 0, 32, -1), 2, 2, protocol_ok
    );
    Expect(protocol_ok, kTest, "upper boundary protocol");
    ExpectEqual(producer, -1, kTest, "producer at N excluded");

    producer = SharedLookupRegion<RecordingOps>(
        *map, MakeRegion(versioned_address, 64, 128, -1), 3, 3, protocol_ok
    );
    Expect(protocol_ok, kTest, "half-open lookup protocol");
    ExpectEqual(producer, -1, kTest, "touching half-open ranges do not overlap");
}

void TestOrderedReclaimFormulaAndExactTurn() {
    constexpr const char *kTest = "ordered-reclaim-exact-turn";
    int64_t candidate = -2;
    Expect(
        SharedComputeOrderedReclaimCandidate(0, 64, candidate),
        kTest, "task 0 reclaim formula"
    );
    ExpectEqual(candidate, -1, kTest, "task 0 reclaim boundary");
    Expect(
        SharedComputeOrderedReclaimCandidate(64, 64, candidate),
        kTest, "task H reclaim formula"
    );
    ExpectEqual(candidate, -1, kTest, "task H reclaim boundary");
    Expect(
        SharedComputeOrderedReclaimCandidate(65, 64, candidate),
        kTest, "task H+1 reclaim formula"
    );
    ExpectEqual(candidate, 0, kTest, "task H+1 inclusive reclaim");
    Expect(
        SharedComputeOrderedReclaimCandidate(1279, 64, candidate),
        kTest, "Case1 final task reclaim formula"
    );
    ExpectEqual(candidate, 1214, kTest, "Case1 final task reclaim boundary");
    Expect(
        !SharedComputeOrderedReclaimCandidate(-1, 64, candidate),
        kTest, "negative task rejected"
    );
    Expect(
        !SharedComputeOrderedReclaimCandidate(0, -1, candidate),
        kTest, "negative heap window rejected"
    );

    {
        auto ahead_map = NewMap();
        StoreControl(&ahead_map->committed_tasks.value, 65);
        StoreControl(&ahead_map->reclaim_upto.value, 1);
        RecordingOps::ResetEvents();
        int64_t rejected = -2;
        Expect(
            !SharedRefreshReclaimForTask<RecordingOps>(
                *ahead_map, 65, 64, rejected
            ),
            kTest, "reclaim state ahead of candidate rejected"
        );
        ExpectEqual(
            LoadControl(&ahead_map->reclaim_upto.value), 1,
            kTest, "ahead reclaim state preserved"
        );
        for (const Event &event : RecordingOps::events) {
            if (event.kind == EventKind::Exchange) {
                Expect(false, kTest, "reclaim regression issued Exchange");
                break;
            }
        }
    }

    auto map = NewMap();
    RecordingOps::DisableEvents();
    const uint64_t address = 0x300000000ULL;
    const uint32_t bucket = TensorMapHash(address);

    Expect(
        TryCommitTask(*map, 0, {MakeRegion(address, 0, 32, 0)}) ==
            CommitResult::Committed,
        kTest, "producer 0 commit"
    );
    Expect(
        TryCommitTask(*map, 1, {MakeRegion(address, 64, 96, 1)}) ==
            CommitResult::Committed,
        kTest, "producer 1 commit"
    );
    const std::vector<SharedRegionValue> empty;
    Expect(
        TryCommitTask(*map, 2, empty, 2) == CommitResult::Committed,
        kTest, "boundary zero-entry task commit"
    );
    ExpectEqual(
        LoadControl(&map->committed_tasks.value), 3,
        kTest, "sequencer before exact turn"
    );
    ExpectEqual(
        LoadControl(&map->reclaim_upto.value), -1,
        kTest, "task N=H does not reclaim"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].head.value), 0,
        kTest, "boundary task keeps head"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].tail.value), 2,
        kTest, "boundary task keeps tail"
    );

    RecordingOps::ResetEvents();
    int64_t reclaim_upto = -2;
    Expect(
        !SharedRefreshReclaimForTask<RecordingOps>(
            *map, 2, 2, reclaim_upto
        ),
        kTest, "stale actor rejected"
    );
    Expect(
        !SharedRefreshReclaimForTask<RecordingOps>(
            *map, 4, 2, reclaim_upto
        ),
        kTest, "future actor rejected"
    );
    for (const Event &event : RecordingOps::events) {
        if (event.kind == EventKind::Exchange ||
            event.kind == EventKind::Flush) {
            Expect(false, kTest, "out-of-turn actor changed shared state");
            break;
        }
    }
    ExpectEqual(
        LoadControl(&map->reclaim_upto.value), -1,
        kTest, "out-of-turn actor preserves reclaim"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].head.value), 0,
        kTest, "out-of-turn actor preserves head"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].tail.value), 2,
        kTest, "out-of-turn actor preserves tail"
    );

    RecordingOps::DisableEvents();
    Expect(
        SharedHasExactTaskTurn<RecordingOps>(*map, 3),
        kTest, "task 3 owns exact turn"
    );
    Expect(
        TryCommitTask(*map, 3, empty, 2) == CommitResult::Committed,
        kTest, "zero-entry task advances ordered reclaim"
    );
    ExpectEqual(
        LoadControl(&map->committed_tasks.value), 4,
        kTest, "zero-entry task publishes commit"
    );
    ExpectEqual(
        LoadControl(&map->reclaim_upto.value), 0,
        kTest, "zero-entry task advances inclusive reclaim"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].head.value), 0,
        kTest, "zero-entry task does not scan unrelated bucket"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].tail.value), 2,
        kTest, "zero-entry task does not append"
    );

    reclaim_upto = LoadControl(&map->reclaim_upto.value);
    Expect(
        SharedRetireBucket<RecordingOps>(*map, bucket, reclaim_upto),
        kTest, "lazy retire at published boundary"
    );
    ExpectEqual(LoadControl(&map->buckets[bucket].head.value), 1, kTest, "producer 0 retired");

    bool protocol_ok = false;
    const int32_t producer = SharedLookupRegion<RecordingOps>(
        *map, MakeRegion(address, 64, 96, -1), 2, 2, protocol_ok
    );
    Expect(protocol_ok, kTest, "surviving producer lookup protocol");
    ExpectEqual(producer, 1, kTest, "producer above reclaim boundary survives");
}

void TestAbsoluteSeqMultipleLapsAndAba() {
    constexpr const char *kTest = "absolute-seq-multiple-laps";
    auto map = NewMap();
    RecordingOps::DisableEvents();
    const uint64_t address = 0x400000000ULL;
    const uint32_t bucket = TensorMapHash(address);
    constexpr uint32_t kIterations = 3 * kMapBucketCapacity + 7;

    for (uint32_t task = 0; task < kIterations; ++task) {
        const CommitResult result = TryCommitTask(
            *map, static_cast<int32_t>(task),
            {MakeRegion(address, 0, 64, static_cast<int32_t>(task))},
            0
        );
        if (result != CommitResult::Committed) {
            std::fprintf(
                stderr, "[FAIL] shared TensorMap ring/%s: commit failed task=%u\n",
                kTest, task
            );
            ++g_failures;
            return;
        }
        if (task == kMapBucketCapacity ||
            task == 2 * kMapBucketCapacity ||
            task == 3 * kMapBucketCapacity) {
            SharedRegionSlot &wrapped =
                map->slots[SharedTensorMapSlotIndex(bucket, task)];
            ExpectEqual(
                LoadControl(&wrapped.seq.value), task, kTest,
                "absolute seq after physical wrap"
            );
        }
    }

    ExpectEqual(
        LoadControl(&map->buckets[bucket].tail.value), kIterations,
        kTest, "absolute tail after laps"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].head.value), kIterations - 1,
        kTest, "single live entry after laps"
    );
    ExpectEqual(
        LoadControl(&map->reclaim_upto.value), kIterations - 2,
        kTest, "three-lap ordered reclaim boundary"
    );
    const uint64_t last_cursor = kIterations - 1;
    SharedRegionSlot &last_slot =
        map->slots[SharedTensorMapSlotIndex(bucket, last_cursor)];
    ExpectEqual(
        LoadControl(&last_slot.seq.value), last_cursor, kTest,
        "latest absolute seq"
    );

    bool protocol_ok = false;
    const int32_t producer = SharedLookupRegion<RecordingOps>(
        *map, MakeRegion(address, 0, 64, -1),
        static_cast<int32_t>(kIterations), 1, protocol_ok
    );
    Expect(protocol_ok, kTest, "post-wrap lookup protocol");
    ExpectEqual(producer, kIterations - 1, kTest, "post-wrap latest producer");

    RecordingOps::ResetEvents();
    RecordingOps::mutate_payload = &last_slot.payload;
    RecordingOps::mutate_seq = &last_slot.seq.value;
    RecordingOps::mutate_seq_value =
        static_cast<int64_t>(last_cursor + kMapBucketCapacity);
    RecordingOps::mutate_once = true;
    SharedRegionValue raced{};
    Expect(
        !SharedReadRegionSlot<RecordingOps>(
            *map, bucket, last_cursor, raced
        ),
        kTest, "double check rejects next-lap seq"
    );
}

void TestCapacityFailureIsAllOrNothing() {
    constexpr const char *kTest = "capacity-all-or-nothing";
    auto map = NewMap();
    RecordingOps::DisableEvents();
    const uint64_t full_address = 0x500000000ULL;
    const uint32_t full_bucket = TensorMapHash(full_address);

    for (uint32_t task = 0; task < kMapBucketCapacity; ++task) {
        const CommitResult result = TryCommitTask(
            *map, static_cast<int32_t>(task),
            {MakeRegion(
                full_address, static_cast<uint64_t>(task) * 16,
                static_cast<uint64_t>(task) * 16 + 8,
                static_cast<int32_t>(task)
            )}
        );
        if (result != CommitResult::Committed) {
            std::fprintf(
                stderr, "[FAIL] shared TensorMap ring/%s: fill failed task=%u\n",
                kTest, task
            );
            ++g_failures;
            return;
        }
    }

    std::vector<LogicalTuple> before;
    Expect(SnapshotLogicalMap(*map, before), kTest, "snapshot before overflow");
    const int64_t full_head = LoadControl(&map->buckets[full_bucket].head.value);
    const int64_t full_tail = LoadControl(&map->buckets[full_bucket].tail.value);
    const uint64_t other_address =
        FindAddressOutsideBucket(0x510000000ULL, full_bucket);
    const uint32_t other_bucket = TensorMapHash(other_address);
    const int64_t other_tail = LoadControl(&map->buckets[other_bucket].tail.value);
    const uint32_t other_slot_index =
        SharedTensorMapSlotIndex(other_bucket, static_cast<uint64_t>(other_tail));
    const int64_t other_seq =
        LoadControl(&map->slots[other_slot_index].seq.value);

    const std::vector<SharedRegionValue> overflowing = {
        MakeRegion(other_address, 0, 8, kMapBucketCapacity),
        MakeRegion(full_address, 4096, 4104, kMapBucketCapacity),
    };
    RecordingOps::ResetEvents();
    int64_t reclaim_upto = -2;
    Expect(
        SharedRefreshReclaimForTask<RecordingOps>(
            *map, kMapBucketCapacity, kMapBucketCapacity, reclaim_upto
        ),
        kTest, "exact turn computes non-retiring boundary"
    );
    ExpectEqual(reclaim_upto, -1, kTest, "full-window reclaim boundary");
    Expect(
        SharedCheckTaskAppend<RecordingOps>(
            *map, overflowing.data(),
            static_cast<uint32_t>(overflowing.size()), reclaim_upto
        ) == SharedAppendCheck::CapacityBlocked,
        kTest, "preflight rejects task if any target bucket is full"
    );
    for (const Event &event : RecordingOps::events) {
        if (event.kind == EventKind::Exchange || event.kind == EventKind::Flush) {
            Expect(false, kTest, "capacity failure performed a state publication");
            break;
        }
    }
    RecordingOps::DisableEvents();

    std::vector<LogicalTuple> after;
    Expect(SnapshotLogicalMap(*map, after), kTest, "snapshot after overflow");
    Expect(after == before, kTest, "failed preflight preserves all logical entries");
    ExpectEqual(
        LoadControl(&map->committed_tasks.value), kMapBucketCapacity,
        kTest, "failed task not committed"
    );
    ExpectEqual(
        LoadControl(&map->buckets[full_bucket].head.value), full_head,
        kTest, "failed task preserves full head"
    );
    ExpectEqual(
        LoadControl(&map->buckets[full_bucket].tail.value), full_tail,
        kTest, "failed task preserves full tail"
    );
    ExpectEqual(
        LoadControl(&map->buckets[other_bucket].tail.value), other_tail,
        kTest, "earlier preflight entry did not partially append"
    );
    ExpectEqual(
        LoadControl(&map->slots[other_slot_index].seq.value), other_seq,
        kTest, "earlier preflight entry did not publish seq"
    );
}

void TestCapacityBlockedAfterSafeRetire() {
    constexpr const char *kTest = "capacity-after-safe-retire";
    auto map = NewMap();
    RecordingOps::DisableEvents();
    const uint64_t full_address = 0x580000000ULL;
    const uint32_t full_bucket = TensorMapHash(full_address);

    for (uint32_t task = 0; task < kMapBucketCapacity; ++task) {
        const CommitResult result = TryCommitTask(
            *map, static_cast<int32_t>(task),
            {MakeRegion(
                full_address, static_cast<uint64_t>(task) * 16,
                static_cast<uint64_t>(task) * 16 + 8,
                static_cast<int32_t>(task)
            )}
        );
        if (result != CommitResult::Committed) {
            std::fprintf(
                stderr, "[FAIL] shared TensorMap ring/%s: fill failed task=%u\n",
                kTest, task
            );
            ++g_failures;
            return;
        }
    }

    const uint64_t other_address =
        FindAddressOutsideBucket(0x590000000ULL, full_bucket);
    const uint32_t other_bucket = TensorMapHash(other_address);
    const int64_t other_tail =
        LoadControl(&map->buckets[other_bucket].tail.value);
    const uint32_t other_slot_index =
        SharedTensorMapSlotIndex(other_bucket, static_cast<uint64_t>(other_tail));
    const int64_t other_seq =
        LoadControl(&map->slots[other_slot_index].seq.value);
    const std::vector<SharedRegionValue> overflowing = {
        MakeRegion(other_address, 0, 8, kMapBucketCapacity),
        MakeRegion(full_address, 4096, 4104, kMapBucketCapacity),
        MakeRegion(full_address, 4112, 4120, kMapBucketCapacity),
    };

    RecordingOps::ResetEvents();
    int64_t reclaim_upto = -2;
    Expect(
        SharedRefreshReclaimForTask<RecordingOps>(
            *map, kMapBucketCapacity, kMapBucketCapacity - 1,
            reclaim_upto
        ),
        kTest, "exact turn advances reclaim to producer 0"
    );
    ExpectEqual(reclaim_upto, 0, kTest, "inclusive stale boundary");
    ExpectEqual(
        LoadControl(&map->reclaim_upto.value), 0,
        kTest, "published stale boundary"
    );
    Expect(
        SharedCheckTaskAppend<RecordingOps>(
            *map, overflowing.data(),
            static_cast<uint32_t>(overflowing.size()), reclaim_upto
        ) == SharedAppendCheck::CapacityBlocked,
        kTest, "two same-bucket entries still exceed capacity"
    );
    for (const Event &event : RecordingOps::events) {
        if (event.kind == EventKind::Flush) {
            Expect(false, kTest, "capacity failure flushed a payload");
            break;
        }
    }
    RecordingOps::DisableEvents();

    ExpectEqual(
        LoadControl(&map->committed_tasks.value), kMapBucketCapacity,
        kTest, "failed task not committed"
    );
    ExpectEqual(
        LoadControl(&map->buckets[full_bucket].head.value), 1,
        kTest, "only stale producer 0 retired"
    );
    ExpectEqual(
        LoadControl(&map->buckets[full_bucket].tail.value), kMapBucketCapacity,
        kTest, "failed task did not publish full-bucket tail"
    );
    ExpectEqual(
        LoadControl(&map->buckets[other_bucket].tail.value), other_tail,
        kTest, "earlier target bucket did not partially append"
    );
    ExpectEqual(
        LoadControl(&map->slots[other_slot_index].seq.value), other_seq,
        kTest, "earlier target bucket did not publish seq"
    );

    std::vector<LogicalTuple> after;
    Expect(SnapshotLogicalMap(*map, after), kTest, "snapshot after safe retire");
    ExpectEqual(
        static_cast<int64_t>(after.size()),
        static_cast<int64_t>(kMapBucketCapacity - 1),
        kTest, "only one stale logical entry removed"
    );
    const bool published_failed_task = std::any_of(
        after.begin(), after.end(),
        [](const LogicalTuple &entry) {
            return entry.producer ==
                   static_cast<int32_t>(kMapBucketCapacity);
        }
    );
    Expect(
        !published_failed_task, kTest,
        "capacity failure published a current-task entry"
    );
}

void TestDeterministicArrivalAndLogicalTupleDifference() {
    constexpr const char *kTest = "deterministic-arrival-logical-diff";
    auto map = NewMap();
    RecordingOps::DisableEvents();
    const uint64_t address_a = 0x600000000ULL;
    const uint64_t address_b =
        FindAddressOutsideBucket(0x610000000ULL, TensorMapHash(address_a));
    const uint64_t address_c =
        FindAddressOutsideBucket(0x620000000ULL, TensorMapHash(address_b));

    const std::vector<std::vector<SharedRegionValue>> deltas = {
        {},
        {MakeRegion(address_a, 0, 64, 1)},
        {
            MakeRegion(address_b, 0, 32, 2),
            MakeRegion(address_c, 64, 96, 2),
        },
        {},
        {MakeRegion(address_a, 0, 64, 4)},
        {MakeRegion(address_b, 16, 48, 5)},
    };
    constexpr uint32_t kTaskCount = 6;
    constexpr uint32_t kArrivalOrder[kTaskCount] = {4, 2, 5, 3, 1, 0};
    bool committed[kTaskCount] = {};
    uint32_t commit_counts[kTaskCount] = {};
    uint32_t committed_count = 0;
    uint32_t rounds = 0;

    while (committed_count < kTaskCount && rounds < kTaskCount + 1) {
        for (uint32_t actor : kArrivalOrder) {
            if (committed[actor]) {
                continue;
            }
            const int64_t heads_before = [&]() {
                int64_t total = 0;
                for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
                    total += LoadControl(&map->buckets[bucket].head.value);
                }
                return total;
            }();
            const int64_t tails_before = [&]() {
                int64_t total = 0;
                for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
                    total += LoadControl(&map->buckets[bucket].tail.value);
                }
                return total;
            }();
            const int64_t reclaim_before =
                LoadControl(&map->reclaim_upto.value);
            const CommitResult result = TryCommitTask(
                *map, static_cast<int32_t>(actor), deltas[actor]
            );
            if (result == CommitResult::Pending) {
                int64_t heads_after = 0;
                int64_t tails_after = 0;
                for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
                    heads_after +=
                        LoadControl(&map->buckets[bucket].head.value);
                    tails_after +=
                        LoadControl(&map->buckets[bucket].tail.value);
                }
                ExpectEqual(
                    heads_after, heads_before, kTest,
                    "reverse actor preserves all heads"
                );
                ExpectEqual(
                    tails_after, tails_before, kTest,
                    "reverse actor preserves all tails"
                );
                ExpectEqual(
                    LoadControl(&map->reclaim_upto.value),
                    reclaim_before, kTest,
                    "reverse actor preserves reclaim"
                );
                continue;
            }
            if (result == CommitResult::Failed) {
                Expect(false, kTest, "in-turn commit failed");
                return;
            }
            committed[actor] = true;
            ++commit_counts[actor];
            ++committed_count;
            if (deltas[actor].empty()) {
                int64_t tails_after = 0;
                for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
                    tails_after += LoadControl(&map->buckets[bucket].tail.value);
                }
                ExpectEqual(
                    tails_after, tails_before, kTest,
                    "zero-entry task does not append"
                );
            }
        }
        ++rounds;
    }

    ExpectEqual(committed_count, kTaskCount, kTest, "all actors eventually commit");
    ExpectEqual(
        LoadControl(&map->committed_tasks.value), kTaskCount,
        kTest, "final ordered sequencer"
    );
    for (uint32_t task = 0; task < kTaskCount; ++task) {
        if (commit_counts[task] != 1) {
            Expect(false, kTest, "each task committed exactly once");
            break;
        }
    }

    std::vector<LogicalTuple> actual;
    Expect(SnapshotLogicalMap(*map, actual), kTest, "final logical snapshot");
    std::vector<LogicalTuple> expected;
    for (const auto &delta : deltas) {
        for (const SharedRegionValue &entry : delta) {
            expected.push_back(ToTuple(entry));
        }
    }
    std::sort(expected.begin(), expected.end());
    Expect(actual == expected, kTest, "logical tuple vector matches fixed reference");
}

}  // namespace

int main() {
    TestAbiResetAndZeroEntryCommit();
    TestPublicationOrderAndDoubleSeqCheck();
    TestVersionsWindowAndMultipleBuckets();
    TestOrderedReclaimFormulaAndExactTurn();
    TestAbsoluteSeqMultipleLapsAndAba();
    TestCapacityFailureIsAllOrNothing();
    TestCapacityBlockedAfterSafeRetire();
    TestDeterministicArrivalAndLogicalTupleDifference();

    if (g_failures != 0) {
        std::fprintf(stderr, "[FAIL] shared TensorMap ring: %d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf(
        "[PASS] shared TensorMap ring: ABI, ordered commit, window, reclaim, "
        "absolute seq, ABA, overflow, logical differential\n"
    );
    std::printf(
        "[NOTE] CPU validates atomic/order hooks only; it does not simulate A5 DCache or DCCI.\n"
    );
    return EXIT_SUCCESS;
}
