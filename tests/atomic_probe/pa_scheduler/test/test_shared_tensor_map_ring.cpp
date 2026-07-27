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
using pa_scheduler::SharedAdvanceReaderDone;
using pa_scheduler::SharedBucketState;
using pa_scheduler::SharedCheckTaskAppend;
using pa_scheduler::SharedComputeReaderReclaimCandidate;
using pa_scheduler::SharedComputeOrderedReclaimCandidate;
using pa_scheduler::SharedHasExactTaskTurn;
using pa_scheduler::SharedLookupRegion;
using pa_scheduler::SharedPreflightTaskAppend;
using pa_scheduler::SharedPublishReclaimCandidate;
using pa_scheduler::SharedPublishTaskCommit;
using pa_scheduler::SharedReadRegionSlot;
using pa_scheduler::SharedRefreshReaderReclaimForTask;
using pa_scheduler::SharedRefreshReclaimForTask;
using pa_scheduler::SharedRegionPayload;
using pa_scheduler::SharedRegionSlot;
using pa_scheduler::SharedRegionValue;
using pa_scheduler::SharedRetireBucket;
using pa_scheduler::SharedTensorMapSidecar;
using pa_scheduler::SharedTensorMapSlotIndex;
using pa_scheduler::SharedTryAppendReaderGatedTask;
using pa_scheduler::TensorMapHash;
using pa_scheduler::kMapBucketCapacity;
using pa_scheduler::kMapBuckets;
using pa_scheduler::kMapCapacity;
using pa_scheduler::kMaxTaskTensors;
using pa_scheduler::kSharedMapEmptySeq;

static_assert(PTO_FDWIC_SHARED_MAP == 1, "this test must compile as the shared TensorMap mode");
static_assert(sizeof(SharedRegionValue) == 32, "shared logical value ABI changed");
static_assert(sizeof(SharedRegionPayload) == 64, "shared payload must occupy one cache line");
static_assert(alignof(SharedRegionPayload) == 64, "shared payload alignment changed");
static_assert(sizeof(SharedRegionSlot) == 128, "shared slot must occupy two cache lines");
static_assert(offsetof(SharedRegionSlot, seq) == 64, "shared seq cache line offset changed");
static_assert(sizeof(SharedBucketState) == 128, "shared bucket control ABI changed");
static_assert(offsetof(SharedBucketState, tail) == 64, "shared head/tail cache lines merged");
#if PTO_FDWIC_TENSORMAP_RING_CAP == 128
static_assert(sizeof(SharedTensorMapSidecar) == 12426880, "shared sidecar ABI changed");
#endif
static_assert(alignof(SharedTensorMapSidecar) == 64, "shared sidecar alignment changed");
static_assert(offsetof(SharedTensorMapSidecar, buckets) == 128, "shared bucket offset changed");
static_assert(
    offsetof(SharedTensorMapSidecar, slots) ==
        offsetof(SharedTensorMapSidecar, buckets) +
        sizeof(SharedBucketState) * kMapBuckets,
    "shared slots must immediately follow the active bucket controls"
);
static_assert(
    offsetof(SharedTensorMapSidecar, shared_outputs) ==
        offsetof(SharedTensorMapSidecar, slots) +
        sizeof(SharedRegionSlot) * kMapCapacity,
    "shared output table must immediately follow the fixed 16K slot pool"
);
static_assert(
    offsetof(SharedTensorMapSidecar, reader_done) ==
        offsetof(SharedTensorMapSidecar, writer_history) +
            sizeof(pa_scheduler::SharedWriterHistoryCell) *
                pa_scheduler::kMaxTasks,
    "shared reader progress must immediately follow writer history"
);
#if PTO_FDWIC_TENSORMAP_RING_CAP == 128
static_assert(offsetof(SharedTensorMapSidecar, slots) == 16512, "default shared slot offset changed");
static_assert(offsetof(SharedTensorMapSidecar, shared_outputs) == 2113664, "default shared output offset changed");
static_assert(offsetof(SharedTensorMapSidecar, shared_heap_cursor) == 11026560, "default shared heap offset changed");
static_assert(
    offsetof(SharedTensorMapSidecar, shared_heap_vend) == 11027072,
    "default shared heap vend offset changed"
);
static_assert(
    offsetof(SharedTensorMapSidecar, shared_vector_cursor) == 11027136,
    "default shared Vector cursor offset changed"
);
static_assert(
    offsetof(SharedTensorMapSidecar, writer_history) == 11027648,
    "default shared writer-history offset changed"
);
static_assert(
    offsetof(SharedTensorMapSidecar, reader_done) == 12420288,
    "default shared reader-progress offset changed"
);
#endif

enum class EventKind : uint8_t {
    Load,
    Exchange,
    CompareExchange,
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
    using InvalidateHook = void (*)(void *);
    using LoadHook = void (*)(void *);

    static std::vector<Event> events;
    static bool record;
    static const void *mutate_payload;
    static volatile int64_t *mutate_seq;
    static int64_t mutate_seq_value;
    static bool mutate_once;
    static const void *invalidate_hook_address;
    static InvalidateHook invalidate_hook;
    static void *invalidate_hook_context;
    static const void *load_hook_address;
    static LoadHook load_hook;
    static void *load_hook_context;

    static int64_t Load(volatile int64_t *address) {
        // load hook 在真正取值前执行，用来固定“reader 已读旧 head、尚未
        // 读 tail”这一控制快照交错。先清空再回调，避免 writer 内部读取
        // 同一 tail 时递归。
        if (load_hook != nullptr &&
            ConstAddress(address) == load_hook_address) {
            const LoadHook callback = load_hook;
            void *const context = load_hook_context;
            load_hook_address = nullptr;
            load_hook = nullptr;
            load_hook_context = nullptr;
            callback(context);
        }
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

    static int64_t CompareExchange(
        volatile int64_t *address, int64_t expected, int64_t desired
    ) {
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(
            address, &observed, desired, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        if (record) {
            events.push_back(
                {
                    EventKind::CompareExchange,
                    ConstAddress(address), desired, observed,
                }
            );
        }
        return observed;
    }

    static void InvalidateRegion(const void *address, uint64_t bytes) {
        if (record) {
            events.push_back(
                {EventKind::Invalidate, address, static_cast<int64_t>(bytes), 0}
            );
        }
        // 单次 hook 在 reader 第一次 seq 检查之后、拷贝 payload 之前执行。
        // 先清空 hook 再回调，允许回调里的 preflight 嵌套读取同一槽而不递归。
        if (invalidate_hook != nullptr &&
            address == invalidate_hook_address) {
            const InvalidateHook callback = invalidate_hook;
            void *const context = invalidate_hook_context;
            invalidate_hook_address = nullptr;
            invalidate_hook = nullptr;
            invalidate_hook_context = nullptr;
            callback(context);
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
        invalidate_hook_address = nullptr;
        invalidate_hook = nullptr;
        invalidate_hook_context = nullptr;
        load_hook_address = nullptr;
        load_hook = nullptr;
        load_hook_context = nullptr;
    }

    static void DisableEvents() {
        events.clear();
        record = false;
        mutate_payload = nullptr;
        mutate_seq = nullptr;
        mutate_seq_value = 0;
        mutate_once = false;
        invalidate_hook_address = nullptr;
        invalidate_hook = nullptr;
        invalidate_hook_context = nullptr;
        load_hook_address = nullptr;
        load_hook = nullptr;
        load_hook_context = nullptr;
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
const void *RecordingOps::invalidate_hook_address = nullptr;
RecordingOps::InvalidateHook RecordingOps::invalidate_hook = nullptr;
void *RecordingOps::invalidate_hook_context = nullptr;
const void *RecordingOps::load_hook_address = nullptr;
RecordingOps::LoadHook RecordingOps::load_hook = nullptr;
void *RecordingOps::load_hook_context = nullptr;

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
    pa_scheduler::InitializeSharedInsertTurns(map);
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
        map.writer_history[task].magic = 0;
        map.writer_history[task].writer_task = 0;
        map.writer_history[task].count = 0;
        map.writer_history[task].reserved = 0;
    }
    for (uint32_t worker = 0;
         worker < pa_scheduler::kWorkers; ++worker) {
        StoreControl(&map.reader_done[worker].value, -1);
    }
}

std::unique_ptr<SharedTensorMapSidecar> NewMap() {
    // sidecar 超过 2 MiB，必须 heap 分配，不能依赖 host 线程栈容量。
    auto map = std::unique_ptr<SharedTensorMapSidecar>(new SharedTensorMapSidecar);
    ResetSharedTensorMap(*map);
    RecordingOps::ResetEvents();
    return map;
}

void TestPhysicalSlotBoundaries() {
    constexpr const char *kTest = "physical-slot-boundaries";
    ExpectEqual(
        SharedTensorMapSlotIndex(0, 0), 0,
        kTest, "first physical slot"
    );
    ExpectEqual(
        SharedTensorMapSlotIndex(
            kMapBuckets - 1U,
            static_cast<uint64_t>(kMapBucketCapacity - 1U)
        ),
        kMapCapacity - 1U,
        kTest, "last physical slot"
    );
    ExpectEqual(
        SharedTensorMapSlotIndex(
            kMapBuckets - 1U,
            static_cast<uint64_t>(kMapBucketCapacity)
        ),
        (kMapBuckets - 1U) * kMapBucketCapacity,
        kTest, "last bucket first wrapped slot"
    );
}

SharedRegionValue MakeRegion(
    uint64_t buffer_addr, uint64_t lo, uint64_t hi, int32_t producer
) {
    return {buffer_addr, lo, hi, producer, 0};
}

uint64_t FindAddressOutsideBucket(uint64_t seed, uint32_t excluded_bucket) {
    if constexpr (kMapBuckets == 1) {
        // 单桶变体不存在“另一个桶”，但调用方仍需要不同的 region identity。
        // 返回 seed 让语义测试继续覆盖同桶多地址；跨桶隔离由专门测试按
        // kMapBuckets>1 条件执行。
        return seed;
    }
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

struct SlowReaderReuseAttempt {
    SharedTensorMapSidecar *map;
    const SharedRegionValue *replacements;
    uint32_t replacement_count;
    uint32_t bucket;
    uint32_t active_readers;
    int32_t heap_window;
    bool fired;
    int64_t reclaim_upto;
    SharedAppendCheck append_result;
    int64_t head;
    int64_t tail;
    int64_t committed_tasks;
    int64_t slot_seq;
    int32_t slot_producer;
};

void AttemptReuseWhileReaderPaused(void *opaque) {
    auto &attempt =
        *static_cast<SlowReaderReuseAttempt *>(opaque);
    attempt.fired = true;
    attempt.append_result =
        SharedTryAppendReaderGatedTask<RecordingOps>(
            *attempt.map, attempt.replacements,
            attempt.replacement_count,
            attempt.active_readers, attempt.heap_window
        );
    attempt.reclaim_upto =
        LoadControl(&attempt.map->reclaim_upto.value);
    attempt.head =
        LoadControl(
            &attempt.map->buckets[attempt.bucket].head.value
        );
    attempt.tail =
        LoadControl(
            &attempt.map->buckets[attempt.bucket].tail.value
        );
    attempt.committed_tasks =
        LoadControl(&attempt.map->committed_tasks.value);
    const uint32_t slot_index =
        SharedTensorMapSlotIndex(attempt.bucket, 0);
    attempt.slot_seq =
        LoadControl(&attempt.map->slots[slot_index].seq.value);
    attempt.slot_producer =
        attempt.map->slots[slot_index].payload.value.producer;
}

struct ConcurrentSafeRetireAttempt {
    SharedTensorMapSidecar *map;
    const SharedRegionValue *replacements;
    uint32_t replacement_count;
    uint32_t active_readers;
    int32_t heap_window;
    bool fired;
    bool candidate_ok;
    bool publish_ok;
    bool append_ok;
    int64_t candidate;
    int64_t reclaim_upto;
    SharedAppendCheck append_check;
};

void RetireSafePrefixWhileReaderPaused(void *opaque) {
    auto &attempt =
        *static_cast<ConcurrentSafeRetireAttempt *>(opaque);
    attempt.fired = true;
    attempt.candidate = -2;
    attempt.reclaim_upto = -2;
    attempt.candidate_ok =
        SharedComputeReaderReclaimCandidate<RecordingOps>(
            *attempt.map, attempt.active_readers,
            attempt.heap_window, attempt.candidate
        );
    attempt.publish_ok =
        attempt.candidate_ok &&
        SharedPublishReclaimCandidate<RecordingOps>(
            *attempt.map, attempt.candidate,
            attempt.reclaim_upto
        );
    attempt.append_check =
        attempt.publish_ok
            ? SharedCheckTaskAppend<RecordingOps>(
                  *attempt.map, attempt.replacements,
                  attempt.replacement_count,
                  attempt.reclaim_upto
              )
            : SharedAppendCheck::ProtocolError;
    attempt.append_ok =
        attempt.append_check == SharedAppendCheck::Ready &&
        SharedAppendPreparedTask<RecordingOps>(
            *attempt.map, attempt.replacements,
            attempt.replacement_count
        );
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
    for (uint32_t worker = 0;
         worker < pa_scheduler::kWorkers; ++worker) {
        if (LoadControl(&map->reader_done[worker].value) != -1) {
            Expect(false, kTest, "reader progress reset");
            break;
        }
    }
    for (uint32_t lane = 1;
         lane < pa_scheduler::kSharedInsertTurnCapacity;
         ++lane) {
        if (LoadControl(
                &map->insert_turn_extra[lane - 1U].value
            ) != -1) {
            Expect(false, kTest, "inactive insert turn reset");
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
    RecordingOps::ResetEvents();
    Expect(
        !SharedPublishTaskCommit<RecordingOps>(*map, 1),
        kTest, "repeated commit is rejected"
    );
    ExpectEqual(
        LoadControl(&map->committed_tasks.value), 2, kTest,
        "repeated commit preserves the advanced frontier"
    );
    Expect(
        RecordingOps::events.size() == 1 &&
            RecordingOps::events[0].kind ==
                EventKind::CompareExchange &&
            RecordingOps::events[0].address ==
                &map->committed_tasks.value &&
            RecordingOps::events[0].argument == 2 &&
            RecordingOps::events[0].result == 2,
        kTest,
        "repeated commit performs one non-mutating CAS"
    );
    StoreControl(&map->committed_tasks.value, 0);
    RecordingOps::ResetEvents();
    Expect(
        !SharedPublishTaskCommit<RecordingOps>(*map, 1),
        kTest, "future commit is rejected"
    );
    ExpectEqual(
        LoadControl(&map->committed_tasks.value), 0, kTest,
        "future commit preserves the earlier frontier"
    );
    Expect(
        RecordingOps::events.size() == 1 &&
            RecordingOps::events[0].kind ==
                EventKind::CompareExchange &&
            RecordingOps::events[0].address ==
                &map->committed_tasks.value &&
            RecordingOps::events[0].argument == 2 &&
            RecordingOps::events[0].result == 0,
        kTest,
        "future commit performs one non-mutating CAS"
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
    uint32_t expected_versioned_bucket_entries = 0;
    for (const auto *task : {&task0, &task1, &task2}) {
        for (const SharedRegionValue &entry : *task) {
            if (TensorMapHash(entry.buffer_addr) == versioned_bucket) {
                ++expected_versioned_bucket_entries;
            }
        }
    }
    ExpectEqual(
        LoadControl(&map->buckets[versioned_bucket].tail.value),
        expected_versioned_bucket_entries,
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

    // 空环正常写入只需既有 reclaim=-1，不应为了未来可能发生的容量反压
    // 无条件扫描全部 reader_done。事件记录锁定 fast path 没有 reader load。
    auto fast_map = NewMap();
    bool fast_readers_closed = true;
    for (uint32_t worker = 0;
         worker < pa_scheduler::kWorkers; ++worker) {
        fast_readers_closed &=
            SharedAdvanceReaderDone<RecordingOps>(
                *fast_map, worker, 0
            );
    }
    Expect(
        fast_readers_closed, kTest,
        "all fast-path actors close task 0 before append"
    );
    RecordingOps::ResetEvents();
    Expect(
        SharedTryAppendReaderGatedTask<RecordingOps>(
            *fast_map, nullptr, 0,
            pa_scheduler::kWorkers, 2
        ) == SharedAppendCheck::Ready &&
            RecordingOps::events.empty(),
        kTest, "empty ordinary batch performs no shared access"
    );
    const SharedRegionValue fast_entry =
        MakeRegion(0x4F0000000ULL, 0, 8, 0);
    RecordingOps::ResetEvents();
    Expect(
        SharedTryAppendReaderGatedTask<RecordingOps>(
            *fast_map, &fast_entry, 1,
            pa_scheduler::kWorkers, 2
        ) == SharedAppendCheck::Ready,
        kTest, "fast path appends without refreshing reader frontier"
    );
    bool loaded_reader_progress = false;
    for (const Event &event : RecordingOps::events) {
        if (event.kind != EventKind::Load) {
            continue;
        }
        for (uint32_t worker = 0;
             worker < pa_scheduler::kWorkers; ++worker) {
            const void *const reader_address =
                const_cast<const int64_t *>(
                    &fast_map->reader_done[worker].value
                );
            loaded_reader_progress |=
                event.address == reader_address;
        }
    }
    Expect(
        !loaded_reader_progress, kTest,
        "ready fast path performs zero reader-progress loads"
    );
    ExpectEqual(
        LoadControl(&fast_map->committed_tasks.value),
        0, kTest,
        "fast reader-gated append does not use global exact turn"
    );
    RecordingOps::DisableEvents();

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
    // fill driver 只负责构造满桶，随后清掉它的旧 exact turn。active
    // reader 尚未关闭任何 task，H=0 也只能得到 reclaim=-1。
    StoreControl(&map->committed_tasks.value, 0);
    RecordingOps::ResetEvents();
    Expect(
        SharedTryAppendReaderGatedTask<RecordingOps>(
            *map, overflowing.data(),
            static_cast<uint32_t>(overflowing.size()), 1, 0
        ) == SharedAppendCheck::CapacityBlocked,
        kTest,
        "reader-gated batch rejects task if any target bucket is full"
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
        LoadControl(&map->committed_tasks.value), 0,
        kTest, "capacity failure does not use global exact turn"
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

    // 另建生产可达的有序写入历史：task 0 只写一条，后续 task 每批最多
    // kMaxTaskTensors 条直至填满。下一 writer 的 reader 前沿只允许回收
    // producer 0，恰好空出一个槽；同桶两项仍必须整批 CapacityBlocked。
    constexpr uint32_t kOneSlotFillTasks =
        1U +
        (kMapBucketCapacity - 1U + kMaxTaskTensors - 1U) /
            kMaxTaskTensors;
    static_assert(
        kOneSlotFillTasks < pa_scheduler::kMaxTasks,
        "one-slot ordered history exceeds task domain"
    );
    auto one_slot_map = NewMap();
    const uint64_t one_slot_address = 0x50A000000ULL;
    const uint32_t one_slot_bucket =
        TensorMapHash(one_slot_address);
    uint32_t one_slot_cursor = 0;
    int32_t one_slot_task = 0;
    while (one_slot_cursor < kMapBucketCapacity) {
        const uint32_t remaining =
            kMapBucketCapacity - one_slot_cursor;
        const uint32_t batch_count =
            one_slot_task == 0
                ? 1U
                : (remaining < kMaxTaskTensors
                       ? remaining
                       : kMaxTaskTensors);
        std::vector<SharedRegionValue> batch;
        batch.reserve(batch_count);
        for (uint32_t index = 0;
             index < batch_count; ++index) {
            const uint64_t lo =
                static_cast<uint64_t>(
                    one_slot_cursor + index
                ) *
                16U;
            batch.push_back(MakeRegion(
                one_slot_address, lo, lo + 8U,
                one_slot_task
            ));
        }
        if (TryCommitTask(
                *one_slot_map, one_slot_task, batch
            ) != CommitResult::Committed) {
            std::fprintf(
                stderr,
                "[FAIL] shared TensorMap ring/%s: "
                "one-slot fill failed task=%d\n",
                kTest, one_slot_task
            );
            ++g_failures;
            return;
        }
        one_slot_cursor += batch_count;
        ++one_slot_task;
    }
    ExpectEqual(
        one_slot_task, kOneSlotFillTasks, kTest,
        "one-slot history uses the expected task count"
    );
    StoreControl(&one_slot_map->committed_tasks.value, 0);
    bool one_slot_reader_closed = true;
    for (int32_t task = 0;
         task <= one_slot_task; ++task) {
        one_slot_reader_closed &=
            SharedAdvanceReaderDone<RecordingOps>(
                *one_slot_map, 0, task
            );
    }
    Expect(
        one_slot_reader_closed, kTest,
        "single reader closes the current ordered writer task"
    );
    const uint64_t next_lo =
        static_cast<uint64_t>(kMapBucketCapacity) * 16U;
    const std::vector<SharedRegionValue> two_for_one_slot = {
        MakeRegion(
            one_slot_address, next_lo, next_lo + 8U,
            one_slot_task
        ),
        MakeRegion(
            one_slot_address, next_lo + 16U,
            next_lo + 24U, one_slot_task
        ),
    };
    RecordingOps::ResetEvents();
    Expect(
        SharedTryAppendReaderGatedTask<RecordingOps>(
            *one_slot_map, two_for_one_slot.data(),
            static_cast<uint32_t>(two_for_one_slot.size()),
            1, one_slot_task
        ) == SharedAppendCheck::CapacityBlocked,
        kTest,
        "one free slot cannot partially accept a two-entry batch"
    );
    bool flushed_partial_entry = false;
    for (const Event &event : RecordingOps::events) {
        flushed_partial_entry |= event.kind == EventKind::Flush;
    }
    Expect(
        !flushed_partial_entry, kTest,
        "one-slot retry publishes no batch payload"
    );
    RecordingOps::DisableEvents();
    ExpectEqual(
        LoadControl(&one_slot_map->reclaim_upto.value),
        0, kTest,
        "closed reader exposes exactly producer 0"
    );
    ExpectEqual(
        LoadControl(
            &one_slot_map->buckets[one_slot_bucket].head.value
        ),
        1, kTest,
        "one safe producer leaves exactly one free slot"
    );
    ExpectEqual(
        LoadControl(
            &one_slot_map->buckets[one_slot_bucket].tail.value
        ),
        kMapBucketCapacity, kTest,
        "blocked two-entry batch preserves tail"
    );
    SharedRegionSlot &first_reusable_slot =
        one_slot_map->slots[
            SharedTensorMapSlotIndex(
                one_slot_bucket, kMapBucketCapacity
            )
        ];
    ExpectEqual(
        LoadControl(&first_reusable_slot.seq.value),
        0, kTest,
        "blocked two-entry batch does not claim the one free slot"
    );
    ExpectEqual(
        LoadControl(&one_slot_map->committed_tasks.value),
        0, kTest,
        "one-slot retry remains independent of exact turn"
    );

    // 容量反压之外，再锁定后项协议损坏同样不能让前项先发布。不同桶形态
    // 使用第二桶 cursor 0；单桶 CAP=16384 形态使用同桶 cursor 1。
    auto protocol_map = NewMap();
    const uint64_t first_address = 0x511000000ULL;
    const uint32_t first_bucket = TensorMapHash(first_address);
    const uint64_t second_address =
        FindAddressOutsideBucket(0x512000000ULL, first_bucket);
    const uint32_t second_bucket = TensorMapHash(second_address);
    const uint64_t second_cursor =
        second_bucket == first_bucket ? 1U : 0U;
    SharedRegionSlot &bad_slot =
        protocol_map->slots[
            SharedTensorMapSlotIndex(
                second_bucket, second_cursor
            )
        ];
    StoreControl(&bad_slot.seq.value, 7);
    const std::vector<SharedRegionValue> invalid_batch = {
        MakeRegion(first_address, 0, 8, 0),
        MakeRegion(second_address, 16, 24, 0),
    };
    RecordingOps::ResetEvents();
    Expect(
        SharedTryAppendReaderGatedTask<RecordingOps>(
            *protocol_map, invalid_batch.data(),
            static_cast<uint32_t>(invalid_batch.size()), 1, 0
        ) == SharedAppendCheck::ProtocolError,
        kTest,
        "later slot corruption rejects the whole reader-gated batch"
    );
    for (const Event &event : RecordingOps::events) {
        if (event.kind == EventKind::Exchange ||
            event.kind == EventKind::Flush ||
            event.kind == EventKind::CompareExchange) {
            Expect(
                false, kTest,
                "protocol rejection published an earlier batch entry"
            );
            break;
        }
    }
    RecordingOps::DisableEvents();
    ExpectEqual(
        LoadControl(
            &protocol_map->buckets[first_bucket].tail.value
        ),
        0, kTest,
        "later protocol error preserves the earlier bucket tail"
    );
    SharedRegionSlot &first_protocol_slot =
        protocol_map->slots[
            SharedTensorMapSlotIndex(first_bucket, 0)
        ];
    ExpectEqual(
        LoadControl(&first_protocol_slot.seq.value),
        kSharedMapEmptySeq, kTest,
        "later protocol error preserves the earlier slot seq"
    );
    ExpectEqual(
        LoadControl(&protocol_map->committed_tasks.value),
        0, kTest,
        "protocol rejection does not use global exact turn"
    );
}

void TestSlowReaderGatesFullBucketReuse() {
    constexpr const char *kTest =
        "slow-reader-gates-full-bucket-reuse";
    constexpr int32_t kHeapWindow = 2;
    constexpr uint32_t kActiveReaders = 2;
    constexpr uint32_t kEntriesPerTask = 8;
    static_assert(
        kEntriesPerTask <= kMaxTaskTensors,
        "reader interleave batch exceeds task tensor capacity"
    );
    static_assert(
        kMapBucketCapacity % kEntriesPerTask == 0,
        "full-ring interleave requires whole task batches"
    );
    constexpr uint32_t kFillTasks =
        kMapBucketCapacity / kEntriesPerTask;
    static_assert(
        kFillTasks > static_cast<uint32_t>(kHeapWindow),
        "fast reader must reach the reclaim boundary"
    );
    static_assert(
        kFillTasks < pa_scheduler::kMaxTasks,
        "full-ring interleave must stay inside reader task domain"
    );

    auto map = NewMap();
    RecordingOps::DisableEvents();
    const uint64_t address = 0x520000000ULL;
    const uint32_t bucket = TensorMapHash(address);

    // 用每 task 八条合法 region 填满同一桶。即使 CAP=16384，future task
    // 也只有 2048，仍处于 reader_done 的 kMaxTasks 合同内。
    for (uint32_t task = 0; task < kFillTasks; ++task) {
        std::vector<SharedRegionValue> entries;
        entries.reserve(kEntriesPerTask);
        for (uint32_t index = 0;
             index < kEntriesPerTask; ++index) {
            const uint64_t cursor =
                static_cast<uint64_t>(task) *
                    kEntriesPerTask +
                index;
            const uint64_t lo = cursor * 32U;
            entries.push_back(MakeRegion(
                address, lo, lo + 8U,
                static_cast<int32_t>(task)
            ));
        }
        if (TryCommitTask(
                *map, static_cast<int32_t>(task),
                entries
            ) != CommitResult::Committed) {
            std::fprintf(
                stderr,
                "[FAIL] shared TensorMap ring/%s: "
                "fill failed task=%u\n",
                kTest, task
            );
            ++g_failures;
            return;
        }
    }

    const int32_t writer_task =
        static_cast<int32_t>(kFillTasks);
    ExpectEqual(
        LoadControl(&map->committed_tasks.value),
        writer_task, kTest, "fill driver reaches the future writer task"
    );
    StoreControl(&map->committed_tasks.value, 0);
    ExpectEqual(
        LoadControl(&map->committed_tasks.value),
        0, kTest, "reader-gated writer has no global exact turn"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].head.value),
        0, kTest, "full ring starts at cursor zero"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].tail.value),
        kMapBucketCapacity, kTest, "target bucket is exactly full"
    );

    // worker 0 已关闭 task 0/1，但仍在 task 2 的 ordinary lookup；
    // worker 1 作为快 reader 已追到 future writer 前一 task。
    bool reader_progress_ok = true;
    for (int32_t task = 0; task <= 1; ++task) {
        reader_progress_ok &=
            SharedAdvanceReaderDone<RecordingOps>(
                *map, 0, task
            );
    }
    for (int32_t task = 0; task <= writer_task; ++task) {
        reader_progress_ok &=
            SharedAdvanceReaderDone<RecordingOps>(
                *map, 1, task
            );
    }
    Expect(
        reader_progress_ok, kTest,
        "slow/fast readers publish contiguous progress"
    );
    ExpectEqual(
        LoadControl(&map->reader_done[0].value),
        1, kTest, "slow reader remains inside task 2"
    );
    ExpectEqual(
        LoadControl(&map->reader_done[1].value),
        writer_task, kTest,
        "writer closes its own task before reader-gated append"
    );

    const uint64_t independent_address =
        FindAddressOutsideBucket(0x521000000ULL, bucket);
    const uint32_t independent_bucket =
        TensorMapHash(independent_address);
    const int64_t independent_tail_before =
        LoadControl(
            &map->buckets[independent_bucket].tail.value
        );
    const uint32_t independent_slot_index =
        SharedTensorMapSlotIndex(
            independent_bucket,
            static_cast<uint64_t>(independent_tail_before)
        );
    const int64_t independent_seq_before =
        LoadControl(
            &map->slots[independent_slot_index].seq.value
        );

    std::vector<SharedRegionValue> replacements;
    replacements.reserve(kEntriesPerTask + 1U);
    if constexpr (kMapBuckets > 1) {
        // 独立空桶 entry 故意排在满桶 batch 前。若组合 helper 错误地
        // 边检查边 append，它会在后项 CapacityBlocked 前留下部分发布。
        replacements.push_back(MakeRegion(
            independent_address, 0, 8, writer_task
        ));
    }
    for (uint32_t index = 0;
         index < kEntriesPerTask; ++index) {
        const uint64_t lo =
            (static_cast<uint64_t>(kMapBucketCapacity) +
             index) *
            32U;
        replacements.push_back(MakeRegion(
            address, lo, lo + 8U, writer_task
        ));
    }

    SharedRegionSlot &first_slot =
        map->slots[SharedTensorMapSlotIndex(bucket, 0)];
    SlowReaderReuseAttempt attempt{};
    attempt.map = map.get();
    attempt.replacements = replacements.data();
    attempt.replacement_count =
        static_cast<uint32_t>(replacements.size());
    attempt.bucket = bucket;
    attempt.active_readers = kActiveReaders;
    attempt.heap_window = kHeapWindow;
    attempt.reclaim_upto = -2;
    attempt.append_result = SharedAppendCheck::ProtocolError;
    attempt.head = -2;
    attempt.tail = -2;
    attempt.committed_tasks = -2;
    attempt.slot_seq = -2;
    attempt.slot_producer = -2;

    // 把 future writer 的 refresh/preflight 精确插在 slow reader 首次
    // seq 检查之后、拷贝 cursor-0 payload 之前。
    RecordingOps::ResetEvents();
    RecordingOps::invalidate_hook_address =
        &first_slot.payload;
    RecordingOps::invalidate_hook =
        AttemptReuseWhileReaderPaused;
    RecordingOps::invalidate_hook_context = &attempt;
    bool protocol_ok = false;
    const int32_t producer =
        SharedLookupRegion<RecordingOps>(
            *map, MakeRegion(address, 0, 8, -1),
            2, kHeapWindow, protocol_ok
        );

    Expect(
        protocol_ok && producer == 0, kTest,
        "paused task-2 lookup still consumes producer 0"
    );
    Expect(attempt.fired, kTest, "paused-reader hook fired");
    Expect(
        attempt.reclaim_upto == -1,
        kTest,
        "slow reader keeps the global reclaim frontier at -1"
    );
    Expect(
        attempt.append_result ==
            SharedAppendCheck::CapacityBlocked,
        kTest,
        "reader-gated batch remains retryable while task 2 is open"
    );
    ExpectEqual(
        attempt.head, 0, kTest,
        "blocked reuse preserves bucket head"
    );
    ExpectEqual(
        attempt.tail, kMapBucketCapacity, kTest,
        "blocked reuse preserves bucket tail"
    );
    ExpectEqual(
        attempt.committed_tasks, 0, kTest,
        "blocked reuse remains independent of global exact turn"
    );
    ExpectEqual(
        attempt.slot_seq, 0, kTest,
        "blocked reuse preserves cursor-0 absolute seq"
    );
    ExpectEqual(
        attempt.slot_producer, 0, kTest,
        "blocked reuse preserves cursor-0 payload"
    );
    if constexpr (kMapBuckets > 1) {
        ExpectEqual(
            LoadControl(
                &map->buckets[independent_bucket].tail.value
            ),
            independent_tail_before, kTest,
            "blocked later bucket does not append the earlier entry"
        );
        ExpectEqual(
            LoadControl(
                &map->slots[independent_slot_index].seq.value
            ),
            independent_seq_before, kTest,
            "blocked later bucket does not publish earlier seq"
        );
    }
    for (const Event &event : RecordingOps::events) {
        if (event.kind == EventKind::Exchange ||
            event.kind == EventKind::Flush ||
            event.kind == EventKind::CompareExchange) {
            Expect(
                false, kTest,
                "paused-reader attempt published shared state"
            );
            break;
        }
    }
    RecordingOps::DisableEvents();

    // lookup 已完整返回后才允许 slow reader 关闭 task 2。此时 Dmin=2，
    // H=2 的 candidate 首次变成 0，producer 0 才可回收。
    Expect(
        SharedAdvanceReaderDone<RecordingOps>(
            *map, 0, 2
        ),
        kTest, "slow reader closes task 2 after its final read"
    );
    ExpectEqual(
        LoadControl(&map->reader_done[0].value),
        2, kTest, "slow reader publishes its closed frontier"
    );
    ExpectEqual(
        LoadControl(&map->reader_done[1].value),
        writer_task, kTest,
        "fast reader frontier remains unchanged"
    );
    Expect(
        SharedTryAppendReaderGatedTask<RecordingOps>(
            *map, replacements.data(),
            static_cast<uint32_t>(replacements.size()),
            kActiveReaders, kHeapWindow
        ) == SharedAppendCheck::Ready,
        kTest,
        "closed reader makes the same whole batch appendable"
    );
    ExpectEqual(
        LoadControl(&map->reclaim_upto.value),
        0, kTest, "closed readers publish producer-0 reclaim"
    );

    ExpectEqual(
        LoadControl(&map->buckets[bucket].head.value),
        kEntriesPerTask, kTest,
        "all task-0 entries retire after reader close"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].tail.value),
        kMapBucketCapacity + kEntriesPerTask,
        kTest, "replacement batch advances tail"
    );
    ExpectEqual(
        LoadControl(&map->committed_tasks.value),
        0, kTest,
        "successful reader-gated append does not publish exact turn"
    );
    if constexpr (kMapBuckets > 1) {
        ExpectEqual(
            LoadControl(
                &map->buckets[independent_bucket].tail.value
            ),
            independent_tail_before + 1, kTest,
            "successful batch appends the independent first entry"
        );
    }
    ExpectEqual(
        LoadControl(&first_slot.seq.value),
        kMapBucketCapacity, kTest,
        "cursor CAP reuses physical slot 0 with a new seq"
    );
    ExpectEqual(
        first_slot.payload.value.producer,
        writer_task, kTest,
        "reused slot contains only the future producer"
    );

    SharedRegionValue old_cursor{};
    Expect(
        !SharedReadRegionSlot<RecordingOps>(
            *map, bucket, 0, old_cursor
        ),
        kTest, "old cursor 0 is rejected after reuse"
    );
    SharedRegionValue new_cursor{};
    Expect(
        SharedReadRegionSlot<RecordingOps>(
            *map, bucket, kMapBucketCapacity,
            new_cursor
        ) &&
            new_cursor.producer == writer_task,
        kTest, "new cursor CAP resolves the replacement"
    );
    std::vector<LogicalTuple> snapshot;
    Expect(
        SnapshotLogicalMap(*map, snapshot), kTest,
        "logical snapshot after reader-gated reuse"
    );
    ExpectEqual(
        snapshot.size(),
        kMapBucketCapacity +
            (kMapBuckets > 1 ? 1U : 0U),
        kTest,
        "whole batch keeps the target full and includes the independent entry"
    );
    const int32_t old_region =
        SharedLookupRegion<RecordingOps>(
            *map, MakeRegion(address, 0, 8, -1),
            3, kHeapWindow, protocol_ok
        );
    Expect(
        protocol_ok && old_region == -1, kTest,
        "task 3 no longer observes expired producer 0"
    );
}

void TestLookupSkipsConcurrentlyRetiredSafePrefix() {
    constexpr const char *kTest =
        "lookup-skips-concurrently-retired-safe-prefix";
    constexpr int32_t kReaderTask = 2;
    constexpr int32_t kHeapWindow = 1;
    constexpr uint32_t kEntriesPerTask = 8;
    constexpr uint32_t kActiveReaders = 1;
    static_assert(
        kMapBucketCapacity % kEntriesPerTask == 0,
        "safe-prefix interleave requires whole task batches"
    );
    constexpr uint32_t kFillTasks =
        kMapBucketCapacity / kEntriesPerTask;
    static_assert(
        kFillTasks + 1U < pa_scheduler::kMaxTasks,
        "safe-prefix writers must stay inside task domain"
    );

    auto map = NewMap();
    RecordingOps::DisableEvents();
    const uint64_t address = 0x530000000ULL;
    const uint32_t bucket = TensorMapHash(address);

    // 用真实整 task append 把一桶填满：cursor 0..7 属于 producer 0，
    // cursor 8..15 属于 producer 1。task 2 的窗口下界为 1，因此前八条
    // 已经失效，但 cursor 8 仍是本次查询必须返回的合法 producer。
    for (uint32_t task = 0; task < kFillTasks; ++task) {
        std::vector<SharedRegionValue> entries;
        entries.reserve(kEntriesPerTask);
        for (uint32_t index = 0;
             index < kEntriesPerTask; ++index) {
            const uint64_t cursor =
                static_cast<uint64_t>(task) *
                    kEntriesPerTask +
                index;
            const uint64_t lo = cursor * 32U;
            entries.push_back(MakeRegion(
                address, lo, lo + 8U,
                static_cast<int32_t>(task)
            ));
        }
        if (TryCommitTask(
                *map, static_cast<int32_t>(task),
                entries
            ) != CommitResult::Committed) {
            std::fprintf(
                stderr,
                "[FAIL] shared TensorMap ring/%s: "
                "fill failed task=%u\n",
                kTest, task
            );
            ++g_failures;
            return;
        }
    }

    // 本门槛刻意撤掉隔离 driver 的 exact-turn 前沿。下面的合法回收只由
    // reader_done 推导，证明 lookup 的并发恢复不依赖 committed_tasks。
    StoreControl(&map->committed_tasks.value, 0);
    Expect(
        SharedAdvanceReaderDone<RecordingOps>(
            *map, 0, 0
        ) &&
            SharedAdvanceReaderDone<RecordingOps>(
                *map, 0, 1
            ),
        kTest, "reader closes tasks before task 2"
    );

    const int32_t writer_task =
        static_cast<int32_t>(kFillTasks);
    std::vector<SharedRegionValue> replacements;
    replacements.reserve(kEntriesPerTask);
    for (uint32_t index = 0;
         index < kEntriesPerTask; ++index) {
        const uint64_t lo =
            (static_cast<uint64_t>(kMapBucketCapacity) +
             index) *
            32U;
        replacements.push_back(MakeRegion(
            address, lo, lo + 8U, writer_task
        ));
    }

    SharedRegionSlot &first_slot =
        map->slots[SharedTensorMapSlotIndex(bucket, 0)];
    ConcurrentSafeRetireAttempt attempt{};
    attempt.map = map.get();
    attempt.replacements = replacements.data();
    attempt.replacement_count =
        static_cast<uint32_t>(replacements.size());
    attempt.active_readers = kActiveReaders;
    attempt.heap_window = kHeapWindow;
    attempt.candidate = -2;
    attempt.reclaim_upto = -2;
    attempt.append_check = SharedAppendCheck::ProtocolError;

    // reader 已读旧 head=0、尚未读取 tail 时，让唯一 writer 回收
    // producer 0 并提交八条 replacement。reader 随后会读到新
    // tail=CAP+8，形成“旧 head + 新 tail”的合法混合快照；lookup 必须
    // 重读 head=8 后继续，而不能因表面跨度大于 CAP 误报协议损坏。
    RecordingOps::ResetEvents();
    RecordingOps::load_hook_address =
        const_cast<const int64_t *>(
            &map->buckets[bucket].tail.value
        );
    RecordingOps::load_hook =
        RetireSafePrefixWhileReaderPaused;
    RecordingOps::load_hook_context = &attempt;
    bool protocol_ok = false;
    const uint64_t target_lo =
        static_cast<uint64_t>(kEntriesPerTask) * 32U;
    const int32_t producer =
        SharedLookupRegion<RecordingOps>(
            *map,
            MakeRegion(
                address, target_lo, target_lo + 8U, -1
            ),
            kReaderTask, kHeapWindow, protocol_ok
        );

    Expect(
        attempt.fired, kTest,
        "safe-retire hook fires between initial head and tail loads"
    );
    Expect(
        attempt.candidate_ok && attempt.candidate == 0 &&
            attempt.publish_ok &&
            attempt.reclaim_upto == 0,
        kTest, "reader frontier admits only producer 0"
    );
    Expect(
        attempt.append_check == SharedAppendCheck::Ready &&
            attempt.append_ok,
        kTest, "writer retires and replaces the safe prefix"
    );
    Expect(
        protocol_ok && producer == 1,
        kTest,
        "lookup repairs old-head/new-tail snapshot and returns producer 1"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].head.value),
        kEntriesPerTask, kTest,
        "safe retire advances head past producer 0"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].tail.value),
        kMapBucketCapacity + kEntriesPerTask,
        kTest, "replacement batch advances tail"
    );
    ExpectEqual(
        LoadControl(&map->committed_tasks.value),
        0, kTest,
        "reader-gated append does not use global exact turn"
    );
    ExpectEqual(
        LoadControl(&first_slot.seq.value),
        kMapBucketCapacity, kTest,
        "physical slot 0 carries the replacement absolute seq"
    );

    size_t head_exchange = RecordingOps::events.size();
    size_t seq_invalidate = RecordingOps::events.size();
    size_t payload_flush = RecordingOps::events.size();
    size_t seq_publish = RecordingOps::events.size();
    size_t tail_publish = RecordingOps::events.size();
    const void *const head_address =
        const_cast<const int64_t *>(
            &map->buckets[bucket].head.value
        );
    const void *const seq_address =
        const_cast<const int64_t *>(&first_slot.seq.value);
    const void *const tail_address =
        const_cast<const int64_t *>(
            &map->buckets[bucket].tail.value
        );
    for (size_t index = 0;
         index < RecordingOps::events.size(); ++index) {
        const Event &event = RecordingOps::events[index];
        if (event.kind == EventKind::Exchange &&
            event.address == head_address &&
            event.argument ==
                static_cast<int64_t>(kEntriesPerTask) &&
            head_exchange == RecordingOps::events.size()) {
            head_exchange = index;
        } else if (
            event.kind == EventKind::Exchange &&
            event.address == seq_address &&
            event.argument == kSharedMapEmptySeq &&
            seq_invalidate == RecordingOps::events.size()
        ) {
            seq_invalidate = index;
        } else if (
            event.kind == EventKind::Flush &&
            event.address == &first_slot.payload &&
            payload_flush == RecordingOps::events.size()
        ) {
            payload_flush = index;
        } else if (
            event.kind == EventKind::Exchange &&
            event.address == seq_address &&
            event.argument ==
                static_cast<int64_t>(kMapBucketCapacity) &&
            seq_publish == RecordingOps::events.size()
        ) {
            seq_publish = index;
        } else if (
            event.kind == EventKind::Exchange &&
            event.address == tail_address &&
            event.argument ==
                static_cast<int64_t>(
                    kMapBucketCapacity + 1U
                ) &&
            tail_publish == RecordingOps::events.size()
        ) {
            tail_publish = index;
        }
    }
    Expect(
        head_exchange < seq_invalidate &&
            seq_invalidate < payload_flush &&
            payload_flush < seq_publish &&
            seq_publish < tail_publish,
        kTest,
        "head publication precedes slot reuse and tail publication"
    );
    RecordingOps::DisableEvents();

    // 再固定更晚的交错：reader task 3 已读 cursor 8 的旧绝对 seq，
    // 尚未拷 payload 时，writer 回收 producer 1 并用 cursor CAP+8
    // 复用同一物理槽。第二次 seq 检查失败后，lookup 只能在新 head
    // 确实越过 cursor 8 时跳到 cursor 16，并返回 producer 2。
    Expect(
        SharedAdvanceReaderDone<RecordingOps>(
            *map, 0, 2
        ),
        kTest, "reader closes task 2 before task 3"
    );
    const int32_t second_writer_task = writer_task + 1;
    std::vector<SharedRegionValue> second_replacements;
    second_replacements.reserve(kEntriesPerTask);
    for (uint32_t index = 0;
         index < kEntriesPerTask; ++index) {
        const uint64_t lo =
            (static_cast<uint64_t>(kMapBucketCapacity) +
             kEntriesPerTask + index) *
            32U;
        second_replacements.push_back(MakeRegion(
            address, lo, lo + 8U, second_writer_task
        ));
    }
    ConcurrentSafeRetireAttempt second_attempt{};
    second_attempt.map = map.get();
    second_attempt.replacements = second_replacements.data();
    second_attempt.replacement_count =
        static_cast<uint32_t>(second_replacements.size());
    second_attempt.active_readers = kActiveReaders;
    second_attempt.heap_window = kHeapWindow;
    second_attempt.candidate = -2;
    second_attempt.reclaim_upto = -2;
    second_attempt.append_check =
        SharedAppendCheck::ProtocolError;

    SharedRegionSlot &second_retired_slot =
        map->slots[
            SharedTensorMapSlotIndex(bucket, kEntriesPerTask)
        ];
    RecordingOps::ResetEvents();
    RecordingOps::invalidate_hook_address =
        &second_retired_slot.payload;
    RecordingOps::invalidate_hook =
        RetireSafePrefixWhileReaderPaused;
    RecordingOps::invalidate_hook_context = &second_attempt;
    bool second_protocol_ok = false;
    const uint64_t second_target_lo =
        static_cast<uint64_t>(2U * kEntriesPerTask) * 32U;
    const int32_t second_producer =
        SharedLookupRegion<RecordingOps>(
            *map,
            MakeRegion(
                address, second_target_lo,
                second_target_lo + 8U, -1
            ),
            kReaderTask + 1, kHeapWindow,
            second_protocol_ok
        );
    Expect(
        second_attempt.fired, kTest,
        "safe-retire hook fires between slot seq checks"
    );
    Expect(
        second_attempt.candidate_ok &&
            second_attempt.candidate == 1 &&
            second_attempt.publish_ok &&
            second_attempt.reclaim_upto == 1,
        kTest, "next reader frontier admits only producer 1"
    );
    Expect(
        second_attempt.append_check ==
                SharedAppendCheck::Ready &&
            second_attempt.append_ok,
        kTest, "writer retires and replaces the next safe prefix"
    );
    Expect(
        second_protocol_ok && second_producer == 2,
        kTest,
        "lookup skips concurrently reused cursor 8 and returns producer 2"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].head.value),
        2U * kEntriesPerTask, kTest,
        "second safe retire advances head past producer 1"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].tail.value),
        kMapBucketCapacity + 2U * kEntriesPerTask,
        kTest, "second replacement batch advances tail"
    );
    ExpectEqual(
        LoadControl(&map->committed_tasks.value),
        0, kTest,
        "both reader-gated appends avoid global exact turn"
    );
    RecordingOps::DisableEvents();

    // 不能把任意 seq 错误都吞成并发回收。head 已停在 cursor 16 时，
    // 破坏 cursor 16 的 seq；二次读取 head 没有越过它，lookup 必须继续
    // fail-closed。
    SharedRegionSlot &target_slot =
        map->slots[
            SharedTensorMapSlotIndex(
                bucket, 2U * kEntriesPerTask
            )
        ];
    StoreControl(
        &target_slot.seq.value,
        static_cast<int64_t>(2U * kEntriesPerTask + 1U)
    );
    bool corrupt_protocol_ok = true;
    const int32_t corrupt_result =
        SharedLookupRegion<RecordingOps>(
            *map,
            MakeRegion(
                address, second_target_lo,
                second_target_lo + 8U, -1
            ),
            kReaderTask + 1, kHeapWindow,
            corrupt_protocol_ok
        );
    Expect(
        !corrupt_protocol_ok && corrupt_result == -1,
        kTest,
        "seq corruption without head advance remains a protocol error"
    );

    // 控制字段真实损坏也不能借异常恢复蒙混过去。没有并发 head 前进时，
    // 人为制造 CAP+1 的跨度，二次读取仍是旧 head，必须直接拒绝。
    auto oversized_map = NewMap();
    const uint32_t oversized_bucket = TensorMapHash(address);
    StoreControl(
        &oversized_map->buckets[oversized_bucket].tail.value,
        static_cast<int64_t>(kMapBucketCapacity + 1U)
    );
    bool oversized_protocol_ok = true;
    const int32_t oversized_result =
        SharedLookupRegion<RecordingOps>(
            *oversized_map,
            MakeRegion(address, 0, 8, -1),
            1, kHeapWindow, oversized_protocol_ok
        );
    Expect(
        !oversized_protocol_ok && oversized_result == -1,
        kTest,
        "oversized control span without head advance remains an error"
    );
}

void TestFullBucketDoesNotBlockIndependentBucket() {
    constexpr const char *kTest = "full-bucket-independent-bucket";
    if constexpr (kMapBuckets == 1) {
        // CAP=16384 的 B=1 形态没有可构造的独立桶；它仍由满环、回绕和
        // 精确复用门槛覆盖，不能伪造一个“不同桶”结论。
        return;
    }

    auto map = NewMap();
    RecordingOps::DisableEvents();
    const uint64_t full_address = 0x540000000ULL;
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
            Expect(false, kTest, "failed to fill the target bucket");
            return;
        }
    }

    const int64_t full_head =
        LoadControl(&map->buckets[full_bucket].head.value);
    const int64_t full_tail =
        LoadControl(&map->buckets[full_bucket].tail.value);
    const uint64_t independent_address =
        FindAddressOutsideBucket(0x550000000ULL, full_bucket);
    const uint32_t independent_bucket =
        TensorMapHash(independent_address);
    Expect(
        independent_bucket != full_bucket, kTest,
        "test address must map to an independent bucket"
    );

    const SharedRegionValue independent = MakeRegion(
        independent_address, 0, 64,
        static_cast<int32_t>(kMapBucketCapacity)
    );
    Expect(
        TryCommitTask(
            *map, static_cast<int32_t>(kMapBucketCapacity),
            {independent}
        ) == CommitResult::Committed,
        kTest, "a full bucket must not become a global capacity gate"
    );
    ExpectEqual(
        LoadControl(&map->buckets[full_bucket].head.value), full_head,
        kTest, "independent append preserves full-bucket head"
    );
    ExpectEqual(
        LoadControl(&map->buckets[full_bucket].tail.value), full_tail,
        kTest, "independent append preserves full-bucket tail"
    );
    ExpectEqual(
        LoadControl(&map->buckets[independent_bucket].tail.value), 1,
        kTest, "independent bucket publishes one entry"
    );
    SharedRegionSlot &slot =
        map->slots[SharedTensorMapSlotIndex(independent_bucket, 0)];
    ExpectEqual(
        LoadControl(&slot.seq.value), 0,
        kTest, "independent bucket publishes its own absolute seq"
    );
}

void TestFullBucketRetireAndReuseExactCapacity() {
    constexpr const char *kTest = "full-bucket-retire-reuse-exact";
    auto map = NewMap();
    RecordingOps::DisableEvents();
    const uint64_t address = 0x560000000ULL;
    const uint32_t bucket = TensorMapHash(address);
    for (uint32_t task = 0; task < kMapBucketCapacity; ++task) {
        const CommitResult result = TryCommitTask(
            *map, static_cast<int32_t>(task),
            {MakeRegion(
                address, static_cast<uint64_t>(task) * 32,
                static_cast<uint64_t>(task) * 32 + 8,
                static_cast<int32_t>(task)
            )}
        );
        if (result != CommitResult::Committed) {
            Expect(false, kTest, "failed to fill the target bucket");
            return;
        }
    }

    constexpr uint32_t kReuse =
        kMapBucketCapacity / 4U < 8U
            ? kMapBucketCapacity / 4U
            : 8U;
    static_assert(kReuse > 0, "exact reuse test requires a non-zero batch");
    std::vector<SharedRegionValue> replacements;
    replacements.reserve(kReuse);
    for (uint32_t index = 0; index < kReuse; ++index) {
        const uint64_t lo =
            (1ULL << 20U) + static_cast<uint64_t>(index) * 32U;
        replacements.push_back(MakeRegion(
            address, lo, lo + 8U,
            static_cast<int32_t>(kMapBucketCapacity)
        ));
    }
    const int32_t heap_window =
        static_cast<int32_t>(kMapBucketCapacity - kReuse);
    Expect(
        TryCommitTask(
            *map, static_cast<int32_t>(kMapBucketCapacity),
            replacements, heap_window
        ) == CommitResult::Committed,
        kTest, "retiring exactly K entries must admit exactly K replacements"
    );
    ExpectEqual(
        LoadControl(&map->reclaim_upto.value), kReuse - 1,
        kTest, "inclusive reclaim boundary"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].head.value), kReuse,
        kTest, "bucket head retires exactly K old entries"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].tail.value),
        kMapBucketCapacity + kReuse,
        kTest, "bucket tail appends exactly K replacement entries"
    );
    for (uint32_t index = 0; index < kReuse; ++index) {
        const uint64_t cursor =
            static_cast<uint64_t>(kMapBucketCapacity) + index;
        SharedRegionSlot &slot =
            map->slots[SharedTensorMapSlotIndex(bucket, cursor)];
        ExpectEqual(
            LoadControl(&slot.seq.value), cursor,
            kTest, "reused physical slot publishes the new absolute seq"
        );
        ExpectEqual(
            slot.payload.value.producer, kMapBucketCapacity,
            kTest, "reused physical slot contains the replacement producer"
        );
    }
    std::vector<LogicalTuple> snapshot;
    Expect(SnapshotLogicalMap(*map, snapshot), kTest, "snapshot after exact reuse");
    ExpectEqual(
        snapshot.size(), kMapBucketCapacity,
        kTest, "exact reuse keeps the bucket logically full"
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

void TestTailOverflowRejectedBeforeMutation() {
    constexpr const char *kTest = "tail-overflow";
    auto map = NewMap();
    RecordingOps::DisableEvents();
    const SharedRegionValue entry =
        MakeRegion(0x6A0000000ULL, 0, 64, 7);
    const uint32_t bucket =
        TensorMapHash(entry.buffer_addr);
    StoreControl(
        &map->buckets[bucket].head.value, INT64_MAX
    );
    StoreControl(
        &map->buckets[bucket].tail.value, INT64_MAX
    );

    RecordingOps::ResetEvents();
    Expect(
        SharedCheckTaskAppend<RecordingOps>(
            *map, &entry, 1, -1
        ) == SharedAppendCheck::ProtocolError,
        kTest,
        "preflight rejects a tail with no representable successor"
    );
    Expect(
        !pa_scheduler::SharedAppendPreparedEntry<RecordingOps>(
            *map, entry
        ),
        kTest,
        "append rejects a tail with no representable successor"
    );
    for (const Event &event : RecordingOps::events) {
        if (event.kind == EventKind::Exchange ||
            event.kind == EventKind::Flush) {
            Expect(
                false, kTest,
                "overflow rejection performed a state publication"
            );
            break;
        }
    }
    RecordingOps::DisableEvents();
    ExpectEqual(
        LoadControl(&map->buckets[bucket].head.value),
        INT64_MAX, kTest, "overflow rejection preserves head"
    );
    ExpectEqual(
        LoadControl(&map->buckets[bucket].tail.value),
        INT64_MAX, kTest, "overflow rejection preserves tail"
    );
}

}  // namespace

int main() {
    TestPhysicalSlotBoundaries();
    TestAbiResetAndZeroEntryCommit();
    TestPublicationOrderAndDoubleSeqCheck();
    TestVersionsWindowAndMultipleBuckets();
    TestOrderedReclaimFormulaAndExactTurn();
    TestAbsoluteSeqMultipleLapsAndAba();
    TestCapacityFailureIsAllOrNothing();
    TestSlowReaderGatesFullBucketReuse();
    TestLookupSkipsConcurrentlyRetiredSafePrefix();
    TestFullBucketDoesNotBlockIndependentBucket();
    TestFullBucketRetireAndReuseExactCapacity();
    TestCapacityBlockedAfterSafeRetire();
    TestDeterministicArrivalAndLogicalTupleDifference();
    TestTailOverflowRejectedBeforeMutation();

    if (g_failures != 0) {
        std::fprintf(stderr, "[FAIL] shared TensorMap ring: %d failure(s)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf(
        "[PASS] shared TensorMap ring CAP=%u buckets=%u: "
        "ABI, ordered commit, slow-reader reclaim, absolute seq, "
        "concurrent safe-prefix retire, ABA, overflow, logical differential\n",
        kMapBucketCapacity, kMapBuckets
    );
    std::printf(
        "[NOTE] CPU validates atomic/order hooks only; it does not simulate A5 DCache or DCCI.\n"
    );
    return EXIT_SUCCESS;
}
