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

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "inner_kernel.h"
// shared_tensor_map.h 必须在未被 AICPU 初始化头间接补齐依赖时独立可编译。
#include "dist_engine/aicore/shared_tensor_map.h"
#include "dist_engine/aicpu/shared_tensor_map_init.h"

namespace {

static_assert(PTO_FDWIC_SHARED_MAP == 1, "shared TensorMap ring tests require the shared artifact identity");

enum class EventKind : uint8_t {
    Load,
    CompareExchange,
    Invalidate,
    Flush,
};

struct Event {
    EventKind kind;
    const void *address;
    int64_t expected;
    int64_t desired;
    int64_t observed;
};

const void *control_address(volatile int64_t *address) { return const_cast<const int64_t *>(address); }

struct RecordingOps {
    inline static std::vector<Event> events{};
    inline static bool record = false;
    inline static const void *mutate_payload = nullptr;
    inline static volatile int64_t *mutate_sequence = nullptr;
    inline static int64_t mutate_sequence_value = 0;
    inline static bool mutate_once = false;
    inline static volatile int64_t *mutate_cas_address = nullptr;
    inline static int64_t mutate_cas_expected = 0;
    inline static int64_t mutate_cas_desired = 0;
    inline static int64_t mutate_cas_value = 0;
    inline static bool mutate_cas_once = false;

    static int64_t Load(volatile int64_t *address) {
        const int64_t value = __atomic_fetch_add(address, int64_t{0}, __ATOMIC_ACQUIRE);
        if (record) {
            events.push_back({EventKind::Load, control_address(address), 0, 0, value});
        }
        return value;
    }

    static int64_t CompareExchange(volatile int64_t *address, int64_t expected, int64_t desired) {
        // 在被测 CAS 的线性化点前确定性模拟一个非法并发 writer。真实
        // exact-turn 路径不允许该竞争；注入只用于证明失败 CAS 不会再把
        // 对方刚发布的控制字覆盖掉。
        if (mutate_cas_once && address == mutate_cas_address && expected == mutate_cas_expected &&
            desired == mutate_cas_desired) {
            __atomic_store_n(address, mutate_cas_value, __ATOMIC_RELEASE);
            mutate_cas_once = false;
        }
        int64_t observed = expected;
        (void
        )__atomic_compare_exchange_n(address, &observed, desired, /*weak=*/false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
        if (record) {
            events.push_back({EventKind::CompareExchange, control_address(address), expected, desired, observed});
        }
        return observed;
    }

    static void InvalidateRegion(const void *address, uint64_t bytes) {
        if (record) {
            events.push_back({EventKind::Invalidate, address, 0, static_cast<int64_t>(bytes), 0});
        }
        // 在第一次 seq 检查与 payload snapshot 之间确定性制造下一 lap 复用，
        // 证明第二次 seq 检查不是装饰。
        if (mutate_once && address == mutate_payload && mutate_sequence != nullptr) {
            __atomic_store_n(mutate_sequence, mutate_sequence_value, __ATOMIC_RELEASE);
            mutate_once = false;
        }
    }

    static void FlushRegion(void *address, uint64_t bytes) {
        if (record) {
            events.push_back({EventKind::Flush, address, 0, static_cast<int64_t>(bytes), 0});
        }
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }

    static void ResetEvents(bool enable = true) {
        events.clear();
        record = enable;
        mutate_payload = nullptr;
        mutate_sequence = nullptr;
        mutate_sequence_value = 0;
        mutate_once = false;
        mutate_cas_address = nullptr;
        mutate_cas_expected = 0;
        mutate_cas_desired = 0;
        mutate_cas_value = 0;
        mutate_cas_once = false;
    }

    static void MutateBeforeCas(volatile int64_t *address, int64_t expected, int64_t desired, int64_t competing_value) {
        if (competing_value == expected || competing_value == desired) {
            throw std::logic_error("CAS competing value must differ from expected and desired");
        }
        mutate_cas_address = address;
        mutate_cas_expected = expected;
        mutate_cas_desired = desired;
        mutate_cas_value = competing_value;
        mutate_cas_once = true;
    }
};

std::unique_ptr<SharedTensorMapState> make_empty_map() {
    auto map = std::make_unique<SharedTensorMapState>();
    std::memset(map.get(), 0xa5, sizeof(*map));
    dist_shared_tensor_map_reset(*map);
    RecordingOps::ResetEvents(false);
    return map;
}

SharedTensorMapValue make_region(uint64_t address, uint64_t lo, uint64_t hi, int32_t producer) {
    return {address, lo, hi, producer, 0};
}

uint64_t find_address_in_bucket(uint32_t target_bucket, uint64_t seed) {
    constexpr uint64_t kSearchLimit = 1ULL << 22;
    for (uint64_t step = 0; step < kSearchLimit; ++step) {
        const uint64_t candidate = seed + step * 64;
        if (dist_tensor_map_hash(candidate) == target_bucket) {
            return candidate;
        }
    }
    throw std::logic_error("failed to find shared TensorMap address in requested bucket");
}

uint64_t find_address_outside_bucket(uint32_t excluded_bucket, uint64_t seed) {
    if constexpr (kMapBuckets == 1) {
        return seed;
    }
    uint64_t candidate = seed;
    while (dist_tensor_map_hash(candidate) == excluded_bucket) {
        candidate += 64;
    }
    return candidate;
}

enum class DriverResult : uint8_t {
    Pending,
    Committed,
    CapacityBlocked,
    ProtocolError,
};

DriverResult try_commit(
    SharedTensorMapState &map, int32_t task_id, const SharedTensorMapValue *entries, uint32_t count,
    int32_t history = INT32_MAX
) {
    if (task_id < 0 || task_id >= kFlagCap) {
        return DriverResult::ProtocolError;
    }
    const int64_t committed = RecordingOps::Load(&map.committed_tasks.v);
    if (committed < task_id) {
        return DriverResult::Pending;
    }
    if (committed > task_id) {
        return DriverResult::ProtocolError;
    }
    int64_t reclaim_upto = -2;
    if (!dist_shared_tensor_map_refresh_reclaim_impl<RecordingOps>(map, task_id, history, reclaim_upto)) {
        return DriverResult::ProtocolError;
    }
    const DistSharedTensorMapAppendCheck check =
        dist_shared_tensor_map_check_task_append_impl<RecordingOps>(map, entries, count, task_id, reclaim_upto);
    if (check == DistSharedTensorMapAppendCheck::CapacityBlocked) {
        return DriverResult::CapacityBlocked;
    }
    if (check != DistSharedTensorMapAppendCheck::Ready ||
        !dist_shared_tensor_map_append_prepared_task_impl<RecordingOps>(map, entries, count, task_id) ||
        !dist_shared_tensor_map_publish_commit_impl<RecordingOps>(map, task_id)) {
        return DriverResult::ProtocolError;
    }
    return DriverResult::Committed;
}

DriverResult try_commit(
    SharedTensorMapState &map, int32_t task_id, const std::vector<SharedTensorMapValue> &entries,
    int32_t history = INT32_MAX
) {
    return try_commit(map, task_id, entries.data(), static_cast<uint32_t>(entries.size()), history);
}

size_t find_event(EventKind kind, const void *address, size_t begin, bool check_desired = false, int64_t desired = 0) {
    for (size_t index = begin; index < RecordingOps::events.size(); ++index) {
        const Event &event = RecordingOps::events[index];
        if (event.kind == kind && event.address == address && (!check_desired || event.desired == desired)) {
            return index;
        }
    }
    return RecordingOps::events.size();
}

void expect_recorded_cas(const void *address, int64_t expected, int64_t desired, int64_t observed) {
    const size_t index = find_event(EventKind::CompareExchange, address, 0, true, desired);
    ASSERT_LT(index, RecordingOps::events.size());
    EXPECT_EQ(RecordingOps::events[index].expected, expected);
    EXPECT_EQ(RecordingOps::events[index].desired, desired);
    EXPECT_EQ(RecordingOps::events[index].observed, observed);
}

TEST(FdwicSharedTensorMapRing, PhysicalBoundariesValueConversionAndZeroEntryCommit) {
    auto map = make_empty_map();
    EXPECT_EQ(dist_shared_tensor_map_slot_index(0, 0), 0U);
    EXPECT_EQ(
        dist_shared_tensor_map_slot_index(kMapBuckets - 1, kMapBucketCapacity - 1), static_cast<uint32_t>(kMapCap - 1)
    );
    EXPECT_EQ(
        dist_shared_tensor_map_slot_index(kMapBuckets - 1, kMapBucketCapacity), (kMapBuckets - 1) * kMapBucketCapacity
    );

    Tensor tensor{};
    tensor.buffer = {0x12340000, 1ULL << 20};
    tensor.start_offset = 3;
    tensor.ndims = 1;
    tensor.dtype = DataType::FLOAT32;
    tensor.is_contiguous = true;
    tensor.shapes[0] = 7;
    tensor.extent_elem_cache = 7;
    const SharedTensorMapValue value = dist_shared_tensor_map_make_value(tensor, 9);
    EXPECT_EQ(value.buf_addr, tensor.buffer.addr);
    EXPECT_EQ(value.lo, 12U);
    EXPECT_EQ(value.hi, 40U);
    EXPECT_EQ(value.producer, 9);
    EXPECT_EQ(value.reserved, 0U);

    const std::vector<SharedTensorMapValue> empty;
    EXPECT_EQ(try_commit(*map, 0, empty), DriverResult::Committed);
    EXPECT_EQ(try_commit(*map, 1, empty), DriverResult::Committed);
    EXPECT_EQ(map->committed_tasks.v, 2);
    EXPECT_EQ(map->reclaim_upto.v, -1);
    for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
        EXPECT_EQ(map->buckets[bucket].head.v, 0);
        EXPECT_EQ(map->buckets[bucket].tail.v, 0);
    }

    EXPECT_FALSE(dist_shared_tensor_map_publish_commit_impl<RecordingOps>(*map, 1));
    EXPECT_EQ(map->committed_tasks.v, 2);
    map->committed_tasks.v = 0;
    EXPECT_FALSE(dist_shared_tensor_map_publish_commit_impl<RecordingOps>(*map, 1));
    EXPECT_EQ(map->committed_tasks.v, 0);
}

TEST(FdwicSharedTensorMapRing, PublicationOrderReaderOrderAndDoubleSequenceRejectAba) {
    auto map = make_empty_map();
    const SharedTensorMapValue entry = make_region(0x100000000ULL, 0, 64, 0);
    const uint32_t bucket = dist_tensor_map_hash(entry.buf_addr);
    SharedTensorMapSlot &slot = map->slots[dist_shared_tensor_map_slot_index(bucket, 0)];

    RecordingOps::ResetEvents();
    EXPECT_EQ(try_commit(*map, 0, {entry}), DriverResult::Committed);

    const void *sequence_address = control_address(&slot.sequence.v);
    const void *payload_address = &slot.payload;
    const void *tail_address = control_address(&map->buckets[bucket].tail.v);
    const void *commit_address = control_address(&map->committed_tasks.v);
    const size_t sequence_claim =
        find_event(EventKind::CompareExchange, sequence_address, 0, true, kSharedTensorMapWritingSequence);
    const size_t payload_invalidate = find_event(EventKind::Invalidate, payload_address, sequence_claim + 1);
    const size_t payload_flush = find_event(EventKind::Flush, payload_address, payload_invalidate + 1);
    const size_t sequence_publish =
        find_event(EventKind::CompareExchange, sequence_address, payload_flush + 1, true, 0);
    const size_t tail_publish = find_event(EventKind::CompareExchange, tail_address, sequence_publish + 1, true, 1);
    const size_t task_commit = find_event(EventKind::CompareExchange, commit_address, tail_publish + 1, true, 1);
    ASSERT_LT(sequence_claim, payload_invalidate);
    ASSERT_LT(payload_invalidate, payload_flush);
    ASSERT_LT(payload_flush, sequence_publish);
    ASSERT_LT(sequence_publish, tail_publish);
    ASSERT_LT(tail_publish, task_commit);
    EXPECT_EQ(RecordingOps::events[sequence_claim].expected, kSharedTensorMapInvalidSequence);
    EXPECT_EQ(RecordingOps::events[sequence_publish].expected, kSharedTensorMapWritingSequence);
    EXPECT_EQ(RecordingOps::events[tail_publish].expected, 0);
    EXPECT_EQ(RecordingOps::events[task_commit].expected, 0);

    RecordingOps::ResetEvents();
    SharedTensorMapValue snapshot{};
    ASSERT_TRUE(dist_shared_tensor_map_read_slot_impl<RecordingOps>(*map, bucket, 0, snapshot));
    EXPECT_EQ(snapshot.buf_addr, entry.buf_addr);
    EXPECT_EQ(snapshot.lo, entry.lo);
    EXPECT_EQ(snapshot.hi, entry.hi);
    EXPECT_EQ(snapshot.producer, entry.producer);
    const size_t first_load = find_event(EventKind::Load, sequence_address, 0);
    const size_t read_invalidate = find_event(EventKind::Invalidate, payload_address, first_load + 1);
    const size_t second_load = find_event(EventKind::Load, sequence_address, read_invalidate + 1);
    ASSERT_LT(first_load, read_invalidate);
    ASSERT_LT(read_invalidate, second_load);

    RecordingOps::ResetEvents();
    RecordingOps::mutate_payload = payload_address;
    RecordingOps::mutate_sequence = &slot.sequence.v;
    RecordingOps::mutate_sequence_value = static_cast<int64_t>(kMapBucketCapacity);
    RecordingOps::mutate_once = true;
    SharedTensorMapValue raced{};
    EXPECT_FALSE(dist_shared_tensor_map_read_slot_impl<RecordingOps>(*map, bucket, 0, raced));
    EXPECT_FALSE(RecordingOps::mutate_once);
}

TEST(FdwicSharedTensorMapRing, ControlCasFailuresDoNotOverwriteCompetingValues) {
    {
        auto map = make_empty_map();
        const SharedTensorMapValue entry = make_region(0x180000000ULL, 0, 64, 0);
        const uint32_t bucket = dist_tensor_map_hash(entry.buf_addr);
        ASSERT_EQ(try_commit(*map, 0, {entry}), DriverResult::Committed);
        const int64_t tail_before = map->buckets[bucket].tail.v;
        const int64_t commit_before = map->committed_tasks.v;
        const SharedTensorMapSlot slot_before = map->slots[dist_shared_tensor_map_slot_index(bucket, 0)];

        RecordingOps::ResetEvents();
        RecordingOps::MutateBeforeCas(&map->buckets[bucket].head.v, 0, 1, 2);
        EXPECT_FALSE(dist_shared_tensor_map_retire_bucket_impl<RecordingOps>(*map, bucket, 0));
        EXPECT_FALSE(RecordingOps::mutate_cas_once);
        EXPECT_EQ(map->buckets[bucket].head.v, 2);
        expect_recorded_cas(control_address(&map->buckets[bucket].head.v), 0, 1, 2);
        EXPECT_EQ(map->buckets[bucket].tail.v, tail_before);
        EXPECT_EQ(map->committed_tasks.v, commit_before);
        EXPECT_EQ(
            std::memcmp(&slot_before, &map->slots[dist_shared_tensor_map_slot_index(bucket, 0)], sizeof(slot_before)), 0
        );
    }

    {
        auto map = make_empty_map();
        map->committed_tasks.v = 65;
        RecordingOps::ResetEvents();
        RecordingOps::MutateBeforeCas(&map->reclaim_upto.v, -1, 0, 7);
        int64_t reclaim_upto = -2;
        EXPECT_FALSE(dist_shared_tensor_map_refresh_reclaim_impl<RecordingOps>(*map, 65, 64, reclaim_upto));
        EXPECT_FALSE(RecordingOps::mutate_cas_once);
        EXPECT_EQ(reclaim_upto, -2);
        EXPECT_EQ(map->reclaim_upto.v, 7);
        expect_recorded_cas(control_address(&map->reclaim_upto.v), -1, 0, 7);
        EXPECT_EQ(map->committed_tasks.v, 65);
    }

    {
        auto map = make_empty_map();
        RecordingOps::ResetEvents();
        RecordingOps::MutateBeforeCas(&map->committed_tasks.v, 0, 1, 2);
        EXPECT_FALSE(dist_shared_tensor_map_publish_commit_impl<RecordingOps>(*map, 0));
        EXPECT_FALSE(RecordingOps::mutate_cas_once);
        EXPECT_EQ(map->committed_tasks.v, 2);
        expect_recorded_cas(control_address(&map->committed_tasks.v), 0, 1, 2);
        EXPECT_EQ(map->reclaim_upto.v, -1);
    }
}

TEST(FdwicSharedTensorMapRing, SlotAcquireCasFailurePublishesNothing) {
    auto map = make_empty_map();
    const SharedTensorMapValue entry = make_region(0x1a0000000ULL, 0, 64, 0);
    const uint32_t bucket = dist_tensor_map_hash(entry.buf_addr);
    SharedTensorMapSlot &slot = map->slots[dist_shared_tensor_map_slot_index(bucket, 0)];
    int64_t reclaim_upto = -2;
    ASSERT_TRUE(dist_shared_tensor_map_refresh_reclaim_impl<RecordingOps>(*map, 0, 64, reclaim_upto));
    ASSERT_EQ(
        dist_shared_tensor_map_check_task_append_impl<RecordingOps>(*map, &entry, 1, 0, reclaim_upto),
        DistSharedTensorMapAppendCheck::Ready
    );
    const SharedTensorMapPayloadLine payload_before = slot.payload;

    RecordingOps::ResetEvents();
    RecordingOps::MutateBeforeCas(
        &slot.sequence.v, kSharedTensorMapInvalidSequence, kSharedTensorMapWritingSequence, 77
    );
    EXPECT_FALSE(dist_shared_tensor_map_append_prepared_task_impl<RecordingOps>(*map, &entry, 1, 0));
    EXPECT_FALSE(RecordingOps::mutate_cas_once);
    EXPECT_EQ(slot.sequence.v, 77);
    expect_recorded_cas(
        control_address(&slot.sequence.v), kSharedTensorMapInvalidSequence, kSharedTensorMapWritingSequence, 77
    );
    EXPECT_EQ(std::memcmp(&payload_before, &slot.payload, sizeof(payload_before)), 0);
    EXPECT_EQ(map->buckets[bucket].tail.v, 0);
    EXPECT_EQ(map->committed_tasks.v, 0);
    for (const Event &event : RecordingOps::events) {
        EXPECT_NE(event.kind, EventKind::Invalidate);
        EXPECT_NE(event.kind, EventKind::Flush);
    }

    // 两个非法 writer 都已通过旧 preflight 时，第一个把槽置为 WRITING，
    // 第二个进入 append 仍必须在 ownership CAS 失败，不能写 payload。
    auto owned_map = make_empty_map();
    SharedTensorMapSlot &owned_slot = owned_map->slots[dist_shared_tensor_map_slot_index(bucket, 0)];
    owned_slot.sequence.v = kSharedTensorMapWritingSequence;
    const SharedTensorMapPayloadLine owned_payload_before = owned_slot.payload;
    EXPECT_FALSE(dist_shared_tensor_map_append_prepared_entry_impl<RecordingOps>(*owned_map, entry, 0));
    EXPECT_EQ(owned_slot.sequence.v, kSharedTensorMapWritingSequence);
    EXPECT_EQ(std::memcmp(&owned_payload_before, &owned_slot.payload, sizeof(owned_payload_before)), 0);
    EXPECT_EQ(owned_map->buckets[bucket].tail.v, 0);
}

TEST(FdwicSharedTensorMapRing, PostAcquireCasFailuresPreserveTheObservedControlWord) {
    const SharedTensorMapValue entry = make_region(0x1c0000000ULL, 16, 80, 0);

    {
        auto map = make_empty_map();
        const uint32_t bucket = dist_tensor_map_hash(entry.buf_addr);
        SharedTensorMapSlot &slot = map->slots[dist_shared_tensor_map_slot_index(bucket, 0)];
        int64_t reclaim_upto = -2;
        ASSERT_TRUE(dist_shared_tensor_map_refresh_reclaim_impl<RecordingOps>(*map, 0, 64, reclaim_upto));
        ASSERT_EQ(
            dist_shared_tensor_map_check_task_append_impl<RecordingOps>(*map, &entry, 1, 0, reclaim_upto),
            DistSharedTensorMapAppendCheck::Ready
        );

        RecordingOps::ResetEvents();
        RecordingOps::MutateBeforeCas(&slot.sequence.v, kSharedTensorMapWritingSequence, 0, 77);
        EXPECT_FALSE(dist_shared_tensor_map_append_prepared_task_impl<RecordingOps>(*map, &entry, 1, 0));
        EXPECT_FALSE(RecordingOps::mutate_cas_once);
        EXPECT_EQ(slot.sequence.v, 77);
        expect_recorded_cas(control_address(&slot.sequence.v), kSharedTensorMapWritingSequence, 0, 77);
        EXPECT_EQ(slot.payload.value.buf_addr, entry.buf_addr);
        EXPECT_EQ(slot.payload.value.lo, entry.lo);
        EXPECT_EQ(slot.payload.value.hi, entry.hi);
        EXPECT_EQ(slot.payload.value.producer, entry.producer);
        EXPECT_EQ(map->buckets[bucket].tail.v, 0);
        EXPECT_EQ(map->committed_tasks.v, 0);
        EXPECT_LT(find_event(EventKind::Flush, &slot.payload, 0), RecordingOps::events.size());
    }

    {
        auto map = make_empty_map();
        const uint32_t bucket = dist_tensor_map_hash(entry.buf_addr);
        SharedTensorMapSlot &slot = map->slots[dist_shared_tensor_map_slot_index(bucket, 0)];
        int64_t reclaim_upto = -2;
        ASSERT_TRUE(dist_shared_tensor_map_refresh_reclaim_impl<RecordingOps>(*map, 0, 64, reclaim_upto));
        ASSERT_EQ(
            dist_shared_tensor_map_check_task_append_impl<RecordingOps>(*map, &entry, 1, 0, reclaim_upto),
            DistSharedTensorMapAppendCheck::Ready
        );

        RecordingOps::ResetEvents();
        RecordingOps::MutateBeforeCas(&map->buckets[bucket].tail.v, 0, 1, 7);
        EXPECT_FALSE(dist_shared_tensor_map_append_prepared_task_impl<RecordingOps>(*map, &entry, 1, 0));
        EXPECT_FALSE(RecordingOps::mutate_cas_once);
        EXPECT_EQ(slot.sequence.v, 0);
        EXPECT_EQ(slot.payload.value.producer, entry.producer);
        EXPECT_EQ(map->buckets[bucket].tail.v, 7);
        expect_recorded_cas(control_address(&map->buckets[bucket].tail.v), 0, 1, 7);
        EXPECT_EQ(map->committed_tasks.v, 0);
    }

    {
        auto map = make_empty_map();
        const uint32_t bucket = dist_tensor_map_hash(entry.buf_addr);
        int64_t reclaim_upto = -2;
        ASSERT_TRUE(dist_shared_tensor_map_refresh_reclaim_impl<RecordingOps>(*map, 0, 64, reclaim_upto));
        ASSERT_EQ(
            dist_shared_tensor_map_check_task_append_impl<RecordingOps>(*map, &entry, 1, 0, reclaim_upto),
            DistSharedTensorMapAppendCheck::Ready
        );
        ASSERT_TRUE(dist_shared_tensor_map_append_prepared_task_impl<RecordingOps>(*map, &entry, 1, 0));

        RecordingOps::ResetEvents();
        RecordingOps::MutateBeforeCas(&map->committed_tasks.v, 0, 1, 2);
        EXPECT_FALSE(dist_shared_tensor_map_publish_commit_impl<RecordingOps>(*map, 0));
        EXPECT_FALSE(RecordingOps::mutate_cas_once);
        EXPECT_EQ(map->committed_tasks.v, 2);
        expect_recorded_cas(control_address(&map->committed_tasks.v), 0, 1, 2);
        EXPECT_EQ(map->buckets[bucket].tail.v, 1);
        EXPECT_EQ(map->slots[dist_shared_tensor_map_slot_index(bucket, 0)].sequence.v, 0);
        size_t commit_cas_count = 0;
        for (const Event &event : RecordingOps::events) {
            if (event.kind == EventKind::CompareExchange && event.address == control_address(&map->committed_tasks.v)) {
                ++commit_cas_count;
            }
        }
        EXPECT_EQ(commit_cas_count, 1U);
    }
}

TEST(FdwicSharedTensorMapRing, LookupUsesHalfOpenHistoryWindowAndMaximumProducer) {
    auto map = make_empty_map();
    const uint64_t versioned_address = 0x200000000ULL;
    const uint32_t versioned_bucket = dist_tensor_map_hash(versioned_address);
    const uint64_t stale_address = find_address_outside_bucket(versioned_bucket, 0x210000000ULL);
    const uint64_t lower_address = find_address_outside_bucket(dist_tensor_map_hash(stale_address), 0x220000000ULL);
    const uint64_t future_address = find_address_outside_bucket(dist_tensor_map_hash(lower_address), 0x230000000ULL);

    EXPECT_EQ(
        try_commit(
            *map, 0,
            {
                make_region(versioned_address, 0, 64, 0),
                make_region(versioned_address, 128, 192, 0),
                make_region(stale_address, 0, 64, 0),
            }
        ),
        DriverResult::Committed
    );
    EXPECT_EQ(
        try_commit(
            *map, 1,
            {
                make_region(versioned_address, 0, 64, 1),
                make_region(lower_address, 0, 64, 1),
            }
        ),
        DriverResult::Committed
    );
    EXPECT_EQ(
        try_commit(
            *map, 2,
            {
                make_region(versioned_address, 0, 64, 2),
                make_region(future_address, 0, 64, 2),
            }
        ),
        DriverResult::Committed
    );

    bool protocol_ok = false;
    EXPECT_EQ(
        dist_shared_tensor_map_lookup_region_impl<RecordingOps>(
            *map, make_region(versioned_address, 16, 32, -1), 3, 2, protocol_ok
        ),
        2
    );
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(
        dist_shared_tensor_map_lookup_region_impl<RecordingOps>(
            *map, make_region(versioned_address, 16, 32, -1), 2, 2, protocol_ok
        ),
        1
    );
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(
        dist_shared_tensor_map_lookup_region_impl<RecordingOps>(
            *map, make_region(stale_address, 0, 32, -1), 3, 2, protocol_ok
        ),
        -1
    );
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(
        dist_shared_tensor_map_lookup_region_impl<RecordingOps>(
            *map, make_region(lower_address, 0, 32, -1), 3, 2, protocol_ok
        ),
        1
    );
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(
        dist_shared_tensor_map_lookup_region_impl<RecordingOps>(
            *map, make_region(future_address, 0, 32, -1), 2, 2, protocol_ok
        ),
        -1
    );
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(
        dist_shared_tensor_map_lookup_region_impl<RecordingOps>(
            *map, make_region(versioned_address, 64, 128, -1), 3, 3, protocol_ok
        ),
        -1
    );
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(
        dist_shared_tensor_map_lookup_region_impl<RecordingOps>(
            *map, make_region(versioned_address, 0, 64, -1), kFlagCap, 3, protocol_ok
        ),
        -1
    );
    EXPECT_FALSE(protocol_ok);
}

TEST(FdwicSharedTensorMapRing, ExactTurnAndInclusiveReclaimAdvanceMonotonically) {
    int64_t candidate = -2;
    ASSERT_TRUE(dist_shared_tensor_map_compute_reclaim(0, 64, candidate));
    EXPECT_EQ(candidate, -1);
    ASSERT_TRUE(dist_shared_tensor_map_compute_reclaim(64, 64, candidate));
    EXPECT_EQ(candidate, -1);
    ASSERT_TRUE(dist_shared_tensor_map_compute_reclaim(65, 64, candidate));
    EXPECT_EQ(candidate, 0);
    ASSERT_TRUE(dist_shared_tensor_map_compute_reclaim(1279, 64, candidate));
    EXPECT_EQ(candidate, 1214);
    EXPECT_FALSE(dist_shared_tensor_map_compute_reclaim(-1, 64, candidate));
    EXPECT_FALSE(dist_shared_tensor_map_compute_reclaim(0, -1, candidate));

    auto map = make_empty_map();
    const uint64_t address = 0x300000000ULL;
    const uint32_t bucket = dist_tensor_map_hash(address);
    EXPECT_EQ(try_commit(*map, 0, {make_region(address, 0, 32, 0)}), DriverResult::Committed);
    EXPECT_EQ(try_commit(*map, 1, {make_region(address, 64, 96, 1)}), DriverResult::Committed);
    EXPECT_EQ(try_commit(*map, 2, std::vector<SharedTensorMapValue>{}, 2), DriverResult::Committed);
    EXPECT_EQ(map->committed_tasks.v, 3);
    EXPECT_EQ(map->reclaim_upto.v, -1);

    int64_t reclaim_upto = -2;
    EXPECT_FALSE(dist_shared_tensor_map_refresh_reclaim_impl<RecordingOps>(*map, 2, 2, reclaim_upto));
    EXPECT_FALSE(dist_shared_tensor_map_refresh_reclaim_impl<RecordingOps>(*map, 4, 2, reclaim_upto));
    EXPECT_EQ(map->reclaim_upto.v, -1);
    EXPECT_EQ(map->buckets[bucket].head.v, 0);
    EXPECT_EQ(map->buckets[bucket].tail.v, 2);

    EXPECT_EQ(try_commit(*map, 3, std::vector<SharedTensorMapValue>{}, 2), DriverResult::Committed);
    EXPECT_EQ(map->committed_tasks.v, 4);
    EXPECT_EQ(map->reclaim_upto.v, 0);
    EXPECT_EQ(map->buckets[bucket].head.v, 0);
    ASSERT_TRUE(dist_shared_tensor_map_retire_bucket_impl<RecordingOps>(*map, bucket, map->reclaim_upto.v));
    EXPECT_EQ(map->buckets[bucket].head.v, 1);

    map->reclaim_upto.v = 2;
    map->committed_tasks.v = 4;
    RecordingOps::ResetEvents();
    EXPECT_FALSE(dist_shared_tensor_map_refresh_reclaim_impl<RecordingOps>(*map, 4, 2, reclaim_upto));
    EXPECT_EQ(map->reclaim_upto.v, 2);
    for (const Event &event : RecordingOps::events) {
        EXPECT_NE(event.kind, EventKind::CompareExchange);
    }
}

TEST(FdwicSharedTensorMapRing, SameTaskCapacityFailurePublishesNothing) {
    auto map = make_empty_map();
    const uint64_t address = 0x400000000ULL;
    const uint32_t bucket = dist_tensor_map_hash(address);
    for (uint32_t task = 0; task + 1 < kMapBucketCapacity; ++task) {
        ASSERT_EQ(
            try_commit(*map, static_cast<int32_t>(task), {make_region(address, task * 8, task * 8 + 4, task)}),
            DriverResult::Committed
        );
    }
    ASSERT_EQ(map->buckets[bucket].tail.v - map->buckets[bucket].head.v, kMapBucketCapacity - 1);

    const int32_t task_id = static_cast<int32_t>(kMapBucketCapacity - 1);
    const std::vector<SharedTensorMapValue> entries = {
        make_region(address, 0x100000, 0x100004, task_id),
        make_region(address, 0x200000, 0x200004, task_id),
    };
    std::vector<std::byte> before(sizeof(*map));
    std::memcpy(before.data(), map.get(), before.size());
    EXPECT_EQ(try_commit(*map, task_id, entries), DriverResult::CapacityBlocked);
    EXPECT_EQ(std::memcmp(before.data(), map.get(), before.size()), 0);
}

TEST(FdwicSharedTensorMapRing, CrossBucketCapacityFailureDoesNotPublishEarlierEntry) {
    if constexpr (kMapBuckets == 1) {
        GTEST_SKIP() << "single-bucket CAP16384 variant has no cross-bucket case";
    }
    auto map = make_empty_map();
    const uint64_t full_address = 0x500000000ULL;
    const uint32_t full_bucket = dist_tensor_map_hash(full_address);
    for (uint32_t task = 0; task < kMapBucketCapacity; ++task) {
        ASSERT_EQ(
            try_commit(
                *map, static_cast<int32_t>(task),
                {make_region(full_address, task * 8, task * 8 + 4, static_cast<int32_t>(task))}
            ),
            DriverResult::Committed
        );
    }
    const uint64_t other_address = find_address_outside_bucket(full_bucket, 0x510000000ULL);
    ASSERT_NE(dist_tensor_map_hash(other_address), full_bucket);
    const int32_t task_id = static_cast<int32_t>(kMapBucketCapacity);
    const std::vector<SharedTensorMapValue> entries = {
        make_region(other_address, 0, 4, task_id),
        make_region(full_address, 0x100000, 0x100004, task_id),
    };
    std::vector<std::byte> before(sizeof(*map));
    std::memcpy(before.data(), map.get(), before.size());
    EXPECT_EQ(try_commit(*map, task_id, entries), DriverResult::CapacityBlocked);
    EXPECT_EQ(std::memcmp(before.data(), map.get(), before.size()), 0);
}

TEST(FdwicSharedTensorMapRing, CapacityFailureAfterRetirePublishesNoTaskData) {
    auto map = make_empty_map();
    const uint64_t address = 0x580000000ULL;
    const uint32_t bucket = dist_tensor_map_hash(address);
    for (uint32_t task = 0; task < kMapBucketCapacity; ++task) {
        ASSERT_EQ(
            try_commit(
                *map, static_cast<int32_t>(task),
                {make_region(address, task * 8, task * 8 + 4, static_cast<int32_t>(task))}
            ),
            DriverResult::Committed
        );
    }

    const int64_t tail_before = map->buckets[bucket].tail.v;
    const int64_t commit_before = map->committed_tasks.v;
    std::vector<std::byte> slots_before(sizeof(map->slots));
    std::memcpy(slots_before.data(), map->slots, slots_before.size());
    std::array<int64_t, kMapBuckets> heads_before{};
    std::array<int64_t, kMapBuckets> tails_before{};
    for (uint32_t index = 0; index < kMapBuckets; ++index) {
        heads_before[index] = map->buckets[index].head.v;
        tails_before[index] = map->buckets[index].tail.v;
    }

    const int32_t task_id = static_cast<int32_t>(kMapBucketCapacity);
    const std::vector<SharedTensorMapValue> entries = {
        make_region(address, 0x100000, 0x100004, task_id),
        make_region(address, 0x200000, 0x200004, task_id),
    };
    // N=CAP、H=CAP-1 只允许回收 producer 0，腾出的一个槽不足以
    // 发布同 task 的两个 entry。head/reclaim 可保留单调推进，但 task
    // payload、seq、tail 与 commit 必须完全未发布。
    EXPECT_EQ(
        try_commit(*map, task_id, entries, static_cast<int32_t>(kMapBucketCapacity - 1)), DriverResult::CapacityBlocked
    );
    EXPECT_EQ(map->reclaim_upto.v, 0);
    EXPECT_EQ(map->buckets[bucket].head.v, heads_before[bucket] + 1);
    EXPECT_EQ(map->buckets[bucket].tail.v, tail_before);
    EXPECT_EQ(map->committed_tasks.v, commit_before);
    EXPECT_EQ(std::memcmp(slots_before.data(), map->slots, slots_before.size()), 0);
    for (uint32_t index = 0; index < kMapBuckets; ++index) {
        if (index != bucket) {
            EXPECT_EQ(map->buckets[index].head.v, heads_before[index]);
        }
        EXPECT_EQ(map->buckets[index].tail.v, tails_before[index]);
    }
}

TEST(FdwicSharedTensorMapRing, FullBucketRetiresExactBatchAndReusesSameTaskSlots) {
    auto map = make_empty_map();
    const uint64_t address = 0x5a0000000ULL;
    const uint32_t bucket = dist_tensor_map_hash(address);
    for (uint32_t task = 0; task < kMapBucketCapacity; ++task) {
        ASSERT_EQ(
            try_commit(
                *map, static_cast<int32_t>(task),
                {make_region(address, task * 8, task * 8 + 4, static_cast<int32_t>(task))}
            ),
            DriverResult::Committed
        );
    }

    constexpr uint32_t kReuse = kMapBucketCapacity / 4U < 8U ? kMapBucketCapacity / 4U : 8U;
    static_assert(kReuse > 0, "exact reuse test requires a non-zero batch");
    const int32_t task_id = static_cast<int32_t>(kMapBucketCapacity);
    std::vector<SharedTensorMapValue> replacements;
    replacements.reserve(kReuse);
    for (uint32_t index = 0; index < kReuse; ++index) {
        const uint64_t lo = (1ULL << 20U) + static_cast<uint64_t>(index) * 32U;
        replacements.push_back(make_region(address, lo, lo + 8U, task_id));
    }
    const int32_t history = static_cast<int32_t>(kMapBucketCapacity - kReuse);
    ASSERT_EQ(try_commit(*map, task_id, replacements, history), DriverResult::Committed);
    EXPECT_EQ(map->reclaim_upto.v, static_cast<int64_t>(kReuse - 1));
    EXPECT_EQ(map->buckets[bucket].head.v, static_cast<int64_t>(kReuse));
    EXPECT_EQ(map->buckets[bucket].tail.v, static_cast<int64_t>(kMapBucketCapacity) + static_cast<int64_t>(kReuse));
    for (uint32_t index = 0; index < kReuse; ++index) {
        const uint64_t cursor = static_cast<uint64_t>(kMapBucketCapacity) + index;
        const SharedTensorMapSlot &reused = map->slots[dist_shared_tensor_map_slot_index(bucket, cursor)];
        EXPECT_EQ(reused.sequence.v, static_cast<int64_t>(cursor));
        EXPECT_EQ(reused.payload.value.buf_addr, replacements[index].buf_addr);
        EXPECT_EQ(reused.payload.value.lo, replacements[index].lo);
        EXPECT_EQ(reused.payload.value.hi, replacements[index].hi);
        EXPECT_EQ(reused.payload.value.producer, task_id);
        EXPECT_EQ(reused.payload.value.reserved, 0U);
    }
}

TEST(FdwicSharedTensorMapRing, ReverseArrivalEventuallyCommitsInTaskOrder) {
    auto map = make_empty_map();
    const SharedTensorMapValue task1 = make_region(0x5d0000000ULL, 0, 4, 1);
    const SharedTensorMapValue task2 = make_region(0x5e0000000ULL, 0, 4, 2);

    std::vector<std::byte> before(sizeof(*map));
    std::memcpy(before.data(), map.get(), before.size());
    EXPECT_EQ(try_commit(*map, 2, {task2}), DriverResult::Pending);
    EXPECT_EQ(std::memcmp(before.data(), map.get(), before.size()), 0);
    EXPECT_EQ(try_commit(*map, 1, {task1}), DriverResult::Pending);
    EXPECT_EQ(std::memcmp(before.data(), map.get(), before.size()), 0);
    EXPECT_EQ(try_commit(*map, 0, std::vector<SharedTensorMapValue>{}), DriverResult::Committed);

    std::memcpy(before.data(), map.get(), before.size());
    EXPECT_EQ(try_commit(*map, 0, std::vector<SharedTensorMapValue>{}), DriverResult::ProtocolError);
    EXPECT_EQ(std::memcmp(before.data(), map.get(), before.size()), 0);
    EXPECT_EQ(try_commit(*map, 2, {task2}), DriverResult::Pending);
    EXPECT_EQ(std::memcmp(before.data(), map.get(), before.size()), 0);
    EXPECT_EQ(try_commit(*map, 1, {task1}), DriverResult::Committed);
    EXPECT_EQ(try_commit(*map, 2, {task2}), DriverResult::Committed);
    EXPECT_EQ(map->committed_tasks.v, 3);
}

TEST(FdwicSharedTensorMapRing, ThreeLapsKeepAbsoluteSequenceAndLatestVersion) {
    auto map = make_empty_map();
    const uint64_t address = 0x600000000ULL;
    const uint32_t bucket = dist_tensor_map_hash(address);
    const int32_t tasks = static_cast<int32_t>(3 * kMapBucketCapacity + 5);
    for (int32_t task = 0; task < tasks; ++task) {
        ASSERT_EQ(try_commit(*map, task, {make_region(address, 0, 64, task)}, 0), DriverResult::Committed)
            << "task=" << task;
    }
    EXPECT_EQ(map->committed_tasks.v, tasks);
    EXPECT_EQ(map->buckets[bucket].head.v, tasks - 1);
    EXPECT_EQ(map->buckets[bucket].tail.v, tasks);
    const uint64_t last_cursor = static_cast<uint64_t>(tasks - 1);
    const SharedTensorMapSlot &last = map->slots[dist_shared_tensor_map_slot_index(bucket, last_cursor)];
    EXPECT_EQ(last.sequence.v, tasks - 1);
    EXPECT_EQ(last.payload.value.producer, tasks - 1);

    bool protocol_ok = false;
    EXPECT_EQ(
        dist_shared_tensor_map_lookup_region_impl<RecordingOps>(
            *map, make_region(address, 0, 64, -1), tasks, 1, protocol_ok
        ),
        tasks - 1
    );
    EXPECT_TRUE(protocol_ok);
}

TEST(FdwicSharedTensorMapRing, ProtocolErrorsRemainDistinctFromMissAndCapacity) {
    auto map = make_empty_map();
    const SharedTensorMapValue valid = make_region(0x700000000ULL, 0, 64, 0);
    SharedTensorMapValue wrong_producer = valid;
    wrong_producer.producer = 1;
    SharedTensorMapValue bad_reserved = valid;
    bad_reserved.reserved = 1;
    SharedTensorMapValue empty_range = valid;
    empty_range.hi = empty_range.lo;
    const int64_t reclaim = map->reclaim_upto.v;

    EXPECT_EQ(
        dist_shared_tensor_map_check_task_append_impl<RecordingOps>(*map, &wrong_producer, 1, 0, reclaim),
        DistSharedTensorMapAppendCheck::ProtocolError
    );
    EXPECT_EQ(
        dist_shared_tensor_map_check_task_append_impl<RecordingOps>(*map, &bad_reserved, 1, 0, reclaim),
        DistSharedTensorMapAppendCheck::ProtocolError
    );
    EXPECT_EQ(
        dist_shared_tensor_map_check_task_append_impl<RecordingOps>(*map, &empty_range, 1, 0, reclaim),
        DistSharedTensorMapAppendCheck::ProtocolError
    );
    EXPECT_EQ(
        dist_shared_tensor_map_check_task_append_impl<RecordingOps>(*map, nullptr, 1, 0, reclaim),
        DistSharedTensorMapAppendCheck::ProtocolError
    );
    EXPECT_EQ(
        dist_shared_tensor_map_check_task_append_impl<RecordingOps>(
            *map, &valid, static_cast<uint32_t>(MAX_TENSOR_ARGS + 1), 0, reclaim
        ),
        DistSharedTensorMapAppendCheck::ProtocolError
    );
    EXPECT_EQ(map->committed_tasks.v, 0);

    std::vector<std::byte> before(sizeof(*map));
    std::memcpy(before.data(), map.get(), before.size());
    EXPECT_EQ(try_commit(*map, kFlagCap, {make_region(valid.buf_addr, 0, 64, kFlagCap)}), DriverResult::ProtocolError);
    EXPECT_EQ(std::memcmp(before.data(), map.get(), before.size()), 0);

    map->committed_tasks.v = kFlagCap;
    std::memcpy(before.data(), map.get(), before.size());
    int64_t bounded_reclaim = -2;
    EXPECT_FALSE(dist_shared_tensor_map_refresh_reclaim_impl<RecordingOps>(*map, kFlagCap, 64, bounded_reclaim));
    EXPECT_EQ(
        dist_shared_tensor_map_check_task_append_impl<RecordingOps>(*map, &valid, 1, kFlagCap, map->reclaim_upto.v),
        DistSharedTensorMapAppendCheck::ProtocolError
    );
    EXPECT_EQ(std::memcmp(before.data(), map.get(), before.size()), 0);
    map->committed_tasks.v = 0;

    const uint32_t bucket = dist_tensor_map_hash(valid.buf_addr);
    map->buckets[bucket].head.v = INT64_MAX;
    map->buckets[bucket].tail.v = INT64_MAX;
    std::memcpy(before.data(), map.get(), before.size());
    EXPECT_EQ(
        dist_shared_tensor_map_check_task_append_impl<RecordingOps>(*map, &valid, 1, 0, reclaim),
        DistSharedTensorMapAppendCheck::ProtocolError
    );
    EXPECT_EQ(std::memcmp(before.data(), map.get(), before.size()), 0);
    EXPECT_FALSE(dist_shared_tensor_map_append_prepared_entry_impl<RecordingOps>(*map, valid, 0));
    EXPECT_EQ(std::memcmp(before.data(), map.get(), before.size()), 0);
    map->buckets[bucket].head.v = 0;
    map->buckets[bucket].tail.v = 0;

    SharedTensorMapSlot &slot = map->slots[dist_shared_tensor_map_slot_index(bucket, 0)];
    slot.sequence.v = kSharedTensorMapWritingSequence;
    SharedTensorMapValue writing_snapshot{};
    EXPECT_FALSE(dist_shared_tensor_map_read_slot_impl<RecordingOps>(*map, bucket, 0, writing_snapshot));
    EXPECT_EQ(
        dist_shared_tensor_map_check_task_append_impl<RecordingOps>(*map, &valid, 1, 0, reclaim),
        DistSharedTensorMapAppendCheck::ProtocolError
    );
    slot.sequence.v = 99;
    EXPECT_EQ(
        dist_shared_tensor_map_check_task_append_impl<RecordingOps>(*map, &valid, 1, 0, reclaim),
        DistSharedTensorMapAppendCheck::ProtocolError
    );
    slot.sequence.v = kSharedTensorMapInvalidSequence;

    ASSERT_EQ(try_commit(*map, 0, {valid}), DriverResult::Committed);
    slot.payload.value.reserved = 1;
    bool protocol_ok = true;
    EXPECT_EQ(
        dist_shared_tensor_map_lookup_region_impl<RecordingOps>(
            *map, make_region(valid.buf_addr, 0, 64, -1), 1, 1, protocol_ok
        ),
        -1
    );
    EXPECT_FALSE(protocol_ok);
}

struct ReferenceEntry {
    SharedTensorMapValue value;
};

int32_t reference_lookup(
    const std::vector<ReferenceEntry> &entries, const SharedTensorMapValue &query, int32_t current_task, int32_t history
) {
    const int32_t lower = current_task > history ? current_task - history : 0;
    int32_t best = -1;
    for (const ReferenceEntry &entry : entries) {
        if (entry.value.producer >= lower && entry.value.producer < current_task &&
            entry.value.buf_addr == query.buf_addr && query.lo < entry.value.hi && entry.value.lo < query.hi) {
            best = std::max(best, entry.value.producer);
        }
    }
    return best;
}

SharedTensorMapValue make_random_region(std::mt19937_64 &random, int32_t producer) {
    const uint64_t address = 0x800000000ULL + (random() % 48) * 0x100000ULL;
    const uint64_t lo = (random() % 96) * 4;
    const uint64_t bytes = (1 + random() % 12) * 4;
    return make_region(address, lo, lo + bytes, producer);
}

TEST(FdwicSharedTensorMapRing, FixedSeedTwelveThousandTasksMatchIndependentReference) {
    constexpr uint64_t kSeed = 0x53485244544d4150ULL;  // "SHRDTMAP"
    constexpr int32_t kHistory = 15;
    constexpr int32_t kTasks = 12000;
    auto map = make_empty_map();
    std::mt19937_64 random(kSeed);
    std::vector<ReferenceEntry> reference;
    reference.reserve(kTasks);

    for (int32_t task = 0; task < kTasks; ++task) {
        const SharedTensorMapValue query = make_random_region(random, -1);
        bool protocol_ok = false;
        const int32_t actual =
            dist_shared_tensor_map_lookup_region_impl<RecordingOps>(*map, query, task, kHistory, protocol_ok);
        ASSERT_TRUE(protocol_ok) << "task=" << task;
        EXPECT_EQ(actual, reference_lookup(reference, query, task, kHistory)) << "task=" << task;

        const SharedTensorMapValue inserted = make_random_region(random, task);
        ASSERT_EQ(try_commit(*map, task, {inserted}, kHistory), DriverResult::Committed) << "task=" << task;
        reference.push_back({inserted});

        const int32_t next_actual =
            dist_shared_tensor_map_lookup_region_impl<RecordingOps>(*map, inserted, task + 1, kHistory, protocol_ok);
        ASSERT_TRUE(protocol_ok) << "task=" << task;
        EXPECT_EQ(next_actual, reference_lookup(reference, inserted, task + 1, kHistory)) << "task=" << task;
    }
}

TEST(FdwicSharedTensorMapRing, ConcreteAicoreOpsRoundTripUsesTheSameStateMachine) {
    auto map = make_empty_map();
    const SharedTensorMapValue entry = make_region(0x900000000ULL, 16, 80, 0);
    int64_t reclaim_upto = -2;
    ASSERT_TRUE(dist_shared_tensor_map_refresh_reclaim(*map, 0, 64, reclaim_upto));
    ASSERT_EQ(
        dist_shared_tensor_map_check_task_append(*map, &entry, 1, 0, reclaim_upto),
        DistSharedTensorMapAppendCheck::Ready
    );
    ASSERT_TRUE(dist_shared_tensor_map_append_prepared_task(*map, &entry, 1, 0));
    ASSERT_TRUE(dist_shared_tensor_map_publish_commit(*map, 0));

    bool protocol_ok = false;
    EXPECT_EQ(
        dist_shared_tensor_map_lookup_region(*map, make_region(entry.buf_addr, 32, 48, -1), 1, 64, protocol_ok), 0
    );
    EXPECT_TRUE(protocol_ok);

    // 不只验证 concrete CAS 成功路径：若适配器误退化成 Exchange，这里会
    // 把竞争值 2 覆写成 desired 1，最终值断言会直接失败。
    map->committed_tasks.v = 2;
    EXPECT_FALSE(dist_shared_tensor_map_publish_commit(*map, 0));
    EXPECT_EQ(map->committed_tasks.v, 2);
}

}  // namespace
