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

#include <atomic>
#include <cstdint>
#include <memory>

#include "dist_engine/common/cross_core_tensor_map_protocol.h"

namespace {

using namespace fdwic::cross_core;

struct HostOps {
    static inline uint32_t flush_count = 0;
    static inline uint32_t invalidate_count = 0;

    static void ResetCounters() {
        flush_count = 0;
        invalidate_count = 0;
    }

    static int64_t Load(volatile int64_t *address) { return __atomic_load_n(address, __ATOMIC_ACQUIRE); }

    static int64_t CompareExchange(volatile int64_t *address, int64_t expected, int64_t desired) {
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(address, &observed, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
        return observed;
    }

    static void FlushRegion(const volatile void *, uint64_t) {
        ++flush_count;
        std::atomic_thread_fence(std::memory_order_release);
    }

    static void InvalidateRegion(const volatile void *, uint64_t) {
        ++invalidate_count;
        std::atomic_thread_fence(std::memory_order_acquire);
    }
};

void ResetMap(CrossCoreTensorMapState &map) {
    for (uint32_t bucket = 0; bucket < kCrossMapBuckets; ++bucket)
        map.tails[bucket].state = 0;
    for (uint32_t slot = 0; slot < kCrossMapCapacity; ++slot)
        map.slots[slot].sequence.state = -1;
    HostOps::ResetCounters();
}

CrossMapValue MakeRegion(uint64_t address, uint64_t lo, uint64_t hi, int32_t producer) {
    return CrossMapValue{address, lo, hi, producer, 0};
}

TEST(FdwicCrossCoreTensorMapProtocol, LayoutSeparatesPayloadSequenceAndBucketControls) {
    EXPECT_EQ(sizeof(CrossMapValue), 32U);
    EXPECT_EQ(sizeof(CrossMapPayload), 64U);
    EXPECT_EQ(sizeof(CrossMapSlot), 128U);
    EXPECT_EQ(offsetof(CrossMapSlot, sequence), 64U);
    EXPECT_EQ(offsetof(CrossCoreTensorMapState, slots), 64U * kCrossMapBuckets);
    EXPECT_EQ(sizeof(CrossCoreTensorMapState), 64U * kCrossMapBuckets + 128U * kCrossMapCapacity);
}

TEST(FdwicCrossCoreTensorMapProtocol, StrictTaskTurnCannotSkipAnUnpublishedPredecessor) {
    volatile int64_t predecessor = -1;
    volatile int64_t completion = -1;

    EXPECT_EQ(InspectMapTaskTurn<HostOps>(predecessor, 7), MapTurnResult::Pending);
    predecessor = 6;
    EXPECT_EQ(InspectMapTaskTurn<HostOps>(predecessor, 7), MapTurnResult::Ready);
    EXPECT_TRUE(PublishMapTaskCompletion<HostOps>(completion, 7));
    EXPECT_EQ(completion, 7);
    EXPECT_FALSE(PublishMapTaskCompletion<HostOps>(completion, 7));
    predecessor = 8;
    EXPECT_EQ(InspectMapTaskTurn<HostOps>(predecessor, 7), MapTurnResult::Invalid);
}

TEST(FdwicCrossCoreTensorMapProtocol, OrderedAppendPublishesPayloadBeforeLookup) {
    auto map = std::make_unique<CrossCoreTensorMapState>();
    ResetMap(*map);
    const uint64_t address = 0x100000;
    const CrossMapValue entries[] = {
        MakeRegion(address, 0, 1024, 3),
        MakeRegion(address, 2048, 4096, 3),
    };

    EXPECT_EQ(AppendCrossMapTask<HostOps>(*map, entries, 2, 3), MapAppendResult::Appended);
    EXPECT_EQ(HostOps::flush_count, 2U);
    EXPECT_EQ(map->tails[CrossMapHash(address)].state, 2);

    bool protocol_ok = false;
    EXPECT_EQ(LookupCrossMap<HostOps>(*map, MakeRegion(address, 512, 768, -1), 4, 64, protocol_ok), 3);
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(HostOps::invalidate_count, 2U);
}

TEST(FdwicCrossCoreTensorMapProtocol, LookupRejectsFutureProducerAndHonorsHistoryWindow) {
    auto map = std::make_unique<CrossCoreTensorMapState>();
    ResetMap(*map);
    const uint64_t address = 0x200000;
    const CrossMapValue old_entry = MakeRegion(address, 0, 256, 1);
    const CrossMapValue future_entry = MakeRegion(address, 0, 256, 9);
    ASSERT_EQ(AppendCrossMapTask<HostOps>(*map, &old_entry, 1, 1), MapAppendResult::Appended);
    ASSERT_EQ(AppendCrossMapTask<HostOps>(*map, &future_entry, 1, 9), MapAppendResult::Appended);

    bool protocol_ok = false;
    EXPECT_EQ(LookupCrossMap<HostOps>(*map, MakeRegion(address, 0, 128, -1), 9, 64, protocol_ok), 1);
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(LookupCrossMap<HostOps>(*map, MakeRegion(address, 0, 128, -1), 9, 4, protocol_ok), -1);
    EXPECT_TRUE(protocol_ok);
}

TEST(FdwicCrossCoreTensorMapProtocol, FullBucketFailsBeforeOverwritingPublishedPayload) {
    auto map = std::make_unique<CrossCoreTensorMapState>();
    ResetMap(*map);
    const uint64_t address = 0x300000;
    for (uint32_t cursor = 0; cursor < kCrossMapBucketCapacity; ++cursor) {
        const CrossMapValue entry = MakeRegion(address, cursor * 64U, cursor * 64U + 32U, cursor);
        ASSERT_EQ(
            AppendCrossMapTask<HostOps>(*map, &entry, 1, static_cast<int32_t>(cursor)), MapAppendResult::Appended
        );
    }
    const uint32_t flushes_before = HostOps::flush_count;
    const CrossMapValue overflow = MakeRegion(address, 0, 16, static_cast<int32_t>(kCrossMapBucketCapacity));
    EXPECT_EQ(
        AppendCrossMapTask<HostOps>(*map, &overflow, 1, static_cast<int32_t>(kCrossMapBucketCapacity)),
        MapAppendResult::CapacityExceeded
    );
    EXPECT_EQ(HostOps::flush_count, flushes_before);
    EXPECT_EQ(map->tails[CrossMapHash(address)].state, static_cast<int64_t>(kCrossMapBucketCapacity));
}

}  // namespace
