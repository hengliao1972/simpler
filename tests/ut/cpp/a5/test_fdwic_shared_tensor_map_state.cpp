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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "dist_engine/aicpu/shared_tensor_map_init.h"

namespace {

constexpr uint8_t kPayloadPattern = 0xa5;

bool shared_payloads_keep_pattern(const SharedTensorMapState &state) {
    for (int32_t slot = 0; slot < kMapCap; ++slot) {
        const auto *bytes = reinterpret_cast<const uint8_t *>(&state.slots[slot].payload);
        for (size_t offset = 0; offset < sizeof(SharedTensorMapPayloadLine); ++offset) {
            if (bytes[offset] != kPayloadPattern) return false;
        }
    }
    return true;
}

TEST(FdwicSharedTensorMapState, SidecarLayoutAppendsWithoutMovingThePrivateTail) {
    constexpr size_t kExpectedStateBytes = 2 * kCacheLine + kMapBuckets * 2 * kCacheLine + kMapCap * 2 * kCacheLine;

    EXPECT_EQ(sizeof(SharedTensorMapValue), 32U);
    EXPECT_EQ(offsetof(SharedTensorMapValue, buf_addr), 0U);
    EXPECT_EQ(offsetof(SharedTensorMapValue, lo), 8U);
    EXPECT_EQ(offsetof(SharedTensorMapValue, hi), 16U);
    EXPECT_EQ(offsetof(SharedTensorMapValue, producer), 24U);
    EXPECT_EQ(offsetof(SharedTensorMapValue, reserved), 28U);
    EXPECT_EQ(sizeof(SharedTensorMapPayloadLine), kCacheLine);
    EXPECT_EQ(sizeof(SharedTensorMapSequenceLine), kCacheLine);
    EXPECT_EQ(offsetof(SharedTensorMapSequenceLine, v), 0U);
    EXPECT_EQ(sizeof(SharedTensorMapSlot), 2 * kCacheLine);
    EXPECT_LT(kSharedTensorMapWritingSequence, kSharedTensorMapInvalidSequence);
    EXPECT_NE(kSharedTensorMapWritingSequence, kSharedTensorMapInvalidSequence);
    EXPECT_EQ(sizeof(SharedTensorMapBucketState), 2 * kCacheLine);
    EXPECT_EQ(sizeof(SharedTensorMapState), kExpectedStateBytes);
    EXPECT_EQ(alignof(SharedTensorMapState), kCacheLine);
    EXPECT_EQ(offsetof(SharedTensorMapState, committed_tasks), 0U);
    EXPECT_EQ(offsetof(SharedTensorMapState, reclaim_upto), kCacheLine);
    EXPECT_EQ(offsetof(SharedTensorMapState, buckets), 2 * kCacheLine);
    EXPECT_EQ(offsetof(SharedTensorMapState, slots), 2 * kCacheLine + sizeof(SharedTensorMapBucketState) * kMapBuckets);
    EXPECT_EQ(offsetof(SharedTensorMapBucketState, head), 0U);
    EXPECT_EQ(offsetof(SharedTensorMapBucketState, tail), kCacheLine);
#if PTO_FDWIC_TENSORMAP_RING_CAP == 128
    EXPECT_EQ(offsetof(SharedTensorMapState, buckets), 128U);
    EXPECT_EQ(offsetof(SharedTensorMapState, slots), 16512U);
    EXPECT_EQ(sizeof(SharedTensorMapState), 2113664U);
#endif

#if PTO_FDWIC_SHARED_MAP
    EXPECT_EQ(offsetof(DistGlobal, shared_tensor_map), kFdwicSharedTensorMapOffset);
    EXPECT_EQ(sizeof(DistGlobal), kFdwicSharedTensorMapOffset + sizeof(SharedTensorMapState));
#else
    EXPECT_EQ(sizeof(DistGlobal), kFdwicSharedTensorMapOffset);
#endif
    EXPECT_LE(sizeof(DistGlobal), kDistEngineGlobalStateSize);
}

#if PTO_FDWIC_SHARED_MAP
TEST(FdwicSharedTensorMapState, AicpuResetInitializesOnlyPublicationAndCursorState) {
    auto state = std::make_unique<SharedTensorMapState>();
    std::memset(state.get(), kPayloadPattern, sizeof(*state));

    dist_shared_tensor_map_reset(*state);

    EXPECT_EQ(state->committed_tasks.v, kSharedTensorMapInitialCommit);
    EXPECT_EQ(state->reclaim_upto.v, kSharedTensorMapInitialReclaim);
    for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
        EXPECT_EQ(state->buckets[bucket].head.v, 0) << "bucket=" << bucket;
        EXPECT_EQ(state->buckets[bucket].tail.v, 0) << "bucket=" << bucket;
    }
    for (int32_t slot = 0; slot < kMapCap; ++slot) {
        EXPECT_EQ(state->slots[slot].sequence.v, kSharedTensorMapInvalidSequence) << "slot=" << slot;
    }
    EXPECT_TRUE(shared_payloads_keep_pattern(*state));

    // 同一 arena 重复运行时必须清掉上一轮的绝对游标和发布 seq。
    state->committed_tasks.v = 37;
    state->reclaim_upto.v = 21;
    state->buckets[kMapBuckets - 1].head.v = 8;
    state->buckets[kMapBuckets - 1].tail.v = 10;
    state->slots[kMapCap - 2].sequence.v = 16382;
    state->slots[kMapCap - 1].sequence.v = kSharedTensorMapWritingSequence;

    dist_shared_tensor_map_reset(*state);

    EXPECT_EQ(state->committed_tasks.v, kSharedTensorMapInitialCommit);
    EXPECT_EQ(state->reclaim_upto.v, kSharedTensorMapInitialReclaim);
    EXPECT_EQ(state->buckets[kMapBuckets - 1].head.v, 0);
    EXPECT_EQ(state->buckets[kMapBuckets - 1].tail.v, 0);
    EXPECT_EQ(state->slots[kMapCap - 2].sequence.v, kSharedTensorMapInvalidSequence);
    EXPECT_EQ(state->slots[kMapCap - 1].sequence.v, kSharedTensorMapInvalidSequence);
    EXPECT_TRUE(shared_payloads_keep_pattern(*state));
}
#endif

}  // namespace
