/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>

#include "dist_engine/common/cross_core_dag_protocol.h"

namespace {

using namespace fdwic::cross_core;

struct HostOps {
    static inline uint32_t flush_count = 0;
    static inline uint32_t invalidate_count = 0;
    static inline uint64_t last_flush_bytes = 0;

    static void Reset() {
        flush_count = 0;
        invalidate_count = 0;
        last_flush_bytes = 0;
    }

    static int64_t Load(volatile int64_t *address) { return __atomic_load_n(address, __ATOMIC_ACQUIRE); }

    static int64_t CompareExchange(volatile int64_t *address, int64_t expected, int64_t desired) {
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(address, &observed, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
        return observed;
    }

    static void FlushRegion(const volatile void *, uint64_t bytes) {
        ++flush_count;
        last_flush_bytes = bytes;
        std::atomic_thread_fence(std::memory_order_release);
    }

    static void InvalidateRegion(const volatile void *, uint64_t) {
        ++invalidate_count;
        std::atomic_thread_fence(std::memory_order_acquire);
    }
};

CrossMapValue Region(uint64_t address, uint64_t lo, uint64_t hi, int32_t producer) {
    return CrossMapValue{address, lo, hi, producer, 0};
}

void ResetCells(DagTaskMetadataCell cells[], uint32_t count) {
    for (uint32_t task = 0; task < count; ++task) cells[task].control.state = 0;
    HostOps::Reset();
}

TEST(FdwicCrossCoreDagProtocol, AtomicControlNeverSharesPayloadCacheLine) {
    EXPECT_EQ(sizeof(DagWriterPayload), 1024U);
    EXPECT_EQ(offsetof(DagTaskMetadataCell, payload), 64U);
    EXPECT_EQ(sizeof(DagTaskMetadataCell), 1088U);
    EXPECT_EQ(alignof(DagTaskMetadataCell), 64U);
}

TEST(FdwicCrossCoreDagProtocol, PayloadIsFlushedBeforePublicationAndCanBeAcquired) {
    HostOps::Reset();
    DagTaskMetadataCell cell{};
    const CrossMapValue writers[] = {Region(0x1000, 0, 128, 7), Region(0x2000, 64, 256, 7)};

    EXPECT_EQ(PublishDagTaskMetadata<HostOps>(cell, 7, writers, 2), DagMetadataPublishResult::Published);
    EXPECT_EQ(HostOps::flush_count, 1U);
    EXPECT_EQ(HostOps::last_flush_bytes, 2U * sizeof(CrossMapValue));

    const DecodedDagMetadataControl decoded = DecodeDagMetadataControl(cell.control.state);
    ASSERT_TRUE(decoded.valid);
    ASSERT_TRUE(decoded.published);
    EXPECT_EQ(decoded.task_id, 7U);
    EXPECT_EQ(decoded.writer_count, 2U);

    uint32_t writer_count = 0;
    EXPECT_EQ(AcquireDagTaskMetadata<HostOps>(cell, 7, writer_count), DagMetadataAcquireResult::Acquired);
    EXPECT_EQ(writer_count, 2U);
    EXPECT_EQ(HostOps::invalidate_count, 1U);
    EXPECT_EQ(cell.payload.writers[1].buffer_address, 0x2000U);
}

TEST(FdwicCrossCoreDagProtocol, EmptyWriterSetPublishesWithoutPayloadDcci) {
    HostOps::Reset();
    DagTaskMetadataCell cell{};
    EXPECT_EQ(PublishDagTaskMetadata<HostOps>(cell, 3, nullptr, 0), DagMetadataPublishResult::Published);
    EXPECT_EQ(HostOps::flush_count, 0U);
    uint32_t writer_count = 99;
    EXPECT_EQ(AcquireDagTaskMetadata<HostOps>(cell, 3, writer_count), DagMetadataAcquireResult::Acquired);
    EXPECT_EQ(writer_count, 0U);
    EXPECT_EQ(HostOps::invalidate_count, 0U);
}

TEST(FdwicCrossCoreDagProtocol, DuplicateOrMalformedPublicationFailsClosed) {
    HostOps::Reset();
    DagTaskMetadataCell cell{};
    const CrossMapValue valid = Region(0x1000, 0, 64, 2);
    const CrossMapValue wrong_owner = Region(0x1000, 0, 64, 1);
    EXPECT_EQ(PublishDagTaskMetadata<HostOps>(cell, 2, &wrong_owner, 1), DagMetadataPublishResult::InvalidInput);
    EXPECT_EQ(PublishDagTaskMetadata<HostOps>(cell, 2, &valid, 1), DagMetadataPublishResult::Published);
    EXPECT_EQ(PublishDagTaskMetadata<HostOps>(cell, 2, &valid, 1), DagMetadataPublishResult::AlreadyPublished);

    uint32_t writer_count = 0;
    EXPECT_EQ(AcquireDagTaskMetadata<HostOps>(cell, 1, writer_count), DagMetadataAcquireResult::InvalidControl);
}

TEST(FdwicCrossCoreDagProtocol, LogicalTaskOrderSelectsLatestOverlappingWriter) {
    constexpr uint32_t kCount = 8;
    auto cells = std::make_unique<DagTaskMetadataCell[]>(kCount);
    ResetCells(cells.get(), kCount);
    for (uint32_t task = 0; task < 6; ++task) {
        CrossMapValue writer = task == 1 ? Region(0x3000, 0, 128, 1) :
                               task == 4 ? Region(0x3000, 64, 192, 4) :
                                           Region(0x9000 + task * 0x100, 0, 64, static_cast<int32_t>(task));
        ASSERT_EQ(
            PublishDagTaskMetadata<HostOps>(cells[task], task, &writer, 1), DagMetadataPublishResult::Published
        );
    }

    int32_t producer = -1;
    EXPECT_EQ(
        FindLatestDagWriter<HostOps>(cells.get(), kCount, Region(0x3000, 96, 112, -1), 6, 6, producer),
        DagWriterLookupResult::Found
    );
    EXPECT_EQ(producer, 4);

    EXPECT_EQ(
        FindLatestDagWriter<HostOps>(cells.get(), kCount, Region(0x3000, 96, 112, -1), 4, 2, producer),
        DagWriterLookupResult::None
    );
    EXPECT_EQ(producer, -1);
}

TEST(FdwicCrossCoreDagProtocol, MissingEarlierSchemaCannotBeSkipped) {
    constexpr uint32_t kCount = 5;
    auto cells = std::make_unique<DagTaskMetadataCell[]>(kCount);
    ResetCells(cells.get(), kCount);
    const CrossMapValue task_zero = Region(0x4000, 0, 64, 0);
    const CrossMapValue task_two = Region(0x5000, 0, 64, 2);
    ASSERT_EQ(PublishDagTaskMetadata<HostOps>(cells[0], 0, &task_zero, 1), DagMetadataPublishResult::Published);
    ASSERT_EQ(PublishDagTaskMetadata<HostOps>(cells[2], 2, &task_two, 1), DagMetadataPublishResult::Published);

    int32_t producer = -1;
    EXPECT_EQ(
        FindLatestDagWriter<HostOps>(cells.get(), kCount, Region(0x4000, 0, 32, -1), 3, 3, producer),
        DagWriterLookupResult::Pending
    );
    EXPECT_EQ(producer, -1);

    ASSERT_EQ(PublishDagTaskMetadata<HostOps>(cells[1], 1, nullptr, 0), DagMetadataPublishResult::Published);
    EXPECT_EQ(
        FindLatestDagWriter<HostOps>(cells.get(), kCount, Region(0x4000, 0, 32, -1), 3, 3, producer),
        DagWriterLookupResult::Found
    );
    EXPECT_EQ(producer, 0);
}

}  // namespace
