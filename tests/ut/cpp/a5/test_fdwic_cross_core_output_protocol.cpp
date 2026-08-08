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

#include "dist_engine/common/cross_core_output_protocol.h"

namespace {

using namespace fdwic::cross_core;

struct alignas(64) TestDescriptor {
    uint64_t words[kExecTensorDescWords];
};

struct HostOps {
    static inline uint32_t flush_count = 0;
    static inline uint32_t invalidate_count = 0;
    static inline uint64_t last_bytes = 0;

    static void Reset() {
        flush_count = 0;
        invalidate_count = 0;
        last_bytes = 0;
    }

    static int64_t Load(volatile int64_t *address) { return __atomic_load_n(address, __ATOMIC_ACQUIRE); }

    static int64_t CompareExchange(volatile int64_t *address, int64_t expected, int64_t desired) {
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(address, &observed, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
        return observed;
    }

    static int64_t FetchAdd(volatile int64_t *address, int64_t delta) {
        return __atomic_fetch_add(address, delta, __ATOMIC_ACQ_REL);
    }

    static void FlushRegion(const volatile void *, uint64_t bytes) {
        ++flush_count;
        last_bytes = bytes;
        __atomic_thread_fence(__ATOMIC_RELEASE);
    }

    static void InvalidateRegion(const volatile void *, uint64_t bytes) {
        ++invalidate_count;
        last_bytes = bytes;
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
    }
};

static_assert(sizeof(TestDescriptor) == kExecTensorDescBytes);

TEST(FdwicCrossCoreOutputProtocol, AtomicControlAndDescriptorsDoNotShareACacheLine) {
    EXPECT_EQ(sizeof(SharedExecControl), 64U);
    EXPECT_EQ(offsetof(CrossCoreOutputCell<TestDescriptor>, descriptors), 64U);
    EXPECT_EQ(alignof(CrossCoreOutputCell<TestDescriptor>), 64U);
    EXPECT_EQ(sizeof(CrossCoreOutputCell<TestDescriptor>), 64U + 32U * 128U);
}

TEST(FdwicCrossCoreOutputProtocol, PublisherFlushesDescriptorsBeforeConsumersAcquire) {
    CrossCoreOutputCell<TestDescriptor> cell{};
    SharedExecControl fatal{};
    HostOps::Reset();
    cell.descriptors[0].words[0] = 0x1234;
    cell.descriptors[1].words[0] = 0x5678;

    EXPECT_EQ(PublishTaskOutputs<HostOps>(cell, 17, 5, 2, fatal), OutputPublishResult::Published);
    EXPECT_EQ(HostOps::flush_count, 1U);
    EXPECT_EQ(HostOps::last_bytes, 2U * sizeof(TestDescriptor));
    const DecodedOutputState published = DecodeOutputState(cell.control.state);
    ASSERT_TRUE(published.valid);
    EXPECT_EQ(published.phase, OutputPhase::Published);
    EXPECT_EQ(published.task_id, 17U);
    EXPECT_EQ(published.build_owner, 5U);
    EXPECT_EQ(published.output_count, 2U);

    EXPECT_EQ(AcquireTaskOutputs<HostOps>(cell, 17, 2, fatal), OutputAcquireResult::Acquired);
    EXPECT_EQ(HostOps::invalidate_count, 1U);
    EXPECT_EQ(HostOps::last_bytes, 2U * sizeof(TestDescriptor));
    EXPECT_EQ(cell.descriptors[0].words[0], 0x1234U);
    EXPECT_EQ(cell.descriptors[1].words[0], 0x5678U);
}

TEST(FdwicCrossCoreOutputProtocol, EmptyOutputListPublishesWithoutPayloadDcci) {
    CrossCoreOutputCell<TestDescriptor> cell{};
    SharedExecControl fatal{};
    HostOps::Reset();

    EXPECT_EQ(PublishTaskOutputs<HostOps>(cell, 3, 2, 0, fatal), OutputPublishResult::Published);
    EXPECT_EQ(HostOps::flush_count, 0U);
    EXPECT_EQ(AcquireTaskOutputs<HostOps>(cell, 3, 0, fatal), OutputAcquireResult::Acquired);
    EXPECT_EQ(HostOps::invalidate_count, 0U);
}

TEST(FdwicCrossCoreOutputProtocol, WrongTaskOrCountCannotAliasAPublishedCell) {
    CrossCoreOutputCell<TestDescriptor> cell{};
    SharedExecControl fatal{};
    HostOps::Reset();
    ASSERT_EQ(PublishTaskOutputs<HostOps>(cell, 11, 9, 1, fatal), OutputPublishResult::Published);

    EXPECT_EQ(AcquireTaskOutputs<HostOps>(cell, 10, 1, fatal), OutputAcquireResult::InvalidControl);
    EXPECT_EQ(AcquireTaskOutputs<HostOps>(cell, 11, 2, fatal), OutputAcquireResult::InvalidControl);
    EXPECT_EQ(HostOps::invalidate_count, 0U);
}

TEST(FdwicCrossCoreOutputProtocol, DuplicatePublicationFailsClosed) {
    CrossCoreOutputCell<TestDescriptor> cell{};
    SharedExecControl fatal{};
    HostOps::Reset();
    ASSERT_EQ(PublishTaskOutputs<HostOps>(cell, 7, 4, 1, fatal), OutputPublishResult::Published);
    EXPECT_EQ(PublishTaskOutputs<HostOps>(cell, 7, 4, 1, fatal), OutputPublishResult::PublishConflict);
    EXPECT_NE(fatal.state, 0);
}

TEST(FdwicCrossCoreOutputProtocol, HeapReservationsAreDisjointAndBounded) {
    SharedExecControl cursor{};
    SharedExecControl fatal{};
    HeapReservation first{};
    HeapReservation second{};
    HostOps::Reset();

    EXPECT_EQ(ReserveOutputHeap<HostOps>(cursor, 1, 3, 1024, 4096, first, fatal), HeapReserveResult::Reserved);
    EXPECT_EQ(ReserveOutputHeap<HostOps>(cursor, 2, 7, 2048, 4096, second, fatal), HeapReserveResult::Reserved);
    EXPECT_EQ(first.begin, 0U);
    EXPECT_EQ(first.end, 1024U);
    EXPECT_EQ(second.begin, 1024U);
    EXPECT_EQ(second.end, 3072U);
    EXPECT_EQ(cursor.state, 3072);

    HeapReservation empty{};
    EXPECT_EQ(ReserveOutputHeap<HostOps>(cursor, 3, 9, 0, 4096, empty, fatal), HeapReserveResult::Empty);
    EXPECT_EQ(empty.begin, 3072U);
    EXPECT_EQ(empty.end, 3072U);
}

TEST(FdwicCrossCoreOutputProtocol, HeapOverflowIsTerminalAfterTheNonRollbackReservation) {
    SharedExecControl cursor{};
    SharedExecControl fatal{};
    HeapReservation reservation{};
    HostOps::Reset();

    EXPECT_EQ(ReserveOutputHeap<HostOps>(cursor, 5, 2, 5120, 4096, reservation, fatal), HeapReserveResult::Exhausted);
    EXPECT_EQ(cursor.state, 5120);
    EXPECT_NE(fatal.state, 0);
}

}  // namespace
