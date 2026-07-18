/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE file.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>

#include <cstdint>

#include "inner_kernel.h"
#include "runtime.h"
#include "dist_engine/common/swimlane.h"

namespace {

class FdwicPollBatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        core_ = {};
        records_[0] = {};
        records_[1] = {};
        g_self = reinterpret_cast<DistCore *>(uintptr_t{1});
        g_fdwic_swimlane_level = kFdwicAtomicSwimlaneLevel;
        g_fdwic_swimlane_core = &core_;
        g_fdwic_swimlane_records = records_;
        g_fdwic_swimlane_records_per_core = 2;
        g_fdwic_atomic_poll_burst = {};
        g_fdwic_atomic_calls = 0;
        g_fdwic_poll_calls = 0;
        g_fdwic_poll_batch_records = 0;
        g_fdwic_atomic_counter_overflow = false;
    }

    void TearDown() override {
        g_self = nullptr;
        g_fdwic_swimlane_level = 0;
        g_fdwic_swimlane_core = nullptr;
        g_fdwic_swimlane_records = nullptr;
        g_fdwic_swimlane_records_per_core = 0;
        g_fdwic_atomic_poll_burst = {};
    }

    FdwicSwimlaneCoreState core_{};
    FdwicSwimlaneRecord records_[2]{};
};

TEST_F(FdwicPollBatchTest, SplitsAtMaximum24BitCountAndReopensExactly) {
    constexpr FdwicAtomicSite kSite = FdwicAtomicSite::StartupPoll;
    constexpr uint32_t kBatchIndex = 0;
    constexpr uint32_t kBatchBit = 1U << kBatchIndex;
    constexpr uint64_t kFirstStart = 111;
    constexpr uint64_t kSecondStart = 222;
    constexpr uint64_t kSecondEnd = 333;

    ASSERT_EQ(fdwic_atomic_poll_batch_index(kSite), static_cast<int32_t>(kBatchIndex));
    g_fdwic_atomic_poll_burst.active_mask = kBatchBit;
    g_fdwic_atomic_poll_burst.start_cycle[kBatchIndex] = kFirstStart;
    g_fdwic_atomic_poll_burst.call_count[kBatchIndex] = kFdwicAtomicPollCountMax - 1;

    // The maximum-th call belongs to the first row and triggers an immediate
    // flush. It must not be dropped or carried into the next batch.
    fdwic_swimlane_accumulate_poll_call(kSite, kFirstStart);

    ASSERT_EQ(core_.count, 1U);
    EXPECT_EQ(g_fdwic_poll_batch_records, 1U);
    EXPECT_EQ(g_fdwic_atomic_poll_burst.active_mask, 0U);
    EXPECT_EQ(g_fdwic_atomic_poll_burst.call_count[kBatchIndex], 0U);
    EXPECT_EQ(records_[0].phase, static_cast<uint16_t>(FdwicSwimlanePhase::Atomic));
    EXPECT_EQ(records_[0].aux, static_cast<uint16_t>(kSite));
    EXPECT_EQ(records_[0].start_cycle, kFirstStart);
    EXPECT_GE(records_[0].end_cycle, records_[0].start_cycle);
    EXPECT_NE(records_[0].flags & kFdwicAtomicPollBatch, 0U);
    EXPECT_EQ(records_[0].flags >> kFdwicAtomicPollCountShift, kFdwicAtomicPollCountMax);

    // The next call starts a fresh row at count one. Closing that row must
    // preserve max+1 calls exactly across the two encoded records.
    fdwic_swimlane_accumulate_poll_call(kSite, kSecondStart);
    ASSERT_EQ(g_fdwic_atomic_poll_burst.active_mask, kBatchBit);
    ASSERT_EQ(g_fdwic_atomic_poll_burst.call_count[kBatchIndex], 1U);
    fdwic_atomic_poll_boundary_at(kSecondEnd);

    ASSERT_EQ(core_.count, 2U);
    EXPECT_EQ(g_fdwic_poll_batch_records, 2U);
    EXPECT_EQ(g_fdwic_atomic_poll_burst.active_mask, 0U);
    EXPECT_EQ(g_fdwic_atomic_poll_burst.call_count[kBatchIndex], 0U);
    EXPECT_EQ(records_[1].start_cycle, kSecondStart);
    EXPECT_EQ(records_[1].end_cycle, kSecondEnd);
    EXPECT_EQ(records_[1].flags >> kFdwicAtomicPollCountShift, 1U);
    const uint64_t represented_calls =
        (records_[0].flags >> kFdwicAtomicPollCountShift) + (records_[1].flags >> kFdwicAtomicPollCountShift);
    EXPECT_EQ(represented_calls, static_cast<uint64_t>(kFdwicAtomicPollCountMax) + 1);
    EXPECT_EQ(core_.dropped, 0U);
    EXPECT_FALSE(g_fdwic_atomic_counter_overflow);
}

}  // namespace
