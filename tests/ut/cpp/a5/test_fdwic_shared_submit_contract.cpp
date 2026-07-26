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

#include <memory>
#include <stdexcept>

#define PTO_FDWIC_TRACE_ENABLED 0
#include "dist_engine/aicore/dist_engine.cpp"  // NOLINT(build/include)

[[noreturn]] void assert_impl(const char *condition, const char *, int) { throw std::logic_error(condition); }

extern "C" void aicpu_orchestration_entry(const L2TaskArgs &) {}
volatile uint8_t *sim_get_reg_base() { return nullptr; }
uint32_t sim_get_physical_core_id() { return 0; }

namespace {

class FdwicSharedSubmitContractTest : public ::testing::Test {
protected:
    void SetUp() override {
        static_assert(PTO_FDWIC_SHARED_MAP == 1);
        static_assert(!kFdwicCompiledBackendReady);
        g_dist_ptr = &g_dist_fallback;
        g_self = worker_.get();
        dist_core_reset(*worker_, CoreType::AIV, /*block=*/0, LANE_AIV0);
        worker_->core_idx = 0;
        g_dist.fatal = 0;
        g_dist.error_code = PTO2_ERROR_NONE;
    }

    void TearDown() override {
        g_self = nullptr;
        g_dist_ptr = nullptr;
    }

    DistSubmitCtx make_context() {
        DistSubmitCtx ctx{};
        ctx.self = worker_.get();
        ctx.task_id = 7;
        return ctx;
    }

    void seed_data_plane_markers(const DistSubmitCtx &ctx) {
        g_dist.shared_tensor_map.committed_tasks.v = 19;
        g_dist.shared_tensor_map.reclaim_upto.v = 11;
        g_dist.shared_tensor_map.buckets[0].head.v = 3;
        g_dist.shared_tensor_map.buckets[0].tail.v = 5;
        g_dist.shared_tensor_map.slots[0].payload.value = SharedTensorMapValue{0x1000, 8, 32, 4, 0};
        g_dist.shared_tensor_map.slots[0].sequence.v = 17;
        task_cell(ctx.task_id).flag = 23;
        task_cell(ctx.task_id).vend = 29;
        worker_->occupied_count = 31;
        worker_->owned_total = 37;
    }

    void expect_data_plane_markers_unchanged(const DistSubmitCtx &ctx) {
        EXPECT_EQ(g_dist.shared_tensor_map.committed_tasks.v, 19);
        EXPECT_EQ(g_dist.shared_tensor_map.reclaim_upto.v, 11);
        EXPECT_EQ(g_dist.shared_tensor_map.buckets[0].head.v, 3);
        EXPECT_EQ(g_dist.shared_tensor_map.buckets[0].tail.v, 5);
        const SharedTensorMapValue &value = g_dist.shared_tensor_map.slots[0].payload.value;
        EXPECT_EQ(value.buf_addr, 0x1000U);
        EXPECT_EQ(value.lo, 8U);
        EXPECT_EQ(value.hi, 32U);
        EXPECT_EQ(value.producer, 4);
        EXPECT_EQ(value.reserved, 0U);
        EXPECT_EQ(g_dist.shared_tensor_map.slots[0].sequence.v, 17);
        EXPECT_EQ(task_cell(ctx.task_id).flag, 23);
        EXPECT_EQ(task_cell(ctx.task_id).vend, 29U);
        EXPECT_EQ(worker_->occupied_count, 31);
        EXPECT_EQ(worker_->owned_total, 37);
    }

    std::unique_ptr<DistCore> worker_ = std::make_unique<DistCore>();
};

TEST_F(FdwicSharedSubmitContractTest, CommittedResultDoesNotLatchFailure) {
    DistSubmitCtx ctx = make_context();

    EXPECT_EQ(
        dist_submit_shared_tensor_map_error_code(DistSharedTensorMapTaskPublishResult::Committed), PTO2_ERROR_NONE
    );
    EXPECT_TRUE(dist_submit_handle_shared_tensor_map_result(ctx, DistSharedTensorMapTaskPublishResult::Committed));
    EXPECT_EQ(worker_->local_index, 0);
    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_NONE);
}

struct SharedPublishFailureCase {
    DistSharedTensorMapTaskPublishResult result;
    int32_t error_code;
};

TEST_F(FdwicSharedSubmitContractTest, EveryFailureClassLatchesItsStructuredCode) {
    const SharedPublishFailureCase cases[] = {
        {DistSharedTensorMapTaskPublishResult::CapacityBlocked, PTO2_ERROR_TENSORMAP_CAPACITY},
        {DistSharedTensorMapTaskPublishResult::ProtocolError, PTO2_ERROR_TENSORMAP_PROTOCOL},
        {DistSharedTensorMapTaskPublishResult::PartialPublish, PTO2_ERROR_TENSORMAP_PARTIAL_PUBLISH},
        {static_cast<DistSharedTensorMapTaskPublishResult>(UINT32_MAX), PTO2_ERROR_TENSORMAP_PROTOCOL},
    };

    for (const SharedPublishFailureCase &test_case : cases) {
        g_dist.fatal = 0;
        g_dist.error_code = PTO2_ERROR_NONE;
        worker_->local_index = 8;
        DistSubmitCtx ctx = make_context();
        seed_data_plane_markers(ctx);

        EXPECT_FALSE(dist_submit_handle_shared_tensor_map_result(ctx, test_case.result));
        EXPECT_EQ(worker_->local_index, kFlagCap);
        EXPECT_EQ(g_dist.fatal, 1);
        EXPECT_EQ(g_dist.error_code, test_case.error_code);
        expect_data_plane_markers_unchanged(ctx);
    }
}

TEST_F(FdwicSharedSubmitContractTest, PreexistingFailureCodeRemainsAuthoritativeForEverySharedFailure) {
    const DistSharedTensorMapTaskPublishResult cases[] = {
        DistSharedTensorMapTaskPublishResult::CapacityBlocked,
        DistSharedTensorMapTaskPublishResult::ProtocolError,
        DistSharedTensorMapTaskPublishResult::PartialPublish,
        static_cast<DistSharedTensorMapTaskPublishResult>(UINT32_MAX),
    };

    for (DistSharedTensorMapTaskPublishResult result : cases) {
        g_dist.fatal = 1;
        g_dist.error_code = PTO2_ERROR_INVALID_ARGS;
        worker_->local_index = 8;
        DistSubmitCtx ctx = make_context();

        EXPECT_FALSE(dist_submit_handle_shared_tensor_map_result(ctx, result));
        EXPECT_EQ(worker_->local_index, kFlagCap);
        EXPECT_EQ(g_dist.fatal, 1);
        EXPECT_EQ(g_dist.error_code, PTO2_ERROR_INVALID_ARGS);
    }
}

TEST(FdwicSharedSubmitStatus, EverySharedFailureCodeIsHostVisible) {
    EXPECT_EQ(
        runtime_status_from_error_codes(PTO2_ERROR_TENSORMAP_CAPACITY, PTO2_ERROR_NONE), -PTO2_ERROR_TENSORMAP_CAPACITY
    );
    EXPECT_EQ(
        runtime_status_from_error_codes(PTO2_ERROR_TENSORMAP_PROTOCOL, PTO2_ERROR_NONE), -PTO2_ERROR_TENSORMAP_PROTOCOL
    );
    EXPECT_EQ(
        runtime_status_from_error_codes(PTO2_ERROR_TENSORMAP_PARTIAL_PUBLISH, PTO2_ERROR_NONE),
        -PTO2_ERROR_TENSORMAP_PARTIAL_PUBLISH
    );
}

}  // namespace
