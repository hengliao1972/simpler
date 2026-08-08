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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

#define PTO_FDWIC_TRACE_ENABLED 0
#include "dist_engine/aicpu/dist_engine.cpp"  // NOLINT(build/include)

void cache_invalidate_range(const void *, size_t) {}

Runtime::Runtime() {
    std::memset(workers, 0, sizeof(workers));
    worker_count = 0;
    dist.shared_addr = 0;
    dist.num_workers = 0;
    for (uint64_t &address : func_id_to_addr_)
        address = 0;
    use_example_exec_time_ = false;
    for (int32_t &duration : example_exec_time_ns_)
        duration = 0;
}

namespace {

class FdwicCrossCoreRegisterTest : public ::testing::Test {
protected:
    void SetUp() override {
        static_assert(PTO_FDWIC_SHARED_MAP == 1);
        static_assert(PTO_FDWIC_SCHEDULER_MODE == 1);
        unsetenv("PTO_DIST_H");
        unsetenv("PTO_DIST_SKIP_EXEC");
        unsetenv("PTO_DIST_WATCHDOG");

        runtime_ = std::make_unique<Runtime>();
        configure_complete_blocks(1);
        rt_.dist_global = &g_dist_fallback;
        rt_.gm_heap = heap_token_.data();
        rt_.gm_heap_size = heap_token_.size();
        g_dist_ptr = &g_dist_fallback;
    }

    void TearDown() override {
        unsetenv("PTO_DIST_H");
        unsetenv("PTO_DIST_SKIP_EXEC");
        unsetenv("PTO_DIST_WATCHDOG");
        g_dist_ptr = nullptr;
    }

    void configure_complete_blocks(int32_t blocks) {
        runtime_->worker_count = 3 * blocks;
        for (int32_t worker = 0; worker < blocks; ++worker)
            runtime_->workers[worker].core_type = CoreType::AIC;
        for (int32_t worker = blocks; worker < 3 * blocks; ++worker)
            runtime_->workers[worker].core_type = CoreType::AIV;
    }

    int32_t register_runtime() {
        return dist_engine_register(&rt_, &orch_args_, runtime_->worker_count, runtime_.get());
    }

    PTO2Runtime rt_{};
    L2TaskArgs orch_args_{};
    std::unique_ptr<Runtime> runtime_;
    alignas(64) std::array<std::byte, 64> heap_token_{};
};

TEST_F(FdwicCrossCoreRegisterTest, AcceptsOneCompleteBlockAndActualHeapCapacity) {
    g_dist_fallback.cross_core_ordinary.fatal.state = 91;
    g_dist_fallback.cross_core_ordinary.heap_cursor.state = 92;
    g_dist_fallback.cross_core_ordinary.execute_owner[0].state = 93;
    g_dist_fallback.cross_core_ordinary.tasks[0].control.state = 94;
    g_dist_fallback.cross_core_ordinary.outputs[0].control.state = 95;
    g_dist_fallback.cross_core_ordinary.tensor_map.tails[0].state = 96;
    g_dist_fallback.cross_core_ordinary.tensor_map.slots[0].sequence.state = 97;

    ASSERT_EQ(register_runtime(), 0);

    EXPECT_EQ(g_dist.num_workers, 3);
    EXPECT_EQ(g_dist.num_blocks, 1);
    EXPECT_EQ(g_dist.heap_size, heap_token_.size());
    EXPECT_EQ(g_dist.layout[0].lane, LANE_AIC);
    EXPECT_EQ(g_dist.layout[1].lane, LANE_AIV0);
    EXPECT_EQ(g_dist.layout[2].lane, LANE_AIV1);
    EXPECT_EQ(g_dist.cross_core_ordinary.fatal.state, 0);
    EXPECT_EQ(g_dist.cross_core_ordinary.heap_cursor.state, 0);
    EXPECT_EQ(g_dist.cross_core_ordinary.execute_owner[0].state, 0);
    EXPECT_EQ(g_dist.cross_core_ordinary.tasks[0].control.state, 0);
    EXPECT_EQ(g_dist.cross_core_ordinary.outputs[0].control.state, 0);
    EXPECT_EQ(g_dist.cross_core_ordinary.tensor_map.tails[0].state, 0);
    EXPECT_EQ(g_dist.cross_core_ordinary.tensor_map.slots[0].sequence.state, -1);
}

TEST_F(FdwicCrossCoreRegisterTest, AcceptsMultipleCompleteBlocksAndZeroHistory) {
    configure_complete_blocks(2);
    ASSERT_EQ(setenv("PTO_DIST_H", "0", 1), 0);

    ASSERT_EQ(register_runtime(), 0);

    EXPECT_EQ(g_dist.num_workers, 6);
    EXPECT_EQ(g_dist.num_blocks, 2);
    EXPECT_EQ(g_dist.H, 0);
    EXPECT_EQ(g_dist.layout[1].lane, LANE_AIC);
    EXPECT_EQ(g_dist.layout[4].lane, LANE_AIV0);
    EXPECT_EQ(g_dist.layout[5].lane, LANE_AIV1);
}

TEST_F(FdwicCrossCoreRegisterTest, RejectsIncompleteBlockTopologyBeforePublishingState) {
    runtime_->worker_count = 2;
    runtime_->workers[0].core_type = CoreType::AIC;
    runtime_->workers[1].core_type = CoreType::AIV;
    runtime_->dist.shared_addr = 0xfeedbeefU;
    g_dist_fallback.frontier = 0x12345678;

    EXPECT_EQ(register_runtime(), -PTO2_ERROR_DIST_CONFIG_INVALID);
    EXPECT_EQ(runtime_->dist.shared_addr, 0U);
    EXPECT_EQ(g_dist_fallback.frontier, 0x12345678);
}

TEST_F(FdwicCrossCoreRegisterTest, RejectsUnknownWorkerRole) {
    runtime_->workers[2].core_type = static_cast<CoreType>(7);

    EXPECT_EQ(register_runtime(), -PTO2_ERROR_DIST_CONFIG_INVALID);
}

}  // namespace
