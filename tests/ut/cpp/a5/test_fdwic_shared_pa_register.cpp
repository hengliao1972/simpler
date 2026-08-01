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

constexpr int32_t kExpectedAicWorkers = 32;
constexpr int32_t kExpectedAivWorkers = 64;
constexpr int32_t kExpectedWorkers = kExpectedAicWorkers + kExpectedAivWorkers;
constexpr int64_t kUntouchedFrontier = 0x12345678;
constexpr int64_t kUntouchedSharedHeapCursor = 0x23456789;

class FdwicSharedPaRegisterTest : public ::testing::Test {
protected:
    void SetUp() override {
        static_assert(PTO_FDWIC_SHARED_MAP == 1);
        unsetenv("PTO_DIST_H");
        unsetenv("PTO_DIST_SKIP_EXEC");
        unsetenv("PTO_DIST_WATCHDOG");

        runtime_ = std::make_unique<Runtime>();
        configure_roles(kExpectedAicWorkers);

        rt_.dist_global = &g_dist_fallback;
        rt_.gm_heap = heap_token_.data();
        rt_.gm_heap_size = kFdwicSharedHeapBytes;
        g_dist_ptr = &g_dist_fallback;
    }

    void TearDown() override {
        unsetenv("PTO_DIST_H");
        unsetenv("PTO_DIST_SKIP_EXEC");
        unsetenv("PTO_DIST_WATCHDOG");
        g_dist_ptr = nullptr;
    }

    void configure_roles(int32_t aic_workers) {
        runtime_->worker_count = kExpectedWorkers;
        for (int32_t worker = 0; worker < kExpectedWorkers; ++worker) {
            runtime_->workers[worker].core_type = worker < aic_workers ? CoreType::AIC : CoreType::AIV;
        }
    }

    int32_t register_runtime(int32_t num_workers = kExpectedWorkers) {
        runtime_->worker_count = num_workers;
        return dist_engine_register(&rt_, &orch_args_, num_workers, runtime_.get());
    }

    void expect_config_failure(int32_t num_workers = kExpectedWorkers) {
        runtime_->dist.shared_addr = 0xfeedbeefU;
        g_dist_fallback.frontier = kUntouchedFrontier;
        g_dist_fallback.shared_pa.shared_heap_cursor[0].v = kUntouchedSharedHeapCursor;

        EXPECT_EQ(register_runtime(num_workers), -PTO2_ERROR_DIST_CONFIG_INVALID);
        EXPECT_EQ(runtime_->dist.shared_addr, 0U);
        EXPECT_EQ(g_dist_fallback.frontier, kUntouchedFrontier);
        EXPECT_EQ(g_dist_fallback.shared_pa.shared_heap_cursor[0].v, kUntouchedSharedHeapCursor);
    }

    PTO2Runtime rt_{};
    L2TaskArgs orch_args_{};
    std::unique_ptr<Runtime> runtime_;
    alignas(64) std::array<std::byte, 64> heap_token_{};
};

TEST_F(FdwicSharedPaRegisterTest, AcceptsFixedDeploymentAndBuildsCompleteTopology) {
    ASSERT_EQ(kFdwicSharedHeapShards, 8U);
    ASSERT_EQ(kFdwicSharedVectorCursorShards, 8U);
    ASSERT_EQ(kFdwicSharedHeapBytes, 256ULL << 20);
    for (uint32_t shard = 0; shard < kFdwicSharedHeapShards; ++shard) {
        g_dist_fallback.shared_pa.shared_heap_cursor[shard].v = 100 + shard;
    }
    g_dist_fallback.shared_pa.shared_heap_vend.v = 4096;

    ASSERT_EQ(register_runtime(), 0);

    EXPECT_EQ(runtime_->dist.shared_addr, reinterpret_cast<uint64_t>(&g_dist_fallback));
    EXPECT_EQ(g_dist_ptr, &g_dist_fallback);
    EXPECT_EQ(g_dist.heap_base, reinterpret_cast<uint8_t *>(heap_token_.data()));
    EXPECT_EQ(g_dist.heap_size, kFdwicSharedHeapBytes);
    EXPECT_EQ(g_dist.num_workers, kExpectedWorkers);
    EXPECT_EQ(g_dist.num_blocks, kExpectedAicWorkers);
    EXPECT_EQ(g_dist.H, kHDefault);
    EXPECT_EQ(g_dist.orch_args, &orch_args_);
    EXPECT_EQ(g_dist.rt, &rt_);
    EXPECT_EQ(g_dist.runtime, runtime_.get());

    for (int32_t block = 0; block < kExpectedAicWorkers; ++block) {
        EXPECT_EQ(g_dist.layout[block].block_id, block);
        EXPECT_EQ(g_dist.layout[block].lane, LANE_AIC);

        const int32_t aiv0 = kExpectedAicWorkers + 2 * block;
        const int32_t aiv1 = aiv0 + 1;
        EXPECT_EQ(g_dist.layout[aiv0].block_id, block);
        EXPECT_EQ(g_dist.layout[aiv0].lane, LANE_AIV0);
        EXPECT_EQ(g_dist.layout[aiv1].block_id, block);
        EXPECT_EQ(g_dist.layout[aiv1].lane, LANE_AIV1);
    }

    for (int32_t group = 0; group < kFinalBarrierGroups; ++group) {
        EXPECT_EQ(g_dist.final_barrier.leaf_arrivals[group].expected, 6);
        EXPECT_EQ(g_dist.final_barrier.leaf_arrivals[group].v, 0);
        EXPECT_EQ(g_dist.final_barrier.leaf_releases[group].v, 0);
    }
    EXPECT_EQ(g_dist.final_barrier.root_arrival.expected, kFinalBarrierGroups);
    EXPECT_EQ(g_dist.final_barrier.root_arrival.v, 0);
    EXPECT_EQ(g_dist.final_barrier.root_release.v, 0);

    for (uint32_t shard = 0; shard < kFdwicSharedHeapShards; ++shard) {
        EXPECT_EQ(g_dist.shared_pa.shared_heap_cursor[shard].v, 0) << "shard=" << shard;
    }
    EXPECT_EQ(g_dist.shared_pa.shared_heap_vend.v, 0);
    for (uint32_t shard = 0; shard < kFdwicSharedVectorCursorShards; ++shard) {
        EXPECT_EQ(g_dist.shared_pa.shared_vector_cursor[shard].v, -1) << "shard=" << shard;
    }
}

TEST_F(FdwicSharedPaRegisterTest, RejectsWorkerCountOutsideFixedDeployment) {
    for (const int32_t workers : {kExpectedWorkers - 1, kExpectedWorkers + 1}) {
        SCOPED_TRACE(testing::Message() << "workers=" << workers);
        expect_config_failure(workers);
    }
}

TEST_F(FdwicSharedPaRegisterTest, RejectsIncompleteRoleLayout) {
    for (const int32_t aic_workers : {kExpectedAicWorkers - 1, kExpectedAicWorkers + 1}) {
        SCOPED_TRACE(testing::Message() << "AIC=" << aic_workers);
        configure_roles(aic_workers);
        expect_config_failure();
    }
}

TEST_F(FdwicSharedPaRegisterTest, RejectsUnknownWorkerRole) {
    runtime_->workers[kExpectedWorkers - 1].core_type = static_cast<CoreType>(7);
    expect_config_failure();
}

TEST_F(FdwicSharedPaRegisterTest, RejectsHeapSizeOutsideFixedDeployment) {
    for (const uint64_t heap_size : {kFdwicSharedHeapBytes - 64, kFdwicSharedHeapBytes + 64}) {
        SCOPED_TRACE(testing::Message() << "heap_size=" << heap_size);
        rt_.gm_heap_size = heap_size;
        expect_config_failure();
    }
}

TEST_F(FdwicSharedPaRegisterTest, RejectsHistoryThatCannotCoverUpToAllocDependency) {
    ASSERT_EQ(setenv("PTO_DIST_H", "3", 1), 0);
    expect_config_failure();
}

}  // namespace
