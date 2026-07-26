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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

// 直接编译 production CPU-sim Submit 实现；只关闭诊断插桩，不复制 Claim、
// Register、Build 或容量失败状态机。
#define PTO_FDWIC_TRACE_ENABLED 0
#include "dist_engine/aicore/dist_engine.cpp"  // NOLINT(build/include)

[[noreturn]] void assert_impl(const char *condition, const char *, int) { throw std::logic_error(condition); }

// CPU-sim production TU 声明了真实 orchestration 入口。该测试直接调用
// compete-first Submit，不进入 orchestration replay，因此只需提供链接定义。
extern "C" void aicpu_orchestration_entry(const L2TaskArgs &) {}
// 完整 production TU 还包含未执行的 worker-finish/platform hooks；这两个
// 链接桩只满足符号解析，不参与本测试的 Submit/Claim/Build 断言。
volatile uint8_t *sim_get_reg_base() { return nullptr; }
uint32_t sim_get_physical_core_id() { return 0; }

namespace {

Tensor make_existing_output(uint64_t address) {
    Tensor tensor{};
    const uint32_t shape[1] = {1};
    tensor.init_external(
        reinterpret_cast<void *>(static_cast<uintptr_t>(address)), sizeof(float), shape, 1, DataType::FLOAT32, 0
    );
    return tensor;
}

Tensor make_existing_output_in_another_bucket(uint64_t address) {
    const uint32_t original_bucket = dist_private_tensor_map_hash(address);
    for (uint64_t candidate = address + 64; candidate < address + (1ULL << 30); candidate += 64) {
        if (dist_private_tensor_map_hash(candidate) != original_bucket) {
            return make_existing_output(candidate);
        }
    }
    throw std::logic_error("failed to find a TensorMap address in another bucket");
}

void fill_output_bucket(DistTensorMap &map, const Tensor &tensor, uint32_t count) {
    for (uint32_t index = 0; index < count; ++index) {
        ASSERT_TRUE(dist_private_tensor_map_insert(map, tensor, /*producer=*/0))
            << "index=" << index << " count=" << count;
    }
}

class FdwicSubmitCapacityTest : public ::testing::Test {
protected:
    void SetUp() override {
        // __CPU_SIM 下 production worker_state.h 默认不绑定 fallback；显式绑定
        // 后，下面所有入口都读写真实 g_dist/g_self。
        g_dist_ptr = &g_dist_fallback;
        g_self = worker_.get();
        g_fdwic_joint_submit_seen = false;
        dist_core_reset(*worker_, CoreType::AIV, /*block=*/0, LANE_AIV0);
        worker_->core_idx = 0;

        g_dist.H = kHDefault;
        g_dist.heap_base = nullptr;
        g_dist.heap_size = 0;
        g_dist.runtime = nullptr;
        g_dist.num_workers = 1;
        g_dist.num_blocks = 1;
        g_dist.fatal = 0;
        g_dist.error_code = PTO2_ERROR_NONE;
        g_dist.blocks[0].any_pub = 0;
        for (int32_t i = 0; i < kPrivateSlots; ++i) {
            g_dist.blocks[0].slots[i].state.v = kWonStateFree;
        }
        for (int32_t shard = 0; shard < kCursorShards; ++shard) {
            g_dist.cube_cursor[shard].v = -1;
            g_dist.vector_cursor[shard].v = -1;
            g_dist.alloc_cursor[shard].v = -1;
        }
    }

    void TearDown() override {
        g_self = nullptr;
        g_dist_ptr = nullptr;
    }

    std::unique_ptr<DistCore> worker_ = std::make_unique<DistCore>();
};

TEST_F(FdwicSubmitCapacityTest, RegisterFailureStopsBuildAndClosesFollowingClaimGate) {
    // 把真实目标 bucket 填到 CAP-1。Register 的第一个
    // OUTPUT_EXISTING 占用最后一槽，第二个精确撞到 per-bucket 上限。
    const Tensor first_output = make_existing_output(0x100000);
    const Tensor second_output = first_output;
    const uint32_t bucket = dist_private_tensor_map_hash(first_output.buffer.addr);
    fill_output_bucket(worker_->map, first_output, kMapBucketCapacity - 1);
    ASSERT_EQ(dist_private_tensor_map_load_head(worker_->map, bucket), 0U);
    ASSERT_EQ(dist_private_tensor_map_load_tail(worker_->map, bucket), kMapBucketCapacity - 1);
    L0TaskArgs args;
    args.add_output(first_output, second_output);
    ASSERT_EQ(args.tensor_count(), 2);
    ASSERT_EQ(args.tag(0), TensorArgType::OUTPUT_EXISTING);
    ASSERT_EQ(args.tag(1), TensorArgType::OUTPUT_EXISTING);

    MixedKernels mixed;
    mixed.aiv0_kernel_id = 7;

    std::array<std::byte, sizeof(worker_->slots)> slots_before{};
    std::memcpy(slots_before.data(), worker_->slots, slots_before.size());
    const int32_t occupied_before = worker_->occupied_count;
    const int32_t owned_before = worker_->owned_total;

    const DistCompeteFirstTicket ticket = dist_submit_compete_first_begin(nullptr, mixed);
    ASSERT_EQ(ticket.ready, 1);
    ASSERT_EQ(ticket.won, 1);
    ASSERT_EQ(ticket.claim_attempted, 1);
    ASSERT_EQ(ticket.task_id, 0);
    ASSERT_EQ(worker_->local_index, 1);
    ASSERT_EQ(g_dist.vector_cursor[0].v, 0);

    (void)dist_submit_compete_first_finish(nullptr, mixed, ticket, args);

    // 第一个 region 已占用本桶最后一槽，第二个 region 未被插入；
    // Register 失败必须在 WinnerBuild/slot 分配之前返回。
    EXPECT_EQ(dist_private_tensor_map_load_head(worker_->map, bucket), 0U);
    EXPECT_EQ(dist_private_tensor_map_load_tail(worker_->map, bucket), kMapBucketCapacity);
    const uint32_t last_slot = dist_private_tensor_map_slot_index(bucket, kMapBucketCapacity - 1);
    EXPECT_EQ(worker_->map.entries[last_slot].buf_addr, first_output.buffer.addr);
    EXPECT_EQ(worker_->occupied_count, occupied_before);
    EXPECT_EQ(worker_->owned_total, owned_before);
    EXPECT_EQ(std::memcmp(slots_before.data(), worker_->slots, slots_before.size()), 0);

    // 失败锁存复用 task-cap sentinel；AICPU 最终可从同一 fatal cache line
    // 取得结构化容量错误。
    EXPECT_EQ(worker_->local_index, kFlagCap);
    EXPECT_EQ(g_dist.fatal, 1);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_TENSORMAP_CAPACITY);
    set_fatal_code(PTO2_ERROR_INVALID_ARGS);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_TENSORMAP_CAPACITY);

    const int64_t claim_cursor_after_failure = g_dist.vector_cursor[0].v;
    const DistCompeteFirstTicket blocked = dist_submit_compete_first_begin(nullptr, mixed);
    EXPECT_EQ(blocked.task_id, kFlagCap);
    EXPECT_EQ(blocked.ready, 0);
    EXPECT_EQ(blocked.won, 0);
    EXPECT_EQ(blocked.claim_attempted, 0);
    EXPECT_EQ(worker_->local_index, kFlagCap);
    EXPECT_EQ(g_dist.vector_cursor[0].v, claim_cursor_after_failure);
}

TEST_F(FdwicSubmitCapacityTest, CrossBucketFailureKeepsTheExistingPrefixPublicationContract) {
    // 当前 Register 合同不是整 task 事务：较早 output 可以先登记，后续
    // output 满环后 task 会 fatal 且不 Build，但已发布前缀不回滚。
    const Tensor full_output = make_existing_output(0x300000);
    const Tensor prefix_output = make_existing_output_in_another_bucket(full_output.buffer.addr);
    const uint32_t full_bucket = dist_private_tensor_map_hash(full_output.buffer.addr);
    const uint32_t prefix_bucket = dist_private_tensor_map_hash(prefix_output.buffer.addr);
    ASSERT_NE(prefix_bucket, full_bucket);
    fill_output_bucket(worker_->map, full_output, kMapBucketCapacity);

    L0TaskArgs args;
    args.add_output(prefix_output, full_output);
    MixedKernels mixed;
    mixed.aiv0_kernel_id = 9;

    std::array<std::byte, sizeof(worker_->slots)> slots_before{};
    std::memcpy(slots_before.data(), worker_->slots, slots_before.size());
    const uint64_t full_head_before = dist_private_tensor_map_load_head(worker_->map, full_bucket);
    const uint64_t full_tail_before = dist_private_tensor_map_load_tail(worker_->map, full_bucket);
    const uint64_t prefix_tail_before = dist_private_tensor_map_load_tail(worker_->map, prefix_bucket);

    const DistCompeteFirstTicket ticket = dist_submit_compete_first_begin(nullptr, mixed);
    ASSERT_EQ(ticket.ready, 1);
    ASSERT_EQ(ticket.won, 1);
    ASSERT_EQ(ticket.task_id, 0);
    (void)dist_submit_compete_first_finish(nullptr, mixed, ticket, args);

    EXPECT_EQ(dist_private_tensor_map_load_head(worker_->map, full_bucket), full_head_before);
    EXPECT_EQ(dist_private_tensor_map_load_tail(worker_->map, full_bucket), full_tail_before);
    EXPECT_EQ(dist_private_tensor_map_load_tail(worker_->map, prefix_bucket), prefix_tail_before + 1);
    EXPECT_EQ(dist_private_tensor_map_lookup(worker_->map, prefix_output), 0);
    EXPECT_EQ(std::memcmp(slots_before.data(), worker_->slots, slots_before.size()), 0);
    EXPECT_EQ(worker_->occupied_count, 0);
    EXPECT_EQ(worker_->owned_total, 0);
    EXPECT_EQ(worker_->local_index, kFlagCap);
    EXPECT_EQ(g_dist.fatal, 1);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_TENSORMAP_CAPACITY);
}

TEST_F(FdwicSubmitCapacityTest, FullBucketFirstStopsBeforeAFreeLaterOutput) {
    // Register 按参数顺序处理并在首个失败处返回。满桶 output 位于前面时，
    // 后续落在空闲桶的 output 不能被登记，也不能改写满桶的任何物理槽。
    const Tensor full_output = make_existing_output(0x500000);
    const Tensor later_free_output = make_existing_output_in_another_bucket(full_output.buffer.addr);
    const uint32_t full_bucket = dist_private_tensor_map_hash(full_output.buffer.addr);
    const uint32_t free_bucket = dist_private_tensor_map_hash(later_free_output.buffer.addr);
    ASSERT_NE(free_bucket, full_bucket);
    fill_output_bucket(worker_->map, full_output, kMapBucketCapacity);

    L0TaskArgs args;
    args.add_output(full_output, later_free_output);
    MixedKernels mixed;
    mixed.aiv0_kernel_id = 11;

    std::vector<std::byte> map_before(sizeof(worker_->map));
    std::memcpy(map_before.data(), &worker_->map, map_before.size());
    std::array<std::byte, sizeof(worker_->slots)> slots_before{};
    std::memcpy(slots_before.data(), worker_->slots, slots_before.size());
    const int32_t occupied_before = worker_->occupied_count;
    const int32_t owned_before = worker_->owned_total;

    const DistCompeteFirstTicket ticket = dist_submit_compete_first_begin(nullptr, mixed);
    ASSERT_EQ(ticket.ready, 1);
    ASSERT_EQ(ticket.won, 1);
    ASSERT_EQ(ticket.task_id, 0);
    (void)dist_submit_compete_first_finish(nullptr, mixed, ticket, args);

    EXPECT_EQ(std::memcmp(map_before.data(), &worker_->map, map_before.size()), 0);
    EXPECT_EQ(dist_private_tensor_map_load_head(worker_->map, full_bucket), 0U);
    EXPECT_EQ(dist_private_tensor_map_load_tail(worker_->map, full_bucket), kMapBucketCapacity);
    EXPECT_EQ(dist_private_tensor_map_load_head(worker_->map, free_bucket), 0U);
    EXPECT_EQ(dist_private_tensor_map_load_tail(worker_->map, free_bucket), 0U);
    EXPECT_EQ(std::memcmp(slots_before.data(), worker_->slots, slots_before.size()), 0);
    EXPECT_EQ(worker_->occupied_count, occupied_before);
    EXPECT_EQ(worker_->owned_total, owned_before);
    EXPECT_EQ(worker_->local_index, kFlagCap);
    EXPECT_EQ(g_dist.fatal, 1);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_TENSORMAP_CAPACITY);
}

}  // namespace
