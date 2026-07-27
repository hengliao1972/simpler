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
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

// Compile the production CPU-sim Submit implementation directly. Disable only
// diagnostics rather than copying the Claim, Register, Build, or failure state machines.
#define PTO_FDWIC_TRACE_ENABLED 0
#include "dist_engine/aicore/dist_engine.cpp"  // NOLINT(build/include)

[[noreturn]] void assert_impl(const char *condition, const char *, int) { throw std::logic_error(condition); }

// The production CPU-sim TU declares the real orchestration entry. These tests
// call compete-first Submit directly, so only a link definition is required.
extern "C" void aicpu_orchestration_entry(const L2TaskArgs &) {}
// The full production TU also contains unused worker-finish and platform hooks.
// These stubs satisfy linkage only and do not participate in the assertions.
volatile uint8_t *sim_get_reg_base() { return nullptr; }
uint32_t sim_get_physical_core_id() { return 0; }

Runtime::Runtime() {
    for (uint64_t &address : func_id_to_addr_) {
        address = 0;
    }
    use_example_exec_time_ = false;
    for (int32_t &duration : example_exec_time_ns_) {
        duration = 0;
    }
}

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
        static_assert(PTO_FDWIC_SHARED_MAP == 0);
        // Production worker_state.h does not bind the fallback under
        // __CPU_SIM. Bind it explicitly so every entry uses real state.
        g_dist_ptr = &g_dist_fallback;
        g_self = worker_.get();
        g_fdwic_joint_submit_seen = false;
        dist_core_reset(*worker_, CoreType::AIV, /*block=*/0, LANE_AIV0);
        worker_->core_idx = 0;

        g_dist.H = kHDefault;
        g_dist.heap_base = nullptr;
        g_dist.heap_size = 0;
        g_dist.runtime = &runtime_;
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

    Runtime runtime_;
    std::unique_ptr<DistCore> worker_ = std::make_unique<DistCore>();
};

TEST_F(FdwicSubmitCapacityTest, RegisterFailureStopsBuildAndClosesFollowingClaimGate) {
    // Fill the target bucket to CAP-1. The first OUTPUT_EXISTING consumes the
    // last slot, and the second one reaches the exact per-bucket limit.
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

    // The first region consumes the final slot and the second is not inserted.
    // Register failure must return before WinnerBuild or slot allocation.
    EXPECT_EQ(dist_private_tensor_map_load_head(worker_->map, bucket), 0U);
    EXPECT_EQ(dist_private_tensor_map_load_tail(worker_->map, bucket), kMapBucketCapacity);
    const uint32_t last_slot = dist_private_tensor_map_slot_index(bucket, kMapBucketCapacity - 1);
    EXPECT_EQ(worker_->map.entries[last_slot].buf_addr, first_output.buffer.addr);
    EXPECT_EQ(worker_->occupied_count, occupied_before);
    EXPECT_EQ(worker_->owned_total, owned_before);
    EXPECT_EQ(std::memcmp(slots_before.data(), worker_->slots, slots_before.size()), 0);

    // Failure latching reuses the task-cap sentinel. AICPU can retrieve the
    // structured capacity error from the same fatal cache line.
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
    // Register is not a whole-task transaction. Earlier outputs may be
    // published before a later full ring makes the task fail without Build.
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
    // Register follows argument order and returns on the first failure. When a
    // full-bucket output comes first, no later free-bucket output is inserted.
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

TEST_F(FdwicSubmitCapacityTest, ManualDependencyInoutKeepsCreatorButSkipsPrivateLookupAndRegister) {
    Tensor manual_output = make_existing_output(0x700000);
    manual_output.manual_dep = true;
    manual_output.owner_task_id = PTO2TaskId::make(0, 0);
    const Tensor normal_output = make_existing_output_in_another_bucket(manual_output.buffer.addr);
    const uint32_t manual_bucket = dist_private_tensor_map_hash(manual_output.buffer.addr);
    const uint32_t normal_bucket = dist_private_tensor_map_hash(normal_output.buffer.addr);
    ASSERT_NE(manual_bucket, normal_bucket);
    ASSERT_TRUE(dist_private_tensor_map_insert(worker_->map, manual_output, /*producer=*/1));
    ASSERT_TRUE(dist_private_tensor_map_insert(worker_->map, normal_output, /*producer=*/2));
    worker_->local_index = 3;

    L0TaskArgs args;
    args.add_inout(manual_output, normal_output);
    ASSERT_EQ(args.tag(0), TensorArgType::INOUT);
    ASSERT_EQ(args.tag(1), TensorArgType::INOUT);
    ASSERT_TRUE(args.tensor(0).ref().manual_dep);
    ASSERT_FALSE(args.tensor(1).ref().manual_dep);
    MixedKernels mixed;
    mixed.aiv0_kernel_id = 13;

    const DistCompeteFirstTicket ticket = dist_submit_compete_first_begin(nullptr, mixed);
    ASSERT_EQ(ticket.ready, 1);
    ASSERT_EQ(ticket.won, 1);
    ASSERT_EQ(ticket.task_id, 3);
    (void)dist_submit_compete_first_finish(nullptr, mixed, ticket, args);

    // manual_dep retains creator 0 but neither consumes map producer 1 nor
    // registers task 3. The normal INOUT consumes producer 2 and registers 3.
    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_NONE);
    EXPECT_EQ(dist_private_tensor_map_load_head(worker_->map, manual_bucket), 0U);
    EXPECT_EQ(dist_private_tensor_map_load_tail(worker_->map, manual_bucket), 1U);
    EXPECT_EQ(dist_private_tensor_map_lookup(worker_->map, manual_output), 1);
    EXPECT_EQ(dist_private_tensor_map_load_head(worker_->map, normal_bucket), 0U);
    EXPECT_EQ(dist_private_tensor_map_load_tail(worker_->map, normal_bucket), 2U);
    EXPECT_EQ(dist_private_tensor_map_lookup(worker_->map, normal_output), 3);
    EXPECT_EQ(worker_->occupied_count, 1);
    const RingSlot *built_slot = nullptr;
    for (int32_t index = 0; index < kPrivateSlots; ++index) {
        const RingSlot &slot = worker_->slots[index];
        if (slot.occupied && slot.built && slot.task_id == 3) {
            ASSERT_EQ(built_slot, nullptr);
            built_slot = &slot;
        }
    }
    ASSERT_NE(built_slot, nullptr);
    ASSERT_EQ(built_slot->fanin_count, 2);
    int32_t creator_count = 0;
    int32_t manual_map_count = 0;
    int32_t normal_map_count = 0;
    for (int32_t index = 0; index < built_slot->fanin_count; ++index) {
        creator_count += built_slot->fanin[index] == 0 ? 1 : 0;
        manual_map_count += built_slot->fanin[index] == 1 ? 1 : 0;
        normal_map_count += built_slot->fanin[index] == 2 ? 1 : 0;
    }
    EXPECT_EQ(creator_count, 1);
    EXPECT_EQ(manual_map_count, 0);
    EXPECT_EQ(normal_map_count, 1);
    EXPECT_EQ(worker_->owned_total, 0);
}

}  // namespace
