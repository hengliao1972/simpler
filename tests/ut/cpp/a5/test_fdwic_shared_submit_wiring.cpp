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

#define PTO_FDWIC_TRACE_ENABLED 0
#include "dist_engine/aicore/dist_engine.cpp"  // NOLINT(build/include)
#include "dist_engine/aicpu/shared_tensor_map_init.h"

[[noreturn]] void assert_impl(const char *condition, const char *, int) { throw std::logic_error(condition); }

extern "C" void aicpu_orchestration_entry(const L2TaskArgs &) {}
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

Tensor make_existing_tensor(uint64_t address) {
    Tensor tensor{};
    const uint32_t shape[1] = {1};
    tensor.init_external(
        reinterpret_cast<void *>(static_cast<uintptr_t>(address)), sizeof(float), shape, 1, DataType::FLOAT32, 0
    );
    return tensor;
}

Tensor make_owned_tensor(uint64_t address, int32_t owner) {
    Tensor tensor = make_existing_tensor(address);
    tensor.owner_task_id = PTO2TaskId::make(0, static_cast<uint32_t>(owner));
    return tensor;
}

const RingSlot *find_built_slot(const DistCore &worker, int32_t task_id) {
    for (int32_t index = 0; index < kPrivateSlots; ++index) {
        const RingSlot &slot = worker.slots[index];
        if (slot.occupied && slot.built && slot.task_id == task_id) {
            return &slot;
        }
    }
    return nullptr;
}

int32_t fanin_occurrences(const RingSlot &slot, int32_t producer) {
    int32_t count = 0;
    for (int32_t index = 0; index < slot.fanin_count; ++index) {
        if (slot.fanin[index] == producer) {
            ++count;
        }
    }
    return count;
}

class FdwicSharedSubmitWiringTest : public ::testing::Test {
protected:
    void SetUp() override {
        static_assert(PTO_FDWIC_SHARED_MAP == 1);
        static_assert(!kFdwicCompiledBackendReady);
        g_dist_ptr = &g_dist_fallback;
        g_self = worker_.get();
        reset_worker(CoreType::AIV, LANE_AIV0);
        dist_shared_tensor_map_reset(g_dist.shared_tensor_map);

        g_fdwic_joint_submit_seen = false;
        g_dist.H = kHDefault;
        g_dist.heap_base = nullptr;
        g_dist.heap_size = 0;
        g_dist.runtime = &runtime_;
        g_dist.num_workers = 1;
        g_dist.num_blocks = 1;
        g_dist.frontier = -1;
        g_dist.fatal = 0;
        g_dist.error_code = PTO2_ERROR_NONE;
        g_dist.blocks[0].any_pub = 0;
        for (int32_t index = 0; index < kPrivateSlots; ++index) {
            g_dist.blocks[0].slots[index].state.v = kWonStateFree;
        }
        for (int32_t shard = 0; shard < kCursorShards; ++shard) {
            g_dist.cube_cursor[shard].v = -1;
            g_dist.vector_cursor[shard].v = -1;
            g_dist.alloc_cursor[shard].v = -1;
        }
        for (int32_t task = 0; task < 256; ++task) {
            reset_task_cell(task);
        }
    }

    void TearDown() override {
        g_self = nullptr;
        g_dist_ptr = nullptr;
    }

    void reset_worker(CoreType role, int32_t lane) {
        dist_core_reset(*worker_, role, /*block=*/0, lane);
        worker_->core_idx = 0;
    }

    void publish_seed_task(const SharedTensorMapValue *entries, uint32_t count, int32_t task, int32_t history) {
        ASSERT_EQ(
            dist_shared_tensor_map_publish_task(g_dist.shared_tensor_map, entries, count, task, history),
            DistSharedTensorMapTaskPublishResult::Committed
        );
        ASSERT_EQ(g_dist.shared_tensor_map.committed_tasks.v, task + 1);
    }

    void fill_one_bucket(const Tensor &tensor, int32_t count) {
        for (int32_t task = 0; task < count; ++task) {
            const SharedTensorMapValue entry = dist_shared_tensor_map_make_value(tensor, task);
            publish_seed_task(&entry, 1, task, kFlagCap - 1);
        }
    }

    std::vector<std::byte> snapshot_shared_tensor_map() const {
        std::vector<std::byte> snapshot(sizeof(g_dist.shared_tensor_map));
        std::memcpy(snapshot.data(), &g_dist.shared_tensor_map, snapshot.size());
        return snapshot;
    }

    Runtime runtime_;
    std::unique_ptr<DistCore> worker_ = std::make_unique<DistCore>();
};

TEST_F(FdwicSharedSubmitWiringTest, KernelWinnerCommitsWholeTaskAndBuildsItsSlot) {
    const Tensor input = make_existing_tensor(0x100000);
    const Tensor inout = make_existing_tensor(0x200000);
    const Tensor output = make_existing_tensor(0x300000);
    const SharedTensorMapValue seed_entries[] = {
        dist_shared_tensor_map_make_value(input, 0),
        dist_shared_tensor_map_make_value(inout, 0),
    };
    publish_seed_task(seed_entries, 2, /*task=*/0, kHDefault);
    worker_->local_index = 1;

    L0TaskArgs args;
    args.add_input(input);
    args.add_inout(inout);
    args.add_output(output);
    MixedKernels mixed;
    mixed.aiv0_kernel_id = 7;

    std::vector<std::byte> private_map_before(sizeof(worker_->map));
    std::memcpy(private_map_before.data(), &worker_->map, private_map_before.size());

    const DistCompeteFirstTicket ticket = dist_submit_compete_first_begin(nullptr, mixed);
    ASSERT_EQ(ticket.task_id, 1);
    ASSERT_EQ(ticket.won, 1);
    (void)dist_submit_compete_first_finish(nullptr, mixed, ticket, args);

    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.shared_tensor_map.committed_tasks.v, 2);
    bool protocol_ok = false;
    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, input, 2, kHDefault, protocol_ok), 0);
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, inout, 2, kHDefault, protocol_ok), 1);
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, output, 2, kHDefault, protocol_ok), 1);
    EXPECT_TRUE(protocol_ok);

    ASSERT_EQ(worker_->occupied_count, 1);
    EXPECT_TRUE(worker_->slots[0].built);
    EXPECT_EQ(worker_->slots[0].task_id, 1);
    ASSERT_EQ(worker_->slots[0].fanin_count, 1);
    EXPECT_EQ(worker_->slots[0].fanin[0], 0);
    EXPECT_EQ(std::memcmp(private_map_before.data(), &worker_->map, private_map_before.size()), 0);
}

TEST_F(FdwicSharedSubmitWiringTest, PaG2LatestInoutWriterFeedsFinalUp) {
    constexpr int32_t kUpKernelId = 15;
    Tensor mi_update = make_owned_tensor(0x510000, /*owner=*/0);
    Tensor li_update = make_owned_tensor(0x520000, /*owner=*/0);
    Tensor oi = make_owned_tensor(0x530000, /*owner=*/0);
    Tensor out_view = make_existing_tensor(0x540000);
    out_view.manual_dep = true;

    const Tensor group0_mi = make_owned_tensor(0x610000, /*owner=*/2);
    const Tensor group0_li = make_owned_tensor(0x620000, /*owner=*/2);
    const Tensor group0_oi_new = make_owned_tensor(0x630000, /*owner=*/3);
    const Tensor group1_mi = make_owned_tensor(0x710000, /*owner=*/6);
    const Tensor group1_li = make_owned_tensor(0x720000, /*owner=*/6);
    const Tensor group1_oi_new = make_owned_tensor(0x730000, /*owner=*/7);

    // Real PA G2 starts with Alloc/QK/SF/PV. Fresh runtime outputs carry their
    // creator IDs but do not enter the ordinary-region TensorMap. Task 1 gets
    // one impossible poison entry for out_view only in this test: a wrong
    // manual_dep lookup would add producer 1 to UP fan-in, while a wrong
    // register would replace its latest producer with task 4 or 8.
    publish_seed_task(nullptr, 0, /*task=*/0, kHDefault);
    const SharedTensorMapValue manual_poison = dist_shared_tensor_map_make_value(out_view, /*producer=*/1);
    publish_seed_task(&manual_poison, 1, /*task=*/1, kHDefault);
    for (int32_t task = 2; task <= 3; ++task) {
        publish_seed_task(nullptr, 0, task, kHDefault);
    }
    worker_->local_index = 4;

    L0TaskArgs up0_args;
    up0_args.add_input(group0_mi, group0_li, group0_oi_new);
    up0_args.add_inout(mi_update, li_update, oi, out_view);
    MixedKernels mixed;
    mixed.aiv0_kernel_id = kUpKernelId;

    std::vector<std::byte> private_map_before(sizeof(worker_->map));
    std::memcpy(private_map_before.data(), &worker_->map, private_map_before.size());

    const DistCompeteFirstTicket up0_ticket = dist_submit_compete_first_begin(nullptr, mixed);
    ASSERT_EQ(up0_ticket.task_id, 4);
    ASSERT_EQ(up0_ticket.won, 1);
    (void)dist_submit_compete_first_finish(nullptr, mixed, up0_ticket, up0_args);

    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_NONE);
    EXPECT_EQ(g_dist.shared_tensor_map.committed_tasks.v, 5);
    const RingSlot *up0_slot = find_built_slot(*worker_, 4);
    ASSERT_NE(up0_slot, nullptr);
    EXPECT_EQ(up0_slot->fanin_count, 3);
    EXPECT_EQ(fanin_occurrences(*up0_slot, 0), 1);
    EXPECT_EQ(fanin_occurrences(*up0_slot, 1), 0);
    EXPECT_EQ(fanin_occurrences(*up0_slot, 2), 1);
    EXPECT_EQ(fanin_occurrences(*up0_slot, 3), 1);

    bool protocol_ok = false;
    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, mi_update, 5, kHDefault, protocol_ok), 4);
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, li_update, 5, kHDefault, protocol_ok), 4);
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, oi, 5, kHDefault, protocol_ok), 4);
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, out_view, 5, kHDefault, protocol_ok), 1);
    EXPECT_TRUE(protocol_ok);

    // QK/SF/PV of group 1 likewise advance the exact turn without publishing
    // ordinary regions. The final UP must observe task 4 as the latest writer
    // of all three accumulators before it publishes task 8.
    for (int32_t task = 5; task <= 7; ++task) {
        publish_seed_task(nullptr, 0, task, kHDefault);
    }
    worker_->local_index = 8;

    L0TaskArgs up1_args;
    up1_args.add_input(group1_mi, group1_li, group1_oi_new);
    up1_args.add_inout(mi_update, li_update, oi, out_view);
    const DistCompeteFirstTicket up1_ticket = dist_submit_compete_first_begin(nullptr, mixed);
    ASSERT_EQ(up1_ticket.task_id, 8);
    ASSERT_EQ(up1_ticket.won, 1);
    (void)dist_submit_compete_first_finish(nullptr, mixed, up1_ticket, up1_args);

    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_NONE);
    EXPECT_EQ(g_dist.shared_tensor_map.committed_tasks.v, 9);
    const RingSlot *up1_slot = find_built_slot(*worker_, 8);
    ASSERT_NE(up1_slot, nullptr);
    EXPECT_EQ(up1_slot->fanin_count, 4);
    EXPECT_EQ(fanin_occurrences(*up1_slot, 0), 1);
    EXPECT_EQ(fanin_occurrences(*up1_slot, 1), 0);
    EXPECT_EQ(fanin_occurrences(*up1_slot, 4), 1);
    EXPECT_EQ(fanin_occurrences(*up1_slot, 6), 1);
    EXPECT_EQ(fanin_occurrences(*up1_slot, 7), 1);

    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, mi_update, 9, kHDefault, protocol_ok), 8);
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, li_update, 9, kHDefault, protocol_ok), 8);
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, oi, 9, kHDefault, protocol_ok), 8);
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, out_view, 9, kHDefault, protocol_ok), 1);
    EXPECT_TRUE(protocol_ok);

    EXPECT_EQ(worker_->occupied_count, 2);
    EXPECT_EQ(std::memcmp(private_map_before.data(), &worker_->map, private_map_before.size()), 0);
}

TEST_F(FdwicSharedSubmitWiringTest, ZeroEntryKernelWinnerAdvancesCommitAndBuilds) {
    L0TaskArgs args;
    MixedKernels mixed;
    mixed.aiv0_kernel_id = 8;
    const std::vector<std::byte> map_before = snapshot_shared_tensor_map();

    const DistCompeteFirstTicket ticket = dist_submit_compete_first_begin(nullptr, mixed);
    ASSERT_EQ(ticket.task_id, 0);
    ASSERT_EQ(ticket.won, 1);
    (void)dist_submit_compete_first_finish(nullptr, mixed, ticket, args);

    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.shared_tensor_map.committed_tasks.v, 1);
    EXPECT_EQ(
        std::memcmp(
            map_before.data() + sizeof(g_dist.shared_tensor_map.committed_tasks.v),
            reinterpret_cast<const std::byte *>(&g_dist.shared_tensor_map) +
                sizeof(g_dist.shared_tensor_map.committed_tasks.v),
            map_before.size() - sizeof(g_dist.shared_tensor_map.committed_tasks.v)
        ),
        0
    );
    ASSERT_EQ(worker_->occupied_count, 1);
    EXPECT_TRUE(worker_->slots[0].built);
    EXPECT_EQ(worker_->slots[0].task_id, 0);
    EXPECT_EQ(worker_->slots[0].fanin_count, 0);
}

TEST_F(FdwicSharedSubmitWiringTest, ZeroEntryJointWinnerCommitsAndPublishesJointWork) {
    reset_worker(CoreType::AIC, LANE_AIC);
    L0TaskArgs args;
    MixedKernels mixed;
    mixed.aic_kernel_id = 12;
    mixed.aiv0_kernel_id = 13;

    const DistCompeteFirstTicket ticket = dist_submit_compete_first_begin(nullptr, mixed);
    ASSERT_EQ(ticket.task_id, 0);
    ASSERT_EQ(ticket.won, 1);
    ASSERT_EQ(ticket.joint, 1);
    ASSERT_EQ(ticket.joint_init, 1);
    (void)dist_submit_compete_first_finish(nullptr, mixed, ticket, args);

    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.shared_tensor_map.committed_tasks.v, 1);
    EXPECT_EQ(g_dist.blocks[0].any_pub, 1);
    int32_t published = 0;
    for (int32_t index = 0; index < kPrivateSlots; ++index) {
        if (g_dist.blocks[0].slots[index].state.v == kWonStatePublished) {
            ++published;
        }
    }
    EXPECT_EQ(published, 1);
    ASSERT_EQ(worker_->occupied_count, 1);
    EXPECT_TRUE(worker_->slots[0].built);
    EXPECT_TRUE(worker_->slots[0].is_multicore);
    EXPECT_EQ(worker_->slots[0].task_id, 0);
}

TEST_F(FdwicSharedSubmitWiringTest, AllocWinnerCommitsAnEmptyMapTransactionBeforeCompletion) {
    L0TaskArgs args;
    const std::vector<std::byte> map_before = snapshot_shared_tensor_map();

    const DistCompeteFirstTicket ticket = dist_alloc_compete_first_begin(nullptr);
    ASSERT_EQ(ticket.task_id, 0);
    ASSERT_EQ(ticket.won, 1);
    (void)dist_alloc_compete_first_finish(nullptr, ticket, args);

    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.shared_tensor_map.committed_tasks.v, 1);
    EXPECT_EQ(task_cell(0).flag, 1);
    EXPECT_EQ(g_dist.frontier, 0);
    EXPECT_EQ(
        std::memcmp(
            map_before.data() + sizeof(g_dist.shared_tensor_map.committed_tasks.v),
            reinterpret_cast<const std::byte *>(&g_dist.shared_tensor_map) +
                sizeof(g_dist.shared_tensor_map.committed_tasks.v),
            map_before.size() - sizeof(g_dist.shared_tensor_map.committed_tasks.v)
        ),
        0
    );
}

TEST_F(FdwicSharedSubmitWiringTest, LegacyKernelUsesTheSameSharedTransactionBoundary) {
    const Tensor output = make_existing_tensor(0x380000);
    L0TaskArgs args;
    args.add_output(output);
    MixedKernels mixed;
    mixed.aiv0_kernel_id = 6;

    (void)dist_submit_impl(nullptr, mixed, args);

    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.shared_tensor_map.committed_tasks.v, 1);
    bool protocol_ok = false;
    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, output, 1, kHDefault, protocol_ok), 0);
    EXPECT_TRUE(protocol_ok);
    ASSERT_EQ(worker_->occupied_count, 1);
    EXPECT_TRUE(worker_->slots[0].built);
    EXPECT_EQ(worker_->slots[0].task_id, 0);
}

TEST_F(FdwicSharedSubmitWiringTest, LegacyAllocUsesTheSameEmptyTransactionBoundary) {
    L0TaskArgs args;

    (void)dist_alloc_tensors(nullptr, args);

    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.shared_tensor_map.committed_tasks.v, 1);
    EXPECT_EQ(task_cell(0).flag, 1);
    EXPECT_EQ(task_cell(0).vend, 0);
    EXPECT_EQ(g_dist.frontier, 0);
}

TEST_F(FdwicSharedSubmitWiringTest, KernelLoserDoesNotReadOrWriteEitherTensorMap) {
    const Tensor input = make_existing_tensor(0x400000);
    const Tensor output = make_existing_tensor(0x400000);
    L0TaskArgs args;
    args.add_input(input);
    args.add_output(output);
    MixedKernels mixed;
    mixed.aiv0_kernel_id = 9;
    g_dist.vector_cursor[0].v = 0;
    g_dist.H = 0;

    worker_->map.alive_floor = -1;
    const uint32_t input_bucket = dist_tensor_map_hash(input.buffer.addr);
    worker_->map.bucket_heads[input_bucket] = 0;
    worker_->map.bucket_tails[input_bucket] = 1;
    MapEntry &stale_entry = worker_->map.entries[dist_private_tensor_map_slot_index(input_bucket, 0)];
    stale_entry.buf_addr = input.buffer.addr;
    stale_entry.lo = 0;
    stale_entry.hi = sizeof(float);
    stale_entry.producer = -2;
    g_dist.shared_tensor_map.buckets[input_bucket].tail.v = 1;
    std::vector<std::byte> private_map_before(sizeof(worker_->map));
    std::memcpy(private_map_before.data(), &worker_->map, private_map_before.size());
    const std::vector<std::byte> shared_map_before = snapshot_shared_tensor_map();

    const DistCompeteFirstTicket ticket = dist_submit_compete_first_begin(nullptr, mixed);
    ASSERT_EQ(ticket.task_id, 0);
    ASSERT_EQ(ticket.won, 0);
    (void)dist_submit_compete_first_finish(nullptr, mixed, ticket, args);

    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(std::memcmp(shared_map_before.data(), &g_dist.shared_tensor_map, shared_map_before.size()), 0);
    EXPECT_EQ(std::memcmp(private_map_before.data(), &worker_->map, private_map_before.size()), 0);
    EXPECT_EQ(worker_->occupied_count, 0);
}

TEST_F(FdwicSharedSubmitWiringTest, GenericScalarLookupFailsClosedWithoutReadingEitherMap) {
    uint32_t scalar = 0x12345678U;
    const Tensor input = make_existing_tensor(reinterpret_cast<uintptr_t>(&scalar));
    const SharedTensorMapValue entry = dist_shared_tensor_map_make_value(input, 0);
    publish_seed_task(&entry, 1, /*task=*/0, kHDefault);
    worker_->local_index = 1;
    task_cell(0).flag = 1;
    worker_->map.alive_floor = 17;
    worker_->map.bucket_heads[0] = 23;
    worker_->map.bucket_tails[0] = 23;
    const std::vector<std::byte> shared_map_before = snapshot_shared_tensor_map();
    std::vector<std::byte> private_map_before(sizeof(worker_->map));
    std::memcpy(private_map_before.data(), &worker_->map, private_map_before.size());

    const uint32_t indices[] = {0};
    EXPECT_EQ(dist_get_tensor_data_impl(nullptr, input, /*ndims=*/1, indices), 0);

    EXPECT_EQ(g_dist.fatal, 1);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_TENSORMAP_PROTOCOL);
    EXPECT_EQ(scalar, 0x12345678U);
    EXPECT_EQ(std::memcmp(shared_map_before.data(), &g_dist.shared_tensor_map, shared_map_before.size()), 0);
    EXPECT_EQ(std::memcmp(private_map_before.data(), &worker_->map, private_map_before.size()), 0);

    g_dist.fatal = 0;
    g_dist.error_code = PTO2_ERROR_NONE;
    dist_set_tensor_data_impl(nullptr, input, /*ndims=*/1, indices, 0xA5A5A5A5U);
    EXPECT_EQ(g_dist.fatal, 1);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_TENSORMAP_PROTOCOL);
    EXPECT_EQ(scalar, 0x12345678U);
}

TEST_F(FdwicSharedSubmitWiringTest, WinnerLookupProtocolFailurePrecedesPublishAndBuild) {
    const Tensor input = make_existing_tensor(0x490000);
    const Tensor output = make_existing_tensor(0x4A0000);
    const uint32_t corrupt_bucket = dist_tensor_map_hash(input.buffer.addr);
    g_dist.shared_tensor_map.buckets[corrupt_bucket].tail.v = 1;
    const std::vector<std::byte> map_before = snapshot_shared_tensor_map();

    L0TaskArgs args;
    args.add_input(input);
    args.add_output(output);
    MixedKernels mixed;
    mixed.aiv0_kernel_id = 10;

    const DistCompeteFirstTicket ticket = dist_submit_compete_first_begin(nullptr, mixed);
    ASSERT_EQ(ticket.task_id, 0);
    ASSERT_EQ(ticket.won, 1);
    (void)dist_submit_compete_first_finish(nullptr, mixed, ticket, args);

    EXPECT_EQ(g_dist.fatal, 1);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_TENSORMAP_PROTOCOL);
    EXPECT_EQ(worker_->local_index, kFlagCap);
    EXPECT_EQ(std::memcmp(map_before.data(), &g_dist.shared_tensor_map, map_before.size()), 0);
    EXPECT_EQ(worker_->occupied_count, 0);
    EXPECT_EQ(task_cell(0).flag, 0);
    EXPECT_EQ(task_cell(0).vend, 0);
    EXPECT_EQ(g_dist.frontier, -1);
}

TEST_F(FdwicSharedSubmitWiringTest, CommitAheadOfWinnerIsAProtocolFailureBeforeBuild) {
    g_dist.shared_tensor_map.committed_tasks.v = 1;
    L0TaskArgs args;
    MixedKernels mixed;
    mixed.aiv0_kernel_id = 11;

    const DistCompeteFirstTicket ticket = dist_submit_compete_first_begin(nullptr, mixed);
    ASSERT_EQ(ticket.task_id, 0);
    ASSERT_EQ(ticket.won, 1);
    (void)dist_submit_compete_first_finish(nullptr, mixed, ticket, args);

    EXPECT_EQ(g_dist.shared_tensor_map.committed_tasks.v, 1);
    EXPECT_EQ(worker_->occupied_count, 0);
    EXPECT_EQ(worker_->local_index, kFlagCap);
    EXPECT_EQ(g_dist.fatal, 1);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_TENSORMAP_PROTOCOL);
    EXPECT_EQ(task_cell(0).flag, 0);
    EXPECT_EQ(task_cell(0).vend, 0);
    EXPECT_EQ(g_dist.frontier, -1);
}

TEST_F(FdwicSharedSubmitWiringTest, AllocCommitAheadSuppressesImmediateCompletion) {
    g_dist.shared_tensor_map.committed_tasks.v = 1;
    L0TaskArgs args;

    const DistCompeteFirstTicket ticket = dist_alloc_compete_first_begin(nullptr);
    ASSERT_EQ(ticket.task_id, 0);
    ASSERT_EQ(ticket.won, 1);
    (void)dist_alloc_compete_first_finish(nullptr, ticket, args);

    EXPECT_EQ(g_dist.shared_tensor_map.committed_tasks.v, 1);
    EXPECT_EQ(worker_->local_index, kFlagCap);
    EXPECT_EQ(g_dist.fatal, 1);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_TENSORMAP_PROTOCOL);
    EXPECT_EQ(task_cell(0).flag, 0);
    EXPECT_EQ(task_cell(0).vend, 0);
    EXPECT_EQ(g_dist.frontier, -1);
}

TEST_F(FdwicSharedSubmitWiringTest, JointWinnerCapacityFailurePrecedesWonDepositAndBuild) {
    reset_worker(CoreType::AIC, LANE_AIC);
    const Tensor output = make_existing_tensor(0x500000);
    fill_one_bucket(output, kMapBucketCapacity);
    g_dist.H = kFlagCap - 1;
    worker_->local_index = kMapBucketCapacity;

    L0TaskArgs args;
    args.add_output(output);
    MixedKernels mixed;
    mixed.aic_kernel_id = 13;
    mixed.aiv0_kernel_id = 14;

    const DistCompeteFirstTicket ticket = dist_submit_compete_first_begin(nullptr, mixed);
    ASSERT_EQ(ticket.task_id, kMapBucketCapacity);
    ASSERT_EQ(ticket.won, 1);
    ASSERT_EQ(ticket.joint, 1);
    ASSERT_EQ(ticket.joint_init, 1);
    (void)dist_submit_compete_first_finish(nullptr, mixed, ticket, args);

    EXPECT_EQ(g_dist.shared_tensor_map.committed_tasks.v, kMapBucketCapacity);
    EXPECT_EQ(g_dist.blocks[0].any_pub, 0);
    for (int32_t index = 0; index < kPrivateSlots; ++index) {
        EXPECT_EQ(g_dist.blocks[0].slots[index].state.v, kWonStateFree);
    }
    EXPECT_EQ(worker_->occupied_count, 0);
    EXPECT_EQ(worker_->local_index, kFlagCap);
    EXPECT_EQ(g_dist.fatal, 1);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_TENSORMAP_CAPACITY);
    EXPECT_EQ(task_cell(kMapBucketCapacity).flag, 0);
    EXPECT_EQ(task_cell(kMapBucketCapacity).vend, 0);
    EXPECT_EQ(g_dist.frontier, -1);
}

}  // namespace
