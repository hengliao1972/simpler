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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>

// The CPU-sim platform and public runtime types both provide their normal
// SPIN_WAIT_HINT definitions. Include them first, then replace only this
// translation unit's final definition with the observation hook below.
#include "inner_kernel.h"
#undef SPIN_WAIT_HINT
#include "pto_runtime2_types.h"

namespace fdwic_shared_multiworker_test {

std::atomic<uint32_t> g_observed_turn_wait_spins{0};
thread_local bool g_observe_turn_wait = false;

void spin_wait_hint() {
    if (g_observe_turn_wait) {
        g_observed_turn_wait_spins.fetch_add(1, std::memory_order_release);
    }
    std::this_thread::yield();
}

}  // namespace fdwic_shared_multiworker_test

// This is a test-only observation seam around the production wait loop. It
// does not replace Claim, TensorMap publication, fanin collection, or Build.
#undef SPIN_WAIT_HINT
#define SPIN_WAIT_HINT() ::fdwic_shared_multiworker_test::spin_wait_hint()
#include "dist_engine/aicore/dist_engine.cpp"  // NOLINT(build/include)
#include "dist_engine/aicpu/shared_tensor_map_init.h"
#undef SPIN_WAIT_HINT

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

int32_t built_slot_count(const DistCore &worker) {
    int32_t count = 0;
    for (int32_t index = 0; index < kPrivateSlots; ++index) {
        if (worker.slots[index].occupied && worker.slots[index].built) {
            ++count;
        }
    }
    return count;
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

class FdwicSharedMultiworkerTest : public ::testing::Test {
protected:
    void SetUp() override {
        static_assert(PTO_FDWIC_SHARED_MAP == 1);
        static_assert(!kFdwicCompiledBackendReady);

        fdwic_shared_multiworker_test::g_observed_turn_wait_spins.store(0, std::memory_order_relaxed);
        g_dist_ptr = &g_dist_fallback;
        dist_shared_tensor_map_reset(g_dist.shared_tensor_map);

        dist_core_reset(*task0_worker_, CoreType::AIV, /*block=*/0, LANE_AIV0);
        dist_core_reset(*task1_worker_, CoreType::AIV, /*block=*/0, LANE_AIV1);
        task0_worker_->core_idx = 1;
        task1_worker_->core_idx = 2;
        g_self = task0_worker_.get();

        g_dist.H = kHDefault;
        g_dist.heap_base = nullptr;
        g_dist.heap_size = 0;
        g_dist.runtime = &runtime_;
        g_dist.num_workers = 2;
        g_dist.num_blocks = 1;
        g_dist.frontier = -1;
        g_dist.fatal = 0;
        g_dist.error_code = PTO2_ERROR_NONE;
        for (int32_t shard = 0; shard < kCursorShards; ++shard) {
            g_dist.cube_cursor[shard].v = -1;
            g_dist.vector_cursor[shard].v = -1;
            g_dist.alloc_cursor[shard].v = -1;
        }
        reset_task_cell(0);
        reset_task_cell(1);
    }

    void TearDown() override {
        fdwic_shared_multiworker_test::g_observe_turn_wait = false;
        g_self = nullptr;
        g_dist_ptr = nullptr;
    }

    Runtime runtime_;
    std::unique_ptr<DistCore> task0_worker_ = std::make_unique<DistCore>();
    std::unique_ptr<DistCore> task1_worker_ = std::make_unique<DistCore>();
};

TEST_F(FdwicSharedMultiworkerTest, FutureTurnWaitsForPriorCommitThenBuildsWithDependency) {
    const Tensor inout = make_existing_tensor(0x600000);
    L0TaskArgs task0_args;
    task0_args.add_inout(inout);
    L0TaskArgs task0_replay_args;
    task0_replay_args.add_inout(inout);
    L0TaskArgs task1_args;
    task1_args.add_inout(inout);
    MixedKernels mixed;
    mixed.aiv0_kernel_id = 21;

    // Holding task0 between Begin and Finish is the deterministic publication
    // latch: it has won Claim, but committed_tasks must remain zero.
    const DistCompeteFirstTicket task0_ticket = dist_submit_compete_first_begin(nullptr, mixed);
    ASSERT_EQ(task0_ticket.task_id, 0);
    ASSERT_EQ(task0_ticket.won, 1);
    ASSERT_EQ(atomic_load(g_dist.shared_tensor_map.committed_tasks.v), 0);

    DistCompeteFirstTicket task0_loser_ticket{};
    DistCompeteFirstTicket task1_ticket{};
    std::atomic<bool> task1_finish_returned{false};
    std::exception_ptr task1_error;
    std::thread task1_thread([&]() {
        try {
            g_self = task1_worker_.get();
            // AIV1 starts from the same replay position. Its task0 Claim loses
            // to AIV0, then the normal per-worker sequence advances to task1.
            task0_loser_ticket = dist_submit_compete_first_begin(nullptr, mixed);
            (void)dist_submit_compete_first_finish(nullptr, mixed, task0_loser_ticket, task0_replay_args);
            task1_ticket = dist_submit_compete_first_begin(nullptr, mixed);
            fdwic_shared_multiworker_test::g_observe_turn_wait = true;
            (void)dist_submit_compete_first_finish(nullptr, mixed, task1_ticket, task1_args);
            fdwic_shared_multiworker_test::g_observe_turn_wait = false;
        } catch (...) {
            task1_error = std::current_exception();
        }
        task1_finish_returned.store(true, std::memory_order_release);
        g_self = nullptr;
    });

    const auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (fdwic_shared_multiworker_test::g_observed_turn_wait_spins.load(std::memory_order_acquire) == 0 &&
           !task1_finish_returned.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < wait_deadline) {
        std::this_thread::yield();
    }

    // Snapshot the blocked state before releasing task0. These values prove
    // task1 did not publish or Build while committed_tasks was behind.
    const bool task1_entered_turn_wait =
        fdwic_shared_multiworker_test::g_observed_turn_wait_spins.load(std::memory_order_acquire) != 0;
    const bool task1_returned_before_release = task1_finish_returned.load(std::memory_order_acquire);
    const int64_t committed_before_release = atomic_load(g_dist.shared_tensor_map.committed_tasks.v);
    const int32_t task0_built_before_release = built_slot_count(*task0_worker_);
    const int32_t task1_built_before_release = built_slot_count(*task1_worker_);

    // Publishing task0 advances the exact turn to one. The waiting production
    // Finish must then collect producer 0, publish task1, and Build exactly once.
    (void)dist_submit_compete_first_finish(nullptr, mixed, task0_ticket, task0_args);
    task1_thread.join();

    EXPECT_TRUE(task1_entered_turn_wait);
    EXPECT_FALSE(task1_returned_before_release);
    EXPECT_EQ(committed_before_release, 0);
    EXPECT_EQ(task0_built_before_release, 0);
    EXPECT_EQ(task1_built_before_release, 0);

    if (task1_error != nullptr) {
        try {
            std::rethrow_exception(task1_error);
        } catch (const std::exception &error) {
            FAIL() << "task1 worker threw: " << error.what();
        } catch (...) {
            FAIL() << "task1 worker threw a non-standard exception";
        }
    }

    EXPECT_EQ(task0_loser_ticket.task_id, 0);
    EXPECT_EQ(task0_loser_ticket.won, 0);
    EXPECT_EQ(task1_ticket.task_id, 1);
    EXPECT_EQ(task1_ticket.won, 1);
    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_NONE);
    EXPECT_EQ(atomic_load(g_dist.shared_tensor_map.committed_tasks.v), 2);

    EXPECT_EQ(task0_worker_->occupied_count, 1);
    EXPECT_EQ(task1_worker_->occupied_count, 1);
    EXPECT_EQ(built_slot_count(*task0_worker_), 1);
    EXPECT_EQ(built_slot_count(*task1_worker_), 1);

    const RingSlot *task0_slot = find_built_slot(*task0_worker_, 0);
    const RingSlot *task1_slot = find_built_slot(*task1_worker_, 1);
    ASSERT_NE(task0_slot, nullptr);
    ASSERT_NE(task1_slot, nullptr);
    EXPECT_EQ(task0_slot->fanin_count, 0);
    ASSERT_EQ(task1_slot->fanin_count, 1);
    EXPECT_EQ(task1_slot->fanin[0], 0);
}

}  // namespace
