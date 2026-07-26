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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

// The CPU-sim platform and public runtime types both provide their normal
// SPIN_WAIT_HINT definitions. Include them first, then replace only this
// translation unit's final definition with the observation hook below.
#include "inner_kernel.h"
#undef SPIN_WAIT_HINT
#include "pto_runtime2_types.h"

namespace fdwic_shared_multiworker_test {

std::atomic<uint32_t> g_observed_turn_wait_spins{0};
std::atomic<uint32_t> g_spins_after_remote_fatal{0};
std::atomic<bool> g_remote_fatal_published{false};
std::atomic<uint32_t> g_single_lane_kernel_calls{0};
thread_local bool g_observe_turn_wait = false;
thread_local bool g_limit_spins_after_remote_fatal = false;

constexpr uint32_t kPostFatalSpinLimit = 1024;

void spin_wait_hint() {
    if (g_observe_turn_wait) {
        g_observed_turn_wait_spins.fetch_add(1, std::memory_order_release);
    }
    if (g_limit_spins_after_remote_fatal && g_remote_fatal_published.load(std::memory_order_acquire) &&
        g_spins_after_remote_fatal.fetch_add(1, std::memory_order_acq_rel) >= kPostFatalSpinLimit) {
        throw std::runtime_error("production wait did not consume the remote fatal");
    }
    std::this_thread::yield();
}

void count_single_lane_kernel(int64_t *) { g_single_lane_kernel_calls.fetch_add(1, std::memory_order_relaxed); }

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

int32_t occupied_slot_count(const DistCore &worker) {
    int32_t count = 0;
    for (int32_t index = 0; index < kPrivateSlots; ++index) {
        if (worker.slots[index].occupied) {
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
        fdwic_shared_multiworker_test::g_spins_after_remote_fatal.store(0, std::memory_order_relaxed);
        fdwic_shared_multiworker_test::g_remote_fatal_published.store(false, std::memory_order_relaxed);
        fdwic_shared_multiworker_test::g_single_lane_kernel_calls.store(0, std::memory_order_relaxed);
        g_dist_ptr = &g_dist_fallback;
        dist_shared_tensor_map_reset(g_dist.shared_tensor_map);

        dist_core_reset(*aic_worker_, CoreType::AIC, /*block=*/0, LANE_AIC);
        dist_core_reset(*task0_worker_, CoreType::AIV, /*block=*/0, LANE_AIV0);
        dist_core_reset(*task1_worker_, CoreType::AIV, /*block=*/0, LANE_AIV1);
        aic_worker_->core_idx = 0;
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
        g_dist.blocks[0].any_pub = 0;
        for (int32_t index = 0; index < kPrivateSlots; ++index) {
            WonSlot &slot = g_dist.blocks[0].slots[index];
            slot.state.v = kWonStateFree;
            slot.remaining.v = 0;
            for (int32_t lane = LANE_AIC; lane <= LANE_AIV1; ++lane) {
                slot.drained[lane].v = kDrainedClaimed;
                slot.lane[lane].present = false;
            }
        }
        for (int32_t shard = 0; shard < kCursorShards; ++shard) {
            g_dist.cube_cursor[shard].v = -1;
            g_dist.vector_cursor[shard].v = -1;
            g_dist.alloc_cursor[shard].v = -1;
        }
        for (int32_t task = 0; task < 16; ++task) {
            reset_task_cell(task);
        }
        for (int32_t group = 0; group < kFinalBarrierGroups; ++group) {
            g_dist.final_barrier.leaf_arrivals[group].v = 0;
            g_dist.final_barrier.leaf_arrivals[group].expected = 0;
            g_dist.final_barrier.leaf_releases[group].v = 0;
        }
        g_dist.final_barrier.root_arrival.v = 0;
        g_dist.final_barrier.root_arrival.expected = 0;
        g_dist.final_barrier.root_release.v = 0;
        g_fdwic_joint_submit_seen = false;
        g_skip_exec = false;
    }

    void TearDown() override {
        fdwic_shared_multiworker_test::g_observe_turn_wait = false;
        fdwic_shared_multiworker_test::g_limit_spins_after_remote_fatal = false;
        g_self = nullptr;
        g_dist_ptr = nullptr;
    }

    Runtime runtime_;
    std::unique_ptr<DistCore> aic_worker_ = std::make_unique<DistCore>();
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

TEST_F(FdwicSharedMultiworkerTest, RemoteFatalInterruptsSlotCapacityWait) {
    // Two permanently blocked slots reach the self-owned capacity threshold.
    // The current task may complete its map transaction, but it must not spin
    // forever or Build after another worker publishes a terminal failure.
    constexpr int32_t kBlockedProducer = 7;
    for (int32_t index = 0; index < kPrivateSlots - kWonReserve; ++index) {
        RingSlot &slot = task0_worker_->slots[index];
        slot.occupied = true;
        slot.built = true;
        slot.task_id = 100 + index;
        slot.fanin_count = 1;
        slot.fanin[0] = kBlockedProducer;
    }
    task0_worker_->occupied_count = kPrivateSlots - kWonReserve;

    L0TaskArgs args;
    MixedKernels mixed;
    mixed.aiv0_kernel_id = 22;
    DistCompeteFirstTicket ticket{};
    std::exception_ptr worker_error;
    std::atomic<bool> worker_returned{false};
    std::thread worker([&]() {
        try {
            g_self = task0_worker_.get();
            ticket = dist_submit_compete_first_begin(nullptr, mixed);
            fdwic_shared_multiworker_test::g_observe_turn_wait = true;
            fdwic_shared_multiworker_test::g_limit_spins_after_remote_fatal = true;
            (void)dist_submit_compete_first_finish(nullptr, mixed, ticket, args);
            fdwic_shared_multiworker_test::g_limit_spins_after_remote_fatal = false;
            fdwic_shared_multiworker_test::g_observe_turn_wait = false;
        } catch (...) {
            worker_error = std::current_exception();
        }
        worker_returned.store(true, std::memory_order_release);
        g_self = nullptr;
    });

    const auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (fdwic_shared_multiworker_test::g_observed_turn_wait_spins.load(std::memory_order_acquire) == 0 &&
           std::chrono::steady_clock::now() < wait_deadline) {
        std::this_thread::yield();
    }
    const bool entered_capacity_wait =
        fdwic_shared_multiworker_test::g_observed_turn_wait_spins.load(std::memory_order_acquire) != 0;
    EXPECT_FALSE(worker_returned.load(std::memory_order_acquire));
    set_fatal_code(PTO2_ERROR_EXPLICIT_ORCH_FATAL);
    fdwic_shared_multiworker_test::g_remote_fatal_published.store(true, std::memory_order_release);
    worker.join();

    EXPECT_TRUE(entered_capacity_wait);
    EXPECT_TRUE(worker_returned.load(std::memory_order_acquire));
    if (worker_error != nullptr) {
        try {
            std::rethrow_exception(worker_error);
        } catch (const std::exception &error) {
            FAIL() << "slot-capacity waiter threw: " << error.what();
        } catch (...) {
            FAIL() << "slot-capacity waiter threw a non-standard exception";
        }
    }
    EXPECT_EQ(ticket.task_id, 0);
    EXPECT_EQ(ticket.won, 1);
    EXPECT_EQ(g_dist.fatal, 1);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_EXPLICIT_ORCH_FATAL);
    EXPECT_EQ(atomic_load(g_dist.shared_tensor_map.committed_tasks.v), 1);
    EXPECT_EQ(task0_worker_->local_index, kFlagCap);
    EXPECT_EQ(task0_worker_->occupied_count, kPrivateSlots - kWonReserve);
    EXPECT_EQ(task_cell(0).flag, 0);
    EXPECT_EQ(task_cell(0).vend, 0);
    EXPECT_EQ(g_dist.frontier, -1);
}

TEST_F(FdwicSharedMultiworkerTest, RemoteFatalInterruptsIncompleteFinalBarrier) {
    // Only one of two expected workers enters the final barrier. A remote fatal
    // means the absent worker will skip FinalDrain, so this worker must leave
    // without waiting for an impossible root/leaf release.
    dist_core_reset(*task0_worker_, CoreType::AIC, /*block=*/0, LANE_AIC);
    task0_worker_->core_idx = 0;
    g_dist.final_barrier.leaf_arrivals[0].expected = 2;
    g_dist.final_barrier.root_arrival.expected = 1;

    std::exception_ptr worker_error;
    std::atomic<bool> worker_returned{false};
    std::thread worker([&]() {
        try {
            g_self = task0_worker_.get();
            fdwic_shared_multiworker_test::g_observe_turn_wait = true;
            fdwic_shared_multiworker_test::g_limit_spins_after_remote_fatal = true;
            dist_submit_drain_to_completion(task0_worker_.get());
            fdwic_shared_multiworker_test::g_limit_spins_after_remote_fatal = false;
            fdwic_shared_multiworker_test::g_observe_turn_wait = false;
        } catch (...) {
            worker_error = std::current_exception();
        }
        worker_returned.store(true, std::memory_order_release);
        g_self = nullptr;
    });

    const auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (fdwic_shared_multiworker_test::g_observed_turn_wait_spins.load(std::memory_order_acquire) == 0 &&
           std::chrono::steady_clock::now() < wait_deadline) {
        std::this_thread::yield();
    }
    const bool entered_final_barrier_wait =
        fdwic_shared_multiworker_test::g_observed_turn_wait_spins.load(std::memory_order_acquire) != 0;
    EXPECT_FALSE(worker_returned.load(std::memory_order_acquire));
    set_fatal_code(PTO2_ERROR_EXPLICIT_ORCH_FATAL);
    fdwic_shared_multiworker_test::g_remote_fatal_published.store(true, std::memory_order_release);
    worker.join();

    EXPECT_TRUE(entered_final_barrier_wait);
    EXPECT_TRUE(worker_returned.load(std::memory_order_acquire));
    if (worker_error != nullptr) {
        try {
            std::rethrow_exception(worker_error);
        } catch (const std::exception &error) {
            FAIL() << "final-barrier waiter threw: " << error.what();
        } catch (...) {
            FAIL() << "final-barrier waiter threw a non-standard exception";
        }
    }
    EXPECT_EQ(g_dist.fatal, 1);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_EXPLICIT_ORCH_FATAL);
    EXPECT_EQ(task0_worker_->local_index, kFlagCap);
    EXPECT_EQ(g_dist.final_barrier.leaf_arrivals[0].v, 1);
    EXPECT_EQ(g_dist.final_barrier.root_arrival.v, 0);
    EXPECT_EQ(g_dist.final_barrier.root_release.v, 0);
    EXPECT_EQ(g_dist.final_barrier.leaf_releases[0].v, 0);
}

TEST_F(FdwicSharedMultiworkerTest, SingleLaneKernelExecutesOnceAndCompletesThroughThreeWorkerFinalDrain) {
    constexpr int32_t kKernelId = 23;
    alignas(PTO2_PACKED_OUTPUT_ALIGN) std::array<uint8_t, 4096> heap{};
    const uint32_t shape[1] = {1};
    TensorCreateInfo output_info(shape, 1, DataType::FLOAT32);
    L0TaskArgs args;
    args.add_output(output_info);

    const ArgDirection signature[] = {ArgDirection::OUT};
    std::vector<uint8_t> callable_storage =
        make_callable<CORE_MAX_TENSOR_ARGS>(signature, 1, nullptr, /*binary_size=*/0);
    CoreCallable *callable = reinterpret_cast<CoreCallable *>(callable_storage.data());
    callable->set_resolved_addr(reinterpret_cast<uint64_t>(&fdwic_shared_multiworker_test::count_single_lane_kernel));
    runtime_.func_id_to_addr_[kKernelId] = reinterpret_cast<uint64_t>(callable);

    g_dist.heap_base = heap.data();
    g_dist.heap_size = heap.size();
    g_dist.num_workers = 3;
    g_dist.num_blocks = 1;
    g_dist.final_barrier.leaf_arrivals[0].expected = 3;
    g_dist.final_barrier.root_arrival.expected = 1;

    MixedKernels mixed;
    mixed.aiv0_kernel_id = kKernelId;

    // All three physical lanes replay task 0. AIV0 wins deterministically;
    // AIC is ineligible, and AIV1 observes the already-published claim.
    g_self = task0_worker_.get();
    const DistCompeteFirstTicket aiv0_ticket = dist_submit_compete_first_begin(nullptr, mixed);
    (void)dist_submit_compete_first_finish(nullptr, mixed, aiv0_ticket, args);

    g_self = aic_worker_.get();
    const DistCompeteFirstTicket aic_ticket = dist_submit_compete_first_begin(nullptr, mixed);
    (void)dist_submit_compete_first_finish(nullptr, mixed, aic_ticket, args);

    g_self = task1_worker_.get();
    const DistCompeteFirstTicket aiv1_ticket = dist_submit_compete_first_begin(nullptr, mixed);
    (void)dist_submit_compete_first_finish(nullptr, mixed, aiv1_ticket, args);

    ASSERT_EQ(aiv0_ticket.task_id, 0);
    ASSERT_EQ(aiv0_ticket.won, 1);
    ASSERT_EQ(aiv0_ticket.joint, 0);
    ASSERT_EQ(aic_ticket.task_id, 0);
    ASSERT_EQ(aic_ticket.won, 0);
    ASSERT_EQ(aic_ticket.claim_attempted, 0);
    ASSERT_EQ(aiv1_ticket.task_id, 0);
    ASSERT_EQ(aiv1_ticket.won, 0);
    ASSERT_EQ(aiv1_ticket.claim_attempted, 1);

    EXPECT_EQ(fdwic_shared_multiworker_test::g_single_lane_kernel_calls.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(atomic_load(g_dist.shared_tensor_map.committed_tasks.v), 1);
    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_NONE);
    EXPECT_EQ(g_dist.blocks[0].any_pub, 0);
    EXPECT_EQ(aic_worker_->local_index, 1);
    EXPECT_EQ(task0_worker_->local_index, 1);
    EXPECT_EQ(task1_worker_->local_index, 1);
    EXPECT_EQ(aic_worker_->heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    EXPECT_EQ(task0_worker_->heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    EXPECT_EQ(task1_worker_->heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    EXPECT_EQ(aic_worker_->occupied_count, 0);
    ASSERT_EQ(task0_worker_->occupied_count, 1);
    EXPECT_EQ(task1_worker_->occupied_count, 0);
    const RingSlot *built = find_built_slot(*task0_worker_, 0);
    ASSERT_NE(built, nullptr);
    EXPECT_EQ(built->func_id, kKernelId);
    EXPECT_EQ(
        built->function_bin_addr, reinterpret_cast<uint64_t>(&fdwic_shared_multiworker_test::count_single_lane_kernel)
    );
    ASSERT_EQ(built->tensor_count, 1);
    EXPECT_EQ(built->tensors[0].buffer.addr, reinterpret_cast<uint64_t>(heap.data()));
    EXPECT_EQ(built->tensors[0].buffer.size, sizeof(float));
    EXPECT_EQ(task_cell(0).flag, 0);
    EXPECT_EQ(task_cell(0).vend, 0);
    EXPECT_EQ(g_dist.frontier, -1);

    std::atomic<uint32_t> ready_count{0};
    std::atomic<bool> start_drain{false};
    std::atomic<bool> aic_returned{false};
    std::atomic<bool> aiv0_returned{false};
    std::atomic<bool> aiv1_returned{false};
    std::exception_ptr aic_error;
    std::exception_ptr aiv0_error;
    std::exception_ptr aiv1_error;

    auto drain_worker = [&](DistCore *worker, std::atomic<bool> &returned, std::exception_ptr &error) {
        g_self = worker;
        ready_count.fetch_add(1, std::memory_order_release);
        while (!start_drain.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        try {
            dist_submit_drain_to_completion(worker);
        } catch (...) {
            error = std::current_exception();
        }
        returned.store(true, std::memory_order_release);
        g_self = nullptr;
    };

    std::thread aic_thread(drain_worker, aic_worker_.get(), std::ref(aic_returned), std::ref(aic_error));
    std::thread aiv0_thread(drain_worker, task0_worker_.get(), std::ref(aiv0_returned), std::ref(aiv0_error));
    std::thread aiv1_thread(drain_worker, task1_worker_.get(), std::ref(aiv1_returned), std::ref(aiv1_error));

    const auto ready_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (ready_count.load(std::memory_order_acquire) != 3 && std::chrono::steady_clock::now() < ready_deadline) {
        std::this_thread::yield();
    }
    const bool all_workers_ready = ready_count.load(std::memory_order_acquire) == 3;
    start_drain.store(true, std::memory_order_release);
    aic_thread.join();
    aiv0_thread.join();
    aiv1_thread.join();

    EXPECT_TRUE(all_workers_ready);
    EXPECT_TRUE(aic_returned.load(std::memory_order_acquire));
    EXPECT_TRUE(aiv0_returned.load(std::memory_order_acquire));
    EXPECT_TRUE(aiv1_returned.load(std::memory_order_acquire));
    EXPECT_EQ(aic_error, nullptr);
    EXPECT_EQ(aiv0_error, nullptr);
    EXPECT_EQ(aiv1_error, nullptr);

    EXPECT_EQ(fdwic_shared_multiworker_test::g_single_lane_kernel_calls.load(std::memory_order_relaxed), 1);
    EXPECT_EQ(aic_worker_->occupied_count, 0);
    EXPECT_EQ(task0_worker_->occupied_count, 0);
    EXPECT_EQ(task1_worker_->occupied_count, 0);
    EXPECT_EQ(occupied_slot_count(*aic_worker_), 0);
    EXPECT_EQ(occupied_slot_count(*task0_worker_), 0);
    EXPECT_EQ(occupied_slot_count(*task1_worker_), 0);
    EXPECT_EQ(built_slot_count(*aic_worker_), 0);
    EXPECT_EQ(built_slot_count(*task0_worker_), 0);
    EXPECT_EQ(built_slot_count(*task1_worker_), 0);
    EXPECT_EQ(aic_worker_->local_index, 1);
    EXPECT_EQ(task0_worker_->local_index, 1);
    EXPECT_EQ(task1_worker_->local_index, 1);
    EXPECT_EQ(task_cell(0).flag, 1);
    EXPECT_EQ(task_cell(0).vend, PTO2_PACKED_OUTPUT_ALIGN);
    EXPECT_EQ(g_dist.frontier, 0);
    EXPECT_EQ(atomic_load(g_dist.shared_tensor_map.committed_tasks.v), 1);
    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_NONE);

    EXPECT_EQ(g_dist.final_barrier.leaf_arrivals[0].v, 3);
    EXPECT_EQ(g_dist.final_barrier.root_arrival.v, 1);
    EXPECT_EQ(g_dist.final_barrier.root_release.v, 1);
    EXPECT_EQ(g_dist.final_barrier.leaf_releases[0].v, 1);
    for (int32_t group = 1; group < kFinalBarrierGroups; ++group) {
        EXPECT_EQ(g_dist.final_barrier.leaf_arrivals[group].v, 0);
        EXPECT_EQ(g_dist.final_barrier.leaf_releases[group].v, 0);
    }
}

}  // namespace
