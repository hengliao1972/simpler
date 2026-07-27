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
#include <cstddef>
#include <cstdint>
#include <cstring>
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
std::atomic<uint32_t> g_joint_kernel_entered[3]{};
std::atomic<uint32_t> g_joint_kernel_exited[3]{};
std::atomic<bool> g_joint_kernel_release[3]{};
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

void run_joint_kernel(int32_t lane) {
    g_joint_kernel_entered[lane].fetch_add(1, std::memory_order_release);
    while (!g_joint_kernel_release[lane].load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    g_joint_kernel_exited[lane].fetch_add(1, std::memory_order_release);
}

}  // namespace fdwic_shared_multiworker_test

// This is a test-only observation seam around the production wait loop. It
// does not replace Claim, TensorMap publication, fanin collection, or Build.
#undef SPIN_WAIT_HINT
#define SPIN_WAIT_HINT() ::fdwic_shared_multiworker_test::spin_wait_hint()
#include "dist_engine/aicore/dist_engine.cpp"  // NOLINT(build/include)
#include "dist_engine/aicpu/shared_tensor_map_init.h"
#undef SPIN_WAIT_HINT

namespace fdwic_shared_multiworker_test {

void count_joint_aic_kernel(int64_t *) { run_joint_kernel(LANE_AIC); }

void count_joint_aiv0_kernel(int64_t *) { run_joint_kernel(LANE_AIV0); }

void count_joint_aiv1_kernel(int64_t *) { run_joint_kernel(LANE_AIV1); }

}  // namespace fdwic_shared_multiworker_test

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

void install_test_callable(
    Runtime &runtime, int32_t kernel_id, void (*kernel)(int64_t *), std::vector<uint8_t> &storage
) {
    const ArgDirection signature[] = {ArgDirection::OUT};
    storage = make_callable<CORE_MAX_TENSOR_ARGS>(signature, 1, nullptr, /*binary_size=*/0);
    CoreCallable *callable = reinterpret_cast<CoreCallable *>(storage.data());
    callable->set_resolved_addr(reinterpret_cast<uint64_t>(kernel));
    runtime.func_id_to_addr_[kernel_id] = reinterpret_cast<uint64_t>(callable);
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

int32_t fanin_occurrences(const RingSlot &slot, int32_t producer) {
    int32_t count = 0;
    for (int32_t index = 0; index < slot.fanin_count; ++index) {
        if (slot.fanin[index] == producer) {
            ++count;
        }
    }
    return count;
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
        for (int32_t lane = LANE_AIC; lane <= LANE_AIV1; ++lane) {
            fdwic_shared_multiworker_test::g_joint_kernel_entered[lane].store(0, std::memory_order_relaxed);
            fdwic_shared_multiworker_test::g_joint_kernel_exited[lane].store(0, std::memory_order_relaxed);
            fdwic_shared_multiworker_test::g_joint_kernel_release[lane].store(false, std::memory_order_relaxed);
        }
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

    void publish_seed_task(const SharedTensorMapValue *entries, uint32_t count, int32_t task) {
        ASSERT_EQ(
            dist_shared_tensor_map_publish_task(g_dist.shared_tensor_map, entries, count, task, kHDefault),
            DistSharedTensorMapTaskPublishResult::Committed
        );
        ASSERT_EQ(atomic_load(g_dist.shared_tensor_map.committed_tasks.v), task + 1);
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

TEST_F(FdwicSharedMultiworkerTest, PaG2FutureFinalUpWaitsForRemoteFirstUpWriterCommit) {
    constexpr int32_t kUpKernelId = 25;
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

    // Seed only the exact-turn transactions for Alloc/QK/SF/PV. The task1
    // out_view entry is an impossible PA value used to detect an accidental
    // manual_dep lookup or register; it is not a model of QK output semantics.
    ASSERT_NO_FATAL_FAILURE(publish_seed_task(nullptr, 0, /*task=*/0));
    const SharedTensorMapValue manual_poison = dist_shared_tensor_map_make_value(out_view, /*producer=*/1);
    ASSERT_NO_FATAL_FAILURE(publish_seed_task(&manual_poison, 1, /*task=*/1));
    ASSERT_NO_FATAL_FAILURE(publish_seed_task(nullptr, 0, /*task=*/2));
    ASSERT_NO_FATAL_FAILURE(publish_seed_task(nullptr, 0, /*task=*/3));

    L0TaskArgs up0_args;
    up0_args.add_input(group0_mi, group0_li, group0_oi_new);
    up0_args.add_inout(mi_update, li_update, oi, out_view);
    up0_args.add_scalar(/*is_first=*/1, /*is_last=*/0);
    L0TaskArgs up1_args;
    up1_args.add_input(group1_mi, group1_li, group1_oi_new);
    up1_args.add_inout(mi_update, li_update, oi, out_view);
    up1_args.add_scalar(/*is_first=*/0, /*is_last=*/1);
    MixedKernels mixed;
    mixed.aiv0_kernel_id = kUpKernelId;

    std::vector<std::byte> task0_private_map_before(sizeof(task0_worker_->map));
    std::vector<std::byte> task1_private_map_before(sizeof(task1_worker_->map));
    std::memcpy(task0_private_map_before.data(), &task0_worker_->map, task0_private_map_before.size());
    std::memcpy(task1_private_map_before.data(), &task1_worker_->map, task1_private_map_before.size());

    // The existing sequential-replay gate proves local_index progression from
    // zero. This focused gate starts directly at the two real G2 UP identities
    // so both workers still perform production Claim and Finish.
    task0_worker_->local_index = 4;
    task1_worker_->local_index = 8;
    const DistCompeteFirstTicket up0_ticket = dist_submit_compete_first_begin(nullptr, mixed);
    ASSERT_EQ(up0_ticket.task_id, 4);
    ASSERT_EQ(up0_ticket.ready, 1);
    ASSERT_EQ(up0_ticket.claim_attempted, 1);
    ASSERT_EQ(up0_ticket.won, 1);
    ASSERT_EQ(atomic_load(g_dist.vector_cursor[0].v), 4);
    ASSERT_EQ(atomic_load(g_dist.shared_tensor_map.committed_tasks.v), 4);

    DistCompeteFirstTicket up1_ticket{};
    std::exception_ptr up1_error;
    std::atomic<bool> up1_finish_returned{false};
    std::thread up1_thread([&]() {
        try {
            g_self = task1_worker_.get();
            up1_ticket = dist_submit_compete_first_begin(nullptr, mixed);
            fdwic_shared_multiworker_test::g_observe_turn_wait = true;
            fdwic_shared_multiworker_test::g_limit_spins_after_remote_fatal = true;
            (void)dist_submit_compete_first_finish(nullptr, mixed, up1_ticket, up1_args);
        } catch (...) {
            up1_error = std::current_exception();
        }
        fdwic_shared_multiworker_test::g_limit_spins_after_remote_fatal = false;
        fdwic_shared_multiworker_test::g_observe_turn_wait = false;
        up1_finish_returned.store(true, std::memory_order_release);
        g_self = nullptr;
    });

    const auto wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (fdwic_shared_multiworker_test::g_observed_turn_wait_spins.load(std::memory_order_acquire) == 0 &&
           !up1_finish_returned.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < wait_deadline) {
        std::this_thread::yield();
    }

    // The observation hook is enabled only around task8 Finish. Together with
    // committed=4 and cursor=8, a positive count proves the future winner
    // reached the production exact-turn loop before task4 was released.
    const bool up1_entered_turn_wait =
        fdwic_shared_multiworker_test::g_observed_turn_wait_spins.load(std::memory_order_acquire) != 0;
    const bool up1_returned_before_release = up1_finish_returned.load(std::memory_order_acquire);
    const int64_t committed_before_release = atomic_load(g_dist.shared_tensor_map.committed_tasks.v);
    const int64_t vector_cursor_before_release = atomic_load(g_dist.vector_cursor[0].v);

    std::exception_ptr release_error;
    int64_t committed_after_up0 = -1;
    std::array<int32_t, 4> latest_after_up0 = {-1, -1, -1, -1};
    std::array<bool, 4> protocol_after_up0 = {false, false, false, false};
    std::array<DistSharedTensorMapTaskPublishResult, 3> filler_results = {
        DistSharedTensorMapTaskPublishResult::ProtocolError,
        DistSharedTensorMapTaskPublishResult::ProtocolError,
        DistSharedTensorMapTaskPublishResult::ProtocolError,
    };
    bool up1_waited_after_up0 = false;
    bool up1_returned_after_up0 = true;
    int64_t committed_while_waiting_after_up0 = -1;

    try {
        // task4 performs the real fanin/register/commit/Build path. Task8 must
        // remain blocked at turn 8 while tasks 5-7 are still absent.
        (void)dist_submit_compete_first_finish(nullptr, mixed, up0_ticket, up0_args);
        committed_after_up0 = atomic_load(g_dist.shared_tensor_map.committed_tasks.v);
        if (committed_after_up0 == 5 && !fatal_set()) {
            latest_after_up0[0] = dist_shared_tensor_map_lookup_tensor(
                g_dist.shared_tensor_map, mi_update, 5, kHDefault, protocol_after_up0[0]
            );
            latest_after_up0[1] = dist_shared_tensor_map_lookup_tensor(
                g_dist.shared_tensor_map, li_update, 5, kHDefault, protocol_after_up0[1]
            );
            latest_after_up0[2] =
                dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, oi, 5, kHDefault, protocol_after_up0[2]);
            latest_after_up0[3] = dist_shared_tensor_map_lookup_tensor(
                g_dist.shared_tensor_map, out_view, 5, kHDefault, protocol_after_up0[3]
            );
        }
    } catch (...) {
        release_error = std::current_exception();
    }

    if (release_error == nullptr && committed_after_up0 == 5 && !fatal_set()) {
        // Observe another exact-turn spin while commit is already five. This
        // directly proves task8 does not resume merely because task4 became
        // visible; turns 5-7 must still publish before its lookup can start.
        const uint32_t spins_after_up0 =
            fdwic_shared_multiworker_test::g_observed_turn_wait_spins.load(std::memory_order_acquire);
        const auto continued_wait_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (fdwic_shared_multiworker_test::g_observed_turn_wait_spins.load(std::memory_order_acquire) ==
                   spins_after_up0 &&
               !up1_finish_returned.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < continued_wait_deadline) {
            std::this_thread::yield();
        }
        up1_returned_after_up0 = up1_finish_returned.load(std::memory_order_acquire);
        committed_while_waiting_after_up0 = atomic_load(g_dist.shared_tensor_map.committed_tasks.v);
        up1_waited_after_up0 =
            fdwic_shared_multiworker_test::g_observed_turn_wait_spins.load(std::memory_order_acquire) >
                spins_after_up0 &&
            !up1_returned_after_up0 && committed_while_waiting_after_up0 == 5;

        try {
            for (int32_t task = 5; task <= 7; ++task) {
                filler_results[task - 5] =
                    dist_shared_tensor_map_publish_task(g_dist.shared_tensor_map, nullptr, 0, task, kHDefault);
            }
        } catch (...) {
            release_error = std::current_exception();
        }
    }

    bool release_completed = release_error == nullptr && committed_after_up0 == 5;
    for (const DistSharedTensorMapTaskPublishResult result : filler_results) {
        release_completed &= result == DistSharedTensorMapTaskPublishResult::Committed;
    }
    if (!release_completed) {
        set_fatal_code(PTO2_ERROR_EXPLICIT_ORCH_FATAL);
        fdwic_shared_multiworker_test::g_remote_fatal_published.store(true, std::memory_order_release);
    }

    const auto completion_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!up1_finish_returned.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < completion_deadline) {
        std::this_thread::yield();
    }
    const bool up1_timed_out = !up1_finish_returned.load(std::memory_order_acquire);
    if (up1_timed_out) {
        set_fatal_code(PTO2_ERROR_EXPLICIT_ORCH_FATAL);
        fdwic_shared_multiworker_test::g_remote_fatal_published.store(true, std::memory_order_release);
    }
    up1_thread.join();

    EXPECT_TRUE(up1_entered_turn_wait);
    EXPECT_FALSE(up1_returned_before_release);
    EXPECT_FALSE(up1_timed_out);
    EXPECT_EQ(committed_before_release, 4);
    EXPECT_EQ(vector_cursor_before_release, 8);
    EXPECT_EQ(committed_after_up0, 5);
    EXPECT_TRUE(up1_waited_after_up0);
    EXPECT_FALSE(up1_returned_after_up0);
    EXPECT_EQ(committed_while_waiting_after_up0, 5);
    for (const bool protocol_ok : protocol_after_up0) {
        EXPECT_TRUE(protocol_ok);
    }
    EXPECT_EQ(latest_after_up0[0], 4);
    EXPECT_EQ(latest_after_up0[1], 4);
    EXPECT_EQ(latest_after_up0[2], 4);
    EXPECT_EQ(latest_after_up0[3], 1);
    for (const DistSharedTensorMapTaskPublishResult result : filler_results) {
        EXPECT_EQ(result, DistSharedTensorMapTaskPublishResult::Committed);
    }

    if (release_error != nullptr) {
        try {
            std::rethrow_exception(release_error);
        } catch (const std::exception &error) {
            ADD_FAILURE() << "task4 release threw: " << error.what();
        } catch (...) {
            ADD_FAILURE() << "task4 release threw a non-standard exception";
        }
    }
    if (up1_error != nullptr) {
        try {
            std::rethrow_exception(up1_error);
        } catch (const std::exception &error) {
            ADD_FAILURE() << "task8 worker threw: " << error.what();
        } catch (...) {
            ADD_FAILURE() << "task8 worker threw a non-standard exception";
        }
    }

    EXPECT_EQ(up1_ticket.task_id, 8);
    EXPECT_EQ(up1_ticket.ready, 1);
    EXPECT_EQ(up1_ticket.claim_attempted, 1);
    EXPECT_EQ(up1_ticket.won, 1);
    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_NONE);
    EXPECT_EQ(atomic_load(g_dist.shared_tensor_map.committed_tasks.v), 9);
    EXPECT_EQ(atomic_load(g_dist.vector_cursor[0].v), 8);
    EXPECT_EQ(task0_worker_->local_index, 5);
    EXPECT_EQ(task1_worker_->local_index, 9);

    EXPECT_EQ(built_slot_count(*task0_worker_), 1);
    EXPECT_EQ(built_slot_count(*task1_worker_), 1);
    const RingSlot *up0_slot = find_built_slot(*task0_worker_, 4);
    const RingSlot *up1_slot = find_built_slot(*task1_worker_, 8);
    ASSERT_NE(up0_slot, nullptr);
    ASSERT_NE(up1_slot, nullptr);
    EXPECT_EQ(up0_slot->fanin_count, 3);
    EXPECT_EQ(fanin_occurrences(*up0_slot, 0), 1);
    EXPECT_EQ(fanin_occurrences(*up0_slot, 1), 0);
    EXPECT_EQ(fanin_occurrences(*up0_slot, 2), 1);
    EXPECT_EQ(fanin_occurrences(*up0_slot, 3), 1);
    EXPECT_EQ(up0_slot->scalar_count, 2);
    EXPECT_EQ(up0_slot->scalars[0], 1);
    EXPECT_EQ(up0_slot->scalars[1], 0);
    EXPECT_EQ(up1_slot->fanin_count, 4);
    EXPECT_EQ(fanin_occurrences(*up1_slot, 0), 1);
    EXPECT_EQ(fanin_occurrences(*up1_slot, 1), 0);
    EXPECT_EQ(fanin_occurrences(*up1_slot, 4), 1);
    EXPECT_EQ(fanin_occurrences(*up1_slot, 6), 1);
    EXPECT_EQ(fanin_occurrences(*up1_slot, 7), 1);
    EXPECT_EQ(up1_slot->scalar_count, 2);
    EXPECT_EQ(up1_slot->scalars[0], 0);
    EXPECT_EQ(up1_slot->scalars[1], 1);

    bool protocol_ok = false;
    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, mi_update, 9, kHDefault, protocol_ok), 8);
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, li_update, 9, kHDefault, protocol_ok), 8);
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, oi, 9, kHDefault, protocol_ok), 8);
    EXPECT_TRUE(protocol_ok);
    EXPECT_EQ(dist_shared_tensor_map_lookup_tensor(g_dist.shared_tensor_map, out_view, 9, kHDefault, protocol_ok), 1);
    EXPECT_TRUE(protocol_ok);

    EXPECT_EQ(std::memcmp(task0_private_map_before.data(), &task0_worker_->map, task0_private_map_before.size()), 0);
    EXPECT_EQ(std::memcmp(task1_private_map_before.data(), &task1_worker_->map, task1_private_map_before.size()), 0);
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

    std::vector<uint8_t> callable_storage;
    install_test_callable(
        runtime_, kKernelId, &fdwic_shared_multiworker_test::count_single_lane_kernel, callable_storage
    );

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

TEST_F(FdwicSharedMultiworkerTest, JointThreeLanePendingFollowersDrainAndLastLaneCompletesOnce) {
    constexpr int32_t kAicKernelId = 31;
    constexpr int32_t kAiv0KernelId = 32;
    constexpr int32_t kAiv1KernelId = 33;
    alignas(PTO2_PACKED_OUTPUT_ALIGN) std::array<uint8_t, 4096> heap{};
    const uint32_t shape[1] = {1};
    TensorCreateInfo output_info(shape, 1, DataType::FLOAT32);
    L0TaskArgs args;
    args.add_output(output_info);

    std::vector<uint8_t> aic_callable;
    std::vector<uint8_t> aiv0_callable;
    std::vector<uint8_t> aiv1_callable;
    install_test_callable(runtime_, kAicKernelId, &fdwic_shared_multiworker_test::count_joint_aic_kernel, aic_callable);
    install_test_callable(
        runtime_, kAiv0KernelId, &fdwic_shared_multiworker_test::count_joint_aiv0_kernel, aiv0_callable
    );
    install_test_callable(
        runtime_, kAiv1KernelId, &fdwic_shared_multiworker_test::count_joint_aiv1_kernel, aiv1_callable
    );

    g_dist.heap_base = heap.data();
    g_dist.heap_size = heap.size();
    g_dist.num_workers = 3;
    g_dist.num_blocks = 1;
    g_dist.final_barrier.leaf_arrivals[0].expected = 3;
    g_dist.final_barrier.root_arrival.expected = 1;

    MixedKernels mixed;
    mixed.aic_kernel_id = kAicKernelId;
    mixed.aiv0_kernel_id = kAiv0KernelId;
    mixed.aiv1_kernel_id = kAiv1KernelId;

    // CPU-sim keeps the "joint submit has been seen" fact in TLS. Each physical
    // worker therefore stays on one host thread from Begin through FinalDrain,
    // matching the persistent A5 worker lifecycle instead of manually seeding
    // the test-only thread-local state.
    DistCore *workers[3] = {aic_worker_.get(), task0_worker_.get(), task1_worker_.get()};
    DistCompeteFirstTicket tickets[3]{};
    std::exception_ptr errors[3];
    std::atomic<bool> allow_begin[3]{};
    std::atomic<bool> begin_done[3]{};
    std::atomic<bool> allow_finish[3]{};
    std::atomic<bool> finish_done[3]{};
    std::atomic<bool> drain_returned[3]{};
    std::atomic<bool> failed[3]{};
    std::atomic<bool> tls_after_begin[3]{};
    std::atomic<bool> tls_after_finish[3]{};
    std::atomic<bool> start_drain{false};
    std::atomic<bool> abort_workers{false};

    auto worker_main = [&](int32_t lane) {
        g_self = workers[lane];
        g_fdwic_joint_submit_seen = false;
        try {
            while (!allow_begin[lane].load(std::memory_order_acquire) &&
                   !abort_workers.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            if (!abort_workers.load(std::memory_order_acquire)) {
                tickets[lane] = dist_submit_compete_first_begin(nullptr, mixed);
                tls_after_begin[lane].store(g_fdwic_joint_submit_seen, std::memory_order_relaxed);
            }
            begin_done[lane].store(true, std::memory_order_release);

            while (!allow_finish[lane].load(std::memory_order_acquire) &&
                   !abort_workers.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            if (!abort_workers.load(std::memory_order_acquire)) {
                (void)dist_submit_compete_first_finish(nullptr, mixed, tickets[lane], args);
                tls_after_finish[lane].store(g_fdwic_joint_submit_seen, std::memory_order_relaxed);
            }
            finish_done[lane].store(true, std::memory_order_release);

            while (!start_drain.load(std::memory_order_acquire) && !abort_workers.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            if (!abort_workers.load(std::memory_order_acquire)) {
                dist_submit_drain_to_completion(workers[lane]);
            }
        } catch (...) {
            errors[lane] = std::current_exception();
            failed[lane].store(true, std::memory_order_release);
        }
        begin_done[lane].store(true, std::memory_order_release);
        finish_done[lane].store(true, std::memory_order_release);
        drain_returned[lane].store(true, std::memory_order_release);
        g_self = nullptr;
    };

    std::thread aic_thread(worker_main, LANE_AIC);
    std::thread aiv0_thread(worker_main, LANE_AIV0);
    std::thread aiv1_thread(worker_main, LANE_AIV1);

    auto release_all_gates = [&](bool abort) {
        if (abort) {
            abort_workers.store(true, std::memory_order_release);
        }
        for (int32_t lane = LANE_AIC; lane <= LANE_AIV1; ++lane) {
            allow_begin[lane].store(true, std::memory_order_release);
            allow_finish[lane].store(true, std::memory_order_release);
            fdwic_shared_multiworker_test::g_joint_kernel_release[lane].store(true, std::memory_order_release);
        }
        start_drain.store(true, std::memory_order_release);
    };
    auto join_all_workers = [&]() {
        if (aic_thread.joinable()) aic_thread.join();
        if (aiv0_thread.joinable()) aiv0_thread.join();
        if (aiv1_thread.joinable()) aiv1_thread.join();
    };
    auto abort_and_join = [&]() {
        release_all_gates(/*abort=*/true);
        join_all_workers();
    };

    const auto protocol_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    auto wait_until = [&](const auto &predicate) {
        while (!predicate() && std::chrono::steady_clock::now() < protocol_deadline) {
            std::this_thread::yield();
        }
        return predicate();
    };

    // AIC is the only eligible anchor for this active mask. Begin it first to
    // make the staged observation deterministic, then let both followers
    // replay the same task on their persistent worker threads.
    allow_begin[LANE_AIC].store(true, std::memory_order_release);
    const bool aic_begin_completed = wait_until([&]() {
        return begin_done[LANE_AIC].load(std::memory_order_acquire);
    });
    if (!aic_begin_completed || failed[LANE_AIC].load(std::memory_order_acquire)) {
        abort_and_join();
        ADD_FAILURE() << "AIC Begin did not complete successfully";
        return;
    }
    allow_begin[LANE_AIV0].store(true, std::memory_order_release);
    allow_begin[LANE_AIV1].store(true, std::memory_order_release);
    const bool all_begins_completed = wait_until([&]() {
        return begin_done[LANE_AIC].load(std::memory_order_acquire) &&
               begin_done[LANE_AIV0].load(std::memory_order_acquire) &&
               begin_done[LANE_AIV1].load(std::memory_order_acquire);
    });
    const bool begins_succeeded =
        aic_begin_completed && all_begins_completed && !failed[LANE_AIC].load(std::memory_order_acquire) &&
        !failed[LANE_AIV0].load(std::memory_order_acquire) && !failed[LANE_AIV1].load(std::memory_order_acquire);
    if (!begins_succeeded) {
        abort_and_join();
        ADD_FAILURE() << "all three persistent workers did not complete Begin successfully";
        return;
    }

    // Finish both losers before the anchor publishes. Their real loser tail
    // calls drain_block_won(), but any_pub is still zero, so both deposits must
    // remain pending until these same threads enter FinalDrain.
    allow_finish[LANE_AIV0].store(true, std::memory_order_release);
    allow_finish[LANE_AIV1].store(true, std::memory_order_release);
    const bool followers_finished = wait_until([&]() {
        return finish_done[LANE_AIV0].load(std::memory_order_acquire) &&
               finish_done[LANE_AIV1].load(std::memory_order_acquire);
    });
    const bool followers_succeeded = begins_succeeded && followers_finished &&
                                     !failed[LANE_AIV0].load(std::memory_order_acquire) &&
                                     !failed[LANE_AIV1].load(std::memory_order_acquire);
    if (!followers_succeeded) {
        abort_and_join();
        ADD_FAILURE() << "both follower Finish calls did not complete successfully";
        return;
    }
    const int64_t committed_before_anchor = atomic_load(g_dist.shared_tensor_map.committed_tasks.v);
    const int32_t any_pub_before_anchor = atomic_load(g_dist.blocks[0].any_pub);
    const int32_t aiv0_occupied_before_anchor = task0_worker_->occupied_count;
    const int32_t aiv1_occupied_before_anchor = task1_worker_->occupied_count;
    const int64_t flag_before_anchor = atomic_load(task_cell(0).flag);
    const uint64_t vend_before_anchor = atomic_load(task_cell(0).vend);
    const int64_t frontier_before_anchor = atomic_load(g_dist.frontier);

    allow_finish[LANE_AIC].store(true, std::memory_order_release);
    const bool anchor_finish_stage_reached = wait_until([&]() {
        return finish_done[LANE_AIC].load(std::memory_order_acquire);
    });
    const bool anchor_finished =
        followers_succeeded && anchor_finish_stage_reached && !failed[LANE_AIC].load(std::memory_order_acquire);
    if (!anchor_finished) {
        abort_and_join();
        ADD_FAILURE() << "AIC anchor Finish did not complete successfully";
        return;
    }

    struct JointDepositSnapshot {
        bool present = false;
        int64_t drained = -1;
        int32_t func_id = INVALID_KERNEL_ID;
        uint64_t function_bin_addr = 0;
        int32_t sub_block_id = -1;
        int32_t tensor_count = -1;
        int32_t fanin_count = -1;
        uint64_t tensor_addr = 0;
        uint64_t tensor_size = 0;
    };
    struct AnchorSlotSnapshot {
        bool found = false;
        bool occupied = false;
        bool built = false;
        bool is_multicore = false;
        int32_t task_id = -1;
        int32_t func_id = INVALID_KERNEL_ID;
        uint64_t function_bin_addr = 0;
        int32_t won_block = -1;
        int32_t won_slot = -1;
        int32_t tensor_count = -1;
        int32_t fanin_count = -1;
        uint64_t tensor_addr = 0;
        uint64_t tensor_size = 0;
    };

    int32_t published_index = -1;
    int32_t published_count = 0;
    int32_t non_free_won_slot_count = 0;
    for (int32_t index = 0; index < kPrivateSlots; ++index) {
        const int64_t state = atomic_load(g_dist.blocks[0].slots[index].state.v);
        if (state != kWonStateFree) {
            ++non_free_won_slot_count;
        }
        if (state == kWonStatePublished) {
            published_index = index;
            ++published_count;
        }
    }
    WonSlot *won_slot = published_index >= 0 ? &g_dist.blocks[0].slots[published_index] : nullptr;
    const RingSlot *anchor_slot = find_built_slot(*aic_worker_, 0);
    JointDepositSnapshot deposits[3]{};
    if (won_slot != nullptr) {
        for (int32_t lane = LANE_AIC; lane <= LANE_AIV1; ++lane) {
            const BuiltSubtask &source = won_slot->lane[lane];
            JointDepositSnapshot &snapshot = deposits[lane];
            snapshot.present = source.present;
            snapshot.drained = atomic_load(won_slot->drained[lane].v);
            snapshot.func_id = source.func_id;
            snapshot.function_bin_addr = source.function_bin_addr;
            snapshot.sub_block_id = source.sub_block_id;
            snapshot.tensor_count = source.tensor_count;
            snapshot.fanin_count = source.fanin_count;
            if (source.tensor_count > 0) {
                snapshot.tensor_addr = source.tensors[0].buffer.addr;
                snapshot.tensor_size = source.tensors[0].buffer.size;
            }
        }
    }
    AnchorSlotSnapshot anchor_snapshot;
    if (anchor_slot != nullptr) {
        anchor_snapshot.found = true;
        anchor_snapshot.occupied = anchor_slot->occupied;
        anchor_snapshot.built = anchor_slot->built;
        anchor_snapshot.is_multicore = anchor_slot->is_multicore;
        anchor_snapshot.task_id = anchor_slot->task_id;
        anchor_snapshot.func_id = anchor_slot->func_id;
        anchor_snapshot.function_bin_addr = anchor_slot->function_bin_addr;
        anchor_snapshot.won_block = anchor_slot->won_block;
        anchor_snapshot.won_slot = anchor_slot->won_slot;
        anchor_snapshot.tensor_count = anchor_slot->tensor_count;
        anchor_snapshot.fanin_count = anchor_slot->fanin_count;
        if (anchor_slot->tensor_count > 0) {
            anchor_snapshot.tensor_addr = anchor_slot->tensors[0].buffer.addr;
            anchor_snapshot.tensor_size = anchor_slot->tensors[0].buffer.size;
        }
    }

    const bool tickets_ok = begins_succeeded && tickets[LANE_AIC].task_id == 0 && tickets[LANE_AIC].won == 1 &&
                            tickets[LANE_AIC].joint == 1 && tickets[LANE_AIC].joint_init == 1 &&
                            tickets[LANE_AIC].claim_attempted == 1 && tickets[LANE_AIC].joint_count == 3 &&
                            tickets[LANE_AIC].joint_block == 0 && tickets[LANE_AIC].kernel_id == kAicKernelId &&
                            tickets[LANE_AIV0].task_id == 0 && tickets[LANE_AIV0].won == 0 &&
                            tickets[LANE_AIV0].joint == 1 && tickets[LANE_AIV0].joint_init == 0 &&
                            tickets[LANE_AIV0].claim_attempted == 0 && tickets[LANE_AIV0].joint_count == 3 &&
                            tickets[LANE_AIV0].joint_block == 0 && tickets[LANE_AIV0].kernel_id == INVALID_KERNEL_ID &&
                            tickets[LANE_AIV1].task_id == 0 && tickets[LANE_AIV1].won == 0 &&
                            tickets[LANE_AIV1].joint == 1 && tickets[LANE_AIV1].joint_init == 0 &&
                            tickets[LANE_AIV1].claim_attempted == 0 && tickets[LANE_AIV1].joint_count == 3 &&
                            tickets[LANE_AIV1].joint_block == 0 && tickets[LANE_AIV1].kernel_id == INVALID_KERNEL_ID;
    const bool no_worker_failed = !failed[LANE_AIC].load(std::memory_order_acquire) &&
                                  !failed[LANE_AIV0].load(std::memory_order_acquire) &&
                                  !failed[LANE_AIV1].load(std::memory_order_acquire);
    const bool published_slot_ok =
        won_slot != nullptr && published_count == 1 && non_free_won_slot_count == 1 &&
        atomic_load(won_slot->remaining.v) == 3 && !deposits[LANE_AIC].present && deposits[LANE_AIV0].present &&
        deposits[LANE_AIV1].present && deposits[LANE_AIV0].drained == kDrainedFree &&
        deposits[LANE_AIV1].drained == kDrainedFree && deposits[LANE_AIV0].func_id == kAiv0KernelId &&
        deposits[LANE_AIV0].function_bin_addr ==
            reinterpret_cast<uint64_t>(&fdwic_shared_multiworker_test::count_joint_aiv0_kernel) &&
        deposits[LANE_AIV0].sub_block_id == 0 && deposits[LANE_AIV0].tensor_count == 1 &&
        deposits[LANE_AIV0].fanin_count == 0 &&
        deposits[LANE_AIV0].tensor_addr == reinterpret_cast<uint64_t>(heap.data()) &&
        deposits[LANE_AIV0].tensor_size == sizeof(float) && deposits[LANE_AIV1].func_id == kAiv1KernelId &&
        deposits[LANE_AIV1].function_bin_addr ==
            reinterpret_cast<uint64_t>(&fdwic_shared_multiworker_test::count_joint_aiv1_kernel) &&
        deposits[LANE_AIV1].sub_block_id == 1 && deposits[LANE_AIV1].tensor_count == 1 &&
        deposits[LANE_AIV1].fanin_count == 0 &&
        deposits[LANE_AIV1].tensor_addr == reinterpret_cast<uint64_t>(heap.data()) &&
        deposits[LANE_AIV1].tensor_size == sizeof(float);
    const bool anchor_slot_ok =
        anchor_snapshot.found && anchor_snapshot.occupied && anchor_snapshot.built && anchor_snapshot.is_multicore &&
        anchor_snapshot.task_id == 0 && anchor_snapshot.func_id == kAicKernelId &&
        anchor_snapshot.function_bin_addr ==
            reinterpret_cast<uint64_t>(&fdwic_shared_multiworker_test::count_joint_aic_kernel) &&
        anchor_snapshot.won_block == 0 && anchor_snapshot.won_slot == published_index &&
        anchor_snapshot.tensor_count == 1 && anchor_snapshot.fanin_count == 0 &&
        anchor_snapshot.tensor_addr == reinterpret_cast<uint64_t>(heap.data()) &&
        anchor_snapshot.tensor_size == sizeof(float) && aic_worker_->occupied_count == 1 &&
        built_slot_count(*aic_worker_) == 1 && task0_worker_->occupied_count == 0 &&
        task1_worker_->occupied_count == 0 && built_slot_count(*task0_worker_) == 0 &&
        built_slot_count(*task1_worker_) == 0;
    const bool pre_drain_ok = aic_begin_completed && all_begins_completed && followers_finished && anchor_finished &&
                              no_worker_failed && tickets_ok && published_slot_ok && anchor_slot_ok;
    if (!pre_drain_ok) {
        abort_and_join();
        EXPECT_TRUE(no_worker_failed);
        EXPECT_TRUE(tickets_ok);
        EXPECT_EQ(published_count, 1);
        EXPECT_EQ(non_free_won_slot_count, 1);
        EXPECT_TRUE(published_slot_ok);
        EXPECT_TRUE(anchor_slot_ok);
        ADD_FAILURE() << "joint task pre-FinalDrain invariants were not established";
        return;
    }
    start_drain.store(true, std::memory_order_release);

    auto all_joint_kernels_entered = [&]() {
        for (int32_t lane = LANE_AIC; lane <= LANE_AIV1; ++lane) {
            if (fdwic_shared_multiworker_test::g_joint_kernel_entered[lane].load(std::memory_order_acquire) != 1) {
                return false;
            }
        }
        return true;
    };
    const bool all_kernels_entered = pre_drain_ok && wait_until(all_joint_kernels_entered);
    const int64_t remaining_before_release = won_slot != nullptr ? atomic_load(won_slot->remaining.v) : -1;
    const int64_t flag_before_release = atomic_load(task_cell(0).flag);
    const int64_t state_before_release = won_slot != nullptr ? atomic_load(won_slot->state.v) : -1;

    auto wait_remaining_and_returned = [&](int64_t expected, int32_t lane) {
        if (won_slot == nullptr) return false;
        return wait_until([&]() {
            return atomic_load(won_slot->remaining.v) == expected &&
                   drain_returned[lane].load(std::memory_order_acquire);
        });
    };

    fdwic_shared_multiworker_test::g_joint_kernel_release[LANE_AIC].store(true, std::memory_order_release);
    const bool anchor_decremented_and_returned = all_kernels_entered && wait_remaining_and_returned(2, LANE_AIC);
    const int64_t flag_after_anchor = atomic_load(task_cell(0).flag);
    const uint64_t vend_after_anchor = atomic_load(task_cell(0).vend);
    const int64_t frontier_after_anchor = atomic_load(g_dist.frontier);
    const int64_t state_after_anchor = won_slot != nullptr ? atomic_load(won_slot->state.v) : -1;
    const uint32_t aiv1_exited_after_anchor =
        fdwic_shared_multiworker_test::g_joint_kernel_exited[LANE_AIV1].load(std::memory_order_acquire);
    const bool aiv1_returned_after_anchor = drain_returned[LANE_AIV1].load(std::memory_order_acquire);

    fdwic_shared_multiworker_test::g_joint_kernel_release[LANE_AIV0].store(true, std::memory_order_release);
    const bool first_follower_decremented_and_returned =
        all_kernels_entered && wait_remaining_and_returned(1, LANE_AIV0);
    const int64_t flag_after_first_follower = atomic_load(task_cell(0).flag);
    const uint64_t vend_after_first_follower = atomic_load(task_cell(0).vend);
    const int64_t frontier_after_first_follower = atomic_load(g_dist.frontier);
    const int64_t state_after_first_follower = won_slot != nullptr ? atomic_load(won_slot->state.v) : -1;
    const uint32_t aiv1_exited_after_first_follower =
        fdwic_shared_multiworker_test::g_joint_kernel_exited[LANE_AIV1].load(std::memory_order_acquire);
    const bool aiv1_returned_after_first_follower = drain_returned[LANE_AIV1].load(std::memory_order_acquire);

    // Always release every test gate before joining, including a failed
    // observation, so this test cannot manufacture its own deadlock.
    release_all_gates(/*abort=*/false);
    join_all_workers();

    EXPECT_TRUE(aic_begin_completed);
    EXPECT_TRUE(all_begins_completed);
    EXPECT_TRUE(begins_succeeded);
    EXPECT_TRUE(followers_finished);
    EXPECT_TRUE(followers_succeeded);
    EXPECT_TRUE(anchor_finish_stage_reached);
    EXPECT_TRUE(anchor_finished);
    EXPECT_TRUE(no_worker_failed);
    EXPECT_TRUE(tickets_ok);
    EXPECT_TRUE(published_slot_ok);
    EXPECT_TRUE(anchor_slot_ok);
    EXPECT_TRUE(pre_drain_ok);
    EXPECT_TRUE(all_kernels_entered);
    EXPECT_EQ(committed_before_anchor, 0);
    EXPECT_EQ(any_pub_before_anchor, 0);
    EXPECT_EQ(aiv0_occupied_before_anchor, 0);
    EXPECT_EQ(aiv1_occupied_before_anchor, 0);
    EXPECT_EQ(flag_before_anchor, 0);
    EXPECT_EQ(vend_before_anchor, 0);
    EXPECT_EQ(frontier_before_anchor, -1);

    for (int32_t lane = LANE_AIC; lane <= LANE_AIV1; ++lane) {
        EXPECT_TRUE(tls_after_begin[lane].load(std::memory_order_relaxed));
        EXPECT_TRUE(tls_after_finish[lane].load(std::memory_order_relaxed));
        EXPECT_TRUE(drain_returned[lane].load(std::memory_order_acquire));
        EXPECT_EQ(errors[lane], nullptr);
    }

    ASSERT_NE(won_slot, nullptr);
    EXPECT_EQ(atomic_load(g_dist.shared_tensor_map.committed_tasks.v), 1);
    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_NONE);
    EXPECT_EQ(g_dist.blocks[0].any_pub, 1);
    EXPECT_EQ(remaining_before_release, 3);
    EXPECT_EQ(flag_before_release, 0);
    EXPECT_EQ(state_before_release, kWonStatePublished);
    EXPECT_TRUE(anchor_decremented_and_returned);
    EXPECT_EQ(flag_after_anchor, 0);
    EXPECT_EQ(vend_after_anchor, 0);
    EXPECT_EQ(frontier_after_anchor, -1);
    EXPECT_EQ(state_after_anchor, kWonStatePublished);
    EXPECT_EQ(aiv1_exited_after_anchor, 0);
    EXPECT_FALSE(aiv1_returned_after_anchor);
    EXPECT_TRUE(first_follower_decremented_and_returned);
    EXPECT_EQ(flag_after_first_follower, 0);
    EXPECT_EQ(vend_after_first_follower, 0);
    EXPECT_EQ(frontier_after_first_follower, -1);
    EXPECT_EQ(state_after_first_follower, kWonStatePublished);
    EXPECT_EQ(aiv1_exited_after_first_follower, 0);
    EXPECT_FALSE(aiv1_returned_after_first_follower);

    EXPECT_FALSE(deposits[LANE_AIC].present);
    EXPECT_TRUE(deposits[LANE_AIV0].present);
    EXPECT_TRUE(deposits[LANE_AIV1].present);
    EXPECT_EQ(deposits[LANE_AIV0].func_id, kAiv0KernelId);
    EXPECT_EQ(deposits[LANE_AIV1].func_id, kAiv1KernelId);
    EXPECT_EQ(deposits[LANE_AIV0].sub_block_id, 0);
    EXPECT_EQ(deposits[LANE_AIV1].sub_block_id, 1);
    EXPECT_EQ(deposits[LANE_AIV0].tensor_count, 1);
    EXPECT_EQ(deposits[LANE_AIV1].tensor_count, 1);
    EXPECT_EQ(deposits[LANE_AIV0].fanin_count, 0);
    EXPECT_EQ(deposits[LANE_AIV1].fanin_count, 0);
    EXPECT_EQ(deposits[LANE_AIV0].tensor_addr, reinterpret_cast<uint64_t>(heap.data()));
    EXPECT_EQ(deposits[LANE_AIV1].tensor_addr, reinterpret_cast<uint64_t>(heap.data()));
    EXPECT_EQ(deposits[LANE_AIV0].tensor_size, sizeof(float));
    EXPECT_EQ(deposits[LANE_AIV1].tensor_size, sizeof(float));
    EXPECT_EQ(
        deposits[LANE_AIV0].function_bin_addr,
        reinterpret_cast<uint64_t>(&fdwic_shared_multiworker_test::count_joint_aiv0_kernel)
    );
    EXPECT_EQ(
        deposits[LANE_AIV1].function_bin_addr,
        reinterpret_cast<uint64_t>(&fdwic_shared_multiworker_test::count_joint_aiv1_kernel)
    );
    EXPECT_EQ(atomic_load(won_slot->drained[LANE_AIV0].v), kDrainedClaimed);
    EXPECT_EQ(atomic_load(won_slot->drained[LANE_AIV1].v), kDrainedClaimed);

    EXPECT_TRUE(anchor_snapshot.is_multicore);
    EXPECT_EQ(anchor_snapshot.func_id, kAicKernelId);
    EXPECT_EQ(
        anchor_snapshot.function_bin_addr,
        reinterpret_cast<uint64_t>(&fdwic_shared_multiworker_test::count_joint_aic_kernel)
    );
    EXPECT_EQ(anchor_snapshot.won_block, 0);
    EXPECT_EQ(anchor_snapshot.won_slot, published_index);
    EXPECT_EQ(anchor_snapshot.tensor_count, 1);
    EXPECT_EQ(anchor_snapshot.fanin_count, 0);
    EXPECT_EQ(anchor_snapshot.tensor_addr, reinterpret_cast<uint64_t>(heap.data()));
    EXPECT_EQ(anchor_snapshot.tensor_size, sizeof(float));
    EXPECT_EQ(aic_worker_->heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    EXPECT_EQ(task0_worker_->heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    EXPECT_EQ(task1_worker_->heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    EXPECT_EQ(aic_worker_->owned_total, 0);
    EXPECT_EQ(task0_worker_->owned_total, 1);
    EXPECT_EQ(task1_worker_->owned_total, 1);

    for (int32_t lane = LANE_AIC; lane <= LANE_AIV1; ++lane) {
        EXPECT_EQ(fdwic_shared_multiworker_test::g_joint_kernel_entered[lane].load(std::memory_order_acquire), 1);
        EXPECT_EQ(fdwic_shared_multiworker_test::g_joint_kernel_exited[lane].load(std::memory_order_acquire), 1);
    }
    EXPECT_EQ(atomic_load(won_slot->remaining.v), 0);
    EXPECT_EQ(atomic_load(won_slot->state.v), kWonStateFree);
    EXPECT_EQ(task_cell(0).flag, 1);
    EXPECT_EQ(task_cell(0).vend, PTO2_PACKED_OUTPUT_ALIGN);
    EXPECT_EQ(g_dist.frontier, 0);
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
