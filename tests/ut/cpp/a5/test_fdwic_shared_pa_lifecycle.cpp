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
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>

// Include the normal CPU-sim declarations first, then replace only this
// translation unit's final spin hint with an observation hook. The hook does
// not replace any production barrier, fatal, slot, or metadata operation.
#define PTO_FDWIC_TRACE_ENABLED 0
#include "inner_kernel.h"
#undef SPIN_WAIT_HINT
#include "pto_runtime2_types.h"

namespace fdwic_shared_pa_lifecycle_test {

std::atomic<uint32_t> g_wait_spins{0};
std::atomic<uint32_t> g_spins_after_remote_fatal{0};
std::atomic<uint32_t> g_kernel_calls{0};
std::atomic<bool> g_remote_fatal_published{false};
thread_local bool g_observe_wait = false;
thread_local bool g_limit_spins_after_remote_fatal = false;

constexpr uint32_t kPostFatalSpinLimit = 8192;

void spin_wait_hint() {
    if (g_observe_wait) {
        g_wait_spins.fetch_add(1, std::memory_order_release);
    }
    if (g_limit_spins_after_remote_fatal &&
        g_remote_fatal_published.load(std::memory_order_acquire) &&
        g_spins_after_remote_fatal.fetch_add(1, std::memory_order_acq_rel) >=
            kPostFatalSpinLimit) {
        throw std::runtime_error("production wait did not consume the remote fatal");
    }
    std::this_thread::yield();
}

void count_kernel(int64_t *) {
    g_kernel_calls.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace fdwic_shared_pa_lifecycle_test

#undef SPIN_WAIT_HINT
#define SPIN_WAIT_HINT() ::fdwic_shared_pa_lifecycle_test::spin_wait_hint()
#include "dist_engine/aicore/dist_engine.cpp"  // NOLINT(build/include)
#include "dist_engine/aicpu/shared_tensor_map_init.h"
#undef SPIN_WAIT_HINT

[[noreturn]] void assert_impl(const char *condition, const char *, int) {
    throw std::logic_error(condition);
}

extern "C" void aicpu_orchestration_entry(const L2TaskArgs &) {}
volatile uint8_t *sim_get_reg_base() { return nullptr; }
uint32_t sim_get_physical_core_id() { return 0; }

Runtime::Runtime() {
    for (uint64_t &address : func_id_to_addr_) address = 0;
    use_example_exec_time_ = false;
    for (int32_t &duration : example_exec_time_ns_) duration = 0;
}

namespace {

struct WorkerThreadResult {
    std::atomic<bool> returned{false};
    std::exception_ptr error;
};

class FdwicSharedPaLifecycleTest : public ::testing::Test {
protected:
    void SetUp() override {
        static_assert(PTO_FDWIC_SHARED_MAP == 1);
        static_assert(kFdwicCompiledBackendReady);

        fdwic_shared_pa_lifecycle_test::g_wait_spins.store(
            0, std::memory_order_relaxed
        );
        fdwic_shared_pa_lifecycle_test::g_spins_after_remote_fatal.store(
            0, std::memory_order_relaxed
        );
        fdwic_shared_pa_lifecycle_test::g_kernel_calls.store(
            0, std::memory_order_relaxed
        );
        fdwic_shared_pa_lifecycle_test::g_remote_fatal_published.store(
            false, std::memory_order_relaxed
        );

        g_dist_ptr = &g_dist_fallback;
        g_self = nullptr;
        g_dist.fatal = 0;
        g_dist.error_code = PTO2_ERROR_NONE;
        g_dist.frontier = -1;
        g_dist.H = kHDefault;
        g_dist.heap_base = nullptr;
        g_dist.heap_size = 0;
        g_dist.runtime = &runtime_;
        g_dist.num_workers = 3;
        g_dist.num_blocks = 1;
        g_dist.replay_done = 0;
        g_dist.started_count = 0;
        g_skip_exec = false;
        g_fdwic_joint_submit_seen = false;

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
        dist_shared_pa_tensor_map_reset(g_dist.shared_pa);

        reset_worker(*aic_worker_, CoreType::AIC, LANE_AIC, 0);
        reset_worker(*aiv0_worker_, CoreType::AIV, LANE_AIV0, 1);
        reset_worker(*aiv1_worker_, CoreType::AIV, LANE_AIV1, 2);
    }

    void TearDown() override {
        fdwic_shared_pa_lifecycle_test::g_observe_wait = false;
        fdwic_shared_pa_lifecycle_test::g_limit_spins_after_remote_fatal =
            false;
        g_self = nullptr;
        g_dist_ptr = nullptr;
    }

    static void reset_worker(
        DistCore &worker, CoreType role, int32_t lane, int32_t core_idx
    ) {
        dist_core_reset(worker, role, /*block=*/0, lane);
        worker.core_idx = core_idx;
    }

    static void observe_wait_and_limit_after_fatal() {
        fdwic_shared_pa_lifecycle_test::g_observe_wait = true;
        fdwic_shared_pa_lifecycle_test::g_limit_spins_after_remote_fatal = true;
    }

    static void stop_observing_wait() {
        fdwic_shared_pa_lifecycle_test::g_limit_spins_after_remote_fatal =
            false;
        fdwic_shared_pa_lifecycle_test::g_observe_wait = false;
    }

    static void wait_until_production_spins() {
        while (fdwic_shared_pa_lifecycle_test::g_wait_spins.load(
                   std::memory_order_acquire
               ) == 0) {
            std::this_thread::yield();
        }
    }

    static void publish_remote_fatal() {
        set_fatal_code(PTO2_ERROR_EXPLICIT_ORCH_FATAL);
        fdwic_shared_pa_lifecycle_test::g_remote_fatal_published.store(
            true, std::memory_order_release
        );
    }

    static void expect_no_thread_error(
        const WorkerThreadResult &result, const char *context
    ) {
        if (result.error == nullptr) return;
        try {
            std::rethrow_exception(result.error);
        } catch (const std::exception &error) {
            FAIL() << context << " threw: " << error.what();
        } catch (...) {
            FAIL() << context << " threw a non-standard exception";
        }
    }

    Runtime runtime_;
    std::unique_ptr<DistCore> aic_worker_ = std::make_unique<DistCore>();
    std::unique_ptr<DistCore> aiv0_worker_ = std::make_unique<DistCore>();
    std::unique_ptr<DistCore> aiv1_worker_ = std::make_unique<DistCore>();
};

TEST_F(
    FdwicSharedPaLifecycleTest,
    ThreeWorkersDrainReadyWorkAndCompleteTheFinalBarrier
) {
    g_dist.final_barrier.leaf_arrivals[0].expected = 3;
    g_dist.final_barrier.root_arrival.expected = 1;

    RingSlot &slot = aiv0_worker_->slots[0];
    slot.occupied = true;
    slot.built = true;
    slot.task_id = 0;
    slot.func_id = 17;
    slot.function_bin_addr = reinterpret_cast<uint64_t>(
        &fdwic_shared_pa_lifecycle_test::count_kernel
    );
    slot.tensor_count = 0;
    slot.scalar_count = 0;
    slot.fanin_count = 0;
    slot.is_multicore = false;
    aiv0_worker_->occupied_count = 1;

    std::array<DistCore *, 3> workers = {
        aic_worker_.get(), aiv0_worker_.get(), aiv1_worker_.get()
    };
    std::array<WorkerThreadResult, 3> results;
    std::array<std::thread, 3> threads;
    std::atomic<uint32_t> ready{0};
    std::atomic<bool> start{false};

    for (size_t index = 0; index < workers.size(); ++index) {
        threads[index] = std::thread([&, index] {
            g_self = workers[index];
            g_fdwic_joint_submit_seen = false;
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            try {
                dist_submit_drain_to_completion(workers[index]);
            } catch (...) {
                results[index].error = std::current_exception();
            }
            results[index].returned.store(true, std::memory_order_release);
            g_self = nullptr;
        });
    }

    while (ready.load(std::memory_order_acquire) != workers.size()) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (std::thread &thread : threads) thread.join();

    for (size_t index = 0; index < results.size(); ++index) {
        EXPECT_TRUE(results[index].returned.load(std::memory_order_acquire));
        expect_no_thread_error(results[index], "FinalDrain worker");
        EXPECT_EQ(workers[index]->occupied_count, 0) << "worker=" << index;
    }
    EXPECT_EQ(
        fdwic_shared_pa_lifecycle_test::g_kernel_calls.load(
            std::memory_order_relaxed
        ),
        1U
    );
    EXPECT_FALSE(slot.occupied);
    EXPECT_FALSE(slot.built);
    EXPECT_EQ(task_cell(0).flag, 1);
    EXPECT_EQ(g_dist.final_barrier.leaf_arrivals[0].v, 3);
    EXPECT_EQ(g_dist.final_barrier.root_arrival.v, 1);
    EXPECT_EQ(g_dist.final_barrier.root_release.v, 1);
    EXPECT_EQ(g_dist.final_barrier.leaf_releases[0].v, 1);
    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_NONE);
}

TEST_F(
    FdwicSharedPaLifecycleTest,
    RemoteFatalInterruptsAnIncompleteFinalBarrier
) {
    g_dist.final_barrier.leaf_arrivals[0].expected = 2;
    g_dist.final_barrier.root_arrival.expected = 1;

    WorkerThreadResult result;
    std::thread worker([&] {
        try {
            g_self = aic_worker_.get();
            g_fdwic_joint_submit_seen = false;
            observe_wait_and_limit_after_fatal();
            dist_submit_drain_to_completion(aic_worker_.get());
            stop_observing_wait();
        } catch (...) {
            result.error = std::current_exception();
        }
        result.returned.store(true, std::memory_order_release);
        g_self = nullptr;
    });

    wait_until_production_spins();
    EXPECT_FALSE(result.returned.load(std::memory_order_acquire));
    publish_remote_fatal();
    worker.join();

    EXPECT_TRUE(result.returned.load(std::memory_order_acquire));
    expect_no_thread_error(result, "incomplete final-barrier waiter");
    EXPECT_EQ(g_dist.fatal, 1);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_EXPLICIT_ORCH_FATAL);
    EXPECT_EQ(aic_worker_->local_index, kFlagCap);
    EXPECT_EQ(g_dist.final_barrier.leaf_arrivals[0].v, 1);
    EXPECT_EQ(g_dist.final_barrier.root_arrival.v, 0);
    EXPECT_EQ(g_dist.final_barrier.root_release.v, 0);
    EXPECT_EQ(g_dist.final_barrier.leaf_releases[0].v, 0);
}

TEST_F(
    FdwicSharedPaLifecycleTest,
    RemoteFatalInterruptsTheSharedSlotCapacityWait
) {
    constexpr int32_t kBlockedProducer = 7;
    for (int32_t index = 0; index < kPrivateSlots - kWonReserve; ++index) {
        RingSlot &slot = aiv0_worker_->slots[index];
        slot.occupied = true;
        slot.built = true;
        slot.task_id = 100 + index;
        slot.fanin_count = 1;
        slot.fanin[0] = kBlockedProducer;
    }
    aiv0_worker_->occupied_count = kPrivateSlots - kWonReserve;
    task_cell(kBlockedProducer).flag = 0;

    WorkerThreadResult result;
    bool capacity_available = true;
    std::thread worker([&] {
        try {
            g_self = aiv0_worker_.get();
            g_fdwic_joint_submit_seen = false;
            observe_wait_and_limit_after_fatal();
            capacity_available =
                dist_submit_wait_slot_capacity(aiv0_worker_.get(), /*task_id=*/3);
            stop_observing_wait();
        } catch (...) {
            result.error = std::current_exception();
        }
        result.returned.store(true, std::memory_order_release);
        g_self = nullptr;
    });

    wait_until_production_spins();
    EXPECT_FALSE(result.returned.load(std::memory_order_acquire));
    publish_remote_fatal();
    worker.join();

    EXPECT_TRUE(result.returned.load(std::memory_order_acquire));
    expect_no_thread_error(result, "slot-capacity waiter");
    EXPECT_FALSE(capacity_available);
    EXPECT_EQ(g_dist.fatal, 1);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_EXPLICIT_ORCH_FATAL);
    EXPECT_EQ(aiv0_worker_->local_index, kFlagCap);
    EXPECT_EQ(aiv0_worker_->occupied_count, kPrivateSlots - kWonReserve);
    for (int32_t index = 0; index < kPrivateSlots - kWonReserve; ++index) {
        EXPECT_TRUE(aiv0_worker_->slots[index].occupied) << "slot=" << index;
        EXPECT_TRUE(aiv0_worker_->slots[index].built) << "slot=" << index;
    }
}

TEST_F(
    FdwicSharedPaLifecycleTest,
    RemoteFatalInterruptsThePredecessorInsertTurnWait
) {
    DistSubmitCtx ctx{};
    ctx.self = aiv0_worker_.get();
    ctx.task_id = 1;
    task_cell(0).deps_prepared = -1;
    g_dist.shared_pa.insert_completion.cells[0].v = -1;

    WorkerThreadResult result;
    bool turn_ready = true;
    int64_t ready_observed = INT64_MIN;
    uint32_t load_count = UINT32_MAX;
    std::thread worker([&] {
        try {
            g_self = aiv0_worker_.get();
            g_fdwic_joint_submit_seen = false;
            observe_wait_and_limit_after_fatal();
            turn_ready = dist_shared_pa_wait_insert_turn(
                ctx, ready_observed, load_count
            );
            stop_observing_wait();
        } catch (...) {
            result.error = std::current_exception();
        }
        result.returned.store(true, std::memory_order_release);
        g_self = nullptr;
    });

    wait_until_production_spins();
    EXPECT_FALSE(result.returned.load(std::memory_order_acquire));
    publish_remote_fatal();
    worker.join();

    EXPECT_TRUE(result.returned.load(std::memory_order_acquire));
    expect_no_thread_error(result, "predecessor insert-turn waiter");
    EXPECT_FALSE(turn_ready);
    EXPECT_EQ(ready_observed, -1);
    EXPECT_EQ(load_count, 0U);
    EXPECT_EQ(g_dist.shared_pa.insert_completion.cells[0].v, -1);
    EXPECT_EQ(task_cell(0).deps_prepared, -1);
    EXPECT_EQ(g_dist.fatal, 1);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_EXPLICIT_ORCH_FATAL);
    EXPECT_EQ(aiv0_worker_->local_index, kFlagCap);
}

}  // namespace
