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
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#define PTO_FDWIC_TRACE_ENABLED 0
#include "dist_engine/aicore/dist_engine.cpp"  // NOLINT(build/include)
#include "dist_engine/aicpu/shared_tensor_map_init.h"
#include "pto_orchestration_api.h"

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

struct SharedPaRunResult {
    bool ok{false};
    uint32_t callbacks{0};
    uint32_t callback_mask{0};
    uint32_t completed_stage{0};
};

class FdwicSharedPaSubmitTest : public ::testing::Test {
protected:
    void SetUp() override {
        static_assert(PTO_FDWIC_SHARED_MAP == 1);
        g_dist_ptr = &g_dist_fallback;
        g_dist.fatal = 0;
        g_dist.error_code = PTO2_ERROR_NONE;
        g_dist.frontier = -1;
        g_dist.H = kHDefault;
        g_dist.heap_size = kFdwicSharedHeapBytes;
        heap_ = std::malloc(kFdwicSharedHeapBytes);
        ASSERT_NE(heap_, nullptr);
        g_dist.heap_base = static_cast<uint8_t *>(heap_);
        g_dist.runtime = &runtime_;
        g_dist.num_workers = 2;
        g_dist.num_blocks = 1;
        g_dist.replay_done = 0;
        g_dist.started_count = 0;
        g_fdwic_joint_submit_seen = false;

        for (int32_t shard = 0; shard < kCursorShards; ++shard) {
            g_dist.cube_cursor[shard].v = -1;
            g_dist.vector_cursor[shard].v = -1;
            g_dist.alloc_cursor[shard].v = -1;
        }
        for (uint32_t task = 0; task < kFdwicSharedPaTaskCapacity; ++task) {
            reset_task_cell(static_cast<int32_t>(task));
        }
        dist_shared_pa_tensor_map_reset(g_dist.shared_pa);

        dist_core_reset(g_dist.cores[0], CoreType::AIC, 0, LANE_AIC);
        g_dist.cores[0].core_idx = 0;
        dist_core_reset(g_dist.cores[1], CoreType::AIV, 0, LANE_AIV0);
        g_dist.cores[1].core_idx = 32;
    }

    void TearDown() override {
        g_self = nullptr;
        g_dist.heap_base = nullptr;
        std::free(heap_);
    }

    static Tensor make_external(uintptr_t address, bool manual_dep = false) {
        const uint32_t shape[1] = {16};
        return make_tensor_external(
            reinterpret_cast<void *>(address), shape, 1, DataType::FLOAT32, manual_dep
        );
    }

    static void replay_one(
        CoreType role, int32_t worker_index, SharedPaRunResult &result,
        uint32_t batches = 1
    ) {
        g_dist_ptr = &g_dist_fallback;
        g_self = &g_dist.cores[worker_index];
        g_fdwic_joint_submit_seen = false;
        const DistSharedPaReplayContext replay = dist_shared_pa_replay_context();
        if (!replay.ready() || replay.role() != role) return;

        Tensor q = make_external(0x100000);
        Tensor k = make_external(0x200000);
        Tensor v = make_external(0x300000);
        Tensor table = make_external(0x400000);
        Tensor out = make_external(0x500000, true);
        const uint32_t vector_shape[1] = {16};
        TensorCreateInfo vector_ci(vector_shape, 1, DataType::FLOAT32);
        const uint32_t scalar_shape[1] = {1};
        TensorCreateInfo scalar_ci(scalar_shape, 1, DataType::FLOAT32);

        for (uint32_t batch = 0; batch < batches; ++batch) {
            const uint32_t stage_base = batch * kFdwicSharedPaTasksPerBatch;
            L0TaskArgs args;
            args.reset();
            SharedTaskOutputs alloc = shared_pa_alloc_tensors_compete_first(
                replay, args, [&](L0TaskArgs &winner_args) {
                    ++result.callbacks;
                    result.callback_mask |= 1U << 0;
                    winner_args.add_output(vector_ci, scalar_ci, scalar_ci);
                }
            );
            if (alloc.size() != 3) return;
            result.completed_stage = stage_base + 1U;
            const FdwicOutputRef oi = alloc.output_ref(0);
            const FdwicOutputRef li_update = alloc.output_ref(1);
            const FdwicOutputRef mi_update = alloc.output_ref(2);

            SharedTaskOutputs qk = shared_pa_submit_aic_compete_first(
                replay, DistSharedPaTaskKind::Qk, 0, args, [&](L0TaskArgs &winner_args) {
                    ++result.callbacks;
                    result.callback_mask |= 1U << 1;
                    winner_args.reset();
                    winner_args.add_input(q, k, table);
                    winner_args.add_output(vector_ci);
                    winner_args.add_scalar(uint64_t{1}, uint64_t{0});
                }
            );
            if (qk.size() != 1) return;
            result.completed_stage = stage_base + 2U;
            const FdwicOutputRef sij = qk.output_ref(0);

            SharedTaskOutputs sf = shared_pa_submit_aiv_compete_first(
                replay, DistSharedPaTaskKind::Sf, 1, args, [&](L0TaskArgs &winner_args) {
                    ++result.callbacks;
                    result.callback_mask |= 1U << 2;
                    winner_args.reset();
                    winner_args.add_input(sij);
                    winner_args.add_output(vector_ci, scalar_ci, scalar_ci);
                    winner_args.add_scalar(uint64_t{1}, uint64_t{1}, uint64_t{1});
                }
            );
            if (sf.size() != 3) return;
            result.completed_stage = stage_base + 3U;
            const FdwicOutputRef pij = sf.output_ref(0);
            const FdwicOutputRef mi = sf.output_ref(1);
            const FdwicOutputRef li = sf.output_ref(2);

            SharedTaskOutputs pv = shared_pa_submit_aic_compete_first(
                replay, DistSharedPaTaskKind::Pv, 2, args, [&](L0TaskArgs &winner_args) {
                    ++result.callbacks;
                    result.callback_mask |= 1U << 3;
                    winner_args.reset();
                    winner_args.add_input(pij, v, table);
                    winner_args.add_output(vector_ci);
                    winner_args.add_scalar(uint64_t{1}, uint64_t{0});
                }
            );
            if (pv.size() != 1) return;
            result.completed_stage = stage_base + 4U;
            const FdwicOutputRef oi_new = pv.output_ref(0);

            SharedTaskOutputs up = shared_pa_submit_aiv_compete_first(
                replay, DistSharedPaTaskKind::Up, 3, args, [&](L0TaskArgs &winner_args) {
                    ++result.callbacks;
                    result.callback_mask |= 1U << 4;
                    winner_args.reset();
                    winner_args.add_input(mi, li, oi_new);
                    winner_args.add_inout(mi_update, li_update, oi, out);
                    winner_args.add_scalar(uint64_t{1}, uint64_t{1});
                }
            );
            if (up.producer_task_id !=
                    static_cast<int32_t>(stage_base + 4U) ||
                !up.empty()) {
                return;
            }
            result.completed_stage = stage_base + 5U;
        }

        uint32_t idle = 0;
        while (g_self->occupied_count != 0 && idle < 1000000) {
            if (drain_phase_b(g_self) == 0) {
                ++idle;
                std::this_thread::yield();
            } else {
                idle = 0;
            }
        }
        result.ok = g_self->occupied_count == 0 && !fdwic_trace_is_fatal();
        (void)role;
    }

    static void reset_ninety_six_workers() {
        g_dist.num_workers = 96;
        g_dist.num_blocks = 32;
        for (uint32_t task = 0; task < kFdwicSharedPaTaskCapacity; ++task) {
            reset_task_cell(static_cast<int32_t>(task));
        }
        dist_shared_pa_tensor_map_reset(g_dist.shared_pa);
        for (int32_t shard = 0; shard < kCursorShards; ++shard) {
            g_dist.cube_cursor[shard].v = -1;
            g_dist.vector_cursor[shard].v = -1;
            g_dist.alloc_cursor[shard].v = -1;
        }
        for (int32_t core = 0; core < 32; ++core) {
            dist_core_reset(g_dist.cores[core], CoreType::AIC, core, LANE_AIC);
            g_dist.cores[core].core_idx = core;
        }
        for (int32_t ordinal = 0; ordinal < 64; ++ordinal) {
            const int32_t core = 32 + ordinal;
            const int32_t lane = (ordinal & 1) == 0 ? LANE_AIV0 : LANE_AIV1;
            dist_core_reset(g_dist.cores[core], CoreType::AIV, ordinal / 2, lane);
            g_dist.cores[core].core_idx = core;
        }
    }

    Runtime runtime_;
    void *heap_{nullptr};
};

TEST_F(FdwicSharedPaSubmitTest, TwoRolesPublishOneFiveTaskBatchAndOnlyWinnersBuildArgs) {
    SharedPaRunResult aic;
    SharedPaRunResult aiv;
    std::thread aic_thread([&] { replay_one(CoreType::AIC, 0, aic); });
    std::thread aiv_thread([&] { replay_one(CoreType::AIV, 1, aiv); });
    aic_thread.join();
    aiv_thread.join();

    EXPECT_TRUE(aic.ok) << "stage=" << aic.completed_stage << " callbacks=" << aic.callback_mask;
    EXPECT_TRUE(aiv.ok) << "stage=" << aiv.completed_stage << " callbacks=" << aiv.callback_mask;
    EXPECT_EQ(aic.callbacks + aiv.callbacks, 5U);
    EXPECT_NE(aic.callback_mask & (1U << 0), 0U);
    EXPECT_EQ(aiv.callback_mask & (1U << 0), 0U);
    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_NONE);
    for (int32_t task = 0; task < 5; ++task) {
        EXPECT_EQ(g_dist.tasks[task].deps_prepared, task);
        EXPECT_EQ(g_dist.tasks[task].flag, 1);
    }
    const uint32_t output_counts[5] = {3, 1, 3, 1, 0};
    for (int32_t task = 0; task < 5; ++task) {
        for (uint32_t slot = 0; slot < output_counts[task]; ++slot) {
            EXPECT_EQ(g_dist.shared_pa.shared_outputs[task].published[slot].v, task);
        }
    }
    EXPECT_EQ(g_dist.shared_pa.shared_outputs[0].last_writer[0].v, 4);
    EXPECT_EQ(g_dist.shared_pa.writer_history[4].magic, kFdwicSharedWriterHistoryMagic);
    EXPECT_EQ(g_dist.shared_pa.writer_history[4].writer_task, 4);
    EXPECT_EQ(g_dist.shared_pa.writer_history[4].count, 3U);
    EXPECT_GT(g_dist.shared_pa.shared_heap_vend.v, 0);
}

TEST_F(FdwicSharedPaSubmitTest, NinetySixWorkersConvergeWithExactlyFiveWinnerCallbacks) {
    reset_ninety_six_workers();

    std::vector<SharedPaRunResult> results(96);
    std::vector<std::thread> workers;
    workers.reserve(96);
    for (int32_t core = 0; core < 96; ++core) {
        workers.emplace_back([&, core] {
            replay_one(core < 32 ? CoreType::AIC : CoreType::AIV, core, results[core]);
        });
    }
    for (std::thread &worker : workers) worker.join();

    uint32_t callback_count = 0;
    for (int32_t core = 0; core < 96; ++core) {
        EXPECT_TRUE(results[core].ok)
            << "core=" << core << " stage=" << results[core].completed_stage
            << " callbacks=" << results[core].callback_mask;
        callback_count += results[core].callbacks;
    }
    EXPECT_EQ(callback_count, 5U);
    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_NONE);
    for (int32_t task = 0; task < 5; ++task) {
        EXPECT_EQ(g_dist.tasks[task].deps_prepared, task);
        EXPECT_EQ(g_dist.tasks[task].flag, 1);
    }
}

TEST_F(FdwicSharedPaSubmitTest, AllocClaimUsesAllWorkersThroughEightLocalGroups) {
    reset_ninety_six_workers();

    uint32_t total_attempts = 0;
    uint32_t total_winners = 0;
    for (uint32_t batch = 0; batch < kFdwicSharedPaBatches; ++batch) {
        const int32_t task_id =
            static_cast<int32_t>(batch * kFdwicSharedPaTasksPerBatch);
        uint32_t task_attempts = 0;
        uint32_t task_winners = 0;
        int32_t winner_core = -1;
        for (int32_t core = 0; core < 96; ++core) {
            DistSharedPaBeginState state{};
            state.self = &g_dist.cores[core];
            state.task_id = task_id;
            g_self = state.self;
            const DistSharedPaReplayContext replay = dist_shared_pa_replay_context();
            ASSERT_TRUE(replay.ready());
            const bool won =
                dist_shared_pa_claim(
                    replay.role(), replay.block_id(),
                    DistSharedPaTaskKind::Alloc, nullptr, state
                );
            const bool expected_attempt = true;
            EXPECT_EQ(state.claim_attempted, expected_attempt)
                << "task=" << task_id << " core=" << core;
            EXPECT_EQ(won, state.won);
            if (state.claim_attempted) {
                ++task_attempts;
                ++total_attempts;
            }
            if (won) {
                ++task_winners;
                ++total_winners;
                winner_core = core;
            }
        }
        EXPECT_EQ(task_attempts, kFdwicSharedWorkers) << "task=" << task_id;
        EXPECT_EQ(task_winners, 1U) << "task=" << task_id;
        ASSERT_GE(winner_core, 0);
        EXPECT_LT(winner_core, static_cast<int32_t>(kFdwicSharedWorkers));
        EXPECT_EQ(g_dist.shared_pa.claim_tournament[task_id].root.owner.v, task_id);
        for (uint32_t group = 0; group < kFdwicSharedAllocClaimTournamentGroups; ++group) {
            EXPECT_EQ(g_dist.shared_pa.claim_tournament[task_id].local[group].owner.v, task_id)
                << "task=" << task_id << " group=" << group;
        }
    }
    EXPECT_EQ(total_attempts, kFdwicSharedPaBatches * kFdwicSharedWorkers);
    EXPECT_EQ(total_winners, kFdwicSharedPaBatches);
    EXPECT_EQ(g_dist.fatal, 0);
}

TEST_F(FdwicSharedPaSubmitTest, NinetySixWorkersConvergeAcrossFullB256TaskSequence) {
    static_assert(
        kFdwicSharedPaBatches * kFdwicSharedPaTasksPerBatch ==
        kFdwicSharedPaTaskCapacity
    );
    reset_ninety_six_workers();

    std::vector<SharedPaRunResult> results(96);
    std::vector<std::thread> workers;
    workers.reserve(96);
    for (int32_t core = 0; core < 96; ++core) {
        workers.emplace_back([&, core] {
            replay_one(
                core < 32 ? CoreType::AIC : CoreType::AIV, core,
                results[core], kFdwicSharedPaBatches
            );
        });
    }
    for (std::thread &worker : workers) worker.join();

    uint32_t callback_count = 0;
    for (int32_t core = 0; core < 96; ++core) {
        EXPECT_TRUE(results[core].ok)
            << "core=" << core << " stage=" << results[core].completed_stage
            << " callbacks=" << results[core].callbacks;
        EXPECT_EQ(results[core].completed_stage, kFdwicSharedPaTaskCapacity);
        callback_count += results[core].callbacks;
    }
    EXPECT_EQ(callback_count, kFdwicSharedPaTaskCapacity);
    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_NONE);
    for (uint32_t task = 0; task < kFdwicSharedPaTaskCapacity; ++task) {
        EXPECT_EQ(g_dist.tasks[task].deps_prepared, static_cast<int64_t>(task))
            << "task=" << task;
        EXPECT_EQ(g_dist.tasks[task].flag, 1) << "task=" << task;
    }
    const uint32_t last_up = kFdwicSharedPaTaskCapacity - 1U;
    EXPECT_EQ(
        g_dist.shared_pa.writer_history[last_up].magic,
        kFdwicSharedWriterHistoryMagic
    );
    EXPECT_EQ(g_dist.shared_pa.writer_history[last_up].writer_task, last_up);
}

TEST_F(FdwicSharedPaSubmitTest, GenericSharedSubmitFailsClosedBeforeClaim) {
    g_self = &g_dist.cores[0];
    L0TaskArgs args;
    args.reset();
    MixedKernels mixed;
    mixed.aic_kernel_id = 0;
    const TaskOutputTensors outputs = dist_submit_impl(nullptr, mixed, args);
    EXPECT_TRUE(outputs.empty());
    EXPECT_NE(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_TENSORMAP_PROTOCOL);
    EXPECT_EQ(g_self->local_index, kFlagCap);
}

TEST_F(FdwicSharedPaSubmitTest, ReplayContextSnapshotsAttachedCoreIdentity) {
    static_assert(!std::is_aggregate<DistSharedPaReplayContext>::value);
    g_self = &g_dist.cores[1];

    const DistSharedPaReplayContext replay = dist_shared_pa_replay_context();

    ASSERT_TRUE(replay.ready());
    EXPECT_EQ(replay.role(), CoreType::AIV);
    EXPECT_EQ(replay.block_id(), g_self->block_id);

    g_self->role = CoreType::AIC;
    EXPECT_EQ(replay.role(), CoreType::AIV);
}

TEST_F(FdwicSharedPaSubmitTest, DefaultReplayFailsBeforeAdvancingCoreOrClaimCursor) {
    g_self = &g_dist.cores[0];
    const DistSharedPaReplayContext replay;

    const DistCompeteFirstTicket ticket = dist_shared_pa_alloc_begin(nullptr, replay);

    EXPECT_EQ(ticket.ready, 0);
    EXPECT_EQ(g_self->local_index, kFlagCap);
    EXPECT_EQ(g_dist.alloc_cursor[0].v, -1);
    EXPECT_NE(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_DIST_CONFIG_INVALID);
}

TEST_F(FdwicSharedPaSubmitTest, WinnerFinishRejectsIdentityChangedAfterReplaySnapshot) {
    g_self = &g_dist.cores[0];
    const DistSharedPaReplayContext replay = dist_shared_pa_replay_context();
    ASSERT_TRUE(replay.ready());
    const DistCompeteFirstTicket ticket = dist_shared_pa_alloc_begin(nullptr, replay);
    ASSERT_EQ(ticket.ready, 1);
    ASSERT_EQ(ticket.won, 1);

    const uint32_t vector_shape[1] = {16};
    TensorCreateInfo vector_ci(vector_shape, 1, DataType::FLOAT32);
    const uint32_t scalar_shape[1] = {1};
    TensorCreateInfo scalar_ci(scalar_shape, 1, DataType::FLOAT32);
    L0TaskArgs args;
    args.reset();
    args.add_output(vector_ci, scalar_ci, scalar_ci);
    g_self->block_id = replay.block_id() + 1;

    EXPECT_FALSE(dist_shared_pa_alloc_finish(nullptr, replay, ticket, &args));
    EXPECT_EQ(g_self->local_index, kFlagCap);
    EXPECT_NE(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_TENSORMAP_PROTOCOL);
    EXPECT_EQ(g_dist.shared_pa.shared_heap_vend.v, 0);
}

TEST_F(FdwicSharedPaSubmitTest, MalformedCreateInfoFailsBeforeMutatingSharedHeapControls) {
    g_self = &g_dist.cores[0];
    L0TaskArgs args;
    args.reset();
    const uint32_t vector_shape[1] = {16};
    TensorCreateInfo malformed(vector_shape, 1, DataType::FLOAT32);
    const uint32_t scalar_shape[1] = {1};
    TensorCreateInfo scalar_ci(scalar_shape, 1, DataType::FLOAT32);
    // Corrupt the descriptor after construction so this exercises the
    // production validation boundary rather than the constructor assertion.
    // Keep it last to prove all outputs are preflighted before the first
    // descriptor is written.
    malformed.ndims = 0;
    args.add_output(scalar_ci, scalar_ci, malformed);

    DistSubmitCtx ctx;
    dist_submit_begin(g_self, args, ctx);
    ASSERT_EQ(ctx.task_id, 0);
    std::memset(ctx.payload, 0xA5, sizeof(*ctx.payload));
    std::vector<uint8_t> payload_before(sizeof(*ctx.payload));
    std::memcpy(payload_before.data(), ctx.payload, payload_before.size());
    DistSharedPaMaterializePlan plan;
    ASSERT_TRUE(dist_shared_pa_validate_and_plan(
        args, ctx.task_id, DistSharedPaTaskKind::Alloc, plan
    ));
    EXPECT_FALSE(dist_shared_pa_materialize_args(args, ctx, plan));

    EXPECT_NE(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_TENSORMAP_PROTOCOL);
    EXPECT_EQ(g_self->local_index, kFlagCap);
    EXPECT_EQ(g_dist.shared_pa.shared_heap_vend.v, 0);
    for (uint32_t shard = 0; shard < kFdwicSharedHeapShards; ++shard) {
        EXPECT_EQ(g_dist.shared_pa.shared_heap_cursor[shard].v, 0)
            << "shard=" << shard;
    }
    for (uint32_t slot = 0; slot < kFdwicSharedOutputMaxPerTask; ++slot) {
        EXPECT_EQ(g_dist.shared_pa.shared_outputs[0].published[slot].v, -1);
        EXPECT_EQ(g_dist.shared_pa.shared_outputs[0].last_writer[slot].v, -1);
    }
    EXPECT_EQ(g_dist.shared_pa.writer_history[0].magic, 0U);
    EXPECT_EQ(g_dist.tasks[0].deps_prepared, -1);
    EXPECT_EQ(ctx.result.size(), 0U);
    EXPECT_EQ(
        std::memcmp(payload_before.data(), ctx.payload, payload_before.size()), 0
    );
}

TEST_F(FdwicSharedPaSubmitTest, FixedStageSchemasProduceExactMaterializePlans) {
    Tensor q = make_external(0x100000);
    Tensor k = make_external(0x200000);
    Tensor v = make_external(0x300000);
    Tensor table = make_external(0x400000);
    Tensor out = make_external(0x500000, true);
    const uint32_t vector_shape[1] = {16};
    TensorCreateInfo vector_ci(vector_shape, 1, DataType::FLOAT32);
    const uint32_t scalar_shape[1] = {1};
    TensorCreateInfo scalar_ci(scalar_shape, 1, DataType::FLOAT32);
    FdwicOutputRef qk_sij{1, 0, 0, 0, 0, 0};
    FdwicOutputRef sf_pij{2, 0, 0, 0, 0, 0};
    FdwicOutputRef sf_mi{2, 1, 0, 0, 0, 0};
    FdwicOutputRef sf_li{2, 2, 0, 0, 0, 0};
    FdwicOutputRef pv_oi{3, 0, 0, 0, 0, 0};
    FdwicOutputRef alloc_oi{0, 0, 0, 0, 0, 0};
    FdwicOutputRef alloc_li{0, 1, 0, 0, 0, 0};
    FdwicOutputRef alloc_mi{0, 2, 0, 0, 0, 0};

    L0TaskArgs alloc;
    alloc.reset();
    alloc.add_output(vector_ci, scalar_ci, scalar_ci);
    DistSharedPaMaterializePlan plan;
    ASSERT_TRUE(dist_shared_pa_validate_and_plan(
        alloc, 0, DistSharedPaTaskKind::Alloc, plan
    ));
    EXPECT_EQ(plan.output_count, 3U);
    EXPECT_EQ(plan.output_start, 0U);
    EXPECT_EQ(plan.register_mask, 0U);

    L0TaskArgs qk;
    qk.reset();
    qk.add_input(q, k, table);
    qk.add_output(vector_ci);
    qk.add_scalar(uint64_t{1}, uint64_t{0});
    ASSERT_TRUE(dist_shared_pa_validate_and_plan(
        qk, 1, DistSharedPaTaskKind::Qk, plan
    ));
    EXPECT_EQ(plan.output_count, 1U);
    EXPECT_EQ(plan.output_start, 3U);
    EXPECT_EQ(plan.register_mask, 0U);

    L0TaskArgs sf;
    sf.reset();
    sf.add_input(qk_sij);
    sf.add_output(vector_ci, scalar_ci, scalar_ci);
    sf.add_scalar(uint64_t{1}, uint64_t{1}, uint64_t{1});
    ASSERT_TRUE(dist_shared_pa_validate_and_plan(
        sf, 2, DistSharedPaTaskKind::Sf, plan
    ));
    EXPECT_EQ(plan.output_count, 3U);
    EXPECT_EQ(plan.output_start, 1U);
    EXPECT_EQ(plan.register_mask, 0U);

    L0TaskArgs pv;
    pv.reset();
    pv.add_input(sf_pij, v, table);
    pv.add_output(vector_ci);
    pv.add_scalar(uint64_t{1}, uint64_t{0});
    ASSERT_TRUE(dist_shared_pa_validate_and_plan(
        pv, 3, DistSharedPaTaskKind::Pv, plan
    ));
    EXPECT_EQ(plan.output_count, 1U);
    EXPECT_EQ(plan.output_start, 3U);
    EXPECT_EQ(plan.register_mask, 0U);

    L0TaskArgs up;
    up.reset();
    up.add_input(sf_mi, sf_li, pv_oi);
    up.add_inout(alloc_mi, alloc_li, alloc_oi, out);
    up.add_scalar(uint64_t{1}, uint64_t{1});
    ASSERT_TRUE(dist_shared_pa_validate_and_plan(
        up, 4, DistSharedPaTaskKind::Up, plan
    ));
    EXPECT_EQ(plan.output_count, 0U);
    EXPECT_EQ(plan.register_mask, (1U << 3) | (1U << 4) | (1U << 5));
}

TEST_F(FdwicSharedPaSubmitTest, InvalidStageSchemasDoNotProduceMaterializePlans) {
    const uint32_t shape[1] = {16};
    TensorCreateInfo ci(shape, 1, DataType::FLOAT32);
    Tensor out_without_manual_dep = make_external(0x500000, false);
    FdwicOutputRef wrong_qk_ref{1, 1, 0, 0, 0, 0};
    FdwicOutputRef sf_mi{2, 1, 0, 0, 0, 0};
    FdwicOutputRef sf_li{2, 2, 0, 0, 0, 0};
    FdwicOutputRef pv_oi{3, 0, 0, 0, 0, 0};
    FdwicOutputRef alloc_oi{0, 0, 0, 0, 0, 0};
    FdwicOutputRef alloc_li{0, 1, 0, 0, 0, 0};
    FdwicOutputRef alloc_mi{0, 2, 0, 0, 0, 0};
    DistSharedPaMaterializePlan plan;

    L0TaskArgs alloc_with_error;
    alloc_with_error.reset();
    alloc_with_error.add_output(ci, ci, ci);
    alloc_with_error.has_error = true;
    EXPECT_FALSE(dist_shared_pa_validate_and_plan(
        alloc_with_error, 0, DistSharedPaTaskKind::Alloc, plan
    ));
    EXPECT_EQ(plan.output_start, 0U);
    EXPECT_EQ(plan.output_count, 0U);

    L0TaskArgs sf_wrong_ref;
    sf_wrong_ref.reset();
    sf_wrong_ref.add_input(wrong_qk_ref);
    sf_wrong_ref.add_output(ci, ci, ci);
    sf_wrong_ref.add_scalar(uint64_t{1}, uint64_t{1}, uint64_t{1});
    EXPECT_FALSE(dist_shared_pa_validate_and_plan(
        sf_wrong_ref, 2, DistSharedPaTaskKind::Sf, plan
    ));
    EXPECT_EQ(plan.output_start, 0U);
    EXPECT_EQ(plan.output_count, 0U);

    L0TaskArgs up_without_manual_dep;
    up_without_manual_dep.reset();
    up_without_manual_dep.add_input(sf_mi, sf_li, pv_oi);
    up_without_manual_dep.add_inout(
        alloc_mi, alloc_li, alloc_oi, out_without_manual_dep
    );
    up_without_manual_dep.add_scalar(uint64_t{1}, uint64_t{1});
    EXPECT_FALSE(dist_shared_pa_validate_and_plan(
        up_without_manual_dep, 4, DistSharedPaTaskKind::Up, plan
    ));
    EXPECT_EQ(plan.output_start, 0U);
    EXPECT_EQ(plan.output_count, 0U);
    EXPECT_EQ(plan.register_mask, 0U);
}

TEST_F(FdwicSharedPaSubmitTest, OversizeOutputFailsBeforePayloadOrPublicationMutation) {
    g_self = &g_dist.cores[0];
    L0TaskArgs args;
    args.reset();
    const uint32_t oversize_shape[1] = {
        static_cast<uint32_t>(kFdwicSharedHeapShardBytes / sizeof(float)) + 1U
    };
    TensorCreateInfo oversize(oversize_shape, 1, DataType::FLOAT32);
    const uint32_t scalar_shape[1] = {1};
    TensorCreateInfo scalar_ci(scalar_shape, 1, DataType::FLOAT32);
    args.add_output(oversize, scalar_ci, scalar_ci);

    DistSubmitCtx ctx;
    dist_submit_begin(g_self, args, ctx);
    ASSERT_EQ(ctx.task_id, 0);
    std::memset(ctx.payload, 0x5A, sizeof(*ctx.payload));
    std::vector<uint8_t> payload_before(sizeof(*ctx.payload));
    std::memcpy(payload_before.data(), ctx.payload, payload_before.size());
    DistSharedPaMaterializePlan plan;
    ASSERT_TRUE(dist_shared_pa_validate_and_plan(
        args, ctx.task_id, DistSharedPaTaskKind::Alloc, plan
    ));
    EXPECT_FALSE(dist_shared_pa_materialize_args(args, ctx, plan));

    EXPECT_NE(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_TENSORMAP_CAPACITY);
    EXPECT_EQ(g_dist.shared_pa.shared_heap_vend.v, 0);
    for (uint32_t shard = 0; shard < kFdwicSharedHeapShards; ++shard) {
        EXPECT_EQ(g_dist.shared_pa.shared_heap_cursor[shard].v, 0);
    }
    for (uint32_t slot = 0; slot < kFdwicSharedOutputMaxPerTask; ++slot) {
        EXPECT_EQ(g_dist.shared_pa.shared_outputs[0].published[slot].v, -1);
        EXPECT_EQ(g_dist.shared_pa.shared_outputs[0].last_writer[slot].v, -1);
    }
    EXPECT_EQ(g_dist.tasks[0].deps_prepared, -1);
    EXPECT_EQ(ctx.result.size(), 0U);
    EXPECT_EQ(
        std::memcmp(payload_before.data(), ctx.payload, payload_before.size()), 0
    );
}

TEST_F(FdwicSharedPaSubmitTest, ExternalScalarReadDoesNotInvokeUnsupportedRegionLookup) {
    g_self = &g_dist.cores[0];
    int32_t values[2] = {17, 29};
    const uint32_t shape[1] = {2};
    const Tensor external =
        make_tensor_external(values, shape, 1, DataType::INT32, false);
    const uint32_t index[1] = {1};
    EXPECT_EQ(dist_get_tensor_data_impl(nullptr, external, 1, index), 29U);
    EXPECT_EQ(g_dist.fatal, 0);
    EXPECT_EQ(g_dist.error_code, PTO2_ERROR_NONE);
}

TEST_F(FdwicSharedPaSubmitTest, SharedDrainDropsReadyPrefixAndSkipsReserveSlots) {
    g_self = &g_dist.cores[0];
    RingSlot &slot = g_self->slots[0];
    slot.occupied = true;
    slot.built = true;
    slot.fanin_count = 2;
    slot.fanin[0] = 0;
    slot.fanin[1] = 1;
    g_self->occupied_count = 1;
    g_dist.tasks[0].flag = 1;
    g_dist.tasks[1].flag = 0;

    EXPECT_EQ(drain_phase_b(g_self), 0);
    ASSERT_EQ(slot.fanin_count, 1);
    EXPECT_EQ(slot.fanin[0], 1);
    EXPECT_TRUE(slot.occupied);

    slot.occupied = false;
    slot.built = false;
    RingSlot &reserve = g_self->slots[kPrivateSlots - 1];
    reserve.occupied = true;
    reserve.built = true;
    reserve.fanin_count = 0;
    g_self->occupied_count = 1;

    EXPECT_EQ(drain_phase_b(g_self), 0);
    EXPECT_TRUE(reserve.occupied);
    EXPECT_EQ(g_self->occupied_count, 1);
}

TEST_F(FdwicSharedPaSubmitTest, OpportunisticEfDrainBacksOffOnlyForOneStalledSlot) {
    DistCore &worker = g_dist.cores[0];
    worker.occupied_count = 1;
    worker.slots[0].occupied = true;
    worker.slots[0].built = false;
    worker.slots_pad[kFdwicSharedEfDrainSkipBudgetByte] = 0;
    worker.slots_pad[kFdwicSharedEfDrainNoProgressByte] = 0;

    EXPECT_EQ(dist_shared_pa_opportunistic_drain(&worker), 0);
    EXPECT_EQ(
        worker.slots_pad[kFdwicSharedEfDrainSkipBudgetByte],
        kFdwicSharedEfDrainNoProgressSkipSubmits
    );
    EXPECT_EQ(worker.slots_pad[kFdwicSharedEfDrainNoProgressByte], 1);

    // The next Submit skips; the following one polls again.
    EXPECT_EQ(dist_shared_pa_opportunistic_drain(&worker), 0);
    EXPECT_EQ(worker.slots_pad[kFdwicSharedEfDrainSkipBudgetByte], 0);
    EXPECT_EQ(worker.slots_pad[kFdwicSharedEfDrainNoProgressByte], 1);

    worker.slots_pad[kFdwicSharedEfDrainNoProgressByte] =
        kFdwicSharedEfDrainLongWaitPollThreshold - 1;
    EXPECT_EQ(dist_shared_pa_opportunistic_drain(&worker), 0);
    EXPECT_EQ(
        worker.slots_pad[kFdwicSharedEfDrainSkipBudgetByte],
        kFdwicSharedEfDrainLongWaitSkipSubmits
    );
    EXPECT_EQ(
        worker.slots_pad[kFdwicSharedEfDrainNoProgressByte],
        kFdwicSharedEfDrainLongWaitPollThreshold
    );

    worker.occupied_count = 2;
    worker.slots[1].occupied = true;
    worker.slots[1].built = false;
    EXPECT_EQ(dist_shared_pa_opportunistic_drain(&worker), 0);
    EXPECT_EQ(worker.slots_pad[kFdwicSharedEfDrainSkipBudgetByte], 0);
    EXPECT_EQ(worker.slots_pad[kFdwicSharedEfDrainNoProgressByte], 0);

    worker.occupied_count = 0;
    EXPECT_EQ(dist_shared_pa_opportunistic_drain(&worker), 0);
    EXPECT_EQ(worker.slots_pad[kFdwicSharedEfDrainSkipBudgetByte], 0);
    EXPECT_EQ(worker.slots_pad[kFdwicSharedEfDrainNoProgressByte], 0);
}

}  // namespace
