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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "dist_engine/aicpu/cross_core_exec_init.h"

namespace {

using namespace fdwic::cross_core;

TEST(FdwicCrossCoreDagState, AppendsMetadataAfterTheProvenExecutionPrefix) {
    static_assert(PTO_FDWIC_SHARED_MAP == 1);
    static_assert(PTO_FDWIC_SCHEDULER_MODE == 2);
    EXPECT_EQ(kFdwicCrossCoreTaskCapacity, 2048U);
    EXPECT_EQ(offsetof(CrossCoreDagState, runtime), 0U);
    EXPECT_EQ(offsetof(CrossCoreDagState, metadata), sizeof(CrossCoreRuntimeState));
    EXPECT_EQ(sizeof(CrossCoreDagState), sizeof(CrossCoreRuntimeState) + 2048U * sizeof(DagTaskMetadataCell));
    EXPECT_EQ(offsetof(DistGlobal, shared_pa), kFdwicSharedTensorMapOffset);
    EXPECT_EQ(offsetof(DistGlobal, cross_core_dag), kFdwicSharedTensorMapOffset + sizeof(SharedPaTensorMapState));
    EXPECT_LE(sizeof(DistGlobal), kDistEngineGlobalStateSize);
}

TEST(FdwicCrossCoreDagState, ResetTouchesEveryPublicationControl) {
    auto state = std::make_unique<CrossCoreDagState>();
    std::memset(state.get(), 0x5a, sizeof(*state));

    dist_cross_core_dag_reset(*state);

    EXPECT_EQ(state->runtime.fatal.state, 0);
    EXPECT_EQ(state->runtime.heap_cursor.state, 0);
    for (uint32_t task = 0; task < kFdwicCrossCoreTaskCapacity; ++task) {
        EXPECT_EQ(state->runtime.execute_owner[task].state, 0) << "task=" << task;
        EXPECT_EQ(state->runtime.tasks[task].control.state, 0) << "task=" << task;
        EXPECT_EQ(state->runtime.outputs[task].control.state, 0) << "task=" << task;
        EXPECT_EQ(state->runtime.build_tournament[task].root.owner.v, -1) << "task=" << task;
        for (uint32_t group = 0; group < kFdwicSharedClaimTournamentMaxGroups; ++group) {
            EXPECT_EQ(state->runtime.build_tournament[task].local[group].owner.v, -1)
                << "task=" << task << " group=" << group;
        }
        EXPECT_EQ(state->metadata[task].control.state, 0) << "task=" << task;
    }
    for (uint32_t bucket = 0; bucket < kCrossMapBuckets; ++bucket) {
        EXPECT_EQ(state->runtime.tensor_map.tails[bucket].state, 0) << "bucket=" << bucket;
    }
    for (uint32_t slot = 0; slot < kCrossMapCapacity; ++slot) {
        EXPECT_EQ(state->runtime.tensor_map.slots[slot].sequence.state, -1) << "slot=" << slot;
    }
    // Reset only publication controls; immutable payload bytes remain diagnostic
    // evidence until their corresponding control is published in the new run.
    EXPECT_EQ(reinterpret_cast<const uint8_t *>(&state->metadata[0].payload)[0], 0x5a);
}

}  // namespace
