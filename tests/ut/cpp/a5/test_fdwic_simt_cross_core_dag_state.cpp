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

constexpr uint8_t kPayloadPattern = 0x5a;

bool BytesEqual(const void *object, size_t offset, size_t bytes, uint8_t value) {
    const auto *data = static_cast<const uint8_t *>(object);
    for (size_t index = offset; index < offset + bytes; ++index) {
        if (data[index] != value) return false;
    }
    return true;
}

TEST(FdwicSimtCrossCoreDagState, AppendsMetadataAfterTheCommonDynamicRequestPrefix) {
    static_assert(PTO_FDWIC_SHARED_MAP == 1);
    static_assert(PTO_FDWIC_SCHEDULER_MODE == 4);
    EXPECT_EQ(kFdwicCrossCoreTaskCapacity, 2048U);
    EXPECT_EQ(offsetof(SimtCrossCoreDagState, runtime), 0U);
    EXPECT_EQ(offsetof(SimtCrossCoreDagState, lifecycle), sizeof(CrossCoreRuntimeState));
    EXPECT_EQ(
        offsetof(SimtCrossCoreDagState, requests), sizeof(CrossCoreRuntimeState) + sizeof(SimtBuilderLifecycleState)
    );
    EXPECT_EQ(
        offsetof(SimtCrossCoreDagState, metadata),
        sizeof(CrossCoreRuntimeState) + sizeof(SimtBuilderLifecycleState) +
            kFdwicCrossCoreTaskCapacity * sizeof(fdwic::cross_core::SimtBuildRequestCell)
    );
    EXPECT_EQ(
        sizeof(SimtCrossCoreDagState),
        sizeof(CrossCoreRuntimeState) + sizeof(SimtBuilderLifecycleState) +
            kFdwicCrossCoreTaskCapacity *
                (sizeof(fdwic::cross_core::SimtBuildRequestCell) + sizeof(fdwic::cross_core::DagTaskMetadataCell))
    );
    EXPECT_EQ(offsetof(DistGlobal, shared_pa), kFdwicSharedTensorMapOffset);
    EXPECT_EQ(offsetof(DistGlobal, simt_cross_core_dag), kFdwicSharedTensorMapOffset + sizeof(SharedPaTensorMapState));
    EXPECT_LE(sizeof(DistGlobal), kDistEngineGlobalStateSize);
}

TEST(FdwicSimtCrossCoreDagState, ResetTouchesEveryControlButPreservesImmutablePayloads) {
    auto state = std::make_unique<SimtCrossCoreDagState>();
    std::memset(state.get(), kPayloadPattern, sizeof(*state));

    dist_simt_cross_core_dag_reset(*state);

    EXPECT_EQ(state->runtime.fatal.state, 0);
    EXPECT_EQ(state->runtime.heap_cursor.state, 0);
    EXPECT_EQ(state->lifecycle.builder_started.state, 0);
    EXPECT_EQ(state->lifecycle.sealed_task_count.state, -1);
    EXPECT_EQ(state->lifecycle.builder_finished.state, 0);
    for (uint32_t task = 0; task < kFdwicCrossCoreTaskCapacity; ++task) {
        EXPECT_EQ(state->runtime.tasks[task].control.state, 0) << "task=" << task;
        EXPECT_EQ(state->requests[task].control.state, 0) << "task=" << task;
        EXPECT_EQ(state->metadata[task].control.state, 0) << "task=" << task;
    }
    EXPECT_TRUE(BytesEqual(
        &state->requests[0], offsetof(fdwic::cross_core::SimtBuildRequestCell, payload),
        sizeof(fdwic::cross_core::SimtBuildRequestStorage), kPayloadPattern
    ));
    EXPECT_TRUE(BytesEqual(
        &state->metadata[kFdwicCrossCoreTaskCapacity - 1U], offsetof(fdwic::cross_core::DagTaskMetadataCell, payload),
        sizeof(fdwic::cross_core::DagWriterPayload), kPayloadPattern
    ));
}

}  // namespace
