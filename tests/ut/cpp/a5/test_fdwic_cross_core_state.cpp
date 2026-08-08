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

constexpr uint8_t kPayloadPattern = 0x5a;

bool BytesEqual(const void *object, size_t offset, size_t bytes, uint8_t value) {
    const auto *data = static_cast<const uint8_t *>(object);
    for (size_t index = offset; index < offset + bytes; ++index) {
        if (data[index] != value) return false;
    }
    return true;
}

TEST(FdwicCrossCoreOrdinaryState, AppendsWithoutMovingTheExistingSharedState) {
    static_assert(PTO_FDWIC_SHARED_MAP == 1);
    static_assert(PTO_FDWIC_SCHEDULER_MODE == 1);
    EXPECT_EQ(kFdwicCrossCoreOrdinaryTaskCapacity, 2048U);
    EXPECT_EQ(offsetof(CrossCoreOrdinaryState, fatal), 0U);
    EXPECT_EQ(offsetof(CrossCoreOrdinaryState, heap_cursor), 64U);
    EXPECT_EQ(offsetof(CrossCoreOrdinaryState, tasks), 128U);
    EXPECT_EQ(offsetof(CrossCoreOrdinaryState, outputs), 128U + 2048U * sizeof(SharedExecCell));
    EXPECT_EQ(
        sizeof(CrossCoreOrdinaryState), 128U + 2048U * (sizeof(SharedExecCell) + sizeof(CrossCoreOutputCell<Tensor>))
    );
    EXPECT_EQ(offsetof(DistGlobal, shared_pa), kFdwicSharedTensorMapOffset);
    EXPECT_EQ(offsetof(DistGlobal, cross_core_ordinary), kFdwicSharedTensorMapOffset + sizeof(SharedPaTensorMapState));
    EXPECT_LE(sizeof(DistGlobal), kDistEngineGlobalStateSize);
}

TEST(FdwicCrossCoreOrdinaryState, ResetTouchesOnlyAtomicPublicationControls) {
    auto state = std::make_unique<CrossCoreOrdinaryState>();
    std::memset(state.get(), kPayloadPattern, sizeof(*state));

    dist_cross_core_ordinary_reset(*state);

    EXPECT_EQ(state->fatal.state, 0);
    EXPECT_EQ(state->heap_cursor.state, 0);
    for (uint32_t task = 0; task < kFdwicCrossCoreOrdinaryTaskCapacity; ++task) {
        EXPECT_EQ(state->tasks[task].control.state, 0) << "task=" << task;
        EXPECT_EQ(state->outputs[task].control.state, 0) << "task=" << task;
        EXPECT_TRUE(BytesEqual(
            &state->tasks[task], offsetof(SharedExecCell, payload), sizeof(ExecPayloadStorage), kPayloadPattern
        )) << "task="
           << task;
        EXPECT_TRUE(BytesEqual(
            &state->outputs[task], offsetof(CrossCoreOutputCell<Tensor>, descriptors),
            sizeof(state->outputs[task].descriptors), kPayloadPattern
        )) << "task="
           << task;
    }
}

}  // namespace
