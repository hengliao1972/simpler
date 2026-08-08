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
#include <cstdint>
#include <cstring>

#include "dist_engine/common/cross_core_exec_protocol.h"

namespace {

using namespace fdwic::cross_core;

struct HostOps {
    static inline uint32_t sequence = 0;
    static inline uint32_t flush_sequence = 0;
    static inline uint32_t built_sequence = 0;
    static inline uint32_t invalidate_sequence = 0;
    static inline uint32_t flush_count = 0;
    static inline uint32_t invalidate_count = 0;

    static void Reset() {
        sequence = 0;
        flush_sequence = 0;
        built_sequence = 0;
        invalidate_sequence = 0;
        flush_count = 0;
        invalidate_count = 0;
    }

    static int64_t Load(volatile int64_t *address) { return __atomic_load_n(address, __ATOMIC_ACQUIRE); }

    static int64_t CompareExchange(volatile int64_t *address, int64_t expected, int64_t desired) {
        const DecodedExecState decoded = DecodeExecState(desired);
        if (decoded.valid && decoded.phase == ExecPhase::Built) built_sequence = ++sequence;
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(address, &observed, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
        return observed;
    }

    static void StorePayloadWord(volatile uint64_t *address, uint64_t value) { *address = value; }

    static void FlushRegion(const volatile void *, uint64_t) {
        ++flush_count;
        flush_sequence = ++sequence;
        __atomic_thread_fence(__ATOMIC_RELEASE);
    }

    static void InvalidateRegion(const volatile void *, uint64_t) {
        ++invalidate_count;
        invalidate_sequence = ++sequence;
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
    }
};

struct PayloadSource {
    alignas(64) uint64_t inline_tensor[kExecTensorDescWords]{};
    alignas(64) uint64_t referenced_tensor[kExecTensorDescWords]{};
    std::array<uint64_t, 2> scalars{0x1234, 0x5678};
    std::array<int32_t, 2> fanin{2, 6};

    PayloadSource() {
        for (uint32_t word = 0; word < kExecTensorDescWords; ++word) {
            inline_tensor[word] = 0x1000 + word;
            referenced_tensor[word] = 0x2000 + word;
        }
    }

    uint64_t TensorReference(uint32_t tensor) const {
        return tensor == 1 ? reinterpret_cast<uintptr_t>(referenced_tensor) : 0;
    }
    uint64_t TensorWord(uint32_t tensor, uint32_t word) const { return tensor == 0 ? inline_tensor[word] : 0; }
    uint64_t Scalar(uint32_t index) const { return scalars[index]; }
    int32_t Fanin(uint32_t index) const { return fanin[index]; }
};

ExecPayloadSpec MakeSpec() {
    return ExecPayloadSpec{
        .task_id = 9,
        .function_address = 0x4000,
        .completion_vend = 0x8000,
        .function_id = 3,
        .tensor_count = 2,
        .scalar_count = 2,
        .fanin_count = 2,
        .engine_class = ExecEngineClass::Aic,
        .flags = 0,
        .multicore_group_id = 0,
        .multicore_rank = 0,
        .multicore_size = 1,
        .tensor_reference_mask = 1U << 1,
    };
}

TEST(FdwicCrossCoreExecProtocol, ControlAndImmutablePayloadNeverShareACacheLine) {
    EXPECT_EQ(sizeof(SharedExecControl), 64U);
    EXPECT_EQ(alignof(SharedExecControl), 64U);
    EXPECT_EQ(offsetof(SharedExecCell, payload), 64U);
    EXPECT_EQ(sizeof(ExecPayloadStorage), 4352U);
    EXPECT_EQ(sizeof(SharedExecCell), 4416U);
    SharedExecCell cell{};
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&cell.payload) % 64U, 0U);
}

TEST(FdwicCrossCoreExecProtocol, StateEncodingPreservesIndependentBuildAndExecuteOwners) {
    const int64_t raw = static_cast<int64_t>(EncodeExecState(ExecPhase::Claimed, 7, 23, ExecEngineClass::Aiv, 11, 913));
    const DecodedExecState decoded = DecodeExecState(raw);
    ASSERT_TRUE(decoded.valid);
    EXPECT_EQ(decoded.phase, ExecPhase::Claimed);
    EXPECT_EQ(decoded.build_owner, 7U);
    EXPECT_EQ(decoded.execute_owner, 23U);
    EXPECT_EQ(decoded.engine_class, ExecEngineClass::Aiv);
    EXPECT_EQ(decoded.payload_lines, 11U);
    EXPECT_EQ(decoded.task_id, 913U);
}

TEST(FdwicCrossCoreExecProtocol, BuildFlushesBeforePublishAndACompatibleOwnerAcquires) {
    SharedExecCell cell{};
    SharedExecControl fatal{};
    PayloadSource source;
    ExecToken token{};
    ResetExecToken(token);
    HostOps::Reset();

    EXPECT_EQ(BuildAndPublishExecPayload<HostOps>(cell, 7, MakeSpec(), source, fatal), ExecBuildResult::Published);
    EXPECT_EQ(HostOps::flush_count, 1U);
    EXPECT_GT(HostOps::flush_sequence, 0U);
    EXPECT_GT(HostOps::built_sequence, HostOps::flush_sequence);

    const DecodedExecState built = DecodeExecState(cell.control.state);
    ASSERT_TRUE(built.valid);
    EXPECT_EQ(built.phase, ExecPhase::Built);
    EXPECT_EQ(built.build_owner, 7U);
    EXPECT_EQ(built.execute_owner, kExecUnboundOwner);

    EXPECT_EQ(
        AcquireExecPayload<HostOps>(cell, 9, 31, ExecEngineClass::Aiv, token, fatal), ExecAcquireResult::Incompatible
    );
    EXPECT_EQ(HostOps::invalidate_count, 0U);

    EXPECT_EQ(
        AcquireExecPayload<HostOps>(cell, 9, 31, ExecEngineClass::Aic, token, fatal), ExecAcquireResult::Acquired
    );
    EXPECT_EQ(HostOps::invalidate_count, 1U);
    EXPECT_GT(HostOps::invalidate_sequence, HostOps::built_sequence);
    EXPECT_EQ(token.build_owner, 7U);
    EXPECT_EQ(token.execute_owner, 31U);
    EXPECT_EQ(token.header.task_id, 9U);
    EXPECT_EQ(token.header.function_id, 3U);
    EXPECT_EQ(token.header.tensor_reference_mask, 1U << 1);

    EXPECT_EQ(PublishExecDone<HostOps>(cell, token, fatal), ExecDoneResult::Done);
    EXPECT_EQ(token.phase, ExecTokenPhase::Idle);
    const DecodedExecState done = DecodeExecState(cell.control.state);
    ASSERT_TRUE(done.valid);
    EXPECT_EQ(done.phase, ExecPhase::Done);
    EXPECT_EQ(done.build_owner, 7U);
    EXPECT_EQ(done.execute_owner, 31U);
}

TEST(FdwicCrossCoreExecProtocol, EmptyOrBuildingCellIsNotMistakenForCorruption) {
    SharedExecCell cell{};
    SharedExecControl fatal{};
    ExecToken token{};
    ResetExecToken(token);
    HostOps::Reset();

    EXPECT_EQ(
        AcquireExecPayload<HostOps>(cell, 37, 2, ExecEngineClass::Aic, token, fatal), ExecAcquireResult::NotBuilt
    );
    EXPECT_EQ(fatal.state, 0);

    cell.control.state =
        static_cast<int64_t>(EncodeExecState(ExecPhase::Building, 5, kExecUnboundOwner, ExecEngineClass::None, 0, 37));
    EXPECT_EQ(
        AcquireExecPayload<HostOps>(cell, 37, 2, ExecEngineClass::Aic, token, fatal), ExecAcquireResult::NotBuilt
    );
    EXPECT_EQ(fatal.state, 0);
}

TEST(FdwicCrossCoreExecProtocol, InvalidFaninFailsClosedBeforeBuiltPublication) {
    SharedExecCell cell{};
    SharedExecControl fatal{};
    PayloadSource source;
    source.fanin[1] = 9;
    HostOps::Reset();

    EXPECT_EQ(BuildAndPublishExecPayload<HostOps>(cell, 4, MakeSpec(), source, fatal), ExecBuildResult::InvalidInput);
    EXPECT_NE(fatal.state, 0);
    const DecodedExecState state = DecodeExecState(cell.control.state);
    ASSERT_TRUE(state.valid);
    EXPECT_EQ(state.phase, ExecPhase::Building);
}

TEST(FdwicCrossCoreExecProtocol, CorruptedPayloadCannotExecuteOrBeClaimedAgain) {
    SharedExecCell cell{};
    SharedExecControl fatal{};
    PayloadSource source;
    ExecToken first{};
    ExecToken second{};
    ResetExecToken(first);
    ResetExecToken(second);
    HostOps::Reset();

    ASSERT_EQ(BuildAndPublishExecPayload<HostOps>(cell, 8, MakeSpec(), source, fatal), ExecBuildResult::Published);
    cell.payload.words[7] = 1;
    EXPECT_EQ(
        AcquireExecPayload<HostOps>(cell, 9, 11, ExecEngineClass::Aic, first, fatal), ExecAcquireResult::InvalidPayload
    );
    EXPECT_EQ(first.phase, ExecTokenPhase::Faulted);
    EXPECT_NE(fatal.state, 0);
    EXPECT_EQ(
        AcquireExecPayload<HostOps>(cell, 9, 12, ExecEngineClass::Aic, second, fatal), ExecAcquireResult::FatalObserved
    );
}

}  // namespace
