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
#include <cstddef>
#include <cstdint>

#include "dist_engine/aicore/cross_core_simt_request_source.h"
#include "dist_engine/common/cross_core_simt_request_protocol.h"

namespace {

using namespace fdwic::cross_core;

struct HostOps {
    static inline uint32_t flush_count = 0;
    static inline uint32_t invalidate_count = 0;
    static inline uint64_t last_bytes = 0;
    static inline uintptr_t flush_addresses[2]{};
    static inline uint64_t flush_bytes[2]{};

    static void Reset() {
        flush_count = 0;
        invalidate_count = 0;
        last_bytes = 0;
        flush_addresses[0] = 0;
        flush_addresses[1] = 0;
        flush_bytes[0] = 0;
        flush_bytes[1] = 0;
    }

    static int64_t Load(volatile int64_t *address) { return __atomic_load_n(address, __ATOMIC_ACQUIRE); }

    static int64_t CompareExchange(volatile int64_t *address, int64_t expected, int64_t desired) {
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(address, &observed, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
        return observed;
    }

    static void StorePayloadWord(volatile uint64_t *address, uint64_t value) { *address = value; }

    static void FlushRegion(const volatile void *address, uint64_t bytes) {
        if (flush_count < 2) {
            flush_addresses[flush_count] = reinterpret_cast<uintptr_t>(address);
            flush_bytes[flush_count] = bytes;
        }
        ++flush_count;
        last_bytes = bytes;
        std::atomic_thread_fence(std::memory_order_release);
    }

    static void InvalidateRegion(const volatile void *, uint64_t bytes) {
        ++invalidate_count;
        last_bytes = bytes;
        std::atomic_thread_fence(std::memory_order_acquire);
    }
};

struct RequestSource {
    uint64_t tensors[2][kExecTensorDescWords]{};
    uint64_t scalars[2]{};
    uint64_t dependencies[2]{};
    TensorArgType tags[2]{TensorArgType::INPUT, TensorArgType::OUTPUT};
    bool references[2]{};

    uint64_t TensorWord(uint32_t tensor, uint32_t word) const { return tensors[tensor][word]; }
    uint64_t Scalar(uint32_t scalar) const { return scalars[scalar]; }
    uint64_t ExplicitDependency(uint32_t dependency) const { return dependencies[dependency]; }
    TensorArgType TensorTag(uint32_t tensor) const { return tags[tensor]; }
    bool TensorIsReference(uint32_t tensor) const { return references[tensor]; }
};

SimtBuildRequestSpec KernelSpec(uint32_t task_id = 9) {
    return SimtBuildRequestSpec{
        task_id, 0x123456780ULL, 17, 2, 2, 2, ExecEngineClass::Aiv, 0,
    };
}

TEST(FdwicSimtBuildRequestProtocol, AtomicControlAndVariablePayloadHaveStableLayout) {
    EXPECT_EQ(sizeof(SimtBuildRequestHeader), 64U);
    EXPECT_EQ(offsetof(SimtBuildRequestCell, payload), 64U);
    EXPECT_EQ(sizeof(SimtBuildRequestStorage), kSimtRequestMaxPayloadBytes);
    EXPECT_EQ(sizeof(SimtBuildRequestCell), 64U + kSimtRequestMaxPayloadBytes);
    EXPECT_EQ(alignof(SimtBuildRequestCell), 64U);
}

TEST(FdwicSimtBuildRequestProtocol, ReservationElectsOneScalarPublisher) {
    SimtBuildRequestCell cell{};
    SharedExecControl fatal{};
    EXPECT_EQ(ReserveSimtBuildRequest<HostOps>(cell, 11, 7, fatal), SimtRequestReserveResult::Reserved);
    EXPECT_EQ(ReserveSimtBuildRequest<HostOps>(cell, 11, 8, fatal), SimtRequestReserveResult::CellUnavailable);
    const DecodedSimtRequestControl state = DecodeSimtRequestControl(cell.control.state);
    ASSERT_TRUE(state.valid);
    EXPECT_EQ(state.phase, SimtRequestPhase::Reserved);
    EXPECT_EQ(state.task_id, 11U);
    EXPECT_EQ(state.publisher_owner, 7U);
    EXPECT_EQ(state.payload_lines, 0U);
}

TEST(FdwicSimtBuildRequestProtocol, PublisherFlushesPackedPayloadBeforePublishedControl) {
    SimtBuildRequestCell cell{};
    SharedExecControl fatal{};
    RequestSource source{};
    for (uint32_t word = 0; word < kExecTensorDescWords; ++word) {
        source.tensors[0][word] = 0x1000U + word;
        source.tensors[1][word] = 0x2000U + word;
    }
    source.scalars[0] = 0xAA;
    source.scalars[1] = 0xBB;
    source.dependencies[0] = 3;
    source.dependencies[1] = 7;
    const SimtBuildRequestSpec spec = KernelSpec();

    HostOps::Reset();
    ASSERT_EQ(ReserveSimtBuildRequest<HostOps>(cell, spec.task_id, 5, fatal), SimtRequestReserveResult::Reserved);
    EXPECT_EQ(
        PublishReservedSimtBuildRequest<HostOps>(cell, 5, spec, source, fatal), SimtRequestPublishResult::Published
    );
    ASSERT_EQ(HostOps::flush_count, 2U);
    EXPECT_EQ(HostOps::flush_addresses[0], reinterpret_cast<uintptr_t>(&cell.payload));
    EXPECT_EQ(HostOps::flush_bytes[0], 6U * kExecCacheLineBytes);
    EXPECT_EQ(HostOps::flush_addresses[1], reinterpret_cast<uintptr_t>(&cell.control));
    EXPECT_EQ(HostOps::flush_bytes[1], kExecCacheLineBytes);

    const DecodedSimtRequestControl control = DecodeSimtRequestControl(cell.control.state);
    ASSERT_TRUE(control.valid);
    EXPECT_EQ(control.phase, SimtRequestPhase::Published);
    EXPECT_EQ(control.payload_lines, 6U);
    EXPECT_EQ(HostOps::last_bytes, kExecCacheLineBytes);

    SimtBuildRequestHeader header{};
    SimtBuildRequestLayout layout{};
    EXPECT_EQ(
        AcquireSimtBuildRequest<HostOps>(cell, spec.task_id, header, layout, fatal), SimtRequestAcquireResult::Acquired
    );
    EXPECT_EQ(HostOps::invalidate_count, 1U);
    EXPECT_EQ(header.task_id, spec.task_id);
    EXPECT_EQ(header.function_address, spec.function_address);
    EXPECT_EQ(header.function_id, spec.function_id);
    EXPECT_EQ(header.engine_class, ExecEngineClass::Aiv);
    EXPECT_EQ(header.tensor_tags[0], static_cast<uint8_t>(TensorArgType::INPUT));
    EXPECT_EQ(header.tensor_tags[1], static_cast<uint8_t>(TensorArgType::OUTPUT));
    EXPECT_EQ(cell.payload.words[SimtRequestTensorWordOffset(0)], 0x1000U);
    EXPECT_EQ(cell.payload.words[SimtRequestTensorWordOffset(1)], 0x2000U);
    EXPECT_EQ(cell.payload.words[SimtRequestTensorWordOffset(1) + 7U], 0x2007U);
    for (uint32_t word = 8; word < kExecTensorDescWords; ++word) {
        EXPECT_EQ(cell.payload.words[SimtRequestTensorWordOffset(1) + word], 0U);
    }
    EXPECT_EQ(cell.payload.words[layout.scalar_word_offset], 0xAAU);
    EXPECT_EQ(cell.payload.words[layout.scalar_word_offset + 1U], 0xBBU);
    EXPECT_EQ(cell.payload.words[layout.explicit_dep_word_offset], 3U);
    EXPECT_EQ(cell.payload.words[layout.explicit_dep_word_offset + 1U], 7U);
}

TEST(FdwicSimtBuildRequestProtocol, ImmediateRequestUsesTheSamePortablePayload) {
    SimtBuildRequestCell cell{};
    SharedExecControl fatal{};
    RequestSource source{};
    source.tags[0] = TensorArgType::OUTPUT;
    const SimtBuildRequestSpec spec{
        4, 0, kExecInvalidFunctionId, 1, 0, 0, ExecEngineClass::Immediate, 0,
    };
    ASSERT_EQ(ReserveSimtBuildRequest<HostOps>(cell, 4, 3, fatal), SimtRequestReserveResult::Reserved);
    EXPECT_EQ(
        PublishReservedSimtBuildRequest<HostOps>(cell, 3, spec, source, fatal), SimtRequestPublishResult::Published
    );
}

TEST(FdwicSimtBuildRequestProtocol, InvalidShapeOrMismatchedPublisherFailsClosed) {
    SimtBuildRequestCell cell{};
    SharedExecControl fatal{};
    RequestSource source{};
    SimtBuildRequestSpec spec = KernelSpec(6);
    ASSERT_EQ(ReserveSimtBuildRequest<HostOps>(cell, 6, 2, fatal), SimtRequestReserveResult::Reserved);
    EXPECT_EQ(
        PublishReservedSimtBuildRequest<HostOps>(cell, 3, spec, source, fatal),
        SimtRequestPublishResult::PublishConflict
    );

    SimtBuildRequestCell invalid_cell{};
    spec.tensor_count = kExecMaxTensors + 1U;
    ASSERT_EQ(ReserveSimtBuildRequest<HostOps>(invalid_cell, 6, 2, fatal), SimtRequestReserveResult::Reserved);
    EXPECT_EQ(
        PublishReservedSimtBuildRequest<HostOps>(invalid_cell, 2, spec, source, fatal),
        SimtRequestPublishResult::InvalidInput
    );
}

TEST(FdwicSimtBuildRequestProtocol, RealL0TaskArgsAreCopiedByValueWithoutLocalReferences) {
    uint32_t shape[1] = {16};
    Tensor input = make_tensor_external(reinterpret_cast<void *>(0x1000), shape, 1, DataType::FLOAT32);
    TensorCreateInfo output(shape, 1, DataType::FLOAT32);
    PTO2TaskId dependencies[2] = {PTO2TaskId::make(0, 1), PTO2TaskId::make(0, 3)};
    L0TaskArgs args;
    args.add_input(input);
    args.add_output(output);
    args.add_scalar(uint64_t{0x1234});
    args.set_dependencies(dependencies, 2);
    ASSERT_TRUE(ValidateSimtL0TaskArgs(args, 5));

    SimtBuildRequestCell cell{};
    SharedExecControl fatal{};
    const SimtBuildRequestSpec spec{
        5, 0x123456780ULL, 17, 2, 1, 2, ExecEngineClass::Aiv, 0,
    };
    const SimtL0TaskArgsRequestSource source{args};
    ASSERT_EQ(ReserveSimtBuildRequest<HostOps>(cell, 5, 2, fatal), SimtRequestReserveResult::Reserved);
    ASSERT_EQ(
        PublishReservedSimtBuildRequest<HostOps>(cell, 2, spec, source, fatal), SimtRequestPublishResult::Published
    );

    const auto *input_words = reinterpret_cast<const uint64_t *>(&input);
    for (uint32_t word = 0; word < kExecTensorDescWords; ++word) {
        EXPECT_EQ(cell.payload.words[SimtRequestTensorWordOffset(0) + word], input_words[word]);
    }
    const auto *create_info_words = reinterpret_cast<const uint64_t *>(&output);
    for (uint32_t word = 0; word < 8; ++word) {
        EXPECT_EQ(cell.payload.words[SimtRequestTensorWordOffset(1) + word], create_info_words[word]);
    }
    for (uint32_t word = 8; word < kExecTensorDescWords; ++word) {
        EXPECT_EQ(cell.payload.words[SimtRequestTensorWordOffset(1) + word], 0U);
    }
    SimtBuildRequestLayout layout{};
    ASSERT_TRUE(ComputeSimtBuildRequestLayout(2, 1, 2, layout));
    EXPECT_EQ(cell.payload.words[layout.scalar_word_offset], 0x1234U);
    EXPECT_EQ(cell.payload.words[layout.explicit_dep_word_offset], dependencies[0].raw);
    EXPECT_EQ(cell.payload.words[layout.explicit_dep_word_offset + 1U], dependencies[1].raw);
}

TEST(FdwicSimtBuildRequestProtocol, RealL0TaskArgsRejectFutureOrCrossRingDependencies) {
    L0TaskArgs args;
    PTO2TaskId future = PTO2TaskId::make(0, 7);
    args.set_dependencies(&future, 1);
    EXPECT_FALSE(ValidateSimtL0TaskArgs(args, 7));

    L0TaskArgs cross_ring;
    PTO2TaskId dependency = PTO2TaskId::make(1, 2);
    cross_ring.set_dependencies(&dependency, 1);
    EXPECT_FALSE(ValidateSimtL0TaskArgs(cross_ring, 7));
}

TEST(FdwicSimtBuildRequestProtocol, SharedOutputReferenceRemainsCompactUntilBuilderResolution) {
    FdwicOutputRef input_ref{3, 0, 0, 0, 0, 0};
    L0TaskArgs args;
    args.add_input(input_ref);
    ASSERT_TRUE(ValidateSimtL0TaskArgs(args, 5, 0));
    EXPECT_FALSE(ValidateSimtL0TaskArgs(args, 5, 1));

    EXPECT_EQ(SimtL0TaskArgsReferenceMask(args), 0x1U);
    const SimtL0TaskArgsRequestSource source{args};
    SimtBuildRequestCell cell{};
    SharedExecControl fatal{};
    const SimtBuildRequestSpec spec{
        5, 0x123456780ULL, 17, 1, 0, 0, ExecEngineClass::Aiv, 0, 0x1,
    };
    ASSERT_EQ(ReserveSimtBuildRequest<HostOps>(cell, 5, 2, fatal), SimtRequestReserveResult::Reserved);
    ASSERT_EQ(
        PublishReservedSimtBuildRequest<HostOps>(cell, 2, spec, source, fatal), SimtRequestPublishResult::Published
    );

    EXPECT_EQ(cell.payload.words[7], 0x1U);
    EXPECT_EQ(cell.payload.words[SimtRequestTensorWordOffset(0, 0x1)], source.TensorWord(0, 0));
    SimtBuildRequestHeader header{};
    SimtBuildRequestLayout layout{};
    ASSERT_EQ(AcquireSimtBuildRequest<HostOps>(cell, 5, header, layout, fatal), SimtRequestAcquireResult::Acquired);
    EXPECT_EQ(header.tensor_reference_mask, 0x1U);
    EXPECT_EQ(layout.inline_tensor_count, 0U);
    EXPECT_EQ(layout.scalar_word_offset, kSimtRequestHeaderWords + 1U);
    EXPECT_EQ(layout.payload_lines, 2U);

    FdwicOutputRef future_ref{5, 0, 0, 0, 0, 0};
    L0TaskArgs future_args;
    future_args.add_input(future_ref);
    EXPECT_FALSE(ValidateSimtL0TaskArgs(future_args, 5, 0));
}

TEST(FdwicSimtBuildRequestProtocol, ReferenceMaskRejectsOutputAndOutOfRangeBits) {
    RequestSource source{};
    source.tags[0] = TensorArgType::OUTPUT;
    source.references[0] = true;
    SimtBuildRequestCell output_cell{};
    SharedExecControl fatal{};
    SimtBuildRequestSpec output_spec{
        5, 0x123456780ULL, 17, 1, 0, 0, ExecEngineClass::Aiv, 0, 0x1,
    };
    ASSERT_EQ(ReserveSimtBuildRequest<HostOps>(output_cell, 5, 2, fatal), SimtRequestReserveResult::Reserved);
    EXPECT_EQ(
        PublishReservedSimtBuildRequest<HostOps>(output_cell, 2, output_spec, source, fatal),
        SimtRequestPublishResult::InvalidInput
    );

    SimtBuildRequestLayout layout{};
    EXPECT_FALSE(ComputeSimtBuildRequestLayout(1, 0, 0, 0x2, layout));
}

TEST(FdwicSimtBuildRequestProtocol, MixedInlineOutputAndReferenceKeepAllOffsetsStable) {
    uint32_t shape[1] = {8};
    Tensor input = make_tensor_external(reinterpret_cast<void *>(0x8000), shape, 1, DataType::FLOAT32);
    TensorCreateInfo output(shape, 1, DataType::FLOAT32);
    FdwicOutputRef prior{2, 1, 0, 0, 0, 0};
    PTO2TaskId dependency = PTO2TaskId::make(0, 1);
    L0TaskArgs args;
    args.add_input(input);
    args.add_output(output);
    args.add_inout(prior);
    args.add_scalar(uint64_t{0x5678});
    args.set_dependencies(&dependency, 1);
    ASSERT_TRUE(ValidateSimtL0TaskArgs(args, 5, 1));
    ASSERT_EQ(SimtL0TaskArgsReferenceMask(args), 0x4U);

    SimtBuildRequestCell cell{};
    SharedExecControl fatal{};
    const SimtBuildRequestSpec spec{
        5, 0x123456780ULL, 17, 3, 1, 1, ExecEngineClass::Aiv, 0, 0x4,
    };
    const SimtL0TaskArgsRequestSource source{args};
    ASSERT_EQ(ReserveSimtBuildRequest<HostOps>(cell, 5, 2, fatal), SimtRequestReserveResult::Reserved);
    ASSERT_EQ(
        PublishReservedSimtBuildRequest<HostOps>(cell, 2, spec, source, fatal), SimtRequestPublishResult::Published
    );

    SimtBuildRequestHeader header{};
    SimtBuildRequestLayout layout{};
    ASSERT_EQ(AcquireSimtBuildRequest<HostOps>(cell, 5, header, layout, fatal), SimtRequestAcquireResult::Acquired);
    EXPECT_EQ(header.tensor_reference_mask, 0x4U);
    EXPECT_EQ(layout.inline_tensor_count, 2U);
    EXPECT_EQ(layout.scalar_word_offset, kSimtRequestHeaderWords + 2U * kExecTensorDescWords + 1U);
    EXPECT_EQ(layout.explicit_dep_word_offset, layout.scalar_word_offset + 1U);
    EXPECT_EQ(cell.payload.words[SimtRequestTensorWordOffset(2, 0x4)], source.TensorWord(2, 0));
    EXPECT_EQ(cell.payload.words[layout.scalar_word_offset], 0x5678U);
    EXPECT_EQ(cell.payload.words[layout.explicit_dep_word_offset], dependency.raw);
}

TEST(FdwicSimtBuildRequestProtocol, ReferenceMaskMustMatchTheSourceReferenceKind) {
    RequestSource source{};
    source.references[0] = true;
    source.tensors[0][0] = 1;
    SimtBuildRequestCell cell{};
    SharedExecControl fatal{};
    const SimtBuildRequestSpec missing_mask{
        5, 0x123456780ULL, 17, 1, 0, 0, ExecEngineClass::Aiv, 0, 0,
    };
    ASSERT_EQ(ReserveSimtBuildRequest<HostOps>(cell, 5, 2, fatal), SimtRequestReserveResult::Reserved);
    EXPECT_EQ(
        PublishReservedSimtBuildRequest<HostOps>(cell, 2, missing_mask, source, fatal),
        SimtRequestPublishResult::InvalidInput
    );
}

}  // namespace
