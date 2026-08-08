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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_S3_DUAL_TASK_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_S3_DUAL_TASK_H

#include <stddef.h>
#include <stdint.h>

#include "gm_probe_support.h"

namespace pa_scheduler::simt_cross_core::s3 {

#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_S3_INLINE __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_S3_INLINE constexpr
#endif

using gm::ExpectedGuardWord;
using gm::FoldDescriptorChecksum;
using gm::kOutputSentinel;
using gm::PackTaskShape;
using gm::PayloadElementCount;
using gm::PayloadTaskId;
using gm::ProbeCellControl;
using gm::ProbeControl;
using gm::ProbeFatalControl;
using gm::ProbeGuard;
using gm::ProbeRoleResult;
using gm::ProbeSimtReport;

constexpr uint64_t kProbeMagic = 0x53494D5453334135ULL;
constexpr uint64_t kProbeVersion = 1U;
constexpr uint64_t kPayloadVersion = 1U;
constexpr uint64_t kVectorPayloadMagic = 0x5333564543544F52ULL;
constexpr uint64_t kCubePayloadMagic = 0x533343554245544BULL;
constexpr uint64_t kResultMagic = 0x5333524553554C54ULL;
constexpr uint32_t kThreadCount = 64U;
constexpr uint32_t kTaskCount = 2U;
constexpr uint32_t kPayloadWords = 8U;
constexpr uint32_t kTileRows = 128U;
constexpr uint32_t kTileColumns = 128U;
constexpr uint32_t kElementCount = kTileRows * kTileColumns;
constexpr uint32_t kTileBytes = kElementCount * sizeof(float);
constexpr uint32_t kVectorTaskIndex = 0U;
constexpr uint32_t kCubeTaskIndex = 1U;
constexpr uint32_t kVectorTaskId = 301U;
constexpr uint32_t kCubeTaskId = 302U;
constexpr uint32_t kBuilderOwner = 32U;
constexpr uint32_t kAicExecutorOwner = 0U;
constexpr uint32_t kAivExecutorOwner = 33U;
constexpr uint32_t kRoleCount = 3U;
constexpr uint64_t kRequiredVisibilityMode = static_cast<uint64_t>(gm::VisibilityMode::ReaderDcci);

enum class ProbeRole : uint64_t {
    AicCubeExecutor = 0U,
    Aiv0Builder = 1U,
    Aiv1VectorExecutor = 2U,
};

enum PayloadWord : uint32_t {
    kPayloadMagicWord = 0U,
    kPayloadVersionWord = 1U,
    kPayloadNonceWord = 2U,
    kPayloadInputAWord = 3U,
    kPayloadInputBWord = 4U,
    kPayloadOutputWord = 5U,
    kPayloadShapeWord = 6U,
    kPayloadChecksumWord = 7U,
};

constexpr uint64_t kEmptyState = EncodeExecState(ExecPhase::Empty, 0U, 0U, ExecEngineClass::None, 0U, 0U);
constexpr uint64_t kVectorBuildingState =
    EncodeExecState(ExecPhase::Building, kBuilderOwner, kUnboundOwner, ExecEngineClass::None, 0U, kVectorTaskId);
constexpr uint64_t kVectorBuiltState =
    EncodeExecState(ExecPhase::Built, kBuilderOwner, kUnboundOwner, ExecEngineClass::Aiv, 1U, kVectorTaskId);
constexpr uint64_t kVectorClaimedState =
    EncodeExecState(ExecPhase::Claimed, kBuilderOwner, kAivExecutorOwner, ExecEngineClass::Aiv, 1U, kVectorTaskId);
constexpr uint64_t kVectorDoneState =
    EncodeExecState(ExecPhase::Done, kBuilderOwner, kAivExecutorOwner, ExecEngineClass::Aiv, 1U, kVectorTaskId);
constexpr uint64_t kCubeBuildingState =
    EncodeExecState(ExecPhase::Building, kBuilderOwner, kUnboundOwner, ExecEngineClass::None, 0U, kCubeTaskId);
constexpr uint64_t kCubeBuiltState =
    EncodeExecState(ExecPhase::Built, kBuilderOwner, kUnboundOwner, ExecEngineClass::Aic, 1U, kCubeTaskId);
constexpr uint64_t kCubeClaimedState =
    EncodeExecState(ExecPhase::Claimed, kBuilderOwner, kAicExecutorOwner, ExecEngineClass::Aic, 1U, kCubeTaskId);
constexpr uint64_t kCubeDoneState =
    EncodeExecState(ExecPhase::Done, kBuilderOwner, kAicExecutorOwner, ExecEngineClass::Aic, 1U, kCubeTaskId);

SIMT_CROSS_CORE_S3_INLINE uint64_t TaskPayloadMagic(uint32_t task_index) {
    return task_index == kVectorTaskIndex ? kVectorPayloadMagic : kCubePayloadMagic;
}

SIMT_CROSS_CORE_S3_INLINE uint32_t TaskId(uint32_t task_index) {
    return task_index == kVectorTaskIndex ? kVectorTaskId : kCubeTaskId;
}

SIMT_CROSS_CORE_S3_INLINE uint64_t
ComputePayloadChecksum(uint32_t task_index, uint64_t nonce, uint64_t input_a, uint64_t input_b, uint64_t output) {
    uint64_t checksum = task_index == kVectorTaskIndex ? 0xA4093822299F31D0ULL : 0x082EFA98EC4E6C89ULL;
    checksum = FoldDescriptorChecksum(checksum, TaskPayloadMagic(task_index), kPayloadMagicWord);
    checksum = FoldDescriptorChecksum(checksum, kPayloadVersion, kPayloadVersionWord);
    checksum = FoldDescriptorChecksum(checksum, nonce, kPayloadNonceWord);
    checksum = FoldDescriptorChecksum(checksum, input_a, kPayloadInputAWord);
    checksum = FoldDescriptorChecksum(checksum, input_b, kPayloadInputBWord);
    checksum = FoldDescriptorChecksum(checksum, output, kPayloadOutputWord);
    checksum = FoldDescriptorChecksum(checksum, PackTaskShape(TaskId(task_index), kElementCount), kPayloadShapeWord);
    return checksum;
}

constexpr float ExpectedVectorInputA(uint64_t nonce, uint32_t index) {
    return static_cast<float>((index + static_cast<uint32_t>(nonce & 63U)) % 251U);
}

constexpr float ExpectedVectorInputB(uint64_t nonce, uint32_t index) {
    return static_cast<float>((index * 5U + static_cast<uint32_t>((nonce >> 8U) & 63U)) % 127U);
}

constexpr float ExpectedVectorOutput(uint64_t nonce, uint32_t index) {
    return ExpectedVectorInputA(nonce, index) + ExpectedVectorInputB(nonce, index);
}

constexpr float ExpectedCubeInputA(uint32_t row, uint32_t column) {
    return row == column ? static_cast<float>(row + 1U) : 0.0F;
}

constexpr float ExpectedCubeInputB(uint64_t nonce, uint32_t row, uint32_t column) {
    const uint32_t nonce_term = static_cast<uint32_t>((nonce >> 16U) & 0xFFU);
    const uint32_t value = (113U * row + 19U * column + 11U * row * column + nonce_term) % 241U;
    return static_cast<float>(value + 1U);
}

constexpr float ExpectedCubeOutput(uint64_t nonce, uint32_t row, uint32_t column) {
    return static_cast<float>(row + 1U) * ExpectedCubeInputB(nonce, row, column);
}

struct alignas(kCacheLineBytes) ProbePayload {
    uint64_t words[kPayloadWords];
};

struct alignas(kCacheLineBytes) ProbeTaskSlot {
    ProbeCellControl cell;
    ProbePayload payload;
};

struct alignas(kCacheLineBytes) ProbeDrainControl {
    uint64_t builder_finished;
    uint64_t done_count;
    uint64_t padding[6];
};

struct alignas(kCacheLineBytes) ProbeFloatTile {
    float values[kElementCount];
};

struct alignas(kCacheLineBytes) ProbeState {
    ProbeControl control;
    ProbeTaskSlot tasks[kTaskCount];
    ProbeFatalControl fatal;
    ProbeDrainControl drain;
    ProbeSimtReport simt_reports[kTaskCount];
    ProbeRoleResult roles[kRoleCount];
    ProbeGuard guard_before_vector_input_a;
    ProbeFloatTile vector_input_a;
    ProbeGuard guard_before_vector_input_b;
    ProbeFloatTile vector_input_b;
    ProbeGuard guard_before_vector_output;
    ProbeFloatTile vector_output;
    ProbeGuard guard_before_cube_input_a;
    ProbeFloatTile cube_input_a;
    ProbeGuard guard_before_cube_input_b;
    ProbeFloatTile cube_input_b;
    ProbeGuard guard_before_cube_output;
    ProbeFloatTile cube_output;
    ProbeGuard guard_after_cube_output;
};

static_assert(kTileBytes == 64U * 1024U, "S3 tile size changed");
static_assert(sizeof(ProbeCellControl) == kCacheLineBytes, "S3 cell ABI changed");
static_assert(sizeof(ProbePayload) == kCacheLineBytes, "S3 payload ABI changed");
static_assert(sizeof(ProbeTaskSlot) == 2U * kCacheLineBytes, "S3 task-slot ABI changed");
static_assert(sizeof(ProbeDrainControl) == kCacheLineBytes, "S3 drain ABI changed");
static_assert(sizeof(ProbeFloatTile) == kTileBytes, "S3 float-tile ABI changed");
static_assert(offsetof(ProbeTaskSlot, payload) == kCacheLineBytes, "S3 payload must not share the control line");
static_assert(
    offsetof(ProbeState, vector_input_a) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, vector_input_b) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, vector_output) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, cube_input_a) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, cube_input_b) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, cube_output) % kCacheLineBytes == 0U && sizeof(ProbeState) % kCacheLineBytes == 0U,
    "S3 GM regions must remain cache-line aligned"
);

#undef SIMT_CROSS_CORE_S3_INLINE

}  // namespace pa_scheduler::simt_cross_core::s3

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_S3_DUAL_TASK_H
