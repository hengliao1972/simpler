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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_S4_MULTI_TASK_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_S4_MULTI_TASK_H

#include <stddef.h>
#include <stdint.h>

#include "gm_probe_support.h"

namespace pa_scheduler::simt_cross_core::s4 {

#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_S4_INLINE __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_S4_INLINE constexpr
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

constexpr uint64_t kProbeMagic = 0x53494D5453344135ULL;
constexpr uint64_t kProbeVersion = 1U;
constexpr uint64_t kPayloadVersion = 1U;
constexpr uint64_t kVectorPayloadMagic = 0x5334564543544F52ULL;
constexpr uint64_t kCubePayloadMagic = 0x533443554245544BULL;
constexpr uint64_t kResultMagic = 0x5334524553554C54ULL;
constexpr uint32_t kBuilderThreadCount = 4U;
constexpr uint32_t kTaskCount = 16U;
constexpr uint32_t kVectorTaskCount = kTaskCount / 2U;
constexpr uint32_t kCubeTaskCount = kTaskCount / 2U;
constexpr uint32_t kPayloadWords = 8U;
constexpr uint32_t kTileRows = 16U;
constexpr uint32_t kTileColumns = 16U;
constexpr uint32_t kElementCount = kTileRows * kTileColumns;
constexpr uint32_t kTileBytes = kElementCount * sizeof(float);
constexpr uint32_t kTaskIdBase = 400U;
constexpr uint32_t kBuilderOwner = 32U;
constexpr uint32_t kAicExecutorOwner = 0U;
constexpr uint32_t kAivExecutorOwner = 33U;
constexpr uint32_t kRoleCount = 3U;
constexpr uint64_t kRequiredVisibilityMode = static_cast<uint64_t>(gm::VisibilityMode::ReaderDcci);

SIMT_CROSS_CORE_S4_INLINE uint64_t EncodeS4State(
    ExecPhase phase, uint32_t build_owner, uint32_t execute_owner, ExecEngineClass engine_class, uint32_t payload_lines,
    uint32_t task_id
) {
    return (static_cast<uint64_t>(phase) << kStatePhaseShift) |
           (static_cast<uint64_t>(build_owner) << kStateBuildOwnerShift) |
           (static_cast<uint64_t>(execute_owner) << kStateExecuteOwnerShift) |
           (static_cast<uint64_t>(engine_class) << kStateEngineShift) |
           (static_cast<uint64_t>(payload_lines) << kStatePayloadLinesShift) |
           (static_cast<uint64_t>(task_id) << kStateTaskIdShift);
}

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

constexpr uint32_t kExecutorExecutedShift = 0U;
constexpr uint32_t kExecutorMaxBusyShift = 16U;
constexpr uint32_t kExecutorBusyBlockedShift = 32U;
constexpr uint64_t kExecutorStatMask = 0xFFFFULL;

SIMT_CROSS_CORE_S4_INLINE bool TaskIsVector(uint32_t task_index) { return (task_index & 1U) == 0U; }

SIMT_CROSS_CORE_S4_INLINE uint32_t TaskOrdinal(uint32_t task_index) { return task_index / 2U; }

SIMT_CROSS_CORE_S4_INLINE uint32_t TaskId(uint32_t task_index) { return kTaskIdBase + task_index; }

SIMT_CROSS_CORE_S4_INLINE ExecEngineClass TaskEngine(uint32_t task_index) {
    return TaskIsVector(task_index) ? ExecEngineClass::Aiv : ExecEngineClass::Aic;
}

SIMT_CROSS_CORE_S4_INLINE uint32_t TaskExecutorOwner(uint32_t task_index) {
    return TaskIsVector(task_index) ? kAivExecutorOwner : kAicExecutorOwner;
}

SIMT_CROSS_CORE_S4_INLINE uint64_t BuildingState(uint32_t task_index) {
    return EncodeS4State(
        ExecPhase::Building, kBuilderOwner, kUnboundOwner, ExecEngineClass::None, 0U, TaskId(task_index)
    );
}

SIMT_CROSS_CORE_S4_INLINE uint64_t BuiltState(uint32_t task_index) {
    return EncodeS4State(
        ExecPhase::Built, kBuilderOwner, kUnboundOwner, TaskEngine(task_index), 1U, TaskId(task_index)
    );
}

SIMT_CROSS_CORE_S4_INLINE uint64_t ClaimedState(uint32_t task_index) {
    return EncodeS4State(
        ExecPhase::Claimed, kBuilderOwner, TaskExecutorOwner(task_index), TaskEngine(task_index), 1U, TaskId(task_index)
    );
}

SIMT_CROSS_CORE_S4_INLINE uint64_t DoneState(uint32_t task_index) {
    return EncodeS4State(
        ExecPhase::Done, kBuilderOwner, TaskExecutorOwner(task_index), TaskEngine(task_index), 1U, TaskId(task_index)
    );
}

SIMT_CROSS_CORE_S4_INLINE uint64_t TaskPayloadMagic(uint32_t task_index) {
    return TaskIsVector(task_index) ? kVectorPayloadMagic : kCubePayloadMagic;
}

SIMT_CROSS_CORE_S4_INLINE uint64_t
ComputePayloadChecksum(uint32_t task_index, uint64_t nonce, uint64_t input_a, uint64_t input_b, uint64_t output) {
    uint64_t checksum = TaskIsVector(task_index) ? 0xA4093822299F31D0ULL : 0x082EFA98EC4E6C89ULL;
    checksum = FoldDescriptorChecksum(checksum, TaskPayloadMagic(task_index), kPayloadMagicWord);
    checksum = FoldDescriptorChecksum(checksum, kPayloadVersion, kPayloadVersionWord);
    checksum = FoldDescriptorChecksum(checksum, nonce, kPayloadNonceWord);
    checksum = FoldDescriptorChecksum(checksum, input_a, kPayloadInputAWord);
    checksum = FoldDescriptorChecksum(checksum, input_b, kPayloadInputBWord);
    checksum = FoldDescriptorChecksum(checksum, output, kPayloadOutputWord);
    checksum = FoldDescriptorChecksum(checksum, PackTaskShape(TaskId(task_index), kElementCount), kPayloadShapeWord);
    return checksum;
}

SIMT_CROSS_CORE_S4_INLINE uint64_t PackExecutorStats(uint32_t executed, uint32_t max_busy, uint32_t busy_blocked) {
    return (static_cast<uint64_t>(executed) << kExecutorExecutedShift) |
           (static_cast<uint64_t>(max_busy) << kExecutorMaxBusyShift) |
           (static_cast<uint64_t>(busy_blocked) << kExecutorBusyBlockedShift);
}

constexpr uint32_t ExecutorStatField(uint64_t stats, uint32_t shift) {
    return static_cast<uint32_t>((stats >> shift) & kExecutorStatMask);
}

constexpr float ExpectedVectorInputA(uint64_t nonce, uint32_t task_ordinal, uint32_t index) {
    return static_cast<float>((index + 7U * task_ordinal + static_cast<uint32_t>(nonce & 63U)) % 251U);
}

constexpr float ExpectedVectorInputB(uint64_t nonce, uint32_t task_ordinal, uint32_t index) {
    return static_cast<float>((5U * index + 11U * task_ordinal + static_cast<uint32_t>((nonce >> 8U) & 63U)) % 127U);
}

constexpr float ExpectedVectorOutput(uint64_t nonce, uint32_t task_ordinal, uint32_t index) {
    return ExpectedVectorInputA(nonce, task_ordinal, index) + ExpectedVectorInputB(nonce, task_ordinal, index);
}

constexpr float ExpectedCubeInputA(uint32_t task_ordinal, uint32_t row, uint32_t column) {
    return row == column ? static_cast<float>(row + task_ordinal + 1U) : 0.0F;
}

constexpr float ExpectedCubeInputB(uint64_t nonce, uint32_t task_ordinal, uint32_t row, uint32_t column) {
    const uint32_t nonce_term = static_cast<uint32_t>((nonce >> 16U) & 0xFFU);
    const uint32_t value = (113U * row + 19U * column + 11U * row * column + 17U * task_ordinal + nonce_term) % 241U;
    return static_cast<float>(value + 1U);
}

constexpr float ExpectedCubeOutput(uint64_t nonce, uint32_t task_ordinal, uint32_t row, uint32_t column) {
    return static_cast<float>(row + task_ordinal + 1U) * ExpectedCubeInputB(nonce, task_ordinal, row, column);
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
    uint64_t vector_done;
    uint64_t cube_done;
    uint64_t done_count;
    uint64_t padding[4];
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
    ProbeFloatTile vector_input_a[kVectorTaskCount];
    ProbeGuard guard_before_vector_input_b;
    ProbeFloatTile vector_input_b[kVectorTaskCount];
    ProbeGuard guard_before_vector_output;
    ProbeFloatTile vector_output[kVectorTaskCount];
    ProbeGuard guard_before_cube_input_a;
    ProbeFloatTile cube_input_a[kCubeTaskCount];
    ProbeGuard guard_before_cube_input_b;
    ProbeFloatTile cube_input_b[kCubeTaskCount];
    ProbeGuard guard_before_cube_output;
    ProbeFloatTile cube_output[kCubeTaskCount];
    ProbeGuard guard_after_cube_output;
};

static_assert(kTaskCount % kBuilderThreadCount == 0U, "S4 thread-stride distribution must be exact");
static_assert(kTileBytes == 1024U, "S4 tile size changed");
static_assert(sizeof(ProbePayload) == kCacheLineBytes, "S4 payload ABI changed");
static_assert(sizeof(ProbeTaskSlot) == 2U * kCacheLineBytes, "S4 task-slot ABI changed");
static_assert(sizeof(ProbeDrainControl) == kCacheLineBytes, "S4 drain ABI changed");
static_assert(sizeof(ProbeFloatTile) == kTileBytes, "S4 float-tile ABI changed");
static_assert(offsetof(ProbeTaskSlot, payload) == kCacheLineBytes, "S4 payload must not share the control line");
static_assert(
    offsetof(ProbeState, vector_input_a) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, vector_input_b) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, vector_output) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, cube_input_a) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, cube_input_b) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, cube_output) % kCacheLineBytes == 0U && sizeof(ProbeState) % kCacheLineBytes == 0U,
    "S4 GM regions must remain cache-line aligned"
);

#undef SIMT_CROSS_CORE_S4_INLINE

}  // namespace pa_scheduler::simt_cross_core::s4

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_S4_MULTI_TASK_H
