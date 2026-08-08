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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_S2_CUBE_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_S2_CUBE_H

#include <stddef.h>
#include <stdint.h>

#include "gm_probe_support.h"

namespace pa_scheduler::simt_cross_core::s2 {

#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_S2_INLINE __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_S2_INLINE constexpr
#endif

using gm::ExpectedGuardWord;
using gm::FoldDescriptorChecksum;
using gm::kOutputSentinel;
using gm::kVisibilityModeCount;
using gm::PackTaskShape;
using gm::PayloadElementCount;
using gm::PayloadTaskId;
using gm::ProbeCellControl;
using gm::ProbeControl;
using gm::ProbeFatalControl;
using gm::ProbeGuard;
using gm::ProbeRoleResult;
using gm::ProbeSimtReport;
using gm::ReaderUsesDcci;
using gm::VisibilityMode;
using gm::WriterUsesDcci;

constexpr uint64_t kProbeMagic = 0x53494D5453324135ULL;
constexpr uint64_t kProbeVersion = 1U;
constexpr uint64_t kPayloadMagic = 0x533243554245544BULL;
constexpr uint64_t kPayloadVersion = 1U;
constexpr uint64_t kResultMagic = 0x5332524553554C54ULL;
constexpr uint32_t kThreadCount = 64U;
constexpr uint32_t kPayloadWords = 8U;
constexpr uint32_t kTileRows = 128U;
constexpr uint32_t kTileColumns = 128U;
constexpr uint32_t kElementCount = kTileRows * kTileColumns;
constexpr uint32_t kTileBytes = kElementCount * sizeof(float);
constexpr uint32_t kTaskId = 202U;
constexpr uint32_t kBuilderOwner = 32U;
constexpr uint32_t kExecutorOwner = 0U;
constexpr uint32_t kObserverOwner = 33U;
constexpr uint32_t kRoleCount = 3U;

enum class ProbeRole : uint64_t {
    AicExecutor = 0U,
    Aiv0Builder = 1U,
    Aiv1Observer = 2U,
};

constexpr uint64_t kEmptyState = EncodeExecState(ExecPhase::Empty, 0U, 0U, ExecEngineClass::None, 0U, 0U);
constexpr uint64_t kBuildingState =
    EncodeExecState(ExecPhase::Building, kBuilderOwner, kUnboundOwner, ExecEngineClass::None, 0U, kTaskId);
constexpr uint64_t kBuiltState =
    EncodeExecState(ExecPhase::Built, kBuilderOwner, kUnboundOwner, ExecEngineClass::Aic, 1U, kTaskId);
constexpr uint64_t kClaimedState =
    EncodeExecState(ExecPhase::Claimed, kBuilderOwner, kExecutorOwner, ExecEngineClass::Aic, 1U, kTaskId);
constexpr uint64_t kDoneState =
    EncodeExecState(ExecPhase::Done, kBuilderOwner, kExecutorOwner, ExecEngineClass::Aic, 1U, kTaskId);

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

SIMT_CROSS_CORE_S2_INLINE uint64_t
ComputePayloadChecksum(uint64_t nonce, uint64_t input_a, uint64_t input_b, uint64_t output) {
    uint64_t checksum = 0x13198A2E03707344ULL;
    checksum = FoldDescriptorChecksum(checksum, kPayloadMagic, kPayloadMagicWord);
    checksum = FoldDescriptorChecksum(checksum, kPayloadVersion, kPayloadVersionWord);
    checksum = FoldDescriptorChecksum(checksum, nonce, kPayloadNonceWord);
    checksum = FoldDescriptorChecksum(checksum, input_a, kPayloadInputAWord);
    checksum = FoldDescriptorChecksum(checksum, input_b, kPayloadInputBWord);
    checksum = FoldDescriptorChecksum(checksum, output, kPayloadOutputWord);
    checksum = FoldDescriptorChecksum(checksum, PackTaskShape(kTaskId, kElementCount), kPayloadShapeWord);
    return checksum;
}

constexpr float ExpectedInputA(uint32_t row, uint32_t column) {
    return row == column ? static_cast<float>(row + 1U) : 0.0F;
}

constexpr float ExpectedInputB(uint64_t nonce, uint32_t row, uint32_t column) {
    const uint32_t nonce_term = static_cast<uint32_t>((nonce >> 8U) & 0xFFU);
    const uint32_t value = (131U * row + 17U * column + 7U * row * column + nonce_term) % 251U;
    return static_cast<float>(value + 1U);
}

constexpr float ExpectedOutput(uint64_t nonce, uint32_t row, uint32_t column) {
    return static_cast<float>(row + 1U) * ExpectedInputB(nonce, row, column);
}

struct alignas(kCacheLineBytes) ProbePayload {
    uint64_t words[kPayloadWords];
};

struct alignas(kCacheLineBytes) ProbeCubeTile {
    float values[kElementCount];
};

struct alignas(kCacheLineBytes) ProbeState {
    ProbeControl control;
    ProbeCellControl cell;
    ProbePayload payload;
    ProbeFatalControl fatal;
    ProbeSimtReport simt_report;
    ProbeRoleResult roles[kRoleCount];
    ProbeGuard guard_before_input_a;
    ProbeCubeTile input_a;
    ProbeGuard guard_before_input_b;
    ProbeCubeTile input_b;
    ProbeGuard guard_before_output;
    ProbeCubeTile output;
    ProbeGuard guard_after_output;
};

static_assert(kTileBytes == 64U * 1024U, "S2 cube tile size changed");
static_assert(sizeof(ProbePayload) == kCacheLineBytes, "S2 payload ABI changed");
static_assert(sizeof(ProbeCubeTile) == kTileBytes, "S2 tile ABI changed");
static_assert(offsetof(ProbeState, payload) == 2U * kCacheLineBytes, "S2 payload offset changed");
static_assert(
    offsetof(ProbeState, input_a) % kCacheLineBytes == 0U && offsetof(ProbeState, input_b) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, output) % kCacheLineBytes == 0U && sizeof(ProbeState) % kCacheLineBytes == 0U,
    "S2 GM regions must remain cache-line aligned"
);

#undef SIMT_CROSS_CORE_S2_INLINE

}  // namespace pa_scheduler::simt_cross_core::s2

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_S2_CUBE_H
