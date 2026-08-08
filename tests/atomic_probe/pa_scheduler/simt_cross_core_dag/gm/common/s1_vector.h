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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_S1_VECTOR_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_S1_VECTOR_H

#include <stddef.h>
#include <stdint.h>

#include "gm_probe_support.h"

namespace pa_scheduler::simt_cross_core::s1 {

#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_S1_INLINE __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_S1_INLINE constexpr
#endif

constexpr uint64_t kProbeMagic = 0x53494D5453314135ULL;
constexpr uint64_t kProbeVersion = 1U;
constexpr uint64_t kPayloadMagic = 0x5331564543544F52ULL;
constexpr uint64_t kPayloadVersion = 1U;
constexpr uint64_t kResultMagic = 0x5331524553554C54ULL;
constexpr uint32_t kThreadCount = 64U;
constexpr uint32_t kPayloadWords = 8U;
constexpr uint32_t kTileRows = 128U;
constexpr uint32_t kTileColumns = 128U;
constexpr uint32_t kElementCount = kTileRows * kTileColumns;
constexpr uint32_t kTileBytes = kElementCount * sizeof(float);
constexpr uint32_t kTaskId = 101U;
constexpr uint32_t kBuilderOwner = 32U;
constexpr uint32_t kExecutorOwner = 33U;
constexpr uint32_t kRoleCount = 3U;

enum class ProbeRole : uint64_t {
    AicObserver = 0U,
    Aiv0Builder = 1U,
    Aiv1Executor = 2U,
};

using gm::ExpectedGuardWord;
using gm::FoldDescriptorChecksum;
using gm::IdentityField;
using gm::kCountBuildAttemptShift;
using gm::kCountBuildWinShift;
using gm::kCountClaimAttemptShift;
using gm::kCountClaimWinShift;
using gm::kIdentityBlockCountShift;
using gm::kIdentityBlockIndexShift;
using gm::kIdentityByteMask;
using gm::kIdentityCoreIdShift;
using gm::kIdentityCoreMask;
using gm::kIdentityRoleShift;
using gm::kIdentitySubblockCountShift;
using gm::kIdentitySubblockShift;
using gm::kOutputSentinel;
using gm::kResultConfigValid;
using gm::kResultPayloadValid;
using gm::kResultWaitDone;
using gm::kVisibilityModeCount;
using gm::PackIdentity;
using gm::PackRoleCounts;
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
using gm::RoleCountField;
using gm::VisibilityMode;
using gm::WriterUsesDcci;

constexpr uint64_t kResultVectorExecuted = gm::kResultTaskExecuted;

constexpr uint64_t kEmptyState = EncodeExecState(ExecPhase::Empty, 0U, 0U, ExecEngineClass::None, 0U, 0U);
constexpr uint64_t kBuildingState =
    EncodeExecState(ExecPhase::Building, kBuilderOwner, kUnboundOwner, ExecEngineClass::None, 0U, kTaskId);
constexpr uint64_t kBuiltState =
    EncodeExecState(ExecPhase::Built, kBuilderOwner, kUnboundOwner, ExecEngineClass::Aiv, 1U, kTaskId);
constexpr uint64_t kClaimedState =
    EncodeExecState(ExecPhase::Claimed, kBuilderOwner, kExecutorOwner, ExecEngineClass::Aiv, 1U, kTaskId);
constexpr uint64_t kDoneState =
    EncodeExecState(ExecPhase::Done, kBuilderOwner, kExecutorOwner, ExecEngineClass::Aiv, 1U, kTaskId);

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

SIMT_CROSS_CORE_S1_INLINE uint64_t
ComputePayloadChecksum(uint64_t nonce, uint64_t input_a, uint64_t input_b, uint64_t output) {
    uint64_t checksum = 0x243F6A8885A308D3ULL;
    checksum = FoldDescriptorChecksum(checksum, kPayloadMagic, kPayloadMagicWord);
    checksum = FoldDescriptorChecksum(checksum, kPayloadVersion, kPayloadVersionWord);
    checksum = FoldDescriptorChecksum(checksum, nonce, kPayloadNonceWord);
    checksum = FoldDescriptorChecksum(checksum, input_a, kPayloadInputAWord);
    checksum = FoldDescriptorChecksum(checksum, input_b, kPayloadInputBWord);
    checksum = FoldDescriptorChecksum(checksum, output, kPayloadOutputWord);
    checksum = FoldDescriptorChecksum(checksum, PackTaskShape(kTaskId, kElementCount), kPayloadShapeWord);
    return checksum;
}

constexpr float ExpectedInputA(uint64_t nonce, uint32_t index) {
    return static_cast<float>((index + static_cast<uint32_t>(nonce & 31U)) % 251U);
}

constexpr float ExpectedInputB(uint64_t nonce, uint32_t index) {
    return static_cast<float>((index * 3U + static_cast<uint32_t>((nonce >> 8U) & 31U)) % 127U);
}

constexpr float ExpectedOutput(uint64_t nonce, uint32_t index) {
    return ExpectedInputA(nonce, index) + ExpectedInputB(nonce, index);
}

struct alignas(kCacheLineBytes) ProbePayload {
    uint64_t words[kPayloadWords];
};

struct alignas(kCacheLineBytes) ProbeVectorTile {
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
    ProbeVectorTile input_a;
    ProbeGuard guard_before_input_b;
    ProbeVectorTile input_b;
    ProbeGuard guard_before_output;
    ProbeVectorTile output;
    ProbeGuard guard_after_output;
};

static_assert(kTileBytes == 64U * 1024U, "S1 vector tile size changed");
static_assert(sizeof(ProbeControl) == kCacheLineBytes, "S1 control ABI changed");
static_assert(sizeof(ProbeCellControl) == kCacheLineBytes, "S1 cell ABI changed");
static_assert(sizeof(ProbePayload) == kCacheLineBytes, "S1 payload ABI changed");
static_assert(sizeof(ProbeFatalControl) == kCacheLineBytes, "S1 fatal ABI changed");
static_assert(sizeof(ProbeSimtReport) == kCacheLineBytes, "S1 SIMT report ABI changed");
static_assert(sizeof(ProbeRoleResult) == 2U * kCacheLineBytes, "S1 role-result ABI changed");
static_assert(sizeof(ProbeGuard) == kCacheLineBytes, "S1 guard ABI changed");
static_assert(sizeof(ProbeVectorTile) == kTileBytes, "S1 tile ABI changed");
static_assert(offsetof(ProbeState, payload) == 2U * kCacheLineBytes, "S1 payload offset changed");
static_assert(
    offsetof(ProbeState, input_a) % kCacheLineBytes == 0U && offsetof(ProbeState, input_b) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, output) % kCacheLineBytes == 0U && sizeof(ProbeState) % kCacheLineBytes == 0U,
    "S1 GM regions must remain cache-line aligned"
);

#undef SIMT_CROSS_CORE_S1_INLINE

}  // namespace pa_scheduler::simt_cross_core::s1

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_S1_VECTOR_H
