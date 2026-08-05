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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_S0_PROBE_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_S0_PROBE_H

#include <stddef.h>
#include <stdint.h>

#include "../../common/shared_protocol.h"

namespace pa_scheduler::simt_cross_core::s0 {

// 这只是 host/device 的函数地址空间标注，不是行为开关。CCEC 把未标注
// 函数视为 host 函数，即使它最终会内联，因此共享值生成器必须显式标注。
#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_S0_INLINE __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_S0_INLINE constexpr
#endif

constexpr uint64_t kProbeMagic = 0x53494D5453304135ULL;
constexpr uint64_t kProbeVersion = 1U;
constexpr uint64_t kResultMagic = 0x5330524553554C54ULL;
constexpr uint32_t kThreadCount = 64U;
constexpr uint32_t kPayloadWords = 8U;
constexpr uint32_t kTaskId = 7U;
constexpr uint32_t kBuilderOwner = 32U;
constexpr uint32_t kExecutorOwner = 33U;

enum class VisibilityMode : uint64_t {
    NoDcci = 0U,
    WriterDcci = 1U,
    ReaderDcci = 2U,
    WriterAndReaderDcci = 3U,
};

constexpr uint64_t kVisibilityModeCount = 4U;

SIMT_CROSS_CORE_S0_INLINE bool WriterUsesDcci(VisibilityMode mode) {
    return mode == VisibilityMode::WriterDcci || mode == VisibilityMode::WriterAndReaderDcci;
}

SIMT_CROSS_CORE_S0_INLINE bool ReaderUsesDcci(VisibilityMode mode) {
    return mode == VisibilityMode::ReaderDcci || mode == VisibilityMode::WriterAndReaderDcci;
}

constexpr uint64_t kEmptyState = EncodeExecState(ExecPhase::Empty, 0U, 0U, ExecEngineClass::None, 0U, 0U);
constexpr uint64_t kBuildingState =
    EncodeExecState(ExecPhase::Building, kBuilderOwner, kUnboundOwner, ExecEngineClass::None, 0U, kTaskId);
constexpr uint64_t kBuiltState =
    EncodeExecState(ExecPhase::Built, kBuilderOwner, kUnboundOwner, ExecEngineClass::Aiv, 1U, kTaskId);
constexpr uint64_t kClaimedState =
    EncodeExecState(ExecPhase::Claimed, kBuilderOwner, kExecutorOwner, ExecEngineClass::Aiv, 1U, kTaskId);
constexpr uint64_t kDoneState =
    EncodeExecState(ExecPhase::Done, kBuilderOwner, kExecutorOwner, ExecEngineClass::Aiv, 1U, kTaskId);

enum ProbeStatus : uint64_t {
    kStatusConfigValid = 1ULL << 0U,
    kStatusAllThreadsObserved = 1ULL << 1U,
    kStatusReserveWon = 1ULL << 2U,
    kStatusPublishWon = 1ULL << 3U,
    kStatusPayloadValid = 1ULL << 4U,
    kStatusClaimWon = 1ULL << 5U,
    kStatusDoneWon = 1ULL << 6U,
    kStatusDuplicateReserveRejected = 1ULL << 7U,
    kStatusFinalDone = 1ULL << 8U,
};

constexpr uint64_t kExpectedStatus = kStatusConfigValid | kStatusAllThreadsObserved | kStatusReserveWon |
                                     kStatusPublishWon | kStatusPayloadValid | kStatusClaimWon | kStatusDoneWon |
                                     kStatusDuplicateReserveRejected | kStatusFinalDone;

SIMT_CROSS_CORE_S0_INLINE uint64_t RotateLeft(uint64_t value, uint32_t shift) {
    return (value << shift) | (value >> (64U - shift));
}

SIMT_CROSS_CORE_S0_INLINE uint64_t ExpectedPayloadWord(uint64_t nonce, uint32_t word) {
    const uint32_t shift = (word % 23U) + 1U;
    return 0xA500000000000000ULL ^ RotateLeft(nonce ^ (0x0101010101010101ULL * (word + 1U)), shift) ^
           (static_cast<uint64_t>(kTaskId) << 32U) ^ word;
}

SIMT_CROSS_CORE_S0_INLINE uint64_t ExpectedThreadWord(uint64_t nonce, uint32_t thread) {
    const uint32_t shift = (thread % 29U) + 1U;
    return 0x5100000000000000ULL ^ RotateLeft(nonce ^ (0x0010001000100010ULL + thread), shift) ^
           static_cast<uint64_t>(thread);
}

SIMT_CROSS_CORE_S0_INLINE uint64_t FoldChecksum(uint64_t checksum, uint64_t word) {
    return RotateLeft(checksum, 9U) ^ word ^ 0x9E3779B97F4A7C15ULL;
}

struct alignas(kCacheLineBytes) ProbeControl {
    uint64_t magic;
    uint64_t version;
    uint64_t thread_count;
    uint64_t payload_words;
    uint64_t launch_nonce;
    uint64_t timeout_ticks;
    uint64_t visibility_mode;
    uint64_t reserved1;
};

struct alignas(kCacheLineBytes) ProbeCellControl {
    uint64_t state;
    uint64_t padding[7];
};

struct alignas(kCacheLineBytes) ProbePayload {
    uint64_t words[kPayloadWords];
};

struct alignas(kCacheLineBytes) ProbeSimtReport {
    uint64_t reserve_observed;
    uint64_t publish_observed;
    uint64_t participating_threads;
    uint64_t payload_words_written;
    uint64_t launch_nonce;
    uint64_t reserved[3];
};

struct alignas(kCacheLineBytes) ProbeThreadWords {
    uint64_t words[kThreadCount];
};

struct alignas(kCacheLineBytes) ProbeResult {
    uint64_t magic;
    uint64_t status;
    uint64_t physical_core_id;
    uint64_t subblock_id;
    uint64_t observed_thread_count;
    uint64_t observed_payload_words;
    uint64_t launch_nonce;
    uint64_t visibility_mode;
    uint64_t reserve_observed;
    uint64_t publish_observed;
    uint64_t claim_observed;
    uint64_t done_observed;
    uint64_t duplicate_reserve_observed;
    uint64_t final_state;
    uint64_t expected_checksum;
    uint64_t observed_checksum;
};

struct alignas(kCacheLineBytes) ProbeState {
    ProbeControl control;
    ProbeCellControl cell;
    ProbePayload payload;
    ProbeSimtReport simt_report;
    ProbeThreadWords thread_words;
    ProbeResult result;
};

static_assert(sizeof(ProbeControl) == kCacheLineBytes, "S0 control ABI changed");
static_assert(sizeof(ProbeCellControl) == kCacheLineBytes, "S0 cell control ABI changed");
static_assert(sizeof(ProbePayload) == kCacheLineBytes, "S0 payload ABI changed");
static_assert(sizeof(ProbeSimtReport) == kCacheLineBytes, "S0 SIMT report ABI changed");
static_assert(sizeof(ProbeThreadWords) == kThreadCount * sizeof(uint64_t), "S0 thread ABI changed");
static_assert(sizeof(ProbeResult) == 2U * kCacheLineBytes, "S0 result ABI changed");
static_assert(offsetof(ProbeState, cell) == kCacheLineBytes, "S0 cell offset changed");
static_assert(offsetof(ProbeState, payload) == 2U * kCacheLineBytes, "S0 payload offset changed");
static_assert(
    offsetof(ProbeState, result) % kCacheLineBytes == 0U && sizeof(ProbeState) % kCacheLineBytes == 0U,
    "S0 probe regions must remain cache-line aligned"
);

#undef SIMT_CROSS_CORE_S0_INLINE

}  // namespace pa_scheduler::simt_cross_core::s0

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_S0_PROBE_H
