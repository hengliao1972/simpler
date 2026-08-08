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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_WARP_CONCURRENCY_PROBE_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_WARP_CONCURRENCY_PROBE_H

#include <stddef.h>
#include <stdint.h>

namespace pa_scheduler::simt_cross_core::warp_concurrency {

#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_WARP_INLINE __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_WARP_INLINE constexpr
#endif

constexpr uint64_t kProbeMagic = 0x57415250434F4E43ULL;
constexpr uint64_t kProbeVersion = 1U;
constexpr uint64_t kReportMagic = 0x574152505245504FULL;
constexpr uint64_t kResultMagic = 0x5741525052455355ULL;
constexpr uint64_t kOutputSentinel = 0xBAD0BAD0BAD0BAD0ULL;
constexpr uint32_t kCacheLineBytes = 64U;
constexpr uint32_t kWarpSize = 32U;
constexpr uint32_t kLaunchThreads = 64U;
constexpr uint32_t kParticipantCount = 2U;
constexpr uint32_t kDefaultPollLimit = 200000U;
constexpr uint64_t kDefaultPollClockBudget = 1000000000ULL;
constexpr uint32_t kDefaultWorkIterations = 20000U;
constexpr uint32_t kMaximumPollLimit = 2000000U;
constexpr uint64_t kMaximumPollClockBudget = 10000000000ULL;
constexpr uint32_t kMaximumWorkIterations = 2000000U;

enum class ProbeMode : uint64_t {
    kAOnly = 1U,
    kBOnly = 2U,
    kSameWarp = 3U,
    kCrossWarp = 4U,
};

enum class HandshakeStatus : uint64_t {
    kUnset = 0U,
    kSuccess = 1U,
    kTimeout = 2U,
    kNotApplicable = 3U,
};

enum class IntervalRelation : uint64_t {
    kUnknown = 0U,
    kDisjoint = 1U,
    kOverlap = 2U,
    kSingle = 3U,
};

enum ProbeStatus : uint64_t {
    kStatusConfigValid = 1ULL << 0U,
    kStatusReportsComplete = 1ULL << 1U,
    kStatusReadyValues = 1ULL << 2U,
    kStatusHandshakeMatchesMode = 1ULL << 3U,
    kStatusIntervalMatchesMode = 1ULL << 4U,
};

constexpr uint64_t kExpectedStatus = kStatusConfigValid | kStatusReportsComplete | kStatusReadyValues |
                                     kStatusHandshakeMatchesMode | kStatusIntervalMatchesMode;

SIMT_CROSS_CORE_WARP_INLINE uint64_t RotateLeft(uint64_t value, uint32_t shift) {
    return (value << shift) | (value >> (64U - shift));
}

SIMT_CROSS_CORE_WARP_INLINE bool IsValidMode(uint64_t raw_mode) {
    return raw_mode >= static_cast<uint64_t>(ProbeMode::kAOnly) &&
           raw_mode <= static_cast<uint64_t>(ProbeMode::kCrossWarp);
}

SIMT_CROSS_CORE_WARP_INLINE uint32_t ExpectedActiveParticipants(ProbeMode mode) {
    return mode == ProbeMode::kAOnly || mode == ProbeMode::kBOnly ? 1U : 2U;
}

SIMT_CROSS_CORE_WARP_INLINE bool ParticipantIsActive(ProbeMode mode, uint32_t participant) {
    return mode == ProbeMode::kAOnly ? participant == 0U :
           mode == ProbeMode::kBOnly ? participant == 1U :
                                       participant < 2U;
}

SIMT_CROSS_CORE_WARP_INLINE uint32_t ExpectedThread(ProbeMode mode, uint32_t participant) {
    if (participant == 0U) {
        return 0U;
    }
    return mode == ProbeMode::kCrossWarp ? kWarpSize : kWarpSize / 2U;
}

SIMT_CROSS_CORE_WARP_INLINE uint64_t ExpectedReady(uint64_t nonce, uint32_t participant) {
    return 0xA100000000000000ULL ^ RotateLeft(nonce, participant * 11U + 7U) ^
           (0x0001000100010001ULL * static_cast<uint64_t>(participant + 1U));
}

SIMT_CROSS_CORE_WARP_INLINE uint64_t ExpectedReportMarker(uint64_t nonce, ProbeMode mode, uint32_t participant) {
    return kReportMagic ^ RotateLeft(nonce, participant * 13U + 3U) ^ (static_cast<uint64_t>(mode) << 32U) ^
           static_cast<uint64_t>(participant + 1U);
}

SIMT_CROSS_CORE_WARP_INLINE uint64_t WorkA(uint64_t nonce, uint32_t iterations) {
    uint64_t value = nonce ^ 0x243F6A8885A308D3ULL;
    for (uint32_t index = 0U; index < iterations; ++index) {
        value = RotateLeft(value ^ (0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(index)), 7U) + 0xD1B54A32D192ED03ULL;
    }
    return value;
}

SIMT_CROSS_CORE_WARP_INLINE uint64_t WorkB(uint64_t nonce, uint32_t iterations) {
    uint64_t value = nonce ^ 0x13198A2E03707344ULL;
    for (uint32_t index = 0U; index < iterations; ++index) {
        value += 0x94D049BB133111EBULL ^ static_cast<uint64_t>(index);
        value = RotateLeft(value, 19U) ^ (value >> 11U) ^ 0xBF58476D1CE4E5B9ULL;
    }
    return value;
}

struct alignas(kCacheLineBytes) ProbeControl {
    uint64_t magic;
    uint64_t version;
    uint64_t mode;
    uint64_t launch_nonce;
    uint64_t poll_limit;
    uint64_t poll_clock_budget;
    uint64_t work_iterations;
    uint64_t reserved;
};

struct alignas(kCacheLineBytes) AtomicCell {
    uint64_t value;
    uint64_t padding[7];
};

struct alignas(kCacheLineBytes) ProbeGuard {
    uint64_t words[8];
};

struct alignas(kCacheLineBytes) ParticipantReport {
    uint64_t marker;
    uint64_t launch_nonce;
    uint64_t mode;
    uint64_t participant;
    uint64_t thread_id;
    uint64_t warp_id;
    uint64_t lane_id;
    uint64_t handshake_status;
    uint64_t poll_count;
    uint64_t observed_peer;
    uint64_t start_clock;
    uint64_t end_clock;
    uint64_t checksum;
    uint64_t published_ready;
    uint64_t reserved[2];
};

struct alignas(kCacheLineBytes) ProbeResult {
    uint64_t magic;
    uint64_t status;
    uint64_t physical_core_id;
    uint64_t subblock_id;
    uint64_t launch_nonce;
    uint64_t mode;
    uint64_t active_reports;
    uint64_t handshake_successes;
    uint64_t handshake_timeouts;
    uint64_t handshake_not_applicable;
    uint64_t ready_a;
    uint64_t ready_b;
    uint64_t interval_relation;
    uint64_t first_start;
    uint64_t last_end;
    uint64_t reserved;
};

struct alignas(kCacheLineBytes) ProbeState {
    ProbeControl control;
    ProbeGuard guard_before_ready_a;
    AtomicCell ready_a;
    ProbeGuard guard_before_ready_b;
    AtomicCell ready_b;
    ProbeGuard guard_before_reports;
    ParticipantReport reports[kParticipantCount];
    ProbeGuard guard_after_reports;
    ProbeResult result;
    ProbeGuard guard_after_result;
};

static_assert(sizeof(ProbeControl) == kCacheLineBytes, "warp control ABI changed");
static_assert(sizeof(AtomicCell) == kCacheLineBytes, "warp atomic cell ABI changed");
static_assert(sizeof(ProbeGuard) == kCacheLineBytes, "warp guard ABI changed");
static_assert(sizeof(ParticipantReport) == 2U * kCacheLineBytes, "warp participant report ABI changed");
static_assert(sizeof(ProbeResult) == 2U * kCacheLineBytes, "warp result ABI changed");
static_assert(
    offsetof(ProbeState, ready_a) % kCacheLineBytes == 0U && offsetof(ProbeState, ready_b) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, reports) % kCacheLineBytes == 0U && offsetof(ProbeState, result) % kCacheLineBytes == 0U &&
        sizeof(ProbeState) % kCacheLineBytes == 0U,
    "warp probe GM regions must remain cache-line aligned"
);

#undef SIMT_CROSS_CORE_WARP_INLINE

}  // namespace pa_scheduler::simt_cross_core::warp_concurrency

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_WARP_CONCURRENCY_PROBE_H
