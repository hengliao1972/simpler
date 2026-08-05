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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_GM_PROBE_SUPPORT_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_GM_PROBE_SUPPORT_H

#include <stdint.h>

#include "../../common/shared_protocol.h"

namespace pa_scheduler::simt_cross_core::gm {

#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_GM_INLINE __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_GM_INLINE constexpr
#endif

enum class VisibilityMode : uint64_t {
    NoDcci = 0U,
    WriterDcci = 1U,
    ReaderDcci = 2U,
    WriterAndReaderDcci = 3U,
};

constexpr uint64_t kVisibilityModeCount = 4U;

SIMT_CROSS_CORE_GM_INLINE bool WriterUsesDcci(VisibilityMode mode) {
    return mode == VisibilityMode::WriterDcci || mode == VisibilityMode::WriterAndReaderDcci;
}

SIMT_CROSS_CORE_GM_INLINE bool ReaderUsesDcci(VisibilityMode mode) {
    return mode == VisibilityMode::ReaderDcci || mode == VisibilityMode::WriterAndReaderDcci;
}

enum ProbeResultStatus : uint64_t {
    kResultConfigValid = 1ULL << 0U,
    kResultPayloadValid = 1ULL << 1U,
    kResultTaskExecuted = 1ULL << 2U,
    kResultWaitDone = 1ULL << 3U,
};

constexpr uint32_t kIdentityRoleShift = 0U;
constexpr uint32_t kIdentityBlockIndexShift = 8U;
constexpr uint32_t kIdentityBlockCountShift = 16U;
constexpr uint32_t kIdentityCoreIdShift = 24U;
constexpr uint32_t kIdentitySubblockShift = 40U;
constexpr uint32_t kIdentitySubblockCountShift = 48U;
constexpr uint64_t kIdentityByteMask = 0xFFULL;
constexpr uint64_t kIdentityCoreMask = 0xFFFFULL;

SIMT_CROSS_CORE_GM_INLINE uint64_t PackIdentity(
    uint32_t role, uint32_t block_index, uint32_t block_count, uint32_t core_id, uint32_t subblock,
    uint32_t subblock_count
) {
    return (static_cast<uint64_t>(role) << kIdentityRoleShift) |
           (static_cast<uint64_t>(block_index) << kIdentityBlockIndexShift) |
           (static_cast<uint64_t>(block_count) << kIdentityBlockCountShift) |
           ((static_cast<uint64_t>(core_id) & kIdentityCoreMask) << kIdentityCoreIdShift) |
           (static_cast<uint64_t>(subblock) << kIdentitySubblockShift) |
           (static_cast<uint64_t>(subblock_count) << kIdentitySubblockCountShift);
}

constexpr uint32_t IdentityField(uint64_t identity, uint32_t shift, uint64_t mask) {
    return static_cast<uint32_t>((identity >> shift) & mask);
}

constexpr uint32_t kCountBuildAttemptShift = 0U;
constexpr uint32_t kCountBuildWinShift = 16U;
constexpr uint32_t kCountClaimAttemptShift = 32U;
constexpr uint32_t kCountClaimWinShift = 48U;
constexpr uint64_t kCountMask = 0xFFFFULL;

SIMT_CROSS_CORE_GM_INLINE uint64_t
PackRoleCounts(uint32_t build_attempts, uint32_t build_wins, uint32_t claim_attempts, uint32_t claim_wins) {
    return (static_cast<uint64_t>(build_attempts) << kCountBuildAttemptShift) |
           (static_cast<uint64_t>(build_wins) << kCountBuildWinShift) |
           (static_cast<uint64_t>(claim_attempts) << kCountClaimAttemptShift) |
           (static_cast<uint64_t>(claim_wins) << kCountClaimWinShift);
}

constexpr uint32_t RoleCountField(uint64_t counts, uint32_t shift) {
    return static_cast<uint32_t>((counts >> shift) & kCountMask);
}

SIMT_CROSS_CORE_GM_INLINE uint64_t RotateLeft(uint64_t value, uint32_t shift) {
    return (value << shift) | (value >> (64U - shift));
}

SIMT_CROSS_CORE_GM_INLINE uint64_t FoldDescriptorChecksum(uint64_t checksum, uint64_t word, uint32_t index) {
    return RotateLeft(checksum ^ word ^ (0x9E3779B97F4A7C15ULL + index), (index % 19U) + 3U);
}

SIMT_CROSS_CORE_GM_INLINE uint64_t PackTaskShape(uint32_t task_id, uint32_t element_count) {
    return (static_cast<uint64_t>(task_id) << 32U) | element_count;
}

SIMT_CROSS_CORE_GM_INLINE uint32_t PayloadTaskId(uint64_t shape) { return static_cast<uint32_t>(shape >> 32U); }

SIMT_CROSS_CORE_GM_INLINE uint32_t PayloadElementCount(uint64_t shape) { return static_cast<uint32_t>(shape); }

constexpr float kOutputSentinel = -123456.0F;

constexpr uint64_t ExpectedGuardWord(uint64_t nonce, uint32_t guard, uint32_t word) {
    return 0xD00DFEED00000000ULL ^ (nonce << 1U) ^ (static_cast<uint64_t>(guard) << 12U) ^ word;
}

struct alignas(kCacheLineBytes) ProbeControl {
    uint64_t magic;
    uint64_t version;
    uint64_t launch_nonce;
    uint64_t timeout_ticks;
    uint64_t visibility_mode;
    uint64_t thread_count;
    uint64_t element_count;
    uint64_t task_id;
};

struct alignas(kCacheLineBytes) ProbeCellControl {
    uint64_t state;
    uint64_t padding[7];
};

struct alignas(kCacheLineBytes) ProbeFatalControl {
    uint64_t state;
    uint64_t padding[7];
};

struct alignas(kCacheLineBytes) ProbeSimtReport {
    uint64_t reserve_observed;
    uint64_t publish_observed;
    uint64_t participating_threads;
    uint64_t payload_words_written;
    uint64_t launch_nonce;
    uint64_t builder_thread;
    uint64_t writer_dcci;
    uint64_t reserved;
};

struct alignas(kCacheLineBytes) ProbeRoleResult {
    uint64_t magic;
    uint64_t identity;
    uint64_t status;
    uint64_t role_counts;
    uint64_t launch_nonce;
    uint64_t visibility_mode;
    uint64_t reserve_observed;
    uint64_t publish_observed;
    uint64_t claim_observed;
    uint64_t done_observed;
    uint64_t final_state;
    uint64_t fatal_state;
    uint64_t observed_payload_checksum;
    uint64_t expected_payload_checksum;
    uint64_t observed_elements;
    uint64_t reserved;
};

struct alignas(kCacheLineBytes) ProbeGuard {
    uint64_t words[kCacheLineBytes / sizeof(uint64_t)];
};

static_assert(sizeof(ProbeControl) == kCacheLineBytes, "GM probe control ABI changed");
static_assert(sizeof(ProbeCellControl) == kCacheLineBytes, "GM probe cell ABI changed");
static_assert(sizeof(ProbeFatalControl) == kCacheLineBytes, "GM probe fatal ABI changed");
static_assert(sizeof(ProbeSimtReport) == kCacheLineBytes, "GM probe SIMT report ABI changed");
static_assert(sizeof(ProbeRoleResult) == 2U * kCacheLineBytes, "GM probe role-result ABI changed");
static_assert(sizeof(ProbeGuard) == kCacheLineBytes, "GM probe guard ABI changed");

#undef SIMT_CROSS_CORE_GM_INLINE

}  // namespace pa_scheduler::simt_cross_core::gm

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_GM_PROBE_SUPPORT_H
