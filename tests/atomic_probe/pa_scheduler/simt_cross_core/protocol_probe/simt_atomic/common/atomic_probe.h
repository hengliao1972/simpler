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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_SIMT_ATOMIC_PROBE_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_SIMT_ATOMIC_PROBE_H

#include <stddef.h>
#include <stdint.h>

#include "shared_protocol.h"

namespace pa_scheduler::simt_cross_core::simt_atomic {

#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_ATOMIC_INLINE __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_ATOMIC_INLINE constexpr
#endif

constexpr uint64_t kProbeMagic = 0x53494D5441544F4DULL;
constexpr uint64_t kProbeVersion = 1U;
constexpr uint64_t kResultMagic = 0x41544F4D52455355ULL;
constexpr uint32_t kWarpSize = 32U;
constexpr uint32_t kMaxThreadCount = 2048U;
constexpr uint32_t kThreadConfigCount = 4U;
constexpr uint32_t kThreadConfigs[kThreadConfigCount] = {32U, 64U, 1024U, 2048U};
constexpr uint64_t kOutputSentinel = 0xBAD0BAD0BAD0BAD0ULL;

enum ProbeStatus : uint64_t {
    kStatusConfigValid = 1ULL << 0U,
    kStatusAllThreadsObserved = 1ULL << 1U,
    kStatusCasUniqueWinner = 1ULL << 2U,
    kStatusCasReturnValues = 1ULL << 3U,
    kStatusCasFinalValue = 1ULL << 4U,
    kStatusAddTicketPermutation = 1ULL << 5U,
    kStatusAddFinalValue = 1ULL << 6U,
    kStatusFullWidth64Bit = 1ULL << 7U,
};

constexpr uint64_t kExpectedStatus = kStatusConfigValid | kStatusAllThreadsObserved | kStatusCasUniqueWinner |
                                     kStatusCasReturnValues | kStatusCasFinalValue | kStatusAddTicketPermutation |
                                     kStatusAddFinalValue | kStatusFullWidth64Bit;

SIMT_CROSS_CORE_ATOMIC_INLINE uint64_t RotateLeft(uint64_t value, uint32_t shift) {
    return (value << shift) | (value >> (64U - shift));
}

SIMT_CROSS_CORE_ATOMIC_INLINE uint64_t ExpectedCasInitial(uint64_t nonce) {
    return 0xC0DEC0DE00000000ULL | (nonce & 0x00000000FFFFFFFFULL);
}

SIMT_CROSS_CORE_ATOMIC_INLINE uint64_t ExpectedCasDesired(uint64_t nonce, uint32_t thread) {
    return 0xA700000000000000ULL | ((nonce >> 8U) & 0x0000FFFFFFFFF000ULL) | static_cast<uint64_t>(thread + 1U);
}

SIMT_CROSS_CORE_ATOMIC_INLINE uint64_t ExpectedAddInitial(uint64_t nonce) {
    return 0x1234567800000000ULL | ((nonce >> 8U) & 0x00000000FFFFF000ULL);
}

SIMT_CROSS_CORE_ATOMIC_INLINE uint64_t ExpectedThreadMarker(uint64_t nonce, uint32_t thread) {
    return 0x5A00000000000000ULL ^ RotateLeft(nonce, (thread % 29U) + 1U) ^
           (0x0010001000100010ULL + static_cast<uint64_t>(thread));
}

struct alignas(kCacheLineBytes) ProbeControl {
    uint64_t magic;
    uint64_t version;
    uint64_t thread_count;
    uint64_t launch_nonce;
    uint64_t cas_initial;
    uint64_t add_initial;
    uint64_t reserved[2];
};

struct alignas(kCacheLineBytes) ProbeAtomicCell {
    uint64_t value;
    uint64_t padding[7];
};

struct alignas(kCacheLineBytes) ProbeThreadValues {
    uint64_t values[kMaxThreadCount];
};

struct alignas(kCacheLineBytes) ProbeGuard {
    uint64_t words[8];
};

struct alignas(kCacheLineBytes) ProbeResult {
    uint64_t magic;
    uint64_t status;
    uint64_t physical_core_id;
    uint64_t subblock_id;
    uint64_t launch_nonce;
    uint64_t requested_thread_count;
    uint64_t observed_thread_count;
    uint64_t warp_size;
    uint64_t cas_initial;
    uint64_t cas_final;
    uint64_t cas_winner_count;
    uint64_t cas_loser_count;
    uint64_t add_initial;
    uint64_t add_final;
    uint64_t add_ticket_in_range;
    uint64_t add_ticket_sum;
};

struct alignas(kCacheLineBytes) ProbeState {
    ProbeControl control;
    ProbeGuard guard_before_cas;
    ProbeAtomicCell cas_cell;
    ProbeGuard guard_before_add;
    ProbeAtomicCell add_cell;
    ProbeGuard guard_before_cas_returns;
    ProbeThreadValues cas_returns;
    ProbeGuard guard_before_add_returns;
    ProbeThreadValues add_returns;
    ProbeGuard guard_before_markers;
    ProbeThreadValues markers;
    ProbeGuard guard_after_markers;
    ProbeResult result;
};

static_assert(sizeof(ProbeControl) == kCacheLineBytes, "SIMT atomic control ABI changed");
static_assert(sizeof(ProbeAtomicCell) == kCacheLineBytes, "SIMT atomic cell ABI changed");
static_assert(sizeof(ProbeThreadValues) == kMaxThreadCount * sizeof(uint64_t), "SIMT atomic thread-value ABI changed");
static_assert(sizeof(ProbeGuard) == kCacheLineBytes, "SIMT atomic guard ABI changed");
static_assert(sizeof(ProbeResult) == 2U * kCacheLineBytes, "SIMT atomic result ABI changed");
static_assert(
    offsetof(ProbeState, cas_cell) % kCacheLineBytes == 0U && offsetof(ProbeState, add_cell) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, cas_returns) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, add_returns) % kCacheLineBytes == 0U &&
        offsetof(ProbeState, markers) % kCacheLineBytes == 0U && sizeof(ProbeState) % kCacheLineBytes == 0U,
    "SIMT atomic GM regions must remain cache-line aligned"
);

#undef SIMT_CROSS_CORE_ATOMIC_INLINE

}  // namespace pa_scheduler::simt_cross_core::simt_atomic

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_SIMT_ATOMIC_PROBE_H
