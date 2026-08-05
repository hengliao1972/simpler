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

// U1 is a four-slot capacity/reuse diagnostic.  AIV0 launches 2048 SIMT
// threads, but only lane0 of each of 64 warps builds its two independent
// tasks.  Every successful slot generation uses volatile AS6 UBUF stores and
// loads followed by ordinary GM word stores; MTE3 is deliberately absent.

#include <pto/common/kernel_meta.hpp>

#include "cce_aicore_intrinsics.h"

#if defined(__DAV_VEC__)
#include "simt_api/asc_simt.h"
#endif

#include "../common/u1_multi_slot.h"

namespace {

using namespace pa_scheduler::simt_cross_core::u1;
namespace g0 = pa_scheduler::simt_cross_core::g0;

constexpr int kSingleCacheLine = 0;
constexpr uint32_t kWatchdogMask = 0x3FFU;
constexpr uint32_t kAtomicCasAttemptLimit = 4096U;
constexpr uintptr_t kUbufRegionOffset = 0U;

__aicore__ __attribute__((always_inline)) inline uint64_t AtomicLoad(__gm__ volatile int64_t *address) {
    return static_cast<uint64_t>(atomicAdd(const_cast<__gm__ int64_t *>(address), static_cast<int64_t>(0)));
}

__aicore__ __attribute__((always_inline)) inline uint64_t
AtomicCas(__gm__ volatile int64_t *address, uint64_t expected, uint64_t desired) {
    return static_cast<uint64_t>(
        atomicCAS(const_cast<__gm__ int64_t *>(address), static_cast<int64_t>(expected), static_cast<int64_t>(desired))
    );
}

__aicore__ __attribute__((always_inline)) inline uint64_t
AtomicFetchAdd(__gm__ volatile int64_t *address, uint64_t increment) {
    return static_cast<uint64_t>(atomicAdd(const_cast<__gm__ int64_t *>(address), static_cast<int64_t>(increment)));
}

__aicore__ __attribute__((always_inline)) inline void StoreDev64(__gm__ uint64_t *address, uint64_t value) {
    __builtin_cce_st_dev(value, address, 0);
}

__aicore__ __attribute__((always_inline)) inline void
PublishFatal(__gm__ U1ProbeState *state, U1FatalReason reason, uint32_t owner, uint32_t task_id) {
    (void)AtomicCas(&state->fatal.value, 0U, EncodeU1Fatal(reason, owner, task_id));
}

__aicore__ __attribute__((always_inline)) inline bool ConfigValid(__gm__ const U1ProbeState *state) {
    return state->control.magic == kProbeMagic && state->control.version == kProbeVersion &&
           state->control.launch_nonce != 0U && state->control.timeout_ticks != 0U &&
           state->control.thread_count == kThreadCount && state->control.warp_count == kWarpCount &&
           state->control.task_count == kTaskCount && state->control.role_count == kRoleCount &&
           state->control.payload_class_count == kPayloadClassCount &&
           state->control.max_payload_lines == kMaxPayloadLines && state->control.words_per_line == kWordsPerLine &&
           state->control.ubuf_alignment_bytes == kUbufAlignmentBytes &&
           state->control.ubuf_slot_count == kUbufSlotCount &&
           state->control.transport_kind == TransportKind::SimtUbufReadToGmWordStore &&
           state->control.ubuf_slot_stride_bytes == kUbufSlotStrideBytes &&
           state->control.ubuf_region_bytes == kUbufRegionBytes &&
           state->control.ubuf_payload_offset_bytes == kUbufPayloadOffsetBytes;
}

__aicore__ __attribute__((always_inline)) inline void PublishRole(
    __gm__ U1RoleReport *destination, uint32_t owner, ProbeRole role, uint32_t physical_block, uint32_t subblock_id,
    uint32_t task_claim_count, uint32_t task_finish_count, uint32_t timeout_count, uint64_t launch_nonce
) {
    __gm__ uint64_t *words = reinterpret_cast<__gm__ uint64_t *>(destination);
    StoreDev64(words + 0U, static_cast<uint64_t>(owner) | (static_cast<uint64_t>(role) << 32U));
    StoreDev64(words + 1U, static_cast<uint64_t>(physical_block) | (static_cast<uint64_t>(subblock_id) << 32U));
    // AIV0 Main Scalar only invokes/joins the SIMT VF and drains.  Its build
    // action count therefore remains zero, as do those of the other roles.
    StoreDev64(words + 2U, static_cast<uint64_t>(0U) | (static_cast<uint64_t>(task_claim_count) << 32U));
    StoreDev64(words + 3U, static_cast<uint64_t>(task_finish_count) | (static_cast<uint64_t>(timeout_count) << 32U));
    StoreDev64(words + 4U, launch_nonce);
    StoreDev64(words + 5U, kResultMagic);
    StoreDev64(words + 6U, 0U);
    StoreDev64(words + 7U, 0U);
    dsb(DSB_ALL);
}

__aicore__ __attribute__((always_inline)) inline void PublishExecReport(
    __gm__ U1ExecReport *destination, uint32_t task_id, uint32_t physical_block, uint32_t subblock_id,
    uint32_t phase_bits, uint32_t payload_lines, uint32_t payload_words, bool checksum_matches, uint64_t launch_nonce,
    uint64_t checksum
) {
    __gm__ uint64_t *words = reinterpret_cast<__gm__ uint64_t *>(destination);
    StoreDev64(words + 0U, static_cast<uint64_t>(task_id) | (static_cast<uint64_t>(kExecutorOwner) << 32U));
    StoreDev64(words + 1U, static_cast<uint64_t>(physical_block) | (static_cast<uint64_t>(subblock_id) << 32U));
    StoreDev64(words + 2U, static_cast<uint64_t>(phase_bits) | (static_cast<uint64_t>(payload_lines) << 32U));
    StoreDev64(words + 3U, static_cast<uint64_t>(payload_words) | (static_cast<uint64_t>(1U) << 32U));
    StoreDev64(
        words + 4U,
        static_cast<uint64_t>(checksum_matches ? 1U : 0U) | (static_cast<uint64_t>(checksum_matches ? 1U : 0U) << 32U)
    );
    StoreDev64(words + 5U, 0U);
    StoreDev64(words + 6U, launch_nonce);
    StoreDev64(words + 7U, checksum);
    dsb(DSB_ALL);
}

__aicore__ bool WaitForTerminal(__gm__ U1ProbeState *state, uint32_t owner, bool *timed_out) {
    *timed_out = false;
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint32_t polls = 0U;
    while (true) {
        if (AtomicLoad(&state->built_count.value) == kTaskCount && AtomicLoad(&state->done_count.value) == kTaskCount) {
            return true;
        }
        if (AtomicLoad(&state->fatal.value) != 0U) {
            return false;
        }
        ++polls;
        if ((polls & kWatchdogMask) == 0U &&
            static_cast<uint64_t>(get_sys_cnt()) - begin > state->control.timeout_ticks) {
            *timed_out = true;
            PublishFatal(state, U1FatalReason::DrainTimeout, owner, kInvalidTaskId);
            return false;
        }
    }
}

#if defined(__DAV_VEC__)

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtRotateLeft(uint64_t value, uint32_t shift) {
    return (value << shift) | (value >> (64U - shift));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint32_t SimtPayloadClass(uint32_t task_id) {
    return ((task_id / kUbufSlotCount) + (kPayloadClassCount - 1U)) & (kPayloadClassCount - 1U);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint32_t SimtPayloadLines(uint32_t task_id) {
    const uint32_t payload_class = SimtPayloadClass(task_id);
    return payload_class == 0U ? 1U : (payload_class == 1U ? 4U : (payload_class == 2U ? 10U : 16U));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtPayloadWord(uint64_t nonce, uint32_t task_id, uint32_t word) {
    const uint32_t shift = ((task_id * 7U + word) % 31U) + 1U;
    return 0x5531000000000000ULL ^ SimtRotateLeft(nonce ^ (0x0101010101010101ULL * (word + 1U)), shift) ^
           (static_cast<uint64_t>(task_id) << 40U) ^ static_cast<uint64_t>(word);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtFoldChecksum(uint64_t checksum, uint64_t word, uint32_t index) {
    return SimtRotateLeft(checksum, 11U) ^ word ^ (0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(index));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t SimtBuildingState(uint32_t task_id) {
    return (static_cast<uint64_t>(g0::ExecPhase::Building) << g0::kStatePhaseShift) |
           (static_cast<uint64_t>(kBuilderOwner) << g0::kStateBuildOwnerShift) |
           (static_cast<uint64_t>(g0::kUnboundOwner) << g0::kStateExecuteOwnerShift) |
           (static_cast<uint64_t>(task_id) << g0::kStateTaskIdShift);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtBuiltState(uint32_t task_id, uint32_t payload_lines) {
    return (static_cast<uint64_t>(g0::ExecPhase::Built) << g0::kStatePhaseShift) |
           (static_cast<uint64_t>(kBuilderOwner) << g0::kStateBuildOwnerShift) |
           (static_cast<uint64_t>(g0::kUnboundOwner) << g0::kStateExecuteOwnerShift) |
           (static_cast<uint64_t>(g0::ExecEngineClass::Aiv) << g0::kStateEngineShift) |
           (static_cast<uint64_t>(payload_lines) << g0::kStatePayloadLinesShift) |
           (static_cast<uint64_t>(task_id) << g0::kStateTaskIdShift);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtFatalValue(U1FatalReason reason, uint32_t task_id) {
    return (static_cast<uint64_t>(reason) << g0::kFatalReasonShift) |
           (static_cast<uint64_t>(kBuilderOwner) << g0::kFatalOwnerShift) |
           (static_cast<uint64_t>(task_id) << g0::kFatalTaskIdShift);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void
SimtPublishFatal(__gm__ uint64_t *fatal, U1FatalReason reason, uint32_t task_id) {
    (void)asc_atomic_cas(fatal, static_cast<uint64_t>(0U), SimtFatalValue(reason, task_id));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void
SimtResetUnpublishedTask(__gm__ uint64_t *task, uint64_t building, __gm__ uint64_t *fatal, uint32_t task_id) {
    if (asc_atomic_cas(task, building, static_cast<uint64_t>(0U)) != building) {
        SimtPublishFatal(fatal, U1FatalReason::TaskPublishConflict, task_id);
    }
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtExpectedUbufGuardWord(uint64_t nonce, uint32_t guard_id, uint32_t word) {
    uint64_t value = kGuardMagic ^ nonce ^ (static_cast<uint64_t>(guard_id) << 32U) ^ word;
    value ^= value >> 27U;
    value *= 0xD6E8FEB86659FD93ULL;
    return value ^ (value >> 31U);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtThreadChecksum(uint64_t nonce, uint32_t thread_id, uint32_t first_task_id, uint32_t last_task_id, uint32_t status) {
    uint64_t checksum = nonce ^ (static_cast<uint64_t>(thread_id) << 32U) ^ first_task_id;
    checksum = SimtFoldChecksum(checksum, last_task_id, 0U);
    checksum = SimtFoldChecksum(checksum, status, 1U);
    return SimtFoldChecksum(checksum, kTransportMarker, 2U);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t SimtSlotFreeState(uint32_t generation) {
    return static_cast<uint64_t>(generation) << 32U;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtSlotBusyState(uint32_t generation, uint32_t task_id) {
    return (static_cast<uint64_t>(generation) << 32U) | (static_cast<uint64_t>(task_id) + 1U);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtWaitForAnchorGate(
    __gm__ uint64_t *anchor_staged, __gm__ uint64_t *anchor_mask, __gm__ uint64_t *fatal, uint64_t timeout_ticks,
    U1FatalReason *failure_reason
) {
    *failure_reason = U1FatalReason::None;
    const uint64_t begin = clock();
    uint32_t polls = 0U;
    while (true) {
        const uint64_t staged = asc_atomic_add(anchor_staged, static_cast<uint64_t>(0U));
        const uint64_t mask = asc_atomic_add(anchor_mask, static_cast<uint64_t>(0U));
        if (staged == kAnchorTaskCount && mask == kAnchorStagedMask) {
            return true;
        }
        if (staged > kAnchorTaskCount || (mask & ~kAnchorStagedMask) != 0U) {
            *failure_reason = U1FatalReason::SlotInvariant;
            return false;
        }
        // Count and identity mask are different GM atomic cachelines.  Even
        // though each publisher updates mask before count, a reader must not
        // turn a transient cross-line visibility skew into a permanent fatal.
        // Only exact count/mask opens the gate; all in-range mismatches retry
        // until the existing watchdog expires.
        if (asc_atomic_add(fatal, static_cast<uint64_t>(0U)) != 0U) {
            return false;
        }
        ++polls;
        if ((polls & kWatchdogMask) == 0U && clock() - begin > timeout_ticks) {
            *failure_reason = U1FatalReason::SlotTimeout;
            return false;
        }
    }
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool
SimtStageAnchorIdentity(__gm__ uint64_t *anchor_mask, uint32_t task_id, U1FatalReason *failure_reason) {
    *failure_reason = U1FatalReason::None;
    const uint64_t bit = static_cast<uint64_t>(1U) << task_id;
    for (uint32_t attempt = 0U; attempt < kAtomicCasAttemptLimit; ++attempt) {
        const uint64_t observed = asc_atomic_add(anchor_mask, static_cast<uint64_t>(0U));
        if ((observed & ~kAnchorStagedMask) != 0U || (observed & bit) != 0U) {
            *failure_reason = U1FatalReason::SlotInvariant;
            return false;
        }
        if (asc_atomic_cas(anchor_mask, observed, observed | bit) == observed) {
            return true;
        }
    }
    *failure_reason = U1FatalReason::SlotTimeout;
    return false;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtDecrementBusy(__gm__ uint64_t *global_busy) {
    for (uint32_t attempt = 0U; attempt < kAtomicCasAttemptLimit; ++attempt) {
        const uint64_t observed = asc_atomic_add(global_busy, static_cast<uint64_t>(0U));
        // An acquire that observes capacity overflow has already incremented
        // this counter.  It must still be able to roll that increment back;
        // rejecting only zero keeps the CAS decrement free of underflow while
        // the acquire path separately records >4 as an invariant failure.
        if (observed == 0U) {
            return false;
        }
        if (asc_atomic_cas(global_busy, observed, observed - 1U) == observed) {
            return true;
        }
    }
    return false;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtReleaseSlot(
    __gm__ uint64_t *slot_state, __gm__ uint64_t *slot_releases, __gm__ uint64_t *global_busy, __gm__ uint64_t *fatal,
    uint32_t generation, uint32_t task_id
) {
    const uint64_t busy_state = SimtSlotBusyState(generation, task_id);
    if (!SimtDecrementBusy(global_busy)) {
        SimtPublishFatal(fatal, U1FatalReason::SlotInvariant, task_id);
        return false;
    }
    if (asc_atomic_cas(slot_state, busy_state, SimtSlotFreeState(generation + 1U)) != busy_state) {
        // The slot is still BUSY if the exact-state CAS failed, so restore the
        // global count before publishing the invariant failure.
        (void)asc_atomic_add(global_busy, static_cast<uint64_t>(1U));
        SimtPublishFatal(fatal, U1FatalReason::SlotInvariant, task_id);
        return false;
    }
    (void)asc_atomic_add(slot_releases, static_cast<uint64_t>(1U));
    return true;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtAcquireSlot(
    __gm__ uint64_t *slot_state, __gm__ uint64_t *slot_acquires, __gm__ uint64_t *slot_releases,
    __gm__ uint64_t *global_busy, __gm__ uint64_t *global_max_busy, __gm__ uint64_t *fatal, uint32_t slot_id,
    uint32_t task_id, uint64_t timeout_ticks, uint32_t *slot_generation, uint32_t *slot_attempts,
    U1FatalReason *failure_reason
) {
    *failure_reason = U1FatalReason::None;
    const uint64_t begin = clock();
    while (true) {
        ++(*slot_attempts);
        const uint64_t observed = asc_atomic_add(slot_state, static_cast<uint64_t>(0U));
        const uint32_t generation = static_cast<uint32_t>(observed >> 32U);
        const uint32_t task_plus_one = static_cast<uint32_t>(observed);
        if (generation >= kSlotReuseCount || task_plus_one > kTaskCount) {
            *failure_reason = U1FatalReason::SlotInvariant;
            return false;
        }
        if (task_plus_one != 0U) {
            const uint32_t owner_task = task_plus_one - 1U;
            if ((owner_task % kUbufSlotCount) != slot_id) {
                *failure_reason = U1FatalReason::SlotInvariant;
                return false;
            }
        } else if (asc_atomic_cas(slot_state, observed, SimtSlotBusyState(generation, task_id)) == observed) {
            *slot_generation = generation;
            // Count the successful slot CAS immediately.  Every subsequent
            // failure path performs the exact generation-advancing release.
            (void)asc_atomic_add(slot_acquires, static_cast<uint64_t>(1U));
            const uint64_t prior_busy = asc_atomic_add(global_busy, static_cast<uint64_t>(1U));
            const uint64_t current_busy = prior_busy + 1U;
            if (prior_busy >= kUbufSlotCount || current_busy > kUbufSlotCount) {
                (void)SimtReleaseSlot(slot_state, slot_releases, global_busy, fatal, generation, task_id);
                *failure_reason = U1FatalReason::SlotInvariant;
                return false;
            }
            uint64_t observed_max = asc_atomic_add(global_max_busy, static_cast<uint64_t>(0U));
            while (observed_max < current_busy) {
                const uint64_t prior_max = asc_atomic_cas(global_max_busy, observed_max, current_busy);
                if (prior_max == observed_max) {
                    break;
                }
                observed_max = prior_max;
            }
            return true;
        }
        if ((*slot_attempts & kWatchdogMask) == 0U) {
            if (asc_atomic_add(fatal, static_cast<uint64_t>(0U)) != 0U) {
                return false;
            }
            if (clock() - begin > timeout_ticks) {
                *failure_reason = U1FatalReason::SlotTimeout;
                return false;
            }
        }
    }
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(kThreadCount) void U1SimtBuildMultiSlot(
    __gm__ uint64_t *task_words, __gm__ uint64_t *slot_state_words, __gm__ uint64_t *slot_acquire_words,
    __gm__ uint64_t *slot_release_words, __gm__ uint64_t *global_busy, __gm__ uint64_t *global_max_busy,
    __gm__ uint64_t *anchor_staged, __gm__ uint64_t *anchor_mask, __gm__ uint64_t *ubuf_guard_checks,
    __gm__ uint64_t *built_count, __gm__ uint64_t *fatal, __gm__ uint64_t *thread_report_words,
    __ubuf__ volatile uint64_t *staging_region, uint64_t nonce, uint64_t timeout_ticks
) {
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    const uint32_t warp = thread / kWarpSize;
    const uint32_t lane = thread % kWarpSize;
    if (thread >= kThreadCount || warp >= kWarpCount || lane != 0U) {
        return;
    }

    constexpr uint32_t kTaskStrideWords = sizeof(U1Task) / sizeof(uint64_t);
    constexpr uint32_t kPayloadOffsetWords = offsetof(U1Task, payload) / sizeof(uint64_t);
    constexpr uint32_t kBuildReportOffsetWords = offsetof(U1Task, build_report) / sizeof(uint64_t);
    constexpr uint32_t kThreadReportStrideWords = sizeof(U1ThreadReport) / sizeof(uint64_t);
    constexpr uint32_t kAtomicStrideWords = sizeof(g0::AtomicLine) / sizeof(uint64_t);
    uint32_t status = kThreadActiveLeader;
    uint32_t slot_attempts = 0U;
    uint32_t build_count = 0U;
    const uint32_t first_task = warp;
    const uint32_t last_task = warp + kWarpCount;

    for (uint32_t task_id = first_task; task_id < kTaskCount; task_id += kWarpCount) {
        __gm__ uint64_t *task = task_words + task_id * kTaskStrideWords;
        const uint32_t slot_id = task_id % kUbufSlotCount;
        const uint32_t payload_lines = SimtPayloadLines(task_id);
        const uint32_t payload_words = payload_lines * kWordsPerLine;
        const uint64_t building = SimtBuildingState(task_id);
        const uint64_t built = SimtBuiltState(task_id, payload_lines);

        if (asc_atomic_cas(task, static_cast<uint64_t>(0U), building) != 0U) {
            SimtPublishFatal(fatal, U1FatalReason::TaskClaimConflict, task_id);
            return;
        }
        status |= kThreadTasksClaimed;

        // Only the four anchors may acquire before the gate opens.  They use
        // distinct slots, fill all 16 payload lines, validate both guards and
        // hold the slots until all four have staged concurrently.
        U1FatalReason gate_failure = U1FatalReason::None;
        if (task_id >= kAnchorTaskCount &&
            !SimtWaitForAnchorGate(anchor_staged, anchor_mask, fatal, timeout_ticks, &gate_failure)) {
            SimtResetUnpublishedTask(task, building, fatal, task_id);
            if (gate_failure != U1FatalReason::None) {
                SimtPublishFatal(fatal, gate_failure, task_id);
            }
            return;
        }

        __gm__ uint64_t *slot_state = slot_state_words + slot_id * kAtomicStrideWords;
        __gm__ uint64_t *slot_acquires = slot_acquire_words + slot_id * kAtomicStrideWords;
        __gm__ uint64_t *slot_releases = slot_release_words + slot_id * kAtomicStrideWords;
        uint32_t generation = 0U;
        U1FatalReason acquire_failure = U1FatalReason::None;
        if (!SimtAcquireSlot(
                slot_state, slot_acquires, slot_releases, global_busy, global_max_busy, fatal, slot_id, task_id,
                timeout_ticks, &generation, &slot_attempts, &acquire_failure
            )) {
            SimtResetUnpublishedTask(task, building, fatal, task_id);
            if (acquire_failure != U1FatalReason::None) {
                SimtPublishFatal(fatal, acquire_failure, task_id);
            }
            return;
        }
        if (task_id < kAnchorTaskCount && generation != 0U) {
            (void)SimtReleaseSlot(slot_state, slot_releases, global_busy, fatal, generation, task_id);
            SimtResetUnpublishedTask(task, building, fatal, task_id);
            SimtPublishFatal(fatal, U1FatalReason::SlotInvariant, task_id);
            return;
        }
        status |= kThreadSlotsAcquired;

        __ubuf__ volatile uint64_t *slot_region = staging_region + slot_id * kUbufSlotStrideWords;
        __ubuf__ volatile uint64_t *staging_payload = slot_region + kUbufPayloadOffsetWords;
        __ubuf__ volatile uint64_t *guard_after = slot_region + kUbufGuardAfterOffsetWords;
        const uint32_t guard_before_id = kGuardBeforeStagingSlots + 2U * slot_id;
        const uint32_t guard_after_id = guard_before_id + 1U;
        for (uint32_t word = 0U; word < kWordsPerLine; ++word) {
            slot_region[word] = SimtExpectedUbufGuardWord(nonce, guard_before_id, word);
            guard_after[word] = SimtExpectedUbufGuardWord(nonce, guard_after_id, word);
        }
        for (uint32_t word = 0U; word < payload_words; ++word) {
            staging_payload[word] = SimtPayloadWord(nonce, task_id, word);
        }
        status |= kThreadPayloadsComplete;

        bool ubuf_guards_valid = true;
        for (uint32_t word = 0U; word < kWordsPerLine; ++word) {
            ubuf_guards_valid = ubuf_guards_valid &&
                                slot_region[word] == SimtExpectedUbufGuardWord(nonce, guard_before_id, word) &&
                                guard_after[word] == SimtExpectedUbufGuardWord(nonce, guard_after_id, word);
        }
        if (!ubuf_guards_valid) {
            (void)SimtReleaseSlot(slot_state, slot_releases, global_busy, fatal, generation, task_id);
            SimtResetUnpublishedTask(task, building, fatal, task_id);
            SimtPublishFatal(fatal, U1FatalReason::UbufGuardCorruption, task_id);
            return;
        }
        (void)asc_atomic_add(ubuf_guard_checks, static_cast<uint64_t>(1U));
        status |= kThreadUbufGuardsValid;

        if (task_id < kAnchorTaskCount) {
            U1FatalReason anchor_failure = U1FatalReason::None;
            if (!SimtStageAnchorIdentity(anchor_mask, task_id, &anchor_failure)) {
                (void)SimtReleaseSlot(slot_state, slot_releases, global_busy, fatal, generation, task_id);
                SimtResetUnpublishedTask(task, building, fatal, task_id);
                SimtPublishFatal(fatal, anchor_failure, task_id);
                return;
            }
            const uint64_t prior_staged = asc_atomic_add(anchor_staged, static_cast<uint64_t>(1U));
            bool anchor_gate_open = false;
            if (prior_staged >= kAnchorTaskCount) {
                anchor_failure = U1FatalReason::SlotInvariant;
            } else {
                anchor_gate_open =
                    SimtWaitForAnchorGate(anchor_staged, anchor_mask, fatal, timeout_ticks, &anchor_failure);
            }
            if (!anchor_gate_open) {
                (void)SimtReleaseSlot(slot_state, slot_releases, global_busy, fatal, generation, task_id);
                SimtResetUnpublishedTask(task, building, fatal, task_id);
                if (anchor_failure != U1FatalReason::None) {
                    SimtPublishFatal(fatal, anchor_failure, task_id);
                }
                return;
            }
        }

        if (asc_atomic_add(fatal, static_cast<uint64_t>(0U)) != 0U) {
            (void)SimtReleaseSlot(slot_state, slot_releases, global_busy, fatal, generation, task_id);
            SimtResetUnpublishedTask(task, building, fatal, task_id);
            return;
        }

        // Diagnostic transport: the same SIMT leader reads volatile UBUF and
        // performs ordinary GM stores.  There is deliberately no MTE3 path.
        __gm__ uint64_t *payload = task + kPayloadOffsetWords;
        uint64_t checksum = 0x243F6A8885A308D3ULL ^ static_cast<uint64_t>(task_id);
        for (uint32_t word = 0U; word < payload_words; ++word) {
            const uint64_t value = staging_payload[word];
            payload[word] = value;
            checksum = SimtFoldChecksum(checksum, value, word);
        }
        status |= kThreadPayloadsComplete;
        asc_threadfence();

        // A foreign role may publish the first fatal while this leader is
        // copying its complete payload to GM.  Recheck at the publication
        // boundary: an unpublished holder must exact-release/reset instead
        // of turning a failed launch's BUILDING task into BUILT.
        if (asc_atomic_add(fatal, static_cast<uint64_t>(0U)) != 0U) {
            (void)SimtReleaseSlot(slot_state, slot_releases, global_busy, fatal, generation, task_id);
            SimtResetUnpublishedTask(task, building, fatal, task_id);
            return;
        }

        if (asc_atomic_cas(task, building, built) != building) {
            (void)SimtReleaseSlot(slot_state, slot_releases, global_busy, fatal, generation, task_id);
            SimtResetUnpublishedTask(task, building, fatal, task_id);
            SimtPublishFatal(fatal, U1FatalReason::TaskPublishConflict, task_id);
            return;
        }
        (void)asc_atomic_add(built_count, static_cast<uint64_t>(1U));
        status |= kThreadTasksPublished;

        if (!SimtReleaseSlot(slot_state, slot_releases, global_busy, fatal, generation, task_id)) {
            return;
        }
        status |= kThreadSlotsReleased;

        __gm__ uint64_t *build_report = task + kBuildReportOffsetWords;
        build_report[0] = static_cast<uint64_t>(task_id) | (static_cast<uint64_t>(thread) << 32U);
        build_report[1] = static_cast<uint64_t>(warp) | (static_cast<uint64_t>(lane) << 32U);
        build_report[2] =
            static_cast<uint64_t>(kExpectedBuildPhaseBits) | (static_cast<uint64_t>(payload_lines) << 32U);
        build_report[3] = static_cast<uint64_t>(payload_words) | (static_cast<uint64_t>(payload_words) << 32U);
        build_report[4] = static_cast<uint64_t>(1U) | (static_cast<uint64_t>(1U) << 32U);
        build_report[5] = static_cast<uint64_t>(slot_id) | (static_cast<uint64_t>(generation) << 32U);
        build_report[6] = nonce;
        build_report[7] = checksum;
        ++build_count;
    }

    if (build_count != kTasksPerWarp) {
        SimtPublishFatal(fatal, U1FatalReason::BuildAborted, first_task);
        return;
    }
    __gm__ uint64_t *thread_report = thread_report_words + thread * kThreadReportStrideWords;
    thread_report[0] = static_cast<uint64_t>(thread) | (static_cast<uint64_t>(warp) << 32U);
    thread_report[1] = static_cast<uint64_t>(lane) | (static_cast<uint64_t>(1U) << 32U);
    thread_report[2] = static_cast<uint64_t>(first_task) | (static_cast<uint64_t>(last_task) << 32U);
    thread_report[3] = static_cast<uint64_t>(build_count) | (static_cast<uint64_t>(status) << 32U);
    thread_report[4] = static_cast<uint64_t>(slot_attempts) | (static_cast<uint64_t>(build_count) << 32U);
    thread_report[5] = nonce;
    thread_report[6] = SimtThreadChecksum(nonce, thread, first_task, last_task, status);
    thread_report[7] = 0U;
    asc_threadfence();
}

// The ordinary AIV1 executor is Scalar.  This unreachable SIMD companion
// preserves mixed SIMD_SIMT metadata without participating in the protocol.
static __simd_vf__ __aicore__ void U1SimdMetadataAnchor(__ubuf__ uint32_t *scratch) { scratch[0] = scratch[0] + 1U; }

__aicore__ void RunBuilder(__gm__ U1ProbeState *state) {
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    const uint32_t subblock = static_cast<uint32_t>(get_subblockid());
    uint32_t timeout_count = 0U;
    if (!ConfigValid(state)) {
        PublishFatal(state, U1FatalReason::InvalidConfig, kBuilderOwner, kInvalidTaskId);
    } else {
        __ubuf__ volatile uint64_t *staging_region = reinterpret_cast<__ubuf__ volatile uint64_t *>(kUbufRegionOffset);
        // Keep the SIMD companion in mixed-function metadata without giving
        // malformed control a path to UBUF.  UINT64_MAX is a valid but
        // reserved diagnostic nonce; this one-word touch completes before the
        // SIMT launch and is overwritten by slot0's guard initialization.
        if (state->control.launch_nonce == UINT64_MAX) {
            U1SimdMetadataAnchor(reinterpret_cast<__ubuf__ uint32_t *>(kUbufRegionOffset));
        }
        cce::async_invoke<U1SimtBuildMultiSlot>(
            cce::dim3{kThreadCount, 1U, 1U}, reinterpret_cast<__gm__ uint64_t *>(&state->tasks[0]),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->slot_states[0].value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->slot_acquire_count[0].value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->slot_release_count[0].value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->global_busy_depth.value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->global_max_busy_depth.value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->anchor_staged_count.value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->anchor_staged_mask.value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->ubuf_guard_check_count.value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->built_count.value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->fatal.value)),
            reinterpret_cast<__gm__ uint64_t *>(&state->builder_threads[0]), staging_region,
            state->control.launch_nonce, state->control.timeout_ticks
        );
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        bool drain_timed_out = false;
        (void)WaitForTerminal(state, kBuilderOwner, &drain_timed_out);
        if (drain_timed_out) {
            ++timeout_count;
        }
    }
    PublishRole(
        &state->roles[static_cast<uint32_t>(ProbeRole::Aiv0Builder)], kBuilderOwner, ProbeRole::Aiv0Builder, block,
        subblock, 0U, 0U, timeout_count, state->control.launch_nonce
    );
}

__aicore__ void RunExecutor(__gm__ U1ProbeState *state) {
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    const uint32_t subblock = static_cast<uint32_t>(get_subblockid());
    uint32_t claims = 0U;
    uint32_t finishes = 0U;
    uint32_t timeout_count = 0U;
    if (!ConfigValid(state)) {
        PublishFatal(state, U1FatalReason::InvalidConfig, kExecutorOwner, kInvalidTaskId);
    } else {
        for (uint32_t task_id = 0U; task_id < kTaskCount; ++task_id) {
            const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
            const uint64_t built = BuiltState(task_id);
            const uint64_t claimed = ClaimedState(task_id);
            uint64_t observed = 0U;
            uint32_t polls = 0U;
            while (true) {
                if (AtomicLoad(&state->fatal.value) != 0U) {
                    break;
                }
                observed = AtomicCas(&state->tasks[task_id].control.state, built, claimed);
                if (observed == built) {
                    break;
                }
                if (observed != EmptyState() && observed != BuildingState(task_id)) {
                    PublishFatal(state, U1FatalReason::InvalidTaskState, kExecutorOwner, task_id);
                    break;
                }
                if (AtomicLoad(&state->fatal.value) != 0U) {
                    break;
                }
                ++polls;
                if ((polls & kWatchdogMask) == 0U &&
                    static_cast<uint64_t>(get_sys_cnt()) - begin > state->control.timeout_ticks) {
                    ++timeout_count;
                    PublishFatal(state, U1FatalReason::TaskWaitTimeout, kExecutorOwner, task_id);
                    break;
                }
            }
            if (observed != built) {
                break;
            }
            // Close the race between the pre-claim fatal load and the
            // BUILT->CLAIMED CAS.  This executor owns the exact CLAIMED value,
            // so it can restore BUILT without publishing any execution work.
            if (AtomicLoad(&state->fatal.value) != 0U) {
                if (AtomicCas(&state->tasks[task_id].control.state, claimed, built) != claimed) {
                    PublishFatal(state, U1FatalReason::TaskClaimConflict, kExecutorOwner, task_id);
                }
                break;
            }
            ++claims;

            const uint32_t payload_lines = PayloadLinesForTask(task_id);
            const uint32_t payload_words = PayloadWordsForTask(task_id);
            __gm__ uint8_t *payload_bytes = reinterpret_cast<__gm__ uint8_t *>(&state->tasks[task_id].payload);
            for (uint32_t line = 0U; line < payload_lines; ++line) {
                dcci(static_cast<__gm__ void *>(payload_bytes + line * kCacheLineBytes), kSingleCacheLine);
            }
            dsb(DSB_ALL);

            uint64_t checksum = 0x243F6A8885A308D3ULL ^ static_cast<uint64_t>(task_id);
            bool payload_valid = true;
            for (uint32_t word = 0U; word < payload_words; ++word) {
                const uint64_t value = state->tasks[task_id].payload.words[word];
                payload_valid =
                    payload_valid && value == ExpectedPayloadWord(state->control.launch_nonce, task_id, word);
                checksum = FoldChecksum(checksum, value, word);
            }

            if (!payload_valid) {
                PublishExecReport(
                    &state->tasks[task_id].exec_report, task_id, block, subblock, kExecClaimed | kExecPayloadRead,
                    payload_lines, payload_words, false, state->control.launch_nonce, checksum
                );
                PublishFatal(state, U1FatalReason::PayloadMismatch, kExecutorOwner, task_id);
                break;
            }
            // A foreign failure may arrive during DCCI/payload validation.
            // Do not turn the locally claimed task into DONE after observing
            // it; restore the exact claim and leave the first fatal intact.
            if (AtomicLoad(&state->fatal.value) != 0U) {
                if (AtomicCas(&state->tasks[task_id].control.state, claimed, built) != claimed) {
                    PublishFatal(state, U1FatalReason::TaskCompleteConflict, kExecutorOwner, task_id);
                }
                break;
            }
            if (AtomicCas(&state->tasks[task_id].control.state, claimed, DoneState(task_id)) != claimed) {
                PublishFatal(state, U1FatalReason::TaskCompleteConflict, kExecutorOwner, task_id);
                break;
            }
            ++finishes;
            (void)AtomicFetchAdd(&state->done_count.value, 1U);
            PublishExecReport(
                &state->tasks[task_id].exec_report, task_id, block, subblock, kExpectedExecPhaseBits, payload_lines,
                payload_words, true, state->control.launch_nonce, checksum
            );
        }
        bool drain_timed_out = false;
        (void)WaitForTerminal(state, kExecutorOwner, &drain_timed_out);
        if (drain_timed_out) {
            ++timeout_count;
        }
    }
    PublishRole(
        &state->roles[static_cast<uint32_t>(ProbeRole::Aiv1Executor)], kExecutorOwner, ProbeRole::Aiv1Executor, block,
        subblock, claims, finishes, timeout_count, state->control.launch_nonce
    );
}

#endif  // defined(__DAV_VEC__)

}  // namespace

#if defined(__DAV_VEC__)

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_cross_core_u1_0_mix_aiv, 1, 2);

extern "C" __global__ __aicore__ void
simt_cross_core_u1_0_mix_aiv(__gm__ pa_scheduler::simt_cross_core::u1::U1ProbeState *state) {
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dsb(DSB_ALL);
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    const uint32_t subblock_dim = static_cast<uint32_t>(get_subblockdim());
    const uint32_t subblock = static_cast<uint32_t>(get_subblockid());
    if (block != 0U || subblock_dim != 2U || subblock >= subblock_dim) {
        const uint32_t reporter_owner = subblock == 1U ? kExecutorOwner : kBuilderOwner;
        PublishFatal(state, U1FatalReason::InvalidTopology, reporter_owner, kInvalidTaskId);
        return;
    }
    if (subblock == 0U) {
        RunBuilder(state);
    } else {
        RunExecutor(state);
    }
}

#else

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_cross_core_u1_0_mix_aic, 1, 2);

extern "C" __global__ __aicore__ void
simt_cross_core_u1_0_mix_aic(__gm__ pa_scheduler::simt_cross_core::u1::U1ProbeState *state) {
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dsb(DSB_ALL);
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    uint32_t timeout_count = 0U;
    if (block != 0U || !ConfigValid(state)) {
        PublishFatal(
            state, block == 0U ? U1FatalReason::InvalidConfig : U1FatalReason::InvalidTopology, 0U, kInvalidTaskId
        );
    } else {
        bool drain_timed_out = false;
        (void)WaitForTerminal(state, 0U, &drain_timed_out);
        if (drain_timed_out) {
            ++timeout_count;
        }
    }
    PublishRole(
        &state->roles[static_cast<uint32_t>(pa_scheduler::simt_cross_core::u1::ProbeRole::AicObserver)], 0U,
        pa_scheduler::simt_cross_core::u1::ProbeRole::AicObserver, block, 0U, 0U, 0U, timeout_count,
        state->control.launch_nonce
    );
}

#endif
