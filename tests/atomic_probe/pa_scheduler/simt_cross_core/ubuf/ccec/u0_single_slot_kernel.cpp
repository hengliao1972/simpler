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

// U0 deliberately diagnoses a SIMT UBUF path without MTE3: AIV0's 64 warp
// leaders serialize through one private UBUF slot, read their words back from
// UBUF and store them to GM directly.  The transport marker and mte3_count==0
// prevent this probe from being mistaken for an MTE3 implementation.

#include <pto/common/kernel_meta.hpp>

#include "cce_aicore_intrinsics.h"

#if defined(__DAV_VEC__)
#include "simt_api/asc_simt.h"
#endif

#include "../common/u0_single_slot.h"

namespace {

using namespace pa_scheduler::simt_cross_core::u0;
namespace g0 = pa_scheduler::simt_cross_core::g0;

constexpr int kSingleCacheLine = 0;
constexpr uint32_t kWatchdogMask = 0x3FFU;
constexpr uintptr_t kUbufSingleSlotOffset = 0U;

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
PublishFatal(__gm__ U0ProbeState *state, U0FatalReason reason, uint32_t owner, uint32_t task_id) {
    (void)AtomicCas(&state->fatal.value, 0U, EncodeU0Fatal(reason, owner, task_id));
}

__aicore__ __attribute__((always_inline)) inline bool ConfigValid(__gm__ const U0ProbeState *state) {
    return state->control.magic == kProbeMagic && state->control.version == kProbeVersion &&
           state->control.launch_nonce != 0U && state->control.timeout_ticks != 0U &&
           state->control.thread_count == kThreadCount && state->control.warp_count == kWarpCount &&
           state->control.task_count == kTaskCount && state->control.role_count == kRoleCount &&
           state->control.payload_class_count == kPayloadClassCount &&
           state->control.max_payload_lines == kMaxPayloadLines && state->control.words_per_line == kWordsPerLine &&
           state->control.ubuf_alignment_bytes == kUbufAlignmentBytes &&
           state->control.ubuf_region_bytes == kUbufRegionBytes &&
           state->control.ubuf_payload_offset_bytes == kUbufPayloadOffsetBytes &&
           state->control.transport_kind == TransportKind::SimtUbufReadToGmWordStore &&
           state->control.ubuf_slot_count == kUbufSlotCount;
}

__aicore__ __attribute__((always_inline)) inline void PublishRole(
    __gm__ U0RoleReport *destination, uint32_t owner, ProbeRole role, uint32_t physical_block, uint32_t subblock_id,
    uint32_t task_claim_count, uint32_t task_finish_count, uint32_t timeout_count, uint64_t launch_nonce
) {
    __gm__ uint64_t *words = reinterpret_cast<__gm__ uint64_t *>(destination);
    StoreDev64(words + 0U, static_cast<uint64_t>(owner) | (static_cast<uint64_t>(role) << 32U));
    StoreDev64(words + 1U, static_cast<uint64_t>(physical_block) | (static_cast<uint64_t>(subblock_id) << 32U));
    // Main Scalar build action count is frozen at zero for every role.  AIV1's
    // legal executor work is reported in the independent claim/finish fields.
    StoreDev64(words + 2U, static_cast<uint64_t>(0U) | (static_cast<uint64_t>(task_claim_count) << 32U));
    StoreDev64(words + 3U, static_cast<uint64_t>(task_finish_count) | (static_cast<uint64_t>(timeout_count) << 32U));
    StoreDev64(words + 4U, launch_nonce);
    StoreDev64(words + 5U, kResultMagic);
    StoreDev64(words + 6U, 0U);
    StoreDev64(words + 7U, 0U);
    dsb(DSB_ALL);
}

__aicore__ __attribute__((always_inline)) inline void PublishExecReport(
    __gm__ U0ExecReport *destination, uint32_t task_id, uint32_t physical_block, uint32_t subblock_id,
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

__aicore__ bool WaitForTerminal(__gm__ U0ProbeState *state, uint32_t owner, bool *timed_out) {
    *timed_out = false;
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint32_t polls = 0U;
    while (true) {
        if (AtomicLoad(&state->done_count.value) == kTaskCount &&
            AtomicLoad(&state->builder_finished_count.value) == kTaskCount &&
            AtomicLoad(&state->executor_finished_count.value) == 1U) {
            return true;
        }
        if (AtomicLoad(&state->fatal.value) != 0U) {
            return false;
        }
        ++polls;
        if ((polls & kWatchdogMask) == 0U &&
            static_cast<uint64_t>(get_sys_cnt()) - begin > state->control.timeout_ticks) {
            *timed_out = true;
            PublishFatal(state, U0FatalReason::DrainTimeout, owner, kInvalidTaskId);
            return false;
        }
    }
}

#if defined(__DAV_VEC__)

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtRotateLeft(uint64_t value, uint32_t shift) {
    return (value << shift) | (value >> (64U - shift));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint32_t SimtPayloadLines(uint32_t task_id) {
    const uint32_t payload_class = task_id & 3U;
    return payload_class == 0U ? 1U : (payload_class == 1U ? 10U : (payload_class == 2U ? 16U : 68U));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtPayloadWord(uint64_t nonce, uint32_t task_id, uint32_t word) {
    const uint32_t shift = ((task_id * 7U + word) % 31U) + 1U;
    return 0x5530000000000000ULL ^ SimtRotateLeft(nonce ^ (0x0101010101010101ULL * (word + 1U)), shift) ^
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
SimtFatalValue(U0FatalReason reason, uint32_t task_id) {
    return (static_cast<uint64_t>(reason) << g0::kFatalReasonShift) |
           (static_cast<uint64_t>(kBuilderOwner) << g0::kFatalOwnerShift) |
           (static_cast<uint64_t>(task_id) << g0::kFatalTaskIdShift);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void
SimtPublishFatal(__gm__ uint64_t *fatal, U0FatalReason reason, uint32_t task_id) {
    (void)asc_atomic_cas(fatal, static_cast<uint64_t>(0U), SimtFatalValue(reason, task_id));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtExpectedUbufGuardWord(uint64_t nonce, uint32_t guard_id, uint32_t word) {
    uint64_t value = kGuardMagic ^ nonce ^ (static_cast<uint64_t>(guard_id) << 32U) ^ word;
    value ^= value >> 27U;
    value *= 0xD6E8FEB86659FD93ULL;
    return value ^ (value >> 31U);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtThreadChecksum(uint64_t nonce, uint32_t thread_id, uint32_t task_id, uint32_t status) {
    uint64_t checksum = nonce ^ (static_cast<uint64_t>(thread_id) << 32U) ^ task_id;
    checksum = SimtFoldChecksum(checksum, status, 0U);
    return SimtFoldChecksum(checksum, kTransportMarker, 1U);
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(kThreadCount) void U0SimtBuildSingleSlot(
    __gm__ uint64_t *task_words, __gm__ uint64_t *slot_owner, __gm__ uint64_t *slot_busy,
    __gm__ uint64_t *slot_max_busy, __gm__ uint64_t *slot_acquires, __gm__ uint64_t *slot_releases,
    __gm__ uint64_t *build_claims, __gm__ uint64_t *built_count, __gm__ uint64_t *ubuf_guard_checks,
    __gm__ uint64_t *builder_finished, __gm__ uint64_t *fatal, __gm__ uint64_t *thread_report_words,
    __ubuf__ volatile uint64_t *staging_region, uint64_t nonce, uint64_t timeout_ticks
) {
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    const uint32_t warp = thread / kWarpSize;
    const uint32_t lane = thread % kWarpSize;
    if (thread >= kThreadCount || warp >= kWarpCount || lane != 0U) {
        return;
    }

    constexpr uint32_t kTaskStrideWords = sizeof(U0Task) / sizeof(uint64_t);
    constexpr uint32_t kPayloadOffsetWords = offsetof(U0Task, payload) / sizeof(uint64_t);
    constexpr uint32_t kBuildReportOffsetWords = offsetof(U0Task, build_report) / sizeof(uint64_t);
    constexpr uint32_t kThreadReportStrideWords = sizeof(U0ThreadReport) / sizeof(uint64_t);
    const uint32_t task_id = warp;
    const uint32_t payload_lines = SimtPayloadLines(task_id);
    const uint32_t payload_words = payload_lines * kWordsPerLine;
    const uint64_t building = SimtBuildingState(task_id);
    const uint64_t built = SimtBuiltState(task_id, payload_lines);
    __gm__ uint64_t *task = task_words + task_id * kTaskStrideWords;
    __gm__ uint64_t *payload = task + kPayloadOffsetWords;
    __gm__ uint64_t *build_report = task + kBuildReportOffsetWords;
    __gm__ uint64_t *thread_report = thread_report_words + thread * kThreadReportStrideWords;
    uint32_t status = kThreadActiveLeader;
    uint32_t slot_attempts = 0U;

    if (asc_atomic_cas(task, static_cast<uint64_t>(0U), building) != 0U) {
        SimtPublishFatal(fatal, U0FatalReason::TaskClaimConflict, task_id);
        return;
    }
    (void)asc_atomic_add(build_claims, static_cast<uint64_t>(1U));
    status |= kThreadTaskClaimed;

    const uint64_t slot_value = static_cast<uint64_t>(warp) + 1U;
    const uint64_t begin = clock();
    while (true) {
        ++slot_attempts;
        if (asc_atomic_cas(slot_owner, kSlotFree, slot_value) == kSlotFree) {
            break;
        }
        if ((slot_attempts & kWatchdogMask) == 0U) {
            if (asc_atomic_add(fatal, static_cast<uint64_t>(0U)) != 0U) {
                return;
            }
            if (clock() - begin > timeout_ticks) {
                SimtPublishFatal(fatal, U0FatalReason::SlotTimeout, task_id);
                return;
            }
        }
    }
    status |= kThreadSlotAcquired;
    const uint64_t slot_ticket = asc_atomic_add(slot_acquires, static_cast<uint64_t>(1U)) + 1U;
    if (asc_atomic_add(slot_busy, static_cast<uint64_t>(1U)) != 0U) {
        SimtPublishFatal(fatal, U0FatalReason::SlotInvariant, task_id);
        return;
    }
    const uint64_t prior_max = asc_atomic_cas(slot_max_busy, static_cast<uint64_t>(0U), static_cast<uint64_t>(1U));
    if (prior_max != 0U && prior_max != 1U) {
        SimtPublishFatal(fatal, U0FatalReason::SlotInvariant, task_id);
        return;
    }

    __ubuf__ volatile uint64_t *staging_payload = staging_region + kUbufPayloadOffsetWords;
    __ubuf__ volatile uint64_t *guard_after = staging_region + kUbufGuardAfterOffsetWords;
    for (uint32_t word = 0U; word < kWordsPerLine; ++word) {
        staging_region[word] = SimtExpectedUbufGuardWord(nonce, kGuardBeforeStagingSlot, word);
        guard_after[word] = SimtExpectedUbufGuardWord(nonce, kGuardAfterStagingSlot, word);
    }
    for (uint32_t word = 0U; word < payload_words; ++word) {
        staging_payload[word] = SimtPayloadWord(nonce, task_id, word);
    }
    status |= kThreadPayloadComplete;

    bool ubuf_guards_valid = true;
    for (uint32_t word = 0U; word < kWordsPerLine; ++word) {
        ubuf_guards_valid = ubuf_guards_valid &&
                            staging_region[word] == SimtExpectedUbufGuardWord(nonce, kGuardBeforeStagingSlot, word) &&
                            guard_after[word] == SimtExpectedUbufGuardWord(nonce, kGuardAfterStagingSlot, word);
    }
    if (!ubuf_guards_valid) {
        SimtPublishFatal(fatal, U0FatalReason::UbufGuardCorruption, task_id);
        if (asc_atomic_add(slot_busy, UINT64_MAX) == 1U &&
            asc_atomic_cas(slot_owner, slot_value, kSlotFree) == slot_value) {
            (void)asc_atomic_add(slot_releases, static_cast<uint64_t>(1U));
        }
        return;
    }
    (void)asc_atomic_add(ubuf_guard_checks, static_cast<uint64_t>(1U));
    status |= kThreadUbufGuardsValid;

    // Diagnostic transport: same SIMT leader reads UBUF and performs ordinary
    // GM word stores.  There is deliberately no MTE3/UBTOOUT instruction.
    uint64_t checksum = 0x243F6A8885A308D3ULL ^ static_cast<uint64_t>(task_id);
    for (uint32_t word = 0U; word < payload_words; ++word) {
        const uint64_t value = staging_payload[word];
        payload[word] = value;
        checksum = SimtFoldChecksum(checksum, value, word);
    }
    asc_threadfence();

    if (asc_atomic_cas(task, building, built) != building) {
        SimtPublishFatal(fatal, U0FatalReason::TaskPublishConflict, task_id);
        return;
    }
    (void)asc_atomic_add(built_count, static_cast<uint64_t>(1U));
    status |= kThreadTaskPublished;

    // Drop busy before exposing FREE so a succeeding leader can never observe
    // slot_owner==FREE while busy_depth still belongs to this owner.
    if (asc_atomic_add(slot_busy, UINT64_MAX) != 1U ||
        asc_atomic_cas(slot_owner, slot_value, kSlotFree) != slot_value) {
        SimtPublishFatal(fatal, U0FatalReason::SlotInvariant, task_id);
        return;
    }
    (void)asc_atomic_add(slot_releases, static_cast<uint64_t>(1U));
    status |= kThreadSlotReleased;

    build_report[0] = static_cast<uint64_t>(task_id) | (static_cast<uint64_t>(thread) << 32U);
    build_report[1] = static_cast<uint64_t>(warp) | (static_cast<uint64_t>(lane) << 32U);
    build_report[2] = static_cast<uint64_t>(kExpectedBuildPhaseBits) | (static_cast<uint64_t>(payload_lines) << 32U);
    build_report[3] = static_cast<uint64_t>(payload_words) | (static_cast<uint64_t>(payload_words) << 32U);
    build_report[4] = static_cast<uint64_t>(1U) | (static_cast<uint64_t>(1U) << 32U);
    build_report[5] = static_cast<uint64_t>(1U) | (slot_ticket << 32U);
    build_report[6] = nonce;
    build_report[7] = checksum;

    thread_report[0] = static_cast<uint64_t>(thread) | (static_cast<uint64_t>(warp) << 32U);
    thread_report[1] = static_cast<uint64_t>(lane) | (static_cast<uint64_t>(1U) << 32U);
    thread_report[2] = static_cast<uint64_t>(task_id) | (static_cast<uint64_t>(status) << 32U);
    thread_report[3] = static_cast<uint64_t>(1U) | (static_cast<uint64_t>(slot_attempts) << 32U);
    thread_report[4] = nonce;
    thread_report[5] = SimtThreadChecksum(nonce, thread, task_id, status);
    thread_report[6] = 0U;
    thread_report[7] = 0U;
    asc_threadfence();

    if (asc_atomic_add(builder_finished, static_cast<uint64_t>(1U)) >= kTaskCount) {
        SimtPublishFatal(fatal, U0FatalReason::SlotInvariant, task_id);
    }
}

// U0's ordinary AIV1 executor is Scalar, so it does not itself create SIMD
// metadata.  Retain the same unreachable companion used by the earlier mixed
// probes: this prevents a shared 1:2 entry from being classified SIMT_VF_ONLY.
// The normal version is finite, therefore it never equals UINT64_MAX and this
// function never touches the UBUF slot.
static __simd_vf__ __aicore__ void U0SimdMetadataAnchor(__ubuf__ uint32_t *scratch) { scratch[0] = scratch[0] + 1U; }

__aicore__ void RunBuilder(__gm__ U0ProbeState *state) {
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    const uint32_t subblock = static_cast<uint32_t>(get_subblockid());
    uint32_t timeout_count = 0U;
    if (!ConfigValid(state)) {
        PublishFatal(state, U0FatalReason::InvalidConfig, kBuilderOwner, kInvalidTaskId);
    } else {
        __ubuf__ volatile uint64_t *staging_region =
            reinterpret_cast<__ubuf__ volatile uint64_t *>(kUbufSingleSlotOffset);
        cce::async_invoke<U0SimtBuildSingleSlot>(
            cce::dim3{kThreadCount, 1U, 1U}, reinterpret_cast<__gm__ uint64_t *>(&state->tasks[0]),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->slot_owner.value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->slot_busy_depth.value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->slot_max_busy_depth.value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->slot_acquire_count.value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->slot_release_count.value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->build_claim_count.value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->built_count.value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->ubuf_guard_check_count.value)),
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->builder_finished_count.value)),
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

__aicore__ void RunExecutor(__gm__ U0ProbeState *state) {
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    const uint32_t subblock = static_cast<uint32_t>(get_subblockid());
    uint32_t claims = 0U;
    uint32_t finishes = 0U;
    uint32_t timeout_count = 0U;
    if (!ConfigValid(state)) {
        PublishFatal(state, U0FatalReason::InvalidConfig, kExecutorOwner, kInvalidTaskId);
    } else {
        for (uint32_t task_id = 0U; task_id < kTaskCount; ++task_id) {
            const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
            const uint64_t built = BuiltState(task_id);
            const uint64_t claimed = ClaimedState(task_id);
            uint64_t observed = 0U;
            uint32_t polls = 0U;
            while (true) {
                observed = AtomicCas(&state->tasks[task_id].control.state, built, claimed);
                if (observed == built) {
                    break;
                }
                if (observed != EmptyState() && observed != BuildingState(task_id)) {
                    PublishFatal(state, U0FatalReason::InvalidTaskState, kExecutorOwner, task_id);
                    break;
                }
                if (AtomicLoad(&state->fatal.value) != 0U) {
                    break;
                }
                ++polls;
                if ((polls & kWatchdogMask) == 0U &&
                    static_cast<uint64_t>(get_sys_cnt()) - begin > state->control.timeout_ticks) {
                    ++timeout_count;
                    PublishFatal(state, U0FatalReason::TaskWaitTimeout, kExecutorOwner, task_id);
                    break;
                }
            }
            if (observed != built) {
                break;
            }
            ++claims;
            (void)AtomicFetchAdd(&state->exec_claim_count.value, 1U);

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
                PublishFatal(state, U0FatalReason::PayloadMismatch, kExecutorOwner, task_id);
                break;
            }
            if (AtomicCas(&state->tasks[task_id].control.state, claimed, DoneState(task_id)) != claimed) {
                PublishFatal(state, U0FatalReason::TaskCompleteConflict, kExecutorOwner, task_id);
                break;
            }
            ++finishes;
            (void)AtomicFetchAdd(&state->done_count.value, 1U);
            PublishExecReport(
                &state->tasks[task_id].exec_report, task_id, block, subblock, kExpectedExecPhaseBits, payload_lines,
                payload_words, true, state->control.launch_nonce, checksum
            );
        }
        if (finishes == kTaskCount) {
            if (AtomicFetchAdd(&state->executor_finished_count.value, 1U) != 0U) {
                PublishFatal(state, U0FatalReason::TaskCompleteConflict, kExecutorOwner, kInvalidTaskId);
            }
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

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_cross_core_u0_0_mix_aiv, 1, 2);

extern "C" __global__ __aicore__ void
simt_cross_core_u0_0_mix_aiv(__gm__ pa_scheduler::simt_cross_core::u0::U0ProbeState *state) {
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dsb(DSB_ALL);
    if (state->control.version == UINT64_MAX) {
        U0SimdMetadataAnchor(reinterpret_cast<__ubuf__ uint32_t *>(kUbufSingleSlotOffset));
    }
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    const uint32_t subblock_dim = static_cast<uint32_t>(get_subblockdim());
    const uint32_t subblock = static_cast<uint32_t>(get_subblockid());
    if (block != 0U || subblock_dim != 2U || subblock >= subblock_dim) {
        const uint32_t reporter_owner = subblock == 1U ? kExecutorOwner : kBuilderOwner;
        PublishFatal(state, U0FatalReason::InvalidTopology, reporter_owner, kInvalidTaskId);
        return;
    }
    if (subblock == 0U) {
        RunBuilder(state);
    } else {
        RunExecutor(state);
    }
}

#else

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_cross_core_u0_0_mix_aic, 1, 2);

extern "C" __global__ __aicore__ void
simt_cross_core_u0_0_mix_aic(__gm__ pa_scheduler::simt_cross_core::u0::U0ProbeState *state) {
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dsb(DSB_ALL);
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    uint32_t timeout_count = 0U;
    if (block != 0U || !ConfigValid(state)) {
        PublishFatal(
            state, block == 0U ? U0FatalReason::InvalidConfig : U0FatalReason::InvalidTopology, 0U, kInvalidTaskId
        );
    } else {
        bool drain_timed_out = false;
        (void)WaitForTerminal(state, 0U, &drain_timed_out);
        if (drain_timed_out) {
            ++timeout_count;
        }
    }
    PublishRole(
        &state->roles[static_cast<uint32_t>(pa_scheduler::simt_cross_core::u0::ProbeRole::AicObserver)], 0U,
        pa_scheduler::simt_cross_core::u0::ProbeRole::AicObserver, block, 0U, 0U, 0U, timeout_count,
        state->control.launch_nonce
    );
}

#endif
