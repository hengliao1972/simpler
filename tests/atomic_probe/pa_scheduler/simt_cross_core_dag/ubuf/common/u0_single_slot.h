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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_U0_SINGLE_SLOT_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_U0_SINGLE_SLOT_H

#include <stddef.h>
#include <stdint.h>

#include "../../common/full_pa_exec_protocol.h"

namespace pa_scheduler::simt_cross_core::u0 {

#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_U0_INLINE __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_U0_INLINE constexpr
#endif

using g0::AtomicLine;
using g0::DecodeExecState;
using g0::EncodeExecState;
using g0::ExecEngineClass;
using g0::ExecPhase;
using g0::kCacheLineBytes;
using g0::kUnboundOwner;

constexpr uint64_t kProbeMagic = 0x53494D5455304135ULL;
constexpr uint64_t kProbeVersion = 2U;
constexpr uint64_t kTransportMarker = 0x55425546474D5354ULL;
constexpr uint64_t kResultMagic = 0x5530524553554C54ULL;
constexpr uint64_t kGuardMagic = 0x5530475541524435ULL;
constexpr uint64_t kPayloadPoisonWord = 0xA5A5A5A5A5A5A5A5ULL;
constexpr uint64_t kReportPoisonWord = 0xD3D3D3D3D3D3D3D3ULL;
constexpr uint8_t kReportPoisonByte = 0xD3U;

constexpr uint32_t kWarpSize = 32U;
constexpr uint32_t kWarpCount = 64U;
constexpr uint32_t kThreadCount = kWarpSize * kWarpCount;
constexpr uint32_t kTaskCount = kWarpCount;
constexpr uint32_t kPayloadClassCount = 4U;
constexpr uint32_t kMaxPayloadLines = 68U;
constexpr uint32_t kWordsPerLine = kCacheLineBytes / sizeof(uint64_t);
constexpr uint32_t kMaxPayloadWords = kMaxPayloadLines * kWordsPerLine;
constexpr uint32_t kUbufAlignmentBytes = kCacheLineBytes;
constexpr uint32_t kUbufGuardLines = 1U;
constexpr uint32_t kUbufPayloadOffsetBytes = kUbufGuardLines * kCacheLineBytes;
constexpr uint32_t kUbufRegionBytes = (kUbufGuardLines + kMaxPayloadLines + kUbufGuardLines) * kCacheLineBytes;
constexpr uint32_t kUbufPayloadOffsetWords = kUbufPayloadOffsetBytes / sizeof(uint64_t);
constexpr uint32_t kUbufRegionWords = kUbufRegionBytes / sizeof(uint64_t);
constexpr uint32_t kUbufGuardAfterOffsetWords = kUbufPayloadOffsetWords + kMaxPayloadWords;
constexpr uint32_t kUbufSlotCount = 1U;
constexpr uint32_t kRoleCount = 3U;
constexpr uint32_t kBuilderOwner = 32U;
constexpr uint32_t kExecutorOwner = 33U;
constexpr uint64_t kSlotFree = 0U;
constexpr uint64_t kDefaultTimeoutTicks = UINT64_C(1000000000);
constexpr uint32_t kInvalidTaskId = UINT32_MAX;
constexpr uint32_t kGuardBeforeTasks = 0U;
constexpr uint32_t kGuardAfterTasks = 1U;
constexpr uint32_t kGuardBeforeRoles = 2U;
constexpr uint32_t kGuardAfterRoles = 3U;
constexpr uint32_t kGuardBeforeBuilderThreads = 4U;
constexpr uint32_t kGuardAfterBuilderThreads = 5U;
constexpr uint32_t kGuardBeforeStagingSlot = 6U;
constexpr uint32_t kGuardAfterStagingSlot = 7U;

enum class TransportKind : uint32_t {
    SimtUbufReadToGmWordStore = 1U,
};

enum class ProbeRole : uint32_t {
    AicObserver = 0U,
    Aiv0Builder = 1U,
    Aiv1Executor = 2U,
};

enum class U0FatalReason : uint8_t {
    None = 0U,
    InvalidConfig = 1U,
    InvalidTopology = 2U,
    TaskClaimConflict = 3U,
    SlotTimeout = 4U,
    SlotInvariant = 5U,
    UbufGuardCorruption = 6U,
    TaskPublishConflict = 7U,
    InvalidTaskState = 8U,
    PayloadMismatch = 9U,
    TaskCompleteConflict = 10U,
    TaskWaitTimeout = 11U,
    DrainTimeout = 12U,
    BuildAborted = 13U,
};

struct DecodedU0Fatal {
    U0FatalReason reason;
    uint32_t reporter_owner;
    uint32_t task_id;
    bool valid;
};

enum BuildPhaseBit : uint32_t {
    kBuildClaimed = 1U << 0U,
    kBuildSlotAcquired = 1U << 1U,
    kBuildUbufComplete = 1U << 2U,
    kBuildUbufGuardsValid = 1U << 3U,
    kBuildGmStoreComplete = 1U << 4U,
    kBuildPublished = 1U << 5U,
    kBuildSlotReleased = 1U << 6U,
};

constexpr uint32_t kExpectedBuildPhaseBits = kBuildClaimed | kBuildSlotAcquired | kBuildUbufComplete |
                                             kBuildUbufGuardsValid | kBuildGmStoreComplete | kBuildPublished |
                                             kBuildSlotReleased;

enum ExecPhaseBit : uint32_t {
    kExecClaimed = 1U << 0U,
    kExecPayloadRead = 1U << 1U,
    kExecPayloadValid = 1U << 2U,
    kExecCompleted = 1U << 3U,
};

constexpr uint32_t kExpectedExecPhaseBits = kExecClaimed | kExecPayloadRead | kExecPayloadValid | kExecCompleted;

enum ThreadStatusBit : uint32_t {
    kThreadActiveLeader = 1U << 0U,
    kThreadTaskClaimed = 1U << 1U,
    kThreadSlotAcquired = 1U << 2U,
    kThreadPayloadComplete = 1U << 3U,
    kThreadUbufGuardsValid = 1U << 4U,
    kThreadTaskPublished = 1U << 5U,
    kThreadSlotReleased = 1U << 6U,
};

constexpr uint32_t kExpectedBuilderThreadStatus = kThreadActiveLeader | kThreadTaskClaimed | kThreadSlotAcquired |
                                                  kThreadPayloadComplete | kThreadUbufGuardsValid |
                                                  kThreadTaskPublished | kThreadSlotReleased;

SIMT_CROSS_CORE_U0_INLINE uint32_t PayloadLinesForTask(uint32_t task_id) {
    const uint32_t payload_class = task_id % kPayloadClassCount;
    return payload_class == 0U ? 1U : (payload_class == 1U ? 10U : (payload_class == 2U ? 16U : 68U));
}

SIMT_CROSS_CORE_U0_INLINE uint32_t PayloadWordsForTask(uint32_t task_id) {
    return PayloadLinesForTask(task_id) * kWordsPerLine;
}

SIMT_CROSS_CORE_U0_INLINE uint64_t SlotOwnerValue(uint32_t warp_id) { return static_cast<uint64_t>(warp_id) + 1U; }

SIMT_CROSS_CORE_U0_INLINE bool U0ReporterOwnerValid(uint32_t owner) {
    return owner == 0U || owner == kBuilderOwner || owner == kExecutorOwner;
}

SIMT_CROSS_CORE_U0_INLINE uint64_t EncodeU0Fatal(U0FatalReason reason, uint32_t reporter_owner, uint32_t task_id) {
    return (static_cast<uint64_t>(reason) << g0::kFatalReasonShift) |
           (static_cast<uint64_t>(reporter_owner) << g0::kFatalOwnerShift) |
           (static_cast<uint64_t>(task_id) << g0::kFatalTaskIdShift);
}

SIMT_CROSS_CORE_U0_INLINE DecodedU0Fatal DecodeU0Fatal(int64_t raw_state) {
    const uint64_t raw = static_cast<uint64_t>(raw_state);
    const U0FatalReason reason = static_cast<U0FatalReason>((raw >> g0::kFatalReasonShift) & g0::kFatalReasonMask);
    const uint32_t owner = static_cast<uint32_t>((raw >> g0::kFatalOwnerShift) & g0::kFatalOwnerMask);
    const uint32_t task_id = static_cast<uint32_t>((raw >> g0::kFatalTaskIdShift) & g0::kFatalTaskIdMask);
    return DecodedU0Fatal{
        reason,
        owner,
        task_id,
        raw != 0U && (raw & ~g0::kFatalKnownMask) == 0U && reason >= U0FatalReason::InvalidConfig &&
            reason <= U0FatalReason::BuildAborted && U0ReporterOwnerValid(owner),
    };
}

SIMT_CROSS_CORE_U0_INLINE uint64_t EmptyState() {
    return EncodeExecState(ExecPhase::Empty, 0U, 0U, ExecEngineClass::None, 0U, 0U);
}

SIMT_CROSS_CORE_U0_INLINE uint64_t BuildingState(uint32_t task_id) {
    return EncodeExecState(ExecPhase::Building, kBuilderOwner, kUnboundOwner, ExecEngineClass::None, 0U, task_id);
}

SIMT_CROSS_CORE_U0_INLINE uint64_t BuiltState(uint32_t task_id) {
    return EncodeExecState(
        ExecPhase::Built, kBuilderOwner, kUnboundOwner, ExecEngineClass::Aiv, PayloadLinesForTask(task_id), task_id
    );
}

SIMT_CROSS_CORE_U0_INLINE uint64_t ClaimedState(uint32_t task_id) {
    return EncodeExecState(
        ExecPhase::Claimed, kBuilderOwner, kExecutorOwner, ExecEngineClass::Aiv, PayloadLinesForTask(task_id), task_id
    );
}

SIMT_CROSS_CORE_U0_INLINE uint64_t DoneState(uint32_t task_id) {
    return EncodeExecState(
        ExecPhase::Done, kBuilderOwner, kExecutorOwner, ExecEngineClass::Aiv, PayloadLinesForTask(task_id), task_id
    );
}

SIMT_CROSS_CORE_U0_INLINE uint64_t RotateLeft(uint64_t value, uint32_t shift) {
    return (value << shift) | (value >> (64U - shift));
}

SIMT_CROSS_CORE_U0_INLINE uint64_t ExpectedPayloadWord(uint64_t nonce, uint32_t task_id, uint32_t word) {
    const uint32_t shift = ((task_id * 7U + word) % 31U) + 1U;
    return 0x5530000000000000ULL ^ RotateLeft(nonce ^ (0x0101010101010101ULL * (word + 1U)), shift) ^
           (static_cast<uint64_t>(task_id) << 40U) ^ static_cast<uint64_t>(word);
}

SIMT_CROSS_CORE_U0_INLINE uint64_t FoldChecksum(uint64_t checksum, uint64_t word, uint32_t index) {
    return RotateLeft(checksum, 11U) ^ word ^ (0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(index));
}

SIMT_CROSS_CORE_U0_INLINE uint64_t ExpectedPayloadChecksum(uint64_t nonce, uint32_t task_id) {
    uint64_t checksum = 0x243F6A8885A308D3ULL ^ static_cast<uint64_t>(task_id);
    const uint32_t payload_words = PayloadWordsForTask(task_id);
    for (uint32_t word = 0U; word < payload_words; ++word) {
        checksum = FoldChecksum(checksum, ExpectedPayloadWord(nonce, task_id, word), word);
    }
    return checksum;
}

SIMT_CROSS_CORE_U0_INLINE uint64_t ExpectedGuardWord(uint64_t nonce, uint32_t guard_id, uint32_t word) {
    uint64_t value = kGuardMagic ^ nonce ^ (static_cast<uint64_t>(guard_id) << 32U) ^ word;
    value ^= value >> 27U;
    value *= 0xD6E8FEB86659FD93ULL;
    return value ^ (value >> 31U);
}

SIMT_CROSS_CORE_U0_INLINE uint64_t
ExpectedThreadChecksum(uint64_t nonce, uint32_t thread_id, uint32_t task_id, uint32_t status) {
    uint64_t checksum = nonce ^ (static_cast<uint64_t>(thread_id) << 32U) ^ task_id;
    checksum = FoldChecksum(checksum, status, 0U);
    return FoldChecksum(checksum, kTransportMarker, 1U);
}

struct alignas(kCacheLineBytes) U0Control {
    uint64_t magic;
    uint64_t version;
    uint64_t launch_nonce;
    uint64_t timeout_ticks;
    uint16_t thread_count;
    uint16_t warp_count;
    uint16_t task_count;
    uint16_t role_count;
    uint16_t payload_class_count;
    uint16_t max_payload_lines;
    uint16_t words_per_line;
    uint16_t ubuf_alignment_bytes;
    uint32_t ubuf_region_bytes;
    uint32_t ubuf_payload_offset_bytes;
    TransportKind transport_kind;
    uint32_t ubuf_slot_count;
};

struct alignas(kCacheLineBytes) U0Guard {
    uint64_t words[kWordsPerLine];
};

struct alignas(kCacheLineBytes) U0TaskControl {
    volatile int64_t state;
    uint8_t padding[kCacheLineBytes - sizeof(int64_t)];
};

struct alignas(kCacheLineBytes) U0TaskPayload {
    volatile uint64_t words[kMaxPayloadWords];
};

struct alignas(kCacheLineBytes) U0BuildReport {
    uint32_t task_id;
    uint32_t builder_thread;
    uint32_t builder_warp;
    uint32_t builder_lane;
    uint32_t phase_bits;
    uint32_t payload_lines;
    uint32_t ubuf_words_written;
    uint32_t gm_words_stored;
    uint32_t claim_count;
    uint32_t publish_count;
    uint32_t release_count;
    uint32_t slot_ticket;
    uint64_t launch_nonce;
    uint64_t payload_checksum;
};

struct alignas(kCacheLineBytes) U0ExecReport {
    uint32_t task_id;
    uint32_t executor_owner;
    uint32_t executor_physical_block;
    uint32_t executor_subblock_id;
    uint32_t phase_bits;
    uint32_t payload_lines;
    uint32_t payload_words_read;
    uint32_t claim_count;
    uint32_t completion_count;
    uint32_t checksum_match_count;
    uint32_t reserved32[2];
    uint64_t launch_nonce;
    uint64_t payload_checksum;
};

struct alignas(kCacheLineBytes) U0Task {
    U0TaskControl control;
    U0TaskPayload payload;
    U0BuildReport build_report;
    U0ExecReport exec_report;
};

struct alignas(kCacheLineBytes) U0ThreadReport {
    uint32_t thread_id;
    uint32_t warp_id;
    uint32_t lane_id;
    uint32_t active_leader;
    uint32_t task_id;
    uint32_t status;
    uint32_t task_attempt_count;
    uint32_t slot_attempt_count;
    uint64_t launch_nonce;
    uint64_t checksum;
    uint64_t reserved[2];
};

struct alignas(kCacheLineBytes) U0RoleReport {
    uint32_t owner;
    ProbeRole role;
    uint32_t physical_block;
    uint32_t subblock_id;
    uint32_t main_scalar_build_action_count;
    uint32_t task_claim_count;
    uint32_t task_finish_count;
    uint32_t timeout_count;
    uint64_t launch_nonce;
    uint64_t result_magic;
    uint64_t reserved[2];
};

struct alignas(kCacheLineBytes) U0ProbeState {
    U0Control control;
    U0Guard guard_before_tasks;
    U0Task tasks[kTaskCount];
    U0Guard guard_after_tasks;
    AtomicLine slot_owner;
    AtomicLine slot_busy_depth;
    AtomicLine slot_max_busy_depth;
    AtomicLine slot_acquire_count;
    AtomicLine slot_release_count;
    AtomicLine build_claim_count;
    AtomicLine built_count;
    AtomicLine exec_claim_count;
    AtomicLine done_count;
    AtomicLine mte3_count;
    AtomicLine ubuf_guard_check_count;
    AtomicLine builder_finished_count;
    AtomicLine executor_finished_count;
    AtomicLine fatal;
    U0Guard guard_before_roles;
    U0RoleReport roles[kRoleCount];
    U0Guard guard_after_roles;
    U0Guard guard_before_builder_threads;
    U0ThreadReport builder_threads[kThreadCount];
    U0Guard guard_after_builder_threads;
};

static_assert(kThreadCount == 2048U && kTaskCount == 64U, "U0 must model 64 warps of 32 threads");
static_assert(
    kUbufRegionBytes == 4480U && kUbufPayloadOffsetBytes == 64U && kUbufRegionBytes % kUbufAlignmentBytes == 0U,
    "U0 guarded UBUF region ABI changed"
);
#if !defined(__CCE_AICORE__)
static_assert(PayloadLinesForTask(0U) == 1U && PayloadLinesForTask(1U) == 10U, "U0 payload classes changed");
static_assert(PayloadLinesForTask(2U) == 16U && PayloadLinesForTask(3U) == 68U, "U0 payload classes changed");
#endif
static_assert(sizeof(U0Control) == kCacheLineBytes, "U0 control ABI changed");
static_assert(sizeof(U0Guard) == kCacheLineBytes, "U0 guard ABI changed");
static_assert(sizeof(U0TaskControl) == kCacheLineBytes, "U0 task control ABI changed");
static_assert(sizeof(U0TaskPayload) == kMaxPayloadLines * kCacheLineBytes, "U0 task payload ABI changed");
static_assert(sizeof(U0BuildReport) == kCacheLineBytes, "U0 build report ABI changed");
static_assert(sizeof(U0ExecReport) == kCacheLineBytes, "U0 exec report ABI changed");
static_assert(sizeof(U0Task) == (kMaxPayloadLines + 3U) * kCacheLineBytes, "U0 task aggregate ABI changed");
static_assert(
    offsetof(U0Task, payload) == kCacheLineBytes &&
        offsetof(U0Task, build_report) == (kMaxPayloadLines + 1U) * kCacheLineBytes &&
        offsetof(U0Task, exec_report) == (kMaxPayloadLines + 2U) * kCacheLineBytes,
    "U0 task subregion offsets changed"
);
static_assert(sizeof(U0ThreadReport) == kCacheLineBytes, "U0 thread report ABI changed");
static_assert(sizeof(U0RoleReport) == kCacheLineBytes, "U0 role report ABI changed");
static_assert(
    offsetof(U0RoleReport, main_scalar_build_action_count) == 16U && offsetof(U0RoleReport, task_claim_count) == 20U &&
        offsetof(U0RoleReport, task_finish_count) == 24U,
    "U0 role task-accounting offsets changed"
);
static_assert(
    offsetof(U0ProbeState, mte3_count) == offsetof(U0ProbeState, done_count) + kCacheLineBytes &&
        offsetof(U0ProbeState, ubuf_guard_check_count) == offsetof(U0ProbeState, mte3_count) + kCacheLineBytes &&
        offsetof(U0ProbeState, builder_finished_count) ==
            offsetof(U0ProbeState, ubuf_guard_check_count) + kCacheLineBytes,
    "U0 direct-store evidence offsets changed"
);
static_assert(
    offsetof(U0ProbeState, tasks) % kCacheLineBytes == 0U &&
        offsetof(U0ProbeState, builder_threads) % kCacheLineBytes == 0U && sizeof(U0ProbeState) % kCacheLineBytes == 0U,
    "all U0 GM regions must remain cache-line aligned"
);

#undef SIMT_CROSS_CORE_U0_INLINE

}  // namespace pa_scheduler::simt_cross_core::u0

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_U0_SINGLE_SLOT_H
