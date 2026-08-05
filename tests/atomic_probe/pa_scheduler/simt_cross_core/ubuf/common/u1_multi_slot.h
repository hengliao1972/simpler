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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_U1_MULTI_SLOT_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_U1_MULTI_SLOT_H

#include <stddef.h>
#include <stdint.h>

#include "../../common/full_pa_exec_protocol.h"

namespace pa_scheduler::simt_cross_core::u1 {

#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_U1_INLINE __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_U1_INLINE constexpr
#endif

using g0::AtomicLine;
using g0::DecodeExecState;
using g0::EncodeExecState;
using g0::ExecEngineClass;
using g0::ExecPhase;
using g0::kCacheLineBytes;
using g0::kUnboundOwner;

constexpr uint64_t kProbeMagic = 0x53494D5455314135ULL;
constexpr uint64_t kProbeVersion = 1U;
constexpr uint64_t kTransportMarker = 0x5542554647314D53ULL;
constexpr uint64_t kResultMagic = 0x5531524553554C54ULL;
constexpr uint64_t kGuardMagic = 0x5531475541524435ULL;
constexpr uint64_t kPayloadPoisonWord = 0xA5A5A5A5A5A5A5A5ULL;
constexpr uint64_t kReportPoisonWord = 0xD3D3D3D3D3D3D3D3ULL;
constexpr uint8_t kReportPoisonByte = 0xD3U;

constexpr uint32_t kWarpSize = 32U;
constexpr uint32_t kWarpCount = 64U;
constexpr uint32_t kThreadCount = kWarpSize * kWarpCount;
constexpr uint32_t kTaskCount = 128U;
constexpr uint32_t kTasksPerWarp = kTaskCount / kWarpCount;
constexpr uint32_t kPayloadClassCount = 4U;
constexpr uint32_t kMaxPayloadLines = 16U;
constexpr uint32_t kWordsPerLine = kCacheLineBytes / sizeof(uint64_t);
constexpr uint32_t kMaxPayloadWords = kMaxPayloadLines * kWordsPerLine;
constexpr uint32_t kUbufAlignmentBytes = kCacheLineBytes;
constexpr uint32_t kUbufGuardLines = 1U;
constexpr uint32_t kUbufSlotCount = 4U;
constexpr uint32_t kUbufPayloadOffsetBytes = kUbufGuardLines * kCacheLineBytes;
constexpr uint32_t kUbufSlotStrideBytes = (kUbufGuardLines + kMaxPayloadLines + kUbufGuardLines) * kCacheLineBytes;
constexpr uint32_t kUbufRegionBytes = kUbufSlotCount * kUbufSlotStrideBytes;
constexpr uint32_t kUbufPayloadOffsetWords = kUbufPayloadOffsetBytes / sizeof(uint64_t);
constexpr uint32_t kUbufSlotStrideWords = kUbufSlotStrideBytes / sizeof(uint64_t);
constexpr uint32_t kUbufRegionWords = kUbufRegionBytes / sizeof(uint64_t);
constexpr uint32_t kUbufGuardAfterOffsetWords = kUbufPayloadOffsetWords + kMaxPayloadWords;
constexpr uint32_t kSlotReuseCount = kTaskCount / kUbufSlotCount;
constexpr uint32_t kAnchorTaskCount = kUbufSlotCount;
constexpr uint64_t kAnchorStagedMask = (uint64_t{1U} << kAnchorTaskCount) - 1U;
constexpr uint32_t kRoleCount = 3U;
constexpr uint32_t kBuilderOwner = 32U;
constexpr uint32_t kExecutorOwner = 33U;
constexpr uint64_t kDefaultTimeoutTicks = UINT64_C(1000000000);
constexpr uint32_t kInvalidTaskId = UINT32_MAX;
constexpr uint32_t kInvalidSlotId = UINT32_MAX;

constexpr uint32_t kGuardBeforeTasks = 0U;
constexpr uint32_t kGuardAfterTasks = 1U;
constexpr uint32_t kGuardBeforeRoles = 2U;
constexpr uint32_t kGuardAfterRoles = 3U;
constexpr uint32_t kGuardBeforeBuilderThreads = 4U;
constexpr uint32_t kGuardAfterBuilderThreads = 5U;
constexpr uint32_t kGuardBeforeStagingSlots = 6U;

enum class TransportKind : uint16_t {
    SimtUbufReadToGmWordStore = 1U,
};

enum class ProbeRole : uint32_t {
    AicObserver = 0U,
    Aiv0Builder = 1U,
    Aiv1Executor = 2U,
};

enum class U1FatalReason : uint8_t {
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

struct DecodedU1Fatal {
    U1FatalReason reason;
    uint32_t reporter_owner;
    uint32_t task_id;
    bool valid;
};

struct DecodedSlotState {
    uint32_t generation;
    uint32_t task_id;
    bool busy;
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
    kThreadTasksClaimed = 1U << 1U,
    kThreadSlotsAcquired = 1U << 2U,
    kThreadPayloadsComplete = 1U << 3U,
    kThreadUbufGuardsValid = 1U << 4U,
    kThreadTasksPublished = 1U << 5U,
    kThreadSlotsReleased = 1U << 6U,
};

constexpr uint32_t kExpectedBuilderThreadStatus = kThreadActiveLeader | kThreadTasksClaimed | kThreadSlotsAcquired |
                                                  kThreadPayloadsComplete | kThreadUbufGuardsValid |
                                                  kThreadTasksPublished | kThreadSlotsReleased;

SIMT_CROSS_CORE_U1_INLINE uint32_t PayloadClassForTask(uint32_t task_id) {
    return ((task_id / kUbufSlotCount) + (kPayloadClassCount - 1U)) % kPayloadClassCount;
}

SIMT_CROSS_CORE_U1_INLINE uint32_t PayloadLinesForClass(uint32_t payload_class) {
    return payload_class == 0U ? 1U : (payload_class == 1U ? 4U : (payload_class == 2U ? 10U : 16U));
}

SIMT_CROSS_CORE_U1_INLINE uint32_t PayloadLinesForTask(uint32_t task_id) {
    return PayloadLinesForClass(PayloadClassForTask(task_id));
}

SIMT_CROSS_CORE_U1_INLINE uint32_t PayloadWordsForTask(uint32_t task_id) {
    return PayloadLinesForTask(task_id) * kWordsPerLine;
}

SIMT_CROSS_CORE_U1_INLINE uint32_t SlotForTask(uint32_t task_id) { return task_id % kUbufSlotCount; }

SIMT_CROSS_CORE_U1_INLINE bool IsAnchorTask(uint32_t task_id) { return task_id < kAnchorTaskCount; }

SIMT_CROSS_CORE_U1_INLINE uint32_t FirstTaskForWarp(uint32_t warp_id) { return warp_id; }

SIMT_CROSS_CORE_U1_INLINE uint32_t LastTaskForWarp(uint32_t warp_id) { return warp_id + kWarpCount; }

SIMT_CROSS_CORE_U1_INLINE uint32_t UbufGuardBeforeId(uint32_t slot_id) {
    return kGuardBeforeStagingSlots + 2U * slot_id;
}

SIMT_CROSS_CORE_U1_INLINE uint32_t UbufGuardAfterId(uint32_t slot_id) { return UbufGuardBeforeId(slot_id) + 1U; }

SIMT_CROSS_CORE_U1_INLINE uint64_t SlotFreeState(uint32_t generation) {
    return static_cast<uint64_t>(generation) << 32U;
}

SIMT_CROSS_CORE_U1_INLINE uint64_t SlotBusyState(uint32_t generation, uint32_t task_id) {
    return (static_cast<uint64_t>(generation) << 32U) | (static_cast<uint64_t>(task_id) + 1U);
}

SIMT_CROSS_CORE_U1_INLINE uint32_t SlotStateGeneration(uint64_t raw_state) {
    return static_cast<uint32_t>(raw_state >> 32U);
}

SIMT_CROSS_CORE_U1_INLINE uint32_t SlotStateTaskPlusOne(uint64_t raw_state) { return static_cast<uint32_t>(raw_state); }

SIMT_CROSS_CORE_U1_INLINE bool SlotStateIsFree(uint64_t raw_state) { return SlotStateTaskPlusOne(raw_state) == 0U; }

SIMT_CROSS_CORE_U1_INLINE uint32_t SlotStateTaskId(uint64_t raw_state) {
    const uint32_t task_plus_one = SlotStateTaskPlusOne(raw_state);
    return task_plus_one == 0U ? kInvalidTaskId : task_plus_one - 1U;
}

SIMT_CROSS_CORE_U1_INLINE DecodedSlotState DecodeSlotState(int64_t raw_state) {
    const uint64_t raw = static_cast<uint64_t>(raw_state);
    const uint32_t task_plus_one = SlotStateTaskPlusOne(raw);
    const bool busy = task_plus_one != 0U;
    return DecodedSlotState{
        SlotStateGeneration(raw),
        busy ? task_plus_one - 1U : kInvalidTaskId,
        busy,
        !busy || task_plus_one <= kTaskCount,
    };
}

SIMT_CROSS_CORE_U1_INLINE bool SlotStateValidForSlot(uint64_t raw_state, uint32_t slot_id) {
    const DecodedSlotState decoded = DecodeSlotState(static_cast<int64_t>(raw_state));
    return slot_id < kUbufSlotCount && decoded.valid && decoded.generation <= kSlotReuseCount &&
           (!decoded.busy || (decoded.generation < kSlotReuseCount && decoded.task_id < kTaskCount &&
                              SlotForTask(decoded.task_id) == slot_id));
}

SIMT_CROSS_CORE_U1_INLINE bool U1ReporterOwnerValid(uint32_t owner) {
    return owner == 0U || owner == kBuilderOwner || owner == kExecutorOwner;
}

SIMT_CROSS_CORE_U1_INLINE uint64_t EncodeU1Fatal(U1FatalReason reason, uint32_t reporter_owner, uint32_t task_id) {
    return (static_cast<uint64_t>(reason) << g0::kFatalReasonShift) |
           (static_cast<uint64_t>(reporter_owner) << g0::kFatalOwnerShift) |
           (static_cast<uint64_t>(task_id) << g0::kFatalTaskIdShift);
}

SIMT_CROSS_CORE_U1_INLINE DecodedU1Fatal DecodeU1Fatal(int64_t raw_state) {
    const uint64_t raw = static_cast<uint64_t>(raw_state);
    const U1FatalReason reason = static_cast<U1FatalReason>((raw >> g0::kFatalReasonShift) & g0::kFatalReasonMask);
    const uint32_t owner = static_cast<uint32_t>((raw >> g0::kFatalOwnerShift) & g0::kFatalOwnerMask);
    const uint32_t task_id = static_cast<uint32_t>((raw >> g0::kFatalTaskIdShift) & g0::kFatalTaskIdMask);
    return DecodedU1Fatal{
        reason,
        owner,
        task_id,
        raw != 0U && (raw & ~g0::kFatalKnownMask) == 0U && reason >= U1FatalReason::InvalidConfig &&
            reason <= U1FatalReason::BuildAborted && U1ReporterOwnerValid(owner) &&
            (task_id < kTaskCount || task_id == kInvalidTaskId),
    };
}

SIMT_CROSS_CORE_U1_INLINE uint64_t EmptyState() {
    return EncodeExecState(ExecPhase::Empty, 0U, 0U, ExecEngineClass::None, 0U, 0U);
}

SIMT_CROSS_CORE_U1_INLINE uint64_t BuildingState(uint32_t task_id) {
    return EncodeExecState(ExecPhase::Building, kBuilderOwner, kUnboundOwner, ExecEngineClass::None, 0U, task_id);
}

SIMT_CROSS_CORE_U1_INLINE uint64_t BuiltState(uint32_t task_id) {
    return EncodeExecState(
        ExecPhase::Built, kBuilderOwner, kUnboundOwner, ExecEngineClass::Aiv, PayloadLinesForTask(task_id), task_id
    );
}

SIMT_CROSS_CORE_U1_INLINE uint64_t ClaimedState(uint32_t task_id) {
    return EncodeExecState(
        ExecPhase::Claimed, kBuilderOwner, kExecutorOwner, ExecEngineClass::Aiv, PayloadLinesForTask(task_id), task_id
    );
}

SIMT_CROSS_CORE_U1_INLINE uint64_t DoneState(uint32_t task_id) {
    return EncodeExecState(
        ExecPhase::Done, kBuilderOwner, kExecutorOwner, ExecEngineClass::Aiv, PayloadLinesForTask(task_id), task_id
    );
}

SIMT_CROSS_CORE_U1_INLINE uint64_t RotateLeft(uint64_t value, uint32_t shift) {
    return (value << shift) | (value >> (64U - shift));
}

SIMT_CROSS_CORE_U1_INLINE uint64_t ExpectedPayloadWord(uint64_t nonce, uint32_t task_id, uint32_t word) {
    const uint32_t shift = ((task_id * 7U + word) % 31U) + 1U;
    return 0x5531000000000000ULL ^ RotateLeft(nonce ^ (0x0101010101010101ULL * (word + 1U)), shift) ^
           (static_cast<uint64_t>(task_id) << 40U) ^ static_cast<uint64_t>(word);
}

SIMT_CROSS_CORE_U1_INLINE uint64_t FoldChecksum(uint64_t checksum, uint64_t word, uint32_t index) {
    return RotateLeft(checksum, 11U) ^ word ^ (0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(index));
}

SIMT_CROSS_CORE_U1_INLINE uint64_t ExpectedPayloadChecksum(uint64_t nonce, uint32_t task_id) {
    uint64_t checksum = 0x243F6A8885A308D3ULL ^ static_cast<uint64_t>(task_id);
    const uint32_t payload_words = PayloadWordsForTask(task_id);
    for (uint32_t word = 0U; word < payload_words; ++word) {
        checksum = FoldChecksum(checksum, ExpectedPayloadWord(nonce, task_id, word), word);
    }
    return checksum;
}

SIMT_CROSS_CORE_U1_INLINE uint64_t ExpectedGuardWord(uint64_t nonce, uint32_t guard_id, uint32_t word) {
    uint64_t value = kGuardMagic ^ nonce ^ (static_cast<uint64_t>(guard_id) << 32U) ^ word;
    value ^= value >> 27U;
    value *= 0xD6E8FEB86659FD93ULL;
    return value ^ (value >> 31U);
}

SIMT_CROSS_CORE_U1_INLINE uint64_t ExpectedThreadChecksum(
    uint64_t nonce, uint32_t thread_id, uint32_t first_task_id, uint32_t last_task_id, uint32_t status
) {
    uint64_t checksum = nonce ^ (static_cast<uint64_t>(thread_id) << 32U) ^ first_task_id;
    checksum = FoldChecksum(checksum, last_task_id, 0U);
    checksum = FoldChecksum(checksum, status, 1U);
    return FoldChecksum(checksum, kTransportMarker, 2U);
}

struct alignas(kCacheLineBytes) U1Control {
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
    uint16_t ubuf_slot_count;
    TransportKind transport_kind;
    uint32_t ubuf_slot_stride_bytes;
    uint32_t ubuf_region_bytes;
    uint32_t ubuf_payload_offset_bytes;
};

struct alignas(kCacheLineBytes) U1Guard {
    uint64_t words[kWordsPerLine];
};

struct alignas(kCacheLineBytes) U1TaskControl {
    volatile int64_t state;
    uint8_t padding[kCacheLineBytes - sizeof(int64_t)];
};

struct alignas(kCacheLineBytes) U1TaskPayload {
    volatile uint64_t words[kMaxPayloadWords];
};

struct alignas(kCacheLineBytes) U1BuildReport {
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
    uint32_t slot_id;
    uint32_t slot_generation;
    uint64_t launch_nonce;
    uint64_t payload_checksum;
};

struct alignas(kCacheLineBytes) U1ExecReport {
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

struct alignas(kCacheLineBytes) U1Task {
    U1TaskControl control;
    U1TaskPayload payload;
    U1BuildReport build_report;
    U1ExecReport exec_report;
};

struct alignas(kCacheLineBytes) U1ThreadReport {
    uint32_t thread_id;
    uint32_t warp_id;
    uint32_t lane_id;
    uint32_t active_leader;
    uint32_t first_task_id;
    uint32_t last_task_id;
    uint32_t task_count;
    uint32_t status;
    uint32_t slot_attempt_count;
    uint32_t build_count;
    uint64_t launch_nonce;
    uint64_t checksum;
    uint64_t reserved;
};

struct alignas(kCacheLineBytes) U1RoleReport {
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

struct alignas(kCacheLineBytes) U1ProbeState {
    U1Control control;
    U1Guard guard_before_tasks;
    U1Task tasks[kTaskCount];
    U1Guard guard_after_tasks;
    AtomicLine slot_states[kUbufSlotCount];
    AtomicLine slot_acquire_count[kUbufSlotCount];
    AtomicLine slot_release_count[kUbufSlotCount];
    AtomicLine global_busy_depth;
    AtomicLine global_max_busy_depth;
    AtomicLine anchor_staged_count;
    AtomicLine anchor_staged_mask;
    AtomicLine ubuf_guard_check_count;
    AtomicLine built_count;
    AtomicLine done_count;
    AtomicLine fatal;
    U1Guard guard_before_roles;
    U1RoleReport roles[kRoleCount];
    U1Guard guard_after_roles;
    U1Guard guard_before_builder_threads;
    U1ThreadReport builder_threads[kThreadCount];
    U1Guard guard_after_builder_threads;
};

static_assert(kThreadCount == 2048U && kWarpCount == 64U, "U1 must model 64 warps of 32 threads");
static_assert(kTaskCount == 128U && kTasksPerWarp == 2U, "each U1 warp leader must own two tasks");
static_assert(kUbufSlotCount == 4U && kSlotReuseCount == 32U, "U1 slot reuse ABI changed");
static_assert(kAnchorStagedMask == 0xFU, "U1 anchor identity mask changed");
static_assert(
    kUbufSlotStrideBytes == 1152U && kUbufRegionBytes == 4608U && kUbufPayloadOffsetBytes == 64U &&
        kUbufSlotStrideBytes % kUbufAlignmentBytes == 0U,
    "U1 guarded four-slot UBUF ABI changed"
);
#if !defined(__CCE_AICORE__)
static_assert(
    PayloadLinesForTask(0U) == 16U && PayloadLinesForTask(4U) == 1U && PayloadLinesForTask(8U) == 4U &&
        PayloadLinesForTask(12U) == 10U,
    "U1 payload-class rotation changed"
);
static_assert(
    SlotForTask(0U) == 0U && SlotForTask(3U) == 3U && SlotForTask(127U) == 3U, "U1 task-to-slot mapping changed"
);
static_assert(
    FirstTaskForWarp(0U) == 0U && LastTaskForWarp(0U) == 64U && FirstTaskForWarp(63U) == 63U &&
        LastTaskForWarp(63U) == 127U,
    "U1 warp-to-task mapping changed"
);
static_assert(
    SlotFreeState(0U) == 0U && SlotFreeState(kSlotReuseCount) == (uint64_t{32U} << 32U) &&
        SlotBusyState(7U, 31U) == ((uint64_t{7U} << 32U) | uint64_t{32U}),
    "U1 slot-state encoding changed"
);
static_assert(
    SlotStateValidForSlot(SlotBusyState(7U, 31U), 3U) && !SlotStateValidForSlot(SlotBusyState(7U, 31U), 0U) &&
        SlotStateValidForSlot(SlotFreeState(kSlotReuseCount), 0U),
    "U1 slot-state contextual validation changed"
);
#endif
static_assert(sizeof(U1Control) == kCacheLineBytes, "U1 control ABI changed");
static_assert(sizeof(U1Guard) == kCacheLineBytes, "U1 guard ABI changed");
static_assert(sizeof(U1TaskControl) == kCacheLineBytes, "U1 task control ABI changed");
static_assert(sizeof(U1TaskPayload) == kMaxPayloadLines * kCacheLineBytes, "U1 task payload ABI changed");
static_assert(sizeof(U1BuildReport) == kCacheLineBytes, "U1 build report ABI changed");
static_assert(sizeof(U1ExecReport) == kCacheLineBytes, "U1 exec report ABI changed");
static_assert(sizeof(U1Task) == (kMaxPayloadLines + 3U) * kCacheLineBytes, "U1 task aggregate ABI changed");
static_assert(
    offsetof(U1Task, payload) == kCacheLineBytes &&
        offsetof(U1Task, build_report) == (kMaxPayloadLines + 1U) * kCacheLineBytes &&
        offsetof(U1Task, exec_report) == (kMaxPayloadLines + 2U) * kCacheLineBytes,
    "U1 task subregion offsets changed"
);
static_assert(sizeof(U1ThreadReport) == kCacheLineBytes, "U1 thread report ABI changed");
static_assert(sizeof(U1RoleReport) == kCacheLineBytes, "U1 role report ABI changed");
static_assert(sizeof(AtomicLine) == kCacheLineBytes, "U1 slot atomics must occupy independent cache lines");
static_assert(
    offsetof(U1BuildReport, slot_id) == 40U && offsetof(U1BuildReport, slot_generation) == 44U,
    "U1 slot provenance report offsets changed"
);
static_assert(
    offsetof(U1ProbeState, tasks) % kCacheLineBytes == 0U &&
        offsetof(U1ProbeState, slot_states) % kCacheLineBytes == 0U &&
        offsetof(U1ProbeState, slot_acquire_count) % kCacheLineBytes == 0U &&
        offsetof(U1ProbeState, slot_release_count) % kCacheLineBytes == 0U &&
        offsetof(U1ProbeState, roles) % kCacheLineBytes == 0U &&
        offsetof(U1ProbeState, builder_threads) % kCacheLineBytes == 0U && sizeof(U1ProbeState) % kCacheLineBytes == 0U,
    "all U1 GM regions must remain cache-line aligned"
);

#undef SIMT_CROSS_CORE_U1_INLINE

}  // namespace pa_scheduler::simt_cross_core::u1

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_U1_MULTI_SLOT_H
