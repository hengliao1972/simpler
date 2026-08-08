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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_U2_FULL_PA_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_U2_FULL_PA_H

#include "../../gm/common/g0_full_pa.h"
#include "ubuf_staging_protocol.h"

namespace pa_scheduler::simt_cross_core::u2 {

#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_U2_INLINE __aicore__ __attribute__((always_inline)) inline
#if defined(__DAV_VEC__)
#define SIMT_CROSS_CORE_U2_SIMT_INLINE __simt_callee__ __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_U2_SIMT_INLINE __aicore__ __attribute__((always_inline)) inline
#endif
#else
#define SIMT_CROSS_CORE_U2_INLINE constexpr
#define SIMT_CROSS_CORE_U2_SIMT_INLINE constexpr
#endif

using g0::AtomicLine;
using g0::ExecPayloadLayout;
using g0::FullPaState;
using g0::TaskKind;
using ubuf_staging::DecodedSlotState;
using ubuf_staging::TransportKind;

constexpr uint64_t kProbeMagic = 0x53494D5455324135ULL;
constexpr uint64_t kProbeVersion = 1U;
constexpr uint64_t kGuardMagic = 0x5532475541524435ULL;
constexpr uint64_t kReportPoisonWord = 0xD3D3D3D3D3D3D3D3ULL;
constexpr uint32_t kBuilderCount = 1U;
constexpr uint32_t kAnchorTaskFirst = 1U;
constexpr uint32_t kAnchorTaskCount = ubuf_staging::kSlotCount;
constexpr uint64_t kAnchorMask = (uint64_t{1U} << kAnchorTaskCount) - 1U;
constexpr uint32_t kMaxTransportReports = g0::kMainKernelTaskCount;
constexpr uint32_t kGuardBeforeSlots = 0U;
constexpr uint32_t kGuardAfterSlots = 1U;
constexpr uint32_t kGuardBeforeReports = 2U;
constexpr uint32_t kGuardAfterReports = 3U;

enum TransportPhaseBit : uint32_t {
    kTransportSlotAcquired = 1U << 0U,
    kTransportUbufComplete = 1U << 1U,
    kTransportGuardsValid = 1U << 2U,
    kTransportGmComplete = 1U << 3U,
    kTransportBuiltPublished = 1U << 4U,
    kTransportSlotReleased = 1U << 5U,
};

constexpr uint32_t kExpectedTransportPhaseBits = kTransportSlotAcquired | kTransportUbufComplete |
                                                 kTransportGuardsValid | kTransportGmComplete |
                                                 kTransportBuiltPublished | kTransportSlotReleased;

struct alignas(g0::kCacheLineBytes) U2Control {
    uint64_t magic;
    uint64_t version;
    uint64_t launch_nonce;
    uint32_t batch_count;
    uint32_t task_count;
    uint32_t kernel_task_count;
    uint32_t builder_count;
    uint16_t slot_count;
    uint16_t max_payload_lines;
    uint16_t words_per_line;
    uint16_t alignment_bytes;
    TransportKind transport_kind;
    uint16_t reserved16;
    uint32_t slot_stride_bytes;
    uint32_t region_bytes;
    uint32_t payload_offset_bytes;
};

struct alignas(g0::kCacheLineBytes) U2Guard {
    uint64_t words[ubuf_staging::kWordsPerLine];
};

struct alignas(g0::kCacheLineBytes) U2TaskStagingReport {
    uint32_t task_id;
    uint32_t slot_id;
    uint32_t generation;
    uint32_t phase_bits;
    uint32_t ubuf_words_written;
    uint32_t gm_words_stored;
    uint32_t guard_check_count;
    uint32_t acquire_count;
    uint32_t release_count;
    uint32_t reserved32[3];
    uint64_t launch_nonce;
    uint64_t payload_checksum;
};

struct alignas(g0::kCacheLineBytes) U2StagingState {
    U2Control control;
    U2Guard guard_before_slots;
    AtomicLine slot_states[ubuf_staging::kSlotCount];
    AtomicLine slot_acquire_count[ubuf_staging::kSlotCount];
    AtomicLine slot_release_count[ubuf_staging::kSlotCount];
    AtomicLine global_busy_depth;
    AtomicLine global_max_busy_depth;
    AtomicLine anchor_staged_count;
    AtomicLine anchor_staged_mask;
    AtomicLine guard_check_count;
    AtomicLine ubuf_words_written;
    AtomicLine gm_words_stored;
    U2Guard guard_after_slots;
    U2Guard guard_before_reports;
    U2TaskStagingReport reports[kMaxTransportReports];
    U2Guard guard_after_reports;
};

struct alignas(g0::kCacheLineBytes) U2FullPaState {
    FullPaState full_pa;
    U2StagingState staging;
};

SIMT_CROSS_CORE_U2_INLINE bool TaskUsesSlot(uint32_t task_id) { return g0::TaskExecutable(g0::TaskKindAt(task_id)); }

SIMT_CROSS_CORE_U2_INLINE uint32_t SlotForTask(uint32_t task_id) { return task_id % ubuf_staging::kSlotCount; }

SIMT_CROSS_CORE_U2_INLINE uint32_t ExpectedGeneration(uint32_t task_id) { return g0::TaskBatch(task_id); }

SIMT_CROSS_CORE_U2_INLINE uint32_t TaskForSlotGeneration(uint32_t slot_id, uint32_t generation) {
    uint32_t kind_ordinal =
        (slot_id + ubuf_staging::kSlotCount - generation % ubuf_staging::kSlotCount) % ubuf_staging::kSlotCount;
    kind_ordinal = kind_ordinal == 0U ? ubuf_staging::kSlotCount : kind_ordinal;
    return generation * g0::kTasksPerBatch + kind_ordinal;
}

SIMT_CROSS_CORE_U2_INLINE bool SlotStateValid(uint64_t raw_state, uint32_t slot_id, uint32_t batches) {
    const DecodedSlotState decoded =
        ubuf_staging::DecodeSlotState(static_cast<int64_t>(raw_state), g0::TaskCount(batches));
    if (slot_id >= ubuf_staging::kSlotCount || !decoded.valid || decoded.generation > batches) {
        return false;
    }
    if (!decoded.busy) {
        return true;
    }
    return decoded.generation < batches && decoded.task_id == TaskForSlotGeneration(slot_id, decoded.generation) &&
           TaskUsesSlot(decoded.task_id) && SlotForTask(decoded.task_id) == slot_id &&
           ExpectedGeneration(decoded.task_id) == decoded.generation;
}

SIMT_CROSS_CORE_U2_INLINE bool IsAnchorTask(uint32_t task_id) {
    return task_id >= kAnchorTaskFirst && task_id < kAnchorTaskFirst + kAnchorTaskCount;
}

SIMT_CROSS_CORE_U2_INLINE uint64_t AnchorBit(uint32_t task_id) {
    return IsAnchorTask(task_id) ? uint64_t{1U} << (task_id - kAnchorTaskFirst) : 0U;
}

SIMT_CROSS_CORE_U2_INLINE uint32_t TransportReportIndex(uint32_t task_id) {
    return g0::TaskBatch(task_id) * g0::kKernelsPerBatch + static_cast<uint32_t>(g0::TaskKindAt(task_id)) - 1U;
}

SIMT_CROSS_CORE_U2_INLINE bool TaskPayloadLayout(uint32_t task_id, ExecPayloadLayout &layout) {
    const g0::TaskExecShape shape = g0::TaskShape(g0::TaskKindAt(task_id));
    return TaskUsesSlot(task_id) &&
           g0::ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout);
}

SIMT_CROSS_CORE_U2_INLINE uint32_t PayloadWrittenWords(TaskKind kind) {
    ExecPayloadLayout layout{};
    const g0::TaskExecShape shape = g0::TaskShape(kind);
    return g0::ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout) ?
               layout.written_words :
               0U;
}

SIMT_CROSS_CORE_U2_INLINE uint32_t PayloadBytes(TaskKind kind) {
    ExecPayloadLayout layout{};
    const g0::TaskExecShape shape = g0::TaskShape(kind);
    return g0::ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout) ?
               layout.payload_bytes :
               0U;
}

SIMT_CROSS_CORE_U2_INLINE uint32_t PayloadLines(TaskKind kind) {
    ExecPayloadLayout layout{};
    const g0::TaskExecShape shape = g0::TaskShape(kind);
    return g0::ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout) ?
               layout.payload_lines :
               0U;
}

SIMT_CROSS_CORE_U2_SIMT_INLINE uint64_t PayloadChecksumSeed(uint64_t nonce, uint32_t task_id, uint32_t written_words) {
    return nonce ^ (static_cast<uint64_t>(task_id) << 32U) ^ written_words;
}

SIMT_CROSS_CORE_U2_SIMT_INLINE uint64_t FoldPayloadChecksum(uint64_t checksum, uint64_t word) {
    return checksum ^ (word + UINT64_C(0x9E3779B97F4A7C15) + (checksum << 6U) + (checksum >> 2U));
}

SIMT_CROSS_CORE_U2_INLINE uint64_t ExpectedGuardWord(uint64_t nonce, uint32_t guard_id, uint32_t word) {
    uint64_t value = kGuardMagic ^ nonce ^ (static_cast<uint64_t>(guard_id) << 32U) ^ word;
    value ^= value >> 27U;
    value *= 0xD6E8FEB86659FD93ULL;
    return value ^ (value >> 31U);
}

SIMT_CROSS_CORE_U2_INLINE uint64_t ExpectedWordsPerBatch() {
    return static_cast<uint64_t>(PayloadWrittenWords(TaskKind::Qk)) + PayloadWrittenWords(TaskKind::Sf) +
           PayloadWrittenWords(TaskKind::Pv) + PayloadWrittenWords(TaskKind::Up);
}

static_assert(g0::kTasksPerBatch == 5U && g0::kKernelsPerBatch == 4U, "U2 full-PA batch shape changed");
static_assert(
    1U * g0::kTasksPerBatch == 5U && 1U * g0::kKernelsPerBatch == 4U && 256U * g0::kTasksPerBatch == 1280U &&
        256U * g0::kKernelsPerBatch == 1024U && 256U * g0::kTasksPerBatch <= g0::kMaxTasks,
    "U2 B1/B256 task capacity changed"
);
static_assert(ubuf_staging::kRegionBytes == 4608U && ubuf_staging::kRegionBytes <= 8192U, "U2 UBUF budget changed");
static_assert(kBuilderCount == g0::kDefaultBuilderCount && kBuilderCount == 1U, "U2 must use one AIV0 builder");
static_assert(kMaxTransportReports == 1024U, "U2 report capacity changed");
static_assert(sizeof(U2Control) == g0::kCacheLineBytes, "U2 control ABI changed");
static_assert(sizeof(U2Guard) == g0::kCacheLineBytes, "U2 guard ABI changed");
static_assert(sizeof(U2TaskStagingReport) == g0::kCacheLineBytes, "U2 staging report ABI changed");
static_assert(sizeof(AtomicLine) == g0::kCacheLineBytes, "U2 slot atomic must own one cache line");
static_assert(offsetof(U2FullPaState, full_pa) == 0U, "U2 FullPaState must remain at device-state offset zero");
static_assert(
    offsetof(U2StagingState, slot_states) % g0::kCacheLineBytes == 0U &&
        offsetof(U2StagingState, reports) % g0::kCacheLineBytes == 0U &&
        sizeof(U2StagingState) % g0::kCacheLineBytes == 0U && sizeof(U2FullPaState) % g0::kCacheLineBytes == 0U,
    "all U2 sidecar regions must remain cache-line aligned"
);
#if !defined(__CCE_AICORE__)
static_assert(
    SlotForTask(1U) == 1U && SlotForTask(2U) == 2U && SlotForTask(3U) == 3U && SlotForTask(4U) == 0U &&
        SlotForTask(6U) == 2U && SlotForTask(9U) == 1U && ExpectedGeneration(9U) == 1U,
    "U2 ordered task-to-slot mapping changed"
);
static_assert(
    TaskForSlotGeneration(1U, 0U) == 1U && TaskForSlotGeneration(2U, 1U) == 6U && TaskForSlotGeneration(1U, 1U) == 9U,
    "U2 slot/generation inverse mapping changed"
);
static_assert(AnchorBit(1U) == 0x1U && AnchorBit(4U) == 0x8U && kAnchorMask == 0xFU, "U2 anchor bits changed");
static_assert(
    PayloadWrittenWords(TaskKind::Qk) == 74U && PayloadWrittenWords(TaskKind::Sf) == 76U &&
        PayloadWrittenWords(TaskKind::Pv) == 75U && PayloadWrittenWords(TaskKind::Up) == 124U &&
        ExpectedWordsPerBatch() == 349U,
    "U2 full-PA payload word counts changed"
);
static_assert(
    PayloadBytes(TaskKind::Qk) == 592U && PayloadBytes(TaskKind::Sf) == 604U && PayloadBytes(TaskKind::Pv) == 596U &&
        PayloadBytes(TaskKind::Up) == 988U && PayloadLines(TaskKind::Qk) == 10U && PayloadLines(TaskKind::Sf) == 10U &&
        PayloadLines(TaskKind::Pv) == 10U && PayloadLines(TaskKind::Up) == 16U,
    "U2 full-PA payload byte/line counts changed"
);
static_assert(
    PayloadWrittenWords(TaskKind::Qk) * sizeof(uint64_t) - PayloadBytes(TaskKind::Qk) == 0U &&
        PayloadWrittenWords(TaskKind::Sf) * sizeof(uint64_t) - PayloadBytes(TaskKind::Sf) == 4U &&
        PayloadWrittenWords(TaskKind::Pv) * sizeof(uint64_t) - PayloadBytes(TaskKind::Pv) == 4U &&
        PayloadWrittenWords(TaskKind::Up) * sizeof(uint64_t) - PayloadBytes(TaskKind::Up) == 4U,
    "U2 partial-tail word accounting changed"
);
static_assert(
    SlotStateValid(ubuf_staging::SlotBusyState(0U, 1U), 1U, 1U) &&
        !SlotStateValid(ubuf_staging::SlotBusyState(0U, 9U), 1U, 2U) &&
        SlotStateValid(ubuf_staging::SlotFreeState(256U), 0U, 256U),
    "U2 ordered slot-state validation changed"
);
#endif

#undef SIMT_CROSS_CORE_U2_INLINE
#undef SIMT_CROSS_CORE_U2_SIMT_INLINE

}  // namespace pa_scheduler::simt_cross_core::u2

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_U2_FULL_PA_H
