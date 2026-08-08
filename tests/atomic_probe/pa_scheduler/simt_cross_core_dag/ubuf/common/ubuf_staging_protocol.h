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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_UBUF_STAGING_PROTOCOL_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_UBUF_STAGING_PROTOCOL_H

#include <stdint.h>

#include "../../common/full_pa_exec_protocol.h"

namespace pa_scheduler::simt_cross_core::ubuf_staging {

#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_UBUF_INLINE __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_UBUF_INLINE constexpr
#endif

using g0::kCacheLineBytes;

constexpr uint32_t kWordsPerLine = kCacheLineBytes / sizeof(uint64_t);
constexpr uint32_t kSlotCount = 4U;
constexpr uint32_t kGuardLines = 1U;
constexpr uint32_t kMaxPayloadLines = 16U;
constexpr uint32_t kMaxPayloadWords = kMaxPayloadLines * kWordsPerLine;
constexpr uint32_t kAlignmentBytes = kCacheLineBytes;
constexpr uint32_t kPayloadOffsetBytes = kGuardLines * kCacheLineBytes;
constexpr uint32_t kPayloadOffsetWords = kPayloadOffsetBytes / sizeof(uint64_t);
constexpr uint32_t kGuardAfterOffsetBytes = kPayloadOffsetBytes + kMaxPayloadLines * kCacheLineBytes;
constexpr uint32_t kGuardAfterOffsetWords = kGuardAfterOffsetBytes / sizeof(uint64_t);
constexpr uint32_t kSlotStrideBytes = (kGuardLines + kMaxPayloadLines + kGuardLines) * kCacheLineBytes;
constexpr uint32_t kSlotStrideWords = kSlotStrideBytes / sizeof(uint64_t);
constexpr uint32_t kRegionBytes = kSlotCount * kSlotStrideBytes;
constexpr uint32_t kRegionWords = kRegionBytes / sizeof(uint64_t);
constexpr uint32_t kInvalidTaskId = UINT32_MAX;

enum class TransportKind : uint16_t {
    SimtUbufReadToGmWordStore = 1U,
};

struct DecodedSlotState {
    uint32_t generation;
    uint32_t task_id;
    bool busy;
    bool valid;
};

SIMT_CROSS_CORE_UBUF_INLINE uint64_t SlotFreeState(uint32_t generation) {
    return static_cast<uint64_t>(generation) << 32U;
}

SIMT_CROSS_CORE_UBUF_INLINE uint64_t SlotBusyState(uint32_t generation, uint32_t task_id) {
    return (static_cast<uint64_t>(generation) << 32U) | (static_cast<uint64_t>(task_id) + 1U);
}

SIMT_CROSS_CORE_UBUF_INLINE uint32_t SlotStateGeneration(uint64_t raw_state) {
    return static_cast<uint32_t>(raw_state >> 32U);
}

SIMT_CROSS_CORE_UBUF_INLINE uint32_t SlotStateTaskPlusOne(uint64_t raw_state) {
    return static_cast<uint32_t>(raw_state);
}

SIMT_CROSS_CORE_UBUF_INLINE bool SlotStateIsFree(uint64_t raw_state) { return SlotStateTaskPlusOne(raw_state) == 0U; }

SIMT_CROSS_CORE_UBUF_INLINE uint32_t SlotStateTaskId(uint64_t raw_state) {
    const uint32_t task_plus_one = SlotStateTaskPlusOne(raw_state);
    return task_plus_one == 0U ? kInvalidTaskId : task_plus_one - 1U;
}

SIMT_CROSS_CORE_UBUF_INLINE DecodedSlotState DecodeSlotState(int64_t raw_state, uint32_t task_count) {
    const uint64_t raw = static_cast<uint64_t>(raw_state);
    const uint32_t task_plus_one = SlotStateTaskPlusOne(raw);
    const bool busy = task_plus_one != 0U;
    return DecodedSlotState{
        SlotStateGeneration(raw),
        busy ? task_plus_one - 1U : kInvalidTaskId,
        busy,
        !busy || task_plus_one <= task_count,
    };
}

SIMT_CROSS_CORE_UBUF_INLINE uint32_t SlotBaseWords(uint32_t slot_id) { return slot_id * kSlotStrideWords; }

SIMT_CROSS_CORE_UBUF_INLINE uint32_t SlotPayloadWords(uint32_t slot_id) {
    return SlotBaseWords(slot_id) + kPayloadOffsetWords;
}

SIMT_CROSS_CORE_UBUF_INLINE uint32_t SlotGuardAfterWords(uint32_t slot_id) {
    return SlotBaseWords(slot_id) + kGuardAfterOffsetWords;
}

static_assert(kCacheLineBytes == 64U && kWordsPerLine == 8U, "UBUF staging cache-line ABI changed");
static_assert(kSlotCount == 4U && kMaxPayloadLines == 16U && kMaxPayloadWords == 128U, "UBUF capacity changed");
static_assert(
    kPayloadOffsetBytes == 64U && kGuardAfterOffsetBytes == 1088U && kSlotStrideBytes == 1152U && kRegionBytes == 4608U,
    "guarded four-slot UBUF geometry changed"
);
static_assert(
    kPayloadOffsetBytes % kAlignmentBytes == 0U && kGuardAfterOffsetBytes % kAlignmentBytes == 0U &&
        kSlotStrideBytes % kAlignmentBytes == 0U && kRegionBytes % kAlignmentBytes == 0U,
    "all UBUF staging regions must remain cache-line aligned"
);
#if !defined(__CCE_AICORE__)
static_assert(
    SlotFreeState(7U) == (uint64_t{7U} << 32U) && SlotBusyState(7U, 31U) == ((uint64_t{7U} << 32U) | uint64_t{32U}) &&
        DecodeSlotState(static_cast<int64_t>(SlotBusyState(7U, 31U)), 128U).task_id == 31U,
    "UBUF slot state encoding changed"
);
static_assert(
    SlotBaseWords(3U) == 432U && SlotPayloadWords(3U) == 440U && SlotGuardAfterWords(3U) == 568U,
    "UBUF slot word offsets changed"
);
#endif

#undef SIMT_CROSS_CORE_UBUF_INLINE

}  // namespace pa_scheduler::simt_cross_core::ubuf_staging

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_UBUF_STAGING_PROTOCOL_H
