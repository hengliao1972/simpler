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

#pragma once

#include <cstddef>
#include <cstdint>

#include "dist_engine/common/cross_core_exec_protocol.h"
#include "fdwic_build_identity.h"

namespace fdwic::cross_core {

constexpr uint32_t kCrossMapCapacity = 16384;
constexpr uint32_t kCrossMapBucketCapacity = static_cast<uint32_t>(PTO_FDWIC_TENSORMAP_RING_CAP);
constexpr uint32_t kCrossMapBuckets = kCrossMapCapacity / kCrossMapBucketCapacity;
constexpr uint32_t kCrossMapBucketMask = kCrossMapBuckets - 1U;
constexpr uint32_t kCrossMapBucketSlotMask = kCrossMapBucketCapacity - 1U;

constexpr uint32_t CrossMapConstexprLog2(uint32_t value) {
    return value <= 1U ? 0U : 1U + CrossMapConstexprLog2(value >> 1U);
}

constexpr uint32_t kCrossMapBucketShift = CrossMapConstexprLog2(kCrossMapBuckets);

static_assert(kCrossMapBucketCapacity >= 32U && kCrossMapBucketCapacity <= kCrossMapCapacity);
static_assert((kCrossMapBucketCapacity & (kCrossMapBucketCapacity - 1U)) == 0);
static_assert(kCrossMapCapacity % kCrossMapBucketCapacity == 0);
static_assert((kCrossMapBuckets & (kCrossMapBuckets - 1U)) == 0);

struct CrossMapValue {
    uint64_t buffer_address;
    uint64_t lo;
    uint64_t hi;
    int32_t producer;
    uint32_t reserved;
};

struct alignas(kExecCacheLineBytes) CrossMapPayload {
    CrossMapValue value;
    uint8_t padding[kExecCacheLineBytes - sizeof(CrossMapValue)];
};

struct alignas(kExecCacheLineBytes) CrossMapSlot {
    CrossMapPayload payload;
    SharedExecControl sequence;
};

struct alignas(kExecCacheLineBytes) CrossCoreTensorMapState {
    SharedExecControl tails[kCrossMapBuckets];
    CrossMapSlot slots[kCrossMapCapacity];
};

static_assert(sizeof(CrossMapValue) == 32);
static_assert(sizeof(CrossMapPayload) == kExecCacheLineBytes);
static_assert(sizeof(CrossMapSlot) == 2 * kExecCacheLineBytes);
static_assert(offsetof(CrossMapSlot, sequence) == kExecCacheLineBytes);
static_assert(offsetof(CrossCoreTensorMapState, slots) == sizeof(SharedExecControl) * kCrossMapBuckets);

enum class MapTurnResult : uint8_t {
    Ready = 0,
    Pending = 1,
    Invalid = 2,
};

enum class MapAppendResult : uint8_t {
    Appended = 0,
    InvalidInput = 1,
    CapacityExceeded = 2,
    ProtocolConflict = 3,
};

PTO_DEVICE_FUNC inline uint32_t CrossMapHash(uint64_t address) {
#if PTO_FDWIC_TENSORMAP_RING_CAP == 16384
    (void)address;
    return 0;
#else
    address *= 0x9E3779B97F4A7C15ULL;
    return static_cast<uint32_t>(address >> (64U - kCrossMapBucketShift)) & kCrossMapBucketMask;
#endif
}

PTO_DEVICE_FUNC inline uint32_t CrossMapSlotIndex(uint32_t bucket, uint64_t cursor) {
    return bucket * kCrossMapBucketCapacity + (static_cast<uint32_t>(cursor) & kCrossMapBucketSlotMask);
}

PTO_DEVICE_FUNC inline bool CrossMapRegionsOverlap(const CrossMapValue &left, const CrossMapValue &right) {
    return left.buffer_address == right.buffer_address && left.lo < right.hi && right.lo < left.hi;
}

template <typename Ops>
PTO_DEVICE_FUNC MapTurnResult InspectMapTaskTurn(__gm__ volatile int64_t &predecessor, int32_t task_id) {
    if (task_id <= 0) return task_id == 0 ? MapTurnResult::Ready : MapTurnResult::Invalid;
    const int64_t observed = Ops::Load(&predecessor);
    if (observed == task_id - 1) return MapTurnResult::Ready;
    return observed == -1 ? MapTurnResult::Pending : MapTurnResult::Invalid;
}

template <typename Ops>
PTO_DEVICE_FUNC bool PublishMapTaskCompletion(__gm__ volatile int64_t &completion, int32_t task_id) {
    if (task_id < 0) return false;
    return Ops::CompareExchange(&completion, -1, task_id) == -1;
}

template <typename Ops>
PTO_DEVICE_FUNC bool
ReadCrossMapSlot(__gm__ CrossCoreTensorMapState &map, uint32_t bucket, uint64_t cursor, CrossMapValue &snapshot) {
    if (bucket >= kCrossMapBuckets || cursor >= kCrossMapBucketCapacity) return false;
    __gm__ CrossMapSlot &slot = map.slots[CrossMapSlotIndex(bucket, cursor)];
    const int64_t expected = static_cast<int64_t>(cursor);
    if (Ops::Load(&slot.sequence.state) != expected) return false;
    Ops::InvalidateRegion(&slot.payload, sizeof(slot.payload));
    snapshot.buffer_address = slot.payload.value.buffer_address;
    snapshot.lo = slot.payload.value.lo;
    snapshot.hi = slot.payload.value.hi;
    snapshot.producer = slot.payload.value.producer;
    snapshot.reserved = slot.payload.value.reserved;
    return Ops::Load(&slot.sequence.state) == expected && snapshot.lo < snapshot.hi && snapshot.producer >= 0 &&
           snapshot.reserved == 0;
}

template <typename Ops>
PTO_DEVICE_FUNC int32_t LookupCrossMap(
    __gm__ CrossCoreTensorMapState &map, const CrossMapValue &query, int32_t consumer_task, int32_t history,
    bool &protocol_ok
) {
    protocol_ok = false;
    if (consumer_task < 0 || history < 0 || query.lo >= query.hi) return -1;
    const uint32_t bucket = CrossMapHash(query.buffer_address);
    const int64_t signed_tail = Ops::Load(&map.tails[bucket].state);
    if (signed_tail < 0 || static_cast<uint64_t>(signed_tail) > kCrossMapBucketCapacity) return -1;
    const int32_t lower = consumer_task > history ? consumer_task - history : 0;
    int32_t best = -1;
    for (uint64_t cursor = 0; cursor < static_cast<uint64_t>(signed_tail); ++cursor) {
        CrossMapValue candidate{};
        if (!ReadCrossMapSlot<Ops>(map, bucket, cursor, candidate)) return -1;
        if (candidate.producer >= lower && candidate.producer < consumer_task &&
            CrossMapRegionsOverlap(candidate, query) && candidate.producer > best) {
            best = candidate.producer;
        }
    }
    protocol_ok = true;
    return best;
}

PTO_DEVICE_FUNC inline uint32_t
EarlierCrossMapEntriesInBucket(const CrossMapValue entries[], uint32_t index, uint32_t bucket) {
    uint32_t earlier = 0;
    for (uint32_t previous = 0; previous < index; ++previous) {
        if (CrossMapHash(entries[previous].buffer_address) == bucket) ++earlier;
    }
    return earlier;
}

template <typename Ops>
PTO_DEVICE_FUNC MapAppendResult AppendCrossMapTask(
    __gm__ CrossCoreTensorMapState &map, const CrossMapValue entries[], uint32_t count, int32_t task_id
) {
    if (task_id < 0 || count > kExecMaxTensors || (count != 0 && entries == nullptr)) {
        return MapAppendResult::InvalidInput;
    }
    for (uint32_t index = 0; index < count; ++index) {
        const CrossMapValue &entry = entries[index];
        if (entry.producer != task_id || entry.reserved != 0 || entry.lo >= entry.hi) {
            return MapAppendResult::InvalidInput;
        }
        const uint32_t bucket = CrossMapHash(entry.buffer_address);
        const int64_t tail = Ops::Load(&map.tails[bucket].state);
        if (tail < 0 || static_cast<uint64_t>(tail) > kCrossMapBucketCapacity) {
            return MapAppendResult::ProtocolConflict;
        }
        const uint32_t earlier = EarlierCrossMapEntriesInBucket(entries, index, bucket);
        const uint64_t cursor = static_cast<uint64_t>(tail) + earlier;
        if (cursor >= kCrossMapBucketCapacity) return MapAppendResult::CapacityExceeded;
        __gm__ CrossMapSlot &slot = map.slots[CrossMapSlotIndex(bucket, cursor)];
        if (Ops::Load(&slot.sequence.state) != -1) return MapAppendResult::ProtocolConflict;
    }

    for (uint32_t index = 0; index < count; ++index) {
        const CrossMapValue &entry = entries[index];
        const uint32_t bucket = CrossMapHash(entry.buffer_address);
        __gm__ SharedExecControl &tail_control = map.tails[bucket];
        const int64_t tail = Ops::Load(&tail_control.state);
        if (tail < 0 || static_cast<uint64_t>(tail) >= kCrossMapBucketCapacity) {
            return MapAppendResult::ProtocolConflict;
        }
        __gm__ CrossMapSlot &slot = map.slots[CrossMapSlotIndex(bucket, static_cast<uint64_t>(tail))];
        slot.payload.value.buffer_address = entry.buffer_address;
        slot.payload.value.lo = entry.lo;
        slot.payload.value.hi = entry.hi;
        slot.payload.value.producer = entry.producer;
        slot.payload.value.reserved = 0;
        Ops::FlushRegion(&slot.payload, sizeof(slot.payload));
        if (Ops::CompareExchange(&slot.sequence.state, -1, tail) != -1) {
            return MapAppendResult::ProtocolConflict;
        }
        if (Ops::CompareExchange(&tail_control.state, tail, tail + 1) != tail) {
            return MapAppendResult::ProtocolConflict;
        }
    }
    return MapAppendResult::Appended;
}

}  // namespace fdwic::cross_core
