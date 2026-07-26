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

#include "dist_engine/common/state.h"
#include "dist_engine/aicore/primitive.h"
#include "dist_engine/aicore/tensor_map_common.h"
#include "dist_engine/common/atomic.h"

namespace {

// Preflight must distinguish capacity exhaustion from corrupted shared state.
// Capacity exhaustion cannot resolve by waiting under exact-turn and must map
// to the structured TensorMap capacity fatal; protocol errors require their
// own convergence path.
enum class DistSharedTensorMapAppendCheck : uint32_t {
    Ready = 0,
    CapacityBlocked = 1,
    ProtocolError = 2,
};

// The task adapter must distinguish rejection before this call publishes
// current-task data from failure after acquiring slot ownership. The latter
// cannot be rolled back or retried: preserve the evidence and fail-stop the
// run. ProtocolError does not claim the whole sidecar is clean; a competing
// writer may already have corrupted shared control state.
enum class DistSharedTensorMapEntryPublishResult : uint32_t {
    Published = 0,
    ProtocolError = 1,
    PartialPublish = 2,
};

enum class DistSharedTensorMapTaskPublishResult : uint32_t {
    Committed = 0,
    CapacityBlocked = 1,
    ProtocolError = 2,
    PartialPublish = 3,
};

PTO_DEVICE_FUNC inline uint32_t dist_shared_tensor_map_slot_index(uint32_t bucket, uint64_t cursor) {
    return dist_tensor_map_slot_index(bucket, cursor);
}

template <typename TensorRef>
PTO_DEVICE_FUNC inline SharedTensorMapValue
dist_shared_tensor_map_make_value(const TensorRef &tensor, int32_t producer) {
    SharedTensorMapValue value{};
    dist_tensor_map_byte_range(tensor, value.buf_addr, value.lo, value.hi);
    value.producer = producer;
    value.reserved = 0;
    return value;
}

// Ops is the only seam between standalone tests and production AICore
// primitives. The state machine does not access g_dist or depend on
// Submit/fatal/wait, so it can be verified before the backend gate is opened.
template <typename Ops>
PTO_DEVICE_FUNC inline bool dist_shared_tensor_map_read_slot_impl(
    __gm__ SharedTensorMapState &map, uint32_t bucket, uint64_t cursor, SharedTensorMapValue &snapshot
) {
    if (bucket >= kMapBuckets || cursor > static_cast<uint64_t>(INT64_MAX)) {
        return false;
    }
    __gm__ SharedTensorMapSlot &slot = map.slots[dist_shared_tensor_map_slot_index(bucket, cursor)];
    const int64_t expected = static_cast<int64_t>(cursor);
    if (Ops::Load(&slot.sequence.v) != expected) {
        return false;
    }
    Ops::InvalidateRegion(&slot.payload, sizeof(slot.payload));
    snapshot.buf_addr = slot.payload.value.buf_addr;
    snapshot.lo = slot.payload.value.lo;
    snapshot.hi = slot.payload.value.hi;
    snapshot.producer = slot.payload.value.producer;
    snapshot.reserved = slot.payload.value.reserved;
    if (Ops::Load(&slot.sequence.v) != expected) {
        return false;
    }
    return snapshot.producer >= 0 && snapshot.reserved == 0 && snapshot.lo < snapshot.hi;
}

// The authoritative lookup window is [max(0, N-H), N). A -1 result can mean a
// normal miss or a sequence/cursor protocol failure, so callers must also
// inspect protocol_ok.
template <typename Ops>
PTO_DEVICE_FUNC inline int32_t dist_shared_tensor_map_lookup_region_impl(
    __gm__ SharedTensorMapState &map, const SharedTensorMapValue &query, int32_t current_task, int32_t history,
    bool &protocol_ok
) {
    protocol_ok = false;
    if (current_task < 0 || current_task >= kFlagCap || history < 0 || query.lo >= query.hi) {
        return -1;
    }
    const uint32_t bucket = dist_tensor_map_hash(query.buf_addr);
    const int64_t head = Ops::Load(&map.buckets[bucket].head.v);
    const int64_t tail = Ops::Load(&map.buckets[bucket].tail.v);
    if (head < 0 || tail < head || static_cast<uint64_t>(tail - head) > kMapBucketCapacity) {
        return -1;
    }

    const int32_t lower = current_task > history ? current_task - history : 0;
    int32_t best = -1;
    for (uint64_t cursor = static_cast<uint64_t>(head); cursor < static_cast<uint64_t>(tail); ++cursor) {
        SharedTensorMapValue candidate{};
        if (!dist_shared_tensor_map_read_slot_impl<Ops>(map, bucket, cursor, candidate)) {
            return -1;
        }
        if (candidate.producer >= lower && candidate.producer < current_task &&
            dist_tensor_map_regions_overlap(candidate, query.buf_addr, query.lo, query.hi) &&
            candidate.producer > best) {
            best = candidate.producer;
        }
    }
    protocol_ok = true;
    return best;
}

template <typename Ops, typename TensorRef>
PTO_DEVICE_FUNC inline int32_t dist_shared_tensor_map_lookup_tensor_impl(
    __gm__ SharedTensorMapState &map, const TensorRef &tensor, int32_t current_task, int32_t history, bool &protocol_ok
) {
    const SharedTensorMapValue query = dist_shared_tensor_map_make_value(tensor, -1);
    return dist_shared_tensor_map_lookup_region_impl<Ops>(map, query, current_task, history, protocol_ok);
}

// Only the unique winner observing committed_tasks==N may retire or append.
// Concurrent unordered writers touching one bucket are outside this ordered
// single-appender protocol.
template <typename Ops>
PTO_DEVICE_FUNC inline bool
dist_shared_tensor_map_retire_bucket_impl(__gm__ SharedTensorMapState &map, uint32_t bucket, int64_t reclaim_upto) {
    if (bucket >= kMapBuckets || reclaim_upto < -1) {
        return false;
    }
    __gm__ SharedTensorMapBucketState &controls = map.buckets[bucket];
    const int64_t original_head = Ops::Load(&controls.head.v);
    const int64_t tail = Ops::Load(&controls.tail.v);
    if (original_head < 0 || tail < original_head || static_cast<uint64_t>(tail - original_head) > kMapBucketCapacity) {
        return false;
    }

    int64_t head = original_head;
    while (head < tail) {
        SharedTensorMapValue value{};
        if (!dist_shared_tensor_map_read_slot_impl<Ops>(map, bucket, static_cast<uint64_t>(head), value)) {
            return false;
        }
        if (value.producer > reclaim_upto) {
            break;
        }
        ++head;
    }
    if (head == original_head) {
        return true;
    }
    const int64_t observed = Ops::CompareExchange(&controls.head.v, original_head, head);
    return observed == original_head;
}

// Winner N has completed its lookups, and later tasks can read no earlier than
// N-H, so the inclusive reclaim boundary is N-H-1. Use int64_t to avoid
// underflow at the boundary.
PTO_DEVICE_FUNC inline bool
dist_shared_tensor_map_compute_reclaim(int32_t current_task, int32_t history, int64_t &candidate) {
    if (current_task < 0 || history < 0) {
        return false;
    }
    candidate = static_cast<int64_t>(current_task) - static_cast<int64_t>(history) - 1;
    if (candidate < -1) {
        candidate = -1;
    }
    return true;
}

template <typename Ops>
PTO_DEVICE_FUNC inline bool
dist_shared_tensor_map_has_exact_turn_impl(__gm__ SharedTensorMapState &map, int32_t current_task) {
    // Tasks and flags share the kFlagCap bounded protocol. Reject an invalid
    // task before changing reclaim, head, or any slot instead of waiting until
    // publish_commit.
    return current_task >= 0 && current_task < kFlagCap && Ops::Load(&map.committed_tasks.v) == current_task;
}

template <typename Ops>
PTO_DEVICE_FUNC inline bool dist_shared_tensor_map_refresh_reclaim_impl(
    __gm__ SharedTensorMapState &map, int32_t current_task, int32_t history, int64_t &reclaim_upto
) {
    if (!dist_shared_tensor_map_has_exact_turn_impl<Ops>(map, current_task)) {
        return false;
    }
    int64_t candidate = -1;
    if (!dist_shared_tensor_map_compute_reclaim(current_task, history, candidate)) {
        return false;
    }
    const int64_t current = Ops::Load(&map.reclaim_upto.v);
    if (current < -1 || candidate < current) {
        return false;
    }
    if (candidate == current) {
        reclaim_upto = current;
        return true;
    }
    const int64_t observed = Ops::CompareExchange(&map.reclaim_upto.v, current, candidate);
    if (observed != current) {
        return false;
    }
    reclaim_upto = candidate;
    return true;
}

PTO_DEVICE_FUNC inline uint32_t
dist_shared_tensor_map_earlier_entries_in_bucket(const SharedTensorMapValue *entries, uint32_t index, uint32_t bucket) {
    uint32_t earlier = 0;
    for (uint32_t previous = 0; previous < index; ++previous) {
        if (dist_tensor_map_hash(entries[previous].buf_addr) == bucket) {
            ++earlier;
        }
    }
    return earlier;
}

// Validate capacity, cursors, and old sequences for the whole task before
// writing any slot. A failed check may retain a proven monotonic reclaim/head
// advance, but must not publish this task's payload, sequence, tail, or commit.
template <typename Ops>
PTO_DEVICE_FUNC inline DistSharedTensorMapAppendCheck dist_shared_tensor_map_check_task_append_impl(
    __gm__ SharedTensorMapState &map, const SharedTensorMapValue *entries, uint32_t count, int32_t current_task,
    int64_t reclaim_upto, uint64_t *planned_cursors = nullptr
) {
    if (current_task < 0 || count > MAX_TENSOR_ARGS || (count != 0 && entries == nullptr) ||
        !dist_shared_tensor_map_has_exact_turn_impl<Ops>(map, current_task) ||
        Ops::Load(&map.reclaim_upto.v) != reclaim_upto) {
        return DistSharedTensorMapAppendCheck::ProtocolError;
    }
    for (uint32_t index = 0; index < count; ++index) {
        const SharedTensorMapValue &entry = entries[index];
        if (entry.producer != current_task || entry.reserved != 0 || entry.lo >= entry.hi) {
            return DistSharedTensorMapAppendCheck::ProtocolError;
        }
        const uint32_t bucket = dist_tensor_map_hash(entry.buf_addr);
        if (!dist_shared_tensor_map_retire_bucket_impl<Ops>(map, bucket, reclaim_upto)) {
            return DistSharedTensorMapAppendCheck::ProtocolError;
        }
        const int64_t head = Ops::Load(&map.buckets[bucket].head.v);
        const int64_t tail = Ops::Load(&map.buckets[bucket].tail.v);
        if (head < 0 || tail < head || static_cast<uint64_t>(tail - head) > kMapBucketCapacity) {
            return DistSharedTensorMapAppendCheck::ProtocolError;
        }
        const uint32_t earlier = dist_shared_tensor_map_earlier_entries_in_bucket(entries, index, bucket);
        const uint64_t occupied = static_cast<uint64_t>(tail - head) + earlier;
        if (occupied >= kMapBucketCapacity) {
            return DistSharedTensorMapAppendCheck::CapacityBlocked;
        }
        const uint64_t cursor = static_cast<uint64_t>(tail) + earlier;
        // Append eventually publishes tail=cursor+1, so INT64_MAX is invalid.
        if (cursor >= static_cast<uint64_t>(INT64_MAX)) {
            return DistSharedTensorMapAppendCheck::ProtocolError;
        }
        __gm__ SharedTensorMapSlot &slot = map.slots[dist_shared_tensor_map_slot_index(bucket, cursor)];
        const int64_t expected_old = cursor < kMapBucketCapacity ? kSharedTensorMapInvalidSequence :
                                                                   static_cast<int64_t>(cursor - kMapBucketCapacity);
        if (Ops::Load(&slot.sequence.v) != expected_old) {
            return DistSharedTensorMapAppendCheck::ProtocolError;
        }
        if (planned_cursors != nullptr) {
            planned_cursors[index] = cursor;
        }
    }
    return DistSharedTensorMapAppendCheck::Ready;
}

template <typename Ops>
PTO_DEVICE_FUNC inline DistSharedTensorMapEntryPublishResult dist_shared_tensor_map_publish_prepared_entry_impl(
    __gm__ SharedTensorMapState &map, const SharedTensorMapValue &entry, int32_t current_task, uint64_t planned_cursor
) {
    if (entry.producer != current_task || entry.reserved != 0 || entry.lo >= entry.hi ||
        planned_cursor >= static_cast<uint64_t>(INT64_MAX)) {
        return DistSharedTensorMapEntryPublishResult::ProtocolError;
    }
    const uint32_t bucket = dist_tensor_map_hash(entry.buf_addr);
    __gm__ SharedTensorMapBucketState &controls = map.buckets[bucket];
    const int64_t head = Ops::Load(&controls.head.v);
    const int64_t tail = Ops::Load(&controls.tail.v);
    if (head < 0 || tail < head || static_cast<uint64_t>(tail) != planned_cursor ||
        static_cast<uint64_t>(tail - head) >= kMapBucketCapacity) {
        return DistSharedTensorMapEntryPublishResult::ProtocolError;
    }
    __gm__ SharedTensorMapSlot &slot = map.slots[dist_shared_tensor_map_slot_index(bucket, planned_cursor)];
    const int64_t expected_old = planned_cursor < kMapBucketCapacity ?
                                     kSharedTensorMapInvalidSequence :
                                     static_cast<int64_t>(planned_cursor - kMapBucketCapacity);
    // WRITING is the exclusive ownership state. A later CAS failure after a
    // successful claim means the exact-turn protocol is broken. Preserve the
    // evidence for fatal convergence; never roll an overwritten payload back
    // to the previous lap sequence.
    const int64_t before_claim = Ops::CompareExchange(&slot.sequence.v, expected_old, kSharedTensorMapWritingSequence);
    if (before_claim != expected_old) {
        return DistSharedTensorMapEntryPublishResult::ProtocolError;
    }

    Ops::InvalidateRegion(&slot.payload, sizeof(slot.payload));
    slot.payload.value.buf_addr = entry.buf_addr;
    slot.payload.value.lo = entry.lo;
    slot.payload.value.hi = entry.hi;
    slot.payload.value.producer = entry.producer;
    slot.payload.value.reserved = 0;
    // Padding carries no protocol state; avoid scalar stores for padding.
    Ops::FlushRegion(&slot.payload, sizeof(slot.payload));

    const int64_t before_publish =
        Ops::CompareExchange(&slot.sequence.v, kSharedTensorMapWritingSequence, static_cast<int64_t>(planned_cursor));
    if (before_publish != kSharedTensorMapWritingSequence) {
        return DistSharedTensorMapEntryPublishResult::PartialPublish;
    }
    const int64_t previous_tail = Ops::CompareExchange(
        &controls.tail.v, static_cast<int64_t>(planned_cursor), static_cast<int64_t>(planned_cursor) + 1
    );
    return previous_tail == static_cast<int64_t>(planned_cursor) ?
               DistSharedTensorMapEntryPublishResult::Published :
               DistSharedTensorMapEntryPublishResult::PartialPublish;
}

template <typename Ops>
PTO_DEVICE_FUNC inline bool
dist_shared_tensor_map_publish_commit_impl(__gm__ SharedTensorMapState &map, int32_t current_task) {
    if (current_task < 0 || current_task >= kFlagCap) {
        return false;
    }
    const int64_t previous =
        Ops::CompareExchange(&map.committed_tasks.v, current_task, static_cast<int64_t>(current_task) + 1);
    return previous == current_task;
}

template <typename Ops>
PTO_DEVICE_FUNC inline DistSharedTensorMapTaskPublishResult dist_shared_tensor_map_publish_prepared_task_impl(
    __gm__ SharedTensorMapState &map, const SharedTensorMapValue *entries, const uint64_t *planned_cursors,
    uint32_t count, int32_t current_task
) {
    if (count > MAX_TENSOR_ARGS || (count != 0 && (entries == nullptr || planned_cursors == nullptr)) ||
        !dist_shared_tensor_map_has_exact_turn_impl<Ops>(map, current_task)) {
        return DistSharedTensorMapTaskPublishResult::ProtocolError;
    }
    for (uint32_t index = 0; index < count; ++index) {
        const DistSharedTensorMapEntryPublishResult result = dist_shared_tensor_map_publish_prepared_entry_impl<Ops>(
            map, entries[index], current_task, planned_cursors[index]
        );
        if (result == DistSharedTensorMapEntryPublishResult::Published) {
            continue;
        }
        if (result == DistSharedTensorMapEntryPublishResult::PartialPublish || index != 0) {
            return DistSharedTensorMapTaskPublishResult::PartialPublish;
        }
        return DistSharedTensorMapTaskPublishResult::ProtocolError;
    }
    if (!dist_shared_tensor_map_publish_commit_impl<Ops>(map, current_task)) {
        return count == 0 ? DistSharedTensorMapTaskPublishResult::ProtocolError :
                            DistSharedTensorMapTaskPublishResult::PartialPublish;
    }
    return DistSharedTensorMapTaskPublishResult::Committed;
}

// Callers must complete lookup before entering this function. It only handles
// reclaim, whole-task preflight, ordered publication, and the final commit.
// A zero-entry task still advances the contiguous task frontier. Any partial
// publication after preflight is unrecoverable and must reach Submit without
// being downgraded to an ordinary protocol rejection.
template <typename Ops>
PTO_DEVICE_FUNC inline DistSharedTensorMapTaskPublishResult dist_shared_tensor_map_publish_task_impl(
    __gm__ SharedTensorMapState &map, const SharedTensorMapValue *entries, uint32_t count, int32_t current_task,
    int32_t history
) {
    if (current_task < 0 || current_task >= kFlagCap || history < 0 || count > MAX_TENSOR_ARGS ||
        (count != 0 && entries == nullptr)) {
        return DistSharedTensorMapTaskPublishResult::ProtocolError;
    }
    for (uint32_t index = 0; index < count; ++index) {
        const SharedTensorMapValue &entry = entries[index];
        if (entry.producer != current_task || entry.reserved != 0 || entry.lo >= entry.hi) {
            return DistSharedTensorMapTaskPublishResult::ProtocolError;
        }
    }

    int64_t reclaim_upto = -2;
    if (!dist_shared_tensor_map_refresh_reclaim_impl<Ops>(map, current_task, history, reclaim_upto)) {
        return DistSharedTensorMapTaskPublishResult::ProtocolError;
    }
    uint64_t planned_cursors[MAX_TENSOR_ARGS];
    const DistSharedTensorMapAppendCheck check = dist_shared_tensor_map_check_task_append_impl<Ops>(
        map, entries, count, current_task, reclaim_upto, planned_cursors
    );
    if (check == DistSharedTensorMapAppendCheck::CapacityBlocked) {
        return DistSharedTensorMapTaskPublishResult::CapacityBlocked;
    }
    if (check != DistSharedTensorMapAppendCheck::Ready) {
        return DistSharedTensorMapTaskPublishResult::ProtocolError;
    }

    return dist_shared_tensor_map_publish_prepared_task_impl<Ops>(map, entries, planned_cursors, count, current_task);
}

struct DistSharedTensorMapAicoreOps {
    PTO_DEVICE_FUNC static int64_t Load(__gm__ volatile int64_t *address) { return atomic_load(*address); }

    PTO_DEVICE_FUNC static int64_t
    CompareExchange(__gm__ volatile int64_t *address, int64_t expected, int64_t desired) {
        return atomic_compare_exchange(*address, expected, desired);
    }

    PTO_DEVICE_FUNC static void InvalidateRegion(__gm__ const void *address, uint64_t bytes) {
        dist_aicore_invalidate_region(address, bytes);
    }

    PTO_DEVICE_FUNC static void FlushRegion(__gm__ void *address, uint64_t bytes) {
        dist_aicore_flush_region(address, bytes);
    }
};

// These concrete wrappers force Host, CPU-sim, and CCEC to instantiate the
// same production primitives. The S2 facade does not call them yet.
PTO_DEVICE_FUNC inline int32_t dist_shared_tensor_map_lookup_region(
    __gm__ SharedTensorMapState &map, const SharedTensorMapValue &query, int32_t current_task, int32_t history,
    bool &protocol_ok
) {
    return dist_shared_tensor_map_lookup_region_impl<DistSharedTensorMapAicoreOps>(
        map, query, current_task, history, protocol_ok
    );
}

template <typename TensorRef>
PTO_DEVICE_FUNC inline int32_t dist_shared_tensor_map_lookup_tensor(
    __gm__ SharedTensorMapState &map, const TensorRef &tensor, int32_t current_task, int32_t history, bool &protocol_ok
) {
    return dist_shared_tensor_map_lookup_tensor_impl<DistSharedTensorMapAicoreOps>(
        map, tensor, current_task, history, protocol_ok
    );
}

PTO_DEVICE_FUNC inline bool dist_shared_tensor_map_refresh_reclaim(
    __gm__ SharedTensorMapState &map, int32_t current_task, int32_t history, int64_t &reclaim_upto
) {
    return dist_shared_tensor_map_refresh_reclaim_impl<DistSharedTensorMapAicoreOps>(
        map, current_task, history, reclaim_upto
    );
}

PTO_DEVICE_FUNC inline DistSharedTensorMapTaskPublishResult dist_shared_tensor_map_publish_task(
    __gm__ SharedTensorMapState &map, const SharedTensorMapValue *entries, uint32_t count, int32_t current_task,
    int32_t history
) {
    return dist_shared_tensor_map_publish_task_impl<DistSharedTensorMapAicoreOps>(
        map, entries, count, current_task, history
    );
}

}  // namespace
