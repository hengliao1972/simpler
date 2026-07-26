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

// preflight 不能把“容量不足”和“共享状态已经损坏”压成同一个 bool。
// exact-turn 下容量不足不会靠等待自行消失，接入 Submit 后应转成结构化
// TensorMap capacity fatal；ProtocolError 则必须走协议错误收敛。
enum class DistSharedTensorMapAppendCheck : uint32_t {
    Ready = 0,
    CapacityBlocked = 1,
    ProtocolError = 2,
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

// Ops 是独立测试与 production AICore primitive 的唯一适配层；状态机本身
// 不访问 g_dist，也不依赖 Submit/fatal/wait，使其可在解除门禁前完整验证。
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

// lookup 的权威窗口为 [max(0,N-H),N)。-1 既可能是正常 miss，也可能是
// seq/游标协议失败，因此调用方必须同时检查 protocol_ok。
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

// 只有 committed_tasks==N 的唯一 winner 可以调用 retire/append。多个无序
// writer 同时修改同一 bucket 不属于这版有序单追加者协议。
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

// winner N 已经完成本任务 lookup，后续任务最早只会读取 N-H，因此
// inclusive 回收上界是 N-H-1。使用 int64_t 避免边界减法溢出。
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
    // task/flag 都使用 kFlagCap 有界协议。边界必须在 reclaim、head 或
    // slot 发生任何修改前拒绝，不能留到最终 publish_commit 才失败。
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

// 在写任何 slot 前完成整任务的容量、cursor 与旧 seq 检查。失败时允许
// reclaim/head 保留已证明正确的单调推进，但本 task 的 payload/seq/tail/
// commit 必须尚未发布。
template <typename Ops>
PTO_DEVICE_FUNC inline DistSharedTensorMapAppendCheck dist_shared_tensor_map_check_task_append_impl(
    __gm__ SharedTensorMapState &map, const SharedTensorMapValue *entries, uint32_t count, int32_t current_task,
    int64_t reclaim_upto
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
        // append 最终要发布 tail=cursor+1，因此 INT64_MAX 本身也不可写。
        if (cursor >= static_cast<uint64_t>(INT64_MAX)) {
            return DistSharedTensorMapAppendCheck::ProtocolError;
        }
        __gm__ SharedTensorMapSlot &slot = map.slots[dist_shared_tensor_map_slot_index(bucket, cursor)];
        const int64_t expected_old = cursor < kMapBucketCapacity ? kSharedTensorMapInvalidSequence :
                                                                   static_cast<int64_t>(cursor - kMapBucketCapacity);
        if (Ops::Load(&slot.sequence.v) != expected_old) {
            return DistSharedTensorMapAppendCheck::ProtocolError;
        }
    }
    return DistSharedTensorMapAppendCheck::Ready;
}

template <typename Ops>
PTO_DEVICE_FUNC inline bool dist_shared_tensor_map_append_prepared_entry_impl(
    __gm__ SharedTensorMapState &map, const SharedTensorMapValue &entry, int32_t current_task
) {
    if (entry.producer != current_task || entry.reserved != 0 || entry.lo >= entry.hi) {
        return false;
    }
    const uint32_t bucket = dist_tensor_map_hash(entry.buf_addr);
    __gm__ SharedTensorMapBucketState &controls = map.buckets[bucket];
    const int64_t head = Ops::Load(&controls.head.v);
    const int64_t tail = Ops::Load(&controls.tail.v);
    if (head < 0 || tail < head || static_cast<uint64_t>(tail - head) >= kMapBucketCapacity) {
        return false;
    }
    const uint64_t cursor = static_cast<uint64_t>(tail);
    if (cursor >= static_cast<uint64_t>(INT64_MAX)) {
        return false;
    }
    __gm__ SharedTensorMapSlot &slot = map.slots[dist_shared_tensor_map_slot_index(bucket, cursor)];
    const int64_t expected_old = cursor < kMapBucketCapacity ? kSharedTensorMapInvalidSequence :
                                                               static_cast<int64_t>(cursor - kMapBucketCapacity);
    // WRITING 是独占 ownership 状态。claim 成功后若后续 CAS 仍失败，
    // 说明 exact-turn 协议已损坏：保留现场并由接入层 fatal 收敛，不能
    // 把已覆写 payload 的槽回滚成旧 lap seq。
    const int64_t before_claim = Ops::CompareExchange(&slot.sequence.v, expected_old, kSharedTensorMapWritingSequence);
    if (before_claim != expected_old) {
        return false;
    }

    Ops::InvalidateRegion(&slot.payload, sizeof(slot.payload));
    slot.payload.value.buf_addr = entry.buf_addr;
    slot.payload.value.lo = entry.lo;
    slot.payload.value.hi = entry.hi;
    slot.payload.value.producer = entry.producer;
    slot.payload.value.reserved = 0;
    // padding 不承载协议，不为填充字节增加 scalar store。
    Ops::FlushRegion(&slot.payload, sizeof(slot.payload));

    const int64_t before_publish =
        Ops::CompareExchange(&slot.sequence.v, kSharedTensorMapWritingSequence, static_cast<int64_t>(cursor));
    if (before_publish != kSharedTensorMapWritingSequence) {
        return false;
    }
    const int64_t previous_tail = Ops::CompareExchange(&controls.tail.v, tail, tail + 1);
    return previous_tail == tail;
}

template <typename Ops>
PTO_DEVICE_FUNC inline bool dist_shared_tensor_map_append_prepared_task_impl(
    __gm__ SharedTensorMapState &map, const SharedTensorMapValue *entries, uint32_t count, int32_t current_task
) {
    if (count > MAX_TENSOR_ARGS || (count != 0 && entries == nullptr) ||
        !dist_shared_tensor_map_has_exact_turn_impl<Ops>(map, current_task)) {
        return false;
    }
    for (uint32_t index = 0; index < count; ++index) {
        if (!dist_shared_tensor_map_append_prepared_entry_impl<Ops>(map, entries[index], current_task)) {
            return false;
        }
    }
    return true;
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

// 这些 concrete wrappers 强制 Host/CPU-sim/CCEC 三类编译器都实例化同一套
// production primitive；S2 尚未从 facade 调用它们。
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

PTO_DEVICE_FUNC inline DistSharedTensorMapAppendCheck dist_shared_tensor_map_check_task_append(
    __gm__ SharedTensorMapState &map, const SharedTensorMapValue *entries, uint32_t count, int32_t current_task,
    int64_t reclaim_upto
) {
    return dist_shared_tensor_map_check_task_append_impl<DistSharedTensorMapAicoreOps>(
        map, entries, count, current_task, reclaim_upto
    );
}

PTO_DEVICE_FUNC inline bool dist_shared_tensor_map_append_prepared_task(
    __gm__ SharedTensorMapState &map, const SharedTensorMapValue *entries, uint32_t count, int32_t current_task
) {
    return dist_shared_tensor_map_append_prepared_task_impl<DistSharedTensorMapAicoreOps>(
        map, entries, count, current_task
    );
}

PTO_DEVICE_FUNC inline bool
dist_shared_tensor_map_publish_commit(__gm__ SharedTensorMapState &map, int32_t current_task) {
    return dist_shared_tensor_map_publish_commit_impl<DistSharedTensorMapAicoreOps>(map, current_task);
}

}  // namespace
