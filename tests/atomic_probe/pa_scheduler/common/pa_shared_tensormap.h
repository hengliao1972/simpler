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

#ifndef PA_SCHEDULER_COMMON_PA_SHARED_TENSORMAP_H
#define PA_SCHEDULER_COMMON_PA_SHARED_TENSORMAP_H

#include "pa_frontend.h"

namespace pa_scheduler {

constexpr int64_t kSharedMapEmptySeq = -1;

enum class SharedAppendCheck : uint32_t {
    Ready = 0,
    CapacityBlocked = 1,
    ProtocolError = 2,
};

PA_DEVICE uint32_t SharedTensorMapSlotIndex(uint32_t bucket, uint64_t cursor) {
    return bucket * kMapBucketCapacity +
           (static_cast<uint32_t>(cursor) & kMapBucketSlotMask);
}

PA_DEVICE bool SharedRegionOverlaps(
    const SharedRegionValue &left, const SharedRegionValue &right
) {
    return left.buffer_addr == right.buffer_addr &&
           left.lo < right.hi && right.lo < left.hi;
}

template <typename TensorReference>
PA_DEVICE SharedRegionValue MakeSharedRegionValue(
    const TensorReference &tensor, int32_t producer
) {
    SharedRegionValue value{};
    TensorByteRange(tensor, value.buffer_addr, value.lo, value.hi);
    value.producer = producer;
    value.reserved = 0;
    return value;
}

// 读侧协议固定为：原子观察绝对 seq、失效 payload、拷出本地快照、再次
// 原子观察同一 seq。任一检查失败都返回 false；上层不得把协议失败静默
// 解释成“没有 producer”。
template <typename Ops>
PA_DEVICE bool SharedReadRegionSlot(
    PA_GM SharedTensorMapSidecar &map, uint32_t bucket, uint64_t cursor,
    SharedRegionValue &snapshot
) {
    if (bucket >= kMapBuckets || cursor > static_cast<uint64_t>(INT64_MAX)) {
        return false;
    }
    PA_GM SharedRegionSlot &slot =
        map.slots[SharedTensorMapSlotIndex(bucket, cursor)];
    const int64_t expected = static_cast<int64_t>(cursor);
    if (Ops::Load(&slot.seq.value) != expected) {
        return false;
    }
    Ops::InvalidateRegion(&slot.payload, sizeof(slot.payload));
    snapshot.buffer_addr = slot.payload.value.buffer_addr;
    snapshot.lo = slot.payload.value.lo;
    snapshot.hi = slot.payload.value.hi;
    snapshot.producer = slot.payload.value.producer;
    snapshot.reserved = slot.payload.value.reserved;
    if (Ops::Load(&slot.seq.value) != expected) {
        return false;
    }
    return snapshot.producer >= 0 && snapshot.reserved == 0 &&
           snapshot.lo < snapshot.hi;
}

// lookup 的权威时间窗为 [current_task-H, current_task)。即使更快 winner
// 已经发布未来 entry，也不能把 producer>=current_task 引入本核 fanin。
template <typename Ops>
PA_DEVICE int32_t SharedLookupRegion(
    PA_GM SharedTensorMapSidecar &map, const SharedRegionValue &query,
    int32_t current_task, int32_t heap_window, bool &protocol_ok
) {
    protocol_ok = false;
    if (current_task < 0 || heap_window < 0 || query.lo >= query.hi) {
        return -1;
    }
    const uint32_t bucket = TensorMapHash(query.buffer_addr);
    const int64_t signed_head = Ops::Load(&map.buckets[bucket].head.value);
    const int64_t signed_tail = Ops::Load(&map.buckets[bucket].tail.value);
    if (signed_head < 0 || signed_tail < signed_head ||
        static_cast<uint64_t>(signed_tail - signed_head) > kMapBucketCapacity) {
        return -1;
    }

    const int32_t lower =
        current_task > heap_window ? current_task - heap_window : 0;
    int32_t best = -1;
    for (uint64_t cursor = static_cast<uint64_t>(signed_head);
         cursor < static_cast<uint64_t>(signed_tail); ++cursor) {
        SharedRegionValue candidate{};
        if (!SharedReadRegionSlot<Ops>(map, bucket, cursor, candidate)) {
            return -1;
        }
        if (candidate.producer >= lower &&
            candidate.producer < current_task &&
            SharedRegionOverlaps(candidate, query) &&
            candidate.producer > best) {
            best = candidate.producer;
        }
    }
    protocol_ok = true;
    return best;
}

template <typename Ops, typename TensorReference>
PA_DEVICE int32_t SharedLookupTensor(
    PA_GM SharedTensorMapSidecar &map, const TensorReference &tensor,
    int32_t current_task, int32_t heap_window, bool &protocol_ok
) {
    const SharedRegionValue query = MakeSharedRegionValue(tensor, -1);
    return SharedLookupRegion<Ops>(
        map, query, current_task, heap_window, protocol_ok
    );
}

// 只有持有 task-order append turn 的 winner 可以调用回收/append 原语；
// 多个无序 writer 同时更新 head/tail 不属于本实现契约。
template <typename Ops>
PA_DEVICE bool SharedRetireBucket(
    PA_GM SharedTensorMapSidecar &map, uint32_t bucket, int64_t reclaim_upto
) {
    if (bucket >= kMapBuckets || reclaim_upto < -1) {
        return false;
    }
    PA_GM SharedBucketState &controls = map.buckets[bucket];
    const int64_t original_head = Ops::Load(&controls.head.value);
    const int64_t tail = Ops::Load(&controls.tail.value);
    if (original_head < 0 || tail < original_head ||
        static_cast<uint64_t>(tail - original_head) > kMapBucketCapacity) {
        return false;
    }

    int64_t head = original_head;
    while (head < tail) {
        SharedRegionValue value{};
        if (!SharedReadRegionSlot<Ops>(
                map, bucket, static_cast<uint64_t>(head), value
            )) {
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
    const int64_t observed = Ops::Exchange(&controls.head.value, head);
    return observed == original_head;
}

// winner N 持有 committed_tasks==N 的唯一 append turn。它已经完成本任务
// 的全部 lookup，后续任务最早只会读取 producer>=N-H，因此 N-H-1 可按
// inclusive 上界回收。用 int64_t 计算，避免 int32_t 边界减法溢出。
PA_DEVICE bool SharedComputeOrderedReclaimCandidate(
    int32_t current_task, int32_t heap_window, int64_t &candidate
) {
    if (current_task < 0 || heap_window < 0) {
        return false;
    }
    candidate =
        static_cast<int64_t>(current_task) -
        static_cast<int64_t>(heap_window) - 1;
    if (candidate < -1) {
        candidate = -1;
    }
    return true;
}

template <typename Ops>
PA_DEVICE bool SharedHasExactTaskTurn(
    PA_GM SharedTensorMapSidecar &map, int32_t current_task
) {
    return current_task >= 0 &&
           Ops::Load(&map.committed_tasks.value) == current_task;
}

// 只有 exact committed turn 的 winner 可以推进 reclaim。先验证
// committed_tasks==current_task，再计算当前任务的 inclusive 回收边界；
// 逆序/陈旧 actor 在任何 head、tail 或 reclaim 写入前失败。
template <typename Ops>
PA_DEVICE bool SharedRefreshReclaimForTask(
    PA_GM SharedTensorMapSidecar &map, int32_t current_task,
    int32_t heap_window,
    int64_t &reclaim_upto
) {
    if (!SharedHasExactTaskTurn<Ops>(map, current_task)) {
        return false;
    }
    int64_t candidate = -1;
    if (!SharedComputeOrderedReclaimCandidate(
            current_task, heap_window, candidate
        )) {
        return false;
    }
    const int64_t current = Ops::Load(&map.reclaim_upto.value);
    if (current < -1 || candidate < current) {
        return false;
    }
    if (candidate == current) {
        reclaim_upto = current;
        return true;
    }
    const int64_t observed =
        Ops::Exchange(&map.reclaim_upto.value, candidate);
    if (observed != current) {
        return false;
    }
    reclaim_upto = candidate;
    return true;
}

PA_DEVICE uint32_t SharedEarlierEntriesInBucket(
    const SharedRegionValue *entries, uint32_t index, uint32_t bucket
) {
    uint32_t earlier = 0;
    for (uint32_t previous = 0; previous < index; ++previous) {
        if (TensorMapHash(entries[previous].buffer_addr) == bucket) {
            ++earlier;
        }
    }
    return earlier;
}

// 在写任何 slot 前完成整任务容量、目标 seq 与 cursor 检查。容量不足时
// 当前任务的 payload/seq/tail/commit 都不发布；检查期间按已发布边界推进
// 的陈旧 head 可以保留。随后 append 失败只可能是协议破坏，调用方应 fatal。
template <typename Ops>
PA_DEVICE SharedAppendCheck SharedCheckTaskAppend(
    PA_GM SharedTensorMapSidecar &map, const SharedRegionValue *entries,
    uint32_t count, int64_t reclaim_upto
) {
    if (count > kMaxTaskTensors) {
        return SharedAppendCheck::ProtocolError;
    }
    for (uint32_t index = 0; index < count; ++index) {
        const SharedRegionValue &entry = entries[index];
        if (entry.producer < 0 || entry.reserved != 0 ||
            entry.lo >= entry.hi) {
            return SharedAppendCheck::ProtocolError;
        }
        const uint32_t bucket = TensorMapHash(entry.buffer_addr);
        if (!SharedRetireBucket<Ops>(map, bucket, reclaim_upto)) {
            return SharedAppendCheck::ProtocolError;
        }
        const int64_t head = Ops::Load(&map.buckets[bucket].head.value);
        const int64_t tail = Ops::Load(&map.buckets[bucket].tail.value);
        if (head < 0 || tail < head) {
            return SharedAppendCheck::ProtocolError;
        }
        const uint32_t earlier =
            SharedEarlierEntriesInBucket(entries, index, bucket);
        const uint64_t occupied =
            static_cast<uint64_t>(tail - head) + earlier;
        if (occupied >= kMapBucketCapacity) {
            return SharedAppendCheck::CapacityBlocked;
        }
        const uint64_t cursor = static_cast<uint64_t>(tail) + earlier;
        if (cursor > static_cast<uint64_t>(INT64_MAX)) {
            return SharedAppendCheck::ProtocolError;
        }
        PA_GM SharedRegionSlot &slot =
            map.slots[SharedTensorMapSlotIndex(bucket, cursor)];
        const int64_t expected_old =
            cursor < kMapBucketCapacity
                ? kSharedMapEmptySeq
                : static_cast<int64_t>(cursor - kMapBucketCapacity);
        if (Ops::Load(&slot.seq.value) != expected_old) {
            return SharedAppendCheck::ProtocolError;
        }
    }
    return SharedAppendCheck::Ready;
}

template <typename Ops>
PA_DEVICE bool SharedPreflightTaskAppend(
    PA_GM SharedTensorMapSidecar &map, const SharedRegionValue *entries,
    uint32_t count, int64_t reclaim_upto
) {
    return SharedCheckTaskAppend<Ops>(
        map, entries, count, reclaim_upto
    ) == SharedAppendCheck::Ready;
}

template <typename Ops>
PA_DEVICE bool SharedAppendPreparedEntry(
    PA_GM SharedTensorMapSidecar &map, const SharedRegionValue &entry
) {
    const uint32_t bucket = TensorMapHash(entry.buffer_addr);
    PA_GM SharedBucketState &controls = map.buckets[bucket];
    const int64_t head = Ops::Load(&controls.head.value);
    const int64_t tail = Ops::Load(&controls.tail.value);
    if (head < 0 || tail < head ||
        static_cast<uint64_t>(tail - head) >= kMapBucketCapacity) {
        return false;
    }
    const uint64_t cursor = static_cast<uint64_t>(tail);
    if (cursor > static_cast<uint64_t>(INT64_MAX)) {
        return false;
    }
    PA_GM SharedRegionSlot &slot =
        map.slots[SharedTensorMapSlotIndex(bucket, cursor)];
    const int64_t expected_old =
        cursor < kMapBucketCapacity
            ? kSharedMapEmptySeq
            : static_cast<int64_t>(cursor - kMapBucketCapacity);
    const int64_t invalidated =
        Ops::Exchange(&slot.seq.value, kSharedMapEmptySeq);
    if (invalidated != expected_old) {
        return false;
    }

    Ops::InvalidateRegion(&slot.payload, sizeof(slot.payload));
    slot.payload.value.buffer_addr = entry.buffer_addr;
    slot.payload.value.lo = entry.lo;
    slot.payload.value.hi = entry.hi;
    slot.payload.value.producer = entry.producer;
    slot.payload.value.reserved = 0;
    // padding 不承载协议字段，也没有 reader/host 消费者；只写完整的
    // 32B value，避免为 cache-line 填充字节增加无意义的 scalar store。
    Ops::FlushRegion(&slot.payload, sizeof(slot.payload));

    const int64_t before_publish =
        Ops::Exchange(&slot.seq.value, static_cast<int64_t>(cursor));
    if (before_publish != kSharedMapEmptySeq) {
        return false;
    }
    const int64_t previous_tail =
        Ops::Exchange(&controls.tail.value, tail + 1);
    return previous_tail == tail;
}

template <typename Ops>
PA_DEVICE bool SharedAppendPreparedTask(
    PA_GM SharedTensorMapSidecar &map, const SharedRegionValue *entries,
    uint32_t count
) {
    for (uint32_t index = 0; index < count; ++index) {
        if (!SharedAppendPreparedEntry<Ops>(map, entries[index])) {
            return false;
        }
    }
    return true;
}

template <typename Ops>
PA_DEVICE bool SharedPublishTaskCommit(
    PA_GM SharedTensorMapSidecar &map, int32_t task_id
) {
    if (task_id < 0 || task_id == INT32_MAX) {
        return false;
    }
    const int64_t previous = Ops::Exchange(
        &map.committed_tasks.value, static_cast<int64_t>(task_id) + 1
    );
    if (previous == task_id) {
        return true;
    }
    // 调用者必须持有 exact turn，因此失败分支不存在合法的并发 publisher。
    // 无条件 Exchange 已经写入 task_id+1；恢复观测到的旧值，避免协议损坏时
    // 把一个更大的前沿写小，或把尚未到达的前沿错误推进。
    (void)Ops::Exchange(&map.committed_tasks.value, previous);
    return false;
}

}  // namespace pa_scheduler

#endif  // PA_SCHEDULER_COMMON_PA_SHARED_TENSORMAP_H
