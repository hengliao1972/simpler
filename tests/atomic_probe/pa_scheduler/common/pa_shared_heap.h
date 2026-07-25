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

#ifndef PA_SCHEDULER_COMMON_PA_SHARED_HEAP_H
#define PA_SCHEDULER_COMMON_PA_SHARED_HEAP_H

#include "pa_model.h"

namespace pa_scheduler {

// shared heap 的首版只验证 PA Case1 当前有界规模，不启用物理区间复用。
// task_base 是 heap 内的物理偏移；aggregate_vend 只是所有 shard 已分配字节
// 的全局累计值，不是任一 shard 的地址，也不能代替旧单 ring HeapGuard。
struct SharedHeapReservation {
    uint64_t task_base;
    uint64_t aggregate_vend;
};
static_assert(sizeof(SharedHeapReservation) == 16, "shared heap reservation ABI changed");

PA_DEVICE uint64_t SharedHeapAlignDown(uint64_t value) {
    return value & ~(kOutputAlignment - 1);
}

PA_DEVICE bool SharedHeapAligned(uint64_t value) {
    return (value & (kOutputAlignment - 1)) == 0;
}

// no-wrap 是本阶段的明确边界：每个 shard 的绝对 cursor 只能从 0 推进到
// shard_span，绝不取模。FetchAdd 返回的旧 cursor 是当前 task 唯一的物理
// 区间；合法并发 writer 可以让它不同于前置 Load 的观察值，不能因此回滚。
//
// 容量竞争若在 FetchAdd 后才被发现，则本轮进入 terminal fatal 并保留已经
// 推进的控制字供 host 取证。并发 allocator 绝不能用 Exchange 恢复预检
// 快照，否则会覆盖其他 winner 的合法进度。
template <class Ops>
PA_DEVICE bool ReserveSharedOutputHeap(
    PA_GM SharedTensorMapSidecar &map, uint32_t task_id, uint64_t total,
    uint64_t heap_size, SharedHeapReservation &reservation
) {
    reservation.task_base = 0;
    reservation.aggregate_vend = 0;

    static_assert(kSharedHeapShards == 8, "standalone shared heap must use eight shards");
    static_assert(
        (kSharedHeapShards & (kSharedHeapShards - 1)) == 0,
        "shared heap shard count must be a power of two"
    );
    static_assert(
        (kOutputAlignment & (kOutputAlignment - 1)) == 0,
        "shared heap alignment must be a power of two"
    );
    if (task_id >= kMaxTasks ||
        heap_size > static_cast<uint64_t>(INT64_MAX)) {
        return false;
    }
    const uint64_t shard_span =
        SharedHeapAlignDown(heap_size / kSharedHeapShards);
    const uint64_t usable_capacity =
        shard_span * kSharedHeapShards;

    const int64_t checked_vend = Ops::Load(&map.shared_heap_vend.value);
    if (checked_vend < 0) {
        return false;
    }
    const uint64_t vend_snapshot = static_cast<uint64_t>(checked_vend);
    if (!SharedHeapAligned(vend_snapshot) ||
        vend_snapshot > usable_capacity) {
        return false;
    }

    // 零输出 task 仍需要取得当前 aggregate vend，供完成协议发布该 task 的
    // progress；但它不读取或推进任一 shard cursor，也不要求可用 heap 空间。
    if (total == 0) {
        reservation.aggregate_vend = vend_snapshot;
        return true;
    }
    if (shard_span == 0) {
        return false;
    }

    if (total > UINT64_MAX - (kOutputAlignment - 1)) {
        return false;
    }
    const uint64_t reserve =
        (total + kOutputAlignment - 1) & ~(kOutputAlignment - 1);
    if (reserve == 0 || reserve > static_cast<uint64_t>(INT64_MAX)) {
        return false;
    }

    if (reserve > shard_span ||
        vend_snapshot > usable_capacity - reserve ||
        vend_snapshot > static_cast<uint64_t>(INT64_MAX) - reserve) {
        return false;
    }

    const uint32_t shard = task_id % kSharedHeapShards;
    PA_GM volatile int64_t *cursor_address =
        &map.shared_heap_cursor[shard].value;
    const int64_t signed_cursor_before = Ops::Load(cursor_address);
    if (signed_cursor_before < 0) {
        return false;
    }
    const uint64_t cursor_before =
        static_cast<uint64_t>(signed_cursor_before);
    if (!SharedHeapAligned(cursor_before) ||
        cursor_before > shard_span - reserve) {
        return false;
    }

    const int64_t observed_cursor = Ops::FetchAdd(
        cursor_address, static_cast<int64_t>(reserve)
    );
    if (observed_cursor < 0) {
        return false;
    }
    const uint64_t cursor = static_cast<uint64_t>(observed_cursor);
    if (!SharedHeapAligned(cursor) || cursor > shard_span - reserve) {
        return false;
    }

    PA_GM volatile int64_t *vend_address = &map.shared_heap_vend.value;
    const int64_t observed_vend = Ops::FetchAdd(
        vend_address, static_cast<int64_t>(reserve)
    );
    if (observed_vend < 0) {
        return false;
    }
    const uint64_t vend = static_cast<uint64_t>(observed_vend);
    if (!SharedHeapAligned(vend) ||
        vend > usable_capacity - reserve ||
        vend > static_cast<uint64_t>(INT64_MAX) - reserve) {
        return false;
    }

    reservation.task_base =
        static_cast<uint64_t>(shard) * shard_span + cursor;
    reservation.aggregate_vend = vend + reserve;
    return true;
}

}  // namespace pa_scheduler

#endif  // PA_SCHEDULER_COMMON_PA_SHARED_HEAP_H
