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

// 调用者必须已经持有 committed_tasks==task_id 的 exact turn。因此两个
// FetchAdd 之间不存在合法并发 writer；若原子返回值仍偏离预检快照，就属于
// 协议异常，冷路径把 cursor/vend 恢复后返回 false，由上层发布 fatal。
//
// no-wrap 是本阶段的明确边界：每个 shard 的绝对 cursor 只能从 0 推进到
// shard_span，绝不取模。这样不依赖尚未证明的 generation 或 H-window 复用。
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
    if (task_id >= kMaxTasks) {
        return false;
    }

    const int64_t signed_vend_before = Ops::Load(&map.shared_heap_vend.value);
    if (signed_vend_before < 0) {
        return false;
    }
    const uint64_t vend_before = static_cast<uint64_t>(signed_vend_before);
    if (!SharedHeapAligned(vend_before) || vend_before > heap_size) {
        return false;
    }

    // 零输出 task 仍需要取得当前 aggregate vend，供完成协议发布该 task 的
    // progress；但它不读取或推进任一 shard cursor，也不要求可用 heap 空间。
    if (total == 0) {
        reservation.aggregate_vend = vend_before;
        return true;
    }

    if (total > UINT64_MAX - (kOutputAlignment - 1)) {
        return false;
    }
    const uint64_t reserve =
        (total + kOutputAlignment - 1) & ~(kOutputAlignment - 1);
    if (reserve == 0 || reserve > static_cast<uint64_t>(INT64_MAX)) {
        return false;
    }

    const uint64_t shard_span =
        SharedHeapAlignDown(heap_size / kSharedHeapShards);
    if (shard_span == 0 || reserve > shard_span ||
        vend_before > heap_size - reserve ||
        vend_before > static_cast<uint64_t>(INT64_MAX) - reserve) {
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
    if (observed_cursor != signed_cursor_before) {
        // exact-turn 契约下没有第三方合法写者，因此恢复预检快照不会覆盖
        // 合法进度；若契约已被破坏，上层会立即 fatal，不得继续分配。
        (void)Ops::Exchange(cursor_address, signed_cursor_before);
        return false;
    }

    PA_GM volatile int64_t *vend_address = &map.shared_heap_vend.value;
    const int64_t observed_vend = Ops::FetchAdd(
        vend_address, static_cast<int64_t>(reserve)
    );
    if (observed_vend != signed_vend_before) {
        (void)Ops::Exchange(vend_address, signed_vend_before);
        (void)Ops::Exchange(cursor_address, signed_cursor_before);
        return false;
    }

    reservation.task_base =
        static_cast<uint64_t>(shard) * shard_span + cursor_before;
    reservation.aggregate_vend = vend_before + reserve;
    return true;
}

}  // namespace pa_scheduler

#endif  // PA_SCHEDULER_COMMON_PA_SHARED_HEAP_H
