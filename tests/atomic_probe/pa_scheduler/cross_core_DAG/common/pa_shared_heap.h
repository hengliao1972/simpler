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

#include "pa_trace.h"

namespace pa_scheduler {

// shared heap 只分配有界、no-wrap 的物理区间，不启用区间复用。
// task_base 是 heap 内的物理偏移；completion_vend 是本 task 已分配区间
// 的物理结束偏移，供跨核 Execute 完成协议携带和 host 精确核验。
// 分配正确性只依赖所属 shard cursor，不再为诊断性全局累计值额外争抢
// 一条所有 builder 共用的返回型 atomic。
struct SharedHeapReservation {
    uint64_t task_base;
    uint64_t completion_vend;
};
static_assert(sizeof(SharedHeapReservation) == 16, "shared heap reservation ABI changed");

PA_DEVICE uint64_t SharedHeapAlignDown(uint64_t value) {
    return value & ~(kOutputAlignment - 1);
}

PA_DEVICE bool SharedHeapAligned(uint64_t value) {
    return (value & (kOutputAlignment - 1)) == 0;
}

// 单元测试默认实例化 ObserveAtomics=false，保持原有简洁 Ops 接口；真实
// scheduler 显式选择 true 并传入本 worker 独占 trace/result。trace-free
// 构建中的 TraceAtomic* 会在编译期退化为原始 Ops，不给性能基线增加分支。
template <typename Ops, bool ObserveAtomics>
PA_DEVICE int64_t SharedHeapAtomicFetchAdd(
    PA_GM volatile int64_t *address, int64_t value, int32_t task_id,
    AtomicSite site, TraceContext *trace, WorkerResult *result
) {
    if constexpr (ObserveAtomics) {
        // 两个 FetchAdd 的旧值都决定本次 reservation，必须使用
        // return-ready 边界，不能按发布型 source-issue 观察。
        return TraceAtomicFetchAdd<Ops>(
            *trace, *result, task_id, site, address, value, true
        );
    } else {
        (void)task_id;
        (void)site;
        (void)trace;
        (void)result;
        return Ops::FetchAdd(address, value);
    }
}

// no-wrap 是本阶段的明确边界：每个 shard 的绝对 cursor 只能从 0 推进到
// shard_span，绝不取模。FetchAdd 返回的旧 cursor 是当前 task 唯一的物理
// 区间；不再用竞态 Load 快照预判随后 RMW 的结果。
//
// 容量竞争若在 FetchAdd 后才被发现，则本轮进入 terminal fatal 并保留已经
// 推进的控制字供 host 取证。并发 allocator 绝不能用 Exchange 恢复旧值，
// 否则会覆盖其他 winner 的合法进度。
template <class Ops, bool ObserveAtomics = false>
PA_DEVICE bool ReserveSharedOutputHeap(
    PA_GM SharedTensorMapSidecar &map, uint32_t task_id, uint64_t total,
    uint64_t heap_size, SharedHeapReservation &reservation,
    TraceContext *trace = nullptr, WorkerResult *result = nullptr
) {
    reservation.task_base = 0;
    reservation.completion_vend = 0;

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
    // shared/no-wrap 模式不使用 task vend 回收 heap。零输出 task 因此不需要
    // 读取任何全局进度；它发布 completion_vend=0 和 ready flag 即可。
    if (total == 0) {
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

    if (reserve > shard_span) {
        return false;
    }

    const uint32_t shard = task_id % kSharedHeapShards;
    PA_GM volatile int64_t *cursor_address =
        &map.shared_heap_cursor[shard].value;
    // FetchAdd 的返回值才是当前 reservation 的线性化 old value。先做
    // Load 得到的仅是可能立刻过期的竞态快照，既不能证明容量，也会为
    // 每个非空 task 增加一次返回型 Atomic；直接消费 RMW 旧值与 SIMT
    // builder 的 heap 合同一致。越界仍是 terminal failure，并保留已经
    // 推进的控制字作为故障现场，绝不回滚覆盖并发 owner。
    const int64_t observed_cursor =
        SharedHeapAtomicFetchAdd<Ops, ObserveAtomics>(
            cursor_address, static_cast<int64_t>(reserve),
            static_cast<int32_t>(task_id),
            AtomicSite::SharedHeapCursorReserve, trace, result
        );
    if (observed_cursor < 0) {
        return false;
    }
    const uint64_t cursor = static_cast<uint64_t>(observed_cursor);
    if (!SharedHeapAligned(cursor) || cursor > shard_span - reserve) {
        return false;
    }

    reservation.task_base =
        static_cast<uint64_t>(shard) * shard_span + cursor;
    reservation.completion_vend = reservation.task_base + reserve;
    return true;
}

}  // namespace pa_scheduler

#endif  // PA_SCHEDULER_COMMON_PA_SHARED_HEAP_H
