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
    int64_t signed_head = Ops::Load(&map.buckets[bucket].head.value);
    const int64_t signed_tail = Ops::Load(&map.buckets[bucket].tail.value);
    if (__builtin_expect(
            signed_head < 0 || signed_tail < signed_head ||
                static_cast<uint64_t>(signed_tail - signed_head) >
                    kMapBucketCapacity,
            0
        )) {
        if (signed_head < 0 || signed_tail < 0) {
            return -1;
        }
        // head/tail 是两个独立 control atomic。reader 先读旧 head 后，
        // writer 可能合法回收前缀、复用槽并发布新 tail，从而让混合快照
        // 暂时呈现 span>CAP。异常支路只重读一次 head；唯有 head 单调
        // 前进后能把同一 tail 重新约束到容量内，才接受该快照。
        const int64_t refreshed_head =
            Ops::Load(&map.buckets[bucket].head.value);
        if (refreshed_head < signed_head ||
            refreshed_head > signed_tail ||
            static_cast<uint64_t>(
                signed_tail - refreshed_head
            ) > kMapBucketCapacity) {
            return -1;
        }
        signed_head = refreshed_head;
    }

    const int32_t lower =
        current_task > heap_window ? current_task - heap_window : 0;
    int32_t best = -1;
    uint64_t cursor = static_cast<uint64_t>(signed_head);
    const uint64_t tail = static_cast<uint64_t>(signed_tail);
    while (cursor < tail) {
        SharedRegionValue candidate{};
        if (__builtin_expect(
                !SharedReadRegionSlot<Ops>(
                    map, bucket, cursor, candidate
                ),
                0
            )) {
            // reader 保存旧 head 后，未来唯一 writer 仍可合法回收
            // producer < current_task-H 的无关前缀，并复用其物理槽。只有
            // head 已单调越过当前 cursor，才能把 seq 双检失败解释为这类
            // 合法复用；否则继续 fail-closed，不能吞掉真实 slot 损坏。
            const int64_t refreshed_head =
                Ops::Load(&map.buckets[bucket].head.value);
            if (refreshed_head < signed_head ||
                refreshed_head > signed_tail ||
                cursor >= static_cast<uint64_t>(refreshed_head)) {
                return -1;
            }
            cursor = static_cast<uint64_t>(refreshed_head);
            continue;
        }
        if (candidate.producer >= lower &&
            candidate.producer < current_task &&
            SharedRegionOverlaps(candidate, query) &&
            candidate.producer > best) {
            best = candidate.producer;
        }
        ++cursor;
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

// append 至少要求唯一、按 task_id 单调推进的 writer；exact turn 与
// writer-ready replay 都可建立 writer 顺序。真正推进 head 还必须由
// reader_done 证明所有更早 reader 已结束；仅有 writer exact turn 并不能
// 证明这一点。通用 intent 仍固定传 reclaim_upto=-1，R4e-a 只先建立
// reader progress 状态与纯公式，不打开任何运行时回收。
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

#if PTO_FDWIC_SHARED_MAP
// reader_done[worker]=N 表示该 worker 已经关闭 task [0,N] 的全部
// ordinary-ring 读取。CAS 只允许 N-1 -> N，重复、跳号和倒退都保留旧值；
// 调用方必须先证明本 task 后续不会再访问 ring，并在 A5 接线前另外闭合
// “读取完成 -> CAS”的编译器/设备顺序。R4e-a 本身不从 PA 热路径调用。
template <typename Ops>
PA_DEVICE bool SharedAdvanceReaderDone(
    PA_GM SharedTensorMapSidecar &map, uint32_t worker,
    int32_t task_id
) {
    if (worker >= kWorkers || task_id < 0 ||
        task_id >= static_cast<int32_t>(kMaxTasks)) {
        return false;
    }
    const int64_t expected =
        static_cast<int64_t>(task_id) - 1;
    return Ops::CompareExchange(
               &map.reader_done[worker].value,
               expected, static_cast<int64_t>(task_id)
           ) == expected;
}

// 对连续 active worker 前缀取最慢完成值 Dmin。窗口 H 允许未来 reader
// 查询 producer >= current_task-H；因此所有 worker 都完成到 Dmin 后，
// inclusive 安全回收上界是 max(-1,Dmin-H)。inactive worker 不参与最小值。
// 任一非法进度都 fail-closed，且失败时不修改 candidate。
template <typename Ops>
PA_DEVICE bool SharedComputeReaderReclaimCandidate(
    PA_GM SharedTensorMapSidecar &map, uint32_t active_workers,
    int32_t heap_window, int64_t &candidate
) {
    if (active_workers == 0 || active_workers > kWorkers ||
        heap_window < 0) {
        return false;
    }
    int64_t minimum_done = INT64_MAX;
    for (uint32_t worker = 0; worker < active_workers; ++worker) {
        const int64_t done =
            Ops::Load(&map.reader_done[worker].value);
        if (done < -1 ||
            done >= static_cast<int64_t>(kMaxTasks)) {
            return false;
        }
        if (done < minimum_done) {
            minimum_done = done;
        }
    }
    int64_t computed =
        minimum_done - static_cast<int64_t>(heap_window);
    if (computed < -1) {
        computed = -1;
    }
    candidate = computed;
    return true;
}
#endif

// 以下 current_task 公式只保留给既有单线程 ordered-ring 隔离 driver，
// 不能作为 generic 多 reader 回收依据。真实接线必须改用上面的最慢
// reader_done 公式，并同时保持唯一 ordered append actor。
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

PA_DEVICE uint32_t SharedInsertTurnLane(int32_t token) {
    return static_cast<uint32_t>(token) &
           kSharedInsertTurnMask;
}

PA_DEVICE PA_GM AtomicLine &SharedInsertTurnLine(
    PA_GM SharedTensorMapSidecar &map, uint32_t lane
) {
    // lane 0 保持原 committed_tasks 地址，G=1 不移动热点控制字。
    // 调用者只传 [0,kSharedInsertTurnCapacity)；active G 之外的物理线
    // 仅供初始化和 host 终态校验。
    if (lane == 0) {
        return map.committed_tasks;
    }
    return map.insert_turn_extra[lane - 1U];
}

PA_DEVICE void InitializeSharedInsertTurns(
    PA_GM SharedTensorMapSidecar &map
) {
    map.committed_tasks.value = 0;
    for (uint32_t lane = 1;
         lane < kSharedInsertTurnCapacity; ++lane) {
        map.insert_turn_extra[lane - 1U].value = -1;
    }
}

// completed_tasks=T 表示 token 0..T 已按序产生。每条 active lane 保存
// 不大于 T 的最大同余 token；从未接收 token 的 lane 与全部 inactive
// lane 保持 -1。该纯公式供 host 和独立测试建立权威终态。
constexpr int64_t SharedInsertTurnTokenAfterTasks(
    uint32_t completed_tasks, uint32_t lane
) {
    if (lane >= kSharedInsertTurnGroups) {
        return -1;
    }
    if (lane > completed_tasks) {
        return lane == 0 ? 0 : -1;
    }
    return static_cast<int64_t>(
        completed_tasks -
        ((completed_tasks - lane) & kSharedInsertTurnMask)
    );
}

PA_DEVICE int64_t SharedInsertTurnPublishExpectedOld(
    int32_t task_id
) {
    const int64_t next =
        static_cast<int64_t>(task_id) + 1;
    return next >=
                   static_cast<int64_t>(
                       kSharedInsertTurnGroups
                   )
               ? next -
                     static_cast<int64_t>(
                         kSharedInsertTurnGroups
                     )
               : -1;
}

enum class SharedInsertTurnState : uint32_t {
    Ready = 0,
    Pending = 1,
    ProtocolError = 2,
};

// insert-turn 是不索引 SchedulerState::tasks 的绝对序列原语，隔离 ring
// 门槛会用它覆盖超过 kMaxTasks 的多圈 cursor，因此这里只限制 int32
// 可表达性。生产 Submit 在触碰任何共享状态前另行要求 task_id 落在
// [0,kMaxTasks)，不能把两层合同合并后破坏大 CAP 原语门槛。
template <typename Ops, bool PreserveReadyDependency = false>
PA_DEVICE SharedInsertTurnState SharedInspectTaskTurnObserved(
    PA_GM SharedTensorMapSidecar &map, int32_t current_task,
    int64_t &observed
) {
    if (current_task < 0) {
        observed = -1;
        return SharedInsertTurnState::ProtocolError;
    }
    const uint32_t lane =
        SharedInsertTurnLane(current_task);
    observed = Ops::Load(
        &SharedInsertTurnLine(map, lane).value
    );
    int64_t compare_observed = observed;
#if PA_BUILD_SWIMLANE && \
    (defined(PA_BUILD_AIC) || defined(PA_BUILD_AIV))
    if constexpr (PreserveReadyDependency) {
        // Ready 分支会向编译器透露 compare_observed == current_task。CCEC
        // 因此只为真正需要计时的 wait 调用，从同一个 atomic 返回寄存器
        // 派生两个编译器不可等同的值：一个只作分支判定，一个通过
        // observed 带到 SYS_CNT 依赖边界。其他协议检查不承担这条 MOV。
        compare_observed = Ops::ForkAtomicResultForBranch(
            observed, observed
        );
    }
#endif
    if (compare_observed == current_task) {
        return SharedInsertTurnState::Ready;
    }
    if (compare_observed < -1 ||
        compare_observed > current_task) {
        return SharedInsertTurnState::ProtocolError;
    }
    // lane 0 从初始化起始终保存非负的 0 mod G token；其他 lane 在
    // 第一次接收 token 前合法保持 -1。已发布旧 token 必须与本 lane
    // 同余，否则表示初始化、越序写或地址计算已经损坏。
    if (compare_observed == -1) {
        return lane == 0
                   ? SharedInsertTurnState::ProtocolError
                   : SharedInsertTurnState::Pending;
    }
    if ((static_cast<uint32_t>(compare_observed) &
         kSharedInsertTurnMask) != lane) {
        return SharedInsertTurnState::ProtocolError;
    }
    return SharedInsertTurnState::Pending;
}

template <typename Ops>
PA_DEVICE SharedInsertTurnState SharedInspectTaskTurn(
    PA_GM SharedTensorMapSidecar &map, int32_t current_task
) {
    int64_t ignored_observed = -1;
    return SharedInspectTaskTurnObserved<Ops>(
        map, current_task, ignored_observed
    );
}

template <typename Ops>
PA_DEVICE bool SharedHasExactTaskTurn(
    PA_GM SharedTensorMapSidecar &map, int32_t current_task
) {
    return SharedInspectTaskTurn<Ops>(
               map, current_task
           ) == SharedInsertTurnState::Ready;
}

// current lane 的 N 是进入有序通道的 grant；G>1 时它会保留到下一代
// token 覆盖，不能单独充当“尚未发布”的锁。生产合同由 Claim 保证
// 每 task 只有一个 owner，同时在写任何 TensorMap 元数据前预检目标
// lane 仍是本 task 应覆盖的旧 token，拒绝已经完成的重复调用。
template <typename Ops>
PA_DEVICE bool SharedCanPublishTaskCommit(
    PA_GM SharedTensorMapSidecar &map, int32_t task_id
) {
    if (task_id < 0 || task_id == INT32_MAX ||
        !SharedHasExactTaskTurn<Ops>(map, task_id)) {
        return false;
    }
    if (kSharedInsertTurnGroups == 1U) {
        // current/next lane 是同一条线，exact N 已经同时证明 CAS 的
        // expected_old=N；避免 G=1 基线多做一次相同地址 load。
        return true;
    }
    const int32_t next = task_id + 1;
    const uint32_t next_lane =
        SharedInsertTurnLane(next);
    const int64_t expected =
        SharedInsertTurnPublishExpectedOld(task_id);
    return Ops::Load(
               &SharedInsertTurnLine(
                    map, next_lane
                ).value
           ) == expected;
}

template <typename Ops>
PA_DEVICE bool SharedPublishTaskCommitAfterPreflightObserved(
    PA_GM SharedTensorMapSidecar &map, int32_t task_id,
    int64_t &observed
) {
    if (task_id < 0 || task_id == INT32_MAX) {
        observed = INT64_MIN;
        return false;
    }
    const int32_t next = task_id + 1;
    const uint32_t next_lane =
        SharedInsertTurnLane(next);
    const int64_t expected =
        SharedInsertTurnPublishExpectedOld(task_id);
    observed = Ops::CompareExchange(
               &SharedInsertTurnLine(
                    map, next_lane
                ).value,
               expected, static_cast<int64_t>(next)
           );
    return observed == expected;
}

template <typename Ops>
PA_DEVICE bool SharedPublishTaskCommitAfterPreflight(
    PA_GM SharedTensorMapSidecar &map, int32_t task_id
) {
    int64_t ignored_observed = INT64_MIN;
    return SharedPublishTaskCommitAfterPreflightObserved<Ops>(
        map, task_id, ignored_observed
    );
}

// reclaim_upto 只允许唯一 ordered append actor 单调发布。这里不从 exact
// turn 推导 actor 唯一性：调用方仍必须先完成 winner/turn 所有权收敛。
// R4e-e 会在真实接线前单独比较 Exchange 与 CAS 的设备成本；当前保持
// 既有单写者发布语义，不把性能选择混入 reader 正确性门槛。
template <typename Ops>
PA_DEVICE bool SharedPublishReclaimCandidate(
    PA_GM SharedTensorMapSidecar &map, int64_t candidate,
    int64_t &reclaim_upto
) {
    if (candidate < -1) {
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

#if PTO_FDWIC_SHARED_MAP
// reader-based refresh 只组合 exact turn、最慢 reader 候选和既有单调发布；
// 不扫描 bucket、不 append，也没有 PA/Submit 调用者。exact turn 仍不等于
// 唯一 actor，外层必须先证明只有一个 ordered append actor 进入本原语。
// active_workers 必须是本轮固定的连续活跃前缀，heap_window 也必须在 ring
// 整个生命周期保持同一权威值；任一参数中途缩小都可能发布不可撤回的过激
// 候选，随后再恢复真实配置已经无法找回被回收的槽。
template <typename Ops>
PA_DEVICE bool SharedRefreshReaderReclaimForTask(
    PA_GM SharedTensorMapSidecar &map, int32_t current_task,
    uint32_t active_workers, int32_t heap_window,
    int64_t &reclaim_upto
) {
    if (!SharedHasExactTaskTurn<Ops>(map, current_task)) {
        return false;
    }
    int64_t candidate = -1;
    if (!SharedComputeReaderReclaimCandidate<Ops>(
            map, active_workers, heap_window, candidate
        )) {
        return false;
    }
    return SharedPublishReclaimCandidate<Ops>(
        map, candidate, reclaim_upto
    );
}
#endif

// 只有 exact committed turn 的 winner 可以推进 reclaim。先验证
// committed_tasks==current_task，再计算当前任务的 inclusive 回收边界；
// 逆序/陈旧 actor 在任何 head、tail 或 reclaim 写入前失败。该旧路径只
// 服务单线程 ordered-ring 隔离 driver；generic 多 reader 使用上面的
// reader-based refresh。
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
    return SharedPublishReclaimCandidate<Ops>(
        map, candidate, reclaim_upto
    );
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
    if (count > kMaxTaskTensors || reclaim_upto < -1) {
        return SharedAppendCheck::ProtocolError;
    }
    for (uint32_t index = 0; index < count; ++index) {
        const SharedRegionValue &entry = entries[index];
        if (entry.producer < 0 || entry.reserved != 0 ||
            entry.lo >= entry.hi) {
            return SharedAppendCheck::ProtocolError;
        }
        const uint32_t bucket = TensorMapHash(entry.buffer_addr);
        // 当前 ordered Submit 固定传 -1，明确表示不回收任何合法
        // producer。该热路径无需读取旧 head slot、invalidate payload 和
        // 双检 seq；下面仍完整执行 head/tail、容量和目标 seq 的
        // fail-closed 预检。非负回收边界继续使用原 retire 协议。
        if (reclaim_upto != -1 &&
            !SharedRetireBucket<Ops>(
                map, bucket, reclaim_upto
            )) {
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
        // 该 entry 发布后 tail 必须递增；cursor==INT64_MAX 也没有
        // 可表达的 next tail，必须在触碰 slot 前拒绝。
        if (cursor >= static_cast<uint64_t>(INT64_MAX)) {
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
    if (tail == INT64_MAX) {
        return false;
    }
    const uint64_t cursor = static_cast<uint64_t>(tail);
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

#if PTO_FDWIC_SHARED_MAP
// 在调用本组合前，当前 worker 必须已经完成本 task 的全部 ordinary
// lookup，并用 SharedAdvanceReaderDone() 恰好关闭一次 reader；append
// actor 还必须由外层 writer-ready replay 保证唯一且按 task id 有序。
//
// 本组合不读取/推进 committed_tasks。正常路径先复用已发布的
// reclaim_upto 做整 task 预检，Ready 时直接 append，不扫描 reader_done；
// 只有容量不足才按固定 active-worker 前缀刷新 reader 回收边界并重试：
// - Ready：整批预检和 append 均已成功；
// - CapacityBlocked：当前 task 没有发布 payload/seq/tail，可在其他
//   reader 前进后重试本函数；安全过期的 head/reclaim 可以保留；
// - ProtocolError：调用层必须终止本轮，不得发布 writer-ready。
// append 阶段若遭遇预检后协议破坏，可能已经发布物理前缀；唯一 writer
// 合同下这不是合法竞争，不能尝试回滚。
template <typename Ops>
PA_DEVICE SharedAppendCheck SharedTryAppendReaderGatedTask(
    PA_GM SharedTensorMapSidecar &map,
    const SharedRegionValue *entries, uint32_t count,
    uint32_t active_workers, int32_t heap_window
) {
    if (entries == nullptr && count != 0) {
        return SharedAppendCheck::ProtocolError;
    }
    // 纯 symbol writer 没有 ordinary entry，不应为一个空 batch 读取
    // reclaim 或扫描 reader 前沿。
    if (count == 0) {
        return SharedAppendCheck::Ready;
    }
    if (active_workers == 0 || active_workers > kWorkers ||
        heap_window < 0) {
        return SharedAppendCheck::ProtocolError;
    }
    int64_t reclaim_upto =
        Ops::Load(&map.reclaim_upto.value);
    if (reclaim_upto < -1) {
        return SharedAppendCheck::ProtocolError;
    }
    SharedAppendCheck check =
        SharedCheckTaskAppend<Ops>(
            map, entries, count, reclaim_upto
        );
    if (check == SharedAppendCheck::ProtocolError) {
        return check;
    }
    if (check == SharedAppendCheck::Ready) {
        return SharedAppendPreparedTask<Ops>(
                   map, entries, count
               )
                   ? SharedAppendCheck::Ready
                   : SharedAppendCheck::ProtocolError;
    }

    int64_t candidate = -1;
    if (!SharedComputeReaderReclaimCandidate<Ops>(
            map, active_workers, heap_window, candidate
        )) {
        return SharedAppendCheck::ProtocolError;
    }
    if (!SharedPublishReclaimCandidate<Ops>(
            map, candidate, reclaim_upto
        )) {
        return SharedAppendCheck::ProtocolError;
    }
    check =
        SharedCheckTaskAppend<Ops>(
            map, entries, count, reclaim_upto
        );
    if (check != SharedAppendCheck::Ready) {
        return check;
    }
    return SharedAppendPreparedTask<Ops>(map, entries, count)
               ? SharedAppendCheck::Ready
               : SharedAppendCheck::ProtocolError;
}
#endif

template <typename Ops>
PA_DEVICE bool SharedPublishTaskCommit(
    PA_GM SharedTensorMapSidecar &map, int32_t task_id
) {
    if (kSharedInsertTurnGroups == 1U) {
        // 保持既有 G=1 原语恰好一次 CAS、零预检 load 的事件形状；
        // CAS(N,N+1) 本身即可拒绝重复、陈旧和 future actor。
        return SharedPublishTaskCommitAfterPreflight<Ops>(
            map, task_id
        );
    }
    if (!SharedCanPublishTaskCommit<Ops>(
            map, task_id
        )) {
        return false;
    }
    return SharedPublishTaskCommitAfterPreflight<Ops>(
        map, task_id
    );
}

}  // namespace pa_scheduler

#endif  // PA_SCHEDULER_COMMON_PA_SHARED_TENSORMAP_H
