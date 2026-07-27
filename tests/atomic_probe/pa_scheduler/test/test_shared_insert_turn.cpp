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

#include "pa_scheduler_core.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <new>
#include <sys/mman.h>

namespace {

using namespace pa_scheduler;

static_assert(
    kSharedInsertTurnGroups ==
        static_cast<uint32_t>(
            PTO_FDWIC_SHARED_INSERT_TURN_GROUPS
        ),
    "shared insert-turn test compiled with inconsistent G"
);
static_assert(
    kSharedInsertTurnCapacity == 8,
    "shared insert-turn test expects eight physical lines"
);

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(
        stderr, "[FAIL] shared insert turn G=%u: %s\n",
        kSharedInsertTurnGroups, message
    );
    ++g_failures;
}

void CheckLaneValue(
    int64_t actual, int64_t expected, uint32_t lane,
    const char *stage
) {
    if (actual == expected) {
        return;
    }
    std::fprintf(
        stderr,
        "[FAIL] shared insert turn G=%u: %s lane=%u "
        "actual=%lld expected=%lld\n",
        kSharedInsertTurnGroups, stage, lane,
        static_cast<long long>(actual),
        static_cast<long long>(expected)
    );
    ++g_failures;
}

// 这里只模拟生产 helper 所需的 acquire load 与 CAS；测试不涉及
// TensorMap payload，也不把宿主原子内存序冒充为 A5 DCache 证据。
struct TurnTestOps {
    static int64_t Load(volatile int64_t *address) {
        return __atomic_load_n(address, __ATOMIC_ACQUIRE);
    }

    static int64_t CompareExchange(
        volatile int64_t *address, int64_t expected,
        int64_t desired
    ) {
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(
            address, &observed, desired, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        return observed;
    }
};

using TurnVector =
    std::array<int64_t, kSharedInsertTurnCapacity>;

SharedTensorMapSidecar *AllocateMap() {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void *memory = mmap(
        nullptr, sizeof(SharedTensorMapSidecar),
        PROT_READ | PROT_WRITE, flags, -1, 0
    );
    if (memory == MAP_FAILED) {
        std::perror("mmap SharedTensorMapSidecar");
        return nullptr;
    }
    return ::new (memory) SharedTensorMapSidecar;
}

void FreeMap(SharedTensorMapSidecar *map) {
    if (map == nullptr) {
        return;
    }
    map->~SharedTensorMapSidecar();
    (void)munmap(map, sizeof(SharedTensorMapSidecar));
}

int64_t LoadLane(
    SharedTensorMapSidecar &map, uint32_t lane
) {
    return TurnTestOps::Load(
        &SharedInsertTurnLine(map, lane).value
    );
}

void StoreLane(
    SharedTensorMapSidecar &map, uint32_t lane,
    int64_t value
) {
    __atomic_store_n(
        &SharedInsertTurnLine(map, lane).value,
        value, __ATOMIC_RELEASE
    );
}

TurnVector SnapshotTurns(SharedTensorMapSidecar &map) {
    TurnVector snapshot{};
    for (uint32_t lane = 0;
         lane < kSharedInsertTurnCapacity; ++lane) {
        snapshot[lane] = LoadLane(map, lane);
    }
    return snapshot;
}

void ExpectNoChange(
    SharedTensorMapSidecar &map,
    const TurnVector &before, const char *stage
) {
    const TurnVector after = SnapshotTurns(map);
    for (uint32_t lane = 0;
         lane < kSharedInsertTurnCapacity; ++lane) {
        CheckLaneValue(
            after[lane], before[lane], lane, stage
        );
    }
}

// 独立 oracle 不调用生产终态公式：active lane 保存不大于 T 的最大
// 同余 token；尚未接收 token 的 lane 以及 inactive lane 都是 -1。
int64_t ExpectedTokenAfterTasks(
    uint32_t completed_tasks, uint32_t lane
) {
    if (lane >= kSharedInsertTurnGroups) {
        return -1;
    }
    if (lane > completed_tasks) {
        return lane == 0 ? 0 : -1;
    }
    const uint32_t distance =
        (completed_tasks - lane) %
        kSharedInsertTurnGroups;
    return static_cast<int64_t>(
        completed_tasks - distance
    );
}

void ExpectTurnVector(
    SharedTensorMapSidecar &map,
    uint32_t completed_tasks, const char *stage
) {
    for (uint32_t lane = 0;
         lane < kSharedInsertTurnCapacity; ++lane) {
        const int64_t expected =
            ExpectedTokenAfterTasks(completed_tasks, lane);
        CheckLaneValue(
            LoadLane(map, lane), expected, lane, stage
        );
        CheckLaneValue(
            SharedInsertTurnTokenAfterTasks(
                completed_tasks, lane
            ),
            expected, lane, "生产终态公式"
        );
    }
}

void TestInitialization(SharedTensorMapSidecar &map) {
    // 先写入脏值，防止 mmap 的零页偶然掩盖未初始化的 extra lane。
    for (uint32_t lane = 0;
         lane < kSharedInsertTurnCapacity; ++lane) {
        StoreLane(
            map, lane,
            static_cast<int64_t>(100U + lane)
        );
    }
    InitializeSharedInsertTurns(map);
    ExpectTurnVector(map, 0, "初始化");
    Check(
        SharedInspectTaskTurn<TurnTestOps>(map, 0) ==
            SharedInsertTurnState::Ready,
        "初始化后 task 0 应拿到 exact turn"
    );
    Check(
        SharedInspectTaskTurn<TurnTestOps>(map, 1) ==
            SharedInsertTurnState::Pending,
        "初始化后的 future task 应处于 pending"
    );
}

void TestRolloverAndFinalVector(
    SharedTensorMapSidecar &map
) {
    InitializeSharedInsertTurns(map);
    const int32_t last_task =
        static_cast<int32_t>(
            2U * kSharedInsertTurnGroups + 2U
        );

    for (int32_t task = 0; task <= last_task; ++task) {
        Check(
            SharedInspectTaskTurn<TurnTestOps>(
                map, task
            ) == SharedInsertTurnState::Ready,
            "顺序 task 发布前必须拿到 exact turn"
        );
        Check(
            SharedCanPublishTaskCommit<TurnTestOps>(
                map, task
            ),
            "顺序 task 的 target lane 预检应成功"
        );

        const TurnVector before = SnapshotTurns(map);
        const int32_t next = task + 1;
        const uint32_t target_lane =
            SharedInsertTurnLane(next);
        CheckLaneValue(
            before[target_lane],
            SharedInsertTurnPublishExpectedOld(task),
            target_lane, "发布前 expected_old"
        );
        Check(
            SharedPublishTaskCommit<TurnTestOps>(
                map, task
            ),
            "顺序 task 发布失败"
        );

        const TurnVector after = SnapshotTurns(map);
        for (uint32_t lane = 0;
             lane < kSharedInsertTurnCapacity; ++lane) {
            const int64_t expected =
                lane == target_lane
                    ? static_cast<int64_t>(next)
                    : before[lane];
            CheckLaneValue(
                after[lane], expected, lane,
                "每步只能修改 next token 的目标 lane"
            );
        }
        ExpectTurnVector(
            map, static_cast<uint32_t>(next),
            "逐步 rollover"
        );

        // G>1 时 current token 可能仍留在原 lane，不能仅凭 exact
        // current 判断重复调用；target-lane preflight 必须拒绝它。
        const TurnVector completed = SnapshotTurns(map);
        Check(
            !SharedCanPublishTaskCommit<TurnTestOps>(
                map, task
            ),
            "重复 task 的 target-lane 预检必须失败"
        );
        Check(
            !SharedPublishTaskCommit<TurnTestOps>(
                map, task
            ),
            "重复 task 不得再次发布"
        );
        ExpectNoChange(
            map, completed, "重复 task 失败后状态"
        );
    }

    ExpectTurnVector(
        map, static_cast<uint32_t>(last_task + 1),
        "0..2G+2 发布后的最终向量"
    );
}

void TestFuturePending(SharedTensorMapSidecar &map) {
    InitializeSharedInsertTurns(map);
    const TurnVector before = SnapshotTurns(map);
    Check(
        SharedInspectTaskTurn<TurnTestOps>(map, 1) ==
            SharedInsertTurnState::Pending,
        "尚未轮到的 task 1 应返回 pending"
    );
    Check(
        !SharedCanPublishTaskCommit<TurnTestOps>(map, 1),
        "future task 不得通过发布预检"
    );
    Check(
        !SharedPublishTaskCommit<TurnTestOps>(map, 1),
        "future task 不得修改 token"
    );
    ExpectNoChange(map, before, "future pending");
}

void TestProtocolErrors(SharedTensorMapSidecar &map) {
    InitializeSharedInsertTurns(map);
    TurnVector before = SnapshotTurns(map);
    Check(
        SharedInspectTaskTurn<TurnTestOps>(map, -1) ==
            SharedInsertTurnState::ProtocolError,
        "负 task id 必须报告协议错误"
    );
    Check(
        !SharedPublishTaskCommit<TurnTestOps>(map, -1),
        "负 task id 不得发布"
    );
    ExpectNoChange(map, before, "负 task id");

    // -1 仅允许尚未接收首个 token 的非零 lane；-2 在任何 lane
    // 都是损坏值，不能被当作“前序尚未完成”继续等待。
    InitializeSharedInsertTurns(map);
    StoreLane(map, 0, -2);
    before = SnapshotTurns(map);
    Check(
        SharedInspectTaskTurn<TurnTestOps>(map, 0) ==
            SharedInsertTurnState::ProtocolError,
        "非法负 token 必须报告协议错误"
    );
    Check(
        !SharedPublishTaskCommit<TurnTestOps>(map, 0),
        "非法负 token 不得发布"
    );
    ExpectNoChange(map, before, "非法负 token");

    // G=1 只有一个同余类，不存在“非负但 residue 错误”的 token；
    // G>1 显式把 lane 1 写成属于 lane 0 的 token 0。
    if (kSharedInsertTurnGroups > 1U) {
        InitializeSharedInsertTurns(map);
        StoreLane(map, 1, 0);
        before = SnapshotTurns(map);
        Check(
            SharedInspectTaskTurn<TurnTestOps>(map, 1) ==
                SharedInsertTurnState::ProtocolError,
            "错误 residue 必须报告协议错误"
        );
        Check(
            !SharedPublishTaskCommit<TurnTestOps>(map, 1),
            "错误 residue 不得发布"
        );
        ExpectNoChange(map, before, "错误 residue");
    }

    // 同余关系正确但 token 已经超过查询 task，也不是 pending。
    InitializeSharedInsertTurns(map);
    const uint32_t future_lane =
        SharedInsertTurnLane(1);
    StoreLane(
        map, future_lane,
        static_cast<int64_t>(
            1U + kSharedInsertTurnGroups
        )
    );
    before = SnapshotTurns(map);
    Check(
        SharedInspectTaskTurn<TurnTestOps>(map, 1) ==
            SharedInsertTurnState::ProtocolError,
        "超过 current task 的 future token 必须报告协议错误"
    );
    Check(
        !SharedPublishTaskCommit<TurnTestOps>(map, 1),
        "future token 损坏状态不得发布"
    );
    ExpectNoChange(map, before, "future token 损坏状态");
}

void TestTargetExpectedOldMismatch(
    SharedTensorMapSidecar &map
) {
    InitializeSharedInsertTurns(map);
    const int32_t task = 0;
    const int32_t next = task + 1;
    const uint32_t target_lane =
        SharedInsertTurnLane(next);

    // 把 target lane 提前写成 desired next，稳定制造 expected_old
    // 不匹配。G=1 时 current 与 target 是同一物理线，因此直接调用
    // AfterPreflight 才能隔离验证最终 CAS 仍然 fail-closed。
    StoreLane(map, target_lane, next);
    const TurnVector before = SnapshotTurns(map);
    Check(
        !SharedCanPublishTaskCommit<TurnTestOps>(
            map, task
        ),
        "target expected_old 不符时预检必须失败"
    );
    Check(
        !SharedPublishTaskCommitAfterPreflight<TurnTestOps>(
            map, task
        ),
        "target expected_old 不符时 CAS 必须失败"
    );
    Check(
        !SharedPublishTaskCommit<TurnTestOps>(
            map, task
        ),
        "target expected_old 不符时完整发布必须失败"
    );
    ExpectNoChange(
        map, before, "target expected_old 不符"
    );
}

void TestRepeatedTask(SharedTensorMapSidecar &map) {
    InitializeSharedInsertTurns(map);
    Check(
        SharedPublishTaskCommit<TurnTestOps>(map, 0),
        "重复发布测试的首次 task 0 发布失败"
    );
    const TurnVector before = SnapshotTurns(map);
    Check(
        !SharedCanPublishTaskCommit<TurnTestOps>(map, 0),
        "已完成 task 0 不得再次通过预检"
    );
    Check(
        !SharedPublishTaskCommit<TurnTestOps>(map, 0),
        "已完成 task 0 不得再次发布"
    );
    Check(
        !SharedPublishTaskCommitAfterPreflight<TurnTestOps>(
            map, 0
        ),
        "重复 task 即使绕过预检也不得覆盖 target lane"
    );
    ExpectNoChange(map, before, "重复 task");
}

}  // namespace

int main() {
    SharedTensorMapSidecar *map = AllocateMap();
    if (map == nullptr) {
        return 1;
    }

    TestInitialization(*map);
    TestRolloverAndFinalVector(*map);
    TestFuturePending(*map);
    TestProtocolErrors(*map);
    TestTargetExpectedOldMismatch(*map);
    TestRepeatedTask(*map);

    FreeMap(map);
    if (g_failures != 0) {
        std::fprintf(
            stderr,
            "[FAIL] shared insert-turn self-test G=%u: "
            "%d failure(s)\n",
            kSharedInsertTurnGroups, g_failures
        );
        return 1;
    }
    std::printf(
        "[PASS] shared insert-turn initialization, rollover, "
        "protocol-error, and final-vector tests G=%u\n",
        kSharedInsertTurnGroups
    );
    return 0;
}
