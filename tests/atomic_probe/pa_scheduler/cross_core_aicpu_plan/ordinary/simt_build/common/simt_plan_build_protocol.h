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

#ifndef PA_SCHEDULER_AICPU_PLAN_ORDINARY_SIMT_BUILD_PROTOCOL_H
#define PA_SCHEDULER_AICPU_PLAN_ORDINARY_SIMT_BUILD_PROTOCOL_H

#include "../../../common/aicpu_plan_protocol.h"

#include <cstdint>

#ifndef PA_SIMT_PLAN_DEVICE
#ifdef PA_DEVICE
// VF TU 在 include 前把 PA_DEVICE 定义为
// __simt_callee__ __aicore__ always_inline；本头的调用链必须继承
// 该属性，否则 CPU inline 门槛会掩盖 VF 不可达问题。
#define PA_SIMT_PLAN_DEVICE PA_DEVICE
#else
#define PA_SIMT_PLAN_DEVICE inline
#endif
#define PA_SIMT_PLAN_UNDEFINE_DEVICE 1
#endif

namespace pa_scheduler::aicpu_plan_simt {

// 首版只启动一个 persistent VF。128 个 SIMT thread 组成 4 个 warp，
// 每个 warp 仅 lane0 执行 Build；其他 lane 只参与 VF 的启动与收口。
constexpr uint32_t kBuilderThreads = 128U;
constexpr uint32_t kWarpSize = 32U;
constexpr uint32_t kBuilderLeaders = kBuilderThreads / kWarpSize;

static_assert(kBuilderThreads % kWarpSize == 0U);
static_assert(kBuilderLeaders == 4U);

PA_SIMT_PLAN_DEVICE constexpr bool IsBuilderLeader(uint32_t thread_id)
{
    return thread_id < kBuilderThreads &&
           (thread_id % kWarpSize) == 0U;
}

PA_SIMT_PLAN_DEVICE constexpr uint32_t BuilderLeaderId(uint32_t thread_id)
{
    return thread_id / kWarpSize;
}

PA_SIMT_PLAN_DEVICE constexpr bool BuilderLeaderIdValid(uint32_t leader_id)
{
    return leader_id < kBuilderLeaders;
}

// build_workers_done 只能证明全局“报到了四次”，不能单独证明
// “四个不同 leader 各报到一次”。VF 调用方必须为每个物理
// leader 保留一份持久的局部状态；这个检查只读写本地值，
// 不会为生产热路增加任何 GM atomic。
enum class LocalLeaderArrivalStatus : uint8_t {
    First,
    Duplicate,
    InvalidLeader,
};

struct BuilderLeaderLocalState {
    bool arrival_recorded = false;
};

PA_SIMT_PLAN_DEVICE LocalLeaderArrivalStatus RecordBuilderLeaderArrivalLocally(
    uint32_t leader_id, BuilderLeaderLocalState &state
)
{
    if (!BuilderLeaderIdValid(leader_id)) {
        return LocalLeaderArrivalStatus::InvalidLeader;
    }
    if (state.arrival_recorded) {
        return LocalLeaderArrivalStatus::Duplicate;
    }
    state.arrival_recorded = true;
    return LocalLeaderArrivalStatus::First;
}

// ordinary/SIMT 与 ordinary/Scalar 共用同一份 closed Plan。外层只做
// 一次 Attach，acquire 并交叉校验 closed/frontier/fatal，然后缓存 N。
// Plan closed 后 closed/frontier 不再改变；每张票不得重复读取它们。
template <typename ConsumerOps>
PA_SIMT_PLAN_DEVICE bool AttachClosedPlan(
    const aicpu_plan::RuntimePlanView &view, uint32_t &attached_task_count
)
{
    attached_task_count = 0U;
    if (view.control == nullptr || view.cells == nullptr ||
        view.capacity == 0U) {
        return false;
    }
    const int64_t closed = ConsumerOps::LoadControl(
        &view.control->closed_task_count.value
    );
    const int64_t frontier = ConsumerOps::LoadControl(
        &view.control->planned_frontier.value
    );
    const int64_t fatal = ConsumerOps::LoadControl(
        &view.control->fatal.value
    );
    if (fatal != 0 || closed < 0 || closed != frontier ||
        closed > static_cast<int64_t>(view.capacity)) {
        return false;
    }
    attached_task_count = static_cast<uint32_t>(closed);
    return true;
}

// Attach 成功后的唯一热路 atomic 是 build_next FetchAdd。区别只在
// 消费者人口：Scalar 是 96 个 worker，SIMT 是 4 个 warp leader，
// 因此正常终态分别是 N+96 与 N+4。ticket 只分配 task id；
// task kind、engine、Tensor/scalar 和 adapter provenance 一律从 PlanCell
// 获取。fatal 在 Build 失败、completion 等待和 arrival/release 边界观察。
template <typename ConsumerOps>
PA_SIMT_PLAN_DEVICE aicpu_plan::BuildReservation TakeAttachedBuildTicket(
    const aicpu_plan::RuntimePlanView &view, uint32_t attached_task_count
)
{
    if (view.control == nullptr || view.cells == nullptr ||
        view.capacity == 0U || attached_task_count > view.capacity) {
        return {aicpu_plan::BuildReservationStatus::Fatal, 0U};
    }
    const int64_t ticket = ConsumerOps::FetchAddControl(
        &view.control->build_next.value, 1
    );
    if (ticket < 0) {
        return {aicpu_plan::BuildReservationStatus::Fatal, 0U};
    }
    if (ticket >= static_cast<int64_t>(attached_task_count)) {
        return {aicpu_plan::BuildReservationStatus::Closed, 0U};
    }
    return {
        aicpu_plan::BuildReservationStatus::Reserved,
        static_cast<uint32_t>(ticket),
    };
}

template <typename ConsumerOps>
PA_SIMT_PLAN_DEVICE aicpu_plan::BuildArrivalStatus ArriveBuilderLeaderOnce(
    const aicpu_plan::RuntimePlanView &view, uint32_t leader_id,
    BuilderLeaderLocalState &local_state
)
{
    if (RecordBuilderLeaderArrivalLocally(leader_id, local_state) !=
        LocalLeaderArrivalStatus::First) {
        return aicpu_plan::BuildArrivalStatus::Invalid;
    }
    return aicpu_plan::ArriveClosedPlanBuildWorker<ConsumerOps>(
        view, kBuilderLeaders
    );
}

template <typename ConsumerOps>
PA_SIMT_PLAN_DEVICE bool PublishBuildRelease(
    const aicpu_plan::RuntimePlanView &view, uint32_t task_count
)
{
    return aicpu_plan::PublishClosedPlanBuildRelease<ConsumerOps>(
        view, task_count, kBuilderLeaders
    );
}

PA_SIMT_PLAN_DEVICE constexpr uint64_t ExpectedBuildTicketCount(
    uint32_t task_count
)
{
    return static_cast<uint64_t>(task_count) + kBuilderLeaders;
}

}  // namespace pa_scheduler::aicpu_plan_simt

#if defined(PA_SIMT_PLAN_UNDEFINE_DEVICE)
#undef PA_SIMT_PLAN_DEVICE
#undef PA_SIMT_PLAN_UNDEFINE_DEVICE
#endif

#endif  // PA_SCHEDULER_AICPU_PLAN_ORDINARY_SIMT_BUILD_PROTOCOL_H
