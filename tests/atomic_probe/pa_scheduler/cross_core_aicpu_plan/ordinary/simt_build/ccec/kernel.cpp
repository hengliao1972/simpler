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

#include <pto/common/kernel_meta.hpp>

#include "cce_aicore_intrinsics.h"
#include "simt_api/asc_simt.h"

#include <cstddef>
#include <cstdint>

#define PA_DEVICE \
    __simt_callee__ __aicore__ __attribute__((always_inline)) inline
#define PA_GM __gm__

#include "../adapter/pa_simt_route_policy.h"
#include "../common/simt_real_state_runtime.h"

PTO_SYNCALL_MIX_AIC_KERNEL_META(pa_scheduler_0_mix_aiv, 1, 2);

namespace {

namespace plan = pa_scheduler::aicpu_plan;
namespace simt = pa_scheduler::aicpu_plan_simt;
namespace adapter = pa_scheduler::aicpu_plan_simt::adapter;

static_assert(
    pa_scheduler::kCompiledRuntimePlanBuildBackend ==
        pa_scheduler::RuntimePlanBuildBackend::Simt
);
static_assert(pa_scheduler::kRuntimePlanBuildWorkers == 4U);
static_assert(pa_scheduler::kWorkers == 96U);
static_assert(simt::kBuilderThreads == 128U);
static_assert(simt::kBuilderLeaders == 4U);

struct SimtOps {
    static constexpr bool kAtomicReturnReadyObserved = true;

    PA_DEVICE static int64_t LoadControl(
        __gm__ volatile int64_t *address
    )
    {
        return asc_atomic_add(
            const_cast<__gm__ int64_t *>(address), int64_t{0}
        );
    }

    PA_DEVICE static int64_t FetchAddControl(
        __gm__ volatile int64_t *address, int64_t value
    )
    {
        return asc_atomic_add(
            const_cast<__gm__ int64_t *>(address), value
        );
    }

    PA_DEVICE static void PublishControl(
        __gm__ volatile int64_t *address, int64_t value
    )
    {
        (void)asc_atomic_exch(
            const_cast<__gm__ int64_t *>(address), value
        );
    }

    PA_DEVICE static int32_t Load(
        __gm__ volatile int32_t *address
    )
    {
        return asc_atomic_add(
            const_cast<__gm__ int32_t *>(address), int32_t{0}
        );
    }

    PA_DEVICE static int64_t Load(
        __gm__ volatile int64_t *address
    )
    {
        return asc_atomic_add(
            const_cast<__gm__ int64_t *>(address), int64_t{0}
        );
    }

    PA_DEVICE static uint64_t Load(
        __gm__ volatile uint64_t *address
    )
    {
        return asc_atomic_add(
            const_cast<__gm__ uint64_t *>(address), uint64_t{0}
        );
    }

    PA_DEVICE static int32_t Exchange(
        __gm__ volatile int32_t *address, int32_t value
    )
    {
        return asc_atomic_exch(
            const_cast<__gm__ int32_t *>(address), value
        );
    }

    PA_DEVICE static int64_t Exchange(
        __gm__ volatile int64_t *address, int64_t value
    )
    {
        return asc_atomic_exch(
            const_cast<__gm__ int64_t *>(address), value
        );
    }

    PA_DEVICE static uint64_t Exchange(
        __gm__ volatile uint64_t *address, uint64_t value
    )
    {
        return asc_atomic_exch(
            const_cast<__gm__ uint64_t *>(address), value
        );
    }

    PA_DEVICE static int64_t CompareExchange(
        __gm__ volatile int64_t *address,
        int64_t expected, int64_t desired
    )
    {
        return asc_atomic_cas(
            const_cast<__gm__ int64_t *>(address), expected, desired
        );
    }

    PA_DEVICE static int64_t FetchAdd(
        __gm__ volatile int64_t *address, int64_t value
    )
    {
        return asc_atomic_add(
            const_cast<__gm__ int64_t *>(address), value
        );
    }

    PA_DEVICE static void StorePayloadWord(
        __gm__ volatile uint64_t *address, uint64_t value
    )
    {
        asc_stcg(const_cast<__gm__ uint64_t *>(address), value);
    }

    PA_DEVICE static void StoreBarrier()
    {
        asc_threadfence();
    }

    PA_DEVICE static uint64_t Now()
    {
        return clock();
    }

    PA_DEVICE static void SpinHint()
    {
    }

    PA_DEVICE static void PreloadBuildDestination(
        __gm__ void *, uint64_t
    )
    {
    }

    PA_DEVICE static void BeforeBuiltPublish(uint32_t)
    {
    }

    PA_DEVICE static void FlushRegion(__gm__ void *, uint64_t)
    {
        // VF writer 的 canonical payload 全部由 asc_stcg bypass 发布；
        // writer 只做 publication fence，DCCI 只属于 reader acquire。
        asc_threadfence();
    }

    PA_DEVICE static void FlushRegion(
        __gm__ volatile uint64_t *address, uint64_t bytes
    )
    {
        FlushRegion(
            static_cast<__gm__ void *>(
                const_cast<__gm__ uint64_t *>(address)
            ), bytes
        );
    }

    PA_DEVICE static void InvalidateRegion(
        __gm__ const void *address, uint64_t bytes
    )
    {
        if (bytes == 0U) return;
        const uint64_t raw = reinterpret_cast<uint64_t>(address);
        if (raw > UINT64_MAX - bytes - 63U) return;
        const uint64_t start = raw & ~uint64_t{63};
        const uint64_t end =
            (raw + bytes + 63U) & ~uint64_t{63};
        for (uint64_t current = start;
             current < end; current += 64U) {
            asc_dcci_single(
                reinterpret_cast<__gm__ void *>(current)
            );
        }
        asc_threadfence();
    }

    PA_DEVICE static void InvalidateRegion(
        __gm__ void *address, uint64_t bytes
    )
    {
        InvalidateRegion(
            static_cast<__gm__ const void *>(address), bytes
        );
    }
};

using Runtime = simt::SimtRealStateRuntime<
    SimtOps, adapter::PaSimtRoutePolicy
>;

PA_DEVICE void PublishFatal(
    __gm__ pa_scheduler::SchedulerState *state
)
{
    if (state == nullptr) return;
    (void)SimtOps::Exchange(&state->fatal.value, int32_t{1});
    (void)SimtOps::Exchange(
        &state->runtime_plan_control.fatal.value, int64_t{1}
    );
}

PA_DEVICE bool LastInsertCompletionPublished(
    Runtime &runtime, uint32_t task_count
)
{
    if (task_count == 0U) return true;
    __gm__ volatile int64_t *completion =
        runtime.InsertCompletion(task_count - 1U);
    return completion != nullptr &&
           SimtOps::Load(completion) ==
               static_cast<int64_t>(task_count - 1U);
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(128) void
BuildClosedCanonicalPlanVf(
    __gm__ pa_scheduler::SchedulerState *state,
    uint32_t identity_preflight_ok
)
{
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    if (!simt::IsBuilderLeader(thread) ||
        identity_preflight_ok == 0U) {
        return;
    }
    const uint32_t leader = simt::BuilderLeaderId(thread);
    const uint32_t build_owner = simt::SimtBuildOwner(leader);

    plan::RuntimePlanView view{nullptr, nullptr, 0U};
    uint32_t attached_task_count = 0U;
    bool attached = false;
    if (state != nullptr) {
        // A5 Scalar/VF 间没有 cache coherence。每个 leader 必须独立
        // acquire AICPU 发布的 storage ref，以及其后整个 Build 会复用的
        // heap window/base/size；不能依赖 AIV0 preflight 的 Scalar cache。
        SimtOps::InvalidateRegion(
            &state->runtime_plan_storage,
            sizeof(state->runtime_plan_storage)
        );
        SimtOps::InvalidateRegion(
            &state->heap_window, sizeof(state->heap_window)
        );
        SimtOps::InvalidateRegion(
            &state->heap_base,
            sizeof(state->heap_base) + sizeof(state->heap_size)
        );
        attached = plan::MakeRuntimePlanView(
                       &state->runtime_plan_control,
                       state->runtime_plan_storage, view
                   ) &&
                   simt::AttachClosedPlan<SimtOps>(
                       view, attached_task_count
                   );
    }

    Runtime runtime{
        state,
        view.capacity,
        UINT32_MAX,
        pa_scheduler::kWatchdogTicks,
        adapter::PaSimtRoutePolicy{},
    };
    bool local_ok = attached && runtime.Valid() &&
                    SimtOps::Load(&state->fatal.value) == 0;
    if (!local_ok) PublishFatal(state);

    while (local_ok) {
        const plan::BuildReservation reservation =
            simt::TakeAttachedBuildTicket<SimtOps>(
                view, attached_task_count
            );
        if (reservation.status ==
            plan::BuildReservationStatus::Reserved) {
            simt::SimtTaskBuildScratch scratch{};
            simt::SimtTaskBuildStatus status =
                simt::SimtTaskBuildStatus::InvalidPlanControl;
            if (runtime.BindTask(reservation.task_id)) {
                status = simt::BuildCanonicalPlanTask<SimtOps>(
                    runtime, view, reservation.task_id,
                    build_owner, scratch
                );
            }
            if (status != simt::SimtTaskBuildStatus::Published) {
                PublishFatal(state);
                local_ok = false;
            }
            continue;
        }
        if (reservation.status !=
            plan::BuildReservationStatus::Closed) {
            PublishFatal(state);
            local_ok = false;
        }
        break;
    }

    // fatal 也必须让四个真实 leader 各自且仅一次走 arrival。只有最后
    // arrival、全局无 fatal 且 completion[N-1]==N-1（N=0 例外）时，
    // 才允许发布 build_release；这证明严格 TensorMap 插入链已封口。
    simt::BuilderLeaderLocalState arrival_state{};
    const plan::BuildArrivalStatus arrival =
        simt::ArriveBuilderLeaderOnce<SimtOps>(
            view, leader, arrival_state
        );
    if (arrival == plan::BuildArrivalStatus::Last) {
        if (!LastInsertCompletionPublished(
                runtime, attached_task_count
            ) ||
            !simt::PublishBuildRelease<SimtOps>(
                view, attached_task_count
            )) {
            PublishFatal(state);
        }
    } else if (arrival == plan::BuildArrivalStatus::Fatal ||
               arrival == plan::BuildArrivalStatus::Invalid) {
        PublishFatal(state);
    }
}

// 该 anchor 只锁定 mixed metadata 中的 SIMD_SIMT_MIX_VF=4；正式状态
// 地址不可能是全 1，运行时不会进入。Build 本身使用上面的 SIMT VF。
static __simd_vf__ __aicore__ void
SimdMetadataAnchor(__ubuf__ uint32_t *scratch)
{
    scratch[0] = scratch[0] + 1U;
}

__aicore__ inline void AcquireBuildIdentity(
    __gm__ pa_scheduler::SchedulerState *state
)
{
    if (state == nullptr) return;
    dcci(
        reinterpret_cast<__gm__ uint8_t *>(&state->config),
        SINGLE_CACHE_LINE
    );
    dcci(
        reinterpret_cast<__gm__ uint8_t *>(&state->pmu_probe),
        SINGLE_CACHE_LINE
    );
    dsb((mem_dsb_t)0);
    __asm__ volatile("" ::: "memory");
}

}  // namespace

// 两个 helper 均由 scalar_build/ccec/kernel.cpp 的独立 AIV continuation
// object 提供，并由最终 version script 局部化。preflight 只读；continuation
// 内部拥有真正的 CcecOps 与 RunScheduler 模板实例。
extern "C" __aicore__ __attribute__((noinline, visibility("hidden")))
bool pa_scheduler_simt_runtime_plan_preflight_aiv(
    __gm__ pa_scheduler::SchedulerState *state
);

extern "C" __aicore__ __attribute__((noinline, visibility("hidden")))
void pa_scheduler_simt_runtime_plan_continuation_aiv(
    __gm__ pa_scheduler::SchedulerState *state,
    uint32_t worker_id,
    uint64_t external_build_begin,
    uint64_t external_build_end,
    uint32_t external_build_active
);

extern "C" __global__ __aicore__ void
pa_scheduler_0_mix_aiv(
    __gm__ pa_scheduler::SchedulerState *state
)
{
    if (reinterpret_cast<uintptr_t>(state) == UINTPTR_MAX) {
        SimdMetadataAnchor(reinterpret_cast<__ubuf__ uint32_t *>(0));
    }

    const uint32_t vector_id = static_cast<uint32_t>(
        get_block_idx() * get_subblockdim() + get_subblockid()
    );
    const uint32_t worker_id = pa_scheduler::kAicWorkers + vector_id;
    if (vector_id != 0U) {
        pa_scheduler_simt_runtime_plan_continuation_aiv(
            state, worker_id, 0U, 0U, 0U
        );
        return;
    }

    // lifecycle/perf 起点必须覆盖本入口首次身份 DCCI 和只读 preflight，
    // 不能从 VF launch 前才开始而漏掉真实启动成本。
    const uint64_t external_build_begin =
        static_cast<uint64_t>(get_sys_cnt());
    // AIV0 在任何本入口 GM write 前 acquire 并完成只读身份核验。
    // 失败路径不启动 writer，但仍启动/等待空 VF，再进入第96个 Scalar
    // continuation；统一 fatal 由 RunScheduler 的 startup gate 发布。
    AcquireBuildIdentity(state);
    const bool identity_ok =
        pa_scheduler_simt_runtime_plan_preflight_aiv(state);
    cce::async_invoke<BuildClosedCanonicalPlanVf>(
        cce::dim3{simt::kBuilderThreads, 1U, 1U},
        state, identity_ok ? 1U : 0U
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    const uint64_t external_build_end =
        static_cast<uint64_t>(get_sys_cnt());

    pa_scheduler_simt_runtime_plan_continuation_aiv(
        state, pa_scheduler::kAicWorkers,
        external_build_begin, external_build_end, 1U
    );
}
