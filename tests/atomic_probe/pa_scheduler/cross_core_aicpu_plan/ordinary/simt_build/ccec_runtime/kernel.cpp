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

#include "../common/simt_real_state_runtime.h"
#include "runtime_report.h"

PTO_SYNCALL_MIX_AIC_KERNEL_META(
    aicpu_plan_simt_runtime_0_mix_aiv, 1, 2
);

namespace {

namespace plan = pa_scheduler::aicpu_plan;
namespace simt = pa_scheduler::aicpu_plan_simt;
namespace gate = pa_scheduler::aicpu_plan_simt::runtime_gate;

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
        // SIMT writer 的 canonical payload 全部由 asc_stcg bypass 写入；
        // writer 只需要 publication fence，DCCI 只属于 reader acquire。
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

// 公共层只校验 canonical engine/function 自洽性，不解释 PA kind、固定
// function table、adapter_data 或 task-id 周期。具体算子的 route policy
// 后续由 adapter 注入；这个 gate 不把 PA 示例固化进公共 SIMT builder。
struct CanonicalRoutePolicy {
    PA_DEVICE bool Validate(
        uint32_t task_id, uint8_t adapter_flags,
        uint16_t adapter_data, uint32_t function_id,
        plan::EngineClass engine
    ) const
    {
        (void)task_id;
        (void)adapter_flags;
        (void)adapter_data;
        if (engine == plan::EngineClass::MetadataOnly) {
            return function_id == plan::kInvalidFunctionId;
        }
        return (engine == plan::EngineClass::Aic ||
                engine == plan::EngineClass::Aiv) &&
               function_id != plan::kInvalidFunctionId;
    }
};

using Runtime =
    simt::SimtRealStateRuntime<SimtOps, CanonicalRoutePolicy>;

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

PA_DEVICE void PublishLeaderReport(
    __gm__ gate::ReportLine *line, const uint64_t *values
)
{
    if (line == nullptr) return;
    __gm__ volatile uint64_t *words = line->words;
    // magic 最后发布；AIV0 join 后仍会逐 cache line invalidate 再读取。
    for (uint32_t word = 1U; word < gate::kReportWords; ++word) {
        SimtOps::StorePayloadWord(&words[word], values[word]);
    }
    SimtOps::StoreBarrier();
    SimtOps::StorePayloadWord(&words[0], gate::kLeaderMagic);
    SimtOps::StoreBarrier();
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(128) void
BuildClosedCanonicalPlanVf(
    __gm__ pa_scheduler::SchedulerState *state,
    __gm__ gate::RuntimeReport *report
)
{
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    if (!simt::IsBuilderLeader(thread)) return;
    const uint32_t leader = simt::BuilderLeaderId(thread);
    const uint32_t build_owner = simt::SimtBuildOwner(leader);
    uint64_t values[gate::kReportWords]{};
    values[gate::LeaderBeginClock] = SimtOps::Now();
    values[gate::LeaderId] = leader;
    values[gate::LeaderBuildOwner] = build_owner;
    values[gate::LeaderLastTask] = UINT32_MAX;
    values[gate::LeaderLastBuildStatus] =
        static_cast<uint64_t>(simt::SimtTaskBuildStatus::Published);
    values[gate::LeaderReservationStatus] =
        static_cast<uint64_t>(plan::BuildReservationStatus::Fatal);
    values[gate::LeaderArrivalStatus] =
        static_cast<uint64_t>(plan::BuildArrivalStatus::Invalid);

    plan::RuntimePlanView view{nullptr, nullptr, 0U};
    uint32_t attached_task_count = 0U;
    bool attached = false;
    if (state != nullptr) {
        // StorageRef 是 Host/AICPU ordinary wire line；四个 leader 都在
        // Attach 前自行 acquire，不能依赖 Scalar 间 cache coherence。
        SimtOps::InvalidateRegion(
            &state->runtime_plan_storage,
            sizeof(state->runtime_plan_storage)
        );
        // 窄 Runtime 的唯一 ordinary SchedulerState 配置读取是 heap
        // window/base/size。它们各自与 fatal atomic 分线，Attach 时一次
        // acquire 后在 leader 本地复用，不能假定 Host 写入已在 DCache。
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
    values[gate::LeaderAttachedTaskCount] = attached_task_count;

    Runtime runtime{
        state,
        view.capacity,
        UINT32_MAX,
        pa_scheduler::kWatchdogTicks,
        CanonicalRoutePolicy{},
    };
    bool local_ok = attached && runtime.Valid() &&
                    SimtOps::Load(&state->fatal.value) == 0;
    if (!local_ok) PublishFatal(state);

    while (local_ok) {
        const plan::BuildReservation reservation =
            simt::TakeAttachedBuildTicket<SimtOps>(
                view, attached_task_count
            );
        values[gate::LeaderReservationStatus] =
            static_cast<uint64_t>(reservation.status);
        if (reservation.status ==
            plan::BuildReservationStatus::Reserved) {
            values[gate::LeaderLastTask] = reservation.task_id;
            ++values[gate::LeaderReservedTasks];
            simt::SimtTaskBuildScratch scratch{};
            simt::SimtTaskBuildStatus status =
                simt::SimtTaskBuildStatus::InvalidPlanControl;
            if (runtime.BindTask(reservation.task_id)) {
                status = simt::BuildCanonicalPlanTask<SimtOps>(
                    runtime, view, reservation.task_id,
                    build_owner, scratch
                );
            } else {
                PublishFatal(state);
            }
            values[gate::LeaderLastBuildStatus] =
                static_cast<uint64_t>(status);
            if (status != simt::SimtTaskBuildStatus::Published) {
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

    // 即使 Build/Attach 已发布 fatal，也必须走唯一一次 arrival 调用并
    // 收口本 leader；公共 arrival 会返回 Fatal，而不会错误发布 release。
    simt::BuilderLeaderLocalState arrival_state{};
    const plan::BuildArrivalStatus arrival =
        simt::ArriveBuilderLeaderOnce<SimtOps>(
            view, leader, arrival_state
        );
    values[gate::LeaderArrivalStatus] =
        static_cast<uint64_t>(arrival);
    if (arrival == plan::BuildArrivalStatus::Last) {
        const bool released = simt::PublishBuildRelease<SimtOps>(
            view, attached_task_count
        );
        values[gate::LeaderReleasePublished] = released ? 1U : 0U;
        if (!released) PublishFatal(state);
    }
    if (!local_ok || arrival == plan::BuildArrivalStatus::Fatal ||
        arrival == plan::BuildArrivalStatus::Invalid) {
        values[gate::LeaderLocalFatal] = 1U;
    }
    values[gate::LeaderEndClock] = SimtOps::Now();
    PublishLeaderReport(
        report == nullptr ? nullptr : &report->leaders[leader],
        values
    );
}

// 只用于把 mixed AIV metadata 明确锁定为 SIMD_SIMT_MIX_VF=4。正常
// SchedulerState 指针非空，运行期不会进入该 metadata anchor。
static __simd_vf__ __aicore__ void
SimdMetadataAnchor(__ubuf__ uint32_t *scratch)
{
    scratch[0] = scratch[0] + 1U;
}

__aicore__ inline void StoreScalarReportWord(
    __gm__ uint64_t *address, uint64_t value
)
{
    // Scalar identity 不能调用 SIMT-only asc_stcg；st_dev 是同一类
    // bypass-DCache GM publication，并且由后续 dsb 收口。
    __builtin_cce_st_dev(value, address, 0);
}

__aicore__ inline uint64_t LoadReportWord(
    __gm__ uint64_t *address
)
{
    return *address;
}

__aicore__ inline int64_t ObserveControl(
    __gm__ volatile int64_t *address
)
{
    return atomicAdd(
        const_cast<__gm__ int64_t *>(address), int64_t{0}
    );
}

__aicore__ inline int32_t ObserveControl(
    __gm__ volatile int32_t *address
)
{
    return atomicAdd(
        const_cast<__gm__ int32_t *>(address), int32_t{0}
    );
}

}  // namespace

extern "C" __aicore__ __attribute__((noinline, visibility("hidden")))
void aicpu_plan_simt_scalar_continuation(
    __gm__ pa_scheduler::SchedulerState *state,
    __gm__ gate::RuntimeReport *report,
    uint32_t task_count, uint64_t validation,
    uint64_t join_clock
);

extern "C" __global__ __aicore__ void
aicpu_plan_simt_runtime_0_mix_aiv(
    __gm__ pa_scheduler::SchedulerState *state,
    __gm__ gate::RuntimeReport *report,
    uint32_t reserved_flags
)
{
    if (reserved_flags == UINT32_MAX) {
        SimdMetadataAnchor(reinterpret_cast<__ubuf__ uint32_t *>(0));
    }
    if (reserved_flags != 0U && state != nullptr) {
        (void)atomicExch(
            const_cast<__gm__ int32_t *>(&state->fatal.value),
            int32_t{1}
        );
        (void)atomicExch(
            const_cast<__gm__ int64_t *>(
                &state->runtime_plan_control.fatal.value
            ), int64_t{1}
        );
    }
    if (report != nullptr &&
        (reinterpret_cast<uintptr_t>(report) %
         gate::kReportLineBytes) != 0U) {
        if (state != nullptr) {
            (void)atomicExch(
                const_cast<__gm__ int32_t *>(&state->fatal.value),
                int32_t{1}
            );
            (void)atomicExch(
                const_cast<__gm__ int64_t *>(
                    &state->runtime_plan_control.fatal.value
                ), int64_t{1}
            );
        }
        // 错误 ABI 仍启动并 join VF，但禁止任何 writer 触碰错位 report。
        report = nullptr;
    }

    // 性能窗口的唯一开始时间必须位于 VF launch 之前，不能让后续
    // Scalar continuation 把 SIMT Build 时间从端到端口径中漏掉。
    const uint64_t begin_clock = static_cast<uint64_t>(get_sys_cnt());
    if (report != nullptr) {
        __gm__ uint64_t *launch = report->launch.words;
        for (uint32_t word = 0U; word < gate::kReportWords; ++word) {
            StoreScalarReportWord(launch + word, 0U);
        }
        StoreScalarReportWord(
            launch + gate::LaunchBeginClock, begin_clock
        );
        dsb((mem_dsb_t)0);
        StoreScalarReportWord(
            launch + gate::LaunchMagic, gate::kLaunchMagic
        );
        dsb((mem_dsb_t)0);
    }

    cce::async_invoke<BuildClosedCanonicalPlanVf>(
        cce::dim3{simt::kBuilderThreads, 1U, 1U}, state, report
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    // fatal 路径也只能从这里越过 V->S join。AIV0 对四条独占 leader
    // report 逐行 invalidate，再验证 N+4 ticket、四次唯一 arrival、真实
    // Exec owner 0..3 与 release；物理 leader identity 使用独立 report
    // 字段，不能依赖 VF writer 的本地 cache 状态。
    const uint64_t join_clock = static_cast<uint64_t>(get_sys_cnt());
    uint64_t validation = gate::kSuccessValidationMask;
    uint64_t reserved_tasks = 0U;
    uint32_t last_arrivals = 0U;
    uint32_t release_publishers = 0U;
    uint32_t attached_task_count = 0U;
    bool attached_count_initialized = false;
    if (report == nullptr) {
        validation = 0U;
    } else {
        for (uint32_t leader = 0U;
             leader < simt::kBuilderLeaders; ++leader) {
            __gm__ gate::ReportLine *line = &report->leaders[leader];
            dcci(
                reinterpret_cast<__gm__ uint8_t *>(line),
                SINGLE_CACHE_LINE
            );
            dcci(
                reinterpret_cast<__gm__ uint8_t *>(line) + 64U,
                SINGLE_CACHE_LINE
            );
        }
        dsb((mem_dsb_t)0);
        __asm__ volatile("" ::: "memory");
        for (uint32_t leader = 0U;
             leader < simt::kBuilderLeaders; ++leader) {
            __gm__ uint64_t *words = report->leaders[leader].words;
            if (LoadReportWord(words + gate::LeaderMagic) !=
                gate::kLeaderMagic) {
                validation &= ~gate::ValidationLeaderMagic;
            }
            if (LoadReportWord(words + gate::LeaderId) != leader ||
                LoadReportWord(words + gate::LeaderBuildOwner) !=
                    simt::SimtBuildOwner(leader)) {
                validation &= ~gate::ValidationLeaderIdentity;
            }
            const uint32_t count = static_cast<uint32_t>(
                LoadReportWord(words + gate::LeaderAttachedTaskCount)
            );
            if (!attached_count_initialized) {
                attached_task_count = count;
                attached_count_initialized = true;
            } else if (count != attached_task_count) {
                validation &= ~gate::ValidationAttachedCount;
            }
            reserved_tasks +=
                LoadReportWord(words + gate::LeaderReservedTasks);
            if (LoadReportWord(words + gate::LeaderLastBuildStatus) !=
                    static_cast<uint64_t>(
                        simt::SimtTaskBuildStatus::Published
                    ) ||
                LoadReportWord(words + gate::LeaderLocalFatal) != 0U) {
                validation &= ~gate::ValidationBuildStatus;
            }
            if (LoadReportWord(words + gate::LeaderReservationStatus) !=
                    static_cast<uint64_t>(
                        plan::BuildReservationStatus::Closed
                    )) {
                validation &= ~gate::ValidationTicketTerminal;
            }
            const uint64_t arrival =
                LoadReportWord(words + gate::LeaderArrivalStatus);
            if (arrival == static_cast<uint64_t>(
                               plan::BuildArrivalStatus::Last
                           )) {
                ++last_arrivals;
            }
            if (arrival != static_cast<uint64_t>(
                               plan::BuildArrivalStatus::Arrived
                           ) &&
                arrival != static_cast<uint64_t>(
                               plan::BuildArrivalStatus::Last
                           )) {
                validation &= ~gate::ValidationArrivalOnce;
            }
            release_publishers += static_cast<uint32_t>(
                LoadReportWord(words + gate::LeaderReleasePublished)
            );
        }
    }

    int64_t build_next = -1;
    int64_t workers_done = -1;
    int64_t release = plan::kBuildReleaseFailed;
    int64_t plan_fatal = 1;
    int32_t scheduler_fatal = 1;
    if (state != nullptr) {
        build_next = ObserveControl(
            &state->runtime_plan_control.build_next.value
        );
        workers_done = ObserveControl(
            &state->runtime_plan_control.build_workers_done.value
        );
        release = ObserveControl(
            &state->runtime_plan_control.build_release.value
        );
        plan_fatal = ObserveControl(
            &state->runtime_plan_control.fatal.value
        );
        scheduler_fatal = ObserveControl(&state->fatal.value);
    }

    if (reserved_tasks != attached_task_count) {
        validation &= ~gate::ValidationReservedCount;
    }
    if (plan_fatal == 0 && scheduler_fatal == 0) {
        if (build_next != static_cast<int64_t>(
                              attached_task_count +
                              simt::kBuilderLeaders
                          )) {
            validation &= ~gate::ValidationTicketTerminal;
        }
        if (workers_done !=
                static_cast<int64_t>(simt::kBuilderLeaders) ||
            last_arrivals != 1U) {
            validation &= ~gate::ValidationArrivalOnce;
        }
        if (release != static_cast<int64_t>(attached_task_count) ||
            release_publishers != 1U) {
            validation &= ~gate::ValidationRelease;
        }
    } else {
        // fatal 不是绕过 join 的理由；走到此处本身就是 failure path 已
        // 完成 VF join 的机器码证据，release 允许保持 Pending/Failed。
        validation = gate::ValidationFatalPathJoined |
                     (validation & (gate::ValidationLeaderMagic |
                                    gate::ValidationLeaderIdentity |
                                    gate::ValidationAttachedCount));
    }

    if (state != nullptr &&
        ((plan_fatal == 0 && scheduler_fatal == 0 &&
          validation != gate::kSuccessValidationMask) ||
         report == nullptr)) {
        (void)atomicExch(
            const_cast<__gm__ int32_t *>(&state->fatal.value),
            int32_t{1}
        );
        (void)atomicExch(
            const_cast<__gm__ int64_t *>(
                &state->runtime_plan_control.fatal.value
            ), int64_t{1}
        );
        plan_fatal = 1;
        scheduler_fatal = 1;
    }

    if (report != nullptr) {
        __gm__ uint64_t *launch = report->launch.words;
        StoreScalarReportWord(
            launch + gate::LaunchJoinClock, join_clock
        );
        StoreScalarReportWord(
            launch + gate::LaunchClosedTaskCount,
            attached_task_count
        );
        StoreScalarReportWord(
            launch + gate::LaunchValidation, validation
        );
        StoreScalarReportWord(
            launch + gate::LaunchTicketsObserved,
            static_cast<uint64_t>(build_next)
        );
        StoreScalarReportWord(
            launch + gate::LaunchArrivalsObserved,
            static_cast<uint64_t>(workers_done)
        );
        StoreScalarReportWord(
            launch + gate::LaunchReleaseObserved,
            static_cast<uint64_t>(release)
        );
        StoreScalarReportWord(
            launch + gate::LaunchPlanFatalObserved,
            static_cast<uint64_t>(plan_fatal)
        );
        StoreScalarReportWord(
            launch + gate::LaunchSchedulerFatalObserved,
            static_cast<uint64_t>(scheduler_fatal)
        );
        StoreScalarReportWord(
            launch + gate::LaunchReservedTasksObserved,
            reserved_tasks
        );
        dsb((mem_dsb_t)0);
    }

    // 该符号必须由另一份 Scalar TU 定义。当前 gate continuation 至少
    // acquire build_release/fatal 并发布独占 GM report；它不是空占位。
    aicpu_plan_simt_scalar_continuation(
        state, report, attached_task_count,
        validation, join_clock
    );
}
