/* Machine-code gate for the narrow SIMT builder's real-state adapter. */

#include "cce_aicore_intrinsics.h"
#include "simt_api/asc_simt.h"

#include <cstddef>
#include <cstdint>

#if defined(PA_SIMT_REAL_STATE_GATE_NOINLINE)
#define PA_DEVICE \
    __simt_callee__ __aicore__ __attribute__((noinline)) inline
#else
#define PA_DEVICE \
    __simt_callee__ __aicore__ __attribute__((always_inline)) inline
#endif
#define PA_GM __gm__

#include "../common/simt_real_state_runtime.h"

#ifndef PA_SIMT_REAL_STATE_GATE_STAGE
#define PA_SIMT_REAL_STATE_GATE_STAGE 5
#endif

namespace {

namespace plan = pa_scheduler::aicpu_plan;
namespace simt = pa_scheduler::aicpu_plan_simt;

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

    PA_DEVICE static void FlushRegion(
        __gm__ void *, uint64_t
    )
    {
        // 本 TU 的 descriptor/history/ordinary/exec payload 全部逐 word
        // 使用 asc_stcg；writer 只做 fence，不执行 reader-side DCCI。
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
        const uint64_t start = raw & ~uint64_t{63};
        const uint64_t end =
            (raw + bytes + 63U) & ~uint64_t{63};
        for (uint64_t current = start; current < end; current += 64U) {
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

struct GateRoutePolicy {
    PA_DEVICE bool Validate(
        uint32_t task_id, uint8_t adapter_flags,
        uint16_t adapter_data, uint32_t function_id,
        plan::EngineClass engine
    ) const
    {
        const bool metadata =
            engine == plan::EngineClass::MetadataOnly;
        return adapter_flags != 0U && adapter_data <= task_id &&
               (metadata
                    ? function_id == plan::kInvalidFunctionId
                    : function_id != plan::kInvalidFunctionId);
    }
};

using Runtime = simt::SimtRealStateRuntime<SimtOps, GateRoutePolicy>;

static __simt_vf__ __aicore__ LAUNCH_BOUND(128) void
RealStateBuildVf(
    __gm__ pa_scheduler::SchedulerState *state,
    __gm__ plan::RuntimeTaskPlanCell *plan_cells,
    uint32_t task_capacity, uint32_t first_task,
    __gm__ uint64_t *reports
)
{
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    if (!simt::IsBuilderLeader(thread)) return;
    const uint32_t leader = simt::BuilderLeaderId(thread);
    const uint32_t task_id = first_task + leader;
    if (task_id >= task_capacity) return;
    Runtime runtime{
        state, task_capacity, UINT32_MAX,
        pa_scheduler::kWatchdogTicks, GateRoutePolicy{},
    };
    simt::SimtTaskBuildStatus status =
        simt::SimtTaskBuildStatus::InvalidPlanControl;
    simt::SimtTaskBuildScratch scratch{};
    if (runtime.BindTask(task_id)) {
#if PA_SIMT_REAL_STATE_GATE_STAGE == 0
        status = runtime.OutputPublished(task_id, 0U) != nullptr &&
                         runtime.OutputLastWriter(task_id, 0U) != nullptr &&
                         runtime.InsertCompletion(task_id) != nullptr &&
                         runtime.WriterHistory(task_id) != nullptr
                     ? simt::SimtTaskBuildStatus::Published
                     : simt::SimtTaskBuildStatus::InvalidPlanControl;
#elif PA_SIMT_REAL_STATE_GATE_STAGE == 1
        status = runtime.PublishMetadataCompletion(task_id, 0U)
                     ? simt::SimtTaskBuildStatus::Published
                     : simt::SimtTaskBuildStatus::MetadataCompletionFailed;
#elif PA_SIMT_REAL_STATE_GATE_STAGE == 2
        uint64_t first_address = 0U;
        uint64_t vend = 0U;
        status = runtime.ReserveOutputHeap(
                     task_id, 1024U, first_address, vend
                 ) && first_address != 0U && vend != 0U
                     ? simt::SimtTaskBuildStatus::Published
                     : simt::SimtTaskBuildStatus::MaterializeFailed;
#elif PA_SIMT_REAL_STATE_GATE_STAGE == 3
        simt::SimtWriterRegion region{
            UINT64_C(0x100000000), 0U, 64U,
            static_cast<int32_t>(task_id), 0U,
        };
        const uint16_t bucket = static_cast<uint16_t>(
            runtime.OrdinaryBucket(region.buffer_addr)
        );
        const uint8_t ordinal = 0U;
        status = runtime.CheckOrdinaryAppend(
                     &region, &bucket, &ordinal, 1U, task_id
                 ) && runtime.AppendOrdinary(
                     &region, &bucket, 1U, task_id
                 )
                     ? simt::SimtTaskBuildStatus::Published
                     : simt::SimtTaskBuildStatus::WriterPublishFailed;
#elif PA_SIMT_REAL_STATE_GATE_STAGE == 4
        simt::SimtCanonicalTensorDesc descriptor{};
        descriptor.buffer_addr = UINT64_C(0x100000000);
        descriptor.buffer_size = 64U;
        descriptor.owner_task_id = simt::kSimtInvalidTaskOwner;
        descriptor.ndims = 1U;
        descriptor.dtype = 0U;
        descriptor.is_contiguous = 1U;
        descriptor.shapes[0] = 16U;
        descriptor.extent_elem_cache = 16U;
        descriptor.strides[0] = 1U;
        int32_t producer = -1;
        status = runtime.LookupOrdinary(
                     descriptor, task_id, producer
                 )
                     ? simt::SimtTaskBuildStatus::Published
                     : simt::SimtTaskBuildStatus::FaninFailed;
#elif PA_SIMT_REAL_STATE_GATE_STAGE == 6
        status = runtime.PublishBuildFatal(task_id, 1U)
                     ? simt::SimtTaskBuildStatus::Published
                     : simt::SimtTaskBuildStatus::InvalidPlanControl;
#elif PA_SIMT_REAL_STATE_GATE_STAGE == 7
        int32_t lower_bound = -1;
        status = runtime.FaninLowerBound(task_id, lower_bound) &&
                         lower_bound >= 0
                     ? simt::SimtTaskBuildStatus::Published
                     : simt::SimtTaskBuildStatus::FaninFailed;
#elif PA_SIMT_REAL_STATE_GATE_STAGE == 8
        status = runtime.ValidateAdapterRoute(
                     1U, static_cast<uint16_t>(task_id),
                     0U, plan::EngineClass::Aic
                 )
                     ? simt::SimtTaskBuildStatus::Published
                     : simt::SimtTaskBuildStatus::InvalidAdapterRoute;
#elif PA_SIMT_REAL_STATE_GATE_STAGE >= 10 && \
      PA_SIMT_REAL_STATE_GATE_STAGE <= 17
        const plan::RuntimePlanView view{
            &state->runtime_plan_control,
            plan_cells, task_capacity,
        };
        status = simt::AcquireCanonicalPlanTask<SimtOps>(
            view, task_id, scratch
        );
#if PA_SIMT_REAL_STATE_GATE_STAGE >= 11
        if (status == simt::SimtTaskBuildStatus::Published &&
            !runtime.ValidateAdapterRoute(
                scratch.header.adapter_flags,
                scratch.header.adapter_data,
                scratch.header.function_id,
                static_cast<plan::EngineClass>(
                    scratch.header.engine_class
                )
            )) {
            status = simt::SimtTaskBuildStatus::InvalidAdapterRoute;
        }
#endif
#if PA_SIMT_REAL_STATE_GATE_STAGE >= 12
        if (status == simt::SimtTaskBuildStatus::Published &&
            !simt::MaterializeAndPublishOutputs<SimtOps>(
                runtime, task_id, scratch
            )) {
            status = simt::SimtTaskBuildStatus::MaterializeFailed;
        }
#endif
#if PA_SIMT_REAL_STATE_GATE_STAGE >= 13
        if (status == simt::SimtTaskBuildStatus::Published &&
            !simt::PrepareWriterDelta(runtime, task_id, scratch)) {
            status = simt::SimtTaskBuildStatus::WriterPublishFailed;
        }
#endif
#if PA_SIMT_REAL_STATE_GATE_STAGE >= 14
        if (status == simt::SimtTaskBuildStatus::Published &&
            !simt::WaitForInsertPredecessor<SimtOps>(
                runtime, task_id
            )) {
            status = simt::SimtTaskBuildStatus::InsertWaitFailed;
        }
#endif
#if PA_SIMT_REAL_STATE_GATE_STAGE >= 15
        if (status == simt::SimtTaskBuildStatus::Published &&
            !simt::PublishWriterMetadataAndCompletion<SimtOps>(
                runtime, task_id, scratch
            )) {
            status = simt::SimtTaskBuildStatus::WriterPublishFailed;
        }
#endif
#if PA_SIMT_REAL_STATE_GATE_STAGE >= 16
        if (status == simt::SimtTaskBuildStatus::Published &&
            !simt::CollectFaninAndResolveTensors<SimtOps>(
                runtime, task_id, scratch
            )) {
            status = simt::SimtTaskBuildStatus::FaninFailed;
        }
#endif
#if PA_SIMT_REAL_STATE_GATE_STAGE >= 17
        if (status == simt::SimtTaskBuildStatus::Published &&
            !simt::PublishTerminalBuildResult<SimtOps>(
                runtime, task_id, simt::SimtBuildOwner(leader), scratch
            )) {
            status = simt::SimtTaskBuildStatus::ExecPublishFailed;
        }
#endif
#else
        const plan::RuntimePlanView view{
            &state->runtime_plan_control,
            plan_cells, task_capacity,
        };
        status = simt::BuildCanonicalPlanTask<SimtOps>(
            runtime, view, task_id,
            simt::SimtBuildOwner(leader), scratch
        );
#endif
    }
    asc_stcg(
        reports + leader,
        static_cast<uint64_t>(status) |
            (static_cast<uint64_t>(scratch.ordinary_count) << 8U) |
            (static_cast<uint64_t>(scratch.symbol_count) << 16U) |
            (static_cast<uint64_t>(scratch.fanin_count) << 24U) |
            (static_cast<uint64_t>(simt::SimtBuildOwner(leader)) << 32U)
    );
}

}  // namespace

extern "C" __global__ __aicore__ void
aicpu_plan_simt_real_state_runtime_gate_0_mix_aiv(
    __gm__ pa_scheduler::SchedulerState *state,
    __gm__ pa_scheduler::aicpu_plan::RuntimeTaskPlanCell *plan_cells,
    uint32_t task_capacity, uint32_t first_task,
    __gm__ uint64_t *reports
)
{
    cce::async_invoke<RealStateBuildVf>(
        cce::dim3(128U), state, plan_cells,
        task_capacity, first_task, reports
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
}
