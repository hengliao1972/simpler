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

// 这个探针不另写一份 PA Build，也不缩短正式语义。它把 Scalar 正式公共
// 头按 VF 可达身份重新实例化，并从真实 static __simt_vf__ 调用完整的
// BuildRuntimePlanTask：Plan acquire/decode、Materialize、严格 ordinary
// TensorMap Register、Fanin 与 SharedExecCell publication 都必须参与代码生成。

#include <pto/common/kernel_meta.hpp>

#include "cce_aicore_intrinsics.h"
#include "simt_api/asc_simt.h"

#include <cstddef>
#include <cstdint>

#define PA_DEVICE \
    __aicore__ __attribute__((always_inline)) inline
#if defined(PA_FULL_BUILD_PROBE_REAL_NOINLINE)
#define PA_DEVICE_NOINLINE \
    static __aicore__ __attribute__((noinline))
#else
#define PA_DEVICE_NOINLINE \
    __aicore__ __attribute__((always_inline)) inline
#endif
#define PA_LOOP_NOUNROLL _Pragma("clang loop unroll(disable)")
#define PA_GM __gm__

// 仅在公共头 include 窗口把其中每个显式 __aicore__（包括局部 lambda）
// 提升成 VF 可达身份。CCEC 的 __aicore__ 本身是预定义宏，必须先撤销
// 再用等价 attribute 组合重建；include 结束恢复原定义，避免把 mixed
// entry 或 static __simt_vf__ 错标成 callee。这不是正式修复，只用于
// 继续暴露“公共头全体函数均 VF 可达”后的下一层编译约束。
#undef __aicore__
#define __aicore__ \
    __simt_callee__ __attribute__((cce_aicore))
#include "../../scalar_build/common/pa_scheduler_core.h"
#undef __aicore__
#define __aicore__ __attribute__((cce_aicore))
#undef PA_DEVICE
#undef PA_DEVICE_NOINLINE
#define PA_DEVICE \
    __simt_callee__ __aicore__ __attribute__((always_inline)) inline
#if defined(PA_FULL_BUILD_PROBE_REAL_NOINLINE)
#define PA_DEVICE_NOINLINE \
    static __simt_callee__ __aicore__ __attribute__((noinline))
#else
#define PA_DEVICE_NOINLINE PA_DEVICE
#endif

PTO_SYNCALL_MIX_AIC_KERNEL_META(aicpu_plan_simt_full_build_probe_0_mix_aiv, 1, 2);

namespace {

// 与 CcecOps 的正确性原语逐项对齐，但所有成员都继承上面的
// __simt_callee__ 身份。性能 hint 保持为空；可见性、返回型原子与 payload
// publication 不能因此被删掉。SIMT API 目前只暴露单行 DCCI 形状，因此
// 本探针只证明编译可达性，不把这一映射宣称成已通过 A5 的内存模型证据。
struct SimtOps {
    static constexpr bool kAtomicReturnReadyObserved = true;

    PA_DEVICE static int64_t LoadControl(
        __gm__ volatile int64_t *address
    ) {
        return asc_atomic_add(
            const_cast<__gm__ int64_t *>(address), int64_t{0}
        );
    }

    PA_DEVICE static int64_t FetchAddControl(
        __gm__ volatile int64_t *address, int64_t value
    ) {
        return asc_atomic_add(
            const_cast<__gm__ int64_t *>(address), value
        );
    }

    PA_DEVICE static void PublishControl(
        __gm__ volatile int64_t *address, int64_t value
    ) {
        (void)asc_atomic_exch(
            const_cast<__gm__ int64_t *>(address), value
        );
    }

    PA_DEVICE static int32_t Load(
        __gm__ volatile int32_t *address
    ) {
        return asc_atomic_add(
            const_cast<__gm__ int32_t *>(address), int32_t{0}
        );
    }

    PA_DEVICE static int64_t Load(
        __gm__ volatile int64_t *address
    ) {
        constexpr int64_t kIdentity = (-9223372036854775807LL - 1LL);
        return asc_atomic_max(
            const_cast<__gm__ int64_t *>(address), kIdentity
        );
    }

    PA_DEVICE static uint64_t Load(
        __gm__ volatile uint64_t *address
    ) {
        return asc_atomic_add(
            const_cast<__gm__ uint64_t *>(address), uint64_t{0}
        );
    }

    PA_DEVICE static int32_t Exchange(
        __gm__ volatile int32_t *address, int32_t value
    ) {
        return asc_atomic_exch(
            const_cast<__gm__ int32_t *>(address), value
        );
    }

    PA_DEVICE static int64_t Exchange(
        __gm__ volatile int64_t *address, int64_t value
    ) {
        return asc_atomic_exch(
            const_cast<__gm__ int64_t *>(address), value
        );
    }

    PA_DEVICE static uint64_t Exchange(
        __gm__ volatile uint64_t *address, uint64_t value
    ) {
        return asc_atomic_exch(
            const_cast<__gm__ uint64_t *>(address), value
        );
    }

    PA_DEVICE static int64_t CompareExchange(
        __gm__ volatile int64_t *address,
        int64_t expected, int64_t desired
    ) {
        return asc_atomic_cas(
            const_cast<__gm__ int64_t *>(address), expected, desired
        );
    }

    PA_DEVICE static int64_t FetchAdd(
        __gm__ volatile int64_t *address, int64_t value
    ) {
        return asc_atomic_add(
            const_cast<__gm__ int64_t *>(address), value
        );
    }

    PA_DEVICE static int64_t FetchMax(
        __gm__ volatile int64_t *address, int64_t value,
        uint64_t &retries
    ) {
        retries = 0U;
        return asc_atomic_max(
            const_cast<__gm__ int64_t *>(address), value
        );
    }

    PA_DEVICE static void StoreBarrier() {
        asc_threadfence();
    }

    PA_DEVICE static uint64_t Now() {
        return clock();
    }

    template <typename T>
    PA_DEVICE static uint64_t NowAfterAtomicResult(T value) {
        (void)value;
        return clock();
    }

    template <typename T>
    PA_DEVICE static T ForkAtomicResultForBranch(
        T value, T &dependency_value
    ) {
        dependency_value = value;
        return value;
    }

    PA_DEVICE static bool PmuWindowStart(
        __gm__ pa_scheduler::SchedulerState *, uint32_t
    ) {
        return false;
    }

    PA_DEVICE static void PmuWindowStop(
        __gm__ pa_scheduler::SchedulerState *, uint32_t, bool
    ) {}

    PA_DEVICE static void SpinHint() {}
    PA_DEVICE static void PreloadDataCache(__gm__ void *) {}
    PA_DEVICE static void PreloadBuildDestination(
        __gm__ void *, uint64_t
    ) {}
    PA_DEVICE static void PreloadPayloadSource(
        __gm__ const void *, uint64_t
    ) {}
    PA_DEVICE static void PreloadTokenDestination(
        __gm__ void *, uint64_t
    ) {}
    PA_DEVICE static void BeforeBuiltPublish(uint32_t) {}
    PA_DEVICE static void BeforePayloadAcquire(uint32_t) {}

    PA_DEVICE static void StorePayloadWord(
        __gm__ volatile uint64_t *address, uint64_t value
    ) {
        asc_stcg(
            const_cast<__gm__ uint64_t *>(address), value
        );
    }

    PA_DEVICE static void StoreTokenPayloadWord(
        __gm__ volatile uint64_t *address, uint64_t value
    ) {
        asc_stcg(
            const_cast<__gm__ uint64_t *>(address), value
        );
    }

    PA_DEVICE static uint64_t LoadPayloadWord(
        __gm__ const volatile uint64_t *address
    ) {
        return asc_ldcg(
            const_cast<__gm__ uint64_t *>(address)
        );
    }

    PA_DEVICE static void InvalidateRegion(
        __gm__ const void *address, uint64_t bytes
    ) {
        if (bytes == 0U) return;
        __gm__ uint8_t *start = const_cast<__gm__ uint8_t *>(
            reinterpret_cast<__gm__ const uint8_t *>(address)
        );
        const uint64_t lines = (bytes + 63U) / 64U;
        for (uint64_t line = 0U; line < lines; ++line) {
            asc_dcci_single(
                static_cast<__gm__ void *>(start + line * 64U)
            );
        }
        asc_threadfence();
    }

    PA_DEVICE static void FlushRegion(
        __gm__ void *address, uint64_t bytes
    ) {
        if (bytes == 0U) return;
        __gm__ uint8_t *start =
            reinterpret_cast<__gm__ uint8_t *>(address);
        const uint64_t lines = (bytes + 63U) / 64U;
        for (uint64_t line = 0U; line < lines; ++line) {
            asc_dcci_single(
                static_cast<__gm__ void *>(start + line * 64U)
            );
        }
        asc_threadfence();
    }

    PA_DEVICE static void Publish(
        __gm__ uint64_t *address, uint64_t value
    ) {
        asc_stcg(address, value);
    }

    PA_DEVICE static void Publish(
        __gm__ uint32_t *address, uint32_t value
    ) {
        asc_stcg(address, value);
    }
};

constexpr uint32_t kBuilderThreads = 128U;
constexpr uint32_t kWarpSize = 32U;
constexpr uint32_t kBuilderLeaders = 4U;
constexpr uint64_t kProbeSuccess = 0x53494D5446554C4CULL;  // "SIMTFULL"

static_assert(
    sizeof(pa_scheduler::TaskArgs) == 1280U,
    "probe must retain the real shared TaskArgs frame"
);
static_assert(
    sizeof(pa_scheduler::SubmitContext) == 408U,
    "probe must retain the real shared SubmitContext frame"
);
static_assert(
    sizeof(pa_scheduler::LocalStats) == 1216U,
    "probe must retain the real trace-free LocalStats frame"
);

static __simt_vf__ __aicore__ LAUNCH_BOUND(kBuilderThreads) void
BuildCanonicalPlanTaskWithScalarImplementation(
    __gm__ pa_scheduler::SchedulerState *state,
    __gm__ pa_scheduler::WorkerState *workers,
    __gm__ pa_scheduler::aicpu_plan::RuntimePlanControl *control,
    __gm__ pa_scheduler::aicpu_plan::RuntimeTaskPlanCell *cells,
    uint32_t capacity, uint32_t first_task_id,
    __gm__ uint64_t *leader_reports
) {
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    const uint32_t warp = thread / kWarpSize;
    const uint32_t lane = thread % kWarpSize;
    if (thread >= kBuilderThreads || lane != 0U ||
        warp >= kBuilderLeaders) {
        return;
    }

    const uint32_t task_id = first_task_id + warp;
    pa_scheduler::aicpu_plan::RuntimePlanView view{
        control, cells, capacity,
    };
    pa_scheduler::SubmitContext context{};
    pa_scheduler::LocalStats stats{};
    bool pmu_context = false;

#if defined(PA_FULL_BUILD_PROBE_PLAN_DECODE_ONLY)
    // 精确保留正式 Plan cell 的 control acquire、payload DCCI、canonical
    // validation 与 PA adapter decode，只切断后续 Materialize/Register。
    // 该门槛用于判断 local/GM pointer 问题是否已经发生在 Plan decoder。
    __gm__ pa_scheduler::aicpu_plan::RuntimeTaskPlanCell &cell =
        cells[task_id];
    const int64_t first = SimtOps::LoadControl(
        &cell.control.value
    );
    const pa_scheduler::aicpu_plan::DecodedPlanCellControl decoded =
        pa_scheduler::aicpu_plan::DecodePlanCellControl(first);
    SimtOps::InvalidateRegion(
        &cell.payload,
        static_cast<uint64_t>(decoded.payload_lines) *
            pa_scheduler::aicpu_plan::kPlanCacheLineBytes
    );
    pa_scheduler::aicpu_plan::RuntimeTaskPlanHeader header{};
    pa_scheduler::aicpu_plan::RuntimeTaskPlanLayout layout{};
    pa_scheduler::aicpu_plan_adapter::PaRuntimeTaskDecodeScratch scratch{};
    pa_scheduler::TaskArgs args{};
    const bool built =
        SimtOps::LoadControl(&cell.control.value) == first &&
        pa_scheduler::aicpu_plan::ValidateRuntimeTaskPlanPayload(
            cell.payload, task_id, decoded.payload_lines,
            header, layout
        ) &&
        pa_scheduler::aicpu_plan_adapter::DecodePaRuntimeTaskPlan(
            cell, header, layout, args, scratch
        );
    asc_stcg(
        leader_reports + warp,
        built ? (kProbeSuccess ^ static_cast<uint64_t>(
                     static_cast<uint32_t>(args.tensor_count)
                 )) : static_cast<uint64_t>(task_id)
    );
#elif defined(PA_FULL_BUILD_PROBE_WRITER_DELTA_DYNAMIC_ADDRESS_SPACE_ONLY) || \
      defined(PA_FULL_BUILD_PROBE_WRITER_DELTA_GM_ONLY)
    // TaskArgs 本身也用一个运行期 kind 区分 owner-local 与 GM
    // TensorDesc。这里不经过 exec adapter，单独强制实例化正式 writer
    // delta 扫描，判断完整 Build 的异地址空间问题是否更早已经发生。
    const uint64_t seed = asc_ldcg(
        leader_reports + kBuilderLeaders + warp
    );
#if defined(PA_FULL_BUILD_PROBE_WRITER_DELTA_GM_ONLY)
    const bool gm_source = true;
#else
    const bool gm_source = (seed & 1U) != 0U;
#endif
    pa_scheduler::TensorDesc local_tensor{};
    local_tensor.buffer_addr = (seed & ~uint64_t{4095}) + 4096U;
    local_tensor.buffer_size = 4096U;
    local_tensor.owner_task_id = task_id == 0U ?
        pa_scheduler::kInvalidTaskId : task_id - 1U;
    local_tensor.start_offset = 0U;
    local_tensor.ndims = 1U;
    local_tensor.dtype = pa_scheduler::DataType::Uint8;
    local_tensor.manual_dep = false;
    local_tensor.is_contiguous = true;
    local_tensor.shapes[0] = 64U;
    local_tensor.extent_elem_cache = 64U;
    pa_scheduler::TaskArgs dynamic_args{};
    dynamic_args.tags[0] = static_cast<int32_t>(
        pa_scheduler::TensorArgType::Inout
    );
    dynamic_args.tensor_count = 1;
    dynamic_args.scalar_count = 0;
    dynamic_args.has_error = false;
    if (gm_source) {
        dynamic_args.tensors[0].pointer.gm_tensor =
            &state->shared_map.shared_outputs[task_id].tensors[0];
        dynamic_args.tensors[0].kind =
            pa_scheduler::TensorRefKind::GmTensor;
    } else {
        dynamic_args.tensors[0].pointer.local_tensor = &local_tensor;
        dynamic_args.tensors[0].kind =
            pa_scheduler::TensorRefKind::LocalTensor;
    }
    pa_scheduler::SubmitContext delta_context{};
    delta_context.task_id = static_cast<int32_t>(task_id);
    delta_context.won = true;
    delta_context.register_mask = 1U;
    delta_context.result.task_id = task_id;
    delta_context.result.count = 0U;
    pa_scheduler::SharedTaskWriterDelta delta{};
    const bool built = pa_scheduler::PrepareSharedTaskWriterDelta(
        dynamic_args, delta_context, delta
    );
    asc_stcg(
        leader_reports + warp,
        built ? (kProbeSuccess ^
                     static_cast<uint64_t>(delta.ordinary_count)) :
                static_cast<uint64_t>(task_id)
    );
#elif defined(PA_FULL_BUILD_PROBE_EXEC_DYNAMIC_ADDRESS_SPACE_ONLY)
    // Scalar adapter 允许同一 tensor slot 在运行期选择 owner-local
    // TensorDesc 或 GM TensorDesc。这个分支刻意保持该合法动态选择，单独
    // 检查异地址空间 union/branch 的 VF machine-code lowering。
    const uint64_t seed = asc_ldcg(
        leader_reports + kBuilderLeaders + warp
    );
    const bool gm_source = (seed & 1U) != 0U;
    pa_scheduler::TensorDesc local_tensor{};
    pa_scheduler::cross_core::ExecPayloadSpec spec{
        task_id,
        0U,
        static_cast<uint64_t>(task_id) + 1U,
        1U,
        1U,
        0U,
        0U,
        pa_scheduler::cross_core::ExecEngineClass::Aiv,
        0U,
        0U,
        0U,
        1U,
        gm_source ? 1U : 0U,
    };
    pa_scheduler::cross_core::PaExecPayloadSource source{};
    if (gm_source) {
        source.tensors[0].gm_tensor =
            &state->shared_map.shared_outputs[task_id]
                 .tensors[0];
        source.gm_tensor_mask = 1U;
        source.reference_mask = 1U;
    } else {
        local_tensor.buffer_addr = seed;
        source.tensors[0].local_tensor = &local_tensor;
        source.gm_tensor_mask = 0U;
        source.reference_mask = 0U;
    }
    pa_scheduler::SharedExecTraceObserver<SimtOps> observer{&stats};
    const bool built =
        pa_scheduler::cross_core::BuildAndPublishExecPayload<
            SimtOps
        >(
            state->exec_cells[task_id], warp, spec, source,
            state->exec_fatal, observer
        ) == pa_scheduler::cross_core::ExecBuildResult::Published;
    asc_stcg(
        leader_reports + warp,
        built ? (kProbeSuccess ^ static_cast<uint64_t>(task_id)) :
                static_cast<uint64_t>(task_id)
    );
#elif defined(PA_FULL_BUILD_PROBE_EXEC_REFERENCE_ONLY)
    // 单独实例化 SharedExecCell 的 GM TensorDesc reference publication。
    // PaExecPayloadSource 的 union 同时保存 local/GM 两种地址空间；这里
    // 固定走真实 GM reference，检验 pointer->wire-address lowering 是否
    // 就是完整 Build 后端崩溃的来源。
    pa_scheduler::cross_core::ExecPayloadSpec spec{
        task_id,
        0U,
        static_cast<uint64_t>(task_id) + 1U,
        1U,
        1U,
        0U,
        0U,
        pa_scheduler::cross_core::ExecEngineClass::Aiv,
        0U,
        0U,
        0U,
        1U,
        1U,
    };
    pa_scheduler::cross_core::PaExecPayloadSource source{};
    source.tensors[0].gm_tensor =
        &state->shared_map.shared_outputs[task_id]
             .tensors[0];
    source.gm_tensor_mask = 1U;
    source.reference_mask = 1U;
    pa_scheduler::SharedExecTraceObserver<SimtOps> observer{&stats};
    const bool built =
        pa_scheduler::cross_core::BuildAndPublishExecPayload<
            SimtOps
        >(
            state->exec_cells[task_id], warp, spec, source,
            state->exec_fatal, observer
        ) == pa_scheduler::cross_core::ExecBuildResult::Published;
    asc_stcg(
        leader_reports + warp,
        built ? (kProbeSuccess ^ static_cast<uint64_t>(task_id)) :
                static_cast<uint64_t>(task_id)
    );
#elif defined(PA_FULL_BUILD_PROBE_WRITER_METADATA_ONLY)
    // 用运行期种子构造一条真正的 ordinary writer delta，强制生成
    // preflight、slot payload、DCCI、seq/tail publication 全链。count=1
    // 不能被优化成 symbol-only/empty fast path。
    const uint64_t seed = asc_ldcg(
        leader_reports + kBuilderLeaders + warp
    );
    context.task_id = static_cast<int32_t>(task_id);
    context.won = true;
    pa_scheduler::SharedTaskWriterDelta delta{};
    delta.prepared_task_id = static_cast<int32_t>(task_id);
    delta.ordinary_count = 1U;
    delta.symbol_count = 0U;
    delta.writer_intent_required = true;
    delta.ordinary_entries[0].buffer_addr =
        (seed & ~uint64_t{4095}) | uint64_t{4096};
    delta.ordinary_entries[0].lo = seed & uint64_t{63};
    delta.ordinary_entries[0].hi =
        delta.ordinary_entries[0].lo + 64U;
    delta.ordinary_entries[0].producer =
        static_cast<int32_t>(task_id);
    delta.ordinary_entries[0].reserved = 0U;
    delta.ordinary_buckets[0] = static_cast<uint16_t>(
        pa_scheduler::TensorMapHash(
            delta.ordinary_entries[0].buffer_addr
        )
    );
    delta.ordinary_bucket_ordinals[0] = 0U;
    const bool built =
        pa_scheduler::PublishSharedTaskWriterMetadata<SimtOps>(
            state, context, delta, stats
        );
    asc_stcg(
        leader_reports + warp,
        built ? (kProbeSuccess ^ static_cast<uint64_t>(task_id)) :
                static_cast<uint64_t>(task_id)
    );
#elif !defined(PA_FULL_BUILD_PROBE_SHELL_ONLY)
    const bool built =
        pa_scheduler::BuildRuntimePlanTask<SimtOps, false>(
            state, workers[warp], view, task_id,
            context, stats, pmu_context
        );
    asc_stcg(
        leader_reports + warp,
        built ? (kProbeSuccess ^ static_cast<uint64_t>(task_id)) :
                static_cast<uint64_t>(task_id)
    );
#else
    // 只在完整实例化被 CCEC backend 拒绝时构建这个工具链壳。它保留
    // static VF、mixed async/wait、SIMT return-ready atomic、DCCI/fence 和
    // GM report，用来把“工具链/MIX_VF 已闭合”与“完整 Build codegen 被
    // address-space lowering 阻塞”分开取证；不能冒充正式 Build 成功。
    const int64_t control_word = SimtOps::LoadControl(
        &control->closed_task_count.value
    );
    if (task_id < capacity) {
        SimtOps::InvalidateRegion(
            &cells[task_id].payload,
            pa_scheduler::aicpu_plan::kPlanCacheLineBytes
        );
    }
    asc_stcg(
        leader_reports + warp,
        static_cast<uint64_t>(control_word) ^
            static_cast<uint64_t>(task_id)
    );
#endif
    asc_threadfence();
}

// 保证 metadata 明确编码 SIMD_SIMT_MIX_VF=4。
static __simd_vf__ __aicore__ void SimdMetadataAnchor(
    __ubuf__ uint32_t *scratch
) {
    scratch[0] = scratch[0] + 1U;
}

}  // namespace

extern "C" __global__ __aicore__ void
aicpu_plan_simt_full_build_probe_0_mix_aiv(
    __gm__ pa_scheduler::SchedulerState *state,
    __gm__ pa_scheduler::WorkerState *workers,
    __gm__ pa_scheduler::aicpu_plan::RuntimePlanControl *control,
    __gm__ pa_scheduler::aicpu_plan::RuntimeTaskPlanCell *cells,
    uint32_t capacity, uint32_t first_task_id,
    __gm__ uint64_t *leader_reports
) {
    if (capacity == UINT32_MAX) {
        SimdMetadataAnchor(
            reinterpret_cast<__ubuf__ uint32_t *>(0)
        );
    }
    cce::async_invoke<BuildCanonicalPlanTaskWithScalarImplementation>(
        cce::dim3{kBuilderThreads, 1U, 1U},
        state, workers, control, cells, capacity,
        first_task_id, leader_reports
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
}
