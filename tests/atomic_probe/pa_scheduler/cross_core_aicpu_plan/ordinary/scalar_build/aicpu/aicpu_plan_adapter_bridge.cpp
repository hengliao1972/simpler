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

#include "aicpu_plan_adapter_bridge.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../../../common/runtime_plan_pipeline_policy.h"
#include "../common/aicpu_plan_pa_adapter.h"

namespace {

using namespace pa_scheduler;
using namespace pa_scheduler::aicpu_plan;
using namespace pa_scheduler::aicpu_plan_adapter;
using pa_scheduler::aicpu_plan_trace::Operation;
using pa_scheduler::aicpu_plan_trace::Scope;
using pa_scheduler::aicpu_plan_trace::State;
using pa_scheduler::aicpu_plan_trace::Target;

#ifndef PA_BUILD_SWIMLANE
#define PA_BUILD_SWIMLANE 0
#endif

#if PA_BUILD_SWIMLANE
State *g_operation_trace = nullptr;
RuntimePlanControl *g_trace_control = nullptr;
RuntimeTaskPlanCell *g_trace_cells = nullptr;
uint32_t g_trace_capacity = 0U;
Scope g_trace_scope = Scope::BackendBind;
uint32_t g_trace_task_id = pa_scheduler::aicpu_plan_trace::kNoTaskId;

struct TraceTarget {
    Target target;
    uint32_t index;
};

TraceTarget ClassifyTraceAddress(const void *address) {
    if (address == nullptr) {
        return {Target::None, pa_scheduler::aicpu_plan_trace::kNoTargetIndex};
    }
    if (g_trace_control != nullptr) {
        if (address == &g_trace_control->fatal.value) {
            return {Target::Fatal, pa_scheduler::aicpu_plan_trace::kNoTargetIndex};
        }
        if (address == &g_trace_control->closed_task_count.value) {
            return {Target::ClosedTaskCount, pa_scheduler::aicpu_plan_trace::kNoTargetIndex};
        }
        if (address == &g_trace_control->planned_frontier.value) {
            return {Target::PlannedFrontier, pa_scheduler::aicpu_plan_trace::kNoTargetIndex};
        }
        if (address == &g_trace_control->build_next.value) {
            return {Target::BuildNext, pa_scheduler::aicpu_plan_trace::kNoTargetIndex};
        }
        if (address == &g_trace_control->build_workers_done.value) {
            return {Target::BuildWorkersDone, pa_scheduler::aicpu_plan_trace::kNoTargetIndex};
        }
        if (address == &g_trace_control->build_release.value) {
            return {Target::BuildRelease, pa_scheduler::aicpu_plan_trace::kNoTargetIndex};
        }
        const uintptr_t raw = reinterpret_cast<uintptr_t>(address);
        const uintptr_t control_begin = reinterpret_cast<uintptr_t>(g_trace_control);
        if (raw >= control_begin && raw < control_begin + sizeof(*g_trace_control)) {
            return {Target::RuntimePlanControl, pa_scheduler::aicpu_plan_trace::kNoTargetIndex};
        }
    }
    if (g_trace_cells != nullptr && g_trace_capacity != 0U) {
        const uintptr_t raw = reinterpret_cast<uintptr_t>(address);
        const uintptr_t cells_begin = reinterpret_cast<uintptr_t>(g_trace_cells);
        const uint64_t cells_bytes = static_cast<uint64_t>(g_trace_capacity) * sizeof(RuntimeTaskPlanCell);
        if (raw >= cells_begin && raw < cells_begin + cells_bytes) {
            const uintptr_t relative = raw - cells_begin;
            const uint32_t index = static_cast<uint32_t>(relative / sizeof(RuntimeTaskPlanCell));
            if (address == &g_trace_cells[index].control.value) {
                return {Target::CellControl, index};
            }
            const uintptr_t within = relative % sizeof(RuntimeTaskPlanCell);
            if (within >= offsetof(RuntimeTaskPlanCell, payload)) {
                return {Target::CellPayload, index};
            }
        }
    }
    return {Target::None, pa_scheduler::aicpu_plan_trace::kNoTargetIndex};
}

bool RecordOperation(
    uint64_t begin_ns, uint64_t end_ns, Operation operation, Target target, uint32_t calls, uint32_t lines,
    uint32_t first_target_index, uint32_t last_target_index, int64_t first_value, int64_t last_value,
    bool merge_consecutive
) {
    return pa_scheduler::aicpu_plan_trace::Append(
        g_operation_trace, begin_ns, end_ns, g_trace_task_id, g_trace_scope, operation, target, calls, lines,
        first_target_index, last_target_index, first_value, last_value, merge_consecutive
    );
}

class TraceScopeGuard {
public:
    TraceScopeGuard(Scope scope, uint32_t task_id) :
        previous_scope_(g_trace_scope),
        previous_task_id_(g_trace_task_id) {
        g_trace_scope = scope;
        g_trace_task_id = task_id;
    }

    ~TraceScopeGuard() {
        g_trace_scope = previous_scope_;
        g_trace_task_id = previous_task_id_;
    }

private:
    Scope previous_scope_;
    uint32_t previous_task_id_;
};
#endif

// 当前 A5 main aicpu_scheduler Path-A 已通过独立同构门禁验证：AICPU
// ordinary payload store 可由 release atomic control 直接发布给 AICore，
// 无需 producer 侧 dc cvac/显式 barrier。Host/SDMA -> AICPU 是另一条非一致
// 路径，复用初始化和 StreamingFuture 的保守协议不能据此顺带删除。
struct AicpuProducerOps {
    // Host 通过 DMA/aclrtMemset 重置同一块 GM 后，AICPU 不会自动丢弃
    // 上一轮留下的 clean-valid control line。AICPU EL0 的 dc ivac 在当前
    // 环境没有可依赖的契约；仓库 Host-DMA -> AICPU 的已验证协议使用
    // dc civac + dsb sy + isb。这里允许 civac 的关键前提是：control 的
    // 唯一写路径 PublishControl 已经执行 cvac + dsb，之后没有 ordinary
    // store，因此进入下一轮时该 line 必须是 clean-valid，而不是 dirty。
    // 当前 AICore 还没有启动，也不存在并发 writer。
    static void DiscardPreviouslyCleanControlLine(
        const volatile int64_t *address
    )
    {
#if PA_BUILD_SWIMLANE
        const uint64_t trace_begin = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
#endif
#if defined(__aarch64__)
        const uintptr_t line =
            reinterpret_cast<uintptr_t>(address) & ~uintptr_t{63U};
        __asm__ volatile("dc civac, %0" : : "r"(line) : "memory");
#else
        (void)address;
#endif
#if PA_BUILD_SWIMLANE
        const uint64_t trace_end = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
        const TraceTarget target = ClassifyTraceAddress(const_cast<const int64_t *>(address));
        (void)RecordOperation(
            trace_begin, trace_end, Operation::CacheDiscardCivac, target.target, 1U, 1U, target.index, target.index, 0,
            0, true
        );
#endif
    }

    static void FinishCacheMaintenance()
    {
        StoreBarrier();
        InstructionBarrier();
    }

    static void InstructionBarrier() {
#if PA_BUILD_SWIMLANE
        const uint64_t trace_begin = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
#endif
#if defined(__aarch64__)
        __asm__ volatile("isb" ::: "memory");
#endif
#if PA_BUILD_SWIMLANE
        const uint64_t trace_end = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
        (void)RecordOperation(
            trace_begin, trace_end, Operation::BarrierIsb, Target::None, 1U, 0U,
            pa_scheduler::aicpu_plan_trace::kNoTargetIndex, pa_scheduler::aicpu_plan_trace::kNoTargetIndex, 0, 0, false
        );
#endif
    }

    static int64_t LoadControl(const volatile int64_t *address)
    {
#if PA_BUILD_SWIMLANE
        const uint64_t trace_begin = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
#endif
        const int64_t value = __atomic_load_n(address, __ATOMIC_ACQUIRE);
#if PA_BUILD_SWIMLANE
        const uint64_t trace_end = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
        const TraceTarget target = ClassifyTraceAddress(const_cast<const int64_t *>(address));
        (void)RecordOperation(
            trace_begin, trace_end, Operation::AtomicLoadAcquire, target.target, 1U, 0U, target.index, target.index,
            value, value, target.target == Target::CellControl
        );
#endif
        return value;
    }

    static void StorePayloadWord(volatile uint64_t *address, uint64_t value)
    {
#if PA_BUILD_SWIMLANE
        const bool observe = g_trace_scope == Scope::TaskPublish;
        const uint64_t trace_begin = observe ? pa_scheduler::aicpu_plan_trace::TimestampNanoseconds() : 0U;
#endif
        *address = value;
#if PA_BUILD_SWIMLANE
        if (observe) {
            const uint64_t trace_end = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
            const TraceTarget target = ClassifyTraceAddress(const_cast<const uint64_t *>(address));
            int64_t signed_value = 0;
            std::memcpy(&signed_value, &value, sizeof(signed_value));
            (void)RecordOperation(
                trace_begin, trace_end, Operation::GmStore, target.target, 1U, 0U, target.index, target.index,
                signed_value, signed_value, false
            );
        }
#endif
    }

    static void FlushRegion(const void *address, uint64_t bytes)
    {
#if PA_BUILD_SWIMLANE
        const uint64_t trace_begin = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
#endif
#if defined(__aarch64__) || PA_BUILD_SWIMLANE
        const uintptr_t begin =
            reinterpret_cast<uintptr_t>(address) & ~uintptr_t{63U};
        const uintptr_t end =
            (reinterpret_cast<uintptr_t>(address) + bytes + 63U) &
            ~uintptr_t{63U};
#else
        (void)address;
        (void)bytes;
#endif
#if defined(__aarch64__)
        for (uintptr_t line = begin; line < end; line += 64U) {
            __asm__ volatile("dc cvac, %0" : : "r"(line) : "memory");
        }
#else
        __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
#if PA_BUILD_SWIMLANE
        const uint64_t trace_end = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
        const TraceTarget target = ClassifyTraceAddress(address);
        const uint32_t lines = static_cast<uint32_t>((end - begin) / 64U);
        (void)RecordOperation(
            trace_begin, trace_end, Operation::CacheCleanCvac, target.target, lines, lines, target.index, target.index,
            0, 0, false
        );
#endif
    }

    static void StoreBarrier()
    {
#if PA_BUILD_SWIMLANE
        const uint64_t trace_begin = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
#endif
#if defined(__aarch64__)
        __asm__ volatile("dsb sy" ::: "memory");
#else
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
#if PA_BUILD_SWIMLANE
        const uint64_t trace_end = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
        (void)RecordOperation(
            trace_begin, trace_end, Operation::BarrierDsbSy, Target::None, 1U, 0U,
            pa_scheduler::aicpu_plan_trace::kNoTargetIndex, pa_scheduler::aicpu_plan_trace::kNoTargetIndex, 0, 0, false
        );
#endif
    }

    static void StoreAndCleanControl(
        volatile int64_t *address, int64_t value
    )
    {
#if PA_BUILD_SWIMLANE
        const uint64_t trace_begin = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
#endif
        *address = value;
        __asm__ volatile("" ::: "memory");
#if PA_BUILD_SWIMLANE
        const uint64_t trace_end = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
        const TraceTarget target = ClassifyTraceAddress(const_cast<const int64_t *>(address));
        (void)RecordOperation(
            trace_begin, trace_end, Operation::GmStore, target.target, 1U, 0U, target.index, target.index, value, value,
            false
        );
#endif
        FlushRegion(const_cast<const int64_t *>(address), sizeof(int64_t));
    }

    static void StoreReleaseControl(
        volatile int64_t *address, int64_t value
    )
    {
#if PA_BUILD_SWIMLANE
        const uint64_t trace_begin = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
#endif
        __atomic_store_n(address, value, __ATOMIC_RELEASE);
#if PA_BUILD_SWIMLANE
        const uint64_t trace_end = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
        const TraceTarget target = ClassifyTraceAddress(const_cast<const int64_t *>(address));
        (void)RecordOperation(
            trace_begin, trace_end, Operation::AtomicStoreRelease, target.target, 1U, 0U, target.index, target.index,
            value, value, true
        );
#endif
    }

    static void PublishControl(volatile int64_t *address, int64_t value)
    {
        StoreAndCleanControl(address, value);
        StoreBarrier();
        InstructionBarrier();
    }
};

// UP 的 has-following/last-in-batch 只有下一个真实 Begin（或
// close）到来后才能确定。Finish callback 里的 L0TaskArgs 及其指向的
// descriptor 都属于 orchestration 栈，不能把指针留到下一个 Begin。
// 所以 Finish 趁 callback 仍存活时把 canonical payload 唯一一次
// Pack 到目标 GM cell，但保持 cell control=Empty。下一个 Begin/close 只补上
// 最终 flags，再以 release atomic 发布 cell control；完整 GM wire 回读校验
// 仅属于显式 debug 构建，不进入正常发布热路。
// PlanAheadClosed 的 AICore consumer 永远不会读 Empty cell 的 payload，
// backend 也不再保留任何 callback 指针；StreamingFuture 仍保留原保守 clean。
constexpr uint64_t kStagedPlanMetadataMagic = 0x5041535441470001ULL;

struct StagedPlanMetadata {
    uint64_t magic;
    RuntimeTaskPlanHeader provisional_header;
    uint32_t payload_lines;
    uint16_t output_count;
    uint16_t reserved;
};

static_assert(
    sizeof(StagedPlanMetadata) <= kAtomicIsolationBytes,
    "staged PA metadata unexpectedly exceeds one isolated line"
);

#if PA_RUNTIME_PLAN_DEBUG_FULL_VALIDATION
bool HeaderMatchesExpected(
    const RuntimeTaskPlanHeader &header,
    const RuntimeTaskPlanHeader &expected
)
{
    if (header.task_id != expected.task_id ||
        header.function_id != expected.function_id ||
        header.tensor_count != expected.tensor_count ||
        header.scalar_count != expected.scalar_count ||
        header.explicit_dep_count != expected.explicit_dep_count ||
        header.output_count != expected.output_count ||
        header.engine_class != expected.engine_class ||
        header.adapter_flags != expected.adapter_flags ||
        header.core_num != expected.core_num ||
        header.require_sync_start != expected.require_sync_start ||
        header.reserved0 != expected.reserved0 ||
        header.adapter_data != expected.adapter_data ||
        header.tensor_reference_mask != expected.tensor_reference_mask ||
        header.abi_version != expected.abi_version) {
        return false;
    }
    for (uint32_t tensor = 0U;
         tensor < aicpu_plan::kMaxTaskTensors; ++tensor) {
        if (header.tensor_tags[tensor] != expected.tensor_tags[tensor]) {
            return false;
        }
    }
    return true;
}
#endif

PlanPublishResult FinalizeStagedPlan(
    const RuntimePlanView &view, const StagedPlanMetadata &metadata,
    uint8_t final_adapter_flags
)
{
    const uint32_t task_id = metadata.provisional_header.task_id;
    if (view.control == nullptr || view.cells == nullptr ||
        task_id >= view.capacity) {
        return PlanPublishResult::CellUnavailable;
    }
    // StreamingFuture 有并发 consumer，发布时必须重新观察共享状态。
    // PlanAheadClosed 的唯一 producer 由 callback 单调 task-id 和 staged
    // magic 锁定，consumer 又要等 Close 后才启动；正常路径不做逐 task
    // 重复检查，Close 的 AdvancePlannedFrontierTo 会一次性验证 [0, N)
    // 的完整连续前缀。
    if constexpr (
        kRuntimePlanPipelineIsStreamingFuture ||
        PA_RUNTIME_PLAN_DEBUG_FULL_VALIDATION
    ) {
        if (AicpuProducerOps::LoadControl(
                &view.control->fatal.value
            ) != 0) {
            return PlanPublishResult::CellUnavailable;
        }
        // StreamingFuture 会和 consumer 并发复用 cell，必须要求当前
        // control 为 Empty。PlanAheadClosed 在 Close 前没有 consumer，
        // 可以直接用本轮 Published 覆盖上一轮同地址 Published；旧值不再
        // 是该 policy 的协议前置条件。
        if constexpr (kRuntimePlanPipelineIsStreamingFuture) {
            if (AicpuProducerOps::LoadControl(
                    &view.cells[task_id].control.value
                ) != 0) {
                return PlanPublishResult::CellUnavailable;
            }
        }
        const int64_t closed = AicpuProducerOps::LoadControl(
            &view.control->closed_task_count.value
        );
        if (closed != kPlanProducerNotReadyTaskCount &&
            closed != kPlanOpenTaskCount) {
            return PlanPublishResult::CellUnavailable;
        }
        const int64_t frontier = AicpuProducerOps::LoadControl(
            &view.control->planned_frontier.value
        );
        if (frontier < 0 || frontier > static_cast<int64_t>(task_id)) {
            return PlanPublishResult::CellUnavailable;
        }
        if (static_cast<int64_t>(task_id) > frontier) {
            const uint32_t predecessor = task_id - 1U;
            const DecodedPlanCellControl decoded = DecodePlanCellControl(
                AicpuProducerOps::LoadControl(
                    &view.cells[predecessor].control.value
                )
            );
            if (!decoded.valid ||
                decoded.phase != PlanCellPhase::Published ||
                decoded.task_id != predecessor) {
                return PlanPublishResult::CellUnavailable;
            }
        }
    }

    RuntimeTaskPlanCell &cell = view.cells[task_id];
    RuntimeTaskPlanHeader expected = metadata.provisional_header;
    expected.adapter_flags = final_adapter_flags;
    AicpuProducerOps::StorePayloadWord(
        &cell.payload.words[2],
        RuntimeTaskPlanHeaderWord(expected, 2U)
    );

    // 正常路径不重读刚刚由同一 producer canonical Pack 的整份 GM
    // payload。专门的协议调试构建可在 Published control 前复核最终
    // wire；无论该门禁是否开启，AICore consumer acquire 后都保留
    // 跨处理器边界上的权威校验。
#if PA_RUNTIME_PLAN_DEBUG_FULL_VALIDATION
    RuntimeTaskPlanHeader header{};
    RuntimeTaskPlanLayout validated_layout{};
#if PA_BUILD_SWIMLANE
    const uint64_t validate_begin = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
#endif
    const bool payload_valid =
        ValidateRuntimeTaskPlanPayload(cell.payload, task_id, metadata.payload_lines, header, validated_layout) &&
        validated_layout.payload_lines == metadata.payload_lines && HeaderMatchesExpected(header, expected) &&
        ValidatePaAdapterMetadata(
            header.task_id, static_cast<EngineClass>(header.engine_class), header.adapter_flags, header.adapter_data
        );
#if PA_BUILD_SWIMLANE
    const uint64_t validate_end = pa_scheduler::aicpu_plan_trace::TimestampNanoseconds();
    (void)RecordOperation(
        validate_begin, validate_end, Operation::ScalarWork, Target::PayloadValidation, 1U, 0U,
        pa_scheduler::aicpu_plan_trace::kNoTargetIndex, pa_scheduler::aicpu_plan_trace::kNoTargetIndex,
        payload_valid ? 1 : 0, payload_valid ? 1 : 0, false
    );
#endif
    if (!payload_valid) {
        return PlanPublishResult::InvalidInput;
    }
#endif

    const int64_t published = static_cast<int64_t>(EncodePlanCellControl(
        PlanCellPhase::Published, metadata.payload_lines, task_id
    ));
    if constexpr (kRuntimePlanPipelineIsStreamingFuture) {
        AicpuProducerOps::FlushRegion(
            &cell.payload,
            metadata.payload_lines * kPlanCacheLineBytes
        );
        AicpuProducerOps::StoreBarrier();
        AicpuProducerOps::PublishControl(&cell.control.value, published);
    } else {
        // 独立 A5 Path-A 门禁验证了同一地址连续复用 20480 轮：ordinary
        // payload + release atomic control 对 AICore 可见，无需 AICPU cvac。
        // AICore 仍以返回型 atomic 观察 control，并在读取 payload 前 DCCI。
        AicpuProducerOps::StoreReleaseControl(
            &cell.control.value, published
        );
    }
    return PlanPublishResult::Published;
}

bool MakeView(
    void *control, void *cells, uint32_t capacity,
    RuntimePlanView &view
)
{
    if (control == nullptr || cells == nullptr || capacity == 0U ||
        capacity > kMaxRuntimeTasks) {
        return false;
    }
    const uint64_t bytes =
        static_cast<uint64_t>(capacity) * sizeof(RuntimeTaskPlanCell);
    RuntimePlanStorageRef storage{};
    storage.cells_base = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(cells)
    );
    storage.cells_bytes = bytes;
    storage.capacity = capacity;
    storage.cell_bytes = sizeof(RuntimeTaskPlanCell);
    storage.abi_version = kRuntimePlanAbiVersion;
    return MakeRuntimePlanView(
        static_cast<RuntimePlanControl *>(control), storage, view
    );
}

}  // namespace

extern "C" int32_t
aicpu_plan_adapter_initialize(void *control, void *cells, uint32_t capacity, State *operation_trace_state) {
    RuntimePlanView view{};
    if (!MakeView(control, cells, capacity, view)) return -1;
#if PA_BUILD_SWIMLANE
    if (operation_trace_state == nullptr || operation_trace_state->records == nullptr ||
        operation_trace_state->capacity == 0U || operation_trace_state->count > operation_trace_state->capacity ||
        operation_trace_state->dropped != 0U || operation_trace_state->reserved != 0U) {
        return -1;
    }
    g_operation_trace = operation_trace_state;
    g_trace_control = view.control;
    g_trace_cells = view.cells;
    g_trace_capacity = view.capacity;
    TraceScopeGuard trace_scope(Scope::BackendBind, pa_scheduler::aicpu_plan_trace::kNoTaskId);
#else
    if (operation_trace_state != nullptr) return -1;
#endif
    // Host 在启动 producer 前已把 closed 置为 NotReady。StreamingFuture
    // 下 AICore 可能已经在另一条 stream 轮询；PlanAheadClosed 下虽然尚未
    // 启动 AICore，也沿用同一安全复用协议。不能对 closed line 做 civac
    // （它可能把旧 N 写回 GM）；用 ordinary store 覆盖本地 stale value
    // 并精确 clean，保证 initialize 失败能从 -2 发布为 -3 ReadyFailed。
    AicpuProducerOps::PublishControl(
        &view.control->closed_task_count.value,
        kPlanProducerNotReadyTaskCount
    );
    // Host aclrtMemset 不会 snoop 上一轮留在 AICPU cache 中的 control。
    // StreamingFuture 的 consumer 会提前领 future ticket，且旧发布协议
    // 保证 control 为 clean-valid，因此在 Ready 前仍完成全容量
    // discard/Empty 校验。PlanAheadClosed 在 Close 前没有 consumer，实际
    // task 会由本轮 release Published 直接覆盖同地址旧值；不再为所有 cell
    // 额外执行 release Empty + acquire readback。
    if constexpr (kRuntimePlanPipelineIsStreamingFuture) {
        for (uint32_t task = 0U; task < capacity; ++task) {
            AicpuProducerOps::DiscardPreviouslyCleanControlLine(
                &view.cells[task].control.value
            );
        }
        AicpuProducerOps::FinishCacheMaintenance();
        for (uint32_t task = 0U; task < capacity; ++task) {
            if (AicpuProducerOps::LoadControl(
                    &view.cells[task].control.value
                ) != 0) {
                (void)PublishRuntimePlanReadyFailed<AicpuProducerOps>(view);
                return -2;
            }
        }
    }
    view.control->planned_frontier.value = 0;
    // StreamingFuture consumer 在 cell control 的 civac+Empty 校验完成前
    // 只允许轮询 closed line；PlanAheadClosed consumer 尚未启动。两条
    // policy 都先保持 NotReady、完成其余 control 的 reset+clean，再在
    // NotReady 下生产 cell。
    view.control->closed_task_count.value =
        kPlanProducerNotReadyTaskCount;
    view.control->build_next.value = 0;
    view.control->build_workers_done.value = 0;
    view.control->build_release.value = kBuildReleasePending;
    view.control->fatal.value = 0;
    AicpuProducerOps::FlushRegion(view.control, sizeof(*view.control));
    AicpuProducerOps::StoreBarrier();
    AicpuProducerOps::InstructionBarrier();
    // 初始化成功后仍保持 NotReady。两种长期保留的 pipeline policy
    // 都允许 producer 在该状态下顺序发布 cell；PlanAheadClosed 直到
    // Close 才开放，StreamingFuture 则可在连续前缀达到门槛后开放。
    if (!RuntimePlanCanPublishReady<AicpuProducerOps>(view)) {
        (void)PublishRuntimePlanReadyFailed<AicpuProducerOps>(view);
        return -3;
    }
    return 0;
}

extern "C" int32_t aicpu_plan_adapter_stage(
    void *control, void *cells, uint32_t capacity,
    const void *l0_task_args, uint32_t task_id, int32_t function_id,
    uint8_t engine_class, uint8_t provisional_adapter_flags,
    uint32_t batch_start,
    void *staged_metadata, uint32_t *payload_lines,
    uint16_t *output_count
)
{
#if PA_BUILD_SWIMLANE
    TraceScopeGuard trace_scope(Scope::TaskStage, task_id);
#endif
    if (staged_metadata == nullptr) return -1;
    auto &metadata =
        *static_cast<StagedPlanMetadata *>(staged_metadata);
    // 先撤销旧 magic，包括其他入参在早期校验就失败的
    // 路径；失败返回后任何旧 pending metadata 都不得被复用。
    metadata.magic = 0U;
    if (l0_task_args == nullptr || payload_lines == nullptr ||
        output_count == nullptr ||
        engine_class > static_cast<uint8_t>(EngineClass::Aiv)) {
        return -1;
    }

    RuntimePlanView view{};
    if (!MakeView(control, cells, capacity, view) ||
        task_id >= view.capacity) {
        return -2;
    }
    // PlanAheadClosed 只有一个串行 producer，且 Close 前无 consumer。
    // 正常路径由 callback 单调 task-id 和 Close 的完整前缀检查共同锁定；
    // 逐 task 状态复核仅留给 debug。
    if constexpr (
        kRuntimePlanPipelineIsStreamingFuture ||
        PA_RUNTIME_PLAN_DEBUG_FULL_VALIDATION
    ) {
        if (AicpuProducerOps::LoadControl(
                &view.control->fatal.value
            ) != 0) {
            return -2;
        }
        if constexpr (kRuntimePlanPipelineIsStreamingFuture) {
            if (AicpuProducerOps::LoadControl(
                    &view.cells[task_id].control.value
                ) != 0) {
                return -2;
            }
        }
        const int64_t closed = AicpuProducerOps::LoadControl(
            &view.control->closed_task_count.value
        );
        if (closed != kPlanProducerNotReadyTaskCount &&
            closed != kPlanOpenTaskCount) {
            return -2;
        }
        const int64_t frontier = AicpuProducerOps::LoadControl(
            &view.control->planned_frontier.value
        );
        if (frontier < 0 || frontier > static_cast<int64_t>(task_id)) {
            return -2;
        }
        if (static_cast<int64_t>(task_id) > frontier) {
            const uint32_t predecessor = task_id - 1U;
            const DecodedPlanCellControl decoded = DecodePlanCellControl(
                AicpuProducerOps::LoadControl(
                    &view.cells[predecessor].control.value
                )
            );
            if (!decoded.valid ||
                decoded.phase != PlanCellPhase::Published ||
                decoded.task_id != predecessor) {
                return -2;
            }
        }
    }

    // 真实 L0TaskArgs 与 standalone TaskArgs 均有独立的 layout static_assert。
    // 这里用 byte copy，而不是把一个 C++ 类型直接别名成另一个类型。
    TaskArgs args;
    static_assert(sizeof(args) == 1280U, "PA adapter expects shared L0TaskArgs ABI");
    std::memcpy(&args, l0_task_args, sizeof(args));

    RuntimeTaskPlanSpec spec{};
    if (!MakePaRuntimeTaskPlanSpec(
            args, task_id, function_id,
            static_cast<EngineClass>(engine_class),
            provisional_adapter_flags, batch_start, spec
        )) {
        return -3;
    }

    RuntimeTaskPlanLayout layout{};
    const PaRuntimeTaskPlanSource source{args};
    if (!PackRuntimeTaskPlan<AicpuProducerOps>(
            view.cells[task_id], spec, source, layout,
            &metadata.provisional_header
        )) {
        // Pack 中途失败只会留下不可见 payload；control 仍为
        // Empty，FinishTask 会立即发布 fatal，consumer 不会 acquire。
        return -4;
    }

    metadata.payload_lines = layout.payload_lines;
    metadata.output_count = spec.output_count;
    metadata.reserved = 0U;
    // magic 最后写，防止中途失败的未发布 GM payload 被下一个
    // Begin 误当成完整 pending task。
    metadata.magic = kStagedPlanMetadataMagic;
    *payload_lines = layout.payload_lines;
    *output_count = spec.output_count;
    return 0;
}

extern "C" int32_t aicpu_plan_adapter_publish_staged(
    void *control, void *cells, uint32_t capacity,
    const void *staged_metadata, uint32_t payload_lines,
    uint8_t final_adapter_flags
)
{
    if (staged_metadata == nullptr || payload_lines == 0U ||
        payload_lines > kMaxPlanPayloadLines) {
        return -1;
    }
    RuntimePlanView view{};
    if (!MakeView(control, cells, capacity, view)) return -2;

    const auto &metadata =
        *static_cast<const StagedPlanMetadata *>(staged_metadata);
    const RuntimeTaskPlanHeader &provisional =
        metadata.provisional_header;
    if (metadata.magic != kStagedPlanMetadataMagic ||
        metadata.payload_lines != payload_lines ||
        provisional.task_id >= capacity) {
        return -3;
    }
#if PA_RUNTIME_PLAN_DEBUG_FULL_VALIDATION
    if (metadata.reserved != 0U ||
        metadata.output_count != provisional.output_count ||
        provisional.tensor_count > aicpu_plan::kMaxTaskTensors ||
        provisional.scalar_count > aicpu_plan::kMaxTaskScalars ||
        provisional.explicit_dep_count >
            aicpu_plan::kMaxExplicitDependencies ||
        provisional.abi_version != kRuntimePlanAbiVersion) {
        return -3;
    }
#endif
#if PA_BUILD_SWIMLANE
    TraceScopeGuard trace_scope(Scope::TaskPublish, provisional.task_id);
#endif

    const EngineClass engine =
        static_cast<EngineClass>(provisional.engine_class);
    if (!ValidatePaAdapterMetadata(
            provisional.task_id, engine, final_adapter_flags,
            provisional.adapter_data
        )) {
        return -4;
    }

    if (FinalizeStagedPlan(
            view, metadata, final_adapter_flags
        ) !=
        PlanPublishResult::Published) {
        return -5;
    }
    // StreamingFuture 需要在生产期发布连续前缀，但
    // PlanAheadClosed 的 consumer 只会在 close 完成后启动。后者在
    // close 一次性验证所有 Published cell 并发布最终 frontier，
    // 避免在同一 planned_frontier cache line 上做每 batch 的重复发布。
    if constexpr (kRuntimePlanPipelineIsStreamingFuture) {
        if ((final_adapter_flags & kSharedPaTicketLastSubmit) == 0U) {
            return 0;
        }
        const uint32_t new_frontier = provisional.task_id + 1U;
#if PA_BUILD_SWIMLANE
        {
            TraceScopeGuard frontier_scope(Scope::FrontierAdvance, provisional.task_id);
#endif
            if (!AdvancePlannedFrontierTo<AicpuProducerOps>(view, provisional.adapter_data, new_frontier)) {
                return -6;
            }
#if PA_BUILD_SWIMLANE
        }
#endif
        // StreamingFuture 的 Ready 只在真实 batch 边界发布，因此
        // 阈值128的G1序列会在连续 frontier=130 时 Open。PlanAheadClosed
        // 在整个生产期都保持 -2；两条路径的短 Plan 都由 close helper
        // 完成同一 Ready -> Closed(N) 握手。
        const int64_t closed = AicpuProducerOps::LoadControl(
            &view.control->closed_task_count.value
        );
        if (new_frontier >= kRuntimePlanReadyPrefillTasks) {
            if (closed == kPlanProducerNotReadyTaskCount) {
#if PA_BUILD_SWIMLANE
                {
                    TraceScopeGuard ready_scope(Scope::ReadyPublish, provisional.task_id);
#endif
                    if (!PublishRuntimePlanReady<AicpuProducerOps>(view)) {
                        return -7;
                    }
#if PA_BUILD_SWIMLANE
                }
#endif
            } else if (closed != kPlanOpenTaskCount) {
                return -7;
            }
        } else if (closed != kPlanProducerNotReadyTaskCount &&
                   closed != kPlanOpenTaskCount) {
            return -7;
        }
    }
    return 0;
}

extern "C" int32_t aicpu_plan_adapter_close(
    void *control, void *cells, uint32_t capacity,
    uint32_t final_task_count
)
{
#if PA_BUILD_SWIMLANE
    TraceScopeGuard trace_scope(Scope::BackendClose, pa_scheduler::aicpu_plan_trace::kNoTaskId);
#endif
    RuntimePlanView view{};
    if (!MakeView(control, cells, capacity, view)) return -1;
    if constexpr (kRuntimePlanPipelineIsPlanAheadClosed) {
        // 串行 policy 在 Close 前不允许任何 Build consumer 已经启动。
        // consumer 只会读本轮逐 task release-Published 并在此处验证过的
        // [0,N)；未用后缀允许保留上一轮值，不参与本轮协议。
        if (AicpuProducerOps::LoadControl(
                &view.control->build_next.value
            ) != 0 ||
            AicpuProducerOps::LoadControl(
                &view.control->build_workers_done.value
                ) != 0) {
            return -3;
        }
        const int64_t frontier = AicpuProducerOps::LoadControl(
            &view.control->planned_frontier.value
        );
        if (frontier != 0 ||
            (final_task_count != 0U &&
             !AdvancePlannedFrontierTo<AicpuProducerOps>(
                 view, 0U, final_task_count
             ))) {
            return -3;
        }
    }
    const int64_t closed = AicpuProducerOps::LoadControl(
        &view.control->closed_task_count.value
    );
    if (closed == kPlanProducerNotReadyTaskCount) {
        // PlanAheadClosed 的任意 N，以及 StreamingFuture 的短 Plan，在
        // 这里先发布 Open，再用下一次独立 control 发布 Close(N)。串行
        // consumer 此时尚未启动；并发 consumer 可观察任一中间状态，
        // 都不会触碰 initialize 仍在维护的 cell。
        if (!PublishRuntimePlanReady<AicpuProducerOps>(view)) return -2;
    } else if (closed != kPlanOpenTaskCount) {
        return -2;
    }
    return CloseRuntimePlan<AicpuProducerOps>(view, final_task_count)
        ? 0
        : -3;
}

extern "C" void aicpu_plan_adapter_publish_fatal(
    void *control, void *cells, uint32_t capacity,
    int64_t error_code
)
{
#if PA_BUILD_SWIMLANE
    TraceScopeGuard trace_scope(Scope::FatalPublish, pa_scheduler::aicpu_plan_trace::kNoTaskId);
#endif
    if (control == nullptr || error_code == 0) return;
    auto *plan_control = static_cast<RuntimePlanControl *>(control);
    AicpuProducerOps::PublishControl(
        &plan_control->fatal.value, error_code
    );
    RuntimePlanView view{};
    if (MakeView(control, cells, capacity, view)) {
        (void)PublishRuntimePlanReadyFailed<AicpuProducerOps>(view);
    }
}
