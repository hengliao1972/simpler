/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
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

// AICPU 与 AICore 不具备 Scalar cache coherence。payload 使用 ordinary
// store + exact clean；control 也使用 ordinary store + exact clean，返回型
// atomic observe 由 AIC/AIV consumer 承担。x86 smoke 只替换为顺序一致 fence。
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
#if defined(__aarch64__)
        const uintptr_t line =
            reinterpret_cast<uintptr_t>(address) & ~uintptr_t{63U};
        __asm__ volatile("dc civac, %0" : : "r"(line) : "memory");
#else
        (void)address;
#endif
    }

    static void FinishCacheMaintenance()
    {
        StoreBarrier();
#if defined(__aarch64__)
        __asm__ volatile("isb" ::: "memory");
#endif
    }

    static int64_t LoadControl(const volatile int64_t *address)
    {
        return __atomic_load_n(address, __ATOMIC_ACQUIRE);
    }

    static void StorePayloadWord(volatile uint64_t *address, uint64_t value)
    {
        *address = value;
    }

    static void FlushRegion(const void *address, uint64_t bytes)
    {
#if defined(__aarch64__)
        const uintptr_t begin =
            reinterpret_cast<uintptr_t>(address) & ~uintptr_t{63U};
        const uintptr_t end =
            (reinterpret_cast<uintptr_t>(address) + bytes + 63U) &
            ~uintptr_t{63U};
        for (uintptr_t line = begin; line < end; line += 64U) {
            __asm__ volatile("dc cvac, %0" : : "r"(line) : "memory");
        }
#else
        (void)address;
        (void)bytes;
        __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
    }

    static void StoreBarrier()
    {
#if defined(__aarch64__)
        __asm__ volatile("dsb sy" ::: "memory");
#else
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
    }

    static void PublishControl(volatile int64_t *address, int64_t value)
    {
        *address = value;
        __asm__ volatile("" ::: "memory");
        FlushRegion(const_cast<const int64_t *>(address), sizeof(int64_t));
        StoreBarrier();
#if defined(__aarch64__)
        __asm__ volatile("isb" ::: "memory");
#endif
    }
};

// UP 的 has-following/last-in-batch 只有下一个真实 Begin（或
// close）到来后才能确定。Finish callback 里的 L0TaskArgs 及其指向的
// descriptor 都属于 orchestration 栈，不能把指针留到下一个 Begin。
// 所以 Finish 趁 callback 仍存活时把 canonical payload 唯一一次
// Pack 到目标 GM cell，但保持 cell control=Empty 且不 clean payload。
// 下一个 Begin/close 只补上最终 flags，校验最终 GM wire，再执行
// payload clean -> barrier -> cell Published。consumer 永远不会读
// Empty cell 的 payload，而 backend 也不再保留任何 callback 指针。
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

PlanPublishResult FinalizeStagedPlan(
    const RuntimePlanView &view, const StagedPlanMetadata &metadata,
    uint8_t final_adapter_flags
)
{
    const uint32_t task_id = metadata.provisional_header.task_id;
    if (view.control == nullptr || view.cells == nullptr ||
        task_id >= view.capacity ||
        AicpuProducerOps::LoadControl(&view.control->fatal.value) != 0 ||
        AicpuProducerOps::LoadControl(
            &view.cells[task_id].control.value
        ) != 0) {
        return PlanPublishResult::CellUnavailable;
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

    RuntimeTaskPlanCell &cell = view.cells[task_id];
    RuntimeTaskPlanHeader expected = metadata.provisional_header;
    expected.adapter_flags = final_adapter_flags;
    AicpuProducerOps::StorePayloadWord(
        &cell.payload.words[2],
        RuntimeTaskPlanHeaderWord(expected, 2U)
    );

    // control 发布前对最终 GM wire 本身做完整校验。这不是
    // 对 staging 副本的替代校验；consumer 之后 acquire 到的就是
    // 此处校验的同一份 bytes。
    RuntimeTaskPlanHeader header{};
    RuntimeTaskPlanLayout validated_layout{};
    if (!ValidateRuntimeTaskPlanPayload(
            cell.payload, task_id, metadata.payload_lines,
            header, validated_layout
        ) ||
        validated_layout.payload_lines != metadata.payload_lines ||
        !HeaderMatchesExpected(header, expected) ||
        !ValidatePaAdapterMetadata(
            header.task_id,
            static_cast<EngineClass>(header.engine_class),
            header.adapter_flags, header.adapter_data
        )) {
        return PlanPublishResult::InvalidInput;
    }

    AicpuProducerOps::FlushRegion(
        &cell.payload,
        metadata.payload_lines * kPlanCacheLineBytes
    );
    AicpuProducerOps::StoreBarrier();
    AicpuProducerOps::PublishControl(
        &cell.control.value,
        static_cast<int64_t>(EncodePlanCellControl(
            PlanCellPhase::Published, metadata.payload_lines, task_id
        ))
    );
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

extern "C" int32_t aicpu_plan_adapter_initialize(
    void *control, void *cells, uint32_t capacity
)
{
    RuntimePlanView view{};
    if (!MakeView(control, cells, capacity, view)) return -1;
    // Host 在启动 producer 前已把 closed 置为 NotReady。StreamingFuture
    // 下 AICore 可能已经在另一条 stream 轮询；PlanAheadClosed 下虽然尚未
    // 启动 AICore，也沿用同一安全复用协议。不能对 closed line 做 civac
    // （它可能把旧 N 写回 GM）；用 ordinary store 覆盖本地 stale value
    // 并精确 clean，保证 initialize 失败能从 -2 发布为 -3 ReadyFailed。
    AicpuProducerOps::PublishControl(
        &view.control->closed_task_count.value,
        kPlanProducerNotReadyTaskCount
    );
    // 同一 storage 的上一轮 Published control 可能仍以 clean-valid 形式
    // 留在 AICPU cache，而本轮 Host aclrtMemset 不会 snoop 它。必须先把
    // 所有 control 的第一条 64B line 丢弃，统一完成 cache maintenance，
    // 之后才允许检查 Empty；不能边 invalidate 边 load。
    for (uint32_t task = 0U; task < capacity; ++task) {
        AicpuProducerOps::DiscardPreviouslyCleanControlLine(
            &view.cells[task].control.value
        );
    }
    AicpuProducerOps::FinishCacheMaintenance();

    // cells 由 Host/AICPU allocator 以零页提供或由 Host 对同一地址重新
    // 清零；这里逐项拒绝非空 control，不覆盖可能仍被 consumer 读取的
    // payload cache line。
    for (uint32_t task = 0U; task < capacity; ++task) {
        if (AicpuProducerOps::LoadControl(
                &view.cells[task].control.value
            ) != 0) {
            (void)PublishRuntimePlanReadyFailed<AicpuProducerOps>(view);
            return -2;
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
#if defined(__aarch64__)
    __asm__ volatile("isb" ::: "memory");
#endif
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
        task_id >= view.capacity ||
        AicpuProducerOps::LoadControl(&view.control->fatal.value) != 0 ||
        AicpuProducerOps::LoadControl(
            &view.cells[task_id].control.value
        ) != 0) {
        return -2;
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
            view.cells[task_id], spec, source, layout
        )) {
        // Pack 中途失败只会留下不可见 payload；control 仍为
        // Empty，FinishTask 会立即发布 fatal，consumer 不会 acquire。
        return -4;
    }

    metadata.provisional_header = DecodeRuntimeTaskPlanHeader(
        view.cells[task_id].payload
    );
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
        metadata.reserved != 0U ||
        metadata.payload_lines != payload_lines ||
        metadata.output_count != provisional.output_count ||
        provisional.task_id >= capacity ||
        provisional.tensor_count > aicpu_plan::kMaxTaskTensors ||
        provisional.scalar_count > aicpu_plan::kMaxTaskScalars ||
        provisional.explicit_dep_count >
            aicpu_plan::kMaxExplicitDependencies ||
        provisional.abi_version != kRuntimePlanAbiVersion) {
        return -3;
    }

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
    // frontier 只是已发布连续前缀的保守摘要。consumer 的
    // future-ticket 直接观察 cell control，所以在真实 batch 尾一次
    // 验证 [batch_start, task_id] 后跳进，避免每 task 一次
    // frontier cvac+dsb+isb。batch_start 来自 Alloc callback provenance。
    if ((final_adapter_flags & kSharedPaTicketLastSubmit) != 0U) {
        const uint32_t new_frontier = provisional.task_id + 1U;
        if (!AdvancePlannedFrontierTo<AicpuProducerOps>(
                view, provisional.adapter_data, new_frontier
            )) {
            return -6;
        }
        // StreamingFuture 的 Ready 只在真实 batch 边界发布，因此
        // 阈值128的G1序列会在连续 frontier=130 时 Open。PlanAheadClosed
        // 在整个生产期都保持 -2；两条路径的短 Plan 都由 close helper
        // 完成同一 Ready -> Closed(N) 握手。
        const int64_t closed = AicpuProducerOps::LoadControl(
            &view.control->closed_task_count.value
        );
        if constexpr (kRuntimePlanPipelineIsStreamingFuture) {
            if (new_frontier >= kRuntimePlanReadyPrefillTasks) {
                if (closed == kPlanProducerNotReadyTaskCount) {
                    if (!PublishRuntimePlanReady<AicpuProducerOps>(view)) {
                        return -7;
                    }
                } else if (closed != kPlanOpenTaskCount) {
                    return -7;
                }
            } else if (closed != kPlanProducerNotReadyTaskCount &&
                       closed != kPlanOpenTaskCount) {
                return -7;
            }
        } else {
            if (closed != kPlanProducerNotReadyTaskCount) return -7;
        }
    }
    return 0;
}

extern "C" int32_t aicpu_plan_adapter_close(
    void *control, void *cells, uint32_t capacity,
    uint32_t final_task_count
)
{
    RuntimePlanView view{};
    if (!MakeView(control, cells, capacity, view)) return -1;
    if constexpr (kRuntimePlanPipelineIsPlanAheadClosed) {
        // 串行 policy 在 Close 前不允许任何 Build consumer 已经启动。
        if (AicpuProducerOps::LoadControl(
                &view.control->build_next.value
            ) != 0 ||
            AicpuProducerOps::LoadControl(
                &view.control->build_workers_done.value
            ) != 0) {
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
