/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */

#include "aicpu_plan_adapter_bridge.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../common/aicpu_plan_pa_adapter.h"

namespace {

using namespace pa_scheduler;
using namespace pa_scheduler::aicpu_plan;
using namespace pa_scheduler::aicpu_plan_adapter;

// AICPU 与 AICore 不具备 Scalar cache coherence。payload 使用 ordinary
// store + exact clean；control 也使用 ordinary store + exact clean，返回型
// atomic observe 由 AIC/AIV consumer 承担。x86 smoke 只替换为顺序一致 fence。
struct AicpuProducerOps {
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

// staging 仅位于 AICPU 私有内存，不需要 cache 操作，也不发布 control。
struct StagingOps {
    static void StorePayloadWord(volatile uint64_t *address, uint64_t value)
    {
        *address = value;
    }
};

struct StagedPlanSource {
    const RuntimeTaskPlanCell &cell;
    RuntimeTaskPlanHeader header;
    RuntimeTaskPlanLayout layout;

    TensorTag TensorTagAt(uint32_t tensor) const
    {
        return static_cast<TensorTag>(header.tensor_tags[tensor]);
    }

    bool TensorIsReference(uint32_t tensor) const
    {
        return (header.tensor_reference_mask & (uint32_t{1} << tensor)) != 0U;
    }

    uint64_t TensorWord(uint32_t tensor, uint32_t word) const
    {
        uint32_t offset = 0U;
        if (!RuntimeTaskPlanTensorWordOffset(header, tensor, offset)) return 0U;
        return cell.payload.words[offset + word];
    }

    uint64_t Scalar(uint32_t scalar) const
    {
        return cell.payload.words[layout.scalar_word_offset + scalar];
    }

    uint64_t ExplicitDependency(uint32_t dependency) const
    {
        return cell.payload.words[
            layout.explicit_dep_word_offset + dependency
        ];
    }
};

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
    // cells 由 Host/AICPU allocator 以零页提供；这里逐项拒绝非空 control，
    // 不用 memset 覆盖可能仍被旧 AICore 读取的 payload cache line。
    for (uint32_t task = 0U; task < capacity; ++task) {
        if (AicpuProducerOps::LoadControl(
                &view.cells[task].control.value
            ) != 0) {
            return -2;
        }
    }
    view.control->planned_frontier.value = 0;
    view.control->closed_task_count.value = kPlanOpenTaskCount;
    view.control->build_next.value = 0;
    view.control->build_workers_done.value = 0;
    view.control->build_release.value = kBuildReleasePending;
    view.control->fatal.value = 0;
    AicpuProducerOps::FlushRegion(view.control, sizeof(*view.control));
    AicpuProducerOps::StoreBarrier();
#if defined(__aarch64__)
    __asm__ volatile("isb" ::: "memory");
#endif
    return 0;
}

extern "C" int32_t aicpu_plan_adapter_stage(
    const void *l0_task_args, uint32_t task_id, int32_t function_id,
    uint8_t engine_class, uint8_t provisional_adapter_flags,
    void *staged_cell, uint32_t *payload_lines, uint16_t *output_count
)
{
    if (l0_task_args == nullptr || staged_cell == nullptr ||
        payload_lines == nullptr || output_count == nullptr ||
        engine_class > static_cast<uint8_t>(EngineClass::Aiv)) {
        return -1;
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
            provisional_adapter_flags, spec
        )) {
        return -2;
    }

    auto *staged = static_cast<RuntimeTaskPlanCell *>(staged_cell);
    std::memset(staged, 0, sizeof(*staged));
    RuntimeTaskPlanLayout layout{};
    const PaRuntimeTaskPlanSource source{args};
    if (!PackRuntimeTaskPlan<StagingOps>(*staged, spec, source, layout)) {
        return -3;
    }
    *payload_lines = layout.payload_lines;
    *output_count = spec.output_count;
    return 0;
}

extern "C" int32_t aicpu_plan_adapter_publish_staged(
    void *control, void *cells, uint32_t capacity,
    const void *staged_cell, uint32_t payload_lines,
    uint8_t final_adapter_flags
)
{
    if (staged_cell == nullptr || payload_lines == 0U ||
        payload_lines > kMaxPlanPayloadLines) {
        return -1;
    }
    RuntimePlanView view{};
    if (!MakeView(control, cells, capacity, view)) return -2;

    const auto &staged =
        *static_cast<const RuntimeTaskPlanCell *>(staged_cell);
    RuntimeTaskPlanHeader header{};
    RuntimeTaskPlanLayout layout{};
    if (!ValidateRuntimeTaskPlanPayload(
            staged.payload,
            DecodeRuntimeTaskPlanHeader(staged.payload).task_id,
            payload_lines, header, layout
        )) {
        return -3;
    }
    const EngineClass engine = static_cast<EngineClass>(header.engine_class);
    if (!ValidatePaAdapterFlags(
            header.task_id, engine, final_adapter_flags
        )) {
        return -4;
    }

    const RuntimeTaskPlanSpec spec{
        header.task_id,
        header.function_id,
        header.tensor_count,
        header.scalar_count,
        header.explicit_dep_count,
        header.output_count,
        engine,
        final_adapter_flags,
        header.core_num,
        header.require_sync_start,
        0U,
        header.tensor_reference_mask,
    };
    const StagedPlanSource source{staged, header, layout};
    if (PublishRuntimeTaskPlan<AicpuProducerOps>(view, spec, source) !=
        PlanPublishResult::Published) {
        return -5;
    }
    if (!AdvancePlannedFrontier<AicpuProducerOps>(view, header.task_id)) {
        return -6;
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
    return CloseRuntimePlan<AicpuProducerOps>(view, final_task_count)
        ? 0
        : -2;
}

extern "C" void aicpu_plan_adapter_publish_fatal(
    void *control, int64_t error_code
)
{
    if (control == nullptr || error_code == 0) return;
    auto *plan_control = static_cast<RuntimePlanControl *>(control);
    AicpuProducerOps::PublishControl(
        &plan_control->fatal.value, error_code
    );
}
