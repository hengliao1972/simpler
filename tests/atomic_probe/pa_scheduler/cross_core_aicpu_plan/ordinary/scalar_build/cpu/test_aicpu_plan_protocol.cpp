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

#include "aicpu_plan_protocol.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

namespace {

using namespace pa_scheduler::aicpu_plan;

constexpr uint32_t kWorkers = 96U;
constexpr uint32_t kPaBlocksPerGroup = 64U;
constexpr uint32_t kPaBlockSize = 128U;
constexpr uint64_t kPayloadPoison = UINT64_C(0xD3D3D3D3D3D3D3D3);

// adapter_flags 是公共协议透传字段；这些 bit 仅属于本测试的 PA 样例，
// 不能反向固化进 common/aicpu_plan_protocol.h。
enum SampleAdapterFlags : uint8_t {
    SampleHasFollowing = 1U << 0U,
    SampleLastInBatch = 1U << 1U,
    SampleLastInPlan = 1U << 2U,
};

struct IoTrace {
    std::atomic<uint32_t> flush_calls{0U};
    std::atomic<uint32_t> invalidate_calls{0U};
    std::atomic<uint64_t> flush_bytes{0U};
    std::atomic<uint64_t> invalidate_bytes{0U};
    std::atomic<uintptr_t> flush_address{0U};
    std::atomic<uintptr_t> invalidate_address{0U};
};

IoTrace g_io_trace;

void ResetIoTrace()
{
    g_io_trace.flush_calls.store(0U, std::memory_order_relaxed);
    g_io_trace.invalidate_calls.store(0U, std::memory_order_relaxed);
    g_io_trace.flush_bytes.store(0U, std::memory_order_relaxed);
    g_io_trace.invalidate_bytes.store(0U, std::memory_order_relaxed);
    g_io_trace.flush_address.store(0U, std::memory_order_relaxed);
    g_io_trace.invalidate_address.store(0U, std::memory_order_relaxed);
}

struct CpuOps {
    static int64_t LoadControl(volatile int64_t *address)
    {
        return __atomic_load_n(address, __ATOMIC_ACQUIRE);
    }

    static void PublishControl(volatile int64_t *address, int64_t value)
    {
        __atomic_store_n(address, value, __ATOMIC_RELEASE);
    }

    static int64_t FetchAddControl(volatile int64_t *address, int64_t value)
    {
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }

    static void StorePayloadWord(volatile uint64_t *address, uint64_t value)
    {
        *address = value;
    }

    static void FlushRegion(const void *address, size_t bytes)
    {
        std::atomic_thread_fence(std::memory_order_release);
        g_io_trace.flush_address.store(reinterpret_cast<uintptr_t>(address), std::memory_order_relaxed);
        g_io_trace.flush_bytes.fetch_add(bytes, std::memory_order_relaxed);
        g_io_trace.flush_calls.fetch_add(1U, std::memory_order_relaxed);
    }

    static void InvalidateRegion(const void *address, size_t bytes)
    {
        std::atomic_thread_fence(std::memory_order_acquire);
        g_io_trace.invalidate_address.store(reinterpret_cast<uintptr_t>(address), std::memory_order_relaxed);
        g_io_trace.invalidate_bytes.fetch_add(bytes, std::memory_order_relaxed);
        g_io_trace.invalidate_calls.fetch_add(1U, std::memory_order_relaxed);
    }

    static void StoreBarrier()
    {
        std::atomic_thread_fence(std::memory_order_release);
    }
};

struct PlanFixture {
    RuntimePlanControl control{};
    std::unique_ptr<RuntimeTaskPlanCell[]> cells;
    uint32_t capacity;

    explicit PlanFixture(uint32_t requested_capacity)
        : cells(new RuntimeTaskPlanCell[requested_capacity]()), capacity(requested_capacity)
    {
        control.planned_frontier.value = 0;
        control.closed_task_count.value = kPlanOpenTaskCount;
        control.build_next.value = 0;
        control.build_workers_done.value = 0;
        control.build_release.value = kBuildReleasePending;
        control.fatal.value = 0;
    }

    RuntimePlanView View()
    {
        RuntimePlanStorageRef storage{};
        storage.cells_base = reinterpret_cast<uint64_t>(cells.get());
        storage.cells_bytes = static_cast<uint64_t>(capacity) * sizeof(RuntimeTaskPlanCell);
        storage.capacity = capacity;
        storage.cell_bytes = sizeof(RuntimeTaskPlanCell);
        storage.abi_version = kRuntimePlanAbiVersion;
        RuntimePlanView view{};
        if (!MakeRuntimePlanView(&control, storage, view)) std::abort();
        return view;
    }
};

struct TaskSource {
    std::array<TensorTag, kMaxTaskTensors> tags{};
    std::array<std::array<uint64_t, kTensorDescWords>, kMaxTaskTensors> tensors{};
    std::array<uint64_t, kMaxTaskScalars> scalars{};
    std::array<uint64_t, kMaxExplicitDependencies> dependencies{};
    uint32_t reference_mask = 0U;

    TensorTag TensorTagAt(uint32_t tensor) const
    {
        return tags[tensor];
    }

    bool TensorIsReference(uint32_t tensor) const
    {
        return (reference_mask & (uint32_t{1} << tensor)) != 0U;
    }

    uint64_t TensorWord(uint32_t tensor, uint32_t word) const
    {
        return tensors[tensor][word];
    }

    uint64_t Scalar(uint32_t scalar) const
    {
        return scalars[scalar];
    }

    uint64_t ExplicitDependency(uint32_t dependency) const
    {
        return dependencies[dependency];
    }
};

struct LogicalTask {
    RuntimeTaskPlanSpec spec{};
    TaskSource source{};
};

[[noreturn]] void Fail(const char *message)
{
    std::fprintf(stderr, "[FAIL] %s\n", message);
    std::exit(1);
}

void Expect(bool condition, const char *message)
{
    if (!condition) Fail(message);
}

uint64_t InlineTensorWord(uint32_t task_id, uint32_t tensor, uint32_t word)
{
    return UINT64_C(0xA500000000000000) ^ (static_cast<uint64_t>(task_id) << 32U) ^
           (static_cast<uint64_t>(tensor) << 16U) ^ word;
}

void FillInlineTensorWords(TaskSource &source, uint32_t task_id)
{
    for (uint32_t tensor = 0U; tensor < kMaxTaskTensors; ++tensor) {
        for (uint32_t word = 0U; word < kTensorDescWords; ++word) {
            source.tensors[tensor][word] = InlineTensorWord(task_id, tensor, word);
        }
    }
}

void SetOutputReference(TaskSource &source, uint32_t tensor, uint32_t producer, uint32_t output_slot)
{
    source.reference_mask |= uint32_t{1} << tensor;
    source.tensors[tensor][0] = static_cast<uint64_t>(producer) | (static_cast<uint64_t>(output_slot) << 32U);
    source.tensors[tensor][1] = 0U;
}

void SetOutputViewReference(
    TaskSource &source, uint32_t tensor, uint32_t producer, uint32_t output_slot,
    uint32_t shape, uint32_t offset
)
{
    source.reference_mask |= uint32_t{1} << tensor;
    const RuntimeOutputReferenceWire reference{
        static_cast<int32_t>(producer), static_cast<int16_t>(output_slot), 1U, 1U, shape, offset,
    };
    source.tensors[tensor][0] = RuntimeOutputReferenceWireWord(reference, 0U);
    source.tensors[tensor][1] = RuntimeOutputReferenceWireWord(reference, 1U);
}

uint32_t CountOutputs(const TaskSource &source, uint32_t tensor_count)
{
    uint32_t outputs = 0U;
    for (uint32_t tensor = 0U; tensor < tensor_count; ++tensor) {
        if (source.tags[tensor] == TensorTag::Output) ++outputs;
    }
    return outputs;
}

RuntimeTaskPlanLayout ExpectedLayout(const LogicalTask &task)
{
    std::array<uint8_t, kMaxTaskTensors> tags{};
    for (uint32_t tensor = 0U; tensor < task.spec.tensor_count; ++tensor) {
        tags[tensor] = static_cast<uint8_t>(task.source.tags[tensor]);
    }
    RuntimeTaskPlanLayout layout{};
    Expect(ComputeRuntimeTaskPlanLayout(task.spec, tags.data(), layout), "logical task has an invalid layout");
    return layout;
}

void PoisonPayload(RuntimeTaskPlanCell &cell)
{
    for (uint32_t word = 0U; word < kMaxPlanPayloadWords; ++word) {
        cell.payload.words[word] = kPayloadPoison;
    }
}

bool AcquiredTaskMatches(
    const RuntimeTaskPlanCell &cell, const LogicalTask &expected, const RuntimeTaskPlanHeader &header,
    const RuntimeTaskPlanLayout &layout
)
{
    if (header.task_id != expected.spec.task_id || header.function_id != expected.spec.function_id ||
        header.tensor_count != expected.spec.tensor_count || header.scalar_count != expected.spec.scalar_count ||
        header.explicit_dep_count != expected.spec.explicit_dep_count ||
        header.output_count != expected.spec.output_count ||
        header.engine_class != static_cast<uint8_t>(expected.spec.engine_class) ||
        header.adapter_flags != expected.spec.adapter_flags ||
        header.core_num != expected.spec.core_num || header.require_sync_start != expected.spec.require_sync_start ||
        header.reserved0 != 0U || header.reserved1 != 0U ||
        header.tensor_reference_mask != expected.spec.tensor_reference_mask ||
        header.abi_version != kRuntimePlanAbiVersion) {
        return false;
    }
    for (uint32_t tensor = 0U; tensor < kMaxTaskTensors; ++tensor) {
        const uint8_t expected_tag = tensor < expected.spec.tensor_count ?
                                         static_cast<uint8_t>(expected.source.tags[tensor]) :
                                         0U;
        if (header.tensor_tags[tensor] != expected_tag) return false;
    }

    const RuntimeTaskPlanLayout expected_layout = ExpectedLayout(expected);
    if (layout.payload_words != expected_layout.payload_words || layout.payload_bytes != expected_layout.payload_bytes ||
        layout.payload_lines != expected_layout.payload_lines ||
        layout.scalar_word_offset != expected_layout.scalar_word_offset ||
        layout.explicit_dep_word_offset != expected_layout.explicit_dep_word_offset) {
        return false;
    }

    uint32_t word = kPlanHeaderWords;
    for (uint32_t tensor = 0U; tensor < expected.spec.tensor_count; ++tensor) {
        const bool reference = (expected.spec.tensor_reference_mask & (uint32_t{1} << tensor)) != 0U;
        const uint32_t meaningful_words = TensorMeaningfulWords(expected.source.tags[tensor], reference);
        for (uint32_t tensor_word = 0U; tensor_word < kTensorCanonicalWords; ++tensor_word) {
            const uint64_t expected_word = tensor_word < meaningful_words ?
                                               expected.source.tensors[tensor][tensor_word] :
                                               0U;
            if (cell.payload.words[word++] != expected_word) return false;
        }
    }
    if (word != layout.scalar_word_offset) return false;
    for (uint32_t scalar = 0U; scalar < expected.spec.scalar_count; ++scalar) {
        if (cell.payload.words[word++] != expected.source.scalars[scalar]) return false;
    }
    if (word != layout.explicit_dep_word_offset) return false;
    for (uint32_t dependency = 0U; dependency < expected.spec.explicit_dep_count; ++dependency) {
        if (cell.payload.words[word++] != expected.source.dependencies[dependency]) return false;
    }
    if (word != layout.payload_words) return false;
    const uint32_t flushed_words = layout.payload_lines * kPlanCacheLineBytes / sizeof(uint64_t);
    while (word < flushed_words) {
        if (cell.payload.words[word++] != 0U) return false;
    }
    return true;
}

LogicalTask MakeMixedLayoutTask(uint32_t task_id)
{
    LogicalTask task{};
    FillInlineTensorWords(task.source, task_id);
    task.spec.task_id = task_id;
    task.spec.function_id = 701U;
    task.spec.tensor_count = kMaxTaskTensors;
    task.spec.scalar_count = kMaxTaskScalars;
    task.spec.explicit_dep_count = kMaxExplicitDependencies;
    task.spec.engine_class = EngineClass::Aiv;
    task.spec.adapter_flags = SampleHasFollowing;
    task.spec.core_num = 7;
    task.spec.require_sync_start = 1U;

    constexpr std::array<TensorTag, 5U> kTags = {
        TensorTag::Input,
        TensorTag::Output,
        TensorTag::Inout,
        TensorTag::OutputExisting,
        TensorTag::NoDependency,
    };
    for (uint32_t tensor = 0U; tensor < kMaxTaskTensors; ++tensor) {
        const TensorTag tag = kTags[tensor % kTags.size()];
        task.source.tags[tensor] = tag;
        if (tag != TensorTag::Output && tensor >= kTags.size()) {
            SetOutputReference(task.source, tensor, tensor % task_id, tensor % kMaxRuntimeOutputsPerTask);
        }
    }
    task.spec.tensor_reference_mask = task.source.reference_mask;
    task.spec.output_count = static_cast<uint16_t>(CountOutputs(task.source, task.spec.tensor_count));
    for (uint32_t scalar = 0U; scalar < kMaxTaskScalars; ++scalar) {
        task.source.scalars[scalar] = UINT64_C(0x5CA1A00000000000) ^ scalar;
    }
    for (uint32_t dependency = 0U; dependency < kMaxExplicitDependencies; ++dependency) {
        task.source.dependencies[dependency] = dependency % task_id;
    }
    return task;
}

LogicalTask MakeMaximumLayoutTask()
{
    LogicalTask task{};
    FillInlineTensorWords(task.source, 0U);
    task.spec.task_id = 0U;
    task.spec.function_id = 702U;
    task.spec.tensor_count = kMaxTaskTensors;
    task.spec.scalar_count = kMaxTaskScalars;
    task.spec.explicit_dep_count = kMaxExplicitDependencies;
    task.spec.output_count = 0U;
    task.spec.engine_class = EngineClass::Aic;
    task.spec.adapter_flags = SampleLastInPlan;
    task.spec.core_num = -1;
    for (uint32_t tensor = 0U; tensor < kMaxTaskTensors; ++tensor) {
        task.source.tags[tensor] = TensorTag::Input;
    }
    for (uint32_t scalar = 0U; scalar < kMaxTaskScalars; ++scalar) {
        task.source.scalars[scalar] = UINT64_C(0x5CA1A10000000000) ^ scalar;
    }
    for (uint32_t dependency = 0U; dependency < kMaxExplicitDependencies; ++dependency) {
        task.source.dependencies[dependency] = dependency;
    }
    return task;
}

LogicalTask MakeSmallInlineTask(uint32_t task_id)
{
    LogicalTask task{};
    FillInlineTensorWords(task.source, task_id);
    task.spec.task_id = task_id;
    task.spec.function_id = 703U;
    task.spec.tensor_count = 2U;
    task.spec.scalar_count = 2U;
    task.spec.explicit_dep_count = task_id == 0U ? 0U : 1U;
    task.spec.output_count = 1U;
    task.spec.engine_class = EngineClass::Aic;
    task.spec.adapter_flags = SampleLastInPlan;
    task.spec.core_num = -1;
    task.source.tags[0] = TensorTag::Input;
    task.source.tags[1] = TensorTag::Output;
    task.source.scalars[0] = task_id;
    task.source.scalars[1] = UINT64_C(0xABCDEF1234567890);
    if (task_id != 0U) task.source.dependencies[0] = task_id - 1U;
    return task;
}

enum class SampleTaskKind : uint8_t {
    Metadata,
    CubeQk,
    VectorSf,
    CubePv,
    VectorUp,
};

EngineClass SampleEngine(SampleTaskKind kind)
{
    if (kind == SampleTaskKind::Metadata) return EngineClass::MetadataOnly;
    if (kind == SampleTaskKind::CubeQk || kind == SampleTaskKind::CubePv) return EngineClass::Aic;
    return EngineClass::Aiv;
}

LogicalTask MakePaSampleTask(
    uint32_t task_id, uint32_t batch, uint32_t batch_start, uint32_t group, uint32_t context,
    uint32_t group_blocks, SampleTaskKind kind, bool has_following, bool last_in_batch, bool last_in_plan
)
{
    LogicalTask task{};
    FillInlineTensorWords(task.source, task_id);
    const EngineClass engine = SampleEngine(kind);
    task.spec.task_id = task_id;
    task.spec.function_id = engine == EngineClass::MetadataOnly ?
                                kInvalidFunctionId :
                                1000U + static_cast<uint32_t>(kind);
    task.spec.engine_class = engine;
    task.spec.adapter_flags = static_cast<uint8_t>(
        (has_following ? static_cast<uint32_t>(SampleHasFollowing) : 0U) |
        (last_in_batch ? static_cast<uint32_t>(SampleLastInBatch) : 0U) |
        (last_in_plan ? static_cast<uint32_t>(SampleLastInPlan) : 0U)
    );
    task.spec.core_num = -1;
    task.spec.require_sync_start = kind == SampleTaskKind::VectorUp ? 1U : 0U;

    if (kind == SampleTaskKind::Metadata) {
        task.spec.tensor_count = 2U;
        task.source.tags[0] = TensorTag::Output;
        task.source.tags[1] = TensorTag::NoDependency;
        task.spec.scalar_count = 3U;
    } else {
        task.spec.tensor_count = 6U;
        task.source.tags[0] = TensorTag::Input;
        task.source.tags[1] = TensorTag::Input;
        task.source.tags[2] = TensorTag::Inout;
        task.source.tags[3] = TensorTag::Output;
        task.source.tags[4] = TensorTag::OutputExisting;
        task.source.tags[5] = TensorTag::NoDependency;
        SetOutputReference(task.source, 0U, task_id - 1U, 0U);
        if (batch_start < task_id) SetOutputReference(task.source, 2U, batch_start, 0U);
        SetOutputReference(task.source, 4U, task_id - 1U, 0U);
        task.spec.scalar_count = 6U;
        task.spec.explicit_dep_count = batch_start + 1U < task_id ? 2U : 1U;
        task.source.dependencies[0] = task_id - 1U;
        if (task.spec.explicit_dep_count == 2U) task.source.dependencies[1] = batch_start;
    }
    task.spec.tensor_reference_mask = task.source.reference_mask;
    task.spec.output_count = static_cast<uint16_t>(CountOutputs(task.source, task.spec.tensor_count));
    task.source.scalars[0] = batch;
    task.source.scalars[1] = batch_start;
    task.source.scalars[2] = group;
    task.source.scalars[3] = context;
    task.source.scalars[4] = group_blocks;
    task.source.scalars[5] = static_cast<uint32_t>(kind);
    return task;
}

uint32_t PaTaskCount(const std::vector<uint32_t> &context_lens)
{
    uint32_t count = 0U;
    for (const uint32_t context : context_lens) {
        const uint32_t blocks = (context + kPaBlockSize - 1U) / kPaBlockSize;
        const uint32_t groups = (blocks + kPaBlocksPerGroup - 1U) / kPaBlocksPerGroup;
        count += 1U + 4U * groups;
    }
    return count;
}

std::vector<LogicalTask> MakePaSamplePlan(const std::vector<uint32_t> &context_lens)
{
    const uint32_t final_count = PaTaskCount(context_lens);
    std::vector<LogicalTask> tasks;
    tasks.reserve(final_count);
    for (uint32_t batch = 0U; batch < context_lens.size(); ++batch) {
        const uint32_t context = context_lens[batch];
        const uint32_t blocks = (context + kPaBlockSize - 1U) / kPaBlockSize;
        const uint32_t groups = (blocks + kPaBlocksPerGroup - 1U) / kPaBlocksPerGroup;
        const uint32_t batch_start = static_cast<uint32_t>(tasks.size());
        uint32_t task_id = static_cast<uint32_t>(tasks.size());
        tasks.push_back(MakePaSampleTask(
            task_id, batch, batch_start, 0U, context, 0U, SampleTaskKind::Metadata, false, groups == 0U,
            task_id + 1U == final_count
        ));
        for (uint32_t group = 0U; group < groups; ++group) {
            const uint32_t consumed_blocks = group * kPaBlocksPerGroup;
            const uint32_t group_blocks =
                blocks - consumed_blocks < kPaBlocksPerGroup ? blocks - consumed_blocks : kPaBlocksPerGroup;
            constexpr std::array<SampleTaskKind, 4U> kGroupTasks = {
                SampleTaskKind::CubeQk,
                SampleTaskKind::VectorSf,
                SampleTaskKind::CubePv,
                SampleTaskKind::VectorUp,
            };
            for (uint32_t within_group = 0U; within_group < kGroupTasks.size(); ++within_group) {
                task_id = static_cast<uint32_t>(tasks.size());
                const bool last = group + 1U == groups && within_group + 1U == kGroupTasks.size();
                tasks.push_back(MakePaSampleTask(
                    task_id, batch, batch_start, group, context, group_blocks, kGroupTasks[within_group],
                    within_group + 1U == kGroupTasks.size() && group + 1U < groups, last,
                    task_id + 1U == final_count
                ));
            }
        }
    }
    Expect(tasks.size() == final_count, "PA sample generator task count mismatch");
    return tasks;
}

void PublishPlan(const RuntimePlanView &view, const std::vector<LogicalTask> &tasks)
{
    for (uint32_t task_id = 0U; task_id < tasks.size(); ++task_id) {
        Expect(tasks[task_id].spec.task_id == task_id, "logical Plan identity is not contiguous");
        PoisonPayload(view.cells[task_id]);
        Expect(
            PublishRuntimeTaskPlan<CpuOps>(view, tasks[task_id].spec, tasks[task_id].source) ==
                PlanPublishResult::Published,
            "PlanCell publication failed"
        );
        Expect(AdvancePlannedFrontier<CpuOps>(view, task_id), "planned frontier did not advance");
    }
    Expect(CloseRuntimePlan<CpuOps>(view, static_cast<uint32_t>(tasks.size())), "Plan close failed");
}

void TestStorageReference()
{
    PlanFixture fixture(3U);
    alignas(kAtomicIsolationBytes) RuntimePlanStorageRef storage{};
    storage.cells_base = reinterpret_cast<uint64_t>(fixture.cells.get());
    storage.cells_bytes = static_cast<uint64_t>(fixture.capacity) * sizeof(RuntimeTaskPlanCell);
    storage.capacity = fixture.capacity;
    storage.cell_bytes = sizeof(RuntimeTaskPlanCell);
    storage.abi_version = kRuntimePlanAbiVersion;
    Expect(RuntimePlanStorageRefValid(storage), "valid Plan storage reference was rejected");
    Expect(reinterpret_cast<uintptr_t>(&storage) % kAtomicIsolationBytes == 0U, "StorageRef object is not 128B aligned");

    RuntimePlanView checked_view{};
    Expect(
        MakeRuntimePlanView(&fixture.control, storage, checked_view),
        "checked Plan view construction rejected valid storage"
    );
    Expect(
        checked_view.control == &fixture.control && checked_view.cells == fixture.cells.get() &&
            checked_view.capacity == fixture.capacity,
        "checked Plan view construction changed storage identity"
    );

    RuntimePlanStorageRef invalid = storage;
    ++invalid.cells_base;
    Expect(!RuntimePlanStorageRefValid(invalid), "misaligned Plan storage reference was accepted");
    invalid = storage;
    --invalid.cells_bytes;
    Expect(!RuntimePlanStorageRefValid(invalid), "short Plan storage allocation was accepted");
    invalid = storage;
    invalid.capacity = 0U;
    Expect(!RuntimePlanStorageRefValid(invalid), "zero Plan capacity was accepted");
    invalid = storage;
    invalid.reserved[2] = 1U;
    Expect(!RuntimePlanStorageRefValid(invalid), "nonzero Plan storage reserved word was accepted");
    checked_view = RuntimePlanView{&fixture.control, fixture.cells.get(), fixture.capacity};
    Expect(
        !MakeRuntimePlanView(nullptr, storage, checked_view) && checked_view.control == nullptr &&
            checked_view.cells == nullptr && checked_view.capacity == 0U,
        "checked Plan view did not fail closed for a null control"
    );
    checked_view = RuntimePlanView{&fixture.control, fixture.cells.get(), fixture.capacity};
    Expect(
        !MakeRuntimePlanView(&fixture.control, invalid, checked_view) && checked_view.control == nullptr &&
            checked_view.cells == nullptr && checked_view.capacity == 0U,
        "checked Plan view did not fail closed for invalid storage"
    );
    std::printf("[PASS] storage-ref\n");
}

void TestLayoutAndPublication()
{
    {
        PlanFixture fixture(18U);
        RuntimePlanView view = fixture.View();
        CpuOps::PublishControl(&fixture.control.planned_frontier.value, 17);
        LogicalTask task = MakeMixedLayoutTask(17U);
        PoisonPayload(view.cells[17]);
        ResetIoTrace();
        Expect(
            PublishRuntimeTaskPlan<CpuOps>(view, task.spec, task.source) == PlanPublishResult::Published,
            "32-tensor mixed PlanCell publication failed"
        );
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        Expect(
            AcquireRuntimeTaskPlan<CpuOps>(view, 17U, header, layout) == PlanAcquireResult::Acquired,
            "32-tensor mixed PlanCell acquire failed"
        );
        Expect(AcquiredTaskMatches(view.cells[17], task, header, layout), "32-tensor mixed layout mismatch");
        Expect(layout.payload_lines > 1U, "mixed PlanCell did not span multiple cache lines");
        Expect(g_io_trace.flush_calls.load(std::memory_order_relaxed) == 1U, "mixed publish flush count mismatch");
        Expect(
            g_io_trace.invalidate_calls.load(std::memory_order_relaxed) == 1U,
            "mixed acquire invalidate count mismatch"
        );
        Expect(
            g_io_trace.flush_bytes.load(std::memory_order_relaxed) ==
                static_cast<uint64_t>(layout.payload_lines) * kPlanCacheLineBytes,
            "mixed publish flushed the wrong byte range"
        );
        Expect(
            g_io_trace.invalidate_bytes.load(std::memory_order_relaxed) ==
                static_cast<uint64_t>(layout.payload_lines) * kPlanCacheLineBytes,
            "mixed acquire invalidated the wrong byte range"
        );
        Expect(
            g_io_trace.flush_address.load(std::memory_order_relaxed) ==
                reinterpret_cast<uintptr_t>(&view.cells[17].payload),
            "mixed publish flushed the wrong address"
        );
        Expect(
            g_io_trace.invalidate_address.load(std::memory_order_relaxed) ==
                reinterpret_cast<uintptr_t>(&view.cells[17].payload),
            "mixed acquire invalidated the wrong address"
        );
    }

    {
        PlanFixture fixture(1U);
        RuntimePlanView view = fixture.View();
        LogicalTask task = MakeMaximumLayoutTask();
        PoisonPayload(view.cells[0]);
        ResetIoTrace();
        Expect(
            PublishRuntimeTaskPlan<CpuOps>(view, task.spec, task.source) == PlanPublishResult::Published,
            "maximum PlanCell publication failed"
        );
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        Expect(
            AcquireRuntimeTaskPlan<CpuOps>(view, 0U, header, layout) == PlanAcquireResult::Acquired,
            "maximum PlanCell acquire failed"
        );
        Expect(AcquiredTaskMatches(view.cells[0], task, header, layout), "maximum PlanCell layout mismatch");
        Expect(layout.payload_words == kMaxPlanPayloadWords, "maximum payload word count mismatch");
        Expect(layout.payload_bytes == kMaxPlanPayloadBytes, "maximum payload byte count mismatch");
        Expect(layout.payload_lines == kMaxPlanPayloadLines, "maximum payload line count mismatch");
        Expect(
            g_io_trace.flush_bytes.load(std::memory_order_relaxed) == kMaxPlanPayloadBytes,
            "maximum PlanCell flush was not exact"
        );
        Expect(
            g_io_trace.invalidate_bytes.load(std::memory_order_relaxed) == kMaxPlanPayloadBytes,
            "maximum PlanCell invalidate was not exact"
        );
    }
    std::printf("[PASS] 32-tensor-ref-inline-scalar-dep-multiline\n");
}

struct UnstableControlOps {
    static volatile int64_t *target;
    static std::atomic<uint32_t> target_reads;

    static int64_t LoadControl(volatile int64_t *address)
    {
        const int64_t value = CpuOps::LoadControl(address);
        if (address == target && target_reads.fetch_add(1U, std::memory_order_relaxed) == 1U) {
            return value ^ static_cast<int64_t>(uint64_t{1} << kPlanPayloadLinesShift);
        }
        return value;
    }

    static void InvalidateRegion(const void *address, size_t bytes)
    {
        CpuOps::InvalidateRegion(address, bytes);
    }
};

volatile int64_t *UnstableControlOps::target = nullptr;
std::atomic<uint32_t> UnstableControlOps::target_reads{0U};

void PublishOne(const RuntimePlanView &view, const LogicalTask &task)
{
    PoisonPayload(view.cells[task.spec.task_id]);
    Expect(
        PublishRuntimeTaskPlan<CpuOps>(view, task.spec, task.source) == PlanPublishResult::Published,
        "valid corruption-test PlanCell publication failed"
    );
}

void TestMalformedPlans()
{
    {
        PlanFixture fixture(2U);
        RuntimePlanView view = fixture.View();
        LogicalTask task = MakeSmallInlineTask(0U);
        PublishOne(view, task);
        Expect(
            PublishRuntimeTaskPlan<CpuOps>(view, task.spec, task.source) == PlanPublishResult::CellUnavailable,
            "duplicate PlanCell publication was accepted"
        );
        Expect(AdvancePlannedFrontier<CpuOps>(view, 0U), "valid duplicate-test frontier advance failed");
        Expect(!AdvancePlannedFrontier<CpuOps>(view, 0U), "duplicate frontier advance was accepted");
        LogicalTask skipped = MakeSmallInlineTask(1U);
        skipped.spec.task_id = 1U;
        Expect(
            PublishRuntimeTaskPlan<CpuOps>(view, skipped.spec, skipped.source) == PlanPublishResult::Published,
            "contiguous PlanCell after duplicate rejection failed"
        );
    }

    {
        PlanFixture fixture(2U);
        RuntimePlanView view = fixture.View();
        LogicalTask missing = MakeSmallInlineTask(1U);
        Expect(
            PublishRuntimeTaskPlan<CpuOps>(view, missing.spec, missing.source) == PlanPublishResult::CellUnavailable,
            "Plan publication skipped a missing task"
        );
        Expect(!CloseRuntimePlan<CpuOps>(view, 1U), "Plan with a missing prefix cell was closed");
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        Expect(
            AcquireRuntimeTaskPlan<CpuOps>(view, 0U, header, layout) == PlanAcquireResult::NotPublished,
            "missing PlanCell was not reported"
        );
        CpuOps::PublishControl(&fixture.control.planned_frontier.value, 1);
        CpuOps::PublishControl(&fixture.control.closed_task_count.value, 1);
        Expect(
            TakeClosedPlanBuildTicket<CpuOps>(view).status == BuildReservationStatus::Reserved,
            "malformed closed Plan did not issue its bounded ticket"
        );
        Expect(
            AcquireRuntimeTaskPlan<CpuOps>(view, 0U, header, layout) == PlanAcquireResult::NotPublished,
            "consumer did not fail closed on a missing reserved PlanCell"
        );
    }

    {
        PlanFixture fixture(1U);
        RuntimePlanView view = fixture.View();
        PublishOne(view, MakeSmallInlineTask(0U));
        view.cells[0].payload.words[7] ^= uint64_t{1} << 32U;
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        Expect(
            AcquireRuntimeTaskPlan<CpuOps>(view, 0U, header, layout) == PlanAcquireResult::InvalidPayload,
            "corrupted payload ABI was accepted"
        );
    }

    {
        PlanFixture fixture(1U);
        RuntimePlanView view = fixture.View();
        PublishOne(view, MakeSmallInlineTask(0U));
        view.cells[0].payload.words[3] = (view.cells[0].payload.words[3] & ~UINT64_C(0xFF)) | UINT64_C(0xFF);
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        Expect(
            AcquireRuntimeTaskPlan<CpuOps>(view, 0U, header, layout) == PlanAcquireResult::InvalidPayload,
            "corrupted tensor tag was accepted"
        );
    }

    {
        PlanFixture fixture(2U);
        RuntimePlanView view = fixture.View();
        CpuOps::PublishControl(&fixture.control.planned_frontier.value, 1);
        LogicalTask future = MakeSmallInlineTask(1U);
        future.source.tags[0] = TensorTag::Input;
        SetOutputReference(future.source, 0U, 1U, 0U);
        future.spec.tensor_reference_mask = future.source.reference_mask;
        Expect(
            PublishRuntimeTaskPlan<CpuOps>(view, future.spec, future.source) == PlanPublishResult::InvalidInput,
            "publisher accepted a future/self output reference"
        );
    }

    {
        PlanFixture fixture(2U);
        RuntimePlanView view = fixture.View();
        CpuOps::PublishControl(&fixture.control.planned_frontier.value, 1);
        LogicalTask invalid_slot = MakeSmallInlineTask(1U);
        SetOutputReference(invalid_slot.source, 0U, 0U, kMaxRuntimeOutputsPerTask);
        invalid_slot.spec.tensor_reference_mask = invalid_slot.source.reference_mask;
        Expect(
            PublishRuntimeTaskPlan<CpuOps>(view, invalid_slot.spec, invalid_slot.source) ==
                PlanPublishResult::InvalidInput,
            "publisher accepted output_slot >= 8"
        );
    }

    {
        PlanFixture fixture(2U);
        RuntimePlanView view = fixture.View();
        CpuOps::PublishControl(&fixture.control.planned_frontier.value, 1);
        LogicalTask malformed_plain = MakeSmallInlineTask(1U);
        SetOutputReference(malformed_plain.source, 0U, 0U, 0U);
        malformed_plain.source.tensors[0][1] = 1U;
        malformed_plain.spec.tensor_reference_mask = malformed_plain.source.reference_mask;
        Expect(
            PublishRuntimeTaskPlan<CpuOps>(view, malformed_plain.spec, malformed_plain.source) ==
                PlanPublishResult::InvalidInput,
            "publisher accepted a plain reference with a nonzero second word"
        );
    }

    {
        PlanFixture fixture(2U);
        RuntimePlanView view = fixture.View();
        CpuOps::PublishControl(&fixture.control.planned_frontier.value, 1);
        LogicalTask malformed_view = MakeSmallInlineTask(1U);
        SetOutputViewReference(malformed_view.source, 0U, 0U, 0U, 2U, UINT32_MAX - 1U);
        malformed_view.spec.tensor_reference_mask = malformed_view.source.reference_mask;
        Expect(
            PublishRuntimeTaskPlan<CpuOps>(view, malformed_view.spec, malformed_view.source) ==
                PlanPublishResult::InvalidInput,
            "publisher accepted an overflowing one-dimensional view"
        );
    }

    {
        PlanFixture fixture(2U);
        RuntimePlanView view = fixture.View();
        CpuOps::PublishControl(&fixture.control.planned_frontier.value, 1);
        LogicalTask plain = MakeSmallInlineTask(1U);
        SetOutputReference(plain.source, 0U, 0U, 0U);
        plain.spec.tensor_reference_mask = plain.source.reference_mask;
        PublishOne(view, plain);
        view.cells[1].payload.words[kPlanHeaderWords + 1U] = 1U;
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        Expect(
            AcquireRuntimeTaskPlan<CpuOps>(view, 1U, header, layout) == PlanAcquireResult::InvalidPayload,
            "consumer accepted a bit-flipped plain-reference second word"
        );
    }

    {
        PlanFixture fixture(2U);
        RuntimePlanView view = fixture.View();
        CpuOps::PublishControl(&fixture.control.planned_frontier.value, 1);
        LogicalTask plain = MakeSmallInlineTask(1U);
        SetOutputReference(plain.source, 0U, 0U, 0U);
        plain.spec.tensor_reference_mask = plain.source.reference_mask;
        PublishOne(view, plain);
        view.cells[1].payload.words[kPlanHeaderWords] = 1U;
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        Expect(
            AcquireRuntimeTaskPlan<CpuOps>(view, 1U, header, layout) == PlanAcquireResult::InvalidPayload,
            "consumer accepted a bit-flipped self reference"
        );
    }

    {
        PlanFixture fixture(2U);
        RuntimePlanView view = fixture.View();
        CpuOps::PublishControl(&fixture.control.planned_frontier.value, 1);
        LogicalTask plain = MakeSmallInlineTask(1U);
        SetOutputReference(plain.source, 0U, 0U, 0U);
        plain.spec.tensor_reference_mask = plain.source.reference_mask;
        PublishOne(view, plain);
        view.cells[1].payload.words[kPlanHeaderWords] =
            static_cast<uint64_t>(kMaxRuntimeOutputsPerTask) << 32U;
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        Expect(
            AcquireRuntimeTaskPlan<CpuOps>(view, 1U, header, layout) == PlanAcquireResult::InvalidPayload,
            "consumer accepted a bit-flipped output_slot >= 8"
        );
    }

    {
        PlanFixture fixture(2U);
        RuntimePlanView view = fixture.View();
        CpuOps::PublishControl(&fixture.control.planned_frontier.value, 1);
        LogicalTask view_task = MakeSmallInlineTask(1U);
        SetOutputViewReference(view_task.source, 0U, 0U, 0U, 16U, 32U);
        view_task.spec.tensor_reference_mask = view_task.source.reference_mask;
        PublishOne(view, view_task);
        view.cells[1].payload.words[kPlanHeaderWords + 1U] =
            static_cast<uint64_t>(16U) | (static_cast<uint64_t>(UINT32_MAX - 15U) << 32U);
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        Expect(
            AcquireRuntimeTaskPlan<CpuOps>(view, 1U, header, layout) == PlanAcquireResult::InvalidPayload,
            "consumer accepted a bit-flipped overflowing view"
        );
    }

    {
        PlanFixture fixture(1U);
        RuntimePlanView view = fixture.View();
        PublishOne(view, MakeSmallInlineTask(0U));
        const int64_t valid = CpuOps::LoadControl(&view.cells[0].control.value);
        CpuOps::PublishControl(
            &view.cells[0].control.value,
            valid | static_cast<int64_t>(uint64_t{1} << 63U)
        );
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        Expect(
            AcquireRuntimeTaskPlan<CpuOps>(view, 0U, header, layout) == PlanAcquireResult::InvalidControl,
            "control with unknown bits was accepted"
        );
    }

    {
        PlanFixture fixture(2U);
        RuntimePlanView view = fixture.View();
        PublishOne(view, MakeSmallInlineTask(0U));
        const DecodedPlanCellControl decoded = DecodePlanCellControl(CpuOps::LoadControl(&view.cells[0].control.value));
        CpuOps::PublishControl(
            &view.cells[0].control.value,
            static_cast<int64_t>(EncodePlanCellControl(PlanCellPhase::Published, decoded.payload_lines, 1U))
        );
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        Expect(
            AcquireRuntimeTaskPlan<CpuOps>(view, 0U, header, layout) == PlanAcquireResult::InvalidControl,
            "control with the wrong task identity was accepted"
        );
    }

    {
        PlanFixture fixture(1U);
        RuntimePlanView view = fixture.View();
        PublishOne(view, MakeSmallInlineTask(0U));
        UnstableControlOps::target = &view.cells[0].control.value;
        UnstableControlOps::target_reads.store(0U, std::memory_order_relaxed);
        RuntimeTaskPlanHeader header{};
        RuntimeTaskPlanLayout layout{};
        Expect(
            AcquireRuntimeTaskPlan<UnstableControlOps>(view, 0U, header, layout) ==
                PlanAcquireResult::InvalidControl,
            "control changed across acquire was accepted"
        );
        UnstableControlOps::target = nullptr;
    }

    {
        LogicalTask task = MakeSmallInlineTask(0U);
        std::array<uint8_t, kMaxTaskTensors> tags{};
        RuntimeTaskPlanLayout layout{};
        task.spec.tensor_count = kMaxTaskTensors + 1U;
        Expect(!ComputeRuntimeTaskPlanLayout(task.spec, tags.data(), layout), "tensor-count overflow was accepted");
        task = MakeSmallInlineTask(0U);
        task.spec.scalar_count = kMaxTaskScalars + 1U;
        Expect(!ComputeRuntimeTaskPlanLayout(task.spec, tags.data(), layout), "scalar-count overflow was accepted");
        task = MakeSmallInlineTask(0U);
        task.spec.explicit_dep_count = kMaxExplicitDependencies + 1U;
        Expect(!ComputeRuntimeTaskPlanLayout(task.spec, tags.data(), layout), "dependency-count overflow was accepted");
        task = MakeSmallInlineTask(0U);
        task.spec.tensor_reference_mask = uint32_t{1} << task.spec.tensor_count;
        Expect(!ComputeRuntimeTaskPlanLayout(task.spec, tags.data(), layout), "reference-mask overflow was accepted");

        PlanFixture fixture(1U);
        RuntimePlanView view = fixture.View();
        LogicalTask outside = MakeSmallInlineTask(1U);
        Expect(
            PublishRuntimeTaskPlan<CpuOps>(view, outside.spec, outside.source) == PlanPublishResult::CellUnavailable,
            "task beyond Plan capacity was accepted"
        );
        Expect(!CloseRuntimePlan<CpuOps>(view, 2U), "Plan close beyond capacity was accepted");
        CpuOps::PublishControl(&fixture.control.planned_frontier.value, 2);
        CpuOps::PublishControl(&fixture.control.closed_task_count.value, 2);
        Expect(
            TakeClosedPlanBuildTicket<CpuOps>(view).status == BuildReservationStatus::Fatal,
            "closed Plan frontier beyond capacity did not fail"
        );
        Expect(CpuOps::LoadControl(&fixture.control.build_next.value) == 0, "fatal ticket path consumed build_next");
    }

    {
        PlanFixture fixture(1U);
        RuntimePlanView view = fixture.View();
        Expect(
            CloseRuntimePlan<CpuOps>(view, 0U),
            "zero-task Plan did not close"
        );
        Expect(
            PublishClosedPlanBuildRelease<CpuOps>(view, 0U, 2U) == false,
            "Build release was published before arrivals"
        );
        Expect(
            ArriveClosedPlanBuildWorker<CpuOps>(view, 2U) == BuildArrivalStatus::Arrived,
            "first Build arrival status mismatch"
        );
        Expect(
            ArriveClosedPlanBuildWorker<CpuOps>(view, 2U) == BuildArrivalStatus::Last,
            "last Build arrival status mismatch"
        );
        CpuOps::PublishControl(&fixture.control.build_next.value, 1);
        Expect(
            PublishClosedPlanBuildRelease<CpuOps>(view, 0U, 2U) == false,
            "Build release accepted an incomplete N+W ticket frontier"
        );
        CpuOps::PublishControl(&fixture.control.build_next.value, 2);
        Expect(
            PublishClosedPlanBuildRelease<CpuOps>(view, 0U, 2U),
            "zero-task Build release rejected the exact N+W frontier"
        );
        Expect(
            ArriveClosedPlanBuildWorker<CpuOps>(view, 2U) == BuildArrivalStatus::Invalid,
            "duplicate Build arrival was accepted"
        );
    }
    std::printf("[PASS] malformed-payload-control-ref-duplicate-missing-overflow\n");
}

void MarkWorkerFailure(PlanFixture &fixture, std::atomic<uint32_t> &failure, uint32_t code)
{
    uint32_t expected = 0U;
    (void)failure.compare_exchange_strong(expected, code, std::memory_order_relaxed);
    CpuOps::PublishControl(&fixture.control.fatal.value, 1);
}

void RunClosedPlanCase(const char *name, const std::vector<uint32_t> &context_lens)
{
    const std::vector<LogicalTask> tasks = MakePaSamplePlan(context_lens);
    const uint32_t task_count = static_cast<uint32_t>(tasks.size());
    PlanFixture fixture(task_count);
    RuntimePlanView view = fixture.View();
    PublishPlan(view, tasks);

    std::vector<std::atomic<uint32_t>> reservations(task_count);
    std::vector<std::atomic<int64_t>> insert_predecessors(task_count);
    for (uint32_t task = 0U; task < task_count; ++task) {
        reservations[task].store(0U, std::memory_order_relaxed);
        insert_predecessors[task].store(INT64_MIN, std::memory_order_relaxed);
    }
    std::atomic<int64_t> insert_frontier{-1};
    std::atomic<uint32_t> completed{0U};
    std::atomic<uint32_t> failure{0U};
    std::atomic<uint32_t> ready{0U};
    std::atomic<bool> start{false};
    std::atomic<uint32_t> last_arrivals{0U};
    std::atomic<uint32_t> released_workers{0U};

    std::vector<std::thread> builders;
    builders.reserve(kWorkers);
    for (uint32_t worker = 0U; worker < kWorkers; ++worker) {
        builders.emplace_back([&, worker] {
            ready.fetch_add(1U, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            while (true) {
                const BuildReservation ticket = TakeClosedPlanBuildTicket<CpuOps>(view);
                if (ticket.status == BuildReservationStatus::Closed) break;
                if (ticket.status != BuildReservationStatus::Reserved || ticket.task_id >= task_count) {
                    MarkWorkerFailure(fixture, failure, 1U);
                    return;
                }
                RuntimeTaskPlanHeader header{};
                RuntimeTaskPlanLayout layout{};
                if (AcquireRuntimeTaskPlan<CpuOps>(view, ticket.task_id, header, layout) !=
                    PlanAcquireResult::Acquired) {
                    MarkWorkerFailure(fixture, failure, 2U);
                    return;
                }
                if (!AcquiredTaskMatches(view.cells[ticket.task_id], tasks[ticket.task_id], header, layout)) {
                    MarkWorkerFailure(fixture, failure, 3U);
                    return;
                }
                if (reservations[ticket.task_id].fetch_add(1U, std::memory_order_relaxed) != 0U) {
                    MarkWorkerFailure(fixture, failure, 4U);
                    return;
                }

                const int64_t predecessor = static_cast<int64_t>(ticket.task_id) - 1;
                while (insert_frontier.load(std::memory_order_acquire) != predecessor) {
                    if (CpuOps::LoadControl(&fixture.control.fatal.value) != 0) return;
                    std::this_thread::yield();
                }
                int64_t expected = predecessor;
                if (!insert_frontier.compare_exchange_strong(
                        expected, static_cast<int64_t>(ticket.task_id), std::memory_order_acq_rel,
                        std::memory_order_acquire
                    )) {
                    MarkWorkerFailure(fixture, failure, 5U);
                    return;
                }
                insert_predecessors[ticket.task_id].store(predecessor, std::memory_order_release);
                completed.fetch_add(1U, std::memory_order_relaxed);
            }

            const BuildArrivalStatus arrival = ArriveClosedPlanBuildWorker<CpuOps>(view, kWorkers);
            if (arrival == BuildArrivalStatus::Last) {
                last_arrivals.fetch_add(1U, std::memory_order_relaxed);
                if (!PublishClosedPlanBuildRelease<CpuOps>(view, task_count, kWorkers)) {
                    MarkWorkerFailure(fixture, failure, 6U);
                    return;
                }
            } else if (arrival != BuildArrivalStatus::Arrived) {
                MarkWorkerFailure(fixture, failure, 7U);
                return;
            }
            while (CpuOps::LoadControl(&fixture.control.build_release.value) != static_cast<int64_t>(task_count)) {
                if (CpuOps::LoadControl(&fixture.control.fatal.value) != 0) return;
                std::this_thread::yield();
            }
            released_workers.fetch_add(1U, std::memory_order_relaxed);
            (void)worker;
        });
    }

    while (ready.load(std::memory_order_acquire) != kWorkers) std::this_thread::yield();
    start.store(true, std::memory_order_release);
    for (std::thread &builder : builders) builder.join();

    if (failure.load(std::memory_order_relaxed) != 0U) {
        std::fprintf(stderr, "[FAIL] %s worker failure code=%u\n", name, failure.load(std::memory_order_relaxed));
        std::exit(1);
    }
    Expect(completed.load(std::memory_order_relaxed) == task_count, "not every closed-Plan task completed Build");
    Expect(
        insert_frontier.load(std::memory_order_acquire) == static_cast<int64_t>(task_count) - 1,
        "strict TensorMap insertion frontier did not close"
    );
    Expect(
        CpuOps::LoadControl(&fixture.control.planned_frontier.value) == task_count,
        "planned frontier terminal count mismatch"
    );
    Expect(
        CpuOps::LoadControl(&fixture.control.closed_task_count.value) == task_count,
        "closed Plan terminal count mismatch"
    );
    Expect(
        CpuOps::LoadControl(&fixture.control.build_next.value) == static_cast<int64_t>(task_count) + kWorkers,
        "central FetchAdd did not close at N+96"
    );
    Expect(
        CpuOps::LoadControl(&fixture.control.build_workers_done.value) == kWorkers,
        "Build worker arrival count mismatch"
    );
    Expect(
        CpuOps::LoadControl(&fixture.control.build_release.value) == task_count,
        "Build release terminal count mismatch"
    );
    Expect(last_arrivals.load(std::memory_order_relaxed) == 1U, "Build barrier did not elect exactly one last worker");
    Expect(released_workers.load(std::memory_order_relaxed) == kWorkers, "not every Build worker observed release");
    for (uint32_t task = 0U; task < task_count; ++task) {
        Expect(reservations[task].load(std::memory_order_relaxed) == 1U, "task was lost or reserved more than once");
        Expect(
            insert_predecessors[task].load(std::memory_order_acquire) == static_cast<int64_t>(task) - 1,
            "strict TensorMap insertion predecessor mismatch"
        );
    }
    std::printf("[PASS] %s tasks=%u workers=%u tickets=%u\n", name, task_count, kWorkers, task_count + kWorkers);
}

}  // namespace

int main()
{
    TestStorageReference();
    TestLayoutAndPublication();
    TestMalformedPlans();
    RunClosedPlanCase("PA-B1", {8193U});
    RunClosedPlanCase("PA-G0", {0U});
    RunClosedPlanCase("PA-G1", {8192U});
    RunClosedPlanCase("PA-mixed", {0U, 1U, 8192U, 8193U});
    std::vector<uint32_t> b256_contexts;
    b256_contexts.reserve(256U);
    constexpr std::array<uint32_t, 4U> kMixedContexts = {0U, 1U, 8192U, 8193U};
    for (uint32_t batch = 0U; batch < 256U; ++batch) {
        b256_contexts.push_back(kMixedContexts[batch % kMixedContexts.size()]);
    }
    RunClosedPlanCase("PA-B256-mixed", b256_contexts);
    return 0;
}
