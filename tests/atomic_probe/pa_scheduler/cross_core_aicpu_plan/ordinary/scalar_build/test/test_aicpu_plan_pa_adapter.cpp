/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "aicpu_plan_pa_adapter.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace {

using namespace pa_scheduler;
using namespace pa_scheduler::aicpu_plan;
using namespace pa_scheduler::aicpu_plan_adapter;

constexpr uint32_t kTaskId = 17U;
constexpr uint32_t kFunctionId = 913U;
constexpr uint64_t kPayloadPoison = UINT64_C(0xD7D7D7D7D7D7D7D7);

// 该值是 PTO2TaskId 的正式 raw ABI：(ring_id << 32) | local_id。
// 测试只传递 raw，不把 host 指针伪装成显式依赖。
constexpr uint64_t PtoTaskIdRaw(uint8_t ring_id, uint32_t local_id)
{
    return (static_cast<uint64_t>(ring_id) << 32U) |
           static_cast<uint64_t>(local_id);
}

// PA adapter_flags 使用现有 shared-ticket byte ABI；公共 Plan 协议只搬运
// 该字节，不解释 task kind。这里选择一个合法的 group-2 PV 样例。
constexpr uint8_t kValidAdapterFlags = static_cast<uint8_t>(
    kSharedPaTicketMetaPresent |
    (2U << kSharedPaTicketGroupShift) |
    static_cast<uint32_t>(TaskKind::Pv)
);

[[noreturn]] void Fail(const char *message)
{
    std::fprintf(stderr, "[FAIL] %s\n", message);
    std::exit(1);
}

void Expect(bool condition, const char *message)
{
    if (!condition) Fail(message);
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

    static int64_t FetchAddControl(
        volatile int64_t *address, int64_t value
    )
    {
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }

    static void StorePayloadWord(
        volatile uint64_t *address, uint64_t value
    )
    {
        *address = value;
    }

    static void FlushRegion(const void *, size_t)
    {
        std::atomic_thread_fence(std::memory_order_release);
    }

    static void InvalidateRegion(const void *, size_t)
    {
        std::atomic_thread_fence(std::memory_order_acquire);
    }

    static void StoreBarrier()
    {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
};

struct PlanFixture {
    RuntimePlanControl control{};
    std::unique_ptr<RuntimeTaskPlanCell[]> cells;
    RuntimePlanStorageRef storage{};
    RuntimePlanView view{};
    uint32_t capacity;

    explicit PlanFixture(uint32_t requested_capacity)
        : cells(new RuntimeTaskPlanCell[requested_capacity]()),
          capacity(requested_capacity)
    {
        control.planned_frontier.value = kTaskId;
        control.closed_task_count.value = kPlanOpenTaskCount;
        control.build_next.value = 0;
        control.build_workers_done.value = 0;
        control.build_release.value = kBuildReleasePending;
        control.fatal.value = 0;
        for (uint32_t cell = 0U; cell < capacity; ++cell) {
            for (uint32_t word = 0U;
                 word < kMaxPlanPayloadWords; ++word) {
                cells[cell].payload.words[word] = kPayloadPoison;
            }
        }
        storage.cells_base = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(cells.get())
        );
        storage.cells_bytes =
            static_cast<uint64_t>(capacity) * sizeof(RuntimeTaskPlanCell);
        storage.capacity = capacity;
        storage.cell_bytes = sizeof(RuntimeTaskPlanCell);
        storage.abi_version = kRuntimePlanAbiVersion;
        Expect(
            MakeRuntimePlanView(&control, storage, view),
            "failed to construct validated runtime Plan view"
        );
    }
};

struct SourceTask {
    TaskArgs args{};
    TensorDesc local_input{};
    TensorDesc gm_no_dependency{};
    TensorCreateInfo fresh_output{};
    std::array<uint64_t, 3U> explicit_dependencies{};

    SourceTask()
    {
        // 故意污染 C++ 对象 padding；wire 必须逐字段编码，不能把这些字节
        // 或任何栈地址原样搬进共享 PlanCell。
        std::memset(&local_input, 0xA5, sizeof(local_input));
        std::memset(&gm_no_dependency, 0xB6, sizeof(gm_no_dependency));
        std::memset(&fresh_output, 0xC7, sizeof(fresh_output));

        const uint32_t input_shape[kMaxTensorDims] = {2U, 3U, 4U, 0U, 0U};
        InitExternalTensor(
            local_input, UINT64_C(0x220000000), input_shape, 3U,
            DataType::Float16, true
        );
        local_input.owner_task_id = 5U;
        local_input.start_offset = 16U;
        local_input.version = 7;

        const uint32_t no_dep_shape[kMaxTensorDims] = {
            8U, 4U, 0U, 0U, 0U,
        };
        InitExternalTensor(
            gm_no_dependency, UINT64_C(0x330000000), no_dep_shape,
            2U, DataType::Float32, false
        );
        gm_no_dependency.owner_task_id = 11U;
        gm_no_dependency.version = 3;

        const uint32_t output_shape[kMaxTensorDims] = {
            4U, 5U, 6U, 0U, 0U,
        };
        InitCreateInfo(
            fresh_output, output_shape, 3U, DataType::Bfloat16
        );
        fresh_output.initial_value = UINT64_C(0x3F800000);
        fresh_output.has_initial_value = true;
        fresh_output.start_offset = 32U;
        fresh_output.version = 9;
        fresh_output.manual_dep = true;
        fresh_output.child_memory = 1U;

        ConstructTaskArgs(args);
        AddLocalTensor(args, local_input, TensorArgType::Input);
        AddOutput(args, fresh_output);
        AddOutputHandleTensor(
            args,
            FdwicOutputRef{7, 1, 1U, 1U, 32U, 4U},
            TensorArgType::Inout
        );
        AddOutputHandleTensor(
            args,
            FdwicOutputRef{2, 3, 0U, 0U, 0U, 0U},
            TensorArgType::OutputExisting
        );
        AddGmTensor(
            args, gm_no_dependency, TensorArgType::NoDependency
        );
        AddScalar(args, UINT64_C(0x0123456789ABCDEF));
        AddScalar(args, UINT64_C(0xFEDCBA9876543210));
        AddScalar(args, UINT64_C(0x000000003F800000));
        args.launch_spec.core_num = 8;
        args.launch_spec.require_sync_start = true;

        explicit_dependencies = {
            PtoTaskIdRaw(0U, 3U),
            PtoTaskIdRaw(2U, 0x12345678U),
            PtoTaskIdRaw(7U, 16U),
        };
        args.explicit_deps = static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(explicit_dependencies.data())
        );
        args.explicit_dep_count = explicit_dependencies.size();
        Expect(!args.has_error, "valid PA source construction failed");
        Expect(args.tensor_count == 5, "valid source must cover five tags");
    }

    SourceTask(const SourceTask &) = delete;
    SourceTask &operator=(const SourceTask &) = delete;
};

bool TensorDescFieldsEqual(
    const TensorDesc &actual, const TensorDesc &expected
)
{
    if (actual.buffer_addr != expected.buffer_addr ||
        actual.buffer_size != expected.buffer_size ||
        actual.owner_task_id != expected.owner_task_id ||
        actual.start_offset != expected.start_offset ||
        actual.version != expected.version ||
        actual.ndims != expected.ndims ||
        actual.dtype != expected.dtype ||
        actual.manual_dep != expected.manual_dep ||
        actual.is_contiguous != expected.is_contiguous ||
        actual.child_memory != expected.child_memory ||
        actual.extent_elem_cache != expected.extent_elem_cache) {
        return false;
    }
    for (uint32_t dim = 0U; dim < kMaxTensorDims; ++dim) {
        if (actual.shapes[dim] != expected.shapes[dim] ||
            actual.strides[dim] != expected.strides[dim]) {
            return false;
        }
    }
    return true;
}

bool CreateInfoFieldsEqual(
    const TensorCreateInfo &actual, const TensorCreateInfo &expected
)
{
    if (actual.initial_value != expected.initial_value ||
        actual.has_initial_value != expected.has_initial_value ||
        actual.reserved0 != expected.reserved0 ||
        actual.start_offset != expected.start_offset ||
        actual.version != expected.version ||
        actual.ndims != expected.ndims ||
        actual.dtype != expected.dtype ||
        actual.manual_dep != expected.manual_dep ||
        actual.is_contiguous != expected.is_contiguous ||
        actual.child_memory != expected.child_memory) {
        return false;
    }
    for (uint32_t dim = 0U; dim < kMaxTensorDims; ++dim) {
        if (actual.shapes[dim] != expected.shapes[dim]) return false;
    }
    return true;
}

bool OutputRefsEqual(
    const FdwicOutputRef &actual, const FdwicOutputRef &expected
)
{
    return actual.producer_task_id == expected.producer_task_id &&
           actual.output_slot == expected.output_slot &&
           actual.flags == expected.flags &&
           actual.view_ndims == expected.view_ndims &&
           actual.view_shape0 == expected.view_shape0 &&
           actual.view_offset0 == expected.view_offset0;
}

void ExpectCanonicalPaddingZero(
    const RuntimeTaskPlanCell &cell,
    const RuntimeTaskPlanHeader &header,
    const RuntimeTaskPlanLayout &layout
)
{
    for (uint32_t tensor = 0U; tensor < header.tensor_count; ++tensor) {
        uint32_t offset = 0U;
        Expect(
            RuntimeTaskPlanTensorWordOffset(header, tensor, offset),
            "failed to locate canonical tensor slot"
        );
        const TensorTag tag =
            static_cast<TensorTag>(header.tensor_tags[tensor]);
        const bool reference =
            (header.tensor_reference_mask & (uint32_t{1} << tensor)) != 0U;
        const uint32_t meaningful = TensorMeaningfulWords(tag, reference);
        for (uint32_t word = meaningful;
             word < kTensorCanonicalWords; ++word) {
            Expect(
                cell.payload.words[offset + word] == 0U,
                "unused canonical tensor words retained source bytes"
            );
        }
        if (!reference && tag != TensorTag::Output) {
            // TensorDesc 的真实字段止于 word11；wire 保留16-word固定槽，
            // adapter 必须把原对象的36B padding canonicalize 为零。
            for (uint32_t word = 12U;
                 word < kTensorCanonicalWords; ++word) {
                Expect(
                    cell.payload.words[offset + word] == 0U,
                    "TensorDesc object padding leaked into Plan payload"
                );
            }
        }
    }
    const uint32_t flushed_words =
        layout.payload_lines * kPlanCacheLineBytes / sizeof(uint64_t);
    for (uint32_t word = layout.payload_words;
         word < flushed_words; ++word) {
        Expect(
            cell.payload.words[word] == 0U,
            "final published cache-line padding was not zeroed"
        );
    }
    for (uint32_t tensor = header.tensor_count;
         tensor < aicpu_plan::kMaxTaskTensors; ++tensor) {
        Expect(
            header.tensor_tags[tensor] == 0U,
            "inactive header tensor tag was not zero"
        );
    }
}

void ExpectNoSourcePointers(
    const RuntimeTaskPlanCell &cell,
    const RuntimeTaskPlanLayout &layout,
    const SourceTask &source
)
{
    const std::array<uint64_t, 5U> forbidden = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&source.args)),
        static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(&source.local_input)
        ),
        static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(&source.gm_no_dependency)
        ),
        static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(&source.fresh_output)
        ),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
            source.explicit_dependencies.data()
        )),
    };
    for (uint32_t word = 0U; word < layout.payload_words; ++word) {
        for (uint64_t address : forbidden) {
            Expect(
                cell.payload.words[word] != address,
                "Plan payload retained a producer-local source pointer"
            );
        }
    }
}

void TestFullPaAdapterRoundTrip()
{
    SourceTask source;
    RuntimeTaskPlanSpec spec{};
    Expect(
        MakePaRuntimeTaskPlanSpec(
            source.args, kTaskId, static_cast<int32_t>(kFunctionId),
            EngineClass::Aic, kValidAdapterFlags, spec
        ),
        "MakePaRuntimeTaskPlanSpec rejected a valid PA task"
    );
    PlanFixture fixture(32U);
    const PaRuntimeTaskPlanSource wire_source{source.args};
    Expect(
        PublishRuntimeTaskPlan<CpuOps>(
            fixture.view, spec, wire_source
        ) == PlanPublishResult::Published,
        "failed to publish valid PA PlanCell"
    );
    Expect(
        AdvancePlannedFrontier<CpuOps>(fixture.view, kTaskId),
        "failed to advance published Plan frontier"
    );

    RuntimeTaskPlanHeader header{};
    RuntimeTaskPlanLayout layout{};
    Expect(
        AcquireRuntimeTaskPlan<CpuOps>(
            fixture.view, kTaskId, header, layout
        ) == PlanAcquireResult::Acquired,
        "failed to acquire valid PA PlanCell"
    );
    Expect(header.task_id == kTaskId, "task id changed across Plan wire");
    Expect(
        header.function_id == kFunctionId,
        "function id changed across Plan wire"
    );
    Expect(
        header.engine_class == static_cast<uint8_t>(EngineClass::Aic),
        "engine class changed across Plan wire"
    );
    Expect(
        header.adapter_flags == kValidAdapterFlags,
        "PA adapter flags changed across Plan wire"
    );
    Expect(header.tensor_count == 5U, "tensor count changed across Plan wire");
    Expect(header.scalar_count == 3U, "scalar count changed across Plan wire");
    Expect(
        header.explicit_dep_count == source.explicit_dependencies.size(),
        "explicit dependency count changed across Plan wire"
    );
    Expect(header.output_count == 1U, "output count changed across Plan wire");
    Expect(header.core_num == 8, "core count changed across Plan wire");
    Expect(header.require_sync_start == 1U, "sync-start flag changed across Plan wire");

    constexpr std::array<TensorArgType, 5U> kExpectedTags = {
        TensorArgType::Input,
        TensorArgType::Output,
        TensorArgType::Inout,
        TensorArgType::OutputExisting,
        TensorArgType::NoDependency,
    };
    for (uint32_t tensor = 0U; tensor < kExpectedTags.size(); ++tensor) {
        Expect(
            header.tensor_tags[tensor] ==
                static_cast<uint8_t>(kExpectedTags[tensor]),
            "one of the five PA tensor tags changed across Plan wire"
        );
    }
    Expect(
        header.tensor_reference_mask ==
            ((uint32_t{1} << 2U) | (uint32_t{1} << 3U)),
        "plain/view output-reference mask changed across Plan wire"
    );

    TaskArgs decoded{};
    PaRuntimeTaskDecodeScratch scratch{};
    Expect(
        DecodePaRuntimeTaskPlan(
            fixture.cells[kTaskId], header, layout, decoded, scratch
        ),
        "failed to decode acquired PA PlanCell"
    );
    Expect(decoded.tensor_count == source.args.tensor_count, "decoded tensor count mismatch");
    Expect(decoded.scalar_count == source.args.scalar_count, "decoded scalar count mismatch");
    Expect(decoded.launch_spec.core_num == 8, "decoded core count mismatch");
    Expect(decoded.launch_spec.require_sync_start, "decoded sync-start flag mismatch");
    for (uint32_t tensor = 0U; tensor < kExpectedTags.size(); ++tensor) {
        Expect(
            TaskTag(decoded, tensor) == kExpectedTags[tensor],
            "decoded PA tensor tag mismatch"
        );
    }
    Expect(
        decoded.tensors[0].kind == TensorRefKind::GmTensor &&
            decoded.tensors[0].pointer.gm_tensor != &source.local_input &&
            TensorDescFieldsEqual(
                *decoded.tensors[0].pointer.gm_tensor,
                source.local_input
            ),
        "local Input TensorDesc was not canonicalized and decoded"
    );
    Expect(
        decoded.tensors[1].kind == TensorRefKind::CreateInfo &&
            decoded.tensors[1].pointer.create_info != &source.fresh_output &&
            CreateInfoFieldsEqual(
                *decoded.tensors[1].pointer.create_info,
                source.fresh_output
            ),
        "Output CreateInfo was not copied and decoded"
    );
    Expect(
        decoded.tensors[2].kind == TensorRefKind::SharedOutputRef &&
            OutputRefsEqual(
                decoded.tensors[2].pointer.output_ref,
                source.args.tensors[2].pointer.output_ref
            ),
        "one-dimensional output view changed across Plan wire"
    );
    Expect(
        decoded.tensors[3].kind == TensorRefKind::SharedOutputRef &&
            OutputRefsEqual(
                decoded.tensors[3].pointer.output_ref,
                source.args.tensors[3].pointer.output_ref
            ),
        "plain output reference changed across Plan wire"
    );
    Expect(
        decoded.tensors[4].kind == TensorRefKind::GmTensor &&
            decoded.tensors[4].pointer.gm_tensor !=
                &source.gm_no_dependency &&
            TensorDescFieldsEqual(
                *decoded.tensors[4].pointer.gm_tensor,
                source.gm_no_dependency
            ),
        "NoDependency GM TensorDesc was not canonicalized and decoded"
    );
    for (uint32_t scalar = 0U;
         scalar < static_cast<uint32_t>(source.args.scalar_count); ++scalar) {
        Expect(
            decoded.scalars[scalar] == source.args.scalars[scalar],
            "scalar changed across Plan wire"
        );
    }
    Expect(
        decoded.explicit_dep_count == source.explicit_dependencies.size(),
        "decoded explicit dependency count mismatch"
    );
    Expect(
        decoded.explicit_deps != source.args.explicit_deps,
        "decoded explicit dependencies retained producer pointer"
    );
    const uint64_t *decoded_dependencies =
        reinterpret_cast<const uint64_t *>(
            static_cast<uintptr_t>(decoded.explicit_deps)
        );
    for (uint32_t dependency = 0U;
         dependency < source.explicit_dependencies.size(); ++dependency) {
        Expect(
            decoded_dependencies[dependency] ==
                source.explicit_dependencies[dependency],
            "PTO2TaskId raw dependency changed across Plan wire"
        );
    }

    for (uint8_t byte : decoded.tensors[0].pointer.gm_tensor->padding) {
        Expect(byte == 0U, "decoded TensorDesc padding is not canonical zero");
    }
    for (uint8_t byte : decoded.tensors[1].pointer.create_info->padding0) {
        Expect(byte == 0U, "decoded CreateInfo padding is not canonical zero");
    }
    ExpectCanonicalPaddingZero(
        fixture.cells[kTaskId], header, layout
    );
    ExpectNoSourcePointers(
        fixture.cells[kTaskId], layout, source
    );
}

void ExpectMakeSpecRejected(
    const TaskArgs &args, int32_t function_id,
    EngineClass engine_class, uint8_t adapter_flags,
    const char *message
)
{
    RuntimeTaskPlanSpec spec{};
    Expect(
        !MakePaRuntimeTaskPlanSpec(
            args, kTaskId, function_id, engine_class,
            adapter_flags, spec
        ),
        message
    );
}

void ExpectPipelineRejected(
    const TaskArgs &args, int32_t function_id,
    EngineClass engine_class, uint8_t adapter_flags,
    const char *message
)
{
    RuntimeTaskPlanSpec spec{};
    if (!MakePaRuntimeTaskPlanSpec(
            args, kTaskId, function_id, engine_class,
            adapter_flags, spec
        )) {
        return;
    }
    PlanFixture fixture(32U);
    const PaRuntimeTaskPlanSource source{args};
    Expect(
        PublishRuntimeTaskPlan<CpuOps>(fixture.view, spec, source) !=
            PlanPublishResult::Published,
        message
    );
}

void TestInvalidPaSourcesFailClosed()
{
    {
        SourceTask source;
        source.args.tensors[2].pointer.output_ref.producer_task_id =
            static_cast<int32_t>(kTaskId);
        ExpectMakeSpecRejected(
            source.args, static_cast<int32_t>(kFunctionId),
            EngineClass::Aic, kValidAdapterFlags,
            "future output reference was accepted"
        );
    }
    {
        SourceTask source;
        source.args.tensors[2].pointer.output_ref.view_ndims = 2U;
        ExpectMakeSpecRejected(
            source.args, static_cast<int32_t>(kFunctionId),
            EngineClass::Aic, kValidAdapterFlags,
            "unsupported multi-dimensional output view was accepted"
        );
    }
    {
        SourceTask source;
        FdwicOutputRef &view =
            source.args.tensors[2].pointer.output_ref;
        view.view_shape0 = 16U;
        view.view_offset0 = UINT32_MAX - 7U;
        ExpectPipelineRejected(
            source.args, static_cast<int32_t>(kFunctionId),
            EngineClass::Aic, kValidAdapterFlags,
            "overflowing one-dimensional output view was accepted"
        );
    }
    {
        SourceTask source;
        source.args.tensors[3].pointer.output_ref.output_slot = 8;
        ExpectMakeSpecRejected(
            source.args, static_cast<int32_t>(kFunctionId),
            EngineClass::Aic, kValidAdapterFlags,
            "output reference slot >= 8 was accepted"
        );
    }
    {
        SourceTask source;
        source.local_input.dtype = DataType::Count;
        ExpectMakeSpecRejected(
            source.args, static_cast<int32_t>(kFunctionId),
            EngineClass::Aic, kValidAdapterFlags,
            "invalid TensorDesc dtype was accepted"
        );
    }
    {
        SourceTask source;
        source.fresh_output.dtype = DataType::Count;
        ExpectMakeSpecRejected(
            source.args, static_cast<int32_t>(kFunctionId),
            EngineClass::Aic, kValidAdapterFlags,
            "invalid CreateInfo dtype was accepted"
        );
    }
    {
        SourceTask source;
        source.gm_no_dependency.ndims = kMaxTensorDims + 1U;
        ExpectMakeSpecRejected(
            source.args, static_cast<int32_t>(kFunctionId),
            EngineClass::Aic, kValidAdapterFlags,
            "TensorDesc rank beyond ABI bound was accepted"
        );
    }
    {
        SourceTask source;
        source.fresh_output.ndims = 0U;
        ExpectMakeSpecRejected(
            source.args, static_cast<int32_t>(kFunctionId),
            EngineClass::Aic, kValidAdapterFlags,
            "zero-rank CreateInfo was accepted"
        );
    }
    {
        SourceTask source;
        source.args.tags[0] = INT32_MAX;
        ExpectMakeSpecRejected(
            source.args, static_cast<int32_t>(kFunctionId),
            EngineClass::Aic, kValidAdapterFlags,
            "unknown PA tensor tag was accepted"
        );
    }
}

void TestInvalidMetadataFailsClosed()
{
    SourceTask source;
    ExpectMakeSpecRejected(
        source.args, static_cast<int32_t>(kFunctionId),
        EngineClass::MetadataOnly, kValidAdapterFlags,
        "metadata-only task with executable function id was accepted"
    );
    ExpectMakeSpecRejected(
        source.args, -1, EngineClass::Aic,
        kValidAdapterFlags,
        "AIC task without executable function id was accepted"
    );
    ExpectMakeSpecRejected(
        source.args, -1, EngineClass::Aiv,
        kValidAdapterFlags,
        "AIV task without executable function id was accepted"
    );

    // PA adapter 必须同时约束 kind 与执行引擎：QK/PV 走 AIC，SF/UP
    // 走 AIV，Alloc 是 MetadataOnly。不能只验证 function_id 的正负。
    ExpectMakeSpecRejected(
        source.args, static_cast<int32_t>(kFunctionId),
        EngineClass::Aiv, kValidAdapterFlags,
        "PV task was accepted on the AIV engine"
    );
    const uint8_t sf_flags = static_cast<uint8_t>(
        kSharedPaTicketMetaPresent |
        (2U << kSharedPaTicketGroupShift) |
        static_cast<uint32_t>(TaskKind::Sf)
    );
    ExpectMakeSpecRejected(
        source.args, static_cast<int32_t>(kFunctionId),
        EngineClass::Aic, sf_flags,
        "SF task was accepted on the AIC engine"
    );
    const uint8_t alloc_flags = static_cast<uint8_t>(
        kSharedPaTicketMetaPresent |
        static_cast<uint32_t>(TaskKind::Alloc)
    );
    ExpectMakeSpecRejected(
        source.args, static_cast<int32_t>(kFunctionId),
        EngineClass::Aic, alloc_flags,
        "Alloc task was accepted as an executable AIC task"
    );
    ExpectMakeSpecRejected(
        source.args, -1, EngineClass::MetadataOnly,
        kValidAdapterFlags,
        "PV task was accepted as metadata-only"
    );
    ExpectPipelineRejected(
        source.args, static_cast<int32_t>(kFunctionId),
        static_cast<EngineClass>(UINT8_MAX), kValidAdapterFlags,
        "unknown engine class was accepted"
    );

    // present bit 缺失，或把 has-following 施加到非 UP task，均不属于
    // PA adapter 的合法业务 byte；公共协议不得替 adapter 猜测其含义。
    ExpectMakeSpecRejected(
        source.args, static_cast<int32_t>(kFunctionId),
        EngineClass::Aic, 0U,
        "PA adapter flag without meta-present bit was accepted"
    );
    const uint8_t invalid_qk_following = static_cast<uint8_t>(
        kSharedPaTicketMetaPresent |
        kSharedPaTicketHasFollowing |
        static_cast<uint32_t>(TaskKind::Qk)
    );
    ExpectMakeSpecRejected(
        source.args, static_cast<int32_t>(kFunctionId),
        EngineClass::Aic, invalid_qk_following,
        "PA adapter flag encoded has-following on a non-UP task"
    );
}

}  // namespace

int main()
{
    TestFullPaAdapterRoundTrip();
    TestInvalidPaSourcesFailClosed();
    TestInvalidMetadataFailsClosed();
    std::puts("[PASS] PA TaskArgs <-> canonical AICPU Plan adapter gate");
    return 0;
}
