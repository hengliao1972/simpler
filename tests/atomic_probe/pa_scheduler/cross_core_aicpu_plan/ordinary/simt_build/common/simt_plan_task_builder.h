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

#ifndef PA_SCHEDULER_AICPU_PLAN_ORDINARY_SIMT_PLAN_TASK_BUILDER_H
#define PA_SCHEDULER_AICPU_PLAN_ORDINARY_SIMT_PLAN_TASK_BUILDER_H

#ifndef PA_DEVICE
#error "PA_DEVICE must name a __simt_callee__ compatible function identity"
#endif
#ifndef PA_GM
#error "PA_GM must name the CCEC GM address space"
#endif

#include "simt_plan_build_protocol.h"
#include "../../scalar_build/common/shared_exec_protocol.h"

#include <cstddef>
#include <cstdint>

namespace pa_scheduler::aicpu_plan_simt {

namespace plan = pa_scheduler::aicpu_plan;
namespace exec = pa_scheduler::cross_core;

constexpr uint32_t kSimtTensorDims = 5U;
constexpr uint32_t kSimtDataTypeCount = 12U;
constexpr uint32_t kSimtOutputAlignment = 1024U;
constexpr uint64_t kSimtInvalidTaskOwner = UINT64_MAX;
constexpr uint32_t kSimtWriterHistoryMagic = 0x57484953U;  // "WHIS"

// 这里不是 TaskArgs 的第二份 pointer union。每个 leader 只把 canonical
// Plan wire 解码成值语义 scratch；任何 descriptor 地址都在最终 Pack 时
// 重新解析，不能把 AICPU callback 栈或 Scalar 私有地址带进 VF。
struct SimtCanonicalTensorDesc {
    uint64_t buffer_addr;
    uint64_t buffer_size;
    uint64_t owner_task_id;
    uint64_t start_offset;
    int32_t version;
    uint32_t ndims;
    uint8_t dtype;
    uint8_t manual_dep;
    uint8_t is_contiguous;
    uint8_t child_memory;
    uint32_t shapes[kSimtTensorDims];
    uint64_t extent_elem_cache;
    uint32_t strides[kSimtTensorDims];
    uint8_t padding[36];
};

struct SimtCanonicalCreateInfo {
    uint64_t initial_value;
    uint8_t has_initial_value;
    uint8_t padding0[7];
    uint64_t reserved0;
    uint64_t start_offset;
    int32_t version;
    uint32_t ndims;
    uint8_t dtype;
    uint8_t manual_dep;
    uint8_t is_contiguous;
    uint8_t child_memory;
    uint32_t shapes[kSimtTensorDims];
};

struct SimtWriterRegion {
    uint64_t buffer_addr;
    uint64_t lo;
    uint64_t hi;
    int32_t producer;
    uint32_t reserved;
};

struct SimtWriterHistoryRecord {
    uint32_t symbol_key;
    int32_t previous_writer;
};

struct alignas(64) SimtWriterHistoryCell {
    uint32_t magic;
    int32_t writer_task;
    uint32_t count;
    uint32_t reserved;
    SimtWriterHistoryRecord entries[plan::kMaxTaskTensors];
    uint8_t padding[48];
};

static_assert(sizeof(SimtCanonicalTensorDesc) == 128U);
static_assert(offsetof(SimtCanonicalTensorDesc, shapes) == 44U);
static_assert(offsetof(SimtCanonicalTensorDesc, extent_elem_cache) == 64U);
static_assert(sizeof(SimtCanonicalCreateInfo) == 64U);
static_assert(sizeof(SimtWriterRegion) == 32U);
static_assert(alignof(SimtWriterRegion) == alignof(uint64_t));
static_assert(offsetof(SimtWriterRegion, buffer_addr) == 0U);
static_assert(offsetof(SimtWriterRegion, lo) == 8U);
static_assert(offsetof(SimtWriterRegion, hi) == 16U);
static_assert(offsetof(SimtWriterRegion, producer) == 24U);
static_assert(offsetof(SimtWriterRegion, reserved) == 28U);
static_assert(sizeof(SimtWriterHistoryRecord) == 8U);
static_assert(alignof(SimtWriterHistoryRecord) == alignof(uint32_t));
static_assert(offsetof(SimtWriterHistoryRecord, symbol_key) == 0U);
static_assert(offsetof(SimtWriterHistoryRecord, previous_writer) == 4U);
static_assert(sizeof(SimtWriterHistoryCell) == 320U);
static_assert(alignof(SimtWriterHistoryCell) == 64U);
static_assert(offsetof(SimtWriterHistoryCell, magic) == 0U);
static_assert(offsetof(SimtWriterHistoryCell, writer_task) == 4U);
static_assert(offsetof(SimtWriterHistoryCell, count) == 8U);
static_assert(offsetof(SimtWriterHistoryCell, reserved) == 12U);
static_assert(offsetof(SimtWriterHistoryCell, entries) == 16U);

enum class SimtTaskBuildStatus : uint8_t {
    Published,
    InvalidPlanControl,
    InvalidPlanPayload,
    UnsupportedOutputView,
    InvalidAdapterRoute,
    MaterializeFailed,
    InsertWaitFailed,
    WriterPublishFailed,
    FaninFailed,
    MetadataCompletionFailed,
    ExecPublishFailed,
};

struct SimtTaskBuildScratch {
    plan::RuntimeTaskPlanHeader header;
    plan::RuntimeTaskPlanLayout layout;
    SimtCanonicalTensorDesc tensors[plan::kMaxTaskTensors];
    SimtCanonicalCreateInfo create_infos[
        plan::kMaxRuntimeOutputsPerTask
    ];
    plan::RuntimeOutputReferenceWire references[
        plan::kMaxTaskTensors
    ];
    uint64_t scalars[plan::kMaxTaskScalars];
    int32_t explicit_dependencies[
        plan::kMaxExplicitDependencies
    ];
    int32_t fanin[exec::kExecMaxFanin];
    SimtWriterRegion ordinary_entries[plan::kMaxTaskTensors];
    uint16_t ordinary_buckets[plan::kMaxTaskTensors];
    uint8_t ordinary_bucket_ordinals[plan::kMaxTaskTensors];
    uint32_t symbol_keys[plan::kMaxTaskTensors];
    int32_t symbol_previous[plan::kMaxTaskTensors];
    uint8_t output_tensor_indices[
        plan::kMaxRuntimeOutputsPerTask
    ];
    uint32_t output_mask;
    uint32_t reference_mask;
    uint32_t output_count;
    uint32_t reserved_output_count;
    uint32_t published_output_count;
    uint32_t ordinary_count;
    uint32_t symbol_count;
    uint32_t fanin_count;
    uint64_t completion_vend;
    bool insert_completion_published;
};

PA_DEVICE uint64_t SimtPackU32Pair(uint32_t low, uint32_t high)
{
    return static_cast<uint64_t>(low) |
           (static_cast<uint64_t>(high) << 32U);
}

// Shared writer-history key ABI is one-based: zero is permanently reserved
// as the invalid/unset key.  Keep the narrow SIMT producer and reader on the
// same encoding used by SharedSymbolHistoryKey/SharedSymbolHistoryReference.
PA_DEVICE bool SimtEncodeSharedSymbolHistoryKey(
    uint32_t producer, uint32_t slot, uint32_t &key
)
{
    if (slot >= plan::kMaxRuntimeOutputsPerTask) return false;
    const uint64_t key64 =
        static_cast<uint64_t>(producer) *
            plan::kMaxRuntimeOutputsPerTask +
        slot + 1U;
    if (key64 > UINT32_MAX) return false;
    key = static_cast<uint32_t>(key64);
    return true;
}

PA_DEVICE bool SimtDecodeSharedSymbolHistoryKey(
    uint32_t key, uint32_t &producer, uint32_t &slot
)
{
    if (key == 0U) return false;
    const uint32_t zero_based_key = key - 1U;
    producer = zero_based_key / plan::kMaxRuntimeOutputsPerTask;
    slot = zero_based_key % plan::kMaxRuntimeOutputsPerTask;
    return true;
}

PA_DEVICE uint64_t SimtElementSize(uint8_t dtype)
{
    constexpr uint8_t sizes[kSimtDataTypeCount] = {
        4U, 2U, 4U, 2U, 1U, 1U, 2U, 8U, 8U, 2U, 4U, 1U,
    };
    return dtype < kSimtDataTypeCount ? sizes[dtype] : 0U;
}

PA_DEVICE uint64_t SimtTensorDescWord(
    const SimtCanonicalTensorDesc &tensor, uint32_t word
)
{
    switch (word) {
        case 0U: return tensor.buffer_addr;
        case 1U: return tensor.buffer_size;
        case 2U: return tensor.owner_task_id;
        case 3U: return tensor.start_offset;
        case 4U:
            return SimtPackU32Pair(
                static_cast<uint32_t>(tensor.version), tensor.ndims
            );
        case 5U:
            return static_cast<uint64_t>(tensor.dtype) |
                   (static_cast<uint64_t>(tensor.manual_dep) << 8U) |
                   (static_cast<uint64_t>(tensor.is_contiguous) << 16U) |
                   (static_cast<uint64_t>(tensor.child_memory) << 24U) |
                   (static_cast<uint64_t>(tensor.shapes[0]) << 32U);
        case 6U:
            return SimtPackU32Pair(tensor.shapes[1], tensor.shapes[2]);
        case 7U:
            return SimtPackU32Pair(tensor.shapes[3], tensor.shapes[4]);
        case 8U: return tensor.extent_elem_cache;
        case 9U:
            return SimtPackU32Pair(tensor.strides[0], tensor.strides[1]);
        case 10U:
            return SimtPackU32Pair(tensor.strides[2], tensor.strides[3]);
        case 11U: return tensor.strides[4];
        default: return 0U;
    }
}

PA_DEVICE void SimtDecodeTensorDesc(
    PA_GM const volatile uint64_t *words,
    SimtCanonicalTensorDesc &tensor
)
{
    tensor.buffer_addr = words[0];
    tensor.buffer_size = words[1];
    tensor.owner_task_id = words[2];
    tensor.start_offset = words[3];
    const uint64_t word4 = words[4];
    const uint64_t word5 = words[5];
    tensor.version = plan::DecodeRuntimeWireInt32(
        static_cast<uint32_t>(word4)
    );
    tensor.ndims = static_cast<uint32_t>(word4 >> 32U);
    tensor.dtype = static_cast<uint8_t>(word5);
    tensor.manual_dep = static_cast<uint8_t>(word5 >> 8U);
    tensor.is_contiguous = static_cast<uint8_t>(word5 >> 16U);
    tensor.child_memory = static_cast<uint8_t>(word5 >> 24U);
    tensor.shapes[0] = static_cast<uint32_t>(word5 >> 32U);
    const uint64_t word6 = words[6];
    const uint64_t word7 = words[7];
    tensor.shapes[1] = static_cast<uint32_t>(word6);
    tensor.shapes[2] = static_cast<uint32_t>(word6 >> 32U);
    tensor.shapes[3] = static_cast<uint32_t>(word7);
    tensor.shapes[4] = static_cast<uint32_t>(word7 >> 32U);
    tensor.extent_elem_cache = words[8];
    const uint64_t word9 = words[9];
    const uint64_t word10 = words[10];
    tensor.strides[0] = static_cast<uint32_t>(word9);
    tensor.strides[1] = static_cast<uint32_t>(word9 >> 32U);
    tensor.strides[2] = static_cast<uint32_t>(word10);
    tensor.strides[3] = static_cast<uint32_t>(word10 >> 32U);
    tensor.strides[4] = static_cast<uint32_t>(words[11]);
}

PA_DEVICE void SimtDecodeCreateInfo(
    PA_GM const volatile uint64_t *words,
    SimtCanonicalCreateInfo &info
)
{
    info.initial_value = words[0];
    info.has_initial_value = static_cast<uint8_t>(words[1]);
    info.reserved0 = words[2];
    info.start_offset = words[3];
    const uint64_t word4 = words[4];
    const uint64_t word5 = words[5];
    info.version = plan::DecodeRuntimeWireInt32(
        static_cast<uint32_t>(word4)
    );
    info.ndims = static_cast<uint32_t>(word4 >> 32U);
    info.dtype = static_cast<uint8_t>(word5);
    info.manual_dep = static_cast<uint8_t>(word5 >> 8U);
    info.is_contiguous = static_cast<uint8_t>(word5 >> 16U);
    info.child_memory = static_cast<uint8_t>(word5 >> 24U);
    info.shapes[0] = static_cast<uint32_t>(word5 >> 32U);
    const uint64_t word6 = words[6];
    const uint64_t word7 = words[7];
    info.shapes[1] = static_cast<uint32_t>(word6);
    info.shapes[2] = static_cast<uint32_t>(word6 >> 32U);
    info.shapes[3] = static_cast<uint32_t>(word7);
    info.shapes[4] = static_cast<uint32_t>(word7 >> 32U);
}

PA_DEVICE bool SimtCheckedMultiplyU64ByU32(
    uint64_t left, uint32_t right, uint64_t &product
)
{
    const uint64_t low_product =
        static_cast<uint64_t>(static_cast<uint32_t>(left)) * right;
    const uint64_t high_product =
        static_cast<uint64_t>(static_cast<uint32_t>(left >> 32U)) *
        right;
    const uint64_t carry = low_product >> 32U;
    if (high_product > UINT32_MAX - carry) return false;
    product = ((high_product + carry) << 32U) |
              static_cast<uint32_t>(low_product);
    return true;
}

PA_DEVICE bool SimtValidateTensorDesc(
    const SimtCanonicalTensorDesc &tensor, uint32_t task_id
)
{
    if (tensor.buffer_addr == 0U || tensor.buffer_size == 0U ||
        tensor.ndims == 0U || tensor.ndims > kSimtTensorDims ||
        tensor.dtype >= kSimtDataTypeCount ||
        tensor.manual_dep > 1U || tensor.is_contiguous > 1U ||
        tensor.start_offset > tensor.buffer_size ||
        (tensor.owner_task_id != kSimtInvalidTaskOwner &&
         (tensor.owner_task_id > INT32_MAX ||
          tensor.owner_task_id >= task_id))) {
        return false;
    }
    for (uint32_t dimension = 0U;
         dimension < tensor.ndims; ++dimension) {
        if (tensor.shapes[dimension] == 0U) return false;
    }
    return true;
}

PA_DEVICE bool SimtCreateInfoBytes(
    const SimtCanonicalCreateInfo &info, uint64_t &bytes
)
{
    bytes = 0U;
    if (info.has_initial_value != 0U || info.reserved0 != 0U ||
        info.start_offset != 0U || info.ndims == 0U ||
        info.ndims > kSimtTensorDims ||
        info.dtype >= kSimtDataTypeCount ||
        info.manual_dep > 1U || info.is_contiguous != 1U ||
        info.child_memory != 0U) {
        return false;
    }
    uint32_t elements = 1U;
    for (uint32_t dimension = 0U;
         dimension < info.ndims; ++dimension) {
        const uint32_t extent = info.shapes[dimension];
        if (extent == 0U || elements > UINT32_MAX / extent) {
            return false;
        }
        elements *= extent;
    }
    const uint64_t element_size = SimtElementSize(info.dtype);
    bytes = static_cast<uint64_t>(elements) * element_size;
    return element_size != 0U && bytes != 0U;
}

PA_DEVICE bool SimtMakeMaterializedTensor(
    const SimtCanonicalCreateInfo &info, uint32_t task_id,
    uint64_t address, uint64_t bytes,
    SimtCanonicalTensorDesc &tensor
)
{
    if (address == 0U || bytes == 0U ||
        !SimtCreateInfoBytes(info, tensor.buffer_size) ||
        tensor.buffer_size != bytes) {
        return false;
    }
    tensor.buffer_addr = address;
    tensor.owner_task_id = task_id;
    tensor.start_offset = info.start_offset;
    tensor.version = info.version;
    tensor.ndims = info.ndims;
    tensor.dtype = info.dtype;
    tensor.manual_dep = info.manual_dep;
    tensor.is_contiguous = info.is_contiguous;
    tensor.child_memory = info.child_memory;
    uint32_t stride = 1U;
    for (uint32_t dimension = 0U;
         dimension < kSimtTensorDims; ++dimension) {
        tensor.shapes[dimension] = info.shapes[dimension];
        tensor.strides[dimension] = 0U;
    }
    for (int32_t dimension = static_cast<int32_t>(info.ndims) - 1;
         dimension >= 0; --dimension) {
        tensor.strides[static_cast<uint32_t>(dimension)] = stride;
        if (stride > UINT32_MAX /
                tensor.shapes[static_cast<uint32_t>(dimension)]) {
            return false;
        }
        stride *= tensor.shapes[static_cast<uint32_t>(dimension)];
    }
    tensor.extent_elem_cache = stride;
    return SimtValidateTensorDesc(tensor, task_id + 1U);
}

PA_DEVICE bool SimtMakeWriterRegion(
    const SimtCanonicalTensorDesc &tensor, uint32_t task_id,
    SimtWriterRegion &region
)
{
    if (!SimtValidateTensorDesc(tensor, task_id)) return false;
    uint64_t extent = tensor.extent_elem_cache;
    if (tensor.is_contiguous != 0U) {
        extent = 1U;
        for (uint32_t dimension = 0U;
             dimension < tensor.ndims; ++dimension) {
            uint64_t next_extent = 0U;
            if (!SimtCheckedMultiplyU64ByU32(
                    extent, tensor.shapes[dimension], next_extent
                )) {
                return false;
            }
            extent = next_extent;
        }
    }
    const uint64_t element_size = SimtElementSize(tensor.dtype);
    uint32_t shift = 0U;
    if (element_size == 8U) shift = 3U;
    else if (element_size == 4U) shift = 2U;
    else if (element_size == 2U) shift = 1U;
    else if (element_size != 1U) return false;
    if (extent == 0U || tensor.start_offset > UINT64_MAX - extent) {
        return false;
    }
    const uint64_t end_offset = tensor.start_offset + extent;
    if (tensor.start_offset > (UINT64_MAX >> shift) ||
        end_offset > (UINT64_MAX >> shift)) {
        return false;
    }
    region = SimtWriterRegion{
        tensor.buffer_addr,
        tensor.start_offset << shift,
        end_offset << shift,
        static_cast<int32_t>(task_id),
        0U,
    };
    return region.lo < region.hi;
}

PA_DEVICE uint64_t SimtAlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

template <typename Ops, typename Runtime>
PA_DEVICE bool SimtPublishFatal(
    Runtime &runtime, uint32_t task_id, SimtTaskBuildStatus status
)
{
    return runtime.PublishBuildFatal(
        task_id, static_cast<uint32_t>(status)
    );
}

template <typename Ops>
PA_DEVICE SimtTaskBuildStatus AcquireCanonicalPlanTask(
    const plan::RuntimePlanView &view, uint32_t task_id,
    SimtTaskBuildScratch &scratch
)
{
    scratch.output_mask = 0U;
    scratch.reference_mask = 0U;
    scratch.output_count = 0U;
    scratch.reserved_output_count = 0U;
    scratch.published_output_count = 0U;
    scratch.ordinary_count = 0U;
    scratch.symbol_count = 0U;
    scratch.fanin_count = 0U;
    scratch.completion_vend = 0U;
    scratch.insert_completion_published = false;
    if (view.control == nullptr || view.cells == nullptr ||
        task_id >= view.capacity) {
        return SimtTaskBuildStatus::InvalidPlanControl;
    }
    PA_GM plan::RuntimeTaskPlanCell &cell = view.cells[task_id];
    const int64_t first = Ops::LoadControl(&cell.control.value);
    const plan::DecodedPlanCellControl decoded =
        plan::DecodePlanCellControl(first);
    if (!decoded.valid ||
        decoded.phase != plan::PlanCellPhase::Published ||
        decoded.task_id != task_id) {
        return SimtTaskBuildStatus::InvalidPlanControl;
    }
    Ops::InvalidateRegion(
        &cell.payload,
        static_cast<uint64_t>(decoded.payload_lines) *
            plan::kPlanCacheLineBytes
    );
    if (Ops::LoadControl(&cell.control.value) != first ||
        !plan::ValidateRuntimeTaskPlanPayload(
            cell.payload, task_id, decoded.payload_lines,
            scratch.header, scratch.layout
        )) {
        return SimtTaskBuildStatus::InvalidPlanPayload;
    }

    uint32_t output = 0U;
    for (uint32_t tensor = 0U;
         tensor < scratch.header.tensor_count; ++tensor) {
        uint32_t word_offset = 0U;
        if (!plan::RuntimeTaskPlanTensorWordOffset(
                scratch.header, tensor, word_offset
            )) {
            return SimtTaskBuildStatus::InvalidPlanPayload;
        }
        PA_GM const volatile uint64_t *words =
            &cell.payload.words[word_offset];
        const plan::TensorTag tag = static_cast<plan::TensorTag>(
            scratch.header.tensor_tags[tensor]
        );
        const bool reference =
            (scratch.header.tensor_reference_mask &
             (uint32_t{1} << tensor)) != 0U;
        if (reference) {
            const plan::RuntimeOutputReferenceWire output_ref =
                plan::DecodeRuntimeOutputReferenceWire(
                    words[0], words[1]
                );
            if (!plan::RuntimeOutputReferenceWireValid(
                    output_ref, task_id
                ) || (tag != plan::TensorTag::Input &&
                       tag != plan::TensorTag::Inout &&
                       tag != plan::TensorTag::OutputExisting)) {
                return SimtTaskBuildStatus::InvalidPlanPayload;
            }
            // Plan ABI v2 保留 1D view，但当前 shared output resolver 没有
            // 可求证的 offset 单位合同。必须在任何 heap/metadata side
            // effect 前拒绝，而不是把 view 静默降级成 plain descriptor。
            if (output_ref.flags != 0U ||
                output_ref.view_ndims != 0U ||
                output_ref.view_shape0 != 0U ||
                output_ref.view_offset0 != 0U) {
                return SimtTaskBuildStatus::UnsupportedOutputView;
            }
            scratch.references[tensor] = output_ref;
            scratch.reference_mask |= uint32_t{1} << tensor;
            continue;
        }
        if (tag == plan::TensorTag::Output) {
            if (output >= plan::kMaxRuntimeOutputsPerTask) {
                return SimtTaskBuildStatus::InvalidPlanPayload;
            }
            SimtDecodeCreateInfo(words, scratch.create_infos[output]);
            uint64_t ignored_bytes = 0U;
            if (!SimtCreateInfoBytes(
                    scratch.create_infos[output], ignored_bytes
                )) {
                return SimtTaskBuildStatus::InvalidPlanPayload;
            }
            scratch.output_tensor_indices[output] =
                static_cast<uint8_t>(tensor);
            scratch.output_mask |= uint32_t{1} << tensor;
            ++output;
            continue;
        }
        SimtDecodeTensorDesc(words, scratch.tensors[tensor]);
        if (!SimtValidateTensorDesc(
                scratch.tensors[tensor], task_id
            )) {
            return SimtTaskBuildStatus::InvalidPlanPayload;
        }
    }
    if (output != scratch.header.output_count) {
        return SimtTaskBuildStatus::InvalidPlanPayload;
    }
    scratch.output_count = output;
    for (uint32_t scalar = 0U;
         scalar < scratch.header.scalar_count; ++scalar) {
        scratch.scalars[scalar] = cell.payload.words[
            scratch.layout.scalar_word_offset + scalar
        ];
    }
    for (uint32_t dependency = 0U;
         dependency < scratch.header.explicit_dep_count;
         ++dependency) {
        const uint64_t producer = cell.payload.words[
            scratch.layout.explicit_dep_word_offset + dependency
        ];
        if (producer >= task_id || producer > INT32_MAX) {
            return SimtTaskBuildStatus::InvalidPlanPayload;
        }
        scratch.explicit_dependencies[dependency] =
            static_cast<int32_t>(producer);
    }
    return SimtTaskBuildStatus::Published;
}

template <typename Ops, typename Runtime>
PA_DEVICE bool MaterializeAndPublishOutputs(
    Runtime &runtime, uint32_t task_id,
    SimtTaskBuildScratch &scratch
)
{
    if (scratch.output_count > plan::kMaxRuntimeOutputsPerTask) {
        return false;
    }
    uint64_t sizes[plan::kMaxRuntimeOutputsPerTask] = {};
    uint64_t total = 0U;
    for (uint32_t output = 0U;
         output < scratch.output_count; ++output) {
        if (!SimtCreateInfoBytes(
                scratch.create_infos[output], sizes[output]
            ) || sizes[output] >
                UINT64_MAX - (kSimtOutputAlignment - 1U)) {
            return false;
        }
        const uint64_t aligned = SimtAlignUp(
            sizes[output], kSimtOutputAlignment
        );
        if (aligned < sizes[output] || total > UINT64_MAX - aligned) {
            return false;
        }
        total += aligned;
    }

    // fresh output 的 last_writer 先从 -1 唯一预留为 producer task。
    // 任何旧轮残留在 descriptor 写入前被拒绝；失败轮进入全局 fatal，
    // 不尝试覆盖其他 leader 的合法进度。
    for (uint32_t output = 0U;
         output < scratch.output_count; ++output) {
        if (Ops::CompareExchange(
                runtime.OutputLastWriter(task_id, output), -1,
                static_cast<int64_t>(task_id)
            ) != -1) {
            return false;
        }
        ++scratch.reserved_output_count;
    }

    uint64_t first_address = 0U;
    if (!runtime.ReserveOutputHeap(
            task_id, total, first_address, scratch.completion_vend
        ) || (total != 0U && first_address == 0U)) {
        return false;
    }
    uint64_t offset = 0U;
    for (uint32_t output = 0U;
         output < scratch.output_count; ++output) {
        const uint32_t tensor = scratch.output_tensor_indices[output];
        if (!SimtMakeMaterializedTensor(
                scratch.create_infos[output], task_id,
                first_address + offset, sizes[output],
                scratch.tensors[tensor]
            )) {
            return false;
        }
        PA_GM volatile uint64_t *destination =
            runtime.OutputDescriptorWords(task_id, output);
        if (destination == nullptr) return false;
        for (uint32_t word = 0U;
             word < plan::kTensorDescWords; ++word) {
            Ops::StorePayloadWord(
                &destination[word],
                SimtTensorDescWord(scratch.tensors[tensor], word)
            );
        }
        offset += SimtAlignUp(sizes[output], kSimtOutputAlignment);
    }
    if (scratch.output_count != 0U) {
        Ops::FlushRegion(
            runtime.OutputDescriptorWords(task_id, 0U),
            static_cast<uint64_t>(scratch.output_count) *
                sizeof(SimtCanonicalTensorDesc)
        );
    }
    for (uint32_t output = 0U;
         output < scratch.output_count; ++output) {
        if (Ops::CompareExchange(
                runtime.OutputPublished(task_id, output), -1,
                static_cast<int64_t>(task_id)
            ) != -1) {
            return false;
        }
        ++scratch.published_output_count;
    }
    return true;
}

template <typename Runtime>
PA_DEVICE bool PrepareWriterDelta(
    Runtime &runtime, uint32_t task_id,
    SimtTaskBuildScratch &scratch
)
{
    scratch.ordinary_count = 0U;
    scratch.symbol_count = 0U;
    for (uint32_t tensor = 0U;
         tensor < scratch.header.tensor_count; ++tensor) {
        const plan::TensorTag tag = static_cast<plan::TensorTag>(
            scratch.header.tensor_tags[tensor]
        );
        if (tag != plan::TensorTag::Inout &&
            tag != plan::TensorTag::OutputExisting) {
            continue;
        }
        if ((scratch.reference_mask &
             (uint32_t{1} << tensor)) != 0U) {
            const plan::RuntimeOutputReferenceWire &reference =
                scratch.references[tensor];
            if (scratch.symbol_count >= plan::kMaxTaskTensors ||
                reference.producer_task_id < 0 ||
                static_cast<uint32_t>(reference.producer_task_id) >=
                    task_id || reference.output_slot < 0) {
                return false;
            }
            uint32_t key = 0U;
            if (!SimtEncodeSharedSymbolHistoryKey(
                    static_cast<uint32_t>(reference.producer_task_id),
                    static_cast<uint32_t>(reference.output_slot), key
                )) {
                return false;
            }
            for (uint32_t previous = 0U;
                 previous < scratch.symbol_count; ++previous) {
                if (scratch.symbol_keys[previous] == key) return false;
            }
            scratch.symbol_keys[scratch.symbol_count++] = key;
            continue;
        }
        const SimtCanonicalTensorDesc &descriptor =
            scratch.tensors[tensor];
        if (descriptor.manual_dep != 0U) continue;
        if (scratch.ordinary_count >= plan::kMaxTaskTensors ||
            !SimtMakeWriterRegion(
                descriptor, task_id,
                scratch.ordinary_entries[scratch.ordinary_count]
            )) {
            return false;
        }
        const uint32_t bucket = runtime.OrdinaryBucket(
            descriptor.buffer_addr
        );
        if (bucket > UINT16_MAX) return false;
        uint32_t ordinal = 0U;
        for (uint32_t previous = 0U;
             previous < scratch.ordinary_count; ++previous) {
            ordinal += scratch.ordinary_buckets[previous] == bucket
                ? 1U : 0U;
        }
        if (ordinal > UINT8_MAX) return false;
        scratch.ordinary_buckets[scratch.ordinary_count] =
            static_cast<uint16_t>(bucket);
        scratch.ordinary_bucket_ordinals[scratch.ordinary_count] =
            static_cast<uint8_t>(ordinal);
        ++scratch.ordinary_count;
    }
    return scratch.ordinary_count + scratch.symbol_count <=
        plan::kMaxTaskTensors;
}

template <typename Ops, typename Runtime>
PA_DEVICE bool WaitForInsertPredecessor(
    Runtime &runtime, uint32_t task_id
)
{
    if (task_id == 0U) return true;
    const int64_t expected = static_cast<int64_t>(task_id) - 1;
    const uint64_t begin = Ops::Now();
    while (true) {
        const int64_t observed = Ops::Load(
            runtime.InsertCompletion(task_id - 1U)
        );
        if (observed == expected) return true;
        if (observed > expected ||
            Ops::Load(runtime.GlobalFatal()) != 0 ||
            Ops::Now() - begin > runtime.WatchdogTicks()) {
            return false;
        }
        Ops::SpinHint();
    }
}

template <typename Ops, typename Runtime>
PA_DEVICE bool PublishWriterMetadataAndCompletion(
    Runtime &runtime, uint32_t task_id,
    SimtTaskBuildScratch &scratch
)
{
    if (!runtime.CheckOrdinaryAppend(
            scratch.ordinary_entries, scratch.ordinary_buckets,
            scratch.ordinary_bucket_ordinals,
            scratch.ordinary_count, task_id
        )) {
        return false;
    }

    if (scratch.symbol_count != 0U) {
        auto *history = runtime.WriterHistory(task_id);
        if (history == nullptr) return false;
        for (uint32_t symbol = 0U;
             symbol < scratch.symbol_count; ++symbol) {
            const uint32_t key = scratch.symbol_keys[symbol];
            uint32_t producer = 0U;
            uint32_t slot = 0U;
            if (!SimtDecodeSharedSymbolHistoryKey(
                    key, producer, slot
                ) || producer >= task_id ||
                Ops::Load(runtime.OutputPublished(producer, slot)) !=
                    static_cast<int64_t>(producer)) {
                return false;
            }
            const int64_t previous = Ops::Load(
                runtime.OutputLastWriter(producer, slot)
            );
            if (previous < static_cast<int64_t>(producer) ||
                previous >= static_cast<int64_t>(task_id)) {
                return false;
            }
            scratch.symbol_previous[symbol] =
                static_cast<int32_t>(previous);
        }
        // A5 Scalar/VF writer 不能用 ordinary GM store 污染本核 DCache 后
        // 再拿 reader-side DCCI 冒充发布。history canonical payload 逐 64bit
        // 使用 bypass stcg；threadfence 完成后，last_writer CAS 才是 reader
        // 可达边界。两个 header word 与每条 record 的 bit packing 显式匹配
        // SimtWriterHistoryCell ABI，未使用 record/padding 不会被 reader 访问。
        PA_GM volatile uint64_t *history_words =
            reinterpret_cast<PA_GM volatile uint64_t *>(history);
        Ops::StorePayloadWord(
            &history_words[0],
            static_cast<uint64_t>(kSimtWriterHistoryMagic) |
                (static_cast<uint64_t>(task_id) << 32U)
        );
        Ops::StorePayloadWord(
            &history_words[1],
            static_cast<uint64_t>(scratch.symbol_count)
        );
        for (uint32_t symbol = 0U;
             symbol < scratch.symbol_count; ++symbol) {
            Ops::StorePayloadWord(
                &history_words[2U + symbol],
                static_cast<uint64_t>(scratch.symbol_keys[symbol]) |
                    (static_cast<uint64_t>(static_cast<uint32_t>(
                         scratch.symbol_previous[symbol]
                     )) << 32U)
            );
        }
        Ops::StoreBarrier();
        for (uint32_t symbol = 0U;
             symbol < scratch.symbol_count; ++symbol) {
            const uint32_t key = scratch.symbol_keys[symbol];
            uint32_t producer = 0U;
            uint32_t slot = 0U;
            if (!SimtDecodeSharedSymbolHistoryKey(
                    key, producer, slot
                ) || Ops::CompareExchange(
                    runtime.OutputLastWriter(producer, slot),
                    scratch.symbol_previous[symbol],
                    static_cast<int64_t>(task_id)
                ) != scratch.symbol_previous[symbol]) {
                return false;
            }
        }
    }

    if (!runtime.AppendOrdinary(
            scratch.ordinary_entries, scratch.ordinary_buckets,
            scratch.ordinary_count, task_id
        )) {
        return false;
    }
    Ops::StoreBarrier();
    const int64_t initial = static_cast<int64_t>(task_id) - 1;
    scratch.insert_completion_published =
        Ops::CompareExchange(
            runtime.InsertCompletion(task_id), initial,
            static_cast<int64_t>(task_id)
        ) == initial;
    return scratch.insert_completion_published;
}

// completion[N] 尚未发布时，没有合法 N+1 reader 可以取得本 task 的
// descriptor publication 证明。失败 owner 因而按严格逆序尽力撤回自己
// 已发布的 fresh-output control。descriptor payload 与 heap FetchAdd 不在
// 这里回滚：本轮已经 terminal fatal，Host 下一轮会重置整份 sidecar；
// 局部回滚 cursor/vend 会覆盖其他 leader 的合法 reservation。
template <typename Ops, typename Runtime>
PA_DEVICE bool RollbackOutputsBeforeInsertCompletion(
    Runtime &runtime, uint32_t task_id,
    SimtTaskBuildScratch &scratch
)
{
    if (scratch.insert_completion_published ||
        scratch.published_output_count > scratch.reserved_output_count ||
        scratch.reserved_output_count > scratch.output_count) {
        return false;
    }
    bool rollback_ok = true;
    for (uint32_t output = scratch.published_output_count;
         output != 0U; --output) {
        const uint32_t slot = output - 1U;
        rollback_ok &= Ops::CompareExchange(
                           runtime.OutputPublished(task_id, slot),
                           static_cast<int64_t>(task_id), -1
                       ) == static_cast<int64_t>(task_id);
    }
    for (uint32_t output = scratch.reserved_output_count;
         output != 0U; --output) {
        const uint32_t slot = output - 1U;
        rollback_ok &= Ops::CompareExchange(
                           runtime.OutputLastWriter(task_id, slot),
                           static_cast<int64_t>(task_id), -1
                       ) == static_cast<int64_t>(task_id);
    }
    if (rollback_ok) {
        scratch.published_output_count = 0U;
        scratch.reserved_output_count = 0U;
    }
    return rollback_ok;
}

PA_DEVICE bool SimtAddFanin(
    SimtTaskBuildScratch &scratch, int32_t producer,
    uint32_t task_id, int32_t reader_lower_bound
)
{
    if (producer < 0) return producer == -1;
    if (reader_lower_bound < 0 ||
        reader_lower_bound > static_cast<int32_t>(task_id) ||
        static_cast<uint32_t>(producer) >= task_id) {
        return false;
    }
    // 与 ordinary TensorMap 的有效窗口保持同一半开区间 [N-H,N)。
    // 窗口外旧 producer 已经由 reclaim/heap-window 合同判为 external，
    // 不能继续塞入跨核 Exec payload；self/future producer 仍是硬错误。
    if (producer < reader_lower_bound) return true;
    for (uint32_t edge = 0U; edge < scratch.fanin_count; ++edge) {
        if (scratch.fanin[edge] == producer) return true;
    }
    if (scratch.fanin_count >= exec::kExecMaxFanin) return false;
    scratch.fanin[scratch.fanin_count++] = producer;
    return true;
}

template <typename Ops, typename Runtime>
PA_DEVICE bool ResolveSymbolWriterBefore(
    Runtime &runtime,
    const plan::RuntimeOutputReferenceWire &reference,
    uint32_t reader_task, int32_t &resolved
)
{
    const uint32_t producer =
        static_cast<uint32_t>(reference.producer_task_id);
    const uint32_t slot = static_cast<uint32_t>(reference.output_slot);
    uint32_t key = 0U;
    if (producer >= reader_task ||
        !SimtEncodeSharedSymbolHistoryKey(producer, slot, key) ||
        Ops::Load(runtime.OutputPublished(producer, slot)) !=
            static_cast<int64_t>(producer)) {
        return false;
    }
    int32_t latest = static_cast<int32_t>(
        Ops::Load(runtime.OutputLastWriter(producer, slot))
    );
    uint32_t traversed = 0U;
    while (latest >= static_cast<int32_t>(reader_task)) {
        if (latest < 0 ||
            static_cast<uint32_t>(latest) >= runtime.TaskCapacity() ||
            ++traversed > reader_task + 1U) {
            return false;
        }
        auto *history =
            runtime.WriterHistory(static_cast<uint32_t>(latest));
        if (history == nullptr) return false;
        Ops::InvalidateRegion(history, sizeof(*history));
        if (history->magic != kSimtWriterHistoryMagic ||
            history->writer_task != latest || history->reserved != 0U ||
            history->count == 0U ||
            history->count > plan::kMaxTaskTensors) {
            return false;
        }
        bool found = false;
        int32_t previous = -1;
        for (uint32_t entry = 0U; entry < history->count; ++entry) {
            if (history->entries[entry].symbol_key != key) continue;
            if (found) return false;
            found = true;
            previous = history->entries[entry].previous_writer;
        }
        if (!found || previous < static_cast<int32_t>(producer) ||
            previous >= latest) {
            return false;
        }
        latest = previous;
    }
    if (latest < static_cast<int32_t>(producer) ||
        latest >= static_cast<int32_t>(reader_task)) {
        return false;
    }
    resolved = latest;
    return true;
}

template <typename Ops, typename Runtime>
PA_DEVICE bool CollectFaninAndResolveTensors(
    Runtime &runtime, uint32_t task_id,
    SimtTaskBuildScratch &scratch
)
{
    scratch.fanin_count = 0U;
    int32_t reader_lower_bound = 0;
    if (task_id > static_cast<uint32_t>(INT32_MAX) ||
        !runtime.FaninLowerBound(task_id, reader_lower_bound) ||
        reader_lower_bound < 0 ||
        reader_lower_bound > static_cast<int32_t>(task_id)) {
        return false;
    }
    for (uint32_t tensor = 0U;
         tensor < scratch.header.tensor_count; ++tensor) {
        const plan::TensorTag tag = static_cast<plan::TensorTag>(
            scratch.header.tensor_tags[tensor]
        );
        if (tag == plan::TensorTag::Output) continue;
        if ((scratch.reference_mask &
             (uint32_t{1} << tensor)) != 0U) {
            const plan::RuntimeOutputReferenceWire &reference =
                scratch.references[tensor];
            int32_t writer = -1;
            if (!ResolveSymbolWriterBefore<Ops>(
                    runtime, reference, task_id, writer
                ) || !SimtAddFanin(
                    scratch, writer, task_id, reader_lower_bound
                )) {
                return false;
            }
            PA_GM volatile uint64_t *source =
                runtime.OutputDescriptorWords(
                    static_cast<uint32_t>(reference.producer_task_id),
                    static_cast<uint32_t>(reference.output_slot)
                );
            if (source == nullptr) return false;
            Ops::InvalidateRegion(
                const_cast<PA_GM uint64_t *>(source),
                sizeof(SimtCanonicalTensorDesc)
            );
            SimtDecodeTensorDesc(source, scratch.tensors[tensor]);
            if (!SimtValidateTensorDesc(
                    scratch.tensors[tensor], task_id
                )) {
                return false;
            }
            continue;
        }

        const SimtCanonicalTensorDesc &descriptor =
            scratch.tensors[tensor];
        if (descriptor.manual_dep != 0U ||
            tag == plan::TensorTag::NoDependency) {
            continue;
        }
        if (descriptor.owner_task_id != kSimtInvalidTaskOwner &&
            !SimtAddFanin(
                scratch, static_cast<int32_t>(
                    descriptor.owner_task_id
                ), task_id, reader_lower_bound
            )) {
            return false;
        }
        if (tag == plan::TensorTag::Input ||
            tag == plan::TensorTag::Inout ||
            tag == plan::TensorTag::OutputExisting) {
            int32_t producer = -1;
            if (!runtime.LookupOrdinary(
                    descriptor, task_id, producer
                ) || !SimtAddFanin(
                    scratch, producer, task_id, reader_lower_bound
                )) {
                return false;
            }
        }
    }
    for (uint32_t dependency = 0U;
         dependency < scratch.header.explicit_dep_count;
         ++dependency) {
        if (!SimtAddFanin(
                scratch, scratch.explicit_dependencies[dependency],
                task_id, reader_lower_bound
            )) {
            return false;
        }
    }
    return true;
}

struct SimtExecPayloadSource {
    const SimtTaskBuildScratch &scratch;

    PA_DEVICE uint64_t TensorWord(
        uint32_t tensor, uint32_t word
    ) const {
        return SimtTensorDescWord(scratch.tensors[tensor], word);
    }

    PA_DEVICE uint64_t TensorReference(uint32_t) const {
        // 第一版把 resolved descriptor 全部内联进 immutable Exec payload。
        // SharedOutputRef 已在 Build 侧解析，executor 不再解释 symbol。
        return 0U;
    }

    PA_DEVICE uint64_t Scalar(uint32_t scalar) const {
        return scratch.scalars[scalar];
    }

    PA_DEVICE int32_t Fanin(uint32_t edge) const {
        return scratch.fanin[edge];
    }
};

template <typename Ops, typename Runtime>
PA_DEVICE bool PublishTerminalBuildResult(
    Runtime &runtime, uint32_t task_id, uint32_t build_owner,
    SimtTaskBuildScratch &scratch
)
{
    const plan::EngineClass engine =
        static_cast<plan::EngineClass>(scratch.header.engine_class);
    if (!runtime.ValidateAdapterRoute(
            scratch.header.adapter_flags,
            scratch.header.adapter_data,
            scratch.header.function_id, engine
        )) {
        return false;
    }
    if (engine == plan::EngineClass::MetadataOnly) {
        return scratch.header.function_id == plan::kInvalidFunctionId &&
               runtime.PublishMetadataCompletion(
                   task_id, scratch.completion_vend
               );
    }
    if (scratch.header.core_num != 1 ||
        scratch.header.require_sync_start != 0U) {
        return false;
    }
    exec::ExecEngineClass exec_engine = exec::ExecEngineClass::None;
    if (engine == plan::EngineClass::Aic) {
        exec_engine = exec::ExecEngineClass::Aic;
    } else if (engine == plan::EngineClass::Aiv) {
        exec_engine = exec::ExecEngineClass::Aiv;
    } else {
        return false;
    }
    const exec::ExecPayloadSpec spec{
        task_id,
        /*function_address=*/0U,
        scratch.completion_vend,
        scratch.header.function_id,
        scratch.header.tensor_count,
        scratch.header.scalar_count,
        static_cast<uint16_t>(scratch.fanin_count),
        exec_engine,
        /*flags=*/0U,
        /*multicore_group_id=*/0U,
        /*multicore_rank=*/0U,
        /*multicore_size=*/1U,
        /*tensor_reference_mask=*/0U,
    };
    const SimtExecPayloadSource source{scratch};
    return exec::BuildAndPublishExecPayload<Ops>(
               runtime.ExecCell(task_id), build_owner,
               spec, source, runtime.ExecFatal()
           ) == exec::ExecBuildResult::Published;
}

// 一个 warp leader 的完整真实 Build 业务叶子：Plan acquire、输出物化、
// task-ID 严格 ordinary/symbol Register、fanin 与 metadata/BUILT 终态。
// ticket 只给 task_id；函数不根据 task_id 推断 kind，也不接受 Host dispatch
// plan。PA adapter 如需核对路由，只能消费 canonical header 的 flags/data。
template <typename Ops, typename Runtime>
PA_DEVICE SimtTaskBuildStatus BuildCanonicalPlanTask(
    Runtime &runtime, const plan::RuntimePlanView &view,
    uint32_t task_id, uint32_t build_owner,
    SimtTaskBuildScratch &scratch
)
{
    SimtTaskBuildStatus status =
        AcquireCanonicalPlanTask<Ops>(view, task_id, scratch);
    if (status != SimtTaskBuildStatus::Published) {
        (void)SimtPublishFatal<Ops>(runtime, task_id, status);
        return status;
    }
    if (!runtime.ValidateAdapterRoute(
            scratch.header.adapter_flags,
            scratch.header.adapter_data,
            scratch.header.function_id,
            static_cast<plan::EngineClass>(
                scratch.header.engine_class
            )
        )) {
        status = SimtTaskBuildStatus::InvalidAdapterRoute;
    } else if (!MaterializeAndPublishOutputs<Ops>(
                   runtime, task_id, scratch
               )) {
        status = SimtTaskBuildStatus::MaterializeFailed;
    } else if (!PrepareWriterDelta(runtime, task_id, scratch)) {
        status = SimtTaskBuildStatus::WriterPublishFailed;
    } else if (!WaitForInsertPredecessor<Ops>(runtime, task_id)) {
        status = SimtTaskBuildStatus::InsertWaitFailed;
    } else if (!PublishWriterMetadataAndCompletion<Ops>(
                   runtime, task_id, scratch
               )) {
        status = SimtTaskBuildStatus::WriterPublishFailed;
    } else if (!CollectFaninAndResolveTensors<Ops>(
                   runtime, task_id, scratch
               )) {
        status = SimtTaskBuildStatus::FaninFailed;
    } else if (!PublishTerminalBuildResult<Ops>(
                   runtime, task_id, build_owner, scratch
               )) {
        status = static_cast<plan::EngineClass>(
                     scratch.header.engine_class
                 ) == plan::EngineClass::MetadataOnly
            ? SimtTaskBuildStatus::MetadataCompletionFailed
            : SimtTaskBuildStatus::ExecPublishFailed;
    }
    if (status != SimtTaskBuildStatus::Published) {
        if (!scratch.insert_completion_published) {
            // 即使 rollback 自身发现 control 冲突，也不能用该结果掩盖原始
            // Build failure；下面仍发布 terminal fatal，Host 保留现场并
            // 在下一轮统一重置。
            (void)RollbackOutputsBeforeInsertCompletion<Ops>(
                runtime, task_id, scratch
            );
        }
        (void)SimtPublishFatal<Ops>(runtime, task_id, status);
    }
    return status;
}

}  // namespace pa_scheduler::aicpu_plan_simt

#endif  // PA_SCHEDULER_AICPU_PLAN_ORDINARY_SIMT_PLAN_TASK_BUILDER_H
