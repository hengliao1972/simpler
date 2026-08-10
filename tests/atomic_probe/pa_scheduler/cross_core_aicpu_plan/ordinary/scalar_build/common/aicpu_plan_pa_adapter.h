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

#ifndef PA_SCHEDULER_CROSS_CORE_AICPU_PLAN_PA_ADAPTER_H
#define PA_SCHEDULER_CROSS_CORE_AICPU_PLAN_PA_ADAPTER_H

// AICPU/CPU 编译单元没有 CCEC 地址空间修饰。adapter 必须像 scheduler
// 入口一样提供窄默认值；CCEC 在 include 前给出的 __aicore__/__gm__ 定义
// 仍优先，不能在这里把真实地址空间静默抹掉。
#ifndef PA_DEVICE
#define PA_AICPU_PLAN_ADAPTER_DEFINED_PA_DEVICE 1
#define PA_DEVICE inline
#endif
#ifndef PA_GM
#define PA_AICPU_PLAN_ADAPTER_DEFINED_PA_GM 1
#define PA_GM
#endif

#include "pa_frontend.h"
#include "../../../common/aicpu_plan_protocol.h"

namespace pa_scheduler::aicpu_plan_adapter {

using aicpu_plan::EngineClass;
using aicpu_plan::RuntimeTaskPlanCell;
using aicpu_plan::RuntimeTaskPlanHeader;
using aicpu_plan::RuntimeTaskPlanLayout;
using aicpu_plan::RuntimeTaskPlanSpec;
using aicpu_plan::TensorTag;

static_assert(
    sizeof(TensorDesc) ==
        aicpu_plan::kTensorDescWords * sizeof(uint64_t),
    "PA TensorDesc and canonical Plan wire size diverged"
);
static_assert(
    sizeof(TensorCreateInfo) ==
        aicpu_plan::kTensorCreateInfoWords * sizeof(uint64_t),
    "PA TensorCreateInfo and canonical Plan wire size diverged"
);
static_assert(
    sizeof(FdwicOutputRef) ==
        aicpu_plan::kOutputReferenceWords * sizeof(uint64_t),
    "PA output reference and canonical Plan wire size diverged"
);

PA_DEVICE uint64_t PackU32Pair(uint32_t low, uint32_t high)
{
    return static_cast<uint64_t>(low) |
           (static_cast<uint64_t>(high) << 32U);
}

// 不复制 C++ 对象的 padding。Plan wire 对每个字段逐项编码，未使用维度和
// 保留字节固定为零；这样 AICPU 与 CCEC 不会因为栈上旧字节得到不同 hash。
template <typename Source>
PA_DEVICE uint64_t CanonicalTensorDescWord(
    const Source &tensor, uint32_t word
)
{
    switch (word) {
        case 0U: return tensor.buffer_addr;
        case 1U: return tensor.buffer_size;
        case 2U: return tensor.owner_task_id;
        case 3U: return tensor.start_offset;
        case 4U:
            return PackU32Pair(
                static_cast<uint32_t>(tensor.version), tensor.ndims
            );
        case 5U:
            return static_cast<uint64_t>(tensor.dtype) |
                   (static_cast<uint64_t>(tensor.manual_dep) << 8U) |
                   (static_cast<uint64_t>(tensor.is_contiguous) << 16U) |
                   (static_cast<uint64_t>(tensor.child_memory) << 24U) |
                   (static_cast<uint64_t>(tensor.shapes[0]) << 32U);
        case 6U: return PackU32Pair(tensor.shapes[1], tensor.shapes[2]);
        case 7U: return PackU32Pair(tensor.shapes[3], tensor.shapes[4]);
        case 8U: return tensor.extent_elem_cache;
        case 9U: return PackU32Pair(tensor.strides[0], tensor.strides[1]);
        case 10U: return PackU32Pair(tensor.strides[2], tensor.strides[3]);
        case 11U: return static_cast<uint64_t>(tensor.strides[4]);
        default: return 0U;
    }
}

PA_DEVICE uint64_t CanonicalCreateInfoWord(
    const TensorCreateInfo &info, uint32_t word
)
{
    switch (word) {
        case 0U: return info.initial_value;
        case 1U: return static_cast<uint64_t>(info.has_initial_value);
        case 2U: return info.reserved0;
        case 3U: return info.start_offset;
        case 4U:
            return PackU32Pair(
                static_cast<uint32_t>(info.version), info.ndims
            );
        case 5U:
            return static_cast<uint64_t>(info.dtype) |
                   (static_cast<uint64_t>(info.manual_dep) << 8U) |
                   (static_cast<uint64_t>(info.is_contiguous) << 16U) |
                   (static_cast<uint64_t>(info.child_memory) << 24U) |
                   (static_cast<uint64_t>(info.shapes[0]) << 32U);
        case 6U: return PackU32Pair(info.shapes[1], info.shapes[2]);
        case 7U: return PackU32Pair(info.shapes[3], info.shapes[4]);
        default: return 0U;
    }
}

PA_DEVICE uint64_t CanonicalOutputRefWord(
    FdwicOutputRef reference, uint32_t word
)
{
    if (word == 0U) {
        return static_cast<uint64_t>(
                   static_cast<uint32_t>(reference.producer_task_id)
               ) |
               (static_cast<uint64_t>(
                    static_cast<uint16_t>(reference.output_slot)
                ) << 32U) |
               (static_cast<uint64_t>(reference.flags) << 48U) |
               (static_cast<uint64_t>(reference.view_ndims) << 56U);
    }
    if (word == 1U) {
        return PackU32Pair(
            reference.view_shape0, reference.view_offset0
        );
    }
    return 0U;
}

template <typename Source>
PA_DEVICE bool ValidateCanonicalTensorDesc(const Source &tensor)
{
    if (tensor.buffer_addr == 0U || tensor.buffer_size == 0U ||
        tensor.ndims == 0U || tensor.ndims > kMaxTensorDims ||
        tensor.dtype >= DataType::Count ||
        tensor.start_offset > tensor.buffer_size) {
        return false;
    }
    for (uint32_t dim = 0U; dim < tensor.ndims; ++dim) {
        if (tensor.shapes[dim] == 0U) return false;
    }
    return true;
}

PA_DEVICE bool ValidateCanonicalCreateInfo(const TensorCreateInfo &info)
{
    if (info.ndims == 0U || info.ndims > kMaxTensorDims ||
        info.dtype >= DataType::Count || info.reserved0 != 0U) {
        return false;
    }
    for (uint32_t dim = 0U; dim < info.ndims; ++dim) {
        if (info.shapes[dim] == 0U) return false;
    }
    return true;
}

// 公共 Plan 协议不认识 PA TaskKind。该转换只存在于 PA adapter，后续
// 其他算子可以拥有自己的 adapter，而不会改变 PlanCell 的 wire ABI。
PA_DEVICE bool PaTensorTag(TensorArgType source, TensorTag &target)
{
    switch (source) {
        case TensorArgType::Input:
            target = TensorTag::Input;
            return true;
        case TensorArgType::Output:
            target = TensorTag::Output;
            return true;
        case TensorArgType::Inout:
            target = TensorTag::Inout;
            return true;
        case TensorArgType::OutputExisting:
            target = TensorTag::OutputExisting;
            return true;
        case TensorArgType::NoDependency:
            target = TensorTag::NoDependency;
            return true;
    }
    return false;
}

PA_DEVICE bool PaTensorTag(TensorTag source, TensorArgType &target)
{
    switch (source) {
        case TensorTag::Input:
            target = TensorArgType::Input;
            return true;
        case TensorTag::Output:
            target = TensorArgType::Output;
            return true;
        case TensorTag::Inout:
            target = TensorArgType::Inout;
            return true;
        case TensorTag::OutputExisting:
            target = TensorArgType::OutputExisting;
            return true;
        case TensorTag::NoDependency:
            target = TensorArgType::NoDependency;
            return true;
    }
    return false;
}

// adapter_flags 沿用 shared PA 已验证的紧凑 meta ABI，但该业务字节只能
// 由 PA adapter 解释。公共 Plan 层原样搬运它，不应知道 Alloc/QK/SF/PV/UP。
// 这里同时锁定 task-id 局部位置与执行引擎，避免合法 wire 被错误业务
// meta 解释后送入另一类 engine。
PA_DEVICE bool ValidatePaAdapterMetadata(
    uint32_t task_id, EngineClass engine_class, uint8_t encoded,
    uint16_t batch_start
)
{
    if ((encoded & kSharedPaTicketMetaPresent) == 0U ||
        task_id >= kMaxTasks) {
        return false;
    }
    const TaskKind kind = static_cast<TaskKind>(
        encoded & kSharedPaTicketKindMask
    );
    const uint32_t group_index =
        (encoded >> kSharedPaTicketGroupShift) &
        kSharedPaTicketGroupMask;
    const bool has_following =
        (encoded & kSharedPaTicketHasFollowing) != 0U;
    const bool is_last =
        (encoded & kSharedPaTicketLastSubmit) != 0U;
    if (kind >= TaskKind::Count ||
        group_index >= kSharedPaMaxBlockGroups ||
        (kind == TaskKind::Alloc &&
         (group_index != 0U || has_following)) ||
        (kind != TaskKind::Up && has_following) ||
        (has_following &&
         group_index + 1U >= kSharedPaMaxBlockGroups) ||
        (is_last &&
         (has_following ||
          (kind != TaskKind::Alloc && kind != TaskKind::Up)))) {
        return false;
    }
    const uint32_t task_offset = SharedPaTaskOffset(kind, group_index);
    // batch_start 是 AICPU 从真实 Alloc callback 维护并随 Plan 发布的
    // provenance。固定 PA offset 只在 adapter 内交叉校验这份显式元数据，
    // 不再用于 Scalar 从 task_id 反推出业务语义。
    if (batch_start > task_id || task_id - batch_start != task_offset) {
        return false;
    }

    const EngineClass expected_engine =
        kind == TaskKind::Alloc
            ? EngineClass::MetadataOnly
            : ((kind == TaskKind::Qk || kind == TaskKind::Pv)
                   ? EngineClass::Aic
                   : EngineClass::Aiv);
    return engine_class == expected_engine;
}

struct PaRuntimeTaskPlanSource {
    const TaskArgs &args;

    PA_DEVICE TensorTag TensorTagAt(uint32_t tensor) const
    {
        TensorTag tag = TensorTag::Input;
        (void)PaTensorTag(TaskTag(args, tensor), tag);
        return tag;
    }

    PA_DEVICE bool TensorIsReference(uint32_t tensor) const
    {
        return args.tensors[tensor].kind ==
            TensorRefKind::SharedOutputRef;
    }

    PA_DEVICE uint64_t TensorWord(
        uint32_t tensor, uint32_t word
    ) const
    {
        const TaskTensorRef &reference = args.tensors[tensor];
        if (reference.kind == TensorRefKind::SharedOutputRef) {
            return CanonicalOutputRefWord(
                reference.pointer.output_ref, word
            );
        }
        if (reference.kind == TensorRefKind::CreateInfo) {
            return CanonicalCreateInfoWord(
                *reference.pointer.create_info, word
            );
        }
        if (reference.kind == TensorRefKind::GmTensor) {
            return CanonicalTensorDescWord(
                *reference.pointer.gm_tensor, word
            );
        }
        return CanonicalTensorDescWord(
            *reference.pointer.local_tensor, word
        );
    }

    PA_DEVICE uint64_t Scalar(uint32_t scalar) const
    {
        return args.scalars[scalar];
    }

    PA_DEVICE uint64_t ExplicitDependency(uint32_t dependency) const
    {
        const uint64_t *dependencies =
            reinterpret_cast<const uint64_t *>(
                static_cast<uintptr_t>(args.explicit_deps)
            );
        return dependencies[dependency];
    }
};

PA_DEVICE bool ValidatePaPlanSource(
    const TaskArgs &args, uint32_t task_id,
    uint32_t &reference_mask, uint16_t &output_count
)
{
    if (task_id >= kMaxTasks || args.has_error ||
        args.tensor_count < 0 ||
        args.tensor_count > static_cast<int32_t>(kMaxTaskTensors) ||
        args.scalar_count < 0 ||
        args.scalar_count > static_cast<int32_t>(kMaxTaskScalars) ||
        args.explicit_dep_count > aicpu_plan::kMaxExplicitDependencies ||
        (args.explicit_dep_count != 0U && args.explicit_deps == 0U) ||
        args.launch_spec.core_num <= 0) {
        return false;
    }
    reference_mask = 0U;
    output_count = 0U;
    for (uint32_t tensor = 0U;
         tensor < static_cast<uint32_t>(args.tensor_count); ++tensor) {
        TensorTag wire_tag = TensorTag::Input;
        if (!PaTensorTag(TaskTag(args, tensor), wire_tag)) return false;
        const TaskTensorRef &reference = args.tensors[tensor];
        if (wire_tag == TensorTag::Output) {
            if (reference.kind != TensorRefKind::CreateInfo ||
                reference.pointer.create_info == nullptr ||
                !ValidateCanonicalCreateInfo(
                    *reference.pointer.create_info
                )) {
                return false;
            }
            ++output_count;
        } else if (reference.kind == TensorRefKind::SharedOutputRef) {
            const FdwicOutputRef output_ref =
                SharedOutputReference(reference);
            if (!IsValidSharedOutputRef(output_ref) ||
                output_ref.producer_task_id < 0 ||
                static_cast<uint32_t>(output_ref.producer_task_id) >=
                    task_id) {
                return false;
            }
            reference_mask |= uint32_t{1} << tensor;
        } else if (reference.kind == TensorRefKind::LocalTensor) {
            if (reference.pointer.local_tensor == nullptr ||
                !ValidateCanonicalTensorDesc(
                    *reference.pointer.local_tensor
                )) {
                return false;
            }
        } else if (reference.kind == TensorRefKind::GmTensor) {
            if (reference.pointer.gm_tensor == nullptr ||
                !ValidateCanonicalTensorDesc(
                    *reference.pointer.gm_tensor
                )) {
                return false;
            }
        } else {
            return false;
        }
    }
    return output_count <= kSharedOutputMaxPerTask;
}

PA_DEVICE bool MakePaRuntimeTaskPlanSpec(
    const TaskArgs &args, uint32_t task_id, int32_t function_id,
    EngineClass engine_class, uint8_t adapter_flags,
    uint32_t batch_start,
    RuntimeTaskPlanSpec &spec
)
{
    uint32_t reference_mask = 0U;
    uint16_t output_count = 0U;
    if (batch_start > UINT16_MAX ||
        !ValidatePaAdapterMetadata(
            task_id, engine_class, adapter_flags,
            static_cast<uint16_t>(batch_start)
        ) ||
        !ValidatePaPlanSource(
            args, task_id, reference_mask, output_count
        )) {
        return false;
    }
    const bool metadata_only = engine_class == EngineClass::MetadataOnly;
    if ((metadata_only && function_id != -1) ||
        (!metadata_only && function_id < 0)) {
        return false;
    }
    spec = RuntimeTaskPlanSpec{
        task_id,
        metadata_only
            ? aicpu_plan::kInvalidFunctionId
            : static_cast<uint32_t>(function_id),
        static_cast<uint16_t>(args.tensor_count),
        static_cast<uint16_t>(args.scalar_count),
        static_cast<uint16_t>(args.explicit_dep_count),
        output_count,
        engine_class,
        adapter_flags,
        args.launch_spec.core_num,
        static_cast<uint8_t>(
            args.launch_spec.require_sync_start ? 1U : 0U
        ),
        0U,
        static_cast<uint16_t>(batch_start),
        reference_mask,
    };
    return true;
}

// Output CreateInfo 与显式依赖需要在 Build 期间保存在 Scalar 私有空间；
// 普通 TensorDesc 继续直接引用已经 acquire 的 immutable GM Plan payload，
// 避免每个 task 再复制最多 4 KiB descriptor。
struct PaRuntimeTaskDecodeScratch {
    TensorCreateInfo create_infos[kSharedOutputMaxPerTask];
    uint64_t explicit_dependencies[
        aicpu_plan::kMaxExplicitDependencies
    ];
};

static_assert(
    sizeof(PaRuntimeTaskDecodeScratch) ==
        sizeof(TensorCreateInfo) * kSharedOutputMaxPerTask +
            sizeof(uint64_t) *
                aicpu_plan::kMaxExplicitDependencies,
    "PA Plan decode scratch unexpectedly grew"
);

PA_DEVICE bool DecodePaRuntimeTaskPlan(
    PA_GM const RuntimeTaskPlanCell &cell,
    const RuntimeTaskPlanHeader &header,
    const RuntimeTaskPlanLayout &layout,
    TaskArgs &args, PaRuntimeTaskDecodeScratch &scratch
)
{
    const EngineClass engine_class =
        static_cast<EngineClass>(header.engine_class);
    const bool metadata_only =
        engine_class == EngineClass::MetadataOnly;
    if (!ValidatePaAdapterMetadata(
            header.task_id, engine_class, header.adapter_flags,
            header.adapter_data
        ) ||
        (metadata_only
             ? header.function_id != aicpu_plan::kInvalidFunctionId
             : header.function_id == aicpu_plan::kInvalidFunctionId) ||
        header.tensor_count > kMaxTaskTensors ||
        header.scalar_count > kMaxTaskScalars ||
        header.explicit_dep_count >
            aicpu_plan::kMaxExplicitDependencies ||
        layout.scalar_word_offset > layout.payload_words ||
        layout.explicit_dep_word_offset > layout.payload_words) {
        return false;
    }
    ConstructTaskArgs(args);
    args.launch_spec.core_num = header.core_num;
    args.launch_spec.require_sync_start =
        header.require_sync_start != 0U;

    uint32_t output_index = 0U;
    for (uint32_t tensor = 0U;
         tensor < header.tensor_count; ++tensor) {
        TensorArgType tag = TensorArgType::Input;
        if (!PaTensorTag(
                static_cast<TensorTag>(header.tensor_tags[tensor]),
                tag
            )) {
            return false;
        }
        uint32_t word_offset = 0U;
        if (!aicpu_plan::RuntimeTaskPlanTensorWordOffset(
                header, tensor, word_offset
            )) {
            return false;
        }
        args.tags[tensor] = TagValue(tag);
        const bool is_reference =
            (header.tensor_reference_mask &
             (uint32_t{1} << tensor)) != 0U;
        if (is_reference) {
            FdwicOutputRef output_ref{};
            uint64_t *destination =
                reinterpret_cast<uint64_t *>(&output_ref);
            destination[0] = cell.payload.words[word_offset];
            destination[1] = cell.payload.words[word_offset + 1U];
            if (!IsValidSharedOutputRef(output_ref) ||
                output_ref.producer_task_id < 0 ||
                static_cast<uint32_t>(output_ref.producer_task_id) >=
                    header.task_id || tag == TensorArgType::Output) {
                return false;
            }
            args.tensors[tensor].pointer.output_ref = output_ref;
            args.tensors[tensor].kind =
                TensorRefKind::SharedOutputRef;
        } else if (tag == TensorArgType::Output) {
            if (output_index >= kSharedOutputMaxPerTask) return false;
            volatile uint64_t *destination =
                reinterpret_cast<volatile uint64_t *>(
                    &scratch.create_infos[output_index]
                );
            for (uint32_t word = 0U;
                 word < sizeof(TensorCreateInfo) / sizeof(uint64_t);
                 ++word) {
                destination[word] =
                    cell.payload.words[word_offset + word];
            }
            args.tensors[tensor].pointer.create_info =
                &scratch.create_infos[output_index++];
            args.tensors[tensor].kind = TensorRefKind::CreateInfo;
        } else {
            // control 已二次确认且 payload 已 exact invalidate；从这里开始
            // descriptor immutable。去掉 volatile 只改变类型限定，不跳过
            // 前面的可见性协议。
            PA_GM const uint64_t *descriptor_words =
                const_cast<PA_GM const uint64_t *>(
                    &cell.payload.words[word_offset]
                );
            args.tensors[tensor].pointer.gm_tensor =
                reinterpret_cast<PA_GM const TensorDesc *>(
                    descriptor_words
                );
            args.tensors[tensor].kind = TensorRefKind::GmTensor;
        }
    }
    args.tensor_count = static_cast<int32_t>(header.tensor_count);
    for (uint32_t scalar = 0U;
         scalar < header.scalar_count; ++scalar) {
        args.scalars[scalar] = cell.payload.words[
            layout.scalar_word_offset + scalar
        ];
    }
    args.scalar_count = static_cast<int32_t>(header.scalar_count);
    for (uint32_t dependency = 0U;
         dependency < header.explicit_dep_count; ++dependency) {
        scratch.explicit_dependencies[dependency] =
            cell.payload.words[
                layout.explicit_dep_word_offset + dependency
            ];
    }
    args.explicit_dep_count = header.explicit_dep_count;
    args.explicit_deps = header.explicit_dep_count == 0U
        ? 0U
        : static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
              &scratch.explicit_dependencies[0]
          ));
    return true;
}

}  // namespace pa_scheduler::aicpu_plan_adapter

#if defined(PA_AICPU_PLAN_ADAPTER_DEFINED_PA_GM)
#undef PA_GM
#undef PA_AICPU_PLAN_ADAPTER_DEFINED_PA_GM
#endif
#if defined(PA_AICPU_PLAN_ADAPTER_DEFINED_PA_DEVICE)
#undef PA_DEVICE
#undef PA_AICPU_PLAN_ADAPTER_DEFINED_PA_DEVICE
#endif

#endif  // PA_SCHEDULER_CROSS_CORE_AICPU_PLAN_PA_ADAPTER_H
