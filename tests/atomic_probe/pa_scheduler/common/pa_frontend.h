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

#ifndef PA_SCHEDULER_COMMON_PA_FRONTEND_H
#define PA_SCHEDULER_COMMON_PA_FRONTEND_H

#include "pa_model.h"

namespace pa_scheduler {

// 这些基址只充当稳定的 tensor identity，供 descriptor、区间重叠和 heap 地址
// 计算使用；winner workload 不解引用这些 synthetic 地址，real-compute 使用独立
// workspace。context_lens 是唯一按真实 GM 指针读取的 PA 前端输入。
constexpr uint64_t kInvalidTaskId = UINT64_MAX;
constexpr uint64_t kSyntheticQueryBase = 0x200000000ULL;
constexpr uint64_t kSyntheticKeyBase = 0x300000000ULL;
constexpr uint64_t kSyntheticValueBase = 0x400000000ULL;
constexpr uint64_t kSyntheticBlockTableBase = 0x500000000ULL;
constexpr uint64_t kSyntheticOutputBase = 0x600000000ULL;
constexpr uint64_t kSyntheticContextLensBase = 0x700000000ULL;
constexpr uint32_t kPaHeads = 16;
constexpr uint32_t kPaHeadDim = 128;
constexpr uint32_t kPaBlockSize = 128;
constexpr uint32_t kPaBlocksPerRequest = 64;
constexpr uint32_t kPaMaxBlocksPerRequest = 256;
constexpr uint64_t kPaScaleBits = 0x3F800000ULL;
constexpr uint32_t kSpmdLocalContextIndex = kMaxTaskTensors + kMaxTaskScalars;
constexpr uint32_t kSpmdGlobalContextIndex = kSpmdLocalContextIndex + 1;
static_assert(kMaxTaskTensors == 32, "PA frontend requires the real 32 tensor slots");
static_assert(kMaxTaskScalars == 16, "PA frontend requires the real 16 scalar slots");
static_assert(kSpmdLocalContextIndex == 48, "PA local-context dispatch index mismatch");
static_assert(kSpmdGlobalContextIndex == 49, "PA global-context dispatch index mismatch");
static_assert(kMaxFanin == 16, "PA frontend requires the real 16 fanin slots");

enum class TensorRefKind : uint8_t {
    LocalTensor = 0,
    GmTensor = 1,
    CreateInfo = 2,
#if PTO_FDWIC_SHARED_MAP
    SharedOutputRef = 3,
#endif
};

#if PTO_FDWIC_SHARED_MAP
// shared fresh Output 使用稳定的 (producer_task_id, output_slot) 符号，
// 不把某个 worker 私有 payload 中的 TensorDesc 指针传给其他 worker。
// 后四个字段预留真实 runtime 的一维 view ABI；PA Case1 本阶段只产生
// flags/view 全零的直接引用，resolver 对其他形态显式 fail-closed。
struct FdwicOutputRef {
    int32_t producer_task_id;
    int16_t output_slot;
    uint8_t flags;
    uint8_t view_ndims;
    uint32_t view_shape0;
    uint32_t view_offset0;
};
static_assert(sizeof(FdwicOutputRef) == 16, "FdwicOutputRef ABI size changed");
static_assert(alignof(FdwicOutputRef) == 4, "FdwicOutputRef ABI alignment changed");
static_assert(offsetof(FdwicOutputRef, producer_task_id) == 0, "shared output producer offset mismatch");
static_assert(offsetof(FdwicOutputRef, output_slot) == 4, "shared output slot offset mismatch");
static_assert(offsetof(FdwicOutputRef, flags) == 6, "shared output flags offset mismatch");
static_assert(offsetof(FdwicOutputRef, view_ndims) == 7, "shared output view-rank offset mismatch");
static_assert(offsetof(FdwicOutputRef, view_shape0) == 8, "shared output view-shape offset mismatch");
static_assert(offsetof(FdwicOutputRef, view_offset0) == 12, "shared output view-offset mismatch");
static_assert(
    __is_trivially_constructible(FdwicOutputRef),
    "FdwicOutputRef must remain trivial for CCEC block-local state"
);

PA_DEVICE FdwicOutputRef InvalidSharedOutputRef() {
    return FdwicOutputRef{-1, -1, 0, 0, 0, 0};
}

// SubmitContext 在 replay 循环中被重复使用，因此 Reset 必须同时写 task id
// 和 count；字段不能使用默认成员初始化，否则 SubmitContext 会产生非平凡
// 构造函数，而 CCEC 禁止 [[block_local]] runtime state 带 ctor/dtor。
struct SharedTaskOutputs {
    int32_t producer_task_id;
    uint32_t output_count;

    PA_DEVICE void Reset(int32_t task_id) {
        producer_task_id = task_id;
        output_count = 0;
    }

    PA_DEVICE bool AddOutputRef(int32_t task_id, int16_t output_slot) {
        if (task_id < 0 || task_id != producer_task_id ||
            output_count >= kSharedOutputMaxPerTask ||
            output_slot != static_cast<int16_t>(output_count)) {
            return false;
        }
        ++output_count;
        return true;
    }

    PA_DEVICE bool Empty() const { return output_count == 0; }
    PA_DEVICE uint32_t Size() const { return output_count; }
    PA_DEVICE int32_t TaskId() const { return producer_task_id; }

    PA_DEVICE FdwicOutputRef OutputRef(uint32_t index) const {
        if (index >= output_count) {
            return InvalidSharedOutputRef();
        }
        return FdwicOutputRef{
            producer_task_id, static_cast<int16_t>(index), 0, 0, 0, 0,
        };
    }
};
static_assert(sizeof(SharedTaskOutputs) == 8, "SharedTaskOutputs ABI size changed");
static_assert(alignof(SharedTaskOutputs) == 4, "SharedTaskOutputs ABI alignment changed");
static_assert(offsetof(SharedTaskOutputs, producer_task_id) == 0, "shared result task offset mismatch");
static_assert(offsetof(SharedTaskOutputs, output_count) == 4, "shared result count offset mismatch");
static_assert(
    __is_trivially_constructible(SharedTaskOutputs),
    "SharedTaskOutputs must remain trivial for CCEC block-local state"
);
#endif

// TaskArgs 同时容纳 orchestration 栈上的 descriptor、GM 中已物化的 descriptor，
// 以及尚待 Materialize 的 CreateInfo。显式 kind 保留生产 TensorRef 的地址空间分支。
union TensorPointer {
    const TensorDesc *local_tensor;
    PA_GM const TensorDesc *gm_tensor;
    const TensorCreateInfo *create_info;
#if PTO_FDWIC_SHARED_MAP
    FdwicOutputRef output_ref;
#endif
};

struct TaskTensorRef {
    TensorPointer pointer;
    TensorRefKind kind;
};
#if PTO_FDWIC_SHARED_MAP
static_assert(sizeof(TaskTensorRef) == 24, "shared TaskTensorRef must match the PA TensorRef ABI");
static_assert(offsetof(TaskTensorRef, pointer) == 0, "shared TaskTensorRef pointer offset mismatch");
static_assert(offsetof(TaskTensorRef, kind) == 16, "shared TaskTensorRef kind offset mismatch");
#else
static_assert(sizeof(TaskTensorRef) == 16, "TaskTensorRef must match the PA TensorRef ABI");
static_assert(offsetof(TaskTensorRef, pointer) == 0, "TaskTensorRef pointer offset mismatch");
static_assert(offsetof(TaskTensorRef, kind) == 8, "TaskTensorRef kind offset mismatch");
#endif

struct PaLaunchSpec {
    int16_t core_num;
    bool require_sync_start;
};
static_assert(sizeof(PaLaunchSpec) == 4, "PA launch spec ABI mismatch");

struct PaAsyncContext {
    uint64_t completion_count;
    uint64_t completion_error_code;
    uint64_t completion_entries;
    uint32_t completion_capacity;
    uint32_t alignment_padding;
    uint64_t task_token;
};
static_assert(sizeof(PaAsyncContext) == 40, "PA async context ABI mismatch");

struct PaLocalContext {
    int32_t block_index;
    int32_t block_count;
    PaAsyncContext async;
};
// Local/GlobalContext 最终放进 RingSlot 的固定 dispatch 参数位 48/49；它们不是
// standalone 自定义参数，offset 必须与真实 SPMD kernel 调用约定一致。
static_assert(sizeof(PaLocalContext) == 48, "PA local context ABI mismatch");

struct PaGlobalContext {
    int32_t sub_block_id;
};
static_assert(sizeof(PaGlobalContext) == 4, "PA global context ABI mismatch");

// PTO2 profiling is enabled in the PA baseline. reset() clears all 160 bytes
// below on every QK/SF/PV/UP argument rebuild, even though Case1 does not ask
// to dump an argument. Keeping this storage and write stream matters to the
// spacing between consecutive Claim operations.
// 这段看似未使用的清零属于真实前端成本，删除会改变各 worker 到达
// Claim 的波形与竞争强度，因此仍按生产构造/reset 顺序执行。
struct PaDumpArgSelection {
    uint64_t dump_arg_mask;
    uint64_t dump_arg_index_ambiguous_mask;
    uint64_t scalar_source_ptrs[kMaxTaskScalars];
    uint8_t scalar_dtypes[kMaxTaskScalars];
};
static_assert(sizeof(PaDumpArgSelection) == 160, "PA dump-selection ABI mismatch");

struct TaskArgs {
    // The real TaskArgsTpl inherits its tag mixin first. TensorArgType is an
    // int32 enum in the PA ABI; keeping tags first also reproduces its offsets.
    // tag 数组位于对象首部不是任意排布；Materialize、fanin 与 register
    // 都会重复扫描/复用这些 tag，错误 offset 会同时改变语义和前端访存成本。
    int32_t tags[kMaxTaskTensors];
    TaskTensorRef tensors[kMaxTaskTensors];
    uint64_t scalars[kMaxTaskScalars];
    int32_t tensor_count;
    int32_t scalar_count;

    bool has_error;
    uint64_t error_msg;
    PaLaunchSpec launch_spec;
    PaDumpArgSelection dump_arg_selection;
    uint64_t explicit_deps;
    uint32_t explicit_dep_count;
    uint8_t cacheline_pad[48];
};
#if PTO_FDWIC_SHARED_MAP
static_assert(sizeof(TaskArgs) == 1280, "shared TaskArgs must match the PA L0TaskArgs ABI size");
static_assert(offsetof(TaskArgs, tags) == 0, "shared TaskArgs tag offset mismatch");
static_assert(offsetof(TaskArgs, tensors) == 128, "shared TaskArgs tensor-ref offset mismatch");
static_assert(offsetof(TaskArgs, scalars) == 896, "shared TaskArgs scalar offset mismatch");
static_assert(offsetof(TaskArgs, tensor_count) == 1024, "shared TaskArgs tensor-count offset mismatch");
static_assert(offsetof(TaskArgs, scalar_count) == 1028, "shared TaskArgs scalar-count offset mismatch");
static_assert(offsetof(TaskArgs, has_error) == 1032, "shared TaskArgs error flag offset mismatch");
static_assert(offsetof(TaskArgs, error_msg) == 1040, "shared TaskArgs error pointer offset mismatch");
static_assert(offsetof(TaskArgs, launch_spec) == 1048, "shared TaskArgs launch-spec offset mismatch");
static_assert(offsetof(TaskArgs, dump_arg_selection) == 1056, "shared TaskArgs dump-selection offset mismatch");
static_assert(offsetof(TaskArgs, explicit_deps) == 1216, "shared TaskArgs dependency pointer offset mismatch");
static_assert(offsetof(TaskArgs, explicit_dep_count) == 1224, "shared TaskArgs dependency count offset mismatch");
static_assert(
    __is_trivially_constructible(TaskArgs),
    "shared TaskArgs must not introduce implicit initialization"
);
#else
static_assert(sizeof(TaskArgs) == 1024, "TaskArgs must match the PA L0TaskArgs ABI size");
static_assert(offsetof(TaskArgs, tags) == 0, "TaskArgs tag offset mismatch");
static_assert(offsetof(TaskArgs, tensors) == 128, "TaskArgs tensor-ref offset mismatch");
static_assert(offsetof(TaskArgs, scalars) == 640, "TaskArgs scalar offset mismatch");
static_assert(offsetof(TaskArgs, tensor_count) == 768, "TaskArgs tensor-count offset mismatch");
static_assert(offsetof(TaskArgs, scalar_count) == 772, "TaskArgs scalar-count offset mismatch");
static_assert(offsetof(TaskArgs, has_error) == 776, "TaskArgs error flag offset mismatch");
static_assert(offsetof(TaskArgs, error_msg) == 784, "TaskArgs error pointer offset mismatch");
static_assert(offsetof(TaskArgs, launch_spec) == 792, "TaskArgs launch-spec offset mismatch");
static_assert(offsetof(TaskArgs, dump_arg_selection) == 800, "TaskArgs dump-selection offset mismatch");
static_assert(offsetof(TaskArgs, explicit_deps) == 960, "TaskArgs dependency pointer offset mismatch");
static_assert(offsetof(TaskArgs, explicit_dep_count) == 968, "TaskArgs dependency count offset mismatch");
#endif

struct TaskOutputs {
    uint64_t task_id;
    uint32_t count;
    PA_GM TensorDesc *tensors[kMaxTaskTensors];
};
static_assert(sizeof(TaskOutputs) == 272, "TaskOutputs must match the PA TaskOutputTensors ABI size");
static_assert(offsetof(TaskOutputs, tensors) == 16, "TaskOutputs tensor pointer offset mismatch");

struct SubmitContext {
    PA_GM WorkerState *self;
    PA_GM TaskPayload *payload;
    int32_t task_id;
    int32_t tensor_count;
    int32_t scalar_count;
    uint32_t register_mask;
    uint64_t output_bytes;
    TaskOutputs result;
#if PTO_FDWIC_SHARED_MAP
    SharedTaskOutputs shared_result;
#endif
    int32_t fanin[kMaxFanin];
    int32_t fanin_count;
    int32_t kernel_id;
    bool won;
    bool joint;
    bool joint_init;
    int32_t joint_block;
    int32_t joint_slot;
    int32_t joint_count;
};
// SubmitContext 贯穿一次 Submit：Begin 绑定 task/payload，Materialize 填充输出与
// register_mask，winner 收集 fanin 并构建 slot。它复刻 DistSubmitCtx 而非诊断结构。
#if PTO_FDWIC_SHARED_MAP
static_assert(sizeof(SubmitContext) == 408, "shared SubmitContext must match DistSubmitCtx");
static_assert(offsetof(SubmitContext, output_bytes) == 32, "shared SubmitContext output-byte offset mismatch");
static_assert(offsetof(SubmitContext, result) == 40, "shared SubmitContext result offset mismatch");
static_assert(offsetof(SubmitContext, shared_result) == 312, "shared SubmitContext result-ref offset mismatch");
static_assert(offsetof(SubmitContext, fanin) == 320, "shared SubmitContext fanin offset mismatch");
static_assert(
    __is_trivially_constructible(SubmitContext),
    "shared SubmitContext must remain trivial for CCEC block-local state"
);
#else
static_assert(sizeof(SubmitContext) == 400, "SubmitContext must match DistSubmitCtx");
static_assert(offsetof(SubmitContext, output_bytes) == 32, "SubmitContext output-byte offset mismatch");
static_assert(offsetof(SubmitContext, result) == 40, "SubmitContext result offset mismatch");
static_assert(offsetof(SubmitContext, fanin) == 312, "SubmitContext fanin offset mismatch");
#endif

#if PTO_FDWIC_SHARED_MAP
using PaOutputHandle = FdwicOutputRef;
using OrchestrationTaskOutputs = SharedTaskOutputs;
#else
using PaOutputHandle = PA_GM TensorDesc *;
using OrchestrationTaskOutputs = TaskOutputs;
#endif

PA_DEVICE const OrchestrationTaskOutputs &OrchestrationOutputs(const SubmitContext &context) {
#if PTO_FDWIC_SHARED_MAP
    return context.shared_result;
#else
    return context.result;
#endif
}

PA_DEVICE PaOutputHandle OutputHandleAt(const OrchestrationTaskOutputs &outputs, uint32_t index) {
#if PTO_FDWIC_SHARED_MAP
    return outputs.OutputRef(index);
#else
    return index < outputs.count ? outputs.tensors[index] : nullptr;
#endif
}

PA_DEVICE PaOutputHandle InvalidPaOutputHandle() {
#if PTO_FDWIC_SHARED_MAP
    return InvalidSharedOutputRef();
#else
    return nullptr;
#endif
}

struct OutputLayout {
    uint64_t buffer_sizes[kMaxTaskTensors];
    uint64_t total_output_size;
};
// 只有 tag=Output 的槽位拥有有效 buffer_sizes；总大小按 1 KiB 对齐累计，随后
// 作为 HeapGuard 的 output_bytes 和本 worker heap_next 的推进量。
static_assert(sizeof(OutputLayout) == 264, "OutputLayout must match DistOutputLayout");

// 该状态保存真实 PA orchestration 在五个 Submit 之间传递的输出 handle：
// private 为本 worker materialize payload 中的 descriptor 指针，shared 为
// (producer_task_id, output_slot) 符号；两者经同一 facade 构建后继参数。
struct PaOrchestrationState {
    TensorDesc query;
    TensorDesc key_cache;
    TensorDesc value_cache;
    TensorDesc block_table;
    TensorDesc context_lens;
    TensorDesc output;
    TensorDesc query_view;
    TensorDesc output_view;

    TensorCreateInfo tile_create_info;
    TensorCreateInfo scalar_create_info;
    TensorCreateInfo qk_create_info;
    TensorCreateInfo sf_create_info;

    // The pointer is supplied by the standalone backend. On A5 it must point
    // at GM so every batch performs the same descriptor-based load as PA.
    PA_GM const volatile int32_t *context_lens_data;
    uint64_t scale_bits;
    uint64_t current_sequence;
    uint64_t current_blocks;
    uint64_t current_block_offset;
    uint64_t current_nblocks;
    uint64_t current_valid_len;
    uint32_t current_batch;

    PaOutputHandle accumulated_output;
    PaOutputHandle accumulated_sum;
    PaOutputHandle accumulated_max;
    PaOutputHandle qk_scores;
    PaOutputHandle sf_probs;
    PaOutputHandle sf_max;
    PaOutputHandle sf_sum;
    PaOutputHandle pv_output;
};
#if PTO_FDWIC_SHARED_MAP
static_assert(sizeof(PaOrchestrationState) == 1472, "shared PA orchestration state size changed");
static_assert(
    offsetof(PaOrchestrationState, accumulated_output) == 1340,
    "shared PA output-handle offset changed"
);
#else
static_assert(sizeof(PaOrchestrationState) == 1408, "private PA orchestration state size changed");
static_assert(
    offsetof(PaOrchestrationState, accumulated_output) == 1344,
    "private PA output-handle offset changed"
);
#endif

PA_DEVICE uint64_t ElementSize(DataType dtype) {
    // 输入 dtype 来自已通过 PA ABI 构造的 descriptor/create-info，必须落在 Count 前；
    // 输出字节数同时用于外部 tensor range 与新 Output 的 heap 大小计算。
    constexpr static uint64_t sizes[static_cast<uint32_t>(DataType::Count)] = {
        4, 2, 4, 2, 1, 1, 2, 8, 8, 2, 4, 1,
    };
    return sizes[static_cast<uint32_t>(dtype)];
}

PA_DEVICE int32_t TagValue(TensorArgType tag) { return static_cast<int32_t>(tag); }

PA_DEVICE TensorArgType TaskTag(const TaskArgs &args, uint32_t index) {
    // index 的有效范围由 tensor_count 保证；集中转换避免各阶段对 int32 ABI tag
    // 做不同解释，Materialize/CollectFanin/Register 因而共享同一分类结果。
    return static_cast<TensorArgType>(args.tags[index]);
}

PA_DEVICE void ClearDumpArgSelection(PaDumpArgSelection &selection) {
    // Volatile stores intentionally preserve the profiling-enabled PA reset
    // traffic even though the standalone winner workload never consumes dump data.
    // volatile 的目的不是同步，而是阻止编译器删掉这段生产基线中存在的写流量。
    volatile uint64_t *masks = &selection.dump_arg_mask;
    masks[0] = 0;
    masks[1] = 0;
    volatile uint64_t *sources = &selection.scalar_source_ptrs[0];
    for (uint32_t index = 0; index < kMaxTaskScalars; ++index) {
        sources[index] = 0;
    }
    volatile uint8_t *dtypes = &selection.scalar_dtypes[0];
    for (uint32_t index = 0; index < kMaxTaskScalars; ++index) {
        dtypes[index] = 0;
    }
}

PA_DEVICE void ConstructTaskArgs(TaskArgs &args) {
    // TensorTagMixin<TensorArgType, 32>::tags_{} is value-initialized by the
    // real L0TaskArgs constructor. TensorRef/scalar slots remain lazy.
    // 构造只初始化真实构造函数会触碰的字段，未使用的 tensor/scalar
    // 槽保持惰性；整对象 memset 会引入 PA 本身没有的额外前端开销。
    volatile int32_t *tags = &args.tags[0];
    for (uint32_t index = 0; index < kMaxTaskTensors; ++index) {
        tags[index] = 0;
    }
    args.tensor_count = 0;
    args.scalar_count = 0;
    args.has_error = false;
    args.error_msg = 0;
    args.launch_spec.core_num = 1;
    args.launch_spec.require_sync_start = false;
    ClearDumpArgSelection(args.dump_arg_selection);
    args.explicit_deps = 0;
    args.explicit_dep_count = 0;
}

PA_DEVICE void ResetTaskArgs(TaskArgs &args) {
    // reset 的输出是不含 tensor/scalar/显式依赖的新逻辑参数表，但保留已分配对象及
    // launch_spec；QK/SF/PV/UP 在同一个 1 KiB TaskArgs 上依次复用这一状态。
    args.tensor_count = 0;
    args.scalar_count = 0;
    ClearDumpArgSelection(args.dump_arg_selection);
    args.explicit_deps = 0;
    args.explicit_dep_count = 0;
    args.has_error = false;
    args.error_msg = 0;
}

PA_DEVICE bool ReserveTensorArgs(TaskArgs &args, int32_t count) {
    // tensor 必须先于 scalar 追加，以保持 dispatch args 的 [tensor..., scalar...]
    // 排列；失败只置 has_error，不发生部分追加。
    if (args.scalar_count != 0 || count < 0 ||
        args.tensor_count + count > static_cast<int32_t>(kMaxTaskTensors)) {
        args.has_error = true;
        return false;
    }
    return true;
}

PA_DEVICE void AppendLocalTensor(TaskArgs &args, const TensorDesc &tensor, TensorArgType tag) {
    const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
    args.tensors[index].pointer.local_tensor = &tensor;
    args.tensors[index].kind = TensorRefKind::LocalTensor;
    args.tags[index] = TagValue(tag);
}

PA_DEVICE void AppendGmTensor(TaskArgs &args, PA_GM const TensorDesc &tensor, TensorArgType tag) {
    const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
    args.tensors[index].pointer.gm_tensor = &tensor;
    args.tensors[index].kind = TensorRefKind::GmTensor;
    args.tags[index] = TagValue(tag);
}

#if PTO_FDWIC_SHARED_MAP
PA_DEVICE void AppendSharedOutputRef(TaskArgs &args, FdwicOutputRef reference, TensorArgType tag) {
    const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
    args.tensors[index].pointer.output_ref = reference;
    args.tensors[index].kind = TensorRefKind::SharedOutputRef;
    args.tags[index] = TagValue(tag);
}

PA_DEVICE bool IsValidSharedOutputRef(FdwicOutputRef reference) {
    if (reference.producer_task_id < 0 ||
        reference.producer_task_id >= static_cast<int32_t>(kMaxTasks) ||
        reference.output_slot < 0 ||
        reference.output_slot >= static_cast<int16_t>(kSharedOutputMaxPerTask) ||
        (reference.flags & ~uint8_t{1}) != 0) {
        return false;
    }
    if ((reference.flags & uint8_t{1}) == 0) {
        return reference.view_ndims == 0 && reference.view_shape0 == 0 &&
               reference.view_offset0 == 0;
    }
    return reference.view_ndims == 1 && reference.view_shape0 != 0;
}

PA_DEVICE bool IsPlainSharedOutputRef(FdwicOutputRef reference) {
    return IsValidSharedOutputRef(reference) && reference.flags == 0;
}

PA_DEVICE bool IsSharedOutputReference(const TaskTensorRef &reference) {
    return reference.kind == TensorRefKind::SharedOutputRef;
}

PA_DEVICE FdwicOutputRef SharedOutputReference(const TaskTensorRef &reference) {
    return reference.pointer.output_ref;
}
#endif

PA_DEVICE void AppendOutput(TaskArgs &args, const TensorCreateInfo &create_info) {
    const uint32_t index = static_cast<uint32_t>(args.tensor_count++);
    args.tensors[index].pointer.create_info = &create_info;
    args.tensors[index].kind = TensorRefKind::CreateInfo;
    args.tags[index] = TagValue(TensorArgType::Output);
}

PA_DEVICE bool ReserveScalarArgs(TaskArgs &args, int32_t count) {
    // 先整体校验容量再由 AddTwo/AddThree 连续写入，保证多 scalar 操作全有或全无。
    if (count < 0 || args.scalar_count + count > static_cast<int32_t>(kMaxTaskScalars)) {
        args.has_error = true;
        return false;
    }
    return true;
}

PA_DEVICE void AppendScalar(TaskArgs &args, uint64_t value) {
    args.scalars[static_cast<uint32_t>(args.scalar_count++)] = value;
}

PA_DEVICE void AddLocalTensor(TaskArgs &args, const TensorDesc &tensor, TensorArgType tag) {
    if (ReserveTensorArgs(args, 1)) AppendLocalTensor(args, tensor, tag);
}

PA_DEVICE void AddGmTensor(TaskArgs &args, PA_GM const TensorDesc &tensor, TensorArgType tag) {
    if (ReserveTensorArgs(args, 1)) AppendGmTensor(args, tensor, tag);
}

PA_DEVICE void AddOutputHandleTensor(TaskArgs &args, PaOutputHandle handle, TensorArgType tag) {
#if PTO_FDWIC_SHARED_MAP
    if (!IsValidSharedOutputRef(handle)) {
        args.has_error = true;
        return;
    }
    if (ReserveTensorArgs(args, 1)) {
        AppendSharedOutputRef(args, handle, tag);
    }
#else
    if (handle == nullptr) {
        args.has_error = true;
        return;
    }
    AddGmTensor(args, *handle, tag);
#endif
}

PA_DEVICE void AddOutput(TaskArgs &args, const TensorCreateInfo &create_info) {
    if (ReserveTensorArgs(args, 1)) AppendOutput(args, create_info);
}

PA_DEVICE void AddScalar(TaskArgs &args, uint64_t value) {
    if (ReserveScalarArgs(args, 1)) AppendScalar(args, value);
}

PA_DEVICE void InitCreateInfo(
    TensorCreateInfo &info, const uint32_t shapes[kMaxTensorDims], uint32_t ndims, DataType dtype
) {
    info.initial_value = 0;
    info.has_initial_value = false;
    info.reserved0 = 0;
    info.start_offset = 0;
    info.version = 0;
    info.ndims = ndims;
    info.dtype = dtype;
    info.manual_dep = false;
    info.is_contiguous = true;
    info.child_memory = 0;
    // TensorCreateInfo's real constructor only writes active dimensions.
    // 只写 ndims 个 shape，保留生产构造器的写入范围，不能为方便把五维全清零。
    for (uint32_t index = 0; index < ndims; ++index) {
        info.shapes[index] = shapes[index];
    }
}

PA_DEVICE void ClearCreateInfo(TensorCreateInfo &info) {
    volatile uint8_t *bytes = reinterpret_cast<volatile uint8_t *>(&info);
    for (uint32_t index = 0; index < sizeof(TensorCreateInfo); ++index) {
        bytes[index] = 0;
    }
}

PA_DEVICE void InitExternalTensor(
    TensorDesc &tensor, uint64_t address, const uint32_t shapes[kMaxTensorDims], uint32_t ndims, DataType dtype,
    bool manual_dep
) {
    // 输入为稳定 backing address、逻辑 shape 和依赖属性；输出是完整连续 descriptor，
    // owner 无效表示它不是本轮 task 产生，row-major stride 从末维向前计算。
    uint64_t elements = 1;
    for (uint32_t index = 0; index < ndims; ++index) {
        elements *= shapes[index];
    }
    tensor.buffer_addr = address;
    tensor.buffer_size = elements * ElementSize(dtype);
    tensor.owner_task_id = kInvalidTaskId;
    tensor.start_offset = 0;
    tensor.version = 0;
    tensor.ndims = ndims;
    tensor.dtype = dtype;
    tensor.manual_dep = manual_dep;
    tensor.is_contiguous = true;
    tensor.child_memory = 0;
    for (uint32_t index = 0; index < kMaxTensorDims; ++index) {
        tensor.shapes[index] = shapes[index];
        tensor.strides[index] = 0;
    }
    uint32_t stride = 1;
    for (int32_t index = static_cast<int32_t>(ndims) - 1; index >= 0; --index) {
        tensor.strides[index] = stride;
        stride *= tensor.shapes[index];
    }
    tensor.extent_elem_cache = stride;
}

PA_DEVICE bool InitTensorFromCreateInfo(
    PA_GM TensorDesc &tensor, const TensorCreateInfo &info, uint64_t address, uint64_t buffer_size
) {
    tensor.buffer_addr = address;
    tensor.buffer_size = buffer_size;
    tensor.owner_task_id = kInvalidTaskId;
    tensor.start_offset = info.start_offset;
    tensor.version = info.version;
    tensor.ndims = info.ndims;
    tensor.dtype = info.dtype;
    tensor.manual_dep = info.manual_dep;
    tensor.is_contiguous = info.is_contiguous;
    tensor.child_memory = info.child_memory;
    for (uint32_t index = 0; index < kMaxTensorDims; ++index) {
        tensor.shapes[index] = info.shapes[index];
    }
    uint32_t stride = 1;
    for (int32_t index = static_cast<int32_t>(tensor.ndims) - 1; index >= 0; --index) {
        tensor.strides[index] = stride;
        stride *= tensor.shapes[index];
    }
    tensor.extent_elem_cache = stride;
    // PA can initialize the backing allocation here. Case1 never requests it;
    // the standalone uses synthetic heap addresses and therefore rejects that
    // unsupported path instead of writing to a fabricated GM pointer.
    // Case1 的 has_initial_value 恒为 false；返回 false 是对未模拟分支的
    // 明确保护，不会在合成地址上伪造初始化写入。
    return !info.has_initial_value;
}

PA_DEVICE uint64_t CreateInfoBytes(const TensorCreateInfo &info) {
    uint64_t elements = 1;
    for (uint32_t index = 0; index < info.ndims; ++index) {
        elements *= info.shapes[index];
    }
    return elements * ElementSize(info.dtype);
}

template <typename Source>
PA_DEVICE void CopyTensorLine1(TensorDesc &destination, const Source &source) {
    // view 只需复制 descriptor 第一条 cache line 的身份/shape 字段，随后由调用方
    // 覆盖 offset、shape、stride 与 extent；不做整 128-byte 拷贝以匹配 PA 写流。
    destination.buffer_addr = source.buffer_addr;
    destination.buffer_size = source.buffer_size;
    destination.owner_task_id = source.owner_task_id;
    destination.start_offset = source.start_offset;
    destination.version = source.version;
    destination.ndims = source.ndims;
    destination.dtype = source.dtype;
    destination.manual_dep = source.manual_dep;
    destination.is_contiguous = source.is_contiguous;
    destination.child_memory = source.child_memory;
    for (uint32_t index = 0; index < kMaxTensorDims; ++index) {
        destination.shapes[index] = source.shapes[index];
    }
}

PA_DEVICE void MakeCallbackOutputView(PaOrchestrationState &orch, uint32_t batch) {
    CopyTensorLine1(orch.output_view, orch.output);
    orch.output_view.start_offset = static_cast<uint64_t>(batch) * kPaHeads * kPaHeadDim;
    orch.output_view.ndims = 2;
    orch.output_view.manual_dep = true;
    orch.output_view.shapes[0] = kPaHeads;
    orch.output_view.shapes[1] = kPaHeadDim;
    orch.output_view.strides[0] = kPaHeadDim;
    orch.output_view.strides[1] = 1;
    orch.output_view.extent_elem_cache = kPaHeads * kPaHeadDim;
}

PA_DEVICE void MakeCallbackQueryView(PaOrchestrationState &orch, uint32_t batch) {
    CopyTensorLine1(orch.query_view, orch.query);
    orch.query_view.start_offset = static_cast<uint64_t>(batch) * kPaHeads * kPaHeadDim;
    orch.query_view.ndims = 2;
    orch.query_view.shapes[0] = kPaHeads;
    orch.query_view.shapes[1] = kPaHeadDim;
    orch.query_view.strides[0] = kPaHeadDim;
    orch.query_view.strides[1] = 1;
    orch.query_view.extent_elem_cache = kPaHeads * kPaHeadDim;
}

PA_DEVICE uint64_t MinU64(uint64_t lhs, uint64_t rhs) { return lhs < rhs ? lhs : rhs; }

PA_DEVICE uint64_t ReadPaContextLength(const PaOrchestrationState &orch, uint32_t batch) {
    if (orch.context_lens_data == nullptr) {
        // Compatibility fallback for a backend that has not yet supplied the
        // 256-int GM buffer. Exact PA runs must pass a non-null pointer.
        // 正式对等运行必须走下方 descriptor+stride 的 GM load；fallback
        // 只用于不具备该缓冲区的兼容后端。
        return kPaBlocksPerRequest * kPaBlockSize;
    }
    const uint64_t flat_index = orch.context_lens.start_offset +
                                static_cast<uint64_t>(batch) * orch.context_lens.strides[0];
    PA_GM const volatile int32_t *value = reinterpret_cast<PA_GM const volatile int32_t *>(
        orch.context_lens.buffer_addr + flat_index * ElementSize(DataType::Int32)
    );
    return static_cast<uint64_t>(*value);
}

PA_DEVICE void PreparePaBlockGroup(PaOrchestrationState &orch, uint64_t block_offset) {
    // 输入 block_offset 位于 [0,current_blocks)；输出 nblocks 最多64，并计算最后
    // 一个 block 的有效 token 数。Case1 只有一个 group，但仍执行通用边界算术。
    orch.current_block_offset = block_offset;
    orch.current_nblocks = MinU64(kPaBlocksPerRequest, orch.current_blocks - block_offset);
    const uint64_t last_block_sequence_start =
        (block_offset + orch.current_nblocks - 1) * kPaBlockSize;
    orch.current_valid_len = MinU64(kPaBlockSize, orch.current_sequence - last_block_sequence_start);
}

PA_DEVICE void BeginPaBatchForCallback(PaOrchestrationState &orch, uint32_t batch) {
    // context GM load 与跨 task 共用算术仍在 Submit 之前；只把最终 descriptor
    // 打包延后到 Claim 后的同步 callback，避免把业务数据流挪入运行时 finish。
    orch.current_batch = batch;
    orch.current_sequence = ReadPaContextLength(orch, batch);
    orch.current_blocks = (orch.current_sequence + kPaBlockSize - 1) / kPaBlockSize;
}

PA_DEVICE void InitPaOrchestration(
    PaOrchestrationState &orch, uint32_t batches, PA_GM const volatile int32_t *context_lens_data
) {
    // 初始化只建立整轮回放共享的外部 descriptor/create-info 模板；每 batch 的 view、
    // context length、动态 QK/SF shape 和返回 descriptor 留给五阶段流按原顺序更新。
    const uint32_t query_shape[kMaxTensorDims] = {batches * kPaHeads, kPaHeadDim, 0, 0, 0};
    const uint32_t cache_shape[kMaxTensorDims] = {
        batches * kPaBlocksPerRequest * kPaBlockSize, kPaHeadDim, 0, 0, 0
    };
    const uint32_t table_shape[kMaxTensorDims] = {batches, kPaMaxBlocksPerRequest, 0, 0, 0};
    const uint32_t context_shape[kMaxTensorDims] = {batches, 0, 0, 0, 0};
    InitExternalTensor(orch.query, kSyntheticQueryBase, query_shape, 2, DataType::Bfloat16, false);
    InitExternalTensor(orch.key_cache, kSyntheticKeyBase, cache_shape, 2, DataType::Bfloat16, false);
    InitExternalTensor(orch.value_cache, kSyntheticValueBase, cache_shape, 2, DataType::Bfloat16, false);
    InitExternalTensor(orch.block_table, kSyntheticBlockTableBase, table_shape, 2, DataType::Int32, false);
    const uint64_t context_address = context_lens_data == nullptr
                                         ? kSyntheticContextLensBase
                                         : reinterpret_cast<uint64_t>(context_lens_data);
    InitExternalTensor(orch.context_lens, context_address, context_shape, 1, DataType::Int32, false);
    InitExternalTensor(orch.output, kSyntheticOutputBase, query_shape, 2, DataType::Float32, false);

    const uint32_t tile_shape[kMaxTensorDims] = {kPaHeads, kPaHeadDim, 0, 0, 0};
    const uint32_t scalar_shape[kMaxTensorDims] = {kPaHeads, 0, 0, 0, 0};
    ClearCreateInfo(orch.tile_create_info);
    ClearCreateInfo(orch.scalar_create_info);
    ClearCreateInfo(orch.qk_create_info);
    ClearCreateInfo(orch.sf_create_info);
    InitCreateInfo(orch.tile_create_info, tile_shape, 2, DataType::Float32);
    InitCreateInfo(orch.scalar_create_info, scalar_shape, 1, DataType::Float32);

    // QK/SF create infos are deliberately not constructed here: in PA they are
    // constructed inside the group after Alloc and QK respectively.
    // 动态 shape 依赖当前 block group，提前构造既不符合业务数据流，也会
    // 把真实发生在两个 Submit 之间的前端工作错误搬到初始化阶段。
    orch.context_lens_data = context_lens_data;
    orch.scale_bits = kPaScaleBits;
    orch.current_sequence = 0;
    orch.current_blocks = 0;
    orch.current_block_offset = 0;
    orch.current_nblocks = 0;
    orch.current_valid_len = 0;
    orch.current_batch = 0;

    orch.accumulated_output = InvalidPaOutputHandle();
    orch.accumulated_sum = InvalidPaOutputHandle();
    orch.accumulated_max = InvalidPaOutputHandle();
    orch.qk_scores = InvalidPaOutputHandle();
    orch.sf_probs = InvalidPaOutputHandle();
    orch.sf_max = InvalidPaOutputHandle();
    orch.sf_sum = InvalidPaOutputHandle();
    orch.pv_output = InvalidPaOutputHandle();
}

PA_DEVICE void InitPaOrchestration(PaOrchestrationState &orch, uint32_t batches) {
    InitPaOrchestration(orch, batches, nullptr);
}

struct CallbackSubmitBuildCounts {
    uint32_t reset_calls;
    uint32_t views_created;
    uint32_t dynamic_create_infos;
    uint32_t tensor_args_added;
    uint32_t scalar_args_added;
};

// builder 只在当前 Submit 栈帧内被同步调用，任何 thunk 都不会被保存或跨 TU。
// 所有 Add* 均无 winner 分支，明确保持 compete-first eager 的全员构参语义。
class CallbackSubmitArgsBuilder {
public:
    PA_DEVICE CallbackSubmitArgsBuilder(TaskArgs &args, TaskKind kind)
        : args_(args), kind_(kind), begin_calls_(0), counts_{} {}

    PA_DEVICE void Begin() {
        if (++begin_calls_ != 1) {
            args_.has_error = true;
            return;
        }
        if (kind_ == TaskKind::Alloc) {
            ConstructTaskArgs(args_);
        } else {
            ResetTaskArgs(args_);
            ++counts_.reset_calls;
        }
    }

    PA_DEVICE void RecordView() { ++counts_.views_created; }
    PA_DEVICE void RecordDynamicCreateInfo() { ++counts_.dynamic_create_infos; }

    template <typename Thunk>
    PA_DEVICE void AddLocalInput(Thunk thunk) {
        if (!Ready()) return;
        const TensorDesc &tensor = thunk();
        AddLocalTensor(args_, tensor, TensorArgType::Input);
        if (!args_.has_error) ++counts_.tensor_args_added;
    }

    template <typename Thunk>
    PA_DEVICE void AddGmInput(Thunk thunk) {
        if (!Ready()) return;
        PA_GM const TensorDesc &tensor = thunk();
        AddGmTensor(args_, tensor, TensorArgType::Input);
        if (!args_.has_error) ++counts_.tensor_args_added;
    }

    template <typename Thunk>
    PA_DEVICE void AddOutputHandleInput(Thunk thunk) {
        if (!Ready()) return;
        const PaOutputHandle handle = thunk();
        AddOutputHandleTensor(args_, handle, TensorArgType::Input);
        if (!args_.has_error) ++counts_.tensor_args_added;
    }

    template <typename Thunk>
    PA_DEVICE void AddOutput(Thunk thunk) {
        if (!Ready()) return;
        const TensorCreateInfo &create_info = thunk();
        pa_scheduler::AddOutput(args_, create_info);
        if (!args_.has_error) ++counts_.tensor_args_added;
    }

    template <typename Thunk>
    PA_DEVICE void AddLocalInout(Thunk thunk) {
        if (!Ready()) return;
        const TensorDesc &tensor = thunk();
        AddLocalTensor(args_, tensor, TensorArgType::Inout);
        if (!args_.has_error) ++counts_.tensor_args_added;
    }

    template <typename Thunk>
    PA_DEVICE void AddGmInout(Thunk thunk) {
        if (!Ready()) return;
        PA_GM const TensorDesc &tensor = thunk();
        AddGmTensor(args_, tensor, TensorArgType::Inout);
        if (!args_.has_error) ++counts_.tensor_args_added;
    }

    template <typename Thunk>
    PA_DEVICE void AddOutputHandleInout(Thunk thunk) {
        if (!Ready()) return;
        const PaOutputHandle handle = thunk();
        AddOutputHandleTensor(args_, handle, TensorArgType::Inout);
        if (!args_.has_error) ++counts_.tensor_args_added;
    }

    template <typename Thunk>
    PA_DEVICE void AddScalar(Thunk thunk) {
        if (!Ready()) return;
        pa_scheduler::AddScalar(args_, thunk());
        if (!args_.has_error) ++counts_.scalar_args_added;
    }

    PA_DEVICE bool Valid() const { return begin_calls_ == 1 && !args_.has_error; }
    PA_DEVICE const CallbackSubmitBuildCounts &Counts() const { return counts_; }

private:
    PA_DEVICE bool Ready() {
        if (begin_calls_ == 1 && !args_.has_error) return true;
        args_.has_error = true;
        return false;
    }

    TaskArgs &args_;
    TaskKind kind_;
    uint32_t begin_calls_;
    CallbackSubmitBuildCounts counts_;
};

PA_DEVICE void AcceptTaskOutputs(
    PaOrchestrationState &orch, TaskKind kind, const OrchestrationTaskOutputs &outputs
) {
    // private 保存本 worker payload descriptor 指针；shared 保存
    // (producer_task_id, output_slot) 符号。上层五阶段 orchestration 只消费
    // PaOutputHandle，不需要在每个业务字段处分散模式宏。
    switch (kind) {
        case TaskKind::Alloc:
            orch.accumulated_output = OutputHandleAt(outputs, 0);
            orch.accumulated_sum = OutputHandleAt(outputs, 1);
            orch.accumulated_max = OutputHandleAt(outputs, 2);
            break;
        case TaskKind::Qk:
            orch.qk_scores = OutputHandleAt(outputs, 0);
            break;
        case TaskKind::Sf:
            orch.sf_probs = OutputHandleAt(outputs, 0);
            orch.sf_max = OutputHandleAt(outputs, 1);
            orch.sf_sum = OutputHandleAt(outputs, 2);
            break;
        case TaskKind::Pv:
            orch.pv_output = OutputHandleAt(outputs, 0);
            break;
        default:
            // UP 只更新既有 Inout，没有新 Output descriptor 需要传给下一阶段。
            break;
    }
}

PA_DEVICE void ResetTensorMap(PA_GM TensorMap &map) {
    // TensorMap 完全属于当前 worker，private ring 的游标和计数都不需要
    // atomic。entry/ABI 保留区保持惰性，未落在 [head, tail) 的槽不可见。
    map.alive_floor = 0;
    map.cleaned_upto = 0;
    map.live_count = 0;
    map.high_water = 0;
    for (uint32_t index = 0; index < kMapBuckets; ++index) {
        map.bucket_heads[index] = 0;
        map.bucket_tails[index] = 0;
    }
    for (uint32_t index = 0; index < kTaskWindow; ++index) {
        map.task_entry_counts[index] = 0;
    }
}

PA_DEVICE uint32_t TensorMapHash(uint64_t address) {
    address *= 0x9E3779B97F4A7C15ULL;
    return static_cast<uint32_t>(address >> (64 - kMapBucketShift)) & kMapBucketMask;
}

template <typename TensorReference>
PA_DEVICE void TensorByteRange(const TensorReference &tensor, uint64_t &address, uint64_t &lo, uint64_t &hi) {
    // identity 先按 backing buffer 地址分桶，再用半开字节区间 [lo, hi) 判断 view
    // 是否重叠。连续 tensor 由 shape 现算 extent，非连续 tensor 使用缓存 extent。
    const uint64_t element_size = ElementSize(tensor.dtype);
    address = tensor.buffer_addr;
    lo = tensor.start_offset * element_size;
    uint64_t extent;
    if (tensor.is_contiguous) {
        extent = 1;
        for (uint32_t index = 0; index < tensor.ndims; ++index) {
            extent *= tensor.shapes[index];
        }
    } else {
        extent = tensor.extent_elem_cache;
    }
    hi = (tensor.start_offset + extent) * element_size;
}

PA_DEVICE uint32_t TensorMapSlotIndex(uint32_t bucket, uint64_t cursor) {
    return bucket * kMapBucketCapacity +
           (static_cast<uint32_t>(cursor) & kMapBucketSlotMask);
}

PA_DEVICE void RetireBucket(PA_GM TensorMap &map, uint32_t bucket) {
    // 同一 worker 按 task_id 单调 append，因此一个桶内的 producer 也单调
    // 不降。只需从最旧槽开始推进到第一个仍在 alive_floor 内的条目。
    // AdvanceTensorMap 不扫 128 个桶；lookup/insert 触达哪个桶，哪个桶才
    // 物理退休。逻辑 live_count 已在 AdvanceTensorMap 中精确扣减。
    uint64_t head = map.bucket_heads[bucket];
    const uint64_t tail = map.bucket_tails[bucket];
    while (head < tail) {
        PA_GM const MapEntry &entry = map.entries[TensorMapSlotIndex(bucket, head)];
        if (entry.producer >= map.alive_floor) {
            break;
        }
        ++head;
    }
    map.bucket_heads[bucket] = head;
}

PA_DEVICE void AdvanceTensorMap(PA_GM TensorMap &map, uint32_t task_id, int32_t heap_window) {
    // PrepareMap 在 Claim 后把存活下界推进到 task_id-H。这里按 producer
    // 精确扣减 logical live_count 并清空循环计数槽；桶内物理 head 留给
    // RetireBucket 惰性推进。TensorMap 与 heap 共享 H，但前者按本 worker
    // task_id 推进，后者按跨核连续 frontier 推进，二者无需同步到同一位置。
    const int32_t new_floor = static_cast<int32_t>(task_id) - heap_window;
    if (new_floor <= map.cleaned_upto) {
        if (new_floor > map.alive_floor) {
            map.alive_floor = new_floor;
        }
        return;
    }
    for (int32_t id = map.cleaned_upto; id < new_floor; ++id) {
        const uint32_t task_slot = static_cast<uint32_t>(id) & kTaskWindowMask;
        const uint32_t retired = map.task_entry_counts[task_slot];
        map.live_count -= retired;
        map.task_entry_counts[task_slot] = 0;
    }
    map.cleaned_upto = new_floor;
    map.alive_floor = new_floor;
}

template <typename TensorReference>
PA_DEVICE bool InsertTensor(PA_GM TensorMap &map, const TensorReference &tensor, int32_t producer) {
    // 每桶独立 append。先惰性退休该桶，再检查固定容量；满时不覆写旧槽、
    // 不推进 tail/计数，明确返回 false 交由 Submit 上层转成 fatal。
    uint64_t address = 0;
    uint64_t lo = 0;
    uint64_t hi = 0;
    TensorByteRange(tensor, address, lo, hi);
    const uint32_t bucket = TensorMapHash(address);
    RetireBucket(map, bucket);
    const uint64_t head = map.bucket_heads[bucket];
    const uint64_t tail = map.bucket_tails[bucket];
    if (tail - head >= kMapBucketCapacity) {
        return false;
    }

    PA_GM MapEntry &entry = map.entries[TensorMapSlotIndex(bucket, tail)];
    entry.buffer_addr = address;
    entry.lo = lo;
    entry.hi = hi;
    entry.producer = producer;
    map.bucket_tails[bucket] = tail + 1;

    const uint32_t task_slot = static_cast<uint32_t>(producer) & kTaskWindowMask;
    ++map.task_entry_counts[task_slot];
    ++map.live_count;
    if (map.live_count > map.high_water) {
        map.high_water = map.live_count;
    }
    return true;
}

template <typename TensorReference>
PA_DEVICE int32_t LookupTensor(PA_GM TensorMap &map, const TensorReference &tensor) {
    uint64_t address = 0;
    uint64_t lo = 0;
    uint64_t hi = 0;
    TensorByteRange(tensor, address, lo, hi);
    const uint32_t bucket = TensorMapHash(address);
    RetireBucket(map, bucket);
    const uint64_t head = map.bucket_heads[bucket];
    const uint64_t tail = map.bucket_tails[bucket];
    int32_t best = -1;
    // 扫描 [head,tail) 的全部合法槽而非依赖 append 顺序提前返回；同一
    // buffer 的多个历史写者中，只接受 producer>=alive_floor 的重叠条目，
    // 并取 producer 最大者，避免依赖退回旧版本。
    for (uint64_t cursor = head; cursor < tail; ++cursor) {
        PA_GM const MapEntry &entry = map.entries[TensorMapSlotIndex(bucket, cursor)];
        if (entry.producer < map.alive_floor) {
            continue;
        }
        if (entry.buffer_addr == address && lo < entry.hi && entry.lo < hi && entry.producer > best) {
            best = entry.producer;
        }
    }
    return best;
}

PA_DEVICE uint64_t TensorOwner(const TaskTensorRef &reference) {
    // CreateInfo 只会出现在 tag=Output 且在 fanin 前已被跳过；这里的输入不变量是
    // LocalTensor/GmTensor，输出为显式 owner 或 kInvalidTaskId。
    if (reference.kind == TensorRefKind::GmTensor) {
        return reference.pointer.gm_tensor->owner_task_id;
    }
    return reference.pointer.local_tensor->owner_task_id;
}

PA_DEVICE int32_t LookupTensorRef(PA_GM TensorMap &map, const TaskTensorRef &reference) {
    // 与 TensorOwner 相同，此辅助入口只接收已存在 descriptor；返回最新重叠 producer，
    // 未登记或已退休则返回 -1。
    if (reference.kind == TensorRefKind::GmTensor) {
        return LookupTensor(map, *reference.pointer.gm_tensor);
    }
    return LookupTensor(map, *reference.pointer.local_tensor);
}

PA_DEVICE void AddFanin(int32_t fanin[kMaxFanin], uint32_t &count, int32_t producer) {
    // owner 与 TensorMap lookup 可能指向同一 producer，先去重再写固定 16 槽数组；
    // Case1 的最大 fanin 为 UP 的 3，正常路径不会截断。
    if (producer < 0) {
        return;
    }
    for (uint32_t index = 0; index < count; ++index) {
        if (fanin[index] == producer) {
            return;
        }
    }
    if (count < kMaxFanin) {
        fanin[count++] = producer;
    }
}

PA_DEVICE uint32_t CollectFanin(
    PA_GM TensorMap &map, const TaskArgs &args, int32_t fanin[kMaxFanin]
) {
    // fanin 只由 winner 收集：先吸收 descriptor 的显式 owner，再对 Input/Inout
    // 查询最新重叠写者；纯 Output 尚未存在，不应成为本次 task 的输入依赖。
    uint32_t count = 0;
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const TensorArgType tag = TaskTag(args, static_cast<uint32_t>(index));
        if (tag == TensorArgType::Output) {
            continue;
        }
        const TaskTensorRef &reference = args.tensors[index];
        // Keep the two address spaces in separate control-flow arms. CCEC's
        // O2/O3 backend rejects a merged pointer phi even when both arms only
        // feed scalar field loads; this is also how PA's production helper is
        // written.
        // 分支重复是后端约束与生产写法的一部分，不应抽成一个混合地址空间指针。
        if (reference.kind == TensorRefKind::GmTensor) {
            PA_GM const TensorDesc &tensor = *reference.pointer.gm_tensor;
            const uint64_t owner = tensor.owner_task_id;
            if (owner != kInvalidTaskId) {
                AddFanin(fanin, count, static_cast<int32_t>(owner & 0xFFFFFFFFU));
            }
            if (tag == TensorArgType::Input || tag == TensorArgType::Inout) {
                AddFanin(fanin, count, LookupTensor(map, tensor));
            }
        } else {
            const TensorDesc &tensor = *reference.pointer.local_tensor;
            const uint64_t owner = tensor.owner_task_id;
            if (owner != kInvalidTaskId) {
                AddFanin(fanin, count, static_cast<int32_t>(owner & 0xFFFFFFFFU));
            }
            if (tag == TensorArgType::Input || tag == TensorArgType::Inout) {
                AddFanin(fanin, count, LookupTensor(map, tensor));
            }
        }
    }
    return count;
}

PA_DEVICE bool InsertExistingTensor(SubmitContext &context, const TaskArgs &args, int32_t index) {
    // 输入 index 来自 register_mask，故必为已有 descriptor 而非 CreateInfo；写入结果
    // 只影响 context.self 对应 worker 的 map，并把当前 task_id 登记为新的 hazard 版本。
    const TaskTensorRef &reference = args.tensors[index];
    if (reference.kind == TensorRefKind::GmTensor) {
        return InsertTensor(context.self->map, *reference.pointer.gm_tensor, context.task_id);
    }
    return InsertTensor(context.self->map, *reference.pointer.local_tensor, context.task_id);
}

PA_DEVICE bool RegisterOutputs(SubmitContext &context, const TaskArgs &args, bool include_existing) {
    // register_mask 只覆盖 Inout/OutputExisting。新 Output 已带本次 owner；现有
    // backing buffer 的新写者则必须登记到本 worker TensorMap，供后继 task 查 hazard。
    if (!include_existing) {
        return true;
    }
    uint32_t register_mask = context.register_mask;
    for (uint32_t index = 0; register_mask != 0; ++index, register_mask >>= 1) {
        if ((register_mask & 1U) != 0) {
            if (!InsertExistingTensor(context, args, static_cast<int32_t>(index))) {
                return false;
            }
        }
    }
    return true;
}

PA_DEVICE uint64_t FrontendAlignUp(uint64_t value, uint64_t alignment) {
    // alignment 在本模型中固定为2的幂1 KiB；返回逻辑 heap 地址，不做 ring 取模。
    return (value + alignment - 1) & ~(alignment - 1);
}

PA_DEVICE bool MaterializeTask(
    PA_GM WorkerState &worker, uint32_t task_id, const TaskArgs &args, SubmitContext &context,
    uint64_t heap_base, uint64_t heap_size
) {
    // 输入是 BeginCallbackSubmit 已绑定的 payload/context 与当前 worker.heap_next；成功输出
    // 包括本 task 的 GM TensorDesc 指针、output_bytes 和推进后的单调 heap_next。
    // 失败不得进入 slot/build 流程，由上层设置 fatal 并终止该 worker 回放。
    // compete-first 路径在这里已完成 Claim，因此不能再把“尚未 Claim”当作
    // 这个共用 helper 的前置条件。
    if (context.payload == nullptr) {
        return false;
    }
#if PTO_FDWIC_SHARED_MAP
    if (task_id >= kMaxTasks ||
        context.shared_result.TaskId() != static_cast<int32_t>(task_id) ||
        context.shared_result.Size() != 0) {
        return false;
    }
#endif
    context.tensor_count = args.tensor_count;
    context.scalar_count = args.scalar_count;
    context.register_mask = 0;

    // DistOutputLayout leaves non-output slots lazy and writes only the sizes
    // selected by output_mask.
    // 第一次 tag 扫描同时产生 output_mask/register_mask；第二次只遍历
    // Output 位，避免读取未初始化的非输出 buffer_sizes。
    OutputLayout layout;
    layout.total_output_size = 0;
    uint32_t output_mask = 0;
    uint32_t output_count = 0;
    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const TensorArgType tag = TaskTag(args, static_cast<uint32_t>(index));
        if (tag == TensorArgType::Inout || tag == TensorArgType::OutputExisting) {
            context.register_mask |= 1U << index;
        }
        if (tag != TensorArgType::Output) {
            continue;
        }
        output_mask |= 1U << index;
        ++output_count;
        layout.buffer_sizes[index] = CreateInfoBytes(*args.tensors[index].pointer.create_info);
        layout.total_output_size += FrontendAlignUp(layout.buffer_sizes[index], kOutputAlignment);
    }
#if PTO_FDWIC_SHARED_MAP
    if (output_count > kSharedOutputMaxPerTask) {
        return false;
    }
#else
    (void)output_count;
#endif

    uint64_t task_base = FrontendAlignUp(worker.heap_next, kOutputAlignment);
    const uint64_t total = layout.total_output_size;
    if (total > heap_size || (total != 0 && heap_base == 0)) {
        return false;
    }
    if (total != 0 && (task_base % heap_size) + total > heap_size) {
        // 单个 task 的输出必须物理连续；若跨 ring 尾部则把逻辑 task_base 推到
        // 下一圈起点。heap_next 仍保持单调，不在这里取模。
        task_base = (task_base / heap_size + 1) * heap_size;
    }

    uint64_t output_offset = 0;
    // 各 Output 在同一 task_base 内按参数顺序排布；result 只收集 Output，索引与
    // TaskArgs 中非输出槽无关，而 payload 仍按原参数 index 保存 descriptor。
    for (int32_t index = 0; output_mask != 0; ++index, output_mask >>= 1) {
        if ((output_mask & 1U) == 0) {
            continue;
        }
        const uint64_t physical = (task_base + output_offset) % heap_size;
        PA_GM TensorDesc &tensor = context.payload->tensors[index];
        if (!InitTensorFromCreateInfo(
                tensor, *args.tensors[index].pointer.create_info, heap_base + physical, layout.buffer_sizes[index]
            )) {
            return false;
        }
        tensor.owner_task_id = task_id;
        const uint32_t output_ordinal = context.result.count;
#if PTO_FDWIC_SHARED_MAP
        if (!context.shared_result.AddOutputRef(
                static_cast<int32_t>(task_id),
                static_cast<int16_t>(output_ordinal)
            )) {
            return false;
        }
#endif
        context.result.tensors[output_ordinal] = &tensor;
        ++context.result.count;
        output_offset += FrontendAlignUp(layout.buffer_sizes[index], kOutputAlignment);
    }
    worker.heap_next = task_base + total;
    context.output_bytes = total;
    return true;
}

PA_DEVICE void CopyTensorFromRef(PA_GM TensorDesc &destination, const TaskTensorRef &reference) {
    // slot 必须拥有 descriptor 快照，不能保存指向 orchestration 栈对象的引用；
    // 按 byte volatile copy 同时兼容 local/GM 源并保留真实 128-byte 搬运量。
    PA_GM volatile uint8_t *destination_bytes = reinterpret_cast<PA_GM volatile uint8_t *>(&destination);
    if (reference.kind == TensorRefKind::GmTensor) {
        PA_GM const volatile uint8_t *source_bytes =
            reinterpret_cast<PA_GM const volatile uint8_t *>(reference.pointer.gm_tensor);
        for (uint32_t byte = 0; byte < sizeof(TensorDesc); ++byte) {
            destination_bytes[byte] = source_bytes[byte];
        }
        return;
    }
    const volatile uint8_t *source_bytes =
        reinterpret_cast<const volatile uint8_t *>(reference.pointer.local_tensor);
    for (uint32_t byte = 0; byte < sizeof(TensorDesc); ++byte) {
        destination_bytes[byte] = source_bytes[byte];
    }
}

PA_DEVICE void CopyGmTensor(PA_GM TensorDesc &destination, PA_GM const TensorDesc &source) {
    // 新 Output 的源 descriptor 已位于 GM payload；单独入口避免把 GM 指针误走
    // local 地址空间分支，输出仍是 slot 内独立副本。
    PA_GM volatile uint8_t *destination_bytes = reinterpret_cast<PA_GM volatile uint8_t *>(&destination);
    PA_GM const volatile uint8_t *source_bytes = reinterpret_cast<PA_GM const volatile uint8_t *>(&source);
    for (uint32_t byte = 0; byte < sizeof(TensorDesc); ++byte) {
        destination_bytes[byte] = source_bytes[byte];
    }
}

PA_DEVICE void PopulateSlotPayload(
    PA_GM LocalSlot &slot, const TaskArgs &args, const SubmitContext &context, const int32_t fanin[kMaxFanin],
    uint32_t fanin_count, int32_t sub_block_id, bool is_multicore, int32_t won_block, int32_t won_slot
) {
    // winner 将活动 descriptor/scalar 复制进私有 slot，dispatch args 指向 slot 内
    // 副本而非 orchestration 临时对象；fanin 随 slot 保存，kernel 执行前逐 flag 检查。
    slot.tensor_count = context.tensor_count;
    slot.scalar_count = context.scalar_count;
    for (int32_t index = 0; index < context.tensor_count; ++index) {
        if (TaskTag(args, static_cast<uint32_t>(index)) == TensorArgType::Output) {
            CopyGmTensor(slot.tensors[index], context.payload->tensors[index]);
#if PTO_FDWIC_SHARED_MAP
        } else if (IsSharedOutputReference(args.tensors[index])) {
            // scheduler resolver 已在 WinnerBuild 内校验 published 与 plain-view
            // 契约，并把共享 descriptor 拷入当前 task 的 payload scratch。
            // slot builder 只复用该快照，不保存符号，也不增加 RingSlot ABI。
            CopyGmTensor(slot.tensors[index], context.payload->tensors[index]);
#endif
        } else {
            CopyTensorFromRef(slot.tensors[index], args.tensors[index]);
        }
        slot.args[index] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&slot.tensors[index]));
    }
    for (int32_t index = 0; index < context.scalar_count; ++index) {
        slot.scalars[index] = args.scalars[index];
        slot.args[context.tensor_count + index] = args.scalars[index];
    }

    PA_GM PaLocalContext &local =
        *reinterpret_cast<PA_GM PaLocalContext *>(&slot.local_context[0]);
    // standalone 每个 task 只由一个 lane kernel 执行，故 block_index/count 固定0/1；
    // async completion 未启用，task_token 保持 invalid，与 PA 普通同步 slot 一致。
    local.block_index = 0;
    local.block_count = 1;
    local.async.completion_count = 0;
    local.async.completion_error_code = 0;
    local.async.completion_entries = 0;
    local.async.completion_capacity = 0;
    local.async.task_token = kInvalidTaskId;
    slot.global_context = static_cast<uint32_t>(sub_block_id);
    slot.args[kSpmdLocalContextIndex] =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&slot.local_context[0]));
    slot.args[kSpmdGlobalContextIndex] =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&slot.global_context));
    slot.fanin_count = fanin_count;
    // fanin 数组只复制有效前缀；执行端以 fanin_count 为边界，未使用尾部保持惰性。
    for (uint32_t index = 0; index < fanin_count; ++index) {
        slot.fanin[index] = fanin[index];
    }
    slot.is_multicore = is_multicore;
    slot.won_block = won_block;
    slot.won_slot = won_slot;
}

PA_DEVICE void BuildSlotPayload(
    PA_GM LocalSlot &slot, uint32_t task_id, uint32_t function_id, uint64_t function_address, const TaskArgs &args,
    const SubmitContext &context, const int32_t fanin[kMaxFanin], uint32_t fanin_count, int32_t sub_block_id = 0,
    bool is_multicore = false, int32_t won_block = -1, int32_t won_slot = -1
) {
    // Match build_ring_slot_from_submit ordering: publish the header first,
    // then copy the active descriptors/scalars and construct dispatch payload.
    // slot 仅由所属 worker 消费，这里的写入次序用于复刻真实构建成本与
    // 状态机；跨核可见性由 task completion 的 flag/vend 协议承担。
    slot.occupied = true;
    slot.task_id = task_id;
    slot.kind = function_id;
    slot.function_address = function_address;
    slot.built = 1;
    PopulateSlotPayload(
        slot, args, context, fanin, fanin_count, sub_block_id, is_multicore, won_block, won_slot
    );
}

// Compatibility overload for a core that has already populated the slot
// header before calling the PA frontend.
// 该入口只补 payload，不改变既有 task/function 头；输出不变量与完整
// BuildSlotPayload 相同，均得到 built 且可由 DrainReady 检查 fanin 的私有 slot。
PA_DEVICE void BuildSlotPayload(
    PA_GM LocalSlot &slot, const TaskArgs &args, const SubmitContext &context, const int32_t fanin[kMaxFanin],
    uint32_t fanin_count
) {
    slot.built = 1;
    PopulateSlotPayload(slot, args, context, fanin, fanin_count, 0, false, -1, -1);
}

}  // namespace pa_scheduler

#endif  // PA_SCHEDULER_COMMON_PA_FRONTEND_H
