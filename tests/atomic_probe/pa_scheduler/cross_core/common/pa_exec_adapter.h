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

#ifndef PA_SCHEDULER_CROSS_CORE_PA_EXEC_ADAPTER_H
#define PA_SCHEDULER_CROSS_CORE_PA_EXEC_ADAPTER_H

// 正式接线由 CPU/CCEC scheduler TU 在定义地址空间修饰后 include；独立
// host 编译门槛也可使用下面的无地址空间窄默认值。
#ifndef PA_DEVICE
#define PA_CROSS_CORE_ADAPTER_DEFINED_PA_DEVICE 1
#define PA_DEVICE inline
#endif
#ifndef PA_GM
#define PA_CROSS_CORE_ADAPTER_DEFINED_PA_GM 1
#define PA_GM
#endif
#ifndef PA_DEVICE_NOINLINE
#define PA_CROSS_CORE_ADAPTER_UNDEFINED_PA_DEVICE_NOINLINE 1
#endif

#include "pa_frontend.h"
#include "shared_exec_protocol.h"

namespace pa_scheduler::cross_core {

// portable 协议按完整 128B descriptor 搬运。这里直接锁定 standalone
// 的真实 PA TensorDesc ABI，而不是只检查一个“足够大”的缓冲区。
static_assert(
    sizeof(TensorDesc) == kExecTensorDescBytes,
    "portable execution payload must carry one complete PA TensorDesc"
);
static_assert(
    alignof(TensorDesc) == alignof(uint64_t) &&
        kExecTensorDescBytes % alignof(TensorDesc) == 0,
    "PA TensorDesc alignment must support exact uint64-word transport"
);
static_assert(
    kMaxTaskTensors == kExecMaxTensors &&
        kMaxTaskScalars == kExecMaxScalars &&
        kMaxFanin == kExecMaxFanin,
    "PA dispatch capacities must match the portable payload contract"
);
static_assert(
    sizeof(PaLocalContext) == kExecLocalContextBytes &&
        sizeof(PaGlobalContext) == kExecGlobalContextBytes,
    "executor context storage must match the PA dispatch ABI"
);

struct PaExecRoute {
    uint32_t function_id;
    ExecEngineClass engine_class;
};

// PA kernel 的 dispatch ABI 不只是“数量不超过上限”，而是每种 task
// 都有唯一的 tensor/scalar/fanin 三元组。把这份合同放在 device adapter
// 内，构建侧和最终发射侧共用；host oracle 不能替代 device fail-closed。
struct PaExecShape {
    uint16_t tensor_count;
    uint16_t scalar_count;
    uint16_t fanin_count;
};

PA_DEVICE bool ResolvePaExecShape(
    TaskKind kind, PaExecShape &shape
) {
    switch (kind) {
        case TaskKind::Qk:
            shape = PaExecShape{4, 2, 0};
            return true;
        case TaskKind::Sf:
            shape = PaExecShape{4, 3, 1};
            return true;
        case TaskKind::Pv:
            shape = PaExecShape{4, 2, 1};
            return true;
        case TaskKind::Up:
            shape = PaExecShape{7, 2, 3};
            return true;
        case TaskKind::Alloc:
        case TaskKind::Count:
            return false;
    }
    return false;
}

PA_DEVICE bool PaExecShapeMatches(
    TaskKind kind, uint32_t tensor_count,
    uint32_t scalar_count, uint32_t fanin_count
) {
    PaExecShape expected{};
    return ResolvePaExecShape(kind, expected) &&
           tensor_count == expected.tensor_count &&
           scalar_count == expected.scalar_count &&
           fanin_count == expected.fanin_count;
}

PA_DEVICE bool PaTaskKindFromExecFunction(
    uint32_t function_id, TaskKind &kind
) {
    switch (function_id) {
        case 0: kind = TaskKind::Qk; return true;
        case 1: kind = TaskKind::Sf; return true;
        case 2: kind = TaskKind::Pv; return true;
        case 3: kind = TaskKind::Up; return true;
        default: kind = TaskKind::Count; return false;
    }
}

// PA adapter 负责 function-id 与合法 engine 的业务映射；portable 协议
// 不认识 QK/SF/PV/UP。Alloc 没有 kernel，Joint 在第一版显式拒绝。
PA_DEVICE bool ResolvePaExecRoute(
    TaskKind kind, int32_t function_id, PaExecRoute &route
) {
    switch (kind) {
        case TaskKind::Qk:
            route = PaExecRoute{0, ExecEngineClass::Aic};
            break;
        case TaskKind::Sf:
            route = PaExecRoute{1, ExecEngineClass::Aiv};
            break;
        case TaskKind::Pv:
            route = PaExecRoute{2, ExecEngineClass::Aic};
            break;
        case TaskKind::Up:
            route = PaExecRoute{3, ExecEngineClass::Aiv};
            break;
        case TaskKind::Alloc:
        case TaskKind::Count:
            return false;
    }
    return function_id >= 0 &&
           static_cast<uint32_t>(function_id) == route.function_id &&
           route.engine_class != ExecEngineClass::Joint;
}

union PaExecTensorAddress {
    const TensorDesc *local_tensor;
    PA_GM const TensorDesc *gm_tensor;
};
static_assert(
    sizeof(PaExecTensorAddress) == sizeof(uint64_t),
    "resolved PA descriptor address must fit one compact pointer slot"
);

// Source 不建立 4 KiB descriptor staging：预检只把每个 active tensor
// 解析为一个 descriptor 地址，并用 gm_tensor_mask 区分地址空间；Pack
// 随后按 64-bit word 直接读取不可变 descriptor。scalar/fanin 很小且
// builder 后续可能复用 TaskArgs/SubmitContext，因此仍按值冻结。
struct PaExecPayloadSource {
    PaExecTensorAddress tensors[kExecMaxTensors];
    uint64_t scalars[kExecMaxScalars];
    int32_t fanin[kExecMaxFanin];
    uint32_t gm_tensor_mask;
    uint32_t reference_mask;

    PA_DEVICE uint64_t TensorWord(uint32_t tensor, uint32_t word) const {
        if ((gm_tensor_mask & (uint32_t{1} << tensor)) != 0) {
            PA_GM const volatile uint64_t *words =
                reinterpret_cast<PA_GM const volatile uint64_t *>(
                    tensors[tensor].gm_tensor
                );
            return words[word];
        }
        const volatile uint64_t *words =
            reinterpret_cast<const volatile uint64_t *>(
                tensors[tensor].local_tensor
            );
        return words[word];
    }

    PA_DEVICE uint64_t TensorReference(uint32_t tensor) const {
        if ((reference_mask & (uint32_t{1} << tensor)) == 0 ||
            (gm_tensor_mask & (uint32_t{1} << tensor)) == 0) {
            return 0;
        }
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
            tensors[tensor].gm_tensor
        ));
    }

    PA_DEVICE uint64_t Scalar(uint32_t scalar) const {
        return scalars[scalar];
    }

    PA_DEVICE int32_t Fanin(uint32_t edge) const {
        return fanin[edge];
    }
};
static_assert(
    sizeof(PaExecPayloadSource) <= 512,
    "PA execution source must stay compact and must not mirror LocalSlot/payload"
);

#if PTO_FDWIC_SHARED_MAP
template <typename Ops, typename Observer>
PA_DEVICE bool ResolvePaExecPayloadSourceAfterFanin(
    PA_GM SchedulerState &state, const TaskArgs &args,
    const SubmitContext &context, uint32_t task_id,
    PaExecPayloadSource &source, Observer &observer
) {
    if (task_id >= kMaxTasks || context.task_id < 0 ||
        static_cast<uint32_t>(context.task_id) != task_id ||
        args.has_error || args.tensor_count < 0 ||
        args.tensor_count > static_cast<int32_t>(kExecMaxTensors) ||
        args.scalar_count < 0 ||
        args.scalar_count > static_cast<int32_t>(kExecMaxScalars) ||
        context.tensor_count != args.tensor_count ||
        context.scalar_count != args.scalar_count ||
        context.fanin_count < 0 ||
        context.fanin_count > static_cast<int32_t>(kExecMaxFanin) ||
        context.result.task_id != task_id || context.payload == nullptr) {
        return false;
    }

    source.gm_tensor_mask = 0;
    source.reference_mask = 0;

    for (int32_t index = 0; index < args.tensor_count; ++index) {
        const uint32_t tensor_index = static_cast<uint32_t>(index);
        const TensorArgType tag = TaskTag(args, tensor_index);
        const TaskTensorRef &reference = args.tensors[tensor_index];
        if (tag == TensorArgType::Output) {
            if (reference.kind != TensorRefKind::CreateInfo) {
                return false;
            }
            // Materialize 已把 fresh Output 放进本 task 的 GM payload。
            source.tensors[tensor_index].gm_tensor =
                &context.payload->tensors[tensor_index];
            source.gm_tensor_mask |= uint32_t{1} << tensor_index;
        } else if (reference.kind == TensorRefKind::SharedOutputRef) {
            const FdwicOutputRef output_ref =
                SharedOutputReference(reference);
            if (!IsPlainSharedOutputRef(output_ref) ||
                output_ref.producer_task_id < 0 ||
                static_cast<uint32_t>(output_ref.producer_task_id) >=
                    task_id) {
                return false;
            }
            PA_GM SharedOutputCell &output_cell =
                state.shared_map.shared_outputs[
                    static_cast<uint32_t>(
                        output_ref.producer_task_id
                    )
                ];
            const uint32_t output_slot =
                static_cast<uint32_t>(output_ref.output_slot);
            // AfterFanin 是硬前置合同：strict insert completion chain 已经
            // 证明 producer 的 descriptor flush 与 published 先于当前
            // Build；latest-writer 解析又验证了该 symbol 的 writer 链。
            // adapter 不得再做返回型 published load，只对 descriptor 做
            // 一次 invalidate；fresh descriptor 此后不可变，Pack 可直接
            // 逐 word 读取，不把 SharedOutputRef 带给 executor。
            observer.InvalidateBuildDescriptor(
                &output_cell.tensors[output_slot],
                sizeof(TensorDesc), task_id
            );
            source.tensors[tensor_index].gm_tensor =
                &output_cell.tensors[output_slot];
            source.gm_tensor_mask |= uint32_t{1} << tensor_index;
        } else if (reference.kind == TensorRefKind::GmTensor) {
            if (reference.pointer.gm_tensor == nullptr) {
                return false;
            }
            source.tensors[tensor_index].gm_tensor =
                reference.pointer.gm_tensor;
            source.gm_tensor_mask |= uint32_t{1} << tensor_index;
        } else if (reference.kind == TensorRefKind::LocalTensor) {
            if (reference.pointer.local_tensor == nullptr) {
                return false;
            }
            source.tensors[tensor_index].local_tensor =
                reference.pointer.local_tensor;
        } else {
            return false;
        }
    }
    for (int32_t index = 0; index < args.scalar_count; ++index) {
        source.scalars[static_cast<uint32_t>(index)] =
            args.scalars[index];
    }
    for (int32_t edge = 0; edge < context.fanin_count; ++edge) {
        const int32_t producer = context.fanin[edge];
        if (producer < 0 || static_cast<uint32_t>(producer) >= task_id) {
            return false;
        }
        source.fanin[static_cast<uint32_t>(edge)] = producer;
    }
    return true;
}

template <typename Ops>
PA_DEVICE bool ResolvePaExecPayloadSourceAfterFanin(
    PA_GM SchedulerState &state, const TaskArgs &args,
    const SubmitContext &context, uint32_t task_id,
    PaExecPayloadSource &source
) {
    DirectExecObserver<Ops> observer{};
    return ResolvePaExecPayloadSourceAfterFanin<Ops>(
        state, args, context, task_id, source, observer
    );
}

PA_DEVICE bool MakePaExecPayloadSpec(
    uint32_t task_id, TaskKind kind, int32_t function_id,
    uint64_t function_address, PA_GM const WorkerState &worker,
    const TaskArgs &args, const SubmitContext &context,
    ExecPayloadSpec &spec
) {
    PaExecRoute route{};
    if (!ResolvePaExecRoute(kind, function_id, route) ||
        context.joint || context.joint_init ||
        args.launch_spec.core_num != 1 ||
        args.tensor_count < 0 || args.scalar_count < 0 ||
        context.fanin_count < 0 ||
        !PaExecShapeMatches(
            kind, static_cast<uint32_t>(args.tensor_count),
            static_cast<uint32_t>(args.scalar_count),
            static_cast<uint32_t>(context.fanin_count)
        ) ||
        context.tensor_count != args.tensor_count ||
        context.scalar_count != args.scalar_count) {
        return false;
    }
    // completion_vend 必须在 Materialize 完成后从 worker.heap_next 取值并
    // 按值冻结进 payload；executor 完成时不得读取 builder 的 WorkerState。
    spec = ExecPayloadSpec{
        task_id,
        function_address,
        worker.heap_next,
        route.function_id,
        static_cast<uint16_t>(args.tensor_count),
        static_cast<uint16_t>(args.scalar_count),
        static_cast<uint16_t>(context.fanin_count),
        route.engine_class,
        0,
        0,
        0,
        1,
        0,
    };
    ExecPayloadLayout layout{};
    return ValidateExecPayloadSpec(spec, layout);
}

template <typename Ops>
PA_DEVICE bool PreparePaExecPublicationAfterFanin(
    PA_GM SchedulerState &state, PA_GM const WorkerState &worker,
    const TaskArgs &args, const SubmitContext &context,
    uint32_t task_id, TaskKind kind, int32_t function_id,
    uint64_t function_address, ExecPayloadSpec &spec,
    PaExecPayloadSource &source
) {
    return MakePaExecPayloadSpec(
               task_id, kind, function_id, function_address,
               worker, args, context, spec
           ) &&
           ResolvePaExecPayloadSourceAfterFanin<Ops>(
               state, args, context, task_id, source
           );
}

PA_DEVICE bool BindPaExecutionTokenDispatchAfterClaim(
    PA_GM ExecutionToken &token,
    PA_GM const WorkerState &executor
) {
    // ClaimAndBindExecPayload 已经把 tensor/scalar args 重绑到本 token；
    // adapter 只补 PA Local/GlobalContext，不能为方便再扫一遍所有参数。
    if (token.control.phase != ExecTokenPhase::WaitingFanin ||
        token.control.engine_class == ExecEngineClass::Joint) {
        return false;
    }
    const ExecPayloadHeader header = ExecutionTokenHeader(token);
    PaExecRoute route{};
    TaskKind kind = TaskKind::Count;
    if (!PaTaskKindFromExecFunction(header.function_id, kind)) {
        return false;
    }
    if (!ResolvePaExecRoute(
            kind, static_cast<int32_t>(header.function_id), route
        ) ||
        !PaExecShapeMatches(
            kind, header.tensor_count,
            header.scalar_count, header.fanin_count
        ) ||
        route.engine_class != header.engine_class) {
        return false;
    }

    PA_GM PaLocalContext &local =
        *reinterpret_cast<PA_GM PaLocalContext *>(
            &token.dispatch.local_context[0]
        );
    local.block_index = 0;
    local.block_count = 1;
    local.async.completion_count = 0;
    local.async.completion_error_code = 0;
    local.async.completion_entries = 0;
    local.async.completion_capacity = 0;
    local.async.alignment_padding = 0;
    local.async.task_token = kInvalidTaskId;

    PA_GM PaGlobalContext &global =
        *reinterpret_cast<PA_GM PaGlobalContext *>(
            &token.dispatch.global_context[0]
        );
    global.sub_block_id = executor.sub_block_id;
    return token.dispatch.args[kExecDispatchLocalContextIndex] ==
               static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
                   &token.dispatch.local_context[0]
               )) &&
           token.dispatch.args[kExecDispatchGlobalContextIndex] ==
               static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
                   &token.dispatch.global_context[0]
               ));
}

// kernel 发射前最后一次核对 executor binding。协议层已经检查 immutable
// shared payload 的 header/range；inline TensorDesc 可以直接引用该
// task-indexed storage，真正可变的 Local/GlobalContext 仍必须使用执行核
// 自己的 token。该检查是 standalone 的真实执行入口合同，不是 host 诊断。
PA_DEVICE bool ValidatePaExecutionTokenDispatch(
    PA_GM const ExecutionToken &token,
    PA_GM const WorkerState &executor, TaskKind expected_kind
) {
    if (token.control.phase != ExecTokenPhase::EngineInflight ||
        expected_kind == TaskKind::Alloc ||
        expected_kind == TaskKind::Count) {
        return false;
    }
    const ExecPayloadHeader header = ExecutionTokenHeader(token);
    PA_GM const ExecPayloadStorage &payload =
        ExecutionTokenPayload(token);
    ExecPayloadLayout layout{};
    PaExecRoute route{};
    TaskKind payload_kind = TaskKind::Count;
    if (!PaTaskKindFromExecFunction(
            header.function_id, payload_kind
        ) ||
        payload_kind != expected_kind ||
        !ResolvePaExecRoute(
            payload_kind,
            static_cast<int32_t>(header.function_id), route
        ) ||
        !PaExecShapeMatches(
            payload_kind, header.tensor_count,
            header.scalar_count, header.fanin_count
        ) ||
        route.engine_class != header.engine_class ||
        route.engine_class != token.control.engine_class ||
        !ComputeExecPayloadLayout(
            header.tensor_count, header.scalar_count,
            header.fanin_count,
            header.tensor_reference_mask, layout
        ) ||
        header.payload_bytes != layout.payload_bytes ||
        token.control.payload_bytes != layout.payload_bytes ||
        token.control.payload_lines != layout.payload_lines) {
        return false;
    }

    const ExecEngineClass executor_engine =
        executor.role == CoreRole::Aic
            ? ExecEngineClass::Aic
            : ExecEngineClass::Aiv;
    if (route.engine_class != executor_engine) {
        return false;
    }
    for (uint32_t tensor = 0;
         tensor < header.tensor_count; ++tensor) {
        const uint32_t word_offset = ExecTensorPayloadWordOffset(
            tensor, header.tensor_reference_mask
        );
        const uint64_t expected_address =
            (header.tensor_reference_mask &
             (uint32_t{1} << tensor)) != 0
            ? payload.words[word_offset]
            : static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
                  &payload.words[word_offset]
              ));
        if (token.dispatch.args[tensor] != expected_address) {
            return false;
        }
    }
    for (uint32_t scalar = 0;
         scalar < header.scalar_count; ++scalar) {
        if (token.dispatch.args[header.tensor_count + scalar] !=
            payload.words[
                layout.scalar_word_offset + scalar
            ]) {
            return false;
        }
    }

    PA_GM const PaLocalContext &local =
        *reinterpret_cast<PA_GM const PaLocalContext *>(
            &token.dispatch.local_context[0]
        );
    PA_GM const PaGlobalContext &global =
        *reinterpret_cast<PA_GM const PaGlobalContext *>(
            &token.dispatch.global_context[0]
        );
    return
        token.dispatch.args[kExecDispatchLocalContextIndex] ==
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
                &token.dispatch.local_context[0]
            )) &&
        token.dispatch.args[kExecDispatchGlobalContextIndex] ==
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
                &token.dispatch.global_context[0]
            )) &&
        local.block_index == 0 && local.block_count == 1 &&
        local.async.completion_count == 0 &&
        local.async.completion_error_code == 0 &&
        local.async.completion_entries == 0 &&
        local.async.completion_capacity == 0 &&
        local.async.alignment_padding == 0 &&
        local.async.task_token == kInvalidTaskId &&
        global.sub_block_id == executor.sub_block_id;
}

template <typename Ops>
PA_DEVICE bool ExecutePaBoundKernel(
    PA_GM SchedulerState *state, PA_GM WorkerState &executor,
    PA_GM ExecutionToken &token, TaskKind expected_kind,
    uint32_t nop_count
) {
    if (!ValidatePaExecutionTokenDispatch(
            token, executor, expected_kind
        )) {
        return false;
    }
    // Ops 收到的是 executor-private token，而不是 builder 的 TaskArgs 或
    // SubmitContext。CPU/CCEC standalone 的计算体仍是受控模拟负载，但其
    // 发射入口必须显式消费这份 binding，才能证明未来真实 dispatch 的边界。
    return Ops::ExecuteBoundKernel(
        state, executor, token, expected_kind, nop_count
    );
}

template <typename Ops>
struct PaExecReadySource {
    PA_GM SchedulerState *state;
    WorkerResult *result;

    PA_DEVICE bool IsReady(int32_t producer) const {
        if (state == nullptr || result == nullptr || producer < 0 ||
            static_cast<uint32_t>(producer) >= kMaxTasks) {
            return false;
        }
        const bool ready =
            // PA_ATOMIC_DCCI_SOURCE_EXEMPT: test-only - 无 TraceContext 的 adapter/协议单测使用直接 ready source
            Ops::Load(&state->tasks[producer].flag) == 1;
        if (ready) {
            ++result->fanin_ready_loads;
        } else {
            ++result->fanin_not_ready_loads;
        }
        return ready;
    }
};

// 正式 scheduler 使用 observer 版本，把 fanin completion load 记入与
// private 路径相同的原子泳道；无 TraceContext 的协议/adapter 单测继续
// 使用上面的直接版本。
template <typename Ops, typename Observer>
struct ObservedPaExecReadySource {
    PA_GM SchedulerState *state;
    WorkerResult *result;
    Observer *observer;

    PA_DEVICE bool IsReady(int32_t producer) const {
        if (state == nullptr || result == nullptr || observer == nullptr ||
            producer < 0 ||
            static_cast<uint32_t>(producer) >= kMaxTasks) {
            return false;
        }
        const bool ready = observer->LoadFaninFlag(
            &state->tasks[producer].flag,
            static_cast<uint32_t>(producer)
        ) == 1;
        if (ready) {
            ++result->fanin_ready_loads;
        } else {
            ++result->fanin_not_ready_loads;
        }
        return ready;
    }
};

struct PaExecSynchronousEngineCompletion {
    PA_DEVICE bool IsComplete(
        PA_GM const ExecutionToken &token
    ) const {
        // standalone 的 ExecuteKernel 只在 AIC/AIV 流水线最终
        // wait 返回后才退出；因此 helper 被调用时 engine 已完成。
        // 未来接 try_wait 时必须替换该 source，不能沿用 true。
        return token.control.phase == ExecTokenPhase::EngineInflight;
    }
};

template <typename Ops>
struct PaExecCompletionSink {
    PA_GM SchedulerState *state;

    PA_DEVICE bool PublishVend(uint32_t task_id, uint64_t vend) {
        if (state == nullptr || task_id >= kMaxTasks ||
            vend % kOutputAlignment != 0) {
            return false;
        }
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: test-only - 无 TraceContext 的 adapter/协议单测使用直接 completion sink
        (void)Ops::Exchange(&state->tasks[task_id].vend, vend);
        return true;
    }

    PA_DEVICE bool PublishFlag(uint32_t task_id) {
        if (state == nullptr || task_id >= kMaxTasks) {
            return false;
        }
        // 与原 CompleteTask 相同：vend 先发布，store barrier 后再发布 flag。
        Ops::StoreBarrier();
        // PA_ATOMIC_DCCI_SOURCE_EXEMPT: test-only - 无 TraceContext 的 adapter/协议单测使用直接 completion sink
        return Ops::Exchange(&state->tasks[task_id].flag, int64_t{1}) == 0;
    }
};


template <typename Ops, typename Observer>
struct ObservedPaExecCompletionSink {
    PA_GM SchedulerState *state;
    Observer *observer;

    PA_DEVICE bool PublishVend(uint32_t task_id, uint64_t vend) {
        if (state == nullptr || observer == nullptr ||
            task_id >= kMaxTasks || vend % kOutputAlignment != 0) {
            return false;
        }
        (void)observer->PublishCompletionVend(
            &state->tasks[task_id].vend, vend, task_id
        );
        return true;
    }

    PA_DEVICE bool PublishFlag(uint32_t task_id) {
        if (state == nullptr || observer == nullptr ||
            task_id >= kMaxTasks) {
            return false;
        }
        Ops::StoreBarrier();
        return observer->PublishCompletionFlag(
                   &state->tasks[task_id].flag,
                   int64_t{1}, task_id
               ) == 0;
    }
};
#endif  // PTO_FDWIC_SHARED_MAP

}  // namespace pa_scheduler::cross_core

#if defined(PA_CROSS_CORE_ADAPTER_UNDEFINED_PA_DEVICE_NOINLINE)
#undef PA_DEVICE_NOINLINE
#undef PA_CROSS_CORE_ADAPTER_UNDEFINED_PA_DEVICE_NOINLINE
#endif
#if defined(PA_CROSS_CORE_ADAPTER_DEFINED_PA_GM)
#undef PA_GM
#undef PA_CROSS_CORE_ADAPTER_DEFINED_PA_GM
#endif
#if defined(PA_CROSS_CORE_ADAPTER_DEFINED_PA_DEVICE)
#undef PA_DEVICE
#undef PA_CROSS_CORE_ADAPTER_DEFINED_PA_DEVICE
#endif

#endif  // PA_SCHEDULER_CROSS_CORE_PA_EXEC_ADAPTER_H
