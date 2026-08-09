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

#pragma once

#if PTO_FDWIC_SCHEDULER_MODE == 1 || PTO_FDWIC_SCHEDULER_MODE == 2 || PTO_FDWIC_SCHEDULER_MODE == 3 || \
    PTO_FDWIC_SCHEDULER_MODE == 4

#include "dist_engine/aicore/cross_core_kernel_classification.h"
#include "dist_engine/aicore/tensor_map_common.h"

namespace {

using fdwic::cross_core::CrossCoreOutputCell;
using fdwic::cross_core::CrossMapValue;
using fdwic::cross_core::ExecAcquireResult;
using fdwic::cross_core::ExecBuildReserveResult;
using fdwic::cross_core::ExecBuildResult;
using fdwic::cross_core::ExecEngineClass;
using fdwic::cross_core::ExecPhase;
using fdwic::cross_core::ExecToken;
using fdwic::cross_core::HeapReservation;
using fdwic::cross_core::HeapReserveResult;
using fdwic::cross_core::MapAppendResult;
using fdwic::cross_core::MapTurnResult;
using fdwic::cross_core::OutputAcquireResult;
using fdwic::cross_core::OutputPublishResult;
using fdwic::cross_core::SharedExecCell;

PTO_DEVICE_FUNC __gm__ CrossCoreRuntimeState &dist_cross_core_runtime_state() {
#if PTO_FDWIC_SCHEDULER_MODE == 1
    return g_dist.cross_core_ordinary;
#elif PTO_FDWIC_SCHEDULER_MODE == 2
    return g_dist.cross_core_dag.runtime;
#elif PTO_FDWIC_SCHEDULER_MODE == 3
    return g_dist.simt_cross_core_ordinary.runtime;
#else
    return g_dist.simt_cross_core_dag.runtime;
#endif
}

PTO_DEVICE_FUNC bool dist_cross_core_fail(int32_t task_id, int32_t error_code) {
    __gm__ DistCore *self = g_self;
    const uint32_t owner = self != nullptr && self->core_idx >= 0 ? static_cast<uint32_t>(self->core_idx) : 0U;
    (void)fdwic::cross_core::PublishExecFatal<DistCrossCoreAicoreOps>(
        dist_cross_core_runtime_state().fatal, fdwic::cross_core::ExecFatalReason::InvalidBuildInput,
        task_id >= 0 ? static_cast<uint32_t>(task_id) : UINT32_MAX, owner
    );
    set_fatal_code(error_code);
    if (self != nullptr) self->local_index = kFlagCap;
    return false;
}

PTO_DEVICE_FUNC bool dist_cross_core_task_valid(const DistSubmitCtx &ctx) {
    return ctx.self != nullptr && ctx.self->core_idx >= 0 &&
           static_cast<uint32_t>(ctx.self->core_idx) <= fdwic::cross_core::kExecMaxOwner && ctx.task_id >= 0 &&
           static_cast<uint32_t>(ctx.task_id) < kFdwicCrossCoreTaskCapacity && ctx.payload != nullptr;
}

PTO_DEVICE_FUNC bool dist_cross_core_create_info_bytes(const TensorCreateInfo &info, uint64_t &bytes) {
    bytes = 0;
    if (info.ndims == 0 || info.ndims > MAX_TENSOR_DIMS ||
        static_cast<uint32_t>(info.dtype) >= static_cast<uint32_t>(DataType::DATA_TYPE_NUM) || info.has_initial_value ||
        info.start_offset != 0 || !info.is_contiguous || info.__pad_flags__ != 0) {
        return false;
    }
    uint32_t elements = 1;
    for (uint32_t dimension = 0; dimension < info.ndims; ++dimension) {
        const uint32_t extent = info.shapes[dimension];
        if (extent == 0 || elements > UINT32_MAX / extent) return false;
        elements *= extent;
    }
    const uint64_t element_bytes = get_element_size(info.dtype);
    if (element_bytes == 0) return false;
    bytes = static_cast<uint64_t>(elements) * element_bytes;
    return true;
}

PTO_DEVICE_FUNC bool dist_cross_core_plan_outputs(
    const L0TaskArgs &args, DistOutputLayout &layout, uint32_t &register_mask, uint32_t &output_mask,
    uint32_t &output_count
) {
    layout.total_output_size = 0;
    register_mask = 0;
    output_mask = 0;
    output_count = 0;
    if (args.tensor_count() < 0 || args.tensor_count() > MAX_TENSOR_ARGS || args.scalar_count() < 0 ||
        args.scalar_count() > MAX_SCALAR_ARGS || args.has_error || args.explicit_dep_count() > kMaxFanin) {
        return false;
    }
    for (int32_t index = 0; index < args.tensor_count(); ++index) {
        const int32_t raw_tag = static_cast<int32_t>(args.tag(index));
        if (raw_tag < static_cast<int32_t>(TensorArgType::INPUT) ||
            raw_tag > static_cast<int32_t>(TensorArgType::NO_DEP)) {
            return false;
        }
        const TensorArgType tag = args.tag(index);
        if (tag != TensorArgType::OUTPUT) {
            if (args.tensor(index).tensor_from_shared_output()) {
                const FdwicOutputRef ref = args.tensor(index).shared_output_ref();
                if (!fdwic_plain_output_ref(ref) || ref.producer_task_id < 0 || ref.output_slot < 0 ||
                    static_cast<uint32_t>(ref.producer_task_id) >= kFdwicCrossCoreTaskCapacity ||
                    static_cast<uint32_t>(ref.output_slot) >= MAX_TENSOR_ARGS) {
                    return false;
                }
                if (tag == TensorArgType::INOUT || tag == TensorArgType::OUTPUT_EXISTING) {
                    register_mask |= uint32_t{1} << static_cast<uint32_t>(index);
                }
                continue;
            }
            if (!args.tensor(index).has_existing_tensor()) return false;
            if ((tag == TensorArgType::INOUT || tag == TensorArgType::OUTPUT_EXISTING) &&
                !dist_submit_tensor_uses_manual_dependency(args, index)) {
                register_mask |= uint32_t{1} << static_cast<uint32_t>(index);
            }
            continue;
        }
        if (!args.tensor(index).has_create_info()) return false;
        uint64_t bytes = 0;
        if (!dist_cross_core_create_info_bytes(args.tensor(index).create_info(), bytes) ||
            bytes > UINT64_MAX - (PTO2_PACKED_OUTPUT_ALIGN - 1U)) {
            return false;
        }
        const uint64_t aligned = PTO2_ALIGN_UP(bytes, PTO2_PACKED_OUTPUT_ALIGN);
        if (aligned < bytes || layout.total_output_size > UINT64_MAX - aligned) return false;
        layout.buffer_sizes[index] = bytes;
        layout.total_output_size += aligned;
        output_mask |= uint32_t{1} << static_cast<uint32_t>(index);
        ++output_count;
    }
    return output_count <= fdwic::cross_core::kOutputMaxDescriptors;
}

PTO_DEVICE_FUNC bool dist_cross_core_copy_existing_tensor(
    __gm__ Tensor &destination, const L0TaskArgs &args, int32_t index, DistSubmitCtx &ctx
) {
    if (args.tensor(index).tensor_from_shared_output()) {
        const FdwicOutputRef ref = args.tensor(index).shared_output_ref();
        if (!fdwic_plain_output_ref(ref) || ref.producer_task_id < 0 || ref.producer_task_id >= ctx.task_id ||
            ref.output_slot < 0 || static_cast<uint32_t>(ref.output_slot) >= MAX_TENSOR_ARGS) {
            return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
        }
        __gm__ CrossCoreOutputCell<Tensor> &producer_outputs =
            dist_cross_core_runtime_state().outputs[static_cast<uint32_t>(ref.producer_task_id)];
        uint32_t polls = 0;
        uint32_t producer_output_count = 0;
        while (true) {
            const OutputAcquireResult result = fdwic::cross_core::AcquirePublishedTaskOutputs<DistCrossCoreAicoreOps>(
                producer_outputs, static_cast<uint32_t>(ref.producer_task_id), producer_output_count,
                dist_cross_core_runtime_state().fatal
            );
            if (result == OutputAcquireResult::Acquired) break;
            if (result != OutputAcquireResult::NotPublished) {
                return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
            }
            SPIN_WAIT_HINT();
            if ((++polls & 1023U) == 0 && fdwic_trace_is_fatal(ctx.task_id)) return false;
        }
        if (static_cast<uint32_t>(ref.output_slot) >= producer_output_count) {
            return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
        }
        Tensor::copy(destination, producer_outputs.descriptors[static_cast<uint32_t>(ref.output_slot)]);
        if (destination.owner_task_id.raw != PTO2TaskId::make(0, static_cast<uint32_t>(ref.producer_task_id)).raw) {
            return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
        }
        return true;
    }
#if defined(__CCE_AICORE__)
    if (args.tensor(index).tensor_from_gm()) {
        Tensor::copy(destination, args.tensor(index).gm_ref());
    } else {
        Tensor::copy(destination, args.tensor(index).ref());
    }
#else
    Tensor::copy(destination, args.tensor(index).ref());
#endif
    return true;
}

PTO_DEVICE_FUNC bool dist_cross_core_materialize_builder(
    const L0TaskArgs &args, DistSubmitCtx &ctx, const DistOutputLayout &layout, uint32_t output_mask,
    uint32_t output_count, HeapReservation &reservation
) {
    if (layout.total_output_size != 0 && g_dist.heap_base == nullptr) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    const HeapReserveResult heap_result = fdwic::cross_core::ReserveOutputHeap<DistCrossCoreAicoreOps>(
        dist_cross_core_runtime_state().heap_cursor, static_cast<uint32_t>(ctx.task_id),
        static_cast<uint32_t>(ctx.self->core_idx), layout.total_output_size, static_cast<uint64_t>(g_dist.heap_size),
        reservation, dist_cross_core_runtime_state().fatal
    );
    if (heap_result != HeapReserveResult::Reserved && heap_result != HeapReserveResult::Empty) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_CAPACITY);
    }

    __gm__ CrossCoreOutputCell<Tensor> &outputs =
        dist_cross_core_runtime_state().outputs[static_cast<uint32_t>(ctx.task_id)];
    uint64_t output_offset = 0;
    uint32_t output_slot = 0;
    for (int32_t index = 0; index < ctx.tensor_count; ++index) {
        if ((output_mask & (uint32_t{1} << static_cast<uint32_t>(index))) == 0) {
            if (!dist_cross_core_copy_existing_tensor(ctx.payload->tensors[index], args, index, ctx)) return false;
            continue;
        }
        __gm__ Tensor &output = outputs.descriptors[output_slot++];
        const uint64_t bytes = layout.buffer_sizes[index];
        init_tensor_from_create_info(
            output, args.tensor(index).create_info(), g_dist.heap_base + reservation.begin + output_offset, bytes
        );
        output.owner_task_id.raw = ctx.result.task_id().raw;
        Tensor::copy(ctx.payload->tensors[index], output);
        ctx.result.materialize_output(output);
        output_offset += PTO2_ALIGN_UP(bytes, PTO2_PACKED_OUTPUT_ALIGN);
    }
    if (output_slot != output_count || output_offset != layout.total_output_size) return false;
    if (fdwic::cross_core::PublishTaskOutputs<DistCrossCoreAicoreOps>(
            outputs, static_cast<uint32_t>(ctx.task_id), static_cast<uint32_t>(ctx.self->core_idx), output_count,
            dist_cross_core_runtime_state().fatal
        ) != OutputPublishResult::Published) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    ctx.output_bytes = layout.total_output_size;
    return true;
}

PTO_DEVICE_FUNC bool dist_cross_core_acquire_outputs(DistSubmitCtx &ctx) {
    __gm__ CrossCoreOutputCell<Tensor> &outputs =
        dist_cross_core_runtime_state().outputs[static_cast<uint32_t>(ctx.task_id)];
    uint32_t polls = 0;
    uint32_t output_count = 0;
    while (true) {
        const OutputAcquireResult result = fdwic::cross_core::AcquirePublishedTaskOutputs<DistCrossCoreAicoreOps>(
            outputs, static_cast<uint32_t>(ctx.task_id), output_count, dist_cross_core_runtime_state().fatal
        );
        if (result == OutputAcquireResult::Acquired) break;
        if (result != OutputAcquireResult::NotPublished) {
            return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
        }
        SPIN_WAIT_HINT();
        if ((++polls & 1023U) == 0 && fdwic_trace_is_fatal(ctx.task_id)) return false;
    }
    for (uint32_t output = 0; output < output_count; ++output) {
        ctx.result.materialize_output(outputs.descriptors[output]);
    }
    return true;
}

PTO_DEVICE_FUNC bool dist_cross_core_wait_map_turn(DistSubmitCtx &ctx) {
    if (ctx.task_id == 0) return true;
    __gm__ volatile int64_t &predecessor = task_cell(ctx.task_id - 1).deps_prepared;
    uint32_t polls = 0;
    while (true) {
        const MapTurnResult result =
            fdwic::cross_core::InspectMapTaskTurn<DistCrossCoreAicoreOps>(predecessor, ctx.task_id);
        if (result == MapTurnResult::Ready) return true;
        if (result == MapTurnResult::Invalid) {
            return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
        }
        SPIN_WAIT_HINT();
        if ((++polls & 1023U) == 0 && fdwic_trace_is_fatal(ctx.task_id)) return false;
    }
}

PTO_DEVICE_FUNC bool dist_cross_core_add_fanin(int32_t fanin[], int32_t &count, int32_t producer) {
    if (producer < 0) return true;
    for (int32_t index = 0; index < count; ++index) {
        if (fanin[index] == producer) return true;
    }
    if (count >= kMaxFanin) return false;
    fanin[count++] = producer;
    return true;
}

PTO_DEVICE_FUNC bool dist_cross_core_make_region(__gm__ const Tensor &tensor, int32_t producer, CrossMapValue &value) {
    uint64_t address = 0;
    uint64_t lo = 0;
    uint64_t hi = 0;
    dist_tensor_map_byte_range(tensor, address, lo, hi);
    if (hi <= lo) return false;
    value = CrossMapValue{address, lo, hi, producer, 0};
    return true;
}

PTO_DEVICE_FUNC bool dist_cross_core_prepare_dependencies(
    const L0TaskArgs &args, DistSubmitCtx &ctx, CrossMapValue writer_entries[], uint32_t &writer_count
) {
    ctx.fanin_count = 0;
    writer_count = 0;
    for (int32_t index = 0; index < ctx.tensor_count; ++index) {
        const TensorArgType tag = args.tag(index);
        __gm__ const Tensor &tensor = ctx.payload->tensors[index];
        if (tag != TensorArgType::OUTPUT) {
            if (tensor.owner_task_id.raw != UINT64_MAX) {
                const int32_t producer = static_cast<int32_t>(tensor.owner_task_id.raw & UINT32_MAX);
                if (producer < 0 || producer >= ctx.task_id ||
                    !dist_cross_core_add_fanin(ctx.fanin, ctx.fanin_count, producer)) {
                    return false;
                }
            }
            if ((tag == TensorArgType::INPUT || tag == TensorArgType::INOUT) && !tensor.manual_dep) {
                CrossMapValue query{};
                if (!dist_cross_core_make_region(tensor, -1, query)) return false;
                bool protocol_ok = false;
                const int32_t producer = fdwic::cross_core::LookupCrossMap<DistCrossCoreAicoreOps>(
                    dist_cross_core_runtime_state().tensor_map, query, ctx.task_id, g_dist.H, protocol_ok
                );
                if (!protocol_ok || !dist_cross_core_add_fanin(ctx.fanin, ctx.fanin_count, producer)) return false;
            }
        }
        if ((ctx.register_mask & (uint32_t{1} << static_cast<uint32_t>(index))) != 0) {
            if (writer_count >= fdwic::cross_core::kExecMaxTensors ||
                !dist_cross_core_make_region(tensor, ctx.task_id, writer_entries[writer_count])) {
                return false;
            }
            ++writer_count;
        }
    }
    for (uint32_t index = 0; index < args.explicit_dep_count(); ++index) {
        const PTO2TaskId dependency = args.explicit_dep(index);
        if (!dependency.is_valid() || dependency.ring() != 0 ||
            dependency.local() >= static_cast<uint32_t>(ctx.task_id) ||
            !dist_cross_core_add_fanin(ctx.fanin, ctx.fanin_count, static_cast<int32_t>(dependency.local()))) {
            return false;
        }
    }
    return true;
}

PTO_DEVICE_FUNC bool
dist_cross_core_publish_map_task(DistSubmitCtx &ctx, const CrossMapValue writer_entries[], uint32_t writer_count) {
    const MapAppendResult append = fdwic::cross_core::AppendCrossMapTask<DistCrossCoreAicoreOps>(
        dist_cross_core_runtime_state().tensor_map, writer_entries, writer_count, ctx.task_id
    );
    if (append != MapAppendResult::Appended) {
        return dist_cross_core_fail(
            ctx.task_id,
            append == MapAppendResult::CapacityExceeded ? PTO2_ERROR_TENSORMAP_CAPACITY : PTO2_ERROR_TENSORMAP_PROTOCOL
        );
    }
    store_barrier();
    if (!fdwic::cross_core::PublishMapTaskCompletion<DistCrossCoreAicoreOps>(
            task_cell(ctx.task_id).deps_prepared, ctx.task_id
        )) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    return true;
}

#if PTO_FDWIC_SCHEDULER_MODE == 2 || PTO_FDWIC_SCHEDULER_MODE == 4
PTO_DEVICE_FUNC __gm__ fdwic::cross_core::DagTaskMetadataCell *dist_cross_core_dag_metadata() {
#if PTO_FDWIC_SCHEDULER_MODE == 2
    return &g_dist.cross_core_dag.metadata[0];
#else
    return &g_dist.simt_cross_core_dag.metadata[0];
#endif
}

PTO_DEVICE_FUNC bool dist_cross_core_dag_prepare_dependencies(const L0TaskArgs &args, DistSubmitCtx &ctx) {
    CrossMapValue writer_entries[fdwic::cross_core::kDagMaxWriterRegions];
    uint32_t writer_count = 0;
    for (int32_t index = 0; index < ctx.tensor_count; ++index) {
        if ((ctx.register_mask & (uint32_t{1} << static_cast<uint32_t>(index))) == 0) continue;
        if (writer_count >= fdwic::cross_core::kDagMaxWriterRegions ||
            !dist_cross_core_make_region(ctx.payload->tensors[index], ctx.task_id, writer_entries[writer_count])) {
            return false;
        }
        ++writer_count;
    }
    __gm__ fdwic::cross_core::DagTaskMetadataCell &metadata =
        dist_cross_core_dag_metadata()[static_cast<uint32_t>(ctx.task_id)];
    if (fdwic::cross_core::PublishDagTaskMetadata<DistCrossCoreAicoreOps>(
            metadata, static_cast<uint32_t>(ctx.task_id), writer_entries, writer_count
        ) != fdwic::cross_core::DagMetadataPublishResult::Published) {
        return false;
    }

    ctx.fanin_count = 0;
    for (int32_t index = 0; index < ctx.tensor_count; ++index) {
        const TensorArgType tag = args.tag(index);
        if (tag == TensorArgType::OUTPUT) continue;
        __gm__ const Tensor &tensor = ctx.payload->tensors[index];
        const uint64_t owner_raw = tensor.owner_task_id.raw;
        if (owner_raw != UINT64_MAX) {
            const uint8_t owner_ring = static_cast<uint8_t>(owner_raw >> 32U);
            const uint32_t owner_local = static_cast<uint32_t>(owner_raw);
            if (owner_ring != 0 || owner_local >= static_cast<uint32_t>(ctx.task_id) ||
                !dist_cross_core_add_fanin(ctx.fanin, ctx.fanin_count, static_cast<int32_t>(owner_local))) {
                return false;
            }
        }
        // OUTPUT_EXISTING remains write-only: register it without a lookup.
        // Only INPUT and INOUT query the latest overlapping writer.
        if ((tag != TensorArgType::INPUT && tag != TensorArgType::INOUT) || tensor.manual_dep) continue;
        CrossMapValue query{};
        if (!dist_cross_core_make_region(tensor, -1, query)) return false;
        uint32_t polls = 0;
        while (true) {
            int32_t producer = -1;
            const fdwic::cross_core::DagWriterLookupResult lookup =
                fdwic::cross_core::FindLatestDagWriter<DistCrossCoreAicoreOps>(
                    dist_cross_core_dag_metadata(), kFdwicCrossCoreTaskCapacity, query,
                    static_cast<uint32_t>(ctx.task_id), static_cast<uint32_t>(g_dist.H), producer
                );
            if (lookup == fdwic::cross_core::DagWriterLookupResult::Found) {
                if (!dist_cross_core_add_fanin(ctx.fanin, ctx.fanin_count, producer)) return false;
                break;
            }
            if (lookup == fdwic::cross_core::DagWriterLookupResult::None) break;
            if (lookup != fdwic::cross_core::DagWriterLookupResult::Pending) return false;
            SPIN_WAIT_HINT();
            if ((++polls & 1023U) == 0 && fdwic_trace_is_fatal(ctx.task_id)) return false;
        }
    }
    for (uint32_t index = 0; index < args.explicit_dep_count(); ++index) {
        const PTO2TaskId dependency = args.explicit_dep(index);
        if (!dependency.is_valid() || dependency.ring() != 0 ||
            dependency.local() >= static_cast<uint32_t>(ctx.task_id) ||
            !dist_cross_core_add_fanin(ctx.fanin, ctx.fanin_count, static_cast<int32_t>(dependency.local()))) {
            return false;
        }
    }
    return true;
}
#endif

struct DistCrossCorePayloadSource {
    __gm__ const DistTaskPayload *payload;
    const L0TaskArgs *args;
    const int32_t *fanin;

    PTO_DEVICE_FUNC uint64_t TensorReference(uint32_t) const { return 0; }

    PTO_DEVICE_FUNC uint64_t TensorWord(uint32_t tensor, uint32_t word) const {
        __gm__ const uint64_t *words = reinterpret_cast<__gm__ const uint64_t *>(&payload->tensors[tensor]);
        return words[word];
    }

    PTO_DEVICE_FUNC uint64_t Scalar(uint32_t scalar) const { return args->scalar(static_cast<int32_t>(scalar)); }

    PTO_DEVICE_FUNC int32_t Fanin(uint32_t index) const { return fanin[index]; }
};

PTO_DEVICE_FUNC bool
dist_cross_core_classify_kernel(const MixedKernels &mixed, ExecEngineClass &engine_class, int32_t &kernel_id) {
    return dist_cross_core_classify_single_lane_kernel(mixed, engine_class, kernel_id);
}

PTO_DEVICE_FUNC bool dist_cross_core_publish_exec(
    DistSubmitCtx &ctx, const L0TaskArgs &args, ExecEngineClass engine_class, int32_t kernel_id,
    const HeapReservation &reservation
) {
    DistCrossCorePayloadSource source{ctx.payload, &args, ctx.fanin};
    const uint64_t function_address =
        engine_class == ExecEngineClass::Immediate ? 0 : dist_aicore_slot_function_addr(g_dist.runtime, kernel_id);
#if !defined(__CCE_AICORE__)
    if (engine_class != ExecEngineClass::Immediate && function_address == 0) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
#endif
    const fdwic::cross_core::ExecPayloadSpec spec{
        static_cast<uint32_t>(ctx.task_id),
        function_address,
        reservation.end,
        engine_class == ExecEngineClass::Immediate ? fdwic::cross_core::kExecInvalidFunctionId :
                                                     static_cast<uint32_t>(kernel_id),
        static_cast<uint16_t>(ctx.tensor_count),
        static_cast<uint16_t>(ctx.scalar_count),
        static_cast<uint16_t>(ctx.fanin_count),
        engine_class,
        0,
        0,
        0,
        1,
        0,
    };
    __gm__ SharedExecCell &cell = dist_cross_core_runtime_state().tasks[static_cast<uint32_t>(ctx.task_id)];
    if (fdwic::cross_core::PublishReservedExecPayload<DistCrossCoreAicoreOps>(
            cell, static_cast<uint32_t>(ctx.self->core_idx), spec, source, dist_cross_core_runtime_state().fatal
        ) != ExecBuildResult::Published) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    if (engine_class != ExecEngineClass::Immediate) return true;

    store_task_vend(ctx.task_id, reservation.end);
    store_barrier();
    publish_task_flag(ctx.task_id);
    if (fdwic::cross_core::PublishImmediateExecDone<DistCrossCoreAicoreOps>(
            cell, static_cast<uint32_t>(ctx.task_id), static_cast<uint32_t>(ctx.self->core_idx),
            dist_cross_core_runtime_state().fatal
        ) != fdwic::cross_core::ExecDoneResult::Done) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    return true;
}

PTO_DEVICE_FUNC bool dist_cross_core_build_ring_slot(__gm__ RingSlot &slot, const ExecToken &token) {
    __gm__ const auto *payload = reinterpret_cast<__gm__ const fdwic::cross_core::ExecPayloadStorage *>(
        static_cast<uintptr_t>(token.payload_address)
    );
    if (payload == nullptr) return false;
    fdwic::cross_core::ExecPayloadLayout layout{};
    if (!fdwic::cross_core::ComputeExecPayloadLayout(
            token.header.tensor_count, token.header.scalar_count, token.header.fanin_count,
            token.header.tensor_reference_mask, layout
        )) {
        return false;
    }

    slot.task_id = static_cast<int32_t>(token.task_id);
    slot.func_id = static_cast<int32_t>(token.header.function_id);
    slot.function_bin_addr = token.header.function_address;
    slot.tensor_count = token.header.tensor_count;
    slot.scalar_count = token.header.scalar_count;
    for (uint32_t tensor = 0; tensor < token.header.tensor_count; ++tensor) {
        const uint32_t word_offset =
            fdwic::cross_core::ExecTensorPayloadWordOffset(tensor, token.header.tensor_reference_mask);
        __gm__ uint64_t *destination = reinterpret_cast<__gm__ uint64_t *>(&slot.tensors[tensor]);
        if ((token.header.tensor_reference_mask & (uint32_t{1} << tensor)) != 0) {
            __gm__ const Tensor *reference =
                reinterpret_cast<__gm__ const Tensor *>(static_cast<uintptr_t>(payload->words[word_offset]));
            if (reference == nullptr) return false;
            dist_aicore_invalidate_region(reference, sizeof(Tensor));
            Tensor::copy(slot.tensors[tensor], *reference);
            continue;
        }
        for (uint32_t word = 0; word < fdwic::cross_core::kExecTensorDescWords; ++word) {
            destination[word] = payload->words[word_offset + word];
        }
    }
    for (uint32_t scalar = 0; scalar < token.header.scalar_count; ++scalar) {
        slot.scalars[scalar] = payload->words[layout.scalar_word_offset + scalar];
    }
    int32_t argument = 0;
    for (int32_t tensor = 0; tensor < slot.tensor_count; ++tensor) {
        slot.args[argument++] = reinterpret_cast<uint64_t>(&slot.tensors[tensor]);
    }
    for (int32_t scalar = 0; scalar < slot.scalar_count; ++scalar)
        slot.args[argument++] = slot.scalars[scalar];
    slot.local_ctx.s_block_idx = 0;
    slot.local_ctx.s_block_num = 1;
    slot.local_ctx.async_ctx.completion_count = nullptr;
    slot.local_ctx.async_ctx.completion_error_code = nullptr;
    slot.local_ctx.async_ctx.completion_entries = nullptr;
    slot.local_ctx.async_ctx.completion_capacity = 0;
    slot.local_ctx.async_ctx.task_token.raw = UINT64_MAX;
    slot.global_ctx.sub_block_id = g_self != nullptr && g_self->lane == LANE_AIV1 ? 1 : 0;
    slot.args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&slot.local_ctx);
    slot.args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&slot.global_ctx);
    slot.fanin_count = token.header.fanin_count;
    for (uint32_t edge = 0; edge < token.header.fanin_count; ++edge) {
        const uint64_t packed = payload->words[layout.fanin_word_offset + edge / 2U];
        slot.fanin[edge] =
            static_cast<int32_t>(edge % 2U == 0 ? static_cast<uint32_t>(packed) : static_cast<uint32_t>(packed >> 32U));
    }
    slot.is_multicore = false;
    slot.won_block = -1;
    slot.won_slot = -1;
    slot.built = true;
    return true;
}

PTO_DEVICE_FUNC ExecEngineClass dist_cross_core_executor_engine(__gm__ DistCore *self) {
    if (self == nullptr) return ExecEngineClass::None;
    if (self->role == CoreType::AIC) return ExecEngineClass::Aic;
    if (self->role == CoreType::AIV) return ExecEngineClass::Aiv;
    return ExecEngineClass::None;
}

PTO_DEVICE_FUNC bool dist_cross_core_win_task_tournament(
    DistSubmitCtx &ctx, __gm__ SharedClaimTournamentTask &tournament, FdwicAtomicSite local_site,
    FdwicAtomicSite root_site, bool &won
);

PTO_DEVICE_FUNC bool dist_cross_core_bind_execution(DistSubmitCtx &ctx, ExecEngineClass task_engine) {
    const ExecEngineClass executor_engine = dist_cross_core_executor_engine(ctx.self);
    if (executor_engine != task_engine) return true;
#if (PTO_FDWIC_SCHEDULER_MODE == 3 || PTO_FDWIC_SCHEDULER_MODE == 4) && defined(__CCE_AICORE__)
    // A topology-selected AIV0 hosts one persistent SIMT builder VF. Its
    // vector unit cannot execute a linked AIV task until the deferred join, so
    // keep the complete builder pool out of execute-owner election. Every
    // remaining engine-compatible worker stays a dynamic candidate.
    if (dist_cross_core_is_simt_builder_worker(ctx.self)) return true;
#endif
#if PTO_FDWIC_SCHEDULER_MODE == 4
    bool tournament_winner = false;
    if (!dist_cross_core_win_task_tournament(
            ctx, dist_cross_core_runtime_state().execute_tournament[static_cast<uint32_t>(ctx.task_id)],
            FdwicAtomicSite::CrossCoreExecuteTournamentLocal, FdwicAtomicSite::CrossCoreExecuteTournamentRoot,
            tournament_winner
        )) {
        return false;
    }
    if (!tournament_winner) return true;
#endif
    // Each compatible worker attempts the dynamic owner CAS once; only the
    // first arrival waits for BUILT. The task-private CAS line does not contend
    // with the builder's SharedExecCell::control. Unlike fixed task-id routing,
    // this neither preselects a busy core nor couples Build and Execute owners.
    __gm__ volatile int64_t &owner_word =
        dist_cross_core_runtime_state().execute_owner[static_cast<uint32_t>(ctx.task_id)].state;
    const int64_t desired_owner = static_cast<int64_t>(ctx.self->core_idx) + 1;
    const int64_t observed_owner = DistCrossCoreAicoreOps::CompareExchange(&owner_word, 0, desired_owner);
    if (observed_owner != 0 && observed_owner != desired_owner) return true;
    __gm__ SharedExecCell &cell = dist_cross_core_runtime_state().tasks[static_cast<uint32_t>(ctx.task_id)];
    uint32_t polls = 0;
    while (true) {
        const int64_t raw_state = DistCrossCoreAicoreOps::Load(&cell.control.state);
        const fdwic::cross_core::DecodedExecState state = fdwic::cross_core::DecodeExecState(raw_state);
        // Build 与 Execute owner 相互独立。两级 Build 仲裁的局部 loser
        // 可能先赢得 Execute owner，而 root winner 尚未把 exec cell 从
        // Empty 推进到 Building；SIMT builder 同样允许这一时序。因此
        // Empty 表示生产者仍在推进，不是控制字损坏。
        if (raw_state == 0) {
            SPIN_WAIT_HINT();
            if ((++polls & 1023U) == 0 && fdwic_trace_is_fatal(ctx.task_id)) return false;
            continue;
        }
        if (!state.valid || state.task_id != static_cast<uint32_t>(ctx.task_id)) {
            return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
        }
        if (state.phase == ExecPhase::Claimed || state.phase == ExecPhase::Done) return true;
        if (state.phase != ExecPhase::Built) {
            SPIN_WAIT_HINT();
            if ((++polls & 1023U) == 0 && fdwic_trace_is_fatal(ctx.task_id)) return false;
            continue;
        }
        if (!dist_submit_wait_slot_capacity(ctx.self, ctx.task_id)) return false;
        ExecToken token{};
        fdwic::cross_core::ResetExecToken(token);
        const ExecAcquireResult acquired = fdwic::cross_core::AcquireExecPayload<DistCrossCoreAicoreOps>(
            cell, static_cast<uint32_t>(ctx.task_id), static_cast<uint32_t>(ctx.self->core_idx), executor_engine, token,
            dist_cross_core_runtime_state().fatal
        );
        if (acquired == ExecAcquireResult::Lost) continue;
        if (acquired == ExecAcquireResult::NotBuilt) continue;
        if (acquired != ExecAcquireResult::Acquired) {
            return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
        }
        __gm__ RingSlot *slot = dist_submit_alloc_slot(ctx.self);
        if (slot == nullptr || !dist_cross_core_build_ring_slot(*slot, token)) {
            if (slot != nullptr) {
                slot->occupied = false;
                slot->built = false;
                --ctx.self->occupied_count;
            }
            return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
        }
        (void)drain_phase_b(ctx.self);
        return true;
    }
}

PTO_DEVICE_FUNC bool dist_cross_core_win_task_tournament(
    DistSubmitCtx &ctx, __gm__ SharedClaimTournamentTask &tournament, FdwicAtomicSite local_site,
    FdwicAtomicSite root_site, bool &won
) {
    won = false;
    if (ctx.self == nullptr || ctx.self->core_idx < 0 || g_dist.num_workers <= 0 ||
        ctx.self->core_idx >= g_dist.num_workers) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    const uint32_t worker_count = static_cast<uint32_t>(g_dist.num_workers);
    const uint32_t groups =
        worker_count < kFdwicSharedClaimTournamentMaxGroups ? worker_count : kFdwicSharedClaimTournamentMaxGroups;
    const uint32_t group = static_cast<uint32_t>(ctx.self->core_idx) % groups;
    const int64_t expected = -1;
    const int64_t desired = static_cast<int64_t>(ctx.task_id);
    const int64_t local_observed = fdwic_trace_atomic_compare_exchange<int64_t>(
        ctx.task_id, local_site, tournament.local[group].owner.v, expected, desired, /*result_used=*/true
    );
    if (local_observed != expected) {
        if (local_observed != desired) return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
        return true;
    }
    const int64_t root_observed = fdwic_trace_atomic_compare_exchange<int64_t>(
        ctx.task_id, root_site, tournament.root.owner.v, expected, desired, /*result_used=*/true
    );
    if (root_observed != expected && root_observed != desired) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    won = root_observed == expected;
    return true;
}

PTO_DEVICE_FUNC bool dist_cross_core_win_build_tournament(DistSubmitCtx &ctx, bool &won) {
    return dist_cross_core_win_task_tournament(
        ctx, dist_cross_core_runtime_state().build_tournament[static_cast<uint32_t>(ctx.task_id)],
        FdwicAtomicSite::CrossCoreBuildTournamentLocal, FdwicAtomicSite::CrossCoreBuildTournamentRoot, won
    );
}

PTO_DEVICE_FUNC bool dist_cross_core_reserve_build(DistSubmitCtx &ctx, bool &build_owner) {
    build_owner = false;
#if PTO_FDWIC_SCHEDULER_MODE == 1 || PTO_FDWIC_SCHEDULER_MODE == 2
    bool tournament_winner = false;
    if (!dist_cross_core_win_build_tournament(ctx, tournament_winner)) return false;
    if (!tournament_winner) return true;
#endif
    __gm__ SharedExecCell &cell = dist_cross_core_runtime_state().tasks[static_cast<uint32_t>(ctx.task_id)];
    const ExecBuildReserveResult reservation_result = fdwic::cross_core::ReserveExecBuild<DistCrossCoreAicoreOps>(
        cell, static_cast<uint32_t>(ctx.task_id), static_cast<uint32_t>(ctx.self->core_idx),
        dist_cross_core_runtime_state().fatal
    );
    build_owner = reservation_result == ExecBuildReserveResult::Reserved;
    if (!build_owner && reservation_result != ExecBuildReserveResult::CellUnavailable) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    return true;
}

PTO_DEVICE_FUNC bool dist_cross_core_finish_builder(
    DistSubmitCtx &ctx, const L0TaskArgs &args, ExecEngineClass engine_class, int32_t kernel_id,
    uint32_t expected_output_count = UINT32_MAX
) {
    ctx.tensor_count = args.tensor_count();
    ctx.scalar_count = args.scalar_count();
    DistOutputLayout layout{};
    uint32_t output_mask = 0;
    uint32_t output_count = 0;
    if (!dist_cross_core_plan_outputs(args, layout, ctx.register_mask, output_mask, output_count)) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    if (expected_output_count != UINT32_MAX && output_count != expected_output_count) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    HeapReservation heap{};
    if (!dist_cross_core_materialize_builder(args, ctx, layout, output_mask, output_count, heap)) {
        return false;
    }
#if PTO_FDWIC_SCHEDULER_MODE == 1 || PTO_FDWIC_SCHEDULER_MODE == 3
    if (!dist_cross_core_wait_map_turn(ctx)) return false;
    CrossMapValue writer_entries[fdwic::cross_core::kExecMaxTensors];
    uint32_t writer_count = 0;
    if (!dist_cross_core_prepare_dependencies(args, ctx, writer_entries, writer_count) ||
        !dist_cross_core_publish_map_task(ctx, writer_entries, writer_count)) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
#else
    if (!dist_cross_core_dag_prepare_dependencies(args, ctx)) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
#endif
    if (!dist_cross_core_publish_exec(ctx, args, engine_class, kernel_id, heap)) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    return true;
}

PTO_DEVICE_FUNC TaskOutputTensors dist_cross_core_finish_submission(
    DistSubmitCtx &ctx, const L0TaskArgs *builder_args, DistSubmitKind kind, ExecEngineClass engine_class,
    int32_t kernel_id, bool build_owner
) {
    if (build_owner) {
        if (builder_args == nullptr || !dist_cross_core_finish_builder(ctx, *builder_args, engine_class, kernel_id)) {
            return ctx.result;
        }
    } else if (!dist_cross_core_acquire_outputs(ctx)) {
        return ctx.result;
    }
    if (kind == DistSubmitKind::Kernel && !dist_cross_core_bind_execution(ctx, engine_class)) return ctx.result;
    return ctx.result;
}

PTO_DEVICE_FUNC TaskOutputTensors
dist_cross_core_submit(const MixedKernels *mixed, const L0TaskArgs &args, DistSubmitKind kind) {
    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, args, ctx);
    if (!dist_cross_core_task_valid(ctx)) {
        (void)dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_CAPACITY);
        return ctx.result;
    }
    (void)drain_phase_b(ctx.self);

    ExecEngineClass engine_class = ExecEngineClass::Immediate;
    int32_t kernel_id = INVALID_KERNEL_ID;
    if (kind == DistSubmitKind::Kernel &&
        (mixed == nullptr || !dist_cross_core_classify_kernel(*mixed, engine_class, kernel_id))) {
        (void)dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
        return ctx.result;
    }
    bool build_owner = false;
    if (!dist_cross_core_reserve_build(ctx, build_owner)) return ctx.result;
    return dist_cross_core_finish_submission(
        ctx, build_owner ? &args : nullptr, kind, engine_class, kernel_id, build_owner
    );
}

PTO_DEVICE_FUNC DistCompeteFirstTicket dist_cross_core_invalid_ticket() {
    DistCompeteFirstTicket ticket{};
    ticket.task_id = kFlagCap;
    ticket.kernel_id = static_cast<int16_t>(INVALID_KERNEL_ID);
    return ticket;
}

PTO_DEVICE_FUNC DistCompeteFirstTicket
dist_cross_core_compete_first_begin(const MixedKernels *mixed, DistSubmitKind kind) {
    DistSubmitCtx ctx;
    dist_submit_begin(nullptr, ctx);
    if (!dist_cross_core_task_valid(ctx)) {
        (void)dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_CAPACITY);
        return dist_cross_core_invalid_ticket();
    }
    (void)drain_phase_b(ctx.self);

    ExecEngineClass engine_class = ExecEngineClass::Immediate;
    int32_t kernel_id = INVALID_KERNEL_ID;
    if (kind == DistSubmitKind::Kernel &&
        (mixed == nullptr || !dist_cross_core_classify_kernel(*mixed, engine_class, kernel_id))) {
        (void)dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
        return dist_cross_core_invalid_ticket();
    }
    bool build_owner = false;
    if (!dist_cross_core_reserve_build(ctx, build_owner)) return dist_cross_core_invalid_ticket();

    DistCompeteFirstTicket ticket{};
    ticket.task_id = ctx.task_id;
    ticket.kernel_id = static_cast<int16_t>(kernel_id);
    // The generic wrapper runs its synchronous argument callback only when
    // ready is nonzero.  For cross-core scheduling that callback belongs only
    // to the independently elected Build owner; observers wait for the output
    // cell and immutable execution packet in Finish.
    ticket.won = static_cast<uint8_t>(build_owner);
    ticket.ready = static_cast<uint8_t>(build_owner);
    return ticket;
}

PTO_DEVICE_FUNC bool dist_cross_core_restore_ticket(
    const DistCompeteFirstTicket &ticket, const MixedKernels *mixed, DistSubmitKind kind, DistSubmitCtx &ctx,
    ExecEngineClass &engine_class, int32_t &kernel_id
) {
    ctx.self = g_self;
    ctx.task_id = ticket.task_id;
    ctx.payload = ctx.self != nullptr && ticket.task_id >= 0 &&
                          static_cast<uint32_t>(ticket.task_id) < kFdwicCrossCoreTaskCapacity ?
                      &ctx.self->task_payloads[ticket.task_id & kTaskPayloadMask] :
                      nullptr;
    ctx.result.set_task_id(PTO2TaskId::make(0, static_cast<uint32_t>(ticket.task_id)));
    ctx.tensor_count = 0;
    ctx.scalar_count = 0;
    ctx.register_mask = 0;
    ctx.output_bytes = 0;
    ctx.fanin_count = 0;
    ctx.kernel_id = ticket.kernel_id;
    ctx.won = ticket.won != 0;
    ctx.joint = false;
    ctx.joint_init = false;
    ctx.joint_block = -1;
    ctx.joint_slot = -1;
    ctx.joint_count = 0;
    ctx.claim_attempted = true;

    engine_class = ExecEngineClass::Immediate;
    kernel_id = INVALID_KERNEL_ID;
    const bool kernel_ok = kind == DistSubmitKind::Alloc ?
                               (mixed == nullptr && ticket.kernel_id == INVALID_KERNEL_ID) :
                               (mixed != nullptr && dist_cross_core_classify_kernel(*mixed, engine_class, kernel_id) &&
                                kernel_id == ticket.kernel_id);
    const bool fields_ok = ticket.won <= 1 && ticket.ready <= 1 && ticket.won == ticket.ready;
    const bool sequence_ok = dist_cross_core_task_valid(ctx) && ctx.self->local_index == ticket.task_id + 1;
    if (kernel_ok && fields_ok && sequence_ok) return true;
    return dist_cross_core_fail(ticket.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
}

PTO_DEVICE_FUNC TaskOutputTensors dist_cross_core_compete_first_finish(
    const MixedKernels *mixed, const DistCompeteFirstTicket &ticket, const L0TaskArgs &args, DistSubmitKind kind
) {
    DistSubmitCtx ctx;
    ExecEngineClass engine_class = ExecEngineClass::None;
    int32_t kernel_id = INVALID_KERNEL_ID;
    if (!dist_cross_core_restore_ticket(ticket, mixed, kind, ctx, engine_class, kernel_id)) return ctx.result;
    const bool build_owner = ticket.won != 0;
    return dist_cross_core_finish_submission(
        ctx, build_owner ? &args : nullptr, kind, engine_class, kernel_id, build_owner
    );
}

PTO_DEVICE_FUNC bool dist_cross_core_compete_first_finish_deferred(
    const MixedKernels *mixed, const DistCompeteFirstTicket &ticket, const L0TaskArgs &args, DistSubmitKind kind,
    uint32_t expected_output_count
) {
    DistSubmitCtx ctx;
    ExecEngineClass engine_class = ExecEngineClass::None;
    int32_t kernel_id = INVALID_KERNEL_ID;
    if (!dist_cross_core_restore_ticket(ticket, mixed, kind, ctx, engine_class, kernel_id)) return false;
    if (expected_output_count > MAX_TENSOR_ARGS) {
        return dist_cross_core_fail(ctx.task_id, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    const bool build_owner = ticket.won != 0;
    if (build_owner && !dist_cross_core_finish_builder(ctx, args, engine_class, kernel_id, expected_output_count)) {
        return false;
    }
    // Observers deliberately skip output acquisition.  They retain only the
    // stable (task_id, output_slot) references supplied by orchestration.  All
    // engine-compatible actors still reach this point so one of them can own
    // Execute independently from the Build owner.
    return kind != DistSubmitKind::Kernel || dist_cross_core_bind_execution(ctx, engine_class);
}

PTO_DEVICE_FUNC TaskOutputTensors dist_cross_core_submit_kernel(const MixedKernels &mixed, const L0TaskArgs &args) {
    return dist_cross_core_submit(&mixed, args, DistSubmitKind::Kernel);
}

PTO_DEVICE_FUNC TaskOutputTensors dist_cross_core_alloc(const L0TaskArgs &args) {
    return dist_cross_core_submit(nullptr, args, DistSubmitKind::Alloc);
}

PTO_DEVICE_FUNC DistCompeteFirstTicket dist_cross_core_submit_compete_first_begin(const MixedKernels &mixed) {
    return dist_cross_core_compete_first_begin(&mixed, DistSubmitKind::Kernel);
}

PTO_DEVICE_FUNC TaskOutputTensors dist_cross_core_submit_compete_first_finish(
    const MixedKernels &mixed, const DistCompeteFirstTicket &ticket, const L0TaskArgs &args
) {
    return dist_cross_core_compete_first_finish(&mixed, ticket, args, DistSubmitKind::Kernel);
}

PTO_DEVICE_FUNC DistCompeteFirstTicket dist_cross_core_alloc_compete_first_begin() {
    return dist_cross_core_compete_first_begin(nullptr, DistSubmitKind::Alloc);
}

PTO_DEVICE_FUNC TaskOutputTensors
dist_cross_core_alloc_compete_first_finish(const DistCompeteFirstTicket &ticket, const L0TaskArgs &args) {
    return dist_cross_core_compete_first_finish(nullptr, ticket, args, DistSubmitKind::Alloc);
}

PTO_DEVICE_FUNC bool dist_cross_core_submit_deferred_compete_first_finish(
    const MixedKernels &mixed, const DistCompeteFirstTicket &ticket, const L0TaskArgs &args,
    uint32_t expected_output_count
) {
    return dist_cross_core_compete_first_finish_deferred(
        &mixed, ticket, args, DistSubmitKind::Kernel, expected_output_count
    );
}

PTO_DEVICE_FUNC bool dist_cross_core_alloc_deferred_compete_first_finish(
    const DistCompeteFirstTicket &ticket, const L0TaskArgs &args, uint32_t expected_output_count
) {
    return dist_cross_core_compete_first_finish_deferred(
        nullptr, ticket, args, DistSubmitKind::Alloc, expected_output_count
    );
}

}  // namespace

#endif  // PTO_FDWIC_SCHEDULER_MODE is a cross-core mode.
