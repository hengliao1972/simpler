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

#if PTO_FDWIC_SCHEDULER_MODE == 3

namespace {

constexpr uint32_t kDistSimtBuilderThreads = 128;
constexpr uint32_t kDistSimtBuilderWarps = kDistSimtBuilderThreads / 32U;
constexpr int64_t kDistSimtBuilderReset = 0;
constexpr int64_t kDistSimtBuilderRunning = 1;
// Use the bounded-wait scale proven by standalone G0. The outer A5 AICPU has a
// three-second execution threshold, so 30G cycles would let that timeout hide
// protocol errors. One billion cycles covers the full request window while
// allowing the builder to publish an explicit fatal first.
constexpr uint64_t kDistSimtBuilderPollBudget = 1000000000ULL;

PTO_DEVICE_FUNC bool dist_simt_cross_core_is_builder_worker(__gm__ const DistCore *self) {
#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)
    // Worker indices follow runtime registration order and do not promise that
    // every AIC precedes every AIV. AICPU maps the first AIV to physical
    // block0/AIV0; use that published topology identity to elect one builder
    // without treating dynamic registration order as a fixed ABI.
    return self != nullptr && self->role == CoreType::AIV && self->block_id == 0 && self->lane == LANE_AIV0;
#else
    (void)self;
    return false;
#endif
}

#if defined(__CCE_AICORE__) && defined(__DAV_VEC__) && defined(PTO_FDWIC_SIMT_DEFINE_BUILDER_VF)

#define DIST_SIMT_CALLEE __simt_callee__ __aicore__ __attribute__((always_inline)) inline

struct DistSimtRequestView {
    __gm__ uint64_t *words;
    uint32_t task_id;
    uint32_t function_id;
    uint64_t function_address;
    uint32_t tensor_count;
    uint32_t scalar_count;
    uint32_t explicit_dep_count;
    uint32_t engine_class;
    uint32_t payload_lines;
};

DIST_SIMT_CALLEE uint64_t dist_simt_atomic_load(__gm__ uint64_t *address) {
    return asc_atomic_add(address, static_cast<uint64_t>(0U));
}

DIST_SIMT_CALLEE uint64_t dist_simt_atomic_cas(__gm__ uint64_t *address, uint64_t expected, uint64_t desired) {
    return asc_atomic_cas(address, expected, desired);
}

DIST_SIMT_CALLEE void dist_simt_store(__gm__ uint64_t *address, uint64_t value) { *address = value; }

DIST_SIMT_CALLEE uint64_t dist_simt_exec_state(
    uint32_t phase, uint32_t build_owner, uint32_t execute_owner, uint32_t engine_class, uint32_t payload_lines,
    uint32_t task_id
) {
    return static_cast<uint64_t>(phase) | (static_cast<uint64_t>(build_owner) << 3U) |
           (static_cast<uint64_t>(execute_owner) << 11U) | (static_cast<uint64_t>(engine_class) << 19U) |
           (static_cast<uint64_t>(payload_lines) << 22U) | (static_cast<uint64_t>(task_id) << 29U);
}

DIST_SIMT_CALLEE void dist_simt_publish_fatal(__gm__ uint64_t *fatal, uint32_t task_id, uint32_t owner) {
    const uint64_t encoded = static_cast<uint64_t>(fdwic::cross_core::ExecFatalReason::InvalidBuildInput) |
                             (static_cast<uint64_t>(owner) << fdwic::cross_core::kExecFatalOwnerShift) |
                             (static_cast<uint64_t>(task_id) << fdwic::cross_core::kExecFatalTaskIdShift);
    (void)dist_simt_atomic_cas(fatal, 0, encoded);
}

DIST_SIMT_CALLEE bool dist_simt_fatal_observed(__gm__ uint64_t *fatal) { return dist_simt_atomic_load(fatal) != 0; }

DIST_SIMT_CALLEE bool dist_simt_wait_value(__gm__ uint64_t *address, uint64_t expected, __gm__ uint64_t *fatal) {
    const uint64_t begin = clock();
    uint32_t polls = 0;
    while (clock() - begin <= kDistSimtBuilderPollBudget) {
        if (dist_simt_atomic_load(address) == expected) return true;
        if ((++polls & 1023U) == 0 && dist_simt_fatal_observed(fatal)) return false;
    }
    return false;
}

DIST_SIMT_CALLEE uint32_t dist_simt_request_tag(const DistSimtRequestView &request, uint32_t tensor) {
    const uint64_t packed = request.words[3U + tensor / 8U];
    return static_cast<uint32_t>(packed >> ((tensor % 8U) * 8U)) & 0xFFU;
}

DIST_SIMT_CALLEE __gm__ uint64_t *dist_simt_request_tensor(const DistSimtRequestView &request, uint32_t tensor) {
    return request.words + fdwic::cross_core::kSimtRequestHeaderWords +
           tensor * fdwic::cross_core::kExecTensorDescWords;
}

DIST_SIMT_CALLEE bool dist_simt_decode_request(
    __gm__ fdwic::cross_core::SimtBuildRequestCell *cell, uint32_t task_id, DistSimtRequestView *request
) {
    __gm__ uint64_t *control = reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&cell->control.state));
    const uint64_t raw = dist_simt_atomic_load(control);
    const uint32_t phase = static_cast<uint32_t>(raw & fdwic::cross_core::kSimtRequestPhaseMask);
    if (phase != static_cast<uint32_t>(fdwic::cross_core::SimtRequestPhase::Published)) return false;
    const uint32_t payload_lines = static_cast<uint32_t>(
        (raw >> fdwic::cross_core::kSimtRequestPayloadLinesShift) & fdwic::cross_core::kSimtRequestPayloadLinesMask
    );
    const uint32_t encoded_task = static_cast<uint32_t>(
        (raw >> fdwic::cross_core::kSimtRequestTaskIdShift) & fdwic::cross_core::kSimtRequestTaskIdMask
    );
    if ((raw & ~fdwic::cross_core::kSimtRequestKnownMask) != 0 || encoded_task != task_id || payload_lines == 0 ||
        payload_lines > fdwic::cross_core::kSimtRequestMaxPayloadLines) {
        return false;
    }

    __gm__ uint64_t *words = const_cast<__gm__ uint64_t *>(cell->payload.words);
    const uint64_t identity = words[0];
    const uint64_t counts = words[2];
    request->words = words;
    request->task_id = static_cast<uint32_t>(identity);
    request->function_id = static_cast<uint32_t>(identity >> 32U);
    request->function_address = words[1];
    request->tensor_count = static_cast<uint16_t>(counts);
    request->scalar_count = static_cast<uint16_t>(counts >> 16U);
    request->explicit_dep_count = static_cast<uint16_t>(counts >> 32U);
    request->engine_class = static_cast<uint8_t>(counts >> 48U);
    request->payload_lines = payload_lines;
    if (request->task_id != task_id || static_cast<uint8_t>(counts >> 56U) != 0 || words[7] != 0 ||
        request->tensor_count > fdwic::cross_core::kExecMaxTensors ||
        request->scalar_count > fdwic::cross_core::kExecMaxScalars ||
        request->explicit_dep_count > fdwic::cross_core::kSimtRequestMaxExplicitDependencies ||
        (request->engine_class != static_cast<uint32_t>(fdwic::cross_core::ExecEngineClass::Aic) &&
         request->engine_class != static_cast<uint32_t>(fdwic::cross_core::ExecEngineClass::Aiv) &&
         request->engine_class != static_cast<uint32_t>(fdwic::cross_core::ExecEngineClass::Immediate))) {
        return false;
    }
    const bool immediate =
        request->engine_class == static_cast<uint32_t>(fdwic::cross_core::ExecEngineClass::Immediate);
    if ((immediate && (request->function_id != UINT32_MAX || request->function_address != 0)) ||
        (!immediate && request->function_id == UINT32_MAX && request->function_address == 0)) {
        return false;
    }
    const uint32_t written_words = fdwic::cross_core::kSimtRequestHeaderWords +
                                   request->tensor_count * fdwic::cross_core::kExecTensorDescWords +
                                   request->scalar_count + request->explicit_dep_count;
    const uint32_t expected_lines = (written_words * sizeof(uint64_t) + fdwic::cross_core::kExecCacheLineBytes - 1U) /
                                    fdwic::cross_core::kExecCacheLineBytes;
    if (expected_lines != payload_lines || dist_simt_atomic_load(control) != raw) return false;
    for (uint32_t tensor = 0; tensor < request->tensor_count; ++tensor) {
        const uint32_t tag = dist_simt_request_tag(*request, tensor);
        if (tag > static_cast<uint32_t>(TensorArgType::NO_DEP)) return false;
    }
    return true;
}

DIST_SIMT_CALLEE uint64_t dist_simt_element_bytes(uint32_t dtype) {
    constexpr uint8_t sizes[] = {4, 2, 4, 2, 1, 1, 2, 8, 8, 2, 4, 1};
    return dtype < static_cast<uint32_t>(DataType::DATA_TYPE_NUM) ? sizes[dtype] : 0;
}

DIST_SIMT_CALLEE bool dist_simt_output_size(__gm__ uint64_t *source, uint64_t *bytes) {
    __gm__ const TensorCreateInfo *info = reinterpret_cast<__gm__ const TensorCreateInfo *>(source);
    if (info->has_initial_value || info->start_offset != 0 || info->ndims == 0 || info->ndims > MAX_TENSOR_DIMS ||
        !info->is_contiguous || info->__pad_flags__ != 0) {
        return false;
    }
    // Match the Scalar ordinary builder's 32-bit shape-product contract so the
    // SIMT mode cannot accept output metadata that Tensor cannot represent.
    uint32_t elements = 1;
    for (uint32_t dimension = 0; dimension < info->ndims; ++dimension) {
        const uint32_t extent = info->shapes[dimension];
        if (extent == 0 || elements > UINT32_MAX / extent) return false;
        elements *= extent;
    }
    const uint64_t element_bytes = dist_simt_element_bytes(static_cast<uint32_t>(info->dtype));
    if (element_bytes == 0 || elements > UINT64_MAX / element_bytes) return false;
    *bytes = elements * element_bytes;
    return true;
}

DIST_SIMT_CALLEE void
dist_simt_copy_words(__gm__ uint64_t *destination, __gm__ const uint64_t *source, uint32_t words) {
    for (uint32_t word = 0; word < words; ++word)
        dist_simt_store(destination + word, source[word]);
}

DIST_SIMT_CALLEE void dist_simt_write_output_descriptor(
    __gm__ uint64_t *destination, __gm__ uint64_t *create_info_words, uint64_t address, uint64_t bytes, uint32_t task_id
) {
    __gm__ const TensorCreateInfo *info = reinterpret_cast<__gm__ const TensorCreateInfo *>(create_info_words);
    const uint32_t shape0 = info->ndims > 0 ? info->shapes[0] : 0;
    const uint32_t shape1 = info->ndims > 1 ? info->shapes[1] : 0;
    const uint32_t shape2 = info->ndims > 2 ? info->shapes[2] : 0;
    const uint32_t shape3 = info->ndims > 3 ? info->shapes[3] : 0;
    const uint32_t shape4 = info->ndims > 4 ? info->shapes[4] : 0;
    uint32_t strides[MAX_TENSOR_DIMS] = {};
    uint32_t extent = 1;
    for (int32_t dimension = static_cast<int32_t>(info->ndims) - 1; dimension >= 0; --dimension) {
        strides[dimension] = extent;
        extent *= info->shapes[dimension];
    }
    dist_simt_store(destination + 0, address);
    dist_simt_store(destination + 1, bytes);
    dist_simt_store(destination + 2, task_id);
    dist_simt_store(destination + 3, 0);
    dist_simt_store(
        destination + 4, static_cast<uint64_t>(info->version) | (static_cast<uint64_t>(info->ndims) << 32U)
    );
    dist_simt_store(
        destination + 5, static_cast<uint64_t>(info->dtype) | (static_cast<uint64_t>(info->manual_dep) << 8U) |
                             (static_cast<uint64_t>(1U) << 16U) | (static_cast<uint64_t>(shape0) << 32U)
    );
    dist_simt_store(destination + 6, static_cast<uint64_t>(shape1) | (static_cast<uint64_t>(shape2) << 32U));
    dist_simt_store(destination + 7, static_cast<uint64_t>(shape3) | (static_cast<uint64_t>(shape4) << 32U));
    dist_simt_store(destination + 8, extent);
    dist_simt_store(destination + 9, static_cast<uint64_t>(strides[0]) | (static_cast<uint64_t>(strides[1]) << 32U));
    dist_simt_store(destination + 10, static_cast<uint64_t>(strides[2]) | (static_cast<uint64_t>(strides[3]) << 32U));
    dist_simt_store(destination + 11, strides[4]);
    for (uint32_t word = 12; word < fdwic::cross_core::kExecTensorDescWords; ++word) {
        dist_simt_store(destination + word, 0);
    }
}

DIST_SIMT_CALLEE bool dist_simt_add_fanin(int32_t fanin[], uint32_t *count, int32_t producer) {
    if (producer < 0) return true;
    for (uint32_t index = 0; index < *count; ++index) {
        if (fanin[index] == producer) return true;
    }
    if (*count >= fdwic::cross_core::kExecMaxFanin) return false;
    fanin[(*count)++] = producer;
    return true;
}

DIST_SIMT_CALLEE bool dist_simt_tensor_region(__gm__ uint64_t *tensor, uint64_t *address, uint64_t *lo, uint64_t *hi) {
    const uint64_t element_bytes = dist_simt_element_bytes(static_cast<uint32_t>(tensor[5]) & 0xFFU);
    const uint32_t dimensions = static_cast<uint32_t>(tensor[4] >> 32U);
    if (tensor[0] == 0 || element_bytes == 0 || dimensions == 0 || dimensions > MAX_TENSOR_DIMS) return false;
    const uint64_t start = tensor[3];
    uint64_t extent = tensor[8];
    const bool contiguous = ((tensor[5] >> 16U) & 0xFFU) != 0;
    if (contiguous) {
        extent = 1;
        for (uint32_t dimension = 0; dimension < dimensions; ++dimension) {
            const uint32_t shape =
                static_cast<uint32_t>(tensor[5U + (dimension + 1U) / 2U] >> ((dimension % 2U == 0 ? 32U : 0U)));
            if (shape == 0 || extent > UINT64_MAX / shape) return false;
            extent *= shape;
        }
    }
    if (extent == 0 || start > UINT64_MAX - extent || (start + extent) > UINT64_MAX / element_bytes) return false;
    *address = tensor[0];
    *lo = start * element_bytes;
    *hi = (start + extent) * element_bytes;
    return *lo < *hi;
}

DIST_SIMT_CALLEE uint32_t dist_simt_map_hash(uint64_t address) {
#if PTO_FDWIC_TENSORMAP_RING_CAP == 16384
    (void)address;
    return 0;
#else
    address *= 0x9E3779B97F4A7C15ULL;
    return static_cast<uint32_t>(address >> (64U - fdwic::cross_core::kCrossMapBucketShift)) &
           fdwic::cross_core::kCrossMapBucketMask;
#endif
}

DIST_SIMT_CALLEE int32_t dist_simt_lookup_map(
    __gm__ fdwic::cross_core::CrossCoreTensorMapState *map, uint64_t address, uint64_t lo, uint64_t hi,
    uint32_t task_id, uint32_t history, bool *valid
) {
    *valid = false;
    const uint32_t bucket = dist_simt_map_hash(address);
    __gm__ uint64_t *tail_address =
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&map->tails[bucket].state));
    const uint64_t tail = dist_simt_atomic_load(tail_address);
    if (tail > fdwic::cross_core::kCrossMapBucketCapacity) return -1;
    const uint32_t lower = task_id > history ? task_id - history : 0;
    int32_t best = -1;
    for (uint64_t cursor = 0; cursor < tail; ++cursor) {
        const uint32_t slot_index = bucket * fdwic::cross_core::kCrossMapBucketCapacity +
                                    (static_cast<uint32_t>(cursor) & fdwic::cross_core::kCrossMapBucketSlotMask);
        __gm__ fdwic::cross_core::CrossMapSlot *slot = &map->slots[slot_index];
        __gm__ uint64_t *sequence =
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&slot->sequence.state));
        if (dist_simt_atomic_load(sequence) != cursor) return -1;
        __gm__ uint64_t *payload = reinterpret_cast<__gm__ uint64_t *>(&slot->payload.value);
        const uint64_t candidate_address = payload[0];
        const uint64_t candidate_lo = payload[1];
        const uint64_t candidate_hi = payload[2];
        const uint64_t identity = payload[3];
        const int32_t producer = static_cast<int32_t>(identity);
        if (static_cast<uint32_t>(identity >> 32U) != 0 || candidate_lo >= candidate_hi || producer < 0 ||
            dist_simt_atomic_load(sequence) != cursor) {
            return -1;
        }
        if (producer >= static_cast<int32_t>(lower) && producer < static_cast<int32_t>(task_id) &&
            candidate_address == address && lo < candidate_hi && candidate_lo < hi && producer > best) {
            best = producer;
        }
    }
    *valid = true;
    return best;
}

DIST_SIMT_CALLEE bool dist_simt_append_map(
    __gm__ fdwic::cross_core::CrossCoreTensorMapState *map, const uint64_t writer_address[], const uint64_t writer_lo[],
    const uint64_t writer_hi[], uint32_t writer_count, uint32_t task_id
) {
    for (uint32_t writer = 0; writer < writer_count; ++writer) {
        if (writer_lo[writer] >= writer_hi[writer]) return false;
        const uint32_t bucket = dist_simt_map_hash(writer_address[writer]);
        __gm__ uint64_t *tail_address =
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&map->tails[bucket].state));
        const uint64_t tail = dist_simt_atomic_load(tail_address);
        if (tail >= fdwic::cross_core::kCrossMapBucketCapacity) return false;
        const uint32_t slot_index = bucket * fdwic::cross_core::kCrossMapBucketCapacity +
                                    (static_cast<uint32_t>(tail) & fdwic::cross_core::kCrossMapBucketSlotMask);
        __gm__ fdwic::cross_core::CrossMapSlot *slot = &map->slots[slot_index];
        __gm__ uint64_t *sequence =
            reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&slot->sequence.state));
        if (dist_simt_atomic_load(sequence) != UINT64_MAX) return false;
        __gm__ uint64_t *payload = reinterpret_cast<__gm__ uint64_t *>(&slot->payload.value);
        dist_simt_store(payload + 0, writer_address[writer]);
        dist_simt_store(payload + 1, writer_lo[writer]);
        dist_simt_store(payload + 2, writer_hi[writer]);
        dist_simt_store(payload + 3, task_id);
        asc_threadfence();
        if (dist_simt_atomic_cas(sequence, UINT64_MAX, tail) != UINT64_MAX ||
            dist_simt_atomic_cas(tail_address, tail, tail + 1U) != tail) {
            return false;
        }
    }
    return true;
}

DIST_SIMT_CALLEE bool dist_simt_prepare_map_and_fanin(
    __gm__ fdwic::cross_core::CrossCoreTensorMapState *map, __gm__ DistTaskCell *task_cells,
    const DistSimtRequestView &request, uint32_t history, int32_t fanin[], uint32_t *fanin_count, __gm__ uint64_t *fatal
) {
    if (request.task_id != 0) {
        __gm__ uint64_t *predecessor = reinterpret_cast<__gm__ uint64_t *>(
            const_cast<__gm__ int64_t *>(&task_cells[request.task_id - 1U].deps_prepared)
        );
        if (!dist_simt_wait_value(predecessor, request.task_id - 1U, fatal)) return false;
    }

    uint64_t writer_address[fdwic::cross_core::kExecMaxTensors] = {};
    uint64_t writer_lo[fdwic::cross_core::kExecMaxTensors] = {};
    uint64_t writer_hi[fdwic::cross_core::kExecMaxTensors] = {};
    uint32_t writer_count = 0;
    *fanin_count = 0;
    for (uint32_t tensor = 0; tensor < request.tensor_count; ++tensor) {
        const uint32_t tag = dist_simt_request_tag(request, tensor);
        if (tag == static_cast<uint32_t>(TensorArgType::OUTPUT)) continue;
        __gm__ uint64_t *descriptor = dist_simt_request_tensor(request, tensor);
        const uint64_t owner_raw = descriptor[2];
        if (owner_raw != UINT64_MAX) {
            const uint32_t producer = static_cast<uint32_t>(owner_raw);
            if (static_cast<uint32_t>(owner_raw >> 32U) != 0 || producer >= request.task_id ||
                !dist_simt_add_fanin(fanin, fanin_count, static_cast<int32_t>(producer))) {
                return false;
            }
        }
        const bool manual_dependency = ((descriptor[5] >> 8U) & 0xFFU) != 0;
        uint64_t address = 0;
        uint64_t lo = 0;
        uint64_t hi = 0;
        if (!dist_simt_tensor_region(descriptor, &address, &lo, &hi)) return false;
        if (!manual_dependency && (tag == static_cast<uint32_t>(TensorArgType::INPUT) ||
                                   tag == static_cast<uint32_t>(TensorArgType::INOUT))) {
            bool lookup_valid = false;
            const int32_t producer =
                dist_simt_lookup_map(map, address, lo, hi, request.task_id, history, &lookup_valid);
            if (!lookup_valid || !dist_simt_add_fanin(fanin, fanin_count, producer)) return false;
        }
        if (!manual_dependency && (tag == static_cast<uint32_t>(TensorArgType::INOUT) ||
                                   tag == static_cast<uint32_t>(TensorArgType::OUTPUT_EXISTING))) {
            if (writer_count >= fdwic::cross_core::kExecMaxTensors) return false;
            writer_address[writer_count] = address;
            writer_lo[writer_count] = lo;
            writer_hi[writer_count] = hi;
            ++writer_count;
        }
    }
    const uint32_t dependency_offset = fdwic::cross_core::kSimtRequestHeaderWords +
                                       request.tensor_count * fdwic::cross_core::kExecTensorDescWords +
                                       request.scalar_count;
    for (uint32_t dependency = 0; dependency < request.explicit_dep_count; ++dependency) {
        const uint64_t raw = request.words[dependency_offset + dependency];
        const uint32_t producer = static_cast<uint32_t>(raw);
        if (static_cast<uint32_t>(raw >> 32U) != 0 || producer >= request.task_id ||
            !dist_simt_add_fanin(fanin, fanin_count, static_cast<int32_t>(producer))) {
            return false;
        }
    }
    if (!dist_simt_append_map(map, writer_address, writer_lo, writer_hi, writer_count, request.task_id)) return false;
    asc_threadfence();
    __gm__ uint64_t *completion =
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&task_cells[request.task_id].deps_prepared));
    return dist_simt_atomic_cas(completion, UINT64_MAX, request.task_id) == UINT64_MAX;
}

DIST_SIMT_CALLEE bool dist_simt_materialize_outputs(
    __gm__ fdwic::cross_core::CrossCoreOutputCell<Tensor> *outputs, __gm__ uint64_t *heap_cursor,
    __gm__ uint8_t *heap_base, uint64_t heap_size, const DistSimtRequestView &request, uint32_t build_owner,
    uint64_t output_bytes[], uint32_t *output_count, uint64_t *heap_begin, uint64_t *heap_end
) {
    *output_count = 0;
    uint64_t total = 0;
    for (uint32_t tensor = 0; tensor < request.tensor_count; ++tensor) {
        if (dist_simt_request_tag(request, tensor) != static_cast<uint32_t>(TensorArgType::OUTPUT)) continue;
        uint64_t bytes = 0;
        if (!dist_simt_output_size(dist_simt_request_tensor(request, tensor), &bytes) ||
            bytes > UINT64_MAX - (PTO2_PACKED_OUTPUT_ALIGN - 1U)) {
            return false;
        }
        output_bytes[*output_count] = bytes;
        const uint64_t aligned = (bytes + PTO2_PACKED_OUTPUT_ALIGN - 1U) & ~(PTO2_PACKED_OUTPUT_ALIGN - 1U);
        if (total > UINT64_MAX - aligned) return false;
        total += aligned;
        ++*output_count;
    }
    if (total != 0 && heap_base == nullptr) return false;
    const uint64_t begin = total == 0 ? dist_simt_atomic_load(heap_cursor) : asc_atomic_add(heap_cursor, total);
    if (begin > heap_size || total > heap_size - begin) return false;
    *heap_begin = begin;
    *heap_end = begin + total;

    uint64_t offset = 0;
    uint32_t output = 0;
    for (uint32_t tensor = 0; tensor < request.tensor_count; ++tensor) {
        if (dist_simt_request_tag(request, tensor) != static_cast<uint32_t>(TensorArgType::OUTPUT)) continue;
        __gm__ uint64_t *destination = reinterpret_cast<__gm__ uint64_t *>(&outputs->descriptors[output]);
        dist_simt_write_output_descriptor(
            destination, dist_simt_request_tensor(request, tensor),
            reinterpret_cast<uint64_t>(heap_base + begin + offset), output_bytes[output], request.task_id
        );
        offset += (output_bytes[output] + PTO2_PACKED_OUTPUT_ALIGN - 1U) & ~(PTO2_PACKED_OUTPUT_ALIGN - 1U);
        ++output;
    }
    if (output != *output_count || offset != total) return false;
    asc_threadfence();
    __gm__ uint64_t *control =
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&outputs->control.state));
    const uint64_t published = static_cast<uint64_t>(fdwic::cross_core::OutputPhase::Published) |
                               (static_cast<uint64_t>(*output_count) << fdwic::cross_core::kOutputStateCountShift) |
                               (static_cast<uint64_t>(build_owner) << fdwic::cross_core::kOutputStateOwnerShift) |
                               (static_cast<uint64_t>(request.task_id) << fdwic::cross_core::kOutputStateTaskIdShift);
    return dist_simt_atomic_cas(control, 0, published) == 0;
}

DIST_SIMT_CALLEE bool dist_simt_publish_exec(
    __gm__ fdwic::cross_core::SharedExecCell *cell, __gm__ DistTaskCell *task_cells,
    __gm__ fdwic::cross_core::CrossCoreOutputCell<Tensor> *outputs, const DistSimtRequestView &request,
    const int32_t fanin[], uint32_t fanin_count, uint32_t output_count, uint64_t heap_end, uint32_t build_owner
) {
    const uint32_t payload_bytes = fdwic::cross_core::kExecCacheLineBytes +
                                   request.tensor_count * fdwic::cross_core::kExecTensorDescBytes +
                                   request.scalar_count * sizeof(uint64_t) + fanin_count * sizeof(int32_t);
    const uint32_t payload_lines =
        (payload_bytes + fdwic::cross_core::kExecCacheLineBytes - 1U) / fdwic::cross_core::kExecCacheLineBytes;
    if (payload_lines == 0 || payload_lines > fdwic::cross_core::kExecMaxPayloadLines) return false;
    __gm__ uint64_t *control = reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&cell->control.state));
    const uint64_t building = dist_simt_exec_state(
        static_cast<uint32_t>(fdwic::cross_core::ExecPhase::Building), build_owner,
        fdwic::cross_core::kExecUnboundOwner, static_cast<uint32_t>(fdwic::cross_core::ExecEngineClass::None), 0,
        request.task_id
    );
    if (dist_simt_atomic_cas(control, 0, building) != 0) return false;

    __gm__ uint64_t *payload = const_cast<__gm__ uint64_t *>(cell->payload.words);
    dist_simt_store(payload + 0, request.task_id);
    dist_simt_store(payload + 1, request.function_address);
    dist_simt_store(payload + 2, heap_end);
    dist_simt_store(
        payload + 3, static_cast<uint64_t>(request.function_id) | (static_cast<uint64_t>(payload_bytes) << 32U)
    );
    dist_simt_store(
        payload + 4,
        static_cast<uint64_t>(request.tensor_count) | (static_cast<uint64_t>(request.scalar_count) << 16U) |
            (static_cast<uint64_t>(fanin_count) << 32U) | (static_cast<uint64_t>(request.engine_class) << 48U)
    );
    dist_simt_store(payload + 5, static_cast<uint64_t>(1U) << 48U);
    dist_simt_store(payload + 6, 0);
    dist_simt_store(payload + 7, 0);

    uint32_t destination = fdwic::cross_core::kExecHeaderWords;
    uint32_t output = 0;
    for (uint32_t tensor = 0; tensor < request.tensor_count; ++tensor) {
        __gm__ const uint64_t *source = nullptr;
        if (dist_simt_request_tag(request, tensor) == static_cast<uint32_t>(TensorArgType::OUTPUT)) {
            if (output >= output_count) return false;
            source = reinterpret_cast<__gm__ const uint64_t *>(&outputs->descriptors[output++]);
        } else {
            source = dist_simt_request_tensor(request, tensor);
        }
        dist_simt_copy_words(payload + destination, source, fdwic::cross_core::kExecTensorDescWords);
        destination += fdwic::cross_core::kExecTensorDescWords;
    }
    if (output != output_count) return false;
    const uint32_t scalar_offset =
        fdwic::cross_core::kSimtRequestHeaderWords + request.tensor_count * fdwic::cross_core::kExecTensorDescWords;
    for (uint32_t scalar = 0; scalar < request.scalar_count; ++scalar) {
        dist_simt_store(payload + destination++, request.words[scalar_offset + scalar]);
    }
    for (uint32_t edge = 0; edge < fanin_count; edge += 2U) {
        uint64_t packed = static_cast<uint32_t>(fanin[edge]);
        if (edge + 1U < fanin_count) packed |= static_cast<uint64_t>(static_cast<uint32_t>(fanin[edge + 1U])) << 32U;
        dist_simt_store(payload + destination++, packed);
    }
    const uint32_t expected_words = fdwic::cross_core::kExecHeaderWords +
                                    request.tensor_count * fdwic::cross_core::kExecTensorDescWords +
                                    request.scalar_count + (fanin_count + 1U) / 2U;
    if (destination != expected_words) return false;
    asc_threadfence();
    const uint64_t built = dist_simt_exec_state(
        static_cast<uint32_t>(fdwic::cross_core::ExecPhase::Built), build_owner, fdwic::cross_core::kExecUnboundOwner,
        request.engine_class, payload_lines, request.task_id
    );
    if (dist_simt_atomic_cas(control, building, built) != building) return false;
    if (request.engine_class != static_cast<uint32_t>(fdwic::cross_core::ExecEngineClass::Immediate)) return true;

    __gm__ uint64_t *task = reinterpret_cast<__gm__ uint64_t *>(&task_cells[request.task_id]);
    dist_simt_store(task + 1, heap_end);
    asc_threadfence();
    if (dist_simt_atomic_cas(task, 0, 1) != 0) return false;
    const uint64_t done = dist_simt_exec_state(
        static_cast<uint32_t>(fdwic::cross_core::ExecPhase::Done), build_owner, build_owner, request.engine_class,
        payload_lines, request.task_id
    );
    return dist_simt_atomic_cas(control, built, done) == built;
}

DIST_SIMT_CALLEE bool dist_simt_build_request(
    __gm__ SimtCrossCoreOrdinaryState *state, __gm__ DistTaskCell *task_cells, __gm__ uint8_t *heap_base,
    uint64_t heap_size, uint32_t history, const DistSimtRequestView &request, uint32_t build_owner
) {
    __gm__ uint64_t *fatal =
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->runtime.fatal.state));
    uint64_t output_bytes[fdwic::cross_core::kExecMaxTensors] = {};
    uint32_t output_count = 0;
    uint64_t heap_begin = 0;
    uint64_t heap_end = 0;
    __gm__ uint64_t *heap_cursor =
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->runtime.heap_cursor.state));
    __gm__ fdwic::cross_core::CrossCoreOutputCell<Tensor> *outputs = &state->runtime.outputs[request.task_id];
    if (!dist_simt_materialize_outputs(
            outputs, heap_cursor, heap_base, heap_size, request, build_owner, output_bytes, &output_count, &heap_begin,
            &heap_end
        )) {
        return false;
    }
    (void)heap_begin;

    int32_t fanin[fdwic::cross_core::kExecMaxFanin] = {};
    uint32_t fanin_count = 0;
    if (!dist_simt_prepare_map_and_fanin(
            &state->runtime.tensor_map, task_cells, request, history, fanin, &fanin_count, fatal
        )) {
        return false;
    }
    return dist_simt_publish_exec(
        &state->runtime.tasks[request.task_id], task_cells, outputs, request, fanin, fanin_count, output_count,
        heap_end, build_owner
    );
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(kDistSimtBuilderThreads) void DistSimtCrossCoreBuild(
    __gm__ SimtCrossCoreOrdinaryState *state, __gm__ DistTaskCell *task_cells, uint64_t heap_base_address,
    uint64_t heap_size, uint32_t history
) {
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    const uint32_t warp = thread / 32U;
    const uint32_t lane = thread % 32U;
    const bool active = lane == 0 && warp < kDistSimtBuilderWarps;
    __gm__ uint8_t *heap_base = reinterpret_cast<__gm__ uint8_t *>(heap_base_address);
    __gm__ uint64_t *fatal =
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->runtime.fatal.state));
    __gm__ uint64_t *started =
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->lifecycle.builder_started.state));
    __gm__ uint64_t *sealed =
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->lifecycle.sealed_task_count.state));
    __gm__ uint64_t *finished =
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->lifecycle.builder_finished.state));
    // builder_started is the first lifecycle field and is proven VF-accessible
    // on A5. Requests follow the three isolated control lines. Derive them from
    // this nearby base so the VF does not regenerate a far member address from
    // a compound state containing the large runtime arrays.
    __gm__ fdwic::cross_core::SimtBuildRequestCell *requests =
        reinterpret_cast<__gm__ fdwic::cross_core::SimtBuildRequestCell *>(
            reinterpret_cast<__gm__ uint8_t *>(started) + sizeof(SimtBuilderLifecycleState)
        );
    if (thread == 0) {
        // Match standalone SimtWaitBuilderStart instruction-for-instruction:
        // thread zero of the sole VF invocation atomically advances Reset to
        // Running.
        const uint64_t observed = asc_atomic_add(started, static_cast<uint64_t>(1U));
        if (observed != static_cast<uint64_t>(kDistSimtBuilderReset)) {
            dist_simt_publish_fatal(fatal, UINT32_MAX, 0);
        } else {
            asc_threadfence();
        }
    }
    // Production launches one persistent VF invocation. The standalone
    // builder_count rendezvous for multiple invocations has no business meaning
    // here. Once thread zero publishes Running, four leaders poll their request
    // streams directly. A leader arriving early naturally waits on an Empty
    // request while Scalars observe Running and publish work.
    if (active) {
        uint32_t task_id = warp;
        while (!dist_simt_fatal_observed(fatal)) {
            if (task_id >= kFdwicCrossCoreTaskCapacity) {
                const uint64_t count = dist_simt_atomic_load(sealed);
                if (count != UINT64_MAX && task_id >= count) break;
                dist_simt_publish_fatal(fatal, task_id, warp);
                break;
            }
            // Avoid re-deriving a far offset from the compound task/map state.
            __gm__ fdwic::cross_core::SimtBuildRequestCell *cell = &requests[task_id];
            __gm__ uint64_t *control =
                reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&cell->control.state));
            const uint64_t wait_begin = clock();
            bool acquired = false;
            DistSimtRequestView request{};
            while (clock() - wait_begin <= kDistSimtBuilderPollBudget) {
                const uint64_t raw = dist_simt_atomic_load(control);
                const uint32_t phase = static_cast<uint32_t>(raw & fdwic::cross_core::kSimtRequestPhaseMask);
                if (phase == static_cast<uint32_t>(fdwic::cross_core::SimtRequestPhase::Published)) {
                    acquired = dist_simt_decode_request(cell, task_id, &request);
                    if (!acquired) dist_simt_publish_fatal(fatal, task_id, warp);
                    break;
                }
                if (raw != 0 && phase != static_cast<uint32_t>(fdwic::cross_core::SimtRequestPhase::Reserved)) {
                    dist_simt_publish_fatal(fatal, task_id, warp);
                    break;
                }
                const uint64_t count = dist_simt_atomic_load(sealed);
                if (count != UINT64_MAX && task_id >= count) break;
                if (dist_simt_fatal_observed(fatal)) break;
            }
            if (!acquired) {
                const uint64_t count = dist_simt_atomic_load(sealed);
                if (!dist_simt_fatal_observed(fatal) && count != UINT64_MAX && task_id >= count) break;
                if (!dist_simt_fatal_observed(fatal)) dist_simt_publish_fatal(fatal, task_id, warp);
                break;
            }
            if (!dist_simt_build_request(state, task_cells, heap_base, heap_size, history, request, warp)) {
                dist_simt_publish_fatal(fatal, task_id, warp);
                break;
            }
            task_id += kDistSimtBuilderWarps;
        }
        const uint64_t prior_finished = asc_atomic_add(finished, static_cast<uint64_t>(1U));
        if (prior_finished >= kDistSimtBuilderWarps) dist_simt_publish_fatal(fatal, UINT32_MAX, warp);
    }
}

#undef DIST_SIMT_CALLEE

#endif  // defined(__CCE_AICORE__) && defined(__DAV_VEC__) && defined(PTO_FDWIC_SIMT_DEFINE_BUILDER_VF)

#if !defined(PTO_FDWIC_SIMT_METADATA_ONLY)

#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)
extern "C" PTO_DEVICE_FUNC void fdwic_simt_cross_core_run_builder(
    __gm__ SimtCrossCoreOrdinaryState *state, __gm__ DistTaskCell *task_cells, uint64_t heap_base_address,
    uint64_t heap_size, uint32_t history
);
#endif

PTO_DEVICE_FUNC bool dist_simt_cross_core_launch_builder(__gm__ DistCore *self) {
#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)
    __gm__ SimtCrossCoreOrdinaryState &state = g_dist.simt_cross_core_ordinary;
    if (!dist_simt_cross_core_is_builder_worker(self)) return true;
    if (DistCrossCoreAicoreOps::Load(&state.lifecycle.builder_started.state) != kDistSimtBuilderReset) {
        return dist_simt_cross_core_fail(self != nullptr ? self->local_index : -1, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
    const uint64_t heap_base_address = reinterpret_cast<uint64_t>(g_dist.heap_base);
    // The VF definition and async_invoke must belong to the registered entry's
    // translation unit so its metadata and VF code offset share one compiler
    // contract. dist_engine only passes dynamic state and does not emit a
    // second local VF.
    fdwic_simt_cross_core_run_builder(
        &state, &g_dist.tasks[0], heap_base_address, static_cast<uint64_t>(g_dist.heap_size),
        static_cast<uint32_t>(g_dist.H)
    );
#else
    (void)self;
#endif
    return true;
}

PTO_DEVICE_FUNC bool dist_simt_cross_core_seal_requests(__gm__ DistCore *self) {
#if defined(__CCE_AICORE__)
    if (self == nullptr || self->local_index < 0 ||
        static_cast<uint32_t>(self->local_index) > kFdwicCrossCoreTaskCapacity) {
        return dist_simt_cross_core_fail(self != nullptr ? self->local_index : -1, PTO2_ERROR_TENSORMAP_CAPACITY);
    }
    __gm__ volatile int64_t &sealed = g_dist.simt_cross_core_ordinary.lifecycle.sealed_task_count.state;
    const int64_t count = self->local_index;
    const int64_t observed = DistCrossCoreAicoreOps::CompareExchange(&sealed, -1, count);
    if (observed != -1 && observed != count) {
        return dist_simt_cross_core_fail(self->local_index, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
#else
    (void)self;
#endif
    return true;
}

PTO_DEVICE_FUNC bool dist_simt_cross_core_join_builder(__gm__ DistCore *self) {
#if defined(__CCE_AICORE__) && defined(__DAV_VEC__)
    if (!dist_simt_cross_core_is_builder_worker(self)) return true;
    // The launch helper already consumed the completion event. Check only the
    // builder's GM completion contract here; a second wait_flag on the same
    // event is invalid.
    __gm__ SimtCrossCoreOrdinaryState &state = g_dist.simt_cross_core_ordinary;
    if (DistCrossCoreAicoreOps::Load(&state.lifecycle.builder_started.state) != kDistSimtBuilderRunning ||
        DistCrossCoreAicoreOps::Load(&state.lifecycle.builder_finished.state) != kDistSimtBuilderWarps ||
        DistCrossCoreAicoreOps::Load(&state.runtime.fatal.state) != 0) {
        return dist_simt_cross_core_fail(self != nullptr ? self->local_index : -1, PTO2_ERROR_TENSORMAP_PROTOCOL);
    }
#else
    (void)self;
#endif
    return true;
}

#endif  // !defined(PTO_FDWIC_SIMT_METADATA_ONLY)

}  // namespace

#endif  // PTO_FDWIC_SCHEDULER_MODE == 3
