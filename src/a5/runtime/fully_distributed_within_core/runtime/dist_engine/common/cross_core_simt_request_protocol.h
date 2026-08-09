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

#include <cstddef>
#include <cstdint>

#include "dist_engine/common/cross_core_exec_protocol.h"
#include "tensor.h"

namespace fdwic::cross_core {

constexpr uint32_t kSimtRequestHeaderWords = kExecCacheLineBytes / sizeof(uint64_t);
constexpr uint32_t kSimtRequestMaxExplicitDependencies = 16;
constexpr uint32_t kSimtRequestMaxPayloadBytes =
    kExecCacheLineBytes + kExecMaxTensors * kExecTensorDescBytes +
    (kExecMaxScalars + kSimtRequestMaxExplicitDependencies) * sizeof(uint64_t);
constexpr uint32_t kSimtRequestMaxPayloadLines =
    (kSimtRequestMaxPayloadBytes + kExecCacheLineBytes - 1U) / kExecCacheLineBytes;
constexpr uint32_t kSimtRequestMaxPayloadWords = kSimtRequestMaxPayloadBytes / sizeof(uint64_t);

static_assert(kSimtRequestMaxPayloadBytes == 4416);
static_assert(kSimtRequestMaxPayloadLines == 69);
static_assert(kSimtRequestMaxPayloadBytes % kExecCacheLineBytes == 0);

enum class SimtRequestPhase : uint8_t {
    Empty = 0,
    Reserved = 1,
    Published = 2,
};

enum class SimtRequestReserveResult : uint8_t {
    Reserved = 0,
    InvalidInput = 1,
    CellUnavailable = 2,
    FatalObserved = 3,
};

enum class SimtRequestPublishResult : uint8_t {
    Published = 0,
    InvalidInput = 1,
    PublishConflict = 2,
    FatalObserved = 3,
};

enum class SimtRequestAcquireResult : uint8_t {
    Acquired = 0,
    NotPublished = 1,
    InvalidControl = 2,
    InvalidPayload = 3,
    FatalObserved = 4,
};

constexpr uint64_t kSimtRequestPhaseShift = 0;
constexpr uint64_t kSimtRequestPhaseMask = 0x3ULL;
constexpr uint64_t kSimtRequestPublisherShift = 2;
constexpr uint64_t kSimtRequestPublisherMask = 0xFFULL;
constexpr uint64_t kSimtRequestPayloadLinesShift = 10;
constexpr uint64_t kSimtRequestPayloadLinesMask = 0x7FULL;
constexpr uint64_t kSimtRequestTaskIdShift = 17;
constexpr uint64_t kSimtRequestTaskIdMask = 0xFFFFFFFFULL;
constexpr uint64_t kSimtRequestKnownMask = (kSimtRequestPhaseMask << kSimtRequestPhaseShift) |
                                           (kSimtRequestPublisherMask << kSimtRequestPublisherShift) |
                                           (kSimtRequestPayloadLinesMask << kSimtRequestPayloadLinesShift) |
                                           (kSimtRequestTaskIdMask << kSimtRequestTaskIdShift);

struct DecodedSimtRequestControl {
    SimtRequestPhase phase;
    uint32_t publisher_owner;
    uint32_t payload_lines;
    uint32_t task_id;
    bool valid;
};

struct SimtBuildRequestSpec {
    uint32_t task_id;
    uint64_t function_address;
    uint32_t function_id;
    uint16_t tensor_count;
    uint16_t scalar_count;
    uint16_t explicit_dep_count;
    ExecEngineClass engine_class;
    uint8_t flags;
    uint32_t tensor_reference_mask;
};

struct SimtBuildRequestHeader {
    uint32_t task_id;
    uint32_t function_id;
    uint64_t function_address;
    uint16_t tensor_count;
    uint16_t scalar_count;
    uint16_t explicit_dep_count;
    ExecEngineClass engine_class;
    uint8_t flags;
    uint8_t tensor_tags[kExecMaxTensors];
    uint32_t tensor_reference_mask;
    uint32_t reserved;
};

struct SimtBuildRequestLayout {
    uint32_t payload_bytes;
    uint32_t payload_lines;
    uint32_t tensor_word_offset;
    uint32_t scalar_word_offset;
    uint32_t explicit_dep_word_offset;
    uint32_t written_words;
    uint32_t tensor_reference_mask;
    uint32_t inline_tensor_count;
};

struct alignas(kExecCacheLineBytes) SimtBuildRequestStorage {
    volatile uint64_t words[kSimtRequestMaxPayloadWords];
};

struct alignas(kExecCacheLineBytes) SimtBuildRequestCell {
    SharedExecControl control;
    SimtBuildRequestStorage payload;
};

static_assert(sizeof(SimtBuildRequestHeader) == kExecCacheLineBytes);
static_assert(sizeof(SimtBuildRequestStorage) == kSimtRequestMaxPayloadBytes);
static_assert(offsetof(SimtBuildRequestCell, payload) == kExecCacheLineBytes);
static_assert(sizeof(SimtBuildRequestCell) == kExecCacheLineBytes + kSimtRequestMaxPayloadBytes);
static_assert(alignof(SimtBuildRequestCell) == kExecCacheLineBytes);

PTO_DEVICE_FUNC inline uint64_t
EncodeSimtRequestControl(SimtRequestPhase phase, uint32_t publisher_owner, uint32_t payload_lines, uint32_t task_id) {
    return (static_cast<uint64_t>(phase) << kSimtRequestPhaseShift) |
           (static_cast<uint64_t>(publisher_owner) << kSimtRequestPublisherShift) |
           (static_cast<uint64_t>(payload_lines) << kSimtRequestPayloadLinesShift) |
           (static_cast<uint64_t>(task_id) << kSimtRequestTaskIdShift);
}

PTO_DEVICE_FUNC inline DecodedSimtRequestControl DecodeSimtRequestControl(int64_t raw_state) {
    const uint64_t raw = static_cast<uint64_t>(raw_state);
    DecodedSimtRequestControl decoded{
        static_cast<SimtRequestPhase>((raw >> kSimtRequestPhaseShift) & kSimtRequestPhaseMask),
        static_cast<uint32_t>((raw >> kSimtRequestPublisherShift) & kSimtRequestPublisherMask),
        static_cast<uint32_t>((raw >> kSimtRequestPayloadLinesShift) & kSimtRequestPayloadLinesMask),
        static_cast<uint32_t>((raw >> kSimtRequestTaskIdShift) & kSimtRequestTaskIdMask),
        false,
    };
    if ((raw & ~kSimtRequestKnownMask) != 0) return decoded;
    switch (decoded.phase) {
    case SimtRequestPhase::Empty:
        decoded.valid = raw == 0;
        break;
    case SimtRequestPhase::Reserved:
        decoded.valid = ExecOwnerValid(decoded.publisher_owner) && decoded.payload_lines == 0;
        break;
    case SimtRequestPhase::Published:
        decoded.valid = ExecOwnerValid(decoded.publisher_owner) && decoded.payload_lines >= 1 &&
                        decoded.payload_lines <= kSimtRequestMaxPayloadLines;
        break;
    default:
        break;
    }
    return decoded;
}

PTO_DEVICE_FUNC inline bool SimtRequestTensorTagValid(TensorArgType tag) {
    switch (tag) {
    case TensorArgType::INPUT:
    case TensorArgType::OUTPUT:
    case TensorArgType::INOUT:
    case TensorArgType::OUTPUT_EXISTING:
    case TensorArgType::NO_DEP:
        return true;
    }
    return false;
}

PTO_DEVICE_FUNC inline uint32_t SimtRequestTensorMaskForCount(uint32_t tensor_count) {
    return tensor_count >= 32U ? UINT32_MAX : ((uint32_t{1} << tensor_count) - 1U);
}

PTO_DEVICE_FUNC inline uint32_t SimtRequestReferenceCount(uint32_t reference_mask) {
    uint32_t count = 0;
    while (reference_mask != 0) {
        count += reference_mask & 1U;
        reference_mask >>= 1U;
    }
    return count;
}

PTO_DEVICE_FUNC inline bool DecodeSimtRequestOutputReference(
    uint64_t encoded, uint32_t consumer_task_id, uint32_t &producer_task_id, uint32_t &output_slot
) {
    const int32_t producer = static_cast<int32_t>(static_cast<uint32_t>(encoded));
    const int16_t slot = static_cast<int16_t>(static_cast<uint16_t>(encoded >> 32U));
    const uint16_t flags_and_view_rank = static_cast<uint16_t>(encoded >> 48U);
    if (producer < 0 || static_cast<uint32_t>(producer) >= consumer_task_id || slot < 0 ||
        static_cast<uint32_t>(slot) >= kExecMaxTensors || flags_and_view_rank != 0) {
        return false;
    }
    producer_task_id = static_cast<uint32_t>(producer);
    output_slot = static_cast<uint32_t>(slot);
    return true;
}

PTO_DEVICE_FUNC inline uint32_t SimtRequestTensorWordOffset(uint32_t tensor, uint32_t reference_mask) {
    const uint32_t preceding_mask = tensor == 0U ? 0U : reference_mask & SimtRequestTensorMaskForCount(tensor);
    return kSimtRequestHeaderWords + tensor * kExecTensorDescWords -
           SimtRequestReferenceCount(preceding_mask) * (kExecTensorDescWords - 1U);
}

PTO_DEVICE_FUNC inline bool ComputeSimtBuildRequestLayout(
    uint32_t tensor_count, uint32_t scalar_count, uint32_t explicit_dep_count, uint32_t tensor_reference_mask,
    SimtBuildRequestLayout &layout
) {
    if (tensor_count > kExecMaxTensors || scalar_count > kExecMaxScalars ||
        explicit_dep_count > kSimtRequestMaxExplicitDependencies ||
        (tensor_reference_mask & ~SimtRequestTensorMaskForCount(tensor_count)) != 0) {
        return false;
    }
    const uint32_t reference_count = SimtRequestReferenceCount(tensor_reference_mask);
    const uint32_t inline_tensor_count = tensor_count - reference_count;
    layout.tensor_word_offset = kSimtRequestHeaderWords;
    layout.scalar_word_offset =
        layout.tensor_word_offset + inline_tensor_count * kExecTensorDescWords + reference_count;
    layout.explicit_dep_word_offset = layout.scalar_word_offset + scalar_count;
    layout.written_words = layout.explicit_dep_word_offset + explicit_dep_count;
    layout.payload_bytes = layout.written_words * sizeof(uint64_t);
    layout.payload_lines = (layout.payload_bytes + kExecCacheLineBytes - 1U) / kExecCacheLineBytes;
    layout.tensor_reference_mask = tensor_reference_mask;
    layout.inline_tensor_count = inline_tensor_count;
    return layout.payload_lines >= 1 && layout.payload_lines <= kSimtRequestMaxPayloadLines &&
           layout.written_words <= kSimtRequestMaxPayloadWords;
}

PTO_DEVICE_FUNC inline bool ComputeSimtBuildRequestLayout(
    uint32_t tensor_count, uint32_t scalar_count, uint32_t explicit_dep_count, SimtBuildRequestLayout &layout
) {
    return ComputeSimtBuildRequestLayout(tensor_count, scalar_count, explicit_dep_count, 0, layout);
}

PTO_DEVICE_FUNC inline bool
ValidateSimtBuildRequestSpec(const SimtBuildRequestSpec &spec, SimtBuildRequestLayout &layout) {
    if (spec.task_id == UINT32_MAX || spec.flags != 0 ||
        (spec.engine_class != ExecEngineClass::Aic && spec.engine_class != ExecEngineClass::Aiv &&
         spec.engine_class != ExecEngineClass::Immediate)) {
        return false;
    }
    const bool immediate = spec.engine_class == ExecEngineClass::Immediate;
    if ((immediate && (spec.function_id != kExecInvalidFunctionId || spec.function_address != 0)) ||
        (!immediate && spec.function_id == kExecInvalidFunctionId && spec.function_address == 0)) {
        return false;
    }
    return ComputeSimtBuildRequestLayout(
        spec.tensor_count, spec.scalar_count, spec.explicit_dep_count, spec.tensor_reference_mask, layout
    );
}

PTO_DEVICE_FUNC inline uint32_t SimtRequestTensorWordOffset(uint32_t tensor) {
    return SimtRequestTensorWordOffset(tensor, 0);
}

PTO_DEVICE_FUNC inline uint64_t PackSimtRequestCounts(const SimtBuildRequestSpec &spec) {
    return static_cast<uint64_t>(spec.tensor_count) | (static_cast<uint64_t>(spec.scalar_count) << 16U) |
           (static_cast<uint64_t>(spec.explicit_dep_count) << 32U) | (static_cast<uint64_t>(spec.engine_class) << 48U) |
           (static_cast<uint64_t>(spec.flags) << 56U);
}

template <typename Ops, typename Source>
PTO_DEVICE_FUNC bool PackSimtBuildRequest(
    __gm__ SimtBuildRequestCell &cell, const SimtBuildRequestSpec &spec, const Source &source,
    SimtBuildRequestLayout &layout
) {
    if (!ValidateSimtBuildRequestSpec(spec, layout)) return false;
    __gm__ volatile uint64_t *destination = cell.payload.words;
    Ops::StorePayloadWord(
        &destination[0], static_cast<uint64_t>(spec.task_id) | (static_cast<uint64_t>(spec.function_id) << 32U)
    );
    Ops::StorePayloadWord(&destination[1], spec.function_address);
    Ops::StorePayloadWord(&destination[2], PackSimtRequestCounts(spec));

    for (uint32_t group = 0; group < 4; ++group) {
        uint64_t packed_tags = 0;
        for (uint32_t within_group = 0; within_group < 8; ++within_group) {
            const uint32_t tensor = group * 8U + within_group;
            uint8_t tag = 0;
            if (tensor < spec.tensor_count) {
                const TensorArgType source_tag = source.TensorTag(tensor);
                if (!SimtRequestTensorTagValid(source_tag)) return false;
                tag = static_cast<uint8_t>(source_tag);
            }
            packed_tags |= static_cast<uint64_t>(tag) << (within_group * 8U);
        }
        Ops::StorePayloadWord(&destination[3U + group], packed_tags);
    }
    Ops::StorePayloadWord(&destination[7], spec.tensor_reference_mask);

    uint32_t word = layout.tensor_word_offset;
    for (uint32_t tensor = 0; tensor < spec.tensor_count; ++tensor) {
        const TensorArgType tag = source.TensorTag(tensor);
        const bool packed_as_reference = (spec.tensor_reference_mask & (uint32_t{1} << tensor)) != 0;
        if (packed_as_reference != source.TensorIsReference(tensor)) return false;
        if (packed_as_reference) {
            if (tag == TensorArgType::OUTPUT) return false;
            uint32_t producer = 0;
            uint32_t output_slot = 0;
            const uint64_t reference = source.TensorWord(tensor, 0);
            if (!DecodeSimtRequestOutputReference(reference, spec.task_id, producer, output_slot)) return false;
            Ops::StorePayloadWord(&destination[word++], reference);
            continue;
        }
        const uint32_t copied_words = tag == TensorArgType::OUTPUT ? 8U : kExecTensorDescWords;
        for (uint32_t desc_word = 0; desc_word < copied_words; ++desc_word) {
            Ops::StorePayloadWord(&destination[word++], source.TensorWord(tensor, desc_word));
        }
        for (uint32_t desc_word = copied_words; desc_word < kExecTensorDescWords; ++desc_word) {
            Ops::StorePayloadWord(&destination[word++], 0);
        }
    }
    for (uint32_t scalar = 0; scalar < spec.scalar_count; ++scalar) {
        Ops::StorePayloadWord(&destination[word++], source.Scalar(scalar));
    }
    for (uint32_t dependency = 0; dependency < spec.explicit_dep_count; ++dependency) {
        Ops::StorePayloadWord(&destination[word++], source.ExplicitDependency(dependency));
    }
    return word == layout.written_words;
}

PTO_DEVICE_FUNC inline SimtBuildRequestHeader
DecodeSimtBuildRequestHeader(__gm__ const SimtBuildRequestStorage &payload) {
    const uint64_t identity = payload.words[0];
    const uint64_t counts = payload.words[2];
    SimtBuildRequestHeader header{
        static_cast<uint32_t>(identity),
        static_cast<uint32_t>(identity >> 32U),
        payload.words[1],
        static_cast<uint16_t>(counts),
        static_cast<uint16_t>(counts >> 16U),
        static_cast<uint16_t>(counts >> 32U),
        static_cast<ExecEngineClass>(static_cast<uint8_t>(counts >> 48U)),
        static_cast<uint8_t>(counts >> 56U),
        {},
        static_cast<uint32_t>(payload.words[7]),
        static_cast<uint32_t>(payload.words[7] >> 32U),
    };
    for (uint32_t tensor = 0; tensor < kExecMaxTensors; ++tensor) {
        const uint64_t packed_tags = payload.words[3U + tensor / 8U];
        header.tensor_tags[tensor] = static_cast<uint8_t>(packed_tags >> ((tensor % 8U) * 8U));
    }
    return header;
}

PTO_DEVICE_FUNC inline bool ValidateSimtBuildRequestPayload(
    __gm__ const SimtBuildRequestStorage &payload, uint32_t expected_task_id, uint32_t published_lines,
    SimtBuildRequestHeader &header, SimtBuildRequestLayout &layout
) {
    header = DecodeSimtBuildRequestHeader(payload);
    const SimtBuildRequestSpec spec{
        header.task_id,
        header.function_address,
        header.function_id,
        header.tensor_count,
        header.scalar_count,
        header.explicit_dep_count,
        header.engine_class,
        header.flags,
        header.tensor_reference_mask,
    };
    if (header.reserved != 0 || header.task_id != expected_task_id || !ValidateSimtBuildRequestSpec(spec, layout) ||
        layout.payload_lines != published_lines) {
        return false;
    }
    for (uint32_t tensor = 0; tensor < header.tensor_count; ++tensor) {
        const TensorArgType tag = static_cast<TensorArgType>(header.tensor_tags[tensor]);
        if (!SimtRequestTensorTagValid(tag)) return false;
        if ((header.tensor_reference_mask & (uint32_t{1} << tensor)) == 0) continue;
        if (tag == TensorArgType::OUTPUT) return false;
        uint32_t producer = 0;
        uint32_t output_slot = 0;
        const uint64_t reference = payload.words[SimtRequestTensorWordOffset(tensor, header.tensor_reference_mask)];
        if (!DecodeSimtRequestOutputReference(reference, header.task_id, producer, output_slot)) return false;
    }
    for (uint32_t tensor = header.tensor_count; tensor < kExecMaxTensors; ++tensor) {
        if (header.tensor_tags[tensor] != 0) return false;
    }
    return true;
}

template <typename Ops>
PTO_DEVICE_FUNC SimtRequestReserveResult ReserveSimtBuildRequest(
    __gm__ SimtBuildRequestCell &cell, uint32_t task_id, uint32_t publisher_owner, __gm__ SharedExecControl &fatal
) {
    if (Ops::Load(&fatal.state) != 0) return SimtRequestReserveResult::FatalObserved;
    if (task_id == UINT32_MAX || !ExecOwnerValid(publisher_owner)) return SimtRequestReserveResult::InvalidInput;
    const int64_t reserved =
        static_cast<int64_t>(EncodeSimtRequestControl(SimtRequestPhase::Reserved, publisher_owner, 0, task_id));
    if (Ops::CompareExchange(&cell.control.state, 0, reserved) != 0) {
        return SimtRequestReserveResult::CellUnavailable;
    }
    return SimtRequestReserveResult::Reserved;
}

template <typename Ops, typename Source>
PTO_DEVICE_FUNC SimtRequestPublishResult PublishReservedSimtBuildRequest(
    __gm__ SimtBuildRequestCell &cell, uint32_t publisher_owner, const SimtBuildRequestSpec &spec, const Source &source,
    __gm__ SharedExecControl &fatal
) {
    if (Ops::Load(&fatal.state) != 0) return SimtRequestPublishResult::FatalObserved;
    SimtBuildRequestLayout expected{};
    if (!ExecOwnerValid(publisher_owner) || !ValidateSimtBuildRequestSpec(spec, expected)) {
        return SimtRequestPublishResult::InvalidInput;
    }
    const int64_t reserved =
        static_cast<int64_t>(EncodeSimtRequestControl(SimtRequestPhase::Reserved, publisher_owner, 0, spec.task_id));
    if (Ops::Load(&cell.control.state) != reserved) return SimtRequestPublishResult::PublishConflict;

    SimtBuildRequestLayout packed{};
    if (!PackSimtBuildRequest<Ops>(cell, spec, source, packed) || packed.payload_bytes != expected.payload_bytes) {
        return SimtRequestPublishResult::InvalidInput;
    }
    Ops::FlushRegion(&cell.payload, static_cast<uint64_t>(packed.payload_lines) * kExecCacheLineBytes);
    const int64_t published = static_cast<int64_t>(
        EncodeSimtRequestControl(SimtRequestPhase::Published, publisher_owner, packed.payload_lines, spec.task_id)
    );
    // The reserve CAS already elected one publisher. The A5 cross-execution-unit
    // publication contract is a local plain store plus DCCI on an isolated
    // control line, followed by a remote atomic load. A second Scalar CAS can
    // leave the VF's returned atomic load observing the old value indefinitely.
    // Exact reads before and after publication retain fail-closed validation.
    Ops::StorePayloadWord(
        reinterpret_cast<__gm__ volatile uint64_t *>(&cell.control.state), static_cast<uint64_t>(published)
    );
    Ops::FlushRegion(&cell.control, kExecCacheLineBytes);
    if (Ops::Load(&cell.control.state) != published) {
        return SimtRequestPublishResult::PublishConflict;
    }
    return SimtRequestPublishResult::Published;
}

template <typename Ops>
PTO_DEVICE_FUNC SimtRequestAcquireResult AcquireSimtBuildRequest(
    __gm__ SimtBuildRequestCell &cell, uint32_t task_id, SimtBuildRequestHeader &header, SimtBuildRequestLayout &layout,
    __gm__ SharedExecControl &fatal
) {
    if (Ops::Load(&fatal.state) != 0) return SimtRequestAcquireResult::FatalObserved;
    const int64_t observed_raw = Ops::Load(&cell.control.state);
    const DecodedSimtRequestControl observed = DecodeSimtRequestControl(observed_raw);
    if (!observed.valid) return SimtRequestAcquireResult::InvalidControl;
    if (observed.phase != SimtRequestPhase::Published) return SimtRequestAcquireResult::NotPublished;
    if (observed.task_id != task_id) return SimtRequestAcquireResult::InvalidControl;

    Ops::InvalidateRegion(&cell.payload, static_cast<uint64_t>(observed.payload_lines) * kExecCacheLineBytes);
    if (Ops::Load(&cell.control.state) != observed_raw) return SimtRequestAcquireResult::InvalidControl;
    if (!ValidateSimtBuildRequestPayload(cell.payload, task_id, observed.payload_lines, header, layout)) {
        return SimtRequestAcquireResult::InvalidPayload;
    }
    return SimtRequestAcquireResult::Acquired;
}

}  // namespace fdwic::cross_core
