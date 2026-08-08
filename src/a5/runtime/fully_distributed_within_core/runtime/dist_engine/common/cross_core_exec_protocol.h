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

#include "data_type.h"
#include "dist_engine/common/target.h"
#include "intrinsic.h"

namespace fdwic::cross_core {

constexpr uint32_t kExecCacheLineBytes = 64;
constexpr uint32_t kExecTensorDescBytes = 128;
constexpr uint32_t kExecTensorDescWords = kExecTensorDescBytes / sizeof(uint64_t);
constexpr uint32_t kExecMaxTensors = 32;
constexpr uint32_t kExecMaxScalars = 16;
constexpr uint32_t kExecMaxFanin = 16;
constexpr uint32_t kExecHeaderWords = kExecCacheLineBytes / sizeof(uint64_t);
constexpr uint32_t kExecMaxPayloadBytes = kExecCacheLineBytes + kExecMaxTensors * kExecTensorDescBytes +
                                          kExecMaxScalars * sizeof(uint64_t) + kExecMaxFanin * sizeof(int32_t);
constexpr uint32_t kExecMaxPayloadLines = (kExecMaxPayloadBytes + kExecCacheLineBytes - 1U) / kExecCacheLineBytes;
constexpr uint32_t kExecMaxPayloadWords = kExecMaxPayloadLines * kExecHeaderWords;
constexpr uint32_t kExecInvalidFunctionId = UINT32_MAX;
constexpr uint32_t kExecMaxOwner = 254;
constexpr uint32_t kExecUnboundOwner = 255;

static_assert(kExecTensorDescBytes % sizeof(uint64_t) == 0);
static_assert(kExecMaxPayloadBytes == 4352);
static_assert(kExecMaxPayloadLines == 68);

enum class ExecPhase : uint8_t {
    Empty = 0,
    Building = 1,
    Built = 2,
    Claimed = 3,
    Done = 4,
};

enum class ExecEngineClass : uint8_t {
    None = 0,
    Aic = 1,
    Aiv = 2,
    Joint = 3,
    Immediate = 4,
};

enum class ExecTokenPhase : uint8_t {
    Idle = 0,
    Acquired = 1,
    Faulted = 2,
};

enum class ExecFatalReason : uint8_t {
    None = 0,
    InvalidBuildInput = 1,
    BuildPackFailed = 2,
    InvalidBuiltControl = 3,
    AcquiredPayloadInvalid = 4,
    ControlPublishConflict = 5,
    CompletionStateConflict = 6,
};

enum class ExecBuildResult : uint8_t {
    Published = 0,
    InvalidInput = 1,
    CellUnavailable = 2,
    PublishConflict = 3,
    FatalObserved = 4,
};

enum class ExecBuildReserveResult : uint8_t {
    Reserved = 0,
    InvalidInput = 1,
    CellUnavailable = 2,
    FatalObserved = 3,
};

enum class ExecAcquireResult : uint8_t {
    Acquired = 0,
    TokenBusy = 1,
    NotBuilt = 2,
    Incompatible = 3,
    Lost = 4,
    InvalidControl = 5,
    InvalidPayload = 6,
    FatalObserved = 7,
};

enum class ExecDoneResult : uint8_t {
    Done = 0,
    TokenNotAcquired = 1,
    StateConflict = 2,
    FatalObserved = 3,
};

constexpr uint64_t kExecStatePhaseShift = 0;
constexpr uint64_t kExecStatePhaseMask = 0x7ULL;
constexpr uint64_t kExecStateBuildOwnerShift = 3;
constexpr uint64_t kExecStateBuildOwnerMask = 0xFFULL;
constexpr uint64_t kExecStateExecuteOwnerShift = 11;
constexpr uint64_t kExecStateExecuteOwnerMask = 0xFFULL;
constexpr uint64_t kExecStateEngineShift = 19;
constexpr uint64_t kExecStateEngineMask = 0x7ULL;
constexpr uint64_t kExecStatePayloadLinesShift = 22;
constexpr uint64_t kExecStatePayloadLinesMask = 0x7FULL;
constexpr uint64_t kExecStateTaskIdShift = 29;
constexpr uint64_t kExecStateTaskIdMask = 0xFFFFFFFFULL;
constexpr uint64_t kExecStateKnownMask =
    (kExecStatePhaseMask << kExecStatePhaseShift) | (kExecStateBuildOwnerMask << kExecStateBuildOwnerShift) |
    (kExecStateExecuteOwnerMask << kExecStateExecuteOwnerShift) | (kExecStateEngineMask << kExecStateEngineShift) |
    (kExecStatePayloadLinesMask << kExecStatePayloadLinesShift) | (kExecStateTaskIdMask << kExecStateTaskIdShift);

constexpr uint64_t kExecFatalReasonMask = 0xFFULL;
constexpr uint64_t kExecFatalOwnerShift = 8;
constexpr uint64_t kExecFatalOwnerMask = 0xFFULL;
constexpr uint64_t kExecFatalTaskIdShift = 16;
constexpr uint64_t kExecFatalTaskIdMask = 0xFFFFFFFFULL;
constexpr uint64_t kExecFatalKnownMask = kExecFatalReasonMask | (kExecFatalOwnerMask << kExecFatalOwnerShift) |
                                         (kExecFatalTaskIdMask << kExecFatalTaskIdShift);

struct DecodedExecState {
    ExecPhase phase;
    uint32_t build_owner;
    uint32_t execute_owner;
    ExecEngineClass engine_class;
    uint32_t payload_lines;
    uint32_t task_id;
    bool valid;
};

struct ExecPayloadLayout {
    uint32_t payload_bytes;
    uint32_t payload_lines;
    uint32_t tensor_word_offset;
    uint32_t scalar_word_offset;
    uint32_t fanin_word_offset;
    uint32_t written_words;
    uint32_t tensor_reference_mask;
};

struct ExecPayloadSpec {
    uint32_t task_id;
    uint64_t function_address;
    uint64_t completion_vend;
    uint32_t function_id;
    uint16_t tensor_count;
    uint16_t scalar_count;
    uint16_t fanin_count;
    ExecEngineClass engine_class;
    uint8_t flags;
    uint32_t multicore_group_id;
    uint16_t multicore_rank;
    uint16_t multicore_size;
    // A set bit stores a stable GM Tensor descriptor address instead of an
    // inline descriptor. The consumer must invalidate that descriptor before
    // dereferencing it on a non-coherent Scalar.
    uint32_t tensor_reference_mask;
};

struct ExecPayloadHeader {
    uint32_t task_id;
    uint64_t function_address;
    uint64_t completion_vend;
    uint32_t function_id;
    uint32_t payload_bytes;
    uint16_t tensor_count;
    uint16_t scalar_count;
    uint16_t fanin_count;
    ExecEngineClass engine_class;
    uint8_t flags;
    uint32_t multicore_group_id;
    uint16_t multicore_rank;
    uint16_t multicore_size;
    uint32_t tensor_reference_mask;
};

struct alignas(kExecCacheLineBytes) SharedExecControl {
    volatile int64_t state;
    uint8_t padding[kExecCacheLineBytes - sizeof(int64_t)];
};

struct alignas(kExecCacheLineBytes) ExecPayloadStorage {
    volatile uint64_t words[kExecMaxPayloadWords];
};

struct alignas(kExecCacheLineBytes) SharedExecCell {
    SharedExecControl control;
    ExecPayloadStorage payload;
};

struct ExecToken {
    ExecTokenPhase phase;
    uint8_t reserved[3];
    uint32_t task_id;
    uint32_t build_owner;
    uint32_t execute_owner;
    ExecEngineClass engine_class;
    uint8_t payload_lines;
    uint16_t reserved2;
    uint64_t payload_address;
    ExecPayloadHeader header;
};

static_assert(sizeof(SharedExecControl) == kExecCacheLineBytes);
static_assert(alignof(SharedExecControl) == kExecCacheLineBytes);
static_assert(offsetof(SharedExecCell, payload) == kExecCacheLineBytes);
static_assert(sizeof(ExecPayloadStorage) == kExecMaxPayloadBytes);
static_assert(sizeof(SharedExecCell) == kExecCacheLineBytes + kExecMaxPayloadBytes);

PTO_DEVICE_FUNC inline bool ExecOwnerValid(uint32_t owner) { return owner <= kExecMaxOwner; }

PTO_DEVICE_FUNC inline bool ExecEngineValid(ExecEngineClass engine_class) {
    return engine_class == ExecEngineClass::Aic || engine_class == ExecEngineClass::Aiv ||
           engine_class == ExecEngineClass::Joint || engine_class == ExecEngineClass::Immediate;
}

PTO_DEVICE_FUNC inline bool ExecExecutorEngineValid(ExecEngineClass engine_class) {
    return engine_class == ExecEngineClass::Aic || engine_class == ExecEngineClass::Aiv ||
           engine_class == ExecEngineClass::Joint;
}

PTO_DEVICE_FUNC inline uint64_t EncodeExecState(
    ExecPhase phase, uint32_t build_owner, uint32_t execute_owner, ExecEngineClass engine_class, uint32_t payload_lines,
    uint32_t task_id
) {
    return (static_cast<uint64_t>(phase) << kExecStatePhaseShift) |
           (static_cast<uint64_t>(build_owner) << kExecStateBuildOwnerShift) |
           (static_cast<uint64_t>(execute_owner) << kExecStateExecuteOwnerShift) |
           (static_cast<uint64_t>(engine_class) << kExecStateEngineShift) |
           (static_cast<uint64_t>(payload_lines) << kExecStatePayloadLinesShift) |
           (static_cast<uint64_t>(task_id) << kExecStateTaskIdShift);
}

PTO_DEVICE_FUNC inline DecodedExecState DecodeExecState(int64_t raw_state) {
    const uint64_t raw = static_cast<uint64_t>(raw_state);
    DecodedExecState decoded{
        static_cast<ExecPhase>((raw >> kExecStatePhaseShift) & kExecStatePhaseMask),
        static_cast<uint32_t>((raw >> kExecStateBuildOwnerShift) & kExecStateBuildOwnerMask),
        static_cast<uint32_t>((raw >> kExecStateExecuteOwnerShift) & kExecStateExecuteOwnerMask),
        static_cast<ExecEngineClass>((raw >> kExecStateEngineShift) & kExecStateEngineMask),
        static_cast<uint32_t>((raw >> kExecStatePayloadLinesShift) & kExecStatePayloadLinesMask),
        static_cast<uint32_t>((raw >> kExecStateTaskIdShift) & kExecStateTaskIdMask),
        false,
    };
    if ((raw & ~kExecStateKnownMask) != 0) return decoded;
    switch (decoded.phase) {
    case ExecPhase::Empty:
        decoded.valid = decoded.build_owner == 0 && decoded.execute_owner == 0 &&
                        decoded.engine_class == ExecEngineClass::None && decoded.payload_lines == 0 &&
                        decoded.task_id == 0;
        break;
    case ExecPhase::Building:
        decoded.valid = ExecOwnerValid(decoded.build_owner) && decoded.execute_owner == kExecUnboundOwner &&
                        decoded.engine_class == ExecEngineClass::None && decoded.payload_lines == 0;
        break;
    case ExecPhase::Built:
        decoded.valid = ExecOwnerValid(decoded.build_owner) && decoded.execute_owner == kExecUnboundOwner &&
                        ExecEngineValid(decoded.engine_class) && decoded.payload_lines >= 1 &&
                        decoded.payload_lines <= kExecMaxPayloadLines;
        break;
    case ExecPhase::Claimed:
    case ExecPhase::Done:
        decoded.valid = ExecOwnerValid(decoded.build_owner) && ExecOwnerValid(decoded.execute_owner) &&
                        ExecEngineValid(decoded.engine_class) && decoded.payload_lines >= 1 &&
                        decoded.payload_lines <= kExecMaxPayloadLines;
        break;
    default:
        break;
    }
    return decoded;
}

PTO_DEVICE_FUNC inline bool ExecFatalReasonValid(ExecFatalReason reason) {
    return reason >= ExecFatalReason::InvalidBuildInput && reason <= ExecFatalReason::CompletionStateConflict;
}

PTO_DEVICE_FUNC inline uint64_t EncodeExecFatal(ExecFatalReason reason, uint32_t owner, uint32_t task_id) {
    return static_cast<uint64_t>(reason) | (static_cast<uint64_t>(owner) << kExecFatalOwnerShift) |
           (static_cast<uint64_t>(task_id) << kExecFatalTaskIdShift);
}

PTO_DEVICE_FUNC inline uint32_t ExecTensorMaskForCount(uint32_t tensor_count) {
    return tensor_count >= 32U ? UINT32_MAX : ((uint32_t{1} << tensor_count) - 1U);
}

PTO_DEVICE_FUNC inline uint32_t ExecPopcount(uint32_t value) {
    uint32_t count = 0;
    while (value != 0) {
        count += value & 1U;
        value >>= 1U;
    }
    return count;
}

PTO_DEVICE_FUNC inline uint32_t ExecTensorPayloadWordOffset(uint32_t tensor, uint32_t reference_mask) {
    const uint32_t preceding = tensor == 0 ? 0 : reference_mask & ExecTensorMaskForCount(tensor);
    return kExecHeaderWords + tensor * kExecTensorDescWords - ExecPopcount(preceding) * (kExecTensorDescWords - 1U);
}

PTO_DEVICE_FUNC inline bool ComputeExecPayloadLayout(
    uint32_t tensor_count, uint32_t scalar_count, uint32_t fanin_count, uint32_t reference_mask,
    ExecPayloadLayout &layout
) {
    if (tensor_count > kExecMaxTensors || scalar_count > kExecMaxScalars || fanin_count > kExecMaxFanin ||
        (reference_mask & ~ExecTensorMaskForCount(tensor_count)) != 0) {
        return false;
    }
    const uint32_t references = ExecPopcount(reference_mask);
    const uint32_t inline_tensors = tensor_count - references;
    layout.tensor_word_offset = kExecHeaderWords;
    layout.scalar_word_offset = layout.tensor_word_offset + inline_tensors * kExecTensorDescWords + references;
    layout.fanin_word_offset = layout.scalar_word_offset + scalar_count;
    layout.written_words = layout.fanin_word_offset + (fanin_count + 1U) / 2U;
    layout.payload_bytes = kExecCacheLineBytes + inline_tensors * kExecTensorDescBytes + references * sizeof(uint64_t) +
                           scalar_count * sizeof(uint64_t) + fanin_count * sizeof(int32_t);
    layout.payload_lines = (layout.payload_bytes + kExecCacheLineBytes - 1U) / kExecCacheLineBytes;
    layout.tensor_reference_mask = reference_mask;
    return layout.payload_lines >= 1 && layout.payload_lines <= kExecMaxPayloadLines &&
           layout.written_words <= kExecMaxPayloadWords;
}

PTO_DEVICE_FUNC inline bool ValidateExecPayloadSpec(const ExecPayloadSpec &spec, ExecPayloadLayout &layout) {
    if (!ExecEngineValid(spec.engine_class) || (spec.flags & ~1U) != 0) {
        return false;
    }
    const bool immediate = spec.engine_class == ExecEngineClass::Immediate;
    if ((immediate && (spec.function_id != kExecInvalidFunctionId || spec.function_address != 0)) ||
        (!immediate && spec.function_id == kExecInvalidFunctionId && spec.function_address == 0)) {
        return false;
    }
    const bool multicore = (spec.flags & 1U) != 0;
    if (immediate && multicore) return false;
    if ((!multicore && (spec.multicore_group_id != 0 || spec.multicore_rank != 0 || spec.multicore_size != 1)) ||
        (multicore && (spec.multicore_size < 2 || spec.multicore_rank >= spec.multicore_size))) {
        return false;
    }
    return ComputeExecPayloadLayout(
        spec.tensor_count, spec.scalar_count, spec.fanin_count, spec.tensor_reference_mask, layout
    );
}

PTO_DEVICE_FUNC inline uint64_t PackExecHeaderWord4(const ExecPayloadSpec &spec) {
    return static_cast<uint64_t>(spec.tensor_count) | (static_cast<uint64_t>(spec.scalar_count) << 16U) |
           (static_cast<uint64_t>(spec.fanin_count) << 32U) | (static_cast<uint64_t>(spec.engine_class) << 48U) |
           (static_cast<uint64_t>(spec.flags) << 56U);
}

template <typename Ops, typename Source>
PTO_DEVICE_FUNC bool PackExecPayload(
    __gm__ SharedExecCell &cell, const ExecPayloadSpec &spec, const Source &source, ExecPayloadLayout &layout
) {
    if (!ValidateExecPayloadSpec(spec, layout)) return false;
    __gm__ volatile uint64_t *destination = cell.payload.words;
    Ops::StorePayloadWord(&destination[0], spec.task_id);
    Ops::StorePayloadWord(&destination[1], spec.function_address);
    Ops::StorePayloadWord(&destination[2], spec.completion_vend);
    Ops::StorePayloadWord(
        &destination[3], static_cast<uint64_t>(spec.function_id) | (static_cast<uint64_t>(layout.payload_bytes) << 32U)
    );
    Ops::StorePayloadWord(&destination[4], PackExecHeaderWord4(spec));
    Ops::StorePayloadWord(
        &destination[5], static_cast<uint64_t>(spec.multicore_group_id) |
                             (static_cast<uint64_t>(spec.multicore_rank) << 32U) |
                             (static_cast<uint64_t>(spec.multicore_size) << 48U)
    );
    Ops::StorePayloadWord(&destination[6], spec.tensor_reference_mask);
    Ops::StorePayloadWord(&destination[7], 0);

    uint32_t word = layout.tensor_word_offset;
    for (uint32_t tensor = 0; tensor < spec.tensor_count; ++tensor) {
        if ((spec.tensor_reference_mask & (uint32_t{1} << tensor)) != 0) {
            const uint64_t reference = source.TensorReference(tensor);
            if (reference == 0 || reference % alignof(uint64_t) != 0) return false;
            Ops::StorePayloadWord(&destination[word++], reference);
            continue;
        }
        for (uint32_t desc_word = 0; desc_word < kExecTensorDescWords; ++desc_word) {
            Ops::StorePayloadWord(&destination[word++], source.TensorWord(tensor, desc_word));
        }
    }
    for (uint32_t scalar = 0; scalar < spec.scalar_count; ++scalar) {
        Ops::StorePayloadWord(&destination[word++], source.Scalar(scalar));
    }
    for (uint32_t edge = 0; edge < spec.fanin_count; edge += 2U) {
        const int32_t low = source.Fanin(edge);
        if (low < 0 || static_cast<uint32_t>(low) >= spec.task_id) return false;
        uint64_t packed = static_cast<uint32_t>(low);
        if (edge + 1U < spec.fanin_count) {
            const int32_t high = source.Fanin(edge + 1U);
            if (high < 0 || static_cast<uint32_t>(high) >= spec.task_id) return false;
            packed |= static_cast<uint64_t>(static_cast<uint32_t>(high)) << 32U;
        }
        Ops::StorePayloadWord(&destination[word++], packed);
    }
    return word == layout.written_words;
}

PTO_DEVICE_FUNC inline ExecPayloadHeader DecodeExecPayloadHeader(__gm__ const ExecPayloadStorage &payload) {
    const uint64_t word3 = payload.words[3];
    const uint64_t word4 = payload.words[4];
    const uint64_t word5 = payload.words[5];
    return ExecPayloadHeader{
        static_cast<uint32_t>(payload.words[0]),
        payload.words[1],
        payload.words[2],
        static_cast<uint32_t>(word3),
        static_cast<uint32_t>(word3 >> 32U),
        static_cast<uint16_t>(word4),
        static_cast<uint16_t>(word4 >> 16U),
        static_cast<uint16_t>(word4 >> 32U),
        static_cast<ExecEngineClass>(static_cast<uint8_t>(word4 >> 48U)),
        static_cast<uint8_t>(word4 >> 56U),
        static_cast<uint32_t>(word5),
        static_cast<uint16_t>(word5 >> 32U),
        static_cast<uint16_t>(word5 >> 48U),
        static_cast<uint32_t>(payload.words[6]),
    };
}

PTO_DEVICE_FUNC inline bool ValidateExecPayload(
    __gm__ const ExecPayloadStorage &payload, uint32_t task_id, ExecEngineClass engine_class, uint32_t published_lines,
    ExecPayloadHeader &header
) {
    header = DecodeExecPayloadHeader(payload);
    ExecPayloadLayout layout{};
    if ((payload.words[0] >> 32U) != 0 || (payload.words[6] >> 32U) != 0 || payload.words[7] != 0 ||
        header.task_id != task_id || header.engine_class != engine_class ||
        ((header.engine_class == ExecEngineClass::Immediate) !=
         (header.function_id == kExecInvalidFunctionId && header.function_address == 0)) ||
        !ComputeExecPayloadLayout(
            header.tensor_count, header.scalar_count, header.fanin_count, header.tensor_reference_mask, layout
        ) ||
        header.payload_bytes != layout.payload_bytes || published_lines != layout.payload_lines) {
        return false;
    }
    for (uint32_t tensor = 0; tensor < header.tensor_count; ++tensor) {
        if ((header.tensor_reference_mask & (uint32_t{1} << tensor)) == 0) continue;
        const uint64_t reference = payload.words[ExecTensorPayloadWordOffset(tensor, header.tensor_reference_mask)];
        if (reference == 0 || reference % alignof(uint64_t) != 0) return false;
    }
    for (uint32_t edge = 0; edge < header.fanin_count; ++edge) {
        const uint64_t packed = payload.words[layout.fanin_word_offset + edge / 2U];
        const int32_t producer =
            static_cast<int32_t>(edge % 2U == 0 ? static_cast<uint32_t>(packed) : static_cast<uint32_t>(packed >> 32U));
        if (producer < 0 || static_cast<uint32_t>(producer) >= task_id) return false;
    }
    const bool multicore = (header.flags & 1U) != 0;
    return (header.flags & ~1U) == 0 &&
           ((!multicore && header.multicore_group_id == 0 && header.multicore_rank == 0 &&
             header.multicore_size == 1) ||
            (multicore && header.multicore_size >= 2 && header.multicore_rank < header.multicore_size));
}

PTO_DEVICE_FUNC inline void ResetExecToken(ExecToken &token) {
    token.phase = ExecTokenPhase::Idle;
    token.task_id = UINT32_MAX;
    token.build_owner = UINT32_MAX;
    token.execute_owner = UINT32_MAX;
    token.engine_class = ExecEngineClass::None;
    token.payload_lines = 0;
    token.payload_address = 0;
    token.header = ExecPayloadHeader{};
}

template <typename Ops>
PTO_DEVICE_FUNC bool
PublishExecFatal(__gm__ SharedExecControl &fatal, ExecFatalReason reason, uint32_t task_id, uint32_t owner) {
    if (!ExecFatalReasonValid(reason) || !ExecOwnerValid(owner)) return false;
    const int64_t desired = static_cast<int64_t>(EncodeExecFatal(reason, owner, task_id));
    return Ops::CompareExchange(&fatal.state, 0, desired) == 0;
}

template <typename Ops>
PTO_DEVICE_FUNC ExecBuildReserveResult
ReserveExecBuild(__gm__ SharedExecCell &cell, uint32_t task_id, uint32_t build_owner, __gm__ SharedExecControl &fatal) {
    if (Ops::Load(&fatal.state) != 0) return ExecBuildReserveResult::FatalObserved;
    if (!ExecOwnerValid(build_owner)) return ExecBuildReserveResult::InvalidInput;
    const int64_t empty = 0;
    const int64_t building = static_cast<int64_t>(
        EncodeExecState(ExecPhase::Building, build_owner, kExecUnboundOwner, ExecEngineClass::None, 0, task_id)
    );
    if (Ops::CompareExchange(&cell.control.state, empty, building) != empty) {
        return ExecBuildReserveResult::CellUnavailable;
    }
    return ExecBuildReserveResult::Reserved;
}

template <typename Ops, typename Source>
PTO_DEVICE_FUNC ExecBuildResult PublishReservedExecPayload(
    __gm__ SharedExecCell &cell, uint32_t build_owner, const ExecPayloadSpec &spec, const Source &source,
    __gm__ SharedExecControl &fatal
) {
    if (Ops::Load(&fatal.state) != 0) return ExecBuildResult::FatalObserved;
    ExecPayloadLayout expected{};
    if (!ExecOwnerValid(build_owner) || !ValidateExecPayloadSpec(spec, expected)) {
        if (ExecOwnerValid(build_owner)) {
            (void)PublishExecFatal<Ops>(fatal, ExecFatalReason::InvalidBuildInput, spec.task_id, build_owner);
        }
        return ExecBuildResult::InvalidInput;
    }
    const int64_t building = static_cast<int64_t>(
        EncodeExecState(ExecPhase::Building, build_owner, kExecUnboundOwner, ExecEngineClass::None, 0, spec.task_id)
    );
    if (Ops::Load(&cell.control.state) != building) {
        (void)PublishExecFatal<Ops>(fatal, ExecFatalReason::ControlPublishConflict, spec.task_id, build_owner);
        return ExecBuildResult::PublishConflict;
    }
    ExecPayloadLayout packed{};
    if (!PackExecPayload<Ops>(cell, spec, source, packed) || packed.payload_bytes != expected.payload_bytes) {
        (void)PublishExecFatal<Ops>(fatal, ExecFatalReason::BuildPackFailed, spec.task_id, build_owner);
        return ExecBuildResult::InvalidInput;
    }
    Ops::FlushRegion(&cell.payload, static_cast<uint64_t>(packed.payload_lines) * kExecCacheLineBytes);
    const int64_t built = static_cast<int64_t>(EncodeExecState(
        ExecPhase::Built, build_owner, kExecUnboundOwner, spec.engine_class, packed.payload_lines, spec.task_id
    ));
    if (Ops::CompareExchange(&cell.control.state, building, built) != building) {
        (void)PublishExecFatal<Ops>(fatal, ExecFatalReason::ControlPublishConflict, spec.task_id, build_owner);
        return ExecBuildResult::PublishConflict;
    }
    return ExecBuildResult::Published;
}

template <typename Ops, typename Source>
PTO_DEVICE_FUNC ExecBuildResult BuildAndPublishExecPayload(
    __gm__ SharedExecCell &cell, uint32_t build_owner, const ExecPayloadSpec &spec, const Source &source,
    __gm__ SharedExecControl &fatal
) {
    switch (ReserveExecBuild<Ops>(cell, spec.task_id, build_owner, fatal)) {
    case ExecBuildReserveResult::Reserved:
        return PublishReservedExecPayload<Ops>(cell, build_owner, spec, source, fatal);
    case ExecBuildReserveResult::InvalidInput:
        return ExecBuildResult::InvalidInput;
    case ExecBuildReserveResult::CellUnavailable:
        return ExecBuildResult::CellUnavailable;
    case ExecBuildReserveResult::FatalObserved:
        return ExecBuildResult::FatalObserved;
    }
    return ExecBuildResult::InvalidInput;
}

PTO_DEVICE_FUNC inline bool ExecEngineCompatible(ExecEngineClass task, ExecEngineClass executor) {
    return task == executor;
}

template <typename Ops>
PTO_DEVICE_FUNC ExecDoneResult PublishImmediateExecDone(
    __gm__ SharedExecCell &cell, uint32_t task_id, uint32_t build_owner, __gm__ SharedExecControl &fatal
) {
    if (Ops::Load(&fatal.state) != 0) return ExecDoneResult::FatalObserved;
    const int64_t observed_raw = Ops::Load(&cell.control.state);
    const DecodedExecState observed = DecodeExecState(observed_raw);
    if (!observed.valid || observed.phase != ExecPhase::Built || observed.task_id != task_id ||
        observed.build_owner != build_owner || observed.engine_class != ExecEngineClass::Immediate) {
        (void)PublishExecFatal<Ops>(fatal, ExecFatalReason::CompletionStateConflict, task_id, build_owner);
        return ExecDoneResult::StateConflict;
    }
    const int64_t done = static_cast<int64_t>(EncodeExecState(
        ExecPhase::Done, build_owner, build_owner, ExecEngineClass::Immediate, observed.payload_lines, task_id
    ));
    if (Ops::CompareExchange(&cell.control.state, observed_raw, done) != observed_raw) {
        (void)PublishExecFatal<Ops>(fatal, ExecFatalReason::CompletionStateConflict, task_id, build_owner);
        return ExecDoneResult::StateConflict;
    }
    return ExecDoneResult::Done;
}

template <typename Ops>
PTO_DEVICE_FUNC ExecAcquireResult AcquireExecPayload(
    __gm__ SharedExecCell &cell, uint32_t task_id, uint32_t execute_owner, ExecEngineClass executor_engine,
    ExecToken &token, __gm__ SharedExecControl &fatal
) {
    if (token.phase != ExecTokenPhase::Idle) return ExecAcquireResult::TokenBusy;
    if (Ops::Load(&fatal.state) != 0) return ExecAcquireResult::FatalObserved;
    if (!ExecOwnerValid(execute_owner) || !ExecExecutorEngineValid(executor_engine)) {
        return ExecAcquireResult::InvalidControl;
    }
    const int64_t observed_raw = Ops::Load(&cell.control.state);
    const DecodedExecState observed = DecodeExecState(observed_raw);
    if (!observed.valid) {
        (void)PublishExecFatal<Ops>(fatal, ExecFatalReason::InvalidBuiltControl, task_id, execute_owner);
        return ExecAcquireResult::InvalidControl;
    }
    if (observed.phase != ExecPhase::Built) return ExecAcquireResult::NotBuilt;
    if (observed.task_id != task_id) {
        (void)PublishExecFatal<Ops>(fatal, ExecFatalReason::InvalidBuiltControl, task_id, execute_owner);
        return ExecAcquireResult::InvalidControl;
    }
    if (!ExecEngineCompatible(observed.engine_class, executor_engine)) return ExecAcquireResult::Incompatible;
    const int64_t claimed = static_cast<int64_t>(EncodeExecState(
        ExecPhase::Claimed, observed.build_owner, execute_owner, observed.engine_class, observed.payload_lines,
        observed.task_id
    ));
    if (Ops::CompareExchange(&cell.control.state, observed_raw, claimed) != observed_raw) {
        return ExecAcquireResult::Lost;
    }
    token.phase = ExecTokenPhase::Acquired;
    token.task_id = task_id;
    token.build_owner = observed.build_owner;
    token.execute_owner = execute_owner;
    token.engine_class = observed.engine_class;
    token.payload_lines = static_cast<uint8_t>(observed.payload_lines);
    token.payload_address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&cell.payload));
    Ops::InvalidateRegion(&cell.payload, static_cast<uint64_t>(observed.payload_lines) * kExecCacheLineBytes);
    if (!ValidateExecPayload(cell.payload, task_id, observed.engine_class, observed.payload_lines, token.header)) {
        token.phase = ExecTokenPhase::Faulted;
        (void)PublishExecFatal<Ops>(fatal, ExecFatalReason::AcquiredPayloadInvalid, task_id, execute_owner);
        return ExecAcquireResult::InvalidPayload;
    }
    return ExecAcquireResult::Acquired;
}

template <typename Ops>
PTO_DEVICE_FUNC ExecDoneResult
PublishExecDone(__gm__ SharedExecCell &cell, ExecToken &token, __gm__ SharedExecControl &fatal) {
    if (token.phase != ExecTokenPhase::Acquired) return ExecDoneResult::TokenNotAcquired;
    if (Ops::Load(&fatal.state) != 0) return ExecDoneResult::FatalObserved;
    const int64_t claimed = static_cast<int64_t>(EncodeExecState(
        ExecPhase::Claimed, token.build_owner, token.execute_owner, token.engine_class, token.payload_lines,
        token.task_id
    ));
    const int64_t done = static_cast<int64_t>(EncodeExecState(
        ExecPhase::Done, token.build_owner, token.execute_owner, token.engine_class, token.payload_lines, token.task_id
    ));
    if (Ops::CompareExchange(&cell.control.state, claimed, done) != claimed) {
        (void)PublishExecFatal<Ops>(
            fatal, ExecFatalReason::CompletionStateConflict, token.task_id, token.execute_owner
        );
        return ExecDoneResult::StateConflict;
    }
    ResetExecToken(token);
    return ExecDoneResult::Done;
}

}  // namespace fdwic::cross_core
