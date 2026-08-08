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

namespace fdwic::cross_core {

constexpr uint32_t kOutputMaxDescriptors = kExecMaxTensors;

enum class OutputPhase : uint8_t {
    Empty = 0,
    Published = 1,
};

enum class OutputPublishResult : uint8_t {
    Published = 0,
    InvalidInput = 1,
    PublishConflict = 2,
    FatalObserved = 3,
};

enum class OutputAcquireResult : uint8_t {
    Acquired = 0,
    NotPublished = 1,
    InvalidInput = 2,
    InvalidControl = 3,
    FatalObserved = 4,
};

enum class HeapReserveResult : uint8_t {
    Reserved = 0,
    Empty = 1,
    InvalidInput = 2,
    Exhausted = 3,
    FatalObserved = 4,
};

struct HeapReservation {
    uint64_t begin;
    uint64_t end;
};

constexpr uint64_t kOutputStatePhaseShift = 0;
constexpr uint64_t kOutputStatePhaseMask = 0x3ULL;
constexpr uint64_t kOutputStateCountShift = 2;
constexpr uint64_t kOutputStateCountMask = 0x3FULL;
constexpr uint64_t kOutputStateOwnerShift = 8;
constexpr uint64_t kOutputStateOwnerMask = 0xFFULL;
constexpr uint64_t kOutputStateTaskIdShift = 16;
constexpr uint64_t kOutputStateTaskIdMask = 0xFFFFFFFFULL;
constexpr uint64_t kOutputStateKnownMask =
    (kOutputStatePhaseMask << kOutputStatePhaseShift) | (kOutputStateCountMask << kOutputStateCountShift) |
    (kOutputStateOwnerMask << kOutputStateOwnerShift) | (kOutputStateTaskIdMask << kOutputStateTaskIdShift);

struct DecodedOutputState {
    OutputPhase phase;
    uint32_t output_count;
    uint32_t build_owner;
    uint32_t task_id;
    bool valid;
};

template <typename Descriptor>
struct alignas(kExecCacheLineBytes) CrossCoreOutputCell {
    SharedExecControl control;
    alignas(kExecCacheLineBytes) Descriptor descriptors[kOutputMaxDescriptors];
};

PTO_DEVICE_FUNC inline uint64_t
EncodeOutputState(OutputPhase phase, uint32_t output_count, uint32_t build_owner, uint32_t task_id) {
    return (static_cast<uint64_t>(phase) << kOutputStatePhaseShift) |
           (static_cast<uint64_t>(output_count) << kOutputStateCountShift) |
           (static_cast<uint64_t>(build_owner) << kOutputStateOwnerShift) |
           (static_cast<uint64_t>(task_id) << kOutputStateTaskIdShift);
}

PTO_DEVICE_FUNC inline DecodedOutputState DecodeOutputState(int64_t raw_state) {
    const uint64_t raw = static_cast<uint64_t>(raw_state);
    DecodedOutputState decoded{
        static_cast<OutputPhase>((raw >> kOutputStatePhaseShift) & kOutputStatePhaseMask),
        static_cast<uint32_t>((raw >> kOutputStateCountShift) & kOutputStateCountMask),
        static_cast<uint32_t>((raw >> kOutputStateOwnerShift) & kOutputStateOwnerMask),
        static_cast<uint32_t>((raw >> kOutputStateTaskIdShift) & kOutputStateTaskIdMask),
        false,
    };
    if ((raw & ~kOutputStateKnownMask) != 0) return decoded;
    if (decoded.phase == OutputPhase::Empty) {
        decoded.valid = raw == 0;
    } else if (decoded.phase == OutputPhase::Published) {
        decoded.valid = decoded.output_count <= kOutputMaxDescriptors && ExecOwnerValid(decoded.build_owner);
    }
    return decoded;
}

template <typename Ops>
PTO_DEVICE_FUNC HeapReserveResult ReserveOutputHeap(
    __gm__ SharedExecControl &cursor, uint32_t task_id, uint32_t build_owner, uint64_t bytes, uint64_t heap_bytes,
    HeapReservation &reservation, __gm__ SharedExecControl &fatal
) {
    reservation = HeapReservation{};
    if (Ops::Load(&fatal.state) != 0) return HeapReserveResult::FatalObserved;
    if (!ExecOwnerValid(build_owner) || bytes > static_cast<uint64_t>(INT64_MAX) ||
        heap_bytes > static_cast<uint64_t>(INT64_MAX)) {
        if (ExecOwnerValid(build_owner)) {
            (void)PublishExecFatal<Ops>(fatal, ExecFatalReason::InvalidBuildInput, task_id, build_owner);
        }
        return HeapReserveResult::InvalidInput;
    }
    if (bytes == 0) {
        const int64_t current = Ops::Load(&cursor.state);
        if (current < 0 || static_cast<uint64_t>(current) > heap_bytes) return HeapReserveResult::InvalidInput;
        reservation.begin = static_cast<uint64_t>(current);
        reservation.end = static_cast<uint64_t>(current);
        return HeapReserveResult::Empty;
    }
    const int64_t observed = Ops::FetchAdd(&cursor.state, static_cast<int64_t>(bytes));
    if (observed < 0 || static_cast<uint64_t>(observed) > UINT64_MAX - bytes) {
        (void)PublishExecFatal<Ops>(fatal, ExecFatalReason::InvalidBuildInput, task_id, build_owner);
        return HeapReserveResult::InvalidInput;
    }
    reservation.begin = static_cast<uint64_t>(observed);
    reservation.end = reservation.begin + bytes;
    if (reservation.end > heap_bytes) {
        (void)PublishExecFatal<Ops>(fatal, ExecFatalReason::InvalidBuildInput, task_id, build_owner);
        return HeapReserveResult::Exhausted;
    }
    return HeapReserveResult::Reserved;
}

template <typename Ops, typename Descriptor>
PTO_DEVICE_FUNC OutputPublishResult PublishTaskOutputs(
    __gm__ CrossCoreOutputCell<Descriptor> &cell, uint32_t task_id, uint32_t build_owner, uint32_t output_count,
    __gm__ SharedExecControl &fatal
) {
    static_assert(sizeof(Descriptor) == kExecTensorDescBytes, "cross-core output descriptor size changed");
    if (Ops::Load(&fatal.state) != 0) return OutputPublishResult::FatalObserved;
    if (!ExecOwnerValid(build_owner) || output_count > kOutputMaxDescriptors) {
        if (ExecOwnerValid(build_owner)) {
            (void)PublishExecFatal<Ops>(fatal, ExecFatalReason::InvalidBuildInput, task_id, build_owner);
        }
        return OutputPublishResult::InvalidInput;
    }
    if (output_count != 0) {
        Ops::FlushRegion(cell.descriptors, static_cast<uint64_t>(output_count) * sizeof(Descriptor));
    }
    const int64_t published =
        static_cast<int64_t>(EncodeOutputState(OutputPhase::Published, output_count, build_owner, task_id));
    if (Ops::CompareExchange(&cell.control.state, 0, published) != 0) {
        (void)PublishExecFatal<Ops>(fatal, ExecFatalReason::ControlPublishConflict, task_id, build_owner);
        return OutputPublishResult::PublishConflict;
    }
    return OutputPublishResult::Published;
}

template <typename Ops, typename Descriptor>
PTO_DEVICE_FUNC OutputAcquireResult AcquireTaskOutputs(
    __gm__ CrossCoreOutputCell<Descriptor> &cell, uint32_t task_id, uint32_t expected_count,
    __gm__ SharedExecControl &fatal
) {
    static_assert(sizeof(Descriptor) == kExecTensorDescBytes, "cross-core output descriptor size changed");
    if (Ops::Load(&fatal.state) != 0) return OutputAcquireResult::FatalObserved;
    if (expected_count > kOutputMaxDescriptors) return OutputAcquireResult::InvalidInput;
    const int64_t observed_raw = Ops::Load(&cell.control.state);
    if (observed_raw == 0) return OutputAcquireResult::NotPublished;
    const DecodedOutputState observed = DecodeOutputState(observed_raw);
    if (!observed.valid || observed.phase != OutputPhase::Published || observed.task_id != task_id ||
        observed.output_count != expected_count) {
        return OutputAcquireResult::InvalidControl;
    }
    if (expected_count != 0) {
        Ops::InvalidateRegion(cell.descriptors, static_cast<uint64_t>(expected_count) * sizeof(Descriptor));
    }
    if (Ops::Load(&cell.control.state) != observed_raw) return OutputAcquireResult::InvalidControl;
    return OutputAcquireResult::Acquired;
}

}  // namespace fdwic::cross_core
