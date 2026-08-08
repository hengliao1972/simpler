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

#include "dist_engine/common/cross_core_tensor_map_protocol.h"

namespace fdwic::cross_core {

// The production runtime has no random-access host task plan. Each Build
// owner therefore publishes its task's minimal writer-region schema. Later
// tasks scan immutable records by logical task id, not publication order.
constexpr uint32_t kDagMaxWriterRegions = kExecMaxTensors;
constexpr uint64_t kDagWriterCountMask = 0x3FULL;
constexpr uint64_t kDagTaskPlusOneShift = 6;
constexpr uint64_t kDagTaskPlusOneMask = 0xFFFFFFFFULL;
constexpr uint64_t kDagControlKnownMask = kDagWriterCountMask | (kDagTaskPlusOneMask << kDagTaskPlusOneShift);

struct alignas(kExecCacheLineBytes) DagWriterPayload {
    CrossMapValue writers[kDagMaxWriterRegions];
};

struct alignas(kExecCacheLineBytes) DagTaskMetadataCell {
    // Keep control on an atomic-only cache line. Payload begins on the next
    // line and is published only after ordinary stores and clean-out finish.
    SharedExecControl control;
    DagWriterPayload payload;
};

static_assert(sizeof(DagWriterPayload) == kDagMaxWriterRegions * sizeof(CrossMapValue));
static_assert(sizeof(DagWriterPayload) % kExecCacheLineBytes == 0);
static_assert(offsetof(DagTaskMetadataCell, payload) == kExecCacheLineBytes);
static_assert(sizeof(DagTaskMetadataCell) == kExecCacheLineBytes + sizeof(DagWriterPayload));

enum class DagMetadataPublishResult : uint8_t {
    Published = 0,
    InvalidInput = 1,
    AlreadyPublished = 2,
};

enum class DagMetadataAcquireResult : uint8_t {
    Acquired = 0,
    NotPublished = 1,
    InvalidControl = 2,
};

enum class DagWriterLookupResult : uint8_t {
    Found = 0,
    None = 1,
    Pending = 2,
    Invalid = 3,
};

struct DecodedDagMetadataControl {
    uint32_t task_id;
    uint32_t writer_count;
    bool published;
    bool valid;
};

PTO_DEVICE_FUNC constexpr uint64_t EncodeDagMetadataControl(uint32_t task_id, uint32_t writer_count) {
    if (writer_count > kDagMaxWriterRegions || task_id == UINT32_MAX) return 0;
    return ((static_cast<uint64_t>(task_id) + 1U) << kDagTaskPlusOneShift) | writer_count;
}

PTO_DEVICE_FUNC constexpr DecodedDagMetadataControl DecodeDagMetadataControl(uint64_t raw) {
    if (raw == 0) return DecodedDagMetadataControl{0, 0, false, true};
    const uint64_t task_plus_one = (raw >> kDagTaskPlusOneShift) & kDagTaskPlusOneMask;
    const uint32_t writer_count = static_cast<uint32_t>(raw & kDagWriterCountMask);
    const bool valid = (raw & ~kDagControlKnownMask) == 0 && task_plus_one != 0 && writer_count <= kDagMaxWriterRegions;
    return DecodedDagMetadataControl{
        valid ? static_cast<uint32_t>(task_plus_one - 1U) : 0U,
        writer_count,
        true,
        valid,
    };
}

PTO_DEVICE_FUNC inline bool DagWriterRegionValid(const CrossMapValue &writer, uint32_t task_id) {
    return writer.buffer_address != 0 && writer.lo < writer.hi && writer.producer == static_cast<int32_t>(task_id) &&
           writer.reserved == 0;
}

template <typename Ops>
PTO_DEVICE_FUNC DagMetadataPublishResult PublishDagTaskMetadata(
    __gm__ DagTaskMetadataCell &cell, uint32_t task_id, const CrossMapValue writers[], uint32_t writer_count
) {
    const uint64_t encoded = EncodeDagMetadataControl(task_id, writer_count);
    if (encoded == 0 || (writer_count != 0 && writers == nullptr)) return DagMetadataPublishResult::InvalidInput;
    for (uint32_t index = 0; index < writer_count; ++index) {
        if (!DagWriterRegionValid(writers[index], task_id)) return DagMetadataPublishResult::InvalidInput;
    }
    if (Ops::Load(&cell.control.state) != 0) return DagMetadataPublishResult::AlreadyPublished;

    for (uint32_t index = 0; index < writer_count; ++index) {
        cell.payload.writers[index].buffer_address = writers[index].buffer_address;
        cell.payload.writers[index].lo = writers[index].lo;
        cell.payload.writers[index].hi = writers[index].hi;
        cell.payload.writers[index].producer = writers[index].producer;
        cell.payload.writers[index].reserved = 0;
    }
    if (writer_count != 0) {
        Ops::FlushRegion(&cell.payload, static_cast<uint64_t>(writer_count) * sizeof(CrossMapValue));
    }
    const int64_t previous = Ops::CompareExchange(&cell.control.state, 0, static_cast<int64_t>(encoded));
    return previous == 0 ? DagMetadataPublishResult::Published : DagMetadataPublishResult::AlreadyPublished;
}

template <typename Ops>
PTO_DEVICE_FUNC DagMetadataAcquireResult
AcquireDagTaskMetadata(__gm__ DagTaskMetadataCell &cell, uint32_t expected_task, uint32_t &writer_count) {
    writer_count = 0;
    const uint64_t first = static_cast<uint64_t>(Ops::Load(&cell.control.state));
    const DecodedDagMetadataControl decoded = DecodeDagMetadataControl(first);
    if (!decoded.valid || (decoded.published && decoded.task_id != expected_task)) {
        return DagMetadataAcquireResult::InvalidControl;
    }
    if (!decoded.published) return DagMetadataAcquireResult::NotPublished;
    if (decoded.writer_count != 0) {
        Ops::InvalidateRegion(&cell.payload, static_cast<uint64_t>(decoded.writer_count) * sizeof(CrossMapValue));
    }
    const uint64_t second = static_cast<uint64_t>(Ops::Load(&cell.control.state));
    if (second != first) return DagMetadataAcquireResult::InvalidControl;
    writer_count = decoded.writer_count;
    return DagMetadataAcquireResult::Acquired;
}

// Scan from task N-1 to the history lower bound. An unpublished candidate
// returns Pending: skipping it could let physical completion order change the
// logical DAG by selecting an older writer.
template <typename Ops>
PTO_DEVICE_FUNC DagWriterLookupResult FindLatestDagWriter(
    __gm__ DagTaskMetadataCell cells[], uint32_t capacity, const CrossMapValue &query, uint32_t consumer_task,
    uint32_t history, int32_t &producer
) {
    producer = -1;
    if (cells == nullptr || consumer_task >= capacity || query.buffer_address == 0 || query.lo >= query.hi) {
        return DagWriterLookupResult::Invalid;
    }
    const uint32_t lower = consumer_task > history ? consumer_task - history : 0U;
    for (uint32_t candidate = consumer_task; candidate > lower;) {
        --candidate;
        uint32_t writer_count = 0;
        const DagMetadataAcquireResult acquired =
            AcquireDagTaskMetadata<Ops>(cells[candidate], candidate, writer_count);
        if (acquired == DagMetadataAcquireResult::NotPublished) return DagWriterLookupResult::Pending;
        if (acquired != DagMetadataAcquireResult::Acquired) return DagWriterLookupResult::Invalid;
        for (uint32_t writer = 0; writer < writer_count; ++writer) {
            // CCEC cannot bind a GM object to a local reference. Control has
            // been acquired and payload invalidated, so snapshot each field
            // locally before applying value-only validation and overlap tests.
            CrossMapValue entry{};
            entry.buffer_address = cells[candidate].payload.writers[writer].buffer_address;
            entry.lo = cells[candidate].payload.writers[writer].lo;
            entry.hi = cells[candidate].payload.writers[writer].hi;
            entry.producer = cells[candidate].payload.writers[writer].producer;
            entry.reserved = cells[candidate].payload.writers[writer].reserved;
            if (!DagWriterRegionValid(entry, candidate)) return DagWriterLookupResult::Invalid;
            if (CrossMapRegionsOverlap(entry, query)) {
                producer = static_cast<int32_t>(candidate);
                return DagWriterLookupResult::Found;
            }
        }
    }
    return DagWriterLookupResult::None;
}

}  // namespace fdwic::cross_core
