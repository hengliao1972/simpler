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

#ifndef PA_SCHEDULER_AICPU_PLAN_OPERATION_TRACE_H
#define PA_SCHEDULER_AICPU_PLAN_OPERATION_TRACE_H

#include <cstdint>
#include <ctime>
#include <limits>

namespace pa_scheduler::aicpu_plan_trace {

constexpr uint32_t kNoTaskId = UINT32_MAX;
constexpr uint32_t kNoTargetIndex = UINT32_MAX;
constexpr uint32_t kFixedRecordCapacity = 64U;
constexpr uint32_t kRecordsPerPlanCell = 32U;

enum class Operation : uint8_t {
    AtomicLoadAcquire = 0,
    CacheCleanCvac = 1,
    CacheDiscardCivac = 2,
    BarrierDsbSy = 3,
    BarrierIsb = 4,
    GmStore = 5,
    ScalarWork = 6,
    AtomicStoreRelease = 7,
    Count = 8,
};

enum class Scope : uint16_t {
    OwnerSetup = 0,
    BackendBind = 1,
    TaskStage = 2,
    TaskPublish = 3,
    FrontierAdvance = 4,
    ReadyPublish = 5,
    BackendClose = 6,
    FatalPublish = 7,
    Count = 8,
};

enum class Target : uint8_t {
    None = 0,
    RequestTensors = 1,
    RequestScalars = 2,
    ContextLens = 3,
    RuntimePlanControl = 4,
    Fatal = 5,
    ClosedTaskCount = 6,
    PlannedFrontier = 7,
    BuildNext = 8,
    BuildWorkersDone = 9,
    BuildRelease = 10,
    CellControl = 11,
    CellPayload = 12,
    PayloadValidation = 13,
    Count = 14,
};

struct alignas(64) Record {
    uint64_t begin_ns;
    uint64_t end_ns;
    int64_t first_value;
    int64_t last_value;
    uint32_t task_id;
    uint32_t calls;
    uint32_t lines;
    uint32_t first_target_index;
    uint32_t last_target_index;
    uint16_t scope;
    uint8_t operation;
    uint8_t target;
    uint8_t reserved[8];
};

struct State {
    Record *records;
    uint32_t capacity;
    uint32_t count;
    uint32_t dropped;
    uint32_t reserved;
};

constexpr uint32_t CapacityForPlanCells(uint32_t plan_capacity) {
    return plan_capacity <= (std::numeric_limits<uint32_t>::max() - kFixedRecordCapacity) / kRecordsPerPlanCell ?
               kFixedRecordCapacity + plan_capacity * kRecordsPerPlanCell :
               0U;
}

inline uint64_t TimestampNanoseconds() {
    constexpr uint64_t kNanosecondsPerSecond = UINT64_C(1000000000);
    timespec timestamp{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) return 0U;
    return static_cast<uint64_t>(timestamp.tv_sec) * kNanosecondsPerSecond + static_cast<uint64_t>(timestamp.tv_nsec);
}

inline bool IsCacheOperation(Operation operation) {
    return operation == Operation::CacheCleanCvac || operation == Operation::CacheDiscardCivac;
}

inline bool Append(
    State *state, uint64_t begin_ns, uint64_t end_ns, uint32_t task_id, Scope scope, Operation operation, Target target,
    uint32_t calls, uint32_t lines, uint32_t first_target_index, uint32_t last_target_index, int64_t first_value,
    int64_t last_value, bool merge_consecutive
) {
    if (state == nullptr) return false;
    const bool cache_operation = IsCacheOperation(operation);
    if (state->records == nullptr || state->capacity == 0U || begin_ns == 0U || end_ns <= begin_ns || calls == 0U ||
        static_cast<uint16_t>(scope) >= static_cast<uint16_t>(Scope::Count) ||
        static_cast<uint8_t>(operation) >= static_cast<uint8_t>(Operation::Count) ||
        static_cast<uint8_t>(target) >= static_cast<uint8_t>(Target::Count) || (cache_operation && lines == 0U) ||
        (!cache_operation && lines != 0U)) {
        ++state->dropped;
        return false;
    }
    if (merge_consecutive && state->count != 0U) {
        Record &previous = state->records[state->count - 1U];
        if (previous.task_id == task_id && previous.scope == static_cast<uint16_t>(scope) &&
            previous.operation == static_cast<uint8_t>(operation) && previous.target == static_cast<uint8_t>(target) &&
            previous.end_ns <= begin_ns && previous.calls <= UINT32_MAX - calls &&
            previous.lines <= UINT32_MAX - lines) {
            previous.end_ns = end_ns;
            previous.last_value = last_value;
            previous.calls += calls;
            previous.lines += lines;
            previous.last_target_index = last_target_index;
            return true;
        }
    }
    if (state->count >= state->capacity) {
        ++state->dropped;
        return false;
    }
    Record record{};
    record.begin_ns = begin_ns;
    record.end_ns = end_ns;
    record.first_value = first_value;
    record.last_value = last_value;
    record.task_id = task_id;
    record.calls = calls;
    record.lines = lines;
    record.first_target_index = first_target_index;
    record.last_target_index = last_target_index;
    record.scope = static_cast<uint16_t>(scope);
    record.operation = static_cast<uint8_t>(operation);
    record.target = static_cast<uint8_t>(target);
    state->records[state->count++] = record;
    return true;
}

static_assert(sizeof(Record) == 64U, "AICPU operation trace ABI changed");
static_assert(alignof(Record) == 64U, "AICPU operation trace alignment changed");
static_assert(sizeof(State) == 24U, "AICPU operation trace state ABI changed");

}  // namespace pa_scheduler::aicpu_plan_trace

#endif  // PA_SCHEDULER_AICPU_PLAN_OPERATION_TRACE_H
