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

#ifndef PA_SCHEDULER_AICPU_PLAN_SIMT_REAL_STATE_RUNTIME_H
#define PA_SCHEDULER_AICPU_PLAN_SIMT_REAL_STATE_RUNTIME_H

#ifndef PA_DEVICE
#error "PA_DEVICE must name the active CPU or __simt_callee__ function identity"
#endif
#ifndef PA_GM
#error "PA_GM must name the active CPU or CCEC GM address space"
#endif

// include 顺序是本桥的 ABI 合同：先以当前 TU 的函数身份实例化真实
// scalar_build 数据结构与 generic TensorMap/heap helper，再引入窄 SIMT
// builder。这里只复用类型和算法；不会创建第二份 GM sidecar。
#include "../../scalar_build/common/pa_shared_tensormap.h"
#include "../../scalar_build/common/pa_shared_heap.h"
#include "simt_plan_task_builder.h"

#include <cstddef>
#include <cstdint>

namespace pa_scheduler::aicpu_plan_simt {

// build_owner 与 execute_owner 是两个独立字段，但真实
// A5 execute/Host policy 仍要求 build_owner < kWorkers。SIMT 四个
// leader 因而映射到 0..3；诊断上的 Build/Execute 角色区分由
// 独立 leader report 和字段位置承担，不伪造 policy 不接受的
// 96..99 owner namespace。
constexpr uint32_t kSimtBuildOwnerLimit = kBuilderLeaders;

PA_DEVICE constexpr uint32_t SimtBuildOwner(uint32_t leader_id)
{
    return BuilderLeaderIdValid(leader_id)
        ? leader_id
        : exec::kExecUnboundOwner;
}

// 以下断言不是“看起来相同”的抽样检查。窄 builder 会把同一片真实 GM
// storage 分别以 wire/value 类型和 production 类型解释，因此所有会跨边界
// 的 size、align、field offset、enum 和 capacity 都必须在本 TU 闭合。
static_assert(kWorkers == 96U);
static_assert(kBuilderLeaders == 4U);
static_assert(kSimtBuildOwnerLimit == 4U);
static_assert(kSimtBuildOwnerLimit <= kWorkers);
static_assert(SimtBuildOwner(0U) == 0U);
static_assert(SimtBuildOwner(3U) == 3U);
static_assert(SimtBuildOwner(4U) == exec::kExecUnboundOwner);

static_assert(sizeof(AtomicLine) == 64U);
static_assert(alignof(AtomicLine) == 64U);
static_assert(offsetof(AtomicLine, value) == 0U);
static_assert(sizeof(AtomicFlagLine) == 64U);
static_assert(alignof(AtomicFlagLine) == 64U);
static_assert(offsetof(AtomicFlagLine, value) == 0U);

static_assert(sizeof(SimtCanonicalTensorDesc) == sizeof(TensorDesc));
static_assert(alignof(SimtCanonicalTensorDesc) == alignof(TensorDesc));
#define PA_SIMT_REAL_DESC_OFFSET(field)                                      \
    static_assert(                                                          \
        offsetof(SimtCanonicalTensorDesc, field) ==                         \
        offsetof(TensorDesc, field)                                         \
    )
PA_SIMT_REAL_DESC_OFFSET(buffer_addr);
PA_SIMT_REAL_DESC_OFFSET(buffer_size);
PA_SIMT_REAL_DESC_OFFSET(owner_task_id);
PA_SIMT_REAL_DESC_OFFSET(start_offset);
PA_SIMT_REAL_DESC_OFFSET(version);
PA_SIMT_REAL_DESC_OFFSET(ndims);
PA_SIMT_REAL_DESC_OFFSET(dtype);
PA_SIMT_REAL_DESC_OFFSET(manual_dep);
PA_SIMT_REAL_DESC_OFFSET(is_contiguous);
PA_SIMT_REAL_DESC_OFFSET(child_memory);
PA_SIMT_REAL_DESC_OFFSET(shapes);
PA_SIMT_REAL_DESC_OFFSET(extent_elem_cache);
PA_SIMT_REAL_DESC_OFFSET(strides);
PA_SIMT_REAL_DESC_OFFSET(padding);
#undef PA_SIMT_REAL_DESC_OFFSET

static_assert(sizeof(SimtWriterRegion) == sizeof(SharedRegionValue));
static_assert(alignof(SimtWriterRegion) == alignof(SharedRegionValue));
#define PA_SIMT_REAL_REGION_OFFSET(field)                                    \
    static_assert(                                                          \
        offsetof(SimtWriterRegion, field) ==                                \
        offsetof(SharedRegionValue, field)                                  \
    )
PA_SIMT_REAL_REGION_OFFSET(buffer_addr);
PA_SIMT_REAL_REGION_OFFSET(lo);
PA_SIMT_REAL_REGION_OFFSET(hi);
PA_SIMT_REAL_REGION_OFFSET(producer);
PA_SIMT_REAL_REGION_OFFSET(reserved);
#undef PA_SIMT_REAL_REGION_OFFSET

static_assert(
    sizeof(SimtWriterHistoryRecord) ==
        sizeof(SharedWriterHistoryRecord)
);
static_assert(
    alignof(SimtWriterHistoryRecord) ==
        alignof(SharedWriterHistoryRecord)
);
static_assert(
    offsetof(SimtWriterHistoryRecord, symbol_key) ==
        offsetof(SharedWriterHistoryRecord, symbol_key)
);
static_assert(
    offsetof(SimtWriterHistoryRecord, previous_writer) ==
        offsetof(SharedWriterHistoryRecord, previous_writer)
);
static_assert(
    sizeof(SimtWriterHistoryCell) == sizeof(SharedWriterHistoryCell)
);
static_assert(
    alignof(SimtWriterHistoryCell) == alignof(SharedWriterHistoryCell)
);
#define PA_SIMT_REAL_HISTORY_OFFSET(field)                                   \
    static_assert(                                                          \
        offsetof(SimtWriterHistoryCell, field) ==                           \
        offsetof(SharedWriterHistoryCell, field)                            \
    )
PA_SIMT_REAL_HISTORY_OFFSET(magic);
PA_SIMT_REAL_HISTORY_OFFSET(writer_task);
PA_SIMT_REAL_HISTORY_OFFSET(count);
PA_SIMT_REAL_HISTORY_OFFSET(reserved);
PA_SIMT_REAL_HISTORY_OFFSET(entries);
PA_SIMT_REAL_HISTORY_OFFSET(padding);
#undef PA_SIMT_REAL_HISTORY_OFFSET
static_assert(kSimtWriterHistoryMagic == kSharedWriterHistoryMagic);

static_assert(sizeof(TaskCell) == 64U);
static_assert(alignof(TaskCell) == 64U);
static_assert(offsetof(TaskCell, flag) == 0U);
static_assert(offsetof(TaskCell, vend) == 8U);
static_assert(offsetof(TaskCell, deps_prepared) == 16U);
static_assert(sizeof(SharedRegionPayload) == 64U);
static_assert(alignof(SharedRegionPayload) == 64U);
static_assert(offsetof(SharedRegionPayload, value) == 0U);
static_assert(sizeof(SharedRegionSlot) == 128U);
static_assert(alignof(SharedRegionSlot) == 64U);
static_assert(offsetof(SharedRegionSlot, payload) == 0U);
static_assert(offsetof(SharedRegionSlot, seq) == 64U);
static_assert(sizeof(SharedBucketState) == 128U);
static_assert(alignof(SharedBucketState) == 64U);
static_assert(offsetof(SharedBucketState, head) == 0U);
static_assert(offsetof(SharedBucketState, tail) == 64U);

static_assert(kMapCapacity == 16384U);
static_assert(kMapBucketCapacity == 128U);
static_assert(kMapBuckets == 128U);
static_assert(kSharedMapEmptySeq == -1);
static_assert(kSharedHeapShards == 8U);
static_assert(kRuntimePlanConsumerCapacity == 4352U);
static_assert(kOutputAlignment == kSimtOutputAlignment);
static_assert(kMaxTensorDims == kSimtTensorDims);
static_assert(kMaxTaskTensors == 32U);
static_assert(kMaxTaskScalars == 16U);
static_assert(kSharedOutputMaxPerTask == 8U);
static_assert(plan::kMaxTaskTensors == 32U);
static_assert(plan::kMaxTaskScalars == 16U);
static_assert(plan::kMaxExplicitDependencies == 16U);
static_assert(plan::kMaxRuntimeOutputsPerTask == 8U);
static_assert(plan::kTensorDescWords == 16U);
static_assert(kMaxTaskTensors == plan::kMaxTaskTensors);
static_assert(kMaxTaskScalars == plan::kMaxTaskScalars);
static_assert(
    kSharedOutputMaxPerTask == plan::kMaxRuntimeOutputsPerTask
);
static_assert(kRuntimePlanConsumerCapacity == kMaxTasks);
static_assert(exec::kExecMaxTensors == kMaxTaskTensors);
static_assert(exec::kExecMaxScalars == kMaxTaskScalars);
static_assert(exec::kExecMaxFanin == plan::kMaxExplicitDependencies);
static_assert(exec::kExecTensorDescWords == plan::kTensorDescWords);
static_assert(exec::kExecInvalidFunctionId == plan::kInvalidFunctionId);

static_assert(sizeof(SharedOutputCell) == 2048U);
static_assert(alignof(SharedOutputCell) == 64U);
static_assert(offsetof(SharedOutputCell, published) == 0U);
static_assert(offsetof(SharedOutputCell, last_writer) == 512U);
static_assert(offsetof(SharedOutputCell, tensors) == 1024U);
static_assert(
    sizeof(SharedOutputCell::published) / sizeof(AtomicLine) ==
        plan::kMaxRuntimeOutputsPerTask
);
static_assert(
    sizeof(SharedOutputCell::last_writer) / sizeof(AtomicLine) ==
        plan::kMaxRuntimeOutputsPerTask
);
static_assert(
    sizeof(SharedOutputCell::tensors) / sizeof(TensorDesc) ==
        plan::kMaxRuntimeOutputsPerTask
);

static_assert(
    sizeof(SharedTensorMapSidecar::buckets) /
            sizeof(SharedBucketState) ==
        kMapBuckets
);
static_assert(
    sizeof(SharedTensorMapSidecar::slots) /
            sizeof(SharedRegionSlot) ==
        kMapCapacity
);
static_assert(
    sizeof(SharedTensorMapSidecar::shared_outputs) /
            sizeof(SharedOutputCell) ==
        kRuntimePlanConsumerCapacity
);
static_assert(
    sizeof(SharedTensorMapSidecar::shared_heap_cursor) /
            sizeof(AtomicLine) ==
        kSharedHeapShards
);
static_assert(
    sizeof(SharedTensorMapSidecar::writer_history) /
            sizeof(SharedWriterHistoryCell) ==
        kRuntimePlanConsumerCapacity
);
static_assert(sizeof(SharedTensorMapSidecar) == 12434560U);
static_assert(alignof(SharedTensorMapSidecar) == 64U);
static_assert(offsetof(SharedTensorMapSidecar, committed_tasks) == 0U);
static_assert(offsetof(SharedTensorMapSidecar, reclaim_upto) == 64U);
static_assert(offsetof(SharedTensorMapSidecar, buckets) == 128U);
static_assert(offsetof(SharedTensorMapSidecar, shared_outputs) == 2113664U);
static_assert(
    offsetof(SharedTensorMapSidecar, shared_heap_cursor) == 11026560U
);
static_assert(
    offsetof(SharedTensorMapSidecar, shared_heap_vend) == 11027072U
);
static_assert(
    offsetof(SharedTensorMapSidecar, writer_history) == 11027648U
);
static_assert(
    offsetof(SharedTensorMapSidecar, slots) ==
        offsetof(SharedTensorMapSidecar, buckets) +
            sizeof(SharedTensorMapSidecar::buckets)
);
static_assert(
    offsetof(SharedTensorMapSidecar, shared_heap_vend) ==
        offsetof(SharedTensorMapSidecar, shared_heap_cursor) +
            sizeof(SharedTensorMapSidecar::shared_heap_cursor)
);
static_assert(
    offsetof(SharedTensorMapSidecar, writer_history) >
        offsetof(SharedTensorMapSidecar, shared_heap_vend)
);

static_assert(
    sizeof(SharedClaimTournamentRoot) ==
        kSharedClaimTournamentNodeStride
);
static_assert(alignof(SharedClaimTournamentRoot) == 64U);
static_assert(offsetof(SharedClaimTournamentRoot, owner) == 0U);
static_assert(
    offsetof(SharedClaimTournamentRoot, insert_completion) == 64U
);
static_assert(
    sizeof(SharedClaimTournamentTask) ==
        kSharedClaimTournamentNodeStride *
            (1U + kSharedClaimTournamentMaxGroups)
);
static_assert(alignof(SharedClaimTournamentTask) == 64U);
static_assert(offsetof(SharedClaimTournamentTask, root) == 0U);
static_assert(
    offsetof(SharedClaimTournamentTask, local) ==
        kSharedClaimTournamentNodeStride
);
static_assert(
    sizeof(SchedulerState::claim_tournament) /
            sizeof(SharedClaimTournamentTask) ==
        kRuntimePlanConsumerCapacity
);
static_assert(
    sizeof(SchedulerState::exec_cells) /
            sizeof(exec::SharedExecCell) ==
        kRuntimePlanConsumerCapacity
);
static_assert(
    offsetof(SchedulerState, shared_map) % alignof(SharedTensorMapSidecar) ==
        0U
);
static_assert(
    (offsetof(SchedulerState, claim_tournament) +
     offsetof(SharedClaimTournamentTask, root) +
     offsetof(SharedClaimTournamentRoot, insert_completion)) % 128U == 0U
);
static_assert(
    offsetof(SchedulerState, exec_cells) % exec::kExecCacheLineBytes == 0U
);
static_assert(
    offsetof(SchedulerState, runtime_plan_control) %
            plan::kAtomicIsolationBytes ==
        0U
);

static_assert(sizeof(exec::SharedExecControl) == 64U);
static_assert(alignof(exec::SharedExecControl) == 64U);
static_assert(offsetof(exec::SharedExecControl, state) == 0U);
static_assert(sizeof(exec::ExecPayloadStorage) == 4352U);
static_assert(alignof(exec::ExecPayloadStorage) == 64U);
static_assert(offsetof(exec::ExecPayloadStorage, words) == 0U);
static_assert(sizeof(exec::SharedExecCell) == 4416U);
static_assert(alignof(exec::SharedExecCell) == 64U);
static_assert(offsetof(exec::SharedExecCell, control) == 0U);
static_assert(offsetof(exec::SharedExecCell, payload) == 64U);
static_assert(sizeof(exec::SharedExecFatalControl) == 64U);
static_assert(alignof(exec::SharedExecFatalControl) == 64U);
static_assert(offsetof(exec::SharedExecFatalControl, state) == 0U);
static_assert(static_cast<uint8_t>(exec::ExecPhase::Empty) == 0U);
static_assert(static_cast<uint8_t>(exec::ExecPhase::Building) == 1U);
static_assert(static_cast<uint8_t>(exec::ExecPhase::Built) == 2U);
static_assert(static_cast<uint8_t>(exec::ExecPhase::Claimed) == 3U);
static_assert(static_cast<uint8_t>(exec::ExecPhase::Done) == 4U);
static_assert(static_cast<uint8_t>(exec::ExecEngineClass::None) == 0U);
static_assert(static_cast<uint8_t>(exec::ExecEngineClass::Aic) == 1U);
static_assert(static_cast<uint8_t>(exec::ExecEngineClass::Aiv) == 2U);
static_assert(static_cast<uint8_t>(exec::ExecEngineClass::Joint) == 3U);

static_assert(static_cast<uint8_t>(plan::EngineClass::MetadataOnly) == 0U);
static_assert(static_cast<uint8_t>(plan::EngineClass::Aic) == 1U);
static_assert(static_cast<uint8_t>(plan::EngineClass::Aiv) == 2U);
static_assert(static_cast<uint8_t>(plan::TensorTag::Input) ==
              static_cast<uint8_t>(TensorArgType::Input));
static_assert(static_cast<uint8_t>(plan::TensorTag::Output) ==
              static_cast<uint8_t>(TensorArgType::Output));
static_assert(static_cast<uint8_t>(plan::TensorTag::Inout) ==
              static_cast<uint8_t>(TensorArgType::Inout));
static_assert(static_cast<uint8_t>(plan::TensorTag::OutputExisting) ==
              static_cast<uint8_t>(TensorArgType::OutputExisting));
static_assert(static_cast<uint8_t>(plan::TensorTag::NoDependency) ==
              static_cast<uint8_t>(TensorArgType::NoDependency));
static_assert(static_cast<uint8_t>(DataType::Float32) == 0U);
static_assert(static_cast<uint8_t>(DataType::Float16) == 1U);
static_assert(static_cast<uint8_t>(DataType::Int32) == 2U);
static_assert(static_cast<uint8_t>(DataType::Int16) == 3U);
static_assert(static_cast<uint8_t>(DataType::Int8) == 4U);
static_assert(static_cast<uint8_t>(DataType::Uint8) == 5U);
static_assert(static_cast<uint8_t>(DataType::Bfloat16) == 6U);
static_assert(static_cast<uint8_t>(DataType::Int64) == 7U);
static_assert(static_cast<uint8_t>(DataType::Uint64) == 8U);
static_assert(static_cast<uint8_t>(DataType::Uint16) == 9U);
static_assert(static_cast<uint8_t>(DataType::Uint32) == 10U);
static_assert(static_cast<uint8_t>(DataType::Bool) == 11U);
static_assert(static_cast<uint8_t>(DataType::Count) == kSimtDataTypeCount);

// RoutePolicy 是算子 adapter 的唯一注入点。公共 Runtime 不解释 PA
// TaskKind、task-id 周期或固定函数表；调用方必须在每次 Build 前 BindTask，
// policy 再用 canonical header 中的 adapter provenance 做业务校验。
template <typename Ops, typename RoutePolicy>
struct SimtRealStateRuntime {
    PA_GM SchedulerState *state;
    uint32_t task_capacity;
    uint32_t bound_task_id;
    uint64_t watchdog_ticks;
    RoutePolicy route_policy;

    PA_DEVICE bool Valid() const
    {
        return state != nullptr && task_capacity != 0U &&
               task_capacity <= kRuntimePlanConsumerCapacity &&
               state->heap_window >= 0 && watchdog_ticks != 0U;
    }

    PA_DEVICE bool BindTask(uint32_t task_id)
    {
        if (!Valid() || task_id >= task_capacity) return false;
        bound_task_id = task_id;
        return true;
    }

    PA_DEVICE PA_GM volatile int64_t *OutputPublished(
        uint32_t task_id, uint32_t output_slot
    ) const
    {
        if (!TaskAndOutputValid(task_id, output_slot)) return nullptr;
        return &state->shared_map.shared_outputs[task_id]
                    .published[output_slot].value;
    }

    PA_DEVICE PA_GM volatile int64_t *OutputLastWriter(
        uint32_t task_id, uint32_t output_slot
    ) const
    {
        if (!TaskAndOutputValid(task_id, output_slot)) return nullptr;
        return &state->shared_map.shared_outputs[task_id]
                    .last_writer[output_slot].value;
    }

    PA_DEVICE PA_GM volatile uint64_t *OutputDescriptorWords(
        uint32_t task_id, uint32_t output_slot
    ) const
    {
        if (!TaskAndOutputValid(task_id, output_slot)) return nullptr;
        return reinterpret_cast<PA_GM volatile uint64_t *>(
            &state->shared_map.shared_outputs[task_id]
                 .tensors[output_slot]
        );
    }

    PA_DEVICE PA_GM volatile int64_t *InsertCompletion(
        uint32_t task_id
    ) const
    {
        return task_id < task_capacity
            ? &state->claim_tournament[task_id]
                   .root.insert_completion.value
            : nullptr;
    }

    PA_DEVICE PA_GM volatile int32_t *GlobalFatal() const
    {
        return state == nullptr ? nullptr : &state->fatal.value;
    }

    PA_DEVICE uint64_t WatchdogTicks() const
    {
        return watchdog_ticks;
    }

    PA_DEVICE uint32_t TaskCapacity() const
    {
        return task_capacity;
    }

    PA_DEVICE bool FaninLowerBound(
        uint32_t reader_task, int32_t &lower_bound
    ) const
    {
        lower_bound = 0;
        if (state == nullptr || reader_task >= task_capacity ||
            reader_task > static_cast<uint32_t>(INT32_MAX) ||
            state->heap_window < 0) {
            return false;
        }
        const int32_t task = static_cast<int32_t>(reader_task);
        lower_bound = task > state->heap_window
            ? task - state->heap_window
            : 0;
        return true;
    }

    PA_DEVICE PA_GM SharedWriterHistoryCell *WriterHistory(
        uint32_t task_id
    ) const
    {
        if (task_id >= task_capacity) return nullptr;
        return &state->shared_map.writer_history[task_id];
    }

    PA_DEVICE bool PublishBuildFatal(
        uint32_t task_id, uint32_t reason
    ) const
    {
        (void)task_id;
        (void)reason;
        if (state == nullptr) return false;
        // Scheduler fatal 与 canonical Plan fatal 都是单调 0->1。它们是
        // 两条真实 atomic-only control，不保存窄 builder 私有原因镜像。
        const int32_t scheduler_before = Ops::Exchange(
            &state->fatal.value, int32_t{1}
        );
        const int64_t plan_before = Ops::Exchange(
            &state->runtime_plan_control.fatal.value, int64_t{1}
        );
        return (scheduler_before == 0 || scheduler_before == 1) &&
               (plan_before == 0 || plan_before == 1);
    }

    PA_DEVICE bool ValidateAdapterRoute(
        uint8_t adapter_flags, uint16_t adapter_data,
        uint32_t function_id, plan::EngineClass engine
    ) const
    {
        return bound_task_id < task_capacity &&
               route_policy.Validate(
                   bound_task_id, adapter_flags, adapter_data,
                   function_id, engine
               );
    }

    PA_DEVICE bool ReserveOutputHeap(
        uint32_t task_id, uint64_t total,
        uint64_t &first_address, uint64_t &aggregate_vend
    ) const
    {
        first_address = 0U;
        aggregate_vend = 0U;
        if (state == nullptr || task_id >= task_capacity ||
            (total != 0U && state->heap_base == 0U) ||
            state->heap_base > UINT64_MAX - state->heap_size) {
            return false;
        }
        SharedHeapReservation reservation{};
        if (!ReserveSharedOutputHeap<Ops, false>(
                state->shared_map, task_id, total,
                state->heap_size, reservation
            ) || reservation.task_base > state->heap_size ||
            total > state->heap_size - reservation.task_base) {
            return false;
        }
        aggregate_vend = reservation.aggregate_vend;
        if (total == 0U) return true;
        if (state->heap_base > UINT64_MAX - reservation.task_base) {
            return false;
        }
        first_address = state->heap_base + reservation.task_base;
        return first_address != 0U;
    }

    PA_DEVICE uint32_t OrdinaryBucket(uint64_t address) const
    {
        return TensorMapHash(address);
    }

    PA_DEVICE bool CheckOrdinaryAppend(
        const SimtWriterRegion *entries, const uint16_t *buckets,
        const uint8_t *bucket_ordinals, uint32_t count,
        uint32_t task_id
    ) const
    {
        if (state == nullptr || task_id >= task_capacity ||
            count > kMaxTaskTensors ||
            (count != 0U && entries == nullptr)) {
            return false;
        }
        // SimtWriterRegion 是 canonical value ABI，不是已启动
        // SharedRegionValue 对象。即使 layout 断言相等，也不能
        // 把整个窄类型数组 reinterpret 成 production 数组。
        SharedRegionValue real_entries[kMaxTaskTensors]{};
        for (uint32_t index = 0U; index < count; ++index) {
            real_entries[index] = MakeRealRegion(entries[index]);
        }
        return SharedCheckPreparedTaskAppend<Ops, false>(
                   state->shared_map, real_entries, buckets,
                   bucket_ordinals, count,
                   /*reclaim_upto=*/-1,
                   static_cast<int32_t>(task_id)
               ) == SharedAppendCheck::Ready;
    }

    PA_DEVICE bool AppendOrdinary(
        const SimtWriterRegion *entries, const uint16_t *buckets,
        uint32_t count, uint32_t task_id
    ) const
    {
        if (state == nullptr || task_id >= task_capacity ||
            count > kMaxTaskTensors ||
            (count != 0U &&
             (entries == nullptr || buckets == nullptr))) {
            return false;
        }
        for (uint32_t index = 0U; index < count; ++index) {
            const SharedRegionValue real_entry =
                MakeRealRegion(entries[index]);
            if (!AppendPreparedEntryStcg(
                    real_entry, buckets[index], task_id
                )) {
                return false;
            }
        }
        return true;
    }

    PA_DEVICE bool LookupOrdinary(
        const SimtCanonicalTensorDesc &descriptor,
        uint32_t reader_task, int32_t &producer
    ) const
    {
        producer = -1;
        if (state == nullptr || reader_task >= task_capacity ||
            descriptor.dtype >= static_cast<uint8_t>(DataType::Count)) {
            return false;
        }
        // 先用窄 builder 已做溢出检查的同一 range 公式构造 query，随后
        // 直接进入真实 generic SharedLookupRegion。避免把 SIMT uint8/bool
        // descriptor 再伪装成 GM TensorDesc；reader 的 [N-H,N) 下界、
        // seq 双检和 atomic->DCCI->payload 次序仍全部来自 production helper。
        SimtWriterRegion narrow_query{};
        if (!SimtMakeWriterRegion(
                descriptor, reader_task, narrow_query
            )) {
            return false;
        }
        const SharedRegionValue real_query{
            narrow_query.buffer_addr,
            narrow_query.lo,
            narrow_query.hi,
            /*producer=*/-1,
            /*reserved=*/0U,
        };
        bool protocol_ok = false;
        producer = SharedLookupRegion<Ops, false, true>(
            state->shared_map, real_query,
            static_cast<int32_t>(reader_task),
            state->heap_window, protocol_ok
        );
        return protocol_ok;
    }

    PA_DEVICE bool PublishMetadataCompletion(
        uint32_t task_id, uint64_t vend
    ) const
    {
        if (state == nullptr || task_id >= task_capacity) return false;
        PA_GM TaskCell &cell = state->tasks[task_id];
        // 与 Scalar CompleteTask 相同：vend 与 flag 都是返回型 atomic，
        // 中间只有 publication fence。旧值校验拒绝重复 completion；这里
        // 绝不能把 vend 改成 stcg 或把二者拆到 mirror cache line。
        const uint64_t prior_vend = Ops::Exchange(&cell.vend, vend);
        if (prior_vend != 0U) return false;
        Ops::StoreBarrier();
        const int64_t prior_flag = Ops::Exchange(
            &cell.flag, int64_t{1}
        );
        return prior_flag == 0;
    }

    PA_DEVICE PA_GM exec::SharedExecCell &ExecCell(
        uint32_t task_id
    ) const
    {
        return state->exec_cells[task_id];
    }

    PA_DEVICE PA_GM exec::SharedExecFatalControl &ExecFatal() const
    {
        return state->exec_fatal;
    }

private:
    PA_DEVICE static SharedRegionValue MakeRealRegion(
        const SimtWriterRegion &entry
    )
    {
        return SharedRegionValue{
            entry.buffer_addr,
            entry.lo,
            entry.hi,
            entry.producer,
            entry.reserved,
        };
    }

    PA_DEVICE bool TaskAndOutputValid(
        uint32_t task_id, uint32_t output_slot
    ) const
    {
        return state != nullptr && task_id < task_capacity &&
               output_slot < kSharedOutputMaxPerTask;
    }

    PA_DEVICE bool AppendPreparedEntryStcg(
        const SharedRegionValue &entry, uint32_t bucket,
        uint32_t
    ) const
    {
        if (bucket >= kMapBuckets || entry.producer < 0 ||
            entry.reserved != 0U || entry.lo >= entry.hi) {
            return false;
        }
        PA_GM SharedBucketState &controls =
            state->shared_map.buckets[bucket];
        const int64_t head = Ops::Load(&controls.head.value);
        const int64_t tail = Ops::Load(&controls.tail.value);
        if (head < 0 || tail < head || tail == INT64_MAX ||
            static_cast<uint64_t>(tail - head) >=
                kMapBucketCapacity) {
            return false;
        }
        const uint64_t cursor = static_cast<uint64_t>(tail);
        PA_GM SharedRegionSlot &slot = state->shared_map.slots[
            SharedTensorMapSlotIndex(bucket, cursor)
        ];
        const int64_t expected_old =
            cursor < kMapBucketCapacity
                ? kSharedMapEmptySeq
                : static_cast<int64_t>(cursor - kMapBucketCapacity);
        if (Ops::Exchange(
                &slot.seq.value, kSharedMapEmptySeq
            ) != expected_old) {
            return false;
        }

        // Scalar generic helper 的 slot/head/tail/absolute-seq 算法保持
        // 不变；唯一平台差异是 SIMT writer 逐 64bit 用 stcg bypass。
        // writer 不执行 DCCI。fence 之后才允许 seq/tail control 发布；
        // reader 仍走真实 SharedReadRegionSlot 的 atomic->DCCI->payload->
        // atomic 双检。
        PA_GM volatile uint64_t *payload_words =
            reinterpret_cast<PA_GM volatile uint64_t *>(
                &slot.payload.value
            );
        Ops::StorePayloadWord(&payload_words[0], entry.buffer_addr);
        Ops::StorePayloadWord(&payload_words[1], entry.lo);
        Ops::StorePayloadWord(&payload_words[2], entry.hi);
        Ops::StorePayloadWord(
            &payload_words[3],
            static_cast<uint64_t>(
                static_cast<uint32_t>(entry.producer)
            ) |
                (static_cast<uint64_t>(entry.reserved) << 32U)
        );
        Ops::StoreBarrier();
        if (Ops::Exchange(
                &slot.seq.value, static_cast<int64_t>(cursor)
            ) != kSharedMapEmptySeq) {
            return false;
        }
        return Ops::Exchange(
                   &controls.tail.value, tail + 1
               ) == tail;
    }
};

}  // namespace pa_scheduler::aicpu_plan_simt

#endif  // PA_SCHEDULER_AICPU_PLAN_SIMT_REAL_STATE_RUNTIME_H
