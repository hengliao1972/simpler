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

#ifndef PA_SCHEDULER_COMMON_PA_TRACE_H
#define PA_SCHEDULER_COMMON_PA_TRACE_H

#include "pa_model.h"

#ifndef PA_DEVICE_NOINLINE
#define PA_DEVICE_NOINLINE PA_DEVICE
#endif

#ifndef PA_LOOP_NOUNROLL
#define PA_LOOP_NOUNROLL
#endif

namespace pa_scheduler {

// 记录区与真实 PA 一样直接拼在定长 Header 后面；worker_id 只选择自己的
// records 分区，避免记录动作本身制造跨核共享写热点。
PA_DEVICE PA_GM TraceStorageRecord *GetTraceRecords(
    PA_GM TraceHeader *header, uint32_t worker_id
) {
    return reinterpret_cast<PA_GM TraceStorageRecord *>(
        reinterpret_cast<PA_GM uint8_t *>(header) +
        sizeof(TraceHeader) +
        static_cast<size_t>(worker_id) * kTraceWorkerBytes +
        kTraceSubmitClaimBytesPerCore
    );
}

#if PTO_FDWIC_SHARED_MAP
PA_DEVICE PA_GM SharedSubmitClaimTraceRecord *
GetSharedSubmitClaimRecords(PA_GM TraceStorageRecord *records) {
    // shared 每 worker 分区内，32B Submit/Claim 区紧邻在通用记录区之前。
    return reinterpret_cast<
        PA_GM SharedSubmitClaimTraceRecord *
    >(
        reinterpret_cast<PA_GM uint8_t *>(records) -
        kTraceSubmitClaimBytesPerCore
    );
}
#endif

struct AtomicPollBurst {
    uint64_t start_cycle[kAtomicPollBatchSiteCount];
    uint32_t call_count[kAtomicPollBatchSiteCount];
    uint32_t active_mask;
    uint32_t enabled_mask;
};

struct TraceContext {
    PA_GM TraceCoreState *core;
    PA_GM TraceStorageRecord *records;
    uint32_t capacity;
    bool atomics_enabled;
    int32_t lane;
    int32_t block_id;
    int32_t core_idx;
    // 本 worker 的记录槽位只由本 scalar 写。把 count/dropped 留在本地，
    // kernel 末尾再一次性发布到 core-state，避免每写一条 32B record 都
    // 额外读写另一条 GM cache line。
    uint32_t record_count;
    uint32_t dropped_records;
    // 轮询调用数留在 worker 私有上下文，最终一次性发布到 core state；
    // 等待热路不为计数再写共享/GM 状态。
    uint64_t poll_calls;
    uint64_t poll_batch_records;
    bool atomic_counter_overflow;
    // DCCI 区域观察与 Atomic 完全独立：calls 可以因 terminal observer
    // 聚合而大于 records，lines 是所有区域实际覆盖的 64B 行数。
    uint64_t dcci_calls;
    uint64_t dcci_lines;
    uint64_t dcci_records;
    bool dcci_counter_overflow;
    AtomicPollBurst poll_burst;
};

PA_DEVICE bool AtomicSwimlaneEnabled(const TraceContext &trace) {
#if PA_BUILD_ATOMIC_SWIMLANE
    (void)trace;
    return true;
#else
    return trace.atomics_enabled;
#endif
}

PA_DEVICE bool TraceStorageAttached(const TraceContext &trace) {
#if PA_BUILD_ATOMIC_SWIMLANE
    // 专用完整泳道构建在调度入口只校验一次 AttachTrace 结果；通过后，
    // 每条阶段/Atomic/DCCI raw 不再重复读取三个不变量。
    (void)trace;
    return true;
#else
    return trace.core != nullptr && trace.records != nullptr &&
           trace.capacity != 0;
#endif
}

// pa_model.h 也向 host 暴露同一 raw ABI 映射，但 CCEC/AscendC 的单个 TU
// 会先以 host 语境包含该头，再实例化 device 调度器。这里保留明确的 device
// 版本，避免设备函数误调用先前已实例化的 __host__ helper。
PA_DEVICE AtomicOp TraceAtomicSiteExpectedOp(AtomicSite site) {
    switch (site) {
        case AtomicSite::StartupIncrement:
        case AtomicSite::ReplayDoneIncrement:
        case AtomicSite::SharedHeapCursorReserve:
        case AtomicSite::SharedHeapVendAdvance:
            return AtomicOp::FetchAdd;
        case AtomicSite::FatalSet:
        case AtomicSite::CompletionVendExchange:
        case AtomicSite::CompletionFlagExchange:
            return AtomicOp::Exchange;
        case AtomicSite::ClaimMax:
        case AtomicSite::FrontierMax:
            return AtomicOp::FetchMax;
        case AtomicSite::SharedInsertTurnHandoff:
        case AtomicSite::SharedMetadataLastWriterCommit:
            return AtomicOp::CompareExchange;
        case AtomicSite::SharedOutputWriterReserve:
            return AtomicOp::FetchMax;
        case AtomicSite::SharedOutputPublishedExchange:
        case AtomicSite::SharedMapAppendSeqResetExchange:
        case AtomicSite::SharedMapAppendSeqPublishExchange:
        case AtomicSite::SharedMapAppendTailExchange:
        case AtomicSite::SharedOutputRollbackExchange:
            return AtomicOp::Exchange;
        default:
            return AtomicOp::Load;
    }
}

PA_DEVICE int32_t TraceAtomicPollBatchIndex(AtomicSite site) {
    switch (site) {
        case AtomicSite::StartupPoll:
            return 0;
        case AtomicSite::FatalPoll:
            return 1;
        case AtomicSite::FaninFlagLoad:
            return 2;
        case AtomicSite::HeapFrontierLoad:
            return 3;
        case AtomicSite::HeapVendLoad:
            return 4;
        case AtomicSite::ReplayDonePoll:
            return 5;
        default:
            return -1;
    }
}

PA_DEVICE AtomicSite TraceAtomicPollBatchSite(uint32_t index) {
    switch (index) {
        case 0:
            return AtomicSite::StartupPoll;
        case 1:
            return AtomicSite::FatalPoll;
        case 2:
            return AtomicSite::FaninFlagLoad;
        case 3:
            return AtomicSite::HeapFrontierLoad;
        case 4:
            return AtomicSite::HeapVendLoad;
        case 5:
            return AtomicSite::ReplayDonePoll;
        default:
            return AtomicSite::Count;
    }
}

PA_DEVICE bool TraceAtomicSiteIsPollBatchable(AtomicSite site) {
    return TraceAtomicPollBatchIndex(site) >= 0 ||
           site == AtomicSite::SharedInsertTurnPoll;
}

// kernel.cpp 会先在 host 语境经 winner_workload.h 包含 pa_model.h。
// 与 Atomic/DCCI 映射相同，设备侧不能调用那次已经实例化的 host helper，
// 因此在本设备头中保留同一 packed ABI 的明确实现。
PA_DEVICE bool TraceCompactFieldsFit(
    int32_t task_id, int32_t function_id,
    TracePhase phase, uint32_t auxiliary
) {
    return task_id >= -1 &&
           task_id < static_cast<int32_t>(kMaxTasks) &&
           function_id >= -1 && function_id <= 3 &&
           static_cast<uint32_t>(phase) <
               static_cast<uint32_t>(TracePhase::Count) &&
           auxiliary <= kCompactTraceAuxiliaryMask;
}

PA_DEVICE uint32_t TracePackCompactFields(
    int32_t task_id, int32_t function_id,
    TracePhase phase, uint32_t auxiliary
) {
    return
        (static_cast<uint32_t>(task_id) &
         kCompactTraceTaskMask) |
        ((static_cast<uint32_t>(function_id) &
          kCompactTraceFunctionMask)
         << kCompactTraceFunctionShift) |
        (static_cast<uint32_t>(phase) <<
         kCompactTracePhaseShift) |
        (auxiliary << kCompactTraceAuxiliaryShift);
}

PA_DEVICE uint32_t TraceAtomicPollBatchMask(AtomicSite site) {
    const int32_t index = TraceAtomicPollBatchIndex(site);
    return index >= 0 && index < 32 ? 1U << static_cast<uint32_t>(index) : 0U;
}

// Attach 只缓存本 worker 的 header 状态、分区首址和物理 lane 信息。控制区
// cache invalidate 在公共调度入口完成，不能随 submit-pmu 编译掉泳道而消失。
template <typename Ops>
PA_DEVICE TraceContext AttachTrace(
    PA_GM SchedulerState *state, PA_GM const WorkerState &worker, uint32_t worker_id
) {
    TraceContext trace{};
    trace.lane = worker.lane;
    trace.block_id = worker.block_id;
    trace.core_idx = static_cast<int32_t>(worker_id);
#if PA_BUILD_TRACE_FREE
    // 诊断 ELF 不含 records 写入、atomic span 或 trace-only SYS_CNT；保留同一
    // TraceContext 形状只是为了复用调度协议源码。
    (void)state;
    return trace;
#else
    const uint64_t base = state->config.trace_base;
    const uint32_t capacity = state->config.trace_records_per_core;
    if ((state->config.trace_enabled & kTracePhasesEnabled) == 0 || base == 0 || capacity == 0 ||
        worker_id >= kWorkers) {
        return trace;
    }
    PA_GM TraceHeader *header = reinterpret_cast<PA_GM TraceHeader *>(base);
    if (header->magic != 0x4653574cU || header->version != 5 ||
        header->record_size_bytes != kTraceRecordSizeBytes ||
        header->records_per_core != capacity ||
        worker_id >= header->num_cores) {
        return trace;
    }
    trace.core = &header->cores[worker_id];
    trace.records = GetTraceRecords(header, worker_id);
    // 成功返回的不变量是 core/records/capacity 同时有效；任一前置条件失败则三者
    // 保持空值，后续 WriteTrace/FlushTraceCore 可无分支地安全退化为 no-op。
    trace.capacity = capacity;
    trace.atomics_enabled = (state->config.trace_enabled & kTraceAtomicsEnabled) != 0;
    // 所有 core-state 字段都等到 FlushTraceCore 一次性发布。记录期间只改
    // TraceContext 本地计数，避免启动和结束各把这条 64B GM 行弄脏一次。
    return trace;
#endif
}

template <bool Profile>
PA_DEVICE void WriteTrace(
    TraceContext &trace, WorkerResult &result, int32_t task_id, int32_t function_id,
    TracePhase trace_phase, ProfilePhase profile_phase, uint64_t start_cycle, uint64_t end_cycle,
    uint32_t flags = 0, uint32_t auxiliary = 0
);

// CCEC 不让栈上的 TraceContext/WorkerResult 引用跨非内联调用。这里仅把
// PollBatch 固定形状的 32-byte GM 写入抽成共享函数，以抑制各 phase 边界
// 内联后的代码膨胀；参数只有 GM 指针与标量，局部 batch 状态仍由调用者维护。
#if !PA_BUILD_TRACE_FREE
PA_DEVICE bool ReserveTraceRecord(
    TraceContext &trace, uint32_t &slot
) {
    // 调用点已经分别由 attached trace 或 atomics_enabled 门控；这里是
    // 每条 raw 都会经过的热路，只保留真正可能变化的容量判断。
    if (__builtin_expect(
            trace.record_count >= trace.capacity, 0
        )) {
        if (trace.dropped_records != UINT32_MAX) {
            ++trace.dropped_records;
        }
        return false;
    }
    slot = trace.record_count++;
    return true;
}

PA_DEVICE void WriteGenericTraceRecordRaw(
    PA_GM TraceStorageRecord *records, uint32_t slot,
    uint64_t start_cycle, uint64_t end_cycle,
    int32_t task_id, int32_t function_id,
    TracePhase phase, uint32_t flags, uint32_t auxiliary
) {
#if PA_BUILD_COMPACT_GENERIC_TRACE
    PA_GM CompactTraceRecord16 &record = records[slot];
    record.start_cycle_low = static_cast<uint32_t>(start_cycle);
    record.end_cycle_low = static_cast<uint32_t>(end_cycle);
    record.flags = flags;
    record.packed = TracePackCompactFields(
        task_id, function_id, phase, auxiliary
    );
#else
    PA_GM TraceRecord &record = records[slot];
    record.start_cycle = start_cycle;
    record.end_cycle = end_cycle;
    record.task_id = task_id;
    record.function_id = function_id;
    record.flags = flags;
    record.phase = static_cast<uint16_t>(phase);
    record.auxiliary = static_cast<uint16_t>(auxiliary);
#endif
}

PA_DEVICE_NOINLINE bool WritePollBatchRecordRaw(
    PA_GM TraceStorageRecord *records, uint32_t slot,
    uint64_t start_cycle, uint64_t end_cycle, uint32_t call_count,
    uint32_t site_id, bool return_ready_end = false
) {
    const AtomicSite site = static_cast<AtomicSite>(site_id);
    WriteGenericTraceRecordRaw(
        records, slot, start_cycle, end_cycle,
        -1, -1, TracePhase::Atomic,
        static_cast<uint32_t>(TraceAtomicSiteExpectedOp(site)) |
            kAtomicResultUsed | kAtomicPollBatch |
            (return_ready_end ? kAtomicReturnReady : 0U) |
            (call_count << kAtomicPollCountShift),
        site_id
    );
    return true;
}

#endif

PA_DEVICE DcciOp TraceDcciSiteExpectedOp(DcciSite site) {
    switch (site) {
        case DcciSite::SharedWriterHistoryFlush:
        case DcciSite::SharedOutputRollbackFlush:
        case DcciSite::SharedOutputDescriptorFlush:
        case DcciSite::SharedRegionAppendFlush:
        case DcciSite::ObserverTraceExport:
            return DcciOp::CleanOut;
        default:
            return DcciOp::Invalidate;
    }
}

PA_DEVICE uint32_t DcciTraceFlags(
    DcciOp op, bool trailing_dsb,
    uint32_t call_count, uint32_t line_count
) {
    return static_cast<uint32_t>(op) |
           (trailing_dsb ? kDcciTrailingDsb : 0U) |
           (call_count << kDcciCallCountShift) |
           (line_count << kDcciLineCountShift);
}

#if !PA_BUILD_TRACE_FREE
// 返回落盘 slot；-1 表示没有可用记录位。该 helper 只写 raw，不修改
// logical 计数，使普通单调用和 observer 两调用聚合共用同一物理格式。
PA_DEVICE_NOINLINE int32_t WriteDcciRecordRaw(
    PA_GM TraceStorageRecord *records, uint32_t slot,
    int32_t task_id, int32_t function_id,
    DcciSite site, DcciOp op, bool trailing_dsb,
    uint32_t call_count, uint32_t line_count,
    uint64_t start_cycle, uint64_t end_cycle
) {
    WriteGenericTraceRecordRaw(
        records, slot, start_cycle, end_cycle,
        task_id, function_id, TracePhase::Dcci,
        DcciTraceFlags(
            op, trailing_dsb, call_count, line_count
        ),
        static_cast<uint32_t>(site)
    );
    return static_cast<int32_t>(slot);
}
#endif

PA_DEVICE int32_t AppendDcciTrace(
    TraceContext &trace, int32_t task_id, int32_t function_id,
    DcciSite site, DcciOp op, bool trailing_dsb,
    uint32_t call_count, uint32_t line_count,
    uint64_t start_cycle, uint64_t end_cycle
) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)task_id;
    (void)function_id;
    (void)site;
    (void)op;
    (void)trailing_dsb;
    (void)call_count;
    (void)line_count;
    (void)start_cycle;
    (void)end_cycle;
    return -1;
#else
    if (!TraceStorageAttached(trace)) {
        return -1;
    }
    const bool shape_valid =
        static_cast<uint32_t>(site) <
            static_cast<uint32_t>(DcciSite::Count) &&
        static_cast<uint32_t>(op) <
            static_cast<uint32_t>(DcciOp::Count) &&
        op == TraceDcciSiteExpectedOp(site) &&
        call_count != 0 && call_count <= kDcciCallCountMask &&
        line_count >= call_count &&
        line_count <= kDcciLineCountMax &&
        end_cycle >= start_cycle
#if PA_BUILD_COMPACT_GENERIC_TRACE
        && TraceCompactFieldsFit(
            task_id, function_id, TracePhase::Dcci,
            static_cast<uint32_t>(site)
        )
#endif
        ;
    const bool counters_fit =
        trace.dcci_calls <= UINT64_MAX - call_count &&
        trace.dcci_lines <= UINT64_MAX - line_count &&
        trace.dcci_records != UINT64_MAX;
    if (!shape_valid || !counters_fit) {
        trace.dcci_counter_overflow = true;
        return -1;
    }
    trace.dcci_calls += call_count;
    trace.dcci_lines += line_count;
    uint32_t reserved_slot = 0;
    if (!ReserveTraceRecord(trace, reserved_slot)) {
        return -1;
    }
    const int32_t slot = WriteDcciRecordRaw(
        trace.records, reserved_slot,
        task_id, function_id, site, op, trailing_dsb,
        call_count, line_count, start_cycle, end_cycle
    );
    if (slot >= 0) {
        ++trace.dcci_records;
    }
    return slot;
#endif
}

PA_DEVICE bool WriteDcciTrace(
    TraceContext &trace, int32_t task_id, int32_t function_id,
    DcciSite site, DcciOp op, bool trailing_dsb,
    uint32_t line_count,
    uint64_t start_cycle, uint64_t end_cycle
) {
    return AppendDcciTrace(
               trace, task_id, function_id, site, op,
               trailing_dsb, 1, line_count,
               start_cycle, end_cycle
           ) >= 0;
}

template <typename Pointer>
PA_DEVICE uint32_t DcciRegionCacheLineCount(
    Pointer address, uint64_t bytes
) {
    if (address == nullptr || bytes == 0) {
        return 0;
    }
    const uint64_t begin = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(address)
    );
    if (bytes > UINT64_MAX - begin ||
        begin + bytes > UINT64_MAX - 63U) {
        return 0;
    }
    const uint64_t aligned_begin = begin & ~UINT64_C(63);
    const uint64_t aligned_end = (begin + bytes + 63U) & ~UINT64_C(63);
    const uint64_t lines = (aligned_end - aligned_begin) / 64U;
    return lines == 0 || lines > kDcciLineCountMax
        ? 0
        : static_cast<uint32_t>(lines);
}

// 区域级观察只在原 DCCI 前后各取一次时钟，并且无论区域覆盖多少条
// cache line 都只写一条 generic 物理记录（当前构建为 16B 或 32B）。
// begin/end 可选回传给已经存在的业务 detail span，避免同一
// FlushRegion 为两套观察重复读取 SYS_CNT。
template <
    typename Ops, bool ObserveDcci, bool IsInvalidate,
    typename Pointer
>
PA_DEVICE uint64_t TraceConfiguredDcciRegion(
    TraceContext *trace, int32_t task_id, int32_t function_id,
    DcciSite site, Pointer address, uint64_t bytes,
    uint64_t *begin_out = nullptr, uint64_t begin_override = 0
) {
    constexpr DcciOp op =
        IsInvalidate ? DcciOp::Invalidate : DcciOp::CleanOut;
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)task_id;
    (void)function_id;
    (void)site;
    (void)begin_override;
    if (begin_out != nullptr) {
        *begin_out = 0;
    }
    if constexpr (IsInvalidate) {
        Ops::InvalidateRegion(address, bytes);
    } else {
        Ops::FlushRegion(address, bytes);
    }
    return 0;
#else
    if constexpr (!ObserveDcci) {
        (void)trace;
        (void)task_id;
        (void)function_id;
        (void)site;
        (void)begin_override;
        if (begin_out != nullptr) {
            *begin_out = 0;
        }
        if constexpr (IsInvalidate) {
            Ops::InvalidateRegion(address, bytes);
        } else {
            Ops::FlushRegion(address, bytes);
        }
        return 0;
    } else {
        const uint32_t line_count =
            DcciRegionCacheLineCount(address, bytes);
        const bool observable =
            trace != nullptr && trace->core != nullptr &&
            trace->records != nullptr && trace->capacity != 0;
        const uint64_t begin = observable
            ? (begin_override != 0 ? begin_override : Ops::Now())
            : 0;
        if (begin_out != nullptr) {
            *begin_out = begin;
        }
        if constexpr (IsInvalidate) {
            Ops::InvalidateRegion(address, bytes);
        } else {
            Ops::FlushRegion(address, bytes);
        }
        if (!observable) {
            return 0;
        }
        const uint64_t end = Ops::Now();
        if (line_count == 0) {
            trace->dcci_counter_overflow = true;
            return end;
        }
        (void)WriteDcciTrace(
            *trace, task_id, function_id, site, op,
            /*trailing_dsb=*/true, line_count, begin, end
        );
        return end;
    }
#endif
}

template <typename Ops, bool ObserveDcci, typename Pointer>
PA_DEVICE uint64_t TraceConfiguredDcciInvalidate(
    TraceContext *trace, int32_t task_id, int32_t function_id,
    DcciSite site, Pointer address, uint64_t bytes,
    uint64_t *begin_out = nullptr, uint64_t begin_override = 0
) {
    return TraceConfiguredDcciRegion<Ops, ObserveDcci, true>(
        trace, task_id, function_id, site,
        address, bytes, begin_out, begin_override
    );
}

template <typename Ops, bool ObserveDcci, typename Pointer>
PA_DEVICE uint64_t TraceConfiguredDcciFlush(
    TraceContext *trace, int32_t task_id, int32_t function_id,
    DcciSite site, Pointer address, uint64_t bytes,
    uint64_t *begin_out = nullptr, uint64_t begin_override = 0
) {
    return TraceConfiguredDcciRegion<Ops, ObserveDcci, false>(
        trace, task_id, function_id, site,
        address, bytes, begin_out, begin_override
    );
}

// 启动控制区必须先经 DCCI 才能读取 trace_base/trace_enabled，因此不能
// 在原语执行时访问 TraceContext。泳道构建只保存两端点，握手成功并
// AttachTrace 后再补写一条 StartupConfigInvalidate；无观察构建严格
// 退化为原始 DCCI，不增加 SYS_CNT 或 cache-line 计算。
template <typename Ops, typename Pointer>
PA_DEVICE void CapturePreAttachDcciInvalidate(
    Pointer address, uint64_t bytes,
    uint64_t &begin_cycle, uint64_t &end_cycle
) {
#if PA_BUILD_TRACE_FREE
    begin_cycle = 0;
    end_cycle = 0;
    Ops::InvalidateRegion(address, bytes);
#else
    begin_cycle = Ops::Now();
    Ops::InvalidateRegion(address, bytes);
    end_cycle = Ops::Now();
#endif
}

// 构建身份不匹配时 trace ABI 本身不可信，不能为了记录 fatal 而解释
// trace_base。中央化这个唯一 pre-attach atomic，仅用于 fail-closed。
template <typename Ops, typename T>
PA_DEVICE T PreAttachAtomicExchange(
    PA_GM volatile T *address, T value
) {
    return Ops::Exchange(address, value);
}

PA_DEVICE uint32_t AtomicTraceFlags(
    AtomicOp op, bool result_used, bool return_ready, bool value_zero = false,
    uint64_t retries = 0
) {
    // 高 24 bit 只能容纳有限重试次数；A5 硬件 atomicMax 当前报告 0，CPU CAS
    // 回归若超过范围则饱和，避免溢出覆盖低位的 op/语义标志。
    constexpr uint64_t kMaxRetries = (1ULL << (32 - kAtomicRetriesShift)) - 1;
    const uint32_t encoded_retries = static_cast<uint32_t>(retries > kMaxRetries ? kMaxRetries : retries);
    return static_cast<uint32_t>(op) | (result_used ? kAtomicResultUsed : 0U) |
           (value_zero ? kAtomicValueZero : 0U) | (return_ready ? kAtomicReturnReady : 0U) |
           (encoded_retries << kAtomicRetriesShift);
}

PA_DEVICE void CountAtomicCall(
    TraceContext &trace, WorkerResult &result, bool poll_batch
) {
    // 单轮 task 数与 2 秒 watchdog 给出了远低于 UINT64_MAX 的物理上界；
    // FlushTraceCore 仍校验 host ABI 的 UINT32 上限。PollBatch 已在
    // active call_count 中累计精确次数，统一延迟到收口时加总，避免每次
    // 轮询重复维护 atomic_calls 与 poll_calls 两个 64-bit 总数。
    (void)trace;
    if (!poll_batch) {
        ++result.atomic_trace_calls;
    }
}

// insert-turn 等待已经在生产循环中维护精确 polls；这里一次性把
// pending polls + 最终 Ready Load 聚合成一条 PollBatch，避免在最热循环
// 内为每次 Load 做 trace 分支、计数和起始状态维护。24-bit 编码不足时
// 明确令本次 trace 闭合失败，不饱和、不拆成随轮询数增长的多条记录。
PA_DEVICE bool WriteAggregateAtomicPollBatch(
    TraceContext &trace, WorkerResult &result, AtomicSite site,
    uint64_t start_cycle, uint64_t end_cycle, uint64_t call_count,
    bool return_ready_end
) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)result;
    (void)site;
    (void)start_cycle;
    (void)end_cycle;
    (void)call_count;
    (void)return_ready_end;
    return false;
#else
    if (!AtomicSwimlaneEnabled(trace)) return false;
    if (!TraceAtomicSiteIsPollBatchable(site) ||
        TraceAtomicSiteExpectedOp(site) != AtomicOp::Load ||
        call_count == 0 || call_count > kAtomicPollCountMax ||
        end_cycle < start_cycle ||
        (return_ready_end &&
         site != AtomicSite::SharedInsertTurnPoll) ||
        result.atomic_trace_calls > UINT64_MAX - call_count ||
        trace.poll_calls > UINT64_MAX - call_count) {
        trace.atomic_counter_overflow = true;
        return false;
    }
    result.atomic_trace_calls += call_count;
    trace.poll_calls += call_count;
    uint32_t slot = 0;
    const bool written =
        ReserveTraceRecord(trace, slot) &&
        WritePollBatchRecordRaw(
        trace.records, slot,
        start_cycle, end_cycle, static_cast<uint32_t>(call_count),
        static_cast<uint32_t>(site), return_ready_end
    );
    if (written) {
        if (trace.poll_batch_records == UINT64_MAX) {
            trace.atomic_counter_overflow = true;
        } else {
            ++trace.poll_batch_records;
        }
    }
    return written;
#endif
}

template <typename Ops>
PA_DEVICE_NOINLINE void AtomicPollBoundaryAtSlow(
    TraceContext &trace, WorkerResult *result, uint64_t end_cycle
) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)result;
    (void)end_cycle;
#else
    const uint32_t active_mask = trace.poll_burst.active_mask;
    // CCEC 默认会把固定 6-site 循环完整展开，再随几十个 phase 边界复制。
    // 禁止展开只控制代码体积；循环次数、site 顺序和同 cycle 关闭语义不变。
    PA_LOOP_NOUNROLL
    for (uint32_t index = 0; index < kAtomicPollBatchSiteCount; ++index) {
        const uint32_t bit = 1U << index;
        if ((active_mask & bit) == 0) continue;
        const uint32_t call_count = trace.poll_burst.call_count[index];
        if (call_count == 0 || call_count > kAtomicPollCountMax) {
            trace.atomic_counter_overflow = true;
            continue;
        }
        if (result != nullptr) {
            if (result->atomic_trace_calls >
                    UINT64_MAX - call_count ||
                trace.poll_calls > UINT64_MAX - call_count) {
                trace.atomic_counter_overflow = true;
                continue;
            }
            result->atomic_trace_calls += call_count;
            trace.poll_calls += call_count;
        }
        const AtomicSite site = TraceAtomicPollBatchSite(index);
        uint32_t slot = 0;
        const bool written =
            ReserveTraceRecord(trace, slot) &&
            WritePollBatchRecordRaw(
            trace.records, slot,
            trace.poll_burst.start_cycle[index], end_cycle,
            call_count, static_cast<uint32_t>(site)
        );
        if (written) {
            if (trace.poll_batch_records == UINT64_MAX) {
                trace.atomic_counter_overflow = true;
            } else {
                ++trace.poll_batch_records;
            }
        }
        trace.poll_burst.call_count[index] = 0;
    }
    trace.poll_burst.active_mask = 0;
#endif
}

template <typename Ops>
PA_DEVICE void AtomicPollBoundaryAt(
    TraceContext &trace, uint64_t end_cycle,
    WorkerResult *result = nullptr
) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)end_cycle;
    (void)result;
#else
    // 绝大多数阶段边界没有活跃等待 episode；先用一个可预测分支返回，
    // 只有真正需要落 PollBatch 时才进入共享慢函数，避免把固定 6-site
    // 收口逻辑复制到每个 TraceTimestamp 调用点。
    if (!AtomicSwimlaneEnabled(trace) ||
        trace.poll_burst.active_mask == 0) {
        return;
    }
    AtomicPollBoundaryAtSlow<Ops>(trace, result, end_cycle);
#endif
}

template <typename Ops>
PA_DEVICE void AtomicPollBoundary(TraceContext &trace, WorkerResult &result) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)result;
#else
    if (trace.poll_burst.active_mask == 0) return;
    AtomicPollBoundaryAt<Ops>(trace, Ops::Now(), &result);
#endif
}

template <typename Ops>
PA_DEVICE uint32_t AtomicPollRegionBegin(
    TraceContext &trace, WorkerResult &result, uint32_t poll_batch_mask
) {
    const uint32_t previous_mask = trace.poll_burst.enabled_mask;
#if PA_BUILD_TRACE_FREE
    (void)result;
    (void)poll_batch_mask;
#else
    if (!AtomicSwimlaneEnabled(trace)) return previous_mask;
    AtomicPollBoundary<Ops>(trace, result);
    trace.poll_burst.enabled_mask = previous_mask | poll_batch_mask;
#endif
    return previous_mask;
}

template <typename Ops>
PA_DEVICE void AtomicPollRegionEnd(
    TraceContext &trace, WorkerResult &result, uint32_t previous_mask
) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)result;
    (void)previous_mask;
#else
    if (!AtomicSwimlaneEnabled(trace)) return;
    AtomicPollBoundary<Ops>(trace, result);
    trace.poll_burst.enabled_mask = previous_mask;
#endif
}

PA_DEVICE bool AtomicPollBatchEnabled(
    TraceContext &trace, AtomicSite site, AtomicOp actual_op
) {
    return AtomicSwimlaneEnabled(trace) && TraceAtomicSiteIsPollBatchable(site) &&
           TraceAtomicSiteExpectedOp(site) == actual_op &&
           (trace.poll_burst.enabled_mask & TraceAtomicPollBatchMask(site)) != 0;
}

template <typename Ops>
PA_DEVICE void AccumulateAtomicPollCall(
    TraceContext &trace, WorkerResult &result, AtomicSite site, uint64_t start_cycle
) {
    const int32_t signed_index = TraceAtomicPollBatchIndex(site);
    if (signed_index < 0) {
        trace.atomic_counter_overflow = true;
        return;
    }
    const uint32_t index = static_cast<uint32_t>(signed_index);
    const uint32_t bit = 1U << index;
    if (__builtin_expect(
            (trace.poll_burst.active_mask & bit) == 0, 0
        )) {
        trace.poll_burst.start_cycle[index] = start_cycle;
        trace.poll_burst.call_count[index] = 0;
        trace.poll_burst.active_mask |= bit;
    }
    uint32_t &call_count = trace.poll_burst.call_count[index];
    ++call_count;
    if (__builtin_expect(
            call_count == kAtomicPollCountMax, 0
        )) {
        AtomicPollBoundary<Ops>(trace, result);
    }
}

template <typename Ops>
PA_DEVICE void WriteAtomicTrace(
    TraceContext &trace, WorkerResult &result, int32_t task_id, AtomicSite site, AtomicOp op,
    uint64_t start_cycle, uint64_t end_cycle, bool result_used, bool return_ready,
    bool value_zero = false, uint64_t retries = 0
) {
    // 一次源码 atomic 只写一条同时含 start/end 的 span；结束时间先于 32B record
    // 写入，因此本条区间不直接包含自己的记录写开销，但下一次竞争到达会受它影响。
    CountAtomicCall(trace, result, false);
#if PA_BUILD_TRACE_FREE
    (void)task_id;
    (void)site;
    (void)op;
    (void)start_cycle;
    (void)end_cycle;
    (void)result_used;
    (void)return_ready;
    (void)value_zero;
    (void)retries;
#else
    WriteTrace<false>(
        trace, result, task_id, -1,
        TracePhase::Atomic, ProfilePhase::ReplayTail,
        start_cycle, end_cycle,
        AtomicTraceFlags(
            op, result_used, return_ready,
            value_zero, retries
        ),
        static_cast<uint32_t>(site)
    );
#endif
}

template <typename Ops, typename T>
PA_DEVICE T TraceAtomicLoad(
    TraceContext &trace, WorkerResult &result, int32_t task_id, AtomicSite site,
    PA_GM volatile T *address, bool result_used = true
) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)result;
    (void)task_id;
    (void)site;
    (void)result_used;
    return Ops::Load(address);
#else
    if (!AtomicSwimlaneEnabled(trace)) return Ops::Load(address);
    const bool poll_batch = result_used && AtomicPollBatchEnabled(trace, site, AtomicOp::Load);
    const int32_t poll_index = poll_batch ? TraceAtomicPollBatchIndex(site) : -1;
    const bool first_in_batch = poll_batch &&
        (trace.poll_burst.active_mask & (1U << static_cast<uint32_t>(poll_index))) == 0;
    const uint64_t begin = !poll_batch || first_in_batch ? Ops::Now() : 0;
    const T old = Ops::Load(address);
    if (poll_batch) {
        CountAtomicCall(trace, result, true);
        AccumulateAtomicPollCall<Ops>(trace, result, site, begin);
        return old;
    }
    // CCEC 只在返回值本来就参与协议判断时插入一条依赖 MOV，再读 SYS_CNT。
    // 这样不会把未消费返回值的 RED/no-return 路径强制改成返回型 ATOM。
    const bool return_ready = result_used && Ops::kAtomicReturnReadyObserved;
    const uint64_t end = result_used ? Ops::NowAfterAtomicResult(old) : Ops::Now();
    WriteAtomicTrace<Ops>(
        trace, result, task_id, site, AtomicOp::Load, begin, end, result_used, return_ready,
        old == static_cast<T>(0)
    );
    return old;
#endif
}

template <typename Ops, typename T>
PA_DEVICE T TraceAtomicExchange(
    TraceContext &trace, WorkerResult &result, int32_t task_id, AtomicSite site,
    PA_GM volatile T *address, T value, bool result_used = false
) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)result;
    (void)task_id;
    (void)site;
    (void)result_used;
    return Ops::Exchange(address, value);
#else
    if (!AtomicSwimlaneEnabled(trace)) return Ops::Exchange(address, value);
    const uint64_t begin = Ops::Now();
    const T old = Ops::Exchange(address, value);
    const bool return_ready = result_used && Ops::kAtomicReturnReadyObserved;
    const uint64_t end = result_used ? Ops::NowAfterAtomicResult(old) : Ops::Now();
    WriteAtomicTrace<Ops>(
        trace, result, task_id, site, AtomicOp::Exchange, begin, end, result_used, return_ready
    );
    return old;
#endif
}

// Handoff 先只捕获 CAS 的 source/return-ready 边界。调用方完成成功判断并
// 固定 Register 父区间后才写 raw，避免 32B 记录写入污染父区间，同时
// 不把 CAS 后的比较和函数返回从 RegisterHandoffNextTurn 中删掉。
template <typename Ops>
PA_DEVICE int64_t CaptureAtomicCompareExchange(
    TraceContext &trace, PA_GM volatile int64_t *address,
    int64_t expected, int64_t desired,
    uint64_t &trace_begin, uint64_t &trace_end
) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    trace_begin = 0;
    trace_end = 0;
    return Ops::CompareExchange(address, expected, desired);
#else
    if (!AtomicSwimlaneEnabled(trace)) {
        trace_begin = 0;
        trace_end = 0;
        return Ops::CompareExchange(address, expected, desired);
    }
    trace_begin = Ops::Now();
    const int64_t old =
        Ops::CompareExchange(address, expected, desired);
    trace_end = Ops::NowAfterAtomicResult(old);
    return old;
#endif
}

template <typename Ops>
PA_DEVICE int64_t TraceAtomicCompareExchange(
    TraceContext &trace, WorkerResult &result, int32_t task_id,
    AtomicSite site, PA_GM volatile int64_t *address,
    int64_t expected, int64_t desired, bool result_used = true
) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)result;
    (void)task_id;
    (void)site;
    (void)result_used;
    return Ops::CompareExchange(address, expected, desired);
#else
    if (!AtomicSwimlaneEnabled(trace)) {
        return Ops::CompareExchange(address, expected, desired);
    }
    const uint64_t begin = Ops::Now();
    const int64_t old =
        Ops::CompareExchange(address, expected, desired);
    const bool return_ready =
        result_used && Ops::kAtomicReturnReadyObserved;
    const uint64_t end = result_used
        ? Ops::NowAfterAtomicResult(old)
        : Ops::Now();
    WriteAtomicTrace<Ops>(
        trace, result, task_id, site,
        AtomicOp::CompareExchange, begin, end,
        result_used, return_ready
    );
    return old;
#endif
}

template <typename Ops>
PA_DEVICE int64_t TraceAtomicFetchAdd(
    TraceContext &trace, WorkerResult &result, int32_t task_id, AtomicSite site,
    PA_GM volatile int64_t *address, int64_t value, bool result_used = false
) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)result;
    (void)task_id;
    (void)site;
    (void)result_used;
    return Ops::FetchAdd(address, value);
#else
    if (!AtomicSwimlaneEnabled(trace)) return Ops::FetchAdd(address, value);
    const uint64_t begin = Ops::Now();
    const int64_t old = Ops::FetchAdd(address, value);
    const bool return_ready = result_used && Ops::kAtomicReturnReadyObserved;
    const uint64_t end = result_used ? Ops::NowAfterAtomicResult(old) : Ops::Now();
    WriteAtomicTrace<Ops>(
        trace, result, task_id, site, AtomicOp::FetchAdd, begin, end, result_used, return_ready
    );
    return old;
#endif
}

template <typename Ops>
PA_DEVICE int64_t TraceAtomicFetchMax(
    TraceContext &trace, WorkerResult &result, int32_t task_id, AtomicSite site,
    PA_GM volatile int64_t *address, int64_t value, uint64_t &retries, bool result_used = true
) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)result;
    (void)task_id;
    (void)site;
    (void)result_used;
    return Ops::FetchMax(address, value, retries);
#else
    if (!AtomicSwimlaneEnabled(trace)) return Ops::FetchMax(address, value, retries);
    const uint64_t begin = Ops::Now();
    const int64_t old = Ops::FetchMax(address, value, retries);
    const bool return_ready = result_used && Ops::kAtomicReturnReadyObserved;
    const uint64_t end = result_used ? Ops::NowAfterAtomicResult(old) : Ops::Now();
    WriteAtomicTrace<Ops>(
        trace, result, task_id, site, AtomicOp::FetchMax, begin, end, result_used,
        return_ready, false, retries
    );
    return old;
#endif
}

// 公共 shared 原语同时服务正式 scheduler 与不创建 TraceContext 的隔离
// 单元测试。所有“可选观察”都在这一中央适配层退化为一次原始 Ops 调用；
// 业务头文件不再自行绕过 TraceAtomic*，源码门槛因而可以机械发现漏接。
template <typename Ops, typename T>
PA_DEVICE T TraceOptionalAtomicLoad(
    TraceContext *trace, WorkerResult *result, int32_t task_id,
    AtomicSite site, PA_GM volatile T *address,
    bool result_used = true
) {
    if (trace == nullptr || result == nullptr) {
        return Ops::Load(address);
    }
    return TraceAtomicLoad<Ops>(
        *trace, *result, task_id, site, address, result_used
    );
}

template <typename Ops, typename T>
PA_DEVICE T TraceOptionalAtomicExchange(
    TraceContext *trace, WorkerResult *result, int32_t task_id,
    AtomicSite site, PA_GM volatile T *address, T value,
    bool result_used = false
) {
    if (trace == nullptr || result == nullptr) {
        return Ops::Exchange(address, value);
    }
    return TraceAtomicExchange<Ops>(
        *trace, *result, task_id, site, address, value,
        result_used
    );
}

template <typename Ops>
PA_DEVICE int64_t TraceOptionalAtomicCompareExchange(
    TraceContext *trace, WorkerResult *result, int32_t task_id,
    AtomicSite site, PA_GM volatile int64_t *address,
    int64_t expected, int64_t desired, bool result_used = true
) {
    if (trace == nullptr || result == nullptr) {
        return Ops::CompareExchange(address, expected, desired);
    }
    return TraceAtomicCompareExchange<Ops>(
        *trace, *result, task_id, site, address,
        expected, desired, result_used
    );
}

template <typename Ops>
PA_DEVICE int64_t TraceOptionalAtomicFetchMax(
    TraceContext *trace, WorkerResult *result, int32_t task_id,
    AtomicSite site, PA_GM volatile int64_t *address,
    int64_t value, uint64_t &retries, bool result_used = true
) {
    if (trace == nullptr || result == nullptr) {
        return Ops::FetchMax(address, value, retries);
    }
    return TraceAtomicFetchMax<Ops>(
        *trace, *result, task_id, site, address, value,
        retries, result_used
    );
}

template <typename Ops, bool ObserveAtomics, typename T>
PA_DEVICE T TraceConfiguredAtomicLoad(
    TraceContext *trace, WorkerResult *result, int32_t task_id,
    AtomicSite site, PA_GM volatile T *address,
    bool result_used = true
) {
    if constexpr (ObserveAtomics) {
        return TraceAtomicLoad<Ops>(
            *trace, *result, task_id, site, address, result_used
        );
    } else {
        (void)trace;
        (void)result;
        (void)task_id;
        (void)site;
        (void)result_used;
        return Ops::Load(address);
    }
}

template <typename Ops, bool ObserveAtomics, typename T>
PA_DEVICE T TraceConfiguredAtomicExchange(
    TraceContext *trace, WorkerResult *result, int32_t task_id,
    AtomicSite site, PA_GM volatile T *address, T value,
    bool result_used = false
) {
    if constexpr (ObserveAtomics) {
        return TraceAtomicExchange<Ops>(
            *trace, *result, task_id, site, address, value,
            result_used
        );
    } else {
        (void)trace;
        (void)result;
        (void)task_id;
        (void)site;
        (void)result_used;
        return Ops::Exchange(address, value);
    }
}

template <typename Ops, bool ObserveAtomics>
PA_DEVICE int64_t TraceConfiguredAtomicCompareExchange(
    TraceContext *trace, WorkerResult *result, int32_t task_id,
    AtomicSite site, PA_GM volatile int64_t *address,
    int64_t expected, int64_t desired, bool result_used = true
) {
    if constexpr (ObserveAtomics) {
        return TraceAtomicCompareExchange<Ops>(
            *trace, *result, task_id, site, address,
            expected, desired, result_used
        );
    } else {
        (void)trace;
        (void)result;
        (void)task_id;
        (void)site;
        (void)result_used;
        return Ops::CompareExchange(address, expected, desired);
    }
}

template <typename Ops, bool ObserveAtomics>
PA_DEVICE int64_t TraceConfiguredAtomicFetchMax(
    TraceContext *trace, WorkerResult *result, int32_t task_id,
    AtomicSite site, PA_GM volatile int64_t *address,
    int64_t value, uint64_t &retries, bool result_used = true
) {
    if constexpr (ObserveAtomics) {
        return TraceAtomicFetchMax<Ops>(
            *trace, *result, task_id, site, address, value,
            retries, result_used
        );
    } else {
        (void)trace;
        (void)result;
        (void)task_id;
        (void)site;
        (void)result_used;
        return Ops::FetchMax(address, value, retries);
    }
}

template <bool Profile>
PA_DEVICE void AccumulatePhase(
    WorkerResult &result, ProfilePhase phase, uint64_t start_cycle, uint64_t end_cycle
) {
#if PA_BUILD_TRACE_FREE
    (void)result;
    (void)phase;
    (void)start_cycle;
    (void)end_cycle;
    return;
#else
    // phase profile 与完整泳道是两套正交机制：即使关闭 records，Profile=true
    // 仍会累计用户当前关注的 Claim/EfDrain/WaitForSlot/HeapGuard 四段。
    if constexpr (Profile) {
        if (phase != ProfilePhase::Claim && phase != ProfilePhase::EfDrain &&
            phase != ProfilePhase::WaitForSlot && phase != ProfilePhase::HeapGuard) {
            return;
        }
        const uint32_t index = static_cast<uint32_t>(phase);
        // 调用方保证 end_cycle>=start_cycle；各后端把 Now() 归一到每 tick 1 ns 的
        // 数值标度，聚合持续时间可直接相加并在 host 侧按 1000 换算为微秒。
        const uint64_t duration = end_cycle - start_cycle;
        result.phase_cycles[index] += duration;
        ++result.phase_calls[index];
    }
#endif
}

#if PTO_FDWIC_SHARED_MAP
template <bool Profile>
PA_DEVICE void WriteSharedClaimTrace(
    TraceContext &trace, WorkerResult &result, uint32_t task_id,
    uint64_t start_cycle, uint64_t end_cycle, bool winner
) {
    AccumulatePhase<Profile>(
        result, ProfilePhase::Claim, start_cycle, end_cycle
    );
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)task_id;
    (void)winner;
#else
    if (!TraceStorageAttached(trace)) {
        return;
    }
#if !PA_BUILD_ATOMIC_SWIMLANE
    if (task_id >= kMaxTasks) {
        return;
    }
    if (start_cycle == 0 || end_cycle < start_cycle ||
        (end_cycle & kSharedClaimWinnerBit) != 0) {
        trace.atomic_counter_overflow = true;
        return;
    }
#endif
    PA_GM SharedSubmitClaimTraceRecord &record =
        GetSharedSubmitClaimRecords(trace.records)[task_id];
    record.claim_begin = start_cycle;
    record.claim_end_and_winner =
        end_cycle | (winner ? kSharedClaimWinnerBit : 0U);
#endif
}

template <bool Profile>
PA_DEVICE void WriteSharedSubmitTrace(
    TraceContext &trace, WorkerResult &result, uint32_t task_id,
    uint64_t start_cycle, uint64_t end_cycle
) {
    AccumulatePhase<Profile>(
        result, ProfilePhase::Submit, start_cycle, end_cycle
    );
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)task_id;
#else
    if (!TraceStorageAttached(trace)) {
        return;
    }
#if !PA_BUILD_ATOMIC_SWIMLANE
    if (task_id >= kMaxTasks) {
        return;
    }
#endif
    PA_GM SharedSubmitClaimTraceRecord &record =
        GetSharedSubmitClaimRecords(trace.records)[task_id];
#if !PA_BUILD_ATOMIC_SWIMLANE
    if (start_cycle == 0 || end_cycle < start_cycle ||
        (end_cycle & kSharedClaimWinnerBit) != 0) {
        trace.atomic_counter_overflow = true;
        return;
    }
#endif
    record.submit_begin = start_cycle;
    record.submit_end = end_cycle;
#endif
}
#endif

template <bool Profile>
PA_DEVICE void WriteTrace(
    TraceContext &trace, WorkerResult &result, int32_t task_id, int32_t function_id, TracePhase trace_phase,
    ProfilePhase profile_phase, uint64_t start_cycle, uint64_t end_cycle, uint32_t flags,
    uint32_t auxiliary
) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)result;
    (void)task_id;
    (void)function_id;
    (void)trace_phase;
    (void)profile_phase;
    (void)start_cycle;
    (void)end_cycle;
    (void)flags;
    (void)auxiliary;
    return;
#else
    // 每段先更新轻量 phase 统计，再按需写 32-byte 原始记录。一个分区只有对应
    // worker 写入；count/dropped 只更新 TraceContext 本地副本，不增加 GM
    // 计数行写入或 atomic。
    AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
    if (!TraceStorageAttached(trace)) {
        return;
    }
#if PA_BUILD_COMPACT_GENERIC_TRACE
    if (__builtin_expect(
            !TraceCompactFieldsFit(
                task_id, function_id, trace_phase, auxiliary
            ),
            0
        )) {
        if (trace.dropped_records != UINT32_MAX) {
            ++trace.dropped_records;
        }
        return;
    }
#endif
    uint32_t slot = 0;
    if (!ReserveTraceRecord(trace, slot)) {
        return;
    }
    WriteGenericTraceRecordRaw(
        trace.records, slot, start_cycle, end_cycle,
        task_id, function_id, trace_phase, flags, auxiliary
    );
#endif
}

template <typename Ops>
PA_DEVICE void ResetTraceLap(
    TraceContext &trace, WorkerResult &result, PA_GM WorkerState &worker
) {
    // lap 是后续 Build/Replay/Alloc 等覆盖式阶段的共同起点，不代表新增嵌套 span。
    // 因此分析时不能把 lap 时长再与其中的 Materialize/Claim/Register 直接相加。
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)result;
    (void)worker;
#else
    (void)result;
    const uint64_t cycle = Ops::Now();
    // 与真实 FDWIC 的 TRACE_LAP_RESET 保持同一边界：等待区 PollBatch
    // 只能覆盖本次逻辑轮询 episode，不能跨进下一段 lap 或计算单元执行。
    AtomicPollBoundaryAt<Ops>(trace, cycle, &result);
    worker.swimlane_last_cycle = cycle;
#endif
}

template <typename Ops>
PA_DEVICE void FlushTraceCore(TraceContext &trace, WorkerResult &result) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)result;
    return;
#else
    if (!TraceStorageAttached(trace)) {
        return;
    }
    // 防御性关闭任何尚未由显式 region end 关闭的等待包；正常路径上 active_mask
    // 应为 0，这里仍保证异常早退不会留下“有逻辑调用、无物理 batch”的半截采集。
    AtomicPollBoundary<Ops>(trace, result);
    PA_GM TraceCoreState &core = *trace.core;
    if (trace.atomic_counter_overflow || result.atomic_trace_calls > UINT32_MAX ||
        trace.poll_calls > UINT32_MAX || trace.poll_batch_records > UINT32_MAX) {
        if (trace.dropped_records != UINT32_MAX) {
            ++trace.dropped_records;
        }
    }
    // terminal row 必须先进入最终 count，随后各记录区 clean 的
    // line_count 才能精确覆盖它自身。shared 的 Submit/Claim 专用区、
    // generic 区和 core-state 各执行一次 FlushRegion，但只生成一条
    // 聚合记录，避免 observer 递归观察自己。
    const uint64_t terminal_begin = Ops::Now();
    const uint64_t final_record_count =
        static_cast<uint64_t>(trace.record_count) + 1U;
    const uint64_t record_bytes =
        final_record_count * sizeof(TraceStorageRecord);
    const uint64_t record_lines =
        (record_bytes + 63U) / 64U;
#if PTO_FDWIC_SHARED_MAP
    const bool submit_claim_window_valid =
        result.submit_begin != 0 &&
        result.submit_end >= result.submit_begin &&
        (result.submit_end & kSharedClaimWinnerBit) == 0;
    if (result.submits > kMaxTasks ||
        !submit_claim_window_valid) {
        if (trace.dropped_records != UINT32_MAX) {
            ++trace.dropped_records;
        }
    }
    const uint64_t submit_claim_count =
        result.submits <= kMaxTasks
            ? result.submits
            : kMaxTasks;
    const uint64_t submit_claim_bytes =
        submit_claim_count *
        sizeof(SharedSubmitClaimTraceRecord);
    const uint64_t submit_claim_lines =
        (submit_claim_bytes + 63U) / 64U;
    constexpr uint32_t kObserverCallCount = 3;
#else
    const uint64_t submit_claim_lines = 0;
    constexpr uint32_t kObserverCallCount = 2;
#endif
    int32_t terminal_slot = -1;
    if (record_lines + submit_claim_lines + 1U <=
        kDcciLineCountMax) {
        terminal_slot = AppendDcciTrace(
            trace, -1, -1, DcciSite::ObserverTraceExport,
            DcciOp::CleanOut, true,
            kObserverCallCount,
            static_cast<uint32_t>(
                record_lines + submit_claim_lines + 1U
            ),
            terminal_begin, terminal_begin
        );
    } else {
        trace.dcci_counter_overflow = true;
    }
    if (trace.dcci_counter_overflow ||
        trace.dcci_calls > UINT32_MAX ||
        trace.dcci_lines > UINT32_MAX ||
        trace.dcci_records > UINT32_MAX) {
        if (trace.dropped_records != UINT32_MAX) {
            ++trace.dropped_records;
        }
    }
    // 正常运行中第一次、也是唯一一次写这条 worker 私有 core-state 行。
    core.count = trace.record_count;
    core.dropped = trace.dropped_records;
    core.atomic_calls =
        static_cast<uint32_t>(result.atomic_trace_calls);
    core.poll_calls = static_cast<uint32_t>(trace.poll_calls);
    core.poll_batch_records =
        static_cast<uint32_t>(trace.poll_batch_records);
    core.core_idx = trace.core_idx;
    core.block_id = trace.block_id;
    core.lane = trace.lane;
    core.dcci_calls = static_cast<uint32_t>(trace.dcci_calls);
    core.dcci_lines = static_cast<uint32_t>(trace.dcci_lines);
    core.dcci_records = static_cast<uint32_t>(trace.dcci_records);
    const uint32_t count =
        trace.record_count < trace.capacity
            ? trace.record_count
            : trace.capacity;
    // A5 侧记录经普通 GM cache 写入，kernel 结束前必须把有效 records 与最后的
    // count/dropped cache line 显式 clean，host 的 D2H 才能得到完整且自洽的快照。
    if (count != 0) {
        Ops::FlushRegion(
            trace.records,
            static_cast<uint64_t>(count) *
                sizeof(TraceStorageRecord)
        );
    }
#if PTO_FDWIC_SHARED_MAP
    if (submit_claim_count != 0) {
        Ops::FlushRegion(
            GetSharedSubmitClaimRecords(trace.records),
            submit_claim_bytes
        );
    }
#endif
    Ops::FlushRegion(&core, sizeof(core));
    if (terminal_slot >= 0) {
        // terminal row 本身已经随 generic records clean 导出；所有
        // observer DSB 之后只能用一次 bypass-DCache store 更新 end，
        // 不能再对同一 cache line 做 read-modify-write。compact 只发布
        // 4B low32，保证同一 64B 内其余三条记录保持原值。
#if PA_BUILD_COMPACT_GENERIC_TRACE
        Ops::Publish(
            &trace.records[
                static_cast<uint32_t>(terminal_slot)
            ].end_cycle_low,
            static_cast<uint32_t>(Ops::Now())
        );
#else
        Ops::Publish(
            &trace.records[
                static_cast<uint32_t>(terminal_slot)
            ].end_cycle,
            Ops::Now()
        );
#endif
    }
#endif
}

template <typename Ops, bool Profile>
PA_DEVICE uint64_t WriteTraceLap(
    TraceContext &trace, PA_GM WorkerState &worker, WorkerResult &result, int32_t task_id,
    int32_t function_id, TracePhase trace_phase, ProfilePhase profile_phase,
    uint32_t flags = 0, uint32_t auxiliary = 0
) {
#if PA_BUILD_TRACE_FREE
    (void)trace;
    (void)worker;
    (void)result;
    (void)task_id;
    (void)function_id;
    (void)trace_phase;
    (void)profile_phase;
    (void)flags;
    (void)auxiliary;
    return 0;
#else
    // lap 记录区间 [上一次 Reset/WriteTraceLap, 当前时刻]，写完立即推进起点。
    // 显式 WriteTrace span 不会修改该起点，这正是生产泳道中阶段可重叠的原因。
    const uint64_t end_cycle = Ops::Now();
    // 真实 FDWIC 在 TRACE_LAP 取到结束时间后先关闭 PollBatch，再写 lap。
    // 复用同一个 end_cycle，避免额外 SYS_CNT 造成可见缝隙。
    AtomicPollBoundaryAt<Ops>(trace, end_cycle, &result);
    WriteTrace<Profile>(
        trace, result, task_id, function_id, trace_phase, profile_phase, worker.swimlane_last_cycle, end_cycle,
        flags, auxiliary
    );
    worker.swimlane_last_cycle = end_cycle;
    return end_cycle;
#endif
}

}  // namespace pa_scheduler

#endif  // PA_SCHEDULER_COMMON_PA_TRACE_H
