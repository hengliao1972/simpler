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
PA_DEVICE PA_GM TraceRecord *GetTraceRecords(PA_GM TraceHeader *header) {
    // 输入必须指向完整且按 64 byte 对齐的 trace buffer；返回值只是首条记录，
    // 调用方还需按 worker_id * capacity 选择自己的分区。
    return reinterpret_cast<PA_GM TraceRecord *>(reinterpret_cast<PA_GM uint8_t *>(header) + sizeof(TraceHeader));
}

struct AtomicPollBurst {
    uint64_t start_cycle[kAtomicPollBatchSiteCount];
    uint32_t call_count[kAtomicPollBatchSiteCount];
    uint32_t active_mask;
    uint32_t enabled_mask;
};

struct TraceContext {
    PA_GM TraceCoreState *core;
    PA_GM TraceRecord *records;
    uint32_t capacity;
    bool atomics_enabled;
    int32_t lane;
    int32_t block_id;
    int32_t core_idx;
    // 轮询调用数留在 worker 私有上下文，最终一次性发布到 core state；
    // 等待热路不为计数再写共享/GM 状态。
    uint64_t poll_calls;
    uint64_t poll_batch_records;
    bool atomic_counter_overflow;
    AtomicPollBurst poll_burst;
};

// pa_model.h 也向 host 暴露同一 raw ABI 映射，但 CCEC/AscendC 的单个 TU
// 会先以 host 语境包含该头，再实例化 device 调度器。这里保留明确的 device
// 版本，避免设备函数误调用先前已实例化的 __host__ helper。
PA_DEVICE AtomicOp TraceAtomicSiteExpectedOp(AtomicSite site) {
    switch (site) {
        case AtomicSite::StartupIncrement:
        case AtomicSite::ReplayDoneIncrement:
            return AtomicOp::FetchAdd;
        case AtomicSite::FatalSet:
        case AtomicSite::CompletionVendExchange:
        case AtomicSite::CompletionFlagExchange:
            return AtomicOp::Exchange;
        case AtomicSite::ClaimMax:
        case AtomicSite::FrontierMax:
            return AtomicOp::FetchMax;
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
    return TraceAtomicPollBatchIndex(site) >= 0;
}

PA_DEVICE uint32_t TraceAtomicSiteMask(AtomicSite site) {
    return 1U << static_cast<uint32_t>(site);
}

// Attach 只缓存本 worker 的 header 状态、分区首址和物理 lane 信息。配置先做
// cache invalidate，确保 A5 worker 看到 host 在 launch 前写入的 trace 开关、地址与 winner 负载配置。
template <typename Ops>
PA_DEVICE TraceContext AttachTrace(
    PA_GM SchedulerState *state, PA_GM const WorkerState &worker, uint32_t worker_id
) {
    TraceContext trace{};
    trace.lane = worker.lane;
    trace.block_id = worker.block_id;
    trace.core_idx = static_cast<int32_t>(worker_id);
    Ops::InvalidateRegion(
        &state->config, sizeof(state->config) + sizeof(state->winner_workload)
    );
    const uint64_t base = state->config.trace_base;
    const uint32_t capacity = state->config.trace_records_per_core;
    if ((state->config.trace_enabled & kTracePhasesEnabled) == 0 || base == 0 || capacity == 0 ||
        worker_id >= kWorkers) {
        return trace;
    }
    PA_GM TraceHeader *header = reinterpret_cast<PA_GM TraceHeader *>(base);
    if (worker_id >= header->num_cores) {
        return trace;
    }
    trace.core = &header->cores[worker_id];
    trace.records = &GetTraceRecords(header)[static_cast<uint64_t>(worker_id) * capacity];
    // 成功返回的不变量是 core/records/capacity 同时有效；任一前置条件失败则三者
    // 保持空值，后续 WriteTrace/FlushTraceCore 可无分支地安全退化为 no-op。
    trace.capacity = capacity;
    trace.atomics_enabled = (state->config.trace_enabled & kTraceAtomicsEnabled) != 0;
    trace.core->count = 0;
    trace.core->dropped = 0;
    trace.core->atomic_calls = 0;
    trace.core->poll_calls = 0;
    trace.core->poll_batch_records = 0;
    trace.core->core_idx = trace.core_idx;
    trace.core->block_id = trace.block_id;
    trace.core->lane = trace.lane;
    return trace;
}

template <bool Profile>
PA_DEVICE void WriteTrace(
    TraceContext &trace, WorkerResult &result, int32_t task_id, int32_t function_id,
    TracePhase trace_phase, ProfilePhase profile_phase, uint64_t start_cycle, uint64_t end_cycle,
    uint32_t flags = 0, uint32_t auxiliary = 0
);

// CCEC 不让栈上的 TraceContext/WorkerResult 引用跨非内联调用。这里仅把
// PollBatch 固定形状的 64-byte GM 写入抽成共享函数，以抑制各 phase 边界
// 内联后的代码膨胀；参数只有 GM 指针与标量，局部 batch 状态仍由调用者维护。
PA_DEVICE_NOINLINE bool WritePollBatchRecordRaw(
    PA_GM TraceCoreState *core, PA_GM TraceRecord *records, uint32_t capacity,
    uint64_t start_cycle, uint64_t end_cycle, uint32_t call_count, uint32_t site_id
) {
    if (core == nullptr || records == nullptr || capacity == 0) {
        return false;
    }
    const uint32_t slot = core->count;
    if (slot >= capacity) {
        core->dropped = core->dropped + 1;
        return false;
    }
    PA_GM TraceRecord &record = records[slot];
    record.start_cycle = start_cycle;
    record.end_cycle = end_cycle;
    record.task_id = -1;
    record.function_id = -1;
    record.phase = static_cast<int32_t>(TracePhase::Atomic);
    record.lane = core->lane;
    record.block_id = core->block_id;
    record.core_idx = core->core_idx;
    const AtomicSite site = static_cast<AtomicSite>(site_id);
    record.flags = static_cast<uint32_t>(TraceAtomicSiteExpectedOp(site)) |
                   kAtomicResultUsed | kAtomicPollBatch |
                   (call_count << kAtomicPollCountShift);
    record.auxiliary = site_id;
    // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
    core->count = slot + 1;
    return true;
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
    if (result.atomic_trace_calls == UINT64_MAX) {
        trace.atomic_counter_overflow = true;
        return;
    }
    ++result.atomic_trace_calls;
    if (!poll_batch) return;
    if (trace.poll_calls == UINT64_MAX) {
        trace.atomic_counter_overflow = true;
        return;
    }
    ++trace.poll_calls;
}

template <typename Ops>
PA_DEVICE void AtomicPollBoundaryAt(
    TraceContext &trace, uint64_t end_cycle
) {
    if (!trace.atomics_enabled || trace.poll_burst.active_mask == 0) return;
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
        const AtomicSite site = TraceAtomicPollBatchSite(index);
        const bool written = WritePollBatchRecordRaw(
            trace.core, trace.records, trace.capacity,
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
}

template <typename Ops>
PA_DEVICE void AtomicPollBoundary(TraceContext &trace, WorkerResult &result) {
    (void)result;
    if (trace.poll_burst.active_mask == 0) return;
    AtomicPollBoundaryAt<Ops>(trace, Ops::Now());
}

template <typename Ops>
PA_DEVICE uint32_t AtomicPollRegionBegin(
    TraceContext &trace, WorkerResult &result, uint32_t site_mask
) {
    const uint32_t previous_mask = trace.poll_burst.enabled_mask;
    if (!trace.atomics_enabled) return previous_mask;
    AtomicPollBoundary<Ops>(trace, result);
    trace.poll_burst.enabled_mask = previous_mask | site_mask;
    return previous_mask;
}

template <typename Ops>
PA_DEVICE void AtomicPollRegionEnd(
    TraceContext &trace, WorkerResult &result, uint32_t previous_mask
) {
    if (!trace.atomics_enabled) return;
    AtomicPollBoundary<Ops>(trace, result);
    trace.poll_burst.enabled_mask = previous_mask;
}

PA_DEVICE bool AtomicPollBatchEnabled(
    TraceContext &trace, AtomicSite site, AtomicOp actual_op
) {
    return trace.atomics_enabled && TraceAtomicSiteIsPollBatchable(site) &&
           TraceAtomicSiteExpectedOp(site) == actual_op &&
           (trace.poll_burst.enabled_mask & TraceAtomicSiteMask(site)) != 0;
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
    if ((trace.poll_burst.active_mask & bit) == 0) {
        trace.poll_burst.start_cycle[index] = start_cycle;
        trace.poll_burst.call_count[index] = 0;
        trace.poll_burst.active_mask |= bit;
    }
    uint32_t &call_count = trace.poll_burst.call_count[index];
    ++call_count;
    if (call_count == kAtomicPollCountMax) {
        AtomicPollBoundary<Ops>(trace, result);
    }
}

template <typename Ops>
PA_DEVICE void WriteAtomicTrace(
    TraceContext &trace, WorkerResult &result, int32_t task_id, AtomicSite site, AtomicOp op,
    uint64_t start_cycle, uint64_t end_cycle, bool result_used, bool return_ready,
    bool value_zero = false, uint64_t retries = 0
) {
    // 一次源码 atomic 只写一条同时含 start/end 的 span；结束时间先于 64B record
    // 写入，因此本条区间不直接包含自己的记录写开销，但下一次竞争到达会受它影响。
    CountAtomicCall(trace, result, false);
    WriteTrace<false>(
        trace, result, task_id, -1, TracePhase::Atomic, ProfilePhase::ReplayTail,
        start_cycle, end_cycle,
        AtomicTraceFlags(op, result_used, return_ready, value_zero, retries),
        static_cast<uint32_t>(site)
    );
}

template <typename Ops, typename T>
PA_DEVICE T TraceAtomicLoad(
    TraceContext &trace, WorkerResult &result, int32_t task_id, AtomicSite site,
    PA_GM volatile T *address, bool result_used = true
) {
    if (!trace.atomics_enabled) return Ops::Load(address);
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
}

template <typename Ops, typename T>
PA_DEVICE T TraceAtomicExchange(
    TraceContext &trace, WorkerResult &result, int32_t task_id, AtomicSite site,
    PA_GM volatile T *address, T value, bool result_used = false
) {
    if (!trace.atomics_enabled) return Ops::Exchange(address, value);
    const uint64_t begin = Ops::Now();
    const T old = Ops::Exchange(address, value);
    const bool return_ready = result_used && Ops::kAtomicReturnReadyObserved;
    const uint64_t end = result_used ? Ops::NowAfterAtomicResult(old) : Ops::Now();
    WriteAtomicTrace<Ops>(
        trace, result, task_id, site, AtomicOp::Exchange, begin, end, result_used, return_ready
    );
    return old;
}

template <typename Ops>
PA_DEVICE int64_t TraceAtomicFetchAdd(
    TraceContext &trace, WorkerResult &result, int32_t task_id, AtomicSite site,
    PA_GM volatile int64_t *address, int64_t value, bool result_used = false
) {
    if (!trace.atomics_enabled) return Ops::FetchAdd(address, value);
    const uint64_t begin = Ops::Now();
    const int64_t old = Ops::FetchAdd(address, value);
    const bool return_ready = result_used && Ops::kAtomicReturnReadyObserved;
    const uint64_t end = result_used ? Ops::NowAfterAtomicResult(old) : Ops::Now();
    WriteAtomicTrace<Ops>(
        trace, result, task_id, site, AtomicOp::FetchAdd, begin, end, result_used, return_ready
    );
    return old;
}

template <typename Ops>
PA_DEVICE int64_t TraceAtomicFetchMax(
    TraceContext &trace, WorkerResult &result, int32_t task_id, AtomicSite site,
    PA_GM volatile int64_t *address, int64_t value, uint64_t &retries, bool result_used = true
) {
    if (!trace.atomics_enabled) return Ops::FetchMax(address, value, retries);
    const uint64_t begin = Ops::Now();
    const int64_t old = Ops::FetchMax(address, value, retries);
    const bool return_ready = result_used && Ops::kAtomicReturnReadyObserved;
    const uint64_t end = result_used ? Ops::NowAfterAtomicResult(old) : Ops::Now();
    WriteAtomicTrace<Ops>(
        trace, result, task_id, site, AtomicOp::FetchMax, begin, end, result_used,
        return_ready, false, retries
    );
    return old;
}

template <bool Profile>
PA_DEVICE void AccumulatePhase(
    WorkerResult &result, ProfilePhase phase, uint64_t start_cycle, uint64_t end_cycle
) {
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
}

template <bool Profile>
PA_DEVICE void WriteTrace(
    TraceContext &trace, WorkerResult &result, int32_t task_id, int32_t function_id, TracePhase trace_phase,
    ProfilePhase profile_phase, uint64_t start_cycle, uint64_t end_cycle, uint32_t flags,
    uint32_t auxiliary
) {
    // 每段先更新轻量 phase 统计，再按需写 64-byte 原始记录。一个分区只有对应
    // worker 写入，因此 count/dropped 保持普通单写者更新，不额外引入 atomic。
    AccumulatePhase<Profile>(result, profile_phase, start_cycle, end_cycle);
    if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
        return;
    }
    PA_GM TraceCoreState &core = *trace.core;
    const uint32_t slot = core.count;
    if (slot >= trace.capacity) {
        core.dropped = core.dropped + 1;
        return;
    }
    PA_GM TraceRecord &record = trace.records[slot];
    record.start_cycle = start_cycle;
    record.end_cycle = end_cycle;
    record.task_id = task_id;
    record.function_id = function_id;
    record.phase = static_cast<int32_t>(trace_phase);
    record.lane = trace.lane;
    record.block_id = trace.block_id;
    record.core_idx = trace.core_idx;
    record.flags = flags;
    record.auxiliary = auxiliary;
    // count 最后更新，使其始终指向下一空槽；单写者条件下无需 reserve/commit 两阶段。
    // 最终 FlushTraceCore 会把记录体先于该计数一并导出。
    core.count = slot + 1;
}

template <typename Ops>
PA_DEVICE void ResetTraceLap(
    TraceContext &trace, WorkerResult &result, PA_GM WorkerState &worker
) {
    // lap 是后续 Build/Replay/Alloc 等覆盖式阶段的共同起点，不代表新增嵌套 span。
    // 因此分析时不能把 lap 时长再与其中的 Materialize/Claim/Register 直接相加。
    (void)result;
    const uint64_t cycle = Ops::Now();
    // 与真实 FDWIC 的 TRACE_LAP_RESET 保持同一边界：等待区 PollBatch
    // 只能覆盖本次逻辑轮询 episode，不能跨进下一段 lap 或计算单元执行。
    AtomicPollBoundaryAt<Ops>(trace, cycle);
    worker.swimlane_last_cycle = cycle;
}

template <typename Ops>
PA_DEVICE void FlushTraceCore(TraceContext &trace, WorkerResult &result) {
    if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
        return;
    }
    // 防御性关闭任何尚未由显式 region end 关闭的等待包；正常路径上 active_mask
    // 应为 0，这里仍保证异常早退不会留下“有逻辑调用、无物理 batch”的半截采集。
    AtomicPollBoundary<Ops>(trace, result);
    PA_GM TraceCoreState &core = *trace.core;
    if (trace.atomic_counter_overflow || result.atomic_trace_calls > UINT32_MAX ||
        trace.poll_calls > UINT32_MAX || trace.poll_batch_records > UINT32_MAX) {
        if (core.dropped != UINT32_MAX) core.dropped = core.dropped + 1;
    }
    core.atomic_calls = static_cast<uint32_t>(result.atomic_trace_calls);
    core.poll_calls = static_cast<uint32_t>(trace.poll_calls);
    core.poll_batch_records = static_cast<uint32_t>(trace.poll_batch_records);
    const uint32_t count = core.count < trace.capacity ? core.count : trace.capacity;
    // A5 侧记录经普通 GM cache 写入，kernel 结束前必须把有效 records 与最后的
    // count/dropped cache line 显式 clean，host 的 D2H 才能得到完整且自洽的快照。
    if (count != 0) {
        Ops::FlushRegion(trace.records, static_cast<uint64_t>(count) * sizeof(TraceRecord));
    }
    Ops::FlushRegion(&core, sizeof(core));
}

template <typename Ops, bool Profile>
PA_DEVICE uint64_t WriteTraceLap(
    TraceContext &trace, PA_GM WorkerState &worker, WorkerResult &result, int32_t task_id,
    int32_t function_id, TracePhase trace_phase, ProfilePhase profile_phase,
    uint32_t flags = 0, uint32_t auxiliary = 0
) {
    // lap 记录区间 [上一次 Reset/WriteTraceLap, 当前时刻]，写完立即推进起点。
    // 显式 WriteTrace span 不会修改该起点，这正是生产泳道中阶段可重叠的原因。
    const uint64_t end_cycle = Ops::Now();
    // 真实 FDWIC 在 TRACE_LAP 取到结束时间后先关闭 PollBatch，再写 lap。
    // 复用同一个 end_cycle，避免额外 SYS_CNT 造成可见缝隙。
    AtomicPollBoundaryAt<Ops>(trace, end_cycle);
    WriteTrace<Profile>(
        trace, result, task_id, function_id, trace_phase, profile_phase, worker.swimlane_last_cycle, end_cycle,
        flags, auxiliary
    );
    worker.swimlane_last_cycle = end_cycle;
    return end_cycle;
}

}  // namespace pa_scheduler

#endif  // PA_SCHEDULER_COMMON_PA_TRACE_H
