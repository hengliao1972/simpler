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

namespace pa_scheduler {

// 记录区与真实 PA 一样直接拼在定长 Header 后面；worker_id 只选择自己的
// records 分区，避免记录动作本身制造跨核共享写热点。
PA_DEVICE PA_GM TraceRecord *GetTraceRecords(PA_GM TraceHeader *header) {
    // 输入必须指向完整且按 64 byte 对齐的 trace buffer；返回值只是首条记录，
    // 调用方还需按 worker_id * capacity 选择自己的分区。
    return reinterpret_cast<PA_GM TraceRecord *>(reinterpret_cast<PA_GM uint8_t *>(header) + sizeof(TraceHeader));
}

struct TraceContext {
    PA_GM TraceCoreState *core;
    PA_GM TraceRecord *records;
    uint32_t capacity;
    int32_t lane;
    int32_t block_id;
    int32_t core_idx;
};

// Attach 只缓存本 worker 的 header 状态、分区首址和物理 lane 信息。配置先做
// cache invalidate，确保 A5 worker 看到 host 在 launch 前写入的 trace 开关与地址。
template <typename Ops>
PA_DEVICE TraceContext AttachTrace(
    PA_GM SchedulerState *state, PA_GM const WorkerState &worker, uint32_t worker_id
) {
    TraceContext trace{nullptr, nullptr, 0, worker.lane, worker.block_id, static_cast<int32_t>(worker_id)};
    Ops::InvalidateRegion(&state->config, sizeof(state->config));
    const uint64_t base = state->config.trace_base;
    const uint32_t capacity = state->config.trace_records_per_core;
    if (state->config.trace_enabled == 0 || base == 0 || capacity == 0 || worker_id >= kWorkers) {
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
    trace.core->count = 0;
    trace.core->dropped = 0;
    return trace;
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
    ProfilePhase profile_phase, uint64_t start_cycle, uint64_t end_cycle, uint32_t flags = 0,
    uint32_t auxiliary = 0
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
PA_DEVICE void ResetTraceLap(PA_GM WorkerState &worker) {
    // lap 是后续 Build/Replay/Alloc 等覆盖式阶段的共同起点，不代表新增嵌套 span。
    // 因此分析时不能把 lap 时长再与其中的 Materialize/Claim/Register 直接相加。
    worker.swimlane_last_cycle = Ops::Now();
}

template <typename Ops>
PA_DEVICE void FlushTraceCore(TraceContext &trace) {
    if (trace.core == nullptr || trace.records == nullptr || trace.capacity == 0) {
        return;
    }
    PA_GM TraceCoreState &core = *trace.core;
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
    WriteTrace<Profile>(
        trace, result, task_id, function_id, trace_phase, profile_phase, worker.swimlane_last_cycle, end_cycle,
        flags, auxiliary
    );
    worker.swimlane_last_cycle = end_cycle;
    return end_cycle;
}

}  // namespace pa_scheduler

#endif  // PA_SCHEDULER_COMMON_PA_TRACE_H
