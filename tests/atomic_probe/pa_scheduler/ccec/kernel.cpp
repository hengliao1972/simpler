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

#include "cce_aicore_intrinsics.h"
#include <pto/common/constants.hpp>
#include <pto/common/kernel_meta.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "pmu_probe.h"
#include "../common/winner_workload.h"

#define PA_DEVICE __aicore__ inline
#define PA_DEVICE_NOINLINE static __aicore__ __attribute__((noinline))
#define PA_LOOP_NOUNROLL _Pragma("clang loop unroll(disable)")
#define PA_GM __gm__
#include "../common/pa_scheduler_core.h"

#define PA_CCEC_OPS_DEFINE_REAL_WORKLOAD 1
#include "ccec_ops.h"
#undef PA_CCEC_OPS_DEFINE_REAL_WORKLOAD

using pa_scheduler_ccec::CcecOps;

namespace {
#if PA_BUILD_SUBMIT_PMU
struct PmuSnapshot {
    uint64_t total_cycles = 0;
    uint32_t vector_busy = 0;
    uint32_t cube_busy = 0;
    uint32_t scalar_busy = 0;
    uint32_t mte1_busy = 0;
    uint32_t mte2_busy = 0;
    uint32_t mte3_busy = 0;
    uint32_t icache_requests = 0;
    uint32_t icache_misses = 0;
    uint32_t fix_busy = 0;
    uint32_t status = 0;
};

struct IcacheShadowSnapshot {
    uint32_t requests = 0;
    uint32_t misses = 0;
};

}  // namespace

namespace pa_scheduler_ccec {

struct SubmitPmuContext {
    uint64_t reg_base = 0;
    uint64_t shadow_requests = 0;
    uint64_t shadow_misses = 0;
    uint64_t phase_requests = 0;
    uint64_t phase_misses = 0;
    // begin 的 shadow read-clear 完成后取起点，end 的 shadow read-clear 之前
    // 取终点；累计值因此不包含两次 PMU 寄存器读取本身。
    uint64_t phase_elapsed_ticks = 0;
    uint64_t phase_begin_tick = 0;
    uint32_t selector_status = 0;
    uint32_t phase_status = pa_scheduler::ccec_pmu::kPhaseStatusRequested;
    uint32_t phase_calls = 0;
    uint32_t begin_reads = 0;
    uint32_t end_reads = 0;
    bool started = false;
    bool phase_armed = false;
    bool boundary_error = false;
};

}  // namespace pa_scheduler_ccec

namespace {

using pa_scheduler_ccec::SubmitPmuContext;

template <uint32_t BlockOffset, uint32_t RegisterOffset>
__aicore__ inline uint32_t ReadPmuRegister(uint64_t reg_base) {
    // 传给 ld_dev 的是重基址后的 __gm__ 指针；相对 offset 均落在编译器允许的 [-2048, 2047]。
    int32_t *block = reinterpret_cast<int32_t *>(reg_base + BlockOffset);
    return static_cast<uint32_t>(ld_dev(block, static_cast<int16_t>(RegisterOffset - BlockOffset)));
}

__aicore__ inline PmuSnapshot ReadObservedCounters(uint64_t reg_base) {
    PmuSnapshot sample;
    sample.vector_busy = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                         pa_scheduler::ccec_pmu::kCnt0Offset>(reg_base);
    sample.cube_busy = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                       pa_scheduler::ccec_pmu::kCnt1Offset>(reg_base);
    sample.scalar_busy = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                         pa_scheduler::ccec_pmu::kCnt2Offset>(reg_base);
    sample.mte1_busy = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                       pa_scheduler::ccec_pmu::kCnt3Offset>(reg_base);
    sample.mte2_busy = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                       pa_scheduler::ccec_pmu::kCnt4Offset>(reg_base);
    // submit-pmu 将 CNT5 留给 shadow I-cache miss。这里不能提前读取，否则
    // read-to-clear 会让随后的 shadow tail 漏计；MTE3 busy 在该诊断构建不可用。
    sample.mte3_busy = 0;
    sample.icache_requests = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                             pa_scheduler::ccec_pmu::kCnt6Offset>(reg_base);
    sample.icache_misses = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                           pa_scheduler::ccec_pmu::kCnt7Offset>(reg_base);
    const uint64_t low = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                         pa_scheduler::ccec_pmu::kTotalLowOffset>(reg_base);
    const uint64_t high = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                          pa_scheduler::ccec_pmu::kTotalHighOffset>(reg_base);
    sample.total_cycles = low | (high << 32);
    return sample;
}

__aicore__ inline IcacheShadowSnapshot ReadShadowCounters(uint64_t reg_base) {
    IcacheShadowSnapshot sample;
    sample.requests = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                      pa_scheduler::ccec_pmu::kCnt8Offset>(reg_base);
    sample.misses = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                    pa_scheduler::ccec_pmu::kCnt5Offset>(reg_base);
    return sample;
}

struct PmuRegisterContext {
    uint64_t reg_base = 0;
    uint32_t status = 0;
    bool shadow_selectors = false;
};

__aicore__ inline PmuRegisterContext ResolvePmuRegisters(__gm__ pa_scheduler::SchedulerState *state) {
    using namespace pa_scheduler::ccec_pmu;
    PmuRegisterContext context;
    const uint32_t physical_core_id = static_cast<uint32_t>(get_coreid()) & kStatusCoreIdMask;
    context.status = kStatusRequested | (physical_core_id << kStatusCoreIdShift);
    const uint64_t table_address = state->pmu_probe.register_table;
    if (state->pmu_probe.magic != kConfigMagicValue || table_address == 0 ||
        physical_core_id >= kPhysicalSubcoreCount) {
        return context;
    }
    context.status |= kStatusCoreIdValid;
    __gm__ const uint64_t *register_bases = reinterpret_cast<__gm__ const uint64_t *>(table_address);
    context.reg_base = register_bases[physical_core_id];
    if (context.reg_base == 0) return context;
    context.status |= kStatusRegMapped;

    // selector 与同 phase 目录内的 owner 逐项核对。CNT5/CNT8 分别重复
    // CNT7/CNT6；CNT9 保持正式 PIPE_UTIL 的 unused(0) 口径。
    if (ReadPmuRegister<kSelectorBlockOffset, kCnt0SelectorOffset>(context.reg_base) == kVectorBusyEvent)
        context.status |= kStatusCnt0Selector;
    if (ReadPmuRegister<kSelectorBlockOffset, kCnt1SelectorOffset>(context.reg_base) == kCubeBusyEvent)
        context.status |= kStatusCnt1Selector;
    if (ReadPmuRegister<kSelectorBlockOffset, kCnt2SelectorOffset>(context.reg_base) == kScalarBusyEvent)
        context.status |= kStatusCnt2Selector;
    if (ReadPmuRegister<kSelectorBlockOffset, kCnt3SelectorOffset>(context.reg_base) == kMte1BusyEvent)
        context.status |= kStatusCnt3Selector;
    if (ReadPmuRegister<kSelectorBlockOffset, kCnt4SelectorOffset>(context.reg_base) == kMte2BusyEvent)
        context.status |= kStatusCnt4Selector;
    const bool cnt5_ok =
        ReadPmuRegister<kSelectorBlockOffset, kCnt5SelectorOffset>(context.reg_base) == kIcacheMissEvent;
    if (cnt5_ok)
        context.status |= kStatusCnt5Selector;
    if (ReadPmuRegister<kSelectorBlockOffset, kCnt6SelectorOffset>(context.reg_base) == kIcacheRequestEvent)
        context.status |= kStatusCnt6Selector;
    if (ReadPmuRegister<kSelectorBlockOffset, kCnt7SelectorOffset>(context.reg_base) == kIcacheMissEvent)
        context.status |= kStatusCnt7Selector;
    const bool cnt8_ok =
        ReadPmuRegister<kSelectorBlockOffset, kCnt8SelectorOffset>(context.reg_base) == kIcacheRequestEvent;
    const bool cnt9_unused =
        ReadPmuRegister<kSelectorBlockOffset, kCnt9SelectorOffset>(context.reg_base) == 0U;
    if (cnt8_ok)
        context.status |= kStatusCnt8Selector;
    context.shadow_selectors = cnt5_ok && cnt8_ok && cnt9_unused;
    return context;
}

__aicore__ inline void PublishPmuSnapshot(
    __gm__ pa_scheduler::WorkerResult &result, const PmuSnapshot &sample
) {
    // 每核独占 sidecar 通过 bypass store 一次性发布；这些写发生在 PMU stop/read 之后，
    // 不进入被导出的 Submit 窗口。
    CcecOps::Publish(&result.pmu_total_cycles, sample.total_cycles);
    CcecOps::Publish(&result.pmu_scalar_busy, sample.scalar_busy);
    CcecOps::Publish(&result.pmu_icache_requests, sample.icache_requests);
    CcecOps::Publish(&result.pmu_icache_misses, sample.icache_misses);
    CcecOps::Publish(&result.pmu_status, sample.status);
    CcecOps::Publish(&result.pmu_vector_busy, sample.vector_busy);
    CcecOps::Publish(&result.pmu_cube_busy, sample.cube_busy);
    CcecOps::Publish(&result.pmu_mte1_busy, sample.mte1_busy);
    CcecOps::Publish(&result.pmu_mte2_busy, sample.mte2_busy);
    CcecOps::Publish(&result.pmu_mte3_busy, sample.mte3_busy);
    // CNT8 已改作 shadow request，submit-pmu 不再发布 fix-busy。
    CcecOps::Publish(&result.pmu_fix_busy, static_cast<uint32_t>(0));
    CcecOps::Publish(&result.pmu_warm_total_cycles, static_cast<uint64_t>(0));
    CcecOps::Publish(&result.pmu_warm_window_ticks, static_cast<uint64_t>(0));
}

__aicore__ inline bool FitsUint32(uint64_t value) {
    return value <= 0xffffffffULL;
}

__aicore__ inline void PublishSubmitPmuContext(
    __gm__ pa_scheduler::WorkerResult &result, const SubmitPmuContext &context
) {
    CcecOps::Publish(&result.pmu_build_variant, pa_scheduler::kBuildVariantSubmitPmu);
    CcecOps::Publish(
        &result.pmu_phase_id,
        static_cast<uint32_t>(pa_scheduler::kCompiledSubmitPmuPhase)
    );
    CcecOps::Publish(&result.pmu_phase_calls, context.phase_calls);
    CcecOps::Publish(&result.pmu_phase_status, context.phase_status);
    CcecOps::Publish(&result.pmu_phase_begin_reads, context.begin_reads);
    CcecOps::Publish(&result.pmu_phase_end_reads, context.end_reads);
    CcecOps::Publish(&result.pmu_phase_elapsed_ticks, context.phase_elapsed_ticks);
    CcecOps::Publish(&result.pmu_phase_icache_requests, static_cast<uint32_t>(context.phase_requests));
    CcecOps::Publish(&result.pmu_phase_icache_misses, static_cast<uint32_t>(context.phase_misses));
    CcecOps::Publish(&result.pmu_shadow_icache_requests, static_cast<uint32_t>(context.shadow_requests));
    CcecOps::Publish(&result.pmu_shadow_icache_misses, static_cast<uint32_t>(context.shadow_misses));
}

}  // namespace

namespace pa_scheduler_ccec {

__aicore__ inline CcecOps::PmuContext CcecOps::PmuWindowStart(
    __gm__ pa_scheduler::SchedulerState *state, uint32_t worker_id
) {
    using namespace pa_scheduler::ccec_pmu;
    (void)worker_id;
    SubmitPmuContext context;
    const WindowMode mode = static_cast<WindowMode>(state->pmu_probe.mode);
    if (mode != WindowMode::SubmitAll) return context;
    // Main AICPU owner 已在 launch 前配置并开启计数；先 stop + snapshot/read-clear，
    // 再解析 selector，避免这些 ld_dev 污染完整 Submit 窗口。
    bisheng::cce::metrics_prof_stop();
    const PmuRegisterContext registers = ResolvePmuRegisters(state);
    context.reg_base = registers.reg_base;
    context.selector_status = registers.status;
    if (registers.shadow_selectors) {
        context.phase_status |= kPhaseStatusShadowSelectors;
    }
    if (context.reg_base == 0) return context;
    (void)ReadObservedCounters(context.reg_base);
    (void)ReadShadowCounters(context.reg_base);
    bisheng::cce::metrics_prof_start();
    context.started = true;
    context.phase_status |= kPhaseStatusWindowStarted;
    return context;
}

__aicore__ inline void CcecOps::PmuPhaseBegin(PmuContext &context) {
    if (!context.started || context.reg_base == 0 || context.phase_armed) {
        context.boundary_error = true;
        return;
    }
    // counter 在运行中读取即清零；begin 之前的片段只进入 shadow whole。
    const IcacheShadowSnapshot sample = ReadShadowCounters(context.reg_base);
    context.shadow_requests += sample.requests;
    context.shadow_misses += sample.misses;
    ++context.begin_reads;
    context.phase_armed = true;
    // get_sys_cnt() 是本机已校准为 1 ns/tick 的 A5 系统计数器。该读取位于
    // begin 的两条 ld_dev 之后，因此不会把 read-clear 成本算进阶段时间。
    context.phase_begin_tick = CcecOps::Now();
}

__aicore__ inline void CcecOps::PmuPhaseEnd(PmuContext &context) {
    // 先取终点再读取 shadow counter，使 end 的两条 ld_dev 同样位于阶段之外。
    const uint64_t phase_end_tick = CcecOps::Now();
    if (!context.started || context.reg_base == 0 || !context.phase_armed) {
        context.boundary_error = true;
        return;
    }
    if (phase_end_tick < context.phase_begin_tick) {
        context.boundary_error = true;
    } else {
        context.phase_elapsed_ticks += phase_end_tick - context.phase_begin_tick;
    }
    // end 读出的片段同时属于完整 shadow 重建与被选中的局部阶段。
    const IcacheShadowSnapshot sample = ReadShadowCounters(context.reg_base);
    context.shadow_requests += sample.requests;
    context.shadow_misses += sample.misses;
    context.phase_requests += sample.requests;
    context.phase_misses += sample.misses;
    ++context.end_reads;
    ++context.phase_calls;
    context.phase_armed = false;
    context.phase_begin_tick = 0;
}

__aicore__ inline void CcecOps::PmuWindowStop(
    __gm__ pa_scheduler::SchedulerState *state, uint32_t worker_id, PmuContext &context
) {
    using namespace pa_scheduler::ccec_pmu;
    __gm__ pa_scheduler::WorkerResult &result = state->results[worker_id];
    PmuSnapshot sample;
    sample.status = context.selector_status;
    if (context.started && context.reg_base != 0) {
        // gate 只在整个 Submit 前后各操作一次。停止后先读从未中途清零的
        // primary counter（不含 shadow CNT5）之后，再读取 CNT8/CNT5 tail
        // 完成软件重建。
        bisheng::cce::metrics_prof_stop();
        sample = ReadObservedCounters(context.reg_base);
        const IcacheShadowSnapshot tail = ReadShadowCounters(context.reg_base);
        context.shadow_requests += tail.requests;
        context.shadow_misses += tail.misses;
        sample.status = context.selector_status | kStatusWindowStarted | kStatusWindowStopped;
        context.phase_status |= kPhaseStatusWindowStopped;
        if (sample.total_cycles != 0) sample.status |= kStatusTotalNonzero;
    }

    if (context.shadow_requests == sample.icache_requests)
        context.phase_status |= kPhaseStatusShadowRequestsMatch;
    if (context.shadow_misses == sample.icache_misses)
        context.phase_status |= kPhaseStatusShadowMissesMatch;
    if (!context.boundary_error && !context.phase_armed &&
        context.begin_reads == context.end_reads && context.end_reads == context.phase_calls)
        context.phase_status |= kPhaseStatusBoundariesBalanced;
    // 两个 shadow counter 是顺序 ld_dev，并非同一时刻的原子快照；局部
    // phase 的 miss/request 边界会错开数条指令，故不能硬性要求局部
    // miss<=request。A5 上运行中 read-to-clear 还会与同周期事件递增竞争，
    // shadow 允许小于未中途读取的 primary，但绝不能反向超过它。primary-
    // shadow 是该次采集可直接给出的局部分段误差包络，而不是要静默吞掉的差值。
    if (context.phase_requests <= context.shadow_requests &&
        context.phase_misses <= context.shadow_misses &&
        context.shadow_misses <= context.shadow_requests &&
        context.shadow_requests <= sample.icache_requests &&
        context.shadow_misses <= sample.icache_misses)
        context.phase_status |= kPhaseStatusValuesOrdered;
    if (FitsUint32(context.shadow_requests) && FitsUint32(context.shadow_misses) &&
        FitsUint32(context.phase_requests) && FitsUint32(context.phase_misses))
        context.phase_status |= kPhaseStatusUint32Fit;

    const bool none_shape =
        pa_scheduler::kCompiledSubmitPmuPhase == pa_scheduler::SubmitPmuPhase::None &&
        context.phase_calls == 0 && context.begin_reads == 0 && context.end_reads == 0 &&
        context.phase_requests == 0 && context.phase_misses == 0 &&
        context.phase_elapsed_ticks == 0;
    const bool running_shape =
        pa_scheduler::kCompiledSubmitPmuPhase != pa_scheduler::SubmitPmuPhase::None &&
        context.phase_calls == state->config.batches * pa_scheduler::kTasksPerBatch &&
        context.begin_reads == context.phase_calls &&
        context.end_reads == context.phase_calls;
    if (none_shape || running_shape)
        context.phase_status |= kPhaseStatusPhaseShape;
    const bool phase_time_valid =
        pa_scheduler::kCompiledSubmitPmuPhase == pa_scheduler::SubmitPmuPhase::None
            ? context.phase_elapsed_ticks == 0
            : context.phase_calls != 0 && context.phase_elapsed_ticks != 0;
    if (phase_time_valid)
        context.phase_status |= kPhaseStatusTimeValid;

    PublishPmuSnapshot(result, sample);
    PublishSubmitPmuContext(result, context);
}
#endif  // PA_BUILD_SUBMIT_PMU

}  // namespace

#if defined(PA_COMPETE_FIRST_SPLIT_FINISH) && defined(PA_BUILD_AIC)
// runtime entry/state-owner TU 每次 launch 只调用一次该 orchestration；它
// 不是 kernel entry，最终由 version script 局部化，避免污染 runtime 入口枚举。
extern "C" __attribute__((noinline, used)) __aicore__ void
pa_scheduler_compete_first_callback_orchestration_aic(
    __gm__ pa_scheduler::SchedulerState *state, uint32_t worker_id
) {
    pa_scheduler::RunScheduler<CcecOps>(state, worker_id, pa_scheduler::CoreRole::Aic);
}
#elif defined(PA_COMPETE_FIRST_SPLIT_FINISH) && defined(PA_BUILD_AIV)
extern "C" __attribute__((noinline, used)) __aicore__ void
pa_scheduler_compete_first_callback_orchestration_aiv(
    __gm__ pa_scheduler::SchedulerState *state, uint32_t worker_id
) {
    pa_scheduler::RunScheduler<CcecOps>(state, worker_id, pa_scheduler::CoreRole::Aiv);
}
#elif defined(PA_BUILD_AIC)
// 同一源码分别按 cube/vec 架构编译；metadata 声明每个物理 block 静态组合 1 个 AIC 与 2 个 AIV。
PTO_SYNCALL_MIX_AIC_KERNEL_META(pa_scheduler_0_mix_aic, 1, 2);

extern "C" __global__ __aicore__ void pa_scheduler_0_mix_aic(__gm__ pa_scheduler::SchedulerState *state) {
    // 32 个物理 block 的 AIC 直接使用 block_idx，形成连续 worker 0..31。
    const uint32_t worker_id = static_cast<uint32_t>(get_block_idx());
    pa_scheduler::RunScheduler<CcecOps>(state, worker_id, pa_scheduler::CoreRole::Aic);
}
#elif defined(PA_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(pa_scheduler_0_mix_aiv, 1, 2);

extern "C" __global__ __aicore__ void pa_scheduler_0_mix_aiv(__gm__ pa_scheduler::SchedulerState *state) {
    // 每个 block 的两个 vector sub-block 展平为 vector_id=2*b+subblock，偏移 32 后形成 worker 32..95。
    const uint32_t vector_id = static_cast<uint32_t>(get_block_idx() * get_subblockdim() + get_subblockid());
    const uint32_t worker_id = pa_scheduler::kAicWorkers + vector_id;
    pa_scheduler::RunScheduler<CcecOps>(state, worker_id, pa_scheduler::CoreRole::Aiv);
}
#else
#error "Compile with PA_BUILD_AIC or PA_BUILD_AIV"
#endif
