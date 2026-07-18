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
#include <pto/common/kernel_meta.hpp>

#include "pmu_probe.h"

#define PA_DEVICE __aicore__ inline
#define PA_GM __gm__
#include "../common/pa_scheduler_core.h"

namespace {

template <uint32_t Count>
__aicore__ inline void EmitNops() {
#pragma unroll
    for (uint32_t index = 0; index < Count; ++index) {
        asm volatile("nop");
    }
}

__aicore__ inline void RuntimeNop(uint32_t count) {
    // 两侧全流水屏障把可调 NOP 段限定为 kernel 模拟体，避免前后调度访存进入被测计算区间。
    __builtin_cce_pipe_barrier(PIPE_ALL);
    while (count >= 256) {
        EmitNops<256>();
        count -= 256;
    }
    if ((count & 128U) != 0) EmitNops<128>();
    if ((count & 64U) != 0) EmitNops<64>();
    if ((count & 32U) != 0) EmitNops<32>();
    if ((count & 16U) != 0) EmitNops<16>();
    if ((count & 8U) != 0) EmitNops<8>();
    if ((count & 4U) != 0) EmitNops<4>();
    if ((count & 2U) != 0) EmitNops<2>();
    if ((count & 1U) != 0) EmitNops<1>();
    __builtin_cce_pipe_barrier(PIPE_ALL);
}

#if defined(PA_BUILD_AIC)
#define PA_ICACHE_TARGET_NAME pa_icache_target_aic
#define PA_ICACHE_MEASURE_NAME pa_icache_measure_aic
#define PA_ICACHE_THRASH_NAME pa_icache_thrash_aic
#elif defined(PA_BUILD_AIV)
#define PA_ICACHE_TARGET_NAME pa_icache_target_aiv
#define PA_ICACHE_MEASURE_NAME pa_icache_measure_aiv
#define PA_ICACHE_THRASH_NAME pa_icache_thrash_aiv
#endif

// cold/warm 两条路径调用同一个 8B 目标函数。入口按 128B I-cache line
// 对齐，构建脚本还会核对符号尺寸、对齐和 target -> harness -> thrash 布局。
__aicore__ static __attribute__((noinline, used, aligned(128), section(".text.pa_icache_target")))
void PA_ICACHE_TARGET_NAME() {
    asm volatile(
        ".rept 1\n"
        "nop\n"
        ".endr\n"
    );
}

__aicore__ static __attribute__((noinline, used, aligned(128), section(".text.pa_icache_thrash")))
void PA_ICACHE_THRASH_NAME();

struct CcecOps {
    static constexpr bool kAtomicReturnReadyObserved = true;

    // 该适配层把平台无关调度器需要的原子、计时、NOP 和 cache 操作逐一映射到 CCEC intrinsic。
    // A5 上 PA 的共享“读取”使用 atomicAdd(addr, 0)，不是普通 GM load；这里保留其 RMW 竞争语义。
    __aicore__ static inline int32_t Load(__gm__ volatile int32_t *address) {
        // atomicAdd 返回加法发生前的值；加数为 0，因此它就是本次共享读取的结果。
        return atomicAdd(const_cast<__gm__ int32_t *>(address), static_cast<int32_t>(0));
    }

    __aicore__ static inline int64_t Load(__gm__ volatile int64_t *address) {
        return atomicAdd(const_cast<__gm__ int64_t *>(address), static_cast<int64_t>(0));
    }

    __aicore__ static inline uint64_t Load(__gm__ volatile uint64_t *address) {
        return atomicAdd(const_cast<__gm__ uint64_t *>(address), static_cast<uint64_t>(0));
    }

    __aicore__ static inline int32_t Exchange(__gm__ volatile int32_t *address, int32_t value) {
        // atomicExch 同样返回旧值；当前 completion/fatal 发布只需要其原子写入副作用。
        return atomicExch(const_cast<__gm__ int32_t *>(address), value);
    }

    __aicore__ static inline int64_t Exchange(__gm__ volatile int64_t *address, int64_t value) {
        return atomicExch(const_cast<__gm__ int64_t *>(address), value);
    }

    __aicore__ static inline uint64_t Exchange(__gm__ volatile uint64_t *address, uint64_t value) {
        return atomicExch(const_cast<__gm__ uint64_t *>(address), value);
    }

    __aicore__ static inline int64_t FetchAdd(__gm__ volatile int64_t *address, int64_t value) {
        // 返回递增前的计数；启动和 replay 屏障只关心全局累加结果，因此调用方不使用该返回值。
        return atomicAdd(const_cast<__gm__ int64_t *>(address), value);
    }

    __aicore__ static inline int64_t FetchMax(__gm__ volatile int64_t *address, int64_t value, uint64_t &retries) {
        // CCEC 直接生成单条硬件 atomicMax，不存在 CPU CAS 循环可观测的重试次数。
        retries = 0;
        // 返回更新前的 cursor/frontier，Claim 用它判定 winner，frontier 扫描用它吸收其他核的进度。
        return atomicMax(const_cast<__gm__ int64_t *>(address), value);
    }

    // PA's A5 OUT_OF_ORDER_STORE_BARRIER is intentionally a no-op; cache
    // coherency is handled by the runtime's DCCI protocol.
    // 这是对生产 A5 契约的刻意复刻，不是遗漏 barrier；若在这里额外插入 dsb，
    // 会改变待测 Submit 热路径。completion 使用 atomic，config/trace 的 cache
    // 可见性则由各自既有的 DCCI 路径处理。
    __aicore__ static inline void StoreBarrier() {}

    __aicore__ static inline uint64_t Now() { return static_cast<uint64_t>(get_sys_cnt()); }

    template <typename T>
    __aicore__ static inline uint64_t NowAfterAtomicResult(T value) {
        static_assert(sizeof(T) == 4 || sizeof(T) == 8, "atomic dependency expects a scalar result");
        uint64_t cycle = 0;
        // 同一个 inline asm 块先真正消费 atomic 返回寄存器，再读取
        // SYS_CNT；编译器不能把 t1 拆到依赖 MOV 之前。AIC/AIV 对该序列
        // 生成相同指令字节，且不增加 DSB/ISB/GM 访存。该边界仍只表示
        // 返回值已可被本核 scalar 消费，不表示跨核全局可见。
        asm volatile(
            "MOV %0, %0\n"
            "MOV %1, SYS_CNT\n"
            : "+l"(value), "=&l"(cycle)
        );
        return cycle;
    }

    __aicore__ static inline void Nop(uint32_t count) { RuntimeNop(count); }

    __aicore__ static inline bool PmuWindowStart(
        __gm__ pa_scheduler::SchedulerState *state, uint32_t worker_id
    );

    __aicore__ static inline void PmuWindowStop(
        __gm__ pa_scheduler::SchedulerState *state, uint32_t worker_id, bool started
    );

    // SPIN_WAIT_HINT is also a no-op in the real A5 inner-kernel contract.
    // 同理不额外插入 nop，让等待循环保留真实 PA 内核“不主动退避”的指令成本。
    __aicore__ static inline void SpinHint() {}

    __aicore__ static inline void InvalidateRegion(__gm__ const void *address, uint64_t bytes) {
        // 逐 cache line 失效并以 dsb 收口，供 worker 在启动时读取 host 刚写入的 standalone 控制区。
        if (bytes == 0) return;
        const uint64_t start = reinterpret_cast<uint64_t>(address) & ~uint64_t{63};
        const uint64_t end = (reinterpret_cast<uint64_t>(address) + bytes + 63) & ~uint64_t{63};
        for (uint64_t current = start; current < end; current += 64) {
            dcci(reinterpret_cast<__gm__ uint8_t *>(current), SINGLE_CACHE_LINE);
        }
        dsb((mem_dsb_t)0);
    }

    __aicore__ static inline void FlushRegion(__gm__ void *address, uint64_t bytes) {
        // 泳道记录先写普通 GM cache，kernel 结束前显式 CACHELINE_OUT，确保 host D2H 能看到完整记录。
        if (bytes == 0) return;
        __asm__ volatile("" ::: "memory");
        const uint64_t start = reinterpret_cast<uint64_t>(address) & ~uint64_t{63};
        const uint64_t end = (reinterpret_cast<uint64_t>(address) + bytes + 63) & ~uint64_t{63};
        for (uint64_t current = start; current < end; current += 64) {
            dcci(reinterpret_cast<__gm__ uint8_t *>(current), SINGLE_CACHE_LINE, CACHELINE_OUT);
        }
        dsb((mem_dsb_t)0);
    }

    __aicore__ static inline void Publish(__gm__ uint64_t *address, uint64_t value) {
        // 每核独占的 WorkerResult 用 bypass-DCache store 发布，host 同步后可直接 D2H，无需共享原子竞争。
        __builtin_cce_st_dev(value, address, 0);
    }

    __aicore__ static inline void Publish(__gm__ uint32_t *address, uint32_t value) {
        __builtin_cce_st_dev(value, address, 0);
    }
};

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

struct IcachePairSnapshot {
    PmuSnapshot cold;
    PmuSnapshot warm;
    uint64_t cold_window_ticks = 0;
    uint64_t warm_window_ticks = 0;
};

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
    sample.mte3_busy = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                       pa_scheduler::ccec_pmu::kCnt5Offset>(reg_base);
    sample.icache_requests = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                             pa_scheduler::ccec_pmu::kCnt6Offset>(reg_base);
    sample.icache_misses = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                           pa_scheduler::ccec_pmu::kCnt7Offset>(reg_base);
    sample.fix_busy = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                      pa_scheduler::ccec_pmu::kCnt8Offset>(reg_base);
    const uint64_t low = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                         pa_scheduler::ccec_pmu::kTotalLowOffset>(reg_base);
    const uint64_t high = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                          pa_scheduler::ccec_pmu::kTotalHighOffset>(reg_base);
    sample.total_cycles = low | (high << 32);
    return sample;
}

__aicore__ inline void AccumulateObserved(const PmuSnapshot &sample, PmuSnapshot *total) {
    total->total_cycles += sample.total_cycles;
    total->vector_busy += sample.vector_busy;
    total->cube_busy += sample.cube_busy;
    total->scalar_busy += sample.scalar_busy;
    total->mte1_busy += sample.mte1_busy;
    total->mte2_busy += sample.mte2_busy;
    total->mte3_busy += sample.mte3_busy;
    total->icache_requests += sample.icache_requests;
    total->icache_misses += sample.icache_misses;
    total->fix_busy += sample.fix_busy;
}

// 禁止内联，确保 cold/warm 经过同一函数体、同一 PMU gate 和同一目标
// callsite；否则编译器复制 harness 后，两条路径的取指状态不再是单变量实验。
__aicore__ static __attribute__((noinline, used, aligned(128), section(".text.pa_icache_harness")))
void PA_ICACHE_MEASURE_NAME(
    uint64_t reg_base, bool warm, PmuSnapshot *total, uint64_t *window_ticks
) {
    // 两条路径都先在窗口外执行 64 KiB capacity sweep。warm 只多一次窗外
    // target 预取，并且发生在 read-clear 之前，避免把预热 miss 计入 warm 窗口。
    bisheng::cce::metrics_prof_stop();
    PA_ICACHE_THRASH_NAME();
    if (warm) PA_ICACHE_TARGET_NAME();
    (void)ReadObservedCounters(reg_base);

    bisheng::cce::metrics_prof_start();
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    PA_ICACHE_TARGET_NAME();
    const uint64_t end = static_cast<uint64_t>(get_sys_cnt());
    bisheng::cce::metrics_prof_stop();

    AccumulateObserved(ReadObservedCounters(reg_base), total);
    *window_ticks += end - begin;
}

// DAV_3510 scalar I-cache 最大为 32 KiB。thrash 放在 target/harness 之后，
// 顺序执行 64 KiB 指令覆盖所有 set 多轮；返回低地址 harness 时不向前预取 target。
__aicore__ static void PA_ICACHE_THRASH_NAME() {
    asm volatile(
        ".rept 16384\n"
        "nop\n"
        ".endr\n"
    );
}

__aicore__ inline void RunIcachePhase(
    uint64_t reg_base, bool warm, uint32_t trials, PmuSnapshot *total, uint64_t *window_ticks
) {
    // 每个 phase 先丢弃一次同分支训练样本，避免逐 trial 交替分支历史把
    // false 路径的目标行提前取回。
    PmuSnapshot discarded;
    uint64_t discarded_ticks = 0;
    PA_ICACHE_MEASURE_NAME(reg_base, warm, &discarded, &discarded_ticks);
    for (uint32_t trial = 0; trial < trials; ++trial) {
        PA_ICACHE_MEASURE_NAME(reg_base, warm, total, window_ticks);
    }
}

__aicore__ inline IcachePairSnapshot RunSingleIcacheProbe(
    uint64_t reg_base, uint32_t trials, uint32_t worker_id
) {
    IcachePairSnapshot pair;
    // 每个 role 内各一半 worker 使用 cold-first/warm-first，抵消两个 phase
    // 的固定时间顺序；phase 内保持相同分支历史。
    if ((worker_id & 1U) == 0U) {
        RunIcachePhase(reg_base, false, trials, &pair.cold, &pair.cold_window_ticks);
        RunIcachePhase(reg_base, true, trials, &pair.warm, &pair.warm_window_ticks);
    } else {
        RunIcachePhase(reg_base, true, trials, &pair.warm, &pair.warm_window_ticks);
        RunIcachePhase(reg_base, false, trials, &pair.cold, &pair.cold_window_ticks);
    }
    return pair;
}

struct PmuRegisterContext {
    uint64_t reg_base = 0;
    uint32_t status = 0;
};

__aicore__ inline PmuRegisterContext ResolvePmuRegisters(__gm__ pa_scheduler::SchedulerState *state) {
    using namespace pa_scheduler::ccec_pmu;
    PmuRegisterContext context;
    const uint32_t physical_core_id = static_cast<uint32_t>(get_coreid()) & kStatusCoreIdMask;
    context.status = kStatusRequested | (physical_core_id << kStatusCoreIdShift);
    const uint64_t table_address =
        static_cast<uint64_t>(state->config.reserved[kConfigRegTableLow]) |
        (static_cast<uint64_t>(state->config.reserved[kConfigRegTableHigh]) << 32);
    if (state->config.reserved[kConfigMagic] != kConfigMagicValue || table_address == 0 ||
        physical_core_id >= kPhysicalSubcoreCount) {
        return context;
    }
    context.status |= kStatusCoreIdValid;
    __gm__ const uint64_t *register_bases = reinterpret_cast<__gm__ const uint64_t *>(table_address);
    context.reg_base = register_bases[physical_core_id];
    if (context.reg_base == 0) return context;
    context.status |= kStatusRegMapped;

    // selector 与 standalone owner 的 A5 PIPE_UTILIZATION 事件表逐项核对，避免
    // 配置或 ABI 错位时仍把 CNT0..8 的数值按 vector/scalar/I-cache 名称导出。
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
    if (ReadPmuRegister<kSelectorBlockOffset, kCnt5SelectorOffset>(context.reg_base) == kMte3BusyEvent)
        context.status |= kStatusCnt5Selector;
    if (ReadPmuRegister<kSelectorBlockOffset, kCnt6SelectorOffset>(context.reg_base) == kIcacheRequestEvent)
        context.status |= kStatusCnt6Selector;
    if (ReadPmuRegister<kSelectorBlockOffset, kCnt7SelectorOffset>(context.reg_base) == kIcacheMissEvent)
        context.status |= kStatusCnt7Selector;
    if (ReadPmuRegister<kSelectorBlockOffset, kCnt8SelectorOffset>(context.reg_base) == kFixBusyEvent)
        context.status |= kStatusCnt8Selector;
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
    CcecOps::Publish(&result.pmu_fix_busy, sample.fix_busy);
    // 非 icache-single 模式也显式清零配对 sidecar，避免同一 device allocation
    // 被后续 run 复用时把陈旧 warm 数据误当成本轮结果。
    CcecOps::Publish(&result.pmu_window_ticks, static_cast<uint64_t>(0));
    CcecOps::Publish(&result.pmu_warm_total_cycles, static_cast<uint64_t>(0));
    CcecOps::Publish(&result.pmu_warm_window_ticks, static_cast<uint64_t>(0));
    CcecOps::Publish(&result.pmu_warm_icache_requests, static_cast<uint32_t>(0));
    CcecOps::Publish(&result.pmu_warm_icache_misses, static_cast<uint32_t>(0));
}

__aicore__ inline void PublishIcachePair(
    __gm__ pa_scheduler::WorkerResult &result, const IcachePairSnapshot &pair
) {
    PublishPmuSnapshot(result, pair.cold);
    CcecOps::Publish(&result.pmu_window_ticks, pair.cold_window_ticks);
    CcecOps::Publish(&result.pmu_warm_total_cycles, pair.warm.total_cycles);
    CcecOps::Publish(&result.pmu_warm_window_ticks, pair.warm_window_ticks);
    CcecOps::Publish(&result.pmu_warm_icache_requests, pair.warm.icache_requests);
    CcecOps::Publish(&result.pmu_warm_icache_misses, pair.warm.icache_misses);
}

__aicore__ inline bool CcecOps::PmuWindowStart(
    __gm__ pa_scheduler::SchedulerState *state, uint32_t worker_id
) {
    using namespace pa_scheduler::ccec_pmu;
    (void)worker_id;
    const WindowMode mode = static_cast<WindowMode>(state->config.reserved[kConfigMode]);
    if (mode != WindowMode::SubmitAll) return false;
    const PmuRegisterContext context = ResolvePmuRegisters(state);
    if (context.reg_base == 0) return false;
    // Main AICPU owner 已在 launch 前配置并开启计数；先 stop + snapshot/read-clear，
    // 再从本 worker 的首个 orchestration 动作开始独立累计。
    bisheng::cce::metrics_prof_stop();
    (void)ReadObservedCounters(context.reg_base);
    bisheng::cce::metrics_prof_start();
    return true;
}

__aicore__ inline void CcecOps::PmuWindowStop(
    __gm__ pa_scheduler::SchedulerState *state, uint32_t worker_id, bool started
) {
    using namespace pa_scheduler::ccec_pmu;
    if (!started) return;
    // 先冻结 gate，再读取 selector 和 counter。若先 ResolvePmuRegisters，九次
    // selector ld_dev 会被误计入本 worker 的 Submit 窗口。
    bisheng::cce::metrics_prof_stop();
    const PmuRegisterContext context = ResolvePmuRegisters(state);
    PmuSnapshot sample;
    sample.status = context.status;
    if (context.reg_base != 0) {
        sample = ReadObservedCounters(context.reg_base);
        sample.status = context.status | kStatusWindowStarted | kStatusWindowStopped;
        if (sample.total_cycles != 0) sample.status |= kStatusTotalNonzero;
    }
    PublishPmuSnapshot(state->results[worker_id], sample);
}

__aicore__ inline void RunPmuProbe(__gm__ pa_scheduler::SchedulerState *state, uint32_t worker_id) {
    using namespace pa_scheduler::ccec_pmu;
    __gm__ pa_scheduler::WorkerResult &result = state->results[worker_id];
    const WindowMode mode = static_cast<WindowMode>(state->config.reserved[kConfigMode]);
    // SubmitAll 已在公共调度器 hook 内完成 start/stop/read/publish；此处保留
    // empty/scalar/scalar-double/icache-single 校准，和正式窗口可独立复验。
    if (mode == WindowMode::SubmitAll) return;

    PmuSnapshot sample;
    if (mode != WindowMode::Off) {
        const PmuRegisterContext context = ResolvePmuRegisters(state);
        sample.status = context.status;
        if (context.reg_base != 0) {
            bisheng::cce::metrics_prof_stop();
            const PmuSnapshot prior = ReadObservedCounters(context.reg_base);
            if (mode == WindowMode::IcacheSingle) {
                IcachePairSnapshot pair = RunSingleIcacheProbe(
                    context.reg_base, state->config.reserved[kConfigIcacheTrials], worker_id
                );
                pair.cold.status = context.status | kStatusWindowStarted | kStatusWindowStopped |
                    kStatusIcachePairObserved;
                if (pair.cold.total_cycles != 0U) pair.cold.status |= kStatusTotalNonzero;
                if (pair.cold.total_cycles < prior.total_cycles) {
                    pair.cold.status |= kStatusPriorSnapshotLarger;
                }
                PublishIcachePair(result, pair);
                return;
            }
            bisheng::cce::metrics_prof_start();
            if (mode == WindowMode::Scalar || mode == WindowMode::ScalarDouble) {
                RuntimeNop(state->config.reserved[kConfigScalarNops]);
            }
            if (mode == WindowMode::ScalarDouble) {
                bisheng::cce::metrics_prof_stop();
                bisheng::cce::metrics_prof_start();
                RuntimeNop(state->config.reserved[kConfigScalarNops]);
            }
            bisheng::cce::metrics_prof_stop();
            sample = ReadObservedCounters(context.reg_base);
            sample.status = context.status | kStatusWindowStarted | kStatusWindowStopped;
            if (sample.total_cycles != 0) sample.status |= kStatusTotalNonzero;
            if (sample.total_cycles < prior.total_cycles) sample.status |= kStatusPriorSnapshotLarger;
        }
    }
    PublishPmuSnapshot(result, sample);
}

}  // namespace

#if defined(PA_BUILD_AIC)
// 同一源码分别按 cube/vec 架构编译；metadata 声明每个物理 block 静态组合 1 个 AIC 与 2 个 AIV。
PTO_SYNCALL_MIX_AIC_KERNEL_META(pa_scheduler_0_mix_aic, 1, 2);

extern "C" __global__ __aicore__ void pa_scheduler_0_mix_aic(__gm__ pa_scheduler::SchedulerState *state) {
    // 32 个物理 block 的 AIC 直接使用 block_idx，形成连续 worker 0..31。
    const uint32_t worker_id = static_cast<uint32_t>(get_block_idx());
    pa_scheduler::RunScheduler<CcecOps>(state, worker_id, pa_scheduler::CoreRole::Aic);
    RunPmuProbe(state, worker_id);
}
#elif defined(PA_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(pa_scheduler_0_mix_aiv, 1, 2);

extern "C" __global__ __aicore__ void pa_scheduler_0_mix_aiv(__gm__ pa_scheduler::SchedulerState *state) {
    // 每个 block 的两个 vector sub-block 展平为 vector_id=2*b+subblock，偏移 32 后形成 worker 32..95。
    const uint32_t vector_id = static_cast<uint32_t>(get_block_idx() * get_subblockdim() + get_subblockid());
    const uint32_t worker_id = pa_scheduler::kAicWorkers + vector_id;
    pa_scheduler::RunScheduler<CcecOps>(state, worker_id, pa_scheduler::CoreRole::Aiv);
    RunPmuProbe(state, worker_id);
}
#else
#error "Compile with PA_BUILD_AIC or PA_BUILD_AIV"
#endif
