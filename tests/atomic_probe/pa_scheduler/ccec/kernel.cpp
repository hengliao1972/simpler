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
    uint32_t scalar_busy = 0;
    uint32_t icache_requests = 0;
    uint32_t icache_misses = 0;
    uint32_t status = 0;
};

template <uint32_t BlockOffset, uint32_t RegisterOffset>
__aicore__ inline uint32_t ReadPmuRegister(uint64_t reg_base) {
    // 传给 ld_dev 的是重基址后的 __gm__ 指针；相对 offset 均落在编译器允许的 [-2048, 2047]。
    int32_t *block = reinterpret_cast<int32_t *>(reg_base + BlockOffset);
    return static_cast<uint32_t>(ld_dev(block, static_cast<int16_t>(RegisterOffset - BlockOffset)));
}

__aicore__ inline PmuSnapshot ReadObservedCounters(uint64_t reg_base) {
    PmuSnapshot sample;
    sample.scalar_busy = ReadPmuRegister<pa_scheduler::ccec_pmu::kCounterBlockOffset,
                                         pa_scheduler::ccec_pmu::kCnt2Offset>(reg_base);
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

__aicore__ inline void RunPmuProbe(__gm__ pa_scheduler::SchedulerState *state, uint32_t worker_id) {
    using namespace pa_scheduler::ccec_pmu;
    __gm__ pa_scheduler::WorkerResult &result = state->results[worker_id];
    PmuSnapshot sample;

    const WindowMode mode = static_cast<WindowMode>(state->config.reserved[kConfigMode]);
    if (mode != WindowMode::Off) {
        sample.status |= kStatusRequested;
        const uint32_t physical_core_id = static_cast<uint32_t>(get_coreid()) & kStatusCoreIdMask;
        sample.status |= physical_core_id << kStatusCoreIdShift;
        const uint64_t table_address =
            static_cast<uint64_t>(state->config.reserved[kConfigRegTableLow]) |
            (static_cast<uint64_t>(state->config.reserved[kConfigRegTableHigh]) << 32);
        if (state->config.reserved[kConfigMagic] == kConfigMagicValue && table_address != 0 &&
            physical_core_id < kPhysicalSubcoreCount) {
            sample.status |= kStatusCoreIdValid;
            __gm__ const uint64_t *register_bases = reinterpret_cast<__gm__ const uint64_t *>(table_address);
            const uint64_t reg_base = register_bases[physical_core_id];
            if (reg_base != 0) {
                sample.status |= kStatusRegMapped;
                const uint32_t selector2 =
                    ReadPmuRegister<kSelectorBlockOffset, kCnt2SelectorOffset>(reg_base);
                const uint32_t selector6 =
                    ReadPmuRegister<kSelectorBlockOffset, kCnt6SelectorOffset>(reg_base);
                const uint32_t selector7 =
                    ReadPmuRegister<kSelectorBlockOffset, kCnt7SelectorOffset>(reg_base);
                if (selector2 == kScalarBusyEvent) sample.status |= kStatusCnt2Selector;
                if (selector6 == kIcacheRequestEvent) sample.status |= kStatusCnt6Selector;
                if (selector7 == kIcacheMissEvent) sample.status |= kStatusCnt7Selector;

                // 外部 profiler 启动 task 时可能已经累计了 scheduler；先冻结并读取一次窗口前快照。
                // A5 的 snapshot 会消费/清除当前累计；stop/start 只负责门控，中间不读取才能续积多段窗口。
                bisheng::cce::metrics_prof_stop();
                const PmuSnapshot prior = ReadObservedCounters(reg_base);
                bisheng::cce::metrics_prof_start();
                if (mode == WindowMode::Scalar || mode == WindowMode::ScalarDouble) {
                    RuntimeNop(state->config.reserved[kConfigScalarNops]);
                }
                if (mode == WindowMode::ScalarDouble) {
                    // 两段相同工作量之间只切 gate、不读取 counter，用于确认 resume 是累计还是重置。
                    bisheng::cce::metrics_prof_stop();
                    bisheng::cce::metrics_prof_start();
                    RuntimeNop(state->config.reserved[kConfigScalarNops]);
                }
                bisheng::cce::metrics_prof_stop();
                sample = ReadObservedCounters(reg_base);
                sample.status |= kStatusRequested | kStatusRegMapped | kStatusCoreIdValid |
                                 (physical_core_id << kStatusCoreIdShift);
                if (selector2 == kScalarBusyEvent) sample.status |= kStatusCnt2Selector;
                if (selector6 == kIcacheRequestEvent) sample.status |= kStatusCnt6Selector;
                if (selector7 == kIcacheMissEvent) sample.status |= kStatusCnt7Selector;
                if (sample.total_cycles != 0) sample.status |= kStatusTotalNonzero;

                // 窗口前 read-clear 之后只做这一次最终 snapshot；重复读取会看到读取路径自身的残余。
                // prior 更大只作为“已从此前 scheduler 累计中隔离”的辅助证据，不作为长负载通用门禁。
                if (sample.total_cycles < prior.total_cycles) sample.status |= kStatusPriorSnapshotLarger;
            }
        }
    }

    // 结果位于每核独占 sidecar，五次 bypass store 不参与 Submit 时间口径。
    CcecOps::Publish(&result.pmu_total_cycles, sample.total_cycles);
    CcecOps::Publish(&result.pmu_scalar_busy, sample.scalar_busy);
    CcecOps::Publish(&result.pmu_icache_requests, sample.icache_requests);
    CcecOps::Publish(&result.pmu_icache_misses, sample.icache_misses);
    CcecOps::Publish(&result.pmu_status, sample.status);
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
