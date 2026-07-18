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

namespace {

using namespace pto;

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
// QK/PV 的首版真实负载使用完整的 128x128 float cube 路径。输入来自独立 GM
// workspace，输出属于当前 worker；每次迭代都等 FIX 写回 GM 后再复用 L0C，
// 因而函数返回就是该模拟 task 的完成边界，而不是单纯的指令发射边界。
static __aicore__ __attribute__((noinline, used)) void pa_real_cube_workload_aic(
    __gm__ float *input_a, __gm__ float *input_b, __gm__ float *output, uint32_t repeats
) {
    constexpr int kTile = static_cast<int>(pa_scheduler::winner_workload::kTileRows);
    constexpr int kBlockAlign = C0_SIZE_BYTE / sizeof(float);
    static_assert(kTile % 16 == 0, "cube M must be 16-aligned");
    static_assert(kTile % kBlockAlign == 0, "cube K/N must satisfy C0 alignment");

    using GlobalData = GlobalTensor<
        float, Shape<1, 1, 1, kTile, kTile>,
        pto::Stride<kTile * kTile, kTile * kTile, kTile * kTile, kTile, 1>>;
    using TileMatA = Tile<
        TileType::Mat, float, kTile, kTile, BLayout::ColMajor,
        kTile, kTile, SLayout::RowMajor, 512>;
    using TileMatB = Tile<
        TileType::Mat, float, kTile, kTile, BLayout::ColMajor,
        kTile, kTile, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<float, kTile, kTile, kTile, kTile>;
    using RightTile = TileRight<float, kTile, kTile, kTile, kTile>;
    using AccTile = TileAcc<float, kTile, kTile, kTile, kTile>;

    GlobalData input_a_global(input_a);
    GlobalData input_b_global(input_b);
    GlobalData output_global(output);
    TileMatA input_a_mat;
    TileMatB input_b_mat;
    LeftTile input_a_l0;
    RightTile input_b_l0;
    AccTile output_l0;
    TASSIGN(input_a_mat, 0x0);
    TASSIGN(input_b_mat, 0x20000);
    TASSIGN(input_a_l0, 0x0);
    TASSIGN(input_b_l0, 0x0);
    TASSIGN(output_l0, 0x0);

    for (uint32_t iteration = 0; iteration < repeats; ++iteration) {
        TLOAD(input_a_mat, input_a_global);
        TLOAD(input_b_mat, input_b_global);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        TMOV(input_a_l0, input_a_mat);
        TMOV(input_b_l0, input_b_mat);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL(output_l0, input_a_l0, input_b_l0);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        TSTORE(output_global, output_l0);
        set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    }
}
#elif defined(PA_BUILD_AIV)
template <bool Multiply>
__aicore__ inline void RunRealVectorWorkload(
    __gm__ float *input_a, __gm__ float *input_b, __gm__ float *output, uint32_t repeats
) {
    constexpr int kRows = static_cast<int>(pa_scheduler::winner_workload::kTileRows);
    constexpr int kCols = static_cast<int>(pa_scheduler::winner_workload::kTileCols);
    using GlobalData = GlobalTensor<
        float, Shape<1, 1, 1, kRows, kCols>, pto::Stride<1, 1, 1, kCols, 1>>;
    using TileData = Tile<
        TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;

    GlobalData input_a_global(input_a);
    GlobalData input_b_global(input_b);
    GlobalData output_global(output);
    TileData input_a_tile(kRows, kCols);
    TileData input_b_tile(kRows, kCols);
    TileData output_tile(kRows, kCols);
    TASSIGN(input_a_tile, 0x0);
    TASSIGN(input_b_tile, 0x10000);
    TASSIGN(output_tile, 0x20000);

    for (uint32_t iteration = 0; iteration < repeats; ++iteration) {
        TLOAD(input_a_tile, input_a_global);
        TLOAD(input_b_tile, input_b_global);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if constexpr (Multiply) {
            TMUL(output_tile, input_a_tile, input_b_tile);
        } else {
            TADD(output_tile, input_a_tile, input_b_tile);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(output_global, output_tile);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    }
}

static __aicore__ __attribute__((noinline, used)) void pa_real_vector_add_workload_aiv(
    __gm__ float *input_a, __gm__ float *input_b, __gm__ float *output, uint32_t repeats
) {
    RunRealVectorWorkload<false>(input_a, input_b, output, repeats);
}

static __aicore__ __attribute__((noinline, used)) void pa_real_vector_mul_workload_aiv(
    __gm__ float *input_a, __gm__ float *input_b, __gm__ float *output, uint32_t repeats
) {
    RunRealVectorWorkload<true>(input_a, input_b, output, repeats);
}
#endif

// 整段真实负载分派保持为 LOCAL noinline 冷路径，避免 workspace 校验、地址计算和
// kind 分支膨胀 scalar-NOP 的 Submit 热代码；正常 NOP 对照只多一次 mode 判断。
static __aicore__ __attribute__((noinline, used)) void pa_execute_real_winner_workload(
    __gm__ pa_scheduler::SchedulerState *state, __gm__ pa_scheduler::WorkerState &worker,
    pa_scheduler::TaskKind kind
) {
    const uint64_t workspace = state->winner_workload.workspace_base;
    const uint32_t repeats = pa_scheduler::WorkloadCountForKind(
        state->winner_workload.repeats, kind
    );
    // 错版 host、截断 workspace 或越界 worker 不能继续解引用 GM。这里不额外
    // 写共享 fatal，避免在正常热路增加 atomic；host 的逐 kind sentinel/数值
    // 闭环会把这种配置错误判为失败。
    if (state->winner_workload.version != pa_scheduler::kWinnerWorkloadConfigVersion ||
        workspace == 0 ||
        state->winner_workload.workspace_bytes < pa_scheduler::winner_workload::kWorkspaceBytes ||
        worker.core_idx < 0 || static_cast<uint32_t>(worker.core_idx) >= pa_scheduler::kWorkers ||
        repeats == 0 || repeats > pa_scheduler::winner_workload::kMaxRealComputeCount) {
        return;
    }
#if defined(PA_BUILD_AIC)
    if (kind != pa_scheduler::TaskKind::Qk && kind != pa_scheduler::TaskKind::Pv) return;
#elif defined(PA_BUILD_AIV)
    if (kind != pa_scheduler::TaskKind::Sf && kind != pa_scheduler::TaskKind::Up) return;
#endif
    __gm__ float *input_a = reinterpret_cast<__gm__ float *>(workspace);
    __gm__ float *input_b = reinterpret_cast<__gm__ float *>(
        workspace + pa_scheduler::winner_workload::kTileBytes
    );
    const uint32_t kind_slot =
        (kind == pa_scheduler::TaskKind::Pv || kind == pa_scheduler::TaskKind::Up) ? 1U : 0U;
    const uint32_t output_tile =
        pa_scheduler::winner_workload::kSharedInputTiles +
        static_cast<uint32_t>(worker.core_idx) *
            pa_scheduler::winner_workload::kOutputTilesPerWorker +
        kind_slot;
    __gm__ float *output = reinterpret_cast<__gm__ float *>(
        workspace + static_cast<uint64_t>(output_tile) *
            pa_scheduler::winner_workload::kTileBytes
    );
#if defined(PA_BUILD_AIC)
    pa_real_cube_workload_aic(input_a, input_b, output, repeats);
#elif defined(PA_BUILD_AIV)
    if (kind == pa_scheduler::TaskKind::Sf) {
        pa_real_vector_add_workload_aiv(input_a, input_b, output, repeats);
    } else {
        pa_real_vector_mul_workload_aiv(input_a, input_b, output, repeats);
    }
#endif
}

#if PA_BUILD_SUBMIT_PMU
struct SubmitPmuContext;
#endif

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

    __aicore__ static inline void ExecuteKernel(
        __gm__ pa_scheduler::SchedulerState *state, __gm__ pa_scheduler::WorkerState &worker,
        pa_scheduler::TaskKind kind, uint32_t nop_count
    ) {
        const auto mode = static_cast<pa_scheduler::WinnerWorkloadMode>(state->winner_workload.mode);
        if (mode != pa_scheduler::WinnerWorkloadMode::RealCompute) {
            RuntimeNop(nop_count);
            return;
        }
        pa_execute_real_winner_workload(state, worker, kind);
    }

#if PA_BUILD_SUBMIT_PMU
    using PmuContext = SubmitPmuContext;

    __aicore__ static inline PmuContext PmuWindowStart(
        __gm__ pa_scheduler::SchedulerState *state, uint32_t worker_id
    );

    __aicore__ static inline void PmuPhaseBegin(PmuContext &context);

    __aicore__ static inline void PmuPhaseEnd(PmuContext &context);

    __aicore__ static inline void PmuWindowStop(
        __gm__ pa_scheduler::SchedulerState *state, uint32_t worker_id, PmuContext &context
    );
#else
    // swimlane 产物不携带 PMU 读寄存器或门控代码；公共调度器保留同一 hook
    // 形状，编译器会把这两个空实现完整消去。
    __aicore__ static inline bool PmuWindowStart(
        __gm__ pa_scheduler::SchedulerState *, uint32_t
    ) { return false; }

    __aicore__ static inline void PmuWindowStop(
        __gm__ pa_scheduler::SchedulerState *, uint32_t, bool
    ) {}
#endif

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

struct SubmitPmuContext {
    uint64_t reg_base = 0;
    uint64_t shadow_requests = 0;
    uint64_t shadow_misses = 0;
    uint64_t phase_requests = 0;
    uint64_t phase_misses = 0;
    uint32_t selector_status = 0;
    uint32_t phase_status = pa_scheduler::ccec_pmu::kPhaseStatusRequested;
    uint32_t phase_calls = 0;
    uint32_t begin_reads = 0;
    uint32_t end_reads = 0;
    bool started = false;
    bool phase_armed = false;
    bool boundary_error = false;
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
    CcecOps::Publish(&result.pmu_window_ticks, static_cast<uint64_t>(0));
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
    CcecOps::Publish(&result.pmu_phase_icache_requests, static_cast<uint32_t>(context.phase_requests));
    CcecOps::Publish(&result.pmu_phase_icache_misses, static_cast<uint32_t>(context.phase_misses));
    CcecOps::Publish(&result.pmu_shadow_icache_requests, static_cast<uint32_t>(context.shadow_requests));
    CcecOps::Publish(&result.pmu_shadow_icache_misses, static_cast<uint32_t>(context.shadow_misses));
}

__aicore__ inline CcecOps::PmuContext CcecOps::PmuWindowStart(
    __gm__ pa_scheduler::SchedulerState *state, uint32_t worker_id
) {
    using namespace pa_scheduler::ccec_pmu;
    (void)worker_id;
    SubmitPmuContext context;
    const WindowMode mode = static_cast<WindowMode>(state->config.reserved[kConfigMode]);
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
}

__aicore__ inline void CcecOps::PmuPhaseEnd(PmuContext &context) {
    if (!context.started || context.reg_base == 0 || !context.phase_armed) {
        context.boundary_error = true;
        return;
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
        context.phase_requests == 0 && context.phase_misses == 0;
    const bool claim_shape =
        pa_scheduler::kCompiledSubmitPmuPhase == pa_scheduler::SubmitPmuPhase::Claim &&
        context.phase_calls == state->config.batches * pa_scheduler::kTasksPerBatch &&
        context.begin_reads == context.phase_calls &&
        context.end_reads == context.phase_calls;
    const bool efdrain_shape =
        pa_scheduler::kCompiledSubmitPmuPhase == pa_scheduler::SubmitPmuPhase::EfDrain &&
        context.phase_calls == state->config.batches * pa_scheduler::kTasksPerBatch &&
        context.begin_reads == context.phase_calls &&
        context.end_reads == context.phase_calls;
    if (none_shape || claim_shape || efdrain_shape)
        context.phase_status |= kPhaseStatusPhaseShape;

    PublishPmuSnapshot(result, sample);
    PublishSubmitPmuContext(result, context);
}
#endif  // PA_BUILD_SUBMIT_PMU

}  // namespace

#if defined(PA_BUILD_AIC)
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
