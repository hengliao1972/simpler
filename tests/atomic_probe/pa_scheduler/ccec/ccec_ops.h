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
#ifndef TESTS_ATOMIC_PROBE_PA_SCHEDULER_CCEC_CCEC_OPS_H
#define TESTS_ATOMIC_PROBE_PA_SCHEDULER_CCEC_CCEC_OPS_H

// 此私有头在 CCEC/PTO 与公共调度头之后包含。kernel.cpp 每个核型只定义
// 一次真实负载实体；split finish TU 复用按核型导出的 dispatcher 与同一份
// inline Ops，避免复制或改写 atomic/cache/计时语义。
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
#include "callback_finish_api.h"
#endif

#if defined(PA_BUILD_AIC)
extern "C" __attribute__((noinline)) __aicore__ void pa_execute_real_winner_workload_aic(
    __gm__ pa_scheduler::SchedulerState *state, __gm__ pa_scheduler::WorkerState *worker,
    pa_scheduler::TaskKind kind
);
#elif defined(PA_BUILD_AIV)
extern "C" __attribute__((noinline)) __aicore__ void pa_execute_real_winner_workload_aiv(
    __gm__ pa_scheduler::SchedulerState *state, __gm__ pa_scheduler::WorkerState *worker,
    pa_scheduler::TaskKind kind
);
#else
#error "Compile CcecOps with PA_BUILD_AIC or PA_BUILD_AIV"
#endif

namespace pa_scheduler_ccec {

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

#if defined(PA_CCEC_OPS_DEFINE_REAL_WORKLOAD) && defined(PA_BUILD_AIC)
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
#elif defined(PA_CCEC_OPS_DEFINE_REAL_WORKLOAD) && defined(PA_BUILD_AIV)
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
#if defined(PA_BUILD_AIC)
        ::pa_execute_real_winner_workload_aic(state, &worker, kind);
#elif defined(PA_BUILD_AIV)
        ::pa_execute_real_winner_workload_aiv(state, &worker, kind);
#endif
    }

#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
    __aicore__ static inline pa_scheduler::CompeteFirstSplitRuntimeState &CompeteFirstSplitState() {
#if defined(PA_BUILD_AIC)
        return ::pa_scheduler_compete_first_callback_state_aic;
#elif defined(PA_BUILD_AIV)
        return ::pa_scheduler_compete_first_callback_state_aiv;
#endif
    }

    __aicore__ static inline bool FinishCompeteFirstCallback(
        const pa_scheduler::CallbackSubmitTicket *ticket, const pa_scheduler::TaskArgs *args
    ) {
#if defined(PA_BUILD_AIC)
        return ::pa_scheduler_compete_first_callback_finish_aic(ticket, args) != 0;
#elif defined(PA_BUILD_AIV)
        return ::pa_scheduler_compete_first_callback_finish_aiv(ticket, args) != 0;
#endif
    }
#endif

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

}  // namespace pa_scheduler_ccec

#if defined(PA_CCEC_OPS_DEFINE_REAL_WORKLOAD)
// 跨 TU 只暴露按架构区分的真实负载分派；Cube/Vector 实体仍保持 LOCAL。
// 最终 mixed ELF 由 version script 把该 strong 定义重新局部化，避免成为 kernel entry。
#if defined(PA_BUILD_AIC)
extern "C" __attribute__((noinline, used)) __aicore__ void pa_execute_real_winner_workload_aic(
    __gm__ pa_scheduler::SchedulerState *state, __gm__ pa_scheduler::WorkerState *worker,
    pa_scheduler::TaskKind kind
) {
#elif defined(PA_BUILD_AIV)
extern "C" __attribute__((noinline, used)) __aicore__ void pa_execute_real_winner_workload_aiv(
    __gm__ pa_scheduler::SchedulerState *state, __gm__ pa_scheduler::WorkerState *worker,
    pa_scheduler::TaskKind kind
) {
#endif
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
        worker->core_idx < 0 || static_cast<uint32_t>(worker->core_idx) >= pa_scheduler::kWorkers ||
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
        static_cast<uint32_t>(worker->core_idx) *
            pa_scheduler::winner_workload::kOutputTilesPerWorker +
        kind_slot;
    __gm__ float *output = reinterpret_cast<__gm__ float *>(
        workspace + static_cast<uint64_t>(output_tile) *
            pa_scheduler::winner_workload::kTileBytes
    );
#if defined(PA_BUILD_AIC)
    pa_scheduler_ccec::pa_real_cube_workload_aic(input_a, input_b, output, repeats);
#elif defined(PA_BUILD_AIV)
    if (kind == pa_scheduler::TaskKind::Sf) {
        pa_scheduler_ccec::pa_real_vector_add_workload_aiv(input_a, input_b, output, repeats);
    } else {
        pa_scheduler_ccec::pa_real_vector_mul_workload_aiv(input_a, input_b, output, repeats);
    }
#endif
}
#endif  // PA_CCEC_OPS_DEFINE_REAL_WORKLOAD

#endif  // TESTS_ATOMIC_PROBE_PA_SCHEDULER_CCEC_CCEC_OPS_H
