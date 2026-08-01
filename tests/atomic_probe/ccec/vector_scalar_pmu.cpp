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

// Single-AIV probe for the scalar-busy classification of the PA vector loop.
// The PMU owner configures the normal A5 PIPE_UTILIZATION map before launch:
//   CNT0=vector(0x501), CNT2=scalar(0x001), CNT4=MTE2(0x202),
//   CNT5=MTE3(0x203), CNT6=I-cache request(0x034), CNT7=I-cache miss(0x035).
// All counters are read-to-clear. Each kernel therefore clears the ten custom
// counters and total, opens one gate around exactly one selected workload, and
// publishes results only after metrics_prof_stop() has closed the gate.

#include <pto/common/constants.hpp>
#include <pto/common/kernel_meta.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#include "vector_scalar_pmu_shared.h"
#include "scalar_pmu_device.h"
// ccec_utils defines legacy sync flag macros, so include it only after PTO's
// same-named constexpr declarations have been parsed.
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(vector_scalar_pmu);

namespace {

using namespace pto;

__aicore__ __attribute__((always_inline)) inline uint64_t ReadSelectorStatus(uint64_t register_base) {
    const uint64_t common = atomic_probe::scalar_pmu::ReadSelectorStatus(
        register_base
    );
    uint64_t status = 0;
    status |= (common & atomic_probe::scalar_pmu::kSelectorVector) != 0
        ? vector_scalar_pmu::kSelectorVector : 0U;
    status |= (common & atomic_probe::scalar_pmu::kSelectorScalar) != 0
        ? vector_scalar_pmu::kSelectorScalar : 0U;
    status |= (common & atomic_probe::scalar_pmu::kSelectorMte2) != 0
        ? vector_scalar_pmu::kSelectorMte2 : 0U;
    status |= (common & atomic_probe::scalar_pmu::kSelectorMte3) != 0
        ? vector_scalar_pmu::kSelectorMte3 : 0U;
    status |= (common & atomic_probe::scalar_pmu::kSelectorIcacheRequest) != 0
        ? vector_scalar_pmu::kSelectorIcacheRequest : 0U;
    status |= (common & atomic_probe::scalar_pmu::kSelectorIcacheMiss) != 0
        ? vector_scalar_pmu::kSelectorIcacheMiss : 0U;
    return status;
}

template <typename GlobalData, typename TileData>
__aicore__ __attribute__((always_inline)) inline void IssueVectorAdd(
    GlobalData &input_a_global, GlobalData &input_b_global, GlobalData &output_global,
    TileData &input_a_tile, TileData &input_b_tile, TileData &output_tile
) {
    TLOAD(input_a_tile, input_a_global);
    TLOAD(input_b_tile, input_b_global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(output_tile, input_a_tile, input_b_tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(output_global, output_tile);
    // 这里只把 MTE3 完成事件发布给 scalar；调用方决定立即等待，还是
    // 先执行与输出无关的 scalar 工作。EVENT_ID7 在等待前不得复用。
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

__aicore__ __attribute__((always_inline)) inline void WaitVectorAddCompletion() {
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

// 串行对照与延后等待必须调用同一份机器码；若让 O3 分别内联，后端可能
// 对两个分支采用不同展开因子，把循环代码生成差异误判成流水重叠。
__aicore__ __attribute__((noinline, used)) void RunScalarNops(uint32_t rounds) {
    for (uint32_t iteration = 0U; iteration < rounds; ++iteration) {
        asm volatile("nop");
    }
}

}  // namespace

extern "C" __global__ __aicore__ void KERNEL_ENTRY(vector_scalar_pmu)(__gm__ vector_scalar_pmu::ProbeState *state) {
    using namespace pto;
    using vector_scalar_pmu::Mode;

    // Runtime profiling may leave the gate enabled before entry. Close it
    // first so control invalidation, object setup and counter clearing stay
    // outside the measured window.
    bisheng::cce::metrics_prof_stop();
    dcci(&state->control, SINGLE_CACHE_LINE);
    dsb(DSB_ALL);

    const uint32_t mode = state->control.mode;
    const uint32_t rounds = state->control.rounds;
    const uint32_t physical_core_id = static_cast<uint32_t>(get_coreid()) & 0x0fffU;

    uint64_t register_base = 0U;
    if (state->control.pmu_register_bases != 0U &&
        physical_core_id < atomic_probe::scalar_pmu::kPhysicalSubcores) {
        __gm__ uint64_t *register_bases = reinterpret_cast<__gm__ uint64_t *>(state->control.pmu_register_bases);
        register_base = register_bases[physical_core_id];
    }

    constexpr int kRows = static_cast<int>(vector_scalar_pmu::kTileRows);
    constexpr int kCols = static_cast<int>(vector_scalar_pmu::kTileCols);
    using GlobalData = GlobalTensor<float, Shape<1, 1, 1, kRows, kCols>, pto::Stride<1, 1, 1, kCols, 1>>;
    using TileData = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;

    GlobalData input_a_global(reinterpret_cast<__gm__ float *>(state->control.input_a));
    GlobalData input_b_global(reinterpret_cast<__gm__ float *>(state->control.input_b));
    GlobalData output_global(reinterpret_cast<__gm__ float *>(state->control.output));
    TileData input_a_tile(kRows, kCols);
    TileData input_b_tile(kRows, kCols);
    TileData output_tile(kRows, kCols);
    TASSIGN(input_a_tile, 0x0);
    TASSIGN(input_b_tile, 0x10000);
    TASSIGN(output_tile, 0x20000);

    uint64_t selector_status = 0U;
    if (register_base != 0U) {
        selector_status = ReadSelectorStatus(register_base);
        atomic_probe::scalar_pmu::ClearCounters(register_base);
    }

    const uint64_t sys_begin = static_cast<uint64_t>(get_sys_cnt());
    bisheng::cce::metrics_prof_start();

    if (mode == static_cast<uint32_t>(Mode::LoopControl)) {
        // Preserve one runtime loop without issuing work to V, MTE2 or MTE3.
        RunScalarNops(rounds);
    } else if (mode == static_cast<uint32_t>(Mode::VectorAdd)) {
        // Keep this body mechanically identical to RunRealVectorWorkload<false>
        // in pa_scheduler/ccec/ccec_ops.h.
        for (uint32_t iteration = 0U; iteration < rounds; ++iteration) {
            IssueVectorAdd(
                input_a_global, input_b_global, output_global,
                input_a_tile, input_b_tile, output_tile
            );
            WaitVectorAddCompletion();
        }
    } else if (mode == static_cast<uint32_t>(Mode::VectorThenScalar)) {
        IssueVectorAdd(
            input_a_global, input_b_global, output_global,
            input_a_tile, input_b_tile, output_tile
        );
        WaitVectorAddCompletion();
        RunScalarNops(rounds);
    } else if (mode == static_cast<uint32_t>(Mode::VectorOverlapScalar)) {
        IssueVectorAdd(
            input_a_global, input_b_global, output_global,
            input_a_tile, input_b_tile, output_tile
        );
        RunScalarNops(rounds);
        WaitVectorAddCompletion();
    }

    bisheng::cce::metrics_prof_stop();
    const uint64_t sys_end = static_cast<uint64_t>(get_sys_cnt());
    const uint64_t ctrl_after_stop = static_cast<uint64_t>(get_ctrl());

    uint64_t total = 0U;
    uint64_t vector_busy = 0U;
    uint64_t scalar_busy = 0U;
    uint64_t mte2_busy = 0U;
    uint64_t mte3_busy = 0U;
    uint64_t icache_request = 0U;
    uint64_t icache_miss = 0U;
    if (register_base != 0U) {
        const auto snapshot =
            atomic_probe::scalar_pmu::ReadSnapshot(register_base);
        vector_busy = snapshot.vector_busy;
        scalar_busy = snapshot.scalar_busy;
        mte2_busy = snapshot.mte2_busy;
        mte3_busy = snapshot.mte3_busy;
        icache_request = snapshot.icache_request;
        icache_miss = snapshot.icache_miss;
        total = snapshot.total;
    }

    __gm__ vector_scalar_pmu::ProbeResult *result = &state->result;
    using atomic_probe::scalar_pmu::Publish64;
    Publish64(&result->sys_ticks, sys_end - sys_begin);
    Publish64(&result->pmu_total_cycles, total);
    Publish64(&result->pmu_vector_busy, vector_busy);
    Publish64(&result->pmu_scalar_busy, scalar_busy);
    Publish64(&result->pmu_mte2_busy, mte2_busy);
    Publish64(&result->pmu_mte3_busy, mte3_busy);
    Publish64(&result->pmu_icache_request, icache_request);
    Publish64(&result->pmu_icache_miss, icache_miss);
    Publish64(&result->physical_core_id, physical_core_id);
    Publish64(&result->pmu_ctrl_after_stop, ctrl_after_stop);
    Publish64(&result->selector_status, selector_status);
    Publish64(&result->observed_mode, mode);
    Publish64(&result->observed_rounds, rounds);
    dsb(DSB_ALL);
}
