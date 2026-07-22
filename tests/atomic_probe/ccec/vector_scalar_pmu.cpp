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
// ccec_utils defines legacy sync flag macros, so include it only after PTO's
// same-named constexpr declarations have been parsed.
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(vector_scalar_pmu);

namespace {

using namespace pto;

constexpr uint32_t kPmuPhysicalSubcores = 108U;
constexpr uint64_t kPmuCounterBlockOffset = 0x4200ULL;
constexpr uint64_t kPmuSelectorBlockOffset = 0x2400ULL;

__aicore__ __attribute__((always_inline)) inline int32_t *PmuCounterBase(uint64_t register_base) {
    return reinterpret_cast<int32_t *>(register_base + kPmuCounterBlockOffset);
}

__aicore__ __attribute__((always_inline)) inline int32_t *PmuSelectorBase(uint64_t register_base) {
    return reinterpret_cast<int32_t *>(register_base + kPmuSelectorBlockOffset);
}

__aicore__ __attribute__((always_inline)) inline void ClearPmuCounters(uint64_t register_base) {
    int32_t *base = PmuCounterBase(register_base);
    (void)ld_dev(base, 0x10);
    (void)ld_dev(base, 0x18);
    (void)ld_dev(base, 0x20);
    (void)ld_dev(base, 0x28);
    (void)ld_dev(base, 0x30);
    (void)ld_dev(base, 0x38);
    (void)ld_dev(base, 0x40);
    (void)ld_dev(base, 0x48);
    (void)ld_dev(base, 0x50);
    (void)ld_dev(base, 0x54);
    (void)ld_dev(base, 0x60);
    (void)ld_dev(base, 0x64);
}

template <int16_t Offset>
__aicore__ __attribute__((always_inline)) inline uint64_t ReadCounter(uint64_t register_base) {
    return static_cast<uint32_t>(ld_dev(PmuCounterBase(register_base), Offset));
}

__aicore__ __attribute__((always_inline)) inline uint64_t ReadPmuTotal(uint64_t register_base) {
    int32_t *base = PmuCounterBase(register_base);
    const uint64_t low = static_cast<uint32_t>(ld_dev(base, 0x60));
    const uint64_t high = static_cast<uint32_t>(ld_dev(base, 0x64));
    return low | (high << 32U);
}

__aicore__ __attribute__((always_inline)) inline uint64_t ReadSelectorStatus(uint64_t register_base) {
    int32_t *base = PmuSelectorBase(register_base);
    uint64_t status = 0U;
    status |= static_cast<uint32_t>(ld_dev(base, 0x100)) == 0x501U ? vector_scalar_pmu::kSelectorVector : 0U;
    status |= static_cast<uint32_t>(ld_dev(base, 0x108)) == 0x001U ? vector_scalar_pmu::kSelectorScalar : 0U;
    status |= static_cast<uint32_t>(ld_dev(base, 0x110)) == 0x202U ? vector_scalar_pmu::kSelectorMte2 : 0U;
    status |= static_cast<uint32_t>(ld_dev(base, 0x114)) == 0x203U ? vector_scalar_pmu::kSelectorMte3 : 0U;
    status |= static_cast<uint32_t>(ld_dev(base, 0x118)) == 0x034U ? vector_scalar_pmu::kSelectorIcacheRequest : 0U;
    status |= static_cast<uint32_t>(ld_dev(base, 0x11c)) == 0x035U ? vector_scalar_pmu::kSelectorIcacheMiss : 0U;
    return status;
}

__aicore__ __attribute__((always_inline)) inline void Publish64(__gm__ uint64_t *address, uint64_t value) {
    __builtin_cce_st_dev(value, address, 0);
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
    if (state->control.pmu_register_bases != 0U && physical_core_id < kPmuPhysicalSubcores) {
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
        ClearPmuCounters(register_base);
    }

    const uint64_t sys_begin = static_cast<uint64_t>(get_sys_cnt());
    bisheng::cce::metrics_prof_start();

    if (mode == static_cast<uint32_t>(Mode::LoopControl)) {
        for (uint32_t iteration = 0U; iteration < rounds; ++iteration) {
            // Preserve one runtime loop iteration without issuing work to V,
            // MTE2 or MTE3. The NOP prevents the loop from being deleted.
            asm volatile("nop");
        }
    } else if (mode == static_cast<uint32_t>(Mode::VectorAdd)) {
        // Keep this body mechanically identical to RunRealVectorWorkload<false>
        // in pa_scheduler/ccec/ccec_ops.h.
        for (uint32_t iteration = 0U; iteration < rounds; ++iteration) {
            TLOAD(input_a_tile, input_a_global);
            TLOAD(input_b_tile, input_b_global);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TADD(output_tile, input_a_tile, input_b_tile);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(output_global, output_tile);
            set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
            wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        }
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
        vector_busy = ReadCounter<0x10>(register_base);
        scalar_busy = ReadCounter<0x20>(register_base);
        mte2_busy = ReadCounter<0x30>(register_base);
        mte3_busy = ReadCounter<0x38>(register_base);
        icache_request = ReadCounter<0x40>(register_base);
        icache_miss = ReadCounter<0x48>(register_base);
        total = ReadPmuTotal(register_base);
    }

    __gm__ vector_scalar_pmu::ProbeResult *result = &state->result;
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
