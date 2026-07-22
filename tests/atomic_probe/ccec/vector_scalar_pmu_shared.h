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

#ifndef TESTS_ATOMIC_PROBE_CCEC_VECTOR_SCALAR_PMU_SHARED_H_
#define TESTS_ATOMIC_PROBE_CCEC_VECTOR_SCALAR_PMU_SHARED_H_

#include <cstddef>
#include <cstdint>

namespace vector_scalar_pmu {

constexpr uint32_t kTileRows = 128U;
constexpr uint32_t kTileCols = 128U;
constexpr uint32_t kTileElements = kTileRows * kTileCols;
constexpr uint32_t kTileBytes = kTileElements * sizeof(float);

// EMPTY measures only the PMU gate. LOOP_CONTROL keeps the same runtime loop
// shape but replaces the pipeline body with one scalar NOP. VECTOR_ADD is the
// exact TLOAD -> MTE2/V -> TADD -> V/MTE3 -> TSTORE -> MTE3/S sequence used by
// the pa_scheduler AIV real workload.
enum class Mode : uint32_t {
    Empty = 0U,
    LoopControl = 1U,
    VectorAdd = 2U,
    Count = 3U,
};

struct alignas(64) ProbeControl {
    uint64_t pmu_register_bases;
    uint64_t input_a;
    uint64_t input_b;
    uint64_t output;
    uint32_t mode;
    uint32_t rounds;
    uint64_t reserved[3];
};

// Every field is widened to 64 bits in the shared ABI even though the custom
// PMU counters are 32 bits. This keeps host formatting unambiguous and gives
// the result two complete cache lines that are published only after PMU stop.
struct alignas(64) ProbeResult {
    uint64_t sys_ticks;
    uint64_t pmu_total_cycles;
    uint64_t pmu_vector_busy;
    uint64_t pmu_scalar_busy;
    uint64_t pmu_mte2_busy;
    uint64_t pmu_mte3_busy;
    uint64_t pmu_icache_request;
    uint64_t pmu_icache_miss;

    uint64_t physical_core_id;
    uint64_t pmu_ctrl_after_stop;
    uint64_t selector_status;
    uint64_t observed_mode;
    uint64_t observed_rounds;
    uint64_t reserved[3];
};

struct alignas(64) ProbeState {
    ProbeControl control;
    ProbeResult result;
};

constexpr uint64_t kSelectorVector = 1ULL << 0;
constexpr uint64_t kSelectorScalar = 1ULL << 1;
constexpr uint64_t kSelectorMte2 = 1ULL << 2;
constexpr uint64_t kSelectorMte3 = 1ULL << 3;
constexpr uint64_t kSelectorIcacheRequest = 1ULL << 4;
constexpr uint64_t kSelectorIcacheMiss = 1ULL << 5;
constexpr uint64_t kRequiredSelectorStatus =
    kSelectorVector | kSelectorScalar | kSelectorMte2 | kSelectorMte3 | kSelectorIcacheRequest | kSelectorIcacheMiss;

static_assert(kTileBytes == 65536U, "the PA vector tile must remain 64 KiB");
static_assert(sizeof(ProbeControl) == 64U, "probe control must occupy one cache line");
static_assert(sizeof(ProbeResult) == 128U, "probe result must occupy two cache lines");
static_assert(offsetof(ProbeState, result) == 64U, "probe result offset changed");
static_assert(sizeof(ProbeState) == 192U, "probe state ABI changed");

}  // namespace vector_scalar_pmu

#endif  // TESTS_ATOMIC_PROBE_CCEC_VECTOR_SCALAR_PMU_SHARED_H_
