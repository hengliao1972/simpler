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

// Pure CCEC A5 cache-preload usage probe. This deliberately uses the compiler
// interfaces directly:
//
//   dc_preload(__gm__ uint64_t *base, int64_t byte_offset)
//   icache_preload(int64_t units_from_current_pc)
//   get_icache_prl_st()
//
// Preload is only a performance hint. It is not DCCI, a memory-ordering
// primitive, cross-core publication, or synchronization.

#include "../cache_preload_shared.h"
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(cache_preload);

namespace {

__aicore__ __attribute__((always_inline)) inline uint64_t ReadOrderedCycle() {
    uint64_t cycle = 0;
    asm volatile("MOV %0, SYS_CNT\n" : "=&l"(cycle) : : "memory");
    return cycle;
}

// The tied operand makes the value under test a compiler dependency of the
// SYS_CNT boundary. These helpers are measurement machinery, not a hardware
// barrier or a cache-coherence operation.
__aicore__ __attribute__((always_inline)) inline uint64_t CycleBeforeValue(uint64_t &value) {
    uint64_t cycle = 0;
    asm volatile("MOV %1, SYS_CNT\n"
                 "MOV %0, %0\n"
                 : "+l"(value), "=&l"(cycle)
                 :
                 : "memory");
    return cycle;
}

__aicore__ __attribute__((always_inline)) inline uint64_t CycleAfterValue(uint64_t &value) {
    uint64_t cycle = 0;
    asm volatile("MOV %0, %0\n"
                 "MOV %1, SYS_CNT\n"
                 : "+l"(value), "=&l"(cycle)
                 :
                 : "memory");
    return cycle;
}

__aicore__ __attribute__((always_inline)) inline uint64_t OpaqueIdentity(uint64_t value) {
    asm volatile("MOV %0, %0\n" : "+l"(value) : : "memory");
    return value;
}

__aicore__ __attribute__((always_inline)) inline void PublishResult(
    __gm__ cache_preload::ProbeResult *result, uint64_t value, uint64_t preparation_checksum, uint64_t gap_checksum,
    uint64_t issue_ticks, uint64_t access_or_work_ticks, uint64_t total_ticks, uint64_t immediate_status,
    uint64_t final_status, uint64_t polls, uint32_t mode, uint32_t target_word, uint32_t gap_rounds
) {
    st_dev_b64(&result->value, value);
    st_dev_b64(&result->preparation_checksum, preparation_checksum);
    st_dev_b64(&result->gap_checksum, gap_checksum);
    st_dev_b64(&result->issue_ticks, issue_ticks);
    st_dev_b64(&result->access_or_work_ticks, access_or_work_ticks);
    st_dev_b64(&result->total_ticks, total_ticks);
    st_dev_b64(&result->icache_immediate_status, immediate_status);
    st_dev_b64(&result->icache_final_status, final_status);
    st_dev_b64(&result->icache_polls, polls);
    st_dev_b64(&result->mode_echo, static_cast<uint64_t>(mode));
    st_dev_b64(&result->target_word_echo, static_cast<uint64_t>(target_word));
    st_dev_b64(&result->gap_rounds_echo, static_cast<uint64_t>(gap_rounds));
}

}  // namespace

// Keep the independent gap in a separate small function. While this function
// runs, current-PC ICache preload can work on the sequential code that follows
// its call site in cache_preload_icache_path.
extern "C" __aicore__ __attribute__((noinline, used)) uint64_t cache_preload_gap(uint64_t seed, uint32_t gap_rounds) {
    return cache_preload::GapOracle(seed, gap_rounds);
}

#define CACHE_PRELOAD_NOPS_1() asm volatile("nop");
#define CACHE_PRELOAD_NOPS_2() CACHE_PRELOAD_NOPS_1() CACHE_PRELOAD_NOPS_1()
#define CACHE_PRELOAD_NOPS_4() CACHE_PRELOAD_NOPS_2() CACHE_PRELOAD_NOPS_2()
#define CACHE_PRELOAD_NOPS_8() CACHE_PRELOAD_NOPS_4() CACHE_PRELOAD_NOPS_4()
#define CACHE_PRELOAD_NOPS_16() CACHE_PRELOAD_NOPS_8() CACHE_PRELOAD_NOPS_8()
#define CACHE_PRELOAD_NOPS_32() CACHE_PRELOAD_NOPS_16() CACHE_PRELOAD_NOPS_16()
#define CACHE_PRELOAD_NOPS_64() CACHE_PRELOAD_NOPS_32() CACHE_PRELOAD_NOPS_32()
#define CACHE_PRELOAD_NOPS_128() CACHE_PRELOAD_NOPS_64() CACHE_PRELOAD_NOPS_64()
#define CACHE_PRELOAD_NOPS_256() CACHE_PRELOAD_NOPS_128() CACHE_PRELOAD_NOPS_128()
#define CACHE_PRELOAD_NOPS_512() CACHE_PRELOAD_NOPS_256() CACHE_PRELOAD_NOPS_256()
#define CACHE_PRELOAD_NOPS_1024() CACHE_PRELOAD_NOPS_512() CACHE_PRELOAD_NOPS_512()
#define CACHE_PRELOAD_NOPS_2048() CACHE_PRELOAD_NOPS_1024() CACHE_PRELOAD_NOPS_1024()
#define CACHE_PRELOAD_NOPS_4096() CACHE_PRELOAD_NOPS_2048() CACHE_PRELOAD_NOPS_2048()
#define CACHE_PRELOAD_NOPS_8192() CACHE_PRELOAD_NOPS_4096() CACHE_PRELOAD_NOPS_4096()

// The evictor is deliberately larger than the documented 16 KiB AIV ICache.
// The build script validates its final linked symbol size and non-overlap with
// the measured current-PC path.
extern "C" __aicore__ __attribute__((noinline, used, aligned(128))) uint64_t cache_preload_icache_evictor(uint64_t seed
) {
    CACHE_PRELOAD_NOPS_8192()
    return cache_preload::ICacheEvictorOracle(seed);
}

// Baseline/async/wait share this exact physical NOP region. icache_preload(2)
// uses the current PC, then the noinline gap provides an overlap window before
// execution reaches the upcoming sequential region.
extern "C" __aicore__ __attribute__((noinline, used, aligned(128))) void cache_preload_icache_path(
    uint32_t mode_value, uint64_t seed, uint32_t target_word, uint32_t gap_rounds, uint64_t preparation_checksum,
    __gm__ cache_preload::ProbeResult *result
) {
    const uint64_t total_begin = ReadOrderedCycle();
    uint64_t issue_ticks = 0;
    uint64_t immediate_status = 0;
    uint64_t final_status = 0;
    uint64_t polls = 0;

    const bool use_preload = mode_value == static_cast<uint32_t>(cache_preload::Mode::ICacheCurrentPcAsync) ||
                             mode_value == static_cast<uint32_t>(cache_preload::Mode::ICacheCurrentPcWait);
    if (use_preload) {
        const uint64_t issue_begin = ReadOrderedCycle();
        icache_preload(static_cast<int64_t>(cache_preload::kICachePreloadUnits));
        const uint64_t issue_end = ReadOrderedCycle();
        issue_ticks = issue_end - issue_begin;

        immediate_status = static_cast<uint64_t>(get_icache_prl_st());
        final_status = immediate_status;
        if (mode_value == static_cast<uint32_t>(cache_preload::Mode::ICacheCurrentPcWait)) {
            while (final_status != 0 && polls < cache_preload::kICachePollLimit) {
                CACHE_PRELOAD_NOPS_16()
                final_status = static_cast<uint64_t>(get_icache_prl_st());
                ++polls;
            }
        }
    }

    const uint64_t gap_checksum = cache_preload_gap(seed, gap_rounds);
    if (use_preload) {
        final_status = static_cast<uint64_t>(get_icache_prl_st());
    }

    uint64_t work_value = seed ^ gap_checksum;
    const uint64_t work_begin = CycleBeforeValue(work_value);
    CACHE_PRELOAD_NOPS_1024()
    work_value = cache_preload::ICacheTargetOracle(work_value);
    const uint64_t work_end = CycleAfterValue(work_value);

    PublishResult(
        result, work_value, preparation_checksum, gap_checksum, issue_ticks, work_end - work_begin,
        work_end - total_begin, immediate_status, final_status, polls, mode_value, target_word, gap_rounds
    );
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(cache_preload)(__gm__ cache_preload::ProbeState *state) {
    if (get_block_idx() != 0) return;

    // Host rewrites only this line for each launch. Invalidate and complete it
    // before consuming the runtime mode/offset/seed.
    dcci(&state->control, SINGLE_CACHE_LINE);
    dsb(DSB_ALL);

    const uint32_t mode_value = state->control.mode;
    const uint32_t target_word = state->control.target_word;
    const uint32_t gap_rounds = state->control.gap_rounds;
    const uint64_t seed = state->control.seed;
    __gm__ cache_preload::ProbeResult *result = &state->result;

    if (mode_value == static_cast<uint32_t>(cache_preload::Mode::DCacheBaseline) ||
        mode_value == static_cast<uint32_t>(cache_preload::Mode::DCachePreload)) {
        __gm__ uint64_t *target = &state->data[target_word];

        // Establish a deterministic cold local-DCache line outside the timed
        // interval. This DCCI is measurement setup, not part of preload usage.
        dcci(target, SINGLE_CACHE_LINE);
        dsb(DSB_ALL);

        const uint64_t total_begin = ReadOrderedCycle();
        uint64_t issue_ticks = 0;
        if (mode_value == static_cast<uint32_t>(cache_preload::Mode::DCachePreload)) {
            const uint64_t issue_begin = ReadOrderedCycle();
            const int64_t byte_offset = static_cast<int64_t>(target_word) * sizeof(uint64_t);
            dc_preload(state->data, byte_offset);
            const uint64_t issue_end = ReadOrderedCycle();
            issue_ticks = issue_end - issue_begin;
        }

        const uint64_t gap_checksum = cache_preload_gap(seed, gap_rounds);

        // OpaqueIdentity is MOV-at-runtime, so the address delta is exactly
        // zero on hardware while still creating a compiler dependency from
        // the independent gap to the ordinary target load.
        const uint64_t opaque_gap = OpaqueIdentity(gap_checksum);
        const uint64_t address_delta = opaque_gap - gap_checksum;
        uint64_t target_address = reinterpret_cast<uint64_t>(target) + address_delta;
        const uint64_t access_begin = CycleBeforeValue(target_address);
        volatile __gm__ uint64_t *volatile_target = reinterpret_cast<volatile __gm__ uint64_t *>(target_address);
        uint64_t value = *volatile_target;
        const uint64_t access_end = CycleAfterValue(value);

        PublishResult(
            result, value, 0, gap_checksum, issue_ticks, access_end - access_begin, access_end - total_begin, 0, 0, 0,
            mode_value, target_word, gap_rounds
        );
        return;
    }

    if (mode_value == static_cast<uint32_t>(cache_preload::Mode::ICacheColdBaseline) ||
        mode_value == static_cast<uint32_t>(cache_preload::Mode::ICacheCurrentPcAsync) ||
        mode_value == static_cast<uint32_t>(cache_preload::Mode::ICacheCurrentPcWait)) {
        const uint64_t preparation_checksum = cache_preload_icache_evictor(seed);
        cache_preload_icache_path(mode_value, seed, target_word, gap_rounds, preparation_checksum, result);
    }
}

#undef CACHE_PRELOAD_NOPS_8192
#undef CACHE_PRELOAD_NOPS_4096
#undef CACHE_PRELOAD_NOPS_2048
#undef CACHE_PRELOAD_NOPS_1024
#undef CACHE_PRELOAD_NOPS_512
#undef CACHE_PRELOAD_NOPS_256
#undef CACHE_PRELOAD_NOPS_128
#undef CACHE_PRELOAD_NOPS_64
#undef CACHE_PRELOAD_NOPS_32
#undef CACHE_PRELOAD_NOPS_16
#undef CACHE_PRELOAD_NOPS_8
#undef CACHE_PRELOAD_NOPS_4
#undef CACHE_PRELOAD_NOPS_2
#undef CACHE_PRELOAD_NOPS_1
