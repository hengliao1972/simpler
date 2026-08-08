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

#include <pto/common/kernel_meta.hpp>

#include "cce_aicore_intrinsics.h"
#include "simt_api/asc_simt.h"

#include "../common/simt_stack_probe.h"

PTO_SYNCALL_AIV_KERNEL_META(simt_cross_core_simt_stack_0_mix_aiv);

namespace {

using namespace pa_scheduler::simt_cross_core::simt_stack;

constexpr int kSingleCacheLine = 0;

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtRotateLeft(uint64_t value, uint32_t shift) {
    return (value << shift) | (value >> (64U - shift));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtFrameStep(uint64_t value, uint32_t frame, uint32_t word) {
    value ^= UINT64_C(0x9E3779B97F4A7C15) + (static_cast<uint64_t>(frame) << 32U) + word;
    return SimtRotateLeft(value, (frame * 11U + word * 7U) % 63U + 1U) + UINT64_C(0xD1B54A32D192ED03);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtDivergencePressure(uint64_t value, uint32_t lane) {
#define SIMT_STACK_OPEN(level, threshold)                                                                                \
    if (lane != threshold) {                                                                                            \
        value = SimtFrameStep(value, 32U + level, threshold);
#define SIMT_STACK_CLOSE(level, threshold)                                                                               \
    }                                                                                                                    \
    else {                                                                                                               \
        value = SimtFrameStep(value, 96U + level, threshold);                                                           \
    }                                                                                                                    \
    value = SimtFrameStep(value, 64U + level, threshold);

    SIMT_STACK_OPEN(0U, 31U)
    SIMT_STACK_OPEN(1U, 30U)
    SIMT_STACK_OPEN(2U, 29U)
    SIMT_STACK_OPEN(3U, 28U)
    SIMT_STACK_OPEN(4U, 27U)
    SIMT_STACK_OPEN(5U, 26U)
    SIMT_STACK_OPEN(6U, 25U)
    SIMT_STACK_OPEN(7U, 24U)
    SIMT_STACK_OPEN(8U, 23U)
    SIMT_STACK_OPEN(9U, 22U)
    SIMT_STACK_OPEN(10U, 21U)
    SIMT_STACK_OPEN(11U, 20U)
    SIMT_STACK_OPEN(12U, 19U)
    SIMT_STACK_OPEN(13U, 18U)
    SIMT_STACK_OPEN(14U, 17U)
    SIMT_STACK_OPEN(15U, 16U)
    SIMT_STACK_OPEN(16U, 15U)
    SIMT_STACK_OPEN(17U, 14U)
    SIMT_STACK_OPEN(18U, 13U)
    SIMT_STACK_OPEN(19U, 12U)
    SIMT_STACK_OPEN(20U, 11U)
    SIMT_STACK_OPEN(21U, 10U)
    SIMT_STACK_OPEN(22U, 9U)
    SIMT_STACK_OPEN(23U, 8U)
    SIMT_STACK_OPEN(24U, 7U)
    SIMT_STACK_OPEN(25U, 6U)
    SIMT_STACK_OPEN(26U, 5U)
    SIMT_STACK_OPEN(27U, 4U)
    SIMT_STACK_OPEN(28U, 3U)
    SIMT_STACK_OPEN(29U, 2U)
    SIMT_STACK_OPEN(30U, 1U)
    value = SimtFrameStep(value, 63U, 0U);
    SIMT_STACK_CLOSE(30U, 1U)
    SIMT_STACK_CLOSE(29U, 2U)
    SIMT_STACK_CLOSE(28U, 3U)
    SIMT_STACK_CLOSE(27U, 4U)
    SIMT_STACK_CLOSE(26U, 5U)
    SIMT_STACK_CLOSE(25U, 6U)
    SIMT_STACK_CLOSE(24U, 7U)
    SIMT_STACK_CLOSE(23U, 8U)
    SIMT_STACK_CLOSE(22U, 9U)
    SIMT_STACK_CLOSE(21U, 10U)
    SIMT_STACK_CLOSE(20U, 11U)
    SIMT_STACK_CLOSE(19U, 12U)
    SIMT_STACK_CLOSE(18U, 13U)
    SIMT_STACK_CLOSE(17U, 14U)
    SIMT_STACK_CLOSE(16U, 15U)
    SIMT_STACK_CLOSE(15U, 16U)
    SIMT_STACK_CLOSE(14U, 17U)
    SIMT_STACK_CLOSE(13U, 18U)
    SIMT_STACK_CLOSE(12U, 19U)
    SIMT_STACK_CLOSE(11U, 20U)
    SIMT_STACK_CLOSE(10U, 21U)
    SIMT_STACK_CLOSE(9U, 22U)
    SIMT_STACK_CLOSE(8U, 23U)
    SIMT_STACK_CLOSE(7U, 24U)
    SIMT_STACK_CLOSE(6U, 25U)
    SIMT_STACK_CLOSE(5U, 26U)
    SIMT_STACK_CLOSE(4U, 27U)
    SIMT_STACK_CLOSE(3U, 28U)
    SIMT_STACK_CLOSE(2U, 29U)
    SIMT_STACK_CLOSE(1U, 30U)
    SIMT_STACK_CLOSE(0U, 31U)

#undef SIMT_STACK_CLOSE
#undef SIMT_STACK_OPEN
    return value;
}

__simt_callee__ __aicore__ __attribute__((noinline)) uint64_t
StackFrame1(uint64_t value) {
    volatile uint64_t words[kFrameWords];
    for (uint32_t word = 0U; word < kFrameWords; ++word) {
        value = SimtFrameStep(value, 1U, word);
        words[word] = value;
    }
    for (uint32_t word = 0U; word < kFrameWords; ++word) {
        value ^= words[word] ^ words[word];
    }
    return value;
}

__simt_callee__ __aicore__ __attribute__((noinline)) uint64_t
StackFrame0(uint64_t value) {
    volatile uint64_t words[kFrameWords];
    value = StackFrame1(value);
    for (uint32_t word = 0U; word < kFrameWords; ++word) {
        value = SimtFrameStep(value, 0U, word);
        words[word] = value;
    }
    for (uint32_t word = 0U; word < kFrameWords; ++word) {
        value ^= words[word] ^ words[word];
    }
    return value;
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(kLaunchThreads) void
SimtStackProbe(__gm__ uint64_t *thread_checksums, __gm__ WarpReport *reports, uint64_t nonce) {
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    const uint32_t warp = thread / kWarpSize;
    const uint32_t lane = thread % kWarpSize;
    if (warp >= kWarpCount) {
        return;
    }

    const uint64_t begin_clock = clock();
    uint64_t checksum = nonce ^ (static_cast<uint64_t>(thread) << 32U) ^ UINT64_C(0x243F6A8885A308D3);
    checksum = SimtDivergencePressure(checksum, lane);
    if (lane < kWarpSize / 2U) {
        checksum = StackFrame0(checksum);
    } else {
        checksum = StackFrame1(checksum);
        checksum = SimtFrameStep(checksum, 127U, lane);
    }
    const uint64_t end_clock = clock();
    asc_stcg(thread_checksums + thread, checksum);
    asc_threadfence();
    if (lane != 0U) {
        return;
    }

    __gm__ uint64_t *report = reinterpret_cast<__gm__ uint64_t *>(&reports[warp]);
    asc_stcg(report + 1U, nonce);
    asc_stcg(report + 2U, static_cast<uint64_t>(thread) | (static_cast<uint64_t>(warp) << 32U));
    asc_stcg(report + 3U, static_cast<uint64_t>(lane) | (static_cast<uint64_t>(kFrameWords) << 32U));
    asc_stcg(report + 4U, checksum);
    asc_stcg(report + 5U, begin_clock);
    asc_stcg(report + 6U, end_clock);
    asc_stcg(report + 7U, static_cast<uint64_t>(0U));
    asc_threadfence();
    asc_stcg(report, kReportMagic);
    asc_threadfence();
}

__aicore__ __attribute__((always_inline)) inline uint64_t LoadDev64(__gm__ uint64_t *address) {
    return static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0));
}

__aicore__ __attribute__((always_inline)) inline void StoreDev64(__gm__ uint64_t *address, uint64_t value) {
    __builtin_cce_st_dev(value, address, 0);
}

}  // namespace

extern "C" __global__ __aicore__ void
simt_cross_core_simt_stack_0_mix_aiv(__gm__ pa_scheduler::simt_cross_core::simt_stack::ProbeState *state) {
    using namespace pa_scheduler::simt_cross_core::simt_stack;

    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dsb(DSB_ALL);
    const bool config_valid = state->control.magic == kProbeMagic && state->control.version == kProbeVersion &&
                              state->control.launch_threads == kLaunchThreads &&
                              state->control.warp_count == kWarpCount && state->control.frame_words == kFrameWords &&
                              state->control.frame_depth == kFrameDepth;
    if (!config_valid) {
        return;
    }

    cce::async_invoke<SimtStackProbe>(
        cce::dim3{kLaunchThreads, 1U, 1U}, &state->thread_checksums[0], &state->reports[0],
        state->control.launch_nonce
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    uint32_t completed_warps = 0U;
    uint64_t checksum_xor = 0U;
    uint64_t status = 1U;
    for (uint32_t warp = 0U; warp < kWarpCount; ++warp) {
        __gm__ uint64_t *report = reinterpret_cast<__gm__ uint64_t *>(&state->reports[warp]);
        if (LoadDev64(report + offsetof(WarpReport, marker) / sizeof(uint64_t)) != kReportMagic ||
            LoadDev64(report + offsetof(WarpReport, launch_nonce) / sizeof(uint64_t)) != state->control.launch_nonce) {
            status = 0U;
            continue;
        }
        ++completed_warps;
        checksum_xor ^= LoadDev64(report + offsetof(WarpReport, checksum) / sizeof(uint64_t));
    }
    __gm__ uint64_t *destination = reinterpret_cast<__gm__ uint64_t *>(&state->result);
    StoreDev64(destination + 0U, kResultMagic);
    StoreDev64(destination + 1U, state->control.launch_nonce);
    StoreDev64(
        destination + 2U, static_cast<uint64_t>(completed_warps) | (static_cast<uint64_t>(kWarpCount) << 32U)
    );
    StoreDev64(destination + 3U, checksum_xor);
    StoreDev64(destination + 4U, status);
    StoreDev64(destination + 5U, 0U);
    StoreDev64(destination + 6U, 0U);
    StoreDev64(destination + 7U, 0U);
    dsb(DSB_ALL);
}
