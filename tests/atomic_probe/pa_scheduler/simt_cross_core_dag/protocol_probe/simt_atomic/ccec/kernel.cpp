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

// 本文件只验证 A5 SIMT thread 对 GM uint64_t CAS/add 的同地址竞争语义。
// 它不验证 UBUF atomic，也不把结果外推到其他数据类型或 atomic 操作。

#include <pto/common/kernel_meta.hpp>

#include "cce_aicore_intrinsics.h"
#include "simt_api/asc_simt.h"

#include "../common/atomic_probe.h"

PTO_SYNCALL_AIV_KERNEL_META(simt_cross_core_atomic_0_mix_aiv);

namespace {

using namespace pa_scheduler::simt_cross_core;
using namespace pa_scheduler::simt_cross_core::simt_atomic;

constexpr int kSingleCacheLine = 0;

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtRotateLeft(uint64_t value, uint32_t shift) {
    return (value << shift) | (value >> (64U - shift));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtExpectedCasDesired(uint64_t nonce, uint32_t thread) {
    return 0xA700000000000000ULL | ((nonce >> 8U) & 0x0000FFFFFFFFF000ULL) | static_cast<uint64_t>(thread + 1U);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtExpectedThreadMarker(uint64_t nonce, uint32_t thread) {
    return 0x5A00000000000000ULL ^ SimtRotateLeft(nonce, (thread % 29U) + 1U) ^
           (0x0010001000100010ULL + static_cast<uint64_t>(thread));
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(kMaxThreadCount) void SimtAtomicContention(
    __gm__ uint64_t *cas_cell, __gm__ uint64_t *add_cell, __gm__ uint64_t *cas_returns, __gm__ uint64_t *add_returns,
    __gm__ uint64_t *markers, uint64_t launch_nonce, uint64_t cas_initial, uint64_t thread_count
) {
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    if (thread >= thread_count) {
        return;
    }

    const uint64_t desired = SimtExpectedCasDesired(launch_nonce, thread);
    const uint64_t cas_observed = asc_atomic_cas(cas_cell, cas_initial, desired);
    const uint64_t add_ticket = asc_atomic_add(add_cell, static_cast<uint64_t>(1U));
    cas_returns[thread] = cas_observed;
    add_returns[thread] = add_ticket;
    markers[thread] = SimtExpectedThreadMarker(launch_nonce, thread);
    asc_threadfence();
}

__aicore__ __attribute__((always_inline)) inline uint64_t LoadDev64(__gm__ uint64_t *address) {
    return static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0));
}

__aicore__ __attribute__((always_inline)) inline void StoreDev64(__gm__ uint64_t *address, uint64_t value) {
    __builtin_cce_st_dev(value, address, 0);
}

__aicore__ void PublishResult(__gm__ ProbeResult *result, const ProbeResult &local) {
    __gm__ uint64_t *destination = reinterpret_cast<__gm__ uint64_t *>(result);
    const uint64_t *source = reinterpret_cast<const uint64_t *>(&local);
    for (uint32_t word = 0U; word < sizeof(ProbeResult) / sizeof(uint64_t); ++word) {
        StoreDev64(destination + word, source[word]);
    }
    dsb(DSB_ALL);
}

__aicore__ bool SupportedThreadCount(uint64_t thread_count) {
    for (uint32_t index = 0U; index < kThreadConfigCount; ++index) {
        if (thread_count == kThreadConfigs[index]) {
            return true;
        }
    }
    return false;
}

}  // namespace

extern "C" __global__ __aicore__ void
simt_cross_core_atomic_0_mix_aiv(__gm__ pa_scheduler::simt_cross_core::simt_atomic::ProbeState *state) {
    using namespace pa_scheduler::simt_cross_core::simt_atomic;

    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dsb(DSB_ALL);

    ProbeResult result{};
    result.magic = kResultMagic;
    result.physical_core_id = static_cast<uint64_t>(get_coreid()) & 0x0FFFU;
    result.subblock_id = static_cast<uint64_t>(get_subblockid());
    result.launch_nonce = state->control.launch_nonce;
    result.requested_thread_count = state->control.thread_count;
    result.warp_size = kWarpSize;
    result.cas_initial = state->control.cas_initial;
    result.add_initial = state->control.add_initial;

    const bool config_valid = state->control.magic == kProbeMagic && state->control.version == kProbeVersion &&
                              SupportedThreadCount(state->control.thread_count) &&
                              state->control.cas_initial == ExpectedCasInitial(state->control.launch_nonce) &&
                              state->control.add_initial == ExpectedAddInitial(state->control.launch_nonce);
    if (!config_valid) {
        PublishResult(&state->result, result);
        return;
    }
    result.status |= kStatusConfigValid;

    cce::async_invoke<SimtAtomicContention>(
        cce::dim3{static_cast<uint32_t>(state->control.thread_count), 1U, 1U}, &state->cas_cell.value,
        &state->add_cell.value, &state->cas_returns.values[0], &state->add_returns.values[0], &state->markers.values[0],
        state->control.launch_nonce, state->control.cas_initial, state->control.thread_count
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    result.cas_final = LoadDev64(&state->cas_cell.value);
    result.add_final = LoadDev64(&state->add_cell.value);
    bool final_matches_desired = false;
    uint64_t add_ticket_sum = 0U;
    for (uint32_t thread = 0U; thread < state->control.thread_count; ++thread) {
        const uint64_t cas_observed = LoadDev64(&state->cas_returns.values[thread]);
        const uint64_t add_ticket = LoadDev64(&state->add_returns.values[thread]);
        const uint64_t marker = LoadDev64(&state->markers.values[thread]);
        result.cas_winner_count += cas_observed == result.cas_initial ? 1U : 0U;
        result.cas_loser_count += cas_observed == result.cas_final ? 1U : 0U;
        result.add_ticket_in_range +=
            add_ticket >= result.add_initial && add_ticket < result.add_initial + state->control.thread_count ? 1U : 0U;
        add_ticket_sum += add_ticket;
        result.observed_thread_count += marker == ExpectedThreadMarker(state->control.launch_nonce, thread) ? 1U : 0U;
        final_matches_desired =
            final_matches_desired || result.cas_final == ExpectedCasDesired(state->control.launch_nonce, thread);
    }
    result.add_ticket_sum = add_ticket_sum;

    if (result.observed_thread_count == state->control.thread_count) {
        result.status |= kStatusAllThreadsObserved;
    }
    if (result.cas_winner_count == 1U) {
        result.status |= kStatusCasUniqueWinner;
    }
    if (result.cas_loser_count + result.cas_winner_count == state->control.thread_count) {
        result.status |= kStatusCasReturnValues;
    }
    if (final_matches_desired) {
        result.status |= kStatusCasFinalValue;
    }
    const uint64_t expected_add_sum = state->control.thread_count * result.add_initial +
                                      state->control.thread_count * (state->control.thread_count - 1U) / 2U;
    if (result.add_ticket_in_range == state->control.thread_count && result.add_ticket_sum == expected_add_sum) {
        result.status |= kStatusAddTicketPermutation;
    }
    if (result.add_final == result.add_initial + state->control.thread_count) {
        result.status |= kStatusAddFinalValue;
    }
    if ((result.cas_initial >> 32U) != 0U && (result.cas_final >> 32U) != 0U && (result.add_initial >> 32U) != 0U &&
        (result.add_final >> 32U) != 0U) {
        result.status |= kStatusFullWidth64Bit;
    }

    PublishResult(&state->result, result);
}
