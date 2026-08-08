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

// S0 只验证 AIV0 上的 SIMT 基础能力和同 AIV GM 可见性。跨 AIV/AIC 的
// 结论留给下一阶段 mixed probe，不能由本文件的同核结果外推。

#include <pto/common/kernel_meta.hpp>

#include "cce_aicore_intrinsics.h"
#include "simt_api/asc_simt.h"

#include "../common/s0_probe.h"

PTO_SYNCALL_AIV_KERNEL_META(simt_cross_core_s0_0_mix_aiv);

namespace {

using namespace pa_scheduler::simt_cross_core;
using namespace pa_scheduler::simt_cross_core::s0;

constexpr int kSingleCacheLine = 0;

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtRotateLeft(uint64_t value, uint32_t shift) {
    return (value << shift) | (value >> (64U - shift));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtExpectedPayloadWord(uint64_t nonce, uint32_t word) {
    const uint32_t shift = (word % 23U) + 1U;
    return 0xA500000000000000ULL ^ SimtRotateLeft(nonce ^ (0x0101010101010101ULL * (word + 1U)), shift) ^
           (static_cast<uint64_t>(kTaskId) << 32U) ^ word;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtExpectedThreadWord(uint64_t nonce, uint32_t thread) {
    const uint32_t shift = (thread % 29U) + 1U;
    return 0x5100000000000000ULL ^ SimtRotateLeft(nonce ^ (0x0010001000100010ULL + thread), shift) ^
           static_cast<uint64_t>(thread);
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(kThreadCount) void S0SimtBuild(
    __gm__ uint64_t *state_word, __gm__ uint64_t *payload_words, __gm__ uint64_t *report_words,
    __gm__ uint64_t *thread_words, uint64_t launch_nonce, uint64_t visibility_mode
) {
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    if (thread >= kThreadCount) {
        return;
    }

    thread_words[thread] = SimtExpectedThreadWord(launch_nonce, thread);
    // 每个 thread 都建立自身普通 GM store 的完成顺序；Main Scalar 仍需
    // 等待整个 VF，不能把这个 fence 当作 VF completion event。
    asc_threadfence();

    if (thread != 0U) {
        return;
    }

    const uint64_t reserve_observed = asc_atomic_cas(state_word, kEmptyState, kBuildingState);
    report_words[0] = reserve_observed;
    report_words[2] = static_cast<uint64_t>(blockDim.x);
    report_words[4] = launch_nonce;
    if (reserve_observed != kEmptyState) {
        report_words[1] = reserve_observed;
        report_words[3] = 0U;
        asc_threadfence();
        return;
    }

    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        payload_words[word] = SimtExpectedPayloadWord(launch_nonce, word);
    }
    report_words[3] = kPayloadWords;

    const auto mode = static_cast<VisibilityMode>(visibility_mode);
    if (mode == VisibilityMode::WriterDcci || mode == VisibilityMode::WriterAndReaderDcci) {
        // 只处理 payload 所在的单独 cache line；control 是 atomic-only
        // cacheline，不能被这条 DCCI 顺带覆盖。
        asc_dcci_single(static_cast<__gm__ void *>(payload_words));
    }
    asc_threadfence();

    const uint64_t publish_observed = asc_atomic_cas(state_word, kBuildingState, kBuiltState);
    report_words[1] = publish_observed;
    // report 是探针诊断数据而非协议 payload。完成前再建一次顺序，随后由
    // Main Scalar 用 ld_dev 读取，避免其可见性污染 payload DCCI 对照。
    asc_threadfence();
}

__aicore__ __attribute__((always_inline)) inline uint64_t LoadDev64(__gm__ uint64_t *address) {
    return static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0));
}

__aicore__ __attribute__((always_inline)) inline void StoreDev64(__gm__ uint64_t *address, uint64_t value) {
    __builtin_cce_st_dev(value, address, 0);
}

__aicore__ __attribute__((always_inline)) inline uint64_t
ScalarCas(__gm__ uint64_t *address, uint64_t expected, uint64_t desired) {
    return atomicCAS(address, expected, desired);
}

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarAtomicLoad(__gm__ uint64_t *address) {
    return atomicAdd(address, static_cast<uint64_t>(0U));
}

__aicore__ void PublishResult(__gm__ ProbeResult *result, const ProbeResult &local) {
    __gm__ uint64_t *destination = reinterpret_cast<__gm__ uint64_t *>(result);
    const uint64_t *source = reinterpret_cast<const uint64_t *>(&local);
    for (uint32_t word = 0U; word < sizeof(ProbeResult) / sizeof(uint64_t); ++word) {
        StoreDev64(destination + word, source[word]);
    }
    dsb(DSB_ALL);
}

}  // namespace

extern "C" __global__ __aicore__ void
simt_cross_core_s0_0_mix_aiv(__gm__ pa_scheduler::simt_cross_core::s0::ProbeState *state) {
    using namespace pa_scheduler::simt_cross_core;
    using namespace pa_scheduler::simt_cross_core::s0;

    // host 在每次 launch 前复用同一地址并改写 nonce/mode。先让本核旧 control
    // cacheline 失效，避免把控制输入陈旧误判成 SIMT payload 可见性问题。
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dsb(DSB_ALL);

    ProbeResult result{};
    result.magic = kResultMagic;
    result.physical_core_id = static_cast<uint64_t>(get_coreid()) & 0x0FFFU;
    result.subblock_id = static_cast<uint64_t>(get_subblockid());
    result.launch_nonce = state->control.launch_nonce;
    result.visibility_mode = state->control.visibility_mode;

    const bool config_valid = state->control.magic == kProbeMagic && state->control.version == kProbeVersion &&
                              state->control.thread_count == kThreadCount &&
                              state->control.payload_words == kPayloadWords &&
                              state->control.visibility_mode < kVisibilityModeCount;
    if (!config_valid) {
        PublishResult(&state->result, result);
        return;
    }
    result.status |= kStatusConfigValid;

    cce::async_invoke<S0SimtBuild>(
        cce::dim3{kThreadCount, 1U, 1U}, &state->cell.state, &state->payload.words[0],
        reinterpret_cast<__gm__ uint64_t *>(&state->simt_report), &state->thread_words.words[0],
        state->control.launch_nonce, state->control.visibility_mode
    );

    // PTO A5 的现有 SIMT gather 实现采用同一 V->S event 收口
    // cce::async_invoke。S0 通过真实 A5 动态结果再次验证这个边界。
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    result.reserve_observed = LoadDev64(&state->simt_report.reserve_observed);
    result.publish_observed = LoadDev64(&state->simt_report.publish_observed);
    const uint64_t reported_threads = LoadDev64(&state->simt_report.participating_threads);
    const uint64_t reported_payload_words = LoadDev64(&state->simt_report.payload_words_written);
    const uint64_t reported_nonce = LoadDev64(&state->simt_report.launch_nonce);
    if (result.reserve_observed == kEmptyState) {
        result.status |= kStatusReserveWon;
    }
    if (result.publish_observed == kBuildingState) {
        result.status |= kStatusPublishWon;
    }

    uint64_t expected_checksum = 0x243F6A8885A308D3ULL;
    uint64_t observed_checksum = 0x243F6A8885A308D3ULL;
    uint64_t observed_threads = 0U;
    for (uint32_t thread = 0U; thread < kThreadCount; ++thread) {
        const uint64_t expected = ExpectedThreadWord(state->control.launch_nonce, thread);
        const uint64_t observed = LoadDev64(&state->thread_words.words[thread]);
        expected_checksum = FoldChecksum(expected_checksum, expected);
        observed_checksum = FoldChecksum(observed_checksum, observed);
        observed_threads += observed == expected ? 1U : 0U;
    }
    result.observed_thread_count = observed_threads;
    if (observed_threads == kThreadCount && reported_threads == kThreadCount &&
        reported_nonce == state->control.launch_nonce) {
        result.status |= kStatusAllThreadsObserved;
    }

    result.claim_observed = ScalarCas(&state->cell.state, kBuiltState, kClaimedState);
    if (result.claim_observed == kBuiltState) {
        result.status |= kStatusClaimWon;
        const auto mode = static_cast<VisibilityMode>(state->control.visibility_mode);
        if (ReaderUsesDcci(mode)) {
            dcci(static_cast<__gm__ void *>(&state->payload), kSingleCacheLine);
            dsb(DSB_ALL);
        }

        uint64_t valid_payload_words = 0U;
        for (uint32_t word = 0U; word < kPayloadWords; ++word) {
            const uint64_t expected = ExpectedPayloadWord(state->control.launch_nonce, word);
            // 这里故意走普通 GM load：它是四组 DCCI 对照的被测路径。
            const uint64_t observed = state->payload.words[word];
            expected_checksum = FoldChecksum(expected_checksum, expected);
            observed_checksum = FoldChecksum(observed_checksum, observed);
            valid_payload_words += observed == expected ? 1U : 0U;
        }
        result.observed_payload_words = valid_payload_words;
        if (valid_payload_words == kPayloadWords && reported_payload_words == kPayloadWords) {
            result.status |= kStatusPayloadValid;
        }

        result.done_observed = ScalarCas(&state->cell.state, kClaimedState, kDoneState);
        if (result.done_observed == kClaimedState) {
            result.status |= kStatusDoneWon;
        }
    } else {
        result.done_observed = result.claim_observed;
    }

    result.duplicate_reserve_observed = ScalarCas(&state->cell.state, kEmptyState, kBuildingState);
    if (result.duplicate_reserve_observed != kEmptyState) {
        result.status |= kStatusDuplicateReserveRejected;
    }
    result.final_state = ScalarAtomicLoad(&state->cell.state);
    if (result.final_state == kDoneState) {
        result.status |= kStatusFinalDone;
    }
    result.expected_checksum = expected_checksum;
    result.observed_checksum = observed_checksum;

    PublishResult(&state->result, result);
}
