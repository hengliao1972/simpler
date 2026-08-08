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

// A1 独立探针：用同一份 A/B 代码区分同 warp 分歧路径串行与跨 warp 独立推进。
// 所有轮询同时受 CLOCK64 deadline 和迭代次数约束，失败模式不会把 kernel 卡死。

#include <pto/common/kernel_meta.hpp>

#include "cce_aicore_intrinsics.h"
#include "simt_api/asc_simt.h"

#include "../common/warp_concurrency_probe.h"

PTO_SYNCALL_AIV_KERNEL_META(simt_cross_core_warp_concurrency_0_mix_aiv);

namespace {

using namespace pa_scheduler::simt_cross_core::warp_concurrency;

constexpr int kSingleCacheLine = 0;

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtRotateLeft(uint64_t value, uint32_t shift) {
    return (value << shift) | (value >> (64U - shift));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtExpectedReady(uint64_t nonce, uint32_t participant) {
    return 0xA100000000000000ULL ^ SimtRotateLeft(nonce, participant * 11U + 7U) ^
           (0x0001000100010001ULL * static_cast<uint64_t>(participant + 1U));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtExpectedReportMarker(uint64_t nonce, uint64_t mode, uint32_t participant) {
    return kReportMagic ^ SimtRotateLeft(nonce, participant * 13U + 3U) ^ (mode << 32U) ^
           static_cast<uint64_t>(participant + 1U);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtWorkA(uint64_t nonce, uint32_t iterations) {
    uint64_t value = nonce ^ 0x243F6A8885A308D3ULL;
    for (uint32_t index = 0U; index < iterations; ++index) {
        value =
            SimtRotateLeft(value ^ (0x9E3779B97F4A7C15ULL + static_cast<uint64_t>(index)), 7U) + 0xD1B54A32D192ED03ULL;
    }
    return value;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtWorkB(uint64_t nonce, uint32_t iterations) {
    uint64_t value = nonce ^ 0x13198A2E03707344ULL;
    for (uint32_t index = 0U; index < iterations; ++index) {
        value += 0x94D049BB133111EBULL ^ static_cast<uint64_t>(index);
        value = SimtRotateLeft(value, 19U) ^ (value >> 11U) ^ 0xBF58476D1CE4E5B9ULL;
    }
    return value;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void StoreReport(
    __gm__ ParticipantReport *report, uint64_t nonce, uint64_t mode, uint32_t participant, uint32_t thread,
    HandshakeStatus handshake_status, uint64_t poll_count, uint64_t observed_peer, uint64_t start_clock,
    uint64_t end_clock, uint64_t checksum, uint64_t published_ready
) {
    report->launch_nonce = nonce;
    report->mode = mode;
    report->participant = participant;
    report->thread_id = thread;
    report->warp_id = thread / kWarpSize;
    report->lane_id = thread % kWarpSize;
    report->handshake_status = static_cast<uint64_t>(handshake_status);
    report->poll_count = poll_count;
    report->observed_peer = observed_peer;
    report->start_clock = start_clock;
    report->end_clock = end_clock;
    report->checksum = checksum;
    report->published_ready = published_ready;
    report->reserved[0] = 0U;
    report->reserved[1] = 0U;
    asc_threadfence();
    report->marker = SimtExpectedReportMarker(nonce, mode, participant);
    asc_threadfence();
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void RunA(
    __gm__ uint64_t *own_ready, __gm__ uint64_t *peer_ready, __gm__ ParticipantReport *report, uint64_t nonce,
    uint64_t mode, uint32_t poll_limit, uint64_t poll_clock_budget, uint32_t work_iterations, bool handshake
) {
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    const uint64_t start_clock = clock();
    const uint64_t own_value = SimtExpectedReady(nonce, 0U);
    uint64_t observed_peer = kOutputSentinel;
    uint64_t poll_count = 0U;
    HandshakeStatus status = HandshakeStatus::kNotApplicable;
    if (handshake) {
        const uint64_t publish_old = asc_atomic_cas(own_ready, static_cast<uint64_t>(0U), own_value);
        const uint64_t poll_start = clock();
        for (uint32_t poll = 0U; poll < poll_limit && clock() - poll_start < poll_clock_budget; ++poll) {
            observed_peer = asc_atomic_add(peer_ready, static_cast<uint64_t>(0U));
            poll_count = static_cast<uint64_t>(poll) + 1U;
            if (observed_peer == SimtExpectedReady(nonce, 1U)) {
                status = publish_old == 0U ? HandshakeStatus::kSuccess : HandshakeStatus::kTimeout;
                break;
            }
        }
        if (status != HandshakeStatus::kSuccess) {
            status = HandshakeStatus::kTimeout;
        }
    }
    const uint64_t checksum = SimtWorkA(nonce, work_iterations);
    const uint64_t end_clock = clock();
    StoreReport(
        report, nonce, mode, 0U, thread, status, poll_count, observed_peer, start_clock, end_clock, checksum,
        handshake ? own_value : 0U
    );
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void RunB(
    __gm__ uint64_t *own_ready, __gm__ uint64_t *peer_ready, __gm__ ParticipantReport *report, uint64_t nonce,
    uint64_t mode, uint32_t poll_limit, uint64_t poll_clock_budget, uint32_t work_iterations, bool handshake
) {
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    const uint64_t start_clock = clock();
    const uint64_t own_value = SimtExpectedReady(nonce, 1U);
    uint64_t observed_peer = kOutputSentinel;
    uint64_t poll_count = 0U;
    HandshakeStatus status = HandshakeStatus::kNotApplicable;
    if (handshake) {
        const uint64_t publish_old = asc_atomic_cas(own_ready, static_cast<uint64_t>(0U), own_value);
        const uint64_t poll_start = clock();
        for (uint32_t poll = 0U; poll < poll_limit && clock() - poll_start < poll_clock_budget; ++poll) {
            observed_peer = asc_atomic_add(peer_ready, static_cast<uint64_t>(0U));
            poll_count = static_cast<uint64_t>(poll) + 1U;
            if (observed_peer == SimtExpectedReady(nonce, 0U)) {
                status = publish_old == 0U ? HandshakeStatus::kSuccess : HandshakeStatus::kTimeout;
                break;
            }
        }
        if (status != HandshakeStatus::kSuccess) {
            status = HandshakeStatus::kTimeout;
        }
    }
    const uint64_t checksum = SimtWorkB(nonce, work_iterations);
    const uint64_t end_clock = clock();
    StoreReport(
        report, nonce, mode, 1U, thread, status, poll_count, observed_peer, start_clock, end_clock, checksum,
        handshake ? own_value : 0U
    );
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(kLaunchThreads) void WarpConcurrencyProbe(
    __gm__ uint64_t *ready_a, __gm__ uint64_t *ready_b, __gm__ ParticipantReport *reports, uint64_t nonce,
    uint64_t mode, uint32_t poll_limit, uint64_t poll_clock_budget, uint32_t work_iterations
) {
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    const uint32_t warp = thread / kWarpSize;
    const uint32_t lane = thread % kWarpSize;

    if (mode == static_cast<uint64_t>(ProbeMode::kAOnly)) {
        if (thread == 0U) {
            RunA(ready_a, ready_b, &reports[0], nonce, mode, poll_limit, poll_clock_budget, work_iterations, false);
        }
        return;
    }
    if (mode == static_cast<uint64_t>(ProbeMode::kBOnly)) {
        if (thread == kWarpSize / 2U) {
            RunB(ready_b, ready_a, &reports[1], nonce, mode, poll_limit, poll_clock_budget, work_iterations, false);
        }
        return;
    }
    if (mode == static_cast<uint64_t>(ProbeMode::kSameWarp)) {
        // 同一 warp 的上下两个半 warp 进入互斥代码路径；leader 分别是 lane 0 和 lane 16。
        if (warp == 0U) {
            if (lane < kWarpSize / 2U) {
                if (lane == 0U) {
                    RunA(
                        ready_a, ready_b, &reports[0], nonce, mode, poll_limit, poll_clock_budget, work_iterations, true
                    );
                }
            } else {
                if (lane == kWarpSize / 2U) {
                    RunB(
                        ready_b, ready_a, &reports[1], nonce, mode, poll_limit, poll_clock_budget, work_iterations, true
                    );
                }
            }
        }
        return;
    }
    if (mode == static_cast<uint64_t>(ProbeMode::kCrossWarp)) {
        // 两个 warp 仍执行 A/B 两条不同路径，但 warp0 与 warp1 拥有独立的推进状态。
        if (warp == 0U) {
            if (lane == 0U) {
                RunA(ready_a, ready_b, &reports[0], nonce, mode, poll_limit, poll_clock_budget, work_iterations, true);
            }
        } else if (warp == 1U) {
            if (lane == 0U) {
                RunB(ready_b, ready_a, &reports[1], nonce, mode, poll_limit, poll_clock_budget, work_iterations, true);
            }
        }
    }
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

}  // namespace

extern "C" __global__ __aicore__ void
simt_cross_core_warp_concurrency_0_mix_aiv(__gm__ pa_scheduler::simt_cross_core::warp_concurrency::ProbeState *state) {
    using namespace pa_scheduler::simt_cross_core::warp_concurrency;

    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dsb(DSB_ALL);

    ProbeResult result{};
    result.magic = kResultMagic;
    result.physical_core_id = static_cast<uint64_t>(get_coreid()) & 0x0FFFU;
    result.subblock_id = static_cast<uint64_t>(get_subblockid());
    result.launch_nonce = state->control.launch_nonce;
    result.mode = state->control.mode;

    const bool config_valid = state->control.magic == kProbeMagic && state->control.version == kProbeVersion &&
                              IsValidMode(state->control.mode) && state->control.poll_limit > 0U &&
                              state->control.poll_limit <= kMaximumPollLimit && state->control.poll_clock_budget > 0U &&
                              state->control.poll_clock_budget <= kMaximumPollClockBudget &&
                              state->control.work_iterations > 0U &&
                              state->control.work_iterations <= kMaximumWorkIterations;
    if (!config_valid) {
        PublishResult(&state->result, result);
        return;
    }
    result.status |= kStatusConfigValid;

    cce::async_invoke<WarpConcurrencyProbe>(
        cce::dim3{kLaunchThreads, 1U, 1U}, &state->ready_a.value, &state->ready_b.value, &state->reports[0],
        state->control.launch_nonce, state->control.mode, static_cast<uint32_t>(state->control.poll_limit),
        state->control.poll_clock_budget, static_cast<uint32_t>(state->control.work_iterations)
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    const ProbeMode mode = static_cast<ProbeMode>(state->control.mode);
    uint64_t starts[kParticipantCount] = {0U, 0U};
    uint64_t ends[kParticipantCount] = {0U, 0U};
    bool reports_complete = true;
    for (uint32_t participant = 0U; participant < kParticipantCount; ++participant) {
        __gm__ uint64_t *words = reinterpret_cast<__gm__ uint64_t *>(&state->reports[participant]);
        const bool active = ParticipantIsActive(mode, participant);
        const uint64_t marker = LoadDev64(words + offsetof(ParticipantReport, marker) / sizeof(uint64_t));
        if (!active) {
            reports_complete = reports_complete && marker == kOutputSentinel;
            continue;
        }
        const uint64_t expected_thread = ExpectedThread(mode, participant);
        const uint64_t handshake = LoadDev64(words + offsetof(ParticipantReport, handshake_status) / sizeof(uint64_t));
        starts[participant] = LoadDev64(words + offsetof(ParticipantReport, start_clock) / sizeof(uint64_t));
        ends[participant] = LoadDev64(words + offsetof(ParticipantReport, end_clock) / sizeof(uint64_t));
        reports_complete =
            reports_complete && marker == ExpectedReportMarker(state->control.launch_nonce, mode, participant) &&
            LoadDev64(words + offsetof(ParticipantReport, launch_nonce) / sizeof(uint64_t)) ==
                state->control.launch_nonce &&
            LoadDev64(words + offsetof(ParticipantReport, mode) / sizeof(uint64_t)) == state->control.mode &&
            LoadDev64(words + offsetof(ParticipantReport, participant) / sizeof(uint64_t)) == participant &&
            LoadDev64(words + offsetof(ParticipantReport, thread_id) / sizeof(uint64_t)) == expected_thread &&
            LoadDev64(words + offsetof(ParticipantReport, warp_id) / sizeof(uint64_t)) == expected_thread / kWarpSize &&
            LoadDev64(words + offsetof(ParticipantReport, lane_id) / sizeof(uint64_t)) == expected_thread % kWarpSize &&
            ends[participant] > starts[participant];
        result.active_reports += 1U;
        result.handshake_successes += handshake == static_cast<uint64_t>(HandshakeStatus::kSuccess) ? 1U : 0U;
        result.handshake_timeouts += handshake == static_cast<uint64_t>(HandshakeStatus::kTimeout) ? 1U : 0U;
        result.handshake_not_applicable +=
            handshake == static_cast<uint64_t>(HandshakeStatus::kNotApplicable) ? 1U : 0U;
    }
    if (reports_complete && result.active_reports == ExpectedActiveParticipants(mode)) {
        result.status |= kStatusReportsComplete;
    }

    result.ready_a = LoadDev64(&state->ready_a.value);
    result.ready_b = LoadDev64(&state->ready_b.value);
    const bool handshake_mode = mode == ProbeMode::kSameWarp || mode == ProbeMode::kCrossWarp;
    const bool ready_values = handshake_mode ? result.ready_a == ExpectedReady(state->control.launch_nonce, 0U) &&
                                                   result.ready_b == ExpectedReady(state->control.launch_nonce, 1U) :
                                               result.ready_a == 0U && result.ready_b == 0U;
    if (ready_values) {
        result.status |= kStatusReadyValues;
    }

    const bool handshake_matches =
        (mode == ProbeMode::kSameWarp && result.handshake_successes == 1U && result.handshake_timeouts == 1U) ||
        (mode == ProbeMode::kCrossWarp && result.handshake_successes == 2U && result.handshake_timeouts == 0U) ||
        ((mode == ProbeMode::kAOnly || mode == ProbeMode::kBOnly) && result.handshake_not_applicable == 1U);
    if (handshake_matches) {
        result.status |= kStatusHandshakeMatchesMode;
    }

    bool interval_matches = false;
    if (mode == ProbeMode::kAOnly || mode == ProbeMode::kBOnly) {
        const uint32_t participant = mode == ProbeMode::kAOnly ? 0U : 1U;
        result.interval_relation = static_cast<uint64_t>(IntervalRelation::kSingle);
        result.first_start = starts[participant];
        result.last_end = ends[participant];
        interval_matches = ends[participant] > starts[participant];
    } else {
        const bool overlap = starts[0] < ends[1] && starts[1] < ends[0];
        result.interval_relation =
            static_cast<uint64_t>(overlap ? IntervalRelation::kOverlap : IntervalRelation::kDisjoint);
        result.first_start = starts[0] < starts[1] ? starts[0] : starts[1];
        result.last_end = ends[0] > ends[1] ? ends[0] : ends[1];
        interval_matches = mode == ProbeMode::kSameWarp ? !overlap : overlap;
    }
    if (interval_matches) {
        result.status |= kStatusIntervalMatchesMode;
    }

    PublishResult(&state->result, result);
}
