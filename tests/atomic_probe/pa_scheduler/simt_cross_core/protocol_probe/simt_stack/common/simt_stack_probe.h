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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_SIMT_STACK_PROBE_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_SIMT_STACK_PROBE_H

#include <cstddef>
#include <cstdint>

namespace pa_scheduler::simt_cross_core::simt_stack {

constexpr uint64_t kProbeMagic = UINT64_C(0x53494D5453544B35);
constexpr uint64_t kProbeVersion = 1U;
constexpr uint64_t kReportMagic = UINT64_C(0x535441434B525054);
constexpr uint64_t kResultMagic = UINT64_C(0x535441434B52534C);
constexpr uint64_t kPoison = UINT64_C(0xC7C7C7C7C7C7C7C7);
constexpr uint32_t kWarpSize = 32U;
constexpr uint32_t kWarpCount = 64U;
constexpr uint32_t kLaunchThreads = kWarpSize * kWarpCount;
constexpr uint32_t kFrameWords = 16U;
constexpr uint32_t kFrameDepth = 2U;
constexpr uint32_t kFrameBytes = kFrameWords * sizeof(uint64_t);
constexpr uint32_t kDivergenceDepth = kWarpSize - 1U;

struct alignas(64) ProbeControl {
    uint64_t magic;
    uint64_t version;
    uint64_t launch_nonce;
    uint32_t launch_threads;
    uint32_t warp_count;
    uint32_t frame_words;
    uint32_t frame_depth;
    uint64_t reserved[3];
};

struct alignas(64) WarpReport {
    uint64_t marker;
    uint64_t launch_nonce;
    uint32_t thread_id;
    uint32_t warp_id;
    uint32_t lane_id;
    uint32_t frame_words;
    uint64_t checksum;
    uint64_t begin_clock;
    uint64_t end_clock;
    uint64_t reserved;
};

struct alignas(64) ProbeResult {
    uint64_t magic;
    uint64_t launch_nonce;
    uint32_t completed_warps;
    uint32_t expected_warps;
    uint64_t checksum_xor;
    uint64_t status;
    uint64_t reserved[3];
};

struct alignas(64) ProbeState {
    ProbeControl control;
    uint64_t thread_checksums[kLaunchThreads];
    WarpReport reports[kWarpCount];
    ProbeResult result;
};

constexpr uint64_t RotateLeft(uint64_t value, uint32_t shift) {
    const uint32_t normalized = shift & 63U;
    return normalized == 0U ? value : (value << normalized) | (value >> (64U - normalized));
}

constexpr uint64_t FrameStep(uint64_t value, uint32_t frame, uint32_t word) {
    value ^= UINT64_C(0x9E3779B97F4A7C15) + (static_cast<uint64_t>(frame) << 32U) + word;
    return RotateLeft(value, (frame * 11U + word * 7U) % 63U + 1U) + UINT64_C(0xD1B54A32D192ED03);
}

inline uint64_t ExpectedThreadChecksum(uint64_t nonce, uint32_t thread) {
    uint64_t value = nonce ^ (static_cast<uint64_t>(thread) << 32U) ^ UINT64_C(0x243F6A8885A308D3);
    const uint32_t lane = thread % kWarpSize;
    uint32_t deepest_level = kDivergenceDepth - 1U;
    bool took_else = false;
    for (uint32_t level = 0U; level < kDivergenceDepth; ++level) {
        const uint32_t threshold = kWarpSize - 1U - level;
        if (lane == threshold) {
            value = FrameStep(value, 96U + level, threshold);
            deepest_level = level;
            took_else = true;
            break;
        }
        value = FrameStep(value, 32U + level, threshold);
    }
    if (!took_else) {
        value = FrameStep(value, 63U, 0U);
    }
    for (uint32_t level_plus_one = deepest_level + 1U; level_plus_one > 0U; --level_plus_one) {
        const uint32_t level = level_plus_one - 1U;
        value = FrameStep(value, 64U + level, kWarpSize - 1U - level);
    }
    const uint32_t frame_count = lane < kWarpSize / 2U ? kFrameDepth : kFrameDepth - 1U;
    for (uint32_t frame = frame_count; frame > 0U; --frame) {
        const uint32_t frame_id = frame - 1U + (kFrameDepth - frame_count);
        for (uint32_t word = 0U; word < kFrameWords; ++word) {
            value = FrameStep(value, frame_id, word);
        }
    }
    if (lane >= kWarpSize / 2U) {
        value = FrameStep(value, 127U, lane);
    }
    return value;
}

inline uint64_t ExpectedChecksum(uint64_t nonce, uint32_t warp) {
    return ExpectedThreadChecksum(nonce, warp * kWarpSize);
}

inline uint64_t ExpectedChecksumXor(uint64_t nonce) {
    uint64_t checksum = 0U;
    for (uint32_t warp = 0U; warp < kWarpCount; ++warp) {
        checksum ^= ExpectedChecksum(nonce, warp);
    }
    return checksum;
}

static_assert(kLaunchThreads == 2048U && kWarpCount == 64U, "SIMT stack topology changed");
static_assert(
    kFrameBytes == 128U && kFrameDepth == 2U && kDivergenceDepth == 31U, "SIMT stack pressure changed"
);
static_assert(sizeof(ProbeControl) == 64U, "SIMT stack control ABI changed");
static_assert(sizeof(WarpReport) == 64U, "SIMT stack report ABI changed");
static_assert(sizeof(ProbeResult) == 64U, "SIMT stack result ABI changed");
static_assert(offsetof(ProbeState, thread_checksums) == 64U, "SIMT stack checksum alignment changed");
static_assert(offsetof(ProbeState, reports) == 64U + kLaunchThreads * sizeof(uint64_t), "SIMT stack report offset changed");

}  // namespace pa_scheduler::simt_cross_core::simt_stack

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_SIMT_STACK_PROBE_H
