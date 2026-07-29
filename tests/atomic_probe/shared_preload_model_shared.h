/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the license.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef TESTS_ATOMIC_PROBE_SHARED_PRELOAD_MODEL_SHARED_H_
#define TESTS_ATOMIC_PROBE_SHARED_PRELOAD_MODEL_SHARED_H_

#include "cache_preload_shared.h"

#include <cstddef>
#include <cstdint>

namespace shared_preload_model {

#if defined(__CCE_AICORE__)
#define SHARED_PRELOAD_MODEL_FN __aicore__
#else
#define SHARED_PRELOAD_MODEL_FN
#endif

constexpr uint32_t kAicWorkers = 32;
constexpr uint32_t kAivWorkers = 64;
constexpr uint32_t kWorkers = kAicWorkers + kAivWorkers;
constexpr uint32_t kCacheLineBytes = 64;
constexpr uint32_t kMaxPayloadBytes = 384;
constexpr uint32_t kWriterHistoryBytes = 40;
constexpr uint32_t kOneDescriptorBytes = 128;
constexpr uint32_t kThreeDescriptorBytes = 384;
constexpr uint32_t kDCacheSamples = 11;
constexpr uint32_t kICacheSamples = 13;
constexpr uint32_t kShortGapRounds = 0;
constexpr uint32_t kOverlapGapRounds = 64;
constexpr uint32_t kICachePreloadUnits = 2;
constexpr uint32_t kControlMagic = 0x53504d44U;  // "SPMD"
constexpr uint32_t kStatusComplete = 0x434f4d50U;  // "COMP"

enum class Experiment : uint32_t {
    Publish = 0,
    Consume = 1,
    ICachePlacement = 2,
};

enum class Mode : uint32_t {
    DCacheBaseline = 0,
    DCachePreload = 1,
    ICacheBaseline = 2,
    ICacheCallerPreload = 3,
    ICacheTargetPreload = 4,
};

struct alignas(64) ProbeControl {
    uint32_t magic;
    uint32_t experiment;
    uint32_t mode;
    uint32_t active_bytes;
    uint32_t gap_rounds;
    uint32_t sample_id;
    uint32_t first_worker;
    uint32_t worker_count;
    uint64_t seed;
    uint64_t reserved[3];
};

struct alignas(64) ProbeResult {
    uint64_t setup_ticks;
    uint64_t issue_ticks;
    uint64_t gap_ticks;
    uint64_t access_ticks;
    uint64_t publish_ticks;
    uint64_t total_ticks;
    uint64_t preparation_checksum;
    uint64_t result_checksum;
    uint64_t icache_immediate_status;
    uint64_t icache_final_status;

    uint32_t worker_id;
    uint32_t experiment;
    uint32_t mode;
    uint32_t active_bytes;
    uint32_t gap_rounds;
    uint32_t sample_id;
    uint32_t status;
    uint32_t reserved0;
    uint64_t reserved[2];
};

struct alignas(64) WorkerData {
    alignas(64) uint8_t source[kMaxPayloadBytes];
    alignas(64) uint8_t destination[kMaxPayloadBytes];
};

SHARED_PRELOAD_MODEL_FN constexpr uint8_t PayloadByte(
    uint32_t worker, uint32_t byte
) {
    return static_cast<uint8_t>(
        0x5aU ^ (worker * 29U) ^ (byte * 17U) ^ (byte >> 3U)
    );
}

SHARED_PRELOAD_MODEL_FN inline uint64_t PayloadChecksum(
    uint32_t worker, uint32_t bytes
) {
    uint64_t value = 0xcbf29ce484222325ULL;
    for (uint32_t byte = 0; byte < bytes; ++byte) {
        value ^= static_cast<uint64_t>(PayloadByte(worker, byte));
        value *= 0x100000001b3ULL;
    }
    return value;
}

SHARED_PRELOAD_MODEL_FN constexpr uint32_t CacheLinesForBytes(
    uint32_t bytes
) {
    return (bytes + kCacheLineBytes - 1U) / kCacheLineBytes;
}

static_assert(sizeof(ProbeControl) == 64, "control must occupy one cache line");
static_assert(sizeof(ProbeResult) == 128, "result must occupy two cache lines");
static_assert(sizeof(WorkerData) == 768, "worker data ABI changed");
static_assert(alignof(WorkerData) == kCacheLineBytes, "worker data must be cacheline aligned");
static_assert(kWriterHistoryBytes == 16U + 3U * 8U, "three-symbol history byte count changed");
static_assert(kOneDescriptorBytes == 2U * kCacheLineBytes, "one TensorDesc must occupy two lines");
static_assert(kThreeDescriptorBytes == 6U * kCacheLineBytes, "three TensorDesc values must occupy six lines");

#undef SHARED_PRELOAD_MODEL_FN

}  // namespace shared_preload_model

#endif  // TESTS_ATOMIC_PROBE_SHARED_PRELOAD_MODEL_SHARED_H_
