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

#ifndef TESTS_ATOMIC_PROBE_CACHE_PRELOAD_SHARED_H_
#define TESTS_ATOMIC_PROBE_CACHE_PRELOAD_SHARED_H_

#include <cstddef>
#include <cstdint>

namespace cache_preload {

#if defined(__ENABLE_ASC_LANG__)
#define CACHE_PRELOAD_SHARED_FN __host__ __aicore__
#elif defined(__CCE_AICORE__)
#define CACHE_PRELOAD_SHARED_FN __aicore__
#else
#define CACHE_PRELOAD_SHARED_FN
#endif

constexpr uint32_t kDataWords = 8192;
constexpr uint32_t kSamples = 9;
constexpr uint32_t kTargetStartWord = 512;
constexpr uint32_t kTargetStrideWords = 64;
constexpr uint32_t kGapRounds = 64;
constexpr uint32_t kICachePreloadUnits = 2;
constexpr uint32_t kICachePollLimit = 4096;
constexpr uint32_t kICacheTargetNops = 1024;
constexpr uint32_t kICacheEvictorNops = 8192;

enum class Mode : uint32_t {
    DCacheBaseline = 0,
    DCachePreload = 1,
    DCacheStoreBaseline = 2,
    DCacheStorePreload = 3,
    DCachePublishBaseline = 4,
    DCachePublishPreload = 5,
    ICacheColdBaseline = 6,
    ICacheCurrentPcAsync = 7,
    ICacheCurrentPcWait = 8,
    Count = 9,
};

CACHE_PRELOAD_SHARED_FN constexpr uint64_t DataValue(uint32_t index) {
    return 0xa500000000000000ULL ^ (static_cast<uint64_t>(index) * 0x9e3779b97f4a7c15ULL);
}

CACHE_PRELOAD_SHARED_FN constexpr uint64_t WriteValue(uint32_t index, uint64_t seed, uint64_t gap_checksum) {
    return 0x5a00000000000000ULL ^ (static_cast<uint64_t>(index) * 0xd6e8feb86659fd93ULL) ^ seed ^ gap_checksum;
}

// This intentionally remains a runtime loop in each device kernel: gap_rounds is
// read from GM and passed into a noinline function. It models useful scalar
// work that can overlap a preload request without sequentially walking the
// upcoming ICache target region.
CACHE_PRELOAD_SHARED_FN inline uint64_t GapOracle(uint64_t seed, uint32_t gap_rounds) {
    uint64_t value = seed ^ 0xd1b54a32d192ed03ULL;
    for (uint32_t round = 0; round < gap_rounds; ++round) {
        value ^= value >> (round % 17U + 1U);
        value *= 0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(round) * 2ULL;
        value += (value << (round % 7U + 1U)) ^ (0x94d049bb133111ebULL + static_cast<uint64_t>(round));
    }
    return value;
}

CACHE_PRELOAD_SHARED_FN constexpr uint64_t ICacheTargetOracle(uint64_t value) {
    value ^= value >> 29U;
    value *= 0x9e3779b185ebca87ULL;
    value += 0xa0761d6478bd642fULL;
    value ^= value >> 31U;
    return value;
}

CACHE_PRELOAD_SHARED_FN constexpr uint64_t ICacheEvictorOracle(uint64_t value) {
    value ^= 0xe7037ed1a0b428dbULL;
    value ^= value >> 23U;
    value *= 0x8ebc6af09c88c6e3ULL;
    value ^= value >> 27U;
    return value;
}

struct alignas(64) ProbeControl {
    uint32_t mode;
    uint32_t target_word;
    uint32_t gap_rounds;
    uint32_t reserved0;
    uint64_t seed;
    uint64_t reserved[5];
};

struct alignas(64) ProbeResult {
    uint64_t value;
    uint64_t preparation_checksum;
    uint64_t gap_checksum;
    uint64_t issue_ticks;
    uint64_t access_or_work_ticks;
    uint64_t store_flush_ticks;
    uint64_t total_ticks;
    uint64_t icache_immediate_status;

    uint64_t icache_final_status;
    uint64_t icache_polls;
    uint64_t mode_echo;
    uint64_t target_word_echo;
    uint64_t gap_rounds_echo;
    uint64_t reserved[3];
};

struct alignas(64) ProbeState {
    ProbeControl control;
    ProbeResult result;
    alignas(64) uint64_t data[kDataWords];
};

static_assert(sizeof(ProbeControl) == 64, "control must occupy one cache line");
static_assert(sizeof(ProbeResult) == 128, "result must occupy two cache lines");
static_assert(offsetof(ProbeState, result) == 64, "result must start on its own cache line");
static_assert(offsetof(ProbeState, data) == 192, "data must start on its own cache line");
static_assert(sizeof(ProbeState) == 192 + kDataWords * sizeof(uint64_t), "probe ABI changed");

#undef CACHE_PRELOAD_SHARED_FN

}  // namespace cache_preload

#endif  // TESTS_ATOMIC_PROBE_CACHE_PRELOAD_SHARED_H_
