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
#ifndef TESTS_ATOMIC_PROBE_CCEC_ST_DEV_LD_DEV_SYNC_SHARED_H
#define TESTS_ATOMIC_PROBE_CCEC_ST_DEV_LD_DEV_SYNC_SHARED_H

#include <cstddef>
#include <cstdint>

namespace st_dev_ld_dev_sync {

constexpr uint32_t kMinAivWorkers = 2;
constexpr uint32_t kDefaultAivWorkers = 20;
constexpr uint32_t kMaxAivWorkers = 20;
constexpr uint32_t kExpectedValue = 0x53594e43U;  // "SYNC"
constexpr uint32_t kResultMagic = 0x534c4453U;    // "SLDS"
constexpr uint64_t kWaitTimeoutTicks = 20000000ULL;  // 20 ms with the A5 1 GHz SYS_CNT.

constexpr uint32_t kFlagInvalidBlockCount = 1U << 0;
constexpr uint32_t kFlagReadTimeout = 1U << 1;

// Protocol requirement: the word polled by ld_dev must own an entire cache
// line. Do not reuse the padding for flags, counters, results, or any other
// live object; unrelated accesses on this line would change the fanout model.
struct alignas(64) SignalLine {
    uint32_t value;
    uint8_t padding[64 - sizeof(uint32_t)];
};

struct alignas(64) WorkerResult {
    uint32_t magic;
    uint32_t block_id;
    uint32_t core_id;
    uint32_t subblock_id;
    uint32_t block_count;
    uint32_t observed_value;
    uint32_t poll_count;
    uint32_t flags;
    uint64_t begin_tick;
    uint64_t observe_tick;
    uint64_t final_barrier_arrive_tick;
    uint64_t end_tick;
};

struct alignas(64) ProbeStorage {
    SignalLine signal;
    WorkerResult workers[kMaxAivWorkers];
    SignalLine guard;
};

static_assert(sizeof(SignalLine) == 64, "signal line must occupy one cache line");
static_assert(alignof(SignalLine) == 64, "signal line must start on a cache-line boundary");
static_assert(sizeof(WorkerResult) == 64, "worker result must occupy one cache line");
static_assert(offsetof(ProbeStorage, signal) % 64 == 0, "signal must be cache-line aligned");
static_assert(offsetof(ProbeStorage, workers) % 64 == 0, "worker results must be cache-line aligned");
static_assert(offsetof(ProbeStorage, workers) == sizeof(SignalLine),
              "no object may share the signal cache line");
static_assert(offsetof(ProbeStorage, guard) % 64 == 0, "guard must be cache-line aligned");

}  // namespace st_dev_ld_dev_sync

#endif  // TESTS_ATOMIC_PROBE_CCEC_ST_DEV_LD_DEV_SYNC_SHARED_H
