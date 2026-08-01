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

#ifndef TESTS_ATOMIC_PROBE_CCEC_ATOMIC_POLL_EXCHANGE_CONTENTION_SHARED_H_
#define TESTS_ATOMIC_PROBE_CCEC_ATOMIC_POLL_EXCHANGE_CONTENTION_SHARED_H_

#include <cstddef>
#include <cstdint>

namespace atomic_poll_exchange_contention {

constexpr uint32_t kRounds = 256;
constexpr uint32_t kPrePolls = 31;
constexpr uint32_t kMinAivs = 2;
constexpr uint32_t kDefaultAivs = 2;
constexpr uint32_t kMaxAivs = 24;
constexpr uint64_t kWaitTimeoutTicks = 20000000ULL;
constexpr uint64_t kArmTicks = 2000ULL;

constexpr uint32_t kHeaderMagic = 0x41505743U;  // "APWC"
constexpr uint32_t kReaderMagic = 0x41505244U;  // "APRD"
constexpr uint32_t kFinishMagic = 0x46494E49U;  // "FINI"
constexpr uint32_t kTimingMagic = 0xA5C00000U;

constexpr uint32_t kFlagReadyTimeout = 1U << 0;
constexpr uint32_t kFlagAckTimeout = 1U << 1;
constexpr uint32_t kFlagTargetOldMismatch = 1U << 2;
constexpr uint32_t kFlagNotifyOldMismatch = 1U << 3;

enum class ProbeMode : uint32_t {
    AddDifferentLine = 0,
    AddSameLineDifferentWord = 1,
    AddTargetThenAway = 2,
    AddSameAddress = 3,
    MaxDifferentLine = 4,
    MaxSameLineDifferentWord = 5,
    MaxTargetThenAway = 6,
    MaxSameAddress = 7,
    Count = 8,
};

struct alignas(64) AtomicLine {
    volatile int64_t value;
    volatile int64_t neighbor;
    int64_t reserved[6];
};

struct alignas(64) ProbeHeader {
    uint32_t magic;
    uint32_t mode;
    uint32_t blocks;
    uint32_t readers;
    uint32_t rounds;
    uint32_t pre_polls;
    uint32_t ready_timeouts;
    uint32_t ack_timeouts;
    uint32_t old_value_mismatches;
    uint32_t final_target_low;
    uint32_t final_signal_low;
    uint32_t writer_core;
    uint32_t writer_subblock;
    uint32_t timeout_ticks;
    uint32_t arm_ticks;
    uint32_t finish;
};

struct alignas(64) TimingRecord {
    uint64_t writer_begin;
    uint64_t writer_end;
    uint64_t notify_begin;
    uint64_t notify_end;
    uint64_t all_ack;
    int64_t old_value;
    uint32_t ready_actual;
    uint32_t ack_actual;
    uint32_t flags;
    uint32_t magic_round;
};

struct alignas(64) ReaderResult {
    uint64_t total_pre_polls;
    uint64_t total_followup_polls;
    uint64_t max_observe_ticks;
    int64_t last_observed;
    uint32_t magic;
    uint32_t block;
    uint32_t core;
    uint32_t subblock;
    uint32_t seen_rounds;
    uint32_t pre_poll_mismatches;
    uint32_t epoch_timeouts;
    uint32_t observe_timeouts;
};

struct alignas(64) ProbeStorage {
    AtomicLine target;
    AtomicLine signal;
    AtomicLine epoch;
    AtomicLine ready;
    AtomicLine ack;
    ProbeHeader header;
    TimingRecord timing[kRounds];
    ReaderResult readers[kMaxAivs];
    AtomicLine guard;
};

constexpr uint32_t ModeIndex(ProbeMode mode) { return static_cast<uint32_t>(mode); }

constexpr bool UsesAtomicMax(ProbeMode mode) { return ModeIndex(mode) >= ModeIndex(ProbeMode::MaxDifferentLine); }

constexpr uint32_t AddressPattern(ProbeMode mode) { return ModeIndex(mode) % 4U; }

constexpr bool PollsTargetUntilPublish(ProbeMode mode) { return AddressPattern(mode) == 3U; }

constexpr bool PollsTargetNeighbor(ProbeMode mode) { return AddressPattern(mode) == 1U; }

constexpr int64_t Token(ProbeMode mode, uint32_t round) {
    return static_cast<int64_t>(
        0x5A00000000000000ULL | (static_cast<uint64_t>(ModeIndex(mode)) << 48U) | static_cast<uint64_t>(round + 1U)
    );
}

static_assert(sizeof(AtomicLine) == 64, "atomic line must occupy one cache line");
static_assert(offsetof(AtomicLine, neighbor) == sizeof(int64_t), "neighbor must share target's cache line");
static_assert(sizeof(ProbeHeader) == 64, "probe header must occupy one cache line");
static_assert(sizeof(TimingRecord) == 64, "timing record must occupy one cache line");
static_assert(sizeof(ReaderResult) == 64, "reader result must occupy one cache line");
static_assert(offsetof(ProbeStorage, target) == 0, "target offset changed");
static_assert(offsetof(ProbeStorage, signal) == 64, "signal must use a distinct cache line");
static_assert(offsetof(ProbeStorage, epoch) == 128, "epoch must use a distinct cache line");
static_assert(offsetof(ProbeStorage, ready) == 192, "ready must use a distinct cache line");
static_assert(offsetof(ProbeStorage, ack) == 256, "ack must use a distinct cache line");
static_assert(offsetof(ProbeStorage, timing) % 64 == 0, "timing records must be cache-line aligned");
static_assert(offsetof(ProbeStorage, readers) % 64 == 0, "reader results must be cache-line aligned");
static_assert(offsetof(ProbeStorage, guard) % 64 == 0, "guard must be cache-line aligned");

}  // namespace atomic_poll_exchange_contention

#endif  // TESTS_ATOMIC_PROBE_CCEC_ATOMIC_POLL_EXCHANGE_CONTENTION_SHARED_H_
