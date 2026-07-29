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
#pragma once

#include <cstddef>
#include <cstdint>

namespace shared_tensor_map_visibility_probe {

inline constexpr uint32_t kAivBlocks = 2;
inline constexpr uint32_t kCacheLineBytes = 64;
inline constexpr uint32_t kRingCapacity = 128;
inline constexpr uint32_t kReuseTasks = 3 * kRingCapacity + 8;
inline constexpr uint32_t kTotalTasks = 2 + kReuseTasks;
inline constexpr uint32_t kDefaultLaunches = 20;
inline constexpr uint32_t kMaxLaunches = 1000;
inline constexpr uint64_t kWaitTimeoutCycles = 100000000;

inline constexpr uint64_t kBufferA0 = 0x0000000100000000ULL;
inline constexpr uint64_t kBufferA1 = 0x000000010007a000ULL;
inline constexpr uint64_t kBufferB = 0x0000000100001000ULL;
inline constexpr uint32_t kExpectedBucketA = 63;
inline constexpr uint32_t kExpectedBucketB = 123;

inline constexpr int64_t kParticipantMagic = 0x53544d5052544349LL;   // "STMPRTCI"
inline constexpr int64_t kParticipantFinish = 0x53544d50444f4e45LL;  // "STMPDONE"

enum class ErrorCode : int64_t {
    None = 0,
    InvalidTopology = 1,
    HashConfiguration = 2,
    ReadyTimeout = 3,
    ReadyOvershoot = 4,
    PublishFailed = 5,
    CommitTimeout = 6,
    CommitOvershoot = 7,
    CommitMismatch = 8,
    ReclaimMismatch = 9,
    HeadMismatch = 10,
    TailMismatch = 11,
    SequenceMismatch = 12,
    ReadFailed = 13,
    PayloadMismatch = 14,
    LookupProtocol = 15,
    LookupMismatch = 16,
    DoneTimeout = 17,
    DoneOvershoot = 18,
};

enum class Phase : int64_t {
    Setup = 0,
    WaitReady = 1,
    Publish = 2,
    WaitCommit = 3,
    CheckControl = 4,
    CheckSlot = 5,
    CheckLookup = 6,
    WaitDone = 7,
};

struct alignas(kCacheLineBytes) AtomicLine {
    volatile int64_t value;
    uint8_t pad[kCacheLineBytes - sizeof(int64_t)];
};
static_assert(sizeof(AtomicLine) == kCacheLineBytes);

struct alignas(kCacheLineBytes) DiagnosticLine {
    volatile int64_t words[8];
};
static_assert(sizeof(DiagnosticLine) == kCacheLineBytes);

struct alignas(kCacheLineBytes) ParticipantLine {
    volatile int64_t words[8];
};
static_assert(sizeof(ParticipantLine) == kCacheLineBytes);

struct alignas(kCacheLineBytes) ProbeControl {
    AtomicLine reader_ready;
    AtomicLine reader_done;
    AtomicLine abort_code;
    AtomicLine first_error_claim;
    AtomicLine finish_count;
    DiagnosticLine first_error;
    DiagnosticLine first_snapshot;
    ParticipantLine participants[kAivBlocks];
    DiagnosticLine guard;
};
static_assert(offsetof(ProbeControl, reader_ready) == 0);
static_assert(offsetof(ProbeControl, reader_done) == 1 * kCacheLineBytes);
static_assert(offsetof(ProbeControl, abort_code) == 2 * kCacheLineBytes);
static_assert(offsetof(ProbeControl, first_error_claim) == 3 * kCacheLineBytes);
static_assert(offsetof(ProbeControl, finish_count) == 4 * kCacheLineBytes);
static_assert(offsetof(ProbeControl, first_error) == 5 * kCacheLineBytes);
static_assert(offsetof(ProbeControl, first_snapshot) == 6 * kCacheLineBytes);
static_assert(offsetof(ProbeControl, participants) == 7 * kCacheLineBytes);
static_assert(offsetof(ProbeControl, guard) == 9 * kCacheLineBytes);
static_assert(sizeof(ProbeControl) == 10 * kCacheLineBytes);

struct KernelArgs {
    uint64_t map_pointer;
    uint64_t control_pointer;
    uint32_t num_blocks;
    uint32_t launch_id;
};
static_assert(sizeof(KernelArgs) == 24);

}  // namespace shared_tensor_map_visibility_probe
