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

#include "data_type.h"

constexpr uint32_t kFdwicSwimlaneMagic = 0x4653574Cu;  // FSWL
constexpr uint32_t kFdwicSwimlaneVersion = 4;
constexpr uint32_t kFdwicSwimlaneTraceSchemaVersion = 4;
constexpr uint32_t kFdwicSwimlaneDefaultRecordsPerCore = 1u << 16;
// Eligible wait-region atomic calls are aggregated into exact-count batches at
// level 4. Reuse the existing 64K partition instead of reserving hundreds of
// thousands of rows per worker for individual spin iterations.
constexpr uint32_t kFdwicAtomicSwimlaneRecordsPerCore = kFdwicSwimlaneDefaultRecordsPerCore;
constexpr uint32_t kFdwicAtomicSwimlaneLevel = 4;
constexpr uint32_t kFdwicPerfClockMode = 1;
static_assert(
    kFdwicAtomicSwimlaneRecordsPerCore % 2 == 0, "32B record partitions must keep every worker base on a 64B boundary"
);

enum class FdwicSwimlanePhase : int32_t {
    Kernel = 0,
    Alloc = 1,
    Build = 2,
    DrainWon = 3,
    Replay = 4,
    RingBp = 5,
    EfDrain = 6,
    Commit = 7,
    Submit = 8,
    Materialize = 9,
    PrepareMap = 10,
    Claim = 11,
    Fanin = 12,
    Register = 13,
    Atomic = 14,
    ClockBaseline = 15,
    // Schema-v4 parent intervals and true Submit-tail actions. The legacy
    // Alloc/Build/Replay IDs stay reserved for archived captures but are no
    // longer emitted by the production runtime.
    OrchestrationReplay = 16,
    FinalDrain = 17,
    WinnerBuild = 18,
    AllocComplete = 19,
    // Unlike the single-lane standalone probe, a production kernel loser
    // really calls drain_block_won(); keep that work as an exclusive child.
    LoserReplay = 20,
    Count = 21,
};

// Atomic/ClockBaseline extend the existing ten-column FDWIC raw record ABI.
// The first fifteen sites intentionally keep the standalone PA probe's stable
// numbering; production-only BlockWon sites are appended and must not reorder
// those existing values.
enum class FdwicAtomicSite : uint32_t {
    StartupIncrement = 0,
    StartupPoll = 1,
    FatalPoll = 2,
    FatalSet = 3,
    ClaimMax = 4,
    FaninFlagLoad = 5,
    CompletionVendExchange = 6,
    CompletionFlagExchange = 7,
    FrontierInitialLoad = 8,
    FrontierFlagLoad = 9,
    FrontierMax = 10,
    HeapFrontierLoad = 11,
    HeapVendLoad = 12,
    ReplayDoneIncrement = 13,
    ReplayDonePoll = 14,
    WonSlotClaimMax = 15,
    WonRemainingExchange = 16,
    WonLaneResetExchange = 17,
    WonLaneDepositExchange = 18,
    WonStatePublishExchange = 19,
    WonAnyPublishExchange = 20,
    WonAnyLoad = 21,
    WonStateLoad = 22,
    WonLaneClaimExchange = 23,
    WonLaneReleaseExchange = 24,
    WonRemainingFetchSub = 25,
    WonStateClearExchange = 26,
    WonDrainedLoad = 27,
    Count = 28,
};

enum class FdwicAtomicOp : uint32_t {
    Load = 0,
    Exchange = 1,
    FetchAdd = 2,
    FetchMax = 3,
    FetchSub = 4,
};

static_assert(
    static_cast<uint32_t>(FdwicAtomicSite::Count) <= (1U << 16), "atomic site id must fit the compact record"
);
static_assert(
    static_cast<uint32_t>(FdwicAtomicSite::Count) <= 32, "atomic site id must fit the 32-bit poll-region site mask"
);

constexpr uint32_t kFdwicAtomicOpMask = 0x0fU;
constexpr uint32_t kFdwicAtomicResultUsed = 1U << 4;
constexpr uint32_t kFdwicAtomicValueZero = 1U << 5;
constexpr uint32_t kFdwicAtomicReturnReady = 1U << 6;
constexpr uint32_t kFdwicAtomicPollBatch = 1U << 7;
constexpr uint32_t kFdwicAtomicRetriesShift = 8;
constexpr uint32_t kFdwicAtomicPollCountShift = 8;
constexpr uint32_t kFdwicAtomicPollCountMax = (1U << (32 - kFdwicAtomicPollCountShift)) - 1;

constexpr uint32_t kFdwicClockAtomicDependency = 1U << 0;
constexpr uint32_t kFdwicClockAtomicDependencyApplied = 1U << 1;

constexpr uint32_t kFdwicClaimWon = 1U << 0;
constexpr uint32_t kFdwicClaimAttempted = 1U << 1;

PTO_DEVICE_FUNC constexpr FdwicAtomicOp fdwic_atomic_site_op(FdwicAtomicSite site) {
    switch (site) {
    case FdwicAtomicSite::StartupIncrement:
    case FdwicAtomicSite::ReplayDoneIncrement:
        return FdwicAtomicOp::FetchAdd;
    case FdwicAtomicSite::FatalSet:
    case FdwicAtomicSite::CompletionVendExchange:
    case FdwicAtomicSite::CompletionFlagExchange:
    case FdwicAtomicSite::WonRemainingExchange:
    case FdwicAtomicSite::WonLaneResetExchange:
    case FdwicAtomicSite::WonLaneDepositExchange:
    case FdwicAtomicSite::WonStatePublishExchange:
    case FdwicAtomicSite::WonAnyPublishExchange:
    case FdwicAtomicSite::WonLaneClaimExchange:
    case FdwicAtomicSite::WonLaneReleaseExchange:
    case FdwicAtomicSite::WonStateClearExchange:
        return FdwicAtomicOp::Exchange;
    case FdwicAtomicSite::ClaimMax:
    case FdwicAtomicSite::FrontierMax:
    case FdwicAtomicSite::WonSlotClaimMax:
        return FdwicAtomicOp::FetchMax;
    case FdwicAtomicSite::WonRemainingFetchSub:
        return FdwicAtomicOp::FetchSub;
    default:
        return FdwicAtomicOp::Load;
    }
}

PTO_DEVICE_FUNC constexpr bool fdwic_atomic_site_result_used(FdwicAtomicSite site) {
    switch (site) {
    case FdwicAtomicSite::StartupIncrement:
    case FdwicAtomicSite::FatalSet:
    case FdwicAtomicSite::CompletionVendExchange:
    case FdwicAtomicSite::CompletionFlagExchange:
    case FdwicAtomicSite::ReplayDoneIncrement:
    case FdwicAtomicSite::WonRemainingExchange:
    case FdwicAtomicSite::WonLaneResetExchange:
    case FdwicAtomicSite::WonLaneDepositExchange:
    case FdwicAtomicSite::WonStatePublishExchange:
    case FdwicAtomicSite::WonAnyPublishExchange:
    case FdwicAtomicSite::WonLaneReleaseExchange:
    case FdwicAtomicSite::WonStateClearExchange:
        return false;
    default:
        return true;
    }
}

// Observation loads used by explicit scheduler wait regions are batchable.
// WonLaneClaimExchange is the sole RMW exception: only its idempotent failed
// retries (old value already kDrainedClaimed) are batched, while the successful
// state transition remains a one-call record.
PTO_DEVICE_FUNC constexpr bool fdwic_atomic_site_is_poll_batchable(FdwicAtomicSite site) {
    switch (site) {
    case FdwicAtomicSite::StartupPoll:
    case FdwicAtomicSite::FatalPoll:
    case FdwicAtomicSite::FaninFlagLoad:
    case FdwicAtomicSite::HeapFrontierLoad:
    case FdwicAtomicSite::HeapVendLoad:
    case FdwicAtomicSite::ReplayDonePoll:
    case FdwicAtomicSite::WonAnyLoad:
    case FdwicAtomicSite::WonStateLoad:
    case FdwicAtomicSite::WonLaneClaimExchange:
    case FdwicAtomicSite::WonDrainedLoad:
        return true;
    default:
        return false;
    }
}

constexpr uint32_t kFdwicAtomicPollBatchSiteCount = 10;
static_assert(kFdwicAtomicPollBatchSiteCount <= 32, "poll-batch sites must fit the 32-bit active mask");

PTO_DEVICE_FUNC constexpr int32_t fdwic_atomic_poll_batch_index(FdwicAtomicSite site) {
    switch (site) {
    case FdwicAtomicSite::StartupPoll:
        return 0;
    case FdwicAtomicSite::FatalPoll:
        return 1;
    case FdwicAtomicSite::FaninFlagLoad:
        return 2;
    case FdwicAtomicSite::HeapFrontierLoad:
        return 3;
    case FdwicAtomicSite::HeapVendLoad:
        return 4;
    case FdwicAtomicSite::ReplayDonePoll:
        return 5;
    case FdwicAtomicSite::WonAnyLoad:
        return 6;
    case FdwicAtomicSite::WonStateLoad:
        return 7;
    case FdwicAtomicSite::WonDrainedLoad:
        return 8;
    case FdwicAtomicSite::WonLaneClaimExchange:
        return 9;
    default:
        return -1;
    }
}

PTO_DEVICE_FUNC constexpr FdwicAtomicSite fdwic_atomic_poll_batch_site(uint32_t index) {
    switch (index) {
    case 0:
        return FdwicAtomicSite::StartupPoll;
    case 1:
        return FdwicAtomicSite::FatalPoll;
    case 2:
        return FdwicAtomicSite::FaninFlagLoad;
    case 3:
        return FdwicAtomicSite::HeapFrontierLoad;
    case 4:
        return FdwicAtomicSite::HeapVendLoad;
    case 5:
        return FdwicAtomicSite::ReplayDonePoll;
    case 6:
        return FdwicAtomicSite::WonAnyLoad;
    case 7:
        return FdwicAtomicSite::WonStateLoad;
    case 8:
        return FdwicAtomicSite::WonDrainedLoad;
    case 9:
        return FdwicAtomicSite::WonLaneClaimExchange;
    default:
        return FdwicAtomicSite::Count;
    }
}

struct FdwicAtomicPollBurst {
    uint64_t start_cycle[kFdwicAtomicPollBatchSiteCount];
    uint32_t call_count[kFdwicAtomicPollBatchSiteCount];
    uint32_t active_mask;
    uint32_t enabled_mask;
};

// perf-clock 只复用每核固定 64B 状态中的既有 32B pad，不分配逐事件
// record。expected_submit_count 由 PA orchestration 明确声明；设备与 host
// 都用它校验最后一个 Submit 的边界，而不是把任意一次 Submit 当作末次。
struct FdwicPerfClockCoreData {
    uint64_t first_submit_start;
    uint64_t last_submit_end;
    uint32_t submit_count;
    uint32_t expected_submit_count;
    uint32_t mode;
    uint32_t final_seen;
};

static_assert(sizeof(FdwicPerfClockCoreData) == 32, "perf-clock data must fit the existing core-state pad");
static_assert(offsetof(FdwicPerfClockCoreData, first_submit_start) == 0, "perf-clock start offset changed");
static_assert(offsetof(FdwicPerfClockCoreData, last_submit_end) == 8, "perf-clock end offset changed");
static_assert(offsetof(FdwicPerfClockCoreData, submit_count) == 16, "perf-clock count offset changed");
static_assert(offsetof(FdwicPerfClockCoreData, expected_submit_count) == 20, "perf-clock expected offset changed");

struct FdwicSwimlaneCoreState {
    volatile uint32_t count;
    volatile uint32_t dropped;
    // Exact number of source-level atomic wrapper calls made by this worker
    // while level-4 tracing was active. Poll batches contribute their encoded
    // call_count rather than one call per Atomic record.
    volatile uint32_t atomic_calls;
    // Calls represented by PollBatch rows and the physical number of those
    // rows. The host derives Atomic rows as
    // atomic_calls - poll_calls + poll_batch_records, then verifies the raw.
    volatile uint32_t poll_calls;
    volatile uint32_t poll_batch_records;
    // Topology is invariant within a worker partition. Store it once here
    // instead of repeating the same 12 bytes in every record.
    volatile int32_t core_idx;
    volatile int32_t block_id;
    volatile int32_t lane;
    union {
        uint32_t pad[8];
        FdwicPerfClockCoreData perf_clock;
    };
} __attribute__((aligned(64)));

static_assert(sizeof(FdwicSwimlaneCoreState) == 64, "FdwicSwimlaneCoreState must occupy one cacheline");
static_assert(offsetof(FdwicSwimlaneCoreState, perf_clock) == 32, "perf-clock must reuse the existing 32B tail");

struct FdwicSwimlaneHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_cores;
    uint32_t records_per_core;
    uint64_t freq_hz;
    FdwicSwimlaneCoreState cores[108];
} __attribute__((aligned(64)));

static_assert(sizeof(FdwicSwimlaneHeader) % 64 == 0, "FdwicSwimlaneHeader must be cacheline aligned");
static_assert(offsetof(FdwicSwimlaneHeader, cores) == 64, "per-core state must start at the second cacheline");

struct FdwicSwimlaneRecord {
    uint64_t start_cycle;
    uint64_t end_cycle;
    int32_t task_id;
    int32_t func_id;
    uint32_t flags;
    uint16_t phase;
    uint16_t aux;
} __attribute__((aligned(32)));

static_assert(sizeof(FdwicSwimlaneRecord) == 32, "FdwicSwimlaneRecord must occupy half a cacheline");
static_assert(alignof(FdwicSwimlaneRecord) == 32, "FdwicSwimlaneRecord alignment changed");
