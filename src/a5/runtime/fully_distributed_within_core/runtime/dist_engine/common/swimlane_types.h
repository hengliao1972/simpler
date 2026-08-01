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
#include "fdwic_build_identity.h"

constexpr uint32_t kFdwicSwimlaneMagic = 0x4653574Cu;  // FSWL
// Production PA worker and per-task tournament topology.  Keep these layout
// constants visible in every translation unit: state.h is shared by the
// generic AArch64 orchestration build, which can include this header before
// the selected TensorMap mode is known.  The shared-only state and hot path
// remain guarded by PTO_FDWIC_SHARED_MAP below.
constexpr uint32_t kFdwicSharedAicWorkers = 32;
constexpr uint32_t kFdwicSharedAivWorkers = 64;
constexpr uint32_t kFdwicSharedWorkers = kFdwicSharedAicWorkers + kFdwicSharedAivWorkers;
constexpr uint32_t kFdwicSharedAllocClaimTournamentGroups = 8;
constexpr uint32_t kFdwicSharedAicClaimTournamentGroups = 6;
constexpr uint32_t kFdwicSharedAivClaimTournamentGroups = 8;
constexpr uint32_t kFdwicSharedClaimTournamentMaxGroups = kFdwicSharedAivClaimTournamentGroups;
constexpr uint32_t kFdwicSharedClaimTournamentNodeStride = 512;
#if PTO_FDWIC_SHARED_MAP
constexpr uint32_t kFdwicSwimlaneVersion = 5;
constexpr uint32_t kFdwicSwimlaneTraceSchemaVersion = 5;
// Schema-v5 stores the two events that every worker produces for every task
// in a fixed Submit/Claim endpoint area. Keep the standalone physical
// capacity so the raw ABI can later admit PA-G4/B256 without being changed.
constexpr uint32_t kFdwicSharedTraceTaskCapacity = 4352;
constexpr uint32_t kFdwicSharedTracePhase1TaskCount = 1280;
constexpr uint32_t kFdwicSharedSubmitClaimRecordSizeBytes = 32;
constexpr uint32_t kFdwicSwimlaneDefaultRecordsPerCore = 28416;
constexpr uint32_t kFdwicSwimlaneRecordSizeBytes = 16;
#else
constexpr uint32_t kFdwicSwimlaneVersion = 4;
constexpr uint32_t kFdwicSwimlaneTraceSchemaVersion = 4;
constexpr uint32_t kFdwicSwimlaneDefaultRecordsPerCore = 1u << 16;
constexpr uint32_t kFdwicSwimlaneRecordSizeBytes = 32;
#endif
// Eligible wait-region atomic calls are aggregated into exact-count batches at
// level 4. Reuse the existing 64K partition instead of reserving hundreds of
// thousands of rows per worker for individual spin iterations.
constexpr uint32_t kFdwicAtomicSwimlaneRecordsPerCore = kFdwicSwimlaneDefaultRecordsPerCore;
constexpr uint32_t kFdwicAtomicSwimlaneLevel = 4;
constexpr uint32_t kFdwicPerfClockMode = 1;
constexpr uint32_t kFdwicPerfClockKernelMode = 2;
static_assert(
    (static_cast<uint64_t>(kFdwicAtomicSwimlaneRecordsPerCore) * kFdwicSwimlaneRecordSizeBytes) % 64 == 0,
    "generic record partitions must keep every worker base on a 64B boundary"
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
#if PTO_FDWIC_SHARED_MAP
    SharedRegisterPublishMetadata = 20,
    SharedMaterializePublishTaskOutputs = 21,
    SharedMaterializePublishTaskOutputsCopy = 22,
    SharedMaterializePublishTaskOutputsFlush = 23,
    Dcci = 24,
    // Non-atomic ld_dev wait that observes the predecessor's monotonic
    // completion word before this winner enters ordered metadata publication.
    SharedRegisterWaitInsertTurnBypassLoad = 25,
    Count = 26,
    // Compile-only poison value for stale schema-v4 shared call sites. It is
    // deliberately outside Count and cannot be encoded by compact-v5.
    LoserReplay = 0x7fff,
#else
    // Unlike the single-lane standalone probe, a production kernel loser
    // really calls drain_block_won(); keep that work as an exclusive child.
    LoserReplay = 20,
    Count = 21,
#endif
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
#if PTO_FDWIC_SHARED_MAP
    SharedHeapVendLoad = 15,
    SharedHeapCursorLoad = 16,
    SharedHeapCursorReserve = 17,
    SharedHeapVendAdvance = 18,
    SharedInsertTurnPoll = 19,
    SharedInsertTurnHandoff = 20,
    SharedWinnerFatalGuardLoad = 21,
    SharedMetadataFatalGuardLoad = 22,
    SharedFaninOutputPublishedLoad = 23,
    SharedMetadataOutputPublishedLoad = 24,
    SharedFaninLastWriterLoad = 25,
    SharedMetadataLastWriterLoad = 26,
    SharedMetadataLastWriterCommit = 27,
    SharedOutputWriterReserve = 28,
    SharedOutputPublishedExchange = 29,
    SharedMapLookupHeadLoad = 30,
    SharedMapLookupTailLoad = 31,
    SharedMapLookupSeqLoad = 32,
    SharedMapAppendHeadLoad = 33,
    SharedMapAppendTailLoad = 34,
    SharedMapAppendSeqLoad = 35,
    SharedMapAppendSeqResetExchange = 36,
    SharedMapAppendSeqPublishExchange = 37,
    SharedMapAppendTailExchange = 38,
    SharedOutputRollbackExchange = 39,
    SharedClaimTournamentLocal = 40,
    SharedClaimTournamentRoot = 41,
    Count = 42,
    // Stale private BlockWon helpers still have to parse while the shared
    // submit path is compiled from the same translation unit. Values outside
    // Count are intentionally unencodable: executing one makes the trace
    // incomplete instead of silently assigning a schema-v5 meaning.
    WonSlotClaimMax = 0x100,
    WonRemainingExchange = 0x101,
    WonLaneResetExchange = 0x102,
    WonLaneDepositExchange = 0x103,
    WonStatePublishExchange = 0x104,
    WonAnyPublishExchange = 0x105,
    WonAnyLoad = 0x106,
    WonStateLoad = 0x107,
    WonLaneClaimExchange = 0x108,
    WonLaneReleaseExchange = 0x109,
    WonRemainingFetchSub = 0x10a,
    WonStateClearExchange = 0x10b,
    WonDrainedLoad = 0x10c,
#else
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
#endif
};

enum class FdwicAtomicOp : uint32_t {
    Load = 0,
    Exchange = 1,
    FetchAdd = 2,
    FetchMax = 3,
#if PTO_FDWIC_SHARED_MAP
    CompareExchange = 4,
    // Compile-only poison for stale private BlockWon code.
    FetchSub = 0x0f,
#else
    FetchSub = 4,
#endif
};

static_assert(
    static_cast<uint32_t>(FdwicAtomicSite::Count) <= (1U << 16), "atomic site id must fit the compact record"
);
static_assert(
    static_cast<uint32_t>(FdwicAtomicSite::Count) <= UINT16_MAX, "atomic site id must fit the raw auxiliary field"
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

#if PTO_FDWIC_SHARED_MAP
enum class FdwicDcciOp : uint32_t {
    Invalidate = 0,
    CleanOut = 1,
    Count = 2,
};

enum class FdwicDcciSite : uint32_t {
    SharedFaninHistoryInvalidate = 0,
    SharedWriterHistoryFlush = 1,
    SharedOutputRollbackFlush = 2,
    SharedOutputDescriptorFlush = 3,
    SharedRegionReadInvalidate = 4,
    SharedRegionAppendInvalidate = 5,
    SharedRegionAppendFlush = 6,
    SharedWinnerBuildDescriptorInvalidate = 7,
    ObserverTraceExport = 8,
    StartupConfigInvalidate = 9,
    Count = 10,
};

constexpr uint32_t kFdwicDcciOpMask = 0x03U;
constexpr uint32_t kFdwicDcciTrailingDsb = 1U << 2;
constexpr uint32_t kFdwicDcciCallCountShift = 3;
constexpr uint32_t kFdwicDcciCallCountMask = 0x0fU;
constexpr uint32_t kFdwicDcciReservedBit = 1U << 7;
constexpr uint32_t kFdwicDcciLineCountShift = 8;
constexpr uint32_t kFdwicDcciLineCountMax = 0x00ffffffU;

PTO_DEVICE_FUNC constexpr FdwicDcciOp fdwic_dcci_site_op(FdwicDcciSite site) {
    switch (site) {
    case FdwicDcciSite::SharedWriterHistoryFlush:
    case FdwicDcciSite::SharedOutputRollbackFlush:
    case FdwicDcciSite::SharedOutputDescriptorFlush:
    case FdwicDcciSite::SharedRegionAppendFlush:
    case FdwicDcciSite::ObserverTraceExport:
        return FdwicDcciOp::CleanOut;
    default:
        return FdwicDcciOp::Invalidate;
    }
}
#endif

PTO_DEVICE_FUNC constexpr FdwicAtomicOp fdwic_atomic_site_op(FdwicAtomicSite site) {
    switch (site) {
    case FdwicAtomicSite::StartupIncrement:
    case FdwicAtomicSite::ReplayDoneIncrement:
#if PTO_FDWIC_SHARED_MAP
    case FdwicAtomicSite::SharedHeapCursorReserve:
    case FdwicAtomicSite::SharedHeapVendAdvance:
#endif
        return FdwicAtomicOp::FetchAdd;
    case FdwicAtomicSite::FatalSet:
    case FdwicAtomicSite::CompletionVendExchange:
    case FdwicAtomicSite::CompletionFlagExchange:
#if PTO_FDWIC_SHARED_MAP
    case FdwicAtomicSite::SharedOutputPublishedExchange:
    case FdwicAtomicSite::SharedMapAppendSeqResetExchange:
    case FdwicAtomicSite::SharedMapAppendSeqPublishExchange:
    case FdwicAtomicSite::SharedMapAppendTailExchange:
    case FdwicAtomicSite::SharedOutputRollbackExchange:
#else
    case FdwicAtomicSite::WonRemainingExchange:
    case FdwicAtomicSite::WonLaneResetExchange:
    case FdwicAtomicSite::WonLaneDepositExchange:
    case FdwicAtomicSite::WonStatePublishExchange:
    case FdwicAtomicSite::WonAnyPublishExchange:
    case FdwicAtomicSite::WonLaneClaimExchange:
    case FdwicAtomicSite::WonLaneReleaseExchange:
    case FdwicAtomicSite::WonStateClearExchange:
#endif
        return FdwicAtomicOp::Exchange;
    case FdwicAtomicSite::ClaimMax:
    case FdwicAtomicSite::FrontierMax:
#if PTO_FDWIC_SHARED_MAP
    case FdwicAtomicSite::SharedOutputWriterReserve:
#else
    case FdwicAtomicSite::WonSlotClaimMax:
#endif
        return FdwicAtomicOp::FetchMax;
#if PTO_FDWIC_SHARED_MAP
    case FdwicAtomicSite::SharedInsertTurnHandoff:
    case FdwicAtomicSite::SharedMetadataLastWriterCommit:
    case FdwicAtomicSite::SharedClaimTournamentLocal:
    case FdwicAtomicSite::SharedClaimTournamentRoot:
        return FdwicAtomicOp::CompareExchange;
#else
    case FdwicAtomicSite::WonRemainingFetchSub:
        return FdwicAtomicOp::FetchSub;
#endif
    default:
        return FdwicAtomicOp::Load;
    }
}

PTO_DEVICE_FUNC constexpr bool fdwic_atomic_site_result_used(FdwicAtomicSite site) {
#if PTO_FDWIC_SHARED_MAP
    switch (site) {
    case FdwicAtomicSite::StartupIncrement:
    case FdwicAtomicSite::FatalSet:
    case FdwicAtomicSite::CompletionVendExchange:
    case FdwicAtomicSite::CompletionFlagExchange:
    case FdwicAtomicSite::ReplayDoneIncrement:
    case FdwicAtomicSite::SharedOutputRollbackExchange:
    case FdwicAtomicSite::Count:
        return false;
    default:
        return static_cast<uint32_t>(site) < static_cast<uint32_t>(FdwicAtomicSite::Count);
    }
#else
    switch (site) {
    case FdwicAtomicSite::StartupPoll:
    case FdwicAtomicSite::FatalPoll:
    case FdwicAtomicSite::ClaimMax:
    case FdwicAtomicSite::FaninFlagLoad:
    case FdwicAtomicSite::FrontierInitialLoad:
    case FdwicAtomicSite::FrontierFlagLoad:
    case FdwicAtomicSite::FrontierMax:
    case FdwicAtomicSite::HeapFrontierLoad:
    case FdwicAtomicSite::HeapVendLoad:
    case FdwicAtomicSite::ReplayDonePoll:
    case FdwicAtomicSite::WonSlotClaimMax:
    case FdwicAtomicSite::WonAnyLoad:
    case FdwicAtomicSite::WonStateLoad:
    case FdwicAtomicSite::WonLaneClaimExchange:
    case FdwicAtomicSite::WonRemainingFetchSub:
    case FdwicAtomicSite::WonDrainedLoad:
        return true;
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
    case FdwicAtomicSite::Count:
        return false;
    }
    return false;
#endif
}

#if PTO_FDWIC_SHARED_MAP
constexpr uint32_t kFdwicAtomicReturnReadySiteCount = 36;
constexpr uint32_t kFdwicAtomicSourceIssueSiteCount = 6;
#else
constexpr uint32_t kFdwicAtomicReturnReadySiteCount = 16;
constexpr uint32_t kFdwicAtomicSourceIssueSiteCount = 12;
#endif
static_assert(
    kFdwicAtomicReturnReadySiteCount + kFdwicAtomicSourceIssueSiteCount ==
        static_cast<uint32_t>(FdwicAtomicSite::Count),
    "every atomic site must have one explicit completion classification"
);

// Observation loads used by explicit scheduler wait regions are batchable.
// WonLaneClaimExchange is the sole RMW exception: only its idempotent failed
// retries (old value already kDrainedClaimed) are batched, while the successful
// state transition remains a one-call record.
PTO_DEVICE_FUNC constexpr bool fdwic_atomic_site_is_poll_batchable(FdwicAtomicSite site) {
#if PTO_FDWIC_SHARED_MAP
    switch (site) {
    case FdwicAtomicSite::StartupPoll:
    case FdwicAtomicSite::FatalPoll:
    case FdwicAtomicSite::FaninFlagLoad:
    case FdwicAtomicSite::HeapFrontierLoad:
    case FdwicAtomicSite::HeapVendLoad:
    case FdwicAtomicSite::ReplayDonePoll:
    case FdwicAtomicSite::SharedInsertTurnPoll:
        return true;
    default:
        return false;
    }
#else
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
#endif
}

#if PTO_FDWIC_SHARED_MAP
constexpr uint32_t kFdwicAtomicPollBatchSiteCount = 6;
#else
constexpr uint32_t kFdwicAtomicPollBatchSiteCount = 10;
#endif
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
#if !PTO_FDWIC_SHARED_MAP
    case FdwicAtomicSite::WonAnyLoad:
        return 6;
    case FdwicAtomicSite::WonStateLoad:
        return 7;
    case FdwicAtomicSite::WonDrainedLoad:
        return 8;
    case FdwicAtomicSite::WonLaneClaimExchange:
        return 9;
#endif
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
#if !PTO_FDWIC_SHARED_MAP
    case 6:
        return FdwicAtomicSite::WonAnyLoad;
    case 7:
        return FdwicAtomicSite::WonStateLoad;
    case 8:
        return FdwicAtomicSite::WonDrainedLoad;
    case 9:
        return FdwicAtomicSite::WonLaneClaimExchange;
#endif
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

// perf-clock-kernel 与普通 perf-clock 共用同一个 32B tail，但把末尾 8B
// 用于逐核 Kernel 累计时间。构建 profile 和外层 mode 字段负责区分两种
// 解释，避免扩大每核 64B cacheline 或增加多核共享 sidecar。
struct FdwicPerfClockKernelCoreData {
    uint64_t first_submit_start;
    uint64_t last_submit_end;
    uint32_t submit_count;
    uint32_t expected_submit_count;
    uint64_t kernel_elapsed_ticks;
};

static_assert(
    sizeof(FdwicPerfClockKernelCoreData) == 32, "perf-clock-kernel data must fit the existing core-state tail"
);
static_assert(
    offsetof(FdwicPerfClockKernelCoreData, kernel_elapsed_ticks) == 24, "perf-clock-kernel elapsed offset changed"
);

struct FdwicSwimlaneCoreState {
    // perf-clock-kernel 关闭 trace 后复用前 20B：count=Kernel 调用数，
    // dropped=聚合错误状态，atomic_calls/poll_calls=0，
    // poll_batch_records=kFdwicPerfClockKernelMode。普通 perf-clock 仍要求
    // 五个字段全零；两种解释不会静默混用。
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
#if PTO_FDWIC_SHARED_MAP
        struct {
            volatile uint32_t dcci_calls;
            volatile uint32_t dcci_lines;
            volatile uint32_t dcci_records;
            uint32_t padding[5];
        };
#endif
        uint32_t pad[8];
        FdwicPerfClockCoreData perf_clock;
        FdwicPerfClockKernelCoreData perf_clock_kernel;
    };
} __attribute__((aligned(64)));

static_assert(sizeof(FdwicSwimlaneCoreState) == 64, "FdwicSwimlaneCoreState must occupy one cacheline");
static_assert(offsetof(FdwicSwimlaneCoreState, perf_clock) == 32, "perf-clock must reuse the existing 32B tail");
static_assert(
    offsetof(FdwicSwimlaneCoreState, perf_clock_kernel) == 32, "perf-clock-kernel must reuse the existing 32B tail"
);
#if PTO_FDWIC_SHARED_MAP
static_assert(offsetof(FdwicSwimlaneCoreState, dcci_calls) == 32, "DCCI counters must reuse the core-state tail");
static_assert(offsetof(FdwicSwimlaneCoreState, dcci_records) == 40, "DCCI counter layout changed");
#endif

struct FdwicSwimlaneHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_cores;
    uint32_t records_per_core;
    uint64_t freq_hz;
#if PTO_FDWIC_SHARED_MAP
    uint32_t record_size_bytes;
    uint32_t reserved[9];
#else
    uint32_t reserved[10];
#endif
    FdwicSwimlaneCoreState cores[108];
} __attribute__((aligned(64)));

static_assert(sizeof(FdwicSwimlaneHeader) % 64 == 0, "FdwicSwimlaneHeader must be cacheline aligned");
#if PTO_FDWIC_SHARED_MAP
static_assert(offsetof(FdwicSwimlaneHeader, record_size_bytes) == 24, "shared record-size offset changed");
static_assert(offsetof(FdwicSwimlaneHeader, reserved) == 28, "shared header padding offset changed");
#else
static_assert(offsetof(FdwicSwimlaneHeader, reserved) == 24, "private header padding offset changed");
#endif
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

#if PTO_FDWIC_SHARED_MAP
struct FdwicCompactSwimlaneRecord {
    uint32_t start_cycle_low;
    uint32_t end_cycle_low;
    uint32_t flags;
    uint32_t packed;
} __attribute__((aligned(16)));

static_assert(
    sizeof(FdwicCompactSwimlaneRecord) == 16 && alignof(FdwicCompactSwimlaneRecord) == 16,
    "compact generic trace record must occupy 16 bytes"
);
static_assert(offsetof(FdwicCompactSwimlaneRecord, flags) == 8, "compact flags offset changed");
static_assert(offsetof(FdwicCompactSwimlaneRecord, packed) == 12, "compact packed offset changed");

constexpr uint32_t kFdwicCompactTraceTaskBits = 13;
constexpr uint32_t kFdwicCompactTraceTaskMask = (1U << kFdwicCompactTraceTaskBits) - 1U;
constexpr uint32_t kFdwicCompactTraceTaskSentinel = kFdwicCompactTraceTaskMask;
constexpr uint32_t kFdwicCompactTraceFunctionShift = 13;
constexpr uint32_t kFdwicCompactTraceFunctionBits = 3;
constexpr uint32_t kFdwicCompactTraceFunctionMask = (1U << kFdwicCompactTraceFunctionBits) - 1U;
constexpr uint32_t kFdwicCompactTraceFunctionSentinel = kFdwicCompactTraceFunctionMask;
constexpr uint32_t kFdwicCompactTracePhaseShift = 16;
constexpr uint32_t kFdwicCompactTracePhaseBits = 5;
constexpr uint32_t kFdwicCompactTracePhaseMask = (1U << kFdwicCompactTracePhaseBits) - 1U;
constexpr uint32_t kFdwicCompactTraceAuxShift = 21;
constexpr uint32_t kFdwicCompactTraceAuxBits = 11;
constexpr uint32_t kFdwicCompactTraceAuxMask = (1U << kFdwicCompactTraceAuxBits) - 1U;

PTO_DEVICE_FUNC constexpr bool fdwic_compact_trace_fields_fit(
    int32_t task_id, int32_t func_id, FdwicSwimlanePhase phase, uint32_t aux
) {
    return task_id >= -1 && task_id < static_cast<int32_t>(kFdwicSharedTraceTaskCapacity) && func_id >= -1 &&
           func_id <= 3 && static_cast<uint32_t>(phase) < static_cast<uint32_t>(FdwicSwimlanePhase::Count) &&
           aux <= kFdwicCompactTraceAuxMask;
}

PTO_DEVICE_FUNC constexpr uint32_t fdwic_pack_compact_trace_fields(
    int32_t task_id, int32_t func_id, FdwicSwimlanePhase phase, uint32_t aux
) {
    return (static_cast<uint32_t>(task_id) & kFdwicCompactTraceTaskMask) |
           ((static_cast<uint32_t>(func_id) & kFdwicCompactTraceFunctionMask)
            << kFdwicCompactTraceFunctionShift) |
           (static_cast<uint32_t>(phase) << kFdwicCompactTracePhaseShift) | (aux << kFdwicCompactTraceAuxShift);
}

struct FdwicSharedSubmitClaimRecord {
    uint64_t claim_begin;
    uint64_t claim_end_and_winner;
    uint64_t submit_begin;
    uint64_t submit_end;
} __attribute__((aligned(32)));

constexpr uint64_t kFdwicSharedClaimWinnerBit = 1ULL << 63;
static_assert(
    sizeof(FdwicSharedSubmitClaimRecord) == kFdwicSharedSubmitClaimRecordSizeBytes &&
        alignof(FdwicSharedSubmitClaimRecord) == 32,
    "shared Submit/Claim record must remain 32 bytes"
);
static_assert(offsetof(FdwicSharedSubmitClaimRecord, submit_begin) == 16, "shared Submit offset changed");

using FdwicSwimlaneStorageRecord = FdwicCompactSwimlaneRecord;
constexpr size_t kFdwicSharedSubmitClaimBytesPerCore =
    static_cast<size_t>(kFdwicSharedTraceTaskCapacity) * sizeof(FdwicSharedSubmitClaimRecord);
constexpr size_t kFdwicSwimlaneGenericBytesPerCore =
    static_cast<size_t>(kFdwicSwimlaneDefaultRecordsPerCore) * sizeof(FdwicSwimlaneStorageRecord);
constexpr size_t kFdwicSwimlaneWorkerBytes =
    kFdwicSharedSubmitClaimBytesPerCore + kFdwicSwimlaneGenericBytesPerCore;
static_assert(kFdwicSwimlaneWorkerBytes == 593920, "shared trace worker stride changed");
#else
using FdwicSwimlaneStorageRecord = FdwicSwimlaneRecord;
constexpr size_t kFdwicSwimlaneGenericBytesPerCore =
    static_cast<size_t>(kFdwicSwimlaneDefaultRecordsPerCore) * sizeof(FdwicSwimlaneStorageRecord);
constexpr size_t kFdwicSwimlaneWorkerBytes = kFdwicSwimlaneGenericBytesPerCore;
static_assert(kFdwicSwimlaneWorkerBytes == (2U << 20), "private trace worker stride changed");
#endif

static_assert(sizeof(FdwicSwimlaneStorageRecord) == kFdwicSwimlaneRecordSizeBytes, "record-size identity changed");
static_assert(kFdwicSwimlaneWorkerBytes % 64 == 0, "worker trace partition must be cache-line aligned");
