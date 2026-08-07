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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_G0_SWIMLANE_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_G0_SWIMLANE_H

#include <stddef.h>
#include <stdint.h>

#include "g0_full_pa.h"

namespace pa_scheduler::simt_cross_core::g0_swimlane {

using namespace pa_scheduler::simt_cross_core::g0;

constexpr uint64_t kTraceMagic = UINT64_C(0x47305357494D4C4E);
constexpr uint64_t kTraceVersion = 5U;
constexpr uint64_t kTracePoison = UINT64_C(0xD3D3D3D3D3D3D3D3);
constexpr uint32_t kTraceTaskCapacity = kDefaultBatches * kTasksPerBatch;
constexpr uint32_t kTraceSimtWriterCount = kMaxBuilderLeaderCount;
constexpr uint32_t kTraceScalarWriterCount = kOwnerCount;
// profiling 变体同时覆盖 1..32 builder。数组按本次构建的最大 writer 数预留；
// control.simt_writer_count 只记录实际启动的 writer。每 writer 容量按其最多
// task 数精确推导，避免降低 warp 数时用固定大数组把 sidecar 放大。
constexpr uint32_t kTraceScalarRecordsPerWriter = 512U;
constexpr uint32_t kTraceNoTask = UINT32_MAX;
// 单 builder 是每个 writer 的最坏任务数。当前协议每 task 最多
// 40 条非 poll/poll-episode 记录，再给 builder 启停/异常保留 16 条；
// 这个静态上界允许 SIMT writer 去掉每条 raw 路径上的分歧容量检查。
constexpr uint32_t kTraceSimtMaxTasksPerWriter =
    (kTraceTaskCapacity + kBuilderWarpCount - 1U) / kBuilderWarpCount;
constexpr uint32_t kTraceSimtMaxRecordsPerTask = 40U;
constexpr uint32_t kTraceSimtFixedRecordsPerWriter = 16U;
constexpr uint32_t kTraceSimtRecordsPerWriter =
    kTraceSimtMaxTasksPerWriter * kTraceSimtMaxRecordsPerTask + kTraceSimtFixedRecordsPerWriter;
static_assert(
    kTraceSimtMaxTasksPerWriter * kTraceSimtMaxRecordsPerTask + kTraceSimtFixedRecordsPerWriter <=
        kTraceSimtRecordsPerWriter,
    "SIMT trace writer capacity no longer covers the statically bounded G0 path"
);

enum class TraceDomain : uint32_t {
    Scalar = 0U,
    Simt = 1U,
    Count = 2U,
};

enum class TraceKind : uint8_t {
    Atomic = 0U,
    Dcci = 1U,
    Count = 2U,
};

enum class AtomicOp : uint8_t {
    Load = 0U,
    Exchange = 1U,
    FetchAdd = 2U,
    CompareExchange = 3U,
    Count = 4U,
};

// 编号是 G0 swimlane raw ABI 的一部分，只能在 Count 前追加。名称按
// 真实协议调用点拆分；相同源码循环的 load 不逐次写 raw，而由对应 Poll
// site 保存一个完整等待区间和精确 call_count。
enum class AtomicSite : uint16_t {
    FatalLoad = 0U,
    FatalSet = 1U,
    SimtBuilderStartedIncrement = 2U,
    SimtBuilderStartedPoll = 3U,
    SimtTaskBuildAttemptIncrement = 4U,
    SimtTaskBuildPreparedIncrement = 5U,
    SimtTaskBuildClaim = 6U,
    SimtHeapShardReserve = 7U,
    SimtHeapVendReserve = 8U,
    SimtHeapVendLoad = 9U,
    SimtTaskBasePublish = 10U,
    SimtCompletionVendReportPublish = 11U,
    SimtOutputPublishedPublish = 12U,
    SimtOutputLastWriterPublish = 13U,
    SimtProducerTaskBasePoll = 14U,
    SimtInsertPredecessorPoll = 15U,
    SimtMetadataLastWriterCommit = 16U,
    SimtInsertCompletionPublish = 17U,
    SimtAllocCompletionVendPublish = 18U,
    SimtAllocCompletionFlagPublish = 19U,
    SimtAllocDoneIncrement = 20U,
    SimtExecBuiltPublish = 21U,
    SimtBuildReportPublish = 22U,
    SimtBuilderFinishedPublish = 23U,
    ScalarDispatchTicket = 24U,
    ScalarProducerTaskBaseLoad = 25U,
    ScalarExecStatePoll = 26U,
    ScalarExecClaim = 27U,
    ScalarFaninFlagPoll = 28U,
    ScalarExecutionWitnessPublish = 29U,
    ScalarCompletionVendPublish = 30U,
    ScalarCompletionFlagPublish = 31U,
    ScalarExecDonePublish = 32U,
    ScalarDoneCountIncrement = 33U,
    ScalarEngineDoneIncrement = 34U,
    ScalarDrainArrive = 35U,
    ScalarDrainArrivalPoll = 36U,
    ScalarDrainVerifyLoad = 37U,
    ScalarRootFinishedPublish = 38U,
    SimtMetadataOutputPublishedPoll = 39U,
    SimtMetadataLastWriterLoad = 40U,
    Count = 41U,
};

enum class DcciOp : uint8_t {
    Invalidate = 0U,
    CleanOut = 1U,
    Count = 2U,
};

enum class DcciSite : uint16_t {
    StartupConfigInvalidate = 0U,
    DispatchTaskIdInvalidate = 1U,
    ExecPayloadInvalidate = 2U,
    TerminalTokenInvalidate = 3U,
    Count = 4U,
};

constexpr uint32_t kAtomicResultUsed = 1U << 0U;
constexpr uint32_t kAtomicReturnReady = 1U << 1U;
constexpr uint32_t kAtomicValueZero = 1U << 2U;
constexpr uint32_t kAtomicPollBatch = 1U << 3U;
constexpr uint32_t kDcciTrailingDsb = 1U << 0U;
constexpr uint32_t kDcciLineCountShift = 8U;
constexpr uint32_t kDcciLineCountMax = 0x00FFFFFFU;

#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_G0_TRACE_INLINE __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_G0_TRACE_INLINE constexpr
#endif

SIMT_CROSS_CORE_G0_TRACE_INLINE AtomicOp AtomicSiteExpectedOp(AtomicSite site) {
    switch (site) {
    case AtomicSite::FatalLoad:
    case AtomicSite::SimtBuilderStartedPoll:
    case AtomicSite::SimtHeapVendLoad:
    case AtomicSite::SimtProducerTaskBasePoll:
    case AtomicSite::SimtInsertPredecessorPoll:
    case AtomicSite::SimtMetadataOutputPublishedPoll:
    case AtomicSite::SimtMetadataLastWriterLoad:
    case AtomicSite::ScalarProducerTaskBaseLoad:
    case AtomicSite::ScalarExecStatePoll:
    case AtomicSite::ScalarFaninFlagPoll:
    case AtomicSite::ScalarDrainArrivalPoll:
    case AtomicSite::ScalarDrainVerifyLoad:
        return AtomicOp::Load;
    case AtomicSite::ScalarCompletionVendPublish:
    case AtomicSite::ScalarCompletionFlagPublish:
        return AtomicOp::Exchange;
    case AtomicSite::SimtBuilderStartedIncrement:
    case AtomicSite::SimtTaskBuildAttemptIncrement:
    case AtomicSite::SimtTaskBuildPreparedIncrement:
    case AtomicSite::SimtHeapShardReserve:
    case AtomicSite::SimtHeapVendReserve:
    case AtomicSite::SimtInsertCompletionPublish:
    case AtomicSite::SimtAllocDoneIncrement:
    case AtomicSite::ScalarDispatchTicket:
    case AtomicSite::ScalarDoneCountIncrement:
    case AtomicSite::ScalarEngineDoneIncrement:
    case AtomicSite::ScalarDrainArrive:
        return AtomicOp::FetchAdd;
    default:
        return AtomicOp::CompareExchange;
    }
}

SIMT_CROSS_CORE_G0_TRACE_INLINE bool AtomicSiteIsPoll(AtomicSite site) {
    return site == AtomicSite::SimtBuilderStartedPoll ||
           site == AtomicSite::SimtProducerTaskBasePoll ||
           site == AtomicSite::SimtInsertPredecessorPoll ||
           site == AtomicSite::SimtMetadataOutputPublishedPoll ||
           site == AtomicSite::ScalarExecStatePoll ||
           site == AtomicSite::ScalarFaninFlagPoll ||
           site == AtomicSite::ScalarDrainArrivalPoll;
}

SIMT_CROSS_CORE_G0_TRACE_INLINE uint32_t PackDcciFlags(uint32_t line_count) {
    return kDcciTrailingDsb | (line_count << kDcciLineCountShift);
}

SIMT_CROSS_CORE_G0_TRACE_INLINE uint32_t DcciLineCount(uint32_t flags) {
    return flags >> kDcciLineCountShift;
}

enum ExecutorTraceBits : uint32_t {
    kExecutorTicketRecorded = 1U << 0U,
    kExecutorClaimRecorded = 1U << 1U,
    kExecutorFaninReadyRecorded = 1U << 2U,
    kExecutorBeginRecorded = 1U << 3U,
    kExecutorEndRecorded = 1U << 4U,
};

constexpr uint32_t kExpectedExecutorTraceBits = kExecutorTicketRecorded | kExecutorClaimRecorded |
                                                kExecutorFaninReadyRecorded | kExecutorBeginRecorded |
                                                kExecutorEndRecorded;

// Builder 与 executor 会同时处理同一 task。两类记录故意拆成独立 cache line，
// 避免 profiling 本身引入同一 cache line 的跨执行单元写竞争。
struct alignas(kCacheLineBytes) BuilderTaskTrace {
    uint64_t launch_nonce;
    uint64_t attempt_begin;
    uint64_t claim_end;
    uint64_t prepare_end;
    uint64_t commit_begin;
    uint64_t commit_end;
    uint64_t report_end;
    uint32_t task_id;
    uint32_t builder_thread;
    uint32_t build_owner;
    uint32_t insert_poll_count;
    uint64_t reserved[7];
};

struct alignas(kCacheLineBytes) ExecutorTaskTrace {
    uint64_t launch_nonce;
    uint64_t ticket_assigned;
    uint64_t claim_end;
    uint64_t fanin_ready;
    uint64_t execute_begin;
    uint64_t execute_end;
    uint32_t task_id;
    uint32_t execute_owner;
    uint32_t task_kind;
    uint32_t phase_bits;
};

struct alignas(kCacheLineBytes) RoleTrace {
    uint64_t launch_nonce;
    uint64_t entry;
    uint64_t config_ready;
    uint64_t work_begin;
    uint64_t work_end;
    uint64_t drain_begin;
    uint64_t drain_end;
    uint64_t exit;
    uint32_t owner;
    uint32_t role;
    uint32_t physical_block;
    uint32_t subblock;
    uint64_t reserved[6];
};

struct alignas(kCacheLineBytes) TraceControl {
    uint64_t magic;
    uint64_t version;
    uint64_t launch_nonce;
    uint64_t tick_ns;
    uint32_t task_count;
    uint32_t kernel_task_count;
    uint32_t builder_count;
    uint32_t role_count;
    uint32_t simt_writer_count;
    uint32_t scalar_writer_count;
    uint32_t simt_records_per_writer;
    uint32_t scalar_records_per_writer;
    uint32_t record_size_bytes;
    uint32_t reserved32;
    uint64_t reserved[7];
};

// 32B 定长 raw：前两字是时间端点，第三字保存 task/site/kind/op，第四
// 字保存 flags/call_count。DCCI 的 flags[31:8] 为实际 cache-line 数；
// atomic PollBatch 的 call_count 为整个等待 episode 的精确 load 次数。
struct alignas(32U) TraceRecord {
    uint64_t begin;
    uint64_t end;
    uint32_t task_id;
    uint16_t site;
    TraceKind kind;
    uint8_t op;
    uint32_t flags;
    uint32_t call_count;
};

struct alignas(kCacheLineBytes) TraceLogControl {
    uint64_t launch_nonce;
    uint64_t atomic_calls;
    uint64_t poll_calls;
    uint64_t dcci_calls;
    uint64_t dcci_lines;
    uint32_t record_count;
    uint32_t dropped_records;
    uint32_t poll_records;
    uint32_t dcci_records;
    uint32_t writer_id;
    TraceDomain domain;
};

struct alignas(kCacheLineBytes) TraceState {
    TraceControl control;
    BuilderTaskTrace builders[kTraceTaskCapacity];
    ExecutorTaskTrace executors[kTraceTaskCapacity];
    RoleTrace roles[kOwnerCount];
    TraceLogControl simt_logs[kTraceSimtWriterCount];
    TraceRecord simt_records[kTraceSimtWriterCount][kTraceSimtRecordsPerWriter];
    TraceLogControl scalar_logs[kTraceScalarWriterCount];
    TraceRecord scalar_records[kTraceScalarWriterCount][kTraceScalarRecordsPerWriter];
};

struct alignas(kCacheLineBytes) G0SwimlaneState {
    FullPaState full_pa;
    TraceState trace;
};

static_assert(kTraceTaskCapacity == 1280U, "G0 B256 trace capacity changed");
static_assert(sizeof(BuilderTaskTrace) == 2U * kCacheLineBytes, "builder trace must occupy two cache lines");
static_assert(sizeof(ExecutorTaskTrace) == kCacheLineBytes, "executor trace must occupy one cache line");
static_assert(sizeof(RoleTrace) == 2U * kCacheLineBytes, "role trace must occupy two cache lines");
static_assert(sizeof(TraceControl) == 2U * kCacheLineBytes, "trace control must occupy two cache lines");
static_assert(sizeof(TraceRecord) == 32U, "G0 atomic/DCCI raw record ABI changed");
static_assert(sizeof(TraceLogControl) == kCacheLineBytes, "one writer control must occupy one cache line");
static_assert(offsetof(G0SwimlaneState, full_pa) == 0U, "profile state must preserve the G0 state base address");
static_assert(
    offsetof(G0SwimlaneState, trace) % kCacheLineBytes == 0U && sizeof(G0SwimlaneState) % kCacheLineBytes == 0U,
    "profile regions must remain cache-line aligned"
);

}  // namespace pa_scheduler::simt_cross_core::g0_swimlane

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_G0_SWIMLANE_H
