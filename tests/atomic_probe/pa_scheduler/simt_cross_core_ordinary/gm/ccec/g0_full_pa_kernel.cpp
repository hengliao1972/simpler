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

// This GM variant keeps the PA task ABI and task graph while moving Build
// into independent SIMT warp leaders. Only real metadata writers enter one
// task-id-ordered global insert chain; Main Scalar launches the VF and joins
// the global drain without building or inserting tasks.

#include <pto/common/kernel_meta.hpp>
#include <pto/pto-inst.hpp>

#include "cce_aicore_intrinsics.h"

#if defined(__DAV_VEC__)
#include "simt_api/asc_simt.h"
#endif

#include "../common/g0_full_pa.h"
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
#include "../common/g0_swimlane.h"
#endif
#include "full_pa_workloads.h"

#if defined(SIMT_CROSS_CORE_G0_SWIMLANE) && !defined(SIMT_CROSS_CORE_G0_DISABLE_SIMT_ATOMIC_TRACE)
#define SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED
#endif

namespace {

using namespace pa_scheduler::simt_cross_core_ordinary::g0;
using namespace pa_scheduler::simt_cross_core_ordinary::g0::device;
using namespace pto;

constexpr int kSingleCacheLine = 0;
constexpr uint32_t kWatchdogMask = 0x3FFU;
constexpr uint32_t kDrainExpectedArrivals = 6U;
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
namespace g0_swimlane = pa_scheduler::simt_cross_core_ordinary::g0_swimlane;
constexpr size_t kG0TraceFromTasksOffsetBytes =
    offsetof(g0_swimlane::G0SwimlaneState, trace) -
    offsetof(g0_swimlane::G0SwimlaneState, full_pa) - offsetof(FullPaState, tasks);
static_assert(
    offsetof(g0_swimlane::G0SwimlaneState, trace) >
            offsetof(g0_swimlane::G0SwimlaneState, full_pa) + offsetof(FullPaState, tasks) &&
        kG0TraceFromTasksOffsetBytes % alignof(g0_swimlane::TraceState) == 0U,
    "G0 trace sidecar must remain aligned and forward-addressable from task_words"
);
#endif

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarAtomicLoad(__gm__ volatile int64_t *address) {
    return static_cast<uint64_t>(atomicAdd(const_cast<__gm__ int64_t *>(address), static_cast<int64_t>(0)));
}

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarAtomicLoad(__gm__ volatile uint64_t *address) {
    return atomicAdd(const_cast<__gm__ uint64_t *>(address), static_cast<uint64_t>(0U));
}

__aicore__ __attribute__((always_inline)) inline uint64_t
ScalarCas(__gm__ volatile int64_t *address, uint64_t expected, uint64_t desired) {
    return static_cast<uint64_t>(
        atomicCAS(const_cast<__gm__ int64_t *>(address), static_cast<int64_t>(expected), static_cast<int64_t>(desired))
    );
}

__aicore__ __attribute__((always_inline)) inline uint64_t
ScalarFetchAdd(__gm__ volatile int64_t *address, uint64_t increment) {
    return static_cast<uint64_t>(atomicAdd(const_cast<__gm__ int64_t *>(address), static_cast<int64_t>(increment)));
}

__aicore__ __attribute__((always_inline)) inline uint64_t
ScalarExchange(__gm__ volatile int64_t *address, uint64_t value) {
    return static_cast<uint64_t>(atomicExch(const_cast<__gm__ int64_t *>(address), static_cast<int64_t>(value)));
}

__aicore__ __attribute__((always_inline)) inline uint64_t LoadDev64(__gm__ const uint64_t *address) {
    return static_cast<uint64_t>(__builtin_cce_ld_dev(const_cast<__gm__ uint64_t *>(address), 0));
}

__aicore__ __attribute__((always_inline)) inline void StoreDev64(__gm__ uint64_t *address, uint64_t value) {
    __builtin_cce_st_dev(value, address, 0);
}

__aicore__ __attribute__((always_inline)) inline uint64_t LoadFatal(__gm__ FullPaState *state) {
    return ScalarAtomicLoad(&state->fatal.state);
}

__aicore__ __attribute__((always_inline)) inline void
PublishFatal(__gm__ FullPaState *state, ExecFatalReason reason, uint32_t owner, uint32_t task_id) {
    (void)ScalarCas(&state->fatal.state, 0U, EncodeExecFatal(reason, owner, task_id));
}

__aicore__ __attribute__((always_inline)) inline bool ConfigValid(__gm__ const FullPaState *state) {
    const uint32_t batches = state->control.batch_count;
    return state->control.magic == kProbeMagic && state->control.version == kProbeVersion &&
           state->control.timeout_ticks != 0U && batches >= 1U && batches <= kDefaultBatches &&
           state->control.task_count == TaskCount(batches) &&
           state->control.kernel_task_count == KernelTaskCount(batches) &&
           state->control.builder_thread_count == kBuilderThreadCount &&
           BuilderCountValid(state->control.builder_count) && state->control.heap_base == kSyntheticHeapBase &&
           state->control.heap_bytes == kHeapBytes && state->control.workspace_base != 0U &&
           state->control.workspace_bytes == kWorkloadBytes && state->control.qk_repeats >= 1U &&
           state->control.sf_repeats >= 1U && state->control.pv_repeats >= 1U && state->control.up_repeats >= 1U &&
           state->exec_dispatch.aic_task_count == batches * 2U && state->exec_dispatch.aiv_task_count == batches * 2U;
}

#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
__aicore__ __attribute__((always_inline)) inline __gm__ g0_swimlane::TraceState *
TraceStateFor(__gm__ FullPaState *state) {
    return &reinterpret_cast<__gm__ g0_swimlane::G0SwimlaneState *>(state)->trace;
}

struct alignas(32U) ScalarTraceContext {
    __gm__ g0_swimlane::TraceLogControl *control;
    __gm__ g0_swimlane::TraceRecord *records;
    uint64_t launch_nonce;
    uint64_t atomic_calls;
    uint64_t poll_calls;
    uint64_t dcci_calls;
    uint64_t dcci_lines;
    uint32_t record_count;
    uint32_t dropped_records;
    uint32_t poll_records;
    uint32_t dcci_records;
    uint32_t owner;
};

struct alignas(32U) ScalarPollEpisode {
    uint64_t begin;
    uint64_t end;
    uint32_t task_id;
    uint32_t call_count;
    g0_swimlane::AtomicSite site;
    uint16_t reserved;
};

static_assert(alignof(ScalarTraceContext) == 32U, "Scalar trace context must be VEC-stack aligned");
static_assert(alignof(ScalarPollEpisode) == 32U, "Scalar poll episode must be VEC-stack aligned");

__aicore__ __attribute__((always_inline)) inline void
InitializeScalarPollEpisode(ScalarPollEpisode *episode) {
    // CCEC 可能把栈上聚合数组的 `{}` 初始化降成不可靠的 VEC-UB
    // copy/store。逐字段标量写入，保证没有领取 task 的 executor 在退出
    // flush 时也一定看到 call_count=0。
    episode->begin = 0U;
    episode->end = 0U;
    episode->task_id = g0_swimlane::kTraceNoTask;
    episode->call_count = 0U;
    episode->site = g0_swimlane::AtomicSite::Count;
    episode->reserved = 0U;
}

__aicore__ __attribute__((always_inline)) inline void
AttachScalarTrace(__gm__ FullPaState *state, uint32_t owner, ScalarTraceContext *trace) {
    // CCEC 可能把聚合零初始化/按值返回降成 UB 上的 VEC
    // store/copy。这里保持 32B 对齐并只做标量逐字段初始化。
    trace->control = &TraceStateFor(state)->scalar_logs[owner];
    trace->records = &TraceStateFor(state)->scalar_records[owner][0];
    trace->launch_nonce = state->control.launch_nonce;
    trace->atomic_calls = 0U;
    trace->poll_calls = 0U;
    trace->dcci_calls = 0U;
    trace->dcci_lines = 0U;
    trace->record_count = 0U;
    trace->dropped_records = 0U;
    trace->poll_records = 0U;
    trace->dcci_records = 0U;
    trace->owner = owner;
}

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarNowAfterAtomicResult(uint64_t value) {
    uint64_t cycle = 0U;
    // 与 cross_core 使用同一条返回依赖序列：先消费 atomic 返回寄存器，
    // 再读 SYS_CNT；不增加 DSB/ISB/GM 访问。
    asm volatile(
        "MOV %0, %0\n"
        "MOV %1, SYS_CNT\n"
        : "+l"(value), "=&l"(cycle)
    );
    return cycle;
}

__aicore__ __attribute__((always_inline)) inline void ScalarTraceWriteRecord(
    ScalarTraceContext &trace, uint64_t begin, uint64_t end, uint32_t task_id, uint16_t site,
    g0_swimlane::TraceKind kind, uint8_t op, uint32_t flags, uint32_t call_count
) {
    if (trace.record_count >= g0_swimlane::kTraceScalarRecordsPerWriter) {
        if (trace.dropped_records != UINT32_MAX) {
            ++trace.dropped_records;
        }
        return;
    }
    __gm__ uint64_t *words =
        reinterpret_cast<__gm__ uint64_t *>(&trace.records[trace.record_count++]);
    const uint64_t meta = static_cast<uint64_t>(task_id) | (static_cast<uint64_t>(site) << 32U) |
                          (static_cast<uint64_t>(kind) << 48U) | (static_cast<uint64_t>(op) << 56U);
    StoreDev64(words + 0U, begin);
    StoreDev64(words + 1U, end);
    StoreDev64(words + 2U, meta);
    StoreDev64(words + 3U, static_cast<uint64_t>(flags) | (static_cast<uint64_t>(call_count) << 32U));
}

__aicore__ __attribute__((always_inline)) inline void ScalarTraceAtomicRecord(
    ScalarTraceContext &trace, uint32_t task_id, g0_swimlane::AtomicSite site, uint64_t begin, uint64_t end,
    g0_swimlane::AtomicOp op, bool result_used, bool value_zero
) {
    // ScalarNowAfterAtomicResult 先消费 atomic 返回寄存器再读
    // SYS_CNT，因此这里可以如实标为 return_ready。
    uint32_t flags = result_used ? g0_swimlane::kAtomicResultUsed | g0_swimlane::kAtomicReturnReady : 0U;
    flags |= value_zero ? g0_swimlane::kAtomicValueZero : 0U;
    ++trace.atomic_calls;
    ScalarTraceWriteRecord(
        trace, begin, end, task_id, static_cast<uint16_t>(site), g0_swimlane::TraceKind::Atomic,
        static_cast<uint8_t>(op), flags, 1U
    );
}

__aicore__ __attribute__((always_inline)) inline void
ScalarTraceFlushPoll(ScalarTraceContext &trace, ScalarPollEpisode *episode) {
    if (episode->call_count == 0U) {
        return;
    }
    // episode->end 在热循环中暂存最后一次 atomic 返回值；只在退出
    // episode 时建立一次返回依赖并读取 SYS_CNT，避免每轮 poll 都
    // 读取时钟而反过来放大等待。
    const uint64_t end = ScalarNowAfterAtomicResult(episode->end);
    trace.atomic_calls += episode->call_count;
    trace.poll_calls += episode->call_count;
    ++trace.poll_records;
    ScalarTraceWriteRecord(
        trace, episode->begin, end, episode->task_id, static_cast<uint16_t>(episode->site),
        g0_swimlane::TraceKind::Atomic, static_cast<uint8_t>(g0_swimlane::AtomicOp::Load),
        g0_swimlane::kAtomicResultUsed | g0_swimlane::kAtomicReturnReady | g0_swimlane::kAtomicPollBatch,
        episode->call_count
    );
    InitializeScalarPollEpisode(episode);
}

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarTracePollLoad(
    ScalarTraceContext &trace, ScalarPollEpisode *episode, uint32_t task_id, g0_swimlane::AtomicSite site,
    __gm__ volatile int64_t *address
) {
    if (episode->call_count != 0U && (episode->task_id != task_id || episode->site != site)) {
        ScalarTraceFlushPoll(trace, episode);
    }
    if (episode->call_count == 0U) {
        episode->begin = static_cast<uint64_t>(get_sys_cnt());
        episode->task_id = task_id;
        episode->site = site;
    }
    const uint64_t old = ScalarAtomicLoad(address);
    episode->end = old;
    if (episode->call_count != UINT32_MAX) {
        ++episode->call_count;
    }
    return old;
}

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarTraceAtomicLoad(
    ScalarTraceContext &trace, uint32_t task_id, g0_swimlane::AtomicSite site,
    __gm__ volatile int64_t *address, bool result_used = true
) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    const uint64_t old = ScalarAtomicLoad(address);
    const uint64_t end = result_used ? ScalarNowAfterAtomicResult(old) : static_cast<uint64_t>(get_sys_cnt());
    ScalarTraceAtomicRecord(
        trace, task_id, site, begin, end, g0_swimlane::AtomicOp::Load, result_used, old == 0U
    );
    return old;
}

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarTraceAtomicLoad(
    ScalarTraceContext &trace, uint32_t task_id, g0_swimlane::AtomicSite site,
    __gm__ volatile uint64_t *address, bool result_used = true
) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    const uint64_t old = ScalarAtomicLoad(address);
    const uint64_t end = result_used ? ScalarNowAfterAtomicResult(old) : static_cast<uint64_t>(get_sys_cnt());
    ScalarTraceAtomicRecord(
        trace, task_id, site, begin, end, g0_swimlane::AtomicOp::Load, result_used, old == 0U
    );
    return old;
}

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarTraceCas(
    ScalarTraceContext &trace, uint32_t task_id, g0_swimlane::AtomicSite site,
    __gm__ volatile int64_t *address, uint64_t expected, uint64_t desired, bool result_used = true
) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    const uint64_t old = ScalarCas(address, expected, desired);
    const uint64_t end = result_used ? ScalarNowAfterAtomicResult(old) : static_cast<uint64_t>(get_sys_cnt());
    ScalarTraceAtomicRecord(
        trace, task_id, site, begin, end, g0_swimlane::AtomicOp::CompareExchange, result_used, false
    );
    return old;
}

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarTraceFetchAdd(
    ScalarTraceContext &trace, uint32_t task_id, g0_swimlane::AtomicSite site,
    __gm__ volatile int64_t *address, uint64_t increment, bool result_used = false
) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    const uint64_t old = ScalarFetchAdd(address, increment);
    const uint64_t end = result_used ? ScalarNowAfterAtomicResult(old) : static_cast<uint64_t>(get_sys_cnt());
    ScalarTraceAtomicRecord(
        trace, task_id, site, begin, end, g0_swimlane::AtomicOp::FetchAdd, result_used, false
    );
    return old;
}

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarTraceExchange(
    ScalarTraceContext &trace, uint32_t task_id, g0_swimlane::AtomicSite site,
    __gm__ volatile int64_t *address, uint64_t value, bool result_used = true
) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    const uint64_t old = ScalarExchange(address, value);
    const uint64_t end = result_used ? ScalarNowAfterAtomicResult(old) : static_cast<uint64_t>(get_sys_cnt());
    ScalarTraceAtomicRecord(
        trace, task_id, site, begin, end, g0_swimlane::AtomicOp::Exchange, result_used, false
    );
    return old;
}

__aicore__ __attribute__((always_inline)) inline void ScalarTraceDcciRecord(
    ScalarTraceContext &trace, uint32_t task_id, g0_swimlane::DcciSite site, g0_swimlane::DcciOp op,
    uint64_t begin, uint64_t end, uint32_t call_count, uint32_t line_count
) {
    trace.dcci_calls += call_count;
    trace.dcci_lines += line_count;
    ++trace.dcci_records;
    ScalarTraceWriteRecord(
        trace, begin, end, task_id, static_cast<uint16_t>(site), g0_swimlane::TraceKind::Dcci,
        static_cast<uint8_t>(op), g0_swimlane::PackDcciFlags(line_count), call_count
    );
}

__aicore__ __attribute__((always_inline)) inline void ScalarTraceFinish(ScalarTraceContext &trace) {
    __gm__ uint64_t *words = reinterpret_cast<__gm__ uint64_t *>(trace.control);
    StoreDev64(words + 1U, trace.atomic_calls);
    StoreDev64(words + 2U, trace.poll_calls);
    StoreDev64(words + 3U, trace.dcci_calls);
    StoreDev64(words + 4U, trace.dcci_lines);
    StoreDev64(
        words + 5U, static_cast<uint64_t>(trace.record_count) |
                        (static_cast<uint64_t>(trace.dropped_records) << 32U)
    );
    StoreDev64(
        words + 6U,
        static_cast<uint64_t>(trace.poll_records) | (static_cast<uint64_t>(trace.dcci_records) << 32U)
    );
    StoreDev64(
        words + 7U, static_cast<uint64_t>(trace.owner) |
                        (static_cast<uint64_t>(g0_swimlane::TraceDomain::Scalar) << 32U)
    );
    StoreDev64(words + 0U, trace.launch_nonce);
}

__aicore__ __attribute__((always_inline)) inline __gm__ uint64_t *
RoleTraceWords(__gm__ FullPaState *state, uint32_t owner) {
    return reinterpret_cast<__gm__ uint64_t *>(&TraceStateFor(state)->roles[owner]);
}

__aicore__ __attribute__((always_inline)) inline void TraceRoleEnter(
    __gm__ FullPaState *state, uint32_t owner, OwnerRole role, uint32_t physical_block, uint32_t subblock,
    uint64_t entry
) {
    __gm__ uint64_t *words = RoleTraceWords(state, owner);
    StoreDev64(words + 1U, entry);
    StoreDev64(words + 8U, static_cast<uint64_t>(owner) | (static_cast<uint64_t>(role) << 32U));
    StoreDev64(words + 9U, static_cast<uint64_t>(physical_block) | (static_cast<uint64_t>(subblock) << 32U));
    StoreDev64(words, state->control.launch_nonce);
}

__aicore__ __attribute__((always_inline)) inline void
TraceRoleTimestamp(__gm__ FullPaState *state, uint32_t owner, uint32_t word) {
    StoreDev64(RoleTraceWords(state, owner) + word, static_cast<uint64_t>(get_sys_cnt()));
}

__aicore__ __attribute__((always_inline)) inline __gm__ uint64_t *
ExecutorTraceWords(__gm__ FullPaState *state, uint32_t task_id) {
    return reinterpret_cast<__gm__ uint64_t *>(&TraceStateFor(state)->executors[task_id]);
}

__aicore__ __attribute__((always_inline)) inline void
TraceExecutorTicket(__gm__ FullPaState *state, uint32_t owner, uint32_t task_id) {
    __gm__ uint64_t *words = ExecutorTraceWords(state, task_id);
    StoreDev64(words + 1U, static_cast<uint64_t>(get_sys_cnt()));
    StoreDev64(
        words + 7U,
        static_cast<uint64_t>(owner) |
            (static_cast<uint64_t>(g0_swimlane::PackExecutorTraceState(
                 TaskKindAt(task_id), g0_swimlane::kExecutorTicketRecorded
             ))
             << 32U)
    );
}

__aicore__ __attribute__((always_inline)) inline void
TraceExecutorClaim(__gm__ FullPaState *state, uint32_t owner, uint32_t task_id) {
    __gm__ uint64_t *words = ExecutorTraceWords(state, task_id);
    StoreDev64(words + 2U, static_cast<uint64_t>(get_sys_cnt()));
    StoreDev64(
        words + 7U,
        static_cast<uint64_t>(owner) |
            (static_cast<uint64_t>(g0_swimlane::PackExecutorTraceState(
                 TaskKindAt(task_id),
                 g0_swimlane::kExecutorTicketRecorded | g0_swimlane::kExecutorClaimRecorded
             ))
             << 32U)
    );
}

__aicore__ __attribute__((always_inline)) inline void
TraceExecutorFaninReady(__gm__ FullPaState *state, uint32_t owner, uint32_t task_id) {
    __gm__ uint64_t *words = ExecutorTraceWords(state, task_id);
    StoreDev64(words + 3U, static_cast<uint64_t>(get_sys_cnt()));
    StoreDev64(
        words + 7U,
        static_cast<uint64_t>(owner) |
            (static_cast<uint64_t>(g0_swimlane::PackExecutorTraceState(
                 TaskKindAt(task_id),
                 g0_swimlane::kExecutorTicketRecorded | g0_swimlane::kExecutorClaimRecorded |
                 g0_swimlane::kExecutorFaninReadyRecorded
             ))
             << 32U)
    );
}

__aicore__ __attribute__((always_inline)) inline void
TraceExecutorBegin(__gm__ FullPaState *state, uint32_t owner, uint32_t task_id) {
    __gm__ uint64_t *words = ExecutorTraceWords(state, task_id);
    StoreDev64(words + 4U, static_cast<uint64_t>(get_sys_cnt()));
    StoreDev64(
        words + 7U,
        static_cast<uint64_t>(owner) |
            (static_cast<uint64_t>(g0_swimlane::PackExecutorTraceState(
                 TaskKindAt(task_id),
                 g0_swimlane::kExecutorTicketRecorded | g0_swimlane::kExecutorClaimRecorded |
                 g0_swimlane::kExecutorFaninReadyRecorded | g0_swimlane::kExecutorBeginRecorded
             ))
             << 32U)
    );
}

__aicore__ __attribute__((always_inline)) inline void
TraceExecutorWorkloadEnd(__gm__ FullPaState *state, uint32_t owner, uint32_t task_id) {
    __gm__ uint64_t *words = ExecutorTraceWords(state, task_id);
    StoreDev64(words + 6U, static_cast<uint64_t>(get_sys_cnt()));
    StoreDev64(
        words + 7U,
        static_cast<uint64_t>(owner) |
            (static_cast<uint64_t>(g0_swimlane::PackExecutorTraceState(
                 TaskKindAt(task_id),
                 g0_swimlane::kExecutorTicketRecorded | g0_swimlane::kExecutorClaimRecorded |
                 g0_swimlane::kExecutorFaninReadyRecorded | g0_swimlane::kExecutorBeginRecorded |
                 g0_swimlane::kExecutorWorkloadBeginRecorded | g0_swimlane::kExecutorWorkloadEndRecorded
             ))
             << 32U)
    );
}

__aicore__ __attribute__((always_inline)) inline void
TraceExecutorEnd(__gm__ FullPaState *state, uint32_t owner, uint32_t task_id) {
    __gm__ uint64_t *words = ExecutorTraceWords(state, task_id);
    StoreDev64(words + 5U, static_cast<uint64_t>(get_sys_cnt()));
    StoreDev64(
        words + 7U,
        static_cast<uint64_t>(owner) |
            (static_cast<uint64_t>(g0_swimlane::PackExecutorTraceState(
                 TaskKindAt(task_id), g0_swimlane::kExpectedExecutorTraceBits
             ))
             << 32U)
    );
    dsb(DSB_ALL);
    StoreDev64(words, state->control.launch_nonce);
}
#endif

#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
#define G0_SCALAR_TRACE_PARAMETER , ScalarTraceContext &trace
#define G0_SCALAR_TRACE_ARGUMENT , trace
#define G0_TRACE_SCALAR_LOAD(trace, task_id, site, address, result_used) \
    ScalarTraceAtomicLoad(trace, task_id, site, address, result_used)
#define G0_TRACE_SCALAR_CAS(trace, task_id, site, address, expected, desired, result_used) \
    ScalarTraceCas(trace, task_id, site, address, expected, desired, result_used)
#define G0_TRACE_SCALAR_FETCH_ADD(trace, task_id, site, address, increment, result_used) \
    ScalarTraceFetchAdd(trace, task_id, site, address, increment, result_used)
#define G0_TRACE_SCALAR_EXCHANGE(trace, task_id, site, address, value, result_used) \
    ScalarTraceExchange(trace, task_id, site, address, value, result_used)

__aicore__ __attribute__((always_inline)) inline uint64_t
TraceLoadFatal(__gm__ FullPaState *state, ScalarTraceContext &trace) {
    return ScalarTraceAtomicLoad(
        trace, g0_swimlane::kTraceNoTask, g0_swimlane::AtomicSite::FatalLoad, &state->fatal.state, true
    );
}

__aicore__ __attribute__((always_inline)) inline void TracePublishFatal(
    __gm__ FullPaState *state, ExecFatalReason reason, uint32_t owner, uint32_t task_id, ScalarTraceContext &trace
) {
    (void)ScalarTraceCas(
        trace, task_id, g0_swimlane::AtomicSite::FatalSet, &state->fatal.state, 0U,
        EncodeExecFatal(reason, owner, task_id), false
    );
}
#else
#define G0_SCALAR_TRACE_PARAMETER
#define G0_SCALAR_TRACE_ARGUMENT
#define G0_TRACE_SCALAR_LOAD(trace, task_id, site, address, result_used) ScalarAtomicLoad(address)
#define G0_TRACE_SCALAR_CAS(trace, task_id, site, address, expected, desired, result_used) \
    ScalarCas(address, expected, desired)
#define G0_TRACE_SCALAR_FETCH_ADD(trace, task_id, site, address, increment, result_used) \
    ScalarFetchAdd(address, increment)
#define G0_TRACE_SCALAR_EXCHANGE(trace, task_id, site, address, value, result_used) ScalarExchange(address, value)
#define TraceLoadFatal(state, trace) LoadFatal(state)
#define TracePublishFatal(state, reason, owner, task_id, trace) PublishFatal(state, reason, owner, task_id)
#endif


#if defined(__DAV_VEC__)

#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
struct alignas(32U) SimtTraceCounters {
    volatile uint64_t atomic_calls;
    volatile uint64_t poll_calls;
    volatile uint32_t record_count;
    volatile uint32_t dropped_records;
    volatile uint32_t poll_records;
    volatile uint32_t reserved;
};

static_assert(sizeof(SimtTraceCounters) == 32U, "SIMT trace counter ABI changed");
static_assert(alignof(SimtTraceCounters) == 32U, "SIMT trace counters must satisfy VEC-stack alignment");

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t SimtTraceClock() {
    return clock();
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtClockAfterAtomicResult(uint64_t value) {
    // asc_atomic_* 与 CLOCK64 都是有副作用的 SIMT builtin；先保留
    // 源码顺序，单独的返回寄存器依赖由 IR/ELF 检查后再定。
    (void)value;
    return SimtTraceClock();
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline __gm__ uint64_t *SimtTraceReserveRecord(
    __gm__ g0_swimlane::TraceRecord *records, SimtTraceCounters *counters,
    uint32_t task_id, uint16_t site, g0_swimlane::TraceKind kind, uint8_t op
) {
    // G0 的 task/writer 上界已由 g0_swimlane.h 静态证明小于容量。
    // 不在每次 raw 写入前引入 SIMT 分歧，避免把 DVG stack
    // 放大到 VF 无法启动。host 仍会检查 record_count 和 dropped=0。
    const uint32_t slot = counters->record_count++;
    __gm__ uint64_t *words = reinterpret_cast<__gm__ uint64_t *>(&records[slot]);
    __gm__ uint32_t *meta_words = reinterpret_cast<__gm__ uint32_t *>(words + 2U);
    const uint32_t encoded_kind_site = static_cast<uint32_t>(site) |
                                       (static_cast<uint32_t>(kind) << 16U) |
                                       (static_cast<uint32_t>(op) << 24U);
    // SIMT store 直接落 GM：使用官方 st.cg（L1 NON_CACHEABLE）。
    // metadata 在 atomic 之前先写完，使 task/site/kind/op 不跨越
    // atomic 与两个 CLOCK64，避免 CCEC 因活跃值过多生成未对齐的
    // VEC-UB spill。
    asc_stcg(meta_words + 0U, task_id);
    asc_stcg(meta_words + 1U, encoded_kind_site);
    return words;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtTraceWriteAttributes(
    __gm__ uint64_t *words, uint32_t flags, uint32_t call_count
) {
    __gm__ uint32_t *attribute_words = reinterpret_cast<__gm__ uint32_t *>(words + 3U);
    asc_stcg(attribute_words + 0U, flags);
    asc_stcg(attribute_words + 1U, call_count);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtTraceWriteTimeEndpoints(
    __gm__ uint64_t *words, uint64_t begin, uint64_t end
) {
    asc_stcg(words + 0U, begin);
    asc_stcg(words + 1U, end);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t SimtTraceAtomicAdd(
    __gm__ g0_swimlane::TraceRecord *records, SimtTraceCounters *counters, uint32_t task_id,
    g0_swimlane::AtomicSite site, __gm__ uint64_t *address, uint64_t value, bool result_used
) {
    const g0_swimlane::AtomicOp op =
        value == 0U ? g0_swimlane::AtomicOp::Load : g0_swimlane::AtomicOp::FetchAdd;
    __gm__ uint64_t *trace_words = SimtTraceReserveRecord(
        records, counters, task_id, static_cast<uint16_t>(site), g0_swimlane::TraceKind::Atomic,
        static_cast<uint8_t>(op)
    );
    // result_used 与 call_count 在发起 atomic 前已知，先写完后不再让
    // attributes 跨 atomic/CLOCK64。value_zero 仅是可选展示信息；
    // SIMT 路径不为它额外保留 atomic 返回值的活跃区间。
    SimtTraceWriteAttributes(trace_words, result_used ? g0_swimlane::kAtomicResultUsed : 0U, 1U);
    const uint64_t begin = SimtTraceClock();
    const uint64_t old = asc_atomic_add(address, value);
    const uint64_t end = result_used ? SimtClockAfterAtomicResult(old) : SimtTraceClock();
    // CCEC 当前不接受 SIMT inline-asm 返回寄存器依赖。这些
    // CLOCK64 边界只能如实标为 source_issue，不冒充 return_ready。
    ++counters->atomic_calls;
    SimtTraceWriteTimeEndpoints(trace_words, begin, end);
    return old;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t SimtTraceAtomicCas(
    __gm__ g0_swimlane::TraceRecord *records, SimtTraceCounters *counters, uint32_t task_id,
    g0_swimlane::AtomicSite site, __gm__ uint64_t *address, uint64_t expected, uint64_t desired, bool result_used
) {
    __gm__ uint64_t *trace_words = SimtTraceReserveRecord(
        records, counters, task_id, static_cast<uint16_t>(site), g0_swimlane::TraceKind::Atomic,
        static_cast<uint8_t>(g0_swimlane::AtomicOp::CompareExchange)
    );
    SimtTraceWriteAttributes(trace_words, result_used ? g0_swimlane::kAtomicResultUsed : 0U, 1U);
    const uint64_t begin = SimtTraceClock();
    const uint64_t old = asc_atomic_cas(address, expected, desired);
    const uint64_t end = result_used ? SimtClockAfterAtomicResult(old) : SimtTraceClock();
    ++counters->atomic_calls;
    SimtTraceWriteTimeEndpoints(trace_words, begin, end);
    return old;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtTracePollRecord(
    __gm__ g0_swimlane::TraceRecord *records, SimtTraceCounters *counters, uint32_t task_id,
    g0_swimlane::AtomicSite site, uint64_t begin, uint32_t call_count
) {
    // SIMT 边界只声明 source_issue；在 episode 退出处取一次 end，
    // 不在每轮 atomic load 后读取 CLOCK64。
    const uint64_t end = SimtTraceClock();
    __gm__ uint64_t *trace_words = SimtTraceReserveRecord(
        records, counters, task_id, static_cast<uint16_t>(site), g0_swimlane::TraceKind::Atomic,
        static_cast<uint8_t>(g0_swimlane::AtomicOp::Load)
    );
    counters->atomic_calls += call_count;
    counters->poll_calls += call_count;
    ++counters->poll_records;
    SimtTraceWriteAttributes(
        trace_words,
        g0_swimlane::kAtomicResultUsed | g0_swimlane::kAtomicPollBatch,
        call_count
    );
    SimtTraceWriteTimeEndpoints(trace_words, begin, end);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtTraceFinish(
    __gm__ g0_swimlane::TraceLogControl *control, SimtTraceCounters *counters,
    uint64_t nonce, uint32_t writer
) {
    __gm__ uint64_t *words = reinterpret_cast<__gm__ uint64_t *>(control);
    words[1] = counters->atomic_calls;
    words[2] = counters->poll_calls;
    words[3] = 0U;
    words[4] = 0U;
    words[5] = static_cast<uint64_t>(counters->record_count) |
               (static_cast<uint64_t>(counters->dropped_records) << 32U);
    words[6] = static_cast<uint64_t>(counters->poll_records);
    words[7] = static_cast<uint64_t>(writer) |
               (static_cast<uint64_t>(g0_swimlane::TraceDomain::Simt) << 32U);
    asc_threadfence();
    words[0] = nonce;
    asc_threadfence();
}

#define G0_SIMT_TRACE_PARAMETER \
    , __gm__ g0_swimlane::TraceRecord *simt_records, SimtTraceCounters *simt_counters
#define G0_SIMT_TRACE_ARGUMENT , simt_records, simt_counters
#define G0_TRACE_SIMT_ADD(simt_trace, task_id, site, address, value, result_used) \
    SimtTraceAtomicAdd(simt_records, simt_counters, task_id, site, address, value, result_used)
#define G0_TRACE_SIMT_CAS(simt_trace, task_id, site, address, expected, desired, result_used) \
    SimtTraceAtomicCas(simt_records, simt_counters, task_id, site, address, expected, desired, result_used)
#else
#define G0_SIMT_TRACE_PARAMETER
#define G0_SIMT_TRACE_ARGUMENT
#define G0_TRACE_SIMT_ADD(simt_trace, task_id, site, address, value, result_used) asc_atomic_add(address, value)
#define G0_TRACE_SIMT_CAS(simt_trace, task_id, site, address, expected, desired, result_used) \
    asc_atomic_cas(address, expected, desired)
#endif

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtFatalValue(ExecFatalReason reason, uint32_t owner, uint32_t task_id) {
    return (static_cast<uint64_t>(reason) << kFatalReasonShift) | (static_cast<uint64_t>(owner) << kFatalOwnerShift) |
           (static_cast<uint64_t>(task_id) << kFatalTaskIdShift);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtPublishFatal(
    __gm__ uint64_t *fatal, ExecFatalReason reason, uint32_t owner, uint32_t task_id G0_SIMT_TRACE_PARAMETER
) {
    (void)G0_TRACE_SIMT_CAS(
        simt_trace, task_id, g0_swimlane::AtomicSite::FatalSet, fatal, static_cast<uint64_t>(0U),
        SimtFatalValue(reason, owner, task_id), false
    );
}


__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t SimtBuilderReportChecksum(
    uint64_t nonce, uint32_t thread_id, uint32_t task_count, uint32_t wins, uint32_t first_task, uint32_t last_task,
    uint32_t attempts, uint32_t prepares, uint32_t commits, uint32_t insert_waits, uint32_t claim_losses
) {
    uint64_t value = nonce ^ (static_cast<uint64_t>(thread_id) << 32U) ^ task_count;
    value ^= static_cast<uint64_t>(wins) << 1U;
    value ^= static_cast<uint64_t>(first_task) << 7U;
    value ^= static_cast<uint64_t>(last_task) << 19U;
    value ^= static_cast<uint64_t>(attempts) << 37U;
    value ^= static_cast<uint64_t>(prepares) << 43U;
    value ^= static_cast<uint64_t>(commits) << 49U;
    value ^= static_cast<uint64_t>(insert_waits) << 55U;
    value ^= static_cast<uint64_t>(claim_losses) << 25U;
    value *= 0x9E3779B97F4A7C15ULL;
    value ^= value >> 29U;
    value *= 0xD6E8FEB86659FD93ULL;
    return value ^ (value >> 31U);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtPublishBuildReportWord(
    __gm__ uint64_t *address, uint64_t value, uint32_t task_id G0_SIMT_TRACE_PARAMETER
) {
    return G0_TRACE_SIMT_CAS(
               simt_trace, task_id, g0_swimlane::AtomicSite::SimtBuildReportPublish, address,
               kReportPoisonWord, value, true
           ) == kReportPoisonWord;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t SimtDescriptorWord(
    uint64_t address, uint64_t buffer_size, uint64_t owner_task, uint64_t start_offset, uint32_t ndims, uint32_t dtype,
    bool manual_dep, uint32_t shape0, uint32_t shape1, uint32_t word
) {
    if (word == 0U) {
        return address;
    }
    if (word == 1U) {
        return buffer_size;
    }
    if (word == 2U) {
        return owner_task;
    }
    if (word == 3U) {
        return start_offset;
    }
    if (word == 4U) {
        return static_cast<uint64_t>(ndims) << 32U;
    }
    if (word == 5U) {
        return static_cast<uint64_t>(dtype) | (static_cast<uint64_t>(manual_dep ? 1U : 0U) << 8U) |
               (static_cast<uint64_t>(1U) << 16U) | (static_cast<uint64_t>(shape0) << 32U);
    }
    if (word == 6U) {
        return shape1;
    }
    if (word == 8U) {
        return ndims == 1U ? shape0 : static_cast<uint64_t>(shape0) * shape1;
    }
    if (word == 9U) {
        const uint32_t stride0 = ndims == 1U ? 1U : shape1;
        const uint32_t stride1 = ndims == 1U ? 0U : 1U;
        return static_cast<uint64_t>(stride0) | (static_cast<uint64_t>(stride1) << 32U);
    }
    return 0U;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtOutputDescriptorWord(uint32_t task_id, uint32_t output_slot, uint64_t task_base, uint32_t word) {
    const uint32_t kind = task_id % kTasksPerBatch;
    uint64_t output_offset = 0U;
    uint32_t ndims = 0U;
    uint32_t dtype = static_cast<uint32_t>(DataType::Float32);
    uint32_t shape0 = 0U;
    uint32_t shape1 = 0U;
    if (kind == static_cast<uint32_t>(TaskKind::Alloc)) {
        ndims = output_slot == 0U ? 2U : 1U;
        shape0 = kPaHeads;
        shape1 = output_slot == 0U ? kPaHeadDim : 0U;
        output_offset = output_slot == 0U ? 0U : (output_slot == 1U ? 8192U : 9216U);
    } else if (kind == static_cast<uint32_t>(TaskKind::Qk)) {
        ndims = 2U;
        shape0 = kPaHeads;
        shape1 = kPaBlocksPerRequest * kPaBlockSize;
    } else if (kind == static_cast<uint32_t>(TaskKind::Sf)) {
        ndims = output_slot == 0U ? 2U : 1U;
        dtype =
            output_slot == 0U ? static_cast<uint32_t>(DataType::Bfloat16) : static_cast<uint32_t>(DataType::Float32);
        shape0 = kPaHeads;
        shape1 = output_slot == 0U ? kPaBlocksPerRequest * kPaBlockSize : 0U;
        output_offset = output_slot == 0U ? 0U : (output_slot == 1U ? 262144U : 263168U);
    } else if (kind == static_cast<uint32_t>(TaskKind::Pv)) {
        ndims = 2U;
        shape0 = kPaHeads;
        shape1 = kPaHeadDim;
    }
    const uint64_t elements = ndims == 1U ? shape0 : static_cast<uint64_t>(shape0) * shape1;
    const uint64_t element_bytes = dtype == static_cast<uint32_t>(DataType::Bfloat16) ? 2U : 4U;
    return SimtDescriptorWord(
        kSyntheticHeapBase + task_base + output_offset, elements * element_bytes, task_id, 0U, ndims, dtype, false,
        shape0, shape1, word
    );
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtExternalDescriptorWord(uint32_t batch_count, uint32_t task_id, uint32_t tensor_index, uint32_t word) {
    const uint32_t kind = task_id % kTasksPerBatch;
    const uint32_t batch = task_id / kTasksPerBatch;
    uint64_t address = 0U;
    uint64_t buffer_size = 0U;
    uint64_t start_offset = 0U;
    uint32_t dtype = static_cast<uint32_t>(DataType::Float32);
    bool manual_dep = false;
    uint32_t shape0 = 0U;
    uint32_t shape1 = 0U;
    if (kind == static_cast<uint32_t>(TaskKind::Qk) && tensor_index == 0U) {
        address = kSyntheticQueryBase;
        buffer_size = static_cast<uint64_t>(batch_count) * kPaHeads * kPaHeadDim * 2U;
        start_offset = static_cast<uint64_t>(batch) * kPaHeads * kPaHeadDim;
        dtype = static_cast<uint32_t>(DataType::Bfloat16);
        shape0 = kPaHeads;
        shape1 = kPaHeadDim;
    } else if ((kind == static_cast<uint32_t>(TaskKind::Qk) && tensor_index == 1U) ||
               (kind == static_cast<uint32_t>(TaskKind::Pv) && tensor_index == 1U)) {
        address = kind == static_cast<uint32_t>(TaskKind::Qk) ? kSyntheticKeyBase : kSyntheticValueBase;
        shape0 = batch_count * kPaBlocksPerRequest * kPaBlockSize;
        shape1 = kPaHeadDim;
        dtype = static_cast<uint32_t>(DataType::Bfloat16);
        buffer_size = static_cast<uint64_t>(shape0) * shape1 * 2U;
    } else if ((kind == static_cast<uint32_t>(TaskKind::Qk) || kind == static_cast<uint32_t>(TaskKind::Pv)) &&
               tensor_index == 2U) {
        address = kSyntheticBlockTableBase;
        shape0 = batch_count;
        shape1 = kPaMaxBlocksPerRequest;
        dtype = static_cast<uint32_t>(DataType::Int32);
        buffer_size = static_cast<uint64_t>(shape0) * shape1 * 4U;
    } else {
        address = kSyntheticOutputBase;
        buffer_size = static_cast<uint64_t>(batch_count) * kPaHeads * kPaHeadDim * 4U;
        start_offset = static_cast<uint64_t>(batch) * kPaHeads * kPaHeadDim;
        dtype = static_cast<uint32_t>(DataType::Float32);
        manual_dep = true;
        shape0 = kPaHeads;
        shape1 = kPaHeadDim;
    }
    return SimtDescriptorWord(
        address, buffer_size, kInvalidTaskId, start_offset, 2U, dtype, manual_dep, shape0, shape1, word
    );
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtStoreDescriptor(
    __gm__ uint64_t *destination, uint64_t address, uint64_t buffer_size, uint64_t owner_task,
    uint64_t start_offset, uint32_t ndims, uint32_t dtype, bool manual_dep, uint32_t shape0,
    uint32_t shape1
) {
    destination[0] = address;
    destination[1] = buffer_size;
    destination[2] = owner_task;
    destination[3] = start_offset;
    destination[4] = static_cast<uint64_t>(ndims) << 32U;
    destination[5] = static_cast<uint64_t>(dtype) |
                     (static_cast<uint64_t>(manual_dep ? 1U : 0U) << 8U) |
                     (static_cast<uint64_t>(1U) << 16U) |
                     (static_cast<uint64_t>(shape0) << 32U);
    destination[6] = shape1;
    destination[7] = 0U;
    destination[8] = ndims == 1U ? shape0 : static_cast<uint64_t>(shape0) * shape1;
    const uint32_t stride0 = ndims == 1U ? 1U : shape1;
    const uint32_t stride1 = ndims == 1U ? 0U : 1U;
    destination[9] = static_cast<uint64_t>(stride0) | (static_cast<uint64_t>(stride1) << 32U);
    destination[10] = 0U;
    destination[11] = 0U;
    destination[12] = 0U;
    destination[13] = 0U;
    destination[14] = 0U;
    destination[15] = 0U;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtStoreOutputDescriptor(
    __gm__ uint64_t *destination, uint32_t task_id, uint32_t output_slot, uint64_t task_base
) {
    const uint32_t kind = task_id % kTasksPerBatch;
    uint64_t output_offset = 0U;
    uint32_t ndims = 0U;
    uint32_t dtype = static_cast<uint32_t>(DataType::Float32);
    uint32_t shape0 = 0U;
    uint32_t shape1 = 0U;
    if (kind == static_cast<uint32_t>(TaskKind::Alloc)) {
        ndims = output_slot == 0U ? 2U : 1U;
        shape0 = kPaHeads;
        shape1 = output_slot == 0U ? kPaHeadDim : 0U;
        output_offset = output_slot == 0U ? 0U : (output_slot == 1U ? 8192U : 9216U);
    } else if (kind == static_cast<uint32_t>(TaskKind::Qk)) {
        ndims = 2U;
        shape0 = kPaHeads;
        shape1 = kPaBlocksPerRequest * kPaBlockSize;
    } else if (kind == static_cast<uint32_t>(TaskKind::Sf)) {
        ndims = output_slot == 0U ? 2U : 1U;
        dtype = output_slot == 0U ? static_cast<uint32_t>(DataType::Bfloat16) :
                                   static_cast<uint32_t>(DataType::Float32);
        shape0 = kPaHeads;
        shape1 = output_slot == 0U ? kPaBlocksPerRequest * kPaBlockSize : 0U;
        output_offset = output_slot == 0U ? 0U : (output_slot == 1U ? 262144U : 263168U);
    } else if (kind == static_cast<uint32_t>(TaskKind::Pv)) {
        ndims = 2U;
        shape0 = kPaHeads;
        shape1 = kPaHeadDim;
    }
    const uint64_t elements = ndims == 1U ? shape0 : static_cast<uint64_t>(shape0) * shape1;
    const uint64_t element_bytes = dtype == static_cast<uint32_t>(DataType::Bfloat16) ? 2U : 4U;
    SimtStoreDescriptor(
        destination, kSyntheticHeapBase + task_base + output_offset, elements * element_bytes,
        task_id, 0U, ndims, dtype, false, shape0, shape1
    );
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SimtStoreExternalDescriptor(
    __gm__ uint64_t *destination, uint32_t batch_count, uint32_t task_id, uint32_t tensor_index
) {
    const uint32_t kind = task_id % kTasksPerBatch;
    const uint32_t batch = task_id / kTasksPerBatch;
    uint64_t address = 0U;
    uint64_t buffer_size = 0U;
    uint64_t start_offset = 0U;
    uint32_t dtype = static_cast<uint32_t>(DataType::Float32);
    bool manual_dep = false;
    uint32_t shape0 = 0U;
    uint32_t shape1 = 0U;
    if (kind == static_cast<uint32_t>(TaskKind::Qk) && tensor_index == 0U) {
        address = kSyntheticQueryBase;
        buffer_size = static_cast<uint64_t>(batch_count) * kPaHeads * kPaHeadDim * 2U;
        start_offset = static_cast<uint64_t>(batch) * kPaHeads * kPaHeadDim;
        dtype = static_cast<uint32_t>(DataType::Bfloat16);
        shape0 = kPaHeads;
        shape1 = kPaHeadDim;
    } else if ((kind == static_cast<uint32_t>(TaskKind::Qk) && tensor_index == 1U) ||
               (kind == static_cast<uint32_t>(TaskKind::Pv) && tensor_index == 1U)) {
        address = kind == static_cast<uint32_t>(TaskKind::Qk) ? kSyntheticKeyBase : kSyntheticValueBase;
        shape0 = batch_count * kPaBlocksPerRequest * kPaBlockSize;
        shape1 = kPaHeadDim;
        dtype = static_cast<uint32_t>(DataType::Bfloat16);
        buffer_size = static_cast<uint64_t>(shape0) * shape1 * 2U;
    } else if ((kind == static_cast<uint32_t>(TaskKind::Qk) || kind == static_cast<uint32_t>(TaskKind::Pv)) &&
               tensor_index == 2U) {
        address = kSyntheticBlockTableBase;
        shape0 = batch_count;
        shape1 = kPaMaxBlocksPerRequest;
        dtype = static_cast<uint32_t>(DataType::Int32);
        buffer_size = static_cast<uint64_t>(shape0) * shape1 * 4U;
    } else {
        address = kSyntheticOutputBase;
        buffer_size = static_cast<uint64_t>(batch_count) * kPaHeads * kPaHeadDim * 4U;
        start_offset = static_cast<uint64_t>(batch) * kPaHeads * kPaHeadDim;
        manual_dep = true;
        shape0 = kPaHeads;
        shape1 = kPaHeadDim;
    }
    SimtStoreDescriptor(
        destination, address, buffer_size, kInvalidTaskId, start_offset, 2U, dtype, manual_dep,
        shape0, shape1
    );
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtWaitBuilderStart(
    __gm__ uint64_t *builder_started, __gm__ uint64_t *fatal, uint64_t timeout_ticks, uint32_t thread,
    uint32_t build_owner, uint32_t builder_count, bool active G0_SIMT_TRACE_PARAMETER
) {
    if (thread == 0U) {
        const uint64_t observed = G0_TRACE_SIMT_ADD(
            simt_trace, g0_swimlane::kTraceNoTask,
            g0_swimlane::AtomicSite::SimtBuilderStartedIncrement, builder_started,
            static_cast<uint64_t>(1U), true
        );
        if (observed >= builder_count) {
            SimtPublishFatal(
                fatal, ExecFatalReason::InvalidBuildInput, build_owner, UINT32_MAX G0_SIMT_TRACE_ARGUMENT
            );
            return false;
        }
        asc_threadfence();
    }
    if (!active) {
        return true;
    }
    const uint64_t begin = clock();
    uint32_t polls = 0U;
    while (clock() - begin <= timeout_ticks) {
        const uint64_t observed = asc_atomic_add(builder_started, static_cast<uint64_t>(0U));
        ++polls;
        if (observed == builder_count) {
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
            SimtTracePollRecord(
                simt_records, simt_counters, g0_swimlane::kTraceNoTask,
                g0_swimlane::AtomicSite::SimtBuilderStartedPoll, begin, polls
            );
#endif
            return true;
        }
        if (observed > builder_count) {
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
            SimtTracePollRecord(
                simt_records, simt_counters, g0_swimlane::kTraceNoTask,
                g0_swimlane::AtomicSite::SimtBuilderStartedPoll, begin, polls
            );
#endif
            SimtPublishFatal(
                fatal, ExecFatalReason::InvalidBuildInput, build_owner, UINT32_MAX G0_SIMT_TRACE_ARGUMENT
            );
            return false;
        }
        if ((polls & kWatchdogMask) == 0U &&
            G0_TRACE_SIMT_ADD(
                simt_trace, g0_swimlane::kTraceNoTask, g0_swimlane::AtomicSite::FatalLoad, fatal,
                static_cast<uint64_t>(0U), true
            ) != 0U) {
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
            SimtTracePollRecord(
                simt_records, simt_counters, g0_swimlane::kTraceNoTask,
                g0_swimlane::AtomicSite::SimtBuilderStartedPoll, begin, polls
            );
#endif
            return false;
        }
    }
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
    SimtTracePollRecord(
        simt_records, simt_counters, g0_swimlane::kTraceNoTask,
        g0_swimlane::AtomicSite::SimtBuilderStartedPoll,
        begin, polls
    );
#endif
    SimtPublishFatal(fatal, ExecFatalReason::Timeout, build_owner, UINT32_MAX G0_SIMT_TRACE_ARGUMENT);
    return false;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtWaitAtomicValue(
    __gm__ uint64_t *address, uint64_t expected, __gm__ uint64_t *fatal, uint64_t timeout_ticks, uint32_t task_id,
    uint32_t build_owner, uint32_t *poll_count G0_SIMT_TRACE_PARAMETER
) {
    const uint64_t begin = clock();
    uint32_t polls = 0U;
    while (clock() - begin <= timeout_ticks) {
        const uint64_t observed = asc_atomic_add(address, static_cast<uint64_t>(0U));
        ++polls;
        if (observed == expected) {
            *poll_count += polls;
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
            SimtTracePollRecord(
                simt_records, simt_counters, task_id,
                g0_swimlane::AtomicSite::SimtInsertPredecessorPoll, begin, polls
            );
#endif
            return true;
        }
        // insert_completion starts at predecessor_id-1 and is incremented
        // exactly once to predecessor_id.  Any third value is corruption, not
        // a slow predecessor, so report it immediately instead of timing out.
        if (observed != expected - 1U) {
            *poll_count += polls;
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
            SimtTracePollRecord(
                simt_records, simt_counters, task_id,
                g0_swimlane::AtomicSite::SimtInsertPredecessorPoll, begin, polls
            );
#endif
            SimtPublishFatal(
                fatal, ExecFatalReason::InsertProtocolFailed, build_owner, task_id G0_SIMT_TRACE_ARGUMENT
            );
            return false;
        }
        if ((polls & kWatchdogMask) == 0U &&
            G0_TRACE_SIMT_ADD(
                simt_trace, task_id, g0_swimlane::AtomicSite::FatalLoad, fatal,
                static_cast<uint64_t>(0U), true
            ) != 0U) {
            *poll_count += polls;
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
            SimtTracePollRecord(
                simt_records, simt_counters, task_id,
                g0_swimlane::AtomicSite::SimtInsertPredecessorPoll, begin, polls
            );
#endif
            return false;
        }
    }
    *poll_count += polls;
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
    SimtTracePollRecord(
        simt_records, simt_counters, task_id,
        g0_swimlane::AtomicSite::SimtInsertPredecessorPoll, begin, polls
    );
#endif
    return false;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtWaitOutputPublished(
    __gm__ uint64_t *address, uint64_t expected, __gm__ uint64_t *fatal, uint64_t timeout_ticks,
    uint32_t task_id, uint32_t build_owner G0_SIMT_TRACE_PARAMETER
) {
    const uint64_t begin = clock();
    uint32_t polls = 0U;
    while (clock() - begin <= timeout_ticks) {
        const uint64_t observed = asc_atomic_add(address, static_cast<uint64_t>(0U));
        ++polls;
        if (observed == expected) {
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
            SimtTracePollRecord(
                simt_records, simt_counters, task_id,
                g0_swimlane::AtomicSite::SimtMetadataOutputPublishedPoll, begin, polls
            );
#endif
            return true;
        }
        if (observed != UINT64_MAX) {
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
            SimtTracePollRecord(
                simt_records, simt_counters, task_id,
                g0_swimlane::AtomicSite::SimtMetadataOutputPublishedPoll, begin, polls
            );
#endif
            SimtPublishFatal(
                fatal, ExecFatalReason::InsertProtocolFailed, build_owner, task_id G0_SIMT_TRACE_ARGUMENT
            );
            return false;
        }
        if ((polls & kWatchdogMask) == 0U &&
            G0_TRACE_SIMT_ADD(
                simt_trace, task_id, g0_swimlane::AtomicSite::FatalLoad, fatal,
                static_cast<uint64_t>(0U), true
            ) != 0U) {
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
            SimtTracePollRecord(
                simt_records, simt_counters, task_id,
                g0_swimlane::AtomicSite::SimtMetadataOutputPublishedPoll, begin, polls
            );
#endif
            return false;
        }
    }
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
    SimtTracePollRecord(
        simt_records, simt_counters, task_id,
        g0_swimlane::AtomicSite::SimtMetadataOutputPublishedPoll, begin, polls
    );
#endif
    return false;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtLoadTaskBase(
    __gm__ uint64_t *task_words, uint32_t producer_task, __gm__ uint64_t *fatal, uint64_t timeout_ticks,
    uint32_t consumer_task, uint32_t build_owner, uint64_t *task_base,
    uint32_t *access_count G0_SIMT_TRACE_PARAMETER
) {
    constexpr uint32_t kTaskStrideWords = sizeof(FullPaTask) / sizeof(uint64_t);
    constexpr uint32_t kBaseReportOffsetWords =
        (offsetof(FullPaTask, allocation) + offsetof(FullPaTaskAllocationReport, task_base_plus_one)) /
        sizeof(uint64_t);
    __gm__ uint64_t *report = task_words + producer_task * kTaskStrideWords + kBaseReportOffsetWords;
    const uint64_t begin = clock();
    uint32_t local_access_count = 0U;
    while (clock() - begin <= timeout_ticks) {
        const uint64_t observed = asc_atomic_add(report, static_cast<uint64_t>(0U));
        ++*access_count;
        ++local_access_count;
        if (observed != 0U) {
            *task_base = observed - 1U;
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
            SimtTracePollRecord(
                simt_records, simt_counters, consumer_task,
                g0_swimlane::AtomicSite::SimtProducerTaskBasePoll,
                begin, local_access_count
            );
#endif
            return true;
        }
        if (((*access_count) & kWatchdogMask) == 0U &&
            G0_TRACE_SIMT_ADD(
                simt_trace, consumer_task, g0_swimlane::AtomicSite::FatalLoad, fatal,
                static_cast<uint64_t>(0U), true
            ) != 0U) {
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
            SimtTracePollRecord(
                simt_records, simt_counters, consumer_task,
                g0_swimlane::AtomicSite::SimtProducerTaskBasePoll,
                begin, local_access_count
            );
#endif
            return false;
        }
    }
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
    SimtTracePollRecord(
        simt_records, simt_counters, consumer_task,
        g0_swimlane::AtomicSite::SimtProducerTaskBasePoll,
        begin, local_access_count
    );
#endif
    return false;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint32_t
SimtTensorSourceTask(uint32_t task_id, uint32_t tensor_index) {
    const uint32_t kind = task_id % kTasksPerBatch;
    if (kind == static_cast<uint32_t>(TaskKind::Qk)) {
        return tensor_index == 3U ? task_id : UINT32_MAX;
    }
    if (kind == static_cast<uint32_t>(TaskKind::Sf)) {
        return tensor_index == 0U ? task_id - 1U : task_id;
    }
    if (kind == static_cast<uint32_t>(TaskKind::Pv)) {
        return tensor_index == 0U ? task_id - 1U : (tensor_index == 3U ? task_id : UINT32_MAX);
    }
    if (kind != static_cast<uint32_t>(TaskKind::Up)) {
        return UINT32_MAX;
    }
    if (tensor_index <= 1U) {
        return task_id - 2U;
    }
    if (tensor_index == 2U) {
        return task_id - 1U;
    }
    return tensor_index <= 5U ? (task_id / kTasksPerBatch) * kTasksPerBatch : UINT32_MAX;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint32_t
SimtTensorOutputSlot(uint32_t task_id, uint32_t tensor_index) {
    const uint32_t kind = task_id % kTasksPerBatch;
    if (kind == static_cast<uint32_t>(TaskKind::Qk)) {
        return 0U;
    }
    if (kind == static_cast<uint32_t>(TaskKind::Sf)) {
        return tensor_index == 0U ? 0U : tensor_index - 1U;
    }
    if (kind == static_cast<uint32_t>(TaskKind::Pv)) {
        return 0U;
    }
    if (tensor_index <= 1U) {
        return tensor_index + 1U;
    }
    if (tensor_index == 2U) {
        return 0U;
    }
    return 5U - tensor_index;
}

// 以下两个 helper 是 full-PA 测试 workload 的 schema adapter，只负责给出
// tensor 数和 access tag。调度优化从 SimtTensorIsSharedWriterIntent 开始，
// 只消费 access tag 和 SharedOutputRef，不按算子/TaskKind 设置快捷路径。
// CCEC 要求 __simt_vf__ 的调用链全部显式标注 __simt_callee__。
__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint32_t
SimtTaskTensorCount(uint32_t task_id) {
    const uint32_t kind = task_id % kTasksPerBatch;
    return kind == static_cast<uint32_t>(TaskKind::Up) ? 7U :
           (kind == static_cast<uint32_t>(TaskKind::Alloc) ? 0U : 4U);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint32_t
SimtTaskOutputCount(uint32_t task_id) {
    const uint32_t kind = task_id % kTasksPerBatch;
    return kind == static_cast<uint32_t>(TaskKind::Alloc) || kind == static_cast<uint32_t>(TaskKind::Sf) ?
               3U :
               (kind == static_cast<uint32_t>(TaskKind::Qk) || kind == static_cast<uint32_t>(TaskKind::Pv) ? 1U :
                                                                                                            0U);
}

// PA schema adapter for allocator demand.  The bulk-reservation optimization
// below consumes only this generic per-task byte count; it does not branch on
// the scheduling identity of any operator stage.
__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtTaskReserveBytes(uint32_t task_id) {
    const uint32_t kind = task_id % kTasksPerBatch;
    return kind == static_cast<uint32_t>(TaskKind::Alloc) ?
               10240U :
               (kind == static_cast<uint32_t>(TaskKind::Qk) ?
                    524288U :
                    (kind == static_cast<uint32_t>(TaskKind::Sf) ?
                         264192U :
                         (kind == static_cast<uint32_t>(TaskKind::Pv) ? 8192U : 0U)));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline TensorAccess
SimtTaskTensorAccessAt(uint32_t task_id, uint32_t tensor_index) {
    const uint32_t kind = task_id % kTasksPerBatch;
    if (kind == static_cast<uint32_t>(TaskKind::Qk)) {
        return tensor_index == 3U ? TensorAccess::Output : TensorAccess::Input;
    }
    if (kind == static_cast<uint32_t>(TaskKind::Sf)) {
        return tensor_index == 0U ? TensorAccess::Input : TensorAccess::Output;
    }
    if (kind == static_cast<uint32_t>(TaskKind::Pv)) {
        return tensor_index == 3U ? TensorAccess::Output : TensorAccess::Input;
    }
    if (kind == static_cast<uint32_t>(TaskKind::Up)) {
        return tensor_index >= 3U ? TensorAccess::Inout : TensorAccess::Input;
    }
    return TensorAccess::NoDependency;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool
SimtTensorIsSharedWriterIntent(uint32_t task_id, uint32_t tensor_index) {
    const TensorAccess access = SimtTaskTensorAccessAt(task_id, tensor_index);
    return (access == TensorAccess::Inout || access == TensorAccess::OutputExisting) &&
           SimtTensorSourceTask(task_id, tensor_index) != UINT32_MAX;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint32_t
SimtMetadataWriterIntentCount(uint32_t task_id) {
    const uint32_t tensor_count = SimtTaskTensorCount(task_id);
    uint32_t count = 0U;
    for (uint32_t tensor = 0U; tensor < tensor_count; ++tensor) {
        count += SimtTensorIsSharedWriterIntent(task_id, tensor) ? 1U : 0U;
    }
    return count;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtMetadataWriterIntentAt(
    uint32_t task_id, uint32_t writer_index, uint32_t *producer_task, uint32_t *output_slot
) {
    const uint32_t tensor_count = SimtTaskTensorCount(task_id);
    uint32_t current = 0U;
    for (uint32_t tensor = 0U; tensor < tensor_count; ++tensor) {
        if (!SimtTensorIsSharedWriterIntent(task_id, tensor)) {
            continue;
        }
        if (current++ == writer_index) {
            *producer_task = SimtTensorSourceTask(task_id, tensor);
            *output_slot = SimtTensorOutputSlot(task_id, tensor);
            return true;
        }
    }
    return false;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint32_t
SimtPreviousMetadataWriterForSymbol(uint32_t task_id, uint32_t producer_task, uint32_t output_slot) {
    if (producer_task >= task_id || output_slot >= kOutputsPerTask) {
        return UINT32_MAX;
    }

    uint32_t candidate = task_id;
    while (candidate > producer_task + 1U) {
        --candidate;
        const uint32_t writer_count = SimtMetadataWriterIntentCount(candidate);
        for (uint32_t writer = 0U; writer < writer_count; ++writer) {
            uint32_t candidate_producer = 0U;
            uint32_t candidate_slot = 0U;
            if (SimtMetadataWriterIntentAt(candidate, writer, &candidate_producer, &candidate_slot) &&
                candidate_producer == producer_task && candidate_slot == output_slot) {
                return candidate;
            }
        }
    }
    return producer_task;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline int32_t
SimtPreviousMetadataWriterTask(uint32_t task_id) {
    if (task_id > kMaxTasks) {
        return -1;
    }
    uint32_t candidate = task_id;
    while (candidate != 0U) {
        --candidate;
        if (SimtMetadataWriterIntentCount(candidate) != 0U) {
            return static_cast<int32_t>(candidate);
        }
    }
    return -1;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtMetadataInsertContractForTask(uint32_t task_id) {
    const uint32_t writer_count = SimtMetadataWriterIntentCount(task_id);
    if (writer_count == 0U || writer_count > kWriterHistoryMaxPerTask) {
        return 0U;
    }
    const int32_t previous = SimtPreviousMetadataWriterTask(task_id);
    return kMetadataInsertContractPresent | writer_count |
           (static_cast<uint64_t>(previous + 1) << kMetadataInsertPreviousWriterShift);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool
SimtMetadataInsertContractPresent(uint64_t contract) {
    return (contract & kMetadataInsertContractPresent) != 0U;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint32_t
SimtMetadataInsertWriterCount(uint64_t contract) {
    return static_cast<uint32_t>(contract & kMetadataInsertWriterCountMask);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline int32_t
SimtMetadataInsertPreviousWriter(uint64_t contract) {
    return static_cast<int32_t>(
               (contract & kMetadataInsertPreviousWriterMask) >> kMetadataInsertPreviousWriterShift
           ) - 1;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint32_t
SimtSharedSymbolKey(uint32_t producer_task, uint32_t output_slot) {
    return producer_task * kOutputsPerTask + output_slot + 1U;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool
SimtDecodeSharedSymbolKey(uint32_t symbol_key, uint32_t *producer_task, uint32_t *output_slot) {
    if (symbol_key == 0U) {
        return false;
    }
    --symbol_key;
    *producer_task = symbol_key / kOutputsPerTask;
    *output_slot = symbol_key % kOutputsPerTask;
    return *producer_task < kMaxTasks && *output_slot < kOutputsPerTask;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtPrepareTask(
    __gm__ uint64_t *task_words, __gm__ uint64_t *heap_words, __gm__ uint64_t *fatal, uint64_t nonce,
    uint64_t timeout_ticks, uint32_t batch_count, uint32_t task_count, uint32_t task_id, uint32_t builder_thread,
    uint32_t build_owner, uint64_t aggregate_cursor, uint64_t *completion_vend, uint32_t *payload_words_written,
    uint32_t *state_access_count G0_SIMT_TRACE_PARAMETER
) {
    constexpr uint32_t kTaskStrideWords = sizeof(FullPaTask) / sizeof(uint64_t);
    constexpr uint32_t kPlanOffsetWords = offsetof(FullPaTask, plan) / sizeof(uint64_t);
    constexpr uint32_t kAllocationOffsetWords = offsetof(FullPaTask, allocation) / sizeof(uint64_t);
    constexpr uint32_t kBaseReportOffsetWords =
        kAllocationOffsetWords + offsetof(FullPaTaskAllocationReport, task_base_plus_one) / sizeof(uint64_t);
    constexpr uint32_t kVendReportOffsetWords =
        kAllocationOffsetWords + offsetof(FullPaTaskAllocationReport, completion_vend_plus_one) / sizeof(uint64_t);
    constexpr uint32_t kOutputsOffsetWords = offsetof(FullPaTask, outputs) / sizeof(uint64_t);
    constexpr uint32_t kOutputTensorOffsetWords =
        kOutputsOffsetWords + offsetof(SharedOutputCell, tensors) / sizeof(uint64_t);
    constexpr uint32_t kPublishedOffsetWords =
        kOutputsOffsetWords + offsetof(SharedOutputCell, published) / sizeof(uint64_t);
    constexpr uint32_t kLastWriterOffsetWords =
        kOutputsOffsetWords + offsetof(SharedOutputCell, last_writer) / sizeof(uint64_t);
    constexpr uint32_t kHistoryOffsetWords = offsetof(FullPaTask, writer_history) / sizeof(uint64_t);
    constexpr uint32_t kExecOffsetWords = offsetof(FullPaTask, exec) / sizeof(uint64_t);
    constexpr uint32_t kExecPayloadOffsetWords =
        kExecOffsetWords + offsetof(SharedExecCell, payload) / sizeof(uint64_t);
    constexpr uint32_t kHeapAtomicStrideWords = sizeof(AtomicLine) / sizeof(uint64_t);
    constexpr uint32_t kAtomicStrideWords = sizeof(AtomicLine) / sizeof(uint64_t);

    __gm__ uint64_t *task = task_words + task_id * kTaskStrideWords;
    const uint32_t kind = task_id % kTasksPerBatch;
    const bool executable = kind != static_cast<uint32_t>(TaskKind::Alloc);
    const uint32_t output_count = SimtTaskOutputCount(task_id);
    const uint64_t reserve = SimtTaskReserveBytes(task_id);
    uint32_t tensor_count = 0U;
    uint32_t scalar_count = 0U;
    uint32_t fanin_count = 0U;
    uint32_t engine = static_cast<uint32_t>(ExecEngineClass::None);
    uint32_t payload_bytes = 0U;
    uint32_t payload_lines = 0U;
    if (kind == static_cast<uint32_t>(TaskKind::Qk)) {
        tensor_count = 4U;
        scalar_count = 2U;
        engine = static_cast<uint32_t>(ExecEngineClass::Aic);
        payload_bytes = 592U;
        payload_lines = 10U;
    } else if (kind == static_cast<uint32_t>(TaskKind::Sf)) {
        tensor_count = 4U;
        scalar_count = 3U;
        fanin_count = 1U;
        engine = static_cast<uint32_t>(ExecEngineClass::Aiv);
        payload_bytes = 604U;
        payload_lines = 10U;
    } else if (kind == static_cast<uint32_t>(TaskKind::Pv)) {
        tensor_count = 4U;
        scalar_count = 2U;
        fanin_count = 1U;
        engine = static_cast<uint32_t>(ExecEngineClass::Aic);
        payload_bytes = 596U;
        payload_lines = 10U;
    } else if (kind == static_cast<uint32_t>(TaskKind::Up)) {
        tensor_count = 7U;
        scalar_count = 2U;
        fanin_count = 3U;
        engine = static_cast<uint32_t>(ExecEngineClass::Aiv);
        payload_bytes = 988U;
        payload_lines = 16U;
    }

    const uint32_t shard = task_id & (kSharedHeapShards - 1U);
    uint64_t task_base = 0U;
    uint64_t vend = aggregate_cursor;
    if (reserve != 0U) {
        const uint64_t cursor = G0_TRACE_SIMT_ADD(
            simt_trace, task_id, g0_swimlane::AtomicSite::SimtHeapShardReserve,
            heap_words + shard * kHeapAtomicStrideWords, reserve, true
        );
        if (cursor > kHeapShardSpan - reserve || aggregate_cursor > kHeapBytes - reserve) {
            SimtPublishFatal(
                fatal, ExecFatalReason::HeapReservationFailed, build_owner, task_id G0_SIMT_TRACE_ARGUMENT
            );
            return false;
        }
        task_base = static_cast<uint64_t>(shard) * kHeapShardSpan + cursor;
        vend = aggregate_cursor + reserve;
    }
    // 静态分片已经证明每个 task 只有一个 builder。allocation report 是
    // 单写、多读的 immutable 发布，不需要再用 CAS 重复争夺所有权。
    asc_stcg(task + kBaseReportOffsetWords, task_base + 1U);
    asc_stcg(task + kVendReportOffsetWords, vend + 1U);
    *completion_vend = vend;

    __gm__ uint64_t *plan = task + kPlanOffsetWords;
    plan[0] = static_cast<uint64_t>(task_id) | (static_cast<uint64_t>(task_id / kTasksPerBatch) << 32U);
    plan[1] = static_cast<uint64_t>(kind) | (static_cast<uint64_t>(engine) << 32U);
    plan[2] = static_cast<uint64_t>(output_count) | (static_cast<uint64_t>(payload_lines) << 32U);
    plan[3] = static_cast<uint64_t>(builder_thread) | (static_cast<uint64_t>(builder_thread / kWarpSize) << 32U);
    const uint32_t encoded_meta =
        kDispatchMetaPresent | (task_id + 1U == task_count ? kDispatchMetaLastSubmit : 0U) | kind;
    const uint32_t exec_route =
        kExecRoutePresent | (executable ? kExecRouteExecutable : 0U) | (engine << kExecRouteEngineShift);
    plan[4] = static_cast<uint64_t>(encoded_meta) | (static_cast<uint64_t>(exec_route) << 8U) |
              (static_cast<uint64_t>(build_owner) << 16U);
    plan[5] = reserve;
    plan[6] = nonce;
    const uint64_t metadata_insert_contract = SimtMetadataInsertContractForTask(task_id);
    plan[7] = metadata_insert_contract;

    if (SimtMetadataInsertContractPresent(metadata_insert_contract)) {
        const uint32_t writer_count = SimtMetadataInsertWriterCount(metadata_insert_contract);
        __gm__ uint64_t *history = task + kHistoryOffsetWords;
        history[0] = static_cast<uint64_t>(kWriterHistoryMagic) | (static_cast<uint64_t>(task_id) << 32U);
        history[1] = writer_count;
        for (uint32_t writer = 0U; writer < writer_count; ++writer) {
            uint32_t producer = 0U;
            uint32_t output_slot = 0U;
            if (!SimtMetadataWriterIntentAt(task_id, writer, &producer, &output_slot) || producer >= task_id ||
                output_slot >= kOutputsPerTask) {
                SimtPublishFatal(
                    fatal, ExecFatalReason::InsertProtocolFailed, build_owner, task_id G0_SIMT_TRACE_ARGUMENT
                );
                return false;
            }
            const uint32_t symbol_key = SimtSharedSymbolKey(producer, output_slot);
            for (uint32_t earlier = 0U; earlier < writer; ++earlier) {
                if (static_cast<uint32_t>(history[2U + earlier]) == symbol_key) {
                    SimtPublishFatal(
                        fatal, ExecFatalReason::InsertProtocolFailed, build_owner, task_id G0_SIMT_TRACE_ARGUMENT
                    );
                    return false;
                }
            }
            const uint32_t previous = SimtPreviousMetadataWriterForSymbol(task_id, producer, output_slot);
            if (previous == UINT32_MAX || previous < producer || previous >= task_id) {
                SimtPublishFatal(
                    fatal, ExecFatalReason::InsertProtocolFailed, build_owner, task_id G0_SIMT_TRACE_ARGUMENT
                );
                return false;
            }
            history[2U + writer] = static_cast<uint64_t>(symbol_key) | (static_cast<uint64_t>(previous) << 32U);
        }
    }

    __gm__ uint64_t *output_tensors = task + kOutputTensorOffsetWords;
    for (uint32_t output = 0U; output < output_count; ++output) {
        SimtStoreOutputDescriptor(
            output_tensors + output * kTensorDescWords, task_id, output, task_base
        );
    }
    // Fresh output descriptor/initial last_writer 同样是静态 owner 单写。
    // 先写 descriptor 和 initial last_writer，fence 后才发布 published；这样
    // metadata writer 一旦观察到 published，就不会与初始化 last_writer 竞争。
    for (uint32_t output = 0U; output < output_count; ++output) {
        __gm__ uint64_t *last_writer = task + kLastWriterOffsetWords + output * kAtomicStrideWords;
        asc_stcg(last_writer, static_cast<uint64_t>(task_id));
    }
    asc_threadfence();
    for (uint32_t output = 0U; output < output_count; ++output) {
        __gm__ uint64_t *published = task + kPublishedOffsetWords + output * kAtomicStrideWords;
        asc_stcg(published, static_cast<uint64_t>(task_id));
    }

    if (executable) {
        __gm__ uint64_t *payload = task + kExecPayloadOffsetWords;
        payload[0] = task_id;
        payload[1] = 0U;
        payload[2] = vend;
        payload[3] = static_cast<uint64_t>(kind - 1U) | (static_cast<uint64_t>(payload_bytes) << 32U);
        payload[4] = static_cast<uint64_t>(tensor_count) | (static_cast<uint64_t>(scalar_count) << 16U) |
                     (static_cast<uint64_t>(fanin_count) << 32U) | (static_cast<uint64_t>(engine) << 48U);
        payload[5] = static_cast<uint64_t>(1U) << 48U;
        payload[6] = 0U;
        payload[7] = 0U;

        uint32_t destination = kPayloadHeaderWords;
        // PA-UP 的 producer 顺序为 PV,PV,SF,Alloc,Alloc,Alloc。相邻 tensor
        // 引用同一 producer 时复用刚取得的 task base，避免对同一只读发布字
        // 重复发起 GM atomic-load；producer 切换后仍重新观察其权威发布值。
        uint32_t cached_producer = UINT32_MAX;
        uint64_t cached_producer_base = 0U;
        for (uint32_t tensor = 0U; tensor < tensor_count; ++tensor) {
            const uint32_t producer = SimtTensorSourceTask(task_id, tensor);
            uint64_t producer_base = 0U;
            if (producer != UINT32_MAX) {
                if (producer == task_id) {
                    producer_base = task_base;
                }
                else if (producer == cached_producer) {
                    producer_base = cached_producer_base;
                }
                else if (!SimtLoadTaskBase(
                               task_words, producer, fatal, timeout_ticks, task_id, build_owner,
                               &producer_base, state_access_count G0_SIMT_TRACE_ARGUMENT
                           )) {
                    SimtPublishFatal(
                        fatal, ExecFatalReason::Timeout, build_owner, task_id G0_SIMT_TRACE_ARGUMENT
                    );
                    return false;
                }
                else {
                    cached_producer = producer;
                    cached_producer_base = producer_base;
                }
            }
            if (producer == UINT32_MAX) {
                SimtStoreExternalDescriptor(payload + destination, batch_count, task_id, tensor);
            } else {
                SimtStoreOutputDescriptor(
                    payload + destination, producer, SimtTensorOutputSlot(task_id, tensor), producer_base
                );
            }
            destination += kTensorDescWords;
        }
        for (uint32_t scalar = 0U; scalar < scalar_count; ++scalar) {
            uint64_t value = 0U;
            if (kind == static_cast<uint32_t>(TaskKind::Qk) || kind == static_cast<uint32_t>(TaskKind::Pv)) {
                value = scalar == 0U ? kPaBlocksPerRequest :
                                       static_cast<uint64_t>(task_id / kTasksPerBatch) * kPaMaxBlocksPerRequest;
            } else if (kind == static_cast<uint32_t>(TaskKind::Sf)) {
                value = scalar == 0U ? kPaScaleBits : (scalar == 1U ? kPaBlocksPerRequest : kPaBlockSize);
            } else {
                value = 1U;
            }
            payload[destination++] = value;
        }
        for (uint32_t fanin = 0U; fanin < fanin_count; fanin += 2U) {
            uint32_t low = 0U;
            uint32_t high = 0U;
            if (kind == static_cast<uint32_t>(TaskKind::Sf) || kind == static_cast<uint32_t>(TaskKind::Pv)) {
                low = task_id - 1U;
            } else {
                low = task_id - 2U;
                high = task_id - 1U;
                if (fanin == 2U) {
                    low = (task_id / kTasksPerBatch) * kTasksPerBatch;
                    high = 0U;
                }
            }
            payload[destination++] = static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32U);
        }
        *payload_words_written = destination;
    } else {
        *payload_words_written = 0U;
    }
    asc_threadfence();
    return true;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtCommitTask(
    __gm__ uint64_t *task_words, __gm__ uint64_t *fatal,
    uint64_t timeout_ticks, uint32_t task_id, uint32_t build_owner, uint64_t completion_vend,
    uint32_t *insert_poll_count, uint32_t *predecessor_wait_count, int64_t *predecessor_observed G0_SIMT_TRACE_PARAMETER
) {
    constexpr uint32_t kTaskStrideWords = sizeof(FullPaTask) / sizeof(uint64_t);
    constexpr uint32_t kCompletionOffsetWords = offsetof(FullPaTask, completion) / sizeof(uint64_t);
    constexpr uint32_t kInsertOffsetWords = offsetof(FullPaTask, insert_completion) / sizeof(uint64_t);
    constexpr uint32_t kPlanOffsetWords = offsetof(FullPaTask, plan) / sizeof(uint64_t);
    constexpr uint32_t kOutputsOffsetWords = offsetof(FullPaTask, outputs) / sizeof(uint64_t);
    constexpr uint32_t kLastWriterOffsetWords =
        kOutputsOffsetWords + offsetof(SharedOutputCell, last_writer) / sizeof(uint64_t);
    constexpr uint32_t kHistoryOffsetWords = offsetof(FullPaTask, writer_history) / sizeof(uint64_t);
    constexpr uint32_t kExecOffsetWords = offsetof(FullPaTask, exec) / sizeof(uint64_t);
    constexpr uint32_t kAtomicStrideWords = sizeof(AtomicLine) / sizeof(uint64_t);
    __gm__ uint64_t *task = task_words + task_id * kTaskStrideWords;
    const uint32_t kind = task_id % kTasksPerBatch;
    const uint64_t metadata_insert_contract = task[kPlanOffsetWords + 7U];
    const bool publishes_metadata = SimtMetadataInsertContractPresent(metadata_insert_contract);
    const uint32_t writer_count = SimtMetadataInsertWriterCount(metadata_insert_contract);
    const int32_t previous_metadata_writer = SimtMetadataInsertPreviousWriter(metadata_insert_contract);
    __gm__ uint64_t *history = task + kHistoryOffsetWords;
    if ((metadata_insert_contract & ~kMetadataInsertContractAllowedMask) != 0U ||
        (!publishes_metadata && metadata_insert_contract != 0U) ||
        (publishes_metadata &&
         (writer_count == 0U || writer_count > kWriterHistoryMaxPerTask ||
          previous_metadata_writer >= static_cast<int32_t>(task_id) ||
          history[0] != (static_cast<uint64_t>(kWriterHistoryMagic) | (static_cast<uint64_t>(task_id) << 32U)) ||
          static_cast<uint32_t>(history[1]) != writer_count || static_cast<uint32_t>(history[1] >> 32U) != 0U))) {
        SimtPublishFatal(
            fatal, ExecFatalReason::InsertProtocolFailed, build_owner, task_id G0_SIMT_TRACE_ARGUMENT
        );
        return false;
    }
    for (uint32_t writer = 0U; writer < writer_count; ++writer) {
        const uint32_t symbol_key = static_cast<uint32_t>(history[2U + writer]);
        const uint32_t previous = static_cast<uint32_t>(history[2U + writer] >> 32U);
        uint32_t producer = 0U;
        uint32_t output_slot = 0U;
        if (!SimtDecodeSharedSymbolKey(symbol_key, &producer, &output_slot) || producer >= task_id ||
            previous < producer || previous >= task_id) {
            SimtPublishFatal(
                fatal, ExecFatalReason::InsertProtocolFailed, build_owner, task_id G0_SIMT_TRACE_ARGUMENT
            );
            return false;
        }
        for (uint32_t earlier = 0U; earlier < writer; ++earlier) {
            if (static_cast<uint32_t>(history[2U + earlier]) == symbol_key) {
                SimtPublishFatal(
                    fatal, ExecFatalReason::InsertProtocolFailed, build_owner, task_id G0_SIMT_TRACE_ARGUMENT
                );
                return false;
            }
        }
        __gm__ uint64_t *published = task_words + producer * kTaskStrideWords + kOutputsOffsetWords +
                                     output_slot * kAtomicStrideWords;
        if (!SimtWaitOutputPublished(
                published, static_cast<uint64_t>(producer), fatal, timeout_ticks,
                task_id, build_owner G0_SIMT_TRACE_ARGUMENT
            )) {
            SimtPublishFatal(
                fatal, ExecFatalReason::Timeout, build_owner, task_id G0_SIMT_TRACE_ARGUMENT
            );
            return false;
        }
    }
    *predecessor_observed = -1;
    if (previous_metadata_writer >= 0) {
        const uint32_t predecessor_task = static_cast<uint32_t>(previous_metadata_writer);
        ++*predecessor_wait_count;
        __gm__ uint64_t *predecessor = task_words + predecessor_task * kTaskStrideWords + kInsertOffsetWords;
        uint32_t polls = 0U;
        if (!SimtWaitAtomicValue(
                predecessor, static_cast<uint64_t>(predecessor_task), fatal, timeout_ticks, task_id, build_owner,
                &polls G0_SIMT_TRACE_ARGUMENT
            )) {
            *insert_poll_count += polls;
            SimtPublishFatal(fatal, ExecFatalReason::Timeout, build_owner, task_id G0_SIMT_TRACE_ARGUMENT);
            return false;
        }
        *insert_poll_count += polls;
        if (*predecessor_observed < static_cast<int64_t>(predecessor_task)) {
            *predecessor_observed = static_cast<int64_t>(predecessor_task);
        }
    }
    if (publishes_metadata) {
        asc_threadfence();
    }
    for (uint32_t writer = 0U; writer < writer_count; ++writer) {
        const uint32_t symbol_key = static_cast<uint32_t>(history[2U + writer]);
        uint32_t producer = 0U;
        uint32_t output_slot = 0U;
        if (!SimtDecodeSharedSymbolKey(symbol_key, &producer, &output_slot)) {
            SimtPublishFatal(
                fatal, ExecFatalReason::InsertProtocolFailed, build_owner, task_id G0_SIMT_TRACE_ARGUMENT
            );
            return false;
        }
        __gm__ uint64_t *last_writer = task_words + producer * kTaskStrideWords + kLastWriterOffsetWords +
                                       output_slot * kAtomicStrideWords;
        // ordinary TensorMap 的全局 predecessor 已保证这里只有一个 writer。
        // last_writer 是串行插入产生的派生元数据，用 non-cacheable store
        // 发布即可；继续做 per-symbol CAS 只会把同一顺序重复串行化。
        asc_stcg(last_writer, static_cast<uint64_t>(task_id));
    }
    if (publishes_metadata) {
        asc_threadfence();
    }

    if (publishes_metadata) {
        // 静态 task owner 与全局 predecessor turn 共同证明 completion cell
        // 只有当前 writer 写入；non-cacheable store 即可完成 baton 发布。
        asc_stcg(task + kInsertOffsetWords, static_cast<uint64_t>(task_id));
    }

    if (kind == static_cast<uint32_t>(TaskKind::Alloc)) {
        // The static residue partition gives this task exactly one builder.
        // Publish immutable vend before the release flag; no claim/CAS is needed.
        asc_stcg(task + kCompletionOffsetWords + 1U, completion_vend);
        asc_threadfence();
        asc_stcg(task + kCompletionOffsetWords, static_cast<uint64_t>(1U));
    } else {
        uint32_t tensor_count = kind == static_cast<uint32_t>(TaskKind::Up) ? 7U : 4U;
        uint32_t scalar_count = kind == static_cast<uint32_t>(TaskKind::Sf) ? 3U : 2U;
        uint32_t fanin_count =
            kind == static_cast<uint32_t>(TaskKind::Up) ? 3U : (kind == static_cast<uint32_t>(TaskKind::Qk) ? 0U : 1U);
        const uint32_t payload_bytes = kCacheLineBytes + tensor_count * kTensorDescBytes +
                                       scalar_count * sizeof(uint64_t) + fanin_count * sizeof(int32_t);
        const uint32_t payload_lines = (payload_bytes + kCacheLineBytes - 1U) / kCacheLineBytes;
        const uint32_t engine =
            kind == static_cast<uint32_t>(TaskKind::Qk) || kind == static_cast<uint32_t>(TaskKind::Pv) ?
                static_cast<uint32_t>(ExecEngineClass::Aic) :
                static_cast<uint32_t>(ExecEngineClass::Aiv);
        const uint64_t built = (static_cast<uint64_t>(ExecPhase::Built) << kStatePhaseShift) |
                               (static_cast<uint64_t>(build_owner) << kStateBuildOwnerShift) |
                               (static_cast<uint64_t>(kUnboundOwner) << kStateExecuteOwnerShift) |
                               (static_cast<uint64_t>(engine) << kStateEngineShift) |
                               (static_cast<uint64_t>(payload_lines) << kStatePayloadLinesShift) |
                               (static_cast<uint64_t>(task_id) << kStateTaskIdShift);
        // completion.vend is immutable build output.  Publish it with BUILT so
        // the unique executor only has to release completion.flag after work.
        asc_stcg(task + kCompletionOffsetWords + 1U, completion_vend);
        asc_threadfence();
        asc_stcg(task + kExecOffsetWords, built);
    }
    return true;
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(kBuilderThreadCount) void G0SimtBuildTasks(
    __gm__ uint64_t *task_words, __gm__ uint64_t *heap_words,
    __gm__ uint64_t *builder_started, __gm__ uint64_t *builder_finished, __gm__ uint64_t *fatal,
    __gm__ uint64_t *thread_report_words, uint64_t nonce, uint64_t timeout_ticks, uint32_t batch_count,
    uint32_t task_count, uint32_t builder_instance, uint32_t builder_count, uint32_t build_owner
)
{
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    const uint32_t warp = thread / kWarpSize;
    const uint32_t lane = thread % kWarpSize;
    const bool active = lane == 0U && warp < kBuilderWarpCount;
    const uint32_t global_thread = builder_instance * kBuilderThreadCount + thread;
    const uint32_t global_warp = builder_instance * kBuilderWarpCount + warp;
    const uint32_t first_assigned_task = builder_instance * kBuilderWarpCount + warp;
    const uint32_t task_stride = builder_count * kBuilderWarpCount;
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    // 保持与生产 G0 完全相同的 async_invoke 参数形状。CCEC 会把
    // VF 参数打包到 UB，额外的 trace pointer 会触发 A5 未对齐访问；
    // sidecar 按已冻结 ABI 从现有 task_words 向后推导。
    __gm__ uint8_t *task_bytes = reinterpret_cast<__gm__ uint8_t *>(task_words);
    __gm__ g0_swimlane::TraceState *trace_state =
        reinterpret_cast<__gm__ g0_swimlane::TraceState *>(task_bytes + kG0TraceFromTasksOffsetBytes);
#endif
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
    // volatile 字段强制 CCEC 逐字段生成标量 SIMT stack 访问，
    // 不允许把 32B 计数器对象合并为要求额外对齐的 VEC 宽访问。
    SimtTraceCounters simt_counter_storage;
    simt_counter_storage.atomic_calls = 0U;
    simt_counter_storage.poll_calls = 0U;
    simt_counter_storage.record_count = 0U;
    simt_counter_storage.dropped_records = 0U;
    simt_counter_storage.poll_records = 0U;
    simt_counter_storage.reserved = 0U;
    SimtTraceCounters *simt_counters = &simt_counter_storage;
    __gm__ g0_swimlane::TraceRecord *simt_records = &trace_state->simt_records[global_warp][0];
    __gm__ g0_swimlane::TraceLogControl *simt_control = &trace_state->simt_logs[global_warp];
#endif
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    __gm__ g0_swimlane::BuilderTaskTrace *builder_traces = &trace_state->builders[0];
#endif
    uint32_t tasks_built = 0U;
    uint32_t prepared = 0U;
    uint32_t committed = 0U;
    uint32_t attempts = 0U;
    uint32_t producer_base_polls = 0U;
    uint32_t insert_waits = 0U;
    uint32_t claim_losses = 0U;
    uint32_t first_task = UINT32_MAX;
    uint32_t last_task = UINT32_MAX;
    uint64_t checksum = 0U;

    constexpr uint32_t kTaskStrideWords = sizeof(FullPaTask) / sizeof(uint64_t);
    constexpr uint32_t kBuildReportOffsetWords = offsetof(FullPaTask, build_report) / sizeof(uint64_t);
    constexpr uint32_t kThreadReportStrideWords = sizeof(FullPaBuilderThreadReport) / sizeof(uint64_t);
    const bool start_ready =
        SimtWaitBuilderStart(
            builder_started, fatal, timeout_ticks, thread, build_owner, builder_count,
            active G0_SIMT_TRACE_ARGUMENT
        );
    bool build_ready = start_ready;
    uint64_t aggregate_cursor = 0U;
    if (active && build_ready) {
        constexpr uint32_t kAggregateVendOffsetWords =
            offsetof(FullPaHeapControl, aggregate_vend) / sizeof(uint64_t);
        uint64_t aggregate_reserve = 0U;
        for (uint32_t task_id = first_assigned_task; task_id < task_count; task_id += task_stride) {
            aggregate_reserve += SimtTaskReserveBytes(task_id);
        }
        aggregate_cursor = G0_TRACE_SIMT_ADD(
            simt_trace, g0_swimlane::kTraceNoTask,
            aggregate_reserve == 0U ? g0_swimlane::AtomicSite::SimtHeapVendLoad :
                                      g0_swimlane::AtomicSite::SimtHeapVendReserve,
            heap_words + kAggregateVendOffsetWords, aggregate_reserve, true
        );
        if (aggregate_reserve > kHeapBytes || aggregate_cursor > kHeapBytes - aggregate_reserve) {
            SimtPublishFatal(
                fatal, ExecFatalReason::HeapReservationFailed, build_owner,
                first_assigned_task G0_SIMT_TRACE_ARGUMENT
            );
            build_ready = false;
        }
    }
    if (active && build_ready) {
        for (uint32_t task_id = first_assigned_task; task_id < task_count; task_id += task_stride) {
            // Static ownership gives this leader an independent task stream.
            // SimtPrepareTask/SimtCommitTask propagate local errors, and every
            // blocking wait already samples the shared fatal word through its
            // watchdog.  A same-address atomic fatal read before every healthy
            // task only serialized all builder warps; the CPU protocol model
            // likewise has no unconditional per-task fatal read.
            ++attempts;
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
            const uint64_t trace_attempt_begin = clock();
#endif
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
            __gm__ g0_swimlane::BuilderTaskTrace *builder_trace = builder_traces + task_id;
            builder_trace->attempt_begin = trace_attempt_begin;
            builder_trace->claim_end = clock();
            builder_trace->task_id = task_id;
            builder_trace->builder_thread = global_thread;
            builder_trace->build_owner = build_owner;
#endif
            uint64_t completion_vend = 0U;
            uint32_t payload_words = 0U;
            if (!SimtPrepareTask(
                    task_words, heap_words, fatal, nonce, timeout_ticks, batch_count, task_count, task_id,
                    global_thread, build_owner, aggregate_cursor, &completion_vend, &payload_words,
                    &producer_base_polls G0_SIMT_TRACE_ARGUMENT
                )) {
                break;
            }
            aggregate_cursor = completion_vend;
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
            builder_trace->prepare_end = clock();
            builder_trace->commit_begin = clock();
#endif
            ++prepared;
            uint32_t task_insert_polls = 0U;
            uint32_t task_predecessor_waits = 0U;
            int64_t predecessor_observed = -1;
            if (!SimtCommitTask(
                    task_words, fatal, timeout_ticks, task_id, build_owner, completion_vend,
                    &task_insert_polls, &task_predecessor_waits, &predecessor_observed G0_SIMT_TRACE_ARGUMENT
                )) {
                break;
            }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
            builder_trace->commit_end = clock();
            builder_trace->insert_poll_count = task_insert_polls;
#endif
            insert_waits += task_predecessor_waits;
            ++committed;
            ++tasks_built;
            if (tasks_built == 1U) {
                first_task = task_id;
            }
            last_task = task_id;

            const uint32_t kind = task_id % kTasksPerBatch;
            const uint32_t output_count = SimtTaskOutputCount(task_id);
            uint32_t phases = kBuildPreparedBit | kBuildOutputsPublishedBit | kBuildInsertCommittedBit;
            phases |= kind == static_cast<uint32_t>(TaskKind::Alloc) ? kBuildAllocCompletedBit : kBuildExecPublishedBit;
            __gm__ uint64_t *report = task_words + task_id * kTaskStrideWords + kBuildReportOffsetWords;
            asc_stcg(
                report,
                static_cast<uint64_t>(task_id) | (static_cast<uint64_t>(global_thread) << 32U)
            );
            asc_stcg(report + 1U, static_cast<uint64_t>(global_warp));
            asc_stcg(
                report + 2U,
                static_cast<uint64_t>(phases) | (static_cast<uint64_t>(output_count) << 32U)
            );
            asc_stcg(
                report + 3U,
                static_cast<uint64_t>(payload_words) | (static_cast<uint64_t>(task_insert_polls) << 32U)
            );
            asc_stcg(report + 4U, static_cast<uint64_t>(predecessor_observed));
            asc_stcg(report + 5U, static_cast<uint64_t>(1U) | (static_cast<uint64_t>(1U) << 32U));
            asc_stcg(report + 6U, static_cast<uint64_t>(1U) | (static_cast<uint64_t>(1U) << 32U));
            asc_stcg(report + 7U, nonce);
            asc_threadfence();
            if (task_id + 1U == task_count) {
                if (G0_TRACE_SIMT_CAS(
                        simt_trace, task_id, g0_swimlane::AtomicSite::SimtBuilderFinishedPublish,
                        builder_finished, static_cast<uint64_t>(0U), static_cast<uint64_t>(1U), true
                    ) != 0U) {
                    SimtPublishFatal(
                        fatal, ExecFatalReason::ControlPublishConflict, build_owner,
                        task_id G0_SIMT_TRACE_ARGUMENT
                    );
                }
            }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
            builder_trace->report_end = clock();
            asc_threadfence();
            builder_trace->launch_nonce = nonce;
            asc_threadfence();
#endif
        }
    }
    if (active) {
        checksum = SimtBuilderReportChecksum(
            nonce, global_thread, task_count, tasks_built, first_task, last_task, attempts, prepared, committed,
            insert_waits, claim_losses
        );

        __gm__ uint64_t *thread_report = thread_report_words + global_thread * kThreadReportStrideWords;
        thread_report[0] = static_cast<uint64_t>(global_thread) | (static_cast<uint64_t>(global_warp) << 32U);
        thread_report[1] = static_cast<uint64_t>(lane) | (static_cast<uint64_t>(1U) << 32U);
        thread_report[2] = static_cast<uint64_t>(tasks_built) | (static_cast<uint64_t>(first_task) << 32U);
        thread_report[3] = static_cast<uint64_t>(last_task) | (static_cast<uint64_t>(attempts) << 32U);
        thread_report[4] = static_cast<uint64_t>(prepared) | (static_cast<uint64_t>(committed) << 32U);
        thread_report[5] = static_cast<uint64_t>(insert_waits) | (static_cast<uint64_t>(claim_losses) << 32U);
        thread_report[6] = nonce;
        thread_report[7] = checksum;
        asc_threadfence();
    }
#if defined(SIMT_CROSS_CORE_G0_SIMT_ATOMIC_TRACE_ENABLED)
    if (active) {
        SimtTraceFinish(simt_control, simt_counters, nonce, global_warp);
    }
#endif
}

#endif  // defined(__DAV_VEC__)

__aicore__ __attribute__((always_inline)) inline void ResetToken(__gm__ ExecutionToken *token) {
    token->control.phase = ExecTokenPhase::Idle;
    token->control.task_id = UINT32_MAX;
    token->control.build_owner = UINT32_MAX;
    token->control.execute_owner = UINT32_MAX;
    token->control.engine_class = ExecEngineClass::None;
    token->control.payload_lines = 0U;
    token->control.payload_bytes = 0U;
    token->control.fanin_ready_prefix = 0U;
    token->control.payload_address = 0U;
    token->control.completion_vend = 0U;
    token->control.function_and_reference = 0U;
    token->control.shape_and_scalar_offset = 0U;
    // Production ResetExecutionToken resets only this control line.  dispatch
    // intentionally retains the last binding for the owner-local token.
}

__aicore__ __attribute__((always_inline)) inline void
PublishTerminalTokenState(
    __gm__ FullPaState *state, uint32_t owner, uint32_t ticket_count G0_SCALAR_TRACE_PARAMETER
) {
    const uint32_t used_tokens = ticket_count < kTokensPerOwner ? ticket_count : kTokensPerOwner;
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    const uint64_t dcci_begin = used_tokens == 0U ? 0U : static_cast<uint64_t>(get_sys_cnt());
#endif
    for (uint32_t slot = 0U; slot < used_tokens; ++slot) {
        __gm__ ExecutionToken *token = &state->tokens[owner][slot];
        dcci(static_cast<__gm__ void *>(&token->control), kSingleCacheLine);
        __gm__ uint8_t *dispatch = reinterpret_cast<__gm__ uint8_t *>(&token->dispatch);
        // The active tensor/scalar prefix occupies lines 0/1.  args[48/49]
        // and local_context share line 6; global_context and padding line 7.
        dcci(static_cast<__gm__ void *>(dispatch), kSingleCacheLine);
        dcci(static_cast<__gm__ void *>(dispatch + kCacheLineBytes), kSingleCacheLine);
        dcci(static_cast<__gm__ void *>(dispatch + 6U * kCacheLineBytes), kSingleCacheLine);
        dcci(static_cast<__gm__ void *>(dispatch + 7U * kCacheLineBytes), kSingleCacheLine);
    }
    if (used_tokens != 0U) {
        // Owner-local tokens have no concurrent ordinary writer.  One terminal
        // clean+invalidate pass publishes both reset control and the retained
        // last binding without adding DCCI to every execution.
        dsb(DSB_ALL);
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
        const uint64_t dcci_end = static_cast<uint64_t>(get_sys_cnt());
        ScalarTraceDcciRecord(
            trace, g0_swimlane::kTraceNoTask, g0_swimlane::DcciSite::TerminalTokenInvalidate,
            g0_swimlane::DcciOp::Invalidate, dcci_begin, dcci_end, 5U * used_tokens, 5U * used_tokens
        );
#endif
    }
}

__aicore__ __attribute__((always_inline)) inline uint32_t BusyTokenCount(__gm__ FullPaState *state, uint32_t owner) {
    uint32_t busy = 0U;
    for (uint32_t slot = 0U; slot < kTokensPerOwner; ++slot) {
        busy += state->tokens[owner][slot].control.phase == ExecTokenPhase::Idle ? 0U : 1U;
    }
    return busy;
}

__aicore__ __attribute__((always_inline)) inline uint64_t PayloadWord(__gm__ const FullPaTask *task, uint32_t word) {
    return task->exec.payload.words[word];
}

__aicore__ __attribute__((always_inline)) inline void
InvalidatePayloadLines(
    __gm__ FullPaTask *task, uint32_t task_id, uint32_t payload_lines G0_SCALAR_TRACE_PARAMETER
) {
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    const uint64_t dcci_begin = static_cast<uint64_t>(get_sys_cnt());
#endif
    __gm__ uint8_t *payload =
        reinterpret_cast<__gm__ uint8_t *>(const_cast<__gm__ uint64_t *>(&task->exec.payload.words[0]));
    for (uint32_t line = 0U; line < payload_lines; ++line) {
        dcci(static_cast<__gm__ void *>(payload + line * kCacheLineBytes), kSingleCacheLine);
    }
    dsb(DSB_ALL);
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    const uint64_t dcci_end = static_cast<uint64_t>(get_sys_cnt());
    ScalarTraceDcciRecord(
        trace, task_id, g0_swimlane::DcciSite::ExecPayloadInvalidate,
        g0_swimlane::DcciOp::Invalidate, dcci_begin, dcci_end, payload_lines, payload_lines
    );
#endif
}

__aicore__ __attribute__((always_inline)) inline bool
BindPublishedPayload(
    __gm__ FullPaState *state, uint32_t owner, __gm__ ExecutionToken *token G0_SCALAR_TRACE_PARAMETER
) {
    const uint32_t task_id = token->control.task_id;
    __gm__ FullPaTask *task = &state->tasks[task_id];
    const TaskKind kind = TaskKindAt(task_id);
    const TaskExecShape shape = TaskShape(kind);
    ExecPayloadLayout layout{};
    if (!IsBuilderOwner(token->control.build_owner, state->control.builder_count) ||
        !OwnerCanExecute(owner, shape.engine_class, state->control.builder_count) ||
        !ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout)) {
        return false;
    }
    InvalidatePayloadLines(task, task_id, layout.payload_lines G0_SCALAR_TRACE_ARGUMENT);
    const uint64_t word0 = PayloadWord(task, 0U);
    const uint64_t word3 = PayloadWord(task, 3U);
    const uint64_t word4 = PayloadWord(task, 4U);
    const uint64_t word5 = PayloadWord(task, 5U);
    if (static_cast<uint32_t>(word0) != task_id || (word0 >> 32U) != 0U || PayloadWord(task, 1U) != 0U ||
        static_cast<uint32_t>(word3) != TaskFunctionId(kind) ||
        static_cast<uint32_t>(word3 >> 32U) != layout.payload_bytes ||
        static_cast<uint16_t>(word4) != shape.tensor_count ||
        static_cast<uint16_t>(word4 >> 16U) != shape.scalar_count ||
        static_cast<uint16_t>(word4 >> 32U) != shape.fanin_count ||
        static_cast<uint8_t>(word4 >> 48U) != static_cast<uint8_t>(shape.engine_class) ||
        static_cast<uint8_t>(word4 >> 56U) != 0U || static_cast<uint32_t>(word5) != 0U ||
        static_cast<uint16_t>(word5 >> 32U) != 0U || static_cast<uint16_t>(word5 >> 48U) != 1U ||
        PayloadWord(task, 6U) != 0U || PayloadWord(task, 7U) != 0U) {
        return false;
    }

    // BUILT is the release publication for an immutable payload.  Executors
    // validate the layout-bearing header before using any offsets, then bind
    // the published descriptor/scalar storage directly.  Reconstructing and
    // comparing every descriptor, scalar and fan-in here duplicated the full
    // independent host oracle after the kernel and added work to every task's
    // Scalar critical path.  The host oracle remains exhaustive; this hot path
    // retains the checks required for safe, signature-driven parsing.
    for (uint32_t tensor_index = 0U; tensor_index < shape.tensor_count; ++tensor_index) {
        const uint32_t offset = layout.tensor_word_offset + tensor_index * kTensorDescWords;
        token->dispatch.args[tensor_index] =
            reinterpret_cast<uint64_t>(const_cast<__gm__ uint64_t *>(&task->exec.payload.words[offset]));
    }
    for (uint32_t scalar = 0U; scalar < shape.scalar_count; ++scalar) {
        const uint64_t value = PayloadWord(task, layout.scalar_word_offset + scalar);
        token->dispatch.args[shape.tensor_count + scalar] = value;
    }

    token->control.payload_lines = layout.payload_lines;
    token->control.payload_bytes = layout.payload_bytes;
    token->control.payload_address =
        reinterpret_cast<uint64_t>(const_cast<__gm__ uint64_t *>(&task->exec.payload.words[0]));
    token->control.completion_vend = PayloadWord(task, 2U);
    token->control.function_and_reference = static_cast<uint64_t>(TaskFunctionId(kind));
    token->control.shape_and_scalar_offset = PackExecutionTokenShapeAndScalarOffset(
        shape.tensor_count, shape.scalar_count, shape.fanin_count, static_cast<uint16_t>(layout.scalar_word_offset)
    );
    __gm__ uint64_t *local_context = reinterpret_cast<__gm__ uint64_t *>(&token->dispatch.local_context[0]);
    local_context[0] = task_id;
    local_context[1] = owner;
    local_context[2] = layout.payload_bytes;
    local_context[3] = layout.payload_lines;
    local_context[4] = shape.fanin_count;
    local_context[5] = token->control.completion_vend;
    *reinterpret_cast<__gm__ uint32_t *>(&token->dispatch.global_context[0]) = state->control.batch_count;
    token->dispatch.args[kDispatchLocalContextIndex] = reinterpret_cast<uint64_t>(&token->dispatch.local_context[0]);
    token->dispatch.args[kDispatchGlobalContextIndex] = reinterpret_cast<uint64_t>(&token->dispatch.global_context[0]);
    return true;
}

__aicore__ __attribute__((always_inline)) inline bool
FaninReady(
    __gm__ FullPaState *state, __gm__ ExecutionToken *token
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    , ScalarPollEpisode *fanin_episode
#endif
    G0_SCALAR_TRACE_PARAMETER
) {
    const uint32_t task_id = token->control.task_id;
    const uint32_t fanin_count = TaskFaninCount(task_id);
    while (token->control.fanin_ready_prefix < fanin_count) {
        const int32_t producer = TaskFanin(task_id, token->control.fanin_ready_prefix);
        if (producer < 0) {
            return false;
        }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
        const uint64_t completion = ScalarTracePollLoad(
            trace, fanin_episode, task_id, g0_swimlane::AtomicSite::ScalarFaninFlagPoll,
            &state->tasks[static_cast<uint32_t>(producer)].completion.flag
        );
#else
        const uint64_t completion =
            ScalarAtomicLoad(&state->tasks[static_cast<uint32_t>(producer)].completion.flag);
#endif
        if (completion != 1U) {
            return false;
        }
        ++token->control.fanin_ready_prefix;
    }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    ScalarTraceFlushPoll(trace, fanin_episode);
#endif
    return true;
}

__aicore__ __attribute__((always_inline)) inline void PublishExecutionWitness(
    __gm__ FullPaState *state, uint32_t owner, uint32_t task_id, TaskKind kind, uint64_t output_checksum,
    uint32_t fanin_ready_prefix G0_SCALAR_TRACE_PARAMETER
) {
    __gm__ FullPaExecutionWitness *witness = &state->tasks[task_id].execution_witness;
    // words[1..3] are immutable identity and words[4..7] remain the host
    // initialized zero image.  A single terminal hash binds every dynamic
    // witness component, avoiding four redundant GM stores and a standalone
    // fence per task.  The caller's release fence orders this state before
    // completion.flag.
    // The global dispatch ticket assigns exactly one executor to each task.
    // Publish the terminal witness word last; no competing writer exists.
    StoreDev64(
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&witness->state)),
        ExecutionWitnessState(
            state->control.launch_nonce, task_id, kind, owner, 1U,
            kCompletionSequenceWorkloadWitnessVendFlagDone, output_checksum, fanin_ready_prefix
        )
    );
}

__aicore__ __attribute__((always_inline)) inline bool
RunClaimedWorkload(
    __gm__ FullPaState *state, uint32_t owner, __gm__ ExecutionToken *token G0_SCALAR_TRACE_PARAMETER
) {
    const uint32_t task_id = token->control.task_id;
    const TaskKind kind = TaskKindAt(task_id);
    __gm__ float *workspace = reinterpret_cast<__gm__ float *>(state->control.workspace_base);
    __gm__ float *input_a = workspace;
    __gm__ float *input_b = workspace + kWorkloadTileElements;
    const uint32_t kind_slot = kind == TaskKind::Pv || kind == TaskKind::Up ? 1U : 0U;
    __gm__ float *output = workspace + (kWorkloadSharedInputTiles + owner * kWorkloadOutputTilesPerOwner + kind_slot) *
                                           kWorkloadTileElements;
    uint64_t output_poison = state->control.launch_nonce ^
                             (static_cast<uint64_t>(task_id) * UINT64_C(0x9E3779B97F4A7C15)) ^
                             UINT64_C(0xD15EA5E0C001D00D);
    if (output_poison == ExpectedWorkloadOutputPair(kind)) {
        output_poison ^= UINT64_C(0xFFFFFFFFFFFFFFFF);
    }
    // Owners reuse one tile per engine kind.  Poison the sampled output before
    // every task so a skipped workload/TSTORE cannot inherit the previous
    // task's valid checksum and publish a false-positive witness.
    StoreDev64(reinterpret_cast<__gm__ uint64_t *>(output), output_poison);
    dsb(DSB_ALL);
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    TraceExecutorBegin(state, owner, task_id);
#endif
#if defined(__DAV_VEC__)
    if (kind == TaskKind::Sf) {
        RunG0VectorAdd(input_a, input_b, output, state->control.sf_repeats);
    } else if (kind == TaskKind::Up) {
        RunG0VectorMultiply(input_a, input_b, output, state->control.up_repeats);
    } else {
        TracePublishFatal(
            state, ExecFatalReason::InvalidTokenPayload, owner, task_id, trace
        );
        return false;
    }
#else
    if (kind == TaskKind::Qk) {
        RunG0CubeMatmul(input_a, input_b, output, state->control.qk_repeats);
    } else if (kind == TaskKind::Pv) {
        RunG0CubeMatmul(input_a, input_b, output, state->control.pv_repeats);
    } else {
        TracePublishFatal(
            state, ExecFatalReason::InvalidTokenPayload, owner, task_id, trace
        );
        return false;
    }
#endif
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    TraceExecutorWorkloadEnd(state, owner, task_id);
#endif
    const uint64_t checksum = LoadDev64(reinterpret_cast<__gm__ const uint64_t *>(output));
    if (checksum != ExpectedWorkloadOutputPair(kind)) {
        TracePublishFatal(
            state, ExecFatalReason::InvalidTokenPayload, owner, task_id, trace
        );
        return false;
    }
    PublishExecutionWitness(
        state, owner, task_id, kind, checksum,
        token->control.fanin_ready_prefix G0_SCALAR_TRACE_ARGUMENT
    );

    __gm__ FullPaTask *task = &state->tasks[task_id];
    // completion.flag is the release boundary consumed by fan-in polling.
    // Make workload output and dynamic witness globally visible first; vend
    // was already published before BUILT by the unique task builder.
    dsb(DSB_ALL);
    StoreDev64(
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&task->completion.flag)), 1U
    );
    StoreDev64(
        reinterpret_cast<__gm__ uint64_t *>(
            const_cast<__gm__ int64_t *>(&task->exec.control.state)
        ),
        DoneState(task_id, token->control.build_owner, owner)
    );
    // Per-task done/aic_done/aiv_done increments duplicated the exact
    // completion total already encoded in every owner's drain contribution.
    // Keeping only that canonical final-drain accounting removes three shared
    // atomic hot spots without weakening end-of-kernel validation.
    return true;
}

__aicore__ __attribute__((always_inline)) inline bool
AdvanceToken(
    __gm__ FullPaState *state, uint32_t owner, __gm__ ExecutionToken *token, FullPaRoleResult *result
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    , ScalarPollEpisode *built_episode, ScalarPollEpisode *fanin_episode
#endif
    G0_SCALAR_TRACE_PARAMETER
) {
    if (token->control.phase == ExecTokenPhase::Idle) {
        return false;
    }
    const uint32_t task_id = token->control.task_id;
    __gm__ FullPaTask *task = &state->tasks[task_id];
    if (token->control.phase == ExecTokenPhase::WaitingBuilt) {
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
        const uint64_t observed = ScalarTracePollLoad(
            trace, built_episode, task_id, g0_swimlane::AtomicSite::ScalarExecStatePoll,
            &task->exec.control.state
        );
#else
        const uint64_t observed = ScalarAtomicLoad(&task->exec.control.state);
#endif
        if (observed == 0U) {
            return false;
        }
        const DecodedExecState decoded = DecodeExecState(static_cast<int64_t>(observed));
        if (decoded.valid && decoded.phase == ExecPhase::Building && decoded.task_id == task_id &&
            IsBuilderOwner(decoded.build_owner, state->control.builder_count) &&
            observed == BuildingState(task_id, decoded.build_owner)) {
            return false;
        }
        if (!decoded.valid || decoded.phase != ExecPhase::Built || decoded.task_id != task_id ||
            !IsBuilderOwner(decoded.build_owner, state->control.builder_count) ||
            observed != BuiltState(task_id, decoded.build_owner)) {
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
            ScalarTraceFlushPoll(trace, built_episode);
#endif
            ++result->claim_lost_count;
            TracePublishFatal(state, ExecFatalReason::InvalidBuiltControl, owner, task_id, trace);
            return false;
        }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
        ScalarTraceFlushPoll(trace, built_episode);
#endif
        token->control.build_owner = decoded.build_owner;
        // The atomic dispatch cursor returns every table ticket once.  No
        // observer consumes an intermediate CLAIMED value, so keep BUILT until
        // the unique executor publishes terminal DONE instead of writing the
        // same control word twice per task.
        ++result->claim_count;
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
        TraceExecutorClaim(state, owner, task_id);
#endif
        token->control.phase = ExecTokenPhase::Binding;
        if (!BindPublishedPayload(state, owner, token G0_SCALAR_TRACE_ARGUMENT)) {
            TracePublishFatal(state, ExecFatalReason::ClaimedPayloadInvalid, owner, task_id, trace);
            return false;
        }
        token->control.phase = ExecTokenPhase::WaitingFanin;
    }
    if (token->control.phase == ExecTokenPhase::WaitingFanin) {
        if (!FaninReady(
                state, token
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
                , fanin_episode
#endif
                G0_SCALAR_TRACE_ARGUMENT
            )) {
            return false;
        }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
        TraceExecutorFaninReady(state, owner, task_id);
#endif
        token->control.phase = ExecTokenPhase::EngineInflight;
    }
    if (token->control.phase == ExecTokenPhase::EngineInflight) {
        token->control.phase = ExecTokenPhase::Completing;
        if (!RunClaimedWorkload(state, owner, token G0_SCALAR_TRACE_ARGUMENT)) {
            return false;
        }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
        TraceExecutorEnd(state, owner, task_id);
#endif
        ++result->execute_count;
        ++result->completed_by_kind[static_cast<uint32_t>(TaskKindAt(task_id))];
        ResetToken(token);
        return true;
    }
    return false;
}

__aicore__ __attribute__((always_inline)) inline uint32_t
LoadDispatchTaskId(
    __gm__ const uint32_t *task_ids, uint32_t ticket G0_SCALAR_TRACE_PARAMETER
) {
    // dispatch 表在 launch 前初始化，kernel 内只读。逐 ticket 做整条
    // cacheline DCCI+DSB 会把同一行重复失效 16 次；一次非缓存 64-bit
    // device load 同时覆盖相邻两个 id，且不存在 polling writer starvation。
    const uint32_t pair_ticket = ticket & ~1U;
    const __gm__ uint64_t *pair_address =
        reinterpret_cast<__gm__ const uint64_t *>(task_ids + pair_ticket);
    const uint64_t pair = LoadDev64(pair_address);
    return (ticket & 1U) == 0U ? static_cast<uint32_t>(pair) : static_cast<uint32_t>(pair >> 32U);
}

__aicore__ __attribute__((always_inline)) inline void
RunExecutor(
    __gm__ FullPaState *state, uint32_t owner, FullPaRoleResult *result G0_SCALAR_TRACE_PARAMETER
) {
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    TraceRoleTimestamp(state, owner, 3U);
#endif
    const ExecEngineClass engine = OwnerEngine(owner, state->control.builder_count);
    __gm__ AtomicLine *cursor =
        engine == ExecEngineClass::Aic ? &state->exec_dispatch.aic_next : &state->exec_dispatch.aiv_next;
    __gm__ uint32_t *task_ids =
        engine == ExecEngineClass::Aic ? &state->exec_dispatch.aic_task_ids[0] : &state->exec_dispatch.aiv_task_ids[0];
    const uint32_t task_count =
        engine == ExecEngineClass::Aic ? state->exec_dispatch.aic_task_count : state->exec_dispatch.aiv_task_count;
    bool exhausted = false;
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    // 不对本地聚合数组做动态寻址。CCEC 在这种场景下可能把
    // `&episodes[slot]` 指到 VEC spill 片段，而不是 C++ 对象字段。
    ScalarPollEpisode built_episode0;
    ScalarPollEpisode fanin_episode0;
    InitializeScalarPollEpisode(&built_episode0);
    InitializeScalarPollEpisode(&fanin_episode0);
#if SIMT_CROSS_CORE_G0_TOKENS_PER_OWNER >= 2
    ScalarPollEpisode built_episode1;
    ScalarPollEpisode fanin_episode1;
    InitializeScalarPollEpisode(&built_episode1);
    InitializeScalarPollEpisode(&fanin_episode1);
#endif
#if SIMT_CROSS_CORE_G0_TOKENS_PER_OWNER >= 3
    ScalarPollEpisode built_episode2;
    ScalarPollEpisode fanin_episode2;
    InitializeScalarPollEpisode(&built_episode2);
    InitializeScalarPollEpisode(&fanin_episode2);
#endif
#if SIMT_CROSS_CORE_G0_TOKENS_PER_OWNER >= 4
    ScalarPollEpisode built_episode3;
    ScalarPollEpisode fanin_episode3;
    InitializeScalarPollEpisode(&built_episode3);
    InitializeScalarPollEpisode(&fanin_episode3);
#endif
#endif
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint32_t iterations = 0U;
    while (true) {
        for (uint32_t slot = 0U; slot < kTokensPerOwner && !exhausted; ++slot) {
            __gm__ ExecutionToken *token = &state->tokens[owner][slot];
            if (token->control.phase != ExecTokenPhase::Idle) {
                continue;
            }
            const uint32_t ticket = static_cast<uint32_t>(G0_TRACE_SCALAR_FETCH_ADD(
                trace, g0_swimlane::kTraceNoTask, g0_swimlane::AtomicSite::ScalarDispatchTicket,
                &cursor->value, 1U, true
            ));
            if (ticket >= task_count) {
                exhausted = true;
                ++result->exhausted_ticket_count;
                break;
            }
            const uint32_t task_id = LoadDispatchTaskId(task_ids, ticket G0_SCALAR_TRACE_ARGUMENT);
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
            TraceExecutorTicket(state, owner, task_id);
#endif
            token->control.phase = ExecTokenPhase::WaitingBuilt;
            token->control.task_id = task_id;
            token->control.build_owner = UINT32_MAX;
            token->control.execute_owner = owner;
            token->control.engine_class = engine;
            token->control.payload_address = reinterpret_cast<uint64_t>(&state->tasks[task_id].exec.payload);
            ++result->ticket_count;
            const uint32_t busy = BusyTokenCount(state, owner);
            result->max_busy_tokens = busy > result->max_busy_tokens ? busy : result->max_busy_tokens;
        }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
        (void)AdvanceToken(
            state, owner, &state->tokens[owner][0U], result,
            &built_episode0, &fanin_episode0, trace
        );
#if SIMT_CROSS_CORE_G0_TOKENS_PER_OWNER >= 2
        (void)AdvanceToken(
            state, owner, &state->tokens[owner][1U], result,
            &built_episode1, &fanin_episode1, trace
        );
#endif
#if SIMT_CROSS_CORE_G0_TOKENS_PER_OWNER >= 3
        (void)AdvanceToken(
            state, owner, &state->tokens[owner][2U], result,
            &built_episode2, &fanin_episode2, trace
        );
#endif
#if SIMT_CROSS_CORE_G0_TOKENS_PER_OWNER >= 4
        (void)AdvanceToken(
            state, owner, &state->tokens[owner][3U], result,
            &built_episode3, &fanin_episode3, trace
        );
#endif
#else
        for (uint32_t slot = 0U; slot < kTokensPerOwner; ++slot) {
            (void)AdvanceToken(
                state, owner, &state->tokens[owner][slot], result G0_SCALAR_TRACE_ARGUMENT
            );
        }
#endif
        if (exhausted && BusyTokenCount(state, owner) == 0U) {
            break;
        }
        ++iterations;
        if ((iterations & kWatchdogMask) == 0U) {
            if (TraceLoadFatal(state, trace) != 0U) {
                break;
            }
            if (static_cast<uint64_t>(get_sys_cnt()) - begin > state->control.timeout_ticks) {
                TracePublishFatal(state, ExecFatalReason::Timeout, owner, UINT32_MAX, trace);
                break;
            }
        }
    }
    if (TraceLoadFatal(state, trace) != 0U) {
        for (uint32_t slot = 0U; slot < kTokensPerOwner; ++slot) {
            ResetToken(&state->tokens[owner][slot]);
        }
    }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    ScalarTraceFlushPoll(trace, &built_episode0);
    ScalarTraceFlushPoll(trace, &fanin_episode0);
#if SIMT_CROSS_CORE_G0_TOKENS_PER_OWNER >= 2
    ScalarTraceFlushPoll(trace, &built_episode1);
    ScalarTraceFlushPoll(trace, &fanin_episode1);
#endif
#if SIMT_CROSS_CORE_G0_TOKENS_PER_OWNER >= 3
    ScalarTraceFlushPoll(trace, &built_episode2);
    ScalarTraceFlushPoll(trace, &fanin_episode2);
#endif
#if SIMT_CROSS_CORE_G0_TOKENS_PER_OWNER >= 4
    ScalarTraceFlushPoll(trace, &built_episode3);
    ScalarTraceFlushPoll(trace, &fanin_episode3);
#endif
#endif
    PublishTerminalTokenState(state, owner, result->ticket_count G0_SCALAR_TRACE_ARGUMENT);
    result->final_busy_tokens = BusyTokenCount(state, owner);
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    TraceRoleTimestamp(state, owner, 4U);
#endif
}

__aicore__ __attribute__((always_inline)) inline void
PublishRoleResult(__gm__ FullPaState *state, uint32_t owner, const FullPaRoleResult &result) {
    // CCEC may scalar-replace this 128-byte local object.  A generic pointer
    // walk over &result then reads compiler spill locations instead of the C++
    // field layout (the real-device symptom contains pieces of the GM base
    // address).  Pack every ABI word from named scalar fields instead.
    __gm__ uint64_t *destination = reinterpret_cast<__gm__ uint64_t *>(&state->roles[owner]);
    StoreDev64(destination + 0U, static_cast<uint64_t>(result.owner) | (static_cast<uint64_t>(result.role) << 32U));
    StoreDev64(
        destination + 1U,
        static_cast<uint64_t>(result.physical_block) | (static_cast<uint64_t>(result.drain_group) << 32U)
    );
    StoreDev64(
        destination + 2U,
        static_cast<uint64_t>(result.build_count) | (static_cast<uint64_t>(result.commit_count) << 32U)
    );
    StoreDev64(
        destination + 3U,
        static_cast<uint64_t>(result.execute_count) | (static_cast<uint64_t>(result.ticket_count) << 32U)
    );
    StoreDev64(
        destination + 4U,
        static_cast<uint64_t>(result.exhausted_ticket_count) | (static_cast<uint64_t>(result.claim_count) << 32U)
    );
    StoreDev64(
        destination + 5U,
        static_cast<uint64_t>(result.claim_lost_count) | (static_cast<uint64_t>(result.max_busy_tokens) << 32U)
    );
    StoreDev64(
        destination + 6U,
        static_cast<uint64_t>(result.final_busy_tokens) | (static_cast<uint64_t>(result.completed_by_kind[0]) << 32U)
    );
    StoreDev64(
        destination + 7U,
        static_cast<uint64_t>(result.completed_by_kind[1]) | (static_cast<uint64_t>(result.completed_by_kind[2]) << 32U)
    );
    StoreDev64(
        destination + 8U,
        static_cast<uint64_t>(result.completed_by_kind[3]) | (static_cast<uint64_t>(result.completed_by_kind[4]) << 32U)
    );
    StoreDev64(
        destination + 9U,
        static_cast<uint64_t>(result.drain_arrival_count) | (static_cast<uint64_t>(result.fatal_count) << 32U)
    );
    StoreDev64(destination + 10U, result.launch_nonce);
    StoreDev64(destination + 11U, result.reserved[0]);
    StoreDev64(destination + 12U, result.reserved[1]);
    StoreDev64(destination + 13U, result.reserved[2]);
    StoreDev64(destination + 14U, result.reserved[3]);
    StoreDev64(destination + 15U, result.reserved[4]);
    dsb(DSB_ALL);
}

__aicore__ __attribute__((always_inline)) inline void
InitializeRoleResult(FullPaRoleResult *result, uint32_t owner, uint32_t builder_count, uint64_t nonce) {
    result->owner = owner;
    result->role = OwnerRoleAt(owner, builder_count);
    result->physical_block = OwnerPhysicalBlock(owner);
    result->drain_group = OwnerDrainGroup(owner);
    result->build_count = 0U;
    result->commit_count = 0U;
    result->execute_count = 0U;
    result->ticket_count = 0U;
    result->exhausted_ticket_count = 0U;
    result->claim_count = 0U;
    result->claim_lost_count = 0U;
    result->max_busy_tokens = 0U;
    result->final_busy_tokens = 0U;
    result->completed_by_kind[0] = 0U;
    result->completed_by_kind[1] = 0U;
    result->completed_by_kind[2] = 0U;
    result->completed_by_kind[3] = 0U;
    result->completed_by_kind[4] = 0U;
    result->drain_arrival_count = 0U;
    result->fatal_count = 0U;
    result->launch_nonce = nonce;
    result->reserved[0] = 0U;
    result->reserved[1] = 0U;
    result->reserved[2] = 0U;
    result->reserved[3] = 0U;
    result->reserved[4] = 0U;
}

__aicore__ __attribute__((always_inline)) inline void
ArriveAndDrain(
    __gm__ FullPaState *state, uint32_t owner, FullPaRoleResult *result G0_SCALAR_TRACE_PARAMETER
) {
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    TraceRoleTimestamp(state, owner, 5U);
#endif
    result->drain_arrival_count = 1U;
    result->fatal_count = TraceLoadFatal(state, trace) == 0U ? 0U : 1U;
    PublishRoleResult(state, owner, *result);
    const int64_t contribution = EncodeDrainContribution(result->execute_count);
    (void)G0_TRACE_SCALAR_FETCH_ADD(
        trace, g0_swimlane::kTraceNoTask, g0_swimlane::AtomicSite::ScalarDrainArrive,
        &state->drain.arrivals[result->drain_group].value, static_cast<uint64_t>(contribution), false
    );
    if (owner != kBuilderOwner) {
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
        TraceRoleTimestamp(state, owner, 6U);
        TraceRoleTimestamp(state, owner, 7U);
#endif
        return;
    }

    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    ScalarPollEpisode drain_episode;
    InitializeScalarPollEpisode(&drain_episode);
#endif
    uint64_t completed = 0U;
    while (true) {
        bool all_arrived = true;
        completed = 0U;
        for (uint32_t group = 0U; group < kDrainGroupCount; ++group) {
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
            const int64_t raw = static_cast<int64_t>(ScalarTracePollLoad(
                trace, &drain_episode, g0_swimlane::kTraceNoTask,
                g0_swimlane::AtomicSite::ScalarDrainArrivalPoll,
                &state->drain.arrivals[group].value
            ));
#else
            const int64_t raw = static_cast<int64_t>(ScalarAtomicLoad(&state->drain.arrivals[group].value));
#endif
            all_arrived = all_arrived && DecodeDrainArrivals(raw) == kDrainExpectedArrivals;
            completed += DecodeDrainCompletions(raw);
        }
        if (all_arrived) {
            break;
        }
        if (static_cast<uint64_t>(get_sys_cnt()) - begin > state->control.timeout_ticks) {
            TracePublishFatal(state, ExecFatalReason::Timeout, owner, UINT32_MAX, trace);
            break;
        }
    }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    ScalarTraceFlushPoll(trace, &drain_episode);
#endif
    if (TraceLoadFatal(state, trace) == 0U) {
        const bool valid = completed == state->control.kernel_task_count &&
                           G0_TRACE_SCALAR_LOAD(
                               trace, g0_swimlane::kTraceNoTask,
                               g0_swimlane::AtomicSite::ScalarDrainVerifyLoad,
                               &state->drain.builder_started.value, true
                           ) == state->control.builder_count &&
                           G0_TRACE_SCALAR_LOAD(
                               trace, g0_swimlane::kTraceNoTask,
                               g0_swimlane::AtomicSite::ScalarDrainVerifyLoad,
                               &state->drain.builder_finished.value, true
                           ) == 1U &&
                           G0_TRACE_SCALAR_LOAD(
                               trace, g0_swimlane::kTraceNoTask,
                               g0_swimlane::AtomicSite::ScalarDrainVerifyLoad,
                               &state->exec_dispatch.aic_next.value, true
                           ) == state->exec_dispatch.aic_task_count + kAicOwnerCount &&
                           G0_TRACE_SCALAR_LOAD(
                               trace, g0_swimlane::kTraceNoTask,
                               g0_swimlane::AtomicSite::ScalarDrainVerifyLoad,
                               &state->exec_dispatch.aiv_next.value, true
                           ) ==
                               state->exec_dispatch.aiv_task_count + AivExecutorCount(state->control.builder_count);
        if (!valid) {
            TracePublishFatal(state, ExecFatalReason::DrainMismatch, owner, UINT32_MAX, trace);
        }
    }
    if (G0_TRACE_SCALAR_CAS(
            trace, g0_swimlane::kTraceNoTask, g0_swimlane::AtomicSite::ScalarRootFinishedPublish,
            &state->drain.root_finished.value, 0U, 1U, true
        ) != 0U) {
        TracePublishFatal(state, ExecFatalReason::DrainMismatch, owner, UINT32_MAX, trace);
    }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    TraceRoleTimestamp(state, owner, 6U);
    TraceRoleTimestamp(state, owner, 7U);
#endif
}

}  // namespace

#if defined(__DAV_VEC__)

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_cross_core_ordinary_g0_0_mix_aiv, 1, 2);

extern "C" __global__ __aicore__ void
simt_cross_core_ordinary_g0_0_mix_aiv(__gm__ pa_scheduler::simt_cross_core_ordinary::g0::FullPaState *state) {
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    const uint64_t startup_dcci_begin = static_cast<uint64_t>(get_sys_cnt());
#endif
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dcci(
        static_cast<__gm__ void *>(reinterpret_cast<__gm__ uint8_t *>(&state->control) + kCacheLineBytes),
        kSingleCacheLine
    );
    dcci(
        static_cast<__gm__ void *>(reinterpret_cast<__gm__ uint8_t *>(&state->exec_dispatch) + 2U * kCacheLineBytes),
        kSingleCacheLine
    );
    dsb(DSB_ALL);
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    const uint64_t startup_dcci_end = static_cast<uint64_t>(get_sys_cnt());
#endif
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    const uint32_t subblock_dim = static_cast<uint32_t>(get_subblockdim());
    const uint32_t subblock = static_cast<uint32_t>(get_subblockid());
    if (block >= kAicOwnerCount || subblock_dim != 2U || subblock >= subblock_dim) {
        PublishFatal(state, ExecFatalReason::InvalidBuildInput, kBuilderOwner, 0U);
        return;
    }
    const uint32_t aiv_id = block * subblock_dim + subblock;
    const uint32_t owner = kBuilderOwner + aiv_id;
    FullPaRoleResult result;
    InitializeRoleResult(&result, owner, state->control.builder_count, state->control.launch_nonce);
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    TraceRoleEnter(state, owner, result.role, result.physical_block, subblock, startup_dcci_begin);
    alignas(32U) ScalarTraceContext trace;
    AttachScalarTrace(state, owner, &trace);
    ScalarTraceDcciRecord(
        trace, g0_swimlane::kTraceNoTask, g0_swimlane::DcciSite::StartupConfigInvalidate,
        g0_swimlane::DcciOp::Invalidate, startup_dcci_begin, startup_dcci_end, 3U, 3U
    );
#endif
    if (!ConfigValid(state)) {
        TracePublishFatal(state, ExecFatalReason::InvalidBuildInput, owner, 0U, trace);
        ArriveAndDrain(state, owner, &result G0_SCALAR_TRACE_ARGUMENT);
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
        ScalarTraceFinish(trace);
#endif
        return;
    }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    TraceRoleTimestamp(state, owner, 2U);
#endif
    if (aiv_id >= state->control.builder_count) {
        RunExecutor(state, owner, &result G0_SCALAR_TRACE_ARGUMENT);
        ArriveAndDrain(state, owner, &result G0_SCALAR_TRACE_ARGUMENT);
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
        ScalarTraceFinish(trace);
#endif
        return;
    }
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    TraceRoleTimestamp(state, owner, 3U);
#endif
    result.reserved[0] = static_cast<uint64_t>(get_sys_cnt());
    cce::async_invoke<G0SimtBuildTasks>(
        cce::dim3{kBuilderThreadCount, 1U, 1U}, reinterpret_cast<__gm__ uint64_t *>(&state->tasks[0]),
        reinterpret_cast<__gm__ uint64_t *>(&state->heap),
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->drain.builder_started.value)),
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->drain.builder_finished.value)),
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->fatal.state)),
        reinterpret_cast<__gm__ uint64_t *>(&state->builder_threads[0]), state->control.launch_nonce,
        state->control.timeout_ticks, state->control.batch_count, state->control.task_count, aiv_id,
        state->control.builder_count, owner
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    result.reserved[1] = static_cast<uint64_t>(get_sys_cnt());
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    TraceRoleTimestamp(state, owner, 4U);
#endif
    ArriveAndDrain(state, owner, &result G0_SCALAR_TRACE_ARGUMENT);
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    ScalarTraceFinish(trace);
#endif
}

#else

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_cross_core_ordinary_g0_0_mix_aic, 1, 2);

extern "C" __global__ __aicore__ void
simt_cross_core_ordinary_g0_0_mix_aic(__gm__ pa_scheduler::simt_cross_core_ordinary::g0::FullPaState *state) {
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    const uint64_t startup_dcci_begin = static_cast<uint64_t>(get_sys_cnt());
#endif
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dcci(
        static_cast<__gm__ void *>(reinterpret_cast<__gm__ uint8_t *>(&state->control) + kCacheLineBytes),
        kSingleCacheLine
    );
    dcci(
        static_cast<__gm__ void *>(reinterpret_cast<__gm__ uint8_t *>(&state->exec_dispatch) + 2U * kCacheLineBytes),
        kSingleCacheLine
    );
    dsb(DSB_ALL);
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    const uint64_t startup_dcci_end = static_cast<uint64_t>(get_sys_cnt());
#endif
    const uint32_t owner = static_cast<uint32_t>(get_block_idx());
    FullPaRoleResult result;
    InitializeRoleResult(&result, owner, state->control.builder_count, state->control.launch_nonce);
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    TraceRoleEnter(
        state, owner, result.role, result.physical_block, static_cast<uint32_t>(get_subblockid()),
        startup_dcci_begin
    );
    alignas(32U) ScalarTraceContext trace;
    AttachScalarTrace(state, owner, &trace);
    ScalarTraceDcciRecord(
        trace, g0_swimlane::kTraceNoTask, g0_swimlane::DcciSite::StartupConfigInvalidate,
        g0_swimlane::DcciOp::Invalidate, startup_dcci_begin, startup_dcci_end, 3U, 3U
    );
#endif
    if (!ConfigValid(state)) {
        TracePublishFatal(state, ExecFatalReason::InvalidBuildInput, owner, 0U, trace);
    } else {
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
        TraceRoleTimestamp(state, owner, 2U);
#endif
        RunExecutor(state, owner, &result G0_SCALAR_TRACE_ARGUMENT);
    }
    ArriveAndDrain(state, owner, &result G0_SCALAR_TRACE_ARGUMENT);
#if defined(SIMT_CROSS_CORE_G0_SWIMLANE)
    ScalarTraceFinish(trace);
#endif
}

#endif
