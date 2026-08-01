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

#include "cce_aicore_intrinsics.h"
#include <pto/common/constants.hpp>
#include <pto/common/kernel_meta.hpp>

#define PA_DEVICE __aicore__ inline
#define PA_DEVICE_NOINLINE static __aicore__ __attribute__((noinline))
#define PA_LOOP_NOUNROLL _Pragma("clang loop unroll(disable)")
#define PA_GM __gm__
#include "../common/pa_scheduler_core.h"
#include "ccec_ops.h"
#include "shared_protocol_litmus_shared.h"

namespace {

using pa_scheduler_ccec::CcecOps;
using pa_scheduler::shared_protocol_litmus::Control;
using pa_scheduler::shared_protocol_litmus::Direction;
using pa_scheduler::shared_protocol_litmus::HistoryChain;
using pa_scheduler::shared_protocol_litmus::ReaderOrdering;
using pa_scheduler::shared_protocol_litmus::ReaderReclaimChain;
using pa_scheduler::shared_protocol_litmus::Scenario;
using pa_scheduler::shared_protocol_litmus::kAicToAiv;
using pa_scheduler::shared_protocol_litmus::kAivToAic;
using pa_scheduler::shared_protocol_litmus::kControlMagic;
using pa_scheduler::shared_protocol_litmus::kControlVersion;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimActiveWorkers;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimAddress;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimAicToAiv;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimAivToAic;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimClosedDone;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimHeapWindow;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimHi;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimInitialDone;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimLo;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimReaderStatus;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimReplacementAddress;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimReplacementHi;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimReplacementLo;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimReclaimerStatus;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimTask;
using pa_scheduler::shared_protocol_litmus::kReaderReclaimWriterTask;
using pa_scheduler::shared_protocol_litmus::kResultMagic;
using pa_scheduler::shared_protocol_litmus::kSymbolCount;

static_assert(
    offsetof(pa_scheduler::SharedWriterHistoryCell, entries) +
            6 * sizeof(pa_scheduler::SharedWriterHistoryRecord) ==
        64,
    "the seventh history record must begin on the second cache line"
);
static_assert(
    kReaderReclaimActiveWorkers == pa_scheduler::kWorkers,
    "reader-reclaim litmus must scan the complete active worker prefix"
);

__aicore__ inline pa_scheduler::FdwicOutputRef
HistoryOutputRef(const HistoryChain &chain, uint32_t slot) {
    return pa_scheduler::FdwicOutputRef{
        chain.producer,
        static_cast<int16_t>(slot),
        0, 0, 0, 0
    };
}

__aicore__ inline void BuildHistoryWriterArgs(
    const HistoryChain &chain, pa_scheduler::TaskArgs &args
) {
    pa_scheduler::ConstructTaskArgs(args);
    for (uint32_t slot = 0; slot < kSymbolCount; ++slot) {
        pa_scheduler::AppendSharedOutputRef(
            args, HistoryOutputRef(chain, slot),
            pa_scheduler::TensorArgType::Inout
        );
    }
}

__aicore__ inline bool PrepareHistoryWriter(
    __gm__ pa_scheduler::SchedulerState *state,
    const HistoryChain &chain, int32_t task_id,
    int32_t expected_predecessor,
    pa_scheduler::LocalStats &stats
) {
    pa_scheduler::TaskArgs args;
    BuildHistoryWriterArgs(chain, args);
    pa_scheduler::SubmitContext context{};
    context.task_id = task_id;
    context.won = true;
    const auto result =
        pa_scheduler::PrepareSharedWriterIntentSet<CcecOps>(
            state, args, context, stats
        );
    return result ==
               pa_scheduler::SharedWriterIntentResult::Published &&
           context.fanin_count == 1 &&
           context.fanin[0] == expected_predecessor &&
           CcecOps::Load(
               &state->tasks[
                    static_cast<uint32_t>(task_id)
                ].deps_prepared
           ) == task_id;
}

__aicore__ inline void PublishHistoryResult(
    __gm__ pa_scheduler::SchedulerState *state,
    uint32_t worker, uint64_t tag, uint64_t status,
    uint64_t quantity, int32_t fanin,
    uint64_t evidence0 = 0, uint64_t evidence1 = 0
) {
    __gm__ pa_scheduler::WorkerResult &result =
        state->results[worker];
    result.submit_begin = kResultMagic | tag;
    result.submit_end = status;
    result.finish_cycle = quantity;
    result.checksum = static_cast<uint64_t>(
        static_cast<int64_t>(fanin)
    );
    result.submits = evidence0;
    result.claim_attempts = evidence1;
    CcecOps::FlushRegion(&result, 64);
}

__aicore__ inline void CaptureReaderSnapshot(
    __gm__ pa_scheduler::SharedTensorMapSidecar &map,
    uint32_t bucket, uint64_t cursor,
    pa_scheduler::SharedRegionValue &snapshot,
    int64_t &seq_before, int64_t &seq_after
) {
    // 与 SharedReadRegionSlot 使用相同的 seq/失效/五字段拷贝/seq 顺序，
    // 但故意把全部判断延后到 reuse-done 之后。否则 CAS 的控制流会先消费
    // payload，三种 ordering 就无法形成有意义的差异。
    __gm__ pa_scheduler::SharedRegionSlot &slot =
        map.slots[
            pa_scheduler::SharedTensorMapSlotIndex(
                bucket, cursor
            )
        ];
    seq_before = CcecOps::Load(&slot.seq.value);
    CcecOps::InvalidateRegion(
        &slot.payload, sizeof(slot.payload)
    );
    snapshot.buffer_addr =
        slot.payload.value.buffer_addr;
    snapshot.lo = slot.payload.value.lo;
    snapshot.hi = slot.payload.value.hi;
    snapshot.producer =
        slot.payload.value.producer;
    snapshot.reserved =
        slot.payload.value.reserved;
    seq_after = CcecOps::Load(&slot.seq.value);
}

PA_DEVICE_NOINLINE bool ValidateReaderSnapshotAfterReuse(
    __gm__ pa_scheduler::SchedulerState *state,
    int32_t expected_signal,
    const pa_scheduler::SharedRegionValue &snapshot,
    int64_t seq_before, int64_t seq_after
) {
    if (state == nullptr || expected_signal < 0 ||
        expected_signal >= static_cast<int32_t>(
            pa_scheduler::kMaxTasks
        )) {
        return false;
    }
    // validator 自己对同一 reuse gate 做第二次真实 atomic load。O3 IR
    // 因而能在一个 noinline 函数内证明：先由 expected_signal 算出
    // TaskCell::deps_prepared，再比较动态 token，成功分支才读取旧快照。
    const int64_t reuse_token = CcecOps::Load(
        &state->tasks[
             static_cast<uint32_t>(expected_signal)
         ].deps_prepared
    );
    if (reuse_token != expected_signal) {
        return false;
    }
    // 使用 bool 位与让五个字段都在 token 成功块中实际读取，避免短路把
    // reserved 单独拆到另一个 CFG 分支后增加自动 IR 判定歧义。
    const bool seq_ok =
        (seq_before == 0) & (seq_after == 0);
    const bool payload_ok =
        (snapshot.buffer_addr == kReaderReclaimAddress) &
        (snapshot.lo == kReaderReclaimLo) &
        (snapshot.hi == kReaderReclaimHi) &
        (snapshot.producer == 0) &
        (snapshot.reserved == 0);
    return seq_ok & payload_ok;
}

PA_DEVICE_NOINLINE bool CloseReaderCompilerClobber(
    __gm__ pa_scheduler::SharedTensorMapSidecar &map,
    uint32_t worker
) {
    // 只约束编译器，不生成设备访存屏障；作为最弱动态对照，不能被解释为
    // ordinary payload 已在设备侧完成。
    __asm__ volatile("" ::: "memory");
    return pa_scheduler::SharedAdvanceReaderDone<CcecOps>(
        map, worker, kReaderReclaimTask
    );
}

PA_DEVICE_NOINLINE bool CloseReaderPayloadDependency(
    __gm__ pa_scheduler::SharedTensorMapSidecar &map,
    uint32_t worker, uint64_t buffer_addr,
    uint64_t lo, uint64_t hi, int32_t producer,
    uint32_t reserved
) {
    // 直接把五个 capture 标量作为 noinline 实参，而不是先在调用者中压成
    // 一个 checksum。最终 O3 IR 因而能沿函数签名审计三组 64b 和两组
    // 32b leaf 是否全部到达下面的 CAS dependency。
    uint32_t checksum = 2166136261U;
    checksum =
        (checksum ^ static_cast<uint32_t>(buffer_addr)) *
        16777619U;
    checksum =
        (checksum ^ static_cast<uint32_t>(
             buffer_addr >> 32
         )) *
        16777619U;
    checksum =
        (checksum ^ static_cast<uint32_t>(lo)) *
        16777619U;
    checksum =
        (checksum ^ static_cast<uint32_t>(lo >> 32)) *
        16777619U;
    checksum =
        (checksum ^ static_cast<uint32_t>(hi)) *
        16777619U;
    checksum =
        (checksum ^ static_cast<uint32_t>(hi >> 32)) *
        16777619U;
    checksum =
        (checksum ^ static_cast<uint32_t>(producer)) *
        16777619U;
    checksum =
        (checksum ^ reserved) * 16777619U;
    const uint32_t saved_checksum = checksum;
    uint32_t opaque_checksum = checksum;
    // MOV 精确保持运行时值，但 tied output 对优化器是不透明的新值。CAS
    // 的 expected/desired 都叠加该差值：正常时差值为零；若不为零，
    // expected 也会偏离当前 1，CAS 因而 fail-closed，不会写入错误 task。
    asm volatile(
        "MOV %0, %0"
        : "+l"(opaque_checksum)
        :
        : "memory"
    );
    const uint32_t dependency =
        opaque_checksum - saved_checksum;
    const int64_t expected =
        static_cast<int64_t>(kReaderReclaimInitialDone) +
        static_cast<int64_t>(dependency);
    const int64_t desired =
        static_cast<int64_t>(kReaderReclaimClosedDone) +
        static_cast<int64_t>(dependency);
    const int64_t observed = CcecOps::CompareExchange(
        &map.reader_done[worker].value, expected, desired
    );
    return dependency == 0 && observed == expected;
}

PA_DEVICE_NOINLINE bool CloseReaderDsbAll(
    __gm__ pa_scheduler::SharedTensorMapSidecar &map,
    uint32_t worker
) {
    // 本机 CANN 头把 mem_dsb_t(0) 定义为 DSB_ALL。两侧 compiler clobber
    // 防止 CCEC 把普通 payload load 穿过设备完成屏障。
    __asm__ volatile("" ::: "memory");
    dsb((mem_dsb_t)0);
    __asm__ volatile("" ::: "memory");
    return pa_scheduler::SharedAdvanceReaderDone<CcecOps>(
        map, worker, kReaderReclaimTask
    );
}

__aicore__ inline bool CloseReaderAfterSnapshot(
    __gm__ pa_scheduler::SharedTensorMapSidecar &map,
    uint32_t worker, ReaderOrdering ordering,
    const pa_scheduler::SharedRegionValue &snapshot
) {
    if (ordering == ReaderOrdering::CompilerClobber) {
        return CloseReaderCompilerClobber(map, worker);
    }
    if (ordering == ReaderOrdering::PayloadDependency) {
        return CloseReaderPayloadDependency(
            map, worker, snapshot.buffer_addr,
            snapshot.lo, snapshot.hi,
            snapshot.producer, snapshot.reserved
        );
    }
    if (ordering == ReaderOrdering::DsbAll) {
        return CloseReaderDsbAll(map, worker);
    }
    return false;
}

__aicore__ inline bool WaitForReaderClosed(
    __gm__ pa_scheduler::SchedulerState *state,
    uint32_t reader_worker, pa_scheduler::LocalStats &stats
) {
    const uint64_t begin = CcecOps::Now();
    uint32_t polls = 0;
    while (true) {
        const int64_t observed = CcecOps::Load(
            &state->shared_map.reader_done[reader_worker].value
        );
        if (observed == kReaderReclaimClosedDone) {
            return true;
        }
        if (observed != kReaderReclaimInitialDone) {
            pa_scheduler::SetFatal<CcecOps>(
                state, stats, kReaderReclaimTask
            );
            return false;
        }
        if (pa_scheduler::WatchdogExpired<CcecOps>(
                state, stats, begin, polls
            )) {
            return false;
        }
    }
}

__aicore__ inline void PublishReaderResult(
    __gm__ pa_scheduler::SchedulerState *state,
    const ReaderReclaimChain &chain, uint64_t status,
    const pa_scheduler::SharedRegionValue &snapshot,
    ReaderOrdering ordering, int64_t closed_done
) {
    __gm__ pa_scheduler::WorkerResult &result =
        state->results[chain.reader_worker];
    result.submit_begin =
        kResultMagic | chain.result_tag | 1U;
    result.submit_end = status;
    result.finish_cycle = snapshot.buffer_addr;
    result.checksum = snapshot.lo;
    result.submits = snapshot.hi;
    result.claim_attempts = static_cast<uint64_t>(
        static_cast<int64_t>(snapshot.producer)
    );
    result.claim_wins = snapshot.reserved;
    result.heap_guards =
        static_cast<uint64_t>(
            static_cast<uint32_t>(ordering)
        ) << 32 |
        static_cast<uint32_t>(closed_done);
    CcecOps::FlushRegion(&result, 64);
}

__aicore__ inline void PublishReclaimerResult(
    __gm__ pa_scheduler::SchedulerState *state,
    const ReaderReclaimChain &chain, uint64_t status,
    int64_t blocked_reclaim,
    pa_scheduler::SharedAppendCheck blocked_check,
    int64_t blocked_head, int64_t blocked_tail,
    int64_t allowed_reclaim,
    int64_t final_head, int64_t final_tail
) {
    __gm__ pa_scheduler::WorkerResult &result =
        state->results[chain.reclaimer_worker];
    result.submit_begin =
        kResultMagic | chain.result_tag | 2U;
    result.submit_end = status;
    result.finish_cycle =
        static_cast<uint64_t>(blocked_reclaim);
    result.checksum =
        static_cast<uint64_t>(blocked_check);
    result.submits = static_cast<uint64_t>(blocked_head);
    result.claim_attempts =
        static_cast<uint64_t>(blocked_tail);
    result.claim_wins =
        static_cast<uint64_t>(allowed_reclaim);
    result.heap_guards =
        static_cast<uint64_t>(
            static_cast<uint32_t>(final_head)
        ) << 32 |
        static_cast<uint32_t>(final_tail);
    CcecOps::FlushRegion(&result, 64);
}

__aicore__ inline void RunWriterB(
    __gm__ pa_scheduler::SchedulerState *state,
    const HistoryChain &chain
) {
    pa_scheduler::LocalStats stats{};
    const bool prepared = PrepareHistoryWriter(
        state, chain, chain.writer_b, chain.producer, stats
    );
    PublishHistoryResult(
        state, chain.writer_b_worker, chain.result_tag | 1U,
        prepared ? 1U : 0U,
        stats.result.shared_symbol_inout_commits,
        chain.producer
    );
}

__aicore__ inline void RunFutureWriters(
    __gm__ pa_scheduler::SchedulerState *state,
    const HistoryChain &chain
) {
    pa_scheduler::LocalStats stats{};
    const bool reader_ready =
        pa_scheduler::WaitForSharedWriterReady<CcecOps>(
            state, chain.reader_past_b_signal, stats
        );
    const bool d_prepared =
        reader_ready &&
        PrepareHistoryWriter(
            state, chain, chain.writer_d,
            chain.writer_b, stats
        );
    const bool e_prepared =
        d_prepared &&
        PrepareHistoryWriter(
            state, chain, chain.writer_e,
            chain.writer_d, stats
        );
    const bool signalled =
        e_prepared &&
        pa_scheduler::PublishSharedWriterReady<CcecOps>(
            state, chain.future_done_signal
        );
    if (!signalled) {
        pa_scheduler::SetFatal<CcecOps>(
            state, stats, chain.writer_e
        );
    }
    const uint64_t status =
        (reader_ready ? 1U : 0U) |
        (d_prepared ? 2U : 0U) |
        (e_prepared ? 4U : 0U) |
        (signalled ? 8U : 0U);
    PublishHistoryResult(
        state, chain.future_worker, chain.result_tag | 2U,
        status, stats.result.shared_symbol_inout_commits,
        chain.writer_d
    );
}

__aicore__ inline void RunSlowReader(
    __gm__ pa_scheduler::SchedulerState *state,
    const HistoryChain &chain
) {
    pa_scheduler::LocalStats stats{};
    const bool b_ready =
        pa_scheduler::WaitForSharedWriterReady<CcecOps>(
            state, chain.writer_b, stats
        );
    // C 通过 B gate 后，先把未来 D/E history 的 header 与第七条 record
    // 所在第二条 cache line 都以普通 GM load 预热成 host 初始化的零。
    // 预热值参与是否发布下一道 gate，保证编译器和 scalar 都必须先消费
    // 这些 load；随后 D/E 才能写回同一地址。
    __gm__ volatile uint64_t *future_d_words =
        reinterpret_cast<__gm__ volatile uint64_t *>(
            &state->shared_map.writer_history[
                static_cast<uint32_t>(chain.writer_d)
            ]
        );
    __gm__ volatile uint64_t *future_e_words =
        reinterpret_cast<__gm__ volatile uint64_t *>(
            &state->shared_map.writer_history[
                static_cast<uint32_t>(chain.writer_e)
            ]
        );
    uint64_t prewarm_first_line = UINT64_MAX;
    uint64_t prewarm_second_line = UINT64_MAX;
    if (b_ready) {
        prewarm_first_line =
            future_d_words[0] | future_e_words[0];
        prewarm_second_line =
            future_d_words[8] | future_e_words[8];
    }
    const bool prewarm_zero =
        b_ready && prewarm_first_line == 0 &&
        prewarm_second_line == 0;
    const bool passed_signal =
        prewarm_zero &&
        pa_scheduler::PublishSharedWriterReady<CcecOps>(
            state, chain.reader_past_b_signal
        );
    if (!passed_signal) {
        pa_scheduler::SetFatal<CcecOps>(
            state, stats, chain.reader_c
        );
    }
    const bool future_ready =
        passed_signal &&
        pa_scheduler::WaitForSharedWriterReady<CcecOps>(
            state, chain.future_done_signal, stats
        );

    pa_scheduler::TaskArgs args;
    pa_scheduler::ConstructTaskArgs(args);
    pa_scheduler::AppendSharedOutputRef(
        args, HistoryOutputRef(chain, kSymbolCount - 1),
        pa_scheduler::TensorArgType::Input
    );
    int32_t fanin[pa_scheduler::kMaxFanin] = {};
    bool protocol_ok = false;
    uint32_t ordinary_lookups = UINT32_MAX;
    uint32_t fanin_count = 0;
    if (future_ready) {
        fanin_count =
            pa_scheduler::CollectSharedFanin<
                CcecOps, false, true
            >(
                state->shared_map, args, chain.reader_c,
                static_cast<int32_t>(state->heap_window),
                stats, fanin, protocol_ok, ordinary_lookups,
                &state->fatal.value
            );
    }
    const bool resolved =
        future_ready && protocol_ok &&
        ordinary_lookups == 0 && fanin_count == 1 &&
        fanin[0] == chain.writer_b;
    if (!resolved) {
        pa_scheduler::SetFatal<CcecOps>(
            state, stats, chain.reader_c
        );
    }
    const uint64_t status =
        (b_ready ? 1U : 0U) |
        (prewarm_zero ? 2U : 0U) |
        (passed_signal ? 4U : 0U) |
        (future_ready ? 8U : 0U) |
        (protocol_ok ? 16U : 0U) |
        (resolved ? 32U : 0U);
    PublishHistoryResult(
        state, chain.reader_worker, chain.result_tag | 3U,
        status, fanin_count,
        fanin_count == 0 ? -1 : fanin[0],
        prewarm_first_line, prewarm_second_line
    );
}

__aicore__ inline void RunHistoryParticipant(
    __gm__ pa_scheduler::SchedulerState *state,
    __gm__ const Control *control, uint32_t worker
) {
    const HistoryChain *chain = nullptr;
    const Direction direction =
        static_cast<Direction>(control->direction);
    if (direction == Direction::AicToAiv) {
        chain = &kAicToAiv;
    } else if (direction == Direction::AivToAic) {
        chain = &kAivToAic;
    } else {
        pa_scheduler::LocalStats stats{};
        pa_scheduler::SetFatal<CcecOps>(state, stats, -1);
        return;
    }

    if (worker == chain->writer_b_worker) {
        RunWriterB(state, *chain);
    } else if (worker == chain->future_worker) {
        RunFutureWriters(state, *chain);
    } else if (worker == chain->reader_worker) {
        RunSlowReader(state, *chain);
    }
}

__aicore__ inline void RunReaderReclaimReader(
    __gm__ pa_scheduler::SchedulerState *state,
    const ReaderReclaimChain &chain,
    ReaderOrdering ordering
) {
    pa_scheduler::LocalStats stats{};
    const bool blocked_ready =
        pa_scheduler::WaitForSharedWriterReady<CcecOps>(
            state, chain.blocked_signal, stats
        );
    pa_scheduler::SharedRegionValue snapshot{};
    const uint32_t bucket =
        pa_scheduler::TensorMapHash(kReaderReclaimAddress);
    int64_t seq_before = -2;
    int64_t seq_after = -2;
    if (blocked_ready) {
        CaptureReaderSnapshot(
            state->shared_map, bucket, 0, snapshot,
            seq_before, seq_after
        );
    }
    const bool close_ok =
        blocked_ready &&
        CloseReaderAfterSnapshot(
            state->shared_map, chain.reader_worker,
            ordering, snapshot
        );
    const int64_t closed_done = CcecOps::Load(
        &state->shared_map
             .reader_done[chain.reader_worker].value
    );
    // CAS 发布后先等 reclaimer 完成 cursor-0 复用，再消费并导出旧快照；
    // 这给缺失 read->publish 顺序的对照留下确定性的覆盖窗口。
    const bool reuse_waited =
        close_ok &&
        pa_scheduler::WaitForSharedWriterReady<CcecOps>(
            state, chain.reuse_done_signal, stats
        );
    const bool reuse_done = reuse_waited;
    const bool snapshot_ok =
        reuse_done &&
        ValidateReaderSnapshotAfterReuse(
            state, chain.reuse_done_signal, snapshot,
            seq_before, seq_after
        );
    const uint64_t status =
        (blocked_ready ? 1U : 0U) |
        (close_ok ? 2U : 0U) |
        (closed_done == kReaderReclaimClosedDone
             ? 4U
             : 0U) |
        (reuse_done ? 8U : 0U) |
        (snapshot_ok ? 16U : 0U);
    if (status != kReaderReclaimReaderStatus) {
        pa_scheduler::SetFatal<CcecOps>(
            state, stats, kReaderReclaimTask
        );
    }
    PublishReaderResult(
        state, chain, status, snapshot, ordering,
        closed_done
    );
}

__aicore__ inline void RunReaderReclaimReclaimer(
    __gm__ pa_scheduler::SchedulerState *state,
    const ReaderReclaimChain &chain
) {
    pa_scheduler::LocalStats stats{};
    const uint32_t bucket =
        pa_scheduler::TensorMapHash(kReaderReclaimAddress);
    pa_scheduler::SharedRegionValue replacement{};
    replacement.buffer_addr =
        kReaderReclaimReplacementAddress;
    replacement.lo = kReaderReclaimReplacementLo;
    replacement.hi = kReaderReclaimReplacementHi;
    replacement.producer = kReaderReclaimWriterTask;
    replacement.reserved = 0;

    int64_t blocked_reclaim = -2;
    const bool blocked_refresh =
        pa_scheduler::SharedRefreshReaderReclaimForTask<
            CcecOps
        >(
            state->shared_map,
            kReaderReclaimWriterTask,
            kReaderReclaimActiveWorkers,
            kReaderReclaimHeapWindow,
            blocked_reclaim
        );
    const pa_scheduler::SharedAppendCheck blocked_check =
        blocked_refresh
            ? pa_scheduler::SharedCheckTaskAppend<CcecOps>(
                  state->shared_map, &replacement, 1,
                  blocked_reclaim
              )
            : pa_scheduler::SharedAppendCheck::ProtocolError;
    const int64_t blocked_head = CcecOps::Load(
        &state->shared_map.buckets[bucket].head.value
    );
    const int64_t blocked_tail = CcecOps::Load(
        &state->shared_map.buckets[bucket].tail.value
    );
    const int64_t blocked_commit = CcecOps::Load(
        &state->shared_map.committed_tasks.value
    );
    __gm__ pa_scheduler::SharedRegionSlot &first_slot =
        state->shared_map.slots[
            pa_scheduler::SharedTensorMapSlotIndex(bucket, 0)
        ];
    const int64_t blocked_seq =
        CcecOps::Load(&first_slot.seq.value);
    CcecOps::InvalidateRegion(
        &first_slot.payload, sizeof(first_slot.payload)
    );
    const uint64_t blocked_address =
        first_slot.payload.value.buffer_addr;
    const uint64_t blocked_lo =
        first_slot.payload.value.lo;
    const uint64_t blocked_hi =
        first_slot.payload.value.hi;
    const int32_t blocked_producer =
        first_slot.payload.value.producer;
    const uint32_t blocked_reserved =
        first_slot.payload.value.reserved;
    const bool blocked_ok =
        blocked_refresh && blocked_reclaim == -1 &&
        blocked_check ==
            pa_scheduler::SharedAppendCheck::CapacityBlocked &&
        blocked_head == 0 &&
        blocked_tail == pa_scheduler::kMapBucketCapacity &&
        blocked_commit == kReaderReclaimWriterTask &&
        blocked_seq == 0 &&
        blocked_address == kReaderReclaimAddress &&
        blocked_lo == kReaderReclaimLo &&
        blocked_hi == kReaderReclaimHi &&
        blocked_producer == 0 &&
        blocked_reserved == 0;
    if (!blocked_ok) {
        pa_scheduler::SetFatal<CcecOps>(
            state, stats, kReaderReclaimWriterTask
        );
    }

    // gate 只通知 reader“阻塞态已被完整取证”；随后 reclaimer 不再等待
    // 其他门值，而是直接轮询真实 reader_done，证明跨核前沿本身可见。
    const bool blocked_signalled =
        blocked_ok &&
        pa_scheduler::PublishSharedWriterReady<CcecOps>(
            state, chain.blocked_signal
        );
    const bool reader_observed =
        blocked_signalled &&
        WaitForReaderClosed(
            state, chain.reader_worker, stats
        );

    int64_t allowed_reclaim = -2;
    const bool allowed_refresh =
        reader_observed &&
        pa_scheduler::SharedRefreshReaderReclaimForTask<
            CcecOps
        >(
            state->shared_map,
            kReaderReclaimWriterTask,
            kReaderReclaimActiveWorkers,
            kReaderReclaimHeapWindow,
            allowed_reclaim
        );
    const pa_scheduler::SharedAppendCheck allowed_check =
        allowed_refresh
            ? pa_scheduler::SharedCheckTaskAppend<CcecOps>(
                  state->shared_map, &replacement, 1,
                  allowed_reclaim
              )
            : pa_scheduler::SharedAppendCheck::ProtocolError;
    const bool appended =
        allowed_check ==
            pa_scheduler::SharedAppendCheck::Ready &&
        pa_scheduler::SharedAppendPreparedTask<CcecOps>(
            state->shared_map, &replacement, 1
        );
    const bool committed =
        appended &&
        pa_scheduler::SharedPublishTaskCommit<CcecOps>(
            state->shared_map,
            kReaderReclaimWriterTask
        );
    // reuse-done 只能表示完整 append+commit 已返回成功。短路依赖避免失败
    // 路径先放行 reader，也让成功路径保留从 payload Flush/seq/tail 返回值，
    // 经 commit 返回值到 gate CAS 的控制链；不能把单独 gate 冒充成发布证据。
    const bool reuse_signalled =
        committed &&
        pa_scheduler::PublishSharedWriterReady<CcecOps>(
            state, chain.reuse_done_signal
        );
    const int64_t final_head = CcecOps::Load(
        &state->shared_map.buckets[bucket].head.value
    );
    const int64_t final_tail = CcecOps::Load(
        &state->shared_map.buckets[bucket].tail.value
    );
    const uint64_t status =
        (blocked_refresh ? 1U : 0U) |
        (blocked_ok ? 2U : 0U) |
        (blocked_signalled ? 4U : 0U) |
        (reader_observed ? 8U : 0U) |
        (allowed_refresh && allowed_reclaim == 0
             ? 16U
             : 0U) |
        (appended ? 32U : 0U) |
        (committed ? 64U : 0U) |
        (reuse_signalled ? 128U : 0U);
    if (status != kReaderReclaimReclaimerStatus) {
        pa_scheduler::SetFatal<CcecOps>(
            state, stats, kReaderReclaimWriterTask
        );
    }
    PublishReclaimerResult(
        state, chain, status, blocked_reclaim,
        blocked_check, blocked_head, blocked_tail,
        allowed_reclaim, final_head, final_tail
    );
}

__aicore__ inline void RunReaderReclaimParticipant(
    __gm__ pa_scheduler::SchedulerState *state,
    __gm__ const Control *control, uint32_t worker
) {
    const ReaderReclaimChain *chain = nullptr;
    const Direction direction =
        static_cast<Direction>(control->direction);
    if (direction == Direction::AicToAiv) {
        chain = &kReaderReclaimAicToAiv;
    } else if (direction == Direction::AivToAic) {
        chain = &kReaderReclaimAivToAic;
    } else {
        pa_scheduler::LocalStats stats{};
        pa_scheduler::SetFatal<CcecOps>(state, stats, -1);
        return;
    }
    const ReaderOrdering ordering =
        static_cast<ReaderOrdering>(
            control->reader_ordering
        );
    if (ordering != ReaderOrdering::CompilerClobber &&
        ordering != ReaderOrdering::PayloadDependency &&
        ordering != ReaderOrdering::DsbAll) {
        pa_scheduler::LocalStats stats{};
        pa_scheduler::SetFatal<CcecOps>(state, stats, -1);
        return;
    }
    if (worker == chain->reader_worker) {
        RunReaderReclaimReader(
            state, *chain, ordering
        );
    } else if (worker == chain->reclaimer_worker) {
        RunReaderReclaimReclaimer(state, *chain);
    }
}

__aicore__ inline void RunSharedProtocolParticipant(
    __gm__ pa_scheduler::SchedulerState *state,
    __gm__ const Control *control, uint32_t worker
) {
    CcecOps::InvalidateRegion(control, sizeof(Control));
    if (control->magic != kControlMagic ||
        control->version != kControlVersion) {
        pa_scheduler::LocalStats stats{};
        pa_scheduler::SetFatal<CcecOps>(state, stats, -1);
        return;
    }
    const Scenario scenario =
        static_cast<Scenario>(control->scenario);
    if (scenario == Scenario::SymbolHistory) {
        if (control->reader_ordering !=
            static_cast<uint32_t>(
                ReaderOrdering::NotApplicable
            )) {
            pa_scheduler::LocalStats stats{};
            pa_scheduler::SetFatal<CcecOps>(
                state, stats, -1
            );
            return;
        }
        RunHistoryParticipant(state, control, worker);
        return;
    }
    if (scenario == Scenario::ReaderReclaim) {
        RunReaderReclaimParticipant(
            state, control, worker
        );
        return;
    }
    pa_scheduler::LocalStats stats{};
    pa_scheduler::SetFatal<CcecOps>(state, stats, -1);
}

}  // namespace

#if defined(PA_BUILD_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(pa_scheduler_0_mix_aic, 1, 2);

extern "C" __global__ __aicore__ void
pa_scheduler_0_mix_aic(
    __gm__ pa_scheduler::SchedulerState *state,
    __gm__ const Control *control
) {
    RunSharedProtocolParticipant(
        state, control,
        static_cast<uint32_t>(get_block_idx())
    );
}
#elif defined(PA_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(pa_scheduler_0_mix_aiv, 1, 2);

extern "C" __global__ __aicore__ void
pa_scheduler_0_mix_aiv(
    __gm__ pa_scheduler::SchedulerState *state,
    __gm__ const Control *control
) {
    const uint32_t vector_id =
        static_cast<uint32_t>(
            get_block_idx() * get_subblockdim() +
            get_subblockid()
    );
    RunSharedProtocolParticipant(
        state, control,
        pa_scheduler::kAicWorkers + vector_id
    );
}
#else
#error "Compile with PA_BUILD_AIC or PA_BUILD_AIV"
#endif
