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
using pa_scheduler::shared_protocol_litmus::Scenario;
using pa_scheduler::shared_protocol_litmus::kAicToAiv;
using pa_scheduler::shared_protocol_litmus::kAivToAic;
using pa_scheduler::shared_protocol_litmus::kControlMagic;
using pa_scheduler::shared_protocol_litmus::kControlVersion;
using pa_scheduler::shared_protocol_litmus::kResultMagic;
using pa_scheduler::shared_protocol_litmus::kSymbolCount;

static_assert(
    offsetof(pa_scheduler::SharedWriterHistoryCell, entries) +
            6 * sizeof(pa_scheduler::SharedWriterHistoryRecord) ==
        64,
    "the seventh history record must begin on the second cache line"
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
        RunHistoryParticipant(state, control, worker);
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
