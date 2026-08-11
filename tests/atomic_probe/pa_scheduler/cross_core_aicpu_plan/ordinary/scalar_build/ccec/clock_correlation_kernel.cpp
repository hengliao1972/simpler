/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */

#include "cce_aicore_intrinsics.h"
#include <pto/common/kernel_meta.hpp>

#include "../aicpu/aicpu_clock_correlation_abi.h"
#define PA_CCEC_CLOCK_CORRELATION_OPS_ONLY 1
#include "ccec_ops.h"
#undef PA_CCEC_CLOCK_CORRELATION_OPS_ONLY

PTO_SYNCALL_AIV_KERNEL_META(pa_clock_correlation_0_mix_aiv);

namespace {

using namespace pa_scheduler::aicpu_clock;
using ClockOps = pa_scheduler_ccec::ClockCorrelationOps;

__aicore__ inline void InvalidateFirstCacheLine(__gm__ uint8_t *address)
{
    ClockOps::InvalidateFirstCacheLine(address);
}

__aicore__ inline int64_t ObserveSequence(__gm__ SequenceLine *line)
{
    InvalidateFirstCacheLine(reinterpret_cast<__gm__ uint8_t *>(line));
    return line->sequence;
}

__aicore__ inline void StoreWord(
    __gm__ void *address, uint32_t word, uint64_t value
)
{
    ClockOps::PublishWord(
        reinterpret_cast<__gm__ uint64_t *>(address) + word, value
    );
}

__aicore__ inline void PublishSequence(
    __gm__ SequenceLine *line, int64_t sequence
)
{
    StoreWord(line, 0U, static_cast<uint64_t>(sequence));
    ClockOps::PublishBarrier();
}

__aicore__ inline void PublishCompletion(
    __gm__ CompletionLine *line, EndpointStatus status,
    uint32_t completed_samples
)
{
    const uint64_t version_status =
        static_cast<uint64_t>(kVersion) |
        (static_cast<uint64_t>(static_cast<uint32_t>(status)) << 32U);
    StoreWord(line, 0U, kCompletionMagic);
    StoreWord(line, 1U, version_status);
    StoreWord(line, 2U, static_cast<uint64_t>(completed_samples));
    ClockOps::PublishBarrier();
}

__aicore__ inline void PublishFailure(
    __gm__ Exchange *exchange, EndpointStatus status,
    uint32_t completed_samples
)
{
    PublishSequence(&exchange->response, kAbortSequence);
    PublishCompletion(&exchange->aicore_completion, status, completed_samples);
}

}  // namespace

extern "C" __global__ __aicore__ void pa_clock_correlation_0_mix_aiv(
    __gm__ pa_scheduler::aicpu_clock::Exchange *exchange
)
{
    using namespace pa_scheduler::aicpu_clock;
    if (exchange == nullptr) return;

    // Config is 128-byte isolated.  The live fields are entirely within its
    // first 64-byte line; the second line is reserved and checked by Host.
    InvalidateFirstCacheLine(
        reinterpret_cast<__gm__ uint8_t *>(&exchange->config)
    );
    const uint64_t magic = exchange->config.magic;
    const uint32_t version = exchange->config.version;
    const uint32_t sample_count = exchange->config.sample_count;
    const uint64_t timeout_ticks = exchange->config.timeout_ns;
    const uint64_t round_nonce = exchange->config.round_nonce;
    if (magic != kExchangeMagic || version != kVersion ||
        sample_count != kSamplesPerRound || timeout_ticks == 0U ||
        round_nonce == 0U) {
        PublishFailure(exchange, EndpointStatus::BadConfig, 0U);
        return;
    }

    uint32_t completed = 0U;
    for (uint32_t sample = 0U; sample < sample_count; ++sample) {
        const int64_t expected = static_cast<int64_t>(sample + 1U);
        const uint64_t wait_begin = ClockOps::Now();
        int64_t request = 0;
        while (request != expected) {
            request = ObserveSequence(&exchange->request);
            if (request < 0) {
                PublishFailure(
                    exchange, EndpointStatus::PeerFailed, completed
                );
                return;
            }
            const uint64_t now = ClockOps::Now();
            if (now < wait_begin || now - wait_begin > timeout_ticks) {
                PublishFailure(exchange, EndpointStatus::Timeout, completed);
                return;
            }
        }

        const uint64_t receive_ticks = ClockOps::Now();
        const uint64_t send_ticks = ClockOps::Now();
        if (receive_ticks == 0U || send_ticks < receive_ticks) {
            PublishFailure(
                exchange, EndpointStatus::ClockReadFailed, completed
            );
            return;
        }
        StoreWord(&exchange->aicore[sample], 0U, receive_ticks);
        StoreWord(&exchange->aicore[sample], 1U, send_ticks);
        ClockOps::PublishBarrier();
        PublishSequence(&exchange->response, expected);
        ++completed;
    }

    PublishCompletion(
        &exchange->aicore_completion, EndpointStatus::Ok, completed
    );
}
