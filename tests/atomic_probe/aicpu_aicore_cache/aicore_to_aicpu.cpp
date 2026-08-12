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
#include <pto/common/kernel_meta.hpp>

#include "aicore_helpers.h"

PTO_SYNCALL_AIV_KERNEL_META(aicore_to_aicpu_cache_probe_0_mix_aiv);

namespace {

using namespace aicpu_aicore_cache_probe;

__aicore__ void PublishPayloadLine(__gm__ PayloadLine *payload, AicorePayloadPublish publication) {
    __asm__ volatile("" ::: "memory");
    if (publication == AicorePayloadPublish::DcciDefault || publication == AicorePayloadPublish::DcciDefaultNoDsb) {
        dcci(payload, SINGLE_CACHE_LINE);
    } else if (publication == AicorePayloadPublish::DcciAll) {
        dcci(payload, SINGLE_CACHE_LINE, CACHELINE_ALL);
    } else if (publication == AicorePayloadPublish::DcciOut || publication == AicorePayloadPublish::DcciOutNoDsb) {
        dcci(payload, SINGLE_CACHE_LINE, CACHELINE_OUT);
    } else if (publication == AicorePayloadPublish::DcciAtomic) {
        dcci(payload, SINGLE_CACHE_LINE, CACHELINE_ATOMIC);
    }
}

__aicore__ void FinishPayloadPublish(AicorePayloadPublish publication) {
    if (publication != AicorePayloadPublish::DcciDefaultNoDsb && publication != AicorePayloadPublish::DcciOutNoDsb &&
        publication != AicorePayloadPublish::NoDcciNoDsb) {
        dsb(DSB_ALL);
    }
}

__aicore__ bool RequiresPostPublishQuietAck(AicorePayloadPublish publication, uint32_t round) {
    const bool no_dsb_dcci =
        publication == AicorePayloadPublish::DcciDefaultNoDsb || publication == AicorePayloadPublish::DcciOutNoDsb;
    return no_dsb_dcci && (round % kDirectQuietAckInterval) == 0U;
}

__aicore__ void DelayBeforePostPublishAck() {
    __asm__ volatile("" ::: "memory");
    for (uint32_t iteration = 0U; iteration < kDirectPostPublishNops; ++iteration) {
        asm volatile("nop");
    }
    __asm__ volatile("" ::: "memory");
}

__aicore__ int64_t PublishControl(
    __gm__ volatile int64_t *address, AicoreControlPublish publication, int64_t expected_old, int64_t desired
) {
    if (publication == AicoreControlPublish::AtomicExchange) {
        return atomicExch(const_cast<__gm__ int64_t *>(address), desired);
    }
    if (publication == AicoreControlPublish::AtomicAdd) {
        return atomicAdd(const_cast<__gm__ int64_t *>(address), desired - expected_old);
    }
    if (publication == AicoreControlPublish::AtomicMax) {
        return atomicMax(const_cast<__gm__ int64_t *>(address), desired);
    }
    if (publication == AicoreControlPublish::AtomicCas) {
        return atomicCAS(const_cast<__gm__ int64_t *>(address), expected_old, desired);
    }
    if (publication == AicoreControlPublish::StDev) {
        __builtin_cce_st_dev(desired, const_cast<__gm__ int64_t *>(address), 0);
        dsb(DSB_ALL);
        return expected_old;
    }

    *address = desired;
    __asm__ volatile("" ::: "memory");
    if (publication == AicoreControlPublish::OrdinaryDcciDefault) {
        dcci(const_cast<__gm__ int64_t *>(address), SINGLE_CACHE_LINE);
    }
    dsb(DSB_ALL);
    return expected_old;
}

}  // namespace

extern "C" __global__ __aicore__ void
aicore_to_aicpu_cache_probe_0_mix_aiv(__gm__ aicpu_aicore_cache_probe::ProbeState *state) {
    using namespace aicpu_aicore_cache_probe;
    InvalidateConfig(&state->config);
    const Direction direction = static_cast<Direction>(state->config.direction);
    const uint32_t case_count = state->config.case_count;
    const uint32_t rounds = state->config.rounds;
    const uint64_t nonce = state->config.nonce;
    const uint64_t timeout_ticks = state->config.aicore_timeout_ticks;
    if (state->config.magic != kProbeMagic || state->config.version != kProbeVersion ||
        direction != Direction::AicoreToAicpu || case_count != kAicoreToAicpuCaseCount || rounds != kRounds ||
        case_count > kMaxCases || timeout_ticks == 0U) {
        (void)atomicAdd(const_cast<__gm__ int64_t *>(&state->aicore_errors.value), int64_t{1});
        return;
    }

    if (!WaitAtomicExact(
            &state->aicpu_initialized.value, static_cast<int64_t>(InitToken(direction, nonce)), timeout_ticks
        )) {
        (void)atomicAdd(const_cast<__gm__ int64_t *>(&state->aicore_errors.value), int64_t{1});
        return;
    }

    uint64_t errors = 0U;
    for (uint32_t case_index = 0U; case_index < case_count; ++case_index) {
        const AicoreToAicpuCase spec = ResolveAicoreToAicpuCase(case_index);
        __gm__ CaseSlot *slot = &state->cases[case_index];
        for (uint32_t round = 0U; round < rounds; ++round) {
            const uint32_t generation = round + 1U;
            if (!WaitAtomicExact(&slot->ready.value, static_cast<int64_t>(generation), timeout_ticks)) {
                ++errors;
                (void)atomicAdd(
                    const_cast<__gm__ int64_t *>(&state->aicore_errors.value), static_cast<int64_t>(errors)
                );
                return;
            }

            for (uint32_t line = 0U; line < spec.payload_lines; ++line) {
                for (uint32_t word = 0U; word < kPayloadWords; ++word) {
                    slot->payload[line].words[word] = PayloadValue(direction, case_index, generation, word, line);
                }
            }
            for (uint32_t line = 0U; line < spec.payload_lines; ++line) {
                PublishPayloadLine(&slot->payload[line], spec.payload_publish);
            }
            FinishPayloadPublish(spec.payload_publish);

            const int64_t expected_old = ControlValue(direction, case_index, round);
            const int64_t expected_new = ControlValue(direction, case_index, generation);
            const int64_t returned =
                PublishControl(&slot->tested_control.value, spec.control_publish, expected_old, expected_new);
            if (returned != expected_old) ++errors;
            if (spec.doorbell_publish == AicoreDoorbellPublish::TestedControl) {
                if (RequiresPostPublishQuietAck(spec.payload_publish, round)) {
                    DelayBeforePostPublishAck();
                    const int64_t primary_ack = atomicAdd(const_cast<__gm__ int64_t *>(&slot->done.value), int64_t{0});
                    if (primary_ack != static_cast<int64_t>(generation)) ++errors;
                }
                continue;
            }
            dsb(DSB_ALL);
            const int64_t done_old =
                atomicExch(const_cast<__gm__ int64_t *>(&slot->done.value), static_cast<int64_t>(generation));
            if (done_old != static_cast<int64_t>(round)) ++errors;
        }
        if (spec.doorbell_publish == AicoreDoorbellPublish::TestedControl &&
            !WaitAtomicExact(&slot->ready.value, static_cast<int64_t>(rounds + 1U), timeout_ticks)) {
            ++errors;
            (void)atomicAdd(const_cast<__gm__ int64_t *>(&state->aicore_errors.value), static_cast<int64_t>(errors));
            return;
        }
    }
    if (errors != 0U) {
        (void)atomicAdd(const_cast<__gm__ int64_t *>(&state->aicore_errors.value), static_cast<int64_t>(errors));
    }
}
