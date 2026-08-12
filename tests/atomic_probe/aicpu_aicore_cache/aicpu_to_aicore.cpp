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

PTO_SYNCALL_AIV_KERNEL_META(aicpu_to_aicore_cache_probe_0_mix_aiv);

namespace {

using namespace aicpu_aicore_cache_probe;

__aicore__ int64_t ObserveControl(__gm__ volatile int64_t *address, AicoreControlObserve observer, int64_t desired) {
    if (observer == AicoreControlObserve::Ordinary) return *address;
    if (observer == AicoreControlObserve::LdDev) return LoadDevice(address);
    if (observer == AicoreControlObserve::AtomicAdd) {
        return atomicAdd(const_cast<__gm__ int64_t *>(address), int64_t{0});
    }
    if (observer == AicoreControlObserve::AtomicMax) {
        constexpr int64_t identity = (-9223372036854775807LL - 1LL);
        return atomicMax(const_cast<__gm__ int64_t *>(address), identity);
    }
    return atomicCAS(const_cast<__gm__ int64_t *>(address), desired, desired);
}

__aicore__ void
ReadPayloadPrimary(__gm__ PayloadLine *payload, AicorePayloadAcquire acquire, uint64_t values[kPayloadWords]) {
    InvalidatePayload(payload, acquire);
    if (acquire == AicorePayloadAcquire::LdDev) {
        ReadDevice(payload, values);
    } else {
        ReadOrdinary(payload, values);
    }
}

}  // namespace

extern "C" __global__ __aicore__ void
aicpu_to_aicore_cache_probe_0_mix_aiv(__gm__ aicpu_aicore_cache_probe::ProbeState *state) {
    using namespace aicpu_aicore_cache_probe;
    InvalidateConfig(&state->config);
    const Direction direction = static_cast<Direction>(state->config.direction);
    const uint32_t case_count = state->config.case_count;
    const uint32_t rounds = state->config.rounds;
    const uint64_t nonce = state->config.nonce;
    const uint64_t timeout_ticks = state->config.aicore_timeout_ticks;
    if (state->config.magic != kProbeMagic || state->config.version != kProbeVersion ||
        direction != Direction::AicpuToAicore || case_count != kAicpuToAicoreCaseCount || rounds != kRounds ||
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

    for (uint32_t case_index = 0U; case_index < case_count; ++case_index) {
        const AicpuToAicoreCase spec = ResolveAicpuToAicoreCase(case_index);
        __gm__ CaseSlot *slot = &state->cases[case_index];
        CaseResult result = NewResult();
        for (uint32_t round = 0U; round < rounds; ++round) {
            const uint32_t generation = round + 1U;
            const int64_t expected_new = ControlValue(direction, case_index, generation);
            const int64_t primed_control = slot->tested_control.value;
            uint64_t primed_payload[kPayloadWords]{};
            ReadOrdinary(&slot->payload[0], primed_payload);
            dsb(DSB_ALL);

            const int64_t ready_old =
                atomicExch(const_cast<__gm__ int64_t *>(&slot->ready.value), static_cast<int64_t>(generation));
            if (ready_old != static_cast<int64_t>(round)) {
                ++result.atomic_return_mismatches;
            }
            __gm__ volatile int64_t *doorbell = spec.doorbell_publish == AicpuDoorbellPublish::TestedControl ?
                                                    &slot->tested_control.value :
                                                    &slot->done.value;
            const int64_t expected_doorbell = spec.doorbell_publish == AicpuDoorbellPublish::TestedControl ?
                                                  expected_new :
                                                  static_cast<int64_t>(generation);
            WaitStats wait_stats{};
            if (!WaitAtomicExact(doorbell, expected_doorbell, timeout_ticks, &wait_stats)) {
                RecordWaitStats(result, wait_stats);
                result.status = static_cast<uint64_t>(Status::PeerTimeout);
                break;
            }
            RecordWaitStats(result, wait_stats);
            dsb(DSB_ALL);

            const int64_t control_primary =
                ObserveControl(&slot->tested_control.value, spec.control_observe, expected_new);
            const int64_t control_reference =
                atomicAdd(const_cast<__gm__ int64_t *>(&slot->tested_control.value), int64_t{0});
            CountValue(
                control_primary, expected_new, primed_control, result.control_primary_fresh,
                result.control_primary_stale, result.control_primary_other
            );
            CountValue(
                control_reference, expected_new, primed_control, result.control_reference_fresh,
                result.control_reference_stale, result.control_reference_other
            );

            uint64_t payload_primary[kPayloadWords]{};
            uint64_t payload_reference[kPayloadWords]{};
            ReadPayloadPrimary(&slot->payload[0], spec.payload_acquire, payload_primary);
            ReadDevice(&slot->payload[0], payload_reference);
            const LineClass primary_class =
                ClassifyLine(payload_primary, primed_payload, direction, case_index, generation);
            const LineClass reference_class =
                ClassifyLine(payload_reference, primed_payload, direction, case_index, generation);
            CountLine(
                primary_class, result.payload_primary_fresh, result.payload_primary_stale, result.payload_primary_other
            );
            CountLine(
                reference_class, result.payload_reference_fresh, result.payload_reference_stale,
                result.payload_reference_other
            );

            if (round == 0U) {
                result.first_control_primary = static_cast<uint64_t>(control_primary);
                result.first_control_reference = static_cast<uint64_t>(control_reference);
                result.first_payload_primary = primary_class.first;
                result.first_payload_reference = reference_class.first;
            }
            if (result.first_bad_round == kNoBadRound &&
                ((control_primary != expected_new && control_primary != primed_control) ||
                 (control_reference != expected_new && control_reference != primed_control) ||
                 (!primary_class.fresh && !primary_class.stale) ||
                 (!reference_class.fresh && !reference_class.stale))) {
                result.first_bad_round = round;
            }
            ++result.rounds_completed;
        }
        if (result.rounds_completed == rounds) {
            const int64_t ack_old =
                atomicExch(const_cast<__gm__ int64_t *>(&slot->ready.value), static_cast<int64_t>(rounds + 1U));
            if (ack_old != static_cast<int64_t>(rounds)) {
                ++result.atomic_return_mismatches;
            }
        }
        PublishResult(&slot->result, result);
    }
}
