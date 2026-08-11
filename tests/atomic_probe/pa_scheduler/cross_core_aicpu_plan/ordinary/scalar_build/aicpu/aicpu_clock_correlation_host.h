/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */

#ifndef PA_SCHEDULER_AICPU_CLOCK_CORRELATION_HOST_H
#define PA_SCHEDULER_AICPU_CLOCK_CORRELATION_HOST_H

#include "aicpu_clock_correlation_abi.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace pa_scheduler::aicpu_clock {

inline bool ClockDifference(uint64_t left, uint64_t right, int64_t *result)
{
    if (result == nullptr) return false;
    if (left >= right) {
        const uint64_t difference = left - right;
        if (difference > static_cast<uint64_t>(INT64_MAX)) return false;
        *result = static_cast<int64_t>(difference);
        return true;
    }
    const uint64_t difference = right - left;
    if (difference > static_cast<uint64_t>(INT64_MAX)) return false;
    *result = -static_cast<int64_t>(difference);
    return true;
}

inline uint64_t SignedIntervalWidth(int64_t lower, int64_t upper)
{
    if (upper < lower) return 0U;
    if (lower >= 0) {
        return static_cast<uint64_t>(upper - lower);
    }
    if (upper <= 0) {
        const uint64_t lower_magnitude =
            static_cast<uint64_t>(-(lower + 1)) + 1U;
        const uint64_t upper_magnitude =
            static_cast<uint64_t>(-(upper + 1)) + 1U;
        return lower_magnitude - upper_magnitude;
    }
    const uint64_t lower_magnitude =
        static_cast<uint64_t>(-(lower + 1)) + 1U;
    return lower_magnitude + static_cast<uint64_t>(upper);
}

inline int64_t SignedIntervalMidpoint(int64_t lower, int64_t upper)
{
    if (lower >= 0 || upper <= 0) {
        return lower + static_cast<int64_t>(
            SignedIntervalWidth(lower, upper) / 2U
        );
    }
    const uint64_t lower_magnitude =
        static_cast<uint64_t>(-(lower + 1)) + 1U;
    const uint64_t upper_magnitude = static_cast<uint64_t>(upper);
    if (upper_magnitude >= lower_magnitude) {
        return static_cast<int64_t>(
            (upper_magnitude - lower_magnitude) / 2U
        );
    }
    const uint64_t negative_difference =
        lower_magnitude - upper_magnitude;
    return -static_cast<int64_t>(
        negative_difference / 2U + negative_difference % 2U
    );
}

inline void InitializeClockCorrelationEvidence(
    ClockCorrelationEvidence *evidence
)
{
    if (evidence == nullptr) return;
    *evidence = ClockCorrelationEvidence{};
    evidence->magic = kEvidenceMagic;
    evidence->version = kVersion;
    evidence->status = static_cast<int32_t>(EvidenceStatus::Incomplete);
    evidence->offset_lower_ns = std::numeric_limits<int64_t>::min();
    evidence->offset_upper_ns = std::numeric_limits<int64_t>::max();
}

inline bool AppendClockCorrelationRound(
    const Exchange &exchange, uint32_t round,
    ClockCorrelationEvidence *evidence
)
{
    if (evidence == nullptr || round >= kCaptureRounds ||
        evidence->magic != kEvidenceMagic || evidence->version != kVersion ||
        evidence->sample_count != round * kSamplesPerRound ||
        evidence->pre_samples !=
            (round == 0U ? 0U : kSamplesPerRound) ||
        evidence->post_samples != 0U ||
        evidence->sample_count > kEvidenceSampleCapacity - kSamplesPerRound ||
        exchange.config.magic != kExchangeMagic ||
        exchange.config.version != kVersion ||
        exchange.config.sample_count != kSamplesPerRound ||
        exchange.config.round_nonce == 0U ||
        exchange.aicpu_completion.magic != kCompletionMagic ||
        exchange.aicpu_completion.version != kVersion ||
        exchange.aicore_completion.magic != kCompletionMagic ||
        exchange.aicore_completion.version != kVersion ||
        exchange.aicpu_completion.status !=
            static_cast<int32_t>(EndpointStatus::Ok) ||
        exchange.aicore_completion.status !=
            static_cast<int32_t>(EndpointStatus::Ok) ||
        exchange.aicpu_completion.completed_samples != kSamplesPerRound ||
        exchange.aicore_completion.completed_samples != kSamplesPerRound) {
        if (evidence != nullptr) {
            evidence->status =
                static_cast<int32_t>(EvidenceStatus::EndpointFailed);
        }
        return false;
    }
    if (round != 0U &&
        exchange.config.round_nonce == evidence->samples[0].round_nonce) {
        evidence->status =
            static_cast<int32_t>(EvidenceStatus::EndpointFailed);
        return false;
    }

    for (uint32_t sample = 0U; sample < kSamplesPerRound; ++sample) {
        ClockCorrelationRawSample &raw =
            evidence->samples[evidence->sample_count];
        raw.aicpu_send_ns = exchange.aicpu[sample].send_ns;
        raw.aicore_receive_ticks = exchange.aicore[sample].receive_ticks;
        raw.aicore_send_ticks = exchange.aicore[sample].send_ticks;
        raw.aicpu_receive_ns = exchange.aicpu[sample].receive_ns;
        raw.round = round;
        raw.index_in_round = sample;
        raw.round_nonce = exchange.config.round_nonce;

        if (raw.aicpu_send_ns == 0U || raw.aicore_receive_ticks == 0U ||
            raw.aicore_send_ticks < raw.aicore_receive_ticks ||
            raw.aicpu_receive_ns <= raw.aicpu_send_ns ||
            !ClockDifference(
                raw.aicore_send_ticks, raw.aicpu_receive_ns,
                &raw.offset_lower_ns
            ) ||
            !ClockDifference(
                raw.aicore_receive_ticks, raw.aicpu_send_ns,
                &raw.offset_upper_ns
            ) ||
            raw.offset_lower_ns > raw.offset_upper_ns) {
            evidence->status = static_cast<int32_t>(
                EvidenceStatus::TimestampOrderInvalid
            );
            return false;
        }
        if (evidence->sample_count != 0U) {
            const ClockCorrelationRawSample &previous =
                evidence->samples[evidence->sample_count - 1U];
            if (raw.aicpu_send_ns < previous.aicpu_receive_ns ||
                raw.aicore_receive_ticks < previous.aicore_send_ticks) {
                evidence->status = static_cast<int32_t>(
                    EvidenceStatus::TimestampOrderInvalid
                );
                return false;
            }
        }
        raw.round_trip_ns =
            raw.aicpu_receive_ns - raw.aicpu_send_ns;
        raw.aicore_service_ticks =
            raw.aicore_send_ticks - raw.aicore_receive_ticks;
        if (raw.aicore_service_ticks > raw.round_trip_ns) {
            evidence->status = static_cast<int32_t>(
                EvidenceStatus::TimestampOrderInvalid
            );
            return false;
        }

        evidence->offset_lower_ns = std::max(
            evidence->offset_lower_ns, raw.offset_lower_ns
        );
        evidence->offset_upper_ns = std::min(
            evidence->offset_upper_ns, raw.offset_upper_ns
        );
        evidence->maximum_round_trip_ns = std::max(
            evidence->maximum_round_trip_ns, raw.round_trip_ns
        );
        evidence->maximum_aicore_service_ticks = std::max(
            evidence->maximum_aicore_service_ticks,
            raw.aicore_service_ticks
        );
        ++evidence->sample_count;
        if (round == 0U) {
            ++evidence->pre_samples;
        } else {
            ++evidence->post_samples;
        }
    }

    if (evidence->offset_lower_ns > evidence->offset_upper_ns) {
        evidence->status = static_cast<int32_t>(
            EvidenceStatus::OffsetIntersectionEmpty
        );
        return false;
    }
    return true;
}

inline bool FinalizeClockCorrelationEvidence(
    ClockCorrelationEvidence *evidence
)
{
    if (evidence == nullptr || evidence->magic != kEvidenceMagic ||
        evidence->version != kVersion ||
        evidence->sample_count != kEvidenceSampleCapacity ||
        evidence->pre_samples != kSamplesPerRound ||
        evidence->post_samples != kSamplesPerRound ||
        evidence->offset_lower_ns > evidence->offset_upper_ns) {
        if (evidence != nullptr &&
            evidence->status ==
                static_cast<int32_t>(EvidenceStatus::Incomplete)) {
            evidence->status =
                static_cast<int32_t>(EvidenceStatus::Incomplete);
        }
        return false;
    }

    const uint64_t interval_width = SignedIntervalWidth(
        evidence->offset_lower_ns, evidence->offset_upper_ns
    );
    evidence->selected_offset_ns = SignedIntervalMidpoint(
        evidence->offset_lower_ns, evidence->offset_upper_ns
    );
    evidence->alignment_error_ns =
        interval_width / 2U + interval_width % 2U;
    if (evidence->alignment_error_ns > kMaximumAlignmentErrorNs) {
        evidence->status = static_cast<int32_t>(
            EvidenceStatus::AlignmentErrorTooLarge
        );
        return false;
    }
    evidence->status = static_cast<int32_t>(EvidenceStatus::Valid);
    return true;
}

inline bool AicpuOwnerIsWithinCorrelationEnvelope(
    const ClockCorrelationEvidence &evidence,
    uint64_t owner_begin_ns, uint64_t owner_end_ns
)
{
    if (evidence.magic != kEvidenceMagic || evidence.version != kVersion ||
        evidence.status != static_cast<int32_t>(EvidenceStatus::Valid) ||
        evidence.sample_count != kEvidenceSampleCapacity ||
        owner_begin_ns == 0U || owner_end_ns <= owner_begin_ns) {
        return false;
    }
    for (uint32_t sample = 0U; sample < kSamplesPerRound; ++sample) {
        // These comparisons are valid because both operands are produced by
        // AICPU CLOCK_MONOTONIC_RAW.  Host timestamps must never enter here.
        if (evidence.samples[sample].aicpu_receive_ns >= owner_begin_ns ||
            evidence.samples[sample + kSamplesPerRound].aicpu_send_ns <=
                owner_end_ns) {
            return false;
        }
    }
    return true;
}

inline bool ClockOffsetSatisfiesEvidence(
    const ClockCorrelationEvidence &evidence, int64_t candidate_offset_ns
)
{
    if (evidence.sample_count != kEvidenceSampleCapacity) return false;
    for (uint32_t sample = 0U; sample < evidence.sample_count; ++sample) {
        if (candidate_offset_ns < evidence.samples[sample].offset_lower_ns ||
            candidate_offset_ns > evidence.samples[sample].offset_upper_ns) {
            return false;
        }
    }
    return true;
}

}  // namespace pa_scheduler::aicpu_clock

#endif  // PA_SCHEDULER_AICPU_CLOCK_CORRELATION_HOST_H
