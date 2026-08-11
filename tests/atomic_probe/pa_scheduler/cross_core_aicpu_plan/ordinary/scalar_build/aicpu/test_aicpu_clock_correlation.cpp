/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */

#include "aicpu_clock_correlation_host.h"

#include <cassert>
#include <cstdint>

namespace {

using namespace pa_scheduler::aicpu_clock;

Exchange MakeRound(
    uint32_t round, int64_t offset, uint64_t outbound_delay,
    uint64_t inbound_delay, uint64_t service_ticks = 7U,
    uint64_t aicpu_base = UINT64_C(1000000000),
    uint64_t sample_stride = UINT64_C(10000)
)
{
    Exchange exchange{};
    exchange.config.magic = kExchangeMagic;
    exchange.config.version = kVersion;
    exchange.config.sample_count = kSamplesPerRound;
    exchange.config.timeout_ns = kDefaultHandshakeTimeoutNs;
    exchange.config.round_nonce = UINT64_C(0x1000) + round;
    exchange.aicpu_completion.magic = kCompletionMagic;
    exchange.aicpu_completion.version = kVersion;
    exchange.aicpu_completion.status =
        static_cast<int32_t>(EndpointStatus::Ok);
    exchange.aicpu_completion.completed_samples = kSamplesPerRound;
    exchange.aicore_completion = exchange.aicpu_completion;
    for (uint32_t sample = 0U; sample < kSamplesPerRound; ++sample) {
        const uint64_t aicpu_send =
            aicpu_base + round * UINT64_C(1000000) +
            sample * sample_stride;
        const uint64_t aicore_receive = static_cast<uint64_t>(
            static_cast<int64_t>(aicpu_send + outbound_delay) + offset
        );
        const uint64_t aicore_send = aicore_receive + service_ticks;
        const uint64_t aicpu_receive = static_cast<uint64_t>(
            static_cast<int64_t>(aicore_send + inbound_delay) - offset
        );
        exchange.aicpu[sample].send_ns = aicpu_send;
        exchange.aicpu[sample].receive_ns = aicpu_receive;
        exchange.aicore[sample].receive_ticks = aicore_receive;
        exchange.aicore[sample].send_ticks = aicore_send;
    }
    return exchange;
}

void TestDistinctEpochRequiresMappedOffset()
{
    constexpr int64_t kDistinctEpochOffset = INT64_C(-687744000000);
    constexpr uint64_t kAicpuBase = UINT64_C(1000000000000000);
    ClockCorrelationEvidence evidence{};
    InitializeClockCorrelationEvidence(&evidence);
    const Exchange pre = MakeRound(
        0U, kDistinctEpochOffset, 70U, 90U, 7U, kAicpuBase
    );
    const Exchange post = MakeRound(
        1U, kDistinctEpochOffset, 80U, 80U, 7U, kAicpuBase
    );
    assert(AppendClockCorrelationRound(pre, 0U, &evidence));
    assert(AppendClockCorrelationRound(post, 1U, &evidence));
    assert(FinalizeClockCorrelationEvidence(&evidence));
    assert(evidence.offset_lower_ns <= kDistinctEpochOffset);
    assert(evidence.offset_upper_ns >= kDistinctEpochOffset);
    assert(ClockOffsetSatisfiesEvidence(
        evidence, evidence.selected_offset_ns));
    // A direct same-epoch concatenation is exactly candidate offset=0.  Every
    // four-timestamp causal interval rejects it for this realistic-scale case.
    assert(!ClockOffsetSatisfiesEvidence(evidence, 0));
}

void TestValidIntersection()
{
    ClockCorrelationEvidence evidence{};
    InitializeClockCorrelationEvidence(&evidence);
    const Exchange pre = MakeRound(0U, 12345, 90U, 110U);
    const Exchange post = MakeRound(1U, 12345, 80U, 120U);
    assert(AppendClockCorrelationRound(pre, 0U, &evidence));
    assert(AppendClockCorrelationRound(post, 1U, &evidence));
    assert(FinalizeClockCorrelationEvidence(&evidence));
    assert(evidence.status == static_cast<int32_t>(EvidenceStatus::Valid));
    assert(evidence.sample_count == kEvidenceSampleCapacity);
    assert(evidence.offset_lower_ns <= 12345);
    assert(evidence.offset_upper_ns >= 12345);
    assert(evidence.alignment_error_ns <= kMaximumAlignmentErrorNs);
}

void TestDisjointRoundsFailClosed()
{
    ClockCorrelationEvidence evidence{};
    InitializeClockCorrelationEvidence(&evidence);
    const Exchange pre = MakeRound(0U, 1000, 10U, 10U);
    const Exchange post = MakeRound(1U, 2000, 10U, 10U);
    assert(AppendClockCorrelationRound(pre, 0U, &evidence));
    assert(!AppendClockCorrelationRound(post, 1U, &evidence));
    assert(evidence.status == static_cast<int32_t>(
        EvidenceStatus::OffsetIntersectionEmpty));
}

void TestTimestampOrderFailsClosed()
{
    ClockCorrelationEvidence evidence{};
    InitializeClockCorrelationEvidence(&evidence);
    Exchange exchange = MakeRound(0U, 123, 10U, 10U);
    exchange.aicore[2].send_ticks = exchange.aicore[2].receive_ticks - 1U;
    assert(!AppendClockCorrelationRound(exchange, 0U, &evidence));
    assert(evidence.status == static_cast<int32_t>(
        EvidenceStatus::TimestampOrderInvalid));
}

void TestHighErrorFailsClosed()
{
    ClockCorrelationEvidence evidence{};
    InitializeClockCorrelationEvidence(&evidence);
    const Exchange pre = MakeRound(
        0U, 5000, 100001U, 100001U, 7U,
        UINT64_C(1000000000), UINT64_C(250000)
    );
    const Exchange post = MakeRound(
        1U, 5000, 100001U, 100001U, 7U,
        UINT64_C(1000000000), UINT64_C(250000)
    );
    assert(AppendClockCorrelationRound(pre, 0U, &evidence));
    assert(AppendClockCorrelationRound(post, 1U, &evidence));
    assert(!FinalizeClockCorrelationEvidence(&evidence));
    assert(evidence.status == static_cast<int32_t>(
        EvidenceStatus::AlignmentErrorTooLarge));
}

void TestEndpointFailureFailsClosed()
{
    ClockCorrelationEvidence evidence{};
    InitializeClockCorrelationEvidence(&evidence);
    Exchange exchange = MakeRound(0U, 123, 10U, 10U);
    exchange.aicpu_completion.status =
        static_cast<int32_t>(EndpointStatus::Timeout);
    assert(!AppendClockCorrelationRound(exchange, 0U, &evidence));
    assert(evidence.status == static_cast<int32_t>(
        EvidenceStatus::EndpointFailed));
}

void TestRoundIdentityFailsClosed()
{
    ClockCorrelationEvidence out_of_order{};
    InitializeClockCorrelationEvidence(&out_of_order);
    const Exchange post = MakeRound(1U, 123, 10U, 10U);
    assert(!AppendClockCorrelationRound(post, 1U, &out_of_order));

    ClockCorrelationEvidence reused_nonce{};
    InitializeClockCorrelationEvidence(&reused_nonce);
    const Exchange pre = MakeRound(0U, 123, 10U, 10U);
    Exchange repeated = MakeRound(1U, 123, 10U, 10U);
    repeated.config.round_nonce = pre.config.round_nonce;
    assert(AppendClockCorrelationRound(pre, 0U, &reused_nonce));
    assert(!AppendClockCorrelationRound(repeated, 1U, &reused_nonce));
}

void TestOwnerMustBeInsideAicpuCausalEnvelope()
{
    constexpr uint64_t kA5ScaleAicpuBase = UINT64_C(676648500000000);
    constexpr uint64_t kObservedHostDomainShiftNs =
        UINT64_C(30787000000);
    ClockCorrelationEvidence evidence{};
    InitializeClockCorrelationEvidence(&evidence);
    const Exchange pre = MakeRound(
        0U, -123456, 40U, 60U, 7U, kA5ScaleAicpuBase
    );
    const Exchange post = MakeRound(
        1U, -123456, 45U, 55U, 7U, kA5ScaleAicpuBase
    );
    assert(AppendClockCorrelationRound(pre, 0U, &evidence));
    assert(AppendClockCorrelationRound(post, 1U, &evidence));
    assert(FinalizeClockCorrelationEvidence(&evidence));

    const uint64_t last_pre_receive =
        evidence.samples[kSamplesPerRound - 1U].aicpu_receive_ns;
    const uint64_t first_post_send =
        evidence.samples[kSamplesPerRound].aicpu_send_ns;
    const uint64_t owner_begin_ns = last_pre_receive + 100U;
    const uint64_t owner_end_ns = first_post_send - 100U;
    assert(AicpuOwnerIsWithinCorrelationEnvelope(
        evidence, owner_begin_ns, owner_end_ns));
    assert(!AicpuOwnerIsWithinCorrelationEnvelope(
        evidence, last_pre_receive, owner_end_ns));
    assert(!AicpuOwnerIsWithinCorrelationEnvelope(
        evidence, owner_begin_ns, first_post_send));

    // A5 showed that Host RAW can be tens of seconds away from AICPU RAW.
    // The valid device envelope remains authoritative and never consumes this
    // unrelated absolute Host value.
    const uint64_t unrelated_host_raw_ns =
        owner_begin_ns + kObservedHostDomainShiftNs;
    assert(unrelated_host_raw_ns > owner_end_ns);
    assert(AicpuOwnerIsWithinCorrelationEnvelope(
        evidence, owner_begin_ns, owner_end_ns));
}

void TestExtremeSignedMidpointDoesNotOverflow()
{
    assert(SignedIntervalWidth(INT64_MIN, INT64_MAX) == UINT64_MAX);
    assert(SignedIntervalMidpoint(INT64_MIN, INT64_MAX) == -1);
    assert(SignedIntervalWidth(INT64_MIN, 0) ==
           (UINT64_C(1) << 63U));
    assert(SignedIntervalMidpoint(INT64_MIN, 0) ==
           -(INT64_C(1) << 62U));
    assert(SignedIntervalMidpoint(-4, 1) == -2);
    assert(SignedIntervalMidpoint(-1, 4) == 1);
}

}  // namespace

int main()
{
    TestValidIntersection();
    TestDisjointRoundsFailClosed();
    TestTimestampOrderFailsClosed();
    TestHighErrorFailsClosed();
    TestEndpointFailureFailsClosed();
    TestRoundIdentityFailsClosed();
    TestOwnerMustBeInsideAicpuCausalEnvelope();
    TestExtremeSignedMidpointDoesNotOverflow();
    TestDistinctEpochRequiresMappedOffset();
    return 0;
}
