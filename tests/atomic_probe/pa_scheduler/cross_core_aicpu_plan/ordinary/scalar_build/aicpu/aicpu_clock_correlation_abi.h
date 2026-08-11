/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */

#ifndef PA_SCHEDULER_AICPU_CLOCK_CORRELATION_ABI_H
#define PA_SCHEDULER_AICPU_CLOCK_CORRELATION_ABI_H

#include <cstddef>
#include <cstdint>

namespace pa_scheduler::aicpu_clock {

// The calibration is a separate AICPU/AIV handshake.  It is never embedded in
// the measured Plan/Build kernel.  Four causal timestamps form one NTP-style
// offset interval:
//
//   AICPU send <= AIV receive <= AIV send <= AICPU receive
//   offset in [aiv_send - aicpu_receive, aiv_receive - aicpu_send]
//
// Both clocks have a one-nanosecond unit on the supported A5 build.  Equality
// of their epochs is deliberately not assumed; the interval intersection is
// the only authority used by the exporter.
constexpr uint64_t kExchangeMagic = UINT64_C(0x5041434c4b455843);
constexpr uint64_t kCompletionMagic = UINT64_C(0x5041434c4b444f4e);
constexpr uint64_t kEvidenceMagic = UINT64_C(0x5041434c4b455649);
constexpr uint32_t kVersion = 2U;
constexpr uint32_t kSamplesPerRound = 4U;
constexpr uint32_t kCaptureRounds = 2U;
constexpr uint32_t kEvidenceSampleCapacity =
    kSamplesPerRound * kCaptureRounds;
constexpr uint64_t kClockUnitsPerSecond = UINT64_C(1000000000);
constexpr uint64_t kDefaultHandshakeTimeoutNs = UINT64_C(100000000);
constexpr uint64_t kMaximumAlignmentErrorNs = UINT64_C(50000);
constexpr int64_t kAbortSequence = -1;

constexpr uint32_t kOwnerCommandRun = 1U;
constexpr uint32_t kOwnerCommandClockCorrelation = 2U;

enum class EndpointStatus : int32_t {
    Pending = 0,
    Ok = 1,
    BadArguments = -1,
    BadConfig = -2,
    ClockReadFailed = -3,
    PeerFailed = -4,
    Timeout = -5,
};

enum class EvidenceStatus : int32_t {
    NotApplicable = 0,
    Valid = 1,
    Incomplete = -1,
    EndpointFailed = -2,
    TimestampOrderInvalid = -3,
    OffsetIntersectionEmpty = -4,
    AlignmentErrorTooLarge = -5,
};

struct alignas(128) ExchangeConfigLine {
    uint64_t magic;
    uint32_t version;
    uint32_t sample_count;
    uint64_t timeout_ns;
    uint64_t round_nonce;
    uint64_t reserved[12];
};

struct alignas(128) SequenceLine {
    volatile int64_t sequence;
    uint64_t reserved[15];
};

// Written only by AICPU.  AIV never writes this cache line.
struct alignas(128) AicpuTimestampLine {
    uint64_t send_ns;
    uint64_t receive_ns;
    uint64_t reserved[14];
};

// Written only by AIV through device stores.  AICPU never writes this line.
struct alignas(128) AicoreTimestampLine {
    uint64_t receive_ticks;
    uint64_t send_ticks;
    uint64_t reserved[14];
};

struct alignas(128) CompletionLine {
    uint64_t magic;
    uint32_t version;
    int32_t status;
    uint32_t completed_samples;
    uint32_t reserved0;
    uint64_t reserved[13];
};

struct alignas(128) Exchange {
    ExchangeConfigLine config;
    SequenceLine request;
    SequenceLine response;
    CompletionLine aicpu_completion;
    CompletionLine aicore_completion;
    AicpuTimestampLine aicpu[kSamplesPerRound];
    AicoreTimestampLine aicore[kSamplesPerRound];
};

// This is the narrow Host -> raw-swimlane exporter interface.  All arithmetic
// is retained as integers so JSON publication never has to reconstruct a clock
// mapping from rounded floating-point values.
struct ClockCorrelationRawSample {
    uint64_t aicpu_send_ns;
    uint64_t aicore_receive_ticks;
    uint64_t aicore_send_ticks;
    uint64_t aicpu_receive_ns;
    int64_t offset_lower_ns;
    int64_t offset_upper_ns;
    uint64_t round_trip_ns;
    uint64_t aicore_service_ticks;
    uint32_t round;
    uint32_t index_in_round;
    uint64_t round_nonce;
};

struct ClockCorrelationEvidence {
    uint64_t magic;
    uint32_t version;
    int32_t status;
    uint32_t sample_count;
    uint32_t pre_samples;
    uint32_t post_samples;
    uint32_t reserved0;
    int64_t offset_lower_ns;
    int64_t offset_upper_ns;
    int64_t selected_offset_ns;
    uint64_t alignment_error_ns;
    uint64_t maximum_round_trip_ns;
    uint64_t maximum_aicore_service_ticks;
    // Deliberately no Host absolute timestamp is present.  A5 exposes Host,
    // AICPU CLOCK_MONOTONIC_RAW, and AICore SYS_CNT as three clock domains.
    // Run identity is proven only with device-domain causal envelopes: pre
    // samples precede Owner/FDWIC and post samples follow them.
    ClockCorrelationRawSample samples[kEvidenceSampleCapacity];
};

static_assert(sizeof(ExchangeConfigLine) == 128U,
              "clock correlation config must occupy one isolated line");
static_assert(sizeof(SequenceLine) == 128U,
              "clock correlation sequence must occupy one isolated line");
static_assert(sizeof(AicpuTimestampLine) == 128U,
              "AICPU timestamps must occupy one isolated line");
static_assert(sizeof(AicoreTimestampLine) == 128U,
              "AICore timestamps must occupy one isolated line");
static_assert(sizeof(CompletionLine) == 128U,
              "clock correlation completion must occupy one isolated line");
static_assert(sizeof(Exchange) == 1664U,
              "clock correlation exchange ABI changed");
static_assert(alignof(Exchange) == 128U,
              "clock correlation exchange alignment changed");
static_assert(offsetof(Exchange, request) == 128U,
              "clock correlation request offset changed");
static_assert(offsetof(Exchange, response) == 256U,
              "clock correlation response offset changed");
static_assert(offsetof(Exchange, aicpu) == 640U,
              "clock correlation AICPU timestamp offset changed");
static_assert(offsetof(Exchange, aicore) == 1152U,
              "clock correlation AICore timestamp offset changed");
static_assert(sizeof(ClockCorrelationRawSample) == 80U,
              "clock correlation raw sample ABI changed");
static_assert(sizeof(ClockCorrelationEvidence) == 720U,
              "clock correlation evidence ABI changed");

}  // namespace pa_scheduler::aicpu_clock

#endif  // PA_SCHEDULER_AICPU_CLOCK_CORRELATION_ABI_H
