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
#ifndef ATOMIC_PROBE_AICPU_AICORE_CACHE_SHARED_H_
#define ATOMIC_PROBE_AICPU_AICORE_CACHE_SHARED_H_

#include <cstddef>
#include <cstdint>

namespace aicpu_aicore_cache_probe {

#if defined(__CCE_AICORE__)
#define CACHE_PROBE_HD __aicore__
#else
#define CACHE_PROBE_HD
#endif

constexpr uint64_t kProbeMagic = UINT64_C(0x4135435055434143);
constexpr uint64_t kResultMagic = UINT64_C(0x4341434845524553);
constexpr uint32_t kProbeVersion = 3U;
constexpr uint32_t kCacheLineBytes = 64U;
constexpr uint32_t kAtomicIsolationBytes = 128U;
constexpr uint32_t kPayloadWords = 8U;
constexpr uint32_t kPayloadLineCapacity = 32U;
constexpr uint32_t kRounds = 4096U;
constexpr uint32_t kMaxCases = 32U;
constexpr uint32_t kDirectQuietAckInterval = 64U;
constexpr uint32_t kDirectPostPublishNops = 1048576U;
constexpr uint64_t kNoBadRound = UINT64_MAX;

enum class Direction : uint32_t {
    AicpuToAicore = 0U,
    AicoreToAicpu = 1U,
};

enum class Expectation : uint32_t {
    Fresh = 0U,
    Stale = 1U,
    ObserveOnly = 2U,
};

enum class Status : uint64_t {
    Ok = 0U,
    BadConfig = 1U,
    InitTimeout = 2U,
    PeerTimeout = 3U,
    InternalError = 4U,
};

enum class AicpuControlPublish : uint32_t {
    OrdinaryCleanDsbIsb = 0U,
    OrdinaryCleanDsb = 1U,
    AtomicStoreRelease = 2U,
    AtomicExchangeSeqCst = 3U,
    AtomicFetchAddSeqCst = 4U,
    AtomicCompareExchangeSeqCst = 5U,
    OrdinaryNoClean = 6U,
    OrdinaryNoCleanDmbIshst = 7U,
    OrdinaryNoCleanDsbSy = 8U,
};

enum class AicpuPayloadPublish : uint32_t {
    OrdinaryClean = 0U,
    OrdinaryNoClean = 1U,
};

enum class AicpuDoorbellPublish : uint32_t {
    OrdinaryCleanDsbIsb = 0U,
    TestedControl = 1U,
    AtomicStoreRelease = 2U,
    AtomicStoreRelaxed = 3U,
    AtomicExchangeSeqCst = 4U,
    DmbIshstThenAtomicRelaxed = 5U,
    DsbSyThenAtomicRelaxed = 6U,
};

enum class AicoreControlObserve : uint32_t {
    Ordinary = 0U,
    LdDev = 1U,
    AtomicAdd = 2U,
    AtomicMax = 3U,
    AtomicCas = 4U,
};

enum class AicorePayloadAcquire : uint32_t {
    Ordinary = 0U,
    LdDev = 1U,
    DcciDefault = 2U,
    DcciAll = 3U,
    DcciOut = 4U,
    DcciAtomic = 5U,
};

enum class AicoreControlPublish : uint32_t {
    AtomicExchange = 0U,
    AtomicAdd = 1U,
    AtomicMax = 2U,
    AtomicCas = 3U,
    OrdinaryDcciDefault = 4U,
    OrdinaryNoDcci = 5U,
    StDev = 6U,
};

enum class AicorePayloadPublish : uint32_t {
    DcciDefault = 0U,
    DcciAll = 1U,
    DcciOut = 2U,
    DcciAtomic = 3U,
    NoDcci = 4U,
    DcciDefaultNoDsb = 5U,
    DcciOutNoDsb = 6U,
    NoDcciNoDsb = 7U,
};

enum class AicoreDoorbellPublish : uint32_t {
    IndependentDone = 0U,
    TestedControl = 1U,
};

enum class AicpuControlObserve : uint32_t {
    AtomicAcquire = 0U,
    AtomicRelaxed = 1U,
    Ordinary = 2U,
    CivacThenOrdinary = 3U,
};

enum class AicpuPayloadAcquire : uint32_t {
    Ordinary = 0U,
    CivacThenOrdinary = 1U,
};

struct AicpuToAicoreCase {
    AicpuControlPublish control_publish;
    AicpuPayloadPublish payload_publish;
    AicoreControlObserve control_observe;
    AicorePayloadAcquire payload_acquire;
    Expectation control_primary;
    Expectation control_reference;
    Expectation payload_primary;
    Expectation payload_reference;
    AicpuDoorbellPublish doorbell_publish = AicpuDoorbellPublish::OrdinaryCleanDsbIsb;
};

struct AicoreToAicpuCase {
    AicoreControlPublish control_publish;
    AicorePayloadPublish payload_publish;
    AicpuControlObserve control_observe;
    AicpuPayloadAcquire payload_acquire;
    Expectation control_primary;
    Expectation control_reference;
    Expectation payload_primary;
    Expectation payload_reference;
    AicoreDoorbellPublish doorbell_publish = AicoreDoorbellPublish::IndependentDone;
    uint32_t payload_lines = 1U;
};

constexpr uint32_t kAicpuToAicoreCaseCount = 32U;
constexpr uint32_t kAicoreToAicpuCaseCount = 20U;

CACHE_PROBE_HD inline AicpuToAicoreCase ResolveAicpuToAicoreCase(uint32_t index) {
    using CP = AicpuControlPublish;
    using PP = AicpuPayloadPublish;
    using CO = AicoreControlObserve;
    using PA = AicorePayloadAcquire;
    using E = Expectation;
    switch (index) {
    case 0U:
        return {CP::OrdinaryCleanDsbIsb,
                PP::OrdinaryClean,
                CO::AtomicAdd,
                PA::DcciDefault,
                E::Fresh,
                E::Fresh,
                E::Fresh,
                E::Fresh};
    case 1U:
        return {CP::OrdinaryCleanDsbIsb,
                PP::OrdinaryClean,
                CO::AtomicMax,
                PA::DcciDefault,
                E::Fresh,
                E::Fresh,
                E::Fresh,
                E::Fresh};
    case 2U:
        return {CP::OrdinaryCleanDsbIsb,
                PP::OrdinaryClean,
                CO::AtomicCas,
                PA::DcciDefault,
                E::Fresh,
                E::Fresh,
                E::Fresh,
                E::Fresh};
    case 3U:
        return {
            CP::OrdinaryCleanDsbIsb, PP::OrdinaryClean, CO::LdDev, PA::LdDev, E::Fresh, E::Fresh, E::Fresh, E::Fresh
        };
    case 4U:
        return {CP::OrdinaryCleanDsbIsb,
                PP::OrdinaryClean,
                CO::Ordinary,
                PA::DcciDefault,
                E::Stale,
                E::Fresh,
                E::Fresh,
                E::Fresh};
    case 5U:
        return {CP::OrdinaryCleanDsbIsb,
                PP::OrdinaryClean,
                CO::AtomicAdd,
                PA::Ordinary,
                E::Fresh,
                E::Fresh,
                E::Stale,
                E::Fresh};
    case 6U:
        return {CP::OrdinaryCleanDsbIsb,
                PP::OrdinaryClean,
                CO::AtomicAdd,
                PA::DcciAll,
                E::Fresh,
                E::Fresh,
                E::Fresh,
                E::Fresh};
    case 7U:
        return {CP::OrdinaryCleanDsbIsb,
                PP::OrdinaryClean,
                CO::AtomicAdd,
                PA::DcciOut,
                E::Fresh,
                E::Fresh,
                E::Fresh,
                E::Fresh};
    case 8U:
        return {CP::OrdinaryCleanDsbIsb,
                PP::OrdinaryClean,
                CO::AtomicAdd,
                PA::DcciAtomic,
                E::Fresh,
                E::Fresh,
                E::Fresh,
                E::Fresh};
    case 9U:
        return {CP::OrdinaryCleanDsb,
                PP::OrdinaryClean,
                CO::AtomicAdd,
                PA::DcciDefault,
                E::Fresh,
                E::Fresh,
                E::Fresh,
                E::Fresh};
    case 10U:
        return {CP::AtomicStoreRelease, PP::OrdinaryClean, CO::AtomicAdd, PA::DcciDefault,
                E::ObserveOnly,         E::ObserveOnly,    E::Fresh,      E::Fresh};
    case 11U:
        return {CP::AtomicExchangeSeqCst, PP::OrdinaryClean, CO::AtomicAdd, PA::DcciDefault,
                E::ObserveOnly,           E::ObserveOnly,    E::Fresh,      E::Fresh};
    case 12U:
        return {CP::OrdinaryNoClean, PP::OrdinaryClean, CO::AtomicAdd, PA::DcciDefault,
                E::ObserveOnly,      E::ObserveOnly,    E::Fresh,      E::Fresh};
    case 13U:
        return {CP::OrdinaryCleanDsbIsb, PP::OrdinaryNoClean, CO::AtomicAdd, PA::DcciDefault, E::Fresh, E::Fresh,
                E::ObserveOnly,          E::ObserveOnly};
    case 14U:
        return {CP::AtomicFetchAddSeqCst, PP::OrdinaryClean, CO::AtomicAdd, PA::DcciDefault,
                E::ObserveOnly,           E::ObserveOnly,    E::Fresh,      E::Fresh};
    case 15U:
        return {
            CP::AtomicCompareExchangeSeqCst,
            PP::OrdinaryClean,
            CO::AtomicAdd,
            PA::DcciDefault,
            E::ObserveOnly,
            E::ObserveOnly,
            E::Fresh,
            E::Fresh
        };
    case 16U:
        return {CP::OrdinaryNoClean, PP::OrdinaryClean, CO::LdDev, PA::DcciDefault,
                E::ObserveOnly,      E::ObserveOnly,    E::Fresh,  E::Fresh};
    case 17U:
        return {CP::OrdinaryCleanDsbIsb, PP::OrdinaryNoClean, CO::AtomicAdd, PA::LdDev, E::Fresh, E::Fresh,
                E::ObserveOnly,          E::ObserveOnly};
    case 18U:
        return {CP::OrdinaryNoClean, PP::OrdinaryNoClean, CO::AtomicAdd,
                PA::DcciDefault,     E::ObserveOnly,      E::ObserveOnly,
                E::ObserveOnly,      E::ObserveOnly,      AicpuDoorbellPublish::TestedControl};
    case 19U:
        return {
            CP::AtomicStoreRelease,
            PP::OrdinaryNoClean,
            CO::AtomicAdd,
            PA::DcciDefault,
            E::Fresh,
            E::Fresh,
            E::Fresh,
            E::Fresh,
            AicpuDoorbellPublish::TestedControl
        };
    case 20U:
        return {CP::AtomicExchangeSeqCst, PP::OrdinaryNoClean, CO::AtomicAdd,
                PA::DcciDefault,          E::ObserveOnly,      E::ObserveOnly,
                E::ObserveOnly,           E::ObserveOnly,      AicpuDoorbellPublish::TestedControl};
    case 21U:
        return {CP::AtomicFetchAddSeqCst, PP::OrdinaryNoClean, CO::AtomicAdd,
                PA::DcciDefault,          E::ObserveOnly,      E::ObserveOnly,
                E::ObserveOnly,           E::ObserveOnly,      AicpuDoorbellPublish::TestedControl};
    case 22U:
        return {
            CP::AtomicCompareExchangeSeqCst,
            PP::OrdinaryNoClean,
            CO::AtomicAdd,
            PA::DcciDefault,
            E::ObserveOnly,
            E::ObserveOnly,
            E::ObserveOnly,
            E::ObserveOnly,
            AicpuDoorbellPublish::TestedControl
        };
    case 23U:
        return {
            CP::OrdinaryNoClean,
            PP::OrdinaryNoClean,
            CO::AtomicAdd,
            PA::DcciDefault,
            E::Fresh,
            E::Fresh,
            E::Fresh,
            E::Fresh,
            AicpuDoorbellPublish::AtomicStoreRelease
        };
    case 24U:
        return {CP::OrdinaryNoClean, PP::OrdinaryNoClean, CO::AtomicAdd,
                PA::DcciDefault,     E::ObserveOnly,      E::ObserveOnly,
                E::ObserveOnly,      E::ObserveOnly,      AicpuDoorbellPublish::AtomicStoreRelaxed};
    case 25U:
        return {CP::OrdinaryNoClean, PP::OrdinaryNoClean, CO::AtomicAdd,
                PA::DcciDefault,     E::ObserveOnly,      E::ObserveOnly,
                E::ObserveOnly,      E::ObserveOnly,      AicpuDoorbellPublish::DmbIshstThenAtomicRelaxed};
    case 26U:
        return {CP::OrdinaryNoClean, PP::OrdinaryNoClean, CO::AtomicAdd,
                PA::DcciDefault,     E::ObserveOnly,      E::ObserveOnly,
                E::ObserveOnly,      E::ObserveOnly,      AicpuDoorbellPublish::DsbSyThenAtomicRelaxed};
    case 27U:
        return {
            CP::AtomicStoreRelease,
            PP::OrdinaryNoClean,
            CO::AtomicAdd,
            PA::Ordinary,
            E::Fresh,
            E::Fresh,
            E::Stale,
            E::Fresh,
            AicpuDoorbellPublish::TestedControl
        };
    case 28U:
        return {
            CP::AtomicStoreRelease,
            PP::OrdinaryNoClean,
            CO::AtomicAdd,
            PA::LdDev,
            E::Fresh,
            E::Fresh,
            E::Fresh,
            E::Fresh,
            AicpuDoorbellPublish::TestedControl
        };
    case 29U:
        return {
            CP::OrdinaryNoClean,
            PP::OrdinaryNoClean,
            CO::Ordinary,
            PA::DcciDefault,
            E::Stale,
            E::Fresh,
            E::Fresh,
            E::Fresh,
            AicpuDoorbellPublish::AtomicStoreRelease
        };
    case 30U:
        return {
            CP::OrdinaryNoCleanDmbIshst,
            PP::OrdinaryNoClean,
            CO::AtomicAdd,
            PA::DcciDefault,
            E::ObserveOnly,
            E::ObserveOnly,
            E::ObserveOnly,
            E::ObserveOnly,
            AicpuDoorbellPublish::TestedControl
        };
    case 31U:
        return {CP::OrdinaryNoCleanDsbSy, PP::OrdinaryNoClean, CO::AtomicAdd,
                PA::DcciDefault,          E::ObserveOnly,      E::ObserveOnly,
                E::ObserveOnly,           E::ObserveOnly,      AicpuDoorbellPublish::TestedControl};
    default:
        return {CP::OrdinaryNoClean, PP::OrdinaryNoClean, CO::Ordinary,   PA::Ordinary,
                E::ObserveOnly,      E::ObserveOnly,      E::ObserveOnly, E::ObserveOnly};
    }
}

CACHE_PROBE_HD inline AicoreToAicpuCase ResolveAicoreToAicpuCase(uint32_t index) {
    using CP = AicoreControlPublish;
    using PP = AicorePayloadPublish;
    using CO = AicpuControlObserve;
    using PA = AicpuPayloadAcquire;
    using E = Expectation;
    switch (index) {
    case 0U:
        return {CP::AtomicExchange, PP::DcciDefault, CO::AtomicAcquire, PA::Ordinary,
                E::Fresh,           E::Fresh,        E::Fresh,          E::Fresh};
    case 1U:
        return {CP::AtomicAdd, PP::DcciDefault, CO::AtomicAcquire, PA::Ordinary,
                E::Fresh,      E::Fresh,        E::Fresh,          E::Fresh};
    case 2U:
        return {CP::AtomicMax, PP::DcciDefault, CO::AtomicAcquire, PA::Ordinary,
                E::Fresh,      E::Fresh,        E::Fresh,          E::Fresh};
    case 3U:
        return {CP::AtomicCas, PP::DcciDefault, CO::AtomicAcquire, PA::Ordinary,
                E::Fresh,      E::Fresh,        E::Fresh,          E::Fresh};
    case 4U:
        return {CP::OrdinaryDcciDefault,
                PP::DcciDefault,
                CO::AtomicAcquire,
                PA::Ordinary,
                E::Fresh,
                E::Fresh,
                E::Fresh,
                E::Fresh};
    case 5U:
        return {CP::AtomicExchange, PP::DcciAll, CO::AtomicAcquire, PA::Ordinary,
                E::Fresh,           E::Fresh,    E::Fresh,          E::Fresh};
    case 6U:
        return {CP::AtomicExchange, PP::DcciOut, CO::AtomicAcquire, PA::Ordinary,
                E::Fresh,           E::Fresh,    E::Fresh,          E::Fresh};
    case 7U:
        return {CP::AtomicExchange, PP::DcciAtomic, CO::AtomicAcquire, PA::Ordinary,
                E::Fresh,           E::Fresh,       E::Fresh,          E::Fresh};
    case 8U:
        return {CP::AtomicExchange, PP::DcciDefault, CO::Ordinary, PA::Ordinary,
                E::Fresh,           E::Fresh,        E::Fresh,     E::Fresh};
    case 9U:
        return {CP::AtomicExchange, PP::DcciDefault, CO::AtomicRelaxed, PA::Ordinary,
                E::Fresh,           E::Fresh,        E::Fresh,          E::Fresh};
    case 10U:
        return {CP::AtomicExchange, PP::DcciDefault, CO::CivacThenOrdinary, PA::CivacThenOrdinary, E::Fresh, E::Fresh,
                E::Fresh,           E::Fresh};
    case 11U:
        return {CP::AtomicExchange, PP::NoDcci, CO::AtomicAcquire, PA::Ordinary,
                E::Fresh,           E::Fresh,   E::ObserveOnly,    E::ObserveOnly};
    case 12U:
        return {CP::OrdinaryNoDcci, PP::DcciDefault, CO::AtomicAcquire, PA::Ordinary,
                E::ObserveOnly,     E::ObserveOnly,  E::Fresh,          E::Fresh};
    case 13U:
        return {CP::StDev,      PP::DcciDefault, CO::AtomicAcquire, PA::Ordinary,
                E::ObserveOnly, E::ObserveOnly,  E::Fresh,          E::Fresh};
    case 14U:
        return {
            CP::AtomicExchange,
            PP::DcciDefault,
            CO::AtomicAcquire,
            PA::Ordinary,
            E::Fresh,
            E::Fresh,
            E::Fresh,
            E::Fresh,
            AicoreDoorbellPublish::TestedControl,
            kPayloadLineCapacity
        };
    case 15U:
        return {
            CP::AtomicExchange,
            PP::DcciDefaultNoDsb,
            CO::AtomicAcquire,
            PA::Ordinary,
            E::Fresh,
            E::Fresh,
            E::Fresh,
            E::Fresh,
            AicoreDoorbellPublish::TestedControl,
            kPayloadLineCapacity
        };
    case 16U:
        return {
            CP::AtomicExchange,
            PP::DcciOut,
            CO::AtomicAcquire,
            PA::Ordinary,
            E::Fresh,
            E::Fresh,
            E::Fresh,
            E::Fresh,
            AicoreDoorbellPublish::TestedControl,
            kPayloadLineCapacity
        };
    case 17U:
        return {
            CP::AtomicExchange,
            PP::DcciOutNoDsb,
            CO::AtomicAcquire,
            PA::Ordinary,
            E::Fresh,
            E::Fresh,
            E::Fresh,
            E::Fresh,
            AicoreDoorbellPublish::TestedControl,
            kPayloadLineCapacity
        };
    case 18U:
        return {
            CP::AtomicExchange,
            PP::NoDcci,
            CO::AtomicAcquire,
            PA::Ordinary,
            E::Fresh,
            E::Fresh,
            E::Stale,
            E::Stale,
            AicoreDoorbellPublish::TestedControl,
            kPayloadLineCapacity
        };
    case 19U:
        return {
            CP::AtomicExchange,
            PP::NoDcciNoDsb,
            CO::AtomicAcquire,
            PA::Ordinary,
            E::Fresh,
            E::Fresh,
            E::Stale,
            E::Stale,
            AicoreDoorbellPublish::TestedControl,
            kPayloadLineCapacity
        };
    default:
        return {CP::OrdinaryNoDcci, PP::NoDcci,     CO::Ordinary,   PA::Ordinary,
                E::ObserveOnly,     E::ObserveOnly, E::ObserveOnly, E::ObserveOnly};
    }
}

struct alignas(kAtomicIsolationBytes) ProbeConfig {
    uint64_t magic;
    uint32_t version;
    uint32_t direction;
    uint32_t case_count;
    uint32_t rounds;
    uint64_t nonce;
    uint64_t aicore_timeout_ticks;
    uint64_t aicpu_timeout_ns;
    uint8_t reserved[80];
};

struct alignas(kAtomicIsolationBytes) AtomicLine {
    volatile int64_t value;
    uint8_t isolation_padding[kAtomicIsolationBytes - sizeof(int64_t)];
};

struct alignas(kAtomicIsolationBytes) PayloadLine {
    volatile uint64_t words[kPayloadWords];
    uint8_t isolation_padding[kAtomicIsolationBytes - kPayloadWords * sizeof(uint64_t)];
};

struct alignas(kAtomicIsolationBytes) CaseResult {
    uint64_t magic;
    uint64_t status;
    uint64_t rounds_completed;
    uint64_t control_primary_fresh;
    uint64_t control_primary_stale;
    uint64_t control_primary_other;
    uint64_t control_reference_fresh;
    uint64_t control_reference_stale;
    uint64_t control_reference_other;
    uint64_t payload_primary_fresh;
    uint64_t payload_primary_stale;
    uint64_t payload_primary_other;
    uint64_t payload_reference_fresh;
    uint64_t payload_reference_stale;
    uint64_t payload_reference_other;
    uint64_t first_bad_round;
    uint64_t first_control_primary;
    uint64_t first_control_reference;
    uint64_t first_payload_primary;
    uint64_t first_payload_reference;
    uint64_t atomic_return_mismatches;
    uint64_t flags;
    uint64_t doorbell_poll_attempts_total;
    uint64_t doorbell_poll_attempts_max;
    uint64_t doorbell_wait_ticks_total;
    uint64_t doorbell_wait_ticks_max;
    uint64_t reserved[6];
};

struct alignas(kAtomicIsolationBytes) CaseSlot {
    AtomicLine ready;
    AtomicLine tested_control;
    AtomicLine done;
    PayloadLine payload[kPayloadLineCapacity];
    CaseResult result;
};

struct alignas(kAtomicIsolationBytes) ProbeState {
    ProbeConfig config;
    AtomicLine aicpu_initialized;
    AtomicLine aicpu_errors;
    AtomicLine aicore_errors;
    CaseSlot cases[kMaxCases];
};

static_assert(sizeof(ProbeConfig) == 128U, "config must occupy two cache lines");
static_assert(sizeof(AtomicLine) == 128U, "atomic control must be isolated by 128 bytes");
static_assert(sizeof(PayloadLine) == 128U, "payload must own two cache lines");
static_assert(sizeof(CaseResult) == 256U, "result layout changed");
static_assert(offsetof(CaseSlot, tested_control) == 128U, "tested control is not isolated");
static_assert(offsetof(CaseSlot, done) == 256U, "done control is not isolated");
static_assert(offsetof(CaseSlot, payload) == 384U, "payload is not isolated");
static_assert(
    offsetof(CaseSlot, result) == 384U + kPayloadLineCapacity * sizeof(PayloadLine), "result is not isolated"
);
static_assert(sizeof(CaseSlot) == 640U + kPayloadLineCapacity * sizeof(PayloadLine), "case stride changed");
static_assert(offsetof(ProbeState, cases) == 512U, "case base changed");
static_assert(sizeof(ProbeState) == 512U + kMaxCases * sizeof(CaseSlot), "probe state layout changed");

CACHE_PROBE_HD inline uint64_t InitToken(Direction direction, uint64_t nonce) {
    return UINT64_C(0x5100000000000000) | (static_cast<uint64_t>(direction) << 52U) |
           (nonce & UINT64_C(0x000fffffffffffff));
}

CACHE_PROBE_HD inline int64_t ControlValue(Direction direction, uint32_t case_index, uint32_t generation) {
    return static_cast<int64_t>(
        UINT64_C(0x1000000000000000) | (static_cast<uint64_t>(direction) << 52U) |
        (static_cast<uint64_t>(case_index + 1U) << 40U) | static_cast<uint64_t>(generation + 1U)
    );
}

CACHE_PROBE_HD inline uint64_t
PayloadValue(Direction direction, uint32_t case_index, uint32_t generation, uint32_t word, uint32_t line_index = 0U) {
    return UINT64_C(0xa500000000000000) ^ (static_cast<uint64_t>(direction) << 55U) ^
           (static_cast<uint64_t>(case_index + 1U) << 47U) ^ (static_cast<uint64_t>(generation + 1U) << 16U) ^
           (static_cast<uint64_t>(line_index * kPayloadWords + word + 1U) * UINT64_C(0x101));
}

inline const char *AicpuToAicoreCaseName(uint32_t index) {
    switch (index) {
    case 0U:
        return "production_atomicAdd_default_dcci";
    case 1U:
        return "control_atomicMax_default_dcci";
    case 2U:
        return "control_atomicCAS_default_dcci";
    case 3U:
        return "control_ld_dev_payload_ld_dev";
    case 4U:
        return "cached_control_ordinary_load";
    case 5U:
        return "cached_payload_without_dcci";
    case 6U:
        return "payload_dcci_explicit_all";
    case 7U:
        return "payload_dcci_out";
    case 8U:
        return "payload_dcci_atomic";
    case 9U:
        return "aicpu_clean_dsb_without_isb";
    case 10U:
        return "aicpu_atomic_store_release_no_cvac";
    case 11U:
        return "aicpu_atomic_exchange_no_cvac";
    case 12U:
        return "aicpu_ordinary_control_without_clean";
    case 13U:
        return "aicpu_payload_without_clean";
    case 14U:
        return "aicpu_atomic_fetch_add_no_cvac";
    case 15U:
        return "aicpu_atomic_compare_exchange_no_cvac";
    case 16U:
        return "aicpu_unclean_control_observed_by_ld_dev";
    case 17U:
        return "aicpu_unclean_payload_observed_by_ld_dev";
    case 18U:
        return "direct_ordinary_no_clean_no_barrier";
    case 19U:
        return "direct_release_no_cvac";
    case 20U:
        return "direct_exchange_no_cvac";
    case 21U:
        return "direct_fetch_add_no_cvac";
    case 22U:
        return "direct_compare_exchange_no_cvac";
    case 23U:
        return "separate_release_doorbell_no_cvac";
    case 24U:
        return "separate_relaxed_doorbell_no_cvac";
    case 25U:
        return "separate_dmb_relaxed_doorbell_no_cvac";
    case 26U:
        return "separate_dsb_relaxed_doorbell_no_cvac";
    case 27U:
        return "direct_release_payload_without_dcci";
    case 28U:
        return "direct_release_payload_ld_dev";
    case 29U:
        return "separate_release_cached_control";
    case 30U:
        return "direct_ordinary_dmb_no_clean";
    case 31U:
        return "direct_ordinary_dsb_no_clean";
    default:
        return "invalid";
    }
}

inline const char *AicoreToAicpuCaseName(uint32_t index) {
    switch (index) {
    case 0U:
        return "production_atomicExch_default_dcci";
    case 1U:
        return "control_atomicAdd_default_dcci";
    case 2U:
        return "control_atomicMax_default_dcci";
    case 3U:
        return "control_atomicCAS_default_dcci";
    case 4U:
        return "control_ordinary_store_dcci";
    case 5U:
        return "payload_dcci_explicit_all";
    case 6U:
        return "payload_dcci_out";
    case 7U:
        return "payload_dcci_atomic";
    case 8U:
        return "aicpu_cached_ordinary_observer";
    case 9U:
        return "aicpu_relaxed_atomic_observer";
    case 10U:
        return "aicpu_redundant_civac_reader";
    case 11U:
        return "aicore_payload_without_dcci";
    case 12U:
        return "aicore_ordinary_control_without_dcci";
    case 13U:
        return "aicore_st_dev_control";
    case 14U:
        return "direct_32line_default_dcci_dsb";
    case 15U:
        return "direct_32line_default_dcci_no_dsb";
    case 16U:
        return "direct_32line_out_dcci_dsb";
    case 17U:
        return "direct_32line_out_dcci_no_dsb";
    case 18U:
        return "direct_32line_no_dcci_dsb";
    case 19U:
        return "direct_32line_no_dcci_no_dsb";
    default:
        return "invalid";
    }
}

#undef CACHE_PROBE_HD

}  // namespace aicpu_aicore_cache_probe

#endif  // ATOMIC_PROBE_AICPU_AICORE_CACHE_SHARED_H_
