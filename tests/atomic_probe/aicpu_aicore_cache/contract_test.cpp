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
#include "shared.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace {

using namespace aicpu_aicore_cache_probe;

[[noreturn]] void Fail(const char *message) {
    std::fprintf(stderr, "[FAIL] %s\n", message);
    std::exit(EXIT_FAILURE);
}

void Check(bool condition, const char *message) {
    if (!condition) Fail(message);
}

void CheckExpectation(Expectation expectation) {
    Check(
        expectation == Expectation::Fresh || expectation == Expectation::Stale ||
            expectation == Expectation::ObserveOnly,
        "invalid expectation"
    );
}

void CheckDoorbell(AicpuDoorbellPublish publication) {
    Check(
        publication == AicpuDoorbellPublish::OrdinaryCleanDsbIsb ||
            publication == AicpuDoorbellPublish::TestedControl ||
            publication == AicpuDoorbellPublish::AtomicStoreRelease ||
            publication == AicpuDoorbellPublish::AtomicStoreRelaxed ||
            publication == AicpuDoorbellPublish::AtomicExchangeSeqCst ||
            publication == AicpuDoorbellPublish::DmbIshstThenAtomicRelaxed ||
            publication == AicpuDoorbellPublish::DsbSyThenAtomicRelaxed,
        "invalid AICPU doorbell publication"
    );
}

void CheckDoorbell(AicoreDoorbellPublish publication) {
    Check(
        publication == AicoreDoorbellPublish::IndependentDone || publication == AicoreDoorbellPublish::TestedControl,
        "invalid AICore doorbell publication"
    );
}

}  // namespace

int main() {
    using namespace aicpu_aicore_cache_probe;
    Check(kAicpuToAicoreCaseCount <= kMaxCases, "AICPU->AICore case count exceeds storage");
    Check(kAicoreToAicpuCaseCount <= kMaxCases, "AICore->AICPU case count exceeds storage");
    Check(kRounds % kDirectQuietAckInterval == 0U, "quiet-ACK sampling interval must divide rounds");

    for (uint32_t index = 0U; index < kAicpuToAicoreCaseCount; ++index) {
        const AicpuToAicoreCase spec = ResolveAicpuToAicoreCase(index);
        Check(std::strcmp(AicpuToAicoreCaseName(index), "invalid") != 0, "unnamed AICPU->AICore case");
        CheckExpectation(spec.control_primary);
        CheckExpectation(spec.control_reference);
        CheckExpectation(spec.payload_primary);
        CheckExpectation(spec.payload_reference);
        CheckDoorbell(spec.doorbell_publish);
        for (uint32_t other = index + 1U; other < kAicpuToAicoreCaseCount; ++other) {
            Check(
                std::strcmp(AicpuToAicoreCaseName(index), AicpuToAicoreCaseName(other)) != 0,
                "duplicate AICPU->AICore case name"
            );
        }
    }

    const AicoreToAicpuCase direct_default_no_dsb = ResolveAicoreToAicpuCase(15U);
    Check(
        direct_default_no_dsb.control_publish == AicoreControlPublish::AtomicExchange &&
            direct_default_no_dsb.payload_publish == AicorePayloadPublish::DcciDefaultNoDsb &&
            direct_default_no_dsb.payload_primary == Expectation::Fresh &&
            direct_default_no_dsb.payload_reference == Expectation::Fresh &&
            direct_default_no_dsb.doorbell_publish == AicoreDoorbellPublish::TestedControl &&
            direct_default_no_dsb.payload_lines == kPayloadLineCapacity,
        "default-DCCI-without-DSB gate must remain direct and require all payload lines fresh"
    );
    const AicoreToAicpuCase direct_out_no_dsb = ResolveAicoreToAicpuCase(17U);
    Check(
        direct_out_no_dsb.control_publish == AicoreControlPublish::AtomicExchange &&
            direct_out_no_dsb.payload_publish == AicorePayloadPublish::DcciOutNoDsb &&
            direct_out_no_dsb.payload_primary == Expectation::Fresh &&
            direct_out_no_dsb.payload_reference == Expectation::Fresh &&
            direct_out_no_dsb.doorbell_publish == AicoreDoorbellPublish::TestedControl &&
            direct_out_no_dsb.payload_lines == kPayloadLineCapacity,
        "OUT-DCCI-without-DSB gate must remain direct and require all payload lines fresh"
    );
    const AicoreToAicpuCase direct_no_dcci_dsb = ResolveAicoreToAicpuCase(18U);
    Check(
        direct_no_dcci_dsb.payload_publish == AicorePayloadPublish::NoDcci &&
            direct_no_dcci_dsb.payload_primary == Expectation::Stale &&
            direct_no_dcci_dsb.payload_reference == Expectation::Stale &&
            direct_no_dcci_dsb.doorbell_publish == AicoreDoorbellPublish::TestedControl &&
            direct_no_dcci_dsb.payload_lines == kPayloadLineCapacity,
        "DSB-without-DCCI negative control must remain direct and require all payload lines stale"
    );

    const AicpuToAicoreCase release_without_cvac = ResolveAicpuToAicoreCase(19U);
    Check(
        release_without_cvac.control_publish == AicpuControlPublish::AtomicStoreRelease &&
            release_without_cvac.payload_publish == AicpuPayloadPublish::OrdinaryNoClean &&
            release_without_cvac.doorbell_publish == AicpuDoorbellPublish::TestedControl,
        "release-without-cvac gate must not gain a hidden clean doorbell"
    );
    for (uint32_t index = 0U; index < kAicoreToAicpuCaseCount; ++index) {
        const AicoreToAicpuCase spec = ResolveAicoreToAicpuCase(index);
        Check(std::strcmp(AicoreToAicpuCaseName(index), "invalid") != 0, "unnamed AICore->AICPU case");
        CheckExpectation(spec.control_primary);
        CheckExpectation(spec.control_reference);
        CheckExpectation(spec.payload_primary);
        CheckExpectation(spec.payload_reference);
        CheckDoorbell(spec.doorbell_publish);
        Check(spec.payload_lines >= 1U && spec.payload_lines <= kPayloadLineCapacity, "invalid payload line count");
        for (uint32_t other = index + 1U; other < kAicoreToAicpuCaseCount; ++other) {
            Check(
                std::strcmp(AicoreToAicpuCaseName(index), AicoreToAicpuCaseName(other)) != 0,
                "duplicate AICore->AICPU case name"
            );
        }
    }

    for (Direction direction : {Direction::AicpuToAicore, Direction::AicoreToAicpu}) {
        for (uint32_t case_index = 0U; case_index < kMaxCases; ++case_index) {
            const int64_t initial = ControlValue(direction, case_index, 0U);
            const int64_t next = ControlValue(direction, case_index, 1U);
            Check(next == initial + 1, "control generations must be unit-stride for atomicAdd cases");
            for (uint32_t word = 0U; word < kPayloadWords; ++word) {
                Check(
                    PayloadValue(direction, case_index, 0U, word) != PayloadValue(direction, case_index, 1U, word),
                    "successive payload generations must differ"
                );
            }
        }
    }

    Check(offsetof(CaseSlot, ready) % kAtomicIsolationBytes == 0U, "ready line alignment");
    Check(offsetof(CaseSlot, tested_control) % kAtomicIsolationBytes == 0U, "tested control alignment");
    Check(offsetof(CaseSlot, done) % kAtomicIsolationBytes == 0U, "done line alignment");
    Check(offsetof(CaseSlot, payload) % kAtomicIsolationBytes == 0U, "payload alignment");
    Check(sizeof(CaseSlot::payload) == kPayloadLineCapacity * sizeof(PayloadLine), "payload region size");
    Check(offsetof(CaseSlot, result) % kAtomicIsolationBytes == 0U, "result alignment");
    std::printf(
        "[PASS] layout=%zu bytes, AICPU->AICore=%u cases, AICore->AICPU=%u cases, rounds=%u\n", sizeof(ProbeState),
        kAicpuToAicoreCaseCount, kAicoreToAicpuCaseCount, kRounds
    );
    return EXIT_SUCCESS;
}
