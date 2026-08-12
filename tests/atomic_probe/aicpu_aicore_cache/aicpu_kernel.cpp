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
#include "common/kernel_args.h"

#include "shared.h"

#include <ctime>

namespace {

using namespace aicpu_aicore_cache_probe;

constexpr uint64_t kNanosecondsPerSecond = UINT64_C(1000000000);

uint64_t MonotonicNanoseconds() {
    timespec timestamp{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) return 0U;
    return static_cast<uint64_t>(timestamp.tv_sec) * kNanosecondsPerSecond + static_cast<uint64_t>(timestamp.tv_nsec);
}

inline void FullBarrier() {
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

inline void InstructionBarrier() {
#if defined(__aarch64__)
    __asm__ volatile("isb" ::: "memory");
#endif
}

inline void StoreBarrierInnerShareable() {
#if defined(__aarch64__)
    __asm__ volatile("dmb ishst" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
}

void CleanRange(const void *address, uint64_t bytes) {
    if (bytes == 0U) return;
#if defined(__aarch64__)
    const uintptr_t begin = reinterpret_cast<uintptr_t>(address) & ~uintptr_t{63U};
    const uintptr_t end = (reinterpret_cast<uintptr_t>(address) + bytes + 63U) & ~uintptr_t{63U};
    for (uintptr_t line = begin; line < end; line += kCacheLineBytes) {
        __asm__ volatile("dc cvac, %0" : : "r"(line) : "memory");
    }
#else
    (void)address;
    __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
}

void DiscardRange(const void *address, uint64_t bytes) {
    if (bytes == 0U) return;
#if defined(__aarch64__)
    const uintptr_t begin = reinterpret_cast<uintptr_t>(address) & ~uintptr_t{63U};
    const uintptr_t end = (reinterpret_cast<uintptr_t>(address) + bytes + 63U) & ~uintptr_t{63U};
    for (uintptr_t line = begin; line < end; line += kCacheLineBytes) {
        __asm__ volatile("dc civac, %0" : : "r"(line) : "memory");
    }
#else
    (void)address;
#endif
    FullBarrier();
    InstructionBarrier();
}

void PublishOrdinaryControl(volatile int64_t *address, int64_t value, bool instruction_barrier) {
    *address = value;
    __asm__ volatile("" ::: "memory");
    CleanRange(const_cast<const int64_t *>(address), sizeof(*address));
    FullBarrier();
    if (instruction_barrier) InstructionBarrier();
}

bool WaitAcquireExact(const volatile int64_t *address, int64_t expected, uint64_t timeout_ns) {
    const uint64_t begin = MonotonicNanoseconds();
    uint32_t polls = 0U;
    while (__atomic_load_n(address, __ATOMIC_ACQUIRE) != expected) {
        ++polls;
        if ((polls & 1023U) == 0U && MonotonicNanoseconds() - begin > timeout_ns) {
            return false;
        }
    }
    return true;
}

void StoreWords(volatile uint64_t *destination, const uint64_t *source, uint32_t words) {
    for (uint32_t word = 0U; word < words; ++word)
        destination[word] = source[word];
}

void ClearWords(volatile uint64_t *destination, uint32_t words) {
    for (uint32_t word = 0U; word < words; ++word)
        destination[word] = 0U;
}

void InitializeState(ProbeState *state, Direction direction, uint32_t case_count) {
    ClearWords(reinterpret_cast<volatile uint64_t *>(&state->aicpu_initialized), sizeof(AtomicLine) / sizeof(uint64_t));
    ClearWords(reinterpret_cast<volatile uint64_t *>(&state->aicpu_errors), sizeof(AtomicLine) / sizeof(uint64_t));
    ClearWords(reinterpret_cast<volatile uint64_t *>(&state->aicore_errors), sizeof(AtomicLine) / sizeof(uint64_t));
    for (uint32_t case_index = 0U; case_index < case_count; ++case_index) {
        CaseSlot &slot = state->cases[case_index];
        ClearWords(reinterpret_cast<volatile uint64_t *>(&slot.ready), sizeof(AtomicLine) / sizeof(uint64_t));
        ClearWords(reinterpret_cast<volatile uint64_t *>(&slot.tested_control), sizeof(AtomicLine) / sizeof(uint64_t));
        slot.tested_control.value = ControlValue(direction, case_index, 0U);
        ClearWords(reinterpret_cast<volatile uint64_t *>(&slot.done), sizeof(AtomicLine) / sizeof(uint64_t));
        for (uint32_t line = 0U; line < kPayloadLineCapacity; ++line) {
            ClearWords(
                reinterpret_cast<volatile uint64_t *>(&slot.payload[line]), sizeof(PayloadLine) / sizeof(uint64_t)
            );
            for (uint32_t word = 0U; word < kPayloadWords; ++word) {
                slot.payload[line].words[word] = PayloadValue(direction, case_index, 0U, word, line);
            }
        }
        ClearWords(reinterpret_cast<volatile uint64_t *>(&slot.result), sizeof(CaseResult) / sizeof(uint64_t));
    }
    const uintptr_t begin = reinterpret_cast<uintptr_t>(&state->aicpu_initialized);
    const uintptr_t end = reinterpret_cast<uintptr_t>(&state->cases[case_count]);
    CleanRange(reinterpret_cast<const void *>(begin), end - begin);
    FullBarrier();
    InstructionBarrier();
}

void PublishAicpuError(ProbeState *state, uint64_t errors) {
    PublishOrdinaryControl(&state->aicpu_errors.value, static_cast<int64_t>(errors), true);
}

void PublishTestedControl(
    volatile int64_t *address, AicpuControlPublish publication, int64_t expected_old, int64_t desired, uint64_t *errors
) {
    if (publication == AicpuControlPublish::OrdinaryCleanDsbIsb) {
        PublishOrdinaryControl(address, desired, true);
    } else if (publication == AicpuControlPublish::OrdinaryCleanDsb) {
        PublishOrdinaryControl(address, desired, false);
    } else if (publication == AicpuControlPublish::AtomicStoreRelease) {
        __atomic_store_n(address, desired, __ATOMIC_RELEASE);
    } else if (publication == AicpuControlPublish::AtomicExchangeSeqCst) {
        const int64_t previous = __atomic_exchange_n(address, desired, __ATOMIC_SEQ_CST);
        if (previous != expected_old) ++*errors;
    } else if (publication == AicpuControlPublish::AtomicFetchAddSeqCst) {
        const int64_t previous = __atomic_fetch_add(address, desired - expected_old, __ATOMIC_SEQ_CST);
        if (previous != expected_old) ++*errors;
    } else if (publication == AicpuControlPublish::AtomicCompareExchangeSeqCst) {
        int64_t expected = expected_old;
        const bool exchanged =
            __atomic_compare_exchange_n(address, &expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        if (!exchanged || expected != expected_old) ++*errors;
    } else if (publication == AicpuControlPublish::OrdinaryNoCleanDmbIshst) {
        StoreBarrierInnerShareable();
        *address = desired;
        __asm__ volatile("" ::: "memory");
    } else if (publication == AicpuControlPublish::OrdinaryNoCleanDsbSy) {
        FullBarrier();
        *address = desired;
        __asm__ volatile("" ::: "memory");
    } else {
        *address = desired;
        __asm__ volatile("" ::: "memory");
    }
}

void PublishDoorbell(
    volatile int64_t *address, AicpuDoorbellPublish publication, int64_t expected_old, int64_t desired, uint64_t *errors
) {
    if (publication == AicpuDoorbellPublish::OrdinaryCleanDsbIsb) {
        PublishOrdinaryControl(address, desired, true);
    } else if (publication == AicpuDoorbellPublish::AtomicStoreRelease) {
        __atomic_store_n(address, desired, __ATOMIC_RELEASE);
    } else if (publication == AicpuDoorbellPublish::AtomicStoreRelaxed) {
        __atomic_store_n(address, desired, __ATOMIC_RELAXED);
    } else if (publication == AicpuDoorbellPublish::AtomicExchangeSeqCst) {
        const int64_t previous = __atomic_exchange_n(address, desired, __ATOMIC_SEQ_CST);
        if (previous != expected_old) ++*errors;
    } else if (publication == AicpuDoorbellPublish::DmbIshstThenAtomicRelaxed) {
        StoreBarrierInnerShareable();
        __atomic_store_n(address, desired, __ATOMIC_RELAXED);
    } else if (publication == AicpuDoorbellPublish::DsbSyThenAtomicRelaxed) {
        FullBarrier();
        __atomic_store_n(address, desired, __ATOMIC_RELAXED);
    }
}

void RunAicpuToAicore(ProbeState *state, const ProbeConfig &config) {
    const Direction direction = Direction::AicpuToAicore;
    uint64_t errors = 0U;
    for (uint32_t case_index = 0U; case_index < config.case_count; ++case_index) {
        const AicpuToAicoreCase spec = ResolveAicpuToAicoreCase(case_index);
        CaseSlot &slot = state->cases[case_index];
        for (uint32_t round = 0U; round < config.rounds; ++round) {
            const uint32_t generation = round + 1U;
            if (!WaitAcquireExact(&slot.ready.value, static_cast<int64_t>(generation), config.aicpu_timeout_ns)) {
                PublishAicpuError(state, errors + 1U);
                return;
            }

            for (uint32_t word = 0U; word < kPayloadWords; ++word) {
                slot.payload[0].words[word] = PayloadValue(direction, case_index, generation, word);
            }
            __asm__ volatile("" ::: "memory");
            if (spec.payload_publish == AicpuPayloadPublish::OrdinaryClean) {
                CleanRange(&slot.payload[0], kCacheLineBytes);
                FullBarrier();
            }

            const int64_t expected_old = ControlValue(direction, case_index, round);
            const int64_t expected_new = ControlValue(direction, case_index, generation);
            PublishTestedControl(&slot.tested_control.value, spec.control_publish, expected_old, expected_new, &errors);
            if (spec.doorbell_publish != AicpuDoorbellPublish::TestedControl) {
                PublishDoorbell(
                    &slot.done.value, spec.doorbell_publish, static_cast<int64_t>(round),
                    static_cast<int64_t>(generation), &errors
                );
            }
        }
        if (!WaitAcquireExact(&slot.ready.value, static_cast<int64_t>(config.rounds + 1U), config.aicpu_timeout_ns)) {
            PublishAicpuError(state, errors + 1U);
            return;
        }
    }
    PublishAicpuError(state, errors);
}

struct LineClass {
    bool fresh;
    bool stale;
    uint64_t first;
};

LineClass ClassifyLine(
    const uint64_t observed[kPayloadWords], const uint64_t primed[kPayloadWords], Direction direction,
    uint32_t case_index, uint32_t generation, uint32_t line_index = 0U
) {
    bool fresh = true;
    bool stale = true;
    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        fresh = fresh && observed[word] == PayloadValue(direction, case_index, generation, word, line_index);
        stale = stale && observed[word] == primed[word];
    }
    return {fresh, stale, observed[0]};
}

void CountValue(
    int64_t observed, int64_t fresh, int64_t stale, uint64_t *fresh_count, uint64_t *stale_count, uint64_t *other_count
) {
    if (observed == fresh) {
        ++*fresh_count;
    } else if (observed == stale) {
        ++*stale_count;
    } else {
        ++*other_count;
    }
}

void CountLine(const LineClass &classification, uint64_t *fresh_count, uint64_t *stale_count, uint64_t *other_count) {
    if (classification.fresh) {
        ++*fresh_count;
    } else if (classification.stale) {
        ++*stale_count;
    } else {
        ++*other_count;
    }
}

int64_t ObserveControl(volatile int64_t *address, AicpuControlObserve observer) {
    if (observer == AicpuControlObserve::CivacThenOrdinary) {
        DiscardRange(const_cast<const int64_t *>(address), sizeof(*address));
        return *address;
    }
    if (observer == AicpuControlObserve::Ordinary) return *address;
    if (observer == AicpuControlObserve::AtomicRelaxed) {
        return __atomic_load_n(address, __ATOMIC_RELAXED);
    }
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

void ReadPayload(const PayloadLine *payload, uint64_t output[kPayloadWords]) {
    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        output[word] = payload->words[word];
    }
}

void PublishResult(CaseResult *destination, const CaseResult &source) {
    StoreWords(
        reinterpret_cast<volatile uint64_t *>(destination), reinterpret_cast<const uint64_t *>(&source),
        sizeof(CaseResult) / sizeof(uint64_t)
    );
    CleanRange(destination, sizeof(CaseResult));
    FullBarrier();
    InstructionBarrier();
}

void RunAicoreToAicpu(ProbeState *state, const ProbeConfig &config) {
    const Direction direction = Direction::AicoreToAicpu;
    uint64_t errors = 0U;
    for (uint32_t case_index = 0U; case_index < config.case_count; ++case_index) {
        const AicoreToAicpuCase spec = ResolveAicoreToAicpuCase(case_index);
        CaseSlot &slot = state->cases[case_index];
        CaseResult result{};
        result.magic = kResultMagic;
        result.status = static_cast<uint64_t>(Status::Ok);
        result.first_bad_round = kNoBadRound;
        for (uint32_t round = 0U; round < config.rounds; ++round) {
            const uint32_t generation = round + 1U;
            const int64_t expected_new = ControlValue(direction, case_index, generation);
            const int64_t primed_control = slot.tested_control.value;
            uint64_t primed_payload[kPayloadLineCapacity][kPayloadWords]{};
            for (uint32_t line = 0U; line < spec.payload_lines; ++line) {
                ReadPayload(&slot.payload[line], primed_payload[line]);
            }
            __atomic_thread_fence(__ATOMIC_ACQUIRE);

            PublishOrdinaryControl(&slot.ready.value, static_cast<int64_t>(generation), true);
            const volatile int64_t *doorbell = spec.doorbell_publish == AicoreDoorbellPublish::TestedControl ?
                                                   &slot.tested_control.value :
                                                   &slot.done.value;
            const int64_t expected_doorbell = spec.doorbell_publish == AicoreDoorbellPublish::TestedControl ?
                                                  expected_new :
                                                  static_cast<int64_t>(generation);
            if (!WaitAcquireExact(doorbell, expected_doorbell, config.aicpu_timeout_ns)) {
                result.status = static_cast<uint64_t>(Status::PeerTimeout);
                ++errors;
                break;
            }

            const int64_t control_primary = ObserveControl(&slot.tested_control.value, spec.control_observe);
            LineClass primary_class{true, true, 0U};
            if (spec.payload_acquire == AicpuPayloadAcquire::CivacThenOrdinary) {
                DiscardRange(&slot.payload[0], spec.payload_lines * sizeof(PayloadLine));
            }
            // Read the last DCCI target first so the no-DSB gate gives the final maintenance operation the
            // shortest possible completion window after the atomic control becomes visible.
            for (uint32_t offset = 0U; offset < spec.payload_lines; ++offset) {
                const uint32_t line = spec.payload_lines - offset - 1U;
                uint64_t payload_primary[kPayloadWords]{};
                ReadPayload(&slot.payload[line], payload_primary);
                const LineClass line_class =
                    ClassifyLine(payload_primary, primed_payload[line], direction, case_index, generation, line);
                primary_class.fresh = primary_class.fresh && line_class.fresh;
                primary_class.stale = primary_class.stale && line_class.stale;
                if (line == 0U) primary_class.first = line_class.first;
            }
            __asm__ volatile("" ::: "memory");
            if (spec.doorbell_publish == AicoreDoorbellPublish::TestedControl) {
                __atomic_store_n(&slot.done.value, static_cast<int64_t>(generation), __ATOMIC_RELEASE);
            }

            // Keep the primary payload read immediately after the tested control observation. In particular, the
            // direct DCCI-without-DSB cases must not gain an AICPU dc civac/dsb reference operation in between.
            DiscardRange(&slot.tested_control, sizeof(slot.tested_control));
            const int64_t control_reference = slot.tested_control.value;
            CountValue(
                control_primary, expected_new, primed_control, &result.control_primary_fresh,
                &result.control_primary_stale, &result.control_primary_other
            );
            CountValue(
                control_reference, expected_new, primed_control, &result.control_reference_fresh,
                &result.control_reference_stale, &result.control_reference_other
            );

            LineClass reference_class{true, true, 0U};
            DiscardRange(&slot.payload[0], spec.payload_lines * sizeof(PayloadLine));
            for (uint32_t line = 0U; line < spec.payload_lines; ++line) {
                uint64_t payload_reference[kPayloadWords]{};
                ReadPayload(&slot.payload[line], payload_reference);
                const LineClass line_class =
                    ClassifyLine(payload_reference, primed_payload[line], direction, case_index, generation, line);
                reference_class.fresh = reference_class.fresh && line_class.fresh;
                reference_class.stale = reference_class.stale && line_class.stale;
                if (line == 0U) reference_class.first = line_class.first;
            }
            CountLine(
                primary_class, &result.payload_primary_fresh, &result.payload_primary_stale,
                &result.payload_primary_other
            );
            CountLine(
                reference_class, &result.payload_reference_fresh, &result.payload_reference_stale,
                &result.payload_reference_other
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
        if (spec.doorbell_publish == AicoreDoorbellPublish::TestedControl && result.rounds_completed == config.rounds) {
            PublishOrdinaryControl(&slot.ready.value, static_cast<int64_t>(config.rounds + 1U), true);
        }
        PublishResult(&slot.result, result);
    }
    PublishAicpuError(state, errors);
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_exec(void *argument) {
    using namespace aicpu_aicore_cache_probe;
    if (argument == nullptr) return 0;
    auto *kernel_args = static_cast<KernelArgs *>(argument);
    if (kernel_args->runtime_args == nullptr) return 0;
    auto *state = reinterpret_cast<ProbeState *>(kernel_args->runtime_args);

    DiscardRange(&state->config, sizeof(state->config));
    const ProbeConfig config = state->config;
    const Direction direction = static_cast<Direction>(config.direction);
    const uint32_t expected_cases =
        direction == Direction::AicpuToAicore ? kAicpuToAicoreCaseCount : kAicoreToAicpuCaseCount;
    if (config.magic != kProbeMagic || config.version != kProbeVersion ||
        (direction != Direction::AicpuToAicore && direction != Direction::AicoreToAicpu) ||
        config.case_count != expected_cases || config.case_count > kMaxCases || config.rounds != kRounds ||
        config.aicpu_timeout_ns == 0U) {
        return 0;
    }

    InitializeState(state, direction, config.case_count);
    PublishOrdinaryControl(
        &state->aicpu_initialized.value, static_cast<int64_t>(InitToken(direction, config.nonce)), true
    );
    if (direction == Direction::AicpuToAicore) {
        RunAicpuToAicore(state, config);
    } else {
        RunAicoreToAicpu(state, config);
    }
    return 0;
}
