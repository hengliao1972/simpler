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

#ifndef PA_SCHEDULER_CROSS_CORE_CCEC_PROBE_SHARED_H
#define PA_SCHEDULER_CROSS_CORE_CCEC_PROBE_SHARED_H

#include <stddef.h>
#include <stdint.h>

// 同一 ABI 头同时供普通 C++ host 和 CCEC device 编译；这里仅根据编译器
// 固有地址空间选择限定符，不引入可改变测试行为的构建宏。
#if defined(__CCE_AICORE__)
#define PA_DEVICE __aicore__ inline
#define PA_GM __gm__
#else
#define PA_DEVICE inline
#define PA_GM
#endif
#include "../common/shared_exec_protocol.h"

namespace pa_scheduler::cross_core::probe {

constexpr uint32_t kProbeMagic = 0x58454343U;
constexpr uint32_t kProbeVersion = 1;
constexpr uint32_t kProbeResultMagic = 0x52534C54U;
constexpr uint32_t kProbeParticipants = 4;
constexpr uint32_t kProbeDirections = 4;
constexpr uint32_t kProbePayloadShapes = 4;
constexpr uint32_t kProbeAcquireModes = 2;
constexpr uint32_t kProbeCaseCount =
    kProbeDirections * kProbePayloadShapes * kProbeAcquireModes;
constexpr uint32_t kProbeTaskBase = 100;
constexpr int64_t kProbeNoFatal = -1;
constexpr uint8_t kProbeGuardBefore = 0x5A;
constexpr uint8_t kProbeGuardAfter = 0xA5;

enum class ProbeDirection : uint32_t {
    AivToAiv = 0,
    AivToAic = 1,
    AicToAiv = 2,
    AicToAic = 3,
};

enum class ProbeDelayPoint : uint32_t {
    MidPack = 0,
    BeforeBuilt = 1,
    BeforeClaim = 2,
    AfterClaim = 3,
};

enum class ProbeAcquireMode : uint32_t {
    ReturnDependency = 0,
    PreDsb = 1,
};

enum ProbeStatus : uint32_t {
    ProbeClaimed = 1U << 0,
    ProbeHeaderValid = 1U << 1,
    ProbePayloadValid = 1U << 2,
    ProbeDone = 1U << 3,
    ProbeTokenIdle = 1U << 4,
};

constexpr uint32_t kProbeExpectedStatus =
    ProbeClaimed | ProbeHeaderValid | ProbePayloadValid |
    ProbeDone | ProbeTokenIdle;

struct ProbePayloadShape {
    uint16_t tensors;
    uint16_t scalars;
    uint16_t fanin;
    uint16_t lines;
};

constexpr ProbePayloadShape kProbeShapes[kProbePayloadShapes] = {
    {0, 0, 0, 1},
    {0, 8, 0, 2},
    {3, 8, 0, 8},
    {32, 16, 16, 68},
};

struct alignas(kExecCacheLineBytes) ProbeControl {
    uint32_t magic;
    uint32_t version;
    uint32_t case_count;
    uint32_t participant_count;
    uint32_t delay_nops;
    uint32_t reserved0;
    uint64_t timeout_ticks;
    uint64_t launch_nonce;
    uint64_t reserved[3];
};

struct alignas(kExecCacheLineBytes) ProbeAtomicLine {
    volatile int64_t value;
    uint8_t padding[kExecCacheLineBytes - sizeof(int64_t)];
};

struct alignas(kExecCacheLineBytes) ProbeResult {
    uint32_t magic;
    uint32_t case_id;
    uint32_t status;
    uint32_t builder_executor;
    uint32_t direction_shape_acquire_delay;
    uint32_t physical_core_id;
    uint32_t subblock_id;
    uint32_t reserved0;
    uint64_t observed_checksum;
    uint64_t expected_checksum;
    uint64_t final_state;
    uint64_t elapsed_ticks;
};

struct alignas(kExecCacheLineBytes) ProbeParticipantResult {
    uint32_t magic;
    uint32_t participant;
    uint32_t role;
    uint32_t block_index;
    uint32_t block_count;
    uint32_t subblock_index;
    uint32_t subblock_count;
    uint32_t physical_core_id;
    uint64_t reserved[4];
};

struct alignas(kExecCacheLineBytes) ProbeState {
    ProbeControl control;
    uint8_t guard_before[kExecCacheLineBytes];
    SharedExecCell cells[kProbeCaseCount];
    uint8_t guard_after[kExecCacheLineBytes];
    ProbeAtomicLine fatal_case;
    SharedExecFatalControl protocol_fatal;
    ExecutionToken tokens[kProbeParticipants];
    ProbeParticipantResult participants[kProbeParticipants];
    ProbeResult results[kProbeCaseCount];
};

static_assert(sizeof(ProbeControl) == 64, "probe control must own one line");
static_assert(sizeof(ProbeAtomicLine) == 64, "probe fatal must own one line");
static_assert(sizeof(ProbeResult) == 64, "probe result must own one line");
static_assert(
    sizeof(ProbeParticipantResult) == 64,
    "participant topology result must own one line"
);
static_assert(
    offsetof(ProbeState, cells) % kExecCacheLineBytes == 0 &&
        offsetof(ProbeState, fatal_case) % kExecCacheLineBytes == 0 &&
        offsetof(ProbeState, protocol_fatal) %
                kExecCacheLineBytes == 0 &&
        offsetof(ProbeState, tokens) % kExecCacheLineBytes == 0 &&
        offsetof(ProbeState, participants) % kExecCacheLineBytes == 0 &&
        offsetof(ProbeState, results) % kExecCacheLineBytes == 0,
    "every probe region must remain cache-line aligned"
);

PA_DEVICE ProbeDirection DirectionForCase(uint32_t case_id) {
    return static_cast<ProbeDirection>(
        case_id /
        (kProbePayloadShapes * kProbeAcquireModes)
    );
}

PA_DEVICE uint32_t ShapeIndexForCase(uint32_t case_id) {
    return (case_id / kProbeAcquireModes) %
           kProbePayloadShapes;
}

PA_DEVICE ProbeAcquireMode AcquireModeForCase(uint32_t case_id) {
    return static_cast<ProbeAcquireMode>(
        case_id % kProbeAcquireModes
    );
}

PA_DEVICE ProbeDelayPoint DelayPointForCase(uint32_t case_id) {
    return static_cast<ProbeDelayPoint>(
        (static_cast<uint32_t>(DirectionForCase(case_id)) +
         ShapeIndexForCase(case_id) +
         static_cast<uint32_t>(AcquireModeForCase(case_id))) % 4U
    );
}

// mixed 1:1 的逻辑参与者顺序固定为：AIC0、AIV0、AIC1、AIV1。
PA_DEVICE uint32_t BuilderForCase(uint32_t case_id) {
    switch (DirectionForCase(case_id)) {
        case ProbeDirection::AivToAiv:
        case ProbeDirection::AivToAic:
            return 1;
        case ProbeDirection::AicToAiv:
        case ProbeDirection::AicToAic:
            return 0;
    }
    return UINT32_MAX;
}

PA_DEVICE uint32_t ExecutorForCase(uint32_t case_id) {
    switch (DirectionForCase(case_id)) {
        case ProbeDirection::AivToAiv:
        case ProbeDirection::AicToAiv:
            return 3;
        case ProbeDirection::AivToAic:
        case ProbeDirection::AicToAic:
            return 2;
    }
    return UINT32_MAX;
}

PA_DEVICE ExecEngineClass EngineForCase(uint32_t case_id) {
    return (ExecutorForCase(case_id) & 1U) == 0
        ? ExecEngineClass::Aic
        : ExecEngineClass::Aiv;
}

PA_DEVICE ExecPayloadSpec SpecForCase(uint32_t case_id) {
    const ProbePayloadShape shape =
        kProbeShapes[ShapeIndexForCase(case_id)];
    const uint32_t task_id = kProbeTaskBase + case_id;
    return ExecPayloadSpec{
        task_id,
        0xC001000000000000ULL | BuilderForCase(case_id),
        0xD000000000000000ULL | case_id,
        0x1000U + case_id,
        shape.tensors,
        shape.scalars,
        shape.fanin,
        EngineForCase(case_id),
        0,
        0,
        0,
        1,
    };
}

PA_DEVICE uint64_t TensorWordValue(
    uint32_t case_id, uint32_t tensor, uint32_t word
) {
    return 0xA000000000000000ULL |
           (static_cast<uint64_t>(case_id) << 32U) |
           (static_cast<uint64_t>(tensor) << 16U) |
           word;
}

PA_DEVICE uint64_t ScalarValue(
    uint32_t case_id, uint32_t scalar
) {
    return 0xB000000000000000ULL |
           (static_cast<uint64_t>(case_id) << 32U) |
           scalar;
}

PA_DEVICE int32_t FaninValue(uint32_t case_id, uint32_t edge) {
    const int32_t task_id = static_cast<int32_t>(
        kProbeTaskBase + case_id
    );
    return task_id - static_cast<int32_t>(edge) - 1;
}

PA_DEVICE uint64_t HashWord(uint64_t hash, uint64_t word) {
    return (hash ^ word) * 1099511628211ULL;
}

PA_DEVICE uint64_t ExpectedPayloadWord(
    uint32_t case_id, const ExecPayloadSpec &spec,
    const ExecPayloadLayout &layout, uint32_t word
) {
    if (word == 0) {
        return PackExecHeaderWord0(spec.task_id);
    }
    if (word == 1) return spec.function_address;
    if (word == 2) return spec.completion_vend;
    if (word == 3) {
        return PackExecHeaderWord3(
            spec.function_id, layout.payload_bytes
        );
    }
    if (word == 4) return PackExecHeaderWord4(spec);
    if (word == 5) return PackExecHeaderWord5(spec);
    if (word < kExecHeaderWords) return 0;
    if (word < layout.scalar_word_offset) {
        const uint32_t relative = word - layout.tensor_word_offset;
        return TensorWordValue(
            case_id,
            relative / kExecTensorDescWords,
            relative % kExecTensorDescWords
        );
    }
    if (word < layout.fanin_word_offset) {
        return ScalarValue(
            case_id, word - layout.scalar_word_offset
        );
    }
    const uint32_t edge =
        (word - layout.fanin_word_offset) * 2U;
    const uint64_t low = static_cast<uint32_t>(
        FaninValue(case_id, edge)
    );
    const uint64_t high = edge + 1U < spec.fanin_count
        ? static_cast<uint32_t>(FaninValue(case_id, edge + 1U))
        : 0U;
    return low | (high << 32U);
}

PA_DEVICE uint32_t PackBuilderExecutor(
    uint32_t builder, uint32_t executor
) {
    return (builder & 0xFFFFU) | (executor << 16U);
}

PA_DEVICE uint32_t PackCaseProperties(uint32_t case_id) {
    return static_cast<uint32_t>(DirectionForCase(case_id)) |
           (ShapeIndexForCase(case_id) << 8U) |
           (static_cast<uint32_t>(AcquireModeForCase(case_id)) << 16U) |
           (static_cast<uint32_t>(DelayPointForCase(case_id)) << 24U);
}

}  // namespace pa_scheduler::cross_core::probe

#undef PA_GM
#undef PA_DEVICE

#endif  // PA_SCHEDULER_CROSS_CORE_CCEC_PROBE_SHARED_H
