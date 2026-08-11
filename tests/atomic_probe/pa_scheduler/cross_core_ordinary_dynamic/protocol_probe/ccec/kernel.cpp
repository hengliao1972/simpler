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

#include "cce_aicore_intrinsics.h"
#include <pto/common/kernel_meta.hpp>

#include "probe_shared.h"

#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif
#ifndef CACHELINE_OUT
#define CACHELINE_OUT 2
#endif

namespace {

using namespace pa_scheduler::cross_core;
using namespace pa_scheduler::cross_core::probe;

constexpr uint32_t kDelayNops = 4096;
constexpr uint32_t kFatalPollMask = 0x3FFU;
constexpr uint64_t kHashSeed = 1469598103934665603ULL;

__aicore__ __attribute__((noinline)) void ProbeDelay() {
#pragma clang loop unroll(disable)
    for (uint32_t index = 0; index < kDelayNops; ++index) {
        asm volatile("nop");
    }
}

struct DeviceCommonOps {
    // S2 首轮保持 preload 关闭；这三个 hook 只锁定候选位置，后续
    // on/off A/B 复用相同 ABI 与正确性 oracle。
    __aicore__ static inline void PreloadBuildDestination(
        __gm__ const void *, uint64_t
    ) {}

    __aicore__ static inline void PreloadPayloadSource(
        __gm__ const void *, uint64_t
    ) {}

    __aicore__ static inline void PreloadTokenDestination(
        __gm__ const void *, uint64_t
    ) {}

    __aicore__ static inline int64_t Load(
        __gm__ volatile int64_t *address
    ) {
        constexpr int64_t identity =
            (-9223372036854775807LL - 1LL);
        return atomicMax(
            const_cast<__gm__ int64_t *>(address), identity
        );
    }

    __aicore__ static inline int64_t CompareExchange(
        __gm__ volatile int64_t *address, int64_t expected,
        int64_t desired
    ) {
        return atomicCAS(
            const_cast<__gm__ int64_t *>(address),
            expected, desired
        );
    }

    __aicore__ static inline void StorePayloadWord(
        __gm__ volatile uint64_t *address, uint64_t value
    ) {
        *address = value;
    }

    __aicore__ static inline uint64_t LoadPayloadWord(
        __gm__ const volatile uint64_t *address
    ) {
        return *address;
    }

    __aicore__ static inline void StoreTokenPayloadWord(
        __gm__ volatile uint64_t *address, uint64_t value
    ) {
        *address = value;
    }

    __aicore__ static inline void FlushRegion(
        __gm__ void *address, uint64_t bytes
    ) {
        if (bytes == 0) return;
        __asm__ volatile("" ::: "memory");
        const uint64_t start =
            reinterpret_cast<uint64_t>(address) & ~uint64_t{63};
        const uint64_t end =
            (reinterpret_cast<uint64_t>(address) + bytes + 63U) &
            ~uint64_t{63};
        for (uint64_t current = start;
             current < end; current += 64U) {
            dcci(
                reinterpret_cast<__gm__ uint8_t *>(current),
                SINGLE_CACHE_LINE, CACHELINE_OUT
            );
        }
        dsb((mem_dsb_t)0);
    }

    __aicore__ static inline void InvalidateRegion(
        __gm__ const void *address, uint64_t bytes
    ) {
        if (bytes == 0) return;
        const uint64_t start =
            reinterpret_cast<uint64_t>(address) & ~uint64_t{63};
        const uint64_t end =
            (reinterpret_cast<uint64_t>(address) + bytes + 63U) &
            ~uint64_t{63};
        for (uint64_t current = start;
             current < end; current += 64U) {
            dcci(
                reinterpret_cast<__gm__ uint8_t *>(current),
                SINGLE_CACHE_LINE
            );
        }
        dsb((mem_dsb_t)0);
        __asm__ volatile("" ::: "memory");
    }
};

struct ProbeReadySource {
    __aicore__ inline bool IsReady(int32_t) const {
        return true;
    }
};

struct ProbeCompletionSink {
    uint32_t vend_task = UINT32_MAX;
    uint32_t flag_task = UINT32_MAX;
    uint64_t vend = 0;

    __aicore__ inline bool PublishVend(
        uint32_t task_id, uint64_t value
    ) {
        vend_task = task_id;
        vend = value;
        return true;
    }

    __aicore__ inline bool PublishFlag(uint32_t task_id) {
        flag_task = task_id;
        return vend_task == task_id;
    }
};

struct ProbeEngineCompletion {
    __aicore__ inline bool IsComplete(
        __gm__ const ExecutionToken &
    ) const {
        return true;
    }
};

struct ReturnDependencyOps : DeviceCommonOps {
    __aicore__ static inline void BeforeBuiltPublish(
        uint32_t task_id
    ) {
        const uint32_t case_id = task_id - kProbeTaskBase;
        if (DelayPointForCase(case_id) ==
            ProbeDelayPoint::BeforeBuilt) {
            ProbeDelay();
        }
    }

    __aicore__ static inline void BeforePayloadAcquire(
        uint32_t task_id
    ) {
        const uint32_t case_id = task_id - kProbeTaskBase;
        if (DelayPointForCase(case_id) ==
            ProbeDelayPoint::AfterClaim) {
            ProbeDelay();
        }
    }
};

struct PreDsbOps : DeviceCommonOps {
    __aicore__ static inline void BeforeBuiltPublish(
        uint32_t task_id
    ) {
        ReturnDependencyOps::BeforeBuiltPublish(task_id);
    }

    __aicore__ static inline void BeforePayloadAcquire(
        uint32_t task_id
    ) {
        // 这是与最小序列对照的保守变体，不进入正式协议默认路径。
        __asm__ volatile("" ::: "memory");
        dsb((mem_dsb_t)0);
        __asm__ volatile("" ::: "memory");
        ReturnDependencyOps::BeforePayloadAcquire(task_id);
    }
};

struct ProbePayloadSource {
    uint32_t case_id;
    uint32_t tensor_count;
    uint32_t scalar_count;

    __aicore__ inline uint64_t TensorWord(
        uint32_t tensor, uint32_t word
    ) const {
        if (DelayPointForCase(case_id) ==
                ProbeDelayPoint::MidPack &&
            word == 0 && tensor == tensor_count / 2U) {
            ProbeDelay();
        }
        return TensorWordValue(case_id, tensor, word);
    }

    __aicore__ inline uint64_t TensorReference(uint32_t) const {
        // 当前 CCEC return-dependency probe 只覆盖内联 descriptor；
        // 引用型 payload 由 standalone PA 与 CPU 协议门槛验证。
        return 0;
    }

    __aicore__ inline uint64_t Scalar(uint32_t scalar) const {
        if (DelayPointForCase(case_id) ==
                ProbeDelayPoint::MidPack &&
            tensor_count == 0 && scalar == scalar_count / 2U) {
            ProbeDelay();
        }
        return ScalarValue(case_id, scalar);
    }

    __aicore__ inline int32_t Fanin(uint32_t edge) const {
        return FaninValue(case_id, edge);
    }
};

__aicore__ inline void PublishFatal(
    __gm__ ProbeState *state, uint32_t case_id
) {
    (void)atomicCAS(
        const_cast<__gm__ int64_t *>(&state->fatal_case.value),
        kProbeNoFatal, static_cast<int64_t>(case_id)
    );
}

__aicore__ inline bool FatalPublished(
    __gm__ ProbeState *state
) {
    return DeviceCommonOps::Load(&state->fatal_case.value) !=
               kProbeNoFatal ||
           ExecFatalPublished<DeviceCommonOps>(
               state->protocol_fatal
           );
}

__aicore__ inline void PublishResult(
    __gm__ ProbeResult *destination, const ProbeResult &result
) {
    destination->magic = result.magic;
    destination->case_id = result.case_id;
    destination->status = result.status;
    destination->builder_executor = result.builder_executor;
    destination->direction_shape_acquire_delay =
        result.direction_shape_acquire_delay;
    destination->physical_core_id = result.physical_core_id;
    destination->subblock_id = result.subblock_id;
    destination->reserved0 = result.reserved0;
    destination->observed_checksum = result.observed_checksum;
    destination->expected_checksum = result.expected_checksum;
    destination->final_state = result.final_state;
    destination->elapsed_ticks = result.elapsed_ticks;
    DeviceCommonOps::FlushRegion(destination, sizeof(ProbeResult));
}

__aicore__ inline void PublishParticipantResult(
    __gm__ ProbeParticipantResult *destination,
    uint32_t participant, uint32_t role
) {
    destination->magic = kProbeResultMagic;
    destination->participant = participant;
    destination->role = role;
    destination->block_index =
        static_cast<uint32_t>(get_block_idx());
    destination->block_count =
        static_cast<uint32_t>(get_block_num());
    destination->subblock_index =
        static_cast<uint32_t>(get_subblockid());
    destination->subblock_count =
        static_cast<uint32_t>(get_subblockdim());
    destination->physical_core_id =
        static_cast<uint32_t>(get_coreid());
    for (uint32_t index = 0; index < 4; ++index) {
        destination->reserved[index] = 0;
    }
    DeviceCommonOps::FlushRegion(
        destination, sizeof(ProbeParticipantResult)
    );
}

__aicore__ inline bool WaitCaseDone(
    __gm__ ProbeState *state, uint32_t case_id
) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint32_t polls = 0;
    while (true) {
        const int64_t raw = DeviceCommonOps::Load(
            &state->cells[case_id].control.state
        );
        const DecodedExecState decoded = DecodeExecState(raw);
        if (decoded.valid && decoded.phase == ExecPhase::Done) {
            return true;
        }
        ++polls;
        if ((polls & kFatalPollMask) == 0U) {
            const uint64_t now =
                static_cast<uint64_t>(get_sys_cnt());
            if (FatalPublished(state) ||
                now - begin > state->control.timeout_ticks) {
                return false;
            }
        }
    }
}

__aicore__ inline void BuildCase(
    __gm__ ProbeState *state, uint32_t case_id,
    uint32_t participant
) {
    const ExecPayloadSpec spec = SpecForCase(case_id);
    const ProbePayloadSource source{
        case_id, spec.tensor_count, spec.scalar_count
    };
    // 纯 header 的 1-line case 没有 Source 回调；仍在 pack 前注入一个
    // 延迟窗口，避免该形状完全漏掉 publisher 被抢占场景。
    if (DelayPointForCase(case_id) == ProbeDelayPoint::MidPack &&
        spec.tensor_count == 0 && spec.scalar_count == 0) {
        ProbeDelay();
    }
    const ExecBuildResult result =
        BuildAndPublishExecPayload<ReturnDependencyOps>(
            state->cells[case_id], participant, spec, source,
            state->protocol_fatal
        );
    if (result != ExecBuildResult::Published) {
        PublishFatal(state, case_id);
    }
}

__aicore__ __attribute__((noinline, used)) ExecClaimResult
CrossCoreClaimMinimal(
    __gm__ SharedExecCell &cell, uint32_t task_id,
    uint32_t participant, ExecEngineClass engine_class,
    __gm__ ExecutionToken &token,
    __gm__ SharedExecFatalControl &fatal
) {
    return ClaimAndBindExecPayload<ReturnDependencyOps>(
        cell, task_id, participant, engine_class, token, fatal
    );
}

__aicore__ __attribute__((noinline, used)) ExecClaimResult
CrossCoreClaimPreDsb(
    __gm__ SharedExecCell &cell, uint32_t task_id,
    uint32_t participant, ExecEngineClass engine_class,
    __gm__ ExecutionToken &token,
    __gm__ SharedExecFatalControl &fatal
) {
    return ClaimAndBindExecPayload<PreDsbOps>(
        cell, task_id, participant, engine_class, token, fatal
    );
}

__aicore__ inline void ExecuteCase(
    __gm__ ProbeState *state, uint32_t case_id,
    uint32_t participant
) {
    const ExecPayloadSpec spec = SpecForCase(case_id);
    __gm__ ExecutionToken &token = state->tokens[participant];
    ProbeResult result{};
    result.magic = kProbeResultMagic;
    result.case_id = case_id;
    result.builder_executor = PackBuilderExecutor(
        BuilderForCase(case_id), participant
    );
    result.direction_shape_acquire_delay =
        PackCaseProperties(case_id);
    result.physical_core_id =
        static_cast<uint32_t>(get_coreid());
    result.subblock_id =
        static_cast<uint32_t>(get_subblockid());
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());

    if (DelayPointForCase(case_id) ==
        ProbeDelayPoint::BeforeClaim) {
        ProbeDelay();
    }
    ExecClaimResult claim = ExecClaimResult::NotBuilt;
    uint32_t polls = 0;
    while (claim == ExecClaimResult::NotBuilt) {
        if (AcquireModeForCase(case_id) ==
            ProbeAcquireMode::ReturnDependency) {
            claim = CrossCoreClaimMinimal(
                state->cells[case_id], spec.task_id,
                participant, spec.engine_class, token,
                state->protocol_fatal
            );
        } else {
            claim = CrossCoreClaimPreDsb(
                state->cells[case_id], spec.task_id,
                participant, spec.engine_class, token,
                state->protocol_fatal
            );
        }
        if (claim != ExecClaimResult::NotBuilt) break;
        ++polls;
        if ((polls & kFatalPollMask) == 0U) {
            const uint64_t now =
                static_cast<uint64_t>(get_sys_cnt());
            if (FatalPublished(state) ||
                now - begin > state->control.timeout_ticks) {
                break;
            }
        }
    }
    if (claim != ExecClaimResult::Claimed) {
        result.final_state = static_cast<uint64_t>(
            DeviceCommonOps::Load(
                &state->cells[case_id].control.state
            )
        );
        result.elapsed_ticks =
            static_cast<uint64_t>(get_sys_cnt()) - begin;
        PublishResult(&state->results[case_id], result);
        PublishFatal(state, case_id);
        return;
    }
    result.status |= ProbeClaimed;

    const ExecPayloadHeader header = ExecutionTokenHeader(token);
    const bool header_valid =
        header.task_id == spec.task_id &&
        header.function_address == spec.function_address &&
        header.completion_vend == spec.completion_vend &&
        header.function_id == spec.function_id &&
        header.tensor_count == spec.tensor_count &&
        header.scalar_count == spec.scalar_count &&
        header.fanin_count == spec.fanin_count &&
        header.engine_class == spec.engine_class &&
        token.control.completion_vend == spec.completion_vend &&
        ExecutionTokenFunctionId(token) == spec.function_id &&
        ExecutionTokenTensorReferenceMask(token) ==
            spec.tensor_reference_mask &&
        ExecutionTokenTensorCount(token) == spec.tensor_count &&
        ExecutionTokenScalarCount(token) == spec.scalar_count &&
        ExecutionTokenFaninCount(token) == spec.fanin_count;
    if (header_valid) result.status |= ProbeHeaderValid;

    ExecPayloadLayout layout{};
    bool payload_valid = ComputeExecPayloadLayout(
        spec.tensor_count, spec.scalar_count,
        spec.fanin_count, layout
    ) && ExecutionTokenScalarWordOffset(token) ==
             layout.scalar_word_offset;
    uint64_t observed_hash = kHashSeed;
    uint64_t expected_hash = kHashSeed;
    if (payload_valid) {
        __gm__ const ExecPayloadStorage &payload =
            ExecutionTokenPayload(token);
        for (uint32_t word = 0;
             word < layout.written_words; ++word) {
            const uint64_t observed = payload.words[word];
            const uint64_t expected = ExpectedPayloadWord(
                case_id, spec, layout, word
            );
            observed_hash = HashWord(observed_hash, observed);
            expected_hash = HashWord(expected_hash, expected);
            payload_valid = payload_valid && observed == expected;
        }
    }
    result.observed_checksum = observed_hash;
    result.expected_checksum = expected_hash;
    if (payload_valid && observed_hash == expected_hash) {
        result.status |= ProbePayloadValid;
    }

    if (!header_valid || !payload_valid ||
        observed_hash != expected_hash ||
        !TryMarkExecutionTokenEngineInflight<
            ReturnDependencyOps>(
            token, ProbeReadySource{}, state->protocol_fatal
        ) ||
        !TryMarkExecutionTokenCompleting<
            ReturnDependencyOps>(
            token, ProbeEngineCompletion{}, state->protocol_fatal
        )) {
        result.final_state = static_cast<uint64_t>(
            DeviceCommonOps::Load(
                &state->cells[case_id].control.state
            )
        );
        result.elapsed_ticks =
            static_cast<uint64_t>(get_sys_cnt()) - begin;
        PublishResult(&state->results[case_id], result);
        PublishFatal(state, case_id);
        return;
    }
    ProbeCompletionSink completion{};
    if (PublishExecDoneAfterCompletion<ReturnDependencyOps>(
            state->cells[case_id], token, completion,
            state->protocol_fatal
        ) == ExecDoneResult::Done) {
        result.status |= ProbeDone;
    }
    if (completion.vend_task != spec.task_id ||
        completion.flag_task != spec.task_id ||
        completion.vend != spec.completion_vend) {
        result.status &= ~ProbeDone;
    }
    result.final_state = static_cast<uint64_t>(
        DeviceCommonOps::Load(
            &state->cells[case_id].control.state
        )
    );
    if (token.control.phase == ExecTokenPhase::Idle) {
        result.status |= ProbeTokenIdle;
    }
    result.elapsed_ticks =
        static_cast<uint64_t>(get_sys_cnt()) - begin;
    PublishResult(&state->results[case_id], result);
    if (result.status != kProbeExpectedStatus) {
        PublishFatal(state, case_id);
    }
}

__aicore__ inline void RunCase(
    __gm__ ProbeState *state, uint32_t case_id,
    uint32_t participant
) {
    if (participant == BuilderForCase(case_id)) {
        BuildCase(state, case_id, participant);
    } else if (participant == ExecutorForCase(case_id)) {
        ExecuteCase(state, case_id, participant);
    }
}

__aicore__ inline void RunParticipant(
    __gm__ ProbeState *state, uint32_t participant,
    uint32_t role
) {
    if (state == nullptr || participant >= kProbeParticipants) {
        return;
    }
    DeviceCommonOps::InvalidateRegion(
        &state->control, sizeof(ProbeControl)
    );
    if (state->control.magic != kProbeMagic ||
        state->control.version != kProbeVersion ||
        state->control.case_count != kProbeCaseCount ||
        state->control.participant_count != kProbeParticipants) {
        PublishFatal(state, UINT32_MAX);
        return;
    }
    PublishParticipantResult(
        &state->participants[participant], participant, role
    );
    for (uint32_t case_id = 0;
         case_id < kProbeCaseCount; ++case_id) {
        if (FatalPublished(state)) return;
        RunCase(state, case_id, participant);
        if (!WaitCaseDone(state, case_id)) {
            PublishFatal(state, case_id);
            return;
        }
    }
}

}  // namespace

#if defined(__DAV_VEC__)
PTO_SYNCALL_MIX_AIC_KERNEL_META(
    cross_core_payload_probe_0_mix_aiv, 1, 1
);

extern "C" __global__ __aicore__ void
cross_core_payload_probe_0_mix_aiv(__gm__ ProbeState *state) {
    const uint32_t vector_id = static_cast<uint32_t>(
        get_block_idx() * get_subblockdim() + get_subblockid()
    );
    RunParticipant(state, vector_id * 2U + 1U, 1U);
}
#else
PTO_SYNCALL_MIX_AIC_KERNEL_META(
    cross_core_payload_probe_0_mix_aic, 1, 1
);

extern "C" __global__ __aicore__ void
cross_core_payload_probe_0_mix_aic(__gm__ ProbeState *state) {
    RunParticipant(
        state, static_cast<uint32_t>(get_block_idx()) * 2U, 0U
    );
}
#endif
