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

// S2 固定启动一个 1 AIC + 2 AIV mixed block。AIV0 只启动 SIMT builder，
// AIC 只 Claim 并执行一个真实 Cube matmul，AIV1 只观察终态。

#include <pto/common/kernel_meta.hpp>

#include "cce_aicore_intrinsics.h"

#if defined(__DAV_VEC__)
#include "simt_api/asc_simt.h"
#else
#include <pto/pto-inst.hpp>
#endif

#include "../common/s2_cube.h"

namespace {

using namespace pa_scheduler::simt_cross_core;
using namespace pa_scheduler::simt_cross_core::gm;
using namespace pa_scheduler::simt_cross_core::s2;

constexpr int kSingleCacheLine = 0;
constexpr uint32_t kFatalPollMask = 0x3FFU;

__aicore__ __attribute__((always_inline)) inline uint64_t LoadDev64(__gm__ uint64_t *address) {
    return static_cast<uint64_t>(__builtin_cce_ld_dev(address, 0));
}

__aicore__ __attribute__((always_inline)) inline void StoreDev64(__gm__ uint64_t *address, uint64_t value) {
    __builtin_cce_st_dev(value, address, 0);
}

__aicore__ __attribute__((always_inline)) inline uint64_t
ScalarCas(__gm__ uint64_t *address, uint64_t expected, uint64_t desired) {
    return atomicCAS(address, expected, desired);
}

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarAtomicLoad(__gm__ uint64_t *address) {
    return atomicAdd(address, static_cast<uint64_t>(0U));
}

__aicore__ __attribute__((always_inline)) inline bool ConfigValid(__gm__ const ProbeState *state) {
    return state->control.magic == kProbeMagic && state->control.version == kProbeVersion &&
           state->control.visibility_mode < kVisibilityModeCount && state->control.thread_count == kThreadCount &&
           state->control.element_count == kElementCount && state->control.task_id == kTaskId &&
           state->control.timeout_ticks != 0U;
}

__aicore__ __attribute__((always_inline)) inline void
InitializeResult(ProbeRoleResult *result, ProbeRole role, __gm__ const ProbeState *state) {
    result->magic = kResultMagic;
    result->identity = PackIdentity(
        static_cast<uint32_t>(role), static_cast<uint32_t>(get_block_idx()), static_cast<uint32_t>(get_block_num()),
        static_cast<uint32_t>(get_coreid()) & 0x0FFFU, static_cast<uint32_t>(get_subblockid()),
        static_cast<uint32_t>(get_subblockdim())
    );
    result->launch_nonce = state->control.launch_nonce;
    result->visibility_mode = state->control.visibility_mode;
}

__aicore__ __attribute__((always_inline)) inline void
PublishRoleResult(__gm__ ProbeRoleResult *destination, const ProbeRoleResult &result) {
    StoreDev64(&destination->magic, result.magic);
    StoreDev64(&destination->identity, result.identity);
    StoreDev64(&destination->status, result.status);
    StoreDev64(&destination->role_counts, result.role_counts);
    StoreDev64(&destination->launch_nonce, result.launch_nonce);
    StoreDev64(&destination->visibility_mode, result.visibility_mode);
    StoreDev64(&destination->reserve_observed, result.reserve_observed);
    StoreDev64(&destination->publish_observed, result.publish_observed);
    StoreDev64(&destination->claim_observed, result.claim_observed);
    StoreDev64(&destination->done_observed, result.done_observed);
    StoreDev64(&destination->final_state, result.final_state);
    StoreDev64(&destination->fatal_state, result.fatal_state);
    StoreDev64(&destination->observed_payload_checksum, result.observed_payload_checksum);
    StoreDev64(&destination->expected_payload_checksum, result.expected_payload_checksum);
    StoreDev64(&destination->observed_elements, result.observed_elements);
    StoreDev64(&destination->reserved, result.reserved);
    dsb(DSB_ALL);
}

__aicore__ __attribute__((always_inline)) inline uint64_t LoadFatal(__gm__ ProbeState *state) {
    return ScalarAtomicLoad(&state->fatal.state);
}

__aicore__ __attribute__((always_inline)) inline void
PublishFatal(__gm__ ProbeState *state, ExecFatalReason reason, uint32_t owner) {
    const uint64_t encoded = (static_cast<uint64_t>(reason) << kFatalReasonShift) |
                             (static_cast<uint64_t>(owner) << kFatalOwnerShift) |
                             (static_cast<uint64_t>(kTaskId) << kFatalTaskIdShift);
    (void)ScalarCas(&state->fatal.state, 0U, encoded);
}

__aicore__ bool WaitForDone(__gm__ ProbeState *state, uint32_t owner) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint32_t polls = 0U;
    while (true) {
        const uint64_t raw = ScalarAtomicLoad(&state->cell.state);
        if (raw == kDoneState) {
            return true;
        }
        if (raw != kEmptyState && raw != kBuildingState && raw != kBuiltState && raw != kClaimedState) {
            PublishFatal(state, ExecFatalReason::InvalidBuiltControl, owner);
            return false;
        }
        ++polls;
        if ((polls & kFatalPollMask) == 0U) {
            if (LoadFatal(state) != 0U) {
                return false;
            }
            if (static_cast<uint64_t>(get_sys_cnt()) - begin > state->control.timeout_ticks) {
                PublishFatal(state, ExecFatalReason::Timeout, owner);
                return false;
            }
        }
    }
}

__aicore__ bool WaitForBuilt(__gm__ ProbeState *state, uint32_t owner) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint32_t polls = 0U;
    while (true) {
        const uint64_t raw = ScalarAtomicLoad(&state->cell.state);
        if (raw == kBuiltState) {
            return true;
        }
        if (raw != kEmptyState && raw != kBuildingState) {
            PublishFatal(state, ExecFatalReason::InvalidBuiltControl, owner);
            return false;
        }
        ++polls;
        if ((polls & kFatalPollMask) == 0U) {
            if (LoadFatal(state) != 0U) {
                return false;
            }
            if (static_cast<uint64_t>(get_sys_cnt()) - begin > state->control.timeout_ticks) {
                PublishFatal(state, ExecFatalReason::Timeout, owner);
                return false;
            }
        }
    }
}

#if defined(__DAV_VEC__)

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtRotateLeft(uint64_t value, uint32_t shift) {
    return (value << shift) | (value >> (64U - shift));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtFoldDescriptorChecksum(uint64_t checksum, uint64_t word, uint32_t index) {
    return SimtRotateLeft(checksum ^ word ^ (0x9E3779B97F4A7C15ULL + index), (index % 19U) + 3U);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtComputePayloadChecksum(uint64_t nonce, uint64_t input_a, uint64_t input_b, uint64_t output) {
    uint64_t checksum = 0x13198A2E03707344ULL;
    checksum = SimtFoldDescriptorChecksum(checksum, kPayloadMagic, kPayloadMagicWord);
    checksum = SimtFoldDescriptorChecksum(checksum, kPayloadVersion, kPayloadVersionWord);
    checksum = SimtFoldDescriptorChecksum(checksum, nonce, kPayloadNonceWord);
    checksum = SimtFoldDescriptorChecksum(checksum, input_a, kPayloadInputAWord);
    checksum = SimtFoldDescriptorChecksum(checksum, input_b, kPayloadInputBWord);
    checksum = SimtFoldDescriptorChecksum(checksum, output, kPayloadOutputWord);
    checksum = SimtFoldDescriptorChecksum(
        checksum, (static_cast<uint64_t>(kTaskId) << 32U) | kElementCount, kPayloadShapeWord
    );
    return checksum;
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(kThreadCount) void S2SimtBuildCubeTask(
    __gm__ uint64_t *state_word, __gm__ uint64_t *payload_words, __gm__ uint64_t *report_words, uint64_t launch_nonce,
    uint64_t input_a, uint64_t input_b, uint64_t output, uint64_t visibility_mode
) {
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    if (thread >= kThreadCount || thread != 0U) {
        return;
    }

    const uint64_t reserve_observed = asc_atomic_cas(state_word, kEmptyState, kBuildingState);
    report_words[0] = reserve_observed;
    report_words[2] = static_cast<uint64_t>(blockDim.x);
    report_words[4] = launch_nonce;
    report_words[5] = thread;
    report_words[6] = 0U;
    if (reserve_observed != kEmptyState) {
        report_words[1] = reserve_observed;
        report_words[3] = 0U;
        asc_threadfence();
        return;
    }

    payload_words[kPayloadMagicWord] = kPayloadMagic;
    payload_words[kPayloadVersionWord] = kPayloadVersion;
    payload_words[kPayloadNonceWord] = launch_nonce;
    payload_words[kPayloadInputAWord] = input_a;
    payload_words[kPayloadInputBWord] = input_b;
    payload_words[kPayloadOutputWord] = output;
    payload_words[kPayloadShapeWord] = (static_cast<uint64_t>(kTaskId) << 32U) | kElementCount;
    payload_words[kPayloadChecksumWord] = SimtComputePayloadChecksum(launch_nonce, input_a, input_b, output);
    report_words[3] = kPayloadWords;

    const auto mode = static_cast<VisibilityMode>(visibility_mode);
    if (mode == VisibilityMode::WriterDcci || mode == VisibilityMode::WriterAndReaderDcci) {
        asc_dcci_single(static_cast<__gm__ void *>(payload_words));
        report_words[6] = 1U;
    }
    asc_threadfence();

    const uint64_t publish_observed = asc_atomic_cas(state_word, kBuildingState, kBuiltState);
    report_words[1] = publish_observed;
    asc_threadfence();
}

// S2 的 AIV 角色运行时不执行 SIMD task；这个本地、永不进入的 companion
// 只让 compiler 为 mixed entry 生成 SIMD_SIMT_MIX_VF metadata，避免把
// 同一调度入口标成 SIMT_VF_ONLY。动态条件来自 GM，防止编译期删除。
static __simd_vf__ __aicore__ void S2SimdMetadataAnchor(__ubuf__ uint32_t *scratch) { scratch[0] = scratch[0] + 1U; }

__aicore__ void RunBuilder(__gm__ ProbeState *state) {
    ProbeRoleResult result{};
    InitializeResult(&result, ProbeRole::Aiv0Builder, state);
    if (!ConfigValid(state)) {
        PublishFatal(state, ExecFatalReason::InvalidBuildInput, kBuilderOwner);
        result.fatal_state = LoadFatal(state);
        PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::Aiv0Builder)], result);
        return;
    }
    result.status |= kResultConfigValid;

    cce::async_invoke<S2SimtBuildCubeTask>(
        cce::dim3{kThreadCount, 1U, 1U}, &state->cell.state, &state->payload.words[0],
        reinterpret_cast<__gm__ uint64_t *>(&state->simt_report), state->control.launch_nonce,
        reinterpret_cast<uint64_t>(&state->input_a.values[0]), reinterpret_cast<uint64_t>(&state->input_b.values[0]),
        reinterpret_cast<uint64_t>(&state->output.values[0]), state->control.visibility_mode
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    result.reserve_observed = LoadDev64(&state->simt_report.reserve_observed);
    result.publish_observed = LoadDev64(&state->simt_report.publish_observed);
    const bool build_won = result.reserve_observed == kEmptyState && result.publish_observed == kBuildingState;
    result.role_counts = PackRoleCounts(1U, build_won ? 1U : 0U, 0U, 0U);
    if (!build_won) {
        PublishFatal(state, ExecFatalReason::PublishConflict, kBuilderOwner);
    }
    if (WaitForDone(state, kBuilderOwner)) {
        result.status |= kResultWaitDone;
    }
    result.final_state = ScalarAtomicLoad(&state->cell.state);
    result.fatal_state = LoadFatal(state);
    PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::Aiv0Builder)], result);
}

__aicore__ void RunAivObserver(__gm__ ProbeState *state) {
    ProbeRoleResult result{};
    InitializeResult(&result, ProbeRole::Aiv1Observer, state);
    if (!ConfigValid(state)) {
        PublishFatal(state, ExecFatalReason::InvalidBuildInput, kObserverOwner);
    } else {
        result.status |= kResultConfigValid;
        if (WaitForDone(state, kObserverOwner)) {
            result.status |= kResultWaitDone;
        }
    }
    result.final_state = ScalarAtomicLoad(&state->cell.state);
    result.fatal_state = LoadFatal(state);
    PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::Aiv1Observer)], result);
}

#else

using namespace pto;

static __aicore__ __attribute__((noinline, used)) void
RunCubeMatmul(__gm__ float *input_a, __gm__ float *input_b, __gm__ float *output) {
    constexpr int kRows = static_cast<int>(kTileRows);
    constexpr int kColumns = static_cast<int>(kTileColumns);
    constexpr int kBlockAlign = C0_SIZE_BYTE / sizeof(float);
    static_assert(kRows % 16 == 0, "S2 Cube M must be 16-aligned");
    static_assert(kColumns % kBlockAlign == 0, "S2 Cube K/N must satisfy C0 alignment");

    using GlobalData = GlobalTensor<
        float, Shape<1, 1, 1, kRows, kColumns>,
        pto::Stride<kRows * kColumns, kRows * kColumns, kRows * kColumns, kColumns, 1>>;
    using TileMatA =
        Tile<TileType::Mat, float, kRows, kColumns, BLayout::ColMajor, kRows, kColumns, SLayout::RowMajor, 512>;
    using TileMatB =
        Tile<TileType::Mat, float, kRows, kColumns, BLayout::ColMajor, kRows, kColumns, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<float, kRows, kColumns, kRows, kColumns>;
    using RightTile = TileRight<float, kRows, kColumns, kRows, kColumns>;
    using AccTile = TileAcc<float, kRows, kColumns, kRows, kColumns>;

    GlobalData input_a_global(input_a);
    GlobalData input_b_global(input_b);
    GlobalData output_global(output);
    TileMatA input_a_mat;
    TileMatB input_b_mat;
    LeftTile input_a_l0;
    RightTile input_b_l0;
    AccTile output_l0;
    TASSIGN(input_a_mat, 0x0);
    TASSIGN(input_b_mat, 0x20000);
    TASSIGN(input_a_l0, 0x0);
    TASSIGN(input_b_l0, 0x0);
    TASSIGN(output_l0, 0x0);

    TLOAD(input_a_mat, input_a_global);
    TLOAD(input_b_mat, input_b_global);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(input_a_l0, input_a_mat);
    TMOV(input_b_l0, input_b_mat);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(output_l0, input_a_l0, input_b_l0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(output_global, output_l0);
    // DONE 必须晚于 FIX 写回 GM 的完成边界。
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
}

__aicore__ void RunAicExecutor(__gm__ ProbeState *state) {
    ProbeRoleResult result{};
    InitializeResult(&result, ProbeRole::AicExecutor, state);
    if (!ConfigValid(state)) {
        PublishFatal(state, ExecFatalReason::InvalidBuildInput, kExecutorOwner);
        result.fatal_state = LoadFatal(state);
        PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::AicExecutor)], result);
        return;
    }
    result.status |= kResultConfigValid;

    if (!WaitForBuilt(state, kExecutorOwner)) {
        result.final_state = ScalarAtomicLoad(&state->cell.state);
        result.fatal_state = LoadFatal(state);
        PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::AicExecutor)], result);
        return;
    }

    result.role_counts = PackRoleCounts(0U, 0U, 1U, 0U);
    result.claim_observed = ScalarCas(&state->cell.state, kBuiltState, kClaimedState);
    if (result.claim_observed != kBuiltState) {
        PublishFatal(state, ExecFatalReason::InvalidBuiltControl, kExecutorOwner);
        result.final_state = ScalarAtomicLoad(&state->cell.state);
        result.fatal_state = LoadFatal(state);
        PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::AicExecutor)], result);
        return;
    }
    result.role_counts = PackRoleCounts(0U, 0U, 1U, 1U);

    const auto mode = static_cast<VisibilityMode>(state->control.visibility_mode);
    if (ReaderUsesDcci(mode)) {
        dcci(static_cast<__gm__ void *>(&state->payload), kSingleCacheLine);
        dsb(DSB_ALL);
    }

    uint64_t payload[kPayloadWords]{};
    __gm__ volatile uint64_t *source = &state->payload.words[0];
    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        payload[word] = source[word];
    }
    const uint64_t expected_input_a = reinterpret_cast<uint64_t>(&state->input_a.values[0]);
    const uint64_t expected_input_b = reinterpret_cast<uint64_t>(&state->input_b.values[0]);
    const uint64_t expected_output = reinterpret_cast<uint64_t>(&state->output.values[0]);
    result.observed_payload_checksum = payload[kPayloadChecksumWord];
    result.expected_payload_checksum =
        ComputePayloadChecksum(state->control.launch_nonce, expected_input_a, expected_input_b, expected_output);
    result.observed_elements = PayloadElementCount(payload[kPayloadShapeWord]);
    const bool payload_valid =
        payload[kPayloadMagicWord] == kPayloadMagic && payload[kPayloadVersionWord] == kPayloadVersion &&
        payload[kPayloadNonceWord] == state->control.launch_nonce && payload[kPayloadInputAWord] == expected_input_a &&
        payload[kPayloadInputBWord] == expected_input_b && payload[kPayloadOutputWord] == expected_output &&
        PayloadTaskId(payload[kPayloadShapeWord]) == kTaskId &&
        PayloadElementCount(payload[kPayloadShapeWord]) == kElementCount &&
        payload[kPayloadChecksumWord] == result.expected_payload_checksum;
    if (payload_valid) {
        result.status |= kResultPayloadValid;
        RunCubeMatmul(
            reinterpret_cast<__gm__ float *>(payload[kPayloadInputAWord]),
            reinterpret_cast<__gm__ float *>(payload[kPayloadInputBWord]),
            reinterpret_cast<__gm__ float *>(payload[kPayloadOutputWord])
        );
        result.status |= kResultTaskExecuted;
    }

    result.done_observed = ScalarCas(&state->cell.state, kClaimedState, kDoneState);
    const bool done_won = result.done_observed == kClaimedState;
    if (done_won) {
        result.status |= kResultWaitDone;
    } else {
        PublishFatal(state, ExecFatalReason::CompletionConflict, kExecutorOwner);
    }
    result.final_state = ScalarAtomicLoad(&state->cell.state);
    result.fatal_state = LoadFatal(state);
    PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::AicExecutor)], result);
}

#endif  // defined(__DAV_VEC__)

}  // namespace

#if defined(__DAV_VEC__)

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_cross_core_s2_0_mix_aiv, 1, 2);

extern "C" __global__ __aicore__ void
simt_cross_core_s2_0_mix_aiv(__gm__ pa_scheduler::simt_cross_core::s2::ProbeState *state) {
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dsb(DSB_ALL);
    if (state->control.version == UINT64_MAX) {
        S2SimdMetadataAnchor(reinterpret_cast<__ubuf__ uint32_t *>(0));
    }

    const uint32_t vector_id = static_cast<uint32_t>(get_block_idx() * get_subblockdim() + get_subblockid());
    if (vector_id == 0U) {
        RunBuilder(state);
    } else if (vector_id == 1U) {
        RunAivObserver(state);
    }
}

#else

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_cross_core_s2_0_mix_aic, 1, 2);

extern "C" __global__ __aicore__ void
simt_cross_core_s2_0_mix_aic(__gm__ pa_scheduler::simt_cross_core::s2::ProbeState *state) {
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dsb(DSB_ALL);
    if (get_block_idx() == 0U) {
        RunAicExecutor(state);
    }
}

#endif
