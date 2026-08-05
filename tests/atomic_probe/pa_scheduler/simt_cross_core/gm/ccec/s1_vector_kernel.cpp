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

// S1 固定启动一个 1 AIC + 2 AIV mixed block。AIV0 只启动 SIMT builder，
// AIV1 只 Claim 并执行一个真实 Vector add，AIC 只观察终态。

#include <pto/common/kernel_meta.hpp>

#include "cce_aicore_intrinsics.h"

#if defined(__DAV_VEC__)
#include <pto/pto-inst.hpp>
#include "simt_api/asc_simt.h"
#endif

#include "../common/s1_vector.h"

namespace {

using namespace pa_scheduler::simt_cross_core;
using namespace pa_scheduler::simt_cross_core::s1;

constexpr int kSingleCacheLine = 0;
constexpr int kCacheLineOut = 2;
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
        role, static_cast<uint32_t>(get_block_idx()), static_cast<uint32_t>(get_block_num()),
        static_cast<uint32_t>(get_coreid()) & 0x0FFFU, static_cast<uint32_t>(get_subblockid()),
        static_cast<uint32_t>(get_subblockdim())
    );
    result->launch_nonce = state->control.launch_nonce;
    result->visibility_mode = state->control.visibility_mode;
}

__aicore__ __attribute__((always_inline)) inline void
PublishRoleResult(__gm__ ProbeRoleResult *destination, const ProbeRoleResult &result) {
    // CCEC 对较大 Scalar 局部 aggregate 做标量替换后，其通用地址不再保证
    // 能按源码字段顺序线性遍历。逐字段 st_dev 既保留 128B ABI，也避免把
    // 编译器内部 spill 次序误当作 C++ 对象布局。
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
    // shared_protocol 的 constexpr encoder 同时承担 host 静态断言；CCEC
    // 不允许运行时从 __aicore__ 调用未标注地址空间的 constexpr 函数，
    // 因此这里按同一已锁定位布局就地编码，不引入第二套字段定义。
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

using namespace pto;

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
    uint64_t checksum = 0x243F6A8885A308D3ULL;
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

static __simt_vf__ __aicore__ LAUNCH_BOUND(kThreadCount) void S1SimtBuildVectorTask(
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

static __aicore__ __attribute__((noinline, used)) void
RunVectorAdd(__gm__ float *input_a, __gm__ float *input_b, __gm__ float *output) {
    constexpr int kRows = static_cast<int>(kTileRows);
    constexpr int kColumns = static_cast<int>(kTileColumns);
    using GlobalData = GlobalTensor<float, Shape<1, 1, 1, kRows, kColumns>, pto::Stride<1, 1, 1, kColumns, 1>>;
    using TileData = Tile<TileType::Vec, float, kRows, kColumns, BLayout::RowMajor, -1, -1>;

    GlobalData input_a_global(input_a);
    GlobalData input_b_global(input_b);
    GlobalData output_global(output);
    TileData input_a_tile(kRows, kColumns);
    TileData input_b_tile(kRows, kColumns);
    TileData output_tile(kRows, kColumns);
    TASSIGN(input_a_tile, 0x0);
    TASSIGN(input_b_tile, 0x10000);
    TASSIGN(output_tile, 0x20000);

    TLOAD(input_a_tile, input_a_global);
    TLOAD(input_b_tile, input_b_global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(output_tile, input_a_tile, input_b_tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(output_global, output_tile);
    // DONE 必须晚于 MTE3 写回 GM 的完成边界。
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

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

    cce::async_invoke<S1SimtBuildVectorTask>(
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

__aicore__ void RunExecutor(__gm__ ProbeState *state) {
    ProbeRoleResult result{};
    InitializeResult(&result, ProbeRole::Aiv1Executor, state);
    if (!ConfigValid(state)) {
        PublishFatal(state, ExecFatalReason::InvalidBuildInput, kExecutorOwner);
        result.fatal_state = LoadFatal(state);
        PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::Aiv1Executor)], result);
        return;
    }
    result.status |= kResultConfigValid;

    if (!WaitForBuilt(state, kExecutorOwner)) {
        result.final_state = ScalarAtomicLoad(&state->cell.state);
        result.fatal_state = LoadFatal(state);
        PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::Aiv1Executor)], result);
        return;
    }

    result.role_counts = PackRoleCounts(0U, 0U, 1U, 0U);
    result.claim_observed = ScalarCas(&state->cell.state, kBuiltState, kClaimedState);
    if (result.claim_observed != kBuiltState) {
        PublishFatal(state, ExecFatalReason::InvalidBuiltControl, kExecutorOwner);
        result.final_state = ScalarAtomicLoad(&state->cell.state);
        result.fatal_state = LoadFatal(state);
        PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::Aiv1Executor)], result);
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
    }

    if (payload_valid) {
        RunVectorAdd(
            reinterpret_cast<__gm__ float *>(payload[kPayloadInputAWord]),
            reinterpret_cast<__gm__ float *>(payload[kPayloadInputBWord]),
            reinterpret_cast<__gm__ float *>(payload[kPayloadOutputWord])
        );
        result.status |= kResultVectorExecuted;
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
    PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::Aiv1Executor)], result);
}

#endif  // defined(__DAV_VEC__)

__aicore__ void RunAicObserver(__gm__ ProbeState *state) {
    ProbeRoleResult result{};
    InitializeResult(&result, ProbeRole::AicObserver, state);
    if (!ConfigValid(state)) {
        PublishFatal(state, ExecFatalReason::InvalidBuildInput, 0U);
    } else {
        result.status |= kResultConfigValid;
        if (WaitForDone(state, 0U)) {
            result.status |= kResultWaitDone;
        }
    }
    result.final_state = ScalarAtomicLoad(&state->cell.state);
    result.fatal_state = LoadFatal(state);
    PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::AicObserver)], result);
}

}  // namespace

#if defined(__DAV_VEC__)

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_cross_core_s1_0_mix_aiv, 1, 2);

extern "C" __global__ __aicore__ void
simt_cross_core_s1_0_mix_aiv(__gm__ pa_scheduler::simt_cross_core::s1::ProbeState *state) {
    // host 每轮复用同一状态地址；配置输入不是 DCCI 对照对象，入口先显式失效。
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dsb(DSB_ALL);

    const uint32_t vector_id = static_cast<uint32_t>(get_block_idx() * get_subblockdim() + get_subblockid());
    if (vector_id == 0U) {
        RunBuilder(state);
    } else if (vector_id == 1U) {
        RunExecutor(state);
    }
}

#else

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_cross_core_s1_0_mix_aic, 1, 2);

extern "C" __global__ __aicore__ void
simt_cross_core_s1_0_mix_aic(__gm__ pa_scheduler::simt_cross_core::s1::ProbeState *state) {
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dsb(DSB_ALL);
    if (get_block_idx() == 0U) {
        RunAicObserver(state);
    }
}

#endif
