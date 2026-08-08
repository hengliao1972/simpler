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

// S3 在同一个 1 AIC + 2 AIV mixed block 内由 AIV0 的两个 SIMT thread
// 分别发布 Vector/Cube task，AIV1 和 AIC 只执行各自兼容的 task。

#include <pto/common/kernel_meta.hpp>
#include <pto/pto-inst.hpp>

#include "cce_aicore_intrinsics.h"

#if defined(__DAV_VEC__)
#include "simt_api/asc_simt.h"
#endif

#include "../common/s3_dual_task.h"

namespace {

using namespace pa_scheduler::simt_cross_core;
using namespace pa_scheduler::simt_cross_core::gm;
using namespace pa_scheduler::simt_cross_core::s3;
using namespace pto;

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
           state->control.visibility_mode == kRequiredVisibilityMode && state->control.thread_count == kThreadCount &&
           state->control.element_count == kElementCount && state->control.task_id == kTaskCount &&
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
PublishFatal(__gm__ ProbeState *state, ExecFatalReason reason, uint32_t owner, uint32_t task_id) {
    const uint64_t encoded = (static_cast<uint64_t>(reason) << kFatalReasonShift) |
                             (static_cast<uint64_t>(owner) << kFatalOwnerShift) |
                             (static_cast<uint64_t>(task_id) << kFatalTaskIdShift);
    (void)ScalarCas(&state->fatal.state, 0U, encoded);
}

__aicore__ bool WaitForBuilt(
    __gm__ ProbeState *state, uint32_t task_index, uint64_t building_state, uint64_t built_state, uint32_t owner
) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint32_t polls = 0U;
    while (true) {
        const uint64_t raw = ScalarAtomicLoad(&state->tasks[task_index].cell.state);
        if (raw == built_state) {
            return true;
        }
        if (raw != kEmptyState && raw != building_state) {
            PublishFatal(state, ExecFatalReason::InvalidBuiltControl, owner, TaskId(task_index));
            return false;
        }
        ++polls;
        if ((polls & kFatalPollMask) == 0U) {
            if (LoadFatal(state) != 0U) {
                return false;
            }
            if (static_cast<uint64_t>(get_sys_cnt()) - begin > state->control.timeout_ticks) {
                PublishFatal(state, ExecFatalReason::Timeout, owner, TaskId(task_index));
                return false;
            }
        }
    }
}

__aicore__ bool WaitForDrain(__gm__ ProbeState *state, uint32_t owner, uint32_t task_id) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint32_t polls = 0U;
    while (true) {
        const uint64_t builder_finished = ScalarAtomicLoad(&state->drain.builder_finished);
        const uint64_t done_count = ScalarAtomicLoad(&state->drain.done_count);
        if (builder_finished == 1U && done_count == kTaskCount) {
            return true;
        }
        if (builder_finished > 1U || done_count > kTaskCount) {
            PublishFatal(state, ExecFatalReason::CompletionConflict, owner, task_id);
            return false;
        }
        ++polls;
        if ((polls & kFatalPollMask) == 0U) {
            if (LoadFatal(state) != 0U) {
                return false;
            }
            if (static_cast<uint64_t>(get_sys_cnt()) - begin > state->control.timeout_ticks) {
                PublishFatal(state, ExecFatalReason::Timeout, owner, task_id);
                return false;
            }
        }
    }
}

__aicore__ __attribute__((always_inline)) inline void CompleteTask(
    __gm__ ProbeState *state, uint32_t task_index, uint64_t claimed_state, uint64_t done_state, uint32_t owner,
    ProbeRoleResult *result
) {
    result->done_observed = ScalarCas(&state->tasks[task_index].cell.state, claimed_state, done_state);
    if (result->done_observed != claimed_state) {
        PublishFatal(state, ExecFatalReason::CompletionConflict, owner, TaskId(task_index));
        return;
    }
    const uint64_t previous_done = atomicAdd(&state->drain.done_count, static_cast<uint64_t>(1U));
    if (previous_done >= kTaskCount) {
        PublishFatal(state, ExecFatalReason::CompletionConflict, owner, TaskId(task_index));
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
SimtComputePayloadChecksum(uint32_t task_index, uint64_t nonce, uint64_t input_a, uint64_t input_b, uint64_t output) {
    const uint64_t magic = task_index == kVectorTaskIndex ? kVectorPayloadMagic : kCubePayloadMagic;
    const uint32_t task_id = task_index == kVectorTaskIndex ? kVectorTaskId : kCubeTaskId;
    uint64_t checksum = task_index == kVectorTaskIndex ? 0xA4093822299F31D0ULL : 0x082EFA98EC4E6C89ULL;
    checksum = SimtFoldDescriptorChecksum(checksum, magic, kPayloadMagicWord);
    checksum = SimtFoldDescriptorChecksum(checksum, kPayloadVersion, kPayloadVersionWord);
    checksum = SimtFoldDescriptorChecksum(checksum, nonce, kPayloadNonceWord);
    checksum = SimtFoldDescriptorChecksum(checksum, input_a, kPayloadInputAWord);
    checksum = SimtFoldDescriptorChecksum(checksum, input_b, kPayloadInputBWord);
    checksum = SimtFoldDescriptorChecksum(checksum, output, kPayloadOutputWord);
    checksum = SimtFoldDescriptorChecksum(
        checksum, (static_cast<uint64_t>(task_id) << 32U) | kElementCount, kPayloadShapeWord
    );
    return checksum;
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(kThreadCount) void S3SimtBuildTasks(
    __gm__ uint64_t *task_words, __gm__ uint64_t *report_words, uint64_t launch_nonce, uint64_t vector_input_a,
    uint64_t vector_input_b, uint64_t vector_output, uint64_t cube_input_a, uint64_t cube_input_b, uint64_t cube_output
) {
    const uint32_t task_index = static_cast<uint32_t>(threadIdx.x);
    if (task_index >= kTaskCount) {
        return;
    }

    constexpr uint32_t kTaskStrideWords = sizeof(ProbeTaskSlot) / sizeof(uint64_t);
    constexpr uint32_t kPayloadOffsetWords = offsetof(ProbeTaskSlot, payload) / sizeof(uint64_t);
    constexpr uint32_t kReportStrideWords = sizeof(ProbeSimtReport) / sizeof(uint64_t);
    __gm__ uint64_t *state_word = task_words + task_index * kTaskStrideWords;
    __gm__ uint64_t *payload_words = state_word + kPayloadOffsetWords;
    __gm__ uint64_t *report = report_words + task_index * kReportStrideWords;
    const bool vector = task_index == kVectorTaskIndex;
    const uint64_t building_state = vector ? kVectorBuildingState : kCubeBuildingState;
    const uint64_t built_state = vector ? kVectorBuiltState : kCubeBuiltState;
    const uint64_t magic = vector ? kVectorPayloadMagic : kCubePayloadMagic;
    const uint32_t task_id = vector ? kVectorTaskId : kCubeTaskId;
    const uint64_t input_a = vector ? vector_input_a : cube_input_a;
    const uint64_t input_b = vector ? vector_input_b : cube_input_b;
    const uint64_t output = vector ? vector_output : cube_output;

    const uint64_t reserve_observed = asc_atomic_cas(state_word, kEmptyState, building_state);
    report[0] = reserve_observed;
    report[2] = static_cast<uint64_t>(blockDim.x);
    report[4] = launch_nonce;
    report[5] = task_index;
    report[6] = 0U;
    if (reserve_observed != kEmptyState) {
        report[1] = reserve_observed;
        report[3] = 0U;
        asc_threadfence();
        return;
    }

    payload_words[kPayloadMagicWord] = magic;
    payload_words[kPayloadVersionWord] = kPayloadVersion;
    payload_words[kPayloadNonceWord] = launch_nonce;
    payload_words[kPayloadInputAWord] = input_a;
    payload_words[kPayloadInputBWord] = input_b;
    payload_words[kPayloadOutputWord] = output;
    payload_words[kPayloadShapeWord] = (static_cast<uint64_t>(task_id) << 32U) | kElementCount;
    payload_words[kPayloadChecksumWord] =
        SimtComputePayloadChecksum(task_index, launch_nonce, input_a, input_b, output);
    report[3] = kPayloadWords;
    asc_threadfence();

    const uint64_t publish_observed = asc_atomic_cas(state_word, building_state, built_state);
    report[1] = publish_observed;
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
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

__aicore__ void RunBuilder(__gm__ ProbeState *state) {
    ProbeRoleResult result{};
    InitializeResult(&result, ProbeRole::Aiv0Builder, state);
    if (!ConfigValid(state)) {
        PublishFatal(state, ExecFatalReason::InvalidBuildInput, kBuilderOwner, 0U);
        result.fatal_state = LoadFatal(state);
        PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::Aiv0Builder)], result);
        return;
    }
    result.status |= kResultConfigValid;

    cce::async_invoke<S3SimtBuildTasks>(
        cce::dim3{kThreadCount, 1U, 1U}, reinterpret_cast<__gm__ uint64_t *>(&state->tasks[0]),
        reinterpret_cast<__gm__ uint64_t *>(&state->simt_reports[0]), state->control.launch_nonce,
        reinterpret_cast<uint64_t>(&state->vector_input_a.values[0]),
        reinterpret_cast<uint64_t>(&state->vector_input_b.values[0]),
        reinterpret_cast<uint64_t>(&state->vector_output.values[0]),
        reinterpret_cast<uint64_t>(&state->cube_input_a.values[0]),
        reinterpret_cast<uint64_t>(&state->cube_input_b.values[0]),
        reinterpret_cast<uint64_t>(&state->cube_output.values[0])
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    uint32_t build_wins = 0U;
    for (uint32_t task_index = 0U; task_index < kTaskCount; ++task_index) {
        const uint64_t reserve = LoadDev64(&state->simt_reports[task_index].reserve_observed);
        const uint64_t publish = LoadDev64(&state->simt_reports[task_index].publish_observed);
        const uint64_t expected_building = task_index == kVectorTaskIndex ? kVectorBuildingState : kCubeBuildingState;
        if (reserve == kEmptyState && publish == expected_building) {
            ++build_wins;
        } else {
            PublishFatal(state, ExecFatalReason::PublishConflict, kBuilderOwner, TaskId(task_index));
        }
    }
    result.role_counts = PackRoleCounts(kTaskCount, build_wins, 0U, 0U);
    const uint64_t builder_finish_old = ScalarCas(&state->drain.builder_finished, 0U, 1U);
    if (builder_finish_old != 0U) {
        PublishFatal(state, ExecFatalReason::CompletionConflict, kBuilderOwner, 0U);
    }
    if (WaitForDrain(state, kBuilderOwner, 0U)) {
        result.status |= kResultWaitDone;
    }
    result.final_state = ScalarAtomicLoad(&state->drain.done_count);
    result.fatal_state = LoadFatal(state);
    PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::Aiv0Builder)], result);
}

__aicore__ void RunVectorExecutor(__gm__ ProbeState *state) {
    ProbeRoleResult result{};
    InitializeResult(&result, ProbeRole::Aiv1VectorExecutor, state);
    if (!ConfigValid(state)) {
        PublishFatal(state, ExecFatalReason::InvalidBuildInput, kAivExecutorOwner, kVectorTaskId);
        result.fatal_state = LoadFatal(state);
        PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::Aiv1VectorExecutor)], result);
        return;
    }
    result.status |= kResultConfigValid;

    if (!WaitForBuilt(state, kVectorTaskIndex, kVectorBuildingState, kVectorBuiltState, kAivExecutorOwner)) {
        result.final_state = ScalarAtomicLoad(&state->tasks[kVectorTaskIndex].cell.state);
        result.fatal_state = LoadFatal(state);
        PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::Aiv1VectorExecutor)], result);
        return;
    }
    result.role_counts = PackRoleCounts(0U, 0U, 1U, 0U);
    result.claim_observed =
        ScalarCas(&state->tasks[kVectorTaskIndex].cell.state, kVectorBuiltState, kVectorClaimedState);
    if (result.claim_observed != kVectorBuiltState) {
        PublishFatal(state, ExecFatalReason::InvalidBuiltControl, kAivExecutorOwner, kVectorTaskId);
        result.final_state = ScalarAtomicLoad(&state->tasks[kVectorTaskIndex].cell.state);
        result.fatal_state = LoadFatal(state);
        PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::Aiv1VectorExecutor)], result);
        return;
    }
    result.role_counts = PackRoleCounts(0U, 0U, 1U, 1U);
    dcci(static_cast<__gm__ void *>(&state->tasks[kVectorTaskIndex].payload), kSingleCacheLine);
    dsb(DSB_ALL);

    uint64_t payload[kPayloadWords]{};
    __gm__ volatile uint64_t *source = &state->tasks[kVectorTaskIndex].payload.words[0];
    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        payload[word] = source[word];
    }
    const uint64_t expected_input_a = reinterpret_cast<uint64_t>(&state->vector_input_a.values[0]);
    const uint64_t expected_input_b = reinterpret_cast<uint64_t>(&state->vector_input_b.values[0]);
    const uint64_t expected_output = reinterpret_cast<uint64_t>(&state->vector_output.values[0]);
    result.observed_payload_checksum = payload[kPayloadChecksumWord];
    result.expected_payload_checksum = ComputePayloadChecksum(
        kVectorTaskIndex, state->control.launch_nonce, expected_input_a, expected_input_b, expected_output
    );
    result.observed_elements = PayloadElementCount(payload[kPayloadShapeWord]);
    const bool payload_valid =
        payload[kPayloadMagicWord] == kVectorPayloadMagic && payload[kPayloadVersionWord] == kPayloadVersion &&
        payload[kPayloadNonceWord] == state->control.launch_nonce && payload[kPayloadInputAWord] == expected_input_a &&
        payload[kPayloadInputBWord] == expected_input_b && payload[kPayloadOutputWord] == expected_output &&
        PayloadTaskId(payload[kPayloadShapeWord]) == kVectorTaskId &&
        PayloadElementCount(payload[kPayloadShapeWord]) == kElementCount &&
        payload[kPayloadChecksumWord] == result.expected_payload_checksum;
    if (payload_valid) {
        result.status |= kResultPayloadValid;
        RunVectorAdd(
            reinterpret_cast<__gm__ float *>(payload[kPayloadInputAWord]),
            reinterpret_cast<__gm__ float *>(payload[kPayloadInputBWord]),
            reinterpret_cast<__gm__ float *>(payload[kPayloadOutputWord])
        );
        result.status |= kResultTaskExecuted;
    } else {
        PublishFatal(state, ExecFatalReason::InvalidPayload, kAivExecutorOwner, kVectorTaskId);
    }
    CompleteTask(state, kVectorTaskIndex, kVectorClaimedState, kVectorDoneState, kAivExecutorOwner, &result);
    if (WaitForDrain(state, kAivExecutorOwner, kVectorTaskId)) {
        result.status |= kResultWaitDone;
    }
    result.final_state = ScalarAtomicLoad(&state->tasks[kVectorTaskIndex].cell.state);
    result.fatal_state = LoadFatal(state);
    PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::Aiv1VectorExecutor)], result);
}

#else

static __aicore__ __attribute__((noinline, used)) void
RunCubeMatmul(__gm__ float *input_a, __gm__ float *input_b, __gm__ float *output) {
    constexpr int kRows = static_cast<int>(kTileRows);
    constexpr int kColumns = static_cast<int>(kTileColumns);
    constexpr int kBlockAlign = C0_SIZE_BYTE / sizeof(float);
    static_assert(kRows % 16 == 0, "S3 Cube M must be 16-aligned");
    static_assert(kColumns % kBlockAlign == 0, "S3 Cube K/N must satisfy C0 alignment");

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
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
}

__aicore__ void RunCubeExecutor(__gm__ ProbeState *state) {
    ProbeRoleResult result{};
    InitializeResult(&result, ProbeRole::AicCubeExecutor, state);
    if (!ConfigValid(state)) {
        PublishFatal(state, ExecFatalReason::InvalidBuildInput, kAicExecutorOwner, kCubeTaskId);
        result.fatal_state = LoadFatal(state);
        PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::AicCubeExecutor)], result);
        return;
    }
    result.status |= kResultConfigValid;

    if (!WaitForBuilt(state, kCubeTaskIndex, kCubeBuildingState, kCubeBuiltState, kAicExecutorOwner)) {
        result.final_state = ScalarAtomicLoad(&state->tasks[kCubeTaskIndex].cell.state);
        result.fatal_state = LoadFatal(state);
        PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::AicCubeExecutor)], result);
        return;
    }
    result.role_counts = PackRoleCounts(0U, 0U, 1U, 0U);
    result.claim_observed = ScalarCas(&state->tasks[kCubeTaskIndex].cell.state, kCubeBuiltState, kCubeClaimedState);
    if (result.claim_observed != kCubeBuiltState) {
        PublishFatal(state, ExecFatalReason::InvalidBuiltControl, kAicExecutorOwner, kCubeTaskId);
        result.final_state = ScalarAtomicLoad(&state->tasks[kCubeTaskIndex].cell.state);
        result.fatal_state = LoadFatal(state);
        PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::AicCubeExecutor)], result);
        return;
    }
    result.role_counts = PackRoleCounts(0U, 0U, 1U, 1U);
    dcci(static_cast<__gm__ void *>(&state->tasks[kCubeTaskIndex].payload), kSingleCacheLine);
    dsb(DSB_ALL);

    uint64_t payload[kPayloadWords]{};
    __gm__ volatile uint64_t *source = &state->tasks[kCubeTaskIndex].payload.words[0];
    for (uint32_t word = 0U; word < kPayloadWords; ++word) {
        payload[word] = source[word];
    }
    const uint64_t expected_input_a = reinterpret_cast<uint64_t>(&state->cube_input_a.values[0]);
    const uint64_t expected_input_b = reinterpret_cast<uint64_t>(&state->cube_input_b.values[0]);
    const uint64_t expected_output = reinterpret_cast<uint64_t>(&state->cube_output.values[0]);
    result.observed_payload_checksum = payload[kPayloadChecksumWord];
    result.expected_payload_checksum = ComputePayloadChecksum(
        kCubeTaskIndex, state->control.launch_nonce, expected_input_a, expected_input_b, expected_output
    );
    result.observed_elements = PayloadElementCount(payload[kPayloadShapeWord]);
    const bool payload_valid =
        payload[kPayloadMagicWord] == kCubePayloadMagic && payload[kPayloadVersionWord] == kPayloadVersion &&
        payload[kPayloadNonceWord] == state->control.launch_nonce && payload[kPayloadInputAWord] == expected_input_a &&
        payload[kPayloadInputBWord] == expected_input_b && payload[kPayloadOutputWord] == expected_output &&
        PayloadTaskId(payload[kPayloadShapeWord]) == kCubeTaskId &&
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
    } else {
        PublishFatal(state, ExecFatalReason::InvalidPayload, kAicExecutorOwner, kCubeTaskId);
    }
    CompleteTask(state, kCubeTaskIndex, kCubeClaimedState, kCubeDoneState, kAicExecutorOwner, &result);
    if (WaitForDrain(state, kAicExecutorOwner, kCubeTaskId)) {
        result.status |= kResultWaitDone;
    }
    result.final_state = ScalarAtomicLoad(&state->tasks[kCubeTaskIndex].cell.state);
    result.fatal_state = LoadFatal(state);
    PublishRoleResult(&state->roles[static_cast<uint32_t>(ProbeRole::AicCubeExecutor)], result);
}

#endif  // defined(__DAV_VEC__)

}  // namespace

#if defined(__DAV_VEC__)

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_cross_core_s3_0_mix_aiv, 1, 2);

extern "C" __global__ __aicore__ void
simt_cross_core_s3_0_mix_aiv(__gm__ pa_scheduler::simt_cross_core::s3::ProbeState *state) {
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dsb(DSB_ALL);
    const uint32_t vector_id = static_cast<uint32_t>(get_block_idx() * get_subblockdim() + get_subblockid());
    if (vector_id == 0U) {
        RunBuilder(state);
    } else if (vector_id == 1U) {
        RunVectorExecutor(state);
    }
}

#else

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_cross_core_s3_0_mix_aic, 1, 2);

extern "C" __global__ __aicore__ void
simt_cross_core_s3_0_mix_aic(__gm__ pa_scheduler::simt_cross_core::s3::ProbeState *state) {
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dsb(DSB_ALL);
    if (get_block_idx() == 0U) {
        RunCubeExecutor(state);
    }
}

#endif
