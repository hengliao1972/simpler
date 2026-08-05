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

// G0/G1 keep the PA task ABI and task graph, but move every
// Build/ordered-insert operation into 64 independent SIMT warp leaders on
// each configured builder AIV. Main Scalar only launches the VF and
// participates in the global drain.

#include <pto/common/kernel_meta.hpp>
#include <pto/pto-inst.hpp>

#include "cce_aicore_intrinsics.h"

#if defined(__DAV_VEC__)
#include "simt_api/asc_simt.h"
#endif

#include "../common/g0_full_pa.h"
#include "full_pa_workloads.h"

namespace {

using namespace pa_scheduler::simt_cross_core::g0;
using namespace pa_scheduler::simt_cross_core::g0::device;
using namespace pto;

constexpr int kSingleCacheLine = 0;
constexpr uint32_t kWatchdogMask = 0x3FFU;
constexpr uint32_t kDrainExpectedArrivals = 6U;

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarAtomicLoad(__gm__ volatile int64_t *address) {
    return static_cast<uint64_t>(atomicAdd(const_cast<__gm__ int64_t *>(address), static_cast<int64_t>(0)));
}

__aicore__ __attribute__((always_inline)) inline uint64_t ScalarAtomicLoad(__gm__ volatile uint64_t *address) {
    return atomicAdd(const_cast<__gm__ uint64_t *>(address), static_cast<uint64_t>(0U));
}

__aicore__ __attribute__((always_inline)) inline uint64_t
ScalarCas(__gm__ volatile int64_t *address, uint64_t expected, uint64_t desired) {
    return static_cast<uint64_t>(
        atomicCAS(const_cast<__gm__ int64_t *>(address), static_cast<int64_t>(expected), static_cast<int64_t>(desired))
    );
}

__aicore__ __attribute__((always_inline)) inline uint64_t
ScalarFetchAdd(__gm__ volatile int64_t *address, uint64_t increment) {
    return static_cast<uint64_t>(atomicAdd(const_cast<__gm__ int64_t *>(address), static_cast<int64_t>(increment)));
}

__aicore__ __attribute__((always_inline)) inline uint64_t
ScalarExchange(__gm__ volatile int64_t *address, uint64_t value) {
    return static_cast<uint64_t>(atomicExch(const_cast<__gm__ int64_t *>(address), static_cast<int64_t>(value)));
}

__aicore__ __attribute__((always_inline)) inline uint64_t LoadDev64(__gm__ const uint64_t *address) {
    return static_cast<uint64_t>(__builtin_cce_ld_dev(const_cast<__gm__ uint64_t *>(address), 0));
}

__aicore__ __attribute__((always_inline)) inline void StoreDev64(__gm__ uint64_t *address, uint64_t value) {
    __builtin_cce_st_dev(value, address, 0);
}

__aicore__ __attribute__((always_inline)) inline uint64_t LoadFatal(__gm__ FullPaState *state) {
    return ScalarAtomicLoad(&state->fatal.state);
}

__aicore__ __attribute__((always_inline)) inline void
PublishFatal(__gm__ FullPaState *state, ExecFatalReason reason, uint32_t owner, uint32_t task_id) {
    (void)ScalarCas(&state->fatal.state, 0U, EncodeExecFatal(reason, owner, task_id));
}

__aicore__ __attribute__((always_inline)) inline bool ConfigValid(__gm__ const FullPaState *state) {
    const uint32_t batches = state->control.batch_count;
    return state->control.magic == kProbeMagic && state->control.version == kProbeVersion &&
           state->control.timeout_ticks != 0U && batches >= 1U && batches <= kDefaultBatches &&
           state->control.task_count == TaskCount(batches) &&
           state->control.kernel_task_count == KernelTaskCount(batches) &&
           state->control.builder_thread_count == kBuilderThreadCount &&
           BuilderCountValid(state->control.builder_count) && state->control.heap_base == kSyntheticHeapBase &&
           state->control.heap_bytes == kHeapBytes && state->control.workspace_base != 0U &&
           state->control.workspace_bytes == kWorkloadBytes && state->control.qk_repeats >= 1U &&
           state->control.sf_repeats >= 1U && state->control.pv_repeats >= 1U && state->control.up_repeats >= 1U &&
           state->exec_dispatch.aic_task_count == batches * 2U && state->exec_dispatch.aiv_task_count == batches * 2U;
}

#if defined(__DAV_VEC__)

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtFatalValue(ExecFatalReason reason, uint32_t owner, uint32_t task_id) {
    return (static_cast<uint64_t>(reason) << kFatalReasonShift) | (static_cast<uint64_t>(owner) << kFatalOwnerShift) |
           (static_cast<uint64_t>(task_id) << kFatalTaskIdShift);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void
SimtPublishFatal(__gm__ uint64_t *fatal, ExecFatalReason reason, uint32_t owner, uint32_t task_id) {
    (void)asc_atomic_cas(fatal, static_cast<uint64_t>(0U), SimtFatalValue(reason, owner, task_id));
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t SimtBuilderReportChecksum(
    uint64_t nonce, uint32_t thread_id, uint32_t task_count, uint32_t wins, uint32_t first_task, uint32_t last_task,
    uint32_t attempts, uint32_t prepares, uint32_t commits, uint32_t insert_waits, uint32_t claim_losses
) {
    uint64_t value = nonce ^ (static_cast<uint64_t>(thread_id) << 32U) ^ task_count;
    value ^= static_cast<uint64_t>(wins) << 1U;
    value ^= static_cast<uint64_t>(first_task) << 7U;
    value ^= static_cast<uint64_t>(last_task) << 19U;
    value ^= static_cast<uint64_t>(attempts) << 37U;
    value ^= static_cast<uint64_t>(prepares) << 43U;
    value ^= static_cast<uint64_t>(commits) << 49U;
    value ^= static_cast<uint64_t>(insert_waits) << 55U;
    value ^= static_cast<uint64_t>(claim_losses) << 25U;
    value *= 0x9E3779B97F4A7C15ULL;
    value ^= value >> 29U;
    value *= 0xD6E8FEB86659FD93ULL;
    return value ^ (value >> 31U);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool
SimtPublishBuildReportWord(__gm__ uint64_t *address, uint64_t value) {
    return asc_atomic_cas(address, kReportPoisonWord, value) == kReportPoisonWord;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t SimtDescriptorWord(
    uint64_t address, uint64_t buffer_size, uint64_t owner_task, uint64_t start_offset, uint32_t ndims, uint32_t dtype,
    bool manual_dep, uint32_t shape0, uint32_t shape1, uint32_t word
) {
    if (word == 0U) {
        return address;
    }
    if (word == 1U) {
        return buffer_size;
    }
    if (word == 2U) {
        return owner_task;
    }
    if (word == 3U) {
        return start_offset;
    }
    if (word == 4U) {
        return static_cast<uint64_t>(ndims) << 32U;
    }
    if (word == 5U) {
        return static_cast<uint64_t>(dtype) | (static_cast<uint64_t>(manual_dep ? 1U : 0U) << 8U) |
               (static_cast<uint64_t>(1U) << 16U) | (static_cast<uint64_t>(shape0) << 32U);
    }
    if (word == 6U) {
        return shape1;
    }
    if (word == 8U) {
        return ndims == 1U ? shape0 : static_cast<uint64_t>(shape0) * shape1;
    }
    if (word == 9U) {
        const uint32_t stride0 = ndims == 1U ? 1U : shape1;
        const uint32_t stride1 = ndims == 1U ? 0U : 1U;
        return static_cast<uint64_t>(stride0) | (static_cast<uint64_t>(stride1) << 32U);
    }
    return 0U;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtOutputDescriptorWord(uint32_t task_id, uint32_t output_slot, uint64_t task_base, uint32_t word) {
    const uint32_t kind = task_id % kTasksPerBatch;
    uint64_t output_offset = 0U;
    uint32_t ndims = 0U;
    uint32_t dtype = static_cast<uint32_t>(DataType::Float32);
    uint32_t shape0 = 0U;
    uint32_t shape1 = 0U;
    if (kind == static_cast<uint32_t>(TaskKind::Alloc)) {
        ndims = output_slot == 0U ? 2U : 1U;
        shape0 = kPaHeads;
        shape1 = output_slot == 0U ? kPaHeadDim : 0U;
        output_offset = output_slot == 0U ? 0U : (output_slot == 1U ? 8192U : 9216U);
    } else if (kind == static_cast<uint32_t>(TaskKind::Qk)) {
        ndims = 2U;
        shape0 = kPaHeads;
        shape1 = kPaBlocksPerRequest * kPaBlockSize;
    } else if (kind == static_cast<uint32_t>(TaskKind::Sf)) {
        ndims = output_slot == 0U ? 2U : 1U;
        dtype =
            output_slot == 0U ? static_cast<uint32_t>(DataType::Bfloat16) : static_cast<uint32_t>(DataType::Float32);
        shape0 = kPaHeads;
        shape1 = output_slot == 0U ? kPaBlocksPerRequest * kPaBlockSize : 0U;
        output_offset = output_slot == 0U ? 0U : (output_slot == 1U ? 262144U : 263168U);
    } else if (kind == static_cast<uint32_t>(TaskKind::Pv)) {
        ndims = 2U;
        shape0 = kPaHeads;
        shape1 = kPaHeadDim;
    }
    const uint64_t elements = ndims == 1U ? shape0 : static_cast<uint64_t>(shape0) * shape1;
    const uint64_t element_bytes = dtype == static_cast<uint32_t>(DataType::Bfloat16) ? 2U : 4U;
    return SimtDescriptorWord(
        kSyntheticHeapBase + task_base + output_offset, elements * element_bytes, task_id, 0U, ndims, dtype, false,
        shape0, shape1, word
    );
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtExternalDescriptorWord(uint32_t batch_count, uint32_t task_id, uint32_t tensor_index, uint32_t word) {
    const uint32_t kind = task_id % kTasksPerBatch;
    const uint32_t batch = task_id / kTasksPerBatch;
    uint64_t address = 0U;
    uint64_t buffer_size = 0U;
    uint64_t start_offset = 0U;
    uint32_t dtype = static_cast<uint32_t>(DataType::Float32);
    bool manual_dep = false;
    uint32_t shape0 = 0U;
    uint32_t shape1 = 0U;
    if (kind == static_cast<uint32_t>(TaskKind::Qk) && tensor_index == 0U) {
        address = kSyntheticQueryBase;
        buffer_size = static_cast<uint64_t>(batch_count) * kPaHeads * kPaHeadDim * 2U;
        start_offset = static_cast<uint64_t>(batch) * kPaHeads * kPaHeadDim;
        dtype = static_cast<uint32_t>(DataType::Bfloat16);
        shape0 = kPaHeads;
        shape1 = kPaHeadDim;
    } else if ((kind == static_cast<uint32_t>(TaskKind::Qk) && tensor_index == 1U) ||
               (kind == static_cast<uint32_t>(TaskKind::Pv) && tensor_index == 1U)) {
        address = kind == static_cast<uint32_t>(TaskKind::Qk) ? kSyntheticKeyBase : kSyntheticValueBase;
        shape0 = batch_count * kPaBlocksPerRequest * kPaBlockSize;
        shape1 = kPaHeadDim;
        dtype = static_cast<uint32_t>(DataType::Bfloat16);
        buffer_size = static_cast<uint64_t>(shape0) * shape1 * 2U;
    } else if ((kind == static_cast<uint32_t>(TaskKind::Qk) || kind == static_cast<uint32_t>(TaskKind::Pv)) &&
               tensor_index == 2U) {
        address = kSyntheticBlockTableBase;
        shape0 = batch_count;
        shape1 = kPaMaxBlocksPerRequest;
        dtype = static_cast<uint32_t>(DataType::Int32);
        buffer_size = static_cast<uint64_t>(shape0) * shape1 * 4U;
    } else {
        address = kSyntheticOutputBase;
        buffer_size = static_cast<uint64_t>(batch_count) * kPaHeads * kPaHeadDim * 4U;
        start_offset = static_cast<uint64_t>(batch) * kPaHeads * kPaHeadDim;
        dtype = static_cast<uint32_t>(DataType::Float32);
        manual_dep = true;
        shape0 = kPaHeads;
        shape1 = kPaHeadDim;
    }
    return SimtDescriptorWord(
        address, buffer_size, kInvalidTaskId, start_offset, 2U, dtype, manual_dep, shape0, shape1, word
    );
}

constexpr uint32_t kSimtClaimFatal = 0U;
constexpr uint32_t kSimtClaimWinner = 1U;
constexpr uint32_t kSimtClaimLost = 2U;

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool
SimtBuilderOwnerValid(uint32_t owner, uint32_t builder_count) {
    return builder_count >= 1U && builder_count <= kMaxBuilderCount && owner >= kBuilderOwner &&
           owner < kBuilderOwner + builder_count;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint64_t
SimtAllocBuildingState(uint64_t nonce, uint32_t task_id, uint32_t build_owner) {
    uint64_t value = kAllocBuildingMagic ^ nonce ^ (static_cast<uint64_t>(task_id) << 19U) ^
                     (static_cast<uint64_t>(build_owner) << 3U);
    value ^= value >> 23U;
    value *= 0xD6E8FEB86659FD93ULL;
    value ^= value >> 31U;
    value |= uint64_t{1} << 63U;
    return value;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool
SimtCompetingExecStateValid(uint64_t raw, uint32_t task_id, uint32_t current_owner, uint32_t builder_count) {
    if ((raw & ~kStateKnownMask) != 0U ||
        static_cast<uint32_t>((raw >> kStateTaskIdShift) & kStateTaskIdMask) != task_id) {
        return false;
    }
    const uint32_t phase = static_cast<uint32_t>((raw >> kStatePhaseShift) & kStatePhaseMask);
    const uint32_t build_owner = static_cast<uint32_t>((raw >> kStateBuildOwnerShift) & kStateBuildOwnerMask);
    const uint32_t execute_owner = static_cast<uint32_t>((raw >> kStateExecuteOwnerShift) & kStateExecuteOwnerMask);
    const uint32_t engine = static_cast<uint32_t>((raw >> kStateEngineShift) & kStateEngineMask);
    const uint32_t payload_lines = static_cast<uint32_t>((raw >> kStatePayloadLinesShift) & kStatePayloadLinesMask);
    if (!SimtBuilderOwnerValid(build_owner, builder_count) || build_owner == current_owner) {
        return false;
    }
    if (phase == static_cast<uint32_t>(ExecPhase::Building)) {
        return execute_owner == kUnboundOwner && engine == static_cast<uint32_t>(ExecEngineClass::None) &&
               payload_lines == 0U;
    }
    const uint32_t kind = task_id % kTasksPerBatch;
    const uint32_t expected_engine =
        kind == static_cast<uint32_t>(TaskKind::Qk) || kind == static_cast<uint32_t>(TaskKind::Pv) ?
            static_cast<uint32_t>(ExecEngineClass::Aic) :
            static_cast<uint32_t>(ExecEngineClass::Aiv);
    const uint32_t expected_lines = kind == static_cast<uint32_t>(TaskKind::Up) ? 16U : 10U;
    if (engine != expected_engine || payload_lines != expected_lines) {
        return false;
    }
    if (phase == static_cast<uint32_t>(ExecPhase::Built)) {
        return execute_owner == kUnboundOwner;
    }
    if (phase != static_cast<uint32_t>(ExecPhase::Claimed) && phase != static_cast<uint32_t>(ExecPhase::Done)) {
        return false;
    }
    return expected_engine == static_cast<uint32_t>(ExecEngineClass::Aic) ?
               execute_owner < kAicOwnerCount :
               execute_owner >= kBuilderOwner + builder_count && execute_owner < kOwnerCount;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint32_t SimtTryClaimTask(
    __gm__ uint64_t *task_words, __gm__ uint64_t *fatal, uint64_t nonce, uint32_t task_id, uint32_t build_owner,
    uint32_t builder_count
) {
    constexpr uint32_t kTaskStrideWords = sizeof(FullPaTask) / sizeof(uint64_t);
    constexpr uint32_t kCompletionOffsetWords = offsetof(FullPaTask, completion) / sizeof(uint64_t);
    constexpr uint32_t kExecOffsetWords = offsetof(FullPaTask, exec) / sizeof(uint64_t);
    __gm__ uint64_t *task = task_words + task_id * kTaskStrideWords;
    if (task_id % kTasksPerBatch == static_cast<uint32_t>(TaskKind::Alloc)) {
        const uint64_t desired = SimtAllocBuildingState(nonce, task_id, build_owner);
        const uint64_t observed = asc_atomic_cas(task + kCompletionOffsetWords, static_cast<uint64_t>(0U), desired);
        if (observed == 0U) {
            return kSimtClaimWinner;
        }
        bool legal_loser = observed == 1U;
        for (uint32_t builder = 0U; builder < builder_count; ++builder) {
            const uint32_t competing_owner = kBuilderOwner + builder;
            legal_loser = legal_loser || (competing_owner != build_owner &&
                                          observed == SimtAllocBuildingState(nonce, task_id, competing_owner));
        }
        if (legal_loser) {
            return kSimtClaimLost;
        }
    } else {
        const uint64_t desired = (static_cast<uint64_t>(ExecPhase::Building) << kStatePhaseShift) |
                                 (static_cast<uint64_t>(build_owner) << kStateBuildOwnerShift) |
                                 (static_cast<uint64_t>(kUnboundOwner) << kStateExecuteOwnerShift) |
                                 (static_cast<uint64_t>(task_id) << kStateTaskIdShift);
        const uint64_t observed = asc_atomic_cas(task + kExecOffsetWords, static_cast<uint64_t>(0U), desired);
        if (observed == 0U) {
            return kSimtClaimWinner;
        }
        if (SimtCompetingExecStateValid(observed, task_id, build_owner, builder_count)) {
            return kSimtClaimLost;
        }
    }
    SimtPublishFatal(fatal, ExecFatalReason::ControlPublishConflict, build_owner, task_id);
    return kSimtClaimFatal;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtWaitBuilderStart(
    __gm__ uint64_t *builder_started, __gm__ uint64_t *fatal, uint64_t timeout_ticks, uint32_t thread,
    uint32_t build_owner, uint32_t builder_count, bool active
) {
    if (thread == 0U) {
        const uint64_t observed = asc_atomic_add(builder_started, static_cast<uint64_t>(1U));
        if (observed >= builder_count) {
            SimtPublishFatal(fatal, ExecFatalReason::InvalidBuildInput, build_owner, UINT32_MAX);
            return false;
        }
        asc_threadfence();
    }
    if (!active) {
        return true;
    }
    const uint64_t begin = clock();
    uint32_t polls = 0U;
    while (clock() - begin <= timeout_ticks) {
        const uint64_t observed = asc_atomic_add(builder_started, static_cast<uint64_t>(0U));
        if (observed == builder_count) {
            return true;
        }
        if (observed > builder_count) {
            SimtPublishFatal(fatal, ExecFatalReason::InvalidBuildInput, build_owner, UINT32_MAX);
            return false;
        }
        ++polls;
        if ((polls & kWatchdogMask) == 0U && asc_atomic_add(fatal, static_cast<uint64_t>(0U)) != 0U) {
            return false;
        }
    }
    SimtPublishFatal(fatal, ExecFatalReason::Timeout, build_owner, UINT32_MAX);
    return false;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtWaitAtomicValue(
    __gm__ uint64_t *address, uint64_t expected, __gm__ uint64_t *fatal, uint64_t timeout_ticks, uint32_t task_id,
    uint32_t build_owner, uint32_t *poll_count
) {
    const uint64_t begin = clock();
    uint32_t polls = 0U;
    while (clock() - begin <= timeout_ticks) {
        const uint64_t observed = asc_atomic_add(address, static_cast<uint64_t>(0U));
        ++polls;
        if (observed == expected) {
            *poll_count += polls;
            return true;
        }
        // insert_completion starts at predecessor_id-1 and is incremented
        // exactly once to predecessor_id.  Any third value is corruption, not
        // a slow predecessor, so report it immediately instead of timing out.
        if (observed != expected - 1U) {
            *poll_count += polls;
            SimtPublishFatal(fatal, ExecFatalReason::InsertProtocolFailed, build_owner, task_id);
            return false;
        }
        if ((polls & kWatchdogMask) == 0U && asc_atomic_add(fatal, static_cast<uint64_t>(0U)) != 0U) {
            *poll_count += polls;
            return false;
        }
    }
    *poll_count += polls;
    return false;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtLoadTaskBase(
    __gm__ uint64_t *task_words, uint32_t producer_task, __gm__ uint64_t *fatal, uint64_t timeout_ticks,
    uint32_t build_owner, uint64_t *task_base, uint32_t *access_count
) {
    constexpr uint32_t kTaskStrideWords = sizeof(FullPaTask) / sizeof(uint64_t);
    constexpr uint32_t kBaseReportOffsetWords =
        (offsetof(FullPaTask, allocation) + offsetof(FullPaTaskAllocationReport, task_base_plus_one)) /
        sizeof(uint64_t);
    __gm__ uint64_t *report = task_words + producer_task * kTaskStrideWords + kBaseReportOffsetWords;
    const uint64_t begin = clock();
    while (clock() - begin <= timeout_ticks) {
        const uint64_t observed = asc_atomic_add(report, static_cast<uint64_t>(0U));
        ++*access_count;
        if (observed != 0U) {
            *task_base = observed - 1U;
            return true;
        }
        if (((*access_count) & kWatchdogMask) == 0U && asc_atomic_add(fatal, static_cast<uint64_t>(0U)) != 0U) {
            return false;
        }
    }
    return false;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint32_t
SimtTensorSourceTask(uint32_t task_id, uint32_t tensor_index) {
    const uint32_t kind = task_id % kTasksPerBatch;
    if (kind == static_cast<uint32_t>(TaskKind::Qk)) {
        return tensor_index == 3U ? task_id : UINT32_MAX;
    }
    if (kind == static_cast<uint32_t>(TaskKind::Sf)) {
        return tensor_index == 0U ? task_id - 1U : task_id;
    }
    if (kind == static_cast<uint32_t>(TaskKind::Pv)) {
        return tensor_index == 0U ? task_id - 1U : (tensor_index == 3U ? task_id : UINT32_MAX);
    }
    if (kind != static_cast<uint32_t>(TaskKind::Up)) {
        return UINT32_MAX;
    }
    if (tensor_index <= 1U) {
        return task_id - 2U;
    }
    if (tensor_index == 2U) {
        return task_id - 1U;
    }
    return tensor_index <= 5U ? (task_id / kTasksPerBatch) * kTasksPerBatch : UINT32_MAX;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline uint32_t
SimtTensorOutputSlot(uint32_t task_id, uint32_t tensor_index) {
    const uint32_t kind = task_id % kTasksPerBatch;
    if (kind == static_cast<uint32_t>(TaskKind::Qk)) {
        return 0U;
    }
    if (kind == static_cast<uint32_t>(TaskKind::Sf)) {
        return tensor_index == 0U ? 0U : tensor_index - 1U;
    }
    if (kind == static_cast<uint32_t>(TaskKind::Pv)) {
        return 0U;
    }
    if (tensor_index <= 1U) {
        return tensor_index + 1U;
    }
    if (tensor_index == 2U) {
        return 0U;
    }
    return 5U - tensor_index;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtPrepareTask(
    __gm__ uint64_t *task_words, __gm__ uint64_t *heap_words, __gm__ uint64_t *fatal, uint64_t nonce,
    uint64_t timeout_ticks, uint32_t batch_count, uint32_t task_count, uint32_t task_id, uint32_t builder_thread,
    uint32_t build_owner, uint64_t *completion_vend, uint32_t *payload_words_written, uint32_t *state_access_count
) {
    constexpr uint32_t kTaskStrideWords = sizeof(FullPaTask) / sizeof(uint64_t);
    constexpr uint32_t kPlanOffsetWords = offsetof(FullPaTask, plan) / sizeof(uint64_t);
    constexpr uint32_t kAllocationOffsetWords = offsetof(FullPaTask, allocation) / sizeof(uint64_t);
    constexpr uint32_t kBaseReportOffsetWords =
        kAllocationOffsetWords + offsetof(FullPaTaskAllocationReport, task_base_plus_one) / sizeof(uint64_t);
    constexpr uint32_t kVendReportOffsetWords =
        kAllocationOffsetWords + offsetof(FullPaTaskAllocationReport, completion_vend_plus_one) / sizeof(uint64_t);
    constexpr uint32_t kOutputsOffsetWords = offsetof(FullPaTask, outputs) / sizeof(uint64_t);
    constexpr uint32_t kOutputTensorOffsetWords =
        kOutputsOffsetWords + offsetof(SharedOutputCell, tensors) / sizeof(uint64_t);
    constexpr uint32_t kPublishedOffsetWords =
        kOutputsOffsetWords + offsetof(SharedOutputCell, published) / sizeof(uint64_t);
    constexpr uint32_t kLastWriterOffsetWords =
        kOutputsOffsetWords + offsetof(SharedOutputCell, last_writer) / sizeof(uint64_t);
    constexpr uint32_t kExecOffsetWords = offsetof(FullPaTask, exec) / sizeof(uint64_t);
    constexpr uint32_t kExecPayloadOffsetWords =
        kExecOffsetWords + offsetof(SharedExecCell, payload) / sizeof(uint64_t);
    constexpr uint32_t kHeapAtomicStrideWords = sizeof(AtomicLine) / sizeof(uint64_t);
    constexpr uint32_t kAtomicStrideWords = sizeof(AtomicLine) / sizeof(uint64_t);
    constexpr uint32_t kAggregateVendOffsetWords = offsetof(FullPaHeapControl, aggregate_vend) / sizeof(uint64_t);

    __gm__ uint64_t *task = task_words + task_id * kTaskStrideWords;
    const uint32_t kind = task_id % kTasksPerBatch;
    const bool executable = kind != static_cast<uint32_t>(TaskKind::Alloc);
    const uint32_t output_count =
        kind == static_cast<uint32_t>(TaskKind::Alloc) || kind == static_cast<uint32_t>(TaskKind::Sf) ?
            3U :
            (kind == static_cast<uint32_t>(TaskKind::Qk) || kind == static_cast<uint32_t>(TaskKind::Pv) ? 1U : 0U);
    const uint64_t reserve = kind == static_cast<uint32_t>(TaskKind::Alloc) ?
                                 10240U :
                                 (kind == static_cast<uint32_t>(TaskKind::Qk) ?
                                      524288U :
                                      (kind == static_cast<uint32_t>(TaskKind::Sf) ?
                                           264192U :
                                           (kind == static_cast<uint32_t>(TaskKind::Pv) ? 8192U : 0U)));
    uint32_t tensor_count = 0U;
    uint32_t scalar_count = 0U;
    uint32_t fanin_count = 0U;
    uint32_t engine = static_cast<uint32_t>(ExecEngineClass::None);
    uint32_t payload_bytes = 0U;
    uint32_t payload_lines = 0U;
    if (kind == static_cast<uint32_t>(TaskKind::Qk)) {
        tensor_count = 4U;
        scalar_count = 2U;
        engine = static_cast<uint32_t>(ExecEngineClass::Aic);
        payload_bytes = 592U;
        payload_lines = 10U;
    } else if (kind == static_cast<uint32_t>(TaskKind::Sf)) {
        tensor_count = 4U;
        scalar_count = 3U;
        fanin_count = 1U;
        engine = static_cast<uint32_t>(ExecEngineClass::Aiv);
        payload_bytes = 604U;
        payload_lines = 10U;
    } else if (kind == static_cast<uint32_t>(TaskKind::Pv)) {
        tensor_count = 4U;
        scalar_count = 2U;
        fanin_count = 1U;
        engine = static_cast<uint32_t>(ExecEngineClass::Aic);
        payload_bytes = 596U;
        payload_lines = 10U;
    } else if (kind == static_cast<uint32_t>(TaskKind::Up)) {
        tensor_count = 7U;
        scalar_count = 2U;
        fanin_count = 3U;
        engine = static_cast<uint32_t>(ExecEngineClass::Aiv);
        payload_bytes = 988U;
        payload_lines = 16U;
    }

    const uint32_t shard = task_id & (kSharedHeapShards - 1U);
    uint64_t task_base = 0U;
    uint64_t vend = 0U;
    if (reserve != 0U) {
        const uint64_t cursor = asc_atomic_add(heap_words + shard * kHeapAtomicStrideWords, reserve);
        const uint64_t observed_vend = asc_atomic_add(heap_words + kAggregateVendOffsetWords, reserve);
        if (cursor > kHeapShardSpan - reserve || observed_vend > kHeapBytes - reserve) {
            SimtPublishFatal(fatal, ExecFatalReason::HeapReservationFailed, build_owner, task_id);
            return false;
        }
        task_base = static_cast<uint64_t>(shard) * kHeapShardSpan + cursor;
        vend = observed_vend + reserve;
    } else {
        vend = asc_atomic_add(heap_words + kAggregateVendOffsetWords, static_cast<uint64_t>(0U));
    }
    if (asc_atomic_cas(task + kBaseReportOffsetWords, 0U, task_base + 1U) != 0U ||
        asc_atomic_cas(task + kVendReportOffsetWords, 0U, vend + 1U) != 0U) {
        SimtPublishFatal(fatal, ExecFatalReason::HeapReservationFailed, build_owner, task_id);
        return false;
    }
    *completion_vend = vend;

    __gm__ uint64_t *plan = task + kPlanOffsetWords;
    plan[0] = static_cast<uint64_t>(task_id) | (static_cast<uint64_t>(task_id / kTasksPerBatch) << 32U);
    plan[1] = static_cast<uint64_t>(kind) | (static_cast<uint64_t>(engine) << 32U);
    plan[2] = static_cast<uint64_t>(output_count) | (static_cast<uint64_t>(payload_lines) << 32U);
    plan[3] = static_cast<uint64_t>(builder_thread) | (static_cast<uint64_t>(builder_thread / kWarpSize) << 32U);
    const uint32_t encoded_meta =
        kDispatchMetaPresent | (task_id + 1U == task_count ? kDispatchMetaLastSubmit : 0U) | kind;
    const uint32_t exec_route =
        kExecRoutePresent | (executable ? kExecRouteExecutable : 0U) | (engine << kExecRouteEngineShift);
    plan[4] = static_cast<uint64_t>(encoded_meta) | (static_cast<uint64_t>(exec_route) << 8U) |
              (static_cast<uint64_t>(build_owner) << 16U);
    plan[5] = reserve;
    plan[6] = nonce;
    plan[7] = 0U;

    __gm__ uint64_t *output_tensors = task + kOutputTensorOffsetWords;
    for (uint32_t output = 0U; output < output_count; ++output) {
        for (uint32_t word = 0U; word < kTensorDescWords; ++word) {
            output_tensors[output * kTensorDescWords + word] =
                SimtOutputDescriptorWord(task_id, output, task_base, word);
        }
    }
    // Fresh outputs are independent across tasks.  Publish each descriptor as
    // soon as this warp leader has completed it; do not pull this work into the
    // strict task[N-1] insert chain below.
    asc_threadfence();
    for (uint32_t output = 0U; output < output_count; ++output) {
        __gm__ uint64_t *published = task + kPublishedOffsetWords + output * kAtomicStrideWords;
        __gm__ uint64_t *last_writer = task + kLastWriterOffsetWords + output * kAtomicStrideWords;
        if (asc_atomic_cas(published, UINT64_MAX, task_id) != UINT64_MAX ||
            asc_atomic_cas(last_writer, UINT64_MAX, task_id) != UINT64_MAX) {
            SimtPublishFatal(fatal, ExecFatalReason::InsertProtocolFailed, build_owner, task_id);
            return false;
        }
    }

    if (executable) {
        __gm__ uint64_t *payload = task + kExecPayloadOffsetWords;
        payload[0] = task_id;
        payload[1] = 0U;
        payload[2] = vend;
        payload[3] = static_cast<uint64_t>(kind - 1U) | (static_cast<uint64_t>(payload_bytes) << 32U);
        payload[4] = static_cast<uint64_t>(tensor_count) | (static_cast<uint64_t>(scalar_count) << 16U) |
                     (static_cast<uint64_t>(fanin_count) << 32U) | (static_cast<uint64_t>(engine) << 48U);
        payload[5] = static_cast<uint64_t>(1U) << 48U;
        payload[6] = 0U;
        payload[7] = 0U;

        uint32_t destination = kPayloadHeaderWords;
        for (uint32_t tensor = 0U; tensor < tensor_count; ++tensor) {
            const uint32_t producer = SimtTensorSourceTask(task_id, tensor);
            uint64_t producer_base = 0U;
            if (producer != UINT32_MAX) {
                if (producer == task_id) {
                    producer_base = task_base;
                } else if (!SimtLoadTaskBase(
                               task_words, producer, fatal, timeout_ticks, build_owner, &producer_base,
                               state_access_count
                           )) {
                    SimtPublishFatal(fatal, ExecFatalReason::Timeout, build_owner, task_id);
                    return false;
                }
            }
            for (uint32_t word = 0U; word < kTensorDescWords; ++word) {
                payload[destination++] =
                    producer == UINT32_MAX ?
                        SimtExternalDescriptorWord(batch_count, task_id, tensor, word) :
                        SimtOutputDescriptorWord(producer, SimtTensorOutputSlot(task_id, tensor), producer_base, word);
            }
        }
        for (uint32_t scalar = 0U; scalar < scalar_count; ++scalar) {
            uint64_t value = 0U;
            if (kind == static_cast<uint32_t>(TaskKind::Qk) || kind == static_cast<uint32_t>(TaskKind::Pv)) {
                value = scalar == 0U ? kPaBlocksPerRequest :
                                       static_cast<uint64_t>(task_id / kTasksPerBatch) * kPaMaxBlocksPerRequest;
            } else if (kind == static_cast<uint32_t>(TaskKind::Sf)) {
                value = scalar == 0U ? kPaScaleBits : (scalar == 1U ? kPaBlocksPerRequest : kPaBlockSize);
            } else {
                value = 1U;
            }
            payload[destination++] = value;
        }
        for (uint32_t fanin = 0U; fanin < fanin_count; fanin += 2U) {
            uint32_t low = 0U;
            uint32_t high = 0U;
            if (kind == static_cast<uint32_t>(TaskKind::Sf) || kind == static_cast<uint32_t>(TaskKind::Pv)) {
                low = task_id - 1U;
            } else {
                low = task_id - 2U;
                high = task_id - 1U;
                if (fanin == 2U) {
                    low = (task_id / kTasksPerBatch) * kTasksPerBatch;
                    high = 0U;
                }
            }
            payload[destination++] = static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32U);
        }
        *payload_words_written = destination;
    } else {
        *payload_words_written = 0U;
    }
    asc_threadfence();
    return true;
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline bool SimtCommitTask(
    __gm__ uint64_t *task_words, __gm__ uint64_t *alloc_done, __gm__ uint64_t *fatal, uint64_t nonce,
    uint64_t timeout_ticks, uint32_t task_id, uint32_t build_owner, uint64_t completion_vend,
    uint32_t *insert_poll_count, int64_t *predecessor_observed
) {
    constexpr uint32_t kTaskStrideWords = sizeof(FullPaTask) / sizeof(uint64_t);
    constexpr uint32_t kCompletionOffsetWords = offsetof(FullPaTask, completion) / sizeof(uint64_t);
    constexpr uint32_t kInsertOffsetWords = offsetof(FullPaTask, insert_completion) / sizeof(uint64_t);
    constexpr uint32_t kOutputsOffsetWords = offsetof(FullPaTask, outputs) / sizeof(uint64_t);
    constexpr uint32_t kLastWriterOffsetWords =
        kOutputsOffsetWords + offsetof(SharedOutputCell, last_writer) / sizeof(uint64_t);
    constexpr uint32_t kHistoryOffsetWords = offsetof(FullPaTask, writer_history) / sizeof(uint64_t);
    constexpr uint32_t kExecOffsetWords = offsetof(FullPaTask, exec) / sizeof(uint64_t);
    __gm__ uint64_t *task = task_words + task_id * kTaskStrideWords;
    const uint32_t kind = task_id % kTasksPerBatch;

    if (task_id != 0U) {
        __gm__ uint64_t *predecessor = task_words + (task_id - 1U) * kTaskStrideWords + kInsertOffsetWords;
        uint32_t polls = 0U;
        if (!SimtWaitAtomicValue(
                predecessor, static_cast<uint64_t>(task_id - 1U), fatal, timeout_ticks, task_id, build_owner, &polls
            )) {
            *insert_poll_count += polls;
            SimtPublishFatal(fatal, ExecFatalReason::Timeout, build_owner, task_id);
            return false;
        }
        *insert_poll_count += polls;
        *predecessor_observed = static_cast<int64_t>(task_id - 1U);
    } else {
        *predecessor_observed = -1;
    }

    if (kind == static_cast<uint32_t>(TaskKind::Up)) {
        const uint32_t alloc = (task_id / kTasksPerBatch) * kTasksPerBatch;
        __gm__ uint64_t *history = task + kHistoryOffsetWords;
        history[0] = static_cast<uint64_t>(kWriterHistoryMagic) | (static_cast<uint64_t>(task_id) << 32U);
        history[1] = 3U;
        history[2] = static_cast<uint64_t>(alloc * kOutputsPerTask + 3U) | (static_cast<uint64_t>(alloc) << 32U);
        history[3] = static_cast<uint64_t>(alloc * kOutputsPerTask + 2U) | (static_cast<uint64_t>(alloc) << 32U);
        history[4] = static_cast<uint64_t>(alloc * kOutputsPerTask + 1U) | (static_cast<uint64_t>(alloc) << 32U);
        asc_threadfence();
    }

    if (kind == static_cast<uint32_t>(TaskKind::Up)) {
        const uint32_t alloc = (task_id / kTasksPerBatch) * kTasksPerBatch;
        __gm__ uint64_t *alloc_last_writer = task_words + alloc * kTaskStrideWords + kLastWriterOffsetWords;
        if (asc_atomic_cas(alloc_last_writer, alloc, task_id) != alloc) {
            SimtPublishFatal(fatal, ExecFatalReason::InsertProtocolFailed, build_owner, task_id);
            return false;
        }
    }

    const uint64_t expected_insert = task_id == 0U ? UINT64_MAX : static_cast<uint64_t>(task_id - 1U);
    const uint64_t insert_observed = asc_atomic_add(task + kInsertOffsetWords, static_cast<uint64_t>(1U));
    if (insert_observed != expected_insert) {
        SimtPublishFatal(fatal, ExecFatalReason::InsertProtocolFailed, build_owner, task_id);
        return false;
    }

    if (kind == static_cast<uint32_t>(TaskKind::Alloc)) {
        if (asc_atomic_cas(task + kCompletionOffsetWords + 1U, 0U, completion_vend) != 0U) {
            SimtPublishFatal(fatal, ExecFatalReason::CompletionPublishFailed, build_owner, task_id);
            return false;
        }
        asc_threadfence();
        const uint64_t alloc_building = SimtAllocBuildingState(nonce, task_id, build_owner);
        if (asc_atomic_cas(task + kCompletionOffsetWords, alloc_building, 1U) != alloc_building) {
            SimtPublishFatal(fatal, ExecFatalReason::CompletionPublishFailed, build_owner, task_id);
            return false;
        }
        (void)asc_atomic_add(alloc_done, static_cast<uint64_t>(1U));
    } else {
        uint32_t tensor_count = kind == static_cast<uint32_t>(TaskKind::Up) ? 7U : 4U;
        uint32_t scalar_count = kind == static_cast<uint32_t>(TaskKind::Sf) ? 3U : 2U;
        uint32_t fanin_count =
            kind == static_cast<uint32_t>(TaskKind::Up) ? 3U : (kind == static_cast<uint32_t>(TaskKind::Qk) ? 0U : 1U);
        const uint32_t payload_bytes = kCacheLineBytes + tensor_count * kTensorDescBytes +
                                       scalar_count * sizeof(uint64_t) + fanin_count * sizeof(int32_t);
        const uint32_t payload_lines = (payload_bytes + kCacheLineBytes - 1U) / kCacheLineBytes;
        const uint32_t engine =
            kind == static_cast<uint32_t>(TaskKind::Qk) || kind == static_cast<uint32_t>(TaskKind::Pv) ?
                static_cast<uint32_t>(ExecEngineClass::Aic) :
                static_cast<uint32_t>(ExecEngineClass::Aiv);
        const uint64_t building = (static_cast<uint64_t>(ExecPhase::Building) << kStatePhaseShift) |
                                  (static_cast<uint64_t>(build_owner) << kStateBuildOwnerShift) |
                                  (static_cast<uint64_t>(kUnboundOwner) << kStateExecuteOwnerShift) |
                                  (static_cast<uint64_t>(task_id) << kStateTaskIdShift);
        const uint64_t built = (static_cast<uint64_t>(ExecPhase::Built) << kStatePhaseShift) |
                               (static_cast<uint64_t>(build_owner) << kStateBuildOwnerShift) |
                               (static_cast<uint64_t>(kUnboundOwner) << kStateExecuteOwnerShift) |
                               (static_cast<uint64_t>(engine) << kStateEngineShift) |
                               (static_cast<uint64_t>(payload_lines) << kStatePayloadLinesShift) |
                               (static_cast<uint64_t>(task_id) << kStateTaskIdShift);
        asc_threadfence();
        if (asc_atomic_cas(task + kExecOffsetWords, building, built) != building) {
            SimtPublishFatal(fatal, ExecFatalReason::ControlPublishConflict, build_owner, task_id);
            return false;
        }
    }
    return true;
}

static __simt_vf__ __aicore__ LAUNCH_BOUND(kBuilderThreadCount) void G0SimtBuildTasks(
    __gm__ uint64_t *task_words, __gm__ uint64_t *heap_words, __gm__ uint64_t *alloc_done,
    __gm__ uint64_t *builder_started, __gm__ uint64_t *builder_finished, __gm__ uint64_t *fatal,
    __gm__ uint64_t *thread_report_words, uint64_t nonce, uint64_t timeout_ticks, uint32_t batch_count,
    uint32_t task_count, uint32_t builder_instance, uint32_t builder_count, uint32_t build_owner
) {
    const uint32_t thread = static_cast<uint32_t>(threadIdx.x);
    const uint32_t warp = thread / kWarpSize;
    const uint32_t lane = thread % kWarpSize;
    const bool active = lane == 0U && warp < kBuilderWarpCount;
    const uint32_t global_thread = builder_instance * kBuilderThreadCount + thread;
    const uint32_t global_warp = builder_instance * kBuilderWarpCount + warp;
    uint32_t tasks_built = 0U;
    uint32_t prepared = 0U;
    uint32_t committed = 0U;
    uint32_t attempts = 0U;
    uint32_t producer_base_polls = 0U;
    uint32_t insert_waits = 0U;
    uint32_t claim_losses = 0U;
    uint32_t first_task = UINT32_MAX;
    uint32_t last_task = UINT32_MAX;
    uint64_t checksum = 0U;

    constexpr uint32_t kTaskStrideWords = sizeof(FullPaTask) / sizeof(uint64_t);
    constexpr uint32_t kBuildReportOffsetWords = offsetof(FullPaTask, build_report) / sizeof(uint64_t);
    constexpr uint32_t kThreadReportStrideWords = sizeof(FullPaBuilderThreadReport) / sizeof(uint64_t);
    const bool start_ready =
        SimtWaitBuilderStart(builder_started, fatal, timeout_ticks, thread, build_owner, builder_count, active);
    if (active && start_ready) {
        for (uint32_t task_id = warp; task_id < task_count; task_id += kBuilderTaskStride) {
            if (asc_atomic_add(fatal, static_cast<uint64_t>(0U)) != 0U) {
                break;
            }
            ++attempts;
            __gm__ uint64_t *report = task_words + task_id * kTaskStrideWords + kBuildReportOffsetWords;
            (void)asc_atomic_add(report + 6U, static_cast<uint64_t>(1U));
            const uint32_t claim = SimtTryClaimTask(task_words, fatal, nonce, task_id, build_owner, builder_count);
            if (claim == kSimtClaimLost) {
                ++claim_losses;
                continue;
            }
            if (claim != kSimtClaimWinner) {
                break;
            }
            (void)asc_atomic_add(report + 6U, static_cast<uint64_t>(1U) << 32U);
            uint64_t completion_vend = 0U;
            uint32_t payload_words = 0U;
            if (!SimtPrepareTask(
                    task_words, heap_words, fatal, nonce, timeout_ticks, batch_count, task_count, task_id,
                    global_thread, build_owner, &completion_vend, &payload_words, &producer_base_polls
                )) {
                break;
            }
            ++prepared;
            uint32_t task_insert_polls = 0U;
            int64_t predecessor_observed = -1;
            if (!SimtCommitTask(
                    task_words, alloc_done, fatal, nonce, timeout_ticks, task_id, build_owner, completion_vend,
                    &task_insert_polls, &predecessor_observed
                )) {
                break;
            }
            insert_waits += task_id == 0U ? 0U : 1U;
            ++committed;
            ++tasks_built;
            if (tasks_built == 1U) {
                first_task = task_id;
            }
            last_task = task_id;

            const uint32_t kind = task_id % kTasksPerBatch;
            const uint32_t output_count =
                kind == static_cast<uint32_t>(TaskKind::Alloc) || kind == static_cast<uint32_t>(TaskKind::Sf) ?
                    3U :
                    (kind == static_cast<uint32_t>(TaskKind::Qk) || kind == static_cast<uint32_t>(TaskKind::Pv) ? 1U :
                                                                                                                  0U);
            uint32_t phases = kBuildPreparedBit | kBuildOutputsPublishedBit | kBuildInsertCommittedBit;
            phases |= kind == static_cast<uint32_t>(TaskKind::Alloc) ? kBuildAllocCompletedBit : kBuildExecPublishedBit;
            if (!SimtPublishBuildReportWord(
                    report, static_cast<uint64_t>(task_id) | (static_cast<uint64_t>(global_thread) << 32U)
                ) ||
                !SimtPublishBuildReportWord(report + 1U, static_cast<uint64_t>(global_warp)) ||
                !SimtPublishBuildReportWord(
                    report + 2U, static_cast<uint64_t>(phases) | (static_cast<uint64_t>(output_count) << 32U)
                ) ||
                !SimtPublishBuildReportWord(
                    report + 3U,
                    static_cast<uint64_t>(payload_words) | (static_cast<uint64_t>(task_insert_polls) << 32U)
                ) ||
                !SimtPublishBuildReportWord(report + 4U, static_cast<uint64_t>(predecessor_observed)) ||
                !SimtPublishBuildReportWord(
                    report + 5U, static_cast<uint64_t>(1U) | (static_cast<uint64_t>(1U) << 32U)
                ) ||
                !SimtPublishBuildReportWord(report + 7U, nonce)) {
                SimtPublishFatal(fatal, ExecFatalReason::ControlPublishConflict, build_owner, task_id);
                break;
            }
            asc_threadfence();
            if (task_id + 1U == task_count) {
                if (asc_atomic_cas(builder_finished, static_cast<uint64_t>(0U), static_cast<uint64_t>(1U)) != 0U) {
                    SimtPublishFatal(fatal, ExecFatalReason::ControlPublishConflict, build_owner, task_id);
                }
            }
        }
    }

    if (active) {
        checksum = SimtBuilderReportChecksum(
            nonce, global_thread, task_count, tasks_built, first_task, last_task, attempts, prepared, committed,
            insert_waits, claim_losses
        );

        __gm__ uint64_t *thread_report = thread_report_words + global_thread * kThreadReportStrideWords;
        thread_report[0] = static_cast<uint64_t>(global_thread) | (static_cast<uint64_t>(global_warp) << 32U);
        thread_report[1] = static_cast<uint64_t>(lane) | (static_cast<uint64_t>(1U) << 32U);
        thread_report[2] = static_cast<uint64_t>(tasks_built) | (static_cast<uint64_t>(first_task) << 32U);
        thread_report[3] = static_cast<uint64_t>(last_task) | (static_cast<uint64_t>(attempts) << 32U);
        thread_report[4] = static_cast<uint64_t>(prepared) | (static_cast<uint64_t>(committed) << 32U);
        thread_report[5] = static_cast<uint64_t>(insert_waits) | (static_cast<uint64_t>(claim_losses) << 32U);
        thread_report[6] = nonce;
        thread_report[7] = checksum;
        asc_threadfence();
    }
}

#endif  // defined(__DAV_VEC__)

__aicore__ __attribute__((always_inline)) inline void ResetToken(__gm__ ExecutionToken *token) {
    token->control.phase = ExecTokenPhase::Idle;
    token->control.task_id = UINT32_MAX;
    token->control.build_owner = UINT32_MAX;
    token->control.execute_owner = UINT32_MAX;
    token->control.engine_class = ExecEngineClass::None;
    token->control.payload_lines = 0U;
    token->control.payload_bytes = 0U;
    token->control.fanin_ready_prefix = 0U;
    token->control.payload_address = 0U;
    token->control.completion_vend = 0U;
    token->control.function_and_reference = 0U;
    token->control.shape_and_scalar_offset = 0U;
    // Production ResetExecutionToken resets only this control line.  dispatch
    // intentionally retains the last binding for the owner-local token.
}

__aicore__ __attribute__((always_inline)) inline void
PublishTerminalTokenState(__gm__ FullPaState *state, uint32_t owner, uint32_t ticket_count) {
    const uint32_t used_tokens = ticket_count < kTokensPerOwner ? ticket_count : kTokensPerOwner;
    for (uint32_t slot = 0U; slot < used_tokens; ++slot) {
        __gm__ ExecutionToken *token = &state->tokens[owner][slot];
        dcci(static_cast<__gm__ void *>(&token->control), kSingleCacheLine);
        __gm__ uint8_t *dispatch = reinterpret_cast<__gm__ uint8_t *>(&token->dispatch);
        // The active tensor/scalar prefix occupies lines 0/1.  args[48/49]
        // and local_context share line 6; global_context and padding line 7.
        dcci(static_cast<__gm__ void *>(dispatch), kSingleCacheLine);
        dcci(static_cast<__gm__ void *>(dispatch + kCacheLineBytes), kSingleCacheLine);
        dcci(static_cast<__gm__ void *>(dispatch + 6U * kCacheLineBytes), kSingleCacheLine);
        dcci(static_cast<__gm__ void *>(dispatch + 7U * kCacheLineBytes), kSingleCacheLine);
    }
    if (used_tokens != 0U) {
        // Owner-local tokens have no concurrent ordinary writer.  One terminal
        // clean+invalidate pass publishes both reset control and the retained
        // last binding without adding DCCI to every execution.
        dsb(DSB_ALL);
    }
}

__aicore__ __attribute__((always_inline)) inline uint32_t BusyTokenCount(__gm__ FullPaState *state, uint32_t owner) {
    uint32_t busy = 0U;
    for (uint32_t slot = 0U; slot < kTokensPerOwner; ++slot) {
        busy += state->tokens[owner][slot].control.phase == ExecTokenPhase::Idle ? 0U : 1U;
    }
    return busy;
}

__aicore__ __attribute__((always_inline)) inline uint64_t PayloadWord(__gm__ const FullPaTask *task, uint32_t word) {
    return task->exec.payload.words[word];
}

__aicore__ __attribute__((always_inline)) inline void
InvalidatePayloadLines(__gm__ FullPaTask *task, uint32_t payload_lines) {
    __gm__ uint8_t *payload =
        reinterpret_cast<__gm__ uint8_t *>(const_cast<__gm__ uint64_t *>(&task->exec.payload.words[0]));
    for (uint32_t line = 0U; line < payload_lines; ++line) {
        dcci(static_cast<__gm__ void *>(payload + line * kCacheLineBytes), kSingleCacheLine);
    }
    dsb(DSB_ALL);
}

__aicore__ __attribute__((always_inline)) inline bool
ValidatePayloadAndBind(__gm__ FullPaState *state, uint32_t owner, __gm__ ExecutionToken *token) {
    const uint32_t task_id = token->control.task_id;
    __gm__ FullPaTask *task = &state->tasks[task_id];
    const TaskKind kind = TaskKindAt(task_id);
    const TaskExecShape shape = TaskShape(kind);
    ExecPayloadLayout layout{};
    if (!IsBuilderOwner(token->control.build_owner, state->control.builder_count) ||
        !OwnerCanExecute(owner, shape.engine_class, state->control.builder_count) ||
        !ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout)) {
        return false;
    }
    InvalidatePayloadLines(task, layout.payload_lines);
    const uint64_t word0 = PayloadWord(task, 0U);
    const uint64_t word3 = PayloadWord(task, 3U);
    const uint64_t word4 = PayloadWord(task, 4U);
    const uint64_t word5 = PayloadWord(task, 5U);
    if (static_cast<uint32_t>(word0) != task_id || (word0 >> 32U) != 0U || PayloadWord(task, 1U) != 0U ||
        static_cast<uint32_t>(word3) != TaskFunctionId(kind) ||
        static_cast<uint32_t>(word3 >> 32U) != layout.payload_bytes ||
        static_cast<uint16_t>(word4) != shape.tensor_count ||
        static_cast<uint16_t>(word4 >> 16U) != shape.scalar_count ||
        static_cast<uint16_t>(word4 >> 32U) != shape.fanin_count ||
        static_cast<uint8_t>(word4 >> 48U) != static_cast<uint8_t>(shape.engine_class) ||
        static_cast<uint8_t>(word4 >> 56U) != 0U || static_cast<uint32_t>(word5) != 0U ||
        static_cast<uint16_t>(word5 >> 32U) != 0U || static_cast<uint16_t>(word5 >> 48U) != 1U ||
        PayloadWord(task, 6U) != 0U || PayloadWord(task, 7U) != 0U) {
        return false;
    }

    for (uint32_t tensor_index = 0U; tensor_index < shape.tensor_count; ++tensor_index) {
        TensorDesc expected{};
        uint32_t producer = 0U;
        uint32_t output_slot = 0U;
        if (PayloadTensorOutputSource(task_id, tensor_index, producer, output_slot)) {
            const uint64_t base_plus_one =
                ScalarAtomicLoad(&state->tasks[producer].allocation.task_base_plus_one.value);
            if (base_plus_one == 0U) {
                return false;
            }
            expected = MakeTaskOutputDescriptor(producer, output_slot, base_plus_one - 1U);
        } else if (!ResolveExternalPayloadTensor(state->control.batch_count, task_id, tensor_index, expected)) {
            return false;
        }
        const uint64_t *expected_words = reinterpret_cast<const uint64_t *>(&expected);
        const uint32_t offset = layout.tensor_word_offset + tensor_index * kTensorDescWords;
        for (uint32_t word = 0U; word < kTensorDescWords; ++word) {
            if (PayloadWord(task, offset + word) != expected_words[word]) {
                return false;
            }
        }
        token->dispatch.args[tensor_index] =
            reinterpret_cast<uint64_t>(const_cast<__gm__ uint64_t *>(&task->exec.payload.words[offset]));
    }
    for (uint32_t scalar = 0U; scalar < shape.scalar_count; ++scalar) {
        const uint64_t value = PayloadWord(task, layout.scalar_word_offset + scalar);
        if (value != TaskScalar(task_id, scalar)) {
            return false;
        }
        token->dispatch.args[shape.tensor_count + scalar] = value;
    }
    for (uint32_t edge = 0U; edge < shape.fanin_count; ++edge) {
        const uint64_t packed = PayloadWord(task, layout.fanin_word_offset + edge / 2U);
        const int32_t actual = static_cast<int32_t>(
            (edge & 1U) == 0U ? static_cast<uint32_t>(packed) : static_cast<uint32_t>(packed >> 32U)
        );
        if (actual != TaskFanin(task_id, edge)) {
            return false;
        }
    }

    token->control.payload_lines = layout.payload_lines;
    token->control.payload_bytes = layout.payload_bytes;
    token->control.payload_address =
        reinterpret_cast<uint64_t>(const_cast<__gm__ uint64_t *>(&task->exec.payload.words[0]));
    token->control.completion_vend = PayloadWord(task, 2U);
    token->control.function_and_reference = static_cast<uint64_t>(TaskFunctionId(kind));
    token->control.shape_and_scalar_offset = PackExecutionTokenShapeAndScalarOffset(
        shape.tensor_count, shape.scalar_count, shape.fanin_count, static_cast<uint16_t>(layout.scalar_word_offset)
    );
    __gm__ uint64_t *local_context = reinterpret_cast<__gm__ uint64_t *>(&token->dispatch.local_context[0]);
    local_context[0] = task_id;
    local_context[1] = owner;
    local_context[2] = layout.payload_bytes;
    local_context[3] = layout.payload_lines;
    local_context[4] = shape.fanin_count;
    local_context[5] = token->control.completion_vend;
    *reinterpret_cast<__gm__ uint32_t *>(&token->dispatch.global_context[0]) = state->control.batch_count;
    token->dispatch.args[kDispatchLocalContextIndex] = reinterpret_cast<uint64_t>(&token->dispatch.local_context[0]);
    token->dispatch.args[kDispatchGlobalContextIndex] = reinterpret_cast<uint64_t>(&token->dispatch.global_context[0]);
    return true;
}

__aicore__ __attribute__((always_inline)) inline bool
FaninReady(__gm__ FullPaState *state, __gm__ ExecutionToken *token) {
    const uint32_t task_id = token->control.task_id;
    const uint32_t fanin_count = TaskFaninCount(task_id);
    while (token->control.fanin_ready_prefix < fanin_count) {
        const int32_t producer = TaskFanin(task_id, token->control.fanin_ready_prefix);
        if (producer < 0 || ScalarAtomicLoad(&state->tasks[static_cast<uint32_t>(producer)].completion.flag) != 1U) {
            return false;
        }
        ++token->control.fanin_ready_prefix;
    }
    return true;
}

__aicore__ __attribute__((always_inline)) inline bool PublishExecutionWitness(
    __gm__ FullPaState *state, uint32_t owner, uint32_t task_id, TaskKind kind, uint64_t output_checksum,
    uint32_t fanin_ready_prefix
) {
    __gm__ FullPaExecutionWitness *witness = &state->tasks[task_id].execution_witness;
    __gm__ uint64_t *words = reinterpret_cast<__gm__ uint64_t *>(witness);
    StoreDev64(words + 1U, state->control.launch_nonce);
    StoreDev64(words + 2U, kExecutionWitnessMagic);
    StoreDev64(words + 3U, static_cast<uint64_t>(task_id) | (static_cast<uint64_t>(kind) << 32U));
    StoreDev64(words + 4U, static_cast<uint64_t>(owner) | (static_cast<uint64_t>(1U) << 32U));
    StoreDev64(words + 5U, kCompletionSequenceWorkloadWitnessVendFlagDone);
    StoreDev64(words + 6U, output_checksum);
    StoreDev64(words + 7U, fanin_ready_prefix);
    dsb(DSB_ALL);
    return ScalarCas(&witness->state, 0U, ExecutionWitnessState(state->control.launch_nonce, task_id, kind, owner)) ==
           0U;
}

__aicore__ __attribute__((always_inline)) inline bool
RunClaimedWorkload(__gm__ FullPaState *state, uint32_t owner, __gm__ ExecutionToken *token) {
    const uint32_t task_id = token->control.task_id;
    const TaskKind kind = TaskKindAt(task_id);
    __gm__ float *workspace = reinterpret_cast<__gm__ float *>(state->control.workspace_base);
    __gm__ float *input_a = workspace;
    __gm__ float *input_b = workspace + kWorkloadTileElements;
    const uint32_t kind_slot = kind == TaskKind::Pv || kind == TaskKind::Up ? 1U : 0U;
    __gm__ float *output = workspace + (kWorkloadSharedInputTiles + owner * kWorkloadOutputTilesPerOwner + kind_slot) *
                                           kWorkloadTileElements;
    uint64_t output_poison = state->control.launch_nonce ^
                             (static_cast<uint64_t>(task_id) * UINT64_C(0x9E3779B97F4A7C15)) ^
                             UINT64_C(0xD15EA5E0C001D00D);
    if (output_poison == ExpectedWorkloadOutputPair(kind)) {
        output_poison ^= UINT64_C(0xFFFFFFFFFFFFFFFF);
    }
    // Owners reuse one tile per engine kind.  Poison the sampled output before
    // every task so a skipped workload/TSTORE cannot inherit the previous
    // task's valid checksum and publish a false-positive witness.
    StoreDev64(reinterpret_cast<__gm__ uint64_t *>(output), output_poison);
    dsb(DSB_ALL);
#if defined(__DAV_VEC__)
    if (kind == TaskKind::Sf) {
        RunG0VectorAdd(input_a, input_b, output, state->control.sf_repeats);
    } else if (kind == TaskKind::Up) {
        RunG0VectorMultiply(input_a, input_b, output, state->control.up_repeats);
    } else {
        PublishFatal(state, ExecFatalReason::InvalidTokenPayload, owner, task_id);
        return false;
    }
#else
    if (kind == TaskKind::Qk) {
        RunG0CubeMatmul(input_a, input_b, output, state->control.qk_repeats);
    } else if (kind == TaskKind::Pv) {
        RunG0CubeMatmul(input_a, input_b, output, state->control.pv_repeats);
    } else {
        PublishFatal(state, ExecFatalReason::InvalidTokenPayload, owner, task_id);
        return false;
    }
#endif
    const uint64_t checksum = LoadDev64(reinterpret_cast<__gm__ const uint64_t *>(output));
    if (checksum != ExpectedWorkloadOutputPair(kind)) {
        PublishFatal(state, ExecFatalReason::InvalidTokenPayload, owner, task_id);
        return false;
    }
    if (!PublishExecutionWitness(state, owner, task_id, kind, checksum, token->control.fanin_ready_prefix)) {
        PublishFatal(state, ExecFatalReason::CompletionPublishFailed, owner, task_id);
        return false;
    }

    __gm__ FullPaTask *task = &state->tasks[task_id];
    if (ScalarExchange(
            reinterpret_cast<__gm__ volatile int64_t *>(const_cast<__gm__ uint64_t *>(&task->completion.vend)),
            token->control.completion_vend
        ) != 0U ||
        ScalarExchange(&task->completion.flag, 1U) != 0U) {
        PublishFatal(state, ExecFatalReason::CompletionPublishFailed, owner, task_id);
        return false;
    }
    const uint64_t claimed = ClaimedState(task_id, token->control.build_owner, owner);
    if (ScalarCas(&task->exec.control.state, claimed, DoneState(task_id, token->control.build_owner, owner)) !=
        claimed) {
        PublishFatal(state, ExecFatalReason::CompletionStateConflict, owner, task_id);
        return false;
    }
    (void)ScalarFetchAdd(&state->drain.done_count.value, 1U);
    if (TaskEngine(kind) == ExecEngineClass::Aic) {
        (void)ScalarFetchAdd(&state->drain.aic_done.value, 1U);
    } else {
        (void)ScalarFetchAdd(&state->drain.aiv_done.value, 1U);
    }
    return true;
}

__aicore__ __attribute__((always_inline)) inline bool
AdvanceToken(__gm__ FullPaState *state, uint32_t owner, __gm__ ExecutionToken *token, FullPaRoleResult *result) {
    if (token->control.phase == ExecTokenPhase::Idle) {
        return false;
    }
    const uint32_t task_id = token->control.task_id;
    __gm__ FullPaTask *task = &state->tasks[task_id];
    if (token->control.phase == ExecTokenPhase::WaitingBuilt) {
        const uint64_t observed = ScalarAtomicLoad(&task->exec.control.state);
        if (observed == 0U) {
            return false;
        }
        const DecodedExecState decoded = DecodeExecState(static_cast<int64_t>(observed));
        if (decoded.valid && decoded.phase == ExecPhase::Building && decoded.task_id == task_id &&
            IsBuilderOwner(decoded.build_owner, state->control.builder_count) &&
            observed == BuildingState(task_id, decoded.build_owner)) {
            return false;
        }
        if (!decoded.valid || decoded.phase != ExecPhase::Built || decoded.task_id != task_id ||
            !IsBuilderOwner(decoded.build_owner, state->control.builder_count) ||
            observed != BuiltState(task_id, decoded.build_owner)) {
            ++result->claim_lost_count;
            PublishFatal(state, ExecFatalReason::InvalidBuiltControl, owner, task_id);
            return false;
        }
        token->control.build_owner = decoded.build_owner;
        const uint64_t claimed = ClaimedState(task_id, decoded.build_owner, owner);
        if (ScalarCas(&task->exec.control.state, observed, claimed) != observed) {
            ++result->claim_lost_count;
            PublishFatal(state, ExecFatalReason::ControlPublishConflict, owner, task_id);
            return false;
        }
        ++result->claim_count;
        token->control.phase = ExecTokenPhase::Binding;
        if (!ValidatePayloadAndBind(state, owner, token)) {
            PublishFatal(state, ExecFatalReason::ClaimedPayloadInvalid, owner, task_id);
            return false;
        }
        token->control.phase = ExecTokenPhase::WaitingFanin;
    }
    if (token->control.phase == ExecTokenPhase::WaitingFanin) {
        if (!FaninReady(state, token)) {
            return false;
        }
        token->control.phase = ExecTokenPhase::EngineInflight;
    }
    if (token->control.phase == ExecTokenPhase::EngineInflight) {
        token->control.phase = ExecTokenPhase::Completing;
        if (!RunClaimedWorkload(state, owner, token)) {
            return false;
        }
        ++result->execute_count;
        ++result->completed_by_kind[static_cast<uint32_t>(TaskKindAt(task_id))];
        ResetToken(token);
        return true;
    }
    return false;
}

__aicore__ __attribute__((always_inline)) inline uint32_t
LoadDispatchTaskId(__gm__ const uint32_t *task_ids, uint32_t ticket) {
    __gm__ const uint32_t *line = task_ids + (ticket & ~15U);
    dcci(static_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(line)), kSingleCacheLine);
    dsb(DSB_ALL);
    return task_ids[ticket];
}

__aicore__ __attribute__((always_inline)) inline void
RunExecutor(__gm__ FullPaState *state, uint32_t owner, FullPaRoleResult *result) {
    const ExecEngineClass engine = OwnerEngine(owner, state->control.builder_count);
    __gm__ AtomicLine *cursor =
        engine == ExecEngineClass::Aic ? &state->exec_dispatch.aic_next : &state->exec_dispatch.aiv_next;
    __gm__ uint32_t *task_ids =
        engine == ExecEngineClass::Aic ? &state->exec_dispatch.aic_task_ids[0] : &state->exec_dispatch.aiv_task_ids[0];
    const uint32_t task_count =
        engine == ExecEngineClass::Aic ? state->exec_dispatch.aic_task_count : state->exec_dispatch.aiv_task_count;
    bool exhausted = false;
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint32_t iterations = 0U;
    while (true) {
        for (uint32_t slot = 0U; slot < kTokensPerOwner && !exhausted; ++slot) {
            __gm__ ExecutionToken *token = &state->tokens[owner][slot];
            if (token->control.phase != ExecTokenPhase::Idle) {
                continue;
            }
            const uint32_t ticket = static_cast<uint32_t>(ScalarFetchAdd(&cursor->value, 1U));
            if (ticket >= task_count) {
                exhausted = true;
                ++result->exhausted_ticket_count;
                break;
            }
            const uint32_t task_id = LoadDispatchTaskId(task_ids, ticket);
            token->control.phase = ExecTokenPhase::WaitingBuilt;
            token->control.task_id = task_id;
            token->control.build_owner = UINT32_MAX;
            token->control.execute_owner = owner;
            token->control.engine_class = engine;
            token->control.payload_address = reinterpret_cast<uint64_t>(&state->tasks[task_id].exec.payload);
            ++result->ticket_count;
            const uint32_t busy = BusyTokenCount(state, owner);
            result->max_busy_tokens = busy > result->max_busy_tokens ? busy : result->max_busy_tokens;
        }
        for (uint32_t slot = 0U; slot < kTokensPerOwner; ++slot) {
            (void)AdvanceToken(state, owner, &state->tokens[owner][slot], result);
        }
        if (exhausted && BusyTokenCount(state, owner) == 0U) {
            break;
        }
        ++iterations;
        if ((iterations & kWatchdogMask) == 0U) {
            if (LoadFatal(state) != 0U) {
                break;
            }
            if (static_cast<uint64_t>(get_sys_cnt()) - begin > state->control.timeout_ticks) {
                PublishFatal(state, ExecFatalReason::Timeout, owner, UINT32_MAX);
                break;
            }
        }
    }
    if (LoadFatal(state) != 0U) {
        for (uint32_t slot = 0U; slot < kTokensPerOwner; ++slot) {
            ResetToken(&state->tokens[owner][slot]);
        }
    }
    PublishTerminalTokenState(state, owner, result->ticket_count);
    result->final_busy_tokens = BusyTokenCount(state, owner);
}

__aicore__ __attribute__((always_inline)) inline void
PublishRoleResult(__gm__ FullPaState *state, uint32_t owner, const FullPaRoleResult &result) {
    // CCEC may scalar-replace this 128-byte local object.  A generic pointer
    // walk over &result then reads compiler spill locations instead of the C++
    // field layout (the real-device symptom contains pieces of the GM base
    // address).  Pack every ABI word from named scalar fields instead.
    __gm__ uint64_t *destination = reinterpret_cast<__gm__ uint64_t *>(&state->roles[owner]);
    StoreDev64(destination + 0U, static_cast<uint64_t>(result.owner) | (static_cast<uint64_t>(result.role) << 32U));
    StoreDev64(
        destination + 1U,
        static_cast<uint64_t>(result.physical_block) | (static_cast<uint64_t>(result.drain_group) << 32U)
    );
    StoreDev64(
        destination + 2U,
        static_cast<uint64_t>(result.build_count) | (static_cast<uint64_t>(result.commit_count) << 32U)
    );
    StoreDev64(
        destination + 3U,
        static_cast<uint64_t>(result.execute_count) | (static_cast<uint64_t>(result.ticket_count) << 32U)
    );
    StoreDev64(
        destination + 4U,
        static_cast<uint64_t>(result.exhausted_ticket_count) | (static_cast<uint64_t>(result.claim_count) << 32U)
    );
    StoreDev64(
        destination + 5U,
        static_cast<uint64_t>(result.claim_lost_count) | (static_cast<uint64_t>(result.max_busy_tokens) << 32U)
    );
    StoreDev64(
        destination + 6U,
        static_cast<uint64_t>(result.final_busy_tokens) | (static_cast<uint64_t>(result.completed_by_kind[0]) << 32U)
    );
    StoreDev64(
        destination + 7U,
        static_cast<uint64_t>(result.completed_by_kind[1]) | (static_cast<uint64_t>(result.completed_by_kind[2]) << 32U)
    );
    StoreDev64(
        destination + 8U,
        static_cast<uint64_t>(result.completed_by_kind[3]) | (static_cast<uint64_t>(result.completed_by_kind[4]) << 32U)
    );
    StoreDev64(
        destination + 9U,
        static_cast<uint64_t>(result.drain_arrival_count) | (static_cast<uint64_t>(result.fatal_count) << 32U)
    );
    StoreDev64(destination + 10U, result.launch_nonce);
    StoreDev64(destination + 11U, result.reserved[0]);
    StoreDev64(destination + 12U, result.reserved[1]);
    StoreDev64(destination + 13U, result.reserved[2]);
    StoreDev64(destination + 14U, result.reserved[3]);
    StoreDev64(destination + 15U, result.reserved[4]);
    dsb(DSB_ALL);
}

__aicore__ __attribute__((always_inline)) inline void
InitializeRoleResult(FullPaRoleResult *result, uint32_t owner, uint32_t builder_count, uint64_t nonce) {
    result->owner = owner;
    result->role = OwnerRoleAt(owner, builder_count);
    result->physical_block = OwnerPhysicalBlock(owner);
    result->drain_group = OwnerDrainGroup(owner);
    result->build_count = 0U;
    result->commit_count = 0U;
    result->execute_count = 0U;
    result->ticket_count = 0U;
    result->exhausted_ticket_count = 0U;
    result->claim_count = 0U;
    result->claim_lost_count = 0U;
    result->max_busy_tokens = 0U;
    result->final_busy_tokens = 0U;
    result->completed_by_kind[0] = 0U;
    result->completed_by_kind[1] = 0U;
    result->completed_by_kind[2] = 0U;
    result->completed_by_kind[3] = 0U;
    result->completed_by_kind[4] = 0U;
    result->drain_arrival_count = 0U;
    result->fatal_count = 0U;
    result->launch_nonce = nonce;
    result->reserved[0] = 0U;
    result->reserved[1] = 0U;
    result->reserved[2] = 0U;
    result->reserved[3] = 0U;
    result->reserved[4] = 0U;
}

__aicore__ __attribute__((always_inline)) inline void
ArriveAndDrain(__gm__ FullPaState *state, uint32_t owner, FullPaRoleResult *result) {
    result->drain_arrival_count = 1U;
    result->fatal_count = LoadFatal(state) == 0U ? 0U : 1U;
    PublishRoleResult(state, owner, *result);
    const int64_t contribution = EncodeDrainContribution(result->execute_count);
    (void)ScalarFetchAdd(&state->drain.arrivals[result->drain_group].value, static_cast<uint64_t>(contribution));
    if (owner != kBuilderOwner) {
        return;
    }

    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    uint64_t completed = 0U;
    while (true) {
        bool all_arrived = true;
        completed = 0U;
        for (uint32_t group = 0U; group < kDrainGroupCount; ++group) {
            const int64_t raw = static_cast<int64_t>(ScalarAtomicLoad(&state->drain.arrivals[group].value));
            all_arrived = all_arrived && DecodeDrainArrivals(raw) == kDrainExpectedArrivals;
            completed += DecodeDrainCompletions(raw);
        }
        if (all_arrived) {
            break;
        }
        if (static_cast<uint64_t>(get_sys_cnt()) - begin > state->control.timeout_ticks) {
            PublishFatal(state, ExecFatalReason::Timeout, owner, UINT32_MAX);
            break;
        }
    }
    if (LoadFatal(state) == 0U) {
        const uint32_t batches = state->control.batch_count;
        const bool valid = completed == state->control.kernel_task_count &&
                           ScalarAtomicLoad(&state->drain.builder_started.value) == state->control.builder_count &&
                           ScalarAtomicLoad(&state->drain.builder_finished.value) == 1U &&
                           ScalarAtomicLoad(&state->drain.done_count.value) == state->control.kernel_task_count &&
                           ScalarAtomicLoad(&state->drain.alloc_done.value) == batches &&
                           ScalarAtomicLoad(&state->drain.aic_done.value) == batches * 2U &&
                           ScalarAtomicLoad(&state->drain.aiv_done.value) == batches * 2U &&
                           ScalarAtomicLoad(&state->exec_dispatch.aic_next.value) ==
                               state->exec_dispatch.aic_task_count + kAicOwnerCount &&
                           ScalarAtomicLoad(&state->exec_dispatch.aiv_next.value) ==
                               state->exec_dispatch.aiv_task_count + AivExecutorCount(state->control.builder_count);
        if (!valid) {
            PublishFatal(state, ExecFatalReason::DrainMismatch, owner, UINT32_MAX);
        }
    }
    if (ScalarCas(&state->drain.root_finished.value, 0U, 1U) != 0U) {
        PublishFatal(state, ExecFatalReason::DrainMismatch, owner, UINT32_MAX);
    }
}

}  // namespace

#if defined(__DAV_VEC__)

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_cross_core_g0_0_mix_aiv, 1, 2);

extern "C" __global__ __aicore__ void
simt_cross_core_g0_0_mix_aiv(__gm__ pa_scheduler::simt_cross_core::g0::FullPaState *state) {
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dcci(
        static_cast<__gm__ void *>(reinterpret_cast<__gm__ uint8_t *>(&state->control) + kCacheLineBytes),
        kSingleCacheLine
    );
    dcci(
        static_cast<__gm__ void *>(reinterpret_cast<__gm__ uint8_t *>(&state->exec_dispatch) + 2U * kCacheLineBytes),
        kSingleCacheLine
    );
    dsb(DSB_ALL);
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    const uint32_t subblock_dim = static_cast<uint32_t>(get_subblockdim());
    const uint32_t subblock = static_cast<uint32_t>(get_subblockid());
    if (block >= kAicOwnerCount || subblock_dim != 2U || subblock >= subblock_dim) {
        PublishFatal(state, ExecFatalReason::InvalidBuildInput, kBuilderOwner, 0U);
        return;
    }
    const uint32_t aiv_id = block * subblock_dim + subblock;
    const uint32_t owner = kBuilderOwner + aiv_id;
    FullPaRoleResult result;
    InitializeRoleResult(&result, owner, state->control.builder_count, state->control.launch_nonce);
    if (!ConfigValid(state)) {
        PublishFatal(state, ExecFatalReason::InvalidBuildInput, owner, 0U);
        ArriveAndDrain(state, owner, &result);
        return;
    }
    if (aiv_id >= state->control.builder_count) {
        RunExecutor(state, owner, &result);
        ArriveAndDrain(state, owner, &result);
        return;
    }
    cce::async_invoke<G0SimtBuildTasks>(
        cce::dim3{kBuilderThreadCount, 1U, 1U}, reinterpret_cast<__gm__ uint64_t *>(&state->tasks[0]),
        reinterpret_cast<__gm__ uint64_t *>(&state->heap),
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->drain.alloc_done.value)),
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->drain.builder_started.value)),
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->drain.builder_finished.value)),
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(&state->fatal.state)),
        reinterpret_cast<__gm__ uint64_t *>(&state->builder_threads[0]), state->control.launch_nonce,
        state->control.timeout_ticks, state->control.batch_count, state->control.task_count, aiv_id,
        state->control.builder_count, owner
    );
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    ArriveAndDrain(state, owner, &result);
}

#else

PTO_SYNCALL_MIX_AIC_KERNEL_META(simt_cross_core_g0_0_mix_aic, 1, 2);

extern "C" __global__ __aicore__ void
simt_cross_core_g0_0_mix_aic(__gm__ pa_scheduler::simt_cross_core::g0::FullPaState *state) {
    dcci(static_cast<__gm__ void *>(&state->control), kSingleCacheLine);
    dcci(
        static_cast<__gm__ void *>(reinterpret_cast<__gm__ uint8_t *>(&state->control) + kCacheLineBytes),
        kSingleCacheLine
    );
    dcci(
        static_cast<__gm__ void *>(reinterpret_cast<__gm__ uint8_t *>(&state->exec_dispatch) + 2U * kCacheLineBytes),
        kSingleCacheLine
    );
    dsb(DSB_ALL);
    const uint32_t owner = static_cast<uint32_t>(get_block_idx());
    FullPaRoleResult result;
    InitializeRoleResult(&result, owner, state->control.builder_count, state->control.launch_nonce);
    if (!ConfigValid(state)) {
        PublishFatal(state, ExecFatalReason::InvalidBuildInput, owner, 0U);
    } else {
        RunExecutor(state, owner, &result);
    }
    ArriveAndDrain(state, owner, &result);
}

#endif
