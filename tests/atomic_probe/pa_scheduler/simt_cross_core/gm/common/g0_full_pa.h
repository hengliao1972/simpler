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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_G0_FULL_PA_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_G0_FULL_PA_H

#include <stddef.h>
#include <stdint.h>

#include "../../common/full_pa_model.h"

namespace pa_scheduler::simt_cross_core::g0 {

#if defined(__CCE_AICORE__)
#define SIMT_CROSS_CORE_G0_ABI_INLINE __aicore__ __attribute__((always_inline)) inline
#else
#define SIMT_CROSS_CORE_G0_ABI_INLINE constexpr
#endif

constexpr uint64_t kGuardMagic = 0x4755415244473041ULL;
constexpr uint64_t kTailPoisonWord = 0xA5A5A5A5A5A5A5A5ULL;
constexpr uint64_t kDescriptorPoisonWord = 0xB4B4B4B4B4B4B4B4ULL;
constexpr uint8_t kTailPoisonByte = 0xA5U;
constexpr uint64_t kReportPoisonWord = 0xD3D3D3D3D3D3D3D3ULL;
constexpr uint64_t kExecutionWitnessMagic = 0x5749544E45535330ULL;
constexpr uint32_t kInvalidBuilderThread = UINT32_MAX;
constexpr uint32_t kBuildPreparedBit = 1U << 0U;
constexpr uint32_t kBuildOutputsPublishedBit = 1U << 1U;
constexpr uint32_t kBuildInsertCommittedBit = 1U << 2U;
constexpr uint32_t kBuildExecPublishedBit = 1U << 3U;
constexpr uint32_t kBuildAllocCompletedBit = 1U << 4U;
constexpr uint64_t kCompletionSequenceWorkloadWitnessVendFlagDone = 0x0102030405ULL;

struct alignas(kCacheLineBytes) FullPaControl {
    uint64_t magic;
    uint64_t version;
    uint64_t launch_nonce;
    uint64_t timeout_ticks;
    uint32_t batch_count;
    uint32_t task_count;
    uint32_t kernel_task_count;
    uint32_t builder_thread_count;
    uint64_t heap_base;
    uint64_t heap_bytes;
    uint64_t workspace_base;
    uint64_t workspace_bytes;
    uint32_t qk_repeats;
    uint32_t sf_repeats;
    uint32_t pv_repeats;
    uint32_t up_repeats;
    uint64_t reserved[4];
};

struct alignas(kCacheLineBytes) FullPaGuard {
    uint64_t words[kCacheLineBytes / sizeof(uint64_t)];
};

struct alignas(kCacheLineBytes) FullPaTaskPlan {
    uint32_t task_id;
    uint32_t batch;
    TaskKind kind;
    ExecEngineClass engine_class;
    uint32_t output_count;
    uint32_t payload_lines;
    uint32_t builder_thread;
    uint32_t builder_warp;
    uint8_t encoded_meta;
    uint8_t exec_route;
    uint16_t reserved16;
    uint64_t reserved_bytes;
    uint64_t launch_nonce;
    uint64_t reserved;
};

struct alignas(kCacheLineBytes) FullPaBuildReport {
    uint32_t task_id;
    uint32_t builder_thread;
    uint32_t builder_warp;
    uint32_t builder_lane;
    uint32_t phase_bits;
    uint32_t output_count;
    uint32_t payload_words;
    uint32_t insert_poll_count;
    int64_t predecessor_observed;
    uint32_t prepare_count;
    uint32_t commit_count;
    uint32_t main_scalar_build_count;
    uint32_t main_scalar_commit_count;
    uint64_t launch_nonce;
};

struct alignas(kCacheLineBytes) FullPaExecutionWitness {
    volatile int64_t state;
    uint64_t launch_nonce;
    uint64_t witness_magic;
    uint32_t task_id;
    TaskKind kind;
    uint32_t execute_owner;
    uint32_t execution_count;
    uint64_t completion_sequence;
    uint64_t output_checksum;
    uint64_t fanin_ready_prefix;
};

struct alignas(kCacheLineBytes) FullPaTaskAllocationReport {
    AtomicLine task_base_plus_one;
    AtomicLine completion_vend_plus_one;
};

struct alignas(kCacheLineBytes) FullPaTask {
    FullPaTaskPlan plan;
    TaskCell completion;
    AtomicLine insert_completion;
    FullPaTaskAllocationReport allocation;
    SharedOutputCell outputs;
    WriterHistoryCell writer_history;
    SharedExecCell exec;
    FullPaBuildReport build_report;
    FullPaExecutionWitness execution_witness;
};

struct alignas(kCacheLineBytes) FullPaFatalControl {
    volatile int64_t state;
    uint8_t padding[kCacheLineBytes - sizeof(int64_t)];
};

struct alignas(kCacheLineBytes) FullPaDrainControl {
    AtomicLine builder_finished;
    AtomicLine done_count;
    AtomicLine alloc_done;
    AtomicLine aic_done;
    AtomicLine aiv_done;
    AtomicLine root_finished;
    AtomicLine arrivals[kDrainGroupCount];
};

struct alignas(kCacheLineBytes) FullPaExecDispatchState {
    AtomicLine aic_next;
    AtomicLine aiv_next;
    uint32_t aic_task_count;
    uint32_t aiv_task_count;
    uint8_t header_padding[kCacheLineBytes - 2U * sizeof(uint32_t)];
    uint32_t aic_task_ids[kAicTaskCapacity];
    uint32_t aiv_task_ids[kAivTaskCapacity];
};

struct alignas(kCacheLineBytes) FullPaHeapControl {
    AtomicLine shard_cursors[kSharedHeapShards];
    AtomicLine aggregate_vend;
};

struct alignas(kCacheLineBytes) FullPaOrdinaryMapCanary {
    AtomicLine head;
    AtomicLine tail;
    AtomicLine lookup_count;
    AtomicLine append_count;
};

struct alignas(kCacheLineBytes) FullPaRoleResult {
    uint32_t owner;
    OwnerRole role;
    uint32_t physical_block;
    uint32_t drain_group;
    uint32_t build_count;
    uint32_t commit_count;
    uint32_t execute_count;
    uint32_t ticket_count;
    uint32_t exhausted_ticket_count;
    uint32_t claim_count;
    uint32_t claim_lost_count;
    uint32_t max_busy_tokens;
    uint32_t final_busy_tokens;
    uint32_t completed_by_kind[static_cast<uint32_t>(TaskKind::Count)];
    uint32_t drain_arrival_count;
    uint32_t fatal_count;
    uint64_t launch_nonce;
    uint64_t reserved[5];
};

struct alignas(kCacheLineBytes) FullPaBuilderThreadReport {
    uint32_t thread_id;
    uint32_t warp_id;
    uint32_t lane_id;
    uint32_t active_leader;
    uint32_t task_count;
    uint32_t first_task;
    uint32_t last_task;
    uint32_t task_state_access_count;
    uint32_t prepare_count;
    uint32_t commit_count;
    uint32_t insert_wait_count;
    uint32_t reserved0;
    uint64_t launch_nonce;
    uint64_t checksum;
};

struct alignas(kCacheLineBytes) FullPaState {
    FullPaControl control;
    FullPaGuard guard_before_tasks;
    FullPaTask tasks[kMaxTasks];
    FullPaGuard guard_after_tasks;
    FullPaFatalControl fatal;
    FullPaDrainControl drain;
    FullPaExecDispatchState exec_dispatch;
    FullPaHeapControl heap;
    FullPaOrdinaryMapCanary ordinary_map;
    FullPaGuard guard_before_tokens;
    ExecutionToken tokens[kOwnerCount][kTokensPerOwner];
    FullPaGuard guard_after_tokens;
    FullPaGuard guard_before_roles;
    FullPaRoleResult roles[kOwnerCount];
    FullPaGuard guard_after_roles;
    FullPaGuard guard_before_builder_threads;
    FullPaBuilderThreadReport builder_threads[kBuilderThreadCount];
    FullPaGuard guard_after_builder_threads;
};

SIMT_CROSS_CORE_G0_ABI_INLINE uint64_t ExpectedGuardWord(uint64_t nonce, uint32_t guard_id, uint32_t word) {
    uint64_t value = kGuardMagic ^ nonce ^ (static_cast<uint64_t>(guard_id) << 32U) ^ word;
    value ^= value >> 29U;
    value *= 0x9E3779B97F4A7C15ULL;
    return value ^ (value >> 31U);
}

SIMT_CROSS_CORE_G0_ABI_INLINE uint64_t
ExpectedDescriptorPoisonWord(uint64_t nonce, uint32_t task_id, uint32_t output_slot, uint32_t word) {
    return kDescriptorPoisonWord ^ nonce ^ (static_cast<uint64_t>(task_id) << 32U) ^
           (static_cast<uint64_t>(output_slot) << 16U) ^ word;
}

SIMT_CROSS_CORE_G0_ABI_INLINE uint64_t ExpectedPayloadPoisonWord(uint64_t nonce, uint32_t task_id, uint32_t word) {
    return kTailPoisonWord ^ nonce ^ (static_cast<uint64_t>(task_id) << 32U) ^ word;
}

SIMT_CROSS_CORE_G0_ABI_INLINE uint64_t BuildingState(uint32_t task_id) {
    return EncodeExecState(ExecPhase::Building, kBuilderOwner, kUnboundOwner, ExecEngineClass::None, 0U, task_id);
}

SIMT_CROSS_CORE_G0_ABI_INLINE uint64_t BuiltState(uint32_t task_id) {
    const TaskExecShape shape = TaskShape(TaskKindAt(task_id));
    ExecPayloadLayout layout{};
    (void)ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout);
    return EncodeExecState(
        ExecPhase::Built, kBuilderOwner, kUnboundOwner, shape.engine_class, layout.payload_lines, task_id
    );
}

SIMT_CROSS_CORE_G0_ABI_INLINE uint64_t ClaimedState(uint32_t task_id, uint32_t execute_owner) {
    const TaskExecShape shape = TaskShape(TaskKindAt(task_id));
    ExecPayloadLayout layout{};
    (void)ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout);
    return EncodeExecState(
        ExecPhase::Claimed, kBuilderOwner, execute_owner, shape.engine_class, layout.payload_lines, task_id
    );
}

SIMT_CROSS_CORE_G0_ABI_INLINE uint64_t DoneState(uint32_t task_id, uint32_t execute_owner) {
    const TaskExecShape shape = TaskShape(TaskKindAt(task_id));
    ExecPayloadLayout layout{};
    (void)ComputeExecPayloadLayout(shape.tensor_count, shape.scalar_count, shape.fanin_count, layout);
    return EncodeExecState(
        ExecPhase::Done, kBuilderOwner, execute_owner, shape.engine_class, layout.payload_lines, task_id
    );
}

SIMT_CROSS_CORE_G0_ABI_INLINE uint64_t
ExecutionWitnessState(uint64_t nonce, uint32_t task_id, TaskKind kind, uint32_t execute_owner) {
    uint64_t value = kExecutionWitnessMagic ^ nonce ^ (static_cast<uint64_t>(task_id) << 17U) ^
                     (static_cast<uint64_t>(kind) << 49U) ^ (static_cast<uint64_t>(execute_owner) << 1U);
    value ^= value >> 23U;
    value *= 0xD6E8FEB86659FD93ULL;
    value ^= value >> 32U;
    return value == 0U ? 1U : value;
}

SIMT_CROSS_CORE_G0_ABI_INLINE bool
PayloadTensorOutputSource(uint32_t task_id, uint32_t tensor_index, uint32_t &producer_task, uint32_t &output_slot) {
    const TaskKind kind = TaskKindAt(task_id);
    if (kind == TaskKind::Qk) {
        if (tensor_index == 3U) {
            producer_task = task_id;
            output_slot = 0U;
            return true;
        }
        return false;
    }
    if (kind == TaskKind::Sf) {
        if (tensor_index == 0U) {
            producer_task = task_id - 1U;
            output_slot = 0U;
        } else if (tensor_index <= 3U) {
            producer_task = task_id;
            output_slot = tensor_index - 1U;
        } else {
            return false;
        }
        return true;
    }
    if (kind == TaskKind::Pv) {
        if (tensor_index == 0U) {
            producer_task = task_id - 1U;
            output_slot = 0U;
        } else if (tensor_index == 3U) {
            producer_task = task_id;
            output_slot = 0U;
        } else {
            return false;
        }
        return true;
    }
    if (kind != TaskKind::Up) {
        return false;
    }
    if (tensor_index <= 1U) {
        producer_task = task_id - 2U;
        output_slot = tensor_index + 1U;
    } else if (tensor_index == 2U) {
        producer_task = task_id - 1U;
        output_slot = 0U;
    } else if (tensor_index <= 5U) {
        producer_task = BatchTaskId(TaskBatch(task_id), TaskKind::Alloc);
        output_slot = 5U - tensor_index;
    } else {
        return false;
    }
    return true;
}

SIMT_CROSS_CORE_G0_ABI_INLINE bool
ResolveExternalPayloadTensor(uint32_t batch_count, uint32_t task_id, uint32_t tensor_index, TensorDesc &tensor) {
    const TaskKind kind = TaskKindAt(task_id);
    if (kind == TaskKind::Qk && tensor_index == 0U) {
        tensor = MakeQueryViewDescriptor(batch_count, TaskBatch(task_id));
        return true;
    }
    if (kind == TaskKind::Qk && tensor_index == 1U) {
        tensor = MakeKeyOrValueDescriptor(batch_count, false);
        return true;
    }
    if ((kind == TaskKind::Qk || kind == TaskKind::Pv) && tensor_index == 2U) {
        tensor = MakeBlockTableDescriptor(batch_count);
        return true;
    }
    if (kind == TaskKind::Pv && tensor_index == 1U) {
        tensor = MakeKeyOrValueDescriptor(batch_count, true);
        return true;
    }
    if (kind == TaskKind::Up && tensor_index == 6U) {
        tensor = MakeOutputViewDescriptor(batch_count, TaskBatch(task_id));
        return true;
    }
    return false;
}

SIMT_CROSS_CORE_G0_ABI_INLINE uint64_t FullPaStateBytes() { return sizeof(FullPaState); }

static_assert(sizeof(FullPaControl) == 128U, "G0 control must occupy two cache lines");
static_assert(sizeof(FullPaGuard) == kCacheLineBytes, "G0 guard ABI changed");
static_assert(sizeof(FullPaTaskPlan) == kCacheLineBytes, "G0 task plan ABI changed");
static_assert(sizeof(FullPaBuildReport) == kCacheLineBytes, "G0 build report ABI changed");
static_assert(sizeof(FullPaExecutionWitness) == kCacheLineBytes, "G0 execution witness ABI changed");
static_assert(offsetof(FullPaExecutionWitness, fanin_ready_prefix) == 56U, "G0 execution fanin witness offset changed");
static_assert(sizeof(FullPaTaskAllocationReport) == 128U, "G0 task allocation report ABI changed");
static_assert(sizeof(FullPaTask) == 7232U, "G0 task aggregate ABI changed");
static_assert(
    offsetof(FullPaTask, completion) == 64U && offsetof(FullPaTask, insert_completion) == 128U &&
        offsetof(FullPaTask, allocation) == 192U && offsetof(FullPaTask, outputs) == 320U &&
        offsetof(FullPaTask, writer_history) == 2368U && offsetof(FullPaTask, exec) == 2688U &&
        offsetof(FullPaTask, build_report) == 7104U && offsetof(FullPaTask, execution_witness) == 7168U,
    "G0 task subregion offsets changed"
);
static_assert(offsetof(FullPaExecDispatchState, aic_task_ids) == 192U, "AIC dispatch ids must begin on their own line");
static_assert(
    offsetof(FullPaExecDispatchState, aiv_task_ids) == 192U + sizeof(uint32_t) * kAicTaskCapacity,
    "AIV dispatch ids offset changed"
);
static_assert(sizeof(FullPaExecDispatchState) == 35008U, "G0 execution dispatch ABI changed");
static_assert(sizeof(FullPaRoleResult) == 128U, "G0 role result ABI changed");
static_assert(
    offsetof(FullPaRoleResult, owner) == 0U && offsetof(FullPaRoleResult, completed_by_kind) == 52U &&
        offsetof(FullPaRoleResult, drain_arrival_count) == 72U && offsetof(FullPaRoleResult, launch_nonce) == 80U &&
        offsetof(FullPaRoleResult, reserved) == 88U,
    "G0 named-field role publication offsets changed"
);
static_assert(sizeof(FullPaBuilderThreadReport) == kCacheLineBytes, "G0 builder thread report ABI changed");
static_assert(
    offsetof(FullPaState, tasks) % kCacheLineBytes == 0U && offsetof(FullPaState, tokens) % kCacheLineBytes == 0U &&
        offsetof(FullPaState, roles) % kCacheLineBytes == 0U &&
        offsetof(FullPaState, builder_threads) % kCacheLineBytes == 0U && sizeof(FullPaState) % kCacheLineBytes == 0U,
    "all G0 GM regions must remain cache-line aligned"
);

#undef SIMT_CROSS_CORE_G0_ABI_INLINE

}  // namespace pa_scheduler::simt_cross_core::g0

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_G0_FULL_PA_H
