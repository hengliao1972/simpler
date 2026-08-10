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

#include "pa_scheduler_core.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using namespace pa_scheduler;

constexpr uint32_t kReplayActors = kWorkers;
constexpr uint32_t kMixedBatches = 5;

uint32_t g_failures = 0;

void Check(bool condition, const char *label) {
    std::printf(
        "[REPLAY_ARGS_TEST] %-62s %s\n",
        label, condition ? "PASS" : "FAIL"
    );
    if (!condition) {
        ++g_failures;
    }
}

void HashWord(uint64_t value, uint64_t &hash) {
    // 参数指纹只服务 CPU 门槛；逐字段混入，避免把 actor 私有对象地址
    // 当成业务值。最终仍单独保存全部 SharedOutputRef 身份做精确比较。
    hash ^= value;
    hash *= UINT64_C(1099511628211);
}

void HashTensorDesc(const TensorDesc &desc, uint64_t &hash) {
    HashWord(desc.buffer_addr, hash);
    HashWord(desc.buffer_size, hash);
    HashWord(desc.owner_task_id, hash);
    HashWord(desc.start_offset, hash);
    HashWord(static_cast<uint32_t>(desc.version), hash);
    HashWord(desc.ndims, hash);
    HashWord(static_cast<uint32_t>(desc.dtype), hash);
    HashWord(desc.manual_dep ? 1U : 0U, hash);
    HashWord(desc.is_contiguous ? 1U : 0U, hash);
    HashWord(desc.child_memory, hash);
    for (uint32_t dim = 0; dim < kMaxTensorDims; ++dim) {
        HashWord(desc.shapes[dim], hash);
        HashWord(desc.strides[dim], hash);
    }
    HashWord(desc.extent_elem_cache, hash);
}

void HashCreateInfo(
    const TensorCreateInfo &info, uint64_t &hash
) {
    HashWord(info.initial_value, hash);
    HashWord(info.has_initial_value ? 1U : 0U, hash);
    HashWord(info.reserved0, hash);
    HashWord(info.start_offset, hash);
    HashWord(static_cast<uint32_t>(info.version), hash);
    HashWord(info.ndims, hash);
    HashWord(static_cast<uint32_t>(info.dtype), hash);
    HashWord(info.manual_dep ? 1U : 0U, hash);
    HashWord(info.is_contiguous ? 1U : 0U, hash);
    HashWord(info.child_memory, hash);
    for (uint32_t dim = 0; dim < kMaxTensorDims; ++dim) {
        HashWord(info.shapes[dim], hash);
    }
}

struct SharedRefFingerprint {
    int32_t producer_task_id;
    int16_t output_slot;
    uint8_t flags;
    uint8_t view_ndims;
    uint32_t view_shape0;
    uint32_t view_offset0;
};

bool SameSharedRef(
    const SharedRefFingerprint &lhs,
    const SharedRefFingerprint &rhs
) {
    return lhs.producer_task_id == rhs.producer_task_id &&
        lhs.output_slot == rhs.output_slot &&
        lhs.flags == rhs.flags &&
        lhs.view_ndims == rhs.view_ndims &&
        lhs.view_shape0 == rhs.view_shape0 &&
        lhs.view_offset0 == rhs.view_offset0;
}

struct TaskFingerprint {
    uint32_t task_id;
    uint32_t batch;
    uint32_t group;
    TaskKind kind;
    uint32_t tensor_count;
    uint32_t scalar_count;
    uint32_t shared_ref_count;
    uint64_t args_hash;
    std::array<SharedRefFingerprint, kMaxTaskTensors>
        shared_refs;
};

bool SameTaskFingerprint(
    const TaskFingerprint &lhs, const TaskFingerprint &rhs
) {
    if (lhs.task_id != rhs.task_id || lhs.batch != rhs.batch ||
        lhs.group != rhs.group || lhs.kind != rhs.kind ||
        lhs.tensor_count != rhs.tensor_count ||
        lhs.scalar_count != rhs.scalar_count ||
        lhs.shared_ref_count != rhs.shared_ref_count ||
        lhs.args_hash != rhs.args_hash) {
        return false;
    }
    for (uint32_t index = 0;
         index < lhs.shared_ref_count; ++index) {
        if (!SameSharedRef(
                lhs.shared_refs[index], rhs.shared_refs[index]
            )) {
            return false;
        }
    }
    return true;
}

struct ReplayCoverage {
    bool refs_precede_consumers = true;
    bool saw_shared_slot0 = false;
    bool saw_shared_slot1 = false;
    bool saw_shared_slot2 = false;
    bool saw_nonzero_view_offset = false;
    bool saw_full_group_shape = false;
    bool saw_short_group_shape = false;
    bool saw_scalar_one = false;
    bool saw_scalar_full_group = false;
};

bool FingerprintTaskArgs(
    const TaskArgs &args, uint32_t task_id,
    uint32_t batch, uint32_t group, TaskKind kind,
    TaskFingerprint &fingerprint, ReplayCoverage &coverage
) {
    if (args.has_error || args.tensor_count < 0 ||
        args.tensor_count > static_cast<int32_t>(kMaxTaskTensors) ||
        args.scalar_count < 0 ||
        args.scalar_count > static_cast<int32_t>(kMaxTaskScalars)) {
        return false;
    }

    fingerprint = {};
    fingerprint.task_id = task_id;
    fingerprint.batch = batch;
    fingerprint.group = group;
    fingerprint.kind = kind;
    fingerprint.tensor_count =
        static_cast<uint32_t>(args.tensor_count);
    fingerprint.scalar_count =
        static_cast<uint32_t>(args.scalar_count);
    uint64_t hash = UINT64_C(1469598103934665603);
    HashWord(task_id, hash);
    HashWord(batch, hash);
    HashWord(group, hash);
    HashWord(static_cast<uint32_t>(kind), hash);
    HashWord(fingerprint.tensor_count, hash);
    HashWord(fingerprint.scalar_count, hash);
    HashWord(args.launch_spec.core_num, hash);
    HashWord(args.launch_spec.require_sync_start ? 1U : 0U, hash);
    HashWord(args.explicit_deps, hash);
    HashWord(args.explicit_dep_count, hash);

    for (uint32_t index = 0;
         index < fingerprint.tensor_count; ++index) {
        const TaskTensorRef &reference = args.tensors[index];
        HashWord(static_cast<uint32_t>(args.tags[index]), hash);
        HashWord(static_cast<uint32_t>(reference.kind), hash);
        switch (reference.kind) {
            case TensorRefKind::LocalTensor:
                if (reference.pointer.local_tensor == nullptr) {
                    return false;
                }
                HashTensorDesc(
                    *reference.pointer.local_tensor, hash
                );
                coverage.saw_nonzero_view_offset |=
                    reference.pointer.local_tensor->start_offset != 0;
                break;
            case TensorRefKind::GmTensor:
                if (reference.pointer.gm_tensor == nullptr) {
                    return false;
                }
                HashTensorDesc(
                    *reference.pointer.gm_tensor, hash
                );
                break;
            case TensorRefKind::CreateInfo:
                if (reference.pointer.create_info == nullptr) {
                    return false;
                }
                HashCreateInfo(
                    *reference.pointer.create_info, hash
                );
                if (reference.pointer.create_info->ndims >= 2) {
                    const uint32_t dynamic_extent =
                        reference.pointer.create_info->shapes[1];
                    coverage.saw_full_group_shape |=
                        dynamic_extent ==
                            kPaBlocksPerRequest * kPaBlockSize;
                    coverage.saw_short_group_shape |=
                        dynamic_extent == kPaBlockSize;
                }
                break;
            case TensorRefKind::SharedOutputRef: {
                const FdwicOutputRef output_ref =
                    SharedOutputReference(reference);
                if (!IsPlainSharedOutputRef(output_ref) ||
                    output_ref.producer_task_id < 0 ||
                    static_cast<uint32_t>(
                        output_ref.producer_task_id
                    ) >= task_id ||
                    output_ref.output_slot < 0 ||
                    static_cast<uint32_t>(output_ref.output_slot) >=
                        kSharedOutputMaxPerTask ||
                    fingerprint.shared_ref_count >=
                        fingerprint.shared_refs.size()) {
                    coverage.refs_precede_consumers = false;
                    return false;
                }
                SharedRefFingerprint &saved =
                    fingerprint.shared_refs[
                        fingerprint.shared_ref_count++
                    ];
                saved = SharedRefFingerprint{
                    output_ref.producer_task_id,
                    output_ref.output_slot,
                    output_ref.flags,
                    output_ref.view_ndims,
                    output_ref.view_shape0,
                    output_ref.view_offset0,
                };
                HashWord(
                    static_cast<uint32_t>(
                        output_ref.producer_task_id
                    ),
                    hash
                );
                HashWord(
                    static_cast<uint16_t>(output_ref.output_slot),
                    hash
                );
                HashWord(output_ref.flags, hash);
                HashWord(output_ref.view_ndims, hash);
                HashWord(output_ref.view_shape0, hash);
                HashWord(output_ref.view_offset0, hash);
                coverage.saw_shared_slot0 |=
                    output_ref.output_slot == 0;
                coverage.saw_shared_slot1 |=
                    output_ref.output_slot == 1;
                coverage.saw_shared_slot2 |=
                    output_ref.output_slot == 2;
                break;
            }
        }
    }
    for (uint32_t index = 0;
         index < fingerprint.scalar_count; ++index) {
        const uint64_t value = args.scalars[index];
        HashWord(value, hash);
        coverage.saw_scalar_one |= value == 1;
        coverage.saw_scalar_full_group |=
            value == kPaBlocksPerRequest;
    }
    fingerprint.args_hash = hash;
    return true;
}

bool BuildArgsForKind(
    TaskKind kind, PaOrchestrationState &orchestration,
    TaskArgs &args, uint32_t batch, LocalStats &stats
) {
    switch (kind) {
        case TaskKind::Alloc:
            return BuildCallbackSubmitArgs<TaskKind::Alloc>(
                orchestration, args, batch, stats
            );
        case TaskKind::Qk:
            return BuildCallbackSubmitArgs<TaskKind::Qk>(
                orchestration, args, batch, stats
            );
        case TaskKind::Sf:
            return BuildCallbackSubmitArgs<TaskKind::Sf>(
                orchestration, args, batch, stats
            );
        case TaskKind::Pv:
            return BuildCallbackSubmitArgs<TaskKind::Pv>(
                orchestration, args, batch, stats
            );
        case TaskKind::Up:
            return BuildCallbackSubmitArgs<TaskKind::Up>(
                orchestration, args, batch, stats
            );
        case TaskKind::Count:
            return false;
    }
    return false;
}

struct BuilderTotals {
    uint64_t arg_resets;
    uint64_t views_created;
    uint64_t dynamic_create_infos;
    uint64_t tensor_args_added;
    uint64_t scalar_args_added;
};

bool SameBuilderTotals(
    const BuilderTotals &lhs, const BuilderTotals &rhs
) {
    return lhs.arg_resets == rhs.arg_resets &&
        lhs.views_created == rhs.views_created &&
        lhs.dynamic_create_infos == rhs.dynamic_create_infos &&
        lhs.tensor_args_added == rhs.tensor_args_added &&
        lhs.scalar_args_added == rhs.scalar_args_added;
}

struct ReplaySummary {
    std::vector<TaskFingerprint> tasks;
    std::array<uint32_t, kMixedBatches> groups_per_batch{};
    std::array<uint32_t, static_cast<uint32_t>(TaskKind::Count)>
        tasks_per_kind{};
    BuilderTotals totals{};
    ReplayCoverage coverage{};
};

bool ReplayOneTask(
    PaOrchestrationState &orchestration, TaskArgs &args,
    LocalStats &stats, uint32_t &next_task_id,
    uint32_t batch, uint32_t group, TaskKind kind,
    ReplaySummary &summary
) {
    const uint32_t task_id = next_task_id++;
    if (task_id >= kMaxTasks || kind >= TaskKind::Count) {
        return false;
    }

    // 与正式 shared callback 相同：每个 actor 先得到稳定 task/slot
    // output symbol；只有随后成为 Build winner 时才同步执行真实参数
    // callback。门槛让 96 个 actor 都走一遍 winner callback，用来证明
    // 任意 actor 赢得同一 task 时会产生相同参数，而不是注入答案表。
    SharedTaskOutputs outputs{};
    outputs.Reset(static_cast<int32_t>(task_id));
    if (!PrepareSharedTaskOutputs(
            outputs, static_cast<int32_t>(task_id), kind
        ) ||
        !BuildArgsForKind(
            kind, orchestration, args, batch, stats
        )) {
        return false;
    }

    TaskFingerprint fingerprint{};
    if (!FingerprintTaskArgs(
            args, task_id, batch, group, kind,
            fingerprint, summary.coverage
        )) {
        return false;
    }
    summary.tasks.push_back(fingerprint);
    ++summary.tasks_per_kind[static_cast<uint32_t>(kind)];
    AcceptTaskOutputs(orchestration, kind, outputs);
    return true;
}

bool ReplayActor(
    PaOrchestrationState &orchestration,
    const int32_t *context_lens, ReplaySummary &summary
) {
    InitPaOrchestration(
        orchestration, kMixedBatches, context_lens
    );
    TaskArgs args{};
    LocalStats stats{};
    uint32_t next_task_id = 0;

    for (uint32_t batch = 0;
         batch < kMixedBatches; ++batch) {
        BeginPaBatchForCallback(orchestration, batch);
        if (!ReplayOneTask(
                orchestration, args, stats, next_task_id,
                batch, /*group=*/0, TaskKind::Alloc, summary
            )) {
            return false;
        }

        uint32_t group = 0;
        for (uint64_t block_offset = 0;
             block_offset < orchestration.current_blocks;
             block_offset += kPaBlocksPerRequest, ++group) {
            // group 数由当前 actor 自己读取 context 后得到的 block 循环
            // 动态决定；测试不构造 SharedPaBatchPlan 或 dispatch table。
            PreparePaBlockGroup(orchestration, block_offset);
            if (!ReplayOneTask(
                    orchestration, args, stats, next_task_id,
                    batch, group, TaskKind::Qk, summary
                ) ||
                !ReplayOneTask(
                    orchestration, args, stats, next_task_id,
                    batch, group, TaskKind::Sf, summary
                ) ||
                !ReplayOneTask(
                    orchestration, args, stats, next_task_id,
                    batch, group, TaskKind::Pv, summary
                ) ||
                !ReplayOneTask(
                    orchestration, args, stats, next_task_id,
                    batch, group, TaskKind::Up, summary
                )) {
                return false;
            }
        }
        summary.groups_per_batch[batch] = group;
    }

    for (uint32_t task_id = 0;
         task_id < summary.tasks.size(); ++task_id) {
        if (summary.tasks[task_id].task_id != task_id) {
            return false;
        }
    }
    summary.totals = BuilderTotals{
        stats.result.arg_resets,
        stats.result.views_created,
        stats.result.dynamic_create_infos,
        stats.result.tensor_args_added,
        stats.result.scalar_args_added,
    };
    return next_task_id == summary.tasks.size();
}

bool SameReplaySummary(
    const ReplaySummary &lhs, const ReplaySummary &rhs
) {
    if (lhs.tasks.size() != rhs.tasks.size() ||
        lhs.groups_per_batch != rhs.groups_per_batch ||
        lhs.tasks_per_kind != rhs.tasks_per_kind ||
        !SameBuilderTotals(lhs.totals, rhs.totals)) {
        return false;
    }
    for (uint32_t task_id = 0;
         task_id < lhs.tasks.size(); ++task_id) {
        if (!SameTaskFingerprint(
                lhs.tasks[task_id], rhs.tasks[task_id]
            )) {
            return false;
        }
    }
    return true;
}

bool CheckNinetySixIndependentReplays(
    ReplaySummary &baseline
) {
    // 这五批分别由真实 block loop 动态形成 G0/G1/G2/G2/G4；数值只
    // 是外部 context 输入，不预先生成 task identity 或 dispatch plan。
    const int32_t context_lens[kMixedBatches] = {
        0, 8192, 8193, 16384, 32768,
    };
    std::array<PaOrchestrationState, kReplayActors> actors{};

    for (uint32_t actor = 0; actor < kReplayActors; ++actor) {
        ReplaySummary observed{};
        if (!ReplayActor(
                actors[actor], context_lens, observed
            )) {
            return false;
        }
        if (actor == 0) {
            baseline = observed;
        } else if (!SameReplaySummary(baseline, observed)) {
            return false;
        }
    }
    return true;
}

bool CheckMixedReplayCoverage(const ReplaySummary &summary) {
    const std::array<uint32_t, kMixedBatches> expected_groups{
        0, 1, 2, 2, 4,
    };
    const uint32_t total_groups = 9;
    return summary.groups_per_batch == expected_groups &&
        summary.tasks.size() ==
            kMixedBatches + 4U * total_groups &&
        summary.tasks_per_kind[
            static_cast<uint32_t>(TaskKind::Alloc)
        ] == kMixedBatches &&
        summary.tasks_per_kind[
            static_cast<uint32_t>(TaskKind::Qk)
        ] == total_groups &&
        summary.tasks_per_kind[
            static_cast<uint32_t>(TaskKind::Sf)
        ] == total_groups &&
        summary.tasks_per_kind[
            static_cast<uint32_t>(TaskKind::Pv)
        ] == total_groups &&
        summary.tasks_per_kind[
            static_cast<uint32_t>(TaskKind::Up)
        ] == total_groups;
}

bool CheckDynamicArgumentCoverage(const ReplaySummary &summary) {
    const ReplayCoverage &coverage = summary.coverage;
    return coverage.refs_precede_consumers &&
        coverage.saw_shared_slot0 &&
        coverage.saw_shared_slot1 &&
        coverage.saw_shared_slot2 &&
        coverage.saw_nonzero_view_offset &&
        coverage.saw_full_group_shape &&
        coverage.saw_short_group_shape &&
        coverage.saw_scalar_one &&
        coverage.saw_scalar_full_group;
}

}  // namespace

int main() {
    ReplaySummary baseline{};
    const bool replay_ok =
        CheckNinetySixIndependentReplays(baseline);
    Check(
        replay_ok,
        "96 independent real callback replays have identical fingerprints"
    );
    Check(
        replay_ok && CheckMixedReplayCoverage(baseline),
        "real block loops cover mixed G0/G1/G2/G4 task identities"
    );
    Check(
        replay_ok && CheckDynamicArgumentCoverage(baseline),
        "refs, producer order, views, dynamic shapes and scalars are covered"
    );

    std::printf(
        "[REPLAY_ARGS_TEST] actors=%u tasks=%zu status=%s\n",
        kReplayActors, baseline.tasks.size(),
        g_failures == 0 ? "PASS" : "FAIL"
    );
    return g_failures == 0 ? 0 : 1;
}
