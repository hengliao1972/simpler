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

// A5 mixed 32 AIC + 64 AIV Build 发放原子拓扑探针。
//
// 两条路径只保留 owner 发放原子，不包含 PA 构参、TensorMap、Build 或执行：
//   1. 当前生产形状：每 task 96 个 local CAS，八个 local winner 再做 root CAS；
//   2. 候选形状：96 核持续从同一个 FetchAdd cursor 取得唯一 task ticket。
//
// 探针回答的是 A5 上“调用量下降是否足以覆盖单地址串行化”，不把 CPU 原子
// 性能当代理，也不直接宣称完整 Submit 会得到相同比例的收益。
#include "build_dispatch_contention_shared.h"
#include "ccec_utils.h"

namespace {

using namespace build_dispatch_probe;

constexpr int64_t kAtomicReadIdentity = (-9223372036854775807LL - 1LL);

__aicore__ inline bool ConfigValid(__gm__ ProbeState *state) {
    __gm__ uint32_t *words = reinterpret_cast<__gm__ uint32_t *>(&state->config);
    return ld_dev_b32(&words[0]) == kConfigMagic && ld_dev_b32(&words[2]) == kWorkers &&
           ld_dev_b32(&words[3]) == kTasks && ld_dev_b32(&words[4]) == kTournamentGroups &&
           ld_dev_b32(&words[5]) == kTournamentNodeStride && ld_dev_b32(&words[1]) <= 2U;
}

__aicore__ inline void WaitForAllWorkers(__gm__ ProbeState *state) {
    const int64_t prior = atomicAdd(const_cast<__gm__ int64_t *>(&state->ready_workers.value), int64_t{1});
    if (prior + 1 == static_cast<int64_t>(kWorkers)) {
        (void)atomicExch(const_cast<__gm__ int64_t *>(&state->start_release.value), int64_t{1});
    }
    while (atomicMax(const_cast<__gm__ int64_t *>(&state->start_release.value), kAtomicReadIdentity) != 1) {
        asm volatile("nop");
    }
}

__aicore__ inline void RunPerTaskTournament(__gm__ ProbeState *state, uint32_t worker_id, ProbeResult &result) {
    const uint32_t group = worker_id % kTournamentGroups;
    for (uint32_t task_id = 0; task_id < kTasks; ++task_id) {
        __gm__ TournamentTask *task = &state->tournaments[task_id];
        ++result.atomic_attempts;
        const int64_t local_old = atomicCAS(
            const_cast<__gm__ int64_t *>(&task->local[group].owner.value), int64_t{-1}, static_cast<int64_t>(task_id)
        );
        if (local_old == -1) {
            ++result.local_wins;
            ++result.atomic_attempts;
            const int64_t root_old = atomicCAS(
                const_cast<__gm__ int64_t *>(&task->root.owner.value), int64_t{-1}, static_cast<int64_t>(task_id)
            );
            if (root_old == -1) {
                ++result.root_wins;
                ++result.valid_tasks;
                result.task_id_sum += task_id;
                result.task_id_xor ^= task_id;
            } else if (root_old != static_cast<int64_t>(task_id)) {
                ++result.errors;
            }
        } else if (local_old != static_cast<int64_t>(task_id)) {
            ++result.errors;
        }
    }
}

__aicore__ inline void RunCentralTicket(__gm__ ProbeState *state, ProbeResult &result) {
    while (true) {
        ++result.atomic_attempts;
        const int64_t ticket = atomicAdd(const_cast<__gm__ int64_t *>(&state->central_ticket.value), int64_t{1});
        if (ticket < 0) {
            ++result.errors;
            return;
        }
        if (ticket >= static_cast<int64_t>(kTasks)) {
            return;
        }
        ++result.valid_tasks;
        result.task_id_sum += static_cast<uint64_t>(ticket);
        result.task_id_xor ^= static_cast<uint64_t>(ticket);
    }
}

__aicore__ inline void
RunProbe(__gm__ ProbeState *state, __gm__ ProbeResult *results, uint32_t worker_id, uint32_t role) {
    ProbeResult local{};
    local.worker_id = worker_id;
    local.role = role;
    const bool valid = worker_id < kWorkers && ConfigValid(state);
    WaitForAllWorkers(state);

    asm volatile("" ::: "memory");
    local.begin_tick = static_cast<uint64_t>(get_sys_cnt());
    if (valid) {
        const uint32_t mode = ld_dev_b32(reinterpret_cast<__gm__ uint32_t *>(&state->config) + 1);
        if (mode == static_cast<uint32_t>(build_dispatch_probe::Mode::PerTaskTournament)) {
            RunPerTaskTournament(state, worker_id, local);
        } else if (mode == static_cast<uint32_t>(build_dispatch_probe::Mode::CentralTicket)) {
            RunCentralTicket(state, local);
        }
        local.completed_mode = mode + 1U;
    } else {
        ++local.errors;
    }
    asm volatile("" ::: "memory");
    local.end_tick = static_cast<uint64_t>(get_sys_cnt());

    __gm__ ProbeResult *result = &results[worker_id];
    result->begin_tick = local.begin_tick;
    result->end_tick = local.end_tick;
    result->task_id_sum = local.task_id_sum;
    result->task_id_xor = local.task_id_xor;
    result->worker_id = local.worker_id;
    result->role = local.role;
    result->valid_tasks = local.valid_tasks;
    result->atomic_attempts = local.atomic_attempts;
    result->local_wins = local.local_wins;
    result->root_wins = local.root_wins;
    result->errors = local.errors;
    result->completed_mode = local.completed_mode;
    dcci(result, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb(DSB_ALL);
}

}  // namespace

#if defined(PA_BUILD_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(build_dispatch_contention_0_mix_aic, 1, 2);

extern "C" __global__ __aicore__ void build_dispatch_contention_0_mix_aic(
    __gm__ build_dispatch_probe::ProbeState *state, __gm__ build_dispatch_probe::ProbeResult *results
) {
    const uint32_t worker_id = static_cast<uint32_t>(get_block_idx());
    RunProbe(state, results, worker_id, 0);
}
#elif defined(PA_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(build_dispatch_contention_0_mix_aiv, 1, 2);

extern "C" __global__ __aicore__ void build_dispatch_contention_0_mix_aiv(
    __gm__ build_dispatch_probe::ProbeState *state, __gm__ build_dispatch_probe::ProbeResult *results
) {
    const uint32_t vector_id = static_cast<uint32_t>(get_block_idx() * get_subblockdim() + get_subblockid());
    const uint32_t worker_id = build_dispatch_probe::kAicWorkers + vector_id;
    RunProbe(state, results, worker_id, 1);
}
#else
#error "Compile once with PA_BUILD_AIC and once with PA_BUILD_AIV"
#endif
