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

#ifndef TESTS_ATOMIC_PROBE_CCEC_BUILD_DISPATCH_CONTENTION_SHARED_H
#define TESTS_ATOMIC_PROBE_CCEC_BUILD_DISPATCH_CONTENTION_SHARED_H

#include <cstddef>
#include <cstdint>

namespace build_dispatch_probe {

constexpr uint32_t kConfigMagic = 0x42544450U;
constexpr uint32_t kAicWorkers = 32;
constexpr uint32_t kAivWorkers = 64;
constexpr uint32_t kWorkers = kAicWorkers + kAivWorkers;
constexpr uint32_t kTasks = 1280;
constexpr uint32_t kTournamentGroups = 8;
constexpr uint32_t kTournamentNodeStride = 512;

enum class Mode : uint32_t {
    PerTaskTournament = 0,
    CentralTicket = 1,
    Empty = 2,
};

struct alignas(64) ProbeConfig {
    uint32_t magic;
    uint32_t mode;
    uint32_t workers;
    uint32_t tasks;
    uint32_t groups;
    uint32_t node_stride;
    uint8_t padding[40];
};
static_assert(sizeof(ProbeConfig) == 64, "probe config must occupy one cache line");

struct alignas(64) AtomicLine {
    volatile int64_t value;
    uint8_t padding[56];
};
static_assert(sizeof(AtomicLine) == 64, "atomic control must occupy one cache line");

// 与 standalone shared Claim 保持相同的 512B 节点间距，避免把更紧凑的
// 地址布局误当成生产 G8 两级 CAS 的性能。
struct alignas(64) TournamentNode {
    AtomicLine owner;
    uint8_t padding[kTournamentNodeStride - sizeof(AtomicLine)];
};
static_assert(sizeof(TournamentNode) == kTournamentNodeStride, "tournament node stride changed");

struct alignas(64) TournamentTask {
    TournamentNode root;
    TournamentNode local[kTournamentGroups];
};
static_assert(
    sizeof(TournamentTask) == kTournamentNodeStride * (kTournamentGroups + 1U), "per-task tournament layout changed"
);

struct alignas(64) ProbeState {
    ProbeConfig config;
    AtomicLine ready_workers;
    AtomicLine start_release;
    AtomicLine central_ticket;
    TournamentTask tournaments[kTasks];
};
static_assert(offsetof(ProbeState, tournaments) == 256, "probe controls must occupy four cache lines");

// 每个 Scalar 只写自己的独占结果行。begin/end 使用同一 1ns SYS_CNT，host
// 以最早 begin 到最晚 end 计算混合 96 核的完整 device span。
struct alignas(64) ProbeResult {
    uint64_t begin_tick;
    uint64_t end_tick;
    uint64_t task_id_sum;
    uint64_t task_id_xor;
    uint32_t worker_id;
    uint32_t role;
    uint32_t valid_tasks;
    uint32_t atomic_attempts;
    uint32_t local_wins;
    uint32_t root_wins;
    uint32_t errors;
    uint32_t completed_mode;
};
static_assert(sizeof(ProbeResult) == 64, "one result must occupy one cache line");

struct KernelArgs {
    uint64_t state_pointer;
    uint64_t results_pointer;
};
static_assert(sizeof(KernelArgs) == 16, "unexpected mixed-kernel argument ABI");

}  // namespace build_dispatch_probe

#endif  // TESTS_ATOMIC_PROBE_CCEC_BUILD_DISPATCH_CONTENTION_SHARED_H
