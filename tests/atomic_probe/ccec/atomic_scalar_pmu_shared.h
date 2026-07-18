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

#ifndef TESTS_ATOMIC_PROBE_CCEC_ATOMIC_SCALAR_PMU_SHARED_H_
#define TESTS_ATOMIC_PROBE_CCEC_ATOMIC_SCALAR_PMU_SHARED_H_

#include <cstddef>
#include <cstdint>

namespace atomic_scalar_pmu {

// 三条路径使用同一份 kernel 和同一个 PMU 读数协议，只替换 gate 内的固定次数工作负载：
// EMPTY 量 gate/read 固有开销；SCALAR_CONTROL 量与 atomic 路径相同的标量递推；
// DEPENDENT_ATOMIC_ADD 让后一条 atomicAdd 的加数依赖前一条返回值，避免多条 atomic 并行掩盖等待时间。
enum class Mode : uint32_t {
    Empty = 0,
    ScalarControl = 1,
    DependentAtomicAdd = 2,
    Count = 3,
};

// Host launch 前只写本 cache line；kernel 在测量窗口中只读。PMU MMIO base 表
// 按 get_coreid() 的低 12 bit 索引，布局与公共 pmu_probe AICPU helper 完全一致。
struct alignas(64) ProbeControl {
    uint64_t pmu_register_bases;
    uint32_t mode;
    uint32_t rounds;
    uint64_t seed;
    uint64_t reserved[5];
};

// Atomic 目标独占 cache line，排除 result/control 的普通 GM 写或 DCCI 对原子值的影响。
struct alignas(64) AtomicTarget {
    volatile uint64_t value;
    uint64_t reserved[7];
};

// 单 AIV 独占写结果 cache line。sys_cycles 是 get_sys_cnt() 前后差；其余四项是同一
// gate 窗口内的 PMU total、scalar busy、I-cache request 和 I-cache miss 原始计数。
struct alignas(64) ProbeResult {
    uint64_t sys_cycles;
    uint64_t pmu_total_cycles;
    uint64_t pmu_scalar_busy;
    uint64_t pmu_icache_request;
    uint64_t pmu_icache_miss;
    uint64_t checksum;
    uint64_t pmu_ctrl_after_stop;
    uint64_t physical_core_id;
};

struct alignas(64) ProbeState {
    ProbeControl control;
    AtomicTarget target;
    ProbeResult result;
};

static_assert(sizeof(ProbeControl) == 64, "probe control must occupy one cache line");
static_assert(sizeof(AtomicTarget) == 64, "atomic target must occupy one cache line");
static_assert(sizeof(ProbeResult) == 64, "probe result must occupy one cache line");
static_assert(offsetof(ProbeState, target) == 64, "atomic target must start on its own cache line");
static_assert(offsetof(ProbeState, result) == 128, "probe result must start on its own cache line");
static_assert(sizeof(ProbeState) == 192, "probe state ABI changed unexpectedly");

}  // namespace atomic_scalar_pmu

#endif  // TESTS_ATOMIC_PROBE_CCEC_ATOMIC_SCALAR_PMU_SHARED_H_
