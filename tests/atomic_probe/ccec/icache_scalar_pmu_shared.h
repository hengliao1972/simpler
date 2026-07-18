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

#ifndef TESTS_ATOMIC_PROBE_CCEC_ICACHE_SCALAR_PMU_SHARED_H_
#define TESTS_ATOMIC_PROBE_CCEC_ICACHE_SCALAR_PMU_SHARED_H_

#include <cstddef>
#include <cstdint>

namespace icache_scalar_pmu {

#if defined(__CCE_AICORE__)
#define ICACHE_SCALAR_PMU_SHARED_FN __aicore__
#else
#define ICACHE_SCALAR_PMU_SHARED_FN
#endif

// 两条路径的 PMU 窗口内都只调用一次、且调用同一个 noinline target：
// WARM 在窗外先调用一次 target；COLD 在窗外执行大于 16 KiB I-cache 容量的 evictor。
// 模式分支本身不进入 PMU gate，因而不会改变窗内 target 的动态指令序列。
enum class Mode : uint32_t {
    WarmTarget = 0,
    ColdTarget = 1,
    Count = 2,
};

// A5 scalar I-cache 容量为 16 KiB。AICore 标量指令为 4B；target 用 2048 条
// volatile NOP 形成约 8 KiB 指令体，evictor 用 8192 条形成约 32 KiB 顺序指令流。
// 构建脚本仍应从最终 ELF 的符号大小复核这两个下界，不能只相信源码常量。
constexpr uint32_t kTargetNopCount = 2048;
constexpr uint32_t kEvictorNopCount = 8192;
constexpr uint64_t kTargetXor = 0xd6e8feb86659fd93ULL;
constexpr uint64_t kTargetMultiplier = 0x9e3779b185ebca87ULL;
constexpr uint64_t kTargetAddend = 0xa0761d6478bd642fULL;
constexpr uint64_t kEvictorXor = 0xe7037ed1a0b428dbULL;
constexpr uint64_t kEvictorMultiplier = 0x8ebc6af09c88c6e3ULL;

// Host 与 device 共用完全相同的无符号递推；uint64_t 溢出按模 2^64 定义。
// checksum 只负责证明 target/evictor 的调用确实发生，不参与 PMU 窗口分类。
ICACHE_SCALAR_PMU_SHARED_FN constexpr uint64_t TargetOracle(uint64_t seed)
{
    uint64_t value = seed ^ kTargetXor;
    value ^= value >> 29;
    value *= kTargetMultiplier;
    value += kTargetAddend;
    value ^= value >> 31;
    return value;
}

ICACHE_SCALAR_PMU_SHARED_FN constexpr uint64_t EvictorOracle(uint64_t seed)
{
    uint64_t value = seed ^ kEvictorXor;
    value ^= value >> 23;
    value *= kEvictorMultiplier;
    value ^= value >> 27;
    return value;
}

// Host launch 前只写本 cache line；kernel 在 PMU 窗口外 DCCI 后读取。
// pmu_register_bases 按 get_coreid() 的低 12 bit 索引，两个 scalar PMU 探针
// 共用同一份 108 physical sub-core MMIO base 表协议。
struct alignas(64) ProbeControl {
    uint64_t pmu_register_bases;
    uint32_t mode;
    uint32_t reserved0;
    uint64_t seed;
    uint64_t reserved[5];
};

// 前五项是同一 target 单次调用窗口内的时间和 PMU 原始计数。
// target_checksum 在两种 mode 下都必须等于 TargetOracle(seed)：这是窗内动态
// 工作负载一致的功能 oracle。preparation_checksum 在 WARM 下也等于 TargetOracle，
// 在 COLD 下等于 EvictorOracle，用来证明对应的窗外准备路径没有被编译器删除。
// 两条 cache line 均由唯一 AIV 用 st_dev 发布，kernel 末尾统一 DSB。
struct alignas(64) ProbeResult {
    uint64_t sys_cycles;
    uint64_t pmu_total_cycles;
    uint64_t pmu_scalar_busy;
    uint64_t pmu_icache_request;
    uint64_t pmu_icache_miss;
    uint64_t target_checksum;
    uint64_t preparation_checksum;
    uint64_t pmu_ctrl_after_stop;

    uint64_t physical_core_id;
    uint64_t mode_echo;
    uint64_t reserved[6];
};

struct alignas(64) ProbeState {
    ProbeControl control;
    ProbeResult result;
};

static_assert(sizeof(ProbeControl) == 64, "probe control must occupy one cache line");
static_assert(sizeof(ProbeResult) == 128, "probe result must occupy two cache lines");
static_assert(offsetof(ProbeState, result) == 64, "probe result must start on its own cache line");
static_assert(sizeof(ProbeState) == 192, "probe state ABI changed unexpectedly");

#undef ICACHE_SCALAR_PMU_SHARED_FN

}  // namespace icache_scalar_pmu

#endif  // TESTS_ATOMIC_PROBE_CCEC_ICACHE_SCALAR_PMU_SHARED_H_
