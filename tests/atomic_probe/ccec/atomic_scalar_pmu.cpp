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

// 测试目标：用单个 AIV、无竞争、无轮询的固定次数窗口，精确区分
// atomicAdd 等待返回的时间是否计入 PMU scalar busy。Host/AICPU 在 launch 前已把
// PMU slot0/1/2 配置为 scalar busy(0x1)、I-cache request(0x34)、I-cache miss(0x35)。
//
// 每次 kernel 的完整时序：
//   1. 先关闭 PMU，再对 host 写入的 control 单独 cache line 做 DCCI + DSB，避免复用
//      ProbeState 时沿用上一次 kernel 的旧 control。Atomic target 不做普通 load/store
//      或 DCCI，始终保持为一条独占 cache line 的 raw atomic 目标。
//   2. 按 physical core id 从 host 传入的寄存器基址表取本 AIV PMU base，然后用
//      read-to-clear 清空所有 counter。
//   3. 同一个 get_sys_cnt 时间窗内执行 metrics_prof_start/stop；三种 mode 只替换
//      gate 内部的固定 rounds 工作负载：
//        EMPTY：不做工作，测 gate 与计时固有开销；
//        SCALAR_CONTROL：只在 scalar 寄存器中执行与 atomic 相同的数据依赖递推；
//        DEPENDENT_ATOMIC_ADD：对独占 target 执行 atomicAdd，下一轮 addend 由上一轮
//        atomicAdd 的返回值计算，不允许多条 atomic 并行隐藏单条等待。
//   4. 关闭 PMU 后才读 total/scalar/request/miss，最后仅用 st_dev 把结果发布到
//      result 独占 cache line，并用 DSB 收口。
//
// CONTROL 与 ATOMIC 共用可由 host 精确复算的递推（uint64_t 模 2^64）：
//   value=seed, delta=1, checksum=0;
//   每轮 old=value/atomicAdd(target, delta) 的返回值，checksum+=old，
//   delta=1+(old&1)；CONTROL 另执行 value+=本轮 delta。
// 解读时应对多个 rounds 取斜率并扣除 EMPTY/CONTROL：若 atomic 的 total 斜率显著
// 增长而 scalar 斜率不同比例增长，atomic 等待不属于 scalar busy；若两者同比例增长，
// 则等待被计入 scalar busy。

#include "atomic_scalar_pmu_shared.h"
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(atomic_scalar_pmu);

namespace {

constexpr uint32_t kPmuPhysicalSubcores = 108;
constexpr uint64_t kPmuCtrl0Offset = 0x4200ULL;

__aicore__ __attribute__((always_inline)) inline int32_t *PmuCounterBase(uint64_t register_base)
{
    // 以 PMU_CTRL_0(0x4200) 为基准后，所有 counter 都落在 ld_dev 的 12-bit immediate 范围内。
    return reinterpret_cast<int32_t *>(register_base + kPmuCtrl0Offset);
}

__aicore__ __attribute__((always_inline)) inline void ClearPmuCounters(uint64_t register_base)
{
    int32_t *base = PmuCounterBase(register_base);
    // A5 PMU counter 是 read-to-clear；显式展开保证每个 ld_dev offset 为编译期常量。
    (void)ld_dev(base, 0x10);
    (void)ld_dev(base, 0x18);
    (void)ld_dev(base, 0x20);
    (void)ld_dev(base, 0x28);
    (void)ld_dev(base, 0x30);
    (void)ld_dev(base, 0x38);
    (void)ld_dev(base, 0x40);
    (void)ld_dev(base, 0x48);
    (void)ld_dev(base, 0x50);
    (void)ld_dev(base, 0x54);
    (void)ld_dev(base, 0x60);
    (void)ld_dev(base, 0x64);
}

__aicore__ __attribute__((always_inline)) inline uint64_t ReadPmuScalar(uint64_t register_base)
{
    return static_cast<uint32_t>(ld_dev(PmuCounterBase(register_base), 0x10));
}

__aicore__ __attribute__((always_inline)) inline uint64_t ReadPmuIcacheRequest(uint64_t register_base)
{
    return static_cast<uint32_t>(ld_dev(PmuCounterBase(register_base), 0x18));
}

__aicore__ __attribute__((always_inline)) inline uint64_t ReadPmuIcacheMiss(uint64_t register_base)
{
    return static_cast<uint32_t>(ld_dev(PmuCounterBase(register_base), 0x20));
}

__aicore__ __attribute__((always_inline)) inline uint64_t ReadPmuTotal(uint64_t register_base)
{
    int32_t *base = PmuCounterBase(register_base);
    const uint64_t low = static_cast<uint32_t>(ld_dev(base, 0x60));
    const uint64_t high = static_cast<uint32_t>(ld_dev(base, 0x64));
    return low | (high << 32);
}

__aicore__ __attribute__((always_inline)) inline void Publish64(__gm__ uint64_t *address, uint64_t value)
{
    __builtin_cce_st_dev(value, address, 0);
}

}  // namespace

extern "C" __global__ __aicore__ void KERNEL_ENTRY(atomic_scalar_pmu)(
    __gm__ atomic_scalar_pmu::ProbeState *state)
{
    using atomic_scalar_pmu::Mode;

    // task-based profiler 可能在入口前已打开 PMU；先关闭，确保 control DCCI 和准备阶段不入窗。
    bisheng::cce::metrics_prof_stop();
    dcci(&state->control, SINGLE_CACHE_LINE);
    dsb(DSB_ALL);

    const uint32_t mode_value = state->control.mode;
    const uint32_t rounds = state->control.rounds;
    const uint64_t seed = state->control.seed;
    const uint32_t physical_core_id = static_cast<uint32_t>(get_coreid()) & 0x0fffU;

    uint64_t register_base = 0;
    if (state->control.pmu_register_bases != 0 && physical_core_id < kPmuPhysicalSubcores) {
        __gm__ uint64_t *register_bases =
            reinterpret_cast<__gm__ uint64_t *>(state->control.pmu_register_bases);
        register_base = register_bases[physical_core_id];
    }
    if (register_base != 0) {
        ClearPmuCounters(register_base);
    }

    uint64_t checksum = 0;
    uint64_t delta = 1;
    uint64_t scalar_value = seed;

    const uint64_t sys_begin = static_cast<uint64_t>(get_sys_cnt());
    bisheng::cce::metrics_prof_start();

    if (mode_value == static_cast<uint32_t>(Mode::ScalarControl)) {
        for (uint32_t round = 0; round < rounds; ++round) {
            const uint64_t old = scalar_value;
            scalar_value += delta;
            checksum += old;
            delta = 1 + (old & 1U);
        }
    } else if (mode_value == static_cast<uint32_t>(Mode::DependentAtomicAdd)) {
        __gm__ uint64_t *target = const_cast<__gm__ uint64_t *>(&state->target.value);
        for (uint32_t round = 0; round < rounds; ++round) {
            // delta 直接依赖上一轮 old；除第一轮外，后一条 atomic 必须等前一条返回。
            const uint64_t old = atomicAdd(target, delta);
            checksum += old;
            delta = 1 + (old & 1U);
        }
    }
    // Empty 和非法 mode 都保持空窗；host 仅会发布 enum 中的三个合法值。

    bisheng::cce::metrics_prof_stop();
    const uint64_t sys_end = static_cast<uint64_t>(get_sys_cnt());
    const uint64_t ctrl_after_stop = static_cast<uint64_t>(get_ctrl());

    uint64_t pmu_total = 0;
    uint64_t pmu_scalar = 0;
    uint64_t pmu_icache_request = 0;
    uint64_t pmu_icache_miss = 0;
    if (register_base != 0) {
        // counter 为 read-to-clear，每项只读一次，且必须在 stop 后执行。
        pmu_scalar = ReadPmuScalar(register_base);
        pmu_icache_request = ReadPmuIcacheRequest(register_base);
        pmu_icache_miss = ReadPmuIcacheMiss(register_base);
        pmu_total = ReadPmuTotal(register_base);
    }

    __gm__ atomic_scalar_pmu::ProbeResult *result = &state->result;
    Publish64(&result->sys_cycles, sys_end - sys_begin);
    Publish64(&result->pmu_total_cycles, pmu_total);
    Publish64(&result->pmu_scalar_busy, pmu_scalar);
    Publish64(&result->pmu_icache_request, pmu_icache_request);
    Publish64(&result->pmu_icache_miss, pmu_icache_miss);
    Publish64(&result->checksum, checksum);
    Publish64(&result->pmu_ctrl_after_stop, ctrl_after_stop);
    Publish64(&result->physical_core_id, physical_core_id);
    dsb(DSB_ALL);
}
