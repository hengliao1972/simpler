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

// 测试目标：用单个 AIV 精确核实“等待 I-cache miss 回填的周期是否计入 PMU
// scalar busy”。Host/AICPU 在 launch 前把 PMU slot0/1/2 配置为 scalar
// busy(0x1)、I-cache request(0x34)、I-cache miss(0x35)。本用例不含 atomic、
// GM 轮询、Vector/Cube/MTE 计算，因此 WARM/COLD 的差值只来自 target 的取指状态。
//
// 每次 kernel 的完整时序：
//   1. 关闭 PMU；对 host 写入的 control 独立 cache line 执行 DCCI + DSB；读取
//      mode、seed、PMU MMIO base，并用 read-to-clear 清空所有 PMU counter。
//   2. 在 PMU 窗口外准备 I-cache：
//        WARM：先调用一次被测 noinline target，使其约 8 KiB 指令体进入 I-cache；
//        COLD：调用约 32 KiB 的 noinline evictor，以超过 16 KiB 容量的顺序
//              指令流替换 I-cache 内容。
//      两条准备路径都返回可精确复算的 checksum，确保调用不能被编译器删除。
//   3. 两条路径在分支后汇合。get_sys_cnt 后打开 PMU，窗口内只从同一个调用点
//      调用同一个 noinline target 一次，然后立即关闭 PMU。两种 mode 的窗内动态
//      指令完全相同，仅 target 调用前的 I-cache 冷热状态不同。
//   4. PMU 关闭后才读取 total/scalar/request/miss；最后仅用 st_dev 发布结果，
//      并用 DSB 收口。Host 必须同时校验 target checksum、准备 checksum、mode echo、
//      physical core id 和 PMU gate 状态。
//
// 预期与判读：
//   - COLD 的 I-cache miss 必须显著高于 WARM，先证明冷热对照确实成立；
//   - 以 COLD-WARM 扣除同一 target 的固定执行成本。若 total 增量与 scalar busy
//     增量近似相同，miss 回填等待计入 scalar busy；若 total 显著增加而 scalar
//     busy 不同比例增加，则该等待形成 scalar-busy gap。
//   - 构建后还必须按最终 ELF 符号大小核实 target >= 8 KiB、evictor >= 32 KiB；
//     否则本用例只能算源码意图，不能算有效的 I-cache 驱逐实验。

#include "icache_scalar_pmu_shared.h"
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(icache_scalar_pmu);

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
    // A5 PMU counter 是 read-to-clear；逐项显式展开，保持 ld_dev offset 为编译期常量。
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

// 递归宏最终展开成固定数量的独立 volatile NOP。这里不用运行时循环，是为了让
// target/evictor 的静态指令 footprint 本身达到指定大小，而不是反复执行一个热循环。
#define ICACHE_PMU_NOPS_1() asm volatile("nop");
#define ICACHE_PMU_NOPS_2() ICACHE_PMU_NOPS_1() ICACHE_PMU_NOPS_1()
#define ICACHE_PMU_NOPS_4() ICACHE_PMU_NOPS_2() ICACHE_PMU_NOPS_2()
#define ICACHE_PMU_NOPS_8() ICACHE_PMU_NOPS_4() ICACHE_PMU_NOPS_4()
#define ICACHE_PMU_NOPS_16() ICACHE_PMU_NOPS_8() ICACHE_PMU_NOPS_8()
#define ICACHE_PMU_NOPS_32() ICACHE_PMU_NOPS_16() ICACHE_PMU_NOPS_16()
#define ICACHE_PMU_NOPS_64() ICACHE_PMU_NOPS_32() ICACHE_PMU_NOPS_32()
#define ICACHE_PMU_NOPS_128() ICACHE_PMU_NOPS_64() ICACHE_PMU_NOPS_64()
#define ICACHE_PMU_NOPS_256() ICACHE_PMU_NOPS_128() ICACHE_PMU_NOPS_128()
#define ICACHE_PMU_NOPS_512() ICACHE_PMU_NOPS_256() ICACHE_PMU_NOPS_256()
#define ICACHE_PMU_NOPS_1024() ICACHE_PMU_NOPS_512() ICACHE_PMU_NOPS_512()
#define ICACHE_PMU_NOPS_2048() ICACHE_PMU_NOPS_1024() ICACHE_PMU_NOPS_1024()
#define ICACHE_PMU_NOPS_4096() ICACHE_PMU_NOPS_2048() ICACHE_PMU_NOPS_2048()
#define ICACHE_PMU_NOPS_8192() ICACHE_PMU_NOPS_4096() ICACHE_PMU_NOPS_4096()

// 保持外部可见、used、noinline：WARM 预热调用和 gate 内测量调用必须指向同一符号，
// 不能被内联成两份物理代码。ELF 检查还会验证本函数的最终符号大小至少为 8 KiB。
extern "C" __aicore__ __attribute__((noinline, used, aligned(128))) uint64_t
icache_scalar_pmu_target(uint64_t seed)
{
    ICACHE_PMU_NOPS_2048()
    return icache_scalar_pmu::TargetOracle(seed);
}

// evictor 同样保持外部可见、used、noinline；8192 条 volatile NOP 形成约 32 KiB
// 顺序指令流。返回值由 host 复算，额外证明 COLD 准备调用确实完成。
extern "C" __aicore__ __attribute__((noinline, used, aligned(128))) uint64_t
icache_scalar_pmu_evictor(uint64_t seed)
{
    ICACHE_PMU_NOPS_8192()
    return icache_scalar_pmu::EvictorOracle(seed);
}

// 把 mode 分支封装在另一个 noinline 函数内：kernel 本体在准备调用返回以后没有
// WARM/COLD 控制流，防止 O3 对共同测量尾部做 tail duplication，进而在 gate 内
// 生成两个物理调用点。该函数本身完全位于 PMU start 之前。
extern "C" __aicore__ __attribute__((noinline, used)) uint64_t
icache_scalar_pmu_prepare(uint32_t mode_value, uint64_t seed)
{
    if (mode_value == static_cast<uint32_t>(icache_scalar_pmu::Mode::WarmTarget)) {
        return icache_scalar_pmu_target(seed);
    }
    if (mode_value == static_cast<uint32_t>(icache_scalar_pmu::Mode::ColdTarget)) {
        return icache_scalar_pmu_evictor(seed);
    }
    return 0;
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(icache_scalar_pmu)(
    __gm__ icache_scalar_pmu::ProbeState *state)
{
    // task-based profiler 可能在入口前已打开 PMU；先关闭，确保准备阶段绝不入窗。
    bisheng::cce::metrics_prof_stop();
    dcci(&state->control, SINGLE_CACHE_LINE);
    dsb(DSB_ALL);

    const uint32_t mode_value = state->control.mode;
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

    const uint64_t preparation_checksum = icache_scalar_pmu_prepare(mode_value, seed);

    // mode 分支在这里结束。以下测量窗口对 WARM/COLD 是同一个静态调用点和同一
    // target 符号，动态指令序列不再依赖 mode。
    const uint64_t sys_begin = static_cast<uint64_t>(get_sys_cnt());
    bisheng::cce::metrics_prof_start();
    const uint64_t target_checksum = icache_scalar_pmu_target(seed);
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

    __gm__ icache_scalar_pmu::ProbeResult *result = &state->result;
    Publish64(&result->sys_cycles, sys_end - sys_begin);
    Publish64(&result->pmu_total_cycles, pmu_total);
    Publish64(&result->pmu_scalar_busy, pmu_scalar);
    Publish64(&result->pmu_icache_request, pmu_icache_request);
    Publish64(&result->pmu_icache_miss, pmu_icache_miss);
    Publish64(&result->target_checksum, target_checksum);
    Publish64(&result->preparation_checksum, preparation_checksum);
    Publish64(&result->pmu_ctrl_after_stop, ctrl_after_stop);
    Publish64(&result->physical_core_id, physical_core_id);
    Publish64(&result->mode_echo, mode_value);
    dsb(DSB_ALL);
}

#undef ICACHE_PMU_NOPS_8192
#undef ICACHE_PMU_NOPS_4096
#undef ICACHE_PMU_NOPS_2048
#undef ICACHE_PMU_NOPS_1024
#undef ICACHE_PMU_NOPS_512
#undef ICACHE_PMU_NOPS_256
#undef ICACHE_PMU_NOPS_128
#undef ICACHE_PMU_NOPS_64
#undef ICACHE_PMU_NOPS_32
#undef ICACHE_PMU_NOPS_16
#undef ICACHE_PMU_NOPS_8
#undef ICACHE_PMU_NOPS_4
#undef ICACHE_PMU_NOPS_2
#undef ICACHE_PMU_NOPS_1
