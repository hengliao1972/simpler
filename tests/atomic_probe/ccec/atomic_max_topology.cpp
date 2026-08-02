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

// A5 atomicMax 地址拓扑探针。
//
// 目标不是模拟 CPU 原子，而是直接回答 Claim 分层方案成立前必须确认的
// 硬件问题：把 N 个同地址 atomicMax 分散到 G 条独立地址后，A5 是否真的
// 能并行处理；以及“组内选举 + 组代表访问根节点”是否比 flat 热点更短。
//
// 每一轮的 task token 单调增加，同一地址上第一个观察到旧值较小的 AIV
// 获胜。轮间使用 FFTS SyncAll，屏障不计入 candidate_ticks，但会进入
// loop_ticks，后者用于观察最慢参与者是否真正提前完成。
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(atomic_max_topology);

namespace {

constexpr uint32_t kModeFlat = 0;
constexpr uint32_t kModeGrouped = 1;
constexpr uint32_t kModeTwoLevel = 2;
constexpr uint32_t kModeBypassPrefilter = 3;
constexpr uint32_t kOpFetchMax = 0;
constexpr uint32_t kOpCompareExchange = 1;
constexpr uint32_t kOpExchange = 2;
constexpr uint32_t kConfigMagic = 0x41544d58U;
constexpr uint32_t kLocalOffsetBytes = 64;
constexpr uint32_t kRootOffsetBytes = 32768;

struct alignas(64) ProbeConfig {
    uint32_t magic;
    uint32_t mode;
    uint32_t group_count;
    uint32_t stride_bytes;
    uint32_t rounds;
    uint32_t operation;
    uint32_t prefilter_nops;
    uint32_t preferred_nops;
    uint8_t padding[32];
};
static_assert(sizeof(ProbeConfig) == 64, "config must occupy one cache line");

struct alignas(64) ProbeResult {
    uint64_t candidate_ticks;
    uint64_t candidate_max_ticks;
    uint64_t loop_ticks;
    uint64_t checksum;
    uint32_t local_wins;
    uint32_t root_wins;
    uint32_t participant;
    uint32_t completed_rounds;
    uint32_t atomic_attempts;
    uint32_t bypass_skips;
    uint32_t preferred_wins;
    uint8_t padding[4];
};
static_assert(sizeof(ProbeResult) == 64, "one result must occupy one cache line");

__aicore__ inline __gm__ int64_t *LineAt(__gm__ uint8_t *storage, uint32_t line, uint32_t stride_bytes) {
    return reinterpret_cast<__gm__ int64_t *>(storage + kLocalOffsetBytes + static_cast<uint64_t>(line) * stride_bytes);
}

__aicore__ inline int64_t Elect(__gm__ int64_t *address, int64_t token, uint32_t operation) {
    if (operation == kOpCompareExchange) {
        return atomicCAS(address, token - 1, token);
    }
    if (operation == kOpExchange) {
        return atomicExch(address, token);
    }
    if (operation == kOpFetchMax) {
        return atomicMax(address, token);
    }
    return token;
}

template <uint32_t Count>
__aicore__ inline void EmitNops() {
#pragma unroll
    for (uint32_t index = 0; index < Count; ++index) {
        asm volatile("nop");
    }
}

__aicore__ inline void DelayByNops(uint32_t count) {
    if (count == 10U) {
        EmitNops<10>();
    } else if (count == 50U) {
        EmitNops<50>();
    } else if (count == 100U) {
        EmitNops<100>();
    } else if (count == 1000U) {
        EmitNops<1000>();
    }
}

}  // namespace

extern "C" __global__ __aicore__ void
KERNEL_ENTRY(atomic_max_topology)(__gm__ uint8_t *storage, __gm__ ProbeResult *results) {
    // host 会在同一块 GM 上连续启动多个变体；普通 scalar load 可能命中
    // 上一次 kernel 留下的 D-cache。配置读取不属于计时窗口，直接复用已
    // 验证的 ld_dev bypass 语义，避免用 DCCI 干扰旁边的原子测试地址。
    __gm__ uint32_t *config_words = reinterpret_cast<__gm__ uint32_t *>(storage);
    const uint32_t magic = ld_dev_b32(&config_words[0]);
    const uint32_t mode = ld_dev_b32(&config_words[1]);
    const uint32_t group_count = ld_dev_b32(&config_words[2]);
    const uint32_t stride_bytes = ld_dev_b32(&config_words[3]);
    const uint32_t rounds = ld_dev_b32(&config_words[4]);
    const uint32_t operation = ld_dev_b32(&config_words[5]);
    const uint32_t prefilter_nops = ld_dev_b32(&config_words[6]);
    const uint32_t preferred_nops = ld_dev_b32(&config_words[7]);
    const uint32_t participant = static_cast<uint32_t>(get_block_idx());
    const uint32_t participants = static_cast<uint32_t>(get_block_num());
    __gm__ ProbeResult *result = &results[participant];

    uint64_t candidate_ticks = 0;
    uint64_t candidate_max_ticks = 0;
    uint64_t checksum = 0;
    uint32_t local_wins = 0;
    uint32_t root_wins = 0;
    uint32_t atomic_attempts = 0;
    uint32_t bypass_skips = 0;
    uint32_t preferred_wins = 0;

    // 参数错误也必须让所有核走相同控制流，避免部分核进入 SyncAll 后挂住。
    const bool valid =
        magic == kConfigMagic && participants != 0 && participants <= 64 && rounds != 0 && stride_bytes >= 64 &&
        (stride_bytes & 63U) == 0 && operation <= kOpExchange &&
        ((mode == kModeFlat && group_count == 1) ||
         ((mode == kModeGrouped || mode == kModeTwoLevel) && group_count != 0 && group_count <= participants) ||
         (mode == kModeBypassPrefilter && group_count == 1 && operation == kOpCompareExchange &&
          (prefilter_nops == 0U || prefilter_nops == 10U || prefilter_nops == 50U || prefilter_nops == 100U) &&
          (preferred_nops == 0U || preferred_nops == 1000U)));

    ccec_sync_all();
    const uint64_t loop_begin = static_cast<uint64_t>(get_sys_cnt());
    if (valid) {
        const uint32_t group = participant % group_count;
        __gm__ int64_t *root = reinterpret_cast<__gm__ int64_t *>(storage + kRootOffsetBytes);

        for (uint32_t round = 0; round < rounds; ++round) {
            ccec_sync_all();
            const int64_t token = static_cast<int64_t>(round + 1U);
            const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());

            if (mode == kModeFlat) {
                ++atomic_attempts;
                const int64_t old = Elect(root, token, operation);
                const bool won = old < token;
                root_wins += won ? 1U : 0U;
                checksum += static_cast<uint64_t>(old + token + (won ? 1 : 0));
            } else if (mode == kModeBypassPrefilter) {
                // 每轮轮换一个首选候选。首选直接 CAS；其余候选只在短暂
                // 本地延后后仍观察到旧 token 时才回退 CAS。ld_dev 读到
                // 旧值只会增加 CAS，最终 owner 仍完全由 CAS 返回值裁定。
                const uint32_t preferred = round % participants;
                bool should_attempt = participant == preferred;
                int64_t observed = token - 1;
                if (should_attempt) {
                    DelayByNops(preferred_nops);
                } else {
                    DelayByNops(prefilter_nops);
                    observed = static_cast<int64_t>(ld_dev_b64(reinterpret_cast<__gm__ uint64_t *>(root)));
                    should_attempt = observed < token;
                }
                if (should_attempt) {
                    ++atomic_attempts;
                    const int64_t old = atomicCAS(root, token - 1, token);
                    const bool won = old == token - 1;
                    root_wins += won ? 1U : 0U;
                    preferred_wins += won && participant == preferred ? 1U : 0U;
                    checksum += static_cast<uint64_t>(old + token + (won ? 11 : 0));
                } else {
                    ++bypass_skips;
                    checksum += static_cast<uint64_t>(observed + token + 13);
                }
            } else {
                __gm__ int64_t *local = LineAt(storage, group, stride_bytes);
                ++atomic_attempts;
                const int64_t local_old = Elect(local, token, operation);
                const bool local_won = local_old < token;
                local_wins += local_won ? 1U : 0U;
                checksum += static_cast<uint64_t>(local_old + token + (local_won ? 3 : 0));

                if (mode == kModeTwoLevel && local_won) {
                    ++atomic_attempts;
                    const int64_t root_old = Elect(root, token, operation);
                    const bool root_won = root_old < token;
                    root_wins += root_won ? 1U : 0U;
                    checksum += static_cast<uint64_t>(root_old + token + (root_won ? 7 : 0));
                }
            }

            const uint64_t end = static_cast<uint64_t>(get_sys_cnt());
            const uint64_t delta = end - begin;
            candidate_ticks += delta;
            if (delta > candidate_max_ticks) candidate_max_ticks = delta;
        }
    }
    ccec_sync_all();
    const uint64_t loop_end = static_cast<uint64_t>(get_sys_cnt());

    // 每个 AIV 只写自己的独占 cache line；整行写完后统一发布给 host。
    result->candidate_ticks = candidate_ticks;
    result->candidate_max_ticks = candidate_max_ticks;
    result->loop_ticks = loop_end - loop_begin;
    result->checksum = checksum;
    result->local_wins = local_wins;
    result->root_wins = root_wins;
    result->participant = participant;
    result->completed_rounds = valid ? rounds : 0;
    result->atomic_attempts = atomic_attempts;
    result->bypass_skips = bypass_skips;
    result->preferred_wins = preferred_wins;
    dcci(result, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb(DSB_ALL);
}
