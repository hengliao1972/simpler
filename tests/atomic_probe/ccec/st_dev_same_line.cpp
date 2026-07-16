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
// 测试目标：比较 logical block0+1 与 logical block0+2 两种 repeated st_dev
// 同-cacheline 路径，并隔离单 AIV 同址路径。host 用独立进程执行三种 mode：
//
//   mode 0：启动两个 AIV，逻辑 AIV0 与 AIV1 写数据，保留原始回归路径；
//   mode 1：启动三个 AIV，仅逻辑 AIV0 与 AIV2 写数据；AIV1 从 kernel 开始到结束
//           都不访问任何被测数据 line，只写 participation、自己的拓扑 marker 并参加每次 SyncAll；
//   mode 2：只启动一个 AIV，验证同核同址连续 st_dev。
//
// mode 0/1 的两核数据路径均执行 100 trial × 257 round：
//   1. 两个活跃 AIV 各写同一 64B line 内自己的 4B slot，循环结束后才 DSB；
//   2. 两个活跃 AIV 各写独占 64B line，循环结束后才 DSB；
//   3. 两个活跃 AIV 写同一 64B line 的不同 slot，但每轮 st_dev 后都 DSB。
//
// 每个 trial 都在两个同步点之间完成“并发写 → DSB → AIV0逐 slot ld_dev 检查”。
// 参与计数、每核执行 marker、结果和各组被测数据互不共用 64B line；两条同-line
// 路径都是正确性门禁：终值必须等于各自最后一轮值；观察到 mismatch 时必须失败，
// 不能增加逐轮 DSB 或把问题存在本身改写为通过条件。这里的分-line 路径只是低压力
// 正确性对照，不代表分-line 安全；独立压力由 st_dev_separate_line_stress 覆盖。
// 逐轮 DSB 路径保留为完成顺序对照。
//
// 本用例直接验证的是逻辑 block 0/1 与 block 0/2。每个 block 的独占 marker line
// 记录 get_coreid()/get_subblockid()，并按本机 PTO-ISA TSYNC_CVID 的 A5 公式计算
// comm_slot。comm_slot 是软件 CV 通信配对编号，不能把它扩大解释为硬件物理组。
//
// mode 2 中 host 只启动一个 AIV，该 AIV 对同一个 4B 地址连续执行
// 257 次 st_dev，再在 loop 末执行一次 DSB 并检查终值；逐写 DSB 路径作为同核 control。
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(st_dev_same_line);

constexpr uint32_t ROUNDS = 257;
constexpr uint32_t TRIALS = 100;
constexpr uint32_t ADJACENT_SAME_BASE = 0;
constexpr uint32_t ADJACENT_SEPARATE_BASE = 16;
constexpr uint32_t ADJACENT_ORDERED_BASE = 48;
constexpr uint32_t GAP_SAME_BASE = 64;
constexpr uint32_t GAP_SEPARATE_BASE = 80;
constexpr uint32_t GAP_ORDERED_BASE = 112;
constexpr uint32_t RESULT_BASE = 128;
constexpr uint32_t PARTICIPATION_BASE = 144;
constexpr uint32_t MARKER_BASE = 160;
constexpr uint32_t CACHELINE_WORDS = 16;
constexpr uint32_t MODE_BLOCK01 = 0;
constexpr uint32_t MODE_BLOCK02 = 1;
constexpr uint32_t MODE_SINGLE_AIV = 2;
constexpr uint32_t A5_GROUPS_PER_DIE = 18;
constexpr uint32_t A5_AIVS_PER_GROUP = 2;
constexpr uint32_t A5_CORES_PER_DIE = A5_GROUPS_PER_DIE * (A5_AIVS_PER_GROUP + 1);

__aicore__ inline uint32_t TrialValue(uint32_t participant, uint32_t trial, uint32_t round)
{
    return 0xA5000000u + participant * 0x01000000u + trial * 0x1000u + round;
}

__aicore__ inline bool PairParticipant(uint32_t participant, uint32_t peer)
{
    return participant == 0 || participant == peer;
}

__aicore__ inline uint32_t PairSlot(uint32_t participant, uint32_t peer)
{
    return participant == peer ? 1u : 0u;
}

__aicore__ inline uint32_t TsyncCommSlot(uint32_t core_id, uint32_t subblock_id)
{
    int32_t die_id = static_cast<int32_t>(core_id / A5_CORES_PER_DIE);
    int32_t local_core_id = static_cast<int32_t>(core_id % A5_CORES_PER_DIE);
    return static_cast<uint32_t>(die_id * static_cast<int32_t>(A5_GROUPS_PER_DIE) +
        (local_core_id - static_cast<int32_t>(A5_GROUPS_PER_DIE) -
         static_cast<int32_t>(subblock_id)) / static_cast<int32_t>(A5_AIVS_PER_GROUP));
}

__aicore__ inline void WriteTopologyMarker(
    __gm__ uint32_t *gx, uint32_t participant, uint32_t mode)
{
    uint32_t core_id = static_cast<uint32_t>(get_coreid());
    uint32_t subblock_id = static_cast<uint32_t>(get_subblockid());
    uint32_t base = MARKER_BASE + participant * CACHELINE_WORDS;
    st_dev_b32(&gx[base + 0], 0xA1A00000u | participant);
    st_dev_b32(&gx[base + 1], participant);
    st_dev_b32(&gx[base + 2], core_id);
    st_dev_b32(&gx[base + 3], subblock_id);
    st_dev_b32(&gx[base + 4], TsyncCommSlot(core_id, subblock_id));
    st_dev_b32(&gx[base + 5], static_cast<uint32_t>(get_subblockdim()));
    st_dev_b32(&gx[base + 6], mode);
}

__aicore__ inline uint32_t RunSameLine(
    __gm__ uint32_t *gx, uint32_t participant, uint32_t peer, uint32_t base,
    uint32_t &first_actual, uint32_t &first_expected)
{
    bool active = PairParticipant(participant, peer);
    uint32_t slot = PairSlot(participant, peer);
    uint32_t errors = 0;
    for (uint32_t trial = 0; trial < TRIALS; ++trial) {
        if (active) {
            for (uint32_t round = 0; round < ROUNDS; ++round) {
                st_dev_b32(&gx[base + slot], TrialValue(slot, trial, round));
            }
            dsb(DSB_ALL);
        }
        ccec_sync_all();
        if (participant == 0) {
            for (uint32_t logical = 0; logical < 2; ++logical) {
                uint32_t expected = TrialValue(logical, trial, ROUNDS - 1);
                uint32_t actual = ld_dev_b32(&gx[base + logical]);
                if (actual != expected) {
                    if (errors == 0) {
                        first_actual = actual;
                        first_expected = expected;
                    }
                    ++errors;
                }
            }
        }
        ccec_sync_all();
    }
    return errors;
}

__aicore__ inline uint32_t RunSeparateLine(
    __gm__ uint32_t *gx, uint32_t participant, uint32_t peer, uint32_t base,
    uint32_t &first_actual, uint32_t &first_expected)
{
    bool active = PairParticipant(participant, peer);
    uint32_t slot = PairSlot(participant, peer);
    uint32_t errors = 0;
    for (uint32_t trial = 0; trial < TRIALS; ++trial) {
        if (active) {
            uint32_t address = base + slot * CACHELINE_WORDS;
            for (uint32_t round = 0; round < ROUNDS; ++round) {
                st_dev_b32(&gx[address], TrialValue(slot, trial, round));
            }
            dsb(DSB_ALL);
        }
        ccec_sync_all();
        if (participant == 0) {
            for (uint32_t logical = 0; logical < 2; ++logical) {
                uint32_t expected = TrialValue(logical, trial, ROUNDS - 1);
                uint32_t actual = ld_dev_b32(&gx[base + logical * CACHELINE_WORDS]);
                if (actual != expected) {
                    if (errors == 0) {
                        first_actual = actual;
                        first_expected = expected;
                    }
                    ++errors;
                }
            }
        }
        ccec_sync_all();
    }
    return errors;
}

__aicore__ inline uint32_t RunSameLineOrdered(
    __gm__ uint32_t *gx, uint32_t participant, uint32_t peer, uint32_t base)
{
    bool active = PairParticipant(participant, peer);
    uint32_t slot = PairSlot(participant, peer);
    uint32_t errors = 0;
    for (uint32_t trial = 0; trial < TRIALS; ++trial) {
        if (active) {
            for (uint32_t round = 0; round < ROUNDS; ++round) {
                st_dev_b32(&gx[base + slot], TrialValue(slot, trial, round));
                dsb(DSB_ALL);
            }
        }
        ccec_sync_all();
        if (participant == 0) {
            for (uint32_t logical = 0; logical < 2; ++logical) {
                uint32_t expected = TrialValue(logical, trial, ROUNDS - 1);
                uint32_t actual = ld_dev_b32(&gx[base + logical]);
                if (actual != expected) {
                    ++errors;
                }
            }
        }
        ccec_sync_all();
    }
    return errors;
}

__aicore__ inline uint32_t RunSingleAddress(
    __gm__ uint32_t *gx, uint32_t base, bool per_write_dsb,
    uint32_t &first_actual, uint32_t &first_expected)
{
    uint32_t errors = 0;
    for (uint32_t trial = 0; trial < TRIALS; ++trial) {
        for (uint32_t round = 0; round < ROUNDS; ++round) {
            st_dev_b32(&gx[base], TrialValue(0, trial, round));
            if (per_write_dsb) {
                dsb(DSB_ALL);
            }
        }
        if (!per_write_dsb) {
            dsb(DSB_ALL);
        }
        uint32_t expected = TrialValue(0, trial, ROUNDS - 1);
        uint32_t actual = ld_dev_b32(&gx[base]);
        if (actual != expected) {
            if (errors == 0) {
                first_actual = actual;
                first_expected = expected;
            }
            ++errors;
        }
    }
    return errors;
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(st_dev_same_line)(
    __gm__ uint32_t *gx, uint32_t mode)
{
    uint32_t participant = (uint32_t)get_block_idx();
    if (mode == MODE_SINGLE_AIV) {
        if (participant != 0) {
            return;
        }
        uint32_t single_first_actual = 0;
        uint32_t single_first_expected = 0;
        uint32_t ordered_first_actual = 0;
        uint32_t ordered_first_expected = 0;
        atomicAdd(&gx[PARTICIPATION_BASE], 1u);
        WriteTopologyMarker(gx, participant, mode);
        dsb(DSB_ALL);

        uint32_t single_errors = RunSingleAddress(
            gx, ADJACENT_SAME_BASE, false, single_first_actual, single_first_expected);
        uint32_t ordered_errors = RunSingleAddress(
            gx, ADJACENT_SEPARATE_BASE, true, ordered_first_actual, ordered_first_expected);

        st_dev_b32(&gx[RESULT_BASE + 0], single_errors);
        st_dev_b32(&gx[RESULT_BASE + 1], ordered_errors);
        st_dev_b32(&gx[RESULT_BASE + 2], 0u);
        st_dev_b32(&gx[RESULT_BASE + 3], single_first_actual);
        st_dev_b32(&gx[RESULT_BASE + 4], single_first_expected);
        st_dev_b32(&gx[RESULT_BASE + 5], TRIALS);
        st_dev_b32(&gx[RESULT_BASE + 6], 0x351051A5u);
        st_dev_b32(&gx[RESULT_BASE + 7], ordered_first_actual);
        st_dev_b32(&gx[RESULT_BASE + 8], ordered_first_expected);
        st_dev_b32(&gx[RESULT_BASE + 9], 0u);
        st_dev_b32(&gx[RESULT_BASE + 10], 0u);
        st_dev_b32(&gx[RESULT_BASE + 11], 0u);
        st_dev_b32(&gx[RESULT_BASE + 12], MODE_SINGLE_AIV);
        st_dev_b32(&gx[RESULT_BASE + 13], 1u);
        st_dev_b32(&gx[RESULT_BASE + 14], 0u);
        st_dev_b32(&gx[RESULT_BASE + 15], 0u);
        dsb(DSB_ALL);
        return;
    }
    if (mode != MODE_BLOCK01 && mode != MODE_BLOCK02) {
        return;
    }

    uint32_t adjacent_first_actual = 0;
    uint32_t adjacent_first_expected = 0;
    uint32_t gap_first_actual = 0;
    uint32_t gap_first_expected = 0;
    uint32_t separate_first_actual = 0;
    uint32_t separate_first_expected = 0;

    atomicAdd(&gx[PARTICIPATION_BASE], 1u);
    WriteTopologyMarker(gx, participant, mode);
    dsb(DSB_ALL);
    ccec_sync_all();

    uint32_t adjacent_same_errors = 0;
    uint32_t adjacent_separate_errors = 0;
    uint32_t adjacent_ordered_errors = 0;
    uint32_t gap_same_errors = 0;
    uint32_t gap_separate_errors = 0;
    uint32_t gap_ordered_errors = 0;
    if (mode == MODE_BLOCK01) {
        adjacent_same_errors = RunSameLine(
            gx, participant, 1, ADJACENT_SAME_BASE,
            adjacent_first_actual, adjacent_first_expected);
        adjacent_separate_errors = RunSeparateLine(
            gx, participant, 1, ADJACENT_SEPARATE_BASE,
            separate_first_actual, separate_first_expected);
        adjacent_ordered_errors = RunSameLineOrdered(
            gx, participant, 1, ADJACENT_ORDERED_BASE);
    } else {
        // AIV1 在 mode 1 的全部三条路径中只到达 SyncAll，不访问 GAP_* data。
        gap_same_errors = RunSameLine(
            gx, participant, 2, GAP_SAME_BASE, gap_first_actual, gap_first_expected);
        gap_separate_errors = RunSeparateLine(
            gx, participant, 2, GAP_SEPARATE_BASE,
            separate_first_actual, separate_first_expected);
        gap_ordered_errors = RunSameLineOrdered(
            gx, participant, 2, GAP_ORDERED_BASE);
    }

    if (participant == 0) {
        st_dev_b32(&gx[RESULT_BASE + 0], adjacent_same_errors);
        st_dev_b32(&gx[RESULT_BASE + 1], adjacent_separate_errors);
        st_dev_b32(&gx[RESULT_BASE + 2], adjacent_ordered_errors);
        st_dev_b32(&gx[RESULT_BASE + 3], adjacent_first_actual);
        st_dev_b32(&gx[RESULT_BASE + 4], adjacent_first_expected);
        st_dev_b32(&gx[RESULT_BASE + 5], TRIALS);
        st_dev_b32(&gx[RESULT_BASE + 6], 0x3510C1A5u);
        st_dev_b32(&gx[RESULT_BASE + 7], gap_same_errors);
        st_dev_b32(&gx[RESULT_BASE + 8], gap_separate_errors);
        st_dev_b32(&gx[RESULT_BASE + 9], gap_ordered_errors);
        st_dev_b32(&gx[RESULT_BASE + 10], gap_first_actual);
        st_dev_b32(&gx[RESULT_BASE + 11], gap_first_expected);
        st_dev_b32(&gx[RESULT_BASE + 12], mode);
        st_dev_b32(
            &gx[RESULT_BASE + 13], mode == MODE_BLOCK01 ? 2u : 3u);
        st_dev_b32(&gx[RESULT_BASE + 14], separate_first_actual);
        st_dev_b32(&gx[RESULT_BASE + 15], separate_first_expected);
        dsb(DSB_ALL);
    }
    ccec_sync_all();
}
