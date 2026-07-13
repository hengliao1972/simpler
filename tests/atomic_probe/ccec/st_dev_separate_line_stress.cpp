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
// 测试目标：只压测两个 AIV 各自独占 64B cacheline 时 repeated st_dev 的终值；本用例
// 不包含任何同-line 数据路径，避免已知同-line mismatch 消耗压力或干扰退出码解释。
//
// 四个 mode 由 runner 放在独立 host 进程中执行：
//   mode 0：启动两个 AIV，逻辑 block0 与 block1 写 allocation 内 line0/line1；
//   mode 1：启动三个 AIV，仅逻辑 block0 与 block2 写 line0/line1；
//   mode 2：与 mode 0 拓扑相同，改写 line1/line2，与旧 separate control offset 对齐；
//   mode 3：与 mode 1 拓扑相同，改写 line1/line2。mode 1/3 中 block1 只写 participation、
//           自己的独占拓扑 marker，并与其他 block 等次数、同顺序参加每次 SyncAll；
//           它从始至终不访问两条被测 data cacheline。
//
// 每个 launch 执行 100 trial；每个活跃 AIV 在每个 trial 中对自己独占 line 的 word0
// 连续执行 257 次 st_dev，只在循环结束后执行一次 DSB。随后所有已启动 AIV SyncAll，
// block0 用 ld_dev 精确检查两个终值都等于各自最后一轮值，再次 SyncAll 后进入下一轮。
// 任一 mismatch 都必须让用例失败，同时保存首错的 actual、expected、trial 与 slot。
//
// 三条候选 data line、result、参与计数、三个 block marker 与 tail guard 各自独占
// 64B；每个 mode 只使用相邻两条 data line，并要求第三条保持全零。marker
// 记录 get_coreid()/get_subblockid()，并按本机 PTO-ISA TSYNC_CVID 的 A5 公式计算
// comm_slot。comm_slot 是软件 CV 通信配对编号，不能把它扩大解释为硬件物理组。
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(st_dev_separate_line_stress);

constexpr uint32_t ROUNDS = 257;
constexpr uint32_t TRIALS = 100;
constexpr uint32_t CACHELINE_WORDS = 16;
constexpr uint32_t DATA_BASE_LINE01 = 0;
constexpr uint32_t DATA_BASE_LINE12 = CACHELINE_WORDS;
constexpr uint32_t RESULT_BASE = 3 * CACHELINE_WORDS;
constexpr uint32_t PARTICIPATION_BASE = 4 * CACHELINE_WORDS;
constexpr uint32_t MARKER_BASE = 5 * CACHELINE_WORDS;
constexpr uint32_t MODE_BLOCK02 = 1;
constexpr uint32_t MODE_BLOCK01_SHIFTED = 2;
constexpr uint32_t MODE_BLOCK02_SHIFTED = 3;
constexpr uint32_t A5_GROUPS_PER_DIE = 18;
constexpr uint32_t A5_AIVS_PER_GROUP = 2;
constexpr uint32_t A5_CORES_PER_DIE = A5_GROUPS_PER_DIE * (A5_AIVS_PER_GROUP + 1);

__aicore__ inline uint32_t TrialValue(uint32_t slot, uint32_t trial, uint32_t round)
{
    return 0xA5000000u + slot * 0x01000000u + trial * 0x1000u + round;
}

__aicore__ inline bool UsesBlock02(uint32_t mode)
{
    return mode == MODE_BLOCK02 || mode == MODE_BLOCK02_SHIFTED;
}

__aicore__ inline uint32_t DataBase(uint32_t mode)
{
    return mode >= MODE_BLOCK01_SHIFTED ? DATA_BASE_LINE12 : DATA_BASE_LINE01;
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
    __gm__ uint32_t *storage, uint32_t participant, uint32_t mode)
{
    uint32_t core_id = static_cast<uint32_t>(get_coreid());
    uint32_t subblock_id = static_cast<uint32_t>(get_subblockid());
    uint32_t base = MARKER_BASE + participant * CACHELINE_WORDS;
    st_dev_b32(&storage[base + 0], 0xA1A00000u | participant);
    st_dev_b32(&storage[base + 1], participant);
    st_dev_b32(&storage[base + 2], core_id);
    st_dev_b32(&storage[base + 3], subblock_id);
    st_dev_b32(&storage[base + 4], TsyncCommSlot(core_id, subblock_id));
    st_dev_b32(&storage[base + 5], static_cast<uint32_t>(get_subblockdim()));
    st_dev_b32(&storage[base + 6], mode);
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(st_dev_separate_line_stress)(
    __gm__ uint32_t *storage, uint32_t mode)
{
    uint32_t participant = static_cast<uint32_t>(get_block_idx());
    uint32_t peer = UsesBlock02(mode) ? 2u : 1u;
    uint32_t data_base = DataBase(mode);
    bool active = participant == 0u || participant == peer;
    uint32_t slot = participant == peer ? 1u : 0u;

    atomicAdd(&storage[PARTICIPATION_BASE], 1u);
    WriteTopologyMarker(storage, participant, mode);
    dsb(DSB_ALL);
    ccec_sync_all();

    uint32_t errors = 0;
    uint32_t slot_errors[2] = {0u, 0u};
    uint32_t first_actual = 0;
    uint32_t first_expected = 0;
    uint32_t first_trial = 0xffffffffu;
    uint32_t first_slot = 0xffffffffu;
    for (uint32_t trial = 0; trial < TRIALS; ++trial) {
        if (active) {
            uint32_t address = data_base + slot * CACHELINE_WORDS;
            for (uint32_t round = 0; round < ROUNDS; ++round) {
                st_dev_b32(&storage[address], TrialValue(slot, trial, round));
            }
            dsb(DSB_ALL);
        }

        ccec_sync_all();
        if (participant == 0u) {
            for (uint32_t logical_slot = 0; logical_slot < 2; ++logical_slot) {
                uint32_t expected = TrialValue(logical_slot, trial, ROUNDS - 1);
                uint32_t actual =
                    ld_dev_b32(&storage[data_base + logical_slot * CACHELINE_WORDS]);
                if (actual != expected) {
                    if (errors == 0u) {
                        first_actual = actual;
                        first_expected = expected;
                        first_trial = trial;
                        first_slot = logical_slot;
                    }
                    ++slot_errors[logical_slot];
                    ++errors;
                }
            }
        }
        ccec_sync_all();
    }

    if (participant == 0u) {
        st_dev_b32(&storage[RESULT_BASE + 0], errors);
        st_dev_b32(&storage[RESULT_BASE + 1], TRIALS * 2u);
        st_dev_b32(&storage[RESULT_BASE + 2], 0x53504C54u);
        st_dev_b32(&storage[RESULT_BASE + 3], mode);
        st_dev_b32(&storage[RESULT_BASE + 4], UsesBlock02(mode) ? 3u : 2u);
        st_dev_b32(&storage[RESULT_BASE + 5], first_actual);
        st_dev_b32(&storage[RESULT_BASE + 6], first_expected);
        st_dev_b32(&storage[RESULT_BASE + 7], first_trial);
        st_dev_b32(&storage[RESULT_BASE + 8], first_slot);
        st_dev_b32(&storage[RESULT_BASE + 9], ROUNDS);
        st_dev_b32(&storage[RESULT_BASE + 10], TRIALS);
        st_dev_b32(&storage[RESULT_BASE + 11], slot_errors[0]);
        st_dev_b32(&storage[RESULT_BASE + 12], slot_errors[1]);
        dsb(DSB_ALL);
    }
    ccec_sync_all();
}
