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
// CCEC 同构实现见 AscendC st_dev_single_core_stress.asc。只启动一个 AIV；写者和
// ld_dev 读者相同，不调用 SyncAll，也不存在跨核数据共享。mode 0..4 只在一组
// repeated st_dev 后执行 DSB，mode 5/6 在每次 st_dev 后立即执行 DSB。
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(st_dev_single_core_stress);

constexpr uint32_t CACHELINE_WORDS = 16;
constexpr uint32_t RESULT_BASE = 3 * CACHELINE_WORDS;
constexpr uint32_t PARTICIPATION_BASE = 4 * CACHELINE_WORDS;
constexpr uint32_t MARKER_BASE = 5 * CACHELINE_WORDS;
constexpr uint32_t ROUNDS = 257;
constexpr uint32_t TRIALS = 100;
constexpr uint32_t INVALID_U32 = 0xffffffffu;

constexpr uint32_t RESULT_MAGIC_VALUE = 0x53314352u;
constexpr uint32_t RESULT_FINISH_VALUE = 0x46494e49u;
constexpr uint32_t MARKER_MAGIC_VALUE = 0x53434f52u;

constexpr uint32_t A5_GROUPS_PER_DIE = 18;
constexpr uint32_t A5_AIVS_PER_GROUP = 2;
constexpr uint32_t A5_CORES_PER_DIE = A5_GROUPS_PER_DIE * (A5_AIVS_PER_GROUP + 1);

enum class ProbeMode : uint32_t {
    SINGLE_LINE0_LOOP_DSB = 0,
    SINGLE_LINE1_LOOP_DSB = 1,
    SINGLE_LINE2_LOOP_DSB = 2,
    DUAL_LINE01_LOOP_DSB = 3,
    DUAL_LINE12_LOOP_DSB = 4,
    SINGLE_LINE1_PER_WRITE_DSB = 5,
    DUAL_LINE12_PER_WRITE_DSB = 6,
};

__aicore__ inline uint32_t DataStartLine(uint32_t mode)
{
    if (mode == static_cast<uint32_t>(ProbeMode::SINGLE_LINE1_LOOP_DSB) ||
        mode == static_cast<uint32_t>(ProbeMode::SINGLE_LINE1_PER_WRITE_DSB) ||
        mode == static_cast<uint32_t>(ProbeMode::DUAL_LINE12_LOOP_DSB) ||
        mode == static_cast<uint32_t>(ProbeMode::DUAL_LINE12_PER_WRITE_DSB)) {
        return 1u;
    }
    return mode == static_cast<uint32_t>(ProbeMode::SINGLE_LINE2_LOOP_DSB) ? 2u : 0u;
}

__aicore__ inline uint32_t SlotCount(uint32_t mode)
{
    return mode == static_cast<uint32_t>(ProbeMode::DUAL_LINE01_LOOP_DSB) ||
        mode == static_cast<uint32_t>(ProbeMode::DUAL_LINE12_LOOP_DSB) ||
        mode == static_cast<uint32_t>(ProbeMode::DUAL_LINE12_PER_WRITE_DSB) ? 2u : 1u;
}

__aicore__ inline bool UsesPerWriteDsb(uint32_t mode)
{
    return mode == static_cast<uint32_t>(ProbeMode::SINGLE_LINE1_PER_WRITE_DSB) ||
        mode == static_cast<uint32_t>(ProbeMode::DUAL_LINE12_PER_WRITE_DSB);
}

__aicore__ inline uint32_t TrialValue(uint32_t mode, uint32_t slot, uint32_t trial, uint32_t round)
{
    return 0xA0000000u + mode * 0x01000000u + slot * 0x00100000u + trial * 0x1000u + round;
}

__aicore__ inline uint32_t TsyncCommSlot(uint32_t core_id, uint32_t subblock_id)
{
    int32_t die_id = static_cast<int32_t>(core_id / A5_CORES_PER_DIE);
    int32_t local_core_id = static_cast<int32_t>(core_id % A5_CORES_PER_DIE);
    return static_cast<uint32_t>(die_id * static_cast<int32_t>(A5_GROUPS_PER_DIE) +
        (local_core_id - static_cast<int32_t>(A5_GROUPS_PER_DIE) -
         static_cast<int32_t>(subblock_id)) / static_cast<int32_t>(A5_AIVS_PER_GROUP));
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(st_dev_single_core_stress)(
    __gm__ uint32_t *storage, uint32_t mode)
{
    uint32_t participant = static_cast<uint32_t>(get_block_idx());
    uint32_t block_num = static_cast<uint32_t>(get_block_num());
    uint32_t data_start_line = DataStartLine(mode);
    uint32_t slots = SlotCount(mode);
    bool per_write_dsb = UsesPerWriteDsb(mode);
    __gm__ uint32_t *result = storage + RESULT_BASE;
    __gm__ uint32_t *participation = storage + PARTICIPATION_BASE;
    __gm__ uint32_t *marker = storage + MARKER_BASE;

    uint32_t core_id = static_cast<uint32_t>(get_coreid());
    uint32_t subblock_id = static_cast<uint32_t>(get_subblockid());
    atomicAdd(&participation[0], 1u);
    (void)atomicExch(&marker[0], MARKER_MAGIC_VALUE);
    (void)atomicExch(&marker[1], participant);
    (void)atomicExch(&marker[2], block_num);
    (void)atomicExch(&marker[3], core_id);
    (void)atomicExch(&marker[4], subblock_id);
    (void)atomicExch(&marker[5], TsyncCommSlot(core_id, subblock_id));
    (void)atomicExch(&marker[6], mode);
    (void)atomicExch(&marker[7], slots);
    (void)atomicExch(&marker[8], data_start_line);
    (void)atomicExch(&marker[9], per_write_dsb ? 1u : 0u);
    (void)atomicExch(&marker[10], 1u);
    dsb(DSB_ALL);

    uint32_t errors = 0;
    uint32_t slot_errors[2] = {0u, 0u};
    uint32_t first_actual = 0;
    uint32_t first_expected = 0;
    uint32_t first_trial = INVALID_U32;
    uint32_t first_slot = INVALID_U32;
    for (uint32_t trial = 0; trial < TRIALS; ++trial) {
        for (uint32_t round = 0; round < ROUNDS; ++round) {
            for (uint32_t slot = 0; slot < slots; ++slot) {
                uint32_t target = (data_start_line + slot) * CACHELINE_WORDS;
                st_dev_b32(&storage[target], TrialValue(mode, slot, trial, round));
                if (per_write_dsb) {
                    dsb(DSB_ALL);
                }
            }
        }
        if (!per_write_dsb) {
            dsb(DSB_ALL);
        }

        for (uint32_t slot = 0; slot < slots; ++slot) {
            uint32_t target = (data_start_line + slot) * CACHELINE_WORDS;
            uint32_t expected = TrialValue(mode, slot, trial, ROUNDS - 1u);
            uint32_t actual = ld_dev_b32(&storage[target]);
            if (actual != expected) {
                if (errors == 0u) {
                    first_actual = actual;
                    first_expected = expected;
                    first_trial = trial;
                    first_slot = slot;
                }
                ++errors;
                ++slot_errors[slot];
            }
        }
    }

    (void)atomicExch(&result[0], RESULT_MAGIC_VALUE);
    (void)atomicExch(&result[1], mode);
    (void)atomicExch(&result[2], TRIALS * slots);
    (void)atomicExch(&result[3], errors);
    (void)atomicExch(&result[4], slot_errors[0]);
    (void)atomicExch(&result[5], slot_errors[1]);
    (void)atomicExch(&result[6], first_actual);
    (void)atomicExch(&result[7], first_expected);
    (void)atomicExch(&result[8], first_trial);
    (void)atomicExch(&result[9], first_slot);
    (void)atomicExch(&result[10], ROUNDS);
    (void)atomicExch(&result[11], TRIALS);
    (void)atomicExch(&result[12], data_start_line);
    (void)atomicExch(&result[13], slots);
    (void)atomicExch(&result[14], per_write_dsb ? 1u : 0u);
    (void)atomicExch(&result[15], RESULT_FINISH_VALUE);
    dsb(DSB_ALL);
}
