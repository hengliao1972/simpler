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
// Minimal two-AIV control for repeated st_dev writes:
//   1. different 4B slots in the same 64B cache line, one DSB after the loop;
//   2. one cache line per AIV, one DSB after the loop;
//   3. different slots in one line, one DSB after every round.
//
// All three cases use the same exact final-value oracle. Case 1 is the
// regression path expected to expose the current cross-AIV same-line issue;
// cases 2 and 3 isolate line sharing and store completion as controls.
// Keep case 1 at loop-end DSB: adding per-round DSB there would mask the
// behavior this regression test is intended to expose.
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(st_dev_same_line);

constexpr uint32_t ROUNDS = 257;
constexpr uint32_t TRIALS = 100;
constexpr uint32_t SAME_BASE = 0;
constexpr uint32_t SEPARATE_BASE = 16;
constexpr uint32_t ORDERED_BASE = 48;
constexpr uint32_t RESULT_BASE = 80;

__aicore__ inline uint32_t TrialValue(uint32_t participant, uint32_t trial, uint32_t round)
{
    return 0xA5000000u + participant * 0x01000000u + trial * 0x1000u + round;
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(st_dev_same_line)(__gm__ uint32_t *gx)
{
    uint32_t participant = (uint32_t)get_block_idx();
    uint32_t same_errors = 0;
    uint32_t separate_errors = 0;
    uint32_t ordered_errors = 0;
    uint32_t first_same_actual = 0;
    uint32_t first_same_expected = 0;

    atomicAdd(&gx[RESULT_BASE + 7], 1u);
    st_dev_b32(&gx[RESULT_BASE + 8 + participant], 0xA1A00000u | participant);
    dsb(DSB_ALL);
    ccec_sync_all();

    for (uint32_t trial = 0; trial < TRIALS; trial++) {
        for (uint32_t round = 0; round < ROUNDS; round++) {
            st_dev_b32(&gx[SAME_BASE + participant], TrialValue(participant, trial, round));
        }
        dsb(DSB_ALL);
        ccec_sync_all();
        if (participant == 0) {
            for (uint32_t p = 0; p < 2; p++) {
                uint32_t expected = TrialValue(p, trial, ROUNDS - 1);
                uint32_t actual = ld_dev_b32(&gx[SAME_BASE + p]);
                if (actual != expected) {
                    if (same_errors == 0) {
                        first_same_actual = actual;
                        first_same_expected = expected;
                    }
                    same_errors++;
                }
            }
        }
        ccec_sync_all();

        uint32_t separate_slot = SEPARATE_BASE + participant * 16;
        for (uint32_t round = 0; round < ROUNDS; round++) {
            st_dev_b32(&gx[separate_slot], TrialValue(participant, trial, round));
        }
        dsb(DSB_ALL);
        ccec_sync_all();
        if (participant == 0) {
            for (uint32_t p = 0; p < 2; p++) {
                uint32_t expected = TrialValue(p, trial, ROUNDS - 1);
                uint32_t actual = ld_dev_b32(&gx[SEPARATE_BASE + p * 16]);
                if (actual != expected) separate_errors++;
            }
        }
        ccec_sync_all();

        for (uint32_t round = 0; round < ROUNDS; round++) {
            st_dev_b32(&gx[ORDERED_BASE + participant], TrialValue(participant, trial, round));
            dsb(DSB_ALL);
        }
        ccec_sync_all();
        if (participant == 0) {
            for (uint32_t p = 0; p < 2; p++) {
                uint32_t expected = TrialValue(p, trial, ROUNDS - 1);
                uint32_t actual = ld_dev_b32(&gx[ORDERED_BASE + p]);
                if (actual != expected) ordered_errors++;
            }
        }
        ccec_sync_all();
    }

    if (participant == 0) {
        st_dev_b32(&gx[RESULT_BASE + 0], same_errors);
        st_dev_b32(&gx[RESULT_BASE + 1], separate_errors);
        st_dev_b32(&gx[RESULT_BASE + 2], ordered_errors);
        st_dev_b32(&gx[RESULT_BASE + 3], first_same_actual);
        st_dev_b32(&gx[RESULT_BASE + 4], first_same_expected);
        st_dev_b32(&gx[RESULT_BASE + 5], TRIALS);
        st_dev_b32(&gx[RESULT_BASE + 6], 0x3510C1A5u);
        dsb(DSB_ALL);
    }
    ccec_sync_all();
}
