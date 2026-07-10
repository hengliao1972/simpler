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

#pragma once

#include "dist_engine/common/runtime_state.h"

#if defined(__CPU_SIM)
#include <cstdio>

void dist_dump_state(int) {
    fprintf(stderr, "\n===== DIST STATE DUMP =====\n");
    fprintf(
        stderr, "frontier=%ld H=%d ring=%zuB replay_done=%ld/%d num_blocks=%d fatal=%d\n",
        static_cast<long>(atom_load(g_dist.frontier, __ATOMIC_RELAXED)), g_dist.H, g_dist.heap_size,
        static_cast<long>(atom_load(g_dist.replay_done, __ATOMIC_RELAXED)), g_dist.num_workers, g_dist.num_blocks,
        atom_load(g_dist.fatal, __ATOMIC_RELAXED)
    );
    fprintf(stderr, "cube_cursor[%d]=", kCursorShards);
    for (int32_t s = 0; s < kCursorShards; s++)
        fprintf(
            stderr, "%ld%s", static_cast<long>(atom_load(g_dist.cube_cursor[s].v, __ATOMIC_RELAXED)),
            s + 1 < kCursorShards ? "," : ""
        );
    fprintf(stderr, " vector_cursor[%d]=", kCursorShards);
    for (int32_t s = 0; s < kCursorShards; s++)
        fprintf(
            stderr, "%ld%s", static_cast<long>(atom_load(g_dist.vector_cursor[s].v, __ATOMIC_RELAXED)),
            s + 1 < kCursorShards ? "," : ""
        );
    fprintf(stderr, "\n");
    for (int32_t c = 0; c < g_dist.num_workers && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = g_dist.cores[c];
        fprintf(
            stderr, "core %d role=%d blk=%d lane=%d replayed=%d occ=%d owned=%d\n", c, static_cast<int>(co.role),
            co.block_id, co.lane, co.local_index, co.occupied_count, co.owned_total
        );
        for (int32_t i = 0; i < kPrivateSlots; i++) {
            RingSlot &s = co.slots[i];
            if (!s.occupied) continue;
            int32_t unmet = -1;
            for (int32_t f = 0; f < s.fanin_count; f++)
                if (s.fanin[f] < 0 || s.fanin[f] >= kFlagCap ||
                    atom_load(task_cell(s.fanin[f]).flag, __ATOMIC_RELAXED) == 0) {
                    unmet = s.fanin[f];
                    break;
                }
            fprintf(
                stderr, "    slot%d tid=%d built=%d mc=%d won=(%d,%d) fanin=%d unmet=%d\n", i, s.task_id, s.built,
                s.is_multicore, s.won_block, s.won_slot, s.fanin_count, unmet
            );
        }
    }
    for (int32_t b = 0; b < g_dist.num_blocks; b++) {
        for (int32_t i = 0; i < kPrivateSlots; i++) {
            WonSlot &w = g_dist.blocks[b].slots[i];
            int32_t st = atom_load(w.state, __ATOMIC_RELAXED);
            if (st == 0) continue;
            fprintf(
                stderr, "  won blk%d slot%d state=%d tid=%d remaining=%ld drained=[%d,%d,%d] present=[%d,%d,%d]\n", b,
                i, st, w.task_id, static_cast<long>(atom_load(w.remaining, __ATOMIC_RELAXED)),
                atom_load(w.drained[0].v, __ATOMIC_RELAXED), atom_load(w.drained[1].v, __ATOMIC_RELAXED),
                atom_load(w.drained[2].v, __ATOMIC_RELAXED), w.lane[0].present, w.lane[1].present, w.lane[2].present
            );
        }
    }
    fprintf(stderr, "===== END DUMP =====\n");
}
#endif
