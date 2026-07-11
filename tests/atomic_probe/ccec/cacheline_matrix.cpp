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
// CCEC equivalent of ascendc/cacheline_matrix.asc.
//
// The synchronization and scalar bypass helpers in ccec_utils.h mirror the
// local CANN 9.1 dav-3510 AscendC implementations. The authoritative runner
// builds AIV-only with CCEC_MATRIX_AIV_ONLY + CCEC_SYNC_AIV_ONLY. The source
// retains a supplementary MIX branch, but MIX is not part of this goal's gate.
#include "ccec_utils.h"

constexpr uint32_t CONTROL_WORD = 256;
constexpr uint32_t MARKER_WORD = CONTROL_WORD + 16;
constexpr uint32_t MATRIX_ROUNDS = 257;
constexpr uint32_t DONE_MAGIC = 0xCCEC3510u;

#ifdef CCEC_MATRIX_AIV_ONLY
CCEC_AIV_ONLY_KERNEL_META(cacheline_matrix);
#else
CCEC_MIX_1_1_KERNEL_META(cacheline_matrix);
#endif

template <typename T>
__aicore__ inline T MatrixValue(uint32_t participant, uint32_t round)
{
    uint64_t value = 0xA5A5A5A5A5A5A5A5ULL;
    value += (uint64_t)(participant + 1) * 0x0101010101010101ULL;
    value += round;
    return static_cast<T>(value);
}

__aicore__ inline uint32_t CoreKind()
{
#ifdef __DAV_VEC__
    return 1;
#else
    return 0;
#endif
}

__aicore__ inline uint32_t ParticipantId(uint32_t block_idx)
{
#ifdef CCEC_MATRIX_AIV_ONLY
    return block_idx;
#else
    return block_idx * 2 + CoreKind();
#endif
}

__aicore__ inline uint32_t ParticipantCount(uint32_t num_blocks)
{
#ifdef CCEC_MATRIX_AIV_ONLY
    return num_blocks;
#else
    return num_blocks * 2;
#endif
}

__aicore__ inline uint32_t MarkerValue(uint32_t participant)
{
    uint32_t core_tag = CoreKind() == 0 ? 0xC1C00000u : 0xC1A00000u;
    return core_tag | participant;
}

template <typename T>
__aicore__ inline void StoreBypass(__gm__ T *address, T value)
{
    __builtin_cce_st_dev(value, address, 0);
    // This oracle requires round N to precede round N+1. Serialize only this
    // probe's repeated same-address writes; this is not a general API rule.
    dsb(DSB_ALL);
}

template <typename T>
__aicore__ inline T LoadBypass(__gm__ T *address)
{
    return static_cast<T>(__builtin_cce_ld_dev(address, 0));
}

template <typename T>
__aicore__ inline void RunWidth(__gm__ uint8_t *storage, uint32_t mode, uint32_t participant,
                                uint32_t participants, __gm__ uint32_t *control)
{
    bool separate_lines = mode >= 4;
    uint32_t byte_offset = separate_lines ? participant * 64 : participant * sizeof(T);
    __gm__ T *slot = reinterpret_cast<__gm__ T *>(&storage[byte_offset]);
    for (uint32_t round = 0; round < MATRIX_ROUNDS; round++) {
        StoreBypass<T>(slot, MatrixValue<T>(participant, round));
    }

    ccec_sync_all();

    if (participant == 0) {
        uint32_t data_errors = 0;
        uint32_t marker_errors = 0;
        for (uint32_t p = 0; p < participants; p++) {
            uint32_t offset = separate_lines ? p * 64 : p * sizeof(T);
            __gm__ T *candidate = reinterpret_cast<__gm__ T *>(&storage[offset]);
            if (LoadBypass<T>(candidate) != MatrixValue<T>(p, MATRIX_ROUNDS - 1)) data_errors++;

            uint32_t marker = ld_dev_b32(&control[16 + p]);
#ifdef CCEC_MATRIX_AIV_ONLY
            uint32_t marker_kind = 1;
#else
            uint32_t marker_kind = p & 1u;
#endif
            uint32_t marker_expected = (marker_kind == 0 ? 0xC1C00000u : 0xC1A00000u) | p;
            if (marker != marker_expected) marker_errors++;
        }
        st_dev_b32(&control[4], data_errors);
        st_dev_b32(&control[5], participants);
        st_dev_b32(&control[6], DONE_MAGIC);
        st_dev_b32(&control[7], mode);
        st_dev_b32(&control[8], marker_errors);
    }

    ccec_sync_all();
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(cacheline_matrix)(
    __gm__ uint8_t *storage, uint32_t mode, uint32_t num_blocks)
{
    __gm__ uint32_t *words = reinterpret_cast<__gm__ uint32_t *>(storage);
    __gm__ uint32_t *control = &words[CONTROL_WORD];
    uint32_t participant = ParticipantId(get_block_idx());
    uint32_t participants = ParticipantCount(num_blocks);

    atomicAdd(&control[1], 1u);
    if (CoreKind() == 0) {
        atomicAdd(&control[2], 1u);
    } else {
        atomicAdd(&control[3], 1u);
    }
    st_dev_b32(&words[MARKER_WORD + participant], MarkerValue(participant));
    ccec_sync_all();

    switch (mode & 3u) {
        case 0:
            RunWidth<uint8_t>(storage, mode, participant, participants, control);
            break;
        case 1:
            RunWidth<uint16_t>(storage, mode, participant, participants, control);
            break;
        case 2:
            RunWidth<uint32_t>(storage, mode, participant, participants, control);
            break;
        default:
            RunWidth<uint64_t>(storage, mode, participant, participants, control);
            break;
    }
}
