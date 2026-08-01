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
// Multi-block QK Matmul Kernel: qi(M, K) @ kj.T(K, N) -> sij(M, N) for each block
//
// Processes n_blocks blocks in a single kernel invocation.
// Per-block kj addresses computed from key_cache base + block_indices lookup.
// qi is shared across all blocks (same query head against different key blocks).
//
// Output layout: n_blocks contiguous (M, N) tiles stacked vertically.
// Block i occupies sij[i*M : (i+1)*M, 0:N].
//
// Optimizations:
//   - qi is staged into L0A once (TMATMUL only reads its operands)
//   - L1/L0B/L0C use precise ping-pong dependencies so adjacent blocks can
//     overlap MTE2, MTE1, M and FIX without a loop-wide PIPE_ALL barrier
//
// Supports two tile configurations via runtime dispatch:
//   Case1: (16, 128) @ (128, 128).T -> (16, 128)
//   Case2: (64, 128) @ (128,  64).T -> (64,  64)
//
// Template: M=q_tile, K=head_dim, N=block_size

#include <cstdint>
// NOLINTBEGIN(clang-diagnostic-error,bugprone-reserved-identifier,bugprone-easily-swappable-parameters,modernize-use-auto)
#include <pto/pto-inst.hpp>

#include "tensor.h"

// NOLINTNEXTLINE(build/namespaces)
using namespace pto;

#include "pipe_sync.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

template <int M, int K, int N>
static __aicore__ void qk_matmul_n_impl(
    __gm__ bfloat16_t *qi_base, __gm__ bfloat16_t *key_base, __gm__ float *sij_base, uint64_t n_blocks,
    __gm__ int32_t *bt, uint64_t bt_offset
) {
    using GlobalA = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<bfloat16_t, Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, 1, K>, Layout::DN>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, bfloat16_t, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;

    using LeftTile = TileLeft<bfloat16_t, M, K, M, K>;
    using RightTile = TileRight<bfloat16_t, K, N, K, N>;
    using AccTile = TileAcc<float, M, N, M, N>;

    // Double-buffer the changing B operand and each independent result.  The
    // constant A operand is copied into L0A only once before the loop.
    constexpr int kBBytes = K * N * static_cast<int>(sizeof(bfloat16_t));
    constexpr int kCBytes = M * N * static_cast<int>(sizeof(float));
    TileMatA aMatTile;
    TileMatB bMatTile[2];
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile[0], 0x20000);
    TASSIGN(bMatTile[1], 0x20000 + kBBytes);

    LeftTile aTile;
    RightTile bTile[2];
    AccTile cTile[2];
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile[0], 0x0);
    TASSIGN(bTile[1], kBBytes);
    TASSIGN(cTile[0], 0x0);
    TASSIGN(cTile[1], kCBytes);

    // Stage the immutable query through L1 into L0A once.  TMATMUL declares
    // both operands as inputs, so later blocks may safely reuse the same L0A
    // tile without copying it again.
    GlobalA qiGlobal(qi_base);
    TLOAD(aMatTile, qiGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    TMOV(aTile, aMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);

    // Prime the three reverse dependencies.  Each slot is initially free:
    //   MTE1 -> MTE2 releases an L1 B slot after TMOV;
    //   M    -> MTE1 releases an L0B slot after TMATMUL;
    //   FIX  -> M releases an L0C slot after TSTORE.
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

    for (uint64_t i = 0; i < n_blocks; i++) {
        const int cur = static_cast<int>(i % 2);
        const ::event_t event = static_cast<::event_t>(cur);
        GlobalB kjGlobal(key_base + bt[bt_offset + i] * N * K);
        GlobalOut sijGlobal(sij_base + i * M * N);

        // Stage 1: GM -> L1[cur].  Do not overwrite this slot until its
        // previous L1 -> L0B transfer has completed.
        wait_flag(PIPE_MTE1, PIPE_MTE2, event);
        TLOAD(bMatTile[cur], kjGlobal);
        set_flag(PIPE_MTE2, PIPE_MTE1, event);

        // Stage 2: L1[cur] -> L0B[cur].  The forward event waits for this
        // block's load; the reverse event protects the previous user.
        wait_flag(PIPE_M, PIPE_MTE1, event);
        wait_flag(PIPE_MTE2, PIPE_MTE1, event);
        TMOV(bTile[cur], bMatTile[cur]);
        set_flag(PIPE_MTE1, PIPE_MTE2, event);
        set_flag(PIPE_MTE1, PIPE_M, event);

        // Stage 3: L0A x L0B[cur] -> L0C[cur].  FIX releases the accumulator
        // slot only after the prior TSTORE has consumed it.
        wait_flag(PIPE_FIX, PIPE_M, event);
        wait_flag(PIPE_MTE1, PIPE_M, event);
        TMATMUL(cTile[cur], aTile, bTile[cur]);
        set_flag(PIPE_M, PIPE_MTE1, event);
        set_flag(PIPE_M, PIPE_FIX, event);

        // Stage 4: L0C[cur] -> GM.  The reverse event lets M reuse this slot
        // two iterations later without draining unrelated pipelines.
        wait_flag(PIPE_M, PIPE_FIX, event);
        TSTORE(sijGlobal, cTile[cur]);
        set_flag(PIPE_FIX, PIPE_M, event);
    }

    // Consume the final release token for every ping-pong slot.  This also
    // drains the last in-flight iteration without leaking event state into
    // the next linked kernel invocation.
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    pipe_sync();
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *qi = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *key_cache = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *block_table_t = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *sij_buf = reinterpret_cast<__gm__ Tensor *>(args[3]);
    uint64_t n_blocks = static_cast<uint64_t>(args[4]);
    uint64_t bt_offset = static_cast<uint64_t>(args[5]);

    __gm__ bfloat16_t *qi_base = reinterpret_cast<__gm__ bfloat16_t *>(qi->buffer.addr) + qi->start_offset;
    __gm__ bfloat16_t *key_base = reinterpret_cast<__gm__ bfloat16_t *>(key_cache->buffer.addr);
    __gm__ float *sij_base = reinterpret_cast<__gm__ float *>(sij_buf->buffer.addr) + sij_buf->start_offset;
    __gm__ int32_t *bt = reinterpret_cast<__gm__ int32_t *>(block_table_t->buffer.addr);

    uint64_t q_tile_size = static_cast<uint64_t>(qi->shapes[0]);

    if (q_tile_size == 16) {
        qk_matmul_n_impl<16, 128, 128>(qi_base, key_base, sij_base, n_blocks, bt, bt_offset);
    } else {
        qk_matmul_n_impl<64, 128, 64>(qi_base, key_base, sij_base, n_blocks, bt, bt_offset);
    }
}

// NOLINTEND(clang-diagnostic-error,bugprone-reserved-identifier,bugprone-easily-swappable-parameters,modernize-use-auto)
