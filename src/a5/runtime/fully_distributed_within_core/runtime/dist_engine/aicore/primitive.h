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

#include "dist_engine/common/target.h"

// AICore-target primitives shared by the direct submit backend. Onboard CCEC
// needs explicit GM cache maintenance; sim AICore shares process memory and only
// needs compiler/CPU fences to preserve ordering.

PTO_DEVICE_FUNC inline void dist_aicore_invalidate_region(__gm__ const void *ptr, uint64_t bytes) {
#if defined(__CCE_AICORE__)
    if (bytes == 0) return;
    const uint64_t start = reinterpret_cast<uint64_t>(ptr) & ~uint64_t{63};
    const uint64_t end = (reinterpret_cast<uint64_t>(ptr) + bytes + 63) & ~uint64_t{63};
    for (uint64_t addr = start; addr < end; addr += 64) {
        dcci(reinterpret_cast<__gm__ uint8_t *>(addr), SINGLE_CACHE_LINE);
    }
    dsb((mem_dsb_t)0);
#else
    (void)ptr;
    (void)bytes;
#endif
}

PTO_DEVICE_FUNC inline void dist_aicore_flush_region(__gm__ void *ptr, uint64_t bytes) {
#if defined(__CCE_AICORE__)
    if (bytes == 0) return;
    __asm__ volatile("" ::: "memory");
    const uint64_t start = reinterpret_cast<uint64_t>(ptr) & ~uint64_t{63};
    const uint64_t end = (reinterpret_cast<uint64_t>(ptr) + bytes + 63) & ~uint64_t{63};
    for (uint64_t addr = start; addr < end; addr += 64) {
        dcci(reinterpret_cast<__gm__ uint8_t *>(addr), SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
    dsb((mem_dsb_t)0);
#else
    (void)ptr;
    (void)bytes;
    atom_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

PTO_DEVICE_FUNC inline void dist_aicore_store_barrier() {
#if defined(__CCE_AICORE__)
    OUT_OF_ORDER_STORE_BARRIER();
#else
    atom_thread_fence(__ATOMIC_RELEASE);
#endif
}

PTO_DEVICE_FUNC inline void dist_aicore_publish_done() {
    OUT_OF_ORDER_STORE_BARRIER();
    write_reg(RegId::COND, MAKE_FIN_VALUE(0));
}
