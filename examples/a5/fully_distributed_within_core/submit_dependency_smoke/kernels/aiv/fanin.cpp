/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstdint>

#if __has_include("inner_kernel.h")
#include "inner_kernel.h"
#elif __has_include(<pto/pto-inst.hpp>)
#include <pto/pto-inst.hpp>
#endif
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#if !defined(__CCE_AICORE__) && !defined(dcci)
#define dcci(...) \
    do {          \
    } while (0)
#endif
#if !defined(__CCE_AICORE__) && !defined(ENTIRE_DATA_CACHE)
#define ENTIRE_DATA_CACHE 0
#endif
#if !defined(__CCE_AICORE__) && !defined(CACHELINE_OUT)
#define CACHELINE_OUT 0
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *seed_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *left_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *right_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *output_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    const uint64_t n = static_cast<uint64_t>(args[4]);

    __gm__ float *seed = reinterpret_cast<__gm__ float *>(seed_tensor->buffer.addr) + seed_tensor->start_offset;
    __gm__ float *left = reinterpret_cast<__gm__ float *>(left_tensor->buffer.addr) + left_tensor->start_offset;
    __gm__ float *right = reinterpret_cast<__gm__ float *>(right_tensor->buffer.addr) + right_tensor->start_offset;
    __gm__ float *output = reinterpret_cast<__gm__ float *>(output_tensor->buffer.addr) + output_tensor->start_offset;
    for (uint64_t i = 0; i < n; i++) {
        output[i] = seed[i] + left[i] + right[i] + 8.0f;
    }
    dcci(output, ENTIRE_DATA_CACHE, CACHELINE_OUT);
}
