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

#include <cstdint>

#if __has_include("inner_kernel.h")
#include "inner_kernel.h"
#elif __has_include(<pto/pto-inst.hpp>)
#include <pto/pto-inst.hpp>
#endif
#include "pipe_sync.h"
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
#if !defined(__CCE_AICORE__) && !defined(SINGLE_CACHE_LINE)
#define SINGLE_CACHE_LINE 0
#endif
#if !defined(__CCE_AICORE__) && !defined(CACHELINE_OUT)
#define CACHELINE_OUT 0
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *input_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *output_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    const uint64_t n = static_cast<uint64_t>(args[2]);
    const uint64_t spin = static_cast<uint64_t>(args[3]);

    // 延迟只用于让缺失 fanin 稳定暴露；数据区域仍按正常 kernel 写回。
    volatile uint64_t sink = 0;
    for (uint64_t iteration = 0; iteration < spin; ++iteration) {
        sink += iteration;
    }

    __gm__ float *input = reinterpret_cast<__gm__ float *>(input_tensor->buffer.addr) + input_tensor->start_offset;
    __gm__ float *output = reinterpret_cast<__gm__ float *>(output_tensor->buffer.addr) + output_tensor->start_offset;
    for (uint64_t index = 0; index < n; ++index) {
        output[index] = input[index] + 7.0f;
    }
    if (sink == UINT64_MAX) output[0] = input[0] + 7.0f;

    for (uint64_t offset = 0; offset < n * sizeof(float); offset += 64U) {
        dcci(reinterpret_cast<__gm__ uint8_t *>(output) + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
    pipe_sync();
}
