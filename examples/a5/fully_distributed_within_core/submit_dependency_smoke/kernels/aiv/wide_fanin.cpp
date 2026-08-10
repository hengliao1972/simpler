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
#include "arg_direction.h"
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
    constexpr uint32_t input_count = MAX_TENSOR_ARGS - 1U;
    __gm__ Tensor *output_tensor = reinterpret_cast<__gm__ Tensor *>(args[input_count]);
    __gm__ float *output = reinterpret_cast<__gm__ float *>(output_tensor->buffer.addr) + output_tensor->start_offset;

    // 所有 Tensor 和 scalar 都被真实消费，防止编译器把上限用例退化。
    for (uint32_t input = 0; input < input_count; ++input) {
        __gm__ Tensor *input_tensor = reinterpret_cast<__gm__ Tensor *>(args[input]);
        __gm__ float *source = reinterpret_cast<__gm__ float *>(input_tensor->buffer.addr) + input_tensor->start_offset;
        output[input] = source[0];
    }
    uint64_t scalar_sum = 0;
    for (uint32_t scalar = 0; scalar < MAX_SCALAR_ARGS; ++scalar) {
        scalar_sum += static_cast<uint64_t>(args[MAX_TENSOR_ARGS + scalar]);
    }
    output[input_count] = static_cast<float>(scalar_sum);

    for (uint32_t offset = 0; offset < MAX_TENSOR_ARGS * sizeof(float); offset += 64U) {
        dcci(reinterpret_cast<__gm__ uint8_t *>(output) + offset, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
    pipe_sync();
}
