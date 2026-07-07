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
#if !defined(__CCE_AICORE__) && !defined(SINGLE_CACHE_LINE)
#define SINGLE_CACHE_LINE 0
#endif
#if !defined(__CCE_AICORE__) && !defined(CACHELINE_OUT)
#define CACHELINE_OUT 0
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *view_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *output_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *dump_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    const uint64_t expected_offset = static_cast<uint64_t>(args[3]);
    const uint64_t expected_n = static_cast<uint64_t>(args[4]);

    __gm__ float *output =
        reinterpret_cast<__gm__ float *>(output_tensor->buffer.addr) + output_tensor->start_offset;
    __gm__ float *dump = reinterpret_cast<__gm__ float *>(dump_tensor->buffer.addr) + dump_tensor->start_offset;
    output[0] = view_tensor->buffer.addr == output_tensor->buffer.addr ? 1.0f : 0.0f;
    output[1] = view_tensor->start_offset == expected_offset ? 1.0f : 0.0f;
    output[2] = view_tensor->shapes[0] == expected_n ? 1.0f : 0.0f;
    output[3] = view_tensor->strides[0] == 1 ? 1.0f : 0.0f;
    dump[0] = view_tensor->start_offset == expected_offset ? 1.0f : 0.0f;
    dump[1] = view_tensor->shapes[0] == expected_n ? 1.0f : 0.0f;
    dump[2] = view_tensor->strides[0] == 1 ? 1.0f : 0.0f;
    dcci(output, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dcci(dump, SINGLE_CACHE_LINE, CACHELINE_OUT);
}
