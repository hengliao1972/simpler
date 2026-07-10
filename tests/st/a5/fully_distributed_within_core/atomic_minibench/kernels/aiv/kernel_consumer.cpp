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
/**
 * Consumer kernel: reads buf (producer output) and writes result.
 *   result[i] = buf[i] + 1
 *
 * If dcci seam is broken, buf contains stale data and result will be wrong.
 *
 * Args: [buf_tensor, result_tensor, round_scalar]
 */

#include <cstdint>

#if __has_include("inner_kernel.h")
#include "inner_kernel.h"
#endif
#if __has_include(<pto/pto-inst.hpp>)
#include <pto/pto-inst.hpp>
#endif
#include "tensor.h"

#if __has_include("pipe_sync.h")
#include "pipe_sync.h"
#endif

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static constexpr int32_t CONSUMER_ELEMS = 16;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *buf_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *result_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);

    __gm__ float *buf = reinterpret_cast<__gm__ float *>(buf_tensor->buffer.addr) + buf_tensor->start_offset;
    __gm__ float *result = reinterpret_cast<__gm__ float *>(result_tensor->buffer.addr) + result_tensor->start_offset;

    for (int32_t i = 0; i < CONSUMER_ELEMS; i++) {
        result[i] = buf[i] + 1.0f;
    }

#if defined(__CCE_AICORE__)
    dcci(reinterpret_cast<__gm__ uint8_t *>(result), SINGLE_CACHE_LINE, CACHELINE_OUT);
#endif
    pipe_sync();
}
