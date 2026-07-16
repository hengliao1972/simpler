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
 * Write a scalar index value into out[0]. Each task writes its own slot,
 * so if the completion flag is lost (cacheline clobber), the slot stays
 * at its initial value (0) and golden comparison fails.
 *
 * Args: [out_tensor, index_scalar]
 *   args[0] = __gm__ Tensor* out  (1 element)
 *   args[1] = int64_t index       (cast to float, written to out[0])
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

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    int64_t index = args[1];

    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;
    out[0] = static_cast<float>(index) + 1.0f;

#if defined(__CCE_AICORE__)
    dcci(reinterpret_cast<__gm__ uint8_t *>(out), SINGLE_CACHE_LINE, CACHELINE_OUT);
#endif
    pipe_sync();
}
