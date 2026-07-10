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
 * Producer kernel: writes a round-dependent pattern into buf[i].
 *   buf[i] = round * 100 + i  (for i in [0, ELEMS))
 *
 * Args: [buf_tensor, round_scalar]
 */

#include <cstdint>

#if __has_include("inner_kernel.h")
#include "inner_kernel.h"
#elif __has_include(<pto/pto-inst.hpp>)
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

static constexpr int32_t PRODUCER_ELEMS = 16;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *buf_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    int64_t round = args[1];

    __gm__ float *buf = reinterpret_cast<__gm__ float *>(buf_tensor->buffer.addr) + buf_tensor->start_offset;

    for (int32_t i = 0; i < PRODUCER_ELEMS; i++) {
        buf[i] = static_cast<float>(round * 100 + i);
    }

    pipe_sync();
}
