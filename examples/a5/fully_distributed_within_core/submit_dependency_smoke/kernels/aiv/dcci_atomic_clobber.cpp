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
#include "tensor.h"
#include "pipe_sync.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

namespace {

constexpr float kOne = 1.0f;
constexpr float kThree = 3.0f;

PTO_DEVICE_FUNC inline float atomic_load_f32(__gm__ volatile float *addr) {
#if defined(__CCE_AICORE__)
    return atomicAdd(const_cast<__gm__ float *>(addr), 0.0f);
#else
    return *addr;
#endif
}

PTO_DEVICE_FUNC inline void atomic_store_f32(__gm__ volatile float *addr, float value) {
#if defined(__CCE_AICORE__)
    atomicExch(const_cast<__gm__ float *>(addr), value);
#else
    *addr = value;
#endif
}

}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *output_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ float *output = reinterpret_cast<__gm__ float *>(output_tensor->buffer.addr) + output_tensor->start_offset;
    __gm__ volatile float *words = output;
    __gm__ volatile float *phase = output + 16;

    while (atomic_load_f32(&phase[0]) != kOne) {}
    pipe_sync();
    atomic_store_f32(&words[0], kThree);
    pipe_sync();
    atomic_store_f32(&phase[1], kOne);
    pipe_sync();
}
