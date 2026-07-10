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

PTO_DEVICE_FUNC inline uint64_t dist_aicore_slot_function_addr(Runtime *runtime, int32_t kernel_id) {
#if defined(__CCE_AICORE__)
    (void)runtime;
    (void)kernel_id;
    return 0;
#else
    return resolve_kernel_addr(runtime, kernel_id);
#endif
}

PTO_DEVICE_FUNC inline void dist_aicore_call_slot_kernel(__gm__ RingSlot &slot) {
#if defined(__CCE_AICORE__)
    const bool is_aic = g_ccec_core_type == static_cast<int32_t>(CoreType::AIC);
    if (is_aic) {
        if (pto_call_linked_kernel_aic != nullptr) {
            pto_call_linked_kernel_aic(slot.func_id, reinterpret_cast<__gm__ int64_t *>(slot.args));
        }
    } else if (pto_call_linked_kernel_aiv != nullptr) {
        pto_call_linked_kernel_aiv(slot.func_id, reinterpret_cast<__gm__ int64_t *>(slot.args));
    }
#else
    typedef void (*KernelFn)(__gm__ int64_t *);
    if (slot.function_bin_addr != 0) {
        KernelFn fn = reinterpret_cast<KernelFn>(slot.function_bin_addr);
        fn(reinterpret_cast<__gm__ int64_t *>(slot.args));
    }
#endif
}
