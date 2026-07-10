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

#include "dist_engine/common/target.h"
#include "callable.h"
#include "inner_kernel.h"
#include "pto_runtime2.h"
#include "pto_types.h"
#include "runtime.h"

#include "dist_engine/aicore/cache_ops.h"  // NOLINT(build/include_subdir)

#include "dist_engine/common/state.h"          // NOLINT(build/include_subdir)
#include "dist_engine/common/atomic.h"         // NOLINT(build/include_subdir)
#include "dist_engine/common/trace.h"          // NOLINT(build/include_subdir)
#include "dist_engine/common/worker_state.h"   // NOLINT(build/include_subdir)
#include "dist_engine/common/runtime_state.h"  // NOLINT(build/include_subdir)
#include "dist_engine/common/debug_dump.h"     // NOLINT(build/include_subdir)
#include "dist_engine/common/sim_control.h"    // NOLINT(build/include_subdir)

#if defined(__CCE_AICORE__)
extern "C" PTO_DEVICE_FUNC __attribute__((weak)) void *memcpy(void *dst, const void *src, unsigned long n) {
    return aicore_memcpy(dst, src, n);
}

extern "C" PTO_DEVICE_FUNC void aicpu_orchestration_entry(const L2TaskArgs &orch_args) __attribute__((weak));
extern "C" PTO_DEVICE_FUNC int32_t pto_call_linked_kernel_aic(int32_t func_id, __gm__ int64_t *args)
    __attribute__((weak));
extern "C" PTO_DEVICE_FUNC int32_t pto_call_linked_kernel_aiv(int32_t func_id, __gm__ int64_t *args)
    __attribute__((weak));

#elif defined(__CPU_SIM)
extern "C" PTO_DEVICE_FUNC void aicpu_orchestration_entry(const L2TaskArgs &orch_args);
#endif

#ifndef SPIN_WAIT_HINT
#define SPIN_WAIT_HINT() ((void)0)
#endif

#include "dist_engine/aicore/primitive.h"           // NOLINT(build/include_subdir)
#include "dist_engine/aicore/api_glue.h"            // NOLINT(build/include_subdir)
#include "dist_engine/aicore/log.h"                 // NOLINT(build/include_subdir)
#include "dist_engine/aicore/tensor_map.h"          // NOLINT(build/include_subdir)
#include "dist_engine/aicore/core_state.h"          // NOLINT(build/include_subdir)
#include "dist_engine/aicore/submit_core.h"         // NOLINT(build/include_subdir)
#include "dist_engine/aicore/submit_helpers.h"      // NOLINT(build/include_subdir)
#include "dist_engine/aicore/tensor_data_access.h"  // NOLINT(build/include_subdir)
#include "dist_engine/aicore/backend.h"             // NOLINT(build/include_subdir)
#include "dist_engine/aicore/submit_direct.h"       // NOLINT(build/include_subdir)
#include "dist_engine/aicore/core_main.h"           // NOLINT(build/include_subdir)
