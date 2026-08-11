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

// 最小失败复现：严格只替换任务要求中的两个公共 device 宏。无需实例化
// entry，CCEC 已会在 BuildCallbackSubmitArgs 模板定义内发现 scalar-only
// callback lambda 调用 simt_callee helper，证明直接文本复用在前端即失败。

#include "cce_aicore_intrinsics.h"
#include "simt_api/asc_simt.h"

#define PA_DEVICE \
    __simt_callee__ __aicore__ __attribute__((always_inline)) inline
#define PA_DEVICE_NOINLINE \
    __simt_callee__ __aicore__ __attribute__((always_inline)) inline
#define PA_LOOP_NOUNROLL _Pragma("clang loop unroll(disable)")
#define PA_GM __gm__

#include "../../scalar_build/common/pa_scheduler_core.h"
