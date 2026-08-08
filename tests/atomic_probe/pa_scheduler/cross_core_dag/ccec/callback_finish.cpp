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

#include "cce_aicore_intrinsics.h"
#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

#define PA_DEVICE __aicore__ inline
#define PA_DEVICE_NOINLINE static __aicore__ __attribute__((noinline))
#define PA_LOOP_NOUNROLL _Pragma("clang loop unroll(disable)")
#define PA_GM __gm__
#include "../common/pa_scheduler_core.h"
#include "ccec_ops.h"

using pa_scheduler_ccec::CcecOps;

// finish 保持唯一的 cross-TU noinline 边界。入口只转交固定 POD ticket 与
// TaskArgs；协议校验、Materialize/Register/Fanin/Build 均在公共实现中完成。
extern "C" {
#if defined(PA_BUILD_AIC)
__attribute__((noinline)) __aicore__ uint32_t
pa_scheduler_compete_first_callback_finish_aic(
    const pa_scheduler::CallbackSubmitTicket *ticket, const pa_scheduler::TaskArgs *args
) {
    return pa_scheduler::FinishSplitCallbackSubmitFromRuntime<CcecOps>(ticket, args);
}
#elif defined(PA_BUILD_AIV)
__attribute__((noinline)) __aicore__ uint32_t
pa_scheduler_compete_first_callback_finish_aiv(
    const pa_scheduler::CallbackSubmitTicket *ticket, const pa_scheduler::TaskArgs *args
) {
    return pa_scheduler::FinishSplitCallbackSubmitFromRuntime<CcecOps>(ticket, args);
}
#else
#error "Compile split finish with PA_BUILD_AIC or PA_BUILD_AIV"
#endif
}
