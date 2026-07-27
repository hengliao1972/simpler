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

#define PA_DEVICE __aicore__ inline
#define PA_DEVICE_NOINLINE static __aicore__ __attribute__((noinline))
#define PA_LOOP_NOUNROLL _Pragma("clang loop unroll(disable)")
#define PA_GM __gm__
#include "../common/pa_scheduler_core.h"
#include "ccec_ops.h"

// 该 TU 只做设备编译器显式实例化，不参与最终 mixed ELF 链接。它保证
// 通用 writer-intent 使用的 CAS、DCCI 和 GM 地址空间签名同时能被 AIC
// 与 AIV 的真实 CcecOps 接受，避免普通 kernel 尚未接线时只解析模板定义。
template pa_scheduler::SharedWriterIntentResult
pa_scheduler::PrepareSharedWriterIntentSet<
    pa_scheduler_ccec::CcecOps>(
    __gm__ pa_scheduler::SchedulerState *,
    const pa_scheduler::TaskArgs &,
    pa_scheduler::SubmitContext &,
    pa_scheduler::LocalStats &
);
