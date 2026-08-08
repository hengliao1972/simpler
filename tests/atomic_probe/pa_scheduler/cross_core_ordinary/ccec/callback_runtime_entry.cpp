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
#include <pto/common/kernel_meta.hpp>

#define PA_DEVICE __aicore__ inline
#define PA_DEVICE_NOINLINE static __aicore__ __attribute__((noinline))
#define PA_LOOP_NOUNROLL _Pragma("clang loop unroll(disable)")
#define PA_GM __gm__
#include "../common/pa_scheduler_core.h"

extern "C" {
#if defined(PA_BUILD_AIC)
// AIC 与 AIV 分别定义自己的 block-local 状态和 launch entry；不能退化成
// 共用弱符号，否则 mixed ELF 中两个核型会误用同一套 Submit 上下文。
[[block_local]] pa_scheduler::CompeteFirstSplitRuntimeState pa_scheduler_compete_first_callback_state_aic;
__aicore__ void
pa_scheduler_compete_first_callback_orchestration_aic(
    __gm__ pa_scheduler::SchedulerState *state, uint32_t worker_id
);
#elif defined(PA_BUILD_AIV)
[[block_local]] pa_scheduler::CompeteFirstSplitRuntimeState pa_scheduler_compete_first_callback_state_aiv;
__aicore__ void
pa_scheduler_compete_first_callback_orchestration_aiv(
    __gm__ pa_scheduler::SchedulerState *state, uint32_t worker_id
);
#else
#error "Compile split runtime entry with PA_BUILD_AIC or PA_BUILD_AIV"
#endif
}

#if defined(PA_BUILD_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(pa_scheduler_0_mix_aic, 1, 2);

extern "C" __global__ __aicore__ void pa_scheduler_0_mix_aic(__gm__ pa_scheduler::SchedulerState *state) {
    // runtime entry 只负责物理核到 worker 的映射；调度主循环位于 caller TU。
    const uint32_t worker_id = static_cast<uint32_t>(get_block_idx());
    pa_scheduler_compete_first_callback_orchestration_aic(state, worker_id);
}
#elif defined(PA_BUILD_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(pa_scheduler_0_mix_aiv, 1, 2);

extern "C" __global__ __aicore__ void pa_scheduler_0_mix_aiv(__gm__ pa_scheduler::SchedulerState *state) {
    // 两个 vector sub-block 展平后继续使用 standalone 的连续 worker 编号。
    const uint32_t vector_id = static_cast<uint32_t>(get_block_idx() * get_subblockdim() + get_subblockid());
    const uint32_t worker_id = pa_scheduler::kAicWorkers + vector_id;
    pa_scheduler_compete_first_callback_orchestration_aiv(state, worker_id);
}
#endif
