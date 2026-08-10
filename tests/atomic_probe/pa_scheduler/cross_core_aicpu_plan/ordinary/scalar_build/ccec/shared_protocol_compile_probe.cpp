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

// 该 TU 只做 shared 通用协议的设备编译器显式实例化，不参与最终 mixed
// ELF。它同时锁定 writer-intent 与 reader-progress/reclaim 使用的 CAS、
// DCCI、GM 地址空间和引用签名，避免普通 PA kernel 尚未接线时只解析模板
// 定义、却从未生成真实 AIC/AIV 代码。
template pa_scheduler::SharedWriterIntentResult
pa_scheduler::PrepareSharedWriterIntentSet<
    pa_scheduler_ccec::CcecOps>(
    __gm__ pa_scheduler::SchedulerState *,
    const pa_scheduler::TaskArgs &,
    pa_scheduler::SubmitContext &,
    pa_scheduler::LocalStats &
);

template bool pa_scheduler::SharedAdvanceReaderDone<
    pa_scheduler_ccec::CcecOps>(
    __gm__ pa_scheduler::SharedTensorMapSidecar &,
    uint32_t, int32_t
);

template bool pa_scheduler::SharedRefreshReaderReclaimForTask<
    pa_scheduler_ccec::CcecOps>(
    __gm__ pa_scheduler::SharedTensorMapSidecar &,
    int32_t, uint32_t, int32_t, int64_t &
);

template pa_scheduler::SharedAppendCheck
pa_scheduler::SharedTryAppendReaderGatedTask<
    pa_scheduler_ccec::CcecOps>(
    __gm__ pa_scheduler::SharedTensorMapSidecar &,
    const pa_scheduler::SharedRegionValue *, uint32_t,
    uint32_t, int32_t
);

// AICPU Plan consumer 尚未接入正式 kernel 前，先在这个不会进入最终
// mixed ELF 的 TU 中强制生成全部 Scalar 侧协议代码。这里不是只检查头
// 文件能否解析：AIC/AIV 都必须真实实例化返回型 atomic observe、中央
// FetchAdd ticket、payload invalidate/acquire、worker arrival 和 release
// Exchange，才能提前暴露 GM 地址空间或 intrinsic 签名错误。
template pa_scheduler::aicpu_plan::BuildReservation
pa_scheduler::aicpu_plan::TakeClosedPlanBuildTicket<
    pa_scheduler_ccec::CcecOps>(
    const pa_scheduler::aicpu_plan::RuntimePlanView &
);

template pa_scheduler::aicpu_plan::BuildArrivalStatus
pa_scheduler::aicpu_plan::ArriveClosedPlanBuildWorker<
    pa_scheduler_ccec::CcecOps>(
    const pa_scheduler::aicpu_plan::RuntimePlanView &, uint32_t
);

template bool
pa_scheduler::aicpu_plan::PublishClosedPlanBuildRelease<
    pa_scheduler_ccec::CcecOps>(
    const pa_scheduler::aicpu_plan::RuntimePlanView &, uint32_t, uint32_t
);

template pa_scheduler::aicpu_plan::PlanAcquireResult
pa_scheduler::aicpu_plan::AcquireRuntimeTaskPlan<
    pa_scheduler_ccec::CcecOps>(
    const pa_scheduler::aicpu_plan::RuntimePlanView &, uint32_t,
    pa_scheduler::aicpu_plan::RuntimeTaskPlanHeader &,
    pa_scheduler::aicpu_plan::RuntimeTaskPlanLayout &
);

// MakeRuntimePlanView 不是模板；单独的 noinline/used 调用点保证它同样
// 经 AIC 与 AIV 后端完成代码生成，而不是只被 host 编译器验证布局。
extern "C" __aicore__ __attribute__((noinline, used)) uint64_t
pa_aicpu_plan_make_view_compile_probe(
    __gm__ pa_scheduler::aicpu_plan::RuntimePlanControl *control,
    __gm__ const pa_scheduler::aicpu_plan::RuntimePlanStorageRef *storage
) {
    pa_scheduler::aicpu_plan::RuntimePlanView view{};
    if (storage == nullptr ||
        !pa_scheduler::aicpu_plan::MakeRuntimePlanView(
            control, *storage, view
        )) {
        return 0U;
    }
    return reinterpret_cast<uint64_t>(view.cells) ^
           static_cast<uint64_t>(view.capacity);
}
