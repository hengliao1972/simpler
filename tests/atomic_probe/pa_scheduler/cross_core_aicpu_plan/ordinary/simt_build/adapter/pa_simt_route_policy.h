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

#ifndef PA_SCHEDULER_AICPU_PLAN_ORDINARY_SIMT_PA_ROUTE_POLICY_H
#define PA_SCHEDULER_AICPU_PLAN_ORDINARY_SIMT_PA_ROUTE_POLICY_H

#ifndef PA_DEVICE
#error "PA_DEVICE must name the active CPU or CCEC function identity"
#endif

// PA 的业务解释只存在于 adapter。SIMT 公共 builder 只把 canonical
// Plan header 的 adapter_flags/adapter_data 原样交给这里，不允许根据
// task_id 周期、Host dispatch plan 或固定 task 表重新构造业务身份。
#include "../../scalar_build/common/aicpu_plan_pa_adapter.h"

#include <cstdint>

namespace pa_scheduler::aicpu_plan_simt::adapter {

// function_id 必须仅由 canonical header 的 PA kind provenance 决定。
// 这里不用 task_id 周期或 batch offset 重建 kind，也不复用依赖 winner
// 语境的 Scalar helper；显式 switch 让四个真实 kernel ID 与 metadata-only
// Alloc 的 invalid ID 在 adapter 边界独立 fail closed。
PA_DEVICE bool PaSimtFunctionIdMatchesAdapterKind(
    uint8_t adapter_flags, uint32_t function_id
)
{
    if ((adapter_flags & kSharedPaTicketMetaPresent) == 0U) {
        return false;
    }
    const TaskKind kind = static_cast<TaskKind>(
        adapter_flags & kSharedPaTicketKindMask
    );
    switch (kind) {
        case TaskKind::Alloc:
            return function_id == aicpu_plan::kInvalidFunctionId;
        case TaskKind::Qk:
            return function_id == 0U;
        case TaskKind::Sf:
            return function_id == 1U;
        case TaskKind::Pv:
            return function_id == 2U;
        case TaskKind::Up:
            return function_id == 3U;
        case TaskKind::Count:
            return false;
    }
    return false;
}

struct PaSimtRoutePolicy {
    PA_DEVICE bool Validate(
        uint32_t task_id, uint8_t adapter_flags,
        uint16_t adapter_data, uint32_t function_id,
        aicpu_plan::EngineClass engine
    ) const
    {
        if (!aicpu_plan_adapter::ValidatePaAdapterMetadata(
                task_id, engine, adapter_flags, adapter_data
            ) ||
            !PaSimtFunctionIdMatchesAdapterKind(
                adapter_flags, function_id
            )) {
            return false;
        }
        return engine == aicpu_plan::EngineClass::MetadataOnly ||
               engine == aicpu_plan::EngineClass::Aic ||
               engine == aicpu_plan::EngineClass::Aiv;
    }
};

}  // namespace pa_scheduler::aicpu_plan_simt::adapter

#endif  // PA_SCHEDULER_AICPU_PLAN_ORDINARY_SIMT_PA_ROUTE_POLICY_H
