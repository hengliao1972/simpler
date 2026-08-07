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

#ifndef PA_SCHEDULER_CROSS_CORE_A5_EXEC_POLICY_H
#define PA_SCHEDULER_CROSS_CORE_A5_EXEC_POLICY_H

#include "shared_exec_protocol.h"

namespace pa_scheduler::cross_core_dag {

// 这是 A5 单 lane 执行的后端 placement 策略，不是 PA 算子规则：
// 32 个 AIC Scalar 后接 64 个 AIV Scalar，AIV worker 按两个 vector lane
// 交错编号。所有满足同一单 engine 合同的算子都可以复用；Joint、固定
// block affinity 和 multicore task 必须选择另一份后端策略。
constexpr uint32_t kA5AicScalarWorkers = 32;
constexpr uint32_t kA5AivScalarWorkers = 64;
constexpr uint32_t kA5ScalarWorkers =
    kA5AicScalarWorkers + kA5AivScalarWorkers;
static_assert(
    kA5AicScalarWorkers > 0 &&
        kA5AivScalarWorkers > 0 &&
        kA5AivScalarWorkers % 2U == 0,
    "A5 single-lane execution requires AIC and paired AIV Scalars"
);
static_assert(
    kA5ScalarWorkers <= kExecUnboundOwner,
    "A5 worker ids must not collide with the unbound owner sentinel"
);

PA_DEVICE bool A5SingleLaneOwnerMatchesEngine(
    uint32_t owner, ExecEngineClass engine_class
) {
    if (owner >= kA5ScalarWorkers) {
        return false;
    }
    switch (engine_class) {
        case ExecEngineClass::Aic:
            return owner < kA5AicScalarWorkers;
        case ExecEngineClass::Aiv:
            return owner >= kA5AicScalarWorkers;
        case ExecEngineClass::None:
        case ExecEngineClass::Joint:
            return false;
    }
    return false;
}

// 单 engine payload 可由任一有效 Scalar 构建；engine 只约束执行 owner。
PA_DEVICE bool A5SingleLaneBuildOwnerEligible(
    uint32_t build_owner, ExecEngineClass engine_class
) {
    if (build_owner >= kA5ScalarWorkers) {
        return false;
    }
    switch (engine_class) {
        case ExecEngineClass::Aic:
        case ExecEngineClass::Aiv:
            return true;
        case ExecEngineClass::None:
        case ExecEngineClass::Joint:
            return false;
    }
    return false;
}

PA_DEVICE bool A5SingleLaneExecuteOwnerEligible(
    uint32_t task_id, uint32_t build_owner,
    ExecEngineClass engine_class, uint32_t execute_owner
) {
    // 双中央 Execute ticket 已经按 engine role 把每个 task 唯一发给一个
    // Scalar。Build/Execute owner 是两次独立决定，但不强制物理核不同。
    // task_id 仍保留在公共签名中，便于终态诊断与未来 placement 扩展。
    (void)task_id;
    return build_owner < kA5ScalarWorkers &&
        A5SingleLaneOwnerMatchesEngine(
            execute_owner, engine_class
        );
}

}  // namespace pa_scheduler::cross_core_dag

#endif  // PA_SCHEDULER_CROSS_CORE_A5_EXEC_POLICY_H
