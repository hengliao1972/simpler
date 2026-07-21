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

#ifndef FDWIC_DIST_ENGINE_AICPU_SUBMIT_PMU_OWNER_H_
#define FDWIC_DIST_ENGINE_AICPU_SUBMIT_PMU_OWNER_H_

#include "dist_engine/common/submit_pmu_types.h"

class Runtime;

/**
 * 判断本轮是否携带合法的 submit-pmu-none 共享头。
 *
 * submit-pmu 与 perf-clock 一样复用 Runtime::dist.swimlane_base，但使用独立
 * magic/version/mode 判型，并要求普通泳道 level/capacity 都为零。返回 true
 * 时，header_out 指向已经完成 AICPU cache invalidate 的共享头。
 */
bool fdwic_submit_pmu_requested(Runtime *runtime, FdwicSubmitPmuHeader **header_out);

/**
 * 在真实 AICore 握手完成后，为本轮 96 个活跃物理子核取得 PMU 所有权。
 *
 * 函数先完整校验 32 AIC + 64 AIV 拓扑，再逐物理核保存、配置、读回。
 * 任一配置失败都会逆序回滚已经取得所有权的槽；返回非零时不得放行业务
 * AICore 进入 dist_core_main。
 */
int fdwic_submit_pmu_owner_configure(Runtime *runtime, FdwicSubmitPmuHeader *header);

/**
 * 恢复 owner 尚未释放的全部物理子核。
 *
 * 恢复以 AICPU 私有 ownership bitmap 为准，按物理 id 107 -> 0 逆序执行；
 * 只有 16 个配置寄存器全部读回原值后才释放对应 bit。失败 bit 会保留，
 * 因而调用方可以幂等重试。
 */
int fdwic_submit_pmu_owner_restore(FdwicSubmitPmuHeader *header);

#endif  // FDWIC_DIST_ENGINE_AICPU_SUBMIT_PMU_OWNER_H_
