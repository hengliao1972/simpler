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

#ifndef PA_SCHEDULER_COMMON_WINNER_WORKLOAD_H
#define PA_SCHEDULER_COMMON_WINNER_WORKLOAD_H

#include "pa_model.h"

#include <stddef.h>
#include <stdint.h>

namespace pa_scheduler::winner_workload {

// 三种 standalone 后端共享完全相同的 GM/host 布局。128x128 float 是 CCEC
// 已在 A5 上验证过的基本形状；后端可以采用不同指令接口，但不能改变输入、
// 输出 tile 编址或 host 数值校验口径。
constexpr uint32_t kTileRows = 128;
constexpr uint32_t kTileCols = 128;
constexpr size_t kTileElements = static_cast<size_t>(kTileRows) * kTileCols;
constexpr size_t kTileBytes = kTileElements * sizeof(float);
constexpr uint32_t kSharedInputTiles = 2;
constexpr uint32_t kOutputTilesPerWorker = 2;
constexpr uint32_t kOutputTiles = kWorkers * kOutputTilesPerWorker;
constexpr size_t kWorkspaceTiles = kSharedInputTiles + kOutputTiles;
constexpr size_t kWorkspaceBytes = kWorkspaceTiles * kTileBytes;
constexpr float kInputAValue = 2.0F;
constexpr float kInputBValue = 3.0F;
constexpr float kExpectedAicValue = 768.0F;
constexpr float kExpectedSfValue = 5.0F;
constexpr float kExpectedUpValue = 6.0F;
constexpr float kOutputSentinel = -12345.0F;

// 256 batch 下，即使同一 AIC 极端地拿到全部 QK/PV，128 次完整 Cube
// 迭代的 CCEC 实测 busy 上界仍低于 32-bit PMU 的 25% 门槛。其他后端也沿用
// 此参数边界，避免相同命令在不同实现上产生不同含义。
constexpr uint32_t kMaxRealComputeCount = 128;

// 1 次用于最小正确性取证。默认次数来自 CCEC 的三个独立 b256 A5 进程；
// AscendC 必须重新标定后才能宣称达到同样时长，不能仅因共享默认参数便沿用
// CCEC 的性能结论。UP 的一次完整 128x128 流水是当前正整数下限。
constexpr WorkloadCounts kRealComputeSmokeCounts{1, 1, 1, 1};
constexpr WorkloadCounts kDefaultRealComputeCounts{6, 28, 4, 1};

static_assert(kTileBytes == 65536, "real-compute tile must occupy 64 KiB");
static_assert(kWorkspaceBytes == 12713984, "real-compute workspace size changed unexpectedly");

}  // namespace pa_scheduler::winner_workload

#endif  // PA_SCHEDULER_COMMON_WINNER_WORKLOAD_H
