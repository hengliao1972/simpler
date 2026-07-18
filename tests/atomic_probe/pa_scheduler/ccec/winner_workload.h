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

#ifndef PA_SCHEDULER_CCEC_WINNER_WORKLOAD_H
#define PA_SCHEDULER_CCEC_WINNER_WORKLOAD_H

#include "../common/pa_model.h"

#include <stddef.h>
#include <stdint.h>

namespace pa_scheduler::ccec_workload {

// 128x128 float 是仓内 A5 QK/PV cube 与 vector 示例共同验证过的基本形状。
// 两个输入由 96 个 worker 只读共享；每个 worker 为本角色的两种 task kind
// 各占一个输出 tile，既避免写竞争，也能分别核对 QK/PV 与 SF/UP 的数值。
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
// 256 batch 下，即使同一 AIC 极端地拿到全部 QK/PV，128 次完整 cube
// 迭代的实测 busy 上界仍低于 32-bit PMU 的 25% 门槛；更大的未经取证值
// 既偏离约 50 us task 目标，也可能让 submit-all 计数整圈回绕后伪装成小值。
constexpr uint32_t kMaxRealComputeCount = 128;

// 1 次仅用于最小 b1 取证。默认次数来自本机 A5 的三个独立 256-batch
// 进程：QK/SF/PV 分别最接近 44.170/53.729/27.626 us。UP 的 1 次
// 128x128 完整 load/vector/store/drain 已是合法下限（约 2.5 us），无法仅靠
// 正整数 repeats 降到 1.565 us；若后续要继续贴准，应缩小 UP tile，而不是用 0 次。
constexpr WorkloadCounts kRealComputeSmokeCounts{1, 1, 1, 1};
constexpr WorkloadCounts kDefaultRealComputeCounts{6, 28, 4, 1};

static_assert(kTileBytes == 65536, "real-compute tile must occupy 64 KiB");
static_assert(kWorkspaceBytes == 12713984, "real-compute workspace size changed unexpectedly");

}  // namespace pa_scheduler::ccec_workload

#endif  // PA_SCHEDULER_CCEC_WINNER_WORKLOAD_H
