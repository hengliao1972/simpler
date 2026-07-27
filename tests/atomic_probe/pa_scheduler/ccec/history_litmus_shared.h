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
#ifndef TESTS_ATOMIC_PROBE_PA_SCHEDULER_CCEC_HISTORY_LITMUS_SHARED_H
#define TESTS_ATOMIC_PROBE_PA_SCHEDULER_CCEC_HISTORY_LITMUS_SHARED_H

#include <cstdint>

namespace pa_scheduler::history_litmus {

constexpr uint32_t kControlMagic = 0x4853544CU;
constexpr uint32_t kControlVersion = 1;
constexpr uint32_t kSharedAbiGeneration = 7;
constexpr uint32_t kSymbolCount = 7;
constexpr uint64_t kResultMagic = 0x484953544F525900ULL;
static_assert(
    pa_scheduler::kBuildIdentityAbiGeneration ==
        kSharedAbiGeneration,
    "history litmus manifest ABI must follow the shared build identity"
);

enum class Direction : uint32_t {
    AicWritersToAivReader = 1,
    AivWritersToAicReader = 2,
};

// 单次 launch 只选择一个方向，避免两组共享历史同时执行后互相掩盖故障。
// control 独占一条 GM cache line；所有 worker 在读取前显式失效该行。
struct alignas(64) Control {
    uint32_t magic;
    uint32_t version;
    uint32_t direction;
    uint32_t reserved0;
    uint64_t launch_nonce;
    uint64_t reserved[5];
};
static_assert(sizeof(Control) == 64, "history litmus control must occupy one cache line");

struct HistoryChain {
    int32_t producer;
    int32_t writer_b;
    int32_t reader_c;
    int32_t writer_d;
    int32_t writer_e;
    int32_t reader_past_b_signal;
    int32_t future_done_signal;
    uint32_t writer_b_worker;
    uint32_t future_worker;
    uint32_t reader_worker;
    uint64_t result_tag;
};

// AIC writer 使用 block0/block1，AIV reader 使用物理 block4 的第一个
// vector 子核。三者不位于同一 mixed block，排除块内偶然共享状态。
constexpr HistoryChain kAicToAiv{
    10, 20, 30, 40, 50, 60, 61,
    0, 1, 40, 0x10
};

// 反向使用 AIV block0/sub1、AIV block1/sub0 和 AIC block2，同样跨越
// 不同物理 mixed block。
constexpr HistoryChain kAivToAic{
    110, 120, 130, 140, 150, 160, 161,
    33, 34, 2, 0x20
};

constexpr uint64_t kWriterBStatus = 1;
constexpr uint64_t kFutureWritersStatus = 0x0F;
constexpr uint64_t kReaderStatus = 0x3F;

}  // namespace pa_scheduler::history_litmus

#endif  // TESTS_ATOMIC_PROBE_PA_SCHEDULER_CCEC_HISTORY_LITMUS_SHARED_H
