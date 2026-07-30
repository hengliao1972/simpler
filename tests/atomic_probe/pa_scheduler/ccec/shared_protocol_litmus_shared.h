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
#ifndef TESTS_ATOMIC_PROBE_PA_SCHEDULER_CCEC_SHARED_PROTOCOL_LITMUS_SHARED_H
#define TESTS_ATOMIC_PROBE_PA_SCHEDULER_CCEC_SHARED_PROTOCOL_LITMUS_SHARED_H

#include <cstdint>

namespace pa_scheduler::shared_protocol_litmus {

constexpr uint32_t kControlMagic = 0x5350524CU;
constexpr uint32_t kControlVersion = 3;
constexpr uint32_t kSharedAbiGeneration = 13;
constexpr uint32_t kSymbolCount = 7;
constexpr uint64_t kResultMagic = 0x484953544F525900ULL;
static_assert(
    pa_scheduler::kBuildIdentityAbiGeneration ==
        kSharedAbiGeneration,
    "shared protocol litmus manifest ABI must follow the shared build identity"
);

enum class Scenario : uint32_t {
    SymbolHistory = 1,
    ReaderReclaim = 2,
};

enum class Direction : uint32_t {
    AicToAiv = 1,
    AivToAic = 2,
};

enum class ReaderOrdering : uint32_t {
    NotApplicable = 0,
    CompilerClobber = 1,
    PayloadDependency = 2,
    DsbAll = 3,
};

// 单次 launch 只选择一个场景和一个方向。后续场景共享同一 mixed ELF，
// 但各自独立初始化和断言，避免不同协议同时执行后互相掩盖故障。
// control 独占一条 GM cache line；所有 worker 在读取前显式失效该行。
struct alignas(64) Control {
    uint32_t magic;
    uint32_t version;
    uint32_t scenario;
    uint32_t direction;
    uint32_t reader_ordering;
    uint32_t reserved0;
    uint64_t launch_nonce;
    uint64_t reserved[4];
};
static_assert(
    sizeof(Control) == 64,
    "shared protocol litmus control must occupy one cache line"
);

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

struct ReaderReclaimChain {
    uint32_t reader_worker;
    uint32_t reclaimer_worker;
    int32_t blocked_signal;
    int32_t reuse_done_signal;
    uint64_t result_tag;
};

// reader/reclaimer 与 history 场景使用互不重叠的 worker、result 和 task gate，
// 使 host 能反向断言未选场景完全没有执行。
constexpr ReaderReclaimChain kReaderReclaimAicToAiv{
    3, 42, 200, 201, 0x40
};
constexpr ReaderReclaimChain kReaderReclaimAivToAic{
    35, 4, 210, 211, 0x80
};

constexpr uint32_t kReaderReclaimActiveWorkers = 96;
constexpr int32_t kReaderReclaimHeapWindow = 2;
constexpr int32_t kReaderReclaimTask = 2;
constexpr int32_t kReaderReclaimWriterTask = 5;
constexpr int32_t kReaderReclaimInitialDone = 1;
constexpr int32_t kReaderReclaimClosedDone = 2;
constexpr uint64_t kReaderReclaimAddress = 0x700000000ULL;
constexpr uint64_t kReaderReclaimLo = 0;
constexpr uint64_t kReaderReclaimHi = 8;
constexpr uint64_t kReaderReclaimReplacementAddress =
    0x7000014C0ULL;
constexpr uint64_t kReaderReclaimReplacementLo = 4096;
constexpr uint64_t kReaderReclaimReplacementHi = 4128;
constexpr uint64_t kReaderReclaimReaderStatus = 0x1F;
constexpr uint64_t kReaderReclaimReclaimerStatus = 0xFF;

}  // namespace pa_scheduler::shared_protocol_litmus

#endif  // TESTS_ATOMIC_PROBE_PA_SCHEDULER_CCEC_SHARED_PROTOCOL_LITMUS_SHARED_H
