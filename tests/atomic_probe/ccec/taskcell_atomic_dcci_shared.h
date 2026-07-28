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
#pragma once

#include <cstddef>
#include <cstdint>

namespace taskcell_atomic_dcci_probe {

#if defined(__CCE_AICORE__)
#define TASKCELL_ATOMIC_DCCI_SHARED_FN __aicore__
#else
#define TASKCELL_ATOMIC_DCCI_SHARED_FN
#endif

inline constexpr uint32_t kAivBlocks = 2;
inline constexpr uint32_t kCacheLineBytes = 64;
inline constexpr uint32_t kScenarioCount = 5;
inline constexpr uint32_t kDefaultTrials = 100;
inline constexpr uint32_t kMaxTrials = 1000;
inline constexpr uint32_t kObservationPolls = 4096;
// 4096 次跨核 atomic 观察在 A5 上存在长尾；等待上限只负责识别真正
// 的协议停滞，不能比一个合法观察窗口更短。
inline constexpr uint64_t kWaitTimeoutTicks = 200000000ULL;

inline constexpr int64_t kInitialDeps = -1;
inline constexpr int64_t kNotApplicable = INT64_MIN;
inline constexpr int64_t kWriterMagic = 0x544344574f4e4501LL;
inline constexpr int64_t kReaderMagic = 0x544344524f4e4501LL;
inline constexpr uint64_t kInitialFlagBase = 0x1100000000000000ULL;
inline constexpr uint64_t kInitialVendBase = 0x2200000000000000ULL;
inline constexpr uint64_t kInitialTaskPaddingBase = 0x3300000000000000ULL;
inline constexpr uint64_t kBuiltFlagBase = 0x4400000000000000ULL;
inline constexpr uint64_t kBuiltVendBase = 0x5500000000000000ULL;
inline constexpr uint64_t kBuiltTaskPaddingBase = 0x6600000000000000ULL;
inline constexpr uint64_t kDedicatedGuardBase = 0x7700000000000000ULL;
inline constexpr uint64_t kStorageGuardBase = 0x7f00000000000000ULL;
inline constexpr int64_t kTokenBase = 0x0010000000000000LL;
enum class Scenario : uint32_t {
    SharedAtomicThenDcci = 0,
    SharedOrdinaryThenDcci = 1,
    DedicatedAtomicThenDcci = 2,
    DedicatedOrdinaryThenDcci = 3,
    DedicatedOrdinaryNoDcci = 4,
};

enum ResultFlag : uint16_t {
    kFlagNone = 0,
    kFlagInvalidTopology = 1U << 0,
    kFlagReadyTimeout = 1U << 1,
    kFlagPhaseTimeout = 1U << 2,
    kFlagAckTimeout = 1U << 3,
    kFlagControlTransition = 1U << 4,
    kFlagAtomicPublishFailed = 1U << 5,
    kFlagPreSawToken = 1U << 6,
    kFlagPostSawToken = 1U << 7,
    kFlagPrePollTimeout = 1U << 8,
    kFlagPostPollTimeout = 1U << 9,
    kFlagUnexpectedPollValue = 1U << 10,
    kFlagPollRegression = 1U << 11,
};

// 复刻 shared PA 当前 64B TaskCell 的字段位置。探针不包含生产头文件，
// 以免把待测 cache 行协议与完整 scheduler 的模板和诊断逻辑耦合。
struct alignas(kCacheLineBytes) TaskCellLine {
    volatile int64_t flag;
    volatile uint64_t vend;
    volatile int64_t deps_prepared;
    volatile uint64_t padding_words[5];
};
static_assert(sizeof(TaskCellLine) == kCacheLineBytes);
static_assert(alignof(TaskCellLine) == kCacheLineBytes);
static_assert(offsetof(TaskCellLine, flag) == 0);
static_assert(offsetof(TaskCellLine, vend) == 8);
static_assert(offsetof(TaskCellLine, deps_prepared) == 16);
static_assert(offsetof(TaskCellLine, padding_words) == 24);

// probe-only 的拆分布局：deps_prepared 独占整条 line，其余 56B 只放
// host 初始化的只读 guard，device 不得以 ordinary load/store 接触。
struct alignas(kCacheLineBytes) DedicatedDepsLine {
    volatile int64_t deps_prepared;
    volatile uint64_t guards[7];
};
static_assert(sizeof(DedicatedDepsLine) == kCacheLineBytes);
static_assert(alignof(DedicatedDepsLine) == kCacheLineBytes);
static_assert(offsetof(DedicatedDepsLine, deps_prepared) == 0);

struct alignas(kCacheLineBytes) AtomicLine {
    volatile int64_t value;
    uint8_t padding[kCacheLineBytes - sizeof(int64_t)];
};
static_assert(sizeof(AtomicLine) == kCacheLineBytes);
static_assert(alignof(AtomicLine) == kCacheLineBytes);

struct alignas(kCacheLineBytes) ResultLine {
    volatile int64_t words[8];
};
static_assert(sizeof(ResultLine) == kCacheLineBytes);
static_assert(alignof(ResultLine) == kCacheLineBytes);

struct alignas(kCacheLineBytes) ProbeStorage {
    TaskCellLine shared_task;
    DedicatedDepsLine dedicated_deps;
    AtomicLine role_claim;
    AtomicLine reader_ready;
    AtomicLine phase;
    AtomicLine reader_ack;
    ResultLine writer_result;
    ResultLine reader_result;
    ResultLine target_snapshot;
    ResultLine guard;
};
static_assert(sizeof(ProbeStorage) == 10 * kCacheLineBytes);
static_assert(offsetof(ProbeStorage, shared_task) == 0 * kCacheLineBytes);
static_assert(offsetof(ProbeStorage, dedicated_deps) == 1 * kCacheLineBytes);
static_assert(offsetof(ProbeStorage, role_claim) == 2 * kCacheLineBytes);
static_assert(offsetof(ProbeStorage, reader_ready) == 3 * kCacheLineBytes);
static_assert(offsetof(ProbeStorage, phase) == 4 * kCacheLineBytes);
static_assert(offsetof(ProbeStorage, reader_ack) == 5 * kCacheLineBytes);
static_assert(offsetof(ProbeStorage, writer_result) == 6 * kCacheLineBytes);
static_assert(offsetof(ProbeStorage, reader_result) == 7 * kCacheLineBytes);
static_assert(offsetof(ProbeStorage, target_snapshot) == 8 * kCacheLineBytes);
static_assert(offsetof(ProbeStorage, guard) == 9 * kCacheLineBytes);

struct KernelArgs {
    uint64_t storage_pointer;
    uint32_t trials;
    uint32_t num_blocks;
};
static_assert(sizeof(KernelArgs) == 16);

TASKCELL_ATOMIC_DCCI_SHARED_FN constexpr bool IsSharedScenario(
    Scenario scenario) {
    return scenario == Scenario::SharedAtomicThenDcci ||
           scenario == Scenario::SharedOrdinaryThenDcci;
}

TASKCELL_ATOMIC_DCCI_SHARED_FN constexpr bool IsAtomicPublishScenario(
    Scenario scenario) {
    return scenario == Scenario::SharedAtomicThenDcci ||
           scenario == Scenario::DedicatedAtomicThenDcci;
}

TASKCELL_ATOMIC_DCCI_SHARED_FN constexpr bool UsesDcci(Scenario scenario) {
    return scenario != Scenario::DedicatedOrdinaryNoDcci;
}

TASKCELL_ATOMIC_DCCI_SHARED_FN constexpr int64_t PublishedToken(
    uint32_t trial) {
    return kTokenBase + static_cast<int64_t>(trial);
}

TASKCELL_ATOMIC_DCCI_SHARED_FN constexpr uint64_t TrialTag(
    Scenario scenario, uint32_t trial) {
    return (static_cast<uint64_t>(scenario) << 32) | trial;
}

TASKCELL_ATOMIC_DCCI_SHARED_FN constexpr uint64_t InitialFlag(
    uint32_t trial) {
    return kInitialFlagBase | trial;
}

TASKCELL_ATOMIC_DCCI_SHARED_FN constexpr uint64_t InitialVend(
    uint32_t trial) {
    return kInitialVendBase | trial;
}

TASKCELL_ATOMIC_DCCI_SHARED_FN constexpr uint64_t InitialTaskPadding(
    uint32_t index, uint32_t trial) {
    return kInitialTaskPaddingBase | (static_cast<uint64_t>(index) << 32) | trial;
}

TASKCELL_ATOMIC_DCCI_SHARED_FN constexpr uint64_t BuiltFlag(
    uint32_t trial) {
    return kBuiltFlagBase | trial;
}

TASKCELL_ATOMIC_DCCI_SHARED_FN constexpr uint64_t BuiltVend(
    uint32_t trial) {
    return kBuiltVendBase | trial;
}

TASKCELL_ATOMIC_DCCI_SHARED_FN constexpr uint64_t BuiltTaskPadding(
    uint32_t index, uint32_t trial) {
    return kBuiltTaskPaddingBase | (static_cast<uint64_t>(index) << 32) | trial;
}

TASKCELL_ATOMIC_DCCI_SHARED_FN constexpr uint64_t DedicatedGuard(
    uint32_t index, uint32_t trial) {
    return kDedicatedGuardBase | (static_cast<uint64_t>(index) << 32) | trial;
}

TASKCELL_ATOMIC_DCCI_SHARED_FN constexpr uint64_t StorageGuard(
    uint32_t index, uint32_t trial) {
    return kStorageGuardBase | (static_cast<uint64_t>(index) << 32) | trial;
}

#undef TASKCELL_ATOMIC_DCCI_SHARED_FN

}  // namespace taskcell_atomic_dcci_probe
