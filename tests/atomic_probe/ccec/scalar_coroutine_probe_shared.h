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

#ifndef TESTS_ATOMIC_PROBE_CCEC_SCALAR_COROUTINE_PROBE_SHARED_H_
#define TESTS_ATOMIC_PROBE_CCEC_SCALAR_COROUTINE_PROBE_SHARED_H_

#include <cstddef>
#include <cstdint>

namespace scalar_coroutine_probe {

#if defined(__CCE_AICORE__)
#define SCALAR_COROUTINE_SHARED_FN __aicore__
#else
#define SCALAR_COROUTINE_SHARED_FN
#endif

constexpr uint32_t kTileRows = 128U;
constexpr uint32_t kTileCols = 128U;
constexpr uint32_t kTileElements = kTileRows * kTileCols;
constexpr uint32_t kTileBytes = kTileElements * sizeof(float);
constexpr uint32_t kTaskRecords = 64U;
constexpr uint32_t kContextWords = 8U;
constexpr uint32_t kControlMagic = 0x434f524fU;  // "CORO"

enum class Role : uint32_t {
    Aic = 0U,
    Aiv = 1U,
};

enum class Mode : uint32_t {
    ContextFifo = 0U,
    EngineThenSchedule = 1U,
    EngineOverlapSchedule = 2U,
    TryWaitCrossCore = 3U,
    TryWaitIntraBlock = 4U,
    TryWaitBufferId = 5U,
    TryWaitTaggedBufferId = 6U,
    TryWaitTaggedDynamicBufferId = 7U,
    TryWaitPollUntilDone = 8U,
    TryWaitFourSlots = 9U,
    TryWaitIdleCost = 10U,
    Count = 11U,
};

enum class ContinuationState : uint32_t {
    Empty = 0U,
    Suspended = 1U,
    Resumed = 2U,
};

struct alignas(64) TaskRecord {
    uint64_t task_id;
    uint64_t next_delta;
    uint64_t ready_token;
    uint64_t payload[5];
};

// 显式 frame 模拟后续调度器必须保存的 continuation。基础探针不借用
// C++ 隐式活跃局部变量：挂起时逐字段复制，随后主动毒化原 context，
// 恢复时再逐字段读回并校验 signature。
struct alignas(64) ContinuationFrame {
    uint64_t words[kContextWords];
    uint64_t signature;
    uint32_t state;
    uint32_t generation;
    uint64_t reserved[6];
};

struct alignas(64) ProbeControl {
    uint64_t pmu_register_bases;
    uint64_t input_a;
    uint64_t input_b;
    uint64_t output;
    uint64_t task_records;
    uint32_t mode;
    uint32_t schedule_iterations;
    uint32_t seed;
    uint32_t magic;
    uint32_t try_wait_buffer_id;
    uint32_t reserved;
};

struct alignas(64) ProbeResult {
    uint64_t sys_ticks;
    uint64_t pmu_total_cycles;
    uint64_t pmu_vector_busy;
    uint64_t pmu_cube_busy;
    uint64_t pmu_scalar_busy;
    uint64_t pmu_mte1_busy;
    uint64_t pmu_mte2_busy;
    uint64_t pmu_mte3_busy;

    uint64_t pmu_fix_busy;
    uint64_t pmu_icache_request;
    uint64_t pmu_icache_miss;
    uint64_t physical_core_id;
    uint64_t selector_status;
    uint64_t context0_signature;
    uint64_t context1_signature;
    uint64_t schedule_checksum;

    uint64_t restored_context0_signature;
    uint64_t restored_context1_signature;
    uint64_t completion_before_wait;
    uint64_t completion_after_wait;
    uint64_t observed_mode;
    uint64_t observed_iterations;
    uint64_t observed_role;
    uint64_t protocol_status;
    uint64_t pmu_ctrl_after_stop;
    uint64_t resume_sequence;
    // try_wait 的返回值按有符号 64 位解释。四个采样点分别验证：
    // engine 发射前、发射后、scalar replay 后以及最终阻塞 wait 后。
    uint64_t try_wait_before_issue;
    uint64_t try_wait_after_issue;
    uint64_t try_wait_after_schedule;
    uint64_t try_wait_after_final_wait;
    uint64_t try_wait_positive_poll_count;
    uint64_t try_wait_poll_final_value;
};

struct alignas(64) ProbeState {
    ProbeControl control;
    ProbeResult result;
};

SCALAR_COROUTINE_SHARED_FN constexpr uint64_t Mix(uint64_t value) {
    value ^= value >> 29U;
    value *= 0x9e3779b185ebca87ULL;
    value ^= value >> 31U;
    return value;
}

SCALAR_COROUTINE_SHARED_FN constexpr uint64_t TaskWord(
    uint32_t record, uint32_t word, uint32_t seed
) {
    return Mix(
        (static_cast<uint64_t>(seed) << 32U) ^
        (static_cast<uint64_t>(record + 1U) * 0x100000001b3ULL) ^
        (static_cast<uint64_t>(word + 3U) * 0xd6e8feb86659fd93ULL)
    );
}

constexpr uint64_t kProtocolContext0Restored = 1ULL << 0;
constexpr uint64_t kProtocolContext1Restored = 1ULL << 1;
constexpr uint64_t kProtocolFifoOrder = 1ULL << 2;
constexpr uint64_t kProtocolCompletionHeldUntilWait = 1ULL << 3;
constexpr uint64_t kRequiredProtocolStatus =
    kProtocolContext0Restored | kProtocolContext1Restored |
    kProtocolFifoOrder | kProtocolCompletionHeldUntilWait;

static_assert(sizeof(TaskRecord) == 64U, "one synthetic task must occupy one cache line");
static_assert(sizeof(ContinuationFrame) == 128U, "continuation frame ABI changed");
static_assert(sizeof(ProbeControl) == 64U, "probe control must occupy one cache line");
static_assert(sizeof(ProbeResult) == 256U, "probe result must occupy four cache lines");
static_assert(offsetof(ProbeState, result) == 64U, "probe result offset changed");
static_assert(sizeof(ProbeState) == 320U, "probe state ABI changed");

#undef SCALAR_COROUTINE_SHARED_FN

}  // namespace scalar_coroutine_probe

#endif  // TESTS_ATOMIC_PROBE_CCEC_SCALAR_COROUTINE_PROBE_SHARED_H_
