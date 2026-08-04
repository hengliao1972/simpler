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

#ifndef TESTS_ATOMIC_PROBE_CCEC_ATOMIC_CONTENTION_CURVE_SHARED_H
#define TESTS_ATOMIC_PROBE_CCEC_ATOMIC_CONTENTION_CURVE_SHARED_H

#include <cstddef>
#include <cstdint>

namespace atomic_contention_curve {

constexpr uint32_t kConfigMagic = 0x4142434eU;
constexpr uint32_t kWorkerMagic = 0x41574b52U;
constexpr uint32_t kMaxAicWorkers = 32U;
constexpr uint32_t kMaxAivWorkers = 64U;
constexpr uint32_t kMaxWorkers = kMaxAicWorkers + kMaxAivWorkers;
constexpr uint32_t kHardwareSubcoresPerDie = 54U;
constexpr uint32_t kHardwareAicSubcoresPerDie = 18U;
constexpr uint32_t kHardwareSubcoreCount = 108U;
constexpr uint32_t kWarmupWaves = 4U;
constexpr uint32_t kMeasuredWaves = 64U;
constexpr uint32_t kTotalWaves = kWarmupWaves + kMeasuredWaves;
constexpr uint32_t kSweeps = 30U;
constexpr uint32_t kMaxStartSpreadTicks = 64U;
constexpr uint64_t kSysCounterHz = 1000000000ULL;
constexpr uint64_t kFirstDeadlineLeadTicks = 100000ULL;
constexpr uint64_t kWaveStrideTicks = 100000ULL;
constexpr uint64_t kWavePublishOffsetTicks = 50000ULL;
constexpr uint64_t kBarrierTimeoutTicks = 1000000000ULL;
constexpr uint64_t kGuardA = 0x13579bdf2468ace0ULL;
constexpr uint64_t kGuardB = 0xfdb97531eca86420ULL;

enum class Scenario : uint32_t {
    Mixed = 0U,
    Aic = 1U,
    Aiv = 2U,
};

enum class Role : uint32_t {
    Aic = 0U,
    Aiv = 1U,
};

enum class AddressLayout : uint32_t {
    Shared = 0U,
    ParticipantStride = 1U,
    // Pure-role probes split by participant id; mixed probes map AIC to the
    // primary address and AIV to the secondary address.
    TwoGroups = 2U,
};

enum StatusFlag : uint32_t {
    kStatusOk = 0U,
    kStatusConfigInvalid = 1U << 0U,
    kStatusBarrierTimeout = 1U << 1U,
    kStatusDeadlineMiss = 1U << 2U,
    kStatusWaveOverrun = 1U << 3U,
};

struct alignas(64) ProbeConfig {
    uint32_t magic;
    uint32_t scenario;
    uint32_t active_workers;
    uint32_t launched_workers;
    uint32_t active_aic;
    uint32_t active_aiv;
    uint32_t block_dim;
    uint32_t total_waves;
    uint32_t warmup_waves;
    uint32_t active_worker_start;
    uint64_t target_address;
    uint32_t address_layout;
    uint32_t target_stride_bytes;
    uint32_t primary_group_workers;
    uint32_t reserved;
};
static_assert(sizeof(ProbeConfig) == 64U, "probe config must occupy one cache line");

struct alignas(64) AtomicLine {
    volatile int64_t value;
    uint8_t padding[56];
};
static_assert(sizeof(AtomicLine) == 64U, "atomic control must occupy one cache line");

struct alignas(64) GuardLine {
    uint64_t first;
    uint64_t second;
    uint8_t padding[48];
};
static_assert(sizeof(GuardLine) == 64U, "guard must occupy one cache line");

struct alignas(64) ProbeState {
    ProbeConfig config;
    AtomicLine ready_workers;
    AtomicLine first_deadline;
    GuardLine guards;
};
static_assert(sizeof(ProbeState) == 256U, "unexpected probe-state layout");

struct alignas(64) WorkerResult {
    uint64_t first_deadline_tick;
    uint64_t publish_begin_tick;
    uint32_t magic;
    uint32_t worker_slot_id;
    uint32_t hardware_core_id;
    uint32_t participant_id;
    uint32_t role;
    uint32_t logical_core_index;
    uint32_t subblock_id;
    uint32_t active;
    uint32_t completed_waves;
    uint32_t status_flags;
    uint8_t padding[8];
};
static_assert(sizeof(WorkerResult) == 64U, "one worker result must occupy one cache line");

struct alignas(64) WaveResult {
    uint64_t begin_tick;
    uint64_t end_tick;
    uint64_t atomic_old;
    uint64_t deadline_tick;
    uint32_t participant_id;
    uint32_t wave;
    uint32_t status_flags;
    uint32_t role;
    uint64_t publish_begin_tick;
    uint64_t publish_ready_tick;
};
static_assert(sizeof(WaveResult) == 64U, "one wave result must occupy one cache line");

struct alignas(64) ProbeResults {
    WorkerResult workers[kMaxWorkers];
    WaveResult waves[kMaxWorkers * kTotalWaves];
};
static_assert(offsetof(ProbeResults, workers) == 0U, "workers must start the result buffer");
static_assert(offsetof(ProbeResults, waves) == sizeof(WorkerResult) * kMaxWorkers, "unexpected wave-result offset");

struct KernelArgs {
    uint64_t state_pointer;
    uint64_t results_pointer;
};
static_assert(sizeof(KernelArgs) == 16U, "unexpected kernel argument ABI");

}  // namespace atomic_contention_curve

#endif  // TESTS_ATOMIC_PROBE_CCEC_ATOMIC_CONTENTION_CURVE_SHARED_H
