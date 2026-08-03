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

// One timed wave contains exactly one same-address atomicAdd from every active
// Scalar.  A GM atomic barrier is used only once, before a common future
// SYS_CNT deadline, so its traffic is outside every measured interval.

#include "atomic_contention_curve_shared.h"
#include "ccec_utils.h"

namespace {

using namespace atomic_contention_curve;

constexpr int64_t kAtomicReadIdentity = (-9223372036854775807LL - 1LL);

struct WorkerIdentity {
    uint32_t worker_slot_id;
    uint32_t hardware_core_id;
    uint32_t role;
    uint32_t logical_core_index;
    uint32_t subblock_id;
};

__aicore__ inline uint32_t LoadConfigWord(__gm__ ProbeState *state, uint32_t index) {
    return ld_dev_b32(reinterpret_cast<__gm__ uint32_t *>(&state->config) + index);
}

__aicore__ inline uint64_t LoadConfigDoubleWord(__gm__ ProbeState *state, uint32_t index) {
    return ld_dev_b64(reinterpret_cast<__gm__ uint64_t *>(&state->config) + index);
}

__aicore__ inline uint64_t ReadOrderedTick() {
    uint64_t tick = 0;
    asm volatile("MOV %0, SYS_CNT\n" : "=l"(tick) : : "memory");
    return tick;
}

template <typename T>
__aicore__ inline uint64_t AtomicResultReadyTick(T value) {
    static_assert(sizeof(T) == 4U || sizeof(T) == 8U, "atomic dependency expects a scalar result");
    uint64_t tick = 0;
    asm volatile("MOV %0, %0\n"
                 "MOV %1, SYS_CNT\n"
                 : "+l"(value), "=&l"(tick)
                 :
                 : "memory");
    return tick;
}

__aicore__ inline bool ConfigValid(__gm__ ProbeState *state, const WorkerIdentity &identity) {
    const uint32_t scenario = LoadConfigWord(state, 1U);
    const uint32_t active = LoadConfigWord(state, 2U);
    const uint32_t launched = LoadConfigWord(state, 3U);
    const uint32_t active_aic = LoadConfigWord(state, 4U);
    const uint32_t active_aiv = LoadConfigWord(state, 5U);
    const uint32_t block_dim = LoadConfigWord(state, 6U);
    const uint32_t active_start = LoadConfigWord(state, 9U);
    const uint64_t target_address = LoadConfigDoubleWord(state, 5U);
    if (LoadConfigWord(state, 0U) != kConfigMagic || active == 0U || active > launched || launched > kMaxWorkers ||
        active_aic + active_aiv != active || block_dim == 0U || block_dim > kMaxWorkers ||
        LoadConfigWord(state, 7U) != kTotalWaves || LoadConfigWord(state, 8U) != kWarmupWaves ||
        active_start > launched || active > launched - active_start || target_address == 0U ||
        target_address % alignof(AtomicLine) != 0U || identity.worker_slot_id >= launched ||
        identity.hardware_core_id >= kHardwareSubcoreCount) {
        return false;
    }

#if defined(ATOMIC_CURVE_BUILD_MIXED_AIC) || defined(ATOMIC_CURVE_BUILD_MIXED_AIV)
    return scenario == static_cast<uint32_t>(Scenario::Mixed) && block_dim <= kMaxAicWorkers &&
           launched == block_dim * 3U && active_start == 0U && active_aic == (active + 2U) / 3U &&
           active_aiv == active - active_aic;
#elif defined(ATOMIC_CURVE_BUILD_AIC)
    return scenario == static_cast<uint32_t>(Scenario::Aic) && block_dim <= kMaxAicWorkers && launched == block_dim &&
           active_aic == active && active_aiv == 0U;
#elif defined(ATOMIC_CURVE_BUILD_AIV)
    return scenario == static_cast<uint32_t>(Scenario::Aiv) && block_dim <= kMaxAivWorkers && launched == block_dim &&
           active_aic == 0U && active_aiv == active;
#else
    return false;
#endif
}

__aicore__ inline bool WaitForFirstDeadline(__gm__ ProbeState *state, uint32_t launched_workers, uint64_t &deadline) {
    const int64_t prior = atomicAdd(const_cast<__gm__ int64_t *>(&state->ready_workers.value), int64_t{1});
    if (prior + 1 == static_cast<int64_t>(launched_workers)) {
        const int64_t future = static_cast<int64_t>(ReadOrderedTick() + kFirstDeadlineLeadTicks);
        (void)atomicExch(const_cast<__gm__ int64_t *>(&state->first_deadline.value), future);
    }

    const uint64_t wait_begin = ReadOrderedTick();
    do {
        const int64_t observed =
            atomicMax(const_cast<__gm__ int64_t *>(&state->first_deadline.value), kAtomicReadIdentity);
        if (observed > 0) {
            deadline = static_cast<uint64_t>(observed);
            return true;
        }
        asm volatile("nop");
    } while (ReadOrderedTick() - wait_begin < kBarrierTimeoutTicks);
    deadline = 0U;
    return false;
}

__aicore__ inline void WaitUntil(uint64_t deadline) {
    while (ReadOrderedTick() < deadline) {
        asm volatile("nop");
    }
}

__aicore__ inline int64_t ExecuteMeasuredAtomic(__gm__ volatile int64_t *target) {
    return atomicAdd(const_cast<__gm__ int64_t *>(target), int64_t{1});
}

__aicore__ inline void PublishWorker(
    __gm__ WorkerResult *destination, const WorkerIdentity &identity, uint32_t active, uint32_t participant_id,
    uint32_t completed_waves, uint32_t status_flags, uint64_t first_deadline, uint64_t publish_begin_tick
) {
    destination->first_deadline_tick = first_deadline;
    destination->publish_begin_tick = publish_begin_tick;
    destination->magic = kWorkerMagic;
    destination->worker_slot_id = identity.worker_slot_id;
    destination->hardware_core_id = identity.hardware_core_id;
    destination->participant_id = active != 0U ? participant_id : UINT32_MAX;
    destination->role = identity.role;
    destination->logical_core_index = identity.logical_core_index;
    destination->subblock_id = identity.subblock_id;
    destination->active = active;
    destination->completed_waves = completed_waves;
    destination->status_flags = status_flags;
    dcci(destination, SINGLE_CACHE_LINE, CACHELINE_OUT);
}

__aicore__ inline void PublishWaveBody(
    __gm__ WaveResult *destination, uint64_t begin, uint64_t end, uint64_t atomic_old, uint64_t deadline,
    uint32_t participant, uint32_t wave, uint32_t role, uint64_t publish_begin
) {
    st_dev_b64(&destination->begin_tick, begin);
    st_dev_b64(&destination->end_tick, end);
    st_dev_b64(&destination->atomic_old, atomic_old);
    st_dev_b64(&destination->deadline_tick, deadline);
    st_dev_b32(&destination->participant_id, participant);
    st_dev_b32(&destination->wave, wave);
    st_dev_b32(&destination->role, role);
    st_dev_b64(&destination->publish_begin_tick, publish_begin);
}

__aicore__ inline void
RunProbe(__gm__ ProbeState *state, __gm__ ProbeResults *results, const WorkerIdentity &identity) {
    uint32_t status = kStatusOk;
    if (!ConfigValid(state, identity)) {
        status |= kStatusConfigInvalid;
        if (identity.worker_slot_id < kMaxWorkers) {
            PublishWorker(
                &results->workers[identity.worker_slot_id], identity, 0U, 0U, 0U, status, 0U, ReadOrderedTick()
            );
        }
        dsb(DSB_ALL);
        return;
    }

    const uint32_t active_workers = LoadConfigWord(state, 2U);
    const uint32_t launched_workers = LoadConfigWord(state, 3U);
    const uint32_t active_start = LoadConfigWord(state, 9U);
    __gm__ volatile int64_t *target = reinterpret_cast<__gm__ volatile int64_t *>(LoadConfigDoubleWord(state, 5U));
    const uint32_t active =
        identity.worker_slot_id >= active_start && identity.worker_slot_id - active_start < active_workers ? 1U : 0U;
    const uint32_t participant_id = active != 0U ? identity.worker_slot_id - active_start : 0U;
    uint64_t first_deadline = 0U;
    if (!WaitForFirstDeadline(state, launched_workers, first_deadline)) {
        status |= kStatusBarrierTimeout;
        PublishWorker(
            &results->workers[identity.worker_slot_id], identity, active, participant_id, 0U, status, first_deadline,
            ReadOrderedTick()
        );
        dsb(DSB_ALL);
        return;
    }

    if (active == 0U) {
        const uint64_t publish_deadline = first_deadline + static_cast<uint64_t>(kTotalWaves) * kWaveStrideTicks;
        WaitUntil(publish_deadline);
        const uint64_t publish_begin = ReadOrderedTick();
        PublishWorker(
            &results->workers[identity.worker_slot_id], identity, 0U, 0U, 0U, status, first_deadline, publish_begin
        );
        dsb(DSB_ALL);
        return;
    }

    // Each wave has a measurement phase and a later publication phase.  The
    // fixed midpoint keeps every result write outside every worker's timed
    // atomic interval; a late measurement or publication fails the launch.
    for (uint32_t wave = 0U; wave < kTotalWaves; ++wave) {
        const uint64_t deadline = first_deadline + static_cast<uint64_t>(wave) * kWaveStrideTicks;
        const uint64_t publish_deadline = deadline + kWavePublishOffsetTicks;
        const uint64_t next_deadline = deadline + kWaveStrideTicks;
        WaitUntil(deadline);
        const uint64_t begin = ReadOrderedTick();
        const int64_t old = ExecuteMeasuredAtomic(target);
        const uint64_t end = AtomicResultReadyTick(old);

        uint32_t wave_status = kStatusOk;
        if (begin < deadline) wave_status |= kStatusDeadlineMiss;
        if (end >= publish_deadline) wave_status |= kStatusWaveOverrun;

        WaitUntil(publish_deadline);
        const uint64_t publish_begin = ReadOrderedTick();
        __gm__ WaveResult *destination = &results->waves[identity.worker_slot_id * kTotalWaves + wave];
        PublishWaveBody(
            destination, begin, end, static_cast<uint64_t>(old), deadline, participant_id, wave, identity.role,
            publish_begin
        );
        dsb(DSB_ALL);
        const uint64_t publish_ready = ReadOrderedTick();
        if (publish_ready >= next_deadline) wave_status |= kStatusWaveOverrun;
        st_dev_b64(&destination->publish_ready_tick, publish_ready);
        st_dev_b32(&destination->status_flags, wave_status);
        dsb(DSB_ALL);
        if (ReadOrderedTick() >= next_deadline) {
            wave_status |= kStatusWaveOverrun;
            st_dev_b32(&destination->status_flags, wave_status);
            dsb(DSB_ALL);
        }
        status |= wave_status;
    }

    const uint64_t publish_deadline = first_deadline + static_cast<uint64_t>(kTotalWaves) * kWaveStrideTicks;
    WaitUntil(publish_deadline);
    const uint64_t publish_begin = ReadOrderedTick();
    PublishWorker(
        &results->workers[identity.worker_slot_id], identity, 1U, participant_id, kTotalWaves, status, first_deadline,
        publish_begin
    );
    dsb(DSB_ALL);
}

}  // namespace

#if defined(ATOMIC_CURVE_BUILD_MIXED_AIC)
PTO_SYNCALL_MIX_AIC_KERNEL_META(atomic_contention_curve_0_mix_aic, 1, 2);

extern "C" __global__ __aicore__ void atomic_contention_curve_0_mix_aic(
    __gm__ atomic_contention_curve::ProbeState *state, __gm__ atomic_contention_curve::ProbeResults *results
) {
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    const uint32_t hardware_core_id = static_cast<uint32_t>(get_coreid()) & 0x0fffU;
    RunProbe(state, results, WorkerIdentity{block * 3U, hardware_core_id, 0U, block, 0U});
}
#elif defined(ATOMIC_CURVE_BUILD_MIXED_AIV)
PTO_SYNCALL_MIX_AIC_KERNEL_META(atomic_contention_curve_0_mix_aiv, 1, 2);

extern "C" __global__ __aicore__ void atomic_contention_curve_0_mix_aiv(
    __gm__ atomic_contention_curve::ProbeState *state, __gm__ atomic_contention_curve::ProbeResults *results
) {
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    const uint32_t subblock = static_cast<uint32_t>(get_subblockid());
    const uint32_t vector_index = block * static_cast<uint32_t>(get_subblockdim()) + subblock;
    const uint32_t hardware_core_id = static_cast<uint32_t>(get_coreid()) & 0x0fffU;
    RunProbe(state, results, WorkerIdentity{block * 3U + 1U + subblock, hardware_core_id, 1U, vector_index, subblock});
}
#elif defined(ATOMIC_CURVE_BUILD_AIC)
PTO_SYNCALL_AIC_KERNEL_META(atomic_contention_curve_aic_0_mix_aic);

extern "C" __global__ __aicore__ void atomic_contention_curve_aic_0_mix_aic(
    __gm__ atomic_contention_curve::ProbeState *state, __gm__ atomic_contention_curve::ProbeResults *results
) {
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    const uint32_t hardware_core_id = static_cast<uint32_t>(get_coreid()) & 0x0fffU;
    RunProbe(state, results, WorkerIdentity{block, hardware_core_id, 0U, block, 0U});
}
#elif defined(ATOMIC_CURVE_BUILD_AIV)
PTO_SYNCALL_AIV_KERNEL_META(atomic_contention_curve_aiv_0_mix_aiv);

extern "C" __global__ __aicore__ void atomic_contention_curve_aiv_0_mix_aiv(
    __gm__ atomic_contention_curve::ProbeState *state, __gm__ atomic_contention_curve::ProbeResults *results
) {
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    const uint32_t hardware_core_id = static_cast<uint32_t>(get_coreid()) & 0x0fffU;
    RunProbe(state, results, WorkerIdentity{block, hardware_core_id, 1U, block, 0U});
}
#else
#error "Select one atomic-contention curve build topology"
#endif
