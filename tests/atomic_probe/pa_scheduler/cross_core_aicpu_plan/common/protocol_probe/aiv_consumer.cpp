/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */
#include "cce_aicore_intrinsics.h"
#include <pto/common/kernel_meta.hpp>

#include "protocol_shared.h"

#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif

PTO_SYNCALL_AIV_KERNEL_META(plan_protocol_probe_0_mix_aiv);

namespace {

using namespace plan_protocol_probe;

constexpr uint64_t kChecksumSeed = UINT64_C(14695981039346656037);
constexpr uint32_t kNoBadTask = UINT32_MAX;

__aicore__ inline int64_t Observe(__gm__ volatile int64_t *address)
{
    return atomicAdd(const_cast<__gm__ int64_t *>(address), int64_t{0});
}

__aicore__ inline void InvalidatePayload(
    __gm__ const RuntimeTaskPlanStorage *payload, uint32_t lines,
    ConsumerResult *result)
{
    for (uint32_t line = 0U; line < lines; ++line) {
        dcci(
            reinterpret_cast<__gm__ uint8_t *>(
                const_cast<__gm__ RuntimeTaskPlanStorage *>(payload)) + line * kCacheLineBytes,
            SINGLE_CACHE_LINE);
        ++result->payload_invalidated_lines;
    }
    dsb(DSB_ALL);
    __asm__ volatile("" ::: "memory");
}

__aicore__ bool ConsumeCell(
    __gm__ ProbeState *state, Workload workload, uint64_t nonce,
    uint32_t task, ConsumerResult *result)
{
    __gm__ RuntimeTaskPlanCell *cell = &state->cells[task];
    const int64_t first = Observe(&cell->control.value);
    ++result->control_observations;
    const DecodedCellControl decoded = DecodeCellControl(first);
    TaskShape shape{};
    if (!ResolveTask(workload, task, &shape)) {
        result->status = static_cast<uint32_t>(Status::BadTaskCount);
        result->first_bad_task = task;
        return false;
    }
    const PayloadLayout layout = LayoutForKind(shape.kind);
    const uint32_t expected_lines = layout.payload_lines;
    if (!decoded.valid || decoded.task_id != task || decoded.payload_lines != expected_lines) {
        result->status = static_cast<uint32_t>(Status::CellControlMismatch);
        result->first_bad_task = task;
        result->first_bad_word = UINT32_MAX;
        return false;
    }

    InvalidatePayload(&cell->payload, decoded.payload_lines, result);
    const int64_t second = Observe(&cell->control.value);
    ++result->control_observations;
    if (second != first) {
        result->status = static_cast<uint32_t>(Status::CellControlMismatch);
        result->first_bad_task = task;
        result->first_bad_word = UINT32_MAX;
        return false;
    }

    for (uint32_t word = 0U; word < decoded.payload_lines * 8U; ++word) {
        const uint64_t actual = cell->payload.words[word];
        const uint64_t expected = ExpectedPayloadWordForShape(shape, layout, nonce, word);
        if (actual != expected) {
            result->status = static_cast<uint32_t>(Status::PayloadMismatch);
            result->first_bad_task = task;
            result->first_bad_word = word;
            return false;
        }
        result->checksum = MixChecksum(result->checksum, actual);
    }
    return true;
}

__aicore__ inline void PublishResult(__gm__ ConsumerResult *destination, const ConsumerResult &result)
{
    const uint64_t word1 = static_cast<uint64_t>(result.status) |
        (static_cast<uint64_t>(result.task_count) << 32U);
    const uint64_t word7 = static_cast<uint64_t>(result.first_bad_task) |
        (static_cast<uint64_t>(result.first_bad_word) << 32U);
    __builtin_cce_st_dev(result.magic, reinterpret_cast<__gm__ uint64_t *>(destination) + 0, 0);
    __builtin_cce_st_dev(word1, reinterpret_cast<__gm__ uint64_t *>(destination) + 1, 0);
    __builtin_cce_st_dev(result.begin_ticks, reinterpret_cast<__gm__ uint64_t *>(destination) + 2, 0);
    __builtin_cce_st_dev(result.end_ticks, reinterpret_cast<__gm__ uint64_t *>(destination) + 3, 0);
    __builtin_cce_st_dev(result.control_observations, reinterpret_cast<__gm__ uint64_t *>(destination) + 4, 0);
    __builtin_cce_st_dev(result.payload_invalidated_lines,
                         reinterpret_cast<__gm__ uint64_t *>(destination) + 5, 0);
    __builtin_cce_st_dev(result.checksum, reinterpret_cast<__gm__ uint64_t *>(destination) + 6, 0);
    __builtin_cce_st_dev(word7, reinterpret_cast<__gm__ uint64_t *>(destination) + 7, 0);
    dsb(DSB_ALL);
}

}  // namespace

extern "C" __global__ __aicore__ void plan_protocol_probe_0_mix_aiv(
    __gm__ plan_protocol_probe::ProbeState *state)
{
    using namespace plan_protocol_probe;
    ConsumerResult result{};
    result.magic = kConsumerResultMagic;
    result.status = static_cast<uint32_t>(Status::Ok);
    result.first_bad_task = kNoBadTask;
    result.first_bad_word = UINT32_MAX;
    result.checksum = kChecksumSeed;

    dcci(&state->config, SINGLE_CACHE_LINE);
    dsb(DSB_ALL);
    __asm__ volatile("" ::: "memory");
    ProbeConfig config{};
    config.magic = state->config.magic;
    config.version = state->config.version;
    config.workload = state->config.workload;
    config.publication = state->config.publication;
    config.expected_tasks = state->config.expected_tasks;
    config.nonce = state->config.nonce;
    config.timeout_ticks = state->config.timeout_ticks;
    config.producer_delay_nops = state->config.producer_delay_nops;
    config.fault_mode = state->config.fault_mode;
    config.fault_task = state->config.fault_task;

    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    result.begin_ticks = begin;
    if (config.magic != kProbeMagic || config.version != kProbeVersion ||
        config.workload > static_cast<uint32_t>(Workload::B256) ||
        config.publication > static_cast<uint32_t>(Publication::PerItemFrontier) ||
        config.fault_mode > static_cast<uint32_t>(FaultMode::SkipLastPayloadCleanAndPoison) ||
        config.expected_tasks == 0U || config.expected_tasks > kMaxTasks) {
        result.status = static_cast<uint32_t>(Status::BadConfig);
        result.end_ticks = static_cast<uint64_t>(get_sys_cnt());
        PublishResult(&state->consumer, result);
        return;
    }

    const auto workload = static_cast<Workload>(config.workload);
    const auto publication = static_cast<Publication>(config.publication);
    const uint32_t expected_tasks = TaskCount(workload);
    if (expected_tasks != config.expected_tasks) {
        result.status = static_cast<uint32_t>(Status::BadTaskCount);
        result.end_ticks = static_cast<uint64_t>(get_sys_cnt());
        PublishResult(&state->consumer, result);
        return;
    }

    uint32_t consumed = 0U;
    int64_t last_frontier = 0;
    while (consumed < expected_tasks) {
        int64_t available = 0;
        if (publication == Publication::CloseOnly) {
            available = Observe(&state->closed_task_count.value);
            ++result.control_observations;
            if (available >= 0 && available != static_cast<int64_t>(expected_tasks)) {
                result.status = static_cast<uint32_t>(Status::CloseMismatch);
                break;
            }
        } else {
            available = Observe(&state->planned_frontier.value);
            ++result.control_observations;
            if (available < last_frontier) {
                result.status = static_cast<uint32_t>(Status::FrontierRegression);
                break;
            }
            last_frontier = available;
        }
        const int64_t fatal = Observe(&state->fatal.value);
        ++result.control_observations;
        if (fatal != 0) {
            result.status = static_cast<uint32_t>(Status::RemoteFatal);
            break;
        }
        if (publication == Publication::CloseOnly && available == kPlanOpen) {
            if (static_cast<uint64_t>(get_sys_cnt()) - begin > config.timeout_ticks) {
                result.status = static_cast<uint32_t>(Status::Timeout);
                break;
            }
            continue;
        }
        if (available < 0 || available > static_cast<int64_t>(expected_tasks)) {
            result.status = static_cast<uint32_t>(Status::FrontierOvershoot);
            break;
        }

        while (consumed < static_cast<uint32_t>(available)) {
            if (!ConsumeCell(state, workload, config.nonce, consumed, &result)) break;
            ++consumed;
        }
        if (result.status != static_cast<uint32_t>(Status::Ok)) break;
        if (static_cast<uint64_t>(get_sys_cnt()) - begin > config.timeout_ticks) {
            result.status = static_cast<uint32_t>(Status::Timeout);
            break;
        }
    }

    if (result.status == static_cast<uint32_t>(Status::Ok)) {
        int64_t closed = kPlanOpen;
        while (closed == kPlanOpen) {
            closed = Observe(&state->closed_task_count.value);
            ++result.control_observations;
            const int64_t fatal = Observe(&state->fatal.value);
            ++result.control_observations;
            if (fatal != 0) {
                result.status = static_cast<uint32_t>(Status::RemoteFatal);
                break;
            }
            if (static_cast<uint64_t>(get_sys_cnt()) - begin > config.timeout_ticks) {
                result.status = static_cast<uint32_t>(Status::Timeout);
                break;
            }
        }
        const int64_t frontier = Observe(&state->planned_frontier.value);
        ++result.control_observations;
        if (result.status == static_cast<uint32_t>(Status::Ok) &&
            (closed != static_cast<int64_t>(expected_tasks) || frontier != closed)) {
            result.status = static_cast<uint32_t>(Status::CloseMismatch);
        }
    }

    result.task_count = consumed;
    result.end_ticks = static_cast<uint64_t>(get_sys_cnt());
    PublishResult(&state->consumer, result);
}
