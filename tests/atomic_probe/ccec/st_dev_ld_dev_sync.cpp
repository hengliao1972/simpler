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
// One-shot 2..20-AIV rendezvous (20 by default):
//   initial SyncAll (outside the measured interval)
//   -> block 0 writes one value with raw st_dev (no explicit DSB)
//   -> blocks 1..19 poll the same value with ld_dev
//   -> final SyncAll (inside the measured interval)
//
// Results are written only after the final timestamp, so result publication is
// not part of the reported synchronization span. No DCCI is used.
#include "ccec_utils.h"
#include "st_dev_ld_dev_sync_shared.h"

CCEC_PROBE_KERNEL_META(st_dev_ld_dev_sync);

namespace {

using st_dev_ld_dev_sync::ProbeStorage;
using st_dev_ld_dev_sync::WorkerResult;

__aicore__ inline uint64_t Timestamp()
{
    __asm__ __volatile__("" ::: "memory");
    return static_cast<uint64_t>(get_sys_cnt());
}

__aicore__ inline void PublishResult(
    __gm__ WorkerResult *result, uint32_t block_id, uint32_t block_count,
    uint32_t observed_value, uint32_t poll_count, uint32_t flags,
    uint64_t begin_tick, uint64_t observe_tick,
    uint64_t final_barrier_arrive_tick, uint64_t end_tick)
{
    st_dev_b32(&result->magic, st_dev_ld_dev_sync::kResultMagic);
    st_dev_b32(&result->block_id, block_id);
    st_dev_b32(&result->core_id, static_cast<uint32_t>(get_coreid()));
    st_dev_b32(&result->subblock_id, static_cast<uint32_t>(get_subblockid()));
    st_dev_b32(&result->block_count, block_count);
    st_dev_b32(&result->observed_value, observed_value);
    st_dev_b32(&result->poll_count, poll_count);
    st_dev_b32(&result->flags, flags);
    st_dev_b64(&result->begin_tick, begin_tick);
    st_dev_b64(&result->observe_tick, observe_tick);
    st_dev_b64(&result->final_barrier_arrive_tick, final_barrier_arrive_tick);
    st_dev_b64(&result->end_tick, end_tick);
    // Host-result publication happens after end_tick and is not part of the
    // tested signal-store protocol or any reported timing interval.
    dsb(DSB_ALL);
}

}  // namespace

extern "C" __global__ __aicore__ void
KERNEL_ENTRY(st_dev_ld_dev_sync)(__gm__ st_dev_ld_dev_sync::ProbeStorage *storage)
{
    const uint32_t block_id = static_cast<uint32_t>(get_block_idx());
    const uint32_t block_count = static_cast<uint32_t>(get_block_num());
    uint32_t observed_value = 0;
    uint32_t poll_count = 0;
    uint32_t flags =
        block_count >= st_dev_ld_dev_sync::kMinAivWorkers &&
        block_count <= st_dev_ld_dev_sync::kMaxAivWorkers
            ? 0U
            : st_dev_ld_dev_sync::kFlagInvalidBlockCount;

    ccec_sync_all();
    const uint64_t begin_tick = Timestamp();

    // signal.value is the only live object on this 64 B cache line. Both the
    // writer's st_dev and every reader's ld_dev target that exclusive line.
    if (block_id == 0U) {
        st_dev_b32(&storage->signal.value, st_dev_ld_dev_sync::kExpectedValue);
        observed_value = st_dev_ld_dev_sync::kExpectedValue;
    } else {
        const uint64_t wait_begin = Timestamp();
        do {
            observed_value = ld_dev_b32(&storage->signal.value);
            ++poll_count;
            if (observed_value == st_dev_ld_dev_sync::kExpectedValue) {
                break;
            }
        } while (Timestamp() - wait_begin < st_dev_ld_dev_sync::kWaitTimeoutTicks);
        if (observed_value != st_dev_ld_dev_sync::kExpectedValue) {
            flags |= st_dev_ld_dev_sync::kFlagReadTimeout;
        }
    }

    const uint64_t observe_tick = Timestamp();
    const uint64_t final_barrier_arrive_tick = Timestamp();
    ccec_sync_all();
    const uint64_t end_tick = Timestamp();

    if (block_id < st_dev_ld_dev_sync::kMaxAivWorkers) {
        PublishResult(
            &storage->workers[block_id], block_id, block_count, observed_value,
            poll_count, flags, begin_tick, observe_tick,
            final_barrier_arrive_tick, end_tick);
    }
}
