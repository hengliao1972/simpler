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

// 模拟 PA completion flag 的一写一轮询路径。block0 是唯一 writer，其余
// AIV 是 reader。每轮 reader 先完成精确 31 次 identity RMW load，再发布
// ready；writer 只在全部 reader ready 后计时一次 AtomicExch(flag, token)。
//
// 每种 load opcode 都比较四类地址关系：
//   different-line：31 次预读和后续轮询都在 signal，计时 exchange 的
//                   target 从未被 reader 访问，且二者相隔 64B；
//   same-line-different-word：reader 轮询 target.neighbor，writer 写
//                            target.value，二者同属一个 64B line；
//   then-away：先对 target 精确读 31 次，ready 后改为轮询 signal，检查已经
//              停止的历史读取是否仍会拖慢 target exchange；
//   same：先对 target 读 31 次并继续同址轮询，直接测持续 identity RMW 与
//         completion exchange 的竞争。
//
// AtomicAdd(0) 对应仓库基线 atomic_load；AtomicMax(INT64_MIN) 对应当前
// shared TensorMap int64_t identity load。target、signal 和所有控制/结果均
// 独占 64B line。控制原子不触碰 target，且所有等待都有有限超时。

#include "atomic_poll_exchange_contention_shared.h"
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(atomic_poll_exchange_contention);

namespace {

using namespace atomic_poll_exchange_contention;

constexpr int64_t kAtomicMaxIdentity = (-9223372036854775807LL - 1LL);

__aicore__ inline uint32_t DeviceModeIndex(ProbeMode mode) { return static_cast<uint32_t>(mode); }

__aicore__ inline bool DeviceUsesAtomicMax(ProbeMode mode) {
    return DeviceModeIndex(mode) >= DeviceModeIndex(ProbeMode::MaxDifferentLine);
}

__aicore__ inline uint32_t DeviceAddressPattern(ProbeMode mode) { return DeviceModeIndex(mode) % 4U; }

__aicore__ inline bool DevicePollsTargetUntilPublish(ProbeMode mode) { return DeviceAddressPattern(mode) == 3U; }

__aicore__ inline int64_t DeviceToken(ProbeMode mode, uint32_t round) {
    return static_cast<int64_t>(
        0x5A00000000000000ULL | (static_cast<uint64_t>(DeviceModeIndex(mode)) << 48U) |
        static_cast<uint64_t>(round + 1U)
    );
}

__aicore__ inline int64_t AtomicAddLoad(__gm__ volatile int64_t *address) {
    return atomicAdd(const_cast<__gm__ int64_t *>(address), static_cast<int64_t>(0));
}

__aicore__ inline int64_t AtomicIdentityLoad(__gm__ volatile int64_t *address, bool use_atomic_max) {
    __gm__ int64_t *mutable_address = const_cast<__gm__ int64_t *>(address);
    return use_atomic_max ? atomicMax(mutable_address, kAtomicMaxIdentity) :
                            atomicAdd(mutable_address, static_cast<int64_t>(0));
}

__aicore__ inline int64_t AtomicExchange(__gm__ volatile int64_t *address, int64_t desired) {
    return static_cast<int64_t>(atomicExch(
        reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ int64_t *>(address)), static_cast<uint64_t>(desired)
    ));
}

template <typename T>
__aicore__ inline uint64_t AtomicResultReadyTick(T value) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "atomic dependency expects a scalar result");
    uint64_t cycle = 0;
    // 与生产泳道的 return-ready hook 相同：在读取 SYS_CNT 的同一汇编
    // 块中消费 atomic 返回值。它只建立本核返回依赖，不增加 DSB/ISB
    // 或 GM 访问，也不宣称跨核可见性完成。
    asm volatile("MOV %0, %0\n"
                 "MOV %1, SYS_CNT\n"
                 : "+l"(value), "=&l"(cycle));
    return cycle;
}

__aicore__ inline bool WaitAtLeast(__gm__ volatile int64_t *address, int64_t target, int64_t &actual) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    do {
        actual = AtomicAddLoad(address);
        if (actual >= target) return true;
    } while (static_cast<uint64_t>(get_sys_cnt()) - begin < kWaitTimeoutTicks);
    actual = AtomicAddLoad(address);
    return actual >= target;
}

__aicore__ inline void ArmReaders() {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    while (static_cast<uint64_t>(get_sys_cnt()) - begin < kArmTicks) {}
}

__aicore__ inline void PublishTiming(
    __gm__ TimingRecord *record, uint64_t writer_begin, uint64_t writer_end, uint64_t notify_begin, uint64_t notify_end,
    uint64_t all_ack, int64_t old_value, uint32_t ready_actual, uint32_t ack_actual, uint32_t flags, uint32_t round
) {
    st_dev_b64(&record->writer_begin, writer_begin);
    st_dev_b64(&record->writer_end, writer_end);
    st_dev_b64(&record->notify_begin, notify_begin);
    st_dev_b64(&record->notify_end, notify_end);
    st_dev_b64(&record->all_ack, all_ack);
    st_dev_b64(&record->old_value, static_cast<uint64_t>(old_value));
    st_dev_b32(&record->ready_actual, ready_actual);
    st_dev_b32(&record->ack_actual, ack_actual);
    st_dev_b32(&record->flags, flags);
    st_dev_b32(&record->magic_round, kTimingMagic | round);
    dsb(DSB_ALL);
}

__aicore__ inline void PublishReader(
    __gm__ ReaderResult *result, uint32_t block, uint64_t total_pre_polls, uint64_t total_followup_polls,
    uint64_t max_observe_ticks, int64_t last_observed, uint32_t seen_rounds, uint32_t pre_poll_mismatches,
    uint32_t epoch_timeouts, uint32_t observe_timeouts
) {
    st_dev_b64(&result->total_pre_polls, total_pre_polls);
    st_dev_b64(&result->total_followup_polls, total_followup_polls);
    st_dev_b64(&result->max_observe_ticks, max_observe_ticks);
    st_dev_b64(&result->last_observed, static_cast<uint64_t>(last_observed));
    st_dev_b32(&result->block, block);
    st_dev_b32(&result->core, static_cast<uint32_t>(get_coreid()));
    st_dev_b32(&result->subblock, static_cast<uint32_t>(get_subblockid()));
    st_dev_b32(&result->seen_rounds, seen_rounds);
    st_dev_b32(&result->pre_poll_mismatches, pre_poll_mismatches);
    st_dev_b32(&result->epoch_timeouts, epoch_timeouts);
    st_dev_b32(&result->observe_timeouts, observe_timeouts);
    dsb(DSB_ALL);
    st_dev_b32(&result->magic, kReaderMagic);
    dsb(DSB_ALL);
}

__aicore__ inline void PublishHeader(
    __gm__ ProbeHeader *header, ProbeMode mode, uint32_t blocks, uint32_t ready_timeouts, uint32_t ack_timeouts,
    uint32_t old_value_mismatches, int64_t final_target, int64_t final_signal, uint32_t finish
) {
    st_dev_b32(&header->magic, kHeaderMagic);
    st_dev_b32(&header->mode, DeviceModeIndex(mode));
    st_dev_b32(&header->blocks, blocks);
    st_dev_b32(&header->readers, blocks - 1U);
    st_dev_b32(&header->rounds, kRounds);
    st_dev_b32(&header->pre_polls, kPrePolls);
    st_dev_b32(&header->ready_timeouts, ready_timeouts);
    st_dev_b32(&header->ack_timeouts, ack_timeouts);
    st_dev_b32(&header->old_value_mismatches, old_value_mismatches);
    st_dev_b32(&header->final_target_low, static_cast<uint32_t>(final_target));
    st_dev_b32(&header->final_signal_low, static_cast<uint32_t>(final_signal));
    st_dev_b32(&header->writer_core, static_cast<uint32_t>(get_coreid()));
    st_dev_b32(&header->writer_subblock, static_cast<uint32_t>(get_subblockid()));
    st_dev_b32(&header->timeout_ticks, static_cast<uint32_t>(kWaitTimeoutTicks));
    st_dev_b32(&header->arm_ticks, static_cast<uint32_t>(kArmTicks));
    dsb(DSB_ALL);
    st_dev_b32(&header->finish, finish);
    dsb(DSB_ALL);
}

__aicore__ inline void WriterMain(__gm__ ProbeStorage *storage, ProbeMode mode, uint32_t blocks) {
    const uint32_t readers = blocks - 1U;
    uint32_t ready_timeouts = 0;
    uint32_t ack_timeouts = 0;
    uint32_t old_value_mismatches = 0;

    for (uint32_t round = 0; round < kRounds; ++round) {
        (void)AtomicExchange(&storage->target.value, 0);
        (void)AtomicExchange(&storage->target.neighbor, 0);
        (void)AtomicExchange(&storage->signal.value, 0);
        (void)AtomicExchange(&storage->epoch.value, static_cast<int64_t>(round + 1U));

        const int64_t ready_target = static_cast<int64_t>(readers) * static_cast<int64_t>(round + 1U);
        int64_t ready_actual = 0;
        const bool ready_ok = WaitAtLeast(&storage->ready.value, ready_target, ready_actual);
        if (!ready_ok) ++ready_timeouts;

        ArmReaders();
        const int64_t token = DeviceToken(mode, round);
        const uint64_t writer_begin = static_cast<uint64_t>(get_sys_cnt());
        const int64_t old_value = AtomicExchange(&storage->target.value, token);
        const uint64_t writer_end = AtomicResultReadyTick(old_value);

        uint64_t notify_begin = writer_end;
        uint64_t notify_end = writer_end;
        int64_t notify_old = 0;
        if (!DevicePollsTargetUntilPublish(mode)) {
            __gm__ volatile int64_t *notify_address =
                DeviceAddressPattern(mode) == 1U ? &storage->target.neighbor : &storage->signal.value;
            notify_begin = static_cast<uint64_t>(get_sys_cnt());
            notify_old = AtomicExchange(notify_address, token);
            notify_end = AtomicResultReadyTick(notify_old);
        }

        const int64_t ack_target = static_cast<int64_t>(readers) * static_cast<int64_t>(round + 1U);
        int64_t ack_actual = 0;
        const bool ack_ok = WaitAtLeast(&storage->ack.value, ack_target, ack_actual);
        const uint64_t all_ack = static_cast<uint64_t>(get_sys_cnt());
        if (!ack_ok) ++ack_timeouts;

        uint32_t flags = 0;
        if (!ready_ok) flags |= kFlagReadyTimeout;
        if (!ack_ok) flags |= kFlagAckTimeout;
        if (old_value != 0) flags |= kFlagTargetOldMismatch;
        if (!DevicePollsTargetUntilPublish(mode) && notify_old != 0) {
            flags |= kFlagNotifyOldMismatch;
        }
        if (flags & (kFlagTargetOldMismatch | kFlagNotifyOldMismatch)) {
            ++old_value_mismatches;
        }
        PublishTiming(
            &storage->timing[round], writer_begin, writer_end, notify_begin, notify_end, all_ack, old_value,
            static_cast<uint32_t>(ready_actual), static_cast<uint32_t>(ack_actual), flags, round
        );
    }

    PublishHeader(
        &storage->header, mode, blocks, ready_timeouts, ack_timeouts, old_value_mismatches,
        AtomicAddLoad(&storage->target.value), AtomicAddLoad(&storage->signal.value), kFinishMagic
    );
}

__aicore__ inline void ReaderMain(__gm__ ProbeStorage *storage, ProbeMode mode, uint32_t block) {
    const uint32_t pattern = DeviceAddressPattern(mode);
    const bool use_atomic_max = DeviceUsesAtomicMax(mode);
    __gm__ volatile int64_t *pre_address = pattern == 0U ? &storage->signal.value :
                                           pattern == 1U ? &storage->target.neighbor :
                                                           &storage->target.value;
    __gm__ volatile int64_t *followup_address = pattern == 1U ? &storage->target.neighbor :
                                                pattern == 3U ? &storage->target.value :
                                                                &storage->signal.value;
    uint64_t total_pre_polls = 0;
    uint64_t total_followup_polls = 0;
    uint64_t max_observe_ticks = 0;
    int64_t last_observed = 0;
    uint32_t seen_rounds = 0;
    uint32_t pre_poll_mismatches = 0;
    uint32_t epoch_timeouts = 0;
    uint32_t observe_timeouts = 0;

    for (uint32_t round = 0; round < kRounds; ++round) {
        int64_t epoch_actual = 0;
        if (!WaitAtLeast(&storage->epoch.value, static_cast<int64_t>(round + 1U), epoch_actual)) {
            ++epoch_timeouts;
        }

        for (uint32_t poll = 0; poll < kPrePolls; ++poll) {
            const int64_t value = AtomicIdentityLoad(pre_address, use_atomic_max);
            ++total_pre_polls;
            if (value != 0) ++pre_poll_mismatches;
        }
        (void)atomicAdd(const_cast<__gm__ int64_t *>(&storage->ready.value), static_cast<int64_t>(1));

        const int64_t token = DeviceToken(mode, round);
        const uint64_t observe_begin = static_cast<uint64_t>(get_sys_cnt());
        bool observed = false;
        do {
            last_observed = AtomicIdentityLoad(followup_address, use_atomic_max);
            ++total_followup_polls;
            if (last_observed == token) {
                observed = true;
                ++seen_rounds;
                break;
            }
        } while (static_cast<uint64_t>(get_sys_cnt()) - observe_begin < kWaitTimeoutTicks);
        const uint64_t observe_ticks = static_cast<uint64_t>(get_sys_cnt()) - observe_begin;
        if (observe_ticks > max_observe_ticks) {
            max_observe_ticks = observe_ticks;
        }
        if (!observed) ++observe_timeouts;

        (void)atomicAdd(const_cast<__gm__ int64_t *>(&storage->ack.value), static_cast<int64_t>(1));
    }

    PublishReader(
        &storage->readers[block], block, total_pre_polls, total_followup_polls, max_observe_ticks, last_observed,
        seen_rounds, pre_poll_mismatches, epoch_timeouts, observe_timeouts
    );
}

}  // namespace

extern "C" __global__ __aicore__ void KERNEL_ENTRY(atomic_poll_exchange_contention)(
    __gm__ atomic_poll_exchange_contention::ProbeStorage *storage, uint32_t mode_value, uint32_t num_blocks
) {
    using namespace atomic_poll_exchange_contention;
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    const uint32_t actual_blocks = static_cast<uint32_t>(get_block_num());
    const uint32_t blocks = actual_blocks == num_blocks ? num_blocks : actual_blocks;
    if (blocks < kMinAivs || blocks > kMaxAivs || mode_value >= DeviceModeIndex(ProbeMode::Count)) {
        if (block == 0U) {
            PublishHeader(&storage->header, static_cast<ProbeMode>(mode_value), blocks, 0, 0, 0, 0, 0, 0xBAD0AB1U);
        }
        return;
    }

    const ProbeMode mode = static_cast<ProbeMode>(mode_value);
    if (block == 0U) {
        WriterMain(storage, mode, blocks);
    } else {
        ReaderMain(storage, mode, block);
    }
}
