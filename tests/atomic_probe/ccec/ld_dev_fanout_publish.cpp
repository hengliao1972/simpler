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
// 测试目标：固定 block0 为唯一 writer，其余所有已启动 AIV 都是 reader；reader 从始至终只用
// raw ld_dev 读取同一个、由 data 独占的 64B cache line。分别精确比较三种 writer 发布序列：
//   mode 0：volatile ordinary scalar GM store -> DSB；不执行 DCCI；
//   mode 1：raw st_dev -> DSB；
//   mode 2：AtomicExch；不额外执行 DSB。
// 这里的 DSB 属于被测发布序列，ordinary mode 绝不补 DCCI，不能把 ordinary dirty line 暗中
// clean 到 GM 后再宣称 ld_dev 可见。
//
// 每轮完整时序：
//   1. 每个 reader 在 ready 独占 line 上 atomicAdd(+1)，随后用 ld_dev 等待 control epoch；
//   2. writer 用 ld_dev 等到所有 reader ready，通过 control 独占 line 上的 AtomicExch 发布 epoch；
//   3. writer 记录 write_start，执行本 mode 的单次写，分别记录“单条写指令周期”和“完整发布序列
//      周期”；所有 reader 同时持续用 ld_dev(data) 等待本轮唯一序列值；
//   4. reader 只有在精确看到本轮值或设备侧有限超时后，才在 ack 独占 line 上 atomicAdd(+1)；
//   5. writer 记录从 write_start 到全部 reader ack 的端到端周期，然后才进入下一轮。因此 writer
//      不会在正常路径上跳过某个序列值，reader 必须逐轮看到全部值，而不只是最终值。
//
// data、host 只写的 launch config、control、ready、ack、timing header/records、每个参与 AIV 的结果
// 和 tail guard 均按 64B 分区；data line 上只有 block0 写和其他 AIV 的 ld_dev 读。控制面使用
// atomic，但与 data 分 line；
// 全用例不执行任何 DCCI。所有轮询都有 get_sys_cnt() 有限超时，即使某种写法对 ld_dev 永不可见，
// kernel 也必须返回明确的 timeout、首错和 timing，而不是死循环。
#include "ccec_utils.h"

CCEC_PROBE_KERNEL_META(ld_dev_fanout_publish);

constexpr uint32_t CACHELINE_WORDS = 16;
constexpr uint32_t NUM_ROUNDS = 64;
constexpr uint32_t MAX_AIVS = 72;
// 36+ launched AIVs first hit a 2ms watchdog under no-backoff fanout pressure. Raising the
// boundary to 20ms did not restore progress: the long tail moved to the new boundary. Keep the
// larger finite bound so low-concurrency timing jitter is not mislabeled as a visibility failure.
constexpr uint32_t WAIT_TIMEOUT_CYCLES = 20000000;  // A5 1GHz sys counter: 20ms.

constexpr uint32_t DATA_BASE = 0 * CACHELINE_WORDS;
constexpr uint32_t CONFIG_BASE = 1 * CACHELINE_WORDS;
constexpr uint32_t CONTROL_BASE = 2 * CACHELINE_WORDS;
constexpr uint32_t READY_BASE = 3 * CACHELINE_WORDS;
constexpr uint32_t ACK_BASE = 4 * CACHELINE_WORDS;
constexpr uint32_t TIMING_HEADER_BASE = 5 * CACHELINE_WORDS;
constexpr uint32_t TIMING_RECORD_BASE = 6 * CACHELINE_WORDS;
constexpr uint32_t TIMING_RECORD_WORDS = 8;
constexpr uint32_t READER_RESULT_BASE = TIMING_RECORD_BASE + NUM_ROUNDS * TIMING_RECORD_WORDS;
constexpr uint32_t GUARD_BASE = READER_RESULT_BASE + MAX_AIVS * CACHELINE_WORDS;
static_assert((GUARD_BASE % CACHELINE_WORDS) == 0, "tail guard must start at a cache line");

constexpr uint32_t TIMING_MAGIC = 0x4c44464eu;       // "LDFN"
constexpr uint32_t TIMING_RECORD_MAGIC = 0x54490000u; // "TI" + round.
constexpr uint32_t PARTICIPANT_MAGIC = 0x4c445250u;  // "LDRP"
constexpr uint32_t FINISH_MAGIC = 0x46494e49u;       // "FINI"
constexpr uint32_t INVALID_U32 = 0xffffffffu;

enum class PublishMode : uint32_t {
    ORDINARY_STORE_DSB = 0,
    ST_DEV_DSB = 1,
    ATOMIC_EXCH = 2,
};

enum TimingHeaderIndex : uint32_t {
    HEADER_MAGIC = 0,
    HEADER_MODE = 1,
    HEADER_ROUNDS = 2,
    HEADER_BLOCKS = 3,
    HEADER_READERS = 4,
    HEADER_COMPLETED_ROUNDS = 5,
    HEADER_READY_TIMEOUTS = 6,
    HEADER_ACK_TIMEOUTS = 7,
    HEADER_FINAL_ACTUAL = 8,
    HEADER_FINAL_EXPECTED = 9,
    HEADER_LAUNCH_ID = 10,
    HEADER_WRITER_CORE = 11,
    HEADER_WRITER_SUBBLOCK = 12,
    HEADER_TIMEOUT_CYCLES = 13,
    HEADER_SEQUENCE_CODE = 14,
    HEADER_FINISH = 15,
};

enum ParticipantIndex : uint32_t {
    PARTICIPANT_HEADER = 0,
    PARTICIPANT_BLOCK = 1,
    PARTICIPANT_CORE = 2,
    PARTICIPANT_SUBBLOCK = 3,
    PARTICIPANT_BLOCK_NUM = 4,
    PARTICIPANT_SEEN = 5,
    PARTICIPANT_DATA_TIMEOUTS = 6,
    PARTICIPANT_CONTROL_TIMEOUTS = 7,
    PARTICIPANT_FIRST_BAD_ROUND = 8,
    PARTICIPANT_FIRST_ACTUAL = 9,
    PARTICIPANT_LAST_ACTUAL = 10,
    PARTICIPANT_TOTAL_POLLS = 11,
    PARTICIPANT_MAX_OBSERVE_CYCLES = 12,
    PARTICIPANT_LAUNCH_ID = 13,
    PARTICIPANT_MODE = 14,
    PARTICIPANT_FINISH = 15,
};

__aicore__ inline uint32_t SequenceValue(uint32_t mode, uint32_t launch_id, uint32_t round)
{
    return 0xa0000000u + (mode + 1u) * 0x01000000u +
        (launch_id & 0xfffu) * 0x1000u + round + 1u;
}

__aicore__ inline uint32_t Elapsed32(uint64_t begin, uint64_t end)
{
    uint64_t delta = end - begin;
    return delta > static_cast<uint64_t>(INVALID_U32) ? INVALID_U32 : static_cast<uint32_t>(delta);
}

__aicore__ inline bool WaitAtLeast(
    __gm__ uint32_t *address, uint32_t target, uint32_t timeout_cycles, uint32_t &actual)
{
    uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    do {
        actual = ld_dev_b32(address);
        if (actual >= target) {
            return true;
        }
    } while (static_cast<uint64_t>(get_sys_cnt()) - begin < timeout_cycles);
    actual = ld_dev_b32(address);
    return actual >= target;
}

__aicore__ inline void PublishWord(__gm__ uint32_t *address, uint32_t value)
{
    (void)atomicExch(address, value);
}

__aicore__ inline void PublishParticipant(
    __gm__ uint32_t *storage, uint32_t block, uint32_t block_num, uint32_t launch_id,
    uint32_t mode, uint32_t seen, uint32_t data_timeouts, uint32_t control_timeouts,
    uint32_t first_bad_round, uint32_t first_actual, uint32_t last_actual,
    uint32_t total_polls, uint32_t max_observe_cycles)
{
    __gm__ uint32_t *result = storage + READER_RESULT_BASE + block * CACHELINE_WORDS;
    PublishWord(&result[PARTICIPANT_HEADER], PARTICIPANT_MAGIC);
    PublishWord(&result[PARTICIPANT_BLOCK], block);
    PublishWord(&result[PARTICIPANT_CORE], static_cast<uint32_t>(get_coreid()));
    PublishWord(&result[PARTICIPANT_SUBBLOCK], static_cast<uint32_t>(get_subblockid()));
    PublishWord(&result[PARTICIPANT_BLOCK_NUM], block_num);
    PublishWord(&result[PARTICIPANT_SEEN], seen);
    PublishWord(&result[PARTICIPANT_DATA_TIMEOUTS], data_timeouts);
    PublishWord(&result[PARTICIPANT_CONTROL_TIMEOUTS], control_timeouts);
    PublishWord(&result[PARTICIPANT_FIRST_BAD_ROUND], first_bad_round);
    PublishWord(&result[PARTICIPANT_FIRST_ACTUAL], first_actual);
    PublishWord(&result[PARTICIPANT_LAST_ACTUAL], last_actual);
    PublishWord(&result[PARTICIPANT_TOTAL_POLLS], total_polls);
    PublishWord(&result[PARTICIPANT_MAX_OBSERVE_CYCLES], max_observe_cycles);
    PublishWord(&result[PARTICIPANT_LAUNCH_ID], launch_id);
    PublishWord(&result[PARTICIPANT_MODE], mode);
    PublishWord(&result[PARTICIPANT_FINISH], FINISH_MAGIC);
}

__aicore__ inline void WriterMain(
    __gm__ uint32_t *storage, uint32_t mode, uint32_t block_num, uint32_t launch_id)
{
    __gm__ uint32_t *data = storage + DATA_BASE;
    __gm__ uint32_t *control = storage + CONTROL_BASE;
    __gm__ uint32_t *ready = storage + READY_BASE;
    __gm__ uint32_t *ack = storage + ACK_BASE;
    volatile __gm__ uint32_t *ordinary_data = data;
    uint32_t readers = block_num - 1u;
    uint32_t ready_timeouts = 0;
    uint32_t ack_timeouts = 0;
    uint32_t completed_rounds = 0;

    for (uint32_t round = 0; round < NUM_ROUNDS; ++round) {
        uint32_t ready_target = readers * (round + 1u);
        uint32_t ready_actual = 0;
        bool ready_ok = WaitAtLeast(ready, ready_target, WAIT_TIMEOUT_CYCLES, ready_actual);
        if (!ready_ok) {
            ++ready_timeouts;
        }

        // Readers start data polling only after this independent atomic epoch publication.
        (void)atomicExch(&control[0], round + 1u);

        uint32_t expected = SequenceValue(mode, launch_id, round);
        uint64_t write_begin = static_cast<uint64_t>(get_sys_cnt());
        uint64_t instruction_end = write_begin;
        uint64_t publish_end = write_begin;
        if (mode == static_cast<uint32_t>(PublishMode::ORDINARY_STORE_DSB)) {
            ordinary_data[0] = expected;
            instruction_end = static_cast<uint64_t>(get_sys_cnt());
            dsb(DSB_ALL);
            publish_end = static_cast<uint64_t>(get_sys_cnt());
        } else if (mode == static_cast<uint32_t>(PublishMode::ST_DEV_DSB)) {
            st_dev_b32(data, expected);
            instruction_end = static_cast<uint64_t>(get_sys_cnt());
            dsb(DSB_ALL);
            publish_end = static_cast<uint64_t>(get_sys_cnt());
        } else {
            (void)atomicExch(data, expected);
            instruction_end = static_cast<uint64_t>(get_sys_cnt());
            publish_end = instruction_end;
        }

        uint32_t ack_target = readers * (round + 1u);
        uint32_t ack_actual = 0;
        bool ack_ok = WaitAtLeast(ack, ack_target, WAIT_TIMEOUT_CYCLES * 4u, ack_actual);
        uint64_t all_ack = static_cast<uint64_t>(get_sys_cnt());
        if (!ack_ok) {
            ++ack_timeouts;
        }

        __gm__ uint32_t *record = storage + TIMING_RECORD_BASE + round * TIMING_RECORD_WORDS;
        PublishWord(&record[0], Elapsed32(write_begin, instruction_end));
        PublishWord(&record[1], Elapsed32(write_begin, publish_end));
        PublishWord(&record[2], Elapsed32(write_begin, all_ack));
        PublishWord(&record[3], ack_actual);
        PublishWord(&record[4], (ready_ok ? 0u : 1u) | (ack_ok ? 0u : 2u));
        PublishWord(&record[5], ld_dev_b32(data));
        PublishWord(&record[6], expected);
        PublishWord(&record[7], TIMING_RECORD_MAGIC | round);
        ++completed_rounds;
    }

    uint32_t final_expected = SequenceValue(mode, launch_id, NUM_ROUNDS - 1u);
    __gm__ uint32_t *header = storage + TIMING_HEADER_BASE;
    PublishWord(&header[HEADER_MAGIC], TIMING_MAGIC);
    PublishWord(&header[HEADER_MODE], mode);
    PublishWord(&header[HEADER_ROUNDS], NUM_ROUNDS);
    PublishWord(&header[HEADER_BLOCKS], block_num);
    PublishWord(&header[HEADER_READERS], readers);
    PublishWord(&header[HEADER_COMPLETED_ROUNDS], completed_rounds);
    PublishWord(&header[HEADER_READY_TIMEOUTS], ready_timeouts);
    PublishWord(&header[HEADER_ACK_TIMEOUTS], ack_timeouts);
    PublishWord(&header[HEADER_FINAL_ACTUAL], ld_dev_b32(data));
    PublishWord(&header[HEADER_FINAL_EXPECTED], final_expected);
    PublishWord(&header[HEADER_LAUNCH_ID], launch_id);
    PublishWord(&header[HEADER_WRITER_CORE], static_cast<uint32_t>(get_coreid()));
    PublishWord(&header[HEADER_WRITER_SUBBLOCK], static_cast<uint32_t>(get_subblockid()));
    PublishWord(&header[HEADER_TIMEOUT_CYCLES], WAIT_TIMEOUT_CYCLES);
    // 0x01=ordinary+DSB, 0x02=st_dev+DSB, 0x03=AtomicExch without explicit DSB.
    PublishWord(&header[HEADER_SEQUENCE_CODE], mode + 1u);
    PublishWord(&header[HEADER_FINISH], FINISH_MAGIC);
    uint32_t writer_local_normal =
        mode == static_cast<uint32_t>(PublishMode::ORDINARY_STORE_DSB) ? ordinary_data[0] : 0u;
    PublishParticipant(storage, 0u, block_num, launch_id, mode, 0u, 0u, 0u,
                       INVALID_U32, 0u, writer_local_normal, 0u, 0u);
    PublishWord(&control[1], FINISH_MAGIC);
}

__aicore__ inline void ReaderMain(
    __gm__ uint32_t *storage, uint32_t mode, uint32_t block_num, uint32_t launch_id,
    uint32_t block)
{
    __gm__ uint32_t *data = storage + DATA_BASE;
    __gm__ uint32_t *control = storage + CONTROL_BASE;
    __gm__ uint32_t *ready = storage + READY_BASE;
    __gm__ uint32_t *ack = storage + ACK_BASE;
    uint32_t seen = 0;
    uint32_t data_timeouts = 0;
    uint32_t control_timeouts = 0;
    uint32_t first_bad_round = INVALID_U32;
    uint32_t first_actual = 0;
    uint32_t last_actual = 0;
    uint32_t total_polls = 0;
    uint32_t max_observe_cycles = 0;

    for (uint32_t round = 0; round < NUM_ROUNDS; ++round) {
        (void)atomicAdd(ready, 1u);
        uint32_t epoch_actual = 0;
        bool epoch_ok = WaitAtLeast(control, round + 1u, WAIT_TIMEOUT_CYCLES, epoch_actual);
        if (!epoch_ok) {
            ++control_timeouts;
        }

        uint32_t expected = SequenceValue(mode, launch_id, round);
        uint64_t observe_begin = static_cast<uint64_t>(get_sys_cnt());
        bool exact = false;
        uint32_t actual = 0;
        do {
            actual = ld_dev_b32(data);
            ++total_polls;
            if (actual == expected) {
                exact = true;
                break;
            }
        } while (static_cast<uint64_t>(get_sys_cnt()) - observe_begin < WAIT_TIMEOUT_CYCLES);
        uint64_t observe_end = static_cast<uint64_t>(get_sys_cnt());
        uint32_t observe_cycles = Elapsed32(observe_begin, observe_end);
        if (observe_cycles > max_observe_cycles) {
            max_observe_cycles = observe_cycles;
        }
        last_actual = actual;
        if (exact) {
            ++seen;
        } else {
            ++data_timeouts;
            if (first_bad_round == INVALID_U32) {
                first_bad_round = round;
                first_actual = actual;
            }
        }
        // Always ack, including timeout, so a failing visibility mode cannot deadlock the writer.
        (void)atomicAdd(ack, 1u);
    }

    PublishParticipant(storage, block, block_num, launch_id, mode, seen, data_timeouts,
                       control_timeouts, first_bad_round, first_actual, last_actual,
                       total_polls, max_observe_cycles);
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(ld_dev_fanout_publish)(
    __gm__ uint32_t *storage, uint32_t mode, uint32_t num_blocks)
{
    uint32_t block = static_cast<uint32_t>(get_block_idx());
    uint32_t actual_blocks = static_cast<uint32_t>(get_block_num());
    uint32_t launch_id = ld_dev_b32(&storage[CONFIG_BASE]);
    // Host constrains this to 2..MAX_AIVS. Preserve a finite, self-describing failure if ABI args differ.
    uint32_t block_num = num_blocks == actual_blocks ? num_blocks : actual_blocks;
    if (block_num < 2u || block_num > MAX_AIVS || mode > 2u) {
        if (block == 0u) {
            __gm__ uint32_t *header = storage + TIMING_HEADER_BASE;
            PublishWord(&header[HEADER_MAGIC], TIMING_MAGIC);
            PublishWord(&header[HEADER_MODE], mode);
            PublishWord(&header[HEADER_BLOCKS], block_num);
            PublishWord(&header[HEADER_FINISH], 0xbad0ab1u);
        }
        return;
    }
    if (block == 0u) {
        WriterMain(storage, mode, block_num, launch_id);
    } else {
        ReaderMain(storage, mode, block_num, launch_id, block);
    }
}
