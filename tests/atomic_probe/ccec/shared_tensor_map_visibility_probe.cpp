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
// A5 production shared TensorMap 跨核可见性探针。
//
// 两个 AIV 对每个 task 轮换 writer/reader。reader 先用普通 scalar load
// 预热旧 committed/reclaim/head/tail/seq/payload cache line，再通过独立诊断
// cache line 告知 writer。writer 只调用 production publish helper；reader
// 只以 production committed_tasks 从 N 变为 N+1 作为交权信号，随后调用
// production read/lookup helper 验证 tail、seq、payload 与 lookup。
//
// 被测窗口不插入 FFTS SyncAll、额外 DSB、额外 DCCI 或自造发布 atomic。
// ready/done 仅位于窗口外，负责阻止 writer 在 reader 完成本 task 取证前进入
// 下一 task。所有轮询都有设备侧有限超时，首错只记录一次。
#include "ccec_utils.h"
#include "inner_kernel.h"
#include "dist_engine/aicore/shared_tensor_map.h"
#include "shared_tensor_map_visibility_probe_shared.h"

using namespace shared_tensor_map_visibility_probe;

static_assert(PTO_FDWIC_SHARED_MAP == 1, "visibility probe must compile the production shared backend");
static_assert(PTO_FDWIC_TENSORMAP_RING_CAP == kRingCapacity, "visibility probe requires CAP=128");
static_assert(kMapBucketCapacity == kRingCapacity);
static_assert(kMapBuckets == 128);

CCEC_PROBE_KERNEL_META(shared_tensor_map_visibility_probe);

enum class WaitResult : uint32_t {
    Reached = 0,
    Aborted = 1,
    Timeout = 2,
    Overshoot = 3,
};

struct TaskPlan {
    SharedTensorMapValue entries[2];
    uint32_t count;
};

__aicore__ inline SharedTensorMapValue MakeEntry(uint64_t buffer, int32_t task, uint32_t ordinal) {
    const uint64_t lower = (static_cast<uint64_t>(static_cast<uint32_t>(task)) * 4ULL + ordinal) * 64ULL;
    SharedTensorMapValue value{};
    value.buf_addr = buffer;
    value.lo = lower;
    value.hi = lower + 32;
    value.producer = task;
    value.reserved = 0;
    return value;
}

__aicore__ inline TaskPlan MakeTaskPlan(int32_t task) {
    TaskPlan plan{};
    if (task == 0) {
        plan.count = 0;
    } else if (task == 1) {
        plan.entries[0] = MakeEntry(kBufferA0, task, 0);
        plan.entries[1] = MakeEntry(kBufferA1, task, 1);
        plan.count = 2;
    } else {
        plan.entries[0] = MakeEntry(kBufferA0, task, 0);
        plan.entries[1] = MakeEntry(kBufferB, task, 1);
        plan.count = 2;
    }
    return plan;
}

__aicore__ inline int64_t LoadControl(__gm__ volatile int64_t *address) {
    return DistSharedTensorMapAicoreOps::Load(address);
}

__aicore__ inline void StoreDiagnostic(__gm__ volatile int64_t *address, int64_t value) {
    (void)atomic_exchange(*address, value);
}

__aicore__ inline void RecordFirstError(
    __gm__ ProbeControl &control, ErrorCode code, Phase phase, int32_t task, uint32_t block, int64_t actual,
    int64_t expected, int64_t aux0, int64_t aux1, int64_t commit, int64_t head, int64_t tail, int64_t sequence,
    const SharedTensorMapValue *snapshot
) {
    if (DistSharedTensorMapAicoreOps::CompareExchange(&control.first_error_claim.value, 0, 1) != 0) {
        return;
    }

    StoreDiagnostic(&control.first_error.words[0], static_cast<int64_t>(code));
    StoreDiagnostic(&control.first_error.words[1], static_cast<int64_t>(task));
    StoreDiagnostic(&control.first_error.words[2], static_cast<int64_t>(block));
    StoreDiagnostic(&control.first_error.words[3], static_cast<int64_t>(phase));
    StoreDiagnostic(&control.first_error.words[4], actual);
    StoreDiagnostic(&control.first_error.words[5], expected);
    StoreDiagnostic(&control.first_error.words[6], aux0);
    StoreDiagnostic(&control.first_error.words[7], aux1);

    StoreDiagnostic(&control.first_snapshot.words[0], commit);
    StoreDiagnostic(&control.first_snapshot.words[1], head);
    StoreDiagnostic(&control.first_snapshot.words[2], tail);
    StoreDiagnostic(&control.first_snapshot.words[3], sequence);
    StoreDiagnostic(
        &control.first_snapshot.words[4], snapshot == nullptr ? 0 : static_cast<int64_t>(snapshot->buf_addr)
    );
    StoreDiagnostic(&control.first_snapshot.words[5], snapshot == nullptr ? 0 : static_cast<int64_t>(snapshot->lo));
    StoreDiagnostic(&control.first_snapshot.words[6], snapshot == nullptr ? 0 : static_cast<int64_t>(snapshot->hi));
    const int64_t producer_reserved = snapshot == nullptr ?
                                          0 :
                                          (static_cast<int64_t>(static_cast<uint32_t>(snapshot->producer)) << 32) |
                                              static_cast<int64_t>(snapshot->reserved);
    StoreDiagnostic(&control.first_snapshot.words[7], producer_reserved);

    (void)DistSharedTensorMapAicoreOps::CompareExchange(&control.abort_code.value, 0, static_cast<int64_t>(code));
}

__aicore__ inline WaitResult
WaitForExact(__gm__ ProbeControl &control, __gm__ volatile int64_t *address, int64_t target, int64_t &actual) {
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    do {
        actual = LoadControl(address);
        if (actual == target) {
            return WaitResult::Reached;
        }
        if (actual > target) {
            return WaitResult::Overshoot;
        }
        if (LoadControl(&control.abort_code.value) != 0) {
            return WaitResult::Aborted;
        }
    } while (static_cast<uint64_t>(get_sys_cnt()) - begin < kWaitTimeoutCycles);

    actual = LoadControl(address);
    if (actual == target) {
        return WaitResult::Reached;
    }
    return actual > target ? WaitResult::Overshoot : WaitResult::Timeout;
}

__aicore__ inline bool ValuesEqual(const SharedTensorMapValue &left, const SharedTensorMapValue &right) {
    return left.buf_addr == right.buf_addr && left.lo == right.lo && left.hi == right.hi &&
           left.producer == right.producer && left.reserved == right.reserved;
}

__aicore__ inline uint64_t
PreheatTask(__gm__ SharedTensorMapState &map, const TaskPlan &plan, uint64_t tail_a, uint64_t tail_b, uint64_t sink) {
    // 这些必须是普通 scalar load，不能改成 ld_dev/atomic；目的正是让 reader
    // 在 writer 发布前持有旧 cache line。
    volatile __gm__ int64_t *commit = &map.committed_tasks.v;
    volatile __gm__ int64_t *reclaim = &map.reclaim_upto.v;
    volatile __gm__ int64_t *head_a = &map.buckets[kExpectedBucketA].head.v;
    volatile __gm__ int64_t *tail_a_ptr = &map.buckets[kExpectedBucketA].tail.v;
    volatile __gm__ int64_t *head_b = &map.buckets[kExpectedBucketB].head.v;
    volatile __gm__ int64_t *tail_b_ptr = &map.buckets[kExpectedBucketB].tail.v;
    sink += static_cast<uint64_t>(*commit) + static_cast<uint64_t>(*reclaim);
    sink += static_cast<uint64_t>(*head_a) + static_cast<uint64_t>(*tail_a_ptr);
    sink += static_cast<uint64_t>(*head_b) + static_cast<uint64_t>(*tail_b_ptr);

    for (uint32_t index = 0; index < plan.count; ++index) {
        const uint32_t bucket = dist_tensor_map_hash(plan.entries[index].buf_addr);
        uint64_t cursor = bucket == kExpectedBucketA ? tail_a : tail_b;
        if (index != 0 && bucket == dist_tensor_map_hash(plan.entries[0].buf_addr)) {
            ++cursor;
        }
        __gm__ SharedTensorMapSlot &slot = map.slots[dist_shared_tensor_map_slot_index(bucket, cursor)];
        volatile __gm__ int64_t *sequence = &slot.sequence.v;
        volatile __gm__ uint64_t *payload = reinterpret_cast<volatile __gm__ uint64_t *>(&slot.payload);
        sink += static_cast<uint64_t>(*sequence);
        for (uint32_t word = 0; word < sizeof(slot.payload) / sizeof(uint64_t); ++word) {
            sink += payload[word] * (word + 1U);
        }
    }
    return sink;
}

__aicore__ inline bool CheckErrorFree(
    __gm__ ProbeControl &control, ErrorCode code, Phase phase, int32_t task, uint32_t block, int64_t actual,
    int64_t expected, int64_t aux0, int64_t aux1, int64_t commit, int64_t head, int64_t tail, int64_t sequence,
    const SharedTensorMapValue *snapshot
) {
    if (actual == expected) {
        return true;
    }
    RecordFirstError(
        control, code, phase, task, block, actual, expected, aux0, aux1, commit, head, tail, sequence, snapshot
    );
    return false;
}

__aicore__ inline bool CheckPublishedEntry(
    __gm__ SharedTensorMapState &map, __gm__ ProbeControl &control, const SharedTensorMapValue &expected,
    uint64_t cursor, int32_t task, uint32_t block, int64_t expected_head, int64_t expected_tail, uint64_t &checks
) {
    const uint32_t bucket = dist_tensor_map_hash(expected.buf_addr);
    const int64_t commit = DistSharedTensorMapAicoreOps::Load(&map.committed_tasks.v);
    const int64_t head = DistSharedTensorMapAicoreOps::Load(&map.buckets[bucket].head.v);
    const int64_t tail = DistSharedTensorMapAicoreOps::Load(&map.buckets[bucket].tail.v);
    ++checks;
    if (!CheckErrorFree(
            control, ErrorCode::HeadMismatch, Phase::CheckControl, task, block, head, expected_head, bucket,
            static_cast<int64_t>(cursor), commit, head, tail, 0, nullptr
        )) {
        return false;
    }
    ++checks;
    if (!CheckErrorFree(
            control, ErrorCode::TailMismatch, Phase::CheckControl, task, block, tail, expected_tail, bucket,
            static_cast<int64_t>(cursor), commit, head, tail, 0, nullptr
        )) {
        return false;
    }

    __gm__ SharedTensorMapSlot &slot = map.slots[dist_shared_tensor_map_slot_index(bucket, cursor)];
    const int64_t sequence = DistSharedTensorMapAicoreOps::Load(&slot.sequence.v);
    ++checks;
    if (!CheckErrorFree(
            control, ErrorCode::SequenceMismatch, Phase::CheckSlot, task, block, sequence, static_cast<int64_t>(cursor),
            bucket, static_cast<int64_t>(cursor), commit, head, tail, sequence, nullptr
        )) {
        return false;
    }

    SharedTensorMapValue snapshot{};
    const bool read_ok =
        dist_shared_tensor_map_read_slot_impl<DistSharedTensorMapAicoreOps>(map, bucket, cursor, snapshot);
    ++checks;
    if (!read_ok) {
        RecordFirstError(
            control, ErrorCode::ReadFailed, Phase::CheckSlot, task, block, 0, 1, bucket, static_cast<int64_t>(cursor),
            commit, head, tail, sequence, &snapshot
        );
        return false;
    }
    ++checks;
    if (!ValuesEqual(snapshot, expected)) {
        RecordFirstError(
            control, ErrorCode::PayloadMismatch, Phase::CheckSlot, task, block, snapshot.producer, expected.producer,
            bucket, static_cast<int64_t>(cursor), commit, head, tail, sequence, &snapshot
        );
        return false;
    }

    bool protocol_ok = false;
    const int32_t producer = dist_shared_tensor_map_lookup_region(map, expected, task + 1, 1, protocol_ok);
    ++checks;
    if (!protocol_ok) {
        RecordFirstError(
            control, ErrorCode::LookupProtocol, Phase::CheckLookup, task, block, producer, expected.producer, bucket,
            static_cast<int64_t>(cursor), commit, head, tail, sequence, &snapshot
        );
        return false;
    }
    ++checks;
    if (producer != expected.producer) {
        RecordFirstError(
            control, ErrorCode::LookupMismatch, Phase::CheckLookup, task, block, producer, expected.producer, bucket,
            static_cast<int64_t>(cursor), commit, head, tail, sequence, &snapshot
        );
        return false;
    }
    return true;
}

__aicore__ inline void PublishParticipant(
    __gm__ ProbeControl &control, uint32_t block, uint32_t completed_tasks, uint32_t writer_tasks,
    uint32_t reader_tasks, uint64_t checks, uint64_t preheat_sink
) {
    __gm__ ParticipantLine &line = control.participants[block];
    StoreDiagnostic(&line.words[0], kParticipantMagic);
    StoreDiagnostic(&line.words[1], static_cast<int64_t>(block));
    const uint64_t topology =
        (static_cast<uint64_t>(static_cast<uint32_t>(get_coreid())) << 32) | static_cast<uint32_t>(get_subblockid());
    StoreDiagnostic(&line.words[2], static_cast<int64_t>(topology));
    StoreDiagnostic(&line.words[3], static_cast<int64_t>(completed_tasks));
    const uint64_t role_counts = (static_cast<uint64_t>(writer_tasks) << 32) | reader_tasks;
    StoreDiagnostic(&line.words[4], static_cast<int64_t>(role_counts));
    StoreDiagnostic(&line.words[5], static_cast<int64_t>(checks));
    StoreDiagnostic(&line.words[6], static_cast<int64_t>(preheat_sink));
    StoreDiagnostic(&line.words[7], kParticipantFinish);
    (void)atomic_fetch_add(control.finish_count.value, int64_t{1});
}

extern "C" __global__ __aicore__ void KERNEL_ENTRY(shared_tensor_map_visibility_probe)(
    __gm__ SharedTensorMapState *map_pointer, __gm__ ProbeControl *control_pointer, uint32_t num_blocks,
    uint32_t launch_id
) {
    (void)launch_id;
    __gm__ SharedTensorMapState &map = *map_pointer;
    __gm__ ProbeControl &control = *control_pointer;
    const uint32_t block = static_cast<uint32_t>(get_block_idx());
    uint32_t completed_tasks = 0;
    uint32_t writer_tasks = 0;
    uint32_t reader_tasks = 0;
    uint64_t checks = 0;
    uint64_t preheat_sink = 0;

    const uint32_t bucket_a0 = dist_tensor_map_hash(kBufferA0);
    const uint32_t bucket_a1 = dist_tensor_map_hash(kBufferA1);
    const uint32_t bucket_b = dist_tensor_map_hash(kBufferB);
    if (num_blocks != kAivBlocks || get_block_num() != kAivBlocks || block >= kAivBlocks) {
        RecordFirstError(
            control, ErrorCode::InvalidTopology, Phase::Setup, -1, block, get_block_num(), kAivBlocks, num_blocks, 0, 0,
            0, 0, 0, nullptr
        );
        PublishParticipant(
            control, block < kAivBlocks ? block : 0, completed_tasks, writer_tasks, reader_tasks, checks, preheat_sink
        );
        return;
    }
    if (bucket_a0 != kExpectedBucketA || bucket_a1 != kExpectedBucketA || bucket_b != kExpectedBucketB ||
        bucket_a0 == bucket_b) {
        RecordFirstError(
            control, ErrorCode::HashConfiguration, Phase::Setup, -1, block, bucket_a0, kExpectedBucketA, bucket_a1,
            bucket_b, 0, 0, 0, 0, nullptr
        );
        PublishParticipant(control, block, completed_tasks, writer_tasks, reader_tasks, checks, preheat_sink);
        return;
    }

    uint64_t tail_a = 0;
    uint64_t tail_b = 0;
    for (int32_t task = 0; task < static_cast<int32_t>(kTotalTasks); ++task) {
        if (LoadControl(&control.abort_code.value) != 0) {
            break;
        }
        const TaskPlan plan = MakeTaskPlan(task);
        const uint32_t writer = static_cast<uint32_t>(task) & 1U;
        const bool is_writer = block == writer;
        const int64_t epoch = static_cast<int64_t>(task) + 1;

        if (!is_writer) {
            preheat_sink = PreheatTask(map, plan, tail_a, tail_b, preheat_sink);
            const int64_t old_ready =
                DistSharedTensorMapAicoreOps::CompareExchange(&control.reader_ready.value, task, epoch);
            if (!CheckErrorFree(
                    control, ErrorCode::ReadyOvershoot, Phase::WaitReady, task, block, old_ready, task, 0, 0,
                    LoadControl(&map.committed_tasks.v), 0, 0, 0, nullptr
                )) {
                break;
            }
        } else {
            int64_t actual = 0;
            const WaitResult wait = WaitForExact(control, &control.reader_ready.value, epoch, actual);
            if (wait != WaitResult::Reached) {
                if (wait != WaitResult::Aborted) {
                    RecordFirstError(
                        control, wait == WaitResult::Timeout ? ErrorCode::ReadyTimeout : ErrorCode::ReadyOvershoot,
                        Phase::WaitReady, task, block, actual, epoch, 0, 0, LoadControl(&map.committed_tasks.v), 0, 0,
                        0, nullptr
                    );
                }
                break;
            }

            const DistSharedTensorMapTaskPublishResult result =
                dist_shared_tensor_map_publish_task(map, plan.count == 0 ? nullptr : plan.entries, plan.count, task, 0);
            ++writer_tasks;
            if (result != DistSharedTensorMapTaskPublishResult::Committed) {
                RecordFirstError(
                    control, ErrorCode::PublishFailed, Phase::Publish, task, block, static_cast<int64_t>(result),
                    static_cast<int64_t>(DistSharedTensorMapTaskPublishResult::Committed), plan.count, 0,
                    LoadControl(&map.committed_tasks.v), 0, 0, 0, nullptr
                );
                break;
            }
        }

        if (!is_writer) {
            int64_t commit = 0;
            const WaitResult wait = WaitForExact(control, &map.committed_tasks.v, epoch, commit);
            if (wait != WaitResult::Reached) {
                if (wait != WaitResult::Aborted) {
                    RecordFirstError(
                        control, wait == WaitResult::Timeout ? ErrorCode::CommitTimeout : ErrorCode::CommitOvershoot,
                        Phase::WaitCommit, task, block, commit, epoch, 0, 0, commit, 0, 0, 0, nullptr
                    );
                }
                break;
            }
            ++reader_tasks;
            ++checks;
            if (!CheckErrorFree(
                    control, ErrorCode::CommitMismatch, Phase::WaitCommit, task, block, commit, epoch, 0, 0, commit, 0,
                    0, 0, nullptr
                )) {
                break;
            }

            const int64_t expected_reclaim = task == 0 ? -1 : task - 1;
            const int64_t reclaim = DistSharedTensorMapAicoreOps::Load(&map.reclaim_upto.v);
            ++checks;
            if (!CheckErrorFree(
                    control, ErrorCode::ReclaimMismatch, Phase::CheckControl, task, block, reclaim, expected_reclaim, 0,
                    0, commit, 0, 0, 0, nullptr
                )) {
                break;
            }

            bool task_ok = true;
            if (task == 0) {
                const int64_t head_a = DistSharedTensorMapAicoreOps::Load(&map.buckets[kExpectedBucketA].head.v);
                const int64_t observed_tail_a =
                    DistSharedTensorMapAicoreOps::Load(&map.buckets[kExpectedBucketA].tail.v);
                const int64_t head_b = DistSharedTensorMapAicoreOps::Load(&map.buckets[kExpectedBucketB].head.v);
                const int64_t observed_tail_b =
                    DistSharedTensorMapAicoreOps::Load(&map.buckets[kExpectedBucketB].tail.v);
                checks += 4;
                task_ok = CheckErrorFree(
                              control, ErrorCode::HeadMismatch, Phase::CheckControl, task, block, head_a, 0,
                              kExpectedBucketA, 0, commit, head_a, observed_tail_a, 0, nullptr
                          ) &&
                          CheckErrorFree(
                              control, ErrorCode::TailMismatch, Phase::CheckControl, task, block, observed_tail_a, 0,
                              kExpectedBucketA, 0, commit, head_a, observed_tail_a, 0, nullptr
                          ) &&
                          CheckErrorFree(
                              control, ErrorCode::HeadMismatch, Phase::CheckControl, task, block, head_b, 0,
                              kExpectedBucketB, 0, commit, head_b, observed_tail_b, 0, nullptr
                          ) &&
                          CheckErrorFree(
                              control, ErrorCode::TailMismatch, Phase::CheckControl, task, block, observed_tail_b, 0,
                              kExpectedBucketB, 0, commit, head_b, observed_tail_b, 0, nullptr
                          );

                bool lookup_ok = false;
                const SharedTensorMapValue empty_query = MakeEntry(kBufferA0, 0, 0);
                const int32_t producer = dist_shared_tensor_map_lookup_region(map, empty_query, 1, 1, lookup_ok);
                checks += 2;
                task_ok = task_ok && lookup_ok && producer == -1;
                if (!lookup_ok || producer != -1) {
                    RecordFirstError(
                        control, lookup_ok ? ErrorCode::LookupMismatch : ErrorCode::LookupProtocol, Phase::CheckLookup,
                        task, block, producer, -1, kExpectedBucketA, 0, commit, head_a, observed_tail_a, 0, nullptr
                    );
                }
            } else {
                const uint64_t old_tail_a = tail_a;
                const uint64_t old_tail_b = tail_b;
                const uint32_t a_count = task == 1 ? 2U : 1U;
                const uint32_t b_count = task >= 2 ? 1U : 0U;
                const int64_t expected_head_a = task == 1 ? 0 : static_cast<int64_t>(old_tail_a);
                const int64_t expected_head_b = task == 2 ? 0 : static_cast<int64_t>(old_tail_b);
                const int64_t expected_tail_a = static_cast<int64_t>(old_tail_a + a_count);
                const int64_t expected_tail_b = static_cast<int64_t>(old_tail_b + b_count);
                for (uint32_t index = 0; index < plan.count; ++index) {
                    const uint32_t bucket = dist_tensor_map_hash(plan.entries[index].buf_addr);
                    uint64_t cursor = bucket == kExpectedBucketA ? old_tail_a : old_tail_b;
                    if (index != 0 && bucket == dist_tensor_map_hash(plan.entries[0].buf_addr)) {
                        ++cursor;
                    }
                    task_ok = task_ok && CheckPublishedEntry(
                                             map, control, plan.entries[index], cursor, task, block,
                                             bucket == kExpectedBucketA ? expected_head_a : expected_head_b,
                                             bucket == kExpectedBucketA ? expected_tail_a : expected_tail_b, checks
                                         );
                    if (!task_ok) {
                        break;
                    }
                }
            }
            if (!task_ok) {
                break;
            }

            const int64_t old_done =
                DistSharedTensorMapAicoreOps::CompareExchange(&control.reader_done.value, task, epoch);
            if (!CheckErrorFree(
                    control, ErrorCode::DoneOvershoot, Phase::WaitDone, task, block, old_done, task, 0, 0, commit, 0, 0,
                    0, nullptr
                )) {
                break;
            }
        } else {
            int64_t actual = 0;
            const WaitResult wait = WaitForExact(control, &control.reader_done.value, epoch, actual);
            if (wait != WaitResult::Reached) {
                if (wait != WaitResult::Aborted) {
                    RecordFirstError(
                        control, wait == WaitResult::Timeout ? ErrorCode::DoneTimeout : ErrorCode::DoneOvershoot,
                        Phase::WaitDone, task, block, actual, epoch, 0, 0, LoadControl(&map.committed_tasks.v), 0, 0, 0,
                        nullptr
                    );
                }
                break;
            }
        }

        if (task == 1) {
            tail_a += 2;
        } else if (task >= 2) {
            ++tail_a;
            ++tail_b;
        }
        ++completed_tasks;
    }

    PublishParticipant(control, block, completed_tasks, writer_tasks, reader_tasks, checks, preheat_sink);
}
