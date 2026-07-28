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

#include "pa_scheduler_core.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <new>
#include <sys/mman.h>
#include <thread>

namespace {

using namespace pa_scheduler;

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] shared writer intent: %s\n", message);
    ++g_failures;
}

// 只验证公共 writer-intent 状态机。CPU acquire/release 保证宿主线程测试
// 没有普通 data race，但不模拟 A5 DCache；CCEC 编译和后续 A5 litmus
// 分别承担设备接口与跨核可见性证据。
struct WriterIntentTestOps {
    static constexpr bool kAtomicReturnReadyObserved = false;
    static volatile int64_t *wait_address;
    static std::atomic<uint64_t> wait_loads;
    static volatile int64_t *cas_trigger_address;
    static volatile int64_t *cas_conflict_address;
    static int64_t cas_conflict_value;

    static int32_t Load(volatile int32_t *address) {
        return __atomic_fetch_add(
            address, static_cast<int32_t>(0), __ATOMIC_ACQUIRE
        );
    }

    static int64_t Load(volatile int64_t *address) {
        if (address == wait_address) {
            wait_loads.fetch_add(1, std::memory_order_release);
        }
        return __atomic_fetch_add(
            address, static_cast<int64_t>(0), __ATOMIC_ACQUIRE
        );
    }

    static int32_t Exchange(
        volatile int32_t *address, int32_t value
    ) {
        return __atomic_exchange_n(
            address, value, __ATOMIC_ACQ_REL
        );
    }

    static int64_t Exchange(
        volatile int64_t *address, int64_t value
    ) {
        return __atomic_exchange_n(
            address, value, __ATOMIC_ACQ_REL
        );
    }

    static uint64_t Exchange(
        volatile uint64_t *address, uint64_t value
    ) {
        return __atomic_exchange_n(
            address, value, __ATOMIC_ACQ_REL
        );
    }

    static int64_t CompareExchange(
        volatile int64_t *address, int64_t expected,
        int64_t desired
    ) {
        // 只供多 symbol terminal-prefix 门槛：在第一项真正 CAS 前，
        // 原子推进第二项，稳定制造“第一项成功、第二项冲突”的时序。
        if (address == cas_trigger_address &&
            cas_conflict_address != nullptr) {
            __atomic_store_n(
                cas_conflict_address, cas_conflict_value,
                __ATOMIC_RELEASE
            );
            cas_trigger_address = nullptr;
            cas_conflict_address = nullptr;
        }
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(
            address, &observed, desired, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        return observed;
    }

    static int64_t FetchMax(
        volatile int64_t *address, int64_t value,
        uint64_t &retries
    ) {
        int64_t current =
            __atomic_load_n(address, __ATOMIC_ACQUIRE);
        retries = 0;
        while (value > current) {
            if (__atomic_compare_exchange_n(
                    address, &current, value, true,
                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
                )) {
                break;
            }
            ++retries;
        }
        return current;
    }

    static void StoreBarrier() {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static void FlushRegion(void *, uint64_t) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static void InvalidateRegion(const void *, uint64_t) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static uint64_t Now() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count()
        );
    }

    template <typename T>
    static uint64_t NowAfterAtomicResult(T value) {
        asm volatile("" : "+r"(value));
        return Now();
    }

    static void SpinHint() { std::this_thread::yield(); }
};

volatile int64_t *WriterIntentTestOps::wait_address = nullptr;
std::atomic<uint64_t> WriterIntentTestOps::wait_loads{0};
volatile int64_t *WriterIntentTestOps::cas_trigger_address =
    nullptr;
volatile int64_t *WriterIntentTestOps::cas_conflict_address =
    nullptr;
int64_t WriterIntentTestOps::cas_conflict_value = -1;

SchedulerState *MapSparseSchedulerState() {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void *memory = mmap(
        nullptr, sizeof(SchedulerState),
        PROT_READ | PROT_WRITE, flags, -1, 0
    );
    if (memory == MAP_FAILED) {
        std::perror("mmap SchedulerState");
        return nullptr;
    }
    return ::new (memory) SchedulerState;
}

void UnmapSparseSchedulerState(SchedulerState *state) {
    if (state != nullptr) {
        (void)munmap(state, sizeof(SchedulerState));
    }
}

void ResetProtocolState(SchedulerState &state) {
    state.fatal.value = 0;
    state.heap_window = kHeapWindow;
    for (uint32_t lane = 0;
         lane < kSharedInsertTurnCapacity; ++lane) {
        SharedInsertTurnLine(
            state.shared_map, lane
        ).value = -10000 - static_cast<int64_t>(lane);
    }
    state.shared_map.reclaim_upto.value = -1;
    for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
        state.shared_map.buckets[bucket].head.value = 0;
        state.shared_map.buckets[bucket].tail.value = 0;
    }
    for (uint32_t slot = 0; slot < kMapCapacity; ++slot) {
        state.shared_map.slots[slot].seq.value =
            kSharedMapEmptySeq;
    }
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        state.shared_map.reader_done[worker].value = -1;
    }
    for (uint32_t task = 0; task < kMaxTasks; ++task) {
        state.tasks[task].deps_prepared = -1;
        SharedOutputCell &outputs =
            state.shared_map.shared_outputs[task];
        for (uint32_t output = 0;
             output < kSharedOutputMaxPerTask; ++output) {
            outputs.published[output].value = -1;
            outputs.last_writer[output].value = -1;
        }
        SharedWriterHistoryCell &history =
            state.shared_map.writer_history[task];
        history.magic = 0;
        history.writer_task = 0;
        history.count = 0;
        history.reserved = 0;
    }
}

void SetInsertCompletionsAfterTasks(
    SchedulerState &state, uint32_t completed_tasks
) {
    for (uint32_t task = 0; task < completed_tasks;
         ++task) {
        state.tasks[task].deps_prepared =
            static_cast<int64_t>(task);
    }
}

bool InsertCompletionsMatch(
    SchedulerState &state, uint32_t completed_tasks
) {
    for (uint32_t task = 0; task < completed_tasks;
         ++task) {
        if (WriterIntentTestOps::Load(
                &state.tasks[task].deps_prepared
            ) != static_cast<int64_t>(task)) {
            return false;
        }
    }
    if (completed_tasks < kMaxTasks &&
        WriterIntentTestOps::Load(
            &state.tasks[completed_tasks].deps_prepared
        ) != -1) {
        return false;
    }
    for (uint32_t lane = 0;
         lane < kSharedInsertTurnCapacity; ++lane) {
        if (WriterIntentTestOps::Load(
                &SharedInsertTurnLine(
                    state.shared_map, lane
                ).value
            ) !=
            -10000 - static_cast<int64_t>(lane)) {
            return false;
        }
    }
    return true;
}

void ResetTaskGate(SchedulerState &state, int32_t task_id) {
    state.tasks[static_cast<uint32_t>(task_id)].flag = 0;
    state.tasks[static_cast<uint32_t>(task_id)].deps_prepared = -1;
}

TensorDesc MakeTensor(
    uint64_t address, uint64_t owner = kInvalidTaskId,
    bool manual_dep = false
) {
    TensorDesc tensor{};
    tensor.buffer_addr = address;
    tensor.buffer_size = 4096;
    tensor.owner_task_id = owner;
    tensor.ndims = 1;
    tensor.dtype = DataType::Float32;
    tensor.manual_dep = manual_dep;
    tensor.is_contiguous = true;
    tensor.shapes[0] = 1024;
    tensor.strides[0] = 1;
    tensor.extent_elem_cache = 1024;
    return tensor;
}

uint64_t FindAddressForBucket(
    uint64_t begin, uint32_t wanted_bucket,
    uint32_t skip_matches = 0
) {
    for (uint64_t address = begin;
         address < begin + (1ULL << 28); address += 4096) {
        if (TensorMapHash(address) != wanted_bucket) {
            continue;
        }
        if (skip_matches == 0) {
            return address;
        }
        --skip_matches;
    }
    return 0;
}

bool WaitUntilObserved(std::atomic<uint64_t> &counter) {
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::seconds(2);
    while (counter.load(std::memory_order_acquire) == 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::yield();
    }
    return true;
}

bool WaitUntilTrue(std::atomic<bool> &value) {
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::seconds(2);
    while (!value.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::yield();
    }
    return true;
}

int64_t LoadReaderDone(
    SchedulerState &state, uint32_t worker
) {
    return WriterIntentTestOps::Load(
        &state.shared_map.reader_done[worker].value
    );
}

void StoreReaderDone(
    SchedulerState &state, uint32_t worker, int64_t value
) {
    __atomic_store_n(
        &state.shared_map.reader_done[worker].value,
        value, __ATOMIC_RELEASE
    );
}

void TestReaderProgressStateMachine(SchedulerState &state) {
    bool reset_ok = true;
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        reset_ok &= LoadReaderDone(state, worker) == -1;
    }
    Check(reset_ok, "all reader progress lines start at -1");

    Check(
        SharedAdvanceReaderDone<WriterIntentTestOps>(
            state.shared_map, 0, 0
        ),
        "reader progress accepts the exact -1 to 0 transition"
    );
    Check(
        !SharedAdvanceReaderDone<WriterIntentTestOps>(
            state.shared_map, 0, 0
        ) &&
            LoadReaderDone(state, 0) == 0,
        "reader progress rejects a duplicate without overwriting"
    );
    Check(
        !SharedAdvanceReaderDone<WriterIntentTestOps>(
            state.shared_map, 0, 2
        ) &&
            LoadReaderDone(state, 0) == 0,
        "reader progress rejects a skipped task without overwriting"
    );
    Check(
        SharedAdvanceReaderDone<WriterIntentTestOps>(
            state.shared_map, 0, 1
        ),
        "reader progress accepts the next contiguous task"
    );
    Check(
        !SharedAdvanceReaderDone<WriterIntentTestOps>(
            state.shared_map, 0, 0
        ) &&
            LoadReaderDone(state, 0) == 1,
        "reader progress rejects a backward task without overwriting"
    );

    StoreReaderDone(
        state, 1, static_cast<int64_t>(kMaxTasks) - 2
    );
    Check(
        SharedAdvanceReaderDone<WriterIntentTestOps>(
            state.shared_map, 1,
            static_cast<int32_t>(kMaxTasks) - 1
        ) &&
            LoadReaderDone(state, 1) ==
                static_cast<int64_t>(kMaxTasks) - 1,
        "reader progress accepts the last task in the shared task domain"
    );
    const int64_t worker0_before = LoadReaderDone(state, 0);
    Check(
        !SharedAdvanceReaderDone<WriterIntentTestOps>(
            state.shared_map, kWorkers, 0
        ),
        "reader progress rejects an out-of-range worker"
    );
    Check(
        !SharedAdvanceReaderDone<WriterIntentTestOps>(
            state.shared_map, 0, -1
        ),
        "reader progress rejects a negative task"
    );
    Check(
        !SharedAdvanceReaderDone<WriterIntentTestOps>(
            state.shared_map, 0,
            static_cast<int32_t>(kMaxTasks)
        ),
        "reader progress rejects a task above the shared domain"
    );
    Check(
        LoadReaderDone(state, 0) == worker0_before,
        "reader progress bounds failures do not touch state"
    );

    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        StoreReaderDone(state, worker, -1);
    }
    const int32_t targets[] = {9, 5, 8};
    bool sequence_ok = true;
    for (uint32_t worker = 0; worker < 3; ++worker) {
        for (int32_t task = 0; task <= targets[worker]; ++task) {
            sequence_ok &=
                SharedAdvanceReaderDone<WriterIntentTestOps>(
                    state.shared_map, worker, task
                );
        }
    }
    Check(
        sequence_ok,
        "three active readers publish contiguous completion sequences"
    );

    // active worker 是连续前缀；前缀之外的脏值不得参与最小值。
    StoreReaderDone(state, 3, -2);
    int64_t candidate = -77;
    Check(
        SharedComputeReaderReclaimCandidate<WriterIntentTestOps>(
            state.shared_map, 3, 0, candidate
        ) &&
            candidate == 5,
        "H=0 exposes the exact active-prefix minimum"
    );
    candidate = -77;
    Check(
        SharedComputeReaderReclaimCandidate<WriterIntentTestOps>(
            state.shared_map, 3, 2, candidate
        ) &&
            candidate == 3,
        "reader minimum 5 and H=2 produce inclusive reclaim 3"
    );
    candidate = -77;
    Check(
        !SharedComputeReaderReclaimCandidate<WriterIntentTestOps>(
            state.shared_map, 4, 2, candidate
        ) &&
            candidate == -77,
        "an invalid active progress line fails without changing output"
    );

    Check(
        SharedAdvanceReaderDone<WriterIntentTestOps>(
            state.shared_map, 0, 10
        ) &&
            SharedAdvanceReaderDone<WriterIntentTestOps>(
                state.shared_map, 2, 9
            ),
        "faster readers may move without changing the slow frontier"
    );
    candidate = -77;
    Check(
        SharedComputeReaderReclaimCandidate<WriterIntentTestOps>(
            state.shared_map, 3, 2, candidate
        ) &&
            candidate == 3,
        "faster-reader progress leaves the slow-reader reclaim unchanged"
    );
    Check(
        SharedAdvanceReaderDone<WriterIntentTestOps>(
            state.shared_map, 1, 6
        ),
        "the slow reader advances by one task"
    );
    candidate = -77;
    Check(
        SharedComputeReaderReclaimCandidate<WriterIntentTestOps>(
            state.shared_map, 3, 2, candidate
        ) &&
            candidate == 4,
        "the reclaim candidate advances only with the slow reader"
    );

    StoreReaderDone(
        state, 1, static_cast<int64_t>(kMaxTasks)
    );
    candidate = -77;
    Check(
        !SharedComputeReaderReclaimCandidate<WriterIntentTestOps>(
            state.shared_map, 3, 2, candidate
        ) &&
            candidate == -77,
        "reader progress above the task domain fails without output"
    );
    StoreReaderDone(state, 1, 6);
    candidate = -77;
    Check(
        !SharedComputeReaderReclaimCandidate<WriterIntentTestOps>(
            state.shared_map, 0, 2, candidate
        ) &&
            candidate == -77,
        "reader reclaim rejects an empty active prefix without output"
    );
    candidate = -77;
    Check(
        !SharedComputeReaderReclaimCandidate<WriterIntentTestOps>(
            state.shared_map, kWorkers + 1, 2, candidate
        ) &&
            candidate == -77,
        "reader reclaim rejects an oversized active prefix without output"
    );
    candidate = -77;
    Check(
        !SharedComputeReaderReclaimCandidate<WriterIntentTestOps>(
            state.shared_map, 3, -1, candidate
        ) &&
            candidate == -77,
        "reader reclaim rejects a negative window without output"
    );
    candidate = -77;
    Check(
        SharedComputeReaderReclaimCandidate<WriterIntentTestOps>(
            state.shared_map, 3, INT32_MAX, candidate
        ) &&
            candidate == -1,
        "a very wide legal window clamps the reclaim candidate to -1"
    );

    // BeginCallbackSubmit 会在 task 读取前推进 local_index。即使该普通字段
    // 已到 6，task 5 的 ordinary lookup 仍可能尚未结束；真正完成前沿 4
    // 在 H=2 时只能回收到 2，不能按 local_index 推到 3/4。
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        StoreReaderDone(state, worker, -1);
    }
    bool local_progress_ok = true;
    for (int32_t task = 0; task <= 4; ++task) {
        local_progress_ok &=
            SharedAdvanceReaderDone<WriterIntentTestOps>(
                state.shared_map, 0, task
            );
    }
    Check(
        local_progress_ok,
        "local-index counterexample prepares reader completion"
    );
    const int32_t saved_local_index =
        state.workers[0].local_index;
    state.workers[0].local_index = 5;
    SubmitContext context{};
    BeginCallbackSubmit(state.workers[0], context);
    Check(
        context.task_id == 5 &&
            state.workers[0].local_index == 6,
        "BeginCallbackSubmit advances local_index before task-5 reads"
    );
    candidate = -77;
    Check(
        SharedComputeReaderReclaimCandidate<WriterIntentTestOps>(
            state.shared_map, 1, 2, candidate
        ) &&
            candidate == 2,
        "reader reclaim is independent of the pre-read local_index"
    );
    state.workers[0].local_index = 100;
    candidate = -77;
    Check(
        SharedComputeReaderReclaimCandidate<WriterIntentTestOps>(
            state.shared_map, 1, 2, candidate
        ) &&
            candidate == 2,
        "changing local_index cannot move the reader completion frontier"
    );
    state.workers[0].local_index = saved_local_index;
}

FdwicOutputRef PublishSingleTestSymbol(
    SchedulerState &state, int32_t producer, uint64_t address
) {
    SharedOutputCell &cell =
        state.shared_map.shared_outputs[
            static_cast<uint32_t>(producer)
        ];
    cell.published[0].value = -1;
    cell.last_writer[0].value = -1;
    TensorDesc tensor = MakeTensor(
        address, static_cast<uint64_t>(producer)
    );
    SubmitContext context{};
    context.task_id = producer;
    context.result.task_id = producer;
    context.result.count = 1;
    context.result.tensors[0] = &tensor;
    context.shared_result.Reset(producer);
    const bool ref_ok =
        context.shared_result.AddOutputRef(producer, 0);
    Check(ref_ok, "single test symbol accepts slot 0");
    if (!ref_ok ||
        !PublishSharedTaskOutputs<WriterIntentTestOps>(
            state.shared_map, context, producer
        )) {
        Check(false, "single test symbol publishes descriptor");
        return InvalidSharedOutputRef();
    }
    return context.shared_result.OutputRef(0);
}

void TestSymbolWriterIntentChain(SchedulerState &state) {
    constexpr int32_t kExistingDependency = 5;
    constexpr int32_t kProducer = 10;
    constexpr int32_t kWriter = 20;
    constexpr int32_t kReader = 30;
    constexpr int32_t kFutureWriter = 40;
    constexpr int32_t kFarFutureWriter = 50;
    constexpr uint32_t kSymbolCount = 7;
    ResetTaskGate(state, kWriter);
    ResetTaskGate(state, kFutureWriter);
    ResetTaskGate(state, kFarFutureWriter);

    SharedOutputCell &cell =
        state.shared_map.shared_outputs[kProducer];
    for (uint32_t output = 0; output < kSymbolCount; ++output) {
        cell.published[output].value = -1;
        cell.last_writer[output].value = -1;
    }

    TensorDesc produced[kSymbolCount] = {};
    FdwicOutputRef output_refs[kSymbolCount] = {};
    SubmitContext producer_context{};
    producer_context.task_id = kProducer;
    producer_context.result.task_id = kProducer;
    producer_context.result.count =
        static_cast<int32_t>(kSymbolCount);
    producer_context.shared_result.Reset(kProducer);
    for (uint32_t output = 0; output < kSymbolCount; ++output) {
        produced[output] = MakeTensor(
            0x410000000ULL +
                static_cast<uint64_t>(output) * 0x10000ULL,
            static_cast<uint64_t>(kProducer)
        );
        producer_context.result.tensors[output] =
            &produced[output];
        Check(
            producer_context.shared_result.AddOutputRef(
                kProducer, static_cast<int16_t>(output)
            ),
            "symbol producer accepts history-test output"
        );
        output_refs[output] =
            producer_context.shared_result.OutputRef(output);
    }
    Check(
        PublishSharedTaskOutputs<WriterIntentTestOps>(
            state.shared_map, producer_context, kProducer
        ),
        "symbol producer publishes descriptor"
    );

    const FdwicOutputRef output_ref = output_refs[0];
    TaskArgs writer_args;
    ConstructTaskArgs(writer_args);
    for (uint32_t output = 0; output < kSymbolCount; ++output) {
        AppendSharedOutputRef(
            writer_args, output_refs[output],
            TensorArgType::Inout
        );
    }
    bool required = false;
    Check(
        InspectSharedWriterIntent(writer_args, required) &&
            required,
        "symbol INOUT is a generic writer intent"
    );

    SubmitContext writer_context{};
    writer_context.task_id = kWriter;
    writer_context.won = true;
    writer_context.fanin_count = 2;
    writer_context.fanin[0] = kExistingDependency;
    writer_context.fanin[1] = kProducer;
    LocalStats writer_stats{};

    TaskArgs reader_args;
    ConstructTaskArgs(reader_args);
    AppendSharedOutputRef(
        reader_args, output_ref, TensorArgType::Input
    );

    std::atomic<bool> reader_finished{false};
    std::atomic<bool> reader_past_writer_gate{false};
    std::atomic<bool> allow_reader_lookup{false};
    bool wait_ok = false;
    bool lookup_ok = false;
    uint32_t ordinary_lookups = UINT32_MAX;
    uint32_t reader_fanin_count = 0;
    int32_t reader_fanin[kMaxFanin] = {};
    LocalStats reader_stats{};
    WriterIntentTestOps::wait_address =
        &state.tasks[kWriter].deps_prepared;
    WriterIntentTestOps::wait_loads.store(
        0, std::memory_order_relaxed
    );
    std::thread reader([&]() {
        wait_ok = WaitForSharedWriterReady<WriterIntentTestOps>(
            &state, kWriter, reader_stats
        );
        if (wait_ok) {
            // C 已经观察到 B 的 writer-ready，但故意停在 symbol lookup
            // 之前；主线程随后让未来 writer D 覆盖 latest cache。这个
            // 时序隔离“门补齐了 B”与“查询仍能找回 B”两项不同性质。
            reader_past_writer_gate.store(
                true, std::memory_order_release
            );
            while (!allow_reader_lookup.load(
                std::memory_order_acquire
            )) {
                std::this_thread::yield();
            }
            reader_fanin_count =
                CollectSharedFanin<
                    WriterIntentTestOps, false, true
                >(
                    state.shared_map, reader_args, kReader,
                    static_cast<int32_t>(state.heap_window),
                    reader_stats, reader_fanin, lookup_ok,
                    ordinary_lookups, &state.fatal.value
                );
        }
        reader_finished.store(true, std::memory_order_release);
    });

    Check(
        WaitUntilObserved(WriterIntentTestOps::wait_loads),
        "symbol reader reaches the delayed writer gate"
    );
    Check(
        !reader_finished.load(std::memory_order_acquire),
        "symbol reader cannot resolve before writer metadata"
    );
    const SharedWriterIntentResult prepared =
        PrepareSharedWriterIntentSet<WriterIntentTestOps>(
            &state, writer_args, writer_context, writer_stats
        );

    Check(
        WaitUntilTrue(reader_past_writer_gate),
        "symbol reader passes B gate before future D publishes"
    );
    TaskArgs future_writer_args;
    ConstructTaskArgs(future_writer_args);
    for (uint32_t output = 0; output < kSymbolCount; ++output) {
        AppendSharedOutputRef(
            future_writer_args, output_refs[output],
            TensorArgType::Inout
        );
    }
    SubmitContext future_writer_context{};
    future_writer_context.task_id = kFutureWriter;
    future_writer_context.won = true;
    LocalStats future_writer_stats{};
    const SharedWriterIntentResult future_prepared =
        PrepareSharedWriterIntentSet<WriterIntentTestOps>(
            &state, future_writer_args,
            future_writer_context, future_writer_stats
        );
    TaskArgs far_future_writer_args;
    ConstructTaskArgs(far_future_writer_args);
    for (uint32_t output = 0; output < kSymbolCount; ++output) {
        AppendSharedOutputRef(
            far_future_writer_args, output_refs[output],
            TensorArgType::Inout
        );
    }
    SubmitContext far_future_writer_context{};
    far_future_writer_context.task_id = kFarFutureWriter;
    far_future_writer_context.won = true;
    LocalStats far_future_writer_stats{};
    const SharedWriterIntentResult far_future_prepared =
        PrepareSharedWriterIntentSet<WriterIntentTestOps>(
            &state, far_future_writer_args,
            far_future_writer_context, far_future_writer_stats
        );
    uint32_t symbol_key = 0;
    uint32_t seventh_symbol_key = 0;
    const bool key_ok =
        SharedSymbolHistoryKey(output_ref, symbol_key) &&
        SharedSymbolHistoryKey(
            output_refs[kSymbolCount - 1],
            seventh_symbol_key
        );
    const SharedWriterHistoryCell &writer_history =
        state.shared_map.writer_history[kWriter];
    const SharedWriterHistoryCell &future_history =
        state.shared_map.writer_history[kFutureWriter];
    const SharedWriterHistoryCell &far_future_history =
        state.shared_map.writer_history[kFarFutureWriter];
    Check(
        key_ok &&
            writer_history.magic ==
                kSharedWriterHistoryMagic &&
            writer_history.writer_task == kWriter &&
            writer_history.count == kSymbolCount &&
            writer_history.entries[0].symbol_key ==
                symbol_key &&
            writer_history.entries[0].previous_writer ==
                kProducer &&
            writer_history.entries[kSymbolCount - 1].symbol_key ==
                seventh_symbol_key,
        "B publishes seven A predecessor records across two cache lines"
    );
    Check(
        future_history.magic ==
                kSharedWriterHistoryMagic &&
            future_history.writer_task == kFutureWriter &&
            future_history.count == kSymbolCount &&
            future_history.entries[0].symbol_key ==
                symbol_key &&
            future_history.entries[0].previous_writer ==
                kWriter,
        "D publishes an immutable B predecessor record"
    );
    Check(
        far_future_history.magic ==
                kSharedWriterHistoryMagic &&
            far_future_history.writer_task == kFarFutureWriter &&
            far_future_history.count == kSymbolCount &&
            far_future_history.entries[0].symbol_key ==
                symbol_key &&
            far_future_history.entries[0].previous_writer ==
                kFutureWriter,
        "E publishes an immutable D predecessor record"
    );
    allow_reader_lookup.store(true, std::memory_order_release);
    reader.join();
    WriterIntentTestOps::wait_address = nullptr;

    Check(
        prepared == SharedWriterIntentResult::Published,
        "symbol writer publishes the generic intent"
    );
    Check(
        writer_context.fanin_count == 2 &&
            writer_context.fanin[0] == kExistingDependency &&
            writer_context.fanin[1] == kProducer,
        "symbol writer retains existing fanin and deduplicates the previous writer"
    );
    Check(
        future_prepared == SharedWriterIntentResult::Published &&
            future_writer_context.fanin_count == 1 &&
            future_writer_context.fanin[0] == kWriter &&
            state.tasks[kFutureWriter].deps_prepared ==
                kFutureWriter,
        "future symbol writer D consumes B before publishing"
    );
    Check(
        far_future_prepared ==
                SharedWriterIntentResult::Published &&
            far_future_writer_context.fanin_count == 1 &&
            far_future_writer_context.fanin[0] ==
                kFutureWriter &&
            state.tasks[kFarFutureWriter].deps_prepared ==
                kFarFutureWriter,
        "far-future symbol writer E consumes D before publishing"
    );
    Check(
        cell.last_writer[0].value == kFarFutureWriter &&
            state.tasks[kWriter].deps_prepared == kWriter,
        "latest cache advances to E after all writer-ready gates"
    );
    Check(
        state.tasks[kWriter].flag == 0,
        "symbol writer-ready is not kernel completion"
    );
    Check(
        wait_ok && lookup_ok &&
            reader_fanin_count == 1 &&
            reader_fanin[0] == kWriter &&
            ordinary_lookups == 0,
        "slow C follows E-to-D-to-B history after latest advances"
    );
}

void TestOutOfOrderSymbolWriterFailsClosed(
    SchedulerState &state
) {
    constexpr int32_t kProducer = 200;
    constexpr int32_t kEarlierWriter = 220;
    constexpr int32_t kLaterWriter = 240;
    state.fatal.value = 0;
    ResetTaskGate(state, kEarlierWriter);
    ResetTaskGate(state, kLaterWriter);
    const FdwicOutputRef output_ref =
        PublishSingleTestSymbol(
            state, kProducer, 0x440000000ULL
        );

    TaskArgs args;
    ConstructTaskArgs(args);
    AppendSharedOutputRef(
        args, output_ref, TensorArgType::Inout
    );
    SubmitContext later_context{};
    later_context.task_id = kLaterWriter;
    later_context.won = true;
    LocalStats later_stats{};
    const SharedWriterIntentResult later_result =
        PrepareSharedWriterIntentSet<WriterIntentTestOps>(
            &state, args, later_context, later_stats
        );

    SubmitContext earlier_context{};
    earlier_context.task_id = kEarlierWriter;
    earlier_context.won = true;
    LocalStats earlier_stats{};
    const SharedWriterIntentResult earlier_result =
        PrepareSharedWriterIntentSet<WriterIntentTestOps>(
            &state, args, earlier_context, earlier_stats
        );
    const SharedOutputCell &cell =
        state.shared_map.shared_outputs[kProducer];
    Check(
        later_result == SharedWriterIntentResult::Published &&
            cell.last_writer[0].value == kLaterWriter,
        "out-of-order setup publishes the later writer first"
    );
    Check(
        earlier_result == SharedWriterIntentResult::Failed &&
            state.fatal.value == 1 &&
            state.tasks[kEarlierWriter].deps_prepared == -1 &&
            cell.last_writer[0].value == kLaterWriter,
        "skipped earlier writer fails closed without moving latest backward"
    );
}

void TestMultiSymbolConflictKeepsTerminalPrefix(
    SchedulerState &state
) {
    constexpr int32_t kProducer0 = 300;
    constexpr int32_t kProducer1 = 301;
    constexpr int32_t kConflictingWriter = 350;
    constexpr int32_t kWriter = 400;
    state.fatal.value = 0;
    ResetTaskGate(state, kWriter);
    const FdwicOutputRef output0 =
        PublishSingleTestSymbol(
            state, kProducer0, 0x450000000ULL
        );
    const FdwicOutputRef output1 =
        PublishSingleTestSymbol(
            state, kProducer1, 0x460000000ULL
        );

    TaskArgs args;
    ConstructTaskArgs(args);
    AppendSharedOutputRef(
        args, output0, TensorArgType::Inout
    );
    AppendSharedOutputRef(
        args, output1, TensorArgType::Inout
    );
    SubmitContext context{};
    context.task_id = kWriter;
    context.won = true;
    LocalStats stats{};
    volatile int64_t *first_latest =
        &state.shared_map.shared_outputs[kProducer0]
             .last_writer[0].value;
    volatile int64_t *second_latest =
        &state.shared_map.shared_outputs[kProducer1]
             .last_writer[0].value;
    WriterIntentTestOps::cas_trigger_address = first_latest;
    WriterIntentTestOps::cas_conflict_address = second_latest;
    WriterIntentTestOps::cas_conflict_value =
        kConflictingWriter;
    const SharedWriterIntentResult result =
        PrepareSharedWriterIntentSet<WriterIntentTestOps>(
            &state, args, context, stats
        );
    WriterIntentTestOps::cas_trigger_address = nullptr;
    WriterIntentTestOps::cas_conflict_address = nullptr;

    const SharedWriterHistoryCell &history =
        state.shared_map.writer_history[kWriter];
    Check(
        result == SharedWriterIntentResult::Failed &&
            state.fatal.value == 1 &&
            state.tasks[kWriter].deps_prepared == -1,
        "multi-symbol CAS conflict terminates without publishing ready"
    );
    Check(
        *first_latest == kWriter &&
            *second_latest == kConflictingWriter &&
            stats.result.shared_symbol_inout_commits == 1,
        "terminal failure preserves and counts only the linearized prefix"
    );
    Check(
        history.magic == kSharedWriterHistoryMagic &&
            history.writer_task == kWriter &&
            history.count == 2 &&
            history.entries[0].previous_writer ==
                kProducer0 &&
            history.entries[1].previous_writer ==
                kProducer1,
        "terminal prefix retains its immutable diagnostic history"
    );
}

void TestOrdinaryWriterIntentChain(SchedulerState &state) {
    constexpr int32_t kFirstWriter = 100;
    constexpr int32_t kSecondWriter = 120;
    constexpr int32_t kReader = 140;
    ResetTaskGate(state, kFirstWriter);
    ResetTaskGate(state, kSecondWriter);
    TensorDesc tensor = MakeTensor(0x420000000ULL);

    TaskArgs first_args;
    ConstructTaskArgs(first_args);
    AddGmTensor(
        first_args, tensor, TensorArgType::OutputExisting
    );
    SubmitContext first_context{};
    first_context.task_id = kFirstWriter;
    first_context.won = true;
    LocalStats first_stats{};
    Check(
        PrepareSharedWriterIntentSet<WriterIntentTestOps>(
            &state, first_args, first_context, first_stats
        ) == SharedWriterIntentResult::Published,
        "ordinary OUTPUT_EXISTING publishes the first writer"
    );
    Check(
        first_context.fanin_count == 0,
        "first external ordinary writer has no predecessor"
    );

    TaskArgs second_args;
    ConstructTaskArgs(second_args);
    AddGmTensor(
        second_args, tensor, TensorArgType::Inout
    );
    SubmitContext second_context{};
    second_context.task_id = kSecondWriter;
    second_context.won = true;
    LocalStats second_stats{};

    TaskArgs reader_args;
    ConstructTaskArgs(reader_args);
    AddGmTensor(reader_args, tensor, TensorArgType::Input);
    std::atomic<bool> reader_finished{false};
    bool wait_ok = false;
    bool lookup_ok = false;
    uint32_t lookup_count = UINT32_MAX;
    uint32_t reader_fanin_count = 0;
    int32_t reader_fanin[kMaxFanin] = {};
    LocalStats reader_stats{};
    WriterIntentTestOps::wait_address =
        &state.tasks[kSecondWriter].deps_prepared;
    WriterIntentTestOps::wait_loads.store(
        0, std::memory_order_relaxed
    );
    std::thread reader([&]() {
        wait_ok = WaitForSharedWriterReady<WriterIntentTestOps>(
            &state, kSecondWriter, reader_stats
        );
        if (wait_ok) {
            reader_fanin_count =
                CollectSharedFanin<
                    WriterIntentTestOps, false, true
                >(
                    state.shared_map, reader_args, kReader,
                    static_cast<int32_t>(state.heap_window),
                    reader_stats, reader_fanin, lookup_ok,
                    lookup_count, &state.fatal.value
                );
        }
        reader_finished.store(true, std::memory_order_release);
    });

    Check(
        WaitUntilObserved(WriterIntentTestOps::wait_loads),
        "ordinary reader reaches the delayed writer gate"
    );
    Check(
        !reader_finished.load(std::memory_order_acquire),
        "ordinary reader cannot resolve before writer metadata"
    );
    const SharedWriterIntentResult prepared =
        PrepareSharedWriterIntentSet<WriterIntentTestOps>(
            &state, second_args, second_context, second_stats
        );
    reader.join();
    WriterIntentTestOps::wait_address = nullptr;

    Check(
        prepared == SharedWriterIntentResult::Published,
        "ordinary INOUT publishes the second writer"
    );
    Check(
        second_context.fanin_count == 1 &&
            second_context.fanin[0] == kFirstWriter,
        "ordinary INOUT consumes the first writer"
    );
    Check(
        state.tasks[kSecondWriter].deps_prepared ==
            kSecondWriter &&
            state.tasks[kSecondWriter].flag == 0,
        "ordinary metadata-ready remains separate from completion"
    );
    Check(
        wait_ok && lookup_ok &&
            reader_fanin_count == 1 &&
            reader_fanin[0] == kSecondWriter &&
            lookup_count == 1,
        "ownerless ordinary INPUT resolves the latest writer"
    );
    Check(
        InsertCompletionsMatch(state, 0),
        "non-adjacent writer-intent preparation leaves the completion chain untouched"
    );
}

void TestWriterDeltaRequiresExactRegisterMask() {
    TensorDesc tensor = MakeTensor(0x455000000ULL);
    const FdwicOutputRef output{0, 0, 0, 0, 0, 0};
    TaskArgs args;
    ConstructTaskArgs(args);
    AddGmTensor(args, tensor, TensorArgType::OutputExisting);
    AddOutputHandleTensor(args, output, TensorArgType::Inout);

    SubmitContext context{};
    context.task_id = 1;
    context.won = true;
    context.result.task_id = 1;
    context.shared_result.Reset(1);
    SharedTaskWriterDelta delta{};

    context.register_mask = 1;
    Check(
        !PrepareSharedTaskWriterDelta(args, context, delta),
        "writer delta rejects a mask that omits one INOUT entry"
    );
    context.register_mask = 2;
    Check(
        !PrepareSharedTaskWriterDelta(args, context, delta),
        "writer delta rejects a mask that omits one ordinary writer"
    );
    context.register_mask = 7;
    Check(
        !PrepareSharedTaskWriterDelta(args, context, delta),
        "writer delta rejects a mask with an extra non-argument bit"
    );
    context.register_mask = 3;
    Check(
        PrepareSharedTaskWriterDelta(args, context, delta) &&
            delta.prepared_task_id == context.task_id &&
            delta.ordinary_count == 1 &&
            delta.symbol_count == 1 &&
            delta.ordinary_buckets[0] ==
                TensorMapHash(tensor.buffer_addr) &&
            delta.ordinary_bucket_ordinals[0] == 0 &&
            delta.symbol_keys[0] == 1 &&
            delta.writer_intent_required,
        "writer delta freezes task identity, writer counts, buckets and symbol keys"
    );

    TaskArgs duplicate_args;
    ConstructTaskArgs(duplicate_args);
    AddOutputHandleTensor(
        duplicate_args, output, TensorArgType::Inout
    );
    AddOutputHandleTensor(
        duplicate_args, output, TensorArgType::OutputExisting
    );
    context.register_mask = 3;
    Check(
        !PrepareSharedTaskWriterDelta(
            duplicate_args, context, delta
        ),
        "writer delta rejects a duplicate precomputed symbol key"
    );

    // 低 32 位看似是合法前任的畸形 owner 也必须在 writer delta
    // 准备阶段拒绝，不能截断成 task 0 后发布 ordinary 元数据。
    tensor.owner_task_id = uint64_t{1} << 32;
    context.task_id = 1;
    context.result.task_id = 1;
    context.register_mask = 3;
    Check(
        !ValidateOrdinarySharedWriterReference(
            tensor, context.task_id
        ) &&
            !PrepareSharedTaskWriterDelta(
                args, context, delta
            ),
        "writer delta rejects owner ids with nonzero high 32 bits"
    );
}

void TestWriterDeltaPrecomputesInterleavedBuckets(
    SchedulerState &state
) {
    ResetProtocolState(state);
    constexpr uint64_t kBase = 0x458000000ULL;
    const uint32_t bucket_a = TensorMapHash(kBase);
    const uint32_t bucket_b =
        (bucket_a + 1U) % kMapBuckets;
    const uint64_t address_a1 =
        FindAddressForBucket(kBase + 4096, bucket_a);
    const uint64_t address_a2 =
        FindAddressForBucket(address_a1 + 4096, bucket_a);
    const uint64_t address_b =
        FindAddressForBucket(kBase + 4096, bucket_b);
    Check(
        address_a1 != 0 && address_a2 != 0 &&
            address_b != 0,
        "interleaved bucket test finds distinct addresses"
    );
    if (address_a1 == 0 || address_a2 == 0 ||
        address_b == 0) {
        return;
    }

    TensorDesc tensors[4] = {
        MakeTensor(kBase),
        MakeTensor(address_b),
        MakeTensor(address_a1),
        MakeTensor(address_a2)
    };
    TaskArgs args;
    ConstructTaskArgs(args);
    for (TensorDesc &tensor : tensors) {
        AddGmTensor(
            args, tensor, TensorArgType::OutputExisting
        );
    }
    SubmitContext context{};
    context.task_id = 0;
    context.won = true;
    context.register_mask = 15;
    context.result.task_id = 0;
    context.shared_result.Reset(0);
    SharedTaskWriterDelta delta{};
    LocalStats stats{};
    Check(
        PrepareSharedTaskWriterDelta(
            args, context, delta
        ) &&
            delta.ordinary_count == 4 &&
            delta.ordinary_buckets[0] == bucket_a &&
            delta.ordinary_buckets[1] == bucket_b &&
            delta.ordinary_buckets[2] == bucket_a &&
            delta.ordinary_buckets[3] == bucket_a &&
            delta.ordinary_bucket_ordinals[0] == 0 &&
            delta.ordinary_bucket_ordinals[1] == 0 &&
            delta.ordinary_bucket_ordinals[2] == 1 &&
            delta.ordinary_bucket_ordinals[3] == 2,
        "writer delta precomputes A,B,A,A buckets and 0,0,1,2 ordinals"
    );
    Check(
        PublishSharedTaskWriterDelta<WriterIntentTestOps>(
            &state, context, delta, stats
        ) &&
            state.shared_map.buckets[bucket_a].tail.value == 3 &&
            state.shared_map.buckets[bucket_b].tail.value == 1 &&
            stats.result.map_inserts == 4,
        "prepared interleaved metadata publishes without reordering"
    );
}

void TestOrderedOrdinaryInsertBeforeLookup(
    SchedulerState &state
) {
    ResetProtocolState(state);
    TensorDesc tensor = MakeTensor(0x460000000ULL);

    TaskArgs first_args;
    ConstructTaskArgs(first_args);
    AddGmTensor(
        first_args, tensor, TensorArgType::OutputExisting
    );
    SubmitContext first_context{};
    first_context.task_id = 0;
    first_context.won = true;
    first_context.register_mask = 1;
    first_context.result.task_id = 0;
    first_context.shared_result.Reset(0);
    SharedTaskWriterDelta first_delta{};
    LocalStats first_stats{};
    Check(
        PrepareSharedTaskWriterDelta(
            first_args, first_context, first_delta
        ) &&
            first_delta.prepared_task_id == 0 &&
            first_delta.ordinary_count == 1 &&
            first_delta.symbol_count == 0 &&
            first_delta.ordinary_buckets[0] ==
                TensorMapHash(tensor.buffer_addr) &&
            first_delta.ordinary_bucket_ordinals[0] == 0 &&
            first_delta.writer_intent_required,
        "ordered ordinary task 0 prepares one writer entry"
    );
    Check(
        PublishSharedTaskWriterDelta<WriterIntentTestOps>(
            &state, first_context, first_delta, first_stats
        ) &&
            InsertCompletionsMatch(state, 1) &&
            first_stats.result.map_inserts == 1 &&
            first_stats.result.shared_symbol_inout_commits == 0,
        "ordered ordinary task 0 publishes before lookup"
    );

    bool lookup_ok = false;
    uint32_t lookup_count = UINT32_MAX;
    int32_t fanin[kMaxFanin] = {};
    const uint32_t first_fanin =
        CollectSharedFanin<
            WriterIntentTestOps, false, true
        >(
            state.shared_map, first_args, 0,
            static_cast<int32_t>(state.heap_window),
            first_stats, fanin, lookup_ok,
            lookup_count, &state.fatal.value
        );
    Check(
        lookup_ok && first_fanin == 0 &&
            lookup_count == 1,
        "task 0 lookup after its own insert excludes itself"
    );

    TaskArgs second_args;
    ConstructTaskArgs(second_args);
    AddGmTensor(second_args, tensor, TensorArgType::Inout);
    SubmitContext second_context{};
    second_context.task_id = 1;
    second_context.won = true;
    second_context.register_mask = 1;
    second_context.result.task_id = 1;
    second_context.shared_result.Reset(1);
    SharedTaskWriterDelta second_delta{};
    LocalStats second_stats{};
    Check(
        PrepareSharedTaskWriterDelta(
            second_args, second_context, second_delta
        ) &&
            PublishSharedTaskWriterDelta<
                WriterIntentTestOps
            >(
                &state, second_context, second_delta,
                second_stats
            ) &&
            InsertCompletionsMatch(state, 2),
        "ordered ordinary task 1 publishes its writer entry"
    );
    lookup_ok = false;
    lookup_count = UINT32_MAX;
    const uint32_t second_fanin =
        CollectSharedFanin<
            WriterIntentTestOps, false, true
        >(
            state.shared_map, second_args, 1,
            static_cast<int32_t>(state.heap_window),
            second_stats, fanin, lookup_ok,
            lookup_count, &state.fatal.value
        );
    Check(
        lookup_ok && second_fanin == 1 &&
            fanin[0] == 0 && lookup_count == 1,
        "task 1 lookup returns task 0 instead of itself"
    );

    TaskArgs reader_args;
    ConstructTaskArgs(reader_args);
    AddGmTensor(reader_args, tensor, TensorArgType::Input);
    SubmitContext reader_context{};
    reader_context.task_id = 2;
    reader_context.won = true;
    reader_context.result.task_id = 2;
    reader_context.shared_result.Reset(2);
    SharedTaskWriterDelta reader_delta{};
    LocalStats reader_stats{};
    Check(
        PrepareSharedTaskWriterDelta(
            reader_args, reader_context, reader_delta
        ) &&
            reader_delta.ordinary_count == 0 &&
            PublishSharedTaskWriterDelta<
                WriterIntentTestOps
            >(
                &state, reader_context, reader_delta,
                reader_stats
            ) &&
            InsertCompletionsMatch(state, 3),
        "empty writer task still publishes its per-task completion"
    );
    lookup_ok = false;
    lookup_count = UINT32_MAX;
    const uint32_t reader_fanin =
        CollectSharedFanin<
            WriterIntentTestOps, false, true
        >(
            state.shared_map, reader_args, 2,
            static_cast<int32_t>(state.heap_window),
            reader_stats, fanin, lookup_ok,
            lookup_count, &state.fatal.value
        );
    Check(
        lookup_ok && reader_fanin == 1 &&
            fanin[0] == 1 && lookup_count == 1,
        "task 2 reader resolves the newest writer below task 2"
    );
}

void TestOrderedSymbolInsertBeforeLookup(
    SchedulerState &state
) {
    ResetProtocolState(state);
    TensorDesc descriptor = MakeTensor(0x470000000ULL, 0);

    TaskArgs producer_args;
    ConstructTaskArgs(producer_args);
    SubmitContext producer_context{};
    producer_context.task_id = 0;
    producer_context.won = true;
    producer_context.result.task_id = 0;
    producer_context.result.count = 1;
    producer_context.result.tensors[0] = &descriptor;
    producer_context.shared_result.Reset(0);
    Check(
        producer_context.shared_result.AddOutputRef(0, 0),
        "fresh symbol test declares output slot 0"
    );
    SharedTaskWriterDelta producer_delta{};
    LocalStats producer_stats{};
    Check(
        PrepareSharedTaskWriterDelta(
            producer_args, producer_context, producer_delta
        ) &&
            PublishSharedTaskWriterDelta<
                WriterIntentTestOps
            >(
                &state, producer_context, producer_delta,
                producer_stats
            ) &&
            state.shared_map.shared_outputs[0]
                    .published[0].value == 0 &&
            InsertCompletionsMatch(state, 1),
        "fresh symbol descriptor is visible before task 0 completion"
    );

    const FdwicOutputRef output{
        0, 0, 0, 0, 0, 0
    };
    TaskArgs writer_args;
    ConstructTaskArgs(writer_args);
    AddOutputHandleTensor(
        writer_args, output, TensorArgType::Inout
    );
    SubmitContext writer_context{};
    writer_context.task_id = 1;
    writer_context.won = true;
    writer_context.register_mask = 1;
    writer_context.result.task_id = 1;
    writer_context.shared_result.Reset(1);
    SharedTaskWriterDelta writer_delta{};
    LocalStats writer_stats{};
    Check(
        PrepareSharedTaskWriterDelta(
            writer_args, writer_context, writer_delta
        ) &&
            writer_delta.prepared_task_id == 1 &&
            writer_delta.ordinary_count == 0 &&
            writer_delta.symbol_count == 1 &&
            writer_delta.symbol_keys[0] == 1 &&
            writer_delta.writer_intent_required &&
            PublishSharedTaskWriterDelta<
                WriterIntentTestOps
            >(
                &state, writer_context, writer_delta,
                writer_stats
            ) &&
            state.shared_map.shared_outputs[0]
                    .last_writer[0].value == 1 &&
            writer_stats.result.map_inserts == 0 &&
            writer_stats.result.shared_symbol_inout_commits == 1 &&
            InsertCompletionsMatch(state, 2),
        "symbol INOUT publishes history before its own lookup"
    );

    bool lookup_ok = false;
    uint32_t lookup_count = UINT32_MAX;
    int32_t fanin[kMaxFanin] = {};
    const uint32_t writer_fanin =
        CollectSharedFanin<
            WriterIntentTestOps, false, true
        >(
            state.shared_map, writer_args, 1,
            static_cast<int32_t>(state.heap_window),
            writer_stats, fanin, lookup_ok,
            lookup_count, &state.fatal.value
        );
    Check(
        lookup_ok && writer_fanin == 1 &&
            fanin[0] == 0 && lookup_count == 0,
        "symbol task 1 walks immutable history back to task 0"
    );

    TaskArgs reader_args;
    ConstructTaskArgs(reader_args);
    AddOutputHandleTensor(
        reader_args, output, TensorArgType::Input
    );
    SubmitContext reader_context{};
    reader_context.task_id = 2;
    reader_context.won = true;
    reader_context.result.task_id = 2;
    reader_context.shared_result.Reset(2);
    SharedTaskWriterDelta reader_delta{};
    LocalStats reader_stats{};
    Check(
        PrepareSharedTaskWriterDelta(
            reader_args, reader_context, reader_delta
        ) &&
            PublishSharedTaskWriterDelta<
                WriterIntentTestOps
            >(
                &state, reader_context, reader_delta,
                reader_stats
            ),
        "symbol reader publishes an empty task transaction"
    );
    lookup_ok = false;
    lookup_count = UINT32_MAX;
    const uint32_t reader_fanin =
        CollectSharedFanin<
            WriterIntentTestOps, false, true
        >(
            state.shared_map, reader_args, 2,
            static_cast<int32_t>(state.heap_window),
            reader_stats, fanin, lookup_ok,
            lookup_count, &state.fatal.value
        );
    Check(
        lookup_ok && reader_fanin == 1 &&
            fanin[0] == 1 && lookup_count == 0,
        "symbol task 2 resolves task 1 as the latest prior writer"
    );
}

void TestOrderedMixedWriterTransaction(
    SchedulerState &state
) {
    ResetProtocolState(state);
    TensorDesc ordinary = MakeTensor(0x478000000ULL);
    TensorDesc seed_output = MakeTensor(0x478100000ULL, 0);

    // task 0 同时建立 ordinary writer 与后续 INOUT 使用的 fresh
    // descriptor。它先完整发布两类元数据，再把有序前沿推进到 1。
    TaskArgs seed_args;
    ConstructTaskArgs(seed_args);
    AddGmTensor(
        seed_args, ordinary, TensorArgType::OutputExisting
    );
    SubmitContext seed_context{};
    seed_context.task_id = 0;
    seed_context.won = true;
    seed_context.register_mask = 1;
    seed_context.result.task_id = 0;
    seed_context.result.count = 1;
    seed_context.result.tensors[0] = &seed_output;
    seed_context.shared_result.Reset(0);
    Check(
        seed_context.shared_result.AddOutputRef(0, 0),
        "mixed transaction seed declares fresh output"
    );
    SharedTaskWriterDelta seed_delta{};
    LocalStats seed_stats{};
    Check(
        PrepareSharedTaskWriterDelta(
            seed_args, seed_context, seed_delta
        ) &&
            PublishSharedTaskWriterDelta<
                WriterIntentTestOps
            >(
                &state, seed_context, seed_delta, seed_stats
            ),
        "mixed transaction seed publishes ordinary and fresh metadata"
    );

    const FdwicOutputRef seed_ref{0, 0, 0, 0, 0, 0};
    TensorDesc next_output = MakeTensor(0x478200000ULL, 1);
    TaskArgs mixed_args;
    ConstructTaskArgs(mixed_args);
    AddGmTensor(
        mixed_args, ordinary, TensorArgType::Inout
    );
    AddOutputHandleTensor(
        mixed_args, seed_ref, TensorArgType::Inout
    );
    SubmitContext mixed_context{};
    mixed_context.task_id = 1;
    mixed_context.won = true;
    mixed_context.register_mask = 3;
    mixed_context.result.task_id = 1;
    mixed_context.result.count = 1;
    mixed_context.result.tensors[0] = &next_output;
    mixed_context.shared_result.Reset(1);
    Check(
        mixed_context.shared_result.AddOutputRef(1, 0),
        "mixed transaction declares its own fresh output"
    );
    SharedTaskWriterDelta mixed_delta{};
    LocalStats mixed_stats{};
    Check(
        PrepareSharedTaskWriterDelta(
            mixed_args, mixed_context, mixed_delta
        ) &&
            mixed_delta.prepared_task_id == 1 &&
            mixed_delta.ordinary_count == 1 &&
            mixed_delta.symbol_count == 1 &&
            mixed_delta.writer_intent_required &&
            PublishSharedTaskWriterDelta<
                WriterIntentTestOps
            >(
                &state, mixed_context, mixed_delta,
                mixed_stats
            ),
        "one ordered transaction publishes ordinary, symbol and fresh metadata"
    );

    const SharedOutputCell &seed_cell =
        state.shared_map.shared_outputs[0];
    const SharedOutputCell &mixed_cell =
        state.shared_map.shared_outputs[1];
    const SharedWriterHistoryCell &history =
        state.shared_map.writer_history[1];
    Check(
        state.fatal.value == 0 &&
            InsertCompletionsMatch(state, 2) &&
            seed_cell.last_writer[0].value == 1 &&
            history.magic == kSharedWriterHistoryMagic &&
            history.writer_task == 1 &&
            history.count == 1 &&
            history.entries[0].symbol_key ==
                mixed_delta.symbol_keys[0] &&
            mixed_cell.published[0].value == 1 &&
            mixed_cell.last_writer[0].value == 1 &&
            mixed_stats.result.map_inserts == 1 &&
            mixed_stats.result.shared_symbol_inout_commits == 1 &&
            mixed_cell.tensors[0].buffer_addr ==
                next_output.buffer_addr,
        "mixed transaction exposes all metadata before task 1 completion"
    );

    bool lookup_ok = false;
    uint32_t lookup_count = UINT32_MAX;
    int32_t fanin[kMaxFanin] = {};
    const uint32_t mixed_fanin =
        CollectSharedFanin<
            WriterIntentTestOps, false, true
        >(
            state.shared_map, mixed_args, 1,
            static_cast<int32_t>(state.heap_window),
            mixed_stats, fanin, lookup_ok,
            lookup_count, &state.fatal.value
        );
    Check(
        lookup_ok && mixed_fanin == 1 &&
            fanin[0] == 0 && lookup_count == 1,
        "mixed transaction lookup deduplicates its two task-0 producers"
    );

    const FdwicOutputRef mixed_ref{1, 0, 0, 0, 0, 0};
    TaskArgs reader_args;
    ConstructTaskArgs(reader_args);
    AddOutputHandleTensor(
        reader_args, mixed_ref, TensorArgType::Input
    );
    SubmitContext reader_context{};
    reader_context.task_id = 2;
    reader_context.won = true;
    reader_context.result.task_id = 2;
    reader_context.shared_result.Reset(2);
    SharedTaskWriterDelta reader_delta{};
    LocalStats reader_stats{};
    Check(
        PrepareSharedTaskWriterDelta(
            reader_args, reader_context, reader_delta
        ) &&
            PublishSharedTaskWriterDelta<
                WriterIntentTestOps
            >(
                &state, reader_context, reader_delta,
                reader_stats
            ),
        "mixed transaction reader publishes an empty completion"
    );
    lookup_ok = false;
    lookup_count = UINT32_MAX;
    const uint32_t reader_fanin =
        CollectSharedFanin<
            WriterIntentTestOps, false, true
        >(
            state.shared_map, reader_args, 2,
            static_cast<int32_t>(state.heap_window),
            reader_stats, fanin, lookup_ok,
            lookup_count, &state.fatal.value
        );
    Check(
        lookup_ok && reader_fanin == 1 &&
            fanin[0] == 1 && lookup_count == 0,
        "downstream reader consumes the fresh output from mixed task 1"
    );
}

void TestStrictLatestFaninWindow(SchedulerState &state) {
    ResetProtocolState(state);
    constexpr int32_t kWindow = 2;
    state.heap_window = kWindow;

    // symbol origin 为 task 0，当前 latest writer 为 task 2。reader 4 的
    // 左边界恰好是 2，必须接收；reader 5 的左边界是 3，必须把同一
    // writer 当作窗口外历史，而不是继续塞进 fanin。
    SharedOutputCell &symbol = state.shared_map.shared_outputs[0];
    symbol.published[0].value = 0;
    symbol.last_writer[0].value = 2;
    SharedWriterHistoryCell &history =
        state.shared_map.writer_history[2];
    history.magic = kSharedWriterHistoryMagic;
    history.writer_task = 2;
    history.count = 1;
    history.reserved = 0;
    history.entries[0].symbol_key = 1;
    history.entries[0].previous_writer = 0;

    const FdwicOutputRef output{0, 0, 0, 0, 0, 0};
    TaskArgs symbol_args;
    ConstructTaskArgs(symbol_args);
    AddOutputHandleTensor(
        symbol_args, output, TensorArgType::Input
    );
    LocalStats symbol_stats{};
    int32_t fanin[kMaxFanin] = {};
    bool protocol_ok = false;
    uint32_t lookup_count = UINT32_MAX;
    const uint32_t boundary_count =
        CollectSharedFanin<
            WriterIntentTestOps, false, true
        >(
            state.shared_map, symbol_args, 4, kWindow,
            symbol_stats, fanin, protocol_ok, lookup_count,
            &state.fatal.value
        );
    Check(
        protocol_ok && boundary_count == 1 &&
            fanin[0] == 2 && lookup_count == 0,
        "strict symbol lookup accepts producer exactly at N-H"
    );
    protocol_ok = false;
    lookup_count = UINT32_MAX;
    const uint32_t expired_count =
        CollectSharedFanin<
            WriterIntentTestOps, false, true
        >(
            state.shared_map, symbol_args, 5, kWindow,
            symbol_stats, fanin, protocol_ok, lookup_count,
            &state.fatal.value
        );
    Check(
        protocol_ok && expired_count == 0 &&
            lookup_count == 0,
        "strict symbol lookup excludes producer below N-H"
    );

    // TensorDesc::owner_task_id 也属于 fanin 来源，必须使用同一窗口。
    // ordinary ring 为空，因此下面结果只由显式 owner 决定。
    TensorDesc owned = MakeTensor(
        0x480000000ULL, 1
    );
    TaskArgs ordinary_args;
    ConstructTaskArgs(ordinary_args);
    AddGmTensor(
        ordinary_args, owned, TensorArgType::Input
    );
    LocalStats ordinary_stats{};
    protocol_ok = false;
    lookup_count = UINT32_MAX;
    const uint32_t owner_boundary_count =
        CollectSharedFanin<
            WriterIntentTestOps, false, true
        >(
            state.shared_map, ordinary_args, 3, kWindow,
            ordinary_stats, fanin, protocol_ok, lookup_count,
            &state.fatal.value
        );
    Check(
        protocol_ok && owner_boundary_count == 1 &&
            fanin[0] == 1 && lookup_count == 1,
        "strict explicit owner accepts producer exactly at N-H"
    );
    protocol_ok = false;
    lookup_count = UINT32_MAX;
    const uint32_t owner_expired_count =
        CollectSharedFanin<
            WriterIntentTestOps, false, true
        >(
            state.shared_map, ordinary_args, 4, kWindow,
            ordinary_stats, fanin, protocol_ok, lookup_count,
            &state.fatal.value
        );
    Check(
        protocol_ok && owner_expired_count == 0 &&
            lookup_count == 1,
        "strict explicit owner excludes producer below N-H"
    );

    owned.owner_task_id = 4;
    protocol_ok = true;
    lookup_count = UINT32_MAX;
    const uint32_t self_count =
        CollectSharedFanin<
            WriterIntentTestOps, false, true
        >(
            state.shared_map, ordinary_args, 4, kWindow,
            ordinary_stats, fanin, protocol_ok, lookup_count,
            &state.fatal.value
        );
    Check(
        !protocol_ok && self_count == 0,
        "strict explicit owner rejects self and future producer"
    );
}

void TestOrderedPublishRejectsFullBucketAtomically(
    SchedulerState &state
) {
    ResetProtocolState(state);
    constexpr uint64_t kAddress = 0x490000000ULL;
    const uint32_t bucket = TensorMapHash(kAddress);
    bool filled = true;
    for (uint32_t cursor = 0;
         cursor < kMapBucketCapacity; ++cursor) {
        const SharedRegionValue entry{
            kAddress, 0, 4096,
            static_cast<int32_t>(cursor), 0
        };
        filled &=
            SharedAppendPreparedEntry<WriterIntentTestOps>(
                state.shared_map, entry
            );
    }
    Check(filled, "full-bucket setup publishes exactly CAP live entries");

    int64_t seq_before[kMapBucketCapacity] = {};
    SharedRegionValue payload_before[kMapBucketCapacity] = {};
    for (uint32_t cursor = 0;
         cursor < kMapBucketCapacity; ++cursor) {
        SharedRegionSlot &slot =
            state.shared_map.slots[
                SharedTensorMapSlotIndex(bucket, cursor)
            ];
        seq_before[cursor] = slot.seq.value;
        payload_before[cursor] = slot.payload.value;
    }
    const int64_t head_before =
        state.shared_map.buckets[bucket].head.value;
    const int64_t tail_before =
        state.shared_map.buckets[bucket].tail.value;

    constexpr int32_t kBlockedTask =
        static_cast<int32_t>(kMapBucketCapacity);
    SetInsertCompletionsAfterTasks(
        state,
        static_cast<uint32_t>(kBlockedTask)
    );
    TensorDesc tensor = MakeTensor(kAddress);
    TaskArgs args;
    ConstructTaskArgs(args);
    AddGmTensor(args, tensor, TensorArgType::OutputExisting);
    SharedOutputCell &symbol =
        state.shared_map.shared_outputs[0];
    symbol.published[0].value = 0;
    symbol.last_writer[0].value = 0;
    const FdwicOutputRef output{0, 0, 0, 0, 0, 0};
    AddOutputHandleTensor(
        args, output, TensorArgType::Inout
    );
    SubmitContext context{};
    context.task_id = kBlockedTask;
    context.won = true;
    context.register_mask = 3;
    context.result.task_id = kBlockedTask;
    TensorDesc fresh = MakeTensor(
        0x491000000ULL,
        static_cast<uint64_t>(kBlockedTask)
    );
    context.result.count = 1;
    context.result.tensors[0] = &fresh;
    context.shared_result.Reset(kBlockedTask);
    Check(
        context.shared_result.AddOutputRef(kBlockedTask, 0),
        "blocked mixed task declares a fresh output"
    );
    SharedTaskWriterDelta delta{};
    LocalStats stats{};
    Check(
        PrepareSharedTaskWriterDelta(args, context, delta) &&
            delta.ordinary_count == 1 &&
            delta.writer_intent_required,
        "blocked task prepares mixed ordinary and symbol writer intent"
    );
    Check(
        !PublishSharedTaskWriterDelta<WriterIntentTestOps>(
            &state, context, delta, stats
        ),
        "ordered publish rejects a full live bucket"
    );

    bool slots_unchanged = true;
    for (uint32_t cursor = 0;
         cursor < kMapBucketCapacity; ++cursor) {
        const SharedRegionSlot &slot =
            state.shared_map.slots[
                SharedTensorMapSlotIndex(bucket, cursor)
            ];
        slots_unchanged &=
            slot.seq.value == seq_before[cursor] &&
            std::memcmp(
                &slot.payload.value, &payload_before[cursor],
                sizeof(SharedRegionValue)
            ) == 0;
    }
    Check(
        state.fatal.value == 1 &&
            InsertCompletionsMatch(
                state,
                static_cast<uint32_t>(kBlockedTask)
            ) &&
            state.shared_map.buckets[bucket].head.value ==
                head_before &&
            state.shared_map.buckets[bucket].tail.value ==
                tail_before &&
            slots_unchanged &&
            symbol.published[0].value == 0 &&
            symbol.last_writer[0].value == 0 &&
            state.shared_map.writer_history[kBlockedTask]
                    .magic == 0 &&
            state.shared_map.shared_outputs[kBlockedTask]
                    .published[0].value == -1 &&
            state.shared_map.shared_outputs[kBlockedTask]
                    .last_writer[0].value == -1 &&
            state.shared_map.shared_outputs[kBlockedTask]
                    .tensors[0].buffer_addr == 0 &&
            stats.result.map_inserts == 0,
        "capacity failure keeps cursor, ring, symbol and fresh metadata unchanged"
    );
}

void TestCommitStatsRequireCompletionCas(
    SchedulerState &state
) {
    ResetProtocolState(state);
    SetInsertCompletionsAfterTasks(state, 1);

    TensorDesc ordinary = MakeTensor(0x492000000ULL);
    TaskArgs args;
    ConstructTaskArgs(args);
    AddGmTensor(
        args, ordinary, TensorArgType::OutputExisting
    );
    SharedOutputCell &symbol =
        state.shared_map.shared_outputs[0];
    symbol.published[0].value = 0;
    symbol.last_writer[0].value = 0;
    const FdwicOutputRef output{0, 0, 0, 0, 0, 0};
    AddOutputHandleTensor(
        args, output, TensorArgType::Inout
    );

    SubmitContext context{};
    context.task_id = 1;
    context.won = true;
    context.register_mask = 3;
    context.result.task_id = 1;
    context.shared_result.Reset(1);
    SharedTaskWriterDelta delta{};
    LocalStats stats{};
    Check(
        PrepareSharedTaskWriterDelta(
            args, context, delta
        ) &&
            delta.prepared_task_id == 1 &&
            delta.ordinary_count == 1 &&
            delta.symbol_count == 1,
        "completion failure test prepares a mixed writer delta"
    );

    // metadata 可以完整落地，但 task-level completion CAS 必须因非法旧值
    // 失败。成功统计只描述完成事务，不能把这个 terminal 前缀算进去。
    state.tasks[1].deps_prepared = 77;
    const uint32_t bucket = TensorMapHash(ordinary.buffer_addr);
    Check(
        !PublishSharedTaskWriterDelta<WriterIntentTestOps>(
            &state, context, delta, stats
        ),
        "completion CAS conflict rejects the ordered transaction"
    );
    Check(
        state.fatal.value == 1 &&
            state.tasks[1].deps_prepared == 77 &&
            state.shared_map.buckets[bucket].tail.value == 1 &&
            symbol.last_writer[0].value == 1 &&
            state.shared_map.writer_history[1].count == 1 &&
            stats.result.map_inserts == 0 &&
            stats.result.shared_symbol_inout_commits == 0,
        "failed completion keeps metadata evidence but records no committed writers"
    );
}

void TestOrdinaryWriterRangeValidation() {
    SharedRegionValue region{};
    TensorDesc contiguous = MakeTensor(0x470000000ULL);
    contiguous.ndims = 2;
    contiguous.shapes[0] = 65535;
    contiguous.shapes[1] = 65537;
    Check(
        MakeValidatedSharedWriterRegion(
            contiguous, 200, region
        ) &&
            region.lo == 0 &&
            region.hi ==
                static_cast<uint64_t>(65535) * 65537 * 4,
        "ordinary contiguous range accepts a uint32-representable shape product"
    );

    contiguous.shapes[0] = 65536;
    contiguous.shapes[1] = 65536;
    Check(
        MakeValidatedSharedWriterRegion(
            contiguous, 200, region
        ) &&
            region.hi == (uint64_t{1} << 34),
        "ordinary contiguous range preserves a valid extent above uint32"
    );

    contiguous.ndims = 3;
    contiguous.shapes[0] = UINT32_MAX;
    contiguous.shapes[1] = UINT32_MAX;
    contiguous.shapes[2] = 2;
    Check(
        !MakeValidatedSharedWriterRegion(
            contiguous, 200, region
        ),
        "ordinary contiguous range rejects a true uint64 shape-product overflow"
    );

    TensorDesc noncontiguous = MakeTensor(0x480000000ULL);
    noncontiguous.is_contiguous = false;
    noncontiguous.dtype = DataType::Uint8;
    noncontiguous.extent_elem_cache =
        static_cast<uint64_t>(UINT32_MAX) + 1;
    Check(
        MakeValidatedSharedWriterRegion(
            noncontiguous, 201, region
        ) &&
            region.lo == 0 &&
            region.hi ==
                static_cast<uint64_t>(UINT32_MAX) + 1,
        "ordinary noncontiguous range preserves its uint64 cached extent"
    );

    noncontiguous.dtype = DataType::Uint64;
    noncontiguous.start_offset = UINT64_MAX / 8;
    noncontiguous.extent_elem_cache = 1;
    Check(
        !MakeValidatedSharedWriterRegion(
            noncontiguous, 201, region
        ),
        "ordinary byte range rejects an end offset that overflows after dtype scaling"
    );
}

void TestManualWriterNeedsNoGate(SchedulerState &state) {
    constexpr int32_t kTask = 160;
    ResetTaskGate(state, kTask);
    TensorDesc manual = MakeTensor(
        0x430000000ULL, kInvalidTaskId, true
    );
    const uint32_t bucket = TensorMapHash(manual.buffer_addr);
    const int64_t tail_before =
        state.shared_map.buckets[bucket].tail.value;

    TaskArgs args;
    ConstructTaskArgs(args);
    AddGmTensor(args, manual, TensorArgType::Inout);
    bool required = true;
    Check(
        InspectSharedWriterIntent(args, required) &&
            !required,
        "manual_dep writer is excluded from automatic intent"
    );
    SubmitContext context{};
    context.task_id = kTask;
    context.won = true;
    LocalStats stats{};
    Check(
        PrepareSharedWriterIntentSet<WriterIntentTestOps>(
            &state, args, context, stats
        ) == SharedWriterIntentResult::NotRequired,
        "manual_dep writer returns without publishing a gate"
    );
    Check(
        state.tasks[kTask].deps_prepared == -1 &&
            state.shared_map.buckets[bucket].tail.value ==
                tail_before,
        "manual_dep writer leaves gate and ordinary map untouched"
    );
}

}  // namespace

int main() {
    SchedulerState *state = MapSparseSchedulerState();
    if (state == nullptr) {
        return 1;
    }
    ResetProtocolState(*state);
    TestReaderProgressStateMachine(*state);
    ResetProtocolState(*state);
    TestSymbolWriterIntentChain(*state);
    TestOrdinaryWriterIntentChain(*state);
    TestWriterDeltaRequiresExactRegisterMask();
    TestWriterDeltaPrecomputesInterleavedBuckets(*state);
    TestOrderedOrdinaryInsertBeforeLookup(*state);
    TestOrderedSymbolInsertBeforeLookup(*state);
    TestOrderedMixedWriterTransaction(*state);
    TestStrictLatestFaninWindow(*state);
    TestOrdinaryWriterRangeValidation();
    TestManualWriterNeedsNoGate(*state);
    const bool fatal_clean = state->fatal.value == 0;
    Check(fatal_clean, "all positive paths leave fatal clear");
    TestCommitStatsRequireCompletionCas(*state);
    TestOrderedPublishRejectsFullBucketAtomically(*state);
    TestOutOfOrderSymbolWriterFailsClosed(*state);
    TestMultiSymbolConflictKeepsTerminalPrefix(*state);
    UnmapSparseSchedulerState(state);
    if (g_failures != 0) {
        std::fprintf(
            stderr, "[FAIL] shared writer intent failures=%d\n",
            g_failures
        );
        return 1;
    }
    std::printf(
        "[PASS] generic shared reader progress, symbol history, "
        "terminal-prefix, and ordinary writer-intent tests\n"
    );
    return 0;
}
