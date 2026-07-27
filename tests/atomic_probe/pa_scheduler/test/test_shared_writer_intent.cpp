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
    state.shared_map.committed_tasks.value = 0;
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
        state.shared_map.committed_tasks.value == 0,
        "non-adjacent ordinary writers do not require exact task turn"
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
    TestOrdinaryWriterRangeValidation();
    TestManualWriterNeedsNoGate(*state);
    const bool fatal_clean = state->fatal.value == 0;
    Check(fatal_clean, "all positive paths leave fatal clear");
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
