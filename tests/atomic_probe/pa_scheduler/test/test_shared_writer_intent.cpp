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

void TestSymbolWriterIntentChain(SchedulerState &state) {
    constexpr int32_t kExistingDependency = 5;
    constexpr int32_t kProducer = 10;
    constexpr int32_t kWriter = 20;
    constexpr int32_t kReader = 30;
    ResetTaskGate(state, kWriter);

    SharedOutputCell &cell =
        state.shared_map.shared_outputs[kProducer];
    cell.published[0].value = -1;
    cell.last_writer[0].value = -1;

    TensorDesc produced = MakeTensor(
        0x410000000ULL,
        static_cast<uint64_t>(kProducer)
    );
    SubmitContext producer_context{};
    producer_context.task_id = kProducer;
    producer_context.result.task_id = kProducer;
    producer_context.result.count = 1;
    producer_context.result.tensors[0] = &produced;
    producer_context.shared_result.Reset(kProducer);
    Check(
        producer_context.shared_result.AddOutputRef(
            kProducer, 0
        ),
        "symbol producer accepts slot 0"
    );
    Check(
        PublishSharedTaskOutputs<WriterIntentTestOps>(
            state.shared_map, producer_context, kProducer
        ),
        "symbol producer publishes descriptor"
    );

    const FdwicOutputRef output_ref =
        producer_context.shared_result.OutputRef(0);
    TaskArgs writer_args;
    ConstructTaskArgs(writer_args);
    AppendSharedOutputRef(
        writer_args, output_ref, TensorArgType::Inout
    );
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
        cell.last_writer[0].value == kWriter &&
            state.tasks[kWriter].deps_prepared == kWriter,
        "symbol writer metadata precedes the ready gate"
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
        "symbol reader resolves the latest writer after the gate"
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
    TestSymbolWriterIntentChain(*state);
    TestOrdinaryWriterIntentChain(*state);
    TestManualWriterNeedsNoGate(*state);
    const bool fatal_clean = state->fatal.value == 0;
    Check(fatal_clean, "all positive paths leave fatal clear");
    UnmapSparseSchedulerState(state);
    if (g_failures != 0) {
        std::fprintf(
            stderr, "[FAIL] shared writer intent failures=%d\n",
            g_failures
        );
        return 1;
    }
    std::printf(
        "[PASS] generic shared symbol/ordinary A->B->C "
        "writer-intent tests\n"
    );
    return 0;
}
