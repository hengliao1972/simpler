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
#include <memory>
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
    std::fprintf(stderr, "[FAIL] %s\n", message);
    ++g_failures;
}

// 该 Ops 只验证公共 symbol helper 的原子状态机和 descriptor 搬运。
// fence 不模拟 A5 DCache；设备缓存可见性仍必须由 CCEC 上板门禁证明。
struct SymbolTestOps {
    static constexpr bool kAtomicReturnReadyObserved = false;
    static volatile int64_t *wait_address;
    static std::atomic<uint64_t> wait_loads;
    static std::atomic<uint64_t> now_calls;
    static std::atomic<uint64_t> spin_calls;

    static int32_t Load(volatile int32_t *address) {
        return __atomic_fetch_add(address, int32_t{0}, __ATOMIC_ACQUIRE);
    }

    static int64_t Load(volatile int64_t *address) {
        if (address == wait_address) {
            wait_loads.fetch_add(1, std::memory_order_release);
        }
        return __atomic_fetch_add(address, int64_t{0}, __ATOMIC_ACQUIRE);
    }

    static int32_t Exchange(volatile int32_t *address, int32_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t Exchange(volatile int64_t *address, int64_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static uint64_t Exchange(volatile uint64_t *address, uint64_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t CompareExchange(
        volatile int64_t *address, int64_t expected, int64_t desired
    ) {
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(
            address, &observed, desired, false,
            __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        );
        return observed;
    }

    static int64_t FetchMax(
        volatile int64_t *address, int64_t value, uint64_t &retries
    ) {
        int64_t current = __atomic_load_n(address, __ATOMIC_ACQUIRE);
        retries = 0;
        while (current < value) {
            if (__atomic_compare_exchange_n(
                    address, &current, value, true, __ATOMIC_ACQ_REL,
                    __ATOMIC_ACQUIRE
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
        now_calls.fetch_add(1, std::memory_order_relaxed);
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

    static void SpinHint() {
        spin_calls.fetch_add(1, std::memory_order_relaxed);
        std::this_thread::yield();
    }
};

volatile int64_t *SymbolTestOps::wait_address = nullptr;
std::atomic<uint64_t> SymbolTestOps::wait_loads{0};
std::atomic<uint64_t> SymbolTestOps::now_calls{0};
std::atomic<uint64_t> SymbolTestOps::spin_calls{0};

// 把第二次读钟直接推进到 watchdog 期限之后，避免用真实 2 秒等待测试
// timeout 终止语义。
struct ExpiredWaitOps : SymbolTestOps {
    static std::atomic<uint64_t> calls;

    static uint64_t Now() {
        const uint64_t call = calls.fetch_add(1, std::memory_order_relaxed);
        return call == 0 ? 0 : kWatchdogTicks + 1;
    }

    static void SpinHint() {}
};

std::atomic<uint64_t> ExpiredWaitOps::calls{0};

// 只在定向测试中模拟“预检后、atomic 执行前”出现的协议异常，覆盖正常
// 单写者 Case1 不会命中的冷回滚分支。
struct PublicationFaultOps : SymbolTestOps {
    using SymbolTestOps::Exchange;

    static volatile int64_t *fetch_race_address;
    static volatile int64_t *exchange_race_address;

    static int64_t FetchMax(
        volatile int64_t *address, int64_t value, uint64_t &retries
    ) {
        if (address == fetch_race_address) {
            __atomic_store_n(address, int64_t{-2}, __ATOMIC_RELEASE);
            fetch_race_address = nullptr;
        }
        return SymbolTestOps::FetchMax(address, value, retries);
    }

    static int64_t Exchange(volatile int64_t *address, int64_t value) {
        if (address == exchange_race_address) {
            __atomic_store_n(address, int64_t{7}, __ATOMIC_RELEASE);
            exchange_race_address = nullptr;
        }
        return SymbolTestOps::Exchange(address, value);
    }
};

volatile int64_t *PublicationFaultOps::fetch_race_address = nullptr;
volatile int64_t *PublicationFaultOps::exchange_race_address = nullptr;

struct SealOrderOps : SymbolTestOps {
    using SymbolTestOps::Exchange;

    static volatile int64_t *output_writer_address;
    static volatile int64_t *published_address;
    static int64_t expected_writer;
    static bool order_ok;

    static int64_t Exchange(volatile int64_t *address, int64_t value) {
        if (address == published_address) {
            order_ok &=
                __atomic_load_n(output_writer_address, __ATOMIC_ACQUIRE) ==
                    expected_writer;
        }
        return SymbolTestOps::Exchange(address, value);
    }
};

volatile int64_t *SealOrderOps::output_writer_address = nullptr;
volatile int64_t *SealOrderOps::published_address = nullptr;
int64_t SealOrderOps::expected_writer = 0;
bool SealOrderOps::order_ok = true;

void ResetSharedState(SharedTensorMapSidecar &map) {
    std::memset(&map, 0, sizeof(map));
    InitializeSharedInsertTurns(map);
    map.reclaim_upto.value = -1;
    for (uint32_t index = 0; index < kMapCapacity; ++index) {
        map.slots[index].seq.value = -1;
    }
    for (uint32_t task = 0; task < kMaxTasks; ++task) {
        for (uint32_t slot = 0; slot < kSharedOutputMaxPerTask; ++slot) {
            map.shared_outputs[task].published[slot].value = -1;
            map.shared_outputs[task].last_writer[slot].value = -1;
        }
    }
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        map.reader_done[worker].value = -1;
    }
}

SchedulerState *MapSparseSchedulerState() {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void *memory = mmap(
        nullptr, sizeof(SchedulerState), PROT_READ | PROT_WRITE,
        flags, -1, 0
    );
    if (memory == MAP_FAILED) {
        std::perror("mmap SchedulerState");
        return nullptr;
    }
    // SchedulerState 是 trivial 类型；匿名映射提供零页，default-init
    // 只建立对象生命周期，避免为定向测试提交约 1 GiB 无关物理页。
    return ::new (memory) SchedulerState;
}

void UnmapSparseSchedulerState(SchedulerState *state) {
    if (state != nullptr) {
        (void)munmap(state, sizeof(SchedulerState));
    }
}

void TestSharedCompletionPublishesWithoutFrontier() {
    SchedulerState *state = MapSparseSchedulerState();
    if (state == nullptr) {
        ++g_failures;
        return;
    }
    state->frontier.value = -1;
    WorkerState &worker = state->workers[0];
    worker.heap_next = 4096;
    LocalStats stats{};

    CompleteTask<SymbolTestOps>(state, worker, 0, stats);

    Check(
        state->tasks[0].vend == worker.heap_next &&
            state->tasks[0].flag == 1,
        "shared completion publishes both vend and ready flag"
    );
    Check(
        state->frontier.value == -1,
        "shared no-wrap completion leaves frontier at its initial value"
    );
    Check(
        stats.result.frontier_initial_loads == 0 &&
            stats.result.frontier_updates == 0 &&
            stats.result.frontier_terminal_loads == 0,
        "shared no-wrap completion performs no frontier helping"
    );
    Check(
        stats.result.cas_retries == 0,
        "shared completion adds no hidden frontier CAS retries"
    );
    UnmapSparseSchedulerState(state);
}

TensorDesc MakeTensor(uint64_t address, uint32_t owner) {
    TensorDesc tensor{};
    tensor.buffer_addr = address;
    tensor.buffer_size = 4096;
    tensor.owner_task_id = owner;
    tensor.ndims = 1;
    tensor.dtype = DataType::Float32;
    tensor.is_contiguous = true;
    tensor.shapes[0] = 1024;
    tensor.strides[0] = 1;
    tensor.extent_elem_cache = 1024;
    return tensor;
}

bool SameTensor(const TensorDesc &left, const TensorDesc &right) {
    return std::memcmp(&left, &right, sizeof(TensorDesc)) == 0;
}

bool AllBytesEqual(const void *object, size_t size, unsigned char expected) {
    const unsigned char *bytes =
        reinterpret_cast<const unsigned char *>(object);
    for (size_t index = 0; index < size; ++index) {
        if (bytes[index] != expected) {
            return false;
        }
    }
    return true;
}

void TestPublishAndResolve() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetSharedState(*map);
    SymbolTestOps::now_calls.store(0, std::memory_order_relaxed);

    TensorDesc first = MakeTensor(0x100000000ULL, 0);
    TensorDesc second = MakeTensor(0x100001000ULL, 0);
    SubmitContext producer{};
    producer.task_id = 0;
    producer.result.task_id = 0;
    producer.result.count = 2;
    producer.result.tensors[0] = &first;
    producer.result.tensors[1] = &second;
    producer.shared_result.Reset(0);
    Check(producer.shared_result.AddOutputRef(0, 0), "producer accepts output slot 0");
    Check(producer.shared_result.AddOutputRef(0, 1), "producer accepts output slot 1");

    Check(
        PublishSharedTaskOutputs<SymbolTestOps>(*map, producer, 0),
        "first publication succeeds"
    );
    Check(map->shared_outputs[0].published[0].value == 0, "slot 0 is published");
    Check(map->shared_outputs[0].published[1].value == 0, "slot 1 is published");
    Check(map->shared_outputs[0].last_writer[0].value == 0, "slot 0 writer starts at producer");
    Check(SameTensor(map->shared_outputs[0].tensors[0], first), "published descriptor is exact");
    TensorDesc replacement = MakeTensor(0x200000000ULL, 0);
    producer.result.tensors[0] = &replacement;
    Check(
        !PublishSharedTaskOutputs<SymbolTestOps>(*map, producer, 0),
        "duplicate publication fails closed"
    );
    Check(
        SameTensor(map->shared_outputs[0].tensors[0], first),
        "failed duplicate publication cannot replace descriptor"
    );
    Check(
        map->shared_outputs[0].published[0].value == 0 &&
            map->shared_outputs[0].last_writer[0].value == 0,
        "failed duplicate publication preserves control state"
    );
    producer.result.tensors[0] = &first;

    TaskArgs input_args;
    ConstructTaskArgs(input_args);
    AppendSharedOutputRef(
        input_args, producer.shared_result.OutputRef(0), TensorArgType::Input
    );
    auto input_payload = std::make_unique<TaskPayload>();
    std::memset(input_payload.get(), 0xA5, sizeof(*input_payload));
    SubmitContext input_context{};
    input_context.task_id = 1;
    input_context.payload = input_payload.get();
    input_context.tensor_count = input_args.tensor_count;
    input_context.scalar_count = input_args.scalar_count;
    LocalStats input_stats{};
    int32_t input_fanin[kMaxFanin] = {};
    bool protocol_ok = false;
    uint32_t ordinary_lookups = UINT32_MAX;
    const uint32_t input_count = CollectSharedFanin<SymbolTestOps>(
        *map, input_args, 1, kHeapWindow, input_stats,
        input_fanin, protocol_ok, ordinary_lookups
    );
    Check(protocol_ok, "plain symbolic INPUT resolves");
    Check(input_count == 1 && input_fanin[0] == 0, "INPUT depends on producer writer");
    Check(ordinary_lookups == 0, "symbol INPUT never enters region map");
    Check(
        input_stats.result.shared_symbol_input_loads == 1,
        "INPUT writer load is counted"
    );
    Check(
        AllBytesEqual(input_payload.get(), sizeof(*input_payload), 0xA5),
        "resolver leaves task payload scratch untouched"
    );
    LocalSlot input_slot{};
    BuildSlotPayload<SymbolTestOps, true>(
        input_slot, 1, static_cast<uint32_t>(FunctionId(TaskKind::Qk)), 0,
        input_args, input_context, input_fanin, input_count, *map
    );
    Check(
        SameTensor(input_slot.tensors[0], first),
        "validated INPUT descriptor lands directly in LocalSlot"
    );
    Check(
        input_slot.args[0] ==
            static_cast<uint64_t>(
                reinterpret_cast<uintptr_t>(&input_slot.tensors[0])
            ),
        "direct INPUT slot descriptor is wired into dispatch args"
    );
    input_payload->tensors[0] = first;
    LocalSlot compatibility_slot{};
    BuildSlotPayload(
        compatibility_slot, input_args, input_context, input_fanin,
        input_count
    );
    Check(
        SameTensor(compatibility_slot.tensors[0], first),
        "legacy prefilled-payload builder contract remains available"
    );

    TaskArgs inout_args;
    ConstructTaskArgs(inout_args);
    AppendSharedOutputRef(
        inout_args, producer.shared_result.OutputRef(0), TensorArgType::Inout
    );
    auto inout_payload = std::make_unique<TaskPayload>();
    std::memset(inout_payload.get(), 0x5A, sizeof(*inout_payload));
    SubmitContext inout_context{};
    inout_context.task_id = 2;
    inout_context.payload = inout_payload.get();
    inout_context.tensor_count = inout_args.tensor_count;
    inout_context.scalar_count = inout_args.scalar_count;
    LocalStats inout_stats{};
    int32_t inout_fanin[kMaxFanin] = {};
    protocol_ok = false;
    ordinary_lookups = UINT32_MAX;
    const uint32_t inout_count = CollectSharedFanin<SymbolTestOps>(
        *map, inout_args, 2, kHeapWindow, inout_stats,
        inout_fanin, protocol_ok, ordinary_lookups
    );
    Check(protocol_ok, "plain symbolic INOUT resolves");
    Check(inout_count == 1 && inout_fanin[0] == 0, "INOUT consumes old writer");
    Check(
        map->shared_outputs[0].last_writer[0].value == 0,
        "read-only INOUT resolve keeps producer writer unchanged"
    );
    Check(
        inout_stats.result.shared_symbol_inout_commits == 0,
        "read-only INOUT resolve publishes no writer statistic"
    );
    Check(
        AllBytesEqual(inout_payload.get(), sizeof(*inout_payload), 0x5A),
        "INOUT resolver also leaves task payload scratch untouched"
    );
    LocalSlot inout_slot{};
    BuildSlotPayload<SymbolTestOps, true>(
        inout_slot, 2, static_cast<uint32_t>(FunctionId(TaskKind::Sf)), 0,
        inout_args, inout_context, inout_fanin, inout_count, *map
    );
    Check(
        SameTensor(inout_slot.tensors[0], first),
        "validated INOUT descriptor lands directly in LocalSlot"
    );
    Check(
        CommitSharedFaninWriters<SymbolTestOps>(
            *map, inout_args, 2, inout_stats
        ),
        "explicit post-build step commits the INOUT writer"
    );
    Check(
        map->shared_outputs[0].last_writer[0].value == 2,
        "INOUT commit advances writer to current task"
    );
    Check(
        inout_stats.result.shared_symbol_inout_commits == 1,
        "successful INOUT writer commit is counted"
    );

    // descriptor identity 仍指向最初 producer，但后继必须依赖最近一次
    // 已提交的 INOUT writer。writer-ready gate 负责阻止 loser 在该提交
    // 之前进入后继；resolver 本身据 last_writer 返回精确的新依赖。
    TaskArgs successor_args;
    ConstructTaskArgs(successor_args);
    AppendSharedOutputRef(
        successor_args, producer.shared_result.OutputRef(0),
        TensorArgType::Input
    );
    LocalStats successor_stats{};
    int32_t successor_fanin[kMaxFanin] = {};
    protocol_ok = false;
    ordinary_lookups = UINT32_MAX;
    const uint32_t successor_count = CollectSharedFanin<SymbolTestOps, true>(
        *map, successor_args, 3, kHeapWindow,
        successor_stats, successor_fanin, protocol_ok, ordinary_lookups,
        nullptr, 0, 2
    );
    Check(protocol_ok, "post-INOUT successor resolves the same shared descriptor");
    Check(
        successor_count == 1 && successor_fanin[0] == 2,
        "post-INOUT successor depends on the latest writer"
    );
    Check(
        SymbolTestOps::now_calls.load(std::memory_order_relaxed) == 0,
        "already-published symbol fast path never reads the watchdog clock"
    );
}

void TestPaTwoGroupWriterReadyGate() {
    SchedulerState *state = MapSparseSchedulerState();
    if (state == nullptr) {
        ++g_failures;
        return;
    }
    ResetSharedState(state->shared_map);
    state->fatal.value = 0;
    constexpr int32_t alloc = 0;
    constexpr int32_t first_sf = 2;
    constexpr int32_t first_pv = 3;
    constexpr int32_t first_up = 4;
    constexpr int32_t second_sf = 6;
    constexpr int32_t second_pv = 7;
    constexpr int32_t second_up = 8;
    state->tasks[first_up].deps_prepared = -1;

    const auto OutputRef = [](int32_t producer, int16_t slot) {
        return FdwicOutputRef{producer, slot, 0, 0, 0, 0};
    };
    const auto SeedOutput = [&](int32_t producer, int16_t slot, uint64_t address) {
        SharedOutputCell &cell =
            state->shared_map.shared_outputs[
                static_cast<uint32_t>(producer)
            ];
        cell.published[static_cast<uint32_t>(slot)].value = producer;
        cell.last_writer[static_cast<uint32_t>(slot)].value = producer;
        cell.tensors[static_cast<uint32_t>(slot)] =
            MakeTensor(address, static_cast<uint32_t>(producer));
    };
    SeedOutput(alloc, 0, 0x310000000ULL);
    SeedOutput(alloc, 1, 0x310001000ULL);
    SeedOutput(alloc, 2, 0x310002000ULL);
    SeedOutput(first_sf, 1, 0x320001000ULL);
    SeedOutput(first_sf, 2, 0x320002000ULL);
    SeedOutput(first_pv, 0, 0x330000000ULL);
    SeedOutput(second_sf, 1, 0x340001000ULL);
    SeedOutput(second_sf, 2, 0x340002000ULL);
    SeedOutput(second_pv, 0, 0x350000000ULL);

    PaOrchestrationState first_orchestration{};
    InitPaOrchestration(first_orchestration, 1, nullptr);
    first_orchestration.current_batch = 0;
    first_orchestration.current_sequence =
        2ULL * kPaBlocksPerRequest * kPaBlockSize;
    first_orchestration.current_blocks = 2ULL * kPaBlocksPerRequest;
    PreparePaBlockGroup(first_orchestration, 0);
    first_orchestration.accumulated_output = OutputRef(alloc, 0);
    first_orchestration.accumulated_sum = OutputRef(alloc, 1);
    first_orchestration.accumulated_max = OutputRef(alloc, 2);
    first_orchestration.sf_max = OutputRef(first_sf, 1);
    first_orchestration.sf_sum = OutputRef(first_sf, 2);
    first_orchestration.pv_output = OutputRef(first_pv, 0);

    TaskArgs first_args;
    LocalStats first_build_stats{};
    Check(
        BuildCallbackSubmitArgs<TaskKind::Up>(
            first_orchestration, first_args, 0, first_build_stats
        ),
        "first PA group builds the real UP argument shape"
    );
    Check(
        first_args.tensor_count == 7 && first_args.scalar_count == 2 &&
            TaskTag(first_args, 0) == TensorArgType::Input &&
            TaskTag(first_args, 1) == TensorArgType::Input &&
            TaskTag(first_args, 2) == TensorArgType::Input &&
            TaskTag(first_args, 3) == TensorArgType::Inout &&
            TaskTag(first_args, 4) == TensorArgType::Inout &&
            TaskTag(first_args, 5) == TensorArgType::Inout &&
            TaskTag(first_args, 6) == TensorArgType::Inout,
        "UP args contain 3 fresh INPUTs, 3 accumulator INOUTs, and output view"
    );
    const bool up_refs_have_expected_kind =
        first_args.tensors[3].kind ==
            TensorRefKind::SharedOutputRef &&
        first_args.tensors[4].kind ==
            TensorRefKind::SharedOutputRef &&
        first_args.tensors[5].kind ==
            TensorRefKind::SharedOutputRef;
    const FdwicOutputRef up_max = up_refs_have_expected_kind
        ? SharedOutputReference(first_args.tensors[3])
        : InvalidSharedOutputRef();
    const FdwicOutputRef up_sum = up_refs_have_expected_kind
        ? SharedOutputReference(first_args.tensors[4])
        : InvalidSharedOutputRef();
    const FdwicOutputRef up_output = up_refs_have_expected_kind
        ? SharedOutputReference(first_args.tensors[5])
        : InvalidSharedOutputRef();
    Check(
        up_refs_have_expected_kind &&
            up_max.producer_task_id == alloc &&
            up_max.output_slot == 2 &&
            up_sum.producer_task_id == alloc &&
            up_sum.output_slot == 1 &&
            up_output.producer_task_id == alloc &&
            up_output.output_slot == 0 &&
            first_args.tensors[6].kind ==
                TensorRefKind::LocalTensor &&
            first_args.tensors[6].pointer.local_tensor != nullptr &&
            first_args.tensors[6]
                .pointer.local_tensor->manual_dep,
        "UP builder fixes accumulator writer order to max/sum/output slots 2/1/0"
    );
    SubmitContext delta_context{};
    delta_context.task_id = first_up;
    delta_context.won = true;
    delta_context.register_mask = 0x78U;
    delta_context.result.task_id = first_up;
    delta_context.shared_result.Reset(first_up);
    SharedTaskWriterDelta exact_delta{};
    const uint32_t alloc_key_base =
        static_cast<uint32_t>(alloc) *
            kSharedOutputMaxPerTask +
        1U;
    Check(
        PrepareSharedTaskWriterDelta(
            first_args, delta_context, exact_delta
        ) &&
            ValidatePreparedPaWriterShape(
                exact_delta, TaskKind::Up, first_up,
                alloc, alloc
            ) &&
            exact_delta.ordinary_count == 0 &&
            exact_delta.symbol_count == 3 &&
            exact_delta.symbol_keys[0] ==
                alloc_key_base + 2U &&
            exact_delta.symbol_keys[1] ==
                alloc_key_base + 1U &&
            exact_delta.symbol_keys[2] ==
                alloc_key_base &&
            exact_delta.writer_intent_required,
        "UP writer delta preserves the exact 2/1/0 callback order"
    );
    SharedTaskWriterDelta malformed_delta = exact_delta;
    const uint32_t saved_middle_key =
        malformed_delta.symbol_keys[1];
    malformed_delta.symbol_keys[1] =
        malformed_delta.symbol_keys[2];
    malformed_delta.symbol_keys[2] = saved_middle_key;
    SharedTaskWriterDelta empty_delta{};
    empty_delta.prepared_task_id = first_sf;
    Check(
        !ValidatePreparedPaWriterShape(
            malformed_delta, TaskKind::Up, first_up,
            alloc, alloc
        ) &&
            !ValidatePreparedPaWriterShape(
                exact_delta, TaskKind::Qk, first_up,
                /*expected_previous=*/-1, alloc
            ) &&
            ValidatePreparedPaWriterShape(
                empty_delta, TaskKind::Sf, first_sf,
                /*expected_previous=*/-1, alloc
            ),
        "PA writer prevalidation rejects malformed UP and accepts an empty non-UP"
    );
    Check(
        first_args.scalars[0] == 1 && first_args.scalars[1] == 0,
        "first PA group carries is_first=1 and is_last=0"
    );

    PaOrchestrationState second_orchestration = first_orchestration;
    PreparePaBlockGroup(second_orchestration, kPaBlocksPerRequest);
    second_orchestration.sf_max = OutputRef(second_sf, 1);
    second_orchestration.sf_sum = OutputRef(second_sf, 2);
    second_orchestration.pv_output = OutputRef(second_pv, 0);
    TaskArgs second_args;
    std::atomic<bool> waiter_finished{false};
    std::atomic<bool> second_build_started{false};
    bool waiter_ok = false;
    bool second_build_ok = false;
    bool second_protocol_ok = false;
    uint32_t second_ordinary_lookups = UINT32_MAX;
    uint32_t second_fanin_count = 0;
    int32_t second_fanin[kMaxFanin] = {};
    LocalStats waiter_stats{};
    LocalStats second_build_stats{};
    LocalStats second_stats{};
    SymbolTestOps::wait_address = &state->tasks[first_up].deps_prepared;
    SymbolTestOps::wait_loads.store(0, std::memory_order_relaxed);
    std::thread loser([&]() {
        waiter_ok = WaitForSharedWriterReady<SymbolTestOps>(
            state, first_up, waiter_stats
        );
        if (waiter_ok) {
            second_build_started.store(true, std::memory_order_release);
            second_build_ok = BuildCallbackSubmitArgs<TaskKind::Up>(
                second_orchestration, second_args, 0, second_build_stats
            );
            if (second_build_ok) {
                second_fanin_count =
                    CollectSharedFanin<SymbolTestOps, true>(
                        state->shared_map, second_args, second_up,
                        kHeapWindow, second_stats, second_fanin,
                        second_protocol_ok, second_ordinary_lookups,
                        nullptr, alloc, first_up
                    );
            }
        }
        waiter_finished.store(true, std::memory_order_release);
    });
    while (SymbolTestOps::wait_loads.load(std::memory_order_acquire) == 0) {
        std::this_thread::yield();
    }
    for (uint32_t spin = 0; spin < 1024; ++spin) {
        std::this_thread::yield();
    }
    Check(
        !waiter_finished.load(std::memory_order_acquire) &&
            !second_build_started.load(std::memory_order_acquire),
        "UP loser cannot build the next group before writer intent is ready"
    );

    LocalStats first_stats{};
    SubmitContext first_context{};
    first_context.task_id = first_up;
    first_context.won = true;
    Check(
        PreparePaSharedWriterIntent<SymbolTestOps>(
            state, first_args, first_context, first_stats
        ),
        "first non-final UP prepares writer intent before winner Build"
    );
    Check(
        first_context.fanin_count == 3 &&
            first_context.fanin[0] == first_sf &&
            first_context.fanin[1] == first_pv &&
            first_context.fanin[2] == alloc &&
            first_stats.result.map_lookups == 0,
        "first UP intent resolves SF/PV/Alloc and excludes manual output view"
    );
    loser.join();
    SymbolTestOps::wait_address = nullptr;
    Check(
        waiter_ok && waiter_finished.load(std::memory_order_acquire) &&
            second_build_started.load(std::memory_order_acquire),
        "UP loser observes the exact gate and then builds the second group"
    );
    Check(
        state->tasks[first_up].flag == 0,
        "writer-ready does not impersonate the first UP completion flag"
    );
    Check(
        second_build_ok && second_args.tensor_count == 7 &&
            second_args.scalar_count == 2 &&
            second_args.scalars[0] == 0 &&
            second_args.scalars[1] == 1,
        "second PA group builds real UP args with is_first=0 and is_last=1"
    );
    Check(
        second_protocol_ok && second_fanin_count == 3 &&
            second_fanin[0] == second_sf &&
            second_fanin[1] == second_pv &&
            second_fanin[2] == first_up,
        "second UP keeps fresh SF/PV producers and uses first UP for accumulators"
    );
    Check(
        second_ordinary_lookups == 0 &&
            first_stats.result.shared_symbol_input_loads == 3 &&
            second_stats.result.shared_symbol_input_loads == 3,
        "both UP intents resolve three symbolic INPUTs and no ordinary region"
    );
    Check(
        !PublishSharedWriterReady<SymbolTestOps>(state, first_up),
        "duplicate writer-ready publication fails closed"
    );
    constexpr int32_t wrong_gate_task = 5;
    state->tasks[wrong_gate_task].deps_prepared = first_up;
    Check(
        !PublishSharedWriterReady<SymbolTestOps>(
            state, wrong_gate_task
        ) &&
            state->tasks[wrong_gate_task].deps_prepared == first_up,
        "writer-ready CAS mismatch preserves the competing gate value"
    );
    LocalStats wrong_gate_stats{};
    Check(
        !WaitForSharedWriterReady<SymbolTestOps>(
            state, wrong_gate_task, wrong_gate_stats
        ) && state->fatal.value == 1,
        "writer-ready wait rejects a different task id and broadcasts fatal"
    );
    state->fatal.value = 0;
    constexpr int32_t timeout_gate_task = 6;
    state->tasks[timeout_gate_task].deps_prepared = -1;
    ExpiredWaitOps::calls.store(0, std::memory_order_relaxed);
    LocalStats timeout_gate_stats{};
    Check(
        !WaitForSharedWriterReady<ExpiredWaitOps>(
            state, timeout_gate_task, timeout_gate_stats
        ) && state->fatal.value == 1,
        "writer-ready wait times out and broadcasts fatal"
    );
    state->fatal.value = 0;

    Check(
        CommitSharedFaninWriters<SymbolTestOps, true>(
            state->shared_map, second_args, second_up, second_stats,
            alloc, first_up
        ),
        "second UP commits over the first UP writer"
    );
    Check(
        state->shared_map.shared_outputs[0].last_writer[0].value == second_up &&
            state->shared_map.shared_outputs[0].last_writer[1].value == second_up &&
            state->shared_map.shared_outputs[0].last_writer[2].value == second_up,
        "all accumulator writers finish at the second UP"
    );
    Check(
        first_stats.result.shared_symbol_inout_commits == 3 &&
            second_stats.result.shared_symbol_inout_commits == 3,
        "two PA groups register exactly three shared writers each"
    );

    LocalStats missing_selector_stats{};
    int32_t missing_selector_fanin[kMaxFanin] = {};
    bool missing_selector_ok = true;
    uint32_t missing_selector_lookups = UINT32_MAX;
    Check(
        CollectSharedFanin<SymbolTestOps, true>(
            state->shared_map, second_args, 12, kHeapWindow,
            missing_selector_stats, missing_selector_fanin,
            missing_selector_ok, missing_selector_lookups,
            nullptr, 1, second_up
        ) == 0 && !missing_selector_ok,
        "chained resolver rejects a producer selector that matches no ref"
    );
    Check(
        !CommitSharedFaninWriters<SymbolTestOps, true>(
            state->shared_map, second_args, 12,
            missing_selector_stats, 1, second_up
        ),
        "chained commit rejects a selector with no writable match"
    );
    LocalStats unchanged_writer_stats{};
    int32_t unchanged_writer_fanin[kMaxFanin] = {};
    bool unchanged_writer_ok = true;
    uint32_t unchanged_writer_lookups = UINT32_MAX;
    Check(
        CollectSharedFanin<SymbolTestOps, true>(
            state->shared_map, second_args, second_up, kHeapWindow,
            unchanged_writer_stats, unchanged_writer_fanin,
            unchanged_writer_ok, unchanged_writer_lookups,
            nullptr, alloc, alloc
        ) == 0 && !unchanged_writer_ok,
        "chained resolver rejects producer==writer as a non-chain"
    );
    Check(
        !CommitSharedFaninWriters<SymbolTestOps, true>(
            state->shared_map, second_args, 12,
            unchanged_writer_stats, alloc, alloc
        ),
        "chained commit also rejects producer==writer before mutation"
    );
    LocalStats invalid_expected_stats{};
    Check(
        !CommitSharedFaninWriters<SymbolTestOps, true>(
            state->shared_map, second_args, 12, invalid_expected_stats,
            alloc, -1
        ),
        "chained commit rejects an invalid expected writer before mutation"
    );
    Check(
        state->shared_map.shared_outputs[0].last_writer[0].value == second_up &&
            state->shared_map.shared_outputs[0].last_writer[1].value == second_up &&
            state->shared_map.shared_outputs[0].last_writer[2].value == second_up,
        "invalid expected writer leaves every accumulator unchanged"
    );

    // 该测试使用真实 PA 的 task-id 顺序和 UP 参数构造，但默认主循环仍
    // 固定每 batch 五 task，GetTaskKind(8) 尚不会返回 UP。这里明确只
    // 证明双组 orchestration 参数与 writer-intent 原语，不冒充完整 replay
    // 已接通。故意回退 writer，确认不能跳过前一 UP。
    for (uint32_t slot = 0; slot < 3; ++slot) {
        state->shared_map.shared_outputs[alloc]
            .last_writer[slot].value = first_up - 1;
    }
    LocalStats stale_stats{};
    int32_t stale_fanin[kMaxFanin] = {};
    bool stale_protocol_ok = true;
    uint32_t stale_ordinary_lookups = UINT32_MAX;
    Check(
        CollectSharedFanin<SymbolTestOps, true>(
            state->shared_map, second_args, second_up, kHeapWindow,
            stale_stats, stale_fanin, stale_protocol_ok,
            stale_ordinary_lookups, nullptr, alloc, first_up
        ) == 0 && !stale_protocol_ok,
        "chained resolver rejects a stale writer instead of skipping a stage"
    );

    UnmapSparseSchedulerState(state);
}

void TestPaWriterIntentPreGateFailuresDoNotPublishGate() {
    SchedulerState *state = MapSparseSchedulerState();
    if (state == nullptr) {
        ++g_failures;
        return;
    }
    ResetSharedState(state->shared_map);
    state->fatal.value = 0;
    state->heap_window = kHeapWindow;
    constexpr int32_t producer = 0;
    constexpr int32_t writer = 4;
    state->tasks[writer].deps_prepared = -1;

    SharedOutputCell &cell = state->shared_map.shared_outputs[producer];
    TaskArgs damaged_writer_args;
    ConstructTaskArgs(damaged_writer_args);
    for (uint32_t slot = 0; slot < 3; ++slot) {
        cell.published[slot].value = producer;
        cell.last_writer[slot].value = producer;
        cell.tensors[slot] =
            MakeTensor(0x360000000ULL + slot * 0x1000ULL, producer);
        AppendSharedOutputRef(
            damaged_writer_args,
            FdwicOutputRef{
                producer, static_cast<int16_t>(slot), 0, 0, 0, 0,
            },
            TensorArgType::Inout
        );
    }
    TensorDesc manual_view = MakeTensor(0x360010000ULL, producer);
    manual_view.manual_dep = true;
    AddLocalTensor(
        damaged_writer_args, manual_view, TensorArgType::Inout
    );
    cell.last_writer[2].value = -1;
    SubmitContext damaged_writer_context{};
    damaged_writer_context.task_id = writer;
    damaged_writer_context.won = true;
    LocalStats damaged_writer_stats{};
    Check(
        !PreparePaSharedWriterIntent<SymbolTestOps>(
            state, damaged_writer_args, damaged_writer_context,
            damaged_writer_stats
        ) &&
            state->fatal.value == 1 &&
            state->tasks[writer].deps_prepared == -1 &&
            cell.last_writer[0].value == producer &&
            cell.last_writer[1].value == producer &&
            cell.last_writer[2].value == -1,
        "read-only intent validation failure changes no writer and publishes no gate"
    );

    state->fatal.value = 0;
    state->tasks[writer].deps_prepared = -1;
    cell.last_writer[2].value = producer;
    PublicationFaultOps::fetch_race_address = &cell.last_writer[1].value;
    SubmitContext partial_commit_context{};
    partial_commit_context.task_id = writer;
    partial_commit_context.won = true;
    LocalStats partial_commit_stats{};
    Check(
        !PreparePaSharedWriterIntent<PublicationFaultOps>(
            state, damaged_writer_args, partial_commit_context,
            partial_commit_stats
        ) &&
            state->fatal.value == 1 &&
            state->tasks[writer].deps_prepared == -1 &&
            cell.last_writer[0].value == writer &&
            cell.last_writer[1].value == writer &&
            cell.last_writer[2].value == producer,
        "partial writer commit keeps terminal prefix evidence but publishes no gate"
    );
    PublicationFaultOps::fetch_race_address = nullptr;

    state->fatal.value = 0;
    state->tasks[writer].deps_prepared = -1;
    cell.last_writer[0].value = producer;
    cell.last_writer[1].value = producer;
    cell.last_writer[2].value = producer;
    TaskArgs missing_accumulator_args = damaged_writer_args;
    missing_accumulator_args.tensor_count = 2;
    SubmitContext missing_accumulator_context{};
    missing_accumulator_context.task_id = writer;
    missing_accumulator_context.won = true;
    LocalStats missing_accumulator_stats{};
    Check(
        !PreparePaSharedWriterIntent<SymbolTestOps>(
            state, missing_accumulator_args, missing_accumulator_context,
            missing_accumulator_stats
        ) &&
            state->fatal.value == 1 &&
            state->tasks[writer].deps_prepared == -1 &&
            cell.last_writer[0].value == producer &&
            cell.last_writer[1].value == producer &&
            cell.last_writer[2].value == producer,
        "missing PA accumulator rejects the fast path before mutation or gate"
    );

    state->fatal.value = 0;
    state->tasks[writer].deps_prepared = -1;
    TaskArgs wrong_accumulator_args = damaged_writer_args;
    wrong_accumulator_args.tensors[0]
        .pointer.output_ref.producer_task_id = 1;
    SubmitContext wrong_accumulator_context{};
    wrong_accumulator_context.task_id = writer;
    wrong_accumulator_context.won = true;
    LocalStats wrong_accumulator_stats{};
    Check(
        !PreparePaSharedWriterIntent<SymbolTestOps>(
            state, wrong_accumulator_args, wrong_accumulator_context,
            wrong_accumulator_stats
        ) &&
            state->fatal.value == 1 &&
            state->tasks[writer].deps_prepared == -1 &&
            cell.last_writer[0].value == producer &&
            cell.last_writer[1].value == producer &&
            cell.last_writer[2].value == producer,
        "PA writer intent rejects three refs that are not the batch accumulators"
    );

    state->fatal.value = 0;
    state->tasks[writer].deps_prepared = -1;
    TaskArgs duplicate_accumulator_args = damaged_writer_args;
    duplicate_accumulator_args.tensors[2]
        .pointer.output_ref.output_slot = 1;
    SubmitContext duplicate_accumulator_context{};
    duplicate_accumulator_context.task_id = writer;
    duplicate_accumulator_context.won = true;
    LocalStats duplicate_accumulator_stats{};
    Check(
        !PreparePaSharedWriterIntent<SymbolTestOps>(
            state, duplicate_accumulator_args,
            duplicate_accumulator_context,
            duplicate_accumulator_stats
        ) &&
            state->fatal.value == 1 &&
            state->tasks[writer].deps_prepared == -1 &&
            cell.last_writer[0].value == producer &&
            cell.last_writer[1].value == producer &&
            cell.last_writer[2].value == producer,
        "PA writer intent rejects duplicate accumulator slots before mutation"
    );

    state->fatal.value = 0;
    state->tasks[writer].deps_prepared = -1;
    TaskArgs missing_view_args = damaged_writer_args;
    missing_view_args.tensor_count = 3;
    SubmitContext missing_view_context{};
    missing_view_context.task_id = writer;
    missing_view_context.won = true;
    LocalStats missing_view_stats{};
    Check(
        !PreparePaSharedWriterIntent<SymbolTestOps>(
            state, missing_view_args, missing_view_context,
            missing_view_stats
        ) &&
            state->fatal.value == 1 &&
            state->tasks[writer].deps_prepared == -1 &&
            cell.last_writer[0].value == producer &&
            cell.last_writer[1].value == producer &&
            cell.last_writer[2].value == producer,
        "PA writer intent rejects a missing manual-dependency output view"
    );

    state->fatal.value = 0;
    state->tasks[writer].deps_prepared = -1;
    TaskArgs duplicate_view_args = damaged_writer_args;
    AddLocalTensor(
        duplicate_view_args, manual_view, TensorArgType::Inout
    );
    SubmitContext duplicate_view_context{};
    duplicate_view_context.task_id = writer;
    duplicate_view_context.won = true;
    LocalStats duplicate_view_stats{};
    Check(
        !PreparePaSharedWriterIntent<SymbolTestOps>(
            state, duplicate_view_args, duplicate_view_context,
            duplicate_view_stats
        ) &&
            state->fatal.value == 1 &&
            state->tasks[writer].deps_prepared == -1 &&
            cell.last_writer[0].value == producer &&
            cell.last_writer[1].value == producer &&
            cell.last_writer[2].value == producer,
        "PA writer intent rejects duplicate manual-dependency writers"
    );

    state->fatal.value = 0;
    state->tasks[writer].deps_prepared = -1;
    TaskArgs errored_args = damaged_writer_args;
    errored_args.has_error = true;
    SubmitContext errored_context{};
    errored_context.task_id = writer;
    errored_context.won = true;
    LocalStats errored_stats{};
    Check(
        !PreparePaSharedWriterIntent<SymbolTestOps>(
            state, errored_args, errored_context, errored_stats
        ) &&
            state->fatal.value == 1 &&
            state->tasks[writer].deps_prepared == -1 &&
            cell.last_writer[0].value == producer &&
            cell.last_writer[1].value == producer &&
            cell.last_writer[2].value == producer,
        "errored frontend args are rejected before writer mutation or gate"
    );

    state->fatal.value = 0;
    state->tasks[writer].deps_prepared = -1;
    TaskArgs invalid_scalar_args = damaged_writer_args;
    invalid_scalar_args.scalar_count =
        static_cast<int32_t>(kMaxTaskScalars) + 1;
    SubmitContext invalid_scalar_context{};
    invalid_scalar_context.task_id = writer;
    invalid_scalar_context.won = true;
    LocalStats invalid_scalar_stats{};
    Check(
        !PreparePaSharedWriterIntent<SymbolTestOps>(
            state, invalid_scalar_args, invalid_scalar_context,
            invalid_scalar_stats
        ) &&
            state->fatal.value == 1 &&
            state->tasks[writer].deps_prepared == -1 &&
            cell.last_writer[0].value == producer &&
            cell.last_writer[1].value == producer &&
            cell.last_writer[2].value == producer,
        "invalid scalar count is rejected before writer mutation or gate"
    );

    state->fatal.value = 0;
    state->tasks[writer].deps_prepared = -1;
    TensorDesc ordinary_writer = MakeTensor(0x370000000ULL, producer);
    ordinary_writer.manual_dep = false;
    TaskArgs ordinary_writer_args = damaged_writer_args;
    AddLocalTensor(
        ordinary_writer_args, ordinary_writer, TensorArgType::Inout
    );
    SubmitContext ordinary_writer_context{};
    ordinary_writer_context.task_id = writer;
    ordinary_writer_context.won = true;
    LocalStats ordinary_writer_stats{};
    Check(
        !PreparePaSharedWriterIntent<SymbolTestOps>(
            state, ordinary_writer_args, ordinary_writer_context,
            ordinary_writer_stats
        ) &&
            state->fatal.value == 1 &&
            state->tasks[writer].deps_prepared == -1 &&
            cell.last_writer[0].value == producer &&
            cell.last_writer[1].value == producer &&
            cell.last_writer[2].value == producer,
        "PA fast path rejects ordinary region writers before mutation or gate"
    );

    UnmapSparseSchedulerState(state);
}

void TestWriterCommitFailuresKeepTerminalEvidence() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetSharedState(*map);

    TaskArgs args;
    ConstructTaskArgs(args);
    AppendSharedOutputRef(
        args, FdwicOutputRef{0, 0, 0, 0, 0, 0},
        TensorArgType::Inout
    );

    LocalStats stats{};
    map->shared_outputs[0].last_writer[0].value = -1;
    Check(
        !CommitSharedFaninWriters<SymbolTestOps>(*map, args, 4, stats),
        "missing producer writer rejects INOUT commit"
    );
    Check(
        map->shared_outputs[0].last_writer[0].value == 4,
        "failed FetchMax keeps terminal over-advance evidence instead of rolling back"
    );
    Check(
        stats.result.shared_symbol_inout_commits == 0,
        "failed writer commit is not counted as success"
    );

    map->shared_outputs[0].last_writer[0].value = 7;
    Check(
        !CommitSharedFaninWriters<SymbolTestOps>(*map, args, 4, stats),
        "future writer rejects INOUT commit"
    );
    Check(
        map->shared_outputs[0].last_writer[0].value == 7,
        "failed FetchMax does not overwrite a later writer"
    );
}

void TestMultiWriterFailureKeepsPartialTerminalEvidence() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetSharedState(*map);
    TaskArgs args;
    ConstructTaskArgs(args);
    for (int32_t producer = 0; producer < 3; ++producer) {
        AppendSharedOutputRef(
            args, FdwicOutputRef{producer, 0, 0, 0, 0, 0},
            TensorArgType::Inout
        );
        map->shared_outputs[static_cast<uint32_t>(producer)]
            .last_writer[0].value = producer;
    }
    // 模拟只读解析完成后第二个 producer 控制字被破坏。第一个提交已经
    // 线性化，不能为了伪造事务性而回滚；第三个尚未触碰。
    map->shared_outputs[1].last_writer[0].value = -1;
    LocalStats stats{};
    Check(
        !CommitSharedFaninWriters<SymbolTestOps>(*map, args, 4, stats),
        "second of three INOUT commits rejects damaged producer writer"
    );
    Check(
        map->shared_outputs[0].last_writer[0].value == 4 &&
            map->shared_outputs[1].last_writer[0].value == 4 &&
            map->shared_outputs[2].last_writer[0].value == 2,
        "partial terminal commit preserves completed, failed, and untouched evidence"
    );
    Check(
        stats.result.shared_symbol_inout_commits == 1,
        "only the writer committed before terminal failure is counted"
    );
}

void TestFailedSealDiscardsBuiltTask() {
    auto worker = std::make_unique<WorkerState>();
    worker->occupied_count = 2;
    worker->slots[0].task_id = 4;
    worker->slots[0].occupied = true;
    worker->slots[0].built = true;
    worker->slots[1].task_id = 3;
    worker->slots[1].occupied = true;
    worker->slots[1].built = true;

    Check(
        DiscardBuiltTask(*worker, 4),
        "failed shared seal finds its just-built private slot"
    );
    Check(
        !worker->slots[0].occupied && !worker->slots[0].built &&
            worker->occupied_count == 1,
        "failed shared seal removes only its own slot from FinalDrain"
    );
    Check(
        worker->slots[1].occupied && worker->slots[1].built,
        "failed shared seal preserves previously built work"
    );
    Check(
        !DiscardBuiltTask(*worker, 9) && worker->occupied_count == 1,
        "missing failed task cannot corrupt occupied accounting"
    );
    worker->slots[0].task_id = 5;
    worker->slots[0].occupied = true;
    worker->slots[0].built = true;
    worker->occupied_count = 0;
    Check(
        !DiscardBuiltTask(*worker, 5),
        "corrupt occupied accounting remains visible to the caller"
    );
    Check(
        !worker->slots[0].occupied && !worker->slots[0].built,
        "terminal discard clears the failed slot despite corrupt accounting"
    );
}

void TestCase1RegistrationBypassesRegionSequencer() {
    SubmitContext context{};

    TaskArgs shared_args;
    ConstructTaskArgs(shared_args);
    AppendSharedOutputRef(
        shared_args, FdwicOutputRef{0, 0, 0, 0, 0, 0},
        TensorArgType::Inout
    );
    context.register_mask = 1;
    Check(
        ValidateEmptySharedRegistration(shared_args, context),
        "shared symbol writer bypasses ordinary region registration"
    );

    TensorDesc manual = MakeTensor(0x360000000ULL, 0);
    manual.manual_dep = true;
    TaskArgs manual_args;
    ConstructTaskArgs(manual_args);
    AddLocalTensor(manual_args, manual, TensorArgType::Inout);
    context.register_mask = 1;
    Check(
        ValidateEmptySharedRegistration(manual_args, context),
        "manual-dependency local writer leaves the region delta empty"
    );

    TaskArgs manual_gm_args;
    ConstructTaskArgs(manual_gm_args);
    AddGmTensor(manual_gm_args, manual, TensorArgType::Inout);
    context.register_mask = 1;
    Check(
        ValidateEmptySharedRegistration(manual_gm_args, context),
        "manual-dependency GM writer leaves the region delta empty"
    );

    TensorDesc ordinary = manual;
    ordinary.manual_dep = false;
    TaskArgs ordinary_args;
    ConstructTaskArgs(ordinary_args);
    AddLocalTensor(ordinary_args, ordinary, TensorArgType::Inout);
    context.register_mask = 1;
    Check(
        !ValidateEmptySharedRegistration(ordinary_args, context),
        "non-manual ordinary writer fails closed instead of entering the ring"
    );

    TaskArgs ordinary_gm_args;
    ConstructTaskArgs(ordinary_gm_args);
    AddGmTensor(ordinary_gm_args, ordinary, TensorArgType::Inout);
    context.register_mask = 1;
    Check(
        !ValidateEmptySharedRegistration(ordinary_gm_args, context),
        "non-manual ordinary GM writer fails closed before region append"
    );

    context.register_mask = 2;
    Check(
        !ValidateEmptySharedRegistration(shared_args, context),
        "registration mask outside active arguments fails closed"
    );
}

void TestPostBuildSealClosesSuccessAndFailurePaths() {
    {
        SchedulerState *state = MapSparseSchedulerState();
        Check(state != nullptr, "writer-failure seal state maps");
        if (state != nullptr) {
            ResetSharedState(state->shared_map);
            state->shared_map.committed_tasks.value = 4;
            WorkerState &worker = state->workers[0];
            worker.occupied_count = 1;
            worker.slots[0].task_id = 4;
            worker.slots[0].occupied = true;
            worker.slots[0].built = true;

            TaskArgs args;
            ConstructTaskArgs(args);
            AppendSharedOutputRef(
                args, FdwicOutputRef{0, 0, 0, 0, 0, 0},
                TensorArgType::Inout
            );
            SubmitContext context{};
            context.task_id = 4;
            context.result.task_id = 4;
            context.shared_result.Reset(4);
            LocalStats stats{};

            Check(
                !PublishSharedWinnerAfterBuild<SymbolTestOps>(
                    state, worker, args, context, 4, TaskKind::Up, stats
                ),
                "writer invariant failure rejects post-build seal"
            );
            Check(
                state->fatal.value == 1 &&
                    state->shared_map.committed_tasks.value == 4,
                "writer failure broadcasts fatal without touching region sequencer"
            );
            Check(
                state->shared_map.shared_outputs[0]
                        .last_writer[0].value == 4,
                "writer failure retains terminal FetchMax evidence"
            );
            Check(
                !worker.slots[0].occupied &&
                    !worker.slots[0].built &&
                    worker.occupied_count == 0,
                "writer failure removes the failed winner from FinalDrain"
            );
            UnmapSparseSchedulerState(state);
        }
    }

    {
        SchedulerState *state = MapSparseSchedulerState();
        Check(state != nullptr, "writer-then-publication-failure state maps");
        if (state != nullptr) {
            ResetSharedState(state->shared_map);
            state->shared_map.committed_tasks.value = 37;
            state->shared_map.shared_outputs[0]
                .last_writer[0].value = 0;
            WorkerState &worker = state->workers[0];
            worker.occupied_count = 1;
            worker.slots[0].task_id = 4;
            worker.slots[0].occupied = true;
            worker.slots[0].built = true;

            TaskArgs args;
            ConstructTaskArgs(args);
            AppendSharedOutputRef(
                args, FdwicOutputRef{0, 0, 0, 0, 0, 0},
                TensorArgType::Inout
            );
            SubmitContext context{};
            context.task_id = 4;
            context.result.task_id = 4;
            TensorDesc output = MakeTensor(0x380000000ULL, 4);
            context.result.count = 1;
            context.result.tensors[0] = &output;
            context.shared_result.Reset(4);
            Check(
                context.shared_result.AddOutputRef(4, 0),
                "writer-publication-failure context declares output"
            );
            LocalStats stats{};

            PublicationFaultOps::exchange_race_address =
                &state->shared_map.shared_outputs[4]
                     .published[0].value;
            Check(
                !PublishSharedWinnerAfterBuild<PublicationFaultOps>(
                    state, worker, args, context, 4, TaskKind::Up, stats
                ),
                "publication failure after writer commit rejects post-build seal"
            );
            Check(
                PublicationFaultOps::exchange_race_address == nullptr &&
                    state->fatal.value == 1,
                "publication failure consumes injection and broadcasts fatal"
            );
            Check(
                state->shared_map.committed_tasks.value == 37 &&
                    state->shared_map.reclaim_upto.value == -1,
                "post-build seal leaves the unused region sequencer untouched"
            );
            Check(
                state->shared_map.shared_outputs[0]
                        .last_writer[0].value == 4 &&
                    stats.result.shared_symbol_inout_commits == 1,
                "publication failure retains the completed writer commit"
            );
            Check(
                state->shared_map.shared_outputs[4]
                        .published[0].value == -1 &&
                    state->shared_map.shared_outputs[4]
                        .last_writer[0].value == -1,
                "publication failure rolls back only the producer output cell"
            );
            Check(
                !worker.slots[0].occupied &&
                    !worker.slots[0].built &&
                    worker.occupied_count == 0,
                "publication failure removes the failed winner slot"
            );
            UnmapSparseSchedulerState(state);
        }
    }

    {
        SchedulerState *state = MapSparseSchedulerState();
        Check(state != nullptr, "publication-failure seal state maps");
        if (state != nullptr) {
            ResetSharedState(state->shared_map);
            state->shared_map.committed_tasks.value = 19;
            state->shared_map.shared_outputs[1]
                .last_writer[0].value = 7;
            WorkerState &worker = state->workers[0];
            worker.occupied_count = 1;
            worker.slots[0].task_id = 1;
            worker.slots[0].occupied = true;
            worker.slots[0].built = true;

            TensorDesc output = MakeTensor(0x390000000ULL, 1);
            TaskArgs args;
            ConstructTaskArgs(args);
            SubmitContext context{};
            context.task_id = 1;
            context.result.task_id = 1;
            context.result.count = 1;
            context.result.tensors[0] = &output;
            context.shared_result.Reset(1);
            Check(
                context.shared_result.AddOutputRef(1, 0),
                "publication-failure context declares output"
            );
            LocalStats stats{};

            Check(
                !PublishSharedWinnerAfterBuild<SymbolTestOps>(
                    state, worker, args, context, 1, TaskKind::Qk, stats
                ),
                "publication invariant failure rejects post-build seal"
            );
            Check(
                state->fatal.value == 1 &&
                    state->shared_map.committed_tasks.value == 19,
                "publication failure is terminal without a global turn"
            );
            Check(
                state->shared_map.shared_outputs[1]
                        .published[0].value == -1 &&
                    state->shared_map.shared_outputs[1]
                        .last_writer[0].value == 7,
                "publication preflight failure exposes no new descriptor"
            );
            Check(
                !worker.slots[0].occupied &&
                    worker.occupied_count == 0,
                "publication failure removes the failed winner slot"
            );
            UnmapSparseSchedulerState(state);
        }
    }

    {
        SchedulerState *state = MapSparseSchedulerState();
        Check(state != nullptr, "successful seal state maps");
        if (state != nullptr) {
            ResetSharedState(state->shared_map);
            state->shared_map.committed_tasks.value = 23;
            WorkerState &worker = state->workers[0];
            worker.occupied_count = 1;
            worker.slots[0].task_id = 1;
            worker.slots[0].occupied = true;
            worker.slots[0].built = true;

            TensorDesc output = MakeTensor(0x3A0000000ULL, 1);
            TaskArgs args;
            ConstructTaskArgs(args);
            SubmitContext context{};
            context.task_id = 1;
            context.result.task_id = 1;
            context.result.count = 1;
            context.result.tensors[0] = &output;
            context.shared_result.Reset(1);
            Check(
                context.shared_result.AddOutputRef(1, 0),
                "successful seal context declares output"
            );
            LocalStats stats{};

            SealOrderOps::output_writer_address =
                &state->shared_map.shared_outputs[1]
                     .last_writer[0].value;
            SealOrderOps::published_address =
                &state->shared_map.shared_outputs[1]
                     .published[0].value;
            SealOrderOps::expected_writer = 1;
            SealOrderOps::order_ok = true;
            Check(
                PublishSharedWinnerAfterBuild<SealOrderOps>(
                    state, worker, args, context, 1, TaskKind::Qk, stats
                ),
                "valid post-build seal succeeds"
            );
            Check(
                SealOrderOps::order_ok,
                "output writer initialization precedes published"
            );
            Check(
                state->fatal.value == 0 &&
                    state->shared_map.committed_tasks.value == 23 &&
                    state->shared_map.shared_outputs[1]
                        .last_writer[0].value == 1 &&
                    state->shared_map.shared_outputs[1]
                        .published[0].value == 1,
                "successful seal publishes output without touching sequencer"
            );
            Check(
                SameTensor(
                    state->shared_map.shared_outputs[1].tensors[0],
                    output
                ),
                "successful seal publishes the exact descriptor"
            );
            Check(
                worker.slots[0].occupied &&
                    worker.slots[0].built &&
                    worker.occupied_count == 1,
                "successful seal preserves executable winner slot"
            );
            SealOrderOps::output_writer_address = nullptr;
            SealOrderOps::published_address = nullptr;
            UnmapSparseSchedulerState(state);
        }
    }
}

void TestPublicationPreflightIsAllOrNothing() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetSharedState(*map);

    TensorDesc first = MakeTensor(0x300000000ULL, 0);
    TensorDesc second = MakeTensor(0x300001000ULL, 0);
    SubmitContext producer{};
    producer.task_id = 0;
    producer.result.task_id = 0;
    producer.result.count = 2;
    producer.result.tensors[0] = &first;
    producer.result.tensors[1] = &second;
    producer.shared_result.Reset(0);
    Check(producer.shared_result.AddOutputRef(0, 0), "preflight adds slot 0");
    Check(producer.shared_result.AddOutputRef(0, 1), "preflight adds slot 1");

    // 人为污染第二槽，验证失败发生在任何 descriptor/前槽控制字写入前。
    map->shared_outputs[0].last_writer[1].value = 7;
    const TensorDesc zero{};
    Check(
        !PublishSharedTaskOutputs<SymbolTestOps>(*map, producer, 0),
        "later occupied slot rejects whole publication"
    );
    Check(
        map->shared_outputs[0].published[0].value == -1 &&
            map->shared_outputs[0].last_writer[0].value == -1,
        "later-slot failure leaves earlier control state untouched"
    );
    Check(
        SameTensor(map->shared_outputs[0].tensors[0], zero) &&
            SameTensor(map->shared_outputs[0].tensors[1], zero),
        "later-slot failure leaves every descriptor untouched"
    );
}

void TestPublicationCommitFaultsRollback() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    TensorDesc first = MakeTensor(0x350000000ULL, 0);
    TensorDesc second = MakeTensor(0x350001000ULL, 0);
    SubmitContext producer{};
    producer.task_id = 0;
    producer.result.task_id = 0;
    producer.result.count = 2;
    producer.result.tensors[0] = &first;
    producer.result.tensors[1] = &second;
    producer.shared_result.Reset(0);
    Check(producer.shared_result.AddOutputRef(0, 0), "fault test adds slot 0");
    Check(producer.shared_result.AddOutputRef(0, 1), "fault test adds slot 1");
    const TensorDesc zero{};

    ResetSharedState(*map);
    PublicationFaultOps::fetch_race_address =
        &map->shared_outputs[0].last_writer[1].value;
    Check(
        !PublishSharedTaskOutputs<PublicationFaultOps>(*map, producer, 0),
        "FetchMax race rejects whole publication"
    );
    Check(
        map->shared_outputs[0].last_writer[0].value == -1 &&
            map->shared_outputs[0].last_writer[1].value == -2,
        "FetchMax failure restores reserved and raced writer values"
    );
    Check(
        map->shared_outputs[0].published[0].value == -1 &&
            map->shared_outputs[0].published[1].value == -1 &&
            SameTensor(map->shared_outputs[0].tensors[0], zero) &&
            SameTensor(map->shared_outputs[0].tensors[1], zero),
        "FetchMax failure publishes no descriptor"
    );

    ResetSharedState(*map);
    PublicationFaultOps::exchange_race_address =
        &map->shared_outputs[0].published[1].value;
    Check(
        !PublishSharedTaskOutputs<PublicationFaultOps>(*map, producer, 0),
        "published Exchange race rejects whole publication"
    );
    Check(
        map->shared_outputs[0].published[0].value == -1 &&
            map->shared_outputs[0].published[1].value == -1 &&
            map->shared_outputs[0].last_writer[0].value == -1 &&
            map->shared_outputs[0].last_writer[1].value == -1,
        "published Exchange failure rolls back every control word"
    );
    Check(
        SameTensor(map->shared_outputs[0].tensors[0], zero) &&
            SameTensor(map->shared_outputs[0].tensors[1], zero),
        "published Exchange failure clears flushed descriptors"
    );
}

void TestConsumerWaitsForDelayedPublication() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetSharedState(*map);

    TensorDesc output = MakeTensor(0x380000000ULL, 0);
    SubmitContext producer{};
    producer.task_id = 0;
    producer.result.task_id = 0;
    producer.result.count = 1;
    producer.result.tensors[0] = &output;
    producer.shared_result.Reset(0);
    Check(
        producer.shared_result.AddOutputRef(0, 0),
        "delayed producer accepts output slot"
    );

    TaskArgs args;
    ConstructTaskArgs(args);
    AppendSharedOutputRef(
        args, producer.shared_result.OutputRef(0), TensorArgType::Input
    );
    auto payload = std::make_unique<TaskPayload>();
    std::memset(payload.get(), 0x3C, sizeof(*payload));
    SubmitContext consumer{};
    consumer.task_id = 1;
    consumer.payload = payload.get();
    consumer.tensor_count = args.tensor_count;
    consumer.scalar_count = args.scalar_count;
    LocalStats stats{};
    int32_t fanin[kMaxFanin] = {};
    bool protocol_ok = false;
    uint32_t ordinary_lookups = UINT32_MAX;
    volatile int32_t fatal = 0;
    std::atomic<bool> publish_ok{false};

    SymbolTestOps::wait_address =
        &map->shared_outputs[0].published[0].value;
    SymbolTestOps::wait_loads.store(0, std::memory_order_relaxed);
    SymbolTestOps::now_calls.store(0, std::memory_order_relaxed);
    std::thread publisher([&] {
        // 等 consumer 已经观察到未发布状态后再发布，避免把本测试退化成
        // “进入 helper 前已经 ready”的普通快路径。
        while (SymbolTestOps::wait_loads.load(std::memory_order_acquire) == 0) {
            std::this_thread::yield();
        }
        publish_ok.store(
            PublishSharedTaskOutputs<SymbolTestOps>(*map, producer, 0),
            std::memory_order_release
        );
    });

    const uint32_t count = CollectSharedFanin<SymbolTestOps>(
        *map, args, 1, kHeapWindow, stats, fanin,
        protocol_ok, ordinary_lookups, &fatal
    );
    publisher.join();
    SymbolTestOps::wait_address = nullptr;

    Check(publish_ok.load(std::memory_order_acquire), "delayed publication succeeds");
    Check(protocol_ok && fatal == 0, "consumer waits without protocol failure");
    Check(
        SymbolTestOps::wait_loads.load(std::memory_order_acquire) > 1,
        "consumer performs at least one unpublished retry"
    );
    Check(
        SymbolTestOps::now_calls.load(std::memory_order_relaxed) >= 1,
        "unpublished slow path establishes a watchdog window"
    );
    Check(count == 1 && fanin[0] == 0, "delayed INPUT closes producer fanin");
    Check(ordinary_lookups == 0, "delayed symbol never enters ordinary map");
    Check(
        AllBytesEqual(payload.get(), sizeof(*payload), 0x3C),
        "delayed resolver leaves task payload scratch untouched"
    );
    LocalSlot slot{};
    BuildSlotPayload<SymbolTestOps, true>(
        slot, 1, static_cast<uint32_t>(FunctionId(TaskKind::Qk)), 0,
        args, consumer, fanin, count, *map
    );
    Check(
        SameTensor(slot.tensors[0], output),
        "delayed descriptor lands directly in LocalSlot after publication"
    );
}

void TestPaUpSplitWriterPublicationStages() {
    SchedulerState *state = MapSparseSchedulerState();
    if (state == nullptr) {
        ++g_failures;
        return;
    }
    ResetSharedState(state->shared_map);
    constexpr int32_t kProducer = 0;
    constexpr int32_t kPredecessor = 3;
    constexpr int32_t kWriter = 4;
    state->tasks[kPredecessor].deps_prepared = -1;
    state->tasks[kWriter].deps_prepared = -1;
    SharedOutputCell &cell =
        state->shared_map.shared_outputs[kProducer];
    for (uint32_t slot = 0; slot < 3; ++slot) {
        cell.published[slot].value = kProducer;
        cell.last_writer[slot].value = kProducer;
    }
    const uint32_t key_base =
        static_cast<uint32_t>(kProducer) *
            kSharedOutputMaxPerTask +
        1U;
    const uint32_t symbol_keys[3] = {
        key_base + 2U, key_base + 1U, key_base
    };
    LocalStats stats{};

    Check(
        PublishTrustedPaUpWriterHistoryPayload<SymbolTestOps>(
            state->shared_map, symbol_keys, kWriter,
            kProducer, kProducer, &stats
        ),
        "PA UP split path prepares immutable history before the turn"
    );
    const SharedWriterHistoryCell &history =
        state->shared_map.writer_history[kWriter];
    bool prepared_but_unpublished =
        history.magic == kSharedWriterHistoryMagic &&
        history.writer_task == kWriter &&
        history.count == 3 &&
        state->tasks[kWriter].deps_prepared == -1;
    for (uint32_t index = 0; index < 3; ++index) {
        prepared_but_unpublished &=
            history.entries[index].symbol_key ==
                symbol_keys[index] &&
            history.entries[index].previous_writer ==
                kProducer &&
            cell.last_writer[index].value == kProducer;
    }
    Check(
        prepared_but_unpublished,
        "prepared PA UP history remains unreachable before last-writer CAS"
    );

    TaskArgs early_reader_args;
    ConstructTaskArgs(early_reader_args);
    AppendSharedOutputRef(
        early_reader_args,
        FdwicOutputRef{kProducer, 0, 0, 0, 0, 0},
        TensorArgType::Input
    );
    int32_t early_fanin[kMaxFanin] = {};
    bool early_protocol_ok = false;
    uint32_t early_ordinary_lookups = UINT32_MAX;
    LocalStats early_stats{};
    const uint32_t early_count =
        CollectSharedFanin<SymbolTestOps, false, true>(
            state->shared_map, early_reader_args,
            /*task_id=*/2, kHeapWindow, early_stats,
            early_fanin, early_protocol_ok,
            early_ordinary_lookups
        );
    Check(
        early_protocol_ok && early_count == 1 &&
            early_fanin[0] == kProducer &&
            cell.last_writer[0].value == kProducer,
        "reader cannot discover a pre-turn orphan history"
    );

    state->tasks[kPredecessor].deps_prepared = kPredecessor;
    int64_t ready_observed = -1;
    Check(
        WaitForSharedTaskInsertTurn<SymbolTestOps>(
            state, kWriter, stats, ready_observed
        ) &&
            ready_observed == kPredecessor &&
            CommitTrustedPaUpLastWriters<SymbolTestOps>(
                state->shared_map, kWriter,
                kProducer, kProducer, &stats
            ),
        "PA UP split path publishes three writers only after acquiring the turn"
    );
    int32_t later_fanin[kMaxFanin] = {};
    bool later_protocol_ok = false;
    uint32_t later_ordinary_lookups = UINT32_MAX;
    LocalStats later_stats{};
    const uint32_t later_count =
        CollectSharedFanin<SymbolTestOps, false, true>(
            state->shared_map, early_reader_args,
            /*task_id=*/5, kHeapWindow, later_stats,
            later_fanin, later_protocol_ok,
            later_ordinary_lookups
        );
    Check(
        later_protocol_ok && later_count == 1 &&
            later_fanin[0] == kWriter &&
            cell.last_writer[0].value == kWriter,
        "reader discovers the prepared history only after last-writer CAS"
    );

    ResetSharedState(state->shared_map);
    for (uint32_t slot = 0; slot < 3; ++slot) {
        cell.published[slot].value = kProducer;
        cell.last_writer[slot].value = kProducer;
    }
    Check(
        PublishTrustedPaUpWriterHistoryPayload<SymbolTestOps>(
            state->shared_map, symbol_keys, kWriter,
            kProducer, kProducer, &stats
        ),
        "partial-CAS setup republishes the immutable history"
    );
    cell.last_writer[1].value = 1;
    Check(
        !CommitTrustedPaUpLastWriters<SymbolTestOps>(
            state->shared_map, kWriter,
            kProducer, kProducer, &stats
        ) &&
            state->shared_map.writer_history[kWriter].magic ==
                kSharedWriterHistoryMagic &&
            cell.last_writer[2].value == kWriter &&
            cell.last_writer[1].value == 1 &&
            cell.last_writer[0].value == kProducer,
        "split PA UP commit preserves its terminal partial-CAS prefix"
    );
    UnmapSparseSchedulerState(state);
}

void TestOrderedPreparedSymbolUsesSinglePublicationCheck() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetSharedState(*map);

    constexpr int32_t kProducer = 0;
    constexpr int32_t kWriter = 1;
    constexpr int32_t kReader = 2;
    TensorDesc output = MakeTensor(0x381000000ULL, kProducer);
    SubmitContext producer{};
    producer.task_id = kProducer;
    producer.result.task_id = kProducer;
    producer.result.count = 1;
    producer.result.tensors[0] = &output;
    producer.shared_result.Reset(kProducer);
    Check(
        producer.shared_result.AddOutputRef(kProducer, 0) &&
            PublishSharedTaskOutputs<SymbolTestOps>(
                *map, producer, kProducer
            ),
        "ordered symbol setup publishes the producer descriptor"
    );
    const FdwicOutputRef output_ref =
        producer.shared_result.OutputRef(0);
    uint32_t symbol_key = 0;
    Check(
        SharedSymbolHistoryKey(output_ref, symbol_key),
        "ordered symbol setup precomputes one packed key"
    );

    volatile int32_t fatal = 0;
    SymbolTestOps::wait_address =
        &map->shared_outputs[kProducer].published[0].value;
    SymbolTestOps::wait_loads.store(0, std::memory_order_relaxed);
    SymbolTestOps::now_calls.store(0, std::memory_order_relaxed);
    SymbolTestOps::spin_calls.store(0, std::memory_order_relaxed);
    const bool committed =
        CommitPreparedSymbolSharedWriterIntentSet<SymbolTestOps>(
            *map, &symbol_key, 1, kWriter, &fatal
        );
    SymbolTestOps::wait_address = nullptr;

    const SharedWriterHistoryCell &history =
        map->writer_history[kWriter];
    Check(committed && fatal == 0, "ordered prepared symbol commit succeeds");
    Check(
        SymbolTestOps::wait_loads.load(std::memory_order_relaxed) == 1,
        "ordered prepared commit checks published exactly once"
    );
    Check(
        SymbolTestOps::now_calls.load(std::memory_order_relaxed) == 0 &&
            SymbolTestOps::spin_calls.load(std::memory_order_relaxed) == 0,
        "ordered prepared commit never opens a watchdog or spins"
    );
    Check(
        history.magic == kSharedWriterHistoryMagic &&
            history.writer_task == kWriter &&
            history.count == 1 &&
            history.entries[0].symbol_key == symbol_key &&
            history.entries[0].previous_writer == kProducer &&
            map->shared_outputs[kProducer].last_writer[0].value ==
                kWriter,
        "ordered prepared commit publishes the exact immutable history"
    );
    TaskArgs reader_args;
    ConstructTaskArgs(reader_args);
    AppendSharedOutputRef(
        reader_args, output_ref, TensorArgType::Input
    );
    LocalStats reader_stats{};
    int32_t fanin[kMaxFanin] = {};
    bool protocol_ok = false;
    uint32_t ordinary_lookups = UINT32_MAX;
    SymbolTestOps::wait_address =
        &map->shared_outputs[kProducer].published[0].value;
    SymbolTestOps::wait_loads.store(0, std::memory_order_relaxed);
    SymbolTestOps::now_calls.store(0, std::memory_order_relaxed);
    SymbolTestOps::spin_calls.store(0, std::memory_order_relaxed);
    const uint32_t fanin_count =
        CollectSharedFanin<SymbolTestOps, false, true>(
            *map, reader_args, kReader, kHeapWindow,
            reader_stats, fanin, protocol_ok, ordinary_lookups,
            &fatal
        );
    SymbolTestOps::wait_address = nullptr;
    Check(
        protocol_ok && fanin_count == 1 &&
            fanin[0] == kWriter && ordinary_lookups == 0,
        "ordered latest-writer lookup still resolves the prepared writer"
    );
    Check(
        SymbolTestOps::wait_loads.load(std::memory_order_relaxed) == 1 &&
            SymbolTestOps::now_calls.load(std::memory_order_relaxed) == 0 &&
            SymbolTestOps::spin_calls.load(std::memory_order_relaxed) == 0,
        "ordered latest-writer lookup also uses one check without waiting"
    );

    ResetSharedState(*map);
    fatal = 0;
    SymbolTestOps::wait_address =
        &map->shared_outputs[kProducer].published[0].value;
    SymbolTestOps::wait_loads.store(0, std::memory_order_relaxed);
    SymbolTestOps::now_calls.store(0, std::memory_order_relaxed);
    SymbolTestOps::spin_calls.store(0, std::memory_order_relaxed);
    const bool missing_rejected =
        !CommitPreparedSymbolSharedWriterIntentSet<SymbolTestOps>(
            *map, &symbol_key, 1, kWriter, &fatal
        );
    SymbolTestOps::wait_address = nullptr;
    Check(
        missing_rejected && fatal == 1,
        "ordered prepared commit rejects an unpublished producer immediately"
    );
    Check(
        SymbolTestOps::wait_loads.load(std::memory_order_relaxed) == 1 &&
            SymbolTestOps::now_calls.load(std::memory_order_relaxed) == 0 &&
            SymbolTestOps::spin_calls.load(std::memory_order_relaxed) == 0,
        "unpublished ordered producer fails after one load without waiting"
    );
    Check(
        map->writer_history[kWriter].magic == 0 &&
            map->shared_outputs[kProducer].last_writer[0].value == -1,
        "unpublished rejection preserves history and writer"
    );

    // 正式 Finish 只有在取得 task N 的 insert turn 后才使用该编译期
    // 分支；前驱 handoff 已经传递 producer descriptor/published 的
    // 可见性。这里刻意只建立后续 last_writer 前置条件并保持 published
    // 为 -1，锁定 helper 本身不会偷偷恢复一次 publication load。
    ResetSharedState(*map);
    map->shared_outputs[kProducer].last_writer[0].value = kProducer;
    fatal = 0;
    SymbolTestOps::wait_address =
        &map->shared_outputs[kProducer].published[0].value;
    SymbolTestOps::wait_loads.store(0, std::memory_order_relaxed);
    const bool trusted_committed =
        CommitPreparedSymbolSharedWriterIntentSet<
            SymbolTestOps, false, false
        >(
            *map, &symbol_key, 1, kWriter, &fatal
        );
    SymbolTestOps::wait_address = nullptr;
    Check(
        trusted_committed && fatal == 0 &&
            SymbolTestOps::wait_loads.load(
                std::memory_order_relaxed
            ) == 0 &&
            map->writer_history[kWriter].magic ==
                kSharedWriterHistoryMagic &&
            map->shared_outputs[kProducer].last_writer[0].value ==
                kWriter,
        "insert-turn trusted commit omits only the publication load"
    );

    // PA 正式路径还从 batch/group 元数据得到精确 previous writer，并
    // 继续由 CAS expected-old 校验共享状态。把 last_writer 地址设为
    // Load 探针，证明该实例不再做 CAS 前预读。
    ResetSharedState(*map);
    map->shared_outputs[kProducer].last_writer[0].value = kProducer;
    fatal = 0;
    SymbolTestOps::wait_address =
        &map->shared_outputs[kProducer].last_writer[0].value;
    SymbolTestOps::wait_loads.store(0, std::memory_order_relaxed);
    const bool expected_previous_committed =
        CommitPreparedSymbolSharedWriterIntentSet<
            SymbolTestOps, false, false, true
        >(
            *map, &symbol_key, 1, kWriter, &fatal,
            nullptr, kProducer
        );
    SymbolTestOps::wait_address = nullptr;
    Check(
        expected_previous_committed && fatal == 0 &&
            SymbolTestOps::wait_loads.load(
                std::memory_order_relaxed
            ) == 0 &&
            map->writer_history[kWriter].entries[0].previous_writer ==
                kProducer &&
            map->shared_outputs[kProducer].last_writer[0].value ==
                kWriter,
        "PA expected-previous commit omits the last-writer preload"
    );

    // 正式 PA UP callback 按 max/sum/output 构造参数，因此专路要求
    // 三个 key 同属 batch Alloc producer且 slot 精确为 2/1/0。
    constexpr int16_t kPaSlotOrder[3] = {2, 1, 0};
    uint32_t pa_up_keys[3] = {};
    for (uint32_t index = 0; index < 3; ++index) {
        FdwicOutputRef pa_ref = output_ref;
        pa_ref.output_slot = kPaSlotOrder[index];
        Check(
            SharedSymbolHistoryKey(pa_ref, pa_up_keys[index]),
            "PA UP setup derives each packed symbol key"
        );
    }
    ResetSharedState(*map);
    for (uint32_t slot = 0; slot < 3; ++slot) {
        map->shared_outputs[kProducer]
            .last_writer[slot].value = kProducer;
    }
    fatal = 0;
    const bool pa_up_committed =
        CommitPreparedSymbolSharedWriterIntentSet<
            SymbolTestOps, false, false, true, true
        >(
            *map, pa_up_keys, 3, kReader, &fatal,
            nullptr, kProducer, kProducer
        );
    bool pa_up_exact =
        pa_up_committed && fatal == 0 &&
        map->writer_history[kReader].count == 3;
    for (uint32_t index = 0; index < 3; ++index) {
        pa_up_exact &=
            map->writer_history[kReader]
                .entries[index].symbol_key ==
                pa_up_keys[index] &&
            map->writer_history[kReader]
                .entries[index].previous_writer ==
                kProducer;
    }
    for (uint32_t slot = 0; slot < 3; ++slot) {
        pa_up_exact &=
            map->shared_outputs[kProducer]
                .last_writer[slot].value == kReader;
    }
    Check(
        pa_up_exact,
        "PA UP compact path keeps the exact callback order index-aligned"
    );

    // 数量或 slot 集合不精确时，必须在 history 首次写入前失败。
    ResetSharedState(*map);
    for (uint32_t slot = 0; slot < 3; ++slot) {
        map->shared_outputs[kProducer]
            .last_writer[slot].value = kProducer;
    }
    fatal = 0;
    const bool pa_up_empty_rejected =
        !CommitPreparedSymbolSharedWriterIntentSet<
            SymbolTestOps, false, false, true, true
        >(
            *map, nullptr, 0, kReader, &fatal,
            nullptr, kProducer, kProducer
        );
    Check(
        pa_up_empty_rejected && fatal == 1 &&
            map->writer_history[kReader].magic == 0 &&
            map->shared_outputs[kProducer]
                .last_writer[0].value == kProducer &&
            map->shared_outputs[kProducer]
                .last_writer[1].value == kProducer &&
            map->shared_outputs[kProducer]
                .last_writer[2].value == kProducer,
        "PA UP compact path rejects a missing writer set before GM publication"
    );

    ResetSharedState(*map);
    for (uint32_t slot = 0; slot < 3; ++slot) {
        map->shared_outputs[kProducer]
            .last_writer[slot].value = kProducer;
    }
    fatal = 0;
    const bool pa_up_count_rejected =
        !CommitPreparedSymbolSharedWriterIntentSet<
            SymbolTestOps, false, false, true, true
        >(
            *map, pa_up_keys, 2, kReader, &fatal,
            nullptr, kProducer, kProducer
        );
    const uint32_t wrong_order_keys[3] = {
        pa_up_keys[0], pa_up_keys[2], pa_up_keys[1]
    };
    fatal = 0;
    const bool pa_up_order_rejected =
        !CommitPreparedSymbolSharedWriterIntentSet<
            SymbolTestOps, false, false, true, true
        >(
            *map, wrong_order_keys, 3, kReader, &fatal,
            nullptr, kProducer, kProducer
        );
    const uint32_t duplicate_keys[3] = {
        pa_up_keys[0], pa_up_keys[0], pa_up_keys[2]
    };
    fatal = 0;
    const bool pa_up_duplicate_rejected =
        !CommitPreparedSymbolSharedWriterIntentSet<
            SymbolTestOps, false, false, true, true
        >(
            *map, duplicate_keys, 3, kReader, &fatal,
            nullptr, kProducer, kProducer
        );
    FdwicOutputRef wrong_producer_ref = output_ref;
    wrong_producer_ref.producer_task_id = kWriter;
    wrong_producer_ref.output_slot = 0;
    uint32_t wrong_producer_key = 0;
    Check(
        SharedSymbolHistoryKey(
            wrong_producer_ref, wrong_producer_key
        ),
        "PA UP negative setup derives a different producer key"
    );
    const uint32_t wrong_producer_keys[3] = {
        pa_up_keys[0], wrong_producer_key, pa_up_keys[2]
    };
    fatal = 0;
    const bool pa_up_producer_rejected =
        !CommitPreparedSymbolSharedWriterIntentSet<
            SymbolTestOps, false, false, true, true
        >(
            *map, wrong_producer_keys, 3, kReader, &fatal,
            nullptr, kProducer, kProducer
        );
    Check(
        pa_up_count_rejected && pa_up_order_rejected &&
            pa_up_duplicate_rejected &&
            pa_up_producer_rejected &&
            map->writer_history[kReader].magic == 0 &&
            map->shared_outputs[kProducer]
                .last_writer[0].value == kProducer &&
            map->shared_outputs[kProducer]
                .last_writer[1].value == kProducer &&
            map->shared_outputs[kProducer]
                .last_writer[2].value == kProducer,
        "PA UP compact path rejects wrong shape before GM publication"
    );

    // 多 symbol CAS 保留原有非事务语义：第二项冲突时第一项已经发布，
    // 后续项保持旧值，history 仍完整描述这一故障现场。
    ResetSharedState(*map);
    for (uint32_t slot = 0; slot < 3; ++slot) {
        map->shared_outputs[kProducer]
            .last_writer[slot].value = kProducer;
    }
    map->shared_outputs[kProducer]
        .last_writer[0].value = kWriter;
    fatal = 0;
    const bool pa_up_partial_rejected =
        !CommitPreparedSymbolSharedWriterIntentSet<
            SymbolTestOps, false, false, true, true
        >(
            *map, pa_up_keys, 3, kReader, &fatal,
            nullptr, kProducer, kProducer
        );
    Check(
        pa_up_partial_rejected &&
            map->writer_history[kReader].magic ==
                kSharedWriterHistoryMagic &&
            map->writer_history[kReader].count == 3 &&
            map->shared_outputs[kProducer]
                .last_writer[2].value == kReader &&
            map->shared_outputs[kProducer]
                .last_writer[0].value == kWriter &&
            map->shared_outputs[kProducer]
                .last_writer[1].value == kReader,
        "PA UP compact path preserves a failed CAS publication prefix"
    );

    ResetSharedState(*map);
    map->shared_outputs[kProducer].last_writer[0].value = kProducer;
    fatal = 0;
    SymbolTestOps::wait_address =
        &map->shared_outputs[kProducer].last_writer[0].value;
    SymbolTestOps::wait_loads.store(0, std::memory_order_relaxed);
    const bool wrong_previous_rejected =
        !CommitPreparedSymbolSharedWriterIntentSet<
            SymbolTestOps, false, false, true
        >(
            *map, &symbol_key, 1, kReader, &fatal,
            nullptr, kWriter
        );
    SymbolTestOps::wait_address = nullptr;
    Check(
        wrong_previous_rejected &&
            SymbolTestOps::wait_loads.load(
                std::memory_order_relaxed
            ) == 0 &&
            map->shared_outputs[kProducer].last_writer[0].value ==
                kProducer,
        "PA expected-previous CAS rejects a mismatched writer"
    );
}

void TestPublicationWaitFailuresFailClosed() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    const FdwicOutputRef output_ref{0, 0, 0, 0, 0, 0};

    ResetSharedState(*map);
    map->shared_outputs[0].published[0].value = 7;
    volatile int32_t fatal = 0;
    SymbolTestOps::now_calls.store(0, std::memory_order_relaxed);
    Check(
        !WaitForSharedOutputPublished<SymbolTestOps>(
            *map, output_ref, &fatal
        ),
        "unexpected publication value is rejected"
    );
    Check(fatal == 1, "unexpected publication value broadcasts fatal");
    Check(
        SymbolTestOps::now_calls.load(std::memory_order_relaxed) == 0,
        "unexpected ready value fails before opening a watchdog window"
    );

    ResetSharedState(*map);
    fatal = 1;
    SymbolTestOps::now_calls.store(0, std::memory_order_relaxed);
    Check(
        !WaitForSharedOutputPublished<SymbolTestOps>(
            *map, output_ref, &fatal
        ),
        "existing fatal terminates unpublished wait"
    );
    Check(
        SymbolTestOps::now_calls.load(std::memory_order_relaxed) == 1,
        "fatal wait exits before a watchdog recheck"
    );

    ResetSharedState(*map);
    fatal = 0;
    ExpiredWaitOps::calls.store(0, std::memory_order_relaxed);
    Check(
        !WaitForSharedOutputPublished<ExpiredWaitOps>(
            *map, output_ref, &fatal
        ),
        "watchdog terminates permanently unpublished symbol"
    );
    Check(fatal == 1, "publication watchdog broadcasts fatal");
    Check(
        ExpiredWaitOps::calls.load(std::memory_order_relaxed) == 2,
        "watchdog clock is read only at slow-path begin and periodic recheck"
    );
}

void TestInvalidReferencesFailClosed() {
    auto map = std::make_unique<SharedTensorMapSidecar>();
    ResetSharedState(*map);
    auto payload = std::make_unique<TaskPayload>();

    const FdwicOutputRef invalid_refs[] = {
        FdwicOutputRef{3, 0, 0, 0, 0, 0},
        FdwicOutputRef{0, 8, 0, 0, 0, 0},
        FdwicOutputRef{0, 0, 1, 1, 16, 0},
    };
    for (const FdwicOutputRef reference : invalid_refs) {
        TaskArgs args;
        ConstructTaskArgs(args);
        AppendSharedOutputRef(args, reference, TensorArgType::Input);
        LocalStats stats{};
        int32_t fanin[kMaxFanin] = {};
        bool protocol_ok = true;
        uint32_t ordinary_lookups = UINT32_MAX;
        (void)CollectSharedFanin<SymbolTestOps>(
            *map, args, 2, kHeapWindow, stats, fanin,
            protocol_ok, ordinary_lookups
        );
        Check(!protocol_ok, "invalid/future/view symbol fails closed");
        Check(ordinary_lookups == 0, "invalid symbol does not enter region map");
    }

    // 第一项是合法 INOUT，第二项才非法；两遍解析必须在第一遍就拒绝，
    // 不能提前改写 writer、payload、统计或输出 fanin。
    TensorDesc published = MakeTensor(0x400000000ULL, 0);
    SubmitContext producer{};
    producer.task_id = 0;
    producer.result.task_id = 0;
    producer.result.count = 1;
    producer.result.tensors[0] = &published;
    producer.shared_result.Reset(0);
    Check(producer.shared_result.AddOutputRef(0, 0), "late-failure producer output");
    Check(
        PublishSharedTaskOutputs<SymbolTestOps>(*map, producer, 0),
        "late-failure producer publishes"
    );

    TaskArgs mixed_args;
    ConstructTaskArgs(mixed_args);
    AppendSharedOutputRef(
        mixed_args, producer.shared_result.OutputRef(0), TensorArgType::Inout
    );
    AppendSharedOutputRef(
        mixed_args, FdwicOutputRef{3, 0, 0, 0, 0, 0},
        TensorArgType::Input
    );
    std::memset(payload.get(), 0xA5, sizeof(*payload));
    LocalStats mixed_stats{};
    int32_t mixed_fanin[kMaxFanin];
    for (uint32_t index = 0; index < kMaxFanin; ++index) {
        mixed_fanin[index] = -77;
    }
    bool mixed_protocol_ok = true;
    uint32_t mixed_ordinary_lookups = UINT32_MAX;
    (void)CollectSharedFanin<SymbolTestOps>(
        *map, mixed_args, 2, kHeapWindow, mixed_stats,
        mixed_fanin, mixed_protocol_ok, mixed_ordinary_lookups
    );
    Check(!mixed_protocol_ok, "late invalid symbol rejects whole resolve");
    Check(
        map->shared_outputs[0].last_writer[0].value == 0,
        "late invalid symbol does not publish earlier INOUT writer"
    );
    Check(
        mixed_stats.result.shared_symbol_input_loads == 0 &&
            mixed_stats.result.shared_symbol_inout_commits == 0,
        "late invalid symbol publishes no resolve statistics"
    );
    Check(
        mixed_fanin[0] == -77,
        "late invalid symbol publishes no fanin"
    );
    Check(
        AllBytesEqual(payload.get(), sizeof(*payload), 0xA5),
        "late invalid symbol does not touch payload scratch"
    );

    // 合法 INPUT 会先命中局部计数，随后才发现非法引用；统计与 fanin
    // 必须等整批验证成功后才发布，不能把局部累计泄露成半次提交。
    TaskArgs late_input_args;
    ConstructTaskArgs(late_input_args);
    AppendSharedOutputRef(
        late_input_args, producer.shared_result.OutputRef(0),
        TensorArgType::Input
    );
    AppendSharedOutputRef(
        late_input_args, FdwicOutputRef{3, 0, 0, 0, 0, 0},
        TensorArgType::Input
    );
    LocalStats late_input_stats{};
    int32_t late_input_fanin[kMaxFanin];
    for (uint32_t index = 0; index < kMaxFanin; ++index) {
        late_input_fanin[index] = -91;
    }
    bool late_input_protocol_ok = true;
    uint32_t late_input_ordinary_lookups = UINT32_MAX;
    (void)CollectSharedFanin<SymbolTestOps>(
        *map, late_input_args, 2, kHeapWindow, late_input_stats,
        late_input_fanin, late_input_protocol_ok,
        late_input_ordinary_lookups
    );
    Check(
        !late_input_protocol_ok,
        "late invalid symbol rejects preceding valid INPUT"
    );
    Check(
        late_input_stats.result.shared_symbol_input_loads == 0,
        "late invalid symbol publishes no partial INPUT statistic"
    );
    Check(
        late_input_fanin[0] == -91,
        "late invalid symbol publishes no partial INPUT fanin"
    );
    Check(
        AllBytesEqual(payload.get(), sizeof(*payload), 0xA5),
        "late invalid INPUT pair leaves payload scratch untouched"
    );

    TaskArgs duplicate_args;
    ConstructTaskArgs(duplicate_args);
    AppendSharedOutputRef(
        duplicate_args, producer.shared_result.OutputRef(0),
        TensorArgType::Inout
    );
    AppendSharedOutputRef(
        duplicate_args, producer.shared_result.OutputRef(0),
        TensorArgType::OutputExisting
    );
    LocalStats duplicate_stats{};
    int32_t duplicate_fanin[kMaxFanin] = {};
    bool duplicate_protocol_ok = true;
    uint32_t duplicate_ordinary_lookups = UINT32_MAX;
    (void)CollectSharedFanin<SymbolTestOps>(
        *map, duplicate_args, 2, kHeapWindow,
        duplicate_stats, duplicate_fanin, duplicate_protocol_ok,
        duplicate_ordinary_lookups
    );
    Check(
        !duplicate_protocol_ok,
        "duplicate write references to one symbol fail before commit"
    );
    Check(
        map->shared_outputs[0].last_writer[0].value == 0,
        "duplicate write rejection preserves producer writer"
    );
}

}  // namespace

int main() {
    TestSharedCompletionPublishesWithoutFrontier();
    TestPublishAndResolve();
    TestPaTwoGroupWriterReadyGate();
    TestPaWriterIntentPreGateFailuresDoNotPublishGate();
    TestWriterCommitFailuresKeepTerminalEvidence();
    TestMultiWriterFailureKeepsPartialTerminalEvidence();
    TestFailedSealDiscardsBuiltTask();
    TestCase1RegistrationBypassesRegionSequencer();
    TestPostBuildSealClosesSuccessAndFailurePaths();
    TestPublicationPreflightIsAllOrNothing();
    TestPublicationCommitFaultsRollback();
    TestConsumerWaitsForDelayedPublication();
    TestPaUpSplitWriterPublicationStages();
    TestOrderedPreparedSymbolUsesSinglePublicationCheck();
    TestPublicationWaitFailuresFailClosed();
    TestInvalidReferencesFailClosed();
    if (g_failures != 0) {
        std::fprintf(stderr, "[FAIL] shared-output symbol tests: %d\n", g_failures);
        return 1;
    }
    std::puts("[PASS] shared-output symbol publish/resolve tests");
    return 0;
}
