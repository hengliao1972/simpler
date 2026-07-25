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
        std::this_thread::yield();
    }
};

volatile int64_t *SymbolTestOps::wait_address = nullptr;
std::atomic<uint64_t> SymbolTestOps::wait_loads{0};
std::atomic<uint64_t> SymbolTestOps::now_calls{0};

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

    // PA Case1 的 fresh symbol 不存在 A -> B(INOUT) -> C(INPUT) 链。
    // writer 已从 producer 改成唯一 UP consumer 后，任何后继复用同一
    // producer handle 都必须 fail-closed，不能脑补成通用 generation 链。
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
    const uint32_t successor_count = CollectSharedFanin<SymbolTestOps>(
        *map, successor_args, 3, kHeapWindow,
        successor_stats, successor_fanin, protocol_ok, ordinary_lookups
    );
    Check(!protocol_ok, "post-INOUT successor is outside the Case1 symbol contract");
    Check(
        successor_count == 0,
        "unsupported writer chain publishes no successor fanin"
    );
    Check(
        SymbolTestOps::now_calls.load(std::memory_order_relaxed) == 0,
        "already-published symbol fast path never reads the watchdog clock"
    );
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
    TestWriterCommitFailuresKeepTerminalEvidence();
    TestMultiWriterFailureKeepsPartialTerminalEvidence();
    TestFailedSealDiscardsBuiltTask();
    TestCase1RegistrationBypassesRegionSequencer();
    TestPostBuildSealClosesSuccessAndFailurePaths();
    TestPublicationPreflightIsAllOrNothing();
    TestPublicationCommitFaultsRollback();
    TestConsumerWaitsForDelayedPublication();
    TestPublicationWaitFailuresFailClosed();
    TestInvalidReferencesFailClosed();
    if (g_failures != 0) {
        std::fprintf(stderr, "[FAIL] shared-output symbol tests: %d\n", g_failures);
        return 1;
    }
    std::puts("[PASS] shared-output symbol publish/resolve tests");
    return 0;
}
