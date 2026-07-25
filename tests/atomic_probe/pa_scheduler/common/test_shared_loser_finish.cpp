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

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <sys/mman.h>
#include <unistd.h>

#define PA_DEVICE inline
#define PA_GM
#define PA_COMPETE_FIRST_SPLIT_FINISH 1
#include "pa_scheduler_core.h"

namespace {

using namespace pa_scheduler;

// 该 Ops 只服务 shared split-finish loser 定向用例。完整接口让公共模板按
// 与 CPU runner 相同的原子语义实例化；测试真正执行的 loser 路径不会调用
// kernel、共享 heap 或 TensorMap。
struct LoserFinishTestOps {
    static constexpr bool kAtomicReturnReadyObserved = false;
    static inline CompeteFirstSplitRuntimeState runtime{};
    static inline uint64_t tick = 0;
    static inline uint64_t finish_api_calls = 0;

    static CompeteFirstSplitRuntimeState &CompeteFirstSplitState() {
        return runtime;
    }

    static int32_t Load(volatile int32_t *address) {
        return __atomic_fetch_add(
            address, static_cast<int32_t>(0), __ATOMIC_ACQUIRE
        );
    }

    static int64_t Load(volatile int64_t *address) {
        return __atomic_fetch_add(
            address, static_cast<int64_t>(0), __ATOMIC_ACQUIRE
        );
    }

    static uint64_t Load(volatile uint64_t *address) {
        return __atomic_fetch_add(
            address, static_cast<uint64_t>(0), __ATOMIC_ACQUIRE
        );
    }

    static int32_t Exchange(volatile int32_t *address, int32_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t Exchange(volatile int64_t *address, int64_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static uint64_t Exchange(
        volatile uint64_t *address, uint64_t value
    ) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t FetchAdd(volatile int64_t *address, int64_t value) {
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t FetchMax(
        volatile int64_t *address, int64_t value, uint64_t &retries
    ) {
        int64_t current = __atomic_load_n(address, __ATOMIC_ACQUIRE);
        retries = 0;
        while (value > current) {
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
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }

    static uint64_t Now() {
        return ++tick;
    }

    template <typename T>
    static uint64_t NowAfterAtomicResult(T value) {
        asm volatile("" : "+r"(value));
        return Now();
    }

    static void ExecuteKernel(
        SchedulerState *, WorkerState &, TaskKind, uint32_t
    ) {}

    static void SpinHint() {}

    static void InvalidateRegion(const void *, uint64_t) {
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }

    static void FlushRegion(void *, uint64_t) {
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
    }

    static void Publish(uint64_t *address, uint64_t value) {
        __atomic_store_n(address, value, __ATOMIC_RELEASE);
    }

    static uint32_t FinishCompeteFirstCallback(
        const CallbackSubmitTicket *, const TaskArgs *
    ) {
        // shared loser 不应到达跨 TU finish。该入口只作为
        // 编译期完整 Ops 接口和动态陷阱；任何调用都会使定向测试失败。
        ++finish_api_calls;
        return 0;
    }
};

// winner 封口故障注入只在指定 int64 控制字上生效。注入发生在真实
// FinishCallbackSubmitBody 内，验证 Materialize、BuildWinner、writer
// commit、published 和失败 slot 撤销的整体顺序。
struct WinnerSealFaultOps : LoserFinishTestOps {
    using LoserFinishTestOps::Exchange;

    static inline volatile int64_t *exchange_race_address = nullptr;
    static inline volatile int64_t *fetch_race_address = nullptr;

    static int64_t Exchange(volatile int64_t *address, int64_t value) {
        if (address == exchange_race_address) {
            __atomic_store_n(address, int64_t{7}, __ATOMIC_RELEASE);
            exchange_race_address = nullptr;
        }
        return LoserFinishTestOps::Exchange(address, value);
    }

    static int64_t FetchMax(
        volatile int64_t *address, int64_t value, uint64_t &retries
    ) {
        if (address == fetch_race_address) {
            __atomic_store_n(address, int64_t{-1}, __ATOMIC_RELEASE);
            fetch_race_address = nullptr;
        }
        return LoserFinishTestOps::FetchMax(address, value, retries);
    }
};

void *MapAnonymous(size_t bytes, bool no_reserve) {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    if (no_reserve) {
        flags |= MAP_NORESERVE;
    }
#else
    (void)no_reserve;
#endif
    return mmap(nullptr, bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
}

SchedulerState *MapSchedulerState() {
    void *memory = MapAnonymous(sizeof(SchedulerState), true);
    if (memory == MAP_FAILED) {
        std::perror("mmap SchedulerState");
        return nullptr;
    }
    return ::new (memory) SchedulerState;
}

void ResetOutputCell(SharedOutputCell &cell) {
    std::memset(&cell, 0, sizeof(cell));
    for (uint32_t slot = 0; slot < kSharedOutputMaxPerTask; ++slot) {
        cell.published[slot].value = -1;
        cell.last_writer[slot].value = -1;
    }
}

TensorDesc MakeTestTensor(uint64_t address, uint32_t owner) {
    TensorDesc tensor{};
    tensor.buffer_addr = address;
    tensor.buffer_size = 1024;
    tensor.owner_task_id = owner;
    tensor.ndims = 1;
    tensor.dtype = DataType::Float32;
    tensor.is_contiguous = true;
    tensor.shapes[0] = 256;
    tensor.strides[0] = 1;
    tensor.extent_elem_cache = 256;
    return tensor;
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

bool RunReadySharedDescriptorDirectToSlotTest() {
    SchedulerState *state = MapSchedulerState();
    if (state == nullptr) {
        return false;
    }
    state->fatal.value = 0;
    state->heap_base = kSyntheticHeapBase;
    state->heap_size = kHeapBytes;
    state->heap_window = kHeapWindow;
    state->shared_map.committed_tasks.value = 0;
    state->shared_map.reclaim_upto.value = -1;
    ResetOutputCell(state->shared_map.shared_outputs[0]);
    ResetOutputCell(state->shared_map.shared_outputs[4]);

    const TensorDesc published =
        MakeTestTensor(kSyntheticHeapBase + 4096, 0);
    SharedOutputCell &producer = state->shared_map.shared_outputs[0];
    producer.tensors[0] = published;
    producer.last_writer[0].value = 0;
    producer.published[0].value = 0;

    WorkerState &worker = state->workers[0];
    worker.role = CoreRole::Aiv;
    worker.core_idx = static_cast<int32_t>(kAicWorkers);
    worker.lane = 1;
    worker.local_index = 4;
    worker.occupied_count = 0;

    TaskArgs args;
    ConstructTaskArgs(args);
    AppendSharedOutputRef(
        args, FdwicOutputRef{0, 0, 0, 0, 0, 0},
        TensorArgType::Input
    );

    SubmitContext context{};
    BeginCallbackSubmit(worker, context);
    bool ok =
        context.task_id == 4 &&
        PrepareSharedTaskOutputs(
            context.shared_result, 4, TaskKind::Up
        );
    std::memset(context.payload, 0xA5, sizeof(*context.payload));

    LocalStats stats{};
    bool pmu_context = false;
    const CallbackSubmitTicket ticket{
        1, 4, static_cast<int16_t>(FunctionId(TaskKind::Up)), 1, 0
    };
    const bool finished =
        FinishCallbackSubmitBody<LoserFinishTestOps, false>(
            state, worker, kTasksPerBatch, args, context, stats,
            pmu_context, ticket
        );

    const LocalSlot &slot = worker.slots[0];
    ok &= finished;
    ok &= state->fatal.value == 0;
    ok &= AllBytesEqual(context.payload, sizeof(*context.payload), 0xA5);
    ok &= worker.occupied_count == 1;
    ok &= slot.occupied && slot.built && slot.task_id == 4;
    ok &= slot.tensor_count == 1;
    ok &= std::memcmp(&slot.tensors[0], &published, sizeof(published)) == 0;
    ok &= slot.args[0] ==
          static_cast<uint64_t>(
              reinterpret_cast<uintptr_t>(&slot.tensors[0])
          );
    ok &= slot.fanin_count == 1 && slot.fanin[0] == 0;
    ok &= context.fanin_count == 1 && context.fanin[0] == 0;
    ok &= stats.result.shared_symbol_input_loads == 1;
    ok &= stats.result.slot_tensor_copies == 1;
    ok &= stats.result.submits == 1;
    ok &= producer.published[0].value == 0;
    ok &= producer.last_writer[0].value == 0;

    (void)munmap(state, sizeof(SchedulerState));
    return ok;
}

bool RunFastLoserSubmitTest() {
    void *state_memory = MapAnonymous(sizeof(SchedulerState), true);
    if (state_memory == MAP_FAILED) {
        std::perror("mmap SchedulerState");
        return false;
    }
    // default-initialization 只建立 trivial 对象生命周期；匿名映射已经按页
    // 提供零值，不能用 value-initialization 触碰整个约 1 GiB 状态。
    auto *state = ::new (state_memory) SchedulerState;
    WorkerState &worker = state->workers[0];
    worker.role = CoreRole::Aic;
    worker.core_idx = 0;
    worker.block_id = 0;
    worker.lane = 0;
    worker.sub_block_id = 0;
    worker.local_index = 0;
    worker.occupied_count = 0;
    state->fatal.value = 0;
    state->heap_base = kSyntheticHeapBase;
    state->heap_size = kHeapBytes;
    state->heap_window = kHeapWindow;
    state->alloc_cursor[0].value = 0;
    state->cube_cursor[1].value = 1;
    state->cube_cursor[3].value = 3;
    for (uint32_t task_id = 0; task_id < kTasksPerBatch; ++task_id) {
        ResetOutputCell(state->shared_map.shared_outputs[task_id]);
    }

    const long raw_page_size = sysconf(_SC_PAGESIZE);
    if (raw_page_size <= 0) {
        std::fprintf(stderr, "[FAIL] invalid host page size\n");
        (void)munmap(state_memory, sizeof(SchedulerState));
        return false;
    }
    const size_t page_size = static_cast<size_t>(raw_page_size);
    const size_t args_bytes =
        (sizeof(TaskArgs) + page_size - 1) / page_size * page_size;
    void *args_memory = MapAnonymous(args_bytes, false);
    if (args_memory == MAP_FAILED) {
        std::perror("mmap TaskArgs");
        (void)munmap(state_memory, sizeof(SchedulerState));
        return false;
    }
    auto *args = ::new (args_memory) TaskArgs;
    ConstructTaskArgs(*args);
    int32_t context_len = 8192;
    PaOrchestrationState orchestration{};
    InitPaOrchestration(orchestration, 1, &context_len);
    BeginPaBatchForCallback(orchestration, 0);
    SubmitContext context{};
    LocalStats stats{};
    bool pmu_context = false;
    LoserFinishTestOps::tick = 0;
    LoserFinishTestOps::finish_api_calls = 0;
    LoserFinishTestOps::runtime = CompeteFirstSplitRuntimeState{};
    const auto is_ref = [](FdwicOutputRef ref, int32_t task, int16_t slot) {
        return IsValidSharedOutputRef(ref) &&
               ref.producer_task_id == task &&
               ref.output_slot == slot &&
               ref.flags == 0;
    };
    bool ok = true;
    ok &= SubmitCallbackTask<
              TaskKind::Alloc, LoserFinishTestOps, false
          >(
              state, worker, kTasksPerBatch, orchestration, *args, 0,
              context, stats, pmu_context
          );
    ok &= context.shared_result.TaskId() == 0;
    ok &= context.shared_result.Size() == 3;
    AcceptTaskOutputs(
        orchestration, TaskKind::Alloc, OrchestrationOutputs(context)
    );
    ok &= is_ref(orchestration.accumulated_output, 0, 0);
    ok &= is_ref(orchestration.accumulated_sum, 0, 1);
    ok &= is_ref(orchestration.accumulated_max, 0, 2);

    // Alloc 的三项轻量 Output 构参已经完成。随后把同一 args 页设为
    // PROT_NONE；四个 kernel loser 若读取上一 task 的任何 args 字段，
    // 测试会在实际访问点直接失败。
    if (mprotect(args_memory, args_bytes, PROT_NONE) != 0) {
        std::perror("mprotect TaskArgs");
        (void)munmap(args_memory, args_bytes);
        (void)munmap(state_memory, sizeof(SchedulerState));
        return false;
    }
    PreparePaBlockGroup(orchestration, 0);
    ok &= SubmitCallbackTask<TaskKind::Qk, LoserFinishTestOps, false>(
        state, worker, kTasksPerBatch, orchestration, *args, 0,
        context, stats, pmu_context
    );
    ok &= context.shared_result.TaskId() == 1;
    ok &= context.shared_result.Size() == 1;
    AcceptTaskOutputs(
        orchestration, TaskKind::Qk, OrchestrationOutputs(context)
    );
    ok &= is_ref(orchestration.qk_scores, 1, 0);
    ok &= SubmitCallbackTask<TaskKind::Sf, LoserFinishTestOps, false>(
        state, worker, kTasksPerBatch, orchestration, *args, 0,
        context, stats, pmu_context
    );
    ok &= context.shared_result.TaskId() == 2;
    ok &= context.shared_result.Size() == 3;
    AcceptTaskOutputs(
        orchestration, TaskKind::Sf, OrchestrationOutputs(context)
    );
    ok &= is_ref(orchestration.sf_probs, 2, 0);
    ok &= is_ref(orchestration.sf_max, 2, 1);
    ok &= is_ref(orchestration.sf_sum, 2, 2);
    ok &= SubmitCallbackTask<TaskKind::Pv, LoserFinishTestOps, false>(
        state, worker, kTasksPerBatch, orchestration, *args, 0,
        context, stats, pmu_context
    );
    ok &= context.shared_result.TaskId() == 3;
    ok &= context.shared_result.Size() == 1;
    AcceptTaskOutputs(
        orchestration, TaskKind::Pv, OrchestrationOutputs(context)
    );
    ok &= is_ref(orchestration.pv_output, 3, 0);
    ok &= SubmitCallbackTask<TaskKind::Up, LoserFinishTestOps, false>(
        state, worker, kTasksPerBatch, orchestration, *args, 0,
        context, stats, pmu_context
    );
    ok &= context.shared_result.TaskId() == 4;
    ok &= context.shared_result.Size() == 0;
    ok &= is_ref(orchestration.accumulated_output, 0, 0);
    ok &= is_ref(orchestration.accumulated_sum, 0, 1);
    ok &= is_ref(orchestration.accumulated_max, 0, 2);

    ok &= LoserFinishTestOps::finish_api_calls == 0;
    ok &= stats.result.submits == kTasksPerBatch;
    ok &= stats.result.claim_attempts == 3;
    ok &= stats.result.claim_wins == 0;
    ok &= stats.result.arg_resets == 0;
    ok &= stats.result.tensor_args_added == 3;
    ok &= stats.result.materialized_outputs == 0;
    ok &= stats.result.map_inserts == 0;
    ok &= stats.result.submit_begin != 0;
    ok &= stats.result.submit_end > stats.result.submit_begin;
    ok &= worker.occupied_count == 0;
    ok &= worker.heap_next == 0;
    ok &= state->fatal.value == 0;
    ok &= state->shared_map.shared_heap_vend.value == 0;
    for (uint32_t shard = 0; shard < kSharedHeapShards; ++shard) {
        ok &= state->shared_map.shared_heap_cursor[shard].value == 0;
    }
    ok &= LoserFinishTestOps::runtime.finish_calls == 0;
    ok &= LoserFinishTestOps::runtime.finish_state_address == 0;
    ok &= LoserFinishTestOps::runtime.task_id_sum == 0;
    for (uint32_t task_id = 0; task_id < kTasksPerBatch; ++task_id) {
        const uint32_t output_count =
            FrontendTaskOutputCount(GetTaskKind(task_id));
        for (uint32_t output = 0; output < output_count; ++output) {
            ok &= state->shared_map.shared_outputs[task_id]
                      .published[output].value == -1;
            ok &= state->shared_map.shared_outputs[task_id]
                      .last_writer[output].value == -1;
        }
        ok &= state->tasks[task_id].flag == 0;
    }
    if (!ok) {
        std::fprintf(
            stderr,
            "[DETAIL] loser_fast task=%d outputs=%u finish_api=%llu "
            "submits=%llu attempts=%llu wins=%llu resets=%llu tensor_args=%llu "
            "materialized=%llu inserts=%llu begin=%llu end=%llu occupied=%u fatal=%d\n",
            context.shared_result.TaskId(),
            context.shared_result.Size(),
            static_cast<unsigned long long>(
                LoserFinishTestOps::finish_api_calls
            ),
            static_cast<unsigned long long>(stats.result.submits),
            static_cast<unsigned long long>(stats.result.claim_attempts),
            static_cast<unsigned long long>(stats.result.claim_wins),
            static_cast<unsigned long long>(stats.result.arg_resets),
            static_cast<unsigned long long>(stats.result.tensor_args_added),
            static_cast<unsigned long long>(
                stats.result.materialized_outputs
            ),
            static_cast<unsigned long long>(stats.result.map_inserts),
            static_cast<unsigned long long>(stats.result.submit_begin),
            static_cast<unsigned long long>(stats.result.submit_end),
            worker.occupied_count,
            state->fatal.value
        );
    }

    // 先恢复权限再撤销映射，便于内存诊断器区分测试刻意保护与真实越界。
    if (mprotect(args_memory, args_bytes, PROT_READ | PROT_WRITE) != 0) {
        std::perror("restore TaskArgs protection");
        ok = false;
    }
    (void)munmap(args_memory, args_bytes);
    (void)munmap(state_memory, sizeof(SchedulerState));
    return ok;
}

bool RunFutureTaskWithoutSequencerTest() {
    SchedulerState *state = MapSchedulerState();
    if (state == nullptr) {
        return false;
    }
    state->fatal.value = 0;
    state->heap_base = kSyntheticHeapBase;
    state->heap_size = kHeapBytes;
    state->heap_window = kHeapWindow;
    state->shared_map.committed_tasks.value = 0;
    state->shared_map.reclaim_upto.value = -1;
    ResetOutputCell(state->shared_map.shared_outputs[6]);

    WorkerState &worker = state->workers[0];
    worker.role = CoreRole::Aic;
    worker.core_idx = 0;
    worker.lane = 0;
    worker.local_index = 6;
    worker.occupied_count = 0;

    TaskArgs args;
    ConstructTaskArgs(args);
    TensorCreateInfo create_info{};
    const uint32_t shape[kMaxTensorDims] = {16, 0, 0, 0, 0};
    InitCreateInfo(create_info, shape, 1, DataType::Float32);
    AddOutput(args, create_info);

    SubmitContext context{};
    BeginCallbackSubmit(worker, context);
    bool ok =
        context.task_id == 6 &&
        PrepareSharedTaskOutputs(
            context.shared_result, 6, TaskKind::Qk
        );
    LocalStats stats{};
    bool pmu_context = false;
    const CallbackSubmitTicket ticket{
        1, 6, static_cast<int16_t>(FunctionId(TaskKind::Qk)), 1, 0
    };
    const bool finished =
        FinishCallbackSubmitBody<LoserFinishTestOps, false>(
            state, worker, 10, args, context, stats, pmu_context, ticket
        );

    ok &= finished;
    ok &= state->fatal.value == 0;
    ok &= state->shared_map.committed_tasks.value == 0;
    ok &= state->shared_map.reclaim_upto.value == -1;
    ok &= state->shared_map.buckets[0].head.value == 0;
    ok &= state->shared_map.buckets[0].tail.value == 0;
    ok &= state->shared_map.shared_outputs[6].published[0].value == 6;
    ok &= state->shared_map.shared_outputs[6].last_writer[0].value == 6;
    ok &= worker.occupied_count == 1;
    ok &= worker.slots[0].occupied && worker.slots[0].built;
    ok &= worker.slots[0].task_id == 6;
    ok &= stats.result.submits == 1;
    ok &= stats.result.materialized_outputs == 1;

    (void)munmap(state, sizeof(SchedulerState));
    return ok;
}

bool RunPreRaisedFatalStopsWinnerTest() {
    SchedulerState *state = MapSchedulerState();
    if (state == nullptr) {
        return false;
    }
    state->fatal.value = 1;
    state->heap_base = kSyntheticHeapBase;
    state->heap_size = kHeapBytes;
    state->heap_window = kHeapWindow;
    state->shared_map.committed_tasks.value = 0;
    state->shared_map.reclaim_upto.value = -1;
    ResetOutputCell(state->shared_map.shared_outputs[6]);

    WorkerState &worker = state->workers[0];
    worker.role = CoreRole::Aic;
    worker.core_idx = 0;
    worker.lane = 0;
    worker.local_index = 6;
    worker.occupied_count = 0;

    TaskArgs args;
    ConstructTaskArgs(args);
    TensorCreateInfo create_info{};
    const uint32_t shape[kMaxTensorDims] = {16, 0, 0, 0, 0};
    InitCreateInfo(create_info, shape, 1, DataType::Float32);
    AddOutput(args, create_info);

    SubmitContext context{};
    BeginCallbackSubmit(worker, context);
    bool ok =
        context.task_id == 6 &&
        PrepareSharedTaskOutputs(
            context.shared_result, 6, TaskKind::Qk
        );
    LocalStats stats{};
    bool pmu_context = false;
    const CallbackSubmitTicket ticket{
        1, 6, static_cast<int16_t>(FunctionId(TaskKind::Qk)), 1, 0
    };
    const bool finished =
        FinishCallbackSubmitBody<LoserFinishTestOps, false>(
            state, worker, 10, args, context, stats, pmu_context, ticket
        );

    // 这里模拟“另一核已广播 fatal，本核随后进入 finish”。终止态必须在
    // shared heap/slot/completion/symbol 任何副作用之前被观察到。
    ok &= !finished;
    ok &= state->fatal.value == 1;
    ok &=
        state->shared_map
            .shared_heap_cursor[6 % kSharedHeapShards].value == 0;
    ok &= state->shared_map.shared_heap_vend.value == 0;
    ok &= state->shared_map.shared_outputs[6]
              .published[0].value == -1;
    ok &= state->shared_map.shared_outputs[6]
              .last_writer[0].value == -1;
    ok &= state->tasks[6].flag == 0;
    ok &= worker.occupied_count == 0;
    ok &= !worker.slots[0].occupied && !worker.slots[0].built;
    ok &= stats.result.materialized_outputs == 0;
    ok &= stats.result.submits == 0;

    (void)munmap(state, sizeof(SchedulerState));
    return ok;
}

bool RunAllocPublicationFailureTest() {
    SchedulerState *state = MapSchedulerState();
    if (state == nullptr) {
        return false;
    }
    state->fatal.value = 0;
    state->heap_base = kSyntheticHeapBase;
    state->heap_size = kHeapBytes;
    state->heap_window = kHeapWindow;
    state->shared_map.committed_tasks.value = 0;
    state->shared_map.reclaim_upto.value = -1;
    ResetOutputCell(state->shared_map.shared_outputs[0]);

    WorkerState &worker = state->workers[0];
    worker.role = CoreRole::Aic;
    worker.core_idx = 0;
    worker.lane = 0;
    worker.local_index = 0;
    worker.occupied_count = 0;

    TaskArgs args;
    ConstructTaskArgs(args);
    TensorCreateInfo create_info{};
    const uint32_t shape[kMaxTensorDims] = {16, 0, 0, 0, 0};
    InitCreateInfo(create_info, shape, 1, DataType::Float32);
    AddOutput(args, create_info);
    AddOutput(args, create_info);
    AddOutput(args, create_info);

    SubmitContext context{};
    BeginCallbackSubmit(worker, context);
    bool ok =
        context.task_id == 0 &&
        PrepareSharedTaskOutputs(
            context.shared_result, 0, TaskKind::Alloc
        );
    LocalStats stats{};
    bool pmu_context = false;
    const CallbackSubmitTicket ticket{
        1, 0, static_cast<int16_t>(FunctionId(TaskKind::Alloc)), 1, 0
    };
    WinnerSealFaultOps::exchange_race_address =
        &state->shared_map.shared_outputs[0].published[0].value;
    const bool finished =
        FinishCallbackSubmitBody<WinnerSealFaultOps, false>(
            state, worker, kTasksPerBatch, args, context, stats,
            pmu_context, ticket
        );

    const TensorDesc zero{};
    ok &= !finished;
    ok &= WinnerSealFaultOps::exchange_race_address == nullptr;
    ok &= state->fatal.value == 1;
    ok &= state->shared_map.committed_tasks.value == 0;
    // Alloc CompleteTask 已先发布 completion；final symbol publication 失败
    // 后不能伪装成可事务回滚，只能依靠 fatal 使整轮结果无效。
    ok &= state->tasks[0].flag == 1;
    ok &= worker.occupied_count == 0;
    for (uint32_t slot = 0; slot < 3; ++slot) {
        ok &= state->shared_map.shared_outputs[0]
                  .published[slot].value == -1;
        ok &= state->shared_map.shared_outputs[0]
                  .last_writer[slot].value == -1;
        ok &= std::memcmp(
                  &state->shared_map.shared_outputs[0].tensors[slot],
                  &zero, sizeof(zero)
              ) == 0;
    }
    ok &= stats.result.materialized_outputs == 3;
    ok &= stats.result.submits == 0;
    ok &= state->shared_map.shared_heap_cursor[0].value == 3072;
    ok &= state->shared_map.shared_heap_vend.value == 3072;

    (void)munmap(state, sizeof(SchedulerState));
    return ok;
}

bool RunQkPublicationFailureTest() {
    SchedulerState *state = MapSchedulerState();
    if (state == nullptr) {
        return false;
    }
    state->fatal.value = 0;
    state->heap_base = kSyntheticHeapBase;
    state->heap_size = kHeapBytes;
    state->heap_window = kHeapWindow;
    state->shared_map.committed_tasks.value = 0;
    state->shared_map.reclaim_upto.value = -1;
    ResetOutputCell(state->shared_map.shared_outputs[1]);

    WorkerState &worker = state->workers[0];
    worker.role = CoreRole::Aic;
    worker.core_idx = 0;
    worker.lane = 0;
    worker.local_index = 1;
    worker.occupied_count = 0;

    TaskArgs args;
    ConstructTaskArgs(args);
    TensorCreateInfo create_info{};
    const uint32_t shape[kMaxTensorDims] = {16, 0, 0, 0, 0};
    InitCreateInfo(create_info, shape, 1, DataType::Float32);
    AddOutput(args, create_info);

    SubmitContext context{};
    BeginCallbackSubmit(worker, context);
    bool ok =
        context.task_id == 1 &&
        PrepareSharedTaskOutputs(
            context.shared_result, 1, TaskKind::Qk
        );
    LocalStats stats{};
    bool pmu_context = false;
    const CallbackSubmitTicket ticket{
        1, 1, static_cast<int16_t>(FunctionId(TaskKind::Qk)), 1, 0
    };
    WinnerSealFaultOps::exchange_race_address =
        &state->shared_map.shared_outputs[1].published[0].value;
    const bool finished =
        FinishCallbackSubmitBody<WinnerSealFaultOps, false>(
            state, worker, kTasksPerBatch, args, context, stats,
            pmu_context, ticket
        );

    const TensorDesc zero{};
    ok &= !finished;
    ok &= WinnerSealFaultOps::exchange_race_address == nullptr;
    ok &= state->fatal.value == 1;
    ok &= state->shared_map.committed_tasks.value == 0;
    ok &= state->shared_map.shared_outputs[1].published[0].value == -1;
    ok &= state->shared_map.shared_outputs[1].last_writer[0].value == -1;
    ok &= std::memcmp(
              &state->shared_map.shared_outputs[1].tensors[0],
              &zero, sizeof(zero)
          ) == 0;
    ok &= worker.occupied_count == 0;
    ok &= !worker.slots[0].occupied && !worker.slots[0].built;
    ok &= state->tasks[1].flag == 0;
    ok &= stats.result.submits == 0;
    ok &= stats.result.materialized_outputs == 1;
    // 64B raw tensor 数据按 shared allocator 的逐 output 1KiB 边界保留，
    // 因而 cursor/vend 验证的是 1024B reserve，而不是 descriptor 数据量。
    ok &= state->shared_map.shared_heap_cursor[1].value == 1024;
    ok &= state->shared_map.shared_heap_vend.value == 1024;

    (void)munmap(state, sizeof(SchedulerState));
    return ok;
}

bool RunUpWriterCommitFailureTest() {
    SchedulerState *state = MapSchedulerState();
    if (state == nullptr) {
        return false;
    }
    state->fatal.value = 0;
    state->heap_base = kSyntheticHeapBase;
    state->heap_size = kHeapBytes;
    state->heap_window = kHeapWindow;
    state->shared_map.committed_tasks.value = 0;
    state->shared_map.reclaim_upto.value = -1;
    ResetOutputCell(state->shared_map.shared_outputs[0]);
    ResetOutputCell(state->shared_map.shared_outputs[4]);
    for (uint32_t slot = 0; slot < 2; ++slot) {
        state->shared_map.shared_outputs[0].published[slot].value = 0;
        state->shared_map.shared_outputs[0].last_writer[slot].value = 0;
        state->shared_map.shared_outputs[0].tensors[slot] =
            MakeTestTensor(kSyntheticHeapBase + slot * 1024, 0);
    }

    WorkerState &worker = state->workers[0];
    worker.role = CoreRole::Aiv;
    worker.core_idx = static_cast<int32_t>(kAicWorkers);
    worker.lane = 1;
    worker.local_index = 4;
    worker.occupied_count = 0;

    TaskArgs args;
    ConstructTaskArgs(args);
    AppendSharedOutputRef(
        args, FdwicOutputRef{0, 0, 0, 0, 0, 0},
        TensorArgType::Inout
    );
    AppendSharedOutputRef(
        args, FdwicOutputRef{0, 1, 0, 0, 0, 0},
        TensorArgType::Inout
    );

    SubmitContext context{};
    BeginCallbackSubmit(worker, context);
    bool ok =
        context.task_id == 4 &&
        PrepareSharedTaskOutputs(
            context.shared_result, 4, TaskKind::Up
        );
    LocalStats stats{};
    bool pmu_context = false;
    const CallbackSubmitTicket ticket{
        1, 4, static_cast<int16_t>(FunctionId(TaskKind::Up)), 1, 0
    };
    WinnerSealFaultOps::fetch_race_address =
        &state->shared_map.shared_outputs[0].last_writer[1].value;
    const bool finished =
        FinishCallbackSubmitBody<WinnerSealFaultOps, false>(
            state, worker, kTasksPerBatch, args, context, stats,
            pmu_context, ticket
        );

    ok &= !finished;
    ok &= WinnerSealFaultOps::fetch_race_address == nullptr;
    ok &= state->fatal.value == 1;
    ok &= state->shared_map.committed_tasks.value == 0;
    ok &= state->shared_map.shared_outputs[0].last_writer[0].value == 4;
    ok &= state->shared_map.shared_outputs[0].last_writer[1].value == 4;
    ok &= state->shared_map.shared_outputs[0].published[0].value == 0;
    ok &= state->shared_map.shared_outputs[0].published[1].value == 0;
    ok &= state->shared_map.shared_outputs[4].published[0].value == -1;
    ok &= stats.result.shared_symbol_inout_commits == 1;
    ok &= context.fanin_count == 1 && context.fanin[0] == 0;
    ok &= worker.occupied_count == 0;
    ok &= !worker.slots[0].occupied && !worker.slots[0].built;
    ok &= state->tasks[4].flag == 0;
    ok &= stats.result.submits == 0;

    (void)munmap(state, sizeof(SchedulerState));
    return ok;
}

}  // namespace

int main() {
    const bool direct_slot_ok = RunReadySharedDescriptorDirectToSlotTest();
    const bool loser_ok = RunFastLoserSubmitTest();
    const bool future_task_ok = RunFutureTaskWithoutSequencerTest();
    const bool fatal_stop_ok = RunPreRaisedFatalStopsWinnerTest();
    const bool alloc_failure_ok = RunAllocPublicationFailureTest();
    const bool qk_failure_ok = RunQkPublicationFailureTest();
    const bool up_failure_ok = RunUpWriterCommitFailureTest();
    if (!direct_slot_ok || !loser_ok || !future_task_ok || !fatal_stop_ok ||
        !alloc_failure_ok || !qk_failure_ok || !up_failure_ok) {
        std::fprintf(
            stderr,
            "[FAIL] shared loser fast-return or winner seal regression: "
            "direct_slot=%d loser_fast=%d future_task=%d fatal_stop=%d "
            "alloc_failure=%d qk_failure=%d up_failure=%d\n",
            direct_slot_ok ? 1 : 0,
            loser_ok ? 1 : 0,
            future_task_ok ? 1 : 0,
            fatal_stop_ok ? 1 : 0,
            alloc_failure_ok ? 1 : 0,
            qk_failure_ok ? 1 : 0,
            up_failure_ok ? 1 : 0
        );
        return 1;
    }
    std::printf(
        "[PASS] shared loser fast-return and winner seal failure paths\n"
    );
    return 0;
}
