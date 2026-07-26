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
};

// non-final UP loser 必须先等待 winner 发布 writer-ready，随后才能返回
// orchestration 构造下一组。该 Ops 在第三次 SpinHint 时用同一原子接口
// 模拟 winner 发布；此前三次读取都观察到 -1，因此测试可以确认 Finish
// 确实进入了等待。TaskArgs 同时保持 PROT_NONE，门放行后也不能读取
// 任何参数字段。
struct GateReleaseLoserFinishTestOps : LoserFinishTestOps {
    static inline volatile int64_t *gate_address = nullptr;
    static inline int64_t gate_value = -1;
    static inline uint32_t spin_calls = 0;
    static inline int64_t release_old = 0;

    static void SpinHint() {
        ++spin_calls;
        if (spin_calls == 3) {
            release_old = LoserFinishTestOps::Exchange(
                gate_address, gate_value
            );
        }
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

bool RunSharedBatchPlanTest() {
    bool ok = true;
    const auto CheckPlan = [&](uint64_t context_length,
                               uint32_t expected_groups,
                               uint32_t expected_tasks) {
        SharedPaBatchPlan plan{};
        bool plan_ok = BuildSharedPaBatchPlan(
            context_length, 17, plan
        );
        plan_ok &= plan.batch_start == 17;
        plan_ok &= plan.group_count == expected_groups;
        plan_ok &= plan.task_count == expected_tasks;
        for (uint32_t offset = 0;
             plan_ok && offset < expected_tasks; ++offset) {
            SharedPaPlannedTask task{};
            plan_ok &= SharedPaPlannedTaskAt(plan, offset, task);
            const TaskKind expected_kind =
                offset == 0
                    ? TaskKind::Alloc
                    : static_cast<TaskKind>(
                          1U + ((offset - 1U) % 4U)
                      );
            const uint32_t expected_group =
                offset == 0 ? 0U : (offset - 1U) / 4U;
            plan_ok &= task.kind == expected_kind;
            plan_ok &= task.group_index == expected_group;
            plan_ok &=
                task.has_following_group ==
                (expected_kind == TaskKind::Up &&
                 expected_group + 1U < expected_groups);
            plan_ok &=
                task.is_last_in_batch ==
                (offset + 1U == expected_tasks);
            plan_ok &=
                SharedPaTaskOffset(task.kind, task.group_index) ==
                offset;
        }
        SharedPaPlannedTask outside{};
        plan_ok &= !SharedPaPlannedTaskAt(
            plan, expected_tasks, outside
        );
        return plan_ok;
    };

    // Alloc 在 group loop 之前；0/1/2/4 组分别对应 1/5/9/17 task。
    ok &= CheckPlan(0, 0, 1);
    ok &= CheckPlan(
        kPaBlocksPerRequest * kPaBlockSize, 1, 5
    );
    ok &= CheckPlan(
        2ULL * kPaBlocksPerRequest * kPaBlockSize, 2, 9
    );
    ok &= CheckPlan(kSharedPaMaxContextLength, 4, 17);

    // 跨过 64-block 边界一个 token 就必须进入第二组。
    ok &= CheckPlan(
        kPaBlocksPerRequest * kPaBlockSize + 1U, 2, 9
    );

    SharedPaBatchPlan rejected{};
    ok &= !BuildSharedPaBatchPlan(
        kSharedPaMaxContextLength + 1U, 0, rejected
    );
    ok &= !BuildSharedPaBatchPlan(
        UINT64_MAX, 0, rejected
    );
    ok &= BuildSharedPaBatchPlan(
        kSharedPaMaxContextLength,
        kMaxTasks - kSharedPaMaxTasksPerBatch, rejected
    );
    ok &= !BuildSharedPaBatchPlan(
        kSharedPaMaxContextLength,
        kMaxTasks - kSharedPaMaxTasksPerBatch + 1U, rejected
    );
    return ok;
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
        1, 4, static_cast<int16_t>(FunctionId(TaskKind::Up)), 1,
        EncodeSharedPaTaskMeta(TaskKind::Up, 0, false)
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

bool RunProtectedArgsLoserTest() {
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
    state->fatal.value = 0;

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
    std::memset(args, 0xA5, sizeof(TaskArgs));
    if (mprotect(args_memory, args_bytes, PROT_NONE) != 0) {
        std::perror("mprotect TaskArgs");
        (void)munmap(args_memory, args_bytes);
        (void)munmap(state_memory, sizeof(SchedulerState));
        return false;
    }

    CompeteFirstSplitRuntimeState &runtime =
        LoserFinishTestOps::CompeteFirstSplitState();
    runtime = CompeteFirstSplitRuntimeState{};
    runtime.scheduler = state;
    runtime.worker = &worker;
    runtime.task_count = 9;
    runtime.worker_id = 0;
    runtime.owner_worker_id = 0;
    runtime.caller_state_address = reinterpret_cast<uint64_t>(&runtime);
    runtime.state_cookie =
        CompeteFirstSplitStateCookie(0, CoreRole::Aic);

    bool ok = true;
    for (uint32_t task_id = 0; task_id < kTasksPerBatch; ++task_id) {
        const TaskKind kind = GetTaskKind(task_id);
        runtime.context = SubmitContext{};
        runtime.context.task_id = static_cast<int32_t>(task_id);
        // Claim loser 的真实返回值固定为 -1；kind 只能由 shared ticket
        // 元数据恢复，不能借 kernel_id 或 task_id % 5 旁路推断。
        runtime.context.kernel_id = -1;
        runtime.context.won = false;
        runtime.context.shared_result.Reset(static_cast<int32_t>(task_id));
        ok &= PrepareSharedTaskOutputs(
            runtime.context.shared_result, static_cast<int32_t>(task_id),
            kind
        );
        const CallbackSubmitTicket ticket{
            static_cast<uint64_t>(task_id + 1),
            task_id,
            -1,
            0,
            EncodeSharedPaTaskMeta(kind, 0, false),
        };
        ok &= FinishSplitCallbackSubmitFromRuntime<LoserFinishTestOps>(
                  &ticket, args
              ) == 1;
    }

    // 再用同一个 task4 验证 non-final UP 门协议。第一次调用已证明普通
    // final UP 不等门；本次只有 has_following bit 改变，Finish 必须先
    // 轮询 gate，再在完全不读保护页参数的前提下闭合 loser 尾部。
    constexpr uint32_t nonfinal_up_task = 4;
    runtime.context = SubmitContext{};
    runtime.context.task_id = static_cast<int32_t>(nonfinal_up_task);
    runtime.context.kernel_id = -1;
    runtime.context.won = false;
    runtime.context.shared_result.Reset(
        static_cast<int32_t>(nonfinal_up_task)
    );
    ok &= PrepareSharedTaskOutputs(
        runtime.context.shared_result,
        static_cast<int32_t>(nonfinal_up_task),
        TaskKind::Up
    );
    volatile int64_t *gate =
        &state->tasks[nonfinal_up_task].deps_prepared;
    __atomic_store_n(gate, int64_t{-1}, __ATOMIC_RELEASE);
    GateReleaseLoserFinishTestOps::gate_address = gate;
    GateReleaseLoserFinishTestOps::gate_value =
        static_cast<int64_t>(nonfinal_up_task);
    GateReleaseLoserFinishTestOps::spin_calls = 0;
    GateReleaseLoserFinishTestOps::release_old = 0;
    const CallbackSubmitTicket nonfinal_up_ticket{
        6,
        nonfinal_up_task,
        -1,
        0,
        EncodeSharedPaTaskMeta(TaskKind::Up, 0, true),
    };
    ok &=
        FinishSplitCallbackSubmitFromRuntime<
            GateReleaseLoserFinishTestOps
        >(&nonfinal_up_ticket, args) == 1;
    ok &= GateReleaseLoserFinishTestOps::spin_calls == 3;
    ok &= GateReleaseLoserFinishTestOps::release_old == -1;
    ok &= __atomic_load_n(gate, __ATOMIC_ACQUIRE) ==
        static_cast<int64_t>(nonfinal_up_task);
    GateReleaseLoserFinishTestOps::gate_address = nullptr;

    // 第二组最终 UP 的 task_id=8，如果沿用 %5 会被误判为 PV。loser
    // 的 function_id 仍为 -1，只有 ticket 中的 group/kind 能恢复语义。
    constexpr uint32_t final_up_task = 8;
    ok &= GetTaskKind(final_up_task) == TaskKind::Pv;
    ok &= FrontendTaskOutputCount(GetTaskKind(final_up_task)) == 1;
    ok &= FrontendTaskOutputCount(TaskKind::Up) == 0;
    runtime.context = SubmitContext{};
    runtime.context.task_id = static_cast<int32_t>(final_up_task);
    runtime.context.kernel_id = -1;
    runtime.context.won = false;
    runtime.context.shared_result.Reset(
        static_cast<int32_t>(final_up_task)
    );
    ok &= PrepareSharedTaskOutputs(
        runtime.context.shared_result, static_cast<int32_t>(final_up_task),
        TaskKind::Up
    );
    const CallbackSubmitTicket final_up_ticket{
        9,
        final_up_task,
        -1,
        0,
        EncodeSharedPaTaskMeta(TaskKind::Up, 1, false),
    };
    ok &= FinishSplitCallbackSubmitFromRuntime<LoserFinishTestOps>(
              &final_up_ticket, args
          ) == 1;

    ok &= runtime.finish_calls == kTasksPerBatch + 2;
    ok &= runtime.protocol_errors == 0;
    ok &= runtime.task_id_sum == 22;
    ok &= runtime.stats.result.submits == kTasksPerBatch + 2;
    ok &= runtime.stats.result.materialized_outputs == 0;
    ok &= runtime.stats.result.shared_symbol_input_loads == 0;
    ok &= runtime.stats.result.shared_symbol_inout_commits == 0;
    ok &= runtime.stats.result.fanin_edges == 0;
    ok &= runtime.stats.result.map_inserts == 0;
    ok &= worker.occupied_count == 0;
    ok &= state->fatal.value == 0;

    // 非法 ticket 必须在进入 Finish body、读取保护页 args 之前拒绝。
    // QK/PV 的输出数同为 1，专门用错误 PV function_id 搭配 QK kind，
    // 防止 shared_result.Size() 恰好相同而掩盖协议错配。
    state->fatal.value = 0;
    runtime.context = SubmitContext{};
    runtime.context.task_id = 1;
    runtime.context.kernel_id = FunctionId(TaskKind::Pv);
    runtime.context.won = true;
    runtime.context.shared_result.Reset(1);
    ok &= PrepareSharedTaskOutputs(
        runtime.context.shared_result, 1, TaskKind::Qk
    );
    const CallbackSubmitTicket wrong_winner_function{
        10,
        1,
        static_cast<int16_t>(FunctionId(TaskKind::Pv)),
        1,
        EncodeSharedPaTaskMeta(TaskKind::Qk, 0, false),
    };
    ok &= FinishSplitCallbackSubmitFromRuntime<LoserFinishTestOps>(
              &wrong_winner_function, args
          ) == 0;
    ok &= runtime.protocol_errors == 1;
    ok &= state->fatal.value == 1;

    // reserved bit 属于未来协议，当前版本必须 fail-closed，不能忽略后
    // 按看似合法的 kind/group 继续解释。
    state->fatal.value = 0;
    runtime.context = SubmitContext{};
    runtime.context.task_id = 1;
    runtime.context.kernel_id = -1;
    runtime.context.won = false;
    runtime.context.shared_result.Reset(1);
    ok &= PrepareSharedTaskOutputs(
        runtime.context.shared_result, 1, TaskKind::Qk
    );
    const CallbackSubmitTicket reserved_bit_ticket{
        11,
        1,
        -1,
        0,
        static_cast<uint8_t>(
            EncodeSharedPaTaskMeta(TaskKind::Qk, 0, false) |
            kSharedPaTicketMetaReserved
        ),
    };
    ok &= FinishSplitCallbackSubmitFromRuntime<LoserFinishTestOps>(
              &reserved_bit_ticket, args
          ) == 0;
    ok &= runtime.protocol_errors == 2;
    ok &= state->fatal.value == 1;

    // loser 的 Claim 结果只能是 -1；即使 kind 与 output count 都正确，
    // 也不能接受伪造的可执行 function_id。
    state->fatal.value = 0;
    runtime.context = SubmitContext{};
    runtime.context.task_id = 1;
    runtime.context.kernel_id = FunctionId(TaskKind::Qk);
    runtime.context.won = false;
    runtime.context.shared_result.Reset(1);
    ok &= PrepareSharedTaskOutputs(
        runtime.context.shared_result, 1, TaskKind::Qk
    );
    const CallbackSubmitTicket wrong_loser_function{
        12,
        1,
        static_cast<int16_t>(FunctionId(TaskKind::Qk)),
        0,
        EncodeSharedPaTaskMeta(TaskKind::Qk, 0, false),
    };
    ok &= FinishSplitCallbackSubmitFromRuntime<LoserFinishTestOps>(
              &wrong_loser_function, args
          ) == 0;

    // 即使 runtime.task_count 或后续 shared plan 自身损坏，固定容量也
    // 必须作为最后一道边界。该 final loser 不进入 Materialize，专门
    // 证明越界不能借“loser 不读 args”绕过 shared-output table 上限。
    state->fatal.value = 0;
    runtime.task_count = kMaxTasks + 1;
    runtime.context = SubmitContext{};
    runtime.context.task_id = static_cast<int32_t>(kMaxTasks);
    runtime.context.kernel_id = -1;
    runtime.context.won = false;
    runtime.context.shared_result.Reset(
        static_cast<int32_t>(kMaxTasks)
    );
    ok &= PrepareSharedTaskOutputs(
        runtime.context.shared_result,
        static_cast<int32_t>(kMaxTasks), TaskKind::Qk
    );
    const CallbackSubmitTicket capacity_overflow_loser{
        13,
        kMaxTasks,
        -1,
        0,
        EncodeSharedPaTaskMeta(TaskKind::Qk, 0, false),
    };
    ok &= FinishSplitCallbackSubmitFromRuntime<LoserFinishTestOps>(
              &capacity_overflow_loser, args
          ) == 0;
    ok &= runtime.finish_calls == kTasksPerBatch + 6;
    ok &= runtime.protocol_errors == 4;
    ok &= runtime.task_id_sum == 25 + kMaxTasks;
    ok &= runtime.stats.result.submits == kTasksPerBatch + 2;
    ok &= worker.occupied_count == 0;
    ok &= state->fatal.value == 1;

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
        1, 6, static_cast<int16_t>(FunctionId(TaskKind::Qk)), 1,
        EncodeSharedPaTaskMeta(TaskKind::Qk, 0, false)
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
        1, 6, static_cast<int16_t>(FunctionId(TaskKind::Qk)), 1,
        EncodeSharedPaTaskMeta(TaskKind::Qk, 0, false)
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
        1, 0, static_cast<int16_t>(FunctionId(TaskKind::Alloc)), 1,
        EncodeSharedPaTaskMeta(TaskKind::Alloc, 0, false)
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
        1, 1, static_cast<int16_t>(FunctionId(TaskKind::Qk)), 1,
        EncodeSharedPaTaskMeta(TaskKind::Qk, 0, false)
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
        1, 4, static_cast<int16_t>(FunctionId(TaskKind::Up)), 1,
        EncodeSharedPaTaskMeta(TaskKind::Up, 0, false)
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

bool RunFatalBlockedSuccessorDrainTest() {
    SchedulerState *state = MapSchedulerState();
    if (state == nullptr) {
        return false;
    }
    state->fatal.value = 0;
    state->tasks[4].flag = 0;

    WorkerState &worker = state->workers[0];
    worker.role = CoreRole::Aiv;
    worker.core_idx = static_cast<int32_t>(kAicWorkers);
    worker.lane = 1;
    worker.occupied_count = kUsableSlots;
    for (uint32_t index = 0; index < kUsableSlots; ++index) {
        LocalSlot &slot = worker.slots[index];
        slot.occupied = true;
        slot.built = true;
        slot.task_id = 8 + index;
        slot.kind = static_cast<uint32_t>(FunctionId(TaskKind::Up));
        slot.fanin_count = 1;
        slot.fanin[0] = 4;
    }

    bool ok = true;
    ok &= !DiscardSharedSlotsAfterReplayFatal<LoserFinishTestOps>(
        state, worker
    );
    ok &= worker.occupied_count == kUsableSlots;
    ok &= worker.slots[0].occupied && worker.slots[1].occupied;

    // 模拟 non-final UP 已放门、后继 slot 已建立，但 task4 最终封口
    // 失败并广播 fatal。RingBp 必须在有界轮询后退出；final barrier
    // 证明无人再生产 slot 后，再统一撤销这些永远等不到 flag 的后继。
    state->fatal.value = 1;
    LocalStats stats{};
    bool fatal_exit = false;
    WaitForSlot<LoserFinishTestOps, false>(
        state, worker, 10, stats, fatal_exit
    );
    ok &= fatal_exit;
    ok &= stats.result.wait_iterations[0] == 1024;
    ok &= worker.occupied_count == kUsableSlots;
    ok &= DiscardSharedSlotsAfterReplayFatal<LoserFinishTestOps>(
        state, worker
    );
    ok &= worker.occupied_count == 0;
    for (uint32_t index = 0; index < kPrivateSlots; ++index) {
        ok &= !worker.slots[index].occupied;
        ok &= !worker.slots[index].built;
    }
    // 清执行资格不抹掉诊断身份，仍可确认阻塞源是 task4。
    ok &= worker.slots[0].task_id == 8;
    ok &= worker.slots[0].fanin_count == 1;
    ok &= worker.slots[0].fanin[0] == 4;
    ok &= state->tasks[4].flag == 0;
    ok &= state->tasks[8].flag == 0;
    for (uint32_t index = 0;
         index < static_cast<uint32_t>(TaskKind::Count) - 1U;
         ++index) {
        ok &= stats.result.kernel_counts[index] == 0;
    }
    for (uint32_t index = 0;
         index < static_cast<uint32_t>(DrainPlace::Count);
         ++index) {
        ok &= stats.result.placement[index] == 0;
    }
    ok &= state->fatal.value == 1;

    // 即使 occupied_count 已损坏，终止清理也不能因返回异常而留下
    // 二次死锁；false 只保留计数不一致证据。
    worker.slots[0].occupied = true;
    worker.slots[0].built = true;
    worker.occupied_count = 2;
    ok &= !DiscardSharedSlotsAfterReplayFatal<LoserFinishTestOps>(
        state, worker
    );
    ok &= worker.occupied_count == 0;
    ok &= !worker.slots[0].occupied && !worker.slots[0].built;

    (void)munmap(state, sizeof(SchedulerState));
    return ok;
}

bool RunTwoGroupExplicitFinishTest() {
    SchedulerState *state = MapSchedulerState();
    if (state == nullptr) {
        return false;
    }
    state->fatal.value = 0;
    state->heap_base = kSyntheticHeapBase;
    state->heap_size = kHeapBytes;
    state->heap_window = kHeapWindow;
    constexpr int32_t alloc = 0;
    constexpr int32_t first_sf = 2;
    constexpr int32_t first_pv = 3;
    constexpr int32_t first_up = 4;
    constexpr int32_t second_sf = 6;
    constexpr int32_t second_pv = 7;
    constexpr int32_t second_up = 8;
    state->tasks[first_up].deps_prepared = -1;
    state->tasks[second_up].deps_prepared = -1;

    const auto OutputRef = [](int32_t producer, int16_t slot) {
        return FdwicOutputRef{producer, slot, 0, 0, 0, 0};
    };
    const auto SeedOutput = [&](int32_t producer, int16_t slot,
                                uint64_t address) {
        SharedOutputCell &cell =
            state->shared_map.shared_outputs[
                static_cast<uint32_t>(producer)
            ];
        cell.published[static_cast<uint32_t>(slot)].value = producer;
        cell.last_writer[static_cast<uint32_t>(slot)].value = producer;
        cell.tensors[static_cast<uint32_t>(slot)] =
            MakeTestTensor(address, static_cast<uint32_t>(producer));
    };
    ResetOutputCell(state->shared_map.shared_outputs[alloc]);
    ResetOutputCell(state->shared_map.shared_outputs[first_sf]);
    ResetOutputCell(state->shared_map.shared_outputs[first_pv]);
    ResetOutputCell(state->shared_map.shared_outputs[first_up]);
    ResetOutputCell(state->shared_map.shared_outputs[second_sf]);
    ResetOutputCell(state->shared_map.shared_outputs[second_pv]);
    ResetOutputCell(state->shared_map.shared_outputs[second_up]);
    SeedOutput(alloc, 0, 0x410000000ULL);
    SeedOutput(alloc, 1, 0x410001000ULL);
    SeedOutput(alloc, 2, 0x410002000ULL);
    SeedOutput(first_sf, 1, 0x420001000ULL);
    SeedOutput(first_sf, 2, 0x420002000ULL);
    SeedOutput(first_pv, 0, 0x430000000ULL);
    SeedOutput(second_sf, 1, 0x440001000ULL);
    SeedOutput(second_sf, 2, 0x440002000ULL);
    SeedOutput(second_pv, 0, 0x450000000ULL);

    PaOrchestrationState first_orchestration{};
    InitPaOrchestration(first_orchestration, 1, nullptr);
    first_orchestration.current_batch = 0;
    first_orchestration.current_sequence =
        2ULL * kPaBlocksPerRequest * kPaBlockSize;
    first_orchestration.current_blocks =
        2ULL * kPaBlocksPerRequest;
    PreparePaBlockGroup(first_orchestration, 0);
    first_orchestration.accumulated_output = OutputRef(alloc, 0);
    first_orchestration.accumulated_sum = OutputRef(alloc, 1);
    first_orchestration.accumulated_max = OutputRef(alloc, 2);
    first_orchestration.sf_max = OutputRef(first_sf, 1);
    first_orchestration.sf_sum = OutputRef(first_sf, 2);
    first_orchestration.pv_output = OutputRef(first_pv, 0);

    TaskArgs first_args;
    LocalStats first_build_stats{};
    bool ok = BuildCallbackSubmitArgs<TaskKind::Up>(
        first_orchestration, first_args, 0, first_build_stats
    );

    WorkerState &worker = state->workers[0];
    worker.role = CoreRole::Aiv;
    worker.core_idx = static_cast<int32_t>(kAicWorkers);
    worker.lane = 1;
    worker.local_index = first_up;
    worker.occupied_count = 0;

    SubmitContext first_context{};
    BeginCallbackSubmit(worker, first_context);
    first_context.won = true;
    first_context.kernel_id = FunctionId(TaskKind::Up);
    ok &= first_context.task_id == first_up;
    ok &= PrepareSharedTaskOutputs(
        first_context.shared_result, first_up, TaskKind::Up
    );
    LocalStats first_stats{};
    bool first_pmu_context = false;
    const CallbackSubmitTicket first_ticket{
        1, first_up,
        static_cast<int16_t>(FunctionId(TaskKind::Up)), 1,
        EncodeSharedPaTaskMeta(TaskKind::Up, 0, true)
    };
    ok &= FinishCallbackSubmitBody<LoserFinishTestOps, false>(
        state, worker, 9, first_args, first_context, first_stats,
        first_pmu_context, first_ticket
    );
    const LocalSlot &first_slot = worker.slots[0];
    ok &= state->fatal.value == 0;
    ok &= state->tasks[first_up].deps_prepared == first_up;
    ok &= state->tasks[first_up].flag == 0;
    ok &= first_stats.result.shared_symbol_input_loads == 3;
    ok &= first_stats.result.shared_symbol_inout_commits == 3;
    ok &= first_stats.result.fanin_edges == 3;
    ok &= first_stats.result.dependency_signature ==
        0x7f405ca7dea83459ULL;
    ok &= first_slot.occupied && first_slot.built &&
        first_slot.task_id == static_cast<uint32_t>(first_up) &&
        first_slot.kind ==
            static_cast<uint32_t>(FunctionId(TaskKind::Up)) &&
        first_slot.fanin_count == 3 &&
        first_slot.fanin[0] == first_sf &&
        first_slot.fanin[1] == first_pv &&
        first_slot.fanin[2] == alloc;

    PaOrchestrationState second_orchestration =
        first_orchestration;
    PreparePaBlockGroup(
        second_orchestration, kPaBlocksPerRequest
    );
    second_orchestration.sf_max = OutputRef(second_sf, 1);
    second_orchestration.sf_sum = OutputRef(second_sf, 2);
    second_orchestration.pv_output = OutputRef(second_pv, 0);
    TaskArgs second_args;
    LocalStats second_build_stats{};
    ok &= BuildCallbackSubmitArgs<TaskKind::Up>(
        second_orchestration, second_args, 0, second_build_stats
    );

    worker.local_index = second_up;
    SubmitContext second_context{};
    BeginCallbackSubmit(worker, second_context);
    second_context.won = true;
    second_context.kernel_id = FunctionId(TaskKind::Up);
    ok &= second_context.task_id == second_up;
    ok &= PrepareSharedTaskOutputs(
        second_context.shared_result, second_up, TaskKind::Up
    );
    LocalStats second_stats{};
    bool second_pmu_context = false;
    const CallbackSubmitTicket second_ticket{
        2, second_up,
        static_cast<int16_t>(FunctionId(TaskKind::Up)), 1,
        EncodeSharedPaTaskMeta(TaskKind::Up, 1, false)
    };
    ok &= FinishCallbackSubmitBody<LoserFinishTestOps, false>(
        state, worker, 9, second_args, second_context,
        second_stats, second_pmu_context, second_ticket
    );
    const LocalSlot &second_slot = worker.slots[1];
    ok &= state->fatal.value == 0;
    ok &= state->tasks[second_up].deps_prepared == -1;
    ok &= second_stats.result.shared_symbol_input_loads == 3;
    ok &= second_stats.result.shared_symbol_inout_commits == 3;
    ok &= second_stats.result.fanin_edges == 3;
    ok &= second_stats.result.dependency_signature ==
        0xf772149f1ca20d6bULL;
    ok &= second_slot.occupied && second_slot.built &&
        second_slot.task_id == static_cast<uint32_t>(second_up) &&
        second_slot.kind ==
            static_cast<uint32_t>(FunctionId(TaskKind::Up)) &&
        second_slot.fanin_count == 3 &&
        second_slot.fanin[0] == second_sf &&
        second_slot.fanin[1] == second_pv &&
        second_slot.fanin[2] == first_up;
    ok &=
        state->shared_map.shared_outputs[alloc]
            .last_writer[0].value == second_up &&
        state->shared_map.shared_outputs[alloc]
            .last_writer[1].value == second_up &&
        state->shared_map.shared_outputs[alloc]
            .last_writer[2].value == second_up;

    (void)munmap(state, sizeof(SchedulerState));
    return ok;
}

}  // namespace

int main() {
    const bool batch_plan_ok = RunSharedBatchPlanTest();
    const bool direct_slot_ok = RunReadySharedDescriptorDirectToSlotTest();
    const bool loser_ok = RunProtectedArgsLoserTest();
    const bool future_task_ok = RunFutureTaskWithoutSequencerTest();
    const bool fatal_stop_ok = RunPreRaisedFatalStopsWinnerTest();
    const bool alloc_failure_ok = RunAllocPublicationFailureTest();
    const bool qk_failure_ok = RunQkPublicationFailureTest();
    const bool up_failure_ok = RunUpWriterCommitFailureTest();
    const bool fatal_drain_ok = RunFatalBlockedSuccessorDrainTest();
    const bool two_group_finish_ok = RunTwoGroupExplicitFinishTest();
    if (!batch_plan_ok || !direct_slot_ok || !loser_ok ||
        !future_task_ok || !fatal_stop_ok ||
        !alloc_failure_ok || !qk_failure_ok || !up_failure_ok ||
        !fatal_drain_ok ||
        !two_group_finish_ok) {
        std::fprintf(
            stderr,
            "[FAIL] shared split-finish loser or winner seal regression\n"
        );
        return 1;
    }
    std::printf(
        "[PASS] shared split-finish loser and winner seal failure paths\n"
    );
    return 0;
}
