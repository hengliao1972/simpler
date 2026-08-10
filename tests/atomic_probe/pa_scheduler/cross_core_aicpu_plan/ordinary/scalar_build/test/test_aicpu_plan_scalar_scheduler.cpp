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
#include "aicpu_plan_pa_adapter.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

namespace {

using namespace pa_scheduler;
using namespace pa_scheduler::aicpu_plan;
using namespace pa_scheduler::aicpu_plan_adapter;

constexpr uint32_t kScalarWorkers = 96U;
constexpr uint32_t kAicWorkers = 32U;
constexpr uint32_t kAivWorkers = 64U;
constexpr uint32_t kPlanCapacity = kRuntimePlanConsumerCapacity;

static_assert(kScalarWorkers == kWorkers, "CPU gate must preserve the 96-Scalar contract");
static_assert(kAicWorkers + kAivWorkers == kScalarWorkers, "AIC/AIV role population changed");
static_assert(kPlanCapacity >= 1280U, "runtime Plan capacity no longer covers PA B256/G1");

[[noreturn]] void Fail(const char *message)
{
    std::fprintf(stderr, "[FAIL] %s\n", message);
    std::exit(1);
}

void Expect(bool condition, const char *message)
{
    if (!condition) Fail(message);
}

// CPU 只模拟协议可见性。A5 的正式实现分别由 exact DCCI 和返回型
// atomic 建立发布/获取关系；这里用 C++ acquire/release 锁定同一 happens-before。
struct CpuPlanOps {
    static int64_t LoadControl(volatile int64_t *address)
    {
        return __atomic_load_n(address, __ATOMIC_ACQUIRE);
    }

    static void PublishControl(volatile int64_t *address, int64_t value)
    {
        __atomic_store_n(address, value, __ATOMIC_RELEASE);
    }

    static int64_t FetchAddControl(volatile int64_t *address, int64_t value)
    {
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }

    static void StorePayloadWord(volatile uint64_t *address, uint64_t value)
    {
        *address = value;
    }

    static void FlushRegion(const void *, size_t)
    {
        std::atomic_thread_fence(std::memory_order_release);
    }

    static void InvalidateRegion(const void *, size_t)
    {
        std::atomic_thread_fence(std::memory_order_acquire);
    }

    static void StoreBarrier()
    {
        std::atomic_thread_fence(std::memory_order_release);
    }
};

struct PlanFixture {
    RuntimePlanControl control{};
    std::unique_ptr<RuntimeTaskPlanCell[]> cells;
    RuntimePlanView view{};

    PlanFixture() : cells(new RuntimeTaskPlanCell[kPlanCapacity]())
    {
        control.planned_frontier.value = 0;
        control.closed_task_count.value = kPlanOpenTaskCount;
        control.build_next.value = 0;
        control.build_workers_done.value = 0;
        control.build_release.value = kBuildReleasePending;
        control.fatal.value = 0;

        RuntimePlanStorageRef storage{};
        storage.cells_base = reinterpret_cast<uint64_t>(cells.get());
        storage.cells_bytes =
            static_cast<uint64_t>(kPlanCapacity) *
            sizeof(RuntimeTaskPlanCell);
        storage.capacity = kPlanCapacity;
        storage.cell_bytes = sizeof(RuntimeTaskPlanCell);
        storage.abi_version = kRuntimePlanAbiVersion;
        Expect(
            MakeRuntimePlanView(&control, storage, view),
            "failed to construct checked runtime Plan view"
        );
    }
};

enum class MinimalExecPhase : uint32_t {
    Empty = 0U,
    Built = 1U,
    Done = 2U,
    MetadataDone = 3U,
};

// 本门槛只证明 Plan 消费、严格插入完成链和 AIC/AIV 路由，不冒充完整
// runtime payload。真实 TensorDesc/参数已经由 DecodePaRuntimeTaskPlan
// 逐字段恢复；最小执行 cell 只保存跨核执行阶段必须识别的稳定身份。
struct alignas(kAtomicIsolationBytes) MinimalExecCell {
    std::atomic<uint32_t> phase{static_cast<uint32_t>(MinimalExecPhase::Empty)};
    uint32_t task_id = UINT32_MAX;
    uint32_t function_id = kInvalidFunctionId;
    uint8_t engine_class = UINT8_MAX;
    uint8_t adapter_flags = 0U;
    uint16_t reserved = 0U;
    uint8_t padding[kAtomicIsolationBytes - 16U]{};
};

static_assert(sizeof(MinimalExecCell) == kAtomicIsolationBytes, "minimal exec cell isolation changed");
static_assert(alignof(MinimalExecCell) == kAtomicIsolationBytes, "minimal exec cell alignment changed");

struct ProducerRecord {
    uint32_t task_id;
    TaskKind kind;
    EngineClass engine;
    uint32_t output_count;
    bool last;
};

EngineClass EngineForKind(TaskKind kind)
{
    if (kind == TaskKind::Alloc) return EngineClass::MetadataOnly;
    if (kind == TaskKind::Qk || kind == TaskKind::Pv) {
        return EngineClass::Aic;
    }
    return EngineClass::Aiv;
}

bool BuildArgsForKind(
    TaskKind kind, PaOrchestrationState &orchestration,
    TaskArgs &args, uint32_t batch, LocalStats &stats
)
{
    switch (kind) {
        case TaskKind::Alloc:
            return BuildCallbackSubmitArgs<TaskKind::Alloc>(
                orchestration, args, batch, stats
            );
        case TaskKind::Qk:
            return BuildCallbackSubmitArgs<TaskKind::Qk>(
                orchestration, args, batch, stats
            );
        case TaskKind::Sf:
            return BuildCallbackSubmitArgs<TaskKind::Sf>(
                orchestration, args, batch, stats
            );
        case TaskKind::Pv:
            return BuildCallbackSubmitArgs<TaskKind::Pv>(
                orchestration, args, batch, stats
            );
        case TaskKind::Up:
            return BuildCallbackSubmitArgs<TaskKind::Up>(
                orchestration, args, batch, stats
            );
        case TaskKind::Count:
            return false;
    }
    return false;
}

bool PublishOneCallbackTask(
    const RuntimePlanView &view, PaOrchestrationState &orchestration,
    TaskArgs &args, LocalStats &stats, uint32_t batch,
    uint32_t group_index, TaskKind kind, bool has_following_group,
    bool last, uint32_t &next_task_id,
    std::vector<ProducerRecord> &records
)
{
    const uint32_t task_id = next_task_id;
    if (task_id >= view.capacity || kind >= TaskKind::Count) return false;

    // 输出符号和参数都由当前 callback 上下文产生。这里没有 Host task
    // descriptor 表，也不从 task_id 的余数反推任务类型。
    SharedTaskOutputs outputs{};
    outputs.Reset(static_cast<int32_t>(task_id));
    if (!PrepareSharedTaskOutputs(
            outputs, static_cast<int32_t>(task_id), kind
        ) ||
        !BuildArgsForKind(
            kind, orchestration, args, batch, stats
        )) {
        return false;
    }

    const uint8_t adapter_flags = EncodeSharedPaTaskMeta(
        kind, group_index, has_following_group, last
    );
    const EngineClass engine = EngineForKind(kind);
    RuntimeTaskPlanSpec spec{};
    if (adapter_flags == 0U ||
        !MakePaRuntimeTaskPlanSpec(
            args, task_id, FunctionId(kind), engine,
            adapter_flags, spec
        )) {
        return false;
    }
    const PaRuntimeTaskPlanSource source{args};
    if (PublishRuntimeTaskPlan<CpuPlanOps>(
            view, spec, source
        ) != PlanPublishResult::Published ||
        !AdvancePlannedFrontier<CpuPlanOps>(view, task_id)) {
        return false;
    }

    records.push_back(ProducerRecord{
        task_id, kind, engine, spec.output_count, last,
    });
    ++next_task_id;
    AcceptTaskOutputs(orchestration, kind, outputs);
    return true;
}

bool ProducePlanFromPaCallbacks(
    const RuntimePlanView &view,
    const std::vector<int32_t> &context_lens,
    std::vector<ProducerRecord> &records
)
{
    if (context_lens.empty() || context_lens.size() > kMaxBatches) {
        return false;
    }
    PaOrchestrationState orchestration{};
    InitPaOrchestration(
        orchestration, static_cast<uint32_t>(context_lens.size()),
        context_lens.data()
    );
    TaskArgs args{};
    LocalStats stats{};
    uint32_t next_task_id = 0U;

    for (uint32_t batch = 0U; batch < context_lens.size(); ++batch) {
        BeginPaBatchForCallback(orchestration, batch);
        if (orchestration.current_sequence >
            kSharedPaMaxContextLength) {
            return false;
        }
        const bool alloc_is_last =
            orchestration.current_blocks == 0U &&
            batch + 1U == context_lens.size();
        if (!PublishOneCallbackTask(
                view, orchestration, args, stats, batch,
                0U, TaskKind::Alloc, false, alloc_is_last,
                next_task_id, records
            )) {
            return false;
        }

        uint32_t group_index = 0U;
        uint64_t block_offset = 0U;
        while (block_offset < orchestration.current_blocks) {
            if (group_index >= kSharedPaMaxBlockGroups) return false;
            PreparePaBlockGroup(orchestration, block_offset);
            if (orchestration.current_nblocks == 0U) return false;
            const bool has_following =
                block_offset + orchestration.current_nblocks <
                orchestration.current_blocks;
            if (!PublishOneCallbackTask(
                    view, orchestration, args, stats, batch,
                    group_index, TaskKind::Qk, false, false,
                    next_task_id, records
                ) ||
                !PublishOneCallbackTask(
                    view, orchestration, args, stats, batch,
                    group_index, TaskKind::Sf, false, false,
                    next_task_id, records
                ) ||
                !PublishOneCallbackTask(
                    view, orchestration, args, stats, batch,
                    group_index, TaskKind::Pv, false, false,
                    next_task_id, records
                )) {
                return false;
            }
            const bool up_is_last =
                !has_following &&
                batch + 1U == context_lens.size();
            if (!PublishOneCallbackTask(
                    view, orchestration, args, stats, batch,
                    group_index, TaskKind::Up, has_following,
                    up_is_last, next_task_id, records
                )) {
                return false;
            }
            block_offset += orchestration.current_nblocks;
            ++group_index;
        }
    }
    return CloseRuntimePlan<CpuPlanOps>(view, next_task_id);
}

struct CaseState {
    std::unique_ptr<std::atomic<uint32_t>[]> build_counts;
    std::unique_ptr<std::atomic<uint32_t>[]> execute_counts;
    std::unique_ptr<std::atomic<int64_t>[]> insert_completion;
    std::unique_ptr<MinimalExecCell[]> exec_cells;
    std::atomic<uint32_t> insertion_sequence{0U};
    std::atomic<uint32_t> metadata_done{0U};
    std::atomic<uint32_t> executable_built{0U};
    std::atomic<uint32_t> build_failure{0U};
    std::atomic<uint32_t> build_ready{0U};
    std::atomic<bool> build_start{false};
    std::atomic<uint32_t> last_arrivals{0U};
    std::atomic<uint32_t> released_workers{0U};

    explicit CaseState(uint32_t task_count)
        : build_counts(new std::atomic<uint32_t>[task_count]),
          execute_counts(new std::atomic<uint32_t>[task_count]),
          insert_completion(new std::atomic<int64_t>[task_count]),
          exec_cells(new MinimalExecCell[task_count])
    {
        for (uint32_t task = 0U; task < task_count; ++task) {
            build_counts[task].store(0U, std::memory_order_relaxed);
            execute_counts[task].store(0U, std::memory_order_relaxed);
            insert_completion[task].store(-1, std::memory_order_relaxed);
        }
    }
};

void MarkFailure(
    PlanFixture &fixture, std::atomic<uint32_t> &failure,
    uint32_t code
)
{
    uint32_t expected = 0U;
    (void)failure.compare_exchange_strong(
        expected, code, std::memory_order_relaxed
    );
    CpuPlanOps::PublishControl(&fixture.control.fatal.value, 1);
}

bool HeaderMatchesDecodedArgs(
    const RuntimeTaskPlanHeader &header,
    const TaskArgs &args
)
{
    if (args.has_error || args.tensor_count != header.tensor_count ||
        args.scalar_count != header.scalar_count ||
        args.explicit_dep_count != header.explicit_dep_count) {
        return false;
    }
    SharedPaTaskMeta meta{};
    return DecodeSharedPaTaskMeta(
        header.adapter_flags, header.task_id, meta
    ) &&
        EngineForKind(meta.kind) ==
            static_cast<EngineClass>(header.engine_class) &&
        FunctionId(meta.kind) ==
            (header.function_id == kInvalidFunctionId
                 ? -1
                 : static_cast<int32_t>(header.function_id));
}

void RunBuildWorkers(
    PlanFixture &fixture, uint32_t task_count, CaseState &state
)
{
    std::vector<std::thread> workers;
    workers.reserve(kScalarWorkers);
    for (uint32_t worker = 0U; worker < kScalarWorkers; ++worker) {
        workers.emplace_back([&, worker] {
            state.build_ready.fetch_add(1U, std::memory_order_release);
            while (!state.build_start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            while (true) {
                const BuildReservation reservation =
                    TakeClosedPlanBuildTicket<CpuPlanOps>(fixture.view);
                if (reservation.status == BuildReservationStatus::Closed) {
                    break;
                }
                if (reservation.status != BuildReservationStatus::Reserved ||
                    reservation.task_id >= task_count) {
                    MarkFailure(fixture, state.build_failure, 1U);
                    return;
                }
                const uint32_t task_id = reservation.task_id;
                RuntimeTaskPlanHeader header{};
                RuntimeTaskPlanLayout layout{};
                if (AcquireRuntimeTaskPlan<CpuPlanOps>(
                        fixture.view, task_id, header, layout
                    ) != PlanAcquireResult::Acquired ||
                    state.build_counts[task_id].fetch_add(
                        1U, std::memory_order_relaxed
                    ) != 0U) {
                    MarkFailure(fixture, state.build_failure, 2U);
                    return;
                }

                TaskArgs decoded_args{};
                PaRuntimeTaskDecodeScratch scratch{};
                if (!DecodePaRuntimeTaskPlan(
                        fixture.view.cells[task_id], header, layout,
                        decoded_args, scratch
                    ) ||
                    !HeaderMatchesDecodedArgs(header, decoded_args)) {
                    MarkFailure(fixture, state.build_failure, 3U);
                    return;
                }

                // ordinary TensorMap 的 payload 写入在这个独立状态机中不
                // 展开；但所有 task（包括无 writer 的 task）必须完成严格
                // N-1 -> N 的 completion 传递，不能越过前驱发布。
                if (task_id != 0U) {
                    while (state.insert_completion[task_id - 1U].load(
                               std::memory_order_acquire
                           ) != static_cast<int64_t>(task_id)) {
                        if (CpuPlanOps::LoadControl(
                                &fixture.control.fatal.value
                            ) != 0) {
                            return;
                        }
                        std::this_thread::yield();
                    }
                }
                if (state.insertion_sequence.fetch_add(
                        1U, std::memory_order_acq_rel
                    ) != task_id) {
                    MarkFailure(fixture, state.build_failure, 4U);
                    return;
                }

                MinimalExecCell &exec = state.exec_cells[task_id];
                exec.task_id = task_id;
                exec.function_id = header.function_id;
                exec.engine_class = header.engine_class;
                exec.adapter_flags = header.adapter_flags;
                const EngineClass engine =
                    static_cast<EngineClass>(header.engine_class);
                if (engine == EngineClass::MetadataOnly) {
                    exec.phase.store(
                        static_cast<uint32_t>(
                            MinimalExecPhase::MetadataDone
                        ),
                        std::memory_order_release
                    );
                    state.metadata_done.fetch_add(
                        1U, std::memory_order_relaxed
                    );
                } else {
                    exec.phase.store(
                        static_cast<uint32_t>(MinimalExecPhase::Built),
                        std::memory_order_release
                    );
                    state.executable_built.fetch_add(
                        1U, std::memory_order_relaxed
                    );
                }
                state.insert_completion[task_id].store(
                    static_cast<int64_t>(task_id) + 1,
                    std::memory_order_release
                );
            }

            const BuildArrivalStatus arrival =
                ArriveClosedPlanBuildWorker<CpuPlanOps>(
                    fixture.view, kScalarWorkers
                );
            if (arrival == BuildArrivalStatus::Last) {
                state.last_arrivals.fetch_add(
                    1U, std::memory_order_relaxed
                );
                if (!PublishClosedPlanBuildRelease<CpuPlanOps>(
                        fixture.view, task_count, kScalarWorkers
                    )) {
                    MarkFailure(fixture, state.build_failure, 5U);
                    return;
                }
            } else if (arrival != BuildArrivalStatus::Arrived) {
                MarkFailure(fixture, state.build_failure, 6U);
                return;
            }
            while (CpuPlanOps::LoadControl(
                       &fixture.control.build_release.value
                   ) != static_cast<int64_t>(task_count)) {
                if (CpuPlanOps::LoadControl(
                        &fixture.control.fatal.value
                    ) != 0) {
                    return;
                }
                std::this_thread::yield();
            }
            state.released_workers.fetch_add(
                1U, std::memory_order_relaxed
            );
            (void)worker;
        });
    }
    while (state.build_ready.load(std::memory_order_acquire) !=
           kScalarWorkers) {
        std::this_thread::yield();
    }
    state.build_start.store(true, std::memory_order_release);
    for (std::thread &worker : workers) worker.join();
}

void RunEngineWorkers(
    EngineClass role, uint32_t worker_count, uint32_t task_count,
    CaseState &state, std::atomic<uint32_t> &done,
    std::atomic<uint32_t> &failure
)
{
    std::atomic<uint32_t> next{0U};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (uint32_t worker = 0U; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker] {
            while (true) {
                const uint32_t task_id = next.fetch_add(
                    1U, std::memory_order_relaxed
                );
                if (task_id >= task_count) break;
                MinimalExecCell &cell = state.exec_cells[task_id];
                const EngineClass engine =
                    static_cast<EngineClass>(cell.engine_class);
                if (engine != role) continue;
                uint32_t expected =
                    static_cast<uint32_t>(MinimalExecPhase::Built);
                if (cell.task_id != task_id ||
                    cell.function_id == kInvalidFunctionId ||
                    !cell.phase.compare_exchange_strong(
                        expected,
                        static_cast<uint32_t>(MinimalExecPhase::Done),
                        std::memory_order_acq_rel,
                        std::memory_order_acquire
                    ) ||
                    state.execute_counts[task_id].fetch_add(
                        1U, std::memory_order_relaxed
                    ) != 0U) {
                    uint32_t zero = 0U;
                    (void)failure.compare_exchange_strong(
                        zero, task_id + 1U,
                        std::memory_order_relaxed
                    );
                    return;
                }
                done.fetch_add(1U, std::memory_order_relaxed);
            }
            (void)worker;
        });
    }
    for (std::thread &worker : workers) worker.join();
}

void RunCase(
    const char *name, const std::vector<int32_t> &context_lens,
    uint32_t expected_tasks
)
{
    PlanFixture fixture;
    std::vector<ProducerRecord> records;
    Expect(
        ProducePlanFromPaCallbacks(
            fixture.view, context_lens, records
        ),
        "dynamic PA callback producer failed"
    );
    const uint32_t task_count = static_cast<uint32_t>(records.size());
    Expect(task_count == expected_tasks, "dynamic PA task count mismatch");
    Expect(
        CpuPlanOps::LoadControl(
            &fixture.control.planned_frontier.value
        ) == static_cast<int64_t>(task_count),
        "producer frontier did not close on dynamic task count"
    );

    uint32_t expected_metadata = 0U;
    uint32_t expected_aic = 0U;
    uint32_t expected_aiv = 0U;
    uint32_t last_count = 0U;
    for (uint32_t task = 0U; task < task_count; ++task) {
        Expect(records[task].task_id == task, "producer task ids are not contiguous");
        if (records[task].engine == EngineClass::MetadataOnly) {
            ++expected_metadata;
        } else if (records[task].engine == EngineClass::Aic) {
            ++expected_aic;
        } else if (records[task].engine == EngineClass::Aiv) {
            ++expected_aiv;
        } else {
            Fail("producer emitted an unknown engine class");
        }
        if (records[task].last) ++last_count;
    }
    Expect(last_count == 1U && records.back().last, "Plan terminal callback marker mismatch");

    CaseState state(task_count);
    RunBuildWorkers(fixture, task_count, state);
    if (state.build_failure.load(std::memory_order_relaxed) != 0U) {
        std::fprintf(
            stderr, "[FAIL] %s Build failure code=%u\n", name,
            state.build_failure.load(std::memory_order_relaxed)
        );
        std::exit(1);
    }

    Expect(
        CpuPlanOps::LoadControl(&fixture.control.build_next.value) ==
            static_cast<int64_t>(task_count + kScalarWorkers),
        "central Build ticket count is not N+96"
    );
    Expect(
        CpuPlanOps::LoadControl(
            &fixture.control.build_workers_done.value
        ) == static_cast<int64_t>(kScalarWorkers),
        "Build worker arrival count is not 96"
    );
    Expect(
        state.last_arrivals.load(std::memory_order_relaxed) == 1U,
        "Build release has more or fewer than one publisher"
    );
    Expect(
        state.released_workers.load(std::memory_order_relaxed) ==
            kScalarWorkers,
        "not every Build worker observed the unique release"
    );
    Expect(
        state.insertion_sequence.load(std::memory_order_acquire) ==
            task_count,
        "strict TensorMap completion chain did not reach N"
    );
    Expect(
        state.metadata_done.load(std::memory_order_relaxed) ==
            expected_metadata,
        "metadata-only completion count mismatch"
    );
    Expect(
        state.executable_built.load(std::memory_order_relaxed) ==
            expected_aic + expected_aiv,
        "executable Build publication count mismatch"
    );

    std::atomic<uint32_t> aic_done{0U};
    std::atomic<uint32_t> aiv_done{0U};
    std::atomic<uint32_t> execute_failure{0U};
    std::thread aic_group([&] {
        RunEngineWorkers(
            EngineClass::Aic, kAicWorkers, task_count,
            state, aic_done, execute_failure
        );
    });
    std::thread aiv_group([&] {
        RunEngineWorkers(
            EngineClass::Aiv, kAivWorkers, task_count,
            state, aiv_done, execute_failure
        );
    });
    aic_group.join();
    aiv_group.join();
    Expect(
        execute_failure.load(std::memory_order_relaxed) == 0U,
        "AIC/AIV execution role consumed an invalid cell"
    );
    Expect(
        aic_done.load(std::memory_order_relaxed) == expected_aic,
        "AIC DONE count mismatch"
    );
    Expect(
        aiv_done.load(std::memory_order_relaxed) == expected_aiv,
        "AIV DONE count mismatch"
    );

    uint32_t terminal_done = 0U;
    for (uint32_t task = 0U; task < task_count; ++task) {
        Expect(
            state.build_counts[task].load(std::memory_order_relaxed) == 1U,
            "task was lost or built more than once"
        );
        Expect(
            state.insert_completion[task].load(
                std::memory_order_acquire
            ) == static_cast<int64_t>(task) + 1,
            "per-task insertion completion value mismatch"
        );
        const MinimalExecPhase phase = static_cast<MinimalExecPhase>(
            state.exec_cells[task].phase.load(std::memory_order_acquire)
        );
        if (records[task].engine == EngineClass::MetadataOnly) {
            Expect(
                phase == MinimalExecPhase::MetadataDone &&
                    state.execute_counts[task].load(
                        std::memory_order_relaxed
                    ) == 0U,
                "metadata-only task entered an execution engine"
            );
        } else {
            Expect(
                phase == MinimalExecPhase::Done &&
                    state.execute_counts[task].load(
                        std::memory_order_relaxed
                    ) == 1U,
                "executable task did not reach DONE exactly once"
            );
        }
        ++terminal_done;
    }
    Expect(terminal_done == task_count, "terminal DONE accounting mismatch");

    std::printf(
        "[PASS] %-14s tasks=%u tickets=%u metadata=%u "
        "executable=%u AIC=%u AIV=%u DONE=%u\n",
        name, task_count, task_count + kScalarWorkers,
        expected_metadata, expected_aic + expected_aiv,
        expected_aic, expected_aiv, terminal_done
    );
}

}  // namespace

int main()
{
    // B1 使用一个跨组边界的实际 context；G0/G1/G2/G4 分别锁定
    // 动态 block-group 数。mixed 同时覆盖空 batch、短尾和跨组 batch。
    RunCase("PA-B1", {8193}, 9U);
    RunCase("PA-G0", {0}, 1U);
    RunCase("PA-G1", {8192}, 5U);
    RunCase("PA-G2", {16384}, 9U);
    RunCase("PA-G4", {32768}, 17U);
    RunCase("PA-mixed", {0, 1, 8192, 8193}, 20U);

    std::vector<int32_t> b256_context_lens(256U, 8192);
    RunCase("PA-B256-G1", b256_context_lens, 1280U);
    std::puts(
        "[BOUNDARY] CPU gate uses a minimal exec cell and models only the "
        "strict TensorMap insertion-completion chain; full TensorMap payload "
        "writes and real kernels remain A5 end-to-end work."
    );
    return 0;
}
