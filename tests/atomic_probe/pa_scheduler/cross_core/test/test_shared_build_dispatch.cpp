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

#include "host_support.h"
#include "pa_scheduler_core.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include <sys/mman.h>

namespace {

using namespace pa_scheduler;
using namespace pa_scheduler::host;

constexpr uint32_t kDispatchWorkers = kWorkers;
constexpr uint32_t kDispatchBatches = kDefaultBatches;
constexpr uint32_t kExpectedTasks = kDispatchBatches * kTasksPerBatch;
constexpr uint32_t kDelayedBuildEvidence = 16;
constexpr uint32_t kProtocolRuns = 10;
constexpr uint64_t kCurrentB256ClaimCas = 133120;

static_assert(kExpectedTasks == 1280, "B256 G1 task count changed");
static_assert(kDispatchWorkers == 96, "A5 Scalar worker count changed");

// 这是未来 device flat plan 的最小候选形状，不携带 TensorDesc 或 winner
// 私有地址。task id 显式存储用于拒绝错位表项；kind/group/末次等信息复用
// 已有 1B meta 编码，batch 与 batch_start 则供随机访问构参直接使用。
struct DispatchTaskIdentity {
    uint32_t task_id;
    uint32_t batch;
    uint32_t batch_start;
    uint8_t encoded_meta;
    uint8_t reserved[3];
};
static_assert(sizeof(DispatchTaskIdentity) == 16, "dispatch task identity must remain 16 bytes");

enum class TicketStatus : uint8_t {
    Acquired,
    Exhausted,
    Invalid,
};

struct TicketResult {
    TicketStatus status;
    uint32_t task_id;
    int64_t observed;
};

struct DispatchControl {
    // 三个控制字各自沿用仓库的 64B AtomicLine，避免在候选协议中先引入
    // 伪共享。CPU 只验证线性化语义，不模拟 A5 DCache/DCCI。
    AtomicLine next_task;
    AtomicLine retired_workers;
    AtomicLine production_closed;
};

struct TaskEvidence {
    std::atomic<int32_t> owner;
    std::atomic<uint32_t> prepare_count;
    std::atomic<uint32_t> build_count;
    std::atomic<int32_t> execute_owner;
    std::atomic<uint32_t> execute_count;
};

struct RunEvidence {
    std::atomic<uint32_t> failures{0};
    std::atomic<uint32_t> prepared_tasks{0};
    std::atomic<uint32_t> published_metadata_writers{0};
    std::atomic<uint32_t> built_tasks{0};
    std::atomic<uint32_t> executed_tasks{0};
    std::atomic<uint32_t> metadata_writer_order{0};
    std::atomic<uint32_t> ticket_fetch_adds{0};
    std::atomic<uint32_t> exec_ticket_fetch_adds{0};
    std::atomic<uint32_t> retire_fetch_adds{0};
    std::atomic<uint32_t> predecessor_loads{0};
    std::atomic<uint32_t> later_built_before_task0{0};
    std::atomic<bool> abort{false};
};

uint32_t MetadataWriterCountBefore(
    const SharedBuildDispatchState &dispatch, uint32_t task_id
) {
    uint32_t count = 0;
    for (uint32_t task = 0; task < task_id; ++task) {
        count += static_cast<uint32_t>(
            (dispatch.metadata_writer_bits[task / 64U] >>
             (task % 64U)) & uint64_t{1}
        );
    }
    return count;
}

void RecordFailure(RunEvidence &evidence);
bool DecodeDispatchIdentity(
    const DispatchTaskIdentity &identity, uint32_t expected_task, uint32_t total_tasks, SharedPaTaskMeta &meta
);

CoreRole WorkerRole(uint32_t worker) { return worker < kAicWorkers ? CoreRole::Aic : CoreRole::Aiv; }

struct TestOps {
    static int64_t Load(volatile int64_t *address) {
        return __atomic_fetch_add(address, static_cast<int64_t>(0), __ATOMIC_ACQUIRE);
    }

    static int64_t FetchAdd(volatile int64_t *address, int64_t value) {
        return __atomic_fetch_add(address, value, __ATOMIC_ACQ_REL);
    }

    static int64_t CompareExchange(volatile int64_t *address, int64_t expected, int64_t desired) {
        int64_t observed = expected;
        (void)__atomic_compare_exchange_n(address, &observed, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
        return observed;
    }

    static int64_t Exchange(volatile int64_t *address, int64_t value) {
        return __atomic_exchange_n(address, value, __ATOMIC_ACQ_REL);
    }
};

bool ConsumeRoleExecTickets(
    uint32_t worker, uint32_t total_tasks,
    const std::vector<DispatchTaskIdentity> &plan,
    SchedulerState &state, TaskEvidence *task_evidence,
    RunEvidence &evidence
) {
    const CoreRole role = WorkerRole(worker);
    const cross_core::ExecEngineClass worker_engine =
        CrossCoreEngineForRole(role);
    AtomicLine *cursor = role == CoreRole::Aic
        ? &state.exec_dispatch.aic_next
        : &state.exec_dispatch.aiv_next;
    const uint32_t task_count = role == CoreRole::Aic
        ? state.exec_dispatch.aic_task_count
        : state.exec_dispatch.aiv_task_count;
    const uint32_t *task_ids = role == CoreRole::Aic
        ? &state.exec_dispatch.aic_task_ids[0]
        : &state.exec_dispatch.aiv_task_ids[0];
    if (task_count > total_tasks) {
        RecordFailure(evidence);
        return false;
    }

    while (!evidence.abort.load(std::memory_order_acquire)) {
        evidence.exec_ticket_fetch_adds.fetch_add(
            1, std::memory_order_relaxed
        );
        const int64_t ordinal = TestOps::FetchAdd(
            &cursor->value, 1
        );
        if (ordinal < 0) {
            RecordFailure(evidence);
            return false;
        }
        if (ordinal >= static_cast<int64_t>(task_count)) {
            return true;
        }
        const uint32_t task_id = task_ids[ordinal];
        SharedPaTaskMeta meta{};
        cross_core::PaExecRoute route{};
        if (task_id >= total_tasks ||
            !DecodeDispatchIdentity(
                plan[task_id], task_id, total_tasks, meta
            ) ||
            meta.kind == TaskKind::Alloc ||
            !cross_core::ResolvePaExecRoute(
                meta.kind, FunctionId(meta.kind), route
            ) ||
            route.engine_class != worker_engine ||
            task_evidence[task_id].build_count.load(
                std::memory_order_acquire
            ) != 1) {
            RecordFailure(evidence);
            return false;
        }

        int32_t expected = -1;
        if (!task_evidence[task_id].execute_owner
                 .compare_exchange_strong(
                     expected, static_cast<int32_t>(worker),
                     std::memory_order_acq_rel,
                     std::memory_order_acquire
                 ) ||
            task_evidence[task_id].execute_count.fetch_add(
                1, std::memory_order_acq_rel
            ) != 0) {
            RecordFailure(evidence);
            return false;
        }
        evidence.executed_tasks.fetch_add(
            1, std::memory_order_release
        );
    }
    return false;
}

class MappedSchedulerState {
public:
    MappedSchedulerState() {
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
        flags |= MAP_NORESERVE;
#endif
        memory_ = mmap(nullptr, sizeof(SchedulerState), PROT_READ | PROT_WRITE, flags, -1, 0);
        if (memory_ != MAP_FAILED) {
            state_ = new (memory_) SchedulerState;
        }
    }

    ~MappedSchedulerState() {
        if (memory_ != MAP_FAILED) {
            (void)munmap(memory_, sizeof(SchedulerState));
        }
    }

    MappedSchedulerState(const MappedSchedulerState &) = delete;
    MappedSchedulerState &operator=(const MappedSchedulerState &) = delete;

    SchedulerState *Get() const { return state_; }

private:
    void *memory_ = MAP_FAILED;
    SchedulerState *state_ = nullptr;
};

void RecordFailure(RunEvidence &evidence) {
    evidence.failures.fetch_add(1, std::memory_order_relaxed);
    evidence.abort.store(true, std::memory_order_release);
}

bool WaitForAtLeast(const std::atomic<uint32_t> &value, uint32_t expected, RunEvidence &evidence) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (value.load(std::memory_order_acquire) < expected) {
        if (evidence.abort.load(std::memory_order_acquire)) {
            return false;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            RecordFailure(evidence);
            return false;
        }
        std::this_thread::yield();
    }
    return true;
}

TicketResult TakeTicket(DispatchControl &control, uint32_t total_tasks, RunEvidence &evidence) {
    evidence.ticket_fetch_adds.fetch_add(1, std::memory_order_relaxed);
    const int64_t observed = TestOps::FetchAdd(&control.next_task.value, 1);
    if (total_tasks == 0 || total_tasks > kMaxTasks || observed < 0) {
        return TicketResult{TicketStatus::Invalid, UINT32_MAX, observed};
    }
    if (observed >= static_cast<int64_t>(total_tasks)) {
        return TicketResult{TicketStatus::Exhausted, UINT32_MAX, observed};
    }
    return TicketResult{
        TicketStatus::Acquired,
        static_cast<uint32_t>(observed),
        observed,
    };
}

bool BuildArgsForKind(TaskKind kind, PaOrchestrationState &orch, TaskArgs &args, uint32_t batch, LocalStats &stats) {
    switch (kind) {
    case TaskKind::Alloc:
        return BuildCallbackSubmitArgs<TaskKind::Alloc>(orch, args, batch, stats);
    case TaskKind::Qk:
        return BuildCallbackSubmitArgs<TaskKind::Qk>(orch, args, batch, stats);
    case TaskKind::Sf:
        return BuildCallbackSubmitArgs<TaskKind::Sf>(orch, args, batch, stats);
    case TaskKind::Pv:
        return BuildCallbackSubmitArgs<TaskKind::Pv>(orch, args, batch, stats);
    case TaskKind::Up:
        return BuildCallbackSubmitArgs<TaskKind::Up>(orch, args, batch, stats);
    case TaskKind::Count:
        return false;
    }
    return false;
}

bool SharedRefsPrecedeTask(const TaskArgs &args, uint32_t task_id) {
    for (int32_t tensor = 0; tensor < args.tensor_count; ++tensor) {
        const TaskTensorRef &reference = args.tensors[static_cast<uint32_t>(tensor)];
        if (reference.kind != TensorRefKind::SharedOutputRef) {
            continue;
        }
        const FdwicOutputRef output_ref = SharedOutputReference(reference);
        if (!IsPlainSharedOutputRef(output_ref) || output_ref.producer_task_id < 0 ||
            static_cast<uint32_t>(output_ref.producer_task_id) >= task_id) {
            return false;
        }
    }
    return true;
}

bool DecodeDispatchIdentity(
    const DispatchTaskIdentity &identity, uint32_t expected_task, uint32_t total_tasks, SharedPaTaskMeta &meta
) {
    if (identity.task_id != expected_task || identity.task_id >= total_tasks ||
        !DecodeSharedPaTaskMeta(identity.encoded_meta, identity.task_id, meta) ||
        meta.batch_start != identity.batch_start || meta.is_last_submit != (identity.task_id + 1U == total_tasks)) {
        return false;
    }
    return true;
}

bool PrepareTicketTask(
    const DispatchTaskIdentity &identity, uint32_t expected_task, uint32_t total_tasks, PaOrchestrationState &orch,
    TaskArgs &args, LocalStats &stats
) {
    SharedPaTaskMeta meta{};
    if (!DecodeDispatchIdentity(identity, expected_task, total_tasks, meta) ||
        !BindSharedPaTaskForRandomAccess(
            orch, identity.batch, identity.batch_start, identity.task_id, meta.kind, meta.group_index
        ) ||
        !BuildArgsForKind(meta.kind, orch, args, identity.batch, stats)) {
        return false;
    }
    return SharedRefsPrecedeTask(args, identity.task_id);
}

uint64_t PlanFingerprint(const std::vector<DispatchTaskIdentity> &plan) {
    uint64_t hash = 1469598103934665603ULL;
    for (const DispatchTaskIdentity &task : plan) {
        const auto *bytes = reinterpret_cast<const uint8_t *>(&task);
        for (size_t index = 0; index < sizeof(task); ++index) {
            hash ^= bytes[index];
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

bool MakeDispatchPlan(
    SchedulerState &state, SharedHostTaskPlan &host_plan, std::vector<DispatchTaskIdentity> &dispatch_plan
) {
    state.config.batches = kDispatchBatches;
    for (uint32_t batch = 0; batch < kDispatchBatches; ++batch) {
        state.context_lens[batch] = 8192;
    }
    std::string error;
    if (!BuildSharedHostTaskPlan(state, &host_plan, &error) ||
        host_plan.total_tasks != kExpectedTasks ||
        !PopulateSharedBuildDispatchPlan(&state, host_plan, &error)) {
        std::fprintf(stderr, "[BUILD_DISPATCH] host plan failed: %s\n", error.c_str());
        return false;
    }

    dispatch_plan.clear();
    dispatch_plan.reserve(host_plan.total_tasks);
    for (const SharedHostPlannedTask &task : host_plan.tasks) {
        const uint8_t encoded = EncodeSharedPaTaskMeta(
            task.kind, task.group_index, task.has_following_group, task.task_id + 1U == host_plan.total_tasks
        );
        if (encoded == 0) {
            return false;
        }
        dispatch_plan.push_back(DispatchTaskIdentity{
            task.task_id,
            task.batch,
            task.batch_start,
            encoded,
            {0, 0, 0},
        });
    }
    return dispatch_plan.size() == host_plan.total_tasks;
}

bool RunDispatchOnce(SchedulerState &state, const std::vector<DispatchTaskIdentity> &plan, uint32_t run) {
    DispatchControl control{};
    control.next_task.value = 0;
    control.retired_workers.value = 0;
    control.production_closed.value = 0;
    state.exec_dispatch.aic_next.value = 0;
    state.exec_dispatch.aiv_next.value = 0;
    RunEvidence evidence;
    auto task_evidence = std::make_unique<TaskEvidence[]>(plan.size());
    auto insert_completion = std::make_unique<AtomicLine[]>(plan.size());
    auto tasks_by_worker = std::make_unique<std::atomic<uint32_t>[]>(kDispatchWorkers);
    for (uint32_t task = 0; task < plan.size(); ++task) {
        task_evidence[task].owner.store(-1, std::memory_order_relaxed);
        task_evidence[task].prepare_count.store(0, std::memory_order_relaxed);
        task_evidence[task].build_count.store(0, std::memory_order_relaxed);
        task_evidence[task].execute_owner.store(-1, std::memory_order_relaxed);
        task_evidence[task].execute_count.store(0, std::memory_order_relaxed);
        insert_completion[task].value =
            SharedInsertCompletionInitialValue(task);
    }
    for (uint32_t worker = 0; worker < kDispatchWorkers; ++worker) {
        tasks_by_worker[worker].store(0, std::memory_order_relaxed);
    }

    std::atomic<uint32_t> workers_ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    workers.reserve(kDispatchWorkers);
    for (uint32_t worker = 0; worker < kDispatchWorkers; ++worker) {
        workers.emplace_back([&, worker]() {
            PaOrchestrationState orch{};
            TaskArgs args{};
            LocalStats stats{};
            InitPaOrchestration(orch, kDispatchBatches, &state.context_lens[0]);
            workers_ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            while (!evidence.abort.load(std::memory_order_acquire)) {
                const TicketResult ticket = TakeTicket(control, static_cast<uint32_t>(plan.size()), evidence);
                if (ticket.status == TicketStatus::Exhausted) {
                    break;
                }
                if (ticket.status != TicketStatus::Acquired) {
                    RecordFailure(evidence);
                    break;
                }
                const uint32_t task_id = ticket.task_id;
                int32_t expected_owner = -1;
                if (!task_evidence[task_id].owner.compare_exchange_strong(
                        expected_owner, static_cast<int32_t>(worker), std::memory_order_acq_rel,
                        std::memory_order_acquire
                    ) ||
                    !PrepareTicketTask(plan[task_id], task_id, static_cast<uint32_t>(plan.size()), orch, args, stats) ||
                    task_evidence[task_id].prepare_count.fetch_add(1, std::memory_order_acq_rel) != 0) {
                    RecordFailure(evidence);
                    break;
                }
                evidence.prepared_tasks.fetch_add(1, std::memory_order_release);

                // 首个 owner 暂停到全局已经完成 96 个 task 的构参。这里验证
                // 准备工作可以越过 task 0 乱序推进；中央 ticket 不承诺 host
                // OS 会公平地让 96 个线程各抢到一个 Build task。
                if (task_id == 0 && !WaitForAtLeast(evidence.prepared_tasks, kDispatchWorkers, evidence)) {
                    break;
                }

                bool publishes_metadata = false;
                int32_t previous_metadata_writer = -1;
                if (!DecodeSharedMetadataWriterPlan(
                        state.build_dispatch, task_id,
                        publishes_metadata,
                        previous_metadata_writer
                    )) {
                    RecordFailure(evidence);
                    break;
                }
                if (previous_metadata_writer >= 0) {
                    while (!evidence.abort.load(std::memory_order_acquire)) {
                        evidence.predecessor_loads.fetch_add(1, std::memory_order_relaxed);
                        const int64_t predecessor = TestOps::Load(
                            &insert_completion[
                                 static_cast<uint32_t>(
                                     previous_metadata_writer
                                 )
                             ].value
                        );
                        if (predecessor == previous_metadata_writer) {
                            break;
                        }
                        if (predecessor !=
                            SharedInsertCompletionInitialValue(
                                static_cast<uint32_t>(
                                    previous_metadata_writer
                                )
                            )) {
                            RecordFailure(evidence);
                            break;
                        }
                        std::this_thread::yield();
                    }
                    if (evidence.abort.load(std::memory_order_acquire)) {
                        break;
                    }
                }

                if (publishes_metadata) {
                    const uint32_t order =
                        evidence.metadata_writer_order.fetch_add(
                            1, std::memory_order_acq_rel
                        );
                    if (order != MetadataWriterCountBefore(
                                     state.build_dispatch, task_id
                                 ) ||
                        TestOps::CompareExchange(
                            &insert_completion[task_id].value,
                            SharedInsertCompletionInitialValue(
                                task_id
                            ),
                            static_cast<int64_t>(task_id)
                        ) != SharedInsertCompletionInitialValue(
                            task_id
                        )) {
                        RecordFailure(evidence);
                        break;
                    }
                    evidence.published_metadata_writers.fetch_add(
                        1, std::memory_order_release
                    );
                }

                // task 0 已完成本 task 的 metadata 计划处理后继续延迟
                // Build。后续 task 必须能继续发布 writer 并完成 Build，
                // 证明严格 writer 链没有错误扩张到按 task-id 保序 Build。
                if (task_id == 0 && !WaitForAtLeast(evidence.built_tasks, kDelayedBuildEvidence, evidence)) {
                    break;
                }
                if (task_id != 0 && task_evidence[0].build_count.load(std::memory_order_acquire) == 0) {
                    evidence.later_built_before_task0.fetch_add(1, std::memory_order_relaxed);
                }
                if (task_evidence[task_id].build_count.fetch_add(1, std::memory_order_acq_rel) != 0) {
                    RecordFailure(evidence);
                    break;
                }
                evidence.built_tasks.fetch_add(1, std::memory_order_release);
                tasks_by_worker[worker].fetch_add(1, std::memory_order_relaxed);
            }

            evidence.retire_fetch_adds.fetch_add(1, std::memory_order_relaxed);
            const int64_t prior = TestOps::FetchAdd(&control.retired_workers.value, 1);
            if (prior < 0 || prior >= static_cast<int64_t>(kDispatchWorkers)) {
                RecordFailure(evidence);
            }
            if (prior + 1 == static_cast<int64_t>(kDispatchWorkers)) {
                if (evidence.prepared_tasks.load(std::memory_order_acquire) != plan.size() ||
                    evidence.published_metadata_writers.load(
                        std::memory_order_acquire
                    ) != MetadataWriterCountBefore(
                        state.build_dispatch,
                        static_cast<uint32_t>(plan.size())
                    ) ||
                    evidence.built_tasks.load(std::memory_order_acquire) != plan.size()) {
                    RecordFailure(evidence);
                }
                (void)TestOps::Exchange(&control.production_closed.value, 1);
            } else {
                while (TestOps::Load(&control.production_closed.value) == 0 &&
                       !evidence.abort.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
            }
            if (!evidence.abort.load(std::memory_order_acquire)) {
                (void)ConsumeRoleExecTickets(
                    worker, static_cast<uint32_t>(plan.size()),
                    plan, state, task_evidence.get(), evidence
                );
            }
        });
    }

    while (workers_ready.load(std::memory_order_acquire) != kDispatchWorkers) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (std::thread &worker : workers) {
        worker.join();
    }

    bool ok = evidence.failures.load(std::memory_order_acquire) == 0 &&
              evidence.prepared_tasks.load(std::memory_order_acquire) == plan.size() &&
              evidence.published_metadata_writers.load(
                  std::memory_order_acquire
              ) == MetadataWriterCountBefore(
                  state.build_dispatch,
                  static_cast<uint32_t>(plan.size())
              ) &&
              evidence.built_tasks.load(std::memory_order_acquire) == plan.size() &&
              evidence.executed_tasks.load(std::memory_order_acquire) == plan.size() - kDispatchBatches &&
              evidence.metadata_writer_order.load(
                  std::memory_order_acquire
              ) == MetadataWriterCountBefore(
                  state.build_dispatch,
                  static_cast<uint32_t>(plan.size())
              ) &&
              evidence.later_built_before_task0.load(std::memory_order_acquire) >= kDelayedBuildEvidence &&
              TestOps::Load(&control.retired_workers.value) == static_cast<int64_t>(kDispatchWorkers) &&
              TestOps::Load(&control.production_closed.value) == 1;
    uint32_t active_workers = 0;
    for (uint32_t worker = 0; worker < kDispatchWorkers; ++worker) {
        active_workers += tasks_by_worker[worker].load(std::memory_order_acquire) != 0 ? 1U : 0U;
    }
    // active_workers 仅作为调度分布诊断量输出。协议要求 96 个 worker 都
    // 启动、退休并参与有限 ticket 收敛，但不要求一个无公平性保证的中央
    // FetchAdd 在 host OS 上恰好向每个线程至少分发一个 Build task。
    for (uint32_t task = 0; task < plan.size(); ++task) {
        ok &= task_evidence[task].owner.load(std::memory_order_acquire) >= 0;
        ok &= task_evidence[task].prepare_count.load(std::memory_order_acquire) == 1;
        ok &= task_evidence[task].build_count.load(std::memory_order_acquire) == 1;
        const bool publishes_metadata =
            ((state.build_dispatch.metadata_writer_bits[
                  task / 64U
              ] >> (task % 64U)) & uint64_t{1}) != 0;
        ok &= TestOps::Load(&insert_completion[task].value) ==
            (publishes_metadata
                 ? static_cast<int64_t>(task)
                 : SharedInsertCompletionInitialValue(task));
        SharedPaTaskMeta meta{};
        ok &= DecodeDispatchIdentity(plan[task], task, static_cast<uint32_t>(plan.size()), meta);
        const int32_t build_owner = task_evidence[task].owner.load(std::memory_order_acquire);
        const int32_t execute_owner = task_evidence[task].execute_owner.load(std::memory_order_acquire);
        if (meta.kind == TaskKind::Alloc) {
            ok &= execute_owner == -1;
            ok &= task_evidence[task].execute_count.load(std::memory_order_acquire) == 0;
        } else {
            cross_core::PaExecRoute route{};
            ok &= cross_core::ResolvePaExecRoute(
                meta.kind, FunctionId(meta.kind), route
            );
            ok &= execute_owner >= 0 &&
                cross_core::A5SingleLaneExecuteOwnerEligible(
                    task, static_cast<uint32_t>(build_owner),
                    route.engine_class,
                    static_cast<uint32_t>(execute_owner)
                );
            ok &= task_evidence[task].execute_count.load(std::memory_order_acquire) == 1;
        }
    }
    const uint32_t expected_ticket_calls = static_cast<uint32_t>(plan.size()) + kDispatchWorkers;
    ok &= evidence.ticket_fetch_adds.load(std::memory_order_acquire) == expected_ticket_calls;
    ok &= TestOps::Load(&control.next_task.value) == static_cast<int64_t>(expected_ticket_calls);
    ok &= evidence.retire_fetch_adds.load(std::memory_order_acquire) == kDispatchWorkers;
    const uint32_t expected_exec_ticket_calls =
        state.exec_dispatch.aic_task_count + kAicWorkers +
        state.exec_dispatch.aiv_task_count + kAivWorkers;
    ok &= evidence.exec_ticket_fetch_adds.load(
              std::memory_order_acquire
          ) == expected_exec_ticket_calls;
    ok &= TestOps::Load(&state.exec_dispatch.aic_next.value) ==
        static_cast<int64_t>(
            state.exec_dispatch.aic_task_count + kAicWorkers
        );
    ok &= TestOps::Load(&state.exec_dispatch.aiv_next.value) ==
        static_cast<int64_t>(
            state.exec_dispatch.aiv_task_count + kAivWorkers
        );

    std::printf(
        "[BUILD_DISPATCH] run=%u status=%s active_workers=%u tickets=%u "
        "exec_tickets=%u metadata_writer_cas=%u executes=%u "
        "later_before_task0=%u predecessor_loads=%u\n",
        run, ok ? "PASS" : "FAIL", active_workers, evidence.ticket_fetch_adds.load(std::memory_order_relaxed),
        evidence.exec_ticket_fetch_adds.load(std::memory_order_relaxed),
        evidence.published_metadata_writers.load(
            std::memory_order_relaxed
        ), evidence.executed_tasks.load(std::memory_order_relaxed),
        evidence.later_built_before_task0.load(std::memory_order_relaxed),
        evidence.predecessor_loads.load(std::memory_order_relaxed)
    );
    return ok;
}

bool CheckInvalidIdentityAndFutureProducer(SchedulerState &state, const std::vector<DispatchTaskIdentity> &plan) {
    if (plan.size() < 3) {
        return false;
    }
    SharedPaTaskMeta meta{};
    DispatchTaskIdentity wrong_task = plan[1];
    ++wrong_task.task_id;
    DispatchTaskIdentity wrong_batch = plan[1];
    ++wrong_batch.batch_start;
    bool ok = !DecodeDispatchIdentity(wrong_task, 1, static_cast<uint32_t>(plan.size()), meta) &&
              !DecodeDispatchIdentity(wrong_batch, 1, static_cast<uint32_t>(plan.size()), meta);

    // task 2 是首批 SF。随机访问 helper 先生成合法 QK producer，再人为
    // 污染为 producer==consumer；与真实 fanin/writer 校验相同的 <N
    // 约束必须拒绝它，不能让未来 TensorMap 条目进入 Build。
    const DispatchTaskIdentity &sf = plan[2];
    PaOrchestrationState orch{};
    TaskArgs args{};
    LocalStats stats{};
    InitPaOrchestration(orch, kDispatchBatches, &state.context_lens[0]);
    ok &= DecodeDispatchIdentity(sf, 2, static_cast<uint32_t>(plan.size()), meta);
    ok &= BindSharedPaTaskForRandomAccess(orch, sf.batch, sf.batch_start, sf.task_id, meta.kind, meta.group_index);
    orch.qk_scores = MakePlainSharedOutputRef(sf.task_id, 0);
    ok &= BuildArgsForKind(meta.kind, orch, args, sf.batch, stats);
    ok &= !SharedRefsPrecedeTask(args, sf.task_id);
    return ok;
}

}  // namespace

int main() {
    MappedSchedulerState mapped_state;
    SchedulerState *state = mapped_state.Get();
    if (state == nullptr) {
        std::perror("mmap SchedulerState");
        return 1;
    }

    SharedHostTaskPlan host_plan;
    std::vector<DispatchTaskIdentity> dispatch_plan;
    bool ok = MakeDispatchPlan(*state, host_plan, dispatch_plan);
    const uint64_t plan_fingerprint = PlanFingerprint(dispatch_plan);
    ok &= CheckInvalidIdentityAndFutureProducer(*state, dispatch_plan);
    for (uint32_t run = 1; run <= kProtocolRuns && ok; ++run) {
        ok &= RunDispatchOnce(*state, dispatch_plan, run);
        ok &= PlanFingerprint(dispatch_plan) == plan_fingerprint;
    }

    const uint64_t ticket_calls = kExpectedTasks + kDispatchWorkers;
    const double call_reduction =
        100.0 * static_cast<double>(kCurrentB256ClaimCas - ticket_calls) / static_cast<double>(kCurrentB256ClaimCas);
    std::printf(
        "[BUILD_DISPATCH] B256 Build ticket calls=%llu legacy Claim CAS=%llu "
        "call-count reduction=%.3f%% (CPU correctness only, not A5 latency)\n",
        static_cast<unsigned long long>(ticket_calls), static_cast<unsigned long long>(kCurrentB256ClaimCas),
        call_reduction
    );
    std::printf(
        "[BUILD_DISPATCH] immutable-plan/unique-ticket/sparse-writer-insert/"
        "out-of-order-build/final-closure status=%s\n",
        ok ? "PASS" : "FAIL"
    );
    return ok ? 0 : 1;
}
