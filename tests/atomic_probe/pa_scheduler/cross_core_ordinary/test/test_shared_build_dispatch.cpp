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

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <sys/mman.h>
#include <thread>
#include <vector>

#define PA_DEVICE inline
#define PA_GM
#include "pa_scheduler_core.h"

namespace {

using namespace pa_scheduler;

constexpr uint32_t kReplayBatches = 4;
constexpr uint32_t kProtocolRuns = 5;
constexpr uint32_t kDelayedWorker = 0;
constexpr uint32_t kExpectedReplayTasks =
    (1U + 4U * 0U) + (1U + 4U * 1U) +
    (1U + 4U * 2U) + (1U + 4U * 4U);

static_assert(kWorkers == 96, "A5 Scalar worker count changed");
static_assert(
    kExpectedReplayTasks == 32,
    "mixed G0/G1/G2/G4 replay task count changed"
);

struct ClaimTestOps {
    static constexpr bool kAtomicReturnReadyObserved = false;

    static int32_t Exchange(
        volatile int32_t *address, int32_t value
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

    static uint64_t Now() { return 0; }

    template <typename T>
    static uint64_t NowAfterAtomicResult(T) {
        return 0;
    }
};

struct TaskEvidence {
    std::atomic<uint32_t> attempts;
    std::atomic<uint32_t> winners;
    std::atomic<uint32_t> losers;
    std::atomic<int32_t> owner;
    std::atomic<int32_t> kind;
};

struct WorkerReplayEvidence {
    uint32_t task_count = 0;
    uint32_t winners = 0;
    uint32_t losers = 0;
    uint64_t fingerprint = 1469598103934665603ULL;
    bool valid = true;
};

template <typename T>
T *MapSparseObject() {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
    flags |= MAP_NORESERVE;
#endif
    void *memory = mmap(
        nullptr, sizeof(T), PROT_READ | PROT_WRITE,
        flags, -1, 0
    );
    if (memory == MAP_FAILED) {
        return nullptr;
    }
    return ::new (memory) T;
}

void MixFingerprint(uint64_t &hash, uint64_t value) {
    for (uint32_t byte = 0; byte < sizeof(value); ++byte) {
        hash ^= (value >> (byte * 8U)) & 0xffU;
        hash *= 1099511628211ULL;
    }
}

uint32_t DynamicReplayTaskCount(
    const std::array<int32_t, kReplayBatches> &context_lens
) {
    PaOrchestrationState orchestration{};
    InitPaOrchestration(
        orchestration, kReplayBatches, context_lens.data()
    );
    uint32_t task_count = 0;
    for (uint32_t batch = 0; batch < kReplayBatches; ++batch) {
        BeginPaBatchForCallback(orchestration, batch);
        SharedPaBatchPlan batch_plan{};
        if (!BuildSharedPaBatchPlan(
                orchestration.current_sequence,
                task_count, batch_plan
            ) || batch_plan.batch_start != task_count) {
            return 0;
        }
        task_count += batch_plan.task_count;
    }
    return task_count;
}

void ResetTaskProtocol(
    SchedulerState &state, TaskEvidence *task_evidence,
    uint32_t task_count
) {
    state.fatal.value = 0;
    for (uint32_t task_id = 0; task_id < task_count; ++task_id) {
        std::memset(
            &state.claim_tournament[task_id], 0xff,
            sizeof(state.claim_tournament[task_id])
        );
        state.tasks[task_id].deps_prepared = -1;
        task_evidence[task_id].attempts.store(
            0, std::memory_order_relaxed
        );
        task_evidence[task_id].winners.store(
            0, std::memory_order_relaxed
        );
        task_evidence[task_id].losers.store(
            0, std::memory_order_relaxed
        );
        task_evidence[task_id].owner.store(
            -1, std::memory_order_relaxed
        );
        task_evidence[task_id].kind.store(
            -1, std::memory_order_relaxed
        );
    }
}

void RecordReplayClaim(
    SchedulerState &state, uint32_t worker_id, CoreRole role,
    uint32_t batch, uint32_t group, uint32_t task_id,
    TaskKind kind, TaskEvidence &task_evidence,
    WorkerReplayEvidence &worker_evidence
) {
    LocalStats stats{};
    const ClaimOutcome outcome = Claim<ClaimTestOps>(
        &state, worker_id, role, task_id, kind, stats
    );

    ++worker_evidence.task_count;
    MixFingerprint(worker_evidence.fingerprint, task_id);
    MixFingerprint(
        worker_evidence.fingerprint,
        static_cast<uint32_t>(kind)
    );
    MixFingerprint(worker_evidence.fingerprint, batch);
    MixFingerprint(worker_evidence.fingerprint, group);

    int32_t recorded_kind = -1;
    if (!task_evidence.kind.compare_exchange_strong(
            recorded_kind, static_cast<int32_t>(kind),
            std::memory_order_acq_rel,
            std::memory_order_acquire
        ) && recorded_kind != static_cast<int32_t>(kind)) {
        worker_evidence.valid = false;
    }
    task_evidence.attempts.fetch_add(
        1, std::memory_order_relaxed
    );

    const int32_t expected_function = kind == TaskKind::Alloc
        ? -1
        : FunctionId(kind);
    if (!outcome.attempted || outcome.retries != 0 ||
        (outcome.won && outcome.function_id != expected_function) ||
        (!outcome.won && outcome.function_id != -1)) {
        worker_evidence.valid = false;
    }

    if (outcome.won) {
        ++worker_evidence.winners;
        task_evidence.winners.fetch_add(
            1, std::memory_order_relaxed
        );
        int32_t no_owner = -1;
        if (!task_evidence.owner.compare_exchange_strong(
                no_owner, static_cast<int32_t>(worker_id),
                std::memory_order_acq_rel,
                std::memory_order_acquire
            )) {
            worker_evidence.valid = false;
        }
    } else {
        ++worker_evidence.losers;
        task_evidence.losers.fetch_add(
            1, std::memory_order_relaxed
        );
    }
}

void ReplayDynamicTaskSequence(
    SchedulerState &state, uint32_t worker_id,
    const std::array<int32_t, kReplayBatches> &context_lens,
    TaskEvidence *task_evidence,
    WorkerReplayEvidence &worker_evidence
) {
    const CoreRole role = worker_id < kAicWorkers
        ? CoreRole::Aic
        : CoreRole::Aiv;
    PaOrchestrationState orchestration{};
    InitPaOrchestration(
        orchestration, kReplayBatches, context_lens.data()
    );

    uint32_t task_id = 0;
    for (uint32_t batch = 0; batch < kReplayBatches; ++batch) {
        BeginPaBatchForCallback(orchestration, batch);
        SharedPaBatchPlan batch_plan{};
        if (!BuildSharedPaBatchPlan(
                orchestration.current_sequence,
                task_id, batch_plan
            ) || batch_plan.batch_start != task_id) {
            worker_evidence.valid = false;
            return;
        }

        // 与真实 callback 回放相同：Alloc 在 group loop 之前；每个动态
        // group 再按 QK/SF/PV/UP 顺序 Submit。这里只验证 owner 仲裁，
        // 不制造 Host task identity，也不做 random-access 参数恢复。
        const uint32_t alloc_task = task_id++;
        RecordReplayClaim(
            state, worker_id, role, batch, 0,
            alloc_task, TaskKind::Alloc,
            task_evidence[alloc_task], worker_evidence
        );
        for (uint32_t group = 0;
             group < batch_plan.group_count; ++group) {
            constexpr std::array<TaskKind, 4> kGroupKinds = {
                TaskKind::Qk, TaskKind::Sf,
                TaskKind::Pv, TaskKind::Up,
            };
            for (TaskKind kind : kGroupKinds) {
                const uint32_t current_task = task_id++;
                RecordReplayClaim(
                    state, worker_id, role, batch, group,
                    current_task, kind,
                    task_evidence[current_task], worker_evidence
                );
            }
        }
        if (task_id !=
            batch_plan.batch_start + batch_plan.task_count) {
            worker_evidence.valid = false;
            return;
        }
    }
}

bool TournamentStateComplete(
    const SchedulerState &state, uint32_t task_id,
    TaskKind kind
) {
    const uint32_t active_groups = kind == TaskKind::Alloc
        ? kSharedAllocClaimTournamentGroups
        : kSharedKernelClaimTournamentGroups;
    const int64_t expected_owner = static_cast<int64_t>(task_id);
    const SharedClaimTournamentTask &tournament =
        state.claim_tournament[task_id];
    if (tournament.root.owner.value != expected_owner ||
        tournament.root.insert_completion.value != -1) {
        return false;
    }
    for (uint32_t group = 0;
         group < kSharedClaimTournamentMaxGroups; ++group) {
        const int64_t expected = group < active_groups
            ? expected_owner
            : -1;
        if (tournament.local[group].owner.value != expected) {
            return false;
        }
    }
    return true;
}

bool RunNoPrebuiltPlanReplay(
    SchedulerState &state,
    const std::array<int32_t, kReplayBatches> &context_lens,
    uint32_t run
) {
    const uint32_t dynamic_task_count =
        DynamicReplayTaskCount(context_lens);
    if (dynamic_task_count != kExpectedReplayTasks) {
        std::fprintf(
            stderr,
            "[NO_PREBUILT_PLAN] invalid dynamic task count=%u\n",
            dynamic_task_count
        );
        return false;
    }

    auto task_evidence =
        std::make_unique<TaskEvidence[]>(dynamic_task_count);
    std::array<WorkerReplayEvidence, kWorkers> worker_evidence{};
    ResetTaskProtocol(
        state, task_evidence.get(), dynamic_task_count
    );

    std::atomic<uint32_t> ready{0};
    std::atomic<uint32_t> non_delayed_done{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    workers.reserve(kWorkers);
    for (uint32_t worker_id = 0;
         worker_id < kWorkers; ++worker_id) {
        workers.emplace_back([&, worker_id]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            if (worker_id == kDelayedWorker) {
                while (non_delayed_done.load(
                           std::memory_order_acquire
                       ) != kWorkers - 1U) {
                    std::this_thread::yield();
                }
            }

            ReplayDynamicTaskSequence(
                state, worker_id, context_lens,
                task_evidence.get(), worker_evidence[worker_id]
            );
            if (worker_id != kDelayedWorker) {
                non_delayed_done.fetch_add(
                    1, std::memory_order_release
                );
            }
        });
    }

    while (ready.load(std::memory_order_acquire) != kWorkers) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (std::thread &worker : workers) {
        worker.join();
    }

    bool ok = state.fatal.value == 0 &&
        non_delayed_done.load(std::memory_order_acquire) ==
            kWorkers - 1U;
    const uint64_t expected_fingerprint =
        worker_evidence[0].fingerprint;
    uint32_t total_winners = 0;
    uint32_t total_losers = 0;
    for (uint32_t worker_id = 0;
         worker_id < kWorkers; ++worker_id) {
        const WorkerReplayEvidence &entry =
            worker_evidence[worker_id];
        ok &= entry.valid &&
            entry.task_count == dynamic_task_count &&
            entry.fingerprint == expected_fingerprint;
        total_winners += entry.winners;
        total_losers += entry.losers;
    }

    // worker0 直到其余 95 个 actor 完成整轮 replay 才进入。它仍能
    // 独立重放相同序列，但不可能成为 owner；因此任何 task 的唯一
    // owner 都由其他 Scalar 动态产生，而不是依赖预制 ticket/指定核。
    ok &= worker_evidence[kDelayedWorker].winners == 0 &&
        worker_evidence[kDelayedWorker].losers ==
            dynamic_task_count;
    for (uint32_t task_id = 0;
         task_id < dynamic_task_count; ++task_id) {
        const TaskEvidence &entry = task_evidence[task_id];
        const int32_t kind_value =
            entry.kind.load(std::memory_order_acquire);
        const int32_t owner =
            entry.owner.load(std::memory_order_acquire);
        ok &= entry.attempts.load(std::memory_order_acquire) ==
                  kWorkers &&
            entry.winners.load(std::memory_order_acquire) == 1 &&
            entry.losers.load(std::memory_order_acquire) ==
                  kWorkers - 1U &&
            owner > static_cast<int32_t>(kDelayedWorker) &&
            owner < static_cast<int32_t>(kWorkers) &&
            kind_value >= static_cast<int32_t>(TaskKind::Alloc) &&
            kind_value < static_cast<int32_t>(TaskKind::Count) &&
            TournamentStateComplete(
                state, task_id,
                static_cast<TaskKind>(kind_value)
            ) &&
            state.tasks[task_id].deps_prepared == -1;
    }
    ok &= total_winners == dynamic_task_count &&
        total_losers ==
            dynamic_task_count * (kWorkers - 1U);

    std::printf(
        "[NO_PREBUILT_PLAN] run=%u status=%s "
        "actors=%u tasks=%u attempts_per_task=%u "
        "winner_per_task=1 losers_per_task=%u "
        "delayed_worker=%u delayed_wins=%u\n",
        run, ok ? "PASS" : "FAIL", kWorkers,
        dynamic_task_count, kWorkers, kWorkers - 1U,
        kDelayedWorker,
        worker_evidence[kDelayedWorker].winners
    );
    return ok;
}

bool ReplayIdentitySealRejectsDivergence(SchedulerState &state) {
    state.fatal.value = 0;
    state.build_dispatch.next_task.value = -1;
    uint32_t identity = kSharedReplayIdentityHashSeed;
    RecordSharedReplayIdentity(
        identity, 0, 0,
        EncodeSharedPaTaskMeta(
            TaskKind::Alloc, 0, false, false
        )
    );
    LocalStats first{};
    LocalStats same{};
    LocalStats different{};
    const bool first_ok = SealSharedReplayIdentity<ClaimTestOps>(
        &state, 1, identity, first
    );
    const bool same_ok = SealSharedReplayIdentity<ClaimTestOps>(
        &state, 1, identity, same
    );
    const bool mismatch_rejected =
        !SealSharedReplayIdentity<ClaimTestOps>(
            &state, 1, identity ^ 1U, different
        );
    const uint64_t expected_packed =
        (static_cast<uint64_t>(identity) << 32U) | 1U;
    const bool ok = first_ok && same_ok && mismatch_rejected &&
        static_cast<uint64_t>(
            state.build_dispatch.next_task.value
        ) == expected_packed &&
        state.fatal.value != 0;
    std::printf(
        "[NO_PREBUILT_PLAN] replay-identity-seal status=%s\n",
        ok ? "PASS" : "FAIL"
    );
    return ok;
}

}  // namespace

int main() {
    SchedulerState *state = MapSparseObject<SchedulerState>();
    if (state == nullptr) {
        std::perror("mmap SchedulerState");
        return 1;
    }

    // 运行时输入覆盖 G0/G1/G2/G4。每个 actor 从 context_len 独立推导
    // sequence；测试中不存在 Host task plan、task identity 表或中央
    // Build ticket。
    const std::array<int32_t, kReplayBatches> context_lens = {
        0,
        static_cast<int32_t>(
            1U * kPaBlocksPerRequest * kPaBlockSize
        ),
        static_cast<int32_t>(
            2U * kPaBlocksPerRequest * kPaBlockSize
        ),
        static_cast<int32_t>(
            4U * kPaBlocksPerRequest * kPaBlockSize
        ),
    };

    bool ok = ReplayIdentitySealRejectsDivergence(*state);
    for (uint32_t run = 1;
         run <= kProtocolRuns && ok; ++run) {
        ok &= RunNoPrebuiltPlanReplay(
            *state, context_lens, run
        );
    }
    (void)munmap(state, sizeof(*state));

    std::printf(
        "[NO_PREBUILT_PLAN] independent-replay/per-task-tournament/"
        "delayed-actor status=%s\n",
        ok ? "PASS" : "FAIL"
    );
    return ok ? 0 : 1;
}
