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

constexpr uint32_t kReplayBatches = 5;
constexpr uint32_t kProtocolRuns = 5;
constexpr uint32_t kDelayedWorker = 0;
constexpr uint32_t kExpectedReplayTasks =
    (1U + 4U * 0U) + (1U + 4U * 1U) +
    (1U + 4U * 2U) + (1U + 4U * 4U) +
    (1U + 4U * 2U);

static_assert(kWorkers == 96, "A5 Scalar worker count changed");
static_assert(
    kExpectedReplayTasks == 41,
    "mixed G0/G1/G2/G4/short-tail replay task count changed"
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

struct DirectReplayTask {
    uint32_t task_id;
    uint32_t batch;
    uint32_t group;
    TaskKind kind;
    uint64_t context_length;
    bool has_following_group;
    bool is_last_submit;
};

template <size_t BatchCount, typename Visitor>
bool VisitDirectReplayTasks(
    const std::array<int32_t, BatchCount> &context_lens,
    Visitor visitor, uint32_t &task_count
) {
    // 这是测试侧独立 oracle：只从外部 context 输入按 block loop
    // 枚举 Alloc/QK/SF/PV/UP，不调用生产 batch-plan/task-at helper。
    task_count = 0;
    constexpr std::array<TaskKind, 4> kGroupKinds = {
        TaskKind::Qk, TaskKind::Sf,
        TaskKind::Pv, TaskKind::Up,
    };
    for (uint32_t batch = 0; batch < BatchCount; ++batch) {
        if (context_lens[batch] < 0) {
            return false;
        }
        const uint64_t context_length =
            static_cast<uint64_t>(context_lens[batch]);
        if (context_length > kSharedPaMaxContextLength) {
            return false;
        }
        const uint64_t block_count =
            (context_length + kPaBlockSize - 1U) /
            kPaBlockSize;
        if (task_count >= kMaxTasks ||
            !visitor(DirectReplayTask{
                task_count++, batch, 0, TaskKind::Alloc,
                context_length, false,
                block_count == 0 && batch + 1U == BatchCount,
            })) {
            return false;
        }

        uint32_t group = 0;
        uint64_t block_offset = 0;
        while (block_offset < block_count) {
            if (group >= kSharedPaMaxBlockGroups) {
                return false;
            }
            const uint64_t remaining =
                block_count - block_offset;
            const uint64_t group_blocks =
                remaining < kPaBlocksPerRequest
                ? remaining : kPaBlocksPerRequest;
            const bool has_following_group =
                block_offset + group_blocks < block_count;
            for (TaskKind kind : kGroupKinds) {
                if (task_count >= kMaxTasks ||
                    !visitor(DirectReplayTask{
                        task_count++, batch, group, kind,
                        context_length,
                        kind == TaskKind::Up &&
                            has_following_group,
                        kind == TaskKind::Up &&
                            !has_following_group &&
                            batch + 1U == BatchCount,
                    })) {
                    return false;
                }
            }
            block_offset += group_blocks;
            ++group;
        }
    }
    return true;
}

template <size_t BatchCount>
bool DirectReplayIdentity(
    const std::array<int32_t, BatchCount> &context_lens,
    uint32_t &task_count, uint32_t &identity_hash
) {
    identity_hash = kSharedReplayIdentityHashSeed;
    return VisitDirectReplayTasks(
        context_lens,
        [&](const DirectReplayTask &task) {
            if (task.kind == TaskKind::Alloc) {
                RecordSharedReplayBatchInput(
                    identity_hash, task.batch,
                    task.context_length
                );
            }
            const uint8_t meta = EncodeSharedPaTaskMeta(
                task.kind, task.group,
                task.has_following_group,
                task.is_last_submit
            );
            if (meta == 0) {
                return false;
            }
            RecordSharedReplayIdentity(
                identity_hash, task.task_id,
                task.batch, meta
            );
            return true;
        },
        task_count
    );
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
    uint32_t task_count = 0;
    worker_evidence.valid = VisitDirectReplayTasks(
        context_lens,
        [&](const DirectReplayTask &task) {
            if (task.kind == TaskKind::Alloc) {
                MixFingerprint(
                    worker_evidence.fingerprint,
                    task.context_length
                );
            }
            RecordReplayClaim(
                state, worker_id, role, task.batch, task.group,
                task.task_id, task.kind,
                task_evidence[task.task_id], worker_evidence
            );
            return worker_evidence.valid;
        },
        task_count
    );
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
    uint32_t dynamic_task_count = 0;
    const bool replay_shape_ok = VisitDirectReplayTasks(
        context_lens,
        [](const DirectReplayTask &) { return true; },
        dynamic_task_count
    );
    if (!replay_shape_ok ||
        dynamic_task_count != kExpectedReplayTasks) {
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

bool ReplayIdentitySealRejectsContextDivergence(
    SchedulerState &state
) {
    state.fatal.value = 0;
    state.replay_identity_seal.line.value = -1;
    const std::array<int32_t, 1> context_one = {1};
    const std::array<int32_t, 1> context_full_group = {8192};
    uint32_t one_task_count = 0;
    uint32_t full_task_count = 0;
    uint32_t one_identity = 0;
    uint32_t full_identity = 0;
    const bool identities_ok =
        DirectReplayIdentity(
            context_one, one_task_count, one_identity
        ) &&
        DirectReplayIdentity(
            context_full_group, full_task_count, full_identity
        );
    LocalStats first{};
    LocalStats same{};
    LocalStats different{};
    const bool first_ok = SealSharedReplayIdentity<ClaimTestOps>(
        &state, one_task_count, one_identity, first
    );
    const bool same_ok = SealSharedReplayIdentity<ClaimTestOps>(
        &state, one_task_count, one_identity, same
    );
    const bool mismatch_rejected =
        !SealSharedReplayIdentity<ClaimTestOps>(
            &state, full_task_count, full_identity, different
        );
    const uint64_t expected_packed =
        (static_cast<uint64_t>(one_identity) << 32U) |
        one_task_count;
    const bool ok = identities_ok &&
        one_task_count == 5 && full_task_count == 5 &&
        one_identity != full_identity &&
        first_ok && same_ok && mismatch_rejected &&
        static_cast<uint64_t>(
            state.replay_identity_seal.line.value
        ) == expected_packed &&
        state.fatal.value != 0;
    std::printf(
        "[NO_PREBUILT_PLAN] replay-context-seal status=%s "
        "tasks=%u/%u hash=%08x/%08x\n",
        ok ? "PASS" : "FAIL", one_task_count,
        full_task_count, one_identity, full_identity
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

    // 运行时输入覆盖 G0/G1/G2/G4，以及 8193 token 形成的第二个
    // 1-block 短尾 group。每个 actor 从 context_len 独立推导 sequence；
    // 测试中不存在 Host task plan、task identity 表或中央 Build ticket。
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
        static_cast<int32_t>(
            kPaBlocksPerRequest * kPaBlockSize + 1U
        ),
    };

    bool ok = ReplayIdentitySealRejectsContextDivergence(*state);
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
