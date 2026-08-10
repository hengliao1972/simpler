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

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "host_support.h"

namespace {

using pa_scheduler::AtomicOp;
using pa_scheduler::AtomicSite;
using pa_scheduler::SharedSubmitClaimTraceRecord;
using pa_scheduler::TaskKind;
using pa_scheduler::TraceHeader;
using pa_scheduler::TracePhase;
using pa_scheduler::TraceRecord;
using pa_scheduler::host::AtomicRecordSchemaValid;
using pa_scheduler::host::ExpandSharedTraceRecords;
using pa_scheduler::host::InitializeTraceHeader;
using pa_scheduler::host::SharedHostTaskPlan;
using pa_scheduler::host::SharedSparseTraceValidator;
using pa_scheduler::host::ValidateTraceHeader;

static_assert(
    static_cast<uint32_t>(AtomicSite::SharedReplayIdentitySeal) == 57U,
    "SharedReplayIdentitySeal is an append-only raw ABI site"
);

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] all-worker replay trace: %s\n", message);
    ++g_failures;
}

TraceRecord MakeRecord(
    TracePhase phase, int32_t task_id, int32_t function_id,
    uint64_t begin, uint64_t end, uint32_t flags = 0,
    uint32_t auxiliary = 0
) {
    TraceRecord record{};
    record.phase = static_cast<uint16_t>(phase);
    record.task_id = task_id;
    record.function_id = function_id;
    record.start_cycle = begin;
    record.end_cycle = end;
    record.flags = flags;
    record.auxiliary = static_cast<uint16_t>(auxiliary);
    return record;
}

uint32_t ReturnReadyCasFlags() {
    return static_cast<uint32_t>(AtomicOp::CompareExchange) |
           pa_scheduler::kAtomicResultUsed |
           pa_scheduler::kAtomicReturnReady;
}

TraceRecord MakeReturnReadyCas(
    AtomicSite site, int32_t task_id, uint64_t begin
) {
    return MakeRecord(
        TracePhase::Atomic, task_id, -1, begin, begin + 3U,
        ReturnReadyCasFlags(), static_cast<uint32_t>(site)
    );
}

bool IsReturnReadyCas(
    const TraceRecord &record, AtomicSite expected_site
) {
    return record.phase == static_cast<uint16_t>(TracePhase::Atomic) &&
           record.auxiliary == static_cast<uint16_t>(expected_site) &&
           (record.flags & pa_scheduler::kAtomicOpMask) ==
               static_cast<uint32_t>(AtomicOp::CompareExchange) &&
           (record.flags & pa_scheduler::kAtomicResultUsed) != 0 &&
           (record.flags & pa_scheduler::kAtomicReturnReady) != 0 &&
           (record.flags & pa_scheduler::kAtomicPollBatch) == 0 &&
           AtomicRecordSchemaValid(record, true);
}

void TestTraceBinaryLayoutAndHeaderGate() {
    Check(
        sizeof(TraceRecord) == 32 && alignof(TraceRecord) == 32 &&
            pa_scheduler::kTraceRecordSizeBytes == 32,
        "generic trace record remains one 32-byte half cache line"
    );
    Check(
        sizeof(SharedSubmitClaimTraceRecord) == 32 &&
            alignof(SharedSubmitClaimTraceRecord) == 32 &&
            pa_scheduler::kTraceSubmitClaimRecordSizeBytes == 32,
        "Submit/Claim compact record remains one 32-byte half cache line"
    );
    Check(
        offsetof(TraceHeader, cores) == 64 &&
            sizeof(TraceHeader) == 6976,
        "trace header ABI remains stable"
    );
    Check(
        pa_scheduler::kTraceSubmitClaimBytesPerCore ==
                static_cast<size_t>(pa_scheduler::kMaxTasks) *
                    sizeof(SharedSubmitClaimTraceRecord) &&
            pa_scheduler::kTraceSubmitClaimBytesPerCore +
                    static_cast<size_t>(
                        pa_scheduler::kTraceRecordsPerCore
                    ) * sizeof(TraceRecord) ==
                pa_scheduler::kTraceWorkerBytes &&
            pa_scheduler::kTraceWorkerBytes == (1U << 20),
        "compact and generic regions exactly fill one worker partition"
    );

    TraceHeader header{};
    InitializeTraceHeader(&header);
    Check(
        header.record_size_bytes == sizeof(TraceRecord) &&
            ValidateTraceHeader(header, "all-worker trace ABI"),
        "initialized trace header accepts the current ABI"
    );
    header.record_size_bytes = 64;
    Check(
        !ValidateTraceHeader(header, "all-worker trace ABI negative"),
        "trace header rejects the obsolete 64-byte record ABI"
    );
}

void TestNewAtomicSchemas() {
    const TraceRecord handoff = MakeReturnReadyCas(
        AtomicSite::SharedInsertTurnHandoff, 3, 100
    );
    Check(
        IsReturnReadyCas(
            handoff, AtomicSite::SharedInsertTurnHandoff
        ),
        "insert completion handoff is a task-indexed return-ready CAS"
    );

    TraceRecord bad = handoff;
    bad.flags =
        (bad.flags & ~pa_scheduler::kAtomicOpMask) |
        static_cast<uint32_t>(AtomicOp::FetchAdd);
    Check(
        !IsReturnReadyCas(
            bad, AtomicSite::SharedInsertTurnHandoff
        ),
        "insert completion handoff rejects the old FetchAdd schema"
    );
    bad = handoff;
    bad.flags &= ~pa_scheduler::kAtomicReturnReady;
    Check(
        !IsReturnReadyCas(
            bad, AtomicSite::SharedInsertTurnHandoff
        ),
        "insert completion handoff must include the CAS return boundary"
    );
    bad = handoff;
    bad.task_id = -1;
    Check(
        !IsReturnReadyCas(
            bad, AtomicSite::SharedInsertTurnHandoff
        ),
        "insert completion handoff must retain its task identity"
    );

    const TraceRecord seal = MakeReturnReadyCas(
        AtomicSite::SharedReplayIdentitySeal, -1, 200
    );
    Check(
        IsReturnReadyCas(seal, AtomicSite::SharedReplayIdentitySeal),
        "site 57 replay identity seal is a return-ready CAS"
    );
    bad = seal;
    bad.flags &= ~pa_scheduler::kAtomicResultUsed;
    Check(
        !IsReturnReadyCas(bad, AtomicSite::SharedReplayIdentitySeal),
        "replay identity seal must consume the CAS observation"
    );
    bad = seal;
    bad.flags =
        (bad.flags & ~pa_scheduler::kAtomicOpMask) |
        static_cast<uint32_t>(AtomicOp::FetchAdd);
    Check(
        !IsReturnReadyCas(bad, AtomicSite::SharedReplayIdentitySeal),
        "replay identity seal rejects a central-ticket FetchAdd shape"
    );
}

struct ReplayEntry {
    uint32_t task_id = 0;
    SharedSubmitClaimTraceRecord endpoints{};
    std::vector<TraceRecord> winner_children;
    std::vector<TraceRecord> control_atomics;
};

struct WorkerReplay {
    std::vector<ReplayEntry> tasks;
    std::vector<TraceRecord> replay_atomics;
};

TaskKind KindForTask(uint32_t task_id) {
    return static_cast<TaskKind>(
        task_id % pa_scheduler::kTasksPerBatch
    );
}

SharedHostTaskPlan MakeTraceOraclePlan(uint32_t total_tasks) {
    SharedHostTaskPlan plan;
    plan.total_tasks = total_tasks;
    plan.tasks.resize(total_tasks);
    for (uint32_t task_id = 0; task_id < total_tasks; ++task_id) {
        plan.tasks[task_id].task_id = task_id;
        plan.tasks[task_id].kind = KindForTask(task_id);
    }
    return plan;
}

int32_t FunctionId(TaskKind kind) {
    return kind == TaskKind::Alloc
        ? -1
        : static_cast<int32_t>(static_cast<uint32_t>(kind) - 1U);
}

bool EndpointWinner(const SharedSubmitClaimTraceRecord &record) {
    return (record.claim_end_and_winner &
            pa_scheduler::kSharedClaimWinnerBit) != 0;
}

uint64_t EndpointClaimEnd(
    const SharedSubmitClaimTraceRecord &record
) {
    return record.claim_end_and_winner &
           ~pa_scheduler::kSharedClaimWinnerBit;
}

bool CompactEndpointsValid(
    const SharedSubmitClaimTraceRecord &record
) {
    const uint64_t claim_end = EndpointClaimEnd(record);
    return record.submit_begin != 0 && record.claim_begin != 0 &&
           (record.submit_begin &
            pa_scheduler::kSharedClaimWinnerBit) == 0 &&
           (record.claim_begin &
            pa_scheduler::kSharedClaimWinnerBit) == 0 &&
           (record.submit_end &
            pa_scheduler::kSharedClaimWinnerBit) == 0 &&
           record.submit_begin <= record.claim_begin &&
           record.claim_begin <= claim_end &&
           claim_end <= record.submit_end;
}

// 历史文件名沿用 sparse_trace，但新门槛刻意不调用旧
// SharedSparseTraceValidator/ExpandSharedTraceRecords：二者当前仍允许逐核
// 空槽和 task-id 跳号，并用 global_task_coverage 拒绝第二个 worker 的同一
// task。这里直接验证设备 raw 的新合同，避免测试继续替 central owner 背书。
bool ValidateAllWorkerReplay(
    const std::vector<WorkerReplay> &workers,
    uint32_t total_tasks
) {
    if (workers.size() != pa_scheduler::kWorkers || total_tasks == 0) {
        return false;
    }

    std::vector<uint32_t> winner_count(total_tasks, 0);
    std::vector<uint32_t> handoff_count(total_tasks, 0);
    for (uint32_t worker = 0; worker < workers.size(); ++worker) {
        const WorkerReplay &replay = workers[worker];
        if (replay.tasks.size() != total_tasks) {
            return false;
        }

        uint32_t seal_count = 0;
        for (const TraceRecord &record : replay.replay_atomics) {
            if (record.auxiliary ==
                static_cast<uint16_t>(
                    AtomicSite::SharedBuildDispatchTicket
                )) {
                return false;
            }
            if (!IsReturnReadyCas(
                    record, AtomicSite::SharedReplayIdentitySeal
                ) || record.task_id != -1) {
                return false;
            }
            ++seal_count;
        }
        if (seal_count != 1) {
            return false;
        }

        for (uint32_t expected_task_id = 0;
             expected_task_id < total_tasks; ++expected_task_id) {
            const ReplayEntry &entry = replay.tasks[expected_task_id];
            // 每核真实 replay 必须是完整 0..N-1 序列。raw 本身按 task_id
            // 定址，显式 identity 再锁住生成侧不得跳号或重复覆盖。
            if (entry.task_id != expected_task_id ||
                !CompactEndpointsValid(entry.endpoints)) {
                return false;
            }

            const bool winner = EndpointWinner(entry.endpoints);
            winner_count[expected_task_id] += winner ? 1U : 0U;
            const TaskKind kind = KindForTask(expected_task_id);
            const TracePhase expected_tail = kind == TaskKind::Alloc
                ? TracePhase::AllocComplete
                : TracePhase::WinnerBuild;
            if (entry.winner_children.size() != (winner ? 1U : 0U)) {
                return false;
            }
            for (const TraceRecord &child : entry.winner_children) {
                if (child.phase != static_cast<uint16_t>(expected_tail) ||
                    child.task_id !=
                        static_cast<int32_t>(expected_task_id) ||
                    child.function_id != FunctionId(kind) ||
                    child.flags != 0 || child.auxiliary != 0 ||
                    child.start_cycle < entry.endpoints.claim_begin ||
                    child.end_cycle < child.start_cycle ||
                    child.end_cycle > entry.endpoints.submit_end) {
                    return false;
                }
            }

            for (const TraceRecord &atomic : entry.control_atomics) {
                if (!IsReturnReadyCas(
                        atomic,
                        AtomicSite::SharedInsertTurnHandoff
                    ) || atomic.task_id !=
                            static_cast<int32_t>(expected_task_id) ||
                    atomic.start_cycle <
                        entry.endpoints.claim_begin ||
                    atomic.end_cycle > entry.endpoints.submit_end) {
                    return false;
                }
                ++handoff_count[expected_task_id];
            }
            // 只有 Build winner 推进严格插入完成字；loser 不产生任何
            // winner 子区间，也不得发布 completion。
            if (entry.control_atomics.size() != (winner ? 1U : 0U)) {
                return false;
            }
        }
    }

    for (uint32_t task_id = 0; task_id < total_tasks; ++task_id) {
        if (winner_count[task_id] != 1 || handoff_count[task_id] != 1) {
            return false;
        }
    }
    return true;
}

std::vector<WorkerReplay> MakeValidReplay(uint32_t total_tasks) {
    std::vector<WorkerReplay> workers(pa_scheduler::kWorkers);
    for (uint32_t worker = 0; worker < pa_scheduler::kWorkers; ++worker) {
        WorkerReplay &replay = workers[worker];
        replay.tasks.reserve(total_tasks);
        for (uint32_t task_id = 0; task_id < total_tasks; ++task_id) {
            // 不把 winner 固定到单个 central worker；五类 task 的 owner
            // 分散在完整 Scalar population 中。
            const uint32_t owner =
                (task_id * 17U + 3U) % pa_scheduler::kWorkers;
            const bool winner = worker == owner;
            const uint64_t base =
                1000U + static_cast<uint64_t>(worker) * 10000U +
                static_cast<uint64_t>(task_id) * 100U;
            ReplayEntry entry{};
            entry.task_id = task_id;
            entry.endpoints = SharedSubmitClaimTraceRecord{
                base + 20U,
                (base + 28U) |
                    (winner
                         ? pa_scheduler::kSharedClaimWinnerBit
                         : 0ULL),
                base + 10U,
                base + 80U,
            };
            if (winner) {
                const TaskKind kind = KindForTask(task_id);
                entry.winner_children.push_back(
                    MakeRecord(
                        kind == TaskKind::Alloc
                            ? TracePhase::AllocComplete
                            : TracePhase::WinnerBuild,
                        static_cast<int32_t>(task_id),
                        FunctionId(kind), base + 50U, base + 65U
                    )
                );
                entry.control_atomics.push_back(
                    MakeReturnReadyCas(
                        AtomicSite::SharedInsertTurnHandoff,
                        static_cast<int32_t>(task_id), base + 40U
                    )
                );
            }
            replay.tasks.push_back(entry);
        }
        replay.replay_atomics.push_back(
            MakeReturnReadyCas(
                AtomicSite::SharedReplayIdentitySeal, -1,
                1000U + static_cast<uint64_t>(worker) * 10000U +
                    static_cast<uint64_t>(total_tasks) * 100U
            )
        );
    }
    return workers;
}

void TestAcceptsCompleteAllWorkerReplay() {
    constexpr uint32_t kTasks = 10;
    const std::vector<WorkerReplay> workers = MakeValidReplay(kTasks);
    Check(
        ValidateAllWorkerReplay(workers, kTasks),
        "96 workers replay every task while each task has one Build winner"
    );

    uint64_t compact_records = 0;
    uint64_t winner_children = 0;
    for (const WorkerReplay &worker : workers) {
        compact_records += worker.tasks.size();
        for (const ReplayEntry &entry : worker.tasks) {
            winner_children += entry.winner_children.size();
        }
    }
    Check(
        compact_records ==
                static_cast<uint64_t>(pa_scheduler::kWorkers) * kTasks &&
            winner_children == kTasks,
        "compact record population is workers*tasks, not one sparse owner per task"
    );

    // 生产导出 API 也必须逐核展开全部 N 个 Claim/Submit，而不是仅展开
    // winner 的稀疏槽。Host plan 在这里仅给泳道补 task kind/function_id，
    // 不承担 owner 或 dispatch 输入。
    const SharedHostTaskPlan plan = MakeTraceOraclePlan(kTasks);
    TraceRecord unused_generic{};
    bool all_expanded = true;
    for (uint32_t worker = 0; worker < workers.size(); ++worker) {
        std::vector<SharedSubmitClaimTraceRecord> compact;
        compact.reserve(kTasks);
        for (const ReplayEntry &entry : workers[worker].tasks) {
            compact.push_back(entry.endpoints);
        }
        std::vector<TraceRecord> logical;
        const bool expanded = ExpandSharedTraceRecords(
            worker, &unused_generic, 0, compact.data(), plan,
            &logical
        );
        bool exact_sequence = expanded && logical.size() == 2U * kTasks;
        for (uint32_t task_id = 0;
             exact_sequence && task_id < kTasks; ++task_id) {
            const TraceRecord &claim = logical[2U * task_id];
            const TraceRecord &submit = logical[2U * task_id + 1U];
            const bool winner =
                EndpointWinner(compact[task_id]);
            exact_sequence &=
                claim.phase ==
                    static_cast<uint16_t>(TracePhase::Claim) &&
                submit.phase ==
                    static_cast<uint16_t>(TracePhase::Submit) &&
                claim.task_id == static_cast<int32_t>(task_id) &&
                submit.task_id == static_cast<int32_t>(task_id) &&
                claim.flags ==
                    (pa_scheduler::kClaimAttempted |
                     (winner ? pa_scheduler::kClaimWon : 0U)) &&
                submit.flags ==
                    (winner ? pa_scheduler::kClaimWon : 0U);
        }
        // worker 0 在这组分散 owner 中恰好没有 winner，可直接让生产
        // validator 闭合完整 loser replay，证明它不再接受 task-id 跳号。
        if (exact_sequence && worker == 0) {
            SharedSparseTraceValidator validator(&plan);
            for (const TraceRecord &record : logical) {
                exact_sequence &= validator.Observe(record);
            }
            exact_sequence &= validator.Closed() &&
                validator.ClaimCount() == kTasks &&
                validator.SubmitCount() == kTasks &&
                validator.WinnerCount() == 0;
        }
        if (!exact_sequence) {
            all_expanded = false;
            break;
        }
    }
    Check(
        all_expanded,
        "host expansion reconstructs every worker's complete replay"
    );
}

void TestRejectsReplayCoverageDrift() {
    constexpr uint32_t kTasks = 5;
    const std::vector<WorkerReplay> valid = MakeValidReplay(kTasks);

    std::vector<WorkerReplay> bad = valid;
    bad[7].tasks.erase(bad[7].tasks.begin() + 2);
    Check(
        !ValidateAllWorkerReplay(bad, kTasks),
        "one worker cannot omit a replayed task"
    );

    bad = valid;
    bad[7].tasks[2].task_id = 1;
    Check(
        !ValidateAllWorkerReplay(bad, kTasks),
        "one worker cannot duplicate an earlier task identity"
    );

    bad = valid;
    bad[7].tasks[2].task_id = 3;
    Check(
        !ValidateAllWorkerReplay(bad, kTasks),
        "one worker cannot skip ahead in task-id order"
    );

    bad = valid;
    bad[7].tasks[2].endpoints = {};
    Check(
        !ValidateAllWorkerReplay(bad, kTasks),
        "an all-zero compact slot is no longer a legal sparse hole"
    );
}

void TestRejectsWinnerAndInsertCompletionDrift() {
    constexpr uint32_t kTasks = 5;
    const std::vector<WorkerReplay> valid = MakeValidReplay(kTasks);
    constexpr uint32_t task_id = 2;
    const uint32_t owner =
        (task_id * 17U + 3U) % pa_scheduler::kWorkers;
    const uint32_t loser = (owner + 1U) % pa_scheduler::kWorkers;

    std::vector<WorkerReplay> bad = valid;
    bad[loser].tasks[task_id].winner_children.push_back(
        bad[owner].tasks[task_id].winner_children.front()
    );
    Check(
        !ValidateAllWorkerReplay(bad, kTasks),
        "loser cannot carry a winner-only business interval"
    );

    bad = valid;
    bad[owner].tasks[task_id].winner_children.clear();
    Check(
        !ValidateAllWorkerReplay(bad, kTasks),
        "the unique winner must retain its winner-only business interval"
    );

    bad = valid;
    bad[loser].tasks[task_id].endpoints.claim_end_and_winner |=
        pa_scheduler::kSharedClaimWinnerBit;
    {
        ReplayEntry &second_winner = bad[loser].tasks[task_id];
        const TaskKind kind = KindForTask(task_id);
        const uint64_t claim_begin =
            second_winner.endpoints.claim_begin;
        second_winner.winner_children.push_back(
            MakeRecord(
                kind == TaskKind::Alloc
                    ? TracePhase::AllocComplete
                    : TracePhase::WinnerBuild,
                static_cast<int32_t>(task_id), FunctionId(kind),
                claim_begin + 20U, claim_begin + 30U
            )
        );
        second_winner.control_atomics.push_back(
            MakeReturnReadyCas(
                AtomicSite::SharedInsertTurnHandoff,
                static_cast<int32_t>(task_id), claim_begin + 10U
            )
        );
    }
    Check(
        !ValidateAllWorkerReplay(bad, kTasks),
        "a task cannot have two global Build winners"
    );

    bad = valid;
    bad[owner].tasks[task_id].endpoints.claim_end_and_winner &=
        ~pa_scheduler::kSharedClaimWinnerBit;
    bad[owner].tasks[task_id].winner_children.clear();
    bad[owner].tasks[task_id].control_atomics.clear();
    Check(
        !ValidateAllWorkerReplay(bad, kTasks),
        "a task cannot finish replay without a Build winner"
    );

    bad = valid;
    bad[owner].tasks[task_id].control_atomics.clear();
    Check(
        !ValidateAllWorkerReplay(bad, kTasks),
        "every task must advance the strict insert completion chain once"
    );

    bad = valid;
    bad[owner].tasks[task_id].control_atomics.push_back(
        bad[owner].tasks[task_id].control_atomics.front()
    );
    Check(
        !ValidateAllWorkerReplay(bad, kTasks),
        "a task cannot publish insert completion twice"
    );

    bad = valid;
    bad[owner].tasks[task_id].control_atomics.front().flags &=
        ~pa_scheduler::kAtomicReturnReady;
    Check(
        !ValidateAllWorkerReplay(bad, kTasks),
        "insert completion CAS must keep its return-ready boundary"
    );
}

void TestRejectsReplaySealAndCentralTicketDrift() {
    constexpr uint32_t kTasks = 5;
    const std::vector<WorkerReplay> valid = MakeValidReplay(kTasks);

    std::vector<WorkerReplay> bad = valid;
    bad[11].replay_atomics.clear();
    Check(
        !ValidateAllWorkerReplay(bad, kTasks),
        "every worker must seal the replayed identity/count once"
    );

    bad = valid;
    bad[11].replay_atomics.push_back(
        bad[11].replay_atomics.front()
    );
    Check(
        !ValidateAllWorkerReplay(bad, kTasks),
        "one worker cannot publish two replay seals"
    );

    bad = valid;
    bad[11].replay_atomics.front().flags &=
        ~pa_scheduler::kAtomicReturnReady;
    Check(
        !ValidateAllWorkerReplay(bad, kTasks),
        "replay seal must expose the return-ready CAS observation"
    );

    bad = valid;
    bad[11].replay_atomics.push_back(
        MakeRecord(
            TracePhase::Atomic, -1, -1, 500, 501,
            static_cast<uint32_t>(AtomicOp::FetchAdd) |
                pa_scheduler::kAtomicResultUsed,
            static_cast<uint32_t>(
                AtomicSite::SharedBuildDispatchTicket
            )
        )
    );
    Check(
        !ValidateAllWorkerReplay(bad, kTasks),
        "all-worker replay rejects the retired central Build ticket"
    );
}

}  // namespace

int main() {
    TestTraceBinaryLayoutAndHeaderGate();
    TestNewAtomicSchemas();
    TestAcceptsCompleteAllWorkerReplay();
    TestRejectsReplayCoverageDrift();
    TestRejectsWinnerAndInsertCompletionDrift();
    TestRejectsReplaySealAndCentralTicketDrift();
    if (g_failures != 0) {
        std::fprintf(
            stderr, "[FAIL] all-worker replay trace tests: %d\n",
            g_failures
        );
        return 1;
    }
    std::printf("[PASS] all-worker replay trace tests\n");
    return 0;
}
