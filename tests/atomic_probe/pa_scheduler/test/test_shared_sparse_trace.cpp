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

#include <cstdio>
#include <vector>

#include "host_support.h"

namespace {

using pa_scheduler::TaskKind;
using pa_scheduler::TracePhase;
using pa_scheduler::TraceHeader;
using pa_scheduler::TraceRecord;
using pa_scheduler::SharedSubmitClaimTraceRecord;
using pa_scheduler::AtomicOp;
using pa_scheduler::AtomicSite;
using pa_scheduler::kAtomicOpMask;
using pa_scheduler::kAtomicPollBatch;
using pa_scheduler::kAtomicPollCountShift;
using pa_scheduler::kAtomicResultUsed;
using pa_scheduler::kAtomicReturnReady;
using pa_scheduler::kTraceRecordSizeBytes;
using pa_scheduler::kTraceRecordsPerCore;
using pa_scheduler::kTraceSubmitClaimBytesPerCore;
using pa_scheduler::kTraceSubmitClaimRecordSizeBytes;
using pa_scheduler::kTraceWorkerBytes;
using pa_scheduler::host::AtomicRecordSchemaValid;
using pa_scheduler::host::ExpandSharedTraceRecords;
using pa_scheduler::host::InitializeTraceHeader;
using pa_scheduler::host::SharedHostTaskPlan;
using pa_scheduler::host::SharedSparseTraceValidator;
using pa_scheduler::host::ValidateTraceHeader;

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) return;
    std::fprintf(stderr, "[FAIL] shared sparse trace: %s\n", message);
    ++g_failures;
}

TraceRecord MakeRecord(
    TracePhase phase, int32_t task_id, int32_t function_id,
    uint64_t begin, uint64_t end, uint32_t flags = 0,
    uint32_t auxiliary = 0
) {
    TraceRecord record{};
    record.phase = static_cast<int32_t>(phase);
    record.task_id = task_id;
    record.function_id = function_id;
    record.start_cycle = begin;
    record.end_cycle = end;
    record.flags = flags;
    record.auxiliary = auxiliary;
    return record;
}

bool SameRecord(const TraceRecord &left, const TraceRecord &right) {
    return left.start_cycle == right.start_cycle &&
           left.end_cycle == right.end_cycle &&
           left.task_id == right.task_id &&
           left.function_id == right.function_id &&
           left.flags == right.flags &&
           left.phase == right.phase &&
           left.auxiliary == right.auxiliary;
}

void TestTraceBinaryLayoutAndHeaderGate() {
    Check(
        sizeof(TraceRecord) == 32 &&
            alignof(TraceRecord) == 32 &&
            kTraceRecordSizeBytes == 32,
        "device trace record must remain a 32-byte half-cache-line"
    );
    Check(
        offsetof(TraceHeader, cores) == 64 &&
            sizeof(TraceHeader) == 6976,
        "record-size field must not move or grow the header core array"
    );
    constexpr size_t partition_bytes =
        static_cast<size_t>(kTraceRecordsPerCore) *
        sizeof(TraceRecord);
    Check(
        sizeof(TraceHeader) % 64 == 0 &&
            partition_bytes % 64 == 0,
        "header and every worker record partition must start on a cache line"
    );
    Check(
        (0U % 64U) + sizeof(TraceRecord) <= 64U &&
            (32U % 64U) + sizeof(TraceRecord) <= 64U &&
            0U / 64U == 32U / 64U &&
            64U / 64U != 32U / 64U,
        "two trace records must fit exactly in one cache line"
    );
    Check(
        sizeof(SharedSubmitClaimTraceRecord) == 32 &&
            alignof(SharedSubmitClaimTraceRecord) == 32 &&
            kTraceSubmitClaimRecordSizeBytes == 32,
        "shared Submit/Claim record must remain 32-byte aligned"
    );
    Check(
        kTraceSubmitClaimBytesPerCore ==
                static_cast<size_t>(pa_scheduler::kMaxTasks) *
                    sizeof(SharedSubmitClaimTraceRecord) &&
            kTraceSubmitClaimBytesPerCore +
                    static_cast<size_t>(kTraceRecordsPerCore) *
                        sizeof(TraceRecord) ==
                kTraceWorkerBytes &&
            kTraceWorkerBytes == (1U << 20),
        "shared compact and generic regions must exactly fill one 1 MiB worker partition"
    );
    Check(
        pa_scheduler::TraceSubmitClaimOffset(0) % 64U == 0 &&
            pa_scheduler::TraceRecordsOffset(0) % 64U == 0 &&
            pa_scheduler::TraceRecordsOffset(0) -
                    pa_scheduler::TraceSubmitClaimOffset(0) ==
                kTraceSubmitClaimBytesPerCore &&
            pa_scheduler::TraceWorkerOffset(1) -
                    pa_scheduler::TraceWorkerOffset(0) ==
                kTraceWorkerBytes,
        "shared compact and generic worker regions must keep cache-line isolation"
    );

    TraceHeader header{};
    InitializeTraceHeader(&header);
    Check(
        header.record_size_bytes == 32 &&
            ValidateTraceHeader(header, "trace ABI self-test"),
        "initialized header must publish and accept a 32-byte raw ABI"
    );
    header.record_size_bytes = 64;
    Check(
        !ValidateTraceHeader(header, "trace ABI negative self-test"),
        "header validation must reject the old 64-byte record ABI"
    );
}

int32_t FunctionId(TaskKind kind) {
    return kind == TaskKind::Alloc
        ? -1
        : static_cast<int32_t>(static_cast<uint32_t>(kind) - 1U);
}

uint32_t IsAlloc(TaskKind kind) {
    return kind == TaskKind::Alloc ? 1U : 0U;
}

struct CompactTraceWindow {
    uint64_t submit_begin;
    uint64_t submit_end;
    uint64_t claim_begin;
    uint64_t claim_end;
};

CompactTraceWindow WindowForTask(
    uint64_t base_cycle, uint32_t task_id
) {
    const uint64_t task_base =
        base_cycle + static_cast<uint64_t>(task_id) * 100U;
    return CompactTraceWindow{
        task_base + 10U,
        task_base + 60U,
        task_base + 20U,
        task_base + 25U,
    };
}

SharedSubmitClaimTraceRecord MakeCompactRecord(
    const CompactTraceWindow &window, bool winner
) {
    return SharedSubmitClaimTraceRecord{
        window.claim_begin,
        window.claim_end |
            (winner
                 ? pa_scheduler::kSharedClaimWinnerBit
                 : 0ULL),
        window.submit_begin,
        window.submit_end,
    };
}

SharedHostTaskPlan MakeCompactTracePlan() {
    constexpr TaskKind kinds[] = {
        TaskKind::Alloc,
        TaskKind::Qk,
        TaskKind::Sf,
        TaskKind::Pv,
        TaskKind::Up,
    };
    SharedHostTaskPlan plan;
    plan.total_tasks =
        static_cast<uint32_t>(
            sizeof(kinds) / sizeof(kinds[0])
        );
    plan.tasks.resize(plan.total_tasks);
    for (uint32_t task_id = 0;
         task_id < plan.total_tasks; ++task_id) {
        plan.tasks[task_id].task_id = task_id;
        plan.tasks[task_id].kind = kinds[task_id];
    }
    return plan;
}

void TestSharedCompactReconstruction() {
    constexpr uint32_t worker = 0;
    constexpr uint64_t base_cycle = 5000;
    const SharedHostTaskPlan plan = MakeCompactTracePlan();
    constexpr bool winners[] = {
        true, false, false, false, false,
    };
    constexpr bool attempted[] = {
        true, false, false, false, false,
    };
    std::vector<SharedSubmitClaimTraceRecord> compact(
        plan.total_tasks
    );
    std::vector<TraceRecord> generic;
    for (uint32_t task_id = 0;
         task_id < plan.total_tasks; ++task_id) {
        const CompactTraceWindow window =
            WindowForTask(base_cycle, task_id);
        compact[task_id] =
            MakeCompactRecord(window, winners[task_id]);
        if (attempted[task_id]) {
            generic.push_back(
                MakeRecord(
                    TracePhase::Atomic,
                    static_cast<int32_t>(task_id), -1,
                    window.claim_begin + 1U,
                    window.claim_begin + 2U,
                    static_cast<uint32_t>(
                        AtomicOp::FetchMax
                    ) |
                        kAtomicResultUsed |
                        kAtomicReturnReady,
                    static_cast<uint32_t>(
                        AtomicSite::ClaimMax
                    )
                )
            );
        }
    }

    std::vector<TraceRecord> logical;
    Check(
        ExpandSharedTraceRecords(
            worker, generic.data(),
            static_cast<uint32_t>(generic.size()),
            compact.data(), plan, &logical
        ),
        "four-endpoint records and generic ClaimMax rows reconstruct"
    );
    Check(
        logical.size() ==
            generic.size() + 2U * plan.total_tasks,
        "reconstruction preserves every generic row and adds Claim/Submit"
    );
    if (logical.size() !=
        generic.size() + 2U * plan.total_tasks) {
        return;
    }

    size_t index = 0;
    for (uint32_t task_id = 0;
         task_id < plan.total_tasks; ++task_id) {
        if (attempted[task_id]) {
            Check(
                logical[index].phase ==
                        static_cast<uint16_t>(
                            TracePhase::Atomic
                        ) &&
                    logical[index].task_id ==
                        static_cast<int32_t>(task_id) &&
                    logical[index].auxiliary ==
                        static_cast<uint16_t>(
                            AtomicSite::ClaimMax
                        ) &&
                    AtomicRecordSchemaValid(
                        logical[index], true
                    ),
                "generic ClaimMax remains an exact return-ready Atomic row"
            );
            ++index;
        }
        const TraceRecord &claim = logical[index++];
        const TraceRecord &submit = logical[index++];
        const CompactTraceWindow window =
            WindowForTask(base_cycle, task_id);
        Check(
            claim.phase ==
                    static_cast<uint16_t>(TracePhase::Claim) &&
                claim.start_cycle == window.claim_begin &&
                claim.end_cycle == window.claim_end &&
                claim.flags ==
                    ((winners[task_id]
                          ? pa_scheduler::kClaimWon
                          : 0U) |
                     (attempted[task_id]
                          ? pa_scheduler::kClaimAttempted
                          : 0U)),
            "Claim reconstructs absolute endpoints and role-derived attempted"
        );
        Check(
            submit.phase ==
                    static_cast<uint16_t>(
                        TracePhase::Submit
                    ) &&
                submit.start_cycle == window.submit_begin &&
                submit.end_cycle == window.submit_end &&
                submit.flags ==
                    (winners[task_id]
                         ? pa_scheduler::kClaimWon
                         : 0U),
            "Submit reconstructs absolute endpoints and winner"
        );
    }
}

void TestSharedCompactStageOnlyReconstruction() {
    constexpr uint32_t worker = 0;
    constexpr uint64_t base_cycle = 6000;
    const SharedHostTaskPlan plan = MakeCompactTracePlan();
    std::vector<SharedSubmitClaimTraceRecord> compact(
        plan.total_tasks
    );
    for (uint32_t task_id = 0;
         task_id < plan.total_tasks; ++task_id) {
        compact[task_id] = MakeCompactRecord(
            WindowForTask(base_cycle, task_id), false
        );
    }
    TraceRecord unused_generic{};
    std::vector<TraceRecord> logical;
    Check(
        ExpandSharedTraceRecords(
            worker, &unused_generic, 0,
            compact.data(), plan, &logical
        ) &&
            logical.size() == 2U * plan.total_tasks,
        "stage-only records expand to Claim/Submit without Atomic"
    );
}

void TestSharedCompactGenericMergeOrder() {
    constexpr uint32_t worker = 0;
    constexpr uint64_t base_cycle = 7000;
    const SharedHostTaskPlan plan = MakeCompactTracePlan();
    std::vector<SharedSubmitClaimTraceRecord> compact(
        plan.total_tasks
    );
    for (uint32_t task_id = 0;
         task_id < plan.total_tasks; ++task_id) {
        compact[task_id] = MakeCompactRecord(
            WindowForTask(base_cycle, task_id), false
        );
    }
    const CompactTraceWindow task0 =
        WindowForTask(base_cycle, 0);
    std::vector<TraceRecord> generic{
        MakeRecord(
            TracePhase::Atomic, 0, -1,
            task0.claim_begin + 1U,
            task0.claim_begin + 2U,
            static_cast<uint32_t>(AtomicOp::FetchMax) |
                kAtomicResultUsed | kAtomicReturnReady,
            static_cast<uint32_t>(AtomicSite::ClaimMax)
        ),
        MakeRecord(
            TracePhase::RingBp, 0, 9,
            task0.claim_end + 1U,
            task0.claim_end + 2U
        ),
    };
    std::vector<TraceRecord> logical;
    Check(
        ExpandSharedTraceRecords(
            worker, generic.data(),
            static_cast<uint32_t>(generic.size()),
            compact.data(), plan, &logical
        ) &&
            logical.size() ==
                generic.size() + 2U * plan.total_tasks &&
            SameRecord(logical[0], generic[0]) &&
            logical[1].phase ==
                static_cast<uint16_t>(TracePhase::Claim) &&
            SameRecord(logical[2], generic[1]) &&
            logical[3].phase ==
                static_cast<uint16_t>(TracePhase::Submit),
        "generic rows merge stably before their enclosing Claim/Submit endpoint"
    );
}

void TestRejectsBadSharedCompactRecords() {
    constexpr uint32_t worker = 0;
    constexpr uint64_t base_cycle = 8000;
    const SharedHostTaskPlan plan = MakeCompactTracePlan();
    std::vector<SharedSubmitClaimTraceRecord> valid(
        plan.total_tasks
    );
    for (uint32_t task_id = 0;
         task_id < plan.total_tasks; ++task_id) {
        valid[task_id] = MakeCompactRecord(
            WindowForTask(base_cycle, task_id), false
        );
    }
    TraceRecord unused_generic{};
    auto accepted_by = [&](
        uint32_t observed_worker,
        const std::vector<SharedSubmitClaimTraceRecord> &records
    ) {
        std::vector<TraceRecord> logical;
        return ExpandSharedTraceRecords(
            observed_worker, &unused_generic, 0,
            records.data(), plan, &logical
        );
    };
    auto rejected = [&](
        const std::vector<SharedSubmitClaimTraceRecord> &records
    ) {
        return !accepted_by(worker, records);
    };

    std::vector<SharedSubmitClaimTraceRecord> bad = valid;
    bad[0].submit_begin = 0;
    Check(rejected(bad), "missing Submit.begin is rejected");

    bad = valid;
    bad[0].claim_end_and_winner =
        bad[0].submit_end + 1U;
    Check(rejected(bad), "Claim outside Submit is rejected");

    bad = valid;
    bad[2].claim_end_and_winner |=
        pa_scheduler::kSharedClaimWinnerBit;
    Check(
        rejected(bad),
        "winner on a role-ineligible Claim is rejected"
    );

    bad = valid;
    bad[1].claim_end_and_winner |=
        pa_scheduler::kSharedClaimWinnerBit;
    Check(
        rejected(bad),
        "winner on a role-eligible but wrong-shard AIC Claim is rejected"
    );
    Check(
        accepted_by(1, bad),
        "winner on the matching AIC Claim shard is accepted"
    );

    bad = valid;
    bad[2].claim_end_and_winner |=
        pa_scheduler::kSharedClaimWinnerBit;
    Check(
        accepted_by(pa_scheduler::kAicWorkers + 2U, bad),
        "winner on the matching AIV-local Claim shard is accepted"
    );
    Check(
        !accepted_by(pa_scheduler::kAicWorkers + 3U, bad),
        "winner on a role-eligible but wrong-shard AIV Claim is rejected"
    );

    bad = valid;
    bad[0].submit_end |=
        pa_scheduler::kSharedClaimWinnerBit;
    Check(
        rejected(bad),
        "winner marker is forbidden in Submit endpoints"
    );

    std::vector<TraceRecord> forbidden{
        MakeRecord(
            TracePhase::Claim, 0, -1,
            base_cycle + 21U, base_cycle + 22U,
            pa_scheduler::kClaimAttempted, 1
        ),
    };
    std::vector<TraceRecord> logical;
    Check(
        !ExpandSharedTraceRecords(
            worker, forbidden.data(), 1,
            valid.data(), plan, &logical
        ),
        "generic stream cannot duplicate dedicated Claim rows"
    );
}

struct TaskTraceBuilder {
    SharedSparseTraceValidator &validator;
    uint64_t tick = 10;
    uint64_t current_submit_begin = 0;
    uint64_t current_claim_begin = 0;
    uint64_t last_efdrain_begin = 0;
    uint64_t last_efdrain_end = 0;

    bool Begin(uint32_t task_id, TaskKind kind, bool winner, bool attempted = true) {
        current_submit_begin = tick;
        current_claim_begin = tick + 2;
        const uint32_t flags =
            (winner ? pa_scheduler::kClaimWon : 0U) |
            (attempted ? pa_scheduler::kClaimAttempted : 0U);
        const bool ok = validator.Observe(
            MakeRecord(
                TracePhase::Claim, static_cast<int32_t>(task_id),
                winner ? FunctionId(kind) : -1,
                current_claim_begin, current_claim_begin + 1,
                flags, IsAlloc(kind)
            )
        );
        tick = current_claim_begin + 1;
        return ok;
    }

    bool FinishWinner(uint32_t task_id, TaskKind kind) {
        const int32_t function_id = FunctionId(kind);
        const uint64_t materialize_begin = tick + 2;
        bool ok = validator.Observe(
            MakeRecord(
                TracePhase::Materialize, static_cast<int32_t>(task_id),
                function_id, materialize_begin, materialize_begin + 3,
                0, IsAlloc(kind)
            )
        );
        ok &= validator.Observe(
            MakeRecord(
                TracePhase::SharedMaterializePublishTaskOutputs,
                static_cast<int32_t>(task_id), function_id,
                materialize_begin + 1, materialize_begin + 3
            )
        );
        ok &= validator.Observe(
            MakeRecord(
                TracePhase::SharedMaterializePublishTaskOutputsCopy,
                static_cast<int32_t>(task_id), function_id,
                materialize_begin + 1, materialize_begin + 2
            )
        );
        ok &= validator.Observe(
            MakeRecord(
                TracePhase::SharedMaterializePublishTaskOutputsFlush,
                static_cast<int32_t>(task_id), function_id,
                materialize_begin + 2, materialize_begin + 3
            )
        );
        uint64_t previous_end = materialize_begin + 3;
        ok &= validator.Observe(
            MakeRecord(
                TracePhase::Register, static_cast<int32_t>(task_id),
                function_id, previous_end, previous_end + 3, 0, 0
            )
        );
        ok &= validator.Observe(
            MakeRecord(
                TracePhase::SharedRegisterPublishMetadata,
                static_cast<int32_t>(task_id), function_id,
                previous_end + 1, previous_end + 2
            )
        );
        previous_end += 3;
        if (kind != TaskKind::Alloc) {
            ok &= validator.Observe(
                MakeRecord(
                    TracePhase::Fanin, static_cast<int32_t>(task_id),
                    function_id, previous_end, previous_end + 2, 0, 3
                )
            );
            previous_end += 2;
        }
        ok &= validator.Observe(
            MakeRecord(
                kind == TaskKind::Alloc
                    ? TracePhase::AllocComplete
                    : TracePhase::WinnerBuild,
                static_cast<int32_t>(task_id), function_id,
                previous_end, previous_end + 4
            )
        );
        previous_end += 4;
        ok &= validator.Observe(
            MakeRecord(
                TracePhase::Submit, static_cast<int32_t>(task_id),
                function_id, current_submit_begin, previous_end + 1,
                pa_scheduler::kClaimWon, IsAlloc(kind)
            )
        );
        last_efdrain_begin = current_submit_begin;
        last_efdrain_end = current_claim_begin;
        tick = previous_end + 1;
        return ok;
    }

    bool FinishLoser(uint32_t task_id, TaskKind kind) {
        // loser 的 Submit 父区间只覆盖轻量返回；它不等待、不读取
        // TensorMap，也不会产生任何 winner-only 子 span。
        const bool ok = validator.Observe(
            MakeRecord(
                TracePhase::Submit, static_cast<int32_t>(task_id), -1,
                current_submit_begin, tick + 1, 0, IsAlloc(kind)
            )
        );
        last_efdrain_begin = current_submit_begin;
        last_efdrain_end = current_claim_begin;
        ++tick;
        return ok;
    }
};

bool OpenAllocWinnerMaterialize(
    SharedSparseTraceValidator &validator,
    uint64_t materialize_begin = 15,
    uint64_t materialize_end = 18
) {
    return validator.Observe(
               MakeRecord(
                   TracePhase::Claim, 0, -1, 12, 13,
                   pa_scheduler::kClaimWon |
                       pa_scheduler::kClaimAttempted,
                   1
               )
           ) &&
           validator.Observe(
               MakeRecord(
                   TracePhase::Materialize, 0, -1,
                   materialize_begin, materialize_end, 0, 1
               )
           );
}

// outputs 包络内固定 copy → flush 两层；copy.end 必须等于 flush.start。
bool ObserveSharedMaterializeOutputNest(
    SharedSparseTraceValidator &validator,
    int32_t task_id,
    int32_t function_id,
    uint64_t outputs_begin,
    uint64_t outputs_end,
    uint64_t copy_end
) {
    return validator.Observe(
               MakeRecord(
                   TracePhase::SharedMaterializePublishTaskOutputs,
                   task_id, function_id, outputs_begin, outputs_end
               )
           ) &&
           validator.Observe(
               MakeRecord(
                   TracePhase::SharedMaterializePublishTaskOutputsCopy,
                   task_id, function_id, outputs_begin, copy_end
               )
           ) &&
           validator.Observe(
               MakeRecord(
                   TracePhase::SharedMaterializePublishTaskOutputsFlush,
                   task_id, function_id, copy_end, outputs_end
               )
           );
}

bool OpenAllocWinnerRegister(
    SharedSparseTraceValidator &validator,
    uint64_t register_begin = 18,
    uint64_t register_end = 24
) {
    return OpenAllocWinnerMaterialize(
               validator, 15, register_begin
           ) &&
           ObserveSharedMaterializeOutputNest(
               validator, 0, -1, 16, register_begin, 17
           ) &&
           validator.Observe(
               MakeRecord(
                   TracePhase::Register, 0, -1,
                   register_begin, register_end
               )
           );
}

void TestAcceptsSparseWinnerAndLoserFlow() {
    SharedSparseTraceValidator validator;
    TaskTraceBuilder trace{validator};
    Check(
        trace.Begin(0, TaskKind::Alloc, true) &&
            trace.FinishWinner(0, TaskKind::Alloc),
        "Alloc winner closes with Materialize/Register/AllocComplete/Submit"
    );
    Check(
        trace.Begin(1, TaskKind::Qk, false) &&
            trace.FinishLoser(1, TaskKind::Qk),
        "attempted QK loser keeps only its Claim/Submit pair"
    );
    Check(
        trace.Begin(2, TaskKind::Sf, false, false) &&
            trace.FinishLoser(2, TaskKind::Sf),
        "role-not-attempted SF loser also keeps only its Claim/Submit pair"
    );
    Check(
        trace.Begin(3, TaskKind::Pv, true) &&
            trace.FinishWinner(3, TaskKind::Pv),
        "ordinary winner includes exactly one Fanin and WinnerBuild"
    );
    Check(
        trace.Begin(4, TaskKind::Up, false) &&
            trace.FinishLoser(4, TaskKind::Up),
        "the final logical task may close through the loser parent"
    );
    Check(validator.Closed(), "mixed sparse flow is fully closed");
    Check(
        validator.EfDrainCount() == 5 &&
            validator.LastEfDrainBegin() == trace.last_efdrain_begin &&
            validator.LastEfDrainEnd() == trace.last_efdrain_end &&
            validator.ClaimCount() == 5 &&
            validator.WinnerCount() == 2 &&
            validator.MaterializeCount() == 2 &&
            validator.FaninCount() == 1 &&
            validator.RegisterCount() == 2 &&
            validator.RegisterMetadataCount() == 2 &&
            validator.MaterializeTaskOutputsCount() == 2 &&
            validator.WinnerTailCount() == 2 &&
            validator.SubmitCount() == 5,
        "sparse flow derives every EfDrain boundary and counts only winner children"
    );
}

void TestRejectsReplayPrefixDrift() {
    {
        SharedSparseTraceValidator validator;
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::EfDrain, 0, -1, 10, 12)
            ),
            "shared sparse raw rejects an explicit EfDrain record"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::Claim, 1, -1, 12, 13)
            ),
            "the first per-core task must be task 0"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::Claim, 0, -1, 13, 12)
            ),
            "Claim rejects an inverted time boundary"
        );
    }
    {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(0, TaskKind::Alloc, false) &&
                trace.FinishLoser(0, TaskKind::Alloc),
            "task 0 loser establishes the sequence-gap test"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::Claim, 2, -1, 20, 21)
            ),
            "a skipped task_id is rejected after a loser"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(
                    TracePhase::Claim, 0, -1, 12, 13,
                    pa_scheduler::kClaimAttempted, 1
                )
            ),
            "valid Claim opens the derived-EfDrain inversion test"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::Submit, 0, -1, 13, 14, 0, 1
                )
            ),
            "Submit.start cannot be later than Claim.start"
        );
    }
}

void TestRejectsMissingSubmit() {
    {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(0, TaskKind::Alloc, false),
            "loser Claim opens the missing-Submit test"
        );
        Check(
            !validator.Closed() &&
                !validator.Observe(
                    MakeRecord(
                        TracePhase::Claim, 1, -1, 20, 21,
                        pa_scheduler::kClaimAttempted, 0
                    )
                ),
            "loser must close Submit before the next task Claim"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerRegister(validator) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::SharedRegisterPublishMetadata,
                        0, -1, 20, 22
                    )
                ) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::AllocComplete, 0, -1, 24, 28
                    )
                ),
            "winner reaches the state that only permits Submit"
        );
        Check(
            !validator.Closed() &&
                !validator.Observe(
                    MakeRecord(TracePhase::Claim, 1, -1, 30, 31)
                ),
            "winner must close Submit before the next task Claim"
        );
    }
}

void TestRejectsEveryLoserOnlyForbiddenPhase() {
    constexpr TracePhase forbidden[] = {
        TracePhase::Materialize,
        TracePhase::PrepareMap,
        TracePhase::Fanin,
        TracePhase::Register,
        TracePhase::SharedRegisterPublishMetadata,
        TracePhase::SharedMaterializePublishTaskOutputs,
        TracePhase::SharedMaterializePublishTaskOutputsCopy,
        TracePhase::SharedMaterializePublishTaskOutputsFlush,
        TracePhase::WinnerBuild,
        TracePhase::AllocComplete,
    };
    for (TracePhase phase : forbidden) {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(0, TaskKind::Alloc, false),
            "loser prefix is accepted before a forbidden phase"
        );
        Check(
            !validator.Observe(
                MakeRecord(phase, 0, -1, 13, 14)
            ),
            "loser rejects every winner-only phase"
        );
    }
    {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(0, TaskKind::Alloc, false) &&
                trace.FinishLoser(0, TaskKind::Alloc),
            "loser closes through exactly one Submit parent"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::Submit, 0, -1, 10, 15, 0, 1)
            ),
            "duplicate loser Submit is rejected"
        );
    }
}

void TestRejectsPrepareMapForWinnerAndOutsideSubmit() {
    {
        SharedSparseTraceValidator validator;
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::PrepareMap, -1, -1, 1, 1)
            ),
            "shared rejects PrepareMap even outside a task flow"
        );
    }
    {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(0, TaskKind::Alloc, true),
            "Alloc winner opens the PrepareMap rejection test"
        );
        Check(
            validator.Observe(
                MakeRecord(
                    TracePhase::Materialize, 0, -1, 15, 18, 0, 1
                )
            ),
            "winner Materialize is accepted before the forbidden marker"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::PrepareMap, 0, -1, 18, 18)
            ),
            "shared winner cannot carry a zero-duration PrepareMap marker"
        );
    }
}

void TestRejectsWinnerShapeDrift() {
    {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(0, TaskKind::Alloc, true),
            "Alloc winner opens the Fanin rejection test"
        );
        Check(
            validator.Observe(
                MakeRecord(
                    TracePhase::Materialize, 0, -1, 15, 18, 0, 1
                )
            ),
            "Alloc Materialize is accepted"
        );
        Check(
            ObserveSharedMaterializeOutputNest(
                validator, 0, -1, 16, 18, 17
            ) &&
                validator.Observe(
                MakeRecord(
                    TracePhase::Register, 0, -1, 18, 20
                )
            ) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::SharedRegisterPublishMetadata,
                        0, -1, 19, 19
                    )
                ),
            "Alloc Register details are accepted before the tail"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::Fanin, 0, -1, 20, 21)
            ),
            "Alloc winner cannot emit Fanin"
        );
    }
    {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(0, TaskKind::Alloc, false) &&
                trace.FinishLoser(0, TaskKind::Alloc) &&
                trace.Begin(1, TaskKind::Qk, true),
            "ordinary winner opens the missing-Fanin test"
        );
        Check(
            validator.Observe(
                MakeRecord(
                    TracePhase::Materialize, 1, 0, 18, 21
                )
            ),
            "ordinary Materialize is accepted"
        );
        Check(
            ObserveSharedMaterializeOutputNest(
                validator, 1, 0, 19, 21, 20
            ) &&
                validator.Observe(
                MakeRecord(TracePhase::Register, 1, 0, 21, 24, 0, 1)
            ) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::SharedRegisterPublishMetadata,
                        1, 0, 22, 23
                    )
                ),
            "ordinary Register details precede Fanin"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::WinnerBuild, 1, 0, 24, 25)
            ),
            "ordinary winner cannot skip Fanin after Register"
        );
    }
    {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(0, TaskKind::Alloc, true),
            "Alloc winner opens the wrong-tail test"
        );
        Check(
            validator.Observe(
                MakeRecord(
                    TracePhase::Materialize, 0, -1, 15, 18, 0, 1
                )
            ) &&
                ObserveSharedMaterializeOutputNest(
                    validator, 0, -1, 16, 18, 17
                ) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::Register, 0, -1, 18, 20
                    )
                ) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::SharedRegisterPublishMetadata,
                        0, -1, 19, 19
                    )
                ),
            "Alloc winner reaches its tail"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::WinnerBuild, 0, -1, 20, 24)
            ),
            "Alloc winner requires AllocComplete rather than WinnerBuild"
        );
    }
    {
        SharedSparseTraceValidator validator;
        TaskTraceBuilder trace{validator};
        Check(
            trace.Begin(0, TaskKind::Alloc, true),
            "incomplete winner opens the closure test"
        );
        Check(
            !validator.Closed(),
            "winner cannot close before all winner-only phases and Submit"
        );
    }
}

void TestMaterializeOutputDetailContract() {
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerMaterialize(validator),
            "Materialize parent opens the missing-output test"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::Register, 0, -1, 18, 24)
            ),
            "winner cannot enter Register before output publication closes"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerMaterialize(validator) &&
                ObserveSharedMaterializeOutputNest(
                    validator, 0, -1, 16, 18, 17
                ),
            "one output nest contained by Materialize is accepted"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::SharedMaterializePublishTaskOutputs,
                    0, -1, 16, 18
                )
            ),
            "duplicate Materialize output detail is rejected"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerMaterialize(validator) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::SharedMaterializePublishTaskOutputs,
                        0, -1, 16, 18
                    )
                ),
            "output parent opens the missing-copy test"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::Register, 0, -1, 18, 24)
            ),
            "winner cannot omit output copy detail"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerMaterialize(validator) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::SharedMaterializePublishTaskOutputs,
                        0, -1, 16, 18
                    )
                ) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::SharedMaterializePublishTaskOutputsCopy,
                        0, -1, 16, 17
                    )
                ),
            "copy detail opens the missing-flush test"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::Register, 0, -1, 18, 24)
            ),
            "winner cannot omit output flush detail"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerMaterialize(validator) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::SharedMaterializePublishTaskOutputs,
                        0, -1, 16, 18
                    )
                ) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::SharedMaterializePublishTaskOutputsCopy,
                        0, -1, 16, 17
                    )
                ),
            "copy detail opens the flush-adjacency test"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::SharedMaterializePublishTaskOutputsFlush,
                    0, -1, 18, 18
                )
            ),
            "flush must start exactly at copy end"
        );
    }
    for (int variant = 0; variant < 4; ++variant) {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerMaterialize(validator),
            "Materialize opens output containment/identity test"
        );
        TraceRecord output = MakeRecord(
            TracePhase::SharedMaterializePublishTaskOutputs,
            0, -1, 16, 18
        );
        if (variant == 0) output.start_cycle = 14;
        if (variant == 1) output.task_id = 1;
        if (variant == 2) output.function_id = 0;
        if (variant == 3) {
            output.flags = 1;
            output.auxiliary = 1;
        }
        Check(
            !validator.Observe(output),
            "Materialize output detail rejects bad boundary, identity, or payload"
        );
    }
}

void TestRegisterMetadataDetailContract() {
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerRegister(validator),
            "Register parent opens the missing-detail test"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::AllocComplete, 0, -1, 24, 28)
            ) &&
                !validator.Closed(),
            "winner cannot omit its Register metadata detail"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerRegister(validator) &&
                validator.Observe(
                    MakeRecord(
                        TracePhase::SharedRegisterPublishMetadata,
                        0, -1, 20, 22
                    )
                ),
            "one contained Register metadata detail is accepted"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::SharedRegisterPublishMetadata,
                    0, -1, 20, 22
                )
            ),
            "duplicate Register metadata detail is rejected"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerRegister(validator),
            "Register parent opens the early-boundary test"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::SharedRegisterPublishMetadata,
                    0, -1, 17, 20
                )
            ),
            "Register metadata cannot begin before its parent"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerRegister(validator),
            "Register parent opens the late-boundary test"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::SharedRegisterPublishMetadata,
                    0, -1, 20, 25
                )
            ),
            "Register metadata cannot end after its parent"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerRegister(validator),
            "Register parent opens the wrong-task test"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::SharedRegisterPublishMetadata,
                    1, -1, 20, 22
                )
            ),
            "Register metadata must keep the parent task identity"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerRegister(validator),
            "Register parent opens the wrong-function test"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::SharedRegisterPublishMetadata,
                    0, 0, 20, 22
                )
            ),
            "Register metadata must keep the parent function identity"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            OpenAllocWinnerRegister(validator),
            "Register parent opens the payload-shape test"
        );
        Check(
            !validator.Observe(
                MakeRecord(
                    TracePhase::SharedRegisterPublishMetadata,
                    0, -1, 20, 22, 1, 1
                )
            ),
            "Register metadata detail requires zero flags and auxiliary"
        );
    }
}

void TestSharedInsertTurnAtomicSchema() {
    TraceRecord poll = MakeRecord(
        TracePhase::Atomic, -1, -1, 100, 200,
        static_cast<uint32_t>(AtomicOp::Load) |
            kAtomicResultUsed | kAtomicPollBatch |
            kAtomicReturnReady |
            (17U << kAtomicPollCountShift),
        static_cast<uint32_t>(
            AtomicSite::SharedInsertTurnPoll
        )
    );
    Check(
        AtomicRecordSchemaValid(poll, true),
        "shared insert-turn aggregate PollBatch schema is accepted"
    );
    TraceRecord direct_poll = poll;
    direct_poll.flags =
        static_cast<uint32_t>(AtomicOp::Load) |
        kAtomicResultUsed | kAtomicReturnReady;
    direct_poll.task_id = 3;
    Check(
        !AtomicRecordSchemaValid(direct_poll, true),
        "shared insert-turn poll cannot masquerade as a direct Load"
    );

    TraceRecord handoff = MakeRecord(
        TracePhase::Atomic, 3, -1, 200, 230,
        static_cast<uint32_t>(
            AtomicOp::CompareExchange
        ) |
            kAtomicResultUsed | kAtomicReturnReady,
        static_cast<uint32_t>(
            AtomicSite::SharedInsertTurnHandoff
        )
    );
    Check(
        AtomicRecordSchemaValid(handoff, true),
        "shared insert-turn handoff CAS schema is accepted"
    );
    TraceRecord anonymous_handoff = handoff;
    anonymous_handoff.task_id = -1;
    Check(
        !AtomicRecordSchemaValid(anonymous_handoff, true),
        "handoff CAS requires its shared winner task identity"
    );
    TraceRecord wrong_handoff_op = handoff;
    wrong_handoff_op.flags =
        (wrong_handoff_op.flags & ~kAtomicOpMask) |
        static_cast<uint32_t>(AtomicOp::Exchange);
    Check(
        !AtomicRecordSchemaValid(wrong_handoff_op, true),
        "handoff site rejects a non-CAS atomic op"
    );
}

void TestPlanClosesAllLogicalTasks() {
    // SchedulerState 保留真实 DistGlobal 约 1 GiB ABI，不能放在线程栈上。
    // 静态零初始化只映射本用例实际触碰的 config/context_lens 页面。
    static pa_scheduler::SchedulerState state{};
    state.config.batches = 2;
    state.context_lens[0] = 0;
    state.context_lens[1] = 0;
    pa_scheduler::host::SharedHostTaskPlan plan;
    Check(
        pa_scheduler::host::BuildSharedHostTaskPlan(state, &plan),
        "two-batch zero-context host plan is valid"
    );
    Check(
        plan.total_tasks == 2 &&
            plan.TaskAt(0)->kind == TaskKind::Alloc &&
            plan.TaskAt(1)->kind == TaskKind::Alloc,
        "authoritative plan contains one Alloc per empty batch"
    );

    SharedSparseTraceValidator validator(&plan);
    TaskTraceBuilder trace{validator};
    Check(
        trace.Begin(0, TaskKind::Alloc, true) &&
            trace.FinishWinner(0, TaskKind::Alloc),
        "first planned Alloc winner closes"
    );
    Check(
        !validator.Closed(),
        "plan-aware validator rejects a truncated per-core replay"
    );
    Check(
        trace.Begin(1, TaskKind::Alloc, false) &&
            trace.FinishLoser(1, TaskKind::Alloc),
        "second planned Alloc loser closes through its Submit parent"
    );
    Check(
        validator.Closed(),
        "plan-aware validator closes only after every logical task"
    );
}

}  // namespace

int main() {
    TestTraceBinaryLayoutAndHeaderGate();
    TestSharedCompactReconstruction();
    TestSharedCompactStageOnlyReconstruction();
    TestSharedCompactGenericMergeOrder();
    TestRejectsBadSharedCompactRecords();
    TestAcceptsSparseWinnerAndLoserFlow();
    TestRejectsReplayPrefixDrift();
    TestRejectsMissingSubmit();
    TestRejectsEveryLoserOnlyForbiddenPhase();
    TestRejectsPrepareMapForWinnerAndOutsideSubmit();
    TestRejectsWinnerShapeDrift();
    TestMaterializeOutputDetailContract();
    TestRegisterMetadataDetailContract();
    TestSharedInsertTurnAtomicSchema();
    TestPlanClosesAllLogicalTasks();
    if (g_failures != 0) {
        std::fprintf(
            stderr, "[FAIL] shared sparse trace tests: %d\n", g_failures
        );
        return 1;
    }
    std::printf("[PASS] shared sparse trace tests\n");
    return 0;
}
