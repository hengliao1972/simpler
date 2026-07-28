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

#include "host_support.h"

namespace {

using pa_scheduler::TaskKind;
using pa_scheduler::TracePhase;
using pa_scheduler::TraceRecord;
using pa_scheduler::AtomicOp;
using pa_scheduler::AtomicSite;
using pa_scheduler::kAtomicOpMask;
using pa_scheduler::kAtomicPollBatch;
using pa_scheduler::kAtomicPollCountShift;
using pa_scheduler::kAtomicResultUsed;
using pa_scheduler::kAtomicReturnReady;
using pa_scheduler::host::AtomicRecordSchemaValid;
using pa_scheduler::host::SharedSparseTraceValidator;

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

int32_t FunctionId(TaskKind kind) {
    return kind == TaskKind::Alloc
        ? -1
        : static_cast<int32_t>(static_cast<uint32_t>(kind) - 1U);
}

uint32_t IsAlloc(TaskKind kind) {
    return kind == TaskKind::Alloc ? 1U : 0U;
}

struct TaskTraceBuilder {
    SharedSparseTraceValidator &validator;
    uint64_t tick = 10;

    bool Begin(uint32_t task_id, TaskKind kind, bool winner, bool attempted = true) {
        const uint64_t submit_begin = tick;
        const uint64_t efdrain_end = tick + 2;
        bool ok = validator.Observe(
            MakeRecord(
                TracePhase::EfDrain, static_cast<int32_t>(task_id), -1,
                submit_begin, efdrain_end
            )
        );
        const uint32_t flags =
            (winner ? pa_scheduler::kClaimWon : 0U) |
            (attempted ? pa_scheduler::kClaimAttempted : 0U);
        ok &= validator.Observe(
            MakeRecord(
                TracePhase::Claim, static_cast<int32_t>(task_id),
                winner ? FunctionId(kind) : -1,
                efdrain_end, efdrain_end + 1, flags, IsAlloc(kind)
            )
        );
        tick = efdrain_end + 1;
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
                function_id, tick - 3, previous_end + 1,
                pa_scheduler::kClaimWon, IsAlloc(kind)
            )
        );
        tick = previous_end + 1;
        return ok;
    }

    bool FinishLoser(uint32_t task_id, TaskKind kind) {
        // loser 的 Submit 父区间只覆盖轻量返回；它不等待、不读取
        // TensorMap，也不会产生任何 winner-only 子 span。
        const bool ok = validator.Observe(
            MakeRecord(
                TracePhase::Submit, static_cast<int32_t>(task_id), -1,
                tick - 3, tick + 1, 0, IsAlloc(kind)
            )
        );
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
               MakeRecord(TracePhase::EfDrain, 0, -1, 10, 12)
           ) &&
           validator.Observe(
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
        "attempted QK loser keeps only its Submit parent"
    );
    Check(
        trace.Begin(2, TaskKind::Sf, false, false) &&
            trace.FinishLoser(2, TaskKind::Sf),
        "role-not-attempted SF loser also keeps only its Submit parent"
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
            validator.ClaimCount() == 5 &&
            validator.WinnerCount() == 2 &&
            validator.MaterializeCount() == 2 &&
            validator.FaninCount() == 1 &&
            validator.RegisterCount() == 2 &&
            validator.RegisterMetadataCount() == 2 &&
            validator.MaterializeTaskOutputsCount() == 2 &&
            validator.WinnerTailCount() == 2 &&
            validator.SubmitCount() == 5,
        "sparse flow counts every Submit parent but only winner-only children"
    );
}

void TestRejectsReplayPrefixDrift() {
    {
        SharedSparseTraceValidator validator;
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::Claim, 0, -1, 10, 11)
            ),
            "Claim cannot appear before the task EfDrain"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::EfDrain, 1, -1, 10, 12)
            ),
            "the first per-core task must be task 0"
        );
    }
    {
        SharedSparseTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::EfDrain, 0, -1, 10, 12)
            ),
            "EfDrain opens the non-contiguous Claim test"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::Claim, 0, -1, 13, 14)
            ),
            "Claim must start exactly at EfDrain.end"
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
                MakeRecord(TracePhase::EfDrain, 2, -1, 20, 22)
            ),
            "a skipped task_id is rejected after a loser"
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
    TestAcceptsSparseWinnerAndLoserFlow();
    TestRejectsReplayPrefixDrift();
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
