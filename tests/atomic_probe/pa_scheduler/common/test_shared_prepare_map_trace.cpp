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

using pa_scheduler::TracePhase;
using pa_scheduler::TraceRecord;
using pa_scheduler::host::SharedPrepareMapTraceValidator;

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) return;
    std::fprintf(stderr, "[FAIL] shared PrepareMap raw marker: %s\n", message);
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

void TestValidSequentialFlow() {
    SharedPrepareMapTraceValidator validator;
    Check(
        validator.Observe(
            MakeRecord(TracePhase::Claim, 0, -1, 10, 12, 3, 1)
        ),
        "task 0 Claim opens the first Submit"
    );
    Check(
        validator.Observe(
            MakeRecord(TracePhase::Materialize, 0, -1, 12, 14, 0, 1)
        ),
        "matching task 0 Materialize is accepted"
    );
    Check(
        validator.Observe(
            MakeRecord(TracePhase::PrepareMap, 0, -1, 14, 14, 0, 1)
        ),
        "marker is anchored to Alloc Materialize.end"
    );
    Check(
        validator.Observe(
            MakeRecord(TracePhase::Submit, 0, -1, 10, 18, 1, 1)
        ),
        "matching task 0 Submit closes the first flow"
    );

    Check(
        validator.Observe(
            MakeRecord(TracePhase::Claim, 1, 0, 20, 21, 3)
        ),
        "task 1 Claim follows task 0"
    );
    Check(
        validator.Observe(
            MakeRecord(TracePhase::Materialize, 1, 0, 21, 24)
        ),
        "matching task 1 Materialize is accepted"
    );
    Check(
        validator.Observe(
            MakeRecord(TracePhase::PrepareMap, 1, 0, 24, 24)
        ),
        "non-Alloc marker is anchored with auxiliary zero"
    );
    Check(
        validator.Observe(
            MakeRecord(TracePhase::Submit, 1, 0, 20, 27, 1)
        ),
        "matching task 1 Submit closes the second flow"
    );
    Check(validator.Closed(), "two sequential Submit flows close");
    Check(
        validator.MaterializeCount() == 2 &&
            validator.MarkerCount() == 2 &&
            validator.SubmitCount() == 2,
        "valid flow counts Materialize, marker and Submit exactly"
    );
}

void TestRejectsTaskSequenceDrift() {
    {
        SharedPrepareMapTraceValidator validator;
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::Claim, 1, 0, 10, 11)
            ),
            "the first per-core Claim must be task 0"
        );
    }
    {
        SharedPrepareMapTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Claim, 0, -1, 10, 11)
            ),
            "task 0 opens the sequence-gap test"
        );
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Materialize, 0, -1, 11, 12)
            ),
            "task 0 Materialize is accepted"
        );
        Check(
            validator.Observe(
                MakeRecord(TracePhase::PrepareMap, 0, -1, 12, 12, 0, 1)
            ),
            "task 0 marker is accepted"
        );
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Submit, 0, -1, 10, 13)
            ),
            "task 0 closes before the sequence gap"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::Claim, 2, 1, 14, 15)
            ),
            "a skipped task_id is rejected"
        );
    }
}

void TestRejectsMaterializeMismatch() {
    {
        SharedPrepareMapTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Claim, 0, -1, 20, 21)
            ),
            "Claim opens the missing-Materialize test"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::PrepareMap, 0, -1, 21, 21, 0, 1)
            ),
            "marker cannot appear before Materialize"
        );
    }
    {
        SharedPrepareMapTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Claim, 0, -1, 30, 31)
            ),
            "Claim opens the duplicate-Materialize test"
        );
        const TraceRecord materialize =
            MakeRecord(TracePhase::Materialize, 0, -1, 31, 33, 0, 1);
        Check(
            validator.Observe(materialize),
            "first Materialize is accepted"
        );
        Check(
            !validator.Observe(materialize),
            "duplicate Materialize is rejected"
        );
    }
    {
        SharedPrepareMapTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Claim, 0, -1, 40, 41)
            ),
            "Claim opens the Materialize identity test"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::Materialize, 0, 0, 41, 43)
            ),
            "Materialize function must match Claim"
        );
    }
}

void TestRejectsMarkerTimingDrift() {
    {
        SharedPrepareMapTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Claim, 0, -1, 50, 51)
            ),
            "Claim opens the marker-anchor test"
        );
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Materialize, 0, -1, 51, 54)
            ),
            "Materialize establishes the marker anchor"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::PrepareMap, 0, -1, 53, 53, 0, 1)
            ),
            "zero-duration marker must equal Materialize.end"
        );
    }
    {
        SharedPrepareMapTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Claim, 0, -1, 60, 61)
            ),
            "Claim opens the non-zero marker test"
        );
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Materialize, 0, -1, 61, 64)
            ),
            "Materialize establishes the non-zero marker anchor"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::PrepareMap, 0, -1, 64, 65, 0, 1)
            ),
            "non-zero PrepareMap duration is rejected"
        );
    }
}

void TestRejectsIdentitySchemaAndMultiplicityDrift() {
    {
        SharedPrepareMapTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Claim, 0, -1, 70, 71)
            ),
            "Claim opens the marker-function test"
        );
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Materialize, 0, -1, 71, 72)
            ),
            "Materialize precedes the marker-function test"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::PrepareMap, 0, 0, 72, 72, 0, 1)
            ),
            "marker function must match Claim"
        );
    }
    {
        SharedPrepareMapTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Claim, 0, -1, 80, 81)
            ),
            "Claim opens the flags test"
        );
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Materialize, 0, -1, 81, 82)
            ),
            "Materialize precedes the flags test"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::PrepareMap, 0, -1, 82, 82, 1, 1)
            ),
            "marker flags must stay zero"
        );
    }
    {
        SharedPrepareMapTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Claim, 0, -1, 90, 91)
            ),
            "Claim opens the auxiliary test"
        );
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Materialize, 0, -1, 91, 92)
            ),
            "Materialize precedes the auxiliary test"
        );
        Check(
            !validator.Observe(
                MakeRecord(TracePhase::PrepareMap, 0, -1, 92, 92)
            ),
            "Alloc marker must set auxiliary=1"
        );
    }
    {
        SharedPrepareMapTraceValidator validator;
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Claim, 0, -1, 100, 101)
            ),
            "Claim opens the duplicate-marker test"
        );
        Check(
            validator.Observe(
                MakeRecord(TracePhase::Materialize, 0, -1, 101, 102)
            ),
            "Materialize precedes the duplicate-marker test"
        );
        const TraceRecord marker =
            MakeRecord(TracePhase::PrepareMap, 0, -1, 102, 102, 0, 1);
        Check(validator.Observe(marker), "first marker is accepted");
        Check(!validator.Observe(marker), "duplicate marker is rejected");
    }
}

void TestRejectsSubmitWithoutCompleteMarkerFlow() {
    SharedPrepareMapTraceValidator validator;
    Check(
        validator.Observe(
            MakeRecord(TracePhase::Claim, 0, -1, 110, 111)
        ),
        "Claim opens the incomplete-flow test"
    );
    Check(
        validator.Observe(
            MakeRecord(TracePhase::Materialize, 0, -1, 111, 112)
        ),
        "Materialize is accepted in the incomplete-flow test"
    );
    Check(
        !validator.Observe(
            MakeRecord(TracePhase::Submit, 0, -1, 110, 114)
        ),
        "Submit cannot close without PrepareMap"
    );
}

}  // namespace

int main() {
    TestValidSequentialFlow();
    TestRejectsTaskSequenceDrift();
    TestRejectsMaterializeMismatch();
    TestRejectsMarkerTimingDrift();
    TestRejectsIdentitySchemaAndMultiplicityDrift();
    TestRejectsSubmitWithoutCompleteMarkerFlow();
    if (g_failures != 0) {
        std::fprintf(
            stderr,
            "[FAIL] shared PrepareMap raw marker tests: %d\n",
            g_failures
        );
        return 1;
    }
    std::printf("[PASS] shared PrepareMap raw marker tests\n");
    return 0;
}
