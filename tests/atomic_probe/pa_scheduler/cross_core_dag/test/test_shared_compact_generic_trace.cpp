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
#include <cstring>

#include "host_support.h"

namespace {

using pa_scheduler::CompactTraceRecord16;
using pa_scheduler::DcciSite;
using pa_scheduler::TraceHeader;
using pa_scheduler::TracePhase;
using pa_scheduler::TraceRecord;
using pa_scheduler::host::DecodeCompactTraceRecord;
using pa_scheduler::host::EncodeCompactTraceRecord;
using pa_scheduler::host::InitializeTraceHeader;
using pa_scheduler::host::ValidateTraceHeader;

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) return;
    std::fprintf(
        stderr, "[FAIL] shared compact generic trace: %s\n",
        message
    );
    ++g_failures;
}

bool SameRecord(
    const TraceRecord &left, const TraceRecord &right
) {
    return
        left.start_cycle == right.start_cycle &&
        left.end_cycle == right.end_cycle &&
        left.task_id == right.task_id &&
        left.function_id == right.function_id &&
        left.flags == right.flags &&
        left.phase == right.phase &&
        left.auxiliary == right.auxiliary;
}

TraceRecord MakeRecord(
    uint64_t start_cycle, uint64_t end_cycle,
    int32_t task_id, int32_t function_id,
    TracePhase phase, uint32_t flags, uint32_t auxiliary
) {
    TraceRecord record{};
    record.start_cycle = start_cycle;
    record.end_cycle = end_cycle;
    record.task_id = task_id;
    record.function_id = function_id;
    record.flags = flags;
    record.phase = static_cast<uint16_t>(phase);
    record.auxiliary = static_cast<uint16_t>(auxiliary);
    return record;
}

void TestLayoutAndHeaderIdentity() {
    Check(
        sizeof(TraceRecord) == 32 &&
            sizeof(CompactTraceRecord16) == 16 &&
            alignof(CompactTraceRecord16) == 16,
        "logical/physical record sizes must remain 32/16 bytes"
    );
    Check(
        pa_scheduler::kTraceRecordSizeBytes == 16 &&
            pa_scheduler::kTraceRecordsPerCore == 28416 &&
            pa_scheduler::kTraceWorkerBytes == 593920,
        "compact build must keep capacity and use the exact short stride"
    );
    Check(
        pa_scheduler::TraceSubmitClaimOffset(0) % 64U == 0 &&
            pa_scheduler::TraceRecordsOffset(0) % 64U == 0 &&
            pa_scheduler::TraceWorkerOffset(1) -
                    pa_scheduler::TraceWorkerOffset(0) ==
                593920U &&
            pa_scheduler::kTraceBytes ==
                sizeof(TraceHeader) +
                    96U * 593920U,
        "all compact worker regions must remain cache-line isolated"
    );

    TraceHeader header{};
    InitializeTraceHeader(&header);
    Check(
        header.record_size_bytes == 16 &&
            ValidateTraceHeader(
                header, "compact trace ABI self-test"
            ),
        "initialized header must accept the 16-byte physical ABI"
    );
    header.record_size_bytes = 32;
    Check(
        !ValidateTraceHeader(
            header, "compact trace ABI negative self-test"
        ),
        "compact header gate must reject a 32-byte mixed artifact"
    );
}

void TestRoundTripAndClockWrap() {
    const uint64_t anchor = UINT64_C(0x00000001fffffff0);
    const uint64_t finish = anchor + 0x100U;
    const TraceRecord source = MakeRecord(
        anchor + 0x20U, anchor + 0x50U,
        static_cast<int32_t>(pa_scheduler::kMaxTasks - 1U),
        3, TracePhase::Materialize, UINT32_MAX,
        pa_scheduler::kCompactTraceAuxiliaryMask
    );
    CompactTraceRecord16 compact{};
    TraceRecord decoded{};
    Check(
        EncodeCompactTraceRecord(source, &compact),
        "maximum legal fields must encode"
    );
    Check(
        compact.start_cycle_low == 0x10U &&
            compact.end_cycle_low == 0x40U,
        "forward timestamps must retain their wrapped low bits"
    );
    Check(
        DecodeCompactTraceRecord(
            compact, anchor, finish, &decoded
        ) &&
            SameRecord(source, decoded),
        "maximum legal fields and a forward wrap must round-trip"
    );

    const TraceRecord sentinel = MakeRecord(
        anchor + 1U, anchor + 2U, -1, -1,
        TracePhase::Atomic, 0xfedcba98U,
        static_cast<uint32_t>(
            pa_scheduler::AtomicSite::SharedOutputRollbackExchange
        )
    );
    Check(
        EncodeCompactTraceRecord(sentinel, &compact) &&
            DecodeCompactTraceRecord(
                compact, anchor, finish, &decoded
            ) &&
            SameRecord(sentinel, decoded),
        "-1 task/function sentinels and full flags must round-trip"
    );
}

void TestStartupBackwardWrap() {
    const uint64_t anchor = UINT64_C(0x0000000200000010);
    const uint64_t finish = anchor + 0x100U;
    const TraceRecord source = MakeRecord(
        anchor - 0x30U, anchor - 0x20U, -1, -1,
        TracePhase::Dcci, 0x30cU,
        static_cast<uint32_t>(
            DcciSite::StartupConfigInvalidate
        )
    );
    CompactTraceRecord16 compact{};
    TraceRecord decoded{};
    Check(
        EncodeCompactTraceRecord(source, &compact) &&
            compact.start_cycle_low == 0xffffffe0U &&
            compact.end_cycle_low == 0xfffffff0U &&
            DecodeCompactTraceRecord(
                compact, anchor, finish, &decoded
            ) &&
            SameRecord(source, decoded),
        "startup DCCI must unfold backward across the low32 wrap"
    );
}

void TestDecodeRejections() {
    const uint64_t anchor = UINT64_C(0x0000000300000100);
    const uint64_t finish = anchor + 0x100U;
    TraceRecord source = MakeRecord(
        anchor + 0x10U, anchor + 0x20U, 7, 2,
        TracePhase::Register, 0x12345678U, 3
    );
    CompactTraceRecord16 compact{};
    TraceRecord decoded{};
    Check(
        EncodeCompactTraceRecord(source, &compact),
        "valid rejection-test seed must encode"
    );
    Check(
        !DecodeCompactTraceRecord(compact, 0, finish, &decoded) &&
            !DecodeCompactTraceRecord(
                compact, anchor, anchor - 1U, &decoded
            ) &&
            !DecodeCompactTraceRecord(
                compact, anchor,
                anchor + (UINT64_C(1) << 32U), &decoded
            ),
        "zero anchor, reversed lifecycle, and a 2^32 window must fail"
    );

    CompactTraceRecord16 invalid = compact;
    invalid.packed =
        (invalid.packed & ~pa_scheduler::kCompactTraceTaskMask) |
        pa_scheduler::kMaxTasks;
    Check(
        !DecodeCompactTraceRecord(
            invalid, anchor, finish, &decoded
        ),
        "reserved task codes must fail"
    );
    invalid = compact;
    invalid.packed &=
        ~(pa_scheduler::kCompactTraceFunctionMask
          << pa_scheduler::kCompactTraceFunctionShift);
    invalid.packed |=
        4U << pa_scheduler::kCompactTraceFunctionShift;
    Check(
        !DecodeCompactTraceRecord(
            invalid, anchor, finish, &decoded
        ),
        "reserved function codes must fail"
    );
    invalid = compact;
    invalid.packed &=
        ~(pa_scheduler::kCompactTracePhaseMask
          << pa_scheduler::kCompactTracePhaseShift);
    invalid.packed |=
        static_cast<uint32_t>(TracePhase::Count)
        << pa_scheduler::kCompactTracePhaseShift;
    Check(
        !DecodeCompactTraceRecord(
            invalid, anchor, finish, &decoded
        ),
        "reserved phase codes must fail"
    );

    invalid = compact;
    invalid.start_cycle_low =
        static_cast<uint32_t>(anchor + 0x30U);
    invalid.end_cycle_low =
        static_cast<uint32_t>(anchor + 0x20U);
    Check(
        !DecodeCompactTraceRecord(
            invalid, anchor, finish, &decoded
        ),
        "a reversed decoded interval must fail"
    );
    invalid = compact;
    invalid.end_cycle_low =
        static_cast<uint32_t>(finish + 1U);
    Check(
        !DecodeCompactTraceRecord(
            invalid, anchor, finish, &decoded
        ),
        "an endpoint outside the worker lifecycle must fail"
    );
}

void TestTerminalEndStoreIsolation() {
    alignas(64) CompactTraceRecord16 records[4]{};
    for (uint32_t index = 0; index < 4; ++index) {
        records[index].start_cycle_low = 0x1000U + index;
        records[index].end_cycle_low = 0x2000U + index;
        records[index].flags = 0x3000U + index;
        records[index].packed = 0x4000U + index;
    }
    unsigned char before[sizeof(records)]{};
    std::memcpy(before, records, sizeof(records));
    records[2].end_cycle_low = 0xdeadbeefU;

    const auto *after =
        reinterpret_cast<const unsigned char *>(records);
    const size_t changed_begin =
        2U * sizeof(CompactTraceRecord16) +
        offsetof(CompactTraceRecord16, end_cycle_low);
    bool isolated = true;
    for (size_t byte = 0; byte < sizeof(records); ++byte) {
        const bool should_change =
            byte >= changed_begin &&
            byte < changed_begin + sizeof(uint32_t);
        isolated &= should_change
            ? before[byte] != after[byte]
            : before[byte] == after[byte];
    }
    Check(
        isolated,
        "terminal 32-bit end publication must not alter neighbors"
    );
}

}  // namespace

int main() {
    TestLayoutAndHeaderIdentity();
    TestRoundTripAndClockWrap();
    TestStartupBackwardWrap();
    TestDecodeRejections();
    TestTerminalEndStoreIsolation();
    if (g_failures != 0) {
        std::fprintf(
            stderr,
            "[FAIL] shared compact generic trace: %d failure(s)\n",
            g_failures
        );
        return 1;
    }
    std::puts(
        "[PASS] shared compact generic trace codec and clock gates"
    );
    return 0;
}
