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

#ifndef PA_AICPU_PLAN_SIMT_CCEC_RUNTIME_REPORT_H
#define PA_AICPU_PLAN_SIMT_CCEC_RUNTIME_REPORT_H

#include <cstddef>
#include <cstdint>

namespace pa_scheduler::aicpu_plan_simt::runtime_gate {

// Report 是独立验证 ABI，不是调度状态。每个 writer 独占一个 A5 128B
// 冲突单元，避免四个 leader 的诊断写互相形成 cache-line sharing，也避免
// report 与 runtime_plan_control 的返回型原子落在同一冲突单元。
constexpr uint32_t kReportLineBytes = 128U;
constexpr uint32_t kReportWords = kReportLineBytes / sizeof(uint64_t);
constexpr uint32_t kLeaderCount = 4U;
constexpr uint64_t kLaunchMagic = UINT64_C(0x53494D544C41554E);  // "SIMTLAUN"
constexpr uint64_t kLeaderMagic = UINT64_C(0x53494D544C454144);  // "SIMTLEAD"
constexpr uint64_t kContinuationMagic =
    UINT64_C(0x5343414C434F4E54);  // "SCALCONT"

struct alignas(kReportLineBytes) ReportLine {
    uint64_t words[kReportWords];
};

struct alignas(kReportLineBytes) RuntimeReport {
    ReportLine launch;
    ReportLine leaders[kLeaderCount];
    ReportLine continuation;
};

enum LaunchWord : uint32_t {
    LaunchMagic = 0U,
    LaunchBeginClock = 1U,
    LaunchJoinClock = 2U,
    LaunchClosedTaskCount = 3U,
    LaunchValidation = 4U,
    LaunchTicketsObserved = 5U,
    LaunchArrivalsObserved = 6U,
    LaunchReleaseObserved = 7U,
    LaunchPlanFatalObserved = 8U,
    LaunchSchedulerFatalObserved = 9U,
    LaunchReservedTasksObserved = 10U,
};

enum LeaderWord : uint32_t {
    LeaderMagic = 0U,
    LeaderBeginClock = 1U,
    LeaderEndClock = 2U,
    LeaderId = 3U,
    LeaderBuildOwner = 4U,
    LeaderAttachedTaskCount = 5U,
    LeaderReservedTasks = 6U,
    LeaderLastTask = 7U,
    LeaderLastBuildStatus = 8U,
    LeaderReservationStatus = 9U,
    LeaderArrivalStatus = 10U,
    LeaderReleasePublished = 11U,
    LeaderLocalFatal = 12U,
};

enum ContinuationWord : uint32_t {
    ContinuationMagic = 0U,
    ContinuationClock = 1U,
    ContinuationTaskCount = 2U,
    ContinuationValidation = 3U,
    ContinuationReleaseObserved = 4U,
    ContinuationPlanFatalObserved = 5U,
    ContinuationSchedulerFatalObserved = 6U,
    ContinuationJoinClock = 7U,
};

enum ValidationBit : uint64_t {
    ValidationLeaderMagic = uint64_t{1} << 0U,
    ValidationLeaderIdentity = uint64_t{1} << 1U,
    ValidationAttachedCount = uint64_t{1} << 2U,
    ValidationReservedCount = uint64_t{1} << 3U,
    ValidationArrivalOnce = uint64_t{1} << 4U,
    ValidationTicketTerminal = uint64_t{1} << 5U,
    ValidationRelease = uint64_t{1} << 6U,
    ValidationFatalPathJoined = uint64_t{1} << 7U,
    ValidationBuildStatus = uint64_t{1} << 8U,
};

constexpr uint64_t kSuccessValidationMask =
    ValidationLeaderMagic | ValidationLeaderIdentity |
    ValidationAttachedCount | ValidationReservedCount |
    ValidationArrivalOnce | ValidationTicketTerminal |
    ValidationRelease | ValidationBuildStatus;

static_assert(sizeof(ReportLine) == kReportLineBytes);
static_assert(alignof(ReportLine) == kReportLineBytes);
static_assert(sizeof(RuntimeReport) == 6U * kReportLineBytes);
static_assert(alignof(RuntimeReport) == kReportLineBytes);
static_assert(offsetof(RuntimeReport, leaders) == kReportLineBytes);
static_assert(offsetof(RuntimeReport, continuation) ==
              5U * kReportLineBytes);

}  // namespace pa_scheduler::aicpu_plan_simt::runtime_gate

#endif  // PA_AICPU_PLAN_SIMT_CCEC_RUNTIME_REPORT_H
