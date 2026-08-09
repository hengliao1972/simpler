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

#include "runtime.h"

#include <cerrno>
#include <inttypes.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "common/platform_config.h"
#include "common/unified_log.h"
#include "dist_engine/common/swimlane_types.h"

namespace {

constexpr uint32_t kFdwicSwimlaneMaxLevel = kFdwicAtomicSwimlaneLevel;
constexpr int32_t kFdwicSwimlanePhaseCount = static_cast<int32_t>(FdwicSwimlanePhase::Count);

struct TraceSummary {
    uint64_t records = 0;
    uint64_t atomic_records = 0;
    uint64_t clock_baseline_records = 0;
    uint64_t atomic_calls = 0;
    uint64_t poll_calls = 0;
    uint64_t poll_batch_records = 0;
#if PTO_FDWIC_SHARED_MAP
    uint64_t dcci_records = 0;
    uint64_t dcci_calls = 0;
    uint64_t dcci_lines = 0;
#endif
    uint64_t dropped_records = 0;
};

bool is_cpu_sim_trace() {
#if defined(SIMPLER_PLATFORM_NAME)
    return std::strcmp(SIMPLER_PLATFORM_NAME, "a5sim") == 0;
#else
    return false;
#endif
}

const char *phase_name(int32_t phase) {
    switch (static_cast<FdwicSwimlanePhase>(phase)) {
    case FdwicSwimlanePhase::Kernel:
        return "Kernel";
    case FdwicSwimlanePhase::Alloc:
        return "Alloc";
    case FdwicSwimlanePhase::Build:
        return "Build";
    case FdwicSwimlanePhase::DrainWon:
        return "DrainWon";
    case FdwicSwimlanePhase::Replay:
        return "Replay";
    case FdwicSwimlanePhase::RingBp:
        return "RingBp";
    case FdwicSwimlanePhase::EfDrain:
        return "EfDrain";
    case FdwicSwimlanePhase::Commit:
        return "Commit";
    case FdwicSwimlanePhase::Submit:
        return "Submit";
    case FdwicSwimlanePhase::Materialize:
        return "Materialize";
    case FdwicSwimlanePhase::PrepareMap:
        return "PrepareMap";
    case FdwicSwimlanePhase::Claim:
        return "Claim";
    case FdwicSwimlanePhase::Fanin:
        return "Fanin";
    case FdwicSwimlanePhase::Register:
        return "Register";
    case FdwicSwimlanePhase::Atomic:
        return "Atomic";
    case FdwicSwimlanePhase::ClockBaseline:
        return "ClockBaseline";
    case FdwicSwimlanePhase::OrchestrationReplay:
        return "OrchestrationReplay";
    case FdwicSwimlanePhase::FinalDrain:
        return "FinalDrain";
    case FdwicSwimlanePhase::WinnerBuild:
        return "WinnerBuild";
    case FdwicSwimlanePhase::AllocComplete:
        return "AllocComplete";
#if PTO_FDWIC_SHARED_MAP
    case FdwicSwimlanePhase::SharedRegisterPublishMetadata:
        return "SharedRegisterPublishMetadata";
    case FdwicSwimlanePhase::SharedMaterializePublishTaskOutputs:
        return "SharedMaterializePublishTaskOutputs";
    case FdwicSwimlanePhase::SharedMaterializePublishTaskOutputsCopy:
        return "SharedMaterializePublishTaskOutputsCopy";
    case FdwicSwimlanePhase::SharedMaterializePublishTaskOutputsFlush:
        return "SharedMaterializePublishTaskOutputsFlush";
    case FdwicSwimlanePhase::Dcci:
        return "Dcci";
    case FdwicSwimlanePhase::SharedRegisterWaitInsertTurnBypassLoad:
        return "SharedRegisterWaitInsertTurnBypassLoad";
#else
    case FdwicSwimlanePhase::LoserReplay:
        return "LoserReplay";
#endif
    case FdwicSwimlanePhase::Count:
        break;
    }
    return "Unknown";
}

const char *atomic_site_name(uint32_t site) {
    static constexpr const char *names[] = {
        "StartupIncrement",
        "StartupPoll",
        "FatalPoll",
        "FatalSet",
        "ClaimMax",
        "FaninFlagLoad",
        "CompletionVendExchange",
        "CompletionFlagExchange",
        "FrontierInitialLoad",
        "FrontierFlagLoad",
        "FrontierMax",
        "HeapFrontierLoad",
        "HeapVendLoad",
        "ReplayDoneIncrement",
        "ReplayDonePoll",
#if PTO_FDWIC_SHARED_MAP
        "SharedHeapVendLoad",
        "SharedHeapCursorLoad",
        "SharedHeapCursorReserve",
        "SharedHeapVendAdvance",
        "SharedInsertPredecessorPoll",
        "SharedInsertCompletionPublish",
        "SharedWinnerFatalGuardLoad",
        "SharedMetadataFatalGuardLoad",
        "SharedFaninOutputPublishedLoad",
        "SharedMetadataOutputPublishedLoad",
        "SharedFaninLastWriterLoad",
        "SharedMetadataLastWriterLoad",
        "SharedMetadataLastWriterCommit",
        "SharedOutputWriterReserve",
        "SharedOutputPublishedExchange",
        "SharedMapLookupHeadLoad",
        "SharedMapLookupTailLoad",
        "SharedMapLookupSeqLoad",
        "SharedMapAppendHeadLoad",
        "SharedMapAppendTailLoad",
        "SharedMapAppendSeqLoad",
        "SharedMapAppendSeqResetExchange",
        "SharedMapAppendSeqPublishExchange",
        "SharedMapAppendTailExchange",
        "SharedOutputRollbackExchange",
        "SharedClaimTournamentLocal",
        "SharedClaimTournamentRoot",
#else
        "WonSlotClaimMax",
        "WonRemainingExchange",
        "WonLaneResetExchange",
        "WonLaneDepositExchange",
        "WonStatePublishExchange",
        "WonAnyPublishExchange",
        "WonAnyLoad",
        "WonStateLoad",
        "WonLaneClaimExchange",
        "WonLaneReleaseExchange",
        "WonRemainingFetchSub",
        "WonStateClearExchange",
        "WonDrainedLoad",
#endif
    };
    static_assert(
        sizeof(names) / sizeof(names[0]) == static_cast<uint32_t>(FdwicAtomicSite::Count),
        "atomic site name table must match the raw ABI"
    );
    return site < sizeof(names) / sizeof(names[0]) ? names[site] : "Unknown";
}

const char *atomic_op_name(uint32_t op) {
#if PTO_FDWIC_SHARED_MAP
    static constexpr const char *names[] = {"Load", "Exchange", "FetchAdd", "FetchMax", "CompareExchange"};
#else
    static constexpr const char *names[] = {"Load", "Exchange", "FetchAdd", "FetchMax", "FetchSub"};
#endif
    return op < sizeof(names) / sizeof(names[0]) ? names[op] : "Unknown";
}

#if PTO_FDWIC_SHARED_MAP
const char *dcci_site_name(uint32_t site) {
    static constexpr const char *names[] = {
        "SharedFaninHistoryInvalidate",
        "SharedWriterHistoryFlush",
        "SharedOutputRollbackFlush",
        "SharedOutputDescriptorFlush",
        "SharedRegionReadInvalidate",
        "SharedRegionAppendInvalidate",
        "SharedRegionAppendFlush",
        "SharedWinnerBuildDescriptorInvalidate",
        "ObserverTraceExport",
        "StartupConfigInvalidate",
    };
    static_assert(
        sizeof(names) / sizeof(names[0]) == static_cast<uint32_t>(FdwicDcciSite::Count),
        "DCCI site name table must match the raw ABI"
    );
    return site < sizeof(names) / sizeof(names[0]) ? names[site] : "Unknown";
}

const char *dcci_op_name(uint32_t op) {
    static constexpr const char *names[] = {"Invalidate", "CleanOut"};
    return op < sizeof(names) / sizeof(names[0]) ? names[op] : "Unknown";
}
#endif

const char *core_type_name(CoreType core_type) {
    switch (core_type) {
    case CoreType::AIC:
        return "aic";
    case CoreType::AIV:
        return "aiv";
    }
    return "unknown";
}

const char *scheduler_mode_name(FdwicSchedulerMode mode) {
    switch (mode) {
    case FdwicSchedulerMode::SameCore:
        return "same_core";
    case FdwicSchedulerMode::CrossCoreOrdinary:
        return "cross_core_ordinary";
    case FdwicSchedulerMode::CrossCoreDag:
        return "cross_core_dag";
    case FdwicSchedulerMode::SimtCrossCoreOrdinary:
        return "simt_cross_core_ordinary";
    case FdwicSchedulerMode::SimtCrossCoreDag:
        return "simt_cross_core_dag";
    }
    return "unknown";
}

bool is_simt_scheduler(FdwicSchedulerMode mode) {
    return mode == FdwicSchedulerMode::SimtCrossCoreOrdinary || mode == FdwicSchedulerMode::SimtCrossCoreDag;
}

bool is_simt_builder_worker(FdwicSchedulerMode mode, CoreType core_type, int32_t block_id, int32_t lane) {
    constexpr int32_t kAiv0Lane = 1;
    if (!is_simt_scheduler(mode) || core_type != CoreType::AIV || block_id < 0 || lane != kAiv0Lane) return false;
    return mode == FdwicSchedulerMode::SimtCrossCoreDag || block_id == 0;
}

uint32_t expected_simt_builder_workers(FdwicSchedulerMode mode, uint32_t block_count) {
    if (mode == FdwicSchedulerMode::SimtCrossCoreDag) return block_count;
    return mode == FdwicSchedulerMode::SimtCrossCoreOrdinary ? 1U : 0U;
}

bool build_expected_core_layout(
    const Runtime *runtime, uint32_t num_cores, int32_t expected_blocks[RUNTIME_MAX_WORKER],
    int32_t expected_lanes[RUNTIME_MAX_WORKER]
) {
    uint32_t aic_count = 0;
    for (uint32_t core = 0; core < num_cores; ++core) {
        const CoreType core_type = runtime->workers[core].core_type;
        if (core_type == CoreType::AIC) {
            expected_blocks[core] = static_cast<int32_t>(aic_count++);
            expected_lanes[core] = 0;
        } else if (core_type != CoreType::AIV) {
            LOG_ERROR("fdwic swimlane core %u has invalid core type %d", core, static_cast<int32_t>(core_type));
            return false;
        }
    }
    if (aic_count == 0) {
        LOG_ERROR("fdwic swimlane topology has no AIC workers");
        return false;
    }

    uint32_t aiv_ordinal = 0;
    for (uint32_t core = 0; core < num_cores; ++core) {
        if (runtime->workers[core].core_type != CoreType::AIV) continue;
        const uint32_t block = aiv_ordinal / 2;
        if (block >= aic_count) {
            LOG_ERROR(
                "fdwic swimlane AIV worker %u cannot map to an AIC block: aic=%u aiv_ordinal=%u", core, aic_count,
                aiv_ordinal
            );
            return false;
        }
        expected_blocks[core] = static_cast<int32_t>(block);
        expected_lanes[core] = 1 + static_cast<int32_t>(aiv_ordinal % 2);
        ++aiv_ordinal;
    }
    if (aiv_ordinal != 2 * aic_count) {
        LOG_ERROR(
            "fdwic swimlane topology must contain two AIV workers per AIC: aic=%u aiv=%u", aic_count, aiv_ordinal
        );
        return false;
    }
    return true;
}

bool atomic_record_schema_valid(const FdwicSwimlaneRecord &record) {
    if (record.aux >= static_cast<uint32_t>(FdwicAtomicSite::Count)) return false;
    const auto site = static_cast<FdwicAtomicSite>(record.aux);
    const uint32_t op = record.flags & kFdwicAtomicOpMask;
    if (op != static_cast<uint32_t>(fdwic_atomic_site_op(site))) return false;

    const bool result_used = (record.flags & kFdwicAtomicResultUsed) != 0;
    const bool return_ready = (record.flags & kFdwicAtomicReturnReady) != 0;
    const bool value_zero = (record.flags & kFdwicAtomicValueZero) != 0;
    const bool poll_batch = (record.flags & kFdwicAtomicPollBatch) != 0;
    const uint32_t payload = record.flags >> kFdwicAtomicRetriesShift;
    if (poll_batch) {
#if PTO_FDWIC_SHARED_MAP
        return fdwic_atomic_site_is_poll_batchable(site) && result_used && !value_zero &&
               (site == FdwicAtomicSite::SharedInsertTurnPoll
                    ? return_ready == !is_cpu_sim_trace()
                    : !return_ready) &&
               payload > 0 &&
               record.task_id == -1 && record.func_id == -1;
#else
        return fdwic_atomic_site_is_poll_batchable(site) && result_used && !return_ready && !value_zero &&
               payload > 0 && record.task_id == -1 && record.func_id == -1;
#endif
    }
#if PTO_FDWIC_SHARED_MAP
    if (site == FdwicAtomicSite::SharedInsertTurnPoll) return false;
#endif
    const bool expected_return_ready = result_used && !is_cpu_sim_trace();
    if (result_used != fdwic_atomic_site_result_used(site) || return_ready != expected_return_ready) return false;
    if (value_zero && op != static_cast<uint32_t>(FdwicAtomicOp::Load)) return false;
    if (payload != 0 && op != static_cast<uint32_t>(FdwicAtomicOp::FetchMax)) return false;
    return record.func_id == -1;
}

uint32_t atomic_record_call_count(const FdwicSwimlaneRecord &record) {
    return (record.flags & kFdwicAtomicPollBatch) != 0 ? record.flags >> kFdwicAtomicPollCountShift : 1U;
}

#if PTO_FDWIC_SHARED_MAP
uint32_t dcci_record_call_count(const FdwicSwimlaneRecord &record) {
    return (record.flags >> kFdwicDcciCallCountShift) & kFdwicDcciCallCountMask;
}

uint32_t dcci_record_line_count(const FdwicSwimlaneRecord &record) {
    return record.flags >> kFdwicDcciLineCountShift;
}

bool dcci_record_schema_valid(const FdwicSwimlaneRecord &record) {
    if (record.aux >= static_cast<uint32_t>(FdwicDcciSite::Count)) return false;
    const auto site = static_cast<FdwicDcciSite>(record.aux);
    const uint32_t op_id = record.flags & kFdwicDcciOpMask;
    if (op_id >= static_cast<uint32_t>(FdwicDcciOp::Count) ||
        static_cast<FdwicDcciOp>(op_id) != fdwic_dcci_site_op(site) ||
        (record.flags & kFdwicDcciReservedBit) != 0 ||
        (record.flags & kFdwicDcciTrailingDsb) == 0) {
        return false;
    }
    const uint32_t calls = dcci_record_call_count(record);
    const uint32_t lines = dcci_record_line_count(record);
    if (calls == 0 || lines < calls) return false;
    if (site == FdwicDcciSite::ObserverTraceExport) {
        return calls == 3 && (record.flags & kFdwicDcciTrailingDsb) != 0 &&
               record.task_id == -1 && record.func_id == -1;
    }
    if (site == FdwicDcciSite::StartupConfigInvalidate) {
        return calls == 1 && (record.flags & kFdwicDcciTrailingDsb) != 0 &&
               record.task_id == -1 && record.func_id == -1;
    }
    return calls == 1 && record.task_id >= 0;
}
#endif

bool claim_record_schema_valid(const FdwicSwimlaneRecord &record) {
    if ((record.flags & ~(kFdwicClaimWon | kFdwicClaimAttempted)) != 0) return false;
    if ((record.flags & kFdwicClaimWon) != 0 && (record.flags & kFdwicClaimAttempted) == 0) return false;
    return record.aux <= 1;
}

bool clock_record_schema_valid(const FdwicSwimlaneRecord &record) {
    if ((record.flags & ~(kFdwicClockAtomicDependency | kFdwicClockAtomicDependencyApplied)) != 0) return false;
    if ((record.flags & kFdwicClockAtomicDependencyApplied) != 0 && (record.flags & kFdwicClockAtomicDependency) == 0) {
        return false;
    }
    const bool dependency = (record.flags & kFdwicClockAtomicDependency) != 0;
    const bool dependency_applied = (record.flags & kFdwicClockAtomicDependencyApplied) != 0;
    if (dependency_applied != (dependency && !is_cpu_sim_trace())) return false;
    return record.task_id == -1 && record.func_id == -1 && record.aux == 0;
}

bool ordinary_record_schema_valid(const FdwicSwimlaneRecord &record) {
    switch (static_cast<FdwicSwimlanePhase>(record.phase)) {
    case FdwicSwimlanePhase::Kernel:
    case FdwicSwimlanePhase::Commit:
        return record.flags <= 1 && record.aux == 0;
    case FdwicSwimlanePhase::DrainWon:
#if PTO_FDWIC_SHARED_MAP
        return false;
#else
        return record.flags == 1 && record.aux < 4;
#endif
    case FdwicSwimlanePhase::RingBp:
        return record.flags == 0 && record.aux <= 1;
    case FdwicSwimlanePhase::Submit:
        return record.flags <= 1 && record.aux <= 1;
    case FdwicSwimlanePhase::Materialize:
#if !PTO_FDWIC_SHARED_MAP
    case FdwicSwimlanePhase::PrepareMap:
#endif
    case FdwicSwimlanePhase::Register:
#if PTO_FDWIC_SHARED_MAP
        return record.flags == 0 && record.aux <= MAX_TENSOR_ARGS;
#else
        return record.flags == 0 && record.aux <= 1;
#endif
    case FdwicSwimlanePhase::Fanin:
        return record.flags == 0 && record.aux <= 16;
    case FdwicSwimlanePhase::Alloc:
    case FdwicSwimlanePhase::Build:
    case FdwicSwimlanePhase::Replay:
        // Schema-v4 reserves the legacy IDs for archived raw files but never
        // accepts newly produced overlapping lap records.
        return false;
    case FdwicSwimlanePhase::EfDrain:
#if PTO_FDWIC_SHARED_MAP
    case FdwicSwimlanePhase::PrepareMap:
        return false;
#else
        return record.flags == 0 && record.aux == 0;
#endif
    case FdwicSwimlanePhase::WinnerBuild:
    case FdwicSwimlanePhase::AllocComplete:
#if !PTO_FDWIC_SHARED_MAP
    case FdwicSwimlanePhase::LoserReplay:
#endif
        return record.flags == 0 && record.aux == 0;
#if PTO_FDWIC_SHARED_MAP
    case FdwicSwimlanePhase::SharedRegisterPublishMetadata:
    case FdwicSwimlanePhase::SharedMaterializePublishTaskOutputs:
    case FdwicSwimlanePhase::SharedMaterializePublishTaskOutputsCopy:
    case FdwicSwimlanePhase::SharedMaterializePublishTaskOutputsFlush:
        return record.flags == 0 && record.aux == 0;
    case FdwicSwimlanePhase::SharedRegisterWaitInsertTurnBypassLoad:
        return record.flags == 0;
#endif
    case FdwicSwimlanePhase::OrchestrationReplay:
    case FdwicSwimlanePhase::FinalDrain:
        return record.task_id == -1 && record.func_id == -1 && record.flags == 0 && record.aux == 0;
    case FdwicSwimlanePhase::Claim:
    case FdwicSwimlanePhase::Atomic:
    case FdwicSwimlanePhase::ClockBaseline:
        return true;
#if PTO_FDWIC_SHARED_MAP
    case FdwicSwimlanePhase::Dcci:
        return true;
#endif
    case FdwicSwimlanePhase::Count:
        return false;
    }
    return false;
}

bool validate_header_and_counts(
    const Runtime *runtime, const FdwicSwimlaneHeader *header, uint32_t level, TraceSummary &summary,
    uint32_t &max_core_records
) {
    const uint32_t expected_cores = runtime->fdwic_swimlane_num_cores_;
    const uint32_t expected_capacity = runtime->fdwic_swimlane_records_per_core_;
#if PTO_FDWIC_SHARED_MAP
    const uint64_t expected_bytes =
        sizeof(FdwicSwimlaneHeader) + static_cast<uint64_t>(expected_cores) * kFdwicSwimlaneWorkerBytes;
#else
    const uint64_t expected_bytes = sizeof(FdwicSwimlaneHeader) + static_cast<uint64_t>(expected_cores) *
                                                                      expected_capacity * sizeof(FdwicSwimlaneRecord);
#endif
    const bool header_valid =
        header->magic == kFdwicSwimlaneMagic && header->version == kFdwicSwimlaneVersion && expected_cores > 0 &&
        expected_cores <= RUNTIME_MAX_WORKER && header->num_cores == expected_cores &&
        runtime->worker_count == static_cast<int>(expected_cores) && expected_capacity > 0 &&
        header->records_per_core == expected_capacity && header->freq_hz == PLATFORM_PROF_SYS_CNT_FREQ &&
        runtime->fdwic_swimlane_bytes_ == expected_bytes && runtime->dist.swimlane_level == level &&
        runtime->dist.swimlane_base == runtime->fdwic_swimlane_dev_base_ &&
        runtime->dist.swimlane_records_per_core == expected_capacity
#if PTO_FDWIC_SHARED_MAP
        && header->record_size_bytes == kFdwicSwimlaneRecordSizeBytes
#endif
        ;
    if (!header_valid) {
        LOG_ERROR(
            "fdwic swimlane invalid header/state: magic=0x%08x version=%u cores=%u/%u worker_count=%d "
            "capacity=%u/%u freq=%llu bytes=%llu/%llu",
            header->magic, header->version, header->num_cores, expected_cores, runtime->worker_count,
            header->records_per_core, expected_capacity, static_cast<unsigned long long>(header->freq_hz),
            static_cast<unsigned long long>(runtime->fdwic_swimlane_bytes_),
            static_cast<unsigned long long>(expected_bytes)
        );
        return false;
    }

    for (uint32_t core = 0; core < expected_cores; ++core) {
        const FdwicSwimlaneCoreState &core_state = header->cores[core];
        summary.records += core_state.count;
        summary.atomic_calls += core_state.atomic_calls;
        summary.poll_calls += core_state.poll_calls;
        summary.poll_batch_records += core_state.poll_batch_records;
#if PTO_FDWIC_SHARED_MAP
        summary.records += 2ULL * kFdwicSharedTracePhase1TaskCount;
        summary.dcci_calls += core_state.dcci_calls;
        summary.dcci_lines += core_state.dcci_lines;
        summary.dcci_records += core_state.dcci_records;
#endif
        summary.dropped_records += core_state.dropped;
        if (core_state.count > max_core_records) max_core_records = core_state.count;
        if (core_state.count > expected_capacity || core_state.dropped != 0) {
            LOG_ERROR(
                "fdwic swimlane core %u is incomplete: count=%u capacity=%u dropped=%u atomic_calls=%u", core,
                core_state.count, expected_capacity, core_state.dropped, core_state.atomic_calls
            );
            return false;
        }
        if (level < kFdwicAtomicSwimlaneLevel &&
            (core_state.atomic_calls != 0 || core_state.poll_calls != 0 || core_state.poll_batch_records != 0
#if PTO_FDWIC_SHARED_MAP
             || core_state.dcci_calls != 0 || core_state.dcci_lines != 0 || core_state.dcci_records != 0
#endif
             )) {
            LOG_ERROR(
                "fdwic swimlane level-%u core %u unexpectedly reports atomic counters: calls=%u poll_calls=%u "
                "poll_batches=%u",
                level, core, core_state.atomic_calls, core_state.poll_calls, core_state.poll_batch_records
            );
            return false;
        }
        if (level >= kFdwicAtomicSwimlaneLevel) {
            if (core_state.poll_calls > core_state.atomic_calls ||
                (core_state.poll_calls == 0) != (core_state.poll_batch_records == 0)) {
                LOG_ERROR(
                    "fdwic swimlane level-4 core %u has invalid poll counters: calls=%u poll_calls=%u "
                    "poll_batches=%u",
                    core, core_state.atomic_calls, core_state.poll_calls, core_state.poll_batch_records
                );
                return false;
            }
            const uint64_t atomic_records =
                static_cast<uint64_t>(core_state.atomic_calls) - core_state.poll_calls + core_state.poll_batch_records;
            if (core_state.count < 2 || atomic_records > core_state.count - 2) {
                LOG_ERROR(
                    "fdwic swimlane level-4 core %u cannot close physical rows: count=%u atomic_records=%llu "
                    "atomic_calls=%u poll_calls=%u poll_batches=%u",
                    core, core_state.count, static_cast<unsigned long long>(atomic_records), core_state.atomic_calls,
                    core_state.poll_calls, core_state.poll_batch_records
                );
                return false;
            }
            summary.atomic_records += atomic_records;
#if PTO_FDWIC_SHARED_MAP
            if (core_state.dcci_records == 0 || core_state.dcci_records > core_state.dcci_calls ||
                core_state.dcci_lines < core_state.dcci_calls || core_state.dcci_records > core_state.count) {
                LOG_ERROR(
                    "fdwic shared swimlane level-4 core %u has invalid DCCI counters: records=%u calls=%u lines=%u",
                    core, core_state.dcci_records, core_state.dcci_calls, core_state.dcci_lines
                );
                return false;
            }
#endif
        }
    }
    if (level >= kFdwicAtomicSwimlaneLevel) {
        summary.clock_baseline_records = 2 * static_cast<uint64_t>(expected_cores);
    }
    return true;
}

#if PTO_FDWIC_SHARED_MAP
bool unfold_shared_compact_clock(uint32_t low, uint64_t anchor, uint64_t &unfolded) {
    constexpr uint64_t kWrap = UINT64_C(1) << 32U;
    constexpr uint64_t kHalfWrap = UINT64_C(1) << 31U;
    uint64_t candidate = (anchor & ~(kWrap - 1U)) | low;
    if (candidate > anchor && candidate - anchor > kHalfWrap) {
        if (candidate < kWrap) return false;
        candidate -= kWrap;
    } else if (anchor > candidate && anchor - candidate > kHalfWrap) {
        if (candidate > UINT64_MAX - kWrap) return false;
        candidate += kWrap;
    }
    unfolded = candidate;
    return true;
}

bool decode_shared_compact_record(
    const FdwicCompactSwimlaneRecord &compact, uint64_t anchor, FdwicSwimlaneRecord &logical
) {
    const uint32_t task_code = compact.packed & kFdwicCompactTraceTaskMask;
    const uint32_t func_code =
        (compact.packed >> kFdwicCompactTraceFunctionShift) & kFdwicCompactTraceFunctionMask;
    const uint32_t phase_code =
        (compact.packed >> kFdwicCompactTracePhaseShift) & kFdwicCompactTracePhaseMask;
    const uint32_t aux = (compact.packed >> kFdwicCompactTraceAuxShift) & kFdwicCompactTraceAuxMask;
    if ((task_code != kFdwicCompactTraceTaskSentinel && task_code >= kFdwicSharedTraceTaskCapacity) ||
        (func_code != kFdwicCompactTraceFunctionSentinel && func_code > 3U) ||
        phase_code >= static_cast<uint32_t>(FdwicSwimlanePhase::Count)) {
        return false;
    }
    uint64_t start = 0;
    if (!unfold_shared_compact_clock(compact.start_cycle_low, anchor, start)) return false;
    const uint64_t duration = static_cast<uint32_t>(compact.end_cycle_low - compact.start_cycle_low);
    if (start > UINT64_MAX - duration) return false;
    logical = {};
    logical.start_cycle = start;
    logical.end_cycle = start + duration;
    logical.task_id =
        task_code == kFdwicCompactTraceTaskSentinel ? -1 : static_cast<int32_t>(task_code);
    logical.func_id =
        func_code == kFdwicCompactTraceFunctionSentinel ? -1 : static_cast<int32_t>(func_code);
    logical.flags = compact.flags;
    logical.phase = static_cast<uint16_t>(phase_code);
    logical.aux = static_cast<uint16_t>(aux);
    return true;
}

enum class SharedPaTaskKind : uint32_t {
    Alloc = 0,
    Qk = 1,
    Sf = 2,
    Pv = 3,
    Up = 4,
};

SharedPaTaskKind shared_pa_task_kind(uint32_t task_id) {
    return static_cast<SharedPaTaskKind>(task_id % 5U);
}

bool shared_claim_attempted(
    bool is_aic, int32_t block_id, uint32_t task_id,
    SharedPaTaskKind kind
) {
    (void)block_id;
    (void)task_id;
    if (kind == SharedPaTaskKind::Alloc) {
        return true;
    }
    return is_aic ? kind == SharedPaTaskKind::Qk || kind == SharedPaTaskKind::Pv
                  : kind == SharedPaTaskKind::Sf || kind == SharedPaTaskKind::Up;
}

uint32_t shared_claim_tournament_groups(SharedPaTaskKind kind) {
    if (kind == SharedPaTaskKind::Alloc) return kFdwicSharedAllocClaimTournamentGroups;
    if (kind == SharedPaTaskKind::Qk || kind == SharedPaTaskKind::Pv) {
        return kFdwicSharedAicClaimTournamentGroups;
    }
    return kFdwicSharedAivClaimTournamentGroups;
}

bool shared_claim_tournament_group(
    uint32_t core, SharedPaTaskKind kind, uint32_t &group
) {
    uint32_t candidate_rank = 0;
    if (kind == SharedPaTaskKind::Alloc) {
        if (core >= kFdwicSharedWorkers) return false;
        candidate_rank = core;
    } else if (kind == SharedPaTaskKind::Qk || kind == SharedPaTaskKind::Pv) {
        if (core >= kFdwicSharedAicWorkers) return false;
        candidate_rank = core;
    } else {
        if (core < kFdwicSharedAicWorkers || core >= kFdwicSharedWorkers) return false;
        candidate_rank = core - kFdwicSharedAicWorkers;
    }
    group = candidate_rank % shared_claim_tournament_groups(kind);
    return true;
}

int32_t shared_task_function_id(SharedPaTaskKind kind) {
    return kind == SharedPaTaskKind::Alloc ? -1 : static_cast<int32_t>(kind) - 1;
}

struct SharedTaskDcciExpectation {
    uint32_t records = 0;
    uint32_t calls_per_record = 0;
    uint32_t lines_per_record = 0;
};

SharedTaskDcciExpectation shared_task_dcci_expectation(SharedPaTaskKind kind, FdwicDcciSite site) {
    switch (kind) {
    case SharedPaTaskKind::Alloc:
        if (site == FdwicDcciSite::SharedOutputDescriptorFlush) return {1, 1, 6};
        break;
    case SharedPaTaskKind::Qk:
        if (site == FdwicDcciSite::SharedOutputDescriptorFlush) return {1, 1, 2};
        break;
    case SharedPaTaskKind::Sf:
        if (site == FdwicDcciSite::SharedOutputDescriptorFlush) return {1, 1, 6};
        if (site == FdwicDcciSite::SharedWinnerBuildDescriptorInvalidate) return {1, 1, 2};
        break;
    case SharedPaTaskKind::Pv:
        if (site == FdwicDcciSite::SharedOutputDescriptorFlush) return {1, 1, 2};
        if (site == FdwicDcciSite::SharedWinnerBuildDescriptorInvalidate) return {1, 1, 2};
        break;
    case SharedPaTaskKind::Up:
        if (site == FdwicDcciSite::SharedWriterHistoryFlush) return {1, 1, 1};
        if (site == FdwicDcciSite::SharedFaninHistoryInvalidate) return {3, 1, 1};
        if (site == FdwicDcciSite::SharedWinnerBuildDescriptorInvalidate) return {6, 1, 2};
        break;
    }
    return {};
}

bool expand_shared_records(
    bool is_aic, int32_t block_id,
    const FdwicSwimlaneRecord *generic_records, uint32_t generic_count,
    const FdwicSharedSubmitClaimRecord *submit_claim_records, std::vector<FdwicSwimlaneRecord> &logical
) {
    if ((generic_count != 0 && generic_records == nullptr) || submit_claim_records == nullptr) return false;
    logical.clear();
    logical.reserve(static_cast<size_t>(generic_count) + 2U * kFdwicSharedTracePhase1TaskCount);
    uint32_t generic_index = 0;
    for (uint32_t task_id = 0; task_id < kFdwicSharedTracePhase1TaskCount; ++task_id) {
        const auto kind = shared_pa_task_kind(task_id);
        const FdwicSharedSubmitClaimRecord &endpoints = submit_claim_records[task_id];
        const bool winner = (endpoints.claim_end_and_winner & kFdwicSharedClaimWinnerBit) != 0;
        const bool attempted =
            shared_claim_attempted(is_aic, block_id, task_id, kind);
        const uint64_t claim_begin = endpoints.claim_begin;
        const uint64_t claim_end = endpoints.claim_end_and_winner & ~kFdwicSharedClaimWinnerBit;
        const uint64_t submit_begin = endpoints.submit_begin;
        const uint64_t submit_end = endpoints.submit_end;
        if (claim_begin == 0 || submit_begin == 0 || claim_end < claim_begin || submit_end < submit_begin ||
            submit_begin > claim_begin || claim_end > submit_end || (winner && !attempted)) {
            return false;
        }
        while (generic_index < generic_count && generic_records[generic_index].end_cycle <= claim_end) {
            const auto &record = generic_records[generic_index++];
            if (record.phase == static_cast<uint16_t>(FdwicSwimlanePhase::Claim) ||
                record.phase == static_cast<uint16_t>(FdwicSwimlanePhase::Submit) ||
                record.phase == static_cast<uint16_t>(FdwicSwimlanePhase::EfDrain)) {
                return false;
            }
            logical.push_back(record);
        }
        const int32_t function_id = winner ? shared_task_function_id(kind) : -1;
        const uint16_t is_alloc = kind == SharedPaTaskKind::Alloc ? 1U : 0U;
        FdwicSwimlaneRecord claim{};
        claim.start_cycle = claim_begin;
        claim.end_cycle = claim_end;
        claim.task_id = static_cast<int32_t>(task_id);
        claim.func_id = function_id;
        claim.flags = (winner ? kFdwicClaimWon : 0U) | (attempted ? kFdwicClaimAttempted : 0U);
        claim.phase = static_cast<uint16_t>(FdwicSwimlanePhase::Claim);
        claim.aux = is_alloc;
        logical.push_back(claim);

        while (generic_index < generic_count && generic_records[generic_index].end_cycle <= submit_end) {
            const auto &record = generic_records[generic_index++];
            if (record.phase == static_cast<uint16_t>(FdwicSwimlanePhase::Claim) ||
                record.phase == static_cast<uint16_t>(FdwicSwimlanePhase::Submit) ||
                record.phase == static_cast<uint16_t>(FdwicSwimlanePhase::EfDrain)) {
                return false;
            }
            logical.push_back(record);
        }
        FdwicSwimlaneRecord submit{};
        submit.start_cycle = submit_begin;
        submit.end_cycle = submit_end;
        submit.task_id = static_cast<int32_t>(task_id);
        submit.func_id = function_id;
        submit.flags = winner ? kFdwicClaimWon : 0U;
        submit.phase = static_cast<uint16_t>(FdwicSwimlanePhase::Submit);
        submit.aux = is_alloc;
        logical.push_back(submit);
    }
    while (generic_index < generic_count) {
        const auto &record = generic_records[generic_index++];
        if (record.phase == static_cast<uint16_t>(FdwicSwimlanePhase::Claim) ||
            record.phase == static_cast<uint16_t>(FdwicSwimlanePhase::Submit) ||
            record.phase == static_cast<uint16_t>(FdwicSwimlanePhase::EfDrain)) {
            return false;
        }
        logical.push_back(record);
    }
    return logical.size() == static_cast<size_t>(generic_count) + 2U * kFdwicSharedTracePhase1TaskCount;
}
#endif

#if !PTO_FDWIC_SHARED_MAP
bool validate_and_write_core(
    const FdwicSwimlaneHeader *header, const FdwicSwimlaneRecord *records, uint32_t core, int32_t expected_block,
    int32_t expected_lane, uint32_t level, std::ofstream &out, bool &first, TraceSummary &observed
) {
    const FdwicSwimlaneCoreState &core_state = header->cores[core];
    if (core_state.core_idx != static_cast<int32_t>(core) || core_state.block_id != expected_block ||
        core_state.lane != expected_lane) {
        LOG_ERROR(
            "fdwic swimlane invalid worker identity: worker=%u core=%d block=%d/%d lane=%d/%d", core,
            core_state.core_idx, core_state.block_id, expected_block, core_state.lane, expected_lane
        );
        return false;
    }
    uint32_t core_atomic_records = 0;
    uint64_t core_atomic_calls = 0;
    uint64_t core_poll_calls = 0;
    uint32_t core_poll_batch_records = 0;
    uint32_t core_clock_records = 0;
    uint32_t core_plain_clock_records = 0;
    uint32_t core_dependency_clock_records = 0;
    uint32_t core_orchestration_records = 0;
    uint32_t core_final_drain_records = 0;
    for (uint32_t index = 0; index < core_state.count; ++index) {
        const FdwicSwimlaneRecord &record = records[index];
        const bool base_valid = record.end_cycle >= record.start_cycle && record.phase < kFdwicSwimlanePhaseCount &&
                                record.task_id >= -1 && record.func_id >= -1;
        bool schema_valid = base_valid;
        if (record.phase == static_cast<int32_t>(FdwicSwimlanePhase::Atomic)) {
            schema_valid = schema_valid && atomic_record_schema_valid(record);
            ++core_atomic_records;
            const uint32_t call_count = atomic_record_call_count(record);
            core_atomic_calls += call_count;
            if ((record.flags & kFdwicAtomicPollBatch) != 0) {
                core_poll_calls += call_count;
                ++core_poll_batch_records;
            }
        } else if (record.phase == static_cast<int32_t>(FdwicSwimlanePhase::Claim)) {
            schema_valid = schema_valid && claim_record_schema_valid(record);
        } else if (record.phase == static_cast<int32_t>(FdwicSwimlanePhase::ClockBaseline)) {
            schema_valid = schema_valid && clock_record_schema_valid(record);
            ++core_clock_records;
            if ((record.flags & kFdwicClockAtomicDependency) != 0) {
                ++core_dependency_clock_records;
            } else {
                ++core_plain_clock_records;
            }
        } else {
            schema_valid = schema_valid && ordinary_record_schema_valid(record);
            if (record.phase == static_cast<int32_t>(FdwicSwimlanePhase::OrchestrationReplay)) {
                ++core_orchestration_records;
            } else if (record.phase == static_cast<int32_t>(FdwicSwimlanePhase::FinalDrain)) {
                ++core_final_drain_records;
            }
        }
        if (!schema_valid) {
            const uint32_t op = record.flags & kFdwicAtomicOpMask;
            LOG_ERROR(
                "fdwic swimlane invalid record: worker=%u index=%u core=%d block=%d lane=%d phase=%d(%s) "
                "task=%d func=%d start=%llu end=%llu flags=0x%08x aux=%u site=%s op=%s",
                core, index, core_state.core_idx, core_state.block_id, core_state.lane, record.phase,
                phase_name(record.phase), record.task_id, record.func_id,
                static_cast<unsigned long long>(record.start_cycle), static_cast<unsigned long long>(record.end_cycle),
                record.flags, record.aux, atomic_site_name(record.aux), atomic_op_name(op)
            );
            return false;
        }
        if (!first) out << ",";
        out << "\n    [" << core_state.core_idx << ", " << core_state.block_id << ", " << core_state.lane << ", "
            << record.task_id << ", " << record.func_id << ", \"" << phase_name(record.phase) << "\", "
            << record.start_cycle << ", " << record.end_cycle << ", " << record.flags << ", " << record.aux << "]";
        first = false;
    }

    if (core_orchestration_records != 1 || core_final_drain_records != 1) {
        LOG_ERROR(
            "fdwic swimlane schema-v4 parent closure failed on core %u: orchestration=%u final_drain=%u", core,
            core_orchestration_records, core_final_drain_records
        );
        return false;
    }

    if (level >= kFdwicAtomicSwimlaneLevel) {
        if (core_atomic_records != static_cast<uint64_t>(core_state.atomic_calls) - core_state.poll_calls +
                                       core_state.poll_batch_records ||
            core_atomic_calls != core_state.atomic_calls || core_poll_calls != core_state.poll_calls ||
            core_poll_batch_records != core_state.poll_batch_records || core_clock_records != 2 ||
            core_plain_clock_records != 1 || core_dependency_clock_records != 1) {
            LOG_ERROR(
                "fdwic swimlane level-4 closure failed on core %u: atomic_records=%u atomic_calls=%llu/%u "
                "poll_calls=%llu/%u poll_batches=%u/%u clock=%u plain_clock=%u dependency_clock=%u",
                core, core_atomic_records, static_cast<unsigned long long>(core_atomic_calls), core_state.atomic_calls,
                static_cast<unsigned long long>(core_poll_calls), core_state.poll_calls, core_poll_batch_records,
                core_state.poll_batch_records, core_clock_records, core_plain_clock_records,
                core_dependency_clock_records
            );
            return false;
        }
    } else if (core_atomic_records != 0 || core_state.atomic_calls != 0 || core_state.poll_calls != 0 ||
               core_state.poll_batch_records != 0 || core_clock_records != 0) {
        LOG_ERROR(
            "fdwic swimlane level-%u contains level-4 records on core %u: atomic_records=%u atomic_calls=%u "
            "poll_calls=%u poll_batches=%u clock=%u",
            level, core, core_atomic_records, core_state.atomic_calls, core_state.poll_calls,
            core_state.poll_batch_records, core_clock_records
        );
        return false;
    }
    observed.records += core_state.count;
    observed.atomic_records += core_atomic_records;
    observed.clock_baseline_records += core_clock_records;
    observed.atomic_calls += core_atomic_calls;
    observed.poll_calls += core_poll_calls;
    observed.poll_batch_records += core_poll_batch_records;
    observed.dropped_records += core_state.dropped;
    return true;
}
#endif

#if PTO_FDWIC_SHARED_MAP
struct SharedTaskTraceShape {
    static constexpr uint32_t kDcciSiteCount = static_cast<uint32_t>(FdwicDcciSite::Count);

    bool claim_seen = false;
    bool submit_seen = false;
    bool winner = false;
    uint32_t claim_local_count = 0;
    uint32_t claim_root_count = 0;
    uint32_t materialize_count = 0;
    uint32_t output_count = 0;
    uint32_t copy_count = 0;
    uint32_t flush_count = 0;
    uint32_t register_count = 0;
    uint32_t wait_insert_turn_count = 0;
    uint32_t metadata_count = 0;
    uint32_t fanin_count = 0;
    uint32_t tail_count = 0;
    uint32_t dcci_records[kDcciSiteCount]{};
    uint32_t dcci_calls[kDcciSiteCount]{};
    uint32_t dcci_lines[kDcciSiteCount]{};
    FdwicSwimlaneRecord claim_local{};
    FdwicSwimlaneRecord claim_root{};
    FdwicSwimlaneRecord claim{};
    FdwicSwimlaneRecord submit{};
    FdwicSwimlaneRecord materialize{};
    FdwicSwimlaneRecord output{};
    FdwicSwimlaneRecord copy{};
    FdwicSwimlaneRecord flush{};
    FdwicSwimlaneRecord reg{};
    FdwicSwimlaneRecord wait_insert_turn{};
    FdwicSwimlaneRecord metadata{};
    FdwicSwimlaneRecord fanin{};
    FdwicSwimlaneRecord tail{};
};

bool interval_contains(const FdwicSwimlaneRecord &outer, const FdwicSwimlaneRecord &inner) {
    return outer.start_cycle <= inner.start_cycle && inner.end_cycle <= outer.end_cycle;
}

bool validate_and_write_shared_core(
    const FdwicSwimlaneHeader *header, const FdwicSwimlaneRecord *records, uint32_t logical_count, uint32_t core,
    int32_t expected_block, int32_t expected_lane, uint32_t level, std::ofstream &out, bool &first,
    TraceSummary &observed, std::vector<uint32_t> &winner_counts,
    std::vector<std::array<uint32_t, kFdwicSharedClaimTournamentMaxGroups>> &root_group_counts
) {
    const FdwicSwimlaneCoreState &core_state = header->cores[core];
    if (core_state.core_idx != static_cast<int32_t>(core) || core_state.block_id != expected_block ||
        core_state.lane != expected_lane) {
        LOG_ERROR(
            "fdwic shared swimlane invalid worker identity: worker=%u core=%d block=%d/%d lane=%d/%d", core,
            core_state.core_idx, core_state.block_id, expected_block, core_state.lane, expected_lane
        );
        return false;
    }
    std::vector<SharedTaskTraceShape> shapes(kFdwicSharedTracePhase1TaskCount);
    std::vector<FdwicSwimlaneRecord> insert_turn_polls;
    std::vector<FdwicSwimlaneRecord> insert_turn_handoffs;
    uint32_t core_atomic_records = 0;
    uint64_t core_atomic_calls = 0;
    uint64_t core_poll_calls = 0;
    uint32_t core_poll_batch_records = 0;
    uint32_t core_clock_records = 0;
    uint32_t core_plain_clock_records = 0;
    uint32_t core_dependency_clock_records = 0;
    uint32_t core_dcci_records = 0;
    uint32_t core_observer_records = 0;
    uint32_t core_startup_dcci_records = 0;
    uint64_t core_dcci_calls = 0;
    uint64_t core_dcci_lines = 0;
    uint32_t core_orchestration_records = 0;
    uint32_t core_final_drain_records = 0;

    for (uint32_t index = 0; index < logical_count; ++index) {
        const FdwicSwimlaneRecord &record = records[index];
        const auto phase = static_cast<FdwicSwimlanePhase>(record.phase);
        bool valid =
            record.end_cycle >= record.start_cycle && record.phase < kFdwicSwimlanePhaseCount &&
            record.task_id >= -1 && record.func_id >= -1;
        if (phase == FdwicSwimlanePhase::Atomic) {
            valid = valid && level >= kFdwicAtomicSwimlaneLevel && atomic_record_schema_valid(record);
            ++core_atomic_records;
            const uint32_t calls = atomic_record_call_count(record);
            core_atomic_calls += calls;
            if ((record.flags & kFdwicAtomicPollBatch) != 0) {
                core_poll_calls += calls;
                ++core_poll_batch_records;
                if (record.aux == static_cast<uint16_t>(FdwicAtomicSite::SharedInsertTurnPoll)) {
                    insert_turn_polls.push_back(record);
                }
            } else if (record.aux == static_cast<uint16_t>(FdwicAtomicSite::SharedInsertTurnHandoff)) {
                insert_turn_handoffs.push_back(record);
            }
            if (valid && record.aux == static_cast<uint16_t>(FdwicAtomicSite::SharedInsertTurnHandoff) &&
                record.task_id < 0) {
                valid = false;
            }
            if (valid &&
                (record.aux == static_cast<uint16_t>(FdwicAtomicSite::SharedClaimTournamentLocal) ||
                 record.aux == static_cast<uint16_t>(FdwicAtomicSite::SharedClaimTournamentRoot))) {
                valid = record.task_id >= 0 &&
                        static_cast<uint32_t>(record.task_id) < kFdwicSharedTracePhase1TaskCount;
                if (valid) {
                    const uint32_t task_id = static_cast<uint32_t>(record.task_id);
                    const auto kind = shared_pa_task_kind(task_id);
                    uint32_t group = 0;
                    valid = shared_claim_tournament_group(core, kind, group);
                    if (valid) {
                        SharedTaskTraceShape &shape = shapes[task_id];
                        if (record.aux ==
                            static_cast<uint16_t>(FdwicAtomicSite::SharedClaimTournamentLocal)) {
                            if (++shape.claim_local_count == 1) shape.claim_local = record;
                        } else {
                            if (++shape.claim_root_count == 1) shape.claim_root = record;
                            ++root_group_counts[task_id][group];
                        }
                    }
                }
            }
        } else if (phase == FdwicSwimlanePhase::Dcci) {
            valid = valid && level >= kFdwicAtomicSwimlaneLevel && dcci_record_schema_valid(record);
            ++core_dcci_records;
            const uint32_t calls = dcci_record_call_count(record);
            const uint32_t lines = dcci_record_line_count(record);
            core_dcci_calls += calls;
            core_dcci_lines += lines;
            if (record.aux == static_cast<uint16_t>(FdwicDcciSite::ObserverTraceExport)) {
                ++core_observer_records;
            } else if (record.aux == static_cast<uint16_t>(FdwicDcciSite::StartupConfigInvalidate)) {
                ++core_startup_dcci_records;
            } else if (valid) {
                valid = record.task_id >= 0 && record.func_id == -1 &&
                        static_cast<uint32_t>(record.task_id) < kFdwicSharedTracePhase1TaskCount;
                if (valid) {
                    const uint32_t task_id = static_cast<uint32_t>(record.task_id);
                    const uint32_t site_id = record.aux;
                    const auto expected = shared_task_dcci_expectation(
                        shared_pa_task_kind(task_id), static_cast<FdwicDcciSite>(site_id)
                    );
                    valid = expected.records != 0 && calls == expected.calls_per_record &&
                            lines == expected.lines_per_record;
                    if (valid) {
                        SharedTaskTraceShape &shape = shapes[task_id];
                        ++shape.dcci_records[site_id];
                        shape.dcci_calls[site_id] += calls;
                        shape.dcci_lines[site_id] += lines;
                    }
                }
            }
        } else if (phase == FdwicSwimlanePhase::ClockBaseline) {
            valid = valid && level >= kFdwicAtomicSwimlaneLevel && clock_record_schema_valid(record);
            ++core_clock_records;
            if ((record.flags & kFdwicClockAtomicDependency) != 0) {
                ++core_dependency_clock_records;
            } else {
                ++core_plain_clock_records;
            }
        } else if (phase == FdwicSwimlanePhase::Claim || phase == FdwicSwimlanePhase::Submit) {
            valid = valid && record.task_id >= 0 &&
                    static_cast<uint32_t>(record.task_id) < kFdwicSharedTracePhase1TaskCount;
            if (valid) {
                const uint32_t task_id = static_cast<uint32_t>(record.task_id);
                const auto kind = shared_pa_task_kind(task_id);
                SharedTaskTraceShape &shape = shapes[task_id];
                const bool is_alloc = kind == SharedPaTaskKind::Alloc;
                if (phase == FdwicSwimlanePhase::Claim) {
                    const bool winner = (record.flags & kFdwicClaimWon) != 0;
                    const bool attempted = shared_claim_attempted(
                        expected_lane == 0, expected_block, task_id, kind
                    );
                    valid =
                        !shape.claim_seen && claim_record_schema_valid(record) &&
                        record.flags ==
                            ((winner ? kFdwicClaimWon : 0U) | (attempted ? kFdwicClaimAttempted : 0U)) &&
                        record.func_id == (winner ? shared_task_function_id(kind) : -1) &&
                        record.aux == (is_alloc ? 1U : 0U);
                    if (valid) {
                        shape.claim_seen = true;
                        shape.winner = winner;
                        shape.claim = record;
                    }
                } else {
                    valid =
                        shape.claim_seen && !shape.submit_seen &&
                        record.flags == (shape.winner ? kFdwicClaimWon : 0U) &&
                        record.func_id == (shape.winner ? shared_task_function_id(kind) : -1) &&
                        record.aux == (is_alloc ? 1U : 0U) &&
                        record.start_cycle <= shape.claim.start_cycle && record.end_cycle >= shape.claim.end_cycle;
                    if (valid) {
                        shape.submit_seen = true;
                        shape.submit = record;
                    }
                }
            }
        } else {
            valid = valid && ordinary_record_schema_valid(record);
            if (phase == FdwicSwimlanePhase::OrchestrationReplay) {
                ++core_orchestration_records;
            } else if (phase == FdwicSwimlanePhase::FinalDrain) {
                ++core_final_drain_records;
            }
            const bool winner_business =
                phase == FdwicSwimlanePhase::Materialize || phase == FdwicSwimlanePhase::Register ||
                phase == FdwicSwimlanePhase::Fanin || phase == FdwicSwimlanePhase::WinnerBuild ||
                phase == FdwicSwimlanePhase::AllocComplete ||
                phase == FdwicSwimlanePhase::SharedRegisterWaitInsertTurnBypassLoad ||
                phase == FdwicSwimlanePhase::SharedRegisterPublishMetadata ||
                phase == FdwicSwimlanePhase::SharedMaterializePublishTaskOutputs ||
                phase == FdwicSwimlanePhase::SharedMaterializePublishTaskOutputsCopy ||
                phase == FdwicSwimlanePhase::SharedMaterializePublishTaskOutputsFlush;
            if (valid && winner_business) {
                valid = record.task_id >= 0 &&
                        static_cast<uint32_t>(record.task_id) < kFdwicSharedTracePhase1TaskCount;
                if (valid) {
                    SharedTaskTraceShape &shape = shapes[static_cast<uint32_t>(record.task_id)];
                    valid = shape.claim_seen && shape.winner &&
                            record.func_id == shared_task_function_id(
                                shared_pa_task_kind(static_cast<uint32_t>(record.task_id))
                            );
                    if (valid) {
                        switch (phase) {
                        case FdwicSwimlanePhase::Materialize:
                            shape.materialize = record;
                            valid = ++shape.materialize_count == 1;
                            break;
                        case FdwicSwimlanePhase::SharedMaterializePublishTaskOutputs:
                            shape.output = record;
                            valid = ++shape.output_count == 1;
                            break;
                        case FdwicSwimlanePhase::SharedMaterializePublishTaskOutputsCopy:
                            shape.copy = record;
                            valid = ++shape.copy_count == 1;
                            break;
                        case FdwicSwimlanePhase::SharedMaterializePublishTaskOutputsFlush:
                            shape.flush = record;
                            valid = ++shape.flush_count == 1;
                            break;
                        case FdwicSwimlanePhase::Register:
                            shape.reg = record;
                            valid = ++shape.register_count == 1;
                            break;
                        case FdwicSwimlanePhase::SharedRegisterWaitInsertTurnBypassLoad:
                            shape.wait_insert_turn = record;
                            valid = ++shape.wait_insert_turn_count == 1;
                            break;
                        case FdwicSwimlanePhase::SharedRegisterPublishMetadata:
                            shape.metadata = record;
                            valid = ++shape.metadata_count == 1;
                            break;
                        case FdwicSwimlanePhase::Fanin:
                            shape.fanin = record;
                            valid = ++shape.fanin_count == 1;
                            break;
                        case FdwicSwimlanePhase::WinnerBuild:
                        case FdwicSwimlanePhase::AllocComplete:
                            shape.tail = record;
                            valid = ++shape.tail_count == 1;
                            break;
                        default:
                            break;
                        }
                    }
                }
            }
            if (valid && phase == FdwicSwimlanePhase::Kernel && record.task_id >= 0) {
                valid = static_cast<uint32_t>(record.task_id) < kFdwicSharedTracePhase1TaskCount &&
                        shapes[static_cast<uint32_t>(record.task_id)].winner;
            }
        }
        if (!valid) {
            LOG_ERROR(
                "fdwic shared swimlane invalid record: worker=%u index=%u phase=%u(%s) task=%d func=%d "
                "start=%llu end=%llu flags=0x%08x aux=%u",
                core, index, record.phase, phase_name(record.phase), record.task_id, record.func_id,
                static_cast<unsigned long long>(record.start_cycle),
                static_cast<unsigned long long>(record.end_cycle), record.flags, record.aux
            );
            return false;
        }
        if (!first) out << ",";
        out << "\n    [" << core_state.core_idx << ", " << core_state.block_id << ", " << core_state.lane << ", "
            << record.task_id << ", " << record.func_id << ", \"" << phase_name(record.phase) << "\", "
            << record.start_cycle << ", " << record.end_cycle << ", " << record.flags << ", " << record.aux << "]";
        first = false;
    }

    for (uint32_t task_id = 0; task_id < kFdwicSharedTracePhase1TaskCount; ++task_id) {
        const auto kind = shared_pa_task_kind(task_id);
        const bool is_alloc = kind == SharedPaTaskKind::Alloc;
        const SharedTaskTraceShape &shape = shapes[task_id];
        bool shape_valid = shape.claim_seen && shape.submit_seen;
        if (level >= kFdwicAtomicSwimlaneLevel) {
            const bool attempted =
                shared_claim_attempted(expected_lane == 0, expected_block, task_id, kind);
            const uint32_t expected_local = attempted ? 1U : 0U;
            const bool tournament_valid =
                shape.claim_local_count == expected_local &&
                (!attempted || interval_contains(shape.claim, shape.claim_local)) &&
                shape.claim_root_count <= 1U &&
                (shape.claim_root_count == 0U ||
                 (shape.claim_local_count == 1U && interval_contains(shape.claim, shape.claim_root))) &&
                (!shape.winner || shape.claim_root_count == 1U);
            if (!tournament_valid) {
                LOG_ERROR(
                    "fdwic shared Claim tournament closure failed: worker=%u task=%u attempted=%d "
                    "local=%u root=%u winner=%d claim=[%llu,%llu]",
                    core, task_id, attempted ? 1 : 0, shape.claim_local_count,
                    shape.claim_root_count, shape.winner ? 1 : 0,
                    static_cast<unsigned long long>(shape.claim.start_cycle),
                    static_cast<unsigned long long>(shape.claim.end_cycle)
                );
                return false;
            }
            for (uint32_t site_id = 0; site_id < SharedTaskTraceShape::kDcciSiteCount; ++site_id) {
                const auto expected =
                    shape.winner
                        ? shared_task_dcci_expectation(kind, static_cast<FdwicDcciSite>(site_id))
                        : SharedTaskDcciExpectation{};
                const uint32_t expected_calls = expected.records * expected.calls_per_record;
                const uint32_t expected_lines = expected.records * expected.lines_per_record;
                if (shape.dcci_records[site_id] != expected.records ||
                    shape.dcci_calls[site_id] != expected_calls ||
                    shape.dcci_lines[site_id] != expected_lines) {
                    LOG_ERROR(
                        "fdwic shared task DCCI closure failed: worker=%u task=%u winner=%d kind=%u "
                        "site=%u(%s) records=%u/%u calls=%u/%u lines=%u/%u",
                        core, task_id, shape.winner ? 1 : 0, static_cast<uint32_t>(kind), site_id,
                        dcci_site_name(site_id), shape.dcci_records[site_id], expected.records,
                        shape.dcci_calls[site_id], expected_calls, shape.dcci_lines[site_id], expected_lines
                    );
                    return false;
                }
            }
        }
        if (shape.winner) {
            shape_valid =
                shape_valid && shape.materialize_count == 1 && shape.output_count == 1 && shape.copy_count == 1 &&
                shape.flush_count == 1 && shape.register_count == 1 && shape.wait_insert_turn_count == 1 &&
                shape.metadata_count == 1 &&
                shape.fanin_count == (is_alloc ? 0U : 1U) && shape.tail_count == 1 &&
                interval_contains(shape.submit, shape.materialize) && interval_contains(shape.materialize, shape.output) &&
                interval_contains(shape.output, shape.copy) && interval_contains(shape.output, shape.flush) &&
                shape.copy.end_cycle == shape.flush.start_cycle && interval_contains(shape.submit, shape.reg) &&
                interval_contains(shape.reg, shape.wait_insert_turn) && interval_contains(shape.reg, shape.metadata) &&
                shape.wait_insert_turn.start_cycle == shape.reg.start_cycle &&
                shape.wait_insert_turn.end_cycle == shape.metadata.start_cycle &&
                ((task_id == 0 && shape.wait_insert_turn.aux == 0) ||
                 (task_id != 0 && shape.wait_insert_turn.aux != 0)) &&
                shape.materialize.end_cycle == shape.reg.start_cycle &&
                (is_alloc ? shape.reg.end_cycle == shape.tail.start_cycle
                          : shape.reg.end_cycle == shape.fanin.start_cycle &&
                                shape.fanin.end_cycle == shape.tail.start_cycle) &&
                shape.tail.end_cycle <= shape.submit.end_cycle &&
                static_cast<FdwicSwimlanePhase>(shape.tail.phase) ==
                    (is_alloc ? FdwicSwimlanePhase::AllocComplete : FdwicSwimlanePhase::WinnerBuild);
        } else {
            shape_valid =
                shape_valid && shape.materialize_count == 0 && shape.output_count == 0 && shape.copy_count == 0 &&
                shape.flush_count == 0 && shape.register_count == 0 && shape.wait_insert_turn_count == 0 &&
                shape.metadata_count == 0 &&
                shape.fanin_count == 0 && shape.tail_count == 0;
        }
        if (!shape_valid) {
            LOG_ERROR(
                "fdwic shared swimlane sparse task closure failed: worker=%u task=%u winner=%d "
                "materialize=%u output=%u/%u/%u register=%u/%u/%u fanin=%u tail=%u",
                core, task_id, shape.winner ? 1 : 0, shape.materialize_count, shape.output_count, shape.copy_count,
                shape.flush_count, shape.register_count, shape.wait_insert_turn_count, shape.metadata_count,
                shape.fanin_count, shape.tail_count
            );
            return false;
        }
        if (shape.winner) ++winner_counts[task_id];
    }
    if (level >= kFdwicAtomicSwimlaneLevel) {
        uint32_t expected_handoff_records = 0;
        if (!insert_turn_polls.empty()) {
            LOG_ERROR(
                "fdwic shared bypass-load insert turn forbids atomic PollBatch rows: records=%zu",
                insert_turn_polls.size()
            );
            return false;
        }
        for (uint32_t task_id = 0; task_id < kFdwicSharedTracePhase1TaskCount; ++task_id) {
            const SharedTaskTraceShape &shape = shapes[task_id];
            if (!shape.winner) continue;
            const uint32_t matching_handoffs = static_cast<uint32_t>(std::count_if(
                insert_turn_handoffs.begin(), insert_turn_handoffs.end(),
                [task_id, &shape](const FdwicSwimlaneRecord &handoff) {
                    return handoff.task_id == static_cast<int32_t>(task_id) &&
                           shape.metadata.end_cycle <= handoff.start_cycle &&
                           handoff.end_cycle <= shape.reg.end_cycle;
                }
            ));
            ++expected_handoff_records;
            if (matching_handoffs != 1) {
                LOG_ERROR(
                    "fdwic shared task %u insert-turn handoff closure failed: records=%u expected=1",
                    task_id, matching_handoffs
                );
                return false;
            }
        }
        if (insert_turn_handoffs.size() != expected_handoff_records) {
            LOG_ERROR(
                "fdwic shared insert-turn orphan records: handoffs=%zu/%u",
                insert_turn_handoffs.size(), expected_handoff_records
            );
            return false;
        }
    }
    const bool parents_closed = core_orchestration_records == 1 && core_final_drain_records == 1;
    bool counters_closed = false;
    if (level >= kFdwicAtomicSwimlaneLevel) {
        counters_closed =
            core_atomic_records ==
                static_cast<uint64_t>(core_state.atomic_calls) - core_state.poll_calls +
                    core_state.poll_batch_records &&
            core_atomic_calls == core_state.atomic_calls && core_poll_calls == core_state.poll_calls &&
            core_poll_batch_records == core_state.poll_batch_records && core_clock_records == 2 &&
            core_plain_clock_records == 1 && core_dependency_clock_records == 1 &&
            core_dcci_records == core_state.dcci_records && core_dcci_calls == core_state.dcci_calls &&
            core_dcci_lines == core_state.dcci_lines && core_observer_records == 1 &&
            core_startup_dcci_records == 1;
    } else {
        counters_closed =
            core_atomic_records == 0 && core_atomic_calls == 0 && core_poll_calls == 0 &&
            core_poll_batch_records == 0 && core_clock_records == 0 && core_dcci_records == 0 &&
            core_dcci_calls == 0 && core_dcci_lines == 0;
    }
    if (!parents_closed || !counters_closed) {
        LOG_ERROR(
            "fdwic shared swimlane core closure failed: worker=%u orchestration=%u final=%u "
            "atomic=%u/%u dcci=%u/%u clock=%u",
            core, core_orchestration_records, core_final_drain_records, core_atomic_records,
            core_state.atomic_calls, core_dcci_records, core_state.dcci_records, core_clock_records
        );
        return false;
    }
    observed.records += logical_count;
    observed.atomic_records += core_atomic_records;
    observed.clock_baseline_records += core_clock_records;
    observed.atomic_calls += core_atomic_calls;
    observed.poll_calls += core_poll_calls;
    observed.poll_batch_records += core_poll_batch_records;
    observed.dcci_records += core_dcci_records;
    observed.dcci_calls += core_dcci_calls;
    observed.dcci_lines += core_dcci_lines;
    observed.dropped_records += core_state.dropped;
    return true;
}
#endif

std::string output_path_from_prefix(const std::string &prefix) {
    if (!prefix.empty() && prefix.back() == '/') return prefix + "l2_swimlane_records.json";
    return prefix + "/l2_swimlane_records.json";
}

std::string output_path(const Runtime *runtime) {
    return output_path_from_prefix(runtime->fdwic_swimlane_output_prefix_);
}

std::string perf_clock_output_path_from_prefix(const std::string &prefix) {
    const char *mode = std::getenv("PTO_FDWIC_PROFILE");
    const bool kernel = mode != nullptr && std::strcmp(mode, "perf-clock-kernel") == 0;
    const char *name = kernel ? "fdwic_perf_clock_kernel_summary.json" : "fdwic_perf_clock_summary.json";
    if (!prefix.empty() && prefix.back() == '/') return prefix + name;
    return prefix + "/" + name;
}

bool perf_clock_kernel_requested() {
    const char *mode = std::getenv("PTO_FDWIC_PROFILE");
    return mode != nullptr && std::strcmp(mode, "perf-clock-kernel") == 0;
}

bool perf_clock_requested() {
    const char *mode = std::getenv("PTO_FDWIC_PROFILE");
    return mode != nullptr && (std::strcmp(mode, "perf-clock") == 0 || std::strcmp(mode, "perf-clock-kernel") == 0);
}

struct PerfClockGroupAggregate {
    uint32_t cores = 0;
    uint64_t elapsed_sum = 0;
    uint64_t elapsed_min = std::numeric_limits<uint64_t>::max();
    uint64_t elapsed_max = 0;
    uint64_t kernel_sum = 0;
    uint64_t kernel_min = std::numeric_limits<uint64_t>::max();
    uint64_t kernel_max = 0;
    uint64_t residual_sum = 0;
    uint64_t residual_min = std::numeric_limits<uint64_t>::max();
    uint64_t residual_max = 0;
    uint64_t kernel_calls_sum = 0;
    uint32_t kernel_calls_min = std::numeric_limits<uint32_t>::max();
    uint32_t kernel_calls_max = 0;

    void add(uint64_t elapsed, uint64_t kernel, uint32_t calls) {
        const uint64_t residual = elapsed - kernel;
        ++cores;
        elapsed_sum += elapsed;
        elapsed_min = std::min(elapsed_min, elapsed);
        elapsed_max = std::max(elapsed_max, elapsed);
        kernel_sum += kernel;
        kernel_min = std::min(kernel_min, kernel);
        kernel_max = std::max(kernel_max, kernel);
        residual_sum += residual;
        residual_min = std::min(residual_min, residual);
        residual_max = std::max(residual_max, residual);
        kernel_calls_sum += calls;
        kernel_calls_min = std::min(kernel_calls_min, calls);
        kernel_calls_max = std::max(kernel_calls_max, calls);
    }
};

bool prepare_output_directory(const std::string &prefix) {
    struct stat info{};
    if (stat(prefix.c_str(), &info) == 0) {
        if (S_ISDIR(info.st_mode)) return true;
        LOG_ERROR("fdwic swimlane output prefix is not a directory: %s", prefix.c_str());
        return false;
    }
    if (errno != ENOENT || mkdir(prefix.c_str(), 0755) != 0) {
        LOG_ERROR("cannot create fdwic swimlane output directory %s: %s", prefix.c_str(), std::strerror(errno));
        return false;
    }
    return true;
}

bool remove_output_if_present(const std::string &path) {
    if (std::remove(path.c_str()) == 0 || errno == ENOENT) return true;
    LOG_ERROR("cannot remove stale fdwic swimlane output %s: %s", path.c_str(), std::strerror(errno));
    return false;
}

bool should_print_trace_export() {
#if defined(SIMPLER_PLATFORM_NAME)
    return std::strcmp(SIMPLER_PLATFORM_NAME, "a5sim") == 0;
#else
    return false;
#endif
}

}  // namespace

extern "C" int fdwic_swimlane_host_init(Runtime *runtime, int num_cores, int level, const char *output_prefix) {
    if (runtime == nullptr) return -1;
    runtime->dist.swimlane_level = 0;
    runtime->dist.swimlane_base = 0;
    runtime->dist.swimlane_records_per_core = 0;
    runtime->fdwic_swimlane_host_shadow_ = nullptr;
    runtime->fdwic_swimlane_dev_allocation_ = 0;
    runtime->fdwic_swimlane_dev_base_ = 0;
    runtime->fdwic_swimlane_bytes_ = 0;
    runtime->fdwic_swimlane_num_cores_ = 0;
    runtime->fdwic_swimlane_records_per_core_ = 0;
    runtime->fdwic_swimlane_output_prefix_[0] = '\0';
    if (level < 0 || level > static_cast<int>(kFdwicSwimlaneMaxLevel)) {
        LOG_ERROR("fdwic swimlane level %d is outside [0, %u]", level, kFdwicSwimlaneMaxLevel);
        return -1;
    }
#if PTO_FDWIC_SHARED_MAP
    if (level == 2 || level == 3) {
        LOG_ERROR("fdwic shared schema-v5 supports only swimlane levels 0, 1, and 4; got %d", level);
        return -1;
    }
#endif
    if (level == 0) return 0;
    if (num_cores <= 0 || num_cores > RUNTIME_MAX_WORKER) return -1;
    if (runtime->host_api.device_malloc == nullptr || runtime->host_api.device_free == nullptr ||
        runtime->host_api.copy_to_device == nullptr || runtime->host_api.copy_from_device == nullptr) {
        return -1;
    }

    const std::string exact_output_prefix = output_prefix == nullptr || output_prefix[0] == '\0' ? "." : output_prefix;
    if (exact_output_prefix.size() >= sizeof(runtime->fdwic_swimlane_output_prefix_)) {
        LOG_ERROR("fdwic swimlane output prefix is too long: %zu bytes", exact_output_prefix.size());
        return -1;
    }
    if (!prepare_output_directory(exact_output_prefix)) return -1;
    const std::string path = output_path_from_prefix(exact_output_prefix);
    if (!remove_output_if_present(path) || !remove_output_if_present(path + ".tmp")) return -1;

    const uint32_t records_per_core = level >= static_cast<int>(kFdwicAtomicSwimlaneLevel) ?
                                          kFdwicAtomicSwimlaneRecordsPerCore :
                                          kFdwicSwimlaneDefaultRecordsPerCore;
#if PTO_FDWIC_SHARED_MAP
    const uint64_t bytes =
        sizeof(FdwicSwimlaneHeader) + static_cast<uint64_t>(num_cores) * kFdwicSwimlaneWorkerBytes;
#else
    const uint64_t bytes =
        sizeof(FdwicSwimlaneHeader) + static_cast<uint64_t>(num_cores) * records_per_core * sizeof(FdwicSwimlaneRecord);
#endif
    constexpr uint64_t kDeviceAlignment = 64;
    if (bytes > std::numeric_limits<size_t>::max() - (kDeviceAlignment - 1)) {
        LOG_ERROR("fdwic swimlane allocation is too large: %llu bytes", static_cast<unsigned long long>(bytes));
        return -1;
    }
    void *host_shadow = std::aligned_alloc(64, sizeof(FdwicSwimlaneHeader));
    if (host_shadow == nullptr) return -1;
    std::memset(host_shadow, 0, sizeof(FdwicSwimlaneHeader));
    FdwicSwimlaneHeader *header = reinterpret_cast<FdwicSwimlaneHeader *>(host_shadow);
    header->magic = kFdwicSwimlaneMagic;
    header->version = kFdwicSwimlaneVersion;
    header->num_cores = static_cast<uint32_t>(num_cores);
    header->records_per_core = records_per_core;
    header->freq_hz = PLATFORM_PROF_SYS_CNT_FREQ;
#if PTO_FDWIC_SHARED_MAP
    header->record_size_bytes = kFdwicSwimlaneRecordSizeBytes;
#endif

    void *dev_allocation = runtime->host_api.device_malloc(static_cast<size_t>(bytes + (kDeviceAlignment - 1)));
    if (dev_allocation == nullptr) {
        std::free(host_shadow);
        return -1;
    }
    const uintptr_t dev_base =
        (reinterpret_cast<uintptr_t>(dev_allocation) + (kDeviceAlignment - 1)) & ~(kDeviceAlignment - 1);
    void *dev = reinterpret_cast<void *>(dev_base);
    if (runtime->host_api.copy_to_device(dev, host_shadow, sizeof(FdwicSwimlaneHeader)) != 0) {
        runtime->host_api.device_free(dev_allocation);
        std::free(host_shadow);
        return -1;
    }

    runtime->fdwic_swimlane_host_shadow_ = host_shadow;
    runtime->fdwic_swimlane_dev_allocation_ = reinterpret_cast<uint64_t>(dev_allocation);
    runtime->fdwic_swimlane_dev_base_ = dev_base;
    runtime->fdwic_swimlane_bytes_ = bytes;
    runtime->fdwic_swimlane_num_cores_ = static_cast<uint32_t>(num_cores);
    runtime->fdwic_swimlane_records_per_core_ = records_per_core;
    std::memcpy(runtime->fdwic_swimlane_output_prefix_, exact_output_prefix.c_str(), exact_output_prefix.size() + 1);
    runtime->dist.swimlane_base = runtime->fdwic_swimlane_dev_base_;
    runtime->dist.swimlane_records_per_core = records_per_core;
    runtime->dist.swimlane_level = static_cast<uint32_t>(level);
    return 1;
}

extern "C" int fdwic_swimlane_host_export(Runtime *runtime) {
    if (runtime == nullptr || runtime->fdwic_swimlane_host_shadow_ == nullptr ||
        runtime->fdwic_swimlane_dev_base_ == 0) {
        return 0;
    }
    const std::string path = output_path(runtime);
    const std::string temporary_path = path + ".tmp";
    if (!remove_output_if_present(temporary_path)) return -1;

    void *dev = reinterpret_cast<void *>(runtime->fdwic_swimlane_dev_base_);
    if (runtime->host_api.copy_from_device(runtime->fdwic_swimlane_host_shadow_, dev, sizeof(FdwicSwimlaneHeader)) !=
        0) {
        LOG_ERROR("fdwic swimlane header D2H copy failed");
        std::remove(temporary_path.c_str());
        return -1;
    }

    FdwicSwimlaneHeader *header = reinterpret_cast<FdwicSwimlaneHeader *>(runtime->fdwic_swimlane_host_shadow_);
    const uint32_t level = runtime->dist.swimlane_level;
    if (level == 0 || level > kFdwicSwimlaneMaxLevel) {
        LOG_ERROR("fdwic swimlane export has invalid level %u", level);
        std::remove(temporary_path.c_str());
        return -1;
    }
    TraceSummary summary;
    uint32_t max_core_records = 0;
    if (!validate_header_and_counts(runtime, header, level, summary, max_core_records)) {
        std::remove(temporary_path.c_str());
        return -1;
    }

    int32_t expected_blocks[RUNTIME_MAX_WORKER] = {};
    int32_t expected_lanes[RUNTIME_MAX_WORKER] = {};
    if (!build_expected_core_layout(runtime, header->num_cores, expected_blocks, expected_lanes)) {
        std::remove(temporary_path.c_str());
        return -1;
    }

#if PTO_FDWIC_SHARED_MAP
    std::vector<FdwicSwimlaneStorageRecord> physical_scratch(max_core_records);
    std::vector<FdwicSwimlaneRecord> decoded_scratch(max_core_records);
    std::vector<FdwicSharedSubmitClaimRecord> submit_claim_scratch(kFdwicSharedTracePhase1TaskCount);
    std::vector<FdwicSwimlaneRecord> logical_scratch;
    std::vector<uint32_t> winner_counts(kFdwicSharedTracePhase1TaskCount);
    std::vector<std::array<uint32_t, kFdwicSharedClaimTournamentMaxGroups>>
        root_group_counts(kFdwicSharedTracePhase1TaskCount);
#else
    FdwicSwimlaneRecord *scratch = nullptr;
    if (max_core_records != 0) {
        const size_t scratch_bytes = static_cast<size_t>(max_core_records) * sizeof(FdwicSwimlaneRecord);
        scratch = static_cast<FdwicSwimlaneRecord *>(std::aligned_alloc(alignof(FdwicSwimlaneRecord), scratch_bytes));
        if (scratch == nullptr) {
            LOG_ERROR("cannot allocate fdwic swimlane per-core scratch: %zu bytes", scratch_bytes);
            std::remove(temporary_path.c_str());
            return -1;
        }
    }
#endif

    std::ofstream out(temporary_path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        LOG_ERROR("cannot open fdwic swimlane temporary output %s: %s", temporary_path.c_str(), std::strerror(errno));
#if !PTO_FDWIC_SHARED_MAP
        std::free(scratch);
#endif
        std::remove(temporary_path.c_str());
        return -1;
    }
#if PTO_FDWIC_SHARED_MAP
    auto fail_export = [&out, &temporary_path]() {
        out.close();
        std::remove(temporary_path.c_str());
        return -1;
    };
#else
    auto fail_export = [&out, &scratch, &temporary_path]() {
        out.close();
        std::free(scratch);
        scratch = nullptr;
        std::remove(temporary_path.c_str());
        return -1;
    };
#endif

    out << "{\n";
    out << "  \"l2_swimlane_level\": " << level << ",\n";
    out << "  \"metadata\": {\n";
#if PTO_FDWIC_SHARED_MAP
    out << "    \"tensormap_mode\": \"shared\",\n";
#endif
    out << "    \"clock_freq_hz\": " << header->freq_hz << ",\n";
    out << "    \"num_cores\": " << header->num_cores << ",\n";
    out << "    \"trace_schema_version\": " << kFdwicSwimlaneTraceSchemaVersion << ",\n";
    out << "    \"raw_trace_version\": " << header->version << ",\n";
    out << "    \"records_per_core\": " << header->records_per_core << ",\n";
    out << "    \"record_size_bytes\": " << kFdwicSwimlaneRecordSizeBytes << ",\n";
    out << "    \"device_trace_bytes\": " << runtime->fdwic_swimlane_bytes_ << ",\n";
    out << "    \"core_types\": [";
    for (uint32_t c = 0; c < header->num_cores; c++) {
        if (c > 0) out << ", ";
        out << "\"" << core_type_name(runtime->workers[c].core_type) << "\"";
    }
    out << "],\n";
    out << "    \"atomic_site_names\": [";
    for (uint32_t site = 0; site < static_cast<uint32_t>(FdwicAtomicSite::Count); ++site) {
        if (site > 0) out << ", ";
        out << "\"" << atomic_site_name(site) << "\"";
    }
    out << "],\n";
    out << "    \"atomic_op_names\": [";
    constexpr uint32_t kAtomicOpCount = 5;
    for (uint32_t op = 0; op < kAtomicOpCount; ++op) {
        if (op > 0) out << ", ";
        out << "\"" << atomic_op_name(op) << "\"";
    }
    out << "],\n";
#if PTO_FDWIC_SHARED_MAP
    out << "    \"dcci_site_names\": [";
    for (uint32_t site = 0; site < static_cast<uint32_t>(FdwicDcciSite::Count); ++site) {
        if (site > 0) out << ", ";
        out << "\"" << dcci_site_name(site) << "\"";
    }
    out << "],\n";
    out << "    \"dcci_op_names\": [";
    for (uint32_t op = 0; op < static_cast<uint32_t>(FdwicDcciOp::Count); ++op) {
        if (op > 0) out << ", ";
        out << "\"" << dcci_op_name(op) << "\"";
    }
    out << "],\n";
#endif
    out << "    \"fdwic_summary\": {\n";
    out << "      \"records\": " << summary.records << ",\n";
    out << "      \"atomic_records\": " << summary.atomic_records << ",\n";
    out << "      \"clock_baseline_records\": " << summary.clock_baseline_records << ",\n";
    out << "      \"atomic_calls\": " << summary.atomic_calls << ",\n";
    out << "      \"batched_poll_calls\": " << summary.poll_calls << ",\n";
    out << "      \"poll_batch_records\": " << summary.poll_batch_records << ",\n";
#if PTO_FDWIC_SHARED_MAP
    out << "      \"dcci_records\": " << summary.dcci_records << ",\n";
    out << "      \"dcci_calls\": " << summary.dcci_calls << ",\n";
    out << "      \"dcci_lines\": " << summary.dcci_lines << ",\n";
#endif
    out << "      \"dropped_records\": " << summary.dropped_records << "\n";
    out << "    }\n";
    out << "  },\n";
    out << "  \"aicore_tasks\": [],\n";
    out << "  \"aicpu_tasks\": [],\n";
    out << "  \"aicpu_scheduler_phases\": [],\n";
    out << "  \"aicpu_orchestrator_phases\": [],\n";
    out << "  \"fdwic_events\": [";
    bool first = true;
    TraceSummary observed;
    const uint64_t records_base = runtime->fdwic_swimlane_dev_base_ + sizeof(FdwicSwimlaneHeader);
    for (uint32_t c = 0; c < header->num_cores; c++) {
        const uint32_t count = header->cores[c].count;
#if PTO_FDWIC_SHARED_MAP
        const uint64_t worker_base = records_base + static_cast<uint64_t>(c) * kFdwicSwimlaneWorkerBytes;
        const size_t submit_claim_bytes =
            static_cast<size_t>(kFdwicSharedTracePhase1TaskCount) * sizeof(FdwicSharedSubmitClaimRecord);
        if (runtime->host_api.copy_from_device(
                submit_claim_scratch.data(), reinterpret_cast<const void *>(worker_base), submit_claim_bytes
            ) != 0) {
            LOG_ERROR("fdwic shared swimlane core %u Submit/Claim D2H copy failed: bytes=%zu", c, submit_claim_bytes);
            return fail_export();
        }
        if (count != 0) {
            const size_t core_bytes = static_cast<size_t>(count) * sizeof(FdwicSwimlaneStorageRecord);
            const void *core_records_dev =
                reinterpret_cast<const void *>(worker_base + kFdwicSharedSubmitClaimBytesPerCore);
            if (runtime->host_api.copy_from_device(physical_scratch.data(), core_records_dev, core_bytes) != 0) {
                LOG_ERROR(
                    "fdwic shared swimlane core %u generic D2H copy failed: records=%u bytes=%zu", c, count, core_bytes
                );
                return fail_export();
            }
        }
        const uint64_t decode_anchor = submit_claim_scratch[0].submit_begin;
        if (decode_anchor == 0) {
            LOG_ERROR("fdwic shared swimlane core %u has no compact clock anchor", c);
            return fail_export();
        }
        for (uint32_t index = 0; index < count; ++index) {
            if (!decode_shared_compact_record(physical_scratch[index], decode_anchor, decoded_scratch[index])) {
                const FdwicCompactSwimlaneRecord &compact = physical_scratch[index];
                LOG_ERROR(
                    "fdwic shared swimlane core %u has invalid compact record %u: "
                    "start_low=0x%08x end_low=0x%08x flags=0x%08x packed=0x%08x anchor=%llu",
                    c, index, compact.start_cycle_low, compact.end_cycle_low, compact.flags, compact.packed,
                    static_cast<unsigned long long>(decode_anchor)
                );
                return fail_export();
            }
        }
        if (!expand_shared_records(
                expected_lanes[c] == 0, expected_blocks[c],
                decoded_scratch.data(), count,
                submit_claim_scratch.data(), logical_scratch
            )) {
            LOG_ERROR(
                "fdwic shared swimlane core %u schema-v5 expansion failed: generic_records=%u",
                c, count
            );
            return fail_export();
        }
        if (!validate_and_write_shared_core(
                header, logical_scratch.data(),
                static_cast<uint32_t>(logical_scratch.size()), c,
                expected_blocks[c], expected_lanes[c], level, out, first,
                observed, winner_counts, root_group_counts
            )) {
            LOG_ERROR(
                "fdwic shared swimlane core %u schema-v5 validation failed: logical_records=%zu",
                c, logical_scratch.size()
            );
            return fail_export();
        }
#else
        if (count != 0) {
            const uint64_t core_offset =
                static_cast<uint64_t>(c) * header->records_per_core * sizeof(FdwicSwimlaneRecord);
            const void *core_records_dev = reinterpret_cast<const void *>(records_base + core_offset);
            const size_t core_bytes = static_cast<size_t>(count) * sizeof(FdwicSwimlaneRecord);
            if (runtime->host_api.copy_from_device(scratch, core_records_dev, core_bytes) != 0) {
                LOG_ERROR("fdwic swimlane core %u D2H copy failed: records=%u bytes=%zu", c, count, core_bytes);
                return fail_export();
            }
        }
        if (!validate_and_write_core(
                header, scratch, c, expected_blocks[c], expected_lanes[c], level, out, first, observed
            )) {
            return fail_export();
        }
#endif
        if (!out) {
            LOG_ERROR("failed while writing fdwic swimlane core %u to %s", c, temporary_path.c_str());
            return fail_export();
        }
    }
#if PTO_FDWIC_SHARED_MAP
    for (uint32_t task_id = 0; task_id < kFdwicSharedTracePhase1TaskCount; ++task_id) {
        if (winner_counts[task_id] != 1) {
            LOG_ERROR(
                "fdwic shared swimlane task %u must have exactly one global winner; got %u",
                task_id, winner_counts[task_id]
            );
            return fail_export();
        }
        if (level >= kFdwicAtomicSwimlaneLevel) {
            const uint32_t groups =
                shared_claim_tournament_groups(shared_pa_task_kind(task_id));
            for (uint32_t group = 0; group < groups; ++group) {
                if (root_group_counts[task_id][group] != 1) {
                    LOG_ERROR(
                        "fdwic shared Claim tournament task %u group %u must have exactly one root contender; got %u",
                        task_id, group, root_group_counts[task_id][group]
                    );
                    return fail_export();
                }
            }
            for (uint32_t group = groups; group < kFdwicSharedClaimTournamentMaxGroups; ++group) {
                if (root_group_counts[task_id][group] != 0) {
                    LOG_ERROR(
                        "fdwic shared Claim tournament task %u inactive group %u has %u root contenders",
                        task_id, group, root_group_counts[task_id][group]
                    );
                    return fail_export();
                }
            }
        }
    }
#endif
    const bool summary_closed =
        observed.records == summary.records && observed.atomic_records == summary.atomic_records &&
        observed.clock_baseline_records == summary.clock_baseline_records &&
        observed.atomic_calls == summary.atomic_calls && observed.poll_calls == summary.poll_calls &&
        observed.poll_batch_records == summary.poll_batch_records &&
#if PTO_FDWIC_SHARED_MAP
        observed.dcci_records == summary.dcci_records && observed.dcci_calls == summary.dcci_calls &&
        observed.dcci_lines == summary.dcci_lines &&
#endif
        observed.dropped_records == summary.dropped_records;
    if (!summary_closed) {
        LOG_ERROR(
            "fdwic swimlane summary closure failed: records=%llu/%llu atomic=%llu/%llu clock=%llu/%llu "
            "calls=%llu/%llu poll_calls=%llu/%llu poll_batches=%llu/%llu dropped=%llu/%llu",
            static_cast<unsigned long long>(observed.records), static_cast<unsigned long long>(summary.records),
            static_cast<unsigned long long>(observed.atomic_records),
            static_cast<unsigned long long>(summary.atomic_records),
            static_cast<unsigned long long>(observed.clock_baseline_records),
            static_cast<unsigned long long>(summary.clock_baseline_records),
            static_cast<unsigned long long>(observed.atomic_calls),
            static_cast<unsigned long long>(summary.atomic_calls), static_cast<unsigned long long>(observed.poll_calls),
            static_cast<unsigned long long>(summary.poll_calls),
            static_cast<unsigned long long>(observed.poll_batch_records),
            static_cast<unsigned long long>(summary.poll_batch_records),
            static_cast<unsigned long long>(observed.dropped_records),
            static_cast<unsigned long long>(summary.dropped_records)
        );
        return fail_export();
    }
    if (!first) out << "\n  ";
    out << "]\n}\n";
    out.close();
#if !PTO_FDWIC_SHARED_MAP
    std::free(scratch);
    scratch = nullptr;
#endif
    if (!out) {
        LOG_ERROR("failed while writing fdwic swimlane output %s", temporary_path.c_str());
        std::remove(temporary_path.c_str());
        return -1;
    }
    if (std::rename(temporary_path.c_str(), path.c_str()) != 0) {
        LOG_ERROR("cannot finalize fdwic swimlane output %s: %s", path.c_str(), std::strerror(errno));
        std::remove(temporary_path.c_str());
        return -1;
    }
    if (should_print_trace_export()) {
        LOG_INFO_V0(
            "fdwic swimlane trace written to %s: records=%llu atomic=%llu clock=%llu calls=%llu poll_calls=%llu "
            "poll_batches=%llu dropped=%llu",
            path.c_str(), static_cast<unsigned long long>(summary.records),
            static_cast<unsigned long long>(summary.atomic_records),
            static_cast<unsigned long long>(summary.clock_baseline_records),
            static_cast<unsigned long long>(summary.atomic_calls), static_cast<unsigned long long>(summary.poll_calls),
            static_cast<unsigned long long>(summary.poll_batch_records),
            static_cast<unsigned long long>(summary.dropped_records)
        );
    }
    return 0;
}

extern "C" void fdwic_swimlane_host_finalize(Runtime *runtime) {
    if (runtime == nullptr) return;
    if (runtime->fdwic_swimlane_dev_allocation_ != 0 && runtime->host_api.device_free != nullptr) {
        runtime->host_api.device_free(reinterpret_cast<void *>(runtime->fdwic_swimlane_dev_allocation_));
    }
    if (runtime->fdwic_swimlane_host_shadow_ != nullptr) {
        std::free(runtime->fdwic_swimlane_host_shadow_);
    }
    runtime->fdwic_swimlane_host_shadow_ = nullptr;
    runtime->fdwic_swimlane_dev_allocation_ = 0;
    runtime->fdwic_swimlane_dev_base_ = 0;
    runtime->fdwic_swimlane_bytes_ = 0;
    runtime->fdwic_swimlane_num_cores_ = 0;
    runtime->fdwic_swimlane_records_per_core_ = 0;
    runtime->dist.swimlane_level = 0;
    runtime->dist.swimlane_base = 0;
    runtime->dist.swimlane_records_per_core = 0;
}

extern "C" int fdwic_perf_clock_host_init(Runtime *runtime, int num_cores, const char *output_prefix) {
    if (!perf_clock_requested()) return 0;
    if (runtime == nullptr) return -1;
    const bool storage_empty =
        runtime->fdwic_swimlane_host_shadow_ == nullptr && runtime->fdwic_swimlane_dev_allocation_ == 0 &&
        runtime->fdwic_swimlane_dev_base_ == 0 && runtime->fdwic_swimlane_bytes_ == 0 &&
        runtime->fdwic_swimlane_num_cores_ == 0 && runtime->fdwic_swimlane_records_per_core_ == 0 &&
        runtime->dist.swimlane_base == 0 && runtime->dist.swimlane_level == 0 &&
        runtime->dist.swimlane_records_per_core == 0;
    if (!storage_empty) {
        LOG_ERROR("fdwic perf-clock found non-empty diagnostic storage; refusing to overwrite a live allocation");
        return -1;
    }
    runtime->fdwic_swimlane_output_prefix_[0] = '\0';
    // This evidence path requires the complete 32-AIC + 64-AIV topology.
    // Reject other topologies instead of presenting partial data as a
    // per-core baseline.
    constexpr int kExpectedAic = 32;
    constexpr int kExpectedAiv = 64;
    constexpr int kExpectedCores = kExpectedAic + kExpectedAiv;
    if (num_cores != kExpectedCores || runtime->worker_count != kExpectedCores) {
        LOG_ERROR(
            "fdwic perf-clock requires 96 workers (32 AIC + 64 AIV): num_cores=%d worker_count=%d", num_cores,
            runtime->worker_count
        );
        return -1;
    }
    if (runtime->host_api.device_malloc == nullptr || runtime->host_api.device_free == nullptr ||
        runtime->host_api.copy_to_device == nullptr || runtime->host_api.copy_from_device == nullptr) {
        return -1;
    }

    const std::string prefix = output_prefix == nullptr || output_prefix[0] == '\0' ? "." : output_prefix;
    if (prefix.size() >= sizeof(runtime->fdwic_swimlane_output_prefix_)) {
        LOG_ERROR("fdwic perf-clock output prefix is too long: %zu bytes", prefix.size());
        return -1;
    }
    if (!prepare_output_directory(prefix)) return -1;
    const std::string path = perf_clock_output_path_from_prefix(prefix);
    if (!remove_output_if_present(path) || !remove_output_if_present(path + ".tmp")) return -1;

    constexpr uint64_t kDeviceAlignment = 64;
    constexpr uint64_t bytes = sizeof(FdwicSwimlaneHeader);
    void *host_shadow = std::aligned_alloc(64, sizeof(FdwicSwimlaneHeader));
    if (host_shadow == nullptr) return -1;
    std::memset(host_shadow, 0, sizeof(FdwicSwimlaneHeader));
    auto *header = reinterpret_cast<FdwicSwimlaneHeader *>(host_shadow);
    header->magic = kFdwicSwimlaneMagic;
    header->version = kFdwicSwimlaneVersion;
    header->num_cores = kExpectedCores;
    header->records_per_core = 0;
    header->freq_hz = PLATFORM_PROF_SYS_CNT_FREQ;

    void *dev_allocation = runtime->host_api.device_malloc(static_cast<size_t>(bytes + (kDeviceAlignment - 1)));
    if (dev_allocation == nullptr) {
        std::free(host_shadow);
        return -1;
    }
    const uintptr_t dev_base =
        (reinterpret_cast<uintptr_t>(dev_allocation) + (kDeviceAlignment - 1)) & ~(kDeviceAlignment - 1);
    if (runtime->host_api.copy_to_device(reinterpret_cast<void *>(dev_base), host_shadow, sizeof(FdwicSwimlaneHeader)) !=
        0) {
        runtime->host_api.device_free(dev_allocation);
        std::free(host_shadow);
        return -1;
    }

    runtime->fdwic_swimlane_host_shadow_ = host_shadow;
    runtime->fdwic_swimlane_dev_allocation_ = reinterpret_cast<uint64_t>(dev_allocation);
    runtime->fdwic_swimlane_dev_base_ = dev_base;
    runtime->fdwic_swimlane_bytes_ = bytes;
    runtime->fdwic_swimlane_num_cores_ = kExpectedCores;
    runtime->fdwic_swimlane_records_per_core_ = 0;
    std::memcpy(runtime->fdwic_swimlane_output_prefix_, prefix.c_str(), prefix.size() + 1);
    // Reuse the handoff address for the fixed header only. Keeping
    // level/records at zero proves this is not a level 1..4 swimlane.
    runtime->dist.swimlane_base = dev_base;
    runtime->dist.swimlane_level = 0;
    runtime->dist.swimlane_records_per_core = 0;
    return 1;
}

extern "C" int fdwic_perf_clock_host_export(Runtime *runtime) {
    if (runtime == nullptr || runtime->fdwic_swimlane_host_shadow_ == nullptr ||
        runtime->fdwic_swimlane_dev_base_ == 0) {
        return 0;
    }
    if (runtime->host_api.copy_from_device(
            runtime->fdwic_swimlane_host_shadow_, reinterpret_cast<void *>(runtime->fdwic_swimlane_dev_base_),
            sizeof(FdwicSwimlaneHeader)
        ) != 0) {
        LOG_ERROR("fdwic perf-clock header D2H copy failed");
        return -1;
    }

    const auto *header = reinterpret_cast<const FdwicSwimlaneHeader *>(runtime->fdwic_swimlane_host_shadow_);
    constexpr uint32_t kExpectedAic = 32;
    constexpr uint32_t kExpectedAiv = 64;
    constexpr uint32_t kExpectedCores = kExpectedAic + kExpectedAiv;
    const bool header_valid =
        header->magic == kFdwicSwimlaneMagic && header->version == kFdwicSwimlaneVersion &&
        header->num_cores == kExpectedCores && header->records_per_core == 0 &&
        header->freq_hz == PLATFORM_PROF_SYS_CNT_FREQ && runtime->worker_count == static_cast<int>(kExpectedCores) &&
        runtime->fdwic_swimlane_bytes_ == sizeof(FdwicSwimlaneHeader) && runtime->dist.swimlane_level == 0 &&
        runtime->dist.swimlane_records_per_core == 0 &&
        runtime->dist.swimlane_base == runtime->fdwic_swimlane_dev_base_;
    if (!header_valid) {
        LOG_ERROR(
            "fdwic perf-clock invalid header/state: magic=0x%08x version=%u cores=%u records=%u freq=%llu bytes=%llu",
            header->magic, header->version, header->num_cores, header->records_per_core,
            static_cast<unsigned long long>(header->freq_hz),
            static_cast<unsigned long long>(runtime->fdwic_swimlane_bytes_)
        );
        return -1;
    }

    const bool kernel_profile = perf_clock_kernel_requested();
    const char *profile_name = kernel_profile ? "perf-clock-kernel" : "perf-clock";
    const uint32_t scheduler_mode_raw = runtime->fdwic_build_identity.scheduler_mode;
    if (scheduler_mode_raw > static_cast<uint32_t>(FdwicSchedulerMode::SimtCrossCoreDag)) {
        LOG_ERROR("fdwic %s has invalid scheduler mode %u", profile_name, scheduler_mode_raw);
        return -1;
    }
    const auto scheduler_mode = static_cast<FdwicSchedulerMode>(scheduler_mode_raw);
    const bool cross_core_e2e = scheduler_mode != FdwicSchedulerMode::SameCore;
    const uint32_t expected_stored_mode = kernel_profile
                                              ? (cross_core_e2e ? kFdwicPerfClockKernelCrossCoreE2eMode
                                                                : kFdwicPerfClockKernelMode)
                                              : (cross_core_e2e ? kFdwicPerfClockCrossCoreE2eMode
                                                                : kFdwicPerfClockMode);
    int32_t expected_blocks[RUNTIME_MAX_WORKER] = {};
    int32_t expected_lanes[RUNTIME_MAX_WORKER] = {};
    if (!build_expected_core_layout(runtime, header->num_cores, expected_blocks, expected_lanes)) return -1;

    uint32_t expected_submits = 0;
    uint32_t replay_cores = 0;
    uint32_t builder_cores = 0;
    uint64_t global_start = std::numeric_limits<uint64_t>::max();
    uint64_t global_end = 0;
    PerfClockGroupAggregate aic;
    PerfClockGroupAggregate aiv;

    for (uint32_t core_id = 0; core_id < header->num_cores; ++core_id) {
        const FdwicSwimlaneCoreState &core = header->cores[core_id];
        uint64_t first_submit_start = 0;
        uint64_t last_submit_end = 0;
        uint64_t kernel_elapsed_ticks = 0;
        uint32_t submit_count = 0;
        uint32_t expected_submit_count = 0;
        uint32_t kernel_calls = 0;
        uint32_t stored_mode = 0;
        uint32_t final_seen = 0;
        bool diagnostic_fields_valid = false;
        if (kernel_profile) {
            const FdwicPerfClockKernelCoreData &clock = core.perf_clock_kernel;
            first_submit_start = clock.first_submit_start;
            last_submit_end = clock.last_submit_end;
            submit_count = clock.submit_count;
            expected_submit_count = clock.expected_submit_count;
            kernel_elapsed_ticks = clock.kernel_elapsed_ticks;
            kernel_calls = core.count;
            stored_mode = core.poll_batch_records;
            final_seen = last_submit_end != 0 ? 1U : 0U;
            const bool call_time_shape =
                (kernel_calls == 0 && kernel_elapsed_ticks == 0) || (kernel_calls != 0 && kernel_elapsed_ticks != 0);
            diagnostic_fields_valid = core.dropped == 0 && core.atomic_calls == 0 && core.poll_calls == 0 &&
                                      stored_mode == expected_stored_mode && call_time_shape;
        } else {
            const FdwicPerfClockCoreData &clock = core.perf_clock;
            first_submit_start = clock.first_submit_start;
            last_submit_end = clock.last_submit_end;
            submit_count = clock.submit_count;
            expected_submit_count = clock.expected_submit_count;
            stored_mode = clock.mode;
            final_seen = clock.final_seen;
            diagnostic_fields_valid = core.count == 0 && core.dropped == 0 && core.atomic_calls == 0 &&
                                      core.poll_calls == 0 && core.poll_batch_records == 0 &&
                                      stored_mode == expected_stored_mode && final_seen == 1;
        }

        const bool identity_valid = core.core_idx == static_cast<int32_t>(core_id) &&
                                    core.block_id == expected_blocks[core_id] && core.lane == expected_lanes[core_id];
        const bool simt_builder = is_simt_builder_worker(
            scheduler_mode, runtime->workers[core_id].core_type, expected_blocks[core_id], expected_lanes[core_id]
        );
        const bool replay_worker = !simt_builder;
        const bool submit_closure_valid = replay_worker
                                              ? expected_submit_count != 0 && submit_count == expected_submit_count
                                              : expected_submit_count == 0 && submit_count == 0;
        const bool window_order_valid =
            (cross_core_e2e || kernel_profile) ? last_submit_end > first_submit_start
                                               : last_submit_end >= first_submit_start;
        const bool clock_valid = submit_closure_valid && first_submit_start != 0 && window_order_valid &&
                                 kernel_elapsed_ticks <= last_submit_end - first_submit_start;
        if (!identity_valid || !diagnostic_fields_valid || !clock_valid) {
            LOG_ERROR(
                "fdwic %s core %u failed closure: core=%d block=%d/%d lane=%d/%d count=%u/%u "
                "start=%llu end=%llu mode=%u final=%u kernel_calls=%u kernel_ticks=%llu status=%u reserved=%u/%u "
                "checks(identity/diagnostic/submit/window/clock)=%u/%u/%u/%u/%u role=%s",
                profile_name, core_id, core.core_idx, core.block_id, expected_blocks[core_id], core.lane,
                expected_lanes[core_id], submit_count, expected_submit_count,
                static_cast<unsigned long long>(first_submit_start), static_cast<unsigned long long>(last_submit_end),
                stored_mode, final_seen, kernel_calls, static_cast<unsigned long long>(kernel_elapsed_ticks),
                core.dropped, core.atomic_calls, core.poll_calls, identity_valid ? 1U : 0U,
                diagnostic_fields_valid ? 1U : 0U, submit_closure_valid ? 1U : 0U, window_order_valid ? 1U : 0U,
                clock_valid ? 1U : 0U, replay_worker ? "replay" : "builder"
            );
            return -1;
        }
        if (replay_worker) {
            ++replay_cores;
            if (expected_submits == 0) expected_submits = expected_submit_count;
            if (expected_submit_count != expected_submits) {
                LOG_ERROR(
                    "fdwic %s expected Submit count differs across replay cores: core=%u expected=%u reference=%u",
                    profile_name, core_id, expected_submit_count, expected_submits
                );
                return -1;
            }
        } else {
            ++builder_cores;
        }
        global_start = std::min(global_start, first_submit_start);
        global_end = std::max(global_end, last_submit_end);
        const uint64_t elapsed = last_submit_end - first_submit_start;
        if (runtime->workers[core_id].core_type == CoreType::AIC) {
            aic.add(elapsed, kernel_elapsed_ticks, kernel_calls);
        } else if (runtime->workers[core_id].core_type == CoreType::AIV) {
            aiv.add(elapsed, kernel_elapsed_ticks, kernel_calls);
        } else {
            LOG_ERROR("fdwic %s core %u has invalid core type", profile_name, core_id);
            return -1;
        }
    }
    const uint32_t expected_builder_cores = expected_simt_builder_workers(scheduler_mode, aic.cores);
    const uint32_t expected_replay_cores = kExpectedCores - expected_builder_cores;
    if (aic.cores != kExpectedAic || aiv.cores != kExpectedAiv || replay_cores != expected_replay_cores ||
        builder_cores != expected_builder_cores || expected_submits == 0 ||
        global_start == std::numeric_limits<uint64_t>::max() || global_end < global_start) {
        LOG_ERROR(
            "fdwic %s topology/global closure failed: AIC=%u/%u AIV=%u/%u replay=%u/%u builder=%u/%u "
            "start=%llu end=%llu",
            profile_name, aic.cores, kExpectedAic, aiv.cores, kExpectedAiv, replay_cores, expected_replay_cores,
            builder_cores, expected_builder_cores, static_cast<unsigned long long>(global_start),
            static_cast<unsigned long long>(global_end)
        );
        return -1;
    }

    uint64_t min_kernel_calls = 0;
    uint64_t max_kernel_calls = 0;
    if (kernel_profile && !cross_core_e2e) {
        if (aic.elapsed_sum == 0 || aiv.elapsed_sum == 0) {
            LOG_ERROR(
                "fdwic perf-clock-kernel requires non-zero role elapsed sums: AIC=%llu AIV=%llu",
                static_cast<unsigned long long>(aic.elapsed_sum), static_cast<unsigned long long>(aiv.elapsed_sum)
            );
            return -1;
        }
        if (expected_submits % 5U != 0) {
            LOG_ERROR("fdwic perf-clock-kernel requires 5*batch Submit shape: expected=%u", expected_submits);
            return -1;
        }
        const uint64_t batches = expected_submits / 5U;
        const uint64_t max_role_calls = 2U * batches;
        min_kernel_calls = batches;
        max_kernel_calls = 4U * batches;
        // In the current PA shape, QK has no fanin and runs no later than the
        // following SF Submit's EfDrain, so AIC must observe at least one call
        // per batch. Other Kernels may land in FinalDrain; validate role and
        // total-call bounds rather than requiring exactly 4*batches.
        if (aic.kernel_calls_sum < batches || aic.kernel_calls_sum > max_role_calls ||
            aiv.kernel_calls_sum > max_role_calls || aic.kernel_calls_sum + aiv.kernel_calls_sum > max_kernel_calls) {
            LOG_ERROR(
                "fdwic perf-clock-kernel call closure failed: AIC=%llu/[%llu,%llu] AIV=%llu/[0,%llu] ALL=%llu/["
                "%llu,%llu]",
                static_cast<unsigned long long>(aic.kernel_calls_sum), static_cast<unsigned long long>(batches),
                static_cast<unsigned long long>(max_role_calls), static_cast<unsigned long long>(aiv.kernel_calls_sum),
                static_cast<unsigned long long>(max_role_calls),
                static_cast<unsigned long long>(aic.kernel_calls_sum + aiv.kernel_calls_sum),
                static_cast<unsigned long long>(batches), static_cast<unsigned long long>(max_kernel_calls)
            );
            return -1;
        }
    }

    const std::string path = perf_clock_output_path_from_prefix(runtime->fdwic_swimlane_output_prefix_);
    const std::string temporary_path = path + ".tmp";
    if (!remove_output_if_present(temporary_path)) return -1;
    std::ofstream out(temporary_path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        LOG_ERROR("cannot open fdwic perf-clock output %s: %s", temporary_path.c_str(), std::strerror(errno));
        return -1;
    }
    const uint64_t global_elapsed = global_end - global_start;
    out << "{\n";
    const char *schema = cross_core_e2e
                             ? (kernel_profile ? "fdwic-cross-core-e2e-clock-kernel-v1"
                                               : "fdwic-cross-core-e2e-clock-v1")
                             : (kernel_profile ? "fdwic-perf-clock-kernel-v1" : "fdwic-perf-clock-v1");
    out << "  \"schema\": \"" << schema << "\",\n";
    out << "  \"mode\": \"" << profile_name << "\",\n";
    if (cross_core_e2e) {
        out << "  \"scheduler_mode\": \"" << scheduler_mode_name(scheduler_mode) << "\",\n";
        out << "  \"scheduler_mode_id\": " << scheduler_mode_raw << ",\n";
        out << "  \"boundary\": \"startup_sync_begin_to_final_drain_end\",\n";
    }
    out << "  \"clock_freq_hz\": " << header->freq_hz << ",\n";
    out << "  \"device_header_bytes\": " << sizeof(FdwicSwimlaneHeader) << ",\n";
    out << "  \"num_cores\": " << header->num_cores << ",\n";
    out << "  \"aic_cores\": " << aic.cores << ",\n";
    out << "  \"aiv_cores\": " << aiv.cores << ",\n";
    if (cross_core_e2e) {
        out << "  \"replay_cores\": " << replay_cores << ",\n";
        out << "  \"builder_cores\": " << builder_cores << ",\n";
        out << "  \"expected_submits_per_replay_core\": " << expected_submits << ",\n";
        out << "  \"global_startup_begin\": " << global_start << ",\n";
        out << "  \"global_final_drain_end\": " << global_end << ",\n";
        out << "  \"global_e2e_span_ticks\": " << global_elapsed << ",\n";
        out << "  \"global_e2e_span_us\": "
            << static_cast<double>(global_elapsed) * 1000000.0 / static_cast<double>(header->freq_hz) << ",\n";
    } else {
        out << "  \"expected_submits_per_core\": " << expected_submits << ",\n";
        out << "  \"global_first_submit_start\": " << global_start << ",\n";
        out << "  \"global_last_submit_end\": " << global_end << ",\n";
        out << "  \"global_submit_span_ticks\": " << global_elapsed << ",\n";
        out << "  \"global_submit_span_us\": "
            << static_cast<double>(global_elapsed) * 1000000.0 / static_cast<double>(header->freq_hz) << ",\n";
    }
    if (kernel_profile) {
        const uint64_t elapsed_sum = aic.elapsed_sum + aiv.elapsed_sum;
        const uint64_t kernel_sum = aic.kernel_sum + aiv.kernel_sum;
        out << "  \"kernel_boundary\": \""
            << (cross_core_e2e ? "linked_kernel_call_within_per_core_startup_to_final_drain_window"
                               : "linked_kernel_call_within_per_core_submit_window")
            << "\",\n";
        out << "  \"kernel_calls\": " << aic.kernel_calls_sum + aiv.kernel_calls_sum << ",\n";
        if (!cross_core_e2e) {
            out << "  \"min_kernel_calls_in_window\": " << min_kernel_calls << ",\n";
            out << "  \"max_kernel_calls_in_window\": " << max_kernel_calls << ",\n";
        }
        out << "  \"kernel_elapsed_ticks_sum\": " << kernel_sum << ",\n";
        out << "  \"non_kernel_residual_ticks_sum\": " << elapsed_sum - kernel_sum << ",\n";
        out << "  \"kernel_core_time_share\": " << static_cast<double>(kernel_sum) / elapsed_sum << ",\n";
    }
    out << "  \"groups\": {\n";
    if (kernel_profile) {
        out << "    \"aic\": {\"cores\": " << aic.cores << ", \"elapsed_min_ticks\": " << aic.elapsed_min
            << ", \"elapsed_max_ticks\": " << aic.elapsed_max << ", \"elapsed_sum_ticks\": " << aic.elapsed_sum
            << ", \"elapsed_mean_ticks\": " << static_cast<double>(aic.elapsed_sum) / aic.cores
            << ", \"kernel_min_ticks\": " << aic.kernel_min << ", \"kernel_max_ticks\": " << aic.kernel_max
            << ", \"kernel_sum_ticks\": " << aic.kernel_sum
            << ", \"kernel_mean_ticks\": " << static_cast<double>(aic.kernel_sum) / aic.cores
            << ", \"kernel_calls_min\": " << aic.kernel_calls_min << ", \"kernel_calls_max\": " << aic.kernel_calls_max
            << ", \"kernel_calls_sum\": " << aic.kernel_calls_sum << ", \"residual_min_ticks\": " << aic.residual_min
            << ", \"residual_max_ticks\": " << aic.residual_max << ", \"residual_sum_ticks\": " << aic.residual_sum
            << ", \"residual_mean_ticks\": " << static_cast<double>(aic.residual_sum) / aic.cores
            << ", \"kernel_core_time_share\": " << static_cast<double>(aic.kernel_sum) / aic.elapsed_sum << "},\n";
        out << "    \"aiv\": {\"cores\": " << aiv.cores << ", \"elapsed_min_ticks\": " << aiv.elapsed_min
            << ", \"elapsed_max_ticks\": " << aiv.elapsed_max << ", \"elapsed_sum_ticks\": " << aiv.elapsed_sum
            << ", \"elapsed_mean_ticks\": " << static_cast<double>(aiv.elapsed_sum) / aiv.cores
            << ", \"kernel_min_ticks\": " << aiv.kernel_min << ", \"kernel_max_ticks\": " << aiv.kernel_max
            << ", \"kernel_sum_ticks\": " << aiv.kernel_sum
            << ", \"kernel_mean_ticks\": " << static_cast<double>(aiv.kernel_sum) / aiv.cores
            << ", \"kernel_calls_min\": " << aiv.kernel_calls_min << ", \"kernel_calls_max\": " << aiv.kernel_calls_max
            << ", \"kernel_calls_sum\": " << aiv.kernel_calls_sum << ", \"residual_min_ticks\": " << aiv.residual_min
            << ", \"residual_max_ticks\": " << aiv.residual_max << ", \"residual_sum_ticks\": " << aiv.residual_sum
            << ", \"residual_mean_ticks\": " << static_cast<double>(aiv.residual_sum) / aiv.cores
            << ", \"kernel_core_time_share\": " << static_cast<double>(aiv.kernel_sum) / aiv.elapsed_sum << "}\n";
    } else {
        out << "    \"aic\": {\"min_ticks\": " << aic.elapsed_min << ", \"max_ticks\": " << aic.elapsed_max
            << ", \"mean_ticks\": " << static_cast<double>(aic.elapsed_sum) / aic.cores << "},\n";
        out << "    \"aiv\": {\"min_ticks\": " << aiv.elapsed_min << ", \"max_ticks\": " << aiv.elapsed_max
            << ", \"mean_ticks\": " << static_cast<double>(aiv.elapsed_sum) / aiv.cores << "}\n";
    }
    out << "  },\n";
    out << "  \"cores\": [\n";
    for (uint32_t core_id = 0; core_id < header->num_cores; ++core_id) {
        const FdwicSwimlaneCoreState &core = header->cores[core_id];
        const uint64_t first_submit_start =
            kernel_profile ? core.perf_clock_kernel.first_submit_start : core.perf_clock.first_submit_start;
        const uint64_t last_submit_end =
            kernel_profile ? core.perf_clock_kernel.last_submit_end : core.perf_clock.last_submit_end;
        const uint32_t submit_count =
            kernel_profile ? core.perf_clock_kernel.submit_count : core.perf_clock.submit_count;
        const uint32_t expected_submit_count = kernel_profile ? core.perf_clock_kernel.expected_submit_count
                                                               : core.perf_clock.expected_submit_count;
        const uint64_t elapsed = last_submit_end - first_submit_start;
        const bool simt_builder = is_simt_builder_worker(
            scheduler_mode, runtime->workers[core_id].core_type, expected_blocks[core_id], expected_lanes[core_id]
        );
        out << "    {\"core_id\": " << core_id << ", \"core_type\": \""
            << core_type_name(runtime->workers[core_id].core_type) << "\", \"block_id\": " << core.block_id
            << ", \"lane\": " << core.lane << ", \"submit_count\": " << submit_count;
        if (cross_core_e2e) {
            out << ", \"expected_submit_count\": " << expected_submit_count << ", \"worker_role\": \""
                << (simt_builder ? "builder" : "replay") << "\", \"startup_begin\": " << first_submit_start
                << ", \"final_drain_end\": " << last_submit_end << ", \"elapsed_ticks\": " << elapsed;
        } else {
            out << ", \"first_submit_start\": " << first_submit_start << ", \"last_submit_end\": "
                << last_submit_end << ", \"elapsed_ticks\": " << elapsed;
        }
        if (kernel_profile) {
            const uint64_t kernel_elapsed_ticks = core.perf_clock_kernel.kernel_elapsed_ticks;
            out << ", \"kernel_elapsed_ticks\": " << kernel_elapsed_ticks << ", \"kernel_calls\": " << core.count
                << ", \"non_kernel_residual_ticks\": " << elapsed - kernel_elapsed_ticks;
        }
        out << "}" << (core_id + 1 == header->num_cores ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    out.close();
    if (!out) {
        LOG_ERROR("failed while writing fdwic perf-clock output %s", temporary_path.c_str());
        std::remove(temporary_path.c_str());
        return -1;
    }
    if (std::rename(temporary_path.c_str(), path.c_str()) != 0) {
        LOG_ERROR("cannot finalize fdwic perf-clock output %s: %s", path.c_str(), std::strerror(errno));
        std::remove(temporary_path.c_str());
        return -1;
    }
    if (kernel_profile) {
        if (cross_core_e2e) {
            LOG_INFO_V0(
                "fdwic perf-clock-kernel written to %s: scheduler=%s cores=%u replay=%u builder=%u "
                "submits/replay-core=%u startup-to-final-drain=%.3fus kernel_calls=%llu kernel_ticks=%llu",
                path.c_str(), scheduler_mode_name(scheduler_mode), header->num_cores, replay_cores, builder_cores,
                expected_submits,
                static_cast<double>(global_elapsed) * 1000000.0 / static_cast<double>(header->freq_hz),
                static_cast<unsigned long long>(aic.kernel_calls_sum + aiv.kernel_calls_sum),
                static_cast<unsigned long long>(aic.kernel_sum + aiv.kernel_sum)
            );
        } else {
            LOG_INFO_V0(
                "fdwic perf-clock-kernel written to %s: cores=%u AIC=%u AIV=%u submits/core=%u span=%.3fus "
                "kernel_calls=%llu/[%llu,%llu] kernel_ticks=%llu",
                path.c_str(), header->num_cores, aic.cores, aiv.cores, expected_submits,
                static_cast<double>(global_elapsed) * 1000000.0 / static_cast<double>(header->freq_hz),
                static_cast<unsigned long long>(aic.kernel_calls_sum + aiv.kernel_calls_sum),
                static_cast<unsigned long long>(min_kernel_calls), static_cast<unsigned long long>(max_kernel_calls),
                static_cast<unsigned long long>(aic.kernel_sum + aiv.kernel_sum)
            );
        }
    } else {
        if (cross_core_e2e) {
            LOG_INFO_V0(
                "fdwic perf-clock written to %s: scheduler=%s cores=%u replay=%u builder=%u "
                "submits/replay-core=%u startup-to-final-drain=%.3fus",
                path.c_str(), scheduler_mode_name(scheduler_mode), header->num_cores, replay_cores, builder_cores,
                expected_submits,
                static_cast<double>(global_elapsed) * 1000000.0 / static_cast<double>(header->freq_hz)
            );
        } else {
            LOG_INFO_V0(
                "fdwic perf-clock written to %s: cores=%u AIC=%u AIV=%u submits/core=%u span=%.3fus", path.c_str(),
                header->num_cores, aic.cores, aiv.cores, expected_submits,
                static_cast<double>(global_elapsed) * 1000000.0 / static_cast<double>(header->freq_hz)
            );
        }
    }
    return 0;
}

extern "C" void fdwic_perf_clock_host_finalize(Runtime *runtime) { fdwic_swimlane_host_finalize(runtime); }
