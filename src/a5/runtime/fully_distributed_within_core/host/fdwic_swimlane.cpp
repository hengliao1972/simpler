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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>

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
    case FdwicSwimlanePhase::LoserReplay:
        return "LoserReplay";
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
    };
    static_assert(
        sizeof(names) / sizeof(names[0]) == static_cast<uint32_t>(FdwicAtomicSite::Count),
        "atomic site name table must match the raw ABI"
    );
    return site < sizeof(names) / sizeof(names[0]) ? names[site] : "Unknown";
}

const char *atomic_op_name(uint32_t op) {
    static constexpr const char *names[] = {"Load", "Exchange", "FetchAdd", "FetchMax", "FetchSub"};
    return op < sizeof(names) / sizeof(names[0]) ? names[op] : "Unknown";
}

const char *core_type_name(CoreType core_type) {
    switch (core_type) {
    case CoreType::AIC:
        return "aic";
    case CoreType::AIV:
        return "aiv";
    }
    return "unknown";
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
        return fdwic_atomic_site_is_poll_batchable(site) && result_used && !return_ready && !value_zero &&
               payload > 0 && record.task_id == -1 && record.func_id == -1;
    }
    const bool expected_return_ready = result_used && !is_cpu_sim_trace();
    if (result_used != fdwic_atomic_site_result_used(site) || return_ready != expected_return_ready) return false;
    if (value_zero && op != static_cast<uint32_t>(FdwicAtomicOp::Load)) return false;
    if (payload != 0 && op != static_cast<uint32_t>(FdwicAtomicOp::FetchMax)) return false;
    return record.func_id == -1;
}

uint32_t atomic_record_call_count(const FdwicSwimlaneRecord &record) {
    return (record.flags & kFdwicAtomicPollBatch) != 0 ? record.flags >> kFdwicAtomicPollCountShift : 1U;
}

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
        return record.flags == 1 && record.aux < 4;
    case FdwicSwimlanePhase::RingBp:
        return record.flags == 0 && record.aux <= 1;
    case FdwicSwimlanePhase::Submit:
        return record.flags <= 1 && record.aux <= 1;
    case FdwicSwimlanePhase::Materialize:
    case FdwicSwimlanePhase::PrepareMap:
    case FdwicSwimlanePhase::Register:
        return record.flags == 0 && record.aux <= 1;
    case FdwicSwimlanePhase::Fanin:
        return record.flags == 0 && record.aux <= 16;
    case FdwicSwimlanePhase::Alloc:
    case FdwicSwimlanePhase::Build:
    case FdwicSwimlanePhase::Replay:
        // Schema-v4 reserves the legacy IDs for archived raw files but never
        // accepts newly produced overlapping lap records.
        return false;
    case FdwicSwimlanePhase::EfDrain:
        return record.flags == 0 && record.aux == 0;
    case FdwicSwimlanePhase::WinnerBuild:
    case FdwicSwimlanePhase::AllocComplete:
    case FdwicSwimlanePhase::LoserReplay:
        return record.flags == 0 && record.aux == 0;
    case FdwicSwimlanePhase::OrchestrationReplay:
    case FdwicSwimlanePhase::FinalDrain:
        return record.task_id == -1 && record.func_id == -1 && record.flags == 0 && record.aux == 0;
    case FdwicSwimlanePhase::Claim:
    case FdwicSwimlanePhase::Atomic:
    case FdwicSwimlanePhase::ClockBaseline:
        return true;
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
    const uint64_t expected_bytes = sizeof(FdwicSwimlaneHeader) + static_cast<uint64_t>(expected_cores) *
                                                                      expected_capacity * sizeof(FdwicSwimlaneRecord);
    const bool header_valid =
        header->magic == kFdwicSwimlaneMagic && header->version == kFdwicSwimlaneVersion && expected_cores > 0 &&
        expected_cores <= RUNTIME_MAX_WORKER && header->num_cores == expected_cores &&
        runtime->worker_count == static_cast<int>(expected_cores) && expected_capacity > 0 &&
        header->records_per_core == expected_capacity && header->freq_hz == PLATFORM_PROF_SYS_CNT_FREQ &&
        runtime->fdwic_swimlane_bytes_ == expected_bytes && runtime->dist.swimlane_level == level &&
        runtime->dist.swimlane_base == runtime->fdwic_swimlane_dev_base_ &&
        runtime->dist.swimlane_records_per_core == expected_capacity;
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
            (core_state.atomic_calls != 0 || core_state.poll_calls != 0 || core_state.poll_batch_records != 0)) {
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
        }
    }
    if (level >= kFdwicAtomicSwimlaneLevel) {
        summary.clock_baseline_records = 2 * static_cast<uint64_t>(expected_cores);
    }
    return true;
}

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

std::string output_path_from_prefix(const std::string &prefix) {
    if (!prefix.empty() && prefix.back() == '/') return prefix + "l2_swimlane_records.json";
    return prefix + "/l2_swimlane_records.json";
}

std::string output_path(const Runtime *runtime) {
    return output_path_from_prefix(runtime->fdwic_swimlane_output_prefix_);
}

std::string perf_clock_output_path_from_prefix(const std::string &prefix) {
    if (!prefix.empty() && prefix.back() == '/') return prefix + "fdwic_perf_clock_summary.json";
    return prefix + "/fdwic_perf_clock_summary.json";
}

bool perf_clock_requested() {
    const char *mode = std::getenv("PTO_FDWIC_PROFILE");
    return mode != nullptr && std::strcmp(mode, "perf-clock") == 0;
}

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
    const uint64_t bytes =
        sizeof(FdwicSwimlaneHeader) + static_cast<uint64_t>(num_cores) * records_per_core * sizeof(FdwicSwimlaneRecord);
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

    std::ofstream out(temporary_path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        LOG_ERROR("cannot open fdwic swimlane temporary output %s: %s", temporary_path.c_str(), std::strerror(errno));
        std::free(scratch);
        std::remove(temporary_path.c_str());
        return -1;
    }
    auto fail_export = [&out, &scratch, &temporary_path]() {
        out.close();
        std::free(scratch);
        scratch = nullptr;
        std::remove(temporary_path.c_str());
        return -1;
    };

    out << "{\n";
    out << "  \"l2_swimlane_level\": " << level << ",\n";
    out << "  \"metadata\": {\n";
    out << "    \"clock_freq_hz\": " << header->freq_hz << ",\n";
    out << "    \"num_cores\": " << header->num_cores << ",\n";
    out << "    \"trace_schema_version\": " << kFdwicSwimlaneTraceSchemaVersion << ",\n";
    out << "    \"raw_trace_version\": " << header->version << ",\n";
    out << "    \"records_per_core\": " << header->records_per_core << ",\n";
    out << "    \"record_size_bytes\": " << sizeof(FdwicSwimlaneRecord) << ",\n";
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
    for (uint32_t op = 0; op <= static_cast<uint32_t>(FdwicAtomicOp::FetchSub); ++op) {
        if (op > 0) out << ", ";
        out << "\"" << atomic_op_name(op) << "\"";
    }
    out << "],\n";
    out << "    \"fdwic_summary\": {\n";
    out << "      \"records\": " << summary.records << ",\n";
    out << "      \"atomic_records\": " << summary.atomic_records << ",\n";
    out << "      \"clock_baseline_records\": " << summary.clock_baseline_records << ",\n";
    out << "      \"atomic_calls\": " << summary.atomic_calls << ",\n";
    out << "      \"batched_poll_calls\": " << summary.poll_calls << ",\n";
    out << "      \"poll_batch_records\": " << summary.poll_batch_records << ",\n";
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
        if (!out) {
            LOG_ERROR("failed while writing fdwic swimlane core %u to %s", c, temporary_path.c_str());
            return fail_export();
        }
    }
    const bool summary_closed =
        observed.records == summary.records && observed.atomic_records == summary.atomic_records &&
        observed.clock_baseline_records == summary.clock_baseline_records &&
        observed.atomic_calls == summary.atomic_calls && observed.poll_calls == summary.poll_calls &&
        observed.poll_batch_records == summary.poll_batch_records &&
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
    std::free(scratch);
    scratch = nullptr;
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
    // 当前证据链只服务真实 PA 的 32 AIC + 64 AIV 全核 Case1/B1。
    // 其他拓扑直接拒绝，避免把部分核数据包装成“每核基线”。
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
    // 只借用已有 handoff 地址传输固定 header；level/records 保持 0，明确
    // 表示这不是 1..4 任一级泳道。
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

    int32_t expected_blocks[RUNTIME_MAX_WORKER] = {};
    int32_t expected_lanes[RUNTIME_MAX_WORKER] = {};
    if (!build_expected_core_layout(runtime, header->num_cores, expected_blocks, expected_lanes)) return -1;

    uint32_t aic_count = 0;
    uint32_t aiv_count = 0;
    uint32_t expected_submits = 0;
    uint64_t global_start = std::numeric_limits<uint64_t>::max();
    uint64_t global_end = 0;
    uint64_t aic_elapsed_sum = 0;
    uint64_t aiv_elapsed_sum = 0;
    uint64_t aic_elapsed_min = std::numeric_limits<uint64_t>::max();
    uint64_t aiv_elapsed_min = std::numeric_limits<uint64_t>::max();
    uint64_t aic_elapsed_max = 0;
    uint64_t aiv_elapsed_max = 0;

    for (uint32_t core_id = 0; core_id < header->num_cores; ++core_id) {
        const FdwicSwimlaneCoreState &core = header->cores[core_id];
        const FdwicPerfClockCoreData &clock = core.perf_clock;
        const bool identity_valid =
            core.core_idx == static_cast<int32_t>(core_id) && core.block_id == expected_blocks[core_id] &&
            core.lane == expected_lanes[core_id];
        const bool counters_clean = core.count == 0 && core.dropped == 0 && core.atomic_calls == 0 &&
                                    core.poll_calls == 0 && core.poll_batch_records == 0;
        const bool clock_valid =
            clock.mode == kFdwicPerfClockMode && clock.final_seen == 1 && clock.expected_submit_count != 0 &&
            clock.submit_count == clock.expected_submit_count && clock.first_submit_start != 0 &&
            clock.last_submit_end >= clock.first_submit_start;
        if (!identity_valid || !counters_clean || !clock_valid) {
            LOG_ERROR(
                "fdwic perf-clock core %u failed closure: core=%d block=%d/%d lane=%d/%d count=%u/%u "
                "start=%llu end=%llu mode=%u final=%u trace_count=%u atomic=%u",
                core_id, core.core_idx, core.block_id, expected_blocks[core_id], core.lane, expected_lanes[core_id],
                clock.submit_count, clock.expected_submit_count,
                static_cast<unsigned long long>(clock.first_submit_start),
                static_cast<unsigned long long>(clock.last_submit_end), clock.mode, clock.final_seen, core.count,
                core.atomic_calls
            );
            return -1;
        }
        if (expected_submits == 0) expected_submits = clock.expected_submit_count;
        if (clock.expected_submit_count != expected_submits) {
            LOG_ERROR(
                "fdwic perf-clock expected Submit count differs across cores: core=%u expected=%u reference=%u",
                core_id, clock.expected_submit_count, expected_submits
            );
            return -1;
        }
        if (clock.first_submit_start < global_start) global_start = clock.first_submit_start;
        if (clock.last_submit_end > global_end) global_end = clock.last_submit_end;
        const uint64_t elapsed = clock.last_submit_end - clock.first_submit_start;
        if (runtime->workers[core_id].core_type == CoreType::AIC) {
            ++aic_count;
            aic_elapsed_sum += elapsed;
            if (elapsed < aic_elapsed_min) aic_elapsed_min = elapsed;
            if (elapsed > aic_elapsed_max) aic_elapsed_max = elapsed;
        } else if (runtime->workers[core_id].core_type == CoreType::AIV) {
            ++aiv_count;
            aiv_elapsed_sum += elapsed;
            if (elapsed < aiv_elapsed_min) aiv_elapsed_min = elapsed;
            if (elapsed > aiv_elapsed_max) aiv_elapsed_max = elapsed;
        } else {
            LOG_ERROR("fdwic perf-clock core %u has invalid core type", core_id);
            return -1;
        }
    }
    if (aic_count != kExpectedAic || aiv_count != kExpectedAiv || global_start == std::numeric_limits<uint64_t>::max() ||
        global_end < global_start) {
        LOG_ERROR(
            "fdwic perf-clock topology/global closure failed: AIC=%u/%u AIV=%u/%u start=%llu end=%llu", aic_count,
            kExpectedAic, aiv_count, kExpectedAiv, static_cast<unsigned long long>(global_start),
            static_cast<unsigned long long>(global_end)
        );
        return -1;
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
    out << "  \"schema\": \"fdwic-perf-clock-v1\",\n";
    out << "  \"mode\": \"perf-clock\",\n";
    out << "  \"clock_freq_hz\": " << header->freq_hz << ",\n";
    out << "  \"device_header_bytes\": " << sizeof(FdwicSwimlaneHeader) << ",\n";
    out << "  \"num_cores\": " << header->num_cores << ",\n";
    out << "  \"aic_cores\": " << aic_count << ",\n";
    out << "  \"aiv_cores\": " << aiv_count << ",\n";
    out << "  \"expected_submits_per_core\": " << expected_submits << ",\n";
    out << "  \"global_first_submit_start\": " << global_start << ",\n";
    out << "  \"global_last_submit_end\": " << global_end << ",\n";
    out << "  \"global_submit_span_ticks\": " << global_elapsed << ",\n";
    out << "  \"global_submit_span_us\": "
        << static_cast<double>(global_elapsed) * 1000000.0 / static_cast<double>(header->freq_hz) << ",\n";
    out << "  \"groups\": {\n";
    out << "    \"aic\": {\"min_ticks\": " << aic_elapsed_min << ", \"max_ticks\": " << aic_elapsed_max
        << ", \"mean_ticks\": " << static_cast<double>(aic_elapsed_sum) / aic_count << "},\n";
    out << "    \"aiv\": {\"min_ticks\": " << aiv_elapsed_min << ", \"max_ticks\": " << aiv_elapsed_max
        << ", \"mean_ticks\": " << static_cast<double>(aiv_elapsed_sum) / aiv_count << "}\n";
    out << "  },\n";
    out << "  \"cores\": [\n";
    for (uint32_t core_id = 0; core_id < header->num_cores; ++core_id) {
        const FdwicSwimlaneCoreState &core = header->cores[core_id];
        const FdwicPerfClockCoreData &clock = core.perf_clock;
        const uint64_t elapsed = clock.last_submit_end - clock.first_submit_start;
        out << "    {\"core_id\": " << core_id << ", \"core_type\": \""
            << core_type_name(runtime->workers[core_id].core_type) << "\", \"block_id\": " << core.block_id
            << ", \"lane\": " << core.lane << ", \"submit_count\": " << clock.submit_count
            << ", \"first_submit_start\": " << clock.first_submit_start << ", \"last_submit_end\": "
            << clock.last_submit_end << ", \"elapsed_ticks\": " << elapsed << "}"
            << (core_id + 1 == header->num_cores ? "\n" : ",\n");
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
    LOG_INFO_V0(
        "fdwic perf-clock written to %s: cores=%u AIC=%u AIV=%u submits/core=%u span=%.3fus", path.c_str(),
        header->num_cores, aic_count, aiv_count, expected_submits,
        static_cast<double>(global_elapsed) * 1000000.0 / static_cast<double>(header->freq_hz)
    );
    return 0;
}

extern "C" void fdwic_perf_clock_host_finalize(Runtime *runtime) { fdwic_swimlane_host_finalize(runtime); }
