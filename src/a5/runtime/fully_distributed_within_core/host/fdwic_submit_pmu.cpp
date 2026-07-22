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

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <sys/stat.h>

#include "common/platform_config.h"
#include "common/unified_log.h"
#include "dist_engine/common/submit_pmu_types.h"
#include "runtime.h"

extern "C" void fdwic_swimlane_host_finalize(Runtime *runtime);

namespace {

constexpr double kPmuCyclesPerNsAll = 1.649844;
constexpr double kPmuCyclesPerNsAic = 1.650062;
constexpr double kPmuCyclesPerNsAiv = 1.649731;

struct SubmitPmuProfile {
    const char *name;
    uint16_t mode;
    FdwicSubmitPmuPhase phase;
    size_t bytes;
    const char *phase_name;
    const char *phase_boundary;
    const char *counter_semantics;
    const char *time_semantics;
};

constexpr SubmitPmuProfile kSubmitPmuProfiles[] = {
    {"submit-pmu-none", kFdwicSubmitPmuModeNone, FdwicSubmitPmuPhase::None, kFdwicSubmitPmuNoneBytes, nullptr, nullptr,
     nullptr, nullptr},
    {"submit-pmu-arg-build", kFdwicSubmitPmuModeArgBuild, FdwicSubmitPmuPhase::ArgBuild, kFdwicSubmitPmuPhaseBytes,
     "arg-build", "claim_end_to_materialize_begin", "running_read_clear_observed_bracket",
     "inner_sys_cnt_between_boundary_observers"},
    {"submit-pmu-empty-bracket", kFdwicSubmitPmuModeEmptyBracket, FdwicSubmitPmuPhase::EmptyBracket,
     kFdwicSubmitPmuPhaseBytes, "empty-bracket", "claim_end_adjacent_empty_bracket",
     "running_read_clear_empty_bracket_calibration", "outer_sys_cnt_around_adjacent_begin_end_pair"},
    {"submit-pmu-materialize", kFdwicSubmitPmuModeMaterialize, FdwicSubmitPmuPhase::Materialize,
     kFdwicSubmitPmuPhaseBytes, "materialize", "materialize_begin_to_materialize_end",
     "running_read_clear_observed_bracket", "inner_sys_cnt_between_boundary_observers"},
    {"submit-pmu-claim", kFdwicSubmitPmuModeClaim, FdwicSubmitPmuPhase::Claim, kFdwicSubmitPmuPhaseBytes, "claim",
     "claim_begin_to_claim_end", "running_read_clear_observed_bracket", "inner_sys_cnt_between_boundary_observers"},
    {"submit-pmu-register", kFdwicSubmitPmuModeRegister, FdwicSubmitPmuPhase::Register, kFdwicSubmitPmuPhaseBytes,
     "register", "register_outputs_call_entry_to_return", "running_read_clear_observed_bracket",
     "inner_sys_cnt_between_boundary_observers"},
    {"submit-pmu-submit-transition", kFdwicSubmitPmuModeSubmitTransition, FdwicSubmitPmuPhase::SubmitTransition,
     kFdwicSubmitPmuPhaseBytes, "submit-transition", "previous_submit_end_to_next_submit_begin",
     "running_read_clear_observed_bracket", "inner_sys_cnt_between_boundary_observers"},
    {"submit-pmu-efdrain-control", kFdwicSubmitPmuModeEfDrainControl, FdwicSubmitPmuPhase::EfDrainControl,
     kFdwicSubmitPmuPhaseBytes, "efdrain-control", "efdrain_begin_to_end_excluding_linked_kernel_calls",
     "discontinuous_running_read_clear_excluding_linked_kernel_calls",
     "discontinuous_sys_cnt_control_segments_excluding_linked_kernel_calls"},
    {"submit-pmu-prepare-map", kFdwicSubmitPmuModePrepareMap, FdwicSubmitPmuPhase::PrepareMap,
     kFdwicSubmitPmuPhaseBytes, "prepare-map", "dist_submit_prepare_map_call_entry_to_return",
     "running_read_clear_observed_bracket", "inner_sys_cnt_between_boundary_observers"},
    {"submit-pmu-fanin", kFdwicSubmitPmuModeFanin, FdwicSubmitPmuPhase::Fanin, kFdwicSubmitPmuPhaseBytes, "fanin",
     "fanin_begin_to_fanin_end", "running_read_clear_observed_bracket", "inner_sys_cnt_between_boundary_observers"},
    {"submit-pmu-winner-build-control", kFdwicSubmitPmuModeWinnerBuild, FdwicSubmitPmuPhase::WinnerBuild,
     kFdwicSubmitPmuPhaseBytes, "winner-build-control", "winner_build_begin_to_end_excluding_linked_kernel_calls",
     "discontinuous_running_read_clear_excluding_linked_kernel_calls",
     "discontinuous_sys_cnt_control_segments_excluding_linked_kernel_calls"},
    {"submit-pmu-alloc-complete-control", kFdwicSubmitPmuModeAllocComplete, FdwicSubmitPmuPhase::AllocComplete,
     kFdwicSubmitPmuPhaseBytes, "alloc-complete-control", "alloc_complete_begin_to_end_excluding_linked_kernel_calls",
     "discontinuous_running_read_clear_excluding_linked_kernel_calls",
     "discontinuous_sys_cnt_control_segments_excluding_linked_kernel_calls"},
};

const SubmitPmuProfile *requested_profile() {
    const char *name = std::getenv("PTO_FDWIC_PROFILE");
    if (name == nullptr) return nullptr;
    for (const SubmitPmuProfile &profile : kSubmitPmuProfiles) {
        if (std::strcmp(name, profile.name) == 0) return &profile;
    }
    return nullptr;
}

const FdwicSubmitPmuPhaseCoreData *phase_records(const FdwicSubmitPmuHeader &header) {
    return reinterpret_cast<const FdwicSubmitPmuPhaseCoreData *>(
        reinterpret_cast<const uint8_t *>(&header) + sizeof(FdwicSubmitPmuHeader)
    );
}

std::string raw_path(const char *prefix) {
    const std::string dir = prefix == nullptr || prefix[0] == '\0' ? "." : prefix;
    return dir.back() == '/' ? dir + "fdwic_submit_pmu_raw.json" : dir + "/fdwic_submit_pmu_raw.json";
}

bool prepare_directory(const std::string &path) {
    struct stat info {};
    if (stat(path.c_str(), &info) == 0) return S_ISDIR(info.st_mode);
    return errno == ENOENT && mkdir(path.c_str(), 0755) == 0;
}

bool remove_if_present(const std::string &path) { return std::remove(path.c_str()) == 0 || errno == ENOENT; }

const char *role_name(CoreType role) { return role == CoreType::AIC ? "aic" : "aiv"; }

bool physical_is_aic(uint32_t physical_id) {
    return physical_id < kFdwicSubmitPmuPhysicalSubcores && physical_id % 54U < 18U;
}

uint32_t bitmap_count(const uint32_t words[kFdwicSubmitPmuBitmapWords]) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < kFdwicSubmitPmuBitmapWords; ++i) {
        count += static_cast<uint32_t>(__builtin_popcount(words[i]));
    }
    return count;
}

bool bitmap_contains(const uint32_t words[kFdwicSubmitPmuBitmapWords], uint32_t physical_id) {
    return physical_id < kFdwicSubmitPmuPhysicalSubcores &&
           (words[physical_id / 32U] & (1U << (physical_id % 32U))) != 0;
}

struct MetricStats {
    uint64_t sum{0};
    uint64_t min{std::numeric_limits<uint64_t>::max()};
    uint64_t max{0};

    void add(uint64_t value) {
        sum += value;
        min = std::min(min, value);
        max = std::max(max, value);
    }
};

struct GroupStats {
    uint32_t cores{0};
    MetricStats total;
    MetricStats scalar_elapsed;
    MetricStats scalar;
    MetricStats requests;
    MetricStats misses;

    void add(const FdwicSubmitPmuCoreData &core) {
        ++cores;
        total.add(core.total_cycles);
        scalar_elapsed.add(core.scalar_submit_elapsed_ticks);
        scalar.add(core.scalar_busy);
        requests.add(core.icache_requests);
        misses.add(core.icache_misses);
    }
};

void write_metric(std::ofstream &out, const char *name, const MetricStats &metric, uint32_t cores, bool comma) {
    out << "        \"" << name << "\": {\"sum\": " << metric.sum << ", \"min\": " << metric.min
        << ", \"mean\": " << static_cast<double>(metric.sum) / cores << ", \"max\": " << metric.max << "}"
        << (comma ? "," : "") << "\n";
}

void write_group(std::ofstream &out, const char *name, const GroupStats &group, bool comma) {
    out << "    \"" << name << "\": {\n";
    out << "      \"cores\": " << group.cores << ",\n";
    write_metric(out, "total_cycles", group.total, group.cores, true);
    write_metric(out, "scalar_submit_elapsed_ticks", group.scalar_elapsed, group.cores, true);
    write_metric(out, "scalar_busy", group.scalar, group.cores, true);
    write_metric(out, "icache_requests", group.requests, group.cores, true);
    write_metric(out, "icache_misses", group.misses, group.cores, true);
    out << "      \"scalar_busy_share\": "
        << static_cast<double>(group.scalar.sum) / static_cast<double>(group.total.sum) << ",\n";
    out << "      \"icache_miss_rate\": "
        << static_cast<double>(group.misses.sum) / static_cast<double>(group.requests.sum) << "\n";
    out << "    }" << (comma ? "," : "") << "\n";
}

struct ValidatedData {
    uint32_t expected_submits{0};
    uint64_t phase_calls_all{0};
    uint64_t phase_calls_aic{0};
    uint64_t phase_calls_aiv{0};
    uint64_t global_start{std::numeric_limits<uint64_t>::max()};
    uint64_t global_end{0};
    uint32_t bitmap_words[kFdwicSubmitPmuBitmapWords]{};
    GroupStats all;
    GroupStats aic;
    GroupStats aiv;
};

bool validate(
    Runtime *runtime, const FdwicSubmitPmuHeader &header, const SubmitPmuProfile &profile, ValidatedData &data
) {
    const bool fixed_header =
        header.magic == kFdwicSubmitPmuMagic && header.version == kFdwicSubmitPmuVersion &&
        header.mode == profile.mode && header.header_bytes == profile.bytes &&
        header.record_bytes == sizeof(FdwicSubmitPmuCoreData) && header.num_cores == kFdwicSubmitPmuExpectedCores &&
        header.expected_aic == kFdwicSubmitPmuExpectedAic && header.expected_aiv == kFdwicSubmitPmuExpectedAiv &&
        header.sys_cnt_freq_hz == PLATFORM_PROF_SYS_CNT_FREQ && header.selectors[0] == kFdwicSubmitPmuCnt2ScalarBusy &&
        header.selectors[1] == kFdwicSubmitPmuCnt5ShadowIcacheMiss &&
        header.selectors[2] == kFdwicSubmitPmuCnt6IcacheRequest &&
        header.selectors[3] == kFdwicSubmitPmuCnt7IcacheMiss &&
        header.selectors[4] == kFdwicSubmitPmuCnt8ShadowIcacheRequest && runtime != nullptr &&
        runtime->worker_count == static_cast<int32_t>(kFdwicSubmitPmuExpectedCores) &&
        runtime->fdwic_swimlane_bytes_ == profile.bytes &&
        runtime->dist.swimlane_base == runtime->fdwic_swimlane_dev_base_ && runtime->dist.swimlane_level == 0 &&
        runtime->dist.swimlane_records_per_core == 0;
    if (!fixed_header) {
        LOG_ERROR("fdwic submit-PMU fixed header/state validation failed");
        return false;
    }

    uint32_t owner_words[kFdwicSubmitPmuBitmapWords]{};
    for (uint32_t word = 0; word < kFdwicSubmitPmuBitmapWords; ++word) {
        owner_words[word] = header.configured_bitmap_words[word];
        data.bitmap_words[word] = owner_words[word];
    }
    const bool owner_valid =
        header.owner_status == kFdwicSubmitPmuRequiredOwnerStatus &&
        header.configured_count == kFdwicSubmitPmuExpectedCores &&
        header.restored_count == kFdwicSubmitPmuExpectedCores && header.configured_aic == kFdwicSubmitPmuExpectedAic &&
        header.configured_aiv == kFdwicSubmitPmuExpectedAiv &&
        header.complete_mixed_triplets == kFdwicSubmitPmuExpectedAic && header.restore_failures == 0 &&
        header.active_after_restore == 0 && bitmap_count(owner_words) == kFdwicSubmitPmuExpectedCores &&
        header.first_failure_field == static_cast<uint32_t>(FdwicSubmitPmuOwnerField::None);
    if (!owner_valid) {
        LOG_ERROR(
            "fdwic submit-PMU owner closure failed: status=0x%x configured=%u restored=%u active=%u failures=%u",
            header.owner_status, header.configured_count, header.restored_count, header.active_after_restore,
            header.restore_failures
        );
        return false;
    }

    const bool phase_mode = fdwic_submit_pmu_mode_has_phase(profile.mode);
    const FdwicSubmitPmuPhaseCoreData *phases = phase_mode ? phase_records(header) : nullptr;
    bool physical_seen[kFdwicSubmitPmuPhysicalSubcores]{};
    uint32_t aic_count = 0;
    uint32_t aiv_count = 0;
    for (uint32_t logical = 0; logical < kFdwicSubmitPmuExpectedCores; ++logical) {
        const FdwicSubmitPmuCoreData &core = header.cores[logical];
        const CoreType role = runtime->workers[logical].core_type;
        const uint32_t expected_block = role == CoreType::AIC ? aic_count : aiv_count / 2U;
        const uint32_t expected_lane = role == CoreType::AIC ? 0U : 1U + aiv_count % 2U;
        if (role == CoreType::AIC) {
            ++aic_count;
        } else if (role == CoreType::AIV) {
            ++aiv_count;
        } else {
            LOG_ERROR("fdwic submit-PMU logical core %u has invalid role", logical);
            return false;
        }
        const uint32_t physical = core.physical_core_id;
        const bool physical_in_range = physical < kFdwicSubmitPmuPhysicalSubcores;
        const bool physical_duplicate = physical_in_range && physical_seen[physical];
        const bool owner_configured = physical_in_range && bitmap_contains(owner_words, physical);
        const bool role_matches_physical = physical_in_range && physical_is_aic(physical) == (role == CoreType::AIC);
        const bool identity_valid = core.logical_core_id == logical && physical_in_range && !physical_duplicate &&
                                    role_matches_physical && core.block_id == expected_block &&
                                    core.lane == expected_lane && owner_configured;
        const bool count_valid = core.expected_submit_count != 0 && core.submit_count == core.expected_submit_count &&
                                 (data.expected_submits == 0 || core.expected_submit_count == data.expected_submits);
        const bool wall_window_valid =
            core.first_submit_start_tick != 0 && core.last_submit_end_tick >= core.first_submit_start_tick;
        const uint64_t wall_elapsed_ticks =
            wall_window_valid ? core.last_submit_end_tick - core.first_submit_start_tick : 0;
        const bool scalar_elapsed_valid =
            core.scalar_submit_elapsed_ticks != 0 && core.scalar_submit_elapsed_ticks <= wall_elapsed_ticks;
        const bool window_valid = wall_window_valid && core.total_cycles != 0;
        const bool counters_valid = core.scalar_busy <= core.total_cycles && core.icache_requests != 0 &&
                                    core.icache_misses <= core.icache_requests &&
                                    core.scalar_busy < kFdwicSubmitPmuCounterRiskThreshold &&
                                    core.icache_requests < kFdwicSubmitPmuCounterRiskThreshold &&
                                    core.icache_misses < kFdwicSubmitPmuCounterRiskThreshold;
        const uint32_t required_core_status = fdwic_submit_pmu_required_core_status(profile.phase);
        const bool status_valid = core.status == required_core_status;
        if (!identity_valid || !count_valid || !window_valid || !scalar_elapsed_valid || !counters_valid ||
            !status_valid) {
            LOG_ERROR(
                "fdwic submit-PMU core %u closure failed: physical=%u block=%u lane=%u count=%u/%u "
                "ticks=%llu..%llu scalar/wall=%llu/%llu total=%llu scalar_busy=%u req=%u miss=%u "
                "status=0x%x/0x%x gates(identity/count/window/scalar/counters/status)=%u/%u/%u/%u/%u/%u "
                "expected_block/lane=%u/%u role=%d physical_in_range/duplicate/bitmap=%u/%u/%u",
                logical, physical, core.block_id, core.lane, core.submit_count, core.expected_submit_count,
                static_cast<unsigned long long>(core.first_submit_start_tick),
                static_cast<unsigned long long>(core.last_submit_end_tick),
                static_cast<unsigned long long>(core.scalar_submit_elapsed_ticks),
                static_cast<unsigned long long>(wall_elapsed_ticks), static_cast<unsigned long long>(core.total_cycles),
                core.scalar_busy, core.icache_requests, core.icache_misses, core.status, required_core_status,
                identity_valid, count_valid, window_valid, scalar_elapsed_valid, counters_valid, status_valid,
                expected_block, expected_lane, static_cast<int32_t>(role), physical_in_range, physical_duplicate,
                owner_configured
            );
            return false;
        }
        if (phase_mode) {
            const FdwicSubmitPmuPhaseCoreData &phase = phases[logical];
            const uint32_t expected_phase_calls =
                fdwic_submit_pmu_expected_phase_calls(profile.phase, core.expected_submit_count);
            const bool dynamic_calls = fdwic_submit_pmu_phase_has_dynamic_calls(profile.phase);
            const uint32_t excluded_kernel_calls = phase.reserved[0];
            const uint32_t phase_shadow_requests = phase.reserved[1];
            const uint32_t phase_shadow_misses = phase.reserved[2];
            const uint64_t expected_boundary_reads = fdwic_submit_pmu_expected_phase_boundary_reads(
                profile.phase, core.expected_submit_count, excluded_kernel_calls
            );
            const bool reserved_valid = phase.reserved[3] == 0U;
            const bool outer_reads_valid =
                phase.phase_begin_reads >= excluded_kernel_calls && phase.phase_end_reads >= excluded_kernel_calls;
            const uint32_t outer_begin_reads =
                outer_reads_valid ? phase.phase_begin_reads - excluded_kernel_calls : UINT32_MAX;
            const uint32_t outer_end_reads =
                outer_reads_valid ? phase.phase_end_reads - excluded_kernel_calls : UINT32_MAX;
            const bool boundary_shape_valid =
                dynamic_calls ?
                    outer_reads_valid && outer_begin_reads == outer_end_reads :
                    expected_phase_calls != 0 && outer_begin_reads == expected_phase_calls &&
                        outer_end_reads == expected_phase_calls && phase.phase_begin_reads == expected_boundary_reads &&
                        phase.phase_end_reads == expected_boundary_reads;
            const bool dynamic_count_valid =
                !dynamic_calls ||
                outer_begin_reads <=
                    fdwic_submit_pmu_dynamic_calls_max_per_core(profile.phase, core.expected_submit_count);
            const bool zero_call_dynamic = dynamic_calls && outer_begin_reads == 0;
            const bool elapsed_valid =
                zero_call_dynamic ?
                    phase.phase_elapsed_ticks == 0 :
                    phase.phase_elapsed_ticks != 0 && phase.phase_elapsed_ticks <= core.scalar_submit_elapsed_ticks;
            const bool zero_values_valid = !zero_call_dynamic || (phase.phase_icache_requests_observed == 0 &&
                                                                  phase.phase_icache_misses_observed == 0);
            const bool shadow_primary_bounded = phase_shadow_misses <= phase_shadow_requests &&
                                                phase_shadow_requests <= core.icache_requests &&
                                                phase_shadow_misses <= core.icache_misses;
            const bool phase_valid = phase.phase_id == static_cast<uint32_t>(profile.phase) && boundary_shape_valid &&
                                     dynamic_count_valid && elapsed_valid && zero_values_valid &&
                                     phase.phase_icache_misses_observed <= phase.phase_icache_requests_observed &&
                                     phase.phase_icache_requests_observed <= phase_shadow_requests &&
                                     phase.phase_icache_misses_observed <= phase_shadow_misses &&
                                     shadow_primary_bounded &&
                                     phase.max_shadow_request_chunk < kFdwicSubmitPmuCounterRiskThreshold &&
                                     phase.max_shadow_miss_chunk < kFdwicSubmitPmuCounterRiskThreshold &&
                                     phase_shadow_requests < kFdwicSubmitPmuCounterRiskThreshold &&
                                     phase_shadow_misses < kFdwicSubmitPmuCounterRiskThreshold &&
                                     phase.status == kFdwicSubmitPmuRequiredPhaseStatus && reserved_valid;
            if (!phase_valid) {
                LOG_ERROR(
                    "fdwic submit-PMU phase core %u closure failed: id=%u reads=%u/%u outer=%u/%u "
                    "expected_outer/boundary=%u/%llu excluded_kernel=%u scalar/wall=%llu/%llu "
                    "phase_ticks=%llu observed=%llu/%llu shadow=%u/%u primary=%u/%u max_chunk=%u/%u "
                    "status=0x%x reserved3=%u",
                    logical, phase.phase_id, phase.phase_begin_reads, phase.phase_end_reads, outer_begin_reads,
                    outer_end_reads, expected_phase_calls, static_cast<unsigned long long>(expected_boundary_reads),
                    excluded_kernel_calls, static_cast<unsigned long long>(core.scalar_submit_elapsed_ticks),
                    static_cast<unsigned long long>(wall_elapsed_ticks),
                    static_cast<unsigned long long>(phase.phase_elapsed_ticks),
                    static_cast<unsigned long long>(phase.phase_icache_requests_observed),
                    static_cast<unsigned long long>(phase.phase_icache_misses_observed), phase_shadow_requests,
                    phase_shadow_misses, core.icache_requests, core.icache_misses, phase.max_shadow_request_chunk,
                    phase.max_shadow_miss_chunk, phase.status, phase.reserved[3]
                );
                return false;
            }
            if (dynamic_calls) {
                data.phase_calls_all += outer_begin_reads;
                (role == CoreType::AIC ? data.phase_calls_aic : data.phase_calls_aiv) += outer_begin_reads;
            }
        }
        physical_seen[physical] = true;
        if (data.expected_submits == 0) data.expected_submits = core.expected_submit_count;
        data.global_start = std::min(data.global_start, core.first_submit_start_tick);
        data.global_end = std::max(data.global_end, core.last_submit_end_tick);
        data.all.add(core);
        (role == CoreType::AIC ? data.aic : data.aiv).add(core);
    }
    if (aic_count != kFdwicSubmitPmuExpectedAic || aiv_count != kFdwicSubmitPmuExpectedAiv ||
        data.global_start == std::numeric_limits<uint64_t>::max() || data.global_end < data.global_start) {
        LOG_ERROR("fdwic submit-PMU topology/window aggregate closure failed");
        return false;
    }
    if (fdwic_submit_pmu_phase_has_dynamic_calls(profile.phase)) {
        const uint64_t expected_all = fdwic_submit_pmu_expected_dynamic_calls_all(profile.phase, data.expected_submits);
        const bool role_calls_fixed = fdwic_submit_pmu_dynamic_calls_have_fixed_roles(profile.phase);
        const uint64_t expected_aic =
            role_calls_fixed ? fdwic_submit_pmu_expected_dynamic_calls_aic(profile.phase, data.expected_submits) : 0U;
        const uint64_t expected_aiv =
            role_calls_fixed ? fdwic_submit_pmu_expected_dynamic_calls_aiv(profile.phase, data.expected_submits) : 0U;
        const bool global_closed = expected_all != 0 && data.phase_calls_all == expected_all &&
                                   data.phase_calls_aic + data.phase_calls_aiv == data.phase_calls_all;
        const bool roles_closed =
            !role_calls_fixed || (data.phase_calls_aic == expected_aic && data.phase_calls_aiv == expected_aiv);
        if (!global_closed || !roles_closed) {
            LOG_ERROR(
                "fdwic submit-PMU dynamic phase call closure failed: actual(all/aic/aiv)=%llu/%llu/%llu "
                "expected_all=%llu fixed_roles=%u expected_aic/aiv=%llu/%llu",
                static_cast<unsigned long long>(data.phase_calls_all),
                static_cast<unsigned long long>(data.phase_calls_aic),
                static_cast<unsigned long long>(data.phase_calls_aiv), static_cast<unsigned long long>(expected_all),
                static_cast<unsigned>(role_calls_fixed), static_cast<unsigned long long>(expected_aic),
                static_cast<unsigned long long>(expected_aiv)
            );
            return false;
        }
    }

    uint32_t complete_triplets = 0;
    for (uint32_t physical = 0; physical < kFdwicSubmitPmuPhysicalSubcores; ++physical) {
        if (!physical_seen[physical] || !physical_is_aic(physical)) continue;
        const uint32_t die_base = physical / 54U * 54U;
        const uint32_t aic_local = physical % 54U;
        const uint32_t aiv0 = die_base + 18U + 2U * aic_local;
        const uint32_t aiv1 = aiv0 + 1U;
        if (aiv1 >= kFdwicSubmitPmuPhysicalSubcores || !physical_seen[aiv0] || !physical_seen[aiv1]) {
            LOG_ERROR("fdwic submit-PMU physical triplet is incomplete for AIC %u", physical);
            return false;
        }
        ++complete_triplets;
    }
    if (complete_triplets != kFdwicSubmitPmuExpectedAic || complete_triplets != header.complete_mixed_triplets) {
        LOG_ERROR("fdwic submit-PMU physical triplet count mismatch: %u", complete_triplets);
        return false;
    }
    return true;
}

}  // namespace

extern "C" int fdwic_submit_pmu_host_init(Runtime *runtime, int num_cores, const char *output_prefix) {
    const SubmitPmuProfile *profile = requested_profile();
    if (profile == nullptr) return 0;
    if (runtime == nullptr || num_cores != static_cast<int>(kFdwicSubmitPmuExpectedCores) ||
        runtime->worker_count != static_cast<int>(kFdwicSubmitPmuExpectedCores)) {
        return -1;
    }
    const bool storage_empty = runtime->fdwic_swimlane_host_shadow_ == nullptr &&
                               runtime->fdwic_swimlane_dev_allocation_ == 0 && runtime->fdwic_swimlane_dev_base_ == 0 &&
                               runtime->fdwic_swimlane_bytes_ == 0 && runtime->dist.swimlane_base == 0 &&
                               runtime->dist.swimlane_level == 0 && runtime->dist.swimlane_records_per_core == 0;
    if (!storage_empty || runtime->host_api.device_malloc == nullptr || runtime->host_api.device_free == nullptr ||
        runtime->host_api.copy_to_device == nullptr || runtime->host_api.copy_from_device == nullptr) {
        return -1;
    }
    const std::string prefix = output_prefix == nullptr || output_prefix[0] == '\0' ? "." : output_prefix;
    if (prefix.size() >= sizeof(runtime->fdwic_swimlane_output_prefix_) || !prepare_directory(prefix)) return -1;
    const std::string path = raw_path(prefix.c_str());
    if (!remove_if_present(path) || !remove_if_present(path + ".tmp")) return -1;

    void *shadow = std::aligned_alloc(64, profile->bytes);
    if (shadow == nullptr) return -1;
    std::memset(shadow, 0, profile->bytes);
    auto *header = reinterpret_cast<FdwicSubmitPmuHeader *>(shadow);
    header->magic = kFdwicSubmitPmuMagic;
    header->version = kFdwicSubmitPmuVersion;
    header->mode = profile->mode;
    header->header_bytes = static_cast<uint32_t>(profile->bytes);
    header->record_bytes = sizeof(FdwicSubmitPmuCoreData);
    header->num_cores = kFdwicSubmitPmuExpectedCores;
    header->expected_aic = kFdwicSubmitPmuExpectedAic;
    header->expected_aiv = kFdwicSubmitPmuExpectedAiv;
    header->sys_cnt_freq_hz = PLATFORM_PROF_SYS_CNT_FREQ;
    header->selectors[0] = kFdwicSubmitPmuCnt2ScalarBusy;
    header->selectors[1] = kFdwicSubmitPmuCnt5ShadowIcacheMiss;
    header->selectors[2] = kFdwicSubmitPmuCnt6IcacheRequest;
    header->selectors[3] = kFdwicSubmitPmuCnt7IcacheMiss;
    header->selectors[4] = kFdwicSubmitPmuCnt8ShadowIcacheRequest;

    constexpr uintptr_t kAlignment = 64;
    void *allocation = runtime->host_api.device_malloc(profile->bytes + kAlignment - 1);
    if (allocation == nullptr) {
        std::free(shadow);
        return -1;
    }
    const uintptr_t base = (reinterpret_cast<uintptr_t>(allocation) + kAlignment - 1) & ~(kAlignment - 1);
    if (runtime->host_api.copy_to_device(reinterpret_cast<void *>(base), shadow, profile->bytes) != 0) {
        runtime->host_api.device_free(allocation);
        std::free(shadow);
        return -1;
    }
    runtime->fdwic_swimlane_host_shadow_ = shadow;
    runtime->fdwic_swimlane_dev_allocation_ = reinterpret_cast<uint64_t>(allocation);
    runtime->fdwic_swimlane_dev_base_ = base;
    runtime->fdwic_swimlane_bytes_ = profile->bytes;
    runtime->fdwic_swimlane_num_cores_ = kFdwicSubmitPmuExpectedCores;
    runtime->fdwic_swimlane_records_per_core_ = 0;
    std::memcpy(runtime->fdwic_swimlane_output_prefix_, prefix.c_str(), prefix.size() + 1);
    runtime->dist.swimlane_base = base;
    runtime->dist.swimlane_level = 0;
    runtime->dist.swimlane_records_per_core = 0;
    return 1;
}

extern "C" int fdwic_submit_pmu_host_export(Runtime *runtime) {
    const SubmitPmuProfile *profile = requested_profile();
    if (profile == nullptr || runtime == nullptr || runtime->fdwic_swimlane_host_shadow_ == nullptr ||
        runtime->fdwic_swimlane_dev_base_ == 0) {
        return 0;
    }
    if (runtime->host_api.copy_from_device(
            runtime->fdwic_swimlane_host_shadow_, reinterpret_cast<void *>(runtime->fdwic_swimlane_dev_base_),
            profile->bytes
        ) != 0) {
        LOG_ERROR("fdwic submit-PMU D2H copy failed");
        return -1;
    }
    const auto &header = *reinterpret_cast<const FdwicSubmitPmuHeader *>(runtime->fdwic_swimlane_host_shadow_);
    ValidatedData data;
    if (!validate(runtime, header, *profile, data)) return -1;

    const std::string path = raw_path(runtime->fdwic_swimlane_output_prefix_);
    const std::string temporary = path + ".tmp";
    if (!remove_if_present(temporary)) return -1;
    std::ofstream out(temporary, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return -1;
    out << std::setprecision(12);
    const uint64_t global_span = data.global_end - data.global_start;
    const bool phase_mode = fdwic_submit_pmu_mode_has_phase(profile->mode);
    const FdwicSubmitPmuPhaseCoreData *phases = phase_mode ? phase_records(header) : nullptr;
    out << "{\n";
    out << "  \"schema\": \"fdwic-submit-pmu-v2\",\n";
    out << "  \"capture\": {\"mode\": \"" << profile->name
        << "\", "
           "\"window_scope\": \"per_core_first_submit_begin_to_last_submit_end\", "
           "\"accepted\": true, \"owner_restore_passed\": true},\n";
    out << "  \"configuration\": {\n";
    out << "    \"num_cores\": 96, \"aic_cores\": 32, \"aiv_cores\": 64,\n";
    out << "    \"expected_submits_per_core\": " << data.expected_submits << ",\n";
    out << "    \"sys_counter_tick_ns\": 1,\n";
    out << "    \"selectors\": {\"cnt0_vector_busy\": " << kFdwicSubmitPmuCnt0VectorBusy
        << ", \"cnt1_cube_busy\": " << kFdwicSubmitPmuCnt1CubeBusy
        << ", \"cnt2_scalar_busy\": " << kFdwicSubmitPmuCnt2ScalarBusy
        << ", \"cnt5_shadow_icache_miss\": " << kFdwicSubmitPmuCnt5ShadowIcacheMiss
        << ", \"cnt6_primary_icache_request\": " << kFdwicSubmitPmuCnt6IcacheRequest
        << ", \"cnt7_primary_icache_miss\": " << kFdwicSubmitPmuCnt7IcacheMiss
        << ", \"cnt8_shadow_icache_request\": " << kFdwicSubmitPmuCnt8ShadowIcacheRequest << "},\n";
    out << "    \"status_required_mask\": " << fdwic_submit_pmu_required_core_status(profile->phase) << ",\n";
    out << "    \"linked_kernel_exclusion\": {\"enabled\": true, "
           "\"boundary\": \"dist_aicore_call_slot_kernel_entry_to_return\", "
           "\"gate_semantics\": \"metrics_prof_stop_before_call_and_start_after_return\", "
           "\"time_denominator\": \"scalar_submit_elapsed_ticks\", "
           "\"wall_tick_semantics\": \"first_submit_start_to_last_submit_end_closure_only\"},\n";
    out << "    \"return_ready_atomic_exclusion\": {\"enabled\": true, "
           "\"classification\": \"result_used_atomic_only\", "
           "\"time_boundary\": \"sys_cnt_before_atomic_to_result_dependent_sys_cnt_after_return\", "
           "\"counter_semantics\": \"pmu_counters_include_atomic_instruction_events\", "
           "\"time_denominator_effect\": \"subtract_return_ready_atomic_elapsed\"},\n";
    out << "    \"counter_width_bits\": {\"total\": 64, \"programmable\": 32},\n";
    out << "    \"programmable_counter_risk_threshold\": " << kFdwicSubmitPmuCounterRiskThreshold << ",\n";
    if (phase_mode) {
        out << "    \"phase\": {\"id\": " << static_cast<uint32_t>(profile->phase) << ", \"name\": \""
            << profile->phase_name << "\", " << "\"boundary\": \"" << profile->phase_boundary << "\", ";
        if (fdwic_submit_pmu_phase_has_dynamic_calls(profile->phase)) {
            const uint64_t expected_all =
                fdwic_submit_pmu_expected_dynamic_calls_all(profile->phase, data.expected_submits);
            if (fdwic_submit_pmu_dynamic_calls_have_fixed_roles(profile->phase)) {
                const uint64_t expected_aic =
                    fdwic_submit_pmu_expected_dynamic_calls_aic(profile->phase, data.expected_submits);
                const uint64_t expected_aiv =
                    fdwic_submit_pmu_expected_dynamic_calls_aiv(profile->phase, data.expected_submits);
                out << "\"call_shape\": \"dynamic_balanced\", \"expected_calls\": {\"all\": " << expected_all
                    << ", \"aic\": " << expected_aic << ", \"aiv\": " << expected_aiv << "}, ";
            } else {
                out << "\"call_shape\": \"dynamic_global\", \"expected_calls\": {\"all\": " << expected_all << "}, ";
            }
        } else {
            const uint32_t expected_phase_calls =
                fdwic_submit_pmu_expected_phase_calls(profile->phase, data.expected_submits);
            out << "\"expected_calls_per_core\": " << expected_phase_calls << ", ";
        }
        out << "\"status_required_mask\": " << kFdwicSubmitPmuRequiredPhaseStatus << ", \"counter_semantics\": \""
            << profile->counter_semantics << "\", " << "\"time_semantics\": \"" << profile->time_semantics << "\"},\n";
    }
    out << "    \"pmu_cycles_per_ns\": {\"all\": " << kPmuCyclesPerNsAll << ", \"aic\": " << kPmuCyclesPerNsAic
        << ", \"aiv\": " << kPmuCyclesPerNsAiv << "}\n";
    out << "  },\n";
    out << "  \"owner\": {\"configure_passed\": true, \"restore_passed\": true, "
           "\"configured_count\": 96, \"configured_aic\": 32, \"configured_aiv\": 64, "
           "\"restored_count\": 96, \"active_after_restore\": 0, \"restore_failures\": 0, "
           "\"configured_bitmap_words\": [";
    for (uint32_t word = 0; word < kFdwicSubmitPmuBitmapWords; ++word) {
        if (word != 0) out << ", ";
        out << data.bitmap_words[word];
    }
    out << "], \"complete_mixed_triplets\": 32},\n";
    out << "  \"window\": {\"global_first_submit_start_tick\": " << data.global_start
        << ", \"global_last_submit_end_tick\": " << data.global_end << ", \"global_submit_span_ticks\": " << global_span
        << ", \"global_submit_span_us\": " << static_cast<double>(global_span) / 1000.0 << "},\n";
    out << "  \"records\": [\n";
    for (uint32_t logical = 0; logical < kFdwicSubmitPmuExpectedCores; ++logical) {
        const auto &core = header.cores[logical];
        const char *role = role_name(runtime->workers[logical].core_type);
        out << "    {\"logical_core_id\": " << core.logical_core_id
            << ", \"physical_core_id\": " << core.physical_core_id << ", \"role\": \"" << role
            << "\", \"block_id\": " << core.block_id << ", \"lane\": " << core.lane
            << ", \"submit_count\": " << core.submit_count
            << ", \"expected_submit_count\": " << core.expected_submit_count
            << ", \"first_submit_start_tick\": " << core.first_submit_start_tick
            << ", \"last_submit_end_tick\": " << core.last_submit_end_tick
            << ", \"submit_elapsed_ticks\": " << core.last_submit_end_tick - core.first_submit_start_tick
            << ", \"scalar_submit_elapsed_ticks\": " << core.scalar_submit_elapsed_ticks
            << ", \"total_cycles\": " << core.total_cycles << ", \"scalar_busy\": " << core.scalar_busy
            << ", \"icache_requests\": " << core.icache_requests << ", \"icache_misses\": " << core.icache_misses
            << ", \"status\": " << core.status;
        if (phase_mode) {
            const FdwicSubmitPmuPhaseCoreData &phase = phases[logical];
            out << ", \"phase_id\": " << phase.phase_id << ", \"phase_elapsed_ticks\": " << phase.phase_elapsed_ticks
                << ", \"phase_icache_requests_observed\": " << phase.phase_icache_requests_observed
                << ", \"phase_icache_misses_observed\": " << phase.phase_icache_misses_observed
                << ", \"phase_begin_reads\": " << phase.phase_begin_reads
                << ", \"phase_end_reads\": " << phase.phase_end_reads
                << ", \"phase_max_shadow_request_chunk\": " << phase.max_shadow_request_chunk
                << ", \"phase_max_shadow_miss_chunk\": " << phase.max_shadow_miss_chunk
                << ", \"phase_status\": " << phase.status << ", \"phase_excluded_kernel_calls\": " << phase.reserved[0]
                << ", \"shadow_icache_requests\": " << phase.reserved[1]
                << ", \"shadow_icache_misses\": " << phase.reserved[2];
        }
        out << "}" << (logical + 1U == kFdwicSubmitPmuExpectedCores ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"validation\": {\"passed\": true, \"trusted_records\": 96, "
           "\"unique_physical_core_ids\": 96, \"aic_records\": 32, \"aiv_records\": 64, "
           "\"mixed_triplets\": 32, \"owner_bitmap_member_records\": 96, \"status_match_records\": 96, "
           "\"selector_match_records\": 96, \"window_started_records\": 96, "
           "\"window_stopped_records\": 96, \"submit_count_closed_records\": 96, "
           "\"scalar_le_total_records\": 96, \"linked_kernel_gate_closed_records\": 96, "
           "\"scalar_submit_elapsed_valid_records\": 96, \"vector_busy_zero_records\": 96, "
           "\"cube_busy_zero_records\": 96, \"return_ready_atomic_time_valid_records\": 96, ";
    if (phase_mode) {
        out << "\"shadow_primary_bounded_records\": 96, \"phase_boundary_closed_records\": 96, "
               "\"phase_shape_match_records\": 96, \"phase_values_ordered_records\": 96, "
               "\"phase_time_within_submit_records\": 96, \"phase_kernel_exclusion_closed_records\": 96, ";
        if (fdwic_submit_pmu_phase_has_dynamic_calls(profile->phase)) {
            out << "\"phase_global_call_count_closed\": true, ";
        }
    } else {
        out << "\"shadow_primary_match_records\": 96, ";
    }
    out << "\"icache_miss_le_request_records\": 96, \"counter_below_risk_threshold_records\": 96},\n";
    out << "  \"summary\": {\n";
    write_group(out, "all", data.all, true);
    write_group(out, "aic", data.aic, true);
    write_group(out, "aiv", data.aiv, false);
    out << "  }\n";
    out << "}\n";
    out.close();
    if (!out || std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::remove(temporary.c_str());
        return -1;
    }
    LOG_INFO_V0("fdwic submit-PMU raw written to %s", path.c_str());
    return 0;
}

extern "C" void fdwic_submit_pmu_host_finalize(Runtime *runtime) { fdwic_swimlane_host_finalize(runtime); }
