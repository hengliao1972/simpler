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
    MetricStats scalar;
    MetricStats requests;
    MetricStats misses;

    void add(const FdwicSubmitPmuCoreData &core) {
        ++cores;
        total.add(core.total_cycles);
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
        const bool window_valid = core.first_submit_start_tick != 0 &&
                                  core.last_submit_end_tick >= core.first_submit_start_tick && core.total_cycles != 0;
        const bool shadow_valid =
            phase_mode ?
                core.shadow_icache_requests <= core.icache_requests && core.shadow_icache_misses <= core.icache_misses :
                core.shadow_icache_requests == core.icache_requests && core.shadow_icache_misses == core.icache_misses;
        const bool counters_valid = core.scalar_busy <= core.total_cycles && core.icache_requests != 0 &&
                                    core.icache_misses <= core.icache_requests &&
                                    core.shadow_icache_misses <= core.shadow_icache_requests && shadow_valid &&
                                    core.scalar_busy < kFdwicSubmitPmuCounterRiskThreshold &&
                                    core.icache_requests < kFdwicSubmitPmuCounterRiskThreshold &&
                                    core.icache_misses < kFdwicSubmitPmuCounterRiskThreshold &&
                                    core.shadow_icache_requests < kFdwicSubmitPmuCounterRiskThreshold &&
                                    core.shadow_icache_misses < kFdwicSubmitPmuCounterRiskThreshold;
        if (!identity_valid || !count_valid || !window_valid || !counters_valid ||
            core.status != kFdwicSubmitPmuRequiredCoreStatus) {
            LOG_ERROR(
                "fdwic submit-PMU core %u closure failed: physical=%u block=%u lane=%u count=%u/%u "
                "ticks=%llu..%llu total=%llu scalar=%u req=%u/%u miss=%u/%u status=0x%x "
                "gates(identity/count/window/counters/status)=%u/%u/%u/%u/%u "
                "expected_block/lane=%u/%u role=%d physical_in_range/duplicate/bitmap=%u/%u/%u",
                logical, physical, core.block_id, core.lane, core.submit_count, core.expected_submit_count,
                static_cast<unsigned long long>(core.first_submit_start_tick),
                static_cast<unsigned long long>(core.last_submit_end_tick),
                static_cast<unsigned long long>(core.total_cycles), core.scalar_busy, core.icache_requests,
                core.shadow_icache_requests, core.icache_misses, core.shadow_icache_misses, core.status, identity_valid,
                count_valid, window_valid, counters_valid, core.status == kFdwicSubmitPmuRequiredCoreStatus,
                expected_block, expected_lane, static_cast<int32_t>(role), physical_in_range, physical_duplicate,
                owner_configured
            );
            return false;
        }
        if (phase_mode) {
            const FdwicSubmitPmuPhaseCoreData &phase = phases[logical];
            const uint32_t expected_phase_calls =
                fdwic_submit_pmu_expected_phase_calls(profile.phase, core.expected_submit_count);
            const bool efdrain_control = profile.phase == FdwicSubmitPmuPhase::EfDrainControl;
            const uint32_t excluded_kernel_calls = efdrain_control ? phase.reserved[0] : 0U;
            const uint64_t expected_boundary_reads = fdwic_submit_pmu_expected_phase_boundary_reads(
                profile.phase, core.expected_submit_count, excluded_kernel_calls
            );
            bool reserved_valid = true;
            for (uint32_t index = efdrain_control ? 1U : 0U; index < 4U; ++index)
                reserved_valid = reserved_valid && phase.reserved[index] == 0U;
            const bool phase_valid =
                expected_phase_calls != 0 && phase.phase_id == static_cast<uint32_t>(profile.phase) &&
                phase.phase_begin_reads == expected_boundary_reads &&
                phase.phase_end_reads == expected_boundary_reads && phase.phase_elapsed_ticks != 0 &&
                phase.phase_elapsed_ticks <= core.last_submit_end_tick - core.first_submit_start_tick &&
                phase.phase_icache_requests_observed <= core.shadow_icache_requests &&
                phase.phase_icache_misses_observed <= core.shadow_icache_misses &&
                phase.max_shadow_request_chunk < kFdwicSubmitPmuCounterRiskThreshold &&
                phase.max_shadow_miss_chunk < kFdwicSubmitPmuCounterRiskThreshold &&
                phase.status == kFdwicSubmitPmuRequiredPhaseStatus && reserved_valid;
            if (!phase_valid) {
                LOG_ERROR(
                    "fdwic submit-PMU phase core %u closure failed: id=%u reads=%u/%u expected=%llu "
                    "excluded_kernel=%u ticks=%llu/%llu observed=%llu/%u,%llu/%u max_chunk=%u/%u "
                    "status=0x%x reserved=%u",
                    logical, phase.phase_id, phase.phase_begin_reads, phase.phase_end_reads,
                    static_cast<unsigned long long>(expected_boundary_reads), excluded_kernel_calls,
                    static_cast<unsigned long long>(phase.phase_elapsed_ticks),
                    static_cast<unsigned long long>(core.last_submit_end_tick - core.first_submit_start_tick),
                    static_cast<unsigned long long>(phase.phase_icache_requests_observed), core.shadow_icache_requests,
                    static_cast<unsigned long long>(phase.phase_icache_misses_observed), core.shadow_icache_misses,
                    phase.max_shadow_request_chunk, phase.max_shadow_miss_chunk, phase.status, reserved_valid
                );
                return false;
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
    out << "  \"schema\": \"fdwic-submit-pmu-v1\",\n";
    out << "  \"capture\": {\"mode\": \"" << profile->name
        << "\", "
           "\"window_scope\": \"per_core_first_submit_begin_to_last_submit_end\", "
           "\"accepted\": true, \"owner_restore_passed\": true},\n";
    out << "  \"configuration\": {\n";
    out << "    \"num_cores\": 96, \"aic_cores\": 32, \"aiv_cores\": 64,\n";
    out << "    \"expected_submits_per_core\": " << data.expected_submits << ",\n";
    out << "    \"sys_counter_tick_ns\": 1,\n";
    out << "    \"selectors\": {\"cnt2_scalar_busy\": 1, \"cnt5_shadow_icache_miss\": 53, "
           "\"cnt6_primary_icache_request\": 52, \"cnt7_primary_icache_miss\": 53, "
           "\"cnt8_shadow_icache_request\": 52},\n";
    out << "    \"status_required_mask\": " << kFdwicSubmitPmuRequiredCoreStatus << ",\n";
    out << "    \"counter_width_bits\": {\"total\": 64, \"programmable\": 32},\n";
    out << "    \"programmable_counter_risk_threshold\": " << kFdwicSubmitPmuCounterRiskThreshold << ",\n";
    if (phase_mode) {
        const uint32_t expected_phase_calls =
            fdwic_submit_pmu_expected_phase_calls(profile->phase, data.expected_submits);
        out << "    \"phase\": {\"id\": " << static_cast<uint32_t>(profile->phase) << ", \"name\": \""
            << profile->phase_name << "\", " << "\"boundary\": \"" << profile->phase_boundary << "\", "
            << "\"expected_calls_per_core\": " << expected_phase_calls
            << ", \"status_required_mask\": " << kFdwicSubmitPmuRequiredPhaseStatus << ", \"counter_semantics\": \""
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
            << ", \"total_cycles\": " << core.total_cycles << ", \"scalar_busy\": " << core.scalar_busy
            << ", \"icache_requests\": " << core.icache_requests << ", \"icache_misses\": " << core.icache_misses
            << ", \"shadow_icache_requests\": " << core.shadow_icache_requests
            << ", \"shadow_icache_misses\": " << core.shadow_icache_misses << ", \"status\": " << core.status;
        if (phase_mode) {
            const FdwicSubmitPmuPhaseCoreData &phase = phases[logical];
            out << ", \"phase_id\": " << phase.phase_id << ", \"phase_elapsed_ticks\": " << phase.phase_elapsed_ticks
                << ", \"phase_icache_requests_observed\": " << phase.phase_icache_requests_observed
                << ", \"phase_icache_misses_observed\": " << phase.phase_icache_misses_observed
                << ", \"phase_begin_reads\": " << phase.phase_begin_reads
                << ", \"phase_end_reads\": " << phase.phase_end_reads
                << ", \"phase_max_shadow_request_chunk\": " << phase.max_shadow_request_chunk
                << ", \"phase_max_shadow_miss_chunk\": " << phase.max_shadow_miss_chunk
                << ", \"phase_status\": " << phase.status;
            if (profile->phase == FdwicSubmitPmuPhase::EfDrainControl) {
                out << ", \"phase_excluded_kernel_calls\": " << phase.reserved[0];
            }
        }
        out << "}" << (logical + 1U == kFdwicSubmitPmuExpectedCores ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"validation\": {\"passed\": true, \"trusted_records\": 96, "
           "\"unique_physical_core_ids\": 96, \"aic_records\": 32, \"aiv_records\": 64, "
           "\"mixed_triplets\": 32, \"owner_bitmap_member_records\": 96, \"status_match_records\": 96, "
           "\"selector_match_records\": 96, \"window_started_records\": 96, "
           "\"window_stopped_records\": 96, \"submit_count_closed_records\": 96, "
           "\"scalar_le_total_records\": 96, ";
    if (phase_mode) {
        out << "\"shadow_primary_bounded_records\": 96, \"phase_boundary_closed_records\": 96, "
               "\"phase_shape_match_records\": 96, \"phase_values_ordered_records\": 96, "
               "\"phase_time_within_submit_records\": 96, ";
        if (profile->phase == FdwicSubmitPmuPhase::EfDrainControl) {
            out << "\"phase_kernel_exclusion_closed_records\": 96, ";
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
