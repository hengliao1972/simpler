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

#include "../probe_host.h"
#include "atomic_contention_curve_shared.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace atomic_contention_curve;

constexpr std::array<uint32_t, 11> kRequestedPopulations = {1U, 2U, 4U, 8U, 16U, 24U, 32U, 48U, 64U, 80U, 96U};
constexpr uint32_t kDie0AicWorkers = 16U;
constexpr uint32_t kDie0AivWorkers = 32U;
constexpr uint32_t kDie0MixedWorkers = 48U;
constexpr size_t kCandidateHugePageBytes = 2U * 1024U * 1024U;
constexpr size_t kCandidateHugePages = 32U;
constexpr size_t kCandidateOffsetsPerPage = 4U;
constexpr std::array<size_t, kCandidateOffsetsPerPage> kCandidateSubOffsets = {0U, 4096U, 65536U, 262144U};
constexpr size_t kStatePoolBytes = kCandidateHugePageBytes * kCandidateHugePages;
constexpr uint32_t kHomeClassificationRepeats = 3U;
constexpr uint64_t kHomeClassificationMinGapTicks = 32U;
constexpr uint64_t kHomePairMaxLocalGapTicks = 8U;
constexpr uint32_t kBankSweeps = 10U;
constexpr uint32_t kBankGapStrideBytes = 64U;
constexpr uint32_t kBankMaxGapBytes = 4096U;
constexpr uint32_t kBankGapCount = kBankMaxGapBytes / kBankGapStrideBytes + 1U;
constexpr uint32_t kBankPhaseCount = 8U;
constexpr std::array<uint32_t, 8> kBankPopulations = {1U, 2U, 4U, 8U, 16U, 24U, 32U, 48U};
constexpr std::array<uint32_t, 8> kBankStrides = {0U, 64U, 128U, 256U, 512U, 1024U, 2048U, 4096U};

struct KernelBinary {
    KernelBinary() = default;
    explicit KernelBinary(const char *kernel_path) :
        path(kernel_path) {}

    std::string path;
    std::vector<char> bytes;
    aclrtBinHandle binary = nullptr;
    aclrtFuncHandle function = nullptr;
};

struct Topology {
    uint32_t active;
    uint32_t active_aic;
    uint32_t active_aiv;
    uint32_t launched;
    uint32_t block_dim;
    uint32_t active_start;
};

struct WaveSummary {
    uint64_t deadline = 0U;
    uint64_t min_begin = 0U;
    uint64_t max_begin = 0U;
    uint64_t max_end = 0U;
    uint64_t start_spread = 0U;
    uint64_t global_span = 0U;
    uint64_t max_worker_elapsed = 0U;
    uint64_t min_publish_begin = 0U;
    uint64_t max_publish_ready = 0U;
    uint64_t old_min = 0U;
    uint64_t old_max = 0U;
    uint64_t old_sum = 0U;
    uint64_t old_xor = 0U;
    bool valid = false;
};

struct DeviceBuffers {
    void *state = nullptr;
    void *target_pool = nullptr;
    std::array<void *, 2> selected_targets{};
    void *results = nullptr;
};

struct RunObservation {
    uint64_t launch_median = 0U;
    std::vector<uint32_t> active_hardware_core_ids;
};

struct BankRawSpec {
    std::ofstream *raw = nullptr;
    const char *experiment = nullptr;
    uint32_t target_offset_bytes = 0U;
    AddressLayout address_layout = AddressLayout::Shared;
    uint32_t target_stride_bytes = 0U;
    uint32_t primary_group_workers = 0U;
};

bool Check(aclError error, const char *label) {
    if (error == ACL_SUCCESS) return true;
    std::fprintf(stderr, "ACL error %d: %s\n", static_cast<int>(error), label);
    return false;
}

bool ReadBinary(KernelBinary *kernel) {
    std::ifstream input(kernel->path, std::ios::binary | std::ios::ate);
    if (!input) {
        std::fprintf(stderr, "Cannot open kernel ELF: %s\n", kernel->path.c_str());
        return false;
    }
    const std::streamsize size = input.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "Kernel ELF is empty: %s\n", kernel->path.c_str());
        return false;
    }
    kernel->bytes.resize(static_cast<size_t>(size));
    input.seekg(0, std::ios::beg);
    if (!input.read(kernel->bytes.data(), size)) {
        std::fprintf(stderr, "Cannot read kernel ELF: %s\n", kernel->path.c_str());
        return false;
    }
    return true;
}

bool LoadKernel(KernelBinary *kernel) {
    if (!ReadBinary(kernel)) return false;
    if (!Check(
            atomic_probe::LoadAicoreBinaryFromData(kernel->bytes.data(), kernel->bytes.size(), &kernel->binary),
            "load atomic contention curve ELF"
        )) {
        return false;
    }
    return Check(
        aclrtBinaryGetFunctionByEntry(kernel->binary, 0U, &kernel->function), "resolve atomic contention curve entry"
    );
}

const char *ScenarioName(Scenario scenario) {
    if (scenario == Scenario::Mixed) return "mixed";
    if (scenario == Scenario::Aic) return "aic";
    return "aiv";
}

const char *CurveProfileName(Scenario scenario, uint32_t gm_home) {
    static constexpr std::array<const char *, 6> names = {
        "mixed_gm0", "mixed_gm1", "aic_gm0", "aic_gm1", "aiv_gm0", "aiv_gm1",
    };
    return names[static_cast<size_t>(scenario) * 2U + gm_home];
}

uint32_t ScenarioCapacity(Scenario scenario) {
    if (scenario == Scenario::Aic) return kDie0AicWorkers;
    if (scenario == Scenario::Aiv) return kDie0AivWorkers;
    return kDie0MixedWorkers;
}

Topology MakeTopology(Scenario scenario, uint32_t population) {
    if (scenario == Scenario::Mixed) {
        const uint32_t active_blocks = (population + 2U) / 3U;
        return Topology{population, active_blocks, population - active_blocks, kMaxWorkers, kMaxAicWorkers, 0U};
    }
    if (scenario == Scenario::Aic) {
        return Topology{population, population, 0U, kMaxAicWorkers, kMaxAicWorkers, 0U};
    }
    return Topology{population, 0U, population, kMaxAivWorkers, kMaxAivWorkers, 0U};
}

uint32_t ExpectedRole(Scenario scenario, uint32_t worker_slot_id) {
    if (scenario == Scenario::Aic) return static_cast<uint32_t>(Role::Aic);
    if (scenario == Scenario::Aiv) return static_cast<uint32_t>(Role::Aiv);
    return worker_slot_id % 3U == 0U ? static_cast<uint32_t>(Role::Aic) : static_cast<uint32_t>(Role::Aiv);
}

uint32_t ExpectedCoreIndex(Scenario scenario, uint32_t worker_slot_id) {
    if (scenario != Scenario::Mixed) return worker_slot_id;
    const uint32_t block = worker_slot_id / 3U;
    const uint32_t position = worker_slot_id % 3U;
    return position == 0U ? block : block * 2U + position - 1U;
}

uint32_t ExpectedSubblock(Scenario scenario, uint32_t worker_slot_id) {
    if (scenario != Scenario::Mixed || worker_slot_id % 3U == 0U) return 0U;
    return worker_slot_id % 3U - 1U;
}

bool IsAicHardwareCore(uint32_t hardware_core_id) {
    return hardware_core_id < kHardwareSubcoreCount &&
           hardware_core_id % kHardwareSubcoresPerDie < kHardwareAicSubcoresPerDie;
}

uint64_t XorRange(uint64_t first, uint64_t count) {
    uint64_t value = 0U;
    for (uint64_t offset = 0U; offset < count; ++offset)
        value ^= first + offset;
    return value;
}

void InitializeState(
    ProbeState *state, Scenario scenario, const Topology &topology, void *target, AddressLayout address_layout,
    uint32_t target_stride_bytes, uint32_t primary_group_workers
) {
    *state = ProbeState{};
    state->config.magic = kConfigMagic;
    state->config.scenario = static_cast<uint32_t>(scenario);
    state->config.active_workers = topology.active;
    state->config.launched_workers = topology.launched;
    state->config.active_aic = topology.active_aic;
    state->config.active_aiv = topology.active_aiv;
    state->config.block_dim = topology.block_dim;
    state->config.total_waves = kTotalWaves;
    state->config.warmup_waves = kWarmupWaves;
    state->config.active_worker_start = topology.active_start;
    state->config.target_address = reinterpret_cast<uint64_t>(target);
    state->config.address_layout = static_cast<uint32_t>(address_layout);
    state->config.target_stride_bytes = target_stride_bytes;
    state->config.primary_group_workers = primary_group_workers;
    state->ready_workers.value = 0;
    state->first_deadline.value = 0;
    state->guards.first = kGuardA;
    state->guards.second = kGuardB;
}

bool StateMatches(
    const ProbeState &state, Scenario scenario, const Topology &topology, const void *target,
    AddressLayout address_layout, uint32_t target_stride_bytes, uint32_t primary_group_workers
) {
    return state.config.magic == kConfigMagic && state.config.scenario == static_cast<uint32_t>(scenario) &&
           state.config.active_workers == topology.active && state.config.launched_workers == topology.launched &&
           state.config.active_aic == topology.active_aic && state.config.active_aiv == topology.active_aiv &&
           state.config.block_dim == topology.block_dim && state.config.total_waves == kTotalWaves &&
           state.config.warmup_waves == kWarmupWaves && state.config.active_worker_start == topology.active_start &&
           state.config.target_address == reinterpret_cast<uint64_t>(target) &&
           state.config.address_layout == static_cast<uint32_t>(address_layout) &&
           state.config.target_stride_bytes == target_stride_bytes &&
           state.config.primary_group_workers == primary_group_workers &&
           state.ready_workers.value == static_cast<int64_t>(topology.launched) && state.first_deadline.value > 0 &&
           state.guards.first == kGuardA && state.guards.second == kGuardB;
}

size_t TargetSpanBytes(AddressLayout address_layout, uint32_t target_stride_bytes, uint32_t active_workers) {
    if (address_layout == AddressLayout::Shared) return sizeof(AtomicLine);
    if (address_layout == AddressLayout::TwoGroups) return target_stride_bytes + sizeof(AtomicLine);
    return static_cast<size_t>(active_workers - 1U) * target_stride_bytes + sizeof(AtomicLine);
}

bool TargetSnapshotMatches(
    const std::vector<int64_t> &snapshot, AddressLayout address_layout, uint32_t target_stride_bytes,
    uint32_t active_workers, uint32_t primary_group_workers
) {
    for (size_t index = 0U; index < snapshot.size(); ++index) {
        int64_t expected = 0;
        if (address_layout == AddressLayout::Shared) {
            if (index == 0U) expected = static_cast<int64_t>(active_workers * kTotalWaves);
        } else if (address_layout == AddressLayout::TwoGroups) {
            if (index == 0U) expected = static_cast<int64_t>(primary_group_workers * kTotalWaves);
            if (index == target_stride_bytes / sizeof(int64_t)) {
                expected = static_cast<int64_t>((active_workers - primary_group_workers) * kTotalWaves);
            }
        } else {
            const size_t stride_words = target_stride_bytes / sizeof(int64_t);
            if (index % stride_words == 0U && index / stride_words < active_workers) {
                expected = static_cast<int64_t>(kTotalWaves);
            }
        }
        if (snapshot[index] != expected) return false;
    }
    return true;
}

bool ValidateWorkers(
    const std::vector<WorkerResult> &workers, Scenario scenario, const Topology &topology, uint64_t first_deadline,
    int32_t expected_active_die
) {
    bool valid = true;
    const uint64_t publish_deadline = first_deadline + static_cast<uint64_t>(kTotalWaves) * kWaveStrideTicks;
    std::array<bool, kHardwareSubcoreCount> hardware_seen{};
    for (uint32_t slot = 0U; slot < topology.launched; ++slot) {
        const WorkerResult &worker = workers[slot];
        const bool active = slot >= topology.active_start && slot - topology.active_start < topology.active;
        const uint32_t participant = active ? slot - topology.active_start : std::numeric_limits<uint32_t>::max();
        const bool hardware_in_range = worker.hardware_core_id < kHardwareSubcoreCount;
        const bool hardware_role_matches =
            hardware_in_range && IsAicHardwareCore(worker.hardware_core_id) ==
                                     (ExpectedRole(scenario, slot) == static_cast<uint32_t>(Role::Aic));
        const bool hardware_unique = hardware_in_range && !hardware_seen[worker.hardware_core_id];
        if (hardware_in_range) hardware_seen[worker.hardware_core_id] = true;
        const bool active_die_matches =
            !active || expected_active_die < 0 ||
            worker.hardware_core_id / kHardwareSubcoresPerDie == static_cast<uint32_t>(expected_active_die);
        valid &= worker.magic == kWorkerMagic && worker.worker_slot_id == slot &&
                 worker.participant_id == participant && worker.role == ExpectedRole(scenario, slot) &&
                 worker.logical_core_index == ExpectedCoreIndex(scenario, slot) &&
                 worker.subblock_id == ExpectedSubblock(scenario, slot) && hardware_role_matches && hardware_unique &&
                 worker.active == (active ? 1U : 0U) && worker.completed_waves == (active ? kTotalWaves : 0U) &&
                 active_die_matches && worker.status_flags == kStatusOk &&
                 worker.first_deadline_tick == first_deadline && worker.publish_begin_tick >= publish_deadline;
    }
    if (scenario == Scenario::Mixed) {
        for (uint32_t block = 0U; block < topology.block_dim; ++block) {
            const uint32_t aic_id = workers[block * 3U].hardware_core_id;
            if (!IsAicHardwareCore(aic_id)) {
                valid = false;
                continue;
            }
            const uint32_t die_base = (aic_id / kHardwareSubcoresPerDie) * kHardwareSubcoresPerDie;
            const uint32_t local_aic = aic_id % kHardwareSubcoresPerDie;
            const uint32_t expected_aiv0 = die_base + kHardwareAicSubcoresPerDie + local_aic * 2U;
            valid &= workers[block * 3U + 1U].hardware_core_id == expected_aiv0 &&
                     workers[block * 3U + 2U].hardware_core_id == expected_aiv0 + 1U;
        }
    }
    return valid;
}

std::string ActiveHardwareCoreIdsJson(const std::vector<WorkerResult> &workers, const Topology &topology) {
    std::string result = "[";
    for (uint32_t participant = 0U; participant < topology.active; ++participant) {
        const uint32_t slot = topology.active_start + participant;
        if (participant != 0U) result += ',';
        result += std::to_string(workers[slot].hardware_core_id);
    }
    result += ']';
    return result;
}

std::pair<bool, WaveSummary> ValidateWave(
    const std::vector<WaveResult> &waves, Scenario scenario, const Topology &topology, uint32_t wave,
    uint64_t first_deadline, AddressLayout address_layout, uint32_t primary_group_workers
) {
    WaveSummary summary{};
    summary.deadline = first_deadline + static_cast<uint64_t>(wave) * kWaveStrideTicks;
    summary.min_begin = std::numeric_limits<uint64_t>::max();
    summary.min_publish_begin = std::numeric_limits<uint64_t>::max();
    summary.old_min = std::numeric_limits<uint64_t>::max();
    std::vector<uint64_t> old_values;
    old_values.reserve(topology.active);
    std::array<std::vector<uint64_t>, 2> group_old_values;
    group_old_values[0].reserve(primary_group_workers);
    group_old_values[1].reserve(topology.active - primary_group_workers);
    bool valid = true;

    for (uint32_t participant = 0U; participant < topology.active; ++participant) {
        const uint32_t slot = topology.active_start + participant;
        const WaveResult &record = waves[slot * kTotalWaves + wave];
        valid &= record.participant_id == participant && record.wave == wave && record.status_flags == kStatusOk &&
                 record.role == ExpectedRole(scenario, slot) && record.deadline_tick == summary.deadline &&
                 record.begin_tick >= summary.deadline && record.end_tick >= record.begin_tick &&
                 record.end_tick < record.publish_begin_tick &&
                 record.publish_begin_tick >= summary.deadline + kWavePublishOffsetTicks &&
                 record.publish_ready_tick >= record.publish_begin_tick &&
                 record.publish_ready_tick < summary.deadline + kWaveStrideTicks;
        summary.min_begin = std::min(summary.min_begin, record.begin_tick);
        summary.max_begin = std::max(summary.max_begin, record.begin_tick);
        summary.max_end = std::max(summary.max_end, record.end_tick);
        summary.max_worker_elapsed = std::max(summary.max_worker_elapsed, record.end_tick - record.begin_tick);
        summary.min_publish_begin = std::min(summary.min_publish_begin, record.publish_begin_tick);
        summary.max_publish_ready = std::max(summary.max_publish_ready, record.publish_ready_tick);
        summary.old_min = std::min(summary.old_min, record.atomic_old);
        summary.old_max = std::max(summary.old_max, record.atomic_old);
        summary.old_sum += record.atomic_old;
        summary.old_xor ^= record.atomic_old;
        old_values.push_back(record.atomic_old);
        if (address_layout == AddressLayout::TwoGroups) {
            const bool primary_group = scenario == Scenario::Mixed ?
                                           ExpectedRole(scenario, slot) == static_cast<uint32_t>(Role::Aic) :
                                           participant < primary_group_workers;
            group_old_values[primary_group ? 0U : 1U].push_back(record.atomic_old);
        }
    }

    for (uint32_t slot = 0U; slot < topology.launched; ++slot) {
        if (slot >= topology.active_start && slot - topology.active_start < topology.active) continue;
        const WaveResult &record = waves[slot * kTotalWaves + wave];
        valid &= record.begin_tick == 0U && record.end_tick == 0U && record.atomic_old == 0U &&
                 record.deadline_tick == 0U && record.participant_id == 0U && record.wave == 0U &&
                 record.status_flags == 0U && record.role == 0U && record.publish_begin_tick == 0U &&
                 record.publish_ready_tick == 0U;
    }

    std::sort(old_values.begin(), old_values.end());
    if (address_layout == AddressLayout::Shared) {
        const uint64_t expected_first = static_cast<uint64_t>(wave) * topology.active;
        for (uint32_t index = 0U; index < topology.active; ++index) {
            valid &= old_values[index] == expected_first + index;
        }
        const uint64_t expected_sum = topology.active * (2U * expected_first + topology.active - 1U) / 2U;
        valid &= summary.old_min == expected_first && summary.old_max == expected_first + topology.active - 1U &&
                 summary.old_sum == expected_sum && summary.old_xor == XorRange(expected_first, topology.active);
    } else if (address_layout == AddressLayout::TwoGroups) {
        const std::array<uint32_t, 2> group_counts = {primary_group_workers, topology.active - primary_group_workers};
        uint64_t expected_min = std::numeric_limits<uint64_t>::max();
        uint64_t expected_max = 0U;
        uint64_t expected_sum = 0U;
        uint64_t expected_xor = 0U;
        for (uint32_t group = 0U; group < group_counts.size(); ++group) {
            std::sort(group_old_values[group].begin(), group_old_values[group].end());
            const uint64_t count = group_counts[group];
            const uint64_t first = static_cast<uint64_t>(wave) * count;
            for (uint64_t index = 0U; index < count; ++index) {
                valid &= group_old_values[group][index] == first + index;
            }
            expected_min = std::min(expected_min, first);
            expected_max = std::max(expected_max, first + count - 1U);
            expected_sum += count * (2U * first + count - 1U) / 2U;
            expected_xor ^= XorRange(first, count);
        }
        valid &= summary.old_min == expected_min && summary.old_max == expected_max &&
                 summary.old_sum == expected_sum && summary.old_xor == expected_xor;
    } else {
        const uint64_t expected_old = wave;
        for (uint64_t old_value : old_values)
            valid &= old_value == expected_old;
        valid &= summary.old_min == expected_old && summary.old_max == expected_old &&
                 summary.old_sum == expected_old * topology.active &&
                 summary.old_xor == (topology.active % 2U == 0U ? 0U : expected_old);
    }

    summary.start_spread = summary.max_begin - summary.min_begin;
    summary.global_span = summary.max_end - summary.min_begin;
    if (wave >= kWarmupWaves) valid &= summary.start_spread <= kMaxStartSpreadTicks;
    valid &= summary.max_end < summary.deadline + kWavePublishOffsetTicks &&
             summary.min_publish_begin >= summary.deadline + kWavePublishOffsetTicks &&
             summary.max_publish_ready < summary.deadline + kWaveStrideTicks;
    summary.valid = valid;
    return {valid, summary};
}

void WriteRawRow(
    std::ofstream &raw, const char *scenario_name, int32_t sweep, uint32_t population, const Topology &topology,
    const std::string &active_hardware_core_ids, uint64_t target_address, int32_t wave, const WaveSummary &summary,
    const char *status
) {
    const uint32_t measured = wave >= static_cast<int32_t>(kWarmupWaves) ? 1U : 0U;
    raw << scenario_name << '\t' << sweep << '\t' << population << '\t' << topology.active_aic << '\t'
        << topology.active_aiv << '\t' << active_hardware_core_ids << '\t' << target_address << '\t' << wave << '\t'
        << measured << '\t' << summary.deadline << '\t' << summary.min_begin << '\t' << summary.max_begin << '\t'
        << summary.max_end << '\t' << summary.start_spread << '\t' << summary.global_span << '\t'
        << summary.max_worker_elapsed << '\t' << summary.min_publish_begin << '\t' << summary.max_publish_ready << '\t'
        << summary.old_min << '\t' << summary.old_max << '\t' << summary.old_sum << '\t' << summary.old_xor << '\t'
        << status << '\n';
}

const char *AddressLayoutName(AddressLayout address_layout) {
    if (address_layout == AddressLayout::Shared) return "shared";
    if (address_layout == AddressLayout::ParticipantStride) return "participant_stride";
    return "two_groups";
}

uint32_t UniqueAddressCount(AddressLayout address_layout, uint32_t active_workers) {
    if (address_layout == AddressLayout::Shared) return 1U;
    if (address_layout == AddressLayout::TwoGroups) return 2U;
    return active_workers;
}

void WriteBankRawRow(
    const BankRawSpec &spec, const char *scenario_name, uint32_t sweep, uint32_t population, const Topology &topology,
    const std::string &active_hardware_core_ids, uint64_t target_address, uint32_t target_span_bytes, uint32_t wave,
    const WaveSummary &summary, const char *status
) {
    const uint32_t measured = wave >= kWarmupWaves ? 1U : 0U;
    *spec.raw << spec.experiment << '\t' << scenario_name << '\t' << sweep << '\t' << population << '\t'
              << topology.active_aic << '\t' << topology.active_aiv << '\t' << active_hardware_core_ids << '\t'
              << target_address << '\t' << spec.target_offset_bytes << '\t' << AddressLayoutName(spec.address_layout)
              << '\t' << spec.target_stride_bytes << '\t' << UniqueAddressCount(spec.address_layout, topology.active)
              << '\t' << target_span_bytes << '\t' << wave << '\t' << measured << '\t' << summary.deadline << '\t'
              << summary.min_begin << '\t' << summary.max_begin << '\t' << summary.max_end << '\t'
              << summary.start_spread << '\t' << summary.global_span << '\t' << summary.max_worker_elapsed << '\t'
              << summary.min_publish_begin << '\t' << summary.max_publish_ready << '\t' << summary.old_min << '\t'
              << summary.old_max << '\t' << summary.old_sum << '\t' << summary.old_xor << '\t' << status << '\n';
}

bool RunOnce(
    KernelBinary &kernel, aclrtStream stream, const DeviceBuffers &device, void *device_target, ProbeState *state,
    std::vector<WorkerResult> *workers, std::vector<WaveResult> *waves, std::ofstream *raw, Scenario scenario,
    uint32_t population, uint32_t sweep, const Topology *topology_override = nullptr,
    const char *scenario_name_override = nullptr, RunObservation *observation = nullptr, bool verbose = true,
    int32_t expected_active_die = -1, const BankRawSpec *bank_raw = nullptr
) {
    const Topology topology = topology_override != nullptr ? *topology_override : MakeTopology(scenario, population);
    const char *scenario_name = scenario_name_override != nullptr ? scenario_name_override : ScenarioName(scenario);
    const AddressLayout address_layout = bank_raw == nullptr ? AddressLayout::Shared : bank_raw->address_layout;
    const uint32_t target_stride_bytes = bank_raw == nullptr ? 0U : bank_raw->target_stride_bytes;
    const uint32_t primary_group_workers = bank_raw == nullptr ? 0U : bank_raw->primary_group_workers;
    const bool target_pattern_valid =
        (address_layout == AddressLayout::Shared && target_stride_bytes == 0U && primary_group_workers == 0U) ||
        (address_layout == AddressLayout::ParticipantStride && target_stride_bytes >= sizeof(AtomicLine) &&
         target_stride_bytes % alignof(AtomicLine) == 0U && primary_group_workers == 0U) ||
        (address_layout == AddressLayout::TwoGroups && target_stride_bytes >= sizeof(AtomicLine) &&
         target_stride_bytes % alignof(AtomicLine) == 0U && primary_group_workers > 0U &&
         primary_group_workers < topology.active);
    if (!target_pattern_valid) {
        std::fprintf(
            stderr, "Invalid host target pattern: layout=%s stride=%u primary_group=%u active=%u.\n",
            AddressLayoutName(address_layout), target_stride_bytes, primary_group_workers, topology.active
        );
        return false;
    }
    const size_t target_span_bytes = TargetSpanBytes(address_layout, target_stride_bytes, topology.active);
    InitializeState(
        state, scenario, topology, device_target, address_layout, target_stride_bytes, primary_group_workers
    );
    std::fill(workers->begin(), workers->end(), WorkerResult{});
    std::fill(waves->begin(), waves->end(), WaveResult{});

    auto *device_results = static_cast<uint8_t *>(device.results);
    if (!Check(
            aclrtMemcpy(device.state, sizeof(ProbeState), state, sizeof(ProbeState), ACL_MEMCPY_HOST_TO_DEVICE),
            "initialize atomic contention state"
        ) ||
        !Check(
            aclrtMemset(device_target, target_span_bytes, 0, target_span_bytes), "clear atomic contention target range"
        ) ||
        !Check(
            aclrtMemset(device.results, sizeof(ProbeResults), 0, sizeof(ProbeResults)),
            "clear atomic contention results"
        )) {
        return false;
    }

    const KernelArgs args{
        reinterpret_cast<uint64_t>(device.state),
        reinterpret_cast<uint64_t>(device.results),
    };
    if (!Check(
            aclrtLaunchKernelWithHostArgs(
                kernel.function, topology.block_dim, stream, nullptr, const_cast<KernelArgs *>(&args), sizeof(args),
                nullptr, 0U
            ),
            "launch atomic contention curve kernel"
        ) ||
        !Check(aclrtSynchronizeStream(stream), "synchronize atomic contention curve kernel") ||
        !Check(
            aclrtMemcpy(state, sizeof(ProbeState), device.state, sizeof(ProbeState), ACL_MEMCPY_DEVICE_TO_HOST),
            "copy atomic contention state"
        ) ||
        !Check(
            aclrtMemcpy(
                workers->data(), sizeof(WorkerResult) * topology.launched,
                device_results + offsetof(ProbeResults, workers), sizeof(WorkerResult) * topology.launched,
                ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "copy atomic contention workers"
        ) ||
        !Check(
            aclrtMemcpy(
                waves->data(), sizeof(WaveResult) * topology.launched * kTotalWaves,
                device_results + offsetof(ProbeResults, waves), sizeof(WaveResult) * topology.launched * kTotalWaves,
                ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "copy atomic contention waves"
        )) {
        return false;
    }

    std::vector<int64_t> target_snapshot(target_span_bytes / sizeof(int64_t));
    if (!Check(
            aclrtMemcpy(
                target_snapshot.data(), target_span_bytes, device_target, target_span_bytes, ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "copy atomic contention target range"
        )) {
        return false;
    }

    const uint64_t first_deadline = static_cast<uint64_t>(state->first_deadline.value);
    const std::string active_hardware_core_ids = ActiveHardwareCoreIdsJson(*workers, topology);
    const bool state_valid = StateMatches(
        *state, scenario, topology, device_target, address_layout, target_stride_bytes, primary_group_workers
    );
    const bool targets_valid = TargetSnapshotMatches(
        target_snapshot, address_layout, target_stride_bytes, topology.active, primary_group_workers
    );
    const bool workers_valid =
        state_valid && ValidateWorkers(*workers, scenario, topology, first_deadline, expected_active_die);
    bool valid = state_valid && targets_valid && workers_valid;
    std::array<WaveSummary, kTotalWaves> summaries{};
    for (uint32_t wave = 0U; wave < kTotalWaves; ++wave) {
        const auto [wave_valid, summary] =
            ValidateWave(*waves, scenario, topology, wave, first_deadline, address_layout, primary_group_workers);
        summaries[wave] = summary;
        valid &= wave_valid;
    }

    if (raw != nullptr) {
        for (uint32_t wave = 0U; wave < kTotalWaves; ++wave) {
            WriteRawRow(
                *raw, scenario_name, static_cast<int32_t>(sweep), population, topology, active_hardware_core_ids,
                reinterpret_cast<uint64_t>(device_target), static_cast<int32_t>(wave), summaries[wave],
                valid && summaries[wave].valid ? "PASS" : "FAIL"
            );
        }
        raw->flush();
    }
    if (bank_raw != nullptr && bank_raw->raw != nullptr) {
        for (uint32_t wave = 0U; wave < kTotalWaves; ++wave) {
            WriteBankRawRow(
                *bank_raw, scenario_name, sweep, population, topology, active_hardware_core_ids,
                reinterpret_cast<uint64_t>(device_target), static_cast<uint32_t>(target_span_bytes), wave,
                summaries[wave], valid && summaries[wave].valid ? "PASS" : "FAIL"
            );
        }
    }

    const auto measured_begin = summaries.begin() + kWarmupWaves;
    uint64_t max_spread = 0U;
    std::vector<uint64_t> measured_spans;
    measured_spans.reserve(kMeasuredWaves);
    for (auto iterator = measured_begin; iterator != summaries.end(); ++iterator) {
        max_spread = std::max(max_spread, iterator->start_spread);
        measured_spans.push_back(iterator->global_span);
    }
    std::sort(measured_spans.begin(), measured_spans.end());
    const uint64_t launch_median = measured_spans[measured_spans.size() / 2U];
    if (observation != nullptr) {
        observation->launch_median = launch_median;
        observation->active_hardware_core_ids.clear();
        observation->active_hardware_core_ids.reserve(topology.active);
        for (uint32_t participant = 0U; participant < topology.active; ++participant) {
            observation->active_hardware_core_ids.push_back(
                (*workers)[topology.active_start + participant].hardware_core_id
            );
        }
    }
    if (verbose) {
        std::printf(
            "[CURVE] sweep=%02u scenario=%-8s N=%2u C=%2u V=%2u blocks=%2u status=%s "
            "median_span=%llu tick max_start_spread=%llu tick\n",
            sweep, scenario_name, population, topology.active_aic, topology.active_aiv, topology.block_dim,
            valid ? "PASS" : "FAIL", static_cast<unsigned long long>(launch_median),
            static_cast<unsigned long long>(max_spread)
        );
    }
    if (!valid) {
        std::fprintf(
            stderr,
            "Semantic validation failed: scenario=%s sweep=%u N=%u layout=%s stride=%u primary_group=%u "
            "state=%s targets=%s workers=%s. "
            "The raw rows are marked FAIL and are not fit-eligible.\n",
            scenario_name, sweep, population, AddressLayoutName(address_layout), target_stride_bytes,
            primary_group_workers, state_valid ? "PASS" : "FAIL", targets_valid ? "PASS" : "FAIL",
            workers_valid ? "PASS" : "FAIL"
        );
        for (uint32_t slot = 0U; slot < topology.launched; ++slot) {
            const WorkerResult &worker = (*workers)[slot];
            std::fprintf(
                stderr,
                "  slot=%u hardware_core=%u magic=0x%08x participant=%u role=%u logical_core=%u "
                "subblock=%u active=%u waves=%u "
                "flags=0x%x first=%llu publish_begin=%llu\n",
                slot, worker.hardware_core_id, worker.magic, worker.participant_id, worker.role,
                worker.logical_core_index, worker.subblock_id, worker.active, worker.completed_waves,
                worker.status_flags, static_cast<unsigned long long>(worker.first_deadline_tick),
                static_cast<unsigned long long>(worker.publish_begin_tick)
            );
        }
        const WaveResult &first = (*waves)[topology.active_start * kTotalWaves];
        std::fprintf(
            stderr,
            "  first-wave begin=%llu end=%llu old=%llu deadline=%llu participant=%u wave=%u flags=0x%x role=%u "
            "publish_begin=%llu publish_ready=%llu\n",
            static_cast<unsigned long long>(first.begin_tick), static_cast<unsigned long long>(first.end_tick),
            static_cast<unsigned long long>(first.atomic_old), static_cast<unsigned long long>(first.deadline_tick),
            first.participant_id, first.wave, first.status_flags, first.role,
            static_cast<unsigned long long>(first.publish_begin_tick),
            static_cast<unsigned long long>(first.publish_ready_tick)
        );
    }
    return valid;
}

void WriteUnsupportedRows(std::ofstream &raw, const DeviceBuffers &device) {
    for (Scenario scenario : {Scenario::Mixed, Scenario::Aic, Scenario::Aiv}) {
        for (uint32_t gm_home = 0U; gm_home < 2U; ++gm_home) {
            for (uint32_t population : kRequestedPopulations) {
                if (population <= ScenarioCapacity(scenario)) continue;
                WriteRawRow(
                    raw, CurveProfileName(scenario, gm_home), -1, population, Topology{}, "[]",
                    reinterpret_cast<uint64_t>(device.selected_targets[gm_home]), -1, WaveSummary{}, "UNSUPPORTED"
                );
            }
        }
    }
}

void WriteRawHeader(std::ofstream &raw) {
    raw << "scenario\tsweep\tN\taic_active\taiv_active\tactive_hardware_core_ids\ttarget_address\twave\tmeasured\t"
           "deadline_tick\tmin_begin_tick\t"
           "max_begin_tick\tmax_end_tick\tstart_spread_tick\tglobal_span_tick\tmax_worker_elapsed_tick\t"
           "min_publish_begin_tick\tmax_publish_ready_tick\t"
           "old_min\told_max\told_sum\told_xor\tstatus\n";
}

void WriteBankRawHeader(std::ofstream &raw) {
    raw << "experiment\tscenario\tsweep\tN\taic_active\taiv_active\tactive_hardware_core_ids\t"
           "target_address\ttarget_offset_bytes\taddress_layout\ttarget_stride_bytes\tunique_addresses\t"
           "target_span_bytes\twave\tmeasured\tdeadline_tick\tmin_begin_tick\tmax_begin_tick\tmax_end_tick\t"
           "start_spread_tick\tglobal_span_tick\tmax_worker_elapsed_tick\tmin_publish_begin_tick\t"
           "max_publish_ready_tick\told_min\told_max\told_sum\told_xor\tstatus\n";
}

bool RunDieLocality(
    std::array<KernelBinary, 3> *kernels, aclrtStream stream, const DeviceBuffers &device, ProbeState *state,
    std::vector<WorkerResult> *workers, std::vector<WaveResult> *waves, std::ofstream &raw
) {
    struct Profile {
        Scenario scenario;
        const char *role_name;
        uint32_t core_die;
        Topology topology;
    };
    const std::array<Profile, 4> profiles{{
        {Scenario::Aic, "aic", 0U, Topology{1U, 1U, 0U, kMaxAicWorkers, kMaxAicWorkers, 0U}},
        {Scenario::Aic, "aic", 1U, Topology{1U, 1U, 0U, kMaxAicWorkers, kMaxAicWorkers, kDie0AicWorkers}},
        {Scenario::Aiv, "aiv", 0U, Topology{1U, 0U, 1U, kMaxAivWorkers, kMaxAivWorkers, 0U}},
        {Scenario::Aiv, "aiv", 1U, Topology{1U, 0U, 1U, kMaxAivWorkers, kMaxAivWorkers, kDie0AivWorkers}},
    }};
    for (uint32_t sweep = 0U; sweep < kSweeps; ++sweep) {
        for (uint32_t offset = 0U; offset < profiles.size() * 2U; ++offset) {
            const uint32_t index = (offset + sweep) % (profiles.size() * 2U);
            const uint32_t gm_home = index / profiles.size();
            const Profile &profile = profiles[index % profiles.size()];
            const std::string name = std::string(profile.role_name) + "_gm" + std::to_string(gm_home) + "_die" +
                                     std::to_string(profile.core_die);
            KernelBinary &kernel = (*kernels)[static_cast<size_t>(profile.scenario)];
            if (!RunOnce(
                    kernel, stream, device, device.selected_targets[gm_home], state, workers, waves, &raw,
                    profile.scenario, 1U, sweep, &profile.topology, name.c_str(), nullptr, true,
                    static_cast<int32_t>(profile.core_die)
                )) {
                return false;
            }
        }
    }
    return true;
}

void *CandidateTarget(const DeviceBuffers &device, size_t candidate) {
    const size_t page = candidate / kCandidateOffsetsPerPage;
    const size_t sub_offset = kCandidateSubOffsets[candidate % kCandidateOffsetsPerPage];
    return static_cast<uint8_t *>(device.target_pool) + page * kCandidateHugePageBytes + sub_offset;
}

uint64_t Median(std::vector<uint64_t> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

bool SelectGmHomes(
    std::array<KernelBinary, 3> *kernels, aclrtStream stream, DeviceBuffers *device, ProbeState *state,
    std::vector<WorkerResult> *workers, std::vector<WaveResult> *waves
) {
    struct HomeCandidate {
        void *target;
        size_t candidate;
        uint64_t die0_median;
        uint64_t die1_median;
    };

    const Topology die0_topology{1U, 0U, 1U, kMaxAivWorkers, kMaxAivWorkers, 0U};
    const Topology die1_topology{1U, 0U, 1U, kMaxAivWorkers, kMaxAivWorkers, kDie0AivWorkers};
    KernelBinary &kernel = (*kernels)[static_cast<size_t>(Scenario::Aiv)];
    const size_t candidate_count = kCandidateHugePages * kCandidateOffsetsPerPage;
    std::array<std::vector<HomeCandidate>, 2> home_candidates;
    for (size_t candidate = 0U; candidate < candidate_count; ++candidate) {
        void *candidate_target = CandidateTarget(*device, candidate);
        std::array<std::vector<uint64_t>, 2> spans;
        std::array<uint32_t, 2> hardware_core_ids{};
        for (uint32_t repeat = 0U; repeat < kHomeClassificationRepeats; ++repeat) {
            for (uint32_t die = 0U; die < 2U; ++die) {
                RunObservation observation{};
                const Topology &topology = die == 0U ? die0_topology : die1_topology;
                if (!RunOnce(
                        kernel, stream, *device, candidate_target, state, workers, waves, nullptr, Scenario::Aiv, 1U,
                        repeat, &topology, die == 0U ? "classify_die0" : "classify_die1", &observation, false,
                        static_cast<int32_t>(die)
                    )) {
                    return false;
                }
                spans[die].push_back(observation.launch_median);
                if (observation.active_hardware_core_ids.size() != 1U) return false;
                hardware_core_ids[die] = observation.active_hardware_core_ids[0];
            }
        }
        const uint64_t die0_median = Median(spans[0]);
        const uint64_t die1_median = Median(spans[1]);
        int32_t home = -1;
        if (die0_median + kHomeClassificationMinGapTicks <= die1_median) home = 0;
        if (die1_median + kHomeClassificationMinGapTicks <= die0_median) home = 1;
        const uintptr_t target_address = reinterpret_cast<uintptr_t>(candidate_target);
        std::printf(
            "[GM_CLASSIFY] candidate=%03zu target=0x%llx die0_core=%u die1_core=%u "
            "die0_median=%llu die1_median=%llu home=%s\n",
            candidate, static_cast<unsigned long long>(target_address), hardware_core_ids[0], hardware_core_ids[1],
            static_cast<unsigned long long>(die0_median), static_cast<unsigned long long>(die1_median),
            home < 0 ? "AMBIGUOUS" : (home == 0 ? "GM0" : "GM1")
        );
        if (home >= 0) {
            home_candidates[static_cast<size_t>(home)].push_back(
                HomeCandidate{candidate_target, candidate, die0_median, die1_median}
            );
        }
    }
    if (home_candidates[0].empty() || home_candidates[1].empty()) {
        std::fprintf(
            stderr, "Cannot find both GM0-local and GM1-local target addresses from %zu candidates.\n", candidate_count
        );
        return false;
    }

    const HomeCandidate *selected_gm0 = nullptr;
    const HomeCandidate *selected_gm1 = nullptr;
    uint64_t selected_local_gap = std::numeric_limits<uint64_t>::max();
    for (const HomeCandidate &gm0 : home_candidates[0]) {
        for (const HomeCandidate &gm1 : home_candidates[1]) {
            const uint64_t local_gap = gm0.die0_median > gm1.die1_median ? gm0.die0_median - gm1.die1_median :
                                                                           gm1.die1_median - gm0.die0_median;
            if (local_gap < selected_local_gap) {
                selected_gm0 = &gm0;
                selected_gm1 = &gm1;
                selected_local_gap = local_gap;
            }
        }
    }
    if (selected_gm0 == nullptr || selected_gm1 == nullptr || selected_local_gap > kHomePairMaxLocalGapTicks) {
        std::fprintf(
            stderr, "Cannot pair GM0/GM1 targets with home-local AIV gap <= %llu ticks; best gap=%llu ticks.\n",
            static_cast<unsigned long long>(kHomePairMaxLocalGapTicks),
            static_cast<unsigned long long>(selected_local_gap)
        );
        return false;
    }
    device->selected_targets[0] = selected_gm0->target;
    device->selected_targets[1] = selected_gm1->target;
    std::printf(
        "[GM_CLASSIFY] selected GM0 candidate=%03zu target=0x%llx local=%llu; "
        "GM1 candidate=%03zu target=0x%llx local=%llu; local_gap=%llu\n",
        selected_gm0->candidate,
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(device->selected_targets[0])),
        static_cast<unsigned long long>(selected_gm0->die0_median), selected_gm1->candidate,
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(device->selected_targets[1])),
        static_cast<unsigned long long>(selected_gm1->die1_median), static_cast<unsigned long long>(selected_local_gap)
    );
    return true;
}

bool BankTargetRangeValid(const DeviceBuffers &device) {
    const uintptr_t pool_begin = reinterpret_cast<uintptr_t>(device.target_pool);
    const uintptr_t pool_end = pool_begin + kStatePoolBytes;
    const uintptr_t target = reinterpret_cast<uintptr_t>(device.selected_targets[0]);
    const size_t max_span =
        TargetSpanBytes(AddressLayout::ParticipantStride, kBankStrides.back(), kBankPopulations.back());
    if (target < pool_begin || target > pool_end || max_span > pool_end - target) {
        std::fprintf(stderr, "GM0 target does not have %zu bytes available inside the target pool.\n", max_span);
        return false;
    }
    const size_t allocation_offset = target - pool_begin;
    const size_t huge_page_offset = allocation_offset % kCandidateHugePageBytes;
    if (target % 512U != 0U || huge_page_offset + max_span > kCandidateHugePageBytes) {
        std::fprintf(
            stderr,
            "GM0 target is unsuitable for the bank sweep: target=0x%llx, 512B_aligned=%s, "
            "huge_page_offset=%zu, max_span=%zu.\n",
            static_cast<unsigned long long>(target), target % 512U == 0U ? "yes" : "no", huge_page_offset, max_span
        );
        return false;
    }
    std::printf(
        "[BANK] GM0 base=0x%llx, allocation_offset=%zu, huge_page_offset=%zu, max_span=%zu bytes\n",
        static_cast<unsigned long long>(target), allocation_offset, huge_page_offset, max_span
    );
    return true;
}

uint32_t RotatedIndex(uint32_t offset, uint32_t count, uint32_t sweep) {
    const uint32_t ordered = sweep % 2U == 0U ? offset : count - 1U - offset;
    return (ordered + sweep) % count;
}

bool RunBankConflictProbe(
    std::array<KernelBinary, 3> *kernels, aclrtStream stream, const DeviceBuffers &device, ProbeState *state,
    std::vector<WorkerResult> *workers, std::vector<WaveResult> *waves, std::ofstream &raw
) {
    if (!BankTargetRangeValid(device)) return false;
    WriteBankRawHeader(raw);
    auto *base = static_cast<uint8_t *>(device.selected_targets[0]);
    const std::array<Scenario, 3> scenarios = {Scenario::Mixed, Scenario::Aic, Scenario::Aiv};
    const std::array<Scenario, 2> calibration_scenarios = {Scenario::Aic, Scenario::Aiv};

    for (uint32_t sweep = 0U; sweep < kBankSweeps; ++sweep) {
        for (uint32_t gap_offset = 0U; gap_offset < kBankGapCount; ++gap_offset) {
            const uint32_t gap_index = RotatedIndex(gap_offset, kBankGapCount, sweep);
            const uint32_t gap_bytes = gap_index * kBankGapStrideBytes;
            for (uint32_t scenario_offset = 0U; scenario_offset < calibration_scenarios.size(); ++scenario_offset) {
                const Scenario scenario =
                    calibration_scenarios[(scenario_offset + sweep + gap_index) % calibration_scenarios.size()];
                BankRawSpec spec{&raw, "single_address", gap_bytes, AddressLayout::Shared, 0U};
                if (!RunOnce(
                        (*kernels)[static_cast<size_t>(scenario)], stream, device, base + gap_bytes, state, workers,
                        waves, nullptr, scenario, 1U, sweep, nullptr, ScenarioName(scenario), nullptr, false, 0, &spec
                    )) {
                    return false;
                }
            }
            for (uint32_t scenario_offset = 0U; scenario_offset < scenarios.size(); ++scenario_offset) {
                const Scenario scenario = scenarios[(scenario_offset + sweep + gap_index) % scenarios.size()];
                const AddressLayout address_layout =
                    gap_bytes == 0U ? AddressLayout::Shared : AddressLayout::ParticipantStride;
                BankRawSpec spec{&raw, "pair_gap", 0U, address_layout, gap_bytes};
                if (!RunOnce(
                        (*kernels)[static_cast<size_t>(scenario)], stream, device, base, state, workers, waves, nullptr,
                        scenario, 2U, sweep, nullptr, ScenarioName(scenario), nullptr, false, 0, &spec
                    )) {
                    return false;
                }
            }
        }
        for (uint32_t phase_offset = 0U; phase_offset < kBankPhaseCount; ++phase_offset) {
            const uint32_t phase_index = RotatedIndex(phase_offset, kBankPhaseCount, sweep);
            const uint32_t phase_bytes = phase_index * kBankGapStrideBytes;
            for (uint32_t scenario_offset = 0U; scenario_offset < scenarios.size(); ++scenario_offset) {
                const Scenario scenario = scenarios[(scenario_offset + sweep + phase_index) % scenarios.size()];
                for (uint32_t adjacent = 0U; adjacent < 2U; ++adjacent) {
                    const AddressLayout address_layout =
                        adjacent == 0U ? AddressLayout::Shared : AddressLayout::ParticipantStride;
                    const uint32_t stride_bytes = adjacent == 0U ? 0U : kBankGapStrideBytes;
                    BankRawSpec spec{
                        &raw, adjacent == 0U ? "pair_phase_same" : "pair_phase_64", phase_bytes, address_layout,
                        stride_bytes
                    };
                    if (!RunOnce(
                            (*kernels)[static_cast<size_t>(scenario)], stream, device, base + phase_bytes, state,
                            workers, waves, nullptr, scenario, 2U, sweep, nullptr, ScenarioName(scenario), nullptr,
                            false, 0, &spec
                        )) {
                        return false;
                    }
                }
            }
        }
        raw.flush();
        std::printf(
            "[BANK] pair sweep %u/%u PASS: %u gaps, %u adjacent-line phases, "
            "AIC/AIV single-address calibration, C+C/V+V/C+V pairs\n",
            sweep + 1U, kBankSweeps, kBankGapCount, kBankPhaseCount
        );
    }

    for (uint32_t sweep = 0U; sweep < kBankSweeps; ++sweep) {
        for (uint32_t scenario_offset = 0U; scenario_offset < scenarios.size(); ++scenario_offset) {
            const Scenario scenario = scenarios[(scenario_offset + sweep) % scenarios.size()];
            for (uint32_t stride_offset = 0U; stride_offset < kBankStrides.size(); ++stride_offset) {
                const uint32_t stride_index = RotatedIndex(stride_offset, kBankStrides.size(), sweep);
                const uint32_t stride_bytes = kBankStrides[stride_index];
                std::vector<uint32_t> populations;
                for (uint32_t population : kBankPopulations) {
                    if (population <= ScenarioCapacity(scenario)) populations.push_back(population);
                }
                if ((sweep + stride_index) % 2U != 0U) std::reverse(populations.begin(), populations.end());
                for (uint32_t population : populations) {
                    const AddressLayout address_layout =
                        stride_bytes == 0U ? AddressLayout::Shared : AddressLayout::ParticipantStride;
                    BankRawSpec spec{&raw, "multicore_curve", 0U, address_layout, stride_bytes};
                    if (!RunOnce(
                            (*kernels)[static_cast<size_t>(scenario)], stream, device, base, state, workers, waves,
                            nullptr, scenario, population, sweep, nullptr, ScenarioName(scenario), nullptr, false, 0,
                            &spec
                        )) {
                        return false;
                    }
                }
            }
        }
        raw.flush();
        std::printf(
            "[BANK] multicore sweep %u/%u PASS: shared control plus %zu distinct-address strides\n", sweep + 1U,
            kBankSweeps, kBankStrides.size() - 1U
        );
    }

    struct TwoGroupPattern {
        const char *experiment;
        uint32_t target_offset_bytes;
        AddressLayout address_layout;
        uint32_t target_stride_bytes;
    };
    const std::array<TwoGroupPattern, 6> two_group_patterns{{
        {"two_group_same", 0U, AddressLayout::Shared, 0U},
        {"two_group_64_same128", 0U, AddressLayout::TwoGroups, 64U},
        {"two_group_64_cross128", 64U, AddressLayout::TwoGroups, 64U},
        {"two_group_128", 0U, AddressLayout::TwoGroups, 128U},
        {"two_group_512", 0U, AddressLayout::TwoGroups, 512U},
        {"two_group_1024", 0U, AddressLayout::TwoGroups, 1024U},
    }};
    for (uint32_t sweep = 0U; sweep < kBankSweeps; ++sweep) {
        for (uint32_t scenario_offset = 0U; scenario_offset < scenarios.size(); ++scenario_offset) {
            const Scenario scenario = scenarios[(scenario_offset + sweep) % scenarios.size()];
            std::vector<uint32_t> populations;
            for (uint32_t population : kBankPopulations) {
                if (population >= 2U && population <= ScenarioCapacity(scenario)) populations.push_back(population);
            }
            if (sweep % 2U != 0U) std::reverse(populations.begin(), populations.end());
            for (uint32_t pattern_offset = 0U; pattern_offset < two_group_patterns.size(); ++pattern_offset) {
                const TwoGroupPattern &pattern =
                    two_group_patterns[(pattern_offset + sweep) % two_group_patterns.size()];
                for (uint32_t population : populations) {
                    const Topology topology = MakeTopology(scenario, population);
                    const uint32_t primary_group_workers =
                        scenario == Scenario::Mixed ? topology.active_aic : (topology.active + 1U) / 2U;
                    BankRawSpec spec{
                        &raw,
                        pattern.experiment,
                        pattern.target_offset_bytes,
                        pattern.address_layout,
                        pattern.target_stride_bytes,
                        pattern.address_layout == AddressLayout::TwoGroups ? primary_group_workers : 0U
                    };
                    if (!RunOnce(
                            (*kernels)[static_cast<size_t>(scenario)], stream, device,
                            base + pattern.target_offset_bytes, state, workers, waves, nullptr, scenario, population,
                            sweep, nullptr, ScenarioName(scenario), nullptr, false, 0, &spec
                        )) {
                        return false;
                    }
                }
            }
        }
        raw.flush();
        std::printf(
            "[BANK] two-group sweep %u/%u PASS: %zu layouts, pure roles split nearly half, mixed split by AIC/AIV\n",
            sweep + 1U, kBankSweeps, two_group_patterns.size()
        );
    }
    return true;
}

std::vector<uint32_t> SupportedPopulations(Scenario scenario, bool descending) {
    std::vector<uint32_t> result;
    for (uint32_t population : kRequestedPopulations) {
        if (population <= ScenarioCapacity(scenario)) result.push_back(population);
    }
    if (descending) std::reverse(result.begin(), result.end());
    return result;
}

bool ResourceLimitsValid(aclrtStream stream) {
    uint32_t cube_limit = 0U;
    uint32_t vector_limit = 0U;
    if (!Check(aclrtGetStreamResLimit(stream, ACL_RT_DEV_RES_CUBE_CORE, &cube_limit), "query cube-core limit") ||
        !Check(aclrtGetStreamResLimit(stream, ACL_RT_DEV_RES_VECTOR_CORE, &vector_limit), "query vector-core limit")) {
        return false;
    }
    std::printf("[TOPOLOGY] runtime cube_limit=%u vector_limit=%u\n", cube_limit, vector_limit);
    if (cube_limit < kMaxAicWorkers || vector_limit < kMaxAivWorkers) {
        std::fprintf(
            stderr, "A5 topology is smaller than the preregistered 32C+64V sweep: cube=%u vector=%u\n", cube_limit,
            vector_limit
        );
        return false;
    }
    return true;
}

bool AllocateBuffers(DeviceBuffers *device) {
    return Check(aclrtMalloc(&device->state, sizeof(ProbeState), ACL_MEM_MALLOC_HUGE_FIRST), "allocate curve state") &&
           Check(
               aclrtMalloc(&device->target_pool, kStatePoolBytes, ACL_MEM_MALLOC_HUGE_FIRST),
               "allocate GM-home candidate target pool"
           ) &&
           Check(
               aclrtMalloc(&device->results, sizeof(ProbeResults), ACL_MEM_MALLOC_HUGE_FIRST), "allocate curve results"
           );
}

bool Cleanup(
    DeviceBuffers *device, std::array<KernelBinary, 3> *kernels, aclrtStream stream, int32_t device_id,
    bool device_initialized
) {
    bool valid = true;
    if (device->results != nullptr) valid &= Check(aclrtFree(device->results), "free curve results");
    if (device->target_pool != nullptr) valid &= Check(aclrtFree(device->target_pool), "free GM-home target pool");
    if (device->state != nullptr) valid &= Check(aclrtFree(device->state), "free curve state");
    for (auto iterator = kernels->rbegin(); iterator != kernels->rend(); ++iterator) {
        if (iterator->binary != nullptr) valid &= Check(aclrtBinaryUnLoad(iterator->binary), "unload curve ELF");
    }
    if (stream != nullptr) valid &= Check(aclrtDestroyStream(stream), "destroy curve stream");
    if (device_initialized) valid &= Check(aclrtResetDevice(device_id), "reset curve device");
    valid &= Check(aclFinalize(), "finalize ACL");
    return valid;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 6) {
        std::fprintf(
            stderr,
            "Usage: %s MIXED_KERNEL AIC_KERNEL AIV_KERNEL RAW_TSV LOCALITY_TSV\n"
            "       %s MIXED_KERNEL AIC_KERNEL AIV_KERNEL --bank BANK_RAW_TSV\n",
            argc > 0 ? argv[0] : "curve_host", argc > 0 ? argv[0] : "curve_host"
        );
        return EXIT_FAILURE;
    }
    const bool bank_mode = std::string(argv[4]) == "--bank";

    std::array<KernelBinary, 3> kernels{{KernelBinary{argv[1]}, KernelBinary{argv[2]}, KernelBinary{argv[3]}}};
    const int32_t device_id = atomic_probe::DeviceId();
    if (device_id < 0 || !Check(aclInit(nullptr), "initialize ACL")) return EXIT_FAILURE;
    bool device_initialized = false;
    aclrtStream stream = nullptr;
    DeviceBuffers device{};
    bool valid = Check(aclrtSetDevice(device_id), "set curve device");
    device_initialized = valid;
    if (valid) valid = Check(aclrtCreateStream(&stream), "create curve stream");
    if (valid) valid = ResourceLimitsValid(stream);
    for (KernelBinary &kernel : kernels) {
        if (valid) valid = LoadKernel(&kernel);
    }
    if (valid) valid = AllocateBuffers(&device);

    auto state = ProbeState{};
    std::vector<WorkerResult> workers(kMaxWorkers);
    std::vector<WaveResult> waves(static_cast<size_t>(kMaxWorkers) * kTotalWaves);
    if (valid) valid = SelectGmHomes(&kernels, stream, &device, &state, &workers, &waves);

    std::ofstream raw;
    std::ofstream locality_raw;
    if (valid) {
        raw.open(bank_mode ? argv[5] : argv[4], std::ios::out | std::ios::trunc);
        if (!bank_mode) locality_raw.open(argv[5], std::ios::out | std::ios::trunc);
        if (!raw || (!bank_mode && !locality_raw)) {
            std::fprintf(stderr, "Cannot create raw result file.\n");
            valid = false;
        }
    }
    if (valid && bank_mode) {
        valid = RunBankConflictProbe(&kernels, stream, device, &state, &workers, &waves, raw);
    } else if (valid) {
        WriteRawHeader(raw);
        WriteRawHeader(locality_raw);
        WriteUnsupportedRows(raw, device);
        const std::array<Scenario, 3> base_order = {Scenario::Mixed, Scenario::Aic, Scenario::Aiv};
        for (uint32_t sweep = 0U; valid && sweep < kSweeps; ++sweep) {
            for (uint32_t offset = 0U; valid && offset < base_order.size(); ++offset) {
                const Scenario scenario = base_order[(offset + sweep) % base_order.size()];
                KernelBinary &kernel = kernels[static_cast<size_t>(scenario)];
                const std::vector<uint32_t> populations = SupportedPopulations(scenario, sweep % 2U != 0U);
                for (uint32_t population : populations) {
                    for (uint32_t home_offset = 0U; home_offset < 2U; ++home_offset) {
                        const uint32_t gm_home = (home_offset + sweep + population) % 2U;
                        if (!RunOnce(
                                kernel, stream, device, device.selected_targets[gm_home], &state, &workers, &waves,
                                &raw, scenario, population, sweep, nullptr, CurveProfileName(scenario, gm_home),
                                nullptr, true, 0
                            )) {
                            valid = false;
                            break;
                        }
                    }
                    if (!valid) break;
                }
            }
        }
        if (valid) {
            valid = RunDieLocality(&kernels, stream, device, &state, &workers, &waves, locality_raw);
        }
    }
    raw.close();
    if (!bank_mode) locality_raw.close();

    const bool cleanup_valid = Cleanup(&device, &kernels, stream, device_id, device_initialized);
    if (valid && cleanup_valid) {
        if (bank_mode) {
            std::printf(
                "[SUMMARY] PASS: bank-conflict pair and multicore probes, %u sweeps, "
                "%u warmup + %u measured waves per launch; raw=%s\n",
                kBankSweeps, kWarmupWaves, kMeasuredWaves, argv[5]
            );
        } else {
            std::printf(
                "[SUMMARY] PASS: %u sweeps, %u warmup + %u measured waves per launch; raw=%s locality=%s\n", kSweeps,
                kWarmupWaves, kMeasuredWaves, argv[4], argv[5]
            );
        }
        return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
}
