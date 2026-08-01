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

#include "scalar_coroutine_probe_shared.h"
#include "pmu_probe_host_support.h"
#include "../probe_host.h"
#include "../pa_scheduler/ccec/pmu_owner_host.h"

#include "acl/acl.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

using atomic_probe::pmu::CheckAcl;
using scalar_coroutine_probe::Mode;
using scalar_coroutine_probe::Role;

constexpr double kAicoreCyclesPerNanosecond = 1.65;
constexpr uint32_t kMeasuredRepeats = 5U;
constexpr uint32_t kTaskSeed = 0x2468ace0U;
constexpr uint64_t kRequiredSelectorStatus = (1ULL << 9U) - 1U;
constexpr std::array<uint32_t, 6U> kIterationValues = {
    0U, 16U, 64U, 256U, 1024U, 4096U,
};

const char *RoleName(Role role) {
    return role == Role::Aic ? "AIC" : "AIV";
}

const char *ModeName(Mode mode) {
    switch (mode) {
    case Mode::ContextFifo: return "CONTEXT_FIFO";
    case Mode::EngineThenSchedule: return "ENGINE_THEN_SCHEDULE";
    case Mode::EngineOverlapSchedule: return "ENGINE_OVERLAP_SCHEDULE";
    case Mode::TryWaitCrossCore: return "TRY_WAIT_CROSS_CORE";
    case Mode::TryWaitIntraBlock: return "TRY_WAIT_INTRA_BLOCK";
    case Mode::TryWaitBufferId: return "TRY_WAIT_BUFFER_ID";
    case Mode::TryWaitTaggedBufferId:
        return "TRY_WAIT_TAGGED_BUFFER_ID";
    case Mode::TryWaitTaggedDynamicBufferId:
        return "TRY_WAIT_TAGGED_DYNAMIC_BUFFER_ID";
    case Mode::TryWaitPollUntilDone:
        return "TRY_WAIT_POLL_UNTIL_DONE";
    case Mode::TryWaitFourSlots:
        return "TRY_WAIT_FOUR_SLOTS";
    case Mode::TryWaitIdleCost:
        return "TRY_WAIT_IDLE_COST";
    default: return "UNKNOWN";
    }
}

struct LoadedKernel {
    std::vector<char> image;
    aclrtBinHandle binary = nullptr;
    aclrtFuncHandle function = nullptr;
};

bool LoadKernel(const std::string &path, LoadedKernel *kernel) {
    kernel->image = atomic_probe::pmu::ReadBinary(path);
    if (kernel->image.empty()) {
        std::fprintf(stderr, "Cannot read kernel binary: %s\n", path.c_str());
        return false;
    }
    return CheckAcl(
               atomic_probe::LoadAicoreBinaryFromData(
                   kernel->image.data(), kernel->image.size(), &kernel->binary
               ),
               "LoadAicoreBinaryFromData(coroutine probe)"
           ) &&
           CheckAcl(
               aclrtBinaryGetFunctionByEntry(
                   kernel->binary, 0U, &kernel->function
               ),
               "aclrtBinaryGetFunctionByEntry(coroutine probe)"
           );
}

struct ContextOracle {
    uint64_t words[scalar_coroutine_probe::kContextWords]{};
};

ContextOracle MakeContext(uint32_t seed, uint64_t task_id) {
    ContextOracle context{};
    for (uint32_t index = 0U;
         index < scalar_coroutine_probe::kContextWords; ++index) {
        context.words[index] = scalar_coroutine_probe::Mix(
            (static_cast<uint64_t>(seed) << 32U) ^ task_id ^
            (static_cast<uint64_t>(index + 1U) *
             0x9e3779b97f4a7c15ULL)
        );
    }
    context.words[0] = task_id;
    return context;
}

uint64_t ContextSignature(const ContextOracle &context) {
    uint64_t signature = 0x6a09e667f3bcc909ULL;
    for (uint32_t index = 0U;
         index < scalar_coroutine_probe::kContextWords; ++index) {
        signature = scalar_coroutine_probe::Mix(
            signature ^ context.words[index] ^
            (static_cast<uint64_t>(index) << 56U)
        );
    }
    return signature;
}

struct ReplayOracle {
    uint64_t checksum = 0U;
    uint64_t context0_signature = 0U;
    uint64_t context1_signature = 0U;
};

ReplayOracle CalculateOracle(
    const std::vector<scalar_coroutine_probe::TaskRecord> &records,
    uint32_t iterations, uint32_t seed
) {
    ReplayOracle oracle{};
    oracle.context0_signature = ContextSignature(
        MakeContext(seed, 0x100000001ULL)
    );
    uint32_t cursor = seed & (scalar_coroutine_probe::kTaskRecords - 1U);
    uint64_t checksum = scalar_coroutine_probe::Mix(
        static_cast<uint64_t>(seed) ^ 0xbb67ae8584caa73bULL
    );
    for (uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        const auto &record = records[cursor];
        const uint64_t payload = record.payload[iteration % 5U];
        checksum = scalar_coroutine_probe::Mix(
            checksum ^ record.task_id ^ payload ^
            (record.ready_token + static_cast<uint64_t>(iteration))
        );
        cursor = (cursor + 1U + static_cast<uint32_t>(record.next_delta) +
                  static_cast<uint32_t>(
                      (record.ready_token ^ checksum) & 1ULL
                  )) &
                 (scalar_coroutine_probe::kTaskRecords - 1U);
    }
    ContextOracle second = MakeContext(
        seed ^ static_cast<uint32_t>(checksum),
        0x200000000ULL | cursor
    );
    second.words[1] ^= checksum;
    oracle.checksum = checksum;
    oracle.context1_signature = ContextSignature(second);
    return oracle;
}

std::vector<scalar_coroutine_probe::TaskRecord> MakeTaskRecords() {
    std::vector<scalar_coroutine_probe::TaskRecord> records(
        scalar_coroutine_probe::kTaskRecords
    );
    for (uint32_t record = 0U;
         record < scalar_coroutine_probe::kTaskRecords; ++record) {
        records[record].task_id = record;
        records[record].next_delta =
            scalar_coroutine_probe::TaskWord(record, 0U, kTaskSeed) & 3ULL;
        records[record].ready_token =
            scalar_coroutine_probe::TaskWord(record, 1U, kTaskSeed);
        for (uint32_t word = 0U; word < 5U; ++word) {
            records[record].payload[word] =
                scalar_coroutine_probe::TaskWord(
                    record, word + 2U, kTaskSeed
                );
        }
    }
    return records;
}

struct Sample {
    scalar_coroutine_probe::ProbeResult result{};
    bool output_ok = false;
    uint32_t configured_buffer_id = 0U;
};

void PrintTryWaitResult(
    Role role, Mode mode, uint32_t iterations, const Sample &sample
) {
    const auto &result = sample.result;
    std::printf(
        "[TRY_WAIT] role=%s mode=%s iterations=%u buffer_id=%u "
        "before_issue=%lld after_issue=%lld after_schedule=%lld "
        "after_final_wait=%lld positive_polls=%llu poll_final=%lld\n",
        RoleName(role), ModeName(mode), iterations,
        sample.configured_buffer_id,
        static_cast<long long>(result.try_wait_before_issue),
        static_cast<long long>(result.try_wait_after_issue),
        static_cast<long long>(result.try_wait_after_schedule),
        static_cast<long long>(result.try_wait_after_final_wait),
        static_cast<unsigned long long>(
            result.try_wait_positive_poll_count
        ),
        static_cast<long long>(result.try_wait_poll_final_value)
    );
}

bool ValidateSample(
    const Sample &sample, Role role, Mode mode, uint32_t iterations,
    const ReplayOracle &oracle,
    const pa_scheduler::pmu_owner::PmuOwnerControl &owner,
    std::string *reason
) {
    const auto &result = sample.result;
    if (result.observed_role != static_cast<uint64_t>(role) ||
        result.observed_mode != static_cast<uint64_t>(mode) ||
        result.observed_iterations != iterations) {
        *reason = "stale-or-mismatched-control";
        return false;
    }
    if (result.physical_core_id >=
            pa_scheduler::pmu_owner::kPhysicalSubcoreCount ||
        !pa_scheduler::pmu_owner::IsConfigured(
            owner, static_cast<uint32_t>(result.physical_core_id)
        )) {
        *reason = "physical-core-not-owned";
        return false;
    }
    if (result.selector_status != kRequiredSelectorStatus) {
        *reason = "selector-map";
        return false;
    }
    if ((result.pmu_ctrl_after_stop & 1ULL) != 0U) {
        *reason = "pmu-gate-still-enabled";
        return false;
    }
    if (result.sys_ticks == 0U || result.pmu_total_cycles == 0U ||
        result.pmu_scalar_busy == 0U) {
        *reason = "zero-cycle-window";
        return false;
    }
    const uint64_t busy_counters[] = {
        result.pmu_vector_busy, result.pmu_cube_busy,
        result.pmu_scalar_busy, result.pmu_mte1_busy,
        result.pmu_mte2_busy, result.pmu_mte3_busy,
        result.pmu_fix_busy,
    };
    for (const uint64_t counter : busy_counters) {
        if (counter > result.pmu_total_cycles) {
            *reason = "busy-counter-exceeds-total";
            return false;
        }
    }
    if (result.pmu_icache_miss > result.pmu_icache_request) {
        *reason = "icache-miss-exceeds-request";
        return false;
    }
    if (result.context0_signature != oracle.context0_signature ||
        result.restored_context0_signature != oracle.context0_signature ||
        result.context1_signature != oracle.context1_signature ||
        result.restored_context1_signature != oracle.context1_signature ||
        result.schedule_checksum != oracle.checksum) {
        *reason = "continuation-or-replay-oracle";
        return false;
    }
    if (result.resume_sequence != 6U ||
        result.protocol_status !=
            scalar_coroutine_probe::kRequiredProtocolStatus) {
        *reason = "continuation-protocol";
        return false;
    }
    const bool idle_cost_mode = mode == Mode::TryWaitIdleCost;
    const bool use_engine = mode != Mode::ContextFifo && !idle_cost_mode;
    if (result.completion_before_wait != 0U ||
        result.completion_after_wait != (use_engine ? 1U : 0U)) {
        *reason = "completion-before-final-wait";
        return false;
    }
    if (use_engine && !sample.output_ok) {
        *reason = "engine-output";
        return false;
    }
    if (use_engine && role == Role::Aiv &&
        (result.pmu_vector_busy == 0U || result.pmu_mte2_busy == 0U ||
         result.pmu_mte3_busy == 0U)) {
        *reason = "aiv-pipeline-counter-zero";
        return false;
    }
    if (use_engine && role == Role::Aic &&
        (result.pmu_cube_busy == 0U || result.pmu_mte1_busy == 0U ||
         result.pmu_mte2_busy == 0U || result.pmu_fix_busy == 0U)) {
        *reason = "aic-pipeline-counter-zero";
        return false;
    }
    const auto signed_value = [](uint64_t value) {
        return static_cast<int64_t>(value);
    };
    if (mode == Mode::TryWaitTaggedBufferId ||
        mode == Mode::TryWaitTaggedDynamicBufferId) {
        if (signed_value(result.try_wait_before_issue) != 0 ||
            signed_value(result.try_wait_after_issue) != 1 ||
            signed_value(result.try_wait_after_schedule) < 0 ||
            signed_value(result.try_wait_after_schedule) > 1 ||
            signed_value(result.try_wait_after_final_wait) != 0) {
            *reason = "tagged-buffer-id-transition";
            return false;
        }
    }
    if (mode == Mode::TryWaitPollUntilDone &&
        (signed_value(result.try_wait_before_issue) != 0 ||
         signed_value(result.try_wait_after_issue) != 1 ||
         signed_value(result.try_wait_after_schedule) != 0 ||
         signed_value(result.try_wait_after_final_wait) != 0 ||
         result.try_wait_positive_poll_count == 0U ||
         signed_value(result.try_wait_poll_final_value) != 0)) {
        *reason = "try-wait-poll-transition";
        return false;
    }
    if (mode == Mode::TryWaitFourSlots &&
        (signed_value(result.try_wait_before_issue) != 0 ||
         signed_value(result.try_wait_after_issue) != 0xf ||
         (result.try_wait_after_schedule & ~0xfULL) != 0U ||
         signed_value(result.try_wait_after_final_wait) != 0)) {
        *reason = "four-slot-buffer-id-transition";
        return false;
    }
    if (idle_cost_mode &&
        signed_value(result.try_wait_poll_final_value) != 0) {
        *reason = "idle-buffer-query-nonzero";
        return false;
    }
    return true;
}

bool RunOne(
    const LoadedKernel &kernel, Role role, Mode mode, uint32_t iterations,
    uint32_t seed, aclrtStream stream, void *state_device,
    void *input_a_device, void *input_b_device, void *output_device,
    void *task_records_device, uint64_t pmu_register_bases,
    const std::vector<scalar_coroutine_probe::TaskRecord> &records,
    const pa_scheduler::pmu_owner::PmuOwnerControl &owner,
    Sample *sample, bool print_raw
) {
    scalar_coroutine_probe::ProbeState state{};
    state.control.pmu_register_bases = pmu_register_bases;
    state.control.input_a = reinterpret_cast<uint64_t>(input_a_device);
    state.control.input_b = reinterpret_cast<uint64_t>(input_b_device);
    state.control.output = reinterpret_cast<uint64_t>(output_device);
    state.control.task_records = reinterpret_cast<uint64_t>(
        task_records_device
    );
    state.control.mode = static_cast<uint32_t>(mode);
    state.control.schedule_iterations = iterations;
    state.control.seed = seed;
    state.control.magic = scalar_coroutine_probe::kControlMagic;
    state.control.try_wait_buffer_id =
        mode == Mode::TryWaitTaggedDynamicBufferId
        ? 24U + ((seed >> 12U) & 7U)
        : mode == Mode::TryWaitFourSlots ? 24U
        : 31U;
    if (!CheckAcl(
            aclrtMemcpy(
                state_device, sizeof(state), &state, sizeof(state),
                ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy(H2D coroutine state)"
        )) {
        return false;
    }
    const bool idle_cost_mode = mode == Mode::TryWaitIdleCost;
    const bool use_engine = mode != Mode::ContextFifo && !idle_cost_mode;
    if (use_engine && !CheckAcl(
            aclrtMemset(
                output_device, scalar_coroutine_probe::kTileBytes, 0xff,
                scalar_coroutine_probe::kTileBytes
            ),
            "aclrtMemset(coroutine output sentinel)"
        )) {
        return false;
    }

    struct KernelArgs {
        uint64_t state_pointer;
    } args{reinterpret_cast<uint64_t>(state_device)};
    static_assert(
        sizeof(KernelArgs) == sizeof(uint64_t),
        "unexpected CCEC kernel argument ABI"
    );
    if (!CheckAcl(
            aclrtLaunchKernelWithHostArgs(
                kernel.function, 1U, stream, nullptr, &args, sizeof(args),
                nullptr, 0U
            ),
            "aclrtLaunchKernelWithHostArgs(coroutine probe)"
        ) ||
        !CheckAcl(
            aclrtSynchronizeStream(stream),
            "aclrtSynchronizeStream(coroutine probe)"
        ) ||
        !CheckAcl(
            aclrtMemcpy(
                &state, sizeof(state), state_device, sizeof(state),
                ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "aclrtMemcpy(D2H coroutine state)"
        )) {
        return false;
    }

    sample->result = state.result;
    sample->configured_buffer_id = state.control.try_wait_buffer_id;
    sample->output_ok = !use_engine;
    if (use_engine) {
        std::vector<float> output(scalar_coroutine_probe::kTileElements);
        if (!CheckAcl(
                aclrtMemcpy(
                    output.data(), scalar_coroutine_probe::kTileBytes,
                    output_device, scalar_coroutine_probe::kTileBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "aclrtMemcpy(D2H coroutine output)"
            )) {
            return false;
        }
        const float expected = role == Role::Aic ? 768.0F : 5.0F;
        sample->output_ok = std::all_of(
            output.begin(), output.end(), [expected](float value) {
                return value == expected;
            }
        );
    }

    const ReplayOracle oracle = CalculateOracle(
        records,
        mode == Mode::TryWaitIdleCost ? 0U : iterations,
        seed
    );
    std::string reason;
    const bool passed = ValidateSample(
        *sample, role, mode, iterations, oracle, owner, &reason
    );
    if (print_raw) {
        std::printf(
            "[RAW] role=%s mode=%s iterations=%u seed=0x%08x "
            "sys_ticks=%llu total=%llu scalar=%llu vector=%llu cube=%llu "
            "mte1=%llu mte2=%llu mte3=%llu fix=%llu icache_req=%llu "
            "icache_miss=%llu resume=%llu protocol=0x%llx output=%s "
            "status=%s%s%s\n",
            RoleName(role), ModeName(mode), iterations, seed,
            static_cast<unsigned long long>(sample->result.sys_ticks),
            static_cast<unsigned long long>(
                sample->result.pmu_total_cycles
            ),
            static_cast<unsigned long long>(
                sample->result.pmu_scalar_busy
            ),
            static_cast<unsigned long long>(
                sample->result.pmu_vector_busy
            ),
            static_cast<unsigned long long>(
                sample->result.pmu_cube_busy
            ),
            static_cast<unsigned long long>(
                sample->result.pmu_mte1_busy
            ),
            static_cast<unsigned long long>(
                sample->result.pmu_mte2_busy
            ),
            static_cast<unsigned long long>(
                sample->result.pmu_mte3_busy
            ),
            static_cast<unsigned long long>(
                sample->result.pmu_fix_busy
            ),
            static_cast<unsigned long long>(
                sample->result.pmu_icache_request
            ),
            static_cast<unsigned long long>(
                sample->result.pmu_icache_miss
            ),
            static_cast<unsigned long long>(
                sample->result.resume_sequence
            ),
            static_cast<unsigned long long>(
                sample->result.protocol_status
            ),
            sample->output_ok ? "PASS" : "FAIL",
            passed ? "PASS" : "FAIL", passed ? "" : " reason=",
            passed ? "" : reason.c_str()
        );
    }
    return passed;
}

double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2U;
    return (values.size() & 1U) != 0U
        ? values[middle]
        : (values[middle - 1U] + values[middle]) / 2.0;
}

void PrintContextSummary(
    Role role, uint32_t iterations, const std::vector<Sample> &samples
) {
    std::vector<double> totals;
    std::vector<double> scalar;
    std::vector<double> ticks;
    for (const Sample &sample : samples) {
        totals.push_back(
            static_cast<double>(sample.result.pmu_total_cycles)
        );
        scalar.push_back(
            static_cast<double>(sample.result.pmu_scalar_busy)
        );
        ticks.push_back(static_cast<double>(sample.result.sys_ticks));
    }
    std::printf(
        "[CONTEXT] role=%s iterations=%u total_cycles=%.3f "
        "total_ns_at_1p65ghz=%.3f scalar_cycles=%.3f sys_ticks=%.3f\n",
        RoleName(role), iterations, Median(totals),
        Median(totals) / kAicoreCyclesPerNanosecond,
        Median(scalar), Median(ticks)
    );
}

void PrintPairSummary(
    Role role, uint32_t iterations,
    const std::vector<Sample> &serial,
    const std::vector<Sample> &overlap
) {
    std::vector<double> total_gains;
    std::vector<double> sys_gains;
    std::vector<double> scalar_gains;
    std::vector<double> gain_ratios;
    std::vector<double> serial_totals;
    std::vector<double> overlap_totals;
    for (size_t index = 0U; index < serial.size(); ++index) {
        const double serial_total = static_cast<double>(
            serial[index].result.pmu_total_cycles
        );
        const double overlap_total = static_cast<double>(
            overlap[index].result.pmu_total_cycles
        );
        serial_totals.push_back(serial_total);
        overlap_totals.push_back(overlap_total);
        total_gains.push_back(serial_total - overlap_total);
        sys_gains.push_back(
            static_cast<double>(serial[index].result.sys_ticks) -
            static_cast<double>(overlap[index].result.sys_ticks)
        );
        scalar_gains.push_back(
            static_cast<double>(serial[index].result.pmu_scalar_busy) -
            static_cast<double>(overlap[index].result.pmu_scalar_busy)
        );
        gain_ratios.push_back(
            (serial_total - overlap_total) / serial_total
        );
    }
    const double total_gain = Median(total_gains);
    const double sys_gain = Median(sys_gains);
    std::printf(
        "[OVERLAP] role=%s iterations=%u serial_total_cycles=%.3f "
        "overlap_total_cycles=%.3f paired_total_gain_cycles=%.3f "
        "paired_total_gain_ns=%.3f paired_sys_gain_ns=%.3f "
        "paired_scalar_gain_cycles=%.3f paired_gain_ratio=%.6f "
        "observation=%s\n",
        RoleName(role), iterations, Median(serial_totals),
        Median(overlap_totals), total_gain,
        total_gain / kAicoreCyclesPerNanosecond, sys_gain,
        Median(scalar_gains), Median(gain_ratios),
        total_gain > 0.0 && sys_gain > 0.0
            ? "OVERLAP_VISIBLE" : "NO_STABLE_GAIN"
    );
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(
            stderr,
            "Usage: %s scalar_coroutine_probe_aic_kernel.o "
            "scalar_coroutine_probe_aiv_kernel.o\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }
    const std::string aic_kernel_path = argv[1];
    const std::string aiv_kernel_path = argv[2];
    const int32_t device = atomic_probe::DeviceId();
    if (device < 0) return EXIT_FAILURE;

    if (!CheckAcl(aclInit(nullptr), "aclInit") ||
        !CheckAcl(aclrtSetDevice(device), "aclrtSetDevice")) {
        return EXIT_FAILURE;
    }
    aclrtStream stream = nullptr;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) {
        return EXIT_FAILURE;
    }
    LoadedKernel aic_kernel;
    LoadedKernel aiv_kernel;
    if (!LoadKernel(aic_kernel_path, &aic_kernel) ||
        !LoadKernel(aiv_kernel_path, &aiv_kernel)) {
        return EXIT_FAILURE;
    }

    void *state_device = nullptr;
    void *input_a_device = nullptr;
    void *input_b_device = nullptr;
    void *output_device = nullptr;
    void *task_records_device = nullptr;
    const size_t task_record_bytes =
        sizeof(scalar_coroutine_probe::TaskRecord) *
        scalar_coroutine_probe::kTaskRecords;
    if (!CheckAcl(
            aclrtMalloc(
                &state_device, sizeof(scalar_coroutine_probe::ProbeState),
                ACL_MEM_MALLOC_NORMAL_ONLY
            ),
            "aclrtMalloc(coroutine state)"
        ) ||
        !CheckAcl(
            aclrtMalloc(
                &input_a_device, scalar_coroutine_probe::kTileBytes,
                ACL_MEM_MALLOC_NORMAL_ONLY
            ),
            "aclrtMalloc(coroutine input A)"
        ) ||
        !CheckAcl(
            aclrtMalloc(
                &input_b_device, scalar_coroutine_probe::kTileBytes,
                ACL_MEM_MALLOC_NORMAL_ONLY
            ),
            "aclrtMalloc(coroutine input B)"
        ) ||
        !CheckAcl(
            aclrtMalloc(
                &output_device, scalar_coroutine_probe::kTileBytes,
                ACL_MEM_MALLOC_NORMAL_ONLY
            ),
            "aclrtMalloc(coroutine output)"
        ) ||
        !CheckAcl(
            aclrtMalloc(
                &task_records_device, task_record_bytes,
                ACL_MEM_MALLOC_NORMAL_ONLY
            ),
            "aclrtMalloc(coroutine task records)"
        )) {
        return EXIT_FAILURE;
    }

    const std::vector<float> input_a(
        scalar_coroutine_probe::kTileElements, 2.0F
    );
    const std::vector<float> input_b(
        scalar_coroutine_probe::kTileElements, 3.0F
    );
    const auto records = MakeTaskRecords();
    if (!CheckAcl(
            aclrtMemcpy(
                input_a_device, scalar_coroutine_probe::kTileBytes,
                input_a.data(), scalar_coroutine_probe::kTileBytes,
                ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy(H2D coroutine input A)"
        ) ||
        !CheckAcl(
            aclrtMemcpy(
                input_b_device, scalar_coroutine_probe::kTileBytes,
                input_b.data(), scalar_coroutine_probe::kTileBytes,
                ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy(H2D coroutine input B)"
        ) ||
        !CheckAcl(
            aclrtMemcpy(
                task_records_device, task_record_bytes, records.data(),
                task_record_bytes, ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy(H2D coroutine task records)"
        )) {
        return EXIT_FAILURE;
    }

    atomic_probe::pmu::RegisterMappings mappings;
    if (!mappings.Initialize(static_cast<uint32_t>(device))) {
        return EXIT_FAILURE;
    }
    const auto &mapped_array = mappings.RegisterBases();
    const std::vector<uint64_t> mapped_bases(
        mapped_array.begin(), mapped_array.end()
    );
    pa_scheduler::pmu_owner::PmuOwnerSession owner;
    const std::string dispatcher_path =
        atomic_probe::pmu::ArtifactBesideKernel(
            aic_kernel_path, "libscalar_coroutine_probe_owner_dispatcher.so"
        );
    const std::string owner_path =
        atomic_probe::pmu::ArtifactBesideKernel(
            aic_kernel_path, "libscalar_coroutine_probe_owner_aicpu.so"
        );
    if (!owner.Initialize(
            static_cast<uint32_t>(device), stream, dispatcher_path,
            owner_path, mapped_bases
        ) ||
        !owner.Configure()) {
        std::fprintf(stderr, "Cannot establish the ten-slot PMU owner.\n");
        return EXIT_FAILURE;
    }

    std::printf(
        "=== A5 scalar coroutine foundation probe ===\n"
        "device=%d repeats=%u task_records=%u aicore_hz=1.65GHz "
        "sys_counter_hz=1GHz\n"
        "contract=explicit_context_save_restore + task_age_fifo + "
        "completion_after_final_wait\n",
        device, kMeasuredRepeats, scalar_coroutine_probe::kTaskRecords
    );

    bool all_passed = true;
    const std::array<std::pair<Role, const LoadedKernel *>, 2U> roles = {{
        {Role::Aic, &aic_kernel},
        {Role::Aiv, &aiv_kernel},
    }};
    for (const auto &[role, kernel] : roles) {
        for (const uint32_t iterations : {0U, 64U}) {
            Sample warmup;
            all_passed &= RunOne(
                *kernel, role, Mode::ContextFifo, iterations,
                0x10000000U ^ iterations, stream, state_device,
                input_a_device, input_b_device, output_device,
                task_records_device, owner.RegisterTableDeviceAddress(),
                records, owner.Control(), &warmup, false
            );
            std::vector<Sample> fifo_samples;
            for (uint32_t repeat = 1U;
                 repeat <= kMeasuredRepeats && all_passed; ++repeat) {
                Sample fifo;
                all_passed &= RunOne(
                    *kernel, role, Mode::ContextFifo, iterations,
                    0x11000000U ^ iterations ^ repeat, stream,
                    state_device, input_a_device, input_b_device,
                    output_device, task_records_device,
                    owner.RegisterTableDeviceAddress(), records,
                    owner.Control(), &fifo, true
                );
                fifo_samples.push_back(fifo);
            }
            if (all_passed) {
                PrintContextSummary(role, iterations, fifo_samples);
            }
        }
        for (const uint32_t iterations : kIterationValues) {
            for (const Mode mode : {
                     Mode::EngineThenSchedule,
                     Mode::EngineOverlapSchedule,
                 }) {
                Sample warmup;
                all_passed &= RunOne(
                    *kernel, role, mode, iterations,
                    0x20000000U ^ iterations, stream, state_device,
                    input_a_device, input_b_device, output_device,
                    task_records_device, owner.RegisterTableDeviceAddress(),
                    records, owner.Control(), &warmup, false
                );
            }
            if (!all_passed) break;

            std::vector<Sample> serial_samples;
            std::vector<Sample> overlap_samples;
            for (uint32_t repeat = 1U;
                 repeat <= kMeasuredRepeats && all_passed; ++repeat) {
                const uint32_t seed =
                    0x30000000U ^ (iterations * 0x9e37U) ^ repeat;
                const std::array<Mode, 2U> order =
                    (repeat & 1U) != 0U
                    ? std::array<Mode, 2U>{
                          Mode::EngineThenSchedule,
                          Mode::EngineOverlapSchedule,
                      }
                    : std::array<Mode, 2U>{
                          Mode::EngineOverlapSchedule,
                          Mode::EngineThenSchedule,
                      };
                Sample serial;
                Sample overlap;
                for (const Mode mode : order) {
                    Sample *sample = mode == Mode::EngineThenSchedule
                        ? &serial : &overlap;
                    all_passed &= RunOne(
                        *kernel, role, mode, iterations, seed, stream,
                        state_device, input_a_device, input_b_device,
                        output_device, task_records_device,
                        owner.RegisterTableDeviceAddress(), records,
                        owner.Control(), sample, true
                    );
                }
                const bool pair_equal =
                    serial.result.schedule_checksum ==
                        overlap.result.schedule_checksum &&
                    serial.result.context0_signature ==
                        overlap.result.context0_signature &&
                    serial.result.context1_signature ==
                        overlap.result.context1_signature &&
                    serial.result.resume_sequence ==
                        overlap.result.resume_sequence;
                std::printf(
                    "[PAIR_ASSERT] role=%s iterations=%u repeat=%u "
                    "same_replay_and_context=%s\n",
                    RoleName(role), iterations, repeat,
                    pair_equal ? "PASS" : "FAIL"
                );
                all_passed &= pair_equal;
                serial_samples.push_back(serial);
                overlap_samples.push_back(overlap);
            }
            if (all_passed) {
                PrintPairSummary(
                    role, iterations, serial_samples, overlap_samples
                );
            }
        }
        // 三种公开 sync_mode 与显式 buffer-token 协议分别独立运行；一次
        // launch 只观察一种状态源。0/4096 两档用于区分“紧跟发射”与
        // “留出足够 scalar 工作后”的返回值，并给 idle 查询提供差量。
        Sample idle_cost_zero;
        Sample idle_cost_many;
        bool have_idle_cost_zero = false;
        bool have_idle_cost_many = false;
        for (const Mode mode : {
                 Mode::TryWaitCrossCore,
                 Mode::TryWaitIntraBlock,
                 Mode::TryWaitBufferId,
                 Mode::TryWaitTaggedBufferId,
                 Mode::TryWaitTaggedDynamicBufferId,
                 Mode::TryWaitPollUntilDone,
                 Mode::TryWaitFourSlots,
                 Mode::TryWaitIdleCost,
             }) {
            for (const uint32_t iterations : {0U, 4096U}) {
                Sample sample;
                all_passed &= RunOne(
                    *kernel, role, mode, iterations,
                    0x40000000U ^
                        (static_cast<uint32_t>(mode) << 20U) ^ iterations,
                    stream, state_device, input_a_device, input_b_device,
                    output_device, task_records_device,
                    owner.RegisterTableDeviceAddress(), records,
                    owner.Control(), &sample, true
                );
                if (all_passed) {
                    PrintTryWaitResult(role, mode, iterations, sample);
                }
                if (mode == Mode::TryWaitIdleCost) {
                    if (iterations == 0U) {
                        idle_cost_zero = sample;
                        have_idle_cost_zero = true;
                    } else {
                        idle_cost_many = sample;
                        have_idle_cost_many = true;
                    }
                }
            }
        }
        if (all_passed && have_idle_cost_zero && have_idle_cost_many) {
            constexpr double kIdleQueries = 4096.0;
            const double cycles_per_query =
                (static_cast<double>(
                     idle_cost_many.result.pmu_total_cycles
                 ) - static_cast<double>(
                     idle_cost_zero.result.pmu_total_cycles
                 )) / kIdleQueries;
            const double sys_ns_per_query =
                (static_cast<double>(idle_cost_many.result.sys_ticks) -
                 static_cast<double>(idle_cost_zero.result.sys_ticks)) /
                kIdleQueries;
            std::printf(
                "[TRY_WAIT_COST] role=%s idle_queries=4096 "
                "amortized_cycles=%.6f ns_at_1p65ghz=%.6f "
                "sys_ns=%.6f\n",
                RoleName(role), cycles_per_query,
                cycles_per_query / kAicoreCyclesPerNanosecond,
                sys_ns_per_query
            );
        }
        if (!all_passed) break;
    }

    const bool owner_cleanup_ok = owner.Finalize();
    mappings.Release();
    bool cleanup_ok = owner_cleanup_ok;
    cleanup_ok &= CheckAcl(
        aclrtFree(task_records_device),
        "aclrtFree(coroutine task records)"
    );
    cleanup_ok &= CheckAcl(
        aclrtFree(output_device), "aclrtFree(coroutine output)"
    );
    cleanup_ok &= CheckAcl(
        aclrtFree(input_b_device), "aclrtFree(coroutine input B)"
    );
    cleanup_ok &= CheckAcl(
        aclrtFree(input_a_device), "aclrtFree(coroutine input A)"
    );
    cleanup_ok &= CheckAcl(
        aclrtFree(state_device), "aclrtFree(coroutine state)"
    );
    cleanup_ok &= CheckAcl(
        aclrtBinaryUnLoad(aiv_kernel.binary),
        "aclrtBinaryUnLoad(coroutine AIV)"
    );
    cleanup_ok &= CheckAcl(
        aclrtBinaryUnLoad(aic_kernel.binary),
        "aclrtBinaryUnLoad(coroutine AIC)"
    );
    cleanup_ok &= CheckAcl(
        aclrtDestroyStream(stream), "aclrtDestroyStream"
    );
    cleanup_ok &= CheckAcl(
        aclrtResetDevice(device), "aclrtResetDevice"
    );
    cleanup_ok &= CheckAcl(aclFinalize(), "aclFinalize");
    std::printf(
        "[SUMMARY] semantic_status=%s pmu_restore_and_cleanup=%s\n",
        all_passed ? "PASS" : "FAIL",
        cleanup_ok ? "PASS" : "FAIL"
    );
    return all_passed && cleanup_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
