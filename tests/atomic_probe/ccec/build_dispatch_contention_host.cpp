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
#include "build_dispatch_contention_shared.h"

#include "runtime/rt.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace build_dispatch_probe;

constexpr uint32_t kRepeats = 11;

struct Sample {
    double device_span_us;
    double max_worker_elapsed_us;
    double begin_spread_us;
    double host_wall_us;
    uint64_t atomic_attempts;
    uint32_t active_workers;
    uint32_t min_tasks;
    uint32_t max_tasks;
};

bool CheckRt(rtError_t error, const char *label) {
    if (error == RT_ERROR_NONE) return true;
    std::fprintf(stderr, "RT error %d: %s\n", static_cast<int>(error), label);
    return false;
}

bool CheckAcl(aclError error, const char *label) {
    if (error == ACL_SUCCESS) return true;
    std::fprintf(stderr, "ACL error %d: %s\n", static_cast<int>(error), label);
    return false;
}

std::vector<char> ReadBinary(const char *path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0) return {};
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) return {};
    return data;
}

uint64_t ExpectedTaskXor() {
    uint64_t value = 0;
    for (uint32_t task = 0; task < kTasks; ++task)
        value ^= task;
    return value;
}

double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

uint64_t Median(std::vector<uint64_t> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2U];
}

const char *ModeName(Mode mode) {
    if (mode == Mode::PerTaskTournament) return "per-task-g8-cas";
    if (mode == Mode::CentralTicket) return "central-fetch-add";
    return "empty";
}

void InitializeState(ProbeState &state, Mode mode) {
    state.config.magic = kConfigMagic;
    state.config.mode = static_cast<uint32_t>(mode);
    state.config.workers = kWorkers;
    state.config.tasks = kTasks;
    state.config.groups = kTournamentGroups;
    state.config.node_stride = kTournamentNodeStride;
    state.ready_workers.value = 0;
    state.start_release.value = 0;
    state.central_ticket.value = 0;
    for (uint32_t task = 0; task < kTasks; ++task) {
        state.tournaments[task].root.owner.value = -1;
        for (uint32_t group = 0; group < kTournamentGroups; ++group) {
            state.tournaments[task].local[group].owner.value = -1;
        }
    }
}

bool ValidateTournamentState(const ProbeState &state) {
    for (uint32_t task = 0; task < kTasks; ++task) {
        if (state.tournaments[task].root.owner.value != static_cast<int64_t>(task)) return false;
        for (uint32_t group = 0; group < kTournamentGroups; ++group) {
            if (state.tournaments[task].local[group].owner.value != static_cast<int64_t>(task)) return false;
        }
    }
    return true;
}

bool RunOnce(
    void *kernel_handle, aclrtStream stream, void *state_device, void *results_device, ProbeState &state,
    std::vector<ProbeResult> &results, Mode mode, uint32_t repeat, Sample *sample
) {
    InitializeState(state, mode);
    std::fill(results.begin(), results.end(), ProbeResult{});
    if (!CheckAcl(
            aclrtMemcpy(state_device, sizeof(state), &state, sizeof(state), ACL_MEMCPY_HOST_TO_DEVICE),
            "initialize mixed dispatch probe state"
        ) ||
        !CheckAcl(
            aclrtMemset(results_device, sizeof(ProbeResult) * kWorkers, 0, sizeof(ProbeResult) * kWorkers),
            "initialize mixed dispatch probe results"
        )) {
        return false;
    }

    void *kernel_args[] = {state_device, results_device};
    rtArgsEx_t args_info{};
    args_info.args = kernel_args;
    args_info.argsSize = sizeof(kernel_args);
    rtTaskCfgInfo_t task_config{};
    const auto host_begin = std::chrono::steady_clock::now();
    if (!CheckRt(
            rtKernelLaunchWithHandleV2(kernel_handle, 0, kAicWorkers, &args_info, nullptr, stream, &task_config),
            "launch mixed dispatch contention probe"
        ) ||
        !CheckAcl(aclrtSynchronizeStream(stream), "synchronize mixed dispatch contention probe")) {
        return false;
    }
    const auto host_end = std::chrono::steady_clock::now();
    if (!CheckAcl(
            aclrtMemcpy(&state, sizeof(state), state_device, sizeof(state), ACL_MEMCPY_DEVICE_TO_HOST),
            "copy mixed dispatch probe state"
        ) ||
        !CheckAcl(
            aclrtMemcpy(
                results.data(), sizeof(ProbeResult) * kWorkers, results_device, sizeof(ProbeResult) * kWorkers,
                ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "copy mixed dispatch probe results"
        )) {
        return false;
    }

    uint64_t min_begin = std::numeric_limits<uint64_t>::max();
    uint64_t max_begin = 0;
    uint64_t max_end = 0;
    uint64_t max_worker_elapsed = 0;
    uint64_t task_sum = 0;
    uint64_t task_xor = 0;
    uint64_t attempts = 0;
    uint64_t local_wins = 0;
    uint64_t root_wins = 0;
    uint32_t valid_tasks = 0;
    uint32_t active_workers = 0;
    uint32_t min_tasks = std::numeric_limits<uint32_t>::max();
    uint32_t max_tasks = 0;
    bool ok = state.ready_workers.value == static_cast<int64_t>(kWorkers) && state.start_release.value == 1;
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        const ProbeResult &result = results[worker];
        const uint32_t expected_role = worker < kAicWorkers ? 0U : 1U;
        ok &= result.worker_id == worker && result.role == expected_role && result.errors == 0 &&
              result.completed_mode == static_cast<uint32_t>(mode) + 1U && result.begin_tick != 0 &&
              result.end_tick >= result.begin_tick;
        min_begin = std::min(min_begin, result.begin_tick);
        max_begin = std::max(max_begin, result.begin_tick);
        max_end = std::max(max_end, result.end_tick);
        max_worker_elapsed = std::max(max_worker_elapsed, result.end_tick - result.begin_tick);
        task_sum += result.task_id_sum;
        task_xor ^= result.task_id_xor;
        attempts += result.atomic_attempts;
        local_wins += result.local_wins;
        root_wins += result.root_wins;
        valid_tasks += result.valid_tasks;
        active_workers += result.valid_tasks != 0 ? 1U : 0U;
        min_tasks = std::min(min_tasks, result.valid_tasks);
        max_tasks = std::max(max_tasks, result.valid_tasks);
    }

    const uint64_t expected_sum = static_cast<uint64_t>(kTasks) * (kTasks - 1U) / 2U;
    if (mode == Mode::PerTaskTournament) {
        ok &= valid_tasks == kTasks && task_sum == expected_sum && task_xor == ExpectedTaskXor() &&
              attempts == static_cast<uint64_t>(kTasks) * (kWorkers + kTournamentGroups) &&
              local_wins == static_cast<uint64_t>(kTasks) * kTournamentGroups && root_wins == kTasks &&
              state.central_ticket.value == 0 && ValidateTournamentState(state);
    } else if (mode == Mode::CentralTicket) {
        ok &= valid_tasks == kTasks && task_sum == expected_sum && task_xor == ExpectedTaskXor() &&
              attempts == static_cast<uint64_t>(kTasks) + kWorkers && local_wins == 0 && root_wins == 0 &&
              state.central_ticket.value == static_cast<int64_t>(kTasks + kWorkers);
    } else {
        ok &= valid_tasks == 0 && task_sum == 0 && task_xor == 0 && attempts == 0 && local_wins == 0 &&
              root_wins == 0 && state.central_ticket.value == 0;
    }

    sample->device_span_us = static_cast<double>(max_end - min_begin) / 1000.0;
    sample->max_worker_elapsed_us = static_cast<double>(max_worker_elapsed) / 1000.0;
    sample->begin_spread_us = static_cast<double>(max_begin - min_begin) / 1000.0;
    sample->host_wall_us = std::chrono::duration<double, std::micro>(host_end - host_begin).count();
    sample->atomic_attempts = attempts;
    sample->active_workers = active_workers;
    sample->min_tasks = min_tasks;
    sample->max_tasks = max_tasks;
    std::printf(
        "[DISPATCH_CONTENTION] repeat=%u mode=%s status=%s span=%.3f us "
        "max_worker=%.3f us begin_spread=%.3f us host=%.3f us atomics=%llu "
        "owners=%u task_range=%u..%u\n",
        repeat, ModeName(mode), ok ? "PASS" : "FAIL", sample->device_span_us, sample->max_worker_elapsed_us,
        sample->begin_spread_us, sample->host_wall_us, static_cast<unsigned long long>(attempts), active_workers,
        min_tasks, max_tasks
    );
    return ok;
}

}  // namespace

int main(int argc, char **argv) {
    const char *kernel_path = argc > 1 ? argv[1] : "./build_dispatch_contention_kernel.o";
    const std::vector<char> binary_data = ReadBinary(kernel_path);
    if (binary_data.empty()) {
        std::fprintf(stderr, "Cannot read mixed kernel binary: %s\n", kernel_path);
        return EXIT_FAILURE;
    }

    const int32_t device = atomic_probe::DeviceId();
    if (device < 0 || !CheckAcl(aclInit(nullptr), "aclInit") || !CheckAcl(aclrtSetDevice(device), "aclrtSetDevice")) {
        return EXIT_FAILURE;
    }
    aclrtStream stream = nullptr;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) return EXIT_FAILURE;

    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, binary_data.data(), binary_data.size()};
    void *kernel_handle = nullptr;
    bool registered_all = true;
    rtError_t register_error = rtRegisterAllKernel(&binary, &kernel_handle);
    if (register_error != RT_ERROR_NONE || kernel_handle == nullptr) {
        registered_all = false;
        register_error = rtBinaryLoadWithoutTilingKey(binary_data.data(), binary_data.size(), &kernel_handle);
    }
    if (!CheckRt(register_error, "register mixed dispatch contention ELF") || kernel_handle == nullptr) {
        return EXIT_FAILURE;
    }

    void *state_device = nullptr;
    void *results_device = nullptr;
    if (!CheckAcl(aclrtMalloc(&state_device, sizeof(ProbeState), ACL_MEM_MALLOC_HUGE_FIRST), "allocate probe state") ||
        !CheckAcl(
            aclrtMalloc(&results_device, sizeof(ProbeResult) * kWorkers, ACL_MEM_MALLOC_HUGE_FIRST),
            "allocate probe results"
        )) {
        return EXIT_FAILURE;
    }
    auto state = std::make_unique<ProbeState>();
    std::vector<ProbeResult> results(kWorkers);
    std::vector<double> tournament_device;
    std::vector<double> ticket_device;
    std::vector<double> empty_device;
    std::vector<double> tournament_worker;
    std::vector<double> ticket_worker;
    std::vector<double> empty_worker;
    std::vector<uint64_t> tournament_attempts;
    std::vector<uint64_t> ticket_attempts;
    bool ok = true;
    for (uint32_t repeat = 1; repeat <= kRepeats; ++repeat) {
        // 三种模式轮换首位，避免把固定的先跑/后跑温度漂移归给某个候选。
        const Mode order[3][3] = {
            {Mode::Empty, Mode::PerTaskTournament, Mode::CentralTicket},
            {Mode::PerTaskTournament, Mode::CentralTicket, Mode::Empty},
            {Mode::CentralTicket, Mode::Empty, Mode::PerTaskTournament},
        };
        for (Mode mode : order[(repeat - 1U) % 3U]) {
            Sample sample{};
            ok &= RunOnce(kernel_handle, stream, state_device, results_device, *state, results, mode, repeat, &sample);
            if (mode == Mode::PerTaskTournament) {
                tournament_device.push_back(sample.device_span_us);
                tournament_worker.push_back(sample.max_worker_elapsed_us);
                tournament_attempts.push_back(sample.atomic_attempts);
            } else if (mode == Mode::CentralTicket) {
                ticket_device.push_back(sample.device_span_us);
                ticket_worker.push_back(sample.max_worker_elapsed_us);
                ticket_attempts.push_back(sample.atomic_attempts);
            } else {
                empty_device.push_back(sample.device_span_us);
                empty_worker.push_back(sample.max_worker_elapsed_us);
            }
        }
    }

    const double tournament_median = Median(tournament_device);
    const double ticket_median = Median(ticket_device);
    const double empty_median = Median(empty_device);
    const double change = 100.0 * (ticket_median - tournament_median) / tournament_median;
    std::printf(
        "[DISPATCH_CONTENTION] summary status=%s repeats=%u workers=%u tasks=%u "
        "empty/g8/ticket_span=%.3f/%.3f/%.3f us "
        "empty/g8/ticket_max_worker=%.3f/%.3f/%.3f us ticket_change=%+.3f%% "
        "g8_atomics=%llu ticket_atomics=%llu\n",
        ok ? "PASS" : "FAIL", kRepeats, kWorkers, kTasks, empty_median, tournament_median, ticket_median,
        Median(empty_worker), Median(tournament_worker), Median(ticket_worker), change,
        static_cast<unsigned long long>(Median(tournament_attempts)),
        static_cast<unsigned long long>(Median(ticket_attempts))
    );

    ok &= CheckAcl(aclrtFree(results_device), "free probe results");
    ok &= CheckAcl(aclrtFree(state_device), "free probe state");
    ok &= CheckRt(
        registered_all ? rtDevBinaryUnRegister(kernel_handle) : rtBinaryUnLoad(kernel_handle),
        "unload mixed dispatch contention ELF"
    );
    ok &= CheckAcl(aclrtDestroyStream(stream), "destroy stream");
    ok &= CheckAcl(aclrtResetDevice(device), "reset device");
    ok &= CheckAcl(aclFinalize(), "aclFinalize");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
