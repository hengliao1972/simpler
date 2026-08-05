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

#include "../common/atomic_probe.h"

#include "acl/acl.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace pa_scheduler::simt_cross_core::simt_atomic;

struct Options {
    std::string kernel_path;
    int32_t device = 0;
    uint32_t runs = 100U;
};

bool ParseUnsigned(const char *raw, uint64_t maximum, uint64_t *value) {
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || parsed > maximum) {
        return false;
    }
    *value = static_cast<uint64_t>(parsed);
    return true;
}

bool ParseOptions(int argc, char **argv, Options *options) {
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--kernel") == 0 && index + 1 < argc) {
            options->kernel_path = argv[++index];
        } else if (std::strcmp(argv[index], "--device") == 0 && index + 1 < argc) {
            uint64_t value = 0U;
            if (!ParseUnsigned(argv[++index], static_cast<uint64_t>(std::numeric_limits<int32_t>::max()), &value)) {
                std::fprintf(stderr, "invalid --device value: %s\n", argv[index]);
                return false;
            }
            options->device = static_cast<int32_t>(value);
        } else if (std::strcmp(argv[index], "--runs") == 0 && index + 1 < argc) {
            uint64_t value = 0U;
            if (!ParseUnsigned(argv[++index], 100000U, &value) || value == 0U) {
                std::fprintf(stderr, "invalid --runs value: %s\n", argv[index]);
                return false;
            }
            options->runs = static_cast<uint32_t>(value);
        } else {
            std::fprintf(stderr, "usage: %s --kernel FILE [--device N] [--runs N]\n", argv[0]);
            return false;
        }
    }
    if (options->kernel_path.empty()) {
        std::fprintf(stderr, "--kernel FILE is required\n");
        return false;
    }
    return true;
}

bool CheckAcl(aclError error, const char *operation) {
    if (error == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "ACL error %d: %s\n", static_cast<int>(error), operation);
    return false;
}

bool ReadBinary(const std::string &path, std::vector<char> *data) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        std::fprintf(stderr, "cannot open kernel binary: %s\n", path.c_str());
        return false;
    }
    const std::streamsize size = input.tellg();
    if (size <= 0) {
        std::fprintf(stderr, "kernel binary is empty: %s\n", path.c_str());
        return false;
    }
    input.seekg(0, std::ios::beg);
    data->resize(static_cast<size_t>(size));
    if (!input.read(data->data(), size)) {
        std::fprintf(stderr, "cannot read kernel binary: %s\n", path.c_str());
        return false;
    }
    return true;
}

class AclSession {
public:
    ~AclSession() {
        if (device_state_ != nullptr) {
            (void)aclrtFree(device_state_);
        }
        if (binary_ != nullptr) {
            (void)aclrtBinaryUnLoad(binary_);
        }
        if (stream_ != nullptr) {
            (void)aclrtDestroyStream(stream_);
        }
        if (device_set_) {
            (void)aclrtResetDevice(device_);
        }
        if (acl_initialized_) {
            (void)aclFinalize();
        }
    }

    bool Initialize(int32_t device, const std::vector<char> &binary_data) {
        device_ = device;
        if (!CheckAcl(aclInit(nullptr), "aclInit")) {
            return false;
        }
        acl_initialized_ = true;
        if (!CheckAcl(aclrtSetDevice(device_), "aclrtSetDevice")) {
            return false;
        }
        device_set_ = true;
        const char *soc_name = aclrtGetSocName();
        if (soc_name == nullptr || soc_name[0] == '\0') {
            std::fprintf(stderr, "aclrtGetSocName returned an empty SoC name\n");
            return false;
        }
        soc_name_ = soc_name;
        if (!CheckAcl(aclrtCreateStream(&stream_), "aclrtCreateStream")) {
            return false;
        }

        aclrtBinaryLoadOption option{};
        option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
        option.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
        aclrtBinaryLoadOptions options{&option, 1U};
        return CheckAcl(
                   aclrtBinaryLoadFromData(binary_data.data(), binary_data.size(), &options, &binary_),
                   "aclrtBinaryLoadFromData"
               ) &&
               CheckAcl(aclrtBinaryGetFunctionByEntry(binary_, 0U, &function_), "aclrtBinaryGetFunctionByEntry") &&
               CheckAcl(
                   aclrtMalloc(&device_state_, sizeof(ProbeState), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(ProbeState)"
               );
    }

    const std::string &SocName() const { return soc_name_; }

    bool Run(ProbeState *state) {
        if (!CheckAcl(
                aclrtMemcpy(device_state_, sizeof(*state), state, sizeof(*state), ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy(H2D ProbeState)"
            )) {
            return false;
        }
        struct KernelArgs {
            uint64_t state_pointer;
        } args{reinterpret_cast<uint64_t>(device_state_)};
        static_assert(sizeof(KernelArgs) == sizeof(uint64_t), "unexpected CCEC kernel argument ABI");
        return CheckAcl(
                   aclrtLaunchKernelWithHostArgs(function_, 1U, stream_, nullptr, &args, sizeof(args), nullptr, 0U),
                   "aclrtLaunchKernelWithHostArgs(SIMT atomic)"
               ) &&
               CheckAcl(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream(SIMT atomic)") &&
               CheckAcl(
                   aclrtMemcpy(state, sizeof(*state), device_state_, sizeof(*state), ACL_MEMCPY_DEVICE_TO_HOST),
                   "aclrtMemcpy(D2H ProbeState)"
               );
    }

private:
    bool acl_initialized_ = false;
    bool device_set_ = false;
    int32_t device_ = 0;
    std::string soc_name_;
    aclrtStream stream_ = nullptr;
    aclrtBinHandle binary_ = nullptr;
    aclrtFuncHandle function_ = nullptr;
    void *device_state_ = nullptr;
};

uint64_t ExpectedGuardWord(uint64_t nonce, uint32_t guard, uint32_t word) {
    return 0xD15EA5E000000000ULL ^ RotateLeft(nonce, (guard + word) % 31U + 1U) ^
           (static_cast<uint64_t>(guard) << 16U) ^ word;
}

void InitializeGuard(ProbeGuard *guard, uint64_t nonce, uint32_t guard_index) {
    for (uint32_t word = 0U; word < 8U; ++word) {
        guard->words[word] = ExpectedGuardWord(nonce, guard_index, word);
    }
}

bool GuardValid(const ProbeGuard &guard, uint64_t nonce, uint32_t guard_index) {
    for (uint32_t word = 0U; word < 8U; ++word) {
        if (guard.words[word] != ExpectedGuardWord(nonce, guard_index, word)) {
            return false;
        }
    }
    return true;
}

void InitializeState(ProbeState *state, uint64_t nonce, uint32_t thread_count) {
    *state = ProbeState{};
    state->control.magic = kProbeMagic;
    state->control.version = kProbeVersion;
    state->control.thread_count = thread_count;
    state->control.launch_nonce = nonce;
    state->control.cas_initial = ExpectedCasInitial(nonce);
    state->control.add_initial = ExpectedAddInitial(nonce);
    state->cas_cell.value = state->control.cas_initial;
    state->add_cell.value = state->control.add_initial;
    for (uint32_t thread = 0U; thread < kMaxThreadCount; ++thread) {
        state->cas_returns.values[thread] = kOutputSentinel;
        state->add_returns.values[thread] = kOutputSentinel;
        state->markers.values[thread] = kOutputSentinel;
    }
    InitializeGuard(&state->guard_before_cas, nonce, 0U);
    InitializeGuard(&state->guard_before_add, nonce, 1U);
    InitializeGuard(&state->guard_before_cas_returns, nonce, 2U);
    InitializeGuard(&state->guard_before_add_returns, nonce, 3U);
    InitializeGuard(&state->guard_before_markers, nonce, 4U);
    InitializeGuard(&state->guard_after_markers, nonce, 5U);
}

struct Validation {
    bool passed = false;
    std::string reason;
};

Validation Validate(const ProbeState &state, uint64_t nonce, uint32_t thread_count) {
    const ProbeResult &result = state.result;
    const uint64_t cas_initial = ExpectedCasInitial(nonce);
    const uint64_t add_initial = ExpectedAddInitial(nonce);
    if (state.control.magic != kProbeMagic || state.control.version != kProbeVersion ||
        state.control.thread_count != thread_count || state.control.launch_nonce != nonce ||
        state.control.cas_initial != cas_initial || state.control.add_initial != add_initial ||
        result.magic != kResultMagic || result.status != kExpectedStatus || result.launch_nonce != nonce ||
        result.requested_thread_count != thread_count || result.observed_thread_count != thread_count ||
        result.warp_size != kWarpSize || result.cas_initial != cas_initial || result.add_initial != add_initial) {
        return {false, "control/result status"};
    }
    if (!GuardValid(state.guard_before_cas, nonce, 0U) || !GuardValid(state.guard_before_add, nonce, 1U) ||
        !GuardValid(state.guard_before_cas_returns, nonce, 2U) ||
        !GuardValid(state.guard_before_add_returns, nonce, 3U) || !GuardValid(state.guard_before_markers, nonce, 4U) ||
        !GuardValid(state.guard_after_markers, nonce, 5U)) {
        return {false, "guard corruption"};
    }

    uint32_t winner_count = 0U;
    uint32_t loser_count = 0U;
    uint64_t add_sum = 0U;
    std::vector<uint64_t> tickets;
    tickets.reserve(thread_count);
    for (uint32_t thread = 0U; thread < thread_count; ++thread) {
        const uint64_t cas_observed = state.cas_returns.values[thread];
        if (cas_observed == cas_initial) {
            ++winner_count;
            if (state.cas_cell.value != ExpectedCasDesired(nonce, thread)) {
                return {false, "CAS winner/final mismatch"};
            }
        } else if (cas_observed == state.cas_cell.value) {
            ++loser_count;
        } else {
            return {false, "CAS old-value mismatch"};
        }
        if (state.markers.values[thread] != ExpectedThreadMarker(nonce, thread)) {
            return {false, "thread marker mismatch"};
        }
        tickets.push_back(state.add_returns.values[thread]);
        add_sum += state.add_returns.values[thread];
    }
    for (uint32_t thread = thread_count; thread < kMaxThreadCount; ++thread) {
        if (state.cas_returns.values[thread] != kOutputSentinel ||
            state.add_returns.values[thread] != kOutputSentinel || state.markers.values[thread] != kOutputSentinel) {
            return {false, "inactive thread wrote output"};
        }
    }
    if (winner_count != 1U || loser_count != thread_count - 1U || result.cas_winner_count != winner_count ||
        result.cas_loser_count != loser_count || result.cas_final != state.cas_cell.value) {
        return {false, "CAS contention count"};
    }

    std::sort(tickets.begin(), tickets.end());
    for (uint32_t ticket = 0U; ticket < thread_count; ++ticket) {
        if (tickets[ticket] != add_initial + ticket) {
            return {false, "atomic-add ticket permutation"};
        }
    }
    if (state.add_cell.value != add_initial + thread_count || result.add_final != state.add_cell.value ||
        result.add_ticket_in_range != thread_count || result.add_ticket_sum != add_sum) {
        return {false, "atomic-add final/aggregate"};
    }
    return {true, {}};
}

}  // namespace

int main(int argc, char **argv) {
    Options options{};
    if (!ParseOptions(argc, argv, &options)) {
        return EXIT_FAILURE;
    }
    std::vector<char> binary_data;
    if (!ReadBinary(options.kernel_path, &binary_data)) {
        return EXIT_FAILURE;
    }
    AclSession session;
    if (!session.Initialize(options.device, binary_data)) {
        return EXIT_FAILURE;
    }
    if (session.SocName().rfind("Ascend950", 0U) != 0U) {
        std::fprintf(
            stderr, "SIMT atomic probe requires A5/Ascend950, but ACL reports: %s\n", session.SocName().c_str()
        );
        return EXIT_FAILURE;
    }
    std::printf(
        "[DEVICE] id=%d soc=%s state_bytes=%zu max_threads=%u warp_size=%u\n", options.device,
        session.SocName().c_str(), sizeof(ProbeState), kMaxThreadCount, kWarpSize
    );

    bool passed = true;
    uint32_t printed_failures = 0U;
    for (uint32_t config = 0U; config < kThreadConfigCount; ++config) {
        const uint32_t thread_count = kThreadConfigs[config];
        uint32_t passes = 0U;
        for (uint32_t run = 0U; run < options.runs; ++run) {
            const uint64_t nonce = 0xA500000000000000ULL ^ (static_cast<uint64_t>(thread_count) << 32U) ^
                                   (static_cast<uint64_t>(run + 1U) << 8U);
            ProbeState state{};
            InitializeState(&state, nonce, thread_count);
            if (!session.Run(&state)) {
                return EXIT_FAILURE;
            }
            const Validation validation = Validate(state, nonce, thread_count);
            passes += validation.passed ? 1U : 0U;
            if (!validation.passed && printed_failures < 8U) {
                std::fprintf(
                    stderr,
                    "[RAW-FAIL] threads=%u run=%u reason=%s status=0x%llx cas_final=0x%llx "
                    "cas_winners=%llu add_final=0x%llx observed_threads=%llu\n",
                    thread_count, run, validation.reason.c_str(), static_cast<unsigned long long>(state.result.status),
                    static_cast<unsigned long long>(state.result.cas_final),
                    static_cast<unsigned long long>(state.result.cas_winner_count),
                    static_cast<unsigned long long>(state.result.add_final),
                    static_cast<unsigned long long>(state.result.observed_thread_count)
                );
                ++printed_failures;
            }
        }
        std::printf(
            "[SUMMARY] threads=%4u warps=%2u CAS+add+returns+guards=%u/%u\n", thread_count, thread_count / kWarpSize,
            passes, options.runs
        );
        passed = passed && passes == options.runs;
    }
    std::printf(
        "[%s] A5 SIMT GM uint64 atomic runs=%u configs=32/64/1024/2048 reused_address=yes\n", passed ? "PASS" : "FAIL",
        options.runs
    );
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
