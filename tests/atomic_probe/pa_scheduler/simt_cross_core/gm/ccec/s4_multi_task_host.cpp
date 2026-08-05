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

#include "../common/s4_multi_task.h"

#include "acl/acl.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace pa_scheduler::simt_cross_core;
using namespace pa_scheduler::simt_cross_core::gm;
using namespace pa_scheduler::simt_cross_core::s4;

constexpr uint64_t kExecutorChecksumSeed = 0x6A09E667F3BCC909ULL;

struct Options {
    std::string kernel_path;
    int32_t device = 0;
    uint32_t runs = 100U;
};

struct DeviceAddresses {
    uint64_t vector_input_a;
    uint64_t vector_input_b;
    uint64_t vector_output;
    uint64_t cube_input_a;
    uint64_t cube_input_b;
    uint64_t cube_output;
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
    return input.read(data->data(), size).good();
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
        if (!CheckAcl(
                aclrtBinaryLoadFromData(binary_data.data(), binary_data.size(), &options, &binary_),
                "aclrtBinaryLoadFromData"
            ) ||
            !CheckAcl(aclrtBinaryGetFunctionByEntry(binary_, 0U, &function_), "aclrtBinaryGetFunctionByEntry") ||
            !CheckAcl(
                aclrtMalloc(&device_state_, sizeof(ProbeState), ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(ProbeState)"
            )) {
            return false;
        }
        if ((reinterpret_cast<uintptr_t>(device_state_) & (kCacheLineBytes - 1U)) != 0U) {
            std::fprintf(stderr, "device ProbeState is not 64-byte aligned: %p\n", device_state_);
            return false;
        }
        return true;
    }

    const std::string &SocName() const { return soc_name_; }

    uint64_t DeviceFieldAddress(size_t offset) const { return reinterpret_cast<uint64_t>(device_state_) + offset; }

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
        static_assert(sizeof(KernelArgs) == sizeof(uint64_t), "unexpected S4 kernel argument ABI");
        return CheckAcl(
                   aclrtLaunchKernelWithHostArgs(function_, 1U, stream_, nullptr, &args, sizeof(args), nullptr, 0U),
                   "aclrtLaunchKernelWithHostArgs(S4 mixed 1:2)"
               ) &&
               CheckAcl(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream(S4)") &&
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

void InitializeGuard(ProbeGuard *guard, uint64_t nonce, uint32_t guard_index) {
    for (uint32_t word = 0U; word < kCacheLineBytes / sizeof(uint64_t); ++word) {
        guard->words[word] = ExpectedGuardWord(nonce, guard_index, word);
    }
}

void InitializeState(ProbeState *state, uint64_t nonce) {
    *state = ProbeState{};
    state->control.magic = kProbeMagic;
    state->control.version = kProbeVersion;
    state->control.launch_nonce = nonce;
    state->control.timeout_ticks = 1000000000ULL;
    state->control.visibility_mode = kRequiredVisibilityMode;
    state->control.thread_count = kBuilderThreadCount;
    state->control.element_count = kElementCount;
    state->control.task_id = kTaskCount;

    ProbeGuard *guards[] = {
        &state->guard_before_vector_input_a, &state->guard_before_vector_input_b, &state->guard_before_vector_output,
        &state->guard_before_cube_input_a,   &state->guard_before_cube_input_b,   &state->guard_before_cube_output,
        &state->guard_after_cube_output,
    };
    for (uint32_t guard = 0U; guard < sizeof(guards) / sizeof(guards[0]); ++guard) {
        InitializeGuard(guards[guard], nonce, guard);
    }
    for (uint32_t ordinal = 0U; ordinal < kVectorTaskCount; ++ordinal) {
        for (uint32_t index = 0U; index < kElementCount; ++index) {
            state->vector_input_a[ordinal].values[index] = ExpectedVectorInputA(nonce, ordinal, index);
            state->vector_input_b[ordinal].values[index] = ExpectedVectorInputB(nonce, ordinal, index);
            state->vector_output[ordinal].values[index] = kOutputSentinel;
        }
    }
    for (uint32_t ordinal = 0U; ordinal < kCubeTaskCount; ++ordinal) {
        for (uint32_t row = 0U; row < kTileRows; ++row) {
            for (uint32_t column = 0U; column < kTileColumns; ++column) {
                const uint32_t index = row * kTileColumns + column;
                state->cube_input_a[ordinal].values[index] = ExpectedCubeInputA(ordinal, row, column);
                state->cube_input_b[ordinal].values[index] = ExpectedCubeInputB(nonce, ordinal, row, column);
                state->cube_output[ordinal].values[index] = kOutputSentinel;
            }
        }
    }
}

bool GuardsValid(const ProbeState &state, uint64_t nonce) {
    const ProbeGuard *guards[] = {
        &state.guard_before_vector_input_a, &state.guard_before_vector_input_b, &state.guard_before_vector_output,
        &state.guard_before_cube_input_a,   &state.guard_before_cube_input_b,   &state.guard_before_cube_output,
        &state.guard_after_cube_output,
    };
    for (uint32_t guard = 0U; guard < sizeof(guards) / sizeof(guards[0]); ++guard) {
        for (uint32_t word = 0U; word < kCacheLineBytes / sizeof(uint64_t); ++word) {
            if (guards[guard]->words[word] != ExpectedGuardWord(nonce, guard, word)) {
                return false;
            }
        }
    }
    return true;
}

bool DataValid(const ProbeState &state, uint64_t nonce) {
    for (uint32_t ordinal = 0U; ordinal < kVectorTaskCount; ++ordinal) {
        for (uint32_t index = 0U; index < kElementCount; ++index) {
            if (state.vector_input_a[ordinal].values[index] != ExpectedVectorInputA(nonce, ordinal, index) ||
                state.vector_input_b[ordinal].values[index] != ExpectedVectorInputB(nonce, ordinal, index) ||
                state.vector_output[ordinal].values[index] != ExpectedVectorOutput(nonce, ordinal, index)) {
                return false;
            }
        }
    }
    for (uint32_t ordinal = 0U; ordinal < kCubeTaskCount; ++ordinal) {
        for (uint32_t row = 0U; row < kTileRows; ++row) {
            for (uint32_t column = 0U; column < kTileColumns; ++column) {
                const uint32_t index = row * kTileColumns + column;
                if (state.cube_input_a[ordinal].values[index] != ExpectedCubeInputA(ordinal, row, column) ||
                    state.cube_input_b[ordinal].values[index] != ExpectedCubeInputB(nonce, ordinal, row, column) ||
                    state.cube_output[ordinal].values[index] != ExpectedCubeOutput(nonce, ordinal, row, column)) {
                    return false;
                }
            }
        }
    }
    return true;
}

uint64_t TileAddress(uint64_t base, uint32_t ordinal) { return base + static_cast<uint64_t>(ordinal) * kTileBytes; }

bool PayloadAndReportValid(
    const ProbeState &state, uint32_t task_index, uint64_t nonce, const DeviceAddresses &addresses
) {
    const uint32_t ordinal = TaskOrdinal(task_index);
    const bool vector = TaskIsVector(task_index);
    const uint64_t input_a = TileAddress(vector ? addresses.vector_input_a : addresses.cube_input_a, ordinal);
    const uint64_t input_b = TileAddress(vector ? addresses.vector_input_b : addresses.cube_input_b, ordinal);
    const uint64_t output = TileAddress(vector ? addresses.vector_output : addresses.cube_output, ordinal);
    const ProbePayload &payload = state.tasks[task_index].payload;
    const ProbeSimtReport &report = state.simt_reports[task_index];
    return state.tasks[task_index].cell.state == DoneState(task_index) &&
           payload.words[kPayloadMagicWord] == TaskPayloadMagic(task_index) &&
           payload.words[kPayloadVersionWord] == kPayloadVersion && payload.words[kPayloadNonceWord] == nonce &&
           payload.words[kPayloadInputAWord] == input_a && payload.words[kPayloadInputBWord] == input_b &&
           payload.words[kPayloadOutputWord] == output &&
           payload.words[kPayloadShapeWord] == PackTaskShape(TaskId(task_index), kElementCount) &&
           payload.words[kPayloadChecksumWord] == ComputePayloadChecksum(task_index, nonce, input_a, input_b, output) &&
           report.reserve_observed == 0U && report.publish_observed == BuildingState(task_index) &&
           report.participating_threads == kBuilderThreadCount && report.payload_words_written == kPayloadWords &&
           report.launch_nonce == nonce && report.builder_thread == BuilderThreadForTask(task_index) &&
           report.writer_dcci == 0U;
}

bool HasStatus(const ProbeRoleResult &result, uint64_t status) { return (result.status & status) != 0U; }

uint32_t IdentityByte(const ProbeRoleResult &result, uint32_t shift) {
    return IdentityField(result.identity, shift, kIdentityByteMask);
}

uint32_t CountField(const ProbeRoleResult &result, uint32_t shift) { return RoleCountField(result.role_counts, shift); }

bool RoleIdentityValid(const ProbeRoleResult &result, ProbeRole role, uint64_t nonce) {
    return result.magic == kResultMagic && IdentityByte(result, kIdentityRoleShift) == static_cast<uint32_t>(role) &&
           IdentityByte(result, kIdentityBlockIndexShift) == 0U &&
           IdentityByte(result, kIdentityBlockCountShift) == 1U && result.launch_nonce == nonce &&
           result.visibility_mode == kRequiredVisibilityMode && HasStatus(result, kResultConfigValid) &&
           result.fatal_state == 0U;
}

uint64_t ExpectedExecutorChecksum(bool vector, uint64_t nonce, const DeviceAddresses &addresses) {
    uint64_t checksum = kExecutorChecksumSeed;
    for (uint32_t task_index = vector ? 0U : 1U; task_index < kTaskCount; task_index += 2U) {
        const uint32_t ordinal = TaskOrdinal(task_index);
        const uint64_t input_a = TileAddress(vector ? addresses.vector_input_a : addresses.cube_input_a, ordinal);
        const uint64_t input_b = TileAddress(vector ? addresses.vector_input_b : addresses.cube_input_b, ordinal);
        const uint64_t output = TileAddress(vector ? addresses.vector_output : addresses.cube_output, ordinal);
        checksum = FoldDescriptorChecksum(
            checksum, ComputePayloadChecksum(task_index, nonce, input_a, input_b, output), task_index
        );
    }
    return checksum;
}

bool ExecutorValid(
    const ProbeRoleResult &result, ProbeRole role, uint64_t nonce, uint32_t last_task, uint64_t expected_checksum
) {
    return RoleIdentityValid(result, role, nonce) && CountField(result, kCountBuildAttemptShift) == 0U &&
           CountField(result, kCountBuildWinShift) == 0U &&
           CountField(result, kCountClaimAttemptShift) == kVectorTaskCount &&
           CountField(result, kCountClaimWinShift) == kVectorTaskCount &&
           result.claim_observed == BuiltState(last_task) && result.done_observed == ClaimedState(last_task) &&
           result.final_state == DoneState(last_task) && HasStatus(result, kResultPayloadValid) &&
           HasStatus(result, kResultTaskExecuted) && HasStatus(result, kResultWaitDone) &&
           result.observed_payload_checksum == expected_checksum &&
           result.expected_payload_checksum == expected_checksum &&
           result.observed_elements == static_cast<uint64_t>(kVectorTaskCount) * kElementCount &&
           ExecutorStatField(result.reserved, kExecutorExecutedShift) == kVectorTaskCount &&
           ExecutorStatField(result.reserved, kExecutorMaxBusyShift) == 1U &&
           ExecutorStatField(result.reserved, kExecutorBusyBlockedShift) == 0U;
}

bool Validate(const ProbeState &state, uint64_t nonce, const DeviceAddresses &addresses) {
    const ProbeRoleResult &aic = state.roles[static_cast<uint32_t>(ProbeRole::AicCubeExecutor)];
    const ProbeRoleResult &builder = state.roles[static_cast<uint32_t>(ProbeRole::Aiv0Builder)];
    const ProbeRoleResult &aiv = state.roles[static_cast<uint32_t>(ProbeRole::Aiv1VectorExecutor)];
    const bool builder_ok = RoleIdentityValid(builder, ProbeRole::Aiv0Builder, nonce) &&
                            IdentityByte(builder, kIdentitySubblockShift) == 0U &&
                            IdentityByte(builder, kIdentitySubblockCountShift) == 2U &&
                            CountField(builder, kCountBuildAttemptShift) == kTaskCount &&
                            CountField(builder, kCountBuildWinShift) == kTaskCount &&
                            CountField(builder, kCountClaimAttemptShift) == 0U &&
                            CountField(builder, kCountClaimWinShift) == 0U &&
                            !HasStatus(builder, kResultPayloadValid) && !HasStatus(builder, kResultTaskExecuted) &&
                            HasStatus(builder, kResultWaitDone) && builder.final_state == kTaskCount;
    const bool aiv_ok =
        IdentityByte(aiv, kIdentitySubblockShift) == 1U && IdentityByte(aiv, kIdentitySubblockCountShift) == 2U &&
        ExecutorValid(
            aiv, ProbeRole::Aiv1VectorExecutor, nonce, kTaskCount - 2U, ExpectedExecutorChecksum(true, nonce, addresses)
        );
    const bool aic_ok = ExecutorValid(
        aic, ProbeRole::AicCubeExecutor, nonce, kTaskCount - 1U, ExpectedExecutorChecksum(false, nonce, addresses)
    );
    bool tasks_ok = true;
    for (uint32_t task_index = 0U; task_index < kTaskCount; ++task_index) {
        tasks_ok = tasks_ok && PayloadAndReportValid(state, task_index, nonce, addresses);
    }
    return state.fatal.state == 0U && state.drain.builder_finished == 1U &&
           state.drain.vector_done == kVectorTaskCount && state.drain.cube_done == kCubeTaskCount &&
           state.drain.done_count == kTaskCount && tasks_ok && builder_ok && aiv_ok && aic_ok &&
           GuardsValid(state, nonce) && DataValid(state, nonce);
}

void PrintFailure(const ProbeState &state, uint32_t run, uint64_t nonce, const DeviceAddresses &addresses) {
    const ProbeRoleResult &aic = state.roles[static_cast<uint32_t>(ProbeRole::AicCubeExecutor)];
    const ProbeRoleResult &builder = state.roles[static_cast<uint32_t>(ProbeRole::Aiv0Builder)];
    const ProbeRoleResult &aiv = state.roles[static_cast<uint32_t>(ProbeRole::Aiv1VectorExecutor)];
    uint32_t first_bad_task = kTaskCount;
    for (uint32_t task = 0U; task < kTaskCount; ++task) {
        if (!PayloadAndReportValid(state, task, nonce, addresses)) {
            first_bad_task = task;
            break;
        }
    }
    std::fprintf(
        stderr,
        "[RAW-FAIL] run=%u fatal=0x%llx drain=%llu/%llu/%llu/%llu bad_task=%u "
        "builder=%u/%u aiv=%u/%u aic=%u/%u status={0x%llx,0x%llx,0x%llx}\n",
        run, static_cast<unsigned long long>(state.fatal.state),
        static_cast<unsigned long long>(state.drain.builder_finished),
        static_cast<unsigned long long>(state.drain.vector_done),
        static_cast<unsigned long long>(state.drain.cube_done), static_cast<unsigned long long>(state.drain.done_count),
        first_bad_task, CountField(builder, kCountBuildWinShift), CountField(builder, kCountBuildAttemptShift),
        CountField(aiv, kCountClaimWinShift), CountField(aiv, kCountClaimAttemptShift),
        CountField(aic, kCountClaimWinShift), CountField(aic, kCountClaimAttemptShift),
        static_cast<unsigned long long>(builder.status), static_cast<unsigned long long>(aiv.status),
        static_cast<unsigned long long>(aic.status)
    );
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
        std::fprintf(stderr, "S4 requires A5/Ascend950, but ACL reports: %s\n", session.SocName().c_str());
        return EXIT_FAILURE;
    }
    const DeviceAddresses addresses{
        session.DeviceFieldAddress(offsetof(ProbeState, vector_input_a)),
        session.DeviceFieldAddress(offsetof(ProbeState, vector_input_b)),
        session.DeviceFieldAddress(offsetof(ProbeState, vector_output)),
        session.DeviceFieldAddress(offsetof(ProbeState, cube_input_a)),
        session.DeviceFieldAddress(offsetof(ProbeState, cube_input_b)),
        session.DeviceFieldAddress(offsetof(ProbeState, cube_output)),
    };
    std::printf(
        "[DEVICE] id=%d soc=%s topology=1AIC+2AIV state_bytes=%zu builder_warps=%u builder_threads=%u tasks=%u\n",
        options.device, session.SocName().c_str(), sizeof(ProbeState), kBuilderWarpCount, kBuilderThreadCount,
        kTaskCount
    );

    auto state = std::make_unique<ProbeState>();
    uint32_t passes = 0U;
    uint32_t printed_failures = 0U;
    for (uint32_t run = 0U; run < options.runs; ++run) {
        const uint64_t nonce = 0xA554000000000000ULL ^ (static_cast<uint64_t>(run + 1U) << 8U);
        InitializeState(state.get(), nonce);
        if (!session.Run(state.get())) {
            return EXIT_FAILURE;
        }
        const bool valid = Validate(*state, nonce, addresses);
        passes += valid ? 1U : 0U;
        if (!valid && printed_failures < 8U) {
            PrintFailure(*state, run, nonce, addresses);
            ++printed_failures;
        }
    }
    const bool passed = passes == options.runs;
    std::printf("[SUMMARY] 4-warp-build+16-state+drain+8-vector+8-cube=%u/%u\n", passes, options.runs);
    std::printf(
        "[%s] S4 mixed multi-task runs=%u reused_address=yes vector_tasks=%u cube_tasks=%u busy_depth=1\n",
        passed ? "PASS" : "FAIL", options.runs, kVectorTaskCount, kCubeTaskCount
    );
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
