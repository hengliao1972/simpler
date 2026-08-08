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

#include "../common/s1_vector.h"

#include "acl/acl.h"

#include <array>
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
using namespace pa_scheduler::simt_cross_core::s1;

struct Options {
    std::string kernel_path;
    int32_t device = 0;
    uint32_t runs = 100U;
};

const char *ModeName(VisibilityMode mode) {
    switch (mode) {
    case VisibilityMode::NoDcci:
        return "NO_DCCI";
    case VisibilityMode::WriterDcci:
        return "WRITER_DCCI";
    case VisibilityMode::ReaderDcci:
        return "READER_DCCI";
    case VisibilityMode::WriterAndReaderDcci:
        return "WRITER_AND_READER_DCCI";
    }
    return "UNKNOWN";
}

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
        static_assert(sizeof(KernelArgs) == sizeof(uint64_t), "unexpected S1 kernel argument ABI");
        if (!CheckAcl(
                aclrtLaunchKernelWithHostArgs(function_, 1U, stream_, nullptr, &args, sizeof(args), nullptr, 0U),
                "aclrtLaunchKernelWithHostArgs(S1 mixed 1:2)"
            ) ||
            !CheckAcl(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream(S1)") ||
            !CheckAcl(
                aclrtMemcpy(state, sizeof(*state), device_state_, sizeof(*state), ACL_MEMCPY_DEVICE_TO_HOST),
                "aclrtMemcpy(D2H ProbeState)"
            )) {
            return false;
        }
        return true;
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

void InitializeState(ProbeState *state, uint64_t nonce, VisibilityMode mode) {
    *state = ProbeState{};
    state->control.magic = kProbeMagic;
    state->control.version = kProbeVersion;
    state->control.launch_nonce = nonce;
    state->control.timeout_ticks = 1000000000ULL;
    state->control.visibility_mode = static_cast<uint64_t>(mode);
    state->control.thread_count = kThreadCount;
    state->control.element_count = kElementCount;
    state->control.task_id = kTaskId;
    state->cell.state = kEmptyState;

    InitializeGuard(&state->guard_before_input_a, nonce, 0U);
    InitializeGuard(&state->guard_before_input_b, nonce, 1U);
    InitializeGuard(&state->guard_before_output, nonce, 2U);
    InitializeGuard(&state->guard_after_output, nonce, 3U);
    for (uint32_t index = 0U; index < kElementCount; ++index) {
        state->input_a.values[index] = ExpectedInputA(nonce, index);
        state->input_b.values[index] = ExpectedInputB(nonce, index);
        state->output.values[index] = kOutputSentinel;
    }
}

bool GuardValid(const ProbeGuard &guard, uint64_t nonce, uint32_t guard_index) {
    for (uint32_t word = 0U; word < kCacheLineBytes / sizeof(uint64_t); ++word) {
        if (guard.words[word] != ExpectedGuardWord(nonce, guard_index, word)) {
            return false;
        }
    }
    return true;
}

bool InputsValid(const ProbeState &state, uint64_t nonce) {
    for (uint32_t index = 0U; index < kElementCount; ++index) {
        if (state.input_a.values[index] != ExpectedInputA(nonce, index) ||
            state.input_b.values[index] != ExpectedInputB(nonce, index)) {
            return false;
        }
    }
    return true;
}

bool OutputMatchesGolden(const ProbeState &state, uint64_t nonce) {
    for (uint32_t index = 0U; index < kElementCount; ++index) {
        if (state.output.values[index] != ExpectedOutput(nonce, index)) {
            return false;
        }
    }
    return true;
}

bool OutputIsSentinel(const ProbeState &state) {
    for (float value : state.output.values) {
        if (value != kOutputSentinel) {
            return false;
        }
    }
    return true;
}

bool HasStatus(const ProbeRoleResult &result, uint64_t status) { return (result.status & status) != 0U; }

uint32_t IdentityByte(const ProbeRoleResult &result, uint32_t shift) {
    return IdentityField(result.identity, shift, kIdentityByteMask);
}

uint32_t CountField(const ProbeRoleResult &result, uint32_t shift) { return RoleCountField(result.role_counts, shift); }

bool RoleCommonValid(const ProbeRoleResult &result, ProbeRole role, uint64_t nonce, VisibilityMode mode) {
    return result.magic == kResultMagic && IdentityByte(result, kIdentityRoleShift) == static_cast<uint32_t>(role) &&
           IdentityByte(result, kIdentityBlockIndexShift) == 0U &&
           IdentityByte(result, kIdentityBlockCountShift) == 1U && HasStatus(result, kResultConfigValid) &&
           result.launch_nonce == nonce && result.visibility_mode == static_cast<uint64_t>(mode) &&
           result.final_state == kDoneState && result.fatal_state == 0U;
}

struct Validation {
    bool control_ok = false;
    bool payload_ok = false;
    const char *reason = "ok";
};

Validation Validate(
    const ProbeState &state, VisibilityMode mode, uint64_t nonce, uint64_t expected_input_a, uint64_t expected_input_b,
    uint64_t expected_output
) {
    Validation validation{};
    const ProbeRoleResult &aic = state.roles[static_cast<uint32_t>(ProbeRole::AicObserver)];
    const ProbeRoleResult &builder = state.roles[static_cast<uint32_t>(ProbeRole::Aiv0Builder)];
    const ProbeRoleResult &executor = state.roles[static_cast<uint32_t>(ProbeRole::Aiv1Executor)];

    const bool guards_ok =
        GuardValid(state.guard_before_input_a, nonce, 0U) && GuardValid(state.guard_before_input_b, nonce, 1U) &&
        GuardValid(state.guard_before_output, nonce, 2U) && GuardValid(state.guard_after_output, nonce, 3U);
    const uint64_t expected_checksum =
        ComputePayloadChecksum(nonce, expected_input_a, expected_input_b, expected_output);
    const bool payload_bytes_ok = state.payload.words[kPayloadMagicWord] == kPayloadMagic &&
                                  state.payload.words[kPayloadVersionWord] == kPayloadVersion &&
                                  state.payload.words[kPayloadNonceWord] == nonce &&
                                  state.payload.words[kPayloadInputAWord] == expected_input_a &&
                                  state.payload.words[kPayloadInputBWord] == expected_input_b &&
                                  state.payload.words[kPayloadOutputWord] == expected_output &&
                                  state.payload.words[kPayloadShapeWord] == PackTaskShape(kTaskId, kElementCount) &&
                                  state.payload.words[kPayloadChecksumWord] == expected_checksum;
    const bool simt_ok =
        state.simt_report.reserve_observed == kEmptyState && state.simt_report.publish_observed == kBuildingState &&
        state.simt_report.participating_threads == kThreadCount &&
        state.simt_report.payload_words_written == kPayloadWords && state.simt_report.launch_nonce == nonce &&
        state.simt_report.builder_thread == 0U && state.simt_report.writer_dcci == (WriterUsesDcci(mode) ? 1U : 0U);
    const bool aic_ok = RoleCommonValid(aic, ProbeRole::AicObserver, nonce, mode) && aic.role_counts == 0U &&
                        !HasStatus(aic, kResultVectorExecuted) && HasStatus(aic, kResultWaitDone);
    const bool builder_ok =
        RoleCommonValid(builder, ProbeRole::Aiv0Builder, nonce, mode) &&
        IdentityByte(builder, kIdentitySubblockShift) == 0U &&
        IdentityByte(builder, kIdentitySubblockCountShift) == 2U &&
        CountField(builder, kCountBuildAttemptShift) == 1U && CountField(builder, kCountBuildWinShift) == 1U &&
        CountField(builder, kCountClaimAttemptShift) == 0U && CountField(builder, kCountClaimWinShift) == 0U &&
        !HasStatus(builder, kResultVectorExecuted) && HasStatus(builder, kResultWaitDone) &&
        builder.reserve_observed == kEmptyState && builder.publish_observed == kBuildingState;
    const bool executor_ok =
        RoleCommonValid(executor, ProbeRole::Aiv1Executor, nonce, mode) &&
        IdentityByte(executor, kIdentitySubblockShift) == 1U &&
        IdentityByte(executor, kIdentitySubblockCountShift) == 2U &&
        CountField(executor, kCountBuildAttemptShift) == 0U && CountField(executor, kCountBuildWinShift) == 0U &&
        CountField(executor, kCountClaimAttemptShift) == 1U && CountField(executor, kCountClaimWinShift) == 1U &&
        executor.claim_observed == kBuiltState && executor.done_observed == kClaimedState &&
        HasStatus(executor, kResultWaitDone);

    validation.control_ok = state.cell.state == kDoneState && state.fatal.state == 0U && guards_ok &&
                            InputsValid(state, nonce) && payload_bytes_ok && simt_ok && aic_ok && builder_ok &&
                            executor_ok;
    if (!validation.control_ok) {
        validation.reason = "protocol-role-guard";
        return validation;
    }

    const bool executor_payload_ok =
        HasStatus(executor, kResultPayloadValid) && HasStatus(executor, kResultVectorExecuted) &&
        executor.observed_payload_checksum == expected_checksum &&
        executor.expected_payload_checksum == expected_checksum && executor.observed_elements == kElementCount;
    validation.payload_ok = executor_payload_ok && OutputMatchesGolden(state, nonce);
    if (!validation.payload_ok) {
        validation.reason = OutputIsSentinel(state) && !HasStatus(executor, kResultVectorExecuted) ?
                                "payload-visibility" :
                                "vector-golden";
    }
    return validation;
}

void PrintRoleDiagnostic(const char *name, const ProbeRoleResult &result) {
    std::fprintf(
        stderr,
        "[ROLE] %s magic=0x%llx role=%llu block=%llu/%llu core=%llu subblock=%llu/%llu "
        "config=%llu build=%llu/%llu claim=%llu/%llu vector=%llu payload=%llu wait=%llu "
        "nonce=0x%llx mode=%llu reserve=0x%llx publish=0x%llx claim_old=0x%llx "
        "done_old=0x%llx final=0x%llx fatal=0x%llx elements=%llu\n",
        name, static_cast<unsigned long long>(result.magic),
        static_cast<unsigned long long>(IdentityByte(result, kIdentityRoleShift)),
        static_cast<unsigned long long>(IdentityByte(result, kIdentityBlockIndexShift)),
        static_cast<unsigned long long>(IdentityByte(result, kIdentityBlockCountShift)),
        static_cast<unsigned long long>(IdentityField(result.identity, kIdentityCoreIdShift, kIdentityCoreMask)),
        static_cast<unsigned long long>(IdentityByte(result, kIdentitySubblockShift)),
        static_cast<unsigned long long>(IdentityByte(result, kIdentitySubblockCountShift)),
        static_cast<unsigned long long>(HasStatus(result, kResultConfigValid)),
        static_cast<unsigned long long>(CountField(result, kCountBuildWinShift)),
        static_cast<unsigned long long>(CountField(result, kCountBuildAttemptShift)),
        static_cast<unsigned long long>(CountField(result, kCountClaimWinShift)),
        static_cast<unsigned long long>(CountField(result, kCountClaimAttemptShift)),
        static_cast<unsigned long long>(HasStatus(result, kResultVectorExecuted)),
        static_cast<unsigned long long>(HasStatus(result, kResultPayloadValid)),
        static_cast<unsigned long long>(HasStatus(result, kResultWaitDone)),
        static_cast<unsigned long long>(result.launch_nonce), static_cast<unsigned long long>(result.visibility_mode),
        static_cast<unsigned long long>(result.reserve_observed),
        static_cast<unsigned long long>(result.publish_observed),
        static_cast<unsigned long long>(result.claim_observed), static_cast<unsigned long long>(result.done_observed),
        static_cast<unsigned long long>(result.final_state), static_cast<unsigned long long>(result.fatal_state),
        static_cast<unsigned long long>(result.observed_elements)
    );
}

void PrintProtocolDiagnostic(
    const ProbeState &state, uint64_t nonce, uint64_t expected_input_a, uint64_t expected_input_b,
    uint64_t expected_output
) {
    std::fprintf(
        stderr,
        "[SIMT] reserve=0x%llx publish=0x%llx threads=%llu words=%llu nonce=0x%llx thread=%llu writer_dcci=%llu\n",
        static_cast<unsigned long long>(state.simt_report.reserve_observed),
        static_cast<unsigned long long>(state.simt_report.publish_observed),
        static_cast<unsigned long long>(state.simt_report.participating_threads),
        static_cast<unsigned long long>(state.simt_report.payload_words_written),
        static_cast<unsigned long long>(state.simt_report.launch_nonce),
        static_cast<unsigned long long>(state.simt_report.builder_thread),
        static_cast<unsigned long long>(state.simt_report.writer_dcci)
    );
    std::fprintf(
        stderr,
        "[HOST-CHECK] guards=%u inputs=%u payload_addr={0x%llx,0x%llx,0x%llx} expected={0x%llx,0x%llx,0x%llx}\n",
        GuardValid(state.guard_before_input_a, nonce, 0U) && GuardValid(state.guard_before_input_b, nonce, 1U) &&
            GuardValid(state.guard_before_output, nonce, 2U) && GuardValid(state.guard_after_output, nonce, 3U),
        InputsValid(state, nonce), static_cast<unsigned long long>(state.payload.words[kPayloadInputAWord]),
        static_cast<unsigned long long>(state.payload.words[kPayloadInputBWord]),
        static_cast<unsigned long long>(state.payload.words[kPayloadOutputWord]),
        static_cast<unsigned long long>(expected_input_a), static_cast<unsigned long long>(expected_input_b),
        static_cast<unsigned long long>(expected_output)
    );
    PrintRoleDiagnostic("AIC", state.roles[static_cast<uint32_t>(ProbeRole::AicObserver)]);
    PrintRoleDiagnostic("AIV0", state.roles[static_cast<uint32_t>(ProbeRole::Aiv0Builder)]);
    PrintRoleDiagnostic("AIV1", state.roles[static_cast<uint32_t>(ProbeRole::Aiv1Executor)]);
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
        std::fprintf(stderr, "S1 requires A5/Ascend950, but ACL reports: %s\n", session.SocName().c_str());
        return EXIT_FAILURE;
    }
    std::printf(
        "[DEVICE] id=%d soc=%s topology=1AIC+2AIV state_bytes=%zu\n", options.device, session.SocName().c_str(),
        sizeof(ProbeState)
    );

    const uint64_t expected_input_a = session.DeviceFieldAddress(offsetof(ProbeState, input_a));
    const uint64_t expected_input_b = session.DeviceFieldAddress(offsetof(ProbeState, input_b));
    const uint64_t expected_output = session.DeviceFieldAddress(offsetof(ProbeState, output));
    std::array<uint32_t, kVisibilityModeCount> control_passes{};
    std::array<uint32_t, kVisibilityModeCount> payload_passes{};
    uint32_t printed_failures = 0U;
    auto state = std::make_unique<ProbeState>();
    for (uint32_t run = 0U; run < options.runs; ++run) {
        for (uint32_t mode_index = 0U; mode_index < kVisibilityModeCount; ++mode_index) {
            const auto mode = static_cast<VisibilityMode>(mode_index);
            const uint64_t nonce = 0xA551000000000000ULL ^ (static_cast<uint64_t>(run + 1U) << 8U) ^ mode_index;
            InitializeState(state.get(), nonce, mode);
            if (!session.Run(state.get())) {
                return EXIT_FAILURE;
            }

            const Validation validation =
                Validate(*state, mode, nonce, expected_input_a, expected_input_b, expected_output);
            control_passes[mode_index] += validation.control_ok ? 1U : 0U;
            payload_passes[mode_index] += validation.control_ok && validation.payload_ok ? 1U : 0U;
            if ((!validation.control_ok || !validation.payload_ok) && printed_failures < 8U) {
                const ProbeRoleResult &executor = state->roles[static_cast<uint32_t>(ProbeRole::Aiv1Executor)];
                std::fprintf(
                    stderr,
                    "[RAW-FAIL] run=%u mode=%s reason=%s cell=0x%llx fatal=0x%llx "
                    "builder=%llu/%llu claim=%llu/%llu payload=%llu vector=%llu "
                    "observed_checksum=0x%llx expected_checksum=0x%llx\n",
                    run, ModeName(mode), validation.reason, static_cast<unsigned long long>(state->cell.state),
                    static_cast<unsigned long long>(state->fatal.state),
                    static_cast<unsigned long long>(
                        CountField(state->roles[static_cast<uint32_t>(ProbeRole::Aiv0Builder)], kCountBuildWinShift)
                    ),
                    static_cast<unsigned long long>(
                        CountField(state->roles[static_cast<uint32_t>(ProbeRole::Aiv0Builder)], kCountBuildAttemptShift)
                    ),
                    static_cast<unsigned long long>(CountField(executor, kCountClaimWinShift)),
                    static_cast<unsigned long long>(CountField(executor, kCountClaimAttemptShift)),
                    static_cast<unsigned long long>(HasStatus(executor, kResultPayloadValid)),
                    static_cast<unsigned long long>(HasStatus(executor, kResultVectorExecuted)),
                    static_cast<unsigned long long>(executor.observed_payload_checksum),
                    static_cast<unsigned long long>(executor.expected_payload_checksum)
                );
                if (!validation.control_ok) {
                    PrintProtocolDiagnostic(*state, nonce, expected_input_a, expected_input_b, expected_output);
                }
                ++printed_failures;
            }
        }
    }

    bool passed = true;
    int32_t minimum_mode = -1;
    for (uint32_t mode_index = 0U; mode_index < kVisibilityModeCount; ++mode_index) {
        const auto mode = static_cast<VisibilityMode>(mode_index);
        const bool stable = payload_passes[mode_index] == 0U || payload_passes[mode_index] == options.runs;
        const bool control_exact = control_passes[mode_index] == options.runs;
        std::printf(
            "[SUMMARY] mode=%-24s protocol=%u/%u payload+golden=%u/%u stable=%s\n", ModeName(mode),
            control_passes[mode_index], options.runs, payload_passes[mode_index], options.runs, stable ? "YES" : "NO"
        );
        passed = passed && control_exact;
        if (minimum_mode < 0 && payload_passes[mode_index] == options.runs) {
            minimum_mode = static_cast<int32_t>(mode_index);
        }
    }

    const uint32_t conservative_mode = static_cast<uint32_t>(VisibilityMode::WriterAndReaderDcci);
    passed = passed && payload_passes[conservative_mode] == options.runs && minimum_mode >= 0;
    if (minimum_mode >= 0) {
        std::printf(
            "[CHARACTERIZATION] AIV0-builder -> AIV1-executor minimum=%s; this S1 probe does not test AIC reads\n",
            ModeName(static_cast<VisibilityMode>(minimum_mode))
        );
    }
    std::printf(
        "[%s] S1 mixed single-Vector runs=%u modes=%llu reused_address=yes golden_elements=%u\n",
        passed ? "PASS" : "FAIL", options.runs, static_cast<unsigned long long>(kVisibilityModeCount), kElementCount
    );
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
