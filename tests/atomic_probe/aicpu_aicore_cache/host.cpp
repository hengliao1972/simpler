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
#include "common/kernel_args.h"
#include "load_aicpu_op.h"
#include "shared.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace aicpu_aicore_cache_probe;

constexpr uint64_t kDefaultAicoreTimeoutTicks = UINT64_C(10000000000);
constexpr uint64_t kDefaultAicpuTimeoutNs = UINT64_C(5000000000);

[[noreturn]] void Fail(const char *message) {
    std::fprintf(stderr, "[FAIL] %s\n", message);
    std::exit(EXIT_FAILURE);
}

void CheckAcl(aclError result, const char *operation) {
    if (result == ACL_SUCCESS) return;
    std::fprintf(stderr, "[FAIL] %s: ACL error %d\n", operation, static_cast<int>(result));
    std::exit(EXIT_FAILURE);
}

void CheckRuntime(int result, const char *operation) {
    if (result == 0) return;
    std::fprintf(stderr, "[FAIL] %s: runtime error %d\n", operation, result);
    std::exit(EXIT_FAILURE);
}

std::vector<uint8_t> ReadBinary(const std::string &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const std::streampos end = input.tellg();
    if (end <= std::streampos(0)) return {};
    std::vector<uint8_t> bytes(static_cast<size_t>(end));
    input.seekg(0, std::ios::beg);
    if (!input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        return {};
    }
    return bytes;
}

const char *ExpectationName(Expectation expectation) {
    if (expectation == Expectation::Fresh) return "fresh";
    if (expectation == Expectation::Stale) return "stale";
    return "observe";
}

bool ValidateDistribution(
    const char *field, Expectation expectation, uint64_t fresh, uint64_t stale, uint64_t other, uint64_t rounds
) {
    const bool complete = fresh + stale + other == rounds;
    bool expected = true;
    if (expectation == Expectation::Fresh) {
        expected = fresh == rounds && stale == 0U && other == 0U;
    } else if (expectation == Expectation::Stale) {
        expected = fresh == 0U && stale == rounds && other == 0U;
    }
    if (!complete || !expected) {
        std::printf(
            "    [FAIL] %s expected=%s fresh/stale/other=%llu/%llu/%llu rounds=%llu\n", field,
            ExpectationName(expectation), static_cast<unsigned long long>(fresh),
            static_cast<unsigned long long>(stale), static_cast<unsigned long long>(other),
            static_cast<unsigned long long>(rounds)
        );
    }
    return complete && expected;
}

struct ExpectedDistributions {
    Expectation control_primary;
    Expectation control_reference;
    Expectation payload_primary;
    Expectation payload_reference;
};

ExpectedDistributions Expectations(Direction direction, uint32_t index) {
    if (direction == Direction::AicpuToAicore) {
        const AicpuToAicoreCase spec = ResolveAicpuToAicoreCase(index);
        return {spec.control_primary, spec.control_reference, spec.payload_primary, spec.payload_reference};
    }
    const AicoreToAicpuCase spec = ResolveAicoreToAicpuCase(index);
    return {spec.control_primary, spec.control_reference, spec.payload_primary, spec.payload_reference};
}

const char *CaseName(Direction direction, uint32_t index) {
    return direction == Direction::AicpuToAicore ? AicpuToAicoreCaseName(index) : AicoreToAicpuCaseName(index);
}

bool ValidateResults(const ProbeState &state, Direction direction) {
    const uint32_t case_count =
        direction == Direction::AicpuToAicore ? kAicpuToAicoreCaseCount : kAicoreToAicpuCaseCount;
    bool valid = state.config.magic == kProbeMagic && state.config.version == kProbeVersion &&
                 state.config.direction == static_cast<uint32_t>(direction) && state.config.case_count == case_count &&
                 state.config.rounds == kRounds &&
                 state.aicpu_initialized.value == static_cast<int64_t>(InitToken(direction, state.config.nonce)) &&
                 state.aicpu_errors.value == 0 && state.aicore_errors.value == 0;
    if (!valid) {
        std::printf(
            "[FAIL] global state: init=0x%llx aicpu_errors=%lld aicore_errors=%lld\n",
            static_cast<unsigned long long>(state.aicpu_initialized.value),
            static_cast<long long>(state.aicpu_errors.value), static_cast<long long>(state.aicore_errors.value)
        );
    }

    std::printf(
        "%-2s  %-43s  %-17s  %-17s  %-17s  %-17s  %s\n", "id", "case", "ctrl primary", "ctrl reference", "data primary",
        "data reference", "result"
    );
    for (uint32_t index = 0U; index < case_count; ++index) {
        const CaseResult &result = state.cases[index].result;
        const ExpectedDistributions expected = Expectations(direction, index);
        bool case_valid = result.magic == kResultMagic && result.status == static_cast<uint64_t>(Status::Ok) &&
                          result.rounds_completed == kRounds && result.atomic_return_mismatches == 0U;
        if (direction == Direction::AicpuToAicore) {
            case_valid = result.doorbell_poll_attempts_total >= kRounds && result.doorbell_poll_attempts_max >= 1U &&
                         result.doorbell_poll_attempts_total >= result.doorbell_poll_attempts_max &&
                         result.doorbell_wait_ticks_total >= result.doorbell_wait_ticks_max && case_valid;
        }
        case_valid = ValidateDistribution(
                         "control primary", expected.control_primary, result.control_primary_fresh,
                         result.control_primary_stale, result.control_primary_other, kRounds
                     ) &&
                     case_valid;
        case_valid = ValidateDistribution(
                         "control reference", expected.control_reference, result.control_reference_fresh,
                         result.control_reference_stale, result.control_reference_other, kRounds
                     ) &&
                     case_valid;
        case_valid = ValidateDistribution(
                         "payload primary", expected.payload_primary, result.payload_primary_fresh,
                         result.payload_primary_stale, result.payload_primary_other, kRounds
                     ) &&
                     case_valid;
        case_valid = ValidateDistribution(
                         "payload reference", expected.payload_reference, result.payload_reference_fresh,
                         result.payload_reference_stale, result.payload_reference_other, kRounds
                     ) &&
                     case_valid;
        std::printf(
            "%2u  %-43s  %3llu/%3llu/%3llu %-6s  %3llu/%3llu/%3llu %-6s  "
            "%3llu/%3llu/%3llu %-6s  %3llu/%3llu/%3llu %-6s  %s\n",
            index, CaseName(direction, index), static_cast<unsigned long long>(result.control_primary_fresh),
            static_cast<unsigned long long>(result.control_primary_stale),
            static_cast<unsigned long long>(result.control_primary_other), ExpectationName(expected.control_primary),
            static_cast<unsigned long long>(result.control_reference_fresh),
            static_cast<unsigned long long>(result.control_reference_stale),
            static_cast<unsigned long long>(result.control_reference_other),
            ExpectationName(expected.control_reference), static_cast<unsigned long long>(result.payload_primary_fresh),
            static_cast<unsigned long long>(result.payload_primary_stale),
            static_cast<unsigned long long>(result.payload_primary_other), ExpectationName(expected.payload_primary),
            static_cast<unsigned long long>(result.payload_reference_fresh),
            static_cast<unsigned long long>(result.payload_reference_stale),
            static_cast<unsigned long long>(result.payload_reference_other),
            ExpectationName(expected.payload_reference), case_valid ? "PASS" : "FAIL"
        );
        if (direction == Direction::AicpuToAicore && result.rounds_completed != 0U) {
            std::printf(
                "    doorbell attempts avg/max=%llu/%llu, wait ticks avg/max=%llu/%llu\n",
                static_cast<unsigned long long>(result.doorbell_poll_attempts_total / result.rounds_completed),
                static_cast<unsigned long long>(result.doorbell_poll_attempts_max),
                static_cast<unsigned long long>(result.doorbell_wait_ticks_total / result.rounds_completed),
                static_cast<unsigned long long>(result.doorbell_wait_ticks_max)
            );
        }
        if (!case_valid) {
            std::printf(
                "    status=%llu rounds=%llu return_mismatch=%llu first_other_round=%llu "
                "first_values={0x%llx,0x%llx,0x%llx,0x%llx}\n",
                static_cast<unsigned long long>(result.status),
                static_cast<unsigned long long>(result.rounds_completed),
                static_cast<unsigned long long>(result.atomic_return_mismatches),
                static_cast<unsigned long long>(result.first_bad_round),
                static_cast<unsigned long long>(result.first_control_primary),
                static_cast<unsigned long long>(result.first_control_reference),
                static_cast<unsigned long long>(result.first_payload_primary),
                static_cast<unsigned long long>(result.first_payload_reference)
            );
        }
        valid = case_valid && valid;
    }
    return valid;
}

struct AicoreKernel {
    aclrtBinHandle binary = nullptr;
    aclrtFuncHandle function = nullptr;
};

AicoreKernel LoadAicore(const std::vector<uint8_t> &bytes) {
    AicoreKernel kernel{};
    CheckAcl(atomic_probe::LoadAicoreBinaryFromData(bytes.data(), bytes.size(), &kernel.binary), "load AICore binary");
    CheckAcl(aclrtBinaryGetFunctionByEntry(kernel.binary, 0U, &kernel.function), "resolve AICore entry");
    return kernel;
}

bool RunDirection(
    Direction direction, aclrtFuncHandle aicore_function, aclrtStream aicore_stream, aclrtStream aicpu_stream,
    host::LoadAicpuOp &loader, void *device_state, ProbeState &host_state, int32_t device_id
) {
    std::memset(&host_state, 0, sizeof(host_state));
    host_state.config.magic = kProbeMagic;
    host_state.config.version = kProbeVersion;
    host_state.config.direction = static_cast<uint32_t>(direction);
    host_state.config.case_count =
        direction == Direction::AicpuToAicore ? kAicpuToAicoreCaseCount : kAicoreToAicpuCaseCount;
    host_state.config.rounds = kRounds;
    host_state.config.nonce = (static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()) &
                               UINT64_C(0x000fffffffffffff)) ^
                              (static_cast<uint64_t>(direction) << 48U);
    host_state.config.aicore_timeout_ticks = kDefaultAicoreTimeoutTicks;
    host_state.config.aicpu_timeout_ns = kDefaultAicpuTimeoutNs;
    CheckAcl(
        aclrtMemcpy(device_state, sizeof(host_state), &host_state, sizeof(host_state), ACL_MEMCPY_HOST_TO_DEVICE),
        "initialize probe state"
    );

    struct AicoreArguments {
        uint64_t state_device;
    } aicore_arguments{reinterpret_cast<uint64_t>(device_state)};
    KernelArgs aicpu_arguments{};
    aicpu_arguments.runtime_args = reinterpret_cast<Runtime *>(device_state);
    aicpu_arguments.device_id = static_cast<uint32_t>(device_id);

    const auto begin = std::chrono::steady_clock::now();
    CheckAcl(
        aclrtLaunchKernelWithHostArgs(
            aicore_function, 1U, aicore_stream, nullptr, &aicore_arguments, sizeof(aicore_arguments), nullptr, 0U
        ),
        "launch AICore participant"
    );
    CheckRuntime(
        loader.LaunchBuiltInOp(static_cast<rtStream_t>(aicpu_stream), &aicpu_arguments, 1, host::KernelNames::RunName),
        "launch AICPU participant"
    );
    CheckAcl(aclrtSynchronizeStream(aicpu_stream), "synchronize AICPU participant");
    CheckAcl(aclrtSynchronizeStream(aicore_stream), "synchronize AICore participant");
    const auto end = std::chrono::steady_clock::now();
    CheckAcl(
        aclrtMemcpy(&host_state, sizeof(host_state), device_state, sizeof(host_state), ACL_MEMCPY_DEVICE_TO_HOST),
        "read probe state"
    );

    const double wall_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - begin).count();
    std::printf(
        "\n=== %s: cases=%u rounds/case=%u wall=%.3f ms ===\n",
        direction == Direction::AicpuToAicore ? "AICPU -> AICore" : "AICore -> AICPU", host_state.config.case_count,
        host_state.config.rounds, wall_ms
    );
    return ValidateResults(host_state, direction);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 5) {
        std::fprintf(stderr, "Usage: %s <aicpu-to-aicore.o> <aicore-to-aicpu.o> <dispatcher.so> <owner.so>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int32_t device_id = atomic_probe::DeviceId();
    if (device_id < 0) return EXIT_FAILURE;
    const std::vector<uint8_t> aicpu_to_aicore_binary = ReadBinary(argv[1]);
    const std::vector<uint8_t> aicore_to_aicpu_binary = ReadBinary(argv[2]);
    const std::vector<uint8_t> dispatcher = ReadBinary(argv[3]);
    const std::vector<uint8_t> owner = ReadBinary(argv[4]);
    if (aicpu_to_aicore_binary.empty() || aicore_to_aicpu_binary.empty() || dispatcher.empty() || owner.empty()) {
        Fail("cannot read one or more probe artifacts");
    }

    CheckAcl(aclInit(nullptr), "initialize ACL");
    CheckAcl(aclrtSetDevice(device_id), "set device");
    aclrtStream aicore_stream = nullptr;
    aclrtStream aicpu_stream = nullptr;
    CheckAcl(aclrtCreateStream(&aicore_stream), "create AICore stream");
    CheckAcl(aclrtCreateStream(&aicpu_stream), "create AICPU stream");
    const AicoreKernel aicpu_to_aicore = LoadAicore(aicpu_to_aicore_binary);
    const AicoreKernel aicore_to_aicpu = LoadAicore(aicore_to_aicpu_binary);

    host::LoadAicpuOp loader;
    CheckRuntime(
        loader.BootstrapDispatcher(
            dispatcher.data(), dispatcher.size(), owner.data(), owner.size(), static_cast<rtStream_t>(aicpu_stream),
            device_id
        ),
        "bootstrap reusable AICPU loader"
    );
    CheckRuntime(loader.Init(), "register AICPU owner");

    void *device_state = nullptr;
    CheckAcl(aclrtMalloc(&device_state, sizeof(ProbeState), ACL_MEM_MALLOC_HUGE_FIRST), "allocate probe state");
    if ((reinterpret_cast<uintptr_t>(device_state) & (kAtomicIsolationBytes - 1U)) != 0U) {
        Fail("probe state is not 128-byte aligned");
    }
    auto host_state = std::make_unique<ProbeState>();

    std::printf(
        "=== A5 AICPU/AICore atomic + cache coherence probe ===\n"
        "device=%d state=%zu bytes isolation=%u bytes rounds=%u\n",
        device_id, sizeof(ProbeState), kAtomicIsolationBytes, kRounds
    );
    const bool forward_ok = RunDirection(
        Direction::AicpuToAicore, aicpu_to_aicore.function, aicore_stream, aicpu_stream, loader, device_state,
        *host_state, device_id
    );
    const bool reverse_ok = RunDirection(
        Direction::AicoreToAicpu, aicore_to_aicpu.function, aicore_stream, aicpu_stream, loader, device_state,
        *host_state, device_id
    );

    CheckAcl(aclrtFree(device_state), "free probe state");
    loader.Finalize();
    CheckAcl(aclrtBinaryUnLoad(aicore_to_aicpu.binary), "unload AICore->AICPU binary");
    CheckAcl(aclrtBinaryUnLoad(aicpu_to_aicore.binary), "unload AICPU->AICore binary");
    CheckAcl(aclrtDestroyStream(aicpu_stream), "destroy AICPU stream");
    CheckAcl(aclrtDestroyStream(aicore_stream), "destroy AICore stream");
    CheckAcl(aclrtResetDevice(device_id), "reset device");
    CheckAcl(aclFinalize(), "finalize ACL");
    std::printf(
        "\n[SUMMARY] AICPU->AICore=%s AICore->AICPU=%s\n", forward_ok ? "PASS" : "FAIL", reverse_ok ? "PASS" : "FAIL"
    );
    return forward_ok && reverse_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
