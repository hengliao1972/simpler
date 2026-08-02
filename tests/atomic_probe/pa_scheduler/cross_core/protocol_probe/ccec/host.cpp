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

#include "../../../probe_host.h"
#include "probe_shared.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace pa_scheduler::cross_core;
using namespace pa_scheduler::cross_core::probe;

struct KernelArgs {
    uint64_t state;
};

struct DeviceResources {
    aclrtStream stream = nullptr;
    aclrtBinHandle binary = nullptr;
    aclrtFuncHandle function = nullptr;
    void *state = nullptr;
};

static_assert(sizeof(KernelArgs) == sizeof(uint64_t), "unexpected kernel argument ABI");

constexpr uint64_t kTimeoutTicks = 2'000'000'000ULL;
constexpr uint64_t kPayloadSentinelBase = 0xE000000000000000ULL;

bool Check(aclError error, const char *expression) {
    return atomic_probe::CheckAcl(
        error, expression, __FILE__, __LINE__
    );
}

std::vector<char> ReadBinary(const std::string &path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return {};
    const std::streamsize size = stream.tellg();
    if (size <= 0) return {};
    stream.seekg(0);
    std::vector<char> data(static_cast<size_t>(size));
    if (!stream.read(data.data(), size)) return {};
    return data;
}

bool ParseRuns(const char *raw, uint32_t *runs) {
    errno = 0;
    char *end = nullptr;
    const unsigned long value = std::strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' ||
        value == 0 || value > 1000) {
        return false;
    }
    *runs = static_cast<uint32_t>(value);
    return true;
}

uint64_t PayloadSentinel(uint32_t case_id, uint32_t word) {
    return kPayloadSentinelBase |
           (static_cast<uint64_t>(case_id) << 32U) | word;
}

uint8_t ControlPaddingSentinel(uint32_t case_id) {
    return static_cast<uint8_t>(0x30U + case_id);
}

void InitializeState(ProbeState &state, uint64_t launch_nonce) {
    std::memset(&state, 0, sizeof(state));
    state.control.magic = kProbeMagic;
    state.control.version = kProbeVersion;
    state.control.case_count = kProbeCaseCount;
    state.control.participant_count = kProbeParticipants;
    state.control.delay_nops = 4096;
    state.control.timeout_ticks = kTimeoutTicks;
    state.control.launch_nonce = launch_nonce;
    std::memset(
        state.guard_before, kProbeGuardBefore,
        sizeof(state.guard_before)
    );
    std::memset(
        state.guard_after, kProbeGuardAfter,
        sizeof(state.guard_after)
    );
    state.fatal_case.value = kProbeNoFatal;
    state.protocol_fatal.state = 0;
    for (uint32_t case_id = 0;
         case_id < kProbeCaseCount; ++case_id) {
        SharedExecCell &cell = state.cells[case_id];
        cell.control.state = 0;
        std::memset(
            cell.control.padding,
            ControlPaddingSentinel(case_id),
            sizeof(cell.control.padding)
        );
        for (uint32_t word = 0;
             word < kExecMaxPayloadWords; ++word) {
            cell.payload.words[word] =
                PayloadSentinel(case_id, word);
        }
    }
    for (ExecutionToken &token : state.tokens) {
        ResetExecutionToken(token);
    }
}

bool BytesAre(
    const uint8_t *bytes, size_t count, uint8_t expected
) {
    for (size_t index = 0; index < count; ++index) {
        if (bytes[index] != expected) return false;
    }
    return true;
}

bool ValidateTopology(const ProbeState &state, std::string *reason) {
    for (uint32_t participant = 0;
         participant < kProbeParticipants; ++participant) {
        const ProbeParticipantResult &result =
            state.participants[participant];
        const uint32_t expected_role = participant & 1U;
        const uint32_t expected_block = participant / 2U;
        if (result.magic != kProbeResultMagic ||
            result.participant != participant ||
            result.role != expected_role ||
            result.block_index != expected_block ||
            result.block_count != 2 ||
            result.subblock_index != 0 ||
            result.subblock_count != 1) {
            *reason = "mixed-topology-participant-" +
                      std::to_string(participant);
            return false;
        }
        for (uint64_t reserved : result.reserved) {
            if (reserved != 0) {
                *reason = "participant-reserved-" +
                          std::to_string(participant);
                return false;
            }
        }
    }
    return true;
}

bool ValidateCase(
    const ProbeState &state, uint32_t case_id,
    std::string *reason
) {
    const ExecPayloadSpec spec = SpecForCase(case_id);
    ExecPayloadLayout layout{};
    if (!ValidateExecPayloadSpec(spec, layout) ||
        layout.payload_lines !=
            kProbeShapes[ShapeIndexForCase(case_id)].lines ||
        layout.written_words !=
            layout.payload_lines * kExecHeaderWords) {
        *reason = "host-layout-" + std::to_string(case_id);
        return false;
    }
    const int64_t expected_state = static_cast<int64_t>(
        EncodeExecState(
            ExecPhase::Done, BuilderForCase(case_id),
            ExecutorForCase(case_id),
            spec.engine_class, layout.payload_lines,
            spec.task_id
        )
    );
    const SharedExecCell &cell = state.cells[case_id];
    if (cell.control.state != expected_state) {
        *reason = "final-state-" + std::to_string(case_id);
        return false;
    }
    const DecodedExecState final_state =
        DecodeExecState(cell.control.state);
    if (!final_state.valid ||
        final_state.phase != ExecPhase::Done ||
        final_state.build_owner != BuilderForCase(case_id) ||
        final_state.execute_owner != ExecutorForCase(case_id)) {
        *reason = "final-owners-" + std::to_string(case_id);
        return false;
    }
    if (!BytesAre(
            cell.control.padding,
            sizeof(cell.control.padding),
            ControlPaddingSentinel(case_id)
        )) {
        *reason = "control-padding-" + std::to_string(case_id);
        return false;
    }
    uint64_t expected_hash = 1469598103934665603ULL;
    for (uint32_t word = 0;
         word < layout.written_words; ++word) {
        const uint64_t expected = ExpectedPayloadWord(
            case_id, spec, layout, word
        );
        if (cell.payload.words[word] != expected) {
            *reason = "payload-word-" + std::to_string(case_id) +
                      "-" + std::to_string(word);
            return false;
        }
        expected_hash = HashWord(expected_hash, expected);
    }
    for (uint32_t word = layout.written_words;
         word < kExecMaxPayloadWords; ++word) {
        if (cell.payload.words[word] !=
            PayloadSentinel(case_id, word)) {
            *reason = "payload-range-overrun-" +
                      std::to_string(case_id) + "-" +
                      std::to_string(word);
            return false;
        }
    }

    const ProbeResult &result = state.results[case_id];
    if (result.magic != kProbeResultMagic ||
        result.case_id != case_id ||
        result.status != kProbeExpectedStatus ||
        result.builder_executor != PackBuilderExecutor(
            BuilderForCase(case_id), ExecutorForCase(case_id)
        ) ||
        result.direction_shape_acquire_delay !=
            PackCaseProperties(case_id) ||
        result.reserved0 != 0 ||
        result.observed_checksum != expected_hash ||
        result.expected_checksum != expected_hash ||
        static_cast<int64_t>(result.final_state) != expected_state ||
        result.elapsed_ticks == 0) {
        *reason = "result-" + std::to_string(case_id);
        return false;
    }
    return true;
}

bool ValidateState(const ProbeState &state, std::string *reason) {
    if (state.control.magic != kProbeMagic ||
        state.control.version != kProbeVersion ||
        state.control.case_count != kProbeCaseCount ||
        state.control.participant_count != kProbeParticipants) {
        *reason = "control";
        return false;
    }
    if (state.fatal_case.value != kProbeNoFatal) {
        *reason = "device-fatal-" +
                  std::to_string(state.fatal_case.value);
        return false;
    }
    if (state.protocol_fatal.state != 0) {
        const DecodedExecFatal fatal =
            DecodeExecFatal(state.protocol_fatal.state);
        *reason = fatal.valid
            ? "protocol-fatal-task-" +
                  std::to_string(fatal.task_id)
            : "protocol-fatal-invalid-record";
        return false;
    }
    if (!BytesAre(
            state.guard_before, sizeof(state.guard_before),
            kProbeGuardBefore
        ) ||
        !BytesAre(
            state.guard_after, sizeof(state.guard_after),
            kProbeGuardAfter
        )) {
        *reason = "outer-guard";
        return false;
    }
    if (!ValidateTopology(state, reason)) return false;
    for (uint32_t case_id = 0;
         case_id < kProbeCaseCount; ++case_id) {
        if (!ValidateCase(state, case_id, reason)) return false;
    }
    for (uint32_t participant = 0;
         participant < kProbeParticipants; ++participant) {
        const ExecutionTokenControl &control =
            state.tokens[participant].control;
        if (control.phase != ExecTokenPhase::Idle ||
            control.task_id != UINT32_MAX ||
            control.build_owner != UINT32_MAX ||
            control.execute_owner != UINT32_MAX ||
            control.engine_class != ExecEngineClass::None ||
            control.payload_lines != 0 ||
            control.payload_bytes != 0 ||
            control.fanin_ready_prefix != 0) {
            *reason = "token-not-idle-" +
                      std::to_string(participant);
            return false;
        }
    }
    return true;
}

bool Cleanup(DeviceResources *device, int32_t device_id) {
    bool ok = true;
    if (device->state != nullptr) {
        ok &= Check(aclrtFree(device->state), "aclrtFree(state)");
        device->state = nullptr;
    }
    if (device->binary != nullptr) {
        ok &= Check(
            aclrtBinaryUnLoad(device->binary),
            "aclrtBinaryUnLoad"
        );
        device->binary = nullptr;
    }
    if (device->stream != nullptr) {
        ok &= Check(
            aclrtDestroyStream(device->stream),
            "aclrtDestroyStream"
        );
        device->stream = nullptr;
    }
    ok &= Check(aclrtResetDevice(device_id), "aclrtResetDevice");
    ok &= Check(aclFinalize(), "aclFinalize");
    return ok;
}

}  // namespace

int main(int argc, char **argv) {
    const std::string kernel_path = argc > 1
        ? argv[1]
        : "./cross_core_payload_probe_kernel.o";
    uint32_t runs = 20;
    if (argc > 2 && !ParseRuns(argv[2], &runs)) {
        std::fprintf(stderr, "Invalid run count: %s\n", argv[2]);
        return EXIT_FAILURE;
    }
    if (argc > 3) {
        std::fprintf(
            stderr, "Usage: %s [kernel.o] [runs]\n", argv[0]
        );
        return EXIT_FAILURE;
    }
    const std::vector<char> kernel_data = ReadBinary(kernel_path);
    if (kernel_data.empty()) {
        std::fprintf(
            stderr, "Cannot read kernel binary: %s\n",
            kernel_path.c_str()
        );
        return EXIT_FAILURE;
    }
    const int32_t device_id = atomic_probe::DeviceId();
    if (device_id < 0) return EXIT_FAILURE;

    DeviceResources device{};
    if (!Check(aclInit(nullptr), "aclInit") ||
        !Check(aclrtSetDevice(device_id), "aclrtSetDevice") ||
        !Check(
            aclrtCreateStream(&device.stream),
            "aclrtCreateStream"
        ) ||
        !Check(
            atomic_probe::LoadAicoreBinaryFromData(
                kernel_data.data(), kernel_data.size(),
                &device.binary
            ),
            "LoadAicoreBinaryFromData"
        ) ||
        !Check(
            aclrtBinaryGetFunctionByEntry(
                device.binary, 0U, &device.function
            ),
            "aclrtBinaryGetFunctionByEntry"
        ) ||
        !Check(
            aclrtMalloc(
                &device.state, sizeof(ProbeState),
                ACL_MEM_MALLOC_HUGE_FIRST
            ),
            "aclrtMalloc(ProbeState)"
        )) {
        Cleanup(&device, device_id);
        return EXIT_FAILURE;
    }
    if (reinterpret_cast<uintptr_t>(device.state) %
            kExecCacheLineBytes != 0) {
        std::fprintf(stderr, "Device ProbeState is not 64B aligned.\n");
        Cleanup(&device, device_id);
        return EXIT_FAILURE;
    }

    std::unique_ptr<ProbeState> host_state(new ProbeState);
    std::printf(
        "=== cross-core portable payload probe ===\n"
        "device=%d kernel=%s bytes=%zu state=%zu "
        "topology=2AIC+2AIV cases=%u runs=%u\n",
        device_id, kernel_path.c_str(), kernel_data.size(),
        sizeof(ProbeState), kProbeCaseCount, runs
    );
    for (uint32_t run = 0; run < runs; ++run) {
        InitializeState(*host_state, run + 1U);
        if (!Check(
                aclrtMemcpy(
                    device.state, sizeof(ProbeState),
                    host_state.get(), sizeof(ProbeState),
                    ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "aclrtMemcpy(H2D ProbeState)"
            )) {
            Cleanup(&device, device_id);
            return EXIT_FAILURE;
        }
        KernelArgs args{
            reinterpret_cast<uint64_t>(device.state)
        };
        if (!Check(
                aclrtLaunchKernelWithHostArgs(
                    device.function, 2U, device.stream, nullptr,
                    &args, sizeof(args), nullptr, 0U
                ),
                "aclrtLaunchKernelWithHostArgs(cross-core probe)"
            ) ||
            !Check(
                aclrtSynchronizeStream(device.stream),
                "aclrtSynchronizeStream(cross-core probe)"
            ) ||
            !Check(
                aclrtMemcpy(
                    host_state.get(), sizeof(ProbeState),
                    device.state, sizeof(ProbeState),
                    ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "aclrtMemcpy(D2H ProbeState)"
            )) {
            Cleanup(&device, device_id);
            return EXIT_FAILURE;
        }
        std::string reason;
        if (!ValidateState(*host_state, &reason)) {
            std::fprintf(
                stderr, "[FAIL] run=%u reason=%s fatal=%lld\n",
                run + 1U, reason.c_str(),
                static_cast<long long>(host_state->fatal_case.value)
            );
            Cleanup(&device, device_id);
            return EXIT_FAILURE;
        }
        std::printf(
            "[PASS] run=%u/%u coreids=%u,%u,%u,%u\n",
            run + 1U, runs,
            host_state->participants[0].physical_core_id,
            host_state->participants[1].physical_core_id,
            host_state->participants[2].physical_core_id,
            host_state->participants[3].physical_core_id
        );
    }
    const bool cleanup_ok = Cleanup(&device, device_id);
    std::printf(
        "[SUMMARY] runs=%u cases_per_run=%u status=%s\n",
        runs, kProbeCaseCount, cleanup_ok ? "PASS" : "FAIL"
    );
    return cleanup_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
