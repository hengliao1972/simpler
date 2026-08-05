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

#include "../common/u0_single_slot.h"

#include "acl/acl.h"

#include <algorithm>
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

using namespace pa_scheduler::simt_cross_core::u0;
namespace g0 = pa_scheduler::simt_cross_core::g0;

constexpr uint64_t kExpectedMte3Count = 0U;

struct Options {
    std::string kernel_path;
    int32_t device = 0;
    uint32_t runs = 100U;
};

struct Failure {
    const char *reason = "unknown";
    uint32_t task = kInvalidTaskId;
    uint32_t index = kInvalidTaskId;
    uint64_t expected = 0U;
    uint64_t actual = 0U;
};

bool Fail(Failure *failure, const char *reason, uint32_t task, uint32_t index, uint64_t expected, uint64_t actual) {
    *failure = Failure{reason, task, index, expected, actual};
    return false;
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
    return input.read(data->data(), size).good();
}

template <typename T>
void FillBytes(T *object, uint8_t value) {
    std::memset(object, value, sizeof(*object));
}

template <typename T>
bool AllBytesEqual(const T &object, uint8_t value) {
    const auto *bytes = reinterpret_cast<const uint8_t *>(&object);
    return std::all_of(bytes, bytes + sizeof(object), [value](uint8_t actual) {
        return actual == value;
    });
}

void InitializeGuard(U0Guard *guard, uint64_t nonce, uint32_t guard_id) {
    for (uint32_t word = 0U; word < kWordsPerLine; ++word) {
        guard->words[word] = ExpectedGuardWord(nonce, guard_id, word);
    }
}

bool ValidateGuard(const U0Guard &guard, uint64_t nonce, uint32_t guard_id, Failure *failure) {
    for (uint32_t word = 0U; word < kWordsPerLine; ++word) {
        const uint64_t expected = ExpectedGuardWord(nonce, guard_id, word);
        if (guard.words[word] != expected) {
            return Fail(
                failure, "guard-corruption", kInvalidTaskId, guard_id * kWordsPerLine + word, expected,
                guard.words[word]
            );
        }
    }
    return true;
}

void InitializeAtomicLine(g0::AtomicLine *line, uint64_t value) {
    line->value = static_cast<int64_t>(value);
    std::memset(line->padding, static_cast<int>(kPayloadPoisonWord & 0xFFU), sizeof(line->padding));
}

bool ValidateAtomicLine(const g0::AtomicLine &line, uint64_t expected_value, uint32_t index, Failure *failure) {
    if (static_cast<uint64_t>(line.value) != expected_value) {
        return Fail(failure, "atomic-value", kInvalidTaskId, index, expected_value, static_cast<uint64_t>(line.value));
    }
    for (uint32_t byte = 0U; byte < sizeof(line.padding); ++byte) {
        if (line.padding[byte] != static_cast<uint8_t>(kPayloadPoisonWord & 0xFFU)) {
            return Fail(
                failure, "atomic-padding", kInvalidTaskId, index, kPayloadPoisonWord & 0xFFU, line.padding[byte]
            );
        }
    }
    return true;
}

void InitializeState(U0ProbeState *state, uint64_t nonce) {
    std::memset(state, 0, sizeof(*state));
    state->control.magic = kProbeMagic;
    state->control.version = kProbeVersion;
    state->control.launch_nonce = nonce;
    state->control.timeout_ticks = kDefaultTimeoutTicks;
    state->control.thread_count = kThreadCount;
    state->control.warp_count = kWarpCount;
    state->control.task_count = kTaskCount;
    state->control.role_count = kRoleCount;
    state->control.payload_class_count = kPayloadClassCount;
    state->control.max_payload_lines = kMaxPayloadLines;
    state->control.words_per_line = kWordsPerLine;
    state->control.ubuf_alignment_bytes = kUbufAlignmentBytes;
    state->control.ubuf_region_bytes = kUbufRegionBytes;
    state->control.ubuf_payload_offset_bytes = kUbufPayloadOffsetBytes;
    state->control.transport_kind = TransportKind::SimtUbufReadToGmWordStore;
    state->control.ubuf_slot_count = kUbufSlotCount;

    InitializeGuard(&state->guard_before_tasks, nonce, kGuardBeforeTasks);
    InitializeGuard(&state->guard_after_tasks, nonce, kGuardAfterTasks);
    InitializeGuard(&state->guard_before_roles, nonce, kGuardBeforeRoles);
    InitializeGuard(&state->guard_after_roles, nonce, kGuardAfterRoles);
    InitializeGuard(&state->guard_before_builder_threads, nonce, kGuardBeforeBuilderThreads);
    InitializeGuard(&state->guard_after_builder_threads, nonce, kGuardAfterBuilderThreads);

    for (uint32_t task_id = 0U; task_id < kTaskCount; ++task_id) {
        U0Task &task = state->tasks[task_id];
        task.control.state = static_cast<int64_t>(EmptyState());
        std::memset(task.control.padding, static_cast<int>(kPayloadPoisonWord & 0xFFU), sizeof(task.control.padding));
        for (uint32_t word = 0U; word < kMaxPayloadWords; ++word) {
            task.payload.words[word] = kPayloadPoisonWord;
        }
        FillBytes(&task.build_report, kReportPoisonByte);
        FillBytes(&task.exec_report, kReportPoisonByte);
    }

    std::array<g0::AtomicLine *, 14U> atomics{
        &state->slot_owner,
        &state->slot_busy_depth,
        &state->slot_max_busy_depth,
        &state->slot_acquire_count,
        &state->slot_release_count,
        &state->build_claim_count,
        &state->built_count,
        &state->exec_claim_count,
        &state->done_count,
        &state->mte3_count,
        &state->ubuf_guard_check_count,
        &state->builder_finished_count,
        &state->executor_finished_count,
        &state->fatal,
    };
    for (g0::AtomicLine *line : atomics) {
        InitializeAtomicLine(line, 0U);
    }
    for (U0RoleReport &role : state->roles) {
        FillBytes(&role, kReportPoisonByte);
    }
    for (U0ThreadReport &report : state->builder_threads) {
        FillBytes(&report, kReportPoisonByte);
    }
}

bool ValidateControl(const U0Control &control, uint64_t nonce, Failure *failure) {
    const bool valid =
        control.magic == kProbeMagic && control.version == kProbeVersion && control.launch_nonce == nonce &&
        control.timeout_ticks == kDefaultTimeoutTicks && control.thread_count == kThreadCount &&
        control.warp_count == kWarpCount && control.task_count == kTaskCount && control.role_count == kRoleCount &&
        control.payload_class_count == kPayloadClassCount && control.max_payload_lines == kMaxPayloadLines &&
        control.words_per_line == kWordsPerLine && control.ubuf_alignment_bytes == kUbufAlignmentBytes &&
        control.ubuf_region_bytes == kUbufRegionBytes && control.ubuf_payload_offset_bytes == kUbufPayloadOffsetBytes &&
        control.transport_kind == TransportKind::SimtUbufReadToGmWordStore && control.ubuf_slot_count == kUbufSlotCount;
    return valid ? true : Fail(failure, "control-mismatch", kInvalidTaskId, 0U, kProbeMagic, control.magic);
}

bool ValidateRole(
    const U0RoleReport &report, ProbeRole role, uint32_t owner, uint32_t subblock, uint32_t claims, uint32_t finishes,
    uint64_t nonce, Failure *failure
) {
    const bool valid = report.owner == owner && report.role == role && report.physical_block == 0U &&
                       report.subblock_id == subblock && report.main_scalar_build_action_count == 0U &&
                       report.task_claim_count == claims && report.task_finish_count == finishes &&
                       report.timeout_count == 0U && report.launch_nonce == nonce &&
                       report.result_magic == kResultMagic && report.reserved[0] == 0U && report.reserved[1] == 0U;
    return valid ? true :
                   Fail(failure, "role-report", kInvalidTaskId, static_cast<uint32_t>(role), owner, report.owner);
}

bool ValidateRun(const U0ProbeState &state, uint64_t nonce, Failure *failure) {
    if (!ValidateControl(state.control, nonce, failure) ||
        !ValidateGuard(state.guard_before_tasks, nonce, kGuardBeforeTasks, failure) ||
        !ValidateGuard(state.guard_after_tasks, nonce, kGuardAfterTasks, failure) ||
        !ValidateGuard(state.guard_before_roles, nonce, kGuardBeforeRoles, failure) ||
        !ValidateGuard(state.guard_after_roles, nonce, kGuardAfterRoles, failure) ||
        !ValidateGuard(state.guard_before_builder_threads, nonce, kGuardBeforeBuilderThreads, failure) ||
        !ValidateGuard(state.guard_after_builder_threads, nonce, kGuardAfterBuilderThreads, failure)) {
        return false;
    }

    const std::array<const g0::AtomicLine *, 14U> atomics{
        &state.slot_owner,
        &state.slot_busy_depth,
        &state.slot_max_busy_depth,
        &state.slot_acquire_count,
        &state.slot_release_count,
        &state.build_claim_count,
        &state.built_count,
        &state.exec_claim_count,
        &state.done_count,
        &state.mte3_count,
        &state.ubuf_guard_check_count,
        &state.builder_finished_count,
        &state.executor_finished_count,
        &state.fatal,
    };
    const std::array<uint64_t, 14U> expected_atomic_values{
        kSlotFree,  0U,         1U,         kTaskCount, kTaskCount,
        kTaskCount, kTaskCount, kTaskCount, kTaskCount, kExpectedMte3Count,
        kTaskCount, kTaskCount, 1U,         0U,
    };
    for (uint32_t index = 0U; index < atomics.size(); ++index) {
        if (!ValidateAtomicLine(*atomics[index], expected_atomic_values[index], index, failure)) {
            return false;
        }
    }

    std::array<bool, kTaskCount + 1U> tickets{};
    for (uint32_t task_id = 0U; task_id < kTaskCount; ++task_id) {
        const U0Task &task = state.tasks[task_id];
        if (static_cast<uint64_t>(task.control.state) != DoneState(task_id)) {
            return Fail(
                failure, "task-state", task_id, 0U, DoneState(task_id), static_cast<uint64_t>(task.control.state)
            );
        }
        for (uint32_t byte = 0U; byte < sizeof(task.control.padding); ++byte) {
            if (task.control.padding[byte] != static_cast<uint8_t>(kPayloadPoisonWord & 0xFFU)) {
                return Fail(
                    failure, "task-control-padding", task_id, byte, kPayloadPoisonWord & 0xFFU,
                    task.control.padding[byte]
                );
            }
        }
        const uint32_t payload_lines = PayloadLinesForTask(task_id);
        const uint32_t payload_words = PayloadWordsForTask(task_id);
        for (uint32_t word = 0U; word < payload_words; ++word) {
            const uint64_t expected = ExpectedPayloadWord(nonce, task_id, word);
            if (task.payload.words[word] != expected) {
                return Fail(failure, "payload-word", task_id, word, expected, task.payload.words[word]);
            }
        }
        for (uint32_t word = payload_words; word < kMaxPayloadWords; ++word) {
            if (task.payload.words[word] != kPayloadPoisonWord) {
                return Fail(failure, "payload-tail", task_id, word, kPayloadPoisonWord, task.payload.words[word]);
            }
        }

        const U0BuildReport &build = task.build_report;
        const uint32_t expected_thread = task_id * kWarpSize;
        if (build.task_id != task_id || build.builder_thread != expected_thread || build.builder_warp != task_id ||
            build.builder_lane != 0U || build.phase_bits != kExpectedBuildPhaseBits ||
            build.payload_lines != payload_lines || build.ubuf_words_written != payload_words ||
            build.gm_words_stored != payload_words || build.claim_count != 1U || build.publish_count != 1U ||
            build.release_count != 1U || build.slot_ticket == 0U || build.slot_ticket > kTaskCount ||
            build.launch_nonce != nonce || build.payload_checksum != ExpectedPayloadChecksum(nonce, task_id)) {
            return Fail(failure, "build-report", task_id, 0U, kExpectedBuildPhaseBits, build.phase_bits);
        }
        if (tickets[build.slot_ticket]) {
            return Fail(failure, "duplicate-slot-ticket", task_id, build.slot_ticket, 0U, 1U);
        }
        tickets[build.slot_ticket] = true;

        const U0ExecReport &exec = task.exec_report;
        if (exec.task_id != task_id || exec.executor_owner != kExecutorOwner || exec.executor_physical_block != 0U ||
            exec.executor_subblock_id != 1U || exec.phase_bits != kExpectedExecPhaseBits ||
            exec.payload_lines != payload_lines || exec.payload_words_read != payload_words || exec.claim_count != 1U ||
            exec.completion_count != 1U || exec.checksum_match_count != 1U || exec.reserved32[0] != 0U ||
            exec.reserved32[1] != 0U || exec.launch_nonce != nonce ||
            exec.payload_checksum != ExpectedPayloadChecksum(nonce, task_id)) {
            return Fail(failure, "exec-report", task_id, 0U, kExpectedExecPhaseBits, exec.phase_bits);
        }
    }
    for (uint32_t ticket = 1U; ticket <= kTaskCount; ++ticket) {
        if (!tickets[ticket]) {
            return Fail(failure, "missing-slot-ticket", kInvalidTaskId, ticket, 1U, 0U);
        }
    }

    for (uint32_t thread = 0U; thread < kThreadCount; ++thread) {
        const U0ThreadReport &report = state.builder_threads[thread];
        if (thread % kWarpSize != 0U) {
            if (!AllBytesEqual(report, kReportPoisonByte)) {
                return Fail(
                    failure, "inactive-lane-mutated", kInvalidTaskId, thread, kReportPoisonWord,
                    *reinterpret_cast<const uint64_t *>(&report)
                );
            }
            continue;
        }
        const uint32_t warp = thread / kWarpSize;
        if (report.thread_id != thread || report.warp_id != warp || report.lane_id != 0U ||
            report.active_leader != 1U || report.task_id != warp || report.status != kExpectedBuilderThreadStatus ||
            report.task_attempt_count != 1U || report.slot_attempt_count == 0U || report.launch_nonce != nonce ||
            report.checksum != ExpectedThreadChecksum(nonce, thread, warp, kExpectedBuilderThreadStatus) ||
            report.reserved[0] != 0U || report.reserved[1] != 0U) {
            return Fail(failure, "builder-thread-report", warp, thread, kExpectedBuilderThreadStatus, report.status);
        }
    }

    return ValidateRole(
               state.roles[static_cast<uint32_t>(ProbeRole::AicObserver)], ProbeRole::AicObserver, 0U, 0U, 0U, 0U,
               nonce, failure
           ) &&
           ValidateRole(
               state.roles[static_cast<uint32_t>(ProbeRole::Aiv0Builder)], ProbeRole::Aiv0Builder, kBuilderOwner, 0U,
               0U, 0U, nonce, failure
           ) &&
           ValidateRole(
               state.roles[static_cast<uint32_t>(ProbeRole::Aiv1Executor)], ProbeRole::Aiv1Executor, kExecutorOwner, 1U,
               kTaskCount, kTaskCount, nonce, failure
           );
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
                "aclrtBinaryLoadFromData(U0 mixed ELF)"
            ) ||
            !CheckAcl(aclrtBinaryGetFunctionByEntry(binary_, 0U, &function_), "aclrtBinaryGetFunctionByEntry(U0)") ||
            !CheckAcl(
                aclrtMalloc(&device_state_, sizeof(U0ProbeState), ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(U0ProbeState)"
            )) {
            return false;
        }
        if ((reinterpret_cast<uintptr_t>(device_state_) & (kCacheLineBytes - 1U)) != 0U) {
            std::fprintf(stderr, "device U0ProbeState is not 64-byte aligned: %p\n", device_state_);
            return false;
        }
        return true;
    }

    bool Run(U0ProbeState *state) {
        if (!CheckAcl(
                aclrtMemcpy(device_state_, sizeof(*state), state, sizeof(*state), ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy(H2D U0ProbeState)"
            )) {
            return false;
        }
        struct KernelArgs {
            uint64_t state_pointer;
        } args{reinterpret_cast<uint64_t>(device_state_)};
        static_assert(sizeof(KernelArgs) == sizeof(uint64_t), "unexpected U0 kernel argument ABI");
        return CheckAcl(
                   aclrtLaunchKernelWithHostArgs(function_, 1U, stream_, nullptr, &args, sizeof(args), nullptr, 0U),
                   "aclrtLaunchKernelWithHostArgs(U0 mixed 1:2)"
               ) &&
               CheckAcl(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream(U0)") &&
               CheckAcl(
                   aclrtMemcpy(state, sizeof(*state), device_state_, sizeof(*state), ACL_MEMCPY_DEVICE_TO_HOST),
                   "aclrtMemcpy(D2H U0ProbeState)"
               );
    }

    const std::string &SocName() const { return soc_name_; }
    uint64_t DeviceStateAddress() const { return reinterpret_cast<uint64_t>(device_state_); }

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
        std::fprintf(stderr, "U0 requires A5/Ascend950, but ACL reports: %s\n", session.SocName().c_str());
        return EXIT_FAILURE;
    }

    std::printf(
        "[DEVICE] id=%d soc=%s topology=1*(1AIC+2AIV) state_bytes=%zu state=0x%llx "
        "builder_threads=%u warps=%u active_leaders=%u tasks=%u payload_lines=1/10/16/68 "
        "ubuf_region=%uB payload_offset=%uB alignment=%uB runs=%u\n",
        options.device, session.SocName().c_str(), sizeof(U0ProbeState),
        static_cast<unsigned long long>(session.DeviceStateAddress()), kThreadCount, kWarpCount, kWarpCount, kTaskCount,
        kUbufRegionBytes, kUbufPayloadOffsetBytes, kUbufAlignmentBytes, options.runs
    );
    std::printf("[TRANSPORT] SIMT volatile UBUF store+load -> ordinary GM word store; MTE3 is intentionally absent "
                "and mte3_count must remain zero\n");

    auto state = std::make_unique<U0ProbeState>();
    uint32_t passes = 0U;
    for (uint32_t run = 0U; run < options.runs; ++run) {
        const uint64_t nonce = UINT64_C(0xA550553000000000) ^ (static_cast<uint64_t>(run) + 1U);
        InitializeState(state.get(), nonce);
        if (!session.Run(state.get())) {
            std::fprintf(
                stderr, "[FAIL] run=%u nonce=0x%llx ACL launch/copy failed\n", run + 1U,
                static_cast<unsigned long long>(nonce)
            );
            break;
        }
        Failure failure{};
        if (!ValidateRun(*state, nonce, &failure)) {
            const uint64_t fatal = static_cast<uint64_t>(state->fatal.value);
            const DecodedU0Fatal decoded_fatal = DecodeU0Fatal(state->fatal.value);
            std::fprintf(
                stderr,
                "[FAIL] run=%u nonce=0x%llx reason=%s task=%u index=%u expected=0x%llx actual=0x%llx "
                "fatal=0x%llx fatal_valid=%u fatal_reason=%u fatal_owner=%u fatal_task=%u\n",
                run + 1U, static_cast<unsigned long long>(nonce), failure.reason, failure.task, failure.index,
                static_cast<unsigned long long>(failure.expected), static_cast<unsigned long long>(failure.actual),
                static_cast<unsigned long long>(fatal), decoded_fatal.valid ? 1U : 0U,
                static_cast<uint32_t>(decoded_fatal.reason), decoded_fatal.reporter_owner, decoded_fatal.task_id
            );
            continue;
        }
        ++passes;
        std::printf(
            "[PASS] run=%u nonce=0x%llx tasks=%u slot_depth=0/1 acquire_release=%u/%u "
            "build_exec_done=%u/%u/%u ubuf_guard_checks=%u mte3=0 same_device_address=yes\n",
            run + 1U, static_cast<unsigned long long>(nonce), kTaskCount, kTaskCount, kTaskCount, kTaskCount,
            kTaskCount, kTaskCount, kTaskCount
        );
    }
    std::printf(
        "[SUMMARY] U0 passes=%u/%u same_address_reuse=%s builder_scalar_build_actions=0 "
        "executor_owner=%u transport=SIMT_UBUF_READ_TO_GM_WORD_STORE mte3=0\n",
        passes, options.runs, options.runs > 1U ? "validated" : "not-requested", kExecutorOwner
    );
    return passes == options.runs ? EXIT_SUCCESS : EXIT_FAILURE;
}
