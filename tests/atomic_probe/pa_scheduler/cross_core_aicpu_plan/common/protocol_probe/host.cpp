/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */
#include "acl/acl.h"

#include "plan_aicpu_loader.h"
#include "protocol_shared.h"

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace plan_protocol_probe;

constexpr uint32_t kProducerCommand = 1U;
constexpr uint64_t kDefaultTimeoutTicks = UINT64_C(10000000000);
constexpr uint64_t kChecksumSeed = UINT64_C(14695981039346656037);

[[noreturn]] void Fail(const char *message)
{
    std::fprintf(stderr, "[FAIL] %s\n", message);
    std::exit(EXIT_FAILURE);
}

void CheckAcl(aclError result, const char *operation)
{
    if (result == ACL_SUCCESS) return;
    std::fprintf(stderr, "[FAIL] %s: ACL error %d\n", operation, static_cast<int>(result));
    std::exit(EXIT_FAILURE);
}

void CheckRuntime(int result, const char *operation)
{
    if (result == 0) return;
    std::fprintf(stderr, "[FAIL] %s: runtime error %d\n", operation, result);
    std::exit(EXIT_FAILURE);
}

std::vector<uint8_t> ReadBinary(const std::string &path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const std::streampos end = input.tellg();
    if (end <= std::streampos(0)) return {};
    std::vector<uint8_t> bytes(static_cast<size_t>(end));
    input.seekg(0, std::ios::beg);
    if (!input.read(reinterpret_cast<char *>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()))) return {};
    return bytes;
}

int32_t DeviceId()
{
    const char *raw = std::getenv("ATOMIC_PROBE_DEVICE");
    if (raw == nullptr || raw[0] == '\0') raw = std::getenv("TASK_DEVICE");
    if (raw == nullptr || raw[0] == '\0') return 0;
    errno = 0;
    char *end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || value < 0 || value > INT32_MAX) {
        Fail("invalid ATOMIC_PROBE_DEVICE/TASK_DEVICE");
    }
    return static_cast<int32_t>(value);
}

uint32_t OptionalUint(const char *name, uint32_t default_value, uint32_t maximum)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') return default_value;
    errno = 0;
    char *end = nullptr;
    const unsigned long value = std::strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || value > maximum) {
        std::fprintf(stderr, "[FAIL] %s must be in [0,%u]\n", name, maximum);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<uint32_t>(value);
}

uint64_t ExpectedChecksum(Workload workload, uint64_t nonce, uint32_t task_limit)
{
    uint64_t checksum = kChecksumSeed;
    for (uint32_t task = 0U; task < task_limit; ++task) {
        const uint32_t lines = PayloadLinesForTask(workload, task);
        for (uint32_t word = 0U; word < lines * 8U; ++word) {
            checksum = MixChecksum(
                checksum, ExpectedPayloadWord(workload, task, nonce, word));
        }
    }
    return checksum;
}

uint64_t ExpectedNegativeChecksum(
    Workload workload, uint64_t nonce, uint32_t fault_task, uint32_t first_bad_word)
{
    uint64_t checksum = ExpectedChecksum(workload, nonce, fault_task);
    for (uint32_t word = 0U; word < first_bad_word; ++word) {
        checksum = MixChecksum(
            checksum, ExpectedPayloadWord(workload, fault_task, nonce, word));
    }
    return checksum;
}

bool ValidatePositivePayload(
    const ProbeState &state, Workload workload, uint64_t nonce,
    uint32_t *bad_task, uint32_t *bad_word)
{
    for (uint32_t task = 0U; task < TaskCount(workload); ++task) {
        const uint32_t lines = PayloadLinesForTask(workload, task);
        const int64_t expected_control = static_cast<int64_t>(EncodeCellControl(lines, task));
        if (state.cells[task].control.value != expected_control) {
            *bad_task = task;
            *bad_word = UINT32_MAX;
            return false;
        }
        for (uint32_t word = 0U; word < lines * 8U; ++word) {
            if (state.cells[task].payload.words[word] !=
                ExpectedPayloadWord(workload, task, nonce, word)) {
                *bad_task = task;
                *bad_word = word;
                return false;
            }
        }
    }
    return true;
}

uint64_t LinesThroughTask(Workload workload, uint32_t inclusive_task)
{
    uint64_t lines = 0U;
    for (uint32_t task = 0U; task <= inclusive_task; ++task) {
        lines += PayloadLinesForTask(workload, task);
    }
    return lines;
}

struct Timing {
    uint64_t producer_ns = 0U;
    uint64_t aiv_ticks = 0U;
    uint64_t wall_ns = 0U;
};

Timing RunOne(
    const char *scenario, Workload workload, Publication publication,
    FaultMode fault_mode, uint32_t fault_task, uint32_t sequence,
    uint32_t delay_nops, void *device_state, ProbeState &host_state,
    aclrtFuncHandle aiv_function, aclrtStream aiv_stream,
    aclrtStream aicpu_stream, PlanAicpuLoader &loader, int32_t device)
{
    const uint32_t tasks = TaskCount(workload);
    std::memset(&host_state, 0xa5, sizeof(host_state));
    host_state.config = ProbeConfig{};
    host_state.config.magic = kProbeMagic;
    host_state.config.version = kProbeVersion;
    host_state.config.workload = static_cast<uint32_t>(workload);
    host_state.config.publication = static_cast<uint32_t>(publication);
    host_state.config.expected_tasks = tasks;
    host_state.config.nonce =
        (static_cast<uint64_t>(sequence + 1U) << 40U) |
        (static_cast<uint64_t>(static_cast<uint32_t>(workload) + 1U) << 32U) |
        (static_cast<uint64_t>(static_cast<uint32_t>(publication) + 1U) << 24U) |
        UINT64_C(0x005a17c3);
    host_state.config.timeout_ticks = kDefaultTimeoutTicks;
    host_state.config.producer_delay_nops = delay_nops;
    host_state.config.fault_mode = static_cast<uint32_t>(fault_mode);
    host_state.config.fault_task = fault_task;
    host_state.producer = ProducerResult{};
    host_state.consumer = ConsumerResult{};
    host_state.planned_frontier.value = 0;
    host_state.closed_task_count.value = kPlanOpen;
    host_state.fatal.value = 0;
    for (uint32_t task = 0U; task < tasks; ++task) {
        host_state.cells[task].control.value = 0;
    }
    const uint64_t nonce = host_state.config.nonce;
    CheckAcl(
        aclrtMemcpy(
            device_state, sizeof(host_state), &host_state, sizeof(host_state),
            ACL_MEMCPY_HOST_TO_DEVICE),
        "initialize reused probe state");

    const AivKernelArgs aiv_args{reinterpret_cast<uint64_t>(device_state)};
    PlanAicpuKernelArgs producer_args{};
    producer_args.runtime_args_device = reinterpret_cast<uint64_t>(device_state);
    producer_args.command = kProducerCommand;
    producer_args.device_id = static_cast<uint32_t>(device);
    const auto wall_begin = std::chrono::steady_clock::now();
    CheckAcl(
        aclrtLaunchKernelWithHostArgs(
            aiv_function, 1U, aiv_stream, nullptr,
            const_cast<AivKernelArgs *>(&aiv_args), sizeof(aiv_args), nullptr, 0U),
        "launch AIV consumer");
    CheckRuntime(
        loader.Launch(aicpu_stream, &producer_args, sizeof(producer_args), 1U),
        "launch AICPU producer");
    CheckAcl(aclrtSynchronizeStream(aicpu_stream), "wait for AICPU producer");
    CheckAcl(aclrtSynchronizeStream(aiv_stream), "wait for AIV consumer");
    const auto wall_end = std::chrono::steady_clock::now();
    CheckAcl(
        aclrtMemcpy(
            &host_state, sizeof(host_state), device_state, sizeof(host_state),
            ACL_MEMCPY_DEVICE_TO_HOST),
        "read reused probe state");

    const bool expect_negative = fault_mode == FaultMode::SkipLastPayloadCleanAndPoison;
    const uint64_t total_lines = TotalPayloadLines(workload);
    const uint64_t expected_omitted = expect_negative ? 1U : 0U;
    const uint64_t expected_control_cleans = publication == Publication::CloseOnly
        ? static_cast<uint64_t>(tasks) + 2U
        : static_cast<uint64_t>(tasks) * 2U + 1U;
    const uint64_t expected_invalidates = expect_negative
        ? LinesThroughTask(workload, fault_task) : total_lines;
    const uint32_t expected_consumed = expect_negative ? fault_task : tasks;
    const uint32_t expected_bad_word = expect_negative
        ? (PayloadLinesForTask(workload, fault_task) - 1U) * 8U : UINT32_MAX;
    const uint64_t expected_checksum = expect_negative
        ? ExpectedNegativeChecksum(workload, nonce, fault_task, expected_bad_word)
        : ExpectedChecksum(workload, nonce, expected_consumed);
    const uint32_t expected_status = static_cast<uint32_t>(
        expect_negative ? Status::PayloadMismatch : Status::Ok);
    uint32_t host_bad_task = UINT32_MAX;
    uint32_t host_bad_word = UINT32_MAX;
    const bool host_payload_ok = expect_negative || ValidatePositivePayload(
        host_state, workload, nonce, &host_bad_task, &host_bad_word);
    const uint32_t fault_lines = expect_negative ? PayloadLinesForTask(workload, fault_task) : 0U;
    const bool negative_detail_ok = !expect_negative ||
        (fault_lines > 1U && host_state.consumer.first_bad_task == fault_task &&
         host_state.consumer.first_bad_word >= (fault_lines - 1U) * 8U &&
         host_state.consumer.first_bad_word < fault_lines * 8U);

    const bool valid =
        host_state.config.magic == kProbeMagic &&
        host_state.config.version == kProbeVersion &&
        host_state.planned_frontier.value == static_cast<int64_t>(tasks) &&
        host_state.closed_task_count.value == static_cast<int64_t>(tasks) &&
        host_state.fatal.value == 0 &&
        host_state.producer.magic == kProducerResultMagic &&
        host_state.producer.status == static_cast<uint32_t>(Status::Ok) &&
        host_state.producer.task_count == tasks &&
        host_state.producer.end_ns >= host_state.producer.begin_ns &&
        host_state.producer.payload_clean_lines == total_lines - expected_omitted &&
        host_state.producer.payload_publish_barriers == tasks &&
        host_state.producer.control_clean_lines == expected_control_cleans &&
        host_state.producer.omitted_clean_lines == expected_omitted &&
        host_state.consumer.magic == kConsumerResultMagic &&
        host_state.consumer.status == expected_status &&
        host_state.consumer.task_count == expected_consumed &&
        host_state.consumer.end_ticks >= host_state.consumer.begin_ticks &&
        host_state.consumer.control_observations != 0U &&
        host_state.consumer.payload_invalidated_lines == expected_invalidates &&
        host_state.consumer.checksum == expected_checksum &&
        (!expect_negative || negative_detail_ok) &&
        (!expect_negative ?
            (host_state.consumer.first_bad_task == UINT32_MAX &&
             host_state.consumer.first_bad_word == UINT32_MAX) : true) &&
        host_payload_ok;

    const uint64_t producer_ns = host_state.producer.end_ns - host_state.producer.begin_ns;
    const uint64_t aiv_ticks = host_state.consumer.end_ticks - host_state.consumer.begin_ticks;
    const uint64_t wall_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_begin).count());
    std::printf(
        "[RESULT] scenario=%s workload=%s mode=%s tasks=%u payload_lines=%llu "
        "expected=%s status=%s producer_ns=%llu aiv_ticks=%llu wall_us=%.3f "
        "payload_clean=%llu payload_dsb=%llu control_clean=%llu observe=%llu "
        "invalidate=%llu omitted=%llu",
        scenario, WorkloadName(workload), PublicationName(publication), tasks,
        static_cast<unsigned long long>(total_lines),
        expect_negative ? "PAYLOAD_MISMATCH" : "OK", valid ? "PASS" : "FAIL",
        static_cast<unsigned long long>(producer_ns),
        static_cast<unsigned long long>(aiv_ticks),
        static_cast<double>(wall_ns) / 1000.0,
        static_cast<unsigned long long>(host_state.producer.payload_clean_lines),
        static_cast<unsigned long long>(host_state.producer.payload_publish_barriers),
        static_cast<unsigned long long>(host_state.producer.control_clean_lines),
        static_cast<unsigned long long>(host_state.consumer.control_observations),
        static_cast<unsigned long long>(host_state.consumer.payload_invalidated_lines),
        static_cast<unsigned long long>(host_state.producer.omitted_clean_lines));
    if (expect_negative || !valid) {
        std::printf(
            " producer_status=%u consumer_status=%u consumer_bad=%u/%u host_bad=%u/%u",
            host_state.producer.status, host_state.consumer.status,
            host_state.consumer.first_bad_task, host_state.consumer.first_bad_word,
            host_bad_task, host_bad_word);
    }
    std::printf("\n");
    if (!valid) Fail("AICPU->AIV multi-line PlanCell validation failed");
    return {producer_ns, aiv_ticks, wall_ns};
}

}  // namespace

int main(int argc, char **argv)
{
    using namespace plan_protocol_probe;
    if (argc != 4) {
        std::fprintf(
            stderr, "Usage: %s <aiv-kernel.o> <dispatcher.so> <aicpu-owner.so>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const uint32_t repeats = OptionalUint("PLAN_PROTOCOL_REPEATS", 1U, 100U);
    if (repeats == 0U) Fail("PLAN_PROTOCOL_REPEATS must be nonzero");
    const uint32_t delay_nops = OptionalUint("PLAN_PROTOCOL_DELAY_NOPS", 0U, 1000000U);
    const int32_t device = DeviceId();
    const std::vector<uint8_t> kernel = ReadBinary(argv[1]);
    if (kernel.empty()) Fail("cannot read AIV kernel");

    CheckAcl(aclInit(nullptr), "initialize ACL");
    CheckAcl(aclrtSetDevice(device), "set device");
    aclrtStream aiv_stream = nullptr;
    aclrtStream aicpu_stream = nullptr;
    CheckAcl(aclrtCreateStream(&aiv_stream), "create AIV stream");
    CheckAcl(aclrtCreateStream(&aicpu_stream), "create AICPU stream");
    aclrtBinaryLoadOption option{};
    option.type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    option.value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
    aclrtBinaryLoadOptions options{&option, 1U};
    aclrtBinHandle binary = nullptr;
    CheckAcl(
        aclrtBinaryLoadFromData(kernel.data(), kernel.size(), &options, &binary),
        "load AIV binary");
    aclrtFuncHandle aiv_function = nullptr;
    CheckAcl(aclrtBinaryGetFunctionByEntry(binary, 0U, &aiv_function), "get AIV entry");

    PlanAicpuLoader loader;
    CheckRuntime(loader.Initialize(argv[2], argv[3], aicpu_stream, device), "initialize Path-A owner");
    void *device_state = nullptr;
    CheckAcl(
        aclrtMalloc(&device_state, sizeof(ProbeState), ACL_MEM_MALLOC_HUGE_FIRST),
        "allocate probe state");
    if ((reinterpret_cast<uintptr_t>(device_state) & (kAtomicIsolationBytes - 1U)) != 0U) {
        Fail("device ProbeState is not 128-byte aligned");
    }
    auto host_state = std::make_unique<ProbeState>();

    std::printf(
        "=== self-contained A5 AICPU->AIV multi-line PlanCell probe ===\n"
        "state_bytes=%zu cell_bytes=%u payload_max={bytes=%u,lines=%u} max_tasks=%u "
        "repeats=%u delay_nops=%u consumer=atomic-observe+exact-DCCI-range+DSB\n",
        sizeof(ProbeState), kCellBytes, kMaxPayloadBytes, kMaxPayloadLines,
        kMaxTasks, repeats, delay_nops);
    const Workload workloads[] = {Workload::G0, Workload::G1, Workload::Mixed, Workload::B256};
    uint32_t sequence = 0U;
    for (Workload workload : workloads) {
        uint64_t close_producer = 0U;
        uint64_t close_aiv = 0U;
        uint64_t close_wall = 0U;
        uint64_t frontier_producer = 0U;
        uint64_t frontier_aiv = 0U;
        uint64_t frontier_wall = 0U;
        for (uint32_t repeat = 0U; repeat < repeats; ++repeat) {
            const Timing close = RunOne(
                "positive-reuse", workload, Publication::CloseOnly,
                FaultMode::None, kNoFaultTask, sequence++, delay_nops,
                device_state, *host_state, aiv_function, aiv_stream,
                aicpu_stream, loader, device);
            const Timing frontier = RunOne(
                "positive-reuse", workload, Publication::PerItemFrontier,
                FaultMode::None, kNoFaultTask, sequence++, delay_nops,
                device_state, *host_state, aiv_function, aiv_stream,
                aicpu_stream, loader, device);
            close_producer += close.producer_ns;
            close_aiv += close.aiv_ticks;
            close_wall += close.wall_ns;
            frontier_producer += frontier.producer_ns;
            frontier_aiv += frontier.aiv_ticks;
            frontier_wall += frontier.wall_ns;
        }
        std::printf(
            "[COMPARE] workload=%s tasks=%u payload_lines=%llu repeats=%u "
            "close_avg={producer_ns=%.1f,aiv_ticks=%.1f,wall_us=%.3f} "
            "frontier_avg={producer_ns=%.1f,aiv_ticks=%.1f,wall_us=%.3f}\n",
            WorkloadName(workload), TaskCount(workload),
            static_cast<unsigned long long>(TotalPayloadLines(workload)), repeats,
            static_cast<double>(close_producer) / repeats,
            static_cast<double>(close_aiv) / repeats,
            static_cast<double>(close_wall) / repeats / 1000.0,
            static_cast<double>(frontier_producer) / repeats,
            static_cast<double>(frontier_aiv) / repeats,
            static_cast<double>(frontier_wall) / repeats / 1000.0);
    }

    // Same allocation, fresh nonce: the final G1 task is a 69-line cell.  The
    // injected run writes all words but omits exactly its last clean, and must
    // be rejected by the AIV.  A subsequent full publication on the same
    // address proves recovery/reuse rather than leaving the negative terminal.
    const uint32_t fault_task = TaskCount(Workload::G1) - 1U;
    (void)RunOne(
        "partial-clean-negative", Workload::G1, Publication::PerItemFrontier,
        FaultMode::SkipLastPayloadCleanAndPoison, fault_task, sequence++, delay_nops,
        device_state, *host_state, aiv_function, aiv_stream,
        aicpu_stream, loader, device);
    (void)RunOne(
        "post-negative-reuse-recovery", Workload::G1, Publication::PerItemFrontier,
        FaultMode::None, kNoFaultTask, sequence++, delay_nops,
        device_state, *host_state, aiv_function, aiv_stream,
        aicpu_stream, loader, device);

    CheckAcl(aclrtFree(device_state), "free probe state");
    CheckRuntime(loader.Finalize(), "finalize Path-A owner");
    CheckAcl(aclrtBinaryUnLoad(binary), "unload AIV binary");
    CheckAcl(aclrtDestroyStream(aicpu_stream), "destroy AICPU stream");
    CheckAcl(aclrtDestroyStream(aiv_stream), "destroy AIV stream");
    CheckAcl(aclrtResetDevice(device), "reset device");
    CheckAcl(aclFinalize(), "finalize ACL");
    std::printf("[SUMMARY] PASS: multi-line positives, reuse, and partial-clean negative validated\n");
    return EXIT_SUCCESS;
}
