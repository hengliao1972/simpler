/*
 * Copyright (c) PyPTO Contributors.
 * SPDX-License-Identifier: CANN-2.0
 */
#include "protocol_shared.h"

#include <ctime>

namespace {

using namespace plan_protocol_probe;

constexpr uint32_t kProducerCommand = 1U;
constexpr uint64_t kNanosecondsPerSecond = UINT64_C(1000000000);

inline uint64_t MonotonicNanoseconds()
{
    timespec timestamp{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) return 0U;
    return static_cast<uint64_t>(timestamp.tv_sec) * kNanosecondsPerSecond +
           static_cast<uint64_t>(timestamp.tv_nsec);
}

inline void FullBarrier()
{
    __asm__ volatile("dsb sy" ::: "memory");
}

inline void InstructionBarrier()
{
    __asm__ volatile("isb" ::: "memory");
}

inline void InvalidateHostLine(const void *address)
{
    const uintptr_t line = reinterpret_cast<uintptr_t>(address) & ~uintptr_t{63U};
    __asm__ volatile("dc civac, %0" : : "r"(line) : "memory");
    FullBarrier();
    InstructionBarrier();
}

inline void CleanLine(const void *address)
{
    const uintptr_t line = reinterpret_cast<uintptr_t>(address) & ~uintptr_t{63U};
    __asm__ volatile("dc cvac, %0" : : "r"(line) : "memory");
}

// This is intentionally an ordinary volatile store followed by exact clean,
// matching the public AICPU ProducerOps contract.  The AIV side is the party
// that performs a return-ready atomic observation.
inline void PublishControl(volatile int64_t *address, int64_t value)
{
    *address = value;
    __asm__ volatile("" ::: "memory");
    CleanLine(const_cast<const int64_t *>(address));
    FullBarrier();
    InstructionBarrier();
}

inline void Delay(uint32_t nops)
{
    for (volatile uint32_t index = 0U; index < nops; ++index) {
        __asm__ volatile("nop");
    }
}

void PublishProducerResult(ProbeState *state, const ProducerResult &result)
{
    state->producer = result;
    CleanLine(&state->producer);
    FullBarrier();
    InstructionBarrier();
}

void Fail(ProbeState *state, Status status, uint32_t tasks, uint64_t begin_ns)
{
    ProducerResult result{};
    result.magic = kProducerResultMagic;
    result.status = static_cast<uint32_t>(status);
    result.task_count = tasks;
    result.begin_ns = begin_ns;
    result.end_ns = MonotonicNanoseconds();
    PublishProducerResult(state, result);
    PublishControl(&state->fatal.value, static_cast<int64_t>(status));
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int plan_protocol_aicpu_exec(void *argument)
{
    using namespace plan_protocol_probe;
    if (argument == nullptr) return 0;
    auto *kernel_args = static_cast<PlanAicpuKernelArgs *>(argument);
    if (kernel_args->runtime_args_device == 0U || kernel_args->command != kProducerCommand) return 0;
    auto *state = reinterpret_cast<ProbeState *>(
        static_cast<uintptr_t>(kernel_args->runtime_args_device));

    InvalidateHostLine(&state->config);
    const ProbeConfig config = state->config;
    const uint64_t begin_ns = MonotonicNanoseconds();
    if (config.magic != kProbeMagic || config.version != kProbeVersion ||
        config.workload > static_cast<uint32_t>(Workload::B256) ||
        config.publication > static_cast<uint32_t>(Publication::PerItemFrontier) ||
        config.fault_mode > static_cast<uint32_t>(FaultMode::SkipLastPayloadCleanAndPoison)) {
        Fail(state, Status::BadConfig, 0U, begin_ns);
        return 0;
    }

    const auto workload = static_cast<Workload>(config.workload);
    const auto publication = static_cast<Publication>(config.publication);
    const auto fault = static_cast<FaultMode>(config.fault_mode);
    const uint32_t task_count = TaskCount(workload);
    if (task_count == 0U || task_count > kMaxTasks || task_count != config.expected_tasks ||
        (fault == FaultMode::SkipLastPayloadCleanAndPoison && config.fault_task >= task_count)) {
        Fail(state, Status::BadTaskCount, task_count, begin_ns);
        return 0;
    }

    uint64_t payload_clean_lines = 0U;
    uint64_t payload_publish_barriers = 0U;
    uint64_t control_clean_lines = 0U;
    uint64_t omitted_clean_lines = 0U;
    for (uint32_t task = 0U; task < task_count; ++task) {
        TaskShape shape{};
        if (!ResolveTask(workload, task, &shape)) {
            Fail(state, Status::ProducerFailure, task, begin_ns);
            return 0;
        }
        const PayloadLayout layout = LayoutForKind(shape.kind);
        const uint32_t payload_lines = layout.payload_lines;
        if (payload_lines == 0U || payload_lines > kMaxPayloadLines) {
            Fail(state, Status::ProducerFailure, task, begin_ns);
            return 0;
        }
        RuntimeTaskPlanCell &cell = state->cells[task];
        for (uint32_t word = 0U; word < payload_lines * 8U; ++word) {
            cell.payload.words[word] =
                ExpectedPayloadWordForShape(shape, layout, config.nonce, word);
        }
        const bool omit_last = fault == FaultMode::SkipLastPayloadCleanAndPoison &&
            task == config.fault_task && payload_lines > 1U;
        for (uint32_t line = 0U; line < payload_lines; ++line) {
            if (omit_last && line + 1U == payload_lines) {
                ++omitted_clean_lines;
                continue;
            }
            CleanLine(reinterpret_cast<const uint8_t *>(&cell.payload) + line * kCacheLineBytes);
            ++payload_clean_lines;
        }
        if (omit_last) {
            // A missing clean alone proved observationally nondeterministic on
            // this A5 path (the line can still become visible through normal
            // cache writeback).  Poison the deliberately uncommitted line so
            // this negative deterministically tests that the consumer checks
            // every advertised line; the poison itself is also left unclean.
            const uint32_t first_word = (payload_lines - 1U) * 8U;
            for (uint32_t word = first_word; word < payload_lines * 8U; ++word) {
                cell.payload.words[word] =
                    ExpectedPayloadWordForShape(shape, layout, config.nonce, word) ^
                    UINT64_C(0xdeadbeefcafef00d);
            }
        }
        FullBarrier();
        ++payload_publish_barriers;
        PublishControl(
            &cell.control.value,
            static_cast<int64_t>(EncodeCellControl(payload_lines, task)));
        ++control_clean_lines;

        if (publication == Publication::PerItemFrontier) {
            PublishControl(&state->planned_frontier.value, static_cast<int64_t>(task + 1U));
            ++control_clean_lines;
            Delay(config.producer_delay_nops);
        }
    }

    if (publication == Publication::CloseOnly) {
        PublishControl(&state->planned_frontier.value, static_cast<int64_t>(task_count));
        ++control_clean_lines;
    }
    PublishControl(&state->closed_task_count.value, static_cast<int64_t>(task_count));
    ++control_clean_lines;
    const uint64_t end_ns = MonotonicNanoseconds();

    ProducerResult result{};
    result.magic = kProducerResultMagic;
    result.status = static_cast<uint32_t>(Status::Ok);
    result.task_count = task_count;
    result.begin_ns = begin_ns;
    result.end_ns = end_ns;
    result.payload_clean_lines = payload_clean_lines;
    result.payload_publish_barriers = payload_publish_barriers;
    result.control_clean_lines = control_clean_lines;
    result.omitted_clean_lines = omitted_clean_lines;
    PublishProducerResult(state, result);
    return 0;
}
