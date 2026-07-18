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

#ifndef PA_SCHEDULER_COMMON_HOST_SUPPORT_H
#define PA_SCHEDULER_COMMON_HOST_SUPPORT_H

#include "pa_model.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace pa_scheduler::host {

// 三种后端共用同一套命令行配置，保证 CPU 语义回归与 A5 上板使用完全相同的工作量。
struct Options {
    std::string kernel_path;
    std::string swimlane_json;
    uint32_t device = 0;
    uint32_t batches = kDefaultBatches;
    uint32_t runs = 5;
    NopCounts nops{kDefaultQkNops, kDefaultSfNops, kDefaultPvNops, kDefaultUpNops};
    bool profile_phases = false;
    bool trace_enabled = true;
    bool trace_atomics = false;
    bool analyze_swimlane = false;
};

enum class ParseStatus {
    Ok,
    Help,
    Error,
};

inline bool ParseUint(const char *raw, uint32_t minimum, uint32_t maximum, uint32_t *value) {
    // 要求整串都能被 strtoul 解析且结果落在给定范围内，拒绝尾随字符和溢出值，
    // 避免参数被部分解析后悄悄改变工作量。
    errno = 0;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || parsed < minimum || parsed > maximum) {
        return false;
    }
    *value = static_cast<uint32_t>(parsed);
    return true;
}

inline bool ParseNopCounts(const char *raw, NopCounts *counts) {
    // 四类 kernel 的 NOP 数必须一次性完整给出，顺序固定为 QK、SF、PV、UP。
    unsigned int qk = 0;
    unsigned int sf = 0;
    unsigned int pv = 0;
    unsigned int up = 0;
    char tail = '\0';
    if (std::sscanf(raw, "%u,%u,%u,%u%c", &qk, &sf, &pv, &up, &tail) != 4) {
        return false;
    }
    constexpr uint32_t kMaxNopCount = 10000000;
    if (qk > kMaxNopCount || sf > kMaxNopCount || pv > kMaxNopCount || up > kMaxNopCount) {
        return false;
    }
    *counts = NopCounts{qk, sf, pv, up};
    return true;
}

inline void PrintUsage(const char *program, bool require_kernel) {
    // require_kernel 只影响 CCEC host 的用法文本，其余 benchmark 参数在三后端完全一致。
    std::fprintf(
        stderr, "Usage: %s%s [--device N] [--batches 1..256] [--runs N] ", program,
        require_kernel ? " --kernel FILE" : ""
    );
    std::fprintf(
        stderr,
        "[--nop-count N | --nop-counts QK,SF,PV,UP] [--profile-phases] [--analyze-swimlane] "
        "[--trace-atomics] [--swimlane-json FILE] [--no-swimlane]\n"
    );
}

inline ParseStatus ParseOptions(int argc, char **argv, bool require_kernel, Options *options) {
    // CCEC host 需要外部 kernel ELF；AscendC 和 CPU 的可执行文件已包含 kernel，因此不需要该参数。
    bool nop_override_seen = false;
    bool swimlane_json_seen = false;
    for (int index = 1; index < argc; ++index) {
        // 无值开关先处理；其余参数统一在消费下一个 argv 前检查缺值，保证错误位置明确。
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            PrintUsage(argv[0], require_kernel);
            return ParseStatus::Help;
        }
        if (argument == "--profile-phases") {
            options->profile_phases = true;
            continue;
        }
        if (argument == "--no-swimlane") {
            options->trace_enabled = false;
            continue;
        }
        if (argument == "--trace-atomics") {
            options->trace_atomics = true;
            continue;
        }
        if (argument == "--analyze-swimlane") {
            options->analyze_swimlane = true;
            continue;
        }
        if (index + 1 >= argc) {
            std::fprintf(stderr, "Missing value after %s\n", argument.c_str());
            return ParseStatus::Error;
        }
        const char *value = argv[++index];
        if (argument == "--kernel" && require_kernel) {
            options->kernel_path = value;
        } else if (argument == "--device") {
            if (!ParseUint(value, 0, INT32_MAX, &options->device)) return ParseStatus::Error;
        } else if (argument == "--batches") {
            if (!ParseUint(value, 1, kMaxBatches, &options->batches)) return ParseStatus::Error;
        } else if (argument == "--runs") {
            if (!ParseUint(value, 1, 1000, &options->runs)) return ParseStatus::Error;
        } else if (argument == "--swimlane-json") {
            if (swimlane_json_seen) {
                std::fprintf(stderr, "Specify --swimlane-json only once.\n");
                return ParseStatus::Error;
            }
            if (*value == '\0') {
                std::fprintf(stderr, "--swimlane-json requires a non-empty path.\n");
                return ParseStatus::Error;
            }
            options->swimlane_json = value;
            swimlane_json_seen = true;
        } else if (argument == "--nop-count") {
            if (nop_override_seen) {
                std::fprintf(stderr, "Specify only one NOP override.\n");
                return ParseStatus::Error;
            }
            uint32_t count = 0;
            if (!ParseUint(value, 0, 10000000, &count)) return ParseStatus::Error;
            options->nops = NopCounts{count, count, count, count};
            nop_override_seen = true;
        } else if (argument == "--nop-counts") {
            if (nop_override_seen) {
                std::fprintf(stderr, "Specify only one NOP override.\n");
                return ParseStatus::Error;
            }
            if (!ParseNopCounts(value, &options->nops)) return ParseStatus::Error;
            nop_override_seen = true;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", argument.c_str());
            return ParseStatus::Error;
        }
    }
    if (require_kernel && options->kernel_path.empty()) {
        std::fprintf(stderr, "--kernel is required\n");
        return ParseStatus::Error;
    }
    if (options->analyze_swimlane && !options->trace_enabled) {
        // 分析和导出都依赖完整 record 缓冲，不能与节省内存的 --no-swimlane 同时使用。
        std::fprintf(stderr, "--analyze-swimlane requires swimlane tracing.\n");
        return ParseStatus::Error;
    }
    if (options->trace_atomics && !options->trace_enabled) {
        std::fprintf(stderr, "--trace-atomics cannot be combined with --no-swimlane.\n");
        return ParseStatus::Error;
    }
    if (!options->swimlane_json.empty() && !options->trace_enabled) {
        std::fprintf(stderr, "--swimlane-json requires swimlane tracing.\n");
        return ParseStatus::Error;
    }
    if (!options->swimlane_json.empty() && options->runs != 1) {
        // 一个文件只对应一次完整采集，禁止多轮运行反复覆盖而丢失轮次边界。
        std::fprintf(stderr, "--swimlane-json requires --runs 1 to avoid overwriting captures.\n");
        return ParseStatus::Error;
    }
    return ParseStatus::Ok;
}

inline void InitializeState(SchedulerState *state, const Options &options) {
    // WorkerState 有意保持真实 PA 每核约 9 MiB 的布局。若 host 每轮清空全部 worker，
    // 会额外触碰并拷贝近 1 GiB 内存；因此只初始化全局前缀和结果区，worker 的活跃字段
    // 由各自 kernel 在启动后复位，这也与真实 PA 的生命周期一致。
    std::memset(state, 0, offsetof(SchedulerState, workers));
    std::memset(&state->config, 0, offsetof(SchedulerState, results) - offsetof(SchedulerState, config));
    std::memset(state->results, 0, sizeof(state->results));
    state->heap_window = kHeapWindow;
    state->heap_base = kSyntheticHeapBase;
    state->heap_size = kHeapBytes;
    state->num_workers = kWorkers;
    state->num_blocks = kAicWorkers;
    state->config.batches = options.batches;
    state->config.workers = kWorkers;
    state->config.nops = options.nops;
    state->config.profile_phases = options.profile_phases ? 1U : 0U;
    for (uint32_t batch = 0; batch < options.batches; ++batch) {
        state->context_lens[batch] = 8192;
    }
    for (uint32_t shard = 0; shard < kCursorShards; ++shard) {
        // -1 表示尚无 task 被 claim；task 0 的 atomicMax 因而也能正常判定唯一 winner。
        state->cube_cursor[shard].value = -1;
        state->vector_cursor[shard].value = -1;
        state->alloc_cursor[shard].value = -1;
    }
    state->frontier.value = -1;
}

inline void ConfigureTrace(SchedulerState *state, const Options &options, const void *trace_base) {
    // device 只持有裸地址和每核容量；TraceHeader/record 缓冲区由 host 单独分配并初始化。
    state->config.trace_enabled = options.trace_enabled
        ? kTracePhasesEnabled | (options.trace_atomics ? kTraceAtomicsEnabled : 0U)
        : 0U;
    state->config.trace_base = options.trace_enabled ? reinterpret_cast<uint64_t>(trace_base) : 0;
    state->config.trace_records_per_core = options.trace_enabled ? kTraceRecordsPerCore : 0;
}

inline void InitializeTraceHeader(TraceHeader *header) {
    // magic、版本、108 槽 header 布局和 1 GHz 频率均与真实 FDWIC 泳道 ABI 对齐。
    std::memset(header, 0, sizeof(*header));
    header->magic = 0x4653574cU;
    header->version = 1;
    header->num_cores = kWorkers;
    header->records_per_core = kTraceRecordsPerCore;
    header->frequency_hz = kSystemCounterHz;
}

// 巨大的 WorkerState 不参与每轮 H2D/D2H；以下三个范围只搬运运行所需的前缀、控制量和结果。
inline constexpr size_t StatePrefixBytes() { return offsetof(SchedulerState, workers); }

inline constexpr size_t ControlBytes() {
    // control sidecar 位于为生产 DistGlobal 保留的总跨度之后，到 results 之前为止。
    return offsetof(SchedulerState, results) - offsetof(SchedulerState, config);
}

inline constexpr size_t ResultBytes() { return sizeof(WorkerResult) * kWorkers; }

struct Metrics {
    // passed 是全部语义断言的合取；submit_span_us 是本用例唯一用于对比 PA 的性能口径。
    bool passed = true;
    double submit_span_us = 0;
};

inline void Expect(bool condition, const char *label, Metrics *metrics) {
    // 所有断言都继续执行，以便一次失败运行尽可能暴露完整状态，而不是遇到首错立即退出。
    std::printf("[ASSERT] %-48s %s\n", label, condition ? "PASS" : "FAIL");
    if (!condition) metrics->passed = false;
}

struct Uint64Distribution {
    uint64_t total = 0;
    double median = 0.0;
    uint64_t p95 = 0;
    uint64_t maximum = 0;
};

inline Uint64Distribution SummarizeUint64(std::vector<uint64_t> values) {
    // 这里按 worker 维度统计累计周期，p95 使用 nearest-rank，避免插值掩盖慢核。
    Uint64Distribution summary;
    if (values.empty()) return summary;

    std::sort(values.begin(), values.end());
    for (uint64_t value : values) summary.total += value;
    const size_t middle = values.size() / 2;
    summary.median = (values.size() & 1U) != 0
        ? static_cast<double>(values[middle])
        : (static_cast<double>(values[middle - 1]) + static_cast<double>(values[middle])) / 2.0;
    const size_t p95_rank = (95U * values.size() + 99U) / 100U;
    summary.p95 = values[p95_rank - 1];
    summary.maximum = values.back();
    return summary;
}

inline void PrintPhaseDiagnostics(const SchedulerState &state) {
    if (state.config.profile_phases == 0) return;

    // WaitForSlot/HeapGuard 没有各自独立命名的 TracePhase；实际发生等待时会写
    // RingBp 记录，汇总诊断则使用 WorkerResult 中的累计周期和等待次数。
    struct PhaseSpec {
        ProfilePhase phase;
        const char *name;
        int32_t wait_event_index;
    };
    const PhaseSpec phases[] = {
        {ProfilePhase::Claim, "Claim", -1},
        {ProfilePhase::EfDrain, "EfDrain", -1},
        {ProfilePhase::WaitForSlot, "WaitForSlot", 0},
        {ProfilePhase::HeapGuard, "HeapGuard", 1},
    };
    const CoreRole roles[] = {CoreRole::Aic, CoreRole::Aiv};
    const char *role_names[] = {"AIC", "AIV"};

    for (uint32_t role_index = 0; role_index < 2; ++role_index) {
        // AIC/AIV 分开统计，避免 32:64 的参与者数量差异掩盖某一类核上的长尾。
        for (const PhaseSpec &phase : phases) {
            std::vector<uint64_t> cycles;
            std::vector<uint64_t> calls;
            std::vector<uint64_t> wait_events;
            const uint32_t phase_index = static_cast<uint32_t>(phase.phase);
            for (uint32_t worker = 0; worker < kWorkers; ++worker) {
                const WorkerResult &result = state.results[worker];
                if (result.role != static_cast<uint64_t>(roles[role_index])) continue;
                cycles.push_back(result.phase_cycles[phase_index]);
                calls.push_back(result.phase_calls[phase_index]);
                wait_events.push_back(
                    phase.wait_event_index < 0 ? 0 : result.wait_events[static_cast<uint32_t>(phase.wait_event_index)]
                );
            }
            const Uint64Distribution cycle_summary = SummarizeUint64(cycles);
            const Uint64Distribution call_summary = SummarizeUint64(calls);
            const Uint64Distribution wait_summary = SummarizeUint64(wait_events);
            std::printf(
                "[PHASE] role=%s phase=%s workers=%zu accumulated_us_median=%.3f "
                "accumulated_us_p95=%.3f accumulated_us_max=%.3f calls_total=%llu "
                "calls_per_worker_median=%.1f calls_per_worker_p95=%llu calls_per_worker_max=%llu "
                "wait_events_total=%llu wait_events_per_worker_median=%.1f "
                "wait_events_per_worker_p95=%llu wait_events_per_worker_max=%llu\n",
                role_names[role_index], phase.name, cycles.size(), cycle_summary.median / 1000.0,
                static_cast<double>(cycle_summary.p95) / 1000.0,
                static_cast<double>(cycle_summary.maximum) / 1000.0,
                static_cast<unsigned long long>(call_summary.total), call_summary.median,
                static_cast<unsigned long long>(call_summary.p95),
                static_cast<unsigned long long>(call_summary.maximum),
                static_cast<unsigned long long>(wait_summary.total), wait_summary.median,
                static_cast<unsigned long long>(wait_summary.p95),
                static_cast<unsigned long long>(wait_summary.maximum)
            );
        }
    }
}

inline const char *TracePhaseName(uint32_t phase) {
    // 名称必须与 l2_swimlane_records.json 的 fdwic_events schema 保持一致。
    const char *names[] = {
        "Kernel", "Alloc", "Build", "DrainWon", "Replay", "RingBp", "EfDrain", "Commit",
        "Submit", "Materialize", "PrepareMap", "Claim", "Fanin", "Register", "Atomic",
        "ClockBaseline",
    };
    return phase < sizeof(names) / sizeof(names[0]) ? names[phase] : "Unknown";
}

inline const char *AtomicSiteName(uint32_t site) {
    // 顺序与 pa_model.h::AtomicSite 的稳定 raw ABI 完全一致。
    const char *names[] = {
        "StartupIncrement", "StartupPoll", "FatalPoll", "FatalSet", "ClaimMax",
        "FaninFlagLoad", "CompletionVendExchange", "CompletionFlagExchange",
        "FrontierInitialLoad", "FrontierFlagLoad", "FrontierMax", "HeapFrontierLoad",
        "HeapVendLoad", "ReplayDoneIncrement", "ReplayDonePoll",
    };
    return site < sizeof(names) / sizeof(names[0]) ? names[site] : "Unknown";
}

inline const char *AtomicOpName(uint32_t op) {
    const char *names[] = {"Load", "Exchange", "FetchAdd", "FetchMax"};
    return op < sizeof(names) / sizeof(names[0]) ? names[op] : "Unknown";
}

inline AtomicOp AtomicSiteOp(AtomicSite site) {
    switch (site) {
        case AtomicSite::StartupIncrement:
        case AtomicSite::ReplayDoneIncrement:
            return AtomicOp::FetchAdd;
        case AtomicSite::FatalSet:
        case AtomicSite::CompletionVendExchange:
        case AtomicSite::CompletionFlagExchange:
            return AtomicOp::Exchange;
        case AtomicSite::ClaimMax:
        case AtomicSite::FrontierMax:
            return AtomicOp::FetchMax;
        default:
            return AtomicOp::Load;
    }
}

inline bool ValidateTraceHeader(const TraceHeader &header, const char *operation) {
    // 在任何 D2H record 搬运前先验证容量和 dropped，防止损坏 header 导致 scratch 越界或导出残缺泳道。
    // 频率也要求精确为 1 GHz，否则后续 ns/us 换算即使 JSON 合法也没有性能意义。
    const bool valid = header.magic == 0x4653574cU && header.version == 1 &&
                       header.num_cores == kWorkers && header.records_per_core == kTraceRecordsPerCore &&
                       header.frequency_hz == kSystemCounterHz;
    bool core_states_valid = true;
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        core_states_valid &= header.cores[worker].count <= kTraceRecordsPerCore;
        core_states_valid &= header.cores[worker].dropped == 0;
    }
    if (!valid || !core_states_valid) {
        std::fprintf(
            stderr,
            "%s rejected an invalid trace header: magic=0x%08x version=%u cores=%u "
            "records_per_core=%u frequency_hz=%llu core_states_valid=%s\n",
            operation, header.magic, header.version, header.num_cores, header.records_per_core,
            static_cast<unsigned long long>(header.frequency_hz), core_states_valid ? "yes" : "no"
        );
    }
    return valid && core_states_valid;
}

template <typename ReadRecords>
inline bool ExportSwimlaneRecords(
    const TraceHeader &header, const std::string &output_path,
    WinnerWorkloadMode workload_mode, const WorkloadCounts &workload_counts,
    ReadRecords read_records
) {
    if (!ValidateTraceHeader(header, "swimlane export")) return false;
    if (workload_mode != WinnerWorkloadMode::ScalarNop &&
        workload_mode != WinnerWorkloadMode::RealCompute) {
        std::fprintf(stderr, "swimlane export rejected invalid winner workload mode.\n");
        return false;
    }

    // 先写同目录临时文件，全部记录写完并关闭后再 rename 替换，避免把半截 JSON
    // 当成有效采集；这里没有 fsync 文件和目录，不承诺掉电后的持久化原子性。
    const std::string temporary_path = output_path + ".tmp";
    std::FILE *output = std::fopen(temporary_path.c_str(), "wb");
    if (output == nullptr) {
        std::fprintf(
            stderr, "Cannot open swimlane output %s: %s\n", temporary_path.c_str(), std::strerror(errno)
        );
        return false;
    }

    // 采用固定 1 MiB stdio 缓冲并逐核流式写出；默认 256 batch 时约 86 万条，
    // 无论实际 batch 数是多少都不在 host 侧一次性聚合全部 JSON 记录。
    std::vector<char> output_buffer(1U << 20);
    std::setvbuf(output, output_buffer.data(), _IOFBF, output_buffer.size());
    std::fprintf(
        output,
        "{\n\"l2_swimlane_level\":1,\n"
        "\"metadata\":{\"clock_freq_hz\":%llu,\"num_cores\":%u,"
        "\"trace_schema_version\":2,"
        "\"winner_workload\":{\"mode\":\"%s\","
        "\"counts\":{\"qk\":%u,\"sf\":%u,\"pv\":%u,\"up\":%u},"
        "\"unit\":\"%s\",\"engine_mapping\":%s},\"core_types\":[",
        static_cast<unsigned long long>(header.frequency_hz), kWorkers,
        workload_mode == WinnerWorkloadMode::RealCompute ? "real-compute" : "scalar-nop",
        workload_counts.qk, workload_counts.sf, workload_counts.pv, workload_counts.up,
        workload_mode == WinnerWorkloadMode::RealCompute
            ? "complete_128x128_engine_pipeline_iteration"
            : "scalar_nop_instruction",
        workload_mode == WinnerWorkloadMode::RealCompute
            ? "{\"qk\":\"cube_matmul\",\"sf\":\"vector_add\","
              "\"pv\":\"cube_matmul\",\"up\":\"vector_mul\"}"
            : "null"
    );
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        std::fprintf(output, "%s\"%s\"", worker == 0 ? "" : ",", worker < kAicWorkers ? "aic" : "aiv");
    }
    std::fprintf(
        output,
        "]},\n\"aicore_tasks\":[],\n\"aicpu_tasks\":[],\n"
        "\"aicpu_scheduler_phases\":[],\n\"aicpu_orchestrator_phases\":[],\n\"fdwic_events\":[\n"
    );
    // fdwic_events 每行固定十列：core、block、lane、task、function、phase、起止周期、flags、aux。

    bool success = true;
    bool first_record = true;
    uint64_t exported_records = 0;
    std::vector<TraceRecord> scratch(kTraceRecordsPerCore);
    constexpr int32_t kTracePhaseCount = static_cast<int32_t>(TracePhase::ClockBaseline) + 1;
    for (uint32_t worker = 0; worker < kWorkers && success; ++worker) {
        // 每次只读取一个 worker 的有效区间；完整 384 MiB trace 缓冲无需整体回拷。
        const uint32_t available = header.cores[worker].count;
        if (available > header.records_per_core) {
            std::fprintf(
                stderr, "Trace core %u count %u exceeds capacity %u.\n", worker, available,
                header.records_per_core
            );
            success = false;
            break;
        }
        if (available != 0 && !read_records(worker, available, scratch.data())) {
            success = false;
            break;
        }
        for (uint32_t index = 0; index < available; ++index) {
            const TraceRecord &record = scratch[index];
            // 这里只检查 lane/block 的合法范围以及 core_idx 是否落在所属 worker 槽，
            // 不把生产者应遵守的 worker↔block/lane 精确映射误说成 exporter 已完成的校验。
            const bool atomic_record = record.phase == static_cast<int32_t>(TracePhase::Atomic);
            const bool claim_record = record.phase == static_cast<int32_t>(TracePhase::Claim);
            const uint32_t atomic_op = record.flags & kAtomicOpMask;
            const bool atomic_result_used = (record.flags & kAtomicResultUsed) != 0;
            const bool atomic_return_ready = (record.flags & kAtomicReturnReady) != 0;
            const bool atomic_return_ready_valid = !atomic_return_ready ||
                (atomic_result_used &&
                 (atomic_op == static_cast<uint32_t>(AtomicOp::Load) ||
                  atomic_op == static_cast<uint32_t>(AtomicOp::FetchMax)));
            const bool atomic_schema_valid = !atomic_record ||
                (record.auxiliary < static_cast<uint32_t>(AtomicSite::Count) &&
                 atomic_op <= static_cast<uint32_t>(AtomicOp::FetchMax) &&
                 atomic_return_ready_valid &&
                 atomic_op == static_cast<uint32_t>(
                     AtomicSiteOp(static_cast<AtomicSite>(record.auxiliary))
                 ));
            const bool claim_schema_valid = !claim_record ||
                ((record.flags & ~(kClaimWon | kClaimAttempted)) == 0 &&
                 ((record.flags & kClaimWon) == 0 || (record.flags & kClaimAttempted) != 0));
            const bool record_valid = record.end_cycle >= record.start_cycle && record.phase >= 0 &&
                                      record.phase < kTracePhaseCount && record.lane >= 0 && record.lane <= 2 &&
                                      record.block_id >= 0 && record.block_id < static_cast<int32_t>(kAicWorkers) &&
                                      record.core_idx == static_cast<int32_t>(worker) && atomic_schema_valid &&
                                      claim_schema_valid;
            if (!record_valid) {
                std::fprintf(
                    stderr,
                    "Invalid trace record at worker=%u index=%u: phase=%d lane=%d block=%d core=%d "
                    "start=%llu end=%llu\n",
                    worker, index, record.phase, record.lane, record.block_id, record.core_idx,
                    static_cast<unsigned long long>(record.start_cycle),
                    static_cast<unsigned long long>(record.end_cycle)
                );
                success = false;
                break;
            }
            std::fprintf(
                output,
                "%s[%d,%d,%d,%d,%d,\"%s\",%llu,%llu,%u,%u]",
                first_record ? "" : ",\n", record.core_idx, record.block_id, record.lane, record.task_id,
                record.function_id, TracePhaseName(static_cast<uint32_t>(record.phase)),
                static_cast<unsigned long long>(record.start_cycle),
                static_cast<unsigned long long>(record.end_cycle), record.flags, record.auxiliary
            );
            first_record = false;
            ++exported_records;
        }
    }
    if (success) std::fprintf(output, "\n]}\n");
    if (std::ferror(output) != 0) {
        std::fprintf(stderr, "Failed while writing swimlane output %s.\n", temporary_path.c_str());
        success = false;
    }
    if (std::fclose(output) != 0) {
        std::fprintf(stderr, "Failed to close swimlane output %s: %s\n", temporary_path.c_str(), std::strerror(errno));
        success = false;
    }
    if (success && std::rename(temporary_path.c_str(), output_path.c_str()) != 0) {
        std::fprintf(
            stderr, "Cannot finalize swimlane output %s: %s\n", output_path.c_str(), std::strerror(errno)
        );
        success = false;
    }
    if (!success) {
        std::remove(temporary_path.c_str());
        return false;
    }
    std::printf(
        "[SWIMLANE] raw_json=%s events=%llu\n", output_path.c_str(),
        static_cast<unsigned long long>(exported_records)
    );
    return true;
}

template <typename ReadRecords>
inline bool AnalyzeSwimlaneRecords(
    const TraceHeader &header, const SchedulerState &state, ReadRecords read_records
) {
    if (!ValidateTraceHeader(header, "swimlane analysis")) return false;

    // 第一组数组统计“每个 worker 在某阶段的累计时间”；task_durations 则保留重点阶段的单事件分布。
    constexpr uint32_t kTracePhaseCount = static_cast<uint32_t>(TracePhase::ClockBaseline) + 1;
    constexpr TracePhase kDetailedPhases[] = {
        TracePhase::EfDrain, TracePhase::Materialize, TracePhase::Claim, TracePhase::Register,
    };
    uint64_t cycles[kWorkers][kTracePhaseCount] = {};
    uint64_t counts[kWorkers][kTracePhaseCount] = {};
    std::vector<uint64_t> task_durations[2][kTasksPerBatch][sizeof(kDetailedPhases) / sizeof(kDetailedPhases[0])];
    std::vector<uint64_t> atomic_durations[2][static_cast<uint32_t>(AtomicSite::Count)];
    uint64_t atomic_return_ready_counts[2][static_cast<uint32_t>(AtomicSite::Count)] = {};
    std::vector<uint64_t> clock_baselines[2];
    std::vector<uint64_t> clock_dependency_baselines[2];
    uint64_t clock_dependency_applied[2] = {};
    std::vector<TraceRecord> scratch(kTraceRecordsPerCore);
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        const uint32_t available = header.cores[worker].count;
        const uint32_t count = std::min(available, header.records_per_core);
        if (count != 0 && !read_records(worker, count, scratch.data())) {
            return false;
        }
        for (uint32_t index = 0; index < count; ++index) {
            const TraceRecord &record = scratch[index];
            if (record.phase < 0 || record.phase >= static_cast<int32_t>(kTracePhaseCount) ||
                record.end_cycle < record.start_cycle) {
                // 分析器面对单条坏记录选择跳过；严格导出路径会直接拒绝，二者服务于不同诊断目的。
                continue;
            }
            const uint32_t phase = static_cast<uint32_t>(record.phase);
            const uint64_t duration = record.end_cycle - record.start_cycle;
            cycles[worker][phase] += duration;
            ++counts[worker][phase];
            if (record.phase == static_cast<int32_t>(TracePhase::Atomic) &&
                record.auxiliary < static_cast<uint32_t>(AtomicSite::Count)) {
                const uint32_t role_index =
                    state.results[worker].role == static_cast<uint64_t>(CoreRole::Aic) ? 0U : 1U;
                atomic_durations[role_index][record.auxiliary].push_back(duration);
                atomic_return_ready_counts[role_index][record.auxiliary] +=
                    (record.flags & kAtomicReturnReady) != 0;
            }
            if (record.phase == static_cast<int32_t>(TracePhase::ClockBaseline)) {
                const uint32_t role_index =
                    state.results[worker].role == static_cast<uint64_t>(CoreRole::Aic) ? 0U : 1U;
                if ((record.flags & kClockAtomicDependency) != 0) {
                    clock_dependency_baselines[role_index].push_back(duration);
                    clock_dependency_applied[role_index] +=
                        (record.flags & kClockAtomicDependencyApplied) != 0;
                } else {
                    clock_baselines[role_index].push_back(duration);
                }
            }
            if (record.task_id >= 0) {
                // task_id % 5 恰好对应 Alloc/QK/SF/PV/UP，这是固定 PA Case1 图的拓扑约束。
                const uint32_t role_index =
                    state.results[worker].role == static_cast<uint64_t>(CoreRole::Aic) ? 0U : 1U;
                const uint32_t kind = static_cast<uint32_t>(record.task_id) % kTasksPerBatch;
                for (uint32_t detail = 0; detail < sizeof(kDetailedPhases) / sizeof(kDetailedPhases[0]); ++detail) {
                    if (phase == static_cast<uint32_t>(kDetailedPhases[detail])) {
                        task_durations[role_index][kind][detail].push_back(duration);
                    }
                }
            }
        }
    }

    const CoreRole roles[] = {CoreRole::Aic, CoreRole::Aiv};
    const char *role_names[] = {"AIC", "AIV"};
    for (uint32_t role_index = 0; role_index < 2; ++role_index) {
        for (uint32_t phase = 0; phase < kTracePhaseCount; ++phase) {
            std::vector<uint64_t> role_cycles;
            uint64_t record_count = 0;
            for (uint32_t worker = 0; worker < kWorkers; ++worker) {
                if (state.results[worker].role != static_cast<uint64_t>(roles[role_index])) continue;
                role_cycles.push_back(cycles[worker][phase]);
                record_count += counts[worker][phase];
            }
            const Uint64Distribution summary = SummarizeUint64(role_cycles);
            std::printf(
                "[TRACE_PHASE] role=%s phase=%s records=%llu accumulated_us_median=%.3f "
                "accumulated_us_p95=%.3f accumulated_us_max=%.3f\n",
                role_names[role_index], TracePhaseName(phase),
                static_cast<unsigned long long>(record_count), summary.median / 1000.0,
                static_cast<double>(summary.p95) / 1000.0,
                static_cast<double>(summary.maximum) / 1000.0
            );
        }
    }
    for (uint32_t role_index = 0; role_index < 2; ++role_index) {
        const Uint64Distribution summary = SummarizeUint64(clock_baselines[role_index]);
        if (!clock_baselines[role_index].empty()) {
            std::printf(
                "[TRACE_CLOCK] role=%s samples=%zu definition=consecutive-sys-cnt-reads "
                "median_ns=%.1f p95_ns=%llu max_ns=%llu\n",
                role_names[role_index], clock_baselines[role_index].size(), summary.median,
                static_cast<unsigned long long>(summary.p95),
                static_cast<unsigned long long>(summary.maximum)
            );
        }
        const Uint64Distribution dependency_summary =
            SummarizeUint64(clock_dependency_baselines[role_index]);
        if (!clock_dependency_baselines[role_index].empty()) {
            std::printf(
                "[TRACE_CLOCK] role=%s samples=%zu definition=atomic-return-dependency-hook "
                "dependency_applied=%llu/%zu median_ns=%.1f p95_ns=%llu max_ns=%llu\n",
                role_names[role_index], clock_dependency_baselines[role_index].size(),
                static_cast<unsigned long long>(clock_dependency_applied[role_index]),
                clock_dependency_baselines[role_index].size(), dependency_summary.median,
                static_cast<unsigned long long>(dependency_summary.p95),
                static_cast<unsigned long long>(dependency_summary.maximum)
            );
        }
    }
    // Atomic 只报告原始括号分布，不扣除计时底噪，也不把 total_cycles
    // 解释成可与 Submit 墙钟直接相加的“atomic 占比”。return-ready 只表示
    // 本核可消费返回值，不表示其他核已经观察到更新。
    for (uint32_t role_index = 0; role_index < 2; ++role_index) {
        for (uint32_t site = 0; site < static_cast<uint32_t>(AtomicSite::Count); ++site) {
            const std::vector<uint64_t> &durations = atomic_durations[role_index][site];
            if (durations.empty()) continue;
            const Uint64Distribution summary = SummarizeUint64(durations);
            const AtomicOp op = AtomicSiteOp(static_cast<AtomicSite>(site));
            const uint64_t return_ready_count = atomic_return_ready_counts[role_index][site];
            const char *boundary = return_ready_count == durations.size()
                ? "return-ready"
                : (return_ready_count == 0 ? "source-issue" : "mixed");
            std::printf(
                "[TRACE_ATOMIC] role=%s site=%s op=%s events=%zu boundary=%s "
                "return_ready=%llu/%zu bracket_cycles_total=%llu median_ns=%.1f "
                "p95_ns=%llu max_ns=%llu\n",
                role_names[role_index], AtomicSiteName(site), AtomicOpName(static_cast<uint32_t>(op)),
                durations.size(), boundary, static_cast<unsigned long long>(return_ready_count),
                durations.size(), static_cast<unsigned long long>(summary.total), summary.median,
                static_cast<unsigned long long>(summary.p95),
                static_cast<unsigned long long>(summary.maximum)
            );
        }
    }
    const char *kind_names[] = {"Alloc", "QK", "SF", "PV", "UP"};
    // 单事件统计按 role 与 task kind 展开，可区分“该 role 真实参与”与“只回放前端”的成本。
    for (uint32_t role_index = 0; role_index < 2; ++role_index) {
        for (uint32_t kind = 0; kind < kTasksPerBatch; ++kind) {
            for (uint32_t detail = 0; detail < sizeof(kDetailedPhases) / sizeof(kDetailedPhases[0]); ++detail) {
                const std::vector<uint64_t> &durations = task_durations[role_index][kind][detail];
                const Uint64Distribution summary = SummarizeUint64(durations);
                std::printf(
                    "[TRACE_TASK] role=%s kind=%s phase=%s events=%zu median_ns=%.1f p95_ns=%llu max_ns=%llu\n",
                    role_names[role_index], kind_names[kind],
                    TracePhaseName(static_cast<uint32_t>(kDetailedPhases[detail])), durations.size(), summary.median,
                    static_cast<unsigned long long>(summary.p95),
                    static_cast<unsigned long long>(summary.maximum)
                );
            }
        }
    }
    return true;
}

inline Metrics Validate(
    const SchedulerState &state, uint32_t run, double host_us, const TraceHeader *trace_header = nullptr
) {
    Metrics metrics;
    // 每个 worker 都回放全部 task。Alloc 由 96 个 worker 全部执行 atomicMax Claim；
    // 其余 kernel task 只有与 active role 匹配的 AIC 或 AIV 参与 Claim。
    const uint32_t batches = state.config.batches;
    const uint32_t task_count = batches * kTasksPerBatch;
    const uint64_t expected_submits = static_cast<uint64_t>(kWorkers) * task_count;
    const uint64_t expected_claims =
        static_cast<uint64_t>(batches) * (kWorkers + kAicWorkers + kAivWorkers + kAicWorkers + kAivWorkers);
    // 上式依次对应 Alloc、QK、SF、PV、UP 的 active worker 数，默认 256 batch 时为 73728。

    // 聚合量分为调度核心计数、kernel 分布、前端操作数和最终状态四组，便于定位语义偏差。
    uint64_t first_submit = UINT64_MAX;
    uint64_t last_submit = 0;
    uint64_t submits = 0;
    uint64_t claims = 0;
    uint64_t wins = 0;
    uint64_t heap_guards = 0;
    uint64_t fanin_ready_loads = 0;
    uint64_t fanin_not_ready_loads = 0;
    uint64_t frontier_initial_loads = 0;
    uint64_t frontier_updates = 0;
    uint64_t frontier_terminal_loads = 0;
    uint64_t atomic_trace_calls = 0;
    uint64_t duplicates = 0;
    uint64_t cas_retries = 0;
    uint64_t joint_polls = 0;
    uint64_t trace_wait_records = 0;
    uint64_t wins_by_kind[5] = {};
    uint64_t kernel_counts[4] = {};
    uint64_t kernel_cycles[4] = {};
    uint64_t kernel_min[4] = {};
    uint64_t kernel_max[4] = {};
    uint64_t placements[3] = {};
    uint64_t phase_calls[static_cast<uint32_t>(ProfilePhase::Count)] = {};
    uint64_t context_reads = 0;
    uint64_t views_created = 0;
    uint64_t dynamic_create_infos = 0;
    uint64_t arg_resets = 0;
    uint64_t tensor_args_added = 0;
    uint64_t scalar_args_added = 0;
    uint64_t materialized_outputs = 0;
    uint64_t map_inserts = 0;
    uint64_t map_lookups = 0;
    uint64_t slot_tensor_copies = 0;
    uint64_t slot_scalar_copies = 0;
    uint64_t fanin_edges = 0;
    bool worker_ids[kWorkers] = {};
    uint32_t aic_count = 0;
    uint32_t aiv_count = 0;
    uint32_t winning_workers = 0;
    uint64_t max_worker_wins = 0;
    bool worker_shape_ok = true;
    bool submit_timestamps_ok = true;
    bool vend_values_ok = true;
    bool frontend_worker_counts_ok = true;
    bool final_worker_state_ok = true;
    bool worker_checksums_ok = true;
    bool fanin_worker_counts_ok = true;
    bool frontier_worker_counts_ok = true;
    bool role_kernel_routing_ok = true;

    // 按真实输出大小、1 KiB 对齐和 256 MiB 环回规则重算每个 task 可接受的最小 vend。
    uint64_t expected_heap_next = 0;
    bool vend_progress_bounds_ok = true;
    uint32_t first_bad_vend = task_count;
    uint64_t first_bad_vend_minimum = 0;
    uint64_t first_bad_vend_actual = 0;
    std::vector<uint64_t> minimum_vends(task_count);
    const uint64_t output_bytes_by_kind[] = {10240, 524288, 264192, 8192, 0};
    for (uint32_t task_id = 0; task_id < task_count; ++task_id) {
        const uint64_t output_bytes = output_bytes_by_kind[task_id % kTasksPerBatch];
        uint64_t task_base = (expected_heap_next + kOutputAlignment - 1) / kOutputAlignment * kOutputAlignment;
        if (output_bytes != 0 && (task_base % state.heap_size) + output_bytes > state.heap_size) {
            task_base = (task_base / state.heap_size + 1) * state.heap_size;
        }
        expected_heap_next = task_base + output_bytes;
        minimum_vends[task_id] = expected_heap_next;
    }
    for (uint32_t task_id = 0; task_id < task_count; ++task_id) {
        // vend 可以大于本 task 的最小末端，因为 winner 发布的是其本地 heap_cursor 快照；
        // 但不能超过该 worker 完整回放所有 task 后的最终 heap_next。
        if (state.tasks[task_id].vend < minimum_vends[task_id] ||
            state.tasks[task_id].vend > expected_heap_next) {
            vend_progress_bounds_ok = false;
            if (first_bad_vend == task_count) {
                first_bad_vend = task_id;
                first_bad_vend_minimum = minimum_vends[task_id];
                first_bad_vend_actual = state.tasks[task_id].vend;
            }
        }
    }
    // TensorMap 只保留 heap window 内仍可能被依赖的四类输出；floor 对应已安全退休的 task 边界。
    const uint64_t expected_map_live = 4ULL * std::min<uint32_t>(batches, 13);
    const uint64_t expected_map_floor = task_count > kHeapWindow + 1 ? task_count - kHeapWindow - 1 : 0;

    for (uint32_t index = 0; index < kWorkers; ++index) {
        // 每核只写自己独占且按 cache line 隔离的 WorkerResult；host 在 kernel 完成后统一汇总，不引入额外 atomic。
        const WorkerResult &result = state.results[index];
        if (result.worker_id < kWorkers) {
            worker_ids[result.worker_id] = true;
        } else {
            worker_shape_ok = false;
        }
        aic_count += result.role == static_cast<uint32_t>(CoreRole::Aic);
        aiv_count += result.role == static_cast<uint32_t>(CoreRole::Aiv);
        worker_shape_ok &= result.submits == task_count;
        worker_shape_ok &= result.max_occupied <= kUsableSlots;
        worker_shape_ok &= result.final_occupied == 0;
        submit_timestamps_ok &= result.submit_begin != 0;
        submit_timestamps_ok &= result.submit_end >= result.submit_begin;
        submit_timestamps_ok &= result.finish_cycle >= result.submit_end;
        first_submit = std::min(first_submit, result.submit_begin);
        last_submit = std::max(last_submit, result.submit_end);
        submits += result.submits;
        claims += result.claim_attempts;
        wins += result.claim_wins;
        if (result.claim_wins != 0) ++winning_workers;
        max_worker_wins = std::max(max_worker_wins, result.claim_wins);
        heap_guards += result.heap_guards;
        fanin_ready_loads += result.fanin_ready_loads;
        fanin_not_ready_loads += result.fanin_not_ready_loads;
        frontier_initial_loads += result.frontier_initial_loads;
        frontier_updates += result.frontier_updates;
        frontier_terminal_loads += result.frontier_terminal_loads;
        atomic_trace_calls += result.atomic_trace_calls;
        duplicates += result.completion_duplicates;
        cas_retries += result.cas_retries;
        joint_polls += result.joint_polls;
        trace_wait_records += result.wait_events[0] + result.wait_events[1];
        context_reads += result.context_reads;
        views_created += result.views_created;
        dynamic_create_infos += result.dynamic_create_infos;
        arg_resets += result.arg_resets;
        tensor_args_added += result.tensor_args_added;
        scalar_args_added += result.scalar_args_added;
        materialized_outputs += result.materialized_outputs;
        map_inserts += result.map_inserts;
        map_lookups += result.map_lookups;
        slot_tensor_copies += result.slot_tensor_copies;
        slot_scalar_copies += result.slot_scalar_copies;
        fanin_edges += result.fanin_edges;
        frontend_worker_counts_ok &= result.context_reads == batches;
        frontend_worker_counts_ok &= result.views_created == static_cast<uint64_t>(batches) * 2;
        frontend_worker_counts_ok &= result.dynamic_create_infos == static_cast<uint64_t>(batches) * 2;
        frontend_worker_counts_ok &= result.arg_resets == static_cast<uint64_t>(batches) * 4;
        frontend_worker_counts_ok &= result.tensor_args_added == static_cast<uint64_t>(batches) * 22;
        frontend_worker_counts_ok &= result.scalar_args_added == static_cast<uint64_t>(batches) * 9;
        frontend_worker_counts_ok &= result.materialized_outputs == static_cast<uint64_t>(batches) * 8;
        frontend_worker_counts_ok &= result.map_inserts == static_cast<uint64_t>(batches) * 4;
        final_worker_state_ok &= result.final_heap_next == expected_heap_next;
        final_worker_state_ok &= result.map_high_water == expected_map_live;
        final_worker_state_ok &= result.map_live_entries == expected_map_live;
        final_worker_state_ok &= result.map_alive_floor == expected_map_floor;
        final_worker_state_ok &= result.map_cleaned_upto == expected_map_floor;
        worker_checksums_ok &= result.checksum == (0xcbf29ce484222325ULL ^ result.worker_id);
        const uint64_t worker_kernel_completions = result.kernel_counts[0] + result.kernel_counts[1] +
                                                   result.kernel_counts[2] + result.kernel_counts[3];
        if (result.role == static_cast<uint32_t>(CoreRole::Aic)) {
            role_kernel_routing_ok &= result.kernel_counts[1] == 0 && result.kernel_counts[3] == 0;
        } else if (result.role == static_cast<uint32_t>(CoreRole::Aiv)) {
            role_kernel_routing_ok &= result.kernel_counts[0] == 0 && result.kernel_counts[2] == 0;
        } else {
            role_kernel_routing_ok = false;
        }
        const uint64_t worker_completions = result.wins[0] + worker_kernel_completions;
        frontier_worker_counts_ok &= result.frontier_initial_loads == worker_completions;
        frontier_worker_counts_ok &= result.frontier_terminal_loads == result.frontier_initial_loads;
        fanin_worker_counts_ok &= result.fanin_ready_loads >= result.fanin_edges;
        if (result.fanin_ready_loads >= result.fanin_edges) {
            // PA 最大 fanin 为 3；每次失败检查最多先重读两个 ready 前缀，再遇到一个 not-ready。
            fanin_worker_counts_ok &=
                result.fanin_ready_loads - result.fanin_edges <= 2 * result.fanin_not_ready_loads;
        }
        for (uint32_t kind = 0; kind < 5; ++kind)
            wins_by_kind[kind] += result.wins[kind];
        for (uint32_t kind = 0; kind < 4; ++kind) {
            kernel_counts[kind] += result.kernel_counts[kind];
            kernel_cycles[kind] += result.kernel_cycles[kind];
            if (result.kernel_min_cycles[kind] != 0 &&
                (kernel_min[kind] == 0 || result.kernel_min_cycles[kind] < kernel_min[kind])) {
                kernel_min[kind] = result.kernel_min_cycles[kind];
            }
            kernel_max[kind] = std::max(kernel_max[kind], result.kernel_max_cycles[kind]);
        }
        for (uint32_t place = 0; place < 3; ++place)
            placements[place] += result.placement[place];
        for (uint32_t phase = 0; phase < static_cast<uint32_t>(ProfilePhase::Count); ++phase)
            phase_calls[phase] += result.phase_calls[phase];
    }
    for (bool seen : worker_ids)
        worker_shape_ok &= seen;

    uint32_t ready_flags = 0;
    for (uint32_t task_id = 0; task_id < task_count; ++task_id) {
        // ready flag 和 vend 是跨核 completion 的最终外部可见状态，不能只依赖 worker 私有计数判断完成。
        ready_flags += state.tasks[task_id].flag == 1;
        vend_values_ok &= state.tasks[task_id].vend != 0;
        vend_values_ok &= state.tasks[task_id].vend % kOutputAlignment == 0;
    }
    const uint64_t kernel_total = kernel_counts[0] + kernel_counts[1] + kernel_counts[2] + kernel_counts[3];
    const uint64_t placement_total = placements[0] + placements[1] + placements[2];
    const uint64_t fanin_loads = fanin_ready_loads + fanin_not_ready_loads;
    const uint64_t frontier_flag_loads = frontier_updates + frontier_terminal_loads;

    // 第一组断言覆盖参与者拓扑、Claim/winner、completion 和最终 drain 等调度主协议。
    Expect(aic_count == kAicWorkers && aiv_count == kAivWorkers, "participant topology is 32 AIC + 64 AIV", &metrics);
    Expect(worker_shape_ok, "all 96 worker markers and private rings are valid", &metrics);
    Expect(submit_timestamps_ok, "all Submit timing markers are valid", &metrics);
    Expect(state.started_count.value == kWorkers, "started_count is 96", &metrics);
    Expect(submits == expected_submits, "replay count is workers * tasks", &metrics);
    Expect(claims == expected_claims, "Claim attempt count matches PA topology", &metrics);
    Expect(wins == task_count, "exactly one winner per task", &metrics);
    Expect(
        wins_by_kind[0] == batches && wins_by_kind[1] == batches && wins_by_kind[2] == batches &&
            wins_by_kind[3] == batches && wins_by_kind[4] == batches,
        "Alloc/QK/SF/PV/UP winners are one per batch", &metrics
    );
    Expect(kernel_total == static_cast<uint64_t>(batches) * 4, "kernel count is four per batch", &metrics);
    Expect(
        kernel_counts[0] == batches && kernel_counts[1] == batches && kernel_counts[2] == batches &&
            kernel_counts[3] == batches,
        "each kernel kind executes once per batch", &metrics
    );
    Expect(
        role_kernel_routing_ok,
        "AIC executes only QK/PV and AIV executes only SF/UP", &metrics
    );
    Expect(heap_guards == static_cast<uint64_t>(batches) * 4, "heap guard count matches output winners", &metrics);
    Expect(
        fanin_worker_counts_ok && fanin_ready_loads >= fanin_edges &&
            fanin_ready_loads - fanin_edges <= 2 * fanin_not_ready_loads,
        "fanin ready/failure load classification is complete", &metrics
    );
    Expect(
        frontier_worker_counts_ok && frontier_initial_loads == task_count,
        "frontier initial loads match completed tasks", &metrics
    );
    Expect(
        frontier_terminal_loads == task_count && frontier_updates >= task_count,
        "frontier ready/update/terminal load identity is exact", &metrics
    );
    Expect(duplicates == 0, "completion flags are published once", &metrics);
    Expect(ready_flags == task_count, "all task flags are ready", &metrics);
    Expect(vend_values_ok, "all published vend values are nonzero and aligned", &metrics);
    Expect(vend_progress_bounds_ok, "every task vend is within PA worker heap progress bounds", &metrics);
    Expect(state.frontier.value == static_cast<int64_t>(task_count) - 1, "frontier reaches the final task", &metrics);
    Expect(state.replay_done.value == kWorkers, "replay_done is 96", &metrics);
    Expect(state.fatal.value == 0, "fatal remains clear", &metrics);
    Expect(placement_total == kernel_total, "EfDrain + RingBp + final placement covers every kernel", &metrics);
    // joint_polls 是为未来 BlockWon 模拟预留的结果字段，当前调度路径没有递增点；
    // 此断言只确认现有输出保持零，不能单独证明 active_count>=2 分支不可达。
    Expect(joint_polls == 0, "single-lane PA performs no BlockWon polling", &metrics);
    // 第二组断言锁定 scalar 前端工作量，防止编译器优化或后续改动悄悄删掉 PA 模拟步骤。
    Expect(frontend_worker_counts_ok, "every worker replays the exact PA frontend operation counts", &metrics);
    Expect(
        context_reads == static_cast<uint64_t>(kWorkers) * batches &&
            views_created == static_cast<uint64_t>(kWorkers) * batches * 2 &&
            dynamic_create_infos == static_cast<uint64_t>(kWorkers) * batches * 2 &&
            arg_resets == static_cast<uint64_t>(kWorkers) * batches * 4 &&
            tensor_args_added == static_cast<uint64_t>(kWorkers) * batches * 22 &&
            scalar_args_added == static_cast<uint64_t>(kWorkers) * batches * 9 &&
            materialized_outputs == static_cast<uint64_t>(kWorkers) * batches * 8 &&
            map_inserts == static_cast<uint64_t>(kWorkers) * batches * 4,
        "global PA frontend operation totals are exact", &metrics
    );
    Expect(
        map_lookups == static_cast<uint64_t>(batches) * 14 &&
            slot_tensor_copies == static_cast<uint64_t>(batches) * 19 &&
            slot_scalar_copies == static_cast<uint64_t>(batches) * 9 &&
            fanin_edges == static_cast<uint64_t>(batches) * 5,
        "winner-only map, slot-copy, and fanin totals are exact", &metrics
    );
    Expect(final_worker_state_ok, "every worker final heap and TensorMap state is exact", &metrics);
    Expect(worker_checksums_ok, "all frontend registration checksums remain clean", &metrics);

    // 三类 Claim cursor 各有四个 shard；按 task_id 重新推导每个 shard 应停留的最后任务。
    int64_t expected_cube[kCursorShards] = {-1, -1, -1, -1};
    int64_t expected_vector[kCursorShards] = {-1, -1, -1, -1};
    int64_t expected_alloc[kCursorShards] = {-1, -1, -1, -1};
    for (uint32_t task_id = 0; task_id < task_count; ++task_id) {
        const TaskKind kind = static_cast<TaskKind>(task_id % kTasksPerBatch);
        int64_t *cursors = kind == TaskKind::Alloc
            ? expected_alloc
            : (kind == TaskKind::Qk || kind == TaskKind::Pv ? expected_cube : expected_vector);
        cursors[task_id % kCursorShards] = task_id;
    }
    bool cursors_ok = true;
    for (uint32_t shard = 0; shard < kCursorShards; ++shard) {
        cursors_ok &= state.cube_cursor[shard].value == expected_cube[shard];
        cursors_ok &= state.vector_cursor[shard].value == expected_vector[shard];
        cursors_ok &= state.alloc_cursor[shard].value == expected_alloc[shard];
    }
    Expect(cursors_ok, "all sharded Claim cursors reach their exact final task", &metrics);

    if (state.config.profile_phases != 0) {
        // profile 开关关闭时这些字段允许保持零，避免把可选诊断本身变成语义门禁。
        Expect(
            phase_calls[static_cast<uint32_t>(ProfilePhase::Claim)] == expected_submits &&
                phase_calls[static_cast<uint32_t>(ProfilePhase::EfDrain)] == expected_submits &&
                phase_calls[static_cast<uint32_t>(ProfilePhase::WaitForSlot)] ==
                    static_cast<uint64_t>(batches) * 4 &&
                phase_calls[static_cast<uint32_t>(ProfilePhase::HeapGuard)] ==
                    static_cast<uint64_t>(batches) * 4,
            "profile call counts match Claim/EfDrain/WaitForSlot/HeapGuard flow", &metrics
        );
    }

    if (state.config.trace_enabled != 0) {
        // 固定阶段记录数加上动态等待记录数，应与所有 worker 的 header count 精确相等。
        bool trace_shape_ok = trace_header != nullptr;
        uint64_t trace_records = 0;
        uint64_t trace_dropped = 0;
        bool per_worker_trace_counts_ok = true;
        if (trace_header != nullptr) {
            trace_shape_ok &= trace_header->magic == 0x4653574cU;
            trace_shape_ok &= trace_header->version == 1;
            trace_shape_ok &= trace_header->num_cores == kWorkers;
            trace_shape_ok &= trace_header->records_per_core == kTraceRecordsPerCore;
            trace_shape_ok &= trace_header->frequency_hz == kSystemCounterHz;
            for (uint32_t worker = 0; worker < kWorkers; ++worker) {
                trace_records += trace_header->cores[worker].count;
                trace_dropped += trace_header->cores[worker].dropped;
                trace_shape_ok &= trace_header->cores[worker].count <= kTraceRecordsPerCore;
                const WorkerResult &result = state.results[worker];
                const uint64_t worker_kernels = result.kernel_counts[0] + result.kernel_counts[1] +
                                                result.kernel_counts[2] + result.kernel_counts[3];
                const uint64_t worker_expected =
                    7 * result.submits + result.claim_wins - result.wins[0] +
                    2 * worker_kernels + result.wait_events[0] + result.wait_events[1] +
                    (((state.config.trace_enabled & kTraceAtomicsEnabled) != 0)
                         ? result.atomic_trace_calls + 2
                         : 0);
                per_worker_trace_counts_ok &= trace_header->cores[worker].count == worker_expected;
            }
        }
        const uint64_t expected_trace_records =
            static_cast<uint64_t>(batches) * (static_cast<uint64_t>(kWorkers) * 35 + 12) +
            trace_wait_records +
            (((state.config.trace_enabled & kTraceAtomicsEnabled) != 0)
                 ? atomic_trace_calls + 2 * kWorkers
                 : 0);
        // 每 batch 固定记录为 96*35+12；RingBp 等真实等待按运行时次数额外加入。
        Expect(trace_shape_ok, "swimlane header and per-worker capacities are valid", &metrics);
        Expect(trace_dropped == 0, "swimlane records fit without drops", &metrics);
        Expect(trace_records == expected_trace_records, "swimlane record count matches PA phase flow", &metrics);
        Expect(per_worker_trace_counts_ok, "every worker swimlane record count is exact", &metrics);
        if ((state.config.trace_enabled & kTraceAtomicsEnabled) != 0) {
            Expect(atomic_trace_calls != 0, "atomic trace captured source-level calls", &metrics);
        } else {
            Expect(atomic_trace_calls == 0, "atomic trace counters stay zero when disabled", &metrics);
        }
        std::printf(
            "[TRACE] records=%llu expected=%llu dropped=%llu bytes=%zu\n",
            static_cast<unsigned long long>(trace_records),
            static_cast<unsigned long long>(expected_trace_records),
            static_cast<unsigned long long>(trace_dropped), kTraceBytes
        );
        std::printf(
            "[ATOMIC_TRACE] enabled=%s calls=%llu definition=per-record-boundary-flags\n",
            (state.config.trace_enabled & kTraceAtomicsEnabled) != 0 ? "yes" : "no",
            static_cast<unsigned long long>(atomic_trace_calls)
        );
    }

    if (first_submit != UINT64_MAX && last_submit >= first_submit) {
        // 性能口径只覆盖最早 Submit.begin 到最晚 Submit.end，不含启动屏障、最终 drain 和 host 同步。
        metrics.submit_span_us = static_cast<double>(last_submit - first_submit) / 1000.0;
    }
    std::printf(
        "[METRIC] run=%u submit_span_us=%.3f host_launch_us=%.3f claims=%llu fanin_loads=%llu cas_retries=%llu\n", run,
        metrics.submit_span_us, host_us, static_cast<unsigned long long>(claims),
        static_cast<unsigned long long>(fanin_loads), static_cast<unsigned long long>(cas_retries)
    );
    const uint64_t submit_completion_ops =
        claims + heap_guards + fanin_loads + 2ULL * task_count + frontier_initial_loads +
        frontier_flag_loads + frontier_updates;
    std::printf(
        "[ATOMIC] submit_completion_ops=%llu fanin_ready=%llu fanin_not_ready=%llu frontier_initial=%llu "
        "frontier_flag=%llu frontier_ready_fetch_max=%llu frontier_terminal=%llu\n",
        static_cast<unsigned long long>(submit_completion_ops),
        static_cast<unsigned long long>(fanin_ready_loads),
        static_cast<unsigned long long>(fanin_not_ready_loads),
        static_cast<unsigned long long>(frontier_initial_loads),
        static_cast<unsigned long long>(frontier_flag_loads),
        static_cast<unsigned long long>(frontier_updates),
        static_cast<unsigned long long>(frontier_terminal_loads)
    );
    std::printf(
        "[WINNERS] active_workers=%u max_wins_per_worker=%llu\n", winning_workers,
        static_cast<unsigned long long>(max_worker_wins)
    );
    std::printf(
        "[PLACEMENT] EfDrain=%llu RingBp=%llu FinalDrain=%llu\n",
        static_cast<unsigned long long>(placements[static_cast<uint32_t>(DrainPlace::EfDrain)]),
        static_cast<unsigned long long>(placements[static_cast<uint32_t>(DrainPlace::RingBackpressure)]),
        static_cast<unsigned long long>(placements[static_cast<uint32_t>(DrainPlace::FinalDrain)])
    );
    // placement 统计回答 kernel 最终在哪个 drain 点执行，与 TracePhase 的累计 span 互补。
    const char *kernel_names[] = {"QK", "SF", "PV", "UP"};
    const uint32_t targets[] = {kTargetQkTicks, kTargetSfTicks, kTargetPvTicks, kTargetUpTicks};
    for (uint32_t kind = 0; kind < 4; ++kind) {
        const double mean =
            kernel_counts[kind] == 0 ? 0.0 : static_cast<double>(kernel_cycles[kind]) / kernel_counts[kind];
        std::printf(
            "[KERNEL] %-2s count=%llu mean_us=%.3f min_us=%.3f max_us=%.3f target_us=%.3f\n", kernel_names[kind],
            static_cast<unsigned long long>(kernel_counts[kind]), mean / 1000.0, kernel_min[kind] / 1000.0,
            kernel_max[kind] / 1000.0, targets[kind] / 1000.0
        );
    }
    PrintPhaseDiagnostics(state);
    if (!metrics.passed) {
        // 失败时补充第一处未完成 task、vend 边界和 worker 进度，避免只有笼统的 ASSERT FAIL。
        uint32_t first_not_ready = task_count;
        for (uint32_t task_id = 0; task_id < task_count; ++task_id) {
            if (state.tasks[task_id].flag != 1) {
                first_not_ready = task_id;
                break;
            }
        }
        uint64_t min_worker_submits = UINT64_MAX;
        uint64_t max_worker_submits = 0;
        uint32_t incomplete_workers = 0;
        uint32_t occupied_workers = 0;
        uint64_t max_final_occupied = 0;
        for (uint32_t worker = 0; worker < kWorkers; ++worker) {
            const WorkerResult &result = state.results[worker];
            min_worker_submits = std::min(min_worker_submits, result.submits);
            max_worker_submits = std::max(max_worker_submits, result.submits);
            incomplete_workers += result.submits != task_count;
            occupied_workers += result.final_occupied != 0;
            max_final_occupied = std::max(max_final_occupied, result.final_occupied);
        }
        const int64_t retire = state.frontier.value - static_cast<int64_t>(kHeapWindow);
        const uint64_t retire_vend =
            retire >= 0 && retire < static_cast<int64_t>(task_count) ? state.tasks[retire].vend : 0;
        std::printf(
            "[FAILURE_STATE] fatal=%d frontier=%lld first_not_ready=%u first_bad_vend=%u "
            "vend_minimum=%llu vend_actual=%llu retire=%lld retire_vend=%llu "
            "worker_submits_min=%llu worker_submits_max=%llu incomplete_workers=%u "
            "final_occupied_workers=%u max_final_occupied=%llu\n",
            state.fatal.value, static_cast<long long>(state.frontier.value), first_not_ready, first_bad_vend,
            static_cast<unsigned long long>(first_bad_vend_minimum),
            static_cast<unsigned long long>(first_bad_vend_actual),
            static_cast<long long>(retire), static_cast<unsigned long long>(retire_vend),
            static_cast<unsigned long long>(min_worker_submits),
            static_cast<unsigned long long>(max_worker_submits), incomplete_workers, occupied_workers,
            static_cast<unsigned long long>(max_final_occupied)
        );
    }
    return metrics;
}

inline double Median(std::vector<double> values) {
    // 多轮 benchmark 只报告中位数；上板基线比较仍应优先采用独立进程首轮。
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if ((values.size() & 1U) != 0) return values[middle];
    return (values[middle - 1] + values[middle]) / 2.0;
}

inline void PrintBanner(const char *backend, const Options &options) {
    // 开始运行前完整打印工作量和大内存占用，便于确认比较口径没有混用。
    std::printf("=== Standalone PA Scheduler Benchmark: %s ===\n", backend);
    std::printf(
        "device=%u batches=%u tasks=%u workers=%u runs=%u nops=%u,%u,%u,%u state_bytes=%zu "
        "swimlane=%s trace_atomics=%s trace_bytes=%zu\n", options.device,
        options.batches, options.batches * kTasksPerBatch, kWorkers, options.runs, options.nops.qk, options.nops.sf,
        options.nops.pv, options.nops.up, sizeof(SchedulerState), options.trace_enabled ? "on" : "off",
        options.trace_atomics ? "on" : "off",
        options.trace_enabled ? kTraceBytes : 0
    );
    if (!options.swimlane_json.empty()) {
        std::printf("swimlane_json=%s\n", options.swimlane_json.c_str());
    }
}

}  // namespace pa_scheduler::host

#endif  // PA_SCHEDULER_COMMON_HOST_SUPPORT_H
