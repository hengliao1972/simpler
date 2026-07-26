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
    FinalBarrierShape final_barrier_shape = FinalBarrierShape::TwoLevel16;
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

inline const char *FinalBarrierShapeName(FinalBarrierShape shape) {
    switch (shape) {
    case FinalBarrierShape::Flat:
        return "flat";
    case FinalBarrierShape::TwoLevel4:
        return "two-4";
    case FinalBarrierShape::TwoLevel8:
        return "two-8";
    case FinalBarrierShape::TwoLevel16:
        return "two-16";
    case FinalBarrierShape::ThreeLevel6x4x4:
        return "three-6x4x4";
    }
    return "invalid";
}

inline bool ParseFinalBarrierShape(const char *raw, FinalBarrierShape *shape) {
    struct Entry {
        const char *name;
        FinalBarrierShape shape;
    };
    constexpr Entry kEntries[] = {
        {"flat", FinalBarrierShape::Flat},
        {"two-4", FinalBarrierShape::TwoLevel4},
        {"two-8", FinalBarrierShape::TwoLevel8},
        {"two-16", FinalBarrierShape::TwoLevel16},
        {"three-6x4x4", FinalBarrierShape::ThreeLevel6x4x4},
    };
    for (const Entry &entry : kEntries) {
        if (std::strcmp(raw, entry.name) == 0) {
            *shape = entry.shape;
            return true;
        }
    }
    return false;
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
        "[--trace-atomics] [--swimlane-json FILE] [--no-swimlane] "
        "[--final-barrier flat|two-4|two-8|two-16|three-6x4x4] (default: two-16)\n"
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
        } else if (argument == "--final-barrier") {
            if (!ParseFinalBarrierShape(value, &options->final_barrier_shape)) {
                std::fprintf(stderr, "Unknown final barrier shape: %s\n", value);
                return ParseStatus::Error;
            }
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
#if PTO_FDWIC_SHARED_MAP
    // shared_map 位于 results 之后，不属于上面的 control/result 任一范围。
    // 先把 payload、bucket 游标和保留字节清零，再建立协议要求的 -1
    // seq/reclaim sentinel；这样每轮复用同一 host/device 分配时不会继承
    // 上一轮 lap。S2.5 不再使用 per-core replay progress。
    std::memset(&state->shared_map, 0, sizeof(state->shared_map));
    state->shared_map.committed_tasks.value = 0;
    state->shared_map.reclaim_upto.value = -1;
    for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
        state->shared_map.buckets[bucket].head.value = 0;
        state->shared_map.buckets[bucket].tail.value = 0;
    }
    for (uint32_t slot = 0; slot < kMapCapacity; ++slot) {
        state->shared_map.slots[slot].seq.value = -1;
    }
    // 每个 task 的 fresh Output 只在本轮使用一次，发布位与最后 writer 都用
    // -1 表示“尚无可消费 descriptor”。TensorDesc 区已由上方 memset 清零；
    // 不对 task_id 取模，避免在本阶段提前引入 generation 语义。
    for (uint32_t task_id = 0; task_id < kMaxTasks; ++task_id) {
        // task 表位于 production prefix，前面的 memset 会把该字段清零；
        // shared 协议必须用 -1 区分“尚未准备”与 task 0 已发布。
        state->tasks[task_id].deps_prepared = -1;
        for (uint32_t slot = 0; slot < kSharedOutputMaxPerTask; ++slot) {
            state->shared_map.shared_outputs[task_id].published[slot].value = -1;
            state->shared_map.shared_outputs[task_id].last_writer[slot].value = -1;
        }
    }
    // shared heap 允许不同 winner 并发推进分片 cursor 与 aggregate vend；
    // 每轮仍必须从绝对零点开始，不能继承上一轮 sidecar 的终态。
    for (uint32_t shard = 0; shard < kSharedHeapShards; ++shard) {
        state->shared_map.shared_heap_cursor[shard].value = 0;
    }
    state->shared_map.shared_heap_vend.value = 0;
    // shared Vector Claim cursor 与 heap cursor 是两套独立状态；-1 表示
    // 尚未 Claim 任一 SF/UP task。每轮完整复位，避免继承旧高水位。
    for (uint32_t shard = 0; shard < kSharedVectorCursorCapacity; ++shard) {
        state->shared_map.shared_vector_cursor[shard].value = -1;
    }
#endif
    state->heap_window = kHeapWindow;
    state->heap_base = kSyntheticHeapBase;
    state->heap_size = kHeapBytes;
    state->num_workers = kWorkers;
    state->num_blocks = kAicWorkers;
    state->config.batches = options.batches;
    state->config.workers = kWorkers;
    state->config.nops = options.nops;
    state->config.profile_phases = options.profile_phases ? 1U : 0U;
    state->config.final_barrier_shape = static_cast<uint32_t>(options.final_barrier_shape);
    state->config.build_identity_magic = kBuildIdentityMagic;
    state->config.build_identity_abi_version = kBuildIdentityAbiVersion;
    state->config.tensor_map_mode = static_cast<uint32_t>(kCompiledTensorMapMode);
    state->config.scheduler_state_size = static_cast<uint32_t>(sizeof(SchedulerState));
    state->pmu_probe.build_variant = kCompiledBuildVariant;
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
    // version=4 表示 phase ABI 已追加父区间和真实 Submit 尾动作；core state
    // 继续携带 weighted atomic/PollBatch 计数和权威拓扑。
    std::memset(header, 0, sizeof(*header));
    header->magic = 0x4653574cU;
    header->version = 4;
    header->num_cores = kWorkers;
    header->records_per_core = kTraceRecordsPerCore;
    header->frequency_hz = kSystemCounterHz;
}

// 巨大的 WorkerState 不参与每轮 H2D/D2H。private 仍只搬前缀、控制量和
// 结果三个既有范围；shared 额外把 results 后的 map sidecar 作为第四个
// 独立范围搬运，不能把约 2 MiB 状态混入 ControlBytes/ResultBytes。
inline constexpr size_t StatePrefixBytes() { return offsetof(SchedulerState, workers); }

inline constexpr size_t ControlBytes() {
    // control sidecar 位于为生产 DistGlobal 保留的总跨度之后，依次覆盖
    // RunConfig、独立 PMU 配置、winner workload、context 和 final barrier。
    return offsetof(SchedulerState, results) - offsetof(SchedulerState, config);
}

inline constexpr size_t ResultBytes() { return sizeof(WorkerResult) * kWorkers; }

inline constexpr size_t SharedSidecarBytes() { return sizeof(SharedTensorMapSidecar); }
#if PTO_FDWIC_SHARED_MAP
static_assert(SharedSidecarBytes() == 11027648, "shared TensorMap transfer size changed");
#else
static_assert(SharedSidecarBytes() == 2113664, "private TensorMap transfer size changed");
#endif

inline constexpr size_t FinalBarrierStateBytes() { return sizeof(FinalBarrierState); }

struct Metrics {
    // lifecycle_* 来自跨核一致的 1 GHz SYS_CNT；host_launch_us 仍只作为包含
    // launch/synchronize 的外层参考，不与设备内分段时间混算。
    bool passed = true;
    double submit_span_us = 0;
    double startup_barrier_span_us = 0;
    double final_barrier_span_us = 0;
    double final_drain_span_us = 0;
    double lifecycle_span_us = 0;
};

inline void Expect(bool condition, const char *label, Metrics *metrics) {
    // 所有断言都继续执行，以便一次失败运行尽可能暴露完整状态，而不是遇到首错立即退出。
    std::printf("[ASSERT] %-48s %s\n", label, condition ? "PASS" : "FAIL");
    if (!condition) metrics->passed = false;
}

inline bool FinalBarrierStateMatches(const FinalBarrierState &barrier, FinalBarrierShape shape) {
    uint32_t leaf_groups = 0;
    int64_t leaf_arrivals = 0;
    uint32_t middle_groups = 0;
    int64_t middle_arrivals = 0;
    int64_t root_arrivals = 0;
    switch (shape) {
    case FinalBarrierShape::Flat:
        break;
    case FinalBarrierShape::TwoLevel4:
        leaf_groups = 4;
        leaf_arrivals = 24;
        root_arrivals = 4;
        break;
    case FinalBarrierShape::TwoLevel8:
        leaf_groups = 8;
        leaf_arrivals = 12;
        root_arrivals = 8;
        break;
    case FinalBarrierShape::TwoLevel16:
        leaf_groups = 16;
        leaf_arrivals = 6;
        root_arrivals = 16;
        break;
    case FinalBarrierShape::ThreeLevel6x4x4:
        leaf_groups = 16;
        leaf_arrivals = 6;
        middle_groups = 4;
        middle_arrivals = 4;
        root_arrivals = 4;
        break;
    default:
        return false;
    }
    bool matches = true;
    for (uint32_t group = 0; group < kFinalBarrierMaxLeafGroups; ++group) {
        const bool active = group < leaf_groups;
        matches &= barrier.leaf_arrivals[group].value == (active ? leaf_arrivals : 0);
        matches &= barrier.leaf_releases[group].value == (active ? 1 : 0);
    }
    for (uint32_t group = 0; group < kFinalBarrierMaxMiddleGroups; ++group) {
        const bool active = group < middle_groups;
        matches &= barrier.middle_arrivals[group].value == (active ? middle_arrivals : 0);
        matches &= barrier.middle_releases[group].value == (active ? 1 : 0);
    }
    matches &= barrier.root_arrival.value == root_arrivals;
    matches &= barrier.root_release.value == (root_arrivals == 0 ? 0 : 1);
    return matches;
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
        "ClockBaseline", "OrchestrationReplay", "FinalDrain", "WinnerBuild",
        "AllocComplete",
    };
    static_assert(
        sizeof(names) / sizeof(names[0]) == static_cast<uint32_t>(TracePhase::Count),
        "TracePhaseName must cover every trace phase"
    );
    return phase < sizeof(names) / sizeof(names[0]) ? names[phase] : "Unknown";
}

inline const char *AtomicSiteName(uint32_t site) {
    // 顺序与 pa_model.h::AtomicSite 的稳定 raw ABI 完全一致。
    const char *names[] = {
        "StartupIncrement", "StartupPoll", "FatalPoll", "FatalSet", "ClaimMax",
        "FaninFlagLoad", "CompletionVendExchange", "CompletionFlagExchange",
        "FrontierInitialLoad", "FrontierFlagLoad", "FrontierMax", "HeapFrontierLoad",
        "HeapVendLoad", "ReplayDoneIncrement", "ReplayDonePoll",
        "SharedHeapVendLoad", "SharedHeapCursorLoad",
        "SharedHeapCursorReserve", "SharedHeapVendAdvance",
    };
    static_assert(
        sizeof(names) / sizeof(names[0]) ==
            static_cast<uint32_t>(AtomicSite::Count),
        "AtomicSiteName must cover every atomic site"
    );
    return site < sizeof(names) / sizeof(names[0]) ? names[site] : "Unknown";
}

inline const char *AtomicOpName(uint32_t op) {
    const char *names[] = {"Load", "Exchange", "FetchAdd", "FetchMax"};
    return op < sizeof(names) / sizeof(names[0]) ? names[op] : "Unknown";
}

inline AtomicOp AtomicSiteOp(AtomicSite site) {
    return AtomicSiteExpectedOp(site);
}

inline bool ValidateTraceHeader(const TraceHeader &header, const char *operation) {
    // 在任何 D2H record 搬运前先验证容量和 dropped，防止损坏 header 导致 scratch 越界或导出残缺泳道。
    // 频率也要求精确为 1 GHz，否则后续 ns/us 换算即使 JSON 合法也没有性能意义。
    const bool valid = header.magic == 0x4653574cU && header.version == 4 &&
                       header.num_cores == kWorkers && header.records_per_core == kTraceRecordsPerCore &&
                       header.frequency_hz == kSystemCounterHz;
    bool core_states_valid = true;
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        const TraceCoreState &core = header.cores[worker];
        core_states_valid &= core.count <= kTraceRecordsPerCore;
        core_states_valid &= core.dropped == 0;
        core_states_valid &= core.poll_calls <= core.atomic_calls;
        core_states_valid &= (core.poll_calls == 0) == (core.poll_batch_records == 0);
        const uint64_t physical_atomic =
            static_cast<uint64_t>(core.atomic_calls) - core.poll_calls + core.poll_batch_records;
        core_states_valid &= physical_atomic <= core.count;
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

struct TraceExportSummary {
    uint64_t records = 0;
    uint64_t atomic_records = 0;
    uint64_t clock_baseline_records = 0;
    uint64_t atomic_calls = 0;
    uint64_t poll_calls = 0;
    uint64_t poll_batch_records = 0;
    uint64_t dropped_records = 0;
};

inline bool SameTraceSummary(const TraceExportSummary &left, const TraceExportSummary &right) {
    return left.records == right.records && left.atomic_records == right.atomic_records &&
           left.clock_baseline_records == right.clock_baseline_records &&
           left.atomic_calls == right.atomic_calls && left.poll_calls == right.poll_calls &&
           left.poll_batch_records == right.poll_batch_records &&
           left.dropped_records == right.dropped_records;
}

inline uint32_t AtomicRecordCallCount(const TraceRecord &record) {
    return (record.flags & kAtomicPollBatch) != 0
        ? record.flags >> kAtomicPollCountShift
        : 1U;
}

inline bool AtomicRecordSchemaValid(const TraceRecord &record, bool atomic_trace_enabled) {
    if (!atomic_trace_enabled || record.auxiliary >= static_cast<uint32_t>(AtomicSite::Count)) {
        return false;
    }
    const AtomicSite site = static_cast<AtomicSite>(record.auxiliary);
#if !PTO_FDWIC_SHARED_MAP
    // private ELF 不得接受 shared-only raw site；否则混用产物或损坏记录会
    // 在 Count/op 校验均通过后被误报成合法 private atomic。
    if (AtomicSiteIsSharedOnly(site)) return false;
#endif
    const uint32_t op = record.flags & kAtomicOpMask;
    if (op != static_cast<uint32_t>(AtomicSiteExpectedOp(site))) return false;

    const bool result_used = (record.flags & kAtomicResultUsed) != 0;
    const bool value_zero = (record.flags & kAtomicValueZero) != 0;
    const bool return_ready = (record.flags & kAtomicReturnReady) != 0;
    const bool poll_batch = (record.flags & kAtomicPollBatch) != 0;
    const uint32_t payload = record.flags >> kAtomicRetriesShift;
    if (poll_batch) {
        return AtomicSiteIsPollBatchable(site) && result_used && !value_zero && !return_ready &&
               payload > 0 && record.task_id == -1 && record.function_id == -1;
    }
    if (result_used != AtomicSiteResultUsed(site) || (return_ready && !result_used)) return false;
    if (value_zero && op != static_cast<uint32_t>(AtomicOp::Load)) return false;
    if (payload != 0 && op != static_cast<uint32_t>(AtomicOp::FetchMax)) return false;
    return record.function_id == -1;
}

inline bool ClockRecordSchemaValid(const TraceRecord &record) {
    const bool dependency = (record.flags & kClockAtomicDependency) != 0;
    const bool dependency_applied = (record.flags & kClockAtomicDependencyApplied) != 0;
    return (record.flags & ~(kClockAtomicDependency | kClockAtomicDependencyApplied)) == 0 &&
           (!dependency_applied || dependency) && record.task_id == -1 &&
           record.function_id == -1 && record.auxiliary == 0;
}

// shared PA Case1 不再执行 ordinary-region PrepareMap，但为保持现有 raw
// schema 仍写一条零时长 marker。这里直接在 host 已有 raw 扫描中闭合
// Claim -> Materialize -> PrepareMap -> Submit 身份、顺序与次数，不增加任何
// device 记录字段。private 的 PrepareMap 是真实动作，因此编译为无约束
// validator。
struct SharedPrepareMapTraceValidator {
    bool Observe(const TraceRecord &record) {
#if PTO_FDWIC_SHARED_MAP
        const auto phase = static_cast<TracePhase>(record.phase);
        if (phase == TracePhase::Claim) {
            if (submit_pending_ || record.task_id != next_task_id_) return false;
            submit_pending_ = true;
            materialize_seen_ = false;
            marker_seen_ = false;
            task_id_ = record.task_id;
            function_id_ = record.function_id;
        } else if (phase == TracePhase::Materialize) {
            if (!submit_pending_ || materialize_seen_ || marker_seen_ ||
                record.task_id != task_id_ ||
                record.function_id != function_id_) {
                return false;
            }
            materialize_seen_ = true;
            materialize_end_ = record.end_cycle;
            ++materialize_count_;
        } else if (phase == TracePhase::PrepareMap) {
            if (!submit_pending_ || !materialize_seen_ || marker_seen_ ||
                record.task_id != task_id_ ||
                record.function_id != function_id_ ||
                record.start_cycle != materialize_end_ ||
                record.end_cycle != materialize_end_ ||
                record.flags != 0 || record.task_id < 0) {
                return false;
            }
            const uint32_t kind =
                static_cast<uint32_t>(record.task_id) % kTasksPerBatch;
            if (record.auxiliary != (kind == 0 ? 1U : 0U)) {
                return false;
            }
            marker_seen_ = true;
            ++marker_count_;
        } else if (phase == TracePhase::Submit) {
            if (!submit_pending_ || !materialize_seen_ || !marker_seen_ ||
                record.task_id != task_id_ ||
                record.function_id != function_id_) {
                return false;
            }
            submit_pending_ = false;
            ++next_task_id_;
            ++submit_count_;
        }
#else
        (void)record;
#endif
        return true;
    }

    bool Closed() const {
#if PTO_FDWIC_SHARED_MAP
        return !submit_pending_ &&
               materialize_count_ == marker_count_ &&
               marker_count_ == submit_count_;
#else
        return true;
#endif
    }

    uint32_t MaterializeCount() const {
        return materialize_count_;
    }

    uint32_t MarkerCount() const {
        return marker_count_;
    }

    uint32_t SubmitCount() const {
        return submit_count_;
    }

private:
    bool submit_pending_ = false;
    bool materialize_seen_ = false;
    bool marker_seen_ = false;
    int32_t next_task_id_ = 0;
    int32_t task_id_ = -1;
    int32_t function_id_ = -1;
    uint64_t materialize_end_ = 0;
    uint32_t materialize_count_ = 0;
    uint32_t marker_count_ = 0;
    uint32_t submit_count_ = 0;
};

inline void ExpectedTraceTopology(uint32_t worker, int32_t *block_id, int32_t *lane) {
    if (worker < kAicWorkers) {
        *block_id = static_cast<int32_t>(worker);
        *lane = 0;
        return;
    }
    const uint32_t vector_id = worker - kAicWorkers;
    *block_id = static_cast<int32_t>(vector_id / 2);
    *lane = static_cast<int32_t>(1 + vector_id % 2);
}

template <typename ReadRecords>
inline bool ExportSwimlaneRecords(
    const TraceHeader &header, const std::string &output_path,
    WinnerWorkloadMode workload_mode, const WorkloadCounts &workload_counts,
    const char *workload_pattern, FinalBarrierShape final_barrier_shape,
    bool atomic_trace_enabled, ReadRecords read_records
) {
    if (!ValidateTraceHeader(header, "swimlane export")) return false;
    if (workload_mode != WinnerWorkloadMode::ScalarNop &&
        workload_mode != WinnerWorkloadMode::RealCompute) {
        std::fprintf(stderr, "swimlane export rejected invalid winner workload mode.\n");
        return false;
    }
    const bool real_compute = workload_mode == WinnerWorkloadMode::RealCompute;
    const bool pattern_valid = workload_pattern != nullptr &&
        ((real_compute &&
          (std::strcmp(workload_pattern, "constant") == 0 ||
           std::strcmp(workload_pattern, "layout-diagnostic") == 0)) ||
         (!real_compute && std::strcmp(workload_pattern, "none") == 0));
    if (!pattern_valid) {
        std::fprintf(stderr, "swimlane export rejected invalid winner workload input pattern.\n");
        return false;
    }

    TraceExportSummary producer_summary;
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        const TraceCoreState &core = header.cores[worker];
        int32_t expected_block = -1;
        int32_t expected_lane = -1;
        ExpectedTraceTopology(worker, &expected_block, &expected_lane);
        if (core.core_idx != static_cast<int32_t>(worker) || core.block_id != expected_block ||
            core.lane != expected_lane) {
            std::fprintf(
                stderr,
                "swimlane export rejected worker topology: worker=%u core=%d block=%d/%d lane=%d/%d\n",
                worker, core.core_idx, core.block_id, expected_block, core.lane, expected_lane
            );
            return false;
        }
        producer_summary.records += core.count;
        producer_summary.atomic_calls += core.atomic_calls;
        producer_summary.poll_calls += core.poll_calls;
        producer_summary.poll_batch_records += core.poll_batch_records;
        producer_summary.dropped_records += core.dropped;
        if (!atomic_trace_enabled) {
            if (core.atomic_calls != 0 || core.poll_calls != 0 || core.poll_batch_records != 0) {
                std::fprintf(
                    stderr,
                    "phase-only swimlane worker %u unexpectedly reports atomic counters: calls=%u polls=%u batches=%u\n",
                    worker, core.atomic_calls, core.poll_calls, core.poll_batch_records
                );
                return false;
            }
            continue;
        }
        if (core.poll_calls > core.atomic_calls ||
            (core.poll_calls == 0) != (core.poll_batch_records == 0)) {
            std::fprintf(
                stderr,
                "atomic swimlane worker %u has invalid counters: calls=%u polls=%u batches=%u\n",
                worker, core.atomic_calls, core.poll_calls, core.poll_batch_records
            );
            return false;
        }
        producer_summary.atomic_records +=
            static_cast<uint64_t>(core.atomic_calls) - core.poll_calls + core.poll_batch_records;
    }
    producer_summary.clock_baseline_records = atomic_trace_enabled ? 2ULL * kWorkers : 0;

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
        "{\n\"l2_swimlane_level\":%u,\n"
        "\"metadata\":{\"clock_freq_hz\":%llu,\"num_cores\":%u,"
        "\"trace_schema_version\":%u,\"final_barrier\":\"%s\","
        "\"winner_workload\":{\"mode\":\"%s\","
        "\"counts\":{\"qk\":%u,\"sf\":%u,\"pv\":%u,\"up\":%u},"
        "\"unit\":\"%s\",\"input_pattern\":\"%s\","
        "\"engine_mapping\":%s},\"core_types\":[",
        atomic_trace_enabled ? 4U : 1U,
        static_cast<unsigned long long>(header.frequency_hz), kWorkers, 4U,
        FinalBarrierShapeName(final_barrier_shape),
        workload_mode == WinnerWorkloadMode::RealCompute ? "real-compute" : "scalar-nop",
        workload_counts.qk, workload_counts.sf, workload_counts.pv, workload_counts.up,
        workload_mode == WinnerWorkloadMode::RealCompute
            ? "complete_128x128_engine_pipeline_iteration"
            : "scalar_nop_instruction",
        workload_pattern,
        workload_mode == WinnerWorkloadMode::RealCompute
            ? "{\"qk\":\"cube_matmul\",\"sf\":\"vector_add\","
              "\"pv\":\"cube_matmul\",\"up\":\"vector_mul\"}"
            : "null"
    );
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        std::fprintf(output, "%s\"%s\"", worker == 0 ? "" : ",", worker < kAicWorkers ? "aic" : "aiv");
    }
    // schema-v4 无论是否开启 atomic 都导出 producer summary；phase-only 的
    // atomic/clock 字段为零，离线分析仍可独立证明 records 与 dropped 闭合。
    std::fprintf(
        output,
        "],\"fdwic_summary\":{\"records\":%llu,\"atomic_records\":%llu,"
        "\"clock_baseline_records\":%llu,\"atomic_calls\":%llu,"
        "\"batched_poll_calls\":%llu,\"poll_batch_records\":%llu,"
        "\"dropped_records\":%llu}",
        static_cast<unsigned long long>(producer_summary.records),
        static_cast<unsigned long long>(producer_summary.atomic_records),
        static_cast<unsigned long long>(producer_summary.clock_baseline_records),
        static_cast<unsigned long long>(producer_summary.atomic_calls),
        static_cast<unsigned long long>(producer_summary.poll_calls),
        static_cast<unsigned long long>(producer_summary.poll_batch_records),
        static_cast<unsigned long long>(producer_summary.dropped_records)
    );
    std::fprintf(
        output,
        "},\n\"aicore_tasks\":[],\n\"aicpu_tasks\":[],\n"
        "\"aicpu_scheduler_phases\":[],\n\"aicpu_orchestrator_phases\":[],\n\"fdwic_events\":[\n"
    );
    // fdwic_events 每行固定十列：core、block、lane、task、function、phase、起止周期、flags、aux。

    bool success = true;
    bool first_record = true;
    uint64_t exported_records = 0;
    TraceExportSummary observed_summary;
    std::vector<TraceRecord> scratch(kTraceRecordsPerCore);
    constexpr int32_t kTracePhaseCount = static_cast<int32_t>(TracePhase::Count);
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
        const TraceCoreState &core = header.cores[worker];
        uint64_t core_atomic_calls = 0;
        uint64_t core_poll_calls = 0;
        uint32_t core_atomic_records = 0;
        uint32_t core_poll_batch_records = 0;
        uint32_t core_clock_records = 0;
        uint32_t core_plain_clock_records = 0;
        uint32_t core_dependency_clock_records = 0;
        bool dependency_applied = false;
        bool direct_result_used_return_ready = false;
        bool direct_result_used_source_issue = false;
        SharedPrepareMapTraceValidator prepare_map_validator;
        for (uint32_t index = 0; index < available; ++index) {
            const TraceRecord &record = scratch[index];
            const bool atomic_record = record.phase == static_cast<int32_t>(TracePhase::Atomic);
            const bool claim_record = record.phase == static_cast<int32_t>(TracePhase::Claim);
            const bool clock_record = record.phase == static_cast<int32_t>(TracePhase::ClockBaseline);
            const bool atomic_schema_valid = !atomic_record ||
                AtomicRecordSchemaValid(record, atomic_trace_enabled);
            const bool claim_schema_valid = !claim_record ||
                ((record.flags & ~(kClaimWon | kClaimAttempted)) == 0 &&
                 ((record.flags & kClaimWon) == 0 || (record.flags & kClaimAttempted) != 0) &&
                 record.auxiliary <= 1);
            const bool clock_schema_valid = !clock_record ||
                (atomic_trace_enabled && ClockRecordSchemaValid(record));
            bool record_valid = record.end_cycle >= record.start_cycle && record.phase >= 0 &&
                                record.phase < kTracePhaseCount && record.task_id >= -1 &&
                                record.function_id >= -1 && record.lane == core.lane &&
                                record.block_id == core.block_id &&
                                record.core_idx == core.core_idx && atomic_schema_valid &&
                                claim_schema_valid && clock_schema_valid;
            if (record_valid && !prepare_map_validator.Observe(record)) {
                record_valid = false;
            }
            if (!record_valid) {
                std::fprintf(
                    stderr,
                    "Invalid trace record at worker=%u index=%u: phase=%d lane=%d block=%d core=%d "
                    "start=%llu end=%llu flags=0x%08x aux=%u\n",
                    worker, index, record.phase, record.lane, record.block_id, record.core_idx,
                    static_cast<unsigned long long>(record.start_cycle),
                    static_cast<unsigned long long>(record.end_cycle), record.flags, record.auxiliary
                );
                success = false;
                break;
            }
            if (atomic_record) {
                ++core_atomic_records;
                const uint32_t call_count = AtomicRecordCallCount(record);
                core_atomic_calls += call_count;
                if ((record.flags & kAtomicPollBatch) != 0) {
                    core_poll_calls += call_count;
                    ++core_poll_batch_records;
                } else if ((record.flags & kAtomicResultUsed) != 0) {
                    if ((record.flags & kAtomicReturnReady) != 0) {
                        direct_result_used_return_ready = true;
                    } else {
                        direct_result_used_source_issue = true;
                    }
                }
            } else if (clock_record) {
                ++core_clock_records;
                if ((record.flags & kClockAtomicDependency) != 0) {
                    ++core_dependency_clock_records;
                    dependency_applied = (record.flags & kClockAtomicDependencyApplied) != 0;
                } else {
                    ++core_plain_clock_records;
                }
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
        if (!success) break;
        bool core_closed = true;
        if (atomic_trace_enabled) {
            const uint64_t expected_atomic_records =
                static_cast<uint64_t>(core.atomic_calls) - core.poll_calls + core.poll_batch_records;
            core_closed = core_atomic_records == expected_atomic_records &&
                          core_atomic_calls == core.atomic_calls && core_poll_calls == core.poll_calls &&
                          core_poll_batch_records == core.poll_batch_records && core_clock_records == 2 &&
                          core_plain_clock_records == 1 && core_dependency_clock_records == 1 &&
                          (!dependency_applied || !direct_result_used_source_issue) &&
                          (dependency_applied || !direct_result_used_return_ready);
        } else {
            core_closed = core_atomic_records == 0 && core_atomic_calls == 0 && core_poll_calls == 0 &&
                          core_poll_batch_records == 0 && core_clock_records == 0;
        }
        core_closed &= prepare_map_validator.Closed();
        if (!core_closed) {
            std::fprintf(
                stderr,
                "swimlane closure failed on worker=%u: physical_atomic=%u logical_atomic=%llu/%u "
                "poll_calls=%llu/%u poll_batches=%u/%u clock=%u plain=%u dependency=%u "
                "dependency_applied=%s direct_ready=%s direct_issue=%s "
                "materializes=%u prepare_map_markers=%u submits=%u\n",
                worker, core_atomic_records, static_cast<unsigned long long>(core_atomic_calls),
                core.atomic_calls, static_cast<unsigned long long>(core_poll_calls), core.poll_calls,
                core_poll_batch_records, core.poll_batch_records, core_clock_records,
                core_plain_clock_records, core_dependency_clock_records,
                dependency_applied ? "yes" : "no", direct_result_used_return_ready ? "yes" : "no",
                direct_result_used_source_issue ? "yes" : "no",
                prepare_map_validator.MaterializeCount(),
                prepare_map_validator.MarkerCount(),
                prepare_map_validator.SubmitCount()
            );
            success = false;
            break;
        }
        observed_summary.records += available;
        observed_summary.atomic_records += core_atomic_records;
        observed_summary.clock_baseline_records += core_clock_records;
        observed_summary.atomic_calls += core_atomic_calls;
        observed_summary.poll_calls += core_poll_calls;
        observed_summary.poll_batch_records += core_poll_batch_records;
        observed_summary.dropped_records += core.dropped;
    }
    if (success && !SameTraceSummary(producer_summary, observed_summary)) {
        std::fprintf(
            stderr,
            "swimlane producer/raw summary mismatch: records=%llu/%llu atomic_records=%llu/%llu "
            "atomic_calls=%llu/%llu poll_calls=%llu/%llu poll_batches=%llu/%llu clock=%llu/%llu\n",
            static_cast<unsigned long long>(observed_summary.records),
            static_cast<unsigned long long>(producer_summary.records),
            static_cast<unsigned long long>(observed_summary.atomic_records),
            static_cast<unsigned long long>(producer_summary.atomic_records),
            static_cast<unsigned long long>(observed_summary.atomic_calls),
            static_cast<unsigned long long>(producer_summary.atomic_calls),
            static_cast<unsigned long long>(observed_summary.poll_calls),
            static_cast<unsigned long long>(producer_summary.poll_calls),
            static_cast<unsigned long long>(observed_summary.poll_batch_records),
            static_cast<unsigned long long>(producer_summary.poll_batch_records),
            static_cast<unsigned long long>(observed_summary.clock_baseline_records),
            static_cast<unsigned long long>(producer_summary.clock_baseline_records)
        );
        success = false;
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
    constexpr uint32_t kTracePhaseCount = static_cast<uint32_t>(TracePhase::Count);
    constexpr TracePhase kDetailedPhases[] = {
        TracePhase::EfDrain, TracePhase::Claim, TracePhase::Materialize, TracePhase::Register,
    };
    uint64_t cycles[kWorkers][kTracePhaseCount] = {};
    uint64_t counts[kWorkers][kTracePhaseCount] = {};
    std::vector<uint64_t> task_durations[2][kTasksPerBatch][sizeof(kDetailedPhases) / sizeof(kDetailedPhases[0])];
    std::vector<uint64_t> atomic_durations[2][static_cast<uint32_t>(AtomicSite::Count)];
    uint64_t atomic_return_ready_counts[2][static_cast<uint32_t>(AtomicSite::Count)] = {};
    std::vector<uint64_t> atomic_poll_windows[2][static_cast<uint32_t>(AtomicSite::Count)];
    uint64_t atomic_poll_calls[2][static_cast<uint32_t>(AtomicSite::Count)] = {};
    std::vector<uint64_t> clock_baselines[2];
    std::vector<uint64_t> clock_dependency_baselines[2];
    uint64_t clock_dependency_applied[2] = {};
    std::vector<TraceRecord> scratch(kTraceRecordsPerCore);
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        SharedPrepareMapTraceValidator prepare_map_validator;
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
            if (!prepare_map_validator.Observe(record)) {
                std::fprintf(
                    stderr,
                    "swimlane analysis rejected shared PrepareMap flow at "
                    "worker=%u index=%u task=%d function=%d start=%llu end=%llu "
                    "flags=0x%08x aux=%u\n",
                    worker, index, record.task_id, record.function_id,
                    static_cast<unsigned long long>(record.start_cycle),
                    static_cast<unsigned long long>(record.end_cycle),
                    record.flags, record.auxiliary
                );
                return false;
            }
            const uint32_t phase = static_cast<uint32_t>(record.phase);
            const uint64_t duration = record.end_cycle - record.start_cycle;
            const bool atomic_poll_batch =
                record.phase == static_cast<int32_t>(TracePhase::Atomic) &&
                (record.flags & kAtomicPollBatch) != 0;
            // PollBatch 的 duration 是一次等待 episode 的包络，允许夹着其他直接
            // atomic/调度代码；不能混入“Atomic 单次括号”的累计时间或分位数。
            if (!atomic_poll_batch) {
                cycles[worker][phase] += duration;
                ++counts[worker][phase];
            }
            if (record.phase == static_cast<int32_t>(TracePhase::Atomic) &&
                record.auxiliary < static_cast<uint32_t>(AtomicSite::Count)) {
                const uint32_t role_index =
                    state.results[worker].role == static_cast<uint64_t>(CoreRole::Aic) ? 0U : 1U;
                if (atomic_poll_batch) {
                    atomic_poll_windows[role_index][record.auxiliary].push_back(duration);
                    atomic_poll_calls[role_index][record.auxiliary] += AtomicRecordCallCount(record);
                } else {
                    atomic_durations[role_index][record.auxiliary].push_back(duration);
                    atomic_return_ready_counts[role_index][record.auxiliary] +=
                        (record.flags & kAtomicReturnReady) != 0;
                }
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
        if (!prepare_map_validator.Closed()) {
            std::fprintf(
                stderr,
                "swimlane analysis rejected unclosed shared PrepareMap flow on "
                "worker=%u: materializes=%u markers=%u submits=%u\n",
                worker, prepare_map_validator.MaterializeCount(),
                prepare_map_validator.MarkerCount(),
                prepare_map_validator.SubmitCount()
            );
            return false;
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
    // 等待聚合只报告 episode 数、精确逻辑调用数与包络分布。window 不能除以
    // calls 当作单次 atomic latency，也不能与 Submit 墙钟直接相加。
    for (uint32_t role_index = 0; role_index < 2; ++role_index) {
        for (uint32_t site = 0; site < static_cast<uint32_t>(AtomicSite::Count); ++site) {
            const std::vector<uint64_t> &windows = atomic_poll_windows[role_index][site];
            if (windows.empty()) continue;
            const Uint64Distribution summary = SummarizeUint64(windows);
            std::printf(
                "[TRACE_ATOMIC_POLL] role=%s site=%s op=%s episodes=%zu logical_calls=%llu "
                "window_definition=wait-episode-envelope median_ns=%.1f p95_ns=%llu max_ns=%llu\n",
                role_names[role_index], AtomicSiteName(site),
                AtomicOpName(static_cast<uint32_t>(AtomicSiteOp(static_cast<AtomicSite>(site)))),
                windows.size(), static_cast<unsigned long long>(atomic_poll_calls[role_index][site]),
                summary.median, static_cast<unsigned long long>(summary.p95),
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

inline uint64_t DependencyEdgeSignatureHost(
    uint32_t consumer, uint32_t producer
) {
    uint64_t value =
        (static_cast<uint64_t>(consumer) << 32U) | producer;
    value ^= value >> 30U;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D049BB133111EBULL;
    value ^= value >> 31U;
    return value;
}

inline uint64_t ExpectedPaDependencySignature(uint32_t batches) {
    uint64_t signature = 0;
    for (uint32_t batch = 0; batch < batches; ++batch) {
        const uint32_t alloc = batch * kTasksPerBatch;
        const uint32_t qk = alloc + 1;
        const uint32_t sf = alloc + 2;
        const uint32_t pv = alloc + 3;
        const uint32_t up = alloc + 4;
        // SF<-QK、PV<-SF；UP 的 SF max/sum 去重为 SF 一条，再依赖
        // PV 和本 batch Alloc 建立的 accumulator。BeginPaBatch 后的 Alloc
        // 会替换 orchestration 中三条累计输出引用，不跨 batch 沿用前一 UP。
        signature ^= DependencyEdgeSignatureHost(sf, qk);
        signature ^= DependencyEdgeSignatureHost(pv, sf);
        signature ^= DependencyEdgeSignatureHost(up, sf);
        signature ^= DependencyEdgeSignatureHost(up, pv);
        signature ^= DependencyEdgeSignatureHost(up, alloc);
    }
    return signature;
}

inline uint32_t SharedTensorMapHashHost(uint64_t address) {
    address *= 0x9E3779B97F4A7C15ULL;
    return static_cast<uint32_t>(
        address >> (64 - kMapBucketShift)
    ) & kMapBucketMask;
}

inline void SharedLogicalHashWord(uint64_t *hash, uint64_t value) {
    // 固定按小端字节折叠，不依赖 host struct padding；private/shared 后续
    // 都以 bucket、region 和 producer 的同一字段序列生成可比较签名。
    for (uint32_t byte = 0; byte < 8; ++byte) {
        *hash ^= (value >> (byte * 8U)) & 0xFFU;
        *hash *= 1099511628211ULL;
    }
}

#if PTO_FDWIC_SHARED_MAP
struct SharedTensorMapValidation {
    bool protocol_ok = true;
    uint64_t total_appends = 0;
    uint64_t physical_entries = 0;
    uint64_t logical_entries = 0;
    uint64_t logical_signature = 1469598103934665603ULL;
};

inline SharedTensorMapValidation ValidateSharedTensorMap(
    const SharedTensorMapSidecar &map
) {
    SharedTensorMapValidation validation;
    validation.protocol_ok &= map.committed_tasks.value == 0;
    validation.protocol_ok &= map.reclaim_upto.value == -1;

    // shared fresh Output 已改由 shared_outputs 直接按
    // (producer_task_id, output_slot) 定位；Case1 中的唯一 ordinary
    // output_view 又是 manual_dep。因此 PA Case1 完全绕过 region ring，
    // committed/reclaim 与 bucket/slot 都必须保持初始化状态。
    const SharedRegionPayload zero_payload{};
    for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
        const int64_t head = map.buckets[bucket].head.value;
        const int64_t tail = map.buckets[bucket].tail.value;
        validation.protocol_ok &= head == 0 && tail == 0;
    }
    for (uint32_t slot = 0; slot < kMapCapacity; ++slot) {
        validation.protocol_ok &= map.slots[slot].seq.value == -1;
        validation.protocol_ok &= std::memcmp(
            &map.slots[slot].payload, &zero_payload,
            sizeof(zero_payload)
        ) == 0;
    }
    return validation;
}
#endif

inline uint32_t ExpectedOutputCount(uint32_t task_id) {
    switch (static_cast<TaskKind>(task_id % kTasksPerBatch)) {
        case TaskKind::Alloc: return 3;
        case TaskKind::Qk: return 1;
        case TaskKind::Sf: return 3;
        case TaskKind::Pv: return 1;
        case TaskKind::Up: return 0;
        case TaskKind::Count: return 0;
    }
    return 0;
}

// host_support 只依赖 pa_model，不能反向 include 设备端 pa_frontend；这里保留
// Case1 协议已固定的 descriptor 常量和 dtype 字节数，避免 host 校验引入设备代码。
constexpr uint32_t kHostPaHeads = 16;
constexpr uint32_t kHostPaHeadDim = 128;
constexpr uint32_t kHostPaBlockSize = 128;
constexpr uint32_t kHostPaBlocksPerRequest = 64;
constexpr uint64_t kHostSyntheticOutputBase = 0x600000000ULL;
constexpr uint64_t kHostInvalidTaskId = UINT64_MAX;

inline uint64_t HostElementSize(DataType dtype) {
    switch (dtype) {
        case DataType::Float32:
        case DataType::Int32:
        case DataType::Uint32:
            return 4;
        case DataType::Float16:
        case DataType::Bfloat16:
        case DataType::Int16:
        case DataType::Uint16:
            return 2;
        case DataType::Int8:
        case DataType::Uint8:
        case DataType::Bool:
            return 1;
        case DataType::Int64:
        case DataType::Uint64:
            return 8;
        case DataType::Count:
            return 0;
    }
    return 0;
}

inline uint64_t ExpectedTaskOutputBytes(uint32_t task_id) {
    constexpr uint64_t kOutputBytesByKind[kTasksPerBatch] = {
        10240, 524288, 264192, 8192, 0
    };
    return kOutputBytesByKind[task_id % kTasksPerBatch];
}

inline uint64_t ExpectedCanonicalTaskBase(uint32_t task_id) {
    constexpr uint64_t kTaskOffsetByKind[kTasksPerBatch] = {
        0, 10240, 534528, 798720, 806912
    };
    return static_cast<uint64_t>(task_id / kTasksPerBatch) * 806912ULL +
           kTaskOffsetByKind[task_id % kTasksPerBatch];
}

#if PTO_FDWIC_SHARED_MAP
inline uint64_t ExpectedSharedHeapShardSpan(uint64_t heap_size) {
    const uint64_t raw = heap_size / kSharedHeapShards;
    return raw / kOutputAlignment * kOutputAlignment;
}
#endif

inline TensorDesc ExpectedCanonicalOutputDescriptor(
    uint32_t task_id, uint32_t output_slot
) {
    // canonical 地址保持 private 连续 heap 布局，只服务跨模式逻辑签名；
    // shared 实际 descriptor 由下方 8-shard oracle 独立验证。
    const uint64_t task_base = ExpectedCanonicalTaskBase(task_id);
    uint64_t output_offset = 0;
    uint64_t buffer_size = 0;
    uint32_t ndims = 0;
    DataType dtype = DataType::Float32;
    uint32_t shapes[kMaxTensorDims] = {};
    const TaskKind kind = static_cast<TaskKind>(task_id % kTasksPerBatch);
    if (kind == TaskKind::Alloc) {
        if (output_slot == 0) {
            buffer_size = 8192;
            ndims = 2;
            shapes[0] = kHostPaHeads;
            shapes[1] = kHostPaHeadDim;
        } else if (output_slot == 1 || output_slot == 2) {
            output_offset = output_slot == 1 ? 8192 : 9216;
            buffer_size = 64;
            ndims = 1;
            shapes[0] = kHostPaHeads;
        }
    } else if (kind == TaskKind::Qk && output_slot == 0) {
        buffer_size = 524288;
        ndims = 2;
        shapes[0] = kHostPaHeads;
        shapes[1] = kHostPaBlocksPerRequest * kHostPaBlockSize;
    } else if (kind == TaskKind::Sf) {
        if (output_slot == 0) {
            buffer_size = 262144;
            ndims = 2;
            dtype = DataType::Bfloat16;
            shapes[0] = kHostPaHeads;
            shapes[1] = kHostPaBlocksPerRequest * kHostPaBlockSize;
        } else if (output_slot == 1 || output_slot == 2) {
            output_offset = output_slot == 1 ? 262144 : 263168;
            buffer_size = 64;
            ndims = 1;
            shapes[0] = kHostPaHeads;
        }
    } else if (kind == TaskKind::Pv && output_slot == 0) {
        buffer_size = 8192;
        ndims = 2;
        shapes[0] = kHostPaHeads;
        shapes[1] = kHostPaHeadDim;
    }

    TensorDesc expected{};
    expected.buffer_addr = kSyntheticHeapBase + task_base + output_offset;
    expected.buffer_size = buffer_size;
    expected.owner_task_id = task_id;
    expected.start_offset = 0;
    expected.version = 0;
    expected.ndims = ndims;
    expected.dtype = dtype;
    expected.manual_dep = false;
    expected.is_contiguous = true;
    expected.child_memory = 0;
    uint32_t stride = 1;
    for (int32_t index = static_cast<int32_t>(ndims) - 1; index >= 0; --index) {
        expected.strides[index] = stride;
        stride *= shapes[index];
    }
    expected.extent_elem_cache = stride;
    for (uint32_t index = 0; index < kMaxTensorDims; ++index) {
        expected.shapes[index] = shapes[index];
    }
    return expected;
}

#if PTO_FDWIC_SHARED_MAP
inline TensorDesc ExpectedSharedOutputDescriptorAtBase(
    uint32_t task_id, uint32_t output_slot, uint64_t task_base
) {
    TensorDesc expected =
        ExpectedCanonicalOutputDescriptor(task_id, output_slot);
    const uint64_t canonical_task_base =
        ExpectedCanonicalTaskBase(task_id);
    const uint64_t output_offset =
        expected.buffer_addr - kSyntheticHeapBase - canonical_task_base;
    expected.buffer_addr =
        kSyntheticHeapBase + task_base + output_offset;
    return expected;
}
#endif

inline bool TensorDescFieldsMatch(
    const TensorDesc &actual, const TensorDesc &expected
) {
    if (actual.buffer_addr != expected.buffer_addr ||
        actual.buffer_size != expected.buffer_size ||
        actual.owner_task_id != expected.owner_task_id ||
        actual.start_offset != expected.start_offset ||
        actual.version != expected.version || actual.ndims != expected.ndims ||
        actual.dtype != expected.dtype || actual.manual_dep != expected.manual_dep ||
        actual.is_contiguous != expected.is_contiguous ||
        actual.child_memory != expected.child_memory ||
        actual.extent_elem_cache != expected.extent_elem_cache) {
        return false;
    }
    for (uint32_t index = 0; index < kMaxTensorDims; ++index) {
        if (actual.shapes[index] != expected.shapes[index] ||
            actual.strides[index] != expected.strides[index]) {
            return false;
        }
    }
    return true;
}

#if PTO_FDWIC_SHARED_MAP
struct SharedOutputValidation {
    bool protocol_ok = true;
    uint64_t published_outputs = 0;
    uint64_t allocated_bytes = 0;
    uint64_t shard_bytes[kSharedHeapShards] = {};
};

struct SharedHeapInterval {
    uint64_t begin;
    uint64_t end;
};

inline SharedOutputValidation ValidateSharedOutputs(
    const SharedTensorMapSidecar &map, uint32_t task_count,
    uint64_t heap_size
) {
    SharedOutputValidation validation;
    const TensorDesc zero_tensor{};
    const uint64_t shard_span = ExpectedSharedHeapShardSpan(heap_size);
    std::vector<SharedHeapInterval> intervals[kSharedHeapShards];
    for (uint32_t task_id = 0; task_id < kMaxTasks; ++task_id) {
        const uint32_t expected_count =
            task_id < task_count ? ExpectedOutputCount(task_id) : 0;
        const SharedOutputCell &cell = map.shared_outputs[task_id];
        uint64_t task_base = 0;
        if (expected_count != 0) {
            const uint64_t output_bytes = ExpectedTaskOutputBytes(task_id);
            const uint32_t shard = task_id % kSharedHeapShards;
            const uint64_t shard_begin =
                static_cast<uint64_t>(shard) * shard_span;
            const uint64_t shard_end = shard_begin + shard_span;
            const uint64_t address = cell.tensors[0].buffer_addr;
            const bool address_ok =
                address >= kSyntheticHeapBase &&
                output_bytes != 0 &&
                output_bytes <= shard_span;
            if (address_ok) {
                task_base = address - kSyntheticHeapBase;
            }
            const bool interval_ok =
                address_ok &&
                task_base % kOutputAlignment == 0 &&
                task_base >= shard_begin &&
                task_base <= shard_end - output_bytes;
            validation.protocol_ok &= interval_ok;
            if (interval_ok) {
                intervals[shard].push_back(
                    {task_base, task_base + output_bytes}
                );
            }
        }
        for (uint32_t slot = 0; slot < kSharedOutputMaxPerTask; ++slot) {
            const bool active = slot < expected_count;
            if (!active) {
                validation.protocol_ok &= cell.published[slot].value == -1;
                validation.protocol_ok &= cell.last_writer[slot].value == -1;
                validation.protocol_ok &= std::memcmp(
                    &cell.tensors[slot], &zero_tensor, sizeof(zero_tensor)
                ) == 0;
                continue;
            }
            const TaskKind kind = static_cast<TaskKind>(task_id % kTasksPerBatch);
            const int64_t expected_writer = kind == TaskKind::Alloc
                ? static_cast<int64_t>(task_id + 4)
                : static_cast<int64_t>(task_id);
            validation.protocol_ok &=
                cell.published[slot].value == static_cast<int64_t>(task_id);
            validation.protocol_ok &= cell.last_writer[slot].value == expected_writer;
            validation.protocol_ok &= TensorDescFieldsMatch(
                cell.tensors[slot],
                ExpectedSharedOutputDescriptorAtBase(
                    task_id, slot, task_base
                )
            );
            ++validation.published_outputs;
        }
    }
    for (uint32_t shard = 0; shard < kSharedHeapShards; ++shard) {
        std::sort(
            intervals[shard].begin(), intervals[shard].end(),
            [](const SharedHeapInterval &left, const SharedHeapInterval &right) {
                return left.begin < right.begin;
            }
        );
        uint64_t next =
            static_cast<uint64_t>(shard) * shard_span;
        for (const SharedHeapInterval &interval : intervals[shard]) {
            validation.protocol_ok &= interval.begin == next;
            next = interval.end;
        }
        validation.shard_bytes[shard] =
            next - static_cast<uint64_t>(shard) * shard_span;
        validation.allocated_bytes += validation.shard_bytes[shard];
    }
    return validation;
}
#endif

struct NormalizedWriterEntry {
    uint64_t buffer_addr;
    uint64_t lo;
    uint64_t hi;
    uint32_t producer;
};

inline void AddNormalizedWriter(
    std::vector<NormalizedWriterEntry> buckets[kMapBuckets],
    const TensorDesc &tensor, uint32_t producer
) {
    const uint64_t element_size = HostElementSize(tensor.dtype);
    uint64_t extent = tensor.is_contiguous ? 1 : tensor.extent_elem_cache;
    if (tensor.is_contiguous) {
        for (uint32_t index = 0; index < tensor.ndims; ++index) {
            extent *= tensor.shapes[index];
        }
    }
    const uint64_t lo = tensor.start_offset * element_size;
    const uint64_t hi = (tensor.start_offset + extent) * element_size;
    buckets[SharedTensorMapHashHost(tensor.buffer_addr)].push_back(
        {tensor.buffer_addr, lo, hi, producer}
    );
}

inline TensorDesc ExpectedManualOutputView(uint32_t batch, uint32_t batches) {
    TensorDesc view{};
    view.buffer_addr = kHostSyntheticOutputBase;
    view.buffer_size = static_cast<uint64_t>(batches) * kHostPaHeads * kHostPaHeadDim * 4;
    view.owner_task_id = kHostInvalidTaskId;
    view.start_offset = static_cast<uint64_t>(batch) * kHostPaHeads * kHostPaHeadDim;
    view.version = 0;
    view.ndims = 2;
    view.dtype = DataType::Float32;
    view.manual_dep = true;
    view.is_contiguous = true;
    view.child_memory = 0;
    view.shapes[0] = kHostPaHeads;
    view.shapes[1] = kHostPaHeadDim;
    view.extent_elem_cache = kHostPaHeads * kHostPaHeadDim;
    view.strides[0] = kHostPaHeadDim;
    view.strides[1] = 1;
    return view;
}

inline uint64_t FinishNormalizedWriterSignature(
    std::vector<NormalizedWriterEntry> buckets[kMapBuckets]
) {
    uint64_t signature = 1469598103934665603ULL;
    for (uint32_t bucket = 0; bucket < kMapBuckets; ++bucket) {
        for (const NormalizedWriterEntry &entry : buckets[bucket]) {
            SharedLogicalHashWord(&signature, bucket);
            SharedLogicalHashWord(&signature, entry.buffer_addr);
            SharedLogicalHashWord(&signature, entry.lo);
            SharedLogicalHashWord(&signature, entry.hi);
            SharedLogicalHashWord(&signature, entry.producer);
        }
    }
    return signature;
}

inline uint64_t ExpectedNormalizedWriterSignature(
    uint32_t batches, uint32_t logical_floor
) {
    std::vector<NormalizedWriterEntry> by_bucket[kMapBuckets];
    for (uint32_t batch = 0; batch < batches; ++batch) {
        const uint32_t alloc = batch * kTasksPerBatch;
        const uint32_t up = alloc + 4;
        if (up < logical_floor) {
            continue;
        }
        // RegisterOutputs 的真实顺序是 max、sum、output、manual output_view。
        AddNormalizedWriter(
            by_bucket, ExpectedCanonicalOutputDescriptor(alloc, 2), up
        );
        AddNormalizedWriter(
            by_bucket, ExpectedCanonicalOutputDescriptor(alloc, 1), up
        );
        AddNormalizedWriter(
            by_bucket, ExpectedCanonicalOutputDescriptor(alloc, 0), up
        );
        AddNormalizedWriter(by_bucket, ExpectedManualOutputView(batch, batches), up);
    }
    return FinishNormalizedWriterSignature(by_bucket);
}

#if PTO_FDWIC_SHARED_MAP
inline uint64_t SharedNormalizedWriterSignature(
    const SharedTensorMapSidecar &map, uint32_t batches, uint32_t logical_floor
) {
    std::vector<NormalizedWriterEntry> by_bucket[kMapBuckets];
    for (uint32_t batch = 0; batch < batches; ++batch) {
        const uint32_t alloc = batch * kTasksPerBatch;
        const uint32_t up = alloc + 4;
        if (up < logical_floor) {
            continue;
        }
        const SharedOutputCell &cell = map.shared_outputs[alloc];
        // 实际 shared descriptor 的 8-shard 地址已经由
        // ValidateSharedOutputs 严格校验。跨模式签名只投影同一个业务
        // output 的 canonical(private 连续 heap)地址，不能把物理分片差异
        // 误判成 writer 拓扑差异。
        AddNormalizedWriter(
            by_bucket, ExpectedCanonicalOutputDescriptor(alloc, 2),
            static_cast<uint32_t>(cell.last_writer[2].value)
        );
        AddNormalizedWriter(
            by_bucket, ExpectedCanonicalOutputDescriptor(alloc, 1),
            static_cast<uint32_t>(cell.last_writer[1].value)
        );
        AddNormalizedWriter(
            by_bucket, ExpectedCanonicalOutputDescriptor(alloc, 0),
            static_cast<uint32_t>(cell.last_writer[0].value)
        );
        AddNormalizedWriter(by_bucket, ExpectedManualOutputView(batch, batches), up);
    }
    return FinishNormalizedWriterSignature(by_bucket);
}
#endif

#if PA_BUILD_PERF_CLOCK
inline bool PerfClockObserverFieldsAreZero(const WorkerResult &result) {
    for (uint32_t kind = 0; kind < 4; ++kind) {
        if (result.kernel_cycles[kind] != 0 ||
            result.kernel_min_cycles[kind] != 0 ||
            result.kernel_max_cycles[kind] != 0) {
            return false;
        }
    }
    for (uint32_t phase = 0;
         phase < static_cast<uint32_t>(ProfilePhase::Count); ++phase) {
        if (result.phase_cycles[phase] != 0 ||
            result.phase_calls[phase] != 0) {
            return false;
        }
    }
    return result.atomic_trace_calls == 0 &&
           result.pmu_total_cycles == 0 &&
           result.pmu_scalar_busy == 0 &&
           result.pmu_icache_requests == 0 &&
           result.pmu_icache_misses == 0 &&
           result.pmu_status == 0 &&
           result.pmu_window_ticks == 0 &&
           result.pmu_warm_total_cycles == 0 &&
           result.pmu_warm_window_ticks == 0 &&
           result.pmu_warm_icache_requests == 0 &&
           result.pmu_warm_icache_misses == 0 &&
           result.pmu_vector_busy == 0 &&
           result.pmu_cube_busy == 0 &&
           result.pmu_mte1_busy == 0 &&
           result.pmu_mte2_busy == 0 &&
           result.pmu_mte3_busy == 0 &&
           result.pmu_fix_busy == 0 &&
           result.pmu_build_variant == 0 &&
           result.pmu_phase_id == 0 &&
           result.pmu_phase_calls == 0 &&
           result.pmu_phase_status == 0 &&
           result.pmu_phase_icache_requests == 0 &&
           result.pmu_phase_icache_misses == 0 &&
           result.pmu_shadow_icache_requests == 0 &&
           result.pmu_shadow_icache_misses == 0 &&
           result.startup_barrier_begin == 0 &&
           result.startup_barrier_end == 0 &&
           result.final_barrier_begin == 0 &&
           result.final_barrier_release == 0 &&
           result.final_barrier_end == 0;
}
#endif

inline Metrics Validate(
    const SchedulerState &state, uint32_t run, double host_us, const TraceHeader *trace_header = nullptr
) {
    Metrics metrics;
    // 每个 worker 都回放全部 task。Alloc 由 96 个 worker 全部执行 atomicMax Claim；
    // 其余 kernel task 只有与 active role 匹配的 AIC 或 AIV 参与 Claim。
    const uint32_t batches = state.config.batches;
    const uint32_t task_count = batches * kTasksPerBatch;
    const bool final_barrier_shape_valid =
        state.config.final_barrier_shape <= static_cast<uint32_t>(FinalBarrierShape::ThreeLevel6x4x4);
    const auto final_barrier_shape = static_cast<FinalBarrierShape>(state.config.final_barrier_shape);
    const uint64_t expected_submits = static_cast<uint64_t>(kWorkers) * task_count;
    const uint64_t expected_claims =
        static_cast<uint64_t>(batches) * (kWorkers + kAicWorkers + kAivWorkers + kAicWorkers + kAivWorkers);
    // 上式依次对应 Alloc、QK、SF、PV、UP 的 active worker 数，默认 256 batch 时为 73728。

    // 聚合量分为调度核心计数、kernel 分布、前端操作数和最终状态四组，便于定位语义偏差。
    uint64_t first_submit = UINT64_MAX;
    uint64_t last_submit = 0;
#if !PA_BUILD_PERF_CLOCK
    uint64_t first_startup_begin = UINT64_MAX;
    uint64_t last_startup_end = 0;
    uint64_t first_final_begin = UINT64_MAX;
    uint64_t last_final_release = 0;
    uint64_t last_final_end = 0;
    std::vector<uint64_t> startup_wait_ticks;
    std::vector<uint64_t> final_release_wait_ticks;
    std::vector<uint64_t> post_release_drain_ticks;
#endif
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
#if !PA_BUILD_PERF_CLOCK
    uint64_t kernel_cycles[4] = {};
    uint64_t kernel_min[4] = {};
    uint64_t kernel_max[4] = {};
#endif
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
    uint64_t dependency_signature = 0;
    uint64_t shared_symbol_input_loads = 0;
    uint64_t shared_symbol_inout_commits = 0;
    bool worker_ids[kWorkers] = {};
    uint32_t aic_count = 0;
    uint32_t aiv_count = 0;
    uint32_t winning_workers = 0;
    uint64_t max_worker_wins = 0;
    bool worker_shape_ok = true;
    bool submit_timestamps_ok = true;
    bool lifecycle_timestamps_ok = true;
#if PA_BUILD_PERF_CLOCK
    bool perf_clock_observer_fields_zero = true;
#endif
    bool vend_values_ok = true;
    bool frontend_worker_counts_ok = true;
    bool final_worker_state_ok = true;
    bool worker_checksums_ok = true;
#if !PTO_FDWIC_SHARED_MAP
    uint64_t private_logical_map_signature = 0;
#endif
    bool fanin_worker_counts_ok = true;
    bool frontier_worker_counts_ok = true;
    bool role_kernel_routing_ok = true;
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
    bool compete_first_split_runtime_oracle_ok = true;
    const uint64_t expected_split_task_id_sum =
        static_cast<uint64_t>(task_count) * (task_count - 1U) / 2U;
#endif

    // private 按连续逻辑 heap 重建逐 task prefix；shared 只按 task_id%8
    // 重建每个 shard 的最终字节总量。并发 FetchAdd 后，某 task 获得的
    // task_base 和 aggregate vend prefix 都不再由 task_id 顺序决定。
    uint64_t expected_heap_next = 0;
    bool vend_progress_bounds_ok = true;
    uint32_t first_bad_vend = task_count;
    uint64_t first_bad_vend_minimum = 0;
    uint64_t first_bad_vend_actual = 0;
    std::vector<uint64_t> minimum_vends(task_count);
#if PTO_FDWIC_SHARED_MAP
    uint64_t expected_shared_heap_cursor[kSharedHeapShards] = {};
    const uint64_t shared_heap_shard_span =
        ExpectedSharedHeapShardSpan(state.heap_size);
    bool shared_heap_capacity_ok = shared_heap_shard_span != 0;
#endif
    for (uint32_t task_id = 0; task_id < task_count; ++task_id) {
        const uint64_t output_bytes = ExpectedTaskOutputBytes(task_id);
#if PTO_FDWIC_SHARED_MAP
        const uint32_t shard = task_id % kSharedHeapShards;
        shared_heap_capacity_ok &=
            expected_shared_heap_cursor[shard] <= shared_heap_shard_span &&
            output_bytes <=
                shared_heap_shard_span -
                    std::min(
                        expected_shared_heap_cursor[shard],
                        shared_heap_shard_span
                    );
        expected_shared_heap_cursor[shard] += output_bytes;
        expected_heap_next += output_bytes;
        minimum_vends[task_id] = output_bytes;
#else
        uint64_t task_base = (expected_heap_next + kOutputAlignment - 1) / kOutputAlignment * kOutputAlignment;
        if (output_bytes != 0 && (task_base % state.heap_size) + output_bytes > state.heap_size) {
            task_base = (task_base / state.heap_size + 1) * state.heap_size;
        }
        expected_heap_next = task_base + output_bytes;
        minimum_vends[task_id] = expected_heap_next;
#endif
    }
#if PTO_FDWIC_SHARED_MAP
    bool shared_heap_state_ok = shared_heap_capacity_ok;
    // 当前 Case1 每个 batch 只有一个 block group；没有后继 group，就不应
    // 触发 shared writer-ready gate。host 只在 kernel 完成后读取该状态，
    // 不给设备热路增加任何观察字段或 atomic。
    bool shared_writer_gates_idle = true;
    for (uint32_t task_id = 0; task_id < task_count; ++task_id) {
        shared_writer_gates_idle &=
            state.tasks[task_id].deps_prepared == -1;
    }
    uint64_t actual_shared_cursor_sum = 0;
    uint64_t expected_shared_cursor_sum = 0;
    for (uint32_t shard = 0; shard < kSharedHeapShards; ++shard) {
        const int64_t raw_cursor =
            state.shared_map.shared_heap_cursor[shard].value;
        shared_heap_state_ok &= raw_cursor >= 0;
        const uint64_t actual_cursor =
            raw_cursor < 0 ? 0 : static_cast<uint64_t>(raw_cursor);
        shared_heap_state_ok &=
            actual_cursor == expected_shared_heap_cursor[shard];
        shared_heap_capacity_ok &=
            actual_cursor <= shared_heap_shard_span;
        actual_shared_cursor_sum += actual_cursor;
        expected_shared_cursor_sum += expected_shared_heap_cursor[shard];
    }
    const int64_t raw_shared_vend =
        state.shared_map.shared_heap_vend.value;
    shared_heap_state_ok &= raw_shared_vend >= 0;
    const uint64_t actual_shared_vend =
        raw_shared_vend < 0 ? 0 : static_cast<uint64_t>(raw_shared_vend);
    shared_heap_state_ok &=
        actual_shared_vend == expected_heap_next &&
        actual_shared_cursor_sum == actual_shared_vend &&
        expected_shared_cursor_sum == expected_heap_next;
    shared_heap_state_ok &= shared_heap_capacity_ok;
#endif
    for (uint32_t task_id = 0; task_id < task_count; ++task_id) {
        // kernel 可以晚于后续 Submit 完成，故 task vend 可以高于本 task
        // reserve 后的 prefix。private 使用确定 task-order prefix；shared
        // 的并发 prefix 只要求覆盖本 task 自身 reserve 且不越过最终 vend。
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
    // private ring 仍保留 heap window 内的四类 writer。shared fresh
    // Output 已迁出 region ring，因此它的 region 摘要和 sequencer 均保持
    // 初值。expected_map_floor 只供跨模式规范化 writer 签名投影使用，
    // 不能解释成 shared sidecar 实际发生过 reclaim。
#if !PTO_FDWIC_SHARED_MAP
    const uint64_t expected_private_map_live =
        static_cast<uint64_t>(kPaCase1MapEntriesPerBatch) *
        std::min<uint32_t>(batches, kPaCase1MaxLiveMapBatches);
#endif
    const uint64_t expected_map_floor = task_count > kHeapWindow + 1 ? task_count - kHeapWindow - 1 : 0;
#if PTO_FDWIC_SHARED_MAP
    const SharedTensorMapValidation shared_map_validation =
        ValidateSharedTensorMap(state.shared_map);
    const SharedOutputValidation shared_output_validation =
        ValidateSharedOutputs(state.shared_map, task_count, state.heap_size);
    bool shared_output_heap_layout_ok =
        shared_output_validation.protocol_ok &&
        shared_output_validation.allocated_bytes == expected_heap_next &&
        shared_output_validation.allocated_bytes == actual_shared_vend;
    for (uint32_t shard = 0; shard < kSharedHeapShards; ++shard) {
        const int64_t raw_cursor =
            state.shared_map.shared_heap_cursor[shard].value;
        shared_output_heap_layout_ok &=
            shared_output_validation.shard_bytes[shard] ==
                expected_shared_heap_cursor[shard] &&
            raw_cursor >= 0 &&
            shared_output_validation.shard_bytes[shard] ==
                static_cast<uint64_t>(raw_cursor);
    }
    const uint64_t shared_normalized_writer_signature =
        shared_output_heap_layout_ok
            ? SharedNormalizedWriterSignature(
                  state.shared_map, batches,
                  static_cast<uint32_t>(expected_map_floor)
              )
            : 0;
#endif
    const uint64_t expected_normalized_writer_signature =
        ExpectedNormalizedWriterSignature(
            batches, static_cast<uint32_t>(expected_map_floor)
        );

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
#if PA_BUILD_PERF_CLOCK
        submit_timestamps_ok &= result.submit_end > result.submit_begin;
        submit_timestamps_ok &= result.finish_cycle == result.submit_end;
        lifecycle_timestamps_ok &=
            result.startup_barrier_begin == 0 &&
            result.startup_barrier_end == 0 &&
            result.final_barrier_begin == 0 &&
            result.final_barrier_release == 0 &&
            result.final_barrier_end == 0;
        perf_clock_observer_fields_zero &=
            PerfClockObserverFieldsAreZero(result);
#else
        submit_timestamps_ok &= result.submit_end >= result.submit_begin;
        submit_timestamps_ok &= result.finish_cycle >= result.submit_end;
        lifecycle_timestamps_ok &= result.startup_barrier_begin != 0;
        lifecycle_timestamps_ok &= result.startup_barrier_end >= result.startup_barrier_begin;
        lifecycle_timestamps_ok &= result.submit_begin >= result.startup_barrier_end;
        lifecycle_timestamps_ok &= result.final_barrier_begin >= result.submit_end;
        lifecycle_timestamps_ok &= result.final_barrier_release >= result.final_barrier_begin;
        lifecycle_timestamps_ok &= result.final_barrier_end >= result.final_barrier_release;
        lifecycle_timestamps_ok &= result.finish_cycle >= result.final_barrier_end;
#endif
        dependency_signature ^= result.dependency_signature;
        shared_symbol_input_loads += result.shared_symbol_input_loads;
        shared_symbol_inout_commits += result.shared_symbol_inout_commits;
        first_submit = std::min(first_submit, result.submit_begin);
        last_submit = std::max(last_submit, result.submit_end);
#if !PA_BUILD_PERF_CLOCK
        first_startup_begin = std::min(first_startup_begin, result.startup_barrier_begin);
        last_startup_end = std::max(last_startup_end, result.startup_barrier_end);
        first_final_begin = std::min(first_final_begin, result.final_barrier_begin);
        last_final_release = std::max(last_final_release, result.final_barrier_release);
        last_final_end = std::max(last_final_end, result.final_barrier_end);
        startup_wait_ticks.push_back(result.startup_barrier_end - result.startup_barrier_begin);
        final_release_wait_ticks.push_back(result.final_barrier_release - result.final_barrier_begin);
        post_release_drain_ticks.push_back(result.final_barrier_end - result.final_barrier_release);
#endif
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
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
        const CoreRole expected_role = index < kAicWorkers ? CoreRole::Aic : CoreRole::Aiv;
        compete_first_split_runtime_oracle_ok &= result.worker_id == index;
        compete_first_split_runtime_oracle_ok &=
            result.compete_first_split_caller_state_address != 0;
        compete_first_split_runtime_oracle_ok &=
            result.compete_first_split_finish_state_address ==
                result.compete_first_split_caller_state_address;
        compete_first_split_runtime_oracle_ok &=
            result.compete_first_split_finish_calls == task_count;
        compete_first_split_runtime_oracle_ok &=
            result.compete_first_split_protocol_errors == 0;
        compete_first_split_runtime_oracle_ok &=
            result.compete_first_split_state_cookie ==
                (kCompeteFirstSplitStateCookieBase ^ static_cast<uint64_t>(index) ^
                 (static_cast<uint64_t>(static_cast<uint32_t>(expected_role)) << 32U));
        compete_first_split_runtime_oracle_ok &=
            result.compete_first_split_task_id_sum == expected_split_task_id_sum;
        compete_first_split_runtime_oracle_ok &=
            result.compete_first_split_owner_worker_id == index;
        compete_first_split_runtime_oracle_ok &=
            result.compete_first_split_reserved == 0;
#endif
#if PTO_FDWIC_SHARED_MAP
        // shared 只让 Alloc 保留全员三个静态 Output 参数；四个重构参 task
        // 与 Materialize 都必须由本核实际 wins[] 精确推导，不能只核对
        // 跨核总数而漏掉 loser 的意外构参。
        const uint64_t alloc_wins = result.wins[static_cast<uint32_t>(TaskKind::Alloc)];
        const uint64_t qk_wins = result.wins[static_cast<uint32_t>(TaskKind::Qk)];
        const uint64_t sf_wins = result.wins[static_cast<uint32_t>(TaskKind::Sf)];
        const uint64_t pv_wins = result.wins[static_cast<uint32_t>(TaskKind::Pv)];
        const uint64_t up_wins = result.wins[static_cast<uint32_t>(TaskKind::Up)];
        frontend_worker_counts_ok &= result.context_reads == batches;
        frontend_worker_counts_ok &=
            result.views_created == qk_wins + up_wins;
        frontend_worker_counts_ok &=
            result.dynamic_create_infos == qk_wins + sf_wins;
        frontend_worker_counts_ok &=
            result.arg_resets == qk_wins + sf_wins + pv_wins + up_wins;
        frontend_worker_counts_ok &=
            result.tensor_args_added ==
                static_cast<uint64_t>(batches) * 3 +
                4 * (qk_wins + sf_wins + pv_wins) + 7 * up_wins;
        frontend_worker_counts_ok &=
            result.scalar_args_added ==
                2 * qk_wins + 3 * sf_wins + 2 * pv_wins + 2 * up_wins;
        frontend_worker_counts_ok &=
            result.materialized_outputs ==
                alloc_wins * 3 + qk_wins + sf_wins * 3 + pv_wins;
        frontend_worker_counts_ok &= result.map_inserts == 0;
        // shared 的权威进度是 sidecar cursor/vend。worker.heap_next 只保存
        // 该 worker 最近一次获胜时观察到的并发 aggregate prefix；不同
        // winner 的 FetchAdd 顺序不由 task_id 决定，因此不能再拿确定的
        // task-order prefix 集合核对。纯 loser 仍必须保持 0。
        const uint64_t nonzero_output_wins =
            alloc_wins + qk_wins + sf_wins + pv_wins;
        const uint64_t own_reserved_bytes =
            alloc_wins * ExpectedTaskOutputBytes(
                static_cast<uint32_t>(TaskKind::Alloc)
            ) +
            qk_wins * ExpectedTaskOutputBytes(
                static_cast<uint32_t>(TaskKind::Qk)
            ) +
            sf_wins * ExpectedTaskOutputBytes(
                static_cast<uint32_t>(TaskKind::Sf)
            ) +
            pv_wins * ExpectedTaskOutputBytes(
                static_cast<uint32_t>(TaskKind::Pv)
            );
        final_worker_state_ok &=
            result.final_heap_next <= expected_heap_next &&
            result.final_heap_next >= own_reserved_bytes &&
            (result.final_heap_next == 0 ||
             result.final_heap_next % kOutputAlignment == 0) &&
            (result.claim_wins != 0 || result.final_heap_next == 0) &&
            (nonzero_output_wins == 0 || result.final_heap_next != 0);
#else
        frontend_worker_counts_ok &= result.context_reads == batches;
        frontend_worker_counts_ok &= result.views_created == static_cast<uint64_t>(batches) * 2;
        frontend_worker_counts_ok &= result.dynamic_create_infos == static_cast<uint64_t>(batches) * 2;
        frontend_worker_counts_ok &= result.arg_resets == static_cast<uint64_t>(batches) * 4;
        frontend_worker_counts_ok &= result.tensor_args_added == static_cast<uint64_t>(batches) * 22;
        frontend_worker_counts_ok &= result.scalar_args_added == static_cast<uint64_t>(batches) * 9;
        frontend_worker_counts_ok &= result.materialized_outputs == static_cast<uint64_t>(batches) * 8;
        frontend_worker_counts_ok &= result.map_inserts == static_cast<uint64_t>(batches) * 4;
        final_worker_state_ok &= result.final_heap_next == expected_heap_next;
#endif
        final_worker_state_ok &= result.map_high_water ==
#if PTO_FDWIC_SHARED_MAP
            0;
#else
            expected_private_map_live;
#endif
        final_worker_state_ok &= result.map_live_entries ==
#if PTO_FDWIC_SHARED_MAP
            0;
#else
            expected_private_map_live;
#endif
        final_worker_state_ok &= result.map_alive_floor ==
#if PTO_FDWIC_SHARED_MAP
            0;
#else
            expected_map_floor;
#endif
        final_worker_state_ok &= result.map_cleaned_upto ==
#if PTO_FDWIC_SHARED_MAP
            0;
#else
            expected_map_floor;
#endif
#if PTO_FDWIC_SHARED_MAP
        // shared 的权威签名由 host 对唯一 sidecar 逐槽生成，worker 不重复
        // 扫描共享 GM，以免把验证 DCCI 成本加入 kernel 生命周期。
        worker_checksums_ok &= result.checksum == 0;
#else
        if (index == 0) {
            private_logical_map_signature = result.checksum;
        } else {
            worker_checksums_ok &=
                result.checksum ==
                private_logical_map_signature;
        }
        worker_checksums_ok &= result.checksum != 0;
#endif
        if (result.role == static_cast<uint32_t>(CoreRole::Aic)) {
            role_kernel_routing_ok &= result.kernel_counts[1] == 0 && result.kernel_counts[3] == 0;
        } else if (result.role == static_cast<uint32_t>(CoreRole::Aiv)) {
            role_kernel_routing_ok &= result.kernel_counts[0] == 0 && result.kernel_counts[2] == 0;
        } else {
            role_kernel_routing_ok = false;
        }
#if PTO_FDWIC_SHARED_MAP
        // shared no-wrap heap 不消费连续 frontier；每核完成只发布 vend/flag。
        // 三个计数必须保持零，防止 private reclaim helping 悄悄回到热路径。
        frontier_worker_counts_ok &=
            result.frontier_initial_loads == 0 &&
            result.frontier_updates == 0 &&
            result.frontier_terminal_loads == 0;
#else
        const uint64_t worker_kernel_completions = result.kernel_counts[0] + result.kernel_counts[1] +
                                                   result.kernel_counts[2] + result.kernel_counts[3];
        const uint64_t worker_completions = result.wins[0] + worker_kernel_completions;
        frontier_worker_counts_ok &= result.frontier_initial_loads == worker_completions;
        frontier_worker_counts_ok &= result.frontier_terminal_loads == result.frontier_initial_loads;
#endif
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
#if !PA_BUILD_PERF_CLOCK
            kernel_cycles[kind] += result.kernel_cycles[kind];
            if (result.kernel_min_cycles[kind] != 0 &&
                (kernel_min[kind] == 0 || result.kernel_min_cycles[kind] < kernel_min[kind])) {
                kernel_min[kind] = result.kernel_min_cycles[kind];
            }
            kernel_max[kind] = std::max(kernel_max[kind], result.kernel_max_cycles[kind]);
#endif
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
#if PTO_FDWIC_SHARED_MAP
        // 无全局 turn 时，零输出 UP 可能在任一非零 reserve 前观察到
        // aggregate vend=0。shared 不使用该值做 heap reclaim，因此 oracle
        // 允许 0；有实际 output reserve 的 task 仍必须发布非零 vend。
        vend_values_ok &=
            ExpectedTaskOutputBytes(task_id) == 0 ||
            state.tasks[task_id].vend != 0;
#else
        vend_values_ok &= state.tasks[task_id].vend != 0;
#endif
        vend_values_ok &= state.tasks[task_id].vend % kOutputAlignment == 0;
    }
    const uint64_t kernel_total = kernel_counts[0] + kernel_counts[1] + kernel_counts[2] + kernel_counts[3];
    const uint64_t placement_total = placements[0] + placements[1] + placements[2];
    const uint64_t fanin_loads = fanin_ready_loads + fanin_not_ready_loads;
    const uint64_t frontier_flag_loads = frontier_updates + frontier_terminal_loads;

    // 第一组断言覆盖参与者拓扑、Claim/winner、completion 和最终 drain 等调度主协议。
    Expect(aic_count == kAicWorkers && aiv_count == kAivWorkers, "participant topology is 32 AIC + 64 AIV", &metrics);
    Expect(
        worker_shape_ok,
        kCompiledTensorMapMode == TensorMapBuildMode::Private
            ? "all 96 worker markers and private rings are valid"
            : "all 96 worker markers and shared-map clients are valid",
        &metrics
    );
#if defined(PA_COMPETE_FIRST_SPLIT_FINISH)
    Expect(
        compete_first_split_runtime_oracle_ok,
        "compete-first caller/finish share one role-specific block-local state",
        &metrics
    );
#endif
    Expect(submit_timestamps_ok, "all Submit timing markers are valid", &metrics);
#if PA_BUILD_PERF_CLOCK
    Expect(
        lifecycle_timestamps_ok,
        "perf-clock lifecycle-only timing fields stay zero",
        &metrics
    );
    Expect(
        perf_clock_observer_fields_zero,
        "perf-clock excludes phase, atomic-trace, PMU, and kernel timing observations",
        &metrics
    );
    Expect(
        state.config.trace_enabled == 0 &&
            state.config.trace_base == 0 &&
            state.config.trace_records_per_core == 0 &&
            state.config.profile_phases == 0,
        "perf-clock runtime trace and phase controls stay disabled",
        &metrics
    );
#else
    Expect(lifecycle_timestamps_ok, "all lifecycle timing markers are valid", &metrics);
#endif
    Expect(final_barrier_shape_valid, "final barrier selector is valid", &metrics);
    const bool flat_final_barrier = final_barrier_shape == FinalBarrierShape::Flat;
    Expect(
        state.started_count.value == static_cast<int64_t>(kWorkers),
        "startup barrier remains flat and reaches all workers", &metrics
    );
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
    Expect(
        heap_guards ==
#if PTO_FDWIC_SHARED_MAP
            0,
#else
            static_cast<uint64_t>(batches) * 4,
#endif
        kCompiledTensorMapMode == TensorMapBuildMode::Private
            ? "private heap guard count matches output winners"
            : "shared no-wrap heap needs no private ring guard",
        &metrics
    );
    Expect(
        fanin_worker_counts_ok && fanin_ready_loads >= fanin_edges &&
            fanin_ready_loads - fanin_edges <= 2 * fanin_not_ready_loads,
        "fanin ready/failure load classification is complete", &metrics
    );
#if PTO_FDWIC_SHARED_MAP
    Expect(
        frontier_worker_counts_ok && frontier_initial_loads == 0,
        "shared no-wrap completion performs no frontier loads", &metrics
    );
    Expect(
        frontier_terminal_loads == 0 && frontier_updates == 0,
        "shared no-wrap completion performs no frontier helping", &metrics
    );
#else
    Expect(
        frontier_worker_counts_ok && frontier_initial_loads == task_count,
        "private frontier initial loads match completed tasks", &metrics
    );
    Expect(
        frontier_terminal_loads == task_count && frontier_updates >= task_count,
        "private frontier ready/update/terminal load identity is exact", &metrics
    );
#endif
    Expect(duplicates == 0, "completion flags are published once", &metrics);
    Expect(ready_flags == task_count, "all task flags are ready", &metrics);
    Expect(
        vend_values_ok,
        kCompiledTensorMapMode == TensorMapBuildMode::Private
            ? "all published vend values are nonzero and aligned"
            : "shared task vends are aligned and nonzero for output reservations",
        &metrics
    );
    Expect(
        vend_progress_bounds_ok,
        kCompiledTensorMapMode == TensorMapBuildMode::Private
            ? "every task vend is within private worker heap progress bounds"
            : "every task vend is within shared aggregate heap progress bounds",
        &metrics
    );
#if PTO_FDWIC_SHARED_MAP
    Expect(
        state.frontier.value == -1,
        "shared no-wrap frontier remains at its initial value", &metrics
    );
    Expect(
        shared_writer_gates_idle,
        "single-group PA leaves every shared writer-ready gate untouched",
        &metrics
    );
#else
    Expect(
        state.frontier.value == static_cast<int64_t>(task_count) - 1,
        "private frontier reaches the final task", &metrics
    );
#endif
    Expect(
        state.replay_done.value == (flat_final_barrier ? static_cast<int64_t>(kWorkers) : 0) &&
            FinalBarrierStateMatches(state.final_barrier, final_barrier_shape),
        "final barrier counters match selected tree", &metrics
    );
    Expect(state.fatal.value == 0, "fatal remains clear", &metrics);
    Expect(placement_total == kernel_total, "EfDrain + RingBp + final placement covers every kernel", &metrics);
    // joint_polls 是为未来 BlockWon 模拟预留的结果字段，当前调度路径没有递增点；
    // 此断言只确认现有输出保持零，不能单独证明 active_count>=2 分支不可达。
    Expect(joint_polls == 0, "single-lane PA performs no BlockWon polling", &metrics);
    // 第二组断言锁定 scalar 前端工作量，防止编译器优化或后续改动悄悄删掉 PA 模拟步骤。
    Expect(
        frontend_worker_counts_ok,
        kCompiledTensorMapMode == TensorMapBuildMode::Private
            ? "every private worker replays the exact eager frontend counts"
            : "shared winner-derived heavy args and materialize counts are exact",
        &metrics
    );
    const uint64_t expected_global_map_inserts =
#if PTO_FDWIC_SHARED_MAP
        0;
#else
        static_cast<uint64_t>(kWorkers) * batches *
        kPaCase1MapEntriesPerBatch;
#endif
    const bool global_frontend_counts_ok =
#if PTO_FDWIC_SHARED_MAP
        context_reads == static_cast<uint64_t>(kWorkers) * batches &&
        views_created == static_cast<uint64_t>(batches) * 2 &&
        dynamic_create_infos == static_cast<uint64_t>(batches) * 2 &&
        arg_resets == static_cast<uint64_t>(batches) * 4 &&
        tensor_args_added ==
            static_cast<uint64_t>(batches) *
                (static_cast<uint64_t>(kWorkers) * 3 + 19) &&
        scalar_args_added ==
            static_cast<uint64_t>(batches) * 9 &&
        materialized_outputs == static_cast<uint64_t>(batches) * 8 &&
        map_inserts == expected_global_map_inserts;
#else
        context_reads == static_cast<uint64_t>(kWorkers) * batches &&
        views_created == static_cast<uint64_t>(kWorkers) * batches * 2 &&
        dynamic_create_infos == static_cast<uint64_t>(kWorkers) * batches * 2 &&
        arg_resets == static_cast<uint64_t>(kWorkers) * batches * 4 &&
        tensor_args_added == static_cast<uint64_t>(kWorkers) * batches * 22 &&
        scalar_args_added == static_cast<uint64_t>(kWorkers) * batches * 9 &&
        materialized_outputs == static_cast<uint64_t>(kWorkers) * batches * 8 &&
        map_inserts == expected_global_map_inserts;
#endif
    Expect(
        global_frontend_counts_ok,
        "global PA frontend operation totals are exact", &metrics
    );
    Expect(
        map_lookups ==
#if PTO_FDWIC_SHARED_MAP
            0 &&
#else
            static_cast<uint64_t>(batches) * 14 &&
#endif
            slot_tensor_copies == static_cast<uint64_t>(batches) * 19 &&
            slot_scalar_copies == static_cast<uint64_t>(batches) * 9 &&
            fanin_edges == static_cast<uint64_t>(batches) * 5,
        "winner-only TensorMap/symbol, slot-copy, and fanin totals are exact", &metrics
    );
    Expect(
        shared_symbol_input_loads ==
#if PTO_FDWIC_SHARED_MAP
            static_cast<uint64_t>(batches) * 5 &&
#else
            0 &&
#endif
        shared_symbol_inout_commits ==
#if PTO_FDWIC_SHARED_MAP
            static_cast<uint64_t>(batches) * 3,
#else
            0,
#endif
        "shared symbol INPUT-load / INOUT-writer-commit totals are exact", &metrics
    );
#if PTO_FDWIC_SHARED_MAP
    std::printf(
        "[SHARED_SYMBOL] published_outputs=%llu input_loads=%llu "
        "inout_writer_commits=%llu\n",
        static_cast<unsigned long long>(
            shared_output_validation.published_outputs
        ),
        static_cast<unsigned long long>(shared_symbol_input_loads),
        static_cast<unsigned long long>(shared_symbol_inout_commits)
    );
#endif
    const uint64_t expected_dependency_signature =
        ExpectedPaDependencySignature(batches);
    Expect(
        dependency_signature == expected_dependency_signature,
        "fanin dependency-edge signature matches PA Case1",
        &metrics
    );
    std::printf(
        "[DEPENDENCY] mode=%s edges=%llu signature=%016llx\n",
        kCompiledTensorMapMode == TensorMapBuildMode::Private
            ? "private"
            : "shared",
        static_cast<unsigned long long>(fanin_edges),
        static_cast<unsigned long long>(dependency_signature)
    );
    Expect(
        final_worker_state_ok,
        kCompiledTensorMapMode == TensorMapBuildMode::Private
            ? "every private worker final heap and TensorMap state is exact"
            : "shared worker heap snapshots are legal and TensorMap summaries are exact",
        &metrics
    );
#if PTO_FDWIC_SHARED_MAP
    Expect(
        shared_heap_state_ok,
        "shared heap cursors, vend sum, and shard capacity are exact",
        &metrics
    );
    std::printf(
        "[SHARED_HEAP] shard_span=%llu cursors=[",
        static_cast<unsigned long long>(shared_heap_shard_span)
    );
    for (uint32_t shard = 0; shard < kSharedHeapShards; ++shard) {
        std::printf(
            "%s%lld", shard == 0 ? "" : ",",
            static_cast<long long>(
                state.shared_map.shared_heap_cursor[shard].value
            )
        );
    }
    std::printf(
        "] cursor_sum=%llu vend=%lld expected_vend=%llu capacity_ok=%u\n",
        static_cast<unsigned long long>(actual_shared_cursor_sum),
        static_cast<long long>(state.shared_map.shared_heap_vend.value),
        static_cast<unsigned long long>(expected_heap_next),
        shared_heap_capacity_ok ? 1U : 0U
    );
    Expect(
        shared_map_validation.protocol_ok &&
            shared_map_validation.total_appends == 0 &&
            shared_map_validation.physical_entries == 0 &&
            shared_map_validation.logical_entries == 0 &&
            shared_map_validation.logical_signature == 1469598103934665603ULL,
        "shared PA Case1 bypass leaves region sequencer untouched",
        &metrics
    );
    Expect(
        shared_output_heap_layout_ok &&
            shared_output_validation.published_outputs ==
                static_cast<uint64_t>(batches) * 8,
        "shared fresh-output descriptors form exact non-overlapping shard coverage",
        &metrics
    );
    Expect(
        shared_output_heap_layout_ok &&
            shared_normalized_writer_signature ==
            expected_normalized_writer_signature,
        "shared symbol projection matches canonical normalized writer signature",
        &metrics
    );
    std::printf(
        "[TENSORMAP] mode=shared committed=%lld reclaim_upto=%lld "
        "region_appends=%llu region_physical=%llu region_logical=%llu "
        "region_raw_signature=%016llx normalized_writer_signature=%016llx "
        "published_outputs=%llu normalized_projection_floor=%llu\n",
        static_cast<long long>(
            state.shared_map.committed_tasks.value
        ),
        static_cast<long long>(state.shared_map.reclaim_upto.value),
        static_cast<unsigned long long>(
            shared_map_validation.total_appends
        ),
        static_cast<unsigned long long>(
            shared_map_validation.physical_entries
        ),
        static_cast<unsigned long long>(
            shared_map_validation.logical_entries
        ),
        static_cast<unsigned long long>(
            shared_map_validation.logical_signature
        ),
        static_cast<unsigned long long>(
            shared_normalized_writer_signature
        ),
        static_cast<unsigned long long>(
            shared_output_validation.published_outputs
        ),
        static_cast<unsigned long long>(expected_map_floor)
    );
#else
    Expect(
        private_logical_map_signature == expected_normalized_writer_signature,
        "private raw TensorMap checksum matches canonical normalized writer signature",
        &metrics
    );
    std::printf(
        "[TENSORMAP] mode=private logical_entries=%llu logical_floor=%llu "
        "region_raw_signature=%016llx normalized_writer_signature=%016llx\n",
        static_cast<unsigned long long>(expected_private_map_live),
        static_cast<unsigned long long>(expected_map_floor),
        static_cast<unsigned long long>(
            private_logical_map_signature
        ),
        static_cast<unsigned long long>(
            expected_normalized_writer_signature
        )
    );
#endif
    Expect(
        worker_checksums_ok,
        "logical TensorMap signature publication is consistent",
        &metrics
    );

    // private 三类 Claim cursor 均为 production-prefix 四分片。S4.14b
    // shared Vector 启用 sidecar 的全部八条物理线；Cube/Alloc 保持
    // 不变。逐 task 重新推导每条物理 cursor 的最终高水位。
    int64_t expected_cube[kCursorShards] = {-1, -1, -1, -1};
#if PTO_FDWIC_SHARED_MAP
    int64_t expected_vector[kSharedVectorCursorCapacity] = {
        -1, -1, -1, -1, -1, -1, -1, -1
    };
#else
    int64_t expected_vector[kCursorShards] = {-1, -1, -1, -1};
#endif
    int64_t expected_alloc[kCursorShards] = {-1, -1, -1, -1};
    for (uint32_t task_id = 0; task_id < task_count; ++task_id) {
        const TaskKind kind = static_cast<TaskKind>(task_id % kTasksPerBatch);
        if (kind == TaskKind::Alloc) {
            expected_alloc[task_id % kCursorShards] = task_id;
        } else if (kind == TaskKind::Qk || kind == TaskKind::Pv) {
            expected_cube[task_id % kCursorShards] = task_id;
        } else {
#if PTO_FDWIC_SHARED_MAP
            expected_vector[task_id % kSharedVectorCursorShards] =
                task_id;
#else
            expected_vector[task_id % kCursorShards] = task_id;
#endif
        }
    }
    bool cursors_ok = true;
    for (uint32_t shard = 0; shard < kCursorShards; ++shard) {
        cursors_ok &= state.cube_cursor[shard].value == expected_cube[shard];
#if PTO_FDWIC_SHARED_MAP
        // shared Vector 不应再触碰旧 production-prefix vector cursor。
        cursors_ok &= state.vector_cursor[shard].value == -1;
#else
        cursors_ok &= state.vector_cursor[shard].value == expected_vector[shard];
#endif
        cursors_ok &= state.alloc_cursor[shard].value == expected_alloc[shard];
    }
#if PTO_FDWIC_SHARED_MAP
    for (uint32_t shard = 0; shard < kSharedVectorCursorCapacity; ++shard) {
        cursors_ok &=
            state.shared_map.shared_vector_cursor[shard].value ==
                expected_vector[shard];
    }
#endif
    Expect(cursors_ok, "all sharded Claim cursors reach their exact final task", &metrics);

    if (state.config.profile_phases != 0) {
        // profile 开关关闭时这些字段允许保持零，避免把可选诊断本身变成语义门禁。
        Expect(
            phase_calls[static_cast<uint32_t>(ProfilePhase::Claim)] == expected_submits &&
                phase_calls[static_cast<uint32_t>(ProfilePhase::EfDrain)] == expected_submits &&
                phase_calls[static_cast<uint32_t>(ProfilePhase::WaitForSlot)] ==
                    static_cast<uint64_t>(batches) * 4 &&
                phase_calls[static_cast<uint32_t>(ProfilePhase::HeapGuard)] ==
#if PTO_FDWIC_SHARED_MAP
                    0,
#else
                    static_cast<uint64_t>(batches) * 4,
#endif
            kCompiledTensorMapMode == TensorMapBuildMode::Private
                ? "private profile calls match Claim/EfDrain/WaitForSlot/HeapGuard"
                : "shared profile calls match Claim/EfDrain/WaitForSlot without private HeapGuard",
            &metrics
        );
    }

    if (state.config.trace_enabled != 0) {
        // 固定阶段记录数加上动态等待记录数，应与所有 worker 的 header count 精确相等。
        bool trace_shape_ok = trace_header != nullptr;
        uint64_t trace_records = 0;
        uint64_t trace_dropped = 0;
        uint64_t physical_atomic_records = 0;
        uint64_t batched_poll_calls = 0;
        uint64_t poll_batch_records = 0;
        bool per_worker_trace_counts_ok = true;
        if (trace_header != nullptr) {
            trace_shape_ok &= trace_header->magic == 0x4653574cU;
            trace_shape_ok &= trace_header->version == 4;
            trace_shape_ok &= trace_header->num_cores == kWorkers;
            trace_shape_ok &= trace_header->records_per_core == kTraceRecordsPerCore;
            trace_shape_ok &= trace_header->frequency_hz == kSystemCounterHz;
            for (uint32_t worker = 0; worker < kWorkers; ++worker) {
                const TraceCoreState &core = trace_header->cores[worker];
                trace_records += core.count;
                trace_dropped += core.dropped;
                trace_shape_ok &= core.count <= kTraceRecordsPerCore;
                int32_t expected_block = -1;
                int32_t expected_lane = -1;
                ExpectedTraceTopology(worker, &expected_block, &expected_lane);
                trace_shape_ok &= core.core_idx == static_cast<int32_t>(worker);
                trace_shape_ok &= core.block_id == expected_block;
                trace_shape_ok &= core.lane == expected_lane;
                const WorkerResult &result = state.results[worker];
                const uint64_t worker_kernels = result.kernel_counts[0] + result.kernel_counts[1] +
                                                result.kernel_counts[2] + result.kernel_counts[3];
                uint64_t worker_physical_atomic = 0;
                if ((state.config.trace_enabled & kTraceAtomicsEnabled) != 0) {
                    trace_shape_ok &= core.atomic_calls == result.atomic_trace_calls;
                    trace_shape_ok &= core.poll_calls <= core.atomic_calls;
                    trace_shape_ok &= (core.poll_calls == 0) == (core.poll_batch_records == 0);
                    worker_physical_atomic =
                        static_cast<uint64_t>(core.atomic_calls) - core.poll_calls + core.poll_batch_records;
                    physical_atomic_records += worker_physical_atomic;
                    batched_poll_calls += core.poll_calls;
                    poll_batch_records += core.poll_batch_records;
                } else {
                    trace_shape_ok &= core.atomic_calls == 0 && core.poll_calls == 0 &&
                                      core.poll_batch_records == 0;
                }
                const uint64_t worker_expected =
                    6 * result.submits + 2 * result.claim_wins - result.wins[0] +
                    2 * worker_kernels + result.wait_events[0] + result.wait_events[1] + 2 +
                    (((state.config.trace_enabled & kTraceAtomicsEnabled) != 0)
                         ? worker_physical_atomic + 2
                         : 0);
                per_worker_trace_counts_ok &= core.count == worker_expected;
            }
        }
        const uint64_t expected_trace_records =
            static_cast<uint64_t>(batches) * (static_cast<uint64_t>(kWorkers) * 30 + 17) +
            trace_wait_records + 2 * kWorkers +
            (((state.config.trace_enabled & kTraceAtomicsEnabled) != 0)
                 ? physical_atomic_records + 2 * kWorkers
                 : 0);
        // 每 batch 的 96*30 是六条每 Submit 固定记录；shared 的
        // PrepareMap 是零时长结构 marker，不代表 region 业务。17 条是
        // winner/Fanin/Kernel/Commit 记录。loser 不再写额外零时长
        // winner marker；两个父 span 再各核固定增加 2 条；
        // RingBp 等真实等待按运行时次数额外加入。
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
            "[ATOMIC_TRACE] enabled=%s logical_calls=%llu physical_records=%llu "
            "batched_poll_calls=%llu poll_batch_records=%llu "
            "closure=physical=logical-batched+batch_records\n",
            (state.config.trace_enabled & kTraceAtomicsEnabled) != 0 ? "yes" : "no",
            static_cast<unsigned long long>(atomic_trace_calls),
            static_cast<unsigned long long>(physical_atomic_records),
            static_cast<unsigned long long>(batched_poll_calls),
            static_cast<unsigned long long>(poll_batch_records)
        );
    }

    if (first_submit != UINT64_MAX && last_submit >= first_submit) {
        // 性能口径只覆盖最早 Submit.begin 到最晚 Submit.end，不含启动屏障、最终 drain 和 host 同步。
        metrics.submit_span_us = static_cast<double>(last_submit - first_submit) / 1000.0;
    }
#if PA_BUILD_PERF_CLOCK
    std::printf(
        "[PERF-CLOCK] run=%u global_start_tick=%llu global_end_tick=%llu "
        "global_span_ticks=%llu scope=first-submit-begin-to-last-submit-end\n",
        run,
        static_cast<unsigned long long>(first_submit),
        static_cast<unsigned long long>(last_submit),
        static_cast<unsigned long long>(
            first_submit == UINT64_MAX || last_submit < first_submit
                ? 0
                : last_submit - first_submit
        )
    );
#else
    if (first_startup_begin != UINT64_MAX && last_startup_end >= first_startup_begin &&
        first_final_begin != UINT64_MAX && last_final_release >= first_final_begin &&
        last_final_end >= first_final_begin && last_final_end >= first_startup_begin) {
        metrics.startup_barrier_span_us = static_cast<double>(last_startup_end - first_startup_begin) / 1000.0;
        metrics.final_barrier_span_us = static_cast<double>(last_final_release - first_final_begin) / 1000.0;
        metrics.final_drain_span_us = static_cast<double>(last_final_end - first_final_begin) / 1000.0;
        metrics.lifecycle_span_us = static_cast<double>(last_final_end - first_startup_begin) / 1000.0;
    }
    const Uint64Distribution startup_wait = SummarizeUint64(startup_wait_ticks);
    const Uint64Distribution final_release_wait = SummarizeUint64(final_release_wait_ticks);
    const Uint64Distribution post_release_drain = SummarizeUint64(post_release_drain_ticks);
    std::printf(
        "[LIFECYCLE] run=%u final_shape=%s startup_span_us=%.3f final_barrier_span_us=%.3f "
        "final_drain_span_us=%.3f lifecycle_span_us=%.3f "
        "worker_startup_wait_median_us=%.3f worker_startup_wait_p95_us=%.3f "
        "worker_final_wait_median_us=%.3f worker_final_wait_p95_us=%.3f "
        "worker_post_release_drain_median_us=%.3f worker_post_release_drain_p95_us=%.3f\n",
        run, FinalBarrierShapeName(final_barrier_shape), metrics.startup_barrier_span_us,
        metrics.final_barrier_span_us, metrics.final_drain_span_us, metrics.lifecycle_span_us,
        startup_wait.median / 1000.0, static_cast<double>(startup_wait.p95) / 1000.0,
        final_release_wait.median / 1000.0, static_cast<double>(final_release_wait.p95) / 1000.0,
        post_release_drain.median / 1000.0, static_cast<double>(post_release_drain.p95) / 1000.0
    );
#endif
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
#if !PA_BUILD_PERF_CLOCK
    const uint32_t targets[] = {kTargetQkTicks, kTargetSfTicks, kTargetPvTicks, kTargetUpTicks};
#endif
    for (uint32_t kind = 0; kind < 4; ++kind) {
#if PA_BUILD_PERF_CLOCK
        std::printf(
            "[KERNEL] %-2s count=%llu timing=disabled-in-perf-clock\n",
            kernel_names[kind],
            static_cast<unsigned long long>(kernel_counts[kind])
        );
#else
        const double mean =
            kernel_counts[kind] == 0 ? 0.0 : static_cast<double>(kernel_cycles[kind]) / kernel_counts[kind];
        std::printf(
            "[KERNEL] %-2s count=%llu mean_us=%.3f min_us=%.3f max_us=%.3f target_us=%.3f\n", kernel_names[kind],
            static_cast<unsigned long long>(kernel_counts[kind]), mean / 1000.0, kernel_min[kind] / 1000.0,
            kernel_max[kind] / 1000.0, targets[kind] / 1000.0
        );
#endif
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
#if PTO_FDWIC_SHARED_MAP
        std::printf(
            "[FAILURE_STATE] fatal=%d frontier=%lld first_not_ready=%u first_bad_vend=%u "
            "vend_minimum=%llu vend_actual=%llu shared_heap_cursors="
            "[%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld] shared_heap_vend=%lld "
            "shared_heap_shard_span=%llu shared_heap_capacity_ok=%d "
            "worker_submits_min=%llu worker_submits_max=%llu incomplete_workers=%u "
            "final_occupied_workers=%u max_final_occupied=%llu\n",
            state.fatal.value, static_cast<long long>(state.frontier.value),
            first_not_ready, first_bad_vend,
            static_cast<unsigned long long>(first_bad_vend_minimum),
            static_cast<unsigned long long>(first_bad_vend_actual),
            static_cast<long long>(state.shared_map.shared_heap_cursor[0].value),
            static_cast<long long>(state.shared_map.shared_heap_cursor[1].value),
            static_cast<long long>(state.shared_map.shared_heap_cursor[2].value),
            static_cast<long long>(state.shared_map.shared_heap_cursor[3].value),
            static_cast<long long>(state.shared_map.shared_heap_cursor[4].value),
            static_cast<long long>(state.shared_map.shared_heap_cursor[5].value),
            static_cast<long long>(state.shared_map.shared_heap_cursor[6].value),
            static_cast<long long>(state.shared_map.shared_heap_cursor[7].value),
            static_cast<long long>(state.shared_map.shared_heap_vend.value),
            static_cast<unsigned long long>(shared_heap_shard_span),
            shared_heap_capacity_ok ? 1 : 0,
            static_cast<unsigned long long>(min_worker_submits),
            static_cast<unsigned long long>(max_worker_submits),
            incomplete_workers, occupied_workers,
            static_cast<unsigned long long>(max_final_occupied)
        );
#else
        const int64_t retire =
            state.frontier.value - static_cast<int64_t>(kHeapWindow);
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
#endif
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
        "device=%u batches=%u tasks=%u workers=%u runs=%u tensormap=%s "
        "nops=%u,%u,%u,%u state_bytes=%zu "
        "final_barrier=%s swimlane=%s trace_atomics=%s trace_bytes=%zu\n", options.device,
        options.batches, options.batches * kTasksPerBatch, kWorkers, options.runs,
        kCompiledTensorMapMode == TensorMapBuildMode::Private ? "private" : "shared",
        options.nops.qk, options.nops.sf,
        options.nops.pv, options.nops.up, sizeof(SchedulerState),
        FinalBarrierShapeName(options.final_barrier_shape), options.trace_enabled ? "on" : "off",
        options.trace_atomics ? "on" : "off",
        options.trace_enabled ? kTraceBytes : 0
    );
    if (!options.swimlane_json.empty()) {
        std::printf("swimlane_json=%s\n", options.swimlane_json.c_str());
    }
}

}  // namespace pa_scheduler::host

#endif  // PA_SCHEDULER_COMMON_HOST_SUPPORT_H
