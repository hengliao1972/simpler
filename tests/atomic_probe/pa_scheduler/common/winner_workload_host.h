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

#ifndef PA_SCHEDULER_COMMON_WINNER_WORKLOAD_HOST_H
#define PA_SCHEDULER_COMMON_WINNER_WORKLOAD_HOST_H

#include "host_support.h"
#include "winner_workload.h"

#include <cstdio>
#include <string>
#include <vector>

namespace pa_scheduler::host {

// winner 负载参数在通用 benchmark parser 前单独剥离，CCEC 可在其后继续剥离
// PMU 参数；AscendC/CPU 则直接把剩余 argv 交给 ParseOptions。这样不把后端
// 私有功能塞入公共 PA 参数结构，也不会复制三套互斥规则。
struct WinnerWorkloadOptions {
    WinnerWorkloadMode mode = WinnerWorkloadMode::ScalarNop;
    WorkloadCounts repeats = winner_workload::kDefaultRealComputeCounts;
    bool counts_explicit = false;
    bool nop_override_explicit = false;
};

inline const char *WinnerWorkloadModeName(WinnerWorkloadMode mode) {
    switch (mode) {
    case WinnerWorkloadMode::ScalarNop:
        return "scalar-nop";
    case WinnerWorkloadMode::RealCompute:
        return "real-compute";
    }
    return "invalid";
}

inline bool ParseWorkloadCounts(const char *raw, WorkloadCounts *counts) {
    unsigned int qk = 0;
    unsigned int sf = 0;
    unsigned int pv = 0;
    unsigned int up = 0;
    char tail = '\0';
    if (std::sscanf(raw, "%u,%u,%u,%u%c", &qk, &sf, &pv, &up, &tail) != 4) return false;
    const uint32_t maximum = winner_workload::kMaxRealComputeCount;
    if (qk == 0 || sf == 0 || pv == 0 || up == 0 ||
        qk > maximum || sf > maximum || pv > maximum || up > maximum) {
        return false;
    }
    *counts = WorkloadCounts{qk, sf, pv, up};
    return true;
}

inline bool ParseWinnerWorkloadOptions(
    int argc, char **argv, WinnerWorkloadOptions *workload, std::vector<char *> *remaining_argv
) {
    bool mode_seen = false;
    bool count_seen = false;
    remaining_argv->clear();
    remaining_argv->push_back(argv[0]);
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--nop-count" || argument == "--nop-counts") {
            workload->nop_override_explicit = true;
        }
        if (argument != "--winner-workload" && argument != "--real-compute-count" &&
            argument != "--real-compute-counts") {
            remaining_argv->push_back(argv[index]);
            continue;
        }
        if (index + 1 >= argc) {
            std::fprintf(stderr, "Missing value after %s\n", argument.c_str());
            return false;
        }
        const char *value = argv[++index];
        if (argument == "--winner-workload") {
            if (mode_seen) {
                std::fprintf(stderr, "Specify --winner-workload only once.\n");
                return false;
            }
            const std::string name = value;
            if (name == "scalar-nop") {
                workload->mode = WinnerWorkloadMode::ScalarNop;
            } else if (name == "real-compute") {
                workload->mode = WinnerWorkloadMode::RealCompute;
            } else {
                std::fprintf(
                    stderr,
                    "Invalid --winner-workload value: %s (expected scalar-nop|real-compute)\n",
                    value
                );
                return false;
            }
            mode_seen = true;
            continue;
        }
        if (count_seen) {
            std::fprintf(stderr, "Specify only one real-compute count override.\n");
            return false;
        }
        if (argument == "--real-compute-count") {
            uint32_t count = 0;
            if (!ParseUint(value, 1, winner_workload::kMaxRealComputeCount, &count)) {
                std::fprintf(stderr, "Invalid --real-compute-count value: %s\n", value);
                return false;
            }
            workload->repeats = WorkloadCounts{count, count, count, count};
        } else if (!ParseWorkloadCounts(value, &workload->repeats)) {
            std::fprintf(stderr, "Invalid --real-compute-counts value: %s\n", value);
            return false;
        }
        count_seen = true;
        workload->counts_explicit = true;
    }
    return true;
}

inline bool ValidateWinnerWorkloadOptions(const WinnerWorkloadOptions &workload) {
    if (workload.mode == WinnerWorkloadMode::RealCompute) {
        if (workload.nop_override_explicit) {
            std::fprintf(
                stderr,
                "--winner-workload real-compute cannot be combined with --nop-count or --nop-counts.\n"
            );
            return false;
        }
        return true;
    }
    if (workload.counts_explicit) {
        std::fprintf(
            stderr,
            "--real-compute-count(s) requires --winner-workload real-compute.\n"
        );
        return false;
    }
    return true;
}

inline void ConfigureWinnerWorkload(
    SchedulerState *state, const WinnerWorkloadOptions &workload, const void *workspace_device
) {
    state->winner_workload.mode = static_cast<uint32_t>(workload.mode);
    state->winner_workload.version = kWinnerWorkloadConfigVersion;
    state->winner_workload.repeats = workload.repeats;
    state->winner_workload.workspace_base = reinterpret_cast<uint64_t>(workspace_device);
    state->winner_workload.workspace_bytes =
        workload.mode == WinnerWorkloadMode::RealCompute ? winner_workload::kWorkspaceBytes : 0;
}

inline void InitializeWinnerWorkloadBuffers(
    std::vector<float> *workspace_image, std::vector<float> *workspace_outputs
) {
    using namespace winner_workload;
    workspace_image->assign(kWorkspaceTiles * kTileElements, kOutputSentinel);
    std::fill_n(workspace_image->begin(), kTileElements, kInputAValue);
    std::fill_n(workspace_image->begin() + kTileElements, kTileElements, kInputBValue);
    workspace_outputs->resize(static_cast<size_t>(kOutputTiles) * kTileElements);
}

inline const char *TaskKindName(TaskKind kind) {
    switch (kind) {
    case TaskKind::Qk:
        return "QK";
    case TaskKind::Sf:
        return "SF";
    case TaskKind::Pv:
        return "PV";
    case TaskKind::Up:
        return "UP";
    default:
        return "invalid";
    }
}

inline bool ValidateRealComputeOutputs(
    const SchedulerState &state, const std::vector<float> &outputs, uint32_t run
) {
    using namespace winner_workload;
    const size_t expected_elements = static_cast<size_t>(kOutputTiles) * kTileElements;
    if (outputs.size() != expected_elements) {
        std::fprintf(
            stderr, "[ASSERT] real-compute output buffer size matches workspace layout FAIL\n"
        );
        return false;
    }

    uint32_t active_tiles = 0;
    uint32_t inactive_tiles = 0;
    for (uint32_t worker = 0; worker < kWorkers; ++worker) {
        const WorkerResult &result = state.results[worker];
        const bool aic = result.role == static_cast<uint32_t>(CoreRole::Aic);
        const TaskKind kinds[2] = {
            aic ? TaskKind::Qk : TaskKind::Sf,
            aic ? TaskKind::Pv : TaskKind::Up,
        };
        for (uint32_t kind_slot = 0; kind_slot < 2; ++kind_slot) {
            const TaskKind kind = kinds[kind_slot];
            const uint32_t kernel_index = static_cast<uint32_t>(kind) - 1;
            const bool active = result.kernel_counts[kernel_index] != 0;
            const float expected = active
                ? (aic ? kExpectedAicValue : (kind == TaskKind::Sf ? kExpectedSfValue : kExpectedUpValue))
                : kOutputSentinel;
            const size_t tile_index =
                static_cast<size_t>(worker) * kOutputTilesPerWorker + kind_slot;
            const size_t begin = tile_index * kTileElements;
            for (size_t element = 0; element < kTileElements; ++element) {
                if (outputs[begin + element] == expected) continue;
                std::fprintf(
                    stderr,
                    "[REAL-COMPUTE-FAIL] run=%u worker=%u kind=%s element=%zu "
                    "expected=%.1f actual=%.9g\n",
                    run, worker, TaskKindName(kind), element, expected,
                    static_cast<double>(outputs[begin + element])
                );
                std::fprintf(
                    stderr,
                    "[ASSERT] real-compute output tiles match role-specific engine results FAIL\n"
                );
                return false;
            }
            active_tiles += active ? 1U : 0U;
            inactive_tiles += active ? 0U : 1U;
        }
    }
    const bool passed = active_tiles != 0 && active_tiles + inactive_tiles == kOutputTiles;
    std::printf(
        "[ASSERT] %-48s %s (active_tiles=%u inactive_sentinel_tiles=%u)\n",
        "real-compute output tiles match role-specific engine results",
        passed ? "PASS" : "FAIL", active_tiles, inactive_tiles
    );
    return passed;
}

inline void PrintWinnerWorkloadConfig(
    const WinnerWorkloadOptions &workload, const NopCounts &nops
) {
    if (workload.mode == WinnerWorkloadMode::RealCompute) {
        std::printf(
            "[WINNER-WORKLOAD] mode=real-compute counts=%u,%u,%u,%u "
            "unit=complete_128x128_engine_pipeline_iteration workspace_bytes=%zu\n",
            workload.repeats.qk, workload.repeats.sf, workload.repeats.pv,
            workload.repeats.up, winner_workload::kWorkspaceBytes
        );
        return;
    }
    std::printf(
        "[WINNER-WORKLOAD] mode=scalar-nop counts=%u,%u,%u,%u "
        "unit=scalar_nop_instruction workspace_bytes=0\n",
        nops.qk, nops.sf, nops.pv, nops.up
    );
}

}  // namespace pa_scheduler::host

#endif  // PA_SCHEDULER_COMMON_WINNER_WORKLOAD_HOST_H
