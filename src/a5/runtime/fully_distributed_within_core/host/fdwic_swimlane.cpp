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

#include "runtime.h"

#include <inttypes.h>
#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "common/platform_config.h"
#include "common/unified_log.h"
#include "dist_engine/common/swimlane_types.h"

namespace {

const char *phase_name(int32_t phase) {
    switch (static_cast<FdwicSwimlanePhase>(phase)) {
    case FdwicSwimlanePhase::Kernel:
        return "Kernel";
    case FdwicSwimlanePhase::Alloc:
        return "Alloc";
    case FdwicSwimlanePhase::Build:
        return "Build";
    case FdwicSwimlanePhase::DrainWon:
        return "DrainWon";
    case FdwicSwimlanePhase::Replay:
        return "Replay";
    case FdwicSwimlanePhase::RingBp:
        return "RingBp";
    case FdwicSwimlanePhase::EfDrain:
        return "EfDrain";
    case FdwicSwimlanePhase::Commit:
        return "Commit";
    case FdwicSwimlanePhase::Submit:
        return "Submit";
    case FdwicSwimlanePhase::Materialize:
        return "Materialize";
    case FdwicSwimlanePhase::PrepareMap:
        return "PrepareMap";
    case FdwicSwimlanePhase::Claim:
        return "Claim";
    case FdwicSwimlanePhase::Fanin:
        return "Fanin";
    case FdwicSwimlanePhase::Register:
        return "Register";
    }
    return "Unknown";
}

const char *core_type_name(uint32_t core_idx) { return (core_idx % 3 == 0) ? "aic" : "aiv"; }

std::string output_path(const Runtime *runtime) {
    std::string base = runtime->fdwic_swimlane_output_prefix_;
    if (base.empty()) base = ".";
    mkdir(base.c_str(), 0755);
    return base + "/l2_swimlane_records.json";
}

bool should_print_trace_export() {
#if defined(SIMPLER_PLATFORM_NAME)
    return std::strcmp(SIMPLER_PLATFORM_NAME, "a5sim") == 0;
#else
    return false;
#endif
}

}  // namespace

extern "C" int fdwic_swimlane_host_init(Runtime *runtime, int num_cores, int enabled, const char *output_prefix) {
    if (runtime == nullptr) return -1;
    runtime->dist.swimlane_enabled = 0;
    runtime->dist.swimlane_base = 0;
    runtime->dist.swimlane_records_per_core = 0;
    runtime->fdwic_swimlane_host_shadow_ = nullptr;
    runtime->fdwic_swimlane_dev_base_ = 0;
    runtime->fdwic_swimlane_bytes_ = 0;
    runtime->fdwic_swimlane_num_cores_ = 0;
    runtime->fdwic_swimlane_records_per_core_ = 0;
    runtime->fdwic_swimlane_output_prefix_[0] = '\0';
    if (!enabled) return 0;
    if (num_cores <= 0 || num_cores > 108) return -1;
    if (runtime->host_api.device_malloc == nullptr || runtime->host_api.device_free == nullptr ||
        runtime->host_api.copy_to_device == nullptr || runtime->host_api.copy_from_device == nullptr) {
        return -1;
    }

    const uint32_t records_per_core = kFdwicSwimlaneDefaultRecordsPerCore;
    const uint64_t bytes =
        sizeof(FdwicSwimlaneHeader) + static_cast<uint64_t>(num_cores) * records_per_core * sizeof(FdwicSwimlaneRecord);
    void *host_shadow = std::malloc(static_cast<size_t>(bytes));
    if (host_shadow == nullptr) return -1;
    std::memset(host_shadow, 0, static_cast<size_t>(bytes));
    FdwicSwimlaneHeader *header = reinterpret_cast<FdwicSwimlaneHeader *>(host_shadow);
    header->magic = kFdwicSwimlaneMagic;
    header->version = kFdwicSwimlaneVersion;
    header->num_cores = static_cast<uint32_t>(num_cores);
    header->records_per_core = records_per_core;
    header->freq_hz = PLATFORM_PROF_SYS_CNT_FREQ;

    void *dev = runtime->host_api.device_malloc(static_cast<size_t>(bytes));
    if (dev == nullptr) {
        std::free(host_shadow);
        return -1;
    }
    if (runtime->host_api.copy_to_device(dev, host_shadow, static_cast<size_t>(bytes)) != 0) {
        runtime->host_api.device_free(dev);
        std::free(host_shadow);
        return -1;
    }

    runtime->fdwic_swimlane_host_shadow_ = host_shadow;
    runtime->fdwic_swimlane_dev_base_ = reinterpret_cast<uint64_t>(dev);
    runtime->fdwic_swimlane_bytes_ = bytes;
    runtime->fdwic_swimlane_num_cores_ = static_cast<uint32_t>(num_cores);
    runtime->fdwic_swimlane_records_per_core_ = records_per_core;
    if (output_prefix != nullptr) {
        std::strncpy(
            runtime->fdwic_swimlane_output_prefix_, output_prefix, sizeof(runtime->fdwic_swimlane_output_prefix_) - 1
        );
        runtime->fdwic_swimlane_output_prefix_[sizeof(runtime->fdwic_swimlane_output_prefix_) - 1] = '\0';
    }
    runtime->dist.swimlane_base = runtime->fdwic_swimlane_dev_base_;
    runtime->dist.swimlane_records_per_core = records_per_core;
    runtime->dist.swimlane_enabled = 1;
    return 1;
}

extern "C" int fdwic_swimlane_host_export(Runtime *runtime) {
    if (runtime == nullptr || runtime->fdwic_swimlane_host_shadow_ == nullptr ||
        runtime->fdwic_swimlane_dev_base_ == 0) {
        return 0;
    }
    void *dev = reinterpret_cast<void *>(runtime->fdwic_swimlane_dev_base_);
    if (runtime->host_api.copy_from_device(
            runtime->fdwic_swimlane_host_shadow_, dev, static_cast<size_t>(runtime->fdwic_swimlane_bytes_)
        ) != 0) {
        LOG_ERROR("fdwic swimlane D2H copy failed");
        return -1;
    }

    FdwicSwimlaneHeader *header = reinterpret_cast<FdwicSwimlaneHeader *>(runtime->fdwic_swimlane_host_shadow_);
    if (header->magic != kFdwicSwimlaneMagic || header->version != kFdwicSwimlaneVersion) {
        LOG_ERROR("fdwic swimlane header mismatch");
        return -1;
    }
    auto *records = reinterpret_cast<FdwicSwimlaneRecord *>(
        static_cast<uint8_t *>(runtime->fdwic_swimlane_host_shadow_) + sizeof(FdwicSwimlaneHeader)
    );
    std::ofstream out(output_path(runtime));
    if (!out.is_open()) return -1;
    out << "{\n";
    out << "  \"l2_swimlane_level\": 1,\n";
    out << "  \"metadata\": {\n";
    out << "    \"clock_freq_hz\": " << header->freq_hz << ",\n";
    out << "    \"num_cores\": " << header->num_cores << ",\n";
    out << "    \"core_types\": [";
    for (uint32_t c = 0; c < header->num_cores; c++) {
        if (c > 0) out << ", ";
        out << "\"" << core_type_name(c) << "\"";
    }
    out << "]\n";
    out << "  },\n";
    out << "  \"aicore_tasks\": [],\n";
    out << "  \"aicpu_tasks\": [],\n";
    out << "  \"aicpu_scheduler_phases\": [],\n";
    out << "  \"aicpu_orchestrator_phases\": [],\n";
    out << "  \"fdwic_events\": [";
    bool first = true;
    for (uint32_t c = 0; c < header->num_cores; c++) {
        const uint32_t count =
            header->cores[c].count < header->records_per_core ? header->cores[c].count : header->records_per_core;
        for (uint32_t i = 0; i < count; i++) {
            const FdwicSwimlaneRecord &r = records[static_cast<uint64_t>(c) * header->records_per_core + i];
            if (r.end_cycle < r.start_cycle) continue;
            if (!first) out << ",";
            out << "\n    [" << r.core_idx << ", " << r.block_id << ", " << r.lane << ", " << r.task_id << ", "
                << r.func_id << ", \"" << phase_name(r.phase) << "\", " << r.start_cycle << ", " << r.end_cycle << ", "
                << r.flags << ", " << r.aux << "]";
            first = false;
        }
    }
    if (!first) out << "\n  ";
    out << "]\n}\n";
    if (should_print_trace_export()) {
        const std::string path = output_path(runtime);
        LOG_INFO_V0("fdwic swimlane trace written to %s", path.c_str());
    }
    return 0;
}

extern "C" void fdwic_swimlane_host_finalize(Runtime *runtime) {
    if (runtime == nullptr) return;
    if (runtime->fdwic_swimlane_dev_base_ != 0 && runtime->host_api.device_free != nullptr) {
        runtime->host_api.device_free(reinterpret_cast<void *>(runtime->fdwic_swimlane_dev_base_));
    }
    if (runtime->fdwic_swimlane_host_shadow_ != nullptr) {
        std::free(runtime->fdwic_swimlane_host_shadow_);
    }
    runtime->fdwic_swimlane_host_shadow_ = nullptr;
    runtime->fdwic_swimlane_dev_base_ = 0;
    runtime->fdwic_swimlane_bytes_ = 0;
    runtime->dist.swimlane_enabled = 0;
    runtime->dist.swimlane_base = 0;
}
