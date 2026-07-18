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

#include "../common/host_support.h"
#include "../common/winner_workload_host.h"
#include "pmu_owner_host.h"
#include "pmu_probe.h"

#include "acl/acl.h"
#include "driver/ascend_hal.h"
#include "runtime/rt.h"

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

bool CheckAcl(aclError error, const char *label) {
    if (error == ACL_SUCCESS) return true;
    std::fprintf(stderr, "ACL error %d: %s\n", static_cast<int>(error), label);
    return false;
}

bool CheckRt(rtError_t error, const char *label) {
    if (error == RT_ERROR_NONE) return true;
    std::fprintf(stderr, "RT error %d: %s\n", static_cast<int>(error), label);
    return false;
}

class ScopedAclDeviceAllocation {
public:
    ScopedAclDeviceAllocation() = default;
    ScopedAclDeviceAllocation(const ScopedAclDeviceAllocation &) = delete;
    ScopedAclDeviceAllocation &operator=(const ScopedAclDeviceAllocation &) = delete;

    ~ScopedAclDeviceAllocation() {
        // 早退路径没有机会汇入末尾 cleanup；这里只负责尽力释放本类新增的
        // real-compute workspace。正常路径会先 Release，再检查 aclrtFree 返回值。
        if (pointer_ != nullptr) (void)aclrtFree(pointer_);
    }

    void **Address() { return &pointer_; }
    void *Get() const { return pointer_; }

    void *Release() {
        void *pointer = pointer_;
        pointer_ = nullptr;
        return pointer;
    }

private:
    void *pointer_ = nullptr;
};

std::vector<char> ReadBinary(const std::string &path) {
    // ELF 整体保存在 vector 中直到 runtime 卸载完成，保证 rtDevBinary_t.data 在整个注册生命周期内有效。
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0) return {};
    std::vector<char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(data.data(), size)) return {};
    return data;
}

struct PmuOptions {
    pa_scheduler::ccec_pmu::WindowMode mode = pa_scheduler::ccec_pmu::WindowMode::Off;
    uint32_t scalar_nops = 100000;
    uint32_t icache_trials = 64;
    std::string json_path;
};

using pa_scheduler::host::ConfigureWinnerWorkload;
using pa_scheduler::host::ParseWinnerWorkloadOptions;
using pa_scheduler::host::ValidateRealComputeOutputs;
using pa_scheduler::host::ValidateWinnerWorkloadOptions;
using pa_scheduler::host::WinnerWorkloadModeName;
using pa_scheduler::host::WinnerWorkloadOptions;

const char *PmuModeName(pa_scheduler::ccec_pmu::WindowMode mode) {
    switch (mode) {
    case pa_scheduler::ccec_pmu::WindowMode::Off:
        return "off";
    case pa_scheduler::ccec_pmu::WindowMode::Empty:
        return "empty";
    case pa_scheduler::ccec_pmu::WindowMode::Scalar:
        return "scalar";
    case pa_scheduler::ccec_pmu::WindowMode::ScalarDouble:
        return "scalar-double";
    case pa_scheduler::ccec_pmu::WindowMode::IcacheSingle:
        return "icache-single";
    case pa_scheduler::ccec_pmu::WindowMode::SubmitAll:
        return "submit-all";
    }
    return "invalid";
}

bool ParsePmuOptions(int argc, char **argv, PmuOptions *pmu, std::vector<char *> *common_argv) {
    // PMU 参数只属于 CCEC 验证分支；先摘出再交给三后端共享 parser，避免 CPU/AscendC 静默接受却不生效。
    bool mode_seen = false;
    bool nops_seen = false;
    bool icache_trials_seen = false;
    bool json_seen = false;
    common_argv->clear();
    common_argv->push_back(argv[0]);
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument != "--pmu-window" && argument != "--pmu-scalar-nops" &&
            argument != "--pmu-icache-trials" && argument != "--pmu-json") {
            common_argv->push_back(argv[index]);
            continue;
        }
        if (index + 1 >= argc) {
            std::fprintf(stderr, "Missing value after %s\n", argument.c_str());
            return false;
        }
        const char *value = argv[++index];
        if (argument == "--pmu-window") {
            if (mode_seen) {
                std::fprintf(stderr, "Specify --pmu-window only once.\n");
                return false;
            }
            const std::string name = value;
            if (name == "off") {
                pmu->mode = pa_scheduler::ccec_pmu::WindowMode::Off;
            } else if (name == "empty") {
                pmu->mode = pa_scheduler::ccec_pmu::WindowMode::Empty;
            } else if (name == "scalar") {
                pmu->mode = pa_scheduler::ccec_pmu::WindowMode::Scalar;
            } else if (name == "scalar-double") {
                pmu->mode = pa_scheduler::ccec_pmu::WindowMode::ScalarDouble;
            } else if (name == "icache-single") {
                pmu->mode = pa_scheduler::ccec_pmu::WindowMode::IcacheSingle;
            } else if (name == "submit-all") {
                pmu->mode = pa_scheduler::ccec_pmu::WindowMode::SubmitAll;
            } else {
                std::fprintf(
                    stderr,
                    "Invalid --pmu-window value: %s "
                    "(expected off|empty|scalar|scalar-double|icache-single|submit-all)\n",
                    value
                );
                return false;
            }
            mode_seen = true;
        } else if (argument == "--pmu-scalar-nops") {
            if (nops_seen || !pa_scheduler::host::ParseUint(value, 0, 10000000, &pmu->scalar_nops)) {
                std::fprintf(stderr, "Invalid or duplicate --pmu-scalar-nops value: %s\n", value);
                return false;
            }
            nops_seen = true;
        } else if (argument == "--pmu-icache-trials") {
            if (icache_trials_seen ||
                !pa_scheduler::host::ParseUint(value, 1, 10000, &pmu->icache_trials)) {
                std::fprintf(stderr, "Invalid or duplicate --pmu-icache-trials value: %s\n", value);
                return false;
            }
            icache_trials_seen = true;
        } else {
            if (json_seen || *value == '\0') {
                std::fprintf(stderr, "Invalid or duplicate --pmu-json path: %s\n", value);
                return false;
            }
            pmu->json_path = value;
            json_seen = true;
        }
    }
    if (nops_seen && pmu->mode != pa_scheduler::ccec_pmu::WindowMode::Scalar &&
        pmu->mode != pa_scheduler::ccec_pmu::WindowMode::ScalarDouble) {
        std::fprintf(stderr, "--pmu-scalar-nops requires a scalar PMU window.\n");
        return false;
    }
    if (icache_trials_seen && pmu->mode != pa_scheduler::ccec_pmu::WindowMode::IcacheSingle) {
        std::fprintf(stderr, "--pmu-icache-trials requires the icache-single PMU window.\n");
        return false;
    }
    return true;
}

using HalResMapFn = int (*)(uint32_t, struct res_map_info *, unsigned long *, uint32_t *);
using HalResUnmapFn = int (*)(uint32_t, struct res_map_info *);

struct PmuRegisterMappings {
    HalResUnmapFn unmap = nullptr;
    std::vector<res_map_info> mapped_resources;
    std::vector<uint64_t> register_bases;
};

bool UnmapPmuRegisters(uint32_t device, PmuRegisterMappings *mappings) {
    bool ok = true;
    if (mappings->unmap != nullptr) {
        for (auto iterator = mappings->mapped_resources.rbegin(); iterator != mappings->mapped_resources.rend();
             ++iterator) {
            const int error = mappings->unmap(device, &*iterator);
            if (error != 0) {
                std::fprintf(stderr, "halResUnmap failed for core %u (rc=%d)\n", iterator->res_id, error);
                ok = false;
            }
        }
    }
    mappings->mapped_resources.clear();
    mappings->register_bases.clear();
    return ok;
}

bool MapPmuRegisters(uint32_t device, PmuRegisterMappings *mappings) {
    using namespace pa_scheduler::ccec_pmu;
    const auto map = reinterpret_cast<HalResMapFn>(dlsym(RTLD_DEFAULT, "halResMap"));
    mappings->unmap = reinterpret_cast<HalResUnmapFn>(dlsym(RTLD_DEFAULT, "halResUnmap"));
    if (map == nullptr || mappings->unmap == nullptr) {
        std::fprintf(stderr, "halResMap/halResUnmap is unavailable in the current CANN driver process.\n");
        return false;
    }

    mappings->register_bases.assign(kPhysicalSubcoreCount, 0);
    mappings->mapped_resources.reserve(kPhysicalAicoreCount);
    for (uint32_t aicore = 0; aicore < kPhysicalAicoreCount; ++aicore) {
        res_map_info info{};
        info.target_proc_type = PROCESS_CP1;
        info.res_type = RES_AICORE;
        info.res_id = aicore;
        unsigned long map_address = 0;
        uint32_t map_bytes = kAicoreMapBytes;
        const int error = map(device, &info, &map_address, &map_bytes);
        if (error != 0 || map_address == 0 || map_bytes < kAicoreMapBytes) {
            std::fprintf(
                stderr, "halResMap failed for core %u (rc=%d address=0x%lx bytes=%u)\n", aicore, error,
                map_address, map_bytes
            );
            (void)UnmapPmuRegisters(device, mappings);
            return false;
        }
        mappings->mapped_resources.push_back(info);

        // 与正式 A5 host_regs.cpp 相同：每个 die 的布局为 18 AIC，随后是 36 AIV。
        const uint32_t die = aicore / kAicorePerDie;
        const uint32_t local = aicore % kAicorePerDie;
        const uint32_t die_base = die * kSubcoresPerDie;
        mappings->register_bases[die_base + local] = static_cast<uint64_t>(map_address);
        mappings->register_bases[die_base + kAicorePerDie + local * 2] =
            static_cast<uint64_t>(map_address) + kAivFirstOffset;
        mappings->register_bases[die_base + kAicorePerDie + local * 2 + 1] =
            static_cast<uint64_t>(map_address) + kAivSecondOffset;
    }
    return true;
}

void ConfigurePmu(pa_scheduler::SchedulerState *state, const PmuOptions &pmu, const void *register_table) {
    using namespace pa_scheduler::ccec_pmu;
    state->config.reserved[kConfigMode] = static_cast<uint32_t>(pmu.mode);
    state->config.reserved[kConfigWorkAmount] =
        pmu.mode == WindowMode::IcacheSingle ? pmu.icache_trials : pmu.scalar_nops;
    StorePointer(state->config.reserved, register_table);
    state->config.reserved[kConfigMagic] = pmu.mode == WindowMode::Off ? 0 : kConfigMagicValue;
}

struct PmuAggregate {
    std::vector<uint64_t> total_cycles;
    std::vector<uint64_t> window_ticks;
    std::vector<uint64_t> warm_total_cycles;
    std::vector<uint64_t> warm_window_ticks;
    std::vector<uint64_t> vector_busy;
    std::vector<uint64_t> cube_busy;
    std::vector<uint64_t> scalar_busy;
    std::vector<uint64_t> mte1_busy;
    std::vector<uint64_t> mte2_busy;
    std::vector<uint64_t> mte3_busy;
    std::vector<uint64_t> icache_requests;
    std::vector<uint64_t> icache_misses;
    std::vector<uint64_t> warm_icache_requests;
    std::vector<uint64_t> warm_icache_misses;
    std::vector<uint64_t> fix_busy;
    uint32_t trusted = 0;
};

void AddPmuSample(const pa_scheduler::WorkerResult &result, PmuAggregate *aggregate) {
    aggregate->total_cycles.push_back(result.pmu_total_cycles);
    aggregate->window_ticks.push_back(result.pmu_window_ticks);
    aggregate->warm_total_cycles.push_back(result.pmu_warm_total_cycles);
    aggregate->warm_window_ticks.push_back(result.pmu_warm_window_ticks);
    aggregate->vector_busy.push_back(result.pmu_vector_busy);
    aggregate->cube_busy.push_back(result.pmu_cube_busy);
    aggregate->scalar_busy.push_back(result.pmu_scalar_busy);
    aggregate->mte1_busy.push_back(result.pmu_mte1_busy);
    aggregate->mte2_busy.push_back(result.pmu_mte2_busy);
    aggregate->mte3_busy.push_back(result.pmu_mte3_busy);
    aggregate->icache_requests.push_back(result.pmu_icache_requests);
    aggregate->icache_misses.push_back(result.pmu_icache_misses);
    aggregate->warm_icache_requests.push_back(result.pmu_warm_icache_requests);
    aggregate->warm_icache_misses.push_back(result.pmu_warm_icache_misses);
    aggregate->fix_busy.push_back(result.pmu_fix_busy);
    aggregate->trusted +=
        (result.pmu_status & pa_scheduler::ccec_pmu::kStatusRequired) == pa_scheduler::ccec_pmu::kStatusRequired;
}

bool PrintSingleIcacheAggregate(
    const char *name, const PmuAggregate &aggregate, uint32_t trials_per_core
) {
    const pa_scheduler::host::Uint64Distribution cold_cycles =
        pa_scheduler::host::SummarizeUint64(aggregate.total_cycles);
    const pa_scheduler::host::Uint64Distribution warm_cycles =
        pa_scheduler::host::SummarizeUint64(aggregate.warm_total_cycles);
    const pa_scheduler::host::Uint64Distribution cold_ticks =
        pa_scheduler::host::SummarizeUint64(aggregate.window_ticks);
    const pa_scheduler::host::Uint64Distribution warm_ticks =
        pa_scheduler::host::SummarizeUint64(aggregate.warm_window_ticks);
    const pa_scheduler::host::Uint64Distribution cold_requests =
        pa_scheduler::host::SummarizeUint64(aggregate.icache_requests);
    const pa_scheduler::host::Uint64Distribution warm_requests =
        pa_scheduler::host::SummarizeUint64(aggregate.warm_icache_requests);
    const pa_scheduler::host::Uint64Distribution cold_misses =
        pa_scheduler::host::SummarizeUint64(aggregate.icache_misses);
    const pa_scheduler::host::Uint64Distribution warm_misses =
        pa_scheduler::host::SummarizeUint64(aggregate.warm_icache_misses);
    const int64_t cycle_delta = static_cast<int64_t>(cold_cycles.total) -
        static_cast<int64_t>(warm_cycles.total);
    const int64_t tick_delta = static_cast<int64_t>(cold_ticks.total) -
        static_cast<int64_t>(warm_ticks.total);
    const int64_t miss_delta = static_cast<int64_t>(cold_misses.total) -
        static_cast<int64_t>(warm_misses.total);
    const uint64_t attempted = static_cast<uint64_t>(trials_per_core) * aggregate.total_cycles.size();
    const double misses_per_trial = attempted == 0U ? 0.0 : static_cast<double>(miss_delta) / attempted;
    const double cycles_per_miss = miss_delta <= 0 ? 0.0 : static_cast<double>(cycle_delta) / miss_delta;
    // 本用例的 get_sys_cnt 已按 1 GHz 时间基准校准，因此一个 tick 对应 1 ns。
    const double ns_per_miss = miss_delta <= 0 ? 0.0 : static_cast<double>(tick_delta) / miss_delta;
    std::printf(
        "[ICACHE-SINGLE-%s] cores=%zu trials_per_core=%u attempted=%llu "
        "cold_cycles=%llu warm_cycles=%llu cycle_delta=%lld cold_ticks=%llu warm_ticks=%llu "
        "tick_delta=%lld cold_req=%llu warm_req=%llu cold_miss=%llu warm_miss=%llu "
        "miss_delta=%lld misses_per_trial=%.6f cycles_per_miss=%.3f ns_per_miss=%.3f\n",
        name, aggregate.total_cycles.size(), trials_per_core, static_cast<unsigned long long>(attempted),
        static_cast<unsigned long long>(cold_cycles.total),
        static_cast<unsigned long long>(warm_cycles.total), static_cast<long long>(cycle_delta),
        static_cast<unsigned long long>(cold_ticks.total),
        static_cast<unsigned long long>(warm_ticks.total), static_cast<long long>(tick_delta),
        static_cast<unsigned long long>(cold_requests.total),
        static_cast<unsigned long long>(warm_requests.total),
        static_cast<unsigned long long>(cold_misses.total),
        static_cast<unsigned long long>(warm_misses.total), static_cast<long long>(miss_delta),
        misses_per_trial, cycles_per_miss, ns_per_miss
    );
    std::printf(
        "[ICACHE-FORMULA-%s] estimated_scalar_icache_time_ns = cnt7_icache_miss * %.3f\n",
        name, ns_per_miss
    );
    return cycle_delta > 0 && tick_delta > 0 && miss_delta == static_cast<int64_t>(attempted);
}

void PrintPmuAggregate(const char *name, const PmuAggregate &aggregate) {
    const pa_scheduler::host::Uint64Distribution total =
        pa_scheduler::host::SummarizeUint64(aggregate.total_cycles);
    const pa_scheduler::host::Uint64Distribution scalar =
        pa_scheduler::host::SummarizeUint64(aggregate.scalar_busy);
    const pa_scheduler::host::Uint64Distribution vector =
        pa_scheduler::host::SummarizeUint64(aggregate.vector_busy);
    const pa_scheduler::host::Uint64Distribution cube =
        pa_scheduler::host::SummarizeUint64(aggregate.cube_busy);
    const pa_scheduler::host::Uint64Distribution mte1 =
        pa_scheduler::host::SummarizeUint64(aggregate.mte1_busy);
    const pa_scheduler::host::Uint64Distribution mte2 =
        pa_scheduler::host::SummarizeUint64(aggregate.mte2_busy);
    const pa_scheduler::host::Uint64Distribution mte3 =
        pa_scheduler::host::SummarizeUint64(aggregate.mte3_busy);
    const pa_scheduler::host::Uint64Distribution requests =
        pa_scheduler::host::SummarizeUint64(aggregate.icache_requests);
    const pa_scheduler::host::Uint64Distribution misses =
        pa_scheduler::host::SummarizeUint64(aggregate.icache_misses);
    const double miss_rate = requests.total == 0 ? 0.0 : 100.0 * misses.total / requests.total;
    std::printf(
        "[PMU-%s] cores=%zu total_sum=%llu total_median=%.1f total_p95=%llu "
        "scalar_busy=%llu vector_busy=%llu cube_busy=%llu mte1_busy=%llu mte2_busy=%llu "
        "mte3_busy=%llu icache_req=%llu icache_miss=%llu miss_rate=%.4f%%\n",
        name, aggregate.total_cycles.size(), static_cast<unsigned long long>(total.total), total.median,
        static_cast<unsigned long long>(total.p95), static_cast<unsigned long long>(scalar.total),
        static_cast<unsigned long long>(vector.total), static_cast<unsigned long long>(cube.total),
        static_cast<unsigned long long>(mte1.total), static_cast<unsigned long long>(mte2.total),
        static_cast<unsigned long long>(mte3.total),
        static_cast<unsigned long long>(requests.total), static_cast<unsigned long long>(misses.total), miss_rate
    );
}

struct PmuValidation {
    uint32_t trusted = 0;
    uint32_t unique_physical_core_ids = 0;
    uint32_t owner_bitmap_members = 0;
    uint32_t exact_worker_slots = 0;
    uint32_t physical_role_matches = 0;
    uint32_t mixed_triplet_matches = 0;
    uint32_t window_started = 0;
    uint32_t window_stopped = 0;
    uint32_t icache_pairs = 0;
    uint32_t icache_calibrated_cores = 0;
    uint32_t prior_snapshot_larger = 0;
    uint32_t submit_engine_workers_expected = 0;
    uint32_t submit_engine_workers_matched = 0;
    uint32_t maximum_programmable_counter = 0;
    bool icache_order_valid = true;
    bool icache_measurement_valid = true;
    bool submit_engine_observation_valid = true;
    bool counter_below_risk_threshold = true;
    bool passed = true;
};

// 32-bit programmable counter 无法仅凭终值证明从未回卷。正式文件采用 25%
// 高水位作为保守拒绝阈值；它只降低风险，不把“未越线”表述成回卷证明。
constexpr uint32_t kProgrammableCounterRiskThreshold = UINT32_MAX / 4U;

bool ValidatePmu(
    const pa_scheduler::SchedulerState &state, uint32_t run, const PmuOptions &pmu,
    const WinnerWorkloadOptions &workload,
    const pa_scheduler::pmu_owner::PmuOwnerControl *owner, PmuValidation *validation
) {
    using namespace pa_scheduler::ccec_pmu;
    if (pmu.mode == WindowMode::Off) {
        *validation = PmuValidation{};
        return true;
    }

    bool seen[kPhysicalSubcoreCount] = {};
    uint32_t trusted = 0;
    uint32_t unique = 0;
    uint32_t owner_members = 0;
    uint32_t exact_worker_slots = 0;
    uint32_t physical_role_matches = 0;
    uint32_t window_started = 0;
    uint32_t window_stopped = 0;
    uint32_t icache_pairs = 0;
    uint32_t icache_calibrated_cores = 0;
    uint32_t prior_larger = 0;
    uint32_t submit_engine_workers_expected = 0;
    uint32_t submit_engine_workers_matched = 0;
    uint32_t maximum_programmable_counter = 0;
    bool icache_order_valid = true;
    uint32_t bad_printed = 0;
    PmuAggregate all;
    PmuAggregate aic;
    PmuAggregate aiv;
    for (uint32_t worker = 0; worker < pa_scheduler::kWorkers; ++worker) {
        const pa_scheduler::WorkerResult &result = state.results[worker];
        const uint32_t status = result.pmu_status;
        const uint32_t core_id = StatusCoreId(status);
        const bool record_trusted = (status & kStatusRequired) == kStatusRequired;
        const bool logical_aic = worker < pa_scheduler::kAicWorkers;
        const bool physical_aic = pa_scheduler::pmu_owner::IsAicPhysicalSlot(core_id);
        trusted += record_trusted;
        owner_members += owner != nullptr && pa_scheduler::pmu_owner::IsConfigured(*owner, core_id);
        exact_worker_slots += result.worker_id == worker;
        physical_role_matches += logical_aic == physical_aic;
        window_started += (status & kStatusWindowStarted) != 0U;
        window_stopped += (status & kStatusWindowStopped) != 0U;
        icache_pairs += (status & kStatusIcachePairObserved) != 0U;
        prior_larger += (status & kStatusPriorSnapshotLarger) != 0;
        if (pmu.mode == WindowMode::SubmitAll &&
            workload.mode == pa_scheduler::WinnerWorkloadMode::RealCompute) {
            const uint64_t submit_engine_tasks =
                result.placement[static_cast<uint32_t>(pa_scheduler::DrainPlace::EfDrain)] +
                result.placement[static_cast<uint32_t>(pa_scheduler::DrainPlace::RingBackpressure)];
            const uint32_t relevant_busy = logical_aic ? result.pmu_cube_busy : result.pmu_vector_busy;
            const bool engine_observation_matches =
                (submit_engine_tasks == 0U) == (relevant_busy == 0U);
            submit_engine_workers_expected += submit_engine_tasks != 0U;
            submit_engine_workers_matched += engine_observation_matches;
        }
        icache_order_valid &= result.pmu_icache_misses <= result.pmu_icache_requests &&
            result.pmu_warm_icache_misses <= result.pmu_warm_icache_requests;
        if (pmu.mode == WindowMode::IcacheSingle) {
            const int64_t worker_cycle_delta = static_cast<int64_t>(result.pmu_total_cycles) -
                static_cast<int64_t>(result.pmu_warm_total_cycles);
            const int64_t worker_tick_delta = static_cast<int64_t>(result.pmu_window_ticks) -
                static_cast<int64_t>(result.pmu_warm_window_ticks);
            const int64_t worker_miss_delta = static_cast<int64_t>(result.pmu_icache_misses) -
                static_cast<int64_t>(result.pmu_warm_icache_misses);
            icache_calibrated_cores += worker_cycle_delta > 0 && worker_tick_delta > 0 &&
                worker_miss_delta == static_cast<int64_t>(pmu.icache_trials) &&
                result.pmu_warm_icache_misses == 0U &&
                result.pmu_icache_misses == pmu.icache_trials;
        }
        const uint32_t programmable[] = {
            result.pmu_vector_busy, result.pmu_cube_busy, result.pmu_scalar_busy,
            result.pmu_mte1_busy, result.pmu_mte2_busy, result.pmu_mte3_busy,
            result.pmu_icache_requests, result.pmu_icache_misses, result.pmu_fix_busy,
            result.pmu_warm_icache_requests, result.pmu_warm_icache_misses,
        };
        for (uint32_t value : programmable) {
            maximum_programmable_counter = std::max(maximum_programmable_counter, value);
        }
        if (core_id < kPhysicalSubcoreCount && !seen[core_id]) {
            seen[core_id] = true;
            ++unique;
        }
        if (!record_trusted && bad_printed < 8) {
            std::printf(
                "[PMU-BAD] worker=%u role=%llu coreid=%u status=0x%08x total=%llu scalar=%u "
                "req=%u miss=%u warm_total=%llu warm_req=%u warm_miss=%u\n",
                worker, static_cast<unsigned long long>(result.role), core_id, status,
                static_cast<unsigned long long>(result.pmu_total_cycles), result.pmu_scalar_busy,
                result.pmu_icache_requests, result.pmu_icache_misses,
                static_cast<unsigned long long>(result.pmu_warm_total_cycles),
                result.pmu_warm_icache_requests, result.pmu_warm_icache_misses
            );
            ++bad_printed;
        }
        AddPmuSample(result, &all);
        AddPmuSample(result, result.role == static_cast<uint32_t>(pa_scheduler::CoreRole::Aic) ? &aic : &aiv);
    }

    uint32_t mixed_triplet_matches = 0U;
    for (uint32_t block = 0U; block < pa_scheduler::kAicWorkers; ++block) {
        const uint32_t aic_id = StatusCoreId(state.results[block].pmu_status);
        if (!pa_scheduler::pmu_owner::IsAicPhysicalSlot(aic_id)) continue;
        const uint32_t die_base = (aic_id / pa_scheduler::pmu_owner::kSubcoresPerDie) *
            pa_scheduler::pmu_owner::kSubcoresPerDie;
        const uint32_t local = aic_id % pa_scheduler::pmu_owner::kSubcoresPerDie;
        const uint32_t expected_aiv0 = die_base + pa_scheduler::pmu_owner::kAicPerDie + local * 2U;
        const uint32_t expected_aiv1 = expected_aiv0 + 1U;
        const uint32_t aiv0_id = StatusCoreId(
            state.results[pa_scheduler::kAicWorkers + block * 2U].pmu_status
        );
        const uint32_t aiv1_id = StatusCoreId(
            state.results[pa_scheduler::kAicWorkers + block * 2U + 1U].pmu_status
        );
        mixed_triplet_matches += aiv0_id == expected_aiv0 && aiv1_id == expected_aiv1;
    }

    PrintPmuAggregate("ALL", all);
    PrintPmuAggregate("AIC", aic);
    PrintPmuAggregate("AIV", aiv);
    bool icache_measurement_ok = true;
    if (pmu.mode == WindowMode::IcacheSingle) {
        const bool all_ok = PrintSingleIcacheAggregate("ALL", all, pmu.icache_trials);
        const bool aic_ok = PrintSingleIcacheAggregate("AIC", aic, pmu.icache_trials);
        const bool aiv_ok = PrintSingleIcacheAggregate("AIV", aiv, pmu.icache_trials);
        icache_measurement_ok = all_ok && aic_ok && aiv_ok &&
            icache_pairs == pa_scheduler::kWorkers &&
            icache_calibrated_cores == pa_scheduler::kWorkers;
    }
    const bool records_ok = trusted == pa_scheduler::kWorkers;
    const bool core_ids_ok = unique == pa_scheduler::kWorkers;
    const bool owner_members_ok = owner_members == pa_scheduler::kWorkers;
    const bool worker_slots_ok = exact_worker_slots == pa_scheduler::kWorkers;
    const bool physical_roles_ok = physical_role_matches == pa_scheduler::kWorkers;
    const bool mixed_triplets_ok = mixed_triplet_matches == pa_scheduler::kAicWorkers;
    const bool windows_started_ok = window_started == pa_scheduler::kWorkers;
    const bool windows_stopped_ok = window_stopped == pa_scheduler::kWorkers;
    const bool submit_engine_observation_ok =
        pmu.mode != WindowMode::SubmitAll ||
        workload.mode != pa_scheduler::WinnerWorkloadMode::RealCompute ||
        submit_engine_workers_matched == pa_scheduler::kWorkers;
    const bool counter_below_risk_threshold =
        maximum_programmable_counter < kProgrammableCounterRiskThreshold;
    std::printf(
        "[PMU] run=%u window=%s calibration_scalar_nops=%u icache_trials=%u trusted=%u/%u "
        "unique_coreids=%u/%u prior_larger=%u/%u icache_pairs=%u/%u calibrated_cores=%u/%u "
        "programmable_max=%u headroom=%u\n", run,
        PmuModeName(pmu.mode), pmu.scalar_nops, pmu.icache_trials, trusted, pa_scheduler::kWorkers,
        unique, pa_scheduler::kWorkers, prior_larger, pa_scheduler::kWorkers,
        icache_pairs, pa_scheduler::kWorkers, icache_calibrated_cores, pa_scheduler::kWorkers,
        maximum_programmable_counter,
        UINT32_MAX - maximum_programmable_counter
    );
    std::printf("[ASSERT] %-48s %s\n", "all PMU records have configured selectors and data",
                records_ok ? "PASS" : "FAIL");
    std::printf("[ASSERT] %-48s %s\n", "all 96 PMU physical subcore ids are unique",
                core_ids_ok ? "PASS" : "FAIL");
    std::printf("[ASSERT] %-48s %s\n", "all PMU physical ids belong to the owner bitmap",
                owner_members_ok ? "PASS" : "FAIL");
    std::printf("[ASSERT] %-48s %s\n", "worker result slots and ids match exactly",
                worker_slots_ok ? "PASS" : "FAIL");
    std::printf("[ASSERT] %-48s %s\n", "logical AIC/AIV roles match physical subcores",
                physical_roles_ok ? "PASS" : "FAIL");
    std::printf("[ASSERT] %-48s %s\n", "all 32 mixed blocks map to physical 1:2 triplets",
                mixed_triplets_ok ? "PASS" : "FAIL");
    std::printf("[ASSERT] %-48s %s\n", "all 96 PMU windows executed start",
                windows_started_ok ? "PASS" : "FAIL");
    std::printf("[ASSERT] %-48s %s\n", "all 96 started PMU windows executed stop",
                windows_stopped_ok ? "PASS" : "FAIL");
    std::printf("[ASSERT] %-48s %s\n", "I-cache misses do not exceed requests",
                icache_order_valid ? "PASS" : "FAIL");
    std::printf("[ASSERT] %-48s %s\n", "programmable counters stay below 25% risk threshold",
                counter_below_risk_threshold ? "PASS" : "FAIL");
    if (pmu.mode == WindowMode::SubmitAll &&
        workload.mode == pa_scheduler::WinnerWorkloadMode::RealCompute) {
        std::printf(
            "[ASSERT] %-48s %s (active_workers=%u matched_workers=%u/%u)\n",
            "Submit placement has matching AIC/AIV engine PMU",
            submit_engine_observation_ok ? "PASS" : "FAIL", submit_engine_workers_expected,
            submit_engine_workers_matched, pa_scheduler::kWorkers
        );
    }
    if (pmu.mode == WindowMode::IcacheSingle) {
        std::printf("[ASSERT] %-48s %s\n", "each cold trial adds exactly one CNT7 I-cache miss",
                    icache_measurement_ok ? "PASS" : "FAIL");
    }
    validation->trusted = trusted;
    validation->unique_physical_core_ids = unique;
    validation->owner_bitmap_members = owner_members;
    validation->exact_worker_slots = exact_worker_slots;
    validation->physical_role_matches = physical_role_matches;
    validation->mixed_triplet_matches = mixed_triplet_matches;
    validation->window_started = window_started;
    validation->window_stopped = window_stopped;
    validation->icache_pairs = icache_pairs;
    validation->icache_calibrated_cores = icache_calibrated_cores;
    validation->prior_snapshot_larger = prior_larger;
    validation->submit_engine_workers_expected = submit_engine_workers_expected;
    validation->submit_engine_workers_matched = submit_engine_workers_matched;
    validation->maximum_programmable_counter = maximum_programmable_counter;
    validation->icache_order_valid = icache_order_valid;
    validation->icache_measurement_valid = icache_measurement_ok;
    validation->submit_engine_observation_valid = submit_engine_observation_ok;
    validation->counter_below_risk_threshold = counter_below_risk_threshold;
    validation->passed = records_ok && core_ids_ok && owner_members_ok && worker_slots_ok &&
        physical_roles_ok && mixed_triplets_ok && windows_started_ok && windows_stopped_ok &&
        icache_order_valid && icache_measurement_ok && submit_engine_observation_ok &&
        counter_below_risk_threshold;
    return validation->passed;
}

void WriteJsonString(std::FILE *output, const std::string &value) {
    std::fputc('"', output);
    for (unsigned char character : value) {
        switch (character) {
        case '"':
            std::fputs("\\\"", output);
            break;
        case '\\':
            std::fputs("\\\\", output);
            break;
        case '\b':
            std::fputs("\\b", output);
            break;
        case '\f':
            std::fputs("\\f", output);
            break;
        case '\n':
            std::fputs("\\n", output);
            break;
        case '\r':
            std::fputs("\\r", output);
            break;
        case '\t':
            std::fputs("\\t", output);
            break;
        default:
            if (character < 0x20U) {
                std::fprintf(output, "\\u%04x", static_cast<unsigned int>(character));
            } else {
                std::fputc(character, output);
            }
        }
    }
    std::fputc('"', output);
}

void WriteMetricDistribution(std::FILE *output, const std::vector<uint64_t> &values) {
    const pa_scheduler::host::Uint64Distribution summary = pa_scheduler::host::SummarizeUint64(values);
    const double mean = values.empty() ? 0.0 : static_cast<double>(summary.total) / values.size();
    std::fprintf(
        output, "{\"sum\":%llu,\"mean\":%.17g,\"median\":%.17g,\"p95\":%llu,\"max\":%llu}",
        static_cast<unsigned long long>(summary.total), mean, summary.median,
        static_cast<unsigned long long>(summary.p95), static_cast<unsigned long long>(summary.maximum)
    );
}

void WritePmuAggregateJson(
    std::FILE *output, const PmuAggregate &aggregate, bool icache_single
) {
    const pa_scheduler::host::Uint64Distribution requests =
        pa_scheduler::host::SummarizeUint64(aggregate.icache_requests);
    const pa_scheduler::host::Uint64Distribution misses =
        pa_scheduler::host::SummarizeUint64(aggregate.icache_misses);
    uint32_t active_cores = 0;
    for (uint64_t cycles : aggregate.total_cycles) active_cores += cycles != 0;
    std::fprintf(
        output, "{\"cores\":%zu,\"active_cores\":%u,\"trusted_cores\":%u,\"total_cycles\":",
        aggregate.total_cycles.size(), active_cores, aggregate.trusted
    );
    WriteMetricDistribution(output, aggregate.total_cycles);
    std::fputs(",\"vector_busy\":", output);
    WriteMetricDistribution(output, aggregate.vector_busy);
    std::fputs(",\"cube_busy\":", output);
    WriteMetricDistribution(output, aggregate.cube_busy);
    std::fputs(",\"scalar_busy\":", output);
    WriteMetricDistribution(output, aggregate.scalar_busy);
    std::fputs(",\"mte1_busy\":", output);
    WriteMetricDistribution(output, aggregate.mte1_busy);
    std::fputs(",\"mte2_busy\":", output);
    WriteMetricDistribution(output, aggregate.mte2_busy);
    std::fputs(",\"mte3_busy\":", output);
    WriteMetricDistribution(output, aggregate.mte3_busy);
    std::fputs(",\"icache_requests\":", output);
    WriteMetricDistribution(output, aggregate.icache_requests);
    std::fputs(",\"icache_misses\":", output);
    WriteMetricDistribution(output, aggregate.icache_misses);
    std::fputs(",\"fix_busy\":", output);
    WriteMetricDistribution(output, aggregate.fix_busy);
    std::fputs(",\"icache_miss_rate\":", output);
    if (requests.total == 0) {
        std::fputs("null", output);
    } else {
        // 全局 miss rate 必须以总 miss/总 request 计算，不能平均逐核百分比。
        std::fprintf(output, "%.17g", static_cast<double>(misses.total) / requests.total);
    }
    if (icache_single) {
        const pa_scheduler::host::Uint64Distribution cold_cycles =
            pa_scheduler::host::SummarizeUint64(aggregate.total_cycles);
        const pa_scheduler::host::Uint64Distribution warm_cycles =
            pa_scheduler::host::SummarizeUint64(aggregate.warm_total_cycles);
        const pa_scheduler::host::Uint64Distribution cold_ticks =
            pa_scheduler::host::SummarizeUint64(aggregate.window_ticks);
        const pa_scheduler::host::Uint64Distribution warm_ticks =
            pa_scheduler::host::SummarizeUint64(aggregate.warm_window_ticks);
        const pa_scheduler::host::Uint64Distribution warm_requests =
            pa_scheduler::host::SummarizeUint64(aggregate.warm_icache_requests);
        const pa_scheduler::host::Uint64Distribution warm_misses =
            pa_scheduler::host::SummarizeUint64(aggregate.warm_icache_misses);
        const int64_t cycle_delta = static_cast<int64_t>(cold_cycles.total) -
            static_cast<int64_t>(warm_cycles.total);
        const int64_t tick_delta = static_cast<int64_t>(cold_ticks.total) -
            static_cast<int64_t>(warm_ticks.total);
        const int64_t miss_delta = static_cast<int64_t>(misses.total) -
            static_cast<int64_t>(warm_misses.total);
        std::fputs(",\"icache_single_pair\":{\"cold_window_ticks\":", output);
        WriteMetricDistribution(output, aggregate.window_ticks);
        std::fputs(",\"warm_total_cycles\":", output);
        WriteMetricDistribution(output, aggregate.warm_total_cycles);
        std::fputs(",\"warm_window_ticks\":", output);
        WriteMetricDistribution(output, aggregate.warm_window_ticks);
        std::fputs(",\"warm_icache_requests\":", output);
        WriteMetricDistribution(output, aggregate.warm_icache_requests);
        std::fputs(",\"warm_icache_misses\":", output);
        WriteMetricDistribution(output, aggregate.warm_icache_misses);
        std::fprintf(
            output,
            ",\"cycle_delta\":%lld,\"tick_delta\":%lld,\"miss_delta\":%lld,"
            "\"cold_request_sum\":%llu,\"warm_request_sum\":%llu,\"ns_per_miss\":",
            static_cast<long long>(cycle_delta), static_cast<long long>(tick_delta),
            static_cast<long long>(miss_delta), static_cast<unsigned long long>(requests.total),
            static_cast<unsigned long long>(warm_requests.total)
        );
        if (miss_delta <= 0) {
            std::fputs("null", output);
        } else {
            std::fprintf(output, "%.17g", static_cast<double>(tick_delta) / miss_delta);
        }
        std::fputc('}', output);
    }
    std::fputc('}', output);
}

uint32_t PmuWindowSegments(const PmuOptions &pmu) {
    const pa_scheduler::ccec_pmu::WindowMode mode = pmu.mode;
    if (mode == pa_scheduler::ccec_pmu::WindowMode::ScalarDouble) return 2U;
    if (mode == pa_scheduler::ccec_pmu::WindowMode::IcacheSingle) return pmu.icache_trials * 2U;
    return 1U;
}

uint32_t CountConfiguredMixedTriplets(const pa_scheduler::pmu_owner::PmuOwnerControl &owner) {
    uint32_t complete = 0U;
    for (uint32_t die_base = 0U;
         die_base < pa_scheduler::pmu_owner::kPhysicalSubcoreCount;
         die_base += pa_scheduler::pmu_owner::kSubcoresPerDie) {
        for (uint32_t local = 0U; local < pa_scheduler::pmu_owner::kAicPerDie; ++local) {
            const uint32_t aic = die_base + local;
            const uint32_t aiv0 = die_base + pa_scheduler::pmu_owner::kAicPerDie + local * 2U;
            const uint32_t aiv1 = aiv0 + 1U;
            complete += pa_scheduler::pmu_owner::IsConfigured(owner, aic) &&
                pa_scheduler::pmu_owner::IsConfigured(owner, aiv0) &&
                pa_scheduler::pmu_owner::IsConfigured(owner, aiv1);
        }
    }
    return complete;
}

bool ExportPmuJson(
    const pa_scheduler::SchedulerState &state, const pa_scheduler::host::Options &options,
    const PmuOptions &pmu, const WinnerWorkloadOptions &workload,
    uint32_t run, double host_us, double submit_span_us,
    const PmuValidation &validation, bool semantic_passed, bool workload_output_passed,
    const pa_scheduler::pmu_owner::PmuOwnerControl &owner, bool restore_passed,
    const std::string &output_path
) {
    using namespace pa_scheduler::ccec_pmu;
    PmuAggregate all;
    PmuAggregate aic;
    PmuAggregate aiv;
    uint32_t active_output_tiles = 0;
    uint64_t ef_drain_kernels = 0;
    uint64_t ring_backpressure_kernels = 0;
    uint64_t final_drain_kernels = 0;
    for (uint32_t worker = 0; worker < pa_scheduler::kWorkers; ++worker) {
        const pa_scheduler::WorkerResult &result = state.results[worker];
        AddPmuSample(result, &all);
        AddPmuSample(result, result.role == static_cast<uint32_t>(pa_scheduler::CoreRole::Aic) ? &aic : &aiv);
        if (result.role == static_cast<uint32_t>(pa_scheduler::CoreRole::Aic)) {
            active_output_tiles += result.kernel_counts[0] != 0;
            active_output_tiles += result.kernel_counts[2] != 0;
        } else if (result.role == static_cast<uint32_t>(pa_scheduler::CoreRole::Aiv)) {
            active_output_tiles += result.kernel_counts[1] != 0;
            active_output_tiles += result.kernel_counts[3] != 0;
        }
        ef_drain_kernels += result.placement[static_cast<uint32_t>(pa_scheduler::DrainPlace::EfDrain)];
        ring_backpressure_kernels +=
            result.placement[static_cast<uint32_t>(pa_scheduler::DrainPlace::RingBackpressure)];
        final_drain_kernels += result.placement[static_cast<uint32_t>(pa_scheduler::DrainPlace::FinalDrain)];
    }

    const auto generated = std::chrono::system_clock::now().time_since_epoch();
    const uint64_t generated_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(generated).count()
    );
    const std::string capture_id = "pa-pmu-" + std::to_string(generated_ns) + "-run" + std::to_string(run);
    const std::string temporary_path = output_path + ".tmp";
    // 临时文件与最终文件都采用 no-replace 语义：并发采集不能截断同名 tmp，
    // 也不能在最终发布时覆盖另一份已经完成的证据文件。
    const int output_fd = open(temporary_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    std::FILE *output = output_fd < 0 ? nullptr : fdopen(output_fd, "wb");
    if (output == nullptr) {
        const int open_error = errno;
        if (output_fd >= 0) {
            (void)close(output_fd);
            (void)std::remove(temporary_path.c_str());
        }
        std::fprintf(
            stderr, "Cannot exclusively create PMU JSON output %s: %s\n", temporary_path.c_str(),
            std::strerror(open_error)
        );
        return false;
    }
    std::vector<char> output_buffer(1U << 20);
    std::setvbuf(output, output_buffer.data(), _IOFBF, output_buffer.size());

    const bool submit_window = IsSubmitWindow(pmu.mode);
    const bool icache_single = pmu.mode == WindowMode::IcacheSingle;
    const bool real_compute = workload.mode == pa_scheduler::WinnerWorkloadMode::RealCompute;
    const bool simulated_task_nops_nonzero = !real_compute &&
        (options.nops.qk != 0U || options.nops.sf != 0U ||
         options.nops.pv != 0U || options.nops.up != 0U);
    const pa_scheduler::WorkloadCounts active_counts = real_compute
        ? workload.repeats
        : pa_scheduler::WorkloadCounts{
              options.nops.qk, options.nops.sf, options.nops.pv, options.nops.up
          };
    const uint32_t owner_bitmap_count = pa_scheduler::pmu_owner::CountConfigured(owner);
    const uint32_t owner_complete_triplets = CountConfiguredMixedTriplets(owner);
    std::fputs("{\n\"schema\":{\"name\":\"pa_scheduler_pmu_phase_windows\",\"version\":3},\n", output);
    std::fputs("\"capture\":{\"capture_id\":", output);
    WriteJsonString(output, capture_id);
    std::fprintf(
        output,
        ",\"generated_unix_time_ns\":%llu,\"run_index\":%u,\"accepted\":true,"
        "\"usable_for_same_configuration_submit_comparison\":%s,"
        "\"usable_as_absolute_real_pa_profile\":false,\"window_scope\":\"%s\","
        "\"pmu_probe_position\":\"%s\",\"scheduler_hot_path_included\":%s,"
        "\"total_sum_is_core_work_not_wall_time\":true,"
        "\"submit_window_excludes_final_drain\":%s,"
        "\"published_after_runtime_cleanup\":true,\"runtime_cleanup_passed\":true,"
        "\"owner_restore_passed\":%s},\n",
        static_cast<unsigned long long>(generated_ns), run,
        submit_window ? "true" : "false",
        submit_window ? "per_worker_orchestration_to_last_submit_return" : "post_scheduler_calibration_probe",
        submit_window ? "inside_RunScheduler" : "after_RunScheduler",
        submit_window ? "true" : "false", submit_window ? "true" : "false",
        restore_passed ? "true" : "false"
    );
    std::fputs("\"configuration\":{\"kernel_path\":", output);
    WriteJsonString(output, options.kernel_path);
    std::fprintf(
        output,
        ",\"device\":%u,\"batches\":%u,\"workers\":%u,\"aic_workers\":%u,\"aiv_workers\":%u,"
        "\"trace_enabled\":%s,\"trace_atomics\":%s,\"profile_phases\":%s,"
        "\"winner_workload\":{\"mode\":",
        options.device, options.batches, pa_scheduler::kWorkers, pa_scheduler::kAicWorkers,
        pa_scheduler::kAivWorkers, options.trace_enabled ? "true" : "false",
        options.trace_atomics ? "true" : "false",
        options.profile_phases ? "true" : "false"
    );
    WriteJsonString(output, WinnerWorkloadModeName(workload.mode));
    std::fputs(",\"input_pattern\":", output);
    WriteJsonString(
        output,
        real_compute ? pa_scheduler::host::RealComputePatternName(workload.pattern) : "none"
    );
    std::fprintf(
        output,
        ",\"config_version\":%u,\"counts\":{\"qk\":%u,\"sf\":%u,\"pv\":%u,\"up\":%u},"
        "\"unit\":",
        pa_scheduler::kWinnerWorkloadConfigVersion, active_counts.qk, active_counts.sf,
        active_counts.pv, active_counts.up
    );
    WriteJsonString(
        output,
        real_compute ? "complete_128x128_engine_pipeline_iteration" : "scalar_nop_instruction"
    );
    std::fprintf(
        output,
        ",\"tile_rows\":%u,\"tile_cols\":%u,\"shared_input_tiles\":%u,"
        "\"output_tiles_per_worker\":%u,\"workspace_bytes\":%zu,\"role_mapping\":",
        real_compute ? pa_scheduler::winner_workload::kTileRows : 0,
        real_compute ? pa_scheduler::winner_workload::kTileCols : 0,
        real_compute ? pa_scheduler::winner_workload::kSharedInputTiles : 0,
        real_compute ? pa_scheduler::winner_workload::kOutputTilesPerWorker : 0,
        real_compute ? pa_scheduler::winner_workload::kWorkspaceBytes : 0
    );
    if (real_compute) {
        std::fputs(
            "{\"qk\":\"cube_matmul\",\"pv\":\"cube_matmul\","
            "\"sf\":\"vector_add\",\"up\":\"vector_mul\"}",
            output
        );
    } else {
        std::fputs("null", output);
    }
    std::fprintf(
        output, ",\"engine_completion_waited_before_task_publish\":%s},\"nop_counts\":",
        real_compute ? "true" : "false"
    );
    if (real_compute) {
        std::fputs("null", output);
    } else {
        std::fprintf(
            output, "{\"qk\":%u,\"sf\":%u,\"pv\":%u,\"up\":%u}",
            options.nops.qk, options.nops.sf, options.nops.pv, options.nops.up
        );
    }
    std::fputs(",\"pmu_window\":", output);
    WriteJsonString(output, PmuModeName(pmu.mode));
    std::fputs(",\"calibration_scalar_nops_per_segment\":", output);
    if (submit_window || icache_single) {
        std::fputs("null", output);
    } else {
        std::fprintf(output, "%u", pmu.scalar_nops);
    }
    std::fputs(",\"icache_single_trials_per_core\":", output);
    if (icache_single) {
        std::fprintf(output, "%u", pmu.icache_trials);
    } else {
        std::fputs("null", output);
    }
    std::fprintf(
        output,
        ",\"window_segments_are_mode_contract_per_record\":true,"
        "\"icache_single_discarded_training_samples_per_core\":%u,"
        "\"icache_single_sys_counter_tick_ns\":%s,"
        "\"host_launch_to_sync_us\":%.17g,\"submit_span_us\":%.17g,"
        "\"selectors\":{\"cnt0_vector_busy\":%u,\"cnt1_cube_busy\":%u,"
        "\"cnt2_scalar_busy\":%u,\"cnt3_mte1_busy\":%u,\"cnt4_mte2_busy\":%u,"
        "\"cnt5_mte3_busy\":%u,\"cnt6_icache_request\":%u,\"cnt7_icache_miss\":%u,"
        "\"cnt8_fix_busy\":%u},\"counter_width_bits\":{\"total\":64,\"programmable\":32},"
        "\"counter_wrap_not_directly_detectable\":true,\"counter_wrap_absence_proven\":false,"
        "\"programmable_counter_risk_threshold\":%u,"
        "\"gate_start_stop_have_pipe_all_barriers\":true,"
        "\"phase_timestamp_calls_present\":true,\"phase_record_writes\":false,"
        "\"atomic_trace\":false,\"profile_accumulation\":false,"
        "\"simulated_task_nop_mechanism_executes_on_scalar\":%s,"
        "\"simulated_task_nops_nonzero\":%s,"
        "\"icache_miss_rate_definition\":\"sum(icache_misses)/sum(icache_requests)\"},\n",
        icache_single ? 2U : 0U, icache_single ? "1" : "null",
        host_us, submit_span_us, kVectorBusyEvent, kCubeBusyEvent, kScalarBusyEvent,
        kMte1BusyEvent, kMte2BusyEvent, kMte3BusyEvent, kIcacheRequestEvent, kIcacheMissEvent,
        kFixBusyEvent, kProgrammableCounterRiskThreshold,
        real_compute ? "false" : "true",
        simulated_task_nops_nonzero ? "true" : "false"
    );
    std::fprintf(
        output,
        "\"validation\":{\"semantic_passed\":%s,\"pmu_passed\":%s,"
        "\"real_compute_output_validation_required\":%s,"
        "\"real_compute_output_validation_passed\":%s,"
        "\"real_compute_active_output_tiles\":%u,"
        "\"real_compute_inactive_sentinel_tiles\":%u,"
        "\"real_compute_output_mismatches\":%u,"
        "\"submit_engine_observation_valid\":%s,"
        "\"submit_engine_workers_expected\":%u,"
        "\"submit_engine_workers_matched\":%u,"
        "\"kernel_placement_counts\":{\"ef_drain\":%llu,\"ring_backpressure\":%llu,"
        "\"final_drain\":%llu},\"trusted_records\":%u,"
        "\"expected_records\":%u,\"unique_physical_core_ids\":%u,\"expected_unique_core_ids\":%u,"
        "\"owner_bitmap_member_records\":%u,\"expected_owner_bitmap_member_records\":%u,"
        "\"exact_worker_slot_records\":%u,\"expected_exact_worker_slot_records\":%u,"
        "\"physical_role_match_records\":%u,\"expected_physical_role_match_records\":%u,"
        "\"mixed_triplet_matches\":%u,\"expected_mixed_triplet_matches\":%u,"
        "\"window_started_records\":%u,\"window_stopped_records\":%u,"
        "\"expected_window_records\":%u,\"prior_snapshot_larger_records\":%u,"
        "\"icache_pair_records\":%u,\"icache_calibrated_cores\":%u,"
        "\"icache_measurement_valid\":%s,"
        "\"icache_miss_le_request\":%s,\"counter_below_risk_threshold\":%s,"
        "\"maximum_programmable_counter\":%u,\"programmable_counter_risk_threshold\":%u,"
        "\"programmable_counter_headroom\":%u},\n",
        semantic_passed ? "true" : "false", validation.passed ? "true" : "false",
        real_compute ? "true" : "false",
        real_compute ? (workload_output_passed ? "true" : "false") : "null",
        real_compute ? active_output_tiles : 0,
        real_compute ? pa_scheduler::winner_workload::kOutputTiles - active_output_tiles : 0,
        real_compute && !workload_output_passed ? 1U : 0U,
        validation.submit_engine_observation_valid ? "true" : "false",
        validation.submit_engine_workers_expected,
        validation.submit_engine_workers_matched,
        static_cast<unsigned long long>(ef_drain_kernels),
        static_cast<unsigned long long>(ring_backpressure_kernels),
        static_cast<unsigned long long>(final_drain_kernels),
        validation.trusted,
        pa_scheduler::kWorkers, validation.unique_physical_core_ids, pa_scheduler::kWorkers,
        validation.owner_bitmap_members, pa_scheduler::kWorkers,
        validation.exact_worker_slots, pa_scheduler::kWorkers,
        validation.physical_role_matches, pa_scheduler::kWorkers,
        validation.mixed_triplet_matches, pa_scheduler::kAicWorkers,
        validation.window_started, validation.window_stopped, pa_scheduler::kWorkers,
        validation.prior_snapshot_larger, validation.icache_pairs,
        validation.icache_calibrated_cores,
        validation.icache_measurement_valid ? "true" : "false",
        validation.icache_order_valid ? "true" : "false",
        validation.counter_below_risk_threshold ? "true" : "false",
        validation.maximum_programmable_counter, kProgrammableCounterRiskThreshold,
        UINT32_MAX - validation.maximum_programmable_counter
    );
    std::fprintf(
        output,
        "\"owner\":{\"mode\":\"main_aicpu_path_a\","
        "\"snapshot_phase\":\"after_configure_before_restore\","
        "\"control_magic\":%u,\"control_version\":%u,\"configure_status\":%d,"
        "\"configured_flag\":%u,\"configured_bitmap_count\":%u,"
        "\"expected\":{\"total\":%u,\"aic\":%u,\"aiv\":%u},"
        "\"active\":{\"total\":%u,\"aic\":%u,\"aiv\":%u},"
        "\"discovered\":{\"total\":%u,\"aic\":%u,\"aiv\":%u},"
        "\"physical_slots_scanned\":%u,\"skipped_physical_slots\":%u,"
        "\"configured_bitmap_word_order\":\"least_significant_physical_ids_first\","
        "\"configured_bitmap_words\":[%u,%u,%u,%u],"
        "\"configured_complete_mixed_triplets\":%u,\"expected_complete_mixed_triplets\":%u,"
        "\"configured_broken_mixed_triplets\":%u,\"restore_passed\":%s},\n",
        owner.magic, owner.version, static_cast<int>(owner.status), owner.configured, owner_bitmap_count,
        owner.expected_total, owner.expected_aic, owner.expected_aiv,
        owner.active_total, owner.active_aic, owner.active_aiv,
        owner.discovered_total, owner.discovered_aic, owner.discovered_aiv,
        pa_scheduler::pmu_owner::kPhysicalSubcoreCount, owner.skipped_total,
        owner.configured_bitmap[0], owner.configured_bitmap[1],
        owner.configured_bitmap[2], owner.configured_bitmap[3],
        owner_complete_triplets, pa_scheduler::kAicWorkers,
        owner_bitmap_count / 3U - owner_complete_triplets, restore_passed ? "true" : "false"
    );
    std::fputs("\"records\":[\n", output);
    for (uint32_t worker = 0; worker < pa_scheduler::kWorkers; ++worker) {
        const pa_scheduler::WorkerResult &result = state.results[worker];
        const uint32_t status = result.pmu_status;
        const uint32_t physical_core_id = StatusCoreId(status);
        const bool trusted = (status & kStatusRequired) == kStatusRequired;
        const bool is_aic = result.role == static_cast<uint32_t>(pa_scheduler::CoreRole::Aic);
        const uint32_t vector_id = is_aic ? 0U : worker - pa_scheduler::kAicWorkers;
        const uint32_t block_id = is_aic ? worker : vector_id / 2U;
        const uint32_t lane = is_aic ? 0U : 1U + vector_id % 2U;
        const uint32_t segments = PmuWindowSegments(pmu);
        const bool owner_bitmap_member = pa_scheduler::pmu_owner::IsConfigured(owner, physical_core_id);
        const bool worker_slot_exact = result.worker_id == worker;
        const bool physical_role_matches =
            is_aic == pa_scheduler::pmu_owner::IsAicPhysicalSlot(physical_core_id);
        const bool window_started = (status & kStatusWindowStarted) != 0U;
        const bool window_stopped = (status & kStatusWindowStopped) != 0U;
        const bool icache_pair_observed = (status & kStatusIcachePairObserved) != 0U;
        std::fprintf(
            output,
            "%s{\"worker_id\":%u,\"physical_core_id\":%u,\"role\":\"%s\",\"block_id\":%u,"
            "\"lane\":%u,\"window_segments\":%u,\"window_segment_count_source\":\"mode_contract\","
            "\"window_started\":%s,\"window_stopped\":%s,\"total_cycles\":%llu,\"vector_busy\":%u,"
            "\"cube_busy\":%u,\"scalar_busy\":%u,\"mte1_busy\":%u,\"mte2_busy\":%u,"
            "\"mte3_busy\":%u,\"icache_requests\":%u,\"icache_misses\":%u,\"fix_busy\":%u,"
            "\"window_ticks\":%llu,\"warm_total_cycles\":%llu,\"warm_window_ticks\":%llu,"
            "\"warm_icache_requests\":%u,\"warm_icache_misses\":%u,"
            "\"status\":%u,\"status_hex\":"
            "\"0x%08x\",\"trusted\":%s,\"physical_core_id_valid\":%s,\"selectors_match\":%s,"
            "\"owner_bitmap_member\":%s,\"worker_slot_exact\":%s,"
            "\"physical_role_matches\":%s,\"icache_pair_observed\":%s,"
            "\"prior_snapshot_larger\":%s}",
            worker == 0 ? "" : ",\n", worker, physical_core_id, is_aic ? "aic" : "aiv", block_id,
            lane, segments, window_started ? "true" : "false", window_stopped ? "true" : "false",
            static_cast<unsigned long long>(result.pmu_total_cycles), result.pmu_vector_busy,
            result.pmu_cube_busy, result.pmu_scalar_busy, result.pmu_mte1_busy, result.pmu_mte2_busy,
            result.pmu_mte3_busy, result.pmu_icache_requests, result.pmu_icache_misses,
            result.pmu_fix_busy, static_cast<unsigned long long>(result.pmu_window_ticks),
            static_cast<unsigned long long>(result.pmu_warm_total_cycles),
            static_cast<unsigned long long>(result.pmu_warm_window_ticks),
            result.pmu_warm_icache_requests, result.pmu_warm_icache_misses,
            status, status, trusted ? "true" : "false",
            (status & kStatusCoreIdValid) != 0 ? "true" : "false",
            (status & (kStatusCnt0Selector | kStatusCnt1Selector | kStatusCnt2Selector |
                       kStatusCnt3Selector | kStatusCnt4Selector | kStatusCnt5Selector |
                       kStatusCnt6Selector | kStatusCnt7Selector | kStatusCnt8Selector)) ==
                    (kStatusCnt0Selector | kStatusCnt1Selector | kStatusCnt2Selector |
                     kStatusCnt3Selector | kStatusCnt4Selector | kStatusCnt5Selector |
                     kStatusCnt6Selector | kStatusCnt7Selector | kStatusCnt8Selector)
                ? "true"
                : "false",
            owner_bitmap_member ? "true" : "false", worker_slot_exact ? "true" : "false",
            physical_role_matches ? "true" : "false",
            icache_pair_observed ? "true" : "false",
            (status & kStatusPriorSnapshotLarger) != 0 ? "true" : "false"
        );
    }
    std::fputs("\n],\n\"summary\":{\"all\":", output);
    WritePmuAggregateJson(output, all, icache_single);
    std::fputs(",\"aic\":", output);
    WritePmuAggregateJson(output, aic, icache_single);
    std::fputs(",\"aiv\":", output);
    WritePmuAggregateJson(output, aiv, icache_single);
    std::fputs("}\n}\n", output);

    bool success = std::ferror(output) == 0;
    int write_error = success ? 0 : EIO;
    if (std::fflush(output) != 0) {
        success = false;
        write_error = errno;
    }
    if (success && fsync(fileno(output)) != 0) {
        success = false;
        write_error = errno;
    }
    if (std::fclose(output) != 0) {
        success = false;
        write_error = errno;
    }
    if (!success) {
        std::fprintf(stderr, "Failed while writing PMU JSON output %s: %s\n", temporary_path.c_str(),
                     std::strerror(write_error));
        (void)std::remove(temporary_path.c_str());
        return false;
    }
    // 同目录 hard-link 在最终名称不存在时原子发布；EEXIST 时保留既有证据，
    // 不采用会替换目标的 POSIX rename。
    if (link(temporary_path.c_str(), output_path.c_str()) != 0) {
        std::fprintf(
            stderr, "Cannot publish PMU JSON without replacement %s -> %s: %s\n",
            temporary_path.c_str(), output_path.c_str(),
            std::strerror(errno)
        );
        (void)std::remove(temporary_path.c_str());
        return false;
    }
    if (unlink(temporary_path.c_str()) != 0) {
        const int unlink_error = errno;
        // 最终文件已链接但事务尚未完成；尽力撤回最终名称，避免失败返回时留下
        // 一份被调用方误认为成功发布的文件。
        (void)unlink(output_path.c_str());
        std::fprintf(
            stderr, "Cannot remove PMU JSON temporary link %s: %s\n", temporary_path.c_str(),
            std::strerror(unlink_error)
        );
        return false;
    }
    std::printf("[PMU-JSON] capture_id=%s records=%u output=%s\n", capture_id.c_str(), pa_scheduler::kWorkers,
                output_path.c_str());
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    // 参数和 ELF 在创建 ACL 资源前完成校验，早期错误不会留下 device、stream 或 kernel handle。
    pa_scheduler::host::Options options;
    PmuOptions pmu_options;
    WinnerWorkloadOptions workload_options;
    std::vector<char *> pmu_argv;
    std::vector<char *> common_argv;
    if (!ParseWinnerWorkloadOptions(argc, argv, &workload_options, &pmu_argv) ||
        !ParsePmuOptions(
            static_cast<int>(pmu_argv.size()), pmu_argv.data(), &pmu_options, &common_argv
        )) {
        return EXIT_FAILURE;
    }
    const pa_scheduler::host::ParseStatus parse_status = pa_scheduler::host::ParseOptions(
        static_cast<int>(common_argv.size()), common_argv.data(), true, &options
    );
    if (parse_status != pa_scheduler::host::ParseStatus::Ok) {
        if (parse_status == pa_scheduler::host::ParseStatus::Help) {
            std::fprintf(
                stderr,
                "CCEC PMU options: [--pmu-window "
                "off|empty|scalar|scalar-double|icache-single|submit-all] "
                "[--pmu-scalar-nops N] [--pmu-icache-trials N] [--pmu-json FILE]\n"
            );
            std::fprintf(
                stderr,
                "CCEC winner workload options: [--winner-workload scalar-nop|real-compute] "
                "[--real-compute-count N | --real-compute-counts QK,SF,PV,UP] "
                "[--real-compute-pattern constant|layout-diagnostic]\n"
            );
            std::fprintf(
                stderr,
                "Default: real-compute, constant, counts=6,28,4,1; "
                "scalar-nop is the calibration compatibility mode.\n"
            );
        }
        return parse_status == pa_scheduler::host::ParseStatus::Help ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (!ValidateWinnerWorkloadOptions(workload_options)) return EXIT_FAILURE;
    if (!pmu_options.json_path.empty() &&
        pmu_options.mode == pa_scheduler::ccec_pmu::WindowMode::Off) {
        std::fprintf(stderr, "--pmu-json requires a non-off --pmu-window.\n");
        return EXIT_FAILURE;
    }
    if (!pmu_options.json_path.empty() && options.runs != 1) {
        // 一个 sidecar 对应一次采集，禁止多轮覆写后丢失逐轮边界。
        std::fprintf(stderr, "--pmu-json requires --runs 1 to avoid overwriting captures.\n");
        return EXIT_FAILURE;
    }
    if (!pmu_options.json_path.empty() &&
        (options.trace_enabled || options.trace_atomics || options.profile_phases ||
         options.analyze_swimlane || !options.swimlane_json.empty())) {
        std::fprintf(
            stderr,
            "--pmu-json requires PMU-only collection: add --no-swimlane and do not enable "
            "phase profiling, atomic tracing, swimlane analysis, or swimlane JSON.\n"
        );
        return EXIT_FAILURE;
    }
    if (!pmu_options.json_path.empty() &&
        (access(pmu_options.json_path.c_str(), F_OK) == 0 ||
         access((pmu_options.json_path + ".tmp").c_str(), F_OK) == 0)) {
        std::fprintf(
            stderr, "Refusing to overwrite an existing PMU JSON or temporary file: %s\n",
            pmu_options.json_path.c_str()
        );
        return EXIT_FAILURE;
    }
    const std::vector<char> binary_data = ReadBinary(options.kernel_path);
    if (binary_data.empty()) {
        std::fprintf(stderr, "Cannot read kernel binary: %s\n", options.kernel_path.c_str());
        return EXIT_FAILURE;
    }
    const bool real_compute =
        workload_options.mode == pa_scheduler::WinnerWorkloadMode::RealCompute;
    std::vector<float> workload_image;
    std::vector<float> workload_outputs;
    if (real_compute) {
        pa_scheduler::host::InitializeWinnerWorkloadBuffers(
            workload_options, &workload_image, &workload_outputs
        );
    }
    pa_scheduler::host::PrintBanner("CCEC", options);
    pa_scheduler::host::PrintWinnerWorkloadConfig(workload_options, options.nops);
    std::printf(
        "[PMU-CONFIG] window=%s calibration_scalar_nops=%u icache_trials=%u source=direct-per-core "
        "owner=main-aicpu-path-a\n",
        PmuModeName(pmu_options.mode), pmu_options.scalar_nops, pmu_options.icache_trials
    );

    // 正常及后处理路径依次完成 ACL 初始化、选卡、stream/ELF/设备区创建、launch/D2H
    // 和尾部清理；初始化、传输或 launch 的早期错误仍按当前实现就地返回。
    if (!CheckAcl(aclInit(nullptr), "aclInit") || !CheckAcl(aclrtSetDevice(options.device), "aclrtSetDevice")) {
        return EXIT_FAILURE;
    }
    aclrtStream stream = nullptr;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) return EXIT_FAILURE;

    rtDevBinary_t binary{RT_DEV_BINARY_MAGIC_ELF, 0, binary_data.data(), binary_data.size()};
    void *kernel_handle = nullptr;
    bool registered_all = true;
    // 先尝试注册带 mixed metadata 的 ELF；若 rtRegisterAllKernel 报错或未返回 handle，
    // 再尝试无 tiling-key 装载。这里仅描述实际回退条件，不假设具体运行时原因。
    rtError_t register_error = rtRegisterAllKernel(&binary, &kernel_handle);
    if (register_error != RT_ERROR_NONE || kernel_handle == nullptr) {
        registered_all = false;
        register_error = rtBinaryLoadWithoutTilingKey(binary_data.data(), binary_data.size(), &kernel_handle);
    }
    if (!CheckRt(register_error, "register mixed AICore ELF") || kernel_handle == nullptr) return EXIT_FAILURE;

    // SchedulerState 保留被测关键 offset、DistCore ABI 和约 1 GiB 生产总跨度；
    // 使用 HUGE_FIRST 降低大块设备内存碎片风险。
    void *state_device = nullptr;
    if (!CheckAcl(
            aclrtMalloc(&state_device, sizeof(pa_scheduler::SchedulerState), ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc(state)"
        )) {
        return EXIT_FAILURE;
    }
    if ((reinterpret_cast<uintptr_t>(state_device) & 63U) != 0) {
        std::fprintf(stderr, "Device state is not 64-byte aligned: %p\n", state_device);
        return EXIT_FAILURE;
    }

    // 真实 PTO 负载使用独立 GM，不解引用调度器中只用于依赖建模的 synthetic tensor 地址。
    // 这里先于 PMU owner 分配；每轮 H2D 初始化虽在 owner 配置之后，但仍位于
    // launch/wall 计时之前，因此两者都不进入 Submit 性能窗口。
    ScopedAclDeviceAllocation workload_allocation;
    if (real_compute &&
        !CheckAcl(
            aclrtMalloc(
                workload_allocation.Address(), pa_scheduler::winner_workload::kWorkspaceBytes,
                ACL_MEM_MALLOC_HUGE_FIRST
            ),
            "aclrtMalloc(real-compute workspace)"
        )) {
        return EXIT_FAILURE;
    }
    void *workload_device = workload_allocation.Get();
    if (real_compute && (reinterpret_cast<uintptr_t>(workload_device) & 63U) != 0) {
        std::fprintf(stderr, "Real-compute workspace is not 64-byte aligned: %p\n", workload_device);
        return EXIT_FAILURE;
    }

    // 泳道区按 96 worker 各 65536 条记录预留，约 384 MiB；关闭泳道时不申请，也不会传递有效 base。
    // 该分配先于 PMU owner 配置，失败时不会留下需要恢复的 selector/MMIO 会话。
    void *trace_device = nullptr;
    if (options.trace_enabled &&
        !CheckAcl(
            aclrtMalloc(&trace_device, pa_scheduler::kTraceBytes, ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc(swimlane trace)"
        )) {
        return EXIT_FAILURE;
    }
    if (options.trace_enabled && (reinterpret_cast<uintptr_t>(trace_device) & 63U) != 0) {
        std::fprintf(stderr, "Device swimlane trace is not 64-byte aligned: %p\n", trace_device);
        return EXIT_FAILURE;
    }

    PmuRegisterMappings pmu_mappings;
    pa_scheduler::pmu_owner::PmuOwnerSession pmu_owner;
    pa_scheduler::pmu_owner::PmuOwnerControl pmu_owner_evidence{};
    bool pmu_owner_evidence_valid = false;
    const void *pmu_registers_device = nullptr;
    if (pmu_options.mode != pa_scheduler::ccec_pmu::WindowMode::Off) {
        if (!MapPmuRegisters(options.device, &pmu_mappings)) return EXIT_FAILURE;
        const std::string dispatcher_path = pa_scheduler::pmu_owner::ArtifactBesideKernel(
            options.kernel_path, "libpa_scheduler_pmu_owner_dispatcher.so"
        );
        const std::string owner_path = pa_scheduler::pmu_owner::ArtifactBesideKernel(
            options.kernel_path, "libpa_scheduler_pmu_owner_aicpu.so"
        );
        if (!pmu_owner.Initialize(
                options.device, stream, dispatcher_path, owner_path, pmu_mappings.register_bases
            ) ||
            !pmu_owner.Configure()) {
            (void)pmu_owner.Finalize();
            (void)UnmapPmuRegisters(options.device, &pmu_mappings);
            return EXIT_FAILURE;
        }
        pmu_owner_evidence = pmu_owner.Control();
        pmu_owner_evidence_valid = true;
        pmu_registers_device = reinterpret_cast<const void *>(pmu_owner.RegisterTableDeviceAddress());
    }

    // host shadow 保留约 1 GiB 总跨度以便按关键 offset 寻址，但每轮传输只选择
    // 共享前缀、控制区和结果区。
    std::unique_ptr<pa_scheduler::SchedulerState> state(new pa_scheduler::SchedulerState);
    pa_scheduler::TraceHeader trace_header{};
    std::vector<double> spans;
    bool execution_ok = true;
    bool all_passed = true;
    bool postprocess_ok = true;
    bool pmu_json_ready = false;
    bool pmu_json_semantic_passed = false;
    bool pmu_json_workload_output_passed = false;
    uint32_t pmu_json_run = 0U;
    double pmu_json_host_us = 0.0;
    double pmu_json_submit_span_us = 0.0;
    PmuValidation pmu_json_validation;
    for (uint32_t run = 1; run <= options.runs; ++run) {
        pa_scheduler::host::InitializeState(state.get(), options);
        pa_scheduler::host::ConfigureTrace(state.get(), options, trace_device);
        ConfigurePmu(state.get(), pmu_options, pmu_registers_device);
        ConfigureWinnerWorkload(state.get(), workload_options, workload_device);
        if (real_compute &&
            !CheckAcl(
                aclrtMemcpy(
                    workload_device, pa_scheduler::winner_workload::kWorkspaceBytes,
                    workload_image.data(), pa_scheduler::winner_workload::kWorkspaceBytes,
                    ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "aclrtMemcpy(H2D real-compute workspace)"
            )) {
            execution_ok = false;
            break;
        }
        if (options.trace_enabled) {
            // 每轮只需重置约 7 KiB header；各 worker 会从 count=0 覆盖自己的记录区，无需清零整块 384 MiB。
            pa_scheduler::host::InitializeTraceHeader(&trace_header);
            if (!CheckAcl(
                    aclrtMemcpy(
                        trace_device, sizeof(trace_header), &trace_header, sizeof(trace_header),
                        ACL_MEMCPY_HOST_TO_DEVICE
                    ),
                    "aclrtMemcpy(H2D swimlane header)"
                )) {
                execution_ok = false;
                break;
            }
        }
        // 为避免每轮搬运约 1 GiB，只 H2D 被测共享前缀和位于生产总跨度之后的
        // standalone 控制区；
        // 每个 worker 的大块私有状态由 device kernel 自行初始化。
        if (!CheckAcl(
                aclrtMemcpy(
                    state_device, pa_scheduler::host::StatePrefixBytes(), state.get(),
                    pa_scheduler::host::StatePrefixBytes(), ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "aclrtMemcpy(H2D state prefix)"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    &static_cast<pa_scheduler::SchedulerState *>(state_device)->config,
                    pa_scheduler::host::ControlBytes(), &state->config, pa_scheduler::host::ControlBytes(),
                    ACL_MEMCPY_HOST_TO_DEVICE
                ),
                "aclrtMemcpy(H2D standalone controls)"
            )) {
            execution_ok = false;
            break;
        }
        void *kernel_args[] = {state_device};
        rtArgsEx_t args_info{};
        args_info.args = kernel_args;
        args_info.argsSize = sizeof(kernel_args);
        rtTaskCfgInfo_t task_config{};
        // launch 维度是 32 个物理 mixed block；ELF metadata 让每个 block 同时产生 1 AIC + 2 AIV，共 96 worker。
        // wall time 在同步完成处截止，包含 launch、完整调度、最终 drain 和 stream 同步，但不包含后续 D2H/JSON。
        const auto wall_begin = std::chrono::steady_clock::now();
        if (!CheckRt(
                rtKernelLaunchWithHandleV2(
                    kernel_handle, 0, pa_scheduler::kAicWorkers, &args_info, nullptr, stream, &task_config
                ),
                "rtKernelLaunchWithHandleV2"
            ) ||
            !CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
            execution_ok = false;
            break;
        }
        const auto wall_end = std::chrono::steady_clock::now();
        const double host_us = std::chrono::duration<double, std::micro>(wall_end - wall_begin).count();
        // D2H 同样避开约 1 GiB 的 worker arena：共享前缀用于 flag/vend/frontier 校验，末尾 results 单独回传。
        if (!CheckAcl(
                aclrtMemcpy(
                    state.get(), pa_scheduler::host::StatePrefixBytes(), state_device,
                    pa_scheduler::host::StatePrefixBytes(), ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "aclrtMemcpy(D2H state prefix)"
            ) ||
            !CheckAcl(
                aclrtMemcpy(
                    state->results, pa_scheduler::host::ResultBytes(),
                    &static_cast<pa_scheduler::SchedulerState *>(state_device)->results[0],
                    pa_scheduler::host::ResultBytes(), ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "aclrtMemcpy(D2H worker results)"
            )) {
            execution_ok = false;
            break;
        }
        if (real_compute &&
            !CheckAcl(
                aclrtMemcpy(
                    workload_outputs.data(),
                    static_cast<size_t>(pa_scheduler::winner_workload::kOutputTiles) *
                        pa_scheduler::winner_workload::kTileBytes,
                    static_cast<uint8_t *>(workload_device) +
                        pa_scheduler::winner_workload::kSharedInputTiles *
                            pa_scheduler::winner_workload::kTileBytes,
                    static_cast<size_t>(pa_scheduler::winner_workload::kOutputTiles) *
                        pa_scheduler::winner_workload::kTileBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "aclrtMemcpy(D2H real-compute outputs)"
            )) {
            execution_ok = false;
            break;
        }
        if (options.trace_enabled &&
            !CheckAcl(
                aclrtMemcpy(
                    &trace_header, sizeof(trace_header), trace_device, sizeof(trace_header),
                    ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "aclrtMemcpy(D2H swimlane header)"
            )) {
            execution_ok = false;
            break;
        }
        // 常规校验只需 header 中的 per-worker count；真实 records 在分析或导出时才按核、按实际 count 懒加载。
        const auto read_trace_records =
            [trace_device](uint32_t worker, uint32_t count, pa_scheduler::TraceRecord *records) {
                // 每核记录区采用固定容量 stride；只复制 header 声明的实际 count，避免 D2H 未使用的尾部空间。
                const uint64_t offset = sizeof(pa_scheduler::TraceHeader) +
                                        static_cast<uint64_t>(worker) * pa_scheduler::kTraceRecordsPerCore *
                                            sizeof(pa_scheduler::TraceRecord);
                return CheckAcl(
                    aclrtMemcpy(
                        records, static_cast<size_t>(count) * sizeof(pa_scheduler::TraceRecord),
                        static_cast<uint8_t *>(trace_device) + offset,
                        static_cast<size_t>(count) * sizeof(pa_scheduler::TraceRecord),
                        ACL_MEMCPY_DEVICE_TO_HOST
                    ),
                    "aclrtMemcpy(D2H swimlane records)"
                );
            };
        // 先完成共享状态、拓扑、计数和 trace header 的语义校验，再允许 raw JSON 成为性能证据。
        const pa_scheduler::host::Metrics metrics = pa_scheduler::host::Validate(
            *state, run, host_us, options.trace_enabled ? &trace_header : nullptr
        );
        all_passed &= metrics.passed;
        const bool workload_passed =
            !real_compute || ValidateRealComputeOutputs(
                *state, workload_options, workload_outputs, run
            );
        all_passed &= workload_passed;
        PmuValidation pmu_validation;
        const bool pmu_passed = ValidatePmu(
            *state, run, pmu_options, workload_options,
            pmu_options.mode == pa_scheduler::ccec_pmu::WindowMode::Off ? nullptr : &pmu_owner.Control(),
            &pmu_validation
        );
        all_passed &= pmu_passed;
        spans.push_back(metrics.submit_span_us);
        if (!pmu_options.json_path.empty()) {
            if (!metrics.passed || !workload_passed || !pmu_passed || !pmu_owner_evidence_valid) {
                std::fprintf(stderr, "PMU JSON rejected because semantic, PMU, or owner validation failed.\n");
                postprocess_ok = false;
                break;
            }
            pmu_json_ready = true;
            pmu_json_semantic_passed = metrics.passed && workload_passed;
            pmu_json_workload_output_passed = workload_passed;
            pmu_json_run = run;
            pmu_json_host_us = host_us;
            pmu_json_submit_span_us = metrics.submit_span_us;
            pmu_json_validation = pmu_validation;
        }
        if (options.analyze_swimlane &&
            !pa_scheduler::host::AnalyzeSwimlaneRecords(trace_header, *state, read_trace_records)) {
            // 后处理错误使用 break 汇入统一 cleanup；与初始化/launch 失败的进程级立即返回语义区分开。
            postprocess_ok = false;
            break;
        }
        if (!options.swimlane_json.empty()) {
            // 只有语义校验通过才把 raw JSON 经“临时文件写完后 rename”发布，
            // 避免把截断或错误调度结果误当成可用性能证据。
            if (!metrics.passed || !workload_passed) {
                std::fprintf(stderr, "Skipping swimlane export because semantic validation failed.\n");
                postprocess_ok = false;
                break;
            }
            if (!pa_scheduler::host::ExportSwimlaneRecords(
                    trace_header, options.swimlane_json, workload_options.mode,
                    real_compute
                        ? workload_options.repeats
                        : pa_scheduler::WorkloadCounts{
                              options.nops.qk, options.nops.sf, options.nops.pv, options.nops.up
                          },
                    real_compute
                        ? pa_scheduler::host::RealComputePatternName(workload_options.pattern)
                        : "none",
                    read_trace_records
                )) {
                postprocess_ok = false;
                break;
            }
        }
    }

    const double median_submit_span_us = spans.empty() ? 0.0 : pa_scheduler::host::Median(spans);
    std::printf(
        "[SUMMARY] runs=%u completed_runs=%zu median_submit_span_us=%.3f "
        "execution_status=%s semantic_status=%s postprocess_status=%s\n",
        options.runs, spans.size(), median_submit_span_us, execution_ok ? "PASS" : "FAIL",
        all_passed ? "PASS" : "FAIL", postprocess_ok ? "PASS" : "FAIL"
    );

    // 后处理失败也统一走设备资源释放、ELF 卸载和 ACL 收尾，避免文件系统错误遗留运行时上下文。
    bool cleanup_ok = true;
    bool pmu_owner_restore_ok = true;
    // 先释放依赖当前 device/context 的大块内存，再卸载 ELF、销毁 stream，最后 reset device 与 finalize ACL。
    if (trace_device != nullptr) {
        cleanup_ok &= CheckAcl(aclrtFree(trace_device), "aclrtFree(swimlane trace)");
    }
    if (pmu_registers_device != nullptr) {
        // owner 必须在 MMIO 映射、device context 和 ACL runtime 仍有效时恢复。
        pmu_owner_restore_ok = pmu_owner.Finalize();
        cleanup_ok &= pmu_owner_restore_ok;
        cleanup_ok &= UnmapPmuRegisters(options.device, &pmu_mappings);
    }
    if (workload_device != nullptr) {
        cleanup_ok &= CheckAcl(
            aclrtFree(workload_allocation.Release()), "aclrtFree(real-compute workspace)"
        );
    }
    cleanup_ok &= CheckAcl(aclrtFree(state_device), "aclrtFree(state)");
    const rtError_t unload_error =
        registered_all ? rtDevBinaryUnRegister(kernel_handle) : rtBinaryUnLoad(kernel_handle);
    cleanup_ok &= CheckRt(unload_error, "unload mixed AICore ELF");
    cleanup_ok &= CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream");
    cleanup_ok &= CheckAcl(aclrtResetDevice(options.device), "aclrtResetDevice");
    cleanup_ok &= CheckAcl(aclFinalize(), "aclFinalize");
    if (!pmu_options.json_path.empty()) {
        if (!pmu_json_ready || !all_passed || !postprocess_ok || !cleanup_ok || !pmu_owner_restore_ok) {
            std::fprintf(stderr, "PMU JSON was not published because the capture or restore transaction failed.\n");
            postprocess_ok = false;
        } else if (!ExportPmuJson(
                       *state, options, pmu_options, workload_options, pmu_json_run,
                       pmu_json_host_us, pmu_json_submit_span_us, pmu_json_validation,
                       pmu_json_semantic_passed, pmu_json_workload_output_passed,
                       pmu_owner_evidence,
                       pmu_owner_restore_ok, pmu_options.json_path
                   )) {
            postprocess_ok = false;
        }
    }
    // 运行语义、后处理和资源清理三者全部成功，进程才返回成功，脚本据此决定是否继续生成 merged 泳道。
    return execution_ok && all_passed && postprocess_ok && cleanup_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
