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
#include "pmu_owner_host.h"
#include "pmu_probe.h"

#include "acl/acl.h"
#include "driver/ascend_hal.h"
#include "runtime/rt.h"

#include <chrono>
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
};

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
    }
    return "invalid";
}

bool ParsePmuOptions(int argc, char **argv, PmuOptions *pmu, std::vector<char *> *common_argv) {
    // PMU 参数只属于 CCEC 验证分支；先摘出再交给三后端共享 parser，避免 CPU/AscendC 静默接受却不生效。
    bool mode_seen = false;
    bool nops_seen = false;
    common_argv->clear();
    common_argv->push_back(argv[0]);
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument != "--pmu-window" && argument != "--pmu-scalar-nops") {
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
            } else {
                std::fprintf(
                    stderr, "Invalid --pmu-window value: %s (expected off|empty|scalar|scalar-double)\n", value
                );
                return false;
            }
            mode_seen = true;
        } else {
            if (nops_seen || !pa_scheduler::host::ParseUint(value, 0, 10000000, &pmu->scalar_nops)) {
                std::fprintf(stderr, "Invalid or duplicate --pmu-scalar-nops value: %s\n", value);
                return false;
            }
            nops_seen = true;
        }
    }
    if (nops_seen && pmu->mode != pa_scheduler::ccec_pmu::WindowMode::Scalar &&
        pmu->mode != pa_scheduler::ccec_pmu::WindowMode::ScalarDouble) {
        std::fprintf(stderr, "--pmu-scalar-nops requires a scalar PMU window.\n");
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
    state->config.reserved[kConfigScalarNops] = pmu.scalar_nops;
    StorePointer(state->config.reserved, register_table);
    state->config.reserved[kConfigMagic] = pmu.mode == WindowMode::Off ? 0 : kConfigMagicValue;
}

struct PmuAggregate {
    std::vector<uint64_t> total_cycles;
    uint64_t scalar_busy = 0;
    uint64_t icache_requests = 0;
    uint64_t icache_misses = 0;
};

void AddPmuSample(const pa_scheduler::WorkerResult &result, PmuAggregate *aggregate) {
    aggregate->total_cycles.push_back(result.pmu_total_cycles);
    aggregate->scalar_busy += result.pmu_scalar_busy;
    aggregate->icache_requests += result.pmu_icache_requests;
    aggregate->icache_misses += result.pmu_icache_misses;
}

void PrintPmuAggregate(const char *name, const PmuAggregate &aggregate) {
    const pa_scheduler::host::Uint64Distribution total =
        pa_scheduler::host::SummarizeUint64(aggregate.total_cycles);
    const double miss_rate = aggregate.icache_requests == 0
        ? 0.0
        : 100.0 * aggregate.icache_misses / aggregate.icache_requests;
    std::printf(
        "[PMU-%s] cores=%zu total_sum=%llu total_median=%.1f total_p95=%llu scalar_busy=%llu "
        "icache_req=%llu icache_miss=%llu miss_rate=%.4f%%\n",
        name, aggregate.total_cycles.size(), static_cast<unsigned long long>(total.total), total.median,
        static_cast<unsigned long long>(total.p95), static_cast<unsigned long long>(aggregate.scalar_busy),
        static_cast<unsigned long long>(aggregate.icache_requests),
        static_cast<unsigned long long>(aggregate.icache_misses), miss_rate
    );
}

bool ValidatePmu(
    const pa_scheduler::SchedulerState &state, uint32_t run, const PmuOptions &pmu,
    const pa_scheduler::pmu_owner::PmuOwnerControl *owner
) {
    using namespace pa_scheduler::ccec_pmu;
    if (pmu.mode == WindowMode::Off) return true;

    bool seen[kPhysicalSubcoreCount] = {};
    uint32_t trusted = 0;
    uint32_t unique = 0;
    uint32_t owner_members = 0;
    uint32_t exact_worker_slots = 0;
    uint32_t physical_role_matches = 0;
    uint32_t prior_larger = 0;
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
        prior_larger += (status & kStatusPriorSnapshotLarger) != 0;
        if (core_id < kPhysicalSubcoreCount && !seen[core_id]) {
            seen[core_id] = true;
            ++unique;
        }
        if (!record_trusted && bad_printed < 8) {
            std::printf(
                "[PMU-BAD] worker=%u role=%llu coreid=%u status=0x%08x total=%llu scalar=%u req=%u miss=%u\n",
                worker, static_cast<unsigned long long>(result.role), core_id, status,
                static_cast<unsigned long long>(result.pmu_total_cycles), result.pmu_scalar_busy,
                result.pmu_icache_requests, result.pmu_icache_misses
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
        const uint32_t aiv0_id = StatusCoreId(
            state.results[pa_scheduler::kAicWorkers + block * 2U].pmu_status
        );
        const uint32_t aiv1_id = StatusCoreId(
            state.results[pa_scheduler::kAicWorkers + block * 2U + 1U].pmu_status
        );
        mixed_triplet_matches += aiv0_id == expected_aiv0 && aiv1_id == expected_aiv0 + 1U;
    }

    PrintPmuAggregate("ALL", all);
    PrintPmuAggregate("AIC", aic);
    PrintPmuAggregate("AIV", aiv);
    const bool records_ok = trusted == pa_scheduler::kWorkers;
    const bool core_ids_ok = unique == pa_scheduler::kWorkers;
    const bool owner_members_ok = owner_members == pa_scheduler::kWorkers;
    const bool worker_slots_ok = exact_worker_slots == pa_scheduler::kWorkers;
    const bool physical_roles_ok = physical_role_matches == pa_scheduler::kWorkers;
    const bool mixed_triplets_ok = mixed_triplet_matches == pa_scheduler::kAicWorkers;
    std::printf(
        "[PMU] run=%u window=%s scalar_nops=%u trusted=%u/%u unique_coreids=%u/%u prior_larger=%u/%u\n", run,
        PmuModeName(pmu.mode), pmu.scalar_nops, trusted, pa_scheduler::kWorkers, unique, pa_scheduler::kWorkers,
        prior_larger, pa_scheduler::kWorkers
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
    return records_ok && core_ids_ok && owner_members_ok && worker_slots_ok &&
        physical_roles_ok && mixed_triplets_ok;
}

}  // namespace

int main(int argc, char **argv) {
    // 参数和 ELF 在创建 ACL 资源前完成校验，早期错误不会留下 device、stream 或 kernel handle。
    pa_scheduler::host::Options options;
    PmuOptions pmu_options;
    std::vector<char *> common_argv;
    if (!ParsePmuOptions(argc, argv, &pmu_options, &common_argv)) return EXIT_FAILURE;
    const pa_scheduler::host::ParseStatus parse_status = pa_scheduler::host::ParseOptions(
        static_cast<int>(common_argv.size()), common_argv.data(), true, &options
    );
    if (parse_status != pa_scheduler::host::ParseStatus::Ok) {
        if (parse_status == pa_scheduler::host::ParseStatus::Help) {
            std::fprintf(
                stderr,
                "CCEC PMU options: [--pmu-window off|empty|scalar|scalar-double] [--pmu-scalar-nops N]\n"
            );
        }
        return parse_status == pa_scheduler::host::ParseStatus::Help ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    const std::vector<char> binary_data = ReadBinary(options.kernel_path);
    if (binary_data.empty()) {
        std::fprintf(stderr, "Cannot read kernel binary: %s\n", options.kernel_path.c_str());
        return EXIT_FAILURE;
    }
    pa_scheduler::host::PrintBanner("CCEC", options);
    std::printf(
        "[PMU-CONFIG] window=%s scalar_nops=%u source=direct-per-core owner=main-aicpu-path-a\n",
        PmuModeName(pmu_options.mode), pmu_options.scalar_nops
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

    PmuRegisterMappings pmu_mappings;
    pa_scheduler::pmu_owner::PmuOwnerSession pmu_owner;
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
        pmu_registers_device = reinterpret_cast<const void *>(pmu_owner.RegisterTableDeviceAddress());
    }

    // 泳道区按 96 worker 各 65536 条记录预留，约 384 MiB；关闭泳道时不申请，也不会传递有效 base。
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

    // host shadow 保留约 1 GiB 总跨度以便按关键 offset 寻址，但每轮传输只选择
    // 共享前缀、控制区和结果区。
    std::unique_ptr<pa_scheduler::SchedulerState> state(new pa_scheduler::SchedulerState);
    pa_scheduler::TraceHeader trace_header{};
    std::vector<double> spans;
    bool all_passed = true;
    bool postprocess_ok = true;
    for (uint32_t run = 1; run <= options.runs; ++run) {
        pa_scheduler::host::InitializeState(state.get(), options);
        pa_scheduler::host::ConfigureTrace(state.get(), options, trace_device);
        ConfigurePmu(state.get(), pmu_options, pmu_registers_device);
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
                return EXIT_FAILURE;
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
            return EXIT_FAILURE;
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
            return EXIT_FAILURE;
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
            return EXIT_FAILURE;
        }
        if (options.trace_enabled &&
            !CheckAcl(
                aclrtMemcpy(
                    &trace_header, sizeof(trace_header), trace_device, sizeof(trace_header),
                    ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "aclrtMemcpy(D2H swimlane header)"
            )) {
            return EXIT_FAILURE;
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
        all_passed &= ValidatePmu(
            *state, run, pmu_options,
            pmu_options.mode == pa_scheduler::ccec_pmu::WindowMode::Off ? nullptr : &pmu_owner.Control()
        );
        spans.push_back(metrics.submit_span_us);
        if (options.analyze_swimlane &&
            !pa_scheduler::host::AnalyzeSwimlaneRecords(trace_header, *state, read_trace_records)) {
            // 后处理错误使用 break 汇入统一 cleanup；与初始化/launch 失败的进程级立即返回语义区分开。
            postprocess_ok = false;
            break;
        }
        if (!options.swimlane_json.empty()) {
            // 只有语义校验通过才把 raw JSON 经“临时文件写完后 rename”发布，
            // 避免把截断或错误调度结果误当成可用性能证据。
            if (!metrics.passed) {
                std::fprintf(stderr, "Skipping swimlane export because semantic validation failed.\n");
                postprocess_ok = false;
                break;
            }
            if (!pa_scheduler::host::ExportSwimlaneRecords(
                    trace_header, options.swimlane_json, read_trace_records
                )) {
                postprocess_ok = false;
                break;
            }
        }
    }

    std::printf(
        "[SUMMARY] runs=%u median_submit_span_us=%.3f semantic_status=%s postprocess_status=%s\n", options.runs,
        pa_scheduler::host::Median(spans), all_passed ? "PASS" : "FAIL", postprocess_ok ? "PASS" : "FAIL"
    );

    // 后处理失败也统一走设备资源释放、ELF 卸载和 ACL 收尾，避免文件系统错误遗留运行时上下文。
    bool cleanup_ok = true;
    // 先释放依赖当前 device/context 的大块内存，再卸载 ELF、销毁 stream，最后 reset device 与 finalize ACL。
    if (trace_device != nullptr) {
        cleanup_ok &= CheckAcl(aclrtFree(trace_device), "aclrtFree(swimlane trace)");
    }
    if (pmu_registers_device != nullptr) {
        // owner 必须在 MMIO 映射、device context 和 ACL runtime 仍有效时恢复。
        cleanup_ok &= pmu_owner.Finalize();
        cleanup_ok &= UnmapPmuRegisters(options.device, &pmu_mappings);
    }
    cleanup_ok &= CheckAcl(aclrtFree(state_device), "aclrtFree(state)");
    const rtError_t unload_error =
        registered_all ? rtDevBinaryUnRegister(kernel_handle) : rtBinaryUnLoad(kernel_handle);
    cleanup_ok &= CheckRt(unload_error, "unload mixed AICore ELF");
    cleanup_ok &= CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream");
    cleanup_ok &= CheckAcl(aclrtResetDevice(options.device), "aclrtResetDevice");
    cleanup_ok &= CheckAcl(aclFinalize(), "aclFinalize");
    // 运行语义、后处理和资源清理三者全部成功，进程才返回成功，脚本据此决定是否继续生成 merged 泳道。
    return all_passed && postprocess_ok && cleanup_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
