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

// 单 AIV、无轮询、固定操作数的 atomic/标量 PMU 对照 host。
// 每个 rounds 都重复 EMPTY -> SCALAR_CONTROL -> DEPENDENT_ATOMIC_ADD：
//   1. EMPTY 给出同位置 PMU gate/read 的固定成本；
//   2. SCALAR_CONTROL 执行与 atomic 路径完全相同的返回值递推和 checksum；
//   3. DEPENDENT_ATOMIC_ADD 让下一条 atomicAdd 的 addend 依赖上一条返回值。
// 因而 (ATOMIC-CONTROL)/rounds 直接回答 atomic 等待周期落在 PMU total、scalar busy
// 中的哪一项，而不混入多核竞争、轮询次数变化或未消费返回值的并行发射。

#include "atomic_scalar_pmu_shared.h"
#include "pmu_probe_control.h"
#include "../probe_host.h"

#include "aicpu_loader/host/load_aicpu_op.h"
#include "common/kernel_args.h"
#include "driver/ascend_hal.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kPhysicalAicoreCount = 36;
constexpr uint32_t kSubcoresPerAicore = 3;
constexpr uint32_t kPhysicalSubcoreCount = kPhysicalAicoreCount * kSubcoresPerAicore;
constexpr uint32_t kAicorePerDie = 18;
constexpr uint32_t kSubcoresPerDie = kAicorePerDie * kSubcoresPerAicore;
constexpr uint32_t kAivBaseInDie = kAicorePerDie;
constexpr uint64_t kSubcoreStride = 0x100000ULL;
constexpr uint32_t kAicoreMapBytes = 0x300000U;

static_assert(
    kPhysicalSubcoreCount == atomic_probe::pmu::kPmuPhysicalSubcores, "PMU table size mismatch"
);

bool CheckAcl(aclError error, const char *label) {
    if (error == ACL_SUCCESS) return true;
    std::fprintf(stderr, "ACL error %d: %s\n", static_cast<int>(error), label);
    return false;
}

std::vector<char> ReadBinary(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0) return {};
    std::vector<char> bytes(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(bytes.data(), size)) return {};
    return bytes;
}

std::string ArtifactBesideKernel(const std::string &kernel_path, const char *name) {
    const size_t slash = kernel_path.find_last_of('/');
    return slash == std::string::npos ? std::string(name) : kernel_path.substr(0, slash + 1) + name;
}

class PmuResources {
public:
    using MapFn = drvError_t (*)(unsigned int, struct res_map_info *, unsigned long *, unsigned int *);
    using UnmapFn = drvError_t (*)(unsigned int, struct res_map_info *);

    ~PmuResources() { RestoreAndUnmap(); }

    bool Initialize(uint32_t device) {
        device_ = device;
        map_ = reinterpret_cast<MapFn>(dlsym(RTLD_DEFAULT, "halResMap"));
        unmap_ = reinterpret_cast<UnmapFn>(dlsym(RTLD_DEFAULT, "halResUnmap"));
        if (map_ == nullptr || unmap_ == nullptr) {
            hal_handle_ = dlopen("libascend_hal.so", RTLD_NOW | RTLD_GLOBAL);
            if (hal_handle_ != nullptr) {
                map_ = reinterpret_cast<MapFn>(dlsym(hal_handle_, "halResMap"));
                unmap_ = reinterpret_cast<UnmapFn>(dlsym(hal_handle_, "halResUnmap"));
            }
        }
        if (map_ == nullptr || unmap_ == nullptr) {
            std::fprintf(stderr, "Cannot resolve halResMap/halResUnmap.\n");
            return false;
        }

        for (uint32_t physical = 0; physical < kPhysicalAicoreCount; ++physical) {
            res_map_info &info = map_info_[physical];
            std::memset(&info, 0, sizeof(info));
            info.target_proc_type = PROCESS_CP1;
            info.res_type = RES_AICORE;
            info.res_id = physical;
            unsigned long map_address = 0;
            unsigned int map_length = kAicoreMapBytes;
            const drvError_t error = map_(device_, &info, &map_address, &map_length);
            if (error != 0 || map_address == 0 || map_length < kAicoreMapBytes) {
                std::fprintf(
                    stderr, "halResMap failed: physical=%u error=%d address=0x%lx length=%u\n", physical,
                    static_cast<int>(error), map_address, map_length
                );
                Unmap();
                return false;
            }
            ++mapped_count_;
            const uint64_t base = static_cast<uint64_t>(map_address);
            const uint32_t die = physical / kAicorePerDie;
            const uint32_t local = physical % kAicorePerDie;
            const uint32_t die_base = die * kSubcoresPerDie;
            register_bases_[die_base + local] = base;
            const uint32_t aiv0 = die_base + kAivBaseInDie + local * 2;
            register_bases_[aiv0] = base + kSubcoreStride;
            register_bases_[aiv0 + 1] = base + 2 * kSubcoreStride;
        }
        return true;
    }

    void RestoreAndUnmap() {
        Unmap();
        if (hal_handle_ != nullptr) {
            dlclose(hal_handle_);
            hal_handle_ = nullptr;
        }
    }

    const std::array<uint64_t, kPhysicalSubcoreCount> &RegisterBases() const { return register_bases_; }

private:
    void Unmap() {
        while (mapped_count_ != 0) {
            --mapped_count_;
            const drvError_t error = unmap_(device_, &map_info_[mapped_count_]);
            if (error != 0) {
                std::fprintf(
                    stderr, "halResUnmap failed: physical=%u error=%d\n", mapped_count_,
                    static_cast<int>(error)
                );
            }
        }
    }

    uint32_t device_ = 0;
    uint32_t mapped_count_ = 0;
    void *hal_handle_ = nullptr;
    MapFn map_ = nullptr;
    UnmapFn unmap_ = nullptr;
    std::array<res_map_info, kPhysicalAicoreCount> map_info_{};
    std::array<uint64_t, kPhysicalSubcoreCount> register_bases_{};
};

bool RunPmuCommand(
    host::LoadAicpuOp &loader, aclrtStream stream, KernelArgs *kernel_args, void *control_device,
    atomic_probe::pmu::PmuControl *control, atomic_probe::pmu::PmuCommand command
) {
    control->command = static_cast<uint32_t>(command);
    control->status = atomic_probe::pmu::kPmuStatusPending;
    kernel_args->enable_profiling_flag = static_cast<uint32_t>(command);
    const int launch_error = loader.LaunchBuiltInOp(stream, kernel_args, 1, host::KernelNames::RunName);
    if (launch_error != 0) {
        std::fprintf(stderr, "AICPU PMU helper launch failed: %d\n", launch_error);
        return false;
    }
    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(PMU helper)") ||
        !CheckAcl(
            aclrtMemcpy(control, sizeof(*control), control_device, sizeof(*control), ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy(D2H PMU control)"
        )) {
        return false;
    }
    const bool expected_state = command == atomic_probe::pmu::PmuCommand::Configure
        ? control->configured == 1 && control->processed_subcores == atomic_probe::pmu::kPmuPhysicalSubcores
        : control->configured == 0 && control->processed_subcores == 0;
    if (control->status != 0 || !expected_state) {
        std::fprintf(
            stderr, "PMU helper failed: command=%u status=%d configured=%u processed=%u\n", control->command,
            static_cast<int>(control->status), control->configured, control->processed_subcores
        );
        return false;
    }
    return true;
}

const char *ModeName(atomic_scalar_pmu::Mode mode) {
    switch (mode) {
        case atomic_scalar_pmu::Mode::Empty: return "EMPTY";
        case atomic_scalar_pmu::Mode::ScalarControl: return "SCALAR_CONTROL";
        case atomic_scalar_pmu::Mode::DependentAtomicAdd: return "DEPENDENT_ATOMIC_ADD";
        default: return "UNKNOWN";
    }
}

bool ParseUint64(const char *text, uint64_t maximum, uint64_t *value) {
    if (text == nullptr || text[0] == '\0') return false;
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || parsed > maximum) return false;
    *value = static_cast<uint64_t>(parsed);
    return true;
}

uint32_t RepeatsFromEnv() {
    const char *raw = std::getenv("ATOMIC_SCALAR_PMU_REPEATS");
    if (raw == nullptr || raw[0] == '\0') return 7;
    uint64_t value = 0;
    if (!ParseUint64(raw, 100, &value) || value == 0) {
        std::fprintf(stderr, "ATOMIC_SCALAR_PMU_REPEATS must be in 1..100: %s\n", raw);
        return 0;
    }
    return static_cast<uint32_t>(value);
}

uint64_t SeedFromEnv(bool *ok) {
    const char *raw = std::getenv("ATOMIC_SCALAR_PMU_SEED");
    if (raw == nullptr || raw[0] == '\0') return 0x1234ULL;
    uint64_t value = 0;
    const bool parsed = ParseUint64(raw, std::numeric_limits<uint64_t>::max(), &value);
    *ok &= parsed;
    if (!parsed) std::fprintf(stderr, "Invalid ATOMIC_SCALAR_PMU_SEED: %s\n", raw);
    return value;
}

std::vector<uint32_t> RoundsFromEnv(bool *ok) {
    const char *raw = std::getenv("ATOMIC_SCALAR_PMU_ROUNDS");
    if (raw == nullptr || raw[0] == '\0') return {0, 1, 4, 16, 64, 256, 1024, 4096, 8192};
    std::vector<uint32_t> rounds;
    const std::string input(raw);
    size_t begin = 0;
    while (begin <= input.size()) {
        const size_t comma = input.find(',', begin);
        const std::string token = input.substr(begin, comma == std::string::npos ? comma : comma - begin);
        uint64_t value = 0;
        if (!ParseUint64(token.c_str(), 1000000, &value)) {
            std::fprintf(stderr, "Invalid ATOMIC_SCALAR_PMU_ROUNDS item: %s\n", token.c_str());
            *ok = false;
            return {};
        }
        rounds.push_back(static_cast<uint32_t>(value));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return rounds;
}

struct Oracle {
    uint64_t final_value;
    uint64_t checksum;
};

Oracle Simulate(uint64_t seed, uint32_t rounds) {
    uint64_t value = seed;
    uint64_t delta = 1;
    uint64_t checksum = 0;
    for (uint32_t round = 0; round < rounds; ++round) {
        const uint64_t old = value;
        value += delta;
        checksum += old;
        delta = 1 + (old & 1ULL);
    }
    return {value, checksum};
}

struct Sample {
    atomic_scalar_pmu::ProbeResult result{};
    uint64_t final_value = 0;
};

bool ValidateSample(
    const Sample &sample, atomic_scalar_pmu::Mode mode, uint32_t rounds, uint64_t seed, std::string *reason
) {
    const Oracle oracle = Simulate(seed, rounds);
    const uint64_t expected_checksum = mode == atomic_scalar_pmu::Mode::Empty ? 0 : oracle.checksum;
    const uint64_t expected_final = mode == atomic_scalar_pmu::Mode::DependentAtomicAdd ? oracle.final_value : seed;
    if (sample.result.checksum != expected_checksum) {
        *reason = "checksum";
        return false;
    }
    if (sample.final_value != expected_final) {
        *reason = "target-final";
        return false;
    }
    if (sample.result.physical_core_id >= kPhysicalSubcoreCount) {
        *reason = "physical-core-id";
        return false;
    }
    if ((sample.result.pmu_ctrl_after_stop & 1ULL) != 0) {
        *reason = "pmu-gate-still-enabled";
        return false;
    }
    if (sample.result.sys_cycles == 0 || sample.result.pmu_total_cycles == 0) {
        *reason = "zero-cycle-window";
        return false;
    }
    if (sample.result.pmu_icache_miss > sample.result.pmu_icache_request) {
        *reason = "icache-miss-exceeds-request";
        return false;
    }
    return true;
}

bool RunOne(
    aclrtFuncHandle function, aclrtStream stream, void *state_device, uint64_t pmu_register_bases,
    atomic_scalar_pmu::Mode mode, uint32_t rounds, uint32_t repeat, uint64_t seed, Sample *sample
) {
    atomic_scalar_pmu::ProbeState state{};
    state.control.pmu_register_bases = pmu_register_bases;
    state.control.mode = static_cast<uint32_t>(mode);
    state.control.rounds = rounds;
    state.control.seed = seed;
    state.target.value = seed;
    if (!CheckAcl(
            aclrtMemcpy(
                state_device, sizeof(state), &state, sizeof(state), ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy(H2D probe state)"
        )) {
        return false;
    }

    struct KernelArgs {
        uint64_t state_pointer;
    } args{reinterpret_cast<uint64_t>(state_device)};
    static_assert(sizeof(KernelArgs) == sizeof(uint64_t), "unexpected CCEC kernel argument ABI");
    if (!CheckAcl(
            aclrtLaunchKernelWithHostArgs(function, 1, stream, nullptr, &args, sizeof(args), nullptr, 0),
            "aclrtLaunchKernelWithHostArgs"
        ) ||
        !CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream(AIV probe)") ||
        !CheckAcl(
            aclrtMemcpy(&state, sizeof(state), state_device, sizeof(state), ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy(D2H probe state)"
        )) {
        return false;
    }

    sample->result = state.result;
    sample->final_value = state.target.value;
    std::string reason;
    const bool semantic_ok = ValidateSample(*sample, mode, rounds, seed, &reason);
    std::printf(
        "[RAW] repeat=%u rounds=%u mode=%s sys_cycles=%llu total=%llu scalar=%llu "
        "icache_req=%llu icache_miss=%llu checksum=%llu final=%llu physical=%llu ctrl=0x%llx status=%s%s%s\n",
        repeat, rounds, ModeName(mode), static_cast<unsigned long long>(sample->result.sys_cycles),
        static_cast<unsigned long long>(sample->result.pmu_total_cycles),
        static_cast<unsigned long long>(sample->result.pmu_scalar_busy),
        static_cast<unsigned long long>(sample->result.pmu_icache_request),
        static_cast<unsigned long long>(sample->result.pmu_icache_miss),
        static_cast<unsigned long long>(sample->result.checksum),
        static_cast<unsigned long long>(sample->final_value),
        static_cast<unsigned long long>(sample->result.physical_core_id),
        static_cast<unsigned long long>(sample->result.pmu_ctrl_after_stop), semantic_ok ? "PASS" : "FAIL",
        semantic_ok ? "" : " reason=", semantic_ok ? "" : reason.c_str()
    );
    return semantic_ok;
}

uint64_t Median(std::vector<uint64_t> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if ((values.size() & 1U) != 0) return values[middle];
    return values[middle - 1] + (values[middle] - values[middle - 1]) / 2;
}

double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return (values.size() & 1U) != 0 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
}

using CounterMember = uint64_t atomic_scalar_pmu::ProbeResult::*;

double PairedDeltaPerOperation(
    const std::vector<Sample> &minuend, const std::vector<Sample> &subtrahend,
    CounterMember member, uint32_t rounds
) {
    std::vector<double> deltas;
    for (size_t index = 0; index < minuend.size(); ++index) {
        deltas.push_back(
            (static_cast<double>(minuend[index].result.*member) -
             static_cast<double>(subtrahend[index].result.*member)) /
            rounds
        );
    }
    return Median(std::move(deltas));
}

void PrintRoundSummary(
    uint32_t rounds, const std::array<std::vector<Sample>, 3> &samples
) {
    struct Metric {
        const char *name;
        CounterMember member;
    };
    constexpr Metric metrics[] = {
        {"sys_cycles", &atomic_scalar_pmu::ProbeResult::sys_cycles},
        {"total", &atomic_scalar_pmu::ProbeResult::pmu_total_cycles},
        {"scalar", &atomic_scalar_pmu::ProbeResult::pmu_scalar_busy},
        {"icache_req", &atomic_scalar_pmu::ProbeResult::pmu_icache_request},
        {"icache_miss", &atomic_scalar_pmu::ProbeResult::pmu_icache_miss},
    };
    for (uint32_t mode_index = 0; mode_index < static_cast<uint32_t>(atomic_scalar_pmu::Mode::Count);
         ++mode_index) {
        const auto mode = static_cast<atomic_scalar_pmu::Mode>(mode_index);
        std::printf("[MEDIAN] rounds=%u mode=%s", rounds, ModeName(mode));
        for (const Metric &metric : metrics) {
            std::vector<uint64_t> values;
            for (const Sample &sample : samples[mode_index]) values.push_back(sample.result.*(metric.member));
            std::printf(" %s=%llu", metric.name, static_cast<unsigned long long>(Median(std::move(values))));
        }
        std::printf("\n");
    }
    if (rounds == 0) return;

    const auto &empty = samples[static_cast<uint32_t>(atomic_scalar_pmu::Mode::Empty)];
    const auto &control = samples[static_cast<uint32_t>(atomic_scalar_pmu::Mode::ScalarControl)];
    const auto &atomic = samples[static_cast<uint32_t>(atomic_scalar_pmu::Mode::DependentAtomicAdd)];
    for (const Metric &metric : metrics) {
        const double control_minus_empty =
            PairedDeltaPerOperation(control, empty, metric.member, rounds);
        const double atomic_minus_control =
            PairedDeltaPerOperation(atomic, control, metric.member, rounds);
        std::printf(
            "[DELTA_PER_OP] rounds=%u metric=%s control_minus_empty=%.6f atomic_minus_control=%.6f\n", rounds,
            metric.name, control_minus_empty, atomic_minus_control
        );
    }

    const double atomic_sys_ns = PairedDeltaPerOperation(
        atomic, control, &atomic_scalar_pmu::ProbeResult::sys_cycles, rounds
    );
    const double atomic_total_cycles = PairedDeltaPerOperation(
        atomic, control, &atomic_scalar_pmu::ProbeResult::pmu_total_cycles, rounds
    );
    const double atomic_scalar_cycles = PairedDeltaPerOperation(
        atomic, control, &atomic_scalar_pmu::ProbeResult::pmu_scalar_busy, rounds
    );
    const double scalar_share = atomic_total_cycles == 0.0 ? 0.0 : atomic_scalar_cycles / atomic_total_cycles;
    std::printf(
        "[ATOMIC_CLASSIFICATION] rounds=%u completion_ns_per_op=%.6f "
        "pmu_total_cycles_per_op=%.6f scalar_busy_cycles_per_op=%.6f scalar_share=%.9f\n",
        rounds, atomic_sys_ns, atomic_total_cycles, atomic_scalar_cycles, scalar_share
    );
}

}  // namespace

int main(int argc, char **argv) {
    const std::string kernel_path = argc > 1 ? argv[1] : "./atomic_scalar_pmu_kernel.o";
    if (argc > 2) {
        std::fprintf(stderr, "Usage: %s [atomic_scalar_pmu_kernel.o]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const uint32_t repeats = RepeatsFromEnv();
    bool options_ok = repeats != 0;
    const uint64_t seed = SeedFromEnv(&options_ok);
    const std::vector<uint32_t> round_values = RoundsFromEnv(&options_ok);
    if (!options_ok || round_values.empty()) return EXIT_FAILURE;

    const int32_t device = atomic_probe::DeviceId();
    if (device < 0) return EXIT_FAILURE;
    const std::vector<char> kernel_data = ReadBinary(kernel_path);
    if (kernel_data.empty()) {
        std::fprintf(stderr, "Cannot read kernel binary: %s\n", kernel_path.c_str());
        return EXIT_FAILURE;
    }

    if (!CheckAcl(aclInit(nullptr), "aclInit") || !CheckAcl(aclrtSetDevice(device), "aclrtSetDevice")) {
        return EXIT_FAILURE;
    }
    aclrtStream stream = nullptr;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) return EXIT_FAILURE;

    aclrtBinHandle binary_handle = nullptr;
    if (!CheckAcl(
            atomic_probe::LoadAicoreBinaryFromData(kernel_data.data(), kernel_data.size(), &binary_handle),
            "LoadAicoreBinaryFromData"
        )) {
        return EXIT_FAILURE;
    }
    aclrtFuncHandle function = nullptr;
    if (!CheckAcl(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function), "aclrtBinaryGetFunctionByEntry")) {
        return EXIT_FAILURE;
    }

    void *state_device = nullptr;
    void *pmu_regs_device = nullptr;
    void *pmu_control_device = nullptr;
    if (!CheckAcl(
            aclrtMalloc(&state_device, sizeof(atomic_scalar_pmu::ProbeState), ACL_MEM_MALLOC_NORMAL_ONLY),
            "aclrtMalloc(probe state)"
        )) {
        return EXIT_FAILURE;
    }
    PmuResources pmu_resources;
    if (!pmu_resources.Initialize(static_cast<uint32_t>(device))) return EXIT_FAILURE;
    const size_t pmu_regs_bytes = sizeof(pmu_resources.RegisterBases());
    if (!CheckAcl(aclrtMalloc(&pmu_regs_device, pmu_regs_bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc(PMU regs)") ||
        !CheckAcl(
            aclrtMemcpy(
                pmu_regs_device, pmu_regs_bytes, pmu_resources.RegisterBases().data(), pmu_regs_bytes,
                ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy(H2D PMU regs)"
        ) ||
        !CheckAcl(
            aclrtMalloc(&pmu_control_device, sizeof(atomic_probe::pmu::PmuControl), ACL_MEM_MALLOC_NORMAL_ONLY),
            "aclrtMalloc(PMU control)"
        )) {
        return EXIT_FAILURE;
    }

    atomic_probe::pmu::PmuControl pmu_control{};
    pmu_control.magic = atomic_probe::pmu::kPmuControlMagic;
    pmu_control.version = atomic_probe::pmu::kPmuControlVersion;
    pmu_control.expected_subcores = atomic_probe::pmu::kPmuPhysicalSubcores;
    if (!CheckAcl(
            aclrtMemcpy(
                pmu_control_device, sizeof(pmu_control), &pmu_control, sizeof(pmu_control),
                ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "aclrtMemcpy(H2D initial PMU control)"
        )) {
        return EXIT_FAILURE;
    }

    const std::string dispatcher_path = ArtifactBesideKernel(kernel_path, "libsimpler_aicpu_dispatcher.so");
    const std::string helper_path = ArtifactBesideKernel(kernel_path, "libatomic_scalar_pmu_aicpu.so");
    const std::vector<char> dispatcher_data = ReadBinary(dispatcher_path);
    const std::vector<char> helper_data = ReadBinary(helper_path);
    if (dispatcher_data.empty() || helper_data.empty()) {
        std::fprintf(stderr, "Cannot read PMU artifacts: %s %s\n", dispatcher_path.c_str(), helper_path.c_str());
        return EXIT_FAILURE;
    }
    host::LoadAicpuOp pmu_loader;
    if (pmu_loader.BootstrapDispatcher(
            dispatcher_data.data(), dispatcher_data.size(), helper_data.data(), helper_data.size(), stream, device
        ) != 0 ||
        pmu_loader.Init() != 0) {
        std::fprintf(stderr, "Cannot initialize PMU AICPU helper.\n");
        return EXIT_FAILURE;
    }
    KernelArgs pmu_kernel_args{};
    pmu_kernel_args.runtime_args = reinterpret_cast<Runtime *>(pmu_control_device);
    pmu_kernel_args.regs = reinterpret_cast<uint64_t>(pmu_regs_device);
    pmu_kernel_args.device_id = static_cast<uint32_t>(device);
    if (!RunPmuCommand(
            pmu_loader, stream, &pmu_kernel_args, pmu_control_device, &pmu_control,
            atomic_probe::pmu::PmuCommand::Configure
        )) {
        return EXIT_FAILURE;
    }

    std::printf(
        "=== Single-AIV dependent atomicAdd scalar-busy PMU probe ===\n"
        "device=%d repeats=%u seed=0x%llx events=total,scalar_busy(0x1),icache_req(0x34),icache_miss(0x35)\n",
        device, repeats, static_cast<unsigned long long>(seed)
    );
    bool all_passed = true;
    for (const uint32_t rounds : round_values) {
        std::array<std::vector<Sample>, 3> samples;
        for (uint32_t repeat = 1; repeat <= repeats; ++repeat) {
            for (uint32_t mode_index = 0; mode_index < static_cast<uint32_t>(atomic_scalar_pmu::Mode::Count);
                 ++mode_index) {
                Sample sample;
                const auto mode = static_cast<atomic_scalar_pmu::Mode>(mode_index);
                const bool passed = RunOne(
                    function, stream, state_device, reinterpret_cast<uint64_t>(pmu_regs_device), mode, rounds,
                    repeat, seed, &sample
                );
                all_passed &= passed;
                samples[mode_index].push_back(sample);
                if (!passed) break;
            }
            if (!all_passed) break;
        }
        if (!all_passed) break;
        PrintRoundSummary(rounds, samples);
    }

    const bool restored = RunPmuCommand(
        pmu_loader, stream, &pmu_kernel_args, pmu_control_device, &pmu_control,
        atomic_probe::pmu::PmuCommand::Restore
    );
    bool cleanup_ok = restored;
    pmu_loader.Finalize();
    pmu_resources.RestoreAndUnmap();
    cleanup_ok &= CheckAcl(aclrtFree(pmu_control_device), "aclrtFree(PMU control)");
    cleanup_ok &= CheckAcl(aclrtFree(pmu_regs_device), "aclrtFree(PMU regs)");
    cleanup_ok &= CheckAcl(aclrtFree(state_device), "aclrtFree(probe state)");
    cleanup_ok &= CheckAcl(aclrtBinaryUnLoad(binary_handle), "aclrtBinaryUnLoad");
    cleanup_ok &= CheckAcl(aclrtDestroyStream(stream), "aclrtDestroyStream");
    cleanup_ok &= CheckAcl(aclrtResetDevice(device), "aclrtResetDevice");
    cleanup_ok &= CheckAcl(aclFinalize(), "aclFinalize");
    std::printf(
        "[SUMMARY] semantic_status=%s pmu_restore_and_cleanup=%s\n", all_passed ? "PASS" : "FAIL",
        cleanup_ok ? "PASS" : "FAIL"
    );
    return all_passed && cleanup_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
