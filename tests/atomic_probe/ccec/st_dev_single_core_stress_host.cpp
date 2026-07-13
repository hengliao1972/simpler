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
#include "../probe_host.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

constexpr uint32_t CACHELINE_BYTES = 64;
constexpr uint32_t CACHELINE_WORDS = 16;
constexpr uint32_t DATA_LINES = 3;
constexpr uint32_t RESULT_BASE = 3 * CACHELINE_WORDS;
constexpr uint32_t PARTICIPATION_BASE = 4 * CACHELINE_WORDS;
constexpr uint32_t MARKER_BASE = 5 * CACHELINE_WORDS;
constexpr uint32_t TAIL_GUARD_BASE = 6 * CACHELINE_WORDS;
constexpr uint32_t STORAGE_LINES = 7;
constexpr uint32_t STORAGE_WORDS = STORAGE_LINES * CACHELINE_WORDS;
constexpr uint32_t NUM_MODES = 7;
constexpr uint32_t ROUNDS = 257;
constexpr uint32_t TRIALS = 100;
constexpr uint32_t DEFAULT_SINGLE_LAUNCHES = 5000;
constexpr uint32_t DEFAULT_DUAL_LAUNCHES = 500;
constexpr uint32_t MAX_LAUNCHES = 1000000;
constexpr uint32_t INVALID_U32 = 0xffffffffu;

constexpr uint32_t RESULT_MAGIC_VALUE = 0x53314352u;
constexpr uint32_t RESULT_FINISH_VALUE = 0x46494e49u;
constexpr uint32_t MARKER_MAGIC_VALUE = 0x53434f52u;

constexpr uint32_t A5_GROUPS_PER_DIE = 18;
constexpr uint32_t A5_AIVS_PER_GROUP = 2;
constexpr uint32_t A5_CORES_PER_DIE = A5_GROUPS_PER_DIE * (A5_AIVS_PER_GROUP + 1);

enum class ProbeMode : uint32_t {
    SINGLE_LINE0_LOOP_DSB = 0,
    SINGLE_LINE1_LOOP_DSB = 1,
    SINGLE_LINE2_LOOP_DSB = 2,
    DUAL_LINE01_LOOP_DSB = 3,
    DUAL_LINE12_LOOP_DSB = 4,
    SINGLE_LINE1_PER_WRITE_DSB = 5,
    DUAL_LINE12_PER_WRITE_DSB = 6,
};

struct KernelArgs {
    uint64_t storage_pointer;
    uint32_t mode;
    uint32_t reserved;
};

static_assert(sizeof(KernelArgs) == 16, "unexpected CCEC kernel argument ABI");

static void Check(aclError error, const char *label)
{
    if (!atomic_probe::CheckAcl(error, label, __FILE__, __LINE__)) std::exit(EXIT_FAILURE);
}

static uint32_t DataStartLine(uint32_t mode)
{
    if (mode == static_cast<uint32_t>(ProbeMode::SINGLE_LINE1_LOOP_DSB) ||
        mode == static_cast<uint32_t>(ProbeMode::SINGLE_LINE1_PER_WRITE_DSB) ||
        mode == static_cast<uint32_t>(ProbeMode::DUAL_LINE12_LOOP_DSB) ||
        mode == static_cast<uint32_t>(ProbeMode::DUAL_LINE12_PER_WRITE_DSB)) {
        return 1u;
    }
    return mode == static_cast<uint32_t>(ProbeMode::SINGLE_LINE2_LOOP_DSB) ? 2u : 0u;
}

static uint32_t SlotCount(uint32_t mode)
{
    return mode == static_cast<uint32_t>(ProbeMode::DUAL_LINE01_LOOP_DSB) ||
        mode == static_cast<uint32_t>(ProbeMode::DUAL_LINE12_LOOP_DSB) ||
        mode == static_cast<uint32_t>(ProbeMode::DUAL_LINE12_PER_WRITE_DSB) ? 2u : 1u;
}

static bool UsesPerWriteDsb(uint32_t mode)
{
    return mode == static_cast<uint32_t>(ProbeMode::SINGLE_LINE1_PER_WRITE_DSB) ||
        mode == static_cast<uint32_t>(ProbeMode::DUAL_LINE12_PER_WRITE_DSB);
}

static const char *ModeLabel(uint32_t mode)
{
    static const char *const labels[NUM_MODES] = {
        "single-line0-loop-dsb", "single-line1-loop-dsb", "single-line2-loop-dsb",
        "dual-line01-loop-dsb", "dual-line12-loop-dsb", "single-line1-per-write-dsb",
        "dual-line12-per-write-dsb"};
    return labels[mode];
}

static uint32_t TrialValue(uint32_t mode, uint32_t slot, uint32_t trial, uint32_t round)
{
    return 0xA0000000u + mode * 0x01000000u + slot * 0x00100000u + trial * 0x1000u + round;
}

static uint32_t TsyncCommSlot(uint32_t core_id, uint32_t subblock_id)
{
    int32_t die_id = static_cast<int32_t>(core_id / A5_CORES_PER_DIE);
    int32_t local_core_id = static_cast<int32_t>(core_id % A5_CORES_PER_DIE);
    return static_cast<uint32_t>(die_id * static_cast<int32_t>(A5_GROUPS_PER_DIE) +
        (local_core_id - static_cast<int32_t>(A5_GROUPS_PER_DIE) -
         static_cast<int32_t>(subblock_id)) / static_cast<int32_t>(A5_AIVS_PER_GROUP));
}

static bool OptionalLaunches(uint32_t mode, uint32_t *launches)
{
    uint32_t default_value = SlotCount(mode) == 1u ? DEFAULT_SINGLE_LAUNCHES : DEFAULT_DUAL_LAUNCHES;
    const char *raw = std::getenv("ATOMIC_PROBE_STRESS_LAUNCHES");
    if (raw == nullptr || raw[0] == '\0') {
        *launches = default_value;
        return true;
    }
    errno = 0;
    char *end = nullptr;
    unsigned long parsed = std::strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || parsed == 0 || parsed > MAX_LAUNCHES) {
        std::fprintf(stderr, "ATOMIC_PROBE_STRESS_LAUNCHES 必须是 1..%u，实际为：%s\n", MAX_LAUNCHES, raw);
        return false;
    }
    *launches = static_cast<uint32_t>(parsed);
    return true;
}

static bool LinesAreIndependent(void *storage)
{
    uintptr_t base = reinterpret_cast<uintptr_t>(storage);
    if ((base & (CACHELINE_BYTES - 1u)) != 0u) return false;
    for (uint32_t line = 1; line < STORAGE_LINES; ++line) {
        uintptr_t previous = base + (line - 1u) * CACHELINE_BYTES;
        uintptr_t current = base + line * CACHELINE_BYTES;
        if (current - previous != CACHELINE_BYTES || (current >> 6) == (previous >> 6)) return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <kernel.o>\n", argv[0]);
        return EXIT_FAILURE;
    }

    uint32_t mode = 0;
    if (!atomic_probe::RequiredUintEnv("ATOMIC_PROBE_MODE", NUM_MODES, &mode)) {
        return EXIT_FAILURE;
    }
    uint32_t launches = 0;
    if (!OptionalLaunches(mode, &launches)) {
        return EXIT_FAILURE;
    }
    uint32_t data_start_line = DataStartLine(mode);
    uint32_t slots = SlotCount(mode);
    bool per_write_dsb = UsesPerWriteDsb(mode);

    int32_t device = atomic_probe::DeviceId();
    if (device < 0) return EXIT_FAILURE;
    Check(aclInit(nullptr), "初始化 ACL");
    Check(aclrtSetDevice(device), "设置 atomic probe 设备");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "创建 single-core stress stream");

    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "无法打开 kernel 文件：%s\n", argv[1]);
        return EXIT_FAILURE;
    }
    size_t binary_size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> binary(binary_size);
    file.read(binary.data(), static_cast<std::streamsize>(binary_size));
    if (!file) return EXIT_FAILURE;

    aclrtBinHandle binary_handle;
    Check(atomic_probe::LoadAicoreBinaryFromData(binary.data(), binary_size, &binary_handle),
          "加载 single-core stress AICore binary");
    aclrtFuncHandle function_handle;
    Check(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function_handle),
          "取得 single-core stress kernel 入口");

    void *device_storage = nullptr;
    Check(aclrtMalloc(&device_storage, STORAGE_WORDS * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
          "分配 single-core stress storage");
    if (!LinesAreIndependent(device_storage)) {
        std::fprintf(stderr, "single-core stress storage 未满足独立 64B line 前提\n");
        Check(aclrtFree(device_storage), "释放未对齐 storage");
        Check(aclrtBinaryUnLoad(binary_handle), "卸载 AICore binary");
        Check(aclrtDestroyStream(stream), "销毁 stream");
        Check(aclrtResetDevice(device), "重置 probe 设备");
        Check(aclFinalize(), "结束 ACL");
        return EXIT_FAILURE;
    }
    uintptr_t allocation = reinterpret_cast<uintptr_t>(device_storage);
    std::printf("allocation_alignment mod128=%zu mod256=%zu mod512=%zu\n",
                static_cast<size_t>(allocation % 128u), static_cast<size_t>(allocation % 256u),
                static_cast<size_t>(allocation % 512u));

    uint64_t total_errors = 0;
    uint64_t total_slot_errors[2] = {0u, 0u};
    uint64_t protocol_failures = 0;
    uint64_t final_snapshot_failures = 0;
    bool have_first_error = false;
    uint32_t first_launch = INVALID_U32;
    uint32_t first_actual = 0;
    uint32_t first_expected = 0;
    uint32_t first_trial = INVALID_U32;
    uint32_t first_slot = INVALID_U32;
    uint32_t first_core = INVALID_U32;
    uint32_t first_subblock = INVALID_U32;
    int32_t first_comm_slot = 0;
    uint32_t progress_step = launches < 10u ? 1u : launches / 10u;

    for (uint32_t launch = 0; launch < launches; ++launch) {
        uint32_t host[STORAGE_WORDS] = {};
        Check(aclrtMemcpy(device_storage, sizeof(host), host, sizeof(host), ACL_MEMCPY_HOST_TO_DEVICE),
              "初始化 single-core stress storage");
        KernelArgs args{static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device_storage)), mode, 0u};
        Check(aclrtLaunchKernelWithHostArgs(function_handle, 1u, stream, nullptr,
                                            &args, sizeof(args), nullptr, 0),
              "启动 single-core stress kernel");
        Check(aclrtSynchronizeStream(stream), "等待 single-core stress kernel");
        Check(aclrtMemcpy(host, sizeof(host), device_storage, sizeof(host), ACL_MEMCPY_DEVICE_TO_HOST),
              "读取 single-core stress 结果");

        uint32_t *result = host + RESULT_BASE;
        uint32_t *participation = host + PARTICIPATION_BASE;
        uint32_t *marker = host + MARKER_BASE;
        bool protocol_exact = result[0] == RESULT_MAGIC_VALUE && result[1] == mode &&
            result[2] == TRIALS * slots && result[3] == result[4] + result[5] &&
            result[10] == ROUNDS && result[11] == TRIALS && result[12] == data_start_line &&
            result[13] == slots && result[14] == (per_write_dsb ? 1u : 0u) &&
            result[15] == RESULT_FINISH_VALUE;

        protocol_exact = protocol_exact && participation[0] == 1u;
        for (uint32_t word = 1; word < CACHELINE_WORDS; ++word) {
            protocol_exact = protocol_exact && participation[word] == 0u;
        }
        protocol_exact = protocol_exact && marker[0] == MARKER_MAGIC_VALUE &&
            marker[1] == 0u && marker[2] == 1u && marker[5] == TsyncCommSlot(marker[3], marker[4]) &&
            marker[6] == mode && marker[7] == slots && marker[8] == data_start_line &&
            marker[9] == (per_write_dsb ? 1u : 0u) && marker[10] == 1u;
        for (uint32_t word = 11; word < CACHELINE_WORDS; ++word) {
            protocol_exact = protocol_exact && marker[word] == 0u;
        }
        if (launch == 0u) {
            first_core = marker[3];
            first_subblock = marker[4];
            first_comm_slot = static_cast<int32_t>(marker[5]);
        }

        for (uint32_t line = 0; line < DATA_LINES; ++line) {
            bool target_line = line >= data_start_line && line < data_start_line + slots;
            uint32_t logical_slot = line - data_start_line;
            for (uint32_t word = 0; word < CACHELINE_WORDS; ++word) {
                uint32_t expected = target_line && word == 0u ?
                    TrialValue(mode, logical_slot, TRIALS - 1u, ROUNDS - 1u) : 0u;
                if (host[line * CACHELINE_WORDS + word] != expected) {
                    if (target_line && word == 0u) {
                        ++final_snapshot_failures;
                    } else {
                        protocol_exact = false;
                    }
                }
            }
        }
        for (uint32_t word = 0; word < CACHELINE_WORDS; ++word) {
            protocol_exact = protocol_exact && host[TAIL_GUARD_BASE + word] == 0u;
        }

        uint32_t launch_errors = result[3];
        total_errors += launch_errors;
        total_slot_errors[0] += result[4];
        total_slot_errors[1] += result[5];
        if (launch_errors == 0u) {
            protocol_exact = protocol_exact && result[6] == 0u && result[7] == 0u &&
                result[8] == INVALID_U32 && result[9] == INVALID_U32;
        } else {
            bool diagnostic_exact = result[6] != result[7] && result[8] < TRIALS && result[9] < slots &&
                result[7] == TrialValue(mode, result[9], result[8], ROUNDS - 1u);
            protocol_exact = protocol_exact && diagnostic_exact;
            if (!have_first_error) {
                have_first_error = true;
                first_launch = launch;
                first_actual = result[6];
                first_expected = result[7];
                first_trial = result[8];
                first_slot = result[9];
                std::printf("FIRST_NONZERO mode=%u launch=%u errors=%u/%u slot0=%u slot1=%u "
                            "trial=%u slot=%u actual=0x%08x expected=0x%08x\n",
                            mode, launch, launch_errors, TRIALS * slots, result[4], result[5],
                            first_trial, first_slot, first_actual, first_expected);
            }
        }
        if (!protocol_exact) {
            ++protocol_failures;
        }

        if ((launch + 1u) % progress_step == 0u || launch + 1u == launches) {
            std::printf("progress mode=%u launches=%u/%u mismatches=%llu/%llu\n", mode, launch + 1u,
                        launches, static_cast<unsigned long long>(total_errors),
                        static_cast<unsigned long long>(launch + 1u) * TRIALS * slots);
        }
    }

    std::printf("topology mode=%u block0(core=%u sub=%u comm_slot=%d)\n",
                mode, first_core, first_subblock, first_comm_slot);
    std::printf("aggregate single-core mode=%u label=%s data_start_line=%u slots=%u per_write_dsb=%u "
                "mismatches=%llu/%llu slot0=%llu/%llu slot1=%llu/%llu "
                "final_snapshot_bad=%llu first_launch=%u first_trial=%u first_slot=%u "
                "actual=0x%08x expected=0x%08x\n",
                mode, ModeLabel(mode), data_start_line, slots, per_write_dsb ? 1u : 0u,
                static_cast<unsigned long long>(total_errors),
                static_cast<unsigned long long>(launches) * TRIALS * slots,
                static_cast<unsigned long long>(total_slot_errors[0]),
                static_cast<unsigned long long>(launches) * TRIALS,
                static_cast<unsigned long long>(total_slot_errors[1]),
                static_cast<unsigned long long>(launches) * TRIALS * (slots - 1u),
                static_cast<unsigned long long>(final_snapshot_failures), first_launch,
                first_trial, first_slot, first_actual, first_expected);

    atomic_probe::Result result;
    result.Expect(protocol_failures == 0u,
                  "单 AIV 参与计数、拓扑 marker、结果头、非目标 data 与 guard 精确");
    result.Expect(total_errors == 0u,
                  "单 AIV repeated st_dev 后的 ld_dev 必须看到每个 slot 的最后写入值");
    result.Expect(final_snapshot_failures == 0u,
                  "每次 launch 返回 host 时，目标 data 的最终快照必须精确");

    Check(aclrtFree(device_storage), "释放 single-core stress storage");
    Check(aclrtBinaryUnLoad(binary_handle), "卸载 AICore binary");
    Check(aclrtDestroyStream(stream), "销毁 stream");
    Check(aclrtResetDevice(device), "重置 probe 设备");
    Check(aclFinalize(), "结束 ACL");
    return result.ExitCode();
}
