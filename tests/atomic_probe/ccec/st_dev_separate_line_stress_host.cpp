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

constexpr uint32_t CACHELINE_WORDS = 16;
constexpr uint32_t STORAGE_WORDS = 9 * CACHELINE_WORDS;
constexpr uint32_t DATA_BASE_LINE01 = 0;
constexpr uint32_t DATA_BASE_LINE12 = CACHELINE_WORDS;
constexpr uint32_t RESULT_BASE = 3 * CACHELINE_WORDS;
constexpr uint32_t PARTICIPATION_BASE = 4 * CACHELINE_WORDS;
constexpr uint32_t MARKER_BASE = 5 * CACHELINE_WORDS;
constexpr uint32_t TAIL_GUARD_BASE = 8 * CACHELINE_WORDS;
constexpr uint32_t CHECKS_PER_LAUNCH = 200;
constexpr uint32_t DEFAULT_LAUNCHES = 500;
constexpr uint32_t MAX_LAUNCHES = 100000;
constexpr uint32_t MODE_BLOCK02 = 1;
constexpr uint32_t MODE_BLOCK01_SHIFTED = 2;
constexpr uint32_t MODE_BLOCK02_SHIFTED = 3;
constexpr uint32_t A5_GROUPS_PER_DIE = 18;
constexpr uint32_t A5_AIVS_PER_GROUP = 2;
constexpr uint32_t A5_CORES_PER_DIE = A5_GROUPS_PER_DIE * (A5_AIVS_PER_GROUP + 1);

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

static bool OptionalLaunches(uint32_t *launches)
{
    const char *raw = std::getenv("ATOMIC_PROBE_STRESS_LAUNCHES");
    if (raw == nullptr || raw[0] == '\0') {
        *launches = DEFAULT_LAUNCHES;
        return true;
    }
    errno = 0;
    char *end = nullptr;
    unsigned long parsed = std::strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || parsed == 0 || parsed > MAX_LAUNCHES) {
        std::fprintf(stderr,
                     "ATOMIC_PROBE_STRESS_LAUNCHES 必须是 1..%u，实际为：%s\n",
                     MAX_LAUNCHES, raw);
        return false;
    }
    *launches = static_cast<uint32_t>(parsed);
    return true;
}

static uint32_t TsyncCommSlot(uint32_t core_id, uint32_t subblock_id)
{
    int32_t die_id = static_cast<int32_t>(core_id / A5_CORES_PER_DIE);
    int32_t local_core_id = static_cast<int32_t>(core_id % A5_CORES_PER_DIE);
    return static_cast<uint32_t>(die_id * static_cast<int32_t>(A5_GROUPS_PER_DIE) +
        (local_core_id - static_cast<int32_t>(A5_GROUPS_PER_DIE) -
         static_cast<int32_t>(subblock_id)) / static_cast<int32_t>(A5_AIVS_PER_GROUP));
}

static bool LineIsZero(const uint32_t *host, uint32_t base)
{
    for (uint32_t word = 0; word < CACHELINE_WORDS; ++word) {
        if (host[base + word] != 0u) return false;
    }
    return true;
}

static bool MarkerExact(const uint32_t *host, uint32_t participant, uint32_t mode)
{
    uint32_t base = MARKER_BASE + participant * CACHELINE_WORDS;
    bool exact = host[base + 0] == (0xA1A00000u | participant) &&
        host[base + 1] == participant &&
        host[base + 4] == TsyncCommSlot(host[base + 2], host[base + 3]) &&
        host[base + 5] > 0u &&
        host[base + 6] == mode;
    for (uint32_t word = 7; word < CACHELINE_WORDS; ++word) {
        exact = exact && host[base + word] == 0u;
    }
    return exact;
}

static uint32_t FinalValue(uint32_t slot)
{
    return 0xA5000000u + slot * 0x01000000u + 99u * 0x1000u + 256u;
}

static bool UsesBlock02(uint32_t mode)
{
    return mode == MODE_BLOCK02 || mode == MODE_BLOCK02_SHIFTED;
}

static uint32_t DataBase(uint32_t mode)
{
    return mode >= MODE_BLOCK01_SHIFTED ? DATA_BASE_LINE12 : DATA_BASE_LINE01;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <kernel.o>\n", argv[0]);
        return EXIT_FAILURE;
    }

    uint32_t mode = 0;
    uint32_t launches = 0;
    if (!atomic_probe::RequiredUintEnv("ATOMIC_PROBE_MODE", 4, &mode) ||
        !OptionalLaunches(&launches)) {
        return EXIT_FAILURE;
    }
    uint32_t participants = UsesBlock02(mode) ? 3u : 2u;
    uint32_t data_base = DataBase(mode);

    int32_t device = atomic_probe::DeviceId();
    if (device < 0) return EXIT_FAILURE;
    Check(aclInit(nullptr), "初始化 ACL");
    Check(aclrtSetDevice(device), "设置 atomic probe 设备");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "创建 separate-line stress stream");

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
          "加载 separate-line stress AICore binary");
    aclrtFuncHandle function_handle;
    Check(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function_handle),
          "取得 separate-line stress kernel 入口");

    void *device_storage = nullptr;
    Check(aclrtMalloc(&device_storage, STORAGE_WORDS * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
          "分配 separate-line stress storage");
    if ((reinterpret_cast<uintptr_t>(device_storage) & 63u) != 0u) {
        std::fprintf(stderr, "storage 首地址未按 64B 对齐，不能执行分-line 测试\n");
        Check(aclrtFree(device_storage), "释放未对齐 storage");
        Check(aclrtBinaryUnLoad(binary_handle), "卸载 AICore binary");
        Check(aclrtDestroyStream(stream), "销毁 stream");
        Check(aclrtResetDevice(device), "重置 probe 设备");
        Check(aclFinalize(), "结束 ACL");
        return EXIT_FAILURE;
    }
    uintptr_t storage_address = reinterpret_cast<uintptr_t>(device_storage);
    std::printf("storage alignment: mod128=%zu mod256=%zu mod512=%zu\n",
                static_cast<size_t>(storage_address & 127u),
                static_cast<size_t>(storage_address & 255u),
                static_cast<size_t>(storage_address & 511u));

    uint64_t total_errors = 0;
    uint64_t total_slot_errors[2] = {0u, 0u};
    uint32_t final_snapshot_failures = 0;
    uint32_t protocol_failures = 0;
    uint32_t first_bad_launch = 0xffffffffu;
    uint32_t first_actual = 0;
    uint32_t first_expected = 0;
    uint32_t first_trial = 0xffffffffu;
    uint32_t first_slot = 0xffffffffu;
    uint32_t first_core[3] = {};
    uint32_t first_subblock[3] = {};
    uint32_t first_comm_slot[3] = {};

    for (uint32_t launch = 0; launch < launches; ++launch) {
        uint32_t host[STORAGE_WORDS] = {};
        Check(aclrtMemcpy(device_storage, sizeof(host), host, sizeof(host),
                          ACL_MEMCPY_HOST_TO_DEVICE),
              "初始化 separate-line stress storage");
        KernelArgs args{
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device_storage)), mode, 0u};
        Check(aclrtLaunchKernelWithHostArgs(function_handle, participants, stream, nullptr,
                                            &args, sizeof(args), nullptr, 0),
              "启动 separate-line stress kernel");
        Check(aclrtSynchronizeStream(stream), "等待 separate-line stress kernel");
        Check(aclrtMemcpy(host, sizeof(host), device_storage, sizeof(host),
                          ACL_MEMCPY_DEVICE_TO_HOST),
              "读取 separate-line stress 结果");

        bool protocol_exact = host[RESULT_BASE + 1] == CHECKS_PER_LAUNCH &&
            host[RESULT_BASE + 2] == 0x53504C54u && host[RESULT_BASE + 3] == mode &&
            host[RESULT_BASE + 4] == participants && host[RESULT_BASE + 9] == 257u &&
            host[RESULT_BASE + 10] == 100u;
        protocol_exact = protocol_exact &&
            host[RESULT_BASE + 0] ==
                host[RESULT_BASE + 11] + host[RESULT_BASE + 12];
        for (uint32_t word = 13; word < CACHELINE_WORDS; ++word) {
            protocol_exact = protocol_exact && host[RESULT_BASE + word] == 0u;
        }

        bool participation_exact = host[PARTICIPATION_BASE] == participants;
        for (uint32_t word = 1; word < CACHELINE_WORDS; ++word) {
            participation_exact =
                participation_exact && host[PARTICIPATION_BASE + word] == 0u;
        }
        for (uint32_t participant = 0; participant < participants; ++participant) {
            protocol_exact = protocol_exact && MarkerExact(host, participant, mode);
            if (launch == 0u) {
                uint32_t base = MARKER_BASE + participant * CACHELINE_WORDS;
                first_core[participant] = host[base + 2];
                first_subblock[participant] = host[base + 3];
                first_comm_slot[participant] = host[base + 4];
            }
        }
        for (uint32_t participant = participants; participant < 3u; ++participant) {
            protocol_exact = protocol_exact &&
                LineIsZero(host, MARKER_BASE + participant * CACHELINE_WORDS);
        }
        uint32_t unused_data_base = data_base == DATA_BASE_LINE01 ?
            2u * CACHELINE_WORDS : DATA_BASE_LINE01;
        protocol_exact = protocol_exact && participation_exact &&
            LineIsZero(host, unused_data_base) && LineIsZero(host, TAIL_GUARD_BASE);
        if (host[data_base] != FinalValue(0) ||
            host[data_base + CACHELINE_WORDS] != FinalValue(1)) {
            ++final_snapshot_failures;
        }
        for (uint32_t word = 1; word < CACHELINE_WORDS; ++word) {
            protocol_exact = protocol_exact && host[data_base + word] == 0u &&
                host[data_base + CACHELINE_WORDS + word] == 0u;
        }
        if (!protocol_exact) ++protocol_failures;

        uint32_t launch_errors = host[RESULT_BASE + 0];
        total_slot_errors[0] += host[RESULT_BASE + 11];
        total_slot_errors[1] += host[RESULT_BASE + 12];
        if (launch_errors == 0u) {
            protocol_exact = host[RESULT_BASE + 5] == 0u &&
                host[RESULT_BASE + 6] == 0u &&
                host[RESULT_BASE + 7] == 0xffffffffu &&
                host[RESULT_BASE + 8] == 0xffffffffu;
            if (!protocol_exact) ++protocol_failures;
        } else {
            bool diagnostic_exact = host[RESULT_BASE + 5] != host[RESULT_BASE + 6] &&
                host[RESULT_BASE + 7] < 100u && host[RESULT_BASE + 8] < 2u;
            if (!diagnostic_exact) ++protocol_failures;
            if (first_bad_launch == 0xffffffffu) {
                first_bad_launch = launch;
                first_actual = host[RESULT_BASE + 5];
                first_expected = host[RESULT_BASE + 6];
                first_trial = host[RESULT_BASE + 7];
                first_slot = host[RESULT_BASE + 8];
            }
            if (first_bad_launch == launch) {
                std::printf(
                    "FIRST_NONZERO mode=%u launch=%u errors=%u/%u slot0=%u/100 "
                    "slot1=%u/100 first_trial=%u slot=%u actual=0x%08x expected=0x%08x\n",
                    mode, launch, launch_errors, CHECKS_PER_LAUNCH,
                    host[RESULT_BASE + 11], host[RESULT_BASE + 12],
                    host[RESULT_BASE + 7], host[RESULT_BASE + 8],
                    host[RESULT_BASE + 5], host[RESULT_BASE + 6]);
            }
        }
        total_errors += launch_errors;

        if ((launch + 1u) % 50u == 0u || launch + 1u == launches) {
            std::printf("progress mode=%u launches=%u/%u mismatches=%llu/%llu\n",
                        mode, launch + 1u, launches,
                        static_cast<unsigned long long>(total_errors),
                        static_cast<unsigned long long>(launch + 1u) * CHECKS_PER_LAUNCH);
        }
    }

    std::printf("topology mode=%u", mode);
    for (uint32_t participant = 0; participant < participants; ++participant) {
        std::printf(" block%u(core=%u sub=%u comm_slot=%u)%s", participant,
                    first_core[participant], first_subblock[participant],
                    first_comm_slot[participant],
                    UsesBlock02(mode) && participant == 1u ? "[data-idle]" : "");
    }
    std::printf("\n");
    std::printf(
        "aggregate separate-line-only mode=%u data_lines=%u/%u mismatches=%llu/%llu slot0=%llu/%llu "
        "slot1=%llu/%llu final_snapshot_bad=%u/%u first_launch=%u "
        "first_trial=%u first_slot=%u actual=0x%08x expected=0x%08x\n",
        mode, data_base / CACHELINE_WORDS,
        data_base / CACHELINE_WORDS + 1u, static_cast<unsigned long long>(total_errors),
        static_cast<unsigned long long>(launches) * CHECKS_PER_LAUNCH,
        static_cast<unsigned long long>(total_slot_errors[0]),
        static_cast<unsigned long long>(launches) * 100u,
        static_cast<unsigned long long>(total_slot_errors[1]),
        static_cast<unsigned long long>(launches) * 100u,
        final_snapshot_failures, launches,
        first_bad_launch, first_trial, first_slot, first_actual, first_expected);

    atomic_probe::Result result;
    result.Expect(protocol_failures == 0u,
                  "每次 launch 的参与计数、拓扑 marker、布局、结果头、未用 data line 与 guard 必须精确");
    result.Expect(total_errors == 0u,
                  "两个活跃 AIV 各写独占 64B line：全部终值必须等于各自最后一轮值");
    result.Expect(final_snapshot_failures == 0u,
                  "每个 launch 返回 host 时，两条独占 data line 的最终快照都必须精确");

    Check(aclrtFree(device_storage), "释放 separate-line stress storage");
    Check(aclrtBinaryUnLoad(binary_handle), "卸载 AICore binary");
    Check(aclrtDestroyStream(stream), "销毁 stream");
    Check(aclrtResetDevice(device), "重置 probe 设备");
    Check(aclFinalize(), "结束 ACL");
    return result.ExitCode();
}
