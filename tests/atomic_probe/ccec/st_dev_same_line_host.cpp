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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

// Host 精确判定：三个 mode 分别在独立进程启动 2、3、1 个 AIV。三核 mode 中，
// 逻辑 AIV1 除独占拓扑 marker 与 SyncAll 外不做数据访问；同-line 路径都必须保持
// 最后一轮精确值，分-line 与逐轮 DSB 路径必须为零错误。
constexpr size_t STORAGE_WORDS = 224;
constexpr uint32_t CACHELINE_WORDS = 16;
constexpr uint32_t RESULT_BASE = 128;
constexpr uint32_t PARTICIPATION_BASE = 144;
constexpr uint32_t MARKER_BASE = 160;
constexpr uint32_t GUARD_BASE = 208;
constexpr uint32_t LAUNCHES = 20;
constexpr uint32_t CHECKS_PER_LAUNCH = 200;
constexpr uint32_t SINGLE_CHECKS_PER_LAUNCH = 100;
constexpr uint32_t MODE_BLOCK01 = 0;
constexpr uint32_t MODE_BLOCK02 = 1;
constexpr uint32_t MODE_SINGLE_AIV = 2;
constexpr uint32_t A5_GROUPS_PER_DIE = 18;
constexpr uint32_t A5_AIVS_PER_GROUP = 2;
constexpr uint32_t A5_CORES_PER_DIE = A5_GROUPS_PER_DIE * (A5_AIVS_PER_GROUP + 1);
constexpr uint32_t NO_COMM_SLOT = 0xffffffffu;

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

static uint32_t TsyncCommSlot(uint32_t core_id, uint32_t subblock_id)
{
    int32_t die_id = static_cast<int32_t>(core_id / A5_CORES_PER_DIE);
    int32_t local_core_id = static_cast<int32_t>(core_id % A5_CORES_PER_DIE);
    return static_cast<uint32_t>(die_id * static_cast<int32_t>(A5_GROUPS_PER_DIE) +
        (local_core_id - static_cast<int32_t>(A5_GROUPS_PER_DIE) -
         static_cast<int32_t>(subblock_id)) / static_cast<int32_t>(A5_AIVS_PER_GROUP));
}

static bool MarkerExact(
    const uint32_t *host, uint32_t participant, uint32_t mode)
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

static bool LineIsZero(const uint32_t *host, uint32_t base)
{
    bool exact = true;
    for (uint32_t word = 0; word < CACHELINE_WORDS; ++word) {
        exact = exact && host[base + word] == 0u;
    }
    return exact;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <kernel.o>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int32_t device = atomic_probe::DeviceId();
    if (device < 0) return EXIT_FAILURE;
    Check(aclInit(nullptr), "aclInit");
    Check(aclrtSetDevice(device), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");

    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file) return EXIT_FAILURE;
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> binary(size);
    file.read(binary.data(), static_cast<std::streamsize>(size));
    if (!file) return EXIT_FAILURE;

    aclrtBinHandle binary_handle;
    Check(atomic_probe::LoadAicoreBinaryFromData(binary.data(), size, &binary_handle),
          "LoadAicoreBinaryFromData");
    aclrtFuncHandle function_handle;
    Check(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function_handle),
          "aclrtBinaryGetFunctionByEntry");

    void *device_storage = nullptr;
    Check(aclrtMalloc(&device_storage, STORAGE_WORDS * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
          "aclrtMalloc");

    uint32_t total_adjacent_same_errors = 0;
    uint32_t total_adjacent_separate_errors = 0;
    uint32_t total_adjacent_ordered_errors = 0;
    uint32_t total_gap_same_errors = 0;
    uint32_t total_gap_separate_errors = 0;
    uint32_t total_gap_ordered_errors = 0;
    atomic_probe::Result result;
    if (std::strstr(argv[0], "atomic_exch_same_line") != nullptr) {
        constexpr uint32_t legacy_result_base = 80;
        uint32_t total_same_errors = 0;
        uint32_t total_separate_errors = 0;
        uint32_t total_ordered_errors = 0;
        for (uint32_t launch = 0; launch < LAUNCHES; ++launch) {
            uint32_t host[STORAGE_WORDS] = {0};
            Check(aclrtMemcpy(
                device_storage, sizeof(host), host, sizeof(host), ACL_MEMCPY_HOST_TO_DEVICE),
                "aclrtMemcpy atomic init");
            uint64_t args =
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device_storage));
            Check(aclrtLaunchKernelWithHostArgs(
                function_handle, 2, stream, nullptr, &args, sizeof(args), nullptr, 0),
                "aclrtLaunchKernelWithHostArgs atomic");
            Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream atomic");
            Check(aclrtMemcpy(
                host, sizeof(host), device_storage, sizeof(host), ACL_MEMCPY_DEVICE_TO_HOST),
                "aclrtMemcpy atomic result");

            bool control_exact = host[legacy_result_base + 5] == 100u &&
                host[legacy_result_base + 6] == 0x3510C1A5u &&
                host[legacy_result_base + 7] == 2u &&
                host[legacy_result_base + 8] == 0xA1A00000u &&
                host[legacy_result_base + 9] == 0xA1A00001u;
            char label[128];
            std::snprintf(label, sizeof(label),
                          "AtomicExch launch=%u：两个 AIV 的参与计数与 marker 必须精确", launch);
            result.Expect(control_exact, label);
            total_same_errors += host[legacy_result_base + 0];
            total_separate_errors += host[legacy_result_base + 1];
            total_ordered_errors += host[legacy_result_base + 2];
        }
        std::printf(
            "aggregate AtomicExch same=%u/%u separate=%u/%u ordered=%u/%u\n",
            total_same_errors, LAUNCHES * CHECKS_PER_LAUNCH,
            total_separate_errors, LAUNCHES * CHECKS_PER_LAUNCH,
            total_ordered_errors, LAUNCHES * CHECKS_PER_LAUNCH);
        result.Expect(total_same_errors == 0,
                      "两个 AIV 同-line AtomicExch 终值必须全部精确");
        result.Expect(total_separate_errors == 0,
                      "两个 AIV 分-line AtomicExch control 必须全部精确");
        result.Expect(total_ordered_errors == 0,
                      "两个 AIV 同-line逐轮 DSB AtomicExch control 必须全部精确");

        Check(aclrtFree(device_storage), "aclrtFree");
        Check(aclrtBinaryUnLoad(binary_handle), "aclrtBinaryUnLoad");
        Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
        Check(aclrtResetDevice(device), "aclrtResetDevice");
        Check(aclFinalize(), "aclFinalize");
        return result.ExitCode();
    }

    uint32_t mode = 0;
    if (!atomic_probe::RequiredUintEnv("ATOMIC_PROBE_MODE", 3, &mode)) {
        return EXIT_FAILURE;
    }
    bool storage_aligned =
        (reinterpret_cast<uintptr_t>(device_storage) & 63u) == 0;
    result.Expect(
        storage_aligned,
        "storage 首地址必须 64B 对齐，保证六条数据路径及协议对象均按设计独占 cacheline");
    for (uint32_t launch = 0;
         mode != MODE_SINGLE_AIV && storage_aligned && launch < LAUNCHES;
         ++launch) {
        uint32_t host[STORAGE_WORDS] = {0};
        Check(aclrtMemcpy(device_storage, sizeof(host), host, sizeof(host), ACL_MEMCPY_HOST_TO_DEVICE),
              "aclrtMemcpy init");
        KernelArgs args{
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device_storage)),
            mode, 0u};
        uint32_t participants = mode == MODE_BLOCK01 ? 2u : 3u;
        Check(aclrtLaunchKernelWithHostArgs(function_handle, participants, stream, nullptr,
                                            &args, sizeof(args), nullptr, 0),
              "aclrtLaunchKernelWithHostArgs");
        Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
        Check(aclrtMemcpy(host, sizeof(host), device_storage, sizeof(host), ACL_MEMCPY_DEVICE_TO_HOST),
              "aclrtMemcpy result");

        bool result_header_exact = host[RESULT_BASE + 5] == 100u &&
            host[RESULT_BASE + 6] == 0x3510C1A5u &&
            host[RESULT_BASE + 12] == mode &&
            host[RESULT_BASE + 13] == participants;
        char label[160];
        std::snprintf(label, sizeof(label),
                      "launch=%u mode=%u：结果头必须记录正确 mode、参与 AIV 数与完整 magic",
                      launch, mode);
        result.Expect(result_header_exact, label);

        uint32_t separate_errors = mode == MODE_BLOCK01 ?
            host[RESULT_BASE + 1] : host[RESULT_BASE + 8];
        bool separate_diag_exact = separate_errors == 0 ?
            (host[RESULT_BASE + 14] == 0u && host[RESULT_BASE + 15] == 0u) :
            (host[RESULT_BASE + 14] != host[RESULT_BASE + 15]);
        std::snprintf(label, sizeof(label),
                      "launch=%u mode=%u：分-line 首错诊断必须与 error count 一致",
                      launch, mode);
        result.Expect(separate_diag_exact, label);

        bool participation_exact = host[PARTICIPATION_BASE] == participants;
        for (uint32_t word = 1; word < CACHELINE_WORDS; ++word) {
            participation_exact = participation_exact && host[PARTICIPATION_BASE + word] == 0u;
        }
        std::snprintf(label, sizeof(label),
                      "launch=%u mode=%u：参与计数 line 仅 word0 为 %u",
                      launch, mode, participants);
        result.Expect(participation_exact, label);

        bool markers_exact = true;
        for (uint32_t participant = 0; participant < participants; ++participant) {
            markers_exact = markers_exact && MarkerExact(host, participant, mode);
        }
        for (uint32_t participant = participants; participant < 3; ++participant) {
            markers_exact = markers_exact &&
                LineIsZero(host, MARKER_BASE + participant * CACHELINE_WORDS);
        }
        std::snprintf(label, sizeof(label),
                      "launch=%u mode=%u：已启动 block 的独占拓扑 marker 自洽，其余 marker 全零",
                      launch, mode);
        result.Expect(markers_exact, label);

        uint32_t comm_slot0 = host[MARKER_BASE + 4];
        uint32_t comm_slot1 = host[MARKER_BASE + CACHELINE_WORDS + 4];
        uint32_t comm_slot2 = mode == MODE_BLOCK02 ?
            host[MARKER_BASE + 2 * CACHELINE_WORDS + 4] : NO_COMM_SLOT;

        bool guard_exact = LineIsZero(host, GUARD_BASE);
        std::snprintf(label, sizeof(label),
                      "launch=%u：未使用 guard cacheline 必须保持全零", launch);
        result.Expect(guard_exact, label);

        total_adjacent_same_errors += host[RESULT_BASE + 0];
        total_adjacent_separate_errors += host[RESULT_BASE + 1];
        total_adjacent_ordered_errors += host[RESULT_BASE + 2];
        total_gap_same_errors += host[RESULT_BASE + 7];
        total_gap_separate_errors += host[RESULT_BASE + 8];
        total_gap_ordered_errors += host[RESULT_BASE + 9];
        if (mode == MODE_BLOCK01) {
            std::printf(
                "topology launch=%u: block0(core=%u sub=%u comm_slot=%u) "
                "block1(core=%u sub=%u comm_slot=%u)\n",
                launch, host[MARKER_BASE + 2], host[MARKER_BASE + 3], comm_slot0,
                host[MARKER_BASE + CACHELINE_WORDS + 2],
                host[MARKER_BASE + CACHELINE_WORDS + 3], comm_slot1);
            std::printf(
                "launch=%u adjacent(0,1): same=%u/%u separate=%u/%u ordered=%u/%u "
                "same_first=0x%08x expected=0x%08x separate_first=0x%08x expected=0x%08x\n",
                launch, host[RESULT_BASE + 0], CHECKS_PER_LAUNCH,
                host[RESULT_BASE + 1], CHECKS_PER_LAUNCH,
                host[RESULT_BASE + 2], CHECKS_PER_LAUNCH,
                host[RESULT_BASE + 3], host[RESULT_BASE + 4],
                host[RESULT_BASE + 14], host[RESULT_BASE + 15]);
        } else {
            std::printf(
                "topology launch=%u: block0(core=%u sub=%u comm_slot=%u) "
                "block1-data-idle(core=%u sub=%u comm_slot=%u) "
                "block2(core=%u sub=%u comm_slot=%u)\n",
                launch, host[MARKER_BASE + 2], host[MARKER_BASE + 3], comm_slot0,
                host[MARKER_BASE + CACHELINE_WORDS + 2],
                host[MARKER_BASE + CACHELINE_WORDS + 3], comm_slot1,
                host[MARKER_BASE + 2 * CACHELINE_WORDS + 2],
                host[MARKER_BASE + 2 * CACHELINE_WORDS + 3], comm_slot2);
            std::printf(
                "launch=%u gap(0,2; AIV1 idle-data): same=%u/%u separate=%u/%u "
                "ordered=%u/%u same_first=0x%08x expected=0x%08x "
                "separate_first=0x%08x expected=0x%08x\n",
                launch, host[RESULT_BASE + 7], CHECKS_PER_LAUNCH,
                host[RESULT_BASE + 8], CHECKS_PER_LAUNCH,
                host[RESULT_BASE + 9], CHECKS_PER_LAUNCH,
                host[RESULT_BASE + 10], host[RESULT_BASE + 11],
                host[RESULT_BASE + 14], host[RESULT_BASE + 15]);
        }
    }

    uint32_t total_single_errors = 0;
    uint32_t total_single_ordered_errors = 0;
    for (uint32_t launch = 0;
         mode == MODE_SINGLE_AIV && storage_aligned && launch < LAUNCHES;
         ++launch) {
        uint32_t host[STORAGE_WORDS] = {0};
        Check(aclrtMemcpy(device_storage, sizeof(host), host, sizeof(host), ACL_MEMCPY_HOST_TO_DEVICE),
              "aclrtMemcpy single init");
        KernelArgs args{
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device_storage)),
            MODE_SINGLE_AIV, 0u};
        Check(aclrtLaunchKernelWithHostArgs(function_handle, 1, stream, nullptr,
                                            &args, sizeof(args), nullptr, 0),
              "aclrtLaunchKernelWithHostArgs single");
        Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream single");
        Check(aclrtMemcpy(host, sizeof(host), device_storage, sizeof(host), ACL_MEMCPY_DEVICE_TO_HOST),
              "aclrtMemcpy single result");

        bool result_header_exact = host[RESULT_BASE + 5] == 100u &&
            host[RESULT_BASE + 6] == 0x351051A5u &&
            host[RESULT_BASE + 12] == MODE_SINGLE_AIV &&
            host[RESULT_BASE + 13] == 1u &&
            host[RESULT_BASE + 14] == 0u && host[RESULT_BASE + 15] == 0u;
        char label[160];
        std::snprintf(label, sizeof(label),
                      "single launch=%u：结果头必须证明单 AIV 模式完整执行 100 trial", launch);
        result.Expect(result_header_exact, label);

        bool participation_exact = host[PARTICIPATION_BASE] == 1u;
        for (uint32_t word = 1; word < CACHELINE_WORDS; ++word) {
            participation_exact = participation_exact && host[PARTICIPATION_BASE + word] == 0u;
        }
        participation_exact = participation_exact &&
            MarkerExact(host, 0, MODE_SINGLE_AIV) &&
            LineIsZero(host, MARKER_BASE + CACHELINE_WORDS) &&
            LineIsZero(host, MARKER_BASE + 2 * CACHELINE_WORDS);
        std::snprintf(label, sizeof(label),
                      "single launch=%u：参与计数仅为 1，只有 block0 marker，block1/2 marker 全零",
                      launch);
        result.Expect(participation_exact, label);
        result.Expect(LineIsZero(host, GUARD_BASE), "单 AIV 模式未使用 guard cacheline 必须全零");

        total_single_errors += host[RESULT_BASE + 0];
        total_single_ordered_errors += host[RESULT_BASE + 1];
        std::printf(
            "single launch=%u core=%u sub=%u comm_slot=%u same-address loop-end=%u/%u "
            "first=0x%08x expected=0x%08x per-write-dsb=%u/%u\n",
            launch, host[MARKER_BASE + 2], host[MARKER_BASE + 3], host[MARKER_BASE + 4],
            host[RESULT_BASE + 0], SINGLE_CHECKS_PER_LAUNCH,
            host[RESULT_BASE + 3], host[RESULT_BASE + 4],
            host[RESULT_BASE + 1], SINGLE_CHECKS_PER_LAUNCH);
    }

    if (mode == MODE_BLOCK01) {
        std::printf(
            "aggregate adjacent(0,1): same=%u/%u separate=%u/%u ordered=%u/%u\n",
            total_adjacent_same_errors, LAUNCHES * CHECKS_PER_LAUNCH,
            total_adjacent_separate_errors, LAUNCHES * CHECKS_PER_LAUNCH,
            total_adjacent_ordered_errors, LAUNCHES * CHECKS_PER_LAUNCH);
        result.Expect(
            total_adjacent_same_errors == 0,
            "AIV0+AIV1 同-line、仅 loop-end DSB：4000 个 slot 终值必须全部精确");
        result.Expect(
            total_adjacent_separate_errors == 0,
            "AIV0+AIV1 分-line control：4000 个 slot 终值必须全部精确");
        result.Expect(
            total_adjacent_ordered_errors == 0,
            "AIV0+AIV1 同-line、逐轮 DSB control：4000 个 slot 终值必须全部精确");
    } else if (mode == MODE_BLOCK02) {
        std::printf(
            "aggregate gap(0,2; AIV1 idle-data): same=%u/%u separate=%u/%u ordered=%u/%u\n",
            total_gap_same_errors, LAUNCHES * CHECKS_PER_LAUNCH,
            total_gap_separate_errors, LAUNCHES * CHECKS_PER_LAUNCH,
            total_gap_ordered_errors, LAUNCHES * CHECKS_PER_LAUNCH);
        result.Expect(
            total_gap_same_errors == 0,
            "AIV0+AIV2 同-line且AIV1全程不访问数据、仅 loop-end DSB：4000 个 slot 终值必须全部精确");
        result.Expect(
            total_gap_separate_errors == 0,
            "AIV0+AIV2 分-line且AIV1全程不访问数据 control：4000 个 slot 终值必须全部精确");
        result.Expect(
            total_gap_ordered_errors == 0,
            "AIV0+AIV2 同-line且AIV1全程不访问数据、逐轮 DSB control：4000 个 slot 终值必须全部精确");
    } else {
        std::printf(
            "aggregate single AIV same-address: loop-end=%u/%u per-write-dsb=%u/%u\n",
            total_single_errors, LAUNCHES * SINGLE_CHECKS_PER_LAUNCH,
            total_single_ordered_errors, LAUNCHES * SINGLE_CHECKS_PER_LAUNCH);
        result.Expect(
            total_single_errors == 0,
            "单 AIV 同一地址连续 257 次 st_dev、仅 loop-end DSB：2000 个终值必须全部精确");
        result.Expect(
            total_single_ordered_errors == 0,
            "单 AIV 同一地址连续 st_dev、逐写 DSB control：2000 个终值必须全部精确");
    }

    Check(aclrtFree(device_storage), "aclrtFree");
    Check(aclrtBinaryUnLoad(binary_handle), "aclrtBinaryUnLoad");
    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    Check(aclrtResetDevice(device), "aclrtResetDevice");
    Check(aclFinalize(), "aclFinalize");
    return result.ExitCode();
}
