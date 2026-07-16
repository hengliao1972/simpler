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

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <set>
#include <utility>
#include <vector>
#include <time.h>

constexpr uint32_t CACHELINE_BYTES = 64;
constexpr uint32_t CACHELINE_WORDS = 16;
constexpr uint32_t NUM_ROUNDS = 64;
constexpr uint32_t MAX_AIVS = 72;
constexpr uint32_t WAIT_TIMEOUT_CYCLES = 20000000;

constexpr uint32_t DATA_BASE = 0 * CACHELINE_WORDS;
constexpr uint32_t CONFIG_BASE = 1 * CACHELINE_WORDS;
constexpr uint32_t CONTROL_BASE = 2 * CACHELINE_WORDS;
constexpr uint32_t READY_BASE = 3 * CACHELINE_WORDS;
constexpr uint32_t ACK_BASE = 4 * CACHELINE_WORDS;
constexpr uint32_t TIMING_HEADER_BASE = 5 * CACHELINE_WORDS;
constexpr uint32_t TIMING_RECORD_BASE = 6 * CACHELINE_WORDS;
constexpr uint32_t TIMING_RECORD_WORDS = 8;
constexpr uint32_t READER_RESULT_BASE = TIMING_RECORD_BASE + NUM_ROUNDS * TIMING_RECORD_WORDS;
constexpr uint32_t GUARD_BASE = READER_RESULT_BASE + MAX_AIVS * CACHELINE_WORDS;
constexpr uint32_t STORAGE_WORDS = GUARD_BASE + CACHELINE_WORDS;

constexpr uint32_t TIMING_MAGIC = 0x4c44464eu;
constexpr uint32_t TIMING_RECORD_MAGIC = 0x54490000u;
constexpr uint32_t PARTICIPANT_MAGIC = 0x4c445250u;
constexpr uint32_t FINISH_MAGIC = 0x46494e49u;
constexpr uint32_t INVALID_U32 = 0xffffffffu;
constexpr uint32_t DEFAULT_LAUNCHES = 3;
constexpr uint32_t MAX_LAUNCHES = 1000;
constexpr double A5_SYS_CNT_HZ = 1000000000.0;

enum TimingHeaderIndex : uint32_t {
    HEADER_MAGIC = 0,
    HEADER_MODE = 1,
    HEADER_ROUNDS = 2,
    HEADER_BLOCKS = 3,
    HEADER_READERS = 4,
    HEADER_COMPLETED_ROUNDS = 5,
    HEADER_READY_TIMEOUTS = 6,
    HEADER_ACK_TIMEOUTS = 7,
    HEADER_FINAL_ACTUAL = 8,
    HEADER_FINAL_EXPECTED = 9,
    HEADER_LAUNCH_ID = 10,
    HEADER_WRITER_CORE = 11,
    HEADER_WRITER_SUBBLOCK = 12,
    HEADER_TIMEOUT_CYCLES = 13,
    HEADER_SEQUENCE_CODE = 14,
    HEADER_FINISH = 15,
};

enum ParticipantIndex : uint32_t {
    PARTICIPANT_HEADER = 0,
    PARTICIPANT_BLOCK = 1,
    PARTICIPANT_CORE = 2,
    PARTICIPANT_SUBBLOCK = 3,
    PARTICIPANT_BLOCK_NUM = 4,
    PARTICIPANT_SEEN = 5,
    PARTICIPANT_DATA_TIMEOUTS = 6,
    PARTICIPANT_CONTROL_TIMEOUTS = 7,
    PARTICIPANT_FIRST_BAD_ROUND = 8,
    PARTICIPANT_FIRST_ACTUAL = 9,
    PARTICIPANT_LAST_ACTUAL = 10,
    PARTICIPANT_TOTAL_POLLS = 11,
    PARTICIPANT_MAX_OBSERVE_CYCLES = 12,
    PARTICIPANT_LAUNCH_ID = 13,
    PARTICIPANT_MODE = 14,
    PARTICIPANT_FINISH = 15,
};

struct KernelArgs {
    uint64_t storage_pointer;
    uint32_t mode;
    uint32_t num_blocks;
};

static_assert(sizeof(KernelArgs) == 16, "unexpected CCEC kernel argument ABI");
static_assert(CONFIG_BASE - DATA_BASE == CACHELINE_WORDS, "data/config must be distinct lines");
static_assert(CONTROL_BASE - CONFIG_BASE == CACHELINE_WORDS, "config/control must be distinct lines");
static_assert(READY_BASE - CONTROL_BASE == CACHELINE_WORDS, "control/ready must be distinct lines");
static_assert(ACK_BASE - READY_BASE == CACHELINE_WORDS, "ready/ack must be distinct lines");
static_assert(TIMING_HEADER_BASE - ACK_BASE == CACHELINE_WORDS, "ack/timing must be distinct lines");
static_assert((READER_RESULT_BASE % CACHELINE_WORDS) == 0, "reader results must start at a cache line");

static void Check(aclError error, const char *label)
{
    if (!atomic_probe::CheckAcl(error, label, __FILE__, __LINE__)) std::exit(EXIT_FAILURE);
}

static bool OptionalUintEnv(
    const char *name, uint32_t default_value, uint32_t minimum, uint32_t maximum, uint32_t *value)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        *value = default_value;
        return true;
    }
    errno = 0;
    char *end = nullptr;
    unsigned long parsed = std::strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || parsed < minimum || parsed > maximum) {
        std::fprintf(stderr, "%s must be in [%u, %u], got: %s\n", name, minimum, maximum, raw);
        return false;
    }
    *value = static_cast<uint32_t>(parsed);
    return true;
}

static uint32_t SequenceValue(uint32_t mode, uint32_t launch_id, uint32_t round)
{
    return 0xa0000000u + (mode + 1u) * 0x01000000u +
        (launch_id & 0xfffu) * 0x1000u + round + 1u;
}

static bool LineIsZero(const std::vector<uint32_t> &storage, uint32_t base)
{
    for (uint32_t word = 0; word < CACHELINE_WORDS; ++word) {
        if (storage[base + word] != 0u) return false;
    }
    return true;
}

static bool UnusedWordsAreZero(
    const std::vector<uint32_t> &storage, uint32_t base, uint32_t first_unused)
{
    for (uint32_t word = first_unused; word < CACHELINE_WORDS; ++word) {
        if (storage[base + word] != 0u) return false;
    }
    return true;
}

struct CycleStats {
    std::vector<uint32_t> samples;

    void Add(uint32_t value)
    {
        samples.push_back(value);
    }

    void Print(const char *label) const
    {
        if (samples.empty()) {
            std::printf("timing %-24s no samples\n", label);
            return;
        }
        std::vector<uint32_t> ordered = samples;
        std::sort(ordered.begin(), ordered.end());
        auto percentile = [&](uint32_t numerator, uint32_t denominator) {
            size_t index = (ordered.size() - 1u) * numerator / denominator;
            return ordered[index];
        };
        uint64_t sum = std::accumulate(ordered.begin(), ordered.end(), uint64_t{0});
        double mean = static_cast<double>(sum) / static_cast<double>(ordered.size());
        double mean_us = mean / (A5_SYS_CNT_HZ / 1000000.0);
        std::printf(
            "timing %-24s n=%zu min=%u p50=%u p95=%u max=%u mean=%.1f cycles (%.3fus @1GHz)\n",
            label, ordered.size(), ordered.front(), percentile(50u, 100u),
            percentile(95u, 100u), ordered.back(), mean, mean_us);
    }
};

static const char *ModeLabel(uint32_t mode)
{
    static const char *const labels[] = {
        "ordinary scalar store -> DSB (no DCCI)",
        "st_dev -> DSB",
        "AtomicExch (no explicit DSB)",
    };
    return labels[mode];
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <kernel.o>\n", argv[0]);
        return EXIT_FAILURE;
    }
    uint32_t mode = 0;
    uint32_t blocks = 0;
    uint32_t launches = 0;
    if (!atomic_probe::RequiredUintEnv("ATOMIC_PROBE_MODE", 3, &mode) ||
        !OptionalUintEnv("ATOMIC_PROBE_AIVS", MAX_AIVS, 2u, MAX_AIVS, &blocks) ||
        !OptionalUintEnv("ATOMIC_PROBE_FANOUT_LAUNCHES", DEFAULT_LAUNCHES, 1u,
                         MAX_LAUNCHES, &launches)) {
        return EXIT_FAILURE;
    }

    int32_t device = atomic_probe::DeviceId();
    if (device < 0) return EXIT_FAILURE;
    Check(aclInit(nullptr), "initialize ACL");
    Check(aclrtSetDevice(device), "set atomic probe device");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "create ld_dev fanout stream");

    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "cannot open kernel file: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    size_t binary_size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> binary(binary_size);
    file.read(binary.data(), static_cast<std::streamsize>(binary_size));
    if (!file) return EXIT_FAILURE;

    aclrtBinHandle binary_handle;
    Check(atomic_probe::LoadAicoreBinaryFromData(binary.data(), binary_size, &binary_handle),
          "load ld_dev fanout AICore binary");
    aclrtFuncHandle function_handle;
    Check(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function_handle),
          "get ld_dev fanout kernel entry");

    void *device_storage = nullptr;
    Check(aclrtMalloc(&device_storage, STORAGE_WORDS * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
          "allocate ld_dev fanout storage");
    uintptr_t storage_address = reinterpret_cast<uintptr_t>(device_storage);
    bool aligned = (storage_address & (CACHELINE_BYTES - 1u)) == 0u;
    std::printf(
        "=== ld_dev fanout publish probe ===\n"
        "mode=%u sequence='%s' launches=%u AIVs=%u readers=%u rounds=%u timeout=%u cycles\n"
        "layout_lines data=0 config=1 control=2 ready=3 ack=4 timing=5..37 reader=38..109 guard=110\n"
        "allocation=0x%zx mod64=%zu mod128=%zu\n",
        mode, ModeLabel(mode), launches, blocks, blocks - 1u, NUM_ROUNDS,
        WAIT_TIMEOUT_CYCLES, static_cast<size_t>(storage_address),
        static_cast<size_t>(storage_address & 63u), static_cast<size_t>(storage_address & 127u));
    if (!aligned) {
        std::fprintf(stderr, "storage is not 64B aligned; refusing to run cache-line probe\n");
        Check(aclrtFree(device_storage), "free unaligned storage");
        Check(aclrtBinaryUnLoad(binary_handle), "unload fanout binary");
        Check(aclrtDestroyStream(stream), "destroy fanout stream");
        Check(aclrtResetDevice(device), "reset probe device");
        Check(aclFinalize(), "finalize ACL");
        return EXIT_FAILURE;
    }

    uint64_t protocol_failures = 0;
    uint64_t expected_seen_total =
        static_cast<uint64_t>(launches) * (blocks - 1u) * NUM_ROUNDS;
    uint64_t seen_total = 0;
    uint64_t data_timeouts = 0;
    uint64_t control_timeouts = 0;
    uint64_t writer_ready_timeouts = 0;
    uint64_t writer_ack_timeouts = 0;
    uint64_t timing_data_mismatches = 0;
    uint64_t final_data_mismatches = 0;
    uint64_t ordinary_local_mismatches = 0;
    double host_us_total = 0.0;
    bool have_first_bad = false;
    uint32_t first_bad_launch = INVALID_U32;
    uint32_t first_bad_reader = INVALID_U32;
    uint32_t first_bad_round = INVALID_U32;
    uint32_t first_bad_actual = 0;
    CycleStats instruction_cycles;
    CycleStats publish_cycles;
    CycleStats end_to_end_cycles;

    for (uint32_t launch = 0; launch < launches; ++launch) {
        std::vector<uint32_t> host(STORAGE_WORDS, 0u);
        host[CONFIG_BASE] = launch;
        Check(aclrtMemcpy(device_storage, STORAGE_WORDS * sizeof(uint32_t), host.data(),
                          STORAGE_WORDS * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE),
              "initialize ld_dev fanout storage");
        KernelArgs args{static_cast<uint64_t>(storage_address), mode, blocks};
        struct timespec start = {}, end = {};
        clock_gettime(CLOCK_MONOTONIC, &start);
        Check(aclrtLaunchKernelWithHostArgs(function_handle, blocks, stream, nullptr,
                                            &args, sizeof(args), nullptr, 0),
              "launch ld_dev fanout kernel");
        Check(aclrtSynchronizeStream(stream), "wait for ld_dev fanout kernel");
        clock_gettime(CLOCK_MONOTONIC, &end);
        host_us_total += (end.tv_sec - start.tv_sec) * 1e6 +
            (end.tv_nsec - start.tv_nsec) / 1e3;
        Check(aclrtMemcpy(host.data(), STORAGE_WORDS * sizeof(uint32_t), device_storage,
                          STORAGE_WORDS * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST),
              "read ld_dev fanout result");

        const uint32_t *header = host.data() + TIMING_HEADER_BASE;
        bool launch_protocol = header[HEADER_MAGIC] == TIMING_MAGIC &&
            header[HEADER_MODE] == mode && header[HEADER_ROUNDS] == NUM_ROUNDS &&
            header[HEADER_BLOCKS] == blocks && header[HEADER_READERS] == blocks - 1u &&
            header[HEADER_COMPLETED_ROUNDS] == NUM_ROUNDS &&
            header[HEADER_LAUNCH_ID] == launch &&
            header[HEADER_TIMEOUT_CYCLES] == WAIT_TIMEOUT_CYCLES &&
            header[HEADER_SEQUENCE_CODE] == mode + 1u && header[HEADER_FINISH] == FINISH_MAGIC;
        writer_ready_timeouts += header[HEADER_READY_TIMEOUTS];
        writer_ack_timeouts += header[HEADER_ACK_TIMEOUTS];
        launch_protocol = launch_protocol && host[CONTROL_BASE] == NUM_ROUNDS &&
            host[CONTROL_BASE + 1u] == FINISH_MAGIC &&
            host[READY_BASE] == (blocks - 1u) * NUM_ROUNDS &&
            host[ACK_BASE] == (blocks - 1u) * NUM_ROUNDS &&
            UnusedWordsAreZero(host, DATA_BASE, 1u) &&
            host[CONFIG_BASE] == launch && UnusedWordsAreZero(host, CONFIG_BASE, 1u) &&
            UnusedWordsAreZero(host, CONTROL_BASE, 2u) &&
            UnusedWordsAreZero(host, READY_BASE, 1u) &&
            UnusedWordsAreZero(host, ACK_BASE, 1u) && LineIsZero(host, GUARD_BASE);

        uint32_t final_expected = SequenceValue(mode, launch, NUM_ROUNDS - 1u);
        if (header[HEADER_FINAL_ACTUAL] != final_expected ||
            header[HEADER_FINAL_EXPECTED] != final_expected || host[DATA_BASE] != final_expected) {
            ++final_data_mismatches;
        }

        for (uint32_t round = 0; round < NUM_ROUNDS; ++round) {
            const uint32_t *record =
                host.data() + TIMING_RECORD_BASE + round * TIMING_RECORD_WORDS;
            uint32_t expected = SequenceValue(mode, launch, round);
            uint32_t ack_target = (blocks - 1u) * (round + 1u);
            bool timing_protocol = record[7] == (TIMING_RECORD_MAGIC | round) &&
                record[3] == ack_target && record[4] == 0u && record[6] == expected;
            launch_protocol = launch_protocol && timing_protocol;
            if (record[5] != expected) ++timing_data_mismatches;
            instruction_cycles.Add(record[0]);
            publish_cycles.Add(record[1]);
            end_to_end_cycles.Add(record[2]);
        }

        std::set<std::pair<uint32_t, uint32_t>> physical_aivs;
        if (launch == 0u) std::printf("topology");
        for (uint32_t participant = 0; participant < blocks; ++participant) {
            const uint32_t *reader =
                host.data() + READER_RESULT_BASE + participant * CACHELINE_WORDS;
            physical_aivs.emplace(reader[PARTICIPANT_CORE], reader[PARTICIPANT_SUBBLOCK]);
            if (launch == 0u) {
                std::printf(" block%u(core=%u sub=%u)", participant,
                            reader[PARTICIPANT_CORE], reader[PARTICIPANT_SUBBLOCK]);
            }
            bool participant_protocol = reader[PARTICIPANT_HEADER] == PARTICIPANT_MAGIC &&
                reader[PARTICIPANT_BLOCK] == participant &&
                reader[PARTICIPANT_BLOCK_NUM] == blocks &&
                reader[PARTICIPANT_LAUNCH_ID] == launch &&
                reader[PARTICIPANT_MODE] == mode && reader[PARTICIPANT_FINISH] == FINISH_MAGIC;
            if (participant == 0u) {
                participant_protocol = participant_protocol &&
                    reader[PARTICIPANT_SEEN] == 0u &&
                    reader[PARTICIPANT_DATA_TIMEOUTS] == 0u &&
                    reader[PARTICIPANT_CONTROL_TIMEOUTS] == 0u &&
                    reader[PARTICIPANT_FIRST_BAD_ROUND] == INVALID_U32;
                if (mode == 0u) {
                    if (reader[PARTICIPANT_LAST_ACTUAL] != final_expected) {
                        ++ordinary_local_mismatches;
                    }
                } else {
                    participant_protocol = participant_protocol &&
                        reader[PARTICIPANT_LAST_ACTUAL] == 0u;
                }
            } else {
                uint32_t participant_seen = reader[PARTICIPANT_SEEN];
                uint32_t participant_data_timeouts = reader[PARTICIPANT_DATA_TIMEOUTS];
                uint32_t participant_control_timeouts = reader[PARTICIPANT_CONTROL_TIMEOUTS];
                seen_total += participant_seen;
                data_timeouts += participant_data_timeouts;
                control_timeouts += participant_control_timeouts;
                participant_protocol = participant_protocol &&
                    participant_seen + participant_data_timeouts == NUM_ROUNDS &&
                    reader[PARTICIPANT_TOTAL_POLLS] > 0u;
                if (participant_data_timeouts == 0u) {
                    participant_protocol = participant_protocol &&
                        reader[PARTICIPANT_FIRST_BAD_ROUND] == INVALID_U32 &&
                        reader[PARTICIPANT_FIRST_ACTUAL] == 0u;
                } else {
                    participant_protocol = participant_protocol &&
                        reader[PARTICIPANT_FIRST_BAD_ROUND] < NUM_ROUNDS;
                    if (!have_first_bad) {
                        have_first_bad = true;
                        first_bad_launch = launch;
                        first_bad_reader = participant;
                        first_bad_round = reader[PARTICIPANT_FIRST_BAD_ROUND];
                        first_bad_actual = reader[PARTICIPANT_FIRST_ACTUAL];
                    }
                }
            }
            launch_protocol = launch_protocol && participant_protocol;
        }
        if (launch == 0u) std::printf("\n");
        launch_protocol = launch_protocol && physical_aivs.size() == blocks;
        for (uint32_t participant = blocks; participant < MAX_AIVS; ++participant) {
            launch_protocol = launch_protocol &&
                LineIsZero(host, READER_RESULT_BASE + participant * CACHELINE_WORDS);
        }
        if (!launch_protocol) ++protocol_failures;

        std::printf(
            "launch=%u kernel_launch=%u writer(core=%u sub=%u) seen=%llu/%llu data_timeout=%llu "
            "control_timeout=%llu ready_timeout=%u ack_timeout=%u final=0x%08x/0x%08x protocol=%s\n",
            launch, header[HEADER_LAUNCH_ID], header[HEADER_WRITER_CORE], header[HEADER_WRITER_SUBBLOCK],
            static_cast<unsigned long long>(seen_total),
            static_cast<unsigned long long>(static_cast<uint64_t>(launch + 1u) *
                                            (blocks - 1u) * NUM_ROUNDS),
            static_cast<unsigned long long>(data_timeouts),
            static_cast<unsigned long long>(control_timeouts),
            header[HEADER_READY_TIMEOUTS], header[HEADER_ACK_TIMEOUTS],
            header[HEADER_FINAL_ACTUAL], final_expected, launch_protocol ? "exact" : "BAD");
    }

    std::printf("\naggregate mode=%u '%s' seen=%llu/%llu data_timeouts=%llu "
                "control_timeouts=%llu ready_timeouts=%llu ack_timeouts=%llu "
                "timing_data_bad=%llu final_bad=%llu ordinary_local_bad=%llu mean_host_launch=%.3fus\n",
                mode, ModeLabel(mode), static_cast<unsigned long long>(seen_total),
                static_cast<unsigned long long>(expected_seen_total),
                static_cast<unsigned long long>(data_timeouts),
                static_cast<unsigned long long>(control_timeouts),
                static_cast<unsigned long long>(writer_ready_timeouts),
                static_cast<unsigned long long>(writer_ack_timeouts),
                static_cast<unsigned long long>(timing_data_mismatches),
                static_cast<unsigned long long>(final_data_mismatches),
                static_cast<unsigned long long>(ordinary_local_mismatches), host_us_total / launches);
    if (have_first_bad) {
        uint32_t expected = SequenceValue(mode, first_bad_launch, first_bad_round);
        std::printf("FIRST_BAD launch=%u reader=%u round=%u actual=0x%08x expected=0x%08x\n",
                    first_bad_launch, first_bad_reader, first_bad_round, first_bad_actual, expected);
    }
    instruction_cycles.Print("writer instruction");
    publish_cycles.Print("writer publish sequence");
    end_to_end_cycles.Print("write-start -> all ack");

    atomic_probe::Result result;
    result.Expect(protocol_failures == 0u,
                  "布局、参与 AIV 的唯一 core/subblock、轮数、epoch、ready/ack、结果与 guard 必须精确");
    result.Expect(control_timeouts == 0u && writer_ready_timeouts == 0u && writer_ack_timeouts == 0u,
                  "独立 cacheline 上的 atomic 控制发布与 ld_dev 轮询不得超时");
    result.Expect(ordinary_local_mismatches == 0u,
                  "ordinary mode 的 writer 本核普通 load 必须看到最终 store，排除 no-op 假失败");
    result.Expect(seen_total == expected_seen_total && data_timeouts == 0u,
                  "每个 reader 必须用 ld_dev 逐轮精确看到唯一 writer 发布的全部序列值");
    result.Expect(timing_data_mismatches == 0u && final_data_mismatches == 0u,
                  "writer 每轮 ld_dev 快照与 kernel 返回后的最终 GM 值必须精确");

    Check(aclrtFree(device_storage), "free ld_dev fanout storage");
    Check(aclrtBinaryUnLoad(binary_handle), "unload ld_dev fanout binary");
    Check(aclrtDestroyStream(stream), "destroy ld_dev fanout stream");
    Check(aclrtResetDevice(device), "reset probe device");
    Check(aclFinalize(), "finalize ACL");
    return result.ExitCode();
}
