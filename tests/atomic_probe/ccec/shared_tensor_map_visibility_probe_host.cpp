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
#include "dist_engine/aicpu/shared_tensor_map_init.h"
#include "shared_tensor_map_visibility_probe_shared.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <utility>
#include <vector>

using namespace shared_tensor_map_visibility_probe;

static_assert(PTO_FDWIC_SHARED_MAP == 1, "visibility host must use shared production layout");
static_assert(PTO_FDWIC_TENSORMAP_RING_CAP == kRingCapacity, "visibility host requires CAP=128");
static_assert(sizeof(SharedTensorMapState) == 2113664);

namespace {

void Check(aclError error, const char *label) {
    if (!atomic_probe::CheckAcl(error, label, __FILE__, __LINE__)) {
        std::exit(EXIT_FAILURE);
    }
}

bool OptionalLaunches(uint32_t *launches) {
    const char *raw = std::getenv("ATOMIC_PROBE_VISIBILITY_LAUNCHES");
    if (raw == nullptr || raw[0] == '\0') {
        *launches = kDefaultLaunches;
        return true;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || parsed == 0 || parsed > kMaxLaunches) {
        std::fprintf(stderr, "ATOMIC_PROBE_VISIBILITY_LAUNCHES must be in [1, %u], got %s\n", kMaxLaunches, raw);
        return false;
    }
    *launches = static_cast<uint32_t>(parsed);
    return true;
}

const char *ErrorLabel(ErrorCode code) {
    switch (code) {
    case ErrorCode::None:
        return "none";
    case ErrorCode::InvalidTopology:
        return "invalid-topology";
    case ErrorCode::HashConfiguration:
        return "hash-configuration";
    case ErrorCode::ReadyTimeout:
        return "ready-timeout";
    case ErrorCode::ReadyOvershoot:
        return "ready-overshoot";
    case ErrorCode::PublishFailed:
        return "publish-failed";
    case ErrorCode::CommitTimeout:
        return "commit-timeout";
    case ErrorCode::CommitOvershoot:
        return "commit-overshoot";
    case ErrorCode::CommitMismatch:
        return "commit-mismatch";
    case ErrorCode::ReclaimMismatch:
        return "reclaim-mismatch";
    case ErrorCode::HeadMismatch:
        return "head-mismatch";
    case ErrorCode::TailMismatch:
        return "tail-mismatch";
    case ErrorCode::SequenceMismatch:
        return "sequence-mismatch";
    case ErrorCode::ReadFailed:
        return "read-failed";
    case ErrorCode::PayloadMismatch:
        return "payload-mismatch";
    case ErrorCode::LookupProtocol:
        return "lookup-protocol";
    case ErrorCode::LookupMismatch:
        return "lookup-mismatch";
    case ErrorCode::DoneTimeout:
        return "done-timeout";
    case ErrorCode::DoneOvershoot:
        return "done-overshoot";
    }
    return "unknown";
}

bool LineIsZero(const DiagnosticLine &line) {
    for (uint32_t index = 0; index < 8; ++index) {
        if (line.words[index] != 0) {
            return false;
        }
    }
    return true;
}

SharedTensorMapValue ExpectedValue(uint64_t buffer, int32_t task, uint32_t ordinal) {
    const uint64_t lower = (static_cast<uint64_t>(static_cast<uint32_t>(task)) * 4ULL + ordinal) * 64ULL;
    return {buffer, lower, lower + 32, task, 0};
}

bool ValuesEqual(const SharedTensorMapValue &left, const SharedTensorMapValue &right) {
    return left.buf_addr == right.buf_addr && left.lo == right.lo && left.hi == right.hi &&
           left.producer == right.producer && left.reserved == right.reserved;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s <kernel.o>\n", argv[0]);
        return EXIT_FAILURE;
    }

    uint32_t launches = 0;
    if (!OptionalLaunches(&launches)) {
        return EXIT_FAILURE;
    }
    const int32_t device = atomic_probe::DeviceId();
    if (device < 0) {
        return EXIT_FAILURE;
    }

    Check(aclInit(nullptr), "initialize ACL");
    Check(aclrtSetDevice(device), "set visibility probe device");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "create visibility probe stream");

    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "cannot open kernel file: %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    const size_t binary_size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> binary(binary_size);
    file.read(binary.data(), static_cast<std::streamsize>(binary_size));
    if (!file) {
        return EXIT_FAILURE;
    }

    aclrtBinHandle binary_handle;
    Check(
        atomic_probe::LoadAicoreBinaryFromData(binary.data(), binary.size(), &binary_handle),
        "load visibility probe AICore binary"
    );
    aclrtFuncHandle function_handle;
    Check(aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function_handle), "get visibility probe kernel entry");

    void *device_map = nullptr;
    void *device_control = nullptr;
    Check(
        aclrtMalloc(&device_map, sizeof(SharedTensorMapState), ACL_MEM_MALLOC_HUGE_FIRST),
        "allocate production shared TensorMap state"
    );
    Check(
        aclrtMalloc(&device_control, sizeof(ProbeControl), ACL_MEM_MALLOC_HUGE_FIRST),
        "allocate visibility probe control"
    );

    const uintptr_t map_address = reinterpret_cast<uintptr_t>(device_map);
    const uintptr_t control_address = reinterpret_cast<uintptr_t>(device_control);
    const bool aligned = (map_address & (kCacheLineBytes - 1U)) == 0 && (control_address & (kCacheLineBytes - 1U)) == 0;
    std::printf(
        "=== production shared TensorMap A5 visibility probe ===\n"
        "backend=shared CAP=%u buckets=%u AIVs=%u tasks/launch=%u "
        "reuse_tasks=%u launches=%u timeout=%llu cycles\n"
        "scenario=zero-entry + same-bucket(A0/A1) + different-bucket(A/B) + "
        "alternating-writer + >3 slot laps\n"
        "addresses map=0x%zx(mod64=%zu) control=0x%zx(mod64=%zu)\n",
        kRingCapacity, kMapBuckets, kAivBlocks, kTotalTasks, kReuseTasks, launches,
        static_cast<unsigned long long>(kWaitTimeoutCycles), static_cast<size_t>(map_address),
        static_cast<size_t>(map_address & 63U), static_cast<size_t>(control_address),
        static_cast<size_t>(control_address & 63U)
    );
    if (!aligned) {
        std::fprintf(stderr, "probe allocations are not 64B aligned\n");
        Check(aclrtFree(device_control), "free unaligned visibility control");
        Check(aclrtFree(device_map), "free unaligned visibility map");
        Check(aclrtBinaryUnLoad(binary_handle), "unload visibility binary");
        Check(aclrtDestroyStream(stream), "destroy visibility stream");
        Check(aclrtResetDevice(device), "reset visibility probe device");
        Check(aclFinalize(), "finalize ACL");
        return EXIT_FAILURE;
    }

    auto host_map = std::make_unique<SharedTensorMapState>();
    auto host_control = std::make_unique<ProbeControl>();
    uint64_t protocol_failures = 0;
    uint64_t semantic_failures = 0;
    std::set<std::pair<uint32_t, uint32_t>> first_topology;

    for (uint32_t launch = 0; launch < launches; ++launch) {
        // 使用 production AICPU reset 形成准确控制初态。payload 保留 A5
        // pattern，供 reader 预热旧 cache line；seq 决定其当前不可达。
        std::memset(host_map.get(), 0xa5, sizeof(*host_map));
        dist_shared_tensor_map_reset(*host_map);
        std::memset(host_control.get(), 0, sizeof(*host_control));
        Check(
            aclrtMemcpy(device_map, sizeof(*host_map), host_map.get(), sizeof(*host_map), ACL_MEMCPY_HOST_TO_DEVICE),
            "initialize production shared TensorMap state"
        );
        Check(
            aclrtMemcpy(
                device_control, sizeof(*host_control), host_control.get(), sizeof(*host_control),
                ACL_MEMCPY_HOST_TO_DEVICE
            ),
            "initialize visibility probe control"
        );

        KernelArgs args{
            static_cast<uint64_t>(map_address),
            static_cast<uint64_t>(control_address),
            kAivBlocks,
            launch,
        };
        Check(
            aclrtLaunchKernelWithHostArgs(
                function_handle, kAivBlocks, stream, nullptr, &args, sizeof(args), nullptr, 0
            ),
            "launch production shared TensorMap visibility probe"
        );
        Check(aclrtSynchronizeStream(stream), "wait for production shared TensorMap visibility probe");
        Check(
            aclrtMemcpy(host_map.get(), sizeof(*host_map), device_map, sizeof(*host_map), ACL_MEMCPY_DEVICE_TO_HOST),
            "read production shared TensorMap state"
        );
        Check(
            aclrtMemcpy(
                host_control.get(), sizeof(*host_control), device_control, sizeof(*host_control),
                ACL_MEMCPY_DEVICE_TO_HOST
            ),
            "read visibility probe control"
        );

        const ErrorCode error = static_cast<ErrorCode>(host_control->abort_code.value);
        bool protocol_ok =
            host_control->reader_ready.value == kTotalTasks && host_control->reader_done.value == kTotalTasks &&
            host_control->finish_count.value == kAivBlocks && host_control->first_error_claim.value == 0 &&
            error == ErrorCode::None && LineIsZero(host_control->first_error) &&
            LineIsZero(host_control->first_snapshot) && LineIsZero(host_control->guard);

        std::set<std::pair<uint32_t, uint32_t>> topology;
        for (uint32_t block = 0; block < kAivBlocks; ++block) {
            const ParticipantLine &participant = host_control->participants[block];
            const uint64_t topology_word = static_cast<uint64_t>(participant.words[2]);
            const uint32_t core = static_cast<uint32_t>(topology_word >> 32);
            const uint32_t subblock = static_cast<uint32_t>(topology_word);
            topology.emplace(core, subblock);
            const uint64_t role_counts = static_cast<uint64_t>(participant.words[4]);
            const uint32_t writer_tasks = static_cast<uint32_t>(role_counts >> 32);
            const uint32_t reader_tasks = static_cast<uint32_t>(role_counts);
            protocol_ok = protocol_ok && participant.words[0] == kParticipantMagic && participant.words[1] == block &&
                          participant.words[3] == kTotalTasks && writer_tasks == kTotalTasks / 2 &&
                          reader_tasks == kTotalTasks / 2 && participant.words[5] > 0 && participant.words[6] != 0 &&
                          participant.words[7] == kParticipantFinish;
        }
        protocol_ok = protocol_ok && topology.size() == kAivBlocks;
        if (launch == 0) {
            first_topology = topology;
        }

        const uint64_t expected_tail_a = 2 + kReuseTasks;
        const uint64_t expected_tail_b = kReuseTasks;
        const uint64_t expected_head_a = expected_tail_a - 1;
        const uint64_t expected_head_b = expected_tail_b - 1;
        bool state_ok = host_map->committed_tasks.v == kTotalTasks &&
                        host_map->reclaim_upto.v == static_cast<int64_t>(kTotalTasks) - 2 &&
                        host_map->buckets[kExpectedBucketA].head.v == static_cast<int64_t>(expected_head_a) &&
                        host_map->buckets[kExpectedBucketA].tail.v == static_cast<int64_t>(expected_tail_a) &&
                        host_map->buckets[kExpectedBucketB].head.v == static_cast<int64_t>(expected_head_b) &&
                        host_map->buckets[kExpectedBucketB].tail.v == static_cast<int64_t>(expected_tail_b);

        const uint64_t cursor_a = expected_tail_a - 1;
        const uint64_t cursor_b = expected_tail_b - 1;
        const SharedTensorMapSlot &slot_a =
            host_map->slots[kExpectedBucketA * kRingCapacity + (cursor_a & (kRingCapacity - 1U))];
        const SharedTensorMapSlot &slot_b =
            host_map->slots[kExpectedBucketB * kRingCapacity + (cursor_b & (kRingCapacity - 1U))];
        const SharedTensorMapValue expected_a = ExpectedValue(kBufferA0, kTotalTasks - 1, 0);
        const SharedTensorMapValue expected_b = ExpectedValue(kBufferB, kTotalTasks - 1, 1);
        state_ok = state_ok && slot_a.sequence.v == static_cast<int64_t>(cursor_a) &&
                   slot_b.sequence.v == static_cast<int64_t>(cursor_b) &&
                   ValuesEqual(slot_a.payload.value, expected_a) && ValuesEqual(slot_b.payload.value, expected_b);

        if (!protocol_ok) {
            ++protocol_failures;
        }
        if (!state_ok) {
            ++semantic_failures;
        }
        std::printf(
            "launch=%u error=%lld(%s) first_claim=%lld ready=%lld/%u "
            "done=%lld/%u finish=%lld/%u commit=%lld/%u reclaim=%lld/%d "
            "A(head=%lld tail=%lld seq=%lld) "
            "B(head=%lld tail=%lld seq=%lld) protocol=%s state=%s\n",
            launch, static_cast<long long>(host_control->abort_code.value), ErrorLabel(error),
            static_cast<long long>(host_control->first_error_claim.value),
            static_cast<long long>(host_control->reader_ready.value), kTotalTasks,
            static_cast<long long>(host_control->reader_done.value), kTotalTasks,
            static_cast<long long>(host_control->finish_count.value), kAivBlocks,
            static_cast<long long>(host_map->committed_tasks.v), kTotalTasks,
            static_cast<long long>(host_map->reclaim_upto.v), static_cast<int32_t>(kTotalTasks) - 2,
            static_cast<long long>(host_map->buckets[kExpectedBucketA].head.v),
            static_cast<long long>(host_map->buckets[kExpectedBucketA].tail.v),
            static_cast<long long>(slot_a.sequence.v),
            static_cast<long long>(host_map->buckets[kExpectedBucketB].head.v),
            static_cast<long long>(host_map->buckets[kExpectedBucketB].tail.v),
            static_cast<long long>(slot_b.sequence.v), protocol_ok ? "exact" : "BAD", state_ok ? "exact" : "BAD"
        );

        if (error != ErrorCode::None) {
            const DiagnosticLine &first = host_control->first_error;
            const DiagnosticLine &snapshot = host_control->first_snapshot;
            std::printf(
                "FIRST_ERROR code=%lld(%s) task=%lld block=%lld phase=%lld "
                "actual=%lld expected=%lld aux0=%lld aux1=%lld "
                "snapshot(commit=%lld head=%lld tail=%lld seq=%lld "
                "buf=0x%016llx lo=%lld hi=%lld producer_reserved=0x%016llx)\n",
                static_cast<long long>(first.words[0]), ErrorLabel(static_cast<ErrorCode>(first.words[0])),
                static_cast<long long>(first.words[1]), static_cast<long long>(first.words[2]),
                static_cast<long long>(first.words[3]), static_cast<long long>(first.words[4]),
                static_cast<long long>(first.words[5]), static_cast<long long>(first.words[6]),
                static_cast<long long>(first.words[7]), static_cast<long long>(snapshot.words[0]),
                static_cast<long long>(snapshot.words[1]), static_cast<long long>(snapshot.words[2]),
                static_cast<long long>(snapshot.words[3]), static_cast<unsigned long long>(snapshot.words[4]),
                static_cast<long long>(snapshot.words[5]), static_cast<long long>(snapshot.words[6]),
                static_cast<unsigned long long>(snapshot.words[7])
            );
        }
    }

    std::printf("topology");
    for (const auto &[core, subblock] : first_topology) {
        std::printf(" (core=%u sub=%u)", core, subblock);
    }
    std::printf(
        "\naggregate launches=%u tasks=%llu zero=%u same_bucket=%u "
        "different_bucket=%u writer_rotations=%llu "
        "A_entries=%u A_laps=%.3f B_entries=%u B_laps=%.3f "
        "protocol_failures=%llu semantic_failures=%llu\n",
        launches, static_cast<unsigned long long>(launches) * kTotalTasks, launches, launches, launches * kReuseTasks,
        static_cast<unsigned long long>(launches) * (kTotalTasks - 1U), 2 + kReuseTasks,
        static_cast<double>(2 + kReuseTasks) / kRingCapacity, kReuseTasks,
        static_cast<double>(kReuseTasks) / kRingCapacity, static_cast<unsigned long long>(protocol_failures),
        static_cast<unsigned long long>(semantic_failures)
    );

    atomic_probe::Result result;
    result.Expect(
        protocol_failures == 0, "two AIVs complete every alternating writer/reader epoch without timeout or first error"
    );
    result.Expect(
        semantic_failures == 0,
        "after observing commit, production tail/seq/payload/read/lookup remain exact across three laps"
    );

    Check(aclrtFree(device_control), "free visibility probe control");
    Check(aclrtFree(device_map), "free production shared TensorMap state");
    Check(aclrtBinaryUnLoad(binary_handle), "unload visibility binary");
    Check(aclrtDestroyStream(stream), "destroy visibility stream");
    Check(aclrtResetDevice(device), "reset visibility probe device");
    Check(aclFinalize(), "finalize ACL");
    return result.ExitCode();
}
