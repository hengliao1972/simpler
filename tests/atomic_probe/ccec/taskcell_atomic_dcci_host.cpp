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
// AIV-only TaskCell atomic/DCCI 探针的 host 端精确判定。
//
// 一次分配全部场景和轮次的存储，并用一次 2-AIV launch 顺序执行全部
// 场景。每个试次使用唯一的 10 条 cache line，不复用目标地址。
#include "../probe_host.h"
#include "taskcell_atomic_dcci_shared.h"

#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

using namespace taskcell_atomic_dcci_probe;

namespace {

constexpr uint64_t kPollCountMask = (1ULL << 24) - 1;
constexpr uint16_t kInformationalFlags =
    kFlagPreSawToken | kFlagPostSawToken;

struct PollRecord {
    int64_t first;
    int64_t final;
    uint32_t count;
    bool saw_token;
};

struct ScenarioStats {
    uint32_t exact = 0;
    uint32_t failed = 0;
    uint32_t pre_visible = 0;
    uint32_t post_visible = 0;
    uint32_t snapshot_visible = 0;
    uint32_t host_visible = 0;
    uint32_t clobbered = 0;
    uint32_t survived = 0;
};

void Check(aclError error, const char *label)
{
    if (!atomic_probe::CheckAcl(error, label, __FILE__, __LINE__)) {
        std::exit(EXIT_FAILURE);
    }
}

bool ParseTrials(const char *raw, uint32_t *trials)
{
    errno = 0;
    char *end = nullptr;
    const unsigned long value = std::strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || value == 0 ||
        value > kMaxTrials) {
        std::fprintf(
            stderr, "trials 必须在 [1, %u]，当前输入：%s\n",
            kMaxTrials, raw);
        return false;
    }
    *trials = static_cast<uint32_t>(value);
    return true;
}

const char *ScenarioName(Scenario scenario)
{
    switch (scenario) {
    case Scenario::SharedAtomicThenDcci:
        return "TaskCell 共线：CAS 发布后 DCCI";
    case Scenario::SharedOrdinaryThenDcci:
        return "TaskCell 共线：普通写后 DCCI";
    case Scenario::DedicatedAtomicThenDcci:
        return "deps 独占行：CAS 发布后 DCCI";
    case Scenario::DedicatedOrdinaryThenDcci:
        return "deps 独占行：普通写后 DCCI";
    case Scenario::DedicatedOrdinaryNoDcci:
        return "deps 独占行：普通写且无 DCCI";
    }
    return "未知场景";
}

void InitializeStorage(ProbeStorage *storage, uint32_t trial)
{
    std::memset(storage, 0, sizeof(*storage));
    storage->shared_task.flag = static_cast<int64_t>(InitialFlag(trial));
    storage->shared_task.vend = InitialVend(trial);
    storage->shared_task.deps_prepared = kInitialDeps;
    for (uint32_t index = 0; index < 5; ++index) {
        storage->shared_task.padding_words[index] =
            InitialTaskPadding(index, trial);
    }
    storage->dedicated_deps.deps_prepared = kInitialDeps;
    for (uint32_t index = 0; index < 7; ++index) {
        storage->dedicated_deps.guards[index] =
            DedicatedGuard(index, trial);
    }
    for (uint32_t index = 0; index < 8; ++index) {
        storage->guard.words[index] =
            static_cast<int64_t>(StorageGuard(index, trial));
    }
}

bool ExactAtomicLine(const AtomicLine &line, int64_t expected)
{
    if (line.value != expected) {
        return false;
    }
    for (uint32_t index = 0; index < sizeof(line.padding); ++index) {
        if (line.padding[index] != 0) {
            return false;
        }
    }
    return true;
}

bool ExactSharedTask(
    const TaskCellLine &line, uint32_t trial, int64_t deps, bool built)
{
    const uint64_t expected_flag =
        built ? BuiltFlag(trial) : InitialFlag(trial);
    const uint64_t expected_vend =
        built ? BuiltVend(trial) : InitialVend(trial);
    if (static_cast<uint64_t>(line.flag) != expected_flag ||
        line.vend != expected_vend || line.deps_prepared != deps) {
        return false;
    }
    for (uint32_t index = 0; index < 5; ++index) {
        const uint64_t expected = built ?
            BuiltTaskPadding(index, trial) :
            InitialTaskPadding(index, trial);
        if (line.padding_words[index] != expected) {
            return false;
        }
    }
    return true;
}

bool ExactDedicatedLine(
    const DedicatedDepsLine &line, uint32_t trial, int64_t deps)
{
    if (line.deps_prepared != deps) {
        return false;
    }
    for (uint32_t index = 0; index < 7; ++index) {
        if (line.guards[index] != DedicatedGuard(index, trial)) {
            return false;
        }
    }
    return true;
}

bool ExactGuard(const ResultLine &line, uint32_t trial)
{
    for (uint32_t index = 0; index < 8; ++index) {
        if (static_cast<uint64_t>(line.words[index]) !=
            StorageGuard(index, trial)) {
            return false;
        }
    }
    return true;
}

bool ExactSharedSnapshot(
    const ResultLine &line, uint32_t trial, int64_t deps, bool built)
{
    const uint64_t expected_flag =
        built ? BuiltFlag(trial) : InitialFlag(trial);
    const uint64_t expected_vend =
        built ? BuiltVend(trial) : InitialVend(trial);
    if (static_cast<uint64_t>(line.words[0]) != expected_flag ||
        static_cast<uint64_t>(line.words[1]) != expected_vend ||
        line.words[2] != deps) {
        return false;
    }
    for (uint32_t index = 0; index < 5; ++index) {
        const uint64_t expected = built ?
            BuiltTaskPadding(index, trial) :
            InitialTaskPadding(index, trial);
        if (static_cast<uint64_t>(line.words[index + 3]) != expected) {
            return false;
        }
    }
    return true;
}

bool ExactDedicatedSnapshot(
    const ResultLine &line, uint32_t trial, int64_t deps)
{
    if (line.words[0] != deps) {
        return false;
    }
    for (uint32_t index = 0; index < 7; ++index) {
        if (static_cast<uint64_t>(line.words[index + 1]) !=
            DedicatedGuard(index, trial)) {
            return false;
        }
    }
    return true;
}

uint16_t ReaderFlags(const ResultLine &line)
{
    return static_cast<uint16_t>(
        static_cast<uint64_t>(line.words[7]) >> 48);
}

PollRecord ReaderPre(const ResultLine &line)
{
    const uint64_t meta = static_cast<uint64_t>(line.words[7]);
    const uint16_t flags = ReaderFlags(line);
    return {
        line.words[3],
        line.words[4],
        static_cast<uint32_t>(meta & kPollCountMask),
        (flags & kFlagPreSawToken) != 0,
    };
}

PollRecord ReaderPost(const ResultLine &line)
{
    const uint64_t meta = static_cast<uint64_t>(line.words[7]);
    const uint16_t flags = ReaderFlags(line);
    return {
        line.words[5],
        line.words[6],
        static_cast<uint32_t>((meta >> 24) & kPollCountMask),
        (flags & kFlagPostSawToken) != 0,
    };
}

bool IsKnownValue(int64_t value, int64_t token)
{
    return value == kInitialDeps || value == token;
}

bool IsMonotonicPair(int64_t first, int64_t final, int64_t token)
{
    return IsKnownValue(first, token) && IsKnownValue(final, token) &&
           !(first == token && final == kInitialDeps);
}

bool IsMonotonicSequence(
    const PollRecord &pre, const PollRecord &post, int64_t gm_after,
    int64_t snapshot, int64_t host, int64_t token)
{
    const int64_t values[] = {
        pre.first, pre.final, post.first, post.final, gm_after,
        snapshot, host,
    };
    bool saw_token = false;
    for (int64_t value : values) {
        if (!IsKnownValue(value, token)) {
            return false;
        }
        if (saw_token && value == kInitialDeps) {
            return false;
        }
        saw_token = saw_token || value == token;
    }
    return true;
}

bool ExactResultHeader(
    const ResultLine &line, int64_t magic, Scenario scenario,
    uint32_t trial)
{
    return line.words[0] == magic &&
           static_cast<uint64_t>(line.words[1]) ==
               TrialTag(scenario, trial);
}

bool ReaderInfoMatches(
    const PollRecord &pre, const PollRecord &post)
{
    // Kernel 的轮询器一旦发现 new 后禁止回退，因此 final 足以与 saw
    // 信息位交叉核对；如果出现 new->old，kernel 会另设硬失败 flag。
    return pre.saw_token == (pre.final != kInitialDeps) &&
           post.saw_token == (post.final != kInitialDeps);
}

void PrintLine(const char *label, const volatile int64_t *words)
{
    std::printf("    %s:", label);
    for (uint32_t index = 0; index < 8; ++index) {
        std::printf(
            " %016" PRIx64, static_cast<uint64_t>(words[index]));
    }
    std::printf("\n");
}

void PrintFailure(
    Scenario scenario, uint32_t trial, const ProbeStorage &storage,
    const PollRecord &pre, const PollRecord &post)
{
    std::printf(
        "[异常] scenario=%u(%s) trial=%u token=%016" PRIx64
        " controls=(ready=%lld phase=%lld ack=%lld)"
        " writer_topology=%016" PRIx64
        " reader_topology=%016" PRIx64
        " writer=(cas_old=%lld local=%lld gm_before=%lld gm_after=%lld"
        " flags=0x%04" PRIx64 ")"
        " reader=(pre=%lld->%lld/%u post=%lld->%lld/%u"
        " flags=0x%04x)\n",
        static_cast<uint32_t>(scenario), ScenarioName(scenario), trial,
        static_cast<uint64_t>(PublishedToken(trial)),
        static_cast<long long>(storage.reader_ready.value),
        static_cast<long long>(storage.phase.value),
        static_cast<long long>(storage.reader_ack.value),
        static_cast<uint64_t>(storage.writer_result.words[2]),
        static_cast<uint64_t>(storage.reader_result.words[2]),
        static_cast<long long>(storage.writer_result.words[3]),
        static_cast<long long>(storage.writer_result.words[4]),
        static_cast<long long>(storage.writer_result.words[5]),
        static_cast<long long>(storage.writer_result.words[6]),
        static_cast<uint64_t>(storage.writer_result.words[7]),
        static_cast<long long>(pre.first),
        static_cast<long long>(pre.final), pre.count,
        static_cast<long long>(post.first),
        static_cast<long long>(post.final), post.count,
        ReaderFlags(storage.reader_result));
    PrintLine(
        "target_snapshot",
        storage.target_snapshot.words);
    if (IsSharedScenario(scenario)) {
        PrintLine(
            "host_target",
            reinterpret_cast<const volatile int64_t *>(
                &storage.shared_task));
    } else {
        PrintLine(
            "host_target",
            reinterpret_cast<const volatile int64_t *>(
                &storage.dedicated_deps));
    }
    PrintLine(
        "storage_guard",
        storage.guard.words);
}

bool ValidateTrial(
    Scenario scenario, uint32_t trial, const ProbeStorage &storage,
    ScenarioStats *stats)
{
    const int64_t token = PublishedToken(trial);
    const ResultLine &writer = storage.writer_result;
    const ResultLine &reader = storage.reader_result;
    const PollRecord pre = ReaderPre(reader);
    const PollRecord post = ReaderPost(reader);
    const uint16_t reader_flags = ReaderFlags(reader);

    bool exact =
        ExactAtomicLine(storage.role_claim, kAivBlocks) &&
        ExactAtomicLine(storage.reader_ready, 1) &&
        ExactAtomicLine(storage.phase, 2) &&
        ExactAtomicLine(storage.reader_ack, 2) &&
        ExactResultHeader(
            writer, kWriterMagic, scenario, trial) &&
        ExactResultHeader(
            reader, kReaderMagic, scenario, trial) &&
        writer.words[2] != reader.words[2] &&
        writer.words[7] == kFlagNone &&
        (reader_flags & ~kInformationalFlags) == 0 &&
        ReaderInfoMatches(pre, post) &&
        ExactGuard(storage.guard, trial);

    const bool atomic_publish = IsAtomicPublishScenario(scenario);
    exact = exact &&
        writer.words[3] ==
            (atomic_publish ? kInitialDeps : kNotApplicable) &&
        writer.words[4] ==
            (atomic_publish ? kNotApplicable : token) &&
        writer.words[5] == kNotApplicable;

    if (scenario == Scenario::SharedAtomicThenDcci) {
        const bool pre_exact =
            pre.count > 0 && pre.final == token && pre.saw_token &&
            IsKnownValue(pre.first, token);
        const bool clobbered =
            post.count == 1 && post.first == kInitialDeps &&
            post.final == kInitialDeps && !post.saw_token &&
            writer.words[6] == kInitialDeps &&
            ExactSharedSnapshot(
                storage.target_snapshot, trial, kInitialDeps, true) &&
            ExactSharedTask(
                storage.shared_task, trial, kInitialDeps, true);
        const bool survived =
            post.count == 1 && post.first == token &&
            post.final == token && post.saw_token &&
            writer.words[6] == token &&
            ExactSharedSnapshot(
                storage.target_snapshot, trial, token, true) &&
            ExactSharedTask(storage.shared_task, trial, token, true);
        exact = exact && pre_exact && (clobbered != survived) &&
                ExactDedicatedLine(
                    storage.dedicated_deps, trial, kInitialDeps);
        stats->clobbered += clobbered ? 1U : 0U;
        stats->survived += survived ? 1U : 0U;
    } else if (scenario == Scenario::SharedOrdinaryThenDcci) {
        exact = exact &&
            pre.count == kObservationPolls &&
            IsMonotonicPair(pre.first, pre.final, token) &&
            post.count > 0 &&
            IsMonotonicPair(post.first, post.final, token) &&
            post.final == token && post.saw_token &&
            writer.words[6] == token &&
            ExactSharedSnapshot(
                storage.target_snapshot, trial, token, true) &&
            ExactSharedTask(storage.shared_task, trial, token, true) &&
            ExactDedicatedLine(
                storage.dedicated_deps, trial, kInitialDeps);
    } else if (scenario == Scenario::DedicatedAtomicThenDcci) {
        exact = exact &&
            pre.count > 0 && pre.final == token && pre.saw_token &&
            IsKnownValue(pre.first, token) &&
            post.count == 1 && post.first == token &&
            post.final == token && post.saw_token &&
            writer.words[6] == token &&
            ExactDedicatedSnapshot(
                storage.target_snapshot, trial, token) &&
            ExactDedicatedLine(storage.dedicated_deps, trial, token) &&
            ExactSharedTask(
                storage.shared_task, trial, kInitialDeps, false);
    } else if (scenario == Scenario::DedicatedOrdinaryThenDcci) {
        exact = exact &&
            pre.count == kObservationPolls &&
            IsMonotonicPair(pre.first, pre.final, token) &&
            post.count > 0 &&
            IsMonotonicPair(post.first, post.final, token) &&
            post.final == token && post.saw_token &&
            writer.words[6] == token &&
            ExactDedicatedSnapshot(
                storage.target_snapshot, trial, token) &&
            ExactDedicatedLine(storage.dedicated_deps, trial, token) &&
            ExactSharedTask(
                storage.shared_task, trial, kInitialDeps, false);
    } else {
        const int64_t snapshot = storage.target_snapshot.words[0];
        const int64_t host = storage.dedicated_deps.deps_prepared;
        exact = exact &&
            pre.count == kObservationPolls &&
            post.count == kObservationPolls &&
            IsMonotonicSequence(
                pre, post, writer.words[6], snapshot, host, token) &&
            ExactDedicatedSnapshot(
                storage.target_snapshot, trial, snapshot) &&
            ExactDedicatedLine(storage.dedicated_deps, trial, host) &&
            ExactSharedTask(
                storage.shared_task, trial, kInitialDeps, false);
    }

    stats->pre_visible += pre.saw_token ? 1U : 0U;
    stats->post_visible += post.saw_token ? 1U : 0U;
    stats->snapshot_visible +=
        storage.target_snapshot.words[
            IsSharedScenario(scenario) ? 2 : 0] == token ? 1U : 0U;
    stats->host_visible +=
        (IsSharedScenario(scenario) ?
            storage.shared_task.deps_prepared :
            storage.dedicated_deps.deps_prepared) == token ? 1U : 0U;
    if (exact) {
        ++stats->exact;
    } else {
        ++stats->failed;
        PrintFailure(scenario, trial, storage, pre, post);
    }
    return exact;
}

void PrintStats(
    Scenario scenario, const ScenarioStats &stats, uint32_t trials)
{
    std::printf(
        "[场景%u] %s：exact=%u/%u failed=%u"
        " pre可见=%u post可见=%u snapshot新值=%u host最终新值=%u",
        static_cast<uint32_t>(scenario), ScenarioName(scenario),
        stats.exact, trials, stats.failed, stats.pre_visible,
        stats.post_visible, stats.snapshot_visible, stats.host_visible);
    if (scenario == Scenario::SharedAtomicThenDcci) {
        std::printf(
            " exact_clobbered=%u exact_survived=%u",
            stats.clobbered, stats.survived);
    }
    std::printf("\n");
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc > 3) {
        std::fprintf(
            stderr, "Usage: %s [kernel.o] [trials:1..%u]\n",
            argv[0], kMaxTrials);
        return EXIT_FAILURE;
    }
    const char *kernel_path =
        argc >= 2 ? argv[1] : "./taskcell_atomic_dcci_kernel.o";
    uint32_t trials = kDefaultTrials;
    if (argc == 3 && !ParseTrials(argv[2], &trials)) {
        return EXIT_FAILURE;
    }

    const int32_t device = atomic_probe::DeviceId();
    if (device < 0) {
        return EXIT_FAILURE;
    }
    Check(aclInit(nullptr), "初始化 ACL");
    Check(aclrtSetDevice(device), "设置 TaskCell 探针设备");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "创建 TaskCell 探针 stream");

    std::ifstream file(kernel_path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "无法打开 kernel 文件：%s\n", kernel_path);
        return EXIT_FAILURE;
    }
    const size_t binary_size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<char> binary(binary_size);
    file.read(binary.data(), static_cast<std::streamsize>(binary_size));
    if (!file) {
        std::fprintf(stderr, "读取 kernel 文件失败：%s\n", kernel_path);
        return EXIT_FAILURE;
    }

    aclrtBinHandle binary_handle = nullptr;
    Check(
        atomic_probe::LoadAicoreBinaryFromData(
            binary.data(), binary.size(), &binary_handle),
        "加载 TaskCell atomic/DCCI AICore binary");
    aclrtFuncHandle function = nullptr;
    Check(
        aclrtBinaryGetFunctionByEntry(binary_handle, 0, &function),
        "取得 TaskCell atomic/DCCI kernel 入口");

    const size_t storage_count =
        static_cast<size_t>(kScenarioCount) * trials;
    const size_t storage_bytes = storage_count * sizeof(ProbeStorage);
    std::vector<ProbeStorage> initial(storage_count);
    for (uint32_t scenario = 0; scenario < kScenarioCount; ++scenario) {
        for (uint32_t trial = 0; trial < trials; ++trial) {
            InitializeStorage(
                &initial[static_cast<size_t>(scenario) * trials + trial],
                trial);
        }
    }

    void *device_storage = nullptr;
    Check(
        aclrtMalloc(
            &device_storage, storage_bytes, ACL_MEM_MALLOC_HUGE_FIRST),
        "一次分配全部场景和轮次的 10-line ProbeStorage");
    if ((reinterpret_cast<uintptr_t>(device_storage) &
         (kCacheLineBytes - 1U)) != 0) {
        std::fprintf(
            stderr, "ProbeStorage 首地址没有按 64B 对齐\n");
        return EXIT_FAILURE;
    }
    Check(
        aclrtMemcpy(
            device_storage, storage_bytes, initial.data(), storage_bytes,
            ACL_MEMCPY_HOST_TO_DEVICE),
        "一次初始化全部唯一 ProbeStorage");

    std::printf(
        "=== AIV-only TaskCell::deps_prepared atomic/DCCI 探针 ===\n"
        "scenarios=%u trials/scenario=%u launches=1 blocks/launch=%u "
        "storage=%zu bytes line=%u bytes\n"
        "DCCI 场景固定 SINGLE_CACHE_LINE + CACHELINE_OUT + DSB；"
        "无 DCCI 场景只观察，不把可见/不可见写成架构保证。\n",
        kScenarioCount, trials, kAivBlocks,
        storage_bytes, kCacheLineBytes);

    std::vector<ScenarioStats> stats(kScenarioCount);
    KernelArgs args{
        static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(device_storage)),
        trials,
        kAivBlocks,
    };
    Check(
        aclrtLaunchKernelWithHostArgs(
            function, kAivBlocks, stream, nullptr, &args,
            sizeof(args), nullptr, 0),
        "启动单次 2-AIV TaskCell atomic/DCCI kernel");
    Check(
        aclrtSynchronizeStream(stream),
        "等待全部 TaskCell atomic/DCCI 场景");

    std::vector<ProbeStorage> observed(storage_count);
    Check(
        aclrtMemcpy(
            observed.data(), storage_bytes, device_storage,
            storage_bytes, ACL_MEMCPY_DEVICE_TO_HOST),
        "一次回读全部唯一 ProbeStorage");

    for (uint32_t scenario_raw = 0; scenario_raw < kScenarioCount;
         ++scenario_raw) {
        const Scenario scenario = static_cast<Scenario>(scenario_raw);
        for (uint32_t trial = 0; trial < trials; ++trial) {
            const size_t index =
                static_cast<size_t>(scenario_raw) * trials + trial;
            (void)ValidateTrial(
                scenario, trial, observed[index],
                &stats[scenario_raw]);
        }
        PrintStats(scenario, stats[scenario_raw], trials);
    }

    atomic_probe::Result result;
    result.Expect(
        stats[0].failed == 0 &&
            stats[0].clobbered + stats[0].survived == trials,
        "场景0只允许完整 clobbered 或完整 survived，拒绝 torn/第四态");
    result.Expect(
        stats[1].failed == 0 && stats[1].post_visible == trials,
        "场景1普通写+DCCI 后 reader/完整 TaskCell/guards 全部精确");
    result.Expect(
        stats[2].failed == 0 && stats[2].post_visible == trials &&
            stats[2].snapshot_visible == trials &&
            stats[2].host_visible == trials,
        "场景2 atomic-only 行 DCCI 前后均为 token，guards 全部精确");
    result.Expect(
        stats[3].failed == 0 && stats[3].post_visible == trials,
        "场景3独占行普通写+DCCI 后 token 与 guards 全部精确");
    result.Expect(
        stats[4].failed == 0,
        "场景4无 DCCI 只接受 old/token 单调序列，不预设最终可见性");

    Check(aclrtFree(device_storage), "释放全部 ProbeStorage");
    Check(aclrtBinaryUnLoad(binary_handle), "卸载 TaskCell 探针 binary");
    Check(aclrtDestroyStream(stream), "销毁 TaskCell 探针 stream");
    Check(aclrtResetDevice(device), "重置 TaskCell 探针设备");
    Check(aclFinalize(), "结束 ACL");
    return result.ExitCode();
}
