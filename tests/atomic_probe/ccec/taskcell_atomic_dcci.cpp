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
// AIV-only TaskCell::deps_prepared 与 DCCI 精确场景探针。
//
// 固定两个 AIV。两个参与者在每个独立 ProbeStorage 的 role_claim line
// 上领取 writer/reader 角色；
// reader_ready、phase、reader_ack 也各自独占 64B atomic-only line，
// 所有控制流量都不会触碰被测 TaskCell 或 dedicated deps line。
// 五类场景和全部 trial 在同一次 kernel 内顺序推进：这既保证每个 trial
// 仍使用从未被 device 访问过的新地址，也规避重复 raw launch 的动态参数
// 更新差异干扰被测 cache 行协议。
//
// 五种场景：
//   0. TaskCell 共线：ordinary 构造旧 deps -> CAS 发布 -> reader 看见新值
//      -> 构造核 DCCI 同一 TaskCell line。
//   1. TaskCell 共线：ordinary 写新 deps -> reader 在 DCCI 前有限轮询
//      -> 构造核 DCCI 同一 TaskCell line。
//   2. deps 独占 line：CAS 发布 -> reader 看见新值 -> 对 atomic-only line DCCI。
//   3. deps 独占 line：ordinary 写新值 -> reader 在 DCCI 前有限轮询 -> DCCI。
//   4. deps 独占 line：ordinary 写新值 -> 全程不 DCCI，作为负对照。
//
// runner 会关闭 scalar 自动 DCCI 和 kernel-end DCCI。任何普通写的可见性
// 只能来自显式 DCCI、自然 writeback/eviction 或设备实现，不能被编译器补写污染。
#include "ccec_utils.h"
#include "taskcell_atomic_dcci_shared.h"

using namespace taskcell_atomic_dcci_probe;

CCEC_PROBE_KERNEL_META(taskcell_atomic_dcci);

namespace {

struct PollResult {
    int64_t first;
    int64_t final;
    uint32_t count;
    bool saw_token;
    bool timed_out;
    bool saw_unexpected;
    bool saw_regression;
};

__aicore__ inline int64_t AtomicLoad(__gm__ volatile int64_t *address)
{
    return atomicAdd(
        const_cast<__gm__ int64_t *>(address), static_cast<int64_t>(0));
}

__aicore__ inline int64_t AtomicCompareExchange(
    __gm__ volatile int64_t *address, int64_t expected, int64_t desired)
{
    return atomicCAS(
        const_cast<__gm__ int64_t *>(address), expected, desired);
}

__aicore__ inline void AtomicStoreResult(
    __gm__ volatile int64_t *address, int64_t value)
{
    (void)atomicExch(
        reinterpret_cast<__gm__ uint64_t *>(
            const_cast<__gm__ int64_t *>(address)),
        static_cast<uint64_t>(value));
}

__aicore__ inline bool WaitForAtLeast(
    __gm__ volatile int64_t *address, int64_t target, uint16_t &flags,
    uint16_t timeout_flag)
{
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    while (true) {
        const int64_t observed = AtomicLoad(address);
        if (observed >= target) {
            if (observed != target) {
                flags |= kFlagControlTransition;
            }
            return observed == target;
        }
        if (static_cast<uint64_t>(get_sys_cnt()) - begin >=
            kWaitTimeoutTicks) {
            flags |= timeout_flag;
            return false;
        }
    }
}

__aicore__ inline void AdvanceControl(
    __gm__ volatile int64_t *address, int64_t expected, int64_t desired,
    uint16_t &flags)
{
    if (AtomicCompareExchange(address, expected, desired) != expected) {
        flags |= kFlagControlTransition;
    }
}

__aicore__ inline PollResult PollFixed(
    __gm__ volatile int64_t *address, int64_t token)
{
    PollResult result{
        kNotApplicable, kNotApplicable, 0, false, false, false, false};
    for (uint32_t index = 0; index < kObservationPolls; ++index) {
        const int64_t value = AtomicLoad(address);
        if (index == 0) {
            result.first = value;
        }
        result.final = value;
        ++result.count;
        if (value == token) {
            result.saw_token = true;
        } else if (value == kInitialDeps) {
            if (result.saw_token) {
                result.saw_regression = true;
            }
        } else {
            result.saw_unexpected = true;
        }
    }
    return result;
}

__aicore__ inline PollResult PollOnce(
    __gm__ volatile int64_t *address, int64_t token)
{
    const int64_t value = AtomicLoad(address);
    return PollResult{
        value,
        value,
        1,
        value == token,
        false,
        value != kInitialDeps && value != token,
        false};
}

__aicore__ inline PollResult PollUntilToken(
    __gm__ volatile int64_t *address, int64_t token)
{
    PollResult result{
        kNotApplicable, kNotApplicable, 0, false, false, false, false};
    const uint64_t begin = static_cast<uint64_t>(get_sys_cnt());
    while (true) {
        const int64_t value = AtomicLoad(address);
        if (result.count == 0) {
            result.first = value;
        }
        result.final = value;
        ++result.count;
        if (value == token) {
            result.saw_token = true;
            return result;
        }
        if (value != kInitialDeps) {
            result.saw_unexpected = true;
        }
        if (static_cast<uint64_t>(get_sys_cnt()) - begin >=
            kWaitTimeoutTicks) {
            result.timed_out = true;
            return result;
        }
    }
}

__aicore__ inline void FlushSingleLine(__gm__ volatile int64_t *address)
{
    // CCEC 9.1 不能把 dcci/dsb 当作普通 GM 的 compiler barrier。
    // 前后空汇编只约束编译器；DCCI+DSB 才是被测设备动作。
    __asm__ volatile("" ::: "memory");
    dcci(
        reinterpret_cast<__gm__ uint8_t *>(
            const_cast<__gm__ int64_t *>(address)),
        SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb(DSB_ALL);
    __asm__ volatile("" ::: "memory");
}

__aicore__ inline void CompleteBeforePublish()
{
    // DSB 负责设备侧完成，空汇编的 memory clobber 负责禁止 CCEC
    // 把 ordinary GM 访问跨过这条完成边界移动。
    __asm__ volatile("" ::: "memory");
    dsb(DSB_ALL);
    __asm__ volatile("" ::: "memory");
}

__aicore__ inline void BuildSharedTask(
    __gm__ TaskCellLine *task, uint32_t trial, int64_t deps)
{
    task->flag = static_cast<int64_t>(BuiltFlag(trial));
    task->vend = BuiltVend(trial);
    task->deps_prepared = deps;
    for (uint32_t index = 0; index < 5; ++index) {
        task->padding_words[index] = BuiltTaskPadding(index, trial);
    }
}

__aicore__ inline uint64_t PackTopology()
{
    return
        (static_cast<uint64_t>(static_cast<uint32_t>(get_coreid())) << 32) |
        static_cast<uint32_t>(get_subblockid());
}

__aicore__ inline uint64_t PackPollMeta(
    uint32_t pre_count, uint32_t post_count, uint16_t flags)
{
    constexpr uint64_t kCountMask = (1ULL << 24) - 1;
    return (static_cast<uint64_t>(flags) << 48) |
           ((static_cast<uint64_t>(post_count) & kCountMask) << 24) |
           (static_cast<uint64_t>(pre_count) & kCountMask);
}

__aicore__ inline void PublishWriterResult(
    __gm__ ResultLine &line, Scenario scenario, uint32_t trial,
    int64_t cas_old, int64_t local_after, int64_t gm_before,
    int64_t gm_after, uint16_t flags)
{
    AtomicStoreResult(
        &line.words[1],
        static_cast<int64_t>(TrialTag(scenario, trial)));
    AtomicStoreResult(
        &line.words[2], static_cast<int64_t>(PackTopology()));
    AtomicStoreResult(&line.words[3], cas_old);
    AtomicStoreResult(&line.words[4], local_after);
    AtomicStoreResult(&line.words[5], gm_before);
    AtomicStoreResult(&line.words[6], gm_after);
    AtomicStoreResult(&line.words[7], static_cast<int64_t>(flags));
    dsb(DSB_ALL);
    AtomicStoreResult(&line.words[0], kWriterMagic);
}

__aicore__ inline void PublishReaderResult(
    __gm__ ResultLine &line, Scenario scenario, uint32_t trial,
    const PollResult &pre, const PollResult &post, uint16_t flags)
{
    if (pre.saw_token) {
        flags |= kFlagPreSawToken;
    }
    if (post.saw_token) {
        flags |= kFlagPostSawToken;
    }
    if (pre.timed_out) {
        flags |= kFlagPrePollTimeout;
    }
    if (post.timed_out) {
        flags |= kFlagPostPollTimeout;
    }
    if (pre.saw_unexpected || post.saw_unexpected) {
        flags |= kFlagUnexpectedPollValue;
    }
    if (pre.saw_regression || post.saw_regression) {
        flags |= kFlagPollRegression;
    }

    AtomicStoreResult(
        &line.words[1],
        static_cast<int64_t>(TrialTag(scenario, trial)));
    AtomicStoreResult(
        &line.words[2], static_cast<int64_t>(PackTopology()));
    AtomicStoreResult(&line.words[3], pre.first);
    AtomicStoreResult(&line.words[4], pre.final);
    AtomicStoreResult(&line.words[5], post.first);
    AtomicStoreResult(&line.words[6], post.final);
    AtomicStoreResult(
        &line.words[7],
        static_cast<int64_t>(
            PackPollMeta(pre.count, post.count, flags)));
    dsb(DSB_ALL);
    AtomicStoreResult(&line.words[0], kReaderMagic);
}

__aicore__ inline void CaptureTargetSnapshot(
    __gm__ volatile int64_t *target, __gm__ ResultLine &snapshot)
{
    __gm__ uint64_t *line =
        reinterpret_cast<__gm__ uint64_t *>(
            const_cast<__gm__ int64_t *>(target));
    for (uint32_t index = 0; index < 8; ++index) {
        AtomicStoreResult(
            &snapshot.words[index],
            static_cast<int64_t>(ld_dev_b64(&line[index])));
    }
    dsb(DSB_ALL);
}

__aicore__ inline void RunWriter(
    __gm__ ProbeStorage &storage, Scenario scenario, uint32_t trial,
    uint32_t num_blocks)
{
    uint16_t flags = kFlagNone;
    if (num_blocks != kAivBlocks) {
        flags |= kFlagInvalidTopology;
    }
    (void)WaitForAtLeast(
        &storage.reader_ready.value, 1, flags, kFlagReadyTimeout);

    __gm__ volatile int64_t *target = IsSharedScenario(scenario) ?
        &storage.shared_task.deps_prepared :
        &storage.dedicated_deps.deps_prepared;
    __gm__ volatile int64_t *snapshot_base = IsSharedScenario(scenario) ?
        &storage.shared_task.flag :
        &storage.dedicated_deps.deps_prepared;
    const int64_t token = PublishedToken(trial);
    int64_t cas_old = kNotApplicable;
    int64_t local_after = kNotApplicable;

    if (scenario == Scenario::SharedAtomicThenDcci) {
        // 先用 ordinary store 构造完整 TaskCell dirty 快照，其中 deps
        // 仍为 -1；随后 CAS 只更新 GM/atomic 路径。
        BuildSharedTask(&storage.shared_task, trial, kInitialDeps);
        CompleteBeforePublish();
        cas_old = AtomicCompareExchange(target, kInitialDeps, token);
        if (cas_old != kInitialDeps) {
            flags |= kFlagAtomicPublishFailed;
        }
        CompleteBeforePublish();
    } else if (scenario == Scenario::SharedOrdinaryThenDcci) {
        BuildSharedTask(&storage.shared_task, trial, token);
        CompleteBeforePublish();
        local_after = storage.shared_task.deps_prepared;
    } else if (scenario == Scenario::DedicatedAtomicThenDcci) {
        // atomic-only 对照禁止任何 ordinary load/store 接触整条 deps line。
        cas_old = AtomicCompareExchange(target, kInitialDeps, token);
        if (cas_old != kInitialDeps) {
            flags |= kFlagAtomicPublishFailed;
        }
        CompleteBeforePublish();
    } else {
        storage.dedicated_deps.deps_prepared = token;
        CompleteBeforePublish();
        local_after = storage.dedicated_deps.deps_prepared;
    }

    __asm__ volatile("" ::: "memory");
    AdvanceControl(&storage.phase.value, 0, 1, flags);
    (void)WaitForAtLeast(
        &storage.reader_ack.value, 1, flags, kFlagAckTimeout);

    // pre 状态已经由远端 reader 的 atomic poll 取证。这里不额外插入
    // ld_dev，确保 DCCI 前没有第三条目标访问路径。
    const int64_t gm_before = kNotApplicable;
    if (UsesDcci(scenario)) {
        FlushSingleLine(target);
    }
    AdvanceControl(&storage.phase.value, 1, 2, flags);
    (void)WaitForAtLeast(
        &storage.reader_ack.value, 2, flags, kFlagAckTimeout);

    // 远端 reader 的 post atomic poll 必须是 DCCI 完成后的第一个目标
    // 访问；等它取证并 ack 后，writer 才做旁路整线快照。
    const int64_t gm_after =
        static_cast<int64_t>(
            ld_dev_b64(
                reinterpret_cast<__gm__ uint64_t *>(
                    const_cast<__gm__ int64_t *>(target))));
    CaptureTargetSnapshot(snapshot_base, storage.target_snapshot);

    PublishWriterResult(
        storage.writer_result, scenario, trial, cas_old, local_after,
        gm_before, gm_after, flags);
}

__aicore__ inline void RunReader(
    __gm__ ProbeStorage &storage, Scenario scenario, uint32_t trial,
    uint32_t num_blocks)
{
    uint16_t flags = kFlagNone;
    if (num_blocks != kAivBlocks) {
        flags |= kFlagInvalidTopology;
    }
    AdvanceControl(&storage.reader_ready.value, 0, 1, flags);
    (void)WaitForAtLeast(
        &storage.phase.value, 1, flags, kFlagPhaseTimeout);

    __gm__ volatile int64_t *target =
        IsSharedScenario(scenario) ?
            &storage.shared_task.deps_prepared :
            &storage.dedicated_deps.deps_prepared;
    const int64_t token = PublishedToken(trial);
    const PollResult pre = IsAtomicPublishScenario(scenario) ?
        PollUntilToken(target, token) :
        PollFixed(target, token);

    AdvanceControl(&storage.reader_ack.value, 0, 1, flags);
    (void)WaitForAtLeast(
        &storage.phase.value, 2, flags, kFlagPhaseTimeout);

    PollResult post{};
    if (scenario == Scenario::SharedAtomicThenDcci ||
        scenario == Scenario::DedicatedAtomicThenDcci) {
        // atomic 发布在 DCCI 前已由 pre 窗口确认；这里只取 DCCI 后状态，
        // 不用长轮询掩盖“新值被覆盖”的瞬时事实。
        post = PollOnce(target, token);
    } else if (scenario == Scenario::DedicatedOrdinaryNoDcci) {
        post = PollFixed(target, token);
    } else {
        post = PollUntilToken(target, token);
    }

    PublishReaderResult(
        storage.reader_result, scenario, trial, pre, post, flags);
    AdvanceControl(&storage.reader_ack.value, 1, 2, flags);
}

}  // namespace

extern "C" __global__ __aicore__ void KERNEL_ENTRY(taskcell_atomic_dcci)(
    __gm__ ProbeStorage *storage, uint32_t trials, uint32_t num_blocks)
{
    if (storage == nullptr || trials == 0 || trials > kMaxTrials) {
        return;
    }
    const uint32_t actual_blocks =
        static_cast<uint32_t>(get_block_num());
    const uint32_t checked_blocks =
        actual_blocks == num_blocks ? num_blocks : 0;

    for (uint32_t scenario_raw = 0; scenario_raw < kScenarioCount;
         ++scenario_raw) {
        const Scenario scenario = static_cast<Scenario>(scenario_raw);
        for (uint32_t trial = 0; trial < trials; ++trial) {
            __gm__ ProbeStorage &current =
                storage[static_cast<uint64_t>(scenario_raw) * trials +
                        trial];
            const int64_t role = atomicAdd(
                const_cast<__gm__ int64_t *>(
                    &current.role_claim.value),
                static_cast<int64_t>(1));
            if (role == 0) {
                RunWriter(
                    current, scenario, trial, checked_blocks);
            } else if (role == 1) {
                RunReader(
                    current, scenario, trial, checked_blocks);
            }
        }
    }
}
