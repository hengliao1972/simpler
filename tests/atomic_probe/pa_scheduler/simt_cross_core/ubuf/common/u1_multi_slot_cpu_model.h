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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_U1_MULTI_SLOT_CPU_MODEL_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_U1_MULTI_SLOT_CPU_MODEL_H

#include "u1_multi_slot.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <thread>

namespace pa_scheduler::simt_cross_core::u1::cpu {

enum class BuildStage : uint32_t {
    AfterClaim = 0U,
    AfterSlotAcquire = 1U,
    AfterHalfStaging = 2U,
    AfterStaging = 3U,
    AfterAnchorArrive = 4U,
    AfterAnchorGate = 5U,
    AfterHalfGmStore = 6U,
    BeforePublish = 7U,
    AfterPublishBeforeRelease = 8U,
    AfterRelease = 9U,
};

enum class BuildOutcome : uint32_t {
    Built = 0U,
    InvalidLeader = 1U,
    ClaimLost = 2U,
    SlotTimeout = 3U,
    AnchorTimeout = 4U,
    Aborted = 5U,
    GuardCorruption = 6U,
    PublishConflict = 7U,
    ReleaseConflict = 8U,
};

enum class ExecOutcome : uint32_t {
    Done = 0U,
    InvalidTask = 1U,
    WaitTimeout = 2U,
    ClaimLost = 3U,
    PayloadMismatch = 4U,
    CompletionConflict = 5U,
    InvalidState = 6U,
    Aborted = 7U,
};

using BuildHook = std::function<void(BuildStage, uint32_t, uint32_t)>;

struct BuildOptions {
    uint32_t slot_poll_budget = 0U;
    uint32_t anchor_poll_budget = 0U;
    BuildHook hook{};
};

struct ExecOptions {
    uint32_t wait_poll_budget = 0U;
};

struct WarpBuildResult {
    BuildOutcome first = BuildOutcome::InvalidLeader;
    BuildOutcome second = BuildOutcome::InvalidLeader;
};

struct alignas(kCacheLineBytes) CpuTaskControl {
    std::atomic<uint64_t> state{EmptyState()};
    std::array<uint8_t, kCacheLineBytes - sizeof(std::atomic<uint64_t>)> padding{};
};

struct alignas(kCacheLineBytes) CpuTask {
    CpuTaskControl control;
    alignas(kCacheLineBytes) std::array<uint64_t, kMaxPayloadWords> payload{};
    U1BuildReport build_report{};
    U1ExecReport exec_report{};
};

struct alignas(kCacheLineBytes) CpuStagingSlot {
    U1Guard guard_before{};
    alignas(kCacheLineBytes) std::array<uint64_t, kMaxPayloadWords> payload{};
    U1Guard guard_after{};
};

static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t), "U1 CPU model requires lock-sized uint64 atomic");
static_assert(offsetof(CpuTask, payload) == kCacheLineBytes, "U1 CPU payload must not share its control line");
static_assert(sizeof(CpuStagingSlot) == kUbufSlotStrideBytes, "U1 CPU staging slot must match the UBUF ABI");

class MultiSlotModel {
public:
    explicit MultiSlotModel(uint64_t nonce) :
        nonce_(nonce) {
        for (CpuTask &task : tasks_) {
            task.control.state.store(EmptyState(), std::memory_order_relaxed);
            task.payload.fill(kPayloadPoisonWord);
            PoisonObject(task.build_report);
            PoisonObject(task.exec_report);
        }
        for (U1ThreadReport &report : builder_threads_) {
            PoisonObject(report);
        }
        for (uint32_t slot = 0U; slot < kUbufSlotCount; ++slot) {
            slot_states_[slot].store(SlotFreeState(0U), std::memory_order_relaxed);
            staging_slots_[slot].payload.fill(kPayloadPoisonWord);
            InitializeSlotGuards(slot);
        }
        InitializeGuard(guard_before_tasks_, kGuardBeforeTasks);
        InitializeGuard(guard_after_tasks_, kGuardAfterTasks);
        InitializeGuard(guard_before_builder_threads_, kGuardBeforeBuilderThreads);
        InitializeGuard(guard_after_builder_threads_, kGuardAfterBuilderThreads);
    }

    MultiSlotModel(const MultiSlotModel &) = delete;
    MultiSlotModel &operator=(const MultiSlotModel &) = delete;

    BuildOutcome BuildTask(uint32_t task_id, uint32_t thread_id, const BuildOptions &options = {}) {
        return BuildTaskImpl(task_id, thread_id, options).outcome;
    }

    WarpBuildResult BuildWarpLeader(uint32_t thread_id, const BuildOptions &options = {}) {
        WarpBuildResult result{};
        if (!ValidWarpLeader(thread_id)) {
            return result;
        }

        const uint32_t warp_id = thread_id / kWarpSize;
        const uint32_t first_task = FirstTaskForWarp(warp_id);
        const uint32_t second_task = LastTaskForWarp(warp_id);
        const BuildAttempt first = BuildTaskImpl(first_task, thread_id, options);
        result.first = first.outcome;
        BuildAttempt second{};
        if (first.outcome == BuildOutcome::Built) {
            second = BuildTaskImpl(second_task, thread_id, options);
            result.second = second.outcome;
        } else {
            result.second = BuildOutcome::Aborted;
        }

        const uint32_t build_count = static_cast<uint32_t>(result.first == BuildOutcome::Built) +
                                     static_cast<uint32_t>(result.second == BuildOutcome::Built);
        uint32_t status = kThreadActiveLeader;
        if (build_count == kTasksPerWarp) {
            status = kExpectedBuilderThreadStatus;
        }
        WriteThreadReport(
            thread_id, first_task, second_task, status, first.slot_attempts + second.slot_attempts, build_count
        );
        builder_finished_count_.fetch_add(1U, std::memory_order_release);
        return result;
    }

    ExecOutcome ExecuteTask(uint32_t task_id, const ExecOptions &options = {}) {
        if (task_id >= kTaskCount) {
            return ExecOutcome::InvalidTask;
        }
        if (fatal_.load(std::memory_order_acquire) != 0U) {
            return ExecOutcome::Aborted;
        }

        uint32_t polls = 0U;
        uint64_t observed = 0U;
        for (;;) {
            if (fatal_.load(std::memory_order_acquire) != 0U) {
                return ExecOutcome::Aborted;
            }
            observed = tasks_[task_id].control.state.load(std::memory_order_acquire);
            if (observed == BuiltState(task_id)) {
                break;
            }
            if (observed != EmptyState() && observed != BuildingState(task_id)) {
                ReportFatal(U1FatalReason::InvalidTaskState, kExecutorOwner, task_id);
                return ExecOutcome::InvalidState;
            }
            if (fatal_.load(std::memory_order_acquire) != 0U) {
                return ExecOutcome::Aborted;
            }
            ++polls;
            if (options.wait_poll_budget != 0U && polls >= options.wait_poll_budget) {
                ReportFatal(U1FatalReason::TaskWaitTimeout, kExecutorOwner, task_id);
                return ExecOutcome::WaitTimeout;
            }
            PollYield(polls);
        }

        uint64_t expected = BuiltState(task_id);
        if (!tasks_[task_id].control.state.compare_exchange_strong(
                expected, ClaimedState(task_id), std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            ReportFatal(U1FatalReason::TaskClaimConflict, kExecutorOwner, task_id);
            return ExecOutcome::ClaimLost;
        }
        if (fatal_.load(std::memory_order_acquire) != 0U) {
            expected = ClaimedState(task_id);
            if (!tasks_[task_id].control.state.compare_exchange_strong(
                    expected, BuiltState(task_id), std::memory_order_release, std::memory_order_acquire
                )) {
                ReportFatal(U1FatalReason::TaskClaimConflict, kExecutorOwner, task_id);
            }
            return ExecOutcome::Aborted;
        }
        exec_claim_count_.fetch_add(1U, std::memory_order_relaxed);

        uint64_t checksum = 0x243F6A8885A308D3ULL ^ static_cast<uint64_t>(task_id);
        bool matches = true;
        const uint32_t payload_words = PayloadWordsForTask(task_id);
        for (uint32_t word = 0U; word < payload_words; ++word) {
            const uint64_t observed_word = tasks_[task_id].payload[word];
            matches = matches && observed_word == ExpectedPayloadWord(nonce_, task_id, word);
            checksum = FoldChecksum(checksum, observed_word, word);
        }
        matches = matches && checksum == ExpectedPayloadChecksum(nonce_, task_id);
        uint32_t phase_bits = kExecClaimed | kExecPayloadRead;
        if (!matches) {
            ReportFatal(U1FatalReason::PayloadMismatch, kExecutorOwner, task_id);
            WriteExecReport(task_id, phase_bits, payload_words, 0U, checksum);
            return ExecOutcome::PayloadMismatch;
        }
        if (fatal_.load(std::memory_order_acquire) != 0U) {
            expected = ClaimedState(task_id);
            if (!tasks_[task_id].control.state.compare_exchange_strong(
                    expected, BuiltState(task_id), std::memory_order_release, std::memory_order_acquire
                )) {
                ReportFatal(U1FatalReason::TaskCompleteConflict, kExecutorOwner, task_id);
            }
            return ExecOutcome::Aborted;
        }
        phase_bits |= kExecPayloadValid | kExecCompleted;
        WriteExecReport(task_id, phase_bits, payload_words, 1U, checksum);

        expected = ClaimedState(task_id);
        if (!tasks_[task_id].control.state.compare_exchange_strong(
                expected, DoneState(task_id), std::memory_order_release, std::memory_order_acquire
            )) {
            ReportFatal(U1FatalReason::TaskCompleteConflict, kExecutorOwner, task_id);
            return ExecOutcome::CompletionConflict;
        }
        done_count_.fetch_add(1U, std::memory_order_acq_rel);
        return ExecOutcome::Done;
    }

    bool ExecuteAll(const ExecOptions &options = {}) {
        for (uint32_t task_id = 0U; task_id < kTaskCount; ++task_id) {
            if (ExecuteTask(task_id, options) != ExecOutcome::Done) {
                executor_finished_count_.fetch_add(1U, std::memory_order_release);
                return false;
            }
        }
        executor_finished_count_.fetch_add(1U, std::memory_order_release);
        return true;
    }

    bool TryReadPublishedPayload(uint32_t task_id, uint64_t &checksum, uint32_t &words_read) const {
        checksum = 0U;
        words_read = 0U;
        if (task_id >= kTaskCount) {
            return false;
        }
        const uint64_t state = tasks_[task_id].control.state.load(std::memory_order_acquire);
        const g0::DecodedExecState decoded = DecodeExecState(state);
        if (!decoded.valid || decoded.task_id != task_id ||
            (decoded.phase != ExecPhase::Built && decoded.phase != ExecPhase::Claimed &&
             decoded.phase != ExecPhase::Done)) {
            return false;
        }

        checksum = 0x243F6A8885A308D3ULL ^ static_cast<uint64_t>(task_id);
        const uint32_t payload_words = PayloadWordsForTask(task_id);
        for (uint32_t word = 0U; word < payload_words; ++word) {
            checksum = FoldChecksum(checksum, tasks_[task_id].payload[word], word);
            ++words_read;
        }
        return true;
    }

    uint64_t nonce() const { return nonce_; }
    uint64_t task_state(uint32_t task_id) const {
        return tasks_[task_id].control.state.load(std::memory_order_acquire);
    }
    uint64_t payload_word(uint32_t task_id, uint32_t word) const { return tasks_[task_id].payload[word]; }
    uint64_t staging_word(uint32_t slot_id, uint32_t word) const { return staging_slots_[slot_id].payload[word]; }
    const U1BuildReport &build_report(uint32_t task_id) const { return tasks_[task_id].build_report; }
    const U1ExecReport &exec_report(uint32_t task_id) const { return tasks_[task_id].exec_report; }
    const U1ThreadReport &builder_thread_report(uint32_t thread_id) const { return builder_threads_[thread_id]; }

    uint64_t slot_state(uint32_t slot_id) const { return slot_states_[slot_id].load(std::memory_order_acquire); }
    uint64_t slot_busy_depth(uint32_t slot_id) const {
        return slot_busy_depth_[slot_id].load(std::memory_order_acquire);
    }
    uint64_t slot_max_busy_depth(uint32_t slot_id) const {
        return slot_max_busy_depth_[slot_id].load(std::memory_order_acquire);
    }
    uint64_t slot_acquire_count(uint32_t slot_id) const {
        return slot_acquire_count_[slot_id].load(std::memory_order_acquire);
    }
    uint64_t slot_release_count(uint32_t slot_id) const {
        return slot_release_count_[slot_id].load(std::memory_order_acquire);
    }
    uint64_t global_busy_depth() const { return global_busy_depth_.load(std::memory_order_acquire); }
    uint64_t global_max_busy_depth() const { return global_max_busy_depth_.load(std::memory_order_acquire); }
    uint64_t anchor_staged_count() const { return anchor_staged_count_.load(std::memory_order_acquire); }
    uint64_t anchor_staged_mask() const { return anchor_staged_mask_.load(std::memory_order_acquire); }
    uint64_t anchor_waiting_count() const { return anchor_waiting_count_.load(std::memory_order_acquire); }
    uint64_t ubuf_guard_check_count() const { return ubuf_guard_check_count_.load(std::memory_order_acquire); }
    uint64_t build_claim_count() const { return build_claim_count_.load(std::memory_order_acquire); }
    uint64_t built_count() const { return built_count_.load(std::memory_order_acquire); }
    uint64_t exec_claim_count() const { return exec_claim_count_.load(std::memory_order_acquire); }
    uint64_t done_count() const { return done_count_.load(std::memory_order_acquire); }
    uint64_t builder_finished_count() const { return builder_finished_count_.load(std::memory_order_acquire); }
    uint64_t executor_finished_count() const { return executor_finished_count_.load(std::memory_order_acquire); }
    uint64_t fatal() const { return fatal_.load(std::memory_order_acquire); }
    uint64_t mte3_count() const { return 0U; }

    void SetTaskStateForTest(uint32_t task_id, uint64_t state) {
        tasks_[task_id].control.state.store(state, std::memory_order_release);
    }

    void SetPayloadWordForTest(uint32_t task_id, uint32_t word, uint64_t value) {
        tasks_[task_id].payload[word] = value;
    }

    void SetAnchorGateForTest(uint64_t count, uint64_t mask) {
        anchor_staged_mask_.store(mask, std::memory_order_release);
        anchor_staged_count_.store(count, std::memory_order_release);
    }

    void CorruptStagingGuardForTest(uint32_t slot_id, bool before = false) {
        U1Guard &guard = before ? staging_slots_[slot_id].guard_before : staging_slots_[slot_id].guard_after;
        guard.words[0] ^= 1U;
    }

    bool TryExactReleaseForTest(uint32_t slot_id, uint32_t task_id, uint32_t generation) {
        return ReleaseOwnedSlot(slot_id, task_id, generation);
    }

    bool TaskPayloadAndTailValid(uint32_t task_id) const {
        const uint32_t payload_words = PayloadWordsForTask(task_id);
        for (uint32_t word = 0U; word < payload_words; ++word) {
            if (tasks_[task_id].payload[word] != ExpectedPayloadWord(nonce_, task_id, word)) {
                return false;
            }
        }
        for (uint32_t word = payload_words; word < kMaxPayloadWords; ++word) {
            if (tasks_[task_id].payload[word] != kPayloadPoisonWord) {
                return false;
            }
        }
        return true;
    }

    bool TaskPayloadIsPoison(uint32_t task_id) const {
        return std::all_of(tasks_[task_id].payload.begin(), tasks_[task_id].payload.end(), [](uint64_t word) {
            return word == kPayloadPoisonWord;
        });
    }

    bool ThreadReportIsPoison(const U1ThreadReport &report) const { return ObjectIsPoison(report); }

    bool StagingGuardsValid(uint32_t slot_id) const {
        return GuardValid(staging_slots_[slot_id].guard_before, UbufGuardBeforeId(slot_id)) &&
               GuardValid(staging_slots_[slot_id].guard_after, UbufGuardAfterId(slot_id));
    }

    bool GuardsValid() const {
        if (!GuardValid(guard_before_tasks_, kGuardBeforeTasks) || !GuardValid(guard_after_tasks_, kGuardAfterTasks) ||
            !GuardValid(guard_before_builder_threads_, kGuardBeforeBuilderThreads) ||
            !GuardValid(guard_after_builder_threads_, kGuardAfterBuilderThreads)) {
            return false;
        }
        for (uint32_t slot = 0U; slot < kUbufSlotCount; ++slot) {
            if (!StagingGuardsValid(slot)) {
                return false;
            }
        }
        return true;
    }

private:
    struct BuildAttempt {
        BuildOutcome outcome = BuildOutcome::InvalidLeader;
        uint32_t slot_attempts = 0U;
    };

    enum class WaitOutcome : uint32_t { Ready, Timeout, Aborted, Invalid };
    enum class AcquireOutcome : uint32_t { Acquired, Timeout, Aborted, Invalid };

    template <typename T>
    static void PoisonObject(T &object) {
        auto *bytes = reinterpret_cast<uint8_t *>(&object);
        std::fill(bytes, bytes + sizeof(T), kReportPoisonByte);
    }

    template <typename T>
    static bool ObjectIsPoison(const T &object) {
        const auto *bytes = reinterpret_cast<const uint8_t *>(&object);
        return std::all_of(bytes, bytes + sizeof(T), [](uint8_t value) {
            return value == kReportPoisonByte;
        });
    }

    static bool ValidWarpLeader(uint32_t thread_id) { return thread_id < kThreadCount && thread_id % kWarpSize == 0U; }

    static bool ValidTaskLeader(uint32_t task_id, uint32_t thread_id) {
        return task_id < kTaskCount && ValidWarpLeader(thread_id) && thread_id / kWarpSize == task_id % kWarpCount;
    }

    static void PollYield(uint32_t polls) {
        if ((polls & 0xFFU) == 0U) {
            std::this_thread::yield();
        }
    }

    static void InvokeHook(const BuildOptions &options, BuildStage stage, uint32_t task_id, uint32_t thread_id) {
        if (options.hook) {
            options.hook(stage, task_id, thread_id);
        }
    }

    void InitializeGuard(U1Guard &guard, uint32_t guard_id) const {
        for (uint32_t word = 0U; word < kWordsPerLine; ++word) {
            guard.words[word] = ExpectedGuardWord(nonce_, guard_id, word);
        }
    }

    bool GuardValid(const U1Guard &guard, uint32_t guard_id) const {
        for (uint32_t word = 0U; word < kWordsPerLine; ++word) {
            if (guard.words[word] != ExpectedGuardWord(nonce_, guard_id, word)) {
                return false;
            }
        }
        return true;
    }

    void InitializeSlotGuards(uint32_t slot_id) {
        InitializeGuard(staging_slots_[slot_id].guard_before, UbufGuardBeforeId(slot_id));
        InitializeGuard(staging_slots_[slot_id].guard_after, UbufGuardAfterId(slot_id));
    }

    void UpdateMaximum(std::atomic<uint64_t> &maximum, uint64_t value) {
        uint64_t observed = maximum.load(std::memory_order_relaxed);
        while (observed < value &&
               !maximum.compare_exchange_weak(observed, value, std::memory_order_relaxed, std::memory_order_relaxed)) {}
    }

    AcquireOutcome
    AcquireSlot(uint32_t slot_id, uint32_t task_id, uint32_t poll_budget, uint32_t &attempts, uint32_t &generation) {
        for (;;) {
            ++attempts;
            uint64_t observed = slot_states_[slot_id].load(std::memory_order_acquire);
            const DecodedSlotState decoded = DecodeSlotState(static_cast<int64_t>(observed));
            if (!SlotStateValidForSlot(observed, slot_id)) {
                return AcquireOutcome::Invalid;
            }
            if (!decoded.busy) {
                if (decoded.generation >= kSlotReuseCount) {
                    return AcquireOutcome::Invalid;
                }
                const uint64_t desired = SlotBusyState(decoded.generation, task_id);
                if (slot_states_[slot_id].compare_exchange_weak(
                        observed, desired, std::memory_order_acq_rel, std::memory_order_acquire
                    )) {
                    generation = decoded.generation;
                    const uint64_t slot_depth = slot_busy_depth_[slot_id].fetch_add(1U, std::memory_order_acq_rel) + 1U;
                    const uint64_t global_depth = global_busy_depth_.fetch_add(1U, std::memory_order_acq_rel) + 1U;
                    UpdateMaximum(slot_max_busy_depth_[slot_id], slot_depth);
                    UpdateMaximum(global_max_busy_depth_, global_depth);
                    slot_acquire_count_[slot_id].fetch_add(1U, std::memory_order_relaxed);
                    if (slot_depth == 1U && global_depth <= kUbufSlotCount) {
                        return AcquireOutcome::Acquired;
                    }
                    (void)ReleaseOwnedSlot(slot_id, task_id, generation);
                    return AcquireOutcome::Invalid;
                }
            }
            if (fatal_.load(std::memory_order_acquire) != 0U) {
                return AcquireOutcome::Aborted;
            }
            if (poll_budget != 0U && attempts >= poll_budget) {
                return AcquireOutcome::Timeout;
            }
            PollYield(attempts);
        }
    }

    bool ReleaseOwnedSlot(uint32_t slot_id, uint32_t task_id, uint32_t generation) {
        if (slot_id >= kUbufSlotCount || task_id >= kTaskCount || generation == UINT32_MAX) {
            return false;
        }
        const uint64_t owned = SlotBusyState(generation, task_id);
        if (slot_states_[slot_id].load(std::memory_order_acquire) != owned) {
            return false;
        }

        const uint64_t slot_depth_observed = slot_busy_depth_[slot_id].load(std::memory_order_acquire);
        const uint64_t global_depth_observed = global_busy_depth_.load(std::memory_order_acquire);
        if (slot_depth_observed == 0U || global_depth_observed == 0U) {
            return false;
        }
        const uint64_t slot_depth_before = slot_busy_depth_[slot_id].fetch_sub(1U, std::memory_order_acq_rel);
        const uint64_t global_depth_before = global_busy_depth_.fetch_sub(1U, std::memory_order_acq_rel);
        uint64_t expected = owned;
        const bool released = slot_states_[slot_id].compare_exchange_strong(
            expected, SlotFreeState(generation + 1U), std::memory_order_release, std::memory_order_acquire
        );
        if (!released) {
            slot_busy_depth_[slot_id].fetch_add(1U, std::memory_order_relaxed);
            global_busy_depth_.fetch_add(1U, std::memory_order_relaxed);
            return false;
        }
        slot_release_count_[slot_id].fetch_add(1U, std::memory_order_relaxed);
        return slot_depth_before == 1U && global_depth_before > 0U && global_depth_before <= kUbufSlotCount;
    }

    WaitOutcome WaitForAnchorGate(uint32_t poll_budget, uint32_t &polls) {
        for (;;) {
            const uint64_t count = anchor_staged_count_.load(std::memory_order_acquire);
            const uint64_t mask = anchor_staged_mask_.load(std::memory_order_acquire);
            if (count == kAnchorTaskCount && mask == kAnchorStagedMask) {
                return WaitOutcome::Ready;
            }
            if (count > kAnchorTaskCount || (mask & ~kAnchorStagedMask) != 0U) {
                return WaitOutcome::Invalid;
            }
            if (fatal_.load(std::memory_order_acquire) != 0U) {
                return WaitOutcome::Aborted;
            }
            ++polls;
            if (poll_budget != 0U && polls >= poll_budget) {
                return WaitOutcome::Timeout;
            }
            PollYield(polls);
        }
    }

    void ResetBuildingTask(uint32_t task_id) {
        uint64_t expected = BuildingState(task_id);
        (void)tasks_[task_id].control.state.compare_exchange_strong(
            expected, EmptyState(), std::memory_order_release, std::memory_order_acquire
        );
    }

    void ReportFatal(U1FatalReason reason, uint32_t owner, uint32_t task_id) {
        uint64_t expected = 0U;
        (void)fatal_.compare_exchange_strong(
            expected, EncodeU1Fatal(reason, owner, task_id), std::memory_order_acq_rel, std::memory_order_acquire
        );
    }

    BuildAttempt BuildTaskImpl(uint32_t task_id, uint32_t thread_id, const BuildOptions &options) {
        BuildAttempt result{};
        if (!ValidTaskLeader(task_id, thread_id)) {
            return result;
        }

        result.outcome = BuildOutcome::Aborted;
        const uint32_t slot_id = SlotForTask(task_id);
        uint32_t phase_bits = 0U;
        uint32_t ubuf_words_written = 0U;
        uint32_t gm_words_stored = 0U;
        uint32_t generation = kInvalidSlotId;
        bool claimed = false;
        bool owns_slot = false;
        bool published = false;

        uint64_t expected = EmptyState();
        if (!tasks_[task_id].control.state.compare_exchange_strong(
                expected, BuildingState(task_id), std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            result.outcome = BuildOutcome::ClaimLost;
            return result;
        }
        claimed = true;
        phase_bits |= kBuildClaimed;
        build_claim_count_.fetch_add(1U, std::memory_order_relaxed);

        try {
            InvokeHook(options, BuildStage::AfterClaim, task_id, thread_id);
            if (!IsAnchorTask(task_id)) {
                uint32_t gate_polls = 0U;
                const WaitOutcome gate = WaitForAnchorGate(options.anchor_poll_budget, gate_polls);
                if (gate != WaitOutcome::Ready) {
                    ResetBuildingTask(task_id);
                    if (gate == WaitOutcome::Timeout) {
                        result.outcome = BuildOutcome::AnchorTimeout;
                        ReportFatal(U1FatalReason::SlotTimeout, kBuilderOwner, task_id);
                    } else if (gate == WaitOutcome::Invalid) {
                        ReportFatal(U1FatalReason::SlotInvariant, kBuilderOwner, task_id);
                    }
                    WriteBuildReport(task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, generation);
                    return result;
                }
            }
            const AcquireOutcome acquire =
                AcquireSlot(slot_id, task_id, options.slot_poll_budget, result.slot_attempts, generation);
            if (acquire != AcquireOutcome::Acquired) {
                if (acquire == AcquireOutcome::Timeout) {
                    ResetBuildingTask(task_id);
                    result.outcome = BuildOutcome::SlotTimeout;
                    ReportFatal(U1FatalReason::SlotTimeout, kBuilderOwner, task_id);
                } else if (acquire == AcquireOutcome::Invalid) {
                    ResetBuildingTask(task_id);
                    result.outcome = BuildOutcome::ReleaseConflict;
                    ReportFatal(U1FatalReason::SlotInvariant, kBuilderOwner, task_id);
                } else {
                    ResetBuildingTask(task_id);
                }
                WriteBuildReport(task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, generation);
                return result;
            }
            owns_slot = true;
            phase_bits |= kBuildSlotAcquired;
            InitializeSlotGuards(slot_id);
            InvokeHook(options, BuildStage::AfterSlotAcquire, task_id, thread_id);

            const uint32_t payload_words = PayloadWordsForTask(task_id);
            for (uint32_t word = 0U; word < payload_words; ++word) {
                staging_slots_[slot_id].payload[word] = ExpectedPayloadWord(nonce_, task_id, word);
                ++ubuf_words_written;
                if (word + 1U == payload_words / 2U) {
                    InvokeHook(options, BuildStage::AfterHalfStaging, task_id, thread_id);
                }
            }
            phase_bits |= kBuildUbufComplete;
            InvokeHook(options, BuildStage::AfterStaging, task_id, thread_id);

            if (!StagingGuardsValid(slot_id)) {
                if (ReleaseOwnedSlot(slot_id, task_id, generation)) {
                    owns_slot = false;
                    phase_bits |= kBuildSlotReleased;
                }
                ResetBuildingTask(task_id);
                ReportFatal(U1FatalReason::UbufGuardCorruption, kBuilderOwner, task_id);
                WriteBuildReport(task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, generation);
                result.outcome = BuildOutcome::GuardCorruption;
                return result;
            }
            phase_bits |= kBuildUbufGuardsValid;
            ubuf_guard_check_count_.fetch_add(1U, std::memory_order_relaxed);

            if (IsAnchorTask(task_id)) {
                const uint64_t anchor_bit = uint64_t{1U} << task_id;
                const uint64_t previous_mask = anchor_staged_mask_.fetch_or(anchor_bit, std::memory_order_acq_rel);
                if ((previous_mask & anchor_bit) != 0U) {
                    if (ReleaseOwnedSlot(slot_id, task_id, generation)) {
                        owns_slot = false;
                        phase_bits |= kBuildSlotReleased;
                    }
                    ResetBuildingTask(task_id);
                    ReportFatal(U1FatalReason::SlotInvariant, kBuilderOwner, task_id);
                    WriteBuildReport(task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, generation);
                    result.outcome = BuildOutcome::Aborted;
                    return result;
                }
                const uint64_t arrived = anchor_staged_count_.fetch_add(1U, std::memory_order_acq_rel) + 1U;
                if (arrived > kAnchorTaskCount) {
                    result.outcome = BuildOutcome::Aborted;
                    if (ReleaseOwnedSlot(slot_id, task_id, generation)) {
                        owns_slot = false;
                        phase_bits |= kBuildSlotReleased;
                    }
                    ResetBuildingTask(task_id);
                    ReportFatal(U1FatalReason::SlotInvariant, kBuilderOwner, task_id);
                    WriteBuildReport(task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, generation);
                    return result;
                }
                InvokeHook(options, BuildStage::AfterAnchorArrive, task_id, thread_id);
                uint32_t anchor_polls = 0U;
                anchor_waiting_count_.fetch_add(1U, std::memory_order_release);
                const WaitOutcome gate = WaitForAnchorGate(options.anchor_poll_budget, anchor_polls);
                anchor_waiting_count_.fetch_sub(1U, std::memory_order_release);
                if (gate != WaitOutcome::Ready) {
                    if (ReleaseOwnedSlot(slot_id, task_id, generation)) {
                        owns_slot = false;
                        phase_bits |= kBuildSlotReleased;
                    }
                    ResetBuildingTask(task_id);
                    if (gate == WaitOutcome::Timeout) {
                        result.outcome = BuildOutcome::AnchorTimeout;
                        ReportFatal(U1FatalReason::SlotTimeout, kBuilderOwner, task_id);
                    } else if (gate == WaitOutcome::Invalid) {
                        ReportFatal(U1FatalReason::SlotInvariant, kBuilderOwner, task_id);
                    }
                    WriteBuildReport(task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, generation);
                    return result;
                }
            }
            InvokeHook(options, BuildStage::AfterAnchorGate, task_id, thread_id);

            for (uint32_t word = 0U; word < payload_words; ++word) {
                tasks_[task_id].payload[word] = staging_slots_[slot_id].payload[word];
                ++gm_words_stored;
                if (word + 1U == payload_words / 2U) {
                    InvokeHook(options, BuildStage::AfterHalfGmStore, task_id, thread_id);
                }
            }
            phase_bits |= kBuildGmStoreComplete;
            InvokeHook(options, BuildStage::BeforePublish, task_id, thread_id);

            if (fatal_.load(std::memory_order_acquire) != 0U) {
                if (ReleaseOwnedSlot(slot_id, task_id, generation)) {
                    owns_slot = false;
                    phase_bits |= kBuildSlotReleased;
                }
                ResetBuildingTask(task_id);
                WriteBuildReport(task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, generation);
                result.outcome = BuildOutcome::Aborted;
                return result;
            }

            expected = BuildingState(task_id);
            if (!tasks_[task_id].control.state.compare_exchange_strong(
                    expected, BuiltState(task_id), std::memory_order_release, std::memory_order_acquire
                )) {
                if (ReleaseOwnedSlot(slot_id, task_id, generation)) {
                    owns_slot = false;
                    phase_bits |= kBuildSlotReleased;
                }
                ResetBuildingTask(task_id);
                ReportFatal(U1FatalReason::TaskPublishConflict, kBuilderOwner, task_id);
                WriteBuildReport(task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, generation);
                result.outcome = BuildOutcome::PublishConflict;
                return result;
            }
            published = true;
            phase_bits |= kBuildPublished;
            built_count_.fetch_add(1U, std::memory_order_relaxed);
            InvokeHook(options, BuildStage::AfterPublishBeforeRelease, task_id, thread_id);

            if (!ReleaseOwnedSlot(slot_id, task_id, generation)) {
                owns_slot = false;
                ReportFatal(U1FatalReason::SlotInvariant, kBuilderOwner, task_id);
                WriteBuildReport(task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, generation);
                result.outcome = BuildOutcome::ReleaseConflict;
                return result;
            }
            owns_slot = false;
            phase_bits |= kBuildSlotReleased;
            InvokeHook(options, BuildStage::AfterRelease, task_id, thread_id);

            WriteBuildReport(
                task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, generation,
                ExpectedPayloadChecksum(nonce_, task_id)
            );
            result.outcome = BuildOutcome::Built;
            return result;
        } catch (...) {
            if (owns_slot && ReleaseOwnedSlot(slot_id, task_id, generation)) {
                owns_slot = false;
                phase_bits |= kBuildSlotReleased;
            }
            if (claimed && !published) {
                ResetBuildingTask(task_id);
            }
            ReportFatal(U1FatalReason::BuildAborted, kBuilderOwner, task_id);
            WriteBuildReport(
                task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, generation,
                published ? ExpectedPayloadChecksum(nonce_, task_id) : 0U
            );
            result.outcome = BuildOutcome::Aborted;
            return result;
        }
    }

    void WriteBuildReport(
        uint32_t task_id, uint32_t thread_id, uint32_t phase_bits, uint32_t ubuf_words_written,
        uint32_t gm_words_stored, uint32_t generation, uint64_t checksum = 0U
    ) {
        U1BuildReport report{};
        report.task_id = task_id;
        report.builder_thread = thread_id;
        report.builder_warp = thread_id / kWarpSize;
        report.builder_lane = thread_id % kWarpSize;
        report.phase_bits = phase_bits;
        report.payload_lines = PayloadLinesForTask(task_id);
        report.ubuf_words_written = ubuf_words_written;
        report.gm_words_stored = gm_words_stored;
        report.claim_count = (phase_bits & kBuildClaimed) != 0U ? 1U : 0U;
        report.publish_count = (phase_bits & kBuildPublished) != 0U ? 1U : 0U;
        report.slot_id = SlotForTask(task_id);
        report.slot_generation = generation;
        report.launch_nonce = nonce_;
        report.payload_checksum = checksum;
        tasks_[task_id].build_report = report;
    }

    void WriteExecReport(
        uint32_t task_id, uint32_t phase_bits, uint32_t payload_words, uint32_t checksum_match_count, uint64_t checksum
    ) {
        U1ExecReport report{};
        report.task_id = task_id;
        report.executor_owner = kExecutorOwner;
        report.executor_physical_block = 0U;
        report.executor_subblock_id = 1U;
        report.phase_bits = phase_bits;
        report.payload_lines = PayloadLinesForTask(task_id);
        report.payload_words_read = payload_words;
        report.claim_count = 1U;
        report.completion_count = (phase_bits & kExecCompleted) != 0U ? 1U : 0U;
        report.checksum_match_count = checksum_match_count;
        report.launch_nonce = nonce_;
        report.payload_checksum = checksum;
        tasks_[task_id].exec_report = report;
    }

    void WriteThreadReport(
        uint32_t thread_id, uint32_t first_task, uint32_t last_task, uint32_t status, uint32_t slot_attempts,
        uint32_t build_count
    ) {
        U1ThreadReport report{};
        report.thread_id = thread_id;
        report.warp_id = thread_id / kWarpSize;
        report.lane_id = thread_id % kWarpSize;
        report.active_leader = 1U;
        report.first_task_id = first_task;
        report.last_task_id = last_task;
        report.task_count = kTasksPerWarp;
        report.status = status;
        report.slot_attempt_count = slot_attempts;
        report.build_count = build_count;
        report.launch_nonce = nonce_;
        report.checksum = ExpectedThreadChecksum(nonce_, thread_id, first_task, last_task, status);
        builder_threads_[thread_id] = report;
    }

    uint64_t nonce_;
    std::array<CpuTask, kTaskCount> tasks_{};
    std::array<U1ThreadReport, kThreadCount> builder_threads_{};
    std::array<CpuStagingSlot, kUbufSlotCount> staging_slots_{};
    std::array<std::atomic<uint64_t>, kUbufSlotCount> slot_states_{};
    std::array<std::atomic<uint64_t>, kUbufSlotCount> slot_busy_depth_{};
    std::array<std::atomic<uint64_t>, kUbufSlotCount> slot_max_busy_depth_{};
    std::array<std::atomic<uint64_t>, kUbufSlotCount> slot_acquire_count_{};
    std::array<std::atomic<uint64_t>, kUbufSlotCount> slot_release_count_{};
    std::atomic<uint64_t> global_busy_depth_{0U};
    std::atomic<uint64_t> global_max_busy_depth_{0U};
    std::atomic<uint64_t> anchor_staged_count_{0U};
    std::atomic<uint64_t> anchor_staged_mask_{0U};
    std::atomic<uint64_t> anchor_waiting_count_{0U};
    std::atomic<uint64_t> ubuf_guard_check_count_{0U};
    std::atomic<uint64_t> build_claim_count_{0U};
    std::atomic<uint64_t> built_count_{0U};
    std::atomic<uint64_t> exec_claim_count_{0U};
    std::atomic<uint64_t> done_count_{0U};
    std::atomic<uint64_t> builder_finished_count_{0U};
    std::atomic<uint64_t> executor_finished_count_{0U};
    std::atomic<uint64_t> fatal_{0U};
    U1Guard guard_before_tasks_{};
    U1Guard guard_after_tasks_{};
    U1Guard guard_before_builder_threads_{};
    U1Guard guard_after_builder_threads_{};
};

}  // namespace pa_scheduler::simt_cross_core::u1::cpu

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_U1_MULTI_SLOT_CPU_MODEL_H
