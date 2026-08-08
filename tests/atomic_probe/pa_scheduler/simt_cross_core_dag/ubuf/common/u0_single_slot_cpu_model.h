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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_U0_SINGLE_SLOT_CPU_MODEL_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_U0_SINGLE_SLOT_CPU_MODEL_H

#include "u0_single_slot.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <thread>

namespace pa_scheduler::simt_cross_core::u0::cpu {

enum class BuildStage : uint32_t {
    AfterClaim = 0U,
    AfterSlotAcquire = 1U,
    AfterHalfStaging = 2U,
    AfterStaging = 3U,
    AfterHalfGmStore = 4U,
    BeforePublish = 5U,
    AfterPublishBeforeRelease = 6U,
    AfterRelease = 7U,
};

enum class BuildOutcome : uint32_t {
    Built = 0U,
    InvalidLeader = 1U,
    ClaimLost = 2U,
    SlotTimeout = 3U,
    Aborted = 4U,
    PublishConflict = 5U,
    ReleaseConflict = 6U,
};

enum class ExecOutcome : uint32_t {
    Done = 0U,
    InvalidLeader = 1U,
    WaitTimeout = 2U,
    ClaimLost = 3U,
    PayloadMismatch = 4U,
    CompletionConflict = 5U,
    InvalidState = 6U,
};

using BuildHook = std::function<void(BuildStage, uint32_t, uint32_t)>;

struct BuildOptions {
    uint32_t slot_poll_budget = 0U;
    BuildHook hook{};
};

struct ExecOptions {
    uint32_t wait_poll_budget = 0U;
};

struct alignas(kCacheLineBytes) CpuTaskControl {
    std::atomic<uint64_t> state{EmptyState()};
    std::array<uint8_t, kCacheLineBytes - sizeof(std::atomic<uint64_t>)> padding{};
};

struct alignas(kCacheLineBytes) CpuTask {
    CpuTaskControl control;
    alignas(kCacheLineBytes) std::array<uint64_t, kMaxPayloadWords> payload{};
    U0BuildReport build_report{};
    U0ExecReport exec_report{};
};

static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t), "U0 CPU model requires lock-sized uint64 atomic");
static_assert(offsetof(CpuTask, payload) == kCacheLineBytes, "U0 CPU payload must not share its atomic control line");

class SingleSlotModel {
public:
    explicit SingleSlotModel(uint64_t nonce) :
        nonce_(nonce) {
        for (CpuTask &task : tasks_) {
            task.control.state.store(EmptyState(), std::memory_order_relaxed);
            task.payload.fill(kPayloadPoisonWord);
            PoisonObject(task.build_report);
            PoisonObject(task.exec_report);
        }
        for (U0ThreadReport &report : builder_threads_) {
            PoisonObject(report);
        }
        InitializeGuard(guard_before_tasks_, kGuardBeforeTasks);
        InitializeGuard(guard_after_tasks_, kGuardAfterTasks);
        InitializeGuard(guard_before_builder_threads_, kGuardBeforeBuilderThreads);
        InitializeGuard(guard_after_builder_threads_, kGuardAfterBuilderThreads);
        InitializeGuard(staging_guard_before_, kGuardBeforeStagingSlot);
        InitializeGuard(staging_guard_after_, kGuardAfterStagingSlot);
        staging_payload_.fill(kPayloadPoisonWord);
    }

    SingleSlotModel(const SingleSlotModel &) = delete;
    SingleSlotModel &operator=(const SingleSlotModel &) = delete;

    BuildOutcome BuildLeader(uint32_t task_id, uint32_t thread_id, const BuildOptions &options = {}) {
        if (!ValidLeader(task_id, thread_id)) {
            return BuildOutcome::InvalidLeader;
        }

        const uint32_t warp_id = thread_id / kWarpSize;
        uint32_t status = kThreadActiveLeader;
        uint32_t slot_attempts = 0U;
        uint32_t phase_bits = 0U;
        uint32_t ubuf_words_written = 0U;
        uint32_t gm_words_stored = 0U;
        uint32_t slot_ticket = 0U;
        bool claimed = false;
        bool owns_slot = false;
        bool published = false;
        bool released = false;

        uint64_t expected = EmptyState();
        if (!tasks_[task_id].control.state.compare_exchange_strong(
                expected, BuildingState(task_id), std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            WriteThreadReport(builder_threads_[thread_id], thread_id, task_id, status, 1U, 0U);
            return BuildOutcome::ClaimLost;
        }
        claimed = true;
        status |= kThreadTaskClaimed;
        phase_bits |= kBuildClaimed;
        build_claim_count_.fetch_add(1U, std::memory_order_relaxed);

        try {
            InvokeHook(options, BuildStage::AfterClaim, task_id, thread_id);
            if (!AcquireSlot(warp_id, options.slot_poll_budget, slot_attempts, slot_ticket)) {
                ReportFatal(U0FatalReason::SlotTimeout, kBuilderOwner, task_id);
                ResetBuildingTask(task_id);
                WriteBuildReport(
                    task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, 0U, 1U, 0U, slot_ticket
                );
                WriteThreadReport(builder_threads_[thread_id], thread_id, task_id, status, 1U, slot_attempts);
                return BuildOutcome::SlotTimeout;
            }
            owns_slot = true;
            status |= kThreadSlotAcquired;
            phase_bits |= kBuildSlotAcquired;
            InvokeHook(options, BuildStage::AfterSlotAcquire, task_id, thread_id);

            const uint32_t payload_words = PayloadWordsForTask(task_id);
            for (uint32_t word = 0U; word < payload_words; ++word) {
                staging_payload_[word] = ExpectedPayloadWord(nonce_, task_id, word);
                ++ubuf_words_written;
                if (word + 1U == payload_words / 2U) {
                    InvokeHook(options, BuildStage::AfterHalfStaging, task_id, thread_id);
                }
            }
            phase_bits |= kBuildUbufComplete;
            InvokeHook(options, BuildStage::AfterStaging, task_id, thread_id);

            if (!StagingGuardsValid()) {
                released = ReleaseOwnedSlot(warp_id);
                owns_slot = false;
                ReportFatal(U0FatalReason::UbufGuardCorruption, kBuilderOwner, task_id);
                WriteBuildReport(
                    task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, 0U, 1U, released ? 1U : 0U,
                    slot_ticket
                );
                WriteThreadReport(builder_threads_[thread_id], thread_id, task_id, status, 1U, slot_attempts);
                return BuildOutcome::Aborted;
            }
            ubuf_guard_check_count_.fetch_add(1U, std::memory_order_relaxed);
            status |= kThreadUbufGuardsValid;
            phase_bits |= kBuildUbufGuardsValid;

            for (uint32_t word = 0U; word < payload_words; ++word) {
                tasks_[task_id].payload[word] = staging_payload_[word];
                ++gm_words_stored;
                if (word + 1U == payload_words / 2U) {
                    InvokeHook(options, BuildStage::AfterHalfGmStore, task_id, thread_id);
                }
            }
            status |= kThreadPayloadComplete;
            phase_bits |= kBuildGmStoreComplete;
            InvokeHook(options, BuildStage::BeforePublish, task_id, thread_id);

            expected = BuildingState(task_id);
            if (!tasks_[task_id].control.state.compare_exchange_strong(
                    expected, BuiltState(task_id), std::memory_order_release, std::memory_order_acquire
                )) {
                ReportFatal(U0FatalReason::TaskPublishConflict, kBuilderOwner, task_id);
                released = ReleaseOwnedSlot(warp_id);
                owns_slot = false;
                WriteBuildReport(
                    task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, 0U, 1U, released ? 1U : 0U,
                    slot_ticket
                );
                WriteThreadReport(builder_threads_[thread_id], thread_id, task_id, status, 1U, slot_attempts);
                return BuildOutcome::PublishConflict;
            }
            published = true;
            status |= kThreadTaskPublished;
            phase_bits |= kBuildPublished;
            built_count_.fetch_add(1U, std::memory_order_relaxed);
            InvokeHook(options, BuildStage::AfterPublishBeforeRelease, task_id, thread_id);

            if (!ReleaseOwnedSlot(warp_id)) {
                owns_slot = false;
                ReportFatal(U0FatalReason::SlotInvariant, kBuilderOwner, task_id);
                WriteBuildReport(
                    task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, 1U, 1U, 0U, slot_ticket
                );
                WriteThreadReport(builder_threads_[thread_id], thread_id, task_id, status, 1U, slot_attempts);
                return BuildOutcome::ReleaseConflict;
            }
            owns_slot = false;
            released = true;
            status |= kThreadSlotReleased;
            phase_bits |= kBuildSlotReleased;
            InvokeHook(options, BuildStage::AfterRelease, task_id, thread_id);

            const uint64_t checksum = ExpectedPayloadChecksum(nonce_, task_id);
            WriteBuildReport(
                task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, 1U, 1U, 1U, slot_ticket, checksum
            );
            WriteThreadReport(builder_threads_[thread_id], thread_id, task_id, status, 1U, slot_attempts);
            builder_finished_count_.fetch_add(1U, std::memory_order_release);
            return BuildOutcome::Built;
        } catch (...) {
            if (owns_slot) {
                released = ReleaseOwnedSlot(warp_id);
                owns_slot = false;
            }
            if (claimed && !published) {
                ResetBuildingTask(task_id);
            }
            ReportFatal(U0FatalReason::BuildAborted, kBuilderOwner, task_id);
            WriteBuildReport(
                task_id, thread_id, phase_bits, ubuf_words_written, gm_words_stored, published ? 1U : 0U, 1U,
                released ? 1U : 0U, slot_ticket
            );
            WriteThreadReport(builder_threads_[thread_id], thread_id, task_id, status, 1U, slot_attempts);
            return BuildOutcome::Aborted;
        }
    }

    ExecOutcome ExecuteTask(uint32_t task_id, const ExecOptions &options = {}) {
        if (task_id >= kTaskCount) {
            return ExecOutcome::InvalidLeader;
        }

        uint32_t polls = 0U;
        uint64_t observed = 0U;
        for (;;) {
            observed = tasks_[task_id].control.state.load(std::memory_order_acquire);
            if (observed == BuiltState(task_id)) {
                break;
            }
            if (observed != EmptyState() && observed != BuildingState(task_id)) {
                ReportFatal(U0FatalReason::InvalidTaskState, kExecutorOwner, task_id);
                return ExecOutcome::InvalidState;
            }
            ++polls;
            if (options.wait_poll_budget != 0U && polls >= options.wait_poll_budget) {
                ReportFatal(U0FatalReason::TaskWaitTimeout, kExecutorOwner, task_id);
                return ExecOutcome::WaitTimeout;
            }
            if ((polls & 0xFFU) == 0U) {
                std::this_thread::yield();
            }
        }

        uint64_t expected = BuiltState(task_id);
        if (!tasks_[task_id].control.state.compare_exchange_strong(
                expected, ClaimedState(task_id), std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            ReportFatal(U0FatalReason::TaskClaimConflict, kExecutorOwner, task_id);
            return ExecOutcome::ClaimLost;
        }
        exec_claim_count_.fetch_add(1U, std::memory_order_relaxed);

        uint64_t checksum = 0x243F6A8885A308D3ULL ^ static_cast<uint64_t>(task_id);
        const uint32_t payload_words = PayloadWordsForTask(task_id);
        for (uint32_t word = 0U; word < payload_words; ++word) {
            checksum = FoldChecksum(checksum, tasks_[task_id].payload[word], word);
        }
        const bool checksum_matches = checksum == ExpectedPayloadChecksum(nonce_, task_id);
        const uint32_t phase_bits = kExecClaimed | kExecPayloadRead | (checksum_matches ? kExecPayloadValid : 0U);
        if (!checksum_matches) {
            ReportFatal(U0FatalReason::PayloadMismatch, kExecutorOwner, task_id);
            WriteExecReport(task_id, phase_bits, payload_words, 1U, 0U, 0U, checksum);
            return ExecOutcome::PayloadMismatch;
        }

        WriteExecReport(task_id, phase_bits | kExecCompleted, payload_words, 1U, 1U, 1U, checksum);
        expected = ClaimedState(task_id);
        if (!tasks_[task_id].control.state.compare_exchange_strong(
                expected, DoneState(task_id), std::memory_order_release, std::memory_order_acquire
            )) {
            ReportFatal(U0FatalReason::TaskCompleteConflict, kExecutorOwner, task_id);
            return ExecOutcome::CompletionConflict;
        }
        const uint64_t completed = done_count_.fetch_add(1U, std::memory_order_acq_rel) + 1U;
        if (completed == kTaskCount) {
            uint64_t expected_finished = 0U;
            if (!executor_finished_count_.compare_exchange_strong(
                    expected_finished, 1U, std::memory_order_release, std::memory_order_acquire
                )) {
                ReportFatal(U0FatalReason::TaskCompleteConflict, kExecutorOwner, task_id);
                return ExecOutcome::CompletionConflict;
            }
        }
        return ExecOutcome::Done;
    }

    bool TryReadPublishedPayload(uint32_t task_id, uint64_t &checksum, uint32_t &words_read) const {
        words_read = 0U;
        checksum = 0U;
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
    const U0BuildReport &build_report(uint32_t task_id) const { return tasks_[task_id].build_report; }
    const U0ExecReport &exec_report(uint32_t task_id) const { return tasks_[task_id].exec_report; }
    const U0ThreadReport &builder_thread_report(uint32_t thread_id) const { return builder_threads_[thread_id]; }

    uint64_t slot_owner() const { return slot_owner_.load(std::memory_order_acquire); }
    uint64_t slot_busy_depth() const { return slot_busy_depth_.load(std::memory_order_acquire); }
    uint64_t slot_max_busy_depth() const { return slot_max_busy_depth_.load(std::memory_order_acquire); }
    uint64_t slot_acquire_count() const { return slot_acquire_count_.load(std::memory_order_acquire); }
    uint64_t slot_release_count() const { return slot_release_count_.load(std::memory_order_acquire); }
    uint64_t build_claim_count() const { return build_claim_count_.load(std::memory_order_acquire); }
    uint64_t built_count() const { return built_count_.load(std::memory_order_acquire); }
    uint64_t exec_claim_count() const { return exec_claim_count_.load(std::memory_order_acquire); }
    uint64_t done_count() const { return done_count_.load(std::memory_order_acquire); }
    uint64_t mte3_count() const { return mte3_count_.load(std::memory_order_acquire); }
    uint64_t ubuf_guard_check_count() const { return ubuf_guard_check_count_.load(std::memory_order_acquire); }
    uint64_t builder_finished_count() const { return builder_finished_count_.load(std::memory_order_acquire); }
    uint64_t executor_finished_count() const { return executor_finished_count_.load(std::memory_order_acquire); }
    uint64_t fatal() const { return fatal_.load(std::memory_order_acquire); }

    void SetTaskStateForTest(uint32_t task_id, uint64_t state) {
        tasks_[task_id].control.state.store(state, std::memory_order_release);
    }

    void CorruptStagingGuardForTest() { staging_guard_after_.words[0] ^= 1U; }

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

    bool ThreadReportIsPoison(const U0ThreadReport &report) const {
        const auto *bytes = reinterpret_cast<const uint8_t *>(&report);
        return std::all_of(bytes, bytes + sizeof(report), [](uint8_t value) {
            return value == kReportPoisonByte;
        });
    }

    bool GuardsValid() const {
        return GuardValid(guard_before_tasks_, kGuardBeforeTasks) && GuardValid(guard_after_tasks_, kGuardAfterTasks) &&
               GuardValid(guard_before_builder_threads_, kGuardBeforeBuilderThreads) &&
               GuardValid(guard_after_builder_threads_, kGuardAfterBuilderThreads) &&
               GuardValid(staging_guard_before_, kGuardBeforeStagingSlot) &&
               GuardValid(staging_guard_after_, kGuardAfterStagingSlot);
    }

private:
    template <typename T>
    static void PoisonObject(T &object) {
        auto *bytes = reinterpret_cast<uint8_t *>(&object);
        std::fill(bytes, bytes + sizeof(T), kReportPoisonByte);
    }

    void InitializeGuard(U0Guard &guard, uint32_t guard_id) const {
        for (uint32_t word = 0U; word < kWordsPerLine; ++word) {
            guard.words[word] = ExpectedGuardWord(nonce_, guard_id, word);
        }
    }

    bool GuardValid(const U0Guard &guard, uint32_t guard_id) const {
        for (uint32_t word = 0U; word < kWordsPerLine; ++word) {
            if (guard.words[word] != ExpectedGuardWord(nonce_, guard_id, word)) {
                return false;
            }
        }
        return true;
    }

    bool StagingGuardsValid() const {
        return GuardValid(staging_guard_before_, kGuardBeforeStagingSlot) &&
               GuardValid(staging_guard_after_, kGuardAfterStagingSlot);
    }

    static bool ValidLeader(uint32_t task_id, uint32_t thread_id) {
        return task_id < kTaskCount && thread_id < kThreadCount && thread_id % kWarpSize == 0U &&
               thread_id / kWarpSize == task_id;
    }

    static void InvokeHook(const BuildOptions &options, BuildStage stage, uint32_t task_id, uint32_t thread_id) {
        if (options.hook) {
            options.hook(stage, task_id, thread_id);
        }
    }

    bool AcquireSlot(uint32_t warp_id, uint32_t poll_budget, uint32_t &attempts, uint32_t &slot_ticket) {
        const uint64_t owner = SlotOwnerValue(warp_id);
        for (;;) {
            ++attempts;
            uint64_t expected = kSlotFree;
            if (slot_owner_.compare_exchange_weak(
                    expected, owner, std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                const uint64_t depth = slot_busy_depth_.fetch_add(1U, std::memory_order_acq_rel) + 1U;
                UpdateMaxBusyDepth(depth);
                slot_ticket = static_cast<uint32_t>(slot_acquire_count_.fetch_add(1U, std::memory_order_relaxed) + 1U);
                return depth == 1U;
            }
            if (poll_budget != 0U && attempts >= poll_budget) {
                return false;
            }
            if ((attempts & 0xFFU) == 0U) {
                std::this_thread::yield();
            }
        }
    }

    bool ReleaseOwnedSlot(uint32_t warp_id) {
        const uint64_t depth_before = slot_busy_depth_.fetch_sub(1U, std::memory_order_acq_rel);
        uint64_t expected = SlotOwnerValue(warp_id);
        const bool owner_released = slot_owner_.compare_exchange_strong(
            expected, kSlotFree, std::memory_order_release, std::memory_order_acquire
        );
        slot_release_count_.fetch_add(1U, std::memory_order_relaxed);
        return depth_before == 1U && owner_released;
    }

    void UpdateMaxBusyDepth(uint64_t depth) {
        uint64_t observed = slot_max_busy_depth_.load(std::memory_order_relaxed);
        while (observed < depth && !slot_max_busy_depth_.compare_exchange_weak(
                                       observed, depth, std::memory_order_relaxed, std::memory_order_relaxed
                                   )) {}
    }

    void ResetBuildingTask(uint32_t task_id) {
        uint64_t expected = BuildingState(task_id);
        (void)tasks_[task_id].control.state.compare_exchange_strong(
            expected, EmptyState(), std::memory_order_release, std::memory_order_acquire
        );
    }

    void ReportFatal(U0FatalReason reason, uint32_t owner, uint32_t task_id) {
        uint64_t expected = 0U;
        (void)fatal_.compare_exchange_strong(
            expected, EncodeU0Fatal(reason, owner, task_id), std::memory_order_acq_rel, std::memory_order_acquire
        );
    }

    void WriteBuildReport(
        uint32_t task_id, uint32_t thread_id, uint32_t phase_bits, uint32_t ubuf_words_written,
        uint32_t gm_words_stored, uint32_t publish_count, uint32_t claim_count, uint32_t release_count,
        uint32_t slot_ticket, uint64_t checksum = 0U
    ) {
        U0BuildReport report{};
        report.task_id = task_id;
        report.builder_thread = thread_id;
        report.builder_warp = thread_id / kWarpSize;
        report.builder_lane = thread_id % kWarpSize;
        report.phase_bits = phase_bits;
        report.payload_lines = PayloadLinesForTask(task_id);
        report.ubuf_words_written = ubuf_words_written;
        report.gm_words_stored = gm_words_stored;
        report.claim_count = claim_count;
        report.publish_count = publish_count;
        report.release_count = release_count;
        report.slot_ticket = slot_ticket;
        report.launch_nonce = nonce_;
        report.payload_checksum = checksum;
        tasks_[task_id].build_report = report;
    }

    void WriteExecReport(
        uint32_t task_id, uint32_t phase_bits, uint32_t payload_words_read, uint32_t claim_count,
        uint32_t completion_count, uint32_t checksum_match_count, uint64_t checksum
    ) {
        U0ExecReport report{};
        report.task_id = task_id;
        report.executor_owner = kExecutorOwner;
        report.executor_physical_block = 0U;
        report.executor_subblock_id = 1U;
        report.phase_bits = phase_bits;
        report.payload_lines = PayloadLinesForTask(task_id);
        report.payload_words_read = payload_words_read;
        report.claim_count = claim_count;
        report.completion_count = completion_count;
        report.checksum_match_count = checksum_match_count;
        report.launch_nonce = nonce_;
        report.payload_checksum = checksum;
        tasks_[task_id].exec_report = report;
    }

    void WriteThreadReport(
        U0ThreadReport &destination, uint32_t thread_id, uint32_t task_id, uint32_t status, uint32_t task_attempt_count,
        uint32_t slot_attempt_count
    ) {
        U0ThreadReport report{};
        report.thread_id = thread_id;
        report.warp_id = thread_id / kWarpSize;
        report.lane_id = thread_id % kWarpSize;
        report.active_leader = 1U;
        report.task_id = task_id;
        report.status = status;
        report.task_attempt_count = task_attempt_count;
        report.slot_attempt_count = slot_attempt_count;
        report.launch_nonce = nonce_;
        report.checksum = ExpectedThreadChecksum(nonce_, thread_id, task_id, status);
        destination = report;
    }

    uint64_t nonce_;
    std::array<CpuTask, kTaskCount> tasks_{};
    std::array<U0ThreadReport, kThreadCount> builder_threads_{};
    std::atomic<uint64_t> slot_owner_{kSlotFree};
    std::atomic<uint64_t> slot_busy_depth_{0U};
    std::atomic<uint64_t> slot_max_busy_depth_{0U};
    std::atomic<uint64_t> slot_acquire_count_{0U};
    std::atomic<uint64_t> slot_release_count_{0U};
    std::atomic<uint64_t> build_claim_count_{0U};
    std::atomic<uint64_t> built_count_{0U};
    std::atomic<uint64_t> exec_claim_count_{0U};
    std::atomic<uint64_t> done_count_{0U};
    std::atomic<uint64_t> mte3_count_{0U};
    std::atomic<uint64_t> ubuf_guard_check_count_{0U};
    std::atomic<uint64_t> builder_finished_count_{0U};
    std::atomic<uint64_t> executor_finished_count_{0U};
    std::atomic<uint64_t> fatal_{0U};
    U0Guard guard_before_tasks_{};
    U0Guard guard_after_tasks_{};
    U0Guard guard_before_builder_threads_{};
    U0Guard guard_after_builder_threads_{};
    U0Guard staging_guard_before_{};
    std::array<uint64_t, kMaxPayloadWords> staging_payload_{};
    U0Guard staging_guard_after_{};
};

}  // namespace pa_scheduler::simt_cross_core::u0::cpu

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_U0_SINGLE_SLOT_CPU_MODEL_H
