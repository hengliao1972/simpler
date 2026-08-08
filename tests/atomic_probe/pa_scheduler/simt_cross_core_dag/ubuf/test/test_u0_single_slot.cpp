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

#include "../common/u0_single_slot_cpu_model.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using namespace pa_scheduler::simt_cross_core;
using namespace pa_scheduler::simt_cross_core::u0;
using namespace pa_scheduler::simt_cross_core::u0::cpu;

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] U0 CPU single-slot: %s\n", message);
    ++g_failures;
}

class PausePoint {
public:
    void Stop() {
        std::unique_lock<std::mutex> lock(mutex_);
        reached_ = true;
        condition_.notify_all();
        condition_.wait(lock, [this] {
            return released_;
        });
    }

    bool WaitUntilReached() {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, std::chrono::seconds(10), [this] {
            return reached_;
        });
    }

    void Release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool reached_ = false;
    bool released_ = false;
};

class StartGate {
public:
    explicit StartGate(uint32_t participants) :
        participants_(participants) {}

    void ArriveAndWait() {
        std::unique_lock<std::mutex> lock(mutex_);
        ++arrived_;
        condition_.notify_all();
        condition_.wait(lock, [this] {
            return open_;
        });
    }

    bool WaitAndOpen() {
        std::unique_lock<std::mutex> lock(mutex_);
        const bool all_arrived = condition_.wait_for(lock, std::chrono::seconds(10), [this] {
            return arrived_ == participants_;
        });
        open_ = true;
        condition_.notify_all();
        return all_arrived;
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    uint32_t participants_;
    uint32_t arrived_ = 0U;
    bool open_ = false;
};

template <typename T>
bool ObjectIsPoison(const T &object) {
    const auto *bytes = reinterpret_cast<const uint8_t *>(&object);
    for (size_t index = 0U; index < sizeof(T); ++index) {
        if (bytes[index] != kReportPoisonByte) {
            return false;
        }
    }
    return true;
}

void TestAbiAndTransportContract() {
    Check(kWarpSize == 32U && kWarpCount == 64U, "logical topology is not 64 warps of 32 threads");
    Check(kThreadCount == 2048U && kTaskCount == 64U, "logical thread/task counts changed");
    Check(
        PayloadLinesForTask(0U) == 1U && PayloadLinesForTask(1U) == 10U && PayloadLinesForTask(2U) == 16U &&
            PayloadLinesForTask(3U) == 68U,
        "payload line classes are not 1/10/16/68"
    );
    Check(
        static_cast<uint32_t>(TransportKind::SimtUbufReadToGmWordStore) == 1U &&
            kTransportMarker == 0x55425546474D5354ULL,
        "direct GM-store transport marker changed"
    );
    Check(
        kUbufAlignmentBytes == 64U && kUbufPayloadOffsetBytes == 64U && kUbufRegionBytes == 4480U &&
            kUbufSlotCount == 1U,
        "guarded single-slot UBUF ABI changed"
    );

    const uint64_t raw_fatal = EncodeU0Fatal(U0FatalReason::InvalidTaskState, kExecutorOwner, 63U);
    const DecodedU0Fatal decoded_fatal = DecodeU0Fatal(static_cast<int64_t>(raw_fatal));
    Check(
        decoded_fatal.valid && decoded_fatal.reason == U0FatalReason::InvalidTaskState &&
            decoded_fatal.reporter_owner == kExecutorOwner && decoded_fatal.task_id == 63U,
        "shared U0 fatal ABI does not round-trip"
    );

    const g0::DecodedExecState building = DecodeExecState(BuildingState(63U));
    const g0::DecodedExecState built = DecodeExecState(BuiltState(63U));
    const g0::DecodedExecState claimed = DecodeExecState(ClaimedState(63U));
    const g0::DecodedExecState done = DecodeExecState(DoneState(63U));
    Check(
        building.valid && building.phase == g0::ExecPhase::Building && building.build_owner == kBuilderOwner &&
            building.execute_owner == g0::kUnboundOwner,
        "BUILDING does not encode physical AIV0 owner32"
    );
    Check(
        built.valid && built.phase == g0::ExecPhase::Built && built.build_owner == kBuilderOwner &&
            built.payload_lines == 68U,
        "BUILT does not encode physical AIV0 owner32 and 68 lines"
    );
    Check(
        claimed.valid && claimed.phase == g0::ExecPhase::Claimed && claimed.build_owner == kBuilderOwner &&
            claimed.execute_owner == kExecutorOwner,
        "CLAIMED does not encode the unique AIV1 executor owner33"
    );
    Check(
        done.valid && done.phase == g0::ExecPhase::Done && done.build_owner == kBuilderOwner &&
            done.execute_owner == kExecutorOwner,
        "DONE physical owners changed"
    );

    SingleSlotModel model(0xA500000000000001ULL);
    Check(model.mte3_count() == 0U, "direct GM-store CPU model reported an MTE3 operation");
    Check(model.GuardsValid(), "fresh model guards are invalid");
}

void TestInactiveLaneAndMismatchedWarpRejected() {
    SingleSlotModel model(0xA500000000000002ULL);
    const BuildOutcome non_leader = model.BuildLeader(0U, 1U);
    const BuildOutcome wrong_warp = model.BuildLeader(1U, 0U);

    Check(non_leader == BuildOutcome::InvalidLeader, "lane1 was accepted as a builder");
    Check(wrong_warp == BuildOutcome::InvalidLeader, "warp0 was accepted for task1");
    Check(
        model.task_state(0U) == EmptyState() && model.task_state(1U) == EmptyState(), "invalid actor changed task state"
    );
    Check(model.TaskPayloadIsPoison(0U) && model.TaskPayloadIsPoison(1U), "invalid actor touched GM payload");
    Check(model.ThreadReportIsPoison(model.builder_thread_report(0U)), "rejected warp leader overwrote its report");
    Check(model.ThreadReportIsPoison(model.builder_thread_report(1U)), "inactive lane report lost poison");
    Check(model.build_claim_count() == 0U && model.slot_acquire_count() == 0U, "invalid actor changed counters");
    Check(model.mte3_count() == 0U && model.GuardsValid(), "invalid actor changed transport count or guards");
}

void TestHalfWriteHiddenUntilPublication() {
    constexpr uint32_t kTask = 3U;
    constexpr uint32_t kThread = kTask * kWarpSize;
    const uint64_t nonce = 0xA500000000000003ULL;
    SingleSlotModel model(nonce);
    PausePoint pause;
    BuildOutcome outcome = BuildOutcome::Aborted;
    BuildOptions options{};
    options.hook = [&](BuildStage stage, uint32_t task_id, uint32_t thread_id) {
        if (stage == BuildStage::AfterHalfGmStore && task_id == kTask && thread_id == kThread) {
            pause.Stop();
        }
    };

    std::thread builder([&] {
        outcome = model.BuildLeader(kTask, kThread, options);
    });
    const bool reached = pause.WaitUntilReached();
    if (!reached) {
        pause.Release();
        builder.join();
        Check(false, "builder did not reach the half-GM-store boundary");
        return;
    }

    Check(model.task_state(kTask) == BuildingState(kTask), "half-written task was not kept BUILDING");
    Check(
        model.slot_owner() == SlotOwnerValue(kTask) && model.slot_busy_depth() == 1U, "half-write lost slot ownership"
    );
    const uint32_t payload_words = PayloadWordsForTask(kTask);
    for (uint32_t word = 0U; word < payload_words / 2U; ++word) {
        Check(
            model.payload_word(kTask, word) == ExpectedPayloadWord(nonce, kTask, word), "first half GM word is wrong"
        );
    }
    for (uint32_t word = payload_words / 2U; word < kMaxPayloadWords; ++word) {
        Check(model.payload_word(kTask, word) == kPayloadPoisonWord, "unwritten GM word lost poison");
    }

    uint64_t premature_checksum = 0U;
    uint32_t premature_words = UINT32_MAX;
    Check(
        !model.TryReadPublishedPayload(kTask, premature_checksum, premature_words) && premature_words == 0U,
        "reader consumed a half-written BUILDING payload"
    );

    pause.Release();
    builder.join();
    Check(outcome == BuildOutcome::Built, "paused builder did not finish after release");
    Check(model.task_state(kTask) == BuiltState(kTask), "completed task was not published BUILT");
    uint64_t checksum = 0U;
    uint32_t words_read = 0U;
    Check(model.TryReadPublishedPayload(kTask, checksum, words_read), "published payload was not readable");
    Check(words_read == payload_words, "published reader consumed the wrong word count");
    Check(checksum == ExpectedPayloadChecksum(nonce, kTask), "published checksum is wrong");
    Check(model.TaskPayloadAndTailValid(kTask), "published payload or poison tail is wrong");

    const U0BuildReport &report = model.build_report(kTask);
    Check(report.phase_bits == kExpectedBuildPhaseBits, "half-write build did not complete all phase evidence");
    Check(
        report.ubuf_words_written == payload_words && report.gm_words_stored == payload_words, "word counts are wrong"
    );
    Check(
        report.claim_count == 1U && report.publish_count == 1U && report.release_count == 1U,
        "lifecycle counts are wrong"
    );
    Check(model.slot_owner() == kSlotFree && model.slot_busy_depth() == 0U, "slot was not released after publication");
    Check(model.slot_max_busy_depth() == 1U && model.mte3_count() == 0U, "single-slot depth or MTE3 count changed");
    Check(model.GuardsValid(), "half-write path damaged a guard");
}

void TestHalfUbufStagingDoesNotTouchGm() {
    constexpr uint32_t kTask = 3U;
    constexpr uint32_t kThread = kTask * kWarpSize;
    SingleSlotModel model(0xA500000000000033ULL);
    PausePoint pause;
    BuildOutcome outcome = BuildOutcome::Aborted;
    BuildOptions options{};
    options.hook = [&](BuildStage stage, uint32_t task_id, uint32_t thread_id) {
        if (stage == BuildStage::AfterHalfStaging && task_id == kTask && thread_id == kThread) {
            pause.Stop();
        }
    };

    std::thread builder([&] {
        outcome = model.BuildLeader(kTask, kThread, options);
    });
    const bool reached = pause.WaitUntilReached();
    if (!reached) {
        pause.Release();
        builder.join();
        Check(false, "builder did not reach the half-UBUF-staging boundary");
        return;
    }

    Check(model.task_state(kTask) == BuildingState(kTask), "half-staged UBUF task was not kept BUILDING");
    Check(
        model.slot_owner() == SlotOwnerValue(kTask) && model.slot_busy_depth() == 1U,
        "half-staged UBUF task lost its slot"
    );
    Check(model.TaskPayloadIsPoison(kTask), "half-staged UBUF data leaked into GM early");
    uint64_t checksum = 0U;
    uint32_t words_read = UINT32_MAX;
    Check(
        !model.TryReadPublishedPayload(kTask, checksum, words_read) && words_read == 0U,
        "executor observed a half-staged UBUF payload"
    );

    pause.Release();
    builder.join();
    Check(outcome == BuildOutcome::Built, "half-staged builder did not finish after release");
    Check(model.TaskPayloadAndTailValid(kTask), "half-staged path produced a bad GM payload or tail");
    Check(model.slot_owner() == kSlotFree && model.slot_busy_depth() == 0U, "half-staged path leaked its slot");
    Check(model.mte3_count() == 0U && model.GuardsValid(), "half-staged path changed MTE3 count or guards");
}

void TestCompleteGmPayloadHiddenBeforePublication() {
    constexpr uint32_t kTask = 2U;
    constexpr uint32_t kThread = kTask * kWarpSize;
    const uint64_t nonce = 0xA500000000000044ULL;
    SingleSlotModel model(nonce);
    PausePoint pause;
    BuildOutcome build_outcome = BuildOutcome::Aborted;
    BuildOptions build_options{};
    build_options.hook = [&](BuildStage stage, uint32_t task_id, uint32_t thread_id) {
        if (stage == BuildStage::BeforePublish && task_id == kTask && thread_id == kThread) {
            pause.Stop();
        }
    };

    std::thread builder([&] {
        build_outcome = model.BuildLeader(kTask, kThread, build_options);
    });
    const bool reached = pause.WaitUntilReached();
    if (!reached) {
        pause.Release();
        builder.join();
        Check(false, "builder did not reach the complete-GM-before-BUILT boundary");
        return;
    }

    Check(model.task_state(kTask) == BuildingState(kTask), "complete GM payload was published before BUILT CAS");
    Check(model.TaskPayloadAndTailValid(kTask), "complete pre-publication GM payload or tail is wrong");
    Check(
        model.slot_owner() == SlotOwnerValue(kTask) && model.slot_busy_depth() == 1U,
        "complete pre-publication payload lost the UBUF slot"
    );
    uint64_t checksum = 0U;
    uint32_t words_read = UINT32_MAX;
    Check(
        !model.TryReadPublishedPayload(kTask, checksum, words_read) && words_read == 0U,
        "executor consumed a complete payload before BUILT publication"
    );

    pause.Release();
    builder.join();
    Check(build_outcome == BuildOutcome::Built, "complete pre-publication builder did not finish");
    Check(model.task_state(kTask) == BuiltState(kTask), "complete payload was not published after pause release");
    Check(model.ubuf_guard_check_count() == 1U, "UBUF guards were not checked before publication");
    Check(model.slot_owner() == kSlotFree && model.slot_busy_depth() == 0U, "pre-publication path leaked slot");
    Check(model.GuardsValid(), "pre-publication path damaged a guard");
}

void TestNoReuseBeforeReleaseAndBoundedTimeout() {
    const uint64_t nonce = 0xA500000000000004ULL;
    SingleSlotModel model(nonce);
    PausePoint pause;
    BuildOutcome first_outcome = BuildOutcome::Aborted;
    BuildOptions first_options{};
    first_options.hook = [&](BuildStage stage, uint32_t task_id, uint32_t) {
        if (stage == BuildStage::AfterPublishBeforeRelease && task_id == 0U) {
            pause.Stop();
        }
    };

    std::thread first_builder([&] {
        first_outcome = model.BuildLeader(0U, 0U, first_options);
    });
    const bool reached = pause.WaitUntilReached();
    if (!reached) {
        pause.Release();
        first_builder.join();
        Check(false, "builder did not reach published-but-owned boundary");
        return;
    }

    Check(model.task_state(0U) == BuiltState(0U), "first task was not published before release pause");
    Check(
        model.slot_owner() == SlotOwnerValue(0U) && model.slot_busy_depth() == 1U, "published builder lost slot early"
    );
    BuildOptions timeout_options{};
    timeout_options.slot_poll_budget = 64U;
    const BuildOutcome timeout_outcome = model.BuildLeader(1U, kWarpSize, timeout_options);
    Check(timeout_outcome == BuildOutcome::SlotTimeout, "contender did not exit at its slot poll budget");
    Check(model.task_state(1U) == EmptyState(), "timed-out task did not roll BUILDING back to EMPTY");
    Check(model.TaskPayloadIsPoison(1U), "timed-out contender wrote GM payload");
    Check(model.slot_owner() == SlotOwnerValue(0U) && model.slot_busy_depth() == 1U, "contender reused an owned slot");
    Check(
        model.slot_acquire_count() == 1U && model.slot_release_count() == 0U, "timeout changed slot ownership counts"
    );

    const DecodedU0Fatal timeout = DecodeU0Fatal(static_cast<int64_t>(model.fatal()));
    Check(
        timeout.valid && timeout.reason == U0FatalReason::SlotTimeout && timeout.reporter_owner == kBuilderOwner &&
            timeout.task_id == 1U,
        "slot timeout did not publish the expected first fatal"
    );

    pause.Release();
    first_builder.join();
    Check(first_outcome == BuildOutcome::Built, "slot owner failed after release pause");
    Check(model.slot_owner() == kSlotFree && model.slot_busy_depth() == 0U, "slot owner did not release");
    Check(model.BuildLeader(1U, kWarpSize) == BuildOutcome::Built, "slot could not be reused after release");
    Check(model.TaskPayloadAndTailValid(1U), "post-release reuse produced a bad payload");
    Check(model.slot_acquire_count() == 2U && model.slot_release_count() == 2U, "post-release counts did not close");
    Check(model.slot_max_busy_depth() == 1U && model.mte3_count() == 0U, "reuse exceeded depth1 or used MTE3");
    Check(model.GuardsValid(), "timeout/reuse path damaged a guard");
}

void TestExceptionCleanupAndRecovery() {
    constexpr uint32_t kTask = 2U;
    constexpr uint32_t kThread = kTask * kWarpSize;
    SingleSlotModel model(0xA500000000000005ULL);
    BuildOptions options{};
    options.hook = [](BuildStage stage, uint32_t, uint32_t) {
        if (stage == BuildStage::AfterHalfStaging) {
            throw std::runtime_error("injected U0 staging failure");
        }
    };

    const BuildOutcome outcome = model.BuildLeader(kTask, kThread, options);
    Check(outcome == BuildOutcome::Aborted, "injected exception escaped or was not reported");
    Check(model.task_state(kTask) == EmptyState(), "exception left task BUILDING");
    Check(model.TaskPayloadIsPoison(kTask), "pre-GM exception changed task payload");
    Check(model.slot_owner() == kSlotFree && model.slot_busy_depth() == 0U, "exception leaked the staging slot");
    Check(
        model.slot_acquire_count() == 1U && model.slot_release_count() == 1U, "exception cleanup counts did not close"
    );
    const DecodedU0Fatal fatal = DecodeU0Fatal(static_cast<int64_t>(model.fatal()));
    Check(
        fatal.valid && fatal.reason == U0FatalReason::BuildAborted && fatal.reporter_owner == kBuilderOwner &&
            fatal.task_id == kTask,
        "exception did not publish BuildAborted from AIV0"
    );
    const U0BuildReport &failed_report = model.build_report(kTask);
    Check((failed_report.phase_bits & kBuildClaimed) != 0U, "exception report lost task claim evidence");
    Check((failed_report.phase_bits & kBuildSlotAcquired) != 0U, "exception report lost slot evidence");
    Check((failed_report.phase_bits & kBuildPublished) == 0U, "exception report falsely claims publication");
    Check(failed_report.release_count == 1U, "exception report lost cleanup release evidence");
    Check(model.BuildLeader(kTask, kThread) == BuildOutcome::Built, "clean retry after exception failed");
    Check(model.TaskPayloadAndTailValid(kTask), "retry after exception produced a bad payload");
    Check(model.slot_acquire_count() == 2U && model.slot_release_count() == 2U, "recovery slot counts did not close");
    Check(model.slot_max_busy_depth() == 1U && model.mte3_count() == 0U, "exception path exceeded depth1 or used MTE3");
    Check(model.GuardsValid(), "exception/recovery damaged a guard");
}

void TestExecutorBoundedTimeout() {
    SingleSlotModel model(0xA500000000000006ULL);
    ExecOptions options{};
    options.wait_poll_budget = 64U;
    const ExecOutcome outcome = model.ExecuteTask(0U, options);
    Check(outcome == ExecOutcome::WaitTimeout, "executor did not exit at its task wait budget");
    Check(model.task_state(0U) == EmptyState(), "executor timeout changed task state");
    Check(model.exec_claim_count() == 0U && model.done_count() == 0U, "executor timeout changed completion counts");
    const DecodedU0Fatal fatal = DecodeU0Fatal(static_cast<int64_t>(model.fatal()));
    Check(
        fatal.valid && fatal.reason == U0FatalReason::TaskWaitTimeout && fatal.reporter_owner == kExecutorOwner &&
            fatal.task_id == 0U,
        "executor timeout did not publish owner33 fatal evidence"
    );
    Check(model.mte3_count() == 0U && model.GuardsValid(), "executor timeout changed MTE3 count or guards");
}

void TestMalformedControlFailsClosed() {
    constexpr uint32_t kTask = 4U;
    SingleSlotModel model(0xA500000000000066ULL);
    const uint64_t malformed = BuiltState(kTask) | (UINT64_C(1) << 63U);
    model.SetTaskStateForTest(kTask, malformed);

    const ExecOutcome outcome = model.ExecuteTask(kTask);
    Check(outcome == ExecOutcome::InvalidState, "malformed control was treated as ordinary Claim loss or timeout");
    Check(model.task_state(kTask) == malformed, "fail-closed executor changed malformed control");
    Check(model.exec_claim_count() == 0U && model.done_count() == 0U, "malformed control changed execution counts");
    const DecodedU0Fatal fatal = DecodeU0Fatal(static_cast<int64_t>(model.fatal()));
    Check(
        fatal.valid && fatal.reason == U0FatalReason::InvalidTaskState && fatal.reporter_owner == kExecutorOwner &&
            fatal.task_id == kTask,
        "malformed control did not publish a shared-ABI InvalidTaskState fatal"
    );
    Check(model.GuardsValid(), "malformed control path damaged a guard");
}

void TestUbufGuardCorruptionFailsClosed() {
    constexpr uint32_t kTask = 5U;
    constexpr uint32_t kThread = kTask * kWarpSize;
    SingleSlotModel model(0xA500000000000077ULL);
    model.CorruptStagingGuardForTest();

    const BuildOutcome outcome = model.BuildLeader(kTask, kThread);
    Check(outcome == BuildOutcome::Aborted, "corrupted UBUF guard did not stop publication");
    Check(model.task_state(kTask) == BuildingState(kTask), "guard failure unexpectedly published or reset task");
    Check(model.TaskPayloadIsPoison(kTask), "guard failure copied staged data into GM");
    Check(model.slot_owner() == kSlotFree && model.slot_busy_depth() == 0U, "guard failure leaked UBUF slot");
    Check(model.ubuf_guard_check_count() == 0U, "failed UBUF guard was counted as valid");
    const DecodedU0Fatal fatal = DecodeU0Fatal(static_cast<int64_t>(model.fatal()));
    Check(
        fatal.valid && fatal.reason == U0FatalReason::UbufGuardCorruption && fatal.reporter_owner == kBuilderOwner &&
            fatal.task_id == kTask,
        "guard corruption did not publish a shared-ABI fatal"
    );
}

bool ValidateConcurrentRound(const SingleSlotModel &model, uint64_t nonce) {
    bool passed = true;
    passed = passed && model.slot_owner() == kSlotFree && model.slot_busy_depth() == 0U;
    passed = passed && model.slot_max_busy_depth() == 1U;
    passed = passed && model.slot_acquire_count() == kTaskCount && model.slot_release_count() == kTaskCount;
    passed = passed && model.build_claim_count() == kTaskCount && model.built_count() == kTaskCount;
    passed = passed && model.exec_claim_count() == kTaskCount && model.done_count() == kTaskCount;
    passed = passed && model.ubuf_guard_check_count() == kTaskCount;
    passed = passed && model.builder_finished_count() == kTaskCount && model.executor_finished_count() == 1U;
    passed = passed && model.mte3_count() == 0U && model.fatal() == 0U && model.GuardsValid();

    std::array<bool, kTaskCount + 1U> ticket_seen{};
    std::array<uint32_t, kPayloadClassCount> payload_class_counts{};
    for (uint32_t task_id = 0U; task_id < kTaskCount; ++task_id) {
        passed = passed && model.task_state(task_id) == DoneState(task_id);
        passed = passed && model.TaskPayloadAndTailValid(task_id);
        const U0BuildReport &build = model.build_report(task_id);
        const uint32_t payload_words = PayloadWordsForTask(task_id);
        passed = passed && build.task_id == task_id && build.builder_thread == task_id * kWarpSize;
        passed = passed && build.builder_warp == task_id && build.builder_lane == 0U;
        passed = passed && build.phase_bits == kExpectedBuildPhaseBits;
        passed = passed && build.payload_lines == PayloadLinesForTask(task_id);
        passed = passed && build.ubuf_words_written == payload_words && build.gm_words_stored == payload_words;
        passed = passed && build.claim_count == 1U && build.publish_count == 1U && build.release_count == 1U;
        passed = passed && build.slot_ticket > 0U && build.slot_ticket <= kTaskCount;
        if (build.slot_ticket <= kTaskCount) {
            passed = passed && !ticket_seen[build.slot_ticket];
            ticket_seen[build.slot_ticket] = true;
        }
        passed = passed && build.launch_nonce == nonce;
        passed = passed && build.payload_checksum == ExpectedPayloadChecksum(nonce, task_id);

        const U0ExecReport &exec = model.exec_report(task_id);
        passed = passed && exec.task_id == task_id && exec.executor_owner == kExecutorOwner;
        passed = passed && exec.executor_physical_block == 0U && exec.executor_subblock_id == 1U;
        passed = passed && exec.phase_bits == kExpectedExecPhaseBits;
        passed = passed && exec.payload_lines == PayloadLinesForTask(task_id);
        passed = passed && exec.payload_words_read == payload_words;
        passed = passed && exec.claim_count == 1U && exec.completion_count == 1U && exec.checksum_match_count == 1U;
        passed = passed && exec.launch_nonce == nonce;
        passed = passed && exec.payload_checksum == ExpectedPayloadChecksum(nonce, task_id);

        const uint32_t leader_thread = task_id * kWarpSize;
        const U0ThreadReport &thread = model.builder_thread_report(leader_thread);
        passed = passed && thread.thread_id == leader_thread && thread.warp_id == task_id && thread.lane_id == 0U;
        passed = passed && thread.active_leader == 1U && thread.task_id == task_id;
        passed = passed && thread.status == kExpectedBuilderThreadStatus && thread.task_attempt_count == 1U;
        passed = passed && thread.slot_attempt_count >= 1U && thread.launch_nonce == nonce;
        passed = passed &&
                 thread.checksum == ExpectedThreadChecksum(nonce, leader_thread, task_id, kExpectedBuilderThreadStatus);
        ++payload_class_counts[task_id % kPayloadClassCount];
    }

    for (uint32_t ticket = 1U; ticket <= kTaskCount; ++ticket) {
        passed = passed && ticket_seen[ticket];
    }
    for (uint32_t count : payload_class_counts) {
        passed = passed && count == kTaskCount / kPayloadClassCount;
    }
    for (uint32_t thread_id = 0U; thread_id < kThreadCount; ++thread_id) {
        if (thread_id % kWarpSize != 0U) {
            passed = passed && model.ThreadReportIsPoison(model.builder_thread_report(thread_id));
        }
    }
    return passed;
}

bool RunConcurrentRound(uint32_t round) {
    const uint64_t nonce = 0xC0DE000000000000ULL ^ static_cast<uint64_t>(round + 1U);
    SingleSlotModel model(nonce);
    StartGate gate(kTaskCount + 1U);
    std::array<BuildOutcome, kTaskCount> build_outcomes{};
    std::array<ExecOutcome, kTaskCount> exec_outcomes{};
    build_outcomes.fill(BuildOutcome::Aborted);
    exec_outcomes.fill(ExecOutcome::WaitTimeout);
    std::array<uint32_t, kTaskCount> build_order{};
    for (uint32_t task_id = 0U; task_id < kTaskCount; ++task_id) {
        build_order[task_id] = task_id;
    }
    std::mt19937_64 random(UINT64_C(0x553052414E444F4D) ^ nonce);
    std::shuffle(build_order.begin(), build_order.end(), random);
    std::array<uint32_t, kTaskCount> jitter_steps{};
    for (uint32_t task_id = 0U; task_id < kTaskCount; ++task_id) {
        jitter_steps[task_id] = static_cast<uint32_t>(random() & 7U);
    }

    std::vector<std::thread> builders;
    builders.reserve(kTaskCount);
    for (uint32_t task_id : build_order) {
        builders.emplace_back([&, task_id] {
            gate.ArriveAndWait();
            for (uint32_t step = 0U; step < jitter_steps[task_id]; ++step) {
                std::this_thread::yield();
            }
            build_outcomes[task_id] = model.BuildLeader(task_id, task_id * kWarpSize);
        });
    }
    std::thread executor([&] {
        gate.ArriveAndWait();
        for (uint32_t task_id = 0U; task_id < kTaskCount; ++task_id) {
            exec_outcomes[task_id] = model.ExecuteTask(task_id);
        }
    });

    const bool all_arrived = gate.WaitAndOpen();
    for (std::thread &builder : builders) {
        builder.join();
    }
    executor.join();
    if (!all_arrived) {
        return false;
    }
    for (BuildOutcome outcome : build_outcomes) {
        if (outcome != BuildOutcome::Built) {
            return false;
        }
    }
    for (ExecOutcome outcome : exec_outcomes) {
        if (outcome != ExecOutcome::Done) {
            return false;
        }
    }
    return ValidateConcurrentRound(model, nonce);
}

bool ParseRounds(int argc, char **argv, uint32_t &rounds) {
    rounds = 8U;
    if (argc == 1) {
        return true;
    }
    if (argc != 3 || std::strcmp(argv[1], "--rounds") != 0) {
        std::fprintf(stderr, "usage: %s [--rounds N]\n", argv[0]);
        return false;
    }
    char *end = nullptr;
    const unsigned long value = std::strtoul(argv[2], &end, 10);
    if (end == argv[2] || *end != '\0' || value == 0U || value > 10000U) {
        std::fprintf(stderr, "invalid --rounds value: %s\n", argv[2]);
        return false;
    }
    rounds = static_cast<uint32_t>(value);
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    uint32_t rounds = 0U;
    if (!ParseRounds(argc, argv, rounds)) {
        return EXIT_FAILURE;
    }

    TestAbiAndTransportContract();
    TestInactiveLaneAndMismatchedWarpRejected();
    TestHalfUbufStagingDoesNotTouchGm();
    TestHalfWriteHiddenUntilPublication();
    TestCompleteGmPayloadHiddenBeforePublication();
    TestNoReuseBeforeReleaseAndBoundedTimeout();
    TestExceptionCleanupAndRecovery();
    TestExecutorBoundedTimeout();
    TestMalformedControlFailsClosed();
    TestUbufGuardCorruptionFailsClosed();
    for (uint32_t round = 0U; round < rounds; ++round) {
        if (!RunConcurrentRound(round)) {
            std::fprintf(stderr, "[FAIL] U0 CPU 64-warp stress round=%u\n", round);
            ++g_failures;
            break;
        }
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "[FAIL] U0 CPU single-slot failures=%d\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf(
        "[PASS] U0 CPU rounds=%u: 64 SIMT leaders, one AIV1 executor, single-slot depth=1, "
        "seeded shuffle, pre-BUILT hidden, fail-closed control/guards, bounded faults, direct GM stores, mte3=0\n",
        rounds
    );
    return EXIT_SUCCESS;
}
