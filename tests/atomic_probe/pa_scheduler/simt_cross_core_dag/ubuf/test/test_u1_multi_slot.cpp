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

#include "../common/u1_multi_slot_cpu_model.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

using namespace pa_scheduler::simt_cross_core;
using namespace pa_scheduler::simt_cross_core::u1;
using namespace pa_scheduler::simt_cross_core::u1::cpu;

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] U1 CPU multi-slot: %s\n", message);
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

bool BootstrapAnchors(MultiSlotModel &model) {
    std::array<BuildOutcome, kAnchorTaskCount> outcomes{};
    outcomes.fill(BuildOutcome::Aborted);
    std::array<std::thread, kAnchorTaskCount> builders;
    for (uint32_t task = 0U; task < kAnchorTaskCount; ++task) {
        builders[task] = std::thread([&, task] {
            outcomes[task] = model.BuildTask(task, task * kWarpSize);
        });
    }
    for (std::thread &builder : builders) {
        builder.join();
    }
    for (BuildOutcome outcome : outcomes) {
        if (outcome != BuildOutcome::Built) {
            return false;
        }
    }
    return true;
}

void TestAbiAndMappingContract() {
    Check(kWarpSize == 32U && kWarpCount == 64U && kThreadCount == 2048U, "SIMT topology changed");
    Check(kTaskCount == 128U && kTasksPerWarp == 2U, "U1 is not 64 leaders times two tasks");
    Check(
        FirstTaskForWarp(0U) == 0U && LastTaskForWarp(0U) == 64U && FirstTaskForWarp(63U) == 63U &&
            LastTaskForWarp(63U) == 127U,
        "warp-to-task mapping changed"
    );
    Check(
        kUbufSlotCount == 4U && kUbufSlotStrideBytes == 1152U && kUbufRegionBytes == 4608U &&
            kUbufPayloadOffsetBytes == 64U && kUbufGuardAfterOffsetWords * sizeof(uint64_t) == 1088U,
        "four guarded UBUF slots no longer occupy 4608 bytes"
    );
    Check(
        3U * kUbufSlotStrideBytes + kUbufPayloadOffsetBytes + (kMaxPayloadLines - 1U) * kCacheLineBytes == 4480U &&
            3U * kUbufSlotStrideBytes + kUbufGuardAfterOffsetWords * sizeof(uint64_t) == 4544U &&
            3U * kUbufSlotStrideBytes + kUbufSlotStrideBytes == 4608U,
        "slot3 final payload/guard offsets 4480..4607 changed"
    );
    Check(
        PayloadLinesForTask(0U) == 16U && PayloadLinesForTask(4U) == 1U && PayloadLinesForTask(8U) == 4U &&
            PayloadLinesForTask(12U) == 10U,
        "payload classes are not the rotated 16/1/4/10 sequence"
    );
    for (uint32_t slot = 0U; slot < kUbufSlotCount; ++slot) {
        std::array<uint32_t, kPayloadClassCount> classes{};
        for (uint32_t task = slot; task < kTaskCount; task += kUbufSlotCount) {
            ++classes[PayloadClassForTask(task)];
            Check(SlotForTask(task) == slot, "task-to-slot mapping changed");
        }
        for (uint32_t count : classes) {
            Check(count == 8U, "one physical slot does not see eight tasks from every payload class");
        }
    }

    const uint64_t busy = SlotBusyState(7U, 60U);
    const DecodedSlotState decoded_busy = DecodeSlotState(static_cast<int64_t>(busy));
    Check(
        decoded_busy.valid && decoded_busy.busy && decoded_busy.generation == 7U && decoded_busy.task_id == 60U,
        "slot generation/owner encoding does not round-trip"
    );
    Check(SlotStateValidForSlot(busy, SlotForTask(60U)), "valid contextual slot state was rejected");
    Check(!SlotStateValidForSlot(busy, 1U), "slot accepted an owner mapped to a different slot");
    const DecodedSlotState final_free = DecodeSlotState(static_cast<int64_t>(SlotFreeState(32U)));
    Check(final_free.valid && !final_free.busy && final_free.generation == 32U, "terminal FREE(32) does not decode");

    const uint64_t raw_fatal = EncodeU1Fatal(U1FatalReason::InvalidTaskState, kExecutorOwner, 127U);
    const DecodedU1Fatal decoded_fatal = DecodeU1Fatal(static_cast<int64_t>(raw_fatal));
    Check(
        decoded_fatal.valid && decoded_fatal.reason == U1FatalReason::InvalidTaskState &&
            decoded_fatal.reporter_owner == kExecutorOwner && decoded_fatal.task_id == 127U,
        "U1 fatal ABI does not round-trip"
    );
    Check(
        !DecodeU1Fatal(static_cast<int64_t>(EncodeU1Fatal(U1FatalReason::InvalidTaskState, kExecutorOwner, 128U)))
             .valid,
        "fatal ABI accepted an out-of-range task"
    );

    MultiSlotModel model(0xA510000000000001ULL);
    Check(model.GuardsValid(), "fresh U1 model guards are invalid");
    Check(model.mte3_count() == 0U, "direct GM-store U1 model reported MTE3");
    Check(model.anchor_staged_count() == 0U && model.anchor_staged_mask() == 0U, "fresh anchor gate is open");
}

void TestInactiveLaneAndWrongWarpRejected() {
    MultiSlotModel model(0xA510000000000002ULL);
    const WarpBuildResult nonleader = model.BuildWarpLeader(1U);
    Check(
        nonleader.first == BuildOutcome::InvalidLeader && nonleader.second == BuildOutcome::InvalidLeader,
        "lane1 was accepted as a warp builder"
    );
    Check(model.BuildTask(64U, kWarpSize) == BuildOutcome::InvalidLeader, "warp1 was accepted for task64");
    Check(model.BuildTask(128U, 0U) == BuildOutcome::InvalidLeader, "out-of-range task was accepted");
    Check(
        model.task_state(0U) == EmptyState() && model.task_state(64U) == EmptyState(), "invalid actor claimed a task"
    );
    Check(model.build_claim_count() == 0U && model.global_busy_depth() == 0U, "invalid actor changed counters");
    Check(model.ThreadReportIsPoison(model.builder_thread_report(0U)), "rejected actor wrote leader report");
    Check(model.ThreadReportIsPoison(model.builder_thread_report(1U)), "inactive lane report lost poison");
}

void TestFourAnchorsHoldFourFullyStagedSlots() {
    const uint64_t nonce = 0xA510000000000003ULL;
    MultiSlotModel model(nonce);
    std::array<PausePoint, kAnchorTaskCount> pauses;
    std::array<BuildOutcome, kAnchorTaskCount> outcomes{};
    outcomes.fill(BuildOutcome::Aborted);
    BuildOptions options{};
    options.hook = [&](BuildStage stage, uint32_t task_id, uint32_t) {
        if (stage == BuildStage::AfterAnchorArrive) {
            pauses[task_id].Stop();
        }
    };

    std::array<std::thread, kAnchorTaskCount> builders;
    for (uint32_t task = 0U; task < kAnchorTaskCount; ++task) {
        builders[task] = std::thread([&, task] {
            outcomes[task] = model.BuildTask(task, task * kWarpSize, options);
        });
    }
    bool all_reached = true;
    for (PausePoint &pause : pauses) {
        all_reached = pause.WaitUntilReached() && all_reached;
    }
    if (!all_reached) {
        for (PausePoint &pause : pauses) {
            pause.Release();
        }
        for (std::thread &builder : builders) {
            builder.join();
        }
        Check(false, "four anchor leaders did not reach the fully-staged hold point");
        return;
    }

    Check(model.anchor_staged_count() == 4U, "anchor count did not reach four");
    Check(model.anchor_staged_mask() == 0xFU, "anchor identity mask is not task0..3");
    Check(model.global_busy_depth() == 4U && model.global_max_busy_depth() == 4U, "four slots were not busy together");
    for (uint32_t slot = 0U; slot < kUbufSlotCount; ++slot) {
        Check(model.slot_state(slot) == SlotBusyState(0U, slot), "anchor does not own its distinct generation0 slot");
        Check(model.slot_busy_depth(slot) == 1U && model.slot_max_busy_depth(slot) == 1U, "slot depth is not one");
        Check(model.slot_acquire_count(slot) == 1U && model.slot_release_count(slot) == 0U, "anchor counts are wrong");
        Check(model.StagingGuardsValid(slot), "anchor damaged its independent slot guards");
        for (uint32_t word = 0U; word < kMaxPayloadWords; ++word) {
            Check(
                model.staging_word(slot, word) == ExpectedPayloadWord(nonce, slot, word),
                "anchor did not fully stage its 16-line payload"
            );
        }
        Check(model.task_state(slot) == BuildingState(slot), "anchor published before the four-slot gate");
        Check(model.TaskPayloadIsPoison(slot), "anchor copied UBUF payload into GM before the gate");
    }
    Check(model.task_state(4U) == EmptyState(), "non-anchor task was touched without a leader");

    for (PausePoint &pause : pauses) {
        pause.Release();
    }
    for (std::thread &builder : builders) {
        builder.join();
    }
    for (uint32_t task = 0U; task < kAnchorTaskCount; ++task) {
        Check(outcomes[task] == BuildOutcome::Built, "anchor did not finish after hold release");
        Check(model.task_state(task) == BuiltState(task), "anchor was not published BUILT");
        Check(model.TaskPayloadAndTailValid(task), "anchor GM payload is wrong");
        const U1BuildReport &report = model.build_report(task);
        Check(report.phase_bits == kExpectedBuildPhaseBits, "anchor report missed a lifecycle phase");
        Check(report.slot_id == task && report.slot_generation == 0U, "anchor report lost slot provenance");
        Check(model.slot_state(task) == SlotFreeState(1U), "anchor release did not advance generation to one");
        Check(model.slot_release_count(task) == 1U, "anchor release count did not close");
    }
    Check(model.global_busy_depth() == 0U && model.fatal() == 0U, "anchor success leaked a slot or fatal");
}

void TestPublicationBoundaryNoEarlyReuseAndRuntimeGeneration() {
    const uint64_t nonce = 0xA510000000000004ULL;
    MultiSlotModel model(nonce);
    Check(BootstrapAnchors(model), "anchor bootstrap failed before publication test");

    constexpr uint32_t kOwnerTask = 60U;
    constexpr uint32_t kContenderTask = 4U;
    constexpr uint32_t kSlot = 0U;
    PausePoint pause;
    BuildOutcome owner_outcome = BuildOutcome::Aborted;
    BuildOptions owner_options{};
    owner_options.hook = [&](BuildStage stage, uint32_t task_id, uint32_t) {
        if (stage == BuildStage::BeforePublish && task_id == kOwnerTask) {
            pause.Stop();
        }
    };
    std::thread owner([&] {
        owner_outcome = model.BuildTask(kOwnerTask, (kOwnerTask % kWarpCount) * kWarpSize, owner_options);
    });
    if (!pause.WaitUntilReached()) {
        pause.Release();
        owner.join();
        Check(false, "runtime-generation owner did not reach pre-publish pause");
        return;
    }

    Check(model.task_state(kOwnerTask) == BuildingState(kOwnerTask), "complete GM payload became visible before BUILT");
    Check(model.TaskPayloadAndTailValid(kOwnerTask), "pre-publish complete GM payload is wrong");
    Check(model.slot_state(kSlot) == SlotBusyState(1U, kOwnerTask), "task60 did not acquire runtime generation one");
    Check(
        !model.TryExactReleaseForTest(kSlot, 0U, 0U), "stale task0/generation0 release cleared the newer task60 owner"
    );
    Check(model.slot_state(kSlot) == SlotBusyState(1U, kOwnerTask), "stale release changed current slot ownership");
    uint64_t checksum = 0U;
    uint32_t words_read = UINT32_MAX;
    Check(
        !model.TryReadPublishedPayload(kOwnerTask, checksum, words_read) && words_read == 0U,
        "executor consumed a complete payload before BUILT publication"
    );

    BuildOptions contender_options{};
    contender_options.slot_poll_budget = 64U;
    const BuildOutcome contender = model.BuildTask(kContenderTask, kContenderTask * kWarpSize, contender_options);
    Check(contender == BuildOutcome::SlotTimeout, "same-slot contender did not stop at its poll budget");
    Check(model.task_state(kContenderTask) == EmptyState(), "slot-timeout contender did not reset BUILDING");
    Check(model.slot_state(kSlot) == SlotBusyState(1U, kOwnerTask), "contender stole or released the owner's slot");
    Check(
        model.slot_acquire_count(kSlot) == 2U && model.slot_release_count(kSlot) == 1U,
        "failed contender changed acquire/release counts"
    );

    pause.Release();
    owner.join();
    Check(owner_outcome == BuildOutcome::Aborted, "owner published after observing the contender's fatal");
    Check(model.task_state(kOwnerTask) == EmptyState(), "fatal cleanup left the owner task BUILDING or BUILT");
    Check(model.slot_state(kSlot) == SlotFreeState(2U), "fatal cleanup did not advance runtime generation");
    Check(model.build_report(kOwnerTask).slot_generation == 1U, "build report derived generation from task id");
    Check(
        model.slot_acquire_count(kSlot) == 2U && model.slot_release_count(kSlot) == 2U,
        "fatal cleanup lifecycle counts did not close"
    );
    Check(model.global_busy_depth() == 0U, "fatal cleanup leaked global busy depth");
    Check(
        !model.TryReadPublishedPayload(kOwnerTask, checksum, words_read) && words_read == 0U,
        "executor observed a payload that fatal cleanup did not publish"
    );
    Check(
        (model.build_report(kOwnerTask).phase_bits & kBuildSlotReleased) != 0U &&
            (model.build_report(kOwnerTask).phase_bits & kBuildPublished) == 0U,
        "fatal cleanup report lost release evidence or claimed publication"
    );
    const DecodedU1Fatal fatal = DecodeU1Fatal(static_cast<int64_t>(model.fatal()));
    Check(
        fatal.valid && fatal.reason == U1FatalReason::SlotTimeout && fatal.task_id == kContenderTask,
        "bounded contender did not publish the expected first fatal"
    );
}

void TestShortLongAlternationDoesNotOverwriteTails() {
    const uint64_t nonce = 0xA510000000000005ULL;
    MultiSlotModel model(nonce);
    Check(BootstrapAnchors(model), "anchor bootstrap failed before short/long alternation");
    constexpr std::array<uint32_t, 4U> kTasks{4U, 8U, 12U, 16U};
    uint32_t expected_generation = 1U;
    for (uint32_t task : kTasks) {
        BuildOptions options{};
        options.hook = [&](BuildStage stage, uint32_t observed_task, uint32_t) {
            if (stage != BuildStage::AfterStaging || observed_task != task) {
                return;
            }
            const uint32_t valid_words = PayloadWordsForTask(task);
            for (uint32_t word = 0U; word < valid_words; ++word) {
                Check(
                    model.staging_word(0U, word) == ExpectedPayloadWord(nonce, task, word),
                    "current task did not stage every valid word"
                );
            }
            for (uint32_t word = valid_words; word < kMaxPayloadWords; ++word) {
                Check(
                    model.staging_word(0U, word) == ExpectedPayloadWord(nonce, 0U, word),
                    "short task overwrote or cleared the untouched staging tail"
                );
            }
        };
        const BuildOutcome outcome = model.BuildTask(task, task * kWarpSize, options);
        Check(outcome == BuildOutcome::Built, "short/long alternating task failed");
        Check(model.TaskPayloadAndTailValid(task), "valid GM prefix or poison tail is wrong");
        Check(model.build_report(task).slot_generation == expected_generation, "sequential slot generation is wrong");
        ++expected_generation;
    }
    Check(model.slot_state(0U) == SlotFreeState(5U), "short/long sequence did not advance slot0 five times");
    Check(model.slot_acquire_count(0U) == 5U && model.slot_release_count(0U) == 5U, "slot0 counts did not close");
    Check(model.fatal() == 0U && model.GuardsValid(), "short/long alternation damaged guards or raised fatal");
}

void TestPrepublishExceptionReleasesExactGeneration() {
    MultiSlotModel model(0xA510000000000006ULL);
    Check(BootstrapAnchors(model), "anchor bootstrap failed before exception test");
    constexpr uint32_t kTask = 5U;
    constexpr uint32_t kSlot = 1U;
    BuildOptions options{};
    options.hook = [](BuildStage stage, uint32_t, uint32_t) {
        if (stage == BuildStage::AfterHalfStaging) {
            throw std::runtime_error("injected U1 staging failure");
        }
    };
    const BuildOutcome outcome = model.BuildTask(kTask, kTask * kWarpSize, options);
    Check(outcome == BuildOutcome::Aborted, "pre-publish exception escaped or was not reported");
    Check(model.task_state(kTask) == EmptyState(), "pre-publish exception left task BUILDING");
    Check(model.TaskPayloadIsPoison(kTask), "pre-GM exception touched task payload");
    Check(model.slot_state(kSlot) == SlotFreeState(2U), "exception cleanup did not advance generation");
    Check(model.slot_busy_depth(kSlot) == 0U && model.global_busy_depth() == 0U, "exception leaked busy depth");
    Check(
        model.slot_acquire_count(kSlot) == 2U && model.slot_release_count(kSlot) == 2U,
        "exception acquire/release counts did not close"
    );
    const U1BuildReport &report = model.build_report(kTask);
    Check(report.slot_id == kSlot && report.slot_generation == 1U, "exception report lost exact slot generation");
    Check(
        (report.phase_bits & (kBuildClaimed | kBuildSlotAcquired | kBuildSlotReleased)) ==
            (kBuildClaimed | kBuildSlotAcquired | kBuildSlotReleased),
        "exception report lost claim/acquire/release evidence"
    );
    Check((report.phase_bits & kBuildPublished) == 0U, "exception report falsely claims publication");
    const DecodedU1Fatal fatal = DecodeU1Fatal(static_cast<int64_t>(model.fatal()));
    Check(
        fatal.valid && fatal.reason == U1FatalReason::BuildAborted && fatal.task_id == kTask,
        "exception did not publish BuildAborted"
    );
}

void TestGuardCorruptionReleasesExactGeneration() {
    MultiSlotModel model(0xA510000000000007ULL);
    Check(BootstrapAnchors(model), "anchor bootstrap failed before guard test");
    constexpr uint32_t kTask = 6U;
    constexpr uint32_t kSlot = 2U;
    BuildOptions options{};
    options.hook = [&](BuildStage stage, uint32_t task_id, uint32_t) {
        if (stage == BuildStage::AfterStaging && task_id == kTask) {
            model.CorruptStagingGuardForTest(kSlot);
        }
    };
    const BuildOutcome outcome = model.BuildTask(kTask, kTask * kWarpSize, options);
    Check(outcome == BuildOutcome::GuardCorruption, "corrupt UBUF guard did not stop publication");
    Check(model.task_state(kTask) == EmptyState(), "guard failure did not reset unpublished task");
    Check(model.TaskPayloadIsPoison(kTask), "guard failure copied staged payload to GM");
    Check(model.slot_state(kSlot) == SlotFreeState(2U), "guard cleanup did not advance generation");
    Check(
        model.slot_acquire_count(kSlot) == 2U && model.slot_release_count(kSlot) == 2U,
        "guard failure acquire/release counts did not close"
    );
    Check(!model.StagingGuardsValid(kSlot), "injected guard corruption was not retained as evidence");
    Check(model.ubuf_guard_check_count() == kAnchorTaskCount, "failed guard was counted as valid");
    const DecodedU1Fatal fatal = DecodeU1Fatal(static_cast<int64_t>(model.fatal()));
    Check(
        fatal.valid && fatal.reason == U1FatalReason::UbufGuardCorruption && fatal.task_id == kTask,
        "guard failure did not publish its precise fatal"
    );
}

void TestAnchorTimeoutReleasesAndAdvancesGeneration() {
    MultiSlotModel model(0xA510000000000008ULL);
    BuildOptions options{};
    options.anchor_poll_budget = 64U;
    const BuildOutcome outcome = model.BuildTask(0U, 0U, options);
    Check(outcome == BuildOutcome::AnchorTimeout, "single anchor did not exit at its gate poll budget");
    Check(model.task_state(0U) == EmptyState(), "timed-out anchor did not reset unpublished task");
    Check(model.anchor_staged_count() == 1U && model.anchor_staged_mask() == 0x1U, "anchor identity evidence is wrong");
    Check(model.slot_state(0U) == SlotFreeState(1U), "anchor timeout did not release with generation+1");
    Check(model.slot_acquire_count(0U) == 1U && model.slot_release_count(0U) == 1U, "timeout counts did not close");
    Check(model.global_busy_depth() == 0U && model.global_max_busy_depth() == 1U, "timeout busy depth is wrong");
    const U1BuildReport &report = model.build_report(0U);
    Check(report.slot_generation == 0U, "anchor timeout report lost acquired generation");
    Check((report.phase_bits & kBuildSlotReleased) != 0U, "anchor timeout report lost exact release evidence");
    Check((report.phase_bits & kBuildPublished) == 0U, "anchor timeout falsely published task");
    const DecodedU1Fatal fatal = DecodeU1Fatal(static_cast<int64_t>(model.fatal()));
    Check(
        fatal.valid && fatal.reason == U1FatalReason::SlotTimeout && fatal.task_id == 0U,
        "anchor timeout did not publish bounded fatal"
    );
}

void TestAnchorCrossLineVisibilitySkewRetries() {
    MultiSlotModel model(0xA510000000000089ULL);
    model.SetAnchorGateForTest(kAnchorTaskCount, kAnchorStagedMask ^ 0x8U);
    BuildOptions options{};
    options.anchor_poll_budget = 1U;
    constexpr uint32_t kTask = 4U;
    const BuildOutcome outcome = model.BuildTask(kTask, kTask * kWarpSize, options);
    Check(outcome == BuildOutcome::AnchorTimeout, "in-range count/mask skew was treated as a permanent invariant");
    Check(model.task_state(kTask) == EmptyState(), "count/mask retry timeout left the task BUILDING");
    Check(model.slot_acquire_count(0U) == 0U, "non-anchor acquired a slot before the exact gate pair was visible");
    const DecodedU1Fatal fatal = DecodeU1Fatal(static_cast<int64_t>(model.fatal()));
    Check(
        fatal.valid && fatal.reason == U1FatalReason::SlotTimeout && fatal.task_id == kTask,
        "count/mask retry did not close through the bounded timeout"
    );
}

void TestAnchorHolderCleansUpAfterForeignFatal() {
    MultiSlotModel model(0xA510000000000088ULL);
    BuildOutcome anchor_outcome = BuildOutcome::Built;
    std::thread anchor([&] {
        anchor_outcome = model.BuildTask(0U, 0U);
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (model.anchor_waiting_count() != 1U && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    if (model.anchor_waiting_count() != 1U) {
        Check(false, "anchor0 did not enter the gate wait while holding slot0");
        model.SetTaskStateForTest(70U, BuiltState(70U) | (UINT64_C(1) << 63U));
        (void)model.ExecuteTask(70U, ExecOptions{1U});
        anchor.join();
        return;
    }

    Check(model.slot_state(0U) == SlotBusyState(0U, 0U), "waiting anchor0 did not retain exact slot ownership");
    Check(model.global_busy_depth() == 1U, "waiting anchor0 is missing from global busy depth");
    constexpr uint32_t kMalformedTask = 70U;
    const uint64_t malformed = BuiltState(kMalformedTask) | (UINT64_C(1) << 63U);
    model.SetTaskStateForTest(kMalformedTask, malformed);
    Check(
        model.ExecuteTask(kMalformedTask) == ExecOutcome::InvalidState,
        "foreign AIV1 path did not publish malformed-control fatal"
    );
    anchor.join();

    Check(anchor_outcome == BuildOutcome::Aborted, "anchor ignored a foreign global fatal");
    Check(model.anchor_waiting_count() == 0U, "anchor wait accounting did not close");
    Check(model.task_state(0U) == EmptyState(), "foreign fatal left unpublished anchor BUILDING");
    Check(model.slot_state(0U) == SlotFreeState(1U), "foreign fatal did not exact-release anchor generation0");
    Check(
        model.slot_acquire_count(0U) == 1U && model.slot_release_count(0U) == 1U, "foreign fatal counts did not close"
    );
    Check(model.global_busy_depth() == 0U, "foreign fatal leaked global busy depth");
    Check((model.build_report(0U).phase_bits & kBuildSlotReleased) != 0U, "foreign fatal report lost release evidence");
    const DecodedU1Fatal fatal = DecodeU1Fatal(static_cast<int64_t>(model.fatal()));
    Check(
        fatal.valid && fatal.reason == U1FatalReason::InvalidTaskState && fatal.reporter_owner == kExecutorOwner &&
            fatal.task_id == kMalformedTask,
        "anchor cleanup overwrote the foreign first fatal"
    );
}

void TestExecutorFailsClosedAndChecksEveryWord() {
    {
        MultiSlotModel model(0xA510000000000009ULL);
        constexpr uint32_t kTask = 70U;
        const uint64_t malformed = BuiltState(kTask) | (UINT64_C(1) << 63U);
        model.SetTaskStateForTest(kTask, malformed);
        const ExecOutcome outcome = model.ExecuteTask(kTask);
        Check(outcome == ExecOutcome::InvalidState, "malformed control was treated as wait or Claim loss");
        Check(model.task_state(kTask) == malformed, "fail-closed executor changed malformed control");
        Check(model.exec_claim_count() == 0U && model.done_count() == 0U, "malformed control changed counts");
        const DecodedU1Fatal fatal = DecodeU1Fatal(static_cast<int64_t>(model.fatal()));
        Check(
            fatal.valid && fatal.reason == U1FatalReason::InvalidTaskState && fatal.reporter_owner == kExecutorOwner &&
                fatal.task_id == kTask,
            "malformed control did not publish AIV1 InvalidTaskState"
        );
    }
    {
        MultiSlotModel model(0xA51000000000000AULL);
        ExecOptions options{};
        options.wait_poll_budget = 64U;
        Check(model.ExecuteTask(0U, options) == ExecOutcome::WaitTimeout, "executor wait was not bounded");
        const DecodedU1Fatal fatal = DecodeU1Fatal(static_cast<int64_t>(model.fatal()));
        Check(
            fatal.valid && fatal.reason == U1FatalReason::TaskWaitTimeout && fatal.reporter_owner == kExecutorOwner,
            "executor timeout did not publish AIV1 fatal"
        );
    }
    {
        MultiSlotModel model(0xA51000000000000BULL);
        Check(BootstrapAnchors(model), "anchor bootstrap failed before word-wise executor oracle");
        model.SetPayloadWordForTest(0U, 0U, model.payload_word(0U, 0U) ^ 1U);
        Check(model.ExecuteTask(0U) == ExecOutcome::PayloadMismatch, "executor accepted a wrong payload word");
        const DecodedU1Fatal fatal = DecodeU1Fatal(static_cast<int64_t>(model.fatal()));
        Check(
            fatal.valid && fatal.reason == U1FatalReason::PayloadMismatch && fatal.task_id == 0U,
            "word mismatch did not publish PayloadMismatch"
        );
    }
    {
        MultiSlotModel model(0xA51000000000000CULL);
        constexpr uint32_t kFirstBuilt = 0U;
        constexpr uint32_t kSecondBuilt = 1U;
        constexpr uint32_t kMalformedTask = 70U;
        model.SetTaskStateForTest(kFirstBuilt, BuiltState(kFirstBuilt));
        model.SetTaskStateForTest(kSecondBuilt, BuiltState(kSecondBuilt));
        model.SetTaskStateForTest(kMalformedTask, BuiltState(kMalformedTask) | (UINT64_C(1) << 63U));
        Check(
            model.ExecuteTask(kMalformedTask) == ExecOutcome::InvalidState,
            "foreign malformed task did not establish the first fatal"
        );
        Check(model.ExecuteTask(kFirstBuilt) == ExecOutcome::Aborted, "executor claimed task0 after foreign fatal");
        Check(model.ExecuteTask(kSecondBuilt) == ExecOutcome::Aborted, "executor claimed task1 after foreign fatal");
        Check(
            model.task_state(kFirstBuilt) == BuiltState(kFirstBuilt) &&
                model.task_state(kSecondBuilt) == BuiltState(kSecondBuilt),
            "foreign fatal changed a remaining BUILT task"
        );
        Check(model.exec_claim_count() == 0U && model.done_count() == 0U, "foreign fatal allowed claim/DONE progress");
        const DecodedU1Fatal fatal = DecodeU1Fatal(static_cast<int64_t>(model.fatal()));
        Check(
            fatal.valid && fatal.reason == U1FatalReason::InvalidTaskState && fatal.task_id == kMalformedTask,
            "post-fatal executor path overwrote the first fatal"
        );
    }
}

bool ValidateConcurrentRound(const MultiSlotModel &model, uint64_t nonce) {
    bool passed = true;
    passed = passed && model.global_busy_depth() == 0U && model.global_max_busy_depth() == kUbufSlotCount;
    passed = passed && model.anchor_staged_count() == kAnchorTaskCount && model.anchor_staged_mask() == 0xFU;
    passed = passed && model.build_claim_count() == kTaskCount && model.built_count() == kTaskCount;
    passed = passed && model.exec_claim_count() == kTaskCount && model.done_count() == kTaskCount;
    passed = passed && model.ubuf_guard_check_count() == kTaskCount;
    passed = passed && model.builder_finished_count() == kWarpCount && model.executor_finished_count() == 1U;
    passed = passed && model.mte3_count() == 0U && model.fatal() == 0U && model.GuardsValid();

    std::array<std::array<bool, kSlotReuseCount>, kUbufSlotCount> generation_seen{};
    std::array<std::array<uint32_t, kPayloadClassCount>, kUbufSlotCount> class_counts{};
    for (uint32_t slot = 0U; slot < kUbufSlotCount; ++slot) {
        passed = passed && model.slot_state(slot) == SlotFreeState(kSlotReuseCount);
        passed = passed && model.slot_busy_depth(slot) == 0U && model.slot_max_busy_depth(slot) == 1U;
        passed = passed && model.slot_acquire_count(slot) == kSlotReuseCount;
        passed = passed && model.slot_release_count(slot) == kSlotReuseCount;
    }

    for (uint32_t task = 0U; task < kTaskCount; ++task) {
        const uint32_t slot = SlotForTask(task);
        const uint32_t payload_words = PayloadWordsForTask(task);
        passed = passed && model.task_state(task) == DoneState(task);
        passed = passed && model.TaskPayloadAndTailValid(task);

        const U1BuildReport &build = model.build_report(task);
        const uint32_t expected_warp = task % kWarpCount;
        passed = passed && build.task_id == task && build.builder_thread == expected_warp * kWarpSize;
        passed = passed && build.builder_warp == expected_warp && build.builder_lane == 0U;
        passed = passed && build.phase_bits == kExpectedBuildPhaseBits;
        passed = passed && build.payload_lines == PayloadLinesForTask(task);
        passed = passed && build.ubuf_words_written == payload_words && build.gm_words_stored == payload_words;
        passed = passed && build.claim_count == 1U && build.publish_count == 1U;
        passed = passed && build.slot_id == slot && build.slot_generation < kSlotReuseCount;
        if (build.slot_generation < kSlotReuseCount) {
            passed = passed && !generation_seen[slot][build.slot_generation];
            generation_seen[slot][build.slot_generation] = true;
        }
        passed = passed && build.launch_nonce == nonce;
        passed = passed && build.payload_checksum == ExpectedPayloadChecksum(nonce, task);
        ++class_counts[slot][PayloadClassForTask(task)];

        const U1ExecReport &exec = model.exec_report(task);
        passed = passed && exec.task_id == task && exec.executor_owner == kExecutorOwner;
        passed = passed && exec.executor_physical_block == 0U && exec.executor_subblock_id == 1U;
        passed = passed && exec.phase_bits == kExpectedExecPhaseBits;
        passed = passed && exec.payload_lines == PayloadLinesForTask(task) && exec.payload_words_read == payload_words;
        passed = passed && exec.claim_count == 1U && exec.completion_count == 1U && exec.checksum_match_count == 1U;
        passed = passed && exec.launch_nonce == nonce;
        passed = passed && exec.payload_checksum == ExpectedPayloadChecksum(nonce, task);
    }

    for (uint32_t slot = 0U; slot < kUbufSlotCount; ++slot) {
        for (bool seen : generation_seen[slot]) {
            passed = passed && seen;
        }
        for (uint32_t count : class_counts[slot]) {
            passed = passed && count == 8U;
        }
    }

    for (uint32_t warp = 0U; warp < kWarpCount; ++warp) {
        const uint32_t thread = warp * kWarpSize;
        const U1ThreadReport &report = model.builder_thread_report(thread);
        passed = passed && report.thread_id == thread && report.warp_id == warp && report.lane_id == 0U;
        passed = passed && report.active_leader == 1U;
        passed = passed && report.first_task_id == FirstTaskForWarp(warp);
        passed = passed && report.last_task_id == LastTaskForWarp(warp);
        passed = passed && report.task_count == kTasksPerWarp && report.build_count == kTasksPerWarp;
        passed = passed && report.status == kExpectedBuilderThreadStatus && report.slot_attempt_count >= kTasksPerWarp;
        passed = passed && report.launch_nonce == nonce;
        passed = passed && report.checksum == ExpectedThreadChecksum(
                                                  nonce, thread, FirstTaskForWarp(warp), LastTaskForWarp(warp),
                                                  kExpectedBuilderThreadStatus
                                              );
    }
    for (uint32_t thread = 0U; thread < kThreadCount; ++thread) {
        if (thread % kWarpSize != 0U) {
            passed = passed && ObjectIsPoison(model.builder_thread_report(thread));
        }
    }
    return passed;
}

bool RunConcurrentRound(uint32_t round) {
    const uint64_t nonce = UINT64_C(0xC1DE000000000000) ^ static_cast<uint64_t>(round + 1U);
    auto model = std::make_unique<MultiSlotModel>(nonce);
    StartGate gate(kWarpCount + 1U);
    std::array<WarpBuildResult, kWarpCount> results{};
    std::array<uint32_t, kWarpCount> order{};
    std::array<uint32_t, kWarpCount> jitter{};
    for (uint32_t warp = 0U; warp < kWarpCount; ++warp) {
        order[warp] = warp;
    }
    std::mt19937_64 random(UINT64_C(0x553152414E444F4D) ^ nonce);
    std::shuffle(order.begin(), order.end(), random);
    for (uint32_t warp = 0U; warp < kWarpCount; ++warp) {
        jitter[warp] = static_cast<uint32_t>(random() & 7U);
    }

    std::vector<std::thread> builders;
    builders.reserve(kWarpCount);
    for (uint32_t warp : order) {
        builders.emplace_back([&, warp] {
            gate.ArriveAndWait();
            for (uint32_t step = 0U; step < jitter[warp]; ++step) {
                std::this_thread::yield();
            }
            results[warp] = model->BuildWarpLeader(warp * kWarpSize);
        });
    }
    bool executor_result = false;
    std::thread executor([&] {
        gate.ArriveAndWait();
        executor_result = model->ExecuteAll();
    });

    const bool all_arrived = gate.WaitAndOpen();
    for (std::thread &builder : builders) {
        builder.join();
    }
    executor.join();
    if (!all_arrived || !executor_result) {
        return false;
    }
    for (const WarpBuildResult &result : results) {
        if (result.first != BuildOutcome::Built || result.second != BuildOutcome::Built) {
            return false;
        }
    }
    return ValidateConcurrentRound(*model, nonce);
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

    TestAbiAndMappingContract();
    TestInactiveLaneAndWrongWarpRejected();
    TestFourAnchorsHoldFourFullyStagedSlots();
    TestPublicationBoundaryNoEarlyReuseAndRuntimeGeneration();
    TestShortLongAlternationDoesNotOverwriteTails();
    TestPrepublishExceptionReleasesExactGeneration();
    TestGuardCorruptionReleasesExactGeneration();
    TestAnchorTimeoutReleasesAndAdvancesGeneration();
    TestAnchorCrossLineVisibilitySkewRetries();
    TestAnchorHolderCleansUpAfterForeignFatal();
    TestExecutorFailsClosedAndChecksEveryWord();
    for (uint32_t round = 0U; round < rounds; ++round) {
        if (!RunConcurrentRound(round)) {
            std::fprintf(stderr, "[FAIL] U1 CPU 64-warp/128-task stress round=%u\n", round);
            ++g_failures;
            break;
        }
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "[FAIL] U1 CPU multi-slot failures=%d\n", g_failures);
        return EXIT_FAILURE;
    }
    std::printf(
        "[PASS] U1 CPU rounds=%u: 2048 threads/64 lane0 leaders/128 tasks, four guarded slots, "
        "anchor mask=0xf and maxbusy=4, runtime generation sets 0..31, pre-BUILT hidden, exact fault cleanup, "
        "word-wise AIV1 validation, seeded concurrency, mte3=0\n",
        rounds
    );
    return EXIT_SUCCESS;
}
