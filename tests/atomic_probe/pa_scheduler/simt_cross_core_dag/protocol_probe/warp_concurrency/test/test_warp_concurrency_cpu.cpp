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

#include "../common/warp_concurrency_model.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace pa_scheduler::simt_cross_core::warp_concurrency;

int g_failures = 0;

void Check(bool condition, const char *message) {
    if (condition) {
        return;
    }
    std::fprintf(stderr, "[FAIL] warp-concurrency CPU oracle: %s\n", message);
    ++g_failures;
}

void TestDivergentMasksAndLeaders() {
    Check((kBranchAMask & kBranchBMask) == 0U, "same-warp branch masks overlap");
    Check((kBranchAMask | kBranchBMask) == kAllLanesMask, "same-warp branch masks do not cover 32 lanes");
    Check(CountActiveLanes(kBranchAMask) == 16U, "branch A is not lanes 0..15");
    Check(CountActiveLanes(kBranchBMask) == 16U, "branch B is not lanes 16..31");
    Check((kBranchAMask & LaneBit(kBranchALeaderLane)) != 0U, "lane 0 is not active in branch A");
    Check((kBranchBMask & LaneBit(kBranchBLeaderLane)) != 0U, "lane 16 is not active in branch B");
}

void CheckSameWarpSerialForwardProgress(DivergentPathOrder path_order) {
    const ForwardProgressReport report = RunSameWarpForwardProgress(path_order);
    const ActorObservation &actor_a = report.actor_a;
    const ActorObservation &actor_b = report.actor_b;
    const ActorObservation &first = path_order == DivergentPathOrder::AThenB ? actor_a : actor_b;
    const ActorObservation &second = path_order == DivergentPathOrder::AThenB ? actor_b : actor_a;

    Check(actor_a.placement.warp_id == actor_b.placement.warp_id, "divergent actors do not share one warp");
    Check(actor_a.placement.outer_branch_mask == kBranchAMask, "branch A outer mask changed");
    Check(actor_b.placement.outer_branch_mask == kBranchBMask, "branch B outer mask changed");
    Check(actor_a.placement.executing_mask == LaneBit(kBranchALeaderLane), "branch A executing mask is not its leader");
    Check(actor_b.placement.executing_mask == LaneBit(kBranchBLeaderLane), "branch B executing mask is not its leader");
    Check(actor_a.placement.leader_lane == kBranchALeaderLane, "branch A leader is not lane 0");
    Check(actor_b.placement.leader_lane == kBranchBLeaderLane, "branch B leader is not lane 16");

    Check(first.final_state == HandshakeState::TimedOut, "first divergent branch did not time out");
    Check(!first.observed_peer, "first divergent branch observed an unscheduled peer");
    Check(first.wait_polls == kHandshakeWaitPollBudget, "first branch did not exhaust its bounded wait");
    Check(first.path_dispatches == kHandshakeWaitPollBudget + 1U, "first branch dispatch count changed");
    Check(
        second.first_dispatch_epoch == first.terminal_dispatch_epoch + 1U,
        "second divergent mask did not start immediately after the first mask reached a terminal state"
    );
    Check(second.final_state == HandshakeState::Succeeded, "second branch did not observe the first publication");
    Check(second.observed_peer, "second branch success did not record its peer observation");
    Check(second.path_dispatches == 2U, "second branch did not publish and then poll exactly once");
    Check(report.a_published && report.b_published, "same-warp leaders did not publish both handshake flags");

    Check(actor_a.leader_mailbox_stores == 1U, "branch A leader publication count is not one");
    Check(actor_b.leader_mailbox_stores == 1U, "branch B leader publication count is not one");
    Check(first.leader_mailbox_loads == kHandshakeWaitPollBudget, "first branch leader poll count changed");
    Check(second.leader_mailbox_loads == 1U, "second branch leader should succeed on its first poll");
    Check(actor_a.non_leader_mailbox_accesses == 0U, "branch A non-leader accessed handshake state");
    Check(actor_b.non_leader_mailbox_accesses == 0U, "branch B non-leader accessed handshake state");
    Check(
        actor_a.outer_branch_lane_steps ==
            static_cast<uint64_t>(CountActiveLanes(kBranchAMask)) * actor_a.path_dispatches,
        "branch A outer-mask accounting changed"
    );
    Check(
        actor_b.outer_branch_lane_steps ==
            static_cast<uint64_t>(CountActiveLanes(kBranchBMask)) * actor_b.path_dispatches,
        "branch B outer-mask accounting changed"
    );
    Check(actor_a.executing_lane_steps == actor_a.path_dispatches, "branch A executed outside its leader lane");
    Check(actor_b.executing_lane_steps == actor_b.path_dispatches, "branch B executed outside its leader lane");
}

void TestSameWarpSerialForwardProgress() {
    CheckSameWarpSerialForwardProgress(DivergentPathOrder::AThenB);
    CheckSameWarpSerialForwardProgress(DivergentPathOrder::BThenA);
}

void TestCrossWarpIndependentForwardProgress() {
    const ForwardProgressReport report = RunCrossWarpForwardProgress();
    const ActorObservation &actor_a = report.actor_a;
    const ActorObservation &actor_b = report.actor_b;

    Check(actor_a.placement.warp_id != actor_b.placement.warp_id, "cross-warp actors share a warp id");
    Check(actor_a.placement.outer_branch_mask == kAllLanesMask, "warp 0 outer branch is not warp-wide");
    Check(actor_b.placement.outer_branch_mask == kAllLanesMask, "warp 1 outer branch is not warp-wide");
    Check(actor_a.placement.executing_mask == LaneBit(0U), "warp 0 executing mask is not lane 0");
    Check(actor_b.placement.executing_mask == LaneBit(0U), "warp 1 executing mask is not lane 0");
    Check(actor_a.placement.leader_lane == 0U, "warp 0 leader is not lane 0");
    Check(actor_b.placement.leader_lane == 0U, "warp 1 leader is not lane 0");

    Check(
        actor_a.first_dispatch_epoch == 1U && actor_b.first_dispatch_epoch == 1U,
        "independent warps did not start in the first dispatch epoch"
    );
    Check(actor_a.final_state == HandshakeState::Succeeded, "warp 0 handshake did not succeed");
    Check(actor_b.final_state == HandshakeState::Succeeded, "warp 1 handshake did not succeed");
    Check(actor_a.observed_peer && actor_b.observed_peer, "cross-warp handshake was not bidirectional");
    Check(
        actor_a.terminal_dispatch_epoch == 2U && actor_b.terminal_dispatch_epoch == 2U,
        "warps did not complete in the second dispatch epoch"
    );
    Check(actor_a.wait_polls == 1U && actor_b.wait_polls == 1U, "cross-warp actors missed peer publication");
    Check(actor_a.path_dispatches == 2U && actor_b.path_dispatches == 2U, "cross-warp dispatch count changed");
    Check(report.a_published && report.b_published, "cross-warp leaders did not publish both handshake flags");

    Check(actor_a.leader_mailbox_stores == 1U, "warp 0 leader publication count is not one");
    Check(actor_b.leader_mailbox_stores == 1U, "warp 1 leader publication count is not one");
    Check(actor_a.leader_mailbox_loads == 1U, "warp 0 leader poll count is not one");
    Check(actor_b.leader_mailbox_loads == 1U, "warp 1 leader poll count is not one");
    Check(actor_a.non_leader_mailbox_accesses == 0U, "warp 0 non-leader accessed handshake state");
    Check(actor_b.non_leader_mailbox_accesses == 0U, "warp 1 non-leader accessed handshake state");
    Check(
        actor_a.outer_branch_lane_steps == static_cast<uint64_t>(kWarpLaneCount) * actor_a.path_dispatches,
        "warp 0 outer-mask accounting changed"
    );
    Check(
        actor_b.outer_branch_lane_steps == static_cast<uint64_t>(kWarpLaneCount) * actor_b.path_dispatches,
        "warp 1 outer-mask accounting changed"
    );
    Check(actor_a.executing_lane_steps == actor_a.path_dispatches, "warp 0 executed outside its leader lane");
    Check(actor_b.executing_lane_steps == actor_b.path_dispatches, "warp 1 executed outside its leader lane");
}

void TestDeterministicElapsedStepOracle() {
    struct StepCase {
        uint32_t actor_a_steps;
        uint32_t actor_b_steps;
    };
    constexpr std::array<StepCase, 5U> kCases{{
        {7U, 11U},
        {1U, 1U},
        {13U, 3U},
        {0U, 9U},
        {8U, 0U},
    }};

    for (const StepCase &step_case : kCases) {
        const ElapsedStepOracle oracle = BuildElapsedStepOracle(step_case.actor_a_steps, step_case.actor_b_steps);
        Check(oracle.a_only == step_case.actor_a_steps, "A-only elapsed steps changed");
        Check(oracle.b_only == step_case.actor_b_steps, "B-only elapsed steps changed");
        Check(
            oracle.same_warp_a_plus_b == oracle.a_only + oracle.b_only,
            "same-warp conceptual elapsed steps are not A-only + B-only"
        );
        Check(
            oracle.cross_warp_a_plus_b == std::max(oracle.a_only, oracle.b_only),
            "cross-warp conceptual elapsed steps are not max(A-only, B-only)"
        );
    }

    const ElapsedStepOracle primary = BuildElapsedStepOracle(7U, 11U);
    Check(primary.same_warp_a_plus_b == 18U, "primary same-warp elapsed oracle is not 18 steps");
    Check(primary.cross_warp_a_plus_b == 11U, "primary cross-warp elapsed oracle is not 11 steps");
    Check(primary.same_warp_a_plus_b > primary.cross_warp_a_plus_b, "primary oracle does not distinguish layouts");
}

}  // namespace

int main() {
    TestDivergentMasksAndLeaders();
    TestSameWarpSerialForwardProgress();
    TestCrossWarpIndependentForwardProgress();
    TestDeterministicElapsedStepOracle();

    if (g_failures != 0) {
        std::fprintf(stderr, "[FAIL] warp-concurrency CPU oracle failures=%d\n", g_failures);
        return EXIT_FAILURE;
    }

    const ElapsedStepOracle oracle = BuildElapsedStepOracle(7U, 11U);
    std::printf(
        "[PASS] warp-concurrency CPU oracle: either same-warp first branch times out; cross-warp bidirectional "
        "success; "
        "steps A=%llu B=%llu same=%llu cross=%llu\n",
        static_cast<unsigned long long>(oracle.a_only), static_cast<unsigned long long>(oracle.b_only),
        static_cast<unsigned long long>(oracle.same_warp_a_plus_b),
        static_cast<unsigned long long>(oracle.cross_warp_a_plus_b)
    );
    return EXIT_SUCCESS;
}
