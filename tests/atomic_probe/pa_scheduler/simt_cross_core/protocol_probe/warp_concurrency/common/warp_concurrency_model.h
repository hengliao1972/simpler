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

#ifndef PA_SCHEDULER_SIMT_CROSS_CORE_WARP_CONCURRENCY_MODEL_H
#define PA_SCHEDULER_SIMT_CROSS_CORE_WARP_CONCURRENCY_MODEL_H

#include <cstdint>

namespace pa_scheduler::simt_cross_core::warp_concurrency {

using LaneMask = uint32_t;

constexpr uint32_t kWarpLaneCount = 32U;
constexpr LaneMask kAllLanesMask = 0xFFFFFFFFU;
constexpr LaneMask kBranchAMask = 0x0000FFFFU;
constexpr LaneMask kBranchBMask = 0xFFFF0000U;
constexpr uint32_t kBranchALeaderLane = 0U;
constexpr uint32_t kBranchBLeaderLane = 16U;
constexpr uint32_t kHandshakeWaitPollBudget = 4U;

constexpr LaneMask LaneBit(uint32_t lane) {
    return lane < kWarpLaneCount ? static_cast<LaneMask>(LaneMask{1U} << lane) : LaneMask{0U};
}

constexpr uint32_t CountActiveLanes(LaneMask mask) {
    uint32_t count = 0U;
    while (mask != 0U) {
        count += mask & 1U;
        mask >>= 1U;
    }
    return count;
}

static_assert((kBranchAMask & kBranchBMask) == 0U, "divergent branch masks must not overlap");
static_assert((kBranchAMask | kBranchBMask) == kAllLanesMask, "divergent branch masks must cover the warp");
static_assert(CountActiveLanes(kBranchAMask) == 16U, "branch A must contain lanes 0..15");
static_assert(CountActiveLanes(kBranchBMask) == 16U, "branch B must contain lanes 16..31");
static_assert((kBranchAMask & LaneBit(kBranchALeaderLane)) != 0U, "branch A leader must be active");
static_assert((kBranchBMask & LaneBit(kBranchBLeaderLane)) != 0U, "branch B leader must be active");

enum class HandshakeRole : uint8_t {
    A,
    B,
};

enum class HandshakeState : uint8_t {
    NotStarted,
    WaitingForPeer,
    Succeeded,
    TimedOut,
};

struct ActorPlacement {
    uint32_t warp_id;
    LaneMask outer_branch_mask;
    LaneMask executing_mask;
    uint32_t leader_lane;
};

enum class DivergentPathOrder : uint8_t {
    AThenB,
    BThenA,
};

struct ActorObservation {
    ActorPlacement placement{};
    HandshakeState final_state = HandshakeState::NotStarted;
    uint32_t first_dispatch_epoch = 0U;
    uint32_t terminal_dispatch_epoch = 0U;
    uint32_t path_dispatches = 0U;
    uint32_t wait_polls = 0U;
    uint64_t outer_branch_lane_steps = 0U;
    uint64_t executing_lane_steps = 0U;
    uint32_t leader_mailbox_stores = 0U;
    uint32_t leader_mailbox_loads = 0U;
    uint32_t non_leader_mailbox_accesses = 0U;
    bool observed_peer = false;
};

struct ForwardProgressReport {
    ActorObservation actor_a{};
    ActorObservation actor_b{};
    bool a_published = false;
    bool b_published = false;
};

namespace detail {

struct HandshakeMailbox {
    bool a_published = false;
    bool b_published = false;
};

class HandshakeActor {
public:
    HandshakeActor(HandshakeRole role, ActorPlacement placement) :
        role_(role),
        placement_(placement) {}

    bool IsTerminal() const { return state_ == HandshakeState::Succeeded || state_ == HandshakeState::TimedOut; }

    void Dispatch(HandshakeMailbox &mailbox, uint32_t dispatch_epoch) {
        if (IsTerminal()) {
            return;
        }

        if (first_dispatch_epoch_ == 0U) {
            first_dispatch_epoch_ = dispatch_epoch;
        }
        ++path_dispatches_;
        outer_branch_lane_steps_ += CountActiveLanes(placement_.outer_branch_mask);
        executing_lane_steps_ += CountActiveLanes(placement_.executing_mask);

        if (state_ == HandshakeState::NotStarted) {
            Publish(mailbox);
            state_ = HandshakeState::WaitingForPeer;
            return;
        }

        ++leader_mailbox_loads_;
        ++wait_polls_;
        if (PeerPublished(mailbox)) {
            observed_peer_ = true;
            state_ = HandshakeState::Succeeded;
            terminal_dispatch_epoch_ = dispatch_epoch;
            return;
        }

        if (wait_polls_ == kHandshakeWaitPollBudget) {
            state_ = HandshakeState::TimedOut;
            terminal_dispatch_epoch_ = dispatch_epoch;
        }
    }

    ActorObservation Observe() const {
        return ActorObservation{
            placement_,
            state_,
            first_dispatch_epoch_,
            terminal_dispatch_epoch_,
            path_dispatches_,
            wait_polls_,
            outer_branch_lane_steps_,
            executing_lane_steps_,
            leader_mailbox_stores_,
            leader_mailbox_loads_,
            0U,
            observed_peer_,
        };
    }

private:
    void Publish(HandshakeMailbox &mailbox) {
        if (role_ == HandshakeRole::A) {
            mailbox.a_published = true;
        } else {
            mailbox.b_published = true;
        }
        ++leader_mailbox_stores_;
    }

    bool PeerPublished(const HandshakeMailbox &mailbox) const {
        return role_ == HandshakeRole::A ? mailbox.b_published : mailbox.a_published;
    }

    HandshakeRole role_;
    ActorPlacement placement_;
    HandshakeState state_ = HandshakeState::NotStarted;
    uint32_t first_dispatch_epoch_ = 0U;
    uint32_t terminal_dispatch_epoch_ = 0U;
    uint32_t path_dispatches_ = 0U;
    uint32_t wait_polls_ = 0U;
    uint64_t outer_branch_lane_steps_ = 0U;
    uint64_t executing_lane_steps_ = 0U;
    uint32_t leader_mailbox_stores_ = 0U;
    uint32_t leader_mailbox_loads_ = 0U;
    bool observed_peer_ = false;
};

inline uint64_t RunSingleElapsed(uint32_t actor_steps) {
    uint64_t elapsed = 0U;
    for (uint32_t step = 0U; step < actor_steps; ++step) {
        ++elapsed;
    }
    return elapsed;
}

inline uint64_t RunSerialElapsed(uint32_t actor_a_steps, uint32_t actor_b_steps) {
    uint64_t elapsed = RunSingleElapsed(actor_a_steps);
    for (uint32_t step = 0U; step < actor_b_steps; ++step) {
        ++elapsed;
    }
    return elapsed;
}

inline uint64_t RunParallelElapsed(uint32_t actor_a_steps, uint32_t actor_b_steps) {
    uint32_t actor_a_completed = 0U;
    uint32_t actor_b_completed = 0U;
    uint64_t elapsed = 0U;
    while (actor_a_completed != actor_a_steps || actor_b_completed != actor_b_steps) {
        if (actor_a_completed != actor_a_steps) {
            ++actor_a_completed;
        }
        if (actor_b_completed != actor_b_steps) {
            ++actor_b_completed;
        }
        ++elapsed;
    }
    return elapsed;
}

}  // namespace detail

inline ForwardProgressReport RunSameWarpForwardProgress(DivergentPathOrder path_order = DivergentPathOrder::AThenB) {
    detail::HandshakeMailbox mailbox{};
    detail::HandshakeActor actor_a{
        HandshakeRole::A,
        ActorPlacement{0U, kBranchAMask, LaneBit(kBranchALeaderLane), kBranchALeaderLane},
    };
    detail::HandshakeActor actor_b{
        HandshakeRole::B,
        ActorPlacement{0U, kBranchBMask, LaneBit(kBranchBLeaderLane), kBranchBLeaderLane},
    };

    uint32_t dispatch_epoch = 0U;
    // A divergent warp cannot switch masks while its current path is blocked;
    // the bounded wait is what eventually lets the warp reach the other path.
    if (path_order == DivergentPathOrder::AThenB) {
        while (!actor_a.IsTerminal()) {
            actor_a.Dispatch(mailbox, ++dispatch_epoch);
        }
        while (!actor_b.IsTerminal()) {
            actor_b.Dispatch(mailbox, ++dispatch_epoch);
        }
    } else {
        while (!actor_b.IsTerminal()) {
            actor_b.Dispatch(mailbox, ++dispatch_epoch);
        }
        while (!actor_a.IsTerminal()) {
            actor_a.Dispatch(mailbox, ++dispatch_epoch);
        }
    }

    return ForwardProgressReport{
        actor_a.Observe(),
        actor_b.Observe(),
        mailbox.a_published,
        mailbox.b_published,
    };
}

inline ForwardProgressReport RunCrossWarpForwardProgress() {
    detail::HandshakeMailbox mailbox{};
    detail::HandshakeActor actor_a{
        HandshakeRole::A,
        ActorPlacement{0U, kAllLanesMask, LaneBit(0U), 0U},
    };
    detail::HandshakeActor actor_b{
        HandshakeRole::B,
        ActorPlacement{1U, kAllLanesMask, LaneBit(0U), 0U},
    };

    uint32_t dispatch_epoch = 0U;
    while (!actor_a.IsTerminal() || !actor_b.IsTerminal()) {
        ++dispatch_epoch;
        actor_a.Dispatch(mailbox, dispatch_epoch);
        actor_b.Dispatch(mailbox, dispatch_epoch);
    }

    return ForwardProgressReport{
        actor_a.Observe(),
        actor_b.Observe(),
        mailbox.a_published,
        mailbox.b_published,
    };
}

struct ElapsedStepOracle {
    uint64_t a_only;
    uint64_t b_only;
    uint64_t same_warp_a_plus_b;
    uint64_t cross_warp_a_plus_b;
};

inline ElapsedStepOracle BuildElapsedStepOracle(uint32_t actor_a_steps, uint32_t actor_b_steps) {
    // These are conceptual elapsed steps, not dispatch events or host wall time.
    return ElapsedStepOracle{
        detail::RunSingleElapsed(actor_a_steps),
        detail::RunSingleElapsed(actor_b_steps),
        detail::RunSerialElapsed(actor_a_steps, actor_b_steps),
        detail::RunParallelElapsed(actor_a_steps, actor_b_steps),
    };
}

}  // namespace pa_scheduler::simt_cross_core::warp_concurrency

#endif  // PA_SCHEDULER_SIMT_CROSS_CORE_WARP_CONCURRENCY_MODEL_H
