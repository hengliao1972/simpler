# A5 block tasks (MIX / 2V): shared control and a path without DCCI

**Date**: 2026-07-14
**Status**: Phase A implemented — `BlockWon`/`WonSlot` removed; winner-gated
SPSC launch (`LaneInbox`) + `DistTaskCell.remaining` live. Only the kernel-arg
`LaunchPayload` still crosses via `dcci`; all control (identity / readiness /
completion) is atomic-only. Phase B (slim launch) remains future work.
**Scope**: `runtime=fully_distributed_within_core`, `platform=a5`
**Authoritative design** (target model, including winner-gated launch):
[fully_distributed_within_core.md][fdwic-spec] §3.1, §5.1, §11.2,
§11.5, §11.7

This note captures how **block-level multi-core tasks** are **implemented
today** on a5 (still `BlockWon` / dual fan-in poll), where `dcci` hurts,
and how to migrate to the spec’s target: SPSC `lane_inbox` +
winner-gated readiness (follower does not collect or poll fan-in).

---

## 1. What a “block task” is

On A5 a hardware **block** is fixed: **1 AIC + 2 AIV** (AIV0, AIV1).
A *block task* here means a **multi-core** submit whose `ActiveMask`
activates two or more core lanes (`popcount(core_mask) ≥ 2`):

| Shape | Lanes | Who claims | Anchor | Follower(s) |
| ----- | ----- | ---------- | ------ | ----------- |
| MIX 1C+1V | AIC, AIV0 | AIC (`cube_cursor`) | AIC | AIV0 |
| MIX 1C+2V | AIC + both AIV | AIC | AIC | AIV0, AIV1 |
| 2V | AIV0, AIV1 | any AIV (`vector_cursor`) | win AIV | sibling |

Single-core 1C / 1V tasks never use the block deposit path: the claim
winner builds only its private `RingSlot` and completes alone.

**Why push instead of “follower waits at N”.** Cube and vector cursors
advance independently. If a follower had to reach submit `N` and then
decide whether *this* block won `N`, a lagging AIC would force AIV to
block (cannot tell “not claimed yet” from “another block won”). The
spec therefore uses **anchor push + async drain** so vector work is not
stalled when cube is behind (§3.1 / §3.2).

---

## 2. Current ownership flow (implementation)

Primary code:

- Claim / publish: `runtime/dist_engine/aicore/submit_runtime.h`
- Drain / execute / complete: `runtime/dist_engine/aicore/submit_core.h`
- Slot population: `runtime/dist_engine/aicore/submit_helpers.h`
- Layout: `runtime/dist_engine/common/state.h`
- A5 cache ops: `runtime/dist_engine/aicore/primitive.h`

Per submit of a multi-core task:

1. Every core replays orch: materialize outputs into its own
   `DistCore` heap/payload, update its private TensorMap.
2. Only the **anchor lane** may `claim` the type cursor
   (`dist_submit_claim_kernel`).
3. On win:
   - Allocate a `WonSlot` in `g_dist.blocks[block_id]` (`alloc_won_slot`).
   - Winner collects fan-in once; `publish_joint_deposits` fills
     `BuiltSubtask` for every *other* active lane (kernel id, tensors,
     scalars, **copied fan-in ids**).
   - Build the anchor’s own private `RingSlot` (`is_multicore=true`,
     `won_block` / `won_slot` back-pointers) **with** `fanin[]`.
4. Followers do **not** claim and do **not** collect fan-in at submit;
   they `drain_block_won()`, copy `BuiltSubtask` (including `fanin[]`)
   into a private `RingSlot`.
5. **Phase B (today):** *both* anchor and follower independently poll
   `task_completed_flag` for each fan-in id before `execute_slot`.
6. After kernel exec, each co-owner
   `decrement_won_remaining_is_last`; the last one publishes
   `DistTaskCell.flag` and advances `frontier`.

Execution always happens from the **private** ring. `block.won` is only
a handover mailbox (§5.1). Dependency *lists* are duplicated into the
follower slot; dependency *polling* is duplicated in Phase B.

---

## 3. Shared data structure (AIC ↔ AIV)

### 3.1 `BlockWon` / `WonSlot`

```text
DistGlobal.blocks[block_id] : BlockWon
  slots[kPrivateSlots] : WonSlot
    state      // Free → Claimed → Published  (atomic, cacheline)
    meta       // task_id                     (cacheline)
    remaining  // joint completion count      (atomic, cacheline)
    drained[3] // per-lane Free/Claimed       (atomic, cacheline each)
    lane[3]    // BuiltSubtask payload        (cacheline-aligned)
  any_pub      // block hint: something published
```

Defined in `state.h`. Capacity today mirrors private slots
(`kPrivateSlots`, typically 4). Doc knobs: `BLOCK_WON_SLOTS` in
[fully_distributed_within_core.md][fdwic-spec] §11.2.

### 3.2 What each field is for

| Field | Role | Writers | Readers |
| ----- | ---- | ------- | ------- |
| `state` | Slot lifecycle | Anchor | Followers / reclaim |
| `meta.task_id` | Identity of deposited joint task | Anchor | Followers |
| `lane[L]` | Deposit payload for lane `L` | Anchor | Lane `L` follower |
| `drained[L]` | Exactly-once drain of lane `L` | Anchor + lane `L` | Lane `L` |
| `remaining` | Joint completion barrier | Anchor + co-owners | Last done |
| `any_pub` | Cheap “any deposit?” hint | Anchor | Followers |

`BuiltSubtask` carries func id, function addr, tensors, scalars,
fan-in ids, and `sub_block_id`. Losers skip fan-in collect at submit
and rely on the deposit for both args and the fan-in list.

### 3.3 Related structures (not block-local control)

- Per-core `DistCore.slots[]` — private execute ring (no AIC↔AIV share).
- Global `cube_cursor` / `vector_cursor` — claim races.
- Global `DistTaskCell` (`flag`, `vend`) — dependency pull + heap
  reclamation.
- Per-core replicated TensorMap / heap — orch run-ahead.

Only `BlockWon` is the **block multiparty control + deposit** object
shared by AIC and its AIV partners.

---

## 4. Where `dcci` is used for this structure

A5 AICore D-cache is not coherent across cores. Atom paths use
`atomicExch` / `atomicAdd` / `atomicMax` (`atomic.h`). Ordinary GM
loads/stores of `meta` / `BuiltSubtask` need explicit maintenance via
`dist_aicore_flush_region` / `dist_aicore_invalidate_region`
(`primitive.h` → `dcci` per 64 B + `dsb`).

**Publish (`publish_joint_deposits`):**

1. Write `meta` and `lane[]`.
2. **Flush** `meta` and all `lane[]` (large region).
3. Atomically set `remaining`, `state=Published`, `any_pub=1`.

**Drain (`drain_block_won` / `has_pending_won`):**

1. Atomically observe `any_pub` / `state` / `drained`.
2. **Invalidate** `lane[my].present`, then full `meta` + `lane[my]`
   before copying into the private ring.

Sim builds omit `dcci` and use CPU fences only.

Cost summary on A5 onboard: flush/invalidate of **full
`BuiltSubtask` copies** dominates vs. the few atomic control words.

---

## 5. Proposal: SPSC launch + winner-gated readiness

### 5.1 Goals

- Delete multiparty `WonSlot` control (`state` / `drained[]` /
  `remaining` / `any_pub`) and the fat cross-core `BuiltSubtask` flush.
- Keep fixed hardware block pairing, async push (no wait-at-N), and a
  single global `flag(N)` when all subtasks complete.
- **Winner owns joint-task dependencies.** Follower neither collects
  fan-in nor polls producer flags for that task.

### 5.2 Design

```text
DistTaskCell[N]  (extend existing global ring)
  flag, vend
  owner_block     // set by claim winner
  remaining       // moved out of WonSlot

Per block, unidirectional SPSC inboxes (single-writer / single-reader):
  AIC  → AIV0_inbox   cells: launch descriptor (no fan-in list)
  AIC  → AIV1_inbox
  AIV* → sibling_inbox   (2V only)
```

**Claim (unchanged).** MIX: AIC on `cube_cursor`. 2V: AIV on
`vector_cursor`.

#### Dependency ownership

| Role | Fan-in collect | Poll `task_completed_flag` | When may execute |
| ---- | -------------- | -------------------------- | ---------------- |
| Winner / anchor | Yes | Yes (own Phase B) | Fan-in ready |
| Follower | No | No | Winner’s ready/launch received |
| Inactive lane / other blocks | No | No | Never for this task |

Ready/launch means **dependencies are satisfied**, **not** that the
winner’s kernel has finished. AIC and AIV still run in parallel after
deps clear.

#### Preferred sequencing

```text
winner:  claim → build RingSlot(with fanin[])
      → Phase B: wait fan-in → release launch into follower SPSC
      → execute own kernel          // may overlap follower
follower: pop launch → RingSlot(fanin_count=0) → invalidate inputs
      → execute
both:     fetch_sub(DistTaskCell.remaining); last sets flag(N)
```

Preferred: **one SPSC message** carrying launch args, published only
when fan-in is ready (avoids a separate payload-then-ready handshake).
Optional variant: deposit args early, set a ready bit later — only if
ready-delay from a busy winner proves costly.

#### Follower orch path

No fan-in collect on lose/follower path. Every core still materializes
and updates TensorMap for address determinism (§9.3); that is unrelated
to fan-in.

#### Launch payload

Carry what the follower needs to run (func id/addr, tensor/scalar args
or thin refs) **without** `fanin[]`. Prefer thin refs + deterministic
heap over full `BuiltSubtask`-sized copies.

#### A5 cache note

Skipping follower flag polls removes N acquires; it does **not** remove
producer **data** visibility. Before kernel entry the follower must
invalidate (or bypass-cache read) input tensor GM using addresses in
the launch descriptor. Ready notify proves “winner observed flags”, not
“follower D-cache is clean”.

#### Completion / back-pressure

`fetch_sub(DistTaskCell[N].remaining)`; last writer sets `flag`.
Inbox-full → anchor skips new joint claims that round.

### 5.3 Effect on `dcci`

| Today | Proposed |
| ----- | -------- |
| Flush/inval full `WonSlot.lane[]` | Flush/inval SPSC launch cell(s) |
| Shared `state` / `drained` / `any_pub` | SPSC head/tail + cell atomics |
| Both cores poll fan-in flags | Winner polls; follower uses ready |
| Complete via `WonSlot.remaining` | Complete via `DistTaskCell.remaining` |
| Follower sees inputs via flag acquire | Follower inval inputs at launch |

### 5.4 What this deliberately does *not* change

- Do **not** reintroduce “follower waits at submit `N` until AIC claims”
  (§3.1 / §3.2).
- Do **not** let vector cores claim MIX (orphans / double-claim).
- Do **not** serialize follower behind winner kernel completion.
- Fixed A5 block pairing; dynamic cross-block MIX stays out of scope
  (§11.7).

### 5.5 Suggested implementation order

1. Winner-only fan-in; launch-on-ready into per-lane SPSC; follower
   slots use `fanin_count=0`.
2. Move `remaining` onto `DistTaskCell`; retarget complete path.
3. Replace `publish_joint_deposits` / `drain_block_won`; delete
   `BlockWon` / `WonSlot` / `any_pub`.
4. Slim launch payload (deterministic heap / thin refs).
5. Validate MIX / 2V on a5 + a5sim; skew (winner busy delaying ready);
   confirm AIC∥AIV overlap after deps clear.

---

## 6. Impact analysis

### 6.1 What improves

| Area | Impact |
| ---- | ------ |
| A5 `dcci` volume | Drop fat `BuiltSubtask` flush/inval (~KiB/lane). |
| Protocol | SPSC instead of multiparty `state`/`drained`/`any_pub`. |
| Dep traffic | One fan-in poll site per joint task (winner only). |
| GM footprint | Delete `BlockWon` array (multi‑MB in dist arena). |
| Drain scan | Precise `head!=tail` vs sticky `any_pub`. |
| Completion | `remaining` beside `flag`/`vend` on `DistTaskCell`. |

### 6.2 Correctness risks (highest first)

**1. Ready-notify latency coupling.**

Today both co-owners poll flags independently. If the winner is busy,
followers may wait even when flags are already true. Mitigate: publish
ready in the same Phase B pass that would execute the anchor slot.
Never tie ready to anchor kernel completion.

**2. Follower input cache maintenance (A5).**

Ready ≠ producer data visibility. Invalidate input tensor lines from
launch args before kernel even with `fanin_count=0`.

**3. Launch payload lifetime.**

SPSC cell must live until follower drains. One launch-on-ready message
is the simplest lifetime story.

**4. `DistTaskCell` ABI / recycle.**

`owner_block` + `remaining` must fit in one cache line with `flag`/
`vend`. Window recycle must clear the new fields.

**5. 2V sibling SPSC.**

Anchor may be AIV0 or AIV1; inbox wiring must be role-symmetric.

### 6.3 Performance tradeoffs

| Change | Likely effect |
| ------ | ------------- |
| Drop fat deposit flush/inval | Large win on a5 onboard MIX/2V. |
| Winner-only fan-in | No loser/follower fan-in orch cost. |
| Ready delayed if winner busy | Follower idle vs dual poll — measure |
| Slim vs fat launch cell | Arg copy/`dcci` scales with payload thinness. |

### 6.4 Code / product surface

Touches a5 FDWIC dist engine (a2a3 twin may need parity):

- `state.h` — delete `WonSlot`/`BlockWon`; extend `DistTaskCell`; add
  SPSC inboxes.
- `submit_{runtime,core,helpers}.h` — claim/build; Phase B publishes
  ready; drain/complete with `fanin_count=0` on followers.
- `control_plane.h`, debug dump, swimlane `DrainWon`.
- Docs: this file + [fully_distributed_within_core.md][fdwic-spec]
  §3.1 / §5.1 / §11.2 when implementing.

Orchestration API (`rt_submit_task` / `MixedKernels`) unchanged.

### 6.5 Test / validation

- `simple_orch_smoke` `mixed=1` / `mixed=2`, a5 + a5sim.
- `submit_dependency_smoke` MIX paths.
- Skew: cube ahead; vector ahead; winner busy while fan-in true.
- Tail drain at `dist_submit_drain_to_completion`.
- Swimlane: AIC∥AIV overlap after deps clear (ready ≠ anchor done).

### 6.6 Landing shape

1. **Phase A:** winner-gated launch-on-ready SPSC + move `remaining`;
   delete multiparty `WonSlot` control (payload may still be compact).
2. **Phase B:** slim launch (thin refs / deterministic heap).
3. **Phase C:** early payload + separate ready bit only if Phase A
   shows ready-delay hurt.

### 6.7 Verdict

Winner-gated readiness is the right dependency model for joint tasks:
followers need not collect or poll fan-in. Gate notify on **dep
satisfaction**, keep A5 input invalidate on follower tensor addresses,
and replace `BlockWon` with SPSC inboxes to remove shared control and
most deposit `dcci`.

---

## 7. References

- Spec: [fully_distributed_within_core.md][fdwic-spec]
- Runtime overview: [`RUNTIME_LOGIC.md`](RUNTIME_LOGIC.md)
- Types: `runtime/dist_engine/common/state.h`
- Submit/drain:
  `runtime/dist_engine/aicore/submit_{runtime,core,helpers}.h`
- Cache primitives: `runtime/dist_engine/aicore/primitive.h`

[fdwic-spec]: ../../../../../docs/fully_distributed_within_core.md
