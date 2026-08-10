# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Strict phase-one PA semantics for the production shared TensorMap schema-v5."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

try:
    from .fdwic_swimlane_schema import (
        KERNEL_EXECUTION_CHILD_PHASES,
        OVERLAY_PHASES,
        CorePartition,
        Event,
        FdwicV4Model,
        SubmitPartition,
        find_containing_event,
        validate_and_partition_v4,
    )
except ImportError:
    from fdwic_swimlane_schema import (  # type: ignore[no-redef]
        KERNEL_EXECUTION_CHILD_PHASES,
        OVERLAY_PHASES,
        CorePartition,
        Event,
        FdwicV4Model,
        SubmitPartition,
        find_containing_event,
        validate_and_partition_v4,
    )

SHARED_V5_PHASES = frozenset(
    {
        "SharedRegisterPublishMetadata",
        "SharedMaterializePublishTaskOutputs",
        "SharedMaterializePublishTaskOutputsCopy",
        "SharedMaterializePublishTaskOutputsFlush",
        "SharedRegisterWaitInsertTurnBypassLoad",
        "Dcci",
    }
)
SHARED_V5_OVERLAY_PHASES = (*OVERLAY_PHASES, "Dcci")
SHARED_V5_FORBIDDEN_PHASES = frozenset(
    {"Alloc", "Build", "Replay", "DrainWon", "EfDrain", "PrepareMap", "LoserReplay"}
)
SHARED_V5_PHASE1_TASK_COUNT = 1280

SHARED_V5_ATOMIC_SITE_NAMES = (
    "StartupIncrement",
    "StartupPoll",
    "FatalPoll",
    "FatalSet",
    "ClaimMax",
    "FaninFlagLoad",
    "CompletionVendExchange",
    "CompletionFlagExchange",
    "FrontierInitialLoad",
    "FrontierFlagLoad",
    "FrontierMax",
    "HeapFrontierLoad",
    "HeapVendLoad",
    "ReplayDoneIncrement",
    "ReplayDonePoll",
    "SharedHeapVendLoad",
    "SharedHeapCursorLoad",
    "SharedHeapCursorReserve",
    "SharedHeapVendAdvance",
    "SharedInsertPredecessorPoll",
    "SharedInsertCompletionPublish",
    "SharedWinnerFatalGuardLoad",
    "SharedMetadataFatalGuardLoad",
    "SharedFaninOutputPublishedLoad",
    "SharedMetadataOutputPublishedLoad",
    "SharedFaninLastWriterLoad",
    "SharedMetadataLastWriterLoad",
    "SharedMetadataLastWriterCommit",
    "SharedOutputWriterReserve",
    "SharedOutputPublishedExchange",
    "SharedMapLookupHeadLoad",
    "SharedMapLookupTailLoad",
    "SharedMapLookupSeqLoad",
    "SharedMapAppendHeadLoad",
    "SharedMapAppendTailLoad",
    "SharedMapAppendSeqLoad",
    "SharedMapAppendSeqResetExchange",
    "SharedMapAppendSeqPublishExchange",
    "SharedMapAppendTailExchange",
    "SharedOutputRollbackExchange",
    "SharedClaimTournamentLocal",
    "SharedClaimTournamentRoot",
    "CrossCoreBuildTournamentLocal",
    "CrossCoreBuildTournamentRoot",
    "CrossCoreExecuteTournamentLocal",
    "CrossCoreExecuteTournamentRoot",
)
SHARED_V5_ATOMIC_OP_NAMES = ("Load", "Exchange", "FetchAdd", "FetchMax", "CompareExchange")
SHARED_V5_ATOMIC_SITE_OP_IDS = {
    0: 2,
    1: 0,
    2: 0,
    3: 1,
    4: 3,
    5: 0,
    6: 1,
    7: 1,
    8: 0,
    9: 0,
    10: 3,
    11: 0,
    12: 0,
    13: 2,
    14: 0,
    15: 0,
    16: 0,
    17: 2,
    18: 2,
    19: 0,
    20: 4,
    21: 0,
    22: 0,
    23: 0,
    24: 0,
    25: 0,
    26: 0,
    27: 4,
    28: 3,
    29: 1,
    30: 0,
    31: 0,
    32: 0,
    33: 0,
    34: 0,
    35: 0,
    36: 1,
    37: 1,
    38: 1,
    39: 1,
    40: 4,
    41: 4,
    42: 4,
    43: 4,
    44: 4,
    45: 4,
}
SHARED_V5_ATOMIC_RESULT_UNUSED_SITE_IDS = frozenset({0, 3, 6, 7, 13, 39})
SHARED_V5_POLL_BATCH_SITE_OP_IDS = {1: 0, 2: 0, 5: 0, 11: 0, 12: 0, 14: 0, 19: 0}

SHARED_V5_DCCI_SITE_NAMES = (
    "SharedFaninHistoryInvalidate",
    "SharedWriterHistoryFlush",
    "SharedOutputRollbackFlush",
    "SharedOutputDescriptorFlush",
    "SharedRegionReadInvalidate",
    "SharedRegionAppendInvalidate",
    "SharedRegionAppendFlush",
    "SharedWinnerBuildDescriptorInvalidate",
    "ObserverTraceExport",
    "StartupConfigInvalidate",
)
SHARED_V5_DCCI_OP_NAMES = ("Invalidate", "CleanOut")
SHARED_V5_DCCI_SITE_OP_IDS = {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 0, 8: 1, 9: 0}
SHARED_V5_WINNER_DCCI_MATRIX = {
    0: ((3, 6),),
    1: ((3, 2),),
    2: ((3, 6), (7, 2)),
    3: ((3, 2), (7, 2)),
    4: (
        (1, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        (7, 2),
        (7, 2),
        (7, 2),
        (7, 2),
        (7, 2),
        (7, 2),
    ),
}


def _contains(parent: Event, child: Event) -> bool:
    return parent.start_cycle <= child.start_cycle and child.end_cycle <= parent.end_cycle


def _overlaps(left: Event, right: Event) -> bool:
    return max(left.start_cycle, right.start_cycle) < min(left.end_cycle, right.end_cycle)


def _expected_layout(core_types: Sequence[str]) -> tuple[tuple[int, int, str], ...]:
    layout: list[tuple[int, int, str] | None] = [None] * len(core_types)
    aic_count = 0
    for core_id, role in enumerate(core_types):
        if role == "aic":
            layout[core_id] = (aic_count, 0, role)
            aic_count += 1
        elif role != "aiv":
            raise ValueError(f"metadata.core_types[{core_id}] has invalid role {role!r}")
    if aic_count == 0:
        raise ValueError("shared schema-v5 topology has no AIC core")
    aiv_ordinal = 0
    for core_id, role in enumerate(core_types):
        if role != "aiv":
            continue
        block_id, sub_lane = divmod(aiv_ordinal, 2)
        if block_id >= aic_count:
            raise ValueError(f"shared schema-v5 AIV core {core_id} has no matching AIC block")
        layout[core_id] = (block_id, sub_lane + 1, role)
        aiv_ordinal += 1
    if aiv_ordinal != 2 * aic_count:
        raise ValueError(
            "shared schema-v5 topology requires two AIV cores per AIC: "
            f"aic={aic_count} aiv={aiv_ordinal}"
        )
    return tuple(item for item in layout if item is not None)


def _task_function_id(task_id: int) -> int:
    kind = task_id % 5
    return -1 if kind == 0 else kind - 1


def _claim_attempted(role: str, block_id: int, task_id: int) -> bool:
    kind = task_id % 5
    if kind == 0:
        return True
    return kind in ({1, 3} if role == "aic" else {2, 4})


def _claim_tournament_groups(task_id: int, aic_count: int) -> int:
    kind = task_id % 5
    if kind == 0:
        return min(8, 3 * aic_count)
    return min(6, aic_count) if kind in (1, 3) else min(8, 2 * aic_count)


def _claim_tournament_group(core_id: int, aic_count: int, task_id: int) -> int:
    kind = task_id % 5
    candidate_rank = core_id if kind in (0, 1, 3) else core_id - aic_count
    return candidate_rank % _claim_tournament_groups(task_id, aic_count)


def _one(events: dict[tuple[int, int], list[Event]], key: tuple[int, int], phase: str, expected: int) -> Event | None:
    rows = events.get(key, [])
    if len(rows) != expected:
        raise ValueError(f"shared schema-v5 {key} requires {expected} {phase} row(s), got {len(rows)}")
    return rows[0] if rows else None


def validate_and_partition_v5(  # noqa: PLR0912, PLR0915
    fdwic_events: Sequence[dict[str, Any]],
    num_cores: int,
    core_types: Sequence[str],
    level: int,
    scheduler_mode: str | None = None,
) -> FdwicV4Model:
    """Validate the shared PA schema-v5 hierarchy and build an exclusive model.

    Phase one intentionally supports the single-group, 1280-task PA stream
    only. Unsupported task plans fail closed instead of being interpreted as
    the fixed ``Alloc,QK,SF,PV,UP`` sequence.
    """

    if scheduler_mode is None:
        scheduler_mode = "same_core" if any(str(row.get("phase")) == "Submit" for row in fdwic_events) else "cross_core"
    valid_scheduler_modes = {
        "same_core",
        "cross_core_ordinary",
        "cross_core_dag",
        "simt_cross_core_ordinary",
        "simt_cross_core_dag",
    }
    if scheduler_mode not in valid_scheduler_modes and scheduler_mode != "cross_core":
        raise ValueError(f"shared schema-v5 has invalid scheduler mode {scheduler_mode!r}")
    if scheduler_mode != "same_core":
        # Cross-core Build/Execute 已与逐核 Submit replay 解耦。它的完整观察
        # 合同只有相邻的 OrchestrationReplay/FinalDrain 父区间以及真实
        # Kernel/Atomic/DCCI 行；复用 v4 的通用父区间模型，不伪造 1280 个
        # same-core Submit/Claim。
        return validate_and_partition_v4(fdwic_events, num_cores, core_types)

    if level not in (1, 4):
        raise ValueError(f"shared schema-v5 supports only level 1 or 4, got {level}")
    if num_cores <= 0 or len(core_types) != num_cores:
        raise ValueError(
            "shared schema-v5 requires metadata.num_cores to match core_types: "
            f"num_cores={num_cores} core_types={len(core_types)}"
        )
    layout = _expected_layout(core_types)
    by_phase: dict[str, dict[tuple[int, int], list[Event]]] = defaultdict(lambda: defaultdict(list))
    parents: dict[int, dict[str, list[Event]]] = {
        core_id: {"OrchestrationReplay": [], "FinalDrain": []} for core_id in range(num_cores)
    }
    submits_by_core: dict[int, list[Event]] = defaultdict(list)
    kernels: list[Event] = []
    kernels_by_task: dict[int, list[Event]] = defaultdict(list)
    polls_by_core: dict[int, list[Event]] = defaultdict(list)
    handoffs: dict[tuple[int, int], list[Event]] = defaultdict(list)
    claim_tournament_locals: dict[tuple[int, int], list[Event]] = defaultdict(list)
    claim_tournament_roots: dict[tuple[int, int], list[Event]] = defaultdict(list)
    business_dcci: dict[tuple[int, int], list[Event]] = defaultdict(list)
    overlay_statistics = {
        phase: {"event_count": 0, "aggregate_duration_cycles": 0}
        for phase in SHARED_V5_OVERLAY_PHASES
    }

    for row_index, raw in enumerate(fdwic_events):
        event = Event.from_mapping(row_index, raw)
        if not 0 <= event.core_id < num_cores:
            raise ValueError(f"fdwic event {row_index} has invalid core_id {event.core_id}")
        block_id, lane, _role = layout[event.core_id]
        if event.block_id != block_id or event.lane != lane:
            raise ValueError(
                f"fdwic event {row_index} has invalid physical identity for core {event.core_id}: "
                f"{event.block_id}/{event.lane} expected={block_id}/{lane}"
            )
        if event.phase in SHARED_V5_FORBIDDEN_PHASES:
            raise ValueError(f"shared schema-v5 forbids phase {event.phase!r}")
        if event.phase in overlay_statistics:
            overlay_statistics[event.phase]["event_count"] += 1
            overlay_statistics[event.phase]["aggregate_duration_cycles"] += event.duration
        if event.phase in ("OrchestrationReplay", "FinalDrain"):
            parents[event.core_id][event.phase].append(event)
        elif event.phase == "Submit":
            submits_by_core[event.core_id].append(event)
        elif event.phase == "Kernel":
            kernels.append(event)
            kernels_by_task[event.task_id].append(event)
        elif event.phase == "Atomic":
            if event.auxiliary == 40:
                claim_tournament_locals[(event.core_id, event.task_id)].append(event)
            elif event.auxiliary == 41:
                claim_tournament_roots[(event.core_id, event.task_id)].append(event)
            elif event.auxiliary == 19:
                polls_by_core[event.core_id].append(event)
            elif event.auxiliary == 20:
                handoffs[(event.core_id, event.task_id)].append(event)
        elif event.phase == "Dcci":
            if event.task_id >= 0:
                business_dcci[(event.core_id, event.task_id)].append(event)
        elif event.phase not in {"ClockBaseline", "Dcci", "Commit", "RingBp"}:
            by_phase[event.phase][(event.core_id, event.task_id)].append(event)

    core_partitions: list[CorePartition] = []
    valid_task_keys: set[tuple[int, int]] = set()
    for core_id in range(num_cores):
        block_id, lane, role = layout[core_id]
        orchestration_rows = parents[core_id]["OrchestrationReplay"]
        final_rows = parents[core_id]["FinalDrain"]
        if len(orchestration_rows) != 1 or len(final_rows) != 1:
            raise ValueError(
                f"core {core_id} requires exactly one OrchestrationReplay and FinalDrain: "
                f"{len(orchestration_rows)}/{len(final_rows)}"
            )
        orchestration = orchestration_rows[0]
        final_drain = final_rows[0]
        if orchestration.end_cycle != final_drain.start_cycle:
            raise ValueError(f"core {core_id} OrchestrationReplay.end must equal FinalDrain.start")
        submits = sorted(
            submits_by_core.get(core_id, []),
            key=lambda event: (event.start_cycle, event.end_cycle, event.row_index),
        )
        task_ids = tuple(event.task_id for event in submits)
        if task_ids != tuple(range(SHARED_V5_PHASE1_TASK_COUNT)):
            raise ValueError(
                f"core {core_id} shared phase-one Submit IDs must be 0.."
                f"{SHARED_V5_PHASE1_TASK_COUNT - 1}; got count={len(task_ids)}"
            )
        for left, right in zip(submits, submits[1:]):
            if _overlaps(left, right):
                raise ValueError(f"core {core_id} has overlapping Submit rows {left.row_index}/{right.row_index}")

        partitions: list[SubmitPartition] = []
        for submit in submits:
            key = (core_id, submit.task_id)
            valid_task_keys.add(key)
            kind = submit.task_id % 5
            is_alloc = kind == 0
            claim = _one(by_phase["Claim"], key, "Claim", 1)
            assert claim is not None
            winner = bool(claim.flags & 1)
            attempted = _claim_attempted(role, block_id, submit.task_id)
            expected_func = _task_function_id(submit.task_id) if winner else -1
            if (
                submit.flags != (1 if winner else 0)
                or submit.auxiliary != (1 if is_alloc else 0)
                or submit.function_id != expected_func
                or claim.flags != ((1 if winner else 0) | (2 if attempted else 0))
                or claim.auxiliary != (1 if is_alloc else 0)
                or claim.function_id != expected_func
                or not _contains(submit, claim)
            ):
                raise ValueError(f"core {core_id} task {submit.task_id} has inconsistent Submit/Claim semantics")
            if not _contains(orchestration, submit) or submit.duration <= 0:
                raise ValueError(f"core {core_id} task {submit.task_id} Submit is outside OrchestrationReplay")
            if level == 4:
                task_locals = claim_tournament_locals.get(key, [])
                expected_locals = 1 if attempted else 0
                if len(task_locals) != expected_locals:
                    raise ValueError(
                        f"core {core_id} task {submit.task_id} requires {expected_locals} "
                        f"Claim tournament local row(s), got {len(task_locals)}"
                    )
                if any(not _contains(claim, row) for row in task_locals):
                    raise ValueError(
                        f"core {core_id} task {submit.task_id} Claim tournament local must be contained by Claim"
                    )
                task_roots = claim_tournament_roots.get(key, [])
                if len(task_roots) > 1 or (task_roots and not task_locals) or (
                    winner and len(task_roots) != 1
                ):
                    raise ValueError(
                        f"core {core_id} task {submit.task_id} has invalid Claim tournament root count "
                        f"{len(task_roots)} for winner={winner}"
                    )
                if any(not _contains(claim, row) for row in task_roots):
                    raise ValueError(
                        f"core {core_id} task {submit.task_id} Claim tournament root must be contained by Claim"
                    )

            expected = 1 if winner else 0
            materialize = _one(by_phase["Materialize"], key, "Materialize", expected)
            register = _one(by_phase["Register"], key, "Register", expected)
            wait_insert_turn = _one(
                by_phase["SharedRegisterWaitInsertTurnBypassLoad"],
                key,
                "bypass-load insert-turn wait detail",
                expected,
            )
            metadata = _one(by_phase["SharedRegisterPublishMetadata"], key, "metadata detail", expected)
            output = _one(by_phase["SharedMaterializePublishTaskOutputs"], key, "task-output detail", expected)
            copy = _one(by_phase["SharedMaterializePublishTaskOutputsCopy"], key, "task-output copy detail", expected)
            flush = _one(
                by_phase["SharedMaterializePublishTaskOutputsFlush"],
                key,
                "task-output flush detail",
                expected,
            )
            fanin = _one(by_phase["Fanin"], key, "Fanin", expected if not is_alloc else 0)
            tail_phase = "AllocComplete" if is_alloc else "WinnerBuild"
            tail = _one(by_phase[tail_phase], key, tail_phase, expected)
            other_tail = "WinnerBuild" if is_alloc else "AllocComplete"
            _one(by_phase[other_tail], key, other_tail, 0)

            children: list[Event] = [
                Event(
                    row_index=-(1 + core_id * SHARED_V5_PHASE1_TASK_COUNT + submit.task_id),
                    core_id=core_id,
                    block_id=block_id,
                    lane=lane,
                    task_id=submit.task_id,
                    function_id=expected_func,
                    phase="EfDrain",
                    start_cycle=submit.start_cycle,
                    end_cycle=claim.start_cycle,
                    flags=0,
                    auxiliary=0,
                ),
                claim,
            ]
            if winner:
                assert materialize is not None
                assert register is not None
                assert wait_insert_turn is not None
                assert metadata is not None
                assert output is not None
                assert copy is not None
                assert flush is not None
                assert tail is not None
                business_rows = (materialize, register, wait_insert_turn, metadata, output, copy, flush, tail)
                if fanin is not None:
                    business_rows += (fanin,)
                if any(row.function_id != expected_func for row in business_rows):
                    raise ValueError(f"core {core_id} task {submit.task_id} winner function IDs disagree")
                if not (
                    _contains(submit, materialize)
                    and _contains(materialize, output)
                    and _contains(output, copy)
                    and _contains(output, flush)
                    and copy.end_cycle == flush.start_cycle
                    and _contains(submit, register)
                    and _contains(register, wait_insert_turn)
                    and _contains(register, metadata)
                    and wait_insert_turn.start_cycle == register.start_cycle
                    and wait_insert_turn.end_cycle == metadata.start_cycle
                    and (
                        (submit.task_id == 0 and wait_insert_turn.auxiliary == 0)
                        or (submit.task_id != 0 and wait_insert_turn.auxiliary > 0)
                    )
                    and claim.end_cycle <= materialize.start_cycle
                    and materialize.end_cycle == register.start_cycle
                    and tail.end_cycle <= submit.end_cycle
                ):
                    raise ValueError(f"core {core_id} task {submit.task_id} has invalid shared detail nesting")
                if is_alloc:
                    if register.end_cycle != tail.start_cycle:
                        raise ValueError(f"core {core_id} alloc task {submit.task_id} has invalid winner tail order")
                else:
                    assert fanin is not None
                    if register.end_cycle != fanin.start_cycle or fanin.end_cycle != tail.start_cycle:
                        raise ValueError(f"core {core_id} task {submit.task_id} has invalid Fanin/tail order")
                children.extend((materialize, register))
                if fanin is not None:
                    children.append(fanin)
                children.append(tail)

                if level == 4:
                    task_dcci = business_dcci.get(key, [])
                    observed_dcci = Counter(
                        (row.auxiliary, row.flags >> 8) for row in task_dcci
                    )
                    expected_dcci = Counter(SHARED_V5_WINNER_DCCI_MATRIX[kind])
                    if observed_dcci != expected_dcci:
                        raise ValueError(
                            f"core {core_id} task {submit.task_id} has invalid winner Dcci matrix: "
                            f"expected={dict(expected_dcci)} observed={dict(observed_dcci)}"
                        )
                    dcci_containers = {
                        0: fanin,
                        1: materialize,
                        3: flush,
                        7: tail,
                    }
                    for row in task_dcci:
                        container = dcci_containers.get(row.auxiliary)
                        if row.function_id != -1 or not _contains(submit, row) or (
                            container is None or not _contains(container, row)
                        ):
                            raise ValueError(
                                f"core {core_id} task {submit.task_id} Dcci site "
                                f"{row.auxiliary} is outside its business phase"
                            )
                    task_handoffs = handoffs.get(key, [])
                    if len(task_handoffs) != 1 or not (
                        metadata.end_cycle <= task_handoffs[0].start_cycle
                        and task_handoffs[0].end_cycle <= register.end_cycle
                    ):
                        raise ValueError(f"core {core_id} task {submit.task_id} has invalid insert-turn handoff")
            elif level == 4 and business_dcci.get(key):
                raise ValueError(
                    f"core {core_id} task {submit.task_id} loser must have zero business Dcci rows"
                )

            ordered = sorted(children, key=lambda event: (event.start_cycle, event.end_cycle, event.row_index))
            for left, right in zip(ordered, ordered[1:]):
                if _overlaps(left, right):
                    raise ValueError(
                        f"core {core_id} task {submit.task_id} has overlapping exclusive children "
                        f"{left.phase}/{right.phase}"
                    )
            partitions.append(SubmitPartition(submit=submit, children=tuple(ordered)))

        core_partitions.append(
            CorePartition(
                core_id=core_id,
                block_id=block_id,
                lane=lane,
                role=role,
                orchestration=orchestration,
                final_drain=final_drain,
                submits=tuple(partitions),
            )
        )

    invalid_kernel_task_ids = sorted(
        task_id
        for task_id in kernels_by_task
        if not 0 <= task_id < SHARED_V5_PHASE1_TASK_COUNT
    )
    if invalid_kernel_task_ids:
        raise ValueError(
            "shared schema-v5 Kernel rows have invalid task IDs: "
            f"{invalid_kernel_task_ids[:8]}"
        )

    aic_count = sum(role == "aic" for role in core_types)
    root_groups_by_task: list[Counter[int]] = [
        Counter() for _ in range(SHARED_V5_PHASE1_TASK_COUNT)
    ]
    if level == 4:
        for (root_core, root_task), rows in claim_tournament_roots.items():
            if 0 <= root_task < SHARED_V5_PHASE1_TASK_COUNT:
                root_groups_by_task[root_task][
                    _claim_tournament_group(root_core, aic_count, root_task)
                ] += len(rows)

    for task_id in range(SHARED_V5_PHASE1_TASK_COUNT):
        winner_cores = [
            core
            for core in core_partitions
            if core.submits[task_id].submit.flags & 1
        ]
        if len(winner_cores) != 1:
            raise ValueError(
                f"shared schema-v5 task {task_id} requires exactly one global winner, got {len(winner_cores)}"
            )
        if level == 4:
            groups = _claim_tournament_groups(task_id, aic_count)
            root_groups = root_groups_by_task[task_id]
            expected_groups = Counter({group: 1 for group in range(groups)})
            if root_groups != expected_groups:
                raise ValueError(
                    f"shared schema-v5 task {task_id} has invalid Claim tournament root groups: "
                    f"expected={dict(expected_groups)} observed={dict(root_groups)}"
                )
        task_kernels = kernels_by_task.get(task_id, [])
        if task_id % 5 == 0:
            if task_kernels:
                raise ValueError(
                    f"shared schema-v5 alloc task {task_id} requires zero Kernel rows, got {len(task_kernels)}"
                )
            continue
        if len(task_kernels) != 1:
            raise ValueError(
                f"shared schema-v5 task {task_id} requires exactly one Kernel row, got {len(task_kernels)}"
            )
        kernel = task_kernels[0]
        winner_core = winner_cores[0]
        if kernel.core_id != winner_core.core_id:
            raise ValueError(
                f"shared schema-v5 task {task_id} Kernel must run on winner core "
                f"{winner_core.core_id}, got core {kernel.core_id}"
            )
        expected_func = _task_function_id(task_id)
        if kernel.function_id != expected_func:
            raise ValueError(
                f"shared schema-v5 task {task_id} Kernel func_id must be "
                f"{expected_func}, got {kernel.function_id}"
            )

    for phase, keyed in by_phase.items():
        orphaned = set(keyed) - valid_task_keys
        if orphaned:
            raise ValueError(f"shared schema-v5 {phase} rows have no matching Submit: {sorted(orphaned)[:8]}")
    if level == 4:
        orphaned_tournament = (
            set(claim_tournament_locals) | set(claim_tournament_roots)
        ) - valid_task_keys
        if orphaned_tournament:
            raise ValueError(
                "shared schema-v5 Claim tournament rows have no matching Submit: "
                f"{sorted(orphaned_tournament)[:8]}"
            )
        orphaned_business_dcci = set(business_dcci) - valid_task_keys
        if orphaned_business_dcci:
            raise ValueError(
                "shared schema-v5 business Dcci rows have no matching Submit: "
                f"{sorted(orphaned_business_dcci)[:8]}"
            )
        expected_handoff_total = sum(
            1 for core in core_partitions for partition in core.submits if partition.submit.flags & 1
        )
        if sum(map(len, polls_by_core.values())) != 0:
            raise ValueError("shared schema-v5 bypass-load insert turn forbids atomic PollBatch rows")
        if sum(map(len, handoffs.values())) != expected_handoff_total:
            raise ValueError("shared schema-v5 has orphan or duplicate insert-turn handoff rows")

    children_by_core = {
        core.core_id: sorted(
            [child for partition in core.submits for child in partition.children],
            key=lambda event: (event.start_cycle, event.end_cycle, event.row_index),
        )
        for core in core_partitions
    }
    submits_for_core = {core.core_id: [partition.submit for partition in core.submits] for core in core_partitions}
    for kernel in kernels:
        core = core_partitions[kernel.core_id]
        child_rows = children_by_core[kernel.core_id]
        child = find_containing_event(
            kernel,
            child_rows,
            [row.start_cycle for row in child_rows],
            "exclusive child",
        )
        submit_rows = submits_for_core[kernel.core_id]
        submit = find_containing_event(
            kernel,
            submit_rows,
            [row.start_cycle for row in submit_rows],
            "Submit",
        )
        if _contains(core.orchestration, kernel):
            if child is not None and child.phase not in KERNEL_EXECUTION_CHILD_PHASES:
                raise ValueError(f"row {kernel.row_index} Kernel has unsupported container {child.phase}")
            if submit is not None and child is None:
                raise ValueError(f"row {kernel.row_index} Kernel is inside Submit residual")
        elif not _contains(core.final_drain, kernel):
            raise ValueError(f"row {kernel.row_index} Kernel is outside both top-level parents")

    return FdwicV4Model(
        cores=tuple(core_partitions),
        kernels=tuple(kernels),
        overlay_statistics=overlay_statistics,
        event_count=len(fdwic_events),
        task_ids=tuple(range(SHARED_V5_PHASE1_TASK_COUNT)),
    )
