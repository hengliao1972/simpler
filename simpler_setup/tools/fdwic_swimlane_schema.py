# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared schema-v4 semantics for production FDWIC swimlane tooling."""

from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

LEGACY_LAP_PHASES = frozenset({"Alloc", "Build", "Replay"})
V4_PHASES = frozenset(
    {
        "OrchestrationReplay",
        "FinalDrain",
        "WinnerBuild",
        "AllocComplete",
        "LoserReplay",
    }
)
V4_EXCLUSIVE_SUBMIT_PHASES = frozenset(
    {
        "EfDrain",
        "Materialize",
        "PrepareMap",
        "Claim",
        "Fanin",
        "Register",
        "WinnerBuild",
        "AllocComplete",
        "LoserReplay",
    }
)
KERNEL_EXECUTION_CHILD_PHASES = frozenset({"EfDrain", "WinnerBuild", "AllocComplete"})
REQUIRED_SUBMIT_PHASES = ("EfDrain", "Materialize", "PrepareMap", "Claim", "Register")
OVERLAY_PHASES = ("Atomic", "ClockBaseline", "Commit", "RingBp", "DrainWon")

_MODEL_PHASES = V4_EXCLUSIVE_SUBMIT_PHASES | {
    "Submit",
    "OrchestrationReplay",
    "FinalDrain",
    "Kernel",
}


@dataclass(frozen=True, slots=True)
class Event:
    """Integer-cycle view of one already row-validated FDWIC event."""

    row_index: int
    core_id: int
    block_id: int
    lane: int
    task_id: int
    function_id: int
    phase: str
    start_cycle: int
    end_cycle: int
    flags: int
    auxiliary: int

    @classmethod
    def from_mapping(cls, row_index: int, event: dict[str, Any]) -> Event:
        return cls(
            row_index=row_index,
            core_id=int(event["core_id"]),
            block_id=int(event["block_id"]),
            lane=int(event["lane"]),
            task_id=int(event["task_id"]),
            function_id=int(event["func_id"]),
            phase=str(event["phase"]),
            start_cycle=int(event["start_cycles"]),
            end_cycle=int(event["end_cycles"]),
            flags=int(event["flags"]),
            auxiliary=int(event["aux"]),
        )

    @property
    def duration(self) -> int:
        return self.end_cycle - self.start_cycle

    @property
    def lane_key(self) -> tuple[int, int]:
        return self.core_id, self.lane


@dataclass(frozen=True, slots=True)
class SubmitPartition:
    submit: Event
    children: tuple[Event, ...]


@dataclass(frozen=True, slots=True)
class CorePartition:
    core_id: int
    block_id: int
    lane: int
    role: str
    orchestration: Event
    final_drain: Event
    submits: tuple[SubmitPartition, ...]


@dataclass(frozen=True, slots=True)
class FdwicV4Model:
    cores: tuple[CorePartition, ...]
    kernels: tuple[Event, ...]
    overlay_statistics: dict[str, dict[str, int]]
    event_count: int
    task_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ResidualSpan:
    core_id: int
    block_id: int
    lane: int
    start_cycle: int
    end_cycle: int
    name: str


def _contains(parent: Event, child: Event) -> bool:
    return parent.start_cycle <= child.start_cycle and child.end_cycle <= parent.end_cycle


def _overlaps(left: Event, right: Event) -> bool:
    return max(left.start_cycle, right.start_cycle) < min(left.end_cycle, right.end_cycle)


def find_containing_event(
    event: Event,
    parents: Sequence[Event],
    starts: Sequence[int],
    container_name: str,
) -> Event | None:
    """Find the containing interval and reject a partial boundary crossing."""

    candidate = bisect.bisect_right(starts, event.start_cycle) - 1
    if candidate >= 0:
        parent = parents[candidate]
        if _contains(parent, event):
            return parent
        if _overlaps(parent, event):
            raise ValueError(f"row {event.row_index} {event.phase} crosses {container_name} row {parent.row_index}")
    next_candidate = candidate + 1
    if next_candidate < len(parents) and _overlaps(parents[next_candidate], event):
        raise ValueError(
            f"row {event.row_index} {event.phase} crosses {container_name} row {parents[next_candidate].row_index}"
        )
    return None


def _expected_layout(core_types: Sequence[str]) -> tuple[tuple[int, int, str], ...]:
    """Mirror the host's dynamic AIC + two-AIV-per-block topology contract."""

    layout: list[tuple[int, int, str] | None] = [None] * len(core_types)
    aic_count = 0
    for core_id, role in enumerate(core_types):
        if role == "aic":
            layout[core_id] = (aic_count, 0, role)
            aic_count += 1
        elif role != "aiv":
            raise ValueError(f"metadata.core_types[{core_id}] has invalid role {role!r}")
    if aic_count == 0:
        raise ValueError("schema-v4 FDWIC topology has no AIC core")

    aiv_ordinal = 0
    for core_id, role in enumerate(core_types):
        if role != "aiv":
            continue
        block_id, lane = divmod(aiv_ordinal, 2)
        if block_id >= aic_count:
            raise ValueError(
                f"schema-v4 AIV core {core_id} cannot map to an AIC block: aic={aic_count} aiv_ordinal={aiv_ordinal}"
            )
        layout[core_id] = (block_id, lane + 1, role)
        aiv_ordinal += 1
    if aiv_ordinal != 2 * aic_count:
        raise ValueError(
            f"schema-v4 FDWIC topology requires exactly two AIV cores per AIC: aic={aic_count} aiv={aiv_ordinal}"
        )
    return tuple(item for item in layout if item is not None)


def _validate_submit_semantics(partition: SubmitPartition) -> None:  # noqa: PLR0912
    submit = partition.submit
    children = partition.children
    counts = Counter(child.phase for child in children)
    for phase in REQUIRED_SUBMIT_PHASES:
        if counts[phase] != 1:
            raise ValueError(
                f"core {submit.core_id} task {submit.task_id} requires exactly one {phase}, got {counts[phase]}"
            )
    if any(count > 1 for count in counts.values()):
        duplicates = sorted(phase for phase, count in counts.items() if count > 1)
        raise ValueError(f"core {submit.core_id} task {submit.task_id} has duplicate exclusive phases {duplicates}")

    is_winner = bool(submit.flags & 1)
    is_alloc = bool(submit.auxiliary)
    expected_sequence: list[str]
    if is_alloc:
        expected_sequence = ["EfDrain", "Materialize", "PrepareMap", "Register", "Claim"]
        if is_winner:
            expected_sequence.append("AllocComplete")
    else:
        expected_sequence = ["EfDrain", "Materialize", "PrepareMap", "Claim"]
        if is_winner:
            expected_sequence.append("Fanin")
        expected_sequence.append("Register")
        expected_sequence.append("WinnerBuild" if is_winner else "LoserReplay")
    actual_sequence = [child.phase for child in children]
    if actual_sequence != expected_sequence:
        raise ValueError(
            f"core {submit.core_id} task {submit.task_id} has invalid exclusive sequence: "
            f"expected={expected_sequence} actual={actual_sequence}"
        )

    claim = next(child for child in children if child.phase == "Claim")
    claim_won = bool(claim.flags & 1)
    claim_attempted = bool(claim.flags & 2)
    if claim_won and not claim_attempted:
        raise ValueError(f"core {submit.core_id} task {submit.task_id} Claim won without an attempt")
    if claim_won != is_winner or bool(claim.auxiliary) != is_alloc:
        raise ValueError(f"core {submit.core_id} task {submit.task_id} Submit/Claim semantics disagree")

    if submit.function_id != claim.function_id:
        raise ValueError(f"core {submit.core_id} task {submit.task_id} Submit/Claim function IDs disagree")
    if is_alloc:
        if submit.function_id != -1:
            raise ValueError(f"core {submit.core_id} alloc task {submit.task_id} must use function_id=-1")
    elif is_winner:
        if submit.function_id < 0:
            raise ValueError(f"core {submit.core_id} kernel winner task {submit.task_id} lacks a function ID")
        for child in children:
            if child.phase in {"Fanin", "Register", "WinnerBuild"} and child.function_id != submit.function_id:
                raise ValueError(
                    f"core {submit.core_id} task {submit.task_id} {child.phase} function ID disagrees with Submit"
                )
    elif submit.function_id != -1:
        raise ValueError(f"core {submit.core_id} kernel loser task {submit.task_id} must use function_id=-1")


def validate_and_partition_v4(  # noqa: PLR0912
    fdwic_events: Sequence[dict[str, Any]],
    num_cores: int,
    core_types: Sequence[str],
) -> FdwicV4Model:
    """Validate production schema-v4 hierarchy and return its shared partition model."""

    if num_cores <= 0 or len(core_types) != num_cores:
        raise ValueError(
            "schema-v4 requires metadata.num_cores to match metadata.core_types: "
            f"num_cores={num_cores} core_types={len(core_types)}"
        )
    expected_layout = _expected_layout(core_types)
    parents: dict[int, dict[str, list[Event]]] = {
        core_id: {"OrchestrationReplay": [], "FinalDrain": []} for core_id in range(num_cores)
    }
    submits_by_core: dict[int, list[Event]] = defaultdict(list)
    children_by_key: dict[tuple[int, int], list[Event]] = defaultdict(list)
    kernels: list[Event] = []
    overlay_statistics = {phase: {"event_count": 0, "aggregate_duration_cycles": 0} for phase in OVERLAY_PHASES}

    for row_index, raw_event in enumerate(fdwic_events):
        core_id = int(raw_event["core_id"])
        if not 0 <= core_id < num_cores:
            raise ValueError(f"fdwic event {row_index} has invalid core_id {core_id}")
        block_id, lane, _role = expected_layout[core_id]
        if int(raw_event["block_id"]) != block_id or int(raw_event["lane"]) != lane:
            raise ValueError(
                f"fdwic event {row_index} has invalid physical identity for core {core_id}: "
                f"block/lane={raw_event['block_id']}/{raw_event['lane']} expected={block_id}/{lane}"
            )
        phase = str(raw_event["phase"])
        duration = int(raw_event["end_cycles"]) - int(raw_event["start_cycles"])
        if phase in overlay_statistics:
            overlay_statistics[phase]["event_count"] += 1
            overlay_statistics[phase]["aggregate_duration_cycles"] += duration
        if phase not in _MODEL_PHASES:
            continue

        event = Event.from_mapping(row_index, raw_event)
        if phase in ("OrchestrationReplay", "FinalDrain"):
            parents[core_id][phase].append(event)
        elif phase == "Submit":
            submits_by_core[core_id].append(event)
        elif phase in V4_EXCLUSIVE_SUBMIT_PHASES:
            children_by_key[(core_id, event.task_id)].append(event)
        elif phase == "Kernel":
            kernels.append(event)

    core_partitions: list[CorePartition] = []
    reference_task_ids: tuple[int, ...] | None = None
    consumed_child_keys: set[tuple[int, int]] = set()
    for core_id in range(num_cores):
        block_id, lane, role = expected_layout[core_id]
        orchestration_rows = parents[core_id]["OrchestrationReplay"]
        final_drain_rows = parents[core_id]["FinalDrain"]
        if len(orchestration_rows) != 1 or len(final_drain_rows) != 1:
            raise ValueError(
                f"core {core_id} requires exactly one OrchestrationReplay and FinalDrain: "
                f"orchestration={len(orchestration_rows)} final_drain={len(final_drain_rows)}"
            )
        orchestration = orchestration_rows[0]
        final_drain = final_drain_rows[0]
        if orchestration.duration < 0 or final_drain.duration < 0:
            raise ValueError(f"core {core_id} has a negative schema-v4 parent duration")
        if orchestration.end_cycle != final_drain.start_cycle:
            raise ValueError(f"core {core_id} OrchestrationReplay.end must equal FinalDrain.start")

        submits = sorted(
            submits_by_core.get(core_id, []),
            key=lambda event: (event.start_cycle, event.end_cycle, event.row_index),
        )
        for previous, current in zip(submits, submits[1:]):
            if _overlaps(previous, current):
                raise ValueError(
                    f"core {core_id} has overlapping Submit rows {previous.row_index} and {current.row_index}"
                )
        task_ids = tuple(submit.task_id for submit in submits)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"core {core_id} has duplicate Submit task IDs")
        if task_ids and task_ids != tuple(range(task_ids[-1] + 1)):
            raise ValueError(f"core {core_id} Submit task IDs are not contiguous 0..N-1: {task_ids}")
        if reference_task_ids is None:
            reference_task_ids = task_ids
        elif task_ids != reference_task_ids:
            raise ValueError(f"core {core_id} Submit task IDs do not match the common replay stream")

        submit_partitions: list[SubmitPartition] = []
        for submit in submits:
            if submit.duration <= 0 or not _contains(orchestration, submit):
                raise ValueError(
                    f"core {core_id} task {submit.task_id} Submit must be positive and contained by OrchestrationReplay"
                )
            key = (core_id, submit.task_id)
            consumed_child_keys.add(key)
            children = sorted(
                children_by_key.get(key, []),
                key=lambda event: (event.start_cycle, event.end_cycle, event.row_index),
            )
            for child in children:
                if not _contains(submit, child):
                    raise ValueError(
                        f"row {child.row_index} {child.phase} is outside core {core_id} task {submit.task_id} Submit"
                    )
            for previous, current in zip(children, children[1:]):
                if _overlaps(previous, current):
                    raise ValueError(
                        f"core {core_id} task {submit.task_id} has overlapping exclusive children "
                        f"rows {previous.row_index} and {current.row_index}"
                    )
            partition = SubmitPartition(submit=submit, children=tuple(children))
            _validate_submit_semantics(partition)
            submit_partitions.append(partition)

        core_partitions.append(
            CorePartition(
                core_id=core_id,
                block_id=block_id,
                lane=lane,
                role=role,
                orchestration=orchestration,
                final_drain=final_drain,
                submits=tuple(submit_partitions),
            )
        )

    orphan_child_keys = set(children_by_key) - consumed_child_keys
    if orphan_child_keys:
        raise ValueError(f"schema-v4 exclusive children have no matching Submit: {sorted(orphan_child_keys)[:8]}")

    exclusive_by_core = {
        core.core_id: sorted(
            [child for partition in core.submits for child in partition.children],
            key=lambda event: (event.start_cycle, event.end_cycle, event.row_index),
        )
        for core in core_partitions
    }
    submits_by_core = {core.core_id: [partition.submit for partition in core.submits] for core in core_partitions}
    exclusive_starts_by_core = {
        core_id: [event.start_cycle for event in events] for core_id, events in exclusive_by_core.items()
    }
    submit_starts_by_core = {
        core_id: [event.start_cycle for event in events] for core_id, events in submits_by_core.items()
    }
    for kernel in kernels:
        core = core_partitions[kernel.core_id]
        inside_orchestration = _contains(core.orchestration, kernel)
        inside_final_drain = _contains(core.final_drain, kernel)
        if inside_orchestration == inside_final_drain:
            raise ValueError(f"row {kernel.row_index} Kernel must be contained by exactly one top-level parent")
        exclusive = exclusive_by_core[kernel.core_id]
        child = find_containing_event(
            kernel,
            exclusive,
            exclusive_starts_by_core[kernel.core_id],
            "exclusive child",
        )
        submits = submits_by_core[kernel.core_id]
        submit = find_containing_event(
            kernel,
            submits,
            submit_starts_by_core[kernel.core_id],
            "Submit",
        )
        if inside_orchestration and child is not None and child.phase not in KERNEL_EXECUTION_CHILD_PHASES:
            raise ValueError(f"row {kernel.row_index} Kernel has unsupported exclusive container {child.phase}")
        if inside_orchestration and submit is not None and child is None:
            raise ValueError(f"row {kernel.row_index} Kernel is inside Submit residual")

    return FdwicV4Model(
        cores=tuple(core_partitions),
        kernels=tuple(kernels),
        overlay_statistics=overlay_statistics,
        event_count=len(fdwic_events),
        task_ids=reference_task_ids or (),
    )


def iter_v4_residual_spans(model: FdwicV4Model) -> Iterator[ResidualSpan]:
    """Yield the exact complement of exclusive children and adjacent Submits."""

    for core in model.cores:
        for previous, current in zip(core.submits, core.submits[1:]):
            start = previous.submit.end_cycle
            end = current.submit.start_cycle
            if end > start:
                yield ResidualSpan(core.core_id, core.block_id, core.lane, start, end, "between_submit_residual")

        for partition in core.submits:
            submit = partition.submit
            cursor = submit.start_cycle
            for child in partition.children:
                if child.start_cycle > cursor:
                    yield ResidualSpan(
                        core.core_id,
                        core.block_id,
                        core.lane,
                        cursor,
                        child.start_cycle,
                        "submit_residual",
                    )
                cursor = child.end_cycle
            if submit.end_cycle > cursor:
                yield ResidualSpan(
                    core.core_id,
                    core.block_id,
                    core.lane,
                    cursor,
                    submit.end_cycle,
                    "submit_tail_gap",
                )
