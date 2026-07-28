#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""按物理 scalar lane 生成 standalone PA Submit 的严格排他 cycle 报告。"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

try:
    # 包方式运行单测时复用同目录 converter 的完整 raw/schema 校验。
    from .swimlane_converter import (
        TASK_KIND_NAMES,
        _derive_v4_task_kinds,
        _load_and_validate,
        _standalone_topology,
    )
except ImportError:
    # 也支持直接执行本脚本，不依赖仓库安装成 Python package。
    from swimlane_converter import (
        TASK_KIND_NAMES,
        _derive_v4_task_kinds,
        _load_and_validate,
        _standalone_topology,
    )


REPORT_SCHEMA_VERSION = 3
EXPECTED_CORES = 96
EXPECTED_AIC_CORES = 32
EXPECTED_AIV_CORES = 64

# v3 的六类显式 child 保持历史口径；v4 只把 winner 的两个真实
# 尾动作加入排他分区。loser 没有尾动作，其剩余时间属于 Submit residual。
# Kernel 只在 EfDrain/FinalDrain 内部再次细分，不会与父区间重复相加。
V3_EXCLUSIVE_SUBMIT_PHASES = (
    "EfDrain",
    "Materialize",
    "PrepareMap",
    "Claim",
    "Fanin",
    "Register",
)
V5_TAIL_PHASES = ("WinnerBuild", "AllocComplete")
V5_EXCLUSIVE_SUBMIT_PHASES = V3_EXCLUSIVE_SUBMIT_PHASES + V5_TAIL_PHASES
PRIVATE_REQUIRED_ON_EVERY_SUBMIT = (
    "EfDrain",
    "Materialize",
    "PrepareMap",
    "Claim",
    "Register",
)
SHARED_REQUIRED_ON_EVERY_SUBMIT = ("EfDrain", "Claim")
SHARED_WINNER_ONLY_PHASES = ("Materialize", "Register")
SHARED_REGISTER_DETAIL_PHASE = "SharedRegisterPublishMetadata"
SHARED_MATERIALIZE_OUTPUT_DETAIL_PHASE = (
    "SharedMaterializePublishTaskOutputs"
)
SHARED_MATERIALIZE_OUTPUT_COPY_PHASE = (
    "SharedMaterializePublishTaskOutputsCopy"
)
SHARED_MATERIALIZE_OUTPUT_FLUSH_PHASE = (
    "SharedMaterializePublishTaskOutputsFlush"
)
LEGACY_SHARED_REGISTER_OUTPUT_DETAIL_PHASE = (
    "SharedRegisterPublishTaskOutputs"
)
LEGACY_SHARED_REGISTER_OUTPUT_COPY_PHASE = (
    "SharedRegisterPublishTaskOutputsCopy"
)
LEGACY_SHARED_REGISTER_OUTPUT_FLUSH_PHASE = (
    "SharedRegisterPublishTaskOutputsFlush"
)
SHARED_OUTPUT_DETAIL_PHASES = (
    SHARED_MATERIALIZE_OUTPUT_DETAIL_PHASE,
    LEGACY_SHARED_REGISTER_OUTPUT_DETAIL_PHASE,
)
SHARED_OUTPUT_COPY_PHASES = (
    SHARED_MATERIALIZE_OUTPUT_COPY_PHASE,
    LEGACY_SHARED_REGISTER_OUTPUT_COPY_PHASE,
)
SHARED_OUTPUT_FLUSH_PHASES = (
    SHARED_MATERIALIZE_OUTPUT_FLUSH_PHASE,
    LEGACY_SHARED_REGISTER_OUTPUT_FLUSH_PHASE,
)
REGISTER_SERIAL_BREAKDOWN_METRICS = (
    "parent",
    "register_wait_predecessor_insert",
    "register_publish_metadata",
    "register_publish_writer_metadata",
    "register_publish_insert_completion",
)
LEGACY_REGISTER_BREAKDOWN_METRICS = (
    *REGISTER_SERIAL_BREAKDOWN_METRICS[:-1],
    "register_publish_task_outputs",
    "register_publish_task_outputs_copy",
    "register_publish_task_outputs_flush",
    "register_publish_task_outputs_residual",
    "register_publish_metadata_epilogue",
    "register_publish_insert_completion",
)
LEGACY_REGISTER_FLAT_PARTITION_METRICS = (
    "register_wait_predecessor_insert",
    "register_publish_writer_metadata",
    "register_publish_task_outputs",
    "register_publish_metadata_epilogue",
    "register_publish_insert_completion",
)
# 报告字段保持向后兼容；新 placement 下五个 legacy output 指标严格为 0，
# fresh-output 的非零明细只出现在 materialize_breakdown。
REGISTER_BREAKDOWN_METRICS = LEGACY_REGISTER_BREAKDOWN_METRICS
REGISTER_FLAT_PARTITION_METRICS = (
    LEGACY_REGISTER_FLAT_PARTITION_METRICS
)
MATERIALIZE_BREAKDOWN_METRICS = (
    "parent",
    "materialize_before_publish_task_outputs",
    "materialize_publish_task_outputs",
    "materialize_publish_task_outputs_copy",
    "materialize_publish_task_outputs_flush",
    "materialize_publish_task_outputs_residual",
    "materialize_after_publish_task_outputs",
)

# 这些记录仍保留调用次数和 aggregate duration，但明确不进入任何闭合式。
OVERLAY_PHASES = (
    "Atomic",
    "ClockBaseline",
    "Commit",
    "RingBp",
    "Build",
    "Replay",
    "Alloc",
    "DrainWon",
)

SUBMIT_PARTITION_METRICS = (
    "efdrain",
    "materialize",
    "prepare_map",
    "claim",
    "fanin",
    "register",
    "submit_residual",
)
V5_SUBMIT_PARTITION_METRICS = (
    *SUBMIT_PARTITION_METRICS[:-1],
    "winner_build",
    "alloc_complete",
    "submit_residual",
)
BASE_ROLE_METRICS = (
    "submit_envelope",
    "submit_union",
    "between_submit_residual",
    *SUBMIT_PARTITION_METRICS,
    "efdrain_kernel_union",
    "efdrain_control",
)
V5_PARENT_METRICS = (
    "orchestration_replay",
    "orchestration_setup",
    "orchestration_tail",
    "final_drain",
    "final_drain_kernel_union",
    "final_drain_residual",
    "worker_completion",
)
PHASE_TO_METRIC = {
    "EfDrain": "efdrain",
    "Materialize": "materialize",
    "PrepareMap": "prepare_map",
    "Claim": "claim",
    "Fanin": "fanin",
    "Register": "register",
    "WinnerBuild": "winner_build",
    "AllocComplete": "alloc_complete",
}
def _exclusive_phases(
    trace_schema_version: int, tensormap_mode: str
) -> tuple[str, ...]:
    if trace_schema_version != 5:
        return V3_EXCLUSIVE_SUBMIT_PHASES
    if tensormap_mode == "shared":
        # shared 的 raw 已从源头禁止 PrepareMap；分析语义也不能继续把它
        # 声称为可能存在的 Submit child。
        return tuple(
            phase
            for phase in V5_EXCLUSIVE_SUBMIT_PHASES
            if phase != "PrepareMap"
        )
    return V5_EXCLUSIVE_SUBMIT_PHASES


def _submit_partition_metrics(trace_schema_version: int) -> tuple[str, ...]:
    return (
        V5_SUBMIT_PARTITION_METRICS
        if trace_schema_version == 5
        else SUBMIT_PARTITION_METRICS
    )


def _role_metrics(trace_schema_version: int) -> tuple[str, ...]:
    if trace_schema_version == 3:
        return BASE_ROLE_METRICS
    return (
        "submit_envelope",
        "submit_union",
        "between_submit_residual",
        *V5_SUBMIT_PARTITION_METRICS,
        "efdrain_kernel_union",
        "efdrain_control",
        *V5_PARENT_METRICS,
    )


@dataclass(frozen=True, slots=True)
class Event:
    """十列 raw ABI 的只读整数视图；row_index 用于给出可追溯错误。"""

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

    @property
    def duration(self) -> int:
        return self.end_cycle - self.start_cycle

    @property
    def lane_key(self) -> tuple[int, int]:
        # core_id 与 lane 同时作为 key，避免以后拓扑扩展时误把不同物理核合并。
        return self.core_id, self.lane


def _event_from_row(index: int, row: tuple[Any, ...]) -> Event:
    return Event(
        row_index=index,
        core_id=int(row[0]),
        block_id=int(row[1]),
        lane=int(row[2]),
        task_id=int(row[3]),
        function_id=int(row[4]),
        phase=str(row[5]),
        start_cycle=int(row[6]),
        end_cycle=int(row[7]),
        flags=int(row[8]),
        auxiliary=int(row[9]),
    )


def _overlaps(left: Event, right: Event) -> bool:
    """raw span 按半开区间解释；首尾相接不算重叠，零时长 instant 不占时间。"""

    return max(left.start_cycle, right.start_cycle) < min(left.end_cycle, right.end_cycle)


def _interval_union_cycles(intervals: Sequence[tuple[int, int]]) -> int:
    """只用整数 cycle 计算区间并集，绝不先换算成浮点微秒。"""

    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + current_end - current_start


def _find_containing_parent(
    event: Event,
    parents: Sequence[Event],
    parent_starts: Sequence[int],
    *,
    parent_name: str,
) -> Event | None:
    """在已排序且互不重叠的父区间中定位唯一容器，并拒绝跨边界截断。"""

    candidate = bisect.bisect_right(parent_starts, event.start_cycle) - 1
    if candidate >= 0:
        parent = parents[candidate]
        if parent.start_cycle <= event.start_cycle and event.end_cycle <= parent.end_cycle:
            return parent

    # 未完整包含时仍要检查相邻父区间；部分相交不能被悄悄当作“父区间外”。
    nearby = {candidate - 1, candidate, candidate + 1, candidate + 2}
    for index in sorted(nearby):
        if 0 <= index < len(parents) and _overlaps(event, parents[index]):
            parent = parents[index]
            raise ValueError(
                f"row {event.row_index} {event.phase} "
                f"[{event.start_cycle},{event.end_cycle}) crosses {parent_name} "
                f"row {parent.row_index} [{parent.start_cycle},{parent.end_cycle})"
            )
    return None


def _median(values: Sequence[int]) -> int | float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    total = ordered[middle - 1] + ordered[middle]
    return total // 2 if total % 2 == 0 else total / 2


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    """p95 使用 nearest-rank；median 只用于横向比较，不参与整数闭合。"""

    if not values:
        raise ValueError("cannot summarize an empty per-core metric")
    ordered = sorted(values)
    p95_index = math.ceil(0.95 * len(ordered)) - 1
    return {
        "median_cycles": _median(ordered),
        "p95_cycles": ordered[p95_index],
        "max_cycles": ordered[-1],
    }


def _validate_capture_identity(
    trace_schema_version: int,
    metadata: dict[str, Any],
    events: Sequence[Event],
) -> tuple[list[str], int]:
    """补足 converter 之外、排他报告必须证明的完整 96 核和 task stream 身份。"""

    if trace_schema_version not in (3, 5):
        raise ValueError(
            "exclusive analysis requires trace_schema_version=3 or 5 because other raw "
            "does not carry producer dropped_records evidence"
        )
    num_cores = int(metadata["num_cores"])
    if num_cores != EXPECTED_CORES:
        raise ValueError(
            f"exclusive analysis requires {EXPECTED_CORES} cores, got {num_cores}"
        )
    core_types = metadata["core_types"]
    expected_types = [
        "aic" if core_id < EXPECTED_AIC_CORES else "aiv"
        for core_id in range(EXPECTED_CORES)
    ]
    if core_types != expected_types:
        raise ValueError("metadata.core_types is not the complete 32 AIC + 64 AIV role map")

    summary = metadata.get("fdwic_summary")
    if not isinstance(summary, dict) or int(summary.get("dropped_records", -1)) != 0:
        raise ValueError("metadata.fdwic_summary.dropped_records must be exactly 0")

    observed_core_ids = {event.core_id for event in events}
    expected_core_ids = set(range(EXPECTED_CORES))
    if observed_core_ids != expected_core_ids:
        missing = sorted(expected_core_ids - observed_core_ids)
        extra = sorted(observed_core_ids - expected_core_ids)
        raise ValueError(f"raw core IDs are incomplete: missing={missing} extra={extra}")
    return expected_types, num_cores


def _validate_and_group_submits(
    events: Sequence[Event],
) -> tuple[dict[tuple[int, int], list[Event]], list[int]]:
    """验证每个 core/lane 的 Submit 不重叠，且 task 0..N-1 顺序完整一致。"""

    by_lane: dict[tuple[int, int], list[Event]] = defaultdict(list)
    for event in events:
        if event.phase == "Submit":
            if event.duration <= 0:
                raise ValueError(f"row {event.row_index} Submit must have positive duration")
            by_lane[event.lane_key].append(event)

    expected_lane_keys = {
        (core_id, _standalone_topology(core_id)[1]) for core_id in range(EXPECTED_CORES)
    }
    if set(by_lane) != expected_lane_keys:
        missing = sorted(expected_lane_keys - set(by_lane))
        extra = sorted(set(by_lane) - expected_lane_keys)
        raise ValueError(f"Submit core/lane IDs are incomplete: missing={missing} extra={extra}")

    reference_task_ids: list[int] | None = None
    for lane_key, submits in by_lane.items():
        submits.sort(key=lambda event: (event.start_cycle, event.end_cycle, event.row_index))
        for previous, current in zip(submits, submits[1:]):
            if current.start_cycle < previous.end_cycle:
                raise ValueError(
                    f"core/lane {lane_key} has overlapping Submit rows "
                    f"{previous.row_index} and {current.row_index}"
                )
        task_ids = [event.task_id for event in submits]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"core/lane {lane_key} has duplicate Submit task IDs")
        if reference_task_ids is None:
            if not task_ids or task_ids != list(range(task_ids[-1] + 1)):
                raise ValueError(
                    f"core/lane {lane_key} Submit task IDs are not contiguous 0..N-1: {task_ids}"
                )
            reference_task_ids = task_ids
        elif task_ids != reference_task_ids:
            raise ValueError(
                f"core/lane {lane_key} Submit task IDs do not match the common task stream"
            )

    assert reference_task_ids is not None
    return by_lane, reference_task_ids


def _associate_exclusive_children(
    events: Sequence[Event],
    submits_by_lane: dict[tuple[int, int], list[Event]],
    exclusive_phases: Sequence[str],
) -> dict[int, list[Event]]:
    """把每条显式 child 严格归入同一 core/lane 上唯一包含它的 Submit。"""

    children: dict[int, list[Event]] = {
        submit.row_index: []
        for submits in submits_by_lane.values()
        for submit in submits
    }
    starts = {
        lane_key: [submit.start_cycle for submit in submits]
        for lane_key, submits in submits_by_lane.items()
    }
    exclusive_phase_set = set(exclusive_phases)
    for event in events:
        if event.phase not in exclusive_phase_set:
            continue
        parents = submits_by_lane[event.lane_key]
        parent = _find_containing_parent(
            event, parents, starts[event.lane_key], parent_name="Submit"
        )
        if parent is None:
            raise ValueError(
                f"row {event.row_index} {event.phase} is outside every Submit "
                f"on core/lane {event.lane_key}"
            )
        # Submit 前端和真实尾动作都描述“当前 task”的 scalar 工作；仅 Kernel
        # 允许在 EfDrain/FinalDrain 中执行前序 task，因此不经过这条关联路径。
        if event.task_id != parent.task_id:
            raise ValueError(
                f"row {event.row_index} {event.phase} task_id={event.task_id} "
                f"does not match containing Submit task_id={parent.task_id}"
            )
        children[parent.row_index].append(event)
    return children


def _associate_shared_register_details(
    events: Sequence[Event],
    children_by_submit: dict[int, list[Event]],
    trace_schema_version: int,
    tensormap_mode: str,
) -> tuple[
    dict[int, Event],
    dict[int, Event],
    dict[int, Event],
    dict[int, Event],
    str,
]:
    """建立 Register→metadata 与 Materialize→outputs 的严格关联。

    ``outputs_by_owner`` 以父 row index 为键。新采集的 owner 是
    Materialize；迁移前 schema-v5 raw 的 owner 是 metadata，并通过返回的
    ``output_placement`` 显式区分，避免把历史数据悄悄套用新口径。
    """

    registers = [
        child
        for children in children_by_submit.values()
        for child in children
        if child.phase == "Register"
    ]
    materializes = [
        child
        for children in children_by_submit.values()
        for child in children
        if child.phase == "Materialize"
    ]
    details = [
        event for event in events if event.phase == SHARED_REGISTER_DETAIL_PHASE
    ]
    output_details = [
        event
        for event in events
        if event.phase in SHARED_OUTPUT_DETAIL_PHASES
    ]
    output_copy_details = [
        event
        for event in events
        if event.phase in SHARED_OUTPUT_COPY_PHASES
    ]
    output_flush_details = [
        event
        for event in events
        if event.phase in SHARED_OUTPUT_FLUSH_PHASES
    ]
    if trace_schema_version != 5 or tensormap_mode != "shared":
        if (
            details
            or output_details
            or output_copy_details
            or output_flush_details
        ):
            raise ValueError(
                "shared Materialize/Register details are only valid for "
                "shared schema-v5"
            )
        return {}, {}, {}, {}, "none"

    def _associate_unique(
        nested: Sequence[Event],
        parents: Sequence[Event],
        *,
        child_name: str,
        parent_name: str,
    ) -> dict[int, Event]:
        parents_by_lane: dict[tuple[int, int], list[Event]] = defaultdict(list)
        for parent in parents:
            parents_by_lane[parent.lane_key].append(parent)
        for lane_parents in parents_by_lane.values():
            lane_parents.sort(
                key=lambda event: (
                    event.start_cycle,
                    event.end_cycle,
                    event.row_index,
                )
            )
        starts = {
            lane_key: [event.start_cycle for event in lane_parents]
            for lane_key, lane_parents in parents_by_lane.items()
        }
        association: dict[int, Event] = {}
        for child in nested:
            lane_parents = parents_by_lane.get(child.lane_key, [])
            parent = _find_containing_parent(
                child,
                lane_parents,
                starts.get(child.lane_key, []),
                parent_name=parent_name,
            )
            if parent is None:
                raise ValueError(
                    f"row {child.row_index} {child.phase} is outside every "
                    f"{parent_name} on core/lane {child.lane_key}"
                )
            if (
                child.core_id != parent.core_id
                or child.lane != parent.lane
                or child.task_id != parent.task_id
                or child.function_id != parent.function_id
            ):
                raise ValueError(
                    f"row {child.row_index} {child.phase} identity does not "
                    f"match {parent_name} row {parent.row_index}"
                )
            previous = association.setdefault(parent.row_index, child)
            if previous is not child:
                raise ValueError(
                    f"{parent_name} row {parent.row_index} has duplicate "
                    f"{child_name} rows {previous.row_index} and "
                    f"{child.row_index}"
                )
        missing = [
            parent.row_index
            for parent in parents
            if parent.row_index not in association
        ]
        if missing:
            raise ValueError(
                "shared schema-v5 requires exactly one "
                f"{child_name} per {parent_name}: "
                f"missing_parent_rows={missing[:8]}"
            )
        if len(association) != len(parents):
            raise AssertionError(
                f"shared {child_name} association is not one-to-one"
            )
        return association

    detail_by_register = _associate_unique(
        details,
        registers,
        child_name=SHARED_REGISTER_DETAIL_PHASE,
        parent_name="Register",
    )

    output_placements = {
        (
            "materialize"
            if event.phase == SHARED_MATERIALIZE_OUTPUT_DETAIL_PHASE
            else "register_legacy"
        )
        for event in output_details
    }
    if len(output_placements) != 1:
        raise ValueError(
            "shared schema-v5 capture must use one task-output placement; "
            f"got {sorted(output_placements)}"
        )
    output_placement = next(iter(output_placements))
    output_parents = (
        materializes if output_placement == "materialize" else details
    )
    output_parent_name = (
        "Materialize"
        if output_placement == "materialize"
        else SHARED_REGISTER_DETAIL_PHASE
    )
    outputs_by_owner = _associate_unique(
        output_details,
        output_parents,
        child_name=(
            SHARED_MATERIALIZE_OUTPUT_DETAIL_PHASE
            if output_placement == "materialize"
            else LEGACY_SHARED_REGISTER_OUTPUT_DETAIL_PHASE
        ),
        parent_name=output_parent_name,
    )

    outputs_by_lane: dict[tuple[int, int], list[Event]] = defaultdict(list)
    for output_detail in output_details:
        outputs_by_lane[output_detail.lane_key].append(output_detail)
    for lane_outputs in outputs_by_lane.values():
        lane_outputs.sort(
            key=lambda event: (event.start_cycle, event.end_cycle, event.row_index)
        )
    output_starts = {
        lane_key: [event.start_cycle for event in lane_outputs]
        for lane_key, lane_outputs in outputs_by_lane.items()
    }

    def _nest_under_outputs(
        children: list[Event],
        phase_name: str,
        association: dict[int, Event],
    ) -> None:
        for child in children:
            lane_outputs = outputs_by_lane.get(child.lane_key, [])
            parent_output = _find_containing_parent(
                child,
                lane_outputs,
                output_starts.get(child.lane_key, []),
                parent_name="task-output publication",
            )
            if parent_output is None:
                raise ValueError(
                    f"row {child.row_index} {child.phase} is outside every "
                    "task-output publication on core/lane "
                    f"{child.lane_key}"
                )
            if (
                child.core_id != parent_output.core_id
                or child.lane != parent_output.lane
                or child.task_id != parent_output.task_id
                or child.function_id != parent_output.function_id
            ):
                raise ValueError(
                    f"row {child.row_index} {child.phase} identity does not "
                    "match task-output publication row "
                    f"{parent_output.row_index}"
                )
            previous = association.setdefault(parent_output.row_index, child)
            if previous is not child:
                raise ValueError(
                    "task-output publication row "
                    f"{parent_output.row_index} has duplicate {phase_name} rows "
                    f"{previous.row_index} and {child.row_index}"
                )
        missing = [
            output_detail.row_index
            for output_detail in output_details
            if output_detail.row_index not in association
        ]
        if missing:
            raise ValueError(
                "shared schema-v5 requires exactly one "
                f"{phase_name} per task-output publication: "
                f"missing_output_rows={missing[:8]}"
            )
        if len(association) != len(output_details):
            raise AssertionError(
                f"shared {phase_name} association is not one-to-one"
            )

    copies_by_outputs: dict[int, Event] = {}
    flushes_by_outputs: dict[int, Event] = {}
    expected_copy_phase = (
        SHARED_MATERIALIZE_OUTPUT_COPY_PHASE
        if output_placement == "materialize"
        else LEGACY_SHARED_REGISTER_OUTPUT_COPY_PHASE
    )
    expected_flush_phase = (
        SHARED_MATERIALIZE_OUTPUT_FLUSH_PHASE
        if output_placement == "materialize"
        else LEGACY_SHARED_REGISTER_OUTPUT_FLUSH_PHASE
    )
    if any(
        event.phase != expected_copy_phase
        for event in output_copy_details
    ) or any(
        event.phase != expected_flush_phase
        for event in output_flush_details
    ):
        raise ValueError(
            "shared schema-v5 task-output parent/copy/flush phase families "
            "must not be mixed"
        )
    _nest_under_outputs(
        output_copy_details,
        expected_copy_phase,
        copies_by_outputs,
    )
    _nest_under_outputs(
        output_flush_details,
        expected_flush_phase,
        flushes_by_outputs,
    )
    for output_row, copy_event in copies_by_outputs.items():
        flush_event = flushes_by_outputs[output_row]
        output_event = next(
            event for event in output_details if event.row_index == output_row
        )
        if not (
            output_event.start_cycle
            <= copy_event.start_cycle
            <= copy_event.end_cycle
            == flush_event.start_cycle
            <= flush_event.end_cycle
            <= output_event.end_cycle
        ):
            raise ValueError(
                f"task-output publication row {output_row} "
                "copy/flush nesting is invalid"
            )
    return (
        detail_by_register,
        outputs_by_owner,
        copies_by_outputs,
        flushes_by_outputs,
        output_placement,
    )


def _build_register_breakdown(
    children_by_submit: dict[int, list[Event]],
    detail_by_register: dict[int, Event],
    outputs_by_owner: dict[int, Event],
    copies_by_outputs: dict[int, Event],
    flushes_by_outputs: dict[int, Event],
    output_placement: str,
    num_cores: int,
    trace_schema_version: int,
    tensormap_mode: str,
) -> dict[str, Any] | None:
    """由 raw 整数边界构造 Register 串行区排他闭合报告。"""

    if trace_schema_version != 5 or tensormap_mode != "shared":
        return None

    registers_by_core: dict[int, list[Event]] = defaultdict(list)
    for children in children_by_submit.values():
        for child in children:
            if child.phase == "Register":
                registers_by_core[child.core_id].append(child)

    per_core: list[dict[str, Any]] = []
    for core_id in range(num_cores):
        metrics = {name: 0 for name in REGISTER_BREAKDOWN_METRICS}
        register_count = 0
        for parent in registers_by_core.get(core_id, []):
            detail = detail_by_register[parent.row_index]
            wait_cycles = detail.start_cycle - parent.start_cycle
            publish_cycles = detail.duration
            if output_placement == "register_legacy":
                output_detail = outputs_by_owner[detail.row_index]
                copy_detail = copies_by_outputs[output_detail.row_index]
                flush_detail = flushes_by_outputs[output_detail.row_index]
                writer_metadata_cycles = (
                    output_detail.start_cycle - detail.start_cycle
                )
                task_outputs_cycles = output_detail.duration
                copy_cycles = copy_detail.duration
                flush_cycles = flush_detail.duration
                outputs_residual_cycles = (
                    task_outputs_cycles - copy_cycles - flush_cycles
                )
                metadata_epilogue_cycles = (
                    detail.end_cycle - output_detail.end_cycle
                )
            else:
                writer_metadata_cycles = publish_cycles
                task_outputs_cycles = 0
                copy_cycles = 0
                flush_cycles = 0
                outputs_residual_cycles = 0
                metadata_epilogue_cycles = 0
            handoff_cycles = parent.end_cycle - detail.end_cycle
            if min(
                wait_cycles,
                publish_cycles,
                writer_metadata_cycles,
                task_outputs_cycles,
                copy_cycles,
                flush_cycles,
                outputs_residual_cycles,
                metadata_epilogue_cycles,
                handoff_cycles,
            ) < 0:
                raise ValueError(
                    f"Register row {parent.row_index} internal partition has "
                    "a negative raw-cycle segment"
                )
            if (
                writer_metadata_cycles
                + task_outputs_cycles
                + metadata_epilogue_cycles
                != publish_cycles
            ):
                raise ValueError(
                    f"{SHARED_REGISTER_DETAIL_PHASE} row {detail.row_index} "
                    "internal partition does not close in raw cycles"
                )
            if (
                copy_cycles + flush_cycles + outputs_residual_cycles
                != task_outputs_cycles
            ):
                raise ValueError(
                    "legacy Register task-output copy/flush residual does not "
                    "close in raw cycles"
                )
            if wait_cycles + publish_cycles + handoff_cycles != parent.duration:
                raise ValueError(
                    f"Register row {parent.row_index} internal partition does not "
                    "close in raw cycles"
                )
            metrics["parent"] += parent.duration
            metrics["register_wait_predecessor_insert"] += wait_cycles
            metrics["register_publish_metadata"] += publish_cycles
            metrics["register_publish_writer_metadata"] += (
                writer_metadata_cycles
            )
            metrics["register_publish_task_outputs"] += task_outputs_cycles
            metrics["register_publish_task_outputs_copy"] += copy_cycles
            metrics["register_publish_task_outputs_flush"] += flush_cycles
            metrics["register_publish_task_outputs_residual"] += (
                outputs_residual_cycles
            )
            metrics["register_publish_metadata_epilogue"] += (
                metadata_epilogue_cycles
            )
            metrics["register_publish_insert_completion"] += handoff_cycles
            register_count += 1

        metadata_children = (
            metrics["register_publish_writer_metadata"]
            + metrics["register_publish_task_outputs"]
            + metrics["register_publish_metadata_epilogue"]
        )
        flat_children = sum(
            metrics[name] for name in REGISTER_FLAT_PARTITION_METRICS
        )
        outputs_children = (
            metrics["register_publish_task_outputs_copy"]
            + metrics["register_publish_task_outputs_flush"]
            + metrics["register_publish_task_outputs_residual"]
        )
        if metadata_children != metrics["register_publish_metadata"]:
            raise ValueError(
                f"core {core_id} aggregate metadata partition does not close"
            )
        if flat_children != metrics["parent"]:
            raise ValueError(
                f"core {core_id} aggregate Register partition does not close"
            )
        if outputs_children != metrics["register_publish_task_outputs"]:
            raise ValueError(
                f"core {core_id} aggregate task-outputs partition does not close"
            )
        block_id, lane, role = _standalone_topology(core_id)
        per_core.append(
            {
                "core_id": core_id,
                "block_id": block_id,
                "lane": lane,
                "role": role,
                "register_count": register_count,
                "metrics_cycles": metrics,
                "closure": {
                    "register": {
                        "parent_cycles": metrics["parent"],
                        "flat_children_cycles": flat_children,
                        "exact": True,
                    },
                    "metadata": {
                        "parent_cycles": metrics[
                            "register_publish_metadata"
                        ],
                        "children_cycles": metadata_children,
                        "exact": True,
                    },
                    "task_outputs": {
                        "parent_cycles": metrics[
                            "register_publish_task_outputs"
                        ],
                        "children_cycles": outputs_children,
                        "exact": True,
                    },
                },
            }
        )

    aggregate = {
        metric: sum(core["metrics_cycles"][metric] for core in per_core)
        for metric in REGISTER_BREAKDOWN_METRICS
    }
    aggregate_metadata_children = (
        aggregate["register_publish_writer_metadata"]
        + aggregate["register_publish_task_outputs"]
        + aggregate["register_publish_metadata_epilogue"]
    )
    aggregate_flat_children = sum(
        aggregate[name] for name in REGISTER_FLAT_PARTITION_METRICS
    )
    aggregate_outputs_children = (
        aggregate["register_publish_task_outputs_copy"]
        + aggregate["register_publish_task_outputs_flush"]
        + aggregate["register_publish_task_outputs_residual"]
    )
    if aggregate_metadata_children != aggregate["register_publish_metadata"]:
        raise AssertionError("aggregate metadata internal partition does not close")
    if aggregate_flat_children != aggregate["parent"]:
        raise AssertionError("aggregate Register internal partition does not close")
    if (
        aggregate_outputs_children
        != aggregate["register_publish_task_outputs"]
    ):
        raise AssertionError(
            "aggregate task-outputs internal partition does not close"
        )

    role_statistics: dict[str, Any] = {}
    for role, expected_count in (
        ("aic", EXPECTED_AIC_CORES),
        ("aiv", EXPECTED_AIV_CORES),
    ):
        role_cores = [core for core in per_core if core["role"] == role]
        if len(role_cores) != expected_count:
            raise ValueError(
                f"Register breakdown role {role} has {len(role_cores)} cores, "
                f"expected {expected_count}"
            )
        role_statistics[role] = {
            "core_count": len(role_cores),
            "metrics": {
                metric: _distribution(
                    [core["metrics_cycles"][metric] for core in role_cores]
                )
                for metric in REGISTER_BREAKDOWN_METRICS
            },
        }

    return {
        "semantics": {
            "parent": "the existing exclusive Register span",
            "register_wait_predecessor_insert": (
                "Register.start to SharedRegisterPublishMetadata.start; "
                "task 0 enters directly, while task N waits for task N-1 to "
                "publish its TensorMap insertion completion"
            ),
            "register_publish_metadata": (
                "the SharedRegisterPublishMetadata raw parent detail"
            ),
            "register_publish_writer_metadata": (
                "the serialized ordinary/symbol writer publication. In new "
                "captures this is the complete SharedRegisterPublishMetadata "
                "span; legacy captures end at task-output publication start"
            ),
            "register_publish_task_outputs": (
                "legacy compatibility field; exactly zero when output_placement "
                "is materialize"
            ),
            "register_publish_task_outputs_copy": (
                "SharedRegisterPublishTaskOutputsCopy; batch TensorDesc copy "
                "into shared_outputs[task].tensors[]"
            ),
            "register_publish_task_outputs_flush": (
                "SharedRegisterPublishTaskOutputsFlush; FlushRegion of the "
                "copied descriptors before StoreBarrier/published"
            ),
            "register_publish_task_outputs_residual": (
                "task-outputs envelope minus copy/flush: pre-check, "
                "last_writer FetchMax, StoreBarrier, and published Exchange"
            ),
            "register_publish_metadata_epilogue": (
                "legacy compatibility field after Register-owned output "
                "publication; exactly zero in new captures"
            ),
            "register_publish_insert_completion": (
                "SharedRegisterPublishMetadata.end to Register.end; publish this "
                "task's TensorMap insertion completion for its successor"
            ),
            "raw_arithmetic": "integer cycle boundaries; merged swimlane is not read",
            "output_placement": output_placement,
            "included_in_submit_additive_totals": {
                "parent": True,
                SHARED_REGISTER_DETAIL_PHASE: False,
                LEGACY_SHARED_REGISTER_OUTPUT_DETAIL_PHASE: False,
                LEGACY_SHARED_REGISTER_OUTPUT_COPY_PHASE: False,
                LEGACY_SHARED_REGISTER_OUTPUT_FLUSH_PHASE: False,
            },
        },
        "event_count": {
            "metadata": len(detail_by_register),
            "task_outputs": (
                len(outputs_by_owner)
                if output_placement == "register_legacy"
                else 0
            ),
            "task_outputs_copy": (
                len(copies_by_outputs)
                if output_placement == "register_legacy"
                else 0
            ),
            "task_outputs_flush": (
                len(flushes_by_outputs)
                if output_placement == "register_legacy"
                else 0
            ),
        },
        "aggregate_core_work": {
            "metrics_cycles": aggregate,
            "closure": {
                "register": {
                    "parent_cycles": aggregate["parent"],
                    "flat_children_cycles": aggregate_flat_children,
                    "exact": True,
                },
                "metadata": {
                    "parent_cycles": aggregate["register_publish_metadata"],
                    "children_cycles": aggregate_metadata_children,
                    "exact": True,
                },
                "task_outputs": {
                    "parent_cycles": aggregate[
                        "register_publish_task_outputs"
                    ],
                    "children_cycles": aggregate_outputs_children,
                    "exact": True,
                },
            },
        },
        "per_role_core_statistics": role_statistics,
        "per_core": per_core,
    }


def _build_materialize_breakdown(
    children_by_submit: dict[int, list[Event]],
    outputs_by_owner: dict[int, Event],
    copies_by_outputs: dict[int, Event],
    flushes_by_outputs: dict[int, Event],
    output_placement: str,
    num_cores: int,
    trace_schema_version: int,
    tensormap_mode: str,
) -> dict[str, Any] | None:
    """闭合新路径的 Materialize→task outputs→copy/flush 两级分区。"""

    if (
        trace_schema_version != 5
        or tensormap_mode != "shared"
        or output_placement != "materialize"
    ):
        return None

    materializes_by_core: dict[int, list[Event]] = defaultdict(list)
    for children in children_by_submit.values():
        for child in children:
            if child.phase == "Materialize":
                materializes_by_core[child.core_id].append(child)

    per_core: list[dict[str, Any]] = []
    for core_id in range(num_cores):
        metrics = {
            name: 0 for name in MATERIALIZE_BREAKDOWN_METRICS
        }
        materialize_count = 0
        for parent in materializes_by_core.get(core_id, []):
            output = outputs_by_owner[parent.row_index]
            copy = copies_by_outputs[output.row_index]
            flush = flushes_by_outputs[output.row_index]
            before_cycles = output.start_cycle - parent.start_cycle
            output_cycles = output.duration
            copy_cycles = copy.duration
            flush_cycles = flush.duration
            output_residual_cycles = (
                output_cycles - copy_cycles - flush_cycles
            )
            after_cycles = parent.end_cycle - output.end_cycle
            if min(
                before_cycles,
                output_cycles,
                copy_cycles,
                flush_cycles,
                output_residual_cycles,
                after_cycles,
            ) < 0:
                raise ValueError(
                    f"Materialize row {parent.row_index} internal partition "
                    "has a negative raw-cycle segment"
                )
            if (
                copy_cycles + flush_cycles + output_residual_cycles
                != output_cycles
            ):
                raise ValueError(
                    f"{SHARED_MATERIALIZE_OUTPUT_DETAIL_PHASE} row "
                    f"{output.row_index} does not close"
                )
            if (
                before_cycles + output_cycles + after_cycles
                != parent.duration
            ):
                raise ValueError(
                    f"Materialize row {parent.row_index} does not close"
                )
            metrics["parent"] += parent.duration
            metrics[
                "materialize_before_publish_task_outputs"
            ] += before_cycles
            metrics["materialize_publish_task_outputs"] += output_cycles
            metrics[
                "materialize_publish_task_outputs_copy"
            ] += copy_cycles
            metrics[
                "materialize_publish_task_outputs_flush"
            ] += flush_cycles
            metrics[
                "materialize_publish_task_outputs_residual"
            ] += output_residual_cycles
            metrics[
                "materialize_after_publish_task_outputs"
            ] += after_cycles
            materialize_count += 1

        flat_children = (
            metrics["materialize_before_publish_task_outputs"]
            + metrics["materialize_publish_task_outputs"]
            + metrics["materialize_after_publish_task_outputs"]
        )
        output_children = (
            metrics["materialize_publish_task_outputs_copy"]
            + metrics["materialize_publish_task_outputs_flush"]
            + metrics["materialize_publish_task_outputs_residual"]
        )
        if flat_children != metrics["parent"]:
            raise ValueError(
                f"core {core_id} aggregate Materialize partition does not close"
            )
        if (
            output_children
            != metrics["materialize_publish_task_outputs"]
        ):
            raise ValueError(
                f"core {core_id} aggregate task-output partition does not close"
            )
        block_id, lane, role = _standalone_topology(core_id)
        per_core.append(
            {
                "core_id": core_id,
                "block_id": block_id,
                "lane": lane,
                "role": role,
                "materialize_count": materialize_count,
                "metrics_cycles": metrics,
                "closure": {
                    "materialize": {
                        "parent_cycles": metrics["parent"],
                        "flat_children_cycles": flat_children,
                        "exact": True,
                    },
                    "task_outputs": {
                        "parent_cycles": metrics[
                            "materialize_publish_task_outputs"
                        ],
                        "children_cycles": output_children,
                        "exact": True,
                    },
                },
            }
        )

    aggregate = {
        metric: sum(core["metrics_cycles"][metric] for core in per_core)
        for metric in MATERIALIZE_BREAKDOWN_METRICS
    }
    aggregate_flat_children = (
        aggregate["materialize_before_publish_task_outputs"]
        + aggregate["materialize_publish_task_outputs"]
        + aggregate["materialize_after_publish_task_outputs"]
    )
    aggregate_output_children = (
        aggregate["materialize_publish_task_outputs_copy"]
        + aggregate["materialize_publish_task_outputs_flush"]
        + aggregate["materialize_publish_task_outputs_residual"]
    )
    if aggregate_flat_children != aggregate["parent"]:
        raise AssertionError(
            "aggregate Materialize internal partition does not close"
        )
    if (
        aggregate_output_children
        != aggregate["materialize_publish_task_outputs"]
    ):
        raise AssertionError(
            "aggregate Materialize task-output partition does not close"
        )

    role_statistics: dict[str, Any] = {}
    for role, expected_count in (
        ("aic", EXPECTED_AIC_CORES),
        ("aiv", EXPECTED_AIV_CORES),
    ):
        role_cores = [core for core in per_core if core["role"] == role]
        if len(role_cores) != expected_count:
            raise ValueError(
                f"Materialize breakdown role {role} has "
                f"{len(role_cores)} cores, expected {expected_count}"
            )
        role_statistics[role] = {
            "core_count": len(role_cores),
            "metrics": {
                metric: _distribution(
                    [
                        core["metrics_cycles"][metric]
                        for core in role_cores
                    ]
                )
                for metric in MATERIALIZE_BREAKDOWN_METRICS
            },
        }

    return {
        "semantics": {
            "parent": "the existing exclusive Materialize span",
            "materialize_before_publish_task_outputs": (
                "Materialize.start to output-publication.start; task "
                "materialization plus writer-delta preparation"
            ),
            "materialize_publish_task_outputs": (
                f"the unique {SHARED_MATERIALIZE_OUTPUT_DETAIL_PHASE} raw detail"
            ),
            "materialize_publish_task_outputs_copy": (
                f"{SHARED_MATERIALIZE_OUTPUT_COPY_PHASE}; batch TensorDesc copy "
                "into shared_outputs[task].tensors[]"
            ),
            "materialize_publish_task_outputs_flush": (
                f"{SHARED_MATERIALIZE_OUTPUT_FLUSH_PHASE}; FlushRegion before "
                "StoreBarrier/published"
            ),
            "materialize_publish_task_outputs_residual": (
                "output envelope minus copy/flush: pre-check, last_writer "
                "FetchMax, StoreBarrier, published Exchange, and helper overhead"
            ),
            "materialize_after_publish_task_outputs": (
                "output-publication.end to Materialize.end; phase closure and "
                "outer timestamp"
            ),
            "raw_arithmetic": (
                "integer cycle boundaries; merged swimlane is not read"
            ),
            "included_in_submit_additive_totals": {
                "parent": True,
                SHARED_MATERIALIZE_OUTPUT_DETAIL_PHASE: False,
                SHARED_MATERIALIZE_OUTPUT_COPY_PHASE: False,
                SHARED_MATERIALIZE_OUTPUT_FLUSH_PHASE: False,
            },
        },
        "event_count": {
            "task_outputs": len(outputs_by_owner),
            "task_outputs_copy": len(copies_by_outputs),
            "task_outputs_flush": len(flushes_by_outputs),
        },
        "aggregate_core_work": {
            "metrics_cycles": aggregate,
            "closure": {
                "materialize": {
                    "parent_cycles": aggregate["parent"],
                    "flat_children_cycles": aggregate_flat_children,
                    "exact": True,
                },
                "task_outputs": {
                    "parent_cycles": aggregate[
                        "materialize_publish_task_outputs"
                    ],
                    "children_cycles": aggregate_output_children,
                    "exact": True,
                },
            },
        },
        "per_role_core_statistics": role_statistics,
        "per_core": per_core,
    }


def _associate_kernels_to_parents(
    events: Sequence[Event],
    parents_by_lane: dict[tuple[int, int], list[Event]],
    *,
    parent_name: str,
) -> tuple[dict[int, list[Event]], set[int]]:
    """给一类互不重叠的父 span 建立 Kernel 包含关系，并返回被消费的 row ID。"""

    for parents in parents_by_lane.values():
        parents.sort(key=lambda event: (event.start_cycle, event.end_cycle, event.row_index))
    starts = {
        lane_key: [event.start_cycle for event in parents]
        for lane_key, parents in parents_by_lane.items()
    }
    kernels_by_parent: dict[int, list[Event]] = {
        parent.row_index: []
        for parents in parents_by_lane.values()
        for parent in parents
    }
    contained_rows: set[int] = set()
    for kernel in (event for event in events if event.phase == "Kernel"):
        parents = parents_by_lane.get(kernel.lane_key, [])
        parent = _find_containing_parent(
            kernel,
            parents,
            starts.get(kernel.lane_key, []),
            parent_name=parent_name,
        )
        if parent is not None:
            kernels_by_parent[parent.row_index].append(kernel)
            contained_rows.add(kernel.row_index)
    return kernels_by_parent, contained_rows


def _associate_efdrain_kernels(
    events: Sequence[Event],
    children_by_submit: dict[int, list[Event]],
) -> tuple[dict[int, list[Event]], set[int]]:
    """只把完整包含于 EfDrain 的 Kernel 纳入其 nested union。"""

    efdrains_by_lane: dict[tuple[int, int], list[Event]] = defaultdict(list)
    for children in children_by_submit.values():
        for event in children:
            if event.phase == "EfDrain":
                efdrains_by_lane[event.lane_key].append(event)
    return _associate_kernels_to_parents(
        events, efdrains_by_lane, parent_name="EfDrain"
    )


def _associate_v4_tail_kernels(
    events: Sequence[Event],
    children_by_submit: dict[int, list[Event]],
) -> tuple[dict[int, list[Event]], set[int], dict[str, int]]:
    """把背压期间执行的 Kernel 唯一归入 WinnerBuild/AllocComplete 父动作。"""

    tails_by_lane: dict[tuple[int, int], list[Event]] = defaultdict(list)
    tail_phase_by_row: dict[int, str] = {}
    for children in children_by_submit.values():
        for event in children:
            if event.phase in ("WinnerBuild", "AllocComplete"):
                tails_by_lane[event.lane_key].append(event)
                tail_phase_by_row[event.row_index] = event.phase
    kernels_by_tail, contained_rows = _associate_kernels_to_parents(
        events, tails_by_lane, parent_name="Submit tail"
    )
    counts = {"WinnerBuild": 0, "AllocComplete": 0}
    for parent_row, kernels in kernels_by_tail.items():
        counts[tail_phase_by_row[parent_row]] += len(kernels)
    return kernels_by_tail, contained_rows, counts


def _group_v4_parents(
    events: Sequence[Event], phase: str
) -> dict[tuple[int, int], list[Event]]:
    """converter 已校验每核恰一条；这里保留列表形状以复用区间定位函数。"""

    parents: dict[tuple[int, int], list[Event]] = defaultdict(list)
    for event in events:
        if event.phase == phase:
            parents[event.lane_key].append(event)
    expected_keys = {
        (core_id, _standalone_topology(core_id)[1]) for core_id in range(EXPECTED_CORES)
    }
    if set(parents) != expected_keys or any(len(items) != 1 for items in parents.values()):
        raise ValueError(f"schema-v5 requires exactly one {phase} per core/lane")
    return parents


def _validate_submit_frontend_contract(
    core_id: int,
    trace_schema_version: int,
    tensormap_mode: str,
    submit: Event,
    counts: Counter[str],
) -> None:
    """按 TensorMap 模式校验 Submit 前端 child 的基数。"""

    winner = bool(submit.flags & 1)
    if trace_schema_version == 5 and tensormap_mode == "shared":
        for phase in SHARED_REQUIRED_ON_EVERY_SUBMIT:
            if counts[phase] != 1:
                raise ValueError(
                    f"core {core_id} task {submit.task_id} shared path requires "
                    f"exactly one {phase}, got {counts[phase]}"
                )
        for phase in SHARED_WINNER_ONLY_PHASES:
            expected_count = 1 if winner else 0
            if counts[phase] != expected_count:
                raise ValueError(
                    f"core {core_id} task {submit.task_id} shared "
                    f"{'winner' if winner else 'loser'} path requires "
                    f"{expected_count} {phase} spans, got {counts[phase]}"
                )
        if counts["PrepareMap"] != 0:
            raise ValueError(
                f"core {core_id} task {submit.task_id} shared path forbids PrepareMap"
            )
    else:
        for phase in PRIVATE_REQUIRED_ON_EVERY_SUBMIT:
            if counts[phase] != 1:
                raise ValueError(
                    f"core {core_id} task {submit.task_id} requires exactly one "
                    f"{phase}, got {counts[phase]}"
                )
    if counts["Fanin"] > 1:
        raise ValueError(
            f"core {core_id} task {submit.task_id} has {counts['Fanin']} Fanin spans"
        )


def _analyze_core(
    core_id: int,
    role: str,
    trace_schema_version: int,
    tensormap_mode: str,
    submits: Sequence[Event],
    children_by_submit: dict[int, list[Event]],
    kernels_by_efdrain: dict[int, list[Event]],
    orchestration: Event | None,
    final_drain: Event | None,
    kernels_by_final_drain: dict[int, list[Event]],
) -> dict[str, Any]:
    """逐 Submit 做整数闭合；v4 继续闭合两个父 span 与 worker completion。"""

    role_metric_names = _role_metrics(trace_schema_version)
    submit_partition_names = _submit_partition_metrics(trace_schema_version)
    metrics = {name: 0 for name in role_metric_names}
    for submit in submits:
        children = sorted(
            children_by_submit[submit.row_index],
            key=lambda event: (event.start_cycle, event.end_cycle, event.row_index),
        )
        counts = Counter(event.phase for event in children)
        winner = bool(submit.flags & 1)
        is_alloc = bool(submit.auxiliary)
        _validate_submit_frontend_contract(
            core_id,
            trace_schema_version,
            tensormap_mode,
            submit,
            counts,
        )
        if trace_schema_version == 5:
            tail_count = sum(counts[phase] for phase in V5_TAIL_PHASES)
            expected_tail_count = 1 if winner else 0
            if tail_count != expected_tail_count or any(
                counts[phase] > 1 for phase in V5_TAIL_PHASES
            ):
                raise ValueError(
                    f"core {core_id} task {submit.task_id} requires "
                    f"{expected_tail_count} WinnerBuild/AllocComplete tails"
                )
            tail = next(
                (event for event in children if event.phase in V5_TAIL_PHASES),
                None,
            )
            if tail is not None:
                latest_frontend_end = max(
                    event.end_cycle
                    for event in children
                    if event.phase not in V5_TAIL_PHASES
                )
                if tail.start_cycle < latest_frontend_end:
                    raise ValueError(
                        f"core {core_id} task {submit.task_id} {tail.phase} must start "
                        "at or after every preceding frontend child"
                    )
                expected_tail = "AllocComplete" if is_alloc else "WinnerBuild"
                if tail.phase != expected_tail:
                    raise ValueError(
                        f"core {core_id} task {submit.task_id} expected {expected_tail}, "
                        f"got {tail.phase}"
                    )
            expected_fanin_count = 1 if winner and not is_alloc else 0
            if counts["Fanin"] != expected_fanin_count:
                raise ValueError(
                    f"core {core_id} task {submit.task_id} "
                    f"{'winner' if winner else 'loser'} path requires "
                    f"{expected_fanin_count} Fanin spans, got {counts['Fanin']}"
                )
        for previous, current in zip(children, children[1:]):
            if current.start_cycle < previous.end_cycle:
                raise ValueError(
                    f"core {core_id} task {submit.task_id} has overlapping exclusive children "
                    f"rows {previous.row_index} ({previous.phase}) and "
                    f"{current.row_index} ({current.phase})"
                )

        child_cycles = 0
        efdrain: Event | None = None
        for child in children:
            duration = child.duration
            if duration < 0:
                raise AssertionError("converter accepted a negative child duration")
            metrics[PHASE_TO_METRIC[child.phase]] += duration
            child_cycles += duration
            if child.phase == "EfDrain":
                efdrain = child
        residual = submit.duration - child_cycles
        if residual < 0 or child_cycles + residual != submit.duration:
            raise ValueError(
                f"core {core_id} task {submit.task_id} Submit partition does not close in raw cycles"
            )
        metrics["submit_union"] += submit.duration
        metrics["submit_residual"] += residual

        assert efdrain is not None
        kernel_union = _interval_union_cycles(
            [
                (kernel.start_cycle, kernel.end_cycle)
                for kernel in kernels_by_efdrain[efdrain.row_index]
            ]
        )
        control = efdrain.duration - kernel_union
        if control < 0 or kernel_union + control != efdrain.duration:
            raise ValueError(
                f"core {core_id} task {submit.task_id} EfDrain partition does not close in raw cycles"
            )
        metrics["efdrain_kernel_union"] += kernel_union
        metrics["efdrain_control"] += control

    first_start = submits[0].start_cycle
    last_end = submits[-1].end_cycle
    between = sum(
        current.start_cycle - previous.end_cycle
        for previous, current in zip(submits, submits[1:])
    )
    envelope = last_end - first_start
    metrics["between_submit_residual"] = between
    metrics["submit_envelope"] = envelope
    if metrics["submit_union"] + between != envelope:
        raise ValueError(f"core {core_id} first/last Submit envelope does not close")
    if sum(metrics[name] for name in submit_partition_names) != metrics["submit_union"]:
        raise ValueError(f"core {core_id} aggregate Submit partition does not close")
    if (
        metrics["efdrain_kernel_union"] + metrics["efdrain_control"]
        != metrics["efdrain"]
    ):
        raise ValueError(f"core {core_id} aggregate EfDrain partition does not close")

    worker_start: int | None = None
    worker_end: int | None = None
    if trace_schema_version == 5:
        if orchestration is None or final_drain is None:
            raise ValueError(f"core {core_id} is missing schema-v5 parent spans")
        if orchestration.end_cycle != final_drain.start_cycle:
            raise ValueError(
                f"core {core_id} OrchestrationReplay.end must equal FinalDrain.start"
            )
        if not (
            orchestration.start_cycle <= first_start
            and last_end <= orchestration.end_cycle
        ):
            raise ValueError(
                f"core {core_id} Submit envelope is outside OrchestrationReplay"
            )
        setup = first_start - orchestration.start_cycle
        tail = orchestration.end_cycle - last_end
        metrics["orchestration_setup"] = setup
        metrics["orchestration_tail"] = tail
        metrics["orchestration_replay"] = orchestration.duration
        orchestration_children = (
            setup + metrics["submit_union"]
            + metrics["between_submit_residual"] + tail
        )
        if orchestration_children != orchestration.duration:
            raise ValueError(f"core {core_id} OrchestrationReplay partition does not close")

        final_kernel_union = _interval_union_cycles(
            [
                (kernel.start_cycle, kernel.end_cycle)
                for kernel in kernels_by_final_drain[final_drain.row_index]
            ]
        )
        final_residual = final_drain.duration - final_kernel_union
        if final_residual < 0 or final_kernel_union + final_residual != final_drain.duration:
            raise ValueError(f"core {core_id} FinalDrain partition does not close")
        metrics["final_drain"] = final_drain.duration
        metrics["final_drain_kernel_union"] = final_kernel_union
        metrics["final_drain_residual"] = final_residual
        metrics["worker_completion"] = final_drain.end_cycle - orchestration.start_cycle
        if (
            orchestration.duration + final_drain.duration
            != metrics["worker_completion"]
        ):
            raise ValueError(f"core {core_id} WorkerCompletion partition does not close")
        worker_start = orchestration.start_cycle
        worker_end = final_drain.end_cycle

    block_id, lane, expected_role = _standalone_topology(core_id)
    if role != expected_role:
        raise AssertionError("role passed to _analyze_core disagrees with standalone topology")
    result = {
        "core_id": core_id,
        "block_id": block_id,
        "lane": lane,
        "role": role,
        "submit_count": len(submits),
        "first_submit_start_cycle": first_start,
        "last_submit_end_cycle": last_end,
        "metrics_cycles": metrics,
    }
    if trace_schema_version == 5:
        result["worker_completion_start_cycle"] = worker_start
        result["worker_completion_end_cycle"] = worker_end
    return result


def _add_residual_segment(
    segments: dict[str, dict[str, int]],
    key: str,
    cycles: int,
    role: str,
) -> None:
    """只在小型汇总报告中累计空白来源，不向 raw/merged 增加逐事件字段。"""

    if cycles <= 0:
        return
    segment = segments.setdefault(
        key,
        {"event_count": 0, "cycles": 0, "aic_cycles": 0, "aiv_cycles": 0},
    )
    segment["event_count"] += 1
    segment["cycles"] += cycles
    segment[f"{role}_cycles"] += cycles


def _residual_breakdown(
    submits_by_lane: dict[tuple[int, int], list[Event]],
    children_by_submit: dict[int, list[Event]],
    task_kind_by_id: dict[int, int] | None,
) -> dict[str, Any]:
    """按相邻既有边界聚合 Submit 内和 Submit 间空白，保持输出规模恒定。"""

    internal_segments: dict[str, dict[str, int]] = {}
    tail_segments: dict[str, dict[str, int]] = {}
    between_segments: dict[str, dict[str, int]] = {}
    internal_total = 0
    tail_total = 0
    between_total = 0
    for (core_id, _lane), submits in submits_by_lane.items():
        role = "aic" if core_id < EXPECTED_AIC_CORES else "aiv"
        for submit in submits:
            cursor = submit.start_cycle
            previous_phase = "SubmitBegin"
            children = sorted(
                children_by_submit[submit.row_index],
                key=lambda event: (event.start_cycle, event.end_cycle, event.row_index),
            )
            for child in children:
                gap = child.start_cycle - cursor
                _add_residual_segment(
                    internal_segments,
                    f"{previous_phase}->{child.phase}",
                    gap,
                    role,
                )
                internal_total += max(gap, 0)
                cursor = child.end_cycle
                previous_phase = child.phase
            tail_gap = submit.end_cycle - cursor
            _add_residual_segment(
                tail_segments,
                f"{previous_phase}->SubmitEnd",
                tail_gap,
                role,
            )
            tail_total += max(tail_gap, 0)

        for previous, current in zip(submits, submits[1:]):
            gap = current.start_cycle - previous.end_cycle
            if task_kind_by_id is None:
                # schema-v3 只存在 private 固定五 task 流；保持历史报告逐字一致。
                previous_kind_id = previous.task_id % len(TASK_KIND_NAMES)
                current_kind_id = current.task_id % len(TASK_KIND_NAMES)
            else:
                previous_kind_id = task_kind_by_id[previous.task_id]
                current_kind_id = task_kind_by_id[current.task_id]
            previous_kind = TASK_KIND_NAMES[previous_kind_id]
            current_kind = TASK_KIND_NAMES[current_kind_id]
            _add_residual_segment(
                between_segments,
                f"{previous_kind}->{current_kind}",
                gap,
                role,
            )
            between_total += max(gap, 0)

    def ordered(segments: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
        return [
            {"boundary": key, **values}
            for key, values in sorted(
                segments.items(), key=lambda item: (-item[1]["cycles"], item[0])
            )
        ]

    return {
        "semantics": (
            "aggregate positive gaps between existing exclusive boundaries; "
            "tail remains residual and is not a standalone business phase"
        ),
        "submit_internal_residual": {
            "total_cycles": internal_total,
            "segments": ordered(internal_segments),
        },
        "submit_tail_residual": {
            "total_cycles": tail_total,
            "segments": ordered(tail_segments),
        },
        "between_submit_residual": {
            "total_cycles": between_total,
            "segments": ordered(between_segments),
        },
    }


def analyze_capture(input_path: Path) -> dict[str, Any]:
    """读取 schema-v3/v4 raw，完成全部门禁后返回可 JSON 序列化报告。"""

    input_path = Path(input_path)
    (
        frequency_hz,
        trace_schema_version,
        rows,
        core_by_block_lane,
        _base_cycle,
        metadata,
    ) = _load_and_validate(input_path)
    # 原始 b256 接近百万行；原地替换规范化 tuple，避免再同时保留一整份 Event
    # 列表。slots 也避免每条 dataclass 单独分配 __dict__。
    for index, row in enumerate(rows):
        rows[index] = _event_from_row(index, row)
    events = cast(list[Event], rows)
    core_types, num_cores = _validate_capture_identity(
        trace_schema_version, metadata, events
    )
    tensormap_mode = (
        str(metadata["tensormap_mode"])
        if trace_schema_version == 5
        else "private"
    )
    if len(core_by_block_lane) != num_cores or set(core_by_block_lane.values()) != set(
        range(num_cores)
    ):
        raise ValueError("block/lane to core mapping is incomplete")

    submits_by_lane, task_ids = _validate_and_group_submits(events)
    task_kind_by_id: dict[int, int] | None = None
    if trace_schema_version == 5:
        # 只复用 Submit 已有 won/is_alloc 两个标量恢复动态 kind；不向
        # 300 MiB 级 raw/merged 添加逐事件字段。
        submit_semantics = {
            (event.core_id, event.task_id): (
                bool(event.flags & 1),
                bool(event.auxiliary),
            )
            for event in events
            if event.phase == "Submit"
        }
        task_kind_by_id = _derive_v4_task_kinds(submit_semantics, num_cores)
    exclusive_phases = _exclusive_phases(
        trace_schema_version, tensormap_mode
    )
    submit_partition_names = _submit_partition_metrics(trace_schema_version)
    role_metric_names = _role_metrics(trace_schema_version)
    children_by_submit = _associate_exclusive_children(
        events, submits_by_lane, exclusive_phases
    )
    (
        detail_by_register,
        outputs_by_owner,
        copies_by_outputs,
        flushes_by_outputs,
        output_placement,
    ) = _associate_shared_register_details(
        events,
        children_by_submit,
        trace_schema_version,
        tensormap_mode,
    )
    register_breakdown = _build_register_breakdown(
        children_by_submit,
        detail_by_register,
        outputs_by_owner,
        copies_by_outputs,
        flushes_by_outputs,
        output_placement,
        num_cores,
        trace_schema_version,
        tensormap_mode,
    )
    materialize_breakdown = _build_materialize_breakdown(
        children_by_submit,
        outputs_by_owner,
        copies_by_outputs,
        flushes_by_outputs,
        output_placement,
        num_cores,
        trace_schema_version,
        tensormap_mode,
    )
    kernels_by_efdrain, efdrain_kernel_rows = _associate_efdrain_kernels(
        events, children_by_submit
    )

    orchestrations_by_lane: dict[tuple[int, int], list[Event]] = {}
    final_drains_by_lane: dict[tuple[int, int], list[Event]] = {}
    kernels_by_final_drain: dict[int, list[Event]] = {}
    final_drain_kernel_rows: set[int] = set()
    tail_kernel_rows: set[int] = set()
    tail_kernel_counts = {"WinnerBuild": 0, "AllocComplete": 0}
    legacy_lap_records = sum(
        event.phase in {"Build", "Replay", "Alloc"} for event in events
    )
    if trace_schema_version == 5:
        if legacy_lap_records != 0:
            raise ValueError("schema-v5 requires zero legacy Alloc/Build/Replay records")
        orchestrations_by_lane = _group_v4_parents(events, "OrchestrationReplay")
        final_drains_by_lane = _group_v4_parents(events, "FinalDrain")
        kernels_by_final_drain, final_drain_kernel_rows = _associate_kernels_to_parents(
            events, final_drains_by_lane, parent_name="FinalDrain"
        )
        _kernels_by_tail, tail_kernel_rows, tail_kernel_counts = (
            _associate_v4_tail_kernels(events, children_by_submit)
        )
        classified_sets = (
            efdrain_kernel_rows,
            tail_kernel_rows,
            final_drain_kernel_rows,
        )
        if any(
            left & right
            for index, left in enumerate(classified_sets)
            for right in classified_sets[index + 1 :]
        ):
            raise ValueError("one Kernel cannot belong to multiple schema-v5 parents")

    per_core = []
    for core_id in range(num_cores):
        lane = _standalone_topology(core_id)[1]
        per_core.append(
            _analyze_core(
                core_id,
                core_types[core_id],
                trace_schema_version,
                tensormap_mode,
                submits_by_lane[(core_id, lane)],
                children_by_submit,
                kernels_by_efdrain,
                (
                    orchestrations_by_lane[(core_id, lane)][0]
                    if trace_schema_version == 5
                    else None
                ),
                (
                    final_drains_by_lane[(core_id, lane)][0]
                    if trace_schema_version == 5
                    else None
                ),
                kernels_by_final_drain,
            )
        )

    aggregate_metrics = {
        metric: sum(core["metrics_cycles"][metric] for core in per_core)
        for metric in role_metric_names
    }
    if (
        register_breakdown is not None
        and register_breakdown["aggregate_core_work"]["metrics_cycles"]["parent"]
        != aggregate_metrics["register"]
    ):
        raise AssertionError(
            "Register breakdown parent does not match the exclusive Submit Register total"
        )
    if (
        materialize_breakdown is not None
        and materialize_breakdown[
            "aggregate_core_work"
        ]["metrics_cycles"]["parent"]
        != aggregate_metrics["materialize"]
    ):
        raise AssertionError(
            "Materialize breakdown parent does not match the exclusive "
            "Submit Materialize total"
        )
    residual_breakdown = _residual_breakdown(
        submits_by_lane, children_by_submit, task_kind_by_id
    )
    if (
        residual_breakdown["submit_internal_residual"]["total_cycles"]
        + residual_breakdown["submit_tail_residual"]["total_cycles"]
        != aggregate_metrics["submit_residual"]
    ):
        raise AssertionError("Submit internal + tail residual breakdown does not close")
    if (
        residual_breakdown["between_submit_residual"]["total_cycles"]
        != aggregate_metrics["between_submit_residual"]
    ):
        raise AssertionError("between-Submit residual boundary breakdown does not close")
    # 占比只进入小型汇总报告，不给 raw/merged 逐事件增加字段。
    residual_breakdown["submit_internal_residual"]["share_of_submit_union"] = (
        residual_breakdown["submit_internal_residual"]["total_cycles"]
        / aggregate_metrics["submit_union"]
    )
    residual_breakdown["submit_tail_residual"]["share_of_submit_union"] = (
        residual_breakdown["submit_tail_residual"]["total_cycles"]
        / aggregate_metrics["submit_union"]
    )
    residual_breakdown["between_submit_residual"]["share_of_submit_envelope"] = (
        residual_breakdown["between_submit_residual"]["total_cycles"]
        / aggregate_metrics["submit_envelope"]
    )
    submit_partition_sum = sum(
        aggregate_metrics[name] for name in submit_partition_names
    )
    envelope_partition_sum = (
        aggregate_metrics["submit_union"]
        + aggregate_metrics["between_submit_residual"]
    )
    efdrain_partition_sum = (
        aggregate_metrics["efdrain_kernel_union"]
        + aggregate_metrics["efdrain_control"]
    )
    if submit_partition_sum != aggregate_metrics["submit_union"]:
        raise AssertionError("per-core Submit closures did not preserve aggregate closure")
    if envelope_partition_sum != aggregate_metrics["submit_envelope"]:
        raise AssertionError("per-core envelope closures did not preserve aggregate closure")
    if efdrain_partition_sum != aggregate_metrics["efdrain"]:
        raise AssertionError("per-core EfDrain closures did not preserve aggregate closure")

    closure: dict[str, Any] = {
        "submit_partition": {
            "parent_cycles": aggregate_metrics["submit_union"],
            "children_plus_residual_cycles": submit_partition_sum,
            "exact": True,
        },
        "submit_envelope": {
            "parent_cycles": aggregate_metrics["submit_envelope"],
            "submit_union_plus_between_cycles": envelope_partition_sum,
            "exact": True,
        },
        "efdrain_partition": {
            "parent_cycles": aggregate_metrics["efdrain"],
            "kernel_union_plus_control_cycles": efdrain_partition_sum,
            "exact": True,
        },
    }
    if trace_schema_version == 5:
        orchestration_partition_sum = (
            aggregate_metrics["orchestration_setup"]
            + aggregate_metrics["submit_union"]
            + aggregate_metrics["between_submit_residual"]
            + aggregate_metrics["orchestration_tail"]
        )
        final_drain_partition_sum = (
            aggregate_metrics["final_drain_kernel_union"]
            + aggregate_metrics["final_drain_residual"]
        )
        worker_completion_sum = (
            aggregate_metrics["orchestration_replay"]
            + aggregate_metrics["final_drain"]
        )
        if orchestration_partition_sum != aggregate_metrics["orchestration_replay"]:
            raise AssertionError("aggregate OrchestrationReplay partition does not close")
        if final_drain_partition_sum != aggregate_metrics["final_drain"]:
            raise AssertionError("aggregate FinalDrain partition does not close")
        if worker_completion_sum != aggregate_metrics["worker_completion"]:
            raise AssertionError("aggregate WorkerCompletion partition does not close")
        closure.update(
            {
                "orchestration_replay": {
                    "parent_cycles": aggregate_metrics["orchestration_replay"],
                    "setup_submit_union_between_tail_cycles": orchestration_partition_sum,
                    "exact": True,
                },
                "final_drain": {
                    "parent_cycles": aggregate_metrics["final_drain"],
                    "kernel_union_plus_residual_cycles": final_drain_partition_sum,
                    "exact": True,
                },
                "worker_completion": {
                    "parent_cycles": aggregate_metrics["worker_completion"],
                    "orchestration_plus_final_drain_cycles": worker_completion_sum,
                    "exact": True,
                },
            }
        )

    role_statistics: dict[str, Any] = {}
    for role, expected_count in (
        ("aic", EXPECTED_AIC_CORES),
        ("aiv", EXPECTED_AIV_CORES),
    ):
        role_cores = [core for core in per_core if core["role"] == role]
        if len(role_cores) != expected_count:
            raise ValueError(
                f"role {role} has {len(role_cores)} cores, expected {expected_count}"
            )
        role_statistics[role] = {
            "core_count": len(role_cores),
            "metrics": {
                metric: _distribution(
                    [core["metrics_cycles"][metric] for core in role_cores]
                )
                for metric in role_metric_names
            },
        }

    all_submits = [
        submit for submits in submits_by_lane.values() for submit in submits
    ]
    global_start = min(submit.start_cycle for submit in all_submits)
    global_end = max(submit.end_cycle for submit in all_submits)

    overlays = {}
    for phase in OVERLAY_PHASES:
        phase_events = [event for event in events if event.phase == phase]
        overlays[phase] = {
            "event_count": len(phase_events),
            "aggregate_duration_cycles": sum(event.duration for event in phase_events),
            "included_in_additive_totals": False,
        }

    kernels = [event for event in events if event.phase == "Kernel"]
    classified_kernel_rows = (
        efdrain_kernel_rows | tail_kernel_rows | final_drain_kernel_rows
    )
    orphan_kernel_count = len(kernels) - len(classified_kernel_rows)
    if orphan_kernel_count < 0:
        raise AssertionError("Kernel containment classification is inconsistent")
    if trace_schema_version == 5 and orphan_kernel_count != 0:
        orphan_rows = sorted(
            event.row_index
            for event in kernels
            if event.row_index not in classified_kernel_rows
        )
        raise ValueError(
            "schema-v5 Kernel must be contained by EfDrain, WinnerBuild, "
            f"AllocComplete, or FinalDrain: orphan_rows={orphan_rows[:8]}"
        )

    validation: dict[str, Any] = {
        "status": "PASS",
        "dropped_records": 0,
        "core_ids_complete": True,
        "role_map_complete": True,
        "task_ids_contiguous_and_equal_per_core": True,
        "exclusive_child_task_ids_match_parent": True,
        "submit_non_overlapping_per_core_lane": True,
        "submit_partition_exact": True,
        "efdrain_partition_exact": True,
        "submit_envelope_partition_exact": True,
    }
    if trace_schema_version == 5:
        validation.update(
            {
                "orchestration_parent_exactly_one_per_core": True,
                "final_drain_parent_exactly_one_per_core": True,
                "parent_boundaries_adjacent": True,
                "legacy_lap_records": 0,
                "orchestration_partition_exact": True,
                "final_drain_partition_exact": True,
                "worker_completion_partition_exact": True,
                "kernel_unique_parent_complete": True,
            }
        )
        if tensormap_mode == "shared":
            validation.update(
                {
                    "register_publish_metadata_exactly_one_per_register": True,
                    "register_detail_identity_matches_parent": True,
                    "register_metadata_partition_exact": True,
                    "register_flat_partition_exact": True,
                    "register_details_excluded_from_submit_additive_totals": True,
                    "task_output_placement": output_placement,
                }
            )
            if output_placement == "materialize":
                validation.update(
                    {
                        "materialize_publish_task_outputs_exactly_one_per_parent": True,
                        "materialize_task_outputs_identity_matches_parent": True,
                        "materialize_partition_exact": True,
                        "materialize_task_outputs_partition_exact": True,
                        "materialize_details_excluded_from_submit_additive_totals": True,
                    }
                )
            else:
                validation.update(
                    {
                        "register_publish_task_outputs_exactly_one_per_metadata": True,
                        "register_task_outputs_identity_matches_metadata": True,
                    }
                )

    semantics: dict[str, Any] = {
        "cycle_arithmetic": "raw_integer_cycles",
        "tensormap_mode": tensormap_mode,
        "exclusive_submit_children": list(exclusive_phases),
        "submit_residual": (
            "Submit minus the union of exclusive children; exactly equals "
            "submit_internal_residual plus submit_tail_residual"
        ),
        "submit_internal_residual": (
            "unattributed prefix and child-to-child gaps inside Submit"
        ),
        "submit_tail_residual": (
            "unattributed suffix from the final exclusive child end to Submit end; "
            "not a standalone business phase"
        ),
        "between_submit_residual": (
            "unattributed gap from one Submit end to the next Submit begin"
        ),
        "efdrain_children": ["KernelUnion", "EfDrainControl"],
        "overlay_phases": list(OVERLAY_PHASES),
        "overlays_are_additive": False,
        "p95_method": "nearest_rank",
    }
    if trace_schema_version == 3:
        semantics["legacy_lap_phases"] = ["Build", "Replay", "Alloc"]
    else:
        semantics.update(
            {
                "orchestration_children": [
                    "OrchestrationSetup",
                    "SubmitUnion",
                    "BetweenSubmitResidual",
                    "OrchestrationTail",
                ],
                "final_drain_children": ["KernelUnion", "FinalDrainResidual"],
                "worker_completion_children": ["OrchestrationReplay", "FinalDrain"],
                "legacy_lap_phases_forbidden": ["Build", "Replay", "Alloc"],
                "kernel_unique_parents": [
                    "EfDrain",
                    "WinnerBuild",
                    "AllocComplete",
                    "FinalDrain",
                ],
            }
        )
        if tensormap_mode == "shared":
            semantics.update(
                {
                    "register_internal_detail": SHARED_REGISTER_DETAIL_PHASE,
                    "register_internal_children": [
                        "RegisterWaitInsertTurn",
                        "RegisterPublishWriterMetadata",
                        "RegisterPublishInsertCompletion",
                    ],
                    "task_output_placement": output_placement,
                    "register_internal_details_are_exclusive_submit_children": False,
                }
            )
            if output_placement == "materialize":
                semantics.update(
                    {
                        "materialize_internal_output_detail": (
                            SHARED_MATERIALIZE_OUTPUT_DETAIL_PHASE
                        ),
                        "materialize_internal_children": [
                            "MaterializeBeforePublishTaskOutputs",
                            "MaterializePublishTaskOutputs",
                            "MaterializeAfterPublishTaskOutputs",
                        ],
                    }
                )
            else:
                semantics.update(
                    {
                        "register_internal_output_detail": (
                            LEGACY_SHARED_REGISTER_OUTPUT_DETAIL_PHASE
                        ),
                        "register_internal_children": [
                            "RegisterWaitInsertTurn",
                            "RegisterPublishWriterMetadata",
                            "RegisterPublishTaskOutputs",
                            "RegisterPublishMetadataEpilogue",
                            "RegisterPublishInsertCompletion",
                        ],
                    }
                )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input": str(input_path),
        "capture": {
            "trace_schema_version": trace_schema_version,
            "tensormap_mode": tensormap_mode,
            "clock_freq_hz": frequency_hz,
            "core_count": num_cores,
            "task_count_per_core": len(task_ids),
            "event_count": len(events),
        },
        "validation": validation,
        "semantics": semantics,
        "global_submit_makespan": {
            "start_cycle": global_start,
            "end_cycle": global_end,
            "duration_cycles": global_end - global_start,
            "duration_us": (global_end - global_start) * 1_000_000 / frequency_hz,
            "semantics": "cross-core wall-clock envelope; not aggregate core-work",
        },
        "aggregate_core_work": {
            "metrics_cycles": aggregate_metrics,
            "closure": closure,
            "semantics": "sum of per-core cycles; not wall-clock duration",
        },
        "materialize_breakdown": materialize_breakdown,
        "register_breakdown": register_breakdown,
        "residual_breakdown": residual_breakdown,
        "per_role_core_statistics": role_statistics,
        "kernel_containment": {
            "total_events": len(kernels),
            "inside_efdrain_events": len(efdrain_kernel_rows),
            "inside_winner_build_events": tail_kernel_counts["WinnerBuild"],
            "inside_alloc_complete_events": tail_kernel_counts["AllocComplete"],
            "inside_submit_tail_events": len(tail_kernel_rows),
            "inside_final_drain_events": len(final_drain_kernel_rows),
            # v3 没有 FinalDrain/真实 tail 父边界，对其父区间外 Kernel
            # 只能标为“无 v4 边界可分类”，不冒充已证实的孤儿。
            "orphan_events": (
                orphan_kernel_count if trace_schema_version == 5 else None
            ),
            "unclassified_without_v5_parent_events": (
                orphan_kernel_count if trace_schema_version == 3 else 0
            ),
        },
        "overlays": overlays,
        "per_core": per_core,
    }


def write_analysis(input_path: Path, output_path: Path) -> Path:
    """先完成全部分析，再用同目录临时文件原子发布 JSON。"""

    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("analysis output path must differ from input raw path")
    report = analyze_capture(input_path)
    document = json.dumps(report, ensure_ascii=False, indent=2) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary.write(document)
            temporary.flush()
            os.fsync(temporary.fileno())
            os.fchmod(temporary.fileno(), 0o644)
            temporary_name = temporary.name
        os.replace(temporary_name, output_path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="schema-v3/v4 l2_swimlane_records.json")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="排他分析 JSON 输出路径",
    )
    arguments = parser.parse_args(argv)
    try:
        output = write_analysis(arguments.input, arguments.output)
    except (OSError, ValueError) as error:
        print(f"exclusive swimlane analysis failed: {error}", file=sys.stderr)
        return 1
    print(f"[SWIMLANE-EXCLUSIVE] input={arguments.input} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
