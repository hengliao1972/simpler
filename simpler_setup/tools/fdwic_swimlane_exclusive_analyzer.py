#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate a strict integer-cycle exclusive report for production FDWIC schema-v4."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from .fdwic_swimlane_schema import (
        KERNEL_EXECUTION_CHILD_PHASES,
        OVERLAY_PHASES,
        V4_EXCLUSIVE_SUBMIT_PHASES,
        Event,
        FdwicV4Model,
        find_containing_event,
        validate_and_partition_v4,
    )
except ImportError:
    from fdwic_swimlane_schema import (  # type: ignore[no-redef]
        KERNEL_EXECUTION_CHILD_PHASES,
        OVERLAY_PHASES,
        V4_EXCLUSIVE_SUBMIT_PHASES,
        Event,
        FdwicV4Model,
        find_containing_event,
        validate_and_partition_v4,
    )


REPORT_SCHEMA_VERSION = 1
PHASE_TO_METRIC = {
    "EfDrain": "efdrain",
    "Materialize": "materialize",
    "PrepareMap": "prepare_map",
    "Claim": "claim",
    "Fanin": "fanin",
    "Register": "register",
    "WinnerBuild": "winner_build",
    "AllocComplete": "alloc_complete",
    "LoserReplay": "loser_replay",
}
SUBMIT_PARTITION_METRICS = (*PHASE_TO_METRIC.values(), "submit_residual")
CORE_METRICS = (
    "submit_envelope",
    "submit_union",
    "between_submit_residual",
    *SUBMIT_PARTITION_METRICS,
    "efdrain_kernel_union",
    "efdrain_control",
    "orchestration_replay",
    "orchestration_setup",
    "orchestration_tail",
    "final_drain",
    "final_drain_kernel_union",
    "final_drain_residual",
    "worker_completion",
)


def _contains(parent: Event, child: Event) -> bool:
    return parent.start_cycle <= child.start_cycle and child.end_cycle <= parent.end_cycle


def _interval_union_cycles(intervals: Sequence[tuple[int, int]]) -> int:
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


def _median(values: Sequence[int]) -> int | float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    total = ordered[middle - 1] + ordered[middle]
    return total // 2 if total % 2 == 0 else total / 2


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    p95_index = math.ceil(0.95 * len(ordered)) - 1
    return {
        "median_cycles": _median(ordered),
        "p95_cycles": ordered[p95_index],
        "max_cycles": ordered[-1],
    }


def _classify_kernels(model: FdwicV4Model) -> tuple[dict[int, list[Event]], dict[int, list[Event]], Counter]:
    """Assign each Kernel to the smallest exclusive or top-level residual container."""

    children_by_core: dict[int, list[Event]] = {}
    submits_by_core: dict[int, list[Event]] = {}
    for core in model.cores:
        children_by_core[core.core_id] = sorted(
            [child for partition in core.submits for child in partition.children],
            key=lambda event: (event.start_cycle, event.end_cycle, event.row_index),
        )
        submits_by_core[core.core_id] = [partition.submit for partition in core.submits]
    child_starts = {core_id: [event.start_cycle for event in events] for core_id, events in children_by_core.items()}
    submit_starts = {core_id: [event.start_cycle for event in events] for core_id, events in submits_by_core.items()}

    kernels_by_child: dict[int, list[Event]] = {}
    kernels_by_final_drain: dict[int, list[Event]] = {core.core_id: [] for core in model.cores}
    counts: Counter = Counter()
    for kernel in model.kernels:
        core = model.cores[kernel.core_id]
        child = find_containing_event(
            kernel,
            children_by_core[kernel.core_id],
            child_starts[kernel.core_id],
            "exclusive child",
        )
        if child is not None:
            kernels_by_child.setdefault(child.row_index, []).append(kernel)
            counts[f"inside_{PHASE_TO_METRIC[child.phase]}_events"] += 1
            continue
        submit = find_containing_event(
            kernel,
            submits_by_core[kernel.core_id],
            submit_starts[kernel.core_id],
            "Submit",
        )
        if submit is not None:
            counts["inside_submit_residual_events"] += 1
        elif _contains(core.orchestration, kernel):
            # Production orchestration may call tensor-data access helpers,
            # which drain executable work between Submit calls. This is valid
            # production behavior absent from the standalone probe.
            counts["inside_orchestration_residual_events"] += 1
        else:
            kernels_by_final_drain[kernel.core_id].append(kernel)
            counts["inside_final_drain_events"] += 1
    counts["total_events"] = len(model.kernels)
    return kernels_by_child, kernels_by_final_drain, counts


def _new_segment() -> dict[str, int]:
    return {"event_count": 0, "cycles": 0, "aic_cycles": 0, "aiv_cycles": 0}


def _add_segment(segments: dict[str, dict[str, int]], key: str, cycles: int, role: str) -> None:
    if cycles <= 0:
        return
    segment = segments.setdefault(key, _new_segment())
    segment["event_count"] += 1
    segment["cycles"] += cycles
    segment[f"{role}_cycles"] += cycles


def _ordered_segments(segments: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    return [
        {"boundary": key, **values}
        for key, values in sorted(segments.items(), key=lambda item: (-item[1]["cycles"], item[0]))
    ]


def analyze_data(  # noqa: PLR0912, PLR0915
    data: dict[str, Any], input_path: Path | None = None
) -> dict[str, Any]:
    """Analyze the validated reader output without converting cycles to float first."""

    if int(data.get("trace_schema_version", 0)) != 4:
        raise ValueError("exclusive FDWIC analysis requires trace_schema_version=4")
    events = data.get("fdwic_events") or []
    model = validate_and_partition_v4(events, int(data.get("num_cores", 0)), data.get("core_types") or [])
    frequency_hz = int(data.get("clock_freq_hz", 0))
    if frequency_hz <= 0:
        raise ValueError("exclusive FDWIC analysis requires a positive clock frequency")

    kernels_by_child, kernels_by_final_drain, kernel_counts = _classify_kernels(model)
    internal_segments: dict[str, dict[str, int]] = {}
    tail_segments: dict[str, dict[str, int]] = {}
    between_segments: dict[str, dict[str, int]] = {}
    internal_total = 0
    tail_total = 0
    between_total = 0
    per_core = []

    for core in model.cores:
        metrics = {name: 0 for name in CORE_METRICS}
        for partition in core.submits:
            submit = partition.submit
            child_cycles = 0
            cursor = submit.start_cycle
            previous_phase = "SubmitBegin"
            for child in partition.children:
                duration = child.duration
                metrics[PHASE_TO_METRIC[child.phase]] += duration
                child_cycles += duration
                gap = child.start_cycle - cursor
                _add_segment(internal_segments, f"{previous_phase}->{child.phase}", gap, core.role)
                internal_total += gap
                cursor = child.end_cycle
                previous_phase = child.phase
                if child.phase == "EfDrain":
                    kernel_union = _interval_union_cycles(
                        [(kernel.start_cycle, kernel.end_cycle) for kernel in kernels_by_child.get(child.row_index, [])]
                    )
                    metrics["efdrain_kernel_union"] += kernel_union
                    metrics["efdrain_control"] += child.duration - kernel_union
            tail_gap = submit.end_cycle - cursor
            _add_segment(tail_segments, f"{previous_phase}->SubmitEnd", tail_gap, core.role)
            tail_total += tail_gap

            residual = submit.duration - child_cycles
            if residual < 0:
                raise ValueError(f"core {core.core_id} task {submit.task_id} Submit partition is negative")
            metrics["submit_union"] += submit.duration
            metrics["submit_residual"] += residual

        for previous, current in zip(core.submits, core.submits[1:]):
            gap = current.submit.start_cycle - previous.submit.end_cycle
            previous_kind = "alloc" if previous.submit.auxiliary else "kernel"
            current_kind = "alloc" if current.submit.auxiliary else "kernel"
            _add_segment(between_segments, f"{previous_kind}->{current_kind}", gap, core.role)
            between_total += gap

        if core.submits:
            first_start = core.submits[0].submit.start_cycle
            last_end = core.submits[-1].submit.end_cycle
            metrics["submit_envelope"] = last_end - first_start
            metrics["between_submit_residual"] = sum(
                current.submit.start_cycle - previous.submit.end_cycle
                for previous, current in zip(core.submits, core.submits[1:])
            )
            metrics["orchestration_setup"] = first_start - core.orchestration.start_cycle
            metrics["orchestration_tail"] = core.orchestration.end_cycle - last_end
        else:
            first_start = None
            last_end = None
            metrics["orchestration_setup"] = core.orchestration.duration

        metrics["orchestration_replay"] = core.orchestration.duration
        final_kernel_union = _interval_union_cycles(
            [(kernel.start_cycle, kernel.end_cycle) for kernel in kernels_by_final_drain[core.core_id]]
        )
        metrics["final_drain"] = core.final_drain.duration
        metrics["final_drain_kernel_union"] = final_kernel_union
        metrics["final_drain_residual"] = core.final_drain.duration - final_kernel_union
        metrics["worker_completion"] = core.final_drain.end_cycle - core.orchestration.start_cycle

        if sum(metrics[name] for name in SUBMIT_PARTITION_METRICS) != metrics["submit_union"]:
            raise ValueError(f"core {core.core_id} aggregate Submit partition does not close")
        if metrics["submit_union"] + metrics["between_submit_residual"] != metrics["submit_envelope"]:
            raise ValueError(f"core {core.core_id} Submit envelope does not close")
        if metrics["efdrain_kernel_union"] + metrics["efdrain_control"] != metrics["efdrain"]:
            raise ValueError(f"core {core.core_id} EfDrain partition does not close")
        orchestration_sum = (
            metrics["orchestration_setup"]
            + metrics["submit_union"]
            + metrics["between_submit_residual"]
            + metrics["orchestration_tail"]
        )
        if orchestration_sum != metrics["orchestration_replay"]:
            raise ValueError(f"core {core.core_id} OrchestrationReplay partition does not close")
        if metrics["final_drain_kernel_union"] + metrics["final_drain_residual"] != metrics["final_drain"]:
            raise ValueError(f"core {core.core_id} FinalDrain partition does not close")
        if metrics["orchestration_replay"] + metrics["final_drain"] != metrics["worker_completion"]:
            raise ValueError(f"core {core.core_id} WorkerCompletion partition does not close")

        per_core.append(
            {
                "core_id": core.core_id,
                "block_id": core.block_id,
                "lane": core.lane,
                "role": core.role,
                "submit_count": len(core.submits),
                "first_submit_start_cycle": first_start,
                "last_submit_end_cycle": last_end,
                "worker_completion_start_cycle": core.orchestration.start_cycle,
                "worker_completion_end_cycle": core.final_drain.end_cycle,
                "metrics_cycles": metrics,
            }
        )

    aggregate_metrics = {metric: sum(core["metrics_cycles"][metric] for core in per_core) for metric in CORE_METRICS}
    submit_partition_sum = sum(aggregate_metrics[name] for name in SUBMIT_PARTITION_METRICS)
    orchestration_partition_sum = (
        aggregate_metrics["orchestration_setup"]
        + aggregate_metrics["submit_union"]
        + aggregate_metrics["between_submit_residual"]
        + aggregate_metrics["orchestration_tail"]
    )
    final_drain_partition_sum = (
        aggregate_metrics["final_drain_kernel_union"] + aggregate_metrics["final_drain_residual"]
    )
    worker_completion_sum = aggregate_metrics["orchestration_replay"] + aggregate_metrics["final_drain"]
    if internal_total + tail_total != aggregate_metrics["submit_residual"]:
        raise AssertionError("Submit internal + tail residual breakdown does not close")
    if between_total != aggregate_metrics["between_submit_residual"]:
        raise AssertionError("between-Submit residual breakdown does not close")

    role_statistics = {}
    for role in ("aic", "aiv"):
        role_cores = [core for core in per_core if core["role"] == role]
        if not role_cores:
            continue
        role_statistics[role] = {
            "core_count": len(role_cores),
            "metrics": {
                metric: _distribution([core["metrics_cycles"][metric] for core in role_cores])
                for metric in CORE_METRICS
            },
        }

    all_submits = [partition.submit for core in model.cores for partition in core.submits]
    worker_start = min(core.orchestration.start_cycle for core in model.cores)
    worker_end = max(core.final_drain.end_cycle for core in model.cores)
    global_submit_makespan = None
    if all_submits:
        submit_start = min(submit.start_cycle for submit in all_submits)
        submit_end = max(submit.end_cycle for submit in all_submits)
        global_submit_makespan = {
            "start_cycle": submit_start,
            "end_cycle": submit_end,
            "duration_cycles": submit_end - submit_start,
            "duration_us": (submit_end - submit_start) * 1_000_000 / frequency_hz,
            "semantics": "cross-core wall-clock Submit envelope; not aggregate core-work",
        }

    overlays = {
        phase: {
            **model.overlay_statistics[phase],
            "included_in_additive_totals": False,
        }
        for phase in OVERLAY_PHASES
    }
    residual_breakdown = {
        "submit_internal_residual": {
            "total_cycles": internal_total,
            "segments": _ordered_segments(internal_segments),
        },
        "submit_tail_residual": {
            "total_cycles": tail_total,
            "segments": _ordered_segments(tail_segments),
        },
        "between_submit_residual": {
            "total_cycles": between_total,
            "segments": _ordered_segments(between_segments),
        },
    }
    if aggregate_metrics["submit_union"]:
        residual_breakdown["submit_internal_residual"]["share_of_submit_union"] = (
            internal_total / aggregate_metrics["submit_union"]
        )
        residual_breakdown["submit_tail_residual"]["share_of_submit_union"] = (
            tail_total / aggregate_metrics["submit_union"]
        )
    if aggregate_metrics["submit_envelope"]:
        residual_breakdown["between_submit_residual"]["share_of_submit_envelope"] = (
            between_total / aggregate_metrics["submit_envelope"]
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "input": str(input_path) if input_path is not None else None,
        "capture": {
            "trace_schema_version": 4,
            "clock_freq_hz": frequency_hz,
            "core_count": len(model.cores),
            "task_count_per_core": len(model.task_ids),
            "event_count": model.event_count,
        },
        "validation": {
            "status": "PASS",
            "dropped_records": 0,
            "physical_topology_complete": True,
            "task_stream_contiguous_and_equal_per_core": True,
            "orchestration_parent_exactly_one_per_core": True,
            "final_drain_parent_exactly_one_per_core": True,
            "parent_boundaries_adjacent": True,
            "legacy_lap_records": 0,
            "exclusive_children_non_overlapping": True,
            "all_integer_cycle_closures_exact": True,
        },
        "semantics": {
            "cycle_arithmetic": "raw_integer_cycles",
            "exclusive_submit_children": sorted(V4_EXCLUSIVE_SUBMIT_PHASES),
            "kernel_execution_submit_children": sorted(KERNEL_EXECUTION_CHILD_PHASES),
            "kernel_execution_top_level_residual": "OrchestrationReplay outside Submit, or FinalDrain",
            "submit_boundary": (
                "starts after dist_submit_begin and ends at the final timestamp before Submit record publication/return"
            ),
            "submit_residual": "Submit minus its non-overlapping exclusive children",
            "submit_residual_contents": (
                "unmarked control and trace-record publication overhead; Kernel execution is forbidden"
            ),
            "orchestration_setup": (
                "OrchestrationReplay.begin to first Submit.begin; includes setup and first dist_submit_begin"
            ),
            "between_submit_residual": (
                "one Submit.end to the next Submit.begin; includes record/return, orchestration work, "
                "and next dist_submit_begin"
            ),
            "orchestration_tail": (
                "last Submit.end to OrchestrationReplay.end; includes final record/return and orchestration epilogue"
            ),
            "worker_completion_boundary": (
                "post-startup OrchestrationReplay.begin through FinalDrain.end; excludes startup and post-window "
                "clock baselines, trace flush, and finish publication"
            ),
            "orchestration_children": [
                "OrchestrationSetup",
                "SubmitUnion",
                "BetweenSubmitResidual",
                "OrchestrationTail",
            ],
            "final_drain_children": ["KernelUnion", "FinalDrainResidual"],
            "worker_completion_children": ["OrchestrationReplay", "FinalDrain"],
            "legacy_lap_phases_forbidden": ["Alloc", "Build", "Replay"],
            "drain_won": "real nested BlockWon action; retained as a non-additive overlay",
            "loser_replay": "real kernel-loser drain_block_won call; exclusive Submit child",
            "alloc_loser_tail": "no fabricated action; remains Submit tail residual",
            "overlays_are_additive": False,
            "p95_method": "nearest_rank",
        },
        "global_worker_completion_makespan": {
            "start_cycle": worker_start,
            "end_cycle": worker_end,
            "duration_cycles": worker_end - worker_start,
            "duration_us": (worker_end - worker_start) * 1_000_000 / frequency_hz,
            "semantics": "cross-core wall-clock envelope; not aggregate core-work",
        },
        "global_submit_makespan": global_submit_makespan,
        "aggregate_core_work": {
            "metrics_cycles": aggregate_metrics,
            "closure": {
                "submit_partition": {
                    "parent_cycles": aggregate_metrics["submit_union"],
                    "children_plus_residual_cycles": submit_partition_sum,
                    "exact": submit_partition_sum == aggregate_metrics["submit_union"],
                },
                "submit_envelope": {
                    "parent_cycles": aggregate_metrics["submit_envelope"],
                    "submit_union_plus_between_cycles": (
                        aggregate_metrics["submit_union"] + aggregate_metrics["between_submit_residual"]
                    ),
                    "exact": True,
                },
                "efdrain_partition": {
                    "parent_cycles": aggregate_metrics["efdrain"],
                    "kernel_union_plus_control_cycles": (
                        aggregate_metrics["efdrain_kernel_union"] + aggregate_metrics["efdrain_control"]
                    ),
                    "exact": True,
                },
                "orchestration_replay": {
                    "parent_cycles": aggregate_metrics["orchestration_replay"],
                    "setup_submit_union_between_tail_cycles": orchestration_partition_sum,
                    "exact": orchestration_partition_sum == aggregate_metrics["orchestration_replay"],
                },
                "final_drain": {
                    "parent_cycles": aggregate_metrics["final_drain"],
                    "kernel_union_plus_residual_cycles": final_drain_partition_sum,
                    "exact": final_drain_partition_sum == aggregate_metrics["final_drain"],
                },
                "worker_completion": {
                    "parent_cycles": aggregate_metrics["worker_completion"],
                    "orchestration_plus_final_drain_cycles": worker_completion_sum,
                    "exact": worker_completion_sum == aggregate_metrics["worker_completion"],
                },
            },
            "semantics": "sum of per-core cycles; not wall-clock duration",
        },
        "residual_breakdown": residual_breakdown,
        "per_role_core_statistics": role_statistics,
        "kernel_containment": dict(kernel_counts),
        "overlays": overlays,
        "per_core": per_core,
    }


def analyze_capture(input_path: Path) -> dict[str, Any]:
    try:
        from .swimlane_converter import read_perf_data  # noqa: PLC0415
    except ImportError:
        from swimlane_converter import read_perf_data  # type: ignore[no-redef]  # noqa: PLC0415

    input_path = Path(input_path)
    return analyze_data(read_perf_data(input_path), input_path)


def write_analysis_data(data: dict[str, Any], input_path: Path, output_path: Path) -> Path:
    output_path = Path(output_path)
    input_path = Path(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("exclusive analysis output must differ from the raw input")
    document = json.dumps(analyze_data(data, input_path), ensure_ascii=False, indent=2) + "\n"
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


def write_analysis(input_path: Path, output_path: Path) -> Path:
    try:
        from .swimlane_converter import read_perf_data  # noqa: PLC0415
    except ImportError:
        from swimlane_converter import read_perf_data  # type: ignore[no-redef]  # noqa: PLC0415

    input_path = Path(input_path)
    return write_analysis_data(read_perf_data(input_path), input_path, output_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="schema-v4 l2_swimlane_records.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output path (default: <input-dir>/swimlane_exclusive_analysis.json)",
    )
    arguments = parser.parse_args(argv)
    output = arguments.output or arguments.input.parent / "swimlane_exclusive_analysis.json"
    try:
        write_analysis(arguments.input, output)
    except (OSError, ValueError) as error:
        print(f"exclusive FDWIC swimlane analysis failed: {error}", file=sys.stderr)
        return 1
    print(f"[SWIMLANE-EXCLUSIVE] input={arguments.input} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
