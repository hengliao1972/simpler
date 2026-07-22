#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the LICENSE file.
# -----------------------------------------------------------------------------------------------------------
"""Build one provenance-aware FDWIC Submit span overview from independent evidence ELFs."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .fdwic_submit_pmu_report import (
        ALLOC_COMPLETE_CAPTURE_MODE,
        ARG_BUILD_CAPTURE_MODE,
        CLAIM_CAPTURE_MODE,
        DEFAULT_INPUT_NAME,
        DEFAULT_PROVENANCE_NAME,
        EFDRAIN_CONTROL_CAPTURE_MODE,
        EMPTY_BRACKET_CAPTURE_MODE,
        FANIN_CAPTURE_MODE,
        LOSER_REPLAY_CAPTURE_MODE,
        MATERIALIZE_CAPTURE_MODE,
        NONE_CAPTURE_MODE,
        PREPARE_MAP_CAPTURE_MODE,
        REGISTER_CAPTURE_MODE,
        SUBMIT_TRANSITION_CAPTURE_MODE,
        WINNER_BUILD_CAPTURE_MODE,
        SubmitPmuCapture,
        load_capture,
        load_provenance,
    )
    from .fdwic_swimlane_exclusive_analyzer import CORE_METRICS, REPORT_SCHEMA_VERSION, analyze_capture
except ImportError:
    from fdwic_submit_pmu_report import (  # type: ignore[no-redef]
        ALLOC_COMPLETE_CAPTURE_MODE,
        ARG_BUILD_CAPTURE_MODE,
        CLAIM_CAPTURE_MODE,
        DEFAULT_INPUT_NAME,
        DEFAULT_PROVENANCE_NAME,
        EFDRAIN_CONTROL_CAPTURE_MODE,
        EMPTY_BRACKET_CAPTURE_MODE,
        FANIN_CAPTURE_MODE,
        LOSER_REPLAY_CAPTURE_MODE,
        MATERIALIZE_CAPTURE_MODE,
        NONE_CAPTURE_MODE,
        PREPARE_MAP_CAPTURE_MODE,
        REGISTER_CAPTURE_MODE,
        SUBMIT_TRANSITION_CAPTURE_MODE,
        WINNER_BUILD_CAPTURE_MODE,
        SubmitPmuCapture,
        load_capture,
        load_provenance,
    )
    from fdwic_swimlane_exclusive_analyzer import (  # type: ignore[no-redef]
        CORE_METRICS,
        REPORT_SCHEMA_VERSION,
        analyze_capture,
    )


OVERVIEW_SCHEMA = "fdwic-submit-span-overview-v1"
DEFAULT_JSON_NAME = "fdwic_submit_span_overview.json"
DEFAULT_HTML_NAME = "fdwic_submit_span_overview.html"
OUTPUT_LOCK_NAME = ".fdwic_submit_span_overview.lock"

PMU_MODE_ORDER = (
    NONE_CAPTURE_MODE,
    EFDRAIN_CONTROL_CAPTURE_MODE,
    CLAIM_CAPTURE_MODE,
    ARG_BUILD_CAPTURE_MODE,
    MATERIALIZE_CAPTURE_MODE,
    PREPARE_MAP_CAPTURE_MODE,
    FANIN_CAPTURE_MODE,
    REGISTER_CAPTURE_MODE,
    WINNER_BUILD_CAPTURE_MODE,
    ALLOC_COMPLETE_CAPTURE_MODE,
    LOSER_REPLAY_CAPTURE_MODE,
    SUBMIT_TRANSITION_CAPTURE_MODE,
    EMPTY_BRACKET_CAPTURE_MODE,
)
EXPECTED_PMU_MODES = frozenset(PMU_MODE_ORDER)

PHASE_PRESENTATION = {
    EFDRAIN_CONTROL_CAPTURE_MODE: ("EfDrain Scalar control", "EfDrain", "control-only"),
    CLAIM_CAPTURE_MODE: ("Claim", "Claim", "same-business-boundary"),
    ARG_BUILD_CAPTURE_MODE: ("ArgBuild", "Claim→Materialize internal residual", "residual-boundary"),
    MATERIALIZE_CAPTURE_MODE: ("Materialize", "Materialize", "same-business-boundary"),
    PREPARE_MAP_CAPTURE_MODE: ("PrepareMap", "PrepareMap", "same-business-boundary"),
    FANIN_CAPTURE_MODE: ("Fanin", "Fanin", "same-business-boundary"),
    REGISTER_CAPTURE_MODE: ("Register", "Register", "same-business-boundary"),
    WINNER_BUILD_CAPTURE_MODE: ("WinnerBuild Scalar control", "WinnerBuild", "control-only"),
    ALLOC_COMPLETE_CAPTURE_MODE: ("AllocComplete Scalar control", "AllocComplete", "control-only"),
    LOSER_REPLAY_CAPTURE_MODE: ("LoserReplay", "LoserReplay", "same-business-boundary"),
    SUBMIT_TRANSITION_CAPTURE_MODE: (
        "SubmitTransition",
        "BetweenSubmitResidual",
        "adjacent-submit-boundary",
    ),
    EMPTY_BRACKET_CAPTURE_MODE: ("EmptyBracket", "—", "observer-calibration"),
}

SUBMIT_PHASES = (
    ("EfDrain", "efdrain"),
    ("Materialize", "materialize"),
    ("PrepareMap", "prepare_map"),
    ("Claim", "claim"),
    ("Fanin", "fanin"),
    ("Register", "register"),
    ("WinnerBuild", "winner_build"),
    ("AllocComplete", "alloc_complete"),
    ("LoserReplay", "loser_replay"),
)


def _fail(message: str) -> None:
    raise ValueError(message)


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _require_int(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{path} must be an integer >= {minimum}")
    return value


def _require_number(value: Any, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        _fail(f"{path} must be a finite number >= {minimum}")
    return float(value)


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{path} must be an object")
    return value


def _validate_swimlane_analysis(data: Mapping[str, Any]) -> tuple[int, int]:  # noqa: PLR0912
    """Reject a partial or non-closing schema-v4 analysis before summarizing it."""

    if data.get("schema_version") != REPORT_SCHEMA_VERSION:
        _fail(f"swimlane analysis schema_version must equal {REPORT_SCHEMA_VERSION}")
    validation = _require_mapping(data.get("validation"), "swimlane.validation")
    required_validation = {
        "status": "PASS",
        "dropped_records": 0,
        "physical_topology_complete": True,
        "task_stream_contiguous_and_equal_per_core": True,
        "orchestration_parent_exactly_one_per_core": True,
        "final_drain_parent_exactly_one_per_core": True,
        "parent_boundaries_adjacent": True,
        "exclusive_children_non_overlapping": True,
        "all_integer_cycle_closures_exact": True,
    }
    for field, expected in required_validation.items():
        if validation.get(field) != expected:
            _fail(f"swimlane.validation.{field} must equal {expected!r}")

    capture = _require_mapping(data.get("capture"), "swimlane.capture")
    if capture.get("trace_schema_version") != 4:
        _fail("swimlane.capture.trace_schema_version must equal 4")
    frequency_hz = _require_int(capture.get("clock_freq_hz"), "swimlane.capture.clock_freq_hz", minimum=1)
    core_count = _require_int(capture.get("core_count"), "swimlane.capture.core_count", minimum=1)
    if core_count != 96:
        _fail("swimlane capture must contain exactly 96 logical cores")
    task_count = _require_int(capture.get("task_count_per_core"), "swimlane.capture.task_count_per_core", minimum=1)

    aggregate = _require_mapping(data.get("aggregate_core_work"), "swimlane.aggregate_core_work")
    metrics = _require_mapping(aggregate.get("metrics_cycles"), "swimlane.aggregate_core_work.metrics_cycles")
    aggregate_values = {
        metric: _require_int(metrics.get(metric), f"swimlane.aggregate_core_work.metrics_cycles.{metric}")
        for metric in CORE_METRICS
    }
    closures = _require_mapping(aggregate.get("closure"), "swimlane.aggregate_core_work.closure")
    required_closures = {
        "submit_partition",
        "submit_envelope",
        "efdrain_partition",
        "orchestration_replay",
        "final_drain",
        "worker_completion",
    }
    if set(closures) != required_closures:
        _fail("swimlane aggregate closure set is incomplete or unexpected")
    for name, closure_value in closures.items():
        closure = _require_mapping(closure_value, f"swimlane.aggregate_core_work.closure.{name}")
        if closure.get("exact") is not True:
            _fail(f"swimlane aggregate closure {name!r} is not exact")

    per_core_value = data.get("per_core")
    if not isinstance(per_core_value, list) or len(per_core_value) != core_count:
        _fail("swimlane.per_core must contain exactly 96 records")
    roles: Counter[str] = Counter()
    core_ids: set[int] = set()
    recomputed = {metric: 0 for metric in CORE_METRICS}
    for index, core_value in enumerate(per_core_value):
        core = _require_mapping(core_value, f"swimlane.per_core[{index}]")
        core_id = _require_int(core.get("core_id"), f"swimlane.per_core[{index}].core_id")
        core_ids.add(core_id)
        role = core.get("role")
        if role not in {"aic", "aiv"}:
            _fail(f"swimlane.per_core[{index}].role must be 'aic' or 'aiv'")
        roles[str(role)] += 1
        if core.get("submit_count") != task_count:
            _fail(f"swimlane.per_core[{index}].submit_count does not match task_count_per_core")
        core_metrics = _require_mapping(core.get("metrics_cycles"), f"swimlane.per_core[{index}].metrics_cycles")
        for metric in CORE_METRICS:
            recomputed[metric] += _require_int(
                core_metrics.get(metric), f"swimlane.per_core[{index}].metrics_cycles.{metric}"
            )
    if core_ids != set(range(core_count)):
        _fail("swimlane logical core IDs must be contiguous 0..95")
    if roles != Counter({"aic": 32, "aiv": 64}):
        _fail("swimlane topology must contain exactly 32 AIC and 64 AIV cores")
    if recomputed != aggregate_values:
        _fail("swimlane aggregate metrics do not equal the sum of per-core metrics")

    residual = _require_mapping(data.get("residual_breakdown"), "swimlane.residual_breakdown")
    internal = _require_mapping(residual.get("submit_internal_residual"), "swimlane residual internal")
    tail = _require_mapping(residual.get("submit_tail_residual"), "swimlane residual tail")
    if (
        _require_int(internal.get("total_cycles"), "swimlane residual internal total")
        + _require_int(tail.get("total_cycles"), "swimlane residual tail total")
        != aggregate_values["submit_residual"]
    ):
        _fail("swimlane internal + tail residual does not close to submit_residual")

    overlays = _require_mapping(data.get("overlays"), "swimlane.overlays")
    for name, overlay_value in overlays.items():
        overlay = _require_mapping(overlay_value, f"swimlane.overlays.{name}")
        if overlay.get("included_in_additive_totals") is not False:
            _fail(f"swimlane overlay {name!r} must remain non-additive")

    makespan = _require_mapping(data.get("global_submit_makespan"), "swimlane.global_submit_makespan")
    makespan_cycles = _require_int(makespan.get("duration_cycles"), "swimlane global Submit duration", minimum=1)
    makespan_us = _require_number(makespan.get("duration_us"), "swimlane global Submit duration_us")
    expected_us = makespan_cycles * 1_000_000 / frequency_hz
    if not math.isclose(makespan_us, expected_us, rel_tol=1e-12, abs_tol=1e-9):
        _fail("swimlane global Submit duration_us does not match its SYS counter frequency")
    return frequency_hz, task_count


def _ticks_to_us(ticks: int | float, frequency_hz: int) -> float:
    return float(ticks) * 1_000_000 / frequency_hz


def _metric_range(data: Mapping[str, Any]) -> dict[str, int]:
    return {"min": int(data["min"]), "max": int(data["max"])}


def _role_metric_range(
    per_core: Sequence[Mapping[str, Any]], metric: str, role: str, frequency_hz: int
) -> dict[str, int | float]:
    values = [int(core["metrics_cycles"][metric]) for core in per_core if core["role"] == role]
    return {
        "min_ticks": min(values),
        "max_ticks": max(values),
        "min_us": _ticks_to_us(min(values), frequency_hz),
        "max_us": _ticks_to_us(max(values), frequency_hz),
    }


def _swimlane_metric_row(
    label: str,
    metric: str,
    metrics: Mapping[str, int],
    denominator: int,
    per_core: Sequence[Mapping[str, Any]],
    frequency_hz: int,
) -> dict[str, Any]:
    ticks = int(metrics[metric])
    return {
        "label": label,
        "metric": metric,
        "core_time_ticks": ticks,
        "core_time_us": _ticks_to_us(ticks, frequency_hz),
        "share": ticks / denominator if denominator else 0.0,
        "aic_per_core": _role_metric_range(per_core, metric, "aic", frequency_hz),
        "aiv_per_core": _role_metric_range(per_core, metric, "aiv", frequency_hz),
    }


def _residual_row(
    label: str,
    metric: str,
    ticks: int,
    denominator: int,
    frequency_hz: int,
) -> dict[str, Any]:
    return {
        "label": label,
        "metric": metric,
        "core_time_ticks": ticks,
        "core_time_us": _ticks_to_us(ticks, frequency_hz),
        "share": ticks / denominator if denominator else 0.0,
        "aic_per_core": None,
        "aiv_per_core": None,
    }


def _partition(name: str, parent_metric: str, parent_ticks: int, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    children_ticks = sum(int(row["core_time_ticks"]) for row in rows)
    if children_ticks != parent_ticks:
        _fail(f"overview partition {name!r} does not close: parent={parent_ticks} children={children_ticks}")
    return {
        "name": name,
        "parent_metric": parent_metric,
        "parent_core_time_ticks": parent_ticks,
        "children_core_time_ticks": children_ticks,
        "exact": True,
        "rows": list(rows),
    }


def _summarize_swimlane(
    analysis: Mapping[str, Any], raw_path: Path, raw_size: int, raw_sha256: str, frequency_hz: int
) -> dict[str, Any]:
    aggregate = analysis["aggregate_core_work"]["metrics_cycles"]
    metrics = {name: int(value) for name, value in aggregate.items()}
    per_core = analysis["per_core"]
    residual = analysis["residual_breakdown"]

    envelope_rows = [
        _swimlane_metric_row(
            "SubmitUnion", "submit_union", metrics, metrics["submit_envelope"], per_core, frequency_hz
        ),
        _swimlane_metric_row(
            "BetweenSubmitResidual",
            "between_submit_residual",
            metrics,
            metrics["submit_envelope"],
            per_core,
            frequency_hz,
        ),
    ]
    internal_ticks = int(residual["submit_internal_residual"]["total_cycles"])
    tail_ticks = int(residual["submit_tail_residual"]["total_cycles"])
    submit_rows = [
        *(
            _swimlane_metric_row(label, metric, metrics, metrics["submit_union"], per_core, frequency_hz)
            for label, metric in SUBMIT_PHASES
        ),
        _residual_row(
            "SubmitInternalResidual",
            "submit_internal_residual",
            internal_ticks,
            metrics["submit_union"],
            frequency_hz,
        ),
        _residual_row(
            "SubmitTailResidual",
            "submit_tail_residual",
            tail_ticks,
            metrics["submit_union"],
            frequency_hz,
        ),
    ]
    efdrain_rows = [
        _swimlane_metric_row(
            "EfDrainKernelUnion",
            "efdrain_kernel_union",
            metrics,
            metrics["efdrain"],
            per_core,
            frequency_hz,
        ),
        _swimlane_metric_row("EfDrainControl", "efdrain_control", metrics, metrics["efdrain"], per_core, frequency_hz),
    ]
    orchestration_rows = [
        _swimlane_metric_row(
            "OrchestrationSetup",
            "orchestration_setup",
            metrics,
            metrics["orchestration_replay"],
            per_core,
            frequency_hz,
        ),
        _swimlane_metric_row(
            "SubmitUnion", "submit_union", metrics, metrics["orchestration_replay"], per_core, frequency_hz
        ),
        _swimlane_metric_row(
            "BetweenSubmitResidual",
            "between_submit_residual",
            metrics,
            metrics["orchestration_replay"],
            per_core,
            frequency_hz,
        ),
        _swimlane_metric_row(
            "OrchestrationTail",
            "orchestration_tail",
            metrics,
            metrics["orchestration_replay"],
            per_core,
            frequency_hz,
        ),
    ]
    final_drain_rows = [
        _swimlane_metric_row(
            "FinalDrainKernelUnion",
            "final_drain_kernel_union",
            metrics,
            metrics["final_drain"],
            per_core,
            frequency_hz,
        ),
        _swimlane_metric_row(
            "FinalDrainResidual",
            "final_drain_residual",
            metrics,
            metrics["final_drain"],
            per_core,
            frequency_hz,
        ),
    ]
    worker_rows = [
        _swimlane_metric_row(
            "OrchestrationReplay",
            "orchestration_replay",
            metrics,
            metrics["worker_completion"],
            per_core,
            frequency_hz,
        ),
        _swimlane_metric_row(
            "FinalDrain", "final_drain", metrics, metrics["worker_completion"], per_core, frequency_hz
        ),
    ]

    residual_segments = {}
    for name in ("submit_internal_residual", "submit_tail_residual", "between_submit_residual"):
        item = residual[name]
        residual_segments[name] = {
            "total_cycles": int(item["total_cycles"]),
            "segments": [
                {
                    "boundary": str(segment["boundary"]),
                    "event_count": int(segment["event_count"]),
                    "cycles": int(segment["cycles"]),
                    "aic_cycles": int(segment["aic_cycles"]),
                    "aiv_cycles": int(segment["aiv_cycles"]),
                }
                for segment in item["segments"]
            ],
        }

    return {
        "source": {
            "raw_path": str(raw_path.resolve()),
            "raw_size": raw_size,
            "raw_sha256": raw_sha256,
            "build_provenance_available": False,
            "identity_limit": (
                "schema-v4 raw was strictly reanalyzed, but this capture has no build-provenance sidecar"
            ),
        },
        "sys_counter_frequency_hz": frequency_hz,
        "sys_counter_semantics": "SYS counter ticks; not PMU cycles",
        "global_submit_makespan": analysis["global_submit_makespan"],
        "global_worker_completion_makespan": analysis["global_worker_completion_makespan"],
        "submit_envelope_partition": _partition(
            "SubmitEnvelope", "submit_envelope", metrics["submit_envelope"], envelope_rows
        ),
        "submit_union_partition": _partition("SubmitUnion", "submit_union", metrics["submit_union"], submit_rows),
        "efdrain_partition": _partition("EfDrain", "efdrain", metrics["efdrain"], efdrain_rows),
        "orchestration_partition": _partition(
            "OrchestrationReplay", "orchestration_replay", metrics["orchestration_replay"], orchestration_rows
        ),
        "final_drain_partition": _partition("FinalDrain", "final_drain", metrics["final_drain"], final_drain_rows),
        "worker_completion_partition": _partition(
            "WorkerCompletion", "worker_completion", metrics["worker_completion"], worker_rows
        ),
        "submit_residual_per_core": {
            "aic": _role_metric_range(per_core, "submit_residual", "aic", frequency_hz),
            "aiv": _role_metric_range(per_core, "submit_residual", "aiv", frequency_hz),
        },
        "residual_segments": residual_segments,
        "kernel_containment": analysis["kernel_containment"],
        "overlays": analysis["overlays"],
    }


def _group_denominator(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cores": int(summary["cores"]),
        "scalar_submit_elapsed_ticks": {
            "sum": int(summary["scalar_submit_elapsed_ticks"]["sum"]),
            **_metric_range(summary["scalar_submit_elapsed_ticks"]),
        },
        "pmu_total_cycles": {
            "sum": int(summary["total_cycles"]["sum"]),
            **_metric_range(summary["total_cycles"]),
        },
        "scalar_busy_cycles": {
            "sum": int(summary["scalar_busy"]["sum"]),
            **_metric_range(summary["scalar_busy"]),
        },
        "primary_icache_requests": {
            "sum": int(summary["icache_requests"]["sum"]),
            **_metric_range(summary["icache_requests"]),
        },
        "primary_icache_misses": {
            "sum": int(summary["icache_misses"]["sum"]),
            **_metric_range(summary["icache_misses"]),
        },
    }


def _phase_group_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cores": int(summary["cores"]),
        "phase_elapsed_ticks": {
            "sum": int(summary["phase_elapsed_ticks"]["sum"]),
            **_metric_range(summary["phase_elapsed_ticks"]),
        },
        "phase_icache_requests_observed": {
            "sum": int(summary["phase_icache_requests_observed"]["sum"]),
            **_metric_range(summary["phase_icache_requests_observed"]),
        },
        "phase_icache_misses_observed": {
            "sum": int(summary["phase_icache_misses_observed"]["sum"]),
            **_metric_range(summary["phase_icache_misses_observed"]),
        },
        "phase_time_share_of_scalar_submit": float(summary["phase_time_share_of_scalar_submit"]),
        "phase_request_observed_share_of_primary": float(summary["phase_request_observed_share_of_primary"]),
        "phase_miss_observed_share_of_primary": float(summary["phase_miss_observed_share_of_primary"]),
        "phase_business_calls": int(summary["phase_business_calls"]),
        "phase_calls_per_core": _metric_range(summary["phase_calls_per_core"]),
        "phase_zero_call_cores": int(summary["phase_zero_call_cores"]),
        "phase_excluded_kernel_calls": int(summary.get("phase_excluded_kernel_calls", 0)),
    }


def _summarize_pmu_capture(
    capture: SubmitPmuCapture,
    provenance: Mapping[str, Any],
    provenance_sha256: str,
    provenance_path: Path,
) -> dict[str, Any]:
    mode = str(capture.data["capture"]["mode"])
    source = {
        "raw_path": str(capture.input_path.resolve()),
        "raw_size": capture.raw_size,
        "raw_sha256": capture.raw_sha256,
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": provenance_sha256,
        "html_report_path": str(capture.input_path.with_name("fdwic_submit_pmu_report.html").resolve()),
        "git_head": str(provenance["build"]["git_head"]),
        "source_fingerprint": str(provenance["build"]["source_fingerprint"]),
        "aicore_kernel_sha256": str(provenance["artifacts"]["aicore_kernel"]["sha256"]),
        "aicore_kernel_text_sha256": str(provenance["artifacts"]["aicore_kernel"]["text"]["sha256"]),
    }
    result = {
        "capture_mode": mode,
        "global_submit_span_us": float(capture.data["window"]["global_submit_span_us"]),
        "expected_submits_per_core": int(capture.data["configuration"]["expected_submits_per_core"]),
        "sys_counter_tick_ns": int(capture.data["configuration"]["sys_counter_tick_ns"]),
        "denominators": {name: _group_denominator(capture.summary[name]) for name in ("all", "aic", "aiv")},
        "measurement_scopes": {
            "scalar_submit_elapsed_ticks": (
                "sum of gate-running SYS counter segments inside the first/last Submit closure, then subtract "
                "result-used return-ready atomic waiting"
            ),
            "pmu_total_scalar_busy_and_primary_icache": (
                "per-core PMU gate nested inside the first/last Submit SYS counter boundary; linked Kernel causes "
                "paired gate pauses, while atomic instruction events remain"
            ),
            "global_submit_span_us": "cross-core first/last Submit closure only",
        },
        "source": source,
    }
    if capture.phase_summary is None:
        result.update(
            {
                "kind": "whole-window",
                "label": "PMU whole gate 与 Scalar Submit 时间分母",
                "boundary_reference": "SubmitEnvelope boundary only",
                "mapping": "whole-window-boundary",
            }
        )
        return result

    label, boundary_reference, mapping = PHASE_PRESENTATION[mode]
    phase = capture.data["configuration"]["phase"]
    result.update(
        {
            "kind": "calibration" if mode == EMPTY_BRACKET_CAPTURE_MODE else "phase",
            "label": label,
            "boundary_reference": boundary_reference,
            "mapping": mapping,
            "phase": {
                "id": int(phase["id"]),
                "name": str(phase["name"]),
                "boundary": str(phase["boundary"]),
                "counter_semantics": str(phase["counter_semantics"]),
                "time_semantics": str(phase["time_semantics"]),
                "groups": {name: _phase_group_summary(capture.phase_summary[name]) for name in ("all", "aic", "aiv")},
            },
        }
    )
    return result


def _coverage_matrix(swimlane: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = [
        {
            "swimlane_region": "WorkerCompletion",
            "pmu_profile": "—",
            "coverage": "not-covered",
            "note": "含 OrchestrationReplay 与 FinalDrain；没有独立 PMU 父窗",
        },
        {
            "swimlane_region": "OrchestrationReplay",
            "pmu_profile": "—",
            "coverage": "not-covered",
            "note": "PMU gate 嵌在首末 Submit SYS closure 内，不等同该父区间",
        },
        {
            "swimlane_region": "OrchestrationSetup",
            "pmu_profile": "—",
            "coverage": "not-covered",
            "note": "当前没有独立 PMU phase",
        },
        {
            "swimlane_region": "SubmitEnvelope",
            "pmu_profile": NONE_CAPTURE_MODE,
            "coverage": "boundary-only",
            "note": "PMU Scalar 分母另行排除 linked Kernel 与 return-ready atomic 时间",
        },
        {
            "swimlane_region": "SubmitUnion",
            "pmu_profile": "—",
            "coverage": "independent-components-only",
            "note": "各 phase 是独立 ELF，不允许求和成 SubmitUnion",
        },
        {
            "swimlane_region": "EfDrain",
            "pmu_profile": EFDRAIN_CONTROL_CAPTURE_MODE,
            "coverage": "control-only",
            "note": "泳道父 span 含 Kernel；PMU 只看排除 Kernel 后的 Scalar control",
        },
        {
            "swimlane_region": "EfDrainKernelUnion",
            "pmu_profile": "—",
            "coverage": "intentionally-excluded",
            "note": "目标是纯 Scalar 归因，linked Kernel 从 PMU gate 排除",
        },
        {
            "swimlane_region": "EfDrainControl",
            "pmu_profile": EFDRAIN_CONTROL_CAPTURE_MODE,
            "coverage": "control-only",
            "note": "与泳道 EfDrain 扣除 Kernel union 后的控制语义对应",
        },
        {
            "swimlane_region": "Materialize",
            "pmu_profile": MATERIALIZE_CAPTURE_MODE,
            "coverage": "same-business-boundary",
            "note": "独立 ELF，只可用自己的分母",
        },
        {
            "swimlane_region": "PrepareMap",
            "pmu_profile": PREPARE_MAP_CAPTURE_MODE,
            "coverage": "same-business-boundary",
            "note": "独立 ELF，只可用自己的分母",
        },
        {
            "swimlane_region": "Claim",
            "pmu_profile": CLAIM_CAPTURE_MODE,
            "coverage": "same-business-boundary",
            "note": "独立 ELF，只可用自己的分母",
        },
        {
            "swimlane_region": "Fanin",
            "pmu_profile": FANIN_CAPTURE_MODE,
            "coverage": "same-business-boundary",
            "note": "动态 winner 调用区间",
        },
        {
            "swimlane_region": "Register",
            "pmu_profile": REGISTER_CAPTURE_MODE,
            "coverage": "same-business-boundary",
            "note": "独立 ELF，只可用自己的分母",
        },
        {
            "swimlane_region": "WinnerBuild",
            "pmu_profile": WINNER_BUILD_CAPTURE_MODE,
            "coverage": "control-only",
            "note": "泳道父 span 可能含 Kernel；PMU 排除 linked Kernel",
        },
        {
            "swimlane_region": "AllocComplete",
            "pmu_profile": ALLOC_COMPLETE_CAPTURE_MODE,
            "coverage": "control-only",
            "note": "泳道父 span 可能含 Kernel；PMU 排除 linked Kernel",
        },
        {
            "swimlane_region": "LoserReplay",
            "pmu_profile": LOSER_REPLAY_CAPTURE_MODE,
            "coverage": "same-business-boundary",
            "note": "真实路径不执行 linked Kernel",
        },
        {
            "swimlane_region": "BetweenSubmitResidual",
            "pmu_profile": SUBMIT_TRANSITION_CAPTURE_MODE,
            "coverage": "adjacent-submit-boundary",
            "note": "聚合每核相邻 Submit 的 N-1 个间隙",
        },
        {
            "swimlane_region": "OrchestrationTail",
            "pmu_profile": "—",
            "coverage": "not-covered",
            "note": "当前没有独立 PMU phase",
        },
        {
            "swimlane_region": "FinalDrain",
            "pmu_profile": "—",
            "coverage": "outside-submit-pmu",
            "note": "发生在末次 Submit 之后，不进入当前 PMU 窗",
        },
        {
            "swimlane_region": "FinalDrainKernelUnion",
            "pmu_profile": "—",
            "coverage": "outside-submit-pmu",
            "note": "末次 Submit 后的 linked Kernel，不属于纯 Submit Scalar 分母",
        },
        {
            "swimlane_region": "FinalDrainResidual",
            "pmu_profile": "—",
            "coverage": "outside-submit-pmu",
            "note": "末次 Submit 后的控制残余",
        },
        {
            "swimlane_region": "SubmitResidual",
            "pmu_profile": ARG_BUILD_CAPTURE_MODE,
            "coverage": "partial",
            "note": "仅 Claim→Materialize internal residual 有同边界 PMU；tail 未覆盖",
        },
        {
            "swimlane_region": "SubmitInternalResidual",
            "pmu_profile": ARG_BUILD_CAPTURE_MODE,
            "coverage": "partial",
            "note": "逐 segment 见下列动态覆盖项",
        },
        {
            "swimlane_region": "SubmitTailResidual",
            "pmu_profile": "—",
            "coverage": "not-covered",
            "note": "逐 segment 列出但当前没有独立 PMU phase",
        },
    ]
    internal_segments = swimlane["residual_segments"]["submit_internal_residual"]["segments"]
    for segment in internal_segments:
        is_arg_build = segment["boundary"] == "Claim->Materialize"
        rows.append(
            {
                "swimlane_region": f"SubmitInternalResidual/{segment['boundary']}",
                "pmu_profile": ARG_BUILD_CAPTURE_MODE if is_arg_build else "—",
                "coverage": "residual-boundary" if is_arg_build else "not-covered",
                "note": "ArgBuild 同边界" if is_arg_build else "当前没有独立 PMU phase",
            }
        )
    for segment in swimlane["residual_segments"]["submit_tail_residual"]["segments"]:
        rows.append(
            {
                "swimlane_region": f"SubmitTailResidual/{segment['boundary']}",
                "pmu_profile": "—",
                "coverage": "not-covered",
                "note": "当前没有独立 PMU phase",
            }
        )
    for overlay_name in swimlane["overlays"]:
        is_atomic = overlay_name == "Atomic"
        rows.append(
            {
                "swimlane_region": f"{overlay_name} overlay",
                "pmu_profile": "—",
                "coverage": "swimlane-overlay",
                "note": (
                    "PMU 时间只扣 result-used return-ready 等待；counter 仍含 atomic 指令事件"
                    if is_atomic
                    else "非加和泳道 overlay；当前没有独立 PMU phase"
                ),
            }
        )
    rows.append(
        {
            "swimlane_region": "观察器校准",
            "pmu_profile": EMPTY_BRACKET_CAPTURE_MODE,
            "coverage": "observer-calibration",
            "note": "不是业务 span，不进入任何业务分布",
        }
    )
    return rows


def _assert_pmu_sources_unchanged(sources: Sequence[Mapping[str, Any]]) -> None:
    for source in sources:
        raw_path = Path(str(source["raw_path"]))
        raw_size, raw_sha = _sha256_file(raw_path)
        if raw_size != source["raw_size"] or raw_sha != source["raw_sha256"]:
            _fail(f"Submit-PMU raw changed while building overview: {raw_path}")
        provenance_path = Path(str(source["provenance_path"]))
        _, provenance_sha = _sha256_file(provenance_path)
        if provenance_sha != source["provenance_sha256"]:
            _fail(f"Submit-PMU provenance changed while building overview: {provenance_path}")


def build_overview(swimlane_raw: Path | str, pmu_dirs: Sequence[Path | str]) -> dict[str, Any]:
    """Strictly reload all source artifacts and return a compact machine-readable overview."""

    raw_path = Path(swimlane_raw)
    if raw_path.name != "l2_swimlane_records.json":
        _fail("swimlane input must be the producer l2_swimlane_records.json, not merged_swimlane.json")
    raw_size, raw_sha256 = _sha256_file(raw_path)
    analysis = analyze_capture(raw_path)
    if _sha256_file(raw_path) != (raw_size, raw_sha256):
        _fail("swimlane raw changed while schema-v4 analysis was running")
    frequency_hz, task_count = _validate_swimlane_analysis(analysis)
    swimlane = _summarize_swimlane(analysis, raw_path, raw_size, raw_sha256, frequency_hz)

    if len(pmu_dirs) != len(EXPECTED_PMU_MODES):
        _fail(f"overview requires exactly {len(EXPECTED_PMU_MODES)} Submit-PMU directories")
    captures: dict[str, dict[str, Any]] = {}
    scenario_keys: set[tuple[str, ...]] = set()
    for raw_dir_value in pmu_dirs:
        raw_dir = Path(raw_dir_value)
        raw_file = raw_dir / DEFAULT_INPUT_NAME if raw_dir.is_dir() else raw_dir
        if raw_file.name != DEFAULT_INPUT_NAME:
            _fail(f"Submit-PMU input must be a directory or {DEFAULT_INPUT_NAME}")
        provenance_file = raw_file.with_name(DEFAULT_PROVENANCE_NAME)
        if not provenance_file.is_file():
            _fail(f"missing Submit-PMU provenance: {provenance_file}")
        capture = load_capture(raw_file)
        provenance, provenance_sha256 = load_provenance(provenance_file, capture)
        mode = str(capture.data["capture"]["mode"])
        if mode in captures:
            _fail(f"duplicate Submit-PMU capture mode: {mode}")
        if mode not in EXPECTED_PMU_MODES:
            _fail(f"unsupported Submit-PMU capture mode: {mode}")
        if int(capture.data["configuration"]["expected_submits_per_core"]) != task_count:
            _fail(f"Submit-PMU {mode} task count does not match the swimlane capture")
        profile_key = tuple(str(value) for value in provenance["build"]["profiled_cache_key"])
        scenario_keys.add(profile_key[:-1])
        captures[mode] = _summarize_pmu_capture(capture, provenance, provenance_sha256, provenance_file)

    missing = EXPECTED_PMU_MODES - captures.keys()
    if missing:
        _fail(f"missing Submit-PMU capture modes: {sorted(missing)!r}")
    if len(scenario_keys) != 1:
        _fail("Submit-PMU provenance entries do not describe one common test/platform/scene")
    ordered = [captures[mode] for mode in PMU_MODE_ORDER]
    sources = [capture["source"] for capture in ordered]
    _assert_pmu_sources_unchanged(sources)

    git_heads = sorted({str(source["git_head"]) for source in sources})
    aicore_hashes = sorted({str(source["aicore_kernel_sha256"]) for source in sources})
    phase_profiles = [item for item in ordered if item["kind"] == "phase"]
    payload = {
        "schema": OVERVIEW_SCHEMA,
        "semantics": {
            "evidence_chains_are_independent": True,
            "cross_elf_absolute_subtraction_allowed": False,
            "cross_elf_phase_shares_additive": False,
            "swimlane_percentages": "same swimlane ELF aggregate core-work partitions",
            "pmu_percentages": "each row divided only by that PMU ELF's own Scalar/primary denominator",
            "swimlane_sys_counter": "capture frequency is a SYS counter conversion, not the 1.65 GHz PMU cycle rate",
            "pmu_scalar_time": (
                "linked vector/cube Kernel and result-used return-ready atomic waiting are excluded; "
                "source-issue atomic is retained"
            ),
            "pmu_counter": (
                "PMU whole gate excludes linked vector/cube Kernel; atomic instruction and observation events "
                "remain included"
            ),
            "pmu_whole_gate": (
                "PMU total/scalar-busy/primary I-cache use a per-core gate nested inside the first/last Submit "
                "SYS counter closure; scalar_submit_elapsed_ticks sums gate-running SYS segments and then "
                "subtracts result-used return-ready atomic waiting"
            ),
            "phase_observed": (
                "running read-clear observations include nearby observer/bookkeeping code and are not a "
                "zero-instrumentation bound"
            ),
        },
        "validation": {
            "status": "PASS",
            "status_scope": "each evidence chain passed its own strict validation",
            "swimlane_schema_v4_reanalyzed": True,
            "swimlane_raw_unchanged": True,
            "swimlane_build_provenance_available": False,
            "swimlane_to_pmu_identity_bound": False,
            "swimlane_to_pmu_alignment": "96-core topology and expected Submit count only",
            "pmu_profiles_complete": True,
            "pmu_profile_count": len(ordered),
            "pmu_raw_and_provenance_unchanged": True,
            "expected_submits_per_core": task_count,
            "scenario_key": list(next(iter(scenario_keys))),
            "git_head_uniform": len(git_heads) == 1,
            "git_heads": git_heads,
            "aicore_kernel_hash_count": len(aicore_hashes),
        },
        "swimlane_elf": swimlane,
        "submit_pmu_elfs": {
            "whole_window": captures[NONE_CAPTURE_MODE],
            "phase_profiles": phase_profiles,
            "calibration": captures[EMPTY_BRACKET_CAPTURE_MODE],
        },
        "coverage_matrix": _coverage_matrix(swimlane),
    }
    return payload


def _fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def _fmt_us(value: int | float) -> str:
    return f"{float(value):,.3f} µs"


def _fmt_pct(value: int | float) -> str:
    return f"{float(value):.3%}"


def _range_text(data: Mapping[str, Any], *, unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    return f"{_fmt_int(data['min'])}–{_fmt_int(data['max'])}{suffix}"


def _partition_html(partition: Mapping[str, Any], *, show_ranges: bool = True) -> str:
    colors = ("#2563eb", "#0d9488", "#7c3aed", "#d97706", "#dc2626", "#0891b2", "#65a30d")
    bars = []
    rows = []
    for index, row in enumerate(partition["rows"]):
        color = colors[index % len(colors)]
        share = float(row["share"])
        bars.append(
            f'<span style="width:{share * 100:.8f}%;background:{color}" '
            f'title="{html.escape(row["label"])} {_fmt_pct(share)}"></span>'
        )
        if show_ranges and row["aic_per_core"] is not None:
            aic = f"{row['aic_per_core']['min_us']:.3f}–{row['aic_per_core']['max_us']:.3f} µs"
            aiv = f"{row['aiv_per_core']['min_us']:.3f}–{row['aiv_per_core']['max_us']:.3f} µs"
        else:
            aic = aiv = "—"
        rows.append(
            "<tr>"
            f'<td><i style="background:{color}"></i><code>{html.escape(row["label"])}</code></td>'
            f"<td>{_fmt_us(row['core_time_us'])}</td><td>{_fmt_pct(share)}</td>"
            f"<td>{aic}</td><td>{aiv}</td>"
            "</tr>"
        )
    return f"""
      <div class="partition">
        <h3>{html.escape(str(partition["name"]))}</h3>
        <div class="stack">{"".join(bars)}</div>
        <div class="table-wrap"><table>
          <thead><tr><th>区域</th><th>Σ core-time</th><th>同父区间占比</th>
            <th>AIC 每核 min–max</th><th>AIV 每核 min–max</th></tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table></div>
      </div>
    """


def _phase_metric_html(
    label: str,
    metric: Mapping[str, Any],
    denominator: int,
    share: float,
    aic: Mapping[str, Any],
    aiv: Mapping[str, Any],
) -> str:
    ratio = (
        f"{_fmt_int(metric['sum'])} / {_fmt_int(denominator)} = {_fmt_pct(share)}"
        if denominator
        else "N/A（本 ELF 分母为 0）"
    )
    return f"""
      <div class="metric">
        <span>{html.escape(label)}</span>
        <strong>{_fmt_int(metric["sum"])}</strong>
        <small>本 ELF：{ratio}</small>
        <small>AIC 每核 {_range_text(aic)}；AIV 每核 {_range_text(aiv)}</small>
      </div>
    """


def _phase_card_html(item: Mapping[str, Any]) -> str:
    groups = item["phase"]["groups"]
    all_group = groups["all"]
    aic = groups["aic"]
    aiv = groups["aiv"]
    source = item["source"]
    calls = (
        f"{_fmt_int(all_group['phase_business_calls'])} / "
        f"{_fmt_int(aic['phase_business_calls'])} / {_fmt_int(aiv['phase_business_calls'])}"
    )
    calls_per_core = (
        f"AIC {_range_text(aic['phase_calls_per_core'])}（零调用核 {aic['phase_zero_call_cores']}）；"
        f"AIV {_range_text(aiv['phase_calls_per_core'])}（零调用核 {aiv['phase_zero_call_cores']}）"
    )
    time_metric = all_group["phase_elapsed_ticks"]
    time_aic = aic["phase_elapsed_ticks"]
    time_aiv = aiv["phase_elapsed_ticks"]
    request_metric = all_group["phase_icache_requests_observed"]
    request_aic = aic["phase_icache_requests_observed"]
    request_aiv = aiv["phase_icache_requests_observed"]
    miss_metric = all_group["phase_icache_misses_observed"]
    miss_aic = aic["phase_icache_misses_observed"]
    miss_aiv = aiv["phase_icache_misses_observed"]
    denominators = item["denominators"]["all"]
    kernel_calls = all_group["phase_excluded_kernel_calls"]
    elapsed_label = (
        "Observer pair empirical elapsed (1 tick = 1 ns)"
        if item["kind"] == "calibration"
        else "Scalar phase core-time (1 tick = 1 ns)"
    )
    time_html = _phase_metric_html(
        elapsed_label,
        time_metric,
        denominators["scalar_submit_elapsed_ticks"]["sum"],
        all_group["phase_time_share_of_scalar_submit"],
        time_aic,
        time_aiv,
    )
    request_html = _phase_metric_html(
        "I-cache request observed",
        request_metric,
        denominators["primary_icache_requests"]["sum"],
        all_group["phase_request_observed_share_of_primary"],
        request_aic,
        request_aiv,
    )
    miss_html = _phase_metric_html(
        "I-cache miss observed",
        miss_metric,
        denominators["primary_icache_misses"]["sum"],
        all_group["phase_miss_observed_share_of_primary"],
        miss_aic,
        miss_aiv,
    )
    return f"""
    <article class="phase-card">
      <header>
        <div><h3>{html.escape(str(item["label"]))}</h3>
          <code>{html.escape(str(item["capture_mode"]))}</code></div>
        <span class="tag">{html.escape(str(item["mapping"]))}</span>
      </header>
      <p class="boundary">{html.escape(str(item["phase"]["boundary"]))} · calls ALL/AIC/AIV {calls}<br>
        calls/core {calls_per_core}</p>
      <div class="metrics">
        {time_html}
        {request_html}
        {miss_html}
      </div>
      <p class="fine">分母只来自本 ELF；excluded linked Kernel calls={_fmt_int(kernel_calls)}；
        global Submit closure={item["global_submit_span_us"]:.3f} µs（不作 phase 分母）。</p>
      <p class="source">git <code>{html.escape(source["git_head"][:12])}</code> · AICore ELF
        <code>{html.escape(source["aicore_kernel_sha256"][:12])}</code> ·
        <a href="{html.escape(source["html_report_path"])}">单份报告</a></p>
    </article>
    """


def _whole_window_html(item: Mapping[str, Any]) -> str:
    groups = item["denominators"]
    all_group = groups["all"]
    aic = groups["aic"]
    aiv = groups["aiv"]

    def card(label: str, key: str, unit: str = "") -> str:
        return f"""
        <div class="metric">
          <span>{html.escape(label)}</span><strong>{_fmt_int(all_group[key]["sum"])}</strong>
          <small>AIC 每核 {_range_text(aic[key], unit=unit)}</small>
          <small>AIV 每核 {_range_text(aiv[key], unit=unit)}</small>
        </div>"""

    source = item["source"]
    return f"""
    <div class="whole-card">
      <h3><code>{NONE_CAPTURE_MODE}</code>：PMU whole gate 与 Scalar Submit 时间分母</h3>
      <p>global 首末 Submit 仅作闭合：<strong>{item["global_submit_span_us"]:.3f} µs</strong>；
        下列数值是 96 核各自累计，不是墙钟。每核先读首个 Submit start tick，再启动 PMU；末次 Submit
        先停止 PMU，再读 end tick，所以 PMU gate 嵌在首末 SYS closure 内。Scalar elapsed 累计 gate-running
        SYS 段，再扣 linked Kernel 与 return-ready atomic 等待；PMU cycle 与 SYS tick 口径不同，不能直接相减。</p>
      <div class="metrics">
        {card("Scalar Submit elapsed (1 tick = 1 ns)", "scalar_submit_elapsed_ticks", "ticks")}
        {card("PMU total cycles", "pmu_total_cycles", "cycles")}
        {card("Scalar busy cycles", "scalar_busy_cycles", "cycles")}
        {card("Primary I-cache request", "primary_icache_requests")}
        {card("Primary I-cache miss", "primary_icache_misses")}
      </div>
      <p class="source">git <code>{html.escape(source["git_head"][:12])}</code> · AICore ELF
        <code>{html.escape(source["aicore_kernel_sha256"][:12])}</code> ·
        <a href="{html.escape(source["html_report_path"])}">单份报告</a></p>
    </div>
    """


def render_overview(payload: Mapping[str, Any]) -> str:
    """Render a self-contained report without inventing a cross-ELF denominator."""

    swimlane = payload["swimlane_elf"]
    pmu = payload["submit_pmu_elfs"]
    validation = payload["validation"]
    phase_cards = "".join(_phase_card_html(item) for item in pmu["phase_profiles"])
    calibration = _phase_card_html(pmu["calibration"])
    coverage_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(row['swimlane_region'])}</code></td>"
        f"<td><code>{html.escape(row['pmu_profile'])}</code></td>"
        f"<td>{html.escape(row['coverage'])}</td><td>{html.escape(row['note'])}</td>"
        "</tr>"
        for row in payload["coverage_matrix"]
    )
    residual_rows_list = []
    for name, value in swimlane["residual_segments"].items():
        for segment in value["segments"]:
            duration_us = segment["cycles"] / swimlane["sys_counter_frequency_hz"] * 1_000_000
            residual_rows_list.append(
                "<tr>"
                f"<td><code>{html.escape(name)}/{html.escape(segment['boundary'])}</code></td>"
                f"<td>{_fmt_int(segment['event_count'])}</td><td>{_fmt_us(duration_us)}</td>"
                f"<td>{_fmt_int(segment['aic_cycles'])}</td>"
                f"<td>{_fmt_int(segment['aiv_cycles'])}</td>"
                "</tr>"
            )
    residual_rows = "".join(residual_rows_list)
    overlay_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(name)}</code></td>"
        f"<td>{_fmt_int(value['event_count'])}</td>"
        f"<td>{_fmt_int(value['aggregate_duration_cycles'])}</td>"
        "<td>否</td></tr>"
        for name, value in swimlane["overlays"].items()
    )
    containment_rows = "".join(
        f"<tr><td><code>{html.escape(name)}</code></td><td>{_fmt_int(value)}</td></tr>"
        for name, value in sorted(swimlane["kernel_containment"].items())
    )
    head_list = "、".join(html.escape(head[:12]) for head in validation["git_heads"])
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>FDWIC Submit 全 span 证据汇总</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#5b6475; --line:#d8dee9; --paper:#f5f7fb; }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
      color:var(--ink); background:var(--paper);
    }}
    main {{ max-width:1320px; margin:auto; padding:28px 20px 64px; }}
    h1 {{ margin:0 0 6px; font-size:30px; }} h2 {{ margin:34px 0 14px; font-size:23px; }} h3 {{ margin:0 0 8px; }}
    code {{ overflow-wrap:anywhere; }}
    .notice {{
      padding:14px 16px; border:1px solid #f59e0b; border-left:5px solid #d97706;
      background:#fffbeb; border-radius:8px;
    }}
    .ok {{ color:#047857; font-weight:700; }} .fine,.source,.boundary {{ color:var(--muted); }}
    .partition,.whole-card,.phase-card,.context {{
      background:white; border:1px solid var(--line); border-radius:12px;
      padding:18px; margin:14px 0; box-shadow:0 2px 10px #1720330a;
    }}
    .stack {{
      display:flex; width:100%; height:22px; overflow:hidden;
      border-radius:6px; background:#e5e7eb; margin:12px 0 16px;
    }}
    .stack span {{ min-width:1px; }}
    .table-wrap {{ max-width:100%; overflow-x:auto; border:1px solid var(--line); border-radius:8px; }}
    table {{ width:100%; border-collapse:collapse; min-width:680px; }}
    th,td {{ padding:9px 11px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
    th {{ background:#f8fafc; }} tr:last-child td {{ border-bottom:0; }}
    td i {{ display:inline-block; width:9px; height:9px; margin-right:7px; border-radius:2px; }}
    .phase-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr)); gap:14px; }}
    .phase-card {{ margin:0; }}
    .phase-card header {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }}
    .phase-card h3 {{ margin:0; }}
    .tag {{ padding:3px 8px; background:#eef2ff; color:#4338ca; border-radius:999px; white-space:nowrap; }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:10px; margin:12px 0; }}
    .metric {{ padding:11px 12px; background:#f8fafc; border:1px solid #e5e7eb; border-radius:8px; min-width:0; }}
    .metric span,.metric small {{ display:block; color:var(--muted); }}
    .metric strong {{ display:block; font-size:19px; margin:2px 0; overflow-wrap:anywhere; }}
    .source a {{ color:#1d4ed8; }}
    @media (max-width:650px) {{
      main {{ padding:18px 10px 40px; }} h1 {{ font-size:24px; }}
      .phase-card header {{ display:block; }} .tag {{ display:inline-block; margin-top:8px; }}
    }}
  </style>
</head>
<body><main>
  <h1>FDWIC Submit 全 span 证据汇总</h1>
  <p><span class="ok">PASS</span> · 96 核 · 每核 {validation["expected_submits_per_core"]:,} 个 Submit ·
    泳道 raw 重新执行 schema-v4 严格闭合 · 13/13 PMU raw + provenance 闭合。
    PASS 只表示两条证据链各自通过门禁，不表示它们已绑定到同一 ELF。</p>
  <div class="notice"><strong>这里有两条互不混算的证据链。</strong>
    泳道图的百分比只在同一个泳道 ELF 的排他时间树中相加；每个 Submit-PMU profile 都是独立 ELF，
    phase 时间、request 和 miss 只能除以本 ELF 自己的 Scalar/primary 分母。不同 ELF 的绝对时间不能相减，
    各 PMU 行的占比也不能相加成 100%。</div>

  <section class="context">
    <h2>口径与来源</h2>
    <p>泳道 SYS counter 频率：<strong>{swimlane["sys_counter_frequency_hz"]:,} Hz</strong>；它用于时间戳换算，
      不是约 1.65 GHz 的 PMU cycle 频率。泳道全局 Submit 墙钟范围：
      <strong>{swimlane["global_submit_makespan"]["duration_us"]:.3f} µs</strong>。</p>
    <p>PMU Scalar 时间已排除 linked Vector/Cube Kernel 和 result-used return-ready atomic 的等待区间；
      source-issue atomic 保留。PMU total、Scalar busy 与 primary I-cache 来自嵌在首末 Submit SYS closure
      内且遇 linked Kernel 会暂停的 PMU gate；Scalar phase 时间占比使用 gate-running SYS 段累计后再扣
      return-ready 等待的独立时间分母。PMU/I-cache counter
      仍含 atomic 指令及观察代码事件，
      局部 observed 不是零插桩函数体的数学上下界。</p>
    <p>本批 PMU 来自 {len(validation["git_heads"])} 组 revision：<code>{head_list}</code>；
      mixed revisions 被如实保留，进一步禁止跨 ELF 数值运算。泳道 raw 没有 build provenance，
      页面只证明 raw SHA 与 schema-v4 重算闭合，
      不伪称已证明当时 ELF 身份；泳道与 PMU 之间仅对齐 96 核拓扑和每核 Submit 数。</p>
  </section>

  <h2>泳道 ELF：同一份业务时间分布</h2>
  <p class="notice">泳道分区展示原始业务 elapsed，不等同于纯 Scalar 时间。Kernel 是父 span 内的嵌套事件；
    当前 analyzer 能精确拆开 EfDrain 与 FinalDrain；其他 containment（本轮包括 WinnerBuild）只有事件数、
    没有独立 Kernel union 时长。因此纯 Scalar 归因只看下方相应 PMU control ELF，不从泳道父 span 猜减。</p>
  {_partition_html(swimlane["submit_envelope_partition"])}
  {_partition_html(swimlane["submit_union_partition"])}
  {_partition_html(swimlane["efdrain_partition"])}

  <details class="context"><summary><strong>外围 Worker / Orchestration / FinalDrain 闭合</strong></summary>
    {_partition_html(swimlane["worker_completion_partition"])}
    {_partition_html(swimlane["orchestration_partition"])}
    {_partition_html(swimlane["final_drain_partition"])}
  </details>

  <h2>泳道 residual 的业务边界</h2>
  <div class="table-wrap"><table><thead><tr><th>边界</th><th>次数</th><th>Σ core-time</th>
    <th>AIC ticks</th><th>AIV ticks</th></tr></thead>
    <tbody>{residual_rows}</tbody></table></div>

  <h2>泳道 Kernel containment 与非加和 overlay</h2>
  <div class="phase-grid">
    <div class="table-wrap"><table><thead><tr><th>Kernel 归属</th><th>事件数</th></tr></thead>
      <tbody>{containment_rows}</tbody></table></div>
    <div class="table-wrap"><table><thead><tr><th>Overlay</th><th>事件数</th>
      <th>累计 ticks</th><th>进入分区</th></tr></thead>
      <tbody>{overlay_rows}</tbody></table></div>
  </div>

  <h2>Submit-PMU ELF：whole gate 与 Scalar 时间分母</h2>
  {_whole_window_html(pmu["whole_window"])}

  <h2>Submit-PMU ELF：各阶段独立归因</h2>
  <p>每张卡的三个百分比都只使用该卡所属 ELF 的 Scalar Submit、primary request、primary miss 分母；
    页面没有也不会生成跨卡合计。</p>
  <div class="phase-grid">{phase_cards}</div>

  <h2>观察器校准（不是业务阶段）</h2>
  <div class="phase-grid">{calibration}</div>

  <h2>泳道区域与 PMU 覆盖矩阵</h2>
  <div class="table-wrap"><table><thead><tr><th>泳道区域</th><th>PMU profile</th><th>关系</th><th>说明</th></tr></thead>
    <tbody>{coverage_rows}</tbody></table></div>

  <p class="fine">机器可读数据见同目录 <code>{DEFAULT_JSON_NAME}</code>。所有逐核分布只呈现 min–max；
    phase extrema 是每核累计完整 phase 的极值，不是单次调用极值。</p>
</main></body></html>
"""


def _stage_text(path: Path, document: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _restore(path: Path, snapshot: bytes | None) -> None:
    if snapshot is None:
        path.unlink(missing_ok=True)
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".restore", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(snapshot)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_pair(json_path: Path, json_document: str, html_path: Path, html_document: str, overwrite: bool) -> None:
    if not overwrite and (json_path.exists() or html_path.exists()):
        _fail("overview output already exists; pass --overwrite to replace the pair")
    previous = {
        json_path: json_path.read_bytes() if json_path.is_file() else None,
        html_path: html_path.read_bytes() if html_path.is_file() else None,
    }
    staged_json = _stage_text(json_path, json_document)
    try:
        staged_html = _stage_text(html_path, html_document)
    except BaseException:
        staged_json.unlink(missing_ok=True)
        raise
    published: list[Path] = []
    try:
        os.replace(staged_json, json_path)
        published.append(json_path)
        os.replace(staged_html, html_path)
        published.append(html_path)
    except BaseException as error:
        restore_errors = []
        for path in published:
            try:
                _restore(path, previous[path])
            except BaseException as restore_error:  # pragma: no cover - catastrophic filesystem failure
                restore_errors.append(f"{path}: {restore_error}")
        if restore_errors:
            message = "overview publication failed and rollback was incomplete: " + "; ".join(restore_errors)
            raise RuntimeError(message) from error
        raise
    finally:
        staged_json.unlink(missing_ok=True)
        staged_html.unlink(missing_ok=True)


def _acquire_output_lock(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / OUTPUT_LOCK_NAME
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError(
            f"another overview publication owns {lock_path}; remove a stale lock only after confirming no writer"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        os.fsync(stream.fileno())
    return lock_path


def write_overview(
    swimlane_raw: Path | str,
    pmu_dirs: Sequence[Path | str],
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Build and publish the fixed JSON/HTML pair with an output lock and rollback."""

    payload = build_overview(swimlane_raw, pmu_dirs)
    json_document = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    html_document = render_overview(payload)

    swimlane_source = payload["swimlane_elf"]["source"]
    swimlane_path = Path(str(swimlane_source["raw_path"]))
    if _sha256_file(swimlane_path) != (swimlane_source["raw_size"], swimlane_source["raw_sha256"]):
        _fail("swimlane raw changed before overview publication")
    pmu_sources = [payload["submit_pmu_elfs"]["whole_window"]["source"]]
    pmu_sources.extend(item["source"] for item in payload["submit_pmu_elfs"]["phase_profiles"])
    pmu_sources.append(payload["submit_pmu_elfs"]["calibration"]["source"])
    _assert_pmu_sources_unchanged(pmu_sources)

    output = Path(output_dir)
    json_path = output / DEFAULT_JSON_NAME
    html_path = output / DEFAULT_HTML_NAME
    lock_path = _acquire_output_lock(output)
    try:
        _publish_pair(json_path, json_document, html_path, html_document, overwrite)
    finally:
        lock_path.unlink(missing_ok=True)
    return json_path, html_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swimlane-raw", required=True, type=Path, help="producer l2_swimlane_records.json")
    parser.add_argument(
        "--pmu-dir",
        required=True,
        action="append",
        type=Path,
        help=f"one directory containing {DEFAULT_INPUT_NAME} and {DEFAULT_PROVENANCE_NAME}; repeat exactly 13 times",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="directory for the fixed JSON/HTML pair")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output pair")
    args = parser.parse_args(argv)
    try:
        json_path, html_path = write_overview(
            args.swimlane_raw,
            args.pmu_dir,
            args.output_dir,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json_path)
    print(html_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
