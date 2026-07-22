# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the LICENSE file.
# -----------------------------------------------------------------------------------------------------------
"""Offline closure tests for the provenance-aware FDWIC Submit span overview."""

from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import simpler_setup.tools.fdwic_submit_span_overview as overview_module
from simpler_setup.tools.fdwic_submit_pmu_report import SubmitPmuCapture

_PER_CORE_METRICS = {
    "submit_envelope": 100,
    "submit_union": 80,
    "between_submit_residual": 20,
    "efdrain": 20,
    "materialize": 10,
    "prepare_map": 5,
    "claim": 7,
    "fanin": 3,
    "register": 8,
    "winner_build": 6,
    "alloc_complete": 4,
    "loser_replay": 5,
    "submit_residual": 12,
    "efdrain_kernel_union": 8,
    "efdrain_control": 12,
    "orchestration_replay": 120,
    "orchestration_setup": 10,
    "orchestration_tail": 10,
    "final_drain": 15,
    "final_drain_kernel_union": 10,
    "final_drain_residual": 5,
    "worker_completion": 135,
}


def _valid_swimlane_analysis() -> dict[str, Any]:
    per_core = [
        {
            "core_id": core_id,
            "role": "aic" if core_id < 32 else "aiv",
            "submit_count": 5,
            "metrics_cycles": dict(_PER_CORE_METRICS),
        }
        for core_id in range(96)
    ]
    aggregate = {metric: value * 96 for metric, value in _PER_CORE_METRICS.items()}
    return {
        "schema_version": overview_module.REPORT_SCHEMA_VERSION,
        "validation": {
            "status": "PASS",
            "dropped_records": 0,
            "physical_topology_complete": True,
            "task_stream_contiguous_and_equal_per_core": True,
            "orchestration_parent_exactly_one_per_core": True,
            "final_drain_parent_exactly_one_per_core": True,
            "parent_boundaries_adjacent": True,
            "exclusive_children_non_overlapping": True,
            "all_integer_cycle_closures_exact": True,
        },
        "capture": {
            "trace_schema_version": 4,
            "clock_freq_hz": 1_000_000_000,
            "core_count": 96,
            "task_count_per_core": 5,
        },
        "aggregate_core_work": {
            "metrics_cycles": aggregate,
            "closure": {
                name: {"exact": True}
                for name in (
                    "submit_partition",
                    "submit_envelope",
                    "efdrain_partition",
                    "orchestration_replay",
                    "final_drain",
                    "worker_completion",
                )
            },
        },
        "per_core": per_core,
        "residual_breakdown": {
            "submit_internal_residual": {
                "total_cycles": 7 * 96,
                "segments": [
                    {
                        "boundary": "Claim->Materialize",
                        "event_count": 5 * 96,
                        "cycles": 4 * 96,
                        "aic_cycles": 4 * 32,
                        "aiv_cycles": 4 * 64,
                    },
                    {
                        "boundary": "Register->WinnerBuild",
                        "event_count": 5 * 96,
                        "cycles": 3 * 96,
                        "aic_cycles": 3 * 32,
                        "aiv_cycles": 3 * 64,
                    },
                ],
            },
            "submit_tail_residual": {
                "total_cycles": 5 * 96,
                "segments": [
                    {
                        "boundary": "LoserReplay->SubmitEnd",
                        "event_count": 5 * 96,
                        "cycles": 5 * 96,
                        "aic_cycles": 5 * 32,
                        "aiv_cycles": 5 * 64,
                    }
                ],
            },
            "between_submit_residual": {
                "total_cycles": 20 * 96,
                "segments": [
                    {
                        "boundary": "SubmitEnd->NextSubmitBegin",
                        "event_count": 4 * 96,
                        "cycles": 20 * 96,
                        "aic_cycles": 20 * 32,
                        "aiv_cycles": 20 * 64,
                    }
                ],
            },
        },
        "overlays": {
            "Atomic": {
                "event_count": 1,
                "aggregate_duration_cycles": 2,
                "included_in_additive_totals": False,
            }
        },
        "global_submit_makespan": {"duration_cycles": 5_000_000, "duration_us": 5_000.0},
        "global_worker_completion_makespan": {"duration_cycles": 5_100_000, "duration_us": 5_100.0},
        "kernel_containment": {"submit_contained": 1, "top_level_residual": 0},
    }


def _metric_summary(cores: int, base: int) -> dict[str, int | float]:
    return {"sum": base * cores, "min": base, "mean": base + 0.5, "max": base + 1}


def _denominator_summary(cores: int) -> dict[str, Any]:
    return {
        "cores": cores,
        "scalar_submit_elapsed_ticks": _metric_summary(cores, 10_000),
        "total_cycles": _metric_summary(cores, 16_000),
        "scalar_busy": _metric_summary(cores, 12_000),
        "icache_requests": _metric_summary(cores, 2_000),
        "icache_misses": _metric_summary(cores, 200),
    }


def _phase_summary(cores: int, calls_per_core: int) -> dict[str, Any]:
    return {
        "cores": cores,
        "phase_elapsed_ticks": _metric_summary(cores, 600),
        "phase_icache_requests_observed": _metric_summary(cores, 180),
        "phase_icache_misses_observed": _metric_summary(cores, 18),
        "phase_time_share_of_scalar_submit": 0.06,
        "phase_request_observed_share_of_primary": 0.09,
        "phase_miss_observed_share_of_primary": 0.09,
        "phase_business_calls": calls_per_core * cores,
        "phase_calls_per_core": _metric_summary(cores, calls_per_core),
        "phase_zero_call_cores": 0,
        "phase_excluded_kernel_calls": 0,
    }


def _fake_capture(raw_path: Path, mode: str) -> SubmitPmuCapture:
    raw_bytes = raw_path.read_bytes()
    phase = None
    phase_summary = None
    if mode != overview_module.NONE_CAPTURE_MODE:
        phase = {
            "id": overview_module.PMU_MODE_ORDER.index(mode),
            "name": mode.removeprefix("submit-pmu-"),
            "boundary": f"{mode}-begin-to-end",
            "counter_semantics": "running_read_clear_observed_bracket",
            "time_semantics": "inner_sys_counter_ticks",
        }
        phase_summary = {
            "all": _phase_summary(96, 5),
            "aic": _phase_summary(32, 5),
            "aiv": _phase_summary(64, 5),
        }
    data = {
        "capture": {"mode": mode},
        "configuration": {"expected_submits_per_core": 5, "sys_counter_tick_ns": 1, "phase": phase},
        "window": {"global_submit_span_us": 5_000.0},
    }
    return SubmitPmuCapture(
        input_path=raw_path,
        raw_size=len(raw_bytes),
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        data=data,
        records=(),
        groups={},
        summary={
            "all": _denominator_summary(96),
            "aic": _denominator_summary(32),
            "aiv": _denominator_summary(64),
        },
        phase_summary=phase_summary,
    )


def _fake_provenance(mode: str, git_head: str = "1" * 40) -> dict[str, Any]:
    return {
        "build": {
            "git_head": git_head,
            "source_fingerprint": f"source-{mode}",
            "profiled_cache_key": ["a5", "fdwic", "Case1", mode],
        },
        "artifacts": {
            "aicore_kernel": {
                "sha256": hashlib.sha256(mode.encode()).hexdigest(),
                "text": {"sha256": hashlib.sha256(f"text-{mode}".encode()).hexdigest()},
            }
        },
    }


@dataclass
class _Evidence:
    swimlane_raw: Path
    analysis: dict[str, Any]
    pmu_dirs: list[Path]
    captures: dict[Path, SubmitPmuCapture]
    provenance: dict[Path, dict[str, Any]]


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Evidence:
    swimlane_raw = tmp_path / "swimlane" / "l2_swimlane_records.json"
    swimlane_raw.parent.mkdir()
    swimlane_raw.write_text('{"schema_version":4}\n', encoding="utf-8")
    analysis = _valid_swimlane_analysis()

    captures: dict[Path, SubmitPmuCapture] = {}
    provenance: dict[Path, dict[str, Any]] = {}
    pmu_dirs = []
    for mode in overview_module.PMU_MODE_ORDER:
        directory = tmp_path / mode
        directory.mkdir()
        raw_path = directory / overview_module.DEFAULT_INPUT_NAME
        raw_path.write_text(f'{{"mode":"{mode}"}}\n', encoding="utf-8")
        provenance_path = directory / overview_module.DEFAULT_PROVENANCE_NAME
        provenance_path.write_text(f'{{"profile":"{mode}"}}\n', encoding="utf-8")
        captures[raw_path.resolve()] = _fake_capture(raw_path, mode)
        provenance[provenance_path.resolve()] = _fake_provenance(mode)
        pmu_dirs.append(directory)

    def fake_load_capture(path: Path | str) -> SubmitPmuCapture:
        return captures[Path(path).resolve()]

    def fake_load_provenance(path: Path | str, capture: SubmitPmuCapture) -> tuple[dict[str, Any], str]:
        del capture
        resolved = Path(path).resolve()
        return provenance[resolved], hashlib.sha256(resolved.read_bytes()).hexdigest()

    monkeypatch.setattr(overview_module, "analyze_capture", lambda path: analysis)
    monkeypatch.setattr(overview_module, "load_capture", fake_load_capture)
    monkeypatch.setattr(overview_module, "load_provenance", fake_load_provenance)
    return _Evidence(swimlane_raw, analysis, pmu_dirs, captures, provenance)


def test_swimlane_analysis_requires_strict_closure_and_builds_exact_partitions(evidence: _Evidence) -> None:
    payload = overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs)
    swimlane = payload["swimlane_elf"]

    for partition_name in (
        "submit_envelope_partition",
        "submit_union_partition",
        "efdrain_partition",
        "orchestration_partition",
        "final_drain_partition",
        "worker_completion_partition",
    ):
        partition = swimlane[partition_name]
        assert partition["exact"] is True
        assert partition["parent_core_time_ticks"] == sum(row["core_time_ticks"] for row in partition["rows"])

    broken_closure = copy.deepcopy(evidence.analysis)
    broken_closure["aggregate_core_work"]["closure"]["submit_partition"]["exact"] = False
    with pytest.raises(ValueError, match="aggregate closure 'submit_partition' is not exact"):
        overview_module._validate_swimlane_analysis(broken_closure)

    broken_partition = copy.deepcopy(evidence.analysis)
    broken_partition["aggregate_core_work"]["metrics_cycles"]["submit_union"] += 96
    for core in broken_partition["per_core"]:
        core["metrics_cycles"]["submit_union"] += 1
    frequency_hz, _ = overview_module._validate_swimlane_analysis(broken_partition)
    raw_size, raw_sha256 = overview_module._sha256_file(evidence.swimlane_raw)
    with pytest.raises(ValueError, match="overview partition 'SubmitEnvelope' does not close"):
        overview_module._summarize_swimlane(
            broken_partition,
            evidence.swimlane_raw,
            raw_size,
            raw_sha256,
            frequency_hz,
        )


def test_complete_thirteen_profile_overview_and_mixed_git_heads_are_accepted(evidence: _Evidence) -> None:
    mixed_mode = overview_module.LOSER_REPLAY_CAPTURE_MODE
    mixed_directory = evidence.pmu_dirs[overview_module.PMU_MODE_ORDER.index(mixed_mode)]
    mixed_path = (mixed_directory / overview_module.DEFAULT_PROVENANCE_NAME).resolve()
    evidence.provenance[mixed_path]["build"]["git_head"] = "2" * 40

    payload = overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs)

    validation = payload["validation"]
    assert validation["status"] == "PASS"
    assert validation["pmu_profile_count"] == 13
    assert validation["pmu_profiles_complete"] is True
    assert validation["git_head_uniform"] is False
    assert validation["git_heads"] == ["1" * 40, "2" * 40]
    assert validation["swimlane_to_pmu_identity_bound"] is False
    assert payload["submit_pmu_elfs"]["whole_window"]["capture_mode"] == overview_module.NONE_CAPTURE_MODE
    assert len(payload["submit_pmu_elfs"]["phase_profiles"]) == 11
    assert payload["submit_pmu_elfs"]["calibration"]["capture_mode"] == overview_module.EMPTY_BRACKET_CAPTURE_MODE


def test_missing_and_duplicate_profiles_are_rejected(evidence: _Evidence) -> None:
    with pytest.raises(ValueError, match="requires exactly 13 Submit-PMU directories"):
        overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs[:-1])

    duplicate_dirs = [*evidence.pmu_dirs[:-1], evidence.pmu_dirs[0]]
    with pytest.raises(ValueError, match=f"duplicate Submit-PMU capture mode: {overview_module.NONE_CAPTURE_MODE}"):
        overview_module.build_overview(evidence.swimlane_raw, duplicate_dirs)


def test_pmu_task_count_must_match_swimlane(evidence: _Evidence) -> None:
    mode = overview_module.CLAIM_CAPTURE_MODE
    profile_directory = evidence.pmu_dirs[overview_module.PMU_MODE_ORDER.index(mode)]
    raw_path = (profile_directory / overview_module.DEFAULT_INPUT_NAME).resolve()
    evidence.captures[raw_path].data["configuration"]["expected_submits_per_core"] = 6

    with pytest.raises(ValueError, match=f"Submit-PMU {mode} task count does not match the swimlane capture"):
        overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs)


def test_coverage_matrix_contains_all_exclusive_spans_and_residuals(evidence: _Evidence) -> None:
    payload = overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs)
    regions = {row["swimlane_region"] for row in payload["coverage_matrix"]}
    exclusive_spans = {label for label, _ in overview_module.SUBMIT_PHASES}

    assert exclusive_spans <= regions
    assert "BetweenSubmitResidual" in regions
    assert "SubmitInternalResidual/Claim->Materialize" in regions
    assert "SubmitInternalResidual/Register->WinnerBuild" in regions
    assert "SubmitTailResidual/LoserReplay->SubmitEnd" in regions
    assert {
        "WorkerCompletion",
        "OrchestrationReplay",
        "OrchestrationSetup",
        "OrchestrationTail",
        "SubmitUnion",
        "SubmitResidual",
        "EfDrainKernelUnion",
        "EfDrainControl",
        "FinalDrain",
        "FinalDrainKernelUnion",
        "FinalDrainResidual",
        "Atomic overlay",
    } <= regions


def test_html_forbids_cross_elf_totals_and_shows_only_min_max(evidence: _Evidence) -> None:
    document = overview_module.render_overview(overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs))

    assert "各 PMU 行的占比也不能相加成 100%" in document
    assert "页面没有也不会生成跨卡合计" in document
    assert "AIC 每核 min–max" in document
    assert "AIV 每核 min–max" in document
    assert "AIC 每核 600–601" in document
    assert "AIV 每核 600–601" in document
    assert "本 ELF：57,600 / 960,000 = 6.000%" in document
    assert "泳道与 PMU 之间仅对齐 96 核拓扑和每核 Submit 数" in document
    for forbidden in ("均值", "median", "p95", "PMU phase 合计", "跨 ELF 合计"):
        assert forbidden not in document


def test_zero_primary_miss_denominator_is_rendered_as_not_applicable(evidence: _Evidence) -> None:
    payload = overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs)
    phase = payload["submit_pmu_elfs"]["phase_profiles"][0]
    phase["denominators"]["all"]["primary_icache_misses"]["sum"] = 0
    phase["phase"]["groups"]["all"]["phase_icache_misses_observed"]["sum"] = 0
    phase["phase"]["groups"]["all"]["phase_miss_observed_share_of_primary"] = 0.0

    document = overview_module.render_overview(payload)

    assert "N/A（本 ELF 分母为 0）" in document
    assert "0 / 0 = 0.000%" not in document


def test_output_lock_rejects_a_concurrent_publisher(tmp_path: Path) -> None:
    lock_path = overview_module._acquire_output_lock(tmp_path)
    try:
        with pytest.raises(ValueError, match="another overview publication owns"):
            overview_module._acquire_output_lock(tmp_path)
    finally:
        lock_path.unlink()


def test_publish_pair_rolls_back_first_file_when_second_publish_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    json_path = tmp_path / overview_module.DEFAULT_JSON_NAME
    html_path = tmp_path / overview_module.DEFAULT_HTML_NAME
    json_path.write_bytes(b"old-json")
    html_path.write_bytes(b"old-html")
    real_replace = os.replace
    failed = False

    def fail_second_publish(source: Path | str, destination: Path | str) -> None:
        nonlocal failed
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path == html_path and source_path.name.endswith(".pending") and not failed:
            failed = True
            raise OSError("injected HTML publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(overview_module.os, "replace", fail_second_publish)
    with pytest.raises(OSError, match="injected HTML publication failure"):
        overview_module._publish_pair(json_path, "new-json", html_path, "new-html", overwrite=True)

    assert failed is True
    assert json_path.read_bytes() == b"old-json"
    assert html_path.read_bytes() == b"old-html"
    assert not list(tmp_path.glob("*.pending"))
    assert not list(tmp_path.glob("*.restore"))
