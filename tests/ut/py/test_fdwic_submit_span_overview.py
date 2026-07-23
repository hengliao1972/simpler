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
import re
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
        "non_scalar_busy_cycles": _metric_summary(cores, 4_000),
        "icache_requests": _metric_summary(cores, 2_000),
        "icache_misses": _metric_summary(cores, 200),
    }


def _phase_summary(cores: int, calls_per_core: int) -> dict[str, Any]:
    return {
        "cores": cores,
        "phase_total_cycles_observed": _metric_summary(cores, 1_000),
        "phase_scalar_busy_observed": _metric_summary(cores, 600),
        "phase_non_scalar_busy_cycles": _metric_summary(cores, 400),
        "shadow_scalar_loss": _metric_summary(cores, 20),
        "phase_elapsed_ticks": _metric_summary(cores, 600),
        "phase_icache_requests_observed": _metric_summary(cores, 180),
        "phase_icache_misses_observed": _metric_summary(cores, 18),
        "phase_total_share_of_pmu_total": 0.0625,
        "phase_scalar_share_of_whole_scalar": 0.05,
        "phase_scalar_busy_share_of_phase_total": 0.6,
        "phase_request_observed_share_of_primary": 0.09,
        "phase_miss_observed_share_of_primary": 0.09,
        "phase_business_calls": calls_per_core * cores,
        "phase_calls_per_core": _metric_summary(cores, calls_per_core),
        "phase_zero_call_cores": 0,
        "phase_end_reads": calls_per_core * cores,
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
            "time_semantics": "boundary_diagnostic_sys_cnt_between_observers",
            "pmu_observation": {"boundary": "running-read-clear"},
        }
        phase_summary = {
            "all": _phase_summary(96, 5),
            "aic": _phase_summary(32, 5),
            "aiv": _phase_summary(64, 5),
        }
    data = {
        "configuration": {
            "num_cores": 96,
            "aic_cores": 32,
            "aiv_cores": 64,
            "expected_submits_per_core": 5,
            "sys_counter_tick_ns": 1,
            "selectors": {"test": "fixed"},
            "linked_kernel_exclusion": {"enabled": True},
            "return_ready_atomic_exclusion": {"enabled": True},
            "counter_width_bits": {"total": 64, "programmable": 32},
            "programmable_counter_risk_threshold": (1 << 30) - 1,
            "pmu_cycles_per_ns": {"all": 1.649844, "aic": 1.650062, "aiv": 1.649731},
            "phase": phase,
        },
        "capture": {
            "mode": mode,
            "window_scope": "per_core_first_submit_begin_to_last_submit_end",
        },
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

    assert swimlane["core_count"] == 96
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


def test_complete_thirteen_profile_overview_accepts_one_revision_and_rejects_mixed_phase_calibration(
    evidence: _Evidence,
) -> None:
    payload = overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs)

    validation = payload["validation"]
    assert validation["status"] == "PASS"
    assert validation["pmu_profile_count"] == 13
    assert validation["pmu_profiles_complete"] is True
    assert validation["git_head_uniform"] is True
    assert validation["git_heads"] == ["1" * 40]
    assert validation["swimlane_to_pmu_identity_bound"] is False
    assert payload["submit_pmu_elfs"]["whole_window"]["capture_mode"] == overview_module.NONE_CAPTURE_MODE
    assert len(payload["submit_pmu_elfs"]["phase_profiles"]) == 11
    assert payload["submit_pmu_elfs"]["calibration"]["capture_mode"] == overview_module.EMPTY_BRACKET_CAPTURE_MODE
    assert payload["schema"] == "fdwic-submit-span-overview-v5"
    assert payload["semantics"]["cross_elf_phase_shares_additive"] is False
    assert payload["semantics"]["cross_elf_synthetic_phase_sum_is_exact_overhead"] is False

    phase_group = payload["submit_pmu_elfs"]["phase_profiles"][0]["phase"]["groups"]["all"]
    assert phase_group["phase_total_cycles_observed"]["sum"] == 96_000
    assert phase_group["phase_scalar_busy_observed"]["sum"] == 57_600
    assert phase_group["phase_non_scalar_busy_cycles"]["sum"] == 38_400
    assert phase_group["phase_total_share_of_pmu_total"] == 0.0625
    assert phase_group["phase_scalar_share_of_whole_scalar"] == 0.05
    assert phase_group["phase_scalar_busy_share_of_phase_total"] == 0.6
    for profile in payload["submit_pmu_elfs"]["phase_profiles"]:
        reference = profile["recording_cost_reference"]
        assert reference["target_capture_mode"] == profile["capture_mode"]
        assert reference["calibration_capture_mode"] == overview_module.EMPTY_BRACKET_CAPTURE_MODE
        assert reference["exact_correction"] is False
        assert reference["raw_whole_denominator_is_unchanged"] is True
        assert reference["whole_recording_cost_is_not_measured"] is True
    synthetic = payload["submit_pmu_elfs"]["synthetic_phase_sum_vs_none"]
    assert synthetic["phase_profile_count"] == 11
    assert synthetic["empty_bracket_excluded_from_raw_phase_sum"] is True
    assert synthetic["empty_bracket_used_as_recording_cost_calibration"] is True
    assert synthetic["formal_partition_closure"] is False
    assert synthetic["exact_recording_cost_correction"] is False

    mixed_mode = overview_module.LOSER_REPLAY_CAPTURE_MODE
    mixed_directory = evidence.pmu_dirs[overview_module.PMU_MODE_ORDER.index(mixed_mode)]
    mixed_path = (mixed_directory / overview_module.DEFAULT_PROVENANCE_NAME).resolve()
    evidence.provenance[mixed_path]["build"]["git_head"] = "2" * 40
    with pytest.raises(
        ValueError,
        match=(
            f"Submit-PMU {mixed_mode} and {overview_module.EMPTY_BRACKET_CAPTURE_MODE} "
            "provenance git heads do not match"
        ),
    ):
        overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs)


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


def test_html_distinguishes_formal_ratios_from_synthetic_cross_elf_diagnostic(evidence: _Evidence) -> None:
    document = overview_module.render_overview(overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs))

    assert all(line == line.rstrip() for line in document.splitlines())
    assert "各 PMU 行的占比也不能相加成 100%" in document
    assert "业务 phase 的主显示值" in document
    assert "先从 raw observed 分子中扣除" in document
    assert "再除以本 ELF 的 raw whole" in document
    assert "raw observed / raw whole 同格保留" in document
    assert "没有测到 whole 窗口中的全部记录工作" in document
    assert "只是更接近业务量级的参考值" in document
    assert "不是“纯业务阶段 / 纯业务整窗”的精确占比" in document
    assert "原始观测合计" in document
    assert "空区间估算的记录代码开销" in document
    assert "扣除上述估算后的参考值" in document
    assert "不是精确校正" in document
    assert "泳道同父区间" in document
    assert "PMU 局部记录扣除参考" in document
    assert "扣局部记录估算后的 Phase PMU / raw whole PMU" in document
    assert "Scalar 局部记录扣除参考" in document
    assert "扣局部记录估算后的 Phase scalar / raw whole scalar" in document
    assert "raw 比例同格保留" in document
    assert '<col class="partition-region"><col class="partition-time"><col class="partition-share">' in document
    assert ".partition table { min-width:920px; table-layout:fixed; }" in document
    assert ".partition table { min-width:1120px; }" not in document
    assert "white-space:normal; vertical-align:top; overflow-wrap:anywhere;" in document
    assert "并非泳道父区间，也不是" in document
    assert "AIC 每核 min–max" in document
    assert "AIV 每核 min–max" in document
    assert "Phase PMU total" in document
    assert "Phase scalar busy" in document
    assert "非 Scalar-busy 残余" in document
    assert "raw Phase total / raw whole total：96,000 / 1,536,000 = 6.250%" in document
    assert "参考 Scalar / 参考 Phase total：N/A；raw 60.000%" in document
    assert "raw Phase scalar / raw whole scalar：57,600 / 1,152,000 = 5.000%" in document
    assert "参考 residual / 参考 Phase total：N/A；raw 40.000%" in document
    assert "AIC 每核 1,000–1,001 cycles（0.606–0.607 µs）" in document
    assert "SYS 边界诊断" in document
    assert "只核验 phase 边界闭合，不作为阶段主时间或占比分母" in document
    assert "1.649844/1.650062/1.649731 cycles/ns" in document
    assert "SYS 边界闭合诊断（raw ticks）" in document
    assert "不参与 PMU 1.65 GHz 时间换算或阶段比例" in document
    assert "不能求和后与" in document
    assert "<code>submit-pmu-none</code> 做正式判等" in document
    assert "Scalar phase core-time" not in document
    assert "Observer pair empirical elapsed" not in document
    assert "泳道与 PMU 之间仅对齐 96 核拓扑和每核 Submit 数" in document
    for forbidden in ("median", "p95", "净性能收益", "精确观察开销"):
        assert forbidden not in document


def test_submit_union_uses_per_core_means_and_decomposes_ten_phase_observations(evidence: _Evidence) -> None:
    payload = overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs)
    pmu = payload["submit_pmu_elfs"]
    phase_by_mode = {str(item["capture_mode"]): item for item in pmu["phase_profiles"]}
    phase_by_metric = {
        metric: phase_by_mode[mode] for metric, mode in overview_module.PARTITION_PMU_MODE_BY_METRIC.items()
    }
    profiles = [phase_by_mode[mode] for mode in overview_module.SUBMIT_UNION_PMU_MODES]
    diagnostic = overview_module._partition_mean_diagnostic(
        payload["swimlane_elf"]["submit_union_partition"],
        phase_by_metric,
        profiles,
        pmu["whole_window"],
        payload["swimlane_elf"]["core_count"],
    )

    assert diagnostic["profiles"] == list(overview_module.SUBMIT_UNION_PMU_MODES)
    assert overview_module.ARG_BUILD_CAPTURE_MODE in diagnostic["profiles"]
    assert overview_module.SUBMIT_TRANSITION_CAPTURE_MODE not in diagnostic["profiles"]
    assert overview_module.NONE_CAPTURE_MODE not in diagnostic["profiles"]
    assert overview_module.EMPTY_BRACKET_CAPTURE_MODE not in diagnostic["profiles"]
    assert diagnostic["direct_mapped_profile_count"] == 9
    assert diagnostic["unmapped_labels"] == ["SubmitInternalResidual", "SubmitTailResidual"]
    decomposition = diagnostic["decomposition"]
    all_group = decomposition["groups"]["all"]
    all_metrics = all_group["metrics"]
    assert all_group["cores"] == 96
    assert all_group["phase_record_pairs"] == 4_800
    assert all_group["phase_record_pairs_per_core"] == 50.0
    assert all_group["phase_business_calls"] == 4_800
    assert all_group["phase_business_calls_per_core"] == 50.0
    assert all_group["empty_calibration_record_pairs"] == 480
    assert all_metrics["pmu_total_cycles"] == {
        "label": "PMU total",
        "unit": "cycles",
        "raw_observed_sum": 960_000,
        "raw_observed_mean": 10_000.0,
        "submit_none_sum": 1_536_000,
        "submit_none_mean": 16_000.0,
        "raw_observed_ratio_to_submit_none": 0.625,
        "empty_cost_per_record_pair": 200.0,
        "recording_cost_estimate_sum": 960_000.0,
        "recording_cost_estimate_mean": 10_000.0,
        "recording_cost_estimate_share_of_raw": 1.0,
        "recording_cost_estimate_ratio_to_submit_none": 0.625,
        "after_recording_cost_reference_sum": 0.0,
        "after_recording_cost_reference_mean": 0.0,
        "after_recording_cost_reference_ratio_to_submit_none": 0.0,
        "phase_field": "phase_total_cycles_observed",
        "submit_none_field": "pmu_total_cycles",
    }
    assert all_metrics["scalar_busy_cycles"]["raw_observed_sum"] == 576_000
    assert all_metrics["scalar_busy_cycles"]["submit_none_sum"] == 1_152_000
    assert all_metrics["scalar_busy_cycles"]["recording_cost_estimate_ratio_to_submit_none"] == 0.5
    assert all_metrics["scalar_busy_cycles"]["after_recording_cost_reference_sum"] == 0
    assert all_metrics["non_scalar_busy_cycles"]["recording_cost_estimate_ratio_to_submit_none"] == 1.0
    assert all_metrics["icache_requests"]["recording_cost_estimate_ratio_to_submit_none"] == 0.9
    assert all_metrics["icache_misses"]["recording_cost_estimate_ratio_to_submit_none"] == 0.9

    changed_profiles = copy.deepcopy(profiles)
    changed_profiles[0]["denominators"]["all"]["pmu_total_cycles"]["sum"] *= 10
    changed = overview_module._partition_mean_diagnostic(
        payload["swimlane_elf"]["submit_union_partition"],
        phase_by_metric,
        changed_profiles,
        pmu["whole_window"],
        payload["swimlane_elf"]["core_count"],
    )
    changed_metric = changed["decomposition"]["groups"]["all"]["metrics"]["pmu_total_cycles"]
    assert changed_metric["raw_observed_sum"] == 960_000
    assert changed_metric["recording_cost_estimate_ratio_to_submit_none"] == 0.625

    document = overview_module.render_overview(payload)
    match = re.search(
        r'<div class="partition">\s*<h3>SubmitUnion</h3>(.*?)</table>',
        document,
        flags=re.DOTALL,
    )
    assert match is not None
    submit_union_html = match.group(1)
    assert "平均每核时间" in submit_union_html
    assert "均值占比对照" in submit_union_html
    assert "泳道每核均值 / 同父区间每核均值" in submit_union_html
    assert "扣局部记录估算后的 Phase PMU 每核均值 / raw whole PMU 每核均值" in submit_union_html
    assert "扣局部记录估算后的 Phase scalar 每核均值 / raw whole scalar 每核均值" in submit_union_html
    assert "Σ core-time" not in submit_union_html
    assert "<code>EfDrain</code></td><td>0.020 µs</td>" in submit_union_html
    assert 'title="0.020 / 0.080 = 25.000%">25.000%</td>' in submit_union_html
    assert "submit-pmu-efdrain-control：参考 每核均值 0.000 / raw whole 16,000.000 = 0.000%" in submit_union_html
    assert "raw 每核均值 1,000.000 / 16,000.000 = 6.250%" in submit_union_html
    assert "submit-pmu-efdrain-control：参考 每核均值 0.000 / raw whole 12,000.000 = 0.000%" in submit_union_html
    assert "raw 每核均值 600.000 / 12,000.000 = 5.000%" in submit_union_html
    assert 'data-partition-total="per-core-mean"' in submit_union_html
    assert "SubmitUnion 平均每核时间合计" in submit_union_html
    assert "<strong>0.080 µs</strong><small>11 个分段每核均值之和</small>" in submit_union_html
    assert "<td><strong>100.000%</strong></td>" in submit_union_html
    assert 'data-phase-profile-count="10"' in submit_union_html
    assert "原始分段观测合计" in submit_union_html
    assert "10,000.000 cycles/core" in submit_union_html
    assert "空区间估算的记录代码开销" in submit_union_html
    assert "扣除上述估算后的参考值" in submit_union_html
    assert "0.000 cycles/core" in submit_union_html
    assert "6,000.000 cycles/core" in submit_union_html
    assert 'data-partition-comparison="submit-pmu-none-mean"' in submit_union_html
    assert "16,000.000 cycles/core" in submit_union_html
    assert "12,000.000 cycles/core" in submit_union_html
    assert "包含 BetweenSubmitResidual / SubmitTransition" in submit_union_html
    assert "raw 含分段记录代码自身开销，不能直接拿来解释业务耗时" in submit_union_html
    assert "偏离 100%" not in submit_union_html
    assert "净观察开销" not in submit_union_html


def test_synthetic_phase_sum_vs_none_uses_raw_metrics_not_summed_phase_shares(evidence: _Evidence) -> None:
    payload = overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs)
    diagnostic = payload["submit_pmu_elfs"]["synthetic_phase_sum_vs_none"]
    all_group = diagnostic["groups"]["all"]
    all_metrics = all_group["metrics"]

    assert diagnostic["included_profiles"] == [
        mode
        for mode in overview_module.PMU_MODE_ORDER
        if mode not in {overview_module.NONE_CAPTURE_MODE, overview_module.EMPTY_BRACKET_CAPTURE_MODE}
    ]
    assert all_group["cores"] == 96
    assert all_group["phase_record_pairs"] == 5_280
    assert all_group["phase_record_pairs_per_core"] == 55.0
    assert all_group["phase_business_calls"] == 5_280
    assert all_group["phase_business_calls_per_core"] == 55.0
    assert all_group["empty_calibration_record_pairs"] == 480
    assert all_metrics["pmu_total_cycles"] == {
        "label": "PMU total",
        "unit": "cycles",
        "raw_observed_sum": 1_056_000,
        "raw_observed_mean": 11_000.0,
        "submit_none_sum": 1_536_000,
        "submit_none_mean": 16_000.0,
        "raw_observed_ratio_to_submit_none": 0.6875,
        "empty_cost_per_record_pair": 200.0,
        "recording_cost_estimate_sum": 1_056_000.0,
        "recording_cost_estimate_mean": 11_000.0,
        "recording_cost_estimate_share_of_raw": 1.0,
        "recording_cost_estimate_ratio_to_submit_none": 0.6875,
        "after_recording_cost_reference_sum": 0.0,
        "after_recording_cost_reference_mean": 0.0,
        "after_recording_cost_reference_ratio_to_submit_none": 0.0,
        "phase_field": "phase_total_cycles_observed",
        "submit_none_field": "pmu_total_cycles",
    }
    assert all_metrics["scalar_busy_cycles"]["raw_observed_sum"] == 633_600
    assert all_metrics["scalar_busy_cycles"]["submit_none_sum"] == 1_152_000
    assert all_metrics["scalar_busy_cycles"]["recording_cost_estimate_ratio_to_submit_none"] == 0.55
    assert all_metrics["scalar_busy_cycles"]["recording_cost_estimate_share_of_raw"] == 1.0
    assert all_metrics["scalar_busy_cycles"]["after_recording_cost_reference_sum"] == 0
    assert all_metrics["non_scalar_busy_cycles"]["recording_cost_estimate_ratio_to_submit_none"] == 1.1
    assert all_metrics["icache_requests"]["recording_cost_estimate_ratio_to_submit_none"] == pytest.approx(0.99)
    assert all_metrics["icache_requests"]["after_recording_cost_reference_sum"] == 0
    assert all_metrics["icache_misses"]["recording_cost_estimate_ratio_to_submit_none"] == pytest.approx(0.99)
    assert all_metrics["icache_misses"]["after_recording_cost_reference_sum"] == pytest.approx(0)
    for group_name in ("all", "aic", "aiv"):
        metrics = diagnostic["groups"][group_name]["metrics"]
        for field in (
            "raw_observed_sum",
            "recording_cost_estimate_sum",
            "after_recording_cost_reference_sum",
        ):
            assert metrics["pmu_total_cycles"][field] == pytest.approx(
                metrics["scalar_busy_cycles"][field] + metrics["non_scalar_busy_cycles"][field]
            )
        for metric in metrics.values():
            assert metric["raw_observed_sum"] == pytest.approx(
                metric["recording_cost_estimate_sum"] + metric["after_recording_cost_reference_sum"]
            )

    document = overview_module.render_overview(payload)
    assert "11 个业务分段的记录开销拆分（含 SubmitTransition）" in document
    assert "原始观测合计" in document
    assert "空区间估算的记录代码开销" in document
    assert "扣除上述估算后的参考值" in document
    assert "<code>submit-pmu-none</code>" in document
    assert "empty-bracket 不参与" not in document
    assert "偏离 100%" not in document
    assert "净观察开销" not in document
    assert "图形封顶 600%" not in document


def test_recording_cost_uses_record_pairs_and_role_weighted_empty_cost(evidence: _Evidence) -> None:
    summary = _phase_summary(32, 5)
    summary["phase_excluded_kernel_calls"] = 7
    summary["phase_end_reads"] += 7
    summarized = overview_module._phase_group_summary(summary)
    assert summarized["phase_business_calls"] == 160
    assert summarized["phase_record_pairs"] == 167

    payload = overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs)
    pmu = payload["submit_pmu_elfs"]
    profile = copy.deepcopy(pmu["phase_profiles"][0])
    role_pairs = {"aic": 32, "aiv": 640}
    role_raw_total = {"aic": 3_200, "aiv": 192_000}
    role_cost = {"aic": 100.0, "aiv": 300.0}
    for role in ("aic", "aiv"):
        phase_group = profile["phase"]["groups"][role]
        phase_group["phase_record_pairs"] = role_pairs[role]
        phase_group["phase_business_calls"] = role_pairs[role]
        phase_group["phase_total_cycles_observed"]["sum"] = role_raw_total[role]
        reference_group = profile["recording_cost_reference"]["groups"][role]
        reference_group["target_record_pairs"] = role_pairs[role]
        metric = reference_group["metrics"]["pmu_total_cycles"]
        metric["raw_phase_observed_sum"] = role_raw_total[role]
        metric["recording_cost_estimate_sum"] = role_cost[role] * role_pairs[role]
        metric["after_recording_cost_reference_sum"] = (
            metric["raw_phase_observed_sum"] - metric["recording_cost_estimate_sum"]
        )

    all_phase = profile["phase"]["groups"]["all"]
    all_phase["phase_record_pairs"] = sum(role_pairs.values())
    all_phase["phase_business_calls"] = sum(role_pairs.values())
    all_phase["phase_total_cycles_observed"]["sum"] = sum(role_raw_total.values())
    all_reference = profile["recording_cost_reference"]["groups"]["all"]
    all_reference["target_record_pairs"] = sum(role_pairs.values())
    all_reference_metric = all_reference["metrics"]["pmu_total_cycles"]
    all_reference_metric["raw_phase_observed_sum"] = sum(role_raw_total.values())
    all_reference_metric["recording_cost_estimate_sum"] = sum(
        role_cost[role] * role_pairs[role] for role in ("aic", "aiv")
    )
    all_reference_metric["after_recording_cost_reference_sum"] = (
        all_reference_metric["raw_phase_observed_sum"] - all_reference_metric["recording_cost_estimate_sum"]
    )

    decomposition = overview_module._synthetic_phase_sum_vs_none(
        [profile],
        pmu["whole_window"],
    )
    groups = decomposition["groups"]
    all_metric = groups["all"]["metrics"]["pmu_total_cycles"]
    aic_estimate = groups["aic"]["metrics"]["pmu_total_cycles"]["recording_cost_estimate_sum"]
    aiv_estimate = groups["aiv"]["metrics"]["pmu_total_cycles"]["recording_cost_estimate_sum"]
    expected_role_weighted = 100.0 * 32 + 300.0 * 640
    naive_all_average = (112_000 / 480) * 672

    assert all_metric["recording_cost_estimate_sum"] == expected_role_weighted
    assert all_metric["recording_cost_estimate_sum"] == aic_estimate + aiv_estimate
    assert all_metric["recording_cost_estimate_sum"] != pytest.approx(naive_all_average)

    more_record_pairs = copy.deepcopy(profile)
    more_record_pairs["phase"]["groups"]["aic"]["phase_excluded_kernel_calls"] += 8
    more_record_pairs["phase"]["groups"]["aic"]["phase_record_pairs"] += 8
    more_record_pairs["phase"]["groups"]["all"]["phase_excluded_kernel_calls"] += 8
    more_record_pairs["phase"]["groups"]["all"]["phase_record_pairs"] += 8
    for group_name in ("aic", "all"):
        reference_group = more_record_pairs["recording_cost_reference"]["groups"][group_name]
        reference_group["target_record_pairs"] += 8
        metric = reference_group["metrics"]["pmu_total_cycles"]
        metric["raw_phase_observed_sum"] += 8 * role_cost["aic"]
        metric["recording_cost_estimate_sum"] += 8 * role_cost["aic"]
    increased = overview_module._synthetic_phase_sum_vs_none(
        [more_record_pairs],
        pmu["whole_window"],
    )
    increased_estimate = increased["groups"]["all"]["metrics"]["pmu_total_cycles"]["recording_cost_estimate_sum"]
    assert increased_estimate == expected_role_weighted + 8 * 100.0


def test_partition_rows_join_only_semantically_matching_phase_profiles(evidence: _Evidence) -> None:
    payload = overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs)
    phases = {item["capture_mode"]: item for item in payload["submit_pmu_elfs"]["phase_profiles"]}

    def update_reference(
        phase: dict[str, Any],
        metric_name: str,
        raw_phase_sum: int,
        reference_sum: int,
    ) -> None:
        values = {
            "all": (raw_phase_sum, reference_sum),
            "aic": (raw_phase_sum / 3, reference_sum / 3),
            "aiv": (raw_phase_sum * 2 / 3, reference_sum * 2 / 3),
        }
        for group_name, (group_raw, group_reference) in values.items():
            metric = phase["recording_cost_reference"]["groups"][group_name]["metrics"][metric_name]
            raw_whole_sum = int(metric["raw_whole_sum"])
            estimate = group_raw - group_reference
            metric.update(
                {
                    "raw_phase_observed_sum": group_raw,
                    "raw_phase_observed_ratio_to_raw_whole": group_raw / raw_whole_sum,
                    "recording_cost_estimate_sum": estimate,
                    "recording_cost_estimate_share_of_raw_phase": estimate / group_raw,
                    "after_recording_cost_reference_sum": group_reference,
                    "after_recording_cost_reference_ratio_to_raw_whole": group_reference / raw_whole_sum,
                }
            )

    claim = phases[overview_module.CLAIM_CAPTURE_MODE]
    claim_all = claim["phase"]["groups"]["all"]
    claim_all["phase_total_cycles_observed"]["sum"] = 192_000
    claim_all["phase_scalar_busy_observed"]["sum"] = 115_200
    claim_all["phase_total_share_of_pmu_total"] = 0.125
    claim_all["phase_scalar_share_of_whole_scalar"] = 0.1
    claim["phase"]["groups"]["aic"]["phase_total_cycles_observed"]["sum"] = 64_000
    claim["phase"]["groups"]["aiv"]["phase_total_cycles_observed"]["sum"] = 128_000
    claim["phase"]["groups"]["aic"]["phase_scalar_busy_observed"]["sum"] = 38_400
    claim["phase"]["groups"]["aiv"]["phase_scalar_busy_observed"]["sum"] = 76_800
    update_reference(claim, "pmu_total_cycles", 192_000, 96_000)
    update_reference(claim, "scalar_busy_cycles", 115_200, 57_600)

    efdrain = phases[overview_module.EFDRAIN_CONTROL_CAPTURE_MODE]
    efdrain_all = efdrain["phase"]["groups"]["all"]
    efdrain_all["phase_total_cycles_observed"]["sum"] = 384_000
    efdrain_all["phase_scalar_busy_observed"]["sum"] = 230_400
    efdrain_all["phase_total_share_of_pmu_total"] = 0.25
    efdrain_all["phase_scalar_share_of_whole_scalar"] = 0.2
    efdrain["phase"]["groups"]["aic"]["phase_total_cycles_observed"]["sum"] = 128_000
    efdrain["phase"]["groups"]["aiv"]["phase_total_cycles_observed"]["sum"] = 256_000
    efdrain["phase"]["groups"]["aic"]["phase_scalar_busy_observed"]["sum"] = 76_800
    efdrain["phase"]["groups"]["aiv"]["phase_scalar_busy_observed"]["sum"] = 153_600
    update_reference(efdrain, "pmu_total_cycles", 384_000, 288_000)
    update_reference(efdrain, "scalar_busy_cycles", 230_400, 172_800)

    document = overview_module.render_overview(payload)

    def row(metric: str) -> str:
        match = re.search(
            rf'<tr data-swimlane-metric="{re.escape(metric)}"[^>]*>.*?</tr>',
            document,
            flags=re.DOTALL,
        )
        assert match is not None
        return match.group(0)

    claim_row = row("claim")
    assert f'data-pmu-profile="{overview_module.CLAIM_CAPTURE_MODE}"' in claim_row
    assert 'data-pmu-mapping="same-business-boundary"' in claim_row
    assert 'data-pmu-reference-share="6.250%"' in claim_row
    assert 'data-pmu-raw-share="12.500%"' in claim_row
    assert 'data-scalar-reference-share="5.000%"' in claim_row
    assert 'data-scalar-raw-share="10.000%"' in claim_row
    assert claim_row.count("<strong>6.250%</strong>") == 1
    assert claim_row.count("<strong>5.000%</strong>") == 1
    assert "raw 12.500%" in claim_row
    assert "raw 10.000%" in claim_row

    efdrain_row = row("efdrain")
    assert f'data-pmu-profile="{overview_module.EFDRAIN_CONTROL_CAPTURE_MODE}"' in efdrain_row
    assert 'data-pmu-mapping="control-only"' in efdrain_row
    assert 'data-pmu-reference-share="18.750%"' in efdrain_row
    assert 'data-pmu-raw-share="25.000%"' in efdrain_row
    assert 'data-scalar-reference-share="15.000%"' in efdrain_row
    assert 'data-scalar-raw-share="20.000%"' in efdrain_row
    assert "<strong>18.750%</strong>" in efdrain_row
    assert "<strong>15.000%</strong>" in efdrain_row
    assert "raw 25.000%" in efdrain_row
    assert "raw 20.000%" in efdrain_row

    expected_links = {
        "between_submit_residual": (
            overview_module.SUBMIT_TRANSITION_CAPTURE_MODE,
            "adjacent-submit-boundary",
        ),
        "efdrain": (overview_module.EFDRAIN_CONTROL_CAPTURE_MODE, "control-only"),
        "efdrain_control": (overview_module.EFDRAIN_CONTROL_CAPTURE_MODE, "control-only"),
        "materialize": (overview_module.MATERIALIZE_CAPTURE_MODE, "same-business-boundary"),
        "prepare_map": (overview_module.PREPARE_MAP_CAPTURE_MODE, "same-business-boundary"),
        "claim": (overview_module.CLAIM_CAPTURE_MODE, "same-business-boundary"),
        "fanin": (overview_module.FANIN_CAPTURE_MODE, "same-business-boundary"),
        "register": (overview_module.REGISTER_CAPTURE_MODE, "same-business-boundary"),
        "winner_build": (overview_module.WINNER_BUILD_CAPTURE_MODE, "control-only"),
        "alloc_complete": (overview_module.ALLOC_COMPLETE_CAPTURE_MODE, "control-only"),
        "loser_replay": (overview_module.LOSER_REPLAY_CAPTURE_MODE, "same-business-boundary"),
    }
    for metric, (capture_mode, mapping) in expected_links.items():
        linked_row = row(metric)
        assert f'data-pmu-profile="{capture_mode}"' in linked_row
        assert f'data-pmu-mapping="{mapping}"' in linked_row

    for metric in (
        "submit_union",
        "submit_internal_residual",
        "submit_tail_residual",
        "efdrain_kernel_union",
        "orchestration_replay",
        "final_drain_kernel_union",
    ):
        unmapped_row = row(metric)
        assert "data-pmu-profile" not in unmapped_row
        assert unmapped_row.count('class="evidence-ratio evidence-na">—</td>') == 2


def test_zero_raw_whole_denominator_is_rendered_as_not_applicable(evidence: _Evidence) -> None:
    payload = overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs)
    phase = payload["submit_pmu_elfs"]["phase_profiles"][0]
    for group_name in ("all", "aic", "aiv"):
        reference_metric = phase["recording_cost_reference"]["groups"][group_name]["metrics"]["pmu_total_cycles"]
        reference_metric["raw_phase_observed_sum"] = 0
        reference_metric["raw_whole_sum"] = 0
        reference_metric["raw_phase_observed_ratio_to_raw_whole"] = None
        reference_metric["recording_cost_estimate_sum"] = 0
        reference_metric["recording_cost_estimate_share_of_raw_phase"] = None
        reference_metric["after_recording_cost_reference_sum"] = 0
        reference_metric["after_recording_cost_reference_ratio_to_raw_whole"] = None

    document = overview_module.render_overview(payload)

    assert 'data-pmu-reference-share="N/A"' in document
    assert 'data-pmu-raw-share="N/A"' in document
    assert "<strong>N/A</strong>" in document
    assert "0 / 0 = 0.000%" not in document


def test_partition_row_preserves_negative_reference_and_keeps_raw_secondary(evidence: _Evidence) -> None:
    payload = overview_module.build_overview(evidence.swimlane_raw, evidence.pmu_dirs)
    phase = next(
        profile
        for profile in payload["submit_pmu_elfs"]["phase_profiles"]
        if profile["capture_mode"] == overview_module.EFDRAIN_CONTROL_CAPTURE_MODE
    )
    for group_name in ("all", "aic", "aiv"):
        metric = phase["recording_cost_reference"]["groups"][group_name]["metrics"]["pmu_total_cycles"]
        raw_sum = float(metric["raw_phase_observed_sum"])
        raw_whole = float(metric["raw_whole_sum"])
        metric["recording_cost_estimate_sum"] = raw_sum * 2
        metric["recording_cost_estimate_share_of_raw_phase"] = 2.0
        metric["after_recording_cost_reference_sum"] = -raw_sum
        metric["after_recording_cost_reference_ratio_to_raw_whole"] = -raw_sum / raw_whole

    document = overview_module.render_overview(payload)
    match = re.search(
        r'<tr data-swimlane-metric="efdrain"[^>]*>.*?</tr>',
        document,
        flags=re.DOTALL,
    )
    assert match is not None
    row = match.group(0)
    assert 'data-pmu-reference-share="-6.250%"' in row
    assert 'data-pmu-raw-share="6.250%"' in row
    assert "<strong>-6.250%</strong>" in row
    assert "raw 6.250%" in row


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
