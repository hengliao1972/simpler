# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Regression tests for the production FDWIC Submit-PMU raw-to-HTML path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

import simpler_setup.tools.fdwic_submit_pmu_report as report_module
from simpler_setup.tools.fdwic_submit_pmu_report import (
    ARG_BUILD_CAPTURE_MODE,
    CLAIM_CAPTURE_MODE,
    DEFAULT_INPUT_NAME,
    DEFAULT_OUTPUT_NAME,
    EMPTY_BRACKET_CAPTURE_MODE,
    MATERIALIZE_CAPTURE_MODE,
    PHASE_REQUIRED_STATUS_MASK,
    REGISTER_CAPTURE_MODE,
    REQUIRED_STATUS_MASK,
    load_capture,
    render_report,
    write_report,
)


def _physical_layout() -> tuple[list[int], list[int]]:
    aic = [*range(16), *range(54, 70)]
    aiv: list[int] = []
    for physical_id in aic:
        die_base = physical_id // 54 * 54
        local = physical_id % 54
        aiv.extend((die_base + 18 + 2 * local, die_base + 18 + 2 * local + 1))
    return aic, aiv


def _bitmap_words(physical_ids: list[int]) -> list[int]:
    words = [0, 0, 0, 0]
    for physical_id in physical_ids:
        words[physical_id // 32] |= 1 << (physical_id % 32)
    return words


def _metric(values: list[int]) -> dict[str, int | float]:
    return {
        "sum": sum(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = [record["total_cycles"] for record in records]
    scalar = [record["scalar_busy"] for record in records]
    requests = [record["icache_requests"] for record in records]
    misses = [record["icache_misses"] for record in records]
    return {
        "cores": len(records),
        "total_cycles": _metric(total),
        "scalar_busy": _metric(scalar),
        "icache_requests": _metric(requests),
        "icache_misses": _metric(misses),
        "scalar_busy_share": sum(scalar) / sum(total),
        "icache_miss_rate": sum(misses) / sum(requests),
    }


def _valid_capture() -> dict[str, Any]:
    aic_physical, aiv_physical = _physical_layout()
    records: list[dict[str, Any]] = []
    for logical_core_id in range(96):
        is_aic = logical_core_id < 32
        if is_aic:
            physical_core_id = aic_physical[logical_core_id]
            block_id = logical_core_id
            lane = 0
            role = "aic"
        else:
            aiv_ordinal = logical_core_id - 32
            physical_core_id = aiv_physical[aiv_ordinal]
            block_id = aiv_ordinal // 2
            lane = 1 + aiv_ordinal % 2
            role = "aiv"
        start = 1_000 + logical_core_id * 3
        end = 11_000 + logical_core_id * 3
        requests = 2_000 + logical_core_id
        misses = 200 + logical_core_id % 5
        records.append(
            {
                "logical_core_id": logical_core_id,
                "physical_core_id": physical_core_id,
                "role": role,
                "block_id": block_id,
                "lane": lane,
                "submit_count": 5,
                "expected_submit_count": 5,
                "first_submit_start_tick": start,
                "last_submit_end_tick": end,
                "submit_elapsed_ticks": end - start,
                "total_cycles": 16_000 + logical_core_id,
                "scalar_busy": 12_000 + logical_core_id,
                "icache_requests": requests,
                "icache_misses": misses,
                "shadow_icache_requests": requests,
                "shadow_icache_misses": misses,
                "status": REQUIRED_STATUS_MASK,
            }
        )

    all_physical = aic_physical + aiv_physical
    grouped = {
        "all": records,
        "aic": records[:32],
        "aiv": records[32:],
    }
    first_tick = min(record["first_submit_start_tick"] for record in records)
    last_tick = max(record["last_submit_end_tick"] for record in records)
    return {
        "schema": "fdwic-submit-pmu-v1",
        "capture": {
            "mode": "submit-pmu-none",
            "window_scope": "per_core_first_submit_begin_to_last_submit_end",
            "accepted": True,
            "owner_restore_passed": True,
        },
        "configuration": {
            "num_cores": 96,
            "aic_cores": 32,
            "aiv_cores": 64,
            "expected_submits_per_core": 5,
            "sys_counter_tick_ns": 1,
            "selectors": {
                "cnt2_scalar_busy": 0x001,
                "cnt5_shadow_icache_miss": 0x035,
                "cnt6_primary_icache_request": 0x034,
                "cnt7_primary_icache_miss": 0x035,
                "cnt8_shadow_icache_request": 0x034,
            },
            "status_required_mask": REQUIRED_STATUS_MASK,
            "counter_width_bits": {"total": 64, "programmable": 32},
            "programmable_counter_risk_threshold": (1 << 30) - 1,
            "pmu_cycles_per_ns": {"all": 1.649844, "aic": 1.650062, "aiv": 1.649731},
        },
        "owner": {
            "configure_passed": True,
            "restore_passed": True,
            "configured_count": 96,
            "configured_aic": 32,
            "configured_aiv": 64,
            "restored_count": 96,
            "active_after_restore": 0,
            "restore_failures": 0,
            "configured_bitmap_words": _bitmap_words(all_physical),
            "complete_mixed_triplets": 32,
        },
        "window": {
            "global_first_submit_start_tick": first_tick,
            "global_last_submit_end_tick": last_tick,
            "global_submit_span_ticks": last_tick - first_tick,
            "global_submit_span_us": (last_tick - first_tick) / 1_000,
        },
        "validation": {
            "passed": True,
            "trusted_records": 96,
            "unique_physical_core_ids": 96,
            "aic_records": 32,
            "aiv_records": 64,
            "mixed_triplets": 32,
            "owner_bitmap_member_records": 96,
            "status_match_records": 96,
            "selector_match_records": 96,
            "window_started_records": 96,
            "window_stopped_records": 96,
            "submit_count_closed_records": 96,
            "scalar_le_total_records": 96,
            "shadow_primary_match_records": 96,
            "icache_miss_le_request_records": 96,
            "counter_below_risk_threshold_records": 96,
        },
        "records": records,
        "summary": {name: _group_summary(group) for name, group in grouped.items()},
    }


def _valid_arg_build_capture() -> dict[str, Any]:
    capture = _valid_capture()
    capture["capture"]["mode"] = ARG_BUILD_CAPTURE_MODE
    capture["configuration"]["phase"] = {
        "id": 1,
        "name": "arg-build",
        "boundary": "claim_end_to_materialize_begin",
        "expected_calls_per_core": 5,
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "running_read_clear_observed_bracket",
        "time_semantics": "inner_sys_cnt_between_boundary_observers",
    }
    for logical_core_id, record in enumerate(capture["records"]):
        record["shadow_icache_requests"] = record["icache_requests"] - 20 - logical_core_id % 3
        record["shadow_icache_misses"] = record["icache_misses"] - 5
        record.update(
            {
                "phase_id": 1,
                "phase_elapsed_ticks": 1_500 + logical_core_id,
                "phase_icache_requests_observed": 400 + logical_core_id,
                "phase_icache_misses_observed": 40 + logical_core_id % 5,
                "phase_begin_reads": 5,
                "phase_end_reads": 5,
                "phase_max_shadow_request_chunk": 120 + logical_core_id % 7,
                "phase_max_shadow_miss_chunk": 15 + logical_core_id % 3,
                "phase_status": PHASE_REQUIRED_STATUS_MASK,
            }
        )
    capture["validation"].pop("shadow_primary_match_records")
    capture["validation"].update(
        {
            "phase_boundary_closed_records": 96,
            "phase_shape_match_records": 96,
            "phase_values_ordered_records": 96,
            "phase_time_within_submit_records": 96,
            "shadow_primary_bounded_records": 96,
        }
    )
    return capture


def _valid_empty_bracket_capture() -> dict[str, Any]:
    capture = _valid_arg_build_capture()
    capture["capture"]["mode"] = EMPTY_BRACKET_CAPTURE_MODE
    capture["configuration"]["phase"] = {
        "id": 2,
        "name": "empty-bracket",
        "boundary": "claim_end_adjacent_empty_bracket",
        "expected_calls_per_core": 5,
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "running_read_clear_empty_bracket_calibration",
        "time_semantics": "outer_sys_cnt_around_adjacent_begin_end_pair",
    }
    for logical_core_id, record in enumerate(capture["records"]):
        record["phase_id"] = 2
        record["phase_elapsed_ticks"] = 200 + logical_core_id
        record["phase_icache_requests_observed"] = 30 + logical_core_id % 5
        record["phase_icache_misses_observed"] = 3 + logical_core_id % 2
    return capture


def _valid_materialize_capture() -> dict[str, Any]:
    capture = _valid_arg_build_capture()
    capture["capture"]["mode"] = MATERIALIZE_CAPTURE_MODE
    capture["configuration"]["phase"] = {
        "id": 3,
        "name": "materialize",
        "boundary": "materialize_begin_to_materialize_end",
        "expected_calls_per_core": 5,
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "running_read_clear_observed_bracket",
        "time_semantics": "inner_sys_cnt_between_boundary_observers",
    }
    for logical_core_id, record in enumerate(capture["records"]):
        record["phase_id"] = 3
        record["phase_elapsed_ticks"] = 800 + logical_core_id
        record["phase_icache_requests_observed"] = 500 + logical_core_id
        record["phase_icache_misses_observed"] = 50 + logical_core_id % 5
    return capture


def _valid_claim_capture() -> dict[str, Any]:
    capture = _valid_arg_build_capture()
    capture["capture"]["mode"] = CLAIM_CAPTURE_MODE
    capture["configuration"]["phase"] = {
        "id": 4,
        "name": "claim",
        "boundary": "claim_begin_to_claim_end",
        "expected_calls_per_core": 5,
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "running_read_clear_observed_bracket",
        "time_semantics": "inner_sys_cnt_between_boundary_observers",
    }
    for logical_core_id, record in enumerate(capture["records"]):
        record["phase_id"] = 4
        record["phase_elapsed_ticks"] = 650 + logical_core_id
        record["phase_icache_requests_observed"] = 200 + logical_core_id
        record["phase_icache_misses_observed"] = 20 + logical_core_id % 5
    return capture


def _valid_register_capture() -> dict[str, Any]:
    capture = _valid_arg_build_capture()
    capture["capture"]["mode"] = REGISTER_CAPTURE_MODE
    capture["configuration"]["phase"] = {
        "id": 5,
        "name": "register",
        "boundary": "register_outputs_call_entry_to_return",
        "expected_calls_per_core": 5,
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "running_read_clear_observed_bracket",
        "time_semantics": "inner_sys_cnt_between_boundary_observers",
    }
    for logical_core_id, record in enumerate(capture["records"]):
        record["phase_id"] = 5
        record["phase_elapsed_ticks"] = 400 + logical_core_id
        record["phase_icache_requests_observed"] = 150 + logical_core_id
        record["phase_icache_misses_observed"] = 15 + logical_core_id % 5
    return capture


def _write_capture(directory: Path, capture: dict[str, Any]) -> Path:
    path = directory / DEFAULT_INPUT_NAME
    path.write_text(json.dumps(capture, indent=2), encoding="utf-8")
    return path


def test_valid_capture_is_recomputed_and_rendered(tmp_path: Path) -> None:
    raw_path = _write_capture(tmp_path, _valid_capture())

    capture = load_capture(raw_path)
    assert len(capture.records) == 96
    assert len(capture.groups["aic"]) == 32
    assert len(capture.groups["aiv"]) == 64

    document = render_report(raw_path)
    assert "真实 FDWIC Submit PMU" in document
    assert "32 AIC + 64 AIV" in document
    assert "Σmiss/Σrequest" in document
    assert "90.000 ns" in document
    assert "不是 Submit 墙钟损失" in document
    assert "非 Scalar-busy 残余不是空闲时间，也不是 I-cache stall" in document
    assert "阶段观察（phase_id=" not in document
    assert "均值 16,047.5；最小 16,000；最大 16,095" in document
    for group_name in ("all", "aic", "aiv"):
        assert f"{capture.summary[group_name]['icache_miss_rate']:.3%}" in document
    assert document.count("<svg") == 4
    assert document.count("<circle") == 4 * 96
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() in document


def test_valid_arg_build_capture_renders_same_elf_phase_observation_first(tmp_path: Path) -> None:
    raw_path = _write_capture(tmp_path, _valid_arg_build_capture())

    capture = load_capture(raw_path)

    assert capture.phase_summary is not None
    all_phase = capture.phase_summary["all"]
    assert all_phase["phase_begin_reads"] == 96 * 5
    assert all_phase["phase_end_reads"] == 96 * 5
    assert all_phase["phase_icache_requests_observed_plus_capture_gap"]["sum"] == all_phase[
        "phase_icache_requests_observed"
    ]["sum"] + sum(record["icache_requests"] - record["shadow_icache_requests"] for record in capture.records)

    document = render_report(raw_path)
    assert document.index("<code>arg-build</code> 阶段观察") < document.index("全局 Submit 时间范围")
    assert "claim_end_to_materialize_begin" in document
    assert "running_read_clear_observed_bracket" in document
    assert "inner_sys_cnt_between_boundary_observers" in document
    assert "时间占比" in document
    assert "Request 观测值 / 加全窗 capture gap" in document
    assert "Miss 观测值 / 加全窗 capture gap" in document
    assert "observed_plus_capture_gap = observed + (primary − shadow)" in document
    assert "插桩 bookkeeping 会进入 sample" in document
    assert "不是原业务" in document
    assert "事件数的数学上下界" in document
    assert "时间占比的分母是同一 ELF 的 Σ每核 Submit elapsed" in document
    assert "request/miss 百分比的分母才是同一 ELF" in document
    assert "每次调用 elapsed / request / miss observed" in document
    assert "不能跨 ELF 相减" in document
    assert "shadow 是 begin/end/final 全部 running read-clear 返回值之和" in document
    for group_name in ("all", "aic", "aiv"):
        phase = capture.phase_summary[group_name]
        assert f"{phase['phase_time_share_of_submit']:.3%}" in document
        assert f"{phase['phase_request_observed_share_of_primary']:.3%}" in document
        assert f"{phase['phase_request_observed_plus_capture_gap_share_of_primary']:.3%}" in document
        assert f"{phase['phase_miss_observed_share_of_primary']:.3%}" in document
        assert f"{phase['phase_miss_observed_plus_capture_gap_share_of_primary']:.3%}" in document
        assert f"{phase['phase_elapsed_ticks_per_call']:,.3f} ns" in document
        assert f"request {phase['phase_icache_requests_observed_per_call']:,.3f}" in document
        assert f"miss {phase['phase_icache_misses_observed_per_call']:,.3f}" in document


def test_valid_empty_bracket_capture_is_reported_as_observer_calibration(tmp_path: Path) -> None:
    raw_path = _write_capture(tmp_path, _valid_empty_bracket_capture())

    capture = load_capture(raw_path)

    assert capture.phase_summary is not None
    document = render_report(raw_path)
    assert document.index("<code>empty-bracket</code> 阶段观察") < document.index("全局 Submit 时间范围")
    assert "phase_id=2" in document
    assert "claim_end_adjacent_empty_bracket" in document
    assert "running_read_clear_empty_bracket_calibration" in document
    assert "outer_sys_cnt_around_adjacent_begin_end_pair" in document
    assert "观察器自成本量尺，不是业务 phase" in document
    assert "phase_elapsed 是紧邻 begin+end 对的外层 SYS_CNT 经验耗时" in document
    assert "含计时边界底噪" in document
    assert "仍只覆盖两次 shadow read-clear 之间" in document
    assert "并不包含 begin 读取前/end 读取后的全部 observer 取指" in document
    assert "两者都只是量级参照，不能跨 ELF 精确扣减" in document
    for group_name in ("all", "aic", "aiv"):
        phase = capture.phase_summary[group_name]
        assert f"{phase['phase_elapsed_ticks_per_call']:,.3f} ns" in document
        assert f"request {phase['phase_icache_requests_observed_per_call']:,.3f}" in document
        assert f"miss {phase['phase_icache_misses_observed_per_call']:,.3f}" in document


def test_empty_bracket_rejects_arg_build_phase_id(tmp_path: Path) -> None:
    capture = _valid_empty_bracket_capture()
    capture["records"][0]["phase_id"] = 1

    with pytest.raises(ValueError, match=r"records\[0\]\.phase_id must equal 2"):
        load_capture(_write_capture(tmp_path, capture))


def test_valid_materialize_capture_tracks_current_business_boundary(tmp_path: Path) -> None:
    raw_path = _write_capture(tmp_path, _valid_materialize_capture())

    capture = load_capture(raw_path)

    assert capture.phase_summary is not None
    assert capture.phase_summary["all"]["phase_begin_reads"] == 96 * 5
    document = render_report(raw_path)
    assert document.index("<code>materialize</code> 阶段观察") < document.index("全局 Submit 时间范围")
    assert "phase_id=3" in document
    assert "materialize_begin_to_materialize_end" in document
    assert "running_read_clear_observed_bracket" in document
    assert "inner_sys_cnt_between_boundary_observers" in document
    assert "观察器自成本量尺，不是业务 phase" not in document


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("id", 1),
        ("name", "arg-build"),
        ("boundary", "claim_end_to_materialize_begin"),
        ("expected_calls_per_core", 4),
        ("counter_semantics", "running_read_clear_empty_bracket_calibration"),
        ("time_semantics", "outer_sys_cnt_around_adjacent_begin_end_pair"),
    ),
)
def test_materialize_rejects_mismatched_phase_configuration(tmp_path: Path, field: str, value: Any) -> None:
    capture = _valid_materialize_capture()
    capture["configuration"]["phase"][field] = value

    with pytest.raises(ValueError, match=r"configuration\.phase"):
        load_capture(_write_capture(tmp_path, capture))


def test_materialize_rejects_wrong_record_phase_id(tmp_path: Path) -> None:
    capture = _valid_materialize_capture()
    capture["records"][0]["phase_id"] = 2

    with pytest.raises(ValueError, match=r"records\[0\]\.phase_id must equal 3"):
        load_capture(_write_capture(tmp_path, capture))


def test_valid_claim_capture_tracks_current_business_boundary(tmp_path: Path) -> None:
    raw_path = _write_capture(tmp_path, _valid_claim_capture())

    capture = load_capture(raw_path)

    assert capture.phase_summary is not None
    assert capture.phase_summary["all"]["phase_begin_reads"] == 96 * 5
    document = render_report(raw_path)
    assert document.index("<code>claim</code> 阶段观察") < document.index("全局 Submit 时间范围")
    assert "phase_id=4" in document
    assert "claim_begin_to_claim_end" in document
    assert "running_read_clear_observed_bracket" in document
    assert "inner_sys_cnt_between_boundary_observers" in document
    assert "观察器自成本量尺，不是业务 phase" not in document


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("id", 3),
        ("name", "materialize"),
        ("boundary", "materialize_begin_to_materialize_end"),
        ("expected_calls_per_core", 4),
        ("counter_semantics", "running_read_clear_empty_bracket_calibration"),
        ("time_semantics", "outer_sys_cnt_around_adjacent_begin_end_pair"),
    ),
)
def test_claim_rejects_mismatched_phase_configuration(tmp_path: Path, field: str, value: Any) -> None:
    capture = _valid_claim_capture()
    capture["configuration"]["phase"][field] = value

    with pytest.raises(ValueError, match=r"configuration\.phase"):
        load_capture(_write_capture(tmp_path, capture))


def test_claim_rejects_wrong_record_phase_id(tmp_path: Path) -> None:
    capture = _valid_claim_capture()
    capture["records"][0]["phase_id"] = 3

    with pytest.raises(ValueError, match=r"records\[0\]\.phase_id must equal 4"):
        load_capture(_write_capture(tmp_path, capture))


def test_valid_register_capture_tracks_call_body_boundary(tmp_path: Path) -> None:
    raw_path = _write_capture(tmp_path, _valid_register_capture())

    capture = load_capture(raw_path)

    assert capture.phase_summary is not None
    assert capture.phase_summary["all"]["phase_begin_reads"] == 96 * 5
    document = render_report(raw_path)
    assert document.index("<code>register</code> 阶段观察") < document.index("全局 Submit 时间范围")
    assert "phase_id=5" in document
    assert "register_outputs_call_entry_to_return" in document
    assert "running_read_clear_observed_bracket" in document
    assert "inner_sys_cnt_between_boundary_observers" in document
    assert "观察器自成本量尺，不是业务 phase" not in document


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("id", 4),
        ("name", "claim"),
        ("boundary", "claim_begin_to_claim_end"),
        ("expected_calls_per_core", 4),
        ("counter_semantics", "running_read_clear_empty_bracket_calibration"),
        ("time_semantics", "outer_sys_cnt_around_adjacent_begin_end_pair"),
    ),
)
def test_register_rejects_mismatched_phase_configuration(tmp_path: Path, field: str, value: Any) -> None:
    capture = _valid_register_capture()
    capture["configuration"]["phase"][field] = value

    with pytest.raises(ValueError, match=r"configuration\.phase"):
        load_capture(_write_capture(tmp_path, capture))


def test_register_rejects_wrong_record_phase_id(tmp_path: Path) -> None:
    capture = _valid_register_capture()
    capture["records"][0]["phase_id"] = 4

    with pytest.raises(ValueError, match=r"records\[0\]\.phase_id must equal 5"):
        load_capture(_write_capture(tmp_path, capture))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("name", "arg-build"),
        ("boundary", "claim_end_to_materialize_begin"),
        ("expected_calls_per_core", 4),
        ("counter_semantics", "running_read_clear_observed_bracket"),
        ("time_semantics", "inner_sys_cnt_between_boundary_observers"),
    ),
)
def test_empty_bracket_rejects_mismatched_phase_configuration(tmp_path: Path, field: str, value: Any) -> None:
    capture = _valid_empty_bracket_capture()
    capture["configuration"]["phase"][field] = value

    with pytest.raises(ValueError, match=r"configuration\.phase"):
        load_capture(_write_capture(tmp_path, capture))


def test_arg_build_rejects_unclosed_phase_boundaries(tmp_path: Path) -> None:
    capture = _valid_arg_build_capture()
    capture["records"][0]["phase_end_reads"] = 4

    with pytest.raises(ValueError, match="phase begin/end reads must both equal 5"):
        load_capture(_write_capture(tmp_path, capture))


def test_arg_build_rejects_shadow_counter_above_primary(tmp_path: Path) -> None:
    capture = _valid_arg_build_capture()
    capture["records"][0]["shadow_icache_requests"] = capture["records"][0]["icache_requests"] + 1

    with pytest.raises(ValueError, match="shadow_icache_requests exceeds icache_requests"):
        load_capture(_write_capture(tmp_path, capture))


def test_arg_build_rejects_observed_counter_above_shadow(tmp_path: Path) -> None:
    capture = _valid_arg_build_capture()
    capture["records"][0]["phase_icache_requests_observed"] = capture["records"][0]["shadow_icache_requests"] + 1

    with pytest.raises(ValueError, match="phase_icache_requests_observed exceeds shadow_icache_requests"):
        load_capture(_write_capture(tmp_path, capture))


def test_arg_build_rejects_phase_time_beyond_submit(tmp_path: Path) -> None:
    capture = _valid_arg_build_capture()
    capture["records"][0]["phase_elapsed_ticks"] = capture["records"][0]["submit_elapsed_ticks"] + 1

    with pytest.raises(ValueError, match="phase_elapsed_ticks exceeds submit_elapsed_ticks"):
        load_capture(_write_capture(tmp_path, capture))


def test_arg_build_rejects_incomplete_phase_status(tmp_path: Path) -> None:
    capture = _valid_arg_build_capture()
    capture["records"][0]["phase_status"] &= ~(1 << 5)

    with pytest.raises(ValueError, match="phase_status must equal 0x3f"):
        load_capture(_write_capture(tmp_path, capture))


def test_arg_build_rejects_mismatched_phase_configuration(tmp_path: Path) -> None:
    capture = _valid_arg_build_capture()
    capture["configuration"]["phase"]["boundary"] = "claim_begin_to_claim_end"

    with pytest.raises(ValueError, match=r"configuration\.phase"):
        load_capture(_write_capture(tmp_path, capture))


def test_arg_build_rejects_deprecated_lower_bound_raw_field(tmp_path: Path) -> None:
    capture = _valid_arg_build_capture()
    record = capture["records"][0]
    record["phase_icache_requests_lower_bound"] = record.pop("phase_icache_requests_observed")

    with pytest.raises(ValueError, match="must use observed phase fields"):
        load_capture(_write_capture(tmp_path, capture))


def test_arg_build_rejects_deprecated_phase_time_validation_name(tmp_path: Path) -> None:
    capture = _valid_arg_build_capture()
    validation = capture["validation"]
    validation["phase_time_bounded_records"] = validation.pop("phase_time_within_submit_records")

    with pytest.raises(ValueError, match="must use phase_time_within_submit_records"):
        load_capture(_write_capture(tmp_path, capture))


@pytest.mark.parametrize(
    "field",
    (
        "phase_boundary_closed_records",
        "phase_shape_match_records",
        "phase_values_ordered_records",
        "phase_time_within_submit_records",
        "shadow_primary_bounded_records",
    ),
)
def test_arg_build_requires_all_producer_phase_validations(tmp_path: Path, field: str) -> None:
    capture = _valid_arg_build_capture()
    capture["validation"][field] = 95

    with pytest.raises(ValueError, match=rf"validation\.{field}"):
        load_capture(_write_capture(tmp_path, capture))


def test_none_rejects_phase_record_fields(tmp_path: Path) -> None:
    capture = _valid_capture()
    capture["records"][0]["phase_id"] = 1

    with pytest.raises(ValueError, match="must not contain phase fields"):
        load_capture(_write_capture(tmp_path, capture))


def test_none_rejects_phase_configuration(tmp_path: Path) -> None:
    capture = _valid_capture()
    capture["configuration"]["phase"] = {}

    with pytest.raises(ValueError, match="must not contain phase"):
        load_capture(_write_capture(tmp_path, capture))


def test_write_report_uses_fixed_default_name_and_atomic_publish(tmp_path: Path) -> None:
    raw_path = _write_capture(tmp_path, _valid_capture())

    output = write_report(raw_path)

    assert output == tmp_path / DEFAULT_OUTPUT_NAME
    assert output.read_text(encoding="utf-8").endswith("</html>\n")
    assert not (tmp_path / f"{DEFAULT_OUTPUT_NAME}.tmp").exists()


def test_write_report_publishes_tmp_with_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_path = _write_capture(tmp_path, _valid_capture())
    real_replace = report_module.os.replace
    calls: list[tuple[Path, Path]] = []

    def record_replace(source: str | Path, destination: str | Path) -> None:
        calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(report_module.os, "replace", record_replace)

    output = write_report(raw_path)

    assert calls == [(tmp_path / f"{DEFAULT_OUTPUT_NAME}.tmp", output)]


def test_float_summary_accepts_the_cpp_emitters_twelve_significant_digits(tmp_path: Path) -> None:
    capture = _valid_capture()
    for group in capture["summary"].values():
        for metric in ("total_cycles", "scalar_busy", "icache_requests", "icache_misses"):
            group[metric]["mean"] = float(f"{group[metric]['mean']:.12g}")
        group["scalar_busy_share"] = float(f"{group['scalar_busy_share']:.12g}")
        group["icache_miss_rate"] = float(f"{group['icache_miss_rate']:.12g}")

    loaded = load_capture(_write_capture(tmp_path, capture))

    assert loaded.summary["all"]["cores"] == 96


def test_counter_immediately_below_risk_threshold_is_accepted(tmp_path: Path) -> None:
    capture = _valid_capture()
    accepted_value = (1 << 30) - 2
    capture["records"][0]["icache_requests"] = accepted_value
    capture["records"][0]["shadow_icache_requests"] = accepted_value
    records = capture["records"]
    capture["summary"] = {
        "all": _group_summary(records),
        "aic": _group_summary(records[:32]),
        "aiv": _group_summary(records[32:]),
    }

    loaded = load_capture(_write_capture(tmp_path, capture))

    assert loaded.records[0]["icache_requests"] == accepted_value


Mutation = Callable[[dict[str, Any]], None]


def _duplicate_physical(capture: dict[str, Any]) -> None:
    capture["records"][1]["physical_core_id"] = capture["records"][0]["physical_core_id"]


def _wrong_role_count(capture: dict[str, Any]) -> None:
    capture["records"][31]["role"] = "aiv"


def _broken_triplet(capture: dict[str, Any]) -> None:
    records = capture["records"]
    records[32]["physical_core_id"] = 50
    physical_ids = [record["physical_core_id"] for record in records]
    capture["owner"]["configured_bitmap_words"] = _bitmap_words(physical_ids)


def _owner_not_restored(capture: dict[str, Any]) -> None:
    capture["owner"]["restore_passed"] = False


def _owner_bitmap_mismatch(capture: dict[str, Any]) -> None:
    capture["owner"]["configured_bitmap_words"][0] &= ~1


def _selector_changed(capture: dict[str, Any]) -> None:
    capture["configuration"]["selectors"]["cnt7_primary_icache_miss"] = 0x36


def _frequency_changed(capture: dict[str, Any]) -> None:
    capture["configuration"]["pmu_cycles_per_ns"]["aiv"] = 1.0


def _schema_changed(capture: dict[str, Any]) -> None:
    capture["schema"] = "fdwic-submit-pmu-v2"


def _status_missing(capture: dict[str, Any]) -> None:
    capture["records"][0]["status"] &= ~(1 << 9)


def _submit_count_mismatch(capture: dict[str, Any]) -> None:
    capture["records"][0]["submit_count"] = 4


def _scalar_exceeds_total(capture: dict[str, Any]) -> None:
    capture["records"][0]["scalar_busy"] = capture["records"][0]["total_cycles"] + 1


def _miss_exceeds_request(capture: dict[str, Any]) -> None:
    capture["records"][0]["icache_misses"] = capture["records"][0]["icache_requests"] + 1
    capture["records"][0]["shadow_icache_misses"] = capture["records"][0]["icache_misses"]


def _request_is_zero(capture: dict[str, Any]) -> None:
    capture["records"][0]["icache_requests"] = 0
    capture["records"][0]["icache_misses"] = 0
    capture["records"][0]["shadow_icache_requests"] = 0
    capture["records"][0]["shadow_icache_misses"] = 0


def _shadow_mismatch(capture: dict[str, Any]) -> None:
    capture["records"][0]["shadow_icache_requests"] += 1


def _counter_reaches_threshold(capture: dict[str, Any]) -> None:
    capture["records"][0]["icache_requests"] = (1 << 30) - 1
    capture["records"][0]["shadow_icache_requests"] = (1 << 30) - 1


def _summary_tampered(capture: dict[str, Any]) -> None:
    capture["summary"]["aiv"]["icache_misses"]["sum"] += 1


def _producer_validation_failed(capture: dict[str, Any]) -> None:
    capture["validation"]["passed"] = False


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (_duplicate_physical, "physical_core_id values must be unique"),
        (_wrong_role_count, "exactly 32 AIC and 64 AIV"),
        (_broken_triplet, "complete 1:2 mixed triplets"),
        (_owner_not_restored, "owner.restore_passed must be true"),
        (_owner_bitmap_mismatch, "configured_bitmap_words must exactly match"),
        (_selector_changed, "configuration.selectors"),
        (_frequency_changed, "configuration.pmu_cycles_per_ns"),
        (_schema_changed, "schema must equal"),
        (_status_missing, "status must equal"),
        (_submit_count_mismatch, "submit_count does not close"),
        (_scalar_exceeds_total, "scalar_busy exceeds total_cycles"),
        (_miss_exceeds_request, "icache_misses exceeds icache_requests"),
        (_request_is_zero, "icache_requests must be an integer >= 1"),
        (_shadow_mismatch, "shadow I-cache counters differ"),
        (_counter_reaches_threshold, "programmable counter reaches the risk threshold"),
        (_summary_tampered, "summary.aiv.icache_misses.sum"),
        (_producer_validation_failed, "validation.passed"),
    ),
)
def test_strict_capture_gates(tmp_path: Path, mutation: Mutation, message: str) -> None:
    capture = _valid_capture()
    mutation(capture)
    raw_path = _write_capture(tmp_path, capture)

    with pytest.raises(ValueError, match=message):
        load_capture(raw_path)


def test_invalid_raw_never_publishes_html(tmp_path: Path) -> None:
    capture = _valid_capture()
    _owner_not_restored(capture)
    raw_path = _write_capture(tmp_path, capture)

    with pytest.raises(ValueError, match="owner.restore_passed must be true"):
        write_report(raw_path)

    assert not (tmp_path / DEFAULT_OUTPUT_NAME).exists()
    assert not (tmp_path / f"{DEFAULT_OUTPUT_NAME}.tmp").exists()


def test_report_requires_the_fixed_raw_filename(tmp_path: Path) -> None:
    path = tmp_path / "run1.json"
    path.write_text(json.dumps(_valid_capture()), encoding="utf-8")

    with pytest.raises(ValueError, match=DEFAULT_INPUT_NAME):
        load_capture(path)
