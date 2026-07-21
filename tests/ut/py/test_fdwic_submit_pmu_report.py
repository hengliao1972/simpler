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
    DEFAULT_INPUT_NAME,
    DEFAULT_OUTPUT_NAME,
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
    assert "均值 16,047.5；最小 16,000；最大 16,095" in document
    for group_name in ("all", "aic", "aiv"):
        assert f"{capture.summary[group_name]['icache_miss_rate']:.3%}" in document
    assert document.count("<svg") == 4
    assert document.count("<circle") == 4 * 96
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() in document


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
