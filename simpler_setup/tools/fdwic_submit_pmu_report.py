#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate a production FDWIC Submit-PMU capture and render its I-cache report."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_NAME = "fdwic-submit-pmu-v1"
DEFAULT_INPUT_NAME = "fdwic_submit_pmu_raw.json"
DEFAULT_OUTPUT_NAME = "fdwic_submit_pmu_report.html"

EXPECTED_CORES = 96
EXPECTED_AIC_CORES = 32
EXPECTED_AIV_CORES = 64
PHYSICAL_CORES = 108
PHYSICAL_CORES_PER_DIE = 54
AIC_CORES_PER_DIE = 18
REQUIRED_STATUS_MASK = (1 << 11) - 1
PHASE_REQUIRED_STATUS_MASK = (1 << 6) - 1
PROGRAMMABLE_COUNTER_RISK_THRESHOLD = 0x3FFFFFFF

NONE_CAPTURE_MODE = "submit-pmu-none"
ARG_BUILD_CAPTURE_MODE = "submit-pmu-arg-build"
EMPTY_BRACKET_CAPTURE_MODE = "submit-pmu-empty-bracket"
MATERIALIZE_CAPTURE_MODE = "submit-pmu-materialize"
CLAIM_CAPTURE_MODE = "submit-pmu-claim"
REGISTER_CAPTURE_MODE = "submit-pmu-register"
ARG_BUILD_PHASE_ID = 1
EMPTY_BRACKET_PHASE_ID = 2
MATERIALIZE_PHASE_ID = 3
CLAIM_PHASE_ID = 4
REGISTER_PHASE_ID = 5
PHASE_CONFIG_BY_MODE = {
    ARG_BUILD_CAPTURE_MODE: {
        "id": ARG_BUILD_PHASE_ID,
        "name": "arg-build",
        "boundary": "claim_end_to_materialize_begin",
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "running_read_clear_observed_bracket",
        "time_semantics": "inner_sys_cnt_between_boundary_observers",
    },
    EMPTY_BRACKET_CAPTURE_MODE: {
        "id": EMPTY_BRACKET_PHASE_ID,
        "name": "empty-bracket",
        "boundary": "claim_end_adjacent_empty_bracket",
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "running_read_clear_empty_bracket_calibration",
        "time_semantics": "outer_sys_cnt_around_adjacent_begin_end_pair",
    },
    MATERIALIZE_CAPTURE_MODE: {
        "id": MATERIALIZE_PHASE_ID,
        "name": "materialize",
        "boundary": "materialize_begin_to_materialize_end",
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "running_read_clear_observed_bracket",
        "time_semantics": "inner_sys_cnt_between_boundary_observers",
    },
    CLAIM_CAPTURE_MODE: {
        "id": CLAIM_PHASE_ID,
        "name": "claim",
        "boundary": "claim_begin_to_claim_end",
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "running_read_clear_observed_bracket",
        "time_semantics": "inner_sys_cnt_between_boundary_observers",
    },
    REGISTER_CAPTURE_MODE: {
        "id": REGISTER_PHASE_ID,
        "name": "register",
        "boundary": "register_outputs_call_entry_to_return",
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "running_read_clear_observed_bracket",
        "time_semantics": "inner_sys_cnt_between_boundary_observers",
    },
}
SUPPORTED_CAPTURE_MODES = {NONE_CAPTURE_MODE, *PHASE_CONFIG_BY_MODE}
PHASE_RECORD_FIELDS = (
    "phase_id",
    "phase_elapsed_ticks",
    "phase_icache_requests_observed",
    "phase_icache_misses_observed",
    "phase_begin_reads",
    "phase_end_reads",
    "phase_max_shadow_request_chunk",
    "phase_max_shadow_miss_chunk",
    "phase_status",
)
DEPRECATED_PHASE_BOUND_FIELDS = (
    "phase_icache_requests_lower_bound",
    "phase_icache_misses_lower_bound",
)

EXPECTED_SELECTORS = {
    "cnt2_scalar_busy": 0x001,
    "cnt5_shadow_icache_miss": 0x035,
    "cnt6_primary_icache_request": 0x034,
    "cnt7_primary_icache_miss": 0x035,
    "cnt8_shadow_icache_request": 0x034,
}
EXPECTED_PMU_CYCLES_PER_NS = {"all": 1.649844, "aic": 1.650062, "aiv": 1.649731}
METRICS = ("total_cycles", "scalar_busy", "icache_requests", "icache_misses")
GROUP_NAMES = ("all", "aic", "aiv")


@dataclass(frozen=True)
class SubmitPmuCapture:
    """A capture that passed all producer/consumer closure checks."""

    input_path: Path
    raw_sha256: str
    data: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    groups: dict[str, tuple[dict[str, Any], ...]]
    summary: dict[str, dict[str, Any]]
    phase_summary: dict[str, dict[str, Any]] | None


def _fail(message: str) -> None:
    raise ValueError(message)


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{path} must be an array")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{path} must be an integer >= {minimum}")
    return value


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive " if positive else ""
        _fail(f"{path} must be a finite {qualifier}number")
    return result


def _same_json_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_same_json_value(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    return type(actual) is type(expected) and actual == expected


def _require_equal(actual: Any, expected: Any, path: str) -> None:
    if not _same_json_value(actual, expected):
        _fail(f"{path} must equal {expected!r}, got {actual!r}")


def _validate_capture_header(data: dict[str, Any]) -> tuple[str, dict[str, Any], int, dict[str, float]]:
    _require_equal(data.get("schema"), SCHEMA_NAME, "schema")
    capture = _object(data.get("capture"), "capture")
    mode = capture.get("mode")
    if not isinstance(mode, str) or mode not in SUPPORTED_CAPTURE_MODES:
        _fail(f"capture.mode must be one of {sorted(SUPPORTED_CAPTURE_MODES)!r}, got {mode!r}")
    _require_equal(
        capture.get("window_scope"),
        "per_core_first_submit_begin_to_last_submit_end",
        "capture.window_scope",
    )
    _require_equal(capture.get("accepted"), True, "capture.accepted")
    _require_equal(capture.get("owner_restore_passed"), True, "capture.owner_restore_passed")

    configuration = _object(data.get("configuration"), "configuration")
    _require_equal(configuration.get("num_cores"), EXPECTED_CORES, "configuration.num_cores")
    _require_equal(configuration.get("aic_cores"), EXPECTED_AIC_CORES, "configuration.aic_cores")
    _require_equal(configuration.get("aiv_cores"), EXPECTED_AIV_CORES, "configuration.aiv_cores")
    expected_submits = _integer(
        configuration.get("expected_submits_per_core"),
        "configuration.expected_submits_per_core",
        minimum=1,
    )
    if mode == NONE_CAPTURE_MODE:
        if "phase" in configuration:
            _fail(f"configuration must not contain phase in {NONE_CAPTURE_MODE}")
    else:
        expected_phase = {
            **PHASE_CONFIG_BY_MODE[mode],
            "expected_calls_per_core": expected_submits,
        }
        _require_equal(
            configuration.get("phase"),
            expected_phase,
            "configuration.phase",
        )
    _require_equal(configuration.get("sys_counter_tick_ns"), 1, "configuration.sys_counter_tick_ns")
    _require_equal(configuration.get("selectors"), EXPECTED_SELECTORS, "configuration.selectors")
    _require_equal(
        configuration.get("status_required_mask"),
        REQUIRED_STATUS_MASK,
        "configuration.status_required_mask",
    )
    _require_equal(
        configuration.get("counter_width_bits"),
        {"total": 64, "programmable": 32},
        "configuration.counter_width_bits",
    )
    _require_equal(
        configuration.get("programmable_counter_risk_threshold"),
        PROGRAMMABLE_COUNTER_RISK_THRESHOLD,
        "configuration.programmable_counter_risk_threshold",
    )
    frequency_data = _object(configuration.get("pmu_cycles_per_ns"), "configuration.pmu_cycles_per_ns")
    _require_equal(frequency_data, EXPECTED_PMU_CYCLES_PER_NS, "configuration.pmu_cycles_per_ns")
    frequencies = {
        name: _number(frequency_data.get(name), f"configuration.pmu_cycles_per_ns.{name}", positive=True)
        for name in GROUP_NAMES
    }
    return mode, configuration, expected_submits, frequencies


def _validate_phase_record(
    record: dict[str, Any],
    prefix: str,
    expected_submits: int,
    submit_elapsed: int,
    expected_phase_id: int,
) -> None:
    _require_equal(record.get("phase_id"), expected_phase_id, f"{prefix}.phase_id")
    phase_elapsed = _integer(record.get("phase_elapsed_ticks"), f"{prefix}.phase_elapsed_ticks", minimum=1)
    phase_requests_observed = _integer(
        record.get("phase_icache_requests_observed"),
        f"{prefix}.phase_icache_requests_observed",
    )
    phase_misses_observed = _integer(
        record.get("phase_icache_misses_observed"),
        f"{prefix}.phase_icache_misses_observed",
    )
    begin_reads = _integer(record.get("phase_begin_reads"), f"{prefix}.phase_begin_reads")
    end_reads = _integer(record.get("phase_end_reads"), f"{prefix}.phase_end_reads")
    max_request_chunk = _integer(
        record.get("phase_max_shadow_request_chunk"),
        f"{prefix}.phase_max_shadow_request_chunk",
    )
    max_miss_chunk = _integer(
        record.get("phase_max_shadow_miss_chunk"),
        f"{prefix}.phase_max_shadow_miss_chunk",
    )

    if begin_reads != expected_submits or end_reads != expected_submits:
        _fail(f"{prefix} phase begin/end reads must both equal {expected_submits}")
    if phase_elapsed > submit_elapsed:
        _fail(f"{prefix}.phase_elapsed_ticks exceeds submit_elapsed_ticks")
    if phase_requests_observed > record["shadow_icache_requests"]:
        _fail(f"{prefix}.phase_icache_requests_observed exceeds shadow_icache_requests")
    if phase_misses_observed > record["shadow_icache_misses"]:
        _fail(f"{prefix}.phase_icache_misses_observed exceeds shadow_icache_misses")
    if max_request_chunk >= PROGRAMMABLE_COUNTER_RISK_THRESHOLD:
        _fail(f"{prefix}.phase_max_shadow_request_chunk reaches the risk threshold 0x3fffffff")
    if max_miss_chunk >= PROGRAMMABLE_COUNTER_RISK_THRESHOLD:
        _fail(f"{prefix}.phase_max_shadow_miss_chunk reaches the risk threshold 0x3fffffff")

    phase_status = _integer(record.get("phase_status"), f"{prefix}.phase_status")
    if phase_status != PHASE_REQUIRED_STATUS_MASK:
        _fail(f"{prefix}.phase_status must equal 0x{PHASE_REQUIRED_STATUS_MASK:x}, got 0x{phase_status:x}")


def _validate_record(
    record_data: Any,
    logical_core_id: int,
    expected_submits: int,
    mode: str,
) -> dict[str, Any]:
    record = _object(record_data, f"records[{logical_core_id}]")
    prefix = f"records[{logical_core_id}]"
    _require_equal(record.get("logical_core_id"), logical_core_id, f"{prefix}.logical_core_id")
    _integer(record.get("physical_core_id"), f"{prefix}.physical_core_id")
    if record.get("role") not in {"aic", "aiv"}:
        _fail(f"{prefix}.role must be 'aic' or 'aiv'")
    _integer(record.get("block_id"), f"{prefix}.block_id")
    _integer(record.get("lane"), f"{prefix}.lane")

    submit_count = _integer(record.get("submit_count"), f"{prefix}.submit_count")
    record_expected_submits = _integer(record.get("expected_submit_count"), f"{prefix}.expected_submit_count")
    if submit_count != expected_submits or record_expected_submits != expected_submits:
        _fail(f"{prefix} submit_count does not close at {expected_submits}")

    start = _integer(record.get("first_submit_start_tick"), f"{prefix}.first_submit_start_tick", minimum=1)
    end = _integer(record.get("last_submit_end_tick"), f"{prefix}.last_submit_end_tick", minimum=1)
    elapsed = _integer(record.get("submit_elapsed_ticks"), f"{prefix}.submit_elapsed_ticks")
    if end < start or elapsed != end - start:
        _fail(f"{prefix} Submit tick window is not closed")

    total = _integer(record.get("total_cycles"), f"{prefix}.total_cycles", minimum=1)
    scalar = _integer(record.get("scalar_busy"), f"{prefix}.scalar_busy")
    requests = _integer(record.get("icache_requests"), f"{prefix}.icache_requests", minimum=1)
    misses = _integer(record.get("icache_misses"), f"{prefix}.icache_misses")
    shadow_requests = _integer(record.get("shadow_icache_requests"), f"{prefix}.shadow_icache_requests")
    shadow_misses = _integer(record.get("shadow_icache_misses"), f"{prefix}.shadow_icache_misses")
    if scalar > total:
        _fail(f"{prefix}.scalar_busy exceeds total_cycles")
    if misses > requests:
        _fail(f"{prefix}.icache_misses exceeds icache_requests")
    if shadow_misses > shadow_requests:
        _fail(f"{prefix}.shadow_icache_misses exceeds shadow_icache_requests")
    if mode == NONE_CAPTURE_MODE:
        if any(field in record for field in (*PHASE_RECORD_FIELDS, *DEPRECATED_PHASE_BOUND_FIELDS)):
            _fail(f"{prefix} must not contain phase fields in {NONE_CAPTURE_MODE}")
        if (requests, misses) != (shadow_requests, shadow_misses):
            _fail(f"{prefix} shadow I-cache counters differ from primary counters")
    else:
        if any(field in record for field in DEPRECATED_PHASE_BOUND_FIELDS):
            _fail(f"{prefix} must use observed phase fields, not lower/upper-bound names")
        if shadow_requests > requests:
            _fail(f"{prefix}.shadow_icache_requests exceeds icache_requests")
        if shadow_misses > misses:
            _fail(f"{prefix}.shadow_icache_misses exceeds icache_misses")
        _validate_phase_record(record, prefix, expected_submits, elapsed, PHASE_CONFIG_BY_MODE[mode]["id"])
    programmable = (scalar, requests, misses, shadow_requests, shadow_misses)
    if any(value >= PROGRAMMABLE_COUNTER_RISK_THRESHOLD for value in programmable):
        _fail(f"{prefix} programmable counter reaches the risk threshold 0x3fffffff")

    status = _integer(record.get("status"), f"{prefix}.status")
    if status != REQUIRED_STATUS_MASK:
        _fail(f"{prefix}.status must equal 0x{REQUIRED_STATUS_MASK:x}, got 0x{status:x}")
    return record


def _validate_logical_layout(records: Sequence[dict[str, Any]]) -> None:
    aic_ordinal = 0
    aiv_ordinal = 0
    for logical_core_id, record in enumerate(records):
        if record["role"] == "aic":
            expected = ("aic", aic_ordinal, 0)
            aic_ordinal += 1
        else:
            expected = ("aiv", aiv_ordinal // 2, 1 + aiv_ordinal % 2)
            aiv_ordinal += 1
        actual = (record["role"], record["block_id"], record["lane"])
        if actual != expected:
            _fail(f"records[{logical_core_id}] logical role/block/lane must equal {expected!r}, got {actual!r}")


def _expected_aiv_physical_ids(aic_physical_id: int) -> tuple[int, int]:
    die_base = aic_physical_id // PHYSICAL_CORES_PER_DIE * PHYSICAL_CORES_PER_DIE
    local_id = aic_physical_id % PHYSICAL_CORES_PER_DIE
    return die_base + AIC_CORES_PER_DIE + 2 * local_id, die_base + AIC_CORES_PER_DIE + 2 * local_id + 1


def _validate_topology(records: Sequence[dict[str, Any]]) -> set[int]:
    physical_ids = [record["physical_core_id"] for record in records]
    if len(set(physical_ids)) != EXPECTED_CORES:
        _fail("physical_core_id values must be unique across all 96 records")
    if any(physical_id >= PHYSICAL_CORES for physical_id in physical_ids):
        _fail("physical_core_id must be in the A5 range [0, 108)")

    aic_physical_ids = {
        record["physical_core_id"]
        for record in records
        if record["physical_core_id"] % PHYSICAL_CORES_PER_DIE < AIC_CORES_PER_DIE
    }
    reported_aic_ids = {record["physical_core_id"] for record in records if record["role"] == "aic"}
    if reported_aic_ids != aic_physical_ids:
        _fail("record roles must match the A5 physical AIC/AIV roles")
    physical_id_set = set(physical_ids)
    complete_triplets = 0
    for aic_physical_id in aic_physical_ids:
        aiv1, aiv2 = _expected_aiv_physical_ids(aic_physical_id)
        if aiv1 in physical_id_set and aiv2 in physical_id_set:
            complete_triplets += 1
    if complete_triplets != EXPECTED_AIC_CORES:
        _fail("records must form exactly 32 complete 1:2 mixed triplets")
    return physical_id_set


def _bitmap_physical_ids(words_data: Any) -> set[int]:
    words = _array(words_data, "owner.configured_bitmap_words")
    if len(words) != 4:
        _fail("owner.configured_bitmap_words must contain exactly four uint32 words")
    result: set[int] = set()
    for word_index, value in enumerate(words):
        word = _integer(value, f"owner.configured_bitmap_words[{word_index}]")
        if word > 0xFFFFFFFF:
            _fail(f"owner.configured_bitmap_words[{word_index}] exceeds uint32")
        for bit in range(32):
            if word & (1 << bit):
                result.add(word_index * 32 + bit)
    if any(physical_id >= PHYSICAL_CORES for physical_id in result):
        _fail("owner.configured_bitmap_words contains an out-of-range physical core")
    return result


def _validate_owner(data: dict[str, Any], physical_ids: set[int]) -> None:
    owner = _object(data.get("owner"), "owner")
    if owner.get("restore_passed") is not True:
        _fail("owner.restore_passed must be true")
    required = {
        "configure_passed": True,
        "configured_count": EXPECTED_CORES,
        "configured_aic": EXPECTED_AIC_CORES,
        "configured_aiv": EXPECTED_AIV_CORES,
        "restored_count": EXPECTED_CORES,
        "active_after_restore": 0,
        "restore_failures": 0,
        "complete_mixed_triplets": EXPECTED_AIC_CORES,
    }
    for field, expected in required.items():
        _require_equal(owner.get(field), expected, f"owner.{field}")
    bitmap_ids = _bitmap_physical_ids(owner.get("configured_bitmap_words"))
    if bitmap_ids != physical_ids:
        _fail("owner.configured_bitmap_words must exactly match all record physical_core_id values")


def _validate_window(data: dict[str, Any], records: Sequence[dict[str, Any]]) -> None:
    window = _object(data.get("window"), "window")
    first_tick = min(record["first_submit_start_tick"] for record in records)
    last_tick = max(record["last_submit_end_tick"] for record in records)
    span_ticks = last_tick - first_tick
    _require_equal(window.get("global_first_submit_start_tick"), first_tick, "window.global_first_submit_start_tick")
    _require_equal(window.get("global_last_submit_end_tick"), last_tick, "window.global_last_submit_end_tick")
    _require_equal(window.get("global_submit_span_ticks"), span_ticks, "window.global_submit_span_ticks")
    span_us = _number(window.get("global_submit_span_us"), "window.global_submit_span_us")
    if not math.isclose(span_us, span_ticks / 1_000, rel_tol=1e-12, abs_tol=1e-9):
        _fail("window.global_submit_span_us does not match the raw Submit ticks")


def _metric_summary(records: Sequence[dict[str, Any]], metric: str) -> dict[str, int | float]:
    values = [record[metric] for record in records]
    total = sum(values)
    return {"sum": total, "min": min(values), "mean": total / len(values), "max": max(values)}


def _group_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = {metric: _metric_summary(records, metric) for metric in METRICS}
    total_cycles = metrics["total_cycles"]["sum"]
    scalar_busy = metrics["scalar_busy"]["sum"]
    requests = metrics["icache_requests"]["sum"]
    misses = metrics["icache_misses"]["sum"]
    return {
        "cores": len(records),
        **metrics,
        "scalar_busy_share": scalar_busy / total_cycles,
        "icache_miss_rate": misses / requests if requests else 0.0,
    }


def _phase_group_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    submit_elapsed = _metric_summary(records, "submit_elapsed_ticks")
    phase_elapsed = _metric_summary(records, "phase_elapsed_ticks")
    requests_observed = _metric_summary(records, "phase_icache_requests_observed")
    misses_observed = _metric_summary(records, "phase_icache_misses_observed")

    requests_observed_plus_gap_values = [
        record["phase_icache_requests_observed"] + record["icache_requests"] - record["shadow_icache_requests"]
        for record in records
    ]
    misses_observed_plus_gap_values = [
        record["phase_icache_misses_observed"] + record["icache_misses"] - record["shadow_icache_misses"]
        for record in records
    ]
    requests_observed_plus_gap = {
        "sum": sum(requests_observed_plus_gap_values),
        "min": min(requests_observed_plus_gap_values),
        "mean": sum(requests_observed_plus_gap_values) / len(requests_observed_plus_gap_values),
        "max": max(requests_observed_plus_gap_values),
    }
    misses_observed_plus_gap = {
        "sum": sum(misses_observed_plus_gap_values),
        "min": min(misses_observed_plus_gap_values),
        "mean": sum(misses_observed_plus_gap_values) / len(misses_observed_plus_gap_values),
        "max": max(misses_observed_plus_gap_values),
    }
    primary_requests = sum(record["icache_requests"] for record in records)
    primary_misses = sum(record["icache_misses"] for record in records)
    phase_calls = sum(record["phase_end_reads"] for record in records)
    return {
        "cores": len(records),
        "submit_elapsed_ticks": submit_elapsed,
        "phase_elapsed_ticks": phase_elapsed,
        "phase_icache_requests_observed": requests_observed,
        "phase_icache_requests_observed_plus_capture_gap": requests_observed_plus_gap,
        "phase_icache_misses_observed": misses_observed,
        "phase_icache_misses_observed_plus_capture_gap": misses_observed_plus_gap,
        "phase_time_share_of_submit": phase_elapsed["sum"] / submit_elapsed["sum"],
        "phase_request_observed_share_of_primary": requests_observed["sum"] / primary_requests,
        "phase_request_observed_plus_capture_gap_share_of_primary": (
            requests_observed_plus_gap["sum"] / primary_requests
        ),
        "phase_miss_observed_share_of_primary": misses_observed["sum"] / primary_misses if primary_misses else 0.0,
        "phase_miss_observed_plus_capture_gap_share_of_primary": (
            misses_observed_plus_gap["sum"] / primary_misses if primary_misses else 0.0
        ),
        "phase_elapsed_ticks_per_call": phase_elapsed["sum"] / phase_calls,
        "phase_icache_requests_observed_per_call": requests_observed["sum"] / phase_calls,
        "phase_icache_misses_observed_per_call": misses_observed["sum"] / phase_calls,
        "phase_begin_reads": sum(record["phase_begin_reads"] for record in records),
        "phase_end_reads": phase_calls,
        "phase_max_shadow_request_chunk": max(record["phase_max_shadow_request_chunk"] for record in records),
        "phase_max_shadow_miss_chunk": max(record["phase_max_shadow_miss_chunk"] for record in records),
    }


def _validate_summary_number(actual: Any, expected: int | float, path: str) -> None:
    if isinstance(expected, int):
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
            _fail(f"{path} does not match the host recomputation: expected {expected!r}, got {actual!r}")
        return
    value = _number(actual, path)
    # The production C++ emitter serializes floating summaries with 12
    # significant digits. Integer sum/min/max remain exact; this tolerance is
    # only for the mean and weighted ratios reconstructed from those integers.
    if not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-12):
        _fail(f"{path} does not match the host recomputation: expected {expected!r}, got {actual!r}")


def _validate_host_summary(data: dict[str, Any], computed: Mapping[str, dict[str, Any]]) -> None:
    supplied = _object(data.get("summary"), "summary")
    for group_name in GROUP_NAMES:
        supplied_group = _object(supplied.get(group_name), f"summary.{group_name}")
        expected_group = computed[group_name]
        _validate_summary_number(supplied_group.get("cores"), expected_group["cores"], f"summary.{group_name}.cores")
        for metric in METRICS:
            supplied_metric = _object(supplied_group.get(metric), f"summary.{group_name}.{metric}")
            for statistic in ("sum", "min", "mean", "max"):
                _validate_summary_number(
                    supplied_metric.get(statistic),
                    expected_group[metric][statistic],
                    f"summary.{group_name}.{metric}.{statistic}",
                )
        for ratio in ("scalar_busy_share", "icache_miss_rate"):
            _validate_summary_number(
                supplied_group.get(ratio),
                expected_group[ratio],
                f"summary.{group_name}.{ratio}",
            )


def _validate_producer_summary(data: dict[str, Any], mode: str) -> None:
    validation = _object(data.get("validation"), "validation")
    required = {
        "passed": True,
        "trusted_records": EXPECTED_CORES,
        "unique_physical_core_ids": EXPECTED_CORES,
        "aic_records": EXPECTED_AIC_CORES,
        "aiv_records": EXPECTED_AIV_CORES,
        "mixed_triplets": EXPECTED_AIC_CORES,
        "owner_bitmap_member_records": EXPECTED_CORES,
        "status_match_records": EXPECTED_CORES,
        "selector_match_records": EXPECTED_CORES,
        "window_started_records": EXPECTED_CORES,
        "window_stopped_records": EXPECTED_CORES,
        "submit_count_closed_records": EXPECTED_CORES,
        "scalar_le_total_records": EXPECTED_CORES,
        "icache_miss_le_request_records": EXPECTED_CORES,
        "counter_below_risk_threshold_records": EXPECTED_CORES,
    }
    if mode == NONE_CAPTURE_MODE:
        required["shadow_primary_match_records"] = EXPECTED_CORES
    else:
        if "phase_time_bounded_records" in validation:
            _fail("validation must use phase_time_within_submit_records, not phase_time_bounded_records")
        required.update(
            {
                "phase_boundary_closed_records": EXPECTED_CORES,
                "phase_shape_match_records": EXPECTED_CORES,
                "phase_values_ordered_records": EXPECTED_CORES,
                "phase_time_within_submit_records": EXPECTED_CORES,
                "shadow_primary_bounded_records": EXPECTED_CORES,
            }
        )
    for field, expected in required.items():
        _require_equal(validation.get(field), expected, f"validation.{field}")


def load_capture(input_path: Path | str) -> SubmitPmuCapture:
    """Read and strictly validate the fixed production Submit-PMU artifact."""

    path = Path(input_path)
    if path.name != DEFAULT_INPUT_NAME:
        _fail(f"Submit-PMU input filename must be {DEFAULT_INPUT_NAME!r}")
    raw_bytes = path.read_bytes()
    decoded = json.loads(raw_bytes)
    data = _object(decoded, "root")
    mode, _, expected_submits, _ = _validate_capture_header(data)

    record_data = _array(data.get("records"), "records")
    if len(record_data) != EXPECTED_CORES:
        _fail("records must contain exactly 96 cores")
    records = tuple(
        _validate_record(value, logical_id, expected_submits, mode) for logical_id, value in enumerate(record_data)
    )
    role_counts = Counter(record["role"] for record in records)
    if role_counts != Counter({"aic": EXPECTED_AIC_CORES, "aiv": EXPECTED_AIV_CORES}):
        _fail("records must contain exactly 32 AIC and 64 AIV cores")

    physical_ids = _validate_topology(records)
    _validate_logical_layout(records)
    _validate_owner(data, physical_ids)
    _validate_window(data, records)
    _validate_producer_summary(data, mode)

    groups = {
        "all": records,
        "aic": tuple(record for record in records if record["role"] == "aic"),
        "aiv": tuple(record for record in records if record["role"] == "aiv"),
    }
    summary = {name: _group_summary(group) for name, group in groups.items()}
    phase_summary = (
        None if mode == NONE_CAPTURE_MODE else {name: _phase_group_summary(group) for name, group in groups.items()}
    )
    _validate_host_summary(data, summary)
    return SubmitPmuCapture(
        path,
        hashlib.sha256(raw_bytes).hexdigest(),
        data,
        records,
        groups,
        summary,
        phase_summary,
    )


def _format_integer(value: int | float) -> str:
    return f"{int(value):,}"


def _format_number(value: int | float, digits: int = 3) -> str:
    return f"{float(value):,.{digits}f}"


def _cycles_to_us(cycles: int | float, cycles_per_ns: float) -> float:
    return float(cycles) / cycles_per_ns / 1_000


def _metric_range(summary: Mapping[str, int | float], *, digits: int = 1) -> str:
    return (
        f"均值 {_format_number(summary['mean'], digits)}；"
        f"最小 {_format_integer(summary['min'])}；最大 {_format_integer(summary['max'])}"
    )


def _group_card(name: str, summary: Mapping[str, Any], cycles_per_ns: float, miss_penalty_ns: float) -> str:
    total = summary["total_cycles"]
    scalar = summary["scalar_busy"]
    requests = summary["icache_requests"]
    misses = summary["icache_misses"]
    residual_mean = float(total["mean"]) - float(scalar["mean"])
    penalty_mean_us = float(misses["mean"]) * miss_penalty_ns / 1_000
    title = {"all": "ALL", "aic": "AIC", "aiv": "AIV"}[name]
    return f"""
      <article class="group-card">
        <h3>{title} · {summary["cores"]} 核</h3>
        <dl>
          <dt>PMU total/core</dt>
          <dd>{_metric_range(total)} cycles<br>
            <small>均值 {_format_number(_cycles_to_us(total["mean"], cycles_per_ns))} µs
              （按 {cycles_per_ns:.6f} cycles/ns 校准）</small>
          </dd>
          <dt>Scalar busy/core</dt>
          <dd>{_metric_range(scalar)} cycles<br>
            <small>加权占比 {float(summary["scalar_busy_share"]):.3%}</small>
          </dd>
          <dt>非 Scalar-busy 残余/core</dt><dd>均值 {_format_number(residual_mean, 1)} cycles</dd>
          <dt>Primary I-cache request/core</dt><dd>{_metric_range(requests)}</dd>
          <dt>Primary I-cache miss/core</dt><dd>{_metric_range(misses)}</dd>
          <dt>加权 miss rate</dt><dd>{float(summary["icache_miss_rate"]):.3%}（Σmiss/Σrequest）</dd>
          <dt>{miss_penalty_ns:g} ns 直觉量尺/core</dt><dd>均值 {_format_number(penalty_mean_us)} µs</dd>
        </dl>
      </article>
    """


def _chart_svg(records: Sequence[dict[str, Any]], metric: str, title: str) -> str:
    width, height = 1_080, 240
    left, right, top, bottom = 64, 20, 32, 42
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [float(record[metric]) for record in records]
    maximum = max(values) or 1.0
    points = []
    for record, value in zip(records, values):
        x = left + record["physical_core_id"] / (PHYSICAL_CORES - 1) * plot_width
        y = top + (1.0 - value / maximum) * plot_height
        color = "#2563eb" if record["role"] == "aic" else "#059669"
        tooltip = html.escape(
            f"physical={record['physical_core_id']} logical={record['logical_core_id']} "
            f"role={record['role'].upper()} {metric}={int(value)}"
        )
        points.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="{color}"><title>{tooltip}</title></circle>')
    return f"""
      <figure class="chart">
        <figcaption>{html.escape(title)}</figcaption>
        <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)} 按物理核分布">
          <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis" />
          <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis" />
          <text x="{left - 8}" y="{top + 4}" text-anchor="end">{_format_number(maximum, 0)}</text>
          <text x="{left - 8}" y="{height - bottom + 4}" text-anchor="end">0</text>
          <text x="{left}" y="{height - 14}">physical 0</text>
          <text x="{width - right}" y="{height - 14}" text-anchor="end">physical 107</text>
          {"".join(points)}
        </svg>
      </figure>
    """


def _per_core_rows(records: Sequence[dict[str, Any]], miss_penalty_ns: float) -> str:
    rows = []
    for record in sorted(records, key=lambda item: item["physical_core_id"]):
        request = record["icache_requests"]
        miss = record["icache_misses"]
        miss_rate = miss / request if request else 0.0
        rows.append(
            "<tr>"
            f"<td>{record['physical_core_id']}</td><td>{record['logical_core_id']}</td>"
            f"<td>{record['role'].upper()}</td><td>{record['block_id']}</td><td>{record['lane']}</td>"
            f"<td>{_format_number(record['submit_elapsed_ticks'] / 1_000)}</td>"
            f"<td>{_format_integer(record['total_cycles'])}</td><td>{_format_integer(record['scalar_busy'])}</td>"
            f"<td>{_format_integer(request)}</td><td>{_format_integer(miss)}</td><td>{miss_rate:.3%}</td>"
            f"<td>{_format_number(miss * miss_penalty_ns / 1_000)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _phase_observed_cell(
    observed: Mapping[str, int | float],
    observed_plus_capture_gap: Mapping[str, int | float],
    observed_share: float,
    observed_plus_capture_gap_share: float,
) -> str:
    return (
        f"<strong>{_format_integer(observed['sum'])} / "
        f"{_format_integer(observed_plus_capture_gap['sum'])}</strong><br>"
        f"<small>{observed_share:.3%} / {observed_plus_capture_gap_share:.3%} of primary</small>"
    )


def _phase_overview(capture: SubmitPmuCapture) -> str:
    if capture.phase_summary is None:
        return ""

    phase = capture.data["configuration"]["phase"]
    phase_name = html.escape(phase["name"])
    phase_boundary = html.escape(phase["boundary"])
    counter_semantics = html.escape(phase["counter_semantics"])
    time_semantics = html.escape(phase["time_semantics"])
    calibration_note = ""
    if phase["id"] == EMPTY_BRACKET_PHASE_ID:
        calibration_note = """
    <p class="notice"><strong>empty-bracket 是每次 begin/end 紧邻执行的观察器自成本量尺，不是业务 phase。</strong>
      phase_elapsed 是紧邻 begin+end 对的外层 SYS_CNT 经验耗时（含计时边界底噪）；request/miss observed
      仍只覆盖两次 shadow read-clear 之间，并不包含 begin 读取前/end 读取后的全部 observer 取指。
      两者都只是量级参照，不能跨 ELF 精确扣减。</p>
        """

    rows = []
    for group_name in GROUP_NAMES:
        group = capture.phase_summary[group_name]
        title = {"all": "ALL", "aic": "AIC", "aiv": "AIV"}[group_name]
        request_observed = _phase_observed_cell(
            group["phase_icache_requests_observed"],
            group["phase_icache_requests_observed_plus_capture_gap"],
            group["phase_request_observed_share_of_primary"],
            group["phase_request_observed_plus_capture_gap_share_of_primary"],
        )
        miss_observed = _phase_observed_cell(
            group["phase_icache_misses_observed"],
            group["phase_icache_misses_observed_plus_capture_gap"],
            group["phase_miss_observed_share_of_primary"],
            group["phase_miss_observed_plus_capture_gap_share_of_primary"],
        )
        rows.append(
            "<tr>"
            f"<th>{title}<small>{group['cores']} 核</small></th>"
            f"<td><strong>{_format_number(group['phase_elapsed_ticks']['sum'] / 1_000)} µs</strong><br>"
            f"<small>Submit core-time {_format_number(group['submit_elapsed_ticks']['sum'] / 1_000)} µs</small></td>"
            f"<td><strong>{group['phase_time_share_of_submit']:.3%}</strong></td>"
            f"<td><strong>{_format_number(group['phase_elapsed_ticks_per_call'])} ns</strong><br>"
            f"<small>request {_format_number(group['phase_icache_requests_observed_per_call'])} / "
            f"miss {_format_number(group['phase_icache_misses_observed_per_call'])}</small></td>"
            f"<td>{request_observed}</td><td>{miss_observed}</td>"
            f"<td>{_format_integer(group['phase_begin_reads'])} / {_format_integer(group['phase_end_reads'])}</td>"
            f"<td>{_format_integer(group['phase_max_shadow_request_chunk'])} / "
            f"{_format_integer(group['phase_max_shadow_miss_chunk'])}</td>"
            "</tr>"
        )
    return f"""
  <section class="phase-overview">
    <h2><code>{phase_name}</code> 阶段观察（phase_id={phase["id"]}）</h2>
    <p class="fine">边界 <code>{phase_boundary}</code> · 计数语义 <code>{counter_semantics}</code> ·
      时间语义 <code>{time_semantics}</code></p>
    {calibration_note}
    <p><strong>这是 running read-clear 观察区间，不是可与其他构建直接相减的独立计时。</strong>
      时间占比的分母是同一 ELF 的 Σ每核 Submit elapsed；request/miss 百分比的分母才是同一 ELF、
      同一次采集的 Submit 整窗 primary。<code>observed_plus_capture_gap = observed + (primary − shadow)</code>；
      边界读数和插桩 bookkeeping 会进入 sample，因此“观测值”和“加全窗 capture gap”都不是原业务
      事件数的数学上下界，也不能跨 ELF 相减。</p>
    <div class="table-wrap phase-table"><table>
      <thead><tr>
        <th>核组</th><th>Phase / Submit core-time</th><th>时间占比</th>
        <th>每次调用 elapsed / request / miss observed</th>
        <th>Request 观测值 / 加全窗 capture gap</th><th>Miss 观测值 / 加全窗 capture gap</th>
        <th>Begin / End reads</th><th>最大 request / miss chunk</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table></div>
  </section>
    """


def _document(capture: SubmitPmuCapture, miss_penalty_ns: float) -> str:
    configuration = capture.data["configuration"]
    frequencies = configuration["pmu_cycles_per_ns"]
    window = capture.data["window"]
    cards = "".join(
        _group_card(name, capture.summary[name], float(frequencies[name]), miss_penalty_ns) for name in GROUP_NAMES
    )
    charts = "".join(
        (
            _chart_svg(capture.records, "total_cycles", "PMU total cycles/core"),
            _chart_svg(capture.records, "scalar_busy", "Scalar busy cycles/core"),
            _chart_svg(capture.records, "icache_requests", "Primary I-cache requests/core"),
            _chart_svg(capture.records, "icache_misses", "Primary I-cache misses/core"),
        )
    )
    rows = _per_core_rows(capture.records, miss_penalty_ns)
    phase_overview = _phase_overview(capture)
    shadow_note = (
        "phase 模式的 shadow 是 begin/end/final 全部 running read-clear 返回值之和，只用于标记全窗 "
        "capture gap，不作为第二份性能数据展示。"
        if capture.phase_summary is not None
        else "shadow 只承担逐核同值闭环，不作为第二份性能数据展示。"
    )
    source_name = html.escape(capture.input_path.name)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>真实 FDWIC Submit PMU I-cache 报告</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#5b6475; --line:#d9dfeb; --panel:#f7f9fc; }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; background:#eef2f7; color:var(--ink);
      font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;
    }}
    main {{ width:min(1500px,calc(100% - 32px)); margin:24px auto 64px; }}
    h1,h2,h3 {{ line-height:1.25; }} h1 {{ margin-bottom:6px; }} h2 {{ margin-top:30px; }}
    .subtitle,.fine {{ color:var(--muted); }}
    .notice {{ border-left:4px solid #d97706; background:#fff7ed; padding:12px 16px; border-radius:6px; }}
    .headline {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin:18px 0; }}
    .headline div,.group-card,.chart,.table-wrap {{
      background:white; border:1px solid var(--line); border-radius:10px; box-shadow:0 2px 10px #1e293b0a;
    }}
    .headline div {{ padding:14px 16px; }} .headline strong {{ display:block; font-size:22px; }}
    .phase-overview {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:4px 16px 16px; }}
    .phase-overview h2 {{ margin-top:14px; }} .phase-overview p {{ max-width:1200px; }}
    .phase-table {{ box-shadow:none; }} .phase-table th:first-child {{ text-align:left; }}
    .phase-table th small {{ display:block; font-weight:400; color:var(--muted); }}
    .groups {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(310px,1fr)); gap:14px; }}
    .group-card {{ padding:16px; min-width:0; }} .group-card h3 {{ margin:0 0 12px; }}
    dl {{ display:grid; grid-template-columns:minmax(120px,.75fr) minmax(0,1.25fr); gap:8px 12px; margin:0; }}
    dt {{ color:var(--muted); }} dd {{ margin:0; overflow-wrap:anywhere; }} small {{ color:var(--muted); }}
    .legend {{ display:flex; gap:18px; }}
    .dot::before {{
      content:""; display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px;
    }}
    .aic::before {{ background:#2563eb; }} .aiv::before {{ background:#059669; }}
    .charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(520px,100%),1fr)); gap:14px; }}
    .chart {{ margin:0; padding:12px; min-width:0; overflow:hidden; }}
    .chart figcaption {{ font-weight:650; margin-bottom:5px; }}
    svg {{ width:100%; height:auto; }}
    .axis {{ stroke:#94a3b8; stroke-width:1; }}
    svg text {{ fill:#64748b; font-size:12px; }}
    .table-wrap {{ max-width:100%; overflow:auto; }}
    table {{ border-collapse:collapse; width:max-content; min-width:100%; white-space:nowrap; }}
    th,td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:right; }}
    th {{ position:sticky; top:0; background:var(--panel); }}
    th:nth-child(3),td:nth-child(3) {{ text-align:center; }} code {{ overflow-wrap:anywhere; }}
  </style>
</head>
<body><main>
  <h1>真实 FDWIC Submit PMU I-cache 报告</h1>
  <p class="subtitle">固定输入 <code>{source_name}</code> · schema <code>{SCHEMA_NAME}</code> · 32 AIC + 64 AIV</p>
  {phase_overview}
  <section class="headline">
    <div><span>全局 Submit 时间范围</span>
      <strong>{_format_number(window["global_submit_span_us"])} µs</strong>
      <small>最早一核首个 Submit 至最晚一核末个 Submit</small>
    </div>
    <div><span>每核 Submit 数</span>
      <strong>{configuration["expected_submits_per_core"]}</strong><small>96 核均已闭环</small>
    </div>
    <div><span>受信记录</span><strong>96 / 96</strong><small>owner、selector、status、拓扑均通过</small></div>
    <div><span>raw SHA-256</span><code>{capture.raw_sha256}</code></div>
  </section>
  <p class="notice"><strong>{miss_penalty_ns:.3f} ns 仅作 I-cache miss 的直觉量尺，不是 Submit 墙钟损失。</strong>
    miss 可能重叠、被流水隐藏，也可能与其他停顿共同出现；不能把 miss×{miss_penalty_ns:g} ns
    当成可直接相减的优化收益。
    非 Scalar-busy 残余不是空闲时间，也不是 I-cache stall。</p>
  <h2>ALL / AIC / AIV 汇总</h2>
  <div class="groups">{cards}</div>
  <h2>逐物理核分布</h2>
  <p class="legend"><span class="dot aic">AIC</span><span class="dot aiv">AIV</span></p>
  <div class="charts">{charts}</div>
  <h2>逐核原始主计数</h2>
  <p class="fine">报告展示 primary request/miss；{shadow_note}</p>
  <div class="table-wrap"><table>
    <thead><tr>
      <th>Physical</th><th>Logical</th><th>Role</th><th>Block</th><th>Lane</th><th>Submit µs</th>
      <th>Total cycles</th><th>Scalar busy</th><th>I$ request</th><th>I$ miss</th>
      <th>Miss rate</th><th>{miss_penalty_ns:g}ns量尺 µs</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</main></body>
</html>
"""


def render_report(input_path: Path | str, *, miss_penalty_ns: float = 90.0) -> str:
    """Return a self-contained HTML report after validating and recomputing the raw capture."""

    penalty = _number(miss_penalty_ns, "miss_penalty_ns", positive=True)
    return _document(load_capture(input_path), penalty)


def write_report(
    input_path: Path | str,
    output_path: Path | str | None = None,
    *,
    miss_penalty_ns: float = 90.0,
) -> Path:
    """Validate first, then atomically publish the fixed-name HTML artifact."""

    input_file = Path(input_path)
    output_file = Path(output_path) if output_path is not None else input_file.with_name(DEFAULT_OUTPUT_NAME)
    if output_file.name != DEFAULT_OUTPUT_NAME:
        _fail(f"Submit-PMU output filename must be {DEFAULT_OUTPUT_NAME!r}")
    document = render_report(input_file, miss_penalty_ns=miss_penalty_ns)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(f"{output_file.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_file)
    finally:
        temporary.unlink(missing_ok=True)
    return output_file


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path(DEFAULT_INPUT_NAME), help=DEFAULT_INPUT_NAME)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=f"output path; basename must be {DEFAULT_OUTPUT_NAME} (default: beside input)",
    )
    parser.add_argument(
        "--miss-penalty-ns",
        type=float,
        default=90.0,
        help="intuitive miss-cost scale only; it is not wall-clock loss (default: 90)",
    )
    args = parser.parse_args(argv)
    try:
        output = write_report(args.input, args.output, miss_penalty_ns=args.miss_penalty_ns)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
