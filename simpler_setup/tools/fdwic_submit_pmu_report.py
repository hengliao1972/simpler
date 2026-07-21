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
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_NAME = "fdwic-submit-pmu-v1"
DEFAULT_INPUT_NAME = "fdwic_submit_pmu_raw.json"
DEFAULT_OUTPUT_NAME = "fdwic_submit_pmu_report.html"
PROVENANCE_SCHEMA_NAME = "fdwic-submit-pmu-provenance-v1"
DEFAULT_PROVENANCE_NAME = "fdwic_submit_pmu_provenance.json"

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
SUBMIT_TRANSITION_CAPTURE_MODE = "submit-pmu-submit-transition"
EFDRAIN_CONTROL_CAPTURE_MODE = "submit-pmu-efdrain-control"
ARG_BUILD_PHASE_ID = 1
EMPTY_BRACKET_PHASE_ID = 2
MATERIALIZE_PHASE_ID = 3
CLAIM_PHASE_ID = 4
REGISTER_PHASE_ID = 5
SUBMIT_TRANSITION_PHASE_ID = 6
EFDRAIN_CONTROL_PHASE_ID = 7
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
    SUBMIT_TRANSITION_CAPTURE_MODE: {
        "id": SUBMIT_TRANSITION_PHASE_ID,
        "name": "submit-transition",
        "boundary": "previous_submit_end_to_next_submit_begin",
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "running_read_clear_observed_bracket",
        "time_semantics": "inner_sys_cnt_between_boundary_observers",
    },
    EFDRAIN_CONTROL_CAPTURE_MODE: {
        "id": EFDRAIN_CONTROL_PHASE_ID,
        "name": "efdrain-control",
        "boundary": "efdrain_begin_to_end_excluding_linked_kernel_calls",
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "discontinuous_running_read_clear_excluding_linked_kernel_calls",
        "time_semantics": "discontinuous_sys_cnt_control_segments_excluding_linked_kernel_calls",
    },
}


def _expected_compile_definitions(profile: str) -> tuple[str, ...]:
    definitions = ["PTO_FDWIC_SUBMIT_PMU=1"]
    if profile != NONE_CAPTURE_MODE:
        definitions.append(f"PTO_FDWIC_SUBMIT_PMU_PHASE_ID={PHASE_CONFIG_BY_MODE[profile]['id']}")
    definitions.append("PTO_FDWIC_TRACE_ENABLED=0")
    return tuple(definitions)


SUPPORTED_CAPTURE_MODES = {NONE_CAPTURE_MODE, *PHASE_CONFIG_BY_MODE}
PHASE_RECORD_FIELDS = (
    "phase_id",
    "phase_elapsed_ticks",
    "phase_icache_requests_observed",
    "phase_icache_misses_observed",
    "phase_begin_reads",
    "phase_end_reads",
    "phase_excluded_kernel_calls",
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
    raw_size: int
    raw_sha256: str
    data: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    groups: dict[str, tuple[dict[str, Any], ...]]
    summary: dict[str, dict[str, Any]]
    phase_summary: dict[str, dict[str, Any]] | None


@dataclass(frozen=True)
class BuildArtifactIdentity:
    """Frozen identity of one executable artifact used by the diagnostic run."""

    path: Path
    sha256: str
    size_bytes: int
    text_sha256: str
    text_size_bytes: int


@dataclass(frozen=True)
class SubmitPmuBuildIdentity:
    """Build-time identity captured before the A5 case starts."""

    profile: str
    profiled_cache_key: tuple[str, ...]
    aicore_extra_cache_key: str
    compile_definitions: tuple[str, ...]
    source_state: str
    source_state_path: Path
    artifacts: tuple[tuple[str, BuildArtifactIdentity], ...]


def _fail(message: str) -> None:
    raise ValueError(message)


_TEXT_SECTION_PATTERN = re.compile(
    r"^\s*\[\s*\d+\]\s+\.text\s+\S+\s+[0-9a-fA-F]+\s+"
    r"([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s",
    re.MULTILINE,
)
_HEX_16_PATTERN = re.compile(r"^[0-9a-f]{16}$")
_HEX_40_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_HEX_64_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _inspect_build_artifact(path: Path | str) -> BuildArtifactIdentity:
    artifact = Path(path).resolve()
    if not artifact.is_file():
        _fail(f"build artifact does not exist: {artifact}")
    data = artifact.read_bytes()
    try:
        result = subprocess.run(
            ["readelf", "-SW", str(artifact)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ValueError("readelf is required to inspect Submit-PMU build provenance") from exc
    if result.returncode != 0:
        _fail(f"readelf failed for build artifact {artifact}: {result.stderr.strip()}")
    match = _TEXT_SECTION_PATTERN.search(result.stdout)
    if match is None:
        _fail(f"build artifact has no literal .text section: {artifact}")
    text_offset, text_size = (int(value, 16) for value in match.groups())
    text_end = text_offset + text_size
    if text_size <= 0 or text_end > len(data):
        _fail(f"build artifact has an invalid .text range: {artifact}")
    text_data = data[text_offset:text_end]
    return BuildArtifactIdentity(
        path=artifact,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        text_sha256=hashlib.sha256(text_data).hexdigest(),
        text_size_bytes=text_size,
    )


def _parse_source_state(source_state: str) -> tuple[str, str, str, str]:
    fields = source_state.split(":")
    if (
        len(fields) != 4
        or fields[0] != "source-v2"
        or _HEX_40_PATTERN.fullmatch(fields[1]) is None
        or _HEX_64_PATTERN.fullmatch(fields[2]) is None
        or _HEX_64_PATTERN.fullmatch(fields[3]) is None
    ):
        _fail("Submit-PMU source state must be source-v2:<git-head>:<source-fingerprint>:<definitions-sha256>")
    return fields[0], fields[1], fields[2], fields[3]


def capture_build_identity(
    *,
    profile: str,
    profiled_cache_key: Sequence[Any],
    aicore_extra_cache_key: str,
    compile_definitions: Sequence[str],
    aicore_kernel: Path | str,
    aicore_build_dir: Path | str,
    host_runtime: Path | str,
) -> SubmitPmuBuildIdentity:
    """Freeze the exact diagnostic ELF/SO identity before running a case."""

    if profile not in {NONE_CAPTURE_MODE, *PHASE_CONFIG_BY_MODE}:
        _fail(f"unsupported Submit-PMU provenance profile {profile!r}")
    if _HEX_16_PATTERN.fullmatch(aicore_extra_cache_key) is None:
        _fail("AICore extra cache key must contain exactly 16 lowercase hex digits")
    definitions = tuple(compile_definitions)
    if definitions != _expected_compile_definitions(profile):
        _fail("Submit-PMU provenance compile definitions do not match the selected profile")
    kernel = Path(aicore_kernel).resolve()
    build_dir = Path(aicore_build_dir).resolve()
    stamp = build_dir / ".git_commit"
    if not stamp.is_file():
        _fail(f"Submit-PMU AICore source-state stamp is missing: {stamp}")
    source_state = stamp.read_text(encoding="utf-8").strip()
    _version, _git_head, _source_fingerprint, definitions_sha256 = _parse_source_state(source_state)
    expected_definitions_sha256 = hashlib.sha256(repr(list(definitions)).encode("utf-8")).hexdigest()
    if definitions_sha256 != expected_definitions_sha256:
        _fail("Submit-PMU source-state definition hash does not match the selected compile definitions")
    if kernel.parent.name != aicore_extra_cache_key or build_dir.parent.name != aicore_extra_cache_key:
        _fail("Submit-PMU AICore output/build paths do not match the selected extra cache key")

    artifact_paths = (
        ("aicore_kernel", kernel),
        ("aic_combined", build_dir / "aicore_aic_combined.o"),
        ("aiv_combined", build_dir / "aicore_aiv_combined.o"),
        ("host_runtime", Path(host_runtime).resolve()),
    )
    artifacts = tuple((name, _inspect_build_artifact(path)) for name, path in artifact_paths)
    cache_key = tuple(str(value) for value in profiled_cache_key)
    if not cache_key or cache_key[-1] != profile:
        _fail("profiled cache key must end with the selected Submit-PMU profile")
    return SubmitPmuBuildIdentity(
        profile=profile,
        profiled_cache_key=cache_key,
        aicore_extra_cache_key=aicore_extra_cache_key,
        compile_definitions=definitions,
        source_state=source_state,
        source_state_path=stamp,
        artifacts=artifacts,
    )


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


def _expected_phase_calls(mode: str, expected_submits: int) -> int:
    if mode == SUBMIT_TRANSITION_CAPTURE_MODE:
        if expected_submits <= 1:
            _fail("submit-pmu-submit-transition requires at least two submits per core")
        return expected_submits - 1
    return expected_submits


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
        expected_calls = _expected_phase_calls(mode, expected_submits)
        expected_phase = {
            **PHASE_CONFIG_BY_MODE[mode],
            "expected_calls_per_core": expected_calls,
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
    mode: str,
    expected_calls: int,
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
    if mode == EFDRAIN_CONTROL_CAPTURE_MODE:
        excluded_kernel_calls = _integer(
            record.get("phase_excluded_kernel_calls"),
            f"{prefix}.phase_excluded_kernel_calls",
        )
        expected_reads = expected_calls + excluded_kernel_calls
    else:
        if "phase_excluded_kernel_calls" in record:
            _fail(
                f"{prefix}.phase_excluded_kernel_calls is only valid in "
                f"{EFDRAIN_CONTROL_CAPTURE_MODE}"
            )
        expected_reads = expected_calls
    max_request_chunk = _integer(
        record.get("phase_max_shadow_request_chunk"),
        f"{prefix}.phase_max_shadow_request_chunk",
    )
    max_miss_chunk = _integer(
        record.get("phase_max_shadow_miss_chunk"),
        f"{prefix}.phase_max_shadow_miss_chunk",
    )

    if begin_reads != expected_reads or end_reads != expected_reads:
        if mode == EFDRAIN_CONTROL_CAPTURE_MODE:
            _fail(
                f"{prefix} phase begin/end reads must both equal expected calls "
                f"{expected_calls} + excluded Kernel calls {excluded_kernel_calls} = {expected_reads}"
            )
        _fail(f"{prefix} phase begin/end reads must both equal {expected_reads}")
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
    elapsed = _integer(record.get("submit_elapsed_ticks"), f"{prefix}.submit_elapsed_ticks", minimum=1)
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
        expected_calls = _expected_phase_calls(mode, expected_submits)
        _validate_phase_record(record, prefix, mode, expected_calls, elapsed, PHASE_CONFIG_BY_MODE[mode]["id"])
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


def _value_summary(values: Sequence[int | float]) -> dict[str, int | float]:
    """Summarize already-derived per-core values without changing their aggregation order."""

    total = sum(values)
    return {"sum": total, "min": min(values), "mean": total / len(values), "max": max(values)}


def _metric_summary(records: Sequence[dict[str, Any]], metric: str) -> dict[str, int | float]:
    return _value_summary([record[metric] for record in records])


def _group_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = {metric: _metric_summary(records, metric) for metric in METRICS}
    total_cycles = metrics["total_cycles"]["sum"]
    scalar_busy = metrics["scalar_busy"]["sum"]
    requests = metrics["icache_requests"]["sum"]
    misses = metrics["icache_misses"]["sum"]
    submit_elapsed_ticks = _metric_summary(records, "submit_elapsed_ticks")
    non_scalar_busy_cycles = _value_summary(
        [record["total_cycles"] - record["scalar_busy"] for record in records]
    )
    pmu_total_cycles_per_submit_tick = _value_summary(
        [record["total_cycles"] / record["submit_elapsed_ticks"] for record in records]
    )
    return {
        "cores": len(records),
        **metrics,
        "submit_elapsed_ticks": submit_elapsed_ticks,
        "non_scalar_busy_cycles": non_scalar_busy_cycles,
        "pmu_total_cycles_per_submit_tick": pmu_total_cycles_per_submit_tick,
        "scalar_busy_share": scalar_busy / total_cycles,
        "icache_miss_rate": misses / requests if requests else 0.0,
    }


def _phase_group_summary(records: Sequence[dict[str, Any]], mode: str) -> dict[str, Any]:
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
    phase_calls = sum(_expected_phase_calls(mode, record["expected_submit_count"]) for record in records)
    summary = {
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
        "phase_end_reads": sum(record["phase_end_reads"] for record in records),
        "phase_max_shadow_request_chunk": max(record["phase_max_shadow_request_chunk"] for record in records),
        "phase_max_shadow_miss_chunk": max(record["phase_max_shadow_miss_chunk"] for record in records),
    }
    if mode == EFDRAIN_CONTROL_CAPTURE_MODE:
        summary["phase_excluded_kernel_calls"] = sum(record["phase_excluded_kernel_calls"] for record in records)
    return summary


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
    kernel_exclusion_validation = "phase_kernel_exclusion_closed_records"
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
    if mode == EFDRAIN_CONTROL_CAPTURE_MODE:
        required[kernel_exclusion_validation] = EXPECTED_CORES
    elif kernel_exclusion_validation in validation:
        _fail(
            f"validation.{kernel_exclusion_validation} is only valid in "
            f"{EFDRAIN_CONTROL_CAPTURE_MODE}"
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
        None
        if mode == NONE_CAPTURE_MODE
        else {name: _phase_group_summary(group, mode) for name, group in groups.items()}
    )
    _validate_host_summary(data, summary)
    return SubmitPmuCapture(
        input_path=path,
        raw_size=len(raw_bytes),
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        data=data,
        records=records,
        groups=groups,
        summary=summary,
        phase_summary=phase_summary,
    )


def _assert_capture_raw_unchanged(capture: SubmitPmuCapture) -> bytes:
    """Re-read one raw and prove it still equals the snapshot accepted by the loader."""

    raw_bytes = capture.input_path.read_bytes()
    if len(raw_bytes) != capture.raw_size or hashlib.sha256(raw_bytes).hexdigest() != capture.raw_sha256:
        _fail("Submit-PMU raw changed after the validated capture snapshot")
    return raw_bytes


def _artifact_payload(identity: BuildArtifactIdentity) -> dict[str, Any]:
    return {
        "path": str(identity.path),
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
        "text": {"sha256": identity.text_sha256, "size_bytes": identity.text_size_bytes},
    }


def _build_provenance_payload(
    capture: SubmitPmuCapture,
    identity: SubmitPmuBuildIdentity,
) -> dict[str, Any]:
    if capture.data["capture"]["mode"] != identity.profile:
        _fail("Submit-PMU build identity profile does not match the raw capture mode")
    source_version, git_head, source_fingerprint, definitions_sha256 = _parse_source_state(identity.source_state)
    stamp = identity.source_state_path
    if not stamp.is_file() or stamp.read_text(encoding="utf-8").strip() != identity.source_state:
        _fail("Submit-PMU AICore source-state stamp changed after the build identity was frozen")

    artifacts: dict[str, Any] = {}
    for name, frozen in identity.artifacts:
        current = _inspect_build_artifact(frozen.path)
        if current != frozen:
            _fail(f"Submit-PMU build artifact changed after identity freeze: {name}")
        artifacts[name] = _artifact_payload(frozen)
    return {
        "schema": PROVENANCE_SCHEMA_NAME,
        "binding": {
            "raw_name": capture.input_path.name,
            "raw_size": capture.raw_size,
            "raw_sha256": capture.raw_sha256,
            "capture_mode": identity.profile,
        },
        "build": {
            "profile": identity.profile,
            "profiled_cache_key": list(identity.profiled_cache_key),
            "aicore_extra_cache_key": identity.aicore_extra_cache_key,
            "compile_definitions": list(identity.compile_definitions),
            "source_state": identity.source_state,
            "source_state_path": str(identity.source_state_path),
            "source_state_version": source_version,
            "git_head": git_head,
            "source_fingerprint": source_fingerprint,
            "definitions_sha256": definitions_sha256,
        },
        "artifacts": artifacts,
    }


def _json_document(data: Mapping[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _exact_keys(data: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(data)
    if actual != expected:
        _fail(
            f"{path} fields do not match the fixed provenance schema: "
            f"expected {sorted(expected)}, got {sorted(actual)}"
        )


def _validate_provenance_artifact(value: Any, path: str) -> None:
    artifact = _object(value, path)
    _exact_keys(artifact, {"path", "sha256", "size_bytes", "text"}, path)
    if not isinstance(artifact["path"], str) or not artifact["path"]:
        _fail(f"{path}.path must be a non-empty string")
    if not isinstance(artifact["sha256"], str) or _HEX_64_PATTERN.fullmatch(artifact["sha256"]) is None:
        _fail(f"{path}.sha256 must contain 64 lowercase hex digits")
    _integer(artifact["size_bytes"], f"{path}.size_bytes", minimum=1)
    text = _object(artifact["text"], f"{path}.text")
    _exact_keys(text, {"sha256", "size_bytes"}, f"{path}.text")
    if not isinstance(text["sha256"], str) or _HEX_64_PATTERN.fullmatch(text["sha256"]) is None:
        _fail(f"{path}.text.sha256 must contain 64 lowercase hex digits")
    _integer(text["size_bytes"], f"{path}.text.size_bytes", minimum=1)
    if text["size_bytes"] > artifact["size_bytes"]:
        _fail(f"{path}.text.size_bytes exceeds the whole artifact size")


def _validate_provenance_data(data: dict[str, Any], capture: SubmitPmuCapture) -> dict[str, Any]:
    """Validate one decoded sidecar against a single accepted raw snapshot."""

    _exact_keys(data, {"schema", "binding", "build", "artifacts"}, "provenance")
    if data["schema"] != PROVENANCE_SCHEMA_NAME:
        _fail(f"provenance.schema must equal {PROVENANCE_SCHEMA_NAME!r}")

    binding = _object(data["binding"], "provenance.binding")
    _exact_keys(binding, {"raw_name", "raw_size", "raw_sha256", "capture_mode"}, "provenance.binding")
    expected_binding = {
        "raw_name": capture.input_path.name,
        "raw_size": capture.raw_size,
        "raw_sha256": capture.raw_sha256,
        "capture_mode": capture.data["capture"]["mode"],
    }
    if binding != expected_binding:
        _fail("provenance.binding does not match the validated Submit-PMU raw")

    build = _object(data["build"], "provenance.build")
    _exact_keys(
        build,
        {
            "profile",
            "profiled_cache_key",
            "aicore_extra_cache_key",
            "compile_definitions",
            "source_state",
            "source_state_path",
            "source_state_version",
            "git_head",
            "source_fingerprint",
            "definitions_sha256",
        },
        "provenance.build",
    )
    if not isinstance(build["source_state"], str):
        _fail("provenance.build.source_state must be a string")
    if not isinstance(build["source_state_path"], str) or not build["source_state_path"]:
        _fail("provenance.build.source_state_path must be a non-empty string")
    source_version, git_head, source_fingerprint, definitions_sha256 = _parse_source_state(build["source_state"])
    if (
        build["profile"] != binding["capture_mode"]
        or build["source_state_version"] != source_version
        or build["git_head"] != git_head
        or build["source_fingerprint"] != source_fingerprint
        or build["definitions_sha256"] != definitions_sha256
    ):
        _fail("provenance.build does not close against its profile/source state")
    cache_key = build["profiled_cache_key"]
    if not isinstance(cache_key, list) or not cache_key or not all(isinstance(value, str) for value in cache_key):
        _fail("provenance.build.profiled_cache_key must be a non-empty string array")
    if cache_key[-1] != build["profile"]:
        _fail("provenance.build.profiled_cache_key must end with the selected profile")
    if not isinstance(build["aicore_extra_cache_key"], str) or _HEX_16_PATTERN.fullmatch(
        build["aicore_extra_cache_key"]
    ) is None:
        _fail("provenance.build.aicore_extra_cache_key must contain 16 lowercase hex digits")
    definitions = build["compile_definitions"]
    if not isinstance(definitions, list) or not definitions or not all(isinstance(value, str) for value in definitions):
        _fail("provenance.build.compile_definitions must be a non-empty string array")
    if tuple(definitions) != _expected_compile_definitions(build["profile"]):
        _fail("provenance.build compile definitions do not match the selected profile")
    if hashlib.sha256(repr(definitions).encode("utf-8")).hexdigest() != definitions_sha256:
        _fail("provenance.build compile definitions do not match definitions_sha256")

    artifacts = _object(data["artifacts"], "provenance.artifacts")
    expected_artifacts = {"aicore_kernel", "aic_combined", "aiv_combined", "host_runtime"}
    _exact_keys(artifacts, expected_artifacts, "provenance.artifacts")
    for name in sorted(expected_artifacts):
        _validate_provenance_artifact(artifacts[name], f"provenance.artifacts.{name}")
    return data


def load_provenance(
    provenance_path: Path | str,
    capture: SubmitPmuCapture,
) -> tuple[dict[str, Any], str]:
    """Validate a provenance sidecar and its immutable binding to one raw capture."""

    path = Path(provenance_path)
    if path.name != DEFAULT_PROVENANCE_NAME:
        _fail(f"Submit-PMU provenance filename must be {DEFAULT_PROVENANCE_NAME!r}")
    raw_bytes = path.read_bytes()
    data = _validate_provenance_data(_object(json.loads(raw_bytes), "provenance"), capture)
    return data, hashlib.sha256(raw_bytes).hexdigest()


def _format_integer(value: int | float) -> str:
    return f"{int(value):,}"


def _format_number(value: int | float, digits: int = 3) -> str:
    return f"{float(value):,.{digits}f}"


def _metric_range(summary: Mapping[str, int | float], *, digits: int = 1) -> str:
    return (
        f"均值 {_format_number(summary['mean'], digits)}；"
        f"最小 {_format_integer(summary['min'])}；最大 {_format_integer(summary['max'])}"
    )


def _scaled_metric_range(
    summary: Mapping[str, int | float],
    scale: float,
    *,
    digits: int = 3,
) -> str:
    """Format mean/min/max after a monotonic unit conversion."""

    return (
        f"均值 {_format_number(float(summary['mean']) * scale, digits)}；"
        f"最小 {_format_number(float(summary['min']) * scale, digits)}；"
        f"最大 {_format_number(float(summary['max']) * scale, digits)}"
    )


def _group_card(name: str, summary: Mapping[str, Any], cycles_per_ns: float, miss_penalty_ns: float) -> str:
    submit_elapsed = summary["submit_elapsed_ticks"]
    total = summary["total_cycles"]
    scalar = summary["scalar_busy"]
    non_scalar_busy = summary["non_scalar_busy_cycles"]
    effective_cycles_per_tick = summary["pmu_total_cycles_per_submit_tick"]
    requests = summary["icache_requests"]
    misses = summary["icache_misses"]
    penalty_mean_us = float(misses["mean"]) * miss_penalty_ns / 1_000
    cycles_to_us = 1.0 / cycles_per_ns / 1_000
    title = {"all": "ALL", "aic": "AIC", "aiv": "AIV"}[name]
    return f"""
      <article class="group-card">
        <h3>{title} · {summary["cores"]} 核</h3>
        <dl>
          <dt>Submit SYS_CNT/core</dt>
          <dd>{_scaled_metric_range(submit_elapsed, 1 / 1_000)} µs<br>
            <small>逐核首末 Submit 窗，不是顶部的跨核全局时间范围</small>
          </dd>
          <dt>PMU total/core</dt>
          <dd>{_metric_range(total)} cycles<br>
            <small>等效时间 {_scaled_metric_range(total, cycles_to_us)} µs
              （按 {cycles_per_ns:.6f} cycles/ns 校准）</small>
          </dd>
          <dt>Scalar busy/core</dt>
          <dd>{_metric_range(scalar)} cycles<br>
            <small>等效时间 {_scaled_metric_range(scalar, cycles_to_us)} µs；
              加权占比 {float(summary["scalar_busy_share"]):.3%}</small>
          </dd>
          <dt>非 Scalar-busy 残余/core</dt>
          <dd>{_metric_range(non_scalar_busy)} cycles<br>
            <small>逐核先算 total−scalar；等效时间
              {_scaled_metric_range(non_scalar_busy, cycles_to_us)} µs</small>
          </dd>
          <dt>PMU-total / SYS-window/core</dt>
          <dd>{_scaled_metric_range(effective_cycles_per_tick, 1.0, digits=6)} cycles/ns<br>
            <small>同 ELF 长窗有效比；不是瞬时 AICore 频率，也不是利用率</small>
          </dd>
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
    expected_calls = phase["expected_calls_per_core"]
    calibration_note = ""
    if phase["id"] == EMPTY_BRACKET_PHASE_ID:
        calibration_note = """
    <p class="notice"><strong>empty-bracket 是每次 begin/end 紧邻执行的观察器自成本量尺，不是业务 phase。</strong>
      phase_elapsed 是紧邻 begin+end 对的外层 SYS_CNT 经验耗时（含计时边界底噪）；request/miss observed
      仍只覆盖两次 shadow read-clear 之间，并不包含 begin 读取前/end 读取后的全部 observer 取指。
      两者都只是量级参照，不能跨 ELF 精确扣减。</p>
        """
    efdrain_control_note = ""
    if phase["id"] == EFDRAIN_CONTROL_PHASE_ID:
        efdrain_control_note = """
    <p class="notice"><strong>本阶段的 elapsed、request 和 miss 都是排除 linked Kernel 后，
      多段 EfDrain control segments 的累计值。</strong>暂停/恢复观察器只切分计数区间；
      每次调用的分母仍是外层 EfDrain 调用次数，不是 Begin/End 读数，也不是排除的 Kernel 调用数。</p>
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
        reads = (
            f"{_format_integer(group['phase_begin_reads'])} / "
            f"{_format_integer(group['phase_end_reads'])}"
        )
        if phase["id"] == EFDRAIN_CONTROL_PHASE_ID:
            reads += (
                "<br><small>排除 linked Kernel 调用 "
                f"{_format_integer(group['phase_excluded_kernel_calls'])} 次</small>"
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
            f"<td>{reads}</td>"
            f"<td>{_format_integer(group['phase_max_shadow_request_chunk'])} / "
            f"{_format_integer(group['phase_max_shadow_miss_chunk'])}</td>"
            "</tr>"
        )
    return f"""
  <section class="phase-overview">
    <h2><code>{phase_name}</code> 阶段观察（phase_id={phase["id"]}）</h2>
    <p class="fine">边界 <code>{phase_boundary}</code> · 计数语义 <code>{counter_semantics}</code> ·
      时间语义 <code>{time_semantics}</code> ·
      <code>expected_calls_per_core={expected_calls}</code></p>
    {calibration_note}
    {efdrain_control_note}
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


def _provenance_overview(provenance: Mapping[str, Any] | None, provenance_sha256: str | None) -> str:
    if provenance is None:
        return ""
    build = provenance["build"]
    artifacts = provenance["artifacts"]
    labels = {
        "aicore_kernel": "AICore final",
        "aic_combined": "AIC combined",
        "aiv_combined": "AIV combined",
        "host_runtime": "Host runtime",
    }
    rows = []
    for name in ("aicore_kernel", "aic_combined", "aiv_combined", "host_runtime"):
        artifact = artifacts[name]
        rows.append(
            "<tr>"
            f"<th>{labels[name]}</th>"
            f"<td>{_format_integer(artifact['size_bytes'])}</td>"
            f"<td><code>{artifact['sha256']}</code></td>"
            f"<td>{_format_integer(artifact['text']['size_bytes'])}</td>"
            f"<td><code>{artifact['text']['sha256']}</code></td>"
            f"<td><code>{html.escape(artifact['path'])}</code></td>"
            "</tr>"
        )
    definitions = "<br>".join(f"<code>{html.escape(value)}</code>" for value in build["compile_definitions"])
    cache_key = " / ".join(html.escape(value) for value in build["profiled_cache_key"])
    return f"""
  <section class="provenance-overview">
    <h2>诊断构建身份</h2>
    <p><strong>该 sidecar 在 case 返回后由实际加载路径冻结，并用 raw SHA 绑定；不会改写 raw，
      也不会给 AICore 热路径增加指令。</strong></p>
    <dl class="provenance-meta">
      <dt>Provenance SHA-256</dt><dd><code>{provenance_sha256}</code></dd>
      <dt>Profile / extra cache</dt><dd><code>{html.escape(build['profile'])}</code> /
        <code>{build['aicore_extra_cache_key']}</code></dd>
      <dt>Profiled cache key</dt><dd><code>{cache_key}</code></dd>
      <dt>Source state</dt><dd><code>{build['source_state']}</code></dd>
      <dt>Source-state stamp</dt><dd><code>{html.escape(build['source_state_path'])}</code></dd>
      <dt>Compile definitions</dt><dd>{definitions}</dd>
    </dl>
    <div class="table-wrap"><table>
      <thead><tr><th>实物</th><th>文件大小</th><th>文件 SHA-256</th>
        <th>.text 大小</th><th>.text SHA-256</th><th>采集时路径</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table></div>
  </section>
    """


def _document(
    capture: SubmitPmuCapture,
    miss_penalty_ns: float,
    provenance: Mapping[str, Any] | None = None,
    provenance_sha256: str | None = None,
) -> str:
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
    provenance_overview = _provenance_overview(provenance, provenance_sha256)
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
    .provenance-overview {{ background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:4px 16px 16px; }}
    .provenance-overview h2 {{ margin-top:14px; }} .provenance-meta {{ max-width:1200px; }}
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
  {provenance_overview}
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
    非 Scalar-busy 残余不是空闲时间，也不是 I-cache stall。卡片中的逐核 PMU 等效时间只解释
    当前 ELF 的采集窗，不能与 perf-clock、swimlane 或另一个 phase ELF 相减。</p>
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
    capture = load_capture(input_path)
    provenance_path = capture.input_path.with_name(DEFAULT_PROVENANCE_NAME)
    provenance = None
    provenance_sha256 = None
    if provenance_path.is_file():
        provenance, provenance_sha256 = load_provenance(provenance_path, capture)
    document = _document(capture, penalty, provenance, provenance_sha256)
    _assert_capture_raw_unchanged(capture)
    return document


def _atomic_write_text(output_file: Path, document: str) -> None:
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


def _stage_text(output_file: Path, document: str, *, purpose: str) -> Path:
    """Durably stage text beside its final path without making it official."""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_file.name}.{purpose}.",
        suffix=".tmp",
        dir=output_file.parent,
    )
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


def _restore_file_snapshot(output_file: Path, previous: bytes | None) -> None:
    """Restore exactly the file state observed before a paired publication."""

    if previous is None:
        output_file.unlink(missing_ok=True)
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_file.name}.rollback.",
        suffix=".tmp",
        dir=output_file.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(previous)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_file)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_provenance_report_pair(
    *,
    capture: SubmitPmuCapture,
    raw_snapshot: bytes,
    provenance_file: Path,
    provenance_document: str,
    output_file: Path,
    report_document: str,
) -> None:
    """Publish the sidecar/report pair or restore the exact previous pair."""

    raw_parent = capture.input_path.resolve().parent
    if provenance_file.resolve().parent != raw_parent or output_file.resolve().parent != raw_parent:
        _fail("Submit-PMU raw, provenance and report must share one output directory")
    if provenance_file == output_file:
        _fail("Submit-PMU provenance and report paths must be distinct")

    previous = {
        provenance_file: provenance_file.read_bytes() if provenance_file.is_file() else None,
        output_file: output_file.read_bytes() if output_file.is_file() else None,
    }
    staged: list[Path] = []
    published = False
    try:
        staged_provenance = _stage_text(provenance_file, provenance_document, purpose="pending")
        staged.append(staged_provenance)
        staged_report = _stage_text(output_file, report_document, purpose="pending")
        staged.append(staged_report)
        if _assert_capture_raw_unchanged(capture) != raw_snapshot:
            _fail("Submit-PMU raw changed while preparing provenance/report artifacts")

        os.replace(staged_provenance, provenance_file)
        published = True
        os.replace(staged_report, output_file)
        if _assert_capture_raw_unchanged(capture) != raw_snapshot:
            _fail("Submit-PMU raw changed while publishing provenance/report artifacts")
    except BaseException as original_error:
        rollback_errors: list[str] = []
        if published:
            for final_path, old_bytes in previous.items():
                try:
                    _restore_file_snapshot(final_path, old_bytes)
                except BaseException as rollback_error:  # pragma: no cover - catastrophic filesystem failure
                    rollback_errors.append(f"{final_path}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError(
                "Submit-PMU paired publication failed and rollback was incomplete: " + "; ".join(rollback_errors)
            ) from original_error
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)


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
    _atomic_write_text(output_file, document)
    return output_file


def write_report_with_provenance(
    input_path: Path | str,
    identity: SubmitPmuBuildIdentity,
    output_path: Path | str | None = None,
    provenance_path: Path | str | None = None,
    *,
    miss_penalty_ns: float = 90.0,
) -> tuple[Path, Path]:
    """Publish provenance and HTML while preserving the C++ producer's raw bytes."""

    penalty = _number(miss_penalty_ns, "miss_penalty_ns", positive=True)
    capture = load_capture(input_path)
    output_file = Path(output_path) if output_path is not None else capture.input_path.with_name(DEFAULT_OUTPUT_NAME)
    provenance_file = (
        Path(provenance_path)
        if provenance_path is not None
        else capture.input_path.with_name(DEFAULT_PROVENANCE_NAME)
    )
    if output_file.name != DEFAULT_OUTPUT_NAME:
        _fail(f"Submit-PMU output filename must be {DEFAULT_OUTPUT_NAME!r}")
    if provenance_file.name != DEFAULT_PROVENANCE_NAME:
        _fail(f"Submit-PMU provenance filename must be {DEFAULT_PROVENANCE_NAME!r}")

    raw_snapshot = _assert_capture_raw_unchanged(capture)
    payload = _build_provenance_payload(capture, identity)
    provenance_document = _json_document(payload)
    validated_provenance = _validate_provenance_data(
        _object(json.loads(provenance_document), "provenance"),
        capture,
    )
    provenance_sha256 = hashlib.sha256(provenance_document.encode("utf-8")).hexdigest()
    document = _document(capture, penalty, validated_provenance, provenance_sha256)
    _publish_provenance_report_pair(
        capture=capture,
        raw_snapshot=raw_snapshot,
        provenance_file=provenance_file,
        provenance_document=provenance_document,
        output_file=output_file,
        report_document=document,
    )
    return output_file, provenance_file


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
