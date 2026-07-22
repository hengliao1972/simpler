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
import struct
from pathlib import Path
from typing import Any, Callable

import pytest

import simpler_setup.tools.fdwic_submit_pmu_report as report_module
from simpler_setup.tools.fdwic_submit_pmu_report import (
    ALLOC_COMPLETE_CAPTURE_MODE,
    ARG_BUILD_CAPTURE_MODE,
    CLAIM_CAPTURE_MODE,
    COMMON_REQUIRED_STATUS_MASK,
    DEFAULT_INPUT_NAME,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_PROVENANCE_NAME,
    EFDRAIN_CONTROL_CAPTURE_MODE,
    EMPTY_BRACKET_CAPTURE_MODE,
    FANIN_CAPTURE_MODE,
    MATERIALIZE_CAPTURE_MODE,
    NONE_REQUIRED_STATUS_MASK,
    PHASE_REQUIRED_STATUS_MASK,
    PREPARE_MAP_CAPTURE_MODE,
    REGISTER_CAPTURE_MODE,
    SUBMIT_TRANSITION_CAPTURE_MODE,
    WINNER_BUILD_CAPTURE_MODE,
    capture_build_identity,
    load_capture,
    load_provenance,
    render_report,
    write_report,
    write_report_with_provenance,
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
    scalar_elapsed = [record["scalar_submit_elapsed_ticks"] for record in records]
    scalar = [record["scalar_busy"] for record in records]
    requests = [record["icache_requests"] for record in records]
    misses = [record["icache_misses"] for record in records]
    return {
        "cores": len(records),
        "total_cycles": _metric(total),
        "scalar_submit_elapsed_ticks": _metric(scalar_elapsed),
        "scalar_busy": _metric(scalar),
        "icache_requests": _metric(requests),
        "icache_misses": _metric(misses),
        "scalar_busy_share": sum(scalar) / sum(total),
        "icache_miss_rate": sum(misses) / sum(requests),
    }


def _refresh_host_aggregates(capture: dict[str, Any]) -> None:
    """Refresh only producer-owned fields after a test mutates records."""

    records = capture["records"]
    capture["summary"] = {
        "all": _group_summary(records),
        "aic": _group_summary(records[:32]),
        "aiv": _group_summary(records[32:]),
    }
    first_tick = min(record["first_submit_start_tick"] for record in records)
    last_tick = max(record["last_submit_end_tick"] for record in records)
    capture["window"].update(
        {
            "global_first_submit_start_tick": first_tick,
            "global_last_submit_end_tick": last_tick,
            "global_submit_span_ticks": last_tick - first_tick,
            "global_submit_span_us": (last_tick - first_tick) / 1_000,
        }
    )


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
                "scalar_submit_elapsed_ticks": end - start - 1_000,
                "total_cycles": 16_000 + logical_core_id,
                "scalar_busy": 12_000 + logical_core_id,
                "icache_requests": requests,
                "icache_misses": misses,
                "status": NONE_REQUIRED_STATUS_MASK,
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
        "schema": "fdwic-submit-pmu-v2",
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
                "cnt0_vector_busy": 0x501,
                "cnt1_cube_busy": 0x301,
                "cnt2_scalar_busy": 0x001,
                "cnt5_shadow_icache_miss": 0x035,
                "cnt6_primary_icache_request": 0x034,
                "cnt7_primary_icache_miss": 0x035,
                "cnt8_shadow_icache_request": 0x034,
            },
            "status_required_mask": NONE_REQUIRED_STATUS_MASK,
            "counter_width_bits": {"total": 64, "programmable": 32},
            "programmable_counter_risk_threshold": (1 << 30) - 1,
            "linked_kernel_exclusion": {
                "enabled": True,
                "boundary": "dist_aicore_call_slot_kernel_entry_to_return",
                "gate_semantics": "metrics_prof_stop_before_call_and_start_after_return",
                "time_denominator": "scalar_submit_elapsed_ticks",
                "wall_tick_semantics": "first_submit_start_to_last_submit_end_closure_only",
            },
            "return_ready_atomic_exclusion": {
                "enabled": True,
                "classification": "result_used_atomic_only",
                "time_boundary": "sys_cnt_before_atomic_to_result_dependent_sys_cnt_after_return",
                "counter_semantics": "pmu_counters_include_atomic_instruction_events",
                "time_denominator_effect": "subtract_return_ready_atomic_elapsed",
            },
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
            "linked_kernel_gate_closed_records": 96,
            "scalar_submit_elapsed_valid_records": 96,
            "vector_busy_zero_records": 96,
            "cube_busy_zero_records": 96,
            "return_ready_atomic_time_valid_records": 96,
        },
        "records": records,
        "summary": {name: _group_summary(group) for name, group in grouped.items()},
    }


def _valid_arg_build_capture() -> dict[str, Any]:
    capture = _valid_capture()
    capture["capture"]["mode"] = ARG_BUILD_CAPTURE_MODE
    capture["configuration"]["status_required_mask"] = COMMON_REQUIRED_STATUS_MASK
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
        record["status"] = COMMON_REQUIRED_STATUS_MASK
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
                "phase_excluded_kernel_calls": 0,
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
            "phase_kernel_exclusion_closed_records": 96,
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


def _valid_prepare_map_capture() -> dict[str, Any]:
    capture = _valid_arg_build_capture()
    capture["capture"]["mode"] = PREPARE_MAP_CAPTURE_MODE
    capture["configuration"]["phase"] = {
        "id": 8,
        "name": "prepare-map",
        "boundary": "dist_submit_prepare_map_call_entry_to_return",
        "expected_calls_per_core": 5,
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "running_read_clear_observed_bracket",
        "time_semantics": "inner_sys_cnt_between_boundary_observers",
    }
    for logical_core_id, record in enumerate(capture["records"]):
        record["phase_id"] = 8
        record["phase_elapsed_ticks"] = 300 + logical_core_id
        record["phase_icache_requests_observed"] = 120 + logical_core_id
        record["phase_icache_misses_observed"] = 12 + logical_core_id % 5
    return capture


def _valid_fanin_capture() -> dict[str, Any]:
    capture = _valid_arg_build_capture()
    capture["capture"]["mode"] = FANIN_CAPTURE_MODE
    capture["configuration"]["phase"] = {
        "id": 9,
        "name": "fanin",
        "boundary": "fanin_begin_to_fanin_end",
        "call_shape": "dynamic_balanced",
        "expected_calls": {"all": 4, "aic": 2, "aiv": 2},
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "running_read_clear_observed_bracket",
        "time_semantics": "inner_sys_cnt_between_boundary_observers",
    }
    capture["validation"]["phase_global_call_count_closed"] = True
    active_calls = {0: 1, 1: 1, 32: 1, 33: 1}
    for logical_core_id, record in enumerate(capture["records"]):
        calls = active_calls.get(logical_core_id, 0)
        record["phase_id"] = 9
        record["phase_elapsed_ticks"] = 75 * calls
        record["phase_icache_requests_observed"] = 40 * calls
        record["phase_icache_misses_observed"] = 2 * calls
        record["phase_begin_reads"] = calls
        record["phase_end_reads"] = calls
    return capture


def _valid_winner_build_capture() -> dict[str, Any]:
    capture = _valid_fanin_capture()
    capture["capture"]["mode"] = WINNER_BUILD_CAPTURE_MODE
    capture["configuration"]["phase"].update(
        {
            "id": 10,
            "name": "winner-build-control",
            "boundary": "winner_build_begin_to_end_excluding_linked_kernel_calls",
            "counter_semantics": "discontinuous_running_read_clear_excluding_linked_kernel_calls",
            "time_semantics": "discontinuous_sys_cnt_control_segments_excluding_linked_kernel_calls",
        }
    )
    capture["validation"]["phase_kernel_exclusion_closed_records"] = 96
    for logical_core_id, record in enumerate(capture["records"]):
        excluded_kernel_calls = int(logical_core_id in {0, 32})
        record["phase_id"] = 10
        record["phase_excluded_kernel_calls"] = excluded_kernel_calls
        record["phase_begin_reads"] += excluded_kernel_calls
        record["phase_end_reads"] += excluded_kernel_calls
    return capture


def _valid_alloc_complete_capture(
    *,
    winner_core_id: int = 0,
    excluded_kernel_calls: int = 2,
) -> dict[str, Any]:
    """Build one B1 AllocComplete call without constraining its AIC/AIV owner."""

    capture = _valid_arg_build_capture()
    capture["capture"]["mode"] = ALLOC_COMPLETE_CAPTURE_MODE
    capture["configuration"]["phase"] = {
        "id": 11,
        "name": "alloc-complete-control",
        "boundary": "alloc_complete_begin_to_end_excluding_linked_kernel_calls",
        "call_shape": "dynamic_global",
        "expected_calls": {"all": 1},
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "discontinuous_running_read_clear_excluding_linked_kernel_calls",
        "time_semantics": "discontinuous_sys_cnt_control_segments_excluding_linked_kernel_calls",
    }
    capture["validation"]["phase_global_call_count_closed"] = True
    for logical_core_id, record in enumerate(capture["records"]):
        business_calls = int(logical_core_id == winner_core_id)
        excluded_calls = excluded_kernel_calls if business_calls else 0
        record.update(
            {
                "phase_id": 11,
                "phase_elapsed_ticks": 75 * business_calls,
                "phase_icache_requests_observed": 40 * business_calls,
                "phase_icache_misses_observed": 2 * business_calls,
                "phase_begin_reads": business_calls + excluded_calls,
                "phase_end_reads": business_calls + excluded_calls,
                "phase_excluded_kernel_calls": excluded_calls,
            }
        )
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


def _valid_submit_transition_capture() -> dict[str, Any]:
    capture = _valid_arg_build_capture()
    capture["capture"]["mode"] = SUBMIT_TRANSITION_CAPTURE_MODE
    capture["configuration"]["phase"] = {
        "id": 6,
        "name": "submit-transition",
        "boundary": "previous_submit_end_to_next_submit_begin",
        "expected_calls_per_core": 4,
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "running_read_clear_observed_bracket",
        "time_semantics": "inner_sys_cnt_between_boundary_observers",
    }
    for logical_core_id, record in enumerate(capture["records"]):
        record["phase_id"] = 6
        record["phase_elapsed_ticks"] = 600 + logical_core_id
        record["phase_icache_requests_observed"] = 180 + logical_core_id
        record["phase_icache_misses_observed"] = 18 + logical_core_id % 5
        record["phase_begin_reads"] = 4
        record["phase_end_reads"] = 4
    return capture


def _valid_efdrain_control_capture() -> dict[str, Any]:
    capture = _valid_arg_build_capture()
    capture["capture"]["mode"] = EFDRAIN_CONTROL_CAPTURE_MODE
    capture["configuration"]["phase"] = {
        "id": 7,
        "name": "efdrain-control",
        "boundary": "efdrain_begin_to_end_excluding_linked_kernel_calls",
        "expected_calls_per_core": 5,
        "status_required_mask": PHASE_REQUIRED_STATUS_MASK,
        "counter_semantics": "discontinuous_running_read_clear_excluding_linked_kernel_calls",
        "time_semantics": "discontinuous_sys_cnt_control_segments_excluding_linked_kernel_calls",
    }
    capture["validation"]["phase_kernel_exclusion_closed_records"] = 96
    for logical_core_id, record in enumerate(capture["records"]):
        excluded_kernel_calls = logical_core_id % 3
        record["phase_id"] = 7
        record["phase_elapsed_ticks"] = 900 + logical_core_id
        record["phase_icache_requests_observed"] = 250 + logical_core_id
        record["phase_icache_misses_observed"] = 25 + logical_core_id % 5
        record["phase_excluded_kernel_calls"] = excluded_kernel_calls
        record["phase_begin_reads"] = 5 + excluded_kernel_calls
        record["phase_end_reads"] = 5 + excluded_kernel_calls
    return capture


def _write_capture(directory: Path, capture: dict[str, Any]) -> Path:
    path = directory / DEFAULT_INPUT_NAME
    path.write_text(json.dumps(capture, indent=2), encoding="utf-8")
    return path


def _write_minimal_elf(
    path: Path,
    *,
    code_section_name: str = ".text",
    code: bytes = b"\x90\x90\xc3",
) -> bytes:
    """Write a minimal ELF64 relocatable object accepted by the host readelf."""

    elf_header_size = 64
    section_header_size = 64
    code_offset = elf_header_size
    section_names = b"\0" + code_section_name.encode("ascii") + b"\0.shstrtab\0"
    section_names_offset = code_offset + len(code)
    section_headers_offset = (section_names_offset + len(section_names) + 7) & ~7
    section_names_name_offset = len(code_section_name) + 2

    ident = b"\x7fELF" + bytes((2, 1, 1, 0, 0)) + b"\0" * 7
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        ident,
        1,  # ET_REL
        62,  # EM_X86_64
        1,
        0,
        0,
        section_headers_offset,
        0,
        elf_header_size,
        0,
        0,
        section_header_size,
        3,
        2,
    )
    null_section = bytes(section_header_size)
    code_section = struct.pack(
        "<IIQQQQIIQQ",
        1,
        1,  # SHT_PROGBITS
        0x6,  # SHF_ALLOC | SHF_EXECINSTR
        0,
        code_offset,
        len(code),
        0,
        0,
        16,
        0,
    )
    section_names_section = struct.pack(
        "<IIQQQQIIQQ",
        section_names_name_offset,
        3,  # SHT_STRTAB
        0,
        0,
        section_names_offset,
        len(section_names),
        0,
        0,
        1,
        0,
    )

    image = bytearray(header)
    image.extend(code)
    image.extend(section_names)
    image.extend(b"\0" * (section_headers_offset - len(image)))
    image.extend(null_section)
    image.extend(code_section)
    image.extend(section_names_section)
    contents = bytes(image)
    path.write_bytes(contents)
    return contents


def _fake_build_identity(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: str = "submit-pmu-none",
) -> report_module.SubmitPmuBuildIdentity:
    """Create a source-v2 identity without depending on real ELF tooling."""

    extra_cache_key = "0123456789abcdef"
    cache_directory = directory / "build" / extra_cache_key
    aicore_build_directory = cache_directory / "aicore_build"
    aicore_build_directory.mkdir(parents=True)
    kernel = cache_directory / "aicore_kernel.o"
    artifact_paths = (
        kernel,
        aicore_build_directory / "aicore_aic_combined.o",
        aicore_build_directory / "aicore_aiv_combined.o",
        directory / "build" / "libfdwic_runtime.so",
    )
    for ordinal, artifact in enumerate(artifact_paths, start=1):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"artifact-{ordinal}-header:text-{ordinal}-payload".encode())

    compile_definitions = ["PTO_FDWIC_SUBMIT_PMU=1"]
    if profile != "submit-pmu-none":
        compile_definitions.append(f"PTO_FDWIC_SUBMIT_PMU_PHASE_ID={report_module.PHASE_CONFIG_BY_MODE[profile]['id']}")
    compile_definitions.append("PTO_FDWIC_TRACE_ENABLED=0")
    frozen_compile_definitions = tuple(compile_definitions)
    definitions_sha256 = hashlib.sha256(repr(compile_definitions).encode()).hexdigest()
    source_state = f"source-v2:{'1' * 40}:{'2' * 64}:{definitions_sha256}"
    (aicore_build_directory / ".git_commit").write_text(source_state, encoding="utf-8")

    def inspect_fake_artifact(path: Path | str) -> report_module.BuildArtifactIdentity:
        artifact = Path(path).resolve()
        data = artifact.read_bytes()
        literal_text = data[data.index(b":") + 1 :]
        return report_module.BuildArtifactIdentity(
            path=artifact,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            text_sha256=hashlib.sha256(literal_text).hexdigest(),
            text_size_bytes=len(literal_text),
        )

    monkeypatch.setattr(report_module, "_inspect_build_artifact", inspect_fake_artifact)
    return capture_build_identity(
        profile=profile,
        profiled_cache_key=("a5", "fdwic", profile),
        aicore_extra_cache_key=extra_cache_key,
        compile_definitions=frozen_compile_definitions,
        aicore_kernel=kernel,
        aicore_build_dir=aicore_build_directory,
        host_runtime=artifact_paths[-1],
    )


def _publish_fake_provenance(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, report_module.SubmitPmuBuildIdentity]:
    raw_path = _write_capture(directory, _valid_capture())
    identity = _fake_build_identity(directory, monkeypatch)
    output_path, provenance_path = write_report_with_provenance(raw_path, identity)
    return output_path, provenance_path, identity


def test_inspect_build_artifact_uses_real_readelf_and_hashes_literal_text(tmp_path: Path) -> None:
    literal_text = b"\x90\x66\x90\xc3\x00\xff"
    artifact = tmp_path / "minimal.o"
    contents = _write_minimal_elf(artifact, code=literal_text)

    identity = report_module._inspect_build_artifact(artifact)

    assert identity.path == artifact.resolve()
    assert identity.size_bytes == len(contents)
    assert identity.sha256 == hashlib.sha256(contents).hexdigest()
    assert identity.text_size_bytes == len(literal_text)
    assert identity.text_sha256 == hashlib.sha256(literal_text).hexdigest()


def test_inspect_build_artifact_rejects_real_elf_without_text_and_invalid_file(tmp_path: Path) -> None:
    no_text = tmp_path / "no-text.o"
    _write_minimal_elf(no_text, code_section_name=".code")
    with pytest.raises(ValueError, match="has no literal .text section"):
        report_module._inspect_build_artifact(no_text)

    invalid = tmp_path / "invalid.o"
    invalid.write_bytes(b"not an ELF object")
    with pytest.raises(ValueError, match="readelf failed for build artifact"):
        report_module._inspect_build_artifact(invalid)


def test_valid_capture_is_recomputed_and_rendered(tmp_path: Path) -> None:
    raw_path = _write_capture(tmp_path, _valid_capture())

    capture = load_capture(raw_path)
    assert len(capture.records) == 96
    assert len(capture.groups["aic"]) == 32
    assert len(capture.groups["aiv"]) == 64
    assert all("shadow_icache_requests" not in record for record in capture.records)
    assert all("shadow_icache_misses" not in record for record in capture.records)
    expected_ratios = {
        "all": (16_000 / 9_000, 16_047.5 / 9_000, 16_095 / 9_000),
        "aic": (16_000 / 9_000, 16_015.5 / 9_000, 16_031 / 9_000),
        "aiv": (16_032 / 9_000, 16_063.5 / 9_000, 16_095 / 9_000),
    }
    for group_name, (minimum, mean, maximum) in expected_ratios.items():
        summary = capture.summary[group_name]
        assert summary["submit_elapsed_ticks"]["min"] == 10_000
        assert summary["submit_elapsed_ticks"]["mean"] == 10_000
        assert summary["submit_elapsed_ticks"]["max"] == 10_000
        assert summary["scalar_submit_elapsed_ticks"]["min"] == 9_000
        assert summary["scalar_submit_elapsed_ticks"]["mean"] == 9_000
        assert summary["scalar_submit_elapsed_ticks"]["max"] == 9_000
        assert summary["scalar_denominator_excluded_wall_ticks"]["min"] == 1_000
        assert summary["scalar_denominator_excluded_wall_ticks"]["mean"] == 1_000
        assert summary["scalar_denominator_excluded_wall_ticks"]["max"] == 1_000
        assert summary["non_scalar_busy_cycles"]["min"] == 4_000
        assert summary["non_scalar_busy_cycles"]["mean"] == 4_000
        assert summary["non_scalar_busy_cycles"]["max"] == 4_000
        ratio = summary["pmu_total_cycles_per_scalar_tick"]
        assert ratio["min"] == pytest.approx(minimum)
        assert ratio["mean"] == pytest.approx(mean)
        assert ratio["max"] == pytest.approx(maximum)

    document = render_report(raw_path)
    assert "真实 FDWIC Scalar Submit PMU" in document
    assert "32 AIC + 64 AIV" in document
    assert "Σmiss/Σrequest" in document
    assert "90.000 ns" in document
    assert "不是 Submit 墙钟损失" in document
    assert "非 Scalar-busy 残余不是空闲时间，也不是 I-cache stall" in document
    assert "Scalar Submit 时间分母/core" in document
    assert "PMU-total / Scalar SYS/core" in document
    assert "不要求精确等于校准频率" in document
    assert "result-used return-ready atomic 的依赖区间只从 Scalar SYS 时间分母" in document
    assert "I-cache/PMU counter 仍包含其指令事件" in document
    assert "source-issue atomic" in document
    assert "不能称为纯 Kernel 时间" in document
    assert "不能与 perf-clock、swimlane 或另一个 phase ELF 相减" in document
    for cycles_per_ns in (1.649844, 1.650062, 1.649731):
        assert f"按 {cycles_per_ns:.6f} cycles/ns 校准" in document
    assert "阶段观察（phase_id=" not in document
    assert "均值 16,047.5；最小 16,000；最大 16,095" in document
    for group_name in ("all", "aic", "aiv"):
        summary = capture.summary[group_name]
        requests = summary["icache_requests"]
        misses = summary["icache_misses"]
        assert (
            f"<dt>Primary I-cache request/core</dt><dd>"
            f"最小 {requests['min']:,}；最大 {requests['max']:,}</dd>"
        ) in document
        assert (
            f"<dt>Primary I-cache miss/core</dt><dd>"
            f"最小 {misses['min']:,}；最大 {misses['max']:,}</dd>"
        ) in document
        assert (
            f"<dt>90 ns 直觉量尺/core</dt>\n"
            f"          <dd>最小 {misses['min'] * 90 / 1_000:,.3f}；"
            f"最大 {misses['max'] * 90 / 1_000:,.3f} µs</dd>"
        ) in document
        assert f"{summary['icache_miss_rate']:.3%}" in document
    assert document.count("<svg") == 5
    assert document.count("<circle") == 5 * 96
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() in document


def test_python_only_derived_metrics_do_not_expand_raw_summary(tmp_path: Path) -> None:
    capture_data = _valid_capture()
    producer_keys = {
        "cores",
        "total_cycles",
        "scalar_submit_elapsed_ticks",
        "scalar_busy",
        "icache_requests",
        "icache_misses",
        "scalar_busy_share",
        "icache_miss_rate",
    }
    for group in capture_data["summary"].values():
        assert set(group) == producer_keys

    capture = load_capture(_write_capture(tmp_path, capture_data))

    for group in capture.summary.values():
        assert {
            "submit_elapsed_ticks",
            "scalar_submit_elapsed_ticks",
            "scalar_denominator_excluded_wall_ticks",
            "non_scalar_busy_cycles",
            "pmu_total_cycles_per_scalar_tick",
        } <= set(group)


def test_non_scalar_busy_extrema_are_computed_per_record(tmp_path: Path) -> None:
    capture_data = _valid_capture()
    records = capture_data["records"]
    records[0]["total_cycles"] = 20_000
    records[0]["scalar_busy"] = 19_000
    records[1]["total_cycles"] = 19_000
    records[1]["scalar_busy"] = 1_000
    _refresh_host_aggregates(capture_data)

    capture = load_capture(_write_capture(tmp_path, capture_data))

    residual = capture.summary["all"]["non_scalar_busy_cycles"]
    assert residual["min"] == 1_000
    assert residual["max"] == 18_000
    assert residual["min"] != (
        capture.summary["all"]["total_cycles"]["min"]
        - capture.summary["all"]["scalar_busy"]["min"]
    )
    assert residual["max"] != (
        capture.summary["all"]["total_cycles"]["max"]
        - capture.summary["all"]["scalar_busy"]["max"]
    )


def test_effective_ratio_is_mean_of_per_core_ratios(tmp_path: Path) -> None:
    capture_data = _valid_capture()
    for logical_core_id, record in enumerate(capture_data["records"]):
        elapsed = 10_000 if logical_core_id % 2 == 0 else 20_000
        record["last_submit_end_tick"] = record["first_submit_start_tick"] + elapsed
        record["submit_elapsed_ticks"] = elapsed
        record["scalar_submit_elapsed_ticks"] = elapsed
        record["total_cycles"] = 10_000 if logical_core_id % 2 == 0 else 40_000
        record["scalar_busy"] = 8_000
    _refresh_host_aggregates(capture_data)

    capture = load_capture(_write_capture(tmp_path, capture_data))

    for group_name in ("all", "aic", "aiv"):
        ratio = capture.summary[group_name]["pmu_total_cycles_per_scalar_tick"]
        assert ratio["min"] == pytest.approx(1.0)
        assert ratio["mean"] == pytest.approx(1.5)
        assert ratio["max"] == pytest.approx(2.0)
        group = capture.groups[group_name]
        ratio_of_sums = sum(record["total_cycles"] for record in group) / sum(
            record["scalar_submit_elapsed_ticks"] for record in group
        )
        assert ratio["mean"] != pytest.approx(ratio_of_sums)


def test_zero_submit_elapsed_is_rejected_before_derived_ratio(tmp_path: Path) -> None:
    capture_data = _valid_capture()
    record = capture_data["records"][0]
    record["last_submit_end_tick"] = record["first_submit_start_tick"]
    record["submit_elapsed_ticks"] = 0

    with pytest.raises(ValueError, match=r"records\[0\]\.submit_elapsed_ticks must be an integer >= 1"):
        load_capture(_write_capture(tmp_path, capture_data))


def test_group_cards_use_role_specific_cycle_calibration(tmp_path: Path) -> None:
    capture = load_capture(_write_capture(tmp_path, _valid_capture()))

    document = render_report(capture.input_path)

    expected_mean_us = {
        "all": 16_047.5 / 1.649844 / 1_000,
        "aic": 16_015.5 / 1.650062 / 1_000,
        "aiv": 16_063.5 / 1.649731 / 1_000,
    }
    for group_name, mean_us in expected_mean_us.items():
        cycles_per_ns = capture.data["configuration"]["pmu_cycles_per_ns"][group_name]
        assert f"按 {cycles_per_ns:.6f} cycles/ns 校准" in document
        assert f"等效时间 均值 {mean_us:,.3f}；" in document


def test_valid_arg_build_capture_renders_same_elf_phase_observation_first(tmp_path: Path) -> None:
    raw_path = _write_capture(tmp_path, _valid_arg_build_capture())

    capture = load_capture(raw_path)

    assert capture.phase_summary is not None
    assert all("shadow_icache_requests" in record for record in capture.records)
    assert all("shadow_icache_misses" in record for record in capture.records)
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
    assert "Request observed（总数 / 逐核 min–max / Primary）" in document
    assert "Miss observed（总数 / 逐核 min–max / Primary）" in document
    assert "observed_plus_capture_gap = observed + (primary − shadow)" in document
    assert "插桩 bookkeeping 会进入 sample" in document
    assert "不是原业务" in document
    assert "事件数的数学上下界" in document
    assert "时间占比的分母是同一 ELF 的 Σ每核 Scalar Submit elapsed" in document
    assert "result-used return-ready atomic 依赖区间只从时间扣除" in document
    assert "I-cache/PMU counter 仍含其指令事件" in document
    assert "source-issue atomic 保留" in document
    assert "request/miss 百分比的分母才是" in document
    assert "同一次采集的 Submit 整窗 primary" in document
    assert "每次调用 elapsed" in document
    assert "每次调用 elapsed / request / miss observed" not in document
    assert "是每核累计" in document
    assert "不是逐调用极值" in document
    assert "不能跨 ELF 相减" in document
    assert "shadow 是 begin/end/final 全部 running read-clear 返回值之和" in document
    for group_name in ("all", "aic", "aiv"):
        phase = capture.phase_summary[group_name]
        group_records = capture.groups[group_name]
        requests = phase["phase_icache_requests_observed"]
        misses = phase["phase_icache_misses_observed"]
        assert phase["primary_icache_requests"] == sum(record["icache_requests"] for record in group_records)
        assert phase["primary_icache_misses"] == sum(record["icache_misses"] for record in group_records)
        assert f"{phase['phase_time_share_of_scalar_submit']:.3%}" in document
        assert f"{phase['phase_request_observed_share_of_primary']:.3%}" in document
        assert f"{phase['phase_request_observed_plus_capture_gap_share_of_primary']:.3%}" in document
        assert f"{phase['phase_miss_observed_share_of_primary']:.3%}" in document
        assert f"{phase['phase_miss_observed_plus_capture_gap_share_of_primary']:.3%}" in document
        assert f"{phase['phase_elapsed_ticks_per_call']:,.3f} ns" in document
        assert f"逐核 最小 {requests['min']:,}；最大 {requests['max']:,}" in document
        assert f"逐核 最小 {misses['min']:,}；最大 {misses['max']:,}" in document
        assert f"Primary {phase['primary_icache_requests']:,}" in document
        assert f"Primary {phase['primary_icache_misses']:,}" in document


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
        requests = phase["phase_icache_requests_observed"]
        misses = phase["phase_icache_misses_observed"]
        assert f"{phase['phase_elapsed_ticks_per_call']:,.3f} ns" in document
        assert f"逐核 最小 {requests['min']:,}；最大 {requests['max']:,}" in document
        assert f"逐核 最小 {misses['min']:,}；最大 {misses['max']:,}" in document


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


def test_valid_prepare_map_capture_tracks_call_body_boundary(tmp_path: Path) -> None:
    raw_path = _write_capture(tmp_path, _valid_prepare_map_capture())

    capture = load_capture(raw_path)

    assert capture.phase_summary is not None
    assert capture.phase_summary["all"]["phase_begin_reads"] == 96 * 5
    document = render_report(raw_path)
    assert document.index("<code>prepare-map</code> 阶段观察") < document.index("全局 Submit 时间范围")
    assert "phase_id=8" in document
    assert "dist_submit_prepare_map_call_entry_to_return" in document
    assert "running_read_clear_observed_bracket" in document
    assert "inner_sys_cnt_between_boundary_observers" in document


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
def test_prepare_map_rejects_mismatched_phase_configuration(tmp_path: Path, field: str, value: Any) -> None:
    capture = _valid_prepare_map_capture()
    capture["configuration"]["phase"][field] = value

    with pytest.raises(ValueError, match=r"configuration\.phase"):
        load_capture(_write_capture(tmp_path, capture))


def test_prepare_map_rejects_wrong_record_phase_id(tmp_path: Path) -> None:
    capture = _valid_prepare_map_capture()
    capture["records"][0]["phase_id"] = 3

    with pytest.raises(ValueError, match=r"records\[0\]\.phase_id must equal 8"):
        load_capture(_write_capture(tmp_path, capture))


def test_valid_fanin_capture_accepts_dynamic_zero_call_cores(tmp_path: Path) -> None:
    raw_path = _write_capture(tmp_path, _valid_fanin_capture())

    capture = load_capture(raw_path)

    assert capture.phase_summary is not None
    assert capture.phase_summary["all"]["phase_begin_reads"] == 4
    assert capture.phase_summary["aic"]["phase_begin_reads"] == 2
    assert capture.phase_summary["aiv"]["phase_begin_reads"] == 2
    assert capture.phase_summary["all"]["phase_zero_call_cores"] == 92
    document = render_report(raw_path)
    assert "<code>fanin</code> 阶段观察" in document
    assert "phase_id=9" in document
    assert "fanin_begin_to_fanin_end" in document
    assert "call_shape=dynamic_balanced" in document
    assert "expected_calls=ALL 4 / AIC 2 / AIV 2" in document
    assert "零调用核 92" in document


def test_fanin_rejects_unbalanced_per_core_boundaries(tmp_path: Path) -> None:
    capture = _valid_fanin_capture()
    capture["records"][0]["phase_end_reads"] = 0

    with pytest.raises(ValueError, match="dynamic phase begin/end reads must be balanced"):
        load_capture(_write_capture(tmp_path, capture))


def test_fanin_rejects_wrong_global_role_call_totals(tmp_path: Path) -> None:
    capture = _valid_fanin_capture()
    for field in ("phase_begin_reads", "phase_end_reads"):
        capture["records"][1][field] = 0
        capture["records"][34][field] = 1
    capture["records"][1]["phase_elapsed_ticks"] = 0
    capture["records"][1]["phase_icache_requests_observed"] = 0
    capture["records"][1]["phase_icache_misses_observed"] = 0
    capture["records"][34]["phase_elapsed_ticks"] = 75
    capture["records"][34]["phase_icache_requests_observed"] = 40
    capture["records"][34]["phase_icache_misses_observed"] = 2

    with pytest.raises(ValueError, match="dynamic phase call totals"):
        load_capture(_write_capture(tmp_path, capture))


def test_fanin_rejects_nonzero_values_on_zero_call_core(tmp_path: Path) -> None:
    capture = _valid_fanin_capture()
    capture["records"][2]["phase_elapsed_ticks"] = 1

    with pytest.raises(ValueError, match="zero-call dynamic phase must have zero"):
        load_capture(_write_capture(tmp_path, capture))


def test_valid_winner_build_control_capture_excludes_linked_kernel_segments(tmp_path: Path) -> None:
    raw_path = _write_capture(tmp_path, _valid_winner_build_capture())

    capture = load_capture(raw_path)

    assert capture.phase_summary is not None
    all_phase = capture.phase_summary["all"]
    assert all_phase["phase_business_calls"] == 4
    assert capture.phase_summary["aic"]["phase_business_calls"] == 2
    assert capture.phase_summary["aiv"]["phase_business_calls"] == 2
    assert all_phase["phase_begin_reads"] == 6
    assert all_phase["phase_end_reads"] == 6
    assert all_phase["phase_excluded_kernel_calls"] == 2
    assert all_phase["phase_elapsed_ticks_per_call"] == pytest.approx(75)
    assert all_phase["phase_elapsed_ticks_per_call"] != pytest.approx(
        all_phase["phase_elapsed_ticks"]["sum"] / all_phase["phase_begin_reads"]
    )
    assert all_phase["phase_icache_requests_observed_per_call"] == pytest.approx(40)
    assert all_phase["phase_icache_misses_observed_per_call"] == pytest.approx(2)
    assert all_phase["phase_calls_per_core"]["min"] == 0
    assert all_phase["phase_calls_per_core"]["max"] == 1
    assert all_phase["phase_zero_call_cores"] == 92
    document = render_report(raw_path)
    assert "<code>winner-build-control</code> 阶段观察" in document
    assert "phase_id=10" in document
    assert "winner_build_begin_to_end_excluding_linked_kernel_calls" in document
    assert "discontinuous_running_read_clear_excluding_linked_kernel_calls" in document
    assert "discontinuous_sys_cnt_control_segments_excluding_linked_kernel_calls" in document
    assert "call_shape=dynamic_balanced" in document
    assert "expected_calls=ALL 4 / AIC 2 / AIV 2" in document
    assert "业务调用 4 次；排除 linked Kernel 调用 2 次" in document
    assert "零调用核 92" in document


def test_winner_build_control_global_shape_uses_business_calls_not_boundary_reads(tmp_path: Path) -> None:
    capture = _valid_winner_build_capture()
    capture["records"][0]["phase_excluded_kernel_calls"] += 1
    capture["records"][0]["phase_elapsed_ticks"] = 0
    capture["records"][0]["phase_icache_requests_observed"] = 0
    capture["records"][0]["phase_icache_misses_observed"] = 0

    with pytest.raises(ValueError, match="dynamic phase call totals"):
        load_capture(_write_capture(tmp_path, capture))


@pytest.mark.parametrize("value", (None, -1))
def test_winner_build_control_requires_nonnegative_excluded_kernel_calls(
    tmp_path: Path,
    value: int | None,
) -> None:
    capture = _valid_winner_build_capture()
    if value is None:
        capture["records"][0].pop("phase_excluded_kernel_calls")
    else:
        capture["records"][0]["phase_excluded_kernel_calls"] = value

    with pytest.raises(ValueError, match=r"records\[0\]\.phase_excluded_kernel_calls must be an integer >= 0"):
        load_capture(_write_capture(tmp_path, capture))


def test_winner_build_control_rejects_excluded_count_above_boundary_reads(tmp_path: Path) -> None:
    capture = _valid_winner_build_capture()
    capture["records"][0]["phase_excluded_kernel_calls"] = 3

    with pytest.raises(ValueError, match="excluded Kernel calls exceed phase begin/end reads"):
        load_capture(_write_capture(tmp_path, capture))


def test_winner_build_control_rejects_unbalanced_business_boundaries(tmp_path: Path) -> None:
    capture = _valid_winner_build_capture()
    capture["records"][0]["phase_end_reads"] -= 1

    with pytest.raises(ValueError, match="must remain balanced after subtracting"):
        load_capture(_write_capture(tmp_path, capture))


def test_winner_build_control_zero_business_calls_require_zero_observed_values(tmp_path: Path) -> None:
    capture = _valid_winner_build_capture()
    capture["records"][0]["phase_excluded_kernel_calls"] = capture["records"][0]["phase_begin_reads"]

    with pytest.raises(ValueError, match="zero-call dynamic phase must have zero"):
        load_capture(_write_capture(tmp_path, capture))


def test_winner_build_control_requires_kernel_exclusion_validation(tmp_path: Path) -> None:
    capture = _valid_winner_build_capture()
    capture["validation"].pop("phase_kernel_exclusion_closed_records")

    with pytest.raises(ValueError, match=r"validation\.phase_kernel_exclusion_closed_records"):
        load_capture(_write_capture(tmp_path, capture))


@pytest.mark.parametrize(
    ("winner_core_id", "winner_role", "other_role"),
    ((0, "aic", "aiv"), (32, "aiv", "aic")),
)
def test_valid_alloc_complete_control_accepts_role_unlocked_b1_winner(
    tmp_path: Path,
    winner_core_id: int,
    winner_role: str,
    other_role: str,
) -> None:
    raw_path = _write_capture(
        tmp_path,
        _valid_alloc_complete_capture(winner_core_id=winner_core_id),
    )

    capture = load_capture(raw_path)

    assert capture.records[winner_core_id]["role"] == winner_role
    assert capture.phase_summary is not None
    all_phase = capture.phase_summary["all"]
    assert all_phase["phase_business_calls"] == 1
    assert capture.phase_summary[winner_role]["phase_business_calls"] == 1
    assert capture.phase_summary[other_role]["phase_business_calls"] == 0
    assert all_phase["phase_begin_reads"] == 3
    assert all_phase["phase_end_reads"] == 3
    assert all_phase["phase_excluded_kernel_calls"] == 2
    assert all_phase["phase_elapsed_ticks_per_call"] == pytest.approx(75)
    assert all_phase["phase_icache_requests_observed_per_call"] == pytest.approx(40)
    assert all_phase["phase_icache_misses_observed_per_call"] == pytest.approx(2)
    assert all_phase["phase_calls_per_core"]["min"] == 0
    assert all_phase["phase_calls_per_core"]["max"] == 1
    assert all_phase["phase_zero_call_cores"] == 95

    document = render_report(raw_path)
    assert "<code>alloc-complete-control</code> 阶段观察" in document
    assert "phase_id=11" in document
    assert "alloc_complete_begin_to_end_excluding_linked_kernel_calls" in document
    assert "discontinuous_running_read_clear_excluding_linked_kernel_calls" in document
    assert "discontinuous_sys_cnt_control_segments_excluding_linked_kernel_calls" in document
    assert "call_shape=dynamic_global" in document
    assert "expected_calls=ALL 1（角色不锁定）" in document
    assert "业务调用 1 次；排除 linked Kernel 调用 2 次" in document
    assert "零调用核 95" in document


def test_alloc_complete_control_rejects_wrong_global_call_total(tmp_path: Path) -> None:
    capture = _valid_alloc_complete_capture()
    winner = capture["records"][0]
    winner["phase_begin_reads"] = winner["phase_excluded_kernel_calls"]
    winner["phase_end_reads"] = winner["phase_excluded_kernel_calls"]
    winner["phase_elapsed_ticks"] = 0
    winner["phase_icache_requests_observed"] = 0
    winner["phase_icache_misses_observed"] = 0

    with pytest.raises(ValueError, match="dynamic phase call totals: global call total must equal 1"):
        load_capture(_write_capture(tmp_path, capture))


def test_alloc_complete_control_rejects_per_core_call_count_above_batch_limit(tmp_path: Path) -> None:
    capture = _valid_alloc_complete_capture()
    winner = capture["records"][0]
    winner["phase_begin_reads"] += 1
    winner["phase_end_reads"] += 1

    with pytest.raises(ValueError, match=r"records\[0\] business phase calls exceed dynamic per-core maximum 1"):
        load_capture(_write_capture(tmp_path, capture))


def test_alloc_complete_control_rejects_unbalanced_business_boundaries(tmp_path: Path) -> None:
    capture = _valid_alloc_complete_capture()
    capture["records"][0]["phase_end_reads"] -= 1

    with pytest.raises(ValueError, match="must remain balanced after subtracting"):
        load_capture(_write_capture(tmp_path, capture))


def test_alloc_complete_control_rejects_nonzero_values_on_zero_call_core(tmp_path: Path) -> None:
    capture = _valid_alloc_complete_capture()
    capture["records"][1]["phase_icache_requests_observed"] = 1

    with pytest.raises(ValueError, match="zero-call dynamic phase must have zero"):
        load_capture(_write_capture(tmp_path, capture))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("id", 10),
        ("name", "winner-build-control"),
        ("boundary", "winner_build_begin_to_end_excluding_linked_kernel_calls"),
        ("call_shape", "dynamic_balanced"),
        ("expected_calls", {"all": 2}),
        ("counter_semantics", "running_read_clear_observed_bracket"),
        ("time_semantics", "inner_sys_cnt_between_boundary_observers"),
    ),
)
def test_alloc_complete_control_rejects_mismatched_phase_configuration(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    capture = _valid_alloc_complete_capture()
    capture["configuration"]["phase"][field] = value

    with pytest.raises(ValueError, match=r"configuration\.phase"):
        load_capture(_write_capture(tmp_path, capture))


def test_alloc_complete_control_rejects_wrong_record_phase_id(tmp_path: Path) -> None:
    capture = _valid_alloc_complete_capture()
    capture["records"][0]["phase_id"] = 10

    with pytest.raises(ValueError, match=r"records\[0\]\.phase_id must equal 11"):
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


def test_valid_submit_transition_capture_uses_submit_count_minus_one(tmp_path: Path) -> None:
    raw_path = _write_capture(tmp_path, _valid_submit_transition_capture())

    capture = load_capture(raw_path)

    assert capture.phase_summary is not None
    assert capture.phase_summary["all"]["phase_begin_reads"] == 96 * 4
    assert capture.phase_summary["all"]["phase_end_reads"] == 96 * 4
    document = render_report(raw_path)
    assert document.index("<code>submit-transition</code> 阶段观察") < document.index("全局 Submit 时间范围")
    assert "phase_id=6" in document
    assert "previous_submit_end_to_next_submit_begin" in document
    assert "expected_calls_per_core=4" in document


def test_submit_transition_accepts_minimum_two_submit_shape(tmp_path: Path) -> None:
    capture = _valid_submit_transition_capture()
    capture["configuration"]["expected_submits_per_core"] = 2
    capture["configuration"]["phase"]["expected_calls_per_core"] = 1
    for record in capture["records"]:
        record["submit_count"] = 2
        record["expected_submit_count"] = 2
        record["phase_begin_reads"] = 1
        record["phase_end_reads"] = 1

    loaded = load_capture(_write_capture(tmp_path, capture))

    assert loaded.phase_summary is not None
    assert loaded.phase_summary["all"]["phase_begin_reads"] == 96
    assert loaded.phase_summary["all"]["phase_end_reads"] == 96


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("id", 5),
        ("name", "register"),
        ("boundary", "register_outputs_call_entry_to_return"),
        ("expected_calls_per_core", 5),
        ("expected_calls_per_core", 3),
        ("counter_semantics", "running_read_clear_empty_bracket_calibration"),
        ("time_semantics", "outer_sys_cnt_around_adjacent_begin_end_pair"),
    ),
)
def test_submit_transition_rejects_mismatched_phase_configuration(tmp_path: Path, field: str, value: Any) -> None:
    capture = _valid_submit_transition_capture()
    capture["configuration"]["phase"][field] = value

    with pytest.raises(ValueError, match=r"configuration\.phase"):
        load_capture(_write_capture(tmp_path, capture))


@pytest.mark.parametrize("reads", [3, 5])
def test_submit_transition_rejects_wrong_phase_call_count(tmp_path: Path, reads: int) -> None:
    capture = _valid_submit_transition_capture()
    capture["records"][0]["phase_begin_reads"] = reads
    capture["records"][0]["phase_end_reads"] = reads

    with pytest.raises(ValueError, match="phase begin/end reads must both equal 4"):
        load_capture(_write_capture(tmp_path, capture))


def test_submit_transition_rejects_wrong_record_phase_id(tmp_path: Path) -> None:
    capture = _valid_submit_transition_capture()
    capture["records"][0]["phase_id"] = 5

    with pytest.raises(ValueError, match=r"records\[0\]\.phase_id must equal 6"):
        load_capture(_write_capture(tmp_path, capture))


def test_submit_transition_rejects_single_submit_capture(tmp_path: Path) -> None:
    capture = _valid_submit_transition_capture()
    capture["configuration"]["expected_submits_per_core"] = 1
    capture["configuration"]["phase"]["expected_calls_per_core"] = 0

    with pytest.raises(ValueError, match="requires at least two submits per core"):
        load_capture(_write_capture(tmp_path, capture))


def test_valid_efdrain_control_capture_excludes_linked_kernel_segments(tmp_path: Path) -> None:
    raw_path = _write_capture(tmp_path, _valid_efdrain_control_capture())

    capture = load_capture(raw_path)

    assert capture.phase_summary is not None
    all_phase = capture.phase_summary["all"]
    outer_calls = 96 * 5
    excluded_kernel_calls = sum(record["phase_excluded_kernel_calls"] for record in capture.records)
    total_elapsed = sum(record["phase_elapsed_ticks"] for record in capture.records)
    total_requests = sum(record["phase_icache_requests_observed"] for record in capture.records)
    total_misses = sum(record["phase_icache_misses_observed"] for record in capture.records)
    assert capture.data["validation"]["phase_kernel_exclusion_closed_records"] == 96
    assert all_phase["phase_excluded_kernel_calls"] == excluded_kernel_calls
    assert all_phase["phase_begin_reads"] == outer_calls + excluded_kernel_calls
    assert all_phase["phase_end_reads"] == outer_calls + excluded_kernel_calls
    assert all_phase["phase_elapsed_ticks_per_call"] == pytest.approx(total_elapsed / outer_calls)
    assert all_phase["phase_elapsed_ticks_per_call"] != pytest.approx(
        total_elapsed / all_phase["phase_end_reads"]
    )
    assert all_phase["phase_icache_requests_observed_per_call"] == pytest.approx(total_requests / outer_calls)
    assert all_phase["phase_icache_misses_observed_per_call"] == pytest.approx(total_misses / outer_calls)

    document = render_report(raw_path)
    assert "<code>efdrain-control</code> 阶段观察" in document
    assert "phase_id=7" in document
    assert "efdrain_begin_to_end_excluding_linked_kernel_calls" in document
    assert "discontinuous_running_read_clear_excluding_linked_kernel_calls" in document
    assert "discontinuous_sys_cnt_control_segments_excluding_linked_kernel_calls" in document
    assert "elapsed、request 和 miss 都排除该 Kernel 整段" in document
    assert "result-used return-ready atomic" in document
    assert "request/miss 仍含 atomic 指令事件" in document
    assert "source-issue atomic" in document
    assert "每次调用的分母仍是外层业务调用次数" in document
    assert "<th>Begin / End reads</th>" in document
    assert (
        f"<small>业务调用 {outer_calls} 次；排除 linked Kernel 调用 {excluded_kernel_calls} 次</small>"
        in document
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("id", 6),
        ("name", "submit-transition"),
        ("boundary", "efdrain_begin_to_end"),
        ("expected_calls_per_core", 4),
        ("counter_semantics", "running_read_clear_observed_bracket"),
        ("time_semantics", "inner_sys_cnt_between_boundary_observers"),
    ),
)
def test_efdrain_control_rejects_mismatched_phase_configuration(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    capture = _valid_efdrain_control_capture()
    capture["configuration"]["phase"][field] = value

    with pytest.raises(ValueError, match=r"configuration\.phase"):
        load_capture(_write_capture(tmp_path, capture))


def test_efdrain_control_requires_excluded_kernel_call_count(tmp_path: Path) -> None:
    capture = _valid_efdrain_control_capture()
    capture["records"][0].pop("phase_excluded_kernel_calls")

    with pytest.raises(
        ValueError,
        match=r"records\[0\]\.phase_excluded_kernel_calls must be an integer >= 0",
    ):
        load_capture(_write_capture(tmp_path, capture))


def test_efdrain_control_rejects_negative_excluded_kernel_call_count(tmp_path: Path) -> None:
    capture = _valid_efdrain_control_capture()
    capture["records"][0]["phase_excluded_kernel_calls"] = -1

    with pytest.raises(
        ValueError,
        match=r"records\[0\]\.phase_excluded_kernel_calls must be an integer >= 0",
    ):
        load_capture(_write_capture(tmp_path, capture))


@pytest.mark.parametrize("field", ("phase_begin_reads", "phase_end_reads"))
def test_efdrain_control_rejects_reads_not_closed_against_excluded_kernel_calls(
    tmp_path: Path,
    field: str,
) -> None:
    capture = _valid_efdrain_control_capture()
    capture["records"][1][field] -= 1

    with pytest.raises(
        ValueError,
        match=r"records\[1\] phase begin/end reads must both equal expected calls 5 "
        r"\+ excluded Kernel calls 1 = 6",
    ):
        load_capture(_write_capture(tmp_path, capture))


def test_efdrain_control_rejects_wrong_record_phase_id(tmp_path: Path) -> None:
    capture = _valid_efdrain_control_capture()
    capture["records"][0]["phase_id"] = 6

    with pytest.raises(ValueError, match=r"records\[0\]\.phase_id must equal 7"):
        load_capture(_write_capture(tmp_path, capture))


@pytest.mark.parametrize("value", (None, 95))
def test_efdrain_control_requires_closed_kernel_exclusion_validation(
    tmp_path: Path,
    value: int | None,
) -> None:
    capture = _valid_efdrain_control_capture()
    if value is None:
        capture["validation"].pop("phase_kernel_exclusion_closed_records")
    else:
        capture["validation"]["phase_kernel_exclusion_closed_records"] = value

    with pytest.raises(ValueError, match=r"validation\.phase_kernel_exclusion_closed_records"):
        load_capture(_write_capture(tmp_path, capture))


@pytest.mark.parametrize(
    "capture_factory",
    (
        _valid_arg_build_capture,
        _valid_empty_bracket_capture,
        _valid_materialize_capture,
        _valid_claim_capture,
        _valid_register_capture,
        _valid_submit_transition_capture,
        _valid_prepare_map_capture,
        _valid_fanin_capture,
        _valid_efdrain_control_capture,
        _valid_winner_build_capture,
        _valid_alloc_complete_capture,
    ),
)
def test_all_phase_modes_require_excluded_kernel_call_count(
    tmp_path: Path,
    capture_factory: Callable[[], dict[str, Any]],
) -> None:
    capture = capture_factory()
    capture["records"][0].pop("phase_excluded_kernel_calls")

    with pytest.raises(
        ValueError,
        match=r"records\[0\]\.phase_excluded_kernel_calls must be an integer >= 0",
    ):
        load_capture(_write_capture(tmp_path, capture))


def test_none_forbids_kernel_exclusion_validation(tmp_path: Path) -> None:
    capture = _valid_capture()
    capture["validation"]["phase_kernel_exclusion_closed_records"] = 96

    with pytest.raises(
        ValueError,
        match=r"phase_kernel_exclusion_closed_records is only valid in",
    ):
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


def test_arg_build_rejects_phase_time_beyond_scalar_denominator(tmp_path: Path) -> None:
    capture = _valid_arg_build_capture()
    capture["records"][0]["phase_elapsed_ticks"] = capture["records"][0]["scalar_submit_elapsed_ticks"] + 1

    with pytest.raises(ValueError, match="phase_elapsed_ticks exceeds scalar_submit_elapsed_ticks"):
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


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("enabled", False),
        ("classification", "all_atomic"),
        ("time_boundary", "atomic_issue_only"),
        ("counter_semantics", "pmu_counters_exclude_atomic_instruction_events"),
        ("time_denominator_effect", "retain_return_ready_atomic_elapsed"),
    ),
)
def test_return_ready_atomic_metadata_is_exact(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    capture = _valid_capture()
    capture["configuration"]["return_ready_atomic_exclusion"][field] = value

    with pytest.raises(ValueError, match=r"configuration\.return_ready_atomic_exclusion"):
        load_capture(_write_capture(tmp_path, capture))


@pytest.mark.parametrize("value", (None, 95))
def test_return_ready_atomic_time_validation_requires_all_cores(
    tmp_path: Path,
    value: int | None,
) -> None:
    capture = _valid_capture()
    if value is None:
        capture["validation"].pop("return_ready_atomic_time_valid_records")
    else:
        capture["validation"]["return_ready_atomic_time_valid_records"] = value

    with pytest.raises(ValueError, match=r"validation\.return_ready_atomic_time_valid_records"):
        load_capture(_write_capture(tmp_path, capture))


def test_none_status_requires_shadow_match_bit(tmp_path: Path) -> None:
    capture = _valid_capture()
    capture["records"][0]["status"] &= ~(1 << 18)

    with pytest.raises(ValueError, match=r"records\[0\]\.status must equal"):
        load_capture(_write_capture(tmp_path, capture))


def test_phase_status_requires_return_ready_atomic_time_bit(tmp_path: Path) -> None:
    capture = _valid_arg_build_capture()
    capture["records"][0]["status"] &= ~(1 << 17)

    with pytest.raises(ValueError, match=r"records\[0\]\.status must equal"):
        load_capture(_write_capture(tmp_path, capture))


def test_scalar_submit_elapsed_must_be_positive(tmp_path: Path) -> None:
    capture = _valid_capture()
    capture["records"][0]["scalar_submit_elapsed_ticks"] = 0

    with pytest.raises(
        ValueError,
        match=r"records\[0\]\.scalar_submit_elapsed_ticks must be an integer >= 1",
    ):
        load_capture(_write_capture(tmp_path, capture))


def test_scalar_submit_elapsed_must_not_exceed_wall(tmp_path: Path) -> None:
    capture = _valid_capture()
    capture["records"][0]["scalar_submit_elapsed_ticks"] = capture["records"][0]["submit_elapsed_ticks"] + 1

    with pytest.raises(ValueError, match=r"scalar_submit_elapsed_ticks exceeds submit_elapsed_ticks"):
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


def test_provenance_publish_preserves_raw_and_closes_sidecar_and_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = _write_capture(tmp_path, _valid_capture())
    raw_before = raw_path.read_bytes()
    identity = _fake_build_identity(tmp_path, monkeypatch)

    output_path, provenance_path = write_report_with_provenance(raw_path, identity)

    assert output_path == tmp_path / DEFAULT_OUTPUT_NAME
    assert provenance_path == tmp_path / DEFAULT_PROVENANCE_NAME
    assert raw_path.read_bytes() == raw_before
    assert provenance_path.read_text(encoding="utf-8").endswith("}\n")
    assert output_path.read_text(encoding="utf-8").endswith("</html>\n")
    assert not (tmp_path / f"{DEFAULT_PROVENANCE_NAME}.tmp").exists()
    assert not (tmp_path / f"{DEFAULT_OUTPUT_NAME}.tmp").exists()

    capture = load_capture(raw_path)
    provenance, provenance_sha256 = load_provenance(provenance_path, capture)
    assert provenance["binding"] == {
        "raw_name": DEFAULT_INPUT_NAME,
        "raw_size": len(raw_before),
        "raw_sha256": hashlib.sha256(raw_before).hexdigest(),
        "capture_mode": "submit-pmu-none",
    }
    build = provenance["build"]
    assert build["profile"] == "submit-pmu-none"
    assert build["profiled_cache_key"][-1] == build["profile"]
    assert build["aicore_extra_cache_key"] == "0123456789abcdef"
    assert build["source_state"].startswith("source-v2:")
    assert build["source_state_version"] == "source-v2"
    assert build["source_state_path"] == str(identity.source_state_path)
    assert build["compile_definitions"] == list(identity.compile_definitions)
    assert build["definitions_sha256"] == hashlib.sha256(
        repr(build["compile_definitions"]).encode()
    ).hexdigest()

    frozen_artifacts = dict(identity.artifacts)
    assert set(provenance["artifacts"]) == {
        "aicore_kernel",
        "aic_combined",
        "aiv_combined",
        "host_runtime",
    }
    for name, frozen in frozen_artifacts.items():
        artifact = provenance["artifacts"][name]
        assert artifact == {
            "path": str(frozen.path),
            "sha256": frozen.sha256,
            "size_bytes": frozen.size_bytes,
            "text": {
                "sha256": frozen.text_sha256,
                "size_bytes": frozen.text_size_bytes,
            },
        }

    document = output_path.read_text(encoding="utf-8")
    assert document == render_report(raw_path)
    assert provenance_sha256 in document
    assert "诊断构建身份" in document
    for label in ("AICore final", "AIC combined", "AIV combined", "Host runtime"):
        assert label in document
    for frozen in frozen_artifacts.values():
        assert frozen.sha256 in document
        assert frozen.text_sha256 in document

    assert not list(tmp_path.glob(".*.pending.*.tmp"))
    assert not list(tmp_path.glob(".*.rollback.*.tmp"))


def test_efdrain_control_build_identity_uses_phase_id_7(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _fake_build_identity(
        tmp_path,
        monkeypatch,
        profile=EFDRAIN_CONTROL_CAPTURE_MODE,
    )

    assert identity.profile == EFDRAIN_CONTROL_CAPTURE_MODE
    assert identity.compile_definitions == (
        "PTO_FDWIC_SUBMIT_PMU=1",
        "PTO_FDWIC_SUBMIT_PMU_PHASE_ID=7",
        "PTO_FDWIC_TRACE_ENABLED=0",
    )


@pytest.mark.parametrize(
    ("profile", "capture_factory", "phase_id"),
    (
        (ARG_BUILD_CAPTURE_MODE, _valid_arg_build_capture, 1),
        (EMPTY_BRACKET_CAPTURE_MODE, _valid_empty_bracket_capture, 2),
        (MATERIALIZE_CAPTURE_MODE, _valid_materialize_capture, 3),
        (CLAIM_CAPTURE_MODE, _valid_claim_capture, 4),
        (REGISTER_CAPTURE_MODE, _valid_register_capture, 5),
        (SUBMIT_TRANSITION_CAPTURE_MODE, _valid_submit_transition_capture, 6),
        (EFDRAIN_CONTROL_CAPTURE_MODE, _valid_efdrain_control_capture, 7),
        (PREPARE_MAP_CAPTURE_MODE, _valid_prepare_map_capture, 8),
        (FANIN_CAPTURE_MODE, _valid_fanin_capture, 9),
        (WINNER_BUILD_CAPTURE_MODE, _valid_winner_build_capture, 10),
        (ALLOC_COMPLETE_CAPTURE_MODE, _valid_alloc_complete_capture, 11),
    ),
)
def test_each_phase_profile_closes_raw_identity_provenance_and_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    capture_factory: Callable[[], dict[str, Any]],
    phase_id: int,
) -> None:
    raw_path = _write_capture(tmp_path, capture_factory())
    raw_before = raw_path.read_bytes()
    identity = _fake_build_identity(tmp_path, monkeypatch, profile=profile)

    output_path, provenance_path = write_report_with_provenance(raw_path, identity)

    capture = load_capture(raw_path)
    provenance, provenance_sha256 = load_provenance(provenance_path, capture)
    assert raw_path.read_bytes() == raw_before
    assert capture.data["capture"]["mode"] == profile
    assert provenance["binding"]["capture_mode"] == profile
    assert provenance["build"]["profile"] == profile
    assert provenance["build"]["compile_definitions"] == [
        "PTO_FDWIC_SUBMIT_PMU=1",
        f"PTO_FDWIC_SUBMIT_PMU_PHASE_ID={phase_id}",
        "PTO_FDWIC_TRACE_ENABLED=0",
    ]
    document = output_path.read_text(encoding="utf-8")
    assert provenance_sha256 in document
    assert f"phase_id={phase_id}" in document
    assert profile in document


@pytest.mark.parametrize("fail_on_final_replace", (1, 2))
@pytest.mark.parametrize("old_pair_exists", (False, True))
def test_provenance_pair_publish_failure_restores_the_exact_previous_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_on_final_replace: int,
    old_pair_exists: bool,
) -> None:
    raw_path = _write_capture(tmp_path, _valid_capture())
    raw_before = raw_path.read_bytes()
    identity = _fake_build_identity(tmp_path, monkeypatch)
    output_path = tmp_path / DEFAULT_OUTPUT_NAME
    provenance_path = tmp_path / DEFAULT_PROVENANCE_NAME
    old_output = b"old-report" if old_pair_exists else None
    old_provenance = b"old-provenance" if old_pair_exists else None
    if old_pair_exists:
        output_path.write_bytes(old_output)
        provenance_path.write_bytes(old_provenance)

    real_replace = report_module.os.replace
    final_replaces = 0

    def fail_selected_final_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal final_replaces
        source_path = Path(source)
        destination_path = Path(destination)
        if ".pending." in source_path.name and destination_path in {provenance_path, output_path}:
            final_replaces += 1
            if final_replaces == fail_on_final_replace:
                raise OSError("injected paired publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(report_module.os, "replace", fail_selected_final_replace)

    with pytest.raises(OSError, match="injected paired publication failure"):
        write_report_with_provenance(raw_path, identity)

    assert raw_path.read_bytes() == raw_before
    if old_pair_exists:
        assert output_path.read_bytes() == old_output
        assert provenance_path.read_bytes() == old_provenance
    else:
        assert not output_path.exists()
        assert not provenance_path.exists()
    assert not list(tmp_path.glob(".*.pending.*.tmp"))
    assert not list(tmp_path.glob(".*.rollback.*.tmp"))


@pytest.mark.parametrize("change_on_check", (2, 3))
def test_provenance_pair_rejects_raw_change_before_or_after_final_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change_on_check: int,
) -> None:
    raw_path = _write_capture(tmp_path, _valid_capture())
    identity = _fake_build_identity(tmp_path, monkeypatch)
    real_assert_unchanged = report_module._assert_capture_raw_unchanged
    checks = 0

    def change_raw_on_selected_check(capture: report_module.SubmitPmuCapture) -> bytes:
        nonlocal checks
        checks += 1
        if checks == change_on_check:
            changed = bytearray(raw_path.read_bytes())
            changed[-2] = ord(" ") if changed[-2] != ord(" ") else ord("\n")
            raw_path.write_bytes(changed)
        return real_assert_unchanged(capture)

    monkeypatch.setattr(report_module, "_assert_capture_raw_unchanged", change_raw_on_selected_check)

    with pytest.raises(ValueError, match="raw changed after the validated capture snapshot"):
        write_report_with_provenance(raw_path, identity)

    assert not (tmp_path / DEFAULT_OUTPUT_NAME).exists()
    assert not (tmp_path / DEFAULT_PROVENANCE_NAME).exists()
    assert not list(tmp_path.glob(".*.pending.*.tmp"))
    assert not list(tmp_path.glob(".*.rollback.*.tmp"))


def test_render_without_provenance_keeps_the_legacy_raw_only_path(tmp_path: Path) -> None:
    raw_path = _write_capture(tmp_path, _valid_capture())

    document = render_report(raw_path)

    assert "真实 FDWIC Scalar Submit PMU" in document
    assert "诊断构建身份" not in document
    assert "Provenance SHA-256" not in document


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("raw_sha256", "0" * 64),
        ("capture_mode", ARG_BUILD_CAPTURE_MODE),
    ),
)
def test_provenance_rejects_raw_binding_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
) -> None:
    _output_path, provenance_path, _identity = _publish_fake_provenance(tmp_path, monkeypatch)
    capture = load_capture(tmp_path / DEFAULT_INPUT_NAME)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["binding"][field] = replacement
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance.binding does not match"):
        load_provenance(provenance_path, capture)


def _change_provenance_source_state(provenance: dict[str, Any]) -> None:
    fields = provenance["build"]["source_state"].split(":")
    fields[1] = "3" * 40
    provenance["build"]["source_state"] = ":".join(fields)


def _change_provenance_definition_hash(provenance: dict[str, Any]) -> None:
    mismatched_hash = "f" * 64
    fields = provenance["build"]["source_state"].split(":")
    fields[3] = mismatched_hash
    provenance["build"]["source_state"] = ":".join(fields)
    provenance["build"]["definitions_sha256"] = mismatched_hash


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (_change_provenance_source_state, "does not close against its profile/source state"),
        (_change_provenance_definition_hash, "compile definitions do not match definitions_sha256"),
    ),
)
def test_provenance_rejects_source_state_or_definition_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    _output_path, provenance_path, _identity = _publish_fake_provenance(tmp_path, monkeypatch)
    capture = load_capture(tmp_path / DEFAULT_INPUT_NAME)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    mutation(provenance)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_provenance(provenance_path, capture)


def test_provenance_rejects_artifact_change_after_identity_freeze_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = _write_capture(tmp_path, _valid_capture())
    raw_before = raw_path.read_bytes()
    identity = _fake_build_identity(tmp_path, monkeypatch)
    changed_artifact = dict(identity.artifacts)["aiv_combined"].path
    changed_artifact.write_bytes(changed_artifact.read_bytes() + b"-changed-after-freeze")

    with pytest.raises(ValueError, match="build artifact changed after identity freeze: aiv_combined"):
        write_report_with_provenance(raw_path, identity)

    assert raw_path.read_bytes() == raw_before
    assert not (tmp_path / DEFAULT_OUTPUT_NAME).exists()
    assert not (tmp_path / DEFAULT_PROVENANCE_NAME).exists()
    assert not (tmp_path / f"{DEFAULT_OUTPUT_NAME}.tmp").exists()
    assert not (tmp_path / f"{DEFAULT_PROVENANCE_NAME}.tmp").exists()


def _remove_required_provenance_field(provenance: dict[str, Any]) -> None:
    provenance["build"].pop("source_state_path")


def _add_unexpected_provenance_field(provenance: dict[str, Any]) -> None:
    provenance["artifacts"]["unexpected"] = {}


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (_remove_required_provenance_field, "provenance.build fields do not match"),
        (_add_unexpected_provenance_field, "provenance.artifacts fields do not match"),
    ),
)
def test_provenance_rejects_missing_or_extra_schema_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    _output_path, provenance_path, _identity = _publish_fake_provenance(tmp_path, monkeypatch)
    capture = load_capture(tmp_path / DEFAULT_INPUT_NAME)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    mutation(provenance)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_provenance(provenance_path, capture)


def test_provenance_apis_reject_wrong_output_filenames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_path = _write_capture(tmp_path, _valid_capture())
    capture = load_capture(raw_path)
    identity = _fake_build_identity(tmp_path, monkeypatch)
    wrong_provenance = tmp_path / "provenance.json"
    wrong_provenance.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=DEFAULT_PROVENANCE_NAME):
        load_provenance(wrong_provenance, capture)
    wrong_provenance.unlink()

    with pytest.raises(ValueError, match=DEFAULT_PROVENANCE_NAME):
        write_report_with_provenance(raw_path, identity, provenance_path=wrong_provenance)
    with pytest.raises(ValueError, match=DEFAULT_OUTPUT_NAME):
        write_report_with_provenance(raw_path, identity, output_path=tmp_path / "report.html")

    assert not wrong_provenance.exists()
    assert not (tmp_path / DEFAULT_PROVENANCE_NAME).exists()
    assert not (tmp_path / DEFAULT_OUTPUT_NAME).exists()


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
    capture["schema"] = "fdwic-submit-pmu-v1"


def _status_missing(capture: dict[str, Any]) -> None:
    capture["records"][0]["status"] &= ~(1 << 9)


def _submit_count_mismatch(capture: dict[str, Any]) -> None:
    capture["records"][0]["submit_count"] = 4


def _scalar_exceeds_total(capture: dict[str, Any]) -> None:
    capture["records"][0]["scalar_busy"] = capture["records"][0]["total_cycles"] + 1


def _miss_exceeds_request(capture: dict[str, Any]) -> None:
    capture["records"][0]["icache_misses"] = capture["records"][0]["icache_requests"] + 1


def _request_is_zero(capture: dict[str, Any]) -> None:
    capture["records"][0]["icache_requests"] = 0
    capture["records"][0]["icache_misses"] = 0


def _none_publishes_shadow(capture: dict[str, Any]) -> None:
    capture["records"][0]["shadow_icache_requests"] = capture["records"][0]["icache_requests"]


def _counter_reaches_threshold(capture: dict[str, Any]) -> None:
    capture["records"][0]["icache_requests"] = (1 << 30) - 1


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
        (_none_publishes_shadow, "must not publish redundant shadow counters"),
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
