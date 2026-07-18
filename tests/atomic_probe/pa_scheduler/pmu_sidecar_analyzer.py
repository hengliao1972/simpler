#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""校验并聚合 standalone PA 调度器的多轮 PMU JSON sidecar。

Host 已为单轮输出 ALL/AIC/AIV summary；本工具从每轮 96 条 worker raw 记录重新
计算同一组统计量，再聚合多个独立进程。它不修改采集文件，也不把 PMU raw total、
scalar busy 或 I-cache miss 的一阶估算冒充 Submit 墙钟时间。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_NAME = "pa_scheduler_pmu_phase_windows"
SCHEMA_VERSIONS = (3, 4)
GROUP_NAMES = ("all", "aic", "aiv")
METRIC_NAMES = (
    "total_cycles",
    "vector_busy",
    "cube_busy",
    "scalar_busy",
    "mte1_busy",
    "mte2_busy",
    "mte3_busy",
    "icache_requests",
    "icache_misses",
    "fix_busy",
)
SUBMIT_PMU_METRIC_NAMES = (
    "total_cycles",
    "vector_busy",
    "cube_busy",
    "scalar_busy",
    "mte1_busy",
    "mte2_busy",
    "icache_requests",
    "icache_misses",
    "shadow_whole_icache_requests",
    "shadow_whole_icache_misses",
    "phase_calls",
    "phase_icache_requests",
    "phase_icache_misses",
)
SUMMARY_FIELDS = ("sum", "mean", "median", "p95", "max")
SUBMIT_PMU_BUILD_VARIANT = "submit-pmu"
SUBMIT_PMU_BUILD_VARIANT_ID = 2
SUBMIT_PMU_PHASE_IDS = {"none": 0, "claim": 1, "efdrain": 2}
TASKS_PER_BATCH = 5
PHASE_STATUS_REQUIRED_MASK = 0x3CF

# schema-v4 只描述 A5 standalone submit-pmu 正式采集，不接受由 JSON 自报的
# 任意缩小拓扑。物理槽按每 die 18 AIC + 36 AIV 排列；当前 runtime 实际开放
# 32 个 AIC 与 64 个 AIV，共组成 32 组 1:2 mixed triplet。
A5_WORKERS = 96
A5_AIC_WORKERS = 32
A5_AIV_WORKERS = 64
A5_PHYSICAL_SUBCORES = 108
A5_AIC_PER_DIE = 18
A5_SUBCORES_PER_DIE = 54
A5_OWNER_BITMAP_WORDS = 4
A5_OWNER_MAGIC = 0x504D554F
A5_OWNER_VERSION = 1

# 这些字段决定两份 sidecar 是否属于同一观察配置。动态时间、capture id 和
# placement 分布不在其中；它们正是多轮运行允许自然变化的结果。
CONFIG_FINGERPRINT_FIELDS = (
    "device",
    "batches",
    "workers",
    "aic_workers",
    "aiv_workers",
    "pmu_window",
    "selectors",
    "counter_width_bits",
    "phase_timestamp_calls_present",
    "phase_record_writes",
    "profile_accumulation",
    "trace_enabled",
    "atomic_trace",
    "gate_start_stop_have_pipe_all_barriers",
    "winner_workload",
)
SUBMIT_PMU_FINGERPRINT_FIELDS = CONFIG_FINGERPRINT_FIELDS + (
    "build_variant",
    "build_variant_id",
    "compiled_phase",
    "compiled_phase_id",
    "primary_window_segments_per_record",
    "unavailable_metrics",
    "phase_boundary_observation_included",
    "phase_counter_pair_snapshot_atomic",
    "primary_counters_read_at_phase_boundaries",
    "phase_shadow_partition_exact_required",
    "phase_values_are_running_read_clear_lower_bounds",
    "cross_phase_elf_sums_valid",
)


@dataclass(frozen=True)
class Capture:
    """一份已通过 raw→summary 和采集门禁的 sidecar。"""

    path: Path
    data: dict[str, Any]
    groups: dict[str, list[dict[str, Any]]]
    fingerprint: str
    schema_version: int


@dataclass(frozen=True)
class PhasePartitionEvidence:
    """一条 raw 记录独立重算出的 shadow 分区证据。"""

    shadow_exact: bool
    shadow_bounded: bool
    request_abs_delta: int
    request_signed_delta: int
    miss_abs_delta: int
    miss_signed_delta: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, label: str) -> int:
    # bool 是 int 的子类，但 JSON true/false 不能静默成为 PMU counter。
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _signed_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nearest_rank_p95(values: Sequence[int]) -> int:
    # 与 CCEC host 使用相同的 nearest-rank 定义：ceil(0.95*N) 对应的顺序统计量。
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _metric_summary(values: Sequence[int]) -> dict[str, int | float]:
    _require(bool(values), "cannot summarize an empty metric")
    return {
        "sum": sum(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "p95": _nearest_rank_p95(values),
        "max": max(values),
    }


def _same_number(lhs: int | float, rhs: Any) -> bool:
    try:
        rhs_number = _number(rhs, "summary value")
    except ValueError:
        return False
    return math.isclose(float(lhs), rhs_number, rel_tol=1e-12, abs_tol=1e-9)


def _configuration_fingerprint(configuration: dict[str, Any], schema_version: int) -> str:
    fields = (
        SUBMIT_PMU_FINGERPRINT_FIELDS if schema_version == 4 else CONFIG_FINGERPRINT_FIELDS
    )
    selected = {field: configuration.get(field) for field in fields}
    # schema version 不是 configuration 字段，但必须进入指纹，防止 v3 历史文件与
    # v4 submit-pmu 恰好具有相同运行参数时被静默聚合。
    selected["schema_version"] = schema_version
    return json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_group_summary(
    path: Path,
    group_name: str,
    records: Sequence[dict[str, Any]],
    expected: Any,
    metric_names: Sequence[str],
    schema_version: int,
) -> None:
    _require(isinstance(expected, dict), f"{path}: summary.{group_name} must be an object")
    _require(
        expected.get("cores") == len(records),
        f"{path}: summary.{group_name}.cores does not match raw records",
    )
    _require(
        expected.get("trusted_cores") == len(records),
        f"{path}: summary.{group_name}.trusted_cores is incomplete",
    )

    for metric in metric_names:
        values = [
            _integer(record.get(metric), f"{path}: records[{index}].{metric}")
            for index, record in enumerate(records)
        ]
        actual = _metric_summary(values)
        reported = expected.get(metric)
        _require(
            isinstance(reported, dict),
            f"{path}: summary.{group_name}.{metric} must be an object",
        )
        for field in SUMMARY_FIELDS:
            _require(
                _same_number(actual[field], reported.get(field)),
                f"{path}: raw summary mismatch at {group_name}.{metric}.{field}: "
                f"raw={actual[field]!r} json={reported.get(field)!r}",
            )

    requests = sum(_integer(record["icache_requests"], "icache_requests") for record in records)
    misses = sum(_integer(record["icache_misses"], "icache_misses") for record in records)
    _require(requests > 0, f"{path}: summary.{group_name} has zero I-cache requests")
    actual_rate = misses / requests
    _require(
        _same_number(actual_rate, expected.get("icache_miss_rate")),
        f"{path}: raw summary mismatch at {group_name}.icache_miss_rate: "
        f"raw={actual_rate!r} json={expected.get('icache_miss_rate')!r}",
    )

    if schema_version == 4:
        phase_requests = sum(
            _integer(record["phase_icache_requests"], "phase_icache_requests")
            for record in records
        )
        phase_misses = sum(
            _integer(record["phase_icache_misses"], "phase_icache_misses")
            for record in records
        )
        reported_rate = expected.get("phase_observed_read_clear_ratio")
        if phase_requests == 0:
            _require(
                reported_rate is None,
                f"{path}: summary.{group_name}.phase_observed_read_clear_ratio "
                "must be null for zero requests",
            )
        else:
            actual_phase_rate = phase_misses / phase_requests
            _require(
                _same_number(actual_phase_rate, reported_rate),
                f"{path}: raw summary mismatch at "
                f"{group_name}.phase_observed_read_clear_ratio: "
                f"raw={actual_phase_rate!r} json={reported_rate!r}",
            )


def _validate_submit_pmu_configuration(path: Path, configuration: dict[str, Any]) -> tuple[str, int]:
    """校验 v4 的编译期构建身份、局部阶段和重复 selector 契约。"""

    _require(
        configuration.get("build_variant") == SUBMIT_PMU_BUILD_VARIANT,
        f"{path}: configuration.build_variant must be {SUBMIT_PMU_BUILD_VARIANT!r}",
    )
    variant_id = _integer(
        configuration.get("build_variant_id"), f"{path}: configuration.build_variant_id"
    )
    _require(
        variant_id == SUBMIT_PMU_BUILD_VARIANT_ID,
        f"{path}: unexpected submit-pmu build_variant_id {variant_id}",
    )
    phase_name = configuration.get("compiled_phase")
    _require(
        isinstance(phase_name, str) and phase_name in SUBMIT_PMU_PHASE_IDS,
        f"{path}: unsupported configuration.compiled_phase {phase_name!r}",
    )
    phase_id = _integer(
        configuration.get("compiled_phase_id"), f"{path}: configuration.compiled_phase_id"
    )
    _require(
        phase_id == SUBMIT_PMU_PHASE_IDS[phase_name],
        f"{path}: compiled phase name/id mismatch",
    )
    _require(
        configuration.get("pmu_window") == "submit-all",
        f"{path}: schema-v4 submit-pmu requires pmu_window='submit-all'",
    )
    _require(
        _integer(
            configuration.get("primary_window_segments_per_record"),
            f"{path}: configuration.primary_window_segments_per_record",
        )
        == 1,
        f"{path}: configuration.primary_window_segments_per_record must be one",
    )
    _require(
        configuration.get("unavailable_metrics") == ["mte3_busy"],
        f"{path}: configuration.unavailable_metrics must identify mte3_busy",
    )

    for field in (
        "trace_enabled",
        "trace_atomics",
        "profile_phases",
        "phase_timestamp_calls_present",
        "phase_record_writes",
        "atomic_trace",
        "profile_accumulation",
        "primary_counters_read_at_phase_boundaries",
        "cross_phase_elf_sums_valid",
    ):
        _require(configuration.get(field) is False, f"{path}: configuration.{field} is not false")
    expected_boundary_observation = phase_name != "none"
    _require(
        configuration.get("phase_boundary_observation_included")
        is expected_boundary_observation,
        f"{path}: configuration.phase_boundary_observation_included does not match the phase",
    )
    _require(
        configuration.get("phase_counter_pair_snapshot_atomic") is False,
        f"{path}: configuration.phase_counter_pair_snapshot_atomic is not false",
    )
    exact_partition_required = phase_name == "none"
    _require(
        configuration.get("phase_shadow_partition_exact_required")
        is exact_partition_required,
        f"{path}: configuration.phase_shadow_partition_exact_required does not match the phase",
    )
    running_lower_bounds = phase_name != "none"
    _require(
        configuration.get("phase_values_are_running_read_clear_lower_bounds")
        is running_lower_bounds,
        f"{path}: configuration.phase_values_are_running_read_clear_lower_bounds does not match the phase",
    )

    selectors = configuration.get("selectors")
    _require(isinstance(selectors, dict), f"{path}: configuration.selectors must be an object")
    expected_selectors = {
        "cnt6_primary_icache_request": 0x034,
        "cnt7_primary_icache_miss": 0x035,
        "cnt5_shadow_icache_miss": 0x035,
        "cnt8_shadow_icache_request": 0x034,
        "cnt9_unused": 0x000,
    }
    for field, expected in expected_selectors.items():
        _require(
            _integer(selectors.get(field), f"{path}: configuration.selectors.{field}") == expected,
            f"{path}: configuration.selectors.{field} is not 0x{expected:03x}",
        )
    return phase_name, phase_id


def _is_aic_physical_slot(physical_id: int) -> bool:
    return (
        physical_id < A5_PHYSICAL_SUBCORES
        and physical_id % A5_SUBCORES_PER_DIE < A5_AIC_PER_DIE
    )


def _expected_logical_triplet(worker_id: int) -> tuple[str, int, int]:
    """按 mixed launch ABI 返回 worker 的 role/block/lane。"""

    if worker_id < A5_AIC_WORKERS:
        return "aic", worker_id, 0
    vector_id = worker_id - A5_AIC_WORKERS
    return "aiv", vector_id // 2, 1 + vector_id % 2


def _physical_aiv_pair(aic_physical_id: int) -> tuple[int, int]:
    """返回一个物理 AIC 槽对应的两个 AIV 槽。"""

    die_base = (aic_physical_id // A5_SUBCORES_PER_DIE) * A5_SUBCORES_PER_DIE
    local_aic = aic_physical_id % A5_SUBCORES_PER_DIE
    first_aiv = die_base + A5_AIC_PER_DIE + local_aic * 2
    return first_aiv, first_aiv + 1


def _require_owner_role_counts(
    path: Path, label: str, value: Any, expected: tuple[int, int, int]
) -> None:
    _require(isinstance(value, dict), f"{path}: owner.{label} must be an object")
    for field, expected_value in zip(("total", "aic", "aiv"), expected):
        actual = _integer(value.get(field), f"{path}: owner.{label}.{field}")
        _require(
            actual == expected_value,
            f"{path}: owner.{label}.{field} must be {expected_value}",
        )


def _validate_submit_pmu_owner(
    path: Path, owner: Any, capture: dict[str, Any]
) -> set[int]:
    """独立重验 configure 后、restore 前保存的 PMU owner 快照。"""

    _require(isinstance(owner, dict), f"{path}: schema-v4 owner must be an object")
    _require(owner.get("mode") == "main_aicpu_path_a", f"{path}: unexpected owner.mode")
    _require(
        owner.get("snapshot_phase") == "after_configure_before_restore",
        f"{path}: unexpected owner.snapshot_phase",
    )
    _require(
        _integer(owner.get("control_magic"), f"{path}: owner.control_magic")
        == A5_OWNER_MAGIC,
        f"{path}: owner.control_magic mismatch",
    )
    _require(
        _integer(owner.get("control_version"), f"{path}: owner.control_version")
        == A5_OWNER_VERSION,
        f"{path}: owner.control_version mismatch",
    )
    _require(
        _integer(owner.get("configure_status"), f"{path}: owner.configure_status") == 0,
        f"{path}: owner.configure_status is not success",
    )
    _require(
        _integer(owner.get("configured_flag"), f"{path}: owner.configured_flag") == 1,
        f"{path}: owner.configured_flag is not one",
    )

    topology = (A5_WORKERS, A5_AIC_WORKERS, A5_AIV_WORKERS)
    for label in ("expected", "active", "discovered"):
        _require_owner_role_counts(path, label, owner.get(label), topology)
    _require(
        _integer(owner.get("physical_slots_scanned"), f"{path}: owner.physical_slots_scanned")
        == A5_PHYSICAL_SUBCORES,
        f"{path}: owner.physical_slots_scanned mismatch",
    )
    _require(
        _integer(
            owner.get("skipped_physical_slots"), f"{path}: owner.skipped_physical_slots"
        )
        == A5_PHYSICAL_SUBCORES - A5_WORKERS,
        f"{path}: owner.skipped_physical_slots mismatch",
    )
    _require(
        owner.get("configured_bitmap_word_order")
        == "least_significant_physical_ids_first",
        f"{path}: unexpected owner.configured_bitmap_word_order",
    )

    words = owner.get("configured_bitmap_words")
    _require(
        isinstance(words, list) and len(words) == A5_OWNER_BITMAP_WORDS,
        f"{path}: owner.configured_bitmap_words must contain four words",
    )
    bitmap_words: list[int] = []
    for index, value in enumerate(words):
        word = _integer(value, f"{path}: owner.configured_bitmap_words[{index}]")
        _require(word <= 0xFFFFFFFF, f"{path}: owner bitmap word exceeds 32 bits")
        bitmap_words.append(word)

    configured_ids = {
        physical_id
        for physical_id in range(A5_OWNER_BITMAP_WORDS * 32)
        if bitmap_words[physical_id // 32] & (1 << (physical_id % 32))
    }
    _require(
        all(physical_id < A5_PHYSICAL_SUBCORES for physical_id in configured_ids),
        f"{path}: owner bitmap sets a bit outside the 108 physical slots",
    )
    configured_count = _integer(
        owner.get("configured_bitmap_count"), f"{path}: owner.configured_bitmap_count"
    )
    _require(
        configured_count == len(configured_ids) == A5_WORKERS,
        f"{path}: owner bitmap count is not exactly 96",
    )
    configured_aic = {physical_id for physical_id in configured_ids if _is_aic_physical_slot(physical_id)}
    configured_aiv = configured_ids - configured_aic
    _require(
        len(configured_aic) == A5_AIC_WORKERS and len(configured_aiv) == A5_AIV_WORKERS,
        f"{path}: owner bitmap is not a 32 AIC / 64 AIV set",
    )

    complete_triplets = sum(
        all(aiv_id in configured_aiv for aiv_id in _physical_aiv_pair(aic_id))
        for aic_id in configured_aic
    )
    broken_triplets = len(configured_aic) - complete_triplets
    _require(
        _integer(
            owner.get("configured_complete_mixed_triplets"),
            f"{path}: owner.configured_complete_mixed_triplets",
        )
        == complete_triplets
        == A5_AIC_WORKERS,
        f"{path}: owner bitmap does not contain 32 complete mixed triplets",
    )
    _require(
        _integer(
            owner.get("expected_complete_mixed_triplets"),
            f"{path}: owner.expected_complete_mixed_triplets",
        )
        == A5_AIC_WORKERS,
        f"{path}: owner.expected_complete_mixed_triplets mismatch",
    )
    _require(
        _integer(
            owner.get("configured_broken_mixed_triplets"),
            f"{path}: owner.configured_broken_mixed_triplets",
        )
        == broken_triplets
        == 0,
        f"{path}: owner bitmap contains a broken mixed triplet",
    )
    _require(owner.get("restore_passed") is True, f"{path}: owner.restore_passed is not true")
    _require(
        owner.get("restore_passed") is capture.get("owner_restore_passed"),
        f"{path}: owner and capture restore results disagree",
    )
    return configured_ids


def _validate_submit_pmu_record(
    path: Path,
    index: int,
    record: dict[str, Any],
    phase_name: str,
    phase_id: int,
    batches: int,
) -> PhasePartitionEvidence:
    """重验 raw phase 分区；claim 只形成 read-to-clear 的上下界。"""

    prefix = f"{path}: records[{index}]"
    _require(
        _integer(record.get("build_variant_id"), f"{prefix}.build_variant_id")
        == SUBMIT_PMU_BUILD_VARIANT_ID,
        f"{prefix} build_variant_id mismatch",
    )
    _require(
        _integer(record.get("compiled_phase_id"), f"{prefix}.compiled_phase_id") == phase_id,
        f"{prefix} compiled_phase_id mismatch",
    )
    phase_status = _integer(record.get("phase_status"), f"{prefix}.phase_status")
    _require(
        (phase_status & PHASE_STATUS_REQUIRED_MASK) == PHASE_STATUS_REQUIRED_MASK,
        f"{prefix} phase_status is incomplete",
    )

    primary_requests = _integer(record.get("icache_requests"), f"{prefix}.icache_requests")
    primary_misses = _integer(record.get("icache_misses"), f"{prefix}.icache_misses")
    shadow_requests = _integer(
        record.get("shadow_whole_icache_requests"), f"{prefix}.shadow_whole_icache_requests"
    )
    shadow_misses = _integer(
        record.get("shadow_whole_icache_misses"), f"{prefix}.shadow_whole_icache_misses"
    )
    shadow_exact = shadow_requests == primary_requests and shadow_misses == primary_misses
    shadow_bounded = shadow_requests <= primary_requests and shadow_misses <= primary_misses
    _require(
        record.get("shadow_matches_primary") is shadow_exact,
        f"{prefix}.shadow_matches_primary disagrees with raw counters",
    )
    _require(
        record.get("shadow_not_greater_than_primary") is shadow_bounded,
        f"{prefix}.shadow_not_greater_than_primary disagrees with raw counters",
    )
    _require(shadow_bounded, f"{prefix} shadow whole exceeds the authoritative primary whole")
    if phase_name == "none":
        _require(shadow_exact, f"{prefix} phase=none requires shadow whole to equal primary")

    request_loss = primary_requests - shadow_requests
    miss_loss = primary_misses - shadow_misses
    _require(
        _integer(record.get("shadow_request_loss"), f"{prefix}.shadow_request_loss")
        == request_loss,
        f"{prefix}.shadow_request_loss disagrees with raw counters",
    )
    _require(
        _integer(record.get("shadow_miss_loss"), f"{prefix}.shadow_miss_loss") == miss_loss,
        f"{prefix}.shadow_miss_loss disagrees with raw counters",
    )

    calls = _integer(record.get("phase_calls"), f"{prefix}.phase_calls")
    begin_reads = _integer(record.get("phase_begin_reads"), f"{prefix}.phase_begin_reads")
    end_reads = _integer(record.get("phase_end_reads"), f"{prefix}.phase_end_reads")
    _require(
        record.get("phase_boundaries_balanced") is True,
        f"{prefix}.phase_boundaries_balanced is not true",
    )
    _require(
        begin_reads == calls and end_reads == calls,
        f"{prefix} phase begin/end boundaries do not match calls",
    )
    primary_segments = _integer(
        record.get("primary_window_segments"), f"{prefix}.primary_window_segments"
    )
    shadow_segments = _integer(
        record.get("shadow_read_segments"), f"{prefix}.shadow_read_segments"
    )
    _require(primary_segments == 1, f"{prefix} primary_window_segments must be one")
    _require(
        shadow_segments == 2 * calls + 1,
        f"{prefix} shadow_read_segments does not match phase boundaries plus tail",
    )

    phase_requests = _integer(
        record.get("phase_icache_requests"), f"{prefix}.phase_icache_requests"
    )
    phase_misses = _integer(
        record.get("phase_icache_misses"), f"{prefix}.phase_icache_misses"
    )
    phase_request_upper = _integer(
        record.get("phase_icache_requests_upper_bound"),
        f"{prefix}.phase_icache_requests_upper_bound",
    )
    phase_miss_upper = _integer(
        record.get("phase_icache_misses_upper_bound"),
        f"{prefix}.phase_icache_misses_upper_bound",
    )
    _require(
        phase_requests <= shadow_requests and phase_misses <= shadow_misses,
        f"{prefix} phase counters exceed the Submit shadow whole",
    )
    _require(
        phase_request_upper == phase_requests + request_loss,
        f"{prefix}.phase_icache_requests_upper_bound is not lower plus shadow loss",
    )
    _require(
        phase_miss_upper == phase_misses + miss_loss,
        f"{prefix}.phase_icache_misses_upper_bound is not lower plus shadow loss",
    )
    _require(
        phase_request_upper <= primary_requests and phase_miss_upper <= primary_misses,
        f"{prefix} phase upper bound exceeds the authoritative primary whole",
    )

    expected_calls = (
        0 if phase_name == "none"
        else batches * TASKS_PER_BATCH if phase_name in ("claim", "efdrain")
        else -1
    )
    _require(calls == expected_calls, f"{prefix} phase_calls does not match the phase contract")
    if phase_name == "none":
        _require(
            phase_requests == 0 and phase_misses == 0,
            f"{prefix} phase=none must have zero phase counters",
        )
        _require(
            phase_request_upper == phase_requests and phase_miss_upper == phase_misses,
            f"{prefix} phase=none lower and upper bounds must be equal",
        )

    return PhasePartitionEvidence(
        shadow_exact=shadow_exact,
        shadow_bounded=shadow_bounded,
        request_abs_delta=abs(shadow_requests - primary_requests),
        request_signed_delta=shadow_requests - primary_requests,
        miss_abs_delta=abs(shadow_misses - primary_misses),
        miss_signed_delta=shadow_misses - primary_misses,
    )


def load_capture(path: Path) -> Capture:
    """读取并完整校验一份历史 v3 或 submit-pmu v4 sidecar。"""

    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    _require(isinstance(data, dict), f"{path}: capture root must be an object")
    schema = data.get("schema")
    _require(
        isinstance(schema, dict) and schema.get("name") == SCHEMA_NAME,
        f"{path}: expected schema name {SCHEMA_NAME}",
    )
    schema_version = _integer(schema.get("version"), f"{path}: schema.version")
    _require(
        schema_version in SCHEMA_VERSIONS,
        f"{path}: expected {SCHEMA_NAME} schema v3 or v4",
    )

    capture = data.get("capture")
    configuration = data.get("configuration")
    validation = data.get("validation")
    records = data.get("records")
    summary = data.get("summary")
    _require(isinstance(capture, dict), f"{path}: capture must be an object")
    _require(isinstance(configuration, dict), f"{path}: configuration must be an object")
    _require(isinstance(validation, dict), f"{path}: validation must be an object")
    _require(isinstance(records, list), f"{path}: records must be an array")
    _require(isinstance(summary, dict), f"{path}: summary must be an object")

    workers = _integer(configuration.get("workers"), f"{path}: configuration.workers")
    aic_workers = _integer(configuration.get("aic_workers"), f"{path}: configuration.aic_workers")
    aiv_workers = _integer(configuration.get("aiv_workers"), f"{path}: configuration.aiv_workers")
    batches = _integer(configuration.get("batches"), f"{path}: configuration.batches")
    _require(workers == aic_workers + aiv_workers, f"{path}: worker role counts do not add up")
    _require(len(records) == workers, f"{path}: record count does not match configuration.workers")
    phase_name: str | None = None
    phase_id: int | None = None
    if schema_version == 4:
        _require(
            (workers, aic_workers, aiv_workers)
            == (A5_WORKERS, A5_AIC_WORKERS, A5_AIV_WORKERS),
            f"{path}: schema-v4 requires the fixed 96/32/64 A5 topology",
        )
        phase_name, phase_id = _validate_submit_pmu_configuration(path, configuration)

    # JSON 只有在运行、PMU、owner Restore 和 runtime cleanup 全部成功后才应发布。
    # 分析器仍逐项重验，防止手工复制或未来 schema 退化绕过发布门禁。
    required_true = (
        (capture, "accepted"),
        (capture, "published_after_runtime_cleanup"),
        (capture, "runtime_cleanup_passed"),
        (capture, "owner_restore_passed"),
        (validation, "semantic_passed"),
        (validation, "pmu_passed"),
        (validation, "icache_measurement_valid"),
        (validation, "icache_miss_le_request"),
        (validation, "counter_below_risk_threshold"),
    )
    for owner, field in required_true:
        _require(owner.get(field) is True, f"{path}: {field} is not true")
    configured_physical_ids: set[int] | None = None
    if schema_version == 4:
        _require(
            validation.get("phase_measurement_valid") is True,
            f"{path}: phase_measurement_valid is not true",
        )
        configured_physical_ids = _validate_submit_pmu_owner(path, data.get("owner"), capture)

    expected_records = (
        "trusted_records",
        "unique_physical_core_ids",
        "owner_bitmap_member_records",
        "exact_worker_slot_records",
        "physical_role_match_records",
        "window_started_records",
        "window_stopped_records",
    )
    for field in expected_records:
        _require(
            validation.get(field) == workers,
            f"{path}: validation.{field} is incomplete",
        )
    if schema_version == 4:
        expected_host_counts = {
            "expected_records": A5_WORKERS,
            "expected_unique_core_ids": A5_WORKERS,
            "expected_owner_bitmap_member_records": A5_WORKERS,
            "expected_exact_worker_slot_records": A5_WORKERS,
            "expected_physical_role_match_records": A5_WORKERS,
            "mixed_triplet_matches": A5_AIC_WORKERS,
            "expected_mixed_triplet_matches": A5_AIC_WORKERS,
            "expected_window_records": A5_WORKERS,
        }
        for field, expected in expected_host_counts.items():
            _require(
                _integer(validation.get(field), f"{path}: validation.{field}") == expected,
                f"{path}: validation.{field} does not match the A5 topology",
            )
        for field in (
            "build_variant_match_records",
            "phase_id_match_records",
            "phase_status_trusted_records",
            "phase_boundary_match_records",
            "phase_call_shape_match_records",
        ):
            _require(
                validation.get(field) == workers,
                f"{path}: validation.{field} is incomplete",
            )
        expected_phase_calls = (
            0 if phase_name == "none" else workers * batches * TASKS_PER_BATCH
        )
        _require(
            validation.get("phase_calls") == expected_phase_calls,
            f"{path}: validation.phase_calls does not match the compiled phase contract",
        )

    groups: dict[str, list[dict[str, Any]]] = {
        "all": records,
        "aic": [record for record in records if record.get("role") == "aic"],
        "aiv": [record for record in records if record.get("role") == "aiv"],
    }

    worker_ids: set[int] = set()
    physical_core_ids: set[int] = set()
    ordered_physical_ids: list[int] = []
    phase_partition_evidence: list[PhasePartitionEvidence] = []
    for index, record in enumerate(records):
        _require(isinstance(record, dict), f"{path}: records[{index}] must be an object")
        worker_id = _integer(record.get("worker_id"), f"{path}: records[{index}].worker_id")
        physical_id = _integer(
            record.get("physical_core_id"), f"{path}: records[{index}].physical_core_id"
        )
        _require(worker_id not in worker_ids, f"{path}: duplicate worker_id {worker_id}")
        _require(
            physical_id not in physical_core_ids,
            f"{path}: duplicate physical_core_id {physical_id}",
        )
        worker_ids.add(worker_id)
        physical_core_ids.add(physical_id)
        ordered_physical_ids.append(physical_id)
        if schema_version == 4:
            _require(
                worker_id == index,
                f"{path}: records[{index}].worker_id does not match its exact worker slot",
            )
            expected_role, expected_block, expected_lane = _expected_logical_triplet(index)
            _require(
                record.get("role") == expected_role,
                f"{path}: records[{index}].role does not match the logical worker topology",
            )
            _require(
                _integer(record.get("block_id"), f"{path}: records[{index}].block_id")
                == expected_block,
                f"{path}: records[{index}].block_id does not match its mixed block",
            )
            _require(
                _integer(record.get("lane"), f"{path}: records[{index}].lane") == expected_lane,
                f"{path}: records[{index}].lane does not match its mixed lane",
            )
            _require(
                physical_id < A5_PHYSICAL_SUBCORES,
                f"{path}: records[{index}].physical_core_id is outside the 108-slot topology",
            )
            _require(
                _is_aic_physical_slot(physical_id) == (expected_role == "aic"),
                f"{path}: records[{index}] logical role does not match its physical slot",
            )
            _require(
                record.get("physical_core_id_valid") is True,
                f"{path}: records[{index}].physical_core_id_valid is not true",
            )
            assert configured_physical_ids is not None
            _require(
                physical_id in configured_physical_ids,
                f"{path}: records[{index}].physical_core_id is absent from the owner bitmap",
            )
        for field in (
            "trusted",
            "selectors_match",
            "owner_bitmap_member",
            "worker_slot_exact",
            "physical_role_matches",
            "window_started",
            "window_stopped",
        ):
            _require(record.get(field) is True, f"{path}: records[{index}].{field} is not true")
        requests = _integer(
            record.get("icache_requests"), f"{path}: records[{index}].icache_requests"
        )
        misses = _integer(
            record.get("icache_misses"), f"{path}: records[{index}].icache_misses"
        )
        _require(misses <= requests, f"{path}: records[{index}] has miss > request")
        if schema_version == 4:
            # 上面的 configuration 校验已保证二者不是 None；显式 assert 只帮助
            # 类型收窄，不替代任何 JSON 运行时门禁。
            assert phase_name is not None and phase_id is not None
            phase_partition_evidence.append(
                _validate_submit_pmu_record(path, index, record, phase_name, phase_id, batches)
            )

    if schema_version == 4:
        assert configured_physical_ids is not None
        _require(
            physical_core_ids == configured_physical_ids,
            f"{path}: worker physical-core set does not exactly equal the owner bitmap",
        )
        for block in range(A5_AIC_WORKERS):
            aic_id = ordered_physical_ids[block]
            expected_aiv0, expected_aiv1 = _physical_aiv_pair(aic_id)
            actual_aiv0 = ordered_physical_ids[A5_AIC_WORKERS + block * 2]
            actual_aiv1 = ordered_physical_ids[A5_AIC_WORKERS + block * 2 + 1]
            _require(
                (actual_aiv0, actual_aiv1) == (expected_aiv0, expected_aiv1),
                f"{path}: mixed block {block} does not map to one physical AIC and its two AIVs",
            )

        shadow_exact_records = sum(item.shadow_exact for item in phase_partition_evidence)
        shadow_bounded_records = sum(item.shadow_bounded for item in phase_partition_evidence)
        _require(
            shadow_bounded_records == A5_WORKERS,
            f"{path}: not all shadow partitions are bounded by primary",
        )
        assert phase_name is not None
        if phase_name == "none":
            _require(
                shadow_exact_records == A5_WORKERS,
                f"{path}: phase=none requires all shadow partitions to be exact",
            )
        _require(
            _integer(
                validation.get("shadow_primary_match_records"),
                f"{path}: validation.shadow_primary_match_records",
            )
            == shadow_exact_records,
            f"{path}: validation.shadow_primary_match_records disagrees with raw records",
        )
        _require(
            _integer(
                validation.get("shadow_primary_bounded_records"),
                f"{path}: validation.shadow_primary_bounded_records",
            )
            == shadow_bounded_records,
            f"{path}: validation.shadow_primary_bounded_records disagrees with raw records",
        )
        unsigned_deltas = {
            "shadow_request_abs_delta_sum": sum(
                item.request_abs_delta for item in phase_partition_evidence
            ),
            "shadow_request_abs_delta_max": max(
                item.request_abs_delta for item in phase_partition_evidence
            ),
            "shadow_miss_abs_delta_sum": sum(
                item.miss_abs_delta for item in phase_partition_evidence
            ),
            "shadow_miss_abs_delta_max": max(
                item.miss_abs_delta for item in phase_partition_evidence
            ),
        }
        signed_deltas = {
            "shadow_request_signed_delta_sum": sum(
                item.request_signed_delta for item in phase_partition_evidence
            ),
            "shadow_miss_signed_delta_sum": sum(
                item.miss_signed_delta for item in phase_partition_evidence
            ),
        }
        for field, expected in unsigned_deltas.items():
            _require(
                _integer(validation.get(field), f"{path}: validation.{field}") == expected,
                f"{path}: validation.{field} disagrees with raw records",
            )
        for field, expected in signed_deltas.items():
            _require(
                _signed_integer(validation.get(field), f"{path}: validation.{field}")
                == expected,
                f"{path}: validation.{field} disagrees with raw records",
            )

    _require(len(groups["aic"]) == aic_workers, f"{path}: AIC raw record count mismatch")
    _require(len(groups["aiv"]) == aiv_workers, f"{path}: AIV raw record count mismatch")

    metric_names = SUBMIT_PMU_METRIC_NAMES if schema_version == 4 else METRIC_NAMES
    for group_name in GROUP_NAMES:
        _validate_group_summary(
            path,
            group_name,
            groups[group_name],
            summary.get(group_name),
            metric_names,
            schema_version,
        )

    return Capture(
        path,
        data,
        groups,
        _configuration_fingerprint(configuration, schema_version),
        schema_version,
    )


def _median(values: Iterable[int | float]) -> int | float:
    return statistics.median(list(values))


def analyze(paths: Sequence[Path], miss_penalty_ns: float = 90.0) -> dict[str, Any]:
    """校验同配置多轮 sidecar，并返回可序列化的跨轮汇总。"""

    _require(bool(paths), "at least one PMU JSON path is required")
    _require(math.isfinite(miss_penalty_ns) and miss_penalty_ns > 0, "miss penalty must be positive")
    captures = [load_capture(path) for path in paths]
    schema_version = captures[0].schema_version
    fingerprint = captures[0].fingerprint
    for capture in captures[1:]:
        _require(
            capture.schema_version == schema_version,
            f"{capture.path}: input schema differs from {captures[0].path}",
        )
        _require(
            capture.fingerprint == fingerprint,
            f"{capture.path}: observation configuration differs from {captures[0].path}",
        )

    per_run: list[dict[str, Any]] = []
    for capture in captures:
        configuration = capture.data["configuration"]
        summary = capture.data["summary"]
        row: dict[str, Any] = {
            "path": str(capture.path),
            "capture_id": capture.data["capture"].get("capture_id"),
            "submit_span_us": _number(
                configuration.get("submit_span_us"),
                f"{capture.path}: configuration.submit_span_us",
            ),
            "groups": {},
        }
        for group_name in GROUP_NAMES:
            group = summary[group_name]
            cores = _integer(group.get("cores"), f"summary.{group_name}.cores")
            requests = _integer(group["icache_requests"].get("sum"), "icache request sum")
            misses = _integer(group["icache_misses"].get("sum"), "icache miss sum")
            row["groups"][group_name] = {
                "cores": cores,
                "icache_requests_sum": requests,
                "icache_misses_sum": misses,
                "icache_miss_rate": misses / requests,
                "icache_requests_per_core": requests / cores,
                "icache_misses_per_core": misses / cores,
                "icache_misses_per_core_median": group["icache_misses"]["median"],
                "icache_misses_per_core_p95": group["icache_misses"]["p95"],
                "first_order_miss_core_equivalent_us": misses * miss_penalty_ns / 1000.0,
                "first_order_miss_per_core_us": misses * miss_penalty_ns / cores / 1000.0,
                "scalar_busy_sum": group["scalar_busy"]["sum"],
                "total_cycles_sum": group["total_cycles"]["sum"],
            }
            if schema_version == 4:
                raw_group = capture.groups[group_name]
                phase_requests = _integer(
                    group["phase_icache_requests"].get("sum"), "phase I-cache request sum"
                )
                phase_misses = _integer(
                    group["phase_icache_misses"].get("sum"), "phase I-cache miss sum"
                )
                phase_calls = _integer(group["phase_calls"].get("sum"), "phase call sum")
                phase_request_uppers = [
                    _integer(
                        record.get("phase_icache_requests_upper_bound"),
                        "phase I-cache request upper bound",
                    )
                    for record in raw_group
                ]
                phase_miss_uppers = [
                    _integer(
                        record.get("phase_icache_misses_upper_bound"),
                        "phase I-cache miss upper bound",
                    )
                    for record in raw_group
                ]
                request_losses = [
                    _integer(record.get("shadow_request_loss"), "shadow request loss")
                    for record in raw_group
                ]
                miss_losses = [
                    _integer(record.get("shadow_miss_loss"), "shadow miss loss")
                    for record in raw_group
                ]
                phase_request_upper_summary = _metric_summary(phase_request_uppers)
                phase_miss_upper_summary = _metric_summary(phase_miss_uppers)
                phase_request_lower_summary = _metric_summary(
                    [
                        _integer(
                            record.get("phase_icache_requests"),
                            "phase I-cache request lower bound",
                        )
                        for record in raw_group
                    ]
                )
                phase_miss_lower_summary = _metric_summary(
                    [
                        _integer(
                            record.get("phase_icache_misses"),
                            "phase I-cache miss lower bound",
                        )
                        for record in raw_group
                    ]
                )
                request_loss_summary = _metric_summary(request_losses)
                miss_loss_summary = _metric_summary(miss_losses)
                row["groups"][group_name].update(
                    {
                        "phase_calls_sum": phase_calls,
                        "phase_calls_per_core": phase_calls / cores,
                        "phase_icache_requests_lower_bound_sum": phase_requests,
                        "phase_icache_requests_upper_bound_sum": sum(phase_request_uppers),
                        "phase_icache_misses_lower_bound_sum": phase_misses,
                        "phase_icache_misses_upper_bound_sum": sum(phase_miss_uppers),
                        "phase_icache_requests_lower_bound_per_core": phase_requests / cores,
                        "phase_icache_requests_upper_bound_per_core": sum(phase_request_uppers)
                        / cores,
                        "phase_icache_misses_lower_bound_per_core": phase_misses / cores,
                        "phase_icache_misses_upper_bound_per_core": sum(phase_miss_uppers) / cores,
                        "phase_icache_requests_lower_bound_per_core_median":
                            phase_request_lower_summary["median"],
                        "phase_icache_requests_lower_bound_per_core_p95":
                            phase_request_lower_summary["p95"],
                        "phase_icache_requests_upper_bound_per_core_median":
                            phase_request_upper_summary["median"],
                        "phase_icache_requests_upper_bound_per_core_p95":
                            phase_request_upper_summary["p95"],
                        "phase_icache_misses_lower_bound_per_core_median":
                            phase_miss_lower_summary["median"],
                        "phase_icache_misses_lower_bound_per_core_p95":
                            phase_miss_lower_summary["p95"],
                        "phase_icache_misses_upper_bound_per_core_median":
                            phase_miss_upper_summary["median"],
                        "phase_icache_misses_upper_bound_per_core_p95":
                            phase_miss_upper_summary["p95"],
                        "shadow_request_loss_sum": sum(request_losses),
                        "shadow_request_loss_per_core": sum(request_losses) / cores,
                        "shadow_request_loss_per_core_median": request_loss_summary["median"],
                        "shadow_request_loss_per_core_p95": request_loss_summary["p95"],
                        "shadow_miss_loss_sum": sum(miss_losses),
                        "shadow_miss_loss_per_core": sum(miss_losses) / cores,
                        "shadow_miss_loss_per_core_median": miss_loss_summary["median"],
                        "shadow_miss_loss_per_core_p95": miss_loss_summary["p95"],
                        "phase_observed_read_clear_ratio": (
                            None if phase_requests == 0 else phase_misses / phase_requests
                        ),
                        "phase_icache_request_lower_bound_share_of_submit":
                            phase_requests / requests,
                        "phase_icache_miss_lower_bound_share_of_submit":
                            phase_misses / misses if misses != 0 else None,
                    }
                )
        per_run.append(row)

    aggregate: dict[str, Any] = {
        "runs": len(per_run),
        "submit_span_us": {
            "median": _median(row["submit_span_us"] for row in per_run),
            "min": min(row["submit_span_us"] for row in per_run),
            "max": max(row["submit_span_us"] for row in per_run),
        },
        "groups": {},
    }
    aggregate_fields = (
        "icache_requests_sum",
        "icache_misses_sum",
        "icache_miss_rate",
        "icache_requests_per_core",
        "icache_misses_per_core",
        "icache_misses_per_core_median",
        "icache_misses_per_core_p95",
        "first_order_miss_core_equivalent_us",
        "first_order_miss_per_core_us",
        "scalar_busy_sum",
        "total_cycles_sum",
    )
    if schema_version == 4:
        aggregate_fields += (
            "phase_calls_sum",
            "phase_calls_per_core",
            "phase_icache_requests_lower_bound_sum",
            "phase_icache_requests_upper_bound_sum",
            "phase_icache_misses_lower_bound_sum",
            "phase_icache_misses_upper_bound_sum",
            "phase_icache_requests_lower_bound_per_core",
            "phase_icache_requests_upper_bound_per_core",
            "phase_icache_misses_lower_bound_per_core",
            "phase_icache_misses_upper_bound_per_core",
            "phase_icache_requests_lower_bound_per_core_median",
            "phase_icache_requests_lower_bound_per_core_p95",
            "phase_icache_requests_upper_bound_per_core_median",
            "phase_icache_requests_upper_bound_per_core_p95",
            "phase_icache_misses_lower_bound_per_core_median",
            "phase_icache_misses_lower_bound_per_core_p95",
            "phase_icache_misses_upper_bound_per_core_median",
            "phase_icache_misses_upper_bound_per_core_p95",
            "shadow_request_loss_sum",
            "shadow_request_loss_per_core",
            "shadow_request_loss_per_core_median",
            "shadow_request_loss_per_core_p95",
            "shadow_miss_loss_sum",
            "shadow_miss_loss_per_core",
            "shadow_miss_loss_per_core_median",
            "shadow_miss_loss_per_core_p95",
            "phase_observed_read_clear_ratio",
            "phase_icache_request_lower_bound_share_of_submit",
            "phase_icache_miss_lower_bound_share_of_submit",
        )
    for group_name in GROUP_NAMES:
        aggregate["groups"][group_name] = {}
        for field in aggregate_fields:
            values = [
                row["groups"][group_name][field]
                for row in per_run
                if row["groups"][group_name][field] is not None
            ]
            aggregate["groups"][group_name][field] = (
                {
                    "median": _median(values),
                    "min": min(values),
                    "max": max(values),
                }
                if values
                else {"median": None, "min": None, "max": None}
            )

    configuration = captures[0].data["configuration"]
    fingerprint_fields = (
        SUBMIT_PMU_FINGERPRINT_FIELDS if schema_version == 4 else CONFIG_FINGERPRINT_FIELDS
    )
    result: dict[str, Any] = {
        "schema": {
            "name": "pa_scheduler_pmu_multi_run_summary",
            "version": 2 if schema_version == 4 else 1,
        },
        "input_schema": {"name": SCHEMA_NAME, "version": schema_version},
        "configuration": {
            field: configuration.get(field) for field in fingerprint_fields
        },
        "estimation": {
            "icache_miss_penalty_ns": miss_penalty_ns,
            "meaning": "controlled_cold_warm_first_order_core_equivalent",
            "not_wall_time": True,
            "not_additive_stall_time": True,
        },
        "actual_exposed_loss": {
            "status": "requires_same_semantics_paired_ab",
            "reason": "A5 submit-all has no verified I-cache stall-cycle counter",
            "required_observations": ("delta_submit_span_us", "delta_icache_misses_per_core"),
        },
        "validation": {
            "raw_to_summary_all_fields_match": True,
            "all_inputs_accepted_and_restored": True,
            "same_observation_configuration": True,
        },
        "per_run": per_run,
        "aggregate": aggregate,
    }
    if schema_version == 4:
        phase_enabled = configuration["compiled_phase"] != "none"
        result["phase_observation"] = {
            "compiled_phase": configuration["compiled_phase"],
            "compiled_phase_id": configuration["compiled_phase_id"],
            "enabled": phase_enabled,
            "primary_whole_authoritative": True,
            "phase_share_is_same_elf_submit_only": True,
            "phase_boundary_observer_perturbed": phase_enabled,
            "phase_counter_pair_snapshot_atomic": False,
            "phase_shadow_partition_exact_required": configuration[
                "phase_shadow_partition_exact_required"
            ],
            "phase_values_are_running_read_clear_lower_bounds": configuration[
                "phase_values_are_running_read_clear_lower_bounds"
            ],
            "phase_value_semantics": (
                "disabled_zero"
                if not phase_enabled
                else "running_read_clear_lower_to_loss_adjusted_upper_bound"
            ),
            "cross_phase_elf_sums_valid": False,
        }
    return result


def _print_text(result: dict[str, Any]) -> None:
    configuration = result["configuration"]
    workload = configuration.get("winner_workload") or {}
    counts = workload.get("counts") or {}
    build_phase = ""
    if result["input_schema"]["version"] == 4:
        build_phase = (
            f" build={configuration.get('build_variant')}"
            f" phase={configuration.get('compiled_phase')}"
        )
    print(
        "[CONFIG] "
        f"window={configuration.get('pmu_window')} batches={configuration.get('batches')} "
        f"workers={configuration.get('workers')} workload={workload.get('mode')} "
        f"counts={counts.get('qk')},{counts.get('sf')},{counts.get('pv')},{counts.get('up')}"
        f"{build_phase}"
    )
    print("[VALIDATION] raw_to_summary=PASS accepted_restore=PASS same_configuration=PASS")
    print(
        "run  submit_us  all_miss/core  all_rate  "
        "aic_miss/core  aic_rate  aiv_miss/core  aiv_rate"
    )
    for index, row in enumerate(result["per_run"], start=1):
        all_group = row["groups"]["all"]
        aic = row["groups"]["aic"]
        aiv = row["groups"]["aiv"]
        print(
            f"{index:>3}  {row['submit_span_us']:>9.3f}  "
            f"{all_group['icache_misses_per_core']:>13.3f}  "
            f"{all_group['icache_miss_rate'] * 100:>7.4f}%  "
            f"{aic['icache_misses_per_core']:>13.3f}  {aic['icache_miss_rate'] * 100:>7.4f}%  "
            f"{aiv['icache_misses_per_core']:>14.3f}  {aiv['icache_miss_rate'] * 100:>7.4f}%"
        )

    aggregate = result["aggregate"]
    all_group = aggregate["groups"]["all"]
    aic = aggregate["groups"]["aic"]
    aiv = aggregate["groups"]["aiv"]
    print(
        "[PRIMARY] "
        f"submit_us={aggregate['submit_span_us']['median']:.3f} "
        f"all_miss_per_core={all_group['icache_misses_per_core']['median']:.3f} "
        f"AIC_miss_per_core={aic['icache_misses_per_core']['median']:.3f} "
        f"AIV_request_per_core={aiv['icache_requests_per_core']['median']:.3f} "
        f"AIV_miss_per_core={aiv['icache_misses_per_core']['median']:.3f} "
        f"AIV_core_miss_p95={aiv['icache_misses_per_core_p95']['median']:.3f} "
        f"AIV_miss_rate={aiv['icache_miss_rate']['median'] * 100:.4f}%"
    )
    if result["input_schema"]["version"] == 4:
        phase = result["phase_observation"]
        if not phase["enabled"]:
            print(
                "[PHASE-BOUNDS] selected=none status=DISABLED "
                "request_per_core=0..0 miss_per_core=0..0 shadow_loss_per_core=request:0,miss:0 "
                "shadow_partition=EXACT_DISABLED"
            )
        else:
            print(
                "[PHASE-BOUNDS] "
                f"selected={phase['compiled_phase']} semantics=RUNNING_READ_CLEAR_BOUNDS "
                f"AIC_request_per_core="
                f"{aic['phase_icache_requests_lower_bound_per_core']['median']:.3f}.."
                f"{aic['phase_icache_requests_upper_bound_per_core']['median']:.3f} "
                f"AIC_miss_per_core="
                f"{aic['phase_icache_misses_lower_bound_per_core']['median']:.3f}.."
                f"{aic['phase_icache_misses_upper_bound_per_core']['median']:.3f} "
                f"AIC_shadow_loss_per_core=request:{aic['shadow_request_loss_per_core']['median']:.3f},"
                f"miss:{aic['shadow_miss_loss_per_core']['median']:.3f} "
                f"AIV_request_per_core="
                f"{aiv['phase_icache_requests_lower_bound_per_core']['median']:.3f}.."
                f"{aiv['phase_icache_requests_upper_bound_per_core']['median']:.3f} "
                f"AIV_miss_per_core="
                f"{aiv['phase_icache_misses_lower_bound_per_core']['median']:.3f}.."
                f"{aiv['phase_icache_misses_upper_bound_per_core']['median']:.3f} "
                f"AIV_shadow_loss_per_core=request:{aiv['shadow_request_loss_per_core']['median']:.3f},"
                f"miss:{aiv['shadow_miss_loss_per_core']['median']:.3f} "
                "shadow_partition=BOUNDED_NOT_EXACT_REQUIRED"
            )
        boundary_state = "PERTURBED" if phase["enabled"] else "DISABLED"
        pair_state = "FALSE" if phase["enabled"] else "NOT_APPLICABLE"
        print(
            "[PERTURBATION] primary_submit_whole=AUTHORITATIVE "
            f"phase_boundary_observer={boundary_state} phase_counter_pair_atomic={pair_state} "
            "cross_phase_elf_sum=INVALID"
        )
    print(
        "[SERIAL-EQUIVALENT] "
        f"penalty={result['estimation']['icache_miss_penalty_ns']:.3f}ns/miss "
        f"AIC_per_core_us={aic['first_order_miss_per_core_us']['median']:.3f} "
        f"AIV_per_core_us={aiv['first_order_miss_per_core_us']['median']:.3f} "
        "meaning=core_equivalent_not_wall_or_additive_stall"
    )
    print(
        "[ACTUAL-EXPOSED-LOSS] status=UNMEASURED "
        "method=requires_same_semantics_paired_AB_delta_submit_and_delta_miss"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="历史 schema-v3 或 submit-pmu schema-v4 JSON sidecars"
    )
    parser.add_argument(
        "--icache-miss-ns",
        type=float,
        default=90.0,
        help="受控 cold/warm 标尺；只用于一阶 core-work 等效估算（默认 90）",
    )
    parser.add_argument("--json", action="store_true", help="输出机器可读的聚合 JSON")
    arguments = parser.parse_args(argv)
    try:
        result = analyze(arguments.inputs, arguments.icache_miss_ns)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"PMU sidecar analysis failed: {error}", file=sys.stderr)
        return 1
    if arguments.json:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        _print_text(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
