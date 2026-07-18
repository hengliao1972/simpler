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
SCHEMA_VERSION = 3
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
SUMMARY_FIELDS = ("sum", "mean", "median", "p95", "max")

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


@dataclass(frozen=True)
class Capture:
    """一份已通过 raw→summary 和采集门禁的 sidecar。"""

    path: Path
    data: dict[str, Any]
    groups: dict[str, list[dict[str, Any]]]
    fingerprint: str


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


def _configuration_fingerprint(configuration: dict[str, Any]) -> str:
    selected = {field: configuration.get(field) for field in CONFIG_FINGERPRINT_FIELDS}
    return json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_group_summary(
    path: Path,
    group_name: str,
    records: Sequence[dict[str, Any]],
    expected: Any,
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

    for metric in METRIC_NAMES:
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


def load_capture(path: Path) -> Capture:
    """读取并完整校验一份 schema-v3 PMU sidecar。"""

    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    _require(isinstance(data, dict), f"{path}: capture root must be an object")
    schema = data.get("schema")
    _require(
        schema == {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        f"{path}: expected {SCHEMA_NAME} schema v{SCHEMA_VERSION}",
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
    _require(workers == aic_workers + aiv_workers, f"{path}: worker role counts do not add up")
    _require(len(records) == workers, f"{path}: record count does not match configuration.workers")

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

    groups: dict[str, list[dict[str, Any]]] = {
        "all": records,
        "aic": [record for record in records if record.get("role") == "aic"],
        "aiv": [record for record in records if record.get("role") == "aiv"],
    }
    _require(len(groups["aic"]) == aic_workers, f"{path}: AIC raw record count mismatch")
    _require(len(groups["aiv"]) == aiv_workers, f"{path}: AIV raw record count mismatch")

    worker_ids: set[int] = set()
    physical_core_ids: set[int] = set()
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

    for group_name in GROUP_NAMES:
        _validate_group_summary(path, group_name, groups[group_name], summary.get(group_name))

    return Capture(path, data, groups, _configuration_fingerprint(configuration))


def _median(values: Iterable[int | float]) -> int | float:
    return statistics.median(list(values))


def analyze(paths: Sequence[Path], miss_penalty_ns: float = 90.0) -> dict[str, Any]:
    """校验同配置多轮 sidecar，并返回可序列化的跨轮汇总。"""

    _require(bool(paths), "at least one PMU JSON path is required")
    _require(math.isfinite(miss_penalty_ns) and miss_penalty_ns > 0, "miss penalty must be positive")
    captures = [load_capture(path) for path in paths]
    fingerprint = captures[0].fingerprint
    for capture in captures[1:]:
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
    for group_name in GROUP_NAMES:
        aggregate["groups"][group_name] = {
            field: {
                "median": _median(row["groups"][group_name][field] for row in per_run),
                "min": min(row["groups"][group_name][field] for row in per_run),
                "max": max(row["groups"][group_name][field] for row in per_run),
            }
            for field in aggregate_fields
        }

    configuration = captures[0].data["configuration"]
    return {
        "schema": {"name": "pa_scheduler_pmu_multi_run_summary", "version": 1},
        "input_schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "configuration": {
            field: configuration.get(field) for field in CONFIG_FINGERPRINT_FIELDS
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


def _print_text(result: dict[str, Any]) -> None:
    configuration = result["configuration"]
    workload = configuration.get("winner_workload") or {}
    counts = workload.get("counts") or {}
    print(
        "[CONFIG] "
        f"window={configuration.get('pmu_window')} batches={configuration.get('batches')} "
        f"workers={configuration.get('workers')} workload={workload.get('mode')} "
        f"counts={counts.get('qk')},{counts.get('sf')},{counts.get('pv')},{counts.get('up')}"
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
    parser.add_argument("inputs", nargs="+", type=Path, help="schema-v3 PMU JSON sidecars")
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
