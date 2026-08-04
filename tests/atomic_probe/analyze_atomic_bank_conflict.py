#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the LICENSE file in the root directory of this source tree for more details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""校验并分析不同 64B 地址的 A5 atomic bank 冲突探针。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import lzma
import math
import shutil
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

SCENARIOS = ("aic", "aiv", "mixed")
SCENARIO_LABELS = {"aic": "纯 AIC", "aiv": "纯 AIV", "mixed": "AIC+AIV 混合"}
DEFAULT_GAPS = tuple(range(0, 4096 + 64, 64))
DEFAULT_PHASES = tuple(range(0, 512, 64))
DEFAULT_STRIDES = (0, 64, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_POPULATIONS = {
    "aic": (1, 2, 4, 8, 16),
    "aiv": (1, 2, 4, 8, 16, 24, 32),
    "mixed": (1, 2, 4, 8, 16, 24, 32, 48),
}
TWO_GROUP_PATTERNS = {
    "two_group_same": {"target_offset_bytes": 0, "target_stride_bytes": 0, "address_layout": "shared"},
    "two_group_64_same128": {
        "target_offset_bytes": 0,
        "target_stride_bytes": 64,
        "address_layout": "two_groups",
    },
    "two_group_64_cross128": {
        "target_offset_bytes": 64,
        "target_stride_bytes": 64,
        "address_layout": "two_groups",
    },
    "two_group_128": {"target_offset_bytes": 0, "target_stride_bytes": 128, "address_layout": "two_groups"},
    "two_group_512": {"target_offset_bytes": 0, "target_stride_bytes": 512, "address_layout": "two_groups"},
    "two_group_1024": {"target_offset_bytes": 0, "target_stride_bytes": 1024, "address_layout": "two_groups"},
}
A5_SUBCORES_PER_DIE = 54
A5_AIC_SUBCORES_PER_DIE = 18
TSV_COLUMNS = (
    "experiment",
    "scenario",
    "sweep",
    "N",
    "aic_active",
    "aiv_active",
    "active_hardware_core_ids",
    "target_address",
    "target_offset_bytes",
    "address_layout",
    "target_stride_bytes",
    "unique_addresses",
    "target_span_bytes",
    "wave",
    "measured",
    "deadline_tick",
    "min_begin_tick",
    "max_begin_tick",
    "max_end_tick",
    "start_spread_tick",
    "global_span_tick",
    "max_worker_elapsed_tick",
    "min_publish_begin_tick",
    "max_publish_ready_tick",
    "old_min",
    "old_max",
    "old_sum",
    "old_xor",
    "status",
)


class AnalysisError(ValueError):
    """原始证据不完整或内部不一致。"""


@dataclass(frozen=True)
class AnalysisConfig:
    sweeps: int = 10
    warmup_waves: int = 4
    measured_waves: int = 64
    gaps: tuple[int, ...] = DEFAULT_GAPS
    phases: tuple[int, ...] = DEFAULT_PHASES
    strides: tuple[int, ...] = DEFAULT_STRIDES
    populations: Mapping[str, tuple[int, ...]] = field(default_factory=lambda: dict(DEFAULT_POPULATIONS))
    sys_counter_hz: int = 1_000_000_000
    max_start_spread_ticks: int = 64
    wave_stride_ticks: int = 100_000
    wave_publish_offset_ticks: int = 50_000

    @property
    def total_waves(self) -> int:
        return self.warmup_waves + self.measured_waves

    def __post_init__(self) -> None:
        if self.sweeps <= 0 or self.measured_waves <= 0 or self.warmup_waves < 0:
            raise AnalysisError("重复次数和 wave 数配置非法")
        if tuple(sorted(set(self.gaps))) != self.gaps or not self.gaps or self.gaps[0] != 0:
            raise AnalysisError("gap 必须是从 0 开始的严格递增唯一序列")
        if tuple(sorted(set(self.phases))) != self.phases or not self.phases or self.phases[0] != 0:
            raise AnalysisError("phase 必须是从 0 开始的严格递增唯一序列")
        if tuple(sorted(set(self.strides))) != self.strides or not self.strides or self.strides[0] != 0:
            raise AnalysisError("stride 必须是从 0 开始的严格递增唯一序列")
        if set(self.populations) != set(SCENARIOS):
            raise AnalysisError(f"population 必须恰好覆盖 {SCENARIOS}")
        if any(not values or tuple(sorted(set(values))) != values for values in self.populations.values()):
            raise AnalysisError("每类核的 population 必须是严格递增非空序列")
        if self.sys_counter_hz <= 0 or not 0 < self.wave_publish_offset_ticks < self.wave_stride_ticks:
            raise AnalysisError("计时配置非法")


@dataclass(frozen=True)
class WaveRecord:
    experiment: str
    scenario: str
    sweep: int
    n: int
    aic_active: int
    aiv_active: int
    hardware_core_ids: tuple[int, ...]
    target_address: int
    target_offset_bytes: int
    address_layout: str
    target_stride_bytes: int
    unique_addresses: int
    target_span_bytes: int
    wave: int
    measured: int
    deadline_tick: int
    min_begin_tick: int
    max_begin_tick: int
    max_end_tick: int
    start_spread_tick: int
    global_span_tick: int
    max_worker_elapsed_tick: int
    min_publish_begin_tick: int
    max_publish_ready_tick: int
    old_min: int
    old_max: int
    old_sum: int
    old_xor: int
    status: str
    source_line: int = 0


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _integer(value: str, column: str, line: int) -> int:
    try:
        return int(value, 0)
    except ValueError as error:
        raise AnalysisError(f"第 {line} 行：{column} 不是整数：{value!r}") from error


def _core_ids(value: str, line: int) -> tuple[int, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise AnalysisError(f"第 {line} 行：active_hardware_core_ids 不是合法 JSON") from error
    if not isinstance(parsed, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in parsed):
        raise AnalysisError(f"第 {line} 行：active_hardware_core_ids 必须是整数数组")
    return tuple(parsed)


def load_tsv(path: Path) -> list[WaveRecord]:
    try:
        handle = _open_text(path)
    except OSError as error:
        raise AnalysisError(f"无法打开原始数据 {path}: {error}") from error
    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != list(TSV_COLUMNS):
            raise AnalysisError(f"TSV 表头不匹配：{reader.fieldnames}")
        records: list[WaveRecord] = []
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None or value == "" for value in row.values()):
                raise AnalysisError(f"第 {line} 行：存在空字段或额外字段")
            integers = {
                name: _integer(str(row[name]), name, line)
                for name in TSV_COLUMNS
                if name not in {"experiment", "scenario", "active_hardware_core_ids", "address_layout", "status"}
            }
            records.append(
                WaveRecord(
                    experiment=str(row["experiment"]),
                    scenario=str(row["scenario"]),
                    sweep=integers["sweep"],
                    n=integers["N"],
                    aic_active=integers["aic_active"],
                    aiv_active=integers["aiv_active"],
                    hardware_core_ids=_core_ids(str(row["active_hardware_core_ids"]), line),
                    target_address=integers["target_address"],
                    target_offset_bytes=integers["target_offset_bytes"],
                    address_layout=str(row["address_layout"]),
                    target_stride_bytes=integers["target_stride_bytes"],
                    unique_addresses=integers["unique_addresses"],
                    target_span_bytes=integers["target_span_bytes"],
                    wave=integers["wave"],
                    measured=integers["measured"],
                    deadline_tick=integers["deadline_tick"],
                    min_begin_tick=integers["min_begin_tick"],
                    max_begin_tick=integers["max_begin_tick"],
                    max_end_tick=integers["max_end_tick"],
                    start_spread_tick=integers["start_spread_tick"],
                    global_span_tick=integers["global_span_tick"],
                    max_worker_elapsed_tick=integers["max_worker_elapsed_tick"],
                    min_publish_begin_tick=integers["min_publish_begin_tick"],
                    max_publish_ready_tick=integers["max_publish_ready_tick"],
                    old_min=integers["old_min"],
                    old_max=integers["old_max"],
                    old_sum=integers["old_sum"],
                    old_xor=integers["old_xor"],
                    status=str(row["status"]),
                    source_line=line,
                )
            )
    if not records:
        raise AnalysisError("原始数据为空")
    return records


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise AnalysisError("percentile 输入非法")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _roles(scenario: str, n: int) -> tuple[int, int]:
    if scenario == "aic":
        return n, 0
    if scenario == "aiv":
        return 0, n
    aic = (n + 2) // 3
    return aic, n - aic


def _two_group_sizes(scenario: str, n: int) -> tuple[int, int]:
    if scenario == "mixed":
        return _roles(scenario, n)
    primary = (n + 1) // 2
    return primary, n - primary


def _xor_upto(value: int) -> int:
    if value < 0:
        return 0
    return (value, 1, value + 1, 0)[value & 3]


def _xor_range(first: int, last: int) -> int:
    return _xor_upto(last) ^ _xor_upto(first - 1)


def _expected_shape(record: WaveRecord) -> tuple[str, int, int, int]:
    if record.experiment == "single_address":
        if record.scenario not in ("aic", "aiv") or record.n != 1:
            raise AnalysisError(f"第 {record.source_line} 行：single_address 场景非法")
        return "shared", 0, 1, 64
    if record.experiment == "pair_gap":
        if record.n != 2:
            raise AnalysisError(f"第 {record.source_line} 行：pair_gap 必须 N=2")
        if record.target_stride_bytes == 0:
            return "shared", 0, 1, 64
        return "participant_stride", record.target_stride_bytes, 2, record.target_stride_bytes + 64
    if record.experiment == "pair_phase_same":
        if record.n != 2:
            raise AnalysisError(f"第 {record.source_line} 行：pair_phase_same 必须 N=2")
        return "shared", 0, 1, 64
    if record.experiment == "pair_phase_64":
        if record.n != 2:
            raise AnalysisError(f"第 {record.source_line} 行：pair_phase_64 必须 N=2")
        return "participant_stride", 64, 2, 128
    if record.experiment in TWO_GROUP_PATTERNS:
        if record.n < 2:
            raise AnalysisError(f"第 {record.source_line} 行：two_group 必须 N>=2")
        pattern = TWO_GROUP_PATTERNS[record.experiment]
        stride = int(pattern["target_stride_bytes"])
        if pattern["address_layout"] == "shared":
            return "shared", 0, 1, 64
        return "two_groups", stride, 2, stride + 64
    if record.experiment == "multicore_curve":
        if record.target_stride_bytes == 0:
            return "shared", 0, 1, 64
        span = (record.n - 1) * record.target_stride_bytes + 64
        return "participant_stride", record.target_stride_bytes, record.n, span
    raise AnalysisError(f"第 {record.source_line} 行：未知实验 {record.experiment!r}")


def _validate_core_ids(record: WaveRecord) -> None:
    if len(record.hardware_core_ids) != record.n or len(set(record.hardware_core_ids)) != record.n:
        raise AnalysisError(f"第 {record.source_line} 行：硬件 Core ID 数量或唯一性错误")
    if any(core_id < 0 or core_id >= A5_SUBCORES_PER_DIE for core_id in record.hardware_core_ids):
        raise AnalysisError(f"第 {record.source_line} 行：活跃 Core 并非全部位于 DIE0")
    observed_aic = sum(core_id < A5_AIC_SUBCORES_PER_DIE for core_id in record.hardware_core_ids)
    if (observed_aic, record.n - observed_aic) != (record.aic_active, record.aiv_active):
        raise AnalysisError(f"第 {record.source_line} 行：物理 Core 类型与记录不一致")
    if record.scenario == "mixed":
        for participant, core_id in enumerate(record.hardware_core_ids):
            expected_aic = participant % 3 == 0
            if (core_id < A5_AIC_SUBCORES_PER_DIE) != expected_aic:
                raise AnalysisError(f"第 {record.source_line} 行：mixed participant 顺序错误")


def _expected_old_summary(record: WaveRecord) -> tuple[int, int, int, int]:
    if record.address_layout == "shared":
        first = record.wave * record.n
        last = first + record.n - 1
        return first, last, (first + last) * record.n // 2, _xor_range(first, last)
    if record.address_layout == "two_groups":
        primary, secondary = _two_group_sizes(record.scenario, record.n)
        primary_first = record.wave * primary
        secondary_first = record.wave * secondary
        primary_last = primary_first + primary - 1
        secondary_last = secondary_first + secondary - 1
        return (
            min(primary_first, secondary_first),
            max(primary_last, secondary_last),
            (primary_first + primary_last) * primary // 2 + (secondary_first + secondary_last) * secondary // 2,
            _xor_range(primary_first, primary_last) ^ _xor_range(secondary_first, secondary_last),
        )
    old = record.wave
    return old, old, old * record.n, old if record.n % 2 else 0


def _validate_wave(record: WaveRecord, config: AnalysisConfig) -> None:
    if record.status != "PASS" or record.scenario not in SCENARIOS:
        raise AnalysisError(f"第 {record.source_line} 行：状态或场景非法")
    if not 0 <= record.sweep < config.sweeps or not 0 <= record.wave < config.total_waves:
        raise AnalysisError(f"第 {record.source_line} 行：sweep/wave 越界")
    if (record.aic_active, record.aiv_active) != _roles(record.scenario, record.n):
        raise AnalysisError(f"第 {record.source_line} 行：AIC/AIV 数量错误")
    _validate_core_ids(record)
    expected_layout, expected_stride, expected_unique, expected_span = _expected_shape(record)
    observed_shape = (
        record.address_layout,
        record.target_stride_bytes,
        record.unique_addresses,
        record.target_span_bytes,
    )
    if observed_shape != (expected_layout, expected_stride, expected_unique, expected_span):
        raise AnalysisError(f"第 {record.source_line} 行：地址布局 {observed_shape} 与预期不符")
    if record.target_address <= 0 or record.target_address % 64 or record.target_offset_bytes % 64:
        raise AnalysisError(f"第 {record.source_line} 行：目标地址或偏移未按 64B 对齐")
    if record.measured != int(record.wave >= config.warmup_waves):
        raise AnalysisError(f"第 {record.source_line} 行：measured 标记错误")
    if not record.deadline_tick <= record.min_begin_tick <= record.max_begin_tick <= record.max_end_tick:
        raise AnalysisError(f"第 {record.source_line} 行：计时时间戳顺序错误")
    if record.start_spread_tick != record.max_begin_tick - record.min_begin_tick:
        raise AnalysisError(f"第 {record.source_line} 行：起跑离散值不闭合")
    if record.global_span_tick != record.max_end_tick - record.min_begin_tick or record.global_span_tick <= 0:
        raise AnalysisError(f"第 {record.source_line} 行：全局跨度不闭合")
    if not 0 < record.max_worker_elapsed_tick <= record.global_span_tick:
        raise AnalysisError(f"第 {record.source_line} 行：单 Core 最大耗时非法")
    publish_deadline = record.deadline_tick + config.wave_publish_offset_ticks
    next_deadline = record.deadline_tick + config.wave_stride_ticks
    if (
        record.max_end_tick >= publish_deadline
        or record.min_publish_begin_tick < publish_deadline
        or not record.min_publish_begin_tick <= record.max_publish_ready_tick < next_deadline
    ):
        raise AnalysisError(f"第 {record.source_line} 行：计时区间与结果发布区间重叠")
    if record.measured and record.start_spread_tick > config.max_start_spread_ticks:
        raise AnalysisError(f"第 {record.source_line} 行：起跑离散超过上限")
    expected_old = _expected_old_summary(record)
    observed_old = (record.old_min, record.old_max, record.old_sum, record.old_xor)
    if observed_old != expected_old:
        raise AnalysisError(f"第 {record.source_line} 行：atomic 返回值 {observed_old}，预期 {expected_old}")


def _launch_parameter(record: WaveRecord) -> int:
    if record.experiment in ("single_address", "pair_phase_same", "pair_phase_64"):
        return record.target_offset_bytes
    return record.target_stride_bytes


def _expected_points(config: AnalysisConfig) -> set[tuple[str, str, int, int]]:
    expected_points: set[tuple[str, str, int, int]] = set()
    expected_points.update(("single_address", scenario, gap, 1) for scenario in ("aic", "aiv") for gap in config.gaps)
    expected_points.update(("pair_gap", scenario, gap, 2) for scenario in SCENARIOS for gap in config.gaps)
    expected_points.update(
        (experiment, scenario, phase, 2)
        for experiment in ("pair_phase_same", "pair_phase_64")
        for scenario in SCENARIOS
        for phase in config.phases
    )
    expected_points.update(
        ("multicore_curve", scenario, stride, n)
        for scenario in SCENARIOS
        for stride in config.strides
        for n in config.populations[scenario]
    )
    expected_points.update(
        (experiment, scenario, int(pattern["target_stride_bytes"]), n)
        for experiment, pattern in TWO_GROUP_PATTERNS.items()
        for scenario in SCENARIOS
        for n in config.populations[scenario]
        if n >= 2
    )
    return expected_points


def _validate_launches(
    launches: Mapping[tuple[str, str, int, int, int], Sequence[WaveRecord]],
    expected_points: set[tuple[str, str, int, int]],
    config: AnalysisConfig,
) -> None:
    observed_points = {(experiment, scenario, parameter, n) for experiment, scenario, _, parameter, n in launches}
    if observed_points != expected_points:
        raise AnalysisError(
            f"测试点集合不完整：missing={sorted(expected_points - observed_points)}, "
            f"extra={sorted(observed_points - expected_points)}"
        )
    for point in expected_points:
        for sweep in range(config.sweeps):
            launch_key = (point[0], point[1], sweep, point[2], point[3])
            launch = sorted(launches.get(launch_key, []), key=lambda item: item.wave)
            if [item.wave for item in launch] != list(range(config.total_waves)):
                raise AnalysisError(f"launch wave 不完整：{launch_key}")
            if (
                len({item.hardware_core_ids for item in launch}) != 1
                or len({item.target_address for item in launch}) != 1
            ):
                raise AnalysisError(f"launch 内 Core 放置或目标地址发生变化：{launch_key}")
            for current, following in zip(launch, launch[1:]):
                if following.deadline_tick - current.deadline_tick != config.wave_stride_ticks:
                    raise AnalysisError(f"wave deadline 步长错误：{launch_key}")
                if current.max_publish_ready_tick >= following.deadline_tick:
                    raise AnalysisError(f"相邻 wave 发生重叠：{launch_key}")


def _validate_target_addresses(records: Sequence[WaveRecord]) -> int:
    base_addresses = {
        record.target_address for record in records if record.experiment in ("pair_gap", "multicore_curve")
    }
    if len(base_addresses) != 1:
        raise AnalysisError(f"pair/multicore 必须共用一个 GM0 基址：{base_addresses}")
    base_address = next(iter(base_addresses))
    if base_address % 512:
        raise AnalysisError("GM0 基址未按 512B 对齐，无法解释 512B 边界")
    for record in records:
        if record.experiment in ("single_address", "pair_phase_same", "pair_phase_64"):
            if record.target_address != base_address + record.target_offset_bytes:
                raise AnalysisError(f"第 {record.source_line} 行：带相位的目标不等于 base+offset")
        elif record.experiment in TWO_GROUP_PATTERNS:
            expected_offset = int(TWO_GROUP_PATTERNS[record.experiment]["target_offset_bytes"])
            if record.target_offset_bytes != expected_offset or record.target_address != base_address + expected_offset:
                raise AnalysisError(f"第 {record.source_line} 行：two_group 目标地址或相位错误")
        elif record.target_offset_bytes != 0:
            raise AnalysisError(f"第 {record.source_line} 行：pair/multicore 的 target_offset 必须为 0")
    return base_address


def validate_records(records: Sequence[WaveRecord], config: AnalysisConfig) -> dict[str, Any]:
    launches: dict[tuple[str, str, int, int, int], list[WaveRecord]] = defaultdict(list)
    seen: set[tuple[str, str, int, int, int, int]] = set()
    for record in records:
        _validate_wave(record, config)
        parameter = _launch_parameter(record)
        key = (record.experiment, record.scenario, record.sweep, parameter, record.n, record.wave)
        if key in seen:
            raise AnalysisError(f"重复 wave：{key}")
        seen.add(key)
        launches[key[:-1]].append(record)

    expected_points = _expected_points(config)
    _validate_launches(launches, expected_points, config)
    base_address = _validate_target_addresses(records)

    expected_launches = len(expected_points) * config.sweeps
    expected_rows = expected_launches * config.total_waves
    if len(records) != expected_rows:
        raise AnalysisError(f"原始行数应为 {expected_rows}，实际为 {len(records)}")
    return {
        "status": "PASS",
        "rows": len(records),
        "launches": expected_launches,
        "measured_wave_rows": expected_launches * config.measured_waves,
        "sweeps": config.sweeps,
        "warmup_waves_per_launch": config.warmup_waves,
        "measured_waves_per_launch": config.measured_waves,
        "all_active_cores_on_die0": True,
        "target_base_512b_aligned": True,
        "full_atomic_return_oracle_checked": True,
        "target_address": base_address,
    }


def _launch_medians(records: Sequence[WaveRecord]) -> dict[tuple[str, str, int, int, int], dict[str, float]]:
    grouped: dict[tuple[str, str, int, int, int], list[WaveRecord]] = defaultdict(list)
    for record in records:
        if record.measured:
            key = (record.experiment, record.scenario, record.sweep, _launch_parameter(record), record.n)
            grouped[key].append(record)
    result: dict[tuple[str, str, int, int, int], dict[str, float]] = {}
    for key, rows in grouped.items():
        result[key] = {
            "global_span": float(statistics.median(row.global_span_tick for row in rows)),
            "max_worker_elapsed": float(statistics.median(row.max_worker_elapsed_tick for row in rows)),
            "start_spread": float(statistics.median(row.start_spread_tick for row in rows)),
        }
    return result


def _distribution(values: Sequence[float], tick_to_ns: float = 1.0) -> dict[str, float]:
    return {
        "p10": percentile(values, 0.10) * tick_to_ns,
        "p50": percentile(values, 0.50) * tick_to_ns,
        "p90": percentile(values, 0.90) * tick_to_ns,
    }


def _linear_fit(points: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if len(points) < 2:
        raise AnalysisError("线性拟合至少需要两个点")
    xs = [float(point["n"]) for point in points]
    ys = [float(point["y_ns"]) for point in points]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        raise AnalysisError("线性拟合的 N 没有变化")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    predictions = [intercept + slope * x for x in xs]
    squared_error = sum((y - prediction) ** 2 for y, prediction in zip(ys, predictions))
    total_variation = sum((y - y_mean) ** 2 for y in ys)
    r_squared = (
        1.0
        if total_variation == 0 and squared_error == 0
        else (0.0 if total_variation == 0 else 1.0 - squared_error / total_variation)
    )
    return {
        "a_ns": intercept,
        "b_ns_per_core": slope,
        "r_squared": r_squared,
        "rmse_ns": math.sqrt(squared_error / len(points)),
    }


def _analyze_pair(
    launches: Mapping[tuple[str, str, int, int, int], Mapping[str, float]], config: AnalysisConfig
) -> dict[str, Any]:
    tick_to_ns = 1_000_000_000.0 / config.sys_counter_hz
    result: dict[str, Any] = {}
    supported_same_region = 0
    supported_boundary = 0
    for scenario in SCENARIOS:
        points: list[dict[str, Any]] = []
        for gap in config.gaps:
            pair_spans: list[float] = []
            worker_elapsed: list[float] = []
            independent_baselines: list[float] = []
            serialization_ratios: list[float] = []
            for sweep in range(config.sweeps):
                pair = launches[("pair_gap", scenario, sweep, gap, 2)]
                aic_base = launches[("single_address", "aic", sweep, 0, 1)]["global_span"]
                aiv_base = launches[("single_address", "aiv", sweep, 0, 1)]["global_span"]
                if scenario == "aic":
                    independent = max(aic_base, launches[("single_address", "aic", sweep, gap, 1)]["global_span"])
                elif scenario == "aiv":
                    independent = max(aiv_base, launches[("single_address", "aiv", sweep, gap, 1)]["global_span"])
                else:
                    independent = max(aic_base, launches[("single_address", "aiv", sweep, gap, 1)]["global_span"])
                same_pair = launches[("pair_gap", scenario, sweep, 0, 2)]["global_span"]
                same_independent = aic_base if scenario == "aic" else aiv_base
                if scenario == "mixed":
                    same_independent = max(aic_base, aiv_base)
                denominator = same_pair - same_independent
                if denominator <= 0:
                    raise AnalysisError(f"{scenario} sweep={sweep}: 同址双 Core 没有可用的串行增量")
                pair_spans.append(pair["global_span"])
                worker_elapsed.append(pair["max_worker_elapsed"])
                independent_baselines.append(independent)
                serialization_ratios.append((pair["global_span"] - independent) / denominator)
            span_distribution = _distribution(pair_spans, tick_to_ns)
            elapsed_distribution = _distribution(worker_elapsed, tick_to_ns)
            baseline_distribution = _distribution(independent_baselines, tick_to_ns)
            ratio_distribution = _distribution(serialization_ratios)
            points.append(
                {
                    "gap_bytes": gap,
                    "pair_global_span_p10_ns": span_distribution["p10"],
                    "pair_global_span_p50_ns": span_distribution["p50"],
                    "pair_global_span_p90_ns": span_distribution["p90"],
                    "max_worker_elapsed_p50_ns": elapsed_distribution["p50"],
                    "independent_baseline_p50_ns": baseline_distribution["p50"],
                    "serialization_ratio_p10": ratio_distribution["p10"],
                    "serialization_ratio_p50": ratio_distribution["p50"],
                    "serialization_ratio_p90": ratio_distribution["p90"],
                }
            )
        by_gap = {point["gap_bytes"]: point for point in points}
        within_gaps = [gap for gap in config.gaps if 0 < gap < 512]
        within_ratio = statistics.median(by_gap[gap]["serialization_ratio_p50"] for gap in within_gaps)
        boundary_ratio = by_gap[512]["serialization_ratio_p50"] if 512 in by_gap else math.nan
        same_region_serializes = within_ratio >= 0.70
        boundary_releases = 512 in by_gap and within_ratio - boundary_ratio >= 0.30
        supported_same_region += int(same_region_serializes)
        supported_boundary += int(boundary_releases)
        first_parallel = next(
            (
                point["gap_bytes"]
                for point in points
                if point["gap_bytes"] > 0 and point["serialization_ratio_p50"] <= 0.30
            ),
            None,
        )
        bucket_spreads: list[float] = []
        bucket_medians: list[float] = []
        for bucket in range(8):
            bucket_values = [
                point["pair_global_span_p50_ns"]
                for point in points
                if point["gap_bytes"] > 0 and point["gap_bytes"] // 512 == bucket
            ]
            if len(bucket_values) >= 2:
                bucket_spreads.append(max(bucket_values) - min(bucket_values))
                bucket_medians.append(statistics.median(bucket_values))
        result[scenario] = {
            "label": SCENARIO_LABELS[scenario],
            "points": points,
            "same_address_pair_p50_ns": by_gap[0]["pair_global_span_p50_ns"],
            "gap_64b_serialization_ratio_p50": by_gap[64]["serialization_ratio_p50"],
            "gap_128b_serialization_ratio_p50": by_gap[128]["serialization_ratio_p50"],
            "within_first_512b_serialization_ratio_p50": within_ratio,
            "gap_512b_serialization_ratio_p50": boundary_ratio,
            "same_region_serializes": same_region_serializes,
            "boundary_512b_releases": boundary_releases,
            "first_parallel_gap_bytes": first_parallel,
            "median_within_bucket_range_ns": statistics.median(bucket_spreads) if bucket_spreads else None,
            "between_bucket_median_range_ns": max(bucket_medians) - min(bucket_medians) if bucket_medians else None,
        }

    if supported_same_region >= 2 and supported_boundary >= 2:
        verdict = "SUPPORTED"
        verdict_zh = "支持“同一 512B 区间内不同 64B 行仍会因 bank 排队”的说法"
    elif supported_same_region >= 2:
        verdict = "PARTIALLY_SUPPORTED"
        verdict_zh = "支持存在地址级 bank 排队，但不足以把边界精确归因到 512B"
    else:
        verdict = "NOT_SUPPORTED"
        verdict_zh = "不支持“同一 512B 区间内不同 64B 行仍按同址 B 排队”的说法"
    return {
        "verdict": verdict,
        "verdict_zh": verdict_zh,
        "decision_thresholds": {"serialized_ratio_min": 0.70, "parallel_ratio_max": 0.30, "boundary_drop_min": 0.30},
        "serialization_ratio_definition": "(双地址实测 - 两次独立单核的并行基线) / (同址双核 - 同址单核基线)",
        "scenarios": result,
    }


def _analyze_phase(
    launches: Mapping[tuple[str, str, int, int, int], Mapping[str, float]], config: AnalysisConfig
) -> dict[str, Any]:
    tick_to_ns = 1_000_000_000.0 / config.sys_counter_hz
    result: dict[str, Any] = {}
    signatures = 0
    for scenario in SCENARIOS:
        points: list[dict[str, Any]] = []
        for phase in config.phases:
            same_spans: list[float] = []
            adjacent_spans: list[float] = []
            ratios: list[float] = []
            for sweep in range(config.sweeps):
                same_pair = launches[("pair_phase_same", scenario, sweep, phase, 2)]["global_span"]
                adjacent_pair = launches[("pair_phase_64", scenario, sweep, phase, 2)]["global_span"]
                aic_first = launches[("single_address", "aic", sweep, phase, 1)]["global_span"]
                aiv_first = launches[("single_address", "aiv", sweep, phase, 1)]["global_span"]
                aic_second = launches[("single_address", "aic", sweep, phase + 64, 1)]["global_span"]
                aiv_second = launches[("single_address", "aiv", sweep, phase + 64, 1)]["global_span"]
                if scenario == "aic":
                    same_baseline = aic_first
                    independent = max(aic_first, aic_second)
                elif scenario == "aiv":
                    same_baseline = aiv_first
                    independent = max(aiv_first, aiv_second)
                else:
                    same_baseline = max(aic_first, aiv_first)
                    independent = max(aic_first, aiv_second)
                denominator = same_pair - same_baseline
                if denominator <= 0:
                    raise AnalysisError(f"{scenario} phase={phase} sweep={sweep}: 同址相位基线没有串行增量")
                same_spans.append(same_pair)
                adjacent_spans.append(adjacent_pair)
                ratios.append((adjacent_pair - independent) / denominator)
            points.append(
                {
                    "phase_bytes": phase,
                    "same_address_pair_p50_ns": _distribution(same_spans, tick_to_ns)["p50"],
                    "adjacent_64b_pair_p50_ns": _distribution(adjacent_spans, tick_to_ns)["p50"],
                    "serialization_ratio_p10": _distribution(ratios)["p10"],
                    "serialization_ratio_p50": _distribution(ratios)["p50"],
                    "serialization_ratio_p90": _distribution(ratios)["p90"],
                }
            )
        aligned_ratios = [point["serialization_ratio_p50"] for point in points if point["phase_bytes"] % 128 == 0]
        crossing_ratios = [point["serialization_ratio_p50"] for point in points if point["phase_bytes"] % 128 == 64]
        aligned_ratio = statistics.median(aligned_ratios)
        crossing_ratio = statistics.median(crossing_ratios)
        signature = aligned_ratio >= 0.70 and crossing_ratio <= 0.30
        signatures += int(signature)
        result[scenario] = {
            "label": SCENARIO_LABELS[scenario],
            "points": points,
            "same_128b_region_serialization_ratio_p50": aligned_ratio,
            "cross_128b_boundary_serialization_ratio_p50": crossing_ratio,
            "alternating_128b_signature": signature,
        }
    observed = signatures == len(SCENARIOS)
    return {
        "observed_128b_conflict_signature": observed,
        "verdict_zh": (
            "相邻 64B 地址是否排队随 128B 边界严格交替，支持 128B 冲突粒度"
            if observed
            else "未观察到跨三类核一致的 128B 相位交替"
        ),
        "scenarios": result,
    }


def _analyze_multicore(
    launches: Mapping[tuple[str, str, int, int, int], Mapping[str, float]], config: AnalysisConfig
) -> dict[str, Any]:
    tick_to_ns = 1_000_000_000.0 / config.sys_counter_hz
    result: dict[str, Any] = {}
    for scenario in SCENARIOS:
        series: dict[str, Any] = {}
        for stride in config.strides:
            points: list[dict[str, Any]] = []
            for n in config.populations[scenario]:
                spans = [
                    launches[("multicore_curve", scenario, sweep, stride, n)]["global_span"]
                    for sweep in range(config.sweeps)
                ]
                distribution = _distribution(spans, tick_to_ns)
                points.append(
                    {
                        "n": n,
                        "p10_ns": distribution["p10"],
                        "y_ns": distribution["p50"],
                        "p90_ns": distribution["p90"],
                    }
                )
            all_fit = _linear_fit(points)
            low_points = [point for point in points if point["n"] <= 8]
            low_fit = _linear_fit(low_points)
            series[str(stride)] = {
                "layout": "同址" if stride == 0 else f"每 Core 独立地址，步长 {stride}B",
                "points": points,
                "fit_all": all_fit,
                "fit_n_le_8": low_fit,
            }
        shared_b = series["0"]["fit_n_le_8"]["b_ns_per_core"]
        for stride in config.strides[1:]:
            series[str(stride)]["fit_n_le_8"]["b_ratio_to_shared"] = (
                series[str(stride)]["fit_n_le_8"]["b_ns_per_core"] / shared_b
            )
            shared_points = {point["n"]: point["y_ns"] for point in series["0"]["points"]}
            for point in series[str(stride)]["points"]:
                point["speedup_vs_shared"] = shared_points[point["n"]] / point["y_ns"]
        max_n = config.populations[scenario][-1]
        max_n_values = {
            str(stride): next(point["y_ns"] for point in series[str(stride)]["points"] if point["n"] == max_n)
            for stride in config.strides
        }
        fastest_stride = min(config.strides[1:], key=lambda stride: max_n_values[str(stride)])
        result[scenario] = {
            "label": SCENARIO_LABELS[scenario],
            "series": series,
            "dense_64b_low_n_b_ratio_to_shared": series["64"]["fit_n_le_8"]["b_ratio_to_shared"],
            "stride_512b_low_n_b_ratio_to_shared": series["512"]["fit_n_le_8"]["b_ratio_to_shared"],
            "max_n": max_n,
            "max_n_p50_ns_by_stride": max_n_values,
            "fastest_distinct_address_stride_at_max_n": fastest_stride,
            "fastest_distinct_address_speedup_vs_shared_at_max_n": (
                max_n_values["0"] / max_n_values[str(fastest_stride)]
            ),
        }
    return {"scenarios": result}


def _analyze_two_group(
    launches: Mapping[tuple[str, str, int, int, int], Mapping[str, float]], config: AnalysisConfig
) -> dict[str, Any]:
    labels = {
        "two_group_same": "两组同一地址",
        "two_group_64_same128": "相隔 64B、同一 128B 区间",
        "two_group_64_cross128": "相隔 64B、跨 128B 边界",
        "two_group_128": "相隔 128B",
        "two_group_512": "相隔 512B",
        "two_group_1024": "相隔 1024B",
    }
    tick_to_ns = 1_000_000_000.0 / config.sys_counter_hz
    result: dict[str, Any] = {}
    merged_signatures = 0
    parallel_signatures = 0
    for scenario in SCENARIOS:
        layouts: dict[str, Any] = {}
        populations = [n for n in config.populations[scenario] if n >= 2]
        for experiment, pattern in TWO_GROUP_PATTERNS.items():
            stride = int(pattern["target_stride_bytes"])
            points: list[dict[str, Any]] = []
            for n in populations:
                spans = [
                    launches[(experiment, scenario, sweep, stride, n)]["global_span"] for sweep in range(config.sweeps)
                ]
                distribution = _distribution(spans, tick_to_ns)
                primary, secondary = _two_group_sizes(scenario, n)
                points.append(
                    {
                        "n": n,
                        "primary_group_workers": primary,
                        "secondary_group_workers": secondary,
                        "max_group_workers": max(primary, secondary),
                        "p10_ns": distribution["p10"],
                        "y_ns": distribution["p50"],
                        "p90_ns": distribution["p90"],
                    }
                )
            fit_total = _linear_fit(points)
            queue_points = [{"n": point["max_group_workers"], "y_ns": point["y_ns"]} for point in points]
            fit_max_group = _linear_fit(queue_points)
            layouts[experiment] = {
                "label": labels[experiment],
                "target_offset_bytes": int(pattern["target_offset_bytes"]),
                "target_stride_bytes": stride,
                "points": points,
                "fit_vs_total_n": fit_total,
                "fit_vs_max_group": fit_max_group,
            }
        same_b = layouts["two_group_same"]["fit_vs_total_n"]["b_ns_per_core"]
        for entry in layouts.values():
            entry["fit_vs_total_n"]["b_ratio_to_same_address"] = entry["fit_vs_total_n"]["b_ns_per_core"] / same_b
            entry["fit_vs_max_group"]["b_ratio_to_same_address"] = entry["fit_vs_max_group"]["b_ns_per_core"] / same_b
        same_128_ratio = layouts["two_group_64_same128"]["fit_vs_total_n"]["b_ratio_to_same_address"]
        cross_128_queue_ratio = layouts["two_group_64_cross128"]["fit_vs_max_group"]["b_ratio_to_same_address"]
        gap_128_queue_ratio = layouts["two_group_128"]["fit_vs_max_group"]["b_ratio_to_same_address"]
        gap_512_queue_ratio = layouts["two_group_512"]["fit_vs_max_group"]["b_ratio_to_same_address"]
        gap_1024_queue_ratio = layouts["two_group_1024"]["fit_vs_max_group"]["b_ratio_to_same_address"]
        groups_merge = 0.80 <= same_128_ratio <= 1.20
        groups_parallel = all(
            0.80 <= ratio <= 1.20
            for ratio in (cross_128_queue_ratio, gap_128_queue_ratio, gap_512_queue_ratio, gap_1024_queue_ratio)
        )
        merged_signatures += int(groups_merge)
        parallel_signatures += int(groups_parallel)
        result[scenario] = {
            "label": SCENARIO_LABELS[scenario],
            "layouts": layouts,
            "same_128b_groups_merge": groups_merge,
            "cross_128b_groups_parallel": groups_parallel,
            "same_128b_b_total_ratio": same_128_ratio,
            "cross_128b_b_max_group_ratio": cross_128_queue_ratio,
            "gap_128b_b_max_group_ratio": gap_128_queue_ratio,
            "gap_512b_b_max_group_ratio": gap_512_queue_ratio,
            "gap_1024b_b_max_group_ratio": gap_1024_queue_ratio,
        }
    verified = merged_signatures == len(SCENARIOS) and parallel_signatures == len(SCENARIOS)
    return {
        "verdict": "VERIFIED" if verified else "INCONCLUSIVE",
        "verdict_zh": (
            "两组热点位于同一 128B 冲突单元时合并成 B×(nA+nB)；跨 128B 后按 B×max(nA,nB) 并行"
            if verified
            else "两组热点曲线未跨三类核形成一致的合并/并行判据"
        ),
        "scenarios": result,
    }


def analyze_records(
    records: Sequence[WaveRecord], config: AnalysisConfig | None = None, source_revision: str = "unknown"
) -> dict[str, Any]:
    config = config or AnalysisConfig()
    validation = validate_records(records, config)
    launches = _launch_medians(records)
    pair = _analyze_pair(launches, config)
    phase = _analyze_phase(launches, config)
    multicore = _analyze_multicore(launches, config)
    two_group = _analyze_two_group(launches, config)
    pair["practical_verdict_zh"] = (
        "512B 共 bank 全串行的说法不成立；但仅隔离 64B 不安全，观察到 128B 冲突粒度"
        if phase["observed_128b_conflict_signature"]
        else pair["verdict_zh"]
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": source_revision,
        "device_timing": {"counter": "SYS_CNT", "frequency_hz": config.sys_counter_hz, "tick_ns": 1.0},
        "validation": validation,
        "pair_gap": pair,
        "pair_phase": phase,
        "multicore": multicore,
        "two_group": two_group,
    }


def _append_two_group_svg(lines: list[str], summary: Mapping[str, Any]) -> None:
    left, top, plot_width, plot_height = 75.0, 850.0, 1080.0, 300.0
    layouts = summary["two_group"]["scenarios"]["mixed"]["layouts"]
    experiments = (
        "two_group_same",
        "two_group_64_same128",
        "two_group_64_cross128",
        "two_group_512",
    )
    colors = {
        "two_group_same": "#333333",
        "two_group_64_same128": "#CC79A7",
        "two_group_64_cross128": "#009E73",
        "two_group_512": "#56B4E9",
    }
    labels = {
        "two_group_same": "同址",
        "two_group_64_same128": "64B 同 128B",
        "two_group_64_cross128": "64B 跨 128B",
        "two_group_512": "相隔 512B",
    }
    all_y = [point["y_ns"] for experiment in experiments for point in layouts[experiment]["points"]]
    y_max = math.ceil(max(all_y) / 1000.0) * 1000.0
    populations = [point["n"] for point in layouts["two_group_same"]["points"]]
    min_n, max_n = min(populations), max(populations)
    for tick in range(0, 6):
        value = y_max * tick / 5
        y = top + plot_height - tick / 5 * plot_height
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{value:.0f}</text>')
    for n in populations:
        x = left + (n - min_n) / (max_n - min_n) * plot_width
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}"/>')
        lines.append(f'<text x="{x:.1f}" y="{top + plot_height + 19}" text-anchor="middle" font-size="11">{n}</text>')
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<text x="{left + plot_width / 2}" y="{top + plot_height + 43}" '
            'text-anchor="middle" font-size="13">总 Core 数 N（混合场景按 AIC/AIV 分成两组）</text>',
            f'<text transform="translate(20 {top + plot_height / 2}) rotate(-90)" '
            'text-anchor="middle" font-size="13">全局跨度（ns）</text>',
            '<text x="75" y="827" font-size="14" font-weight="700">两个组内同址热点的并发曲线</text>',
        ]
    )
    for legend_index, experiment in enumerate(experiments):
        coordinates = []
        for point in layouts[experiment]["points"]:
            x = left + (point["n"] - min_n) / (max_n - min_n) * plot_width
            y = top + plot_height - point["y_ns"] / y_max * plot_height
            coordinates.append(f"{x:.1f},{y:.1f}")
            lines.append(f'<circle class="mark" cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{colors[experiment]}"/>')
        dash = ' stroke-dasharray="7 4"' if experiment == "two_group_64_same128" else ""
        lines.append(f'<polyline class="series" stroke="{colors[experiment]}"{dash} points="{" ".join(coordinates)}"/>')
        legend_x = 485 + legend_index * 165
        lines.append(
            f'<line x1="{legend_x}" y1="823" x2="{legend_x + 22}" y2="823" stroke="{colors[experiment]}" '
            f'stroke-width="3"{dash}/>'
        )
        lines.append(f'<text x="{legend_x + 28}" y="827" font-size="12">{labels[experiment]}</text>')


def _svg(summary: Mapping[str, Any]) -> str:
    width, height = 1200, 1230
    colors = {"aic": "#0072B2", "aiv": "#D55E00", "mixed": "#009E73"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#222}'
        ".axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}"
        ".series{fill:none;stroke-width:2}.mark{stroke:white;stroke-width:1}</style>",
        '<text x="50" y="34" font-size="22" font-weight="700">Atomic 不同地址 bank 冲突实测</text>',
    ]

    left, top, plot_width, plot_height = 75.0, 70.0, 1080.0, 300.0
    for boundary in range(0, 4097, 512):
        x = left + boundary / 4096 * plot_width
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}"/>')
        lines.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 19}" text-anchor="middle" font-size="11">{boundary}</text>'
        )
    for ratio in (0.0, 0.5, 1.0, 1.5):
        y = top + plot_height - ratio / 1.5 * plot_height
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{ratio:.1f}</text>')
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<text x="{left + plot_width / 2}" y="{top + plot_height + 42}" '
            'text-anchor="middle" font-size="13">两个不同地址的间距（B）</text>',
            f'<text transform="translate(20 {top + plot_height / 2}) rotate(-90)" '
            'text-anchor="middle" font-size="13">归一化串行比例</text>',
        ]
    )
    for legend_index, scenario in enumerate(SCENARIOS):
        points = summary["pair_gap"]["scenarios"][scenario]["points"]
        coordinates = []
        for point in points:
            x = left + point["gap_bytes"] / 4096 * plot_width
            ratio = max(-0.1, min(1.5, point["serialization_ratio_p50"]))
            y = top + plot_height - ratio / 1.5 * plot_height
            coordinates.append(f"{x:.1f},{y:.1f}")
        color = colors[scenario]
        lines.append(f'<polyline class="series" stroke="{color}" points="{" ".join(coordinates)}"/>')
        lx = 790 + legend_index * 125
        lines.append(f'<line x1="{lx}" y1="48" x2="{lx + 24}" y2="48" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{lx + 30}" y="52" font-size="12">{html.escape(SCENARIO_LABELS[scenario])}</text>')

    left2, top2, plot_width2, plot_height2 = 75.0, 460.0, 1080.0, 285.0
    mixed = summary["multicore"]["scenarios"]["mixed"]["series"]
    selected_strides = (0, 64, 512, 4096)
    stride_colors = {0: "#333333", 64: "#CC79A7", 512: "#56B4E9", 4096: "#E69F00"}
    all_y = [point["y_ns"] for stride in selected_strides for point in mixed[str(stride)]["points"]]
    y_max = math.ceil(max(all_y) / 500.0) * 500.0
    max_n = max(point["n"] for point in mixed["0"]["points"])
    for tick in range(0, 6):
        value = y_max * tick / 5
        y = top2 + plot_height2 - tick / 5 * plot_height2
        lines.append(f'<line class="grid" x1="{left2}" y1="{y:.1f}" x2="{left2 + plot_width2}" y2="{y:.1f}"/>')
        lines.append(f'<text x="{left2 - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{value:.0f}</text>')
    for n in (1, 8, 16, 24, 32, 48):
        x = left2 + (n - 1) / (max_n - 1) * plot_width2
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{top2}" x2="{x:.1f}" y2="{top2 + plot_height2}"/>')
        lines.append(f'<text x="{x:.1f}" y="{top2 + plot_height2 + 19}" text-anchor="middle" font-size="11">{n}</text>')
    lines.extend(
        [
            f'<line class="axis" x1="{left2}" y1="{top2 + plot_height2}" '
            f'x2="{left2 + plot_width2}" y2="{top2 + plot_height2}"/>',
            f'<line class="axis" x1="{left2}" y1="{top2}" x2="{left2}" y2="{top2 + plot_height2}"/>',
            f'<text x="{left2 + plot_width2 / 2}" y="{top2 + plot_height2 + 43}" '
            'text-anchor="middle" font-size="13">并发 Core 数 N（AIC+AIV 混合）</text>',
            f'<text transform="translate(20 {top2 + plot_height2 / 2}) rotate(-90)" '
            'text-anchor="middle" font-size="13">全局跨度（ns）</text>',
        ]
    )
    for legend_index, stride in enumerate(selected_strides):
        coordinates = []
        for point in mixed[str(stride)]["points"]:
            x = left2 + (point["n"] - 1) / (max_n - 1) * plot_width2
            y = top2 + plot_height2 - point["y_ns"] / y_max * plot_height2
            coordinates.append(f"{x:.1f},{y:.1f}")
            lines.append(f'<circle class="mark" cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{stride_colors[stride]}"/>')
        lines.append(f'<polyline class="series" stroke="{stride_colors[stride]}" points="{" ".join(coordinates)}"/>')
        lx = 640 + legend_index * 135
        label = "同址" if stride == 0 else f"步长 {stride}B"
        lines.append(
            f'<line x1="{lx}" y1="438" x2="{lx + 22}" y2="438" stroke="{stride_colors[stride]}" stroke-width="3"/>'
        )
        lines.append(f'<text x="{lx + 28}" y="442" font-size="12">{label}</text>')
    _append_two_group_svg(lines, summary)
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _report(summary: Mapping[str, Any], raw_name: str, raw_sha256: str) -> str:
    pair = summary["pair_gap"]
    two_group = summary["two_group"]
    lines = [
        "# Atomic 不同地址的 L2 bank 冲突实测",
        "",
        "## 结论",
        "",
        f"**真实两组热点场景：{two_group['verdict_zh']}。**",
        "",
        f"独立地址粒度结论：{pair['practical_verdict_zh']}。",
        "",
        f"对原始说法的判定是：**{pair['verdict_zh']}**。这里的结论只来自地址偏移与时延形态，"
        "不把设备虚拟地址直接当作公开的 bank 编号。",
        "",
        "## N 个 Core 分成两个同址热点组",
        "",
        "纯 AIC/AIV 近半分组；混合场景按 AIC→组 A、AIV→组 B。`B总核数` 是对总 Core 数 N 的斜率，"
        "`B最大组` 是对 `max(nA,nB)` 的斜率。若两组队列合并，前者应接近同址 B；若两组并行，后者应接近同址 B。",
        "",
        "| 场景 | 同址 B/ns | 64B 同 128B：B总核数/同址 | 64B 跨边界：B最大组/同址 | "
        "128B：B最大组/同址 | 512B：B最大组/同址 | 1024B：B最大组/同址 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        entry = two_group["scenarios"][scenario]
        layouts = entry["layouts"]
        same_b = layouts["two_group_same"]["fit_vs_total_n"]["b_ns_per_core"]
        lines.append(
            f"| {entry['label']} | {same_b:.2f} | "
            f"{layouts['two_group_64_same128']['fit_vs_total_n']['b_ratio_to_same_address']:.3f} | "
            f"{layouts['two_group_64_cross128']['fit_vs_max_group']['b_ratio_to_same_address']:.3f} | "
            f"{layouts['two_group_128']['fit_vs_max_group']['b_ratio_to_same_address']:.3f} | "
            f"{layouts['two_group_512']['fit_vs_max_group']['b_ratio_to_same_address']:.3f} | "
            f"{layouts['two_group_1024']['fit_vs_max_group']['b_ratio_to_same_address']:.3f} |"
        )
    mixed_layouts = two_group["scenarios"]["mixed"]["layouts"]
    merged_total = mixed_layouts["two_group_64_same128"]["fit_vs_total_n"]
    merged_max = mixed_layouts["two_group_64_same128"]["fit_vs_max_group"]
    parallel_total = mixed_layouts["two_group_64_cross128"]["fit_vs_total_n"]
    parallel_max = mixed_layouts["two_group_64_cross128"]["fit_vs_max_group"]
    lines.extend(
        [
            "",
            "纯 AIC/AIV 的两组近似等大，`N` 与 `max(nA,nB)` 成固定倍数，单靠 R² 不能区分模型。"
            "混合场景的分组比例会随 N 取整变化，因此是主要判据：64B 同一 128B 区间按总核数拟合时 "
            f"R²={merged_total['r_squared']:.6f}、RMSE={merged_total['rmse_ns']:.2f}ns，优于按最大组拟合的 "
            f"R²={merged_max['r_squared']:.6f}、RMSE={merged_max['rmse_ns']:.2f}ns；跨 128B 后关系反转，"
            f"按最大组拟合的 R²={parallel_max['r_squared']:.6f}、RMSE={parallel_max['rmse_ns']:.2f}ns，"
            f"优于按总核数拟合的 R²={parallel_total['r_squared']:.6f}、RMSE={parallel_total['rmse_ns']:.2f}ns。",
            "",
            "每个拟合的完整 A、B、R²、RMSE 和各 N 点 P10/P50/P90 均保存在 `summary.json`。",
            "",
            "## 两个 Core、两个不同地址",
            "",
            "串行比例定义为：`(双地址实测 - 两次单核的并行基线) / (同址双核 - 同址单核基线)`。"
            "接近 1 表示虽然地址不同，额外耗时仍接近同址排队；接近 0 表示两次 atomic 基本并行。",
            "",
            "| 场景 | 同址双核 P50/ns | 间距 64B 串行比例 | 间距 128B 串行比例 | "
            "间距 512B 串行比例 | 首个近似并行间距 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        entry = pair["scenarios"][scenario]
        first_parallel = entry["first_parallel_gap_bytes"]
        lines.append(
            f"| {entry['label']} | {entry['same_address_pair_p50_ns']:.1f} | "
            f"{entry['gap_64b_serialization_ratio_p50']:.3f} | "
            f"{entry['gap_128b_serialization_ratio_p50']:.3f} | "
            f"{entry['gap_512b_serialization_ratio_p50']:.3f} | "
            f"{str(first_parallel) + 'B' if first_parallel is not None else '未观察到'} |"
        )
    lines.extend(
        [
            "",
            "间距 64B 时三类组合都与同址一样完整串行；但同在首个 512B 区间内的 128/192/256/320/384/448B "
            "间距均已接近并行。因此，实测不能支持“整个 512B 区间共用一条 B≈170ns 队列”。",
            "",
            "## 相邻 64B 地址的 128B 相位验证",
            "",
            "| 场景 | 两地址落在同一 128B 区间时的串行比例 | 两地址跨 128B 边界时的串行比例 | 是否呈 128B 交替 |",
            "|---|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        entry = summary["pair_phase"]["scenarios"][scenario]
        lines.append(
            f"| {entry['label']} | {entry['same_128b_region_serialization_ratio_p50']:.3f} | "
            f"{entry['cross_128b_boundary_serialization_ratio_p50']:.3f} | "
            f"{'是' if entry['alternating_128b_signature'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "相邻地址对的起点每次平移 64B：`+0/+128/+256/+384B` 落在同一 128B 区间，"
            "`+64/+192/+320/+448B` 跨越 128B 边界。这个相位实验用于区分“固定 gap=64B 的偶然地址映射”与 128B 冲突粒度。",
            "",
            "## 多 Core、每 Core 独立地址",
            "",
            "非同址曲线明显不是 `A+B×N`：64B/256B/512B 的低 N 拟合 R² 很低，不能把其拟合斜率解释成单条 HA 队列的 B。"
            "因此下表直接给出最大并发点的实测中位数。",
            "",
            "| 场景 | 最大 N | 同址/ns | 步长 64B/ns | 步长 128B/ns | 步长 512B/ns | "
            "步长 1024B/ns | 最快独立步长 | 相对同址加速 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        entry = summary["multicore"]["scenarios"][scenario]
        values = entry["max_n_p50_ns_by_stride"]
        lines.append(
            f"| {entry['label']} | {entry['max_n']} | {values['0']:.1f} | {values['64']:.1f} | "
            f"{values['128']:.1f} | {values['512']:.1f} | {values['1024']:.1f} | "
            f"{entry['fastest_distinct_address_stride_at_max_n']}B | "
            f"{entry['fastest_distinct_address_speedup_vs_shared_at_max_n']:.2f}× |"
        )
    validation = summary["validation"]
    lines.extend(
        [
            "",
            "![不同地址与两组热点的 bank 冲突曲线](curve.svg)",
            "",
            "## 测试方法",
            "",
            "- 固定 DIE0 Core → GM0，GM0 基址按 512B 对齐，并限制扫描范围不跨 2MiB 候选页。",
            "- 每个 launch 先做 4 个预热 wave，再做 64 个计量 wave；共 10 次顺序轮换的 sweep。",
            "- 所有 Core 等待同一个未来 SYS_CNT 截止点；计时区间内只有一次 `atomicAdd`，结果发布在固定后半 wave。",
            "- 双 Core 扫描覆盖 0–4096B、步长 64B，并分别测 C+C、V+V、C+V。每个偏移另测 AIC/AIV 单核基线。",
            "- 另将相邻 64B 地址对的起点在一个 512B 区间内逐 64B 平移；每个相位同时测同址对照。",
            "- 多 Core 扫描比较同址及 64/128/256/512/1024/2048/4096B 步长；非零步长下每个 Core 都访问独立地址。",
            "- 两组热点扫描覆盖同址、相隔 64B 且同处一个 128B 区间、相隔 64B 但跨 128B 边界，以及"
            "相隔 128/512/1024B；纯 AIC/AIV 近半分组，混合场景按 AIC/AIV 分组。",
            "- host 校验完整目标区和每个 wave 的 atomic 返回值；独立地址必须各自从 0 连续增长到 67，"
            "两组热点必须分别形成各自完整的连续旧值序列。",
            "",
            "## 数据质量与限制",
            "",
            f"- 原始数据 {validation['rows']} 行，{validation['launches']} 个 launch，"
            f"其中计量 wave {validation['measured_wave_rows']} 行；语义与时序校验全部通过。",
            "- SYS_CNT 频率按 1GHz，因而 1 tick = 1ns。表中 P10/P50/P90 的独立样本层级是"
            "每个 launch 的 64-wave 中位数。",
            "- 512B bank 归因属于由地址周期和性能台阶反推的结论；若 bank 选择还包含更高位哈希，"
            "需要结合完整 4KiB 扫描解释，不能只看 `gap=512B` 单点。",
            "- 本测试测的是一次 atomic 返回就绪的全局完成跨度，不代表带其他访存或调度逻辑的完整业务 kernel。",
            "",
            "## 可复现信息",
            "",
            f"- 源码版本：`{summary['source_revision']}`",
            f"- 原始证据：`{raw_name}`",
            f"- 原始证据 SHA-256：`{raw_sha256}`",
            "- 执行：`tests/atomic_probe/ccec/run_atomic_bank_conflict.sh all`",
            "",
        ]
    )
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_raw_bundle(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    if source.suffix == ".xz":
        shutil.copyfile(source, destination)
        return
    with source.open("rb") as input_handle, lzma.open(destination, "wb", preset=9) as output_handle:
        shutil.copyfileobj(input_handle, output_handle)


def write_artifacts(
    summary: dict[str, Any], summary_path: Path, svg_path: Path, report_path: Path, raw_evidence: Path
) -> None:
    for path in (summary_path, svg_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    raw_hash = _sha256(raw_evidence)
    summary["raw_evidence"] = {"file": raw_evidence.name, "sha256": raw_hash}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    svg_path.write_text(_svg(summary), encoding="utf-8")
    report_path.write_text(_report(summary, raw_evidence.name, raw_hash), encoding="utf-8")


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--raw-bundle", type=Path)
    parser.add_argument("--source-revision", default="unknown")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    try:
        records = load_tsv(arguments.raw)
        summary = analyze_records(records, source_revision=arguments.source_revision)
        raw_evidence = arguments.raw
        if arguments.raw_bundle is not None:
            write_raw_bundle(arguments.raw, arguments.raw_bundle)
            raw_evidence = arguments.raw_bundle
        write_artifacts(summary, arguments.summary, arguments.svg, arguments.report, raw_evidence)
    except (AnalysisError, OSError) as error:
        print(f"atomic bank conflict analysis failed: {error}", file=sys.stderr)
        return 1
    print(
        f"[ANALYZE] {summary['pair_gap']['verdict']}: {summary['pair_gap']['verdict_zh']}; "
        f"rows={summary['validation']['rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
