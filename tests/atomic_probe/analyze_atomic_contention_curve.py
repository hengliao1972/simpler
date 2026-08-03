#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validate and fit the A5 same-address atomic contention measurements."""

from __future__ import annotations

import argparse
import csv
import html
import json
import lzma
import math
import statistics
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

BASE_SCENARIOS = ("aic", "aiv", "mixed")
GM_HOMES = (0, 1)
SCENARIOS = tuple(f"{scenario}_gm{gm_home}" for scenario in BASE_SCENARIOS for gm_home in GM_HOMES)
LOCALITY_PROFILES = {
    f"{role}_gm{gm_home}_die{core_die}": (f"{role}_gm{gm_home}", core_die, gm_home)
    for role in ("aic", "aiv")
    for gm_home in GM_HOMES
    for core_die in (0, 1)
}
BASE_SCENARIO_LABELS = {
    "aic": "纯 AIC",
    "aiv": "纯 AIV",
    "mixed": "AIC+AIV 混合",
}
SCENARIO_LABELS = {
    f"{scenario}_gm{gm_home}": f"{label}：DIE0 Core → GM{gm_home}"
    for scenario, label in BASE_SCENARIO_LABELS.items()
    for gm_home in GM_HOMES
}
FIT_LABELS_ZH = {
    "STRONG": "强拟合",
    "APPROXIMATE": "近似拟合",
    "NOT_SUPPORTED": "不支持",
}
REQUESTED_N = (1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96)
_SUPPORTED_BY_ROLE = {
    "aic": (1, 2, 4, 8, 16),
    "aiv": (1, 2, 4, 8, 16, 24, 32),
    "mixed": (1, 2, 4, 8, 16, 24, 32, 48),
}
DEFAULT_SUPPORTED_POINTS = {
    f"{scenario}_gm{gm_home}": points for scenario, points in _SUPPORTED_BY_ROLE.items() for gm_home in GM_HOMES
}
DEFAULT_UNSUPPORTED_POINTS = {
    scenario: tuple(n for n in REQUESTED_N if n not in supported)
    for scenario, supported in DEFAULT_SUPPORTED_POINTS.items()
}
A5_PHYSICAL_SUBCORES = 108
A5_SUBCORES_PER_DIE = 54
A5_AIC_SLOTS_PER_DIE = 18
TSV_COLUMNS = (
    "scenario",
    "sweep",
    "N",
    "aic_active",
    "aiv_active",
    "active_hardware_core_ids",
    "target_address",
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
    """Raised when raw evidence is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class AnalysisConfig:
    """Expected capture shape; defaults describe the formal device run."""

    supported_points: Mapping[str, tuple[int, ...]] = field(default_factory=lambda: dict(DEFAULT_SUPPORTED_POINTS))
    unsupported_points: Mapping[str, tuple[int, ...]] = field(default_factory=lambda: dict(DEFAULT_UNSUPPORTED_POINTS))
    sweeps: int = 30
    warmup_waves: int = 4
    measured_waves: int = 64
    sys_counter_hz: int = 1_000_000_000
    max_start_spread_ticks: int = 64
    wave_stride_ticks: int = 100_000
    wave_publish_offset_ticks: int = 50_000
    min_home_classification_gap_ticks: int = 32
    max_matched_home_local_gap_ticks: int = 8

    def __post_init__(self) -> None:
        if self.sweeps <= 0:
            raise AnalysisError("expected sweeps must be positive")
        if self.warmup_waves < 0 or self.measured_waves <= 0:
            raise AnalysisError("wave counts must be non-negative, with at least one measured wave")
        if self.sys_counter_hz <= 0:
            raise AnalysisError("SYS_CNT frequency must be positive")
        if self.max_start_spread_ticks < 0:
            raise AnalysisError("maximum start spread must be non-negative")
        if not 0 < self.wave_publish_offset_ticks < self.wave_stride_ticks:
            raise AnalysisError("wave publication offset must be inside the wave stride")
        if self.min_home_classification_gap_ticks <= 0:
            raise AnalysisError("GM-home classification gap must be positive")
        if self.max_matched_home_local_gap_ticks < 0:
            raise AnalysisError("matched GM-home local gap must be non-negative")
        scenario_keys = set(self.supported_points) | set(self.unsupported_points)
        if scenario_keys != set(SCENARIOS):
            raise AnalysisError(f"config must describe exactly {SCENARIOS}, got {sorted(scenario_keys)}")
        for scenario in SCENARIOS:
            supported = tuple(self.supported_points[scenario])
            unsupported = tuple(self.unsupported_points[scenario])
            if not supported:
                raise AnalysisError(f"{scenario}: at least one supported N is required")
            if len(set(supported)) != len(supported) or len(set(unsupported)) != len(unsupported):
                raise AnalysisError(f"{scenario}: duplicate N in expected point configuration")
            if set(supported) & set(unsupported):
                raise AnalysisError(f"{scenario}: supported and unsupported N overlap")
            if any(value <= 0 for value in supported + unsupported):
                raise AnalysisError(f"{scenario}: N must be positive")

    @property
    def total_waves(self) -> int:
        return self.warmup_waves + self.measured_waves


@dataclass(frozen=True)
class WaveRecord:
    scenario: str
    sweep: int
    n: int
    aic_active: int
    aiv_active: int
    active_hardware_core_ids: tuple[int, ...]
    target_address: int
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


def _parse_int(raw: str, column: str, line: int) -> int:
    try:
        return int(raw, 0)
    except ValueError as error:
        raise AnalysisError(f"line {line}: {column} is not an integer: {raw!r}") from error


def _parse_hardware_core_ids(raw: str, line: int) -> tuple[int, ...]:
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as error:
        raise AnalysisError(f"line {line}: active_hardware_core_ids is not valid JSON: {raw!r}") from error
    if not isinstance(values, list):
        raise AnalysisError(f"line {line}: active_hardware_core_ids must be a JSON array")
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise AnalysisError(f"line {line}: active_hardware_core_ids[{index}] must be a JSON integer, got {value!r}")
        result.append(value)
    return tuple(result)


def load_tsv(path: Path) -> list[WaveRecord]:
    """Load the exact host TSV schema without weakening column provenance."""

    try:
        if path.suffix == ".xz":
            handle = lzma.open(path, "rt", encoding="utf-8", newline="")
        else:
            handle = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise AnalysisError(f"cannot open raw TSV {path}: {error}") from error
    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != list(TSV_COLUMNS):
            raise AnalysisError(f"unexpected TSV header: expected {list(TSV_COLUMNS)}, got {reader.fieldnames}")
        records: list[WaveRecord] = []
        for line, row in enumerate(reader, start=2):
            if None in row:
                raise AnalysisError(f"line {line}: unexpected extra TSV fields")
            if any(value is None or value == "" for value in row.values()):
                raise AnalysisError(f"line {line}: empty TSV field")
            records.append(
                WaveRecord(
                    scenario=str(row["scenario"]),
                    sweep=_parse_int(str(row["sweep"]), "sweep", line),
                    n=_parse_int(str(row["N"]), "N", line),
                    aic_active=_parse_int(str(row["aic_active"]), "aic_active", line),
                    aiv_active=_parse_int(str(row["aiv_active"]), "aiv_active", line),
                    active_hardware_core_ids=_parse_hardware_core_ids(str(row["active_hardware_core_ids"]), line),
                    target_address=_parse_int(str(row["target_address"]), "target_address", line),
                    wave=_parse_int(str(row["wave"]), "wave", line),
                    measured=_parse_int(str(row["measured"]), "measured", line),
                    deadline_tick=_parse_int(str(row["deadline_tick"]), "deadline_tick", line),
                    min_begin_tick=_parse_int(str(row["min_begin_tick"]), "min_begin_tick", line),
                    max_begin_tick=_parse_int(str(row["max_begin_tick"]), "max_begin_tick", line),
                    max_end_tick=_parse_int(str(row["max_end_tick"]), "max_end_tick", line),
                    start_spread_tick=_parse_int(str(row["start_spread_tick"]), "start_spread_tick", line),
                    global_span_tick=_parse_int(str(row["global_span_tick"]), "global_span_tick", line),
                    max_worker_elapsed_tick=_parse_int(
                        str(row["max_worker_elapsed_tick"]), "max_worker_elapsed_tick", line
                    ),
                    min_publish_begin_tick=_parse_int(
                        str(row["min_publish_begin_tick"]), "min_publish_begin_tick", line
                    ),
                    max_publish_ready_tick=_parse_int(
                        str(row["max_publish_ready_tick"]), "max_publish_ready_tick", line
                    ),
                    old_min=_parse_int(str(row["old_min"]), "old_min", line),
                    old_max=_parse_int(str(row["old_max"]), "old_max", line),
                    old_sum=_parse_int(str(row["old_sum"]), "old_sum", line),
                    old_xor=_parse_int(str(row["old_xor"]), "old_xor", line),
                    status=str(row["status"]),
                    source_line=line,
                )
            )
    if not records:
        raise AnalysisError(f"raw TSV is empty: {path}")
    return records


def _base_scenario(scenario: str) -> str:
    for base in BASE_SCENARIOS:
        if scenario.startswith(f"{base}_gm"):
            return base
    raise AnalysisError(f"unknown scenario: {scenario}")


def _scenario_gm_home(scenario: str) -> int:
    if scenario not in SCENARIOS:
        raise AnalysisError(f"unknown curve scenario: {scenario}")
    return int(scenario[-1])


def _active_roles(scenario: str, n: int) -> tuple[int, int]:
    base = _base_scenario(scenario)
    if base == "aic":
        return n, 0
    if base == "aiv":
        return 0, n
    if base == "mixed":
        aic = (n + 2) // 3
        return aic, n - aic
    raise AnalysisError(f"unknown scenario: {scenario}")


def _hardware_role(core_id: int) -> str:
    return "aic" if core_id % A5_SUBCORES_PER_DIE < A5_AIC_SLOTS_PER_DIE else "aiv"


def _participant_role(scenario: str, participant: int) -> str:
    base = _base_scenario(scenario)
    if base in ("aic", "aiv"):
        return base
    if base == "mixed":
        return "aic" if participant % 3 == 0 else "aiv"
    raise AnalysisError(f"unknown scenario: {scenario}")


def _validate_hardware_core_ids(record: WaveRecord, expected_core_die: int) -> None:
    location = _record_location(record)
    core_ids = record.active_hardware_core_ids
    if len(core_ids) != record.n:
        raise AnalysisError(
            f"{location}: active_hardware_core_ids must contain N={record.n} IDs, found {len(core_ids)}"
        )
    if len(set(core_ids)) != len(core_ids):
        raise AnalysisError(f"{location}: active_hardware_core_ids must be unique")
    for participant, core_id in enumerate(core_ids):
        if not 0 <= core_id < A5_PHYSICAL_SUBCORES:
            raise AnalysisError(
                f"{location}: active_hardware_core_ids[{participant}]={core_id} is outside the A5 physical range "
                f"[0,{A5_PHYSICAL_SUBCORES})"
            )
        if core_id // A5_SUBCORES_PER_DIE != expected_core_die:
            raise AnalysisError(
                f"{location}: active_hardware_core_ids[{participant}]={core_id} is outside physical die "
                f"{expected_core_die}"
            )
    observed_aic = sum(_hardware_role(core_id) == "aic" for core_id in core_ids)
    observed_aiv = len(core_ids) - observed_aic
    if (observed_aic, observed_aiv) != (record.aic_active, record.aiv_active):
        raise AnalysisError(
            f"{location}: hardware core role counts are {observed_aic}+{observed_aiv}, expected "
            f"{record.aic_active}+{record.aiv_active}"
        )
    for participant, core_id in enumerate(core_ids):
        expected_role = _participant_role(record.scenario, participant)
        if _hardware_role(core_id) != expected_role:
            raise AnalysisError(
                f"{location}: participant {participant} must use an {expected_role.upper()} physical slot, "
                f"got hardware core {core_id}"
            )
    if _base_scenario(record.scenario) == "mixed":
        for block_start in range(0, len(core_ids), 3):
            aic_id = core_ids[block_start]
            die_base = aic_id // A5_SUBCORES_PER_DIE * A5_SUBCORES_PER_DIE
            local_aic = aic_id % A5_SUBCORES_PER_DIE
            expected_aiv0 = die_base + A5_AIC_SLOTS_PER_DIE + 2 * local_aic
            for offset, expected_id in ((1, expected_aiv0), (2, expected_aiv0 + 1)):
                participant = block_start + offset
                if participant < len(core_ids) and core_ids[participant] != expected_id:
                    raise AnalysisError(
                        f"{location}: mixed participant {participant} hardware core {core_ids[participant]} "
                        f"does not match AIC triplet rooted at {aic_id}; expected {expected_id}"
                    )


def _xor_upto(value: int) -> int:
    if value < 0:
        return 0
    remainder = value & 3
    if remainder == 0:
        return value
    if remainder == 1:
        return 1
    if remainder == 2:
        return value + 1
    return 0


def _xor_range(first: int, last: int) -> int:
    return _xor_upto(last) ^ _xor_upto(first - 1)


def _record_location(record: WaveRecord) -> str:
    return f"line {record.source_line}" if record.source_line else "record"


def _matched_home_local_summary(profiles: Mapping[str, Mapping[str, Any]], config: AnalysisConfig) -> dict[str, Any]:
    tick_to_ns = 1_000_000_000.0 / config.sys_counter_hz
    maximum_gap_ns = config.max_matched_home_local_gap_ticks * tick_to_ns
    result: dict[str, Any] = {}
    for role in ("aic", "aiv"):
        gm0_local = float(profiles[f"{role}_gm0_die0"]["launch_median_p50_ns"])
        gm1_local = float(profiles[f"{role}_gm1_die1"]["launch_median_p50_ns"])
        gap = abs(gm0_local - gm1_local)
        if gap > maximum_gap_ns:
            raise AnalysisError(
                f"{role} GM0/GM1 home-local baselines differ by {gap} ns, exceeding {maximum_gap_ns} ns"
            )
        result[role] = {
            "gm0_home_local_p50_ns": gm0_local,
            "gm1_home_local_p50_ns": gm1_local,
            "absolute_gap_ns": gap,
        }
    return result


def _validate_unsupported(record: WaveRecord) -> None:
    location = _record_location(record)
    if record.status != "UNSUPPORTED":
        raise AnalysisError(f"{location}: unsupported point must use status=UNSUPPORTED")
    numeric_zero = (
        record.aic_active,
        record.aiv_active,
        record.measured,
        record.deadline_tick,
        record.min_begin_tick,
        record.max_begin_tick,
        record.max_end_tick,
        record.start_spread_tick,
        record.global_span_tick,
        record.max_worker_elapsed_tick,
        record.min_publish_begin_tick,
        record.max_publish_ready_tick,
        record.old_min,
        record.old_max,
        record.old_sum,
        record.old_xor,
    )
    if (
        record.sweep != -1
        or record.wave != -1
        or record.active_hardware_core_ids
        or record.target_address <= 0
        or record.target_address % 64 != 0
        or any(numeric_zero)
    ):
        raise AnalysisError(f"{location}: malformed UNSUPPORTED sentinel for {record.scenario} N={record.n}")


def _validate_supported_record(record: WaveRecord, config: AnalysisConfig, expected_core_die: int = 0) -> None:
    location = _record_location(record)
    if record.status != "PASS":
        raise AnalysisError(f"{location}: supported point {record.scenario} N={record.n} has status={record.status}")
    expected_aic, expected_aiv = _active_roles(record.scenario, record.n)
    if (record.aic_active, record.aiv_active) != (expected_aic, expected_aiv):
        raise AnalysisError(
            f"{location}: {record.scenario} N={record.n} active roles are "
            f"{record.aic_active}+{record.aiv_active}, expected {expected_aic}+{expected_aiv}"
        )
    if record.target_address <= 0 or record.target_address % 64 != 0:
        raise AnalysisError(f"{location}: target_address must be a nonzero 64-byte-aligned device address")
    _validate_hardware_core_ids(record, expected_core_die)
    expected_measured = int(record.wave >= config.warmup_waves)
    if record.measured != expected_measured:
        raise AnalysisError(f"{location}: wave {record.wave} measured={record.measured}, expected {expected_measured}")
    if record.min_begin_tick < record.deadline_tick:
        raise AnalysisError(f"{location}: a worker began before the common deadline")
    if not record.min_begin_tick <= record.max_begin_tick <= record.max_end_tick:
        raise AnalysisError(f"{location}: timestamp ordering is invalid")
    if record.start_spread_tick != record.max_begin_tick - record.min_begin_tick:
        raise AnalysisError(f"{location}: start_spread_tick does not close against begin ticks")
    if record.global_span_tick != record.max_end_tick - record.min_begin_tick:
        raise AnalysisError(f"{location}: global_span_tick does not close against global endpoints")
    if record.global_span_tick <= 0:
        raise AnalysisError(f"{location}: global span must be positive")
    if not 0 < record.max_worker_elapsed_tick <= record.global_span_tick:
        raise AnalysisError(f"{location}: max worker elapsed is outside the global span")
    publish_deadline = record.deadline_tick + config.wave_publish_offset_ticks
    next_deadline = record.deadline_tick + config.wave_stride_ticks
    if record.max_end_tick >= publish_deadline:
        raise AnalysisError(f"{location}: measured interval reaches the result-publication phase")
    if record.min_publish_begin_tick < publish_deadline:
        raise AnalysisError(f"{location}: result publication began before its fixed phase")
    if not record.min_publish_begin_tick <= record.max_publish_ready_tick < next_deadline:
        raise AnalysisError(f"{location}: result publication does not finish before the next wave")
    if record.measured == 1 and record.start_spread_tick > config.max_start_spread_ticks:
        raise AnalysisError(
            f"{location}: start spread {record.start_spread_tick} exceeds {config.max_start_spread_ticks} ticks"
        )
    first_old = record.wave * record.n
    last_old = first_old + record.n - 1
    expected_sum = (first_old + last_old) * record.n // 2
    expected_xor = _xor_range(first_old, last_old)
    observed_old = (record.old_min, record.old_max, record.old_sum, record.old_xor)
    expected_old = (first_old, last_old, expected_sum, expected_xor)
    if observed_old != expected_old:
        raise AnalysisError(f"{location}: returned-old oracle is {observed_old}, expected {expected_old}")


def _validate_launch_waves(
    launch: Sequence[WaveRecord], scenario: str, n: int, sweep: int, config: AnalysisConfig
) -> None:
    expected_waves = set(range(config.total_waves))
    waves = {record.wave for record in launch}
    if waves != expected_waves:
        missing = sorted(expected_waves - waves)
        extra = sorted(waves - expected_waves)
        raise AnalysisError(f"{scenario} N={n} sweep={sweep}: wave set mismatch, missing={missing}, extra={extra}")
    ordered_launch = sorted(launch, key=lambda record: record.wave)
    placements = {record.active_hardware_core_ids for record in ordered_launch}
    if len(placements) != 1:
        raise AnalysisError(f"{scenario} N={n} sweep={sweep}: hardware core placement changes across waves")
    target_addresses = {record.target_address for record in ordered_launch}
    if len(target_addresses) != 1:
        raise AnalysisError(f"{scenario} N={n} sweep={sweep}: target address changes across waves")
    for current, following in zip(ordered_launch, ordered_launch[1:]):
        if following.deadline_tick - current.deadline_tick != config.wave_stride_ticks:
            raise AnalysisError(f"{scenario} N={n} sweep={sweep}: wave deadline stride is invalid")
        if current.max_publish_ready_tick >= following.deadline_tick:
            raise AnalysisError(
                f"{scenario} N={n} sweep={sweep}: wave {current.wave} publication overlaps wave {following.wave}"
            )


def validate_records(records: Sequence[WaveRecord], config: AnalysisConfig) -> dict[str, Any]:
    """Fail closed on the capture shape and every per-wave semantic summary."""

    expected_supported = {(scenario, n) for scenario in SCENARIOS for n in config.supported_points[scenario]}
    expected_unsupported = {(scenario, n) for scenario in SCENARIOS for n in config.unsupported_points[scenario]}
    observed_supported: set[tuple[str, int]] = set()
    observed_unsupported: set[tuple[str, int]] = set()
    seen_keys: set[tuple[str, int, int, int]] = set()
    records_by_launch: dict[tuple[str, int, int], list[WaveRecord]] = {}

    for record in records:
        point = (record.scenario, record.n)
        if point in expected_unsupported:
            _validate_unsupported(record)
            if point in observed_unsupported:
                raise AnalysisError(f"duplicate UNSUPPORTED sentinel for {record.scenario} N={record.n}")
            observed_unsupported.add(point)
            continue
        if point not in expected_supported:
            raise AnalysisError(f"{_record_location(record)}: unexpected point {record.scenario} N={record.n}")
        if not 0 <= record.sweep < config.sweeps:
            raise AnalysisError(f"{_record_location(record)}: sweep is outside 0..{config.sweeps - 1}")
        if not 0 <= record.wave < config.total_waves:
            raise AnalysisError(f"{_record_location(record)}: wave is outside 0..{config.total_waves - 1}")
        key = (record.scenario, record.n, record.sweep, record.wave)
        if key in seen_keys:
            raise AnalysisError(f"duplicate wave row for {key}")
        seen_keys.add(key)
        _validate_supported_record(record, config)
        observed_supported.add(point)
        records_by_launch.setdefault((record.scenario, record.n, record.sweep), []).append(record)

    if observed_supported != expected_supported:
        missing = sorted(expected_supported - observed_supported)
        extra = sorted(observed_supported - expected_supported)
        raise AnalysisError(f"supported point set mismatch: missing={missing}, extra={extra}")
    if observed_unsupported != expected_unsupported:
        missing = sorted(expected_unsupported - observed_unsupported)
        extra = sorted(observed_unsupported - expected_unsupported)
        raise AnalysisError(f"unsupported point set mismatch: missing={missing}, extra={extra}")

    for scenario, n in sorted(expected_supported):
        for sweep in range(config.sweeps):
            launch = records_by_launch.get((scenario, n, sweep), [])
            _validate_launch_waves(launch, scenario, n, sweep, config)

    expected_rows = len(expected_unsupported) + sum(
        config.sweeps * config.total_waves * len(config.supported_points[scenario]) for scenario in SCENARIOS
    )
    if len(records) != expected_rows:
        raise AnalysisError(f"expected {expected_rows} total rows, found {len(records)}")
    supported_rows = expected_rows - len(expected_unsupported)
    target_addresses_by_home = {
        gm_home: {
            record.target_address
            for record in records
            if record.status == "PASS" and _scenario_gm_home(record.scenario) == gm_home
        }
        for gm_home in GM_HOMES
    }
    if any(len(addresses) != 1 for addresses in target_addresses_by_home.values()):
        raise AnalysisError(f"each GM home must use exactly one target address: {target_addresses_by_home}")
    gm_addresses = {gm_home: next(iter(addresses)) for gm_home, addresses in target_addresses_by_home.items()}
    if gm_addresses[0] == gm_addresses[1]:
        raise AnalysisError("GM0 and GM1 curves must use different target addresses")
    return {
        "status": "PASS",
        "rows": len(records),
        "supported_points": len(expected_supported),
        "unsupported_points": len(expected_unsupported),
        "supported_launches": len(expected_supported) * config.sweeps,
        "measured_wave_rows": len(expected_supported) * config.sweeps * config.measured_waves,
        "hardware_placement": {
            "status": "PASS",
            "rows": supported_rows,
            "launches": len(expected_supported) * config.sweeps,
            "physical_core_id_range": [0, A5_PHYSICAL_SUBCORES],
            "participant_order_checked": True,
            "mixed_triplets_checked": True,
            "constant_within_launch": True,
            "all_active_cores_on_die0": True,
        },
        "target_addresses": {f"gm{gm_home}": address for gm_home, address in gm_addresses.items()},
    }


def analyze_die_locality(records: Sequence[WaveRecord], config: AnalysisConfig) -> dict[str, Any]:
    """Validate reciprocal N=1 measurements used to classify GM0 and GM1."""

    seen_keys: set[tuple[str, int, int]] = set()
    records_by_launch: dict[tuple[str, int], list[WaveRecord]] = {}
    for record in records:
        if record.scenario not in LOCALITY_PROFILES:
            raise AnalysisError(f"{_record_location(record)}: unexpected locality profile {record.scenario!r}")
        if record.n != 1 or not 0 <= record.sweep < config.sweeps or not 0 <= record.wave < config.total_waves:
            raise AnalysisError(f"{_record_location(record)}: malformed locality row")
        key = (record.scenario, record.sweep, record.wave)
        if key in seen_keys:
            raise AnalysisError(f"duplicate locality wave row for {key}")
        seen_keys.add(key)

        curve_scenario, expected_die, _gm_home = LOCALITY_PROFILES[record.scenario]
        normalized = replace(record, scenario=curve_scenario)
        _validate_supported_record(normalized, config, expected_die)
        records_by_launch.setdefault((record.scenario, record.sweep), []).append(record)

    expected_rows = len(LOCALITY_PROFILES) * config.sweeps * config.total_waves
    if len(records) != expected_rows:
        raise AnalysisError(f"expected {expected_rows} locality rows, found {len(records)}")

    profiles: dict[str, Any] = {}
    tick_to_ns = 1_000_000_000.0 / config.sys_counter_hz
    for profile, (curve_scenario, expected_die, gm_home) in LOCALITY_PROFILES.items():
        launch_medians: list[float] = []
        measured_spans: list[int] = []
        hardware_ids: set[int] = set()
        profile_target_addresses: set[int] = set()
        for sweep in range(config.sweeps):
            launch = records_by_launch.get((profile, sweep), [])
            normalized_launch = [replace(record, scenario=curve_scenario) for record in launch]
            _validate_launch_waves(normalized_launch, curve_scenario, 1, sweep, config)
            measured = [record.global_span_tick for record in launch if record.measured == 1]
            if len(measured) != config.measured_waves:
                raise AnalysisError(
                    f"{profile} sweep={sweep}: expected {config.measured_waves} measured waves, found {len(measured)}"
                )
            launch_medians.append(float(statistics.median(measured)))
            measured_spans.extend(measured)
            hardware_ids.update(record.active_hardware_core_ids[0] for record in launch)
            profile_target_addresses.update(record.target_address for record in launch)
        if len(profile_target_addresses) != 1:
            raise AnalysisError(f"{profile}: target address changes across launches: {profile_target_addresses}")
        profiles[profile] = {
            "role": _base_scenario(curve_scenario),
            "gm_home": gm_home,
            "core_die": expected_die,
            "target_address": next(iter(profile_target_addresses)),
            "hardware_core_ids": sorted(hardware_ids),
            "launches": config.sweeps,
            "measured_waves_per_launch": config.measured_waves,
            "launch_median_p10_ns": percentile(launch_medians, 0.10) * tick_to_ns,
            "launch_median_p50_ns": percentile(launch_medians, 0.50) * tick_to_ns,
            "launch_median_p90_ns": percentile(launch_medians, 0.90) * tick_to_ns,
            "measured_wave_min_ns": min(measured_spans) * tick_to_ns,
            "measured_wave_max_ns": max(measured_spans) * tick_to_ns,
        }

    target_addresses: dict[int, int] = {}
    for gm_home in GM_HOMES:
        addresses = {int(entry["target_address"]) for entry in profiles.values() if int(entry["gm_home"]) == gm_home}
        if len(addresses) != 1:
            raise AnalysisError(f"GM{gm_home} locality profiles do not use exactly one target address: {addresses}")
        target_addresses[gm_home] = next(iter(addresses))
    if target_addresses[0] == target_addresses[1]:
        raise AnalysisError("GM0 and GM1 locality profiles must use different target addresses")

    matched_home_local = _matched_home_local_summary(profiles, config)
    comparisons: dict[str, Any] = {}
    tick_to_ns = 1_000_000_000.0 / config.sys_counter_hz
    minimum_gap_ns = config.min_home_classification_gap_ticks * tick_to_ns
    for gm_home in GM_HOMES:
        for role in ("aic", "aiv"):
            die0 = float(profiles[f"{role}_gm{gm_home}_die0"]["launch_median_p50_ns"])
            die1 = float(profiles[f"{role}_gm{gm_home}_die1"]["launch_median_p50_ns"])
            faster_die = 0 if die0 < die1 else 1
            gap = abs(die0 - die1)
            if faster_die != gm_home or gap < minimum_gap_ns:
                raise AnalysisError(
                    f"{role} GM{gm_home} home classification does not close: die0={die0}, die1={die1}, "
                    f"expected faster DIE{gm_home} with gap >= {minimum_gap_ns} ns"
                )
            slower = max(die0, die1)
            faster = min(die0, die1)
            comparisons[f"{role}_gm{gm_home}"] = {
                "role": role,
                "gm_home": gm_home,
                "die0_p50_ns": die0,
                "die1_p50_ns": die1,
                "die0_minus_die1_ns": die0 - die1,
                "absolute_gap_ns": gap,
                "faster_die": faster_die,
                "slower_over_faster_ratio": slower / faster,
            }
    return {
        "validation": {
            "status": "PASS",
            "rows": len(records),
            "profiles": len(LOCALITY_PROFILES),
            "launches": len(LOCALITY_PROFILES) * config.sweeps,
            "physical_die_checked_from_get_coreid": True,
            "reciprocal_home_classification": True,
            "minimum_home_gap_ns": minimum_gap_ns,
            "maximum_matched_home_local_gap_ns": (config.max_matched_home_local_gap_ticks * tick_to_ns),
        },
        "target_addresses": {f"gm{gm_home}": address for gm_home, address in target_addresses.items()},
        "profiles": profiles,
        "comparisons": comparisons,
        "matched_home_local": matched_home_local,
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile over an already independent sample level."""

    if not values:
        raise AnalysisError("cannot compute a percentile of no values")
    if not 0.0 <= quantile <= 1.0:
        raise AnalysisError(f"percentile quantile is outside [0, 1]: {quantile}")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _point_summaries(records: Sequence[WaveRecord], config: AnalysisConfig, scenario: str) -> list[dict[str, Any]]:
    tick_to_ns = 1_000_000_000.0 / config.sys_counter_hz
    result: list[dict[str, Any]] = []
    for n in config.supported_points[scenario]:
        point_rows = [
            record for record in records if record.scenario == scenario and record.n == n and record.measured == 1
        ]
        launch_medians: list[float] = []
        for sweep in range(config.sweeps):
            spans = [record.global_span_tick for record in point_rows if record.sweep == sweep]
            if len(spans) != config.measured_waves:
                raise AnalysisError(
                    f"{scenario} N={n} sweep={sweep}: expected {config.measured_waves} measured spans, "
                    f"found {len(spans)}"
                )
            launch_medians.append(float(statistics.median(spans)))
        p10_ticks = percentile(launch_medians, 0.10)
        p50_ticks = percentile(launch_medians, 0.50)
        p90_ticks = percentile(launch_medians, 0.90)
        expected_aic, expected_aiv = _active_roles(scenario, n)
        start_spreads = [record.start_spread_tick for record in point_rows]
        result.append(
            {
                "n": n,
                "aic_active": expected_aic,
                "aiv_active": expected_aiv,
                "launches": config.sweeps,
                "measured_waves_per_launch": config.measured_waves,
                "launch_median_p10_ticks": p10_ticks,
                "launch_median_p50_ticks": p50_ticks,
                "launch_median_p90_ticks": p90_ticks,
                "p10_ns": p10_ticks * tick_to_ns,
                "y_ns": p50_ticks * tick_to_ns,
                "p90_ns": p90_ticks * tick_to_ns,
                "start_spread_p50_ticks": percentile(start_spreads, 0.50),
                "start_spread_p90_ticks": percentile(start_spreads, 0.90),
                "start_spread_max_ticks": max(start_spreads),
            }
        )
    return result


def _linear_fit(points: list[dict[str, Any]], tick_resolution_ns: float) -> dict[str, Any]:
    if len(points) < 2:
        raise AnalysisError("linear fit requires at least two N points")
    xs = [float(point["n"]) for point in points]
    ys = [float(point["y_ns"]) for point in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0.0:
        raise AnalysisError("linear fit requires distinct N values")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean

    squared_errors = 0.0
    max_abs_residual = (-1.0, -1)
    max_abs_relative = (-1.0, -1)
    for point in points:
        predicted = intercept + slope * float(point["n"])
        residual = float(point["y_ns"]) - predicted
        relative = None if predicted == 0.0 else residual / predicted
        half_repeat_band = (float(point["p90_ns"]) - float(point["p10_ns"])) / 2.0
        scale = max(half_repeat_band, tick_resolution_ns)
        point["predicted_ns"] = predicted
        point["residual_ns"] = residual
        point["relative_residual"] = relative
        point["scaled_residual"] = residual / scale
        point["predicted_inside_p10_p90"] = point["p10_ns"] <= predicted <= point["p90_ns"]
        squared_errors += residual * residual
        if abs(residual) > max_abs_residual[0]:
            max_abs_residual = (abs(residual), int(point["n"]))
        if relative is not None and abs(relative) > max_abs_relative[0]:
            max_abs_relative = (abs(relative), int(point["n"]))

    total_variation = sum((value - y_mean) ** 2 for value in ys)
    if total_variation == 0.0:
        r_squared = 1.0 if squared_errors <= 1e-24 else 0.0
    else:
        r_squared = 1.0 - squared_errors / total_variation
    rmse = math.sqrt(squared_errors / len(points))
    t1 = intercept + slope
    a_over_b = None if slope == 0.0 else intercept / slope
    parallel_fraction = None if t1 == 0.0 else intercept / t1
    serial_fraction = None if t1 == 0.0 else slope / t1
    max_relative = None if max_abs_relative[0] < 0.0 else max_abs_relative[0]
    dynamic_range = max(ys) - min(ys)
    normalized_rmse = None if dynamic_range == 0.0 else rmse / dynamic_range
    normalized_max_residual = None if dynamic_range == 0.0 else max_abs_residual[0] / dynamic_range
    if intercept >= 0.0 and slope > 0.0 and r_squared >= 0.99 and max_relative is not None and max_relative <= 0.05:
        pointwise_label = "STRONG"
    elif slope > 0.0 and r_squared >= 0.95 and max_relative is not None and max_relative <= 0.10:
        pointwise_label = "APPROXIMATE"
    else:
        pointwise_label = "NOT_SUPPORTED"
    if (
        intercept >= 0.0
        and slope > 0.0
        and r_squared >= 0.999
        and normalized_rmse is not None
        and normalized_rmse <= 0.005
        and normalized_max_residual is not None
        and normalized_max_residual <= 0.01
    ):
        label = "STRONG"
    elif (
        slope > 0.0
        and r_squared >= 0.99
        and normalized_rmse is not None
        and normalized_rmse <= 0.02
        and normalized_max_residual is not None
        and normalized_max_residual <= 0.05
    ):
        label = "APPROXIMATE"
    else:
        label = "NOT_SUPPORTED"
    return {
        "model": "T(N) = A + B*N",
        "point_count": len(points),
        "a_ns": intercept,
        "b_ns_per_scalar": slope,
        "t1_ns": t1,
        "r_squared": r_squared,
        "rmse_ns": rmse,
        "dynamic_range_ns": dynamic_range,
        "normalized_rmse": normalized_rmse,
        "normalized_max_abs_residual": normalized_max_residual,
        "max_abs_residual_ns": max_abs_residual[0],
        "max_abs_residual_n": max_abs_residual[1],
        "max_abs_relative_residual": max_relative,
        "max_abs_relative_residual_n": max_abs_relative[1] if max_relative is not None else None,
        "a_over_b": a_over_b,
        "parallel_fraction_at_n1": parallel_fraction,
        "serial_fraction_at_n1": serial_fraction,
        "label": label,
        "strict_pointwise_label": pointwise_label,
    }


def analyze_records(records: Sequence[WaveRecord], config: AnalysisConfig | None = None) -> dict[str, Any]:
    """Validate raw rows, apply hierarchical medians, and fit each scenario."""

    active_config = config or AnalysisConfig()
    validation = validate_records(records, active_config)
    scenarios: dict[str, Any] = {}
    tick_resolution_ns = 1_000_000_000.0 / active_config.sys_counter_hz
    for scenario in SCENARIOS:
        points = _point_summaries(records, active_config, scenario)
        sensitivity_fits = {
            f"n_ge_{cutoff}": _linear_fit(
                [dict(point) for point in points if int(point["n"]) >= cutoff], tick_resolution_ns
            )
            for cutoff in (2, 4)
            if sum(int(point["n"]) >= cutoff for point in points) >= 2
        }
        scenarios[scenario] = {
            "label": SCENARIO_LABELS[scenario],
            "points": points,
            "fit": _linear_fit(points, tick_resolution_ns),
            "sensitivity_fits": sensitivity_fits,
        }
    comparisons: dict[str, Any] = {}
    for base in BASE_SCENARIOS:
        local_fit = scenarios[f"{base}_gm0"]["fit"]
        remote_fit = scenarios[f"{base}_gm1"]["fit"]
        delta_a = float(remote_fit["a_ns"]) - float(local_fit["a_ns"])
        delta_b = float(remote_fit["b_ns_per_scalar"]) - float(local_fit["b_ns_per_scalar"])
        equal_time_n = None if delta_b == 0.0 else -delta_a / delta_b
        if equal_time_n is not None and equal_time_n <= 0.0:
            equal_time_n = None
        comparisons[base] = {
            "label": BASE_SCENARIO_LABELS[base],
            "gm1_minus_gm0_a_ns": delta_a,
            "gm1_minus_gm0_b_ns_per_scalar": delta_b,
            "gm1_over_gm0_b_ratio": float(remote_fit["b_ns_per_scalar"]) / float(local_fit["b_ns_per_scalar"]),
            "gm1_minus_gm0_t1_ns": float(remote_fit["t1_ns"]) - float(local_fit["t1_ns"]),
            "equal_time_n": equal_time_n,
        }
    unsupported = [
        {"scenario": scenario, "n": n, "reason": "physical_scalar_capacity"}
        for scenario in SCENARIOS
        for n in active_config.unsupported_points[scenario]
    ]
    return {
        "schema": {"name": "atomic-contention-curve-analysis", "version": 5},
        "configuration": {
            "sys_counter_hz": active_config.sys_counter_hz,
            "tick_resolution_ns": tick_resolution_ns,
            "sweeps": active_config.sweeps,
            "warmup_waves_per_launch": active_config.warmup_waves,
            "measured_waves_per_launch": active_config.measured_waves,
            "launch_statistic": "median(global_span_tick across measured waves)",
            "point_statistic": "p10/p50/p90 across launch medians with linear interpolation",
            "fit_weighting": "equal weight per N point",
            "global_span_definition": "max(active end_tick) - min(active begin_tick)",
            "hardware_placement_column": "active_hardware_core_ids as a compact JSON integer array",
            "target_address_column": "device virtual address of the isolated 64-byte atomic line",
            "hardware_role_topology": "A5 108 slots: 54 per die, first 18 slots per die are AIC",
            "curve_scope": "all active AIC/AIV participants are verified on physical DIE0",
            "start_spread_policy": (
                f"fail measured waves if max(begin_tick)-min(begin_tick) > {active_config.max_start_spread_ticks}"
            ),
            "wave_stride_ticks": active_config.wave_stride_ticks,
            "wave_publish_offset_ticks": active_config.wave_publish_offset_ticks,
            "publication_policy": (
                "measurement must finish before the fixed publication phase; "
                "publication must finish before the next wave"
            ),
            "fit_labels": {
                "STRONG": "A>=0, B>0, R^2>=0.999, normalized RMSE<=0.5%, normalized max residual<=1%",
                "APPROXIMATE": "B>0, R^2>=0.99, normalized RMSE<=2%, normalized max residual<=5%",
                "NOT_SUPPORTED": "otherwise",
                "strict_pointwise_label": (
                    "secondary diagnostic: STRONG requires max point-relative residual<=5%; APPROXIMATE requires <=10%"
                ),
            },
        },
        "validation": validation,
        "scenarios": scenarios,
        "gm1_vs_gm0": comparisons,
        "unsupported": unsupported,
    }


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]], right_aligned: set[int]) -> list[str]:
    """Render a compact table whose source pipes satisfy aligned-delimiter lint."""

    text_rows = [[str(value) for value in row] for row in rows]

    def display_width(value: str) -> int:
        return sum(2 if unicodedata.east_asian_width(character) in "FW" else 1 for character in value)

    text_headers = [str(header) for header in headers]
    delimiters = [
        "-" * (display_width(header) - 1) + ":" if index in right_aligned else "-" * display_width(header)
        for index, header in enumerate(text_headers)
    ]

    def render_row(row: Sequence[str]) -> str:
        return "| " + " | ".join(row) + " |"

    return [render_row(text_headers), render_row(delimiters), *(render_row(row) for row in text_rows)]


def render_markdown(summary: Mapping[str, Any]) -> str:
    """Render the paired local/remote GM fits and their validation evidence."""

    fits = [summary["scenarios"][scenario]["fit"] for scenario in SCENARIOS]
    configuration = summary["configuration"]
    placement = summary["validation"]["hardware_placement"]
    min_r_squared = min(float(fit["r_squared"]) for fit in fits)
    delta_as = [float(entry["gm1_minus_gm0_a_ns"]) for entry in summary["gm1_vs_gm0"].values()]
    delta_bs = [float(entry["gm1_minus_gm0_b_ns_per_scalar"]) for entry in summary["gm1_vs_gm0"].values()]
    delta_t1s = [float(entry["gm1_minus_gm0_t1_ns"]) for entry in summary["gm1_vs_gm0"].values()]
    max_b_ratio_deviation = max(
        abs(float(entry["gm1_over_gm0_b_ratio"]) - 1.0) for entry in summary["gm1_vs_gm0"].values()
    )
    if max_b_ratio_deviation <= 0.02:
        parameter_interpretation = (
            "三类 B 的相对差均不超过 2%，应视为斜率近似相同；本次同 DIE/跨 DIE 差异主要体现在 A。"
        )
    elif min(delta_as) > 0.0 and min(delta_bs) > 0.0:
        parameter_interpretation = "跨 DIE 的固定项和边际项都更高，差距随 N 增大。"
    elif min(delta_as) > 0.0 and max(delta_bs) < 0.0:
        parameter_interpretation = "跨 DIE 的固定项更高、边际项更低，差距随 N 缩小，并可能在大 N 处交叉。"
    else:
        parameter_interpretation = "三类 Scalar 的参数变化方向不完全一致，应分别查看下表，不能合并成单一常数。"
    fit_statement = (
        "六条曲线全部达到强拟合门槛" if all(fit["label"] == "STRONG" for fit in fits) else "六条曲线的判定见下表"
    )
    lines = [
        "# DIE0 Core 访问 GM0/GM1 的同地址 Atomic 并发曲线",
        "",
        "本文中的 GM0 是经双向 N=1 镜像时延判定为 DIE0 本地的目标地址；GM1 是用同一方法判定为",
        "DIE1 本地的目标地址。因此，DIE0 Core → GM0 是同 DIE，DIE0 Core → GM1 是跨 DIE。",
        "该编号是本探针的操作性命名，不是 CANN 接口返回的物理 HA 编号。",
        "",
        "每条曲线的所有活跃 Scalar 都由 `get_coreid` 确认在物理 DIE0。主观测量是一个 wave 的",
        f"`max(active end_tick) - min(active begin_tick)`；先取每次 launch 内 "
        f"{configuration['measured_waves_per_launch']} 个计量 wave 的中位数，",
        f"再对 {configuration['sweeps']} 次 launch 取中位数。分析不删除性能离群样本。",
        "",
        "## 直接结论",
        "",
        (
            f"{fit_statement}，最低 R² 为 {min_r_squared:.6f}。"
            f"在发起 Core 固定为 DIE0 后，将目标从 GM0 换到 GM1，ΔA 为 "
            f"{min(delta_as):+.3f} 至 {max(delta_as):+.3f} ns，ΔB 为 "
            f"{min(delta_bs):+.3f} 至 {max(delta_bs):+.3f} ns/Scalar。"
        ),
        "",
        parameter_interpretation,
        "",
        (
            "不能只比较斜率 B：N=1 的拟合值是 `T(1)=A+B`，所以跨 DIE 的 N=1 差值是 "
            f"`ΔT(1)=ΔA+ΔB`，三类场景为 {min(delta_t1s):.3f} 至 {max(delta_t1s):.3f} ns。"
        ),
        "",
        (
            f"Raw 中 {placement['rows']} 行、{placement['launches']} 个曲线 launch 的物理 placement、"
            "目标地址一致性、返回旧值全排列和发布阶段边界均已闭合。"
        ),
        "",
        "## 拟合汇总",
        "",
    ]
    fit_rows: list[list[str]] = []
    for scenario in SCENARIOS:
        entry = summary["scenarios"][scenario]
        fit = entry["fit"]
        fit_rows.append(
            [
                str(entry["label"]),
                str(fit["point_count"]),
                _format_number(fit["a_ns"]),
                _format_number(fit["b_ns_per_scalar"]),
                _format_number(fit["t1_ns"]),
                _format_number(fit["r_squared"], 6),
                _format_number(fit["rmse_ns"]),
                FIT_LABELS_ZH[fit["label"]],
            ]
        )
    lines.extend(
        _markdown_table(
            ["场景", "点数", "A（ns）", "B（ns/Scalar）", "T(1)（ns）", "R²", "RMSE（ns）", "判定"],
            fit_rows,
            {1, 2, 3, 4, 5, 6},
        )
    )
    delta_rows = [
        [
            str(entry["label"]),
            _format_number(entry["gm1_minus_gm0_a_ns"]),
            _format_number(entry["gm1_minus_gm0_b_ns_per_scalar"]),
            _format_number(entry["gm1_over_gm0_b_ratio"]),
            _format_number(entry["gm1_minus_gm0_t1_ns"]),
            _format_number(entry["equal_time_n"]),
        ]
        for entry in summary["gm1_vs_gm0"].values()
    ]
    lines.extend(["", "## GM1 相对 GM0 的参数变化", ""])
    lines.extend(
        _markdown_table(
            ["场景", "ΔA（ns）", "ΔB（ns/Scalar）", "B1/B0", "ΔT(1)（ns）", "拟合交点 N"],
            delta_rows,
            {1, 2, 3, 4, 5},
        )
    )

    locality = summary.get("die_locality")
    if locality is not None:
        locality_rows: list[list[str]] = []
        for profile in LOCALITY_PROFILES:
            entry = locality["profiles"][profile]
            locality_rows.append(
                [
                    str(entry["role"]).upper(),
                    f"GM{entry['gm_home']}",
                    f"DIE{entry['core_die']}",
                    ",".join(str(core_id) for core_id in entry["hardware_core_ids"]),
                    f"0x{int(entry['target_address']):x}",
                    _format_number(entry["launch_median_p10_ns"]),
                    _format_number(entry["launch_median_p50_ns"]),
                    _format_number(entry["launch_median_p90_ns"]),
                ]
            )
        lines.extend(
            [
                "",
                "## GM 归属的镜像校验",
                "",
                "GM0/GM1 不由未公开的地址位规则推测，而由固定 Core 在两个 DIE 上的 N=1 镜像时延分类。",
                (
                    "AIC 和 AIV 均必须满足 GM0 在 DIE0 更快、GM1 在 DIE1 更快，且中位数差值不小于 "
                    f"{locality['validation']['minimum_home_gap_ns']:.0f} ns。"
                ),
                (
                    "为避免把不同地址档位误当成 DIE 差异，GM0 与 GM1 的各自本地 N=1 中位数差值还必须"
                    f"不超过 {locality['validation']['maximum_matched_home_local_gap_ns']:.0f} ns。"
                ),
                "",
            ]
        )
        lines.extend(
            _markdown_table(
                ["Scalar", "目标", "Core DIE", "`get_coreid`", "目标地址", "P10（ns）", "中位数（ns）", "P90（ns）"],
                locality_rows,
                {3, 5, 6, 7},
            )
        )
        matched_rows = [
            [
                role.upper(),
                _format_number(entry["gm0_home_local_p50_ns"]),
                _format_number(entry["gm1_home_local_p50_ns"]),
                _format_number(entry["absolute_gap_ns"]),
            ]
            for role, entry in locality["matched_home_local"].items()
        ]
        lines.extend(["", "### GM0/GM1 本地基准匹配", ""])
        lines.extend(
            _markdown_table(
                ["Scalar", "GM0 本地（ns）", "GM1 本地（ns）", "绝对差（ns）"],
                matched_rows,
                {1, 2, 3},
            )
        )

    for base in BASE_SCENARIOS:
        lines.extend(["", f"## {BASE_SCENARIO_LABELS[base]} 逐点结果", ""])
        point_rows: list[list[str]] = []
        for gm_home in GM_HOMES:
            entry = summary["scenarios"][f"{base}_gm{gm_home}"]
            for point in entry["points"]:
                point_rows.append(
                    [
                        f"GM{gm_home}",
                        str(point["n"]),
                        str(point["aic_active"]),
                        str(point["aiv_active"]),
                        _format_number(point["p10_ns"]),
                        _format_number(point["y_ns"]),
                        _format_number(point["p90_ns"]),
                        _format_number(point["predicted_ns"]),
                        _format_number(point["residual_ns"]),
                    ]
                )
        lines.extend(
            _markdown_table(
                ["目标", "`N`", "AIC", "AIV", "P10（ns）", "中位数（ns）", "P90（ns）", "预测值（ns）", "残差（ns）"],
                point_rows,
                set(range(1, 9)),
            )
        )

    unsupported = summary.get("unsupported", [])
    if unsupported:
        unsupported_rows = [
            [SCENARIO_LABELS[row["scenario"]], str(row["n"]), "超出 DIE0 可用 Scalar 容量"] for row in unsupported
        ]
        lines.extend(["", "## 不支持的点", ""])
        lines.extend(_markdown_table(["场景", "`N`", "原因"], unsupported_rows, {1}))
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "GM0/GM1 是本用例根据双向 N=1 时延定义的操作性归属，不是由 CANN 公开接口返回的 HA 元数据。",
            "强拟合证明 `A+B*N` 是本次设备、二进制和测试形状下的良好一阶模型，但不能单独把 A 和 B",
            "精确分解为某一级 SoC 互连或 HA 内部阶段。",
            "",
        ]
    )
    return "\n".join(lines)


def render_svg(summary: Mapping[str, Any]) -> str:
    """Render one paired GM0/GM1 panel for each Scalar composition."""

    width = 1260
    height = 500
    outer = 36
    gap = 28
    panel_width = (width - 2 * outer - 2 * gap) / 3.0
    panel_height = 320.0
    plot_top = 92.0
    plot_bottom = plot_top + panel_height
    colors = {0: "#2563eb", 1: "#dc2626"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>DIE0 Core 访问 GM0/GM1 的同地址 atomic 并发拟合</title>",
        "<desc>三个面板对比纯 AIC、纯 AIV 和混合 Scalar 从 DIE0 访问 GM0 与 GM1 的曲线。</desc>",
        "<style>",
        ".axis{stroke:#334155;stroke-width:1}.grid{stroke:#e2e8f0;stroke-width:1}.fit{fill:none;stroke-width:2}",
        ".point{stroke:#fff;stroke-width:1.5}.band{stroke-width:1.5}.label{font:12px sans-serif;fill:#334155}",
        ".title{font:bold 16px sans-serif;fill:#0f172a}.metric{font:11px monospace;fill:#475569}",
        "</style>",
        '<text x="36" y="28" class="title">DIE0 Core → GM0/GM1：T(N) = A + B*N</text>',
        '<line x1="885" y1="23" x2="909" y2="23" stroke="#2563eb" stroke-width="3"/>',
        '<text x="915" y="27" class="label">GM0（DIE0 本地）</text>',
        '<line x1="1052" y1="23" x2="1076" y2="23" stroke="#dc2626" stroke-width="3"/>',
        '<text x="1082" y="27" class="label">GM1（DIE1 本地）</text>',
    ]
    for index, base in enumerate(BASE_SCENARIOS):
        entries = {gm_home: summary["scenarios"][f"{base}_gm{gm_home}"] for gm_home in GM_HOMES}
        all_points = [point for entry in entries.values() for point in entry["points"]]
        panel_left = outer + index * (panel_width + gap)
        plot_left = panel_left + 52.0
        plot_right = panel_left + panel_width - 14.0
        xs = [float(point["n"]) for point in all_points]
        y_values = [float(point[key]) for point in all_points for key in ("p10_ns", "p90_ns", "predicted_ns")]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(0.0, *y_values)
        y_max = max(y_values)
        if x_max == x_min:
            x_max += 1.0
        if y_max == y_min:
            y_max += 1.0
        y_max += 0.08 * (y_max - y_min)

        def sx(value: float) -> float:
            return plot_left + (value - x_min) * (plot_right - plot_left) / (x_max - x_min)

        def sy(value: float) -> float:
            return plot_bottom - (value - y_min) * panel_height / (y_max - y_min)

        elements.append(f'<g class="panel" data-scenario="{base}">')
        elements.append(
            f'<text x="{panel_left + panel_width / 2:.1f}" y="72" text-anchor="middle" class="title">'
            f"{html.escape(BASE_SCENARIO_LABELS[base])}</text>"
        )
        for tick in range(5):
            value = y_min + (y_max - y_min) * tick / 4.0
            y = sy(value)
            elements.append(
                f'<line x1="{plot_left:.1f}" y1="{y:.1f}" x2="{plot_right:.1f}" y2="{y:.1f}" class="grid"/>'
            )
            elements.append(
                f'<text x="{plot_left - 7:.1f}" y="{y + 4:.1f}" text-anchor="end" class="label">{value:.0f}</text>'
            )
        elements.append(
            f'<line x1="{plot_left:.1f}" y1="{plot_top:.1f}" x2="{plot_left:.1f}" y2="{plot_bottom:.1f}" class="axis"/>'
        )
        elements.append(
            f'<line x1="{plot_left:.1f}" y1="{plot_bottom:.1f}" x2="{plot_right:.1f}" '
            f'y2="{plot_bottom:.1f}" class="axis"/>'
        )
        for gm_home in GM_HOMES:
            entry = entries[gm_home]
            color = colors[gm_home]
            coordinates = " ".join(
                f"{sx(float(point['n'])):.1f},{sy(float(point['predicted_ns'])):.1f}" for point in entry["points"]
            )
            elements.append(f'<polyline points="{coordinates}" class="fit" stroke="{color}"/>')
            for point in entry["points"]:
                x = sx(float(point["n"]))
                low = sy(float(point["p10_ns"]))
                high = sy(float(point["p90_ns"]))
                center = sy(float(point["y_ns"]))
                elements.append(
                    f'<line x1="{x:.1f}" y1="{low:.1f}" x2="{x:.1f}" y2="{high:.1f}" class="band" stroke="{color}"/>'
                )
                elements.append(f'<circle cx="{x:.1f}" cy="{center:.1f}" r="3.8" class="point" fill="{color}"/>')
        for point in entries[0]["points"]:
            x = sx(float(point["n"]))
            elements.append(
                f'<text x="{x:.1f}" y="{plot_bottom + 17:.1f}" text-anchor="middle" class="label">{point["n"]}</text>'
            )
        elements.append(
            f'<text x="{panel_left + panel_width / 2:.1f}" y="{plot_bottom + 38:.1f}" '
            'text-anchor="middle" class="label">并发 Scalar 数（N）</text>'
        )
        for line_index, gm_home in enumerate(GM_HOMES):
            fit = entries[gm_home]["fit"]
            elements.append(
                f'<text x="{plot_left:.1f}" y="{plot_bottom + 58 + line_index * 17:.1f}" class="metric" '
                f'style="fill:{colors[gm_home]}">GM{gm_home}: A={fit["a_ns"]:.2f}  '
                f"B={fit['b_ns_per_scalar']:.2f}  R²={fit['r_squared']:.6f}</text>"
            )
        elements.append("</g>")
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def write_artifacts(summary: Mapping[str, Any], summary_path: Path, svg_path: Path, report_path: Path) -> None:
    """Publish all derived artifacts only after analysis has completed."""

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    svg_path.write_text(render_svg(summary), encoding="utf-8")
    report_path.write_text(render_markdown(summary), encoding="utf-8")


def _record_fields(record: WaveRecord) -> tuple[Any, ...]:
    return (
        record.scenario,
        record.sweep,
        record.n,
        record.aic_active,
        record.aiv_active,
        json.dumps(record.active_hardware_core_ids, separators=(",", ":")),
        record.target_address,
        record.wave,
        record.measured,
        record.deadline_tick,
        record.min_begin_tick,
        record.max_begin_tick,
        record.max_end_tick,
        record.start_spread_tick,
        record.global_span_tick,
        record.max_worker_elapsed_tick,
        record.min_publish_begin_tick,
        record.max_publish_ready_tick,
        record.old_min,
        record.old_max,
        record.old_sum,
        record.old_xor,
        record.status,
    )


def write_raw_bundle(records: Sequence[WaveRecord], output_directory: Path) -> list[Path]:
    """Write one compact, reviewable raw TSV stream per physical scenario."""

    output_directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for scenario in SCENARIOS:
        output = output_directory / f"raw_{scenario}.tsv.xz"
        with lzma.open(output, "wt", encoding="utf-8", newline="", preset=9) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(TSV_COLUMNS)
            writer.writerows(_record_fields(record) for record in records if record.scenario == scenario)
        outputs.append(output)
    locality_records = [record for record in records if record.scenario in LOCALITY_PROFILES]
    if locality_records:
        output = output_directory / "raw_locality.tsv.xz"
        with lzma.open(output, "wt", encoding="utf-8", newline="", preset=9) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(TSV_COLUMNS)
            writer.writerows(_record_fields(record) for record in locality_records)
        outputs.append(output)
    return outputs


def _default_artifact_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}{suffix}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly validate and fit the A5 same-address atomic contention TSV.")
    parser.add_argument("input", type=Path, nargs="+", help="raw TSV or per-scenario TSV.xz from the device runner")
    parser.add_argument("--summary", "--summary-json", dest="summary_json", type=Path, help="output summary JSON path")
    parser.add_argument("--svg", type=Path, help="output three-panel SVG path")
    parser.add_argument("--report", type=Path, help="output Markdown report path")
    parser.add_argument("--source-revision", help="source revision recorded in the summary")
    parser.add_argument("--raw-bundle-dir", type=Path, help="write compressed per-scenario raw evidence here")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    primary_input = args.input[0]
    summary_path = args.summary_json or _default_artifact_path(primary_input, "_summary.json")
    svg_path = args.svg or _default_artifact_path(primary_input, "_fit.svg")
    report_path = args.report or _default_artifact_path(primary_input, "_report.md")
    try:
        records = [record for input_path in args.input for record in load_tsv(input_path)]
        curve_records = [record for record in records if record.scenario in SCENARIOS]
        locality_records = [record for record in records if record.scenario in LOCALITY_PROFILES]
        unknown_scenarios = sorted({record.scenario for record in records} - set(SCENARIOS) - set(LOCALITY_PROFILES))
        if unknown_scenarios:
            raise AnalysisError(f"unknown scenario names: {unknown_scenarios}")
        summary = analyze_records(curve_records)
        if locality_records:
            locality = analyze_die_locality(locality_records, AnalysisConfig())
            if locality["target_addresses"] != summary["validation"]["target_addresses"]:
                raise AnalysisError(
                    "curve and reciprocal locality rows disagree on GM0/GM1 target addresses: "
                    f"curve={summary['validation']['target_addresses']}, locality={locality['target_addresses']}"
                )
            summary["die_locality"] = locality
        summary["source_tsv"] = [input_path.name for input_path in args.input]
        if args.source_revision:
            summary["source_revision"] = args.source_revision
        write_artifacts(summary, summary_path, svg_path, report_path)
        if args.raw_bundle_dir:
            write_raw_bundle(records, args.raw_bundle_dir)
    except (AnalysisError, OSError) as error:
        print(f"atomic contention analysis failed: {error}", file=sys.stderr)
        return 2
    print(
        f"[ATOMIC_CURVE] validation=PASS rows={len(records)} summary={summary_path} svg={svg_path} report={report_path}"
    )
    for scenario in SCENARIOS:
        fit = summary["scenarios"][scenario]["fit"]
        print(
            f"[ATOMIC_CURVE] scenario={scenario} A_ns={fit['a_ns']:.6f} "
            f"B_ns_per_scalar={fit['b_ns_per_scalar']:.6f} R2={fit['r_squared']:.9f} "
            f"RMSE_ns={fit['rmse_ns']:.6f} label={fit['label']}"
        )
    if "die_locality" in summary:
        for profile, comparison in summary["die_locality"]["comparisons"].items():
            print(
                f"[ATOMIC_LOCALITY] profile={profile} die0_ns={comparison['die0_p50_ns']:.3f} "
                f"die1_ns={comparison['die1_p50_ns']:.3f} "
                f"delta_ns={comparison['die0_minus_die1_ns']:.3f} faster_die={comparison['faster_die']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
