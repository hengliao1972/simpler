# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the LICENSE file in the root directory of this source tree for more details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Atomic 不同地址 bank 冲突分析器的回归测试。"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

try:
    from .analyze_atomic_bank_conflict import (
        SCENARIOS,
        TSV_COLUMNS,
        TWO_GROUP_PATTERNS,
        AnalysisConfig,
        AnalysisError,
        WaveRecord,
        analyze_records,
        load_tsv,
        write_artifacts,
        write_raw_bundle,
    )
except ImportError:
    from analyze_atomic_bank_conflict import (
        SCENARIOS,
        TSV_COLUMNS,
        TWO_GROUP_PATTERNS,
        AnalysisConfig,
        AnalysisError,
        WaveRecord,
        analyze_records,
        load_tsv,
        write_artifacts,
        write_raw_bundle,
    )


def _core_ids(scenario: str, n: int) -> tuple[int, ...]:
    if scenario == "aic":
        return tuple(range(n))
    if scenario == "aiv":
        return tuple(range(18, 18 + n))
    result: list[int] = []
    for block in range((n + 2) // 3):
        result.extend((block, 18 + block * 2, 19 + block * 2))
    return tuple(result[:n])


def _roles(scenario: str, n: int) -> tuple[int, int]:
    if scenario == "aic":
        return n, 0
    if scenario == "aiv":
        return 0, n
    aic = (n + 2) // 3
    return aic, n - aic


def _two_group_sizes(scenario: str, n: int) -> tuple[int, int]:
    aic, aiv = _roles(scenario, n)
    if scenario == "mixed":
        return aic, aiv
    primary = (n + 1) // 2
    return primary, n - primary


def _xor_upto(value: int) -> int:
    if value < 0:
        return 0
    return (value, 1, value + 1, 0)[value & 3]


def _record(
    config: AnalysisConfig,
    experiment: str,
    scenario: str,
    sweep: int,
    parameter: int,
    n: int,
    wave: int,
    span: int,
) -> WaveRecord:
    base = 0x200000
    two_group_pattern = TWO_GROUP_PATTERNS.get(experiment)
    phased = experiment in ("single_address", "pair_phase_same", "pair_phase_64")
    target_offset = int(two_group_pattern["target_offset_bytes"]) if two_group_pattern else (parameter if phased else 0)
    if two_group_pattern:
        stride = int(two_group_pattern["target_stride_bytes"])
        address_layout = str(two_group_pattern["address_layout"])
    elif experiment in ("single_address", "pair_phase_same"):
        stride = 0
        address_layout = "shared"
    elif experiment == "pair_phase_64":
        stride = 64
        address_layout = "participant_stride"
    else:
        stride = parameter
        address_layout = "shared" if stride == 0 else "participant_stride"
    shared = address_layout == "shared"
    two_groups = address_layout == "two_groups"
    aic, aiv = _roles(scenario, n)
    deadline = 1_000_000_000 + sweep * 10_000_000 + wave * config.wave_stride_ticks
    min_begin = deadline + 2
    max_begin = min_begin + 4
    if shared:
        first = wave * n
        last = first + n - 1
        old = (first, last, (first + last) * n // 2, _xor_upto(last) ^ _xor_upto(first - 1))
        unique_addresses = 1
        target_span = 64
    elif two_groups:
        primary, secondary = _two_group_sizes(scenario, n)
        primary_first = wave * primary
        primary_last = primary_first + primary - 1
        secondary_first = wave * secondary
        secondary_last = secondary_first + secondary - 1
        old = (
            min(primary_first, secondary_first),
            max(primary_last, secondary_last),
            (primary_first + primary_last) * primary // 2 + (secondary_first + secondary_last) * secondary // 2,
            _xor_upto(primary_last)
            ^ _xor_upto(primary_first - 1)
            ^ _xor_upto(secondary_last)
            ^ _xor_upto(secondary_first - 1),
        )
        unique_addresses = 2
        target_span = stride + 64
    else:
        old = (wave, wave, wave * n, wave if n % 2 else 0)
        unique_addresses = n
        target_span = (n - 1) * stride + 64
    return WaveRecord(
        experiment=experiment,
        scenario=scenario,
        sweep=sweep,
        n=n,
        aic_active=aic,
        aiv_active=aiv,
        hardware_core_ids=_core_ids(scenario, n),
        target_address=base + target_offset,
        target_offset_bytes=target_offset,
        address_layout=address_layout,
        target_stride_bytes=stride,
        unique_addresses=unique_addresses,
        target_span_bytes=target_span,
        wave=wave,
        measured=int(wave >= config.warmup_waves),
        deadline_tick=deadline,
        min_begin_tick=min_begin,
        max_begin_tick=max_begin,
        max_end_tick=min_begin + span,
        start_spread_tick=4,
        global_span_tick=span,
        max_worker_elapsed_tick=span - 1,
        min_publish_begin_tick=deadline + config.wave_publish_offset_ticks + 2,
        max_publish_ready_tick=deadline + config.wave_publish_offset_ticks + 100,
        old_min=old[0],
        old_max=old[1],
        old_sum=old[2],
        old_xor=old[3],
        status="PASS",
    )


def _append_two_group_records(
    rows: list[WaveRecord],
    config: AnalysisConfig,
    sweep: int,
    sweep_jitter: int,
    scenario: str,
    scenario_offset: int,
) -> None:
    for experiment, pattern in TWO_GROUP_PATTERNS.items():
        for n in config.populations[scenario]:
            if n < 2:
                continue
            primary, secondary = _two_group_sizes(scenario, n)
            merged = experiment in ("two_group_same", "two_group_64_same128")
            queue_depth = n if merged else max(primary, secondary)
            span = 80 + scenario_offset + 170 * queue_depth + sweep_jitter
            for wave in range(config.total_waves):
                rows.append(
                    _record(
                        config,
                        experiment,
                        scenario,
                        sweep,
                        int(pattern["target_stride_bytes"]),
                        n,
                        wave,
                        span,
                    )
                )


def _records(config: AnalysisConfig) -> list[WaveRecord]:
    rows: list[WaveRecord] = []
    for sweep in range(config.sweeps):
        sweep_jitter = (-2, 0, 2)[sweep]
        for gap in config.gaps:
            for scenario, single_latency in (("aic", 200), ("aiv", 210)):
                for wave in range(config.total_waves):
                    rows.append(
                        _record(config, "single_address", scenario, sweep, gap, 1, wave, single_latency + sweep_jitter)
                    )
            for scenario in SCENARIOS:
                baseline = 200 if scenario == "aic" else 210
                if scenario == "mixed":
                    baseline = 210
                extra = 170 if gap == 0 else (165 if gap < 512 else 10)
                for wave in range(config.total_waves):
                    rows.append(
                        _record(config, "pair_gap", scenario, sweep, gap, 2, wave, baseline + extra + sweep_jitter)
                    )
        for phase in config.phases:
            for scenario in SCENARIOS:
                baseline = 200 if scenario == "aic" else 210
                if scenario == "mixed":
                    baseline = 210
                adjacent_extra = 170 if phase % 128 == 0 else 10
                for wave in range(config.total_waves):
                    rows.append(
                        _record(
                            config,
                            "pair_phase_same",
                            scenario,
                            sweep,
                            phase,
                            2,
                            wave,
                            baseline + 170 + sweep_jitter,
                        )
                    )
                    rows.append(
                        _record(
                            config,
                            "pair_phase_64",
                            scenario,
                            sweep,
                            phase,
                            2,
                            wave,
                            baseline + adjacent_extra + sweep_jitter,
                        )
                    )
        for scenario in SCENARIOS:
            scenario_offset = {"aic": 0, "aiv": 5, "mixed": 10}[scenario]
            for stride in config.strides:
                slope = {0: 170, 64: 160, 128: 40, 512: 20, 1024: 5, 4096: 145}[stride]
                for n in config.populations[scenario]:
                    span = 80 + scenario_offset + slope * n + sweep_jitter
                    for wave in range(config.total_waves):
                        rows.append(_record(config, "multicore_curve", scenario, sweep, stride, n, wave, span))
            _append_two_group_records(rows, config, sweep, sweep_jitter, scenario, scenario_offset)
    return rows


def _write_tsv(path: Path, records: list[WaveRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        for record in records:
            writer.writerow(
                (
                    record.experiment,
                    record.scenario,
                    record.sweep,
                    record.n,
                    record.aic_active,
                    record.aiv_active,
                    json.dumps(record.hardware_core_ids, separators=(",", ":")),
                    record.target_address,
                    record.target_offset_bytes,
                    record.address_layout,
                    record.target_stride_bytes,
                    record.unique_addresses,
                    record.target_span_bytes,
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
            )


class AtomicBankConflictAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AnalysisConfig(
            sweeps=3,
            warmup_waves=1,
            measured_waves=4,
            gaps=(0, 64, 128, 192, 256, 320, 384, 448, 512, 1024),
            phases=(0, 64, 128, 192),
            strides=(0, 64, 128, 512, 1024, 4096),
            populations={scenario: (1, 2, 4, 8) for scenario in SCENARIOS},
        )
        self.records = _records(self.config)

    def test_supports_same_512b_region_serialization_and_boundary_release(self) -> None:
        summary = analyze_records(self.records, self.config, source_revision="synthetic")
        self.assertEqual(summary["validation"]["status"], "PASS")
        self.assertEqual(summary["pair_gap"]["verdict"], "SUPPORTED")
        self.assertTrue(summary["pair_phase"]["observed_128b_conflict_signature"])
        self.assertEqual(summary["two_group"]["verdict"], "VERIFIED")
        for scenario in SCENARIOS:
            pair = summary["pair_gap"]["scenarios"][scenario]
            self.assertGreater(pair["within_first_512b_serialization_ratio_p50"], 0.9)
            self.assertLess(pair["gap_512b_serialization_ratio_p50"], 0.1)
            multicore = summary["multicore"]["scenarios"][scenario]
            self.assertAlmostEqual(multicore["dense_64b_low_n_b_ratio_to_shared"], 160 / 170)
            self.assertAlmostEqual(multicore["stride_512b_low_n_b_ratio_to_shared"], 20 / 170)
            two_group = summary["two_group"]["scenarios"][scenario]
            self.assertAlmostEqual(two_group["same_128b_b_total_ratio"], 1.0)
            self.assertAlmostEqual(two_group["cross_128b_b_max_group_ratio"], 1.0)
            self.assertAlmostEqual(two_group["gap_128b_b_max_group_ratio"], 1.0)
            self.assertAlmostEqual(two_group["gap_512b_b_max_group_ratio"], 1.0)
            self.assertAlmostEqual(two_group["gap_1024b_b_max_group_ratio"], 1.0)

    def test_rejects_corrupted_independent_address_return_oracle(self) -> None:
        index = next(
            index
            for index, record in enumerate(self.records)
            if record.experiment == "pair_gap" and record.target_stride_bytes == 64 and record.wave == 2
        )
        corrupted = list(self.records)
        corrupted[index] = replace(corrupted[index], old_max=99)
        with self.assertRaisesRegex(AnalysisError, "atomic 返回值"):
            analyze_records(corrupted, self.config)

    def test_rejects_corrupted_two_group_return_oracle(self) -> None:
        index = next(
            index
            for index, record in enumerate(self.records)
            if record.experiment == "two_group_128"
            and record.scenario == "mixed"
            and record.n == 8
            and record.wave == 2
        )
        corrupted = list(self.records)
        corrupted[index] = replace(corrupted[index], old_sum=corrupted[index].old_sum + 1)
        with self.assertRaisesRegex(AnalysisError, "atomic 返回值"):
            analyze_records(corrupted, self.config)

    def test_tsv_bundle_and_artifacts_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.tsv"
            bundle = root / "raw.tsv.xz"
            _write_tsv(raw, self.records)
            loaded = load_tsv(raw)
            self.assertEqual(len(loaded), len(self.records))
            summary = analyze_records(loaded, self.config, source_revision="synthetic")
            write_raw_bundle(raw, bundle)
            summary_path = root / "summary.json"
            svg_path = root / "curve.svg"
            report_path = root / "report.md"
            write_artifacts(summary, summary_path, svg_path, report_path, bundle)
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8"))["validation"]["status"], "PASS")
            ET.parse(svg_path)
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("## 结论", report)
            self.assertIn("SHA-256", report)
            self.assertGreater(bundle.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
