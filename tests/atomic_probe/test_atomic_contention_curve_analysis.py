# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression tests for the dependency-free atomic contention curve analyzer."""

from __future__ import annotations

import csv
import json
import operator
import tempfile
import unittest
import xml.etree.ElementTree as ET
from dataclasses import replace
from functools import reduce
from pathlib import Path
from typing import Callable

try:
    from .analyze_atomic_contention_curve import (
        SCENARIOS,
        TSV_COLUMNS,
        AnalysisConfig,
        AnalysisError,
        WaveRecord,
        analyze_die_locality,
        analyze_records,
        load_tsv,
        write_artifacts,
        write_raw_bundle,
    )
except ImportError:
    from analyze_atomic_contention_curve import (
        SCENARIOS,
        TSV_COLUMNS,
        AnalysisConfig,
        AnalysisError,
        WaveRecord,
        analyze_die_locality,
        analyze_records,
        load_tsv,
        write_artifacts,
        write_raw_bundle,
    )


def _active_roles(scenario: str, n: int) -> tuple[int, int]:
    base = scenario.split("_gm", maxsplit=1)[0]
    if base == "aic":
        return n, 0
    if base == "aiv":
        return 0, n
    aic = (n + 2) // 3
    return aic, n - aic


def _xor_range(first: int, last: int) -> int:
    return reduce(operator.xor, range(first, last + 1), 0)


def _hardware_core_ids(scenario: str, n: int) -> tuple[int, ...]:
    base = scenario.split("_gm", maxsplit=1)[0]
    aic_ids = tuple(core_id for core_id in range(108) if core_id % 54 < 18)[:32]
    aiv_ids = tuple(core_id for core_id in range(108) if core_id % 54 >= 18)[:64]
    if base == "aic":
        return aic_ids[:n]
    if base == "aiv":
        return aiv_ids[:n]
    result: list[int] = []
    for block in range((n + 2) // 3):
        result.extend((aic_ids[block], aiv_ids[2 * block], aiv_ids[2 * block + 1]))
    return tuple(result[:n])


def _record(
    config: AnalysisConfig,
    scenario: str,
    n: int,
    sweep: int,
    wave: int,
    span: int,
) -> WaveRecord:
    aic, aiv = _active_roles(scenario, n)
    deadline = 1_000_000_000 + sweep * 10_000_000 + wave * config.wave_stride_ticks
    min_begin = deadline + 2
    max_begin = min_begin + 3
    first_old = wave * n
    last_old = first_old + n - 1
    return WaveRecord(
        scenario=scenario,
        sweep=sweep,
        n=n,
        aic_active=aic,
        aiv_active=aiv,
        active_hardware_core_ids=_hardware_core_ids(scenario, n),
        target_address=0x100000 if scenario.endswith("gm0") else 0x110000,
        wave=wave,
        measured=int(wave >= config.warmup_waves),
        deadline_tick=deadline,
        min_begin_tick=min_begin,
        max_begin_tick=max_begin,
        max_end_tick=min_begin + span,
        start_spread_tick=3,
        global_span_tick=span,
        max_worker_elapsed_tick=span - 1,
        min_publish_begin_tick=deadline + config.wave_publish_offset_ticks + 2,
        max_publish_ready_tick=deadline + config.wave_publish_offset_ticks + 102,
        old_min=first_old,
        old_max=last_old,
        old_sum=(first_old + last_old) * n // 2,
        old_xor=_xor_range(first_old, last_old),
        status="PASS",
    )


def _unsupported(scenario: str, n: int) -> WaveRecord:
    return WaveRecord(
        scenario=scenario,
        sweep=-1,
        n=n,
        aic_active=0,
        aiv_active=0,
        active_hardware_core_ids=(),
        target_address=0x100000 if scenario.endswith("gm0") else 0x110000,
        wave=-1,
        measured=0,
        deadline_tick=0,
        min_begin_tick=0,
        max_begin_tick=0,
        max_end_tick=0,
        start_spread_tick=0,
        global_span_tick=0,
        max_worker_elapsed_tick=0,
        min_publish_begin_tick=0,
        max_publish_ready_tick=0,
        old_min=0,
        old_max=0,
        old_sum=0,
        old_xor=0,
        status="UNSUPPORTED",
    )


def _records(
    config: AnalysisConfig,
    span_for: Callable[[str, int, int, int], int],
) -> list[WaveRecord]:
    rows: list[WaveRecord] = []
    for scenario in SCENARIOS:
        for n in config.supported_points[scenario]:
            for sweep in range(config.sweeps):
                for wave in range(config.total_waves):
                    rows.append(_record(config, scenario, n, sweep, wave, span_for(scenario, n, sweep, wave)))
        rows.extend(_unsupported(scenario, n) for n in config.unsupported_points[scenario])
    return rows


def _locality_records(config: AnalysisConfig) -> list[WaveRecord]:
    profiles = {
        "aic_gm0_die0": ("aic_gm0", (0,), 207),
        "aic_gm0_die1": ("aic_gm0", (54,), 280),
        "aic_gm1_die0": ("aic_gm1", (0,), 280),
        "aic_gm1_die1": ("aic_gm1", (54,), 207),
        "aiv_gm0_die0": ("aiv_gm0", (18,), 208),
        "aiv_gm0_die1": ("aiv_gm0", (72,), 281),
        "aiv_gm1_die0": ("aiv_gm1", (18,), 281),
        "aiv_gm1_die1": ("aiv_gm1", (72,), 208),
    }
    rows: list[WaveRecord] = []
    for profile, (scenario, hardware_ids, baseline) in profiles.items():
        for sweep in range(config.sweeps):
            for wave in range(config.total_waves):
                record = _record(config, scenario, 1, sweep, wave, baseline + sweep % 2)
                rows.append(replace(record, scenario=profile, active_hardware_core_ids=hardware_ids))
    return rows


def _write_tsv(path: Path, records: list[WaveRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        for record in records:
            writer.writerow(
                (
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
            )


class AtomicContentionCurveAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        points = {scenario: (1, 2, 4) for scenario in SCENARIOS}
        unsupported = {scenario: () for scenario in SCENARIOS}
        self.config = AnalysisConfig(
            supported_points=points,
            unsupported_points=unsupported,
            sweeps=3,
            warmup_waves=1,
            measured_waves=4,
        )

    def test_exact_linear_curve_closes_hierarchical_fit(self) -> None:
        sweep_offsets = (-2, 0, 2)

        def span_for(_scenario: str, n: int, sweep: int, wave: int) -> int:
            wave_offset = 0 if wave == 0 else (-3, -1, 1, 3)[wave - 1]
            return 100 + 20 * n + sweep_offsets[sweep] + wave_offset

        summary = analyze_records(_records(self.config, span_for), self.config)
        self.assertEqual(summary["validation"]["status"], "PASS")
        for scenario in SCENARIOS:
            fit = summary["scenarios"][scenario]["fit"]
            self.assertAlmostEqual(fit["a_ns"], 100.0)
            self.assertAlmostEqual(fit["b_ns_per_scalar"], 20.0)
            self.assertAlmostEqual(fit["r_squared"], 1.0)
            self.assertAlmostEqual(fit["rmse_ns"], 0.0)
            self.assertAlmostEqual(fit["a_over_b"], 5.0)
            self.assertAlmostEqual(fit["parallel_fraction_at_n1"], 5.0 / 6.0)
            self.assertAlmostEqual(fit["serial_fraction_at_n1"], 1.0 / 6.0)
            self.assertEqual(fit["label"], "STRONG")
            self.assertTrue(all(point["residual_ns"] == 0.0 for point in summary["scenarios"][scenario]["points"]))

    def test_point_center_is_median_of_launch_medians(self) -> None:
        points = {scenario: (1, 2) for scenario in SCENARIOS}
        config = AnalysisConfig(
            supported_points=points,
            unsupported_points={scenario: () for scenario in SCENARIOS},
            sweeps=2,
            warmup_waves=0,
            measured_waves=4,
        )

        def span_for(_scenario: str, n: int, sweep: int, wave: int) -> int:
            if n == 1 and sweep == 0:
                return (10, 20, 100, 110)[wave]
            if n == 1:
                return 30
            return 80

        summary = analyze_records(_records(config, span_for), config)
        point = summary["scenarios"]["aic_gm0"]["points"][0]
        self.assertEqual(point["launch_median_p50_ticks"], 45.0)
        self.assertNotEqual(point["launch_median_p50_ticks"], 30.0)

    def test_gm_parameter_delta_reports_a_positive_crossing(self) -> None:
        def span_for(scenario: str, n: int, _sweep: int, _wave: int) -> int:
            return 200 + 10 * n if scenario.endswith("gm1") else 100 + 20 * n

        summary = analyze_records(_records(self.config, span_for), self.config)
        for comparison in summary["gm1_vs_gm0"].values():
            self.assertAlmostEqual(comparison["gm1_minus_gm0_a_ns"], 100.0)
            self.assertAlmostEqual(comparison["gm1_minus_gm0_b_ns_per_scalar"], -10.0)
            self.assertAlmostEqual(comparison["equal_time_n"], 10.0)

    def test_missing_wave_is_rejected_without_silent_replacement(self) -> None:
        rows = _records(self.config, lambda _scenario, n, _sweep, _wave: 100 + 20 * n)
        rows.pop()
        with self.assertRaisesRegex(AnalysisError, "wave set mismatch|expected .* total rows"):
            analyze_records(rows, self.config)

    def test_overlapping_scheduled_waves_are_rejected(self) -> None:
        rows = _records(self.config, lambda _scenario, n, _sweep, _wave: 100 + 20 * n)
        row = rows[0]
        rows[0] = WaveRecord(
            **{**row.__dict__, "max_end_tick": row.deadline_tick + 100_000, "global_span_tick": 99_998}
        )
        with self.assertRaisesRegex(AnalysisError, "publication phase|overlaps wave"):
            analyze_records(rows, self.config)

    def test_result_publication_phase_is_strictly_bounded(self) -> None:
        baseline = _records(self.config, lambda _scenario, n, _sweep, _wave: 100 + 20 * n)
        mutations = (
            (
                {"min_publish_begin_tick": baseline[0].deadline_tick + self.config.wave_publish_offset_ticks - 1},
                "before its fixed phase",
            ),
            (
                {"max_publish_ready_tick": baseline[0].deadline_tick + self.config.wave_stride_ticks},
                "does not finish before the next wave",
            ),
        )
        for changes, message in mutations:
            with self.subTest(changes=changes):
                rows = list(baseline)
                rows[0] = WaveRecord(**{**rows[0].__dict__, **changes})
                with self.assertRaisesRegex(AnalysisError, message):
                    analyze_records(rows, self.config)

    def test_die_locality_profiles_are_validated_and_summarized(self) -> None:
        rows = _locality_records(self.config)
        locality = analyze_die_locality(rows, self.config)
        self.assertEqual(locality["validation"]["status"], "PASS")
        self.assertEqual(locality["profiles"]["aic_gm0_die0"]["hardware_core_ids"], [0])
        self.assertEqual(locality["profiles"]["aiv_gm1_die1"]["hardware_core_ids"], [72])
        self.assertEqual(locality["comparisons"]["aic_gm0"]["die0_minus_die1_ns"], -73.0)
        self.assertEqual(locality["comparisons"]["aiv_gm1"]["faster_die"], 1)
        self.assertEqual(locality["matched_home_local"]["aic"]["absolute_gap_ns"], 0.0)

        malformed = list(rows)
        die1_index = next(index for index, record in enumerate(malformed) if record.scenario == "aic_gm0_die1")
        malformed[die1_index] = replace(malformed[die1_index], active_hardware_core_ids=(0,))
        with self.assertRaisesRegex(AnalysisError, "outside physical die 1"):
            analyze_die_locality(malformed, self.config)

        unmatched = [
            replace(
                record,
                max_end_tick=record.min_begin_tick + 230,
                global_span_tick=230,
                max_worker_elapsed_tick=229,
            )
            if record.scenario == "aic_gm1_die1"
            else record
            for record in rows
        ]
        with self.assertRaisesRegex(AnalysisError, "home-local baselines differ"):
            analyze_die_locality(unmatched, self.config)

    def test_die_locality_is_rendered_and_bundled(self) -> None:
        curve_rows = _records(self.config, lambda _scenario, n, _sweep, _wave: 100 + 20 * n)
        locality_rows = _locality_records(self.config)
        summary = analyze_records(curve_rows, self.config)
        summary["die_locality"] = analyze_die_locality(locality_rows, self.config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_artifacts(summary, root / "summary.json", root / "curve.svg", root / "report.md")
            report = (root / "report.md").read_text(encoding="utf-8")
            self.assertIn("GM 归属的镜像校验", report)
            self.assertIn("不小于 32 ns", report)
            outputs = write_raw_bundle(curve_rows + locality_rows, root / "raw")
            self.assertEqual(outputs[-1].name, "raw_locality.tsv.xz")
            self.assertEqual(len(load_tsv(outputs[-1])), len(locality_rows))

    def test_tsv_round_trip_and_artifacts(self) -> None:
        self.assertEqual(
            TSV_COLUMNS[3:8],
            ("aic_active", "aiv_active", "active_hardware_core_ids", "target_address", "wave"),
        )
        rows = _records(self.config, lambda _scenario, n, _sweep, _wave: 100 + 20 * n)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.tsv"
            summary_path = root / "summary.json"
            svg_path = root / "fit.svg"
            report_path = root / "report.md"
            _write_tsv(raw, rows)
            loaded = load_tsv(raw)
            self.assertEqual(loaded[0].active_hardware_core_ids, rows[0].active_hardware_core_ids)
            summary = analyze_records(loaded, self.config)
            write_artifacts(summary, summary_path, svg_path, report_path)

            persisted = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["schema"]["name"], "atomic-contention-curve-analysis")
            self.assertEqual(persisted["schema"]["version"], 5)
            svg_root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
            panels = [element for element in svg_root if element.tag.endswith("g") and element.get("class") == "panel"]
            self.assertEqual(len(panels), 3)
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("A+B*N", report)
            self.assertIn("物理 placement", report)
            self.assertIn("结论边界", report)

            bundle = write_raw_bundle(loaded, root / "raw")
            self.assertEqual(
                [path.name for path in bundle],
                [
                    "raw_aic_gm0.tsv.xz",
                    "raw_aic_gm1.tsv.xz",
                    "raw_aiv_gm0.tsv.xz",
                    "raw_aiv_gm1.tsv.xz",
                    "raw_mixed_gm0.tsv.xz",
                    "raw_mixed_gm1.tsv.xz",
                ],
            )
            bundled_rows = [record for path in bundle for record in load_tsv(path)]
            self.assertEqual(analyze_records(bundled_rows, self.config)["validation"]["status"], "PASS")

    def test_hardware_core_id_json_must_be_an_integer_array(self) -> None:
        rows = _records(self.config, lambda _scenario, n, _sweep, _wave: 100 + 20 * n)
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw.tsv"
            malformed = (
                ("0,18", "is not valid JSON"),
                ('{"core":0}', "must be a JSON array"),
                ("[true]", r"\[0\] must be a JSON integer"),
            )
            for raw_value, message in malformed:
                with self.subTest(raw_value=raw_value):
                    _write_tsv(raw, rows)
                    lines = raw.read_text(encoding="utf-8").splitlines()
                    fields = lines[1].split("\t")
                    fields[TSV_COLUMNS.index("active_hardware_core_ids")] = raw_value
                    lines[1] = "\t".join(fields)
                    raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(AnalysisError, message):
                        load_tsv(raw)

    def test_hardware_core_id_count_uniqueness_range_and_role_are_validated(self) -> None:
        baseline = _records(self.config, lambda _scenario, n, _sweep, _wave: 100 + 20 * n)
        aic_n2 = next(index for index, row in enumerate(baseline) if row.scenario == "aic_gm0" and row.n == 2)
        mixed_n2 = next(index for index, row in enumerate(baseline) if row.scenario == "mixed_gm0" and row.n == 2)
        mutations = (
            (aic_n2, (0,), "must contain N=2 IDs"),
            (aic_n2, (0, 0), "must be unique"),
            (aic_n2, (0, 108), "outside the A5 physical range"),
            (aic_n2, (0, 18), "role counts"),
            (mixed_n2, (18, 0), "participant 0 must use an AIC physical slot"),
            (mixed_n2, (0, 20), "does not match AIC triplet"),
        )
        for row_index, core_ids, message in mutations:
            with self.subTest(core_ids=core_ids):
                rows = list(baseline)
                row = rows[row_index]
                rows[row_index] = WaveRecord(**{**row.__dict__, "active_hardware_core_ids": core_ids})
                with self.assertRaisesRegex(AnalysisError, message):
                    analyze_records(rows, self.config)

    def test_hardware_core_placement_is_constant_within_one_launch(self) -> None:
        rows = _records(self.config, lambda _scenario, n, _sweep, _wave: 100 + 20 * n)
        row_index = next(
            index
            for index, row in enumerate(rows)
            if row.scenario == "aic_gm0" and row.n == 1 and row.sweep == 0 and row.wave == 1
        )
        row = rows[row_index]
        rows[row_index] = WaveRecord(**{**row.__dict__, "active_hardware_core_ids": (1,)})
        with self.assertRaisesRegex(AnalysisError, "hardware core placement changes across waves"):
            analyze_records(rows, self.config)

    def test_target_address_is_constant_and_gm_homes_are_distinct(self) -> None:
        baseline = _records(self.config, lambda _scenario, n, _sweep, _wave: 100 + 20 * n)
        wave_index = next(
            index
            for index, row in enumerate(baseline)
            if row.scenario == "aic_gm0" and row.n == 1 and row.sweep == 0 and row.wave == 1
        )
        changed_wave = list(baseline)
        changed_wave[wave_index] = replace(changed_wave[wave_index], target_address=0x120000)
        with self.assertRaisesRegex(AnalysisError, "target address changes across waves"):
            analyze_records(changed_wave, self.config)

        same_address = [
            replace(row, target_address=0x100000) if row.scenario.endswith("gm1") else row for row in baseline
        ]
        with self.assertRaisesRegex(AnalysisError, "GM0 and GM1 curves must use different target addresses"):
            analyze_records(same_address, self.config)

    def test_unsupported_sentinel_is_strictly_validated(self) -> None:
        points = {scenario: (1, 2) for scenario in SCENARIOS}
        unsupported = {scenario: ((4,) if scenario == "aic_gm0" else ()) for scenario in SCENARIOS}
        config = AnalysisConfig(
            supported_points=points,
            unsupported_points=unsupported,
            sweeps=2,
            warmup_waves=1,
            measured_waves=2,
        )
        rows = _records(config, lambda _scenario, n, _sweep, _wave: 100 + 20 * n)
        sentinel_index = next(index for index, row in enumerate(rows) if row.status == "UNSUPPORTED")
        sentinel = rows[sentinel_index]
        rows[sentinel_index] = WaveRecord(**{**sentinel.__dict__, "global_span_tick": 1})
        with self.assertRaisesRegex(AnalysisError, "malformed UNSUPPORTED sentinel"):
            analyze_records(rows, config)

        rows = _records(config, lambda _scenario, n, _sweep, _wave: 100 + 20 * n)
        sentinel_index = next(index for index, row in enumerate(rows) if row.status == "UNSUPPORTED")
        sentinel = rows[sentinel_index]
        rows[sentinel_index] = WaveRecord(**{**sentinel.__dict__, "active_hardware_core_ids": (0,)})
        with self.assertRaisesRegex(AnalysisError, "malformed UNSUPPORTED sentinel"):
            analyze_records(rows, config)


if __name__ == "__main__":
    unittest.main()
