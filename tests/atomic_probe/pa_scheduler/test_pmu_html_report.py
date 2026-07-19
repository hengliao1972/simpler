#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""standalone submit-pmu HTML 可视报告的生成与失败原子性回归。"""

from __future__ import annotations

import json
import re
import stat
import tempfile
import unittest
from pathlib import Path

try:
    from .pmu_html_report import (
        DEFAULT_AIC_PMU_CYCLES_PER_NS,
        DEFAULT_AIV_PMU_CYCLES_PER_NS,
        DEFAULT_PMU_CYCLES_PER_NS,
        PMU_CALIBRATION_CYCLE_DELTA,
        PMU_CALIBRATION_SYS_TICK_NS,
        _cycles_to_us,
        _phase_share_metric,
        default_output_path,
        render_report,
        write_report,
    )
    from .pmu_sidecar_analyzer import analyze
    from .test_pmu_sidecar_analyzer import _submit_pmu_capture, _submit_pmu_summary
except ImportError:
    from pmu_html_report import (
        DEFAULT_AIC_PMU_CYCLES_PER_NS,
        DEFAULT_AIV_PMU_CYCLES_PER_NS,
        DEFAULT_PMU_CYCLES_PER_NS,
        PMU_CALIBRATION_CYCLE_DELTA,
        PMU_CALIBRATION_SYS_TICK_NS,
        _cycles_to_us,
        _phase_share_metric,
        default_output_path,
        render_report,
        write_report,
    )
    from pmu_sidecar_analyzer import analyze
    from test_pmu_sidecar_analyzer import _submit_pmu_capture, _submit_pmu_summary


class PmuHtmlReportTest(unittest.TestCase):
    def _write_capture(
        self, directory: str, capture: dict, name: str = "submit_icache_raw.json"
    ) -> Path:
        path = Path(directory) / name
        path.write_text(json.dumps(capture, ensure_ascii=False), encoding="utf-8")
        return path

    def test_default_output_uses_descriptive_report_name(self) -> None:
        self.assertEqual(
            default_output_path(Path("result/submit_icache_raw.json")),
            Path("result/submit_icache_report.html"),
        )
        self.assertEqual(
            default_output_path(Path("result/custom.json")),
            Path("result/custom_report.html"),
        )

    def test_none_report_reuses_analyzer_values_and_has_96_rows(self) -> None:
        capture = _submit_pmu_capture(phase="none")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_capture(directory, capture)
            analysis = analyze([path])
            document = render_report(path)

        aiv = analysis["per_run"][0]["groups"]["aiv"]
        self.assertIn("Standalone PA Submit I-cache 报告", document)
        self.assertIn(f"{aiv['icache_misses_per_core']:,.2f}", document)
        self.assertIn(f"{aiv['icache_miss_rate'] * 100:.4f}%", document)
        self.assertIn("PRIMARY ↔ SHADOW EXACT 96/96", document)
        self.assertIn("局部 phase：none", document)
        self.assertIn("局部阶段总览：none", document)
        self.assertIn("不适用：</strong>该 ELF 未编译局部阶段", document)
        self.assertLess(
            document.index("局部阶段总览：none"),
            document.index('<section class="cards" aria-label="关键指标">'),
        )
        self.assertNotIn('data-metric="time"', document)
        self.assertIn("PMU total 与 scalar busy", document)
        self.assertIn(f"{aiv['total_cycles_sum'] / 64:,.2f}", document)
        self.assertIn(f"{aiv['scalar_busy_sum'] / 64:,.2f}", document)
        self.assertIn("非 Scalar-busy 残余", document)
        self.assertIn("它不是空闲时间，也不是 I-cache stall", document)
        self.assertIn("不提供 scalar_wait_ib_time/scalar_wait_time", document)
        self.assertIn("完整 Submit（最早开始 → 最晚结束）", document)
        self.assertIn("96 核逐核 PMU total 平均值（校准）", document)
        self.assertNotIn("包络", document)
        self.assertEqual(document.count('class="pmu-role-card"'), 3)
        self.assertEqual(document.count('class="pmu-compact-table"'), 3)
        self.assertEqual(document.count('data-stat="min"'), 3)
        self.assertEqual(document.count('data-stat="mean"'), 3)
        self.assertEqual(document.count('data-stat="max"'), 3)
        self.assertNotIn('data-stat="median"', document)
        self.assertNotIn('data-stat="p95"', document)
        self.assertIn(".pmu-role-grid { display:grid;", document)
        self.assertIn(".pmu-role-card { min-width:0;", document)
        self.assertIn('<div class="table-scroll">\n    <table class="icache-table">', document)
        self.assertGreaterEqual(document.count('class="table-scroll"'), 2)
        self.assertIn("total_cycles=", document)
        self.assertIn("不是 Submit 墙钟损失", document)
        self.assertIn("PMU cycle 频率校准", document)
        self.assertIn(f"PMU cycle_delta = {PMU_CALIBRATION_CYCLE_DELTA:,}", document)
        self.assertIn(
            f"SYS_CNT tick_delta = {PMU_CALIBRATION_SYS_TICK_NS:,} ns", document
        )
        self.assertIn(f"ALL = <strong>{DEFAULT_PMU_CYCLES_PER_NS:.6f}</strong>", document)
        self.assertIn(f"AIC = <strong>{DEFAULT_AIC_PMU_CYCLES_PER_NS:.6f}</strong>", document)
        self.assertIn(f"AIV = <strong>{DEFAULT_AIV_PMU_CYCLES_PER_NS:.6f}</strong>", document)
        all_group = analysis["per_run"][0]["groups"]["all"]
        all_total_per_core = all_group["total_cycles_sum"] / all_group["cores"]
        self.assertIn(
            f'<div class="value">{_cycles_to_us(all_total_per_core, DEFAULT_PMU_CYCLES_PER_NS):,.3f} µs</div>',
            document,
        )
        self.assertIn("生成器：pmu_html_report schema v5", document)
        self.assertEqual(document.count('data-worker-id="'), 96)
        self.assertNotIn("http://", document)
        self.assertNotIn("https://", document)
        self.assertNotIn("<script", document.lower())

    def test_pmu_role_cards_derive_min_mean_max_from_validated_records(self) -> None:
        capture = _submit_pmu_capture(phase="none")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_capture(directory, capture)
            document = render_report(path)

        frequencies = {
            "all": DEFAULT_PMU_CYCLES_PER_NS,
            "aic": DEFAULT_AIC_PMU_CYCLES_PER_NS,
            "aiv": DEFAULT_AIV_PMU_CYCLES_PER_NS,
        }
        for group_name in ("all", "aic", "aiv"):
            group_records = (
                capture["records"]
                if group_name == "all"
                else [record for record in capture["records"] if record["role"] == group_name]
            )
            self.assertNotIn("min", capture["summary"][group_name]["total_cycles"])
            card_match = re.search(
                rf'<article class="pmu-role-card" data-pmu-group="{group_name}">(.*?)</article>',
                document,
                re.DOTALL,
            )
            self.assertIsNotNone(card_match)
            card = card_match.group(1)
            cycles_per_ns = frequencies[group_name]
            for metric in ("total_cycles", "scalar_busy"):
                values = [int(record[metric]) for record in group_records]
                for value in (min(values), max(values)):
                    self.assertIn(f"{value:,} cycle", card)
                    self.assertIn(f"≈{_cycles_to_us(value, cycles_per_ns):,.3f} µs", card)
            self.assertEqual(card.count('data-stat="'), 3)

    def test_icache_overview_shows_sum_and_per_core_min_mean_max(self) -> None:
        capture = _submit_pmu_capture(phase="none")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_capture(directory, capture)
            document = render_report(path)

        all_records = capture["records"]
        all_requests = [int(record["icache_requests"]) for record in all_records]
        all_misses = [int(record["icache_misses"]) for record in all_records]
        for label, values in (
            ("request", all_requests),
            ("miss", all_misses),
        ):
            card_match = re.search(
                rf'<div class="card"><div class="label">完整 Submit primary I-cache {label}'
                rf'（96 核总和）</div><div class="value">{sum(values):,}</div>'
                rf'<div class="muted">逐核平均 {sum(values) / len(values):,.2f}</div>'
                rf'<div class="muted">逐核最小 {min(values):,} · 逐核最大 {max(values):,}</div></div>',
                document,
            )
            self.assertIsNotNone(card_match)

        table_match = re.search(
            r'<table class="icache-table">(.*?)</table>', document, re.DOTALL
        )
        self.assertIsNotNone(table_match)
        table = table_match.group(1)
        self.assertIn("request min/core", table)
        self.assertIn("request mean/core", table)
        self.assertIn("request max/core", table)
        self.assertIn("miss min/core", table)
        self.assertIn("miss mean/core", table)
        self.assertIn("miss max/core", table)
        for role in ("aic", "aiv"):
            records = [record for record in all_records if record["role"] == role]
            requests = [int(record["icache_requests"]) for record in records]
            misses = [int(record["icache_misses"]) for record in records]
            expected_cells = (
                min(requests),
                sum(requests) / len(requests),
                max(requests),
                min(misses),
                sum(misses) / len(misses),
                max(misses),
            )
            row_match = re.search(
                rf'<tr><td><span class="role role-{role}">{role.upper()}</span></td>'
                rf'<td>{len(records):,}</td>'
                rf'<td>{expected_cells[0]:,}</td>'
                rf'<td>{expected_cells[1]:,.2f}</td>'
                rf'<td>{expected_cells[2]:,}</td>'
                rf'<td>{expected_cells[3]:,}</td>'
                rf'<td>{expected_cells[4]:,.2f}</td>'
                rf'<td>{expected_cells[5]:,}</td>',
                table,
            )
            self.assertIsNotNone(row_match)

        self.assertNotIn("median", table.lower())
        self.assertNotIn("p95", table.lower())

    def test_role_specific_frequency_converts_per_core_cycles(self) -> None:
        capture = _submit_pmu_capture(phase="none")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_capture(directory, capture)
            document = render_report(path)

        for role, cycles_per_ns in (
            ("aic", DEFAULT_AIC_PMU_CYCLES_PER_NS),
            ("aiv", DEFAULT_AIV_PMU_CYCLES_PER_NS),
        ):
            record = next(item for item in capture["records"] if item["role"] == role)
            total = int(record["total_cycles"])
            self.assertIn(
                f'<span class="cycle-value">{total:,} cycle</span>'
                f'<span class="cycle-time">≈{_cycles_to_us(total, cycles_per_ns):,.3f} µs</span>',
                document,
            )

    def test_frequency_override_changes_group_and_role_conversion(self) -> None:
        capture = _submit_pmu_capture(phase="none")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_capture(directory, capture)
            document = render_report(
                path,
                pmu_cycles_per_ns=2.0,
                aic_pmu_cycles_per_ns=4.0,
                aiv_pmu_cycles_per_ns=8.0,
            )

        all_total_per_core = capture["summary"]["all"]["total_cycles"]["sum"] / 96
        self.assertIn(f'<div class="value">{all_total_per_core / 2.0 / 1000:,.3f} µs</div>', document)
        self.assertIn("ALL = <strong>2.000000</strong>", document)
        self.assertIn("AIC = <strong>4.000000</strong>", document)
        self.assertIn("AIV = <strong>8.000000</strong>", document)
        for role, cycles_per_ns in (("aic", 4.0), ("aiv", 8.0)):
            record = next(item for item in capture["records"] if item["role"] == role)
            self.assertIn(
                f"calibrated_time={int(record['total_cycles']) / cycles_per_ns / 1000:,.3f}us",
                document,
            )

    def test_invalid_pmu_frequency_is_rejected_before_publish(self) -> None:
        capture = _submit_pmu_capture(phase="none")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_capture(directory, capture)
            output = default_output_path(path)
            with self.assertRaisesRegex(ValueError, "pmu_cycles_per_ns must be finite and positive"):
                write_report(path, pmu_cycles_per_ns=0.0)
            self.assertFalse(output.exists())

    def test_running_phase_shows_bounds_and_observer_warning(self) -> None:
        capture = _submit_pmu_capture(phase="claim")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_capture(directory, capture)
            analysis = analyze([path])
            document = render_report(path)

        group = analysis["per_run"][0]["groups"]["all"]
        self.assertIn("局部阶段总览：claim", document)
        self.assertIn("局部阶段详细数据：claim", document)
        self.assertIn("带边界扰动的诊断区间", document)
        self.assertIn(f"{group['phase_calls_sum']:,}", document)
        self.assertIn(
            f"{group['phase_icache_misses_lower_bound_sum']:,}.."
            f"{group['phase_icache_misses_upper_bound_sum']:,}",
            document,
        )
        self.assertLess(
            document.index("局部阶段总览：claim"),
            document.index('<section class="cards" aria-label="关键指标">'),
        )
        self.assertEqual(document.count('class="phase-share-card"'), 3)
        self.assertEqual(document.count('class="phase-share-metric"'), 9)
        self.assertEqual(document.count('class="phase-share-track '), 9)
        self.assertEqual(document.count('data-metric="time"'), 3)
        for group_name in ("all", "aic", "aiv"):
            phase_group = analysis["per_run"][0]["groups"][group_name]
            time_share = phase_group["phase_time_share_of_submit"]
            self.assertIn(
                f'data-phase-group="{group_name}" data-metric="time" '
                f'data-time-share="{time_share:.12f}"',
                document,
            )
            self.assertIn(f"{time_share * 100:.4f}%", document)
            self.assertIn(
                f"平均 {phase_group['phase_elapsed_per_core_us']:,.3f} µs/核",
                document,
            )
            for metric, lower_key, upper_key in (
                (
                    "request",
                    "phase_icache_request_lower_bound_share_of_submit",
                    "phase_icache_request_upper_bound_share_of_submit",
                ),
                (
                    "miss",
                    "phase_icache_miss_lower_bound_share_of_submit",
                    "phase_icache_miss_upper_bound_share_of_submit",
                ),
            ):
                self.assertIn(
                    f'data-phase-group="{group_name}" data-metric="{metric}" '
                    f'data-lower-share="{phase_group[lower_key]:.12f}" '
                    f'data-upper-share="{phase_group[upper_key]:.12f}"',
                    document,
                )
                lower_percent = phase_group[lower_key] * 100.0
                upper_percent = phase_group[upper_key] * 100.0
                metric_pattern = re.compile(
                    rf'data-phase-group="{group_name}" data-metric="{metric}" '
                    rf'data-lower-share="{phase_group[lower_key]:.12f}" '
                    rf'data-upper-share="{phase_group[upper_key]:.12f}".*?'
                    rf'下界 {lower_percent:.4f}%，上界 {upper_percent:.4f}%.*?'
                    rf'class="phase-share-upper" style="width:{upper_percent:.6f}%".*?'
                    rf'class="phase-share-lower" style="width:{lower_percent:.6f}%".*?'
                    rf'class="phase-share-upper-marker" style="left:{upper_percent:.6f}%"',
                    re.DOTALL,
                )
                self.assertRegex(document, metric_pattern)
        self.assertEqual(
            document.count(
                '<div class="phase-share-axis"><span>0%</span><span>50%</span><span>100%</span></div>'
            ),
            9,
        )
        self.assertIn('class="phase-table-scroll"', document)
        self.assertIn('class="phase-table"', document)
        self.assertIn('<details class="phase-table-details">', document)
        self.assertNotIn('<details class="phase-table-details" open>', document)
        self.assertIn(
            '<div class="phase-table-scroll">\n        <table class="phase-table">',
            document,
        )
        self.assertIn("展开 ALL / AIC / AIV 完整数字表", document)
        self.assertIn(".phase-panel { overflow:hidden; }", document)
        self.assertIn(
            ".phase-table-scroll { width:100%; max-width:100%; overflow-x:auto;",
            document,
        )
        self.assertIn('class="phase-plot-scroll"', document)
        self.assertIn(
            ".phase-plot-scroll { width:100%; max-width:100%; overflow-x:auto;",
            document,
        )
        self.assertIn("不同 phase ELF 的局部值不能相加", document)
        self.assertIn("phase_icache_misses=", document)
        self.assertIn(
            "Σ阶段 SYS_CNT / Σ同核首个 submit_begin 计时点到末个 submit_end 计时点 SYS_CNT",
            document,
        )
        self.assertIn("不包含两侧 ld_dev", document)

    def test_legacy_v4_phase_time_is_explicitly_unavailable(self) -> None:
        capture = _submit_pmu_capture(phase="claim", schema_version=4)
        for field in (
            "phase_time_observation_included",
            "phase_time_sys_counter_tick_ns",
            "phase_time_boundary",
            "phase_time_excludes_shadow_read_overhead",
            "phase_time_includes_timestamp_overhead",
            "phase_time_share_definition",
            "phase_time_denominator_scope",
        ):
            capture["configuration"].pop(field)
        capture["validation"].pop("phase_time_valid_records")
        capture["validation"].pop("phase_time_measurement_valid")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_capture(directory, capture)
            document = render_report(path)

        self.assertEqual(document.count('data-metric="time" data-time-share=""'), 3)
        self.assertIn("本次 raw 未采集阶段 SYS_CNT", document)
        self.assertNotIn("None%", document)
        self.assertNotIn("nan%", document.lower())

    def test_phase_share_equal_bounds_keep_visible_value_and_marker(self) -> None:
        fragment = _phase_share_metric("AIC", "I-cache miss", "miss", 0.25, 0.25, 10, 10)
        self.assertIn("25.0000%..25.0000%", fragment)
        self.assertIn('class="phase-share-upper" style="width:25.000000%"', fragment)
        self.assertIn('class="phase-share-lower" style="width:25.000000%"', fragment)
        self.assertIn('class="phase-share-upper-marker" style="left:25.000000%"', fragment)

    def test_phase_share_zero_denominator_is_explicitly_unavailable(self) -> None:
        fragment = _phase_share_metric("AIC", "I-cache miss", "miss", None, None, 0, 0)
        self.assertIn("比例不可计算", fragment)
        self.assertNotIn("None%", fragment)
        self.assertNotIn("nan%", fragment.lower())

    def test_zero_aic_miss_has_an_explicit_non_dividing_comparison(self) -> None:
        capture = _submit_pmu_capture(phase="none")
        for record in capture["records"]:
            if record["role"] == "aic":
                record["icache_misses"] = 0
                record["shadow_whole_icache_misses"] = 0
        groups = {
            "all": capture["records"],
            "aic": [record for record in capture["records"] if record["role"] == "aic"],
            "aiv": [record for record in capture["records"] if record["role"] == "aiv"],
        }
        capture["summary"] = {
            name: _submit_pmu_summary(records) for name, records in groups.items()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_capture(directory, capture)
            document = render_report(path)

        self.assertIn("AIC miss/core 为 0，AIV 相对变化不可计算", document)

    def test_dynamic_strings_and_raw_link_are_escaped(self) -> None:
        capture = _submit_pmu_capture(phase="none")
        capture["capture"]["capture_id"] = '<capture & "probe">'
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_capture(directory, capture, "submit_<icache>_raw.json")
            document = render_report(path)

        self.assertNotIn('<capture & "probe">', document)
        self.assertIn("&lt;capture &amp; &quot;probe&quot;&gt;", document)
        self.assertIn("submit_%3Cicache%3E_raw.json", document)
        self.assertIn("submit_&lt;icache&gt;_raw.json", document)

    def test_write_report_is_complete_readable_and_leaves_no_temp(self) -> None:
        capture = _submit_pmu_capture(phase="none")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_capture(directory, capture)
            output = write_report(path)
            mode = stat.S_IMODE(output.stat().st_mode)
            temporary_files = list(Path(directory).glob(".*.tmp"))

            self.assertEqual(output.name, "submit_icache_report.html")
            self.assertEqual(mode, 0o644)
            self.assertTrue(output.read_text(encoding="utf-8").endswith("</html>\n"))
            self.assertEqual(temporary_files, [])

    def test_invalid_raw_does_not_publish_html(self) -> None:
        capture = _submit_pmu_capture(phase="none")
        capture["records"][0]["icache_misses"] = capture["records"][0]["icache_requests"] + 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_capture(directory, capture)
            output = default_output_path(path)
            with self.assertRaisesRegex(ValueError, "miss > request|raw summary mismatch"):
                write_report(path)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
