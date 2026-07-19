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
    from .pmu_html_report import _phase_share_metric, default_output_path, render_report, write_report
    from .pmu_sidecar_analyzer import analyze
    from .test_pmu_sidecar_analyzer import _submit_pmu_capture, _submit_pmu_summary
except ImportError:
    from pmu_html_report import _phase_share_metric, default_output_path, render_report, write_report
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
        self.assertIn("PMU total 与 scalar busy", document)
        self.assertIn(f"{aiv['total_cycles_sum'] / 64:,.2f}", document)
        self.assertIn(f"{aiv['scalar_busy_sum'] / 64:,.2f}", document)
        self.assertIn("非 Scalar-busy 残余", document)
        self.assertIn("它不是空闲时间，也不是 I-cache stall", document)
        self.assertIn("不提供 scalar_wait_ib_time/scalar_wait_time", document)
        self.assertIn("<th>PMU total</th><th>scalar busy</th>", document)
        self.assertIn("total_cycles=", document)
        self.assertIn("不是 Submit 墙钟损失", document)
        self.assertEqual(document.count('data-worker-id="'), 96)
        self.assertNotIn("http://", document)
        self.assertNotIn("https://", document)
        self.assertNotIn("<script", document.lower())

    def test_running_phase_shows_bounds_and_observer_warning(self) -> None:
        capture = _submit_pmu_capture(phase="claim")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_capture(directory, capture)
            analysis = analyze([path])
            document = render_report(path)

        group = analysis["per_run"][0]["groups"]["all"]
        self.assertIn("局部 phase：claim", document)
        self.assertIn("带边界扰动的 running read-clear 区间", document)
        self.assertIn(f"{group['phase_calls_sum']:,}", document)
        self.assertIn(
            f"{group['phase_icache_misses_lower_bound_sum']:,}.."
            f"{group['phase_icache_misses_upper_bound_sum']:,}",
            document,
        )
        self.assertIn("局部占同一 ELF 完整 Submit primary", document)
        self.assertEqual(document.count('class="phase-share-card"'), 3)
        self.assertEqual(document.count('class="phase-share-metric"'), 6)
        self.assertEqual(document.count('class="phase-share-track '), 6)
        for group_name in ("all", "aic", "aiv"):
            phase_group = analysis["per_run"][0]["groups"][group_name]
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
            6,
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
