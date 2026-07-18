#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""standalone PMU sidecar 多轮分析器的 raw 门禁与聚合回归。"""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

try:
    from .pmu_sidecar_analyzer import METRIC_NAMES, analyze, load_capture
except ImportError:
    from pmu_sidecar_analyzer import METRIC_NAMES, analyze, load_capture


def _p95(values: Sequence[int]) -> int:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cores": len(records),
        "active_cores": len(records),
        "trusted_cores": len(records),
    }
    for metric in METRIC_NAMES:
        values = [record[metric] for record in records]
        result[metric] = {
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2]
            if len(values) % 2
            else (sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]) / 2,
            "p95": _p95(values),
            "max": max(values),
        }
    result["icache_miss_rate"] = (
        result["icache_misses"]["sum"] / result["icache_requests"]["sum"]
    )
    return result


def _capture(offset: int = 0, window: str = "submit-all") -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for worker_id, role in enumerate(("aic", "aiv", "aiv")):
        base = 100 + worker_id * 10 + offset
        record: dict[str, Any] = {
            "worker_id": worker_id,
            "physical_core_id": 10 + worker_id,
            "role": role,
            "trusted": True,
            "selectors_match": True,
            "owner_bitmap_member": True,
            "worker_slot_exact": True,
            "physical_role_matches": True,
            "window_started": True,
            "window_stopped": True,
        }
        for metric_index, metric in enumerate(METRIC_NAMES):
            record[metric] = base + metric_index
        # miss/request 取独立、直观的数值，便于断言聚合公式。
        record["icache_requests"] = 1000 + base
        record["icache_misses"] = 100 + base // 10
        records.append(record)

    groups = {
        "all": records,
        "aic": [record for record in records if record["role"] == "aic"],
        "aiv": [record for record in records if record["role"] == "aiv"],
    }
    return {
        "schema": {"name": "pa_scheduler_pmu_phase_windows", "version": 3},
        "capture": {
            "capture_id": f"capture-{offset}",
            "accepted": True,
            "published_after_runtime_cleanup": True,
            "runtime_cleanup_passed": True,
            "owner_restore_passed": True,
        },
        "configuration": {
            "device": 0,
            "batches": 256,
            "workers": 3,
            "aic_workers": 1,
            "aiv_workers": 2,
            "pmu_window": window,
            "submit_span_us": 5000.0 + offset,
            "selectors": {"cnt6_icache_request": 0x034, "cnt7_icache_miss": 0x035},
            "counter_width_bits": {"total": 64, "programmable": 32},
            "phase_timestamp_calls_present": True,
            "phase_record_writes": False,
            "profile_accumulation": False,
            "trace_enabled": False,
            "atomic_trace": False,
            "gate_start_stop_have_pipe_all_barriers": True,
            "winner_workload": {
                "mode": "real-compute",
                "counts": {"qk": 6, "sf": 28, "pv": 4, "up": 1},
                "unit": "complete_128x128_engine_pipeline_iteration",
            },
        },
        "validation": {
            "semantic_passed": True,
            "pmu_passed": True,
            "icache_measurement_valid": True,
            "icache_miss_le_request": True,
            "counter_below_risk_threshold": True,
            "trusted_records": 3,
            "unique_physical_core_ids": 3,
            "owner_bitmap_member_records": 3,
            "exact_worker_slot_records": 3,
            "physical_role_match_records": 3,
            "window_started_records": 3,
            "window_stopped_records": 3,
        },
        "records": records,
        "summary": {name: _summary(group) for name, group in groups.items()},
    }


class PmuSidecarAnalyzerTest(unittest.TestCase):
    def _write(self, directory: str, name: str, capture: dict[str, Any]) -> Path:
        path = Path(directory) / name
        path.write_text(json.dumps(capture), encoding="utf-8")
        return path

    def test_valid_raw_is_recomputed_and_multiple_runs_are_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self._write(directory, "first.json", _capture(0))
            second = self._write(directory, "second.json", _capture(20))
            result = analyze([first, second], miss_penalty_ns=90.0)

        self.assertTrue(result["validation"]["raw_to_summary_all_fields_match"])
        self.assertEqual(result["aggregate"]["runs"], 2)
        self.assertEqual(result["aggregate"]["submit_span_us"]["median"], 5010.0)
        expected_misses = [
            _capture(offset)["summary"]["all"]["icache_misses"]["sum"]
            for offset in (0, 20)
        ]
        self.assertEqual(
            result["aggregate"]["groups"]["all"]["icache_misses_sum"]["median"],
            sum(expected_misses) / 2,
        )
        self.assertTrue(result["estimation"]["not_wall_time"])
        self.assertTrue(result["estimation"]["not_additive_stall_time"])
        self.assertEqual(
            result["actual_exposed_loss"]["status"],
            "requires_same_semantics_paired_ab",
        )
        expected_aiv_misses_per_core = [
            _capture(offset)["summary"]["aiv"]["icache_misses"]["sum"] / 2
            for offset in (0, 20)
        ]
        self.assertEqual(
            result["aggregate"]["groups"]["aiv"]["icache_misses_per_core"]["median"],
            sum(expected_aiv_misses_per_core) / 2,
        )

    def test_tampered_host_summary_is_rejected(self) -> None:
        capture = _capture()
        capture["summary"]["aiv"]["icache_misses"]["sum"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "tampered.json", capture)
            with self.assertRaisesRegex(ValueError, "raw summary mismatch"):
                load_capture(path)

    def test_different_observation_configurations_cannot_be_merged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self._write(directory, "submit.json", _capture(window="submit-all"))
            second = self._write(directory, "empty.json", _capture(window="empty"))
            with self.assertRaisesRegex(ValueError, "observation configuration differs"):
                analyze([first, second])

    def test_failed_restore_is_rejected(self) -> None:
        capture = _capture()
        capture["capture"]["owner_restore_passed"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "restore_failed.json", capture)
            with self.assertRaisesRegex(ValueError, "owner_restore_passed is not true"):
                load_capture(path)

    def test_duplicate_physical_core_is_rejected(self) -> None:
        capture = _capture()
        capture["records"][1]["physical_core_id"] = capture["records"][0]["physical_core_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "duplicate.json", capture)
            with self.assertRaisesRegex(ValueError, "duplicate physical_core_id"):
                load_capture(path)

    def test_miss_greater_than_request_is_rejected_before_summary_use(self) -> None:
        capture = _capture()
        capture["records"][0]["icache_misses"] = capture["records"][0]["icache_requests"] + 1
        # 同步重算 summary，证明失败来自逐核物理约束，而不是 summary 不一致。
        records = capture["records"]
        capture["summary"] = {
            "all": _summary(records),
            "aic": _summary([record for record in records if record["role"] == "aic"]),
            "aiv": _summary([record for record in records if record["role"] == "aiv"]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "invalid_miss.json", capture)
            with self.assertRaisesRegex(ValueError, "miss > request"):
                load_capture(path)


if __name__ == "__main__":
    unittest.main()
