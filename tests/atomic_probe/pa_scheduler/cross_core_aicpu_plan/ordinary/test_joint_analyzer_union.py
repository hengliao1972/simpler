#!/usr/bin/env python3
"""Regression tests for honest AICPU/AICore interval-union overlap."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


FIXTURES_PATH = Path(__file__).with_name(
    "test_plan_swimlane_converter.py"
)
FIXTURES_SPEC = importlib.util.spec_from_file_location(
    "joint_analyzer_union_fixtures",
    FIXTURES_PATH,
)
assert FIXTURES_SPEC is not None and FIXTURES_SPEC.loader is not None
FIXTURES = importlib.util.module_from_spec(FIXTURES_SPEC)
sys.modules[FIXTURES_SPEC.name] = FIXTURES
FIXTURES_SPEC.loader.exec_module(FIXTURES)


class JointAnalyzerIntervalUnionTest(unittest.TestCase):
    def test_streaming_producer_in_build_envelope_hole_reports_no_overlap(
        self,
    ) -> None:
        capture = FIXTURES._simt_runtime_plan_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, list)
            if row[5] != "RuntimePlanBuild":
                continue
            if int(row[0]) < FIXTURES.CORE_COUNT // 2:
                row[6], row[7] = 1_000, 1_085
            else:
                row[6], row[7] = 1_150, 1_200
        # The mapped AICPU interval is [1090, 1140], strictly inside the
        # global Build envelope but outside every per-core Build interval.
        FIXTURES._set_aicpu_timing(
            capture,
            990,
            1_040,
            minimum_fdwic_start_tick=1_000,
            maximum_fdwic_end_tick=1_250,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "build_union_hole.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            report = FIXTURES.ANALYZER_MODULE.analyze_capture(path)

        producer = report["aicpu_runtime_plan_producer"]
        self.assertEqual(producer["pipeline_relationship"], "no-overlap")
        self.assertEqual(
            producer["runtime_plan_build_interval_union"]
            ["producer_overlap"],
            {
                "minimum_cycles": 0,
                "midpoint_cycles": 0,
                "maximum_cycles": 0,
                "minimum_us": 0.0,
                "midpoint_us": 0.0,
                "maximum_us": 0.0,
            },
        )
        self.assertEqual(
            report["aicpu_aicore_joint_timing"]["build_overlap"],
            producer["runtime_plan_build_interval_union"]
            ["producer_overlap"],
        )

    def test_final_drain_overlap_uses_union_not_global_envelope(
        self,
    ) -> None:
        capture = FIXTURES._simt_runtime_plan_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, list)
            core_id = int(row[0])
            if row[5] == "RuntimePlanBuild" and (
                core_id >= FIXTURES.CORE_COUNT // 2
            ):
                row[6], row[7] = 1_000, 1_040
            elif row[5] == "FinalDrain":
                if core_id < FIXTURES.CORE_COUNT // 2:
                    row[6], row[7] = 1_208, 1_250
                else:
                    row[6], row[7] = 1_050, 1_080
        # Build union still covers [1090, 1140], while the FinalDrain union
        # has a real [1080, 1208] hole containing the producer.
        FIXTURES._set_aicpu_timing(
            capture,
            990,
            1_040,
            minimum_fdwic_start_tick=1_000,
            maximum_fdwic_end_tick=1_250,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final_drain_union_hole.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            report = FIXTURES.ANALYZER_MODULE.analyze_capture(path)

        producer = report["aicpu_runtime_plan_producer"]
        self.assertEqual(
            producer["aicore_final_drain_envelope"],
            {"start_cycle": 1_050, "end_cycle": 1_250},
        )
        final_drain_union = producer[
            "aicore_final_drain_interval_union"
        ]
        self.assertEqual(
            final_drain_union["global_interval_union_cycles"],
            72,
        )
        self.assertEqual(
            final_drain_union["producer_overlap"],
            {
                "minimum_cycles": 0,
                "midpoint_cycles": 0,
                "maximum_cycles": 0,
                "minimum_us": 0.0,
                "midpoint_us": 0.0,
                "maximum_us": 0.0,
            },
        )
        self.assertEqual(
            report["aicpu_aicore_joint_timing"][
                "final_drain_overlap"
            ],
            final_drain_union["producer_overlap"],
        )
        self.assertNotIn(
            "overlap",
            producer["runtime_plan_build_envelope"],
        )
        self.assertNotIn(
            "overlap",
            producer["aicore_final_drain_envelope"],
        )


if __name__ == "__main__":
    unittest.main()
