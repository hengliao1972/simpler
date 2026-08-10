#!/usr/bin/env python3
"""AICPU Runtime Plan 泳道 raw ABI 的最小离线门槛。"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

CONVERTER = Path(__file__).with_name("swimlane_converter.py")
SPEC = importlib.util.spec_from_file_location("aicpu_plan_swimlane_converter", CONVERTER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ANALYZER = Path(__file__).with_name("swimlane_exclusive_analyzer.py")
ANALYZER_SPEC = importlib.util.spec_from_file_location("aicpu_plan_swimlane_exclusive_analyzer", ANALYZER)
assert ANALYZER_SPEC is not None and ANALYZER_SPEC.loader is not None
ANALYZER_MODULE = importlib.util.module_from_spec(ANALYZER_SPEC)
sys.modules[ANALYZER_SPEC.name] = ANALYZER_MODULE
PREVIOUS_CONVERTER = sys.modules.get("swimlane_converter")
sys.modules["swimlane_converter"] = MODULE
try:
    ANALYZER_SPEC.loader.exec_module(ANALYZER_MODULE)
finally:
    if PREVIOUS_CONVERTER is None:
        sys.modules.pop("swimlane_converter", None)
    else:
        sys.modules["swimlane_converter"] = PREVIOUS_CONVERTER


CORE_COUNT = 96


def _topology(core_id: int) -> tuple[int, int, str]:
    if core_id < 32:
        return core_id, 0, "aic"
    vector_id = core_id - 32
    return vector_id // 2, 1 + vector_id % 2, "aiv"


def _row(
    core_id: int,
    task_id: int,
    phase: str,
    start: int,
    end: int,
    *,
    function_id: int = -1,
    auxiliary: int = 0,
) -> list[object]:
    block_id, lane, _role = _topology(core_id)
    return [
        core_id,
        block_id,
        lane,
        task_id,
        function_id,
        phase,
        start,
        end,
        0,
        auxiliary,
    ]


def _runtime_plan_capture() -> dict[str, object]:
    rows: list[list[object]] = []
    for core_id in range(CORE_COUNT):
        rows.extend(
            [
                _row(core_id, -1, "RuntimePlanBuild", 1_000, 1_200),
                _row(core_id, -1, "FinalDrain", 1_208, 1_250),
            ]
        )

    owners = (11, 9, 1, 1, 5)
    starts = (1_020, 1_022, 1_024, 1_060, 1_026)
    for task_id, core_id in enumerate(owners):
        function_id = -1 if task_id == 0 else task_id - 1
        start = starts[task_id]
        materialize_end = start + 10
        register_end = materialize_end + 6
        rows.extend(
            [
                _row(
                    core_id,
                    task_id,
                    "Materialize",
                    start,
                    materialize_end,
                    function_id=function_id,
                    auxiliary=int(task_id == 0),
                ),
                _row(
                    core_id,
                    task_id,
                    "SharedMaterializePublishTaskOutputs",
                    start + 5,
                    materialize_end - 1,
                    function_id=function_id,
                ),
                _row(
                    core_id,
                    task_id,
                    "SharedMaterializePublishTaskOutputsCopy",
                    start + 5,
                    start + 6,
                    function_id=function_id,
                ),
                _row(
                    core_id,
                    task_id,
                    "SharedMaterializePublishTaskOutputsFlush",
                    start + 6,
                    materialize_end - 2,
                    function_id=function_id,
                ),
                _row(
                    core_id,
                    task_id,
                    "Register",
                    materialize_end,
                    register_end,
                    function_id=function_id,
                ),
                _row(
                    core_id,
                    task_id,
                    "SharedRegisterPublishMetadata",
                    materialize_end + 1,
                    register_end - 2,
                    function_id=function_id,
                ),
            ]
        )
        if task_id == 0:
            rows.append(
                _row(
                    core_id,
                    task_id,
                    "AllocComplete",
                    register_end,
                    register_end + 4,
                )
            )
        else:
            rows.extend(
                [
                    _row(
                        core_id,
                        task_id,
                        "Fanin",
                        register_end,
                        register_end + 3,
                        function_id=function_id,
                    ),
                    _row(
                        core_id,
                        task_id,
                        "WinnerBuild",
                        register_end + 3,
                        register_end + 9,
                        function_id=function_id,
                    ),
                ]
            )

    rows.append(
        _row(
            0,
            1,
            "Kernel",
            1_210,
            1_215,
            function_id=0,
        )
    )

    return {
        "l2_swimlane_level": 1,
        "metadata": {
            "clock_freq_hz": 1_000_000_000,
            "num_cores": CORE_COUNT,
            "trace_schema_version": 5,
            "tensormap_mode": "shared",
            "submit_topology": "aicpu_plan_central_build",
            "core_types": [_topology(core_id)[2] for core_id in range(CORE_COUNT)],
            "shared_metadata_writer_tasks": [4],
            "shared_metadata_prefix_tasks": list(range(5)),
            "fdwic_summary": {
                "records": len(rows),
                "atomic_records": 0,
                "clock_baseline_records": 0,
                "atomic_calls": 0,
                "batched_poll_calls": 0,
                "poll_batch_records": 0,
                "dropped_records": 0,
            },
        },
        "fdwic_events": rows,
    }


class RuntimePlanConverterAbiTest(unittest.TestCase):
    def test_runtime_plan_phase_and_topology_are_explicit(self) -> None:
        self.assertEqual(
            MODULE.PHASE_NAMES["RuntimePlanBuild"],
            "runtime_plan_build",
        )
        self.assertIn("RuntimePlanBuild", MODULE.V5_PHASES)
        self.assertIn(
            "aicpu_plan_central_build",
            MODULE.CENTRAL_BUILD_TOPOLOGIES,
        )
        self.assertIn(
            "aicpu_plan_central_build",
            MODULE.STRICT_INSERT_TOPOLOGIES,
        )

    def test_runtime_plan_atomic_sites_are_append_only_and_complete(self) -> None:
        expected_names = {
            58: "runtime_plan_fatal_load",
            59: "runtime_plan_fatal_publish",
            60: "runtime_plan_closed_load",
            61: "runtime_plan_frontier_load",
            62: "runtime_plan_build_next_fetch_add",
            63: "runtime_plan_build_next_load",
            64: "runtime_plan_cell_control_load",
            65: "runtime_plan_workers_done_fetch_add",
            66: "runtime_plan_workers_done_load",
            67: "runtime_plan_build_release_load",
            68: "runtime_plan_build_release_publish",
            69: "runtime_plan_last_insert_completion_load",
        }
        self.assertEqual(
            {site: MODULE.ATOMIC_SITE_NAMES[site] for site in range(58, 70)},
            expected_names,
        )
        self.assertEqual(
            {site: MODULE.ATOMIC_SITE_OP_IDS[site] for site in range(58, 70)},
            {
                58: 0,
                59: 1,
                60: 0,
                61: 0,
                62: 2,
                63: 0,
                64: 0,
                65: 2,
                66: 0,
                67: 0,
                68: 1,
                69: 0,
            },
        )
        self.assertEqual(
            MODULE.RUNTIME_PLAN_ATOMIC_SITE_IDS,
            set(range(58, 70)),
        )
        self.assertTrue({58, 60, 67}.issubset(MODULE.POLL_BATCH_SITE_OP_IDS))
        self.assertTrue({59, 68}.issubset(MODULE.ATOMIC_RESULT_UNUSED_SITE_IDS))


class RuntimePlanExclusiveAnalyzerTest(unittest.TestCase):
    def test_central_build_closes_without_legacy_replay_or_claim(self) -> None:
        capture = _runtime_plan_capture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "l2_swimlane_records.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            report = ANALYZER_MODULE.analyze_capture(path)

        self.assertEqual(report["validation"]["status"], "PASS")
        self.assertEqual(report["capture"]["task_count_global"], 5)
        self.assertEqual(report["capture"]["planned_build_count"], 5)
        self.assertNotIn("winner_loser_actor_closure", report)
        self.assertNotIn("global_submit_makespan", report)
        self.assertEqual(report["validation"]["submit_records"], 0)
        self.assertEqual(report["validation"]["claim_records"], 0)
        self.assertEqual(report["overlays"]["Replay"]["event_count"], 0)

        metrics = report["aggregate_core_work"]["metrics_cycles"]
        self.assertEqual(metrics["runtime_plan_build"], 19_200)
        self.assertEqual(metrics["planned_build_union"], 120)
        self.assertEqual(metrics["runtime_plan_build_residual"], 19_080)
        self.assertEqual(metrics["build_to_final_drain_residual"], 768)
        self.assertEqual(metrics["final_drain"], 4_032)
        self.assertEqual(metrics["worker_completion"], 24_000)
        self.assertEqual(
            report["kernel_containment"]["inside_final_drain_events"],
            1,
        )
        self.assertEqual(report["per_core"][1]["planned_build_count"], 2)
        self.assertEqual(report["per_core"][0]["planned_build_count"], 0)
        self.assertEqual(
            report["per_core"][0]["metrics_cycles"]["runtime_plan_build_residual"],
            200,
        )
        self.assertEqual(
            report["materialize_breakdown"]["aggregate_core_work"]["metrics_cycles"]["parent"],
            50,
        )
        self.assertEqual(
            report["register_breakdown"]["aggregate_core_work"]["metrics_cycles"]["parent"],
            30,
        )
        self.assertIs(
            report["aggregate_core_work"]["closure"]["runtime_plan_build_partition"]["exact"],
            True,
        )
        self.assertEqual(
            report["semantics"]["runtime_plan_build_children"],
            [
                "Materialize",
                "Register",
                "Fanin",
                "WinnerBuild",
                "AllocComplete",
                "RuntimePlanBuildResidual",
            ],
        )

    def test_central_build_rejects_final_drain_overlap(self) -> None:
        capture = _runtime_plan_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        final_drain = next(row for row in rows if row[0] == 0 and row[5] == "FinalDrain")
        final_drain[6] = 1_199
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "l2_swimlane_records.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "FinalDrain overlaps RuntimePlanBuild",
            ):
                ANALYZER_MODULE.analyze_capture(path)

    def test_central_build_rejects_legacy_register_output_placement(
        self,
    ) -> None:
        capture = _runtime_plan_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        for task_id in range(5):
            metadata = next(row for row in rows if row[3] == task_id and row[5] == "SharedRegisterPublishMetadata")
            output_rows = [
                row
                for row in rows
                if row[3] == task_id and str(row[5]).startswith("SharedMaterializePublishTaskOutputs")
            ]
            output_rows.sort(key=lambda row: len(str(row[5])))
            output, copy, flush = output_rows
            output[5] = "SharedRegisterPublishTaskOutputs"
            copy[5] = "SharedRegisterPublishTaskOutputsCopy"
            flush[5] = "SharedRegisterPublishTaskOutputsFlush"
            output[6:8] = [metadata[6], metadata[7]]
            copy[6:8] = [metadata[6], int(metadata[6]) + 1]
            flush[6:8] = [int(metadata[6]) + 1, metadata[7]]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "l2_swimlane_records.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "requires Materialize-owned task-output publication",
            ):
                ANALYZER_MODULE.analyze_capture(path)


if __name__ == "__main__":
    unittest.main()
