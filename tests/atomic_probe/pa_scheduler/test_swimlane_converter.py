#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""standalone 泳道转换器的最小布局回归。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    # `python -m unittest tests.atomic_probe...` 以 namespace package 导入。
    from .swimlane_converter import _derive_v4_task_kinds, convert
except ImportError:
    # 也保留从本目录直接执行脚本的用法。
    from swimlane_converter import _derive_v4_task_kinds, convert


def _standalone_topology(core_id: int) -> tuple[int, int, str]:
    """返回 standalone 固定 32 AIC + 64 AIV 拓扑中的 block/lane/type。"""
    if core_id < 32:
        return core_id, 0, "aic"
    vector_id = core_id - 32
    return vector_id // 2, 1 + vector_id % 2, "aiv"


def _v3_capture(
    rows: list[list[object]],
    *,
    num_cores: int = 1,
    add_clock_baselines: bool = True,
    dependency_applied: bool = True,
) -> dict[str, object]:
    """构造带 producer weighted summary 的最小 schema-v3 raw。"""
    all_rows = [list(row) for row in rows]
    if add_clock_baselines:
        dependency_flags = 0x3 if dependency_applied else 0x1
        for core_id in range(num_cores):
            block_id, lane, _ = _standalone_topology(core_id)
            start = 10 + core_id * 4
            all_rows.extend(
                [
                    [
                        core_id,
                        block_id,
                        lane,
                        -1,
                        -1,
                        "ClockBaseline",
                        start,
                        start + 1,
                        0,
                        0,
                    ],
                    [
                        core_id,
                        block_id,
                        lane,
                        -1,
                        -1,
                        "ClockBaseline",
                        start + 2,
                        start + 3,
                        dependency_flags,
                        0,
                    ],
                ]
            )
    atomic_rows = [row for row in all_rows if row[5] == "Atomic"]
    batch_rows = [row for row in atomic_rows if int(row[8]) & 0x80]
    batch_calls = sum((int(row[8]) >> 8) & 0xFFFFFF for row in batch_rows)
    core_types = [_standalone_topology(core_id)[2] for core_id in range(num_cores)]
    summary = {
        "records": len(all_rows),
        "atomic_records": len(atomic_rows),
        "clock_baseline_records": sum(
            row[5] == "ClockBaseline" for row in all_rows
        ),
        "atomic_calls": len(atomic_rows) - len(batch_rows) + batch_calls,
        "batched_poll_calls": batch_calls,
        "poll_batch_records": len(batch_rows),
        "dropped_records": 0,
    }
    return {
        "l2_swimlane_level": 4,
        "metadata": {
            "clock_freq_hz": 1_000_000_000,
            "num_cores": num_cores,
            "trace_schema_version": 3,
            "core_types": core_types,
            "fdwic_summary": summary,
        },
        "fdwic_events": all_rows,
    }


def _v4_capture(
    rows: list[list[object]],
    *,
    num_cores: int = 1,
    add_parents: bool = True,
    tensormap_mode: str = "private",
) -> dict[str, object]:
    """构造 phase-only schema-v4 raw；调用者显式提供 Claim/Submit/尾动作。"""
    all_rows = [list(row) for row in rows]
    if add_parents:
        for core_id in range(num_cores):
            block_id, lane, _ = _standalone_topology(core_id)
            submit_ends = [
                int(row[7])
                for row in all_rows
                if int(row[0]) == core_id and row[5] == "Submit"
            ]
            orchestration_end = max([200, *submit_ends])
            all_rows.extend(
                [
                    [
                        core_id,
                        block_id,
                        lane,
                        -1,
                        -1,
                        "OrchestrationReplay",
                        90,
                        orchestration_end,
                        0,
                        0,
                    ],
                    [
                        core_id,
                        block_id,
                        lane,
                        -1,
                        -1,
                        "FinalDrain",
                        orchestration_end,
                        orchestration_end + 20,
                        0,
                        0,
                    ],
                ]
            )
    return {
        "l2_swimlane_level": 1,
        "metadata": {
            "clock_freq_hz": 1_000_000_000,
            "num_cores": num_cores,
            "trace_schema_version": 4,
            "tensormap_mode": tensormap_mode,
            "core_types": [
                _standalone_topology(core_id)[2] for core_id in range(num_cores)
            ],
            "fdwic_summary": {
                "records": len(all_rows),
                "atomic_records": 0,
                "clock_baseline_records": 0,
                "atomic_calls": 0,
                "batched_poll_calls": 0,
                "poll_batch_records": 0,
                "dropped_records": 0,
            },
        },
        "fdwic_events": all_rows,
    }


class SwimlaneConverterLayoutTest(unittest.TestCase):
    def test_v4_requires_explicit_tensormap_mode(self) -> None:
        capture = _v4_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 120, 0x3, 1],
                [0, 0, 0, 0, -1, "AllocComplete", 120, 130, 0, 0],
                [0, 0, 0, 0, -1, "Submit", 100, 140, 1, 1],
            ]
        )
        metadata = capture["metadata"]
        assert isinstance(metadata, dict)
        metadata.pop("tensormap_mode")
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tensormap_mode"):
                convert(input_path, output_path)

    def test_v4_shared_rejects_private_prepare_map_record(self) -> None:
        capture = _v4_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 115, 0x3, 1],
                [0, 0, 0, 0, -1, "Materialize", 116, 120, 0, 1],
                [0, 0, 0, 0, -1, "PrepareMap", 120, 120, 0, 1],
                [0, 0, 0, 0, -1, "Register", 121, 125, 0, 0],
                [0, 0, 0, 0, -1, "AllocComplete", 126, 130, 0, 0],
                [0, 0, 0, 0, -1, "Submit", 100, 140, 1, 1],
            ],
            tensormap_mode="shared",
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not contain PrepareMap"):
                convert(input_path, output_path)

    def test_v4_shared_register_detail_splits_parent_with_one_raw_row(self) -> None:
        rows = [
            [0, 0, 0, 0, -1, "Claim", 110, 115, 0x3, 1],
            [0, 0, 0, 0, -1, "Materialize", 115, 120, 0, 1],
            [0, 0, 0, 0, -1, "Register", 120, 140, 0, 0],
            [
                0,
                0,
                0,
                0,
                -1,
                "SharedRegisterPublishMetadata",
                124,
                134,
                0,
                0,
            ],
            [0, 0, 0, 0, -1, "AllocComplete", 140, 145, 0, 0],
            [0, 0, 0, 0, -1, "Submit", 100, 150, 1, 1],
        ]
        capture = _v4_capture(rows, tensormap_mode="shared")
        raw_rows = capture["fdwic_events"]
        assert isinstance(raw_rows, list)
        # 两条 parent 由 helper 加入；Register 拆分本身只多一个设备 raw。
        self.assertEqual(len(raw_rows), len(rows) + 2)
        self.assertEqual(
            sum(row[5] == "SharedRegisterPublishMetadata" for row in raw_rows),
            1,
        )

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            events = json.loads(output_path.read_text(encoding="utf-8"))["traceEvents"]

        parent = next(event for event in events if event.get("name") == "register#0")
        child_names = (
            "register.wait_insert_turn#0",
            "register.publish_metadata#0",
            "register.handoff_next_turn#0",
        )
        children = {
            event["name"]: event
            for event in events
            if event.get("name") in child_names
        }
        self.assertEqual(set(children), set(child_names))
        for child in children.values():
            self.assertEqual(
                set(child), {"ph", "name", "pid", "tid", "ts", "dur"}
            )
        self.assertEqual(
            children["register.wait_insert_turn#0"]["ts"], parent["ts"]
        )
        self.assertAlmostEqual(
            children["register.wait_insert_turn#0"]["ts"]
            + children["register.wait_insert_turn#0"]["dur"],
            children["register.publish_metadata#0"]["ts"],
        )
        self.assertAlmostEqual(
            children["register.publish_metadata#0"]["ts"]
            + children["register.publish_metadata#0"]["dur"],
            children["register.handoff_next_turn#0"]["ts"],
        )
        self.assertAlmostEqual(
            children["register.handoff_next_turn#0"]["ts"]
            + children["register.handoff_next_turn#0"]["dur"],
            parent["ts"] + parent["dur"],
        )
        self.assertAlmostEqual(
            sum(child["dur"] for child in children.values()), parent["dur"]
        )

    def test_v4_shared_register_detail_is_required_exactly_once_for_winner(
        self,
    ) -> None:
        base_rows = [
            [0, 0, 0, 0, -1, "Claim", 110, 115, 0x3, 1],
            [0, 0, 0, 0, -1, "Register", 120, 140, 0, 0],
            [0, 0, 0, 0, -1, "AllocComplete", 140, 145, 0, 0],
            [0, 0, 0, 0, -1, "Submit", 100, 150, 1, 1],
        ]
        detail = [
            0,
            0,
            0,
            0,
            -1,
            "SharedRegisterPublishMetadata",
            124,
            134,
            0,
            0,
        ]
        cases = {
            "missing": base_rows,
            "duplicate": [*base_rows, detail, list(detail)],
        }
        for label, rows in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                capture = _v4_capture(rows, tensormap_mode="shared")
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError, "requires exactly one SharedRegisterPublishMetadata"
                ):
                    convert(input_path, output_path)

    def test_v4_shared_register_detail_rejects_bad_boundary_or_identity(
        self,
    ) -> None:
        base_rows = [
            [0, 0, 0, 0, -1, "Claim", 110, 115, 0x3, 1],
            [0, 0, 0, 0, -1, "Register", 120, 140, 0, 0],
            [0, 0, 0, 0, -1, "AllocComplete", 140, 145, 0, 0],
            [0, 0, 0, 0, -1, "Submit", 100, 150, 1, 1],
        ]
        cases = {
            "outside_parent": [
                0,
                0,
                0,
                0,
                -1,
                "SharedRegisterPublishMetadata",
                119,
                134,
                0,
                0,
            ],
            "different_function": [
                0,
                0,
                0,
                0,
                0,
                "SharedRegisterPublishMetadata",
                124,
                134,
                0,
                0,
            ],
        }
        for label, detail in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                capture = _v4_capture(
                    [*base_rows, detail], tensormap_mode="shared"
                )
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                expected = (
                    "outside Register parent"
                    if label == "outside_parent"
                    else "identity differs"
                )
                with self.assertRaisesRegex(ValueError, expected):
                    convert(input_path, output_path)

    def test_v4_rejects_shared_register_detail_in_private_mode(self) -> None:
        capture = _v4_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 115, 0x3, 1],
                [0, 0, 0, 0, -1, "Register", 120, 140, 0, 0],
                [
                    0,
                    0,
                    0,
                    0,
                    -1,
                    "SharedRegisterPublishMetadata",
                    124,
                    134,
                    0,
                    0,
                ],
                [0, 0, 0, 0, -1, "AllocComplete", 140, 145, 0, 0],
                [0, 0, 0, 0, -1, "Submit", 100, 150, 1, 1],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "only valid for shared TensorMap"
            ):
                convert(input_path, output_path)

    def test_v4_shared_register_detail_is_forbidden_for_loser(self) -> None:
        capture = _v4_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 115, 0x2, 1],
                [
                    0,
                    0,
                    0,
                    0,
                    -1,
                    "SharedRegisterPublishMetadata",
                    124,
                    134,
                    0,
                    0,
                ],
                [0, 0, 0, 0, -1, "Submit", 100, 150, 0, 1],
            ],
            tensormap_mode="shared",
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "and none for losers"):
                convert(input_path, output_path)

    def test_v4_shared_register_parent_is_forbidden_for_loser(self) -> None:
        capture = _v4_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 115, 0x2, 1],
                [0, 0, 0, 0, -1, "Register", 120, 140, 0, 0],
                [0, 0, 0, 0, -1, "Submit", 100, 150, 0, 1],
            ],
            tensormap_mode="shared",
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "Register parent for each winner and none for losers"
            ):
                convert(input_path, output_path)

    def test_v4_shared_rejects_register_parent_without_claim(self) -> None:
        capture = _v4_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 115, 0x3, 1],
                [0, 0, 0, 0, -1, "Register", 120, 140, 0, 0],
                [
                    0,
                    0,
                    0,
                    0,
                    -1,
                    "SharedRegisterPublishMetadata",
                    124,
                    134,
                    0,
                    0,
                ],
                [0, 0, 0, 0, -1, "AllocComplete", 140, 145, 0, 0],
                [0, 0, 0, 0, -1, "Submit", 100, 150, 1, 1],
                # task 9 没有 Claim/Submit，不能让独立 converter 静默接收。
                [0, 0, 0, 9, -1, "Register", 151, 152, 0, 0],
            ],
            tensormap_mode="shared",
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "Register parents have no matching Claim"
            ):
                convert(input_path, output_path)

    def test_v4_splits_internal_and_tail_residual_without_repeated_fields(self) -> None:
        capture = _v4_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 120, 0x3, 1],
                [0, 0, 0, 0, -1, "AllocComplete", 120, 130, 0, 0],
                [0, 0, 0, 0, -1, "Submit", 100, 140, 1, 1],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            merged = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(merged["metadata"]["trace_schema_version"], 4)
        events = merged["traceEvents"]
        orchestration = next(
            event for event in events if event.get("name") == "orchestration_replay"
        )
        self.assertNotIn("args", orchestration)
        self.assertNotIn("cat", orchestration)
        residuals = [event for event in events if event.get("name") == "submit_residual"]
        tails = [
            event for event in events if event.get("name") == "submit_tail_gap"
        ]
        self.assertEqual(
            residuals,
            [
                {"ph": "X", "name": "submit_residual", "pid": 0, "tid": 0, "ts": 0.01, "dur": 0.01},
            ],
        )
        self.assertEqual(
            tails,
            [
                {
                    "ph": "X",
                    "name": "submit_tail_gap",
                    "pid": 0,
                    "tid": 0,
                    "ts": 0.04,
                    "dur": 0.01,
                }
            ],
        )
        for event in (*residuals, *tails):
            self.assertEqual(set(event), {"ph", "name", "pid", "tid", "ts", "dur"})

    def test_v4_marks_between_submit_gap_without_loser_marker(self) -> None:
        capture = _v4_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 120, 0x2, 1],
                [0, 0, 0, 0, -1, "Submit", 100, 140, 0, 1],
                [0, 0, 0, 1, -1, "Claim", 170, 180, 0x2, 0],
                [0, 0, 0, 1, -1, "Submit", 160, 200, 0, 0],
                [0, 0, 0, 2, -1, "Claim", 210, 220, 0x2, 0],
                [0, 0, 0, 2, -1, "Submit", 200, 240, 0, 0],
                [0, 0, 0, 3, -1, "Claim", 250, 260, 0x2, 0],
                [0, 0, 0, 3, -1, "Submit", 240, 280, 0, 0],
                [0, 0, 0, 4, -1, "Claim", 290, 300, 0x2, 0],
                [0, 0, 0, 4, -1, "Submit", 280, 320, 0, 0],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            events = json.loads(output_path.read_text(encoding="utf-8"))["traceEvents"]

        internal = [event for event in events if event.get("name") == "submit_residual"]
        tails = [
            event for event in events if event.get("name") == "submit_tail_gap"
        ]
        between = [
            event for event in events if event.get("name") == "between_submit_residual"
        ]
        self.assertEqual(len(internal), 5)
        self.assertEqual(len(tails), 5)
        self.assertEqual(
            between,
            [
                {
                    "ph": "X",
                    "name": "between_submit_residual",
                    "pid": 0,
                    "tid": 0,
                    "ts": 0.05,
                    "dur": 0.02,
                }
            ],
        )
        # 每个完整 G1 task 仍只生成真实补集，不增加设备 raw 记录。
        self.assertEqual(len(internal) + len(tails) + len(between), 11)
        for event in (*internal, *tails, *between):
            self.assertEqual(set(event), {"ph", "name", "pid", "tid", "ts", "dur"})

    def test_v4_rejects_legacy_lap_phase(self) -> None:
        capture = _v4_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 120, 0, 0],
                [0, 0, 0, 0, -1, "Submit", 100, 140, 0, 1],
                [0, 0, 0, 0, -1, "Replay", 100, 120, 0, 0],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbids legacy lap phase"):
                convert(input_path, output_path)

    def test_v4_rejects_unused_drain_won_phase(self) -> None:
        capture = _v4_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 120, 0, 0],
                [0, 0, 0, 0, -1, "Submit", 100, 140, 0, 0],
                [0, 0, 0, 0, -1, "DrainWon", 121, 122, 0, 0],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbids unused legacy phase"):
                convert(input_path, output_path)

    def test_v4_rejects_missing_winner_tail(self) -> None:
        capture = _v4_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 120, 0x3, 1],
                [0, 0, 0, 0, -1, "Submit", 100, 140, 1, 1],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tail mismatch"):
                convert(input_path, output_path)

    def test_v4_rejects_task_kind_that_disagrees_with_task_id(self) -> None:
        capture = _v4_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 101, 102, 0, 1],
                [0, 0, 0, 0, -1, "Submit", 100, 105, 0, 1],
                # Submit 给出的动态 plan 是 QK；Claim 篡改成 Alloc。
                [0, 0, 0, 1, -1, "Claim", 111, 112, 0, 1],
                [0, 0, 0, 1, -1, "Submit", 110, 115, 0, 0],
                [0, 0, 0, 2, -1, "Claim", 121, 122, 0, 0],
                [0, 0, 0, 2, -1, "Submit", 120, 125, 0, 0],
                [0, 0, 0, 3, -1, "Claim", 131, 132, 0, 0],
                [0, 0, 0, 3, -1, "Submit", 130, 135, 0, 0],
                [0, 0, 0, 4, -1, "Claim", 141, 142, 0, 0],
                [0, 0, 0, 4, -1, "Submit", 140, 145, 0, 0],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "task-kind mismatch"):
                convert(input_path, output_path)

    def test_v4_derives_g0_g1_g2_g4_and_mixed_task_kinds(self) -> None:
        """动态类型来自 Alloc 边界，不再把全局 task_id 按五取模。"""

        # mixed=[G0,G1,G2,G4]，各 batch 的 task 数为 1/5/9/17。
        group_counts = (0, 1, 2, 4)
        alloc_task_ids: list[int] = []
        expected_kinds: dict[int, int] = {}
        next_task = 0
        for group_count in group_counts:
            alloc_task_ids.append(next_task)
            expected_kinds[next_task] = 0
            next_task += 1
            for _group in range(group_count):
                for kind_id in range(1, 5):
                    expected_kinds[next_task] = kind_id
                    next_task += 1

        semantics = {
            (core_id, task_id): (
                False,
                task_id in alloc_task_ids,
            )
            for core_id in range(3)
            for task_id in range(next_task)
        }
        self.assertEqual(_derive_v4_task_kinds(semantics, 3), expected_kinds)
        # G0 后紧邻下一个 batch Alloc，形成合法 Alloc->Alloc；随后 task 5
        # 是该 G1 的 UP。旧全局 task_id % 5 会把两者分别错判为 QK/Alloc。
        self.assertEqual([expected_kinds[0], expected_kinds[1]], [0, 0])
        self.assertEqual(expected_kinds[5], 4)

    def test_v4_rejects_cross_core_alloc_marker_disagreement(self) -> None:
        semantics = {
            (core_id, task_id): (False, task_id == 0)
            for core_id in range(2)
            for task_id in range(5)
        }
        semantics[(1, 1)] = (False, True)
        with self.assertRaisesRegex(ValueError, "Alloc marker differs across cores"):
            _derive_v4_task_kinds(semantics, 2)

    def test_v4_rejects_incomplete_dynamic_group(self) -> None:
        semantics = {
            (0, task_id): (False, task_id == 0)
            for task_id in range(4)
        }
        with self.assertRaisesRegex(ValueError, "complete QK/SF/PV/UP groups"):
            _derive_v4_task_kinds(semantics, 1)

    def test_v4_requires_both_parent_spans(self) -> None:
        capture = _v4_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 120, 0, 0],
                [0, 0, 0, 0, -1, "Submit", 100, 140, 0, 1],
            ],
            add_parents=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires exactly one schema-v4"):
                convert(input_path, output_path)

    def test_v4_rejects_removed_loser_replay_phase(self) -> None:
        capture = _v4_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 120, 0x2, 1],
                [0, 0, 0, 0, -1, "LoserReplay", 120, 120, 0, 0],
                [0, 0, 0, 0, -1, "Submit", 100, 140, 0, 1],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown phase 'LoserReplay'"):
                convert(input_path, output_path)

    def test_real_compute_metadata_is_preserved_and_visible(self) -> None:
        # raw 与 merged 都必须自描述真实 engine 负载；否则同名 QK/SF/PV/UP
        # span 无法与历史 scalar-NOP 泳道区分。
        workload = {
            "mode": "real-compute",
            "counts": {"qk": 6, "sf": 28, "pv": 4, "up": 1},
            "unit": "complete_128x128_engine_pipeline_iteration",
            "input_pattern": "layout-diagnostic",
            "engine_mapping": {
                "qk": "cube_matmul",
                "sf": "vector_add",
                "pv": "cube_matmul",
                "up": "vector_mul",
            },
        }
        capture = {
            "l2_swimlane_level": 1,
            "metadata": {
                "clock_freq_hz": 1_000_000_000,
                "num_cores": 1,
                "trace_schema_version": 2,
                "winner_workload": workload,
                "core_types": ["AIC"],
            },
            "fdwic_events": [[0, 0, 0, 1, 0, "Kernel", 100, 200, 0, 0]],
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            emitted, blocks, base_cycle = convert(input_path, output_path)
            merged = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual((emitted, blocks, base_cycle), (2, 1, 100))
        self.assertEqual(merged["metadata"]["winner_workload"], workload)
        capture_event = next(
            event for event in merged["traceEvents"]
            if event.get("name") == "pa_scheduler.capture"
        )
        self.assertEqual(capture_event["args"]["winner_workload"], workload)

    def test_invalid_real_compute_input_pattern_is_rejected(self) -> None:
        capture = {
            "l2_swimlane_level": 1,
            "metadata": {
                "clock_freq_hz": 1_000_000_000,
                "num_cores": 1,
                "trace_schema_version": 2,
                "winner_workload": {
                    "mode": "real-compute",
                    "counts": {"qk": 1, "sf": 1, "pv": 1, "up": 1},
                    "unit": "complete_128x128_engine_pipeline_iteration",
                    "input_pattern": "unknown-layout",
                    "engine_mapping": {
                        "qk": "cube_matmul",
                        "sf": "vector_add",
                        "pv": "cube_matmul",
                        "up": "vector_mul",
                    },
                },
                "core_types": ["AIC"],
            },
            "fdwic_events": [[0, 0, 0, 1, 0, "Kernel", 100, 200, 0, 0]],
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input_pattern is invalid"):
                convert(input_path, output_path)
            self.assertFalse(output_path.exists())

    def test_atomic_and_clock_share_the_scalar_lane(self) -> None:
        # 同一 mixed block 放一条 AIC 和一条 AIV0；Atomic 是 AIC scalar
        # 上 Claim 的子区间，ClockBaseline 也是 AIV0 scalar 指令，而
        # Kernel 是 AIV0 计算单元上的独立区间。
        capture = {
            "l2_swimlane_level": 4,
            "metadata": {
                "clock_freq_hz": 1_000_000_000,
                "num_cores": 2,
                "trace_schema_version": 2,
                "core_types": ["AIC", "AIV"],
            },
            "fdwic_events": [
                # Claim flags: attempted(bit1)，本核输了所以 winner(bit0)=0。
                [0, 0, 0, 7, -1, "Claim", 100, 200, 0x2, 0],
                # flags: FetchMax(3) | result-used(bit4) | return-ready(bit6)。
                [0, 0, 0, 7, -1, "Atomic", 120, 160, 0x53, 4],
                # Exchange(1) 的旧值未消费，只能标 source-issue。
                [0, 0, 0, 7, -1, "Atomic", 161, 162, 0x01, 7],
                # flags: dependency-hook(bit0) | dependency-applied(bit1)。
                [1, 0, 1, -1, -1, "ClockBaseline", 101, 102, 0x3, 0],
                [1, 0, 1, 7, 0, "Kernel", 140, 180, 0, 0],
                # 同一 AIV0 上的 role-filtered Claim，没有 atomic。
                [1, 0, 1, 8, -1, "Claim", 201, 220, 0x0, 0],
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            emitted, blocks, base_cycle = convert(input_path, output_path)
            merged = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual((emitted, blocks, base_cycle), (6, 1, 100))
        events = merged["traceEvents"]
        thread_names = {
            (event["pid"], event["tid"]): event["args"]["name"]
            for event in events
            if event.get("ph") == "M" and event.get("name") == "thread_name"
        }
        self.assertEqual(thread_names[(0, 0)], "AIC (core0)")
        self.assertEqual(thread_names[(0, 1)], "AIV0 (core1)")
        self.assertEqual(thread_names[(0, 4)], "AIV0·kernel (core1)")
        self.assertFalse(any("·atomic" in name for name in thread_names.values()))

        ready_atomic = next(event for event in events if event.get("cat") == "atomic.return_ready")
        issue_atomic = next(event for event in events if event.get("cat") == "atomic.source_issue")
        clock = next(event for event in events if event.get("cat") == "scalar_clock")
        kernel = next(event for event in events if event.get("name") == "QK#7")
        self.assertEqual((ready_atomic["pid"], ready_atomic["tid"]), (0, 0))
        self.assertEqual((issue_atomic["pid"], issue_atomic["tid"]), (0, 0))
        self.assertEqual((clock["pid"], clock["tid"]), (0, 1))
        self.assertEqual((kernel["pid"], kernel["tid"]), (0, 4))
        self.assertEqual(
            ready_atomic["name"], "atomic.return_ready.claim_max.fetch_max#7"
        )
        self.assertEqual(
            issue_atomic["name"], "atomic.source_issue.completion_flag_exchange.exchange#7"
        )
        self.assertEqual(ready_atomic["args"]["execution_unit"], "scalar")
        self.assertEqual(issue_atomic["args"]["execution_unit"], "scalar")
        self.assertEqual(clock["args"]["execution_unit"], "scalar")
        self.assertEqual(ready_atomic["args"]["completion_boundary"], "return_value_ready")
        self.assertEqual(issue_atomic["args"]["completion_boundary"], "source_issue_bracket")
        attempted_claim = next(event for event in events if event.get("name") == "claim.lost#7")
        skipped_claim = next(event for event in events if event.get("name") == "claim.not_attempted#8")
        self.assertTrue(attempted_claim["args"]["claim_attempted"])
        self.assertFalse(skipped_claim["args"]["claim_attempted"])
        self.assertEqual(attempted_claim["args"]["claim_attempted_source"], "raw_flag")

    def test_v1_claim_attempt_uses_contained_atomic_evidence(self) -> None:
        # 历史 raw 没有 attempted bit。只在同一 capture 真有 claim_max 记录时
        # 恢复该语义，不根据 task_id 或 core role 猜测。
        capture = {
            "l2_swimlane_level": 1,
            "metadata": {
                "clock_freq_hz": 1_000_000_000,
                "num_cores": 1,
                "core_types": ["AIC"],
            },
            "fdwic_events": [
                [0, 0, 0, 1, -1, "Claim", 100, 200, 0, 0],
                [0, 0, 0, 1, -1, "Atomic", 120, 160, 0x50, 4],
                [0, 0, 0, 2, -1, "Claim", 210, 230, 0, 0],
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            events = json.loads(output_path.read_text(encoding="utf-8"))["traceEvents"]

        attempted = next(event for event in events if event.get("name") == "claim.lost#1")
        unknown = next(event for event in events if event.get("name") == "claim#2")
        self.assertTrue(attempted["args"]["claim_attempted"])
        self.assertIsNone(unknown["args"]["claim_attempted"])
        self.assertEqual(attempted["args"]["claim_attempted_source"], "contained_claim_max")
        self.assertEqual(
            unknown["args"]["claim_attempted_source"],
            "unknown_v1_without_matching_claim_max",
        )

    def test_v2_claim_states_do_not_require_atomic_records(self) -> None:
        # v2 raw 直接携带 attempted/won，因此关闭 --trace-atomics 后仍能
        # 区分三种 Claim 状态，不依赖 converter 从业务拓扑推断。
        capture = {
            "l2_swimlane_level": 1,
            "metadata": {
                "clock_freq_hz": 1_000_000_000,
                "num_cores": 1,
                "trace_schema_version": 2,
                "core_types": ["AIC"],
            },
            "fdwic_events": [
                [0, 0, 0, 1, -1, "Claim", 100, 110, 0x0, 0],
                [0, 0, 0, 2, -1, "Claim", 120, 140, 0x2, 0],
                [0, 0, 0, 3, 0, "Claim", 150, 180, 0x3, 0],
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            events = json.loads(output_path.read_text(encoding="utf-8"))["traceEvents"]

        names = {event.get("name") for event in events}
        self.assertIn("claim.not_attempted#1", names)
        self.assertIn("claim.lost#2", names)
        self.assertIn("claim.won#3", names)

    def test_v2_rejects_winner_without_attempt(self) -> None:
        capture = {
            "l2_swimlane_level": 1,
            "metadata": {
                "clock_freq_hz": 1_000_000_000,
                "num_cores": 1,
                "trace_schema_version": 2,
                "core_types": ["AIC"],
            },
            "fdwic_events": [[0, 0, 0, 1, 0, "Claim", 100, 110, 0x1, 0]],
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid Claim flags"):
                convert(input_path, output_path)

    def test_v3_poll_batches_preserve_exact_calls_on_scalar_lane(self) -> None:
        # standalone 只允许六类显式等待区 observation load 聚合；每个
        # PollBatch 都必须保留精确 call_count，但不能伪装成单次延迟。
        sites = {
            1: "startup_poll",
            2: "fatal_poll",
            5: "fanin_flag_load",
            11: "heap_frontier_load",
            12: "heap_vend_load",
            14: "replay_done_poll",
        }
        call_count = 12_345
        flags = (call_count << 8) | 0x90
        for site_id, site_name in sites.items():
            with self.subTest(site=site_name), tempfile.TemporaryDirectory() as directory:
                capture = _v3_capture(
                    [[0, 0, 0, -1, -1, "Atomic", 100, 900, flags, site_id]]
                )
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                emitted, blocks, base_cycle = convert(input_path, output_path)
                merged = json.loads(output_path.read_text(encoding="utf-8"))

                self.assertEqual((emitted, blocks, base_cycle), (3, 1, 10))
                batch = next(
                    event
                    for event in merged["traceEvents"]
                    if event.get("cat") == "atomic.poll_batch"
                )
                self.assertEqual(
                    batch["name"], f"atomic.poll_batch.{site_name}.load×{call_count}"
                )
                self.assertEqual((batch["pid"], batch["tid"]), (0, 0))
                self.assertEqual(batch["args"]["call_count"], call_count)
                self.assertEqual(batch["args"]["poll_window_cycles"], 800)
                self.assertEqual(
                    batch["args"]["duration_semantics"],
                    "logical_poll_episode_envelope_not_single_atomic_latency",
                )
                self.assertEqual(
                    batch["args"]["batch_semantics"], "observation_load_calls"
                )
                self.assertTrue(
                    batch["args"]["may_contain_interleaved_direct_atomics"]
                )
                self.assertNotIn("cycles", batch["args"])
                self.assertNotIn("completion_boundary", batch["args"])
                self.assertNotIn("return_ready_observed", batch["args"])
                self.assertEqual(
                    merged["metadata"]["fdwic_summary"]["atomic_calls"],
                    call_count,
                )

    def test_v3_poll_batch_accepts_maximum_24_bit_count(self) -> None:
        call_count = 0xFFFFFF
        capture = _v3_capture(
            [[
                0,
                0,
                0,
                -1,
                -1,
                "Atomic",
                100,
                900,
                (call_count << 8) | 0x90,
                1,
            ]]
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            events = json.loads(output_path.read_text(encoding="utf-8"))["traceEvents"]

        batch = next(event for event in events if event.get("cat") == "atomic.poll_batch")
        self.assertEqual(batch["args"]["call_count"], call_count)

    def test_v3_rejects_invalid_poll_batch_schema(self) -> None:
        valid = (7 << 8) | 0x90
        cases = (
            (0x90, 5, -1, -1),  # call_count=0
            ((7 << 8) | 0x91, 5, -1, -1),  # observation 只能是 Load
            (valid, 9, -1, -1),  # frontier scan 不是显式等待区
            ((7 << 8) | 0xB0, 5, -1, -1),  # batch 没有 value_zero
            ((7 << 8) | 0xD0, 5, -1, -1),  # batch 没有 return-ready
            (valid, 5, 0, -1),  # batch 不归属单个 task
            (valid, 5, -1, 0),  # batch 不归属 kernel function
        )
        for flags, site, task_id, func_id in cases:
            with self.subTest(flags=flags, site=site), tempfile.TemporaryDirectory() as directory:
                capture = _v3_capture(
                    [[
                        0,
                        0,
                        0,
                        task_id,
                        func_id,
                        "Atomic",
                        100,
                        110,
                        flags,
                        site,
                    ]]
                )
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid Atomic PollBatch"):
                    convert(input_path, output_path)
                self.assertFalse(output_path.exists())

    def test_v2_reserves_poll_batch_flag(self) -> None:
        capture = {
            "l2_swimlane_level": 4,
            "metadata": {
                "clock_freq_hz": 1_000_000_000,
                "num_cores": 1,
                "trace_schema_version": 2,
                "core_types": ["aic"],
            },
            "fdwic_events": [
                [0, 0, 0, -1, -1, "Atomic", 100, 110, (7 << 8) | 0x90, 5]
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid Atomic PollBatch"):
                convert(input_path, output_path)

    def test_v3_rejects_invalid_direct_atomic_schema(self) -> None:
        cases = (
            (0x51, 4, -1),  # ClaimMax 的 op 必须是 FetchMax
            (0x12, 0, -1),  # StartupIncrement 不消费返回值
            (0x42, 0, -1),  # 未消费返回值不能声明 return-ready
            (0x73, 4, -1),  # value_zero 只属于 Load
            ((1 << 8) | 0x50, 1, -1),  # retry payload 只属于 FetchMax
            (0x50, 19, -1),  # 当前 AtomicSite::Count 以外的未定义站点
            (0x53, 4, 0),  # Atomic 不携带 function id
        )
        for flags, site, func_id in cases:
            with self.subTest(flags=flags, site=site), tempfile.TemporaryDirectory() as directory:
                capture = _v3_capture(
                    [[0, 0, 0, 7, func_id, "Atomic", 100, 110, flags, site]]
                )
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid direct Atomic"):
                    convert(input_path, output_path)

    def test_v3_exports_shared_heap_return_ready_sites(self) -> None:
        rows = [
            [0, 0, 0, 7, -1, "Atomic", 100, 110, 0x50, 15],
            [0, 0, 0, 7, -1, "Atomic", 111, 121, 0x50, 16],
            [0, 0, 0, 7, -1, "Atomic", 122, 132, 0x52, 17],
            [0, 0, 0, 7, -1, "Atomic", 133, 143, 0x52, 18],
        ]
        capture = _v3_capture(rows, dependency_applied=True)
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            events = json.loads(output_path.read_text(encoding="utf-8"))[
                "traceEvents"
            ]

        names = {
            event["name"]
            for event in events
            if event.get("cat") == "atomic.return_ready"
        }
        self.assertEqual(
            names,
            {
                "atomic.return_ready.shared_heap_vend_load.load#7",
                "atomic.return_ready.shared_heap_cursor_load.load#7",
                "atomic.return_ready.shared_heap_cursor_reserve.fetch_add#7",
                "atomic.return_ready.shared_heap_vend_advance.fetch_add#7",
            },
        )

    def test_v3_direct_boundary_must_match_core_clock_baseline(self) -> None:
        # baseline 声明该后端应用依赖钩子，消费返回值的直接 Atomic 却没有
        # return-ready bit；converter 必须拒绝这种自相矛盾的 raw。
        capture = _v3_capture(
            [[0, 0, 0, 7, -1, "Atomic", 100, 110, 0x13, 4]],
            dependency_applied=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match.*ClockBaseline"):
                convert(input_path, output_path)

    def test_v3_source_issue_direct_boundary_matches_cpu_baseline(self) -> None:
        # CPU/A5Sim 的依赖基线明确声明 dependency_applied=0；消费返回值的
        # direct span 因而保留 source-issue，不能被 converter 擅自升级。
        capture = _v3_capture(
            [[0, 0, 0, 7, -1, "Atomic", 100, 110, 0x13, 4]],
            dependency_applied=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            events = json.loads(output_path.read_text(encoding="utf-8"))["traceEvents"]

        direct = next(
            event for event in events if event.get("cat") == "atomic.source_issue"
        )
        self.assertEqual(direct["args"]["call_count"], 1)
        self.assertTrue(direct["args"]["result_used"])
        self.assertFalse(direct["args"]["return_ready_observed"])

    def test_v3_rejects_invalid_clock_baseline_schema(self) -> None:
        cases = (
            (0x2, -1, -1, 0),  # applied 不能脱离 dependency bit
            (0x4, -1, -1, 0),  # 未定义 flag
            (0x0, 0, -1, 0),  # baseline 不归属 task
            (0x0, -1, 0, 0),  # baseline 不归属 function
            (0x0, -1, -1, 1),  # aux 必须为零
        )
        for flags, task_id, func_id, auxiliary in cases:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                capture = _v3_capture([], add_clock_baselines=False)
                capture["fdwic_events"] = [
                    [
                        0,
                        0,
                        0,
                        task_id,
                        func_id,
                        "ClockBaseline",
                        10,
                        11,
                        flags,
                        auxiliary,
                    ]
                ]
                summary = capture["metadata"]["fdwic_summary"]
                summary["records"] = 1
                summary["clock_baseline_records"] = 1
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid ClockBaseline"):
                    convert(input_path, output_path)

    def test_v3_requires_two_clock_baselines_per_core(self) -> None:
        capture = _v3_capture([], add_clock_baselines=False)
        capture["fdwic_events"] = [
            [0, 0, 0, -1, -1, "ClockBaseline", 10, 11, 0, 0]
        ]
        summary = capture["metadata"]["fdwic_summary"]
        summary["records"] = 1
        summary["clock_baseline_records"] = 1
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires exactly one plain"):
                convert(input_path, output_path)

    def test_v3_rejects_each_broken_weighted_summary_field(self) -> None:
        rows = [
            [0, 0, 0, -1, -1, "Atomic", 100, 200, (17 << 8) | 0x90, 1],
            [0, 0, 0, 4, -1, "Atomic", 210, 220, 0x53, 4],
        ]
        keys = (
            "records",
            "atomic_records",
            "clock_baseline_records",
            "atomic_calls",
            "batched_poll_calls",
            "poll_batch_records",
            "dropped_records",
        )
        for key in keys:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                capture = _v3_capture(rows)
                capture["metadata"]["fdwic_summary"][key] += 1
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, rf"fdwic_summary\.{key}"):
                    convert(input_path, output_path)

    def test_v3_weighted_summary_closes_mixed_direct_and_batches(self) -> None:
        rows = [
            [0, 0, 0, -1, -1, "Atomic", 100, 200, (17 << 8) | 0x90, 1],
            [0, 0, 0, -1, -1, "Atomic", 201, 250, (9 << 8) | 0x90, 14],
            [0, 0, 0, 4, -1, "Atomic", 251, 260, 0x53, 4],
        ]
        capture = _v3_capture(rows)
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            emitted, _, _ = convert(input_path, output_path)
            merged = json.loads(output_path.read_text(encoding="utf-8"))

        summary = merged["metadata"]["fdwic_summary"]
        self.assertEqual(summary["records"], 5)
        self.assertEqual(summary["atomic_records"], 3)
        self.assertEqual(summary["clock_baseline_records"], 2)
        self.assertEqual(summary["atomic_calls"], 27)
        self.assertEqual(summary["batched_poll_calls"], 26)
        self.assertEqual(summary["poll_batch_records"], 2)
        self.assertEqual(summary["dropped_records"], 0)
        self.assertEqual(emitted, 5)
        atomic_events = [
            event
            for event in merged["traceEvents"]
            if str(event.get("cat", "")).startswith("atomic.")
        ]
        self.assertEqual(len(atomic_events), summary["atomic_records"])

    def test_v3_requires_level4_and_producer_summary(self) -> None:
        capture = _v3_capture(
            [[0, 0, 0, -1, -1, "Atomic", 100, 110, (3 << 8) | 0x90, 14]]
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"

            capture["l2_swimlane_level"] = 1
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires l2_swimlane_level=4"):
                convert(input_path, output_path)

            capture["l2_swimlane_level"] = 4
            del capture["metadata"]["fdwic_summary"]
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fdwic_summary is required"):
                convert(input_path, output_path)

    def test_v3_rejects_non_standalone_topology(self) -> None:
        cases = {
            "core_type": lambda capture: capture["metadata"]["core_types"].__setitem__(
                0, "aiv"
            ),
            "block": lambda capture: capture["fdwic_events"][0].__setitem__(1, 1),
            "lane": lambda capture: capture["fdwic_events"][0].__setitem__(2, 1),
        }
        for name, mutate in cases.items():
            with self.subTest(field=name), tempfile.TemporaryDirectory() as directory:
                capture = _v3_capture([])
                mutate(capture)
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "does not match standalone topology"):
                    convert(input_path, output_path)
                self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
