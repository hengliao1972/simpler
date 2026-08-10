#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportIndexIssue=false
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

# 测试数据故意使用松散 JSON object 并与 converter 的 raw 换行对齐；
# 保留当前审计格式，避免单个站点回归引起整文件机械重排。
# fmt: off

try:
    # `python -m unittest tests.atomic_probe...` 以 namespace package 导入。
    from .swimlane_converter import (
        _derive_v4_task_kinds,
        _iter_v5_cross_core_semantic_gap_spans,
        _restore_v5_shared_efdrain,
        convert,
    )
except ImportError:
    # 也保留从本目录直接执行脚本的用法。
    from swimlane_converter import (
        _derive_v4_task_kinds,
        _iter_v5_cross_core_semantic_gap_spans,
        _restore_v5_shared_efdrain,
        convert,
    )


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


def _v5_capture(
    rows: list[list[object]],
    *,
    num_cores: int = 1,
    add_parents: bool = True,
    tensormap_mode: str = "private",
    metadata_writer_tasks: list[int] | None = None,
    metadata_prefix_tasks: list[int] | None = None,
) -> dict[str, object]:
    """构造 phase-only schema-v5 raw；调用者显式提供 Claim/Submit/尾动作。"""
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
    metadata: dict[str, object] = {
        "clock_freq_hz": 1_000_000_000,
        "num_cores": num_cores,
        "trace_schema_version": 5,
        "tensormap_mode": tensormap_mode,
        "core_types": [
            _standalone_topology(core_id)[2]
            for core_id in range(num_cores)
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
    }
    if tensormap_mode == "shared":
        if metadata_writer_tasks is None:
            # 大多数旧 fixture 描述“每个 winner 都发布 metadata”的
            # 旧协议；显式稀疏链 fixture 会传入准确 writer task 列表。
            metadata_writer_tasks = sorted(
                {
                    int(row[3])
                    for row in all_rows
                    if row[5] == "Submit" and int(row[8]) & 0x1
                }
            )
        metadata["shared_metadata_writer_tasks"] = list(
            metadata_writer_tasks
        )
        if metadata_prefix_tasks is None:
            metadata_prefix_tasks = sorted(
                {
                    int(row[3])
                    for row in all_rows
                    if row[5] == "Submit" and int(row[8]) & 0x1
                }
            )
        metadata["shared_metadata_prefix_tasks"] = list(
            metadata_prefix_tasks
        )
    return {
        "l2_swimlane_level": 1,
        "metadata": metadata,
        "fdwic_events": all_rows,
    }


def _refresh_summary(capture: dict[str, object]) -> None:
    """按当前 raw 行重新生成 producer weighted summary。"""

    rows = capture["fdwic_events"]
    metadata = capture["metadata"]
    assert isinstance(rows, list)
    assert isinstance(metadata, dict)
    atomic_rows = [row for row in rows if row[5] == "Atomic"]
    batch_rows = [row for row in atomic_rows if int(row[8]) & 0x80]
    batch_calls = sum((int(row[8]) >> 8) & 0xFFFFFF for row in batch_rows)
    dcci_rows = [row for row in rows if row[5] == "Dcci"]
    summary = {
        "records": len(rows),
        "atomic_records": len(atomic_rows),
        "clock_baseline_records": sum(
            row[5] == "ClockBaseline" for row in rows
        ),
        "atomic_calls": len(atomic_rows) - len(batch_rows) + batch_calls,
        "batched_poll_calls": batch_calls,
        "poll_batch_records": len(batch_rows),
        "dropped_records": 0,
    }
    if dcci_rows:
        summary.update(
            {
                "dcci_records": len(dcci_rows),
                "dcci_calls": sum(
                    (int(row[8]) >> 3) & 0xF for row in dcci_rows
                ),
                "dcci_lines": sum(
                    (int(row[8]) >> 8) & 0xFFFFFF
                    for row in dcci_rows
                ),
            }
        )
    metadata["fdwic_summary"] = summary


def _v5_shared_register_atomic_capture(
    *,
    dependency_applied: bool = True,
    metadata_prefix_tasks: list[int] | None = None,
) -> dict[str, object]:
    """构造五个 winner、两个真实 metadata writer 的稀疏链 v5 raw。

    task 0 是首个 writer，因此没有前驱；task 1..4 各等待 task 0，
    但只有 task 0/4 发布自己的 completion。
    """

    if metadata_prefix_tasks is None:
        metadata_prefix_tasks = [0, 1, 2, 3, 4]
    rows: list[list[object]] = []
    for task_id in range(5):
        base = 100 + task_id * 50
        is_alloc = task_id == 0
        function_id = -1 if is_alloc else task_id - 1
        rows.extend(
            [
                [
                    0,
                    0,
                    0,
                    task_id,
                    -1,
                    "Claim",
                    base + 10,
                    base + 15,
                    0x3,
                    1 if is_alloc else 0,
                ],
                [
                    0,
                    0,
                    0,
                    task_id,
                    function_id,
                    "Materialize",
                    base + 15,
                    base + 20,
                    0,
                    1,
                ],
                [
                    0,
                    0,
                    0,
                    task_id,
                    function_id,
                    "Register",
                    base + 20,
                    base + 40,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    task_id,
                    function_id,
                    "SharedRegisterPublishMetadata",
                    base + 24,
                    base + 34,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    task_id,
                    function_id,
                    "SharedRegisterPublishTaskOutputs",
                    base + 29,
                    base + 32,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    task_id,
                    function_id,
                    "SharedRegisterPublishTaskOutputsCopy",
                    base + 29,
                    base + 30,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    task_id,
                    function_id,
                    "SharedRegisterPublishTaskOutputsFlush",
                    base + 30,
                    base + 32,
                    0,
                    0,
                ],
            ]
        )
        if is_alloc:
            rows.append(
                [
                    0,
                    0,
                    0,
                    task_id,
                    -1,
                    "AllocComplete",
                    base + 40,
                    base + 45,
                    0,
                    0,
                ]
            )
        else:
            rows.extend(
                [
                    [
                        0,
                        0,
                        0,
                        task_id,
                        function_id,
                        "Fanin",
                        base + 40,
                        base + 43,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        task_id,
                        function_id,
                        "WinnerBuild",
                        base + 43,
                        base + 45,
                        0,
                        0,
                    ],
                ]
            )
        rows.append(
            [
                0,
                0,
                0,
                task_id,
                -1,
                "Submit",
                base,
                base + 50,
                1,
                1 if is_alloc else 0,
            ]
        )

    capture = _v5_capture(
        rows,
        tensormap_mode="shared",
        metadata_writer_tasks=[0, 4],
        metadata_prefix_tasks=metadata_prefix_tasks,
    )
    metadata = capture["metadata"]
    assert isinstance(metadata, dict)
    metadata["submit_topology"] = "central_ticket"
    capture["l2_swimlane_level"] = 4
    capture_rows = capture["fdwic_events"]
    assert isinstance(capture_rows, list)
    dependency_flags = 0x3 if dependency_applied else 0x1
    capture_rows.extend(
        [
            [0, 0, 0, -1, -1, "ClockBaseline", 10, 11, 0, 0],
            [
                0,
                0,
                0,
                -1,
                -1,
                "ClockBaseline",
                12,
                13,
                dependency_flags,
                0,
            ],
        ]
    )
    for task_id in range(5):
        base = 100 + task_id * 50
        if task_id > 0 and task_id in metadata_prefix_tasks:
            # 三次 Load（两次 Pending + 最后一次 Ready）聚成一条等待
            # episode；task 0 没有前驱，不产生此记录。
            capture_rows.append(
                [
                    0,
                    0,
                    0,
                    -1,
                    -1,
                    "Atomic",
                    base + 20,
                    base + 24,
                    (3 << 8) | 0xD0,
                    19,
                ]
            )
        if task_id in (0, 4):
            # 旧 central-ticket fixture 仍只为真实 writer 构造完成记录，
            # 但 site20 的稳定原语已统一成消费返回值的 CAS。
            handoff_flags = 0x14 | (0x40 if dependency_applied else 0)
            capture_rows.append(
                [
                    0,
                    0,
                    0,
                    task_id,
                    -1,
                    "Atomic",
                    base + 34,
                    base + 40,
                    handoff_flags,
                    20,
                ]
            )
    _refresh_summary(capture)
    return capture


class SwimlaneConverterLayoutTest(unittest.TestCase):
    def test_v5_shared_restores_efdrain_without_raw_record_growth(
        self,
    ) -> None:
        capture = _v5_shared_register_atomic_capture()
        rows = capture["fdwic_events"]
        metadata = capture["metadata"]
        assert isinstance(rows, list)
        assert isinstance(metadata, dict)
        raw_count = len(rows)
        self.assertFalse(any(row[5] == "EfDrain" for row in rows))
        self.assertEqual(metadata["fdwic_summary"]["records"], raw_count)
        restored_rows = [tuple(row) for row in rows]
        _restore_v5_shared_efdrain(restored_rows, 5, "shared")
        restored_efdrains = [
            row for row in restored_rows if row[5] == "EfDrain"
        ]
        self.assertEqual(len(restored_efdrains), 5)
        # 五个 task 都是 winner；EfDrain 属于 scalar Submit 前端，不把
        # QK/SF/PV/UP 的 function_id 错挂到派生事件上。
        self.assertTrue(all(row[4] == -1 for row in restored_efdrains))

        def converted_efdrains(
            directory: str,
            source: dict[str, object],
        ) -> list[tuple[str, float, float]]:
            input_path = Path(directory) / "derived.raw.json"
            output_path = Path(directory) / "derived.merged.json"
            input_path.write_text(json.dumps(source), encoding="utf-8")
            convert(input_path, output_path)
            merged = json.loads(output_path.read_text(encoding="utf-8"))
            return sorted(
                (
                    str(event["name"]),
                    float(event["ts"]),
                    float(event["dur"]),
                )
                for event in merged["traceEvents"]
                if str(event.get("name", "")).startswith("efdrain#")
            )

        with tempfile.TemporaryDirectory() as directory:
            derived = converted_efdrains(directory, capture)

        self.assertEqual(len(derived), 5)
        self.assertEqual(
            [name for name, _start, _duration in derived],
            [f"efdrain#{task_id}" for task_id in range(5)],
        )
        self.assertTrue(
            all(duration == 0.01 for _name, _start, duration in derived)
        )

    def test_v5_shared_efdrain_derivation_rejects_invalid_evidence(
        self,
    ) -> None:
        for label, expected_error in (
            ("missing_claim", "Claim keys do not match Submit keys"),
            ("inverted_boundary", "Claim is outside or inverted"),
            ("explicit_efdrain", "must not contain explicit EfDrain"),
        ):
            with self.subTest(label=label):
                capture = _v5_shared_register_atomic_capture()
                rows = capture["fdwic_events"]
                assert isinstance(rows, list)
                if label == "missing_claim":
                    rows[:] = [
                        row
                        for row in rows
                        if not (
                            row[0] == 0
                            and row[3] == 0
                            and row[5] == "Claim"
                        )
                    ]
                elif label == "inverted_boundary":
                    claim = next(
                        row
                        for row in rows
                        if row[0] == 0
                        and row[3] == 0
                        and row[5] == "Claim"
                    )
                    claim[6:8] = [95, 99]
                else:
                    rows.append(
                        [0, 0, 0, 0, -1, "EfDrain", 100, 110, 0, 0]
                    )
                _refresh_summary(capture)

                with tempfile.TemporaryDirectory() as directory:
                    input_path = Path(directory) / "raw.json"
                    output_path = Path(directory) / "merged.json"
                    input_path.write_text(
                        json.dumps(capture), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(
                        ValueError, expected_error
                    ):
                        convert(input_path, output_path)

    def test_v5_private_accepts_startup_and_terminal_dcci_rows(self) -> None:
        capture = _v5_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 120, 0x3, 1],
                [0, 0, 0, 0, -1, "AllocComplete", 120, 130, 0, 0],
                [0, 0, 0, 0, -1, "Submit", 100, 140, 1, 1],
            ]
        )
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        # startup: invalidate + DSB, one call across three lines。
        rows.append([0, 0, 0, -1, -1, "Dcci", 92, 96, 0x30C, 9])
        # terminal observer: clean + DSB, two calls across two lines。
        rows.append([0, 0, 0, -1, -1, "Dcci", 220, 224, 0x215, 8])
        _refresh_summary(capture)

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            merged = json.loads(output_path.read_text(encoding="utf-8"))

        names = {
            event.get("name") for event in merged["traceEvents"]
        }
        self.assertIn(
            "dcci.startup_config_invalidate.invalidate×1.lines3#-1",
            names,
        )
        self.assertIn(
            "dcci.observer_trace_export.clean_out×2.lines2#-1",
            names,
        )
        summary = merged["metadata"]["fdwic_summary"]
        self.assertEqual(summary["dcci_records"], 2)
        self.assertEqual(summary["dcci_calls"], 3)
        self.assertEqual(summary["dcci_lines"], 5)

    def test_v5_shared_accepts_three_region_terminal_dcci_row(self) -> None:
        capture = _v5_shared_register_atomic_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        # shared observer 依次 clean 专用 Submit/Claim 区、通用记录区和
        # core state，因此一条聚合 row 表示三次区域原语。
        rows.extend(
            [
                [0, 0, 0, -1, -1, "Dcci", 92, 96, 0x30C, 9],
                [0, 0, 0, -1, -1, "Dcci", 420, 424, 0x31D, 8],
            ]
        )
        _refresh_summary(capture)

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            merged = json.loads(output_path.read_text(encoding="utf-8"))

        names = {
            event.get("name") for event in merged["traceEvents"]
        }
        self.assertIn(
            "dcci.observer_trace_export.clean_out×3.lines3#-1",
            names,
        )
        summary = merged["metadata"]["fdwic_summary"]
        self.assertEqual(summary["dcci_records"], 2)
        self.assertEqual(summary["dcci_calls"], 4)
        self.assertEqual(summary["dcci_lines"], 6)

    def test_v5_rejects_invalid_startup_dcci_identity(self) -> None:
        capture = _v5_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 120, 0x3, 1],
                [0, 0, 0, 0, -1, "AllocComplete", 120, 130, 0, 0],
                [0, 0, 0, 0, -1, "Submit", 100, 140, 1, 1],
            ]
        )
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        rows.extend(
            [
                [0, 0, 0, 0, -1, "Dcci", 92, 96, 0x30C, 9],
                [0, 0, 0, -1, -1, "Dcci", 220, 224, 0x215, 8],
            ]
        )
        _refresh_summary(capture)
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "invalid startup Dcci fields"
            ):
                convert(input_path, output_path)

    def test_v4_requires_explicit_tensormap_mode(self) -> None:
        capture = _v5_capture(
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
        capture = _v5_capture(
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
            [
                0,
                0,
                0,
                0,
                -1,
                "SharedRegisterPublishTaskOutputs",
                129,
                132,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                -1,
                "SharedRegisterPublishTaskOutputsCopy",
                129,
                130,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                -1,
                "SharedRegisterPublishTaskOutputsFlush",
                130,
                132,
                0,
                0,
            ],
            [0, 0, 0, 0, -1, "AllocComplete", 140, 145, 0, 0],
            [0, 0, 0, 0, -1, "Submit", 100, 150, 1, 1],
        ]
        capture = _v5_capture(rows, tensormap_mode="shared")
        raw_rows = capture["fdwic_events"]
        assert isinstance(raw_rows, list)
        # 两条全核 parent 由 helper 加入；Register 使用 metadata 父 detail
        # 加 task-outputs 及 copy/flush 两层子 detail。
        self.assertEqual(len(raw_rows), len(rows) + 2)
        self.assertEqual(
            sum(row[5] == "SharedRegisterPublishMetadata" for row in raw_rows),
            1,
        )
        self.assertEqual(
            sum(
                row[5] == "SharedRegisterPublishTaskOutputs"
                for row in raw_rows
            ),
            1,
        )
        self.assertEqual(
            sum(
                row[5] == "SharedRegisterPublishTaskOutputsCopy"
                for row in raw_rows
            ),
            1,
        )
        self.assertEqual(
            sum(
                row[5] == "SharedRegisterPublishTaskOutputsFlush"
                for row in raw_rows
            ),
            1,
        )

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            events = json.loads(output_path.read_text(encoding="utf-8"))["traceEvents"]

        parent = next(event for event in events if event.get("name") == "register#0")
        flat_child_names = (
            "register.enter_tensormap_insert_chain#0",
            "register.publish_writer_metadata"
            "[ordinary_tensormap_entries=0]#0",
            "register.publish_task_outputs#0",
            "register.publish_metadata_epilogue#0",
            "register.publish_tensormap_insert_completion#0",
        )
        nested_output_names = (
            "register.publish_task_outputs.copy#0",
            "register.publish_task_outputs.flush#0",
        )
        metadata_name = "register.publish_metadata#0"
        children = {
            event["name"]: event
            for event in events
            if event.get("name")
            in (*flat_child_names, metadata_name, *nested_output_names)
        }
        self.assertEqual(
            set(children),
            {*flat_child_names, metadata_name, *nested_output_names},
        )
        self.assertAlmostEqual(
            children["register.publish_task_outputs.copy#0"]["ts"],
            children["register.publish_task_outputs#0"]["ts"],
        )
        self.assertAlmostEqual(
            children["register.publish_task_outputs.copy#0"]["ts"]
            + children["register.publish_task_outputs.copy#0"]["dur"],
            children["register.publish_task_outputs.flush#0"]["ts"],
        )
        self.assertAlmostEqual(
            children["register.publish_task_outputs.flush#0"]["ts"]
            + children["register.publish_task_outputs.flush#0"]["dur"],
            children["register.publish_task_outputs#0"]["ts"]
            + children["register.publish_task_outputs#0"]["dur"],
        )
        for child in children.values():
            self.assertEqual(
                set(child), {"ph", "name", "pid", "tid", "ts", "dur"}
            )
        self.assertEqual(
            children["register.enter_tensormap_insert_chain#0"]["ts"],
            parent["ts"],
        )
        self.assertAlmostEqual(
            children["register.enter_tensormap_insert_chain#0"]["ts"]
            + children[
                "register.enter_tensormap_insert_chain#0"
            ]["dur"],
            children[
                "register.publish_writer_metadata"
                "[ordinary_tensormap_entries=0]#0"
            ]["ts"],
        )
        self.assertAlmostEqual(
            children[
                "register.publish_writer_metadata"
                "[ordinary_tensormap_entries=0]#0"
            ]["ts"]
            + children[
                "register.publish_writer_metadata"
                "[ordinary_tensormap_entries=0]#0"
            ]["dur"],
            children["register.publish_task_outputs#0"]["ts"],
        )
        self.assertAlmostEqual(
            children["register.publish_task_outputs#0"]["ts"]
            + children["register.publish_task_outputs#0"]["dur"],
            children["register.publish_metadata_epilogue#0"]["ts"],
        )
        self.assertAlmostEqual(
            children["register.publish_metadata_epilogue#0"]["ts"]
            + children["register.publish_metadata_epilogue#0"]["dur"],
            children["register.publish_tensormap_insert_completion#0"]["ts"],
        )
        self.assertAlmostEqual(
            children["register.publish_tensormap_insert_completion#0"]["ts"]
            + children[
                "register.publish_tensormap_insert_completion#0"
            ]["dur"],
            parent["ts"] + parent["dur"],
        )
        self.assertAlmostEqual(
            sum(children[name]["dur"] for name in flat_child_names),
            parent["dur"],
        )
        self.assertAlmostEqual(
            sum(
                children[name]["dur"]
                for name in (
                    "register.publish_writer_metadata"
                    "[ordinary_tensormap_entries=0]#0",
                    "register.publish_task_outputs#0",
                    "register.publish_metadata_epilogue#0",
                )
            ),
            children[metadata_name]["dur"],
        )

    def test_v5_materialize_output_detail_leaves_register_serial_only(
        self,
    ) -> None:
        rows = [
            [0, 0, 0, 0, -1, "Claim", 110, 115, 0x3, 1],
            [0, 0, 0, 0, -1, "Materialize", 115, 125, 0, 1],
            [
                0,
                0,
                0,
                0,
                -1,
                "SharedMaterializePublishTaskOutputs",
                120,
                124,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                -1,
                "SharedMaterializePublishTaskOutputsCopy",
                120,
                121,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                -1,
                "SharedMaterializePublishTaskOutputsFlush",
                121,
                123,
                0,
                0,
            ],
            # descriptor flush 与业务 flush span 使用完全相同的端点；
            # merged 必须先输出业务父区间，再输出 DCCI overlay。
            [0, 0, 0, 0, -1, "Dcci", 121, 123, 0x10D, 3],
            [0, 0, 0, 0, -1, "Register", 125, 140, 0, 0],
            [
                0,
                0,
                0,
                0,
                -1,
                "SharedRegisterPublishMetadata",
                129,
                135,
                0,
                0,
            ],
            # SharedOutputRef 的 writer commit 会消费 CAS 返回值，因此泳道
            # 必须明确显示为 return_ready；相邻 DCCI 仍归属同一个 Register
            # writer metadata 子区间。
            [0, 0, 0, 0, -1, "Atomic", 130, 131, 0x54, 27],
            [0, 0, 0, 0, -1, "Dcci", 131, 132, 0x10D, 1],
            [0, 0, 0, 0, -1, "Atomic", 136, 137, 0x54, 20],
            # all-worker-replay 已由逐 task completion 链证明 producer
            # 发布完成；这里是一条协议校验 Load，而不是等待 PollBatch。
            [0, 0, 0, 0, -1, "Atomic", 137, 138, 0x50, 23],
            [0, 0, 0, 0, -1, "AllocComplete", 140, 145, 0, 0],
            [0, 0, 0, 0, -1, "Submit", 100, 150, 1, 1],
            [0, 0, 0, -1, -1, "ClockBaseline", 10, 11, 0, 0],
            [0, 0, 0, -1, -1, "ClockBaseline", 12, 13, 0x3, 0],
            [0, 0, 0, -1, -1, "Dcci", 220, 224, 0x31D, 8],
        ]
        capture = _v5_capture(rows, tensormap_mode="shared")
        capture["l2_swimlane_level"] = 4
        _refresh_summary(capture)
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            events = json.loads(
                output_path.read_text(encoding="utf-8")
            )["traceEvents"]

        names = {event.get("name") for event in events}
        positions = {
            event.get("name"): index
            for index, event in enumerate(events)
        }
        self.assertIn(
            "materialize.publish_shared_output_descriptors#0", names
        )
        self.assertIn(
            "materialize.publish_shared_output_descriptors"
            ".copy_tensor_descs[GM]#0",
            names,
        )
        self.assertIn(
            "materialize.publish_shared_output_descriptors"
            ".flush_tensor_descs[DCCI]#0",
            names,
        )
        self.assertIn(
            "register.enter_tensormap_insert_chain#0", names
        )
        self.assertIn(
            "register.publish_writer_metadata"
            "[ordinary_tensormap_entries=0]#0",
            names,
        )
        self.assertIn(
            "register.publish_tensormap_insert_completion#0", names
        )
        self.assertIn(
            "atomic.return_ready.shared_output_ref_last_writer_commit"
            ".compare_exchange#0",
            names,
        )
        self.assertIn(
            "atomic.return_ready."
            "shared_output_ref_fanin_output_published_load.load#0",
            names,
        )
        self.assertIn(
            "dcci.shared_output_ref_writer_history_flush"
            ".clean_out×1.lines1#0",
            names,
        )
        self.assertIn(
            "dcci.shared_output_descriptor_flush"
            ".clean_out×1.lines1#0",
            names,
        )
        self.assertLess(
            positions["materialize#0"],
            positions[
                "materialize.publish_shared_output_descriptors#0"
            ],
        )
        self.assertLess(
            positions[
                "materialize.publish_shared_output_descriptors"
                ".flush_tensor_descs[DCCI]#0"
            ],
            positions[
                "dcci.shared_output_descriptor_flush"
                ".clean_out×1.lines1#0"
            ],
        )
        self.assertLess(
            positions["register#0"],
            positions[
                "register.publish_writer_metadata"
                "[ordinary_tensormap_entries=0]#0"
            ],
        )
        self.assertLess(
            positions[
                "register.publish_writer_metadata"
                "[ordinary_tensormap_entries=0]#0"
            ],
            positions[
                "dcci.shared_output_ref_writer_history_flush"
                ".clean_out×1.lines1#0"
            ],
        )
        self.assertNotIn("register.publish_metadata#0", names)
        self.assertNotIn("register.publish_task_outputs#0", names)
        self.assertNotIn("register.publish_metadata_epilogue#0", names)

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
                capture = _v5_capture(rows, tensormap_mode="shared")
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError, "requires exactly one SharedRegisterPublishMetadata"
                ):
                    convert(input_path, output_path)

    def test_v5_task_outputs_detail_is_strictly_nested_once(self) -> None:
        for label in ("missing", "duplicate", "outside_metadata", "wrong_identity"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                capture = _v5_shared_register_atomic_capture()
                rows = capture["fdwic_events"]
                assert isinstance(rows, list)
                output_detail = next(
                    row
                    for row in rows
                    if row[0] == 0
                    and row[3] == 0
                    and row[5] == "SharedRegisterPublishTaskOutputs"
                )
                if label == "missing":
                    rows.remove(output_detail)
                elif label == "duplicate":
                    rows.append(list(output_detail))
                elif label == "outside_metadata":
                    output_detail[6] = 123
                else:
                    output_detail[4] = 0
                _refresh_summary(capture)
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                expected = {
                    "missing": "requires exactly one SharedRegisterPublishTaskOutputs",
                    "duplicate": "requires exactly one SharedRegisterPublishTaskOutputs",
                    "outside_metadata": "outside SharedRegisterPublishMetadata",
                    "wrong_identity": "identity differs",
                }[label]
                with self.assertRaisesRegex(ValueError, expected):
                    convert(input_path, output_path)

    def test_v5_rejects_old_schema_v4_raw(self) -> None:
        capture = _v5_capture(
            [
                [0, 0, 0, 0, -1, "Claim", 110, 120, 0x3, 1],
                [0, 0, 0, 0, -1, "AllocComplete", 120, 130, 0, 0],
                [0, 0, 0, 0, -1, "Submit", 100, 140, 1, 1],
            ]
        )
        metadata = capture["metadata"]
        assert isinstance(metadata, dict)
        metadata["trace_schema_version"] = 4
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "unsupported metadata.trace_schema_version: 4"
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
                output_detail = [
                    detail[0],
                    detail[1],
                    detail[2],
                    detail[3],
                    detail[4],
                    "SharedRegisterPublishTaskOutputs",
                    129,
                    132,
                    0,
                    0,
                ]
                capture = _v5_capture(
                    [*base_rows, detail, output_detail],
                    tensormap_mode="shared",
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
        capture = _v5_capture(
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
                [
                    0,
                    0,
                    0,
                    0,
                    -1,
                    "SharedRegisterPublishTaskOutputs",
                    129,
                    132,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    -1,
                    "SharedRegisterPublishTaskOutputs",
                    129,
                    132,
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
        capture = _v5_capture(
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
        capture = _v5_capture(
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
        capture = _v5_capture(
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
                [
                    0,
                    0,
                    0,
                    0,
                    -1,
                    "SharedRegisterPublishTaskOutputs",
                    129,
                    132,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    -1,
                    "SharedRegisterPublishTaskOutputsCopy",
                    129,
                    130,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    0,
                    0,
                    -1,
                    "SharedRegisterPublishTaskOutputsFlush",
                    130,
                    132,
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
        capture = _v5_capture(
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

        self.assertEqual(merged["metadata"]["trace_schema_version"], 5)
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
                {"ph": "X", "name": "submit_residual", "pid": 0, "tid": 1, "ts": 0.01, "dur": 0.01},
            ],
        )
        self.assertEqual(
            tails,
            [
                {
                    "ph": "X",
                    "name": "submit_tail_gap",
                    "pid": 0,
                    "tid": 1,
                    "ts": 0.04,
                    "dur": 0.01,
                }
            ],
        )
        for event in (*residuals, *tails):
            self.assertEqual(set(event), {"ph", "name", "pid", "tid", "ts", "dur"})

    def test_v4_marks_between_submit_gap_without_loser_marker(self) -> None:
        capture = _v5_capture(
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
                    "tid": 1,
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
        capture = _v5_capture(
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
        capture = _v5_capture(
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
        capture = _v5_capture(
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
        capture = _v5_capture(
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

    def test_v5_central_ticket_derives_global_unique_task_plan(self) -> None:
        semantics = {
            (0, 0): (True, True),
            (7, 1): (True, False),
            (3, 2): (True, False),
            (95, 3): (True, False),
            (32, 4): (True, False),
        }
        self.assertEqual(
            _derive_v4_task_kinds(
                semantics, 96, "central_ticket"
            ),
            {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        )

    def test_v5_central_ticket_rejects_duplicate_task_owner(self) -> None:
        semantics = {
            (0, 0): (True, True),
            (1, 0): (True, True),
            (2, 1): (True, False),
            (3, 2): (True, False),
            (4, 3): (True, False),
            (5, 4): (True, False),
        }
        with self.assertRaisesRegex(ValueError, "multiple Submit owners"):
            _derive_v4_task_kinds(
                semantics, 96, "central_ticket"
            )

    def test_v4_requires_both_parent_spans(self) -> None:
        capture = _v5_capture(
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
            with self.assertRaisesRegex(ValueError, "requires exactly one schema-v5"):
                convert(input_path, output_path)

    def test_v4_rejects_removed_loser_replay_phase(self) -> None:
        capture = _v5_capture(
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
        # pid=4 的 mixed block 放一条 AIC 和一条 AIV0；这会复现旧版
        # AIC tid=0 被 Perfetto 映射到主线程 tid=pid=4、再与 AIV0 kernel
        # tid=4 碰撞的问题。Atomic 是 AIC scalar
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
                [0, 4, 0, 7, -1, "Claim", 100, 200, 0x2, 0],
                # flags: FetchMax(3) | result-used(bit4) | return-ready(bit6)。
                [0, 4, 0, 7, -1, "Atomic", 120, 160, 0x53, 4],
                # Exchange(1) 的旧值未消费，只能标 source-issue。
                [0, 4, 0, 7, -1, "Atomic", 161, 162, 0x01, 7],
                # flags: dependency-hook(bit0) | dependency-applied(bit1)。
                [1, 4, 1, -1, -1, "ClockBaseline", 101, 102, 0x3, 0],
                [1, 4, 1, 7, 0, "Kernel", 140, 180, 0, 0],
                # 同一 AIV0 上的 role-filtered Claim，没有 atomic。
                [1, 4, 1, 8, -1, "Claim", 201, 220, 0x0, 0],
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
        self.assertEqual(thread_names[(4, 1)], "AIC (core0)")
        self.assertEqual(thread_names[(4, 2)], "AIV0 (core1)")
        self.assertEqual(thread_names[(4, 4)], "AIC·kernel (core0)")
        self.assertEqual(thread_names[(4, 5)], "AIV0·kernel (core1)")
        self.assertNotIn((4, 0), thread_names)
        self.assertFalse(any("·atomic" in name for name in thread_names.values()))

        ready_atomic = next(event for event in events if event.get("cat") == "atomic.return_ready")
        issue_atomic = next(event for event in events if event.get("cat") == "atomic.source_issue")
        clock = next(event for event in events if event.get("cat") == "scalar_clock")
        kernel = next(event for event in events if event.get("name") == "QK#7")
        self.assertEqual((ready_atomic["pid"], ready_atomic["tid"]), (4, 1))
        self.assertEqual((issue_atomic["pid"], issue_atomic["tid"]), (4, 1))
        self.assertEqual((clock["pid"], clock["tid"]), (4, 2))
        self.assertEqual((kernel["pid"], kernel["tid"]), (4, 5))
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

    def test_block_tracks_keep_scalar_then_kernel_order(self) -> None:
        # 每条物理 lane 只需出现一条 scalar 记录，converter 就应为该
        # block 建齐对应的 scalar/kernel 两条轨道，并通过显式 sort index
        # 固定为 AIC/AIV0/AIV1 scalar 在前、三条 kernel 在后。
        capture = {
            "l2_swimlane_level": 4,
            "metadata": {
                "clock_freq_hz": 1_000_000_000,
                "num_cores": 3,
                "trace_schema_version": 2,
                "core_types": ["AIC", "AIV", "AIV"],
            },
            "fdwic_events": [
                [0, 7, 0, 0, -1, "Claim", 100, 110, 0, 0],
                [1, 7, 1, 0, -1, "Claim", 100, 110, 0, 0],
                [2, 7, 2, 0, -1, "Claim", 100, 110, 0, 0],
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            merged = json.loads(output_path.read_text(encoding="utf-8"))

        events = merged["traceEvents"]
        thread_names = {
            (event["pid"], event["tid"]): event["args"]["name"]
            for event in events
            if event.get("ph") == "M" and event.get("name") == "thread_name"
        }
        thread_sort_indices = {
            (event["pid"], event["tid"]): event["args"]["sort_index"]
            for event in events
            if event.get("ph") == "M"
            and event.get("name") == "thread_sort_index"
        }
        expected_names = [
            "AIC (core0)",
            "AIV0 (core1)",
            "AIV1 (core2)",
            "AIC·kernel (core0)",
            "AIV0·kernel (core1)",
            "AIV1·kernel (core2)",
        ]
        ordered_tracks = sorted(
            thread_names,
            key=lambda track: thread_sort_indices[track],
        )
        self.assertEqual(ordered_tracks, [(7, tid) for tid in range(1, 7)])
        self.assertEqual(
            [thread_names[track] for track in ordered_tracks], expected_names
        )
        self.assertEqual(
            [thread_sort_indices[track] for track in ordered_tracks],
            list(range(1, 7)),
        )

    def test_v4_shared_register_atomics_are_named_on_scalar_lane(self) -> None:
        capture = _v5_shared_register_atomic_capture()
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            merged = json.loads(output_path.read_text(encoding="utf-8"))

        events = merged["traceEvents"]
        poll = next(
            event
            for event in events
            if event.get("name")
            == (
                "atomic.poll_batch.return_ready."
                "shared_insert_predecessor_poll.load×3"
            )
        )
        handoff = next(
            event
            for event in events
            if event.get("name")
            == (
                "atomic.return_ready.shared_insert_completion_publish."
                "compare_exchange#0"
            )
        )
        register = next(
            event for event in events if event.get("name") == "register#0"
        )
        self.assertEqual((poll["pid"], poll["tid"]), (0, 1))
        self.assertEqual((handoff["pid"], handoff["tid"]), (0, 1))
        self.assertEqual((register["pid"], register["tid"]), (0, 1))
        # schema-v5 为控制近 300 MiB 产物，只保留 Perfetto duration
        # 必需字段；poll_batch/return_ready/source_issue/site/op/call_count 已完整编码
        # 在可见名称中。
        self.assertEqual(
            set(poll), {"ph", "name", "pid", "tid", "ts", "dur"}
        )
        self.assertEqual(poll["ph"], "X")
        self.assertEqual(
            set(handoff), {"ph", "name", "pid", "tid", "ts", "dur"}
        )
        summary = merged["metadata"]["fdwic_summary"]
        self.assertEqual(summary["atomic_records"], 6)
        self.assertEqual(summary["atomic_calls"], 14)
        self.assertEqual(summary["batched_poll_calls"], 12)
        self.assertEqual(summary["poll_batch_records"], 4)
        thread_names = {
            event["args"]["name"]
            for event in events
            if event.get("ph") == "M"
            and event.get("name") == "thread_name"
        }
        self.assertFalse(any("·atomic" in name for name in thread_names))

    def test_v5_shared_output_publication_waits_are_aggregate_only(
        self,
    ) -> None:
        capture = _v5_shared_register_atomic_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        rows.extend(
            [
                # fanin 与 metadata writer 各自等待两个 Load；二者都只
                # 允许一条 return-ready PollBatch，不允许逐 Load raw。
                [0, 0, 0, -1, -1, "Atomic", 114, 116, 0x2D0, 23],
                [0, 0, 0, -1, -1, "Atomic", 116, 118, 0x2D0, 24],
            ]
        )
        _refresh_summary(capture)

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            names = {
                event["name"]
                for event in json.loads(
                    output_path.read_text(encoding="utf-8")
                )["traceEvents"]
            }

        self.assertIn(
            "atomic.poll_batch.return_ready."
            "shared_output_ref_fanin_output_published_load.load×2",
            names,
        )
        self.assertIn(
            "atomic.poll_batch.return_ready."
            "shared_output_ref_metadata_output_published_load.load×2",
            names,
        )

        for site_id in (23, 24):
            with self.subTest(site_id=site_id), tempfile.TemporaryDirectory() as directory:
                invalid = _v5_shared_register_atomic_capture()
                invalid_rows = invalid["fdwic_events"]
                assert isinstance(invalid_rows, list)
                invalid_rows.append(
                    [0, 0, 0, -1, -1, "Atomic", 114, 116, 0x50, site_id]
                )
                _refresh_summary(invalid)
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError, "aggregate-only Atomic site"
                ):
                    convert(input_path, output_path)

    def test_v5_shared_metadata_writer_plan_is_required_and_exact(
        self,
    ) -> None:
        cases = (
            ("missing", None, "must be an array"),
            ("duplicate", [0, 0, 4], "strictly increasing"),
            ("unknown", [0, 99], "contains unknown tasks"),
        )
        for label, writer_tasks, expected in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                capture = _v5_shared_register_atomic_capture()
                metadata = capture["metadata"]
                assert isinstance(metadata, dict)
                if writer_tasks is None:
                    metadata.pop("shared_metadata_writer_tasks")
                else:
                    metadata["shared_metadata_writer_tasks"] = writer_tasks
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    convert(input_path, output_path)

    def test_v5_shared_metadata_prefix_plan_is_required_and_exact(
        self,
    ) -> None:
        cases = (
            ("missing", None, "must be an array"),
            ("duplicate", [0, 1, 1, 4], "strictly increasing"),
            ("unknown", [0, 1, 2, 3, 99], "contains unknown tasks"),
            ("omits_writer", [0, 1, 2, 3], "writers must also require"),
        )
        for label, prefix_tasks, expected in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                capture = _v5_shared_register_atomic_capture()
                metadata = capture["metadata"]
                assert isinstance(metadata, dict)
                if prefix_tasks is None:
                    metadata.pop("shared_metadata_prefix_tasks")
                else:
                    metadata["shared_metadata_prefix_tasks"] = prefix_tasks
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    convert(input_path, output_path)

    def test_v5_shared_sparse_prefix_waits_only_actual_writer_tasks(
        self,
    ) -> None:
        capture = _v5_shared_register_atomic_capture(
            metadata_prefix_tasks=[0, 4]
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            merged = json.loads(output_path.read_text(encoding="utf-8"))
            atomic_polls = [
                event
                for event in merged["traceEvents"]
                if event.get("name") ==
                "atomic.poll_batch.return_ready."
                "shared_insert_predecessor_poll.load×3"
            ]
            self.assertEqual(len(atomic_polls), 1)

    def test_v5_shared_per_symbol_dag_accepts_device_derived_poll_set(
        self,
    ) -> None:
        capture = _v5_shared_register_atomic_capture()
        metadata = capture["metadata"]
        rows = capture["fdwic_events"]
        assert isinstance(metadata, dict)
        assert isinstance(rows, list)
        metadata["shared_metadata_ordering"] = "per_symbol_dag"
        metadata.pop("shared_metadata_prefix_tasks")
        # task 0/4 是同一 symbol 的 writer。模拟 device DAG 只让 task 4
        # 等 task 0；中间三个 task 不因全局 writer 前缀产生假等待。
        rows[:] = [
            row
            for row in rows
            if not (
                row[5] == "Atomic"
                and int(row[9]) == 19
                and int(row[6]) in (170, 220, 270)
            )
        ]
        _refresh_summary(capture)

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            merged = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            merged["metadata"]["shared_metadata_ordering"],
            "per_symbol_dag",
        )
        atomic_polls = [
            event
            for event in merged["traceEvents"]
            if str(event.get("name", "")).startswith(
                "atomic.poll_batch.return_ready."
                "shared_insert_predecessor_poll.load"
            )
        ]
        self.assertEqual(len(atomic_polls), 1)

    def test_v5_shared_per_symbol_dag_rejects_host_prefix_authority(
        self,
    ) -> None:
        capture = _v5_shared_register_atomic_capture()
        metadata = capture["metadata"]
        assert isinstance(metadata, dict)
        metadata["shared_metadata_ordering"] = "per_symbol_dag"
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "shared_metadata_prefix_tasks must be absent",
            ):
                convert(input_path, output_path)

    def test_v5_shared_per_symbol_dag_rejects_duplicate_poll(self) -> None:
        capture = _v5_shared_register_atomic_capture()
        metadata = capture["metadata"]
        rows = capture["fdwic_events"]
        assert isinstance(metadata, dict)
        assert isinstance(rows, list)
        metadata["shared_metadata_ordering"] = "per_symbol_dag"
        metadata.pop("shared_metadata_prefix_tasks")
        poll = next(
            row
            for row in rows
            if row[5] == "Atomic" and int(row[9]) == 19
        )
        rows.append(list(poll))
        _refresh_summary(capture)
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "per_symbol_dag allows at most one",
            ):
                convert(input_path, output_path)

    def test_v5_shared_claim_tournament_atomics_are_named(self) -> None:
        capture = _v5_shared_register_atomic_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        # 两级 CAS 都消费返回值来判断 local/root owner，因此必须显示为
        # return_ready；task_id 后缀让它们能和同一个 Claim 直接对应。
        rows.extend(
            [
                [0, 0, 0, 0, -1, "Atomic", 111, 112, 0x54, 40],
                [0, 0, 0, 0, -1, "Atomic", 112, 113, 0x54, 41],
            ]
        )
        _refresh_summary(capture)

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            merged = json.loads(output_path.read_text(encoding="utf-8"))

        atomic_names = {
            event["name"]
            for event in merged["traceEvents"]
            if str(event.get("name", "")).startswith("atomic.return_ready.")
        }
        self.assertIn(
            "atomic.return_ready.shared_claim_tournament_local."
            "compare_exchange#0",
            atomic_names,
        )
        self.assertIn(
            "atomic.return_ready.shared_claim_tournament_root."
            "compare_exchange#0",
            atomic_names,
        )

    def test_v5_cross_core_exec_atomic_and_dcci_sites_are_named(self) -> None:
        # 复用一份结构闭合的 shared Submit，再把真实 cross-core 执行包
        # 的 Build/EfDrain 原语压入 task 1 的时间窗；EfDrain 父区间仍按
        # 正式格式由 Submit.start->Claim.start 离线恢复。这里只验证 raw ABI、
        # 边界语义和可见名称，不用 CPU 时间模拟 A5 延迟。
        capture = _v5_shared_register_atomic_capture()
        capture_rows = capture["fdwic_events"]
        assert isinstance(capture_rows, list)
        capture_rows.extend(
            [
                [0, 0, 0, 1, -1, "Atomic", 152, 153, 0x50, 43],
                [0, 0, 0, 1, -1, "Atomic", 153, 154, 0x50, 45],
                [0, 0, 0, 1, -1, "Atomic", 154, 155, 0x54, 48],
                [0, 0, 0, 1, -1, "Dcci", 155, 156, 0xA0C, 12],
                [0, 0, 0, 1, -1, "Atomic", 156, 157, 0x01, 49],
                [0, 0, 0, 1, -1, "Atomic", 157, 158, 0x51, 50],
                [0, 0, 0, 1, -1, "Atomic", 158, 159, 0x54, 51],
                [0, 0, 0, -1, -1, "Atomic", 385, 386, 0x50, 55],
                [0, 0, 0, -1, -1, "Atomic", 386, 387, 0x01, 53],
                [0, 0, 0, 1, -1, "Atomic", 193, 194, 0x54, 46],
                [0, 0, 0, 1, -1, "Dcci", 193, 194, 0xA0D, 11],
                [0, 0, 0, 1, -1, "Atomic", 194, 195, 0x54, 47],
                [0, 0, 0, -1, -1, "Dcci", 390, 391, 0x31D, 8],
            ]
        )
        _refresh_summary(capture)

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            merged = json.loads(output_path.read_text(encoding="utf-8"))

        names = {
            event["name"]
            for event in merged["traceEvents"]
            if event.get("ph") == "X"
        }
        self.assertIn(
            "atomic.return_ready.shared_exec_fatal_load.load#1", names
        )
        self.assertIn(
            "atomic.return_ready.shared_exec_cell_state_load.load#1",
            names,
        )
        self.assertIn(
            "atomic.return_ready.shared_exec_claim.compare_exchange#1",
            names,
        )
        self.assertIn(
            "atomic.source_issue.shared_exec_completion_vend_publish."
            "exchange#1",
            names,
        )
        self.assertIn(
            "atomic.return_ready.shared_exec_completion_flag_publish."
            "exchange#1",
            names,
        )
        self.assertIn(
            "atomic.return_ready.shared_exec_done_publish."
            "compare_exchange#1",
            names,
        )
        self.assertTrue(
            any(
                name.startswith(
                    "atomic.return_ready."
                    "shared_exec_drain_arrival_poll.load#"
                )
                for name in names
            )
        )
        self.assertIn(
            "atomic.source_issue.shared_exec_drain_release_publish."
            "exchange#-1",
            names,
        )
        self.assertIn(
            "atomic.return_ready.shared_exec_build_reserve."
            "compare_exchange#1",
            names,
        )
        self.assertIn(
            "atomic.return_ready.shared_exec_built_publish."
            "compare_exchange#1",
            names,
        )
        self.assertIn(
            "dcci.shared_exec_payload_invalidate.invalidate×1.lines10#1",
            names,
        )
        self.assertIn(
            "dcci.shared_exec_payload_flush.clean_out×1.lines10#1",
            names,
        )

    def test_v5_cross_core_winner_build_pack_span_uses_existing_edges(
        self,
    ) -> None:
        capture = _v5_shared_register_atomic_capture()
        rows = capture["fdwic_events"]
        metadata = capture["metadata"]
        assert isinstance(rows, list)
        assert isinstance(metadata, dict)
        metadata["submit_topology"] = "central_ticket"

        # 扩大最后一个 task 的父区间，放入正常成功 Build 的 reserve CAS
        # 与 payload-flush DCCI。两者之间是唯一可从现有记录严格推出的
        # Preload/Pack 边界；转换器不得为此增加 raw 行。
        for row in rows:
            if row[0] == 0 and row[3] == 4 and row[5] == "WinnerBuild":
                row[6:8] = [343, 380]
            elif row[0] == 0 and row[3] == 4 and row[5] == "Materialize":
                # Claim.end=315；把 Materialize.start 后移两个 tick，验证
                # 同一条 residual 会按中央 ticket 语义重命名为 ArgBuild。
                row[6] = 317
            elif row[0] == 0 and row[3] == 4 and row[5] == "Submit":
                row[7] = 390
            elif row[0] == 0 and row[5] == "OrchestrationReplay":
                row[7] = 390
            elif row[0] == 0 and row[5] == "FinalDrain":
                row[6:8] = [390, 410]
        rows.extend(
            [
                [0, 0, 0, 4, -1, "Atomic", 348, 349, 0x54, 46],
                [0, 0, 0, 4, -1, "Dcci", 360, 361, 0x100D, 11],
                [0, 0, 0, -1, -1, "Dcci", 400, 401, 0x31D, 8],
            ]
        )
        raw_count = len(rows)
        _refresh_summary(capture)

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            merged = json.loads(output_path.read_text(encoding="utf-8"))

        pack_events = [
            event
            for event in merged["traceEvents"]
            if event.get("name")
            == "winner_build.pack_execution_payload[GM+Scalar]#4"
        ]
        self.assertEqual(len(pack_events), 1)
        self.assertEqual(pack_events[0]["ts"], 0.339)
        self.assertEqual(pack_events[0]["dur"], 0.011)
        self.assertEqual(len(rows), raw_count)

        # 同一条 Claim.end -> Materialize.start 原 residual 已按中央 ticket
        # 的真实业务重命名，没有再并排生成第二条派生事件。
        self.assertEqual(
            sum(
                event.get("name")
                == "arg_build.plan_and_construct_args[GM+Scalar]#4"
                for event in merged["traceEvents"]
            ),
            1,
        )

    def test_v5_cross_core_large_gaps_use_existing_boundaries_only(
        self,
    ) -> None:
        # 这里只验证离线推导合同；时间刻意使用 1 GHz 下的 tick，使 1 us
        # 门槛可直接读成 1000 tick。每段两端都来自现有父区间或原语 raw，
        # generator 不得修改、复制或补造设备记录。
        rows = [
            # Materialize：Scalar layout -> heap atomic -> GM descriptor/delta
            (0, 0, 0, 10, 0, "Materialize", 10_000, 30_000, 0, 0),
            (0, 0, 0, 10, -1, "Atomic", 15_000, 16_000, 0x50, 15),
            (0, 0, 0, 10, -1, "Atomic", 17_000, 18_000, 0x52, 18),
            (
                0, 0, 0, 10, 0,
                "SharedMaterializePublishTaskOutputs",
                21_000, 29_000, 0, 0,
            ),
            # Fanin：首次共享 Load、history invalidate 后的 GM 读取，以及
            # 最后把验证结果提交到本地 fanin 前缀。
            (0, 0, 0, 11, 1, "Fanin", 40_000, 52_000, 0, 0),
            (0, 0, 0, 3, -1, "Atomic", 43_000, 44_000, 0x50, 23),
            (0, 0, 0, 3, -1, "Dcci", 45_000, 46_000, 0x10C, 0),
            (0, 0, 0, 4, -1, "Atomic", 48_000, 49_000, 0x50, 23),
            (0, 0, 0, 4, -1, "Atomic", 49_000, 50_000, 0x50, 25),
            # WinnerBuild：source resolve 中嵌套 DCCI，随后 reserve/pack/flush。
            (0, 0, 0, 11, 1, "WinnerBuild", 60_000, 80_000, 0, 0),
            (0, 0, 0, 11, -1, "Dcci", 62_000, 63_000, 0x10C, 10),
            (0, 0, 0, 11, -1, "Atomic", 65_000, 66_000, 0x54, 46),
            (0, 0, 0, 11, -1, "Dcci", 72_000, 73_000, 0x10D, 11),
            (0, 0, 0, 11, -1, "Atomic", 74_000, 75_000, 0x54, 47),
            # EfDrain：payload invalidate 后 GM copy/bind，再逐 fanin 推进。
            (0, 0, 0, 12, -1, "EfDrain", 90_000, 130_500, 0, 0),
            (0, 0, 0, 20, -1, "Atomic", 92_000, 93_000, 0x50, 45),
            (0, 0, 0, 20, -1, "Atomic", 95_000, 96_000, 0x54, 48),
            (0, 0, 0, 20, -1, "Dcci", 97_000, 98_000, 0x10C, 12),
            (0, 0, 0, 7, -1, "Atomic", 103_000, 104_000, 0x50, 5),
            (0, 0, 0, 8, -1, "Atomic", 106_000, 107_000, 0x50, 5),
            (0, 0, 0, 21, -1, "Atomic", 109_000, 110_000, 0x50, 45),
            (0, 0, 0, 22, -1, "Atomic", 112_000, 113_000, 0x50, 45),
            (0, 0, 0, -1, -1, "Atomic", 115_000, 116_000, 0x50, 2),
            (0, 0, 0, 20, -1, "Atomic", 117_000, 118_000, 0x54, 51),
            (0, 0, 0, 9, -1, "Atomic", 120_000, 121_000, 0x50, 5),
            # 500 tick 的尾缝低于 1 us，不能继续膨胀 merged。
            (0, 0, 0, 23, -1, "Atomic", 129_500, 130_000, 0x50, 45),
        ]
        raw_count = len(rows)
        spans = list(
            _iter_v5_cross_core_semantic_gap_spans(
                rows, 1_000_000_000, 5, "shared", "central_ticket"
            )
        )
        self.assertEqual(len(rows), raw_count)
        by_name = {span[5]: span[3:5] for span in spans}
        self.assertEqual(
            by_name["materialize.validate_output_layout[Scalar]#10"],
            (10_000, 15_000),
        )
        self.assertEqual(
            by_name[
                "materialize.write_descriptors_and_prepare_writer_delta"
                "[GM+Scalar]#10"
            ],
            (18_000, 21_000),
        )
        self.assertEqual(
            by_name["fanin.scan_and_validate_refs[GM+Scalar]#11"],
            (40_000, 43_000),
        )
        self.assertEqual(
            by_name["fanin.read_writer_history[GM+Scalar]#11"],
            (46_000, 48_000),
        )
        self.assertEqual(
            by_name["fanin.commit_validated_edges[Scalar]#11"],
            (50_000, 52_000),
        )
        self.assertEqual(
            by_name[
                "winner_build.resolve_payload_sources[GM+Scalar]#11"
            ],
            (60_000, 65_000),
        )
        self.assertEqual(
            by_name[
                "execute.bind_payload_and_rebuild_args[GM+Scalar]#20"
            ],
            (98_000, 103_000),
        )
        self.assertIn(
            "execute.advance_fanin_prefix[GM+Scalar]", by_name
        )
        self.assertIn(
            "efdrain.scan_exec_candidates[GM+Scalar]", by_name
        )
        self.assertIn(
            "efdrain.evaluate_exec_claim[GM+Scalar]#20", by_name
        )
        self.assertIn(
            "efdrain.return_and_check_fatal[GM+Scalar]", by_name
        )
        self.assertIn(
            "execute.recycle_token_and_continue[GM+Scalar]#20", by_name
        )

    def test_v5_cross_core_final_drain_gaps_ignore_poll_batch_overlay(
        self,
    ) -> None:
        # PollBatch 覆盖整个 FinalDrain，但它只是多次 atomic load 的汇总
        # 时间区间，中间仍会执行 GM/Scalar 工作。派生器必须忽略它作为
        # 切点，并用已有 direct Atomic/DCCI/Kernel/Commit 端点补齐大缝。
        rows = [
            (0, 0, 0, -1, -1, "FinalDrain", 200_000, 260_000, 0, 0),
            (
                0, 0, 0, -1, -1, "Atomic",
                200_000, 260_000, 0x290, 5,
            ),
            (0, 0, 0, 100, -1, "Atomic", 202_000, 202_200, 0x50, 45),
            (0, 0, 0, 100, -1, "Atomic", 204_000, 204_200, 0x54, 48),
            (0, 0, 0, 100, -1, "Dcci", 205_000, 205_200, 0x10C, 12),
            (0, 0, 0, 101, -1, "Atomic", 209_000, 209_200, 0x50, 45),
            (0, 0, 0, 101, -1, "Atomic", 211_000, 211_200, 0x50, 45),
            (0, 0, 0, 101, -1, "Atomic", 213_000, 213_200, 0x54, 48),
            (0, 0, 0, 101, -1, "Dcci", 214_000, 214_200, 0x10C, 12),
            (0, 0, 0, 101, 1, "Kernel", 220_000, 230_000, 0, 0),
            (0, 0, 0, 101, 1, "Commit", 231_000, 231_000, 0, 0),
            (0, 0, 0, 102, -1, "Atomic", 234_000, 234_200, 0x50, 45),
            (0, 0, 0, -1, -1, "Atomic", 237_000, 237_200, 0x52, 52),
            # 非 root 用独立父区间验证 arrival 后只执行本核关闭逻辑。
            (1, 1, 0, -1, -1, "FinalDrain", 301_500, 306_000, 0, 0),
            (1, 1, 0, -1, -1, "Atomic", 302_000, 302_200, 0x52, 52),
        ]
        raw_count = len(rows)
        spans = list(
            _iter_v5_cross_core_semantic_gap_spans(
                rows, 1_000_000_000, 5, "shared", "central_ticket"
            )
        )
        self.assertEqual(len(rows), raw_count)
        by_name = {span[5]: span[3:5] for span in spans}
        self.assertEqual(
            by_name["final_drain.inspect_tokens_and_plan[GM+Scalar]"],
            (200_000, 202_000),
        )
        self.assertEqual(
            by_name[
                "final_drain.evaluate_exec_claim[GM+Scalar]#100"
            ],
            (202_200, 204_000),
        )
        self.assertEqual(
            by_name[
                "final_drain.defer_token_and_scan_candidates"
                "[AtomicPoll+GM+Scalar]"
            ],
            (205_200, 209_000),
        )
        self.assertEqual(
            by_name[
                "final_drain.reobserve_blocked_candidate[GM+Scalar]#101"
            ],
            (209_200, 211_000),
        )
        self.assertEqual(
            by_name[
                "final_drain.wait_fanin_and_prepare_engine"
                "[AtomicPoll+GM+Scalar]#101"
            ],
            (214_200, 220_000),
        )
        self.assertEqual(
            by_name[
                "final_drain.recycle_token_and_scan_candidates[GM+Scalar]"
            ],
            (231_000, 234_000),
        )
        self.assertEqual(
            by_name[
                "final_drain.verify_local_drain_and_encode_arrival"
                "[GM+Scalar]"
            ],
            (234_200, 237_000),
        )
        self.assertEqual(
            by_name[
                "final_drain.root_wait_arrivals_and_validate"
                "[AtomicPoll+GM+Scalar]"
            ],
            (237_200, 260_000),
        )
        self.assertEqual(
            by_name["final_drain.close_after_arrival[Scalar]"],
            (302_200, 306_000),
        )
        self.assertTrue(
            all(end - start >= 1_000 for *_, start, end, _name in spans)
        )

    def test_v4_shared_task_zero_forbids_insert_turn_poll_batch(self) -> None:
        capture = _v5_shared_register_atomic_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        # task 0 的 Register.start->metadata.start 仍保留为闭合前段，但
        # task 0 没有前驱，不能伪造 SharedInsertTurnPoll。
        rows.append(
            [
                0,
                0,
                0,
                -1,
                -1,
                "Atomic",
                120,
                124,
                (1 << 8) | 0xD0,
                19,
            ]
        )
        _refresh_summary(capture)
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "none before the first writer.*expected=0",
            ):
                convert(input_path, output_path)
            self.assertFalse(output_path.exists())

    def test_v4_shared_register_atomic_schema_is_fail_closed(self) -> None:
        cases = {
            "poll_direct": (
                19,
                [0, 0, 0, -1, -1, "Atomic", 120, 124, 0x50, 19],
            ),
            "poll_wrong_op": (
                19,
                [0, 0, 0, -1, -1, "Atomic", 120, 124, (3 << 8) | 0xD1, 19],
            ),
            "handoff_wrong_op": (
                20,
                [0, 0, 0, 0, -1, "Atomic", 134, 140, 0x50, 20],
            ),
            "handoff_without_task": (
                20,
                [0, 0, 0, -1, -1, "Atomic", 134, 140, 0x54, 20],
            ),
        }
        for label, (site_id, replacement) in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                capture = _v5_shared_register_atomic_capture()
                rows = capture["fdwic_events"]
                assert isinstance(rows, list)
                row_index = next(
                    index
                    for index, row in enumerate(rows)
                    if row[5] == "Atomic" and row[9] == site_id
                )
                rows[row_index] = replacement
                _refresh_summary(capture)
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError,
                    "aggregate-only Atomic site|"
                    "invalid Atomic PollBatch|invalid direct Atomic",
                ):
                    convert(input_path, output_path)
                self.assertFalse(output_path.exists())

    def test_v4_shared_register_atomic_structure_closes_per_winner(
        self,
    ) -> None:
        cases = (
            ("missing_poll", "SharedInsertTurnPoll PollBatch"),
            ("duplicate_poll", "SharedInsertTurnPoll PollBatch"),
            ("poll_boundary", "SharedInsertTurnPoll PollBatch"),
            ("missing_handoff", "SharedInsertTurnHandoff direct CompareExchange"),
            ("duplicate_handoff", "SharedInsertTurnHandoff direct CompareExchange"),
            ("handoff_boundary", "identity or boundary"),
            ("handoff_task", "SharedInsertTurnHandoff direct CompareExchange"),
        )
        for label, expected in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                capture = _v5_shared_register_atomic_capture()
                rows = capture["fdwic_events"]
                assert isinstance(rows, list)
                poll = next(
                    row for row in rows if row[5] == "Atomic" and row[9] == 19
                )
                handoff = next(
                    row for row in rows if row[5] == "Atomic" and row[9] == 20
                )
                if label == "missing_poll":
                    rows.remove(poll)
                elif label == "duplicate_poll":
                    rows.append(list(poll))
                elif label == "poll_boundary":
                    poll[6] = int(poll[6]) + 1
                elif label == "missing_handoff":
                    rows.remove(handoff)
                elif label == "duplicate_handoff":
                    rows.append(list(handoff))
                elif label == "handoff_boundary":
                    handoff[6] = int(handoff[6]) - 1
                elif label == "handoff_task":
                    handoff[3] = 1
                else:
                    self.fail(f"unhandled mutation {label}")
                _refresh_summary(capture)
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    convert(input_path, output_path)
                self.assertFalse(output_path.exists())

    def test_v4_shared_register_atomics_keep_cpu_source_issue_boundary(
        self,
    ) -> None:
        capture = _v5_shared_register_atomic_capture(dependency_applied=False)
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        for row in rows:
            if row[5] == "Atomic" and row[9] in (19, 20):
                row[8] = int(row[8]) & ~0x40
        _refresh_summary(capture)
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            names = {
                event["name"]
                for event in json.loads(
                    output_path.read_text(encoding="utf-8")
                )["traceEvents"]
            }

        self.assertIn(
            "atomic.poll_batch.source_issue.shared_insert_predecessor_poll.load×3",
            names,
        )
        self.assertIn(
            "atomic.source_issue.shared_insert_completion_publish.compare_exchange#0",
            names,
        )

    def test_v4_shared_poll_return_ready_requires_dependency_evidence(self) -> None:
        capture = _v5_shared_register_atomic_capture(dependency_applied=False)
        # Handoff 是 source-issue；先删除它，精确验证 PollBatch 自己的门禁。
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        capture["fdwic_events"] = [
            row for row in rows if not (row[5] == "Atomic" and row[9] == 20)
        ]
        _refresh_summary(capture)
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "PollBatch return_ready=True.*ClockBaseline",
            ):
                convert(input_path, output_path)
            self.assertFalse(output_path.exists())

    def test_shared_register_atomic_sites_require_shared_schema_v4(self) -> None:
        cases = (
            [0, 0, 0, -1, -1, "Atomic", 100, 110, (3 << 8) | 0x90, 19],
            [0, 0, 0, 0, -1, "Atomic", 100, 110, 0x54, 20],
        )
        for row in cases:
            with self.subTest(site=row[9]), tempfile.TemporaryDirectory() as directory:
                capture = _v3_capture([row])
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError, "requires shared schema-v5"
                ):
                    convert(input_path, output_path)
                self.assertFalse(output_path.exists())

        for site_id, row in (
            (19, [0, 0, 0, -1, -1, "Atomic", 120, 124, (3 << 8) | 0xD0, 19]),
            (20, [0, 0, 0, 0, -1, "Atomic", 134, 140, 0x54, 20]),
        ):
            with self.subTest(private_v4_site=site_id), tempfile.TemporaryDirectory() as directory:
                capture = _v5_capture(
                    [
                        [0, 0, 0, 0, -1, "Claim", 110, 115, 0x3, 1],
                        [0, 0, 0, 0, -1, "AllocComplete", 140, 145, 0, 0],
                        [0, 0, 0, 0, -1, "Submit", 100, 150, 1, 1],
                    ],
                    tensormap_mode="private",
                )
                capture["l2_swimlane_level"] = 4
                rows = capture["fdwic_events"]
                assert isinstance(rows, list)
                rows.extend(
                    [
                        [0, 0, 0, -1, -1, "ClockBaseline", 10, 11, 0, 0],
                        [0, 0, 0, -1, -1, "ClockBaseline", 12, 13, 0x3, 0],
                        row,
                    ]
                )
                _refresh_summary(capture)
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError, "requires shared schema-v5"
                ):
                    convert(input_path, output_path)
                self.assertFalse(output_path.exists())

    def test_v4_shared_loser_forbids_insert_turn_atomics(self) -> None:
        for site_id, row in (
            (19, [0, 0, 0, -1, -1, "Atomic", 120, 124, (3 << 8) | 0xD0, 19]),
            (20, [0, 0, 0, 0, -1, "Atomic", 134, 140, 0x54, 20]),
        ):
            with self.subTest(site=site_id), tempfile.TemporaryDirectory() as directory:
                capture = _v5_capture(
                    [
                        # 唯一 task 明确是 Alloc loser：没有 Register owner。
                        [0, 0, 0, 0, -1, "Claim", 110, 115, 0x2, 1],
                        [0, 0, 0, 0, -1, "Submit", 100, 150, 0, 1],
                    ],
                    tensormap_mode="shared",
                )
                capture["l2_swimlane_level"] = 4
                rows = capture["fdwic_events"]
                assert isinstance(rows, list)
                rows.extend(
                    [
                        [0, 0, 0, -1, -1, "ClockBaseline", 10, 11, 0, 0],
                        [0, 0, 0, -1, -1, "ClockBaseline", 12, 13, 0x3, 0],
                        row,
                    ]
                )
                _refresh_summary(capture)
                input_path = Path(directory) / "raw.json"
                output_path = Path(directory) / "merged.json"
                input_path.write_text(json.dumps(capture), encoding="utf-8")
                with self.assertRaisesRegex(
                    ValueError, "orphan or duplicate SharedInsertTurn"
                ):
                    convert(input_path, output_path)
                self.assertFalse(output_path.exists())

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
                self.assertEqual((batch["pid"], batch["tid"]), (0, 1))
                self.assertEqual(batch["ph"], "X")
                self.assertAlmostEqual(batch["dur"], 0.8)
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

    def test_v5_final_drain_nests_poll_windows_and_scalar_task(self) -> None:
        capture = _v5_shared_register_atomic_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        # helper 的 FinalDrain=[350,370)。两类 PollBatch 和 task.execute
        # 都有 raw 权威区间，且严格形成父子包含关系；merged 应按同一
        # scalar tid 的嵌套 slice 展示，而不是把 poll 压成单点。
        rows.extend(
            [
                [0, 0, 0, -1, -1, "Atomic", 351, 369, (11 << 8) | 0x90, 5],
                [0, 0, 0, -1, -1, "Atomic", 352, 368, (19 << 8) | 0x90, 14],
                [0, 0, 0, 2, 1, "Kernel", 355, 365, 0, 0],
                [0, 0, 0, 2, 1, "Commit", 366, 366, 0, 0],
            ]
        )
        _refresh_summary(capture)

        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "raw.json"
            output_path = Path(directory) / "merged.json"
            input_path.write_text(json.dumps(capture), encoding="utf-8")
            convert(input_path, output_path)
            events = json.loads(output_path.read_text(encoding="utf-8"))[
                "traceEvents"
            ]

        final_drain = next(
            event for event in events if event.get("name") == "final_drain"
        )
        scalar_task = next(
            event
            for event in events
            if event.get("name") == "task.execute.SF#2"
        )
        kernel = next(
            event for event in events if event.get("name") == "SF#2"
        )
        fanin_poll = next(
            event
            for event in events
            if event.get("name")
            == "atomic.poll_batch.fanin_flag_load.load×11"
        )
        replay_poll = next(
            event
            for event in events
            if event.get("name")
            == "atomic.poll_batch.replay_done_poll.load×19"
        )

        self.assertEqual((final_drain["pid"], final_drain["tid"]), (0, 1))
        self.assertEqual((scalar_task["pid"], scalar_task["tid"]), (0, 1))
        self.assertEqual((kernel["pid"], kernel["tid"]), (0, 4))
        self.assertEqual(
            (scalar_task["ts"], scalar_task["dur"]),
            (kernel["ts"], kernel["dur"]),
        )
        self.assertLessEqual(final_drain["ts"], scalar_task["ts"])
        self.assertGreaterEqual(
            final_drain["ts"] + final_drain["dur"],
            scalar_task["ts"] + scalar_task["dur"],
        )
        for outer, inner in (
            (final_drain, fanin_poll),
            (fanin_poll, replay_poll),
            (replay_poll, scalar_task),
        ):
            self.assertEqual((outer["pid"], outer["tid"]), (0, 1))
            self.assertEqual((inner["pid"], inner["tid"]), (0, 1))
            self.assertEqual((outer["ph"], inner["ph"]), ("X", "X"))
            self.assertLessEqual(outer["ts"], inner["ts"])
            self.assertGreaterEqual(
                outer["ts"] + outer["dur"],
                inner["ts"] + inner["dur"],
            )
            self.assertLess(events.index(outer), events.index(inner))

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
            (0x50, 43, -1),  # cross-core shared 站点不能出现在 schema-v3
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
