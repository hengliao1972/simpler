#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""standalone PA Submit 排他分析器的闭合、门禁与原子发布回归。"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

try:
    from .swimlane_exclusive_analyzer import (
        OVERLAY_PHASES,
        SUBMIT_PARTITION_METRICS,
        analyze_capture,
        main,
        write_analysis,
    )
except ImportError:
    from swimlane_exclusive_analyzer import (
        OVERLAY_PHASES,
        SUBMIT_PARTITION_METRICS,
        analyze_capture,
        main,
        write_analysis,
    )


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
    flags: int = 0,
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
        flags,
        auxiliary,
    ]


def _refresh_summary(capture: dict[str, object]) -> None:
    """修改 fixture 后同步 producer summary，使失败精确落在待测门禁。"""

    rows = capture["fdwic_events"]
    assert isinstance(rows, list)
    atomic_rows = [row for row in rows if row[5] == "Atomic"]
    poll_rows = [row for row in atomic_rows if int(row[8]) & 0x80]
    poll_calls = sum((int(row[8]) >> 8) & 0xFFFFFF for row in poll_rows)
    metadata = capture["metadata"]
    assert isinstance(metadata, dict)
    old_summary = metadata.get("fdwic_summary")
    dropped = int(old_summary.get("dropped_records", 0)) if isinstance(old_summary, dict) else 0
    metadata["fdwic_summary"] = {
        "records": len(rows),
        "atomic_records": len(atomic_rows),
        "clock_baseline_records": sum(row[5] == "ClockBaseline" for row in rows),
        "atomic_calls": len(atomic_rows) - len(poll_rows) + poll_calls,
        "batched_poll_calls": poll_calls,
        "poll_batch_records": len(poll_rows),
        "dropped_records": dropped,
    }


def _capture() -> dict[str, object]:
    """构造完整 32 AIC + 64 AIV、每核两个 Submit 的 schema-v3 证据。"""

    rows: list[list[object]] = []
    for core_id in range(CORE_COUNT):
        # schema-v3 每核必须有 plain/dependency 各一条。dependency_applied=1，
        # 所有消费返回值的 Atomic 因而使用 return_ready 边界。
        clock_start = 10 + core_id * 4
        rows.append(_row(core_id, -1, "ClockBaseline", clock_start, clock_start + 1))
        rows.append(
            _row(
                core_id,
                -1,
                "ClockBaseline",
                clock_start + 2,
                clock_start + 3,
                flags=0x3,
            )
        )

        base = 1000 + core_id
        for task_id, submit_offset, submit_duration in ((0, 0, 100), (1, 120, 80)):
            start = base + submit_offset
            if task_id == 0:
                # child 总计 53 cycle，SubmitResidual=47；EfDrain 内 Kernel union=10。
                efdrain = (start + 1, start + 21)
                kernel = (start + 4, start + 14)
                materialize = (start + 25, start + 35)
                prepare = (start + 36, start + 41)
                claim = (start + 43, start + 51)
                fanin = (start + 52, start + 55)
                register = (start + 56, start + 63)
            else:
                # child 总计 36 cycle，SubmitResidual=44；EfDrain 内 Kernel union=4。
                efdrain = (start + 1, start + 11)
                kernel = (start + 3, start + 7)
                materialize = (start + 14, start + 22)
                prepare = (start + 23, start + 27)
                claim = (start + 29, start + 35)
                fanin = (start + 36, start + 38)
                register = (start + 40, start + 46)

            rows.extend(
                [
                    _row(core_id, task_id, "EfDrain", *efdrain),
                    _row(core_id, task_id, "Kernel", *kernel, function_id=task_id),
                    _row(core_id, task_id, "Materialize", *materialize),
                    _row(core_id, task_id, "PrepareMap", *prepare),
                    _row(
                        core_id,
                        task_id,
                        "Claim",
                        *claim,
                        flags=0x2,
                        auxiliary=1 if task_id == 0 else 0,
                    ),
                    _row(core_id, task_id, "Fanin", *fanin, function_id=task_id),
                    _row(core_id, task_id, "Register", *register),
                    # fetch_max + result_used + return_ready，site=ClaimMax(4)。
                    _row(
                        core_id,
                        task_id,
                        "Atomic",
                        claim[0] + 1,
                        claim[0] + 3,
                        flags=0x53,
                        auxiliary=4,
                    ),
                    _row(core_id, task_id, "Commit", start + 70, start + 70),
                    # 三种 lap marker 故意覆盖显式 child，用于证明它们仅为 Overlay。
                    _row(core_id, task_id, "Build", start + 25, start + 65),
                    _row(core_id, task_id, "Replay", start + 14, start + 60),
                    _row(core_id, task_id, "Alloc", start + 25, start + 70),
                    _row(core_id, task_id, "Submit", start, start + submit_duration),
                ]
            )

    capture: dict[str, object] = {
        "l2_swimlane_level": 4,
        "metadata": {
            "clock_freq_hz": 1_000_000_000,
            "num_cores": CORE_COUNT,
            "trace_schema_version": 3,
            "tensor_map_mode": "private",
            "core_types": [_topology(core_id)[2] for core_id in range(CORE_COUNT)],
        },
        "fdwic_events": rows,
    }
    _refresh_summary(capture)
    return capture


def _v4_capture() -> dict[str, object]:
    """在 v3 结构样本上替换真实尾动作，并补齐两个首尾相接的父 span。"""

    capture = _capture()
    capture["l2_swimlane_level"] = 4
    metadata = capture["metadata"]
    assert isinstance(metadata, dict)
    metadata["trace_schema_version"] = 4
    source_rows = capture["fdwic_events"]
    assert isinstance(source_rows, list)
    rows: list[list[object]] = []
    for original in source_rows:
        row = list(original)
        core_id = int(row[0])
        task_id = int(row[3])
        phase = str(row[5])
        winner = (core_id, task_id) in {(0, 0), (1, 1)}
        if phase in {"Build", "Replay", "Alloc"}:
            continue
        if phase == "Fanin" and (not winner or task_id == 0):
            continue
        base = 1000 + core_id
        submit_start = base + (0 if task_id == 0 else 120)
        if phase == "Claim":
            # schema-v4 跟随 compete-first eager 生产路径：Claim 先于
            # callback 构参与 Materialize。两类 task 都保留 v3 fixture
            # 的 Claim 时长，只调整边界顺序。
            row[6:8] = (
                [submit_start + 25, submit_start + 33]
                if task_id == 0
                else [submit_start + 14, submit_start + 20]
            )
            row[8] = 0x3 if winner else 0x2
        elif phase == "Materialize":
            row[6:8] = (
                [submit_start + 36, submit_start + 46]
                if task_id == 0
                else [submit_start + 24, submit_start + 32]
            )
        elif phase == "PrepareMap":
            row[6:8] = (
                [submit_start + 47, submit_start + 52]
                if task_id == 0
                else [submit_start + 33, submit_start + 37]
            )
        elif phase == "Fanin":
            row[6:8] = [submit_start + 40, submit_start + 42]
        elif phase == "Register":
            row[6:8] = (
                [submit_start + 56, submit_start + 63]
                if task_id == 0
                else [submit_start + 43, submit_start + 49]
            )
        elif phase == "Atomic":
            row[6:8] = (
                [submit_start + 26, submit_start + 28]
                if task_id == 0
                else [submit_start + 15, submit_start + 17]
            )
        elif phase == "Submit":
            row[8] = 1 if winner else 0
            row[9] = 1 if task_id == 0 else 0
        rows.append(row)

        if phase != "Submit":
            continue
        submit_start = int(row[6])
        if winner and task_id == 0:
            rows.append(
                _row(
                    core_id,
                    task_id,
                    "AllocComplete",
                    submit_start + 65,
                    submit_start + 75,
                )
            )
        elif winner:
            rows.append(
                _row(
                    core_id,
                    task_id,
                    "WinnerBuild",
                    submit_start + 50,
                    submit_start + 60,
                    function_id=task_id % 5 - 1,
                )
            )

    for core_id in range(CORE_COUNT):
        base = 1000 + core_id
        rows.extend(
            [
                _row(core_id, -1, "OrchestrationReplay", base - 10, base + 210),
                _row(core_id, -1, "FinalDrain", base + 210, base + 250),
                _row(core_id, 1, "Kernel", base + 220, base + 230, function_id=0),
            ]
        )
    capture["fdwic_events"] = rows
    _refresh_summary(capture)
    return capture


def _shared_v4_capture(*, batches: int = 4) -> dict[str, object]:
    """构造 shared Alloc 仅由四个固定 owner 记录的完整 schema-v4 样本。"""

    owner_by_shard = (0, 34, 37, 3)
    logical_task_count = batches * 5
    rows: list[list[object]] = []
    for core_id in range(CORE_COUNT):
        base = 1000 + core_id
        for task_id in range(logical_task_count):
            is_alloc = task_id % 5 == 0
            if is_alloc and owner_by_shard[task_id % 4] != core_id:
                continue
            start = base + task_id * 20
            rows.extend(
                [
                    _row(core_id, task_id, "EfDrain", start + 1, start + 2),
                    _row(
                        core_id,
                        task_id,
                        "Claim",
                        start + 3,
                        start + 4,
                        flags=0x2,
                        auxiliary=1 if is_alloc else 0,
                    ),
                    _row(core_id, task_id, "Materialize", start + 5, start + 6),
                    _row(core_id, task_id, "PrepareMap", start + 6, start + 7),
                    _row(core_id, task_id, "Register", start + 8, start + 9),
                    _row(
                        core_id,
                        task_id,
                        "Submit",
                        start,
                        start + 10,
                        auxiliary=1 if is_alloc else 0,
                    ),
                ]
            )
        orchestration_end = base + logical_task_count * 20
        rows.extend(
            [
                _row(
                    core_id,
                    -1,
                    "OrchestrationReplay",
                    base - 10,
                    orchestration_end,
                ),
                _row(
                    core_id,
                    -1,
                    "FinalDrain",
                    orchestration_end,
                    orchestration_end + 10,
                ),
            ]
        )

    capture: dict[str, object] = {
        "l2_swimlane_level": 1,
        "metadata": {
            "clock_freq_hz": 1_000_000_000,
            "num_cores": CORE_COUNT,
            "trace_schema_version": 4,
            "tensor_map_mode": "shared",
            "core_types": [_topology(core_id)[2] for core_id in range(CORE_COUNT)],
        },
        "fdwic_events": rows,
    }
    _refresh_summary(capture)
    return capture


class SwimlaneExclusiveAnalyzerTest(unittest.TestCase):
    def _write(self, directory: str, capture: dict[str, object]) -> Path:
        path = Path(directory) / "l2_swimlane_records.json"
        path.write_text(json.dumps(capture, ensure_ascii=False), encoding="utf-8")
        return path

    def test_valid_capture_closes_all_integer_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, _capture())
            report = analyze_capture(path)

        self.assertEqual(report["validation"]["status"], "PASS")
        self.assertEqual(report["capture"]["core_count"], 96)
        self.assertEqual(report["capture"]["logical_task_count"], 2)
        self.assertEqual(
            report["capture"]["profiled_submit_count_per_core"],
            {"min": 2, "max": 2},
        )
        self.assertEqual(report["global_submit_makespan"]["duration_cycles"], 295)

        metrics = report["aggregate_core_work"]["metrics_cycles"]
        self.assertEqual(metrics["submit_envelope"], 19_200)
        self.assertEqual(metrics["submit_union"], 17_280)
        self.assertEqual(metrics["between_submit_residual"], 1_920)
        self.assertEqual(metrics["efdrain"], 2_880)
        self.assertEqual(metrics["materialize"], 1_728)
        self.assertEqual(metrics["prepare_map"], 864)
        self.assertEqual(metrics["claim"], 1_344)
        self.assertEqual(metrics["fanin"], 480)
        self.assertEqual(metrics["register"], 1_248)
        self.assertEqual(metrics["submit_residual"], 8_736)
        self.assertEqual(metrics["efdrain_kernel_union"], 1_344)
        self.assertEqual(metrics["efdrain_control"], 1_536)
        self.assertEqual(
            sum(metrics[name] for name in SUBMIT_PARTITION_METRICS),
            metrics["submit_union"],
        )
        self.assertEqual(
            metrics["submit_union"] + metrics["between_submit_residual"],
            metrics["submit_envelope"],
        )
        self.assertEqual(
            metrics["efdrain_kernel_union"] + metrics["efdrain_control"],
            metrics["efdrain"],
        )
        residual = report["residual_breakdown"]
        self.assertEqual(residual["submit_internal_residual"]["total_cycles"], 1_920)
        self.assertEqual(residual["submit_tail_residual"]["total_cycles"], 6_816)
        self.assertEqual(residual["between_submit_residual"]["total_cycles"], 1_920)
        self.assertEqual(
            residual["submit_internal_residual"]["total_cycles"]
            + residual["submit_tail_residual"]["total_cycles"],
            metrics["submit_residual"],
        )
        for section in (
            "submit_internal_residual",
            "submit_tail_residual",
            "between_submit_residual",
        ):
            segments = residual[section]["segments"]
            self.assertEqual(
                sum(segment["cycles"] for segment in segments),
                residual[section]["total_cycles"],
            )
            for segment in segments:
                self.assertEqual(
                    segment["aic_cycles"] + segment["aiv_cycles"],
                    segment["cycles"],
                )
        self.assertEqual(len(report["per_core"]), 96)
        self.assertIsNone(report["kernel_containment"]["orphan_events"])
        self.assertEqual(
            report["kernel_containment"]["unclassified_without_v4_parent_events"],
            0,
        )

    def test_v4_closes_true_tails_and_worker_parent_hierarchy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, _v4_capture())
            report = analyze_capture(path)

        self.assertEqual(report["capture"]["trace_schema_version"], 4)
        self.assertEqual(
            report["semantics"]["exclusive_submit_children"][-2:],
            ["WinnerBuild", "AllocComplete"],
        )
        validation = report["validation"]
        self.assertIs(validation["parent_boundaries_adjacent"], True)
        self.assertEqual(validation["legacy_lap_records"], 0)
        self.assertIs(validation["worker_completion_partition_exact"], True)

        metrics = report["aggregate_core_work"]["metrics_cycles"]
        self.assertEqual(metrics["submit_union"], 17_280)
        self.assertEqual(metrics["fanin"], 2)
        self.assertEqual(metrics["winner_build"], 10)
        self.assertEqual(metrics["alloc_complete"], 10)
        self.assertEqual(metrics["submit_residual"], 9_194)
        self.assertEqual(metrics["orchestration_setup"], 960)
        self.assertEqual(metrics["orchestration_tail"], 960)
        self.assertEqual(metrics["orchestration_replay"], 21_120)
        self.assertEqual(metrics["final_drain"], 3_840)
        self.assertEqual(metrics["final_drain_kernel_union"], 960)
        self.assertEqual(metrics["final_drain_residual"], 2_880)
        self.assertEqual(metrics["worker_completion"], 24_960)
        residual = report["residual_breakdown"]
        self.assertEqual(residual["submit_internal_residual"]["total_cycles"], 2_689)
        self.assertEqual(residual["submit_tail_residual"]["total_cycles"], 6_505)
        self.assertEqual(residual["between_submit_residual"]["total_cycles"], 1_920)
        self.assertAlmostEqual(
            residual["submit_internal_residual"]["share_of_submit_union"],
            2_689 / 17_280,
        )
        self.assertAlmostEqual(
            residual["submit_tail_residual"]["share_of_submit_union"],
            6_505 / 17_280,
        )
        self.assertAlmostEqual(
            residual["between_submit_residual"]["share_of_submit_envelope"],
            1_920 / 19_200,
        )
        tail_boundaries = {
            segment["boundary"]
            for segment in residual["submit_tail_residual"]["segments"]
        }
        self.assertIn("Register->SubmitEnd", tail_boundaries)
        self.assertIn("AllocComplete->SubmitEnd", tail_boundaries)
        self.assertIn("WinnerBuild->SubmitEnd", tail_boundaries)
        self.assertTrue(
            all(boundary.endswith("->SubmitEnd") for boundary in tail_boundaries)
        )
        self.assertTrue(
            all(
                not segment["boundary"].endswith("->SubmitEnd")
                for segment in residual["submit_internal_residual"]["segments"]
            )
        )
        internal_boundaries = {
            segment["boundary"]
            for segment in residual["submit_internal_residual"]["segments"]
        }
        self.assertIn("Claim->Materialize", internal_boundaries)

        closure = report["aggregate_core_work"]["closure"]
        for name in (
            "submit_partition",
            "submit_envelope",
            "efdrain_partition",
            "orchestration_replay",
            "final_drain",
            "worker_completion",
        ):
            self.assertIs(closure[name]["exact"], True)
        self.assertEqual(report["kernel_containment"]["inside_efdrain_events"], 192)
        self.assertEqual(report["kernel_containment"]["inside_final_drain_events"], 96)
        self.assertEqual(report["kernel_containment"]["orphan_events"], 0)
        self.assertEqual(
            report["kernel_containment"]["unclassified_without_v4_parent_events"],
            0,
        )

    def test_v4_shared_validates_sparse_alloc_owners_and_reports_profiled_counts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, _shared_v4_capture())
            report = analyze_capture(path)

        capture = report["capture"]
        self.assertEqual(capture["tensor_map_mode"], "shared")
        self.assertEqual(capture["logical_task_count"], 20)
        self.assertEqual(
            capture["profiled_submit_count_per_core"],
            {"min": 16, "max": 17},
        )
        self.assertEqual(capture["profiled_submit_count_total"], 1540)
        self.assertIs(
            report["validation"]["submit_task_ids_match_tensor_map_mode"],
            True,
        )
        self.assertNotIn(
            "task_ids_contiguous_and_equal_per_core", report["validation"]
        )
        self.assertEqual(
            report["semantics"]["shared_alloc_owner_by_shard"],
            [0, 34, 37, 3],
        )
        self.assertEqual(report["per_core"][0]["submit_count"], 17)
        self.assertEqual(report["per_core"][34]["submit_count"], 17)
        self.assertEqual(report["per_core"][37]["submit_count"], 17)
        self.assertEqual(report["per_core"][3]["submit_count"], 17)
        self.assertEqual(report["per_core"][1]["submit_count"], 16)

        # core1 的首个 Alloc 早退落入 OrchestrationSetup；后续三个
        # 稀疏 Alloc 则扩大相邻 full-path Submit 间的 residual。
        core1_metrics = report["per_core"][1]["metrics_cycles"]
        self.assertEqual(core1_metrics["orchestration_setup"], 30)
        self.assertEqual(core1_metrics["between_submit_residual"], 210)

    def test_v4_shared_rejects_alloc_submit_on_non_owner(self) -> None:
        capture = _shared_v4_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        # 复制 core0/task0 的完整路径到非 owner core1；所有 row key、
        # parent containment 和 Claim/Submit 对仍合法，应由 mode 精确集合拒绝。
        for source in list(rows):
            if source[0] != 0 or source[3] != 0:
                continue
            copied = list(source)
            copied[0] = 1
            copied[1] = 1
            copied[2] = 0
            copied[6] = int(copied[6]) + 1
            copied[7] = int(copied[7]) + 1
            rows.append(copied)
        _refresh_summary(capture)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "shared full-path set"):
                analyze_capture(path)

    def test_v4_shared_rejects_missing_non_alloc_full_path(self) -> None:
        capture = _shared_v4_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        capture["fdwic_events"] = [
            row for row in rows if not (row[0] == 1 and row[3] == 1)
        ]
        _refresh_summary(capture)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "shared full-path set"):
                analyze_capture(path)

    def test_v4_phase_only_capture_still_has_dropped_evidence_and_closes(self) -> None:
        capture = _v4_capture()
        capture["l2_swimlane_level"] = 1
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        capture["fdwic_events"] = [
            row for row in rows if row[5] not in {"Atomic", "ClockBaseline"}
        ]
        _refresh_summary(capture)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            report = analyze_capture(path)

        self.assertEqual(report["validation"]["dropped_records"], 0)
        self.assertEqual(report["overlays"]["Atomic"]["event_count"], 0)
        self.assertEqual(report["overlays"]["ClockBaseline"]["event_count"], 0)
        self.assertIs(
            report["aggregate_core_work"]["closure"]["worker_completion"]["exact"],
            True,
        )

    def test_v4_parent_boundary_gap_is_rejected(self) -> None:
        capture = _v4_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        final_drain = next(
            row for row in rows if row[0] == 0 and row[5] == "FinalDrain"
        )
        final_drain[6] = int(final_drain[6]) + 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(
                ValueError, "OrchestrationReplay.end must equal FinalDrain.start"
            ):
                analyze_capture(path)

    def test_v4_submit_must_stay_inside_orchestration_parent(self) -> None:
        capture = _v4_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        orchestration = next(
            row for row in rows if row[0] == 0 and row[5] == "OrchestrationReplay"
        )
        orchestration[6] = 1001
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "outside OrchestrationReplay"):
                analyze_capture(path)

    def test_v4_tail_cannot_precede_frontend_children(self) -> None:
        capture = _v4_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        winner_tail = next(
            row
            for row in rows
            if row[0] == 1 and row[3] == 1 and row[5] == "WinnerBuild"
        )
        winner_tail[6] = int(winner_tail[6]) - 20
        winner_tail[7] = int(winner_tail[7]) - 20
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "must start at or after"):
                analyze_capture(path)

    def test_v4_loser_cannot_carry_winner_only_fanin(self) -> None:
        capture = _v4_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        rows.append(_row(0, 1, "Fanin", 1156, 1158, function_id=0))
        _refresh_summary(capture)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "loser path requires 0 Fanin"):
                analyze_capture(path)

    def test_v4_kernel_must_have_one_supported_parent(self) -> None:
        capture = _v4_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        # core0 两个 Submit 之间的 orchestration gap 不是合法 Kernel 容器。
        rows.append(_row(0, 0, "Kernel", 1105, 1110, function_id=0))
        _refresh_summary(capture)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "Kernel must be contained"):
                analyze_capture(path)

    def test_v4_kernel_inside_winner_build_is_classified(self) -> None:
        capture = _v4_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        winner_build = next(
            row
            for row in rows
            if row[0] == 1 and row[3] == 1 and row[5] == "WinnerBuild"
        )
        rows.append(
            _row(
                1,
                0,
                "Kernel",
                int(winner_build[6]) + 2,
                int(winner_build[6]) + 5,
                function_id=0,
            )
        )
        _refresh_summary(capture)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            report = analyze_capture(path)

        containment = report["kernel_containment"]
        self.assertEqual(containment["inside_winner_build_events"], 1)
        self.assertEqual(containment["inside_submit_tail_events"], 1)
        self.assertEqual(containment["orphan_events"], 0)

    def test_v4_final_drain_kernel_crossing_boundary_is_rejected(self) -> None:
        capture = _v4_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        final_drain = next(
            row for row in rows if row[0] == 0 and row[5] == "FinalDrain"
        )
        final_kernel = next(
            row
            for row in rows
            if row[0] == 0 and row[5] == "Kernel" and int(row[6]) > int(final_drain[6])
        )
        final_kernel[7] = int(final_drain[7]) + 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "crosses FinalDrain"):
                analyze_capture(path)

    def test_role_statistics_are_separate_from_global_and_core_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, _capture())
            report = analyze_capture(path)

        roles = report["per_role_core_statistics"]
        self.assertEqual(roles["aic"]["core_count"], 32)
        self.assertEqual(roles["aiv"]["core_count"], 64)
        for role in ("aic", "aiv"):
            envelope = roles[role]["metrics"]["submit_envelope"]
            self.assertEqual(envelope["median_cycles"], 200)
            self.assertEqual(envelope["p95_cycles"], 200)
            self.assertEqual(envelope["max_cycles"], 200)
        self.assertEqual(
            report["global_submit_makespan"]["semantics"],
            "cross-core wall-clock envelope of recorded full-path Submit spans; "
            "not aggregate core-work",
        )
        self.assertEqual(
            report["aggregate_core_work"]["semantics"],
            "sum of per-core cycles; not wall-clock duration",
        )

    def test_all_overlay_phases_are_reported_but_never_added(self) -> None:
        capture = _capture()
        with tempfile.TemporaryDirectory() as directory:
            baseline_path = self._write(directory, capture)
            baseline = analyze_capture(baseline_path)

            # 大幅拉长所有 Atomic/Build/Replay/Alloc span；排他结果必须保持不变。
            rows = capture["fdwic_events"]
            assert isinstance(rows, list)
            for row in rows:
                if row[5] in {"Atomic", "Build", "Replay", "Alloc"}:
                    row[7] = int(row[7]) + 10_000
            changed_path = Path(directory) / "changed.json"
            changed_path.write_text(json.dumps(capture), encoding="utf-8")
            changed = analyze_capture(changed_path)

        self.assertEqual(
            baseline["aggregate_core_work"]["metrics_cycles"],
            changed["aggregate_core_work"]["metrics_cycles"],
        )
        self.assertEqual(set(changed["overlays"]), set(OVERLAY_PHASES))
        for phase in OVERLAY_PHASES:
            self.assertIs(changed["overlays"][phase]["included_in_additive_totals"], False)
        self.assertIs(changed["semantics"]["overlays_are_additive"], False)

    def test_overlapping_submits_on_same_core_lane_are_rejected(self) -> None:
        capture = _capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        first_end = next(
            int(row[7])
            for row in rows
            if row[0] == 0 and row[3] == 0 and row[5] == "Submit"
        )
        second = next(
            row for row in rows if row[0] == 0 and row[3] == 1 and row[5] == "Submit"
        )
        second[6] = first_end - 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "overlapping Submit"):
                analyze_capture(path)

    def test_overlapping_exclusive_children_are_rejected(self) -> None:
        capture = _capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        efdrain_end = next(
            int(row[7])
            for row in rows
            if row[0] == 0 and row[3] == 0 and row[5] == "EfDrain"
        )
        materialize = next(
            row for row in rows if row[0] == 0 and row[3] == 0 and row[5] == "Materialize"
        )
        materialize[6] = efdrain_end - 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "overlapping exclusive children"):
                analyze_capture(path)

    def test_exclusive_child_task_must_match_containing_submit(self) -> None:
        capture = _v4_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        first_claim = next(
            row for row in rows if row[0] == 0 and row[3] == 0 and row[5] == "Claim"
        )
        second_claim = next(
            row for row in rows if row[0] == 0 and row[3] == 1 and row[5] == "Claim"
        )
        # 只交换时间，不改 task/flags。converter 的 schema 键与 winner 语义仍
        # 合法；分析器必须拒绝 task1 Claim 被时间包含进 task0 Submit 的伪闭合。
        first_claim[6:8], second_claim[6:8] = second_claim[6:8], first_claim[6:8]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "does not match containing Submit"):
                analyze_capture(path)

    def test_v4_rejects_removed_loser_replay_phase(self) -> None:
        capture = _v4_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        rows.append(_row(2, 0, "LoserReplay", 1065, 1065))
        _refresh_summary(capture)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "unknown phase 'LoserReplay'"):
                analyze_capture(path)

    def test_kernel_crossing_efdrain_boundary_is_rejected(self) -> None:
        capture = _capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        efdrain_end = next(
            int(row[7])
            for row in rows
            if row[0] == 0 and row[3] == 0 and row[5] == "EfDrain"
        )
        kernel = next(
            row for row in rows if row[0] == 0 and row[3] == 0 and row[5] == "Kernel"
        )
        kernel[6] = efdrain_end - 1
        kernel[7] = efdrain_end + 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "crosses EfDrain"):
                analyze_capture(path)

    def test_missing_required_child_is_rejected(self) -> None:
        capture = _capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        capture["fdwic_events"] = [
            row
            for row in rows
            if not (row[0] == 0 and row[3] == 0 and row[5] == "Register")
        ]
        _refresh_summary(capture)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "requires exactly one Register"):
                analyze_capture(path)

    def test_incomplete_task_stream_is_rejected(self) -> None:
        capture = _capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        capture["fdwic_events"] = [
            row for row in rows if not (row[0] == 95 and row[3] == 1)
        ]
        _refresh_summary(capture)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "task IDs do not match"):
                analyze_capture(path)

    def test_nonzero_dropped_records_is_rejected(self) -> None:
        capture = _capture()
        metadata = capture["metadata"]
        assert isinstance(metadata, dict)
        summary = metadata["fdwic_summary"]
        assert isinstance(summary, dict)
        summary["dropped_records"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "dropped_records"):
                analyze_capture(path)

    def test_old_schema_without_dropped_evidence_is_rejected(self) -> None:
        capture = _capture()
        capture["l2_swimlane_level"] = 1
        metadata = capture["metadata"]
        assert isinstance(metadata, dict)
        metadata["trace_schema_version"] = 2
        metadata.pop("fdwic_summary")
        # v2 不允许 v3 ClockBaseline/Atomic flags，删去它们后应由排他分析器
        # 因缺少 dropped 证据拒绝，而不是先落入 converter 的 flags 门禁。
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        capture["fdwic_events"] = [
            row for row in rows if row[5] not in {"ClockBaseline", "Atomic"}
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            with self.assertRaisesRegex(ValueError, "requires trace_schema_version=3"):
                analyze_capture(path)

    def test_write_is_atomic_and_failure_keeps_existing_output(self) -> None:
        capture = _capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        second = next(
            row for row in rows if row[0] == 0 and row[3] == 1 and row[5] == "Submit"
        )
        second[6] = 1099
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, capture)
            output = Path(directory) / "exclusive.json"
            output.write_text("existing-good-output\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlapping Submit"):
                write_analysis(path, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing-good-output\n")
            self.assertEqual(list(Path(directory).glob(".exclusive.json.*.tmp")), [])

    def test_cli_requires_output_and_publishes_complete_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, _capture())
            output = Path(directory) / "exclusive.json"
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main([str(path)])
            self.assertEqual(raised.exception.code, 2)

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(path), "-o", str(output)]), 0)
            document = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(document["validation"]["status"], "PASS")
            self.assertTrue(output.read_text(encoding="utf-8").endswith("}\n"))
            self.assertEqual(list(Path(directory).glob(".exclusive.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
