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
    from .swimlane_converter import convert
except ImportError:
    # 也保留从本目录直接执行脚本的用法。
    from swimlane_converter import convert


class SwimlaneConverterLayoutTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
