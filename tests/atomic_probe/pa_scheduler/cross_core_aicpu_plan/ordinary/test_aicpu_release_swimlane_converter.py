#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""AICPU release atomic 泳道事件的最小离线门槛。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from tests.atomic_probe.pa_scheduler.cross_core_aicpu_plan.ordinary.test_plan_swimlane_converter import (
    MODULE,
    _runtime_plan_capture,
)


class AicpuReleaseSwimlaneConverterTest(unittest.TestCase):
    def test_release_store_is_rendered_as_aicpu_atomic(self) -> None:
        capture = cast(dict[str, Any], _runtime_plan_capture())
        metadata = cast(dict[str, Any], capture["metadata"])
        tasks = cast(list[dict[str, Any]], capture["aicpu_tasks"])
        publish_begin = int(tasks[0]["publish_begin_ns"])
        metadata["aicpu_operation_trace"] = {
            "enabled": True,
            "records": 1,
            "record_bytes": 64,
            "dropped": 0,
        }
        capture["aicpu_operations"] = [
            {
                "sequence": 0,
                "task_id": 0,
                "scope": "task_publish",
                "operation": "atomic_store_release",
                "target": "cell_control",
                "clock": "aicpu_monotonic_raw_ns",
                "begin_ns": publish_begin + 1,
                "end_ns": publish_begin + 2,
                "calls": 1,
                "lines": 0,
                "first_target_index": 0,
                "last_target_index": 0,
                "first_value": 2,
                "last_value": 2,
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "aicpu_release_raw.json"
            merged_path = Path(directory) / "aicpu_release_merged.json"
            raw_path.write_text(json.dumps(capture), encoding="utf-8")
            MODULE.convert(raw_path, merged_path)
            merged = json.loads(merged_path.read_text(encoding="utf-8"))

        release_events = [
            event
            for event in merged["traceEvents"]
            if event.get("name") == "atomic.task_publish.cell_control.store_release#0"
        ]
        self.assertEqual(len(release_events), 1)
        self.assertEqual(release_events[0]["cat"], "aicpu.atomic")
        self.assertEqual(release_events[0]["tid"], MODULE.AICPU_TASK_THREAD_ID)
        self.assertGreater(release_events[0]["dur"], 0)


if __name__ == "__main__":
    unittest.main()
