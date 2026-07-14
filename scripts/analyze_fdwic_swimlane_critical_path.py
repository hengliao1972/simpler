#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Summarize fdwic swimlane records by critical core span."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

Event = list[Any]

SUBMIT_PHASE = "Submit"
LAP_PHASES = {"Replay", "Build", "Alloc"}


def latest_trace() -> Path:
    traces = list(Path("outputs").glob("TestPagedAttentionUnroll_Case1_*/l2_swimlane_records.json"))
    if not traces:
        raise FileNotFoundError("no PA Case1 fdwic swimlane trace found under outputs/")
    return max(traces, key=lambda path: path.stat().st_mtime)


def _field_int(event: Event, index: int) -> int:
    return int(event[index])


def _field_str(event: Event, index: int) -> str:
    return str(event[index])


def phase_rows(events: list[Event]) -> dict[str, list[int]]:
    rows: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for event in events:
        phase = _field_str(event, 5)
        dur = _field_int(event, 7) - _field_int(event, 6)
        rows[phase][0] += 1
        rows[phase][1] += dur
        rows[phase][2] = max(rows[phase][2], dur)
    return rows


def interval_union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    total += current_end - current_start
    return total


def submit_exclusive_rows(events: list[Event]) -> dict[str, list[int]]:
    submit_events: list[Event] = []
    child_intervals_by_core_task: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for event in events:
        phase = _field_str(event, 5)
        if phase == SUBMIT_PHASE:
            submit_events.append(event)
        elif phase not in LAP_PHASES:
            child_intervals_by_core_task[(_field_int(event, 0), _field_int(event, 2))].append(
                (_field_int(event, 6), _field_int(event, 7))
            )

    rows: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for event in submit_events:
        submit_start = _field_int(event, 6)
        submit_end = _field_int(event, 7)
        child_intervals: list[tuple[int, int]] = []
        for child_start, child_end in child_intervals_by_core_task[(_field_int(event, 0), _field_int(event, 2))]:
            if child_start >= submit_start and child_end <= submit_end:
                child_intervals.append((child_start, child_end))
        submit_ns = submit_end - submit_start
        child_ns = interval_union_ns(child_intervals)
        exclusive_ns = max(0, submit_ns - child_ns)
        rows["SubmitExclusive"][0] += 1
        rows["SubmitExclusive"][1] += exclusive_ns
        rows["SubmitExclusive"][2] = max(rows["SubmitExclusive"][2], exclusive_ns)
        rows["SubmitChildren"][0] += 1
        rows["SubmitChildren"][1] += child_ns
        rows["SubmitChildren"][2] = max(rows["SubmitChildren"][2], child_ns)
    return rows


def print_phase_table(events: list[Event], span_ns: int, title: str) -> None:
    print(title)
    print(f"{'phase':12s} {'count':>8s} {'sum_us':>12s} {'avg_ns':>10s} {'max_us':>9s} {'span%':>8s}")
    for phase, (count, total_ns, max_ns) in sorted(
        phase_rows(events).items(), key=lambda item: item[1][1], reverse=True
    ):
        avg_ns = total_ns / count if count else 0
        span_pct = total_ns * 100 / span_ns if span_ns > 0 else 0
        print(f"{phase:12s} {count:8d} {total_ns / 1000:12.3f} {avg_ns:10.1f} {max_ns / 1000:9.3f} {span_pct:7.1f}%")


def print_submit_exclusive_table(events: list[Event], span_ns: int, title: str) -> None:
    print(title)
    print(f"{'metric':15s} {'count':>8s} {'sum_us':>12s} {'avg_ns':>10s} {'max_us':>9s} {'span%':>8s}")
    for metric, (count, total_ns, max_ns) in sorted(
        submit_exclusive_rows(events).items(), key=lambda item: item[1][1], reverse=True
    ):
        avg_ns = total_ns / count if count else 0
        span_pct = total_ns * 100 / span_ns if span_ns > 0 else 0
        print(f"{metric:15s} {count:8d} {total_ns / 1000:12.3f} {avg_ns:10.1f} {max_ns / 1000:9.3f} {span_pct:7.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path, help="l2_swimlane_records.json; defaults to latest PA Case1")
    parser.add_argument("--top", type=int, default=8, help="number of core spans to print")
    args = parser.parse_args()

    path = args.trace or latest_trace()
    data = json.loads(path.read_text())
    events = cast(list[Event], data.get("fdwic_events") or [])
    if not events:
        print(f"{path}\nevents 0")
        return 0

    start_ns = min(_field_int(event, 6) for event in events)
    end_ns = max(_field_int(event, 7) for event in events)
    global_span_ns = end_ns - start_ns

    by_core: dict[int, list[Event]] = defaultdict(list)
    for event in events:
        by_core[_field_int(event, 0)].append(event)

    core_stats: list[tuple[int, int, int, int, int]] = []
    for core, core_events in by_core.items():
        core_start = min(_field_int(event, 6) for event in core_events)
        core_end = max(_field_int(event, 7) for event in core_events)
        core_stats.append((core_end - core_start, core, len(core_events), core_start, core_end))
    core_stats.sort(reverse=True)

    max_core_span_ns, top_core, top_count, _top_start, _top_end = core_stats[0]
    print(path)
    print(
        "events "
        f"{len(events)} global_span_us {global_span_ns / 1000:.3f} "
        f"max_core_span_us {max_core_span_ns / 1000:.3f} "
        f"top_core {top_core} top_core_events {top_count}"
    )
    print("\nTop core spans:")
    print(f"{'core':>4s} {'events':>8s} {'span_us':>12s} {'offset_start_us':>15s} {'offset_end_us':>13s}")
    for span_ns, core, count, core_start, core_end in core_stats[: args.top]:
        print(
            f"{core:4d} {count:8d} {span_ns / 1000:12.3f} "
            f"{(core_start - start_ns) / 1000:15.3f} {(core_end - start_ns) / 1000:13.3f}"
        )

    print()
    print_phase_table(by_core[top_core], max_core_span_ns, f"Top core {top_core} phase totals:")
    print()
    print_submit_exclusive_table(by_core[top_core], max_core_span_ns, f"Top core {top_core} submit exclusive:")
    print()
    print_phase_table(events, global_span_ns, "All-core phase totals:")
    print()
    print_submit_exclusive_table(events, global_span_ns, "All-core submit exclusive:")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
