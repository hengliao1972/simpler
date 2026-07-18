# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Atomic-event coverage for the shared FDWIC swimlane converter."""

import json

import pytest

from simpler_setup.tools.swimlane_converter import generate_chrome_trace_json, read_perf_data


def _capture(
    rows,
    *,
    trace_schema_version=None,
    num_cores=1,
    add_clock_baselines=True,
    clock_dependency_applied=True,
):
    rows = list(rows)
    if trace_schema_version == 3 and add_clock_baselines:
        dependency_flags = 0x3 if clock_dependency_applied else 0x1
        for core_id in range(num_cores):
            block_id, lane = divmod(core_id, 3)
            start = 10 + 4 * core_id
            rows.extend(
                [
                    [core_id, block_id, lane, -1, -1, "ClockBaseline", start, start + 1, 0, 0],
                    [core_id, block_id, lane, -1, -1, "ClockBaseline", start + 2, start + 3, dependency_flags, 0],
                ]
            )
    metadata = {
        "clock_freq_hz": 1_000_000_000,
        "num_cores": num_cores,
        "core_types": ["aic" if core_id % 3 == 0 else "aiv" for core_id in range(num_cores)],
    }
    if trace_schema_version is not None:
        metadata["trace_schema_version"] = trace_schema_version
    if trace_schema_version == 3:
        atomic_rows = [row for row in rows if row[5] == "Atomic"]
        batch_rows = [row for row in atomic_rows if int(row[8]) & (1 << 7)]
        batch_calls = sum((int(row[8]) >> 8) & 0xFFFFFF for row in batch_rows)
        metadata["fdwic_summary"] = {
            "records": len(rows),
            "atomic_records": len(atomic_rows),
            "clock_baseline_records": sum(row[5] == "ClockBaseline" for row in rows),
            "atomic_calls": len(atomic_rows) - len(batch_rows) + batch_calls,
            "batched_poll_calls": batch_calls,
            "poll_batch_records": len(batch_rows),
            "dropped_records": 0,
        }
    return {
        "l2_swimlane_level": 4 if trace_schema_version == 3 else 1,
        "metadata": metadata,
        "aicore_tasks": [],
        "aicpu_tasks": [],
        "aicpu_scheduler_phases": [],
        "aicpu_orchestrator_phases": [],
        "fdwic_events": rows,
    }


def _convert(tmp_path, capture, *, pass_metadata=True):
    raw_path = tmp_path / "l2_swimlane_records.json"
    merged_path = tmp_path / "merged_swimlane.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")
    data = read_perf_data(raw_path)
    kwargs = {"fdwic_events": data.get("fdwic_events")}
    if pass_metadata:
        kwargs.update(
            trace_schema_version=data["trace_schema_version"],
            clock_freq_hz=data["clock_freq_hz"],
        )
    generate_chrome_trace_json(data["tasks"], merged_path, **kwargs)
    return data, json.loads(merged_path.read_text(encoding="utf-8"))["traceEvents"]


def test_atomic_and_clock_stay_on_scalar_lane_and_preserve_atomic_count(tmp_path):
    rows = [
        [0, 0, 0, 7, -1, "Claim", 100, 200, 0x2, 0],
        [0, 0, 0, 7, -1, "Atomic", 120, 160, 0x53, 15],
        [0, 0, 0, 7, -1, "Atomic", 161, 170, 0x54, 25],
        [1, 0, 1, -1, -1, "ClockBaseline", 101, 102, 0x3, 0],
        [1, 0, 1, 7, 0, "Kernel", 140, 180, 0, 0],
    ]
    _, events = _convert(tmp_path, _capture(rows, trace_schema_version=2, num_cores=2))

    thread_names = {
        event["tid"]: event["args"]["name"]
        for event in events
        if event.get("ph") == "M" and event.get("name") == "thread_name"
    }
    assert thread_names[0] == "AIC (core0)"
    assert thread_names[1] == "AIV0 (core1)"
    assert thread_names[4] == "AIV0·kernel (core1)"
    assert not any("·atomic" in name for name in thread_names.values())

    atomic_events = [event for event in events if event.get("args", {}).get("phase") == "atomic"]
    clock = next(event for event in events if event.get("cat") == "scalar_clock")
    kernel = next(event for event in events if event.get("name") == "f0#7")
    assert len(atomic_events) == sum(row[5] == "Atomic" for row in rows)
    assert all(event["tid"] == 0 for event in atomic_events)
    assert clock["tid"] == 1
    assert clock["args"]["ticks"] == 1
    assert clock["args"]["clock_freq_hz"] == 1_000_000_000
    assert kernel["tid"] == 4

    fetch_sub = next(event for event in atomic_events if event["args"]["op"] == "fetch_sub")
    assert fetch_sub["name"] == "atomic.return_ready.won_remaining_fetch_sub.fetch_sub#7"
    assert fetch_sub["cat"] == "atomic.return_ready"
    assert fetch_sub["args"]["site_id"] == 25
    assert fetch_sub["args"]["op_id"] == 4
    assert fetch_sub["args"]["cycles"] == 9
    assert fetch_sub["args"]["call_count"] == 1
    assert isinstance(fetch_sub["args"]["cycles"], int)
    assert fetch_sub["args"]["execution_unit"] == "scalar"


@pytest.mark.parametrize(
    ("site_id", "site_name", "op_id", "op_name"),
    [
        (1, "startup_poll", 0, "load"),
        (2, "fatal_poll", 0, "load"),
        (5, "fanin_flag_load", 0, "load"),
        (11, "heap_frontier_load", 0, "load"),
        (12, "heap_vend_load", 0, "load"),
        (14, "replay_done_poll", 0, "load"),
        (21, "won_any_load", 0, "load"),
        (22, "won_state_load", 0, "load"),
        (23, "won_lane_claim_exchange", 1, "exchange"),
        (27, "won_drained_load", 0, "load"),
    ],
)
def test_v3_poll_batch_preserves_exact_call_count_without_fake_atomic_latency(
    tmp_path, site_id, site_name, op_id, op_name
):
    poll_count = 12345
    flags = (poll_count << 8) | 0x80 | 0x10 | op_id
    rows = [[0, 0, 0, -1, -1, "Atomic", 100, 900, flags, site_id]]

    data, events = _convert(tmp_path, _capture(rows, trace_schema_version=3))

    assert data["trace_schema_version"] == 3
    batch = next(event for event in events if event.get("cat") == "atomic.poll_batch")
    assert batch["name"] == f"atomic.poll_batch.{site_name}.{op_name}×{poll_count}"
    assert batch["tid"] == 0
    assert batch["args"]["call_count"] == poll_count
    assert batch["args"]["phase"] == "atomic_poll_batch"
    assert batch["args"]["is_poll_batch"] is True
    expected_semantics = "idempotent_failed_exchange_retries" if site_id == 23 else "observation_load_calls"
    assert batch["args"]["batch_semantics"] == expected_semantics
    assert batch["args"]["duration_semantics"] == "logical_poll_episode_envelope_not_single_atomic_latency"
    assert batch["args"]["may_contain_interleaved_direct_atomics"] is True
    assert batch["args"]["poll_window_cycles"] == 800
    assert batch["args"]["estimate_formula"] == "call_count * calibrated_atomic_cost"
    assert "cycles" not in batch["args"]
    assert "completion_boundary" not in batch["args"]
    assert "return_ready_observed" not in batch["args"]


def test_v3_poll_batch_accepts_maximum_24_bit_call_count(tmp_path):
    poll_count = 0xFFFFFF
    flags = (poll_count << 8) | 0x90
    data, events = _convert(
        tmp_path,
        _capture([[0, 0, 0, -1, -1, "Atomic", 100, 900, flags, 1]], trace_schema_version=3),
    )

    batch = next(event for event in events if event.get("cat") == "atomic.poll_batch")
    assert batch["args"]["call_count"] == poll_count
    assert data["fdwic_summary"]["atomic_calls"] == poll_count


@pytest.mark.parametrize(
    ("trace_schema_version", "flags", "site"),
    [
        (2, (7 << 8) | 0x90, 5),  # schema v2 reserves bit 7
        (3, 0x90, 5),  # zero call_count
        (3, (7 << 8) | 0x91, 5),  # Exchange is not a batchable observation load
        (3, (7 << 8) | 0x90, 9),  # frontier scans are not explicit wait-region polling
        (3, (7 << 8) | 0x90, 23),  # failed lane-claim retries must retain their Exchange op
        (3, (7 << 8) | 0x93, 15),  # WonSlot FetchMax is protocol-changing, not a retry batch
        (3, (7 << 8) | 0xB0, 5),  # batch has no single-load value_zero meaning
        (3, (7 << 8) | 0xD0, 5),  # batch has no return-ready boundary
    ],
)
def test_poll_batch_rejects_invalid_schema_or_flags(tmp_path, trace_schema_version, flags, site):
    capture = _capture(
        [[0, 0, 0, -1, -1, "Atomic", 100, 110, flags, site]],
        trace_schema_version=trace_schema_version,
    )
    raw_path = tmp_path / "l2_swimlane_records.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid Atomic PollBatch"):
        read_perf_data(raw_path)


@pytest.mark.parametrize(
    ("flags", "site", "func_id"),
    [
        (0x51, 4, -1),  # ClaimMax is FetchMax, not Exchange.
        (0x12, 0, -1),  # StartupIncrement does not consume its FetchAdd result.
        (0x42, 0, -1),  # return_ready cannot exist without a consumed result.
        (0x73, 4, -1),  # value_zero is defined only for Load.
        ((1 << 8) | 0x51, 23, -1),  # retry payload is defined only for FetchMax.
        (0x50, 28, -1),  # Unknown site.
        (0x53, 4, 0),  # Atomic records never carry a func id.
    ],
)
def test_v3_rejects_invalid_direct_atomic_schema(tmp_path, flags, site, func_id):
    capture = _capture(
        [[0, 0, 0, 7, func_id, "Atomic", 100, 110, flags, site]],
        trace_schema_version=3,
    )
    raw_path = tmp_path / "l2_swimlane_records.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid direct Atomic"):
        read_perf_data(raw_path)


@pytest.mark.parametrize(
    ("task_id", "func_id", "flags", "aux"),
    [
        (-1, -1, 0x2, 0),  # applied requires the dependency bit.
        (-1, -1, 0x4, 0),  # Unknown flag bit.
        (0, -1, 0, 0),
        (-1, 0, 0, 0),
        (-1, -1, 0, 99),
    ],
)
def test_v3_rejects_invalid_clock_baseline_schema(tmp_path, task_id, func_id, flags, aux):
    capture = _capture(
        [[0, 0, 0, task_id, func_id, "ClockBaseline", 100, 110, flags, aux]],
        trace_schema_version=3,
        add_clock_baselines=False,
    )
    raw_path = tmp_path / "l2_swimlane_records.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid ClockBaseline"):
        read_perf_data(raw_path)


@pytest.mark.parametrize(("start", "end"), [(-1, 10), (11, 10), (0, 1 << 64)])
def test_v3_rejects_invalid_cycle_range(tmp_path, start, end):
    capture = _capture(
        [[0, 0, 0, 7, -1, "Atomic", start, end, 0x53, 4]],
        trace_schema_version=3,
    )
    raw_path = tmp_path / "l2_swimlane_records.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid cycle range"):
        read_perf_data(raw_path)


@pytest.mark.parametrize("flags", [-1, 1 << 32])
def test_v3_rejects_non_uint32_flags(tmp_path, flags):
    capture = _capture(
        [[0, 0, 0, 7, -1, "Atomic", 100, 110, flags, 4]],
        trace_schema_version=3,
    )
    raw_path = tmp_path / "l2_swimlane_records.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid uint32 flags"):
        read_perf_data(raw_path)


@pytest.mark.parametrize(
    ("clock_dependency_applied", "direct_flags"),
    [
        (True, 0x53),  # Real A5: consumed result has a return-ready boundary.
        (False, 0x13),  # A5Sim: source bracket only.
    ],
)
def test_v3_direct_return_boundary_matches_per_core_clock_baseline(
    tmp_path, clock_dependency_applied, direct_flags
):
    capture = _capture(
        [[0, 0, 0, 7, -1, "Atomic", 100, 110, direct_flags, 4]],
        trace_schema_version=3,
        clock_dependency_applied=clock_dependency_applied,
    )
    data, _ = _convert(tmp_path, capture)
    assert data["fdwic_summary"]["clock_baseline_records"] == 2


def test_v3_rejects_direct_return_boundary_that_disagrees_with_clock_baseline(tmp_path):
    capture = _capture(
        [[0, 0, 0, 7, -1, "Atomic", 100, 110, 0x13, 4]],
        trace_schema_version=3,
        clock_dependency_applied=True,
    )
    raw_path = tmp_path / "l2_swimlane_records.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match.*ClockBaseline"):
        read_perf_data(raw_path)


def test_v3_requires_two_clock_baselines_per_core(tmp_path):
    capture = _capture([], trace_schema_version=3, num_cores=2, add_clock_baselines=False)
    capture["fdwic_events"] = [[0, 0, 0, -1, -1, "ClockBaseline", 10, 11, 0, 0]]
    capture["metadata"]["fdwic_summary"]["records"] = 1
    capture["metadata"]["fdwic_summary"]["clock_baseline_records"] = 1
    raw_path = tmp_path / "l2_swimlane_records.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")

    with pytest.raises(ValueError, match="requires exactly one plain and one dependency ClockBaseline"):
        read_perf_data(raw_path)


@pytest.mark.parametrize(
    "summary_key",
    [
        "records",
        "atomic_records",
        "clock_baseline_records",
        "atomic_calls",
        "batched_poll_calls",
        "poll_batch_records",
        "dropped_records",
    ],
)
def test_v3_rejects_any_broken_weighted_summary_field(tmp_path, summary_key):
    rows = [
        [0, 0, 0, -1, -1, "Atomic", 100, 200, (17 << 8) | 0x90, 1],
        [0, 0, 0, 4, -1, "Atomic", 210, 220, 0x53, 4],
    ]
    capture = _capture(rows, trace_schema_version=3)
    capture["metadata"]["fdwic_summary"][summary_key] += 1
    raw_path = tmp_path / "l2_swimlane_records.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"fdwic_summary\.{summary_key}"):
        read_perf_data(raw_path)


def test_schema_v3_requires_level4_and_weighted_summary(tmp_path):
    capture = _capture([[0, 0, 0, -1, -1, "Atomic", 100, 110, (3 << 8) | 0x90, 14]], trace_schema_version=3)
    capture["l2_swimlane_level"] = 1
    raw_path = tmp_path / "l2_swimlane_records.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")
    with pytest.raises(ValueError, match="requires l2_swimlane_level=4"):
        read_perf_data(raw_path)

    capture["l2_swimlane_level"] = 4
    del capture["metadata"]["fdwic_summary"]
    raw_path.write_text(json.dumps(capture), encoding="utf-8")
    with pytest.raises(ValueError, match="fdwic_summary is required"):
        read_perf_data(raw_path)


def test_v2_claim_flags_encode_all_three_states(tmp_path):
    rows = [
        [0, 0, 0, 1, -1, "Claim", 100, 110, 0x0, 0],
        [0, 0, 0, 2, -1, "Claim", 120, 140, 0x2, 0],
        [0, 0, 0, 3, 0, "Claim", 150, 180, 0x3, 1],
    ]
    data, events = _convert(tmp_path, _capture(rows, trace_schema_version=2))

    assert data["trace_schema_version"] == 2
    by_name = {event.get("name"): event for event in events}
    assert by_name["claim.not_attempted#1"]["args"]["claim_attempted"] is False
    assert by_name["claim.lost#2"]["args"]["claim_attempted"] is True
    assert by_name["claim.won#3"]["args"]["claim_won"] is True
    assert all(
        by_name[name]["args"]["claim_attempted_source"] == "raw_flag"
        for name in ("claim.not_attempted#1", "claim.lost#2", "claim.won#3")
    )


@pytest.mark.parametrize("flags", [0x1, 0x4])
def test_v2_rejects_invalid_claim_flags(tmp_path, flags):
    capture = _capture(
        [[0, 0, 0, 1, -1, "Claim", 100, 110, flags, 0]],
        trace_schema_version=2,
    )
    raw_path = tmp_path / "l2_swimlane_records.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid Claim flags"):
        read_perf_data(raw_path)


def test_v1_claim_attempt_requires_contained_claim_max_evidence(tmp_path):
    rows = [
        [0, 0, 0, 1, -1, "Claim", 100, 200, 0, 0],
        [0, 0, 0, 1, -1, "Atomic", 120, 160, 0x53, 4],
        [0, 0, 0, 2, -1, "Claim", 210, 230, 0, 0],
        [0, 0, 0, 2, -1, "Atomic", 231, 240, 0x53, 4],
    ]
    data, events = _convert(tmp_path, _capture(rows), pass_metadata=False)

    assert data["trace_schema_version"] == 1
    by_name = {event.get("name"): event for event in events}
    claim_max = next(event for event in events if event.get("args", {}).get("site_id") == 4)
    assert claim_max["name"] == "atomic.return_ready.claim_max.fetch_max#1"
    assert by_name["claim.lost#1"]["args"]["claim_attempted"] is True
    assert by_name["claim.lost#1"]["args"]["claim_attempted_source"] == "contained_claim_max"
    assert by_name["claim#2"]["args"]["claim_attempted"] is None
    assert by_name["claim#2"]["args"]["claim_attempted_source"] == "unknown_v1_without_matching_claim_max"
