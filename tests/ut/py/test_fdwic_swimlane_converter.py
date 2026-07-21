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

from simpler_setup.tools.fdwic_swimlane_exclusive_analyzer import analyze_data, write_analysis_data
from simpler_setup.tools.fdwic_swimlane_schema import validate_and_partition_v4
from simpler_setup.tools.swimlane_converter import generate_chrome_trace_json, read_perf_data


def _capture(
    rows,
    *,
    trace_schema_version=None,
    num_cores=1,
    add_clock_baselines=True,
    clock_dependency_applied=True,
    level=None,
):
    rows = list(rows)
    if level is None:
        level = 4 if trace_schema_version == 3 else 1
    if trace_schema_version in (3, 4) and level == 4 and add_clock_baselines:
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
    if trace_schema_version in (3, 4):
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
        "l2_swimlane_level": level,
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
    if pass_metadata:
        generate_chrome_trace_json(
            data["tasks"],
            merged_path,
            fdwic_events=data.get("fdwic_events"),
            trace_schema_version=data["trace_schema_version"],
            clock_freq_hz=data["clock_freq_hz"],
            fdwic_num_cores=data.get("num_cores", 0),
            fdwic_core_types=data.get("core_types"),
        )
    else:
        generate_chrome_trace_json(data["tasks"], merged_path, fdwic_events=data.get("fdwic_events"))
    return data, json.loads(merged_path.read_text(encoding="utf-8"))["traceEvents"]


def _v4_rows(num_cores=3):
    rows = []
    for core_id in range(num_cores):
        block_id, lane = divmod(core_id, 3)
        offset = core_id * 1_000
        rows.extend(
            [
                [core_id, block_id, lane, -1, -1, "OrchestrationReplay", 100 + offset, 300 + offset, 0, 0],
                [core_id, block_id, lane, -1, -1, "FinalDrain", 300 + offset, 360 + offset, 0, 0],
                # task 0 is deliberately a kernel (not standalone's task_id % 5 Alloc rule).
                [core_id, block_id, lane, 0, 7, "Submit", 110 + offset, 190 + offset, 1, 0],
                [core_id, block_id, lane, 0, -1, "EfDrain", 110 + offset, 120 + offset, 0, 0],
                [core_id, block_id, lane, 99, 7, "Kernel", 112 + offset, 116 + offset, 0, 0],
                [core_id, block_id, lane, 0, 7, "Claim", 120 + offset, 130 + offset, 3, 0],
                # The callback builds eager arguments without adding a raw phase.
                [core_id, block_id, lane, 0, -1, "Materialize", 135 + offset, 145 + offset, 0, 0],
                [core_id, block_id, lane, 0, -1, "PrepareMap", 145 + offset, 155 + offset, 0, 0],
                [core_id, block_id, lane, 0, 7, "Fanin", 155 + offset, 160 + offset, 0, 2],
                [core_id, block_id, lane, 0, 7, "Register", 160 + offset, 170 + offset, 0, 1],
                [core_id, block_id, lane, 0, 7, "WinnerBuild", 170 + offset, 185 + offset, 0, 0],
                # DrainWon is real in production and remains a nested overlay.
                [core_id, block_id, lane, 0, 7, "DrainWon", 172 + offset, 174 + offset, 1, 1],
                [core_id, block_id, lane, 0, 7, "Kernel", 175 + offset, 180 + offset, 0, 0],
                # task 1 is deliberately Alloc; kind comes from aux, never task_id modulo arithmetic.
                [core_id, block_id, lane, 1, -1, "Submit", 200 + offset, 270 + offset, 0, 1],
                [core_id, block_id, lane, 1, -1, "EfDrain", 200 + offset, 210 + offset, 0, 0],
                [core_id, block_id, lane, 1, -1, "Claim", 210 + offset, 220 + offset, 2, 1],
                # Alloc uses the same Claim-first callback gap as kernel Submit.
                [core_id, block_id, lane, 1, -1, "Materialize", 225 + offset, 235 + offset, 0, 1],
                [core_id, block_id, lane, 1, -1, "PrepareMap", 235 + offset, 240 + offset, 0, 1],
                [core_id, block_id, lane, 1, -1, "Register", 240 + offset, 250 + offset, 0, 0],
                # Tensor-data waits may execute a kernel between Submit calls in production.
                [core_id, block_id, lane, 88, 8, "Kernel", 280 + offset, 290 + offset, 0, 0],
                [core_id, block_id, lane, 77, 9, "Kernel", 310 + offset, 330 + offset, 0, 0],
            ]
        )
    return rows


@pytest.fixture
def v4_business_and_atomic_capture():
    """Combine production business spans with direct and batched-poll atomic overlays."""

    rows = _v4_rows()
    rows.extend(
        [
            # ClaimMax consumes the FetchMax result, so its end is a return-ready boundary.
            [0, 0, 0, 0, -1, "Atomic", 121, 128, 0x53, 4],
            # Fanin polling keeps its exact logical call count without becoming exclusive work.
            [0, 0, 0, -1, -1, "Atomic", 156, 159, (7 << 8) | 0x90, 5],
        ]
    )
    return _capture(rows, trace_schema_version=4, num_cores=3, level=4)


def _v4_single_path_rows(*, is_alloc, is_winner, claim_first=True, num_cores=3):
    """Build one Submit per core for an exact path-order contract test."""

    rows = []
    for core_id in range(num_cores):
        block_id, lane = divmod(core_id, 3)
        offset = core_id * 1_000
        function_id = -1 if is_alloc or not is_winner else 7
        submit_flags = 1 if is_winner else 0
        claim_flags = 0x2 | submit_flags
        rows.extend(
            [
                [core_id, block_id, lane, -1, -1, "OrchestrationReplay", 100 + offset, 300 + offset, 0, 0],
                [core_id, block_id, lane, -1, -1, "FinalDrain", 300 + offset, 340 + offset, 0, 0],
                [
                    core_id,
                    block_id,
                    lane,
                    0,
                    function_id,
                    "Submit",
                    110 + offset,
                    250 + offset,
                    submit_flags,
                    int(is_alloc),
                ],
            ]
        )

        cursor = 110

        def add_phase(phase, *, func_id=-1, flags=0, aux=0):
            nonlocal cursor
            rows.append(
                [
                    core_id,
                    block_id,
                    lane,
                    0,
                    func_id,
                    phase,
                    cursor + offset,
                    cursor + 10 + offset,
                    flags,
                    aux,
                ]
            )
            cursor += 10

        add_phase("EfDrain")
        if claim_first:
            add_phase("Claim", func_id=function_id, flags=claim_flags, aux=int(is_alloc))
            cursor += 5  # Existing Claim/Materialize boundaries expose eager callback work as residual.
            add_phase("Materialize", aux=int(is_alloc))
            add_phase("PrepareMap", aux=int(is_alloc))
        else:
            # The live one-shot API deliberately retains its original order.
            add_phase("Materialize", aux=int(is_alloc))
            add_phase("PrepareMap", aux=int(is_alloc))
            if is_alloc:
                add_phase("Register")
            add_phase("Claim", func_id=function_id, flags=claim_flags, aux=int(is_alloc))

        if not is_alloc:
            if is_winner:
                add_phase("Fanin", func_id=function_id, aux=2)
            add_phase("Register", func_id=function_id, aux=1)
            add_phase("WinnerBuild" if is_winner else "LoserReplay", func_id=function_id)
        elif claim_first:
            add_phase("Register")
            if is_winner:
                add_phase("AllocComplete")
        elif is_winner:
            add_phase("AllocComplete")
    return rows


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
def test_v3_direct_return_boundary_matches_per_core_clock_baseline(tmp_path, clock_dependency_applied, direct_flags):
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


@pytest.mark.parametrize(
    ("is_alloc", "is_winner", "expected_sequence"),
    [
        (
            False,
            True,
            ("EfDrain", "Claim", "Materialize", "PrepareMap", "Fanin", "Register", "WinnerBuild"),
        ),
        (False, False, ("EfDrain", "Claim", "Materialize", "PrepareMap", "Register", "LoserReplay")),
        (True, True, ("EfDrain", "Claim", "Materialize", "PrepareMap", "Register", "AllocComplete")),
        (True, False, ("EfDrain", "Claim", "Materialize", "PrepareMap", "Register")),
    ],
)
def test_v4_accepts_claim_first_submit_paths(tmp_path, is_alloc, is_winner, expected_sequence):
    capture = _capture(
        _v4_single_path_rows(is_alloc=is_alloc, is_winner=is_winner),
        trace_schema_version=4,
        num_cores=3,
    )
    raw_path = tmp_path / f"claim_first_{int(is_alloc)}_{int(is_winner)}.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")
    data = read_perf_data(raw_path)

    model = validate_and_partition_v4(data["fdwic_events"], data["num_cores"], data["core_types"])
    assert all(
        tuple(child.phase for child in core.submits[0].children) == expected_sequence for core in model.cores
    )

    report = analyze_data(data, raw_path)
    internal = report["residual_breakdown"]["submit_internal_residual"]
    assert internal["total_cycles"] == 15
    assert internal["segments"] == [
        {
            "boundary": "Claim->Materialize",
            "event_count": 3,
            "cycles": 15,
            "aic_cycles": 5,
            "aiv_cycles": 10,
        }
    ]


@pytest.mark.parametrize(
    ("is_alloc", "expected_sequence"),
    [
        (
            False,
            ("EfDrain", "Materialize", "PrepareMap", "Claim", "Fanin", "Register", "WinnerBuild"),
        ),
        (True, ("EfDrain", "Materialize", "PrepareMap", "Register", "Claim", "AllocComplete")),
    ],
    ids=["kernel", "alloc"],
)
def test_v4_accepts_live_one_shot_submit_order(tmp_path, is_alloc, expected_sequence):
    capture = _capture(
        _v4_single_path_rows(is_alloc=is_alloc, is_winner=True, claim_first=False),
        trace_schema_version=4,
        num_cores=3,
    )
    raw_path = tmp_path / f"one_shot_order_{int(is_alloc)}.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")
    data = read_perf_data(raw_path)
    model = validate_and_partition_v4(data["fdwic_events"], data["num_cores"], data["core_types"])
    assert all(
        tuple(child.phase for child in core.submits[0].children) == expected_sequence for core in model.cores
    )


def test_v4_production_hierarchy_generates_thin_events_and_exact_residuals(tmp_path):
    data, events = _convert(tmp_path, _capture(_v4_rows(), trace_schema_version=4, num_cores=3))

    assert data["trace_schema_version"] == 4
    duration_events = [event for event in events if event.get("ph") == "X"]
    assert all("args" not in event and "cat" not in event for event in duration_events)
    assert any(event["name"] == "orchestration_replay" for event in duration_events)
    assert any(event["name"] == "final_drain" for event in duration_events)
    assert any(event["name"] == "winner_build#0" for event in duration_events)
    assert any(event["name"] == "drain_won#0" for event in duration_events)
    assert not any(event["name"].startswith(("build#", "replay#", "alloc#")) for event in duration_events)

    residuals = [
        event
        for event in duration_events
        if event["name"] in {"submit_residual", "submit_tail_gap", "between_submit_residual"}
    ]
    assert sum(event["name"] == "submit_residual" for event in residuals) == 6
    assert sum(event["name"] == "submit_tail_gap" for event in residuals) == 6
    assert sum(event["name"] == "between_submit_residual" for event in residuals) == 3


def test_v4_business_and_atomic_overlays_merge_without_changing_exclusive_partition(
    tmp_path, v4_business_and_atomic_capture
):
    data, events = _convert(tmp_path, v4_business_and_atomic_capture)

    claim = next(event for event in events if event.get("name") == "claim.won#0" and event["tid"] == 0)
    direct = next(event for event in events if event.get("name") == "atomic.return_ready.claim_max.fetch_max#0")
    poll = next(event for event in events if event.get("name") == "atomic.poll_batch.fanin_flag_load.load×7")
    assert claim["pid"] == direct["pid"] == poll["pid"] == 0
    assert claim["tid"] == direct["tid"] == poll["tid"] == 0
    assert claim["ts"] <= direct["ts"] < direct["ts"] + direct["dur"] <= claim["ts"] + claim["dur"]
    assert data["fdwic_summary"]["atomic_records"] == 2
    assert data["fdwic_summary"]["atomic_calls"] == 8
    assert data["fdwic_summary"]["poll_batch_records"] == 1

    report = analyze_data(data, tmp_path / "l2_swimlane_records.json")
    baseline_path = tmp_path / "business_only.json"
    baseline_path.write_text(
        json.dumps(_capture(_v4_rows(), trace_schema_version=4, num_cores=3, level=4)),
        encoding="utf-8",
    )
    baseline_report = analyze_data(read_perf_data(baseline_path), baseline_path)

    assert report["validation"]["status"] == "PASS"
    assert report["aggregate_core_work"] == baseline_report["aggregate_core_work"]
    assert report["residual_breakdown"] == baseline_report["residual_breakdown"]
    assert report["overlays"]["Atomic"] == {
        "event_count": 2,
        "aggregate_duration_cycles": 10,
        "included_in_additive_totals": False,
    }
    assert report["aggregate_core_work"]["closure"]["submit_partition"]["exact"] is True


def test_v4_residuals_reuse_the_combined_reader_time_origin(tmp_path):
    capture = _capture(_v4_rows(), trace_schema_version=4, num_cores=3)
    capture["aicore_tasks"] = [[0, 123, 0, 50, 60]]
    raw_path = tmp_path / "l2_swimlane_records.json"
    merged_path = tmp_path / "merged_swimlane.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")
    data = read_perf_data(raw_path)

    generate_chrome_trace_json(
        [],
        merged_path,
        fdwic_events=data["fdwic_events"],
        trace_schema_version=4,
        clock_freq_hz=data["clock_freq_hz"],
        fdwic_num_cores=data["num_cores"],
        fdwic_core_types=data["core_types"],
    )
    events = json.loads(merged_path.read_text(encoding="utf-8"))["traceEvents"]

    between = next(
        event
        for event in events
        if event.get("name") == "between_submit_residual" and event["pid"] == 0 and event["tid"] == 0
    )
    assert between["ts"] == pytest.approx(0.14)
    assert between["dur"] == pytest.approx(0.01)


def test_v4_exclusive_report_closes_production_parents_and_kernel_containment(tmp_path):
    capture = _capture(_v4_rows(), trace_schema_version=4, num_cores=3)
    raw_path = tmp_path / "l2_swimlane_records.json"
    report_path = tmp_path / "swimlane_exclusive_analysis.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")
    data = read_perf_data(raw_path)

    report = analyze_data(data, raw_path)
    written = write_analysis_data(data, raw_path, report_path)

    assert written == report_path
    assert json.loads(report_path.read_text(encoding="utf-8"))["validation"]["status"] == "PASS"
    assert report["capture"]["task_count_per_core"] == 2
    assert report["aggregate_core_work"]["closure"]["submit_partition"]["exact"] is True
    assert report["aggregate_core_work"]["closure"]["orchestration_replay"]["exact"] is True
    assert report["aggregate_core_work"]["closure"]["final_drain"]["exact"] is True
    internal = report["residual_breakdown"]["submit_internal_residual"]
    assert internal["total_cycles"] == 30
    assert internal["segments"] == [
        {
            "boundary": "Claim->Materialize",
            "event_count": 6,
            "cycles": 30,
            "aic_cycles": 10,
            "aiv_cycles": 20,
        }
    ]
    assert report["kernel_containment"]["inside_efdrain_events"] == 3
    assert report["kernel_containment"]["inside_winner_build_events"] == 3
    assert report["kernel_containment"]["inside_orchestration_residual_events"] == 3
    assert report["kernel_containment"]["inside_final_drain_events"] == 3
    assert report["overlays"]["DrainWon"]["event_count"] == 3
    assert report["overlays"]["DrainWon"]["included_in_additive_totals"] is False


def test_v4_reader_rejects_kernel_crossing_exclusive_boundary(tmp_path):
    rows = _v4_rows()
    kernel = next(row for row in rows if row[0] == 0 and row[3] == 99 and row[5] == "Kernel")
    kernel[6:8] = [115, 125]
    raw_path = tmp_path / "l2_swimlane_records.json"
    raw_path.write_text(
        json.dumps(_capture(rows, trace_schema_version=4, num_cores=3)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Kernel crosses exclusive child"):
        read_perf_data(raw_path)


def test_v4_reader_rejects_kernel_inside_submit_residual(tmp_path):
    rows = _v4_rows()
    kernel = next(row for row in rows if row[0] == 0 and row[3] == 99 and row[5] == "Kernel")
    kernel[6:8] = [186, 189]
    raw_path = tmp_path / "l2_swimlane_records.json"
    raw_path.write_text(
        json.dumps(_capture(rows, trace_schema_version=4, num_cores=3)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Kernel is inside Submit residual"):
        read_perf_data(raw_path)


def test_v4_supports_all_existing_fdwic_collection_levels(tmp_path):
    for level in (1, 2, 3):
        capture = _capture(_v4_rows(), trace_schema_version=4, num_cores=3, level=level)
        raw_path = tmp_path / f"l2_swimlane_records_level{level}.json"
        raw_path.write_text(json.dumps(capture), encoding="utf-8")
        assert read_perf_data(raw_path)["l2_swimlane_level"] == level


def test_v4_uses_dynamic_two_block_core_topology(tmp_path):
    capture = _capture(_v4_rows(num_cores=6), trace_schema_version=4, num_cores=6)
    raw_path = tmp_path / "l2_swimlane_records.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")

    report = analyze_data(read_perf_data(raw_path), raw_path)

    assert report["capture"]["core_count"] == 6
    assert report["per_role_core_statistics"]["aic"]["core_count"] == 2
    assert report["per_role_core_statistics"]["aiv"]["core_count"] == 4


@pytest.mark.parametrize("legacy_phase", ["Alloc", "Build", "Replay"])
def test_v4_rejects_legacy_overlapping_lap_phases(tmp_path, legacy_phase):
    rows = _v4_rows()
    rows.append([0, 0, 0, 0, 7, legacy_phase, 170, 185, 0, 0])
    capture = _capture(rows, trace_schema_version=4, num_cores=3)
    raw_path = tmp_path / "l2_swimlane_records.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")

    with pytest.raises(ValueError, match="forbids legacy lap phase"):
        read_perf_data(raw_path)


def test_v4_rejects_missing_parent_wrong_tail_and_overlapping_children(tmp_path):
    mutations = []

    missing_parent = [row for row in _v4_rows() if not (row[0] == 1 and row[5] == "FinalDrain")]
    mutations.append((missing_parent, "exactly one OrchestrationReplay and FinalDrain"))

    wrong_tail = _v4_rows()
    next(row for row in wrong_tail if row[0] == 0 and row[5] == "WinnerBuild")[5] = "AllocComplete"
    mutations.append((wrong_tail, "invalid exclusive sequence"))

    overlapping = _v4_rows()
    next(row for row in overlapping if row[0] == 0 and row[5] == "Materialize")[6] = 129
    mutations.append((overlapping, "overlapping exclusive children"))

    for index, (rows, message) in enumerate(mutations):
        capture = _capture(rows, trace_schema_version=4, num_cores=3)
        raw_path = tmp_path / f"invalid_{index}.json"
        raw_path.write_text(json.dumps(capture), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            read_perf_data(raw_path)


def test_v4_requires_zero_drop_producer_summary_even_without_atomic_rows(tmp_path):
    capture = _capture(_v4_rows(), trace_schema_version=4, num_cores=3)
    capture["metadata"]["fdwic_summary"]["dropped_records"] = 1
    raw_path = tmp_path / "l2_swimlane_records.json"
    raw_path.write_text(json.dumps(capture), encoding="utf-8")

    with pytest.raises(ValueError, match=r"fdwic_summary\.dropped_records"):
        read_perf_data(raw_path)
