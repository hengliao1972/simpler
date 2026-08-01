# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json

import pytest

from simpler_setup.tools.fdwic_shared_swimlane_schema import (
    SHARED_V5_ATOMIC_OP_NAMES,
    SHARED_V5_ATOMIC_SITE_NAMES,
    SHARED_V5_DCCI_OP_NAMES,
    SHARED_V5_DCCI_SITE_NAMES,
    SHARED_V5_DCCI_SITE_OP_IDS,
    SHARED_V5_PHASE1_TASK_COUNT,
    _claim_attempted,
)
from simpler_setup.tools.fdwic_swimlane_exclusive_analyzer import analyze_data
from simpler_setup.tools.swimlane_converter import generate_chrome_trace_json, read_perf_data

_SYNTHETIC_AIC_CORES = 4
_SYNTHETIC_AIV_CORES = 2 * _SYNTHETIC_AIC_CORES
_SYNTHETIC_CORE_COUNT = _SYNTHETIC_AIC_CORES + _SYNTHETIC_AIV_CORES


def _append_row(rows, core, lane, task, func, phase, start, end, flags=0, aux=0):
    block = core if lane == 0 else (core - _SYNTHETIC_AIC_CORES) // 2
    rows.append([core, block, lane, task, func, phase, start, end, flags, aux])


def _dcci_flags(site, lines):
    return SHARED_V5_DCCI_SITE_OP_IDS[site] | (1 << 2) | (1 << 3) | (lines << 8)


def _refresh_summary(raw):
    rows = raw["fdwic_events"]
    atomic_rows = [row for row in rows if row[5] == "Atomic"]
    dcci_rows = [row for row in rows if row[5] == "Dcci"]
    poll_rows = [row for row in atomic_rows if row[8] & (1 << 7)]
    raw["metadata"]["fdwic_summary"] = {
        "records": len(rows),
        "atomic_records": len(atomic_rows),
        "clock_baseline_records": sum(row[5] == "ClockBaseline" for row in rows),
        "atomic_calls": sum(
            (row[8] >> 8) & 0xFFFFFF if row[8] & (1 << 7) else 1
            for row in atomic_rows
        ),
        "batched_poll_calls": sum((row[8] >> 8) & 0xFFFFFF for row in poll_rows),
        "poll_batch_records": len(poll_rows),
        "dcci_records": len(dcci_rows),
        "dcci_calls": sum((row[8] >> 3) & 0xF for row in dcci_rows),
        "dcci_lines": sum(row[8] >> 8 for row in dcci_rows),
        "dropped_records": 0,
    }


def _write_mutated_capture(path, raw):
    _refresh_summary(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")


def _shared_capture(level=1):  # noqa: PLR0912
    rows = []
    core_types = ["aic"] * _SYNTHETIC_AIC_CORES + ["aiv"] * _SYNTHETIC_AIV_CORES
    lanes = [0] * _SYNTHETIC_AIC_CORES + [
        1 + ordinal % 2 for ordinal in range(_SYNTHETIC_AIV_CORES)
    ]
    for core, (role, lane) in enumerate(zip(core_types, lanes)):
        orchestration_start = 1_000_000 + core * 2_000_000
        if level == 4:
            _append_row(rows, core, lane, -1, -1, "Dcci", orchestration_start - 40, orchestration_start - 30, 268, 9)
            _append_row(rows, core, lane, -1, -1, "ClockBaseline", orchestration_start - 20, orchestration_start - 19)
            _append_row(
                rows,
                core,
                lane,
                -1,
                -1,
                "ClockBaseline",
                orchestration_start - 18,
                orchestration_start - 17,
                1,
            )
        last_submit_end = 0
        for task in range(SHARED_V5_PHASE1_TASK_COUNT):
            kind = task % 5
            task_func = -1 if kind == 0 else kind - 1
            attempted = kind == 0 or (
                (role == "aic" and kind in (1, 3))
                or (role == "aiv" and kind in (2, 4))
            )
            root_contender = (
                (kind == 0 and core < 8)
                or (kind in (1, 3) and core < min(6, _SYNTHETIC_AIC_CORES))
                or (
                    kind in (2, 4)
                    and _SYNTHETIC_AIC_CORES <= core < _SYNTHETIC_AIC_CORES + min(8, _SYNTHETIC_AIV_CORES)
                )
            )
            winner_core = {
                0: 0,
                1: 0,
                2: _SYNTHETIC_AIC_CORES,
                3: 0,
                4: _SYNTHETIC_AIC_CORES,
            }[kind]
            winner = core == winner_core
            func = task_func if winner else -1
            submit_start = orchestration_start + 10 + task * 400
            claim_start = submit_start + 10
            claim_end = submit_start + 20
            submit_end = submit_start + 300
            claim_flags = (1 if winner else 0) | (2 if attempted else 0)
            _append_row(rows, core, lane, task, func, "Claim", claim_start, claim_end, claim_flags, kind == 0)
            if level == 4 and attempted:
                _append_row(
                    rows,
                    core,
                    lane,
                    task,
                    -1,
                    "Atomic",
                    claim_start + 2,
                    claim_start + 4,
                    4 | (1 << 4),
                    40,
                )
            if level == 4 and root_contender:
                _append_row(
                    rows,
                    core,
                    lane,
                    task,
                    -1,
                    "Atomic",
                    claim_start + 5,
                    claim_end - 2,
                    4 | (1 << 4),
                    41,
                )
            if winner:
                _append_row(rows, core, lane, task, func, "Materialize", submit_start + 30, submit_start + 100)
                _append_row(
                    rows,
                    core,
                    lane,
                    task,
                    func,
                    "SharedMaterializePublishTaskOutputs",
                    submit_start + 40,
                    submit_start + 70,
                )
                _append_row(
                    rows,
                    core,
                    lane,
                    task,
                    func,
                    "SharedMaterializePublishTaskOutputsCopy",
                    submit_start + 40,
                    submit_start + 50,
                )
                _append_row(
                    rows,
                    core,
                    lane,
                    task,
                    func,
                    "SharedMaterializePublishTaskOutputsFlush",
                    submit_start + 50,
                    submit_start + 70,
                )
                _append_row(rows, core, lane, task, func, "Register", submit_start + 100, submit_start + 170)
                _append_row(
                    rows,
                    core,
                    lane,
                    task,
                    func,
                    "SharedRegisterWaitInsertTurnBypassLoad",
                    submit_start + 100,
                    submit_start + 110,
                    0,
                    0 if task == 0 else 1,
                )
                _append_row(
                    rows,
                    core,
                    lane,
                    task,
                    func,
                    "SharedRegisterPublishMetadata",
                    submit_start + 110,
                    submit_start + 140,
                )
                if level == 4:
                    if kind in (0, 1, 2, 3):
                        descriptor_lines = {0: 6, 1: 2, 2: 6, 3: 2}[kind]
                        _append_row(
                            rows,
                            core,
                            lane,
                            task,
                            -1,
                            "Dcci",
                            submit_start + 55,
                            submit_start + 60,
                            _dcci_flags(3, descriptor_lines),
                            3,
                        )
                    elif kind == 4:
                        _append_row(
                            rows,
                            core,
                            lane,
                            task,
                            -1,
                            "Dcci",
                            submit_start + 75,
                            submit_start + 80,
                            _dcci_flags(1, 1),
                            1,
                        )
                    _append_row(
                        rows,
                        core,
                        lane,
                        task,
                        -1,
                        "Atomic",
                        submit_start + 145,
                        submit_start + 150,
                        4 | (1 << 4),
                        20,
                    )
                if kind != 0:
                    _append_row(rows, core, lane, task, func, "Fanin", submit_start + 170, submit_start + 190)
                    _append_row(rows, core, lane, task, func, "WinnerBuild", submit_start + 190, submit_start + 210)
                    _append_row(rows, core, lane, task, func, "Kernel", submit_start + 192, submit_start + 208)
                    if level == 4 and kind in (2, 3):
                        _append_row(
                            rows,
                            core,
                            lane,
                            task,
                            -1,
                            "Dcci",
                            submit_start + 195,
                            submit_start + 200,
                            _dcci_flags(7, 2),
                            7,
                        )
                    elif level == 4 and kind == 4:
                        for offset in (172, 177, 182):
                            _append_row(
                                rows,
                                core,
                                lane,
                                task,
                                -1,
                                "Dcci",
                                submit_start + offset,
                                submit_start + offset + 2,
                                _dcci_flags(0, 1),
                                0,
                            )
                        for offset in (191, 194, 197, 200, 203, 206):
                            _append_row(
                                rows,
                                core,
                                lane,
                                task,
                                -1,
                                "Dcci",
                                submit_start + offset,
                                submit_start + offset + 2,
                                _dcci_flags(7, 2),
                                7,
                            )
                else:
                    _append_row(rows, core, lane, task, -1, "AllocComplete", submit_start + 170, submit_start + 210)
            _append_row(
                rows,
                core,
                lane,
                task,
                func,
                "Submit",
                submit_start,
                submit_end,
                1 if winner else 0,
                kind == 0,
            )
            last_submit_end = submit_end
        orchestration_end = last_submit_end + 10
        _append_row(
            rows,
            core,
            lane,
            -1,
            -1,
            "OrchestrationReplay",
            orchestration_start,
            orchestration_end,
        )
        _append_row(rows, core, lane, -1, -1, "FinalDrain", orchestration_end, orchestration_end + 20)
        if level == 4:
            _append_row(
                rows,
                core,
                lane,
                -1,
                -1,
                "Dcci",
                orchestration_end + 30,
                orchestration_end + 40,
                1 | (1 << 2) | (3 << 3) | (3 << 8),
                8,
            )

    atomic_rows = [row for row in rows if row[5] == "Atomic"]
    dcci_rows = [row for row in rows if row[5] == "Dcci"]
    poll_rows = [row for row in atomic_rows if row[8] & (1 << 7)]
    return {
        "l2_swimlane_level": level,
        "metadata": {
            "clock_freq_hz": 1_000_000_000,
            "num_cores": _SYNTHETIC_CORE_COUNT,
            "trace_schema_version": 5,
            "raw_trace_version": 5,
            "tensormap_mode": "shared",
            "records_per_core": 28416,
            "record_size_bytes": 16,
            "atomic_site_names": list(SHARED_V5_ATOMIC_SITE_NAMES),
            "atomic_op_names": list(SHARED_V5_ATOMIC_OP_NAMES),
            "dcci_site_names": list(SHARED_V5_DCCI_SITE_NAMES),
            "dcci_op_names": list(SHARED_V5_DCCI_OP_NAMES),
            "core_types": core_types,
            "fdwic_summary": {
                "records": len(rows),
                "atomic_records": len(atomic_rows),
                "clock_baseline_records": sum(row[5] == "ClockBaseline" for row in rows),
                "atomic_calls": sum((row[8] >> 8) & 0xFFFFFF if row[8] & (1 << 7) else 1 for row in atomic_rows),
                "batched_poll_calls": sum((row[8] >> 8) & 0xFFFFFF for row in poll_rows),
                "poll_batch_records": len(poll_rows),
                "dcci_records": len(dcci_rows),
                "dcci_calls": sum((row[8] >> 3) & 0xF for row in dcci_rows),
                "dcci_lines": sum(row[8] >> 8 for row in dcci_rows),
                "dropped_records": 0,
            },
        },
        "aicore_tasks": [],
        "aicpu_tasks": [],
        "aicpu_scheduler_phases": [],
        "aicpu_orchestrator_phases": [],
        "fdwic_events": rows,
    }


@pytest.fixture
def shared_level1_raw(tmp_path):
    path = tmp_path / "l2_swimlane_records.json"
    path.write_text(json.dumps(_shared_capture()), encoding="utf-8")
    return path


@pytest.fixture
def shared_level4_raw(tmp_path):
    path = tmp_path / "l2_swimlane_records.json"
    path.write_text(json.dumps(_shared_capture(level=4)), encoding="utf-8")
    return path


def test_shared_v5_level1_converts_and_closes_exclusive_model(shared_level1_raw, tmp_path):
    data = read_perf_data(shared_level1_raw)
    assert data["trace_schema_version"] == 5
    assert data["tensormap_mode"] == "shared"
    assert (
        len(data["fdwic_events"])
        > 2 * _SYNTHETIC_CORE_COUNT * SHARED_V5_PHASE1_TASK_COUNT
    )

    report = analyze_data(data, shared_level1_raw)
    assert report["validation"]["status"] == "PASS"
    assert report["capture"]["trace_schema_version"] == 5
    assert report["capture"]["task_count_per_core"] == SHARED_V5_PHASE1_TASK_COUNT

    merged = tmp_path / "merged_swimlane.json"
    generate_chrome_trace_json(
        data["tasks"],
        merged,
        fdwic_events=data["fdwic_events"],
        trace_schema_version=5,
        clock_freq_hz=data["clock_freq_hz"],
        fdwic_num_cores=data["num_cores"],
        fdwic_core_types=data["core_types"],
    )
    names = {event.get("name") for event in json.loads(merged.read_text(encoding="utf-8"))["traceEvents"]}
    assert "efdrain#0" in names
    assert "materialize.publish_shared_output_descriptors#0" in names
    assert "register.wait_insert_turn.ld_dev×0#0" in names
    assert "register.wait_insert_turn.ld_dev×1#1" in names


def test_shared_v5_claim_attempted_matches_full_alloc_tournament_contract(shared_level1_raw):
    raw = json.loads(shared_level1_raw.read_text(encoding="utf-8"))
    claims = [row for row in raw["fdwic_events"] if row[5] == "Claim"]
    attempted = [row for row in claims if row[8] & (1 << 1)]
    alloc_attempted = [row for row in attempted if row[3] % 5 == 0]

    assert len(claims) == _SYNTHETIC_CORE_COUNT * SHARED_V5_PHASE1_TASK_COUNT
    assert len(attempted) == 9_216
    assert len(alloc_attempted) == 3_072
    assert {row[0] for row in alloc_attempted} == set(range(_SYNTHETIC_CORE_COUNT))

    production_attempts = 0
    for task in range(SHARED_V5_PHASE1_TASK_COUNT):
        production_attempts += sum(
            _claim_attempted("aic", block, task) for block in range(32)
        )
        production_attempts += sum(
            _claim_attempted("aiv", block, task)
            for block in range(32)
            for _lane in range(2)
        )
    assert production_attempts == 73_728


def test_shared_v5_rejects_atomic_name_table_drift(shared_level1_raw):
    raw = json.loads(shared_level1_raw.read_text(encoding="utf-8"))
    raw["metadata"]["atomic_site_names"][20] = "SharedInsertTurnHandoff"
    shared_level1_raw.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="atomic_site_names"):
        read_perf_data(shared_level1_raw)


def test_shared_v5_requires_one_global_winner_per_task(shared_level1_raw):
    raw = json.loads(shared_level1_raw.read_text(encoding="utf-8"))
    winner_phases = {
        "Materialize",
        "SharedMaterializePublishTaskOutputs",
        "SharedMaterializePublishTaskOutputsCopy",
        "SharedMaterializePublishTaskOutputsFlush",
        "Register",
        "SharedRegisterWaitInsertTurnBypassLoad",
        "SharedRegisterPublishMetadata",
        "Fanin",
        "WinnerBuild",
    }
    rows = []
    for row in raw["fdwic_events"]:
        core, _block, _lane, task, _func, phase, *_rest = row
        if core == 0 and task == 1:
            if phase in winner_phases:
                continue
            if phase == "Claim":
                row[4] = -1
                row[8] = 1 << 1
            elif phase == "Submit":
                row[4] = -1
                row[8] = 0
        rows.append(row)
    raw["fdwic_events"] = rows
    raw["metadata"]["fdwic_summary"]["records"] = len(rows)
    shared_level1_raw.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="task 1 requires exactly one global winner, got 0"):
        read_perf_data(shared_level1_raw)


def test_shared_v5_requires_winner_bypass_wait_detail(shared_level1_raw):
    raw = json.loads(shared_level1_raw.read_text(encoding="utf-8"))
    raw["fdwic_events"] = [
        row
        for row in raw["fdwic_events"]
        if not (
            row[0] == 0
            and row[3] == 1
            and row[5] == "SharedRegisterWaitInsertTurnBypassLoad"
        )
    ]
    _write_mutated_capture(shared_level1_raw, raw)

    with pytest.raises(
        ValueError,
        match=r"\(0, 1\) requires 1 bypass-load insert-turn wait detail row\(s\), got 0",
    ):
        read_perf_data(shared_level1_raw)


def test_shared_v5_requires_one_kernel_for_every_nonalloc_task(shared_level1_raw):
    raw = json.loads(shared_level1_raw.read_text(encoding="utf-8"))
    raw["fdwic_events"] = [
        row
        for row in raw["fdwic_events"]
        if not (row[3] == 1 and row[5] == "Kernel")
    ]
    _write_mutated_capture(shared_level1_raw, raw)

    with pytest.raises(ValueError, match=r"task 1 requires exactly one Kernel row, got 0"):
        read_perf_data(shared_level1_raw)


@pytest.mark.parametrize("mutation", ["duplicate", "alloc"])
def test_shared_v5_rejects_extra_or_duplicate_kernel(shared_level1_raw, mutation):
    raw = json.loads(shared_level1_raw.read_text(encoding="utf-8"))
    if mutation == "duplicate":
        kernel = next(
            row
            for row in raw["fdwic_events"]
            if row[3] == 1 and row[5] == "Kernel"
        )
        raw["fdwic_events"].append(list(kernel))
        message = r"task 1 requires exactly one Kernel row, got 2"
    else:
        tail = next(
            row
            for row in raw["fdwic_events"]
            if row[3] == 0 and row[5] == "AllocComplete"
        )
        _append_row(
            raw["fdwic_events"],
            tail[0],
            tail[2],
            0,
            -1,
            "Kernel",
            tail[6] + 1,
            tail[7] - 1,
        )
        message = r"alloc task 0 requires zero Kernel rows, got 1"
    _write_mutated_capture(shared_level1_raw, raw)

    with pytest.raises(ValueError, match=message):
        read_perf_data(shared_level1_raw)


def test_shared_v5_rejects_kernel_on_nonwinner_core(shared_level1_raw):
    raw = json.loads(shared_level1_raw.read_text(encoding="utf-8"))
    kernel = next(
        row
        for row in raw["fdwic_events"]
        if row[3] == 1 and row[5] == "Kernel"
    )
    assert kernel[0] == 0
    kernel[0] = 1
    kernel[1] = 1
    kernel[2] = 0
    _write_mutated_capture(shared_level1_raw, raw)

    with pytest.raises(ValueError, match=r"task 1 Kernel must run on winner core 0, got core 1"):
        read_perf_data(shared_level1_raw)


def test_shared_v5_rejects_kernel_function_drift(shared_level1_raw):
    raw = json.loads(shared_level1_raw.read_text(encoding="utf-8"))
    kernel = next(
        row
        for row in raw["fdwic_events"]
        if row[3] == 1 and row[5] == "Kernel"
    )
    kernel[4] = 1
    _write_mutated_capture(shared_level1_raw, raw)

    with pytest.raises(ValueError, match=r"task 1 Kernel func_id must be 0, got 1"):
        read_perf_data(shared_level1_raw)


def test_shared_v5_level4_validates_insert_turn_bypass_wait_atomic_handoff_and_dcci_closure(shared_level4_raw):
    data = read_perf_data(shared_level4_raw)
    report = analyze_data(data, shared_level4_raw)
    bypass_waits = [
        event
        for event in data["fdwic_events"]
        if event["phase"] == "SharedRegisterWaitInsertTurnBypassLoad"
    ]

    assert report["validation"]["status"] == "PASS"
    assert data["l2_swimlane_level"] == 4
    assert report["overlays"]["Atomic"]["event_count"] > 0
    assert report["overlays"]["Atomic"]["event_count"] == 18_688
    assert report["overlays"]["Dcci"]["event_count"] == 4_120
    assert len(bypass_waits) == SHARED_V5_PHASE1_TASK_COUNT
    assert sum(event["aux"] == 0 for event in bypass_waits) == 1
    assert all(event["aux"] > 0 for event in bypass_waits if event["task_id"] != 0)


def test_shared_v5_level4_rejects_missing_tournament_local(shared_level4_raw):
    raw = json.loads(shared_level4_raw.read_text(encoding="utf-8"))
    raw["fdwic_events"] = [
        row
        for row in raw["fdwic_events"]
        if not (row[0] == 0 and row[3] == 1 and row[5] == "Atomic" and row[9] == 40)
    ]
    _write_mutated_capture(shared_level4_raw, raw)

    with pytest.raises(ValueError, match="tournament local"):
        read_perf_data(shared_level4_raw)


def test_shared_v5_level4_rejects_tournament_local_for_nonattempted_core(shared_level4_raw):
    raw = json.loads(shared_level4_raw.read_text(encoding="utf-8"))
    claim = next(
        row
        for row in raw["fdwic_events"]
        if row[0] == _SYNTHETIC_AIC_CORES and row[3] == 1 and row[5] == "Claim"
    )
    assert not claim[8] & (1 << 1)
    _append_row(
        raw["fdwic_events"],
        claim[0],
        claim[2],
        claim[3],
        -1,
        "Atomic",
        claim[6] + 2,
        claim[7] - 2,
        4 | (1 << 4),
        40,
    )
    _write_mutated_capture(shared_level4_raw, raw)

    with pytest.raises(ValueError, match="tournament local"):
        read_perf_data(shared_level4_raw)


def test_shared_v5_level4_rejects_missing_tournament_root_group(shared_level4_raw):
    raw = json.loads(shared_level4_raw.read_text(encoding="utf-8"))
    raw["fdwic_events"] = [
        row
        for row in raw["fdwic_events"]
        if not (row[0] == 0 and row[3] == 1 and row[5] == "Atomic" and row[9] == 41)
    ]
    _write_mutated_capture(shared_level4_raw, raw)

    with pytest.raises(ValueError, match="tournament root"):
        read_perf_data(shared_level4_raw)


def test_shared_v5_level4_rejects_duplicate_tournament_root_group(shared_level4_raw):
    raw = json.loads(shared_level4_raw.read_text(encoding="utf-8"))
    claim = next(
        row
        for row in raw["fdwic_events"]
        if row[0] == 0 and row[3] == 1 and row[5] == "Claim"
    )
    _append_row(
        raw["fdwic_events"], claim[0], claim[2], claim[3], -1,
        "Atomic", claim[6] + 5, claim[7] - 2, 4 | (1 << 4), 41,
    )
    _write_mutated_capture(shared_level4_raw, raw)

    with pytest.raises(ValueError, match="tournament root count"):
        read_perf_data(shared_level4_raw)


def test_shared_v5_level4_rejects_missing_winner_dcci(shared_level4_raw):
    raw = json.loads(shared_level4_raw.read_text(encoding="utf-8"))
    raw["fdwic_events"] = [
        row
        for row in raw["fdwic_events"]
        if not (row[0] == 0 and row[3] == 0 and row[5] == "Dcci" and row[9] == 3)
    ]
    _write_mutated_capture(shared_level4_raw, raw)

    with pytest.raises(ValueError, match="Dcci matrix"):
        read_perf_data(shared_level4_raw)


def test_shared_v5_level4_rejects_wrong_winner_dcci_lines(shared_level4_raw):
    raw = json.loads(shared_level4_raw.read_text(encoding="utf-8"))
    row = next(
        row
        for row in raw["fdwic_events"]
        if row[0] == _SYNTHETIC_AIC_CORES and row[3] == 2 and row[5] == "Dcci" and row[9] == 3
    )
    row[8] = (row[8] & 0xFF) | (5 << 8)
    _write_mutated_capture(shared_level4_raw, raw)

    with pytest.raises(ValueError, match="Dcci matrix"):
        read_perf_data(shared_level4_raw)


def test_shared_v5_level4_rejects_loser_business_dcci(shared_level4_raw):
    raw = json.loads(shared_level4_raw.read_text(encoding="utf-8"))
    submit = next(
        row
        for row in raw["fdwic_events"]
        if row[0] == 1 and row[3] == 1 and row[5] == "Submit"
    )
    assert not submit[8] & 1
    _append_row(
        raw["fdwic_events"],
        submit[0],
        submit[2],
        submit[3],
        -1,
        "Dcci",
        submit[6] + 50,
        submit[6] + 55,
        _dcci_flags(3, 2),
        3,
    )
    _write_mutated_capture(shared_level4_raw, raw)

    with pytest.raises(ValueError, match=r"loser.*Dcci"):
        read_perf_data(shared_level4_raw)
