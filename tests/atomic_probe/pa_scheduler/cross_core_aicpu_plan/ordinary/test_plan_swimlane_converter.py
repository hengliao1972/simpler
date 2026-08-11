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
CLOCK_OFFSET = 100


def _aicpu_timing(
    owner_begin_ns: int,
    owner_end_ns: int,
    *,
    minimum_fdwic_start_tick: int,
    maximum_fdwic_end_tick: int,
    offset: int = CLOCK_OFFSET,
    half_width: int = 1,
) -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
]:
    if half_width < 1:
        raise ValueError("test correlation half-width must be positive")
    samples: list[dict[str, object]] = []
    round_trip = 2 * half_width + 2
    spacing = round_trip + 20
    last_pre_start = min(
        owner_begin_ns - round_trip - 20,
        minimum_fdwic_start_tick - offset - half_width - 22,
    )
    pre_start = max(1, last_pre_start - 3 * spacing)
    starts = [pre_start + spacing * index for index in range(4)]
    first_post_start = max(
        owner_end_ns + 20,
        maximum_fdwic_end_tick - offset - half_width + 20,
    )
    starts.extend(
        first_post_start + spacing * index for index in range(4)
    )
    for index, start in enumerate(starts):
        # For half_width=1 this is a four-timestamp interval
        # [offset-1, offset+1].  Wider negative fixtures increase the AICPU
        # round trip symmetrically while retaining causal ordering.
        receive_ns = start + 2 * half_width + 2
        aicore_receive = start + offset + half_width
        aicore_send = receive_ns + offset - half_width
        samples.append({
            "aicpu_send_ns": start,
            "aicore_receive_tick": aicore_receive,
            "aicore_send_tick": aicore_send,
            "aicpu_receive_ns": receive_ns,
            "offset_lower_tick": offset - half_width,
            "offset_upper_tick": offset + half_width,
            "round_trip_ns": receive_ns - start,
            "aicore_service_ticks": aicore_send - aicore_receive,
            "round": index // 4,
            "index_in_round": index % 4,
            "round_nonce": 1_111 if index < 4 else 2_222,
        })
    correlation = {
        "status": "valid",
        "version": 2,
        "pre_samples": 4,
        "post_samples": 4,
        "scale_num": 1_000_000_000,
        "scale_den": 1_000_000_000,
        "offset_lower_tick": offset - half_width,
        "offset_upper_tick": offset + half_width,
        "offset_mid_tick": offset,
        "alignment_error_tick": half_width,
        "maximum_round_trip_ns": round_trip,
        "maximum_aicore_service_ticks": 2,
        "samples": samples,
    }
    causal_bracket = {
        "derivation": "pre-post-four-timestamp-samples",
        "aicpu_clock": "aicpu_monotonic_raw_ns",
        "aicpu_pre_receive_max_ns": max(
            int(sample["aicpu_receive_ns"])
            for sample in samples[:4]
        ),
        "aicpu_post_send_min_ns": min(
            int(sample["aicpu_send_ns"])
            for sample in samples[4:]
        ),
        "aicore_clock": "aicore_sys_cnt_tick",
        "aicore_pre_send_max_tick": max(
            int(sample["aicore_send_tick"])
            for sample in samples[:4]
        ),
        "aicore_post_receive_min_tick": min(
            int(sample["aicore_receive_tick"])
            for sample in samples[4:]
        ),
    }
    phases = [{
        "name": "RuntimePlanProducer",
        "clock": "aicpu_monotonic_raw_ns",
        "begin_ns": owner_begin_ns,
        "end_ns": owner_end_ns,
    }]
    return correlation, causal_bracket, phases


def _set_aicpu_timing(
    capture: dict[str, object],
    owner_begin_ns: int,
    owner_end_ns: int,
    *,
    minimum_fdwic_start_tick: int = 1_000,
    maximum_fdwic_end_tick: int = 1_250,
    half_width: int = 1,
) -> None:
    metadata = capture["metadata"]
    assert isinstance(metadata, dict)
    correlation, causal_bracket, phases = _aicpu_timing(
        owner_begin_ns,
        owner_end_ns,
        minimum_fdwic_start_tick=minimum_fdwic_start_tick,
        maximum_fdwic_end_tick=maximum_fdwic_end_tick,
        half_width=half_width,
    )
    metadata["runtime_plan_producer_domain"] = "aicpu"
    metadata["profiling_primary_view"] = (
        "joint_aicpu_aicore_structure"
    )
    metadata["clock_correlation_warmup_before_pipeline"] = True
    metadata["timing_scope"] = "calibrated-structural-capture"
    metadata["performance_representative"] = False
    metadata["aicpu_aicore_clock_correlation"] = correlation
    metadata["aicpu_aicore_causal_capture_bracket"] = causal_bracket
    owner_span = owner_end_ns - owner_begin_ns
    phase_points = [
        owner_begin_ns,
        owner_begin_ns + owner_span // 16,
        owner_begin_ns + owner_span // 8,
        owner_begin_ns + owner_span * 7 // 8,
        owner_begin_ns + owner_span * 15 // 16,
        owner_end_ns,
    ]
    capture["aicpu_scheduler_phases"] = [
        {
            "name": name,
            "clock": "aicpu_monotonic_raw_ns",
            "begin_ns": phase_points[index],
            "end_ns": phase_points[index + 1],
        }
        for index, name in enumerate((
            "OwnerSetup",
            "BackendBind",
            "Orchestration",
            "BackendClose",
            "OwnerFinalize",
        ))
    ]
    orchestration_begin = phase_points[2]
    orchestration_end = phase_points[3]
    task_window = (orchestration_end - orchestration_begin) // 5
    tasks: list[dict[str, object]] = []
    task_identities = (
        ("Alloc", -1, "metadata", 3),
        ("QK", 0, "aic", 1),
        ("SF", 1, "aiv", 3),
        ("PV", 2, "aic", 1),
        ("UP", 3, "aiv", 0),
    )
    previous_publish_end = orchestration_begin
    for task_id, (kind, function_id, engine, output_count) in enumerate(
        task_identities
    ):
        build_begin = previous_publish_end + 1
        begin_end = build_begin + max(1, task_window // 8)
        finish_begin = begin_end + max(1, task_window // 3)
        finish_end = finish_begin + max(1, task_window // 4)
        if task_id + 1 == len(task_identities):
            publish_begin = phase_points[3] + 1
        else:
            publish_begin = finish_end + max(1, task_window // 8)
        publish_end = publish_begin + max(1, task_window // 8)
        tasks.append({
            "task_id": task_id,
            "task_kind": kind,
            "function_id": function_id,
            "engine": engine,
            "group": 0,
            "output_count": output_count,
            "payload_lines": 4 + task_id,
            "clock": "aicpu_monotonic_raw_ns",
            "build_begin_ns": build_begin,
            "begin_end_ns": begin_end,
            "finish_begin_ns": finish_begin,
            "finish_end_ns": finish_end,
            "publish_begin_ns": publish_begin,
            "publish_end_ns": publish_end,
        })
        previous_publish_end = publish_end
    capture["aicpu_tasks"] = tasks
    capture["aicpu_orchestrator_phases"] = phases


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

    capture: dict[str, object] = {
        "l2_swimlane_level": 1,
        "metadata": {
            "clock_freq_hz": 1_000_000_000,
            "num_cores": CORE_COUNT,
            "trace_schema_version": 5,
            "tensormap_mode": "shared",
            "submit_topology": "aicpu_plan_central_build",
            "runtime_plan_abi": 3,
            "runtime_plan_build_backend": "scalar",
            "runtime_plan_build_workers": CORE_COUNT,
            "runtime_plan_execute_workers": CORE_COUNT,
            "runtime_plan_build_trace_coverage": "scalar-task-detail",
            "runtime_plan_producer_task_count": 5,
            "runtime_plan_task_count": 5,
            "runtime_plan_task_kinds": [0, 1, 2, 3, 4],
            "runtime_plan_terminal": {
                "planned_frontier": 5,
                "closed_task_count": 5,
                "build_next": 5 + CORE_COUNT,
                "build_workers_done": CORE_COUNT,
                "build_release": 5,
                "fatal": 0,
            },
            "pipeline": "plan-ahead-closed",
            "launch_order": "plan-sync-before-aicore",
            "producer_ready": "closed",
            "consumer_admission": "closed-only",
            "prefill": 0,
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
    _set_aicpu_timing(capture, 200, 800)
    return capture


def _simt_runtime_plan_capture() -> dict[str, object]:
    capture = _runtime_plan_capture()
    rows = capture["fdwic_events"]
    metadata = capture["metadata"]
    assert isinstance(rows, list)
    assert isinstance(metadata, dict)
    scalar_build_phases = {
        "Materialize", "Register", "Fanin", "WinnerBuild",
        "AllocComplete", "SharedRegisterPublishMetadata",
        "SharedMaterializePublishTaskOutputs",
        "SharedMaterializePublishTaskOutputsCopy",
        "SharedMaterializePublishTaskOutputsFlush",
    }
    rows[:] = [
        row for row in rows if str(row[5]) not in scalar_build_phases
    ]
    metadata.update({
        "runtime_plan_build_backend": "simt",
        "runtime_plan_build_workers": 4,
        "runtime_plan_execute_workers": CORE_COUNT,
        "runtime_plan_build_trace_coverage":
            "simt-coarse-direct-state",
        "runtime_plan_terminal": {
            "planned_frontier": 5,
            "closed_task_count": 5,
            "build_next": 9,
            "build_workers_done": 4,
            "build_release": 5,
            "fatal": 0,
        },
        "pipeline": "streaming-future",
        "launch_order": "dual-stream-overlap",
        "producer_ready": "prefill",
        "consumer_admission": "ready-future-ticket",
        "prefill": 128,
    })
    summary = metadata["fdwic_summary"]
    assert isinstance(summary, dict)
    summary["records"] = len(rows)
    _set_aicpu_timing(capture, 200, 1_150)
    return capture


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
        self.assertTrue({58, 60, 64, 67}.issubset(MODULE.POLL_BATCH_SITE_OP_IDS))
        self.assertTrue({59, 68}.issubset(MODULE.ATOMIC_RESULT_UNUSED_SITE_IDS))

    def test_pipeline_metadata_is_required_and_cross_checked(self) -> None:
        capture = _runtime_plan_capture()
        metadata = capture["metadata"]
        assert isinstance(metadata, dict)
        metadata.pop("pipeline")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_pipeline.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata.pipeline"):
                MODULE._load_and_validate(path)

        capture = _runtime_plan_capture()
        metadata = capture["metadata"]
        assert isinstance(metadata, dict)
        metadata.update(
            {
                "pipeline": "streaming-future",
                "launch_order": "plan-sync-before-aicore",
                "producer_ready": "prefill",
                "consumer_admission": "ready-future-ticket",
                "prefill": 128,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed_pipeline.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata.launch_order"):
                MODULE._load_and_validate(path)

    def test_real_aicpu_producer_is_a_separate_mapped_perfetto_lane(
        self,
    ) -> None:
        capture = _runtime_plan_capture()
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "aicpu_lane_raw.json"
            merged_path = Path(directory) / "aicpu_lane_merged.json"
            raw_path.write_text(json.dumps(capture), encoding="utf-8")
            _events, _blocks, base_cycle = MODULE.convert(
                raw_path, merged_path
            )
            merged = json.loads(
                merged_path.read_text(encoding="utf-8")
            )

        self.assertEqual(base_cycle, 300)
        producer_events = [
            event
            for event in merged["traceEvents"]
            if event.get("ph") == "X"
            and event.get("name") == "RuntimePlanProducer"
        ]
        self.assertEqual(len(producer_events), 1)
        producer = producer_events[0]
        self.assertEqual(producer["pid"], MODULE.AICPU_PROCESS_ID)
        self.assertEqual(producer["tid"], MODULE.AICPU_THREAD_ID)
        self.assertEqual(producer["ts"], 0.0)
        self.assertEqual(producer["dur"], 0.6)
        self.assertEqual(
            merged["metadata"]["runtime_plan_producer_domain"],
            "aicpu",
        )
        self.assertIs(
            merged["metadata"][
                "clock_correlation_warmup_before_pipeline"
            ],
            True,
        )
        self.assertEqual(
            merged["metadata"]["timing_scope"],
            "calibrated-structural-capture",
        )
        self.assertIs(
            merged["metadata"]["performance_representative"],
            False,
        )
        self.assertNotIn(
            "runtime_plan_pipeline_clock_bracket", merged["metadata"]
        )
        self.assertNotIn("host_pipeline_e2e", merged["metadata"])
        self.assertNotIn("timing_primary_metric", merged["metadata"])
        self.assertNotIn("aicore_time_scope", merged["metadata"])
        self.assertEqual(
            merged["metadata"]["profiling_primary_view"],
            "joint_aicpu_aicore_structure",
        )
        self.assertIn(
            "aicpu_aicore_causal_capture_bracket",
            merged["metadata"],
        )
        samples = merged["metadata"][
            "aicpu_aicore_clock_correlation"
        ]["samples"]
        self.assertEqual(len(samples), 8)
        self.assertEqual(
            {sample["round_nonce"] for sample in samples[:4]},
            {1_111},
        )
        self.assertEqual(
            {sample["round_nonce"] for sample in samples[4:]},
            {2_222},
        )
        aicpu_task_events = [
            event
            for event in merged["traceEvents"]
            if event.get("ph") == "X"
            and event.get("pid") == MODULE.AICPU_PROCESS_ID
            and str(event.get("name", "")).startswith("task_plan.")
        ]
        self.assertEqual(len(aicpu_task_events), 30)
        self.assertTrue(all(event["dur"] > 0 for event in aicpu_task_events))
        self.assertEqual(
            {
                event["name"].split("#", maxsplit=1)[0]
                for event in aicpu_task_events
                if event["cat"] == "aicpu.task_plan.detail"
            },
            {
                "task_plan.begin",
                "task_plan.orchestration",
                "task_plan.stage_payload",
                "task_plan.defer_publish",
                "task_plan.publish",
            },
        )
        self.assertEqual(
            {
                event["name"]
                for event in aicpu_task_events
                if event["name"].startswith("task_plan.Alloc")
            },
            {"task_plan.Alloc#0"},
        )
        owner_phase_events = [
            event
            for event in merged["traceEvents"]
            if event.get("ph") == "X"
            and event.get("pid") == MODULE.AICPU_PROCESS_ID
            and event.get("name") in {
                "OwnerSetup", "BackendBind", "Orchestration",
                "BackendClose", "OwnerFinalize",
            }
        ]
        self.assertEqual(len(owner_phase_events), 5)

    def test_aicpu_operation_trace_is_nested_in_taskplan_lane(self) -> None:
        capture = _runtime_plan_capture()
        tasks = capture["aicpu_tasks"]
        assert isinstance(tasks, list) and isinstance(tasks[0], dict)
        publish_begin = int(tasks[0]["publish_begin_ns"])
        phases = capture["aicpu_scheduler_phases"]
        assert isinstance(phases, list) and isinstance(phases[0], dict)
        owner_setup_begin = int(phases[0]["begin_ns"])
        metadata = capture["metadata"]
        assert isinstance(metadata, dict)
        metadata["aicpu_operation_trace"] = {
            "enabled": True,
            "records": 7,
            "record_bytes": 64,
            "dropped": 0,
        }
        operation_specs = (
            (
                -1, "owner_setup", "cache_discard_civac",
                "request_tensors", owner_setup_begin + 1, 2, 2, -1,
            ),
            (
                0, "task_publish", "atomic_load_acquire",
                "fatal", publish_begin + 1, 1, 0, -1,
            ),
            (
                0, "task_publish", "gm_store",
                "cell_payload", publish_begin + 2, 1, 0, 0,
            ),
            (
                0, "task_publish", "scalar_work",
                "payload_validation", publish_begin + 3, 1, 0, -1,
            ),
            (
                0, "task_publish", "cache_clean_cvac",
                "cell_payload", publish_begin + 4, 4, 4, 0,
            ),
            (
                0, "task_publish", "barrier_dsb_sy",
                "none", publish_begin + 5, 1, 0, -1,
            ),
            (
                0, "task_publish", "barrier_isb",
                "none", publish_begin + 6, 1, 0, -1,
            ),
        )
        capture["aicpu_operations"] = [
            {
                "sequence": sequence,
                "task_id": task_id,
                "scope": scope,
                "operation": operation,
                "target": target,
                "clock": "aicpu_monotonic_raw_ns",
                "begin_ns": begin_ns,
                "end_ns": begin_ns + 1,
                "calls": calls,
                "lines": lines,
                "first_target_index": target_index,
                "last_target_index": target_index,
                "first_value": 1 if operation == "scalar_work" else 0,
                "last_value": 1 if operation == "scalar_work" else 0,
            }
            for sequence, (
                task_id, scope, operation, target, begin_ns,
                calls, lines, target_index,
            ) in enumerate(operation_specs)
        ]
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "aicpu_operation_raw.json"
            merged_path = Path(directory) / "aicpu_operation_merged.json"
            raw_path.write_text(json.dumps(capture), encoding="utf-8")
            MODULE.convert(raw_path, merged_path)
            merged = json.loads(
                merged_path.read_text(encoding="utf-8")
            )

        operation_events = [
            event
            for event in merged["traceEvents"]
            if event.get("cat") in {
                "aicpu.atomic", "aicpu.cache", "aicpu.barrier",
                "aicpu.gm", "aicpu.scalar",
            }
        ]
        self.assertEqual(len(operation_events), 7)
        events_by_name = {
            str(event["name"]): event for event in operation_events
        }
        self.assertEqual(
            set(events_by_name),
            {
                "cache.owner_setup.request_tensors.dc_civac×2.lines2#owner",
                "atomic.task_publish.fatal.load_acquire#0",
                "gm.task_publish.cell_payload.store#0",
                "scalar.task_publish.payload_validation#0",
                "cache.task_publish.cell_payload.dc_cvac×4.lines4#0",
                "barrier.task_publish.dsb_sy#0",
                "barrier.task_publish.isb#0",
            },
        )
        self.assertEqual(
            events_by_name[
                "cache.owner_setup.request_tensors.dc_civac×2.lines2#owner"
            ]["tid"],
            MODULE.AICPU_THREAD_ID,
        )
        self.assertTrue(
            all(
                event["tid"] == MODULE.AICPU_TASK_THREAD_ID
                for name, event in events_by_name.items()
                if name.endswith("#0")
            )
        )
        self.assertTrue(all(event["dur"] > 0 for event in operation_events))

    def test_aicpu_operation_trace_malformed_input_fails_closed(self) -> None:
        def make_capture() -> dict[str, object]:
            capture = _runtime_plan_capture()
            tasks = capture["aicpu_tasks"]
            metadata = capture["metadata"]
            assert isinstance(tasks, list) and isinstance(tasks[0], dict)
            assert isinstance(metadata, dict)
            publish_begin = int(tasks[0]["publish_begin_ns"])
            metadata["aicpu_operation_trace"] = {
                "enabled": True,
                "records": 1,
                "record_bytes": 64,
                "dropped": 0,
            }
            capture["aicpu_operations"] = [{
                "sequence": 0,
                "task_id": 0,
                "scope": "task_publish",
                "operation": "atomic_load_acquire",
                "target": "fatal",
                "clock": "aicpu_monotonic_raw_ns",
                "begin_ns": publish_begin + 1,
                "end_ns": publish_begin + 2,
                "calls": 1,
                "lines": 0,
                "first_target_index": -1,
                "last_target_index": -1,
                "first_value": 0,
                "last_value": 0,
            }]
            return capture

        cases = (
            (
                "dropped record",
                lambda capture: capture["metadata"][
                    "aicpu_operation_trace"
                ].__setitem__("dropped", 1),
                "complete non-empty",
            ),
            (
                "summary count mismatch",
                lambda capture: capture["metadata"][
                    "aicpu_operation_trace"
                ].__setitem__("records", 2),
                "complete non-empty",
            ),
            (
                "invalid atomic target",
                lambda capture: capture["aicpu_operations"][0].__setitem__(
                    "target", "none"
                ),
                "operation semantics",
            ),
            (
                "outside parent interval",
                lambda capture: capture["aicpu_operations"][0].__setitem__(
                    "end_ns",
                    capture["aicpu_tasks"][0]["publish_end_ns"] + 1,
                ),
                "parent containment",
            ),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                capture = make_capture()
                mutate(capture)
                raw_path = Path(directory) / "malformed_operation.json"
                merged_path = Path(directory) / "merged.json"
                raw_path.write_text(json.dumps(capture), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.convert(raw_path, merged_path)
                self.assertFalse(merged_path.exists())

    def test_host_cpu_domain_is_explicit_and_never_fakes_an_aicpu_lane(
        self,
    ) -> None:
        capture = _runtime_plan_capture()
        metadata = capture["metadata"]
        assert isinstance(metadata, dict)
        metadata["runtime_plan_producer_domain"] = "host-cpu"
        metadata["profiling_primary_view"] = (
            "host_cpu_functional_structure"
        )
        metadata["clock_correlation_warmup_before_pipeline"] = False
        metadata["timing_scope"] = "host-cpu-functional-capture"
        metadata.pop("aicpu_aicore_clock_correlation")
        metadata.pop("aicpu_aicore_causal_capture_bracket")
        capture["aicpu_tasks"] = []
        capture["aicpu_scheduler_phases"] = []
        capture["aicpu_orchestrator_phases"] = []
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "host_cpu.json"
            merged_path = Path(directory) / "host_cpu_merged.json"
            raw_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "--host-cpu-functional"
            ):
                MODULE.convert(raw_path, merged_path)
            self.assertFalse(merged_path.exists())
            with self.assertRaisesRegex(
                ValueError, "--host-cpu-functional"
            ):
                ANALYZER_MODULE.analyze_capture(raw_path)
            MODULE.convert(
                raw_path,
                merged_path,
                allow_host_cpu_functional=True,
            )
            merged = json.loads(
                merged_path.read_text(encoding="utf-8")
            )
            report = ANALYZER_MODULE.analyze_capture(
                raw_path,
                allow_host_cpu_functional=True,
            )
        self.assertFalse(any(
            event.get("name") == "RuntimePlanProducer"
            for event in merged["traceEvents"]
        ))
        self.assertEqual(
            report["capture"]["runtime_plan_producer_domain"],
            "host-cpu",
        )
        self.assertEqual(
            merged["metadata"]["profiling_scope"],
            "host-cpu-functional-only",
        )
        self.assertIs(merged["metadata"]["joint_profiling"], False)
        self.assertEqual(report["validation"]["status"], "FUNCTIONAL_ONLY")
        self.assertIs(report["capture"]["joint_profiling"], False)
        self.assertNotIn("aicpu_runtime_plan_producer", report)
        self.assertNotIn("aicpu_aicore_joint_timing", report)

    def test_missing_producer_domain_cannot_emit_merged_or_analyzer_pass(
        self,
    ) -> None:
        capture = _runtime_plan_capture()
        metadata = capture["metadata"]
        assert isinstance(metadata, dict)
        metadata.pop("runtime_plan_producer_domain")
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "missing_domain.json"
            merged_path = Path(directory) / "merged.json"
            raw_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "joint AICPU/AICore"):
                MODULE.convert(raw_path, merged_path)
            self.assertFalse(merged_path.exists())
            with self.assertRaisesRegex(ValueError, "joint AICPU/AICore"):
                ANALYZER_MODULE.analyze_capture(raw_path)

    def test_aicpu_producer_schema_fails_closed_on_missing_or_unknown(
        self,
    ) -> None:
        cases: list[tuple[str, dict[str, object], str]] = []
        missing_primary_view = _runtime_plan_capture()
        metadata = missing_primary_view["metadata"]
        assert isinstance(metadata, dict)
        metadata.pop("profiling_primary_view")
        cases.append((
            "missing_primary_view",
            missing_primary_view,
            "profiling_primary_view",
        ))

        unknown_primary_view = _runtime_plan_capture()
        metadata = unknown_primary_view["metadata"]
        assert isinstance(metadata, dict)
        metadata["profiling_primary_view"] = "host_pipeline_e2e"
        cases.append((
            "unknown_primary_view",
            unknown_primary_view,
            "profiling_primary_view",
        ))

        missing_phase = _runtime_plan_capture()
        missing_phase.pop("aicpu_orchestrator_phases")
        cases.append(("missing_phase", missing_phase, "orchestrator_phases"))

        missing_correlation = _runtime_plan_capture()
        metadata = missing_correlation["metadata"]
        assert isinstance(metadata, dict)
        metadata.pop("aicpu_aicore_clock_correlation")
        cases.append((
            "missing_correlation",
            missing_correlation,
            "clock_correlation",
        ))

        unknown_phase = _runtime_plan_capture()
        phases = unknown_phase["aicpu_orchestrator_phases"]
        assert isinstance(phases, list) and isinstance(phases[0], dict)
        phases[0]["name"] = "SyntheticBuild"
        cases.append(("unknown_phase", unknown_phase, "RuntimePlanProducer"))

        unknown_status = _runtime_plan_capture()
        metadata = unknown_status["metadata"]
        assert isinstance(metadata, dict)
        correlation = metadata["aicpu_aicore_clock_correlation"]
        assert isinstance(correlation, dict)
        correlation["status"] = "estimated"
        cases.append(("unknown_status", unknown_status, "status"))

        performance_claim = _runtime_plan_capture()
        metadata = performance_claim["metadata"]
        assert isinstance(metadata, dict)
        metadata["performance_representative"] = True
        cases.append((
            "performance_claim",
            performance_claim,
            "performance_representative",
        ))

        with tempfile.TemporaryDirectory() as directory:
            for name, capture, expected in cases:
                with self.subTest(name=name):
                    path = Path(directory) / f"{name}.json"
                    path.write_text(json.dumps(capture), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, expected):
                        MODULE._load_and_validate(path)

    def test_aicpu_taskplan_detail_requires_complete_real_trace(
        self,
    ) -> None:
        cases: list[tuple[str, dict[str, object], str]] = []

        missing_tasks = _runtime_plan_capture()
        missing_tasks.pop("aicpu_tasks")
        cases.append(("missing_tasks", missing_tasks, "must be an array"))

        incomplete_tasks = _runtime_plan_capture()
        tasks = incomplete_tasks["aicpu_tasks"]
        assert isinstance(tasks, list)
        tasks.pop()
        cases.append((
            "incomplete_tasks",
            incomplete_tasks,
            "must exactly cover",
        ))

        wrong_identity = _runtime_plan_capture()
        tasks = wrong_identity["aicpu_tasks"]
        assert isinstance(tasks, list) and isinstance(tasks[2], dict)
        tasks[2]["engine"] = "aic"
        cases.append((
            "wrong_identity",
            wrong_identity,
            "invalid identity or callback timeline",
        ))

        broken_owner_partition = _runtime_plan_capture()
        phases = broken_owner_partition["aicpu_scheduler_phases"]
        assert isinstance(phases, list) and isinstance(phases[1], dict)
        phases[1]["begin_ns"] = int(phases[1]["begin_ns"]) + 1
        cases.append((
            "broken_owner_partition",
            broken_owner_partition,
            "exact non-empty partition",
        ))

        oversized_payload = _runtime_plan_capture()
        tasks = oversized_payload["aicpu_tasks"]
        assert isinstance(tasks, list) and isinstance(tasks[0], dict)
        tasks[0]["payload_lines"] = 70
        cases.append((
            "oversized_payload",
            oversized_payload,
            "invalid identity or callback timeline",
        ))

        with tempfile.TemporaryDirectory() as directory:
            for name, capture, expected in cases:
                with self.subTest(name=name):
                    path = Path(directory) / f"{name}.json"
                    path.write_text(json.dumps(capture), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, expected):
                        MODULE._load_and_validate(path)

            maximum_payload = _runtime_plan_capture()
            tasks = maximum_payload["aicpu_tasks"]
            assert isinstance(tasks, list) and isinstance(tasks[0], dict)
            tasks[0]["payload_lines"] = 69
            path = Path(directory) / "maximum_payload.json"
            path.write_text(json.dumps(maximum_payload), encoding="utf-8")
            MODULE._load_and_validate(path)

    def test_aicpu_producer_rejects_empty_or_reversed_intervals(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name, end_ns in (("empty", 200), ("reversed", 199)):
                capture = _runtime_plan_capture()
                phases = capture["aicpu_orchestrator_phases"]
                assert isinstance(phases, list)
                phase = phases[0]
                assert isinstance(phase, dict)
                phase["end_ns"] = end_ns
                path = Path(directory) / f"{name}.json"
                path.write_text(json.dumps(capture), encoding="utf-8")
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "non-empty"
                ):
                    MODULE._load_and_validate(path)

    def test_aicpu_correlation_rejects_empty_intersection_and_reused_nonce(
        self,
    ) -> None:
        empty_intersection = _runtime_plan_capture()
        metadata = empty_intersection["metadata"]
        assert isinstance(metadata, dict)
        correlation = metadata["aicpu_aicore_clock_correlation"]
        assert isinstance(correlation, dict)
        samples = correlation["samples"]
        assert isinstance(samples, list)
        for sample in samples[4:]:
            assert isinstance(sample, dict)
            sample["aicore_receive_tick"] = int(
                sample["aicore_receive_tick"]
            ) + 100
            sample["aicore_send_tick"] = int(
                sample["aicore_send_tick"]
            ) + 100
            sample["offset_lower_tick"] = int(
                sample["offset_lower_tick"]
            ) + 100
            sample["offset_upper_tick"] = int(
                sample["offset_upper_tick"]
            ) + 100

        reused_nonce = _runtime_plan_capture()
        metadata = reused_nonce["metadata"]
        assert isinstance(metadata, dict)
        correlation = metadata["aicpu_aicore_clock_correlation"]
        assert isinstance(correlation, dict)
        samples = correlation["samples"]
        assert isinstance(samples, list)
        for sample in samples[4:]:
            assert isinstance(sample, dict)
            sample["round_nonce"] = 1_111

        with tempfile.TemporaryDirectory() as directory:
            for name, capture, expected in (
                ("empty_intersection", empty_intersection, "empty offset intersection"),
                ("reused_nonce", reused_nonce, "different nonces"),
            ):
                with self.subTest(name=name):
                    path = Path(directory) / f"{name}.json"
                    path.write_text(json.dumps(capture), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, expected):
                        MODULE._load_and_validate(path)

    def test_aicpu_correlation_rejects_alignment_error_above_50us(
        self,
    ) -> None:
        capture = _runtime_plan_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, list)
            row[6] = int(row[6]) + 1_000_000
            row[7] = int(row[7]) + 1_000_000
        _set_aicpu_timing(
            capture,
            999_200,
            1_001_100,
            minimum_fdwic_start_tick=1_001_000,
            maximum_fdwic_end_tick=1_001_250,
            half_width=50_001,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "high_error.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "50 us limit"):
                MODULE._load_and_validate(path)

    def test_aicpu_direct_tick_splicing_is_rejected(self) -> None:
        capture = _runtime_plan_capture()
        phases = capture["aicpu_orchestrator_phases"]
        assert isinstance(phases, list)
        phase = phases[0]
        assert isinstance(phase, dict)
        phase["begin_tick"] = phase.pop("begin_ns")
        phase["end_tick"] = phase.pop("end_ns")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "direct_splice.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "direct AICore-tick splicing"):
                MODULE._load_and_validate(path)

    def test_host_timing_metadata_is_explicitly_forbidden(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for field, value in (
                (
                    "runtime_plan_pipeline_clock_bracket",
                    {
                        "clock": "host_monotonic_raw_ns",
                        "begin_ns": 1,
                        "end_ns": 2,
                    },
                ),
                (
                    "host_pipeline_e2e",
                    {
                        "clock": "host_steady_duration_only",
                        "duration_ns": 1_080,
                    },
                ),
                ("timing_primary_metric", "pipeline_e2e"),
                ("aicore_time_scope", "aicore-stream-wall"),
            ):
                capture = _runtime_plan_capture()
                metadata = capture["metadata"]
                assert isinstance(metadata, dict)
                metadata[field] = value
                path = Path(directory) / f"forbidden_{field}.json"
                path.write_text(json.dumps(capture), encoding="utf-8")
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "Host-derived timing metadata"
                ):
                    MODULE._load_and_validate(path)

    def test_aicpu_owner_must_fit_strict_sample_derived_window(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for boundary in ("begin", "end"):
                capture = _runtime_plan_capture()
                metadata = capture["metadata"]
                phases = capture["aicpu_orchestrator_phases"]
                scheduler_phases = capture["aicpu_scheduler_phases"]
                assert isinstance(metadata, dict)
                assert isinstance(phases, list) and isinstance(phases[0], dict)
                assert isinstance(scheduler_phases, list)
                bracket = metadata["aicpu_aicore_causal_capture_bracket"]
                assert isinstance(bracket, dict)
                if boundary == "begin":
                    phases[0]["begin_ns"] = bracket[
                        "aicpu_pre_receive_max_ns"
                    ]
                    assert isinstance(scheduler_phases[0], dict)
                    scheduler_phases[0]["begin_ns"] = phases[0][
                        "begin_ns"
                    ]
                else:
                    phases[0]["end_ns"] = bracket[
                        "aicpu_post_send_min_ns"
                    ]
                    assert isinstance(scheduler_phases[-1], dict)
                    scheduler_phases[-1]["end_ns"] = phases[0][
                        "end_ns"
                    ]
                path = Path(directory) / f"owner_{boundary}.json"
                path.write_text(json.dumps(capture), encoding="utf-8")
                with self.subTest(boundary=boundary), self.assertRaisesRegex(
                    ValueError, "outside the strict AICPU"
                ):
                    MODULE._load_and_validate(path)

    def test_aicore_rows_must_fit_causal_calibration_window(self) -> None:
        capture = _runtime_plan_capture()
        metadata = capture["metadata"]
        assert isinstance(metadata, dict)
        bracket = metadata["aicpu_aicore_causal_capture_bracket"]
        assert isinstance(bracket, dict)
        correlation = metadata["aicpu_aicore_clock_correlation"]
        assert isinstance(correlation, dict)
        samples = correlation["samples"]
        assert isinstance(samples, list)
        for index, sample in enumerate(samples[4:]):
            assert isinstance(sample, dict)
            shift = -40
            for field in (
                "aicpu_send_ns",
                "aicpu_receive_ns",
                "aicore_receive_tick",
                "aicore_send_tick",
            ):
                sample[field] = int(sample[field]) + shift
        bracket["aicpu_post_send_min_ns"] = min(
            int(sample["aicpu_send_ns"])
            for sample in samples[4:]
        )
        bracket["aicore_post_receive_min_tick"] = min(
            int(sample["aicore_receive_tick"])
            for sample in samples[4:]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outside_causal_window.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the strict pre-send"):
                MODULE._load_and_validate(path)

    def test_legacy_abi2_cannot_emit_merged_or_analyzer_pass(
        self,
    ) -> None:
        capture = _runtime_plan_capture()
        metadata = capture["metadata"]
        assert isinstance(metadata, dict)
        for field in (
            "runtime_plan_abi", "pipeline", "launch_order",
            "producer_ready", "consumer_admission", "prefill",
            "runtime_plan_build_backend",
            "runtime_plan_build_workers",
            "runtime_plan_execute_workers",
            "runtime_plan_build_trace_coverage",
            "runtime_plan_producer_task_count",
            "runtime_plan_task_count", "runtime_plan_task_kinds",
            "runtime_plan_terminal",
            "runtime_plan_producer_domain",
            "profiling_primary_view",
            "clock_correlation_warmup_before_pipeline",
            "timing_scope",
            "performance_representative",
            "aicpu_aicore_clock_correlation",
            "aicpu_aicore_causal_capture_bracket",
        ):
            metadata.pop(field)
        capture["aicpu_tasks"] = []
        capture["aicpu_scheduler_phases"] = []
        capture["aicpu_orchestrator_phases"] = []

        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "legacy.json"
            merged_path = Path(directory) / "merged.json"
            raw_path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "legacy ABI2"):
                MODULE.convert(raw_path, merged_path)
            self.assertFalse(merged_path.exists())
            with self.assertRaisesRegex(ValueError, "legacy ABI2"):
                ANALYZER_MODULE.analyze_capture(raw_path)

    def test_closed_atomic_fast_path_is_exact_but_streaming_is_bounded(
        self,
    ) -> None:
        task_count = 5
        counts = {
            58: CORE_COUNT + 1,
            59: 0,
            60: CORE_COUNT + 1,
            61: CORE_COUNT,
            62: task_count + CORE_COUNT,
            63: 1,
            64: 2 * task_count,
            65: CORE_COUNT,
            66: 1,
            67: CORE_COUNT + 1,
            68: 1,
            69: 1,
        }
        MODULE._validate_runtime_plan_atomic_closure(
            "plan-ahead-closed", "scalar", "scalar-task-detail",
            counts, task_count, CORE_COUNT, CORE_COUNT,
        )
        MODULE._validate_runtime_plan_atomic_closure(
            "streaming-future", "scalar", "scalar-task-detail",
            counts, task_count, CORE_COUNT, CORE_COUNT,
        )

        overlapped_counts = dict(counts)
        overlapped_counts[61] += 7
        overlapped_counts[64] += 101
        MODULE._validate_runtime_plan_atomic_closure(
            "streaming-future", "scalar", "scalar-task-detail",
            overlapped_counts, task_count, CORE_COUNT, CORE_COUNT,
        )
        with self.assertRaisesRegex(
            ValueError, "pipeline=plan-ahead-closed"
        ):
            MODULE._validate_runtime_plan_atomic_closure(
                "plan-ahead-closed", "scalar", "scalar-task-detail",
                overlapped_counts, task_count, CORE_COUNT, CORE_COUNT,
            )

    def test_simt_coarse_capture_uses_direct_state_without_scalar_children(
        self,
    ) -> None:
        capture = _simt_runtime_plan_capture()
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "simt.json"
            merged_path = Path(directory) / "simt_merged.json"
            raw_path.write_text(json.dumps(capture), encoding="utf-8")
            MODULE.convert(raw_path, merged_path)
            merged = json.loads(
                merged_path.read_text(encoding="utf-8")
            )
        self.assertEqual(
            merged["metadata"]["runtime_plan_build_backend"], "simt"
        )
        self.assertEqual(
            merged["metadata"]["runtime_plan_task_count"], 5
        )
        self.assertFalse(
            any(
                row[5] == "Materialize"
                for row in capture["fdwic_events"]
            )
        )

    def test_simt_backend_worker_identity_and_scalar_children_are_rejected(
        self,
    ) -> None:
        capture = _simt_runtime_plan_capture()
        metadata = capture["metadata"]
        assert isinstance(metadata, dict)
        metadata["runtime_plan_build_workers"] = CORE_COUNT
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simt_bad_workers.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "runtime_plan_build_workers"
            ):
                MODULE._load_and_validate(path)

        capture = _simt_runtime_plan_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        rows.append(
            _row(0, 0, "Materialize", 1_020, 1_030, auxiliary=1)
        )
        metadata = capture["metadata"]
        assert isinstance(metadata, dict)
        summary = metadata["fdwic_summary"]
        assert isinstance(summary, dict)
        summary["records"] = len(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simt_fake_scalar_child.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "must not contain Scalar Build child"
            ):
                MODULE._load_and_validate(path)

    def test_scalar_detailed_capture_still_requires_every_materialize(
        self,
    ) -> None:
        capture = _runtime_plan_capture()
        rows = capture["fdwic_events"]
        assert isinstance(rows, list)
        rows[:] = [
            row for row in rows
            if not (row[3] == 4 and row[5] == "Materialize")
        ]
        metadata = capture["metadata"]
        assert isinstance(metadata, dict)
        summary = metadata["fdwic_summary"]
        assert isinstance(summary, dict)
        summary["records"] = len(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scalar_missing_materialize.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE._load_and_validate(path)

    def test_simt_atomic_raw_closes_continuations_and_not_vf_rows(
        self,
    ) -> None:
        counts = {
            58: 2 * CORE_COUNT,
            59: 0,
            60: 2 * CORE_COUNT,
            61: 2 * CORE_COUNT,
            67: CORE_COUNT,
        }
        MODULE._validate_runtime_plan_atomic_closure(
            "streaming-future", "simt",
            "simt-coarse-direct-state", counts, 5, 4, CORE_COUNT,
        )
        counts[65] = 4
        with self.assertRaisesRegex(ValueError, "must not counterfeit VF"):
            MODULE._validate_runtime_plan_atomic_closure(
                "streaming-future", "simt",
                "simt-coarse-direct-state", counts, 5, 4,
                CORE_COUNT,
            )


class RuntimePlanExclusiveAnalyzerTest(unittest.TestCase):
    def test_simt_coarse_analyzer_does_not_invent_scalar_children(
        self,
    ) -> None:
        capture = _simt_runtime_plan_capture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simt_l2_swimlane_records.json"
            path.write_text(json.dumps(capture), encoding="utf-8")
            report = ANALYZER_MODULE.analyze_capture(path)

        self.assertEqual(report["validation"]["status"], "PASS")
        self.assertEqual(
            report["capture"]["runtime_plan_build_backend"], "simt"
        )
        self.assertEqual(report["capture"]["task_count_global"], 5)
        self.assertEqual(report["capture"]["planned_build_count"], 0)
        self.assertNotIn("materialize_breakdown", report)
        self.assertNotIn("register_breakdown", report)
        self.assertEqual(
            report["semantics"]["runtime_plan_build_children"],
            ["RuntimePlanBuildResidual"],
        )
        producer = report["aicpu_runtime_plan_producer"]
        self.assertEqual(producer["duration_ns"], 950)
        self.assertEqual(
            producer["runtime_plan_build_interval_union"]
            ["producer_overlap"],
            {
                "minimum_cycles": 200,
                "midpoint_cycles": 200,
                "maximum_cycles": 200,
                "minimum_us": 0.2,
                "midpoint_us": 0.2,
                "maximum_us": 0.2,
            },
        )
        self.assertEqual(
            report["aicpu_aicore_joint_timing"]["pipeline"],
            "streaming-future",
        )
        metrics = report["aggregate_core_work"]["metrics_cycles"]
        self.assertEqual(metrics["planned_build_union"], 0)
        self.assertEqual(
            metrics["runtime_plan_build_residual"],
            metrics["runtime_plan_build"],
        )

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
        producer = report["aicpu_runtime_plan_producer"]
        self.assertEqual(producer["duration_ns"], 600)
        self.assertEqual(
            producer["clock_alignment"]["alignment_error_us"],
            0.001,
        )
        self.assertEqual(
            producer["runtime_plan_build_interval_union"]
            ["producer_overlap"][
                "maximum_cycles"
            ],
            0,
        )
        self.assertEqual(
            producer[
                "producer_to_earliest_runtime_plan_build_gap"
            ],
            {
                "minimum_cycles": 99,
                "midpoint_cycles": 100,
                "maximum_cycles": 101,
                "minimum_us": 0.099,
                "midpoint_us": 0.1,
                "maximum_us": 0.101,
            },
        )
        self.assertFalse(
            producer["included_in_96_core_exclusive_closure"]
        )
        joint = report["aicpu_aicore_joint_timing"]
        self.assertEqual(
            joint["timing_scope"],
            "calibrated-structural-capture",
        )
        self.assertIs(joint["performance_representative"], False)
        self.assertIn("execute_kernel_union_overlap", joint)
        self.assertIn("final_drain_overlap", joint)
        self.assertTrue(
            report["validation"]
            ["all_aicore_events_inside_causal_capture_window"]
        )
        self.assertNotIn("pipeline_e2e", joint)

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
