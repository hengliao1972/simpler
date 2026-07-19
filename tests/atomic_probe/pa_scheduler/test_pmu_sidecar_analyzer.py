#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""standalone PMU sidecar 多轮分析器的 raw 门禁与聚合回归。"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

try:
    from .pmu_sidecar_analyzer import (
        A5_AIC_PER_DIE,
        A5_AIC_WORKERS,
        A5_AIV_WORKERS,
        A5_OWNER_MAGIC,
        A5_OWNER_VERSION,
        A5_PHYSICAL_SUBCORES,
        A5_SUBCORES_PER_DIE,
        A5_WORKERS,
        METRIC_NAMES,
        PHASE_STATUS_REQUIRED_MASK,
        PROGRAMMABLE_COUNTER_RISK_THRESHOLD,
        SUBMIT_PMU_METRIC_NAMES,
        TASKS_PER_BATCH,
        analyze,
        load_capture,
    )
except ImportError:
    from pmu_sidecar_analyzer import (
        A5_AIC_PER_DIE,
        A5_AIC_WORKERS,
        A5_AIV_WORKERS,
        A5_OWNER_MAGIC,
        A5_OWNER_VERSION,
        A5_PHYSICAL_SUBCORES,
        A5_SUBCORES_PER_DIE,
        A5_WORKERS,
        METRIC_NAMES,
        PHASE_STATUS_REQUIRED_MASK,
        PROGRAMMABLE_COUNTER_RISK_THRESHOLD,
        SUBMIT_PMU_METRIC_NAMES,
        TASKS_PER_BATCH,
        analyze,
        load_capture,
    )


def _p95(values: Sequence[int]) -> int:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _summary_for_metrics(
    records: list[dict[str, Any]], metric_names: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cores": len(records),
        "active_cores": len(records),
        "trusted_cores": len(records),
    }
    for metric in metric_names:
        values = [record[metric] for record in records]
        result[metric] = {
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2]
            if len(values) % 2
            else (sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]) / 2,
            "p95": _p95(values),
            "max": max(values),
        }
    result["icache_miss_rate"] = (
        result["icache_misses"]["sum"] / result["icache_requests"]["sum"]
    )
    return result


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return _summary_for_metrics(records, METRIC_NAMES)


def _submit_pmu_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = _summary_for_metrics(records, SUBMIT_PMU_METRIC_NAMES)
    phase_requests = result["phase_icache_requests"]["sum"]
    phase_misses = result["phase_icache_misses"]["sum"]
    result["phase_observed_read_clear_ratio"] = (
        None if phase_requests == 0 else phase_misses / phase_requests
    )
    return result


def _submit_pmu_physical_id(worker_id: int) -> int:
    """构造与 host mixed launch 相同的 32 组物理 1:2 triplet。"""

    if worker_id < A5_AIC_WORKERS:
        block = worker_id
        die_base = 0 if block < A5_AIC_WORKERS // 2 else A5_SUBCORES_PER_DIE
        return die_base + block % (A5_AIC_WORKERS // 2)
    vector_id = worker_id - A5_AIC_WORKERS
    block = vector_id // 2
    aic_id = _submit_pmu_physical_id(block)
    die_base = (aic_id // A5_SUBCORES_PER_DIE) * A5_SUBCORES_PER_DIE
    local_aic = aic_id % A5_SUBCORES_PER_DIE
    return die_base + A5_AIC_PER_DIE + local_aic * 2 + vector_id % 2


def _submit_pmu_bitmap_words() -> list[int]:
    words = [0, 0, 0, 0]
    for worker_id in range(A5_WORKERS):
        physical_id = _submit_pmu_physical_id(worker_id)
        words[physical_id // 32] |= 1 << (physical_id % 32)
    return words


def _capture(offset: int = 0, window: str = "submit-all") -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for worker_id, role in enumerate(("aic", "aiv", "aiv")):
        base = 100 + worker_id * 10 + offset
        record: dict[str, Any] = {
            "worker_id": worker_id,
            "physical_core_id": 10 + worker_id,
            "role": role,
            "trusted": True,
            "selectors_match": True,
            "owner_bitmap_member": True,
            "worker_slot_exact": True,
            "physical_role_matches": True,
            "window_started": True,
            "window_stopped": True,
        }
        for metric_index, metric in enumerate(METRIC_NAMES):
            record[metric] = base + metric_index
        record["total_cycles"] = base + len(METRIC_NAMES)
        # miss/request 取独立、直观的数值，便于断言聚合公式。
        record["icache_requests"] = 1000 + base
        record["icache_misses"] = 100 + base // 10
        records.append(record)

    groups = {
        "all": records,
        "aic": [record for record in records if record["role"] == "aic"],
        "aiv": [record for record in records if record["role"] == "aiv"],
    }
    return {
        "schema": {"name": "pa_scheduler_pmu_phase_windows", "version": 3},
        "capture": {
            "capture_id": f"capture-{offset}",
            "accepted": True,
            "published_after_runtime_cleanup": True,
            "runtime_cleanup_passed": True,
            "owner_restore_passed": True,
        },
        "configuration": {
            "device": 0,
            "batches": 256,
            "workers": 3,
            "aic_workers": 1,
            "aiv_workers": 2,
            "pmu_window": window,
            "submit_span_us": 5000.0 + offset,
            "selectors": {"cnt6_icache_request": 0x034, "cnt7_icache_miss": 0x035},
            "counter_width_bits": {"total": 64, "programmable": 32},
            "phase_timestamp_calls_present": True,
            "phase_record_writes": False,
            "profile_accumulation": False,
            "trace_enabled": False,
            "atomic_trace": False,
            "gate_start_stop_have_pipe_all_barriers": True,
            "winner_workload": {
                "mode": "real-compute",
                "counts": {"qk": 6, "sf": 28, "pv": 4, "up": 1},
                "unit": "complete_128x128_engine_pipeline_iteration",
            },
        },
        "validation": {
            "semantic_passed": True,
            "pmu_passed": True,
            "icache_measurement_valid": True,
            "icache_miss_le_request": True,
            "counter_below_risk_threshold": True,
            "trusted_records": 3,
            "unique_physical_core_ids": 3,
            "owner_bitmap_member_records": 3,
            "exact_worker_slot_records": 3,
            "physical_role_match_records": 3,
            "window_started_records": 3,
            "window_stopped_records": 3,
        },
        "records": records,
        "summary": {name: _summary(group) for name, group in groups.items()},
    }


def _submit_pmu_capture(offset: int = 0, phase: str = "claim") -> dict[str, Any]:
    phase_ids = {
        "none": 0,
        "claim": 1,
        "efdrain": 2,
        "materialize": 4,
        "register": 5,
    }
    phase_id = phase_ids[phase]
    batches = 2
    records: list[dict[str, Any]] = []
    for worker_id in range(A5_WORKERS):
        role = "aic" if worker_id < A5_AIC_WORKERS else "aiv"
        vector_id = 0 if role == "aic" else worker_id - A5_AIC_WORKERS
        block_id = worker_id if role == "aic" else vector_id // 2
        lane = 0 if role == "aic" else 1 + vector_id % 2
        base = 100 + worker_id * 10 + offset
        calls_per_worker = 0 if phase == "none" else batches * TASKS_PER_BATCH
        primary_requests = 1000 + base
        primary_misses = 100 + base // 10
        phase_requests = 0 if calls_per_worker == 0 else 100 + worker_id * 10 + offset
        phase_misses = 0 if calls_per_worker == 0 else 10 + worker_id + offset // 10
        request_loss = 0 if calls_per_worker == 0 else 1 + worker_id % 3
        miss_loss = 0 if calls_per_worker == 0 else 1 + worker_id % 2
        shadow_requests = primary_requests - request_loss
        shadow_misses = primary_misses - miss_loss
        record: dict[str, Any] = {
            "worker_id": worker_id,
            "physical_core_id": _submit_pmu_physical_id(worker_id),
            "role": role,
            "block_id": block_id,
            "lane": lane,
            "trusted": True,
            "physical_core_id_valid": True,
            "selectors_match": True,
            "owner_bitmap_member": True,
            "worker_slot_exact": True,
            "physical_role_matches": True,
            "window_started": True,
            "window_stopped": True,
            "build_variant_id": 2,
            "compiled_phase_id": phase_id,
            "phase_status": PHASE_STATUS_REQUIRED_MASK | (0x30 if calls_per_worker == 0 else 0),
            "phase_calls": calls_per_worker,
            "phase_expected_calls": calls_per_worker,
            "phase_begin_reads": calls_per_worker,
            "phase_end_reads": calls_per_worker,
            "primary_window_segments": 1,
            "shadow_read_segments": 2 * calls_per_worker + 1,
            "phase_icache_requests": phase_requests,
            "phase_icache_misses": phase_misses,
            "phase_icache_requests_upper_bound": phase_requests + request_loss,
            "phase_icache_misses_upper_bound": phase_misses + miss_loss,
            "shadow_whole_icache_requests": shadow_requests,
            "shadow_whole_icache_misses": shadow_misses,
            "shadow_matches_primary": request_loss == 0 and miss_loss == 0,
            "shadow_not_greater_than_primary": True,
            "shadow_request_loss": request_loss,
            "shadow_miss_loss": miss_loss,
            "phase_boundaries_balanced": True,
        }
        for metric_index, metric in enumerate(SUBMIT_PMU_METRIC_NAMES):
            record.setdefault(metric, base + metric_index)
        # total 是同一窗口的包络，fixture 也必须满足 scalar_busy <= total_cycles。
        record["total_cycles"] = base + len(SUBMIT_PMU_METRIC_NAMES)
        record["icache_requests"] = primary_requests
        record["icache_misses"] = primary_misses
        # setdefault 不覆盖上面按 phase 契约填写的五个扩展字段。
        records.append(record)

    groups = {
        "all": records,
        "aic": [record for record in records if record["role"] == "aic"],
        "aiv": [record for record in records if record["role"] == "aiv"],
    }
    workers = len(records)
    exact_records = sum(record["shadow_matches_primary"] for record in records)
    bounded_records = sum(record["shadow_not_greater_than_primary"] for record in records)
    acceptable_records = sum(
        record["shadow_matches_primary"]
        if record["phase_expected_calls"] == 0
        else record["shadow_not_greater_than_primary"]
        for record in records
    )
    request_losses = [record["shadow_request_loss"] for record in records]
    miss_losses = [record["shadow_miss_loss"] for record in records]
    return {
        "schema": {"name": "pa_scheduler_pmu_phase_windows", "version": 4},
        "capture": {
            "capture_id": f"submit-pmu-{phase}-{offset}",
            "accepted": True,
            "published_after_runtime_cleanup": True,
            "runtime_cleanup_passed": True,
            "owner_restore_passed": True,
        },
        "configuration": {
            "build_variant": "submit-pmu",
            "build_variant_id": 2,
            "compiled_phase": phase,
            "compiled_phase_id": phase_id,
            "device": 0,
            "batches": batches,
            "workers": workers,
            "aic_workers": A5_AIC_WORKERS,
            "aiv_workers": A5_AIV_WORKERS,
            "pmu_window": "submit-all",
            "primary_window_segments_per_record": 1,
            "unavailable_metrics": ["mte3_busy"],
            "submit_span_us": 5000.0 + offset,
            "selectors": {
                "cnt0_vector_busy": 0x501,
                "cnt1_cube_busy": 0x301,
                "cnt2_scalar_busy": 0x001,
                "cnt3_mte1_busy": 0x701,
                "cnt4_mte2_busy": 0x202,
                "cnt5_shadow_icache_miss": 0x035,
                "cnt6_primary_icache_request": 0x034,
                "cnt7_primary_icache_miss": 0x035,
                "cnt8_shadow_icache_request": 0x034,
                "cnt9_unused": 0x000,
            },
            "counter_width_bits": {"total": 64, "programmable": 32},
            "phase_timestamp_calls_present": False,
            "phase_record_writes": False,
            "profile_accumulation": False,
            "trace_enabled": False,
            "trace_atomics": False,
            "atomic_trace": False,
            "profile_phases": False,
            "gate_start_stop_have_pipe_all_barriers": True,
            "phase_boundary_observation_included": phase != "none",
            "phase_counter_pair_snapshot_atomic": False,
            "primary_counters_read_at_phase_boundaries": False,
            "phase_shadow_partition_exact_required": phase == "none",
            "phase_values_are_running_read_clear_lower_bounds": phase != "none",
            "cross_phase_elf_sums_valid": False,
            "winner_workload": {
                "mode": "real-compute",
                "counts": {"qk": 6, "sf": 28, "pv": 4, "up": 1},
                "unit": "complete_128x128_engine_pipeline_iteration",
            },
        },
        "validation": {
            "semantic_passed": True,
            "pmu_passed": True,
            "icache_measurement_valid": True,
            "icache_miss_le_request": True,
            "counter_below_risk_threshold": True,
            "phase_measurement_valid": True,
            "trusted_records": workers,
            "unique_physical_core_ids": workers,
            "owner_bitmap_member_records": workers,
            "exact_worker_slot_records": workers,
            "physical_role_match_records": workers,
            "window_started_records": workers,
            "window_stopped_records": workers,
            "build_variant_match_records": workers,
            "phase_id_match_records": workers,
            "phase_status_trusted_records": workers,
            "shadow_primary_match_records": exact_records,
            "shadow_primary_bounded_records": bounded_records,
            "phase_shadow_acceptable_records": acceptable_records,
            "shadow_request_abs_delta_sum": sum(request_losses),
            "shadow_request_abs_delta_max": max(request_losses),
            "shadow_request_signed_delta_sum": -sum(request_losses),
            "shadow_miss_abs_delta_sum": sum(miss_losses),
            "shadow_miss_abs_delta_max": max(miss_losses),
            "shadow_miss_signed_delta_sum": -sum(miss_losses),
            "phase_boundary_match_records": workers,
            "phase_call_shape_match_records": workers,
            "phase_calls": sum(record["phase_calls"] for record in records),
            "phase_expected_calls": sum(record["phase_expected_calls"] for record in records),
            "expected_records": A5_WORKERS,
            "expected_unique_core_ids": A5_WORKERS,
            "expected_owner_bitmap_member_records": A5_WORKERS,
            "expected_exact_worker_slot_records": A5_WORKERS,
            "expected_physical_role_match_records": A5_WORKERS,
            "mixed_triplet_matches": A5_AIC_WORKERS,
            "expected_mixed_triplet_matches": A5_AIC_WORKERS,
            "expected_window_records": A5_WORKERS,
        },
        "owner": {
            "mode": "main_aicpu_path_a",
            "snapshot_phase": "after_configure_before_restore",
            "control_magic": A5_OWNER_MAGIC,
            "control_version": A5_OWNER_VERSION,
            "configure_status": 0,
            "configured_flag": 1,
            "configured_bitmap_count": A5_WORKERS,
            "expected": {
                "total": A5_WORKERS,
                "aic": A5_AIC_WORKERS,
                "aiv": A5_AIV_WORKERS,
            },
            "active": {
                "total": A5_WORKERS,
                "aic": A5_AIC_WORKERS,
                "aiv": A5_AIV_WORKERS,
            },
            "discovered": {
                "total": A5_WORKERS,
                "aic": A5_AIC_WORKERS,
                "aiv": A5_AIV_WORKERS,
            },
            "physical_slots_scanned": A5_PHYSICAL_SUBCORES,
            "skipped_physical_slots": A5_PHYSICAL_SUBCORES - A5_WORKERS,
            "configured_bitmap_word_order": "least_significant_physical_ids_first",
            "configured_bitmap_words": _submit_pmu_bitmap_words(),
            "configured_complete_mixed_triplets": A5_AIC_WORKERS,
            "expected_complete_mixed_triplets": A5_AIC_WORKERS,
            "configured_broken_mixed_triplets": 0,
            "restore_passed": True,
        },
        "records": records,
        "summary": {name: _submit_pmu_summary(group) for name, group in groups.items()},
    }


class PmuSidecarAnalyzerTest(unittest.TestCase):
    def _write(self, directory: str, name: str, capture: dict[str, Any]) -> Path:
        path = Path(directory) / name
        path.write_text(json.dumps(capture), encoding="utf-8")
        return path

    def test_valid_raw_is_recomputed_and_multiple_runs_are_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self._write(directory, "first.json", _capture(0))
            second = self._write(directory, "second.json", _capture(20))
            result = analyze([first, second], miss_penalty_ns=90.0)

        self.assertTrue(result["validation"]["raw_to_summary_all_fields_match"])
        self.assertEqual(result["aggregate"]["runs"], 2)
        self.assertEqual(result["aggregate"]["submit_span_us"]["median"], 5010.0)
        expected_misses = [
            _capture(offset)["summary"]["all"]["icache_misses"]["sum"]
            for offset in (0, 20)
        ]
        self.assertEqual(
            result["aggregate"]["groups"]["all"]["icache_misses_sum"]["median"],
            sum(expected_misses) / 2,
        )
        self.assertTrue(result["estimation"]["not_wall_time"])
        self.assertTrue(result["estimation"]["not_additive_stall_time"])
        self.assertEqual(
            result["actual_exposed_loss"]["status"],
            "requires_same_semantics_paired_ab",
        )
        expected_aiv_misses_per_core = [
            _capture(offset)["summary"]["aiv"]["icache_misses"]["sum"] / 2
            for offset in (0, 20)
        ]
        self.assertEqual(
            result["aggregate"]["groups"]["aiv"]["icache_misses_per_core"]["median"],
            sum(expected_aiv_misses_per_core) / 2,
        )

    def test_tampered_host_summary_is_rejected(self) -> None:
        capture = _capture()
        capture["summary"]["aiv"]["icache_misses"]["sum"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "tampered.json", capture)
            with self.assertRaisesRegex(ValueError, "raw summary mismatch"):
                load_capture(path)

    def test_different_observation_configurations_cannot_be_merged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self._write(directory, "submit.json", _capture(window="submit-all"))
            second = self._write(directory, "empty.json", _capture(window="empty"))
            with self.assertRaisesRegex(ValueError, "observation configuration differs"):
                analyze([first, second])

    def test_failed_restore_is_rejected(self) -> None:
        capture = _capture()
        capture["capture"]["owner_restore_passed"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "restore_failed.json", capture)
            with self.assertRaisesRegex(ValueError, "owner_restore_passed is not true"):
                load_capture(path)

    def test_duplicate_physical_core_is_rejected(self) -> None:
        capture = _capture()
        capture["records"][1]["physical_core_id"] = capture["records"][0]["physical_core_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "duplicate.json", capture)
            with self.assertRaisesRegex(ValueError, "duplicate physical_core_id"):
                load_capture(path)

    def test_miss_greater_than_request_is_rejected_before_summary_use(self) -> None:
        capture = _capture()
        capture["records"][0]["icache_misses"] = capture["records"][0]["icache_requests"] + 1
        # 同步重算 summary，证明失败来自逐核物理约束，而不是 summary 不一致。
        records = capture["records"]
        capture["summary"] = {
            "all": _summary(records),
            "aic": _summary([record for record in records if record["role"] == "aic"]),
            "aiv": _summary([record for record in records if record["role"] == "aiv"]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "invalid_miss.json", capture)
            with self.assertRaisesRegex(ValueError, "miss > request"):
                load_capture(path)

    def test_submit_pmu_v4_claim_is_recomputed_and_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self._write(directory, "claim-first.json", _submit_pmu_capture(0, "claim"))
            second = self._write(directory, "claim-second.json", _submit_pmu_capture(10, "claim"))
            result = analyze([first, second])

        self.assertEqual(result["input_schema"]["version"], 4)
        self.assertEqual(result["schema"]["version"], 2)
        self.assertEqual(result["phase_observation"]["compiled_phase"], "claim")
        self.assertTrue(result["phase_observation"]["enabled"])
        self.assertFalse(result["phase_observation"]["cross_phase_elf_sums_valid"])
        self.assertEqual(
            result["phase_observation"]["phase_value_semantics"],
            "running_read_clear_lower_to_loss_adjusted_upper_bound",
        )
        expected_aiv_phase_misses = [
            _submit_pmu_capture(offset, "claim")["summary"]["aiv"][
                "phase_icache_misses"
            ]["sum"]
            / A5_AIV_WORKERS
            for offset in (0, 10)
        ]
        self.assertEqual(
            result["aggregate"]["groups"]["aiv"][
                "phase_icache_misses_lower_bound_per_core"
            ]["median"],
            sum(expected_aiv_phase_misses) / 2,
        )
        self.assertGreater(
            result["aggregate"]["groups"]["aiv"][
                "phase_icache_miss_lower_bound_share_of_submit"
            ]["median"],
            0,
        )
        claim_aiv = result["aggregate"]["groups"]["aiv"]
        self.assertGreater(
            claim_aiv["phase_icache_requests_upper_bound_per_core"]["median"],
            claim_aiv["phase_icache_requests_lower_bound_per_core"]["median"],
        )
        self.assertGreater(claim_aiv["shadow_request_loss_per_core"]["median"], 0)
        for row in result["per_run"]:
            for group_name in ("all", "aic", "aiv"):
                group = row["groups"][group_name]
                self.assertEqual(
                    group["phase_icache_requests_upper_bound_sum"]
                    - group["phase_icache_requests_lower_bound_sum"],
                    group["shadow_request_loss_sum"],
                )
                self.assertEqual(
                    group["phase_icache_misses_upper_bound_sum"]
                    - group["phase_icache_misses_lower_bound_sum"],
                    group["shadow_miss_loss_sum"],
                )
                self.assertLessEqual(
                    group["phase_icache_requests_lower_bound_per_core_median"],
                    group["phase_icache_requests_upper_bound_per_core_median"],
                )
                self.assertLessEqual(
                    group["phase_icache_misses_lower_bound_per_core_p95"],
                    group["phase_icache_misses_upper_bound_per_core_p95"],
                )
                self.assertEqual(
                    group["phase_icache_request_lower_bound_share_of_submit"],
                    group["phase_icache_requests_lower_bound_sum"]
                    / group["icache_requests_sum"],
                )
                self.assertEqual(
                    group["phase_icache_request_upper_bound_share_of_submit"],
                    group["phase_icache_requests_upper_bound_sum"]
                    / group["icache_requests_sum"],
                )
                self.assertEqual(
                    group["phase_icache_miss_lower_bound_share_of_submit"],
                    group["phase_icache_misses_lower_bound_sum"] / group["icache_misses_sum"],
                )
                self.assertEqual(
                    group["phase_icache_miss_upper_bound_share_of_submit"],
                    group["phase_icache_misses_upper_bound_sum"] / group["icache_misses_sum"],
                )
        self.assertLess(
            _submit_pmu_capture()["validation"]["shadow_request_signed_delta_sum"], 0
        )

    def test_submit_pmu_v4_none_has_zero_disabled_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "none.json", _submit_pmu_capture(0, "none"))
            result = analyze([path])

        self.assertFalse(result["phase_observation"]["enabled"])
        phase = result["aggregate"]["groups"]["aiv"]
        self.assertEqual(phase["phase_calls_sum"]["median"], 0)
        self.assertEqual(phase["phase_icache_misses_lower_bound_per_core"]["median"], 0)
        self.assertIsNone(phase["phase_observed_read_clear_ratio"]["median"])
        self.assertEqual(
            phase["phase_icache_miss_lower_bound_share_of_submit"]["median"], 0
        )
        self.assertEqual(
            phase["phase_icache_miss_upper_bound_share_of_submit"]["median"], 0
        )
        self.assertEqual(
            phase["phase_icache_requests_lower_bound_per_core"]["median"],
            phase["phase_icache_requests_upper_bound_per_core"]["median"],
        )
        self.assertEqual(phase["shadow_request_loss_sum"]["median"], 0)

    def test_submit_pmu_v4_requires_fixed_a5_topology(self) -> None:
        capture = _submit_pmu_capture()
        # 仍保持 workers=aic+aiv，证明拒绝原因是 schema-v4 的固定 A5 拓扑，
        # 不是原有的自报计数加和检查。
        capture["configuration"]["aic_workers"] = A5_AIC_WORKERS - 1
        capture["configuration"]["aiv_workers"] = A5_AIV_WORKERS + 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "wrong-topology.json", capture)
            with self.assertRaisesRegex(ValueError, "fixed 96/32/64 A5 topology"):
                load_capture(path)

    def test_submit_pmu_v4_logical_worker_triplet_is_recomputed(self) -> None:
        mutations = (
            (0, "worker_id", A5_WORKERS, "exact worker slot"),
            (0, "role", "aiv", "logical worker topology"),
            (0, "block_id", 1, "mixed block"),
            (A5_AIC_WORKERS, "lane", 2, "mixed lane"),
            (0, "physical_core_id_valid", False, "physical_core_id_valid"),
        )
        for record_index, field, value, message in mutations:
            with self.subTest(field=field):
                capture = _submit_pmu_capture()
                capture["records"][record_index][field] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = self._write(directory, f"wrong-{field}.json", capture)
                    with self.assertRaisesRegex(ValueError, message):
                        load_capture(path)

    def test_submit_pmu_v4_physical_range_and_triplets_are_recomputed(self) -> None:
        capture = _submit_pmu_capture()
        capture["records"][0]["physical_core_id"] = A5_PHYSICAL_SUBCORES
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "out-of-range-core.json", capture)
            with self.assertRaisesRegex(ValueError, "outside the 108-slot topology"):
                load_capture(path)

        capture = _submit_pmu_capture()
        # 交换两个 block 的 AIV0：ID 仍唯一、仍属于 owner、物理角色仍是 AIV，
        # 只有逐 block 的 1:2 关系被破坏。
        first = A5_AIC_WORKERS
        second = A5_AIC_WORKERS + 2
        capture["records"][first]["physical_core_id"], capture["records"][second][
            "physical_core_id"
        ] = (
            capture["records"][second]["physical_core_id"],
            capture["records"][first]["physical_core_id"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "crossed-aiv-triplets.json", capture)
            with self.assertRaisesRegex(ValueError, "mixed block 0"):
                load_capture(path)

    def test_submit_pmu_v4_owner_control_fields_are_rechecked(self) -> None:
        mutations = (
            (("control_magic",), 0, "control_magic mismatch"),
            (("control_version",), A5_OWNER_VERSION + 1, "control_version mismatch"),
            (("configure_status",), 1, "configure_status is not success"),
            (("configured_flag",), 0, "configured_flag is not one"),
            (("expected", "total"), A5_WORKERS - 1, "owner.expected.total"),
            (("active", "aic"), A5_AIC_WORKERS - 1, "owner.active.aic"),
            (("discovered", "aiv"), A5_AIV_WORKERS - 1, "owner.discovered.aiv"),
            (("physical_slots_scanned",), A5_PHYSICAL_SUBCORES - 1, "physical_slots_scanned"),
            (("skipped_physical_slots",), 11, "skipped_physical_slots"),
            (("configured_bitmap_count",), A5_WORKERS - 1, "bitmap count"),
            (("configured_complete_mixed_triplets",), A5_AIC_WORKERS - 1, "complete mixed triplets"),
            (("configured_broken_mixed_triplets",), 1, "broken mixed triplet"),
            (("restore_passed",), False, "owner.restore_passed"),
        )
        for keys, value, message in mutations:
            with self.subTest(field=".".join(keys)):
                capture = _submit_pmu_capture()
                target = capture["owner"]
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = self._write(directory, f"bad-owner-{keys[-1]}.json", capture)
                    with self.assertRaisesRegex(ValueError, message):
                        load_capture(path)

    def test_submit_pmu_v4_owner_object_and_bitmap_are_rechecked(self) -> None:
        capture = _submit_pmu_capture()
        del capture["owner"]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "missing-owner.json", capture)
            with self.assertRaisesRegex(ValueError, "schema-v4 owner must be an object"):
                load_capture(path)

        capture = _submit_pmu_capture()
        capture["owner"]["configured_bitmap_words"][3] |= 1 << 31
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "owner-high-bit.json", capture)
            with self.assertRaisesRegex(ValueError, "outside the 108 physical slots"):
                load_capture(path)

        capture = _submit_pmu_capture()
        words = capture["owner"]["configured_bitmap_words"]
        # 用另一个完整 triplet 替换 block15 对应的三个位。bitmap 本身仍是
        # 96/32/64 且 32 组完整 triplet，但不再等于 worker raw 的物理 ID 集合。
        for physical_id in (15, 48, 49):
            words[physical_id // 32] &= ~(1 << (physical_id % 32))
        for physical_id in (16, 50, 51):
            words[physical_id // 32] |= 1 << (physical_id % 32)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "owner-record-set-mismatch.json", capture)
            with self.assertRaisesRegex(ValueError, "absent from the owner bitmap"):
                load_capture(path)

    def test_submit_pmu_v4_host_triplet_count_is_not_blindly_trusted(self) -> None:
        capture = _submit_pmu_capture()
        capture["validation"]["mixed_triplet_matches"] -= 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "host-triplet-count.json", capture)
            with self.assertRaisesRegex(ValueError, "mixed_triplet_matches"):
                load_capture(path)

    def test_submit_pmu_fixed_phase_calls_are_exact_per_worker(self) -> None:
        for phase in ("claim", "efdrain", "materialize", "register"):
            with self.subTest(phase=phase):
                capture = _submit_pmu_capture(phase=phase)
                first = capture["records"][0]
                second = capture["records"][1]
                first["phase_calls"] -= 1
                first["phase_begin_reads"] -= 1
                first["phase_end_reads"] -= 1
                first["shadow_read_segments"] -= 2
                second["phase_calls"] += 1
                second["phase_begin_reads"] += 1
                second["phase_end_reads"] += 1
                second["shadow_read_segments"] += 2
                records = capture["records"]
                capture["summary"] = {
                    "all": _submit_pmu_summary(records),
                    "aic": _submit_pmu_summary(records[:A5_AIC_WORKERS]),
                    "aiv": _submit_pmu_summary(records[A5_AIC_WORKERS:]),
                }
                # 全局 calls、每条 begin/end 与 shadow segment 都保持闭合，
                # 只有逐核固定流 phase 的调用契约被破坏。
                with tempfile.TemporaryDirectory() as directory:
                    path = self._write(
                        directory, f"redistributed-{phase}-calls.json", capture
                    )
                    with self.assertRaisesRegex(ValueError, "phase_calls does not match"):
                        load_capture(path)

    def test_submit_pmu_efdrain_is_an_independent_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "efdrain.json", _submit_pmu_capture(phase="efdrain"))
            result = analyze([path])

        self.assertEqual(result["phase_observation"]["compiled_phase"], "efdrain")
        self.assertEqual(
            result["aggregate"]["groups"]["all"]["phase_calls_per_core"]["median"],
            2 * TASKS_PER_BATCH,
        )

    def test_submit_pmu_none_rejects_nonzero_phase_counters(self) -> None:
        capture = _submit_pmu_capture(phase="none")
        capture["records"][0]["phase_icache_requests"] = 1
        capture["records"][0]["phase_icache_requests_upper_bound"] = 1
        records = capture["records"]
        capture["summary"] = {
            "all": _submit_pmu_summary(records),
            "aic": _submit_pmu_summary(records[:A5_AIC_WORKERS]),
            "aiv": _submit_pmu_summary(records[A5_AIC_WORKERS:]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "none-nonzero-phase.json", capture)
            with self.assertRaisesRegex(ValueError, "zero calls must have zero"):
                load_capture(path)

    def test_submit_pmu_shadow_greater_than_primary_is_rejected_from_raw(self) -> None:
        capture = _submit_pmu_capture()
        record = capture["records"][0]
        record["shadow_whole_icache_requests"] = record["icache_requests"] + 1
        record["shadow_not_greater_than_primary"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "shadow-greater-than-primary.json", capture)
            with self.assertRaisesRegex(ValueError, "shadow whole exceeds"):
                load_capture(path)

    def test_submit_pmu_shadow_miss_greater_than_request_is_rejected(self) -> None:
        capture = _submit_pmu_capture()
        record = capture["records"][0]
        record["shadow_whole_icache_requests"] = record["shadow_whole_icache_misses"] - 1
        records = capture["records"]
        capture["summary"] = {
            "all": _submit_pmu_summary(records),
            "aic": _submit_pmu_summary(records[:A5_AIC_WORKERS]),
            "aiv": _submit_pmu_summary(records[A5_AIC_WORKERS:]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "shadow-miss-greater-than-request.json", capture)
            with self.assertRaisesRegex(ValueError, "shadow I-cache miss > request"):
                load_capture(path)

    def test_submit_pmu_shadow_booleans_loss_and_upper_are_recomputed(self) -> None:
        mutations = (
            ("shadow_matches_primary", True, "shadow_matches_primary disagrees"),
            ("shadow_not_greater_than_primary", False, "shadow_not_greater_than_primary disagrees"),
            ("shadow_request_loss", 999, "shadow_request_loss disagrees"),
            (
                "phase_icache_requests_upper_bound",
                999,
                "upper_bound is not lower plus shadow loss",
            ),
            ("shadow_miss_loss", 999, "shadow_miss_loss disagrees"),
            (
                "phase_icache_misses_upper_bound",
                999,
                "upper_bound is not lower plus shadow loss",
            ),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                capture = _submit_pmu_capture()
                capture["records"][0][field] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = self._write(directory, f"fake-{field}.json", capture)
                    with self.assertRaisesRegex(ValueError, message):
                        load_capture(path)

    def test_submit_pmu_host_shadow_counts_and_deltas_are_recomputed(self) -> None:
        fields = (
            "shadow_primary_match_records",
            "shadow_primary_bounded_records",
            "shadow_request_abs_delta_sum",
            "shadow_request_abs_delta_max",
            "shadow_request_signed_delta_sum",
            "shadow_miss_abs_delta_sum",
            "shadow_miss_abs_delta_max",
            "shadow_miss_signed_delta_sum",
        )
        for field in fields:
            with self.subTest(field=field):
                capture = _submit_pmu_capture()
                capture["validation"][field] += 1
                with tempfile.TemporaryDirectory() as directory:
                    path = self._write(directory, f"tampered-{field}.json", capture)
                    with self.assertRaisesRegex(ValueError, field):
                        load_capture(path)

    def test_submit_pmu_unbalanced_phase_boundary_is_rejected_from_raw(self) -> None:
        capture = _submit_pmu_capture()
        capture["records"][0]["phase_end_reads"] -= 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "boundary-mismatch.json", capture)
            with self.assertRaisesRegex(ValueError, "boundaries do not match calls"):
                load_capture(path)

        capture = _submit_pmu_capture()
        capture["records"][0]["shadow_read_segments"] -= 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "segment-mismatch.json", capture)
            with self.assertRaisesRegex(ValueError, "shadow_read_segments does not match"):
                load_capture(path)

    def test_submit_pmu_record_build_and_phase_ids_are_rechecked(self) -> None:
        for field, value, message in (
            ("build_variant_id", 1, "build_variant_id mismatch"),
            ("compiled_phase_id", 0, "compiled_phase_id mismatch"),
        ):
            with self.subTest(field=field):
                capture = _submit_pmu_capture()
                capture["records"][0][field] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = self._write(directory, f"bad-{field}.json", capture)
                    with self.assertRaisesRegex(ValueError, message):
                        load_capture(path)

    def test_submit_pmu_configuration_and_host_gate_are_rechecked(self) -> None:
        cases = (
            ("build_variant", "swimlane", "configuration.build_variant"),
            ("compiled_phase_id", 0, "compiled phase name/id mismatch"),
            (
                "phase_boundary_observation_included",
                False,
                "phase_boundary_observation_included does not match",
            ),
            (
                "phase_counter_pair_snapshot_atomic",
                True,
                "phase_counter_pair_snapshot_atomic is not false",
            ),
            (
                "phase_shadow_partition_exact_required",
                True,
                "phase_shadow_partition_exact_required does not match",
            ),
            (
                "phase_values_are_running_read_clear_lower_bounds",
                False,
                "phase_values_are_running_read_clear_lower_bounds does not match",
            ),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                capture = _submit_pmu_capture()
                capture["configuration"][field] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = self._write(directory, f"bad-config-{field}.json", capture)
                    with self.assertRaisesRegex(ValueError, message):
                        load_capture(path)

        capture = _submit_pmu_capture()
        capture["configuration"]["selectors"]["cnt5_shadow_icache_miss"] = 0x203
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "bad-shadow-selector.json", capture)
            with self.assertRaisesRegex(ValueError, "cnt5_shadow_icache_miss"):
                load_capture(path)

        capture = _submit_pmu_capture()
        capture["configuration"]["selectors"]["cnt2_scalar_busy"] = 0xDEAD
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "bad-scalar-selector.json", capture)
            with self.assertRaisesRegex(ValueError, "cnt2_scalar_busy"):
                load_capture(path)

        capture = _submit_pmu_capture()
        capture["configuration"]["counter_width_bits"]["programmable"] = 64
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "bad-programmable-counter-width.json", capture)
            with self.assertRaisesRegex(ValueError, "programmable must be 32"):
                load_capture(path)

        capture = _submit_pmu_capture()
        capture["validation"]["phase_boundary_match_records"] -= 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "bad-host-phase-gate.json", capture)
            with self.assertRaisesRegex(ValueError, "phase_boundary_match_records is incomplete"):
                load_capture(path)

        capture = _submit_pmu_capture()
        capture["validation"]["phase_call_shape_match_records"] -= 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "bad-host-phase-call-shape.json", capture)
            with self.assertRaisesRegex(
                ValueError, "phase_call_shape_match_records is incomplete"
            ):
                load_capture(path)

    def test_submit_pmu_rejects_retired_phase_id_three(self) -> None:
        capture = _submit_pmu_capture()
        capture["configuration"]["compiled_phase"] = "wait-for-slot"
        capture["configuration"]["compiled_phase_id"] = 3
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "retired-phase-id-three.json", capture)
            with self.assertRaisesRegex(ValueError, "unsupported configuration.compiled_phase"):
                load_capture(path)

    def test_submit_pmu_rejects_scalar_busy_above_total(self) -> None:
        capture = _submit_pmu_capture()
        capture["records"][0]["scalar_busy"] = capture["records"][0]["total_cycles"] + 1
        records = capture["records"]
        capture["summary"] = {
            "all": _submit_pmu_summary(records),
            "aic": _submit_pmu_summary(records[:A5_AIC_WORKERS]),
            "aiv": _submit_pmu_summary(records[A5_AIC_WORKERS:]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "scalar-busy-above-total.json", capture)
            with self.assertRaisesRegex(ValueError, "scalar_busy > total_cycles"):
                load_capture(path)

    def test_submit_pmu_rejects_zero_total_cycles(self) -> None:
        capture = _submit_pmu_capture()
        capture["records"][0]["total_cycles"] = 0
        capture["records"][0]["scalar_busy"] = 0
        records = capture["records"]
        capture["summary"] = {
            "all": _submit_pmu_summary(records),
            "aic": _submit_pmu_summary(records[:A5_AIC_WORKERS]),
            "aiv": _submit_pmu_summary(records[A5_AIC_WORKERS:]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "zero-total-cycles.json", capture)
            with self.assertRaisesRegex(ValueError, "zero total_cycles"):
                load_capture(path)

    def test_submit_pmu_recomputes_programmable_counter_risk_threshold(self) -> None:
        capture = _submit_pmu_capture()
        capture["records"][0]["vector_busy"] = PROGRAMMABLE_COUNTER_RISK_THRESHOLD
        records = capture["records"]
        capture["summary"] = {
            "all": _submit_pmu_summary(records),
            "aic": _submit_pmu_summary(records[:A5_AIC_WORKERS]),
            "aiv": _submit_pmu_summary(records[A5_AIC_WORKERS:]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "counter-risk-threshold.json", capture)
            with self.assertRaisesRegex(ValueError, "32-bit counter risk threshold"):
                load_capture(path)

    def test_integer_summary_fields_require_exact_equality(self) -> None:
        capture = _submit_pmu_capture()
        for worker_id, record in enumerate(capture["records"]):
            record["total_cycles"] = 10**12 + worker_id
        records = capture["records"]
        capture["summary"] = {
            "all": _submit_pmu_summary(records),
            "aic": _submit_pmu_summary(records[:A5_AIC_WORKERS]),
            "aiv": _submit_pmu_summary(records[A5_AIC_WORKERS:]),
        }
        capture["summary"]["all"]["total_cycles"]["sum"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "inexact-large-integer-summary.json", capture)
            with self.assertRaisesRegex(ValueError, "all.total_cycles.sum"):
                load_capture(path)

    def test_submit_pmu_phase_counter_order_is_rechecked(self) -> None:
        # 两个 ld_dev 不是原子配对快照；边界漂移可使局部 miss 略大于 request，
        # 只要两者分别不超过各自的 Submit whole 就仍是合法 raw 观察。
        capture = _submit_pmu_capture()
        capture["records"][0]["phase_icache_misses"] = (
            capture["records"][0]["phase_icache_requests"] + 5
        )
        capture["records"][0]["phase_icache_misses_upper_bound"] = (
            capture["records"][0]["phase_icache_misses"]
            + capture["records"][0]["shadow_miss_loss"]
        )
        records = capture["records"]
        capture["summary"] = {
            "all": _submit_pmu_summary(records),
            "aic": _submit_pmu_summary(
                [record for record in records if record["role"] == "aic"]
            ),
            "aiv": _submit_pmu_summary(
                [record for record in records if record["role"] == "aiv"]
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "phase-miss-above-request.json", capture)
            load_capture(path)

        mutations = (
            ("phase_icache_misses", 200, "phase counters exceed"),
            ("phase_icache_requests", 2000, "phase counters exceed"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field):
                capture = _submit_pmu_capture()
                capture["records"][0][field] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = self._write(directory, f"bad-{field}.json", capture)
                    with self.assertRaisesRegex(ValueError, message):
                        load_capture(path)

    def test_submit_pmu_phase_summary_tampering_is_rejected(self) -> None:
        capture = _submit_pmu_capture()
        capture["summary"]["aiv"]["phase_icache_misses"]["sum"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "phase-summary-tampered.json", capture)
            with self.assertRaisesRegex(ValueError, "raw summary mismatch"):
                load_capture(path)

        capture = _submit_pmu_capture()
        capture["summary"]["aiv"]["phase_observed_read_clear_ratio"] += 0.01
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, "phase-ratio-tampered.json", capture)
            with self.assertRaisesRegex(ValueError, "phase_observed_read_clear_ratio"):
                load_capture(path)

    def test_different_schema_or_submit_pmu_phase_cannot_be_merged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            legacy = self._write(directory, "legacy.json", _capture())
            claim = self._write(directory, "claim.json", _submit_pmu_capture(0, "claim"))
            none = self._write(directory, "none.json", _submit_pmu_capture(0, "none"))
            with self.assertRaisesRegex(ValueError, "input schema differs"):
                analyze([legacy, claim])
            with self.assertRaisesRegex(ValueError, "observation configuration differs"):
                analyze([claim, none])


if __name__ == "__main__":
    unittest.main()
