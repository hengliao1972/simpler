#!/usr/bin/env python3
# pyright: reportArgumentType=false
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""把 standalone PA 调度器的原始 FDWIC 记录转换为 Perfetto 泳道。

脚本只使用 Python 标准库和调用者给出的本地 JSON，不 import ``simpler_setup``
或仓库外模块。输出遵循 Chrome Trace Event 格式，可直接载入 Perfetto。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TextIO

# 该 standalone converter 刻意保留与 raw ABI 审计对齐的手工换行；
# 全文重排会产生上千行无语义 diff，掩盖真实协议改动。
# fmt: off

# 与真实 PA swimlane converter 使用同一套阶段命名。这里保留 ringbp、
# efdrain 等既有拼写，避免同一阶段在两类泳道中被 Perfetto 分成不同名称。
PHASE_NAMES = {
    "Kernel": "kernel",
    "Alloc": "alloc",
    "Build": "build",
    "DrainWon": "drain_won",
    "Replay": "replay",
    "RingBp": "ringbp",
    "EfDrain": "efdrain",
    "Commit": "commit",
    "Submit": "submit",
    "Materialize": "materialize",
    "PrepareMap": "prepare_map",
    "Claim": "claim",
    "Fanin": "fanin",
    "Register": "register",
    "Atomic": "atomic",
    "Dcci": "dcci",
    "ClockBaseline": "clock_baseline",
    "OrchestrationReplay": "orchestration_replay",
    "FinalDrain": "final_drain",
    "WinnerBuild": "winner_build",
    "AllocComplete": "alloc_complete",
    "SharedRegisterPublishMetadata": "register.publish_metadata",
    "SharedMaterializePublishTaskOutputs": (
        "materialize.publish_shared_output_descriptors"
    ),
    "SharedMaterializePublishTaskOutputsCopy": (
        "materialize.publish_shared_output_descriptors.copy_tensor_descs[GM]"
    ),
    "SharedMaterializePublishTaskOutputsFlush": (
        "materialize.publish_shared_output_descriptors.flush_tensor_descs[DCCI]"
    ),
    "RuntimePlanBuild": "runtime_plan_build",
    # 兼容迁移前已经落盘的 schema-v5 raw；新采集只会写上面的
    # SharedMaterialize* 名称，旧名称仍按其当时的 Register 归属解释。
    "SharedRegisterPublishTaskOutputs": "register.publish_task_outputs",
    "SharedRegisterPublishTaskOutputsCopy": (
        "register.publish_task_outputs.copy"
    ),
    "SharedRegisterPublishTaskOutputsFlush": (
        "register.publish_task_outputs.flush"
    ),
}
LEGACY_LAP_PHASES = {"Alloc", "Build", "Replay"}
V5_PHASES = {
    "OrchestrationReplay",
    "FinalDrain",
    "WinnerBuild",
    "AllocComplete",
    "SharedRegisterPublishMetadata",
    "SharedMaterializePublishTaskOutputs",
    "SharedMaterializePublishTaskOutputsCopy",
    "SharedMaterializePublishTaskOutputsFlush",
    "SharedRegisterPublishTaskOutputs",
    "SharedRegisterPublishTaskOutputsCopy",
    "SharedRegisterPublishTaskOutputsFlush",
    "Dcci",
    "RuntimePlanBuild",
}
# schema-v5 已有区间足以在离线侧取补集；这些 phase 之外的
# Atomic、Kernel、RingBp 等是嵌套或 Overlay，不能再从 Submit 扣一次。
V5_EXCLUSIVE_SUBMIT_PHASES = {
    "EfDrain",
    "Materialize",
    "PrepareMap",
    "Claim",
    "Fanin",
    "Register",
    "WinnerBuild",
    "AllocComplete",
}
# TaskKind/function ABI 固定为五种“类型”，但 shared TensorMap 每个 batch
# 可以有 0..4 组 QK/SF/PV/UP，运行时 task 数不再固定为五个。
TASK_KIND_NAMES = ("Alloc", "QK", "SF", "PV", "UP")
KERNEL_NAMES = {
    function_id: task_kind
    for function_id, task_kind in enumerate(TASK_KIND_NAMES[1:])
}
# 一个物理 mixed block 的三条 runtime lane：AIC、AIV0、AIV1。
LANE_NAMES = {0: "AIC", 1: "AIV0", 2: "AIV1"}

# AICPU producer uses CLOCK_MONOTONIC_RAW while the AICore rows use SYS_CNT.
# They happen to have the same one-nanosecond rate on A5, but their epochs are
# different.  A merged lane is therefore legal only after the eight-sample
# four-timestamp correlation below has closed to at most 50 us.
AICPU_PROCESS_ID = 1_000_000
AICPU_THREAD_ID = 1
AICPU_TASK_THREAD_ID = 2
AICPU_CORRELATION_SAMPLE_COUNT = 8
AICPU_CORRELATION_PRE_SAMPLES = 4
AICPU_CORRELATION_POST_SAMPLES = 4
AICPU_CORRELATION_VERSION = 2
AICPU_CLOCK_SCALE_DEN = 1_000_000_000
AICPU_MAX_ALIGNMENT_ERROR_NS = 50_000
AICPU_PRODUCER_PHASE_NAME = "RuntimePlanProducer"
AICPU_PRODUCER_CLOCK = "aicpu_monotonic_raw_ns"
AICPU_OWNER_PHASE_NAMES = (
    "OwnerSetup",
    "BackendBind",
    "Orchestration",
    "BackendClose",
    "OwnerFinalize",
)
AICPU_TASK_KIND_NAMES = ("Alloc", "QK", "SF", "PV", "UP")
AICPU_TASK_ENGINE_NAMES = ("metadata", "aic", "aiv")
AICPU_MAX_PLAN_PAYLOAD_LINES = 69
AICPU_OPERATION_SCOPES = (
    "owner_setup",
    "backend_bind",
    "task_stage",
    "task_publish",
    "frontier_advance",
    "ready_publish",
    "backend_close",
    "fatal_publish",
)
AICPU_OPERATION_NAMES = (
    "atomic_load_acquire",
    "cache_clean_cvac",
    "cache_discard_civac",
    "barrier_dsb_sy",
    "barrier_isb",
    "gm_store",
    "scalar_work",
    "atomic_store_release",
)
AICPU_OPERATION_TARGETS = (
    "none",
    "request_tensors",
    "request_scalars",
    "context_lens",
    "runtime_plan_control",
    "fatal",
    "closed_task_count",
    "planned_frontier",
    "build_next",
    "build_workers_done",
    "build_release",
    "cell_control",
    "cell_payload",
    "payload_validation",
)


def _scalar_thread_id(lane: int) -> int:
    """返回避开 Perfetto 主线程保留值 0 的 scalar track id。"""
    return lane + 1


def _kernel_thread_id(lane: int) -> int:
    """返回排在全部 scalar 轨道之后的 kernel track id。"""
    return lane + 4


# Atomic raw ABI：auxiliary 存放调用点，flags 低 4 位存放操作类型。这里的
# 数值必须与 standalone C++ AtomicSite/AtomicOp 枚举保持一致；未知值仍会
# 以 site_<id>/op_<id> 完整导出，便于识别版本不匹配，不会伪装成已知操作。
ATOMIC_SITE_NAMES = {
    0: "startup_increment",
    1: "startup_poll",
    2: "fatal_poll",
    3: "fatal_set",
    4: "claim_max",
    5: "fanin_flag_load",
    6: "completion_vend_exchange",
    7: "completion_flag_exchange",
    8: "frontier_initial_load",
    9: "frontier_flag_load",
    10: "frontier_max",
    11: "heap_frontier_load",
    12: "heap_vend_load",
    13: "replay_done_increment",
    14: "replay_done_poll",
    15: "shared_heap_vend_load",
    16: "shared_heap_cursor_load",
    17: "shared_heap_cursor_reserve",
    18: "shared_heap_vend_advance",
    19: "shared_insert_predecessor_poll",
    20: "shared_insert_completion_publish",
    21: "shared_winner_fatal_guard_load",
    22: "shared_metadata_fatal_guard_load",
    23: "shared_output_ref_fanin_output_published_load",
    24: "shared_output_ref_metadata_output_published_load",
    25: "shared_output_ref_fanin_last_writer_load",
    26: "shared_output_ref_metadata_last_writer_load",
    27: "shared_output_ref_last_writer_commit",
    28: "shared_output_writer_reserve",
    29: "shared_output_published_exchange",
    30: "shared_tensormap_lookup_head_load",
    31: "shared_tensormap_lookup_tail_load",
    32: "shared_tensormap_lookup_seq_load",
    33: "shared_tensormap_append_head_load",
    34: "shared_tensormap_append_tail_load",
    35: "shared_tensormap_append_seq_load",
    36: "shared_tensormap_append_seq_reset_exchange",
    37: "shared_tensormap_append_seq_publish_exchange",
    38: "shared_tensormap_append_tail_exchange",
    39: "shared_output_rollback_exchange",
    40: "shared_claim_tournament_local",
    41: "shared_claim_tournament_root",
    42: "shared_build_dispatch_ticket",
    43: "shared_exec_fatal_load",
    44: "shared_exec_fatal_set",
    45: "shared_exec_cell_state_load",
    46: "shared_exec_build_reserve",
    47: "shared_exec_built_publish",
    48: "shared_exec_claim",
    49: "shared_exec_completion_vend_publish",
    50: "shared_exec_completion_flag_publish",
    51: "shared_exec_done_publish",
    52: "shared_exec_drain_arrive",
    53: "shared_exec_drain_release_publish",
    54: "shared_exec_drain_release_poll",
    55: "shared_exec_drain_arrival_poll",
    56: "shared_exec_dispatch_ticket",
    57: "shared_replay_identity_seal",
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
ATOMIC_OP_NAMES = {
    0: "load",
    1: "exchange",
    2: "fetch_add",
    3: "fetch_max",
    4: "compare_exchange",
}

# schema-v3/4 的校验表必须与 standalone C++ 的稳定 AtomicSite 编号一致。
# 0..14 是既有 common/private 站点，15..18 是 shared heap，19/20 是
# shared Register 插入轮次的等待 Load 与返回型 CompareExchange 发布；真实 PA 的 BlockWon
# 不属于本用例，不能为了兼容生产 converter 凭空放宽。
ATOMIC_SITE_OP_IDS = {
    0: 2,
    1: 0,
    2: 0,
    3: 1,
    4: 3,
    5: 0,
    6: 1,
    7: 1,
    8: 0,
    9: 0,
    10: 3,
    11: 0,
    12: 0,
    13: 2,
    14: 0,
    15: 0,
    16: 0,
    17: 2,
    18: 2,
    19: 0,
    20: 4,
    21: 0,
    22: 0,
    23: 0,
    24: 0,
    25: 0,
    26: 0,
    27: 4,
    28: 3,
    29: 1,
    30: 0,
    31: 0,
    32: 0,
    33: 0,
    34: 0,
    35: 0,
    36: 1,
    37: 1,
    38: 1,
    39: 1,
    40: 4,
    41: 4,
    42: 2,
    43: 0,
    44: 4,
    45: 0,
    46: 4,
    47: 4,
    48: 4,
    49: 1,
    50: 1,
    51: 4,
    52: 2,
    53: 1,
    54: 0,
    55: 0,
    56: 2,
    57: 4,
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
}
# 这些发布型调用不消费 atomic 返回的旧值；其余 standalone site 的
# 返回值都参与协议判断。v3 输入必须与源码语义完全一致。
ATOMIC_RESULT_UNUSED_SITE_IDS = {
    0, 3, 6, 7, 13, 39, 49, 53, 59, 68,
}
# common/private 的六类等待 Load 与 shared Register insert-turn Load 可以
# 合并；frontier 扫描和 Claim 即使调用很多次也必须继续保留逐调用记录。
POLL_BATCH_SITE_OP_IDS = {
    1: 0,
    2: 0,
    5: 0,
    11: 0,
    12: 0,
    14: 0,
    19: 0,
    23: 0,
    24: 0,
    54: 0,
    55: 0,
    58: 0,
    60: 0,
    64: 0,
    67: 0,
}
SHARED_REGISTER_ATOMIC_SITE_IDS = {19, 20}
SCHEMA_V5_SHARED_ATOMIC_SITE_IDS = set(range(19, 70))
SHARED_INSERT_TURN_POLL_SITE_ID = 19
SHARED_INSERT_TURN_HANDOFF_SITE_ID = 20
SHARED_OUTPUT_PUBLISHED_POLL_SITE_IDS = {23, 24}
AGGREGATE_ONLY_POLL_SITE_IDS = {SHARED_INSERT_TURN_POLL_SITE_ID}
SHARED_CLAIM_TOURNAMENT_SITE_IDS = {40, 41}
RUNTIME_PLAN_ATOMIC_SITE_IDS = set(range(58, 70))
RUNTIME_PLAN_CELL_CONTROL_SITE_ID = 64
RUNTIME_PLAN_TASK_LOCAL_SITE_IDS = {64, 69}

# ``central_ticket`` 是旧 cross-core raw；AICPU Plan-ahead 同样是全局
# 唯一 Build owner，但不再伪造 Submit/Claim endpoint。两者可以
# 共用 cross-core execution 的离线派生，不能共用 task 身份来源。
CENTRAL_BUILD_TOPOLOGIES = {
    "central_ticket", "aicpu_plan_central_build",
}
STRICT_INSERT_TOPOLOGIES = {
    "all_worker_replay", "aicpu_plan_central_build",
}

ATOMIC_RESULT_USED = 1 << 4
ATOMIC_VALUE_ZERO = 1 << 5
ATOMIC_RETURN_READY = 1 << 6
ATOMIC_POLL_BATCH = 1 << 7
ATOMIC_PAYLOAD_SHIFT = 8
ATOMIC_PAYLOAD_MASK = 0xFFFFFF

# DCCI raw ABI 与 Atomic 独立复用 flags/aux。一次区域原语只生成一条
# 记录；observer 最终导出把 records/core 两次 clean 聚合成一条 terminal
# 记录，因此 call_count 与物理 row 数不能混为一谈。
DCCI_SITE_NAMES = {
    0: "shared_output_ref_fanin_history_invalidate",
    1: "shared_output_ref_writer_history_flush",
    2: "shared_output_rollback_flush",
    3: "shared_output_descriptor_flush",
    4: "shared_region_read_invalidate",
    5: "shared_region_append_invalidate",
    6: "shared_region_append_flush",
    7: "shared_winner_build_descriptor_invalidate",
    8: "observer_trace_export",
    9: "startup_config_invalidate",
    10: "shared_exec_build_source_descriptor_invalidate",
    11: "shared_exec_payload_flush",
    12: "shared_exec_payload_invalidate",
    13: "shared_exec_token_descriptor_invalidate",
    14: "startup_context_lens_invalidate",
    15: "runtime_plan_payload_acquire",
    16: "runtime_plan_storage_ref_acquire",
}
DCCI_OP_NAMES = {
    0: "invalidate",
    1: "clean_out",
}
DCCI_SITE_OP_IDS = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 0,
    5: 0,
    6: 1,
    7: 0,
    8: 1,
    9: 0,
    10: 0,
    11: 1,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
}
DCCI_SHARED_ONLY_SITE_IDS = set(range(8)) | set(range(10, 17))
DCCI_OBSERVER_SITE_ID = 8
DCCI_STARTUP_SITE_ID = 9
DCCI_CONTEXT_LENS_SITE_ID = 14
DCCI_OP_MASK = 0x3
DCCI_TRAILING_DSB = 1 << 2
DCCI_CALL_COUNT_SHIFT = 3
DCCI_CALL_COUNT_MASK = 0xF
DCCI_RESERVED_BIT = 1 << 7
DCCI_LINE_COUNT_SHIFT = 8
DCCI_LINE_COUNT_MASK = 0xFFFFFF


# schema-v3 只由本目录的 standalone producer 生成；其 worker 编号与
# 32 AIC + 64 AIV 的 mixed-block 映射是 raw ABI 的一部分，converter 不再
# 只检查“同一 block/lane 不重复”这个弱条件。
def _standalone_topology(core_id: int) -> tuple[int, int, str]:
    if core_id < 32:
        return core_id, 0, "aic"
    vector_id = core_id - 32
    return vector_id // 2, 1 + vector_id % 2, "aiv"


# 把可转为整数的 raw 标量归一为 int，并在错误中保留精确字段路径。
def _integer(value: Any, label: str) -> int:
    # 这是兼容 JSON 数值/数值字符串的宽松归一，不负责强制原始 JSON 类型必须为 int。
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not an integer: {value!r}") from error


def _ceil_div(numerator: int, denominator: int) -> int:
    """Exact mathematical ceil(numerator / denominator), including negatives."""
    return -((-numerator) // denominator)


def _scaled_floor(value: int, scale_num: int, scale_den: int) -> int:
    return value * scale_num // scale_den


def _scaled_ceil(value: int, scale_num: int, scale_den: int) -> int:
    return _ceil_div(value * scale_num, scale_den)


def _scaled_nearest(value: int, scale_num: int, scale_den: int) -> int:
    # All AICPU timestamps are nonnegative.  Half-up avoids Python's
    # float/banker's-rounding and keeps the conversion reproducible in C++.
    return (value * scale_num + scale_den // 2) // scale_den


def _validate_aicpu_runtime_plan_producer(  # noqa: PLR0912, PLR0915
    data: dict[str, Any],
    metadata: dict[str, Any],
    frequency_hz: int,
    runtime_plan_abi: int,
    *,
    allow_host_cpu_functional: bool,
) -> dict[str, Any] | None:
    """Validate and map the real AICPU Plan owner into the SYS_CNT domain.

    The Host-computed offset fields are retained as evidence but are not
    trusted: every per-sample interval and the final intersection are rebuilt
    from the four causal timestamps using integer arithmetic.
    """

    forbidden_host_timing_fields = (
        "runtime_plan_pipeline_clock_bracket",
        "host_pipeline_e2e",
        "timing_primary_metric",
        "aicore_time_scope",
    )
    present_host_timing_fields = [
        field for field in forbidden_host_timing_fields
        if field in metadata
    ]
    if present_host_timing_fields:
        raise ValueError(
            "Host-derived timing metadata is forbidden by the joint "
            "AICPU/AICore profiling ABI: "
            f"{present_host_timing_fields}; only the AICPU and AICore clock "
            "domains may participate"
        )

    producer_domain = metadata.get("runtime_plan_producer_domain")
    if producer_domain is None:
        raise ValueError(
            "metadata.runtime_plan_producer_domain is required for "
            "joint AICPU/AICore Runtime Plan profiling"
        )
    if producer_domain not in ("aicpu", "host-cpu"):
        raise ValueError(
            "metadata.runtime_plan_producer_domain must be aicpu or host-cpu"
        )
    if producer_domain == "host-cpu" and not allow_host_cpu_functional:
        raise ValueError(
            "joint AICPU/AICore profiling requires "
            "metadata.runtime_plan_producer_domain='aicpu'; host-cpu is "
            "functional-only and requires explicit --host-cpu-functional"
        )
    expected_primary_view = (
        "joint_aicpu_aicore_structure"
        if producer_domain == "aicpu"
        else "host_cpu_functional_structure"
    )
    primary_view = metadata.get("profiling_primary_view")
    if primary_view != expected_primary_view:
        raise ValueError(
            "metadata.profiling_primary_view must be "
            f"{expected_primary_view!r} for producer_domain={producer_domain}"
        )

    warmup_before_pipeline = metadata.get(
        "clock_correlation_warmup_before_pipeline"
    )
    timing_scope = metadata.get("timing_scope")
    performance_representative = metadata.get(
        "performance_representative"
    )
    expected_warmup = producer_domain == "aicpu"
    expected_timing_scope = (
        "calibrated-structural-capture"
        if expected_warmup
        else "host-cpu-functional-capture"
    )
    if (
        type(warmup_before_pipeline) is not bool
        or warmup_before_pipeline is not expected_warmup
    ):
        raise ValueError(
            "metadata.clock_correlation_warmup_before_pipeline must be "
            f"{str(expected_warmup).lower()} for producer_domain={producer_domain}"
        )
    if timing_scope != expected_timing_scope:
        raise ValueError(
            f"metadata.timing_scope must be {expected_timing_scope!r} for "
            f"producer_domain={producer_domain}"
        )
    if type(performance_representative) is not bool or performance_representative:
        raise ValueError(
            "metadata.performance_representative must be false for a "
            "swimlane structural/functional capture"
        )
    orchestrator_phases = data.get("aicpu_orchestrator_phases")
    aicpu_tasks = data.get("aicpu_tasks")
    aicpu_operations = data.get("aicpu_operations")
    scheduler_phases = data.get("aicpu_scheduler_phases")
    operation_summary = metadata.get("aicpu_operation_trace")
    if producer_domain == "host-cpu":
        for field, value in (
            ("aicpu_tasks", aicpu_tasks),
            ("aicpu_operations", aicpu_operations),
            ("aicpu_scheduler_phases", scheduler_phases),
            ("aicpu_orchestrator_phases", orchestrator_phases),
        ):
            if value not in (None, []):
                raise ValueError(
                    f"{field} must be empty for runtime_plan_producer_domain=host-cpu"
                )
        if "aicpu_aicore_clock_correlation" in metadata:
            raise ValueError(
                "metadata.aicpu_aicore_clock_correlation is only valid for "
                "runtime_plan_producer_domain=aicpu"
            )
        if "aicpu_aicore_causal_capture_bracket" in metadata:
            raise ValueError(
                "metadata.aicpu_aicore_causal_capture_bracket is only valid "
                "for runtime_plan_producer_domain=aicpu"
            )
        if operation_summary is not None:
            expected_summary = {
                "enabled": False,
                "records": 0,
                "record_bytes": 0,
                "dropped": 0,
            }
            if operation_summary != expected_summary:
                raise ValueError(
                    "metadata.aicpu_operation_trace must be disabled for "
                    "runtime_plan_producer_domain=host-cpu"
                )
        metadata["profiling_scope"] = "host-cpu-functional-only"
        metadata["joint_profiling"] = False
        return None

    if not isinstance(aicpu_tasks, list):
        raise ValueError("aicpu_tasks must be an array")
    if aicpu_operations is None and operation_summary is None:
        aicpu_operations = []
        metadata["aicpu_operation_trace"] = {
            "enabled": False,
            "records": 0,
            "record_bytes": 0,
            "dropped": 0,
            "legacy_absent": True,
        }
    elif not isinstance(aicpu_operations, list):
        raise ValueError("aicpu_operations must be an array")
    else:
        expected_operation_summary_fields = {
            "enabled", "records", "record_bytes", "dropped"
        }
        if (
            not isinstance(operation_summary, dict)
            or set(operation_summary) != expected_operation_summary_fields
            or operation_summary.get("enabled") is not True
            or _integer(
                operation_summary.get("records"),
                "metadata.aicpu_operation_trace.records",
            ) != len(aicpu_operations)
            or _integer(
                operation_summary.get("record_bytes"),
                "metadata.aicpu_operation_trace.record_bytes",
            ) != 64
            or _integer(
                operation_summary.get("dropped"),
                "metadata.aicpu_operation_trace.dropped",
            ) != 0
            or not aicpu_operations
        ):
            raise ValueError(
                "metadata.aicpu_operation_trace must describe a complete "
                "non-empty 64-byte AICPU operation trace"
            )
    if not isinstance(scheduler_phases, list):
        raise ValueError("aicpu_scheduler_phases must be an array")
    if not isinstance(orchestrator_phases, list):
        raise ValueError("aicpu_orchestrator_phases must be an array")
    if len(orchestrator_phases) != 1:
        raise ValueError(
            "aicpu_orchestrator_phases must contain exactly one real "
            "RuntimePlanProducer phase"
        )
    phase = orchestrator_phases[0]
    if not isinstance(phase, dict):
        raise ValueError("aicpu_orchestrator_phases[0] must be a JSON object")
    expected_phase_fields = {"name", "clock", "begin_ns", "end_ns"}
    if set(phase) != expected_phase_fields:
        raise ValueError(
            "aicpu_orchestrator_phases[0] must contain exactly "
            f"{sorted(expected_phase_fields)}; direct AICore-tick splicing is forbidden"
        )
    if phase.get("name") != AICPU_PRODUCER_PHASE_NAME:
        raise ValueError(
            "aicpu_orchestrator_phases[0].name must be RuntimePlanProducer"
        )
    if phase.get("clock") != AICPU_PRODUCER_CLOCK:
        raise ValueError(
            "aicpu_orchestrator_phases[0].clock must be "
            f"{AICPU_PRODUCER_CLOCK!r}"
        )
    owner_begin_ns = _integer(
        phase.get("begin_ns"), "aicpu_orchestrator_phases[0].begin_ns"
    )
    owner_end_ns = _integer(
        phase.get("end_ns"), "aicpu_orchestrator_phases[0].end_ns"
    )
    if owner_begin_ns <= 0 or owner_end_ns <= owner_begin_ns:
        raise ValueError(
            "aicpu_orchestrator_phases[0] requires a non-empty positive "
            f"interval: begin_ns={owner_begin_ns} end_ns={owner_end_ns}"
        )

    if len(scheduler_phases) != len(AICPU_OWNER_PHASE_NAMES):
        raise ValueError(
            "aicpu_scheduler_phases must contain exactly the five real "
            "RuntimePlanProducer phases"
        )
    normalized_scheduler_phases: list[dict[str, Any]] = []
    previous_phase_end = owner_begin_ns
    for index, expected_name in enumerate(AICPU_OWNER_PHASE_NAMES):
        raw_phase = scheduler_phases[index]
        label = f"aicpu_scheduler_phases[{index}]"
        if not isinstance(raw_phase, dict):
            raise ValueError(f"{label} must be a JSON object")
        if set(raw_phase) != expected_phase_fields:
            raise ValueError(
                f"{label} must contain exactly {sorted(expected_phase_fields)}"
            )
        if raw_phase.get("name") != expected_name:
            raise ValueError(
                f"{label}.name must be {expected_name!r}"
            )
        if raw_phase.get("clock") != AICPU_PRODUCER_CLOCK:
            raise ValueError(
                f"{label}.clock must be {AICPU_PRODUCER_CLOCK!r}"
            )
        phase_begin = _integer(
            raw_phase.get("begin_ns"), f"{label}.begin_ns"
        )
        phase_end = _integer(
            raw_phase.get("end_ns"), f"{label}.end_ns"
        )
        if phase_begin != previous_phase_end or phase_end <= phase_begin:
            raise ValueError(
                "aicpu_scheduler_phases must be an exact non-empty "
                "partition of RuntimePlanProducer"
            )
        normalized_scheduler_phases.append({
            "name": expected_name,
            "raw_clock": AICPU_PRODUCER_CLOCK,
            "raw_begin_ns": phase_begin,
            "raw_end_ns": phase_end,
        })
        previous_phase_end = phase_end
    if previous_phase_end != owner_end_ns:
        raise ValueError(
            "aicpu_scheduler_phases must close exactly at "
            "RuntimePlanProducer.end_ns"
        )

    producer_task_count = _integer(
        metadata.get("runtime_plan_producer_task_count"),
        "metadata.runtime_plan_producer_task_count",
    )
    task_kind_ids = metadata.get("runtime_plan_task_kinds")
    if (
        producer_task_count <= 0
        or len(aicpu_tasks) != producer_task_count
        or not isinstance(task_kind_ids, list)
        or len(task_kind_ids) != producer_task_count
    ):
        raise ValueError(
            "aicpu_tasks must exactly cover "
            "metadata.runtime_plan_producer_task_count"
        )
    expected_task_fields = {
        "task_id",
        "task_kind",
        "function_id",
        "engine",
        "group",
        "output_count",
        "payload_lines",
        "clock",
        "build_begin_ns",
        "begin_end_ns",
        "finish_begin_ns",
        "finish_end_ns",
        "publish_begin_ns",
        "publish_end_ns",
    }
    orchestration_begin_ns = int(
        normalized_scheduler_phases[2]["raw_begin_ns"]
    )
    orchestration_end_ns = int(
        normalized_scheduler_phases[2]["raw_end_ns"]
    )
    backend_closed_ns = int(
        normalized_scheduler_phases[3]["raw_end_ns"]
    )
    normalized_tasks: list[dict[str, Any]] = []
    previous_publish_end = orchestration_begin_ns
    previous_kind = -1
    previous_group = 0
    for index, raw_task in enumerate(aicpu_tasks):
        label = f"aicpu_tasks[{index}]"
        if not isinstance(raw_task, dict):
            raise ValueError(f"{label} must be a JSON object")
        if set(raw_task) != expected_task_fields:
            raise ValueError(
                f"{label} must contain exactly {sorted(expected_task_fields)}"
            )
        task_id = _integer(raw_task.get("task_id"), f"{label}.task_id")
        kind_id = _integer(task_kind_ids[index], f"runtime_plan_task_kinds[{index}]")
        function_id = _integer(
            raw_task.get("function_id"), f"{label}.function_id"
        )
        group = _integer(raw_task.get("group"), f"{label}.group")
        output_count = _integer(
            raw_task.get("output_count"), f"{label}.output_count"
        )
        payload_lines = _integer(
            raw_task.get("payload_lines"), f"{label}.payload_lines"
        )
        if kind_id < 0 or kind_id >= len(AICPU_TASK_KIND_NAMES):
            raise ValueError(f"{label} has invalid task kind {kind_id}")
        expected_group = 0
        if kind_id != 0:
            if kind_id == 1:
                expected_group = (
                    0 if previous_kind == 0 else previous_group + 1
                )
            else:
                expected_group = previous_group
        expected_function = -1 if kind_id == 0 else kind_id - 1
        expected_engine = AICPU_TASK_ENGINE_NAMES[
            0 if kind_id == 0 else 1 if kind_id in (1, 3) else 2
        ]
        expected_outputs = (3, 1, 3, 1, 0)[kind_id]
        timestamps = {
            field: _integer(raw_task.get(field), f"{label}.{field}")
            for field in (
                "build_begin_ns",
                "begin_end_ns",
                "finish_begin_ns",
                "finish_end_ns",
                "publish_begin_ns",
                "publish_end_ns",
            )
        }
        if (
            task_id != index
            or raw_task.get("task_kind") != AICPU_TASK_KIND_NAMES[kind_id]
            or function_id != expected_function
            or raw_task.get("engine") != expected_engine
            or group != expected_group
            or group < 0
            or group >= 4
            or output_count != expected_outputs
            or payload_lines <= 0
            or payload_lines > AICPU_MAX_PLAN_PAYLOAD_LINES
            or raw_task.get("clock") != AICPU_PRODUCER_CLOCK
            or timestamps["build_begin_ns"] < previous_publish_end
            or not (
                orchestration_begin_ns <= timestamps["build_begin_ns"]
                < timestamps["begin_end_ns"]
                < timestamps["finish_begin_ns"]
                < timestamps["finish_end_ns"]
                < timestamps["publish_begin_ns"]
                < timestamps["publish_end_ns"]
                <= backend_closed_ns
            )
            or timestamps["finish_end_ns"] > orchestration_end_ns
            or (
                index + 1 < producer_task_count
                and timestamps["publish_end_ns"] > orchestration_end_ns
            )
            or (
                index + 1 == producer_task_count
                and timestamps["publish_begin_ns"] < orchestration_end_ns
            )
        ):
            raise ValueError(
                f"{label} has invalid identity or callback timeline"
            )
        normalized_tasks.append({
            "task_id": task_id,
            "task_kind": AICPU_TASK_KIND_NAMES[kind_id],
            "task_kind_id": kind_id,
            "function_id": function_id,
            "engine": expected_engine,
            "group": group,
            "output_count": output_count,
            "payload_lines": payload_lines,
            "raw_clock": AICPU_PRODUCER_CLOCK,
            **{f"raw_{field}": value for field, value in timestamps.items()},
        })
        previous_publish_end = timestamps["publish_end_ns"]
        previous_kind = kind_id
        previous_group = group

    expected_operation_fields = {
        "sequence",
        "task_id",
        "scope",
        "operation",
        "target",
        "clock",
        "begin_ns",
        "end_ns",
        "calls",
        "lines",
        "first_target_index",
        "last_target_index",
        "first_value",
        "last_value",
    }
    normalized_operations: list[dict[str, Any]] = []
    previous_operation_end = owner_begin_ns
    for index, raw_operation in enumerate(aicpu_operations):
        label = f"aicpu_operations[{index}]"
        if not isinstance(raw_operation, dict):
            raise ValueError(f"{label} must be a JSON object")
        if set(raw_operation) != expected_operation_fields:
            raise ValueError(
                f"{label} must contain exactly "
                f"{sorted(expected_operation_fields)}"
            )
        sequence = _integer(
            raw_operation.get("sequence"), f"{label}.sequence"
        )
        task_id = _integer(
            raw_operation.get("task_id"), f"{label}.task_id"
        )
        scope = raw_operation.get("scope")
        operation = raw_operation.get("operation")
        target = raw_operation.get("target")
        begin_ns = _integer(
            raw_operation.get("begin_ns"), f"{label}.begin_ns"
        )
        end_ns = _integer(
            raw_operation.get("end_ns"), f"{label}.end_ns"
        )
        calls = _integer(raw_operation.get("calls"), f"{label}.calls")
        lines = _integer(raw_operation.get("lines"), f"{label}.lines")
        first_target_index = _integer(
            raw_operation.get("first_target_index"),
            f"{label}.first_target_index",
        )
        last_target_index = _integer(
            raw_operation.get("last_target_index"),
            f"{label}.last_target_index",
        )
        first_value = _integer(
            raw_operation.get("first_value"), f"{label}.first_value"
        )
        last_value = _integer(
            raw_operation.get("last_value"), f"{label}.last_value"
        )
        task_scoped = scope in {
            "task_stage", "task_publish", "frontier_advance",
            "ready_publish",
        }
        if scope == "owner_setup":
            parent_begin = int(
                normalized_scheduler_phases[0]["raw_begin_ns"]
            )
            parent_end = int(
                normalized_scheduler_phases[0]["raw_end_ns"]
            )
        elif scope == "backend_bind":
            parent_begin = int(
                normalized_scheduler_phases[1]["raw_begin_ns"]
            )
            parent_end = int(
                normalized_scheduler_phases[1]["raw_end_ns"]
            )
        elif scope == "task_stage" and 0 <= task_id < len(normalized_tasks):
            parent_begin = int(
                normalized_tasks[task_id]["raw_finish_begin_ns"]
            )
            parent_end = int(
                normalized_tasks[task_id]["raw_finish_end_ns"]
            )
        elif (
            scope in {"task_publish", "frontier_advance", "ready_publish"}
            and 0 <= task_id < len(normalized_tasks)
        ):
            parent_begin = int(
                normalized_tasks[task_id]["raw_publish_begin_ns"]
            )
            parent_end = int(
                normalized_tasks[task_id]["raw_publish_end_ns"]
            )
        elif scope == "backend_close":
            parent_begin = int(
                normalized_scheduler_phases[3]["raw_begin_ns"]
            )
            parent_end = int(
                normalized_scheduler_phases[3]["raw_end_ns"]
            )
        else:
            parent_begin = 0
            parent_end = 0
        cache_operation = operation in {
            "cache_clean_cvac", "cache_discard_civac"
        }
        cell_target = target in {"cell_control", "cell_payload"}
        operation_target_valid = (
            operation == "atomic_load_acquire"
            and target in {
                "fatal", "closed_task_count", "planned_frontier",
                "build_next", "build_workers_done", "build_release",
                "cell_control",
            }
        ) or (
            cache_operation
            and target not in {"none", "payload_validation"}
        ) or (
            operation in {"barrier_dsb_sy", "barrier_isb"}
            and target == "none"
        ) or (
            operation == "gm_store"
            and target in {
                "fatal", "closed_task_count", "planned_frontier",
                "cell_control", "cell_payload",
            }
        ) or (
            operation == "atomic_store_release"
            and target == "cell_control"
        ) or (
            operation == "scalar_work"
            and target == "payload_validation"
        )
        if (
            sequence != index
            or scope not in AICPU_OPERATION_SCOPES
            or operation not in AICPU_OPERATION_NAMES
            or target not in AICPU_OPERATION_TARGETS
            or raw_operation.get("clock") != AICPU_PRODUCER_CLOCK
            or not operation_target_valid
            or calls <= 0
            or (cache_operation and (lines <= 0 or calls != lines))
            or (not cache_operation and lines != 0)
            or (task_scoped and not (0 <= task_id < len(normalized_tasks)))
            or (not task_scoped and task_id != -1)
            or parent_begin <= 0
            or parent_end <= parent_begin
            or begin_ns < previous_operation_end
            or begin_ns < parent_begin
            or end_ns <= begin_ns
            or end_ns > parent_end
            or (
                cell_target
                and (
                    first_target_index < 0
                    or last_target_index < first_target_index
                )
            )
            or (
                not cell_target
                and (first_target_index != -1 or last_target_index != -1)
            )
            or (
                operation not in {
                    "atomic_load_acquire", "gm_store", "scalar_work",
                    "atomic_store_release",
                }
                and (first_value != 0 or last_value != 0)
            )
        ):
            raise ValueError(
                f"{label} has invalid identity, operation semantics, or "
                "parent containment"
            )
        normalized_operations.append({
            "sequence": sequence,
            "task_id": task_id,
            "scope": scope,
            "operation": operation,
            "target": target,
            "raw_clock": AICPU_PRODUCER_CLOCK,
            "raw_begin_ns": begin_ns,
            "raw_end_ns": end_ns,
            "calls": calls,
            "lines": lines,
            "first_target_index": first_target_index,
            "last_target_index": last_target_index,
            "first_value": first_value,
            "last_value": last_value,
        })
        previous_operation_end = end_ns

    correlation = metadata.get("aicpu_aicore_clock_correlation")
    if not isinstance(correlation, dict):
        raise ValueError(
            "metadata.aicpu_aicore_clock_correlation must be a JSON object"
        )
    expected_correlation_fields = {
        "status",
        "version",
        "pre_samples",
        "post_samples",
        "scale_num",
        "scale_den",
        "offset_lower_tick",
        "offset_upper_tick",
        "offset_mid_tick",
        "alignment_error_tick",
        "maximum_round_trip_ns",
        "maximum_aicore_service_ticks",
        "samples",
    }
    if set(correlation) != expected_correlation_fields:
        raise ValueError(
            "metadata.aicpu_aicore_clock_correlation must contain exactly "
            f"{sorted(expected_correlation_fields)}"
        )
    if correlation.get("status") != "valid":
        raise ValueError(
            "metadata.aicpu_aicore_clock_correlation.status must be 'valid'"
        )
    version = _integer(
        correlation.get("version"),
        "metadata.aicpu_aicore_clock_correlation.version",
    )
    pre_samples = _integer(
        correlation.get("pre_samples"),
        "metadata.aicpu_aicore_clock_correlation.pre_samples",
    )
    post_samples = _integer(
        correlation.get("post_samples"),
        "metadata.aicpu_aicore_clock_correlation.post_samples",
    )
    if (
        version != AICPU_CORRELATION_VERSION
        or pre_samples != AICPU_CORRELATION_PRE_SAMPLES
        or post_samples != AICPU_CORRELATION_POST_SAMPLES
    ):
        raise ValueError(
            "AICPU/AICore correlation requires version=2 and an exact 4+4 "
            "pre/post sample population"
        )
    scale_num = _integer(
        correlation.get("scale_num"),
        "metadata.aicpu_aicore_clock_correlation.scale_num",
    )
    scale_den = _integer(
        correlation.get("scale_den"),
        "metadata.aicpu_aicore_clock_correlation.scale_den",
    )
    if scale_num != frequency_hz or scale_den != AICPU_CLOCK_SCALE_DEN:
        raise ValueError(
            "AICPU/AICore correlation scale must be the raw trace frequency "
            "over 1e9: "
            f"scale={scale_num}/{scale_den} frequency_hz={frequency_hz}"
        )
    samples = correlation.get("samples")
    if not isinstance(samples, list) or len(samples) != AICPU_CORRELATION_SAMPLE_COUNT:
        raise ValueError(
            "metadata.aicpu_aicore_clock_correlation.samples must contain "
            f"exactly {AICPU_CORRELATION_SAMPLE_COUNT} four-timestamp samples"
        )

    expected_sample_fields = {
        "aicpu_send_ns",
        "aicore_receive_tick",
        "aicore_send_tick",
        "aicpu_receive_ns",
        "offset_lower_tick",
        "offset_upper_tick",
        "round_trip_ns",
        "aicore_service_ticks",
        "round",
        "index_in_round",
        "round_nonce",
    }
    normalized_samples: list[dict[str, int]] = []
    intersection_lower: int | None = None
    intersection_upper: int | None = None
    previous_aicpu_receive = 0
    previous_aicore_send = 0
    round_nonces: list[int | None] = [None, None]
    maximum_round_trip = 0
    maximum_service = 0
    for index, raw_sample in enumerate(samples):
        label = f"metadata.aicpu_aicore_clock_correlation.samples[{index}]"
        if not isinstance(raw_sample, dict):
            raise ValueError(f"{label} must be a JSON object")
        if set(raw_sample) != expected_sample_fields:
            raise ValueError(
                f"{label} must contain exactly {sorted(expected_sample_fields)}"
            )
        sample = {
            field: _integer(raw_sample.get(field), f"{label}.{field}")
            for field in expected_sample_fields
        }
        aicpu_send = sample["aicpu_send_ns"]
        aicpu_receive = sample["aicpu_receive_ns"]
        aicore_receive = sample["aicore_receive_tick"]
        aicore_send = sample["aicore_send_tick"]
        if (
            aicpu_send <= 0
            or aicore_receive <= 0
            or aicpu_receive <= aicpu_send
            or aicore_send < aicore_receive
        ):
            raise ValueError(
                f"{label} violates four-timestamp causal order"
            )
        if index != 0 and (
            aicpu_send < previous_aicpu_receive
            or aicore_receive < previous_aicore_send
        ):
            raise ValueError(
                f"{label} is not ordered after the previous correlation sample"
            )
        previous_aicpu_receive = aicpu_receive
        previous_aicore_send = aicore_send

        # offset >= AICore.send - scale*AICPU.receive
        # offset <= AICore.receive - scale*AICPU.send
        # ceil/floor are intentionally asymmetric to preserve a causal integer
        # interval at non-1GHz frequencies.
        computed_lower = aicore_send - _scaled_floor(
            aicpu_receive, scale_num, scale_den
        )
        computed_upper = aicore_receive - _scaled_ceil(
            aicpu_send, scale_num, scale_den
        )
        computed_round_trip = aicpu_receive - aicpu_send
        computed_service = aicore_send - aicore_receive
        expected_round = index // AICPU_CORRELATION_PRE_SAMPLES
        expected_index = index % AICPU_CORRELATION_PRE_SAMPLES
        if computed_lower > computed_upper:
            raise ValueError(f"{label} has an empty causal offset interval")
        if (
            sample["offset_lower_tick"] != computed_lower
            or sample["offset_upper_tick"] != computed_upper
            or sample["round_trip_ns"] != computed_round_trip
            or sample["aicore_service_ticks"] != computed_service
            or sample["round"] != expected_round
            or sample["index_in_round"] != expected_index
            or sample["round_nonce"] <= 0
        ):
            raise ValueError(
                f"{label} derived fields do not match its raw four timestamps "
                "or 4+4 round identity"
            )
        if round_nonces[expected_round] is None:
            round_nonces[expected_round] = sample["round_nonce"]
        elif round_nonces[expected_round] != sample["round_nonce"]:
            raise ValueError(
                f"{label} has a different nonce inside correlation round "
                f"{expected_round}"
            )
        intersection_lower = (
            computed_lower
            if intersection_lower is None
            else max(intersection_lower, computed_lower)
        )
        intersection_upper = (
            computed_upper
            if intersection_upper is None
            else min(intersection_upper, computed_upper)
        )
        normalized_samples.append(sample)
        maximum_round_trip = max(
            maximum_round_trip, computed_round_trip
        )
        maximum_service = max(maximum_service, computed_service)

    assert intersection_lower is not None and intersection_upper is not None
    if intersection_lower > intersection_upper:
        raise ValueError(
            "metadata.aicpu_aicore_clock_correlation samples have an empty "
            "offset intersection"
        )
    if round_nonces[0] == round_nonces[1]:
        raise ValueError(
            "AICPU/AICore pre/post correlation rounds must use different nonces"
        )
    pre_samples_raw = normalized_samples[
        :AICPU_CORRELATION_PRE_SAMPLES
    ]
    post_samples_raw = normalized_samples[
        -AICPU_CORRELATION_POST_SAMPLES:
    ]
    computed_causal_bracket = {
        "aicpu_pre_receive_max_ns": max(
            sample["aicpu_receive_ns"] for sample in pre_samples_raw
        ),
        "aicpu_post_send_min_ns": min(
            sample["aicpu_send_ns"] for sample in post_samples_raw
        ),
        "aicore_pre_send_max_tick": max(
            sample["aicore_send_tick"] for sample in pre_samples_raw
        ),
        "aicore_post_receive_min_tick": min(
            sample["aicore_receive_tick"] for sample in post_samples_raw
        ),
    }
    causal_bracket = metadata.get(
        "aicpu_aicore_causal_capture_bracket"
    )
    if not isinstance(causal_bracket, dict):
        raise ValueError(
            "metadata.aicpu_aicore_causal_capture_bracket must be a JSON object"
        )
    expected_bracket_fields = {
        "derivation",
        "aicpu_clock",
        "aicpu_pre_receive_max_ns",
        "aicpu_post_send_min_ns",
        "aicore_clock",
        "aicore_pre_send_max_tick",
        "aicore_post_receive_min_tick",
    }
    if set(causal_bracket) != expected_bracket_fields:
        raise ValueError(
            "metadata.aicpu_aicore_causal_capture_bracket must contain "
            f"exactly {sorted(expected_bracket_fields)}"
        )
    if (
        causal_bracket.get("derivation")
        != "pre-post-four-timestamp-samples"
        or causal_bracket.get("aicpu_clock") != AICPU_PRODUCER_CLOCK
        or causal_bracket.get("aicore_clock") != "aicore_sys_cnt_tick"
    ):
        raise ValueError(
            "metadata.aicpu_aicore_causal_capture_bracket has invalid "
            "derivation or clock identities"
        )
    declared_causal_bracket = {
        field: _integer(
            causal_bracket.get(field),
            f"metadata.aicpu_aicore_causal_capture_bracket.{field}",
        )
        for field in computed_causal_bracket
    }
    if declared_causal_bracket != computed_causal_bracket:
        raise ValueError(
            "metadata.aicpu_aicore_causal_capture_bracket does not match "
            "the eight raw correlation samples"
        )
    if not (
        computed_causal_bracket["aicpu_pre_receive_max_ns"]
        < owner_begin_ns
        < owner_end_ns
        < computed_causal_bracket["aicpu_post_send_min_ns"]
    ):
        raise ValueError(
            "RuntimePlanProducer is outside the strict AICPU "
            "pre-receive/post-send causal bracket"
        )

    computed_mid = (intersection_lower + intersection_upper) // 2
    computed_error = max(
        computed_mid - intersection_lower,
        intersection_upper - computed_mid,
    )
    declared = {
        field: _integer(
            correlation.get(field),
            f"metadata.aicpu_aicore_clock_correlation.{field}",
        )
        for field in (
            "offset_lower_tick",
            "offset_upper_tick",
            "offset_mid_tick",
            "alignment_error_tick",
        )
    }
    expected = {
        "offset_lower_tick": intersection_lower,
        "offset_upper_tick": intersection_upper,
        "offset_mid_tick": computed_mid,
        "alignment_error_tick": computed_error,
    }
    if declared != expected:
        raise ValueError(
            "metadata.aicpu_aicore_clock_correlation aggregate does not "
            f"match the raw-sample intersection: declared={declared} expected={expected}"
        )
    declared_maximum_round_trip = _integer(
        correlation.get("maximum_round_trip_ns"),
        "metadata.aicpu_aicore_clock_correlation.maximum_round_trip_ns",
    )
    declared_maximum_service = _integer(
        correlation.get("maximum_aicore_service_ticks"),
        "metadata.aicpu_aicore_clock_correlation.maximum_aicore_service_ticks",
    )
    if (
        declared_maximum_round_trip != maximum_round_trip
        or declared_maximum_service != maximum_service
    ):
        raise ValueError(
            "AICPU/AICore correlation maximum RTT/service aggregates do not "
            "match the raw samples"
        )
    maximum_error_tick = _ceil_div(
        frequency_hz * AICPU_MAX_ALIGNMENT_ERROR_NS,
        AICPU_CLOCK_SCALE_DEN,
    )
    if computed_error > maximum_error_tick:
        raise ValueError(
            "AICPU/AICore clock alignment error exceeds the 50 us limit: "
            f"error_tick={computed_error} limit_tick={maximum_error_tick}"
        )

    mapped_begin = (
        _scaled_nearest(owner_begin_ns, scale_num, scale_den)
        + computed_mid
    )
    mapped_end = (
        _scaled_nearest(owner_end_ns, scale_num, scale_den)
        + computed_mid
    )
    if mapped_begin <= 0 or mapped_end <= mapped_begin:
        raise ValueError(
            "mapped RuntimePlanProducer interval is empty or outside the "
            "positive AICore tick domain"
        )
    mapped_aicpu_pre_receive_max = (
        _scaled_nearest(
            computed_causal_bracket["aicpu_pre_receive_max_ns"],
            scale_num,
            scale_den,
        )
        + computed_mid
    )
    mapped_aicpu_post_send_min = (
        _scaled_nearest(
            computed_causal_bracket["aicpu_post_send_min_ns"],
            scale_num,
            scale_den,
        )
        + computed_mid
    )
    if not (
        0 < mapped_aicpu_pre_receive_max
        < mapped_begin
        < mapped_end
        < mapped_aicpu_post_send_min
    ):
        raise ValueError(
            "mapped RuntimePlanProducer does not lie inside the mapped "
            "AICPU causal calibration window"
        )
    mapped_scheduler_phases: list[dict[str, Any]] = []
    for scheduler_phase in normalized_scheduler_phases:
        phase_begin = (
            _scaled_nearest(
                int(scheduler_phase["raw_begin_ns"]),
                scale_num,
                scale_den,
            )
            + computed_mid
        )
        phase_end = (
            _scaled_nearest(
                int(scheduler_phase["raw_end_ns"]),
                scale_num,
                scale_den,
            )
            + computed_mid
        )
        if phase_begin < mapped_begin or phase_end > mapped_end or phase_end <= phase_begin:
            raise ValueError(
                "mapped AICPU owner phase is empty or outside "
                "RuntimePlanProducer"
            )
        mapped_scheduler_phases.append({
            **scheduler_phase,
            "mapped_begin_tick": phase_begin,
            "mapped_end_tick": phase_end,
        })

    mapped_tasks: list[dict[str, Any]] = []
    timestamp_fields = (
        "build_begin_ns",
        "begin_end_ns",
        "finish_begin_ns",
        "finish_end_ns",
        "publish_begin_ns",
        "publish_end_ns",
    )
    for task in normalized_tasks:
        mapped_task = dict(task)
        for field in timestamp_fields:
            mapped_task[f"mapped_{field[:-3]}_tick"] = (
                _scaled_nearest(
                    int(task[f"raw_{field}"]),
                    scale_num,
                    scale_den,
                )
                + computed_mid
            )
        mapped_points = [
            int(mapped_task[f"mapped_{field[:-3]}_tick"])
            for field in timestamp_fields
        ]
        if not (
            mapped_begin
            <= mapped_points[0]
            and all(
                begin < end
                for begin, end in zip(mapped_points, mapped_points[1:])
            )
            and mapped_points[-1]
            <= mapped_end
        ):
            raise ValueError(
                "mapped AICPU TaskPlan detail is empty, reordered, or "
                "outside RuntimePlanProducer"
            )
        mapped_tasks.append(mapped_task)
    mapped_operations: list[dict[str, Any]] = []
    for operation in normalized_operations:
        operation_begin = (
            _scaled_nearest(
                int(operation["raw_begin_ns"]), scale_num, scale_den
            )
            + computed_mid
        )
        operation_end = (
            _scaled_nearest(
                int(operation["raw_end_ns"]), scale_num, scale_den
            )
            + computed_mid
        )
        if (
            operation_begin < mapped_begin
            or operation_end <= operation_begin
            or operation_end > mapped_end
        ):
            raise ValueError(
                "mapped AICPU operation is empty or outside "
                "RuntimePlanProducer"
            )
        mapped_operations.append({
            **operation,
            "mapped_begin_tick": operation_begin,
            "mapped_end_tick": operation_end,
        })
    normalized = {
        "name": AICPU_PRODUCER_PHASE_NAME,
        "raw_clock": AICPU_PRODUCER_CLOCK,
        "raw_begin_ns": owner_begin_ns,
        "raw_end_ns": owner_end_ns,
        "duration_ns": owner_end_ns - owner_begin_ns,
        "mapped_begin_tick": mapped_begin,
        "mapped_end_tick": mapped_end,
        "alignment_error_tick": computed_error,
        "alignment_error_us": computed_error * 1_000_000 / frequency_hz,
        "maximum_alignment_error_tick": maximum_error_tick,
        "mapping": "aicore_tick=round(scale_num*aicpu_ns/scale_den)+offset_mid_tick",
    }
    metadata["aicpu_runtime_plan_producer"] = normalized
    metadata["aicpu_runtime_plan_scheduler_phases"] = (
        mapped_scheduler_phases
    )
    metadata["aicpu_runtime_plan_tasks"] = mapped_tasks
    metadata["aicpu_runtime_plan_operations"] = mapped_operations
    metadata["profiling_scope"] = "aicpu-aicore-joint"
    metadata["joint_profiling"] = True
    metadata["aicpu_aicore_causal_capture_window"] = {
        **computed_causal_bracket,
        "mapped_aicpu_pre_receive_max_tick":
            mapped_aicpu_pre_receive_max,
        "mapped_aicpu_post_send_min_tick":
            mapped_aicpu_post_send_min,
        "alignment_error_tick": computed_error,
    }
    causal_bracket.update(computed_causal_bracket)
    # Normalize integer-compatible JSON strings before the merged metadata is
    # emitted; the original timestamps and every raw sample remain present.
    correlation.update(expected)
    correlation["scale_num"] = scale_num
    correlation["scale_den"] = scale_den
    correlation["version"] = version
    correlation["pre_samples"] = pre_samples
    correlation["post_samples"] = post_samples
    correlation["maximum_round_trip_ns"] = maximum_round_trip
    correlation["maximum_aicore_service_ticks"] = maximum_service
    correlation["samples"] = normalized_samples
    return normalized


RUNTIME_PLAN_PIPELINE_METADATA = {
    "plan-ahead-closed": {
        "launch_order": "plan-sync-before-aicore",
        "producer_ready": "closed",
        "consumer_admission": "closed-only",
        "prefill": 0,
    },
    "streaming-future": {
        "launch_order": "dual-stream-overlap",
        "producer_ready": "prefill",
        "consumer_admission": "ready-future-ticket",
    },
}


def _validate_runtime_plan_atomic_closure(
    pipeline: str,
    build_backend: str,
    build_trace_coverage: str,
    atomic_calls_by_site: dict[int, int],
    task_count: int,
    build_workers: int,
    execute_workers: int,
) -> None:
    """Validate only the Runtime Plan atomics covered by the raw trace.

    Scalar Build owns detailed TraceAtomic records.  Formal SIMT Build uses
    direct VF operations, so its W=4 ticket/arrival closure is proved by the
    separately validated terminal control snapshot; the raw contains only the
    96 Scalar continuation Attach/Wait observations.  Do not invent VF rows.
    """

    if build_backend == "simt":
        if build_trace_coverage != "simt-coarse-direct-state":
            raise ValueError(
                "SIMT Runtime Plan requires "
                "runtime_plan_build_trace_coverage="
                "'simt-coarse-direct-state'"
            )
        # These sites belong exclusively to the untraced VF Build path.  A
        # nonzero raw count would contradict the declared coarse coverage.
        for site in (62, 63, 64, 65, 66, 68, 69):
            actual_calls = atomic_calls_by_site.get(site, 0)
            if actual_calls != 0:
                raise ValueError(
                    "SIMT coarse Runtime Plan raw must not counterfeit VF "
                    f"Atomic rows: site={ATOMIC_SITE_NAMES[site]} "
                    f"actual={actual_calls}"
                )
        minimum_continuation_calls = {
            58: execute_workers,
            60: execute_workers,
            61: execute_workers,
            67: execute_workers,
        }
        if pipeline == "streaming-future":
            # Ready attach and terminal release validation each read the final
            # Plan identity once per Scalar; pre-ready/release polling may add
            # timing-dependent calls.
            minimum_continuation_calls.update({
                58: 2 * execute_workers,
                60: 2 * execute_workers,
                61: 2 * execute_workers,
            })
        for site, minimum_calls in minimum_continuation_calls.items():
            actual_calls = atomic_calls_by_site.get(site, 0)
            if actual_calls < minimum_calls:
                raise ValueError(
                    "SIMT Runtime Plan continuation Atomic closure is "
                    f"incomplete: pipeline={pipeline} "
                    f"site={ATOMIC_SITE_NAMES[site]} "
                    f"actual={actual_calls} minimum={minimum_calls}"
                )
        if atomic_calls_by_site.get(59, 0) != 0:
            raise ValueError(
                "SIMT Runtime Plan success raw must not publish fatal"
            )
        return

    if build_backend != "scalar" or (
        build_trace_coverage != "scalar-task-detail"
    ):
        raise ValueError(
            "Scalar Runtime Plan requires "
            "runtime_plan_build_trace_coverage='scalar-task-detail'"
        )

    exact_plan_atomic_calls = {
        62: task_count + build_workers,
        63: 1,
        65: build_workers,
        66: 1,
        68: 1,
        59: 0,
        69: 1 if task_count != 0 else 0,
    }
    if pipeline == "plan-ahead-closed":
        exact_plan_atomic_calls.update({
            61: execute_workers,
            64: 2 * task_count,
        })
    for site, expected_calls in exact_plan_atomic_calls.items():
        actual_calls = atomic_calls_by_site.get(site, 0)
        if actual_calls != expected_calls:
            raise ValueError(
                "AICPU Plan Atomic call count mismatch: "
                f"pipeline={pipeline} site={ATOMIC_SITE_NAMES[site]} "
                f"actual={actual_calls} expected={expected_calls}"
            )

    minimum_plan_atomic_calls = {
        # Every worker attaches, and the last arrival repeats the terminal
        # fatal/closed/release checks before publishing Build release.
        58: execute_workers + 1,
        60: execute_workers + 1,
        67: execute_workers + 1,
    }
    if pipeline == "streaming-future":
        minimum_plan_atomic_calls.update({
            # Ready attach/release observe frontier, while future tickets can
            # add timing-dependent cell-control polls before publication.
            61: execute_workers,
            64: 2 * task_count,
        })
    for site, minimum_calls in minimum_plan_atomic_calls.items():
        actual_calls = atomic_calls_by_site.get(site, 0)
        if actual_calls < minimum_calls:
            raise ValueError(
                "AICPU Plan Atomic poll closure is incomplete: "
                f"pipeline={pipeline} site={ATOMIC_SITE_NAMES[site]} "
                f"actual={actual_calls} minimum={minimum_calls}"
            )


def _derive_v4_task_kinds(  # noqa: PLR0912
    submit_semantics: dict[tuple[int, int], tuple[bool, bool]],
    num_cores: int,
    submit_topology: str = "all_worker_replay",
) -> dict[int, int]:
    """从 Submit 已有的 Alloc 标记恢复动态 task 类型流。

    schema-v5 的 ``Submit.auxiliary`` 已逐核记录 ``is_alloc``，因此无需给
    设备 raw 再增加 task-kind 字段。旧 ``all_worker_replay`` 要求每核具有
    完全一致的连续 task 流；``central_ticket`` 则要求全局 task 0..N-1
    恰好各出现一次。两种拓扑随后使用同一 PA batch 形状门禁：
    ``Alloc + 0..4 × (QK,SF,PV,UP)``。返回稳定 TaskKind 编号：Alloc=0，
    QK/SF/PV/UP=1..4。
    """

    if num_cores <= 0:
        raise ValueError(f"schema-v5 task plan requires positive num_cores, got {num_cores}")
    if submit_topology not in ("all_worker_replay", "central_ticket"):
        raise ValueError(
            f"unsupported schema-v5 submit_topology: {submit_topology!r}"
        )

    task_ids_by_core: dict[int, list[int]] = {
        core_id: [] for core_id in range(num_cores)
    }
    for (core_id, task_id) in submit_semantics:
        if core_id not in task_ids_by_core:
            raise ValueError(
                f"schema-v5 Submit task plan has out-of-range core {core_id}"
            )
        if task_id < 0:
            raise ValueError(
                f"schema-v5 Submit task plan has negative task_id {task_id}"
            )
        task_ids_by_core[core_id].append(task_id)

    reference_task_ids: list[int] | None = None
    if submit_topology == "central_ticket":
        owners_by_task: dict[int, int] = {}
        for (core_id, task_id) in submit_semantics:
            previous_owner = owners_by_task.setdefault(task_id, core_id)
            if previous_owner != core_id:
                raise ValueError(
                    "schema-v5 central-ticket task has multiple Submit owners: "
                    f"task={task_id} owners={previous_owner},{core_id}"
                )
        reference_task_ids = sorted(owners_by_task)
        if (
            not reference_task_ids
            or reference_task_ids
            != list(range(reference_task_ids[-1] + 1))
        ):
            raise ValueError(
                "schema-v5 central-ticket Submit task IDs must cover global "
                f"0..N-1 exactly once: task_ids={reference_task_ids}"
            )
    else:
        for core_id in range(num_cores):
            task_ids = sorted(task_ids_by_core[core_id])
            if reference_task_ids is None:
                if not task_ids or task_ids != list(range(task_ids[-1] + 1)):
                    raise ValueError(
                        "schema-v5 Submit task IDs must be contiguous 0..N-1 on every "
                        f"core: core={core_id} task_ids={task_ids}"
                    )
                reference_task_ids = task_ids
            elif task_ids != reference_task_ids:
                raise ValueError(
                    "schema-v5 Submit task IDs differ across cores: "
                    f"core={core_id} task_ids={task_ids}"
                )

    assert reference_task_ids is not None
    alloc_by_task: dict[int, bool] = {}
    for task_id in reference_task_ids:
        markers = {
            semantics[1]
            for (core_id, observed_task_id), semantics in submit_semantics.items()
            if observed_task_id == task_id
        }
        if len(markers) != 1:
            raise ValueError(
                "schema-v5 Submit Alloc marker differs across cores for "
                f"task {task_id}"
            )
        alloc_by_task[task_id] = markers.pop()

    alloc_task_ids = [
        task_id for task_id in reference_task_ids if alloc_by_task[task_id]
    ]
    if not alloc_task_ids or alloc_task_ids[0] != 0:
        raise ValueError("schema-v5 dynamic task plan must begin with task 0 Alloc")

    task_kind_by_id: dict[int, int] = {}
    interval_ends = [*alloc_task_ids[1:], len(reference_task_ids)]
    for alloc_task_id, interval_end in zip(alloc_task_ids, interval_ends):
        interval_length = interval_end - alloc_task_id
        payload_tasks = interval_length - 1
        if payload_tasks % 4 != 0 or not 0 <= payload_tasks // 4 <= 4:
            raise ValueError(
                "schema-v5 dynamic batch must contain Alloc plus 0..4 complete "
                "QK/SF/PV/UP groups: "
                f"alloc_task={alloc_task_id} interval_length={interval_length}"
            )
        task_kind_by_id[alloc_task_id] = 0
        for offset in range(1, interval_length):
            task_kind_by_id[alloc_task_id + offset] = 1 + (offset - 1) % 4

    if set(task_kind_by_id) != set(reference_task_ids):
        raise AssertionError("schema-v5 dynamic task plan derivation is incomplete")
    return task_kind_by_id


def _runtime_plan_task_kinds_from_metadata(
    raw_task_kinds: Any,
    task_count: int,
) -> dict[int, int]:
    """Validate the post-run Runtime Plan identity exported by the Host.

    This metadata is the only per-task identity available for formal SIMT,
    whose VF intentionally has no Scalar Materialize/Register/Fanin trace.
    Reuse the PA batch-shape validator, but never turn the result into fake raw
    owners or child spans.
    """

    if not isinstance(raw_task_kinds, list):
        raise ValueError(
            "metadata.runtime_plan_task_kinds must be an array"
        )
    task_kinds = [
        _integer(
            kind,
            f"metadata.runtime_plan_task_kinds[{task_id}]",
        )
        for task_id, kind in enumerate(raw_task_kinds)
    ]
    if len(task_kinds) != task_count:
        raise ValueError(
            "metadata.runtime_plan_task_kinds length must equal "
            "metadata.runtime_plan_task_count: "
            f"length={len(task_kinds)} task_count={task_count}"
        )
    if any(kind not in range(5) for kind in task_kinds):
        raise ValueError(
            "metadata.runtime_plan_task_kinds values must be in [0, 4]"
        )
    submit_semantics = {
        (0, task_id): (True, kind == 0)
        for task_id, kind in enumerate(task_kinds)
    }
    derived = _derive_v4_task_kinds(
        submit_semantics, 1, "central_ticket"
    )
    declared = {
        task_id: kind for task_id, kind in enumerate(task_kinds)
    }
    if declared != derived:
        raise ValueError(
            "metadata.runtime_plan_task_kinds does not match the PA "
            "Alloc + QK/SF/PV/UP batch shape"
        )
    return declared


def _validate_runtime_plan_terminal_metadata(
    raw_terminal: Any,
    task_count: int,
    build_workers: int,
) -> dict[str, int]:
    if not isinstance(raw_terminal, dict):
        raise ValueError("metadata.runtime_plan_terminal must be an object")
    expected = {
        "planned_frontier": task_count,
        "closed_task_count": task_count,
        "build_next": task_count + build_workers,
        "build_workers_done": build_workers,
        "build_release": task_count,
        "fatal": 0,
    }
    if set(raw_terminal) != set(expected):
        raise ValueError(
            "metadata.runtime_plan_terminal must contain exactly "
            f"{sorted(expected)}"
        )
    terminal = {
        field: _integer(
            raw_terminal.get(field),
            f"metadata.runtime_plan_terminal.{field}",
        )
        for field in expected
    }
    for field, expected_value in expected.items():
        actual_value = terminal[field]
        if actual_value != expected_value:
            raise ValueError(
                "metadata.runtime_plan_terminal does not close for the "
                f"declared Build population: field={field} "
                f"actual={actual_value} expected={expected_value}"
            )
    return terminal


# 读取 raw JSON，校验十列结构、字段范围与可转整数值，并返回规范化视图。
def _load_and_validate(  # noqa: PLR0912, PLR0915
    input_path: Path,
    *,
    allow_host_cpu_functional: bool = False,
) -> tuple[int, int, list[tuple[Any, ...]], dict[tuple[int, int], int], int, dict[str, Any]]:
    # raw 文件沿用真实 l2_swimlane_records.json 的十列 fdwic_events ABI：
    # core、block、lane、task、func、phase、start、end、flags、aux。
    with input_path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, dict):
        raise ValueError("capture root must be a JSON object")

    # 先验证顶层 schema 和时钟元数据；时钟频率是 cycle 转时间的唯一依据，
    # 不允许由 converter 根据平台名称猜测。
    level = _integer(data.get("l2_swimlane_level"), "l2_swimlane_level")
    if level not in (1, 2, 3, 4):
        raise ValueError(f"unsupported l2_swimlane_level: {level}")
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    frequency_hz = _integer(metadata.get("clock_freq_hz"), "metadata.clock_freq_hz")
    if frequency_hz <= 0:
        raise ValueError("metadata.clock_freq_hz must be positive")
    # v1 是旧 raw，Claim flags 只有 winner bit；v2 追加 attempted bit；
    # v3 再加入精确计数 PollBatch；v5 追加排他父区间、真实尾动作 span
    # 以及 Materialize→task outputs 与 Register→metadata detail。迁移前
    # 已落盘的 v5 Register→metadata→task outputs 仍只读兼容。v4 raw 不再接受，
    # 避免缺少 task-outputs 边界的旧采集被伪装成新细分。
    # 不认识的新版本直接拒绝，避免把新 flags 按旧语义误读。
    trace_schema_version = _integer(metadata.get("trace_schema_version", 1), "metadata.trace_schema_version")
    if trace_schema_version not in (1, 2, 3, 5):
        raise ValueError(f"unsupported metadata.trace_schema_version: {trace_schema_version}")
    if trace_schema_version == 3 and level != 4:
        raise ValueError("metadata.trace_schema_version=3 requires l2_swimlane_level=4")
    if trace_schema_version == 5 and level not in (1, 4):
        raise ValueError(
            "metadata.trace_schema_version=5 requires l2_swimlane_level=1 or 4"
        )
    tensormap_mode = metadata.get("tensormap_mode")
    submit_topology = metadata.get(
        "submit_topology", "all_worker_replay"
    )
    runtime_plan_build_backend: str | None = None
    runtime_plan_build_workers = 0
    runtime_plan_execute_workers = 0
    runtime_plan_build_trace_coverage: str | None = None
    runtime_plan_task_count: int | None = None
    runtime_plan_task_kinds: dict[int, int] | None = None
    runtime_plan_abi = 0
    inferred_legacy_runtime_plan_identity = False
    if trace_schema_version == 5:
        if tensormap_mode not in ("private", "shared"):
            raise ValueError(
                "metadata.tensormap_mode must be private or shared for "
                "trace_schema_version=5"
            )
        if submit_topology not in (
            "all_worker_replay", "central_ticket",
            "aicpu_plan_central_build",
        ):
            raise ValueError(
                "metadata.submit_topology must be all_worker_replay or "
                "central_ticket or aicpu_plan_central_build for "
                "trace_schema_version=5"
            )
        if (
            submit_topology in CENTRAL_BUILD_TOPOLOGIES
            and tensormap_mode != "shared"
        ):
            raise ValueError(
                "central-build submit_topology requires shared "
                "TensorMap mode"
            )
        if submit_topology == "aicpu_plan_central_build":
            pipeline = metadata.get("pipeline")
            inferred_legacy_policy = pipeline is None
            runtime_plan_abi_raw = metadata.get("runtime_plan_abi")
            runtime_plan_abi = (
                2
                if runtime_plan_abi_raw is None
                else _integer(
                    runtime_plan_abi_raw,
                    "metadata.runtime_plan_abi",
                )
            )
            if runtime_plan_abi <= 0:
                raise ValueError(
                    "metadata.runtime_plan_abi must be positive"
                )
            if runtime_plan_abi_raw is None or runtime_plan_abi < 3:
                raise ValueError(
                    "joint AICPU/AICore profiling requires explicit "
                    "metadata.runtime_plan_abi>=3; legacy ABI2 AICore-only "
                    "captures are not accepted"
                )
            if pipeline is None:
                # Legacy central-build captures predate the policy field and
                # came only from the closed Plan-ahead path.  Preserve their
                # replayability, but never infer a missing field as streaming.
                if runtime_plan_abi >= 3:
                    raise ValueError(
                        "metadata.pipeline is required for Runtime Plan "
                        "ABI v3 and later"
                    )
                policy_fields = (
                    "launch_order", "producer_ready",
                    "consumer_admission", "prefill",
                )
                partial_fields = [
                    field for field in policy_fields
                    if field in metadata
                ]
                if partial_fields:
                    raise ValueError(
                        "legacy AICPU Plan capture has partial pipeline "
                        f"metadata without metadata.pipeline: {partial_fields}"
                    )
                pipeline = "plan-ahead-closed"
                metadata["pipeline"] = pipeline
                metadata.update(
                    RUNTIME_PLAN_PIPELINE_METADATA[pipeline]
                )
                metadata["inferred_legacy_policy"] = True
            if pipeline not in RUNTIME_PLAN_PIPELINE_METADATA:
                raise ValueError(
                    "metadata.pipeline must be plan-ahead-closed or "
                    "streaming-future for AICPU Plan central build"
                )
            if (
                not inferred_legacy_policy
                and (
                    runtime_plan_abi_raw is None
                    or runtime_plan_abi < 3
                )
            ):
                raise ValueError(
                    "explicit metadata.pipeline requires "
                    "metadata.runtime_plan_abi>=3"
                )
            metadata["runtime_plan_abi"] = runtime_plan_abi
            expected_pipeline_metadata = (
                RUNTIME_PLAN_PIPELINE_METADATA[str(pipeline)]
            )
            for field in (
                "launch_order", "producer_ready",
                "consumer_admission",
            ):
                expected_value = expected_pipeline_metadata[field]
                actual_value = metadata.get(field)
                if actual_value != expected_value:
                    raise ValueError(
                        f"metadata.{field}={actual_value!r} does not match "
                        f"pipeline={pipeline!r}; expected {expected_value!r}"
                    )
            prefill = _integer(
                metadata.get("prefill"), "metadata.prefill"
            )
            if pipeline == "plan-ahead-closed":
                if prefill != 0:
                    raise ValueError(
                        "metadata.prefill must be 0 for "
                        "pipeline='plan-ahead-closed'"
                    )
            elif prefill <= 0:
                raise ValueError(
                    "metadata.prefill must be positive for "
                    "pipeline='streaming-future'"
                )
            metadata["prefill"] = prefill

            identity_fields = (
                "runtime_plan_build_backend",
                "runtime_plan_build_workers",
                "runtime_plan_execute_workers",
                "runtime_plan_build_trace_coverage",
                "runtime_plan_producer_task_count",
                "runtime_plan_task_count",
                "runtime_plan_task_kinds",
                "runtime_plan_terminal",
            )
            if inferred_legacy_policy:
                # ABI2 historical captures predate both policy and backend
                # identity.  They came only from Scalar W=96; reject partial
                # modern metadata rather than guessing a mixed contract.
                present_identity_fields = [
                    field for field in identity_fields
                    if field in metadata
                ]
                if present_identity_fields:
                    raise ValueError(
                        "legacy ABI2 AICPU Plan capture has partial modern "
                        "Runtime Plan identity: "
                        f"{present_identity_fields}"
                    )
                runtime_plan_build_backend = "scalar"
                runtime_plan_build_workers = 96
                runtime_plan_execute_workers = 96
                runtime_plan_build_trace_coverage = (
                    "scalar-task-detail"
                )
                inferred_legacy_runtime_plan_identity = True
                metadata.update({
                    "runtime_plan_build_backend": "scalar",
                    "runtime_plan_build_workers": 96,
                    "runtime_plan_execute_workers": 96,
                    "runtime_plan_build_trace_coverage":
                        "scalar-task-detail",
                    "inferred_legacy_runtime_plan_identity": True,
                })
            else:
                runtime_plan_build_backend_raw = metadata.get(
                    "runtime_plan_build_backend"
                )
                if runtime_plan_build_backend_raw not in (
                    "scalar", "simt"
                ):
                    raise ValueError(
                        "metadata.runtime_plan_build_backend must be "
                        "scalar or simt"
                    )
                runtime_plan_build_backend = str(
                    runtime_plan_build_backend_raw
                )
                runtime_plan_build_workers = _integer(
                    metadata.get("runtime_plan_build_workers"),
                    "metadata.runtime_plan_build_workers",
                )
                expected_build_workers = (
                    4
                    if runtime_plan_build_backend == "simt"
                    else 96
                )
                if runtime_plan_build_workers != expected_build_workers:
                    raise ValueError(
                        "metadata.runtime_plan_build_workers does not "
                        "match metadata.runtime_plan_build_backend: "
                        f"backend={runtime_plan_build_backend} "
                        f"actual={runtime_plan_build_workers} "
                        f"expected={expected_build_workers}"
                    )
                runtime_plan_execute_workers = _integer(
                    metadata.get("runtime_plan_execute_workers"),
                    "metadata.runtime_plan_execute_workers",
                )
                if runtime_plan_execute_workers != 96:
                    raise ValueError(
                        "metadata.runtime_plan_execute_workers must be 96"
                    )
                expected_trace_coverage = (
                    "simt-coarse-direct-state"
                    if runtime_plan_build_backend == "simt"
                    else "scalar-task-detail"
                )
                runtime_plan_build_trace_coverage_raw = metadata.get(
                    "runtime_plan_build_trace_coverage"
                )
                if (
                    runtime_plan_build_trace_coverage_raw
                    != expected_trace_coverage
                ):
                    raise ValueError(
                        "metadata.runtime_plan_build_trace_coverage does "
                        "not match metadata.runtime_plan_build_backend: "
                        f"actual={runtime_plan_build_trace_coverage_raw!r} "
                        f"expected={expected_trace_coverage!r}"
                    )
                runtime_plan_build_trace_coverage = (
                    expected_trace_coverage
                )
                runtime_plan_task_count = _integer(
                    metadata.get("runtime_plan_task_count"),
                    "metadata.runtime_plan_task_count",
                )
                if runtime_plan_task_count <= 0:
                    raise ValueError(
                        "metadata.runtime_plan_task_count must be positive"
                    )
                runtime_plan_producer_task_count = _integer(
                    metadata.get("runtime_plan_producer_task_count"),
                    "metadata.runtime_plan_producer_task_count",
                )
                if (
                    runtime_plan_producer_task_count
                    != runtime_plan_task_count
                ):
                    raise ValueError(
                        "metadata.runtime_plan_producer_task_count must "
                        "equal the direct-state runtime_plan_task_count: "
                        f"producer={runtime_plan_producer_task_count} "
                        f"direct_state={runtime_plan_task_count}"
                    )
                runtime_plan_task_kinds = (
                    _runtime_plan_task_kinds_from_metadata(
                        metadata.get("runtime_plan_task_kinds"),
                        runtime_plan_task_count,
                    )
                )
                metadata["runtime_plan_task_kinds"] = [
                    runtime_plan_task_kinds[task_id]
                    for task_id in range(runtime_plan_task_count)
                ]
                metadata["runtime_plan_terminal"] = (
                    _validate_runtime_plan_terminal_metadata(
                        metadata.get("runtime_plan_terminal"),
                        runtime_plan_task_count,
                        runtime_plan_build_workers,
                    )
                )
    elif tensormap_mode is not None:
        raise ValueError(
            "metadata.tensormap_mode is only valid for trace_schema_version=5"
        )
    elif "submit_topology" in metadata:
        raise ValueError(
            "metadata.submit_topology is only valid for trace_schema_version=5"
        )
    metadata_writer_tasks_raw = metadata.get(
        "shared_metadata_writer_tasks"
    )
    metadata_prefix_tasks_raw = metadata.get(
        "shared_metadata_prefix_tasks"
    )
    shared_metadata_ordering = metadata.get(
        "shared_metadata_ordering", "global_writer_chain"
    )
    shared_metadata_writer_tasks: tuple[int, ...] = ()
    shared_metadata_prefix_tasks: tuple[int, ...] = ()
    if trace_schema_version == 5 and tensormap_mode == "shared":
        if shared_metadata_ordering not in (
            "global_writer_chain", "per_symbol_dag"
        ):
            raise ValueError(
                "metadata.shared_metadata_ordering must be "
                "global_writer_chain or per_symbol_dag for shared schema-v5"
            )
        metadata["shared_metadata_ordering"] = shared_metadata_ordering
        if not isinstance(metadata_writer_tasks_raw, list):
            raise ValueError(
                "metadata.shared_metadata_writer_tasks must be an array "
                "for shared schema-v5"
            )
        normalized_writer_tasks = tuple(
            _integer(
                task_id,
                f"metadata.shared_metadata_writer_tasks[{index}]",
            )
            for index, task_id in enumerate(metadata_writer_tasks_raw)
        )
        if any(task_id < 0 for task_id in normalized_writer_tasks) or any(
            left >= right
            for left, right in zip(
                normalized_writer_tasks,
                normalized_writer_tasks[1:],
            )
        ):
            raise ValueError(
                "metadata.shared_metadata_writer_tasks must be strictly "
                "increasing nonnegative task ids"
            )
        shared_metadata_writer_tasks = normalized_writer_tasks
        metadata["shared_metadata_writer_tasks"] = list(
            normalized_writer_tasks
        )
        if (
            shared_metadata_ordering == "per_symbol_dag"
            and metadata_prefix_tasks_raw is not None
        ):
            raise ValueError(
                "metadata.shared_metadata_prefix_tasks must be absent when "
                "shared_metadata_ordering=per_symbol_dag"
            )
        if (
            shared_metadata_ordering == "global_writer_chain"
            and not isinstance(metadata_prefix_tasks_raw, list)
        ):
            raise ValueError(
                "metadata.shared_metadata_prefix_tasks must be an array "
                "for shared schema-v5 global_writer_chain"
            )
        if shared_metadata_ordering == "global_writer_chain":
            normalized_prefix_tasks = tuple(
                _integer(
                    task_id,
                    f"metadata.shared_metadata_prefix_tasks[{index}]",
                )
                for index, task_id in enumerate(metadata_prefix_tasks_raw)
            )
            if any(task_id < 0 for task_id in normalized_prefix_tasks) or any(
                left >= right
                for left, right in zip(
                    normalized_prefix_tasks,
                    normalized_prefix_tasks[1:],
                )
            ):
                raise ValueError(
                    "metadata.shared_metadata_prefix_tasks must be strictly "
                    "increasing nonnegative task ids"
                )
            shared_metadata_prefix_tasks = normalized_prefix_tasks
            metadata["shared_metadata_prefix_tasks"] = list(
                normalized_prefix_tasks
            )
    elif metadata_writer_tasks_raw is not None:
        raise ValueError(
            "metadata.shared_metadata_writer_tasks is only valid for "
            "shared schema-v5"
        )
    elif metadata_prefix_tasks_raw is not None:
        raise ValueError(
            "metadata.shared_metadata_prefix_tasks is only valid for "
            "shared schema-v5"
        )
    elif "shared_metadata_ordering" in metadata:
        raise ValueError(
            "metadata.shared_metadata_ordering is only valid for shared "
            "schema-v5"
        )
    num_cores = _integer(metadata.get("num_cores"), "metadata.num_cores")
    if num_cores <= 0:
        raise ValueError("metadata.num_cores must be positive")
    if (
        submit_topology == "aicpu_plan_central_build"
        and num_cores != runtime_plan_execute_workers
    ):
        raise ValueError(
            "metadata.num_cores must equal "
            "metadata.runtime_plan_execute_workers for AICPU Plan "
            f"central build: num_cores={num_cores} "
            f"execute_workers={runtime_plan_execute_workers}"
        )
    core_types = metadata.get("core_types")
    if not isinstance(core_types, list) or len(core_types) != num_cores:
        raise ValueError("metadata.core_types length must equal metadata.num_cores")
    if trace_schema_version >= 3:
        if num_cores > 96:
            raise ValueError("schema-v3+ standalone metadata.num_cores must not exceed 96")
        for core_id, core_type in enumerate(core_types):
            expected_type = _standalone_topology(core_id)[2]
            if core_type != expected_type:
                raise ValueError(
                    f"metadata.core_types[{core_id}]={core_type!r} does not match "
                    f"standalone topology {expected_type!r}"
                )
    winner_workload = metadata.get("winner_workload")
    if winner_workload is not None:
        if not isinstance(winner_workload, dict):
            raise ValueError("metadata.winner_workload must be a JSON object")
        workload_mode = winner_workload.get("mode")
        if workload_mode not in ("scalar-nop", "real-compute"):
            raise ValueError("metadata.winner_workload.mode must be scalar-nop or real-compute")
        workload_counts = winner_workload.get("counts")
        if not isinstance(workload_counts, dict):
            raise ValueError("metadata.winner_workload.counts must be a JSON object")
        normalized_counts: dict[str, int] = {}
        for kind in ("qk", "sf", "pv", "up"):
            value = _integer(
                workload_counts.get(kind), f"metadata.winner_workload.counts.{kind}"
            )
            if value < 0 or (workload_mode == "real-compute" and value == 0):
                raise ValueError(
                    f"metadata.winner_workload.counts.{kind} is invalid for {workload_mode}"
                )
            normalized_counts[kind] = value
        expected_unit = (
            "complete_128x128_engine_pipeline_iteration"
            if workload_mode == "real-compute"
            else "scalar_nop_instruction"
        )
        if winner_workload.get("unit") != expected_unit:
            raise ValueError(
                f"metadata.winner_workload.unit must be {expected_unit!r} for {workload_mode}"
            )
        # input_pattern 是 real-compute 布局诊断新增的可选元数据。旧 schema-v2
        # 文件没有该字段，仍保持可读；新采集若给出则必须与 workload 模式一致。
        input_pattern = winner_workload.get("input_pattern")
        if input_pattern is not None:
            valid_patterns = (
                {"constant", "layout-diagnostic"}
                if workload_mode == "real-compute"
                else {"none"}
            )
            if input_pattern not in valid_patterns:
                raise ValueError(
                    "metadata.winner_workload.input_pattern is invalid for "
                    f"{workload_mode}"
                )
        engine_mapping = winner_workload.get("engine_mapping")
        if workload_mode == "real-compute":
            expected_mapping = {
                "qk": "cube_matmul",
                "sf": "vector_add",
                "pv": "cube_matmul",
                "up": "vector_mul",
            }
            if engine_mapping != expected_mapping:
                raise ValueError("metadata.winner_workload.engine_mapping is invalid")
        elif engine_mapping is not None:
            raise ValueError("scalar-nop metadata.winner_workload.engine_mapping must be null")
        # 后续 merged 顶层与 instant event 使用经过整数归一的同一份配置。
        winner_workload["counts"] = normalized_counts

    rows = data.get("fdwic_events")
    if not isinstance(rows, list) or not rows:
        raise ValueError("fdwic_events must be a non-empty array")

    # Perfetto metadata 需要从 (block, lane) 找回稳定的 core 编号；同一 lane
    # 若在 raw 中映射到两个 core，说明采集已损坏，不能继续生成误导性泳道。
    core_by_block_lane: dict[tuple[int, int], int] = {}
    base_cycle: int | None = None
    observed_summary = {
        "records": len(rows),
        "atomic_records": 0,
        "clock_baseline_records": 0,
        "atomic_calls": 0,
        "batched_poll_calls": 0,
        "poll_batch_records": 0,
        "dcci_records": 0,
        "dcci_calls": 0,
        "dcci_lines": 0,
        # dropped 无法从已经导出的有效行反推；v3+ 必须由 producer summary
        # 明确承诺为零，下面再逐字段核对。
        "dropped_records": 0,
    }
    atomic_calls_by_site: dict[int, int] = {}
    v3_clock_rows: dict[int, dict[str, int | bool | None]] = {
        core_id: {"plain": 0, "dependency": 0, "return_ready": None}
        for core_id in range(num_cores)
    }
    v3_result_used_direct_rows: list[tuple[int, int, bool]] = []
    v3_return_ready_poll_batch_rows: list[tuple[int, int, bool]] = []
    v5_observer_dcci_rows = {core_id: 0 for core_id in range(num_cores)}
    v4_parent_counts: dict[int, dict[str, int]] = {
        core_id: {
            "OrchestrationReplay": 0,
            "RuntimePlanBuild": 0,
            "FinalDrain": 0,
        }
        for core_id in range(num_cores)
    }
    v4_parent_spans: dict[
        tuple[int, str], tuple[int, int]
    ] = {}
    v4_claims: dict[tuple[int, int], tuple[bool, bool, bool]] = {}
    v4_submits: set[tuple[int, int]] = set()
    v4_submit_semantics: dict[tuple[int, int], tuple[bool, bool]] = {}
    v4_tails: dict[tuple[int, int], tuple[str, int]] = {}
    v4_materializes: dict[
        tuple[int, int], list[tuple[Any, ...]]
    ] = {}
    v4_registers: dict[tuple[int, int], list[tuple[Any, ...]]] = {}
    v4_shared_register_details: dict[
        tuple[int, int], list[tuple[Any, ...]]
    ] = {}
    v4_shared_register_output_details: dict[
        tuple[int, int], list[tuple[Any, ...]]
    ] = {}
    v4_shared_register_output_copy_details: dict[
        tuple[int, int], list[tuple[Any, ...]]
    ] = {}
    v4_shared_register_output_flush_details: dict[
        tuple[int, int], list[tuple[Any, ...]]
    ] = {}
    v4_shared_insert_turn_polls: list[tuple[Any, ...]] = []
    v4_shared_insert_turn_handoffs: dict[
        tuple[int, int], list[tuple[Any, ...]]
    ] = {}
    # 逐行在写输出前检查列数、范围和可转整数的字段。任一行不满足这些约束
    # 都会整体拒绝输入，不生成缺少关键阶段的“部分可看”泳道。
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 10:
            raise ValueError(f"fdwic_events[{index}] must contain exactly 10 fields")
        core_id = _integer(row[0], f"fdwic_events[{index}].core_id")
        block_id = _integer(row[1], f"fdwic_events[{index}].block_id")
        lane = _integer(row[2], f"fdwic_events[{index}].lane")
        task_id = _integer(row[3], f"fdwic_events[{index}].task_id")
        function_id = _integer(row[4], f"fdwic_events[{index}].function_id")
        phase = str(row[5])
        start_cycle = _integer(row[6], f"fdwic_events[{index}].start_cycle")
        end_cycle = _integer(row[7], f"fdwic_events[{index}].end_cycle")
        flags = _integer(row[8], f"fdwic_events[{index}].flags")
        auxiliary = _integer(row[9], f"fdwic_events[{index}].auxiliary")
        if not 0 <= core_id < num_cores:
            raise ValueError(f"fdwic_events[{index}] has out-of-range core_id {core_id}")
        if block_id < 0:
            raise ValueError(f"fdwic_events[{index}] has negative block_id {block_id}")
        if lane not in LANE_NAMES:
            raise ValueError(f"fdwic_events[{index}] has invalid lane {lane}")
        if phase not in PHASE_NAMES:
            raise ValueError(f"fdwic_events[{index}] has unknown phase {phase!r}")
        if trace_schema_version == 5 and phase in LEGACY_LAP_PHASES:
            raise ValueError(
                f"fdwic_events[{index}] schema-v5 forbids legacy lap phase {phase!r}"
            )
        if trace_schema_version == 5 and phase == "DrainWon":
            raise ValueError(
                f"fdwic_events[{index}] schema-v5 forbids unused legacy phase 'DrainWon'"
            )
        if trace_schema_version < 5 and phase in V5_PHASES:
            raise ValueError(
                f"fdwic_events[{index}] phase {phase!r} requires trace_schema_version=5"
            )
        if trace_schema_version >= 3:
            if task_id < -1 or function_id < -1 or auxiliary < 0:
                raise ValueError(
                    f"fdwic_events[{index}] has invalid v3+ base fields: "
                    f"task={task_id} func={function_id} aux={auxiliary}"
                )
            if not 0 <= flags <= 0xFFFFFFFF:
                raise ValueError(
                    f"fdwic_events[{index}] has invalid uint32 flags {flags}"
                )
            expected_block, expected_lane, _ = _standalone_topology(core_id)
            if block_id != expected_block or lane != expected_lane:
                raise ValueError(
                    f"fdwic_events[{index}] block/lane={block_id}/{lane} does not match "
                    f"standalone topology {expected_block}/{expected_lane} for core {core_id}"
                )
        if phase == "Claim" and trace_schema_version >= 2:
            if flags & ~0x3 or (flags & 0x1 and not flags & 0x2):
                raise ValueError(
                    f"fdwic_events[{index}] has invalid Claim flags 0x{flags:x}"
                )
            if trace_schema_version >= 3 and auxiliary > 1:
                raise ValueError(
                    f"fdwic_events[{index}] has invalid Claim auxiliary {auxiliary}"
                )
        if phase == "Atomic":
            poll_batch = bool(flags & ATOMIC_POLL_BATCH)
            atomic_op = flags & 0xF
            result_used = bool(flags & ATOMIC_RESULT_USED)
            value_zero = bool(flags & ATOMIC_VALUE_ZERO)
            return_ready = bool(flags & ATOMIC_RETURN_READY)
            payload = (flags >> ATOMIC_PAYLOAD_SHIFT) & ATOMIC_PAYLOAD_MASK
            # central-ticket 的 output-published 站点是真正的等待 episode，
            # 只能导出聚合 PollBatch；same-core all-worker-replay 已由逐 task
            # insert-completion 链证明 producer 发布完成，两个站点只是一次
            # 协议校验 Load，必须允许 direct return-ready 记录。site19 在两种
            # 拓扑下始终是等待 episode，仍严格禁止逐 Load raw。
            aggregate_only_poll = (
                auxiliary in AGGREGATE_ONLY_POLL_SITE_IDS
                or (
                    submit_topology in CENTRAL_BUILD_TOPOLOGIES
                    and auxiliary in SHARED_OUTPUT_PUBLISHED_POLL_SITE_IDS
                )
            )
            if auxiliary in SCHEMA_V5_SHARED_ATOMIC_SITE_IDS and not (
                trace_schema_version == 5 and tensormap_mode == "shared"
            ):
                raise ValueError(
                    f"fdwic_events[{index}] has invalid direct Atomic "
                    f"site={auxiliary}: shared schema-v5 site requires "
                    "shared schema-v5"
                )
            if aggregate_only_poll and not poll_batch:
                raise ValueError(
                    f"fdwic_events[{index}] aggregate-only Atomic site "
                    f"{auxiliary} must use PollBatch"
                )
            if poll_batch:
                return_ready_valid = (
                    not return_ready
                    or auxiliary in AGGREGATE_ONLY_POLL_SITE_IDS
                    or auxiliary in SHARED_OUTPUT_PUBLISHED_POLL_SITE_IDS
                )
                if (
                    trace_schema_version not in (3, 5)
                    or level != 4
                    or payload == 0
                    or POLL_BATCH_SITE_OP_IDS.get(auxiliary) != atomic_op
                    or not result_used
                    or value_zero
                    or not return_ready_valid
                    or task_id != -1
                    or function_id != -1
                ):
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid Atomic PollBatch "
                        f"site={auxiliary} flags=0x{flags:x}"
                    )
                if aggregate_only_poll:
                    v3_return_ready_poll_batch_rows.append(
                        (index, core_id, return_ready)
                    )
            elif trace_schema_version in (3, 5):
                if level != 4:
                    raise ValueError(
                        f"fdwic_events[{index}] Atomic requires l2_swimlane_level=4"
                    )
                expected_atomic_op = ATOMIC_SITE_OP_IDS.get(auxiliary)
                expected_result_used = (
                    auxiliary in ATOMIC_SITE_OP_IDS
                    and auxiliary not in ATOMIC_RESULT_UNUSED_SITE_IDS
                )
                if auxiliary in RUNTIME_PLAN_ATOMIC_SITE_IDS:
                    task_local_control = (
                        auxiliary in RUNTIME_PLAN_TASK_LOCAL_SITE_IDS
                    )
                    if task_local_control != (task_id >= 0):
                        raise ValueError(
                            f"fdwic_events[{index}] Runtime Plan Atomic "
                            f"site={auxiliary} has invalid task_id={task_id}"
                        )
                if (
                    auxiliary == SHARED_INSERT_TURN_HANDOFF_SITE_ID
                    and submit_topology in STRICT_INSERT_TOPOLOGIES
                ):
                    # Scalar cross-core 的全员真实 replay 让每个 task（包括
                    # 空 writer）用返回型 CAS 发布独立 insert completion。
                    # 旧稀疏 writer FetchAdd 已退出正式协议；稳定 site 编号
                    # 保持不变，schema-v5 按当前拓扑要求 CAS/return-ready。
                    expected_atomic_op = 4
                    expected_result_used = True
                if (
                    expected_atomic_op != atomic_op
                    or result_used != expected_result_used
                    or (return_ready and not result_used)
                    or (value_zero and atomic_op != 0)
                    or (payload and atomic_op != 3)
                    or function_id != -1
                    or (
                        auxiliary in (
                            {SHARED_INSERT_TURN_HANDOFF_SITE_ID}
                            | SHARED_CLAIM_TOURNAMENT_SITE_IDS
                        )
                        and task_id < 0
                    )
                ):
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid direct Atomic "
                        f"site={auxiliary} flags=0x{flags:x}"
                    )
                if result_used:
                    v3_result_used_direct_rows.append((index, core_id, return_ready))
            observed_summary["atomic_records"] += 1
            atomic_calls_by_site[auxiliary] = (
                atomic_calls_by_site.get(auxiliary, 0)
                + (payload if poll_batch else 1)
            )
            if poll_batch:
                observed_summary["atomic_calls"] += payload
                observed_summary["batched_poll_calls"] += payload
                observed_summary["poll_batch_records"] += 1
            else:
                observed_summary["atomic_calls"] += 1
            if trace_schema_version == 5 and auxiliary in (
                SHARED_INSERT_TURN_POLL_SITE_ID,
                SHARED_INSERT_TURN_HANDOFF_SITE_ID,
            ):
                atomic_record = (
                    core_id,
                    block_id,
                    lane,
                    task_id,
                    function_id,
                    phase,
                    start_cycle,
                    end_cycle,
                    flags,
                    auxiliary,
                )
                if auxiliary == SHARED_INSERT_TURN_POLL_SITE_ID:
                    v4_shared_insert_turn_polls.append(atomic_record)
                else:
                    v4_shared_insert_turn_handoffs.setdefault(
                        (core_id, task_id), []
                    ).append(atomic_record)
        elif phase == "ClockBaseline":
            observed_summary["clock_baseline_records"] += 1
            if trace_schema_version in (3, 5):
                if level != 4:
                    raise ValueError(
                        f"fdwic_events[{index}] ClockBaseline requires l2_swimlane_level=4"
                    )
                dependency = bool(flags & 0x1)
                dependency_applied = bool(flags & 0x2)
                if (
                    flags & ~0x3
                    or (dependency_applied and not dependency)
                    or task_id != -1
                    or function_id != -1
                    or auxiliary != 0
                ):
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid ClockBaseline "
                        f"flags=0x{flags:x} auxiliary={auxiliary}"
                    )
                clock_state = v3_clock_rows[core_id]
                if dependency:
                    clock_state["dependency"] = int(clock_state["dependency"]) + 1
                    clock_state["return_ready"] = dependency_applied
                else:
                    clock_state["plain"] = int(clock_state["plain"]) + 1
        elif phase == "Dcci":
            if trace_schema_version != 5:
                raise ValueError(
                    f"fdwic_events[{index}] Dcci requires trace_schema_version=5"
                )
            op_id = flags & DCCI_OP_MASK
            call_count = (
                flags >> DCCI_CALL_COUNT_SHIFT
            ) & DCCI_CALL_COUNT_MASK
            line_count = (
                flags >> DCCI_LINE_COUNT_SHIFT
            ) & DCCI_LINE_COUNT_MASK
            if (
                auxiliary not in DCCI_SITE_OP_IDS
                or DCCI_SITE_OP_IDS[auxiliary] != op_id
                or op_id not in DCCI_OP_NAMES
                or flags & DCCI_RESERVED_BIT
                or not flags & DCCI_TRAILING_DSB
                or call_count == 0
                or line_count < call_count
            ):
                raise ValueError(
                    f"fdwic_events[{index}] has invalid Dcci "
                    f"site={auxiliary} flags=0x{flags:x}"
                )
            if auxiliary == DCCI_OBSERVER_SITE_ID:
                expected_observer_calls = (
                    3 if tensormap_mode == "shared" else 2
                )
                if (
                    call_count != expected_observer_calls
                    or task_id != -1
                    or function_id != -1
                ):
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid observer Dcci fields"
                    )
                v5_observer_dcci_rows[core_id] += 1
            elif auxiliary == DCCI_STARTUP_SITE_ID:
                if call_count != 1 or task_id != -1 or function_id != -1:
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid startup Dcci fields"
                    )
            elif auxiliary == DCCI_CONTEXT_LENS_SITE_ID:
                if (
                    tensormap_mode != "shared"
                    or call_count != 1
                    or task_id != -1
                    or function_id != -1
                ):
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid context-lens Dcci fields"
                    )
            elif auxiliary == 16:
                if (
                    tensormap_mode != "shared"
                    or call_count != 1
                    or task_id != -1
                    or function_id != -1
                ):
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid Runtime Plan "
                        "storage-ref Dcci fields"
                    )
            elif (
                tensormap_mode != "shared"
                or auxiliary not in DCCI_SHARED_ONLY_SITE_IDS
                or call_count != 1
                or task_id < 0
            ):
                raise ValueError(
                    f"fdwic_events[{index}] has invalid shared Dcci fields"
                )
            observed_summary["dcci_records"] += 1
            observed_summary["dcci_calls"] += call_count
            observed_summary["dcci_lines"] += line_count
        if trace_schema_version == 5:
            task_key = (core_id, task_id)
            if tensormap_mode == "shared" and phase == "PrepareMap":
                raise ValueError(
                    f"fdwic_events[{index}] shared schema-v5 must not contain PrepareMap"
                )
            if phase == "SharedRegisterPublishMetadata":
                if tensormap_mode != "shared":
                    raise ValueError(
                        f"fdwic_events[{index}] SharedRegisterPublishMetadata "
                        "is only valid for shared TensorMap"
                    )
                if task_id < 0 or flags != 0 or auxiliary != 0:
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid "
                        "SharedRegisterPublishMetadata fields"
                    )
                v4_shared_register_details.setdefault(task_key, []).append(
                    (
                        core_id,
                        block_id,
                        lane,
                        task_id,
                        function_id,
                        phase,
                        start_cycle,
                        end_cycle,
                        flags,
                        auxiliary,
                    )
                )
            elif phase in (
                "SharedMaterializePublishTaskOutputs",
                "SharedRegisterPublishTaskOutputs",
            ):
                if tensormap_mode != "shared":
                    raise ValueError(
                        f"fdwic_events[{index}] {phase} "
                        "is only valid for shared TensorMap"
                    )
                if task_id < 0 or flags != 0 or auxiliary != 0:
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid {phase} fields"
                    )
                v4_shared_register_output_details.setdefault(
                    task_key, []
                ).append(
                    (
                        core_id,
                        block_id,
                        lane,
                        task_id,
                        function_id,
                        phase,
                        start_cycle,
                        end_cycle,
                        flags,
                        auxiliary,
                    )
                )
            elif phase in (
                "SharedMaterializePublishTaskOutputsCopy",
                "SharedMaterializePublishTaskOutputsFlush",
                "SharedRegisterPublishTaskOutputsCopy",
                "SharedRegisterPublishTaskOutputsFlush",
            ):
                if tensormap_mode != "shared":
                    raise ValueError(
                        f"fdwic_events[{index}] {phase} "
                        "is only valid for shared TensorMap"
                    )
                if task_id < 0 or flags != 0 or auxiliary != 0:
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid {phase} fields"
                    )
                bucket = (
                    v4_shared_register_output_copy_details
                    if phase.endswith("Copy")
                    else v4_shared_register_output_flush_details
                )
                bucket.setdefault(task_key, []).append(
                    (
                        core_id,
                        block_id,
                        lane,
                        task_id,
                        function_id,
                        phase,
                        start_cycle,
                        end_cycle,
                        flags,
                        auxiliary,
                    )
                )
            if phase in (
                "OrchestrationReplay", "RuntimePlanBuild", "FinalDrain",
            ):
                if task_id != -1 or function_id != -1 or flags != 0 or auxiliary != 0:
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid schema-v5 parent {phase} fields"
                    )
                v4_parent_counts[core_id][phase] += 1
                v4_parent_spans[(core_id, phase)] = (
                    start_cycle, end_cycle,
                )
            elif phase == "Submit":
                if task_id < 0 or flags > 1 or auxiliary > 1:
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid schema-v5 Submit fields"
                    )
                if task_key in v4_submits:
                    raise ValueError(
                        f"core {core_id} has duplicate schema-v5 Submit for task {task_id}"
                    )
                v4_submits.add(task_key)
                v4_submit_semantics[task_key] = (bool(flags & 1), bool(auxiliary))
            elif phase == "Claim":
                if task_id < 0 or task_key in v4_claims:
                    raise ValueError(
                        f"core {core_id} has invalid or duplicate schema-v5 Claim for task {task_id}"
                    )
                v4_claims[task_key] = (
                    bool(flags & 0x2), bool(flags & 0x1), bool(auxiliary)
                )
            elif phase == "Materialize":
                if task_id < 0:
                    raise ValueError(
                        f"fdwic_events[{index}] Materialize requires non-negative task_id"
                    )
                v4_materializes.setdefault(task_key, []).append(
                    (
                        core_id,
                        block_id,
                        lane,
                        task_id,
                        function_id,
                        phase,
                        start_cycle,
                        end_cycle,
                        flags,
                        auxiliary,
                    )
                )
            elif phase == "Register":
                if task_id < 0:
                    raise ValueError(
                        f"fdwic_events[{index}] Register requires non-negative task_id"
                    )
                v4_registers.setdefault(task_key, []).append(
                    (
                        core_id,
                        block_id,
                        lane,
                        task_id,
                        function_id,
                        phase,
                        start_cycle,
                        end_cycle,
                        flags,
                        auxiliary,
                    )
                )
            elif phase in ("WinnerBuild", "AllocComplete"):
                if task_id < 0 or flags != 0 or auxiliary != 0:
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid schema-v5 tail {phase} fields"
                    )
                if task_key in v4_tails:
                    raise ValueError(
                        f"core {core_id} has duplicate schema-v5 tail for task {task_id}"
                    )
                if phase == "WinnerBuild" and function_id not in KERNEL_NAMES:
                    raise ValueError(
                        f"fdwic_events[{index}] WinnerBuild has invalid function_id "
                        f"{function_id}"
                    )
                if phase == "AllocComplete" and function_id != -1:
                    raise ValueError(
                        f"fdwic_events[{index}] AllocComplete requires function_id=-1"
                    )
                # task 类型必须等所有核的 Submit Alloc 标记齐全后再推导；
                # 这里仅保存尾动作自己的权威 function，避免恢复固定 %5 假设。
                v4_tails[task_key] = (phase, function_id)
        if start_cycle <= 0 or end_cycle < start_cycle:
            raise ValueError(
                f"fdwic_events[{index}] has invalid cycles start={start_cycle} end={end_cycle}"
            )
        key = (block_id, lane)
        previous_core = core_by_block_lane.setdefault(key, core_id)
        if previous_core != core_id:
            raise ValueError(
                f"block {block_id} lane {lane} maps to both core {previous_core} and core {core_id}"
            )
        # 所有 X 事件共同减去最早 start，既避免大整数转 float 的精度损失，
        # 也让设备 SYS_CNT 的绝对值不影响 Perfetto 横轴。
        base_cycle = start_cycle if base_cycle is None else min(base_cycle, start_cycle)
        rows[index] = (
            core_id,
            block_id,
            lane,
            task_id,
            function_id,
            phase,
            start_cycle,
            end_cycle,
            flags,
            auxiliary,
        )

    assert base_cycle is not None
    if submit_topology == "aicpu_plan_central_build":
        aicpu_producer = _validate_aicpu_runtime_plan_producer(
            data,
            metadata,
            frequency_hz,
            runtime_plan_abi,
            allow_host_cpu_functional=allow_host_cpu_functional,
        )
        if aicpu_producer is not None:
            causal_window = metadata[
                "aicpu_aicore_causal_capture_window"
            ]
            minimum_fdwic_start = min(int(row[6]) for row in rows)
            maximum_fdwic_end = max(int(row[7]) for row in rows)
            if not (
                int(causal_window["aicore_pre_send_max_tick"])
                < minimum_fdwic_start
                <= maximum_fdwic_end
                < int(causal_window["aicore_post_receive_min_tick"])
            ):
                raise ValueError(
                    "AICore FDWIC events are outside the strict pre-send/"
                    "post-receive causal calibration window: "
                    f"events=[{minimum_fdwic_start},{maximum_fdwic_end}] "
                    "window=("
                    f"{causal_window['aicore_pre_send_max_tick']},"
                    f"{causal_window['aicore_post_receive_min_tick']})"
                )
            causal_window.update({
                "minimum_fdwic_start_tick": minimum_fdwic_start,
                "maximum_fdwic_end_tick": maximum_fdwic_end,
                "aicore_pre_to_fdwic_gap_tick":
                    minimum_fdwic_start
                    - int(causal_window["aicore_pre_send_max_tick"]),
                "fdwic_to_aicore_post_gap_tick":
                    int(causal_window["aicore_post_receive_min_tick"])
                    - maximum_fdwic_end,
            })
            # The merged origin includes the producer; subtracting only the
            # earliest FDWIC cycle would otherwise clip a valid Plan-ahead
            # lane into negative Perfetto time.
            base_cycle = min(
                base_cycle,
                int(aicpu_producer["mapped_begin_tick"]),
            )
    if trace_schema_version in (3, 5) and level == 4:
        # 每核两条基线同时证明采集完整性和该后端是否真正应用了
        # atomic 返回值依赖；所有消费返回值的直接记录必须与本核证据一致。
        # SharedInsertTurnPoll 的最终 PollBatch 可以额外带 return_ready，
        # 但它仍表示整个轮询 episode，不能解释成单次 Load 延迟。
        for core_id, clock_state in v3_clock_rows.items():
            if clock_state["plain"] != 1 or clock_state["dependency"] != 1:
                raise ValueError(
                    f"core {core_id} requires exactly one plain and one dependency "
                    f"ClockBaseline: plain={clock_state['plain']} "
                    f"dependency={clock_state['dependency']}"
                )
        for row_index, core_id, return_ready in v3_result_used_direct_rows:
            expected_return_ready = bool(v3_clock_rows[core_id]["return_ready"])
            if return_ready != expected_return_ready:
                raise ValueError(
                    f"fdwic_events[{row_index}] direct Atomic return_ready={return_ready} "
                    f"does not match core {core_id} ClockBaseline "
                    f"dependency_applied={expected_return_ready}"
                )
        for row_index, core_id, return_ready in v3_return_ready_poll_batch_rows:
            expected_return_ready = bool(v3_clock_rows[core_id]["return_ready"])
            if return_ready != expected_return_ready:
                raise ValueError(
                    f"fdwic_events[{row_index}] PollBatch "
                    f"return_ready={return_ready} "
                    f"does not match core {core_id} ClockBaseline "
                    f"dependency_applied={expected_return_ready}"
                )

    if trace_schema_version == 5:
        for core_id, counts in v4_parent_counts.items():
            expected_parent_counts = {
                "OrchestrationReplay": (
                    0
                    if submit_topology ==
                    "aicpu_plan_central_build"
                    else 1
                ),
                "RuntimePlanBuild": (
                    1
                    if submit_topology ==
                    "aicpu_plan_central_build"
                    else 0
                ),
                "FinalDrain": 1,
            }
            for phase, expected_count in expected_parent_counts.items():
                count = counts[phase]
                if count != expected_count:
                    raise ValueError(
                        f"core {core_id} requires {expected_count} "
                        f"schema-v5 {phase}: count={count}"
                    )
        if submit_topology == "aicpu_plan_central_build":
            if v4_claims or v4_submits or v4_submit_semantics:
                raise ValueError(
                    "AICPU Plan central build must not contain legacy "
                    "Submit/Claim records"
                )
            build_child_phases = {
                "Materialize", "Register", "Fanin", "WinnerBuild",
                "AllocComplete", "SharedRegisterPublishMetadata",
                "SharedMaterializePublishTaskOutputs",
                "SharedMaterializePublishTaskOutputsCopy",
                "SharedMaterializePublishTaskOutputsFlush",
            }
            if runtime_plan_build_backend == "scalar":
                # Scalar Plan-ahead has no Submit/Claim raw. Materialize is
                # every central-ticket owner's first task identity record;
                # normalize it to winner semantics only inside validation.
                owners_by_task: dict[int, int] = {}
                for task_key, materializes in v4_materializes.items():
                    if len(materializes) != 1:
                        raise ValueError(
                            "AICPU Plan task requires exactly one Materialize "
                            f"owner at {task_key}: count={len(materializes)}"
                        )
                    core_id, task_id = task_key
                    previous_owner = owners_by_task.setdefault(
                        task_id, core_id
                    )
                    if previous_owner != core_id:
                        raise ValueError(
                            "AICPU Plan task has multiple Build owners: "
                            f"task={task_id} owners={previous_owner},{core_id}"
                        )
                    is_alloc = bool(int(materializes[0][9]))
                    v4_claims[task_key] = (True, True, is_alloc)
                    v4_submit_semantics[task_key] = (True, is_alloc)
                    v4_submits.add(task_key)
            elif runtime_plan_build_backend == "simt":
                # The formal VF does not call Scalar trace helpers.  Its task
                # identity comes from validated metadata/direct terminal state;
                # accepting Scalar children here would silently relabel a mixed
                # artifact as SIMT.
                scalar_build_children = [
                    str(row[5]) for row in rows
                    if str(row[5]) in build_child_phases
                ]
                if scalar_build_children or (
                    v4_shared_insert_turn_polls
                    or v4_shared_insert_turn_handoffs
                ):
                    raise ValueError(
                        "SIMT coarse Runtime Plan capture must not contain "
                        "Scalar Build child or insert-completion trace"
                    )
            else:
                raise AssertionError(
                    "AICPU Plan central build backend was not normalized"
                )

            for row in rows:
                core_id = int(row[0])
                phase = str(row[5])
                plan_evidence = (
                    phase in build_child_phases
                    or (
                        phase == "Atomic"
                        and int(row[9]) in
                        RUNTIME_PLAN_ATOMIC_SITE_IDS
                    )
                    or (
                        phase == "Dcci"
                        and int(row[9]) in (15, 16)
                    )
                )
                if not plan_evidence:
                    continue
                parent_start, parent_end = v4_parent_spans[
                    (core_id, "RuntimePlanBuild")
                ]
                if not (
                    parent_start <= int(row[6]) <= int(row[7])
                    <= parent_end
                ):
                    raise ValueError(
                        "AICPU Plan Build evidence is outside its per-core "
                        f"RuntimePlanBuild parent: core={core_id} "
                        f"phase={phase}"
                    )
        if set(v4_claims) != v4_submits:
            raise ValueError("schema-v5 Claim keys do not match Submit keys")
        if (
            submit_topology == "aicpu_plan_central_build"
            and runtime_plan_build_backend == "simt"
        ):
            if runtime_plan_task_kinds is None:
                raise AssertionError(
                    "SIMT Runtime Plan task metadata was not normalized"
                )
            task_kind_by_id = dict(runtime_plan_task_kinds)
        else:
            task_kind_by_id = _derive_v4_task_kinds(
                v4_submit_semantics, num_cores,
                (
                    "central_ticket"
                    if submit_topology == "aicpu_plan_central_build"
                    else str(submit_topology)
                ),
            )
            if submit_topology == "aicpu_plan_central_build":
                derived_task_count = len(task_kind_by_id)
                if inferred_legacy_runtime_plan_identity:
                    runtime_plan_task_count = derived_task_count
                    runtime_plan_task_kinds = dict(task_kind_by_id)
                    metadata["runtime_plan_task_count"] = (
                        derived_task_count
                    )
                    metadata["runtime_plan_task_kinds"] = [
                        task_kind_by_id[task_id]
                        for task_id in range(derived_task_count)
                    ]
                elif (
                    runtime_plan_task_count != derived_task_count
                    or runtime_plan_task_kinds != task_kind_by_id
                ):
                    raise ValueError(
                        "Scalar Build child trace does not match the "
                        "declared Runtime Plan task count/kinds"
                    )
        if (
            submit_topology == "aicpu_plan_central_build"
            and level == 4
        ):
            task_count = len(task_kind_by_id)
            assert runtime_plan_build_backend is not None
            assert runtime_plan_build_trace_coverage is not None
            _validate_runtime_plan_atomic_closure(
                str(metadata["pipeline"]),
                runtime_plan_build_backend,
                runtime_plan_build_trace_coverage,
                atomic_calls_by_site, task_count,
                runtime_plan_build_workers,
                runtime_plan_execute_workers,
            )
        writer_task_set = set(shared_metadata_writer_tasks)
        metadata_prefix_task_set = set(
            shared_metadata_prefix_tasks
        )
        handoff_task_set = (
            set()
            if runtime_plan_build_backend == "simt"
            else metadata_prefix_task_set
            if submit_topology in STRICT_INSERT_TOPOLOGIES
            else writer_task_set
        )
        unknown_writer_tasks = writer_task_set - set(task_kind_by_id)
        unknown_prefix_tasks = (
            metadata_prefix_task_set - set(task_kind_by_id)
        )
        if tensormap_mode == "shared":
            if unknown_writer_tasks:
                raise ValueError(
                    "metadata.shared_metadata_writer_tasks contains unknown "
                    f"tasks: {sorted(unknown_writer_tasks)[:8]}"
                )
            if unknown_prefix_tasks:
                raise ValueError(
                    "metadata.shared_metadata_prefix_tasks contains unknown "
                    f"tasks: {sorted(unknown_prefix_tasks)[:8]}"
                )
            if (
                shared_metadata_ordering == "global_writer_chain"
                and not writer_task_set.issubset(
                    metadata_prefix_task_set
                )
            ):
                raise ValueError(
                    "metadata writers must also require the strict metadata "
                    "prefix"
                )
            if (
                submit_topology in STRICT_INSERT_TOPOLOGIES
                and metadata_prefix_task_set
                != (
                    set(task_kind_by_id)
                    if runtime_plan_build_backend == "simt"
                    else {
                        task_id
                        for (_core_id, task_id), (
                            _attempted,
                            won,
                            _is_alloc,
                        ) in v4_claims.items()
                        if won
                    }
                )
            ):
                raise ValueError(
                    "all_worker_replay shared metadata prefix must contain "
                    "every winning task exactly once"
                )
        previous_writer_by_task: dict[int, int | None] = {}
        previous_writer: int | None = None
        for planned_task_id in sorted(task_kind_by_id):
            previous_writer_by_task[planned_task_id] = previous_writer
            if planned_task_id in writer_task_set:
                previous_writer = planned_task_id
        matched_per_symbol_dag_polls = 0
        for task_key, (attempted, won, is_alloc) in v4_claims.items():
            submit_won, submit_alloc = v4_submit_semantics[task_key]
            task_kind = task_kind_by_id[task_key[1]]
            expected_alloc = task_kind == 0
            if is_alloc != expected_alloc or submit_alloc != expected_alloc:
                raise ValueError(
                    f"schema-v5 task-kind mismatch at {task_key}: "
                    f"expected_alloc={expected_alloc}"
                )
            if won and not attempted:
                raise ValueError(f"schema-v5 Claim won without attempt at {task_key}")
            if submit_topology in CENTRAL_BUILD_TOPOLOGIES and (
                not attempted or not won
            ):
                raise ValueError(
                    "schema-v5 central-ticket Submit owner must have an "
                    f"attempted winner Claim at {task_key}"
                )
            if submit_won != won or submit_alloc != is_alloc:
                raise ValueError(f"schema-v5 Submit/Claim semantics mismatch at {task_key}")
            expected_tail = (
                ("AllocComplete", -1)
                if is_alloc
                else ("WinnerBuild", task_kind - 1)
            )
            actual_tail = v4_tails.get(task_key)
            # 只为 winner 记录真实尾动作；loser 没有尾记录，其剩余时间
            # 由 Submit 的离线补集表示。
            tail_valid = actual_tail == expected_tail if won else actual_tail is None
            if not tail_valid:
                raise ValueError(
                    f"schema-v5 tail mismatch at {task_key}: "
                    f"expected {expected_tail if won else 'no winner tail'}, "
                    f"got {actual_tail}"
                )
            if tensormap_mode == "shared":
                materializes = v4_materializes.get(task_key, [])
                parents = v4_registers.get(task_key, [])
                expected_parent_count = 1 if won else 0
                if len(parents) != expected_parent_count:
                    raise ValueError(
                        "shared schema-v5 requires exactly one Register parent "
                        "for each winner and none for losers at "
                        f"{task_key}: count={len(parents)} won={won}"
                    )
                details = v4_shared_register_details.get(task_key, [])
                if len(details) != (1 if won else 0):
                    raise ValueError(
                        "shared schema-v5 requires exactly one "
                        "SharedRegisterPublishMetadata for each winner and none "
                        f"for losers at {task_key}: count={len(details)} won={won}"
                )
                if not won:
                    continue
                parent = parents[0]
                detail = details[0]
                output_details = v4_shared_register_output_details.get(
                    task_key, []
                )
                if len(output_details) != 1:
                    raise ValueError(
                        "shared schema-v5 requires exactly one "
                        "SharedRegisterPublishTaskOutputs or "
                        "SharedMaterializePublishTaskOutputs for each winner "
                        "and none "
                        f"for losers at {task_key}: "
                        f"count={len(output_details)} won={won}"
                    )
                output_detail = output_details[0]
                output_phase = str(output_detail[5])
                outputs_in_materialize = (
                    output_phase ==
                    "SharedMaterializePublishTaskOutputs"
                )
                if outputs_in_materialize and len(materializes) != 1:
                    raise ValueError(
                        "shared schema-v5 requires exactly one Materialize parent "
                        "for each winner using Materialize task-output publication "
                        f"at {task_key}: count={len(materializes)}"
                    )
                if parent[:5] != detail[:5]:
                    raise ValueError(
                        "shared schema-v5 Register detail identity differs from "
                        f"its parent at {task_key}"
                    )
                output_parent = (
                    materializes[0]
                    if outputs_in_materialize
                    else detail
                )
                if output_parent[:5] != output_detail[:5]:
                    raise ValueError(
                        "shared schema-v5 task-outputs detail identity differs "
                        f"from {output_parent[5]} at {task_key}"
                    )
                parent_start, parent_end = int(parent[6]), int(parent[7])
                detail_start, detail_end = int(detail[6]), int(detail[7])
                if not (
                    parent_start
                    <= detail_start
                    <= detail_end
                    <= parent_end
                ):
                    raise ValueError(
                        "shared schema-v5 SharedRegisterPublishMetadata is outside "
                        f"Register parent at {task_key}"
                    )
                output_start = int(output_detail[6])
                output_end = int(output_detail[7])
                output_parent_start = int(output_parent[6])
                output_parent_end = int(output_parent[7])
                if not (
                    output_parent_start
                    <= output_start
                    <= output_end
                    <= output_parent_end
                ):
                    raise ValueError(
                        f"shared schema-v5 {output_phase} is outside "
                        f"{output_parent[5]} at {task_key}"
                    )
                copy_details = v4_shared_register_output_copy_details.get(
                    task_key, []
                )
                flush_details = v4_shared_register_output_flush_details.get(
                    task_key, []
                )
                if len(copy_details) != 1:
                    raise ValueError(
                        "shared schema-v5 requires exactly one "
                        "SharedRegisterPublishTaskOutputsCopy for each winner "
                        f"and none for losers at {task_key}: "
                        f"count={len(copy_details)} won={won}"
                    )
                if len(flush_details) != 1:
                    raise ValueError(
                        "shared schema-v5 requires exactly one "
                        "SharedRegisterPublishTaskOutputsFlush for each winner "
                        f"and none for losers at {task_key}: "
                        f"count={len(flush_details)} won={won}"
                    )
                copy_detail = copy_details[0]
                flush_detail = flush_details[0]
                expected_copy_phase = (
                    "SharedMaterializePublishTaskOutputsCopy"
                    if outputs_in_materialize
                    else "SharedRegisterPublishTaskOutputsCopy"
                )
                expected_flush_phase = (
                    "SharedMaterializePublishTaskOutputsFlush"
                    if outputs_in_materialize
                    else "SharedRegisterPublishTaskOutputsFlush"
                )
                if (
                    copy_detail[5] != expected_copy_phase
                    or flush_detail[5] != expected_flush_phase
                ):
                    raise ValueError(
                        "shared schema-v5 task-output detail families are mixed "
                        f"at {task_key}: parent={output_phase} "
                        f"copy={copy_detail[5]} flush={flush_detail[5]}"
                    )
                if output_detail[:5] != copy_detail[:5]:
                    raise ValueError(
                        "shared schema-v5 task-outputs copy identity differs "
                        f"from {output_phase} at {task_key}"
                    )
                if output_detail[:5] != flush_detail[:5]:
                    raise ValueError(
                        "shared schema-v5 task-outputs flush identity differs "
                        f"from {output_phase} at {task_key}"
                    )
                copy_start = int(copy_detail[6])
                copy_end = int(copy_detail[7])
                flush_start = int(flush_detail[6])
                flush_end = int(flush_detail[7])
                if not (
                    output_start
                    <= copy_start
                    <= copy_end
                    == flush_start
                    <= flush_end
                    <= output_end
                ):
                    raise ValueError(
                        "shared schema-v5 task-outputs copy/flush nesting is "
                        f"invalid at {task_key}: "
                        f"outputs=[{output_start},{output_end}) "
                        f"copy=[{copy_start},{copy_end}) "
                        f"flush=[{flush_start},{flush_end})"
                    )
                if level == 4:
                    matching_polls = [
                        poll
                        for poll in v4_shared_insert_turn_polls
                        if poll[:3] == parent[:3]
                        and int(poll[6]) == parent_start
                        and int(poll[7]) == detail_start
                    ]
                    if shared_metadata_ordering == "per_symbol_dag":
                        # DAG 由 device Build 动态推导，host 不再提供逐 task
                        # 前驱权威。设备把一个 task 的全部精确前驱轮询聚成
                        # 至多一条 PollBatch；converter 只闭合数量上限和真实
                        # Register 前段边界，不按 PA task 形状臆造前驱。
                        if len(matching_polls) > 1:
                            raise ValueError(
                                "shared schema-v5 per_symbol_dag allows at "
                                "most one SharedInsertTurnPoll PollBatch on "
                                "Register.start->metadata.start at "
                                f"{task_key}: count={len(matching_polls)}"
                            )
                        matched_per_symbol_dag_polls += len(matching_polls)
                    elif submit_topology in STRICT_INSERT_TOPOLOGIES:
                        # same-core shared 仍采用逐 task insert-completion 链：
                        # task N 等 N-1，并由每个 winner 发布自己的 completion。
                        # 真正改写 writer metadata 的 task 仍由 writer 列表独立
                        # 描述，不能把二者混成同一种业务动作。
                        expected_poll_count = (
                            1
                            if task_key[1] in metadata_prefix_task_set
                            and any(
                                prefix_task < task_key[1]
                                for prefix_task in metadata_prefix_task_set
                            )
                            else 0
                        )
                        if len(matching_polls) != expected_poll_count:
                            raise ValueError(
                                "shared schema-v5 all_worker_replay requires "
                                "one SharedInsertTurnPoll PollBatch for every "
                                "task after the first insert completion at "
                                f"{task_key}: count={len(matching_polls)} "
                                f"expected={expected_poll_count}"
                            )
                    else:
                        # 全局稀疏链只等待严格早于当前 task 的最近 metadata
                        # writer；host prefix 在该旧协议中是独立结构权威。
                        expected_poll_count = (
                            1
                            if task_key[1] in metadata_prefix_task_set
                            and previous_writer_by_task[task_key[1]] is not None
                            else 0
                        )
                        if len(matching_polls) != expected_poll_count:
                            raise ValueError(
                                "shared schema-v5 level4 requires exactly one "
                                "SharedInsertTurnPoll PollBatch when a previous "
                                "metadata writer exists and none before the first "
                                "writer on "
                                "Register.start->metadata.start at "
                                f"{task_key}: count={len(matching_polls)} "
                                f"expected={expected_poll_count}"
                            )
                    handoffs = v4_shared_insert_turn_handoffs.get(
                        task_key, []
                    )
                    expected_handoff_count = (
                        1 if task_key[1] in handoff_task_set else 0
                    )
                    if len(handoffs) != expected_handoff_count:
                        raise ValueError(
                            "shared schema-v5 level4 requires exactly one "
                            "SharedInsertTurnHandoff direct "
                            "CompareExchange per "
                            "required insert completion and none otherwise at "
                            f"{task_key}: count={len(handoffs)} "
                            f"expected={expected_handoff_count}"
                        )
                    if handoffs:
                        handoff = handoffs[0]
                        handoff_start = int(handoff[6])
                        handoff_end = int(handoff[7])
                        if handoff[:3] != parent[:3] or not (
                            detail_end
                            <= handoff_start
                            <= handoff_end
                            <= parent_end
                        ):
                            raise ValueError(
                                "shared schema-v5 SharedInsertTurnHandoff "
                                "identity or boundary is outside "
                                "metadata.end->Register.end "
                                f"at {task_key}"
                            )

        if tensormap_mode == "shared":
            orphan_parent_keys = set(v4_registers) - set(v4_claims)
            if orphan_parent_keys:
                raise ValueError(
                    "shared schema-v5 Register parents have no matching Claim: "
                    f"{sorted(orphan_parent_keys)[:8]}"
                )
            orphan_detail_keys = set(v4_shared_register_details) - set(v4_claims)
            if orphan_detail_keys:
                raise ValueError(
                    "shared schema-v5 Register details have no matching Claim: "
                    f"{sorted(orphan_detail_keys)[:8]}"
                )
            orphan_output_detail_keys = (
                set(v4_shared_register_output_details) - set(v4_claims)
            )
            if orphan_output_detail_keys:
                raise ValueError(
                    "shared schema-v5 task-output details have no matching Claim: "
                    f"{sorted(orphan_output_detail_keys)[:8]}"
                )
            orphan_copy_detail_keys = (
                set(v4_shared_register_output_copy_details) - set(v4_claims)
            )
            if orphan_copy_detail_keys:
                raise ValueError(
                    "shared schema-v5 task-output copy details have no matching "
                    f"Claim: {sorted(orphan_copy_detail_keys)[:8]}"
                )
            orphan_flush_detail_keys = (
                set(v4_shared_register_output_flush_details) - set(v4_claims)
            )
            if orphan_flush_detail_keys:
                raise ValueError(
                    "shared schema-v5 task-output flush details have no matching "
                    f"Claim: {sorted(orphan_flush_detail_keys)[:8]}"
                )
            for task_key, output_details in (
                v4_shared_register_output_details.items()
            ):
                won = v4_claims.get(task_key, (False, False, False))[1]
                if len(output_details) != (1 if won else 0):
                    raise ValueError(
                        "shared schema-v5 requires exactly one "
                        "SharedRegisterPublishTaskOutputs for each winner and none "
                        f"for losers at {task_key}: "
                        f"count={len(output_details)} won={won}"
                    )
            for task_key, copy_details in (
                v4_shared_register_output_copy_details.items()
            ):
                won = v4_claims.get(task_key, (False, False, False))[1]
                if len(copy_details) != (1 if won else 0):
                    raise ValueError(
                        "shared schema-v5 requires exactly one "
                        "SharedRegisterPublishTaskOutputsCopy for each winner "
                        f"and none for losers at {task_key}: "
                        f"count={len(copy_details)} won={won}"
                    )
            for task_key, flush_details in (
                v4_shared_register_output_flush_details.items()
            ):
                won = v4_claims.get(task_key, (False, False, False))[1]
                if len(flush_details) != (1 if won else 0):
                    raise ValueError(
                        "shared schema-v5 requires exactly one "
                        "SharedRegisterPublishTaskOutputsFlush for each winner "
                        f"and none for losers at {task_key}: "
                        f"count={len(flush_details)} won={won}"
                    )
            if level == 4:
                winner_count = sum(
                    won for _attempted, won, _is_alloc in v4_claims.values()
                )
                if shared_metadata_ordering == "per_symbol_dag":
                    if (
                        len(v4_shared_insert_turn_polls)
                        != matched_per_symbol_dag_polls
                    ):
                        raise ValueError(
                            "shared schema-v5 per_symbol_dag has orphan "
                            "SharedInsertTurnPoll records: "
                            f"records={len(v4_shared_insert_turn_polls)} "
                            f"matched_registers={matched_per_symbol_dag_polls} "
                            f"all_winners={winner_count}"
                        )
                elif submit_topology in STRICT_INSERT_TOPOLOGIES:
                    predecessor_wait_winner_count = sum(
                        1
                        for (_core_id, task_id), (
                            _attempted,
                            won,
                            _is_alloc,
                        ) in v4_claims.items()
                        if won
                        and task_id in metadata_prefix_task_set
                        and any(
                            prefix_task < task_id
                            for prefix_task in metadata_prefix_task_set
                        )
                    )
                    if (
                        len(v4_shared_insert_turn_polls)
                        != predecessor_wait_winner_count
                    ):
                        raise ValueError(
                            "shared schema-v5 all_worker_replay has orphan or "
                            "duplicate SharedInsertTurnPoll records: "
                            f"records={len(v4_shared_insert_turn_polls)} "
                            "expected_predecessor_wait_winners="
                            f"{predecessor_wait_winner_count} "
                            f"all_winners={winner_count}"
                        )
                else:
                    predecessor_wait_winner_count = sum(
                        1
                        for (_core_id, task_id), (
                            _attempted,
                            won,
                            _is_alloc,
                        ) in v4_claims.items()
                        if won and
                        task_id in metadata_prefix_task_set and
                        previous_writer_by_task[task_id] is not None
                    )
                    if (
                        len(v4_shared_insert_turn_polls)
                        != predecessor_wait_winner_count
                    ):
                        raise ValueError(
                            "shared schema-v5 level4 has orphan or duplicate "
                            "SharedInsertTurnPoll records: "
                            f"records={len(v4_shared_insert_turn_polls)} "
                            "expected_predecessor_wait_winners="
                            f"{predecessor_wait_winner_count} "
                            f"all_winners={winner_count}"
                        )
                handoff_count = sum(
                    len(items)
                    for items in v4_shared_insert_turn_handoffs.values()
                )
                if handoff_count != len(handoff_task_set):
                    raise ValueError(
                        "shared schema-v5 level4 has orphan or duplicate "
                        "SharedInsertTurnHandoff records: "
                        f"records={handoff_count} "
                        f"expected_handoffs={len(handoff_task_set)} "
                        f"winners={winner_count}"
                    )

    if trace_schema_version >= 3:
        producer_summary = metadata.get("fdwic_summary")
        if not isinstance(producer_summary, dict):
            raise ValueError(
                "metadata.fdwic_summary is required for trace_schema_version>=3"
            )
        dcci_keys = ("dcci_records", "dcci_calls", "dcci_lines")
        dcci_declared = any(key in producer_summary for key in dcci_keys)
        if observed_summary["dcci_records"] != 0 and not dcci_declared:
            raise ValueError(
                "raw Dcci records require dcci_records/dcci_calls/dcci_lines "
                "in metadata.fdwic_summary"
            )
        if dcci_declared:
            if not all(key in producer_summary for key in dcci_keys):
                raise ValueError(
                    "metadata.fdwic_summary must declare all three DCCI counters"
                )
            if trace_schema_version != 5:
                raise ValueError(
                    "DCCI summary counters require trace_schema_version=5"
                )
            invalid = {
                core_id: count
                for core_id, count in v5_observer_dcci_rows.items()
                if count != 1
            }
            if invalid:
                raise ValueError(
                    "schema-v5 DCCI capture requires exactly one "
                    "ObserverTraceExport record per core; "
                    f"invalid={invalid}"
                )
        for key, observed_value in observed_summary.items():
            # 迁移前 schema-v5 capture 没有 DCCI 行，也没有三项 summary。
            # 一旦任一新字段出现，就必须按上面的完整合同严格闭合。
            if key in dcci_keys and not dcci_declared:
                continue
            producer_value = _integer(
                producer_summary.get(key), f"metadata.fdwic_summary.{key}"
            )
            if producer_value != observed_value:
                raise ValueError(
                    f"metadata.fdwic_summary.{key}={producer_value} "
                    f"does not match raw value {observed_value}"
                )
    return frequency_hz, trace_schema_version, rows, core_by_block_lane, base_cycle, metadata


def _restore_v5_shared_efdrain(
    rows: list[tuple[Any, ...]],
    trace_schema_version: int,
    tensormap_mode: str | None,
) -> None:
    """用 Submit.start 与 Claim.start 离线恢复 shared EfDrain。

    新 shared raw 不再为每个 Submit 写一条 EfDrain 记录。该区间的两端
    已分别由 Submit 和 Claim 权威记录，离线恢复既不扩张 raw ABI，也不会
    污染设备侧 ``fdwic_summary.records``。采集与加工使用同一份代码，
    因此 shared schema-v5 一旦出现显式 EfDrain 就直接拒绝，避免同时
    维护设备记录与离线派生两个口径。
    """

    if trace_schema_version != 5 or tensormap_mode != "shared":
        return

    submits: dict[tuple[int, int], tuple[Any, ...]] = {}
    claims: dict[tuple[int, int], tuple[Any, ...]] = {}
    for row in rows:
        phase = str(row[5])
        if phase == "EfDrain":
            raise ValueError(
                "shared schema-v5 raw must not contain explicit EfDrain"
            )
        if phase not in ("Submit", "Claim"):
            continue
        key = (int(row[0]), int(row[3]))
        bucket = submits if phase == "Submit" else claims
        if key in bucket:
            raise ValueError(
                f"shared schema-v5 has duplicate {phase} for {key}"
            )
        bucket[key] = row

    if set(claims) != set(submits):
        missing = sorted(set(submits) - set(claims))
        orphan = sorted(set(claims) - set(submits))
        raise ValueError(
            "shared schema-v5 EfDrain derivation requires exactly one Claim "
            f"per Submit: missing={missing[:8]} orphan={orphan[:8]}"
        )

    for key in sorted(submits):
        submit = submits[key]
        claim = claims[key]
        if tuple(int(value) for value in claim[:5]) != tuple(
            int(value) for value in submit[:5]
        ):
            raise ValueError(
                "shared schema-v5 Claim identity differs from Submit for "
                f"{key}"
            )
        submit_start = int(submit[6])
        submit_end = int(submit[7])
        claim_start = int(claim[6])
        claim_end = int(claim[7])
        if not (
            submit_start
            <= claim_start
            <= claim_end
            <= submit_end
        ):
            raise ValueError(
                "shared schema-v5 cannot derive EfDrain because Claim is "
                f"outside or inverted relative to Submit at {key}: "
                f"Submit=[{submit_start},{submit_end}) "
                f"Claim=[{claim_start},{claim_end})"
            )

        expected = (
            int(submit[0]),
            int(submit[1]),
            int(submit[2]),
            int(submit[3]),
            # EfDrain 是 Submit 前端的 scalar 控制区，不属于某个计算
            # function；历史设备记录和新离线事件都固定使用 -1。
            -1,
            "EfDrain",
            submit_start,
            claim_start,
            0,
            0,
        )
        rows.append(expected)


def _iter_v5_cross_core_winner_build_pack_spans(
    rows: list[tuple[Any, ...]],
    trace_schema_version: int,
    tensormap_mode: str | None,
    submit_topology: str | None,
) -> Iterator[tuple[int, int, int, int, int, str]]:
    """用 reserve 与 payload flush 边界标出本地打包区间。

    cross-core 正常 Build 的顺序固定为：reserve CAS、Preload/Pack、
    payload DCCI、BUILT CAS。因此 reserve.end 到 payload-flush.start
    正好包围纯 Scalar 的 Preload/Pack，不依赖正常路径保留多余的 fatal
    读取。该区间完全在离线侧派生，不增加设备 raw 记录；错误路径缺少
    任一边界时不猜测，也不伪造该 span。
    """

    if (
        trace_schema_version != 5
        or tensormap_mode != "shared"
        or submit_topology not in CENTRAL_BUILD_TOPOLOGIES
    ):
        return

    parents: dict[tuple[int, int, int], tuple[Any, ...]] = {}
    reserves: dict[
        tuple[int, int, int], list[tuple[Any, ...]]
    ] = {}
    payload_flushes: dict[
        tuple[int, int, int], list[tuple[Any, ...]]
    ] = {}
    for row in rows:
        core_id, _block_id, lane, task_id, _function_id, phase, *_rest = row
        key = (int(core_id), int(lane), int(task_id))
        if phase == "WinnerBuild":
            if key in parents:
                raise ValueError(
                    "schema-v5 cross-core payload-pack derivation found "
                    f"duplicate WinnerBuild {key}"
                )
            parents[key] = row
        elif phase == "Atomic" and int(row[9]) == 46:
            reserves.setdefault(key, []).append(row)
        elif phase == "Dcci" and int(row[9]) == 11:
            payload_flushes.setdefault(key, []).append(row)

    for key in sorted(parents):
        parent = parents[key]
        parent_start = int(parent[6])
        parent_end = int(parent[7])
        contained_reserves = [
            row
            for row in reserves.get(key, [])
            if parent_start <= int(row[6])
            and int(row[7]) <= parent_end
        ]
        contained_flushes = [
            row
            for row in payload_flushes.get(key, [])
            if parent_start <= int(row[6])
            and int(row[7]) <= parent_end
        ]
        if len(contained_reserves) != 1 or len(contained_flushes) != 1:
            continue
        pack_start = int(contained_reserves[0][7])
        pack_end = int(contained_flushes[0][6])
        if pack_end <= pack_start:
            continue
        yield (
            int(parent[0]),
            int(parent[1]),
            int(parent[2]),
            pack_start,
            pack_end,
            (
                "winner_build.pack_execution_payload[GM+Scalar]"
                f"#{int(parent[3])}"
            ),
        )


def _iter_v5_cross_core_semantic_gap_spans(  # noqa: PLR0912, PLR0915
    rows: list[tuple[Any, ...]],
    frequency_hz: int,
    trace_schema_version: int,
    tensormap_mode: str | None,
    submit_topology: str | None,
) -> Iterator[tuple[int, int, int, int, int, str]]:
    """用现有原语端点标出 cross-core 中可证明的大块非原语代码。

    这些 span 只解释两个已经相邻且有源码合同的 raw 边界之间执行了什么：
    Atomic、DCCI 与 Kernel 本身仍由设备 raw 单独显示。只导出不少于 1 us
    的区间，避免为了补小缝继续膨胀 merged；raw ABI、设备写入量和被测
    调度路径均不改变。

    名称中的 ``GM+Scalar`` 表示源码同时搬运/读取 GM 状态并执行 Scalar
    校验或打包，不表示整段都是 GM stall；精确 DCCI/Atomic 仍以嵌套事件
    为准。无法由现有端点唯一确定业务含义的区间不猜测、不生成。
    """

    if (
        trace_schema_version != 5
        or tensormap_mode != "shared"
        or submit_topology not in CENTRAL_BUILD_TOPOLOGIES
    ):
        return

    minimum_ticks = max(1, (frequency_hz + 999_999) // 1_000_000)
    rows_by_lane: dict[
        tuple[int, int], list[tuple[Any, ...]]
    ] = {}
    for row in rows:
        rows_by_lane.setdefault(
            (int(row[0]), int(row[2])), []
        ).append(row)

    def emit_if_large(
        parent: tuple[Any, ...], start: int, end: int, name: str
    ) -> Iterator[tuple[int, int, int, int, int, str]]:
        if end - start < minimum_ticks:
            return
        yield (
            int(parent[0]), int(parent[1]), int(parent[2]),
            start, end, name,
        )

    def contained(
        parent: tuple[Any, ...],
    ) -> list[tuple[Any, ...]]:
        start = int(parent[6])
        end = int(parent[7])
        return [
            row
            for row in rows_by_lane[
                (int(parent[0]), int(parent[2]))
            ]
            if row is not parent
            and start <= int(row[6])
            and int(row[7]) <= end
        ]

    # Materialize 前段只扫描 block-local TaskArgs/SubmitContext，计算输出
    # 布局；第一个 shared-heap Load 是它首次进入共享 heap 协议的权威边界。
    # heap 原语结束到 descriptor 发布开始之间，则会写 GM TaskPayload 并在
    # Scalar 上准备 writer delta。两段都由既有端点唯一界定。
    for parent in (row for row in rows if row[5] == "Materialize"):
        children = contained(parent)
        heap_loads = [
            row for row in children
            if row[5] == "Atomic" and int(row[9]) == 15
        ]
        heap_ops = [
            row for row in children
            if row[5] == "Atomic" and 15 <= int(row[9]) <= 18
        ]
        output_parents = [
            row for row in children
            if row[5] == "SharedMaterializePublishTaskOutputs"
        ]
        if (
            len(heap_loads) != 1
            or not heap_ops
            or len(output_parents) != 1
        ):
            continue
        first_heap = heap_loads[0]
        last_heap_end = max(int(row[7]) for row in heap_ops)
        output_parent = output_parents[0]
        task_id = int(parent[3])
        yield from emit_if_large(
            parent, int(parent[6]), int(first_heap[6]),
            f"materialize.validate_output_layout[Scalar]#{task_id}",
        )
        if last_heap_end <= int(output_parent[6]):
            yield from emit_if_large(
                parent, last_heap_end, int(output_parent[6]),
                (
                    "materialize.write_descriptors_and_prepare_writer_delta"
                    f"[GM+Scalar]#{task_id}"
                ),
            )

    # Fanin.start 到第一次共享原语之间固定执行参数引用扫描与合法性校验。
    # SharedOutputRef 路径随后进入 output-published atomic；ordinary tensor
    # 路径随后进入 TensorMap tail atomic。两种都可能读取描述符，因此使用
    # GM+Scalar，而不把这段误标成第一次 atomic 的耗时。
    for parent in (row for row in rows if row[5] == "Fanin"):
        children = contained(parent)
        first_boundaries = [
            row for row in children
            if row[5] in ("Atomic", "Dcci")
            and not (
                row[5] == "Atomic"
                and int(row[8]) & ATOMIC_POLL_BATCH
            )
        ]
        if not first_boundaries:
            continue
        first = min(
            first_boundaries,
            key=lambda row: (int(row[6]), int(row[7])),
        )
        yield from emit_if_large(
            parent, int(parent[6]), int(first[6]),
            f"fanin.scan_and_validate_refs[GM+Scalar]#{int(parent[3])}",
        )
        ordered_boundaries = sorted(
            first_boundaries,
            key=lambda row: (int(row[6]), int(row[7]), str(row[5])),
        )
        previous = ordered_boundaries[0]
        cursor = int(previous[7])
        for current in ordered_boundaries[1:]:
            current_start = int(current[6])
            if (
                current_start > cursor
                and previous[5] == "Dcci"
                and int(previous[9]) == 0
                and current[5] == "Atomic"
                and int(current[9]) == 23
            ):
                yield from emit_if_large(
                    parent, cursor, current_start,
                    (
                        "fanin.read_writer_history[GM+Scalar]"
                        f"#{int(parent[3])}"
                    ),
                )
            if int(current[7]) > cursor:
                cursor = int(current[7])
                previous = current
        if (
            previous[5] == "Atomic"
            and int(previous[9]) == 25
            and int(parent[7]) > cursor
        ):
            yield from emit_if_large(
                parent, cursor, int(parent[7]),
                f"fanin.commit_validated_edges[Scalar]#{int(parent[3])}",
            )

    # WinnerBuild 的 source 解析区可以包含若干 descriptor invalidate；用
    # 外层 span 把它们与周围 GM 地址解析/Scalar 校验归在同一业务阶段，
    # DCCI 子事件仍保持原有精确位置。reserve.end -> payload-flush.start 的
    # payload 打包由上面的专用派生器生成，避免重复覆盖同一区间。
    for parent in (row for row in rows if row[5] == "WinnerBuild"):
        children = contained(parent)
        reserves = [
            row for row in children
            if row[5] == "Atomic" and int(row[9]) == 46
        ]
        if len(reserves) != 1:
            continue
        reserve = reserves[0]
        yield from emit_if_large(
            parent, int(parent[6]), int(reserve[6]),
            (
                "winner_build.resolve_payload_sources[GM+Scalar]"
                f"#{int(parent[3])}"
            ),
        )

    # EfDrain 中只处理已由相邻 direct Atomic/DCCI/Kernel 端点证明的几类
    # 高频大缝。PollBatch 是等待 episode 包络而非连续原子耗时，既不能拿来
    # 切 direct gap，也不能把其内部时间重新命名成 GM/Scalar。
    for parent in (row for row in rows if row[5] == "EfDrain"):
        boundaries = [
            row for row in contained(parent)
            if row[5] in ("Atomic", "Dcci", "Kernel")
            and not (
                row[5] == "Atomic"
                and int(row[8]) & ATOMIC_POLL_BATCH
            )
        ]
        boundaries.sort(
            key=lambda row: (int(row[6]), int(row[7]), str(row[5]))
        )
        cursor = int(parent[6])
        previous: tuple[Any, ...] | None = None
        for current in boundaries:
            current_start = int(current[6])
            current_end = int(current[7])
            if current_start > cursor:
                previous_phase = None if previous is None else str(previous[5])
                previous_site = (
                    None if previous is None else int(previous[9])
                )
                current_phase = str(current[5])
                current_site = int(current[9])
                name: str | None = None
                task_id: int | None = None
                if previous is None and (
                    (current_phase == "Atomic" and current_site in (5, 45))
                    or current_phase == "Kernel"
                ):
                    name = "efdrain.inspect_tokens_and_plan[GM+Scalar]"
                elif previous_phase == "Dcci" and previous_site == 12 and (
                    (current_phase == "Atomic" and current_site == 5)
                    or current_phase == "Kernel"
                ):
                    assert previous is not None
                    task_id = int(previous[3])
                    name = "execute.bind_payload_and_rebuild_args[GM+Scalar]"
                elif (
                    previous_phase == "Atomic"
                    and previous_site == 5
                    and current_phase == "Atomic"
                    and current_site == 5
                ):
                    name = "execute.advance_fanin_prefix[GM+Scalar]"
                elif (
                    previous_phase == "Atomic"
                    and previous_site == 5
                    and current_phase == "Atomic"
                    and current_site == 45
                ):
                    name = "efdrain.defer_token_and_scan_plan[GM+Scalar]"
                elif (
                    previous_phase == "Atomic"
                    and previous_site == 45
                    and current_phase == "Atomic"
                    and current_site == 45
                ):
                    name = "efdrain.scan_exec_candidates[GM+Scalar]"
                elif (
                    previous_phase == "Atomic"
                    and previous_site == 45
                    and current_phase == "Atomic"
                    and current_site == 48
                ):
                    assert previous is not None
                    task_id = int(previous[3])
                    name = "efdrain.evaluate_exec_claim[GM+Scalar]"
                elif (
                    previous_phase == "Atomic"
                    and previous_site == 45
                    and current_phase == "Atomic"
                    and current_site == 2
                ):
                    name = "efdrain.return_and_check_fatal[GM+Scalar]"
                elif (
                    previous_phase == "Atomic"
                    and previous_site == 51
                    and current_phase == "Atomic"
                    and current_site in (5, 45)
                ):
                    assert previous is not None
                    task_id = int(previous[3])
                    name = "execute.recycle_token_and_continue[GM+Scalar]"
                if name is not None:
                    if task_id is not None:
                        name = f"{name}#{task_id}"
                    yield from emit_if_large(
                        parent, cursor, current_start, name,
                    )
            if current_end > cursor:
                cursor = current_end
                previous = current

    # FinalDrain 不再生产 Build，只反复推进 owner-local token、扫描可执行
    # candidate，并在本核排空后参加全局 drain 汇合。设备侧 PollBatch 记录
    # 覆盖整个轮询 episode，里面既有 fanin/drain atomic load，也夹着 GM
    # control/payload 读取和 Scalar 判断，不能把其总时长冒充成 atomic 指令
    # 延迟。因此这里仍以逐调用 direct Atomic、DCCI、Kernel、Commit 为切点，
    # 再把相邻切点间不少于 1 us 的业务代码离线标出；不增加 raw 记录。
    for parent in (row for row in rows if row[5] == "FinalDrain"):
        boundaries = [
            row for row in contained(parent)
            if row[5] in ("Atomic", "Dcci", "Kernel", "Commit")
            and not (
                row[5] == "Atomic"
                and int(row[8]) & ATOMIC_POLL_BATCH
            )
        ]
        boundaries.sort(
            key=lambda row: (int(row[6]), int(row[7]), str(row[5]))
        )
        cursor = int(parent[6])
        previous: tuple[Any, ...] | None = None
        for current in boundaries:
            current_start = int(current[6])
            current_end = int(current[7])
            if current_start > cursor:
                previous_phase = None if previous is None else str(previous[5])
                previous_site = (
                    None if previous is None else int(previous[9])
                )
                previous_task = (
                    None if previous is None else int(previous[3])
                )
                current_phase = str(current[5])
                current_site = int(current[9])
                current_task = int(current[3])
                name: str | None = None

                # Kernel 前的长区间可能包含多轮 fanin flag load；PollBatch
                # 已在泳道中显示其调用次数，这里只补齐同一时间内的 GM
                # token/payload 访问、Scalar ready 判断与 engine 参数准备。
                if current_phase == "Kernel":
                    name = (
                        "final_drain.wait_fanin_and_prepare_engine"
                        f"[AtomicPoll+GM+Scalar]#{current_task}"
                    )
                elif current_phase == "Atomic" and current_site == 45:
                    if previous is None:
                        name = (
                            "final_drain.inspect_tokens_and_plan[GM+Scalar]"
                        )
                    elif previous_phase == "Dcci" and previous_site == 12:
                        name = (
                            "final_drain.defer_token_and_scan_candidates"
                            "[AtomicPoll+GM+Scalar]"
                        )
                    elif previous_phase == "Commit":
                        name = (
                            "final_drain.recycle_token_and_scan_candidates"
                            "[GM+Scalar]"
                        )
                    elif previous_phase == "Atomic" and previous_site == 45:
                        if previous_task == current_task:
                            name = (
                                "final_drain.reobserve_blocked_candidate"
                                f"[GM+Scalar]#{current_task}"
                            )
                        else:
                            name = (
                                "final_drain.scan_exec_candidates[GM+Scalar]"
                            )
                elif (
                    current_phase == "Atomic"
                    and current_site == 48
                    and previous_phase == "Atomic"
                    and previous_site == 45
                    and previous_task == current_task
                ):
                    name = (
                        "final_drain.evaluate_exec_claim[GM+Scalar]"
                        f"#{current_task}"
                    )
                elif current_phase == "Atomic" and current_site == 52:
                    name = (
                        "final_drain.verify_local_drain_and_encode_arrival"
                        "[GM+Scalar]"
                    )

                if name is not None:
                    yield from emit_if_large(
                        parent, cursor, current_start, name,
                    )
            if current_end > cursor:
                cursor = current_end
                previous = current

        # 非 root 发布 arrival 后直接关闭本核；root 继续轮询所有 arrival
        # group 并核对完成数。后者包含 PollBatch 记录所汇总的 atomic loads，
        # 因而显式标成混合区间，不能解释为纯 Scalar 或纯 atomic。
        if int(parent[7]) > cursor and previous is not None:
            if previous[5] == "Atomic" and int(previous[9]) == 52:
                tail_name = (
                    "final_drain.root_wait_arrivals_and_validate"
                    "[AtomicPoll+GM+Scalar]"
                    if int(parent[0]) == 0
                    else "final_drain.close_after_arrival[Scalar]"
                )
                yield from emit_if_large(
                    parent, cursor, int(parent[7]), tail_name,
                )


# 写一个 Chrome Trace Event，并统一处理数组元素间的逗号。
def _emit_event(output: TextIO, event: dict[str, Any], first: bool) -> bool:
    # 逐事件写出，避免再在内存中构造一份体积可达数百 MiB 的 merged 列表。
    if not first:
        output.write(",\n")
    json.dump(event, output, ensure_ascii=False, separators=(",", ":"))
    return False


def _iter_v5_residual_spans(  # noqa: PLR0912
    rows: list[tuple[Any, ...]],
    tensormap_mode: str | None = None,
    submit_topology: str | None = None,
) -> Iterator[tuple[int, int, int, int, int, str]]:
    """只用既有 Submit/child 边界生成逐段补集，不改 raw ABI。"""

    if submit_topology == "aicpu_plan_central_build":
        # Plan-ahead 没有 Submit endpoint；它的全核父区间由
        # RuntimePlanBuild raw 直接给出。用 Materialize 反推虚假
        # Submit 会重新引入已删除的 replay 语义。
        return

    submits_by_lane: dict[tuple[int, int], list[tuple[Any, ...]]] = {}
    submit_by_task: dict[tuple[int, int, int], tuple[Any, ...]] = {}
    children_by_task: dict[tuple[int, int, int], list[tuple[Any, ...]]] = {}
    for row in rows:
        core_id, _block_id, lane, task_id, _function_id, phase, *_rest = row
        lane_key = (int(core_id), int(lane))
        task_key = (int(core_id), int(lane), int(task_id))
        if phase == "Submit":
            submits_by_lane.setdefault(lane_key, []).append(row)
            if task_key in submit_by_task:
                raise ValueError(f"schema-v5 residual synthesis found duplicate Submit {task_key}")
            submit_by_task[task_key] = row
        elif phase in V5_EXCLUSIVE_SUBMIT_PHASES:
            children_by_task.setdefault(task_key, []).append(row)

    orphan_child_keys = set(children_by_task) - set(submit_by_task)
    if orphan_child_keys:
        raise ValueError(
            "schema-v5 residual synthesis found children without matching Submit: "
            f"{sorted(orphan_child_keys)[:8]}"
        )

    # 先按每个 scalar lane 标记相邻 Submit 之间的真实空白。
    for lane_key in sorted(submits_by_lane):
        submits = sorted(
            submits_by_lane[lane_key], key=lambda row: (int(row[6]), int(row[7]))
        )
        for previous, current in zip(submits, submits[1:]):
            previous_end = int(previous[7])
            current_start = int(current[6])
            if current_start < previous_end:
                raise ValueError(f"schema-v5 Submit spans overlap on core/lane {lane_key}")
            if current_start > previous_end:
                yield (
                    int(previous[0]),
                    int(previous[1]),
                    int(previous[2]),
                    previous_end,
                    current_start,
                    "between_submit_residual",
                )

    # Submit 内部只扣除同 task 的互斥 child；每个不连续补集段
    # 单独生成一条最小 Perfetto X event，不伪造跨空白的连续区间。
    # 只有“最后一个已知 child.end -> Submit.end”使用 tail 名称；
    # 前缀和 child-to-child gap 继续保留中性 residual，不冒充业务阶段。
    for task_key in sorted(submit_by_task):
        submit = submit_by_task[task_key]
        submit_start = int(submit[6])
        submit_end = int(submit[7])
        cursor = submit_start
        children = sorted(
            children_by_task.get(task_key, []),
            key=lambda row: (int(row[6]), int(row[7]), str(row[5])),
        )
        previous_phase: str | None = None
        for child in children:
            child_start = int(child[6])
            child_end = int(child[7])
            if child_start < submit_start or child_end > submit_end:
                raise ValueError(
                    f"schema-v5 {child[5]} child is outside Submit {task_key}"
                )
            if child_start < cursor:
                raise ValueError(
                    f"schema-v5 exclusive children overlap in Submit {task_key}"
                )
            if child_start > cursor:
                gap_name = "submit_residual"
                if (
                    tensormap_mode == "shared"
                    and submit_topology in CENTRAL_BUILD_TOPOLOGIES
                    and previous_phase == "Claim"
                    and child[5] == "Materialize"
                ):
                    # 中央 ticket 的 Claim.end -> Materialize.start 恰好是
                    # dispatch plan 解码、winner context 绑定和 TaskArgs 构造；
                    # 复用原 residual 的同一条离线事件，不增加 merged 行数。
                    gap_name = (
                        "arg_build.plan_and_construct_args[GM+Scalar]"
                        f"#{int(submit[3])}"
                    )
                yield (
                    int(submit[0]),
                    int(submit[1]),
                    int(submit[2]),
                    cursor,
                    child_start,
                    gap_name,
                )
            cursor = max(cursor, child_end)
            previous_phase = str(child[5])
        if submit_end > cursor:
            yield (
                int(submit[0]),
                int(submit[1]),
                int(submit[2]),
                cursor,
                submit_end,
                # 与旧 "submit_residual" 等长，重分类后不增加 merged 字节。
                "submit_tail_gap",
            )


def _iter_v5_shared_register_derived_spans(
    rows: list[tuple[Any, ...]],
    metadata_writer_tasks: set[int],
    metadata_prefix_tasks: set[int],
    submit_topology: str,
) -> Iterator[tuple[int, int, int, int, int, str]]:
    """用 Register 与 metadata 边界补出非 raw 串行段。

    新采集的 task outputs 已属于 Materialize，因此 Register 只合成等待、
    writer metadata 与插入完成发布。writer 名称复用 Register raw 已有的
    auxiliary，直接显示 ordinary TensorMap entry 数，不增加设备字段。
    迁移前 raw 仍按旧 outputs 子区间恢复 metadata epilogue，保证历史
    泳道可重放。
    """

    parents: dict[tuple[int, int], tuple[Any, ...]] = {}
    details: dict[tuple[int, int], tuple[Any, ...]] = {}
    output_details: dict[tuple[int, int], tuple[Any, ...]] = {}
    for row in rows:
        core_id, _block_id, _lane, task_id, _function_id, phase, *_rest = row
        task_key = (int(core_id), int(task_id))
        if phase == "Register":
            parents[task_key] = row
        elif phase == "SharedRegisterPublishMetadata":
            details[task_key] = row
        elif phase in (
            "SharedMaterializePublishTaskOutputs",
            "SharedRegisterPublishTaskOutputs",
        ):
            output_details[task_key] = row

    for task_key in sorted(details):
        parent = parents[task_key]
        detail = details[task_key]
        output_detail = output_details[task_key]
        core_id, block_id, lane, task_id = (
            int(parent[0]),
            int(parent[1]),
            int(parent[2]),
            int(parent[3]),
        )
        if submit_topology in STRICT_INSERT_TOPOLOGIES:
            has_predecessor = any(
                prefix_task < task_id
                for prefix_task in metadata_prefix_tasks
            )
            predecessor_span_name = (
                "register.wait_predecessor_tensormap_insert"
                if has_predecessor
                else "register.enter_tensormap_insert_chain"
            )
        else:
            has_previous_writer = any(
                writer_task < task_id
                for writer_task in metadata_writer_tasks
            )
            predecessor_span_name = (
                "register.wait_predecessor_tensormap_writer"
                if has_previous_writer
                else "register.enter_tensormap_writer_chain"
            )
        publishes_metadata = task_id in metadata_writer_tasks
        publishes_insert_completion = (
            task_id in metadata_prefix_tasks
            if submit_topology in STRICT_INSERT_TOPOLOGIES
            else publishes_metadata
        )
        yield (
            core_id,
            block_id,
            lane,
            int(parent[6]),
            int(detail[6]),
            f"{predecessor_span_name}#{task_id}",
        )
        ordinary_tensormap_entries = int(parent[9])
        writer_metadata_name = (
            (
                "register.publish_writer_metadata"
                if publishes_metadata
                else "register.no_writer_metadata"
            )
            + f"[ordinary_tensormap_entries={ordinary_tensormap_entries}]"
            + f"#{task_id}"
        )
        if output_detail[5] == "SharedRegisterPublishTaskOutputs":
            yield (
                core_id,
                block_id,
                lane,
                int(detail[6]),
                int(output_detail[6]),
                writer_metadata_name,
            )
            yield (
                core_id,
                block_id,
                lane,
                int(output_detail[7]),
                int(detail[7]),
                f"register.publish_metadata_epilogue#{task_id}",
            )
        else:
            yield (
                core_id,
                block_id,
                lane,
                int(detail[6]),
                int(detail[7]),
                writer_metadata_name,
            )
        yield (
            core_id,
            block_id,
            lane,
            int(detail[7]),
            int(parent[7]),
            (
                "register.publish_tensormap_insert_completion"
                if publishes_insert_completion
                else "register.metadata_chain_epilogue"
            ) + f"#{task_id}",
        )


def _merged_item_sort_key(
    item: tuple[Any, ...],
) -> tuple[int, int, int, int, int, str]:
    """按物理轨道建立父区间优先的确定性导入顺序。

    设备在阶段结束时才写父记录，所以 raw 的物理顺序天然是“子事件在前、
    父区间在后”。Perfetto 的同轨 slice 建栈不能直接使用这个落盘顺序。
    这里仅重排离线 merged：同一轨道 start 升序、end 降序，保证外层先
    导入；完全同区间时业务 span 先于 Atomic/DCCI overlay。
    """

    if item and item[0] == "derived":
        _, _core_id, block_id, lane, start, end, name = item
        return (
            int(block_id), _scalar_thread_id(int(lane)), int(start), -int(end),
            0, str(name),
        )
    if item and item[0] == "scalar_task":
        (
            _scalar_task,
            _core_id,
            block_id,
            lane,
            task_id,
            function_id,
            _phase_raw,
            start,
            end,
            _flags,
            _auxiliary,
        ) = item
        kernel_name = KERNEL_NAMES.get(
            int(function_id), f"f{int(function_id)}"
        )
        return (
            int(block_id),
            _scalar_thread_id(int(lane)),
            int(start),
            -int(end),
            3,
            f"task.execute.{kernel_name}#{int(task_id)}",
        )
    (
        _core_id,
        block_id,
        lane,
        _task_id,
        _function_id,
        phase_raw,
        start,
        end,
        flags,
        _auxiliary,
    ) = item
    phase = PHASE_NAMES[str(phase_raw)]
    thread_id = (
        _kernel_thread_id(int(lane))
        if phase in {"kernel", "commit"}
        else _scalar_thread_id(int(lane))
    )
    if phase == "atomic" and int(flags) & ATOMIC_POLL_BATCH:
        # PollBatch 是等待 episode 的父包络；完全同端点时也必须先于
        # task.execute 与 direct Atomic 导入，才能形成稳定的嵌套层级。
        overlay_priority = 2
    elif phase in ("atomic", "dcci"):
        overlay_priority = 4
    else:
        overlay_priority = 1
    return (
        int(block_id), thread_id, int(start), -int(end),
        overlay_priority, str(phase_raw),
    )


# 完成一次 raw 到 merged 的转换，成功时返回事件数、block 数和基准 cycle。
def convert(  # noqa: PLR0912, PLR0915
    input_path: Path,
    output_path: Path,
    *,
    allow_host_cpu_functional: bool = False,
) -> tuple[int, int, int]:
    (
        frequency_hz,
        trace_schema_version,
        rows,
        core_by_block_lane,
        base_cycle,
        capture_metadata,
    ) = _load_and_validate(
        input_path,
        allow_host_cpu_functional=allow_host_cpu_functional,
    )
    _restore_v5_shared_efdrain(
        rows,
        trace_schema_version,
        capture_metadata.get("tensormap_mode"),
    )
    # 禁止原地转换；否则创建临时文件或最终 replace 时可能破坏唯一一份 raw。
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")

    # 始终先写同目录临时文件，完整 flush/fsync 后再原子替换目标；转换失败
    # 时删除临时文件，不把半截 JSON 留作可加载的正式产物。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    # Chrome Trace Event 的 ts/dur 约定使用微秒；displayTimeUnit="ns" 只控制
    # Perfetto 的显示精度。1 GHz A5 counter 因此每 tick 对应 0.001 us。
    factor = 1_000_000.0 / float(frequency_hz)
    blocks = sorted({block_id for block_id, _ in core_by_block_lane})
    # v1 raw 没有 Claim attempted bit。若同一份 capture 确实含逐 atomic
    # 记录，则可以用同核、同 task 且时间被 Claim 完整包含的
    # claim_max 作为实测证据恢复 attempted；不含 atomic 时保留 unknown，
    # 绝不根据 task kind 或 AIC/AIV role 在转换器中猜业务路由。
    legacy_claim_max_spans: dict[tuple[int, int, int, int], list[tuple[int, int]]] = {}
    has_atomic_trace = False
    if trace_schema_version == 1:
        for core_id, block_id, lane, task_id, _, phase, start, end, _, auxiliary in rows:
            if phase != "Atomic":
                continue
            has_atomic_trace = True
            if auxiliary == 4:  # AtomicSite::ClaimMax
                key = (core_id, block_id, lane, task_id)
                legacy_claim_max_spans.setdefault(key, []).append((start, end))
    # 新布局已经把 output descriptor 发布移入 Materialize。对应 Register
    # metadata raw 与离线合成的 writer span 使用完全相同的边界；merged
    # 只保留带 ordinary TensorMap 数量的合成事件，避免同轨同区间互相遮挡。
    # 旧 Register-placement raw 不在此集合中，仍按历史嵌套结构完整输出。
    materialize_output_tasks = {
        (int(row[0]), int(row[3]))
        for row in rows
        if row[5] == "SharedMaterializePublishTaskOutputs"
    }
    # merged 的顺序是显示合同的一部分：先收集 raw 引用与离线派生 span，
    # 再按物理轨道做父区间优先排序。这里只增加轻量 tuple/reference，
    # 不构造数十万份 event dict；JSON 仍逐事件流式写出。
    ordered_items: list[tuple[Any, ...]] = [
        row
        for row in rows
        if not (
            trace_schema_version == 5
            and row[5] == "SharedRegisterPublishMetadata"
            and (int(row[0]), int(row[3]))
                in materialize_output_tasks
        )
    ]
    if trace_schema_version >= 4:
        # DrainReady 在 scalar 控制流中同步调用 ExecuteKernel。Kernel raw 的
        # begin/end 因而也是该 task 占用 scalar 调度路径的权威边界；除了
        # 计算单元轨，还要把同一区间投影到 scalar 轨。这里只复用 raw，
        # 不新增设备记录，也不把投影冒充第二份物理执行。
        ordered_items.extend(
            ("scalar_task", *row)
            for row in rows
            if row[5] == "Kernel" and int(row[4]) >= 0
        )
    if trace_schema_version == 5:
        ordered_items.extend(
            ("derived", *span)
            for span in _iter_v5_cross_core_winner_build_pack_spans(
                rows,
                trace_schema_version,
                capture_metadata.get("tensormap_mode"),
                capture_metadata.get("submit_topology"),
            )
        )
        ordered_items.extend(
            ("derived", *span)
            for span in _iter_v5_cross_core_semantic_gap_spans(
                rows,
                frequency_hz,
                trace_schema_version,
                capture_metadata.get("tensormap_mode"),
                capture_metadata.get("submit_topology"),
            )
        )
        ordered_items.extend(
            ("derived", *span)
            for span in _iter_v5_shared_register_derived_spans(
                rows,
                set(
                    capture_metadata.get(
                        "shared_metadata_writer_tasks", []
                    )
                ),
                set(
                    capture_metadata.get(
                        "shared_metadata_prefix_tasks", []
                    )
                ),
                str(capture_metadata.get("submit_topology", "all_worker_replay")),
            )
        )
        ordered_items.extend(
            ("derived", *span)
            for span in _iter_v5_residual_spans(
                rows,
                capture_metadata.get("tensormap_mode"),
                capture_metadata.get("submit_topology"),
            )
        )
    ordered_items.sort(key=_merged_item_sort_key)
    first = True
    emitted = 0
    # 临时文件的整个生命周期都在 try 内；包括 Ctrl-C 在内的异常都会先清理
    # .tmp 再向上传播。格式/IO 错误由 main 简短报告，Ctrl-C 保留默认中断行为。
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            output.write('{"displayTimeUnit":"ns","metadata":')
            json.dump(capture_metadata, output, ensure_ascii=False, separators=(",", ":"))
            output.write(',"traceEvents":[\n')
            winner_workload = capture_metadata.get("winner_workload")
            if winner_workload is not None:
                first = _emit_event(
                    output,
                    {
                        "ph": "i",
                        "s": "g",
                        "name": "pa_scheduler.capture",
                        "pid": 0,
                        "tid": 0,
                        "ts": 0,
                        "args": {"winner_workload": winner_workload},
                    },
                    first,
                )
                emitted += 1
            aicpu_producer = capture_metadata.get(
                "aicpu_runtime_plan_producer"
            )
            if aicpu_producer is not None:
                # AICPU is a real independent process/track.  Its raw
                # CLOCK_MONOTONIC_RAW interval is mapped only after rebuilding
                # the causal offset intersection; it is never appended as if
                # it were already in the SYS_CNT domain.
                for metadata_event in (
                    {
                        "ph": "M",
                        "name": "process_name",
                        "pid": AICPU_PROCESS_ID,
                        "args": {"name": "AICPU"},
                    },
                    {
                        "ph": "M",
                        "name": "process_sort_index",
                        "pid": AICPU_PROCESS_ID,
                        "args": {"sort_index": -1},
                    },
                    {
                        "ph": "M",
                        "name": "thread_name",
                        "pid": AICPU_PROCESS_ID,
                        "tid": AICPU_THREAD_ID,
                        "args": {"name": AICPU_PRODUCER_PHASE_NAME},
                    },
                    {
                        "ph": "M",
                        "name": "thread_sort_index",
                        "pid": AICPU_PROCESS_ID,
                        "tid": AICPU_THREAD_ID,
                        "args": {"sort_index": AICPU_THREAD_ID},
                    },
                    {
                        "ph": "M",
                        "name": "thread_name",
                        "pid": AICPU_PROCESS_ID,
                        "tid": AICPU_TASK_THREAD_ID,
                        "args": {"name": "TaskPlan"},
                    },
                    {
                        "ph": "M",
                        "name": "thread_sort_index",
                        "pid": AICPU_PROCESS_ID,
                        "tid": AICPU_TASK_THREAD_ID,
                        "args": {"sort_index": AICPU_TASK_THREAD_ID},
                    },
                ):
                    first = _emit_event(output, metadata_event, first)
                producer_begin = int(
                    aicpu_producer["mapped_begin_tick"]
                )
                producer_end = int(
                    aicpu_producer["mapped_end_tick"]
                )
                first = _emit_event(
                    output,
                    {
                        "ph": "X",
                        "name": AICPU_PRODUCER_PHASE_NAME,
                        "cat": "aicpu.orchestrator",
                        "pid": AICPU_PROCESS_ID,
                        "tid": AICPU_THREAD_ID,
                        "ts": round(
                            (producer_begin - base_cycle) * factor, 3
                        ),
                        "dur": round(
                            (producer_end - producer_begin) * factor, 3
                        ),
                        "args": {
                            "raw_clock": aicpu_producer["raw_clock"],
                            "raw_begin_ns": aicpu_producer[
                                "raw_begin_ns"
                            ],
                            "raw_end_ns": aicpu_producer["raw_end_ns"],
                            "mapped_begin_tick": producer_begin,
                            "mapped_end_tick": producer_end,
                            "alignment_error_tick": aicpu_producer[
                                "alignment_error_tick"
                            ],
                            "alignment_error_us": aicpu_producer[
                                "alignment_error_us"
                            ],
                            "correlation_status": "valid",
                        },
                    },
                    first,
                )
                emitted += 1
                for owner_phase in capture_metadata[
                    "aicpu_runtime_plan_scheduler_phases"
                ]:
                    phase_begin = int(owner_phase["mapped_begin_tick"])
                    phase_end = int(owner_phase["mapped_end_tick"])
                    first = _emit_event(
                        output,
                        {
                            "ph": "X",
                            "name": owner_phase["name"],
                            "cat": "aicpu.owner",
                            "pid": AICPU_PROCESS_ID,
                            "tid": AICPU_THREAD_ID,
                            "ts": round(
                                (phase_begin - base_cycle) * factor, 3
                            ),
                            "dur": round(
                                (phase_end - phase_begin) * factor, 3
                            ),
                            "args": {
                                "raw_clock": owner_phase["raw_clock"],
                                "raw_begin_ns": owner_phase["raw_begin_ns"],
                                "raw_end_ns": owner_phase["raw_end_ns"],
                            },
                        },
                        first,
                    )
                    emitted += 1
                for task in capture_metadata[
                    "aicpu_runtime_plan_tasks"
                ]:
                    task_id = int(task["task_id"])
                    task_kind = str(task["task_kind"])
                    common_args = {
                        "task_id": task_id,
                        "task_kind": task_kind,
                        "function_id": int(task["function_id"]),
                        "engine": task["engine"],
                        "group": int(task["group"]),
                        "output_count": int(task["output_count"]),
                        "payload_lines": int(task["payload_lines"]),
                    }
                    task_begin = int(task["mapped_build_begin_tick"])
                    task_end = int(task["mapped_publish_end_tick"])
                    first = _emit_event(
                        output,
                        {
                            "ph": "X",
                            "name": f"task_plan.{task_kind}#{task_id}",
                            "cat": "aicpu.task_plan",
                            "pid": AICPU_PROCESS_ID,
                            "tid": AICPU_TASK_THREAD_ID,
                            "ts": round(
                                (task_begin - base_cycle) * factor, 3
                            ),
                            "dur": round(
                                (task_end - task_begin) * factor, 3
                            ),
                            "args": common_args,
                        },
                        first,
                    )
                    emitted += 1
                    detail_spans = (
                        (
                            "begin",
                            int(task["mapped_build_begin_tick"]),
                            int(task["mapped_begin_end_tick"]),
                        ),
                        (
                            "orchestration",
                            int(task["mapped_begin_end_tick"]),
                            int(task["mapped_finish_begin_tick"]),
                        ),
                        (
                            "stage_payload",
                            int(task["mapped_finish_begin_tick"]),
                            int(task["mapped_finish_end_tick"]),
                        ),
                        (
                            "defer_publish",
                            int(task["mapped_finish_end_tick"]),
                            int(task["mapped_publish_begin_tick"]),
                        ),
                        (
                            "publish",
                            int(task["mapped_publish_begin_tick"]),
                            int(task["mapped_publish_end_tick"]),
                        ),
                    )
                    for detail_name, detail_begin, detail_end in detail_spans:
                        first = _emit_event(
                            output,
                            {
                                "ph": "X",
                                "name": (
                                    f"task_plan.{detail_name}#{task_id}"
                                ),
                                "cat": "aicpu.task_plan.detail",
                                "pid": AICPU_PROCESS_ID,
                                "tid": AICPU_TASK_THREAD_ID,
                                "ts": round(
                                    (detail_begin - base_cycle) * factor,
                                    3,
                                ),
                                "dur": round(
                                    (detail_end - detail_begin) * factor,
                                    3,
                                ),
                                "args": common_args,
                            },
                            first,
                        )
                        emitted += 1
                for operation in capture_metadata[
                    "aicpu_runtime_plan_operations"
                ]:
                    task_id = int(operation["task_id"])
                    scope = str(operation["scope"])
                    operation_name = str(operation["operation"])
                    target = str(operation["target"])
                    calls = int(operation["calls"])
                    lines = int(operation["lines"])
                    suffix = f"#{task_id}" if task_id >= 0 else "#owner"
                    call_suffix = f"×{calls}" if calls > 1 else ""
                    if operation_name in {
                        "atomic_load_acquire", "atomic_store_release"
                    }:
                        category = "aicpu.atomic"
                        instruction = (
                            "load_acquire"
                            if operation_name == "atomic_load_acquire"
                            else "store_release"
                        )
                        event_name = (
                            f"atomic.{scope}.{target}.{instruction}"
                            f"{call_suffix}{suffix}"
                        )
                    elif operation_name in {
                        "cache_clean_cvac", "cache_discard_civac"
                    }:
                        category = "aicpu.cache"
                        instruction = (
                            "dc_cvac"
                            if operation_name == "cache_clean_cvac"
                            else "dc_civac"
                        )
                        event_name = (
                            f"cache.{scope}.{target}.{instruction}"
                            f"×{calls}.lines{lines}{suffix}"
                        )
                    elif operation_name in {
                        "barrier_dsb_sy", "barrier_isb"
                    }:
                        category = "aicpu.barrier"
                        instruction = (
                            "dsb_sy"
                            if operation_name == "barrier_dsb_sy"
                            else "isb"
                        )
                        event_name = (
                            f"barrier.{scope}.{instruction}"
                            f"{call_suffix}{suffix}"
                        )
                    elif operation_name == "gm_store":
                        category = "aicpu.gm"
                        event_name = (
                            f"gm.{scope}.{target}.store"
                            f"{call_suffix}{suffix}"
                        )
                    else:
                        category = "aicpu.scalar"
                        event_name = (
                            f"scalar.{scope}.{target}"
                            f"{call_suffix}{suffix}"
                        )
                    operation_begin = int(
                        operation["mapped_begin_tick"]
                    )
                    operation_end = int(operation["mapped_end_tick"])
                    first = _emit_event(
                        output,
                        {
                            "ph": "X",
                            "name": event_name,
                            "cat": category,
                            "pid": AICPU_PROCESS_ID,
                            "tid": (
                                AICPU_TASK_THREAD_ID
                                if task_id >= 0
                                else AICPU_THREAD_ID
                            ),
                            "ts": round(
                                (operation_begin - base_cycle) * factor,
                                3,
                            ),
                            "dur": round(
                                (operation_end - operation_begin) * factor,
                                3,
                            ),
                            "args": {
                                "sequence": int(operation["sequence"]),
                                "task_id": task_id,
                                "scope": scope,
                                "operation": operation_name,
                                "target": target,
                                "calls": calls,
                                "lines": lines,
                                "first_target_index": int(
                                    operation["first_target_index"]
                                ),
                                "last_target_index": int(
                                    operation["last_target_index"]
                                ),
                                "first_value": int(
                                    operation["first_value"]
                                ),
                                "last_value": int(
                                    operation["last_value"]
                                ),
                                "raw_clock": operation["raw_clock"],
                                "raw_begin_ns": int(
                                    operation["raw_begin_ns"]
                                ),
                                "raw_end_ns": int(
                                    operation["raw_end_ns"]
                                ),
                            },
                        },
                        first,
                    )
                    emitted += 1
            # 每个物理 block 建一个 process；每条硬件 lane 再拆成 runtime
            # 与 kernel 两个 thread，避免等待/提交阶段覆盖 kernel 执行条。
            for block_id in blocks:
                first = _emit_event(
                    output,
                    {"ph": "M", "name": "process_name", "pid": block_id, "args": {"name": f"block{block_id}"}},
                    first,
                )
                first = _emit_event(
                    output,
                    {
                        "ph": "M",
                        "name": "process_sort_index",
                        "pid": block_id,
                        "args": {"sort_index": block_id},
                    },
                    first,
                )
                # 同一 block 固定先列出三条 scalar，再列出三条 kernel。
                # thread_sort_index 是 Perfetto 的权威显示顺序；TID 也采用
                # 同样的 1..6 排列，确保不识别该元数据的查看器仍能稳定回退。
                thread_tracks: list[tuple[int, str]] = []
                for lane, lane_name in LANE_NAMES.items():
                    core_id = core_by_block_lane.get((block_id, lane))
                    if core_id is not None:
                        thread_tracks.append(
                            (
                                _scalar_thread_id(lane),
                                f"{lane_name} (core{core_id})",
                            )
                        )
                for lane, lane_name in LANE_NAMES.items():
                    core_id = core_by_block_lane.get((block_id, lane))
                    if core_id is not None:
                        thread_tracks.append(
                            (
                                _kernel_thread_id(lane),
                                f"{lane_name}·kernel (core{core_id})",
                            )
                        )
                for thread_id, thread_name in thread_tracks:
                    first = _emit_event(
                        output,
                        {
                            "ph": "M",
                            "name": "thread_name",
                            "pid": block_id,
                            "tid": thread_id,
                            "args": {"name": thread_name},
                        },
                        first,
                    )
                    first = _emit_event(
                        output,
                        {
                            "ph": "M",
                            "name": "thread_sort_index",
                            "pid": block_id,
                            "tid": thread_id,
                            "args": {"sort_index": thread_id},
                        },
                        first,
                    )
            for item in ordered_items:
                if item[0] == "scalar_task":
                    (
                        _scalar_task,
                        core_id,
                        block_id,
                        lane,
                        task_id,
                        function_id,
                        _phase_raw,
                        start,
                        end,
                        _flags,
                        _auxiliary,
                    ) = item
                    kernel_name = KERNEL_NAMES.get(
                        int(function_id), f"f{int(function_id)}"
                    )
                    event = {
                        "ph": "X",
                        "name": (
                            f"task.execute.{kernel_name}#{int(task_id)}"
                        ),
                        "pid": int(block_id),
                        "tid": _scalar_thread_id(int(lane)),
                        "ts": round((int(start) - base_cycle) * factor, 3),
                        "dur": round((int(end) - int(start)) * factor, 3),
                        "args": {
                            "phase": "scalar_task_execution",
                            "task_id": int(task_id),
                            "func_id": int(function_id),
                            "core": int(core_id),
                            "kernel": kernel_name,
                            "execution_unit": "scalar",
                            "projection_source": (
                                "synchronous_execute_kernel_bracket"
                            ),
                        },
                        "cat": "scalar_task",
                    }
                    if trace_schema_version == 5:
                        event.pop("args")
                        event.pop("cat")
                    first = _emit_event(output, event, first)
                    emitted += 1
                    continue
                if item[0] == "derived":
                    (
                        _derived,
                        _core_id,
                        block_id,
                        lane,
                        start,
                        end,
                        name,
                    ) = item
                    first = _emit_event(
                        output,
                        {
                            "ph": "X",
                            "name": name,
                            "pid": block_id,
                            "tid": _scalar_thread_id(lane),
                            "ts": round(
                                (start - base_cycle) * factor, 3
                            ),
                            "dur": round(
                                (end - start) * factor, 3
                            ),
                        },
                        first,
                    )
                    emitted += 1
                    continue
                row = item
                core_id, block_id, lane, task_id, function_id, phase_raw, start, end, flags, auxiliary = row
                phase = PHASE_NAMES[phase_raw]
                scalar_thread_id = _scalar_thread_id(lane)
                kernel_thread_id = _kernel_thread_id(lane)
                # Kernel/Commit 放到对应计算单元子泳道；Atomic/ClockBaseline
                # 都是 AIC/AIV 对应 scalar 上执行的指令，必须与 runtime 阶段共用
                # 非零 scalar TID。这样既避免 tid=0 被 Perfetto 重映射为主线程，
                # 又让 atomic span 作为 Claim/Fanin/轮询等阶段的子区间叠加显示，
                # 不会伪装成 AIC/AIV 之外的第三类执行单元。
                if phase == "claim":
                    claim_attempted: bool | None
                    claim_attempted_source: str
                    if trace_schema_version >= 2:
                        claim_attempted = bool(flags & 0x2)
                        claim_attempted_source = "raw_flag"
                    elif has_atomic_trace:
                        key = (core_id, block_id, lane, task_id)
                        matched_claim_max = any(
                            atomic_start >= start and atomic_end <= end
                            for atomic_start, atomic_end in legacy_claim_max_spans.get(key, [])
                        )
                        # 命中的 claim_max 能正向证明 attempted；但 v1 raw 没有
                        # 显式“atomic 记录完整”元数据，未命中不能反向证明
                        # not_attempted，因此保留 unknown。
                        claim_attempted = True if matched_claim_max else None
                        claim_attempted_source = (
                            "contained_claim_max"
                            if matched_claim_max
                            else "unknown_v1_without_matching_claim_max"
                        )
                    else:
                        claim_attempted = None
                        claim_attempted_source = "unknown_v1_without_atomic_trace"
                    claim_won = bool(flags & 0x1)
                    if claim_attempted is False:
                        name = f"claim.not_attempted#{task_id}"
                    elif claim_attempted is True:
                        name = f"claim.{'won' if claim_won else 'lost'}#{task_id}"
                    else:
                        name = f"claim#{task_id}"
                    thread_id = scalar_thread_id
                elif phase == "atomic":
                    atomic_site_id = auxiliary
                    atomic_op_id = flags & 0xF
                    atomic_site = ATOMIC_SITE_NAMES.get(atomic_site_id, f"site_{atomic_site_id}")
                    atomic_op = ATOMIC_OP_NAMES.get(atomic_op_id, f"op_{atomic_op_id}")
                    atomic_poll_batch = (
                        trace_schema_version >= 3 and bool(flags & ATOMIC_POLL_BATCH)
                    )
                    if atomic_poll_batch:
                        atomic_call_count = (
                            flags >> ATOMIC_PAYLOAD_SHIFT
                        ) & ATOMIC_PAYLOAD_MASK
                        if (
                            atomic_site_id in AGGREGATE_ONLY_POLL_SITE_IDS
                            or atomic_site_id
                            in SHARED_OUTPUT_PUBLISHED_POLL_SITE_IDS
                        ):
                            poll_boundary_tag = (
                                "return_ready"
                                if flags & ATOMIC_RETURN_READY
                                else "source_issue"
                            )
                            name = (
                                f"atomic.poll_batch.{poll_boundary_tag}."
                                f"{atomic_site}.{atomic_op}"
                                f"×{atomic_call_count}"
                            )
                        else:
                            name = (
                                f"atomic.poll_batch.{atomic_site}.{atomic_op}"
                                f"×{atomic_call_count}"
                            )
                    else:
                        # 边界直接写入 span 名称，打开泳道后无需点开 args
                        # 就能区分“本核返回值可消费”和“只包围源码发射”。
                        atomic_boundary_tag = (
                            "return_ready"
                            if flags & ATOMIC_RETURN_READY
                            else "source_issue"
                        )
                        name = (
                            f"atomic.{atomic_boundary_tag}.{atomic_site}."
                            f"{atomic_op}#{task_id}"
                        )
                    thread_id = scalar_thread_id
                elif phase == "dcci":
                    dcci_site_id = auxiliary
                    dcci_op_id = flags & DCCI_OP_MASK
                    dcci_site = DCCI_SITE_NAMES.get(
                        dcci_site_id, f"site_{dcci_site_id}"
                    )
                    dcci_op = DCCI_OP_NAMES.get(
                        dcci_op_id, f"op_{dcci_op_id}"
                    )
                    dcci_call_count = (
                        flags >> DCCI_CALL_COUNT_SHIFT
                    ) & DCCI_CALL_COUNT_MASK
                    dcci_line_count = (
                        flags >> DCCI_LINE_COUNT_SHIFT
                    ) & DCCI_LINE_COUNT_MASK
                    name = (
                        f"dcci.{dcci_site}.{dcci_op}"
                        f"×{dcci_call_count}.lines{dcci_line_count}"
                        f"#{task_id}"
                    )
                    thread_id = scalar_thread_id
                elif phase == "clock_baseline":
                    name = (
                        "clock.atomic_return_dependency_hook"
                        if flags & 1
                        else "clock.consecutive_sys_cnt_reads"
                    )
                    thread_id = scalar_thread_id
                elif phase == "kernel" and function_id >= 0:
                    name = f"{KERNEL_NAMES.get(function_id, f'f{function_id}')}#{task_id}"
                    thread_id = kernel_thread_id
                elif phase == "commit":
                    name = f"commit#{task_id}"
                    thread_id = kernel_thread_id
                elif phase in (
                    "orchestration_replay", "runtime_plan_build",
                    "final_drain",
                ):
                    name = phase
                    thread_id = scalar_thread_id
                else:
                    name = f"{phase}#{task_id}"
                    thread_id = scalar_thread_id
                event = {
                    "ph": "X",
                    "name": name,
                    "pid": block_id,
                    "tid": thread_id,
                    "ts": round((start - base_cycle) * factor, 3),
                    "dur": round((end - start) * factor, 3),
                    "args": {
                        "phase": phase,
                        "task_id": task_id,
                        "func_id": function_id,
                        "core": core_id,
                        # mc 是兼容真实 merged schema 的字段名，只原样承载 flags
                        # bit0；它与 aux 的实际含义均需结合 phase 解读，例如 Claim
                        # 可表示 winner，而 Fanin/HeapGuard 的 aux 各有自己的计数语义。
                        "mc": flags & 1,
                        "aux": auxiliary,
                    },
                }
                if phase == "atomic":
                    # PollBatch 的 start/end 是逻辑等待 episode 包络，
                    # 不是单次 atomic completion boundary，也不是其中
                    # call_count 次 load 的独占耗时。保留 X 区间是为了让
                    # Perfetto 按真实包含关系把其他 poll、task.execute 和
                    # direct Atomic 画到下层；不得用 dur/call_count 估算单次
                    # atomic 延迟。
                    if atomic_poll_batch:
                        event["args"] = {
                            "phase": "atomic_poll_batch",
                            "task_id": task_id,
                            "func_id": function_id,
                            "core": core_id,
                            "site": atomic_site,
                            "site_id": atomic_site_id,
                            "op": atomic_op,
                            "op_id": atomic_op_id,
                            "call_count": atomic_call_count,
                            "poll_window_cycles": end - start,
                            "estimate_formula": "call_count * calibrated_atomic_cost",
                            "is_poll_batch": True,
                            "batch_semantics": "observation_load_calls",
                            "duration_semantics": (
                                "logical_poll_episode_envelope_not_single_atomic_latency"
                            ),
                            "may_contain_interleaved_direct_atomics": True,
                            "flags": flags,
                            "execution_unit": "scalar",
                        }
                        event["cat"] = "atomic.poll_batch"
                    else:
                        # 直接 Atomic 的 flags/aux 有独立 ABI，不沿用普通
                        # phase 的 mc 语义。cycles 保留原始整数，避免短
                        # atomic 经微秒浮点换算后丢失 tick 精度。
                        event["args"] = {
                            "phase": phase,
                            "task_id": task_id,
                            "func_id": function_id,
                            "core": core_id,
                            "site": atomic_site,
                            "site_id": atomic_site_id,
                            "op": atomic_op,
                            "op_id": atomic_op_id,
                            "call_count": 1,
                            "cycles": end - start,
                            "result_used": bool(flags & ATOMIC_RESULT_USED),
                            "return_ready_observed": bool(flags & ATOMIC_RETURN_READY),
                            "completion_boundary": (
                                "return_value_ready"
                                if flags & ATOMIC_RETURN_READY
                                else "source_issue_bracket"
                            ),
                            "flags": flags,
                            "execution_unit": "scalar",
                        }
                        # 分类同样带边界，便于 Perfetto 过滤和分组；二者
                        # 仍在同一 AIC/AIV scalar lane，不伪造并行执行单元。
                        event["cat"] = f"atomic.{atomic_boundary_tag}"
                        # bit5 只对 Load 有意义；bits8..31 只对 FetchMax
                        # 表示饱和后的 retry 数。
                        if atomic_op_id == 0:
                            event["args"]["value_zero"] = bool(
                                flags & ATOMIC_VALUE_ZERO
                            )
                        if atomic_op_id == 3:
                            event["args"]["retries"] = (
                                flags >> ATOMIC_PAYLOAD_SHIFT
                            ) & ATOMIC_PAYLOAD_MASK
                elif phase == "dcci":
                    event["args"] = {
                        "phase": phase,
                        "task_id": task_id,
                        "func_id": function_id,
                        "core": core_id,
                        "site": dcci_site,
                        "site_id": dcci_site_id,
                        "op": dcci_op,
                        "op_id": dcci_op_id,
                        "call_count": dcci_call_count,
                        "cache_line_count": dcci_line_count,
                        "trailing_dsb": bool(flags & DCCI_TRAILING_DSB),
                        "cycles": end - start,
                        "execution_unit": "scalar",
                        "flags": flags,
                    }
                    event["cat"] = "dcci"
                elif phase == "claim":
                    event["args"] = {
                        "phase": phase,
                        "task_id": task_id,
                        "func_id": function_id,
                        "core": core_id,
                        "claim_attempted": claim_attempted,
                        "claim_won": claim_won,
                        "claim_attempted_source": claim_attempted_source,
                        "claim_path": "alloc" if auxiliary == 1 else "kernel",
                        "execution_unit": "scalar",
                        "flags": flags,
                    }
                    event["cat"] = "scalar_scheduler"
                elif phase == "clock_baseline":
                    dependency_hook = bool(flags & 1)
                    event["args"] = {
                        "phase": phase,
                        "core": core_id,
                        "ticks": end - start,
                        "clock_freq_hz": frequency_hz,
                        "definition": (
                            "atomic-return-dependency-hook"
                            if dependency_hook
                            else "consecutive-sys-cnt-reads"
                        ),
                        "dependency_applied": bool(flags & 2) if dependency_hook else False,
                        "execution_unit": "scalar",
                    }
                    event["cat"] = "scalar_clock"
                if trace_schema_version == 5:
                    # schema-v5 的 merged 只承担可视化：阶段、atomic site/op/
                    # boundary、task 和 poll 次数均已编码在 name，轨道与时间由
                    # pid/tid/ts/dur 给出。十列权威字段完整保留在同目录 raw，
                    # 不再逐事件复制近 100 MiB 的 args/cat。
                    event.pop("args", None)
                    event.pop("cat", None)
                first = _emit_event(output, event, first)
                emitted += 1
            output.write("\n]}\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return emitted, len(blocks), base_cycle


# 只解析显式 input/output，不扫描仓库 outputs，也不选择“最新”文件。
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # 强制 -o 使覆盖目标可审查，避免脱仓后因 cwd 不同写到意外目录。
    parser = argparse.ArgumentParser(
        description="Convert standalone PA fdwic_events JSON to a Chrome/Perfetto swimlane trace."
    )
    parser.add_argument("input", type=Path, help="l2_swimlane_records.json produced by the standalone runner")
    parser.add_argument("-o", "--output", type=Path, required=True, help="merged_swimlane.json output path")
    parser.add_argument(
        "--host-cpu-functional",
        action="store_true",
        help=(
            "explicitly accept a host-cpu functional capture; output is "
            "marked non-joint and is not profiling evidence"
        ),
    )
    return parser.parse_args(argv)


# 命令行错误边界：预期的输入、格式和文件系统错误统一返回 1。
def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        events, blocks, base_cycle = convert(
            args.input,
            args.output,
            allow_host_cpu_functional=args.host_cpu_functional,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        # 不吞掉错误原因，但也不向普通使用者输出长 traceback；convert 已保证
        # 失败路径不会留下临时 merged 文件。
        print(f"swimlane conversion failed: {error}", file=sys.stderr)
        return 1
    print(
        f"[SWIMLANE] merged_json={args.output} events={events} blocks={blocks} base_cycle={base_cycle}"
    )
    print(f"Open https://ui.perfetto.dev/ and load {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
