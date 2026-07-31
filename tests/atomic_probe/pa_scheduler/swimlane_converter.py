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
from collections.abc import Iterator
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
        "materialize.publish_shared_output_descriptors.copy_tensor_descs"
    ),
    "SharedMaterializePublishTaskOutputsFlush": (
        "materialize.publish_shared_output_descriptors.flush_tensor_descs"
    ),
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
# shared Register 插入轮次的等待 Load 与交接 CAS；真实 PA 的 BlockWon
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
}
# 这些发布型调用不消费 atomic 返回的旧值；其余 standalone site 的
# 返回值都参与协议判断。v3 输入必须与源码语义完全一致。
ATOMIC_RESULT_UNUSED_SITE_IDS = {0, 3, 6, 7, 13, 39}
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
}
SHARED_REGISTER_ATOMIC_SITE_IDS = {19, 20}
SCHEMA_V5_SHARED_ATOMIC_SITE_IDS = set(range(19, 42))
SHARED_INSERT_TURN_POLL_SITE_ID = 19
SHARED_INSERT_TURN_HANDOFF_SITE_ID = 20
SHARED_CLAIM_TOURNAMENT_SITE_IDS = {40, 41}

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
}
DCCI_SHARED_ONLY_SITE_IDS = set(range(8))
DCCI_OBSERVER_SITE_ID = 8
DCCI_STARTUP_SITE_ID = 9
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


def _derive_v4_task_kinds(
    submit_semantics: dict[tuple[int, int], tuple[bool, bool]],
    num_cores: int,
) -> dict[int, int]:
    """从每核 Submit 的既有 Alloc 标记恢复动态 task 类型流。

    schema-v5 的 ``Submit.auxiliary`` 已逐核记录 ``is_alloc``，因此无需给
    设备 raw 再增加 task-kind 字段。这里先要求 96 核（或测试给定核数）
    具有完全一致的连续 task 流，再按相邻 Alloc 边界验证每个 batch 必须是
    ``Alloc + 0..4 × (QK,SF,PV,UP)``。返回值使用稳定 TaskKind 编号：
    Alloc=0，QK/SF/PV/UP=1..4。
    """

    if num_cores <= 0:
        raise ValueError(f"schema-v5 task plan requires positive num_cores, got {num_cores}")

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
            submit_semantics[(core_id, task_id)][1]
            for core_id in range(num_cores)
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


# 读取 raw JSON，校验十列结构、字段范围与可转整数值，并返回规范化视图。
def _load_and_validate(  # noqa: PLR0912, PLR0915
    input_path: Path,
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
    if trace_schema_version == 5:
        if tensormap_mode not in ("private", "shared"):
            raise ValueError(
                "metadata.tensormap_mode must be private or shared for "
                "trace_schema_version=5"
            )
    elif tensormap_mode is not None:
        raise ValueError(
            "metadata.tensormap_mode is only valid for trace_schema_version=5"
        )
    num_cores = _integer(metadata.get("num_cores"), "metadata.num_cores")
    if num_cores <= 0:
        raise ValueError("metadata.num_cores must be positive")
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
    v3_clock_rows: dict[int, dict[str, int | bool | None]] = {
        core_id: {"plain": 0, "dependency": 0, "return_ready": None}
        for core_id in range(num_cores)
    }
    v3_result_used_direct_rows: list[tuple[int, int, bool]] = []
    v3_insert_turn_poll_batch_rows: list[tuple[int, int, bool]] = []
    v5_observer_dcci_rows = {core_id: 0 for core_id in range(num_cores)}
    v4_parent_counts: dict[int, dict[str, int]] = {
        core_id: {"OrchestrationReplay": 0, "FinalDrain": 0}
        for core_id in range(num_cores)
    }
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
            if auxiliary in SCHEMA_V5_SHARED_ATOMIC_SITE_IDS and not (
                trace_schema_version == 5 and tensormap_mode == "shared"
            ):
                raise ValueError(
                    f"fdwic_events[{index}] has invalid direct Atomic "
                    f"site={auxiliary}: shared schema-v5 site requires "
                    "shared schema-v5"
                )
            if (
                auxiliary == SHARED_INSERT_TURN_POLL_SITE_ID
                and not poll_batch
            ):
                raise ValueError(
                    f"fdwic_events[{index}] SharedInsertTurnPoll must use PollBatch"
                )
            if poll_batch:
                return_ready_valid = (
                    not return_ready
                    or auxiliary == SHARED_INSERT_TURN_POLL_SITE_ID
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
                if auxiliary == SHARED_INSERT_TURN_POLL_SITE_ID:
                    v3_insert_turn_poll_batch_rows.append(
                        (index, core_id, return_ready)
                    )
            elif trace_schema_version in (3, 5):
                if level != 4:
                    raise ValueError(
                        f"fdwic_events[{index}] Atomic requires l2_swimlane_level=4"
                    )
                expected_result_used = (
                    auxiliary in ATOMIC_SITE_OP_IDS
                    and auxiliary not in ATOMIC_RESULT_UNUSED_SITE_IDS
                )
                if (
                    ATOMIC_SITE_OP_IDS.get(auxiliary) != atomic_op
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
            if phase in ("OrchestrationReplay", "FinalDrain"):
                if task_id != -1 or function_id != -1 or flags != 0 or auxiliary != 0:
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid schema-v5 parent {phase} fields"
                    )
                v4_parent_counts[core_id][phase] += 1
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
        for row_index, core_id, return_ready in v3_insert_turn_poll_batch_rows:
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
            for phase, count in counts.items():
                if count != 1:
                    raise ValueError(
                        f"core {core_id} requires exactly one schema-v5 {phase}: count={count}"
                    )
        if set(v4_claims) != v4_submits:
            raise ValueError("schema-v5 Claim keys do not match Submit keys")
        task_kind_by_id = _derive_v4_task_kinds(v4_submit_semantics, num_cores)
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
                    # per-task predecessor chain 中 task 0 没有前驱，不执行
                    # insert-turn Load；其余 task 的 winner 各产生一条聚合
                    # PollBatch。Register 的前段仍由父/detail 边界表示，
                    # 不能因为 task 0 没有 atomic 就删掉该闭合区间。
                    expected_poll_count = 0 if task_key[1] == 0 else 1
                    if len(matching_polls) != expected_poll_count:
                        raise ValueError(
                            "shared schema-v5 level4 requires exactly one "
                            "SharedInsertTurnPoll PollBatch for every nonzero-task "
                            "winner and none for task 0 on "
                            "Register.start->metadata.start at "
                            f"{task_key}: count={len(matching_polls)} "
                            f"expected={expected_poll_count}"
                        )
                    handoffs = v4_shared_insert_turn_handoffs.get(
                        task_key, []
                    )
                    if len(handoffs) != 1:
                        raise ValueError(
                            "shared schema-v5 level4 requires exactly one "
                            "SharedInsertTurnHandoff direct CAS per winner at "
                            f"{task_key}: count={len(handoffs)}"
                        )
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
                            "shared schema-v5 SharedInsertTurnHandoff identity "
                            "or boundary is outside metadata.end->Register.end "
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
                nonzero_task_winner_count = sum(
                    1
                    for (_core_id, task_id), (
                        _attempted,
                        won,
                        _is_alloc,
                    ) in v4_claims.items()
                    if won and task_id > 0
                )
                if (
                    len(v4_shared_insert_turn_polls)
                    != nonzero_task_winner_count
                ):
                    raise ValueError(
                        "shared schema-v5 level4 has orphan or duplicate "
                        "SharedInsertTurnPoll records: "
                        f"records={len(v4_shared_insert_turn_polls)} "
                        "expected_nonzero_task_winners="
                        f"{nonzero_task_winner_count} "
                        f"all_winners={winner_count}"
                    )
                handoff_count = sum(
                    len(items)
                    for items in v4_shared_insert_turn_handoffs.values()
                )
                if handoff_count != winner_count:
                    raise ValueError(
                        "shared schema-v5 level4 has orphan or duplicate "
                        "SharedInsertTurnHandoff records: "
                        f"records={handoff_count} winners={winner_count}"
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


# 写一个 Chrome Trace Event，并统一处理数组元素间的逗号。
def _emit_event(output: TextIO, event: dict[str, Any], first: bool) -> bool:
    # 逐事件写出，避免再在内存中构造一份体积可达数百 MiB 的 merged 列表。
    if not first:
        output.write(",\n")
    json.dump(event, output, ensure_ascii=False, separators=(",", ":"))
    return False


def _iter_v5_residual_spans(
    rows: list[tuple[Any, ...]],
) -> Iterator[tuple[int, int, int, int, int, str]]:
    """只用既有 Submit/child 边界生成逐段补集，不改 raw ABI。"""

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
                yield (
                    int(submit[0]),
                    int(submit[1]),
                    int(submit[2]),
                    cursor,
                    child_start,
                    "submit_residual",
                )
            cursor = max(cursor, child_end)
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
        yield (
            core_id,
            block_id,
            lane,
            int(parent[6]),
            int(detail[6]),
            f"register.wait_predecessor_tensormap_insert#{task_id}",
        )
        ordinary_tensormap_entries = int(parent[9])
        writer_metadata_name = (
            "register.publish_writer_metadata"
            f"[ordinary_tensormap_entries={ordinary_tensormap_entries}]"
            f"#{task_id}"
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
            f"register.publish_tensormap_insert_completion#{task_id}",
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
            int(block_id), int(lane), int(start), -int(end),
            0, str(name),
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
        _flags,
        _auxiliary,
    ) = item
    phase = PHASE_NAMES[str(phase_raw)]
    thread_id = (
        int(lane) + 3
        if phase in {"kernel", "commit"}
        else int(lane)
    )
    overlay_priority = 2 if phase in ("atomic", "dcci") else 1
    return (
        int(block_id), thread_id, int(start), -int(end),
        overlay_priority, str(phase_raw),
    )


# 完成一次 raw 到 merged 的转换，成功时返回事件数、block 数和基准 cycle。
def convert(  # noqa: PLR0912, PLR0915
    input_path: Path, output_path: Path
) -> tuple[int, int, int]:
    (
        frequency_hz,
        trace_schema_version,
        rows,
        core_by_block_lane,
        base_cycle,
        capture_metadata,
    ) = _load_and_validate(input_path)
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
    if trace_schema_version == 5:
        ordered_items.extend(
            ("derived", *span)
            for span in _iter_v5_shared_register_derived_spans(
                rows
            )
        )
        ordered_items.extend(
            ("derived", *span)
            for span in _iter_v5_residual_spans(rows)
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
                for lane, lane_name in LANE_NAMES.items():
                    core_id = core_by_block_lane.get((block_id, lane))
                    if core_id is None:
                        continue
                    for thread_id, thread_name in (
                        (lane, f"{lane_name} (core{core_id})"),
                        (lane + 3, f"{lane_name}·kernel (core{core_id})"),
                    ):
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
            for item in ordered_items:
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
                            "tid": lane,
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
                # Kernel/Commit 放到 lane+3 的计算单元子泳道；Atomic/ClockBaseline
                # 都是 AIC/AIV 对应 scalar 上执行的指令，必须与 runtime 阶段共用
                # lane 0..2。这样 atomic span 作为 Claim/Fanin/轮询等阶段的子区间
                # 叠加显示，不会伪装成 AIC/AIV 之外的第三类执行单元。
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
                    thread_id = lane
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
                            atomic_site_id
                            == SHARED_INSERT_TURN_POLL_SITE_ID
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
                    thread_id = lane
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
                    thread_id = lane
                elif phase == "clock_baseline":
                    name = (
                        "clock.atomic_return_dependency_hook"
                        if flags & 1
                        else "clock.consecutive_sys_cnt_reads"
                    )
                    thread_id = lane
                elif phase == "kernel" and function_id >= 0:
                    name = f"{KERNEL_NAMES.get(function_id, f'f{function_id}')}#{task_id}"
                    thread_id = lane + 3
                elif phase == "commit":
                    name = f"commit#{task_id}"
                    thread_id = lane + 3
                elif phase in ("orchestration_replay", "final_drain"):
                    name = phase
                    thread_id = lane
                else:
                    name = f"{phase}#{task_id}"
                    thread_id = lane
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
                    # PollBatch 表示显式等待区内的逻辑调用次数；它的 span
                    # 只是 episode 包络，可能与其他 site 或直接 Atomic 交错，
                    # 因而绝不能伪装成一次 atomic 的 completion boundary。
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
def _parse_args() -> argparse.Namespace:
    # 强制 -o 使覆盖目标可审查，避免脱仓后因 cwd 不同写到意外目录。
    parser = argparse.ArgumentParser(
        description="Convert standalone PA fdwic_events JSON to a Chrome/Perfetto swimlane trace."
    )
    parser.add_argument("input", type=Path, help="l2_swimlane_records.json produced by the standalone runner")
    parser.add_argument("-o", "--output", type=Path, required=True, help="merged_swimlane.json output path")
    return parser.parse_args()


# 命令行错误边界：预期的输入、格式和文件系统错误统一返回 1。
def main() -> int:
    args = _parse_args()
    try:
        events, blocks, base_cycle = convert(args.input, args.output)
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
