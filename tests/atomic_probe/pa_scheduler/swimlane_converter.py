#!/usr/bin/env python3
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
    "ClockBaseline": "clock_baseline",
    "OrchestrationReplay": "orchestration_replay",
    "FinalDrain": "final_drain",
    "WinnerBuild": "winner_build",
    "AllocComplete": "alloc_complete",
}
LEGACY_LAP_PHASES = {"Alloc", "Build", "Replay"}
V4_PHASES = {
    "OrchestrationReplay",
    "FinalDrain",
    "WinnerBuild",
    "AllocComplete",
}
# schema-v4 已有区间足以在离线侧取补集；这些 phase 之外的
# Atomic、Kernel、RingBp 等是嵌套或 Overlay，不能再从 Submit 扣一次。
V4_EXCLUSIVE_SUBMIT_PHASES = {
    "EfDrain",
    "Materialize",
    "PrepareMap",
    "Claim",
    "Fanin",
    "Register",
    "WinnerBuild",
    "AllocComplete",
}
KERNEL_NAMES = {0: "QK", 1: "SF", 2: "PV", 3: "UP"}
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
}
ATOMIC_OP_NAMES = {
    0: "load",
    1: "exchange",
    2: "fetch_add",
    3: "fetch_max",
}

# schema-v3 的校验表必须与 standalone C++ 的稳定 AtomicSite 编号一致。
# 这里只描述本独立调度器实际实现的 0..14；真实 PA 追加的 BlockWon site
# 不属于本用例，不能为了兼容生产 converter 在这里凭空放宽输入。
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
}
# 这些发布型调用不消费 atomic 返回的旧值；其余 standalone site 的
# 返回值都参与协议判断。v3 输入必须与源码语义完全一致。
ATOMIC_RESULT_UNUSED_SITE_IDS = {0, 3, 6, 7, 13}
# 只有显式 scheduler 等待区中的六类 observation load 可以合并。
# frontier 扫描和 Claim 即使调用很多次也必须继续保留逐调用记录。
POLL_BATCH_SITE_OP_IDS = {1: 0, 2: 0, 5: 0, 11: 0, 12: 0, 14: 0}

ATOMIC_RESULT_USED = 1 << 4
ATOMIC_VALUE_ZERO = 1 << 5
ATOMIC_RETURN_READY = 1 << 6
ATOMIC_POLL_BATCH = 1 << 7
ATOMIC_PAYLOAD_SHIFT = 8
ATOMIC_PAYLOAD_MASK = 0xFFFFFF


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


# 读取 raw JSON，校验十列结构、字段范围与可转整数值，并返回规范化视图。
def _load_and_validate(
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
    # v3 再加入精确计数 PollBatch；v4 追加排他父区间与真实尾动作 span，
    # 并要求 phase-only/atomic 两种观察级别都带 producer summary。
    # 不认识的新版本直接拒绝，避免把新 flags 按旧语义误读。
    trace_schema_version = _integer(metadata.get("trace_schema_version", 1), "metadata.trace_schema_version")
    if trace_schema_version not in (1, 2, 3, 4):
        raise ValueError(f"unsupported metadata.trace_schema_version: {trace_schema_version}")
    if trace_schema_version == 3 and level != 4:
        raise ValueError("metadata.trace_schema_version=3 requires l2_swimlane_level=4")
    if trace_schema_version == 4 and level not in (1, 4):
        raise ValueError(
            "metadata.trace_schema_version=4 requires l2_swimlane_level=1 or 4"
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
        # dropped 无法从已经导出的有效行反推；v3+ 必须由 producer summary
        # 明确承诺为零，下面再逐字段核对。
        "dropped_records": 0,
    }
    v3_clock_rows: dict[int, dict[str, int | bool | None]] = {
        core_id: {"plain": 0, "dependency": 0, "return_ready": None}
        for core_id in range(num_cores)
    }
    v3_result_used_direct_rows: list[tuple[int, int, bool]] = []
    v4_parent_counts: dict[int, dict[str, int]] = {
        core_id: {"OrchestrationReplay": 0, "FinalDrain": 0}
        for core_id in range(num_cores)
    }
    v4_claims: dict[tuple[int, int], tuple[bool, bool, bool]] = {}
    v4_submits: set[tuple[int, int]] = set()
    v4_submit_semantics: dict[tuple[int, int], tuple[bool, bool]] = {}
    v4_tails: dict[tuple[int, int], str] = {}
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
        if trace_schema_version == 4 and phase in LEGACY_LAP_PHASES:
            raise ValueError(
                f"fdwic_events[{index}] schema-v4 forbids legacy lap phase {phase!r}"
            )
        if trace_schema_version == 4 and phase == "DrainWon":
            raise ValueError(
                f"fdwic_events[{index}] schema-v4 forbids unused legacy phase 'DrainWon'"
            )
        if trace_schema_version < 4 and phase in V4_PHASES:
            raise ValueError(
                f"fdwic_events[{index}] phase {phase!r} requires trace_schema_version=4"
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
            if poll_batch:
                if (
                    trace_schema_version not in (3, 4)
                    or level != 4
                    or payload == 0
                    or POLL_BATCH_SITE_OP_IDS.get(auxiliary) != atomic_op
                    or not result_used
                    or value_zero
                    or return_ready
                    or task_id != -1
                    or function_id != -1
                ):
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid Atomic PollBatch "
                        f"site={auxiliary} flags=0x{flags:x}"
                    )
            elif trace_schema_version in (3, 4):
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
        elif phase == "ClockBaseline":
            observed_summary["clock_baseline_records"] += 1
            if trace_schema_version in (3, 4):
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
        if trace_schema_version == 4:
            task_key = (core_id, task_id)
            if phase in ("OrchestrationReplay", "FinalDrain"):
                if task_id != -1 or function_id != -1 or flags != 0 or auxiliary != 0:
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid schema-v4 parent {phase} fields"
                    )
                v4_parent_counts[core_id][phase] += 1
            elif phase == "Submit":
                if task_id < 0 or flags > 1 or auxiliary > 1:
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid schema-v4 Submit fields"
                    )
                if task_key in v4_submits:
                    raise ValueError(
                        f"core {core_id} has duplicate schema-v4 Submit for task {task_id}"
                    )
                v4_submits.add(task_key)
                v4_submit_semantics[task_key] = (bool(flags & 1), bool(auxiliary))
            elif phase == "Claim":
                if task_id < 0 or task_key in v4_claims:
                    raise ValueError(
                        f"core {core_id} has invalid or duplicate schema-v4 Claim for task {task_id}"
                    )
                v4_claims[task_key] = (
                    bool(flags & 0x2), bool(flags & 0x1), bool(auxiliary)
                )
            elif phase in ("WinnerBuild", "AllocComplete"):
                if task_id < 0 or flags != 0 or auxiliary != 0:
                    raise ValueError(
                        f"fdwic_events[{index}] has invalid schema-v4 tail {phase} fields"
                    )
                if task_key in v4_tails:
                    raise ValueError(
                        f"core {core_id} has duplicate schema-v4 tail for task {task_id}"
                    )
                if phase == "WinnerBuild":
                    expected_function = task_id % 5 - 1
                    if task_id % 5 == 0 or function_id != expected_function:
                        raise ValueError(
                            f"fdwic_events[{index}] WinnerBuild function/task mismatch"
                        )
                elif function_id != -1 or task_id % 5 != 0:
                    raise ValueError(
                        f"fdwic_events[{index}] AllocComplete must belong to an Alloc task"
                    )
                v4_tails[task_key] = phase
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
    if trace_schema_version in (3, 4) and level == 4:
        # 每核两条基线同时证明采集完整性和该后端是否真正应用了
        # atomic 返回值依赖；所有消费返回值的直接记录必须与本核证据一致。
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

    if trace_schema_version == 4:
        for core_id, counts in v4_parent_counts.items():
            for phase, count in counts.items():
                if count != 1:
                    raise ValueError(
                        f"core {core_id} requires exactly one schema-v4 {phase}: count={count}"
                    )
        if set(v4_claims) != v4_submits:
            raise ValueError("schema-v4 Claim keys do not match Submit keys")
        for task_key, (attempted, won, is_alloc) in v4_claims.items():
            submit_won, submit_alloc = v4_submit_semantics[task_key]
            expected_alloc = task_key[1] % 5 == 0
            if is_alloc != expected_alloc or submit_alloc != expected_alloc:
                raise ValueError(
                    f"schema-v4 task-kind mismatch at {task_key}: "
                    f"expected_alloc={expected_alloc}"
                )
            if won and not attempted:
                raise ValueError(f"schema-v4 Claim won without attempt at {task_key}")
            if submit_won != won or submit_alloc != is_alloc:
                raise ValueError(f"schema-v4 Submit/Claim semantics mismatch at {task_key}")
            expected_tail = "AllocComplete" if is_alloc else "WinnerBuild"
            actual_tail = v4_tails.get(task_key)
            # 只为 winner 记录真实尾动作；loser 没有尾记录，其剩余时间
            # 由 Submit 的离线补集表示。
            tail_valid = actual_tail == expected_tail if won else actual_tail is None
            if not tail_valid:
                raise ValueError(
                    f"schema-v4 tail mismatch at {task_key}: "
                    f"expected {expected_tail if won else 'no winner tail'}, got {actual_tail}"
                )

    if trace_schema_version >= 3:
        producer_summary = metadata.get("fdwic_summary")
        if not isinstance(producer_summary, dict):
            raise ValueError(
                "metadata.fdwic_summary is required for trace_schema_version>=3"
            )
        for key, observed_value in observed_summary.items():
            producer_value = _integer(
                producer_summary.get(key), f"metadata.fdwic_summary.{key}"
            )
            if producer_value != observed_value:
                raise ValueError(
                    f"metadata.fdwic_summary.{key}={producer_value} "
                    f"does not match raw value {observed_value}"
                )
    return frequency_hz, trace_schema_version, rows, core_by_block_lane, base_cycle, metadata


# 写一个 Chrome Trace Event，并统一处理数组元素间的逗号。
def _emit_event(output: TextIO, event: dict[str, Any], first: bool) -> bool:
    # 逐事件写出，避免再在内存中构造一份体积可达数百 MiB 的 merged 列表。
    if not first:
        output.write(",\n")
    json.dump(event, output, ensure_ascii=False, separators=(",", ":"))
    return False


def _iter_v4_residual_spans(
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
                raise ValueError(f"schema-v4 residual synthesis found duplicate Submit {task_key}")
            submit_by_task[task_key] = row
        elif phase in V4_EXCLUSIVE_SUBMIT_PHASES:
            children_by_task.setdefault(task_key, []).append(row)

    orphan_child_keys = set(children_by_task) - set(submit_by_task)
    if orphan_child_keys:
        raise ValueError(
            "schema-v4 residual synthesis found children without matching Submit: "
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
                raise ValueError(f"schema-v4 Submit spans overlap on core/lane {lane_key}")
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
                    f"schema-v4 {child[5]} child is outside Submit {task_key}"
                )
            if child_start < cursor:
                raise ValueError(
                    f"schema-v4 exclusive children overlap in Submit {task_key}"
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


# 完成一次 raw 到 merged 的转换，成功时返回事件数、block 数和基准 cycle。
def convert(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    (
        frequency_hz,
        trace_schema_version,
        rows,
        core_by_block_lane,
        base_cycle,
        capture_metadata,
    ) = _load_and_validate(input_path)
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
            for row in rows:
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
                if trace_schema_version == 4:
                    # schema-v4 的 merged 只承担可视化：阶段、atomic site/op/
                    # boundary、task 和 poll 次数均已编码在 name，轨道与时间由
                    # pid/tid/ts/dur 给出。十列权威字段完整保留在同目录 raw，
                    # 不再逐事件复制近 100 MiB 的 args/cat。
                    event.pop("args", None)
                    event.pop("cat", None)
                first = _emit_event(output, event, first)
                emitted += 1
            if trace_schema_version == 4:
                # 补集是纯离线合成件：不增加设备 trace record、SYS_CNT
                # 或 raw 字段。事件只保留 Perfetto X 必需的 6 个字段，
                # 不复制每条 raw 已有的 args，控制 merged 体积增量。
                for _core_id, block_id, lane, start, end, name in _iter_v4_residual_spans(rows):
                    first = _emit_event(
                        output,
                        {
                            "ph": "X",
                            "name": name,
                            "pid": block_id,
                            "tid": lane,
                            "ts": round((start - base_cycle) * factor, 3),
                            "dur": round((end - start) * factor, 3),
                        },
                        first,
                    )
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
