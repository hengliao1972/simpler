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
}
KERNEL_NAMES = {0: "QK", 1: "SF", 2: "PV", 3: "UP"}
# 一个物理 mixed block 的三条 runtime lane：AIC、AIV0、AIV1。
LANE_NAMES = {0: "AIC", 1: "AIV0", 2: "AIV1"}


# 把可转为整数的 raw 标量归一为 int，并在错误中保留精确字段路径。
def _integer(value: Any, label: str) -> int:
    # 这是兼容 JSON 数值/数值字符串的宽松归一，不负责强制原始 JSON 类型必须为 int。
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not an integer: {value!r}") from error


# 读取 raw JSON，校验十列结构、字段范围与可转整数值，并返回规范化视图。
def _load_and_validate(input_path: Path) -> tuple[int, list[tuple[Any, ...]], dict[tuple[int, int], int], int]:
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
    num_cores = _integer(metadata.get("num_cores"), "metadata.num_cores")
    if num_cores <= 0:
        raise ValueError("metadata.num_cores must be positive")
    core_types = metadata.get("core_types")
    if not isinstance(core_types, list) or len(core_types) != num_cores:
        raise ValueError("metadata.core_types length must equal metadata.num_cores")

    rows = data.get("fdwic_events")
    if not isinstance(rows, list) or not rows:
        raise ValueError("fdwic_events must be a non-empty array")

    # Perfetto metadata 需要从 (block, lane) 找回稳定的 core 编号；同一 lane
    # 若在 raw 中映射到两个 core，说明采集已损坏，不能继续生成误导性泳道。
    core_by_block_lane: dict[tuple[int, int], int] = {}
    base_cycle: int | None = None
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
    return frequency_hz, rows, core_by_block_lane, base_cycle


# 写一个 Chrome Trace Event，并统一处理数组元素间的逗号。
def _emit_event(output: TextIO, event: dict[str, Any], first: bool) -> bool:
    # 逐事件写出，避免再在内存中构造一份体积可达数百 MiB 的 merged 列表。
    if not first:
        output.write(",\n")
    json.dump(event, output, ensure_ascii=False, separators=(",", ":"))
    return False


# 完成一次 raw 到 merged 的转换，成功时返回事件数、block 数和基准 cycle。
def convert(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    frequency_hz, rows, core_by_block_lane, base_cycle = _load_and_validate(input_path)
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
    first = True
    emitted = 0
    # 临时文件的整个生命周期都在 try 内；包括 Ctrl-C 在内的异常都会先清理
    # .tmp 再向上传播。格式/IO 错误由 main 简短报告，Ctrl-C 保留默认中断行为。
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            output.write('{"displayTimeUnit":"ns","traceEvents":[\n')
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
                # Kernel 和 Commit 放到 lane+3 的 kernel 子泳道；其他阶段留在
                # lane 0..2 的 runtime 泳道。这与真实 PA merged 文件的布局一致。
                if phase == "kernel" and function_id >= 0:
                    name = f"{KERNEL_NAMES.get(function_id, f'f{function_id}')}#{task_id}"
                    thread_id = lane + 3
                elif phase == "commit":
                    name = f"commit#{task_id}"
                    thread_id = lane + 3
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
