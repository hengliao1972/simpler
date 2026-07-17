#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

# 用脚本自身位置锚定所有构建产物、转换器和输出目录；从任意 cwd 调用都
# 不会回退到 simpler 仓库中的同名工具。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  ./run.sh build  ccec|ascendc|cpu|all
  ./run.sh run    ccec|ascendc|cpu|all [benchmark options]
  ./run.sh smoke  ccec|ascendc|cpu|all [--device N]
  ./run.sh swimlane ccec|ascendc|cpu|all [benchmark options]

Benchmark options:
  --device N
  --batches 1..256
  --runs N
  --nop-count N
  --nop-counts QK,SF,PV,UP
  --profile-phases
  --analyze-swimlane
  --swimlane-json FILE
  --no-swimlane

The swimlane action performs exactly one run and writes both the raw capture
and merged Perfetto JSON below this directory's outputs/ folder. It rejects
--runs, --swimlane-json, and --no-swimlane because those are managed by the action.

The all target always uses the requested implementation order:
CCEC, AscendC, then CPU.
EOF
}

require_file() {
    # run/smoke/swimlane 都只消费本目录 build/ 下已经生成的后端产物，
    # 缺失时明确提示对应 build action，而不是临时猜测编译命令。
    if [[ ! -f "$1" ]]; then
        echo "Missing build artifact: $1" >&2
        echo "Run: $0 build $2" >&2
        exit 1
    fi
}

build_backend() {
    # 后端自己的 build.sh 是唯一构建入口；all 的先后顺序由下方 BACKENDS
    # 固定为 CCEC、AscendC、CPU，便于按用户要求分阶段复现。
    case "$1" in
        ccec|ascendc|cpu)
            "$SCRIPT_DIR/$1/build.sh"
            ;;
        *)
            echo "Unknown backend: $1" >&2
            exit 1
            ;;
    esac
}

run_backend() {
    local backend="$1"
    shift
    # 所有 benchmark 参数原样透传给同一套 host option parser。CCEC 额外
    # 传入本目录内的 mixed ELF，其余两个后端直接启动本地可执行文件。
    case "$backend" in
        ccec)
            local host="$SCRIPT_DIR/build/ccec/pa_scheduler_host"
            local kernel="$SCRIPT_DIR/build/ccec/pa_scheduler_kernel.o"
            require_file "$host" ccec
            require_file "$kernel" ccec
            "$host" --kernel "$kernel" "$@"
            ;;
        ascendc)
            local executable="$SCRIPT_DIR/build/ascendc/pa_scheduler_ascendc"
            require_file "$executable" ascendc
            "$executable" "$@"
            ;;
        cpu)
            local executable="$SCRIPT_DIR/build/cpu/pa_scheduler_cpu"
            require_file "$executable" cpu
            "$executable" "$@"
            ;;
        *)
            echo "Unknown backend: $backend" >&2
            exit 1
            ;;
    esac
}

reject_managed_swimlane_options() {
    # swimlane action 必须独占轮数、raw 路径和 trace 开关，才能保证每个
    # backend 恰好对应一对 raw/merged 文件且不会发生多轮覆盖。
    for argument in "$@"; do
        case "$argument" in
            --runs|--swimlane-json|--no-swimlane)
                echo "The swimlane action manages $argument; do not pass it explicitly." >&2
                exit 1
                ;;
        esac
    done
}

# 顶层先解释 action/backend；run 的其余参数交给共享 parser，build、smoke 和
# swimlane 再分别处理自己的约束或默认注入项。参数不足会在创建目录前失败。
if [[ $# -lt 2 ]]; then
    usage >&2
    exit 1
fi

ACTION="$1"
BACKEND="$2"
shift 2

if [[ "$BACKEND" == "all" ]]; then
    # 该顺序也是组合构建、运行和泳道采集的稳定对外约定。
    BACKENDS=(ccec ascendc cpu)
else
    BACKENDS=("$BACKEND")
fi

case "$ACTION" in
    build)
        # build 只选择后端，不接收 benchmark 参数；这样编译配置不会被运行时
        # 选项暗中改变，三种实现的构建命令也保持可独立复现。
        if [[ $# -ne 0 ]]; then
            echo "The build action does not accept benchmark options." >&2
            exit 1
        fi
        for backend in "${BACKENDS[@]}"; do
            build_backend "$backend"
        done
        ;;
    run)
        # run 不替用户补默认覆盖项，完整参数校验交给各后端共享的 Options parser。
        for backend in "${BACKENDS[@]}"; do
            run_backend "$backend" "$@"
        done
        ;;
    smoke)
        # smoke 仍启动全部 96 个 worker，并默认注入 1 batch、1 run、0 NOP；
        # 后置用户参数仍由共享 parser 处理。它用于快速检查原子协议、拓扑和
        # 最终状态，不作为性能数据。
        for backend in "${BACKENDS[@]}"; do
            run_backend "$backend" --batches 1 --runs 1 --nop-count 0 "$@"
        done
        ;;
    swimlane)
        # swimlane 是“采集 + 转换”的事务边界：runner 失败则不转换，converter
        # 失败则 action 非零退出；已成功写完的 raw 会保留用于排查转换问题。
        reject_managed_swimlane_options "$@"
        # 仅需要 Python 标准库；允许用户用 PYTHON 指向自己的虚拟环境，
        # 但转换脚本始终取自当前 pa_scheduler 目录。
        PYTHON_BIN="${PYTHON:-python3}"
        if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
            echo "Python executable not found: $PYTHON_BIN" >&2
            exit 1
        fi
        if [[ ! -f "$SCRIPT_DIR/swimlane_converter.py" ]]; then
            echo "Missing local converter: $SCRIPT_DIR/swimlane_converter.py" >&2
            exit 1
        fi
        # UTC 秒级时间加当前 shell PID 避免并行采集目录冲突；所有产物保持
        # 在本目录 outputs/ 下，复制 pa_scheduler 后仍可原样工作。
        OUTPUT_ROOT="$SCRIPT_DIR/outputs/pa_scheduler_swimlane_$(date -u +%Y%m%d_%H%M%S)_$$"
        mkdir -p "$OUTPUT_ROOT"
        # all 模式下每个 backend 使用独立子目录，避免同名 raw/merged 互相覆盖；
        # 某一后端失败后 set -e 停止，之前已完成后端的产物仍可单独检查。
        for backend in "${BACKENDS[@]}"; do
            BACKEND_OUTPUT="$OUTPUT_ROOT/$backend"
            RAW_JSON="$BACKEND_OUTPUT/l2_swimlane_records.json"
            MERGED_JSON="$BACKEND_OUTPUT/merged_swimlane.json"
            mkdir -p "$BACKEND_OUTPUT"
            # runner 先执行单轮严格语义校验并流式写 raw；成功后才调用本地
            # converter 生成 Perfetto 文件。set -e 保证任一步失败即停止。
            run_backend "$backend" --runs 1 --swimlane-json "$RAW_JSON" "$@"
            "$PYTHON_BIN" "$SCRIPT_DIR/swimlane_converter.py" "$RAW_JSON" -o "$MERGED_JSON"
        done
        echo "[SWIMLANE] output_root=$OUTPUT_ROOT"
        ;;
    *)
        # 未知 action 不尝试推断用户意图，也不会触发任何构建或设备操作。
        echo "Unknown action: $ACTION" >&2
        usage >&2
        exit 1
        ;;
esac
