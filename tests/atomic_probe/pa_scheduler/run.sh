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
  ./run.sh build-submit-pmu ccec none|claim|efdrain|materialize|register
  ./run.sh submit-pmu ccec none|claim|efdrain|materialize|register [benchmark options]

Benchmark options:
  --device N
  --batches 1..256
  --runs N
  --nop-count N
  --nop-counts QK,SF,PV,UP
  --profile-phases
  --analyze-swimlane
  --trace-atomics
  --swimlane-json FILE
  --no-swimlane

CCEC-only PMU probe options (selectors are owned by the standalone Main AICPU helper):
  --pmu-window off|empty|scalar|scalar-double|icache-single|submit-all
  --pmu-scalar-nops N
  --pmu-icache-trials N
  --pmu-json FILE

Standalone winner workload options (CCEC, AscendC, and CPU):
  --winner-workload scalar-nop|real-compute
  --real-compute-count N
  --real-compute-counts QK,SF,PV,UP
  --real-compute-pattern constant|layout-diagnostic

real-compute is the default. Its CCEC-calibrated A5 counts are QK/SF/PV/UP=6/28/4/1;
one count is one complete 128x128 load/engine/store/completion-wait pipeline
per winner task, not a scalar NOP count. Explicit count options override those
four defaults. AscendC must be calibrated independently; CPU only preserves
the arithmetic and routing semantics and is not an A5 timing reference.
The constant pattern is the performance default. layout-diagnostic uses a
weighted diagonal A and asymmetric B to detect transpose/stride/reorder bugs.

The swimlane action enables atomic tracing by default. For the lower-level run
action, --trace-atomics still requires swimlane tracing; add
--analyze-swimlane to print the per-role/per-site timing distributions.
--pmu-json requires --runs 1 and a non-off PMU window. For submit-pmu, a
successful raw capture also generates a self-contained HTML report beside it;
submit_icache_raw.json maps to submit_icache_report.html. PMU probe options are
CCEC-only and cannot target all.

The submit-pmu action is a separate CCEC-only build. It fixes one PMU-only run
covering the complete Submit window. phase=none performs no internal snapshots;
phase=claim/efdrain/materialize/register reports running read-clear lower/loss-adjusted upper bounds
for one compile-time phase while CNT6/7 retain the authoritative whole-window counters.

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
            if [[ "$1" == "ccec" ]]; then
                "$SCRIPT_DIR/ccec/build.sh" swimlane
            else
                "$SCRIPT_DIR/$1/build.sh"
            fi
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

validate_submit_pmu_phase() {
    case "$1" in
        none|claim|efdrain|materialize|register) ;;
        *)
            echo "Unknown submit-pmu phase: $1 (expected none|claim|efdrain|materialize|register)" >&2
            exit 1
            ;;
    esac
}

submit_pmu_artifact_failure() {
    local phase="$1"
    local reason="$2"
    echo "Invalid submit-pmu artifact set for phase '$phase': $reason" >&2
    echo "Run: $0 build-submit-pmu ccec $phase" >&2
    return 1
}

validate_submit_pmu_artifacts() {
    local phase="$1"
    local build_dir="$2"
    local phase_id
    case "$phase" in
        none) phase_id=0 ;;
        claim) phase_id=1 ;;
        efdrain) phase_id=2 ;;
        materialize) phase_id=4 ;;
        register) phase_id=5 ;;
        *) submit_pmu_artifact_failure "$phase" "unsupported phase"; return 1 ;;
    esac

    local manifest_name="submit_pmu_artifacts.manifest"
    local manifest="$build_dir/$manifest_name"
    local artifacts=(
        pa_scheduler_host
        pa_scheduler_kernel.o
        libpa_scheduler_pmu_owner_aicpu.so
        libpa_scheduler_pmu_owner_dispatcher.so
    )
    if [[ ! -x "$build_dir/${artifacts[0]}" ]]; then
        submit_pmu_artifact_failure "$phase" "host runner is missing, empty, or not executable"
        return 1
    fi
    local artifact
    for artifact in "${artifacts[@]:1}"; do
        if [[ ! -s "$build_dir/$artifact" ]]; then
            submit_pmu_artifact_failure "$phase" "artifact is missing or empty: $artifact"
            return 1
        fi
    done
    if [[ ! -s "$manifest" ]]; then
        submit_pmu_artifact_failure "$phase" "ready manifest is missing or empty"
        return 1
    fi
    if ! command -v sha256sum >/dev/null 2>&1; then
        submit_pmu_artifact_failure "$phase" "sha256sum is unavailable"
        return 1
    fi

    # manifest 固定为四行身份头和四行校验和；既检查 phase/variant，也拒绝
    # 漏项、增项、绝对路径或重复文件，避免 sha256sum 只校验到一个子集。
    local manifest_lines=()
    mapfile -t manifest_lines < "$manifest"
    if [[ ${#manifest_lines[@]} -ne 8 ||
          "${manifest_lines[0]}" != "# schema=pa_scheduler_submit_pmu_artifacts/v1" ||
          "${manifest_lines[1]}" != "# variant=submit-pmu" ||
          "${manifest_lines[2]}" != "# phase=$phase" ||
          "${manifest_lines[3]}" != "# phase_id=$phase_id" ]]; then
        submit_pmu_artifact_failure "$phase" "manifest schema, variant, or phase metadata does not match"
        return 1
    fi
    local index digest filename extra
    for index in "${!artifacts[@]}"; do
        digest=""
        filename=""
        extra=""
        read -r digest filename extra <<< "${manifest_lines[index + 4]}"
        if [[ ! "$digest" =~ ^[[:xdigit:]]{64}$ ||
              "$filename" != "${artifacts[index]}" || -n "$extra" ]]; then
            submit_pmu_artifact_failure "$phase" "manifest checksum entry $((index + 1)) is malformed or out of order"
            return 1
        fi
    done
    if ! (cd "$build_dir" && sha256sum --check --strict --status "$manifest_name"); then
        submit_pmu_artifact_failure "$phase" "one or more artifact SHA256 values do not match"
        return 1
    fi
    echo "[CHECK] submit-pmu artifact manifest verified: $manifest"
}

reject_managed_submit_pmu_options() {
    # 这些参数定义诊断 ELF 与窗口边界，必须由 action 独占；允许用户只传
    # device/batches/workload 和可选的 --pmu-json。
    for argument in "$@"; do
        case "$argument" in
            --kernel|--kernel=*|--runs|--runs=*|--pmu-window|--pmu-window=*|\
            --no-swimlane|--profile-phases|--trace-atomics|--analyze-swimlane|\
            --swimlane-json|--swimlane-json=*|--pmu-scalar-nops|--pmu-scalar-nops=*|\
            --pmu-icache-trials|--pmu-icache-trials=*)
                echo "The submit-pmu action manages or forbids $argument." >&2
                exit 1
                ;;
        esac
    done
}

run_submit_pmu() {
    local phase="$1"
    shift
    local build_dir="$SCRIPT_DIR/build/ccec/submit-pmu/$phase"
    local host="$build_dir/pa_scheduler_host"
    local kernel="$build_dir/pa_scheduler_kernel.o"
    local pmu_json=""
    local expect_pmu_json_value=false
    local argument
    # host 仍是 raw 文件的唯一写入者；这里只提取同一个路径，在 host 成功且
    # raw 已原子发布后调用独立分析器生成可视 HTML。原参数保持原样透传。
    for argument in "$@"; do
        if [[ "$expect_pmu_json_value" == true ]]; then
            pmu_json="$argument"
            expect_pmu_json_value=false
            continue
        fi
        case "$argument" in
            --pmu-json) expect_pmu_json_value=true ;;
            --pmu-json=*) pmu_json="${argument#--pmu-json=}" ;;
        esac
    done
    validate_submit_pmu_artifacts "$phase" "$build_dir"
    "$host" --kernel "$kernel" --runs 1 --no-swimlane --pmu-window submit-all "$@"
    if [[ -n "$pmu_json" ]]; then
        local python_bin="${PYTHON:-python3}"
        if ! command -v "$python_bin" >/dev/null 2>&1; then
            echo "Python executable not found for PMU HTML report: $python_bin" >&2
            return 1
        fi
        if [[ ! -f "$SCRIPT_DIR/pmu_html_report.py" ]]; then
            echo "Missing local PMU HTML report generator: $SCRIPT_DIR/pmu_html_report.py" >&2
            return 1
        fi
        "$python_bin" "$SCRIPT_DIR/pmu_html_report.py" "$pmu_json"
    fi
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

reject_ccec_pmu_options_for_non_ccec() {
    local backend="$1"
    shift
    if [[ "$backend" == "ccec" ]]; then
        return
    fi

    # PMU selector、校准 NOP 和导出路径只由 CCEC 分支消费；winner workload
    # 已由三个后端共同解析。这里在展开 all 前拒绝 PMU，避免先启动 CCEC、
    # 再由其他后端迟到报错。
    for argument in "$@"; do
        case "$argument" in
            --pmu-window|--pmu-window=*|--pmu-scalar-nops|--pmu-scalar-nops=*|\
            --pmu-icache-trials|--pmu-icache-trials=*|--pmu-json|--pmu-json=*)
                echo "CCEC-only option $argument is not supported by backend '$backend'." >&2
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

# 后端约束必须早于 build/run/smoke/swimlane 的任何文件创建、构建或设备动作。
reject_ccec_pmu_options_for_non_ccec "$BACKEND" "$@"

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
        # smoke 仍启动全部 96 个 worker，并显式注入 1 batch、1 run、scalar-nop=0；
        # 后置用户参数仍由共享 parser 处理。它用于快速检查原子协议、拓扑和
        # 最终状态，不作为性能数据。
        for argument in "$@"; do
            case "$argument" in
                --batches|--batches=*|--runs|--runs=*|--nop-count|--nop-count=*|\
                --nop-counts|--nop-counts=*|--winner-workload|--winner-workload=*|\
                --real-compute-count|--real-compute-count=*|--real-compute-counts|\
                --real-compute-counts=*|--real-compute-pattern|--real-compute-pattern=*)
                    echo "The smoke action fixes b1/r1/scalar-nop=0; use the run action for real-compute." >&2
                    exit 1
                    ;;
            esac
        done
        for backend in "${BACKENDS[@]}"; do
            run_backend "$backend" --batches 1 --runs 1 \
                --winner-workload scalar-nop --nop-count 0 "$@"
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
            # 用户要求泳道默认带齐逐 atomic 性能。重复传入 --trace-atomics
            # 只是幂等布尔开关，不会产生两份记录；直接 run 仍可选择 phase-only。
            run_backend "$backend" --runs 1 --trace-atomics --swimlane-json "$RAW_JSON" "$@"
            "$PYTHON_BIN" "$SCRIPT_DIR/swimlane_converter.py" "$RAW_JSON" -o "$MERGED_JSON"
        done
        echo "[SWIMLANE] output_root=$OUTPUT_ROOT"
        ;;
    build-submit-pmu)
        if [[ "$BACKEND" != "ccec" || $# -ne 1 ]]; then
            echo "Usage: $0 build-submit-pmu ccec none|claim|efdrain|materialize|register" >&2
            exit 1
        fi
        PHASE="$1"
        validate_submit_pmu_phase "$PHASE"
        "$SCRIPT_DIR/ccec/build.sh" submit-pmu "$PHASE"
        ;;
    submit-pmu)
        if [[ "$BACKEND" != "ccec" || $# -lt 1 ]]; then
            echo "Usage: $0 submit-pmu ccec none|claim|efdrain|materialize|register [benchmark options]" >&2
            exit 1
        fi
        PHASE="$1"
        shift
        validate_submit_pmu_phase "$PHASE"
        reject_managed_submit_pmu_options "$@"
        run_submit_pmu "$PHASE" "$@"
        ;;
    *)
        # 未知 action 不尝试推断用户意图，也不会触发任何构建或设备操作。
        echo "Unknown action: $ACTION" >&2
        usage >&2
        exit 1
        ;;
esac
