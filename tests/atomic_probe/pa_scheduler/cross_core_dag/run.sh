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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# converter/analyzer 属于 pa_scheduler 公共工具；本目录只拥有 DAG 实现。
PA_SCHEDULER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TENSORMAP_MODE="shared"

usage() {
    cat <<'EOF'
Usage:
  ./run.sh build            cpu|ccec
  ./run.sh run              cpu|ccec [benchmark options]
  ./run.sh smoke            cpu|ccec [--device N]
  ./run.sh swimlane         cpu|ccec [benchmark options]
  ./run.sh build-perf-clock cpu|ccec
  ./run.sh perf-clock       cpu|ccec [benchmark options]

本目录固定为 Scalar cross-core DAG shared TensorMap PA：
  - 不接受 private/shared 模式参数；
  - 不构建 AscendC；
  - 首阶段只保留 swimlane 与无泳道端到端构建，不提供 submit-PMU；
  - perf-clock 是现有命令名，唯一结果覆盖 startup 起点到 FinalDrain 结束。

常用 benchmark 参数：
  --device N
  --batches 1..512
  --runs N
  --nop-count N
  --nop-counts QK,SF,PV,UP
  --winner-workload scalar-nop|real-compute
  --real-compute-count N
  --real-compute-counts QK,SF,PV,UP
  --real-compute-pattern constant|layout-diagnostic
  --shared-context-lens C0[,C1...]

Scalar cross-core DAG 由 execution drain 直接收口，不再使用 final-barrier 选项。

PA_SHARED_INSERT_TURN_GROUPS 可选择已有 shared 插入完成链身份：
  1|2|4|8|16|32|64|128（默认 1）。构建与运行必须使用同一值，
  CCEC manifest 会拒绝交叉消费不同身份的产物。
EOF
}

validate_backend() {
    case "$1" in
        cpu|ccec) ;;
        *)
            echo "Unknown backend: $1 (expected cpu|ccec)" >&2
            exit 1
            ;;
    esac
}

validate_groups() {
    local groups="${PA_SHARED_INSERT_TURN_GROUPS:-1}"
    case "$groups" in
        1|2|4|8|16|32|64|128) ;;
        *)
            echo "PA_SHARED_INSERT_TURN_GROUPS must be a power of two from 1 through 128." >&2
            exit 1
            ;;
    esac
}

cpu_executable_path() {
    local variant="$1"
    local groups="${PA_SHARED_INSERT_TURN_GROUPS:-1}"
    local binary="pa_scheduler_cpu"
    if [[ "$groups" != "1" ]]; then
        binary="pa_scheduler_cpu_turn_g${groups}"
    fi
    printf '%s/build/cpu/%s/%s/%s\n' \
        "$SCRIPT_DIR" "$TENSORMAP_MODE" "$variant" "$binary"
}

build_backend() {
    local backend="$1"
    local variant="$2"
    "$SCRIPT_DIR/$backend/build.sh" "$variant"
}

ccec_artifact_failure() {
    local variant="$1"
    local reason="$2"
    echo "Invalid CCEC Scalar cross-core DAG artifact set ($variant): $reason" >&2
    if [[ "$variant" == "perf-clock" ]]; then
        echo "Run: $0 build-perf-clock ccec" >&2
    else
        echo "Run: $0 build ccec" >&2
    fi
    return 1
}

validate_ccec_artifacts() {
    local variant="$1"
    local build_dir="$SCRIPT_DIR/build/ccec/$TENSORMAP_MODE/$variant"
    local manifest_name="pa_scheduler_artifacts.manifest"
    local manifest="$build_dir/$manifest_name"
    local artifacts=(pa_scheduler_host pa_scheduler_kernel.o)
    local expected_groups="${PA_SHARED_INSERT_TURN_GROUPS:-1}"
    local expected_generic_bytes=32
    local expected_stride=1048576
    if [[ "$variant" == "swimlane" ]]; then
        expected_generic_bytes=16
        expected_stride=593920
    elif [[ "$variant" != "perf-clock" ]]; then
        ccec_artifact_failure "$variant" "unsupported variant"
        return 1
    fi

    if [[ ! -x "$build_dir/pa_scheduler_host" ||
          ! -s "$build_dir/pa_scheduler_kernel.o" || ! -s "$manifest" ]]; then
        ccec_artifact_failure "$variant" "host, kernel, or ready manifest is missing"
        return 1
    fi
    if ! command -v sha256sum >/dev/null 2>&1; then
        ccec_artifact_failure "$variant" "sha256sum is unavailable"
        return 1
    fi

    local lines=()
    mapfile -t lines < "$manifest"
    if [[ ${#lines[@]} -ne 14 ||
          "${lines[0]:-}" != "# schema=pa_scheduler_artifacts/v4" ||
          "${lines[1]:-}" != "# tensormap_mode=shared" ||
          "${lines[2]:-}" != "# tensormap_mode_id=1" ||
          "${lines[3]:-}" != "# tensormap_ring_cap=128" ||
          "${lines[4]:-}" != "# shared_insert_turn_groups=$expected_groups" ||
          "${lines[5]:-}" != "# generic_record_bytes=$expected_generic_bytes" ||
          "${lines[6]:-}" != "# submit_claim_record_bytes=32" ||
          "${lines[7]:-}" != "# records_per_core=28416" ||
          "${lines[8]:-}" != "# worker_stride_bytes=$expected_stride" ||
          "${lines[9]:-}" != "# variant=$variant" ||
          "${lines[10]:-}" != "# phase=none" ||
          "${lines[11]:-}" != "# phase_id=0" ]]; then
        ccec_artifact_failure "$variant" "manifest identity or trace layout does not match"
        return 1
    fi

    local index digest filename extra
    for index in "${!artifacts[@]}"; do
        read -r digest filename extra <<< "${lines[index + 12]}"
        if [[ ! "$digest" =~ ^[[:xdigit:]]{64}$ ||
              "$filename" != "${artifacts[index]}" || -n "${extra:-}" ]]; then
            ccec_artifact_failure "$variant" "manifest checksum entry is malformed"
            return 1
        fi
    done
    if ! (cd "$build_dir" && sha256sum --check --strict --status "$manifest_name"); then
        ccec_artifact_failure "$variant" "artifact SHA256 values do not match"
        return 1
    fi
    echo "[CHECK] CCEC manifest verified: $manifest"
}

run_backend() {
    local backend="$1"
    local variant="$2"
    shift 2
    case "$backend" in
        cpu)
            local executable
            executable="$(cpu_executable_path "$variant")"
            if [[ ! -x "$executable" ]]; then
                echo "Missing CPU artifact: $executable" >&2
                if [[ "$variant" == "perf-clock" ]]; then
                    echo "Run: $0 build-perf-clock cpu" >&2
                else
                    echo "Run: $0 build cpu" >&2
                fi
                return 1
            fi
            "$executable" "$@"
            ;;
        ccec)
            local build_dir="$SCRIPT_DIR/build/ccec/$TENSORMAP_MODE/$variant"
            validate_ccec_artifacts "$variant"
            "$build_dir/pa_scheduler_host" \
                --kernel "$build_dir/pa_scheduler_kernel.o" "$@"
            ;;
    esac
}

reject_fixed_options() {
    local action="$1"
    shift
    local argument
    for argument in "$@"; do
        case "$argument" in
            --kernel|--kernel=*|--runs|--runs=*|--swimlane-json|--swimlane-json=*|\
            --no-swimlane|--profile-phases|--trace-atomics|--analyze-swimlane|\
            --pmu-window|--pmu-window=*|--pmu-scalar-nops|--pmu-scalar-nops=*|\
            --pmu-icache-trials|--pmu-icache-trials=*|--pmu-json|--pmu-json=*)
                echo "The $action action manages or forbids $argument." >&2
                exit 1
                ;;
        esac
    done
}

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 1
fi

ACTION="$1"
BACKEND="$2"
shift 2
validate_backend "$BACKEND"
validate_groups

# 首阶段不提供 submit-PMU。即使复制来的 host 仍保留底层诊断解析能力，
# 统一入口也必须在任何构建、运行或文件创建前拒绝这些参数。
for argument in "$@"; do
    case "$argument" in
        --pmu-window|--pmu-window=*|--pmu-scalar-nops|--pmu-scalar-nops=*|\
        --pmu-icache-trials|--pmu-icache-trials=*|--pmu-json|--pmu-json=*)
            echo "submit-PMU is not available in the Scalar cross-core DAG stage: $argument" >&2
            exit 1
            ;;
    esac
done

case "$ACTION" in
    build)
        if [[ $# -ne 0 ]]; then
            echo "The build action does not accept benchmark options." >&2
            exit 1
        fi
        build_backend "$BACKEND" swimlane
        ;;
    run)
        run_backend "$BACKEND" swimlane "$@"
        ;;
    smoke)
        for argument in "$@"; do
            case "$argument" in
                --batches|--batches=*|--runs|--runs=*|--nop-count|--nop-count=*|\
                --nop-counts|--nop-counts=*|--winner-workload|--winner-workload=*|\
                --real-compute-count|--real-compute-count=*|--real-compute-counts|\
                --real-compute-counts=*|--real-compute-pattern|--real-compute-pattern=*)
                    echo "The smoke action fixes b1/r1/scalar-nop=0." >&2
                    exit 1
                    ;;
            esac
        done
        run_backend "$BACKEND" swimlane --batches 1 --runs 1 \
            --winner-workload scalar-nop --nop-count 0 "$@"
        ;;
    swimlane)
        reject_fixed_options swimlane "$@"
        PYTHON_BIN="${PYTHON:-python3}"
        if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
            echo "Python executable not found: $PYTHON_BIN" >&2
            exit 1
        fi
        CONVERTER="$PA_SCHEDULER_DIR/swimlane_converter.py"
        ANALYZER="$PA_SCHEDULER_DIR/swimlane_exclusive_analyzer.py"
        if [[ ! -f "$CONVERTER" || ! -f "$ANALYZER" ]]; then
            echo "Missing pa_scheduler swimlane converter or analyzer." >&2
            exit 1
        fi
        OUTPUT_ROOT="$PA_SCHEDULER_DIR/outputs/pa_scheduler_cross_core_dag_swimlane_$(date -u +%Y%m%d_%H%M%S)_$$"
        BACKEND_OUTPUT="$OUTPUT_ROOT/$BACKEND"
        RAW_JSON="$BACKEND_OUTPUT/l2_swimlane_records.json"
        MERGED_JSON="$BACKEND_OUTPUT/merged_swimlane.json"
        EXCLUSIVE_JSON="$BACKEND_OUTPUT/swimlane_exclusive_analysis.json"
        mkdir -p "$BACKEND_OUTPUT"
        run_backend "$BACKEND" swimlane \
            --runs 1 --trace-atomics --swimlane-json "$RAW_JSON" "$@"
        "$PYTHON_BIN" "$CONVERTER" "$RAW_JSON" -o "$MERGED_JSON"
        "$PYTHON_BIN" "$ANALYZER" "$RAW_JSON" -o "$EXCLUSIVE_JSON"
        echo "[SWIMLANE] output_root=$OUTPUT_ROOT"
        ;;
    build-perf-clock)
        if [[ $# -ne 0 ]]; then
            echo "The build-perf-clock action does not accept benchmark options." >&2
            exit 1
        fi
        build_backend "$BACKEND" perf-clock
        ;;
    perf-clock)
        reject_fixed_options perf-clock "$@"
        run_backend "$BACKEND" perf-clock --runs 1 --no-swimlane "$@"
        ;;
    *)
        echo "Unknown action: $ACTION" >&2
        usage >&2
        exit 1
        ;;
esac
