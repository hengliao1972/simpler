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
PA_SCHEDULER_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
SCALAR_ROOT="$(cd "$SCRIPT_DIR/../scalar_build" && pwd)"
TENSORMAP_MODE="shared"

usage() {
    cat <<'EOF'
Usage:
  ./run.sh build            ccec
  ./run.sh run              ccec [benchmark options]
  ./run.sh smoke            ccec [--device N]
  ./run.sh swimlane         ccec [benchmark options]
  ./run.sh build-perf-clock ccec
  ./run.sh verify           ccec [swimlane|perf-clock]
  ./run.sh perf-clock       ccec [benchmark options]

本目录固定为 AICPU closed Plan + ordinary TensorMap + SIMT Build：
  - AICPU producer 只发布 canonical Plan，不发布 Host task identity；
  - AIV0 启动 128-thread VF，四个 warp leader 动态领取 Build ticket；
  - VF join 后，32 AIC + 64 AIV Scalar 共同 Execute/FinalDrain；
  - 当前只提供 CCEC 正式模式；CPU 测试是 build.sh 的可行性门槛。

PA_SHARED_INSERT_TURN_GROUPS 可选 1|2|4|8|16|32|64|128（默认 1）。
EOF
}

validate_backend() {
    if [[ "$1" != "ccec" ]]; then
        echo "ordinary SIMT formal mode supports only ccec, got: $1" >&2
        exit 1
    fi
}

validate_groups() {
    case "${PA_SHARED_INSERT_TURN_GROUPS:-1}" in
        1|2|4|8|16|32|64|128) ;;
        *)
            echo "PA_SHARED_INSERT_TURN_GROUPS must be a power of two from 1 through 128." >&2
            exit 1
            ;;
    esac
}

artifact_failure() {
    local variant="$1"
    local reason="$2"
    echo "Invalid ordinary SIMT CCEC artifact set ($variant): $reason" >&2
    if [[ "$variant" == "perf-clock" ]]; then
        echo "Run: $0 build-perf-clock ccec" >&2
    else
        echo "Run: $0 build ccec" >&2
    fi
    return 1
}

validate_artifacts() {
    local variant="$1"
    local build_dir="$SCRIPT_DIR/build/ccec/$TENSORMAP_MODE/$variant"
    local manifest_name="pa_scheduler_artifacts.manifest"
    local manifest="$build_dir/$manifest_name"
    local artifacts=(
        pa_scheduler_host
        pa_scheduler_kernel.o
        libpa_scheduler_plan_aicpu.so
        libpa_scheduler_plan_dispatcher.so
    )
    local expected_groups="${PA_SHARED_INSERT_TURN_GROUPS:-1}"
    local expected_generic_bytes=32
    local expected_stride=1048576
    if [[ "$variant" == "swimlane" ]]; then
        expected_generic_bytes=16
        expected_stride=593920
    elif [[ "$variant" != "perf-clock" ]]; then
        artifact_failure "$variant" "unsupported variant"
        return 1
    fi

    if [[ ! -x "$build_dir/pa_scheduler_host" ||
          ! -s "$build_dir/pa_scheduler_kernel.o" ||
          ! -s "$build_dir/libpa_scheduler_plan_aicpu.so" ||
          ! -s "$build_dir/libpa_scheduler_plan_dispatcher.so" ||
          ! -s "$manifest" ]]; then
        artifact_failure "$variant" \
            "host, kernel, AICPU owner, dispatcher, or manifest is missing"
        return 1
    fi
    if ! command -v sha256sum >/dev/null 2>&1; then
        artifact_failure "$variant" "sha256sum is unavailable"
        return 1
    fi

    local lines=()
    mapfile -t lines < "$manifest"
    if [[ ${#lines[@]} -ne 26 ||
          "${lines[0]:-}" != "# schema=pa_scheduler_artifacts/v6" ||
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
          "${lines[11]:-}" != "# phase_id=0" ||
          "${lines[12]:-}" != "# runtime_plan_abi=2" ||
          "${lines[13]:-}" != "# runtime_plan_cell_bytes=4608" ||
          "${lines[14]:-}" != "# runtime_plan_capacity=4352" ||
          "${lines[15]:-}" != "# plan_owner_entry=plan_protocol_aicpu_exec" ||
          "${lines[16]:-}" != "# scheduler_input=aicpu_closed_runtime_plan" ||
          "${lines[17]:-}" != "# runtime_plan_build_backend=simt" ||
          "${lines[18]:-}" != "# runtime_plan_build_backend_id=1" ||
          "${lines[19]:-}" != "# runtime_plan_build_workers=4" ||
          "${lines[20]:-}" != "# runtime_plan_build_leaders=4" ||
          "${lines[21]:-}" != "# runtime_plan_execute_workers=96" ]]; then
        artifact_failure "$variant" \
            "manifest backend/ABI/trace identity does not match"
        return 1
    fi

    local index digest filename extra
    for index in "${!artifacts[@]}"; do
        read -r digest filename extra <<< "${lines[index + 22]}"
        if [[ ! "$digest" =~ ^[[:xdigit:]]{64}$ ||
              "$filename" != "${artifacts[index]}" ||
              -n "${extra:-}" ]]; then
            artifact_failure "$variant" \
                "manifest checksum entry is malformed"
            return 1
        fi
    done
    if ! (cd "$build_dir" &&
          sha256sum --check --strict --status "$manifest_name"); then
        artifact_failure "$variant" "artifact SHA256 values do not match"
        return 1
    fi

    # manifest 必须晚于正式入口、SIMT adapter/runtime、被复用的 Scalar
    # continuation/Host/AICPU producer 和公共 Plan 协议。这里只比较源码
    # mtime，不读取 Host 产生的 task 身份或业务计划。
    local stale_source
    stale_source="$({
        find \
            "$SCRIPT_DIR/common" \
            "$SCRIPT_DIR/adapter" \
            "$SCRIPT_DIR/ccec" \
            "$SCRIPT_DIR/cpu" \
            "$SCRIPT_DIR/test" \
            "$SCALAR_ROOT/common" \
            "$SCALAR_ROOT/ccec" \
            "$SCALAR_ROOT/aicpu" \
            "$SCRIPT_DIR/../../common" \
            -type f \
            \( -name '*.h' -o -name '*.cpp' -o -name '*.sh' -o -name '*.map' \) \
            -newer "$manifest" -print
        for source in \
            "$REPO_ROOT/examples/a5/fully_distributed_within_core/paged_attention_unroll/kernels/orchestration/paged_attention_orch.cpp" \
            "$REPO_ROOT/tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/common/protocol_probe/plan_aicpu_dispatcher.cpp";
        do
            if [[ "$source" -nt "$manifest" ]]; then
                printf '%s\n' "$source"
            fi
        done
    } | head -n 1)"
    if [[ -n "$stale_source" ]]; then
        artifact_failure "$variant" \
            "source is newer than the atomic four-artifact manifest: $stale_source"
        return 1
    fi
    echo "[CHECK] ordinary SIMT CCEC manifest verified: $manifest"
}

run_variant() {
    local variant="$1"
    shift
    local build_dir="$SCRIPT_DIR/build/ccec/$TENSORMAP_MODE/$variant"
    validate_artifacts "$variant"
    "$build_dir/pa_scheduler_host" \
        --kernel "$build_dir/pa_scheduler_kernel.o" "$@"
}

reject_managed_options() {
    local action="$1"
    shift
    local argument
    for argument in "$@"; do
        case "$argument" in
            --kernel|--kernel=*|--swimlane-json|--swimlane-json=*|\
            --no-swimlane|--trace-atomics|--analyze-swimlane|\
            --pmu-window|--pmu-window=*|--pmu-json|--pmu-json=*)
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

case "$ACTION" in
    build)
        [[ $# -eq 0 ]] || {
            echo "The build action does not accept benchmark options." >&2
            exit 1
        }
        "$SCRIPT_DIR/ccec/build.sh" swimlane
        ;;
    run)
        run_variant swimlane "$@"
        ;;
    smoke)
        run_variant swimlane --batches 1 --runs 1 \
            --winner-workload scalar-nop --nop-count 0 "$@"
        ;;
    swimlane)
        reject_managed_options swimlane "$@"
        PYTHON_BIN="${PYTHON:-python3}"
        CONVERTER="$PA_SCHEDULER_DIR/swimlane_converter.py"
        ANALYZER="$PA_SCHEDULER_DIR/swimlane_exclusive_analyzer.py"
        if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 ||
           [[ ! -f "$CONVERTER" || ! -f "$ANALYZER" ]]; then
            echo "Missing Python or pa_scheduler swimlane tools." >&2
            exit 1
        fi
        OUTPUT_ROOT="$PA_SCHEDULER_DIR/outputs/pa_scheduler_aicpu_plan_simt_ordinary_swimlane_$(date -u +%Y%m%d_%H%M%S)_$$"
        BACKEND_OUTPUT="$OUTPUT_ROOT/ccec"
        RAW_JSON="$BACKEND_OUTPUT/l2_swimlane_records.json"
        mkdir -p "$BACKEND_OUTPUT"
        run_variant swimlane \
            --runs 1 --trace-atomics --swimlane-json "$RAW_JSON" "$@"
        "$PYTHON_BIN" "$CONVERTER" "$RAW_JSON" \
            -o "$BACKEND_OUTPUT/merged_swimlane.json"
        "$PYTHON_BIN" "$ANALYZER" "$RAW_JSON" \
            -o "$BACKEND_OUTPUT/swimlane_exclusive_analysis.json"
        echo "[SWIMLANE] output_root=$OUTPUT_ROOT"
        ;;
    build-perf-clock)
        [[ $# -eq 0 ]] || {
            echo "The build-perf-clock action does not accept benchmark options." >&2
            exit 1
        }
        "$SCRIPT_DIR/ccec/build.sh" perf-clock
        ;;
    verify)
        VERIFY_VARIANT="${1:-perf-clock}"
        if [[ $# -gt 1 ||
              ("$VERIFY_VARIANT" != "swimlane" &&
               "$VERIFY_VARIANT" != "perf-clock") ]]; then
            echo "Usage: $0 verify ccec [swimlane|perf-clock]" >&2
            exit 1
        fi
        validate_artifacts "$VERIFY_VARIANT"
        ;;
    perf-clock)
        reject_managed_options perf-clock "$@"
        run_variant perf-clock --no-swimlane "$@"
        ;;
    *)
        echo "Unknown action: $ACTION" >&2
        usage >&2
        exit 1
        ;;
esac
