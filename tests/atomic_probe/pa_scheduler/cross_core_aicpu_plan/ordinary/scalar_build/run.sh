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
# converter/analyzer 属于 pa_scheduler 公共工具；本目录只拥有
# AICPU Plan + ordinary TensorMap + Scalar Build 实现。
PA_SCHEDULER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
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

本目录固定为 AICPU Plan + Scalar Build + ordinary TensorMap PA：
  - 不接受 private/shared 模式参数；
  - 不构建 AscendC；
  - 首阶段只保留 swimlane 与无泳道端到端构建，不提供 submit-PMU；
  - plan-ahead-closed（默认）先完整 Close Plan 再启动 Scalar Build；
  - streaming-future 在 prefill 后允许 Scalar 持有 future ticket。
    两者都不执行 96 核 replay/Claim。
  - perf-clock 是现有命令名；主口径 pipeline_e2e 覆盖 Plan producer
    起点到 Scalar FinalDrain 结束，plan/scalar 子时间只用于归因。
    streaming-future 的 aicore_time_us 会包含 Host 先等 Plan stream
    的时间，只是 AICore 上界，不能作为两策略 A/B 主口径。

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

Plan Build 由 execution drain 直接收口，不再使用 final-barrier 选项。

PA_SHARED_INSERT_TURN_GROUPS 可选择已有 shared 插入完成链身份：
  1|2|4|8|16|32|64|128（默认 1）。构建与运行必须使用同一值，
  CCEC manifest 会拒绝交叉消费不同身份的产物。

PA_RUNTIME_PLAN_PIPELINE_POLICY 可选：
  0|plan-ahead-closed（默认），不接受显式 prefill；
  1|streaming-future，PA_AICPU_PLAN_READY_PREFILL_TASKS 默认 128。
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

resolve_pipeline_policy() {
    local policy_input="${PA_RUNTIME_PLAN_PIPELINE_POLICY:-0}"
    case "$policy_input" in
        0|plan-ahead-closed)
            PIPELINE_NAME="plan-ahead-closed"
            PIPELINE_KEY="$PIPELINE_NAME"
            LAUNCH_ORDER="plan-sync-before-aicore"
            PRODUCER_READY="closed"
            CONSUMER_ADMISSION="closed-only"
            READY_PREFILL_TASKS=0
            SCHEDULER_INPUT="aicpu_closed_runtime_plan"
            if [[ -n "${PA_AICPU_PLAN_READY_PREFILL_TASKS+x}" ]]; then
                echo "plan-ahead-closed does not accept PA_AICPU_PLAN_READY_PREFILL_TASKS." >&2
                exit 1
            fi
            ;;
        1|streaming-future)
            PIPELINE_NAME="streaming-future"
            LAUNCH_ORDER="dual-stream-overlap"
            PRODUCER_READY="prefill"
            CONSUMER_ADMISSION="ready-future-ticket"
            READY_PREFILL_TASKS="${PA_AICPU_PLAN_READY_PREFILL_TASKS:-128}"
            if [[ ! "$READY_PREFILL_TASKS" =~ ^[0-9]+$ ]] ||
               ((READY_PREFILL_TASKS == 0 || READY_PREFILL_TASKS > 32768)); then
                echo "PA_AICPU_PLAN_READY_PREFILL_TASKS must be an integer in [1, 32768]." >&2
                exit 1
            fi
            PIPELINE_KEY="streaming-future-p${READY_PREFILL_TASKS}"
            SCHEDULER_INPUT="aicpu_streaming_runtime_plan"
            ;;
        *)
            echo "PA_RUNTIME_PLAN_PIPELINE_POLICY must be 0|plan-ahead-closed or 1|streaming-future." >&2
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
    printf '%s/build/cpu/%s/%s/%s/%s\n' \
        "$SCRIPT_DIR" "$TENSORMAP_MODE" "$PIPELINE_KEY" "$variant" "$binary"
}

build_backend() {
    local backend="$1"
    local variant="$2"
    "$SCRIPT_DIR/$backend/build.sh" "$variant"
}

ccec_artifact_failure() {
    local variant="$1"
    local reason="$2"
    echo "Invalid CCEC cross-core shared artifact set ($variant): $reason" >&2
    if [[ "$variant" == "perf-clock" ]]; then
        echo "Run: $0 build-perf-clock ccec" >&2
    else
        echo "Run: $0 build ccec" >&2
    fi
    return 1
}

validate_ccec_artifacts() {
    local variant="$1"
    local build_dir="$SCRIPT_DIR/build/ccec/$TENSORMAP_MODE/$PIPELINE_KEY/$variant"
    local manifest_name="pa_scheduler_artifacts.manifest"
    local manifest="$build_dir/$manifest_name"
    local artifacts=(
        pa_scheduler_host
        pa_scheduler_kernel.o
        pa_scheduler_clock_correlation_kernel.o
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
        ccec_artifact_failure "$variant" "unsupported variant"
        return 1
    fi

    if [[ ! -x "$build_dir/pa_scheduler_host" ||
          ! -s "$build_dir/pa_scheduler_kernel.o" ||
          ! -s "$build_dir/pa_scheduler_clock_correlation_kernel.o" ||
          ! -s "$build_dir/libpa_scheduler_plan_aicpu.so" ||
          ! -s "$build_dir/libpa_scheduler_plan_dispatcher.so" ||
          ! -s "$manifest" ]]; then
        ccec_artifact_failure "$variant" \
            "host, kernel, AICPU owner, dispatcher, or ready manifest is missing"
        return 1
    fi
    if ! command -v sha256sum >/dev/null 2>&1; then
        ccec_artifact_failure "$variant" "sha256sum is unavailable"
        return 1
    fi

    local lines=()
    mapfile -t lines < "$manifest"
    if [[ ${#lines[@]} -ne 30 ||
          "${lines[0]:-}" != "# schema=pa_scheduler_artifacts/v8" ||
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
          "${lines[12]:-}" != "# runtime_plan_abi=3" ||
          "${lines[13]:-}" != "# runtime_plan_cell_bytes=4608" ||
          "${lines[14]:-}" != "# runtime_plan_capacity=4352" ||
          "${lines[15]:-}" != "# plan_owner_entry=plan_protocol_aicpu_exec" ||
          "${lines[16]:-}" != "# scheduler_input=$SCHEDULER_INPUT" ||
          "${lines[17]:-}" != "# pipeline=$PIPELINE_NAME" ||
          "${lines[18]:-}" != "# launch_order=$LAUNCH_ORDER" ||
          "${lines[19]:-}" != "# producer_ready=$PRODUCER_READY" ||
          "${lines[20]:-}" != "# consumer_admission=$CONSUMER_ADMISSION" ||
          "${lines[21]:-}" != "# prefill=$READY_PREFILL_TASKS" ||
          "${lines[22]:-}" != "# clock_correlation_abi=2" ||
          "${lines[23]:-}" != "# clock_correlation_samples=8" ||
          "${lines[24]:-}" != "# clock_correlation_max_alignment_error_ns=50000" ]]; then
        ccec_artifact_failure "$variant" "manifest identity or trace layout does not match"
        return 1
    fi

    local index digest filename extra
    for index in "${!artifacts[@]}"; do
        read -r digest filename extra <<< "${lines[index + 25]}"
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
    # manifest 必须晚于会进入 Host/Kernel/AICPU owner 的本目录源码。该门槛
    # 专门拒绝“只重编 Host 后把旧 Kernel 重新写进 manifest”的混合产物；
    # 完整 build.sh 会在五个镜像全部成功后最后原子发布 manifest。
    local stale_source
    stale_source="$({
        find \
            "$SCRIPT_DIR/common" \
            "$SCRIPT_DIR/ccec" \
            "$SCRIPT_DIR/aicpu" \
            "$SCRIPT_DIR/../../common" \
            -type f \
            \( -name '*.h' -o -name '*.cpp' -o -name '*.sh' -o -name '*.map' \) \
            -newer "$manifest" -print
        for source in \
            "$REPO_ROOT/examples/a5/fully_distributed_within_core/paged_attention_unroll/kernels/orchestration/paged_attention_orch.cpp" \
            "$REPO_ROOT/src/a5/runtime/fully_distributed_within_core/orchestration/pto_orchestration_api.h" \
            "$REPO_ROOT/src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/dist_engine_api.h" \
            "$REPO_ROOT/src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/common/target.h" \
            "$REPO_ROOT/src/a5/runtime/fully_distributed_within_core/runtime/dist_engine/aicore/dist_engine.cpp" \
            "$REPO_ROOT/tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/common/protocol_probe/plan_aicpu_dispatcher.cpp";
        do
            if [[ "$source" -nt "$manifest" ]]; then
                printf '%s\n' "$source"
            fi
        done
    } | head -n 1)"
    if [[ -n "$stale_source" ]]; then
        ccec_artifact_failure "$variant" \
            "source is newer than the atomic five-artifact manifest: $stale_source"
        return 1
    fi
    echo "[CHECK] CCEC manifest verified: $manifest"
    echo "[POLICY] pipeline=$PIPELINE_NAME launch_order=$LAUNCH_ORDER producer_ready=$PRODUCER_READY consumer_admission=$CONSUMER_ADMISSION prefill=$READY_PREFILL_TASKS"
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
            local build_dir="$SCRIPT_DIR/build/ccec/$TENSORMAP_MODE/$PIPELINE_KEY/$variant"
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
            --runs|--runs=*)
                if [[ "$action" != "perf-clock" ]]; then
                    echo "The $action action manages or forbids $argument." >&2
                    exit 1
                fi
                ;;
            --kernel|--kernel=*|--swimlane-json|--swimlane-json=*|\
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
resolve_pipeline_policy
echo "[POLICY] pipeline=$PIPELINE_NAME launch_order=$LAUNCH_ORDER producer_ready=$PRODUCER_READY consumer_admission=$CONSUMER_ADMISSION prefill=$READY_PREFILL_TASKS"

# CCEC kernel path is owned by the verified manifest directory.  Reject every
# user spelling before dispatch so a later duplicate option cannot replace the
# SHA-checked kernel/adjacent SO set inside Host parsing.
if [[ "$BACKEND" == "ccec" ]]; then
    for argument in "$@"; do
        case "$argument" in
            --kernel|--kernel=*)
                echo "The $ACTION ccec action manages $argument." >&2
                exit 1
                ;;
        esac
    done
fi

# 首阶段不提供 submit-PMU。即使复制来的 host 仍保留底层诊断解析能力，
# 统一入口也必须在任何构建、运行或文件创建前拒绝这些参数。
for argument in "$@"; do
    case "$argument" in
        --pmu-window|--pmu-window=*|--pmu-scalar-nops|--pmu-scalar-nops=*|\
        --pmu-icache-trials|--pmu-icache-trials=*|--pmu-json|--pmu-json=*)
            echo "submit-PMU is not available in the first cross-core PA stage: $argument" >&2
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
        OUTPUT_ROOT="$PA_SCHEDULER_DIR/outputs/pa_scheduler_aicpu_plan_scalar_ordinary_${PIPELINE_KEY}_swimlane_$(date -u +%Y%m%d_%H%M%S)_$$"
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
        run_backend "$BACKEND" perf-clock --no-swimlane "$@"
        ;;
    *)
        echo "Unknown action: $ACTION" >&2
        usage >&2
        exit 1
        ;;
esac
