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
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
S0_ROOT="$SCRIPT_DIR/protocol_probe"
S0_BUILD="$S0_ROOT/build/ccec"
ATOMIC_ROOT="$S0_ROOT/simt_atomic"
ATOMIC_BUILD="$ATOMIC_ROOT/build/ccec"
WARP_ROOT="$S0_ROOT/warp_concurrency"
WARP_BUILD="$WARP_ROOT/build/ccec"
GM_ROOT="$SCRIPT_DIR/gm"
GM_BUILD="$GM_ROOT/build/ccec"
G0_BUILD_MANIFEST="$GM_BUILD/g0_build_manifest.sha256"
G0_SWIMLANE_BUILD_MANIFEST="$GM_BUILD/g0_swimlane_build_manifest.sha256"
G0_SWIMLANE_ACL_CONFIG="$GM_ROOT/ccec/g0_swimlane_acl.json"
UBUF_ROOT="$SCRIPT_DIR/ubuf"
U0_BUILD="$UBUF_ROOT/build/ccec"
U0_BUILD_MANIFEST="$U0_BUILD/u0_build_manifest.sha256"
U1_BUILD="$UBUF_ROOT/build/ccec"
U1_BUILD_MANIFEST="$U1_BUILD/u1_build_manifest.sha256"

usage() {
    cat <<'EOF'
Usage:
  ./run.sh build-s0
  ./run.sh run-s0 --device N [--runs N]
  ./run.sh build-atomic
  ./run.sh run-atomic --device N [--runs N]
  ./run.sh build-warp
  ./run.sh run-warp --device N [--runs N]
  ./run.sh build-s1
  ./run.sh run-s1 --device N [--runs N]
  ./run.sh build-s2
  ./run.sh run-s2 --device N [--runs N]
  ./run.sh build-s3
  ./run.sh run-s3 --device N [--runs N]
  ./run.sh build-s4
  ./run.sh run-s4 --device N [--runs N]
  ./run.sh build-g0
  ./run.sh run-g0 [--device N] [--batches 1|256] [--runs N]
  ./run.sh build-g1
  ./run.sh run-g1 [--device N] [--batches 1|256] [--runs N]
  ./run.sh build-gm
  ./run.sh run-gm --builders 1..8 [--device N] [--batches 1|256] [--runs N]
  ./run.sh build-g0-swimlane
  ./run.sh run-g0-swimlane [--device N] [--batches 1|256] --runs 1 --swimlane-json FILE
  ./run.sh build-g1-swimlane
  ./run.sh run-g1-swimlane [--device N] [--batches 1|256] --runs 1 --swimlane-json FILE
  ./run.sh build-gm-swimlane
  ./run.sh run-gm-swimlane --builders 1..8 [--device N] [--batches 1|256] --runs 1 --swimlane-json FILE
  ./run.sh build-u0
  ./run.sh run-u0 [--device N] [--runs N]
  ./run.sh build-u1
  ./run.sh run-u1 [--device N] [--runs N]

build-s0 运行 CPU optimized/ASan/UBSan/TSan，并构建、静态检查 CCEC/ELF。
run-s0 只运行已构建的真实 A5 探针；调用前仍需按仓库规则执行 A5 precheck。
build-atomic 运行 SIMT atomic CPU 三套测试，并构建、静态检查 CCEC/ELF。
run-atomic 在真实 A5 上验证 32/64/1024/2048 线程的 GM uint64 CAS/add 同地址竞争。
build-warp 运行同 warp 串行/跨 warp 独立推进的 CPU 三套测试，并构建、静态检查 CCEC/ELF。
run-warp 在真实 A5 上用 bounded handshake 和 CLOCK64 区间区分同 warp 与跨 warp 执行。
build-s1 运行单 Vector 的 CPU 三套测试，并构建、静态检查 1:2 mixed CCEC/ELF。
run-s1 运行 AIV0 SIMT builder -> AIV1 Vector executor 的真实 A5 四模式探针。
build-s2 运行单 Cube 的 CPU 三套测试，并构建、静态检查 1:2 mixed CCEC/ELF。
run-s2 运行 AIV0 SIMT builder -> AIC Cube executor 的真实 A5 四模式探针。
build-s3 运行 Vector+Cube 双 task 的 CPU 三套测试和 mixed CCEC/ELF 静态检查。
run-s3 运行 AIV0 双 task builder、AIV1 Vector executor、AIC Cube executor 的真实 A5 探针。
build-s4 运行 4 个 SIMT warp/128 thread 构建 16 task 的 CPU 三套测试和 mixed CCEC/ELF 静态检查。
run-s4 运行 AIV0 构建 16 task、AIV1/AIC 各以 busy depth 1 执行 8 task 的真实 A5 探针。
build-g0 运行完整 PA 调度的 CPU 三套测试，并构建、静态检查 G0 CCEC/bitcode/mixed ELF/ACL host。
run-g0 先做 A5 precheck、构建清单校验和 600 秒有界运行，再由 512 SIMT thread/16 个 lane0 leader
并发构建、AIC/AIV 执行完整 PA DAG；
build-g1 复用同一份参数化源码，验证单/双 builder CPU 模型并重建完整 PA CCEC/ELF/ACL host。
run-g1 执行 AIV0+AIV1 两个独立 VF：各 512 SIMT thread/16 个 lane0 leader，按 task-id 静态分片，
每个 task 只有一个固定 builder；其余 94 个 AIC/AIV owner 只执行完整 PA DAG，每个 warp 仍严格只有 lane0 工作。
build-gm 在同一份源码上验证 1..8 个 builder，并重建通用 GM 产物。
run-gm 显式接收 --builders N；AIV0..AIV(N-1) 只构建，剩余 AIV 只执行，task 在 N*16 个 leader 间唯一分片。
build-g0-swimlane 构建与生产 G0 分离的 profiling 变体；builder/executor 按 task 写独立 cache line。
run-g0-swimlane 在真实 A5 上导出 Chrome Trace JSON；该数据用于观察时序，性能结论使用关闭埋点的 run-g0。
build-g1-swimlane 复用同一 profiling 源码并验证双 builder trace 容量与 writer 映射。
run-g1-swimlane 导出 AIV0+AIV1 双 builder Chrome Trace JSON；性能结论使用关闭埋点的 run-g1。
build-gm-swimlane 构建覆盖 1..8 builder writer 上界的 profiling 产物。
run-gm-swimlane 显式接收 --builders N，并为对应 N-builder 配置导出一份不覆盖旧文件的泳道图。
build-u0 运行 UBUF 单槽 CPU 三套测试，并构建、静态检查 U0 CCEC/bitcode/mixed ELF/ACL host。
run-u0 在真实 A5 上验证 AIV0 的 64 个 lane0 leader 竞争单个 UBUF slot，由同一
SIMT leader 读回 UBUF 并直接写 GM；AIV1 Scalar 只执行，mte3_count 必须为 0。
build-u1 运行 UBUF 四槽/128 task CPU 三套测试，并构建、静态检查 U1 CCEC/bitcode/mixed ELF/ACL host。
run-u1 在真实 A5 上验证 4×1152 B UBUF slot 的并发驻留、generation 复用和 128 task 全量收口。
环境提供 task-submit 时，必须先在锁外做 A5 precheck，再从其 --run 命令内调用；run-g0 会把
$TASK_DEVICE 自动注入 host，锁内调用不得再传 --device。run-g1/run-u0/run-u1 遵循同一规则。
EOF
}

require_a5_access() {
    local stage_label="$1"
    local precheck_rc=0
    if "$REPO_ROOT/.claude/skills/onboard-arch-precheck/check.sh" a5; then
        precheck_rc=0
    else
        precheck_rc=$?
    fi
    if (( precheck_rc != 0 )); then
        if (( precheck_rc != 1 )); then
            echo "$stage_label A5 arch precheck rejected this silicon (exit=$precheck_rc); refusing hardware access." >&2
            exit "$precheck_rc"
        fi
        if [[ -n "${TASK_DEVICE:-}" ]] || command -v task-submit >/dev/null 2>&1; then
            echo "$stage_label A5 arch precheck failed; refusing the submitted/managed hardware run." >&2
            exit 1
        fi
        printf '%s\n' \
            '[WARN] A5 arch precheck unavailable; proceeding only because this is an explicitly authorized unlocked run' \
            >&2
    fi
    if [[ -z "${TASK_DEVICE:-}" ]] && command -v task-submit >/dev/null 2>&1; then
        printf '%s\n' \
            "$ACTION detected task-submit but is not running inside a submitted task; run the A5 precheck first, then wrap $ACTION with task-submit --device auto --device-num 1 --run \"...\"." \
            >&2
        exit 1
    fi
    if [[ -z "${TASK_DEVICE:-}" ]]; then
        printf '%s\n' \
            '[WARN] task-submit not found; running unlocked — results may be noisy if any other process is on this NPU' \
            >&2
    fi
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
fi

ACTION="$1"
shift
case "$ACTION" in
    build-s0)
        if [[ $# -ne 0 ]]; then
            echo "build-s0 does not accept additional arguments." >&2
            exit 1
        fi
        "$S0_ROOT/cpu/build.sh"
        "$S0_ROOT/ccec/build.sh"
        ;;
    run-s0)
        if [[ ! -x "$S0_BUILD/simt_cross_core_s0_host" ||
              ! -s "$S0_BUILD/simt_cross_core_s0_kernel.o" ]]; then
            echo "S0 artifacts are missing; run: $0 build-s0" >&2
            exit 1
        fi
        "$S0_BUILD/simt_cross_core_s0_host" \
            --kernel "$S0_BUILD/simt_cross_core_s0_kernel.o" "$@"
        ;;
    build-atomic)
        if [[ $# -ne 0 ]]; then
            echo "build-atomic does not accept additional arguments." >&2
            exit 1
        fi
        "$ATOMIC_ROOT/cpu/build.sh"
        "$ATOMIC_ROOT/ccec/build.sh"
        ;;
    run-atomic)
        if [[ ! -x "$ATOMIC_BUILD/simt_cross_core_atomic_host" ||
              ! -s "$ATOMIC_BUILD/simt_cross_core_atomic_kernel.o" ]]; then
            echo "SIMT atomic artifacts are missing; run: $0 build-atomic" >&2
            exit 1
        fi
        "$ATOMIC_BUILD/simt_cross_core_atomic_host" \
            --kernel "$ATOMIC_BUILD/simt_cross_core_atomic_kernel.o" "$@"
        ;;
    build-warp)
        if [[ $# -ne 0 ]]; then
            echo "build-warp does not accept additional arguments." >&2
            exit 1
        fi
        "$WARP_ROOT/cpu/build.sh"
        "$WARP_ROOT/ccec/build.sh"
        ;;
    run-warp)
        if [[ ! -x "$WARP_BUILD/simt_cross_core_warp_concurrency_host" ||
              ! -s "$WARP_BUILD/simt_cross_core_warp_concurrency_kernel.o" ]]; then
            echo "warp-concurrency artifacts are missing; run: $0 build-warp" >&2
            exit 1
        fi
        "$WARP_BUILD/simt_cross_core_warp_concurrency_host" \
            --kernel "$WARP_BUILD/simt_cross_core_warp_concurrency_kernel.o" "$@"
        ;;
    build-s1)
        if [[ $# -ne 0 ]]; then
            echo "build-s1 does not accept additional arguments." >&2
            exit 1
        fi
        "$GM_ROOT/cpu/build_s1.sh"
        "$GM_ROOT/ccec/build_s1.sh"
        ;;
    run-s1)
        if [[ ! -x "$GM_BUILD/simt_cross_core_s1_host" ||
              ! -s "$GM_BUILD/simt_cross_core_s1_kernel.o" ]]; then
            echo "S1 artifacts are missing; run: $0 build-s1" >&2
            exit 1
        fi
        "$GM_BUILD/simt_cross_core_s1_host" \
            --kernel "$GM_BUILD/simt_cross_core_s1_kernel.o" "$@"
        ;;
    build-s2)
        if [[ $# -ne 0 ]]; then
            echo "build-s2 does not accept additional arguments." >&2
            exit 1
        fi
        "$GM_ROOT/cpu/build_s2.sh"
        "$GM_ROOT/ccec/build_s2.sh"
        ;;
    run-s2)
        if [[ ! -x "$GM_BUILD/simt_cross_core_s2_host" ||
              ! -s "$GM_BUILD/simt_cross_core_s2_kernel.o" ]]; then
            echo "S2 artifacts are missing; run: $0 build-s2" >&2
            exit 1
        fi
        "$GM_BUILD/simt_cross_core_s2_host" \
            --kernel "$GM_BUILD/simt_cross_core_s2_kernel.o" "$@"
        ;;
    build-s3)
        if [[ $# -ne 0 ]]; then
            echo "build-s3 does not accept additional arguments." >&2
            exit 1
        fi
        "$GM_ROOT/cpu/build_s3.sh"
        "$GM_ROOT/ccec/build_s3.sh"
        ;;
    run-s3)
        if [[ ! -x "$GM_BUILD/simt_cross_core_s3_host" ||
              ! -s "$GM_BUILD/simt_cross_core_s3_kernel.o" ]]; then
            echo "S3 artifacts are missing; run: $0 build-s3" >&2
            exit 1
        fi
        "$GM_BUILD/simt_cross_core_s3_host" \
            --kernel "$GM_BUILD/simt_cross_core_s3_kernel.o" "$@"
        ;;
    build-s4)
        if [[ $# -ne 0 ]]; then
            echo "build-s4 does not accept additional arguments." >&2
            exit 1
        fi
        "$GM_ROOT/cpu/build_s4.sh"
        "$GM_ROOT/ccec/build_s4.sh"
        ;;
    run-s4)
        if [[ ! -x "$GM_BUILD/simt_cross_core_s4_host" ||
              ! -s "$GM_BUILD/simt_cross_core_s4_kernel.o" ]]; then
            echo "S4 artifacts are missing; run: $0 build-s4" >&2
            exit 1
        fi
        "$GM_BUILD/simt_cross_core_s4_host" \
            --kernel "$GM_BUILD/simt_cross_core_s4_kernel.o" "$@"
        ;;
    build-g0|build-g1|build-gm)
        if [[ $# -ne 0 ]]; then
            echo "$ACTION does not accept additional arguments." >&2
            exit 1
        fi
        "$GM_ROOT/cpu/build_g0.sh"
        "$GM_ROOT/ccec/build_g0.sh"
        ;;
    build-g0-swimlane|build-g1-swimlane|build-gm-swimlane)
        if [[ $# -ne 0 ]]; then
            echo "$ACTION does not accept additional arguments." >&2
            exit 1
        fi
        "$GM_ROOT/cpu/build_g0.sh"
        SIMT_CROSS_CORE_G0_VARIANT=swimlane "$GM_ROOT/ccec/build_g0.sh"
        ;;
    build-u0)
        if [[ $# -ne 0 ]]; then
            echo "build-u0 does not accept additional arguments." >&2
            exit 1
        fi
        "$UBUF_ROOT/cpu/build_u0.sh"
        "$UBUF_ROOT/ccec/build_u0.sh"
        ;;
    build-u1)
        if [[ $# -ne 0 ]]; then
            echo "build-u1 does not accept additional arguments." >&2
            exit 1
        fi
        "$UBUF_ROOT/cpu/build_u1.sh"
        "$UBUF_ROOT/ccec/build_u1.sh"
        ;;
    run-g0|run-g1|run-gm)
        full_pa_stage="${ACTION#run-}"
        full_pa_fixed_builder_args=(--builders 1)
        full_pa_builder_arg_required=0
        if [[ "$ACTION" == "run-g1" ]]; then
            full_pa_fixed_builder_args=(--builders 2)
        elif [[ "$ACTION" == "run-gm" ]]; then
            full_pa_fixed_builder_args=()
            full_pa_builder_arg_required=1
        fi
        if [[ ! -x "$GM_BUILD/simt_cross_core_g0_host" ||
              ! -s "$GM_BUILD/simt_cross_core_g0_kernel.o" ||
              ! -s "$G0_BUILD_MANIFEST" ]]; then
            echo "${full_pa_stage^^} artifacts are missing; run: $0 build-$full_pa_stage" >&2
            exit 1
        fi
        if ! (cd "$SCRIPT_DIR" && sha256sum --check --status "$G0_BUILD_MANIFEST"); then
            echo "${full_pa_stage^^} sources and runtime artifacts do not match the successful-build manifest; run: $0 build-$full_pa_stage" >&2
            exit 1
        fi
        require_a5_access "${full_pa_stage^^}"
        full_pa_device_args=()
        full_pa_builder_arg_seen=0
        for argument in "$@"; do
            case "$argument" in
                --kernel|--kernel=*)
                    echo "$ACTION uses the kernel covered by its successful-build manifest; do not override --kernel." >&2
                    exit 1
                    ;;
                --builders|--builders=*)
                    if [[ "$full_pa_builder_arg_required" -eq 0 ]]; then
                        echo "$ACTION fixes its builder count; use run-gm for an explicit 1..8 builder scan." >&2
                        exit 1
                    fi
                    full_pa_builder_arg_seen=1
                    ;;
                --device|--device=*)
                    if [[ -n "${TASK_DEVICE:-}" ]]; then
                        echo "$ACTION injects --device from TASK_DEVICE inside task-submit; do not pass --device explicitly." >&2
                        exit 1
                    fi
                    ;;
            esac
        done
        if [[ "$full_pa_builder_arg_required" -eq 1 && "$full_pa_builder_arg_seen" -ne 1 ]]; then
            echo "run-gm requires an explicit --builders N value in 1..8." >&2
            exit 1
        fi
        if [[ -n "${TASK_DEVICE:-}" ]]; then
            full_pa_device_args=(--device "$TASK_DEVICE")
        fi
        timeout --foreground 600s "$GM_BUILD/simt_cross_core_g0_host" \
            --kernel "$GM_BUILD/simt_cross_core_g0_kernel.o" "${full_pa_fixed_builder_args[@]}" \
            "${full_pa_device_args[@]}" "$@"
        ;;
    run-g0-swimlane|run-g1-swimlane|run-gm-swimlane)
        swimlane_fixed_builder_args=(--builders 1)
        swimlane_builder_arg_required=0
        swimlane_label="G0"
        swimlane_build_action="build-g0-swimlane"
        if [[ "$ACTION" == "run-g1-swimlane" ]]; then
            swimlane_fixed_builder_args=(--builders 2)
            swimlane_label="G1"
            swimlane_build_action="build-g1-swimlane"
        elif [[ "$ACTION" == "run-gm-swimlane" ]]; then
            swimlane_fixed_builder_args=()
            swimlane_builder_arg_required=1
            swimlane_label="GM"
            swimlane_build_action="build-gm-swimlane"
        fi
        if [[ ! -x "$GM_BUILD/simt_cross_core_g0_swimlane_host" ||
              ! -s "$GM_BUILD/simt_cross_core_g0_swimlane_kernel.o" ||
              ! -s "$G0_SWIMLANE_BUILD_MANIFEST" ]]; then
            echo "$swimlane_label swimlane artifacts are missing; run: $0 $swimlane_build_action" >&2
            exit 1
        fi
        if ! (cd "$SCRIPT_DIR" && sha256sum --check --status "$G0_SWIMLANE_BUILD_MANIFEST"); then
            echo "G0 swimlane sources and runtime artifacts do not match the successful-build manifest; rebuild it." >&2
            exit 1
        fi
        require_a5_access "$swimlane_label swimlane"
        full_pa_device_args=()
        swimlane_builder_arg_seen=0
        for argument in "$@"; do
            case "$argument" in
                --kernel|--kernel=*|--acl-config|--acl-config=*)
                    echo "$ACTION fixes its kernel and ACL stack config; do not override them." >&2
                    exit 1
                    ;;
                --builders|--builders=*)
                    if [[ "$swimlane_builder_arg_required" -eq 0 ]]; then
                        echo "$ACTION fixes its builder count; use run-gm-swimlane for an explicit 1..8 builder trace." >&2
                        exit 1
                    fi
                    swimlane_builder_arg_seen=1
                    ;;
                --device|--device=*)
                    if [[ -n "${TASK_DEVICE:-}" ]]; then
                        echo "$ACTION injects --device from TASK_DEVICE; do not pass it explicitly." >&2
                        exit 1
                    fi
                    ;;
            esac
        done
        if [[ "$swimlane_builder_arg_required" -eq 1 && "$swimlane_builder_arg_seen" -ne 1 ]]; then
            echo "run-gm-swimlane requires an explicit --builders N value in 1..8." >&2
            exit 1
        fi
        if [[ -n "${TASK_DEVICE:-}" ]]; then
            full_pa_device_args=(--device "$TASK_DEVICE")
        fi
        timeout --foreground 600s "$GM_BUILD/simt_cross_core_g0_swimlane_host" \
            --kernel "$GM_BUILD/simt_cross_core_g0_swimlane_kernel.o" "${swimlane_fixed_builder_args[@]}" \
            --acl-config "$G0_SWIMLANE_ACL_CONFIG" \
            "${full_pa_device_args[@]}" "$@"
        ;;
    run-u0|run-u1)
        ubuf_stage="${ACTION#run-}"
        ubuf_build="$U0_BUILD"
        ubuf_manifest="$U0_BUILD_MANIFEST"
        if [[ "$ACTION" == "run-u1" ]]; then
            ubuf_build="$U1_BUILD"
            ubuf_manifest="$U1_BUILD_MANIFEST"
        fi
        ubuf_host="$ubuf_build/simt_cross_core_${ubuf_stage}_host"
        ubuf_kernel="$ubuf_build/simt_cross_core_${ubuf_stage}_kernel.o"
        if [[ ! -x "$ubuf_host" || ! -s "$ubuf_kernel" || ! -s "$ubuf_manifest" ]]; then
            echo "${ubuf_stage^^} artifacts are missing; run: $0 build-$ubuf_stage" >&2
            exit 1
        fi
        if ! (cd "$SCRIPT_DIR" && sha256sum --check --status "$ubuf_manifest"); then
            echo "${ubuf_stage^^} sources and runtime artifacts do not match the successful-build manifest; run: $0 build-$ubuf_stage" >&2
            exit 1
        fi
        require_a5_access "${ubuf_stage^^}"
        ubuf_device_args=()
        for argument in "$@"; do
            case "$argument" in
                --kernel|--kernel=*)
                    echo "$ACTION uses the kernel covered by its successful-build manifest; do not override --kernel." >&2
                    exit 1
                    ;;
                --device|--device=*)
                    if [[ -n "${TASK_DEVICE:-}" ]]; then
                        echo "$ACTION injects --device from TASK_DEVICE inside task-submit; do not pass --device explicitly." >&2
                        exit 1
                    fi
                    ;;
            esac
        done
        if [[ -n "${TASK_DEVICE:-}" ]]; then
            ubuf_device_args=(--device "$TASK_DEVICE")
        fi
        timeout --foreground 300s "$ubuf_host" --kernel "$ubuf_kernel" "${ubuf_device_args[@]}" "$@"
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
