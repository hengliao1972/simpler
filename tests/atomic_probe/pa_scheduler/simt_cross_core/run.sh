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
run-g0 先做 A5 precheck、构建清单校验和 600 秒有界运行，再由 2048 SIMT thread/64 个 lane0 leader
并发构建、AIC/AIV 执行完整 PA DAG；
环境提供 task-submit 时，必须先在锁外做 A5 precheck，再从其 --run 命令内调用；run-g0 会把
$TASK_DEVICE 自动注入 host，锁内调用不得再传 --device。
EOF
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
    build-g0)
        if [[ $# -ne 0 ]]; then
            echo "build-g0 does not accept additional arguments." >&2
            exit 1
        fi
        "$GM_ROOT/cpu/build_g0.sh"
        "$GM_ROOT/ccec/build_g0.sh"
        ;;
    run-g0)
        if [[ ! -x "$GM_BUILD/simt_cross_core_g0_host" ||
              ! -s "$GM_BUILD/simt_cross_core_g0_kernel.o" ||
              ! -s "$G0_BUILD_MANIFEST" ]]; then
            echo "G0 artifacts are missing; run: $0 build-g0" >&2
            exit 1
        fi
        if ! (cd "$SCRIPT_DIR" && sha256sum --check --status "$G0_BUILD_MANIFEST"); then
            echo "G0 sources and runtime artifacts do not match the successful-build manifest; run: $0 build-g0" >&2
            exit 1
        fi
        precheck_rc=0
        if "$REPO_ROOT/.claude/skills/onboard-arch-precheck/check.sh" a5; then
            precheck_rc=0
        else
            precheck_rc=$?
        fi
        if (( precheck_rc != 0 )); then
            if (( precheck_rc != 1 )); then
                echo "G0 A5 arch precheck rejected this silicon (exit=$precheck_rc); refusing hardware access." >&2
                exit "$precheck_rc"
            fi
            if [[ -n "${TASK_DEVICE:-}" ]] || command -v task-submit >/dev/null 2>&1; then
                echo "G0 A5 arch precheck failed; refusing the submitted/managed hardware run." >&2
                exit 1
            fi
            printf '%s\n' \
                '[WARN] A5 arch precheck unavailable; proceeding only because this is an explicitly authorized unlocked run' \
                >&2
        fi
        if [[ -z "${TASK_DEVICE:-}" ]] && command -v task-submit >/dev/null 2>&1; then
            printf '%s\n' \
                'run-g0 detected task-submit but is not running inside a submitted task; run the A5 precheck first, then wrap run-g0 with task-submit --device auto --device-num 1 --run "...".' \
                >&2
            exit 1
        fi
        if [[ -z "${TASK_DEVICE:-}" ]]; then
            printf '%s\n' \
                '[WARN] task-submit not found; running unlocked — results may be noisy if any other process is on this NPU' \
                >&2
        fi
        g0_device_args=()
        for argument in "$@"; do
            case "$argument" in
                --kernel|--kernel=*)
                    echo "run-g0 uses the kernel covered by its successful-build manifest; do not override --kernel." >&2
                    exit 1
                    ;;
                --device|--device=*)
                    if [[ -n "${TASK_DEVICE:-}" ]]; then
                        echo "run-g0 injects --device from TASK_DEVICE inside task-submit; do not pass --device explicitly." >&2
                        exit 1
                    fi
                    ;;
            esac
        done
        if [[ -n "${TASK_DEVICE:-}" ]]; then
            g0_device_args=(--device "$TASK_DEVICE")
        fi
        timeout --foreground 600s "$GM_BUILD/simt_cross_core_g0_host" \
            --kernel "$GM_BUILD/simt_cross_core_g0_kernel.o" "${g0_device_args[@]}" "$@"
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
