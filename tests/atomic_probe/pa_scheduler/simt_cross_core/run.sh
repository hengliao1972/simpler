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
S0_ROOT="$SCRIPT_DIR/protocol_probe"
S0_BUILD="$S0_ROOT/build/ccec"
ATOMIC_ROOT="$S0_ROOT/simt_atomic"
ATOMIC_BUILD="$ATOMIC_ROOT/build/ccec"
WARP_ROOT="$S0_ROOT/warp_concurrency"
WARP_BUILD="$WARP_ROOT/build/ccec"
GM_ROOT="$SCRIPT_DIR/gm"
GM_BUILD="$GM_ROOT/build/ccec"

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
    *)
        usage >&2
        exit 1
        ;;
esac
