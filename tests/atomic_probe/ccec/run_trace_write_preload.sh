#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the LICENSE file for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the license.
# -----------------------------------------------------------------------------------------------------------

# Build and run the isolated 1:2 mixed A5 trace-write preload experiment.
#
# Usage:
#   ./run_trace_write_preload.sh
#   ./run_trace_write_preload.sh build
#   ./run_trace_write_preload.sh run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/trace_write_preload"
ACTION="${1:-all}"
if [[ "$ACTION" != "all" && "$ACTION" != "build" && "$ACTION" != "run" ]]; then
    echo "Usage: $0 [all|build|run]" >&2
    exit 1
fi
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [all|build|run]" >&2
    exit 1
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set; source the CANN environment first." >&2
    exit 1
fi

CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD="$ASCEND_HOME_PATH/bin/ld.lld"
CXX_BIN="${CXX:-g++}"
READELF_BIN="${READELF:-readelf}"
PTO_INCLUDE_ROOT="${PTO_ISA_ROOT:-$ASCEND_HOME_PATH/x86_64-linux}"
KERNEL="$BUILD_DIR/trace_write_preload_kernel.o"
HOST="$BUILD_DIR/trace_write_preload_host"

if [[ ! -x "$CCEC" || ! -x "$LD" ]]; then
    echo "ccec or ld.lld is missing under ASCEND_HOME_PATH=$ASCEND_HOME_PATH" >&2
    exit 1
fi
if [[ ! -f "$PTO_INCLUDE_ROOT/include/pto/common/kernel_meta.hpp" ]]; then
    echo "PTO metadata header is missing: $PTO_INCLUDE_ROOT/include/pto/common/kernel_meta.hpp" >&2
    exit 1
fi
if ! command -v "$CXX_BIN" >/dev/null 2>&1; then
    echo "Host C++ compiler is missing: $CXX_BIN" >&2
    exit 1
fi
if ! command -v "$READELF_BIN" >/dev/null 2>&1; then
    echo "readelf is required: $READELF_BIN" >&2
    exit 1
fi

check_kernel_elf() {
    if [[ ! -f "$KERNEL" ]]; then
        echo "Kernel ELF is missing: $KERNEL" >&2
        exit 1
    fi
    local symbols sections
    symbols="$("$READELF_BIN" --symbols --wide "$KERNEL")"
    sections="$("$READELF_BIN" --sections --wide "$KERNEL")"
    for entry in trace_write_preload_0_mix_aic trace_write_preload_0_mix_aiv; do
        if ! awk -v name="$entry" \
            '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {found = 1} END {exit !found}' \
            <<<"$symbols"; then
            echo "Missing non-empty mixed entry: $entry" >&2
            exit 1
        fi
        if [[ "$sections" != *".ascend.meta.$entry"* ]]; then
            echo "Missing metadata section: .ascend.meta.$entry" >&2
            exit 1
        fi
    done
    echo "[CHECK] 1:2 mixed AIC/AIV entries and metadata PASS"
}

build_probe() {
    mkdir -p "$BUILD_DIR"
    local common_flags=(
        -c -O3 -g -x cce -Wall -std=c++17
        --cce-aicore-only
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=false
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-dcci-before-kernel-end=false
        -I"$SCRIPT_DIR"
        -I"$PTO_INCLUDE_ROOT/include"
    )

    echo "[BUILD] CCEC AIC trace-write preload entry"
    "$CCEC" "${common_flags[@]}" \
        --cce-aicore-arch=dav-c310-cube \
        -DTRACE_WRITE_PRELOAD_BUILD_AIC \
        -o "$BUILD_DIR/trace_write_preload_aic.o" \
        "$SCRIPT_DIR/trace_write_preload.cpp"
    echo "[BUILD] CCEC AIV trace-write preload entry"
    "$CCEC" "${common_flags[@]}" \
        --cce-aicore-arch=dav-c310-vec \
        -DTRACE_WRITE_PRELOAD_BUILD_AIV \
        -o "$BUILD_DIR/trace_write_preload_aiv.o" \
        "$SCRIPT_DIR/trace_write_preload.cpp"
    echo "[BUILD] Static 1:2 mixed AICore ELF"
    "$LD" -m aicorelinux -Ttext=0 -static \
        -o "$KERNEL" \
        "$BUILD_DIR/trace_write_preload_aic.o" \
        "$BUILD_DIR/trace_write_preload_aiv.o"
    check_kernel_elf

    echo "[BUILD] host runner"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
        -I"$ASCEND_HOME_PATH/include" \
        "$SCRIPT_DIR/trace_write_preload_host.cpp" \
        -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -lascendcl \
        -o "$HOST"
    echo "[BUILD] complete: $BUILD_DIR"
}

run_probe() {
    if [[ ! -x "$HOST" || ! -f "$KERNEL" ]]; then
        echo "Build artifacts are incomplete; run '$0 build' first." >&2
        exit 1
    fi
    check_kernel_elf
    echo "[RUN] device=${ATOMIC_PROBE_DEVICE:-${TASK_DEVICE:-0}}"
    timeout "${TRACE_WRITE_PRELOAD_TIMEOUT:-240}" "$HOST" "$KERNEL"
}

echo "=== CCEC Trace Write Preload Experiment ==="
echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git_head=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "git_branch=$(git -C "$SCRIPT_DIR" branch --show-current 2>/dev/null || echo unknown)"
echo "ascend_home=$ASCEND_HOME_PATH"
echo "ccec=$($CCEC --version 2>&1 | head -n 1)"
echo "action=$ACTION"

if [[ "$ACTION" == "all" || "$ACTION" == "build" ]]; then
    build_probe
fi
if [[ "$ACTION" == "all" || "$ACTION" == "run" ]]; then
    run_probe
fi
