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
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
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
HCC="$ASCEND_HOME_PATH/tools/hcc/bin/aarch64-target-linux-gnu-g++"
CXX_BIN="${CXX:-g++}"
READELF_BIN="${READELF:-readelf}"
PTO_INCLUDE_ROOT="${PTO_ISA_ROOT:-$ASCEND_HOME_PATH/x86_64-linux}"
for tool in "$CCEC" "$LD" "$HCC" "$CXX_BIN" "$READELF_BIN"; do
    if ! command -v "$tool" >/dev/null 2>&1 && [[ ! -x "$tool" ]]; then
        echo "Missing required tool: $tool" >&2
        exit 1
    fi
done
if [[ ! -f "$PTO_INCLUDE_ROOT/include/pto/common/kernel_meta.hpp" ]]; then
    echo "Missing PTO metadata header under $PTO_INCLUDE_ROOT/include" >&2
    exit 1
fi

AICPU_TO_AICORE_KERNEL="$BUILD_DIR/aicpu_to_aicore_kernel.o"
AICORE_TO_AICPU_KERNEL="$BUILD_DIR/aicore_to_aicpu_kernel.o"
DISPATCHER="$BUILD_DIR/libsimpler_aicpu_dispatcher.so"
OWNER="$BUILD_DIR/libaicpu_aicore_cache_probe.so"
HOST="$BUILD_DIR/aicpu_aicore_cache_probe_host"
CONTRACT="$BUILD_DIR/aicpu_aicore_cache_contract"

compile_aicore() {
    local source="$1"
    local object="$2"
    local binary="$3"
    local entry="$4"
    "$CCEC" \
        -c -O3 -g -x cce -Wall -std=c++17 \
        --cce-aicore-only \
        --cce-aicore-arch=dav-c310-vec \
        -mllvm -cce-aicore-stack-size=0x8000 \
        -mllvm -cce-aicore-function-stack-size=0x8000 \
        -mllvm -cce-aicore-record-overflow=false \
        -mllvm -cce-aicore-addr-transform \
        -mllvm -cce-aicore-dcci-insert-for-scalar=false \
        -mllvm -cce-aicore-dcci-before-kernel-end=false \
        -I"$SCRIPT_DIR" \
        -I"$PTO_INCLUDE_ROOT/include" \
        -o "$object" \
        "$source"
    "$LD" -m aicorelinux -Ttext=0 -static -o "$binary" "$object"
    local symbols sections
    symbols="$("$READELF_BIN" --symbols --wide "$binary")"
    sections="$("$READELF_BIN" --sections --wide "$binary")"
    if [[ "$symbols" != *" $entry"* || "$sections" != *".ascend.meta.$entry"* ]]; then
        echo "Missing AIV entry or metadata for $entry" >&2
        exit 1
    fi
}

build_probe() {
    mkdir -p "$BUILD_DIR"
    echo "[BUILD] host layout/matrix contract"
    "$CXX_BIN" -O2 -g -std=c++17 -Wall -Wextra -Werror \
        -I"$SCRIPT_DIR" \
        "$SCRIPT_DIR/contract_test.cpp" \
        -o "$CONTRACT"
    "$CONTRACT"

    echo "[BUILD] AICPU -> AICore consumer"
    compile_aicore \
        "$SCRIPT_DIR/aicpu_to_aicore.cpp" \
        "$BUILD_DIR/aicpu_to_aicore.o" \
        "$AICPU_TO_AICORE_KERNEL" \
        "aicpu_to_aicore_cache_probe_0_mix_aiv"

    echo "[BUILD] AICore -> AICPU producer"
    compile_aicore \
        "$SCRIPT_DIR/aicore_to_aicpu.cpp" \
        "$BUILD_DIR/aicore_to_aicpu.o" \
        "$AICORE_TO_AICPU_KERNEL" \
        "aicore_to_aicpu_cache_probe_0_mix_aiv"

    echo "[BUILD] reusable repository AICPU bootstrap dispatcher"
    "$HCC" -shared -fPIC -O3 -g -std=gnu++17 -Wall -Wextra -Werror \
        -Wl,--build-id \
        -I"$REPO_ROOT/src/common" \
        "$REPO_ROOT/src/common/aicpu_loader/device/aicpu_dispatcher.cpp" \
        -ldl \
        -o "$DISPATCHER"

    echo "[BUILD] AICPU cache/atomic participant"
    "$HCC" -shared -fPIC -O3 -g -std=gnu++17 -Wall -Wextra -Werror \
        -Wl,--build-id \
        -I"$SCRIPT_DIR" \
        -I"$REPO_ROOT/src/a5/platform/include" \
        "$SCRIPT_DIR/aicpu_kernel.cpp" \
        -o "$OWNER"

    echo "[BUILD] host runner using src/common/aicpu_loader"
    "$CXX_BIN" -O2 -g -std=c++17 -Wall -Wextra -Werror -Wno-deprecated-declarations \
        -I"$SCRIPT_DIR" \
        -I"$REPO_ROOT/tests/atomic_probe" \
        -I"$REPO_ROOT/src/common/aicpu_loader/host" \
        -I"$REPO_ROOT/src/a5/platform/include" \
        -I"$REPO_ROOT/src/common" \
        -I"$REPO_ROOT/src/common/log/include" \
        -I"$ASCEND_HOME_PATH/include" \
        -I"$ASCEND_HOME_PATH/pkg_inc" \
        -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
        -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
        "$SCRIPT_DIR/host.cpp" \
        "$REPO_ROOT/src/common/aicpu_loader/host/load_aicpu_op.cpp" \
        "$REPO_ROOT/src/common/log/host_log.cpp" \
        "$REPO_ROOT/src/common/log/unified_log_host.cpp" \
        -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -lascendcl -lruntime -ldl -pthread \
        -o "$HOST"
    echo "[BUILD] complete: $BUILD_DIR"
}

run_probe() {
    for artifact in \
        "$AICPU_TO_AICORE_KERNEL" "$AICORE_TO_AICPU_KERNEL" \
        "$DISPATCHER" "$OWNER" "$HOST"; do
        if [[ ! -e "$artifact" ]]; then
            echo "Missing build artifact: $artifact; run '$0 build' first." >&2
            exit 1
        fi
    done
    echo "[RUN] device=${ATOMIC_PROBE_DEVICE:-${TASK_DEVICE:-0}}"
    timeout 180 \
        "$HOST" "$AICPU_TO_AICORE_KERNEL" "$AICORE_TO_AICPU_KERNEL" \
        "$DISPATCHER" "$OWNER"
}

if [[ "$ACTION" == "all" || "$ACTION" == "build" ]]; then
    build_probe
fi
if [[ "$ACTION" == "all" || "$ACTION" == "run" ]]; then
    run_probe
fi
