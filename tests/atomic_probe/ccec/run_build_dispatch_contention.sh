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
BUILD_DIR="$SCRIPT_DIR/build/build_dispatch_contention"
ACTION="${1:-all}"
if [[ "$ACTION" != "all" && "$ACTION" != "build" && "$ACTION" != "run" ]]; then
    echo "Usage: $0 [all|build|run]" >&2
    exit 1
fi

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set; source the CANN 9.1 environment first." >&2
    exit 1
fi
if [[ -z "${PTO_ISA_ROOT:-}" || ! -f "$PTO_ISA_ROOT/include/pto/common/kernel_meta.hpp" ]]; then
    echo "PTO_ISA_ROOT must contain include/pto/common/kernel_meta.hpp." >&2
    exit 1
fi

CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD="$ASCEND_HOME_PATH/bin/ld.lld"
READELF_BIN="${READELF:-readelf}"
mkdir -p "$BUILD_DIR"

COMMON_FLAGS=(
    -c -O3 -g -x cce -Wall -std=c++17
    --cce-aicore-only
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false
    -I"$SCRIPT_DIR"
    -I"$PTO_ISA_ROOT/include"
)

if [[ "$ACTION" != "run" ]]; then
    echo "[BUILD] mixed AIC dispatch contention entry"
    "$CCEC" "${COMMON_FLAGS[@]}" --cce-aicore-arch=dav-c310-cube -DPA_BUILD_AIC \
        -o "$BUILD_DIR/build_dispatch_contention_aic.o" \
        "$SCRIPT_DIR/build_dispatch_contention.cpp"

    echo "[BUILD] mixed AIV dispatch contention entry"
    "$CCEC" "${COMMON_FLAGS[@]}" --cce-aicore-arch=dav-c310-vec -DPA_BUILD_AIV \
        -o "$BUILD_DIR/build_dispatch_contention_aiv.o" \
        "$SCRIPT_DIR/build_dispatch_contention.cpp"

    echo "[BUILD] static 1:2 mixed dispatch contention ELF"
    "$LD" -m aicorelinux -Ttext=0 -static \
        -o "$BUILD_DIR/build_dispatch_contention_kernel.o" \
        "$BUILD_DIR/build_dispatch_contention_aic.o" \
        "$BUILD_DIR/build_dispatch_contention_aiv.o"

    symbol_table="$("$READELF_BIN" --symbols --wide "$BUILD_DIR/build_dispatch_contention_kernel.o")"
    section_table="$("$READELF_BIN" --sections --wide "$BUILD_DIR/build_dispatch_contention_kernel.o")"
    for entry in build_dispatch_contention_0_mix_aic build_dispatch_contention_0_mix_aiv; do
        if ! awk -v name="$entry" \
            '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
             END {exit count != 1}' <<<"$symbol_table"; then
            echo "Missing unique non-empty mixed entry: $entry" >&2
            exit 1
        fi
        if [[ "$section_table" != *".ascend.meta.$entry"* ]]; then
            echo "Missing mixed metadata section for $entry" >&2
            exit 1
        fi
    done
    if [[ -n "$("$READELF_BIN" --relocs --wide "$BUILD_DIR/build_dispatch_contention_kernel.o" | sed -n '/Relocation section/p')" ]]; then
        echo "Mixed dispatch contention ELF must not retain relocations." >&2
        exit 1
    fi
    echo "[CHECK] mixed entries, metadata and relocation closure PASS"

    echo "[BUILD] mixed dispatch contention host"
    g++ -O2 -std=c++17 -Wall -Wextra -Werror -Wno-deprecated-declarations \
        -I"$SCRIPT_DIR" \
        -I"$ASCEND_HOME_PATH/include" \
        -I"$ASCEND_HOME_PATH/pkg_inc" \
        -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
        -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
        "$SCRIPT_DIR/build_dispatch_contention_host.cpp" \
        -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -lascendcl -lruntime \
        -o "$BUILD_DIR/build_dispatch_contention_host"
fi

if [[ "$ACTION" != "build" ]]; then
    echo "[RUN] device=${ATOMIC_PROBE_DEVICE:-${TASK_DEVICE:-0}}"
    "$BUILD_DIR/build_dispatch_contention_host" \
        "$BUILD_DIR/build_dispatch_contention_kernel.o"
fi
