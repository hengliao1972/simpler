#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the LICENSE file for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the license.
# -----------------------------------------------------------------------------------------------------------

# Build and run the isolated PA-shared cache-preload model.
#
# Usage:
#   ./run_shared_preload_model.sh
#   ./run_shared_preload_model.sh build
#   ./run_shared_preload_model.sh run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/shared_preload_model"
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
KERNEL="$BUILD_DIR/shared_preload_model_kernel.o"
HOST="$BUILD_DIR/shared_preload_model_host"

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

symbol_address_hex=""
symbol_size=""
read_unique_function_symbol() {
    local elf="$1"
    local name="$2"
    local -a matches=()
    mapfile -t matches < <(
        "$READELF_BIN" --symbols --wide "$elf" |
            awk -v symbol="$name" \
                '$4 == "FUNC" && $7 != "UND" && $8 == symbol {print $2 " " $3}'
    )
    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "Expected exactly one defined FUNC named $name; found ${#matches[@]}" >&2
        exit 1
    fi
    read -r symbol_address_hex symbol_size <<<"${matches[0]}"
    if [[ ! "$symbol_address_hex" =~ ^[0-9a-fA-F]+$ ||
          ! "$symbol_size" =~ ^[0-9]+$ ]]; then
        echo "Cannot parse ELF symbol $name: ${matches[0]}" >&2
        exit 1
    fi
}

check_kernel_elf() {
    if [[ ! -f "$KERNEL" ]]; then
        echo "Kernel ELF is missing: $KERNEL" >&2
        exit 1
    fi
    local symbols sections
    symbols="$("$READELF_BIN" --symbols --wide "$KERNEL")"
    sections="$("$READELF_BIN" --sections --wide "$KERNEL")"
    for entry in shared_preload_model_0_mix_aic shared_preload_model_0_mix_aiv; do
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

    read_unique_function_symbol "$KERNEL" \
        "shared_preload_model_icache_target"
    local target_address_hex="$symbol_address_hex"
    local target_size="$symbol_size"
    read_unique_function_symbol "$KERNEL" \
        "shared_preload_model_icache_evictor"
    local evictor_address_hex="$symbol_address_hex"
    local evictor_size="$symbol_size"
    read_unique_function_symbol "$KERNEL" \
        "shared_preload_model_icache_caller"
    local caller_address_hex="$symbol_address_hex"
    local caller_size="$symbol_size"

    local target_address=$((16#$target_address_hex))
    local evictor_address=$((16#$evictor_address_hex))
    local caller_address=$((16#$caller_address_hex))
    local target_end=$((target_address + target_size))
    local evictor_end=$((evictor_address + evictor_size))
    local caller_end=$((caller_address + caller_size))
    local caller_forward_window_end=$((caller_address + 4096))

    if ((target_size < 4096)); then
        echo "ICache target is smaller than 4096B: $target_size" >&2
        exit 1
    fi
    if ((evictor_size < 32768)); then
        echo "ICache evictor is smaller than 32768B: $evictor_size" >&2
        exit 1
    fi
    if ((target_address % 128 != 0 ||
         evictor_address % 128 != 0 ||
         caller_address % 128 != 0)); then
        printf 'ICache symbols are not 128B aligned: target=0x%x evictor=0x%x caller=0x%x\n' \
            "$target_address" "$evictor_address" "$caller_address" >&2
        exit 1
    fi
    if ! ((target_end <= evictor_address || evictor_end <= target_address)); then
        echo "ICache target and evictor overlap" >&2
        exit 1
    fi
    if ! ((target_end <= caller_address || caller_end <= target_address)); then
        echo "ICache target and caller overlap" >&2
        exit 1
    fi
    if ! ((target_end <= caller_address ||
           target_address >= caller_forward_window_end)); then
        printf 'Target overlaps caller current-PC 4096B window: target=[0x%x,0x%x) caller=[0x%x,0x%x)\n' \
            "$target_address" "$target_end" "$caller_address" \
            "$caller_forward_window_end" >&2
        exit 1
    fi

    printf '[CHECK] mixed entries/meta PASS; target=0x%x/%uB; evictor=0x%x/%uB; caller=0x%x/%uB; caller-target-window=disjoint\n' \
        "$target_address" "$target_size" "$evictor_address" \
        "$evictor_size" "$caller_address" "$caller_size"
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

    echo "[BUILD] CCEC AIC shared preload model"
    "$CCEC" "${common_flags[@]}" \
        --cce-aicore-arch=dav-c310-cube \
        -DSHARED_PRELOAD_MODEL_BUILD_AIC \
        -o "$BUILD_DIR/shared_preload_model_aic.o" \
        "$SCRIPT_DIR/shared_preload_model.cpp"
    echo "[BUILD] CCEC AIV shared preload model"
    "$CCEC" "${common_flags[@]}" \
        --cce-aicore-arch=dav-c310-vec \
        -DSHARED_PRELOAD_MODEL_BUILD_AIV \
        -o "$BUILD_DIR/shared_preload_model_aiv.o" \
        "$SCRIPT_DIR/shared_preload_model.cpp"
    echo "[BUILD] Static 1:2 mixed AICore ELF"
    "$LD" -m aicorelinux -Ttext=0 -static \
        -o "$KERNEL" \
        "$BUILD_DIR/shared_preload_model_aic.o" \
        "$BUILD_DIR/shared_preload_model_aiv.o"
    check_kernel_elf

    echo "[BUILD] host runner"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
        -I"$ASCEND_HOME_PATH/include" \
        "$SCRIPT_DIR/shared_preload_model_host.cpp" \
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
    timeout "${SHARED_PRELOAD_MODEL_TIMEOUT:-180}" \
        "$HOST" "$KERNEL"
}

echo "=== CCEC PA-shared Cache Preload Model ==="
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
