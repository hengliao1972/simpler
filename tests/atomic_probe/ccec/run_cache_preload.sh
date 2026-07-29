#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Build and run the pure-CCEC A5 cache-preload reference probe without
# changing ccec/run_all.sh.
#
# Usage:
#   ./run_cache_preload.sh        # build, linked-ELF checks, and A5 run
#   ./run_cache_preload.sh build  # build and linked-ELF checks only
#   ./run_cache_preload.sh run    # recheck existing ELF and run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/cache_preload"
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
    echo "ASCEND_HOME_PATH is not set; source the local CANN environment first." >&2
    exit 1
fi

CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD="$ASCEND_HOME_PATH/bin/ld.lld"
CXX_BIN="${CXX:-g++}"
READELF_BIN="${READELF:-readelf}"
PTO_INCLUDE_ROOT="${PTO_ISA_ROOT:-$ASCEND_HOME_PATH/x86_64-linux}"
KERNEL="$BUILD_DIR/cache_preload_kernel.o"
HOST="$BUILD_DIR/cache_preload_host"

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
            awk -v symbol="$name" '$4 == "FUNC" && $8 == symbol {print $2 " " $3}'
    )
    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "Expected exactly one FUNC symbol named $name; found ${#matches[@]}" >&2
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
    local symbols sections entry="cache_preload_0_mix_aiv"
    symbols="$("$READELF_BIN" --symbols --wide "$KERNEL")"
    sections="$("$READELF_BIN" --sections --wide "$KERNEL")"
    if [[ "$symbols" != *" $entry"* ||
          "$sections" != *".ascend.meta.$entry"* ]]; then
        echo "Missing AIV entry or metadata for $entry" >&2
        exit 1
    fi

    read_unique_function_symbol "$KERNEL" "cache_preload_icache_path"
    local path_address_hex="$symbol_address_hex"
    local path_size="$symbol_size"
    read_unique_function_symbol "$KERNEL" "cache_preload_icache_evictor"
    local evictor_address_hex="$symbol_address_hex"
    local evictor_size="$symbol_size"
    read_unique_function_symbol "$KERNEL" "cache_preload_gap"

    local path_address=$((16#$path_address_hex))
    local evictor_address=$((16#$evictor_address_hex))
    local path_end=$((path_address + path_size))
    local evictor_end=$((evictor_address + evictor_size))
    if ((path_size < 4096)); then
        echo "current-PC path is smaller than 4096B: $path_size" >&2
        exit 1
    fi
    if ((evictor_size < 32768)); then
        echo "ICache evictor is smaller than 32768B: $evictor_size" >&2
        exit 1
    fi
    if ((path_address % 128 != 0 || evictor_address % 128 != 0)); then
        printf 'ICache symbols are not 128B-aligned: path=0x%x evictor=0x%x\n' \
            "$path_address" "$evictor_address" >&2
        exit 1
    fi
    if ! ((path_end <= evictor_address || evictor_end <= path_address)); then
        printf 'path/evictor overlap: path=[0x%x,0x%x) evictor=[0x%x,0x%x)\n' \
            "$path_address" "$path_end" "$evictor_address" "$evictor_end" >&2
        exit 1
    fi
    printf '[CHECK] AIV entry/meta PASS; current-PC path=0x%x/%uB; evictor=0x%x/%uB; ' \
        "$path_address" "$path_size" "$evictor_address" "$evictor_size"
    echo "alignment=128B ranges=non-overlap"
}

build_probe() {
    mkdir -p "$BUILD_DIR"
    local common_flags=(
        -c -O3 -g -x cce -Wall -std=c++17
        --cce-aicore-only
        --cce-aicore-arch=dav-c310-vec
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=false
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-dcci-before-kernel-end=false
        -DCCEC_SYNC_AIV_ONLY
        -I"$SCRIPT_DIR"
        -I"$PTO_INCLUDE_ROOT/include"
    )

    echo "[BUILD] pure CCEC cache preload kernel (dav-c310-vec)"
    "$CCEC" "${common_flags[@]}" \
        -o "$BUILD_DIR/cache_preload_vec.o" \
        "$SCRIPT_DIR/cache_preload.cpp"
    "$LD" -m aicorelinux -Ttext=0 -static \
        -o "$KERNEL" "$BUILD_DIR/cache_preload_vec.o"
    check_kernel_elf

    echo "[BUILD] host runner"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
        -I"$ASCEND_HOME_PATH/include" \
        "$SCRIPT_DIR/cache_preload_host.cpp" \
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
    timeout "${CACHE_PRELOAD_TIMEOUT:-120}" "$HOST" "$KERNEL"
}

echo "=== Pure CCEC Cache Preload Probe ==="
echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git_head=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "ascend_home=$ASCEND_HOME_PATH"
echo "ccec=$($CCEC --version 2>&1 | head -n 1)"
echo "action=$ACTION"

if [[ "$ACTION" == "all" || "$ACTION" == "build" ]]; then
    build_probe
fi
if [[ "$ACTION" == "all" || "$ACTION" == "run" ]]; then
    run_probe
fi
