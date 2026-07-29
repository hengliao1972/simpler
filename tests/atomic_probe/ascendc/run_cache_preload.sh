#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Build and run the AscendC counterpart of the cache-preload probe. The
# dedicated entry point adds final embedded-AICore ELF size/alignment checks
# that the generic _run_asc_probe.sh intentionally does not perform.
#
# Usage:
#   ./run_cache_preload.sh
#   ./run_cache_preload.sh build
#   ./run_cache_preload.sh run
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

BISHENG="${BISHENG:-$ASCEND_HOME_PATH/bin/bisheng}"
OBJCOPY_BIN="${OBJCOPY:-objcopy}"
READELF_BIN="${READELF:-readelf}"
SOURCE="$SCRIPT_DIR/cache_preload_probe.asc"
EXECUTABLE="$BUILD_DIR/cache_preload_probe"
AICORE_ELF="$BUILD_DIR/cache_preload_aicore.o"

if [[ ! -x "$BISHENG" ]]; then
    echo "bisheng is missing: $BISHENG" >&2
    exit 1
fi
if ! command -v "$OBJCOPY_BIN" >/dev/null 2>&1; then
    echo "objcopy is required: $OBJCOPY_BIN" >&2
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

extract_and_check_aicore() {
    if [[ ! -x "$EXECUTABLE" ]]; then
        echo "AscendC executable is missing: $EXECUTABLE" >&2
        exit 1
    fi
    "$OBJCOPY_BIN" \
        --dump-section ".aicore_binary=$AICORE_ELF" \
        "$EXECUTABLE"
    if [[ ! -s "$AICORE_ELF" ]]; then
        echo "Embedded .aicore_binary extraction failed: $AICORE_ELF" >&2
        exit 1
    fi

    local symbols sections entry="cache_preload_ascendc_probe"
    symbols="$("$READELF_BIN" --symbols --wide "$AICORE_ELF")"
    sections="$("$READELF_BIN" --sections --wide "$AICORE_ELF")"
    if [[ "$symbols" != *" $entry"* ||
          "$sections" != *".ascend.meta.$entry"* ]]; then
        echo "Missing AIV entry or metadata for $entry" >&2
        exit 1
    fi

    read_unique_function_symbol \
        "$AICORE_ELF" "cache_preload_ascendc_icache_path.vector"
    local path_address_hex="$symbol_address_hex"
    local path_size="$symbol_size"
    read_unique_function_symbol \
        "$AICORE_ELF" "cache_preload_ascendc_icache_evictor.vector"
    local evictor_address_hex="$symbol_address_hex"
    local evictor_size="$symbol_size"
    read_unique_function_symbol \
        "$AICORE_ELF" "cache_preload_ascendc_gap.vector"

    local path_address=$((16#$path_address_hex))
    local evictor_address=$((16#$evictor_address_hex))
    local path_end=$((path_address + path_size))
    local evictor_end=$((evictor_address + evictor_size))
    if ((path_size < 4096)); then
        echo "AscendC current-PC path is smaller than 4096B: $path_size" >&2
        exit 1
    fi
    if ((evictor_size < 32768)); then
        echo "AscendC ICache evictor is smaller than 32768B: $evictor_size" >&2
        exit 1
    fi
    if ((path_address % 128 != 0 || evictor_address % 128 != 0)); then
        printf 'AscendC ICache symbols are not 128B-aligned: path=0x%x evictor=0x%x\n' \
            "$path_address" "$evictor_address" >&2
        exit 1
    fi
    if ! ((path_end <= evictor_address || evictor_end <= path_address)); then
        printf 'AscendC path/evictor overlap: path=[0x%x,0x%x) evictor=[0x%x,0x%x)\n' \
            "$path_address" "$path_end" "$evictor_address" "$evictor_end" >&2
        exit 1
    fi
    printf '[CHECK] AscendC AIV entry/meta PASS; current-PC path=0x%x/%uB; ' \
        "$path_address" "$path_size"
    printf 'evictor=0x%x/%uB; alignment=128B ranges=non-overlap\n' \
        "$evictor_address" "$evictor_size"
}

build_probe() {
    mkdir -p "$BUILD_DIR"
    echo "[BUILD] AscendC cache preload probe (dav-3510)"
    "$BISHENG" -xasc "$SOURCE" --npu-arch=dav-3510 \
        -o "$EXECUTABLE"
    extract_and_check_aicore
    echo "[BUILD] complete: $BUILD_DIR"
}

run_probe() {
    extract_and_check_aicore
    echo "[RUN] device=${ATOMIC_PROBE_DEVICE:-${TASK_DEVICE:-0}}"
    timeout "${CACHE_PRELOAD_TIMEOUT:-120}" "$EXECUTABLE"
}

echo "=== AscendC Cache Preload Probe ==="
echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git_head=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "ascend_home=$ASCEND_HOME_PATH"
echo "bisheng=$($BISHENG --version 2>&1 | head -n 1)"
echo "action=$ACTION"

if [[ "$ACTION" == "all" || "$ACTION" == "build" ]]; then
    build_probe
fi
if [[ "$ACTION" == "all" || "$ACTION" == "run" ]]; then
    run_probe
fi
