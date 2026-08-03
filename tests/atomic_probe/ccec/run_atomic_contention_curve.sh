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
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
BUILD_DIR="$SCRIPT_DIR/build/atomic_contention_curve"
RESULT_DIR="$SCRIPT_DIR/../results/atomic_contention_curve"
ACTION="${1:-all}"
if [[ "$ACTION" != "all" && "$ACTION" != "build" && "$ACTION" != "run" && "$ACTION" != "analyze" ]]; then
    echo "Usage: $0 [all|build|run|analyze]" >&2
    exit 1
fi
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [all|build|run|analyze]" >&2
    exit 1
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set; source the local CANN environment first." >&2
    exit 1
fi

CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD="$ASCEND_HOME_PATH/bin/ld.lld"
READELF_BIN="${READELF:-readelf}"
PTO_INCLUDE_ROOT="${PTO_ISA_ROOT:-$ASCEND_HOME_PATH/x86_64-linux}"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
GCC15_ROOT="${GCC15_ROOT:-${HOME}/.local/gcc-15/root}"
if [[ -n "${CXX:-}" ]]; then
    CXX_BIN="$CXX"
elif [[ -x "$GCC15_ROOT/usr/bin/g++-15" ]]; then
    CXX_BIN="$GCC15_ROOT/usr/bin/g++-15"
    export LD_LIBRARY_PATH="$GCC15_ROOT/usr/lib/x86_64-linux-gnu:$GCC15_ROOT/usr/lib/gcc/x86_64-linux-gnu/15${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    CXX_BIN="g++"
fi

for executable in "$CCEC" "$LD" "$CXX_BIN" "$READELF_BIN" "$PYTHON_BIN"; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        echo "Required executable is missing: $executable" >&2
        exit 1
    fi
done
if [[ ! -f "$PTO_INCLUDE_ROOT/include/pto/common/kernel_meta.hpp" ]]; then
    echo "PTO metadata header is missing under $PTO_INCLUDE_ROOT/include." >&2
    exit 1
fi

MIXED_KERNEL="$BUILD_DIR/atomic_contention_curve_mixed_kernel.o"
AIC_KERNEL="$BUILD_DIR/atomic_contention_curve_aic_kernel.o"
AIV_KERNEL="$BUILD_DIR/atomic_contention_curve_aiv_kernel.o"
HOST="$BUILD_DIR/atomic_contention_curve_host"
RAW="$BUILD_DIR/raw_samples.tsv"
RAW_LOCALITY="$BUILD_DIR/raw_locality.tsv"
RAW_CURVE_BUNDLES=(
    "$RESULT_DIR/raw_aic_gm0.tsv.xz"
    "$RESULT_DIR/raw_aic_gm1.tsv.xz"
    "$RESULT_DIR/raw_aiv_gm0.tsv.xz"
    "$RESULT_DIR/raw_aiv_gm1.tsv.xz"
    "$RESULT_DIR/raw_mixed_gm0.tsv.xz"
    "$RESULT_DIR/raw_mixed_gm1.tsv.xz"
)
RAW_LOCALITY_BUNDLE="$RESULT_DIR/raw_locality.tsv.xz"
SUMMARY="$RESULT_DIR/summary.json"
SVG="$RESULT_DIR/curve.svg"
REPORT="$RESULT_DIR/report.md"

check_entry() {
    local elf="$1"
    local entry="$2"
    local symbols sections
    symbols="$("$READELF_BIN" --symbols --wide "$elf")"
    sections="$("$READELF_BIN" --sections --wide "$elf")"
    if ! awk -v name="$entry" \
        '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
         END {exit count != 1}' <<<"$symbols"; then
        echo "Missing unique non-empty kernel entry: $entry in $elf" >&2
        exit 1
    fi
    if [[ "$sections" != *".ascend.meta.$entry"* ]]; then
        echo "Missing kernel metadata section: $entry in $elf" >&2
        exit 1
    fi
}

check_closed_elf() {
    local elf="$1"
    if [[ -n "$("$READELF_BIN" --relocs --wide "$elf" | sed -n '/Relocation section/p')" ]]; then
        echo "Kernel ELF retains relocations: $elf" >&2
        exit 1
    fi
}

build_probe() {
    mkdir -p "$BUILD_DIR"
    local common_flags=(
        -c -O3 -g -x cce -Wall -std=c++17
        --cce-aicore-only
        -mllvm -cce-aicore-record-overflow=false
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-dcci-before-kernel-end=false
        -I"$SCRIPT_DIR"
        -I"$PTO_INCLUDE_ROOT/include"
    )

    echo "[BUILD] mixed 1C:2V atomic contention entries"
    "$CCEC" "${common_flags[@]}" --cce-aicore-arch=dav-c310-cube \
        -DATOMIC_CURVE_BUILD_MIXED_AIC \
        -o "$BUILD_DIR/atomic_contention_curve_mixed_aic.o" \
        "$SCRIPT_DIR/atomic_contention_curve.cpp"
    "$CCEC" "${common_flags[@]}" --cce-aicore-arch=dav-c310-vec \
        -DATOMIC_CURVE_BUILD_MIXED_AIV \
        -o "$BUILD_DIR/atomic_contention_curve_mixed_aiv.o" \
        "$SCRIPT_DIR/atomic_contention_curve.cpp"
    "$LD" -m aicorelinux -Ttext=0 -static \
        -o "$MIXED_KERNEL" \
        "$BUILD_DIR/atomic_contention_curve_mixed_aic.o" \
        "$BUILD_DIR/atomic_contention_curve_mixed_aiv.o"

    echo "[BUILD] pure AIC atomic contention entry"
    "$CCEC" "${common_flags[@]}" --cce-aicore-arch=dav-c310-cube \
        -DATOMIC_CURVE_BUILD_AIC \
        -o "$BUILD_DIR/atomic_contention_curve_aic.o" \
        "$SCRIPT_DIR/atomic_contention_curve.cpp"
    "$LD" -m aicorelinux -Ttext=0 -static \
        -o "$AIC_KERNEL" "$BUILD_DIR/atomic_contention_curve_aic.o"

    echo "[BUILD] pure AIV atomic contention entry"
    "$CCEC" "${common_flags[@]}" --cce-aicore-arch=dav-c310-vec \
        -DATOMIC_CURVE_BUILD_AIV \
        -o "$BUILD_DIR/atomic_contention_curve_aiv.o" \
        "$SCRIPT_DIR/atomic_contention_curve.cpp"
    "$LD" -m aicorelinux -Ttext=0 -static \
        -o "$AIV_KERNEL" "$BUILD_DIR/atomic_contention_curve_aiv.o"

    check_entry "$MIXED_KERNEL" atomic_contention_curve_0_mix_aic
    check_entry "$MIXED_KERNEL" atomic_contention_curve_0_mix_aiv
    check_entry "$AIC_KERNEL" atomic_contention_curve_aic_0_mix_aic
    check_entry "$AIV_KERNEL" atomic_contention_curve_aiv_0_mix_aiv
    check_closed_elf "$MIXED_KERNEL"
    check_closed_elf "$AIC_KERNEL"
    check_closed_elf "$AIV_KERNEL"
    echo "[CHECK] topology entries, metadata, and relocation closure PASS"

    echo "[BUILD] atomic contention curve host"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror -Wno-deprecated-declarations \
        -I"$SCRIPT_DIR" \
        -I"$ASCEND_HOME_PATH/include" \
        -I"$ASCEND_HOME_PATH/pkg_inc" \
        -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
        -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
        "$SCRIPT_DIR/atomic_contention_curve_host.cpp" \
        -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -lascendcl -lruntime \
        -o "$HOST"
    echo "[BUILD] complete: $BUILD_DIR"
}

run_probe() {
    for artifact in "$MIXED_KERNEL" "$AIC_KERNEL" "$AIV_KERNEL" "$HOST"; do
        if [[ ! -s "$artifact" ]]; then
            echo "Missing build artifact: $artifact" >&2
            exit 1
        fi
    done
    mkdir -p "$RESULT_DIR"
    echo "[RUN] device=${ATOMIC_PROBE_DEVICE:-${TASK_DEVICE:-0}}"
    "$HOST" "$MIXED_KERNEL" "$AIC_KERNEL" "$AIV_KERNEL" "$RAW" "$RAW_LOCALITY"
}

analyze_probe() {
    local raw_inputs=()
    local bundle_args=()
    if [[ -s "$RAW" && -s "$RAW_LOCALITY" ]]; then
        raw_inputs=("$RAW" "$RAW_LOCALITY")
        bundle_args=(--raw-bundle-dir "$RESULT_DIR")
    else
        for raw_part in "${RAW_CURVE_BUNDLES[@]}"; do
            if [[ ! -s "$raw_part" ]]; then
                echo "Raw result input is missing: $raw_part" >&2
                exit 1
            fi
            raw_inputs+=("$raw_part")
        done
        if [[ ! -s "$RAW_LOCALITY_BUNDLE" ]]; then
            echo "Raw locality input is missing: $RAW_LOCALITY_BUNDLE" >&2
            exit 1
        fi
        raw_inputs+=("$RAW_LOCALITY_BUNDLE")
    fi
    local source_revision
    source_revision="$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD)"
    if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
        source_revision="${source_revision}+dirty"
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/../analyze_atomic_contention_curve.py" \
        "${raw_inputs[@]}" \
        --summary "$SUMMARY" \
        --svg "$SVG" \
        --report "$REPORT" \
        --source-revision "$source_revision" \
        "${bundle_args[@]}"
    echo "[RESULT] $SUMMARY"
    echo "[RESULT] $SVG"
    echo "[RESULT] $REPORT"
}

case "$ACTION" in
    all)
        build_probe
        run_probe
        analyze_probe
        ;;
    build)
        build_probe
        ;;
    run)
        run_probe
        analyze_probe
        ;;
    analyze)
        analyze_probe
        ;;
esac
