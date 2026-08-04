#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the LICENSE file in the root directory of this source tree for more details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
BUILD_DIR="$SCRIPT_DIR/build/atomic_contention_curve"
RESULT_DIR="$SCRIPT_DIR/../results/atomic_bank_conflict"
ACTION="${1:-all}"
if [[ "$ACTION" != "all" && "$ACTION" != "build" && "$ACTION" != "run" && "$ACTION" != "analyze" ]]; then
    echo "Usage: $0 [all|build|run|analyze]" >&2
    exit 1
fi
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [all|build|run|analyze]" >&2
    exit 1
fi

MIXED_KERNEL="$BUILD_DIR/atomic_contention_curve_mixed_kernel.o"
AIC_KERNEL="$BUILD_DIR/atomic_contention_curve_aic_kernel.o"
AIV_KERNEL="$BUILD_DIR/atomic_contention_curve_aiv_kernel.o"
HOST="$BUILD_DIR/atomic_contention_curve_host"
RAW="$BUILD_DIR/raw_bank_conflict.tsv"
RAW_BUNDLE="$RESULT_DIR/raw_samples.tsv.xz"
SUMMARY="$RESULT_DIR/summary.json"
SVG="$RESULT_DIR/curve.svg"
REPORT="$RESULT_DIR/report.md"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

build_probe() {
    "$SCRIPT_DIR/run_atomic_contention_curve.sh" build
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
    "$HOST" "$MIXED_KERNEL" "$AIC_KERNEL" "$AIV_KERNEL" --bank "$RAW"
}

analyze_probe() {
    local raw_input="$RAW"
    local bundle_args=(--raw-bundle "$RAW_BUNDLE")
    if [[ ! -s "$raw_input" ]]; then
        raw_input="$RAW_BUNDLE"
        bundle_args=()
    fi
    if [[ ! -s "$raw_input" ]]; then
        echo "Raw result input is missing: $RAW or $RAW_BUNDLE" >&2
        exit 1
    fi
    local source_revision
    source_revision="$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD)"
    if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
        source_revision="${source_revision}+dirty"
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/../analyze_atomic_bank_conflict.py" \
        "$raw_input" \
        --summary "$SUMMARY" \
        --svg "$SVG" \
        --report "$REPORT" \
        --source-revision "$source_revision" \
        "${bundle_args[@]}"
    echo "[RESULT] $SUMMARY"
    echo "[RESULT] $SVG"
    echo "[RESULT] $REPORT"
    echo "[RESULT] $RAW_BUNDLE"
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
