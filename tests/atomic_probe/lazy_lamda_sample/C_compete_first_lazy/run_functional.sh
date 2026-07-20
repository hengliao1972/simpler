#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/compete-first-lazy"
OUTPUT="$SCRIPT_DIR/output/functional.txt"

if [[ ! -x "$BUILD_DIR/pa_scheduler_host" || ! -f "$BUILD_DIR/pa_scheduler_kernel.o" ]]; then
    echo "Missing clean build; run ./build.sh first." >&2
    exit 1
fi
"$BUILD_DIR/pa_scheduler_host" \
    --kernel "$BUILD_DIR/pa_scheduler_kernel.o" \
    --device 0 --batches 1 --runs 1 \
    --winner-workload real-compute --pmu-window off --no-swimlane | tee "$OUTPUT"
if ! grep -Eq '^\[SUMMARY\].*completed_runs=1.*execution_status=PASS semantic_status=PASS postprocess_status=PASS$' "$OUTPUT"; then
    echo "Functional oracle failed for compete-first-lazy" >&2
    exit 1
fi
if ! grep -Fq '[LAZY_SAMPLE_FRONTEND] shape=compete-first-lazy views=97/97 tensor_args=1162/1162 scalar_args=9/9 resets=384/384' "$OUTPUT"; then
    echo "Lazy frontend-count oracle failed for compete-first-lazy" >&2
    exit 1
fi
