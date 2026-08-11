#!/usr/bin/env bash
# CPU mapping gate for the narrow SIMT builder's production-state adapter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE="$MODE_DIR/test/test_simt_real_state_runtime.cpp"
BUILD_DIR="$SCRIPT_DIR/build/real_state_runtime"
CXX="${CXX:-g++}"

mkdir -p "$BUILD_DIR"

COMMON_FLAGS=(
    -std=c++17
    -Wall
    -Wextra
    -Werror
    -Wno-unused-function
    -pthread
)

echo "[BUILD] O2 real-state mapping gate"
"$CXX" "${COMMON_FLAGS[@]}" -O2 \
    "$SOURCE" -o "$BUILD_DIR/test_o2"
"$BUILD_DIR/test_o2"

echo "[BUILD] ASan real-state mapping gate"
"$CXX" "${COMMON_FLAGS[@]}" -O1 -g \
    -fsanitize=address -fno-omit-frame-pointer \
    "$SOURCE" -o "$BUILD_DIR/test_asan"
ASAN_OPTIONS=detect_leaks=0 "$BUILD_DIR/test_asan"

echo "[BUILD] UBSan real-state mapping gate"
"$CXX" "${COMMON_FLAGS[@]}" -O1 -g \
    -fsanitize=undefined -fno-sanitize-recover=all \
    "$SOURCE" -o "$BUILD_DIR/test_ubsan"
"$BUILD_DIR/test_ubsan"

echo "[PASS] CPU adapter maps SchedulerState without a mirror sidecar"
