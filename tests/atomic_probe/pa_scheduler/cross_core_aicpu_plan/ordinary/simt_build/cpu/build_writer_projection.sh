#!/usr/bin/env bash
# Independent CPU gate for the Plan-driven Host writer projection.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE="$MODE_DIR/test/test_runtime_plan_writer_projection.cpp"
BUILD_DIR="$SCRIPT_DIR/build/writer_projection"
CXX="${CXX:-g++}"

mkdir -p "$BUILD_DIR"

COMMON_FLAGS=(
    -std=c++17
    -Wall
    -Wextra
    -Werror
)

echo "[BUILD] O2 runtime Plan writer-projection gate"
"$CXX" "${COMMON_FLAGS[@]}" -O2 \
    "$SOURCE" -o "$BUILD_DIR/test_o2"
"$BUILD_DIR/test_o2"

echo "[BUILD] ASan runtime Plan writer-projection gate"
"$CXX" "${COMMON_FLAGS[@]}" -O1 -g \
    -fsanitize=address -fno-omit-frame-pointer \
    "$SOURCE" -o "$BUILD_DIR/test_asan"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
    "$BUILD_DIR/test_asan"

echo "[BUILD] UBSan runtime Plan writer-projection gate"
"$CXX" "${COMMON_FLAGS[@]}" -O1 -g \
    -fsanitize=undefined -fno-sanitize-recover=all \
    -fno-omit-frame-pointer \
    "$SOURCE" -o "$BUILD_DIR/test_ubsan"
UBSAN_OPTIONS=halt_on_error=1 \
    "$BUILD_DIR/test_ubsan"

echo "[PASS] runtime Plan writer-projection CPU gates"
