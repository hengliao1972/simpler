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
GM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$GM_ROOT/build/cpu"
SOURCE="$GM_ROOT/test/test_s2_cube.cpp"
CXX_BIN="${CXX:-g++}"

mkdir -p "$BUILD_DIR"

COMMON_FLAGS=(
    -std=c++17 -pthread -Wall -Wextra -Werror
    -I"$GM_ROOT/common"
)

echo "[BUILD] S2 CPU single-Cube optimized"
"$CXX_BIN" -O2 "${COMMON_FLAGS[@]}" \
    "$SOURCE" -o "$BUILD_DIR/test_s2_cube"

echo "[TEST] S2 CPU single-Cube optimized"
timeout --foreground 60s "$BUILD_DIR/test_s2_cube" --rounds 16

echo "[BUILD] S2 CPU single-Cube ASan+UBSan"
"$CXX_BIN" -O1 -g -fno-omit-frame-pointer \
    -fsanitize=address,undefined "${COMMON_FLAGS[@]}" \
    "$SOURCE" -o "$BUILD_DIR/test_s2_cube_asan_ubsan"

echo "[TEST] S2 CPU single-Cube ASan+UBSan"
timeout --foreground 90s \
    "$BUILD_DIR/test_s2_cube_asan_ubsan" --rounds 8

echo "[BUILD] S2 CPU single-Cube TSan"
"$CXX_BIN" -O1 -g -fno-omit-frame-pointer \
    -fsanitize=thread "${COMMON_FLAGS[@]}" \
    "$SOURCE" -o "$BUILD_DIR/test_s2_cube_tsan"

echo "[TEST] S2 CPU single-Cube TSan"
timeout --foreground 120s \
    "$BUILD_DIR/test_s2_cube_tsan" --rounds 4
