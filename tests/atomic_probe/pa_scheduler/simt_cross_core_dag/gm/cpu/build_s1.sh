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
SOURCE="$GM_ROOT/test/test_s1_vector.cpp"
CXX_BIN="${CXX:-g++}"

mkdir -p "$BUILD_DIR"

COMMON_FLAGS=(
    -std=c++17 -pthread -Wall -Wextra -Werror
    -I"$GM_ROOT/common"
)

echo "[BUILD] S1 CPU single-Vector optimized"
"$CXX_BIN" -O2 "${COMMON_FLAGS[@]}" \
    "$SOURCE" -o "$BUILD_DIR/test_s1_vector"

echo "[TEST] S1 CPU single-Vector optimized"
timeout --foreground 45s "$BUILD_DIR/test_s1_vector" --rounds 64

echo "[BUILD] S1 CPU single-Vector ASan+UBSan"
"$CXX_BIN" -O1 -g -fno-omit-frame-pointer \
    -fsanitize=address,undefined "${COMMON_FLAGS[@]}" \
    "$SOURCE" -o "$BUILD_DIR/test_s1_vector_asan_ubsan"

echo "[TEST] S1 CPU single-Vector ASan+UBSan"
timeout --foreground 60s \
    "$BUILD_DIR/test_s1_vector_asan_ubsan" --rounds 32

echo "[BUILD] S1 CPU single-Vector TSan"
"$CXX_BIN" -O1 -g -fno-omit-frame-pointer \
    -fsanitize=thread "${COMMON_FLAGS[@]}" \
    "$SOURCE" -o "$BUILD_DIR/test_s1_vector_tsan"

echo "[TEST] S1 CPU single-Vector TSan"
timeout --foreground 90s \
    "$BUILD_DIR/test_s1_vector_tsan" --rounds 16
