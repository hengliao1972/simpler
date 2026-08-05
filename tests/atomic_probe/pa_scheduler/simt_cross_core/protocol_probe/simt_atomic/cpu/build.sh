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
PROBE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SIMT_ROOT="$(cd "$PROBE_ROOT/../.." && pwd)"
BUILD_DIR="$PROBE_ROOT/build/cpu"
SOURCE="$PROBE_ROOT/test/test_simt_atomic.cpp"
CXX_BIN="${CXX:-g++}"

mkdir -p "$BUILD_DIR"

COMMON_FLAGS=(
    -std=c++17 -pthread -Wall -Wextra -Werror
    -I"$PROBE_ROOT/common"
    -I"$SIMT_ROOT/common"
)

echo "[BUILD] SIMT atomic CPU optimized"
"$CXX_BIN" -O2 "${COMMON_FLAGS[@]}" \
    "$SOURCE" -o "$BUILD_DIR/test_simt_atomic"
echo "[TEST] SIMT atomic CPU optimized"
timeout --foreground 30s "$BUILD_DIR/test_simt_atomic" --rounds 16

echo "[BUILD] SIMT atomic CPU ASan+UBSan"
"$CXX_BIN" -O1 -g -fno-omit-frame-pointer \
    -fsanitize=address,undefined "${COMMON_FLAGS[@]}" \
    "$SOURCE" -o "$BUILD_DIR/test_simt_atomic_asan_ubsan"
echo "[TEST] SIMT atomic CPU ASan+UBSan"
timeout --foreground 45s "$BUILD_DIR/test_simt_atomic_asan_ubsan" --rounds 8

echo "[BUILD] SIMT atomic CPU TSan"
"$CXX_BIN" -O1 -g -fno-omit-frame-pointer \
    -fsanitize=thread "${COMMON_FLAGS[@]}" \
    "$SOURCE" -o "$BUILD_DIR/test_simt_atomic_tsan"
echo "[TEST] SIMT atomic CPU TSan"
timeout --foreground 60s "$BUILD_DIR/test_simt_atomic_tsan" --rounds 4
