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
UBUF_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SIMT_ROOT="$(cd "$UBUF_ROOT/.." && pwd)"
BUILD_DIR="$UBUF_ROOT/build/cpu"
SOURCE="$UBUF_ROOT/test/test_u1_multi_slot.cpp"
CXX_BIN="${CXX:-g++}"

mkdir -p "$BUILD_DIR"

COMMON_FLAGS=(
    -std=c++17 -pthread -Wall -Wextra -Werror
    -I"$UBUF_ROOT/common"
    -I"$SIMT_ROOT/common"
)

echo "[BUILD] U1 CPU four-slot optimized"
"$CXX_BIN" -O2 "${COMMON_FLAGS[@]}" "$SOURCE" -o "$BUILD_DIR/test_u1_multi_slot"
echo "[TEST] U1 CPU four-slot optimized"
timeout --foreground 180s "$BUILD_DIR/test_u1_multi_slot" --rounds 8

echo "[BUILD] U1 CPU four-slot ASan+UBSan"
"$CXX_BIN" -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined \
    "${COMMON_FLAGS[@]}" "$SOURCE" -o "$BUILD_DIR/test_u1_multi_slot_asan_ubsan"
echo "[TEST] U1 CPU four-slot ASan+UBSan"
timeout --foreground 240s "$BUILD_DIR/test_u1_multi_slot_asan_ubsan" --rounds 2

echo "[BUILD] U1 CPU four-slot TSan"
"$CXX_BIN" -O1 -g -fno-omit-frame-pointer -fsanitize=thread \
    "${COMMON_FLAGS[@]}" "$SOURCE" -o "$BUILD_DIR/test_u1_multi_slot_tsan"
echo "[TEST] U1 CPU four-slot TSan"
timeout --foreground 300s "$BUILD_DIR/test_u1_multi_slot_tsan" --rounds 2
