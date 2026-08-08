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
SIMT_ROOT="$(cd "$GM_ROOT/.." && pwd)"
BUILD_DIR="$GM_ROOT/build/cpu"
SOURCE="$GM_ROOT/test/test_g0_full_pa.cpp"
CXX_BIN="${CXX:-g++}"
BUILDER_WARP_COUNT="${SIMT_CROSS_CORE_GM_BUILDER_WARPS:-5}"
TOKEN_COUNT="${SIMT_CROSS_CORE_GM_TOKENS_PER_OWNER:-1}"
DISPATCH_WINDOW_BATCHES="${SIMT_CROSS_CORE_GM_DISPATCH_WINDOW_BATCHES:-256}"

if [[ ! "$BUILDER_WARP_COUNT" =~ ^[0-9]+$ ]] ||
   (( BUILDER_WARP_COUNT < 1 || BUILDER_WARP_COUNT > 64 )); then
    echo "SIMT_CROSS_CORE_GM_BUILDER_WARPS must be an integer in 1..64." >&2
    exit 1
fi
if [[ ! "$DISPATCH_WINDOW_BATCHES" =~ ^[0-9]+$ ]] ||
   (( DISPATCH_WINDOW_BATCHES < 1 || DISPATCH_WINDOW_BATCHES > 256 )); then
    echo "SIMT_CROSS_CORE_GM_DISPATCH_WINDOW_BATCHES must be an integer in 1..256." >&2
    exit 1
fi
if [[ ! "$TOKEN_COUNT" =~ ^[0-9]+$ ]] || (( TOKEN_COUNT < 1 || TOKEN_COUNT > 4 )); then
    echo "SIMT_CROSS_CORE_GM_TOKENS_PER_OWNER must be an integer in 1..4." >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

COMMON_FLAGS=(
    -std=c++17 -pthread -Wall -Wextra -Werror
    "-DSIMT_CROSS_CORE_G0_BUILDER_WARP_COUNT=$BUILDER_WARP_COUNT"
    "-DSIMT_CROSS_CORE_G0_TOKENS_PER_OWNER=$TOKEN_COUNT"
    "-DSIMT_CROSS_CORE_G0_DISPATCH_WINDOW_BATCHES=$DISPATCH_WINDOW_BATCHES"
    -I"$GM_ROOT/common"
    -I"$SIMT_ROOT/common"
)

echo "[BUILD] G0 CPU full PA optimized"
"$CXX_BIN" -O2 "${COMMON_FLAGS[@]}" "$SOURCE" -o "$BUILD_DIR/test_g0_full_pa"
echo "[TEST] G0 CPU full PA optimized"
timeout --foreground 90s "$BUILD_DIR/test_g0_full_pa" --rounds 4

echo "[BUILD] G0 CPU full PA ASan+UBSan"
"$CXX_BIN" -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined \
    "${COMMON_FLAGS[@]}" "$SOURCE" -o "$BUILD_DIR/test_g0_full_pa_asan_ubsan"
echo "[TEST] G0 CPU full PA ASan+UBSan"
timeout --foreground 120s "$BUILD_DIR/test_g0_full_pa_asan_ubsan" --rounds 2

echo "[BUILD] G0 CPU full PA TSan"
"$CXX_BIN" -O1 -g -fno-omit-frame-pointer -fsanitize=thread \
    "${COMMON_FLAGS[@]}" "$SOURCE" -o "$BUILD_DIR/test_g0_full_pa_tsan"
echo "[TEST] G0 CPU full PA TSan"
timeout --foreground 180s "$BUILD_DIR/test_g0_full_pa_tsan" --rounds 2
