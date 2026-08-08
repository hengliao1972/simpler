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
SOURCE="$PROBE_ROOT/test/test_warp_concurrency_cpu.cpp"
CXX_BIN="${CXX:-g++}"
BUILD_DIR="$(mktemp -d /tmp/simt-warp-concurrency-cpu.XXXXXX)"

cleanup() {
    rm -rf -- "$BUILD_DIR"
}
trap cleanup EXIT

COMMON_FLAGS=(
    -std=c++17 -pthread -Wall -Wextra -Werror -pedantic
    -I"$PROBE_ROOT/common"
)

echo "[BUILD] warp-concurrency CPU oracle optimized"
"$CXX_BIN" -O2 "${COMMON_FLAGS[@]}" \
    "$SOURCE" -o "$BUILD_DIR/test_warp_concurrency_cpu"
echo "[TEST] warp-concurrency CPU oracle optimized"
timeout --foreground 30s "$BUILD_DIR/test_warp_concurrency_cpu"

echo "[BUILD] warp-concurrency CPU oracle ASan+UBSan"
"$CXX_BIN" -O1 -g -fno-omit-frame-pointer \
    -fsanitize=address,undefined "${COMMON_FLAGS[@]}" \
    "$SOURCE" -o "$BUILD_DIR/test_warp_concurrency_cpu_asan_ubsan"
echo "[TEST] warp-concurrency CPU oracle ASan+UBSan"
timeout --foreground 45s "$BUILD_DIR/test_warp_concurrency_cpu_asan_ubsan"

echo "[BUILD] warp-concurrency CPU oracle TSan"
"$CXX_BIN" -O1 -g -fno-omit-frame-pointer \
    -fsanitize=thread "${COMMON_FLAGS[@]}" \
    "$SOURCE" -o "$BUILD_DIR/test_warp_concurrency_cpu_tsan"
echo "[TEST] warp-concurrency CPU oracle TSan"
timeout --foreground 60s "$BUILD_DIR/test_warp_concurrency_cpu_tsan"
