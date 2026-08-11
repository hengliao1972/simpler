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
MODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$MODE_ROOT/build/cpu"
CXX_BIN="${CXX:-g++}"
SOURCE="$MODE_ROOT/test/test_narrow_simt_plan_task_builder.cpp"
COMMON_FLAGS=(
    -std=c++17
    -pthread
    -Wall
    -Wextra
    -Werror
)

mkdir -p "$BUILD_DIR"

"$CXX_BIN" \
    -O2 \
    "${COMMON_FLAGS[@]}" \
    "$SOURCE" \
    -o "$BUILD_DIR/test_narrow_simt_plan_task_builder"

timeout --foreground 30s \
    "$BUILD_DIR/test_narrow_simt_plan_task_builder"

"$CXX_BIN" \
    -O1 -g \
    "${COMMON_FLAGS[@]}" \
    -fsanitize=address,undefined \
    -fno-omit-frame-pointer \
    "$SOURCE" \
    -o "$BUILD_DIR/test_narrow_simt_plan_task_builder_sanitize"

ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
timeout --foreground 60s \
    "$BUILD_DIR/test_narrow_simt_plan_task_builder_sanitize"
