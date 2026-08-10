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

SANITIZER_FLAGS=()
if [[ "${PA_AICPU_PLAN_SIMT_SANITIZE:-0}" == "1" ]]; then
    SANITIZER_FLAGS=(
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
fi

mkdir -p "$BUILD_DIR"
"$CXX_BIN" \
    -O2 -std=c++17 -pthread -Wall -Wextra -Werror \
    "${SANITIZER_FLAGS[@]}" \
    "$MODE_ROOT/test/test_simt_plan_build_protocol.cpp" \
    -o "$BUILD_DIR/test_simt_plan_build_protocol"

"$BUILD_DIR/test_simt_plan_build_protocol"
