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
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/ordinary/scalar_build/cpu"
CXX_BIN="${CXX:-g++}"

SANITIZER_FLAGS=()
if [[ "${PA_AICPU_PLAN_GATE_SANITIZE:-0}" == "1" ]]; then
    SANITIZER_FLAGS=(
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
fi

mkdir -p "$BUILD_DIR"
"$CXX_BIN" \
    -O2 -std=c++17 -pthread -Wall -Wextra -Werror \
    "${SANITIZER_FLAGS[@]}" \
    -DPTO_FDWIC_SHARED_MAP=1 \
    -DPA_BUILD_SWIMLANE=0 \
    -DPA_BUILD_SUBMIT_PMU=0 \
    -DPA_BUILD_PERF_CLOCK=0 \
    -I"$ROOT_DIR/common" \
    -I"$ROOT_DIR/ordinary/scalar_build/common" \
    "$ROOT_DIR/ordinary/scalar_build/test/test_aicpu_plan_scalar_scheduler.cpp" \
    -o "$BUILD_DIR/test_aicpu_plan_scalar_scheduler"

"$BUILD_DIR/test_aicpu_plan_scalar_scheduler"
