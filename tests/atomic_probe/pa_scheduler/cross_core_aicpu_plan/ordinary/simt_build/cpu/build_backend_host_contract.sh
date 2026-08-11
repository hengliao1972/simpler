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
SIMT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ORDINARY_ROOT="$(cd "$SIMT_ROOT/.." && pwd)"
SCALAR_ROOT="$ORDINARY_ROOT/scalar_build"
BUILD_DIR="$SIMT_ROOT/build/cpu/backend_host_contract"
CXX_BIN="${CXX:-g++}"
TEST_SOURCE="$SIMT_ROOT/test/test_simt_backend_host_contract.cpp"

mkdir -p "$BUILD_DIR"

COMMON_FLAGS=(
    -O2
    -std=c++17
    -Wall
    -Wextra
    -Werror
    -DPTO_FDWIC_SHARED_MAP=1
    -DPTO_FDWIC_TENSORMAP_RING_CAP=128
    -DPA_BUILD_SWIMLANE=1
    -DPA_BUILD_SUBMIT_PMU=0
    -DPA_BUILD_PERF_CLOCK=0
    -I"$SCALAR_ROOT/common"
)

echo "[BUILD] Scalar backend Host contract (backend=0, W=96)"
"$CXX_BIN" "${COMMON_FLAGS[@]}" \
    -DPA_RUNTIME_PLAN_BUILD_BACKEND=0 \
    -DPA_RUNTIME_PLAN_BUILD_WORKERS=96 \
    "$TEST_SOURCE" \
    -o "$BUILD_DIR/test_scalar_backend_host_contract"

echo "[TEST] Scalar backend Host contract"
"$BUILD_DIR/test_scalar_backend_host_contract"

echo "[BUILD] SIMT backend Host contract (backend=1, W=4)"
"$CXX_BIN" "${COMMON_FLAGS[@]}" \
    -DPA_RUNTIME_PLAN_BUILD_BACKEND=1 \
    -DPA_RUNTIME_PLAN_BUILD_WORKERS=4 \
    "$TEST_SOURCE" \
    -o "$BUILD_DIR/test_simt_backend_host_contract"

echo "[TEST] SIMT backend Host contract"
"$BUILD_DIR/test_simt_backend_host_contract"

echo "[PASS] Scalar/SIMT Host validation contracts compile and pass under -Werror."
