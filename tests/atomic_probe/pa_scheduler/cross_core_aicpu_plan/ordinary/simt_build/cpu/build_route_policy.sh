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
BUILD_DIR="$SIMT_ROOT/build/cpu/route_policy"
CXX_BIN="${CXX:-g++}"
SOURCE="$SIMT_ROOT/test/test_pa_simt_route_policy.cpp"
OUTPUT="$BUILD_DIR/test_pa_simt_route_policy"

mkdir -p "$BUILD_DIR"

echo "[BUILD] ordinary SIMT PA exact route policy"
"$CXX_BIN" \
    -O2 -std=c++17 -Wall -Wextra -Werror \
    -DPTO_FDWIC_SHARED_MAP=1 \
    -DPTO_FDWIC_TENSORMAP_RING_CAP=128 \
    -DPTO_FDWIC_SHARED_INSERT_TURN_GROUPS=1 \
    -DPA_RUNTIME_PLAN_BUILD_BACKEND=1 \
    -DPA_RUNTIME_PLAN_BUILD_WORKERS=4 \
    "$SOURCE" -o "$OUTPUT"

"$OUTPUT"
echo "[PASS] ordinary SIMT PA route rejects same-engine wrong function IDs"
