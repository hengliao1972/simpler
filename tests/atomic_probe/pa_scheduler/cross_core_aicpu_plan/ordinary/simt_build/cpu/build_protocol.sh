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

# PA Case1 的 ordinary_count 当前恒为零，不能用它证明 SIMT Build 保留了
# 通用 TensorMap 语义。这个独立门槛把四个 warp leader 的动态 ticket
# 直接接到现有 generic Publish/append/fanin 实现，覆盖同 key 多 writer、
# writer<consumer 查询、空 writer 交棒，以及重复/越序 fail-closed。
"$CXX_BIN" \
    -O2 -std=c++17 -pthread -Wall -Wextra -Werror \
    -DPTO_FDWIC_SHARED_MAP=1 \
    -DPA_BUILD_SWIMLANE=1 \
    "${SANITIZER_FLAGS[@]}" \
    -I"$MODE_ROOT/../scalar_build/common" \
    "$MODE_ROOT/test/test_simt_ordinary_writer_gate.cpp" \
    -o "$BUILD_DIR/test_simt_ordinary_writer_gate"

timeout --foreground 20s "$BUILD_DIR/test_simt_ordinary_writer_gate"
