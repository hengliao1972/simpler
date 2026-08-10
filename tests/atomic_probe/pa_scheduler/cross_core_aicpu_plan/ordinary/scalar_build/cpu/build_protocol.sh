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

mkdir -p "$BUILD_DIR"
"$CXX_BIN" \
    -O2 -std=c++17 -Wall -Wextra -Werror -pthread \
    -I"$ROOT_DIR/common" \
    "$SCRIPT_DIR/test_aicpu_plan_protocol.cpp" \
    -o "$BUILD_DIR/test_aicpu_plan_protocol"

"$BUILD_DIR/test_aicpu_plan_protocol"

# PA 只是公共 Plan ABI 的首个 adapter 样例。该独立门槛使用真实
# standalone TaskArgs/TensorDesc/CreateInfo/FdwicOutputRef，对完整
# MakeSpec -> Publish -> Acquire -> Decode 链路做逐字段检查；同时以严格
# 告警锁住 canonical padding、无源指针 payload 与非法输入 fail-closed。
"$CXX_BIN" \
    -O2 -std=c++17 -Wall -Wextra -Werror -pthread \
    -DPTO_FDWIC_SHARED_MAP=1 \
    -DPA_BUILD_SWIMLANE=0 \
    -I"$ROOT_DIR/common" \
    -I"$ROOT_DIR/ordinary/scalar_build/common" \
    "$ROOT_DIR/ordinary/scalar_build/test/test_aicpu_plan_pa_adapter.cpp" \
    -o "$BUILD_DIR/test_aicpu_plan_pa_adapter"

"$BUILD_DIR/test_aicpu_plan_pa_adapter"
