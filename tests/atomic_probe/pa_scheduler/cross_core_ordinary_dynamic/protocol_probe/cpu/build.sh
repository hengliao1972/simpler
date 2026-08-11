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
CROSS_CORE_ROOT="$(cd "$PROBE_ROOT/.." && pwd)"
BUILD_DIR="$PROBE_ROOT/build/cpu"
CXX_BIN="${CXX:-g++}"

mkdir -p "$BUILD_DIR"

echo "[BUILD] cross-core shared execution protocol CPU gate"
"$CXX_BIN" -O2 -std=c++17 -pthread -Wall -Wextra -Werror \
    -I"$CROSS_CORE_ROOT/common" \
    "$PROBE_ROOT/test/test_shared_exec_protocol.cpp" \
    -o "$BUILD_DIR/test_shared_exec_protocol"

echo "[TEST] cross-core shared execution protocol CPU gate"
timeout --foreground 30s "$BUILD_DIR/test_shared_exec_protocol"
