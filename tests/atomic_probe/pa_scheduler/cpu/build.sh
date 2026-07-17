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
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/cpu"
CXX_BIN="${CXX:-g++}"

# CPU 后端只依赖 C++17、pthread 和本目录 common/，不需要 CANN。
# CXX 可显式指向用户目录下的 g++-15，未设置时沿用当前 PATH 中的 g++。

# CPU build 与设备 build 使用平行目录，便于 run.sh 根据 backend 做严格选择，
# 也避免把 host 回归二进制误当成 A5 产物。
mkdir -p "$BUILD_DIR"

echo "[BUILD] CPU scheduler executable"
# -pthread 同时提供编译期线程宏和链接期 pthread 支持；严格告警用于防止
# CPU 等价层因类型或原子接口变化而静默偏离设备端公共协议。
"$CXX_BIN" -O3 -std=c++17 -pthread -Wall -Wextra -Werror \
    -I"$ROOT_DIR/common" \
    "$SCRIPT_DIR/main.cpp" \
    -o "$BUILD_DIR/pa_scheduler_cpu"

# set -e 保证编译或链接失败时不会打印 complete，也不会在组合构建中继续
# 后续步骤；只有成功退出的构建才被本脚本声明为可运行产物。
echo "[BUILD] complete: $BUILD_DIR/pa_scheduler_cpu"
