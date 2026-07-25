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
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [private|shared]" >&2
    exit 1
fi
TENSORMAP_MODE="${1:-private}"
case "$TENSORMAP_MODE" in
    private) TENSORMAP_MODE_ID=0 ;;
    shared) TENSORMAP_MODE_ID=1 ;;
    *)
        echo "Unknown TensorMap mode: $TENSORMAP_MODE (expected private|shared)" >&2
        exit 1
        ;;
esac
BUILD_DIR="$ROOT_DIR/build/cpu/$TENSORMAP_MODE/swimlane"
CXX_BIN="${CXX:-g++}"

# CPU 后端只依赖 C++17、pthread 和本目录 common/，不需要 CANN。
# CXX 可显式指向用户目录下的 g++-15，未设置时沿用当前 PATH 中的 g++。

# CPU build 与设备 build 使用平行目录，便于 run.sh 根据 backend 做严格选择，
# 也避免把 host 回归二进制误当成 A5 产物。
mkdir -p "$BUILD_DIR"

echo "[BUILD] CPU scheduler executable"
# -pthread 同时提供编译期线程宏和链接期 pthread 支持；严格告警用于防止
# CPU 等价层因类型或原子接口变化而静默偏离设备端公共协议。private/shared
# 都实例化同一 scheduler，模式宏只选择各自已经接线的 TensorMap backend。
"$CXX_BIN" -O3 -std=c++17 -pthread -Wall -Wextra -Werror \
    "-DPTO_FDWIC_SHARED_MAP=$TENSORMAP_MODE_ID" \
    -I"$ROOT_DIR/common" \
    "$SCRIPT_DIR/main.cpp" \
    -o "$BUILD_DIR/pa_scheduler_cpu"

# PollBatch 是 common/ 中的设备/CPU 共用模板。这里用普通 C++17 编译器
# 直接实例化并执行边界自测；任一断言失败都会借助 set -e 阻止构建成功。
echo "[BUILD] atomic PollBatch boundary self-test"
"$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
    "-DPTO_FDWIC_SHARED_MAP=$TENSORMAP_MODE_ID" \
    -DPA_BUILD_SWIMLANE=1 \
    -I"$ROOT_DIR/common" \
    "$ROOT_DIR/common/test_atomic_poll_batch.cpp" \
    -o "$BUILD_DIR/test_atomic_poll_batch"

echo "[TEST] atomic PollBatch boundary self-test"
"$BUILD_DIR/test_atomic_poll_batch"

# TensorMap 独立回归不启动 96 个 worker，也不运行模拟 kernel：private
# 覆盖单线程 ring 的回收回绕与 reference 差分；shared 覆盖有序发布、
# seq/ABA、慢核回收和整任务容量预检，两者不混用并发纪律。
if [[ "$TENSORMAP_MODE" == "private" ]]; then
    echo "[BUILD] private TensorMap ring self-test"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
        -DPTO_FDWIC_SHARED_MAP=0 \
        -DPA_BUILD_SWIMLANE=1 \
        -I"$ROOT_DIR/common" \
        "$ROOT_DIR/common/test_private_tensor_map_ring.cpp" \
        -o "$BUILD_DIR/test_private_tensor_map_ring"

    echo "[TEST] private TensorMap ring self-test"
    "$BUILD_DIR/test_private_tensor_map_ring"
else
    echo "[BUILD] shared TensorMap ordered-ring self-test"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
        -DPTO_FDWIC_SHARED_MAP=1 \
        -DPA_BUILD_SWIMLANE=1 \
        -I"$ROOT_DIR/common" \
        "$ROOT_DIR/common/test_shared_tensor_map_ring.cpp" \
        -o "$BUILD_DIR/test_shared_tensor_map_ring"

    echo "[TEST] shared TensorMap ordered-ring self-test"
    "$BUILD_DIR/test_shared_tensor_map_ring"

    # fresh-output symbol 与 region ring 是两条独立协议。该用例单独锁定
    # descriptor 发布、INPUT last-writer load、INOUT exchange、payload
    # scratch 解析及非法引用 fail-closed，避免只靠完整 96 线程回放偶然覆盖。
    echo "[BUILD] shared-output symbol self-test"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
        -DPTO_FDWIC_SHARED_MAP=1 \
        -DPA_BUILD_SWIMLANE=1 \
        -I"$ROOT_DIR/common" \
        "$ROOT_DIR/common/test_shared_output_symbols.cpp" \
        -o "$BUILD_DIR/test_shared_output_symbols"

    echo "[TEST] shared-output symbol self-test"
    "$BUILD_DIR/test_shared_output_symbols"

    # shared heap 与 region/symbol 协议分开验证：这里按 exact-turn 前提
    # 锁定 8 shard、1 KiB 对齐、首版禁止 wrap、零输出及异常原子回滚。
    echo "[BUILD] shared heap no-wrap reserve self-test"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
        -DPTO_FDWIC_SHARED_MAP=1 \
        -DPA_BUILD_SWIMLANE=1 \
        -I"$ROOT_DIR/common" \
        "$ROOT_DIR/common/test_shared_heap_reserve.cpp" \
        -o "$BUILD_DIR/test_shared_heap_reserve"

    echo "[TEST] shared heap no-wrap reserve self-test"
    "$BUILD_DIR/test_shared_heap_reserve"

    # Materialize 在触碰 shared cursor 前必须完成数量、引用、shape/stride
    # 和地址区间预检；定向用例保证失败不会留下半次 heap 推进。
    echo "[BUILD] shared winner materialize self-test"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
        -DPTO_FDWIC_SHARED_MAP=1 \
        -DPA_BUILD_SWIMLANE=1 \
        -I"$ROOT_DIR/common" \
        "$ROOT_DIR/common/test_shared_materialize.cpp" \
        -o "$BUILD_DIR/test_shared_materialize"

    echo "[TEST] shared winner materialize self-test"
    "$BUILD_DIR/test_shared_materialize"
fi

# set -e 保证编译或链接失败时不会打印 complete，也不会在组合构建中继续
# 后续步骤；只有成功退出的构建才被本脚本声明为可运行产物。
echo "[BUILD] complete: $BUILD_DIR/pa_scheduler_cpu"
