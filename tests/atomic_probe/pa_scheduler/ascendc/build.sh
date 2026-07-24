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
BUILD_DIR="$ROOT_DIR/build/ascendc/$TENSORMAP_MODE/swimlane"

# 所有源码和公共头都从 pa_scheduler 目录解析；外部只需要用户安装的
# CANN 工具链和运行库，不搜索 simpler 仓库根目录。

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    # 非交互 shell 不保证自动 source CANN 环境，缺失时直接失败，避免误用
    # PATH 中其他版本的 bisheng。
    echo "ASCEND_HOME_PATH is not set; source the CANN 9.1 set_env.sh first." >&2
    exit 1
fi

BISHENG="$ASCEND_HOME_PATH/bin/bisheng"
if [[ ! -x "$BISHENG" ]]; then
    # 只接受当前 ASCEND_HOME_PATH 下的编译器，确保构建与运行库来自同一套 CANN。
    echo "bisheng is missing under ASCEND_HOME_PATH=$ASCEND_HOME_PATH" >&2
    exit 1
fi

# build 产物固定放回 pa_scheduler/build/ascendc；目录可重复创建，重新构建会
# 原位替换可执行文件，不向源码目录散落中间产物。
mkdir -p "$BUILD_DIR"

echo "[BUILD] AscendC 1:2 mixed host + kernel executable"
# -xasc 会把本文件中的 host main 与 AscendC kernel 构建为一个可执行文件。
# __MIX_CORE_AIC_RATION__ 的拼写来自 bisheng mixed-core ABI；值 2 必须与
# kernel 上的 __mix__(1, 2) 一致，否则 launch metadata 不能表达 32+64 拓扑。
# 调度器已经在协议边界显式执行 dcci，关闭自动 scalar DCCI 可避免编译器
# 额外插入 cache 操作并扰动待测的 atomic/Submit 时序。
"$BISHENG" -O3 -xasc \
    "$SCRIPT_DIR/pa_scheduler.asc" \
    --npu-arch=dav-3510 \
    "-DPTO_FDWIC_SHARED_MAP=$TENSORMAP_MODE_ID" \
    -D__MIX_CORE_AIC_RATION__=2 \
    -I"$ROOT_DIR/common" \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -mllvm -cce-aicore-dcci-before-kernel-end=false \
    -o "$BUILD_DIR/pa_scheduler_ascendc"

echo "[BUILD] complete: $BUILD_DIR/pa_scheduler_ascendc"
