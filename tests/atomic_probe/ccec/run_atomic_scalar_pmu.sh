#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 独立构建并运行单 AIV atomic/scalar PMU 对照；不修改 run_all.sh，也不下载 PTO_ISA。
# 用法：
#   ./run_atomic_scalar_pmu.sh          # build + run
#   ./run_atomic_scalar_pmu.sh build    # 仅构建
#   ./run_atomic_scalar_pmu.sh run      # 仅运行已有产物
# 可选环境：ATOMIC_PROBE_DEVICE、ATOMIC_SCALAR_PMU_REPEATS、
#           ATOMIC_SCALAR_PMU_ROUNDS（逗号分隔）、ATOMIC_SCALAR_PMU_SEED。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
BUILD_DIR="$SCRIPT_DIR/build/atomic_scalar_pmu"
ACTION="${1:-all}"
if [[ "$ACTION" != "all" && "$ACTION" != "build" && "$ACTION" != "run" ]]; then
    echo "Usage: $0 [all|build|run]" >&2
    exit 1
fi
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [all|build|run]" >&2
    exit 1
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set; source the local CANN environment first." >&2
    exit 1
fi

CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD="$ASCEND_HOME_PATH/bin/ld.lld"
HCC="$ASCEND_HOME_PATH/tools/hcc/bin/aarch64-target-linux-gnu-g++"
CXX_BIN="${CXX:-g++}"
READELF_BIN="${READELF:-readelf}"
# 优先复用调用者指定的本机 PTO_ISA；未指定时使用 CANN 安装包自带的 metadata header。
PTO_INCLUDE_ROOT="${PTO_ISA_ROOT:-$ASCEND_HOME_PATH/x86_64-linux}"

if [[ ! -x "$CCEC" || ! -x "$LD" || ! -x "$HCC" ]]; then
    echo "CCEC, ld.lld, or AICPU cross compiler is missing under $ASCEND_HOME_PATH" >&2
    exit 1
fi
if [[ ! -f "$PTO_INCLUDE_ROOT/include/pto/common/kernel_meta.hpp" ]]; then
    echo "Local PTO metadata header is missing: $PTO_INCLUDE_ROOT/include/pto/common/kernel_meta.hpp" >&2
    exit 1
fi
if ! command -v "$READELF_BIN" >/dev/null 2>&1; then
    echo "readelf is required." >&2
    exit 1
fi

KERNEL="$BUILD_DIR/atomic_scalar_pmu_kernel.o"
HOST="$BUILD_DIR/atomic_scalar_pmu_host"

build_probe() {
    mkdir -p "$BUILD_DIR"
    local common_flags=(
        -c -O3 -g -x cce -Wall -std=c++17
        --cce-aicore-only
        --cce-aicore-arch=dav-c310-vec
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=false
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-dcci-before-kernel-end=false
        -DCCEC_SYNC_AIV_ONLY
        -I"$SCRIPT_DIR"
        -I"$PTO_INCLUDE_ROOT/include"
    )

    echo "[BUILD] single-AIV CCEC kernel"
    "$CCEC" "${common_flags[@]}" \
        -o "$BUILD_DIR/atomic_scalar_pmu_vec.o" \
        "$SCRIPT_DIR/atomic_scalar_pmu.cpp"
    "$LD" -m aicorelinux -Ttext=0 -static \
        -o "$KERNEL" "$BUILD_DIR/atomic_scalar_pmu_vec.o"

    local symbols sections entry="atomic_scalar_pmu_0_mix_aiv"
    symbols="$("$READELF_BIN" --symbols --wide "$KERNEL")"
    sections="$("$READELF_BIN" --sections --wide "$KERNEL")"
    if [[ "$symbols" != *" $entry"* || "$sections" != *".ascend.meta.$entry"* ]]; then
        echo "Missing AIV entry or metadata for $entry" >&2
        exit 1
    fi
    echo "[CHECK] AIV-only entry and metadata present"

    # 两个 scalar PMU 探针复用同一份 108-subcore 配置/恢复 helper。
    echo "[BUILD] AICPU PMU configure/restore helper"
    "$HCC" -shared -fPIC -O3 -g -std=gnu++17 -Wall -Wextra -Werror \
        -Wl,--build-id \
        -I"$SCRIPT_DIR" \
        -I"$REPO_ROOT/src/a5/platform/include" \
        "$SCRIPT_DIR/pmu_probe_aicpu.cpp" \
        "$REPO_ROOT/src/a5/platform/onboard/aicpu/inner_platform_regs.cpp" \
        -o "$BUILD_DIR/libatomic_scalar_pmu_aicpu.so"

    echo "[BUILD] AICPU bootstrap dispatcher"
    "$HCC" -shared -fPIC -O3 -g -std=gnu++17 -Wall -Wextra \
        -Wl,--build-id \
        -I"$REPO_ROOT/src/common" \
        "$REPO_ROOT/src/common/aicpu_loader/device/aicpu_dispatcher.cpp" \
        -ldl \
        -o "$BUILD_DIR/libsimpler_aicpu_dispatcher.so"

    echo "[BUILD] host runner"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror -Wno-deprecated-declarations \
        -I"$REPO_ROOT/src/common" \
        -I"$REPO_ROOT/src/common/log/include" \
        -I"$REPO_ROOT/src/a5/platform/include" \
        -I"$ASCEND_HOME_PATH/include" \
        -I"$ASCEND_HOME_PATH/pkg_inc" \
        -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
        -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
        "$SCRIPT_DIR/atomic_scalar_pmu_host.cpp" \
        "$REPO_ROOT/src/common/aicpu_loader/host/load_aicpu_op.cpp" \
        "$REPO_ROOT/src/common/log/host_log.cpp" \
        "$REPO_ROOT/src/common/log/unified_log_host.cpp" \
        -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -lascendcl -lruntime -ldl -pthread \
        -o "$HOST"
    echo "[BUILD] complete: $BUILD_DIR"
}

run_probe() {
    if [[ ! -x "$HOST" || ! -f "$KERNEL" || ! -f "$BUILD_DIR/libatomic_scalar_pmu_aicpu.so" ||
          ! -f "$BUILD_DIR/libsimpler_aicpu_dispatcher.so" ]]; then
        echo "Build artifacts are incomplete; run '$0 build' first." >&2
        exit 1
    fi
    echo "[RUN] device=${ATOMIC_PROBE_DEVICE:-${TASK_DEVICE:-0}}"
    timeout "${ATOMIC_PROBE_TIMEOUT:-120}" "$HOST" "$KERNEL"
}

if [[ "$ACTION" == "all" || "$ACTION" == "build" ]]; then
    build_probe
fi
if [[ "$ACTION" == "all" || "$ACTION" == "run" ]]; then
    run_probe
fi
