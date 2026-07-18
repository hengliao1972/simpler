#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 独立构建并运行单 AIV WARM/COLD I-cache PMU 配对；不修改 run_all.sh，
# 也不下载 PTO_ISA。最终 ELF 的符号尺寸、128B 对齐和区间不重叠是运行前硬条件。
# 用法：
#   ./run_icache_scalar_pmu.sh          # build + run（默认 11 对）
#   ./run_icache_scalar_pmu.sh build    # 仅构建和检查 ELF
#   ./run_icache_scalar_pmu.sh run      # 仅检查已有 ELF 后运行
# 可选环境：ATOMIC_PROBE_DEVICE、ICACHE_SCALAR_PMU_REPEATS（11..101）、
#           ICACHE_SCALAR_PMU_SEED、ICACHE_SCALAR_PMU_TIMEOUT。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
BUILD_DIR="$SCRIPT_DIR/build/icache_scalar_pmu"
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
if ! command -v "$CXX_BIN" >/dev/null 2>&1; then
    echo "Host C++ compiler is missing: $CXX_BIN" >&2
    exit 1
fi
if ! command -v "$READELF_BIN" >/dev/null 2>&1; then
    echo "readelf is required: $READELF_BIN" >&2
    exit 1
fi

KERNEL="$BUILD_DIR/icache_scalar_pmu_kernel.o"
HOST="$BUILD_DIR/icache_scalar_pmu_host"

symbol_address_hex=""
symbol_size=""
read_unique_function_symbol() {
    local elf="$1"
    local name="$2"
    local -a matches=()
    mapfile -t matches < <(
        "$READELF_BIN" --symbols --wide "$elf" |
            awk -v symbol="$name" '$4 == "FUNC" && $8 == symbol {print $2 " " $3}'
    )
    if [[ ${#matches[@]} -ne 1 ]]; then
        echo "Expected exactly one FUNC symbol named $name; found ${#matches[@]}" >&2
        exit 1
    fi
    read -r symbol_address_hex symbol_size <<<"${matches[0]}"
    if [[ ! "$symbol_address_hex" =~ ^[0-9a-fA-F]+$ || ! "$symbol_size" =~ ^[0-9]+$ ]]; then
        echo "Cannot parse ELF symbol $name: ${matches[0]}" >&2
        exit 1
    fi
}

check_kernel_elf() {
    if [[ ! -f "$KERNEL" ]]; then
        echo "Kernel ELF is missing: $KERNEL" >&2
        exit 1
    fi

    local symbols sections entry="icache_scalar_pmu_0_mix_aiv"
    symbols="$("$READELF_BIN" --symbols --wide "$KERNEL")"
    sections="$("$READELF_BIN" --sections --wide "$KERNEL")"
    if [[ "$symbols" != *" $entry"* || "$sections" != *".ascend.meta.$entry"* ]]; then
        echo "Missing AIV entry or metadata for $entry" >&2
        exit 1
    fi

    read_unique_function_symbol "$KERNEL" "icache_scalar_pmu_target"
    local target_address_hex="$symbol_address_hex"
    local target_size="$symbol_size"
    read_unique_function_symbol "$KERNEL" "icache_scalar_pmu_evictor"
    local evictor_address_hex="$symbol_address_hex"
    local evictor_size="$symbol_size"

    local target_address=$((16#$target_address_hex))
    local evictor_address=$((16#$evictor_address_hex))
    local target_end=$((target_address + target_size))
    local evictor_end=$((evictor_address + evictor_size))
    if ((target_size < 8192)); then
        echo "target is smaller than 8192B: $target_size" >&2
        exit 1
    fi
    if ((evictor_size < 32768)); then
        echo "evictor is smaller than 32768B: $evictor_size" >&2
        exit 1
    fi
    if ((target_address % 128 != 0)); then
        echo "target address is not 128B-aligned: 0x$target_address_hex" >&2
        exit 1
    fi
    if ((evictor_address % 128 != 0)); then
        echo "evictor address is not 128B-aligned: 0x$evictor_address_hex" >&2
        exit 1
    fi
    if ! ((target_end <= evictor_address || evictor_end <= target_address)); then
        printf 'target/evictor ranges overlap: target=[0x%x,0x%x) evictor=[0x%x,0x%x)\n' \
            "$target_address" "$target_end" "$evictor_address" "$evictor_end" >&2
        exit 1
    fi
    printf '[CHECK] AIV entry/meta PASS; target address=0x%x size=%u; evictor address=0x%x size=%u; ' \
        "$target_address" "$target_size" "$evictor_address" "$evictor_size"
    echo "alignment=128B ranges=non-overlap"
}

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

    echo "[BUILD] single-AIV CCEC WARM/COLD I-cache kernel"
    "$CCEC" "${common_flags[@]}" \
        -o "$BUILD_DIR/icache_scalar_pmu_vec.o" \
        "$SCRIPT_DIR/icache_scalar_pmu.cpp"
    "$LD" -m aicorelinux -Ttext=0 -static \
        -o "$KERNEL" "$BUILD_DIR/icache_scalar_pmu_vec.o"
    check_kernel_elf

    # 与 atomic_scalar_pmu 复用同一份 108-subcore PMU 配置/恢复源码；
    # 仅产物名独立，避免两个 probe 的 build 目录相互依赖。
    echo "[BUILD] AICPU PMU configure/restore helper"
    "$HCC" -shared -fPIC -O3 -g -std=gnu++17 -Wall -Wextra -Werror \
        -Wl,--build-id \
        -I"$SCRIPT_DIR" \
        -I"$REPO_ROOT/src/a5/platform/include" \
        "$SCRIPT_DIR/pmu_probe_aicpu.cpp" \
        "$REPO_ROOT/src/a5/platform/onboard/aicpu/inner_platform_regs.cpp" \
        -o "$BUILD_DIR/libicache_scalar_pmu_aicpu.so"

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
        "$SCRIPT_DIR/icache_scalar_pmu_host.cpp" \
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
    if [[ ! -x "$HOST" || ! -f "$KERNEL" || ! -f "$BUILD_DIR/libicache_scalar_pmu_aicpu.so" ||
          ! -f "$BUILD_DIR/libsimpler_aicpu_dispatcher.so" ]]; then
        echo "Build artifacts are incomplete; run '$0 build' first." >&2
        exit 1
    fi
    # run-only 也重新检查 ELF，不能绕开尺寸/对齐/区间 oracle。
    check_kernel_elf
    echo "[RUN] device=${ATOMIC_PROBE_DEVICE:-${TASK_DEVICE:-0}} pairs=${ICACHE_SCALAR_PMU_REPEATS:-11}"
    timeout "${ICACHE_SCALAR_PMU_TIMEOUT:-120}" "$HOST" "$KERNEL"
}

if [[ "$ACTION" == "all" || "$ACTION" == "build" ]]; then
    build_probe
fi
if [[ "$ACTION" == "all" || "$ACTION" == "run" ]]; then
    run_probe
fi
