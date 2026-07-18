#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 任一工具失败、未定义变量或管道中间失败都立即终止，避免继续使用半成品 device ELF。
set -euo pipefail

# 所有输入和产物都从脚本自身位置解析，调用者无需位于仓库根目录。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/ccec"

# 编译只依赖本目录源码与用户安装的 CANN/PTO 头，不引用 pa_scheduler 目录外的 simpler 构建产物。
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set; source the CANN 9.1 set_env.sh first." >&2
    exit 1
fi

CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD="$ASCEND_HOME_PATH/bin/ld.lld"
HCC="$ASCEND_HOME_PATH/tools/hcc/bin/aarch64-target-linux-gnu-g++"
CXX_BIN="${CXX:-g++}"
READELF_BIN="${READELF:-readelf}"
PTO_INCLUDE_ROOT="${PTO_ISA_ROOT:-$ASCEND_HOME_PATH/x86_64-linux}"

# ccec/ld.lld 必须来自当前已 source 的 CANN；host 编译器和 readelf 允许用户通过环境变量替换。
if [[ ! -x "$CCEC" || ! -x "$LD" || ! -x "$HCC" ]]; then
    echo "CCEC, ld.lld, or the AICPU HCC compiler is missing under ASCEND_HOME_PATH=$ASCEND_HOME_PATH" >&2
    exit 1
fi
if ! command -v "$READELF_BIN" >/dev/null 2>&1; then
    echo "readelf is required to verify the mixed AICore ELF." >&2
    exit 1
fi
if [[ ! -f "$PTO_INCLUDE_ROOT/include/pto/common/kernel_meta.hpp" ]]; then
    echo "PTO kernel metadata header is missing under $PTO_INCLUDE_ROOT/include" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

# 关闭编译器自动插入的 scalar DCCI，由 kernel.cpp 中与 PA 对齐的显式失效/回写协议负责 cache 可见性。
# 两种架构共用这些 ABI、栈和优化参数，避免 AIC/AIV 对共享 SchedulerState 产生不同解释。
COMMON_FLAGS=(
    -c -O3 -g -x cce -Wall -std=c++17
    --cce-aicore-only
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false
    -I"$ROOT_DIR/common"
    -I"$PTO_INCLUDE_ROOT/include"
)

# 同一入口源码分别面向 cube 与 vector ISA 编译，宏只选择各自的全局入口和 mixed metadata。
echo "[BUILD] CCEC AIC entry (dav-c310-cube)"
"$CCEC" "${COMMON_FLAGS[@]}" \
    --cce-aicore-arch=dav-c310-cube \
    -DPA_BUILD_AIC \
    -o "$BUILD_DIR/pa_scheduler_aic.o" \
    "$SCRIPT_DIR/kernel.cpp"

echo "[BUILD] CCEC AIV entry (dav-c310-vec)"
"$CCEC" "${COMMON_FLAGS[@]}" \
    --cce-aicore-arch=dav-c310-vec \
    -DPA_BUILD_AIV \
    -o "$BUILD_DIR/pa_scheduler_aiv.o" \
    "$SCRIPT_DIR/kernel.cpp"

# 静态链接把两个 device object 合成一个可由 runtime 按 1:2 比例启动的 mixed AICore ELF。
echo "[BUILD] Static 1:2 mixed AICore ELF"
"$LD" -m aicorelinux -Ttext=0 -static \
    -o "$BUILD_DIR/pa_scheduler_kernel.o" \
    "$BUILD_DIR/pa_scheduler_aic.o" \
    "$BUILD_DIR/pa_scheduler_aiv.o"

SYMBOL_TABLE="$("$READELF_BIN" --symbols --wide "$BUILD_DIR/pa_scheduler_kernel.o")"
SECTION_TABLE="$("$READELF_BIN" --sections --wide "$BUILD_DIR/pa_scheduler_kernel.o")"
# 构建成功不等于 mixed launch 可用：同时检查两个入口符号及其 metadata section，缺一即拒绝产物。
# `set -e` 同时保证 readelf 自身失败时不会拿空字符串继续做伪检查。
for entry in pa_scheduler_0_mix_aic pa_scheduler_0_mix_aiv; do
    if [[ "$SYMBOL_TABLE" != *" $entry"* ]]; then
        echo "Missing mixed-kernel entry: $entry" >&2
        exit 1
    fi
    if [[ "$SECTION_TABLE" != *".ascend.meta.$entry"* ]]; then
        echo "Missing mixed-kernel metadata section: .ascend.meta.$entry" >&2
        exit 1
    fi
done
echo "[CHECK] both 1:2 mixed entries and metadata sections are present"

check_icache_probe_layout() {
    local role="$1"
    local target="pa_icache_target_${role}"
    local harness="pa_icache_measure_${role}"
    local thrash="pa_icache_thrash_${role}"
    local target_record
    local harness_record
    local thrash_record
    target_record="$(awk -v name="$target" '$4 == "FUNC" && index($NF, name) != 0 {print $2, $3; exit}' <<<"$SYMBOL_TABLE")"
    harness_record="$(awk -v name="$harness" '$4 == "FUNC" && index($NF, name) != 0 {print $2, $3; exit}' <<<"$SYMBOL_TABLE")"
    thrash_record="$(awk -v name="$thrash" '$4 == "FUNC" && index($NF, name) != 0 {print $2, $3; exit}' <<<"$SYMBOL_TABLE")"
    if [[ -z "$target_record" || -z "$harness_record" || -z "$thrash_record" ]]; then
        echo "Missing I-cache probe symbols for $role" >&2
        exit 1
    fi

    local target_hex target_size harness_hex harness_size thrash_hex thrash_size
    read -r target_hex target_size <<<"$target_record"
    read -r harness_hex harness_size <<<"$harness_record"
    read -r thrash_hex thrash_size <<<"$thrash_record"
    local target_address=$((16#$target_hex))
    local harness_address=$((16#$harness_hex))
    local thrash_address=$((16#$thrash_hex))
    if (( target_address % 128 != 0 || target_size == 0 || target_size > 16 )); then
        echo "Invalid single-fetch-block I-cache target for $role: address=0x$target_hex size=$target_size" >&2
        exit 1
    fi
    if (( thrash_size < 65536 )); then
        echo "I-cache thrash body is smaller than 64 KiB for $role: size=$thrash_size" >&2
        exit 1
    fi
    if (( harness_address % 128 != 0 || target_address + 128 > harness_address ||
          harness_address + harness_size > thrash_address )); then
        echo "I-cache layout must be target -> harness -> thrash for $role" >&2
        exit 1
    fi
    echo "[CHECK] $role I-cache target=0x$target_hex/$target_size harness=0x$harness_hex/$harness_size "\
         "thrash=0x$thrash_hex/$thrash_size"
}

# 编译器不能把目标扩到两个 16B fetch block，也不能折叠 64 KiB 冲刷体；否则
# cold/warm 虽然仍可能产生数字，却不再代表可解释的单次 I-cache miss。
check_icache_probe_layout aic
check_icache_probe_layout aiv

# PMU selector/CTRL 的所有权必须由主 aicpu_scheduler 配置并在退出前恢复。
# standalone 目录内自带 Path-A dispatcher 与 owner：前者负责把 owner SO
# 落到设备预安装目录，后者由 mode=0 JSON 注册并通过统一入口执行命令。
echo "[BUILD] self-contained AICPU PMU dispatcher"
"$HCC" -shared -fPIC -O3 -g -std=gnu++17 -Wall -Wextra -Werror \
    -Wl,--build-id \
    "$SCRIPT_DIR/pmu_owner_dispatcher.cpp" \
    -o "$BUILD_DIR/libpa_scheduler_pmu_owner_dispatcher.so"

echo "[BUILD] self-contained AICPU PMU owner"
"$HCC" -shared -fPIC -O3 -g -std=gnu++17 -Wall -Wextra -Werror \
    -Wl,--build-id \
    -I"$SCRIPT_DIR" \
    "$SCRIPT_DIR/pmu_owner_aicpu.cpp" \
    -o "$BUILD_DIR/libpa_scheduler_pmu_owner_aicpu.so"

OWNER_HEADER="$("$READELF_BIN" --file-header "$BUILD_DIR/libpa_scheduler_pmu_owner_aicpu.so")"
OWNER_SYMBOLS="$("$READELF_BIN" --dyn-syms --wide "$BUILD_DIR/libpa_scheduler_pmu_owner_aicpu.so")"
DISPATCHER_HEADER="$("$READELF_BIN" --file-header "$BUILD_DIR/libpa_scheduler_pmu_owner_dispatcher.so")"
DISPATCHER_SYMBOLS="$("$READELF_BIN" --dyn-syms --wide "$BUILD_DIR/libpa_scheduler_pmu_owner_dispatcher.so")"
if [[ "$OWNER_HEADER" != *"Type:                              DYN"* ||
      "$OWNER_HEADER" != *"Machine:                           AArch64"* ||
      "$DISPATCHER_HEADER" != *"Type:                              DYN"* ||
      "$DISPATCHER_HEADER" != *"Machine:                           AArch64"* ]]; then
    echo "PMU dispatcher and owner must both be AArch64 shared objects." >&2
    exit 1
fi
if [[ "$OWNER_SYMBOLS" != *" simpler_aicpu_exec"* ]]; then
    echo "Missing main AICPU PMU owner entry: simpler_aicpu_exec" >&2
    exit 1
fi
for entry in StaticTileFwkBackendKernelServer DynTileFwkBackendKernelServerInit DynTileFwkBackendKernelServer; do
    if [[ "$DISPATCHER_SYMBOLS" != *" $entry"* ]]; then
        echo "Missing AICPU PMU dispatcher entry: $entry" >&2
        exit 1
    fi
done
echo "[CHECK] Path-A dispatcher and main AICPU PMU owner exports are present"

# host runner 只链接用户 CANN 9.1 的 ACL/runtime，并写入同一安装目录的 rpath，运行时不需要 simpler 动态库。
# `-Werror` 让 host API 签名或尺寸类型变化在构建期暴露，避免到上板阶段才出现参数截断。
echo "[BUILD] CCEC host runner"
"$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror -Wno-deprecated-declarations \
    -I"$ROOT_DIR/common" \
    -I"$ASCEND_HOME_PATH/include" \
    -I"$ASCEND_HOME_PATH/pkg_inc" \
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
    "$SCRIPT_DIR/host.cpp" \
    -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -lascendcl -lruntime \
    -ldl \
    -o "$BUILD_DIR/pa_scheduler_host"

echo "[BUILD] complete: $BUILD_DIR"
