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
SIMT_ROOT="$(cd "$PROBE_ROOT/.." && pwd)"
BUILD_DIR="$PROBE_ROOT/build/ccec"
KERNEL_SOURCE="$SCRIPT_DIR/kernel.cpp"
HOST_SOURCE="$SCRIPT_DIR/host.cpp"
DEVICE_OBJECT="$BUILD_DIR/simt_cross_core_s0_aiv.o"
DEVICE_BITCODE="$BUILD_DIR/simt_cross_core_s0_aiv.bc"
BITCODE_DUMP="$BUILD_DIR/simt_cross_core_s0_aiv.bc.dump"
KERNEL_ELF="$BUILD_DIR/simt_cross_core_s0_kernel.o"
HOST_BINARY="$BUILD_DIR/simt_cross_core_s0_host"
ENTRY="simt_cross_core_s0_0_mix_aiv"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set; source the CANN environment first." >&2
    exit 1
fi

CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD_LLD="$ASCEND_HOME_PATH/bin/ld.lld"
READELF_BIN="${READELF:-readelf}"
if command -v llvm-bcanalyzer >/dev/null 2>&1; then
    LLVM_BCANALYZER="$(command -v llvm-bcanalyzer)"
else
    LLVM_BCANALYZER="/opt/mlir-debug/bin/llvm-bcanalyzer"
fi
if command -v g++-15 >/dev/null 2>&1; then
    GXX15="$(command -v g++-15)"
elif [[ -n "${GCC15_ROOT:-}" && -x "$GCC15_ROOT/usr/bin/g++-15" ]]; then
    GXX15="$GCC15_ROOT/usr/bin/g++-15"
else
    echo "g++-15 is required for the ACL host runner." >&2
    exit 1
fi

for tool in "$CCEC" "$LD_LLD" "$LLVM_BCANALYZER" "$GXX15"; do
    if [[ ! -x "$tool" ]]; then
        echo "required executable is missing: $tool" >&2
        exit 1
    fi
done
if ! command -v "$READELF_BIN" >/dev/null 2>&1; then
    echo "readelf is required for device ELF validation." >&2
    exit 1
fi
if ! python3 -c 'import msobjdump' >/dev/null 2>&1; then
    echo "the CANN msobjdump module is required for metadata validation." >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

COMMON_DEVICE_FLAGS=(
    -c -O3 -g -x cce -Wall -std=c++17
    --cce-aicore-only
    --cce-aicore-arch=dav-c310-vec
    -Wno-missing-braces
    -Wno-unused-variable
    -Wno-unused-function
    -Wno-unneeded-internal-declaration
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false
    -I"$ASCEND_HOME_PATH/x86_64-linux/include"
    -I"$ASCEND_HOME_PATH/x86_64-linux/asc"
    -I"$ASCEND_HOME_PATH/x86_64-linux/asc/include"
    -I"$SIMT_ROOT/common"
    -I"$PROBE_ROOT/common"
)

echo "[CHECK] S0 source closure and publication sequence"
if rg -n '#include.*(cross_core_ordinary|ops-nn)' "$SIMT_ROOT" \
        -g '*.h' -g '*.cpp'; then
    echo "S0 must not include cross_core_ordinary or ops-nn source files." >&2
    exit 1
fi

publication_block="$({
    awk '
        /for \(uint32_t word = 0U; word < kPayloadWords; \+\+word\)/ {inside = 1}
        inside {print}
        /const uint64_t publish_observed = asc_atomic_cas/ {exit}
    ' "$KERNEL_SOURCE"
})"
payload_line="$(grep -nF 'payload_words[word] = SimtExpectedPayloadWord' <<<"$publication_block" | cut -d: -f1)"
dcci_line="$(grep -nF 'asc_dcci_single' <<<"$publication_block" | cut -d: -f1)"
fence_line="$(grep -nF 'asc_threadfence();' <<<"$publication_block" | tail -n 1 | cut -d: -f1)"
publish_line="$(grep -nF 'const uint64_t publish_observed = asc_atomic_cas' <<<"$publication_block" | cut -d: -f1)"
if [[ -z "$payload_line" || -z "$dcci_line" || -z "$fence_line" || -z "$publish_line" ]] ||
   ! (( payload_line < dcci_line && dcci_line < fence_line && fence_line < publish_line )); then
    echo "S0 source must preserve payload stores < optional writer DCCI < thread fence < BUILT CAS." >&2
    exit 1
fi

echo "[BUILD] CCEC AIV object"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" \
    -o "$DEVICE_OBJECT" "$KERNEL_SOURCE"

echo "[BUILD] CCEC optimized SIMT bitcode"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" \
    -Xclang -emit-llvm-bc \
    -o "$DEVICE_BITCODE" "$KERNEL_SOURCE"

# 当前 CCEC SIMT bitcode 使用 amdgpu_cs_chain。环境里的 llvm-dis 会因
# 版本不匹配拒绝该合法模块；llvm-bcanalyzer 可以完整解析 record 和
# value-symbol-table，因此这里检查 lowered intrinsic inventory，不伪称
# 已完成逐 SSA 的文本 IR 顺序验证。
"$LLVM_BCANALYZER" -dump "$DEVICE_BITCODE" > "$BITCODE_DUMP"
required_bitcode_symbols=(
    "S0SimtBuild"
    "llvm.hivm.store.vfsimt.info"
    "llvm.hivm.get.TID.X"
    "llvm.hivm.atom.CAS.G.u64"
    "llvm.hivm.fence.workitems"
    "llvm.hivm.DCCI.DST"
    "llvm.hivm.SET.FLAG.IMM"
    "llvm.hivm.WAIT.FLAG.IMM"
    "llvm.hivm.LD.DEV.u64.GM"
    "llvm.hivm.ST.DEV.u64"
)
for symbol in "${required_bitcode_symbols[@]}"; do
    if ! grep -Fq "$symbol" "$BITCODE_DUMP"; then
        echo "optimized SIMT bitcode is missing: $symbol" >&2
        exit 1
    fi
done
echo "[CHECK] bitcode contains SIMT launch, 64-bit CAS, fence, DCCI matrix and V/S completion intrinsics"

echo "[BUILD] AIV-only S0 ELF"
"$LD_LLD" -m aicorelinux -Ttext=0 -static \
    -o "$KERNEL_ELF" "$DEVICE_OBJECT"

SYMBOLS="$($READELF_BIN --symbols --wide --sym-base=10 "$KERNEL_ELF")"
SECTIONS="$($READELF_BIN --sections --wide "$KERNEL_ELF")"
RELOCATIONS="$($READELF_BIN --relocs --wide "$KERNEL_ELF")"

if ! awk -v name="$ENTRY" \
    '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
     END {exit count != 1}' <<<"$SYMBOLS"; then
    echo "final ELF must export one non-empty $ENTRY." >&2
    exit 1
fi
global_functions="$(awk '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" {print $NF}' <<<"$SYMBOLS")"
if [[ "$global_functions" != "$ENTRY" ]]; then
    echo "final ELF exports unexpected GLOBAL functions:" >&2
    printf '%s\n' "$global_functions" >&2
    exit 1
fi
if ! awk '$4 == "FUNC" && $5 == "LOCAL" && $3 + 0 > 0 && $NF ~ /S0SimtBuild.*_simt_entry$/ {count++}
          END {exit count != 1}' <<<"$SYMBOLS"; then
    echo "final ELF must retain exactly one non-empty LOCAL SIMT entry." >&2
    exit 1
fi
undefined_globals="$(awk '$5 == "GLOBAL" && $7 == "UND" {print $NF}' <<<"$SYMBOLS")"
if [[ -n "$undefined_globals" ]]; then
    echo "final ELF contains undefined GLOBAL symbols:" >&2
    printf '%s\n' "$undefined_globals" >&2
    exit 1
fi
if [[ "$RELOCATIONS" != *"There are no relocations"* ]]; then
    echo "final S0 ELF unexpectedly retains relocations." >&2
    exit 1
fi
metadata_sections="$(awk '{for (i=1; i<=NF; ++i) if ($i ~ /^\.ascend\.meta\./) print $i}' <<<"$SECTIONS")"
if [[ "$metadata_sections" != ".ascend.meta.$ENTRY" ]]; then
    echo "final ELF metadata sections are not exact: $metadata_sections" >&2
    exit 1
fi

METADATA_OUTPUT="$(python3 -m msobjdump -d "$KERNEL_ELF")"
if [[ "$METADATA_OUTPUT" != *"KERNEL_TYPE: MIX_AIV_MAIN"* ||
      "$METADATA_OUTPUT" != *"MIX_TASK_RATION: [0:1]"* ]]; then
    echo "S0 AIV metadata is not MIX_AIV_MAIN [0:1]:" >&2
    printf '%s\n' "$METADATA_OUTPUT" >&2
    exit 1
fi
echo "[CHECK] ELF exports only AIV Main entry; SIMT entry is local; metadata is MIX_AIV_MAIN [0:1]"

echo "[BUILD] GCC 15 ACL host ($($GXX15 -dumpfullversion))"
"$GXX15" -O2 -std=c++17 -Wall -Wextra -Werror \
    -Wno-deprecated-declarations \
    -I"$PROBE_ROOT/common" \
    -I"$ASCEND_HOME_PATH/include" \
    -I"$ASCEND_HOME_PATH/pkg_inc" \
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
    "$HOST_SOURCE" \
    -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -lascendcl -lruntime -ldl \
    -o "$HOST_BINARY"

if [[ ! -s "$KERNEL_ELF" || ! -x "$HOST_BINARY" ]]; then
    echo "S0 build did not produce the required kernel and host artifacts." >&2
    exit 1
fi

echo "[BUILD] S0 CCEC complete"
echo "[BUILD] kernel: $KERNEL_ELF"
echo "[BUILD] host:   $HOST_BINARY"
