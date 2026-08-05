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
GM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SIMT_ROOT="$(cd "$GM_ROOT/.." && pwd)"
BUILD_DIR="$GM_ROOT/build/ccec"
KERNEL_SOURCE="$SCRIPT_DIR/s1_vector_kernel.cpp"
HOST_SOURCE="$SCRIPT_DIR/s1_vector_host.cpp"
AIC_OBJECT="$BUILD_DIR/simt_cross_core_s1_aic.o"
AIV_OBJECT="$BUILD_DIR/simt_cross_core_s1_aiv.o"
AIV_BITCODE="$BUILD_DIR/simt_cross_core_s1_aiv.bc"
AIV_BITCODE_DUMP="$BUILD_DIR/simt_cross_core_s1_aiv.bc.dump"
KERNEL_ELF="$BUILD_DIR/simt_cross_core_s1_kernel.o"
HOST_BINARY="$BUILD_DIR/simt_cross_core_s1_host"
AIC_ENTRY="simt_cross_core_s1_0_mix_aic"
AIV_ENTRY="simt_cross_core_s1_0_mix_aiv"

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
    echo "readelf is required for mixed-ELF validation." >&2
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
    -Wno-logical-op-parentheses
    -Wno-unused-but-set-variable
    -Wno-bitwise-op-parentheses
    -Wno-unused-local-typedef
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
    -I"$GM_ROOT/common"
    -I"$SIMT_ROOT/common"
)

echo "[CHECK] S1 source closure and publication/completion order"
if rg -n '#include.*(cross_core|ops-nn)' "$SIMT_ROOT" -g '*.h' -g '*.cpp'; then
    echo "S1 must not include cross_core or ops-nn source files." >&2
    exit 1
fi

payload_line="$(grep -nF 'payload_words[kPayloadMagicWord] = kPayloadMagic' "$KERNEL_SOURCE" | cut -d: -f1)"
writer_dcci_line="$(grep -nF 'asc_dcci_single(static_cast<__gm__ void *>(payload_words))' "$KERNEL_SOURCE" | cut -d: -f1)"
publish_fence_line="$(grep -nF 'asc_threadfence();' "$KERNEL_SOURCE" | sed -n '2p' | cut -d: -f1)"
publish_cas_line="$(grep -nF 'const uint64_t publish_observed = asc_atomic_cas' "$KERNEL_SOURCE" | cut -d: -f1)"
if [[ -z "$payload_line" || -z "$writer_dcci_line" || -z "$publish_fence_line" || -z "$publish_cas_line" ]] ||
   ! (( payload_line < writer_dcci_line && writer_dcci_line < publish_fence_line &&
         publish_fence_line < publish_cas_line )); then
    echo "S1 must preserve payload stores < optional writer DCCI < thread fence < BUILT CAS." >&2
    exit 1
fi

tload_line="$(grep -nF 'TLOAD(input_a_tile, input_a_global)' "$KERNEL_SOURCE" | cut -d: -f1)"
tadd_line="$(grep -nF 'TADD(output_tile, input_a_tile, input_b_tile)' "$KERNEL_SOURCE" | cut -d: -f1)"
tstore_line="$(grep -nF 'TSTORE(output_global, output_tile)' "$KERNEL_SOURCE" | cut -d: -f1)"
completion_wait_line="$(grep -nF 'wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7)' "$KERNEL_SOURCE" | cut -d: -f1)"
executor_call_line="$(awk '/__aicore__ void RunExecutor/ {inside=1} inside && /RunVectorAdd\(/ {print NR; exit}' "$KERNEL_SOURCE")"
done_cas_line="$(grep -nF 'result.done_observed = ScalarCas' "$KERNEL_SOURCE" | cut -d: -f1)"
if [[ -z "$tload_line" || -z "$tadd_line" || -z "$tstore_line" || -z "$completion_wait_line" ||
      -z "$executor_call_line" || -z "$done_cas_line" ]] ||
   ! (( tload_line < tadd_line && tadd_line < tstore_line && tstore_line < completion_wait_line &&
         executor_call_line < done_cas_line )); then
    echo "S1 must preserve TLOAD < TADD < TSTORE < MTE3->S completion and RunVectorAdd < DONE CAS." >&2
    exit 1
fi

echo "[BUILD] CCEC S1 AIC observer (dav-c310-cube)"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" --cce-aicore-arch=dav-c310-cube \
    -o "$AIC_OBJECT" "$KERNEL_SOURCE"

echo "[BUILD] CCEC S1 AIV SIMT-builder/Vector-executor (dav-c310-vec)"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" --cce-aicore-arch=dav-c310-vec \
    -o "$AIV_OBJECT" "$KERNEL_SOURCE"

echo "[BUILD] CCEC S1 optimized AIV bitcode"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" --cce-aicore-arch=dav-c310-vec \
    -Xclang -emit-llvm-bc -o "$AIV_BITCODE" "$KERNEL_SOURCE"
"$LLVM_BCANALYZER" -dump "$AIV_BITCODE" > "$AIV_BITCODE_DUMP"
required_bitcode_symbols=(
    "S1SimtBuildVectorTask"
    "llvm.hivm.store.vfsimt.info"
    "llvm.hivm.get.TID.X"
    "llvm.hivm.atom.CAS.G.u64"
    "llvm.hivm.atom.ADD.G.u64"
    "llvm.hivm.fence.workitems"
    "llvm.hivm.DCCI.DST"
    "llvm.hivm.DSB"
    "llvm.hivm.vldsx1.v64f32"
    "llvm.hivm.vadd.s.x.v64f32"
    "llvm.hivm.vstsx1.v64f32"
    "llvm.hivm.SET.FLAG.IMM"
    "llvm.hivm.WAIT.FLAG.IMM"
    "llvm.hivm.ST.DEV.u64"
)
for symbol in "${required_bitcode_symbols[@]}"; do
    if ! grep -Fq "$symbol" "$AIV_BITCODE_DUMP"; then
        echo "optimized S1 AIV bitcode is missing: $symbol" >&2
        exit 1
    fi
done
echo "[CHECK] bitcode contains SIMT publication, atomic polling, DCCI and real Vector-add intrinsics"

echo "[BUILD] Static S1 1:2 mixed AICore ELF"
"$LD_LLD" -m aicorelinux -Ttext=0 -static \
    -o "$KERNEL_ELF" "$AIC_OBJECT" "$AIV_OBJECT"

SYMBOLS="$("$READELF_BIN" --symbols --wide --sym-base=10 "$KERNEL_ELF")"
SECTIONS="$("$READELF_BIN" --sections --wide "$KERNEL_ELF")"
RELOCATIONS="$("$READELF_BIN" --relocs --wide "$KERNEL_ELF")"

for entry in "$AIC_ENTRY" "$AIV_ENTRY"; do
    if ! awk -v name="$entry" \
        '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
         END {exit count != 1}' <<<"$SYMBOLS"; then
        echo "final S1 ELF must define one non-empty GLOBAL entry: $entry" >&2
        exit 1
    fi
    if [[ "$SECTIONS" != *".ascend.meta.$entry"* ]]; then
        echo "final S1 ELF is missing metadata: .ascend.meta.$entry" >&2
        exit 1
    fi
done
global_functions="$(awk '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" {print $NF}' <<<"$SYMBOLS")"
expected_global_functions="$(printf '%s\n%s\n' "$AIC_ENTRY" "$AIV_ENTRY")"
if [[ "$global_functions" != "$expected_global_functions" ]]; then
    echo "final S1 ELF exports unexpected GLOBAL functions:" >&2
    printf '%s\n' "$global_functions" >&2
    exit 1
fi
if ! awk '$4 == "FUNC" && $5 == "LOCAL" && $3 + 0 > 0 && $NF ~ /S1SimtBuildVectorTask.*_simt_entry$/ {count++}
          END {exit count != 1}' <<<"$SYMBOLS"; then
    echo "final S1 ELF must retain one non-empty LOCAL SIMT builder entry." >&2
    exit 1
fi
if ! awk '$4 == "FUNC" && $5 == "LOCAL" && $3 + 0 > 0 && $NF ~ /RunVectorAdd/ {count++}
          END {exit count != 1}' <<<"$SYMBOLS"; then
    echo "final S1 ELF must retain one non-empty LOCAL Vector-add function." >&2
    exit 1
fi
undefined_globals="$(awk '$5 == "GLOBAL" && $7 == "UND" {print $NF}' <<<"$SYMBOLS")"
if [[ -n "$undefined_globals" ]]; then
    echo "final S1 ELF contains undefined GLOBAL symbols:" >&2
    printf '%s\n' "$undefined_globals" >&2
    exit 1
fi
if [[ "$RELOCATIONS" != *"There are no relocations"* ]]; then
    echo "final S1 ELF unexpectedly retains relocations." >&2
    exit 1
fi

METADATA_OUTPUT="$(python3 -m msobjdump -d "$KERNEL_ELF")"
if [[ "$(grep -Fc 'KERNEL_TYPE: MIX_AIC_MAIN' <<<"$METADATA_OUTPUT")" -ne 2 ||
      "$(grep -Fc 'MIX_TASK_RATION: [1:2]' <<<"$METADATA_OUTPUT")" -ne 2 ]]; then
    echo "S1 mixed metadata is not an exact pair of MIX_AIC_MAIN [1:2] entries:" >&2
    printf '%s\n' "$METADATA_OUTPUT" >&2
    exit 1
fi
AIV_META_HEX="$("$READELF_BIN" -x ".ascend.meta.$AIV_ENTRY" "$KERNEL_ELF")"
if [[ "$AIV_META_HEX" != *"0c000400 04000000"* || "$AIV_META_HEX" != *"07000400 00200000"* ]]; then
    echo "S1 AIV metadata must encode SIMD_SIMT_MIX_VF=4 and 8 KiB SIMT share memory." >&2
    printf '%s\n' "$AIV_META_HEX" >&2
    exit 1
fi
VECTOR_UB_BYTES=$((3 * 64 * 1024))
SIMT_SHARE_BYTES=$((8 * 1024))
MAX_LOCAL_BYTES=$((224 * 1024))
if (( VECTOR_UB_BYTES + SIMT_SHARE_BYTES > MAX_LOCAL_BYTES )); then
    echo "S1 Vector UB plus SIMT share memory exceeds the 224 KiB A5 budget." >&2
    exit 1
fi
echo "[CHECK] ELF exports only two mixed entries; AIV is SIMD_SIMT_MIX_VF and UB budget is 200/224 KiB"

echo "[BUILD] GCC 15 S1 ACL host ($("$GXX15" -dumpfullversion))"
"$GXX15" -O2 -std=c++17 -Wall -Wextra -Werror \
    -Wno-deprecated-declarations \
    -I"$GM_ROOT/common" \
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
    echo "S1 build did not produce the required kernel and host artifacts." >&2
    exit 1
fi

echo "[BUILD] S1 CCEC complete"
echo "[BUILD] kernel: $KERNEL_ELF"
echo "[BUILD] host:   $HOST_BINARY"
