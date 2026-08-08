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
SIMT_ROOT="$(cd "$PROBE_ROOT/../.." && pwd)"
BUILD_DIR="$PROBE_ROOT/build/ccec"
KERNEL_SOURCE="$SCRIPT_DIR/kernel.cpp"
HOST_SOURCE="$SCRIPT_DIR/host.cpp"
ACL_CONFIG="$SCRIPT_DIR/simt_stack_acl.json"
DEVICE_OBJECT="$BUILD_DIR/simt_cross_core_simt_stack_aiv.o"
DEVICE_BITCODE="$BUILD_DIR/simt_cross_core_simt_stack_aiv.bc"
BITCODE_DUMP="$BUILD_DIR/simt_cross_core_simt_stack_aiv.bc.dump"
KERNEL_ELF="$BUILD_DIR/simt_cross_core_simt_stack_kernel.o"
HOST_BINARY="$BUILD_DIR/simt_cross_core_simt_stack_host"
BUILD_MANIFEST="$BUILD_DIR/simt_stack_build_manifest.sha256"
ENTRY="simt_cross_core_simt_stack_0_mix_aiv"

STACK_BUILD_INPUTS=(
    run.sh
    protocol_probe/simt_stack/common/simt_stack_probe.h
    protocol_probe/simt_stack/ccec/kernel.cpp
    protocol_probe/simt_stack/ccec/host.cpp
    protocol_probe/simt_stack/ccec/simt_stack_acl.json
    protocol_probe/simt_stack/ccec/build.sh
)

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
if ! command -v "$READELF_BIN" >/dev/null 2>&1 || ! command -v rg >/dev/null 2>&1; then
    echo "readelf and rg are required for SIMT stack validation." >&2
    exit 1
fi
if ! python3 -c 'import msobjdump' >/dev/null 2>&1; then
    echo "the CANN msobjdump module is required for metadata validation." >&2
    exit 1
fi
if ! python3 - "$ACL_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    config = json.load(source)
if config.get("StackSize") != {"simt_stack_size": 1024, "simt_divergence_stack_size": 512}:
    raise SystemExit(1)
PY
then
    echo "SIMT stack ACL config must encode 1024 B SIMT and 512 B divergence stack limits." >&2
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
    -I"$PROBE_ROOT/common"
)

echo "[CHECK] SIMT stack source closure and ACL-init ordering"
if rg -n '#include.*(cross_core_ordinary|ops-nn)' "$PROBE_ROOT" -g '*.h' -g '*.cpp'; then
    echo "SIMT stack probe must not include cross_core_ordinary or ops-nn source files." >&2
    exit 1
fi
for required in \
    'LAUNCH_BOUND(kLaunchThreads)' \
    'SimtDivergencePressure(checksum, lane)' \
    'checksum = StackFrame0(checksum)' \
    'asc_stcg(thread_checksums + thread, checksum)' \
    'cce::dim3{kLaunchThreads, 1U, 1U}' \
    'aclInit(acl_config.c_str())'; do
    if ! grep -Fq "$required" "$KERNEL_SOURCE" "$HOST_SOURCE"; then
        echo "SIMT stack probe is missing required source contract: $required" >&2
        exit 1
    fi
done
if grep -Fq 'rtDeviceSetLimit' "$HOST_SOURCE"; then
    echo "SIMT stack probe must configure stacks through the first aclInit call, not a post-init runtime limit." >&2
    exit 1
fi
invoke_line="$(grep -nF 'cce::async_invoke<SimtStackProbe>' "$KERNEL_SOURCE" | cut -d: -f1)"
wait_line="$(grep -nF 'wait_flag(PIPE_V, PIPE_S, EVENT_ID0)' "$KERNEL_SOURCE" | cut -d: -f1)"
scan_line="$(grep -nF 'uint32_t completed_warps = 0U;' "$KERNEL_SOURCE" | cut -d: -f1)"
if [[ -z "$invoke_line" || -z "$wait_line" || -z "$scan_line" ]] ||
   ! (( invoke_line < wait_line && wait_line < scan_line )); then
    echo "SIMT stack source must preserve VF invoke < V/S wait < Scalar validation." >&2
    exit 1
fi

echo "[BUILD] CCEC SIMT stack AIV object"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" -o "$DEVICE_OBJECT" "$KERNEL_SOURCE"

echo "[BUILD] CCEC optimized SIMT stack bitcode"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" -Xclang -emit-llvm-bc -o "$DEVICE_BITCODE" "$KERNEL_SOURCE"
"$LLVM_BCANALYZER" -dump "$DEVICE_BITCODE" > "$BITCODE_DUMP"
required_bitcode_symbols=(
    "SimtStackProbe"
    "StackFrame0"
    "StackFrame1"
    "llvm.hivm.store.vfsimt.info"
    "llvm.hivm.get.TID.X"
    "llvm.hivm.get.CLOCK64"
    "llvm.hivm.fence.workitems"
    "llvm.hivm.SET.FLAG.IMM"
    "llvm.hivm.WAIT.FLAG.IMM"
    "llvm.hivm.LD.DEV.u64.GM"
    "llvm.hivm.ST.DEV.u64"
)
for symbol in "${required_bitcode_symbols[@]}"; do
    if ! grep -Fq "$symbol" "$BITCODE_DUMP"; then
        echo "optimized SIMT stack bitcode is missing: $symbol" >&2
        exit 1
    fi
done
echo "[CHECK] bitcode retains 2048-thread VF, two stack frames, 31-level divergence, CLOCK64 and V/S wait"

echo "[BUILD] AIV-only SIMT stack ELF"
"$LD_LLD" -m aicorelinux -Ttext=0 -static -o "$KERNEL_ELF" "$DEVICE_OBJECT"

SYMBOLS="$($READELF_BIN --symbols --wide --sym-base=10 "$KERNEL_ELF")"
SECTIONS="$($READELF_BIN --sections --wide "$KERNEL_ELF")"
RELOCATIONS="$($READELF_BIN --relocs --wide "$KERNEL_ELF")"
if ! awk -v name="$ENTRY" \
    '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
     END {exit count != 1}' <<<"$SYMBOLS"; then
    echo "final SIMT stack ELF must export one non-empty $ENTRY." >&2
    exit 1
fi
global_functions="$(awk '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" {print $NF}' <<<"$SYMBOLS")"
if [[ "$global_functions" != "$ENTRY" ]]; then
    echo "final SIMT stack ELF exports unexpected GLOBAL functions:" >&2
    printf '%s\n' "$global_functions" >&2
    exit 1
fi
for local_pattern in 'SimtStackProbe.*_simt_entry$' 'StackFrame0' 'StackFrame1'; do
    if ! awk -v pattern="$local_pattern" \
        '$4 == "FUNC" && $5 == "LOCAL" && $3 + 0 > 0 && $NF ~ pattern {found=1} END {exit !found}' \
        <<<"$SYMBOLS"; then
        echo "final SIMT stack ELF is missing non-empty LOCAL function: $local_pattern" >&2
        exit 1
    fi
done
undefined_globals="$(awk '$5 == "GLOBAL" && $7 == "UND" {print $NF}' <<<"$SYMBOLS")"
if [[ -n "$undefined_globals" ]]; then
    echo "final SIMT stack ELF contains undefined GLOBAL symbols:" >&2
    printf '%s\n' "$undefined_globals" >&2
    exit 1
fi
if [[ "$RELOCATIONS" != *"There are no relocations"* ]]; then
    echo "final SIMT stack ELF unexpectedly retains relocations." >&2
    exit 1
fi
metadata_sections="$(awk '{for (i=1; i<=NF; ++i) if ($i ~ /^\.ascend\.meta\./) print $i}' <<<"$SECTIONS")"
if [[ "$metadata_sections" != ".ascend.meta.$ENTRY" ]]; then
    echo "final SIMT stack ELF metadata sections are not exact: $metadata_sections" >&2
    exit 1
fi
METADATA_OUTPUT="$(python3 -m msobjdump -d "$KERNEL_ELF")"
if [[ "$METADATA_OUTPUT" != *"KERNEL_TYPE: MIX_AIV_MAIN"* ||
      "$METADATA_OUTPUT" != *"MIX_TASK_RATION: [0:1]"* ]]; then
    echo "SIMT stack metadata is not MIX_AIV_MAIN [0:1]:" >&2
    printf '%s\n' "$METADATA_OUTPUT" >&2
    exit 1
fi
AIV_META_HEX="$($READELF_BIN -x ".ascend.meta.$ENTRY" "$KERNEL_ELF")"
read -r VF_TYPE COMPILER_ALLOC_UB_BYTES SU_STACK_BYTES SIMT_WARP_STACK_BYTES SIMT_DVG_STACK_BYTES < <(
    AIV_META_HEX="$AIV_META_HEX" python3 - <<'PY'
import os
import re
import struct

blob = bytearray()
for line in os.environ["AIV_META_HEX"].splitlines():
    fields = line.split()
    if not fields or not fields[0].startswith("0x"):
        continue
    for field in fields[1:]:
        if re.fullmatch(r"[0-9A-Fa-f]{8}", field) is None:
            break
        blob.extend(bytes.fromhex(field))
values = {}
offset = 0
while offset + 4 <= len(blob):
    tag, size = struct.unpack_from("<HH", blob, offset)
    offset += 4
    if offset + size > len(blob):
        raise SystemExit("truncated SIMT stack metadata TLV")
    values[tag] = int.from_bytes(blob[offset : offset + size], "little")
    offset += size
print(*(values.get(tag, -1) for tag in (12, 7, 8, 9, 10)))
PY
)
if (( VF_TYPE != 3 || COMPILER_ALLOC_UB_BYTES != 8192 || SU_STACK_BYTES <= 0 ||
      SIMT_WARP_STACK_BYTES <= 0 || SIMT_WARP_STACK_BYTES > 1024 ||
      SIMT_DVG_STACK_BYTES <= 0 || SIMT_DVG_STACK_BYTES > 512 )); then
    echo "SIMT stack metadata must encode VF-only=3, 8 KiB compiler UB, <=1024 B warp stack and <=512 B DVG stack." >&2
    printf '%s\n' "$AIV_META_HEX" >&2
    exit 1
fi
echo "[CHECK] stack metadata: compiler_ub=$COMPILER_ALLOC_UB_BYTES SU=$SU_STACK_BYTES SIMT=$SIMT_WARP_STACK_BYTES DVG=$SIMT_DVG_STACK_BYTES bytes"

echo "[BUILD] GCC 15 SIMT stack ACL host ($($GXX15 -dumpfullversion))"
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
    echo "SIMT stack build did not produce the required kernel and host artifacts." >&2
    exit 1
fi
manifest_tmp="$BUILD_MANIFEST.tmp.$$"
(
    cd "$SIMT_ROOT"
    sha256sum \
        "${STACK_BUILD_INPUTS[@]}" \
        protocol_probe/simt_stack/build/ccec/simt_cross_core_simt_stack_kernel.o \
        protocol_probe/simt_stack/build/ccec/simt_cross_core_simt_stack_host
) > "$manifest_tmp"
mv "$manifest_tmp" "$BUILD_MANIFEST"

echo "[BUILD] SIMT stack CCEC complete"
echo "[BUILD] kernel:   $KERNEL_ELF"
echo "[BUILD] host:     $HOST_BINARY"
echo "[BUILD] config:   $ACL_CONFIG"
echo "[BUILD] manifest: $BUILD_MANIFEST"
