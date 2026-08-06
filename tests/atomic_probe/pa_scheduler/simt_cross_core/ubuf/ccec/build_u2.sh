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
UBUF_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SIMT_ROOT="$(cd "$UBUF_ROOT/.." && pwd)"
GM_ROOT="$SIMT_ROOT/gm"
BUILD_DIR="$UBUF_ROOT/build/ccec"
KERNEL_SOURCE="$GM_ROOT/ccec/g0_full_pa_kernel.cpp"
WORKLOAD_SOURCE="$GM_ROOT/ccec/full_pa_workloads.h"
HOST_SOURCE="$GM_ROOT/ccec/g0_full_pa_host.cpp"
U2_HEADER="$UBUF_ROOT/common/u2_full_pa.h"
STAGING_HEADER="$UBUF_ROOT/common/ubuf_staging_protocol.h"
TRANSPORT_HEADER="$UBUF_ROOT/common/u2_payload_transport.h"
TRANSPORT_PROBE_SOURCE="$SCRIPT_DIR/u2_payload_transport_ir_probe.cpp"
ACL_CONFIG="$SCRIPT_DIR/u2_acl.json"
AIC_OBJECT="$BUILD_DIR/simt_cross_core_u2_aic.o"
AIV_OBJECT="$BUILD_DIR/simt_cross_core_u2_aiv.o"
AIC_BITCODE="$BUILD_DIR/simt_cross_core_u2_aic.bc"
AIV_BITCODE="$BUILD_DIR/simt_cross_core_u2_aiv.bc"
AIC_BITCODE_DUMP="$BUILD_DIR/simt_cross_core_u2_aic.bc.dump"
AIV_BITCODE_DUMP="$BUILD_DIR/simt_cross_core_u2_aiv.bc.dump"
TRANSPORT_PROBE_BITCODE="$BUILD_DIR/simt_cross_core_u2_transport_probe.bc"
TRANSPORT_PROBE_DUMP="$BUILD_DIR/simt_cross_core_u2_transport_probe.bc.dump"
KERNEL_ELF="$BUILD_DIR/simt_cross_core_u2_kernel.o"
HOST_BINARY="$BUILD_DIR/simt_cross_core_u2_host"
BUILD_MANIFEST="$BUILD_DIR/u2_build_manifest.sha256"
AIC_ENTRY="simt_cross_core_u2_0_mix_aic"
AIV_ENTRY="simt_cross_core_u2_0_mix_aiv"
U2_BUILD_INPUTS=(
    run.sh
    common/full_pa_exec_protocol.h
    common/full_pa_model.h
    gm/common/g0_full_pa.h
    gm/ccec/full_pa_workloads.h
    gm/ccec/g0_full_pa_kernel.cpp
    gm/ccec/g0_full_pa_host.cpp
    ubuf/common/ubuf_staging_protocol.h
    ubuf/common/u2_full_pa.h
    ubuf/common/u2_payload_transport.h
    ubuf/ccec/build_u2.sh
    ubuf/ccec/u2_payload_transport_ir_probe.cpp
    ubuf/ccec/u2_acl.json
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
for source in "$KERNEL_SOURCE" "$WORKLOAD_SOURCE" "$HOST_SOURCE" "$U2_HEADER" "$STAGING_HEADER" \
              "$TRANSPORT_HEADER" "$TRANSPORT_PROBE_SOURCE" "$ACL_CONFIG"; do
    if [[ ! -s "$source" ]]; then
        echo "required U2 source is missing: $source" >&2
        exit 1
    fi
done
if ! command -v "$READELF_BIN" >/dev/null 2>&1 || ! command -v rg >/dev/null 2>&1; then
    echo "readelf and rg are required for U2 validation." >&2
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
if config.get("StackSize") != {"simt_stack_size": 1536, "simt_divergence_stack_size": 1536}:
    raise SystemExit(1)
PY
then
    echo "U2 ACL config must encode 1536 B SIMT and 1536 B divergence stack limits." >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

COMMON_DEVICE_FLAGS=(
    -c -O3 -g -x cce -Wall -std=c++17
    -DSIMT_CROSS_CORE_U2
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
    -mllvm -cce-vf-stack-size=0x3800
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-record-stack-size=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false
    -I"$ASCEND_HOME_PATH/x86_64-linux/include"
    -I"$ASCEND_HOME_PATH/x86_64-linux/asc"
    -I"$ASCEND_HOME_PATH/x86_64-linux/asc/include"
    -I"$GM_ROOT/common"
    -I"$SIMT_ROOT/common"
    -I"$UBUF_ROOT/common"
)

echo "[CHECK] U2 full-PA single-builder guarded four-slot UBUF transport"
if ! grep -Fq 'aclInit(acl_config)' "$HOST_SOURCE" || grep -Fq 'rtDeviceSetLimit' "$HOST_SOURCE"; then
    echo "U2 must configure SIMT stacks through the first aclInit call and must not use post-init runtime limits." >&2
    exit 1
fi
if ! grep -Fq 'full_pa->control.builder_count == u2::kBuilderCount' "$KERNEL_SOURCE" ||
   ! grep -Fq 'constexpr uint32_t kBuilderCount = 1U;' "$U2_HEADER" ||
   ! grep -Fq 'cce::async_invoke<U2SimtBuildTasks>' "$KERNEL_SOURCE" ||
   ! grep -Fq 'reinterpret_cast<__ubuf__ volatile uint64_t *>(kU2UbufRegionOffset)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'TransportKind::SimtUbufReadToGmWordStore' "$HOST_SOURCE"; then
    echo "U2 must retain one AIV0 builder and an observable volatile UBUF-to-ordinary-GM transport." >&2
    exit 1
fi
if ! grep -Fq 'u2::SimtCopyPayloadWordsToGm(staging_payload, payload, destination)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'inline void SimtCopyPayloadWordsToGm(' "$TRANSPORT_HEADER" ||
   ! grep -Fq 'const uint32_t requested_words = static_cast<uint32_t>(destination[kProbeMaxWords]);' \
       "$TRANSPORT_PROBE_SOURCE" ||
   ! grep -Fq 'u2::SimtCopyPayloadWordsToGm(staging, destination, written_words)' \
       "$TRANSPORT_PROBE_SOURCE" ||
   ! grep -Fq 'u2::PayloadChecksumSeed(nonce, task_id, words)' "$HOST_SOURCE" ||
   ! grep -Fq 'u2::FoldPayloadChecksum(checksum, state.tasks[task_id].exec.payload.words[word])' "$HOST_SOURCE"; then
    echo "U2 must reuse the copy-only transport helper and independently verify the public checksum formula." >&2
    exit 1
fi
if ! grep -Fq 'for (uint32_t attempt = 0U; attempt < kU2AtomicCasAttemptLimit' "$KERNEL_SOURCE" ||
   ! grep -Fq 'asc_atomic_max(global_max_busy, current_busy)' "$KERNEL_SOURCE" ||
   ! grep -Fq '__attribute__((noinline)) bool SimtPrepareTask(' "$KERNEL_SOURCE" ||
   ! grep -Fq 'SimtU2ReleaseSlot(u2_staging, fatal, task_id, task_id / kTasksPerBatch)' "$KERNEL_SOURCE"; then
    echo "U2 must bound remaining CAS loops, use native atomic-max, keep prepare out of the divergent caller and preserve exact slot cleanup on failure." >&2
    exit 1
fi

ubuf_store_line="$(grep -nF 'staging_payload[destination++] =' "$KERNEL_SOURCE" | head -1 | cut -d: -f1)"
ubuf_complete_line="$(grep -nF 'bool guards_valid = true;' "$KERNEL_SOURCE" | cut -d: -f1)"
gm_store_line="$(grep -nF 'u2::SimtCopyPayloadWordsToGm(staging_payload, payload, destination)' \
    "$KERNEL_SOURCE" | cut -d: -f1)"
gm_complete_line="$(grep -nF 'asc_atomic_add(u2_staging + kU2GmWordsOffsetWords, destination)' \
    "$KERNEL_SOURCE" | cut -d: -f1)"
commit_line="$(grep -nF 'if (!SimtCommitTask(' "$KERNEL_SOURCE" | cut -d: -f1)"
release_line="$(grep -nF 'if (!SimtU2ReleaseSlot(u2_staging, fatal, task_id, task_id / kTasksPerBatch))' \
    "$KERNEL_SOURCE" | cut -d: -f1)"
report_phase_line="$(grep -nF 'static_cast<uint64_t>(u2::kExpectedTransportPhaseBits) << 32U' \
    "$KERNEL_SOURCE" | cut -d: -f1)"
if [[ -z "$ubuf_store_line" || -z "$ubuf_complete_line" || -z "$gm_store_line" || -z "$gm_complete_line" ||
      -z "$commit_line" || -z "$release_line" || -z "$report_phase_line" ]] ||
   ! (( ubuf_store_line < ubuf_complete_line && ubuf_complete_line < gm_store_line &&
         gm_store_line < gm_complete_line && gm_complete_line < commit_line && commit_line < release_line &&
         release_line < report_phase_line )); then
    echo "U2 order must remain UBUF stage -> GM copy -> strict commit/BUILT -> slot release." >&2
    exit 1
fi
if ! grep -Fq 'u2-global-max-busy-depth' "$HOST_SOURCE" ||
   ! grep -Fq 'u2-inactive-report-mutated' "$HOST_SOURCE" ||
   ! grep -Fq 'u2-report-payload-checksum' "$HOST_SOURCE" ||
   ! grep -Fq 'payload-tail-mutated' "$HOST_SOURCE"; then
    echo "U2 host must independently validate B1/B256 slot reuse, reports, checksum and payload tail." >&2
    exit 1
fi

echo "[BUILD] CCEC U2 AIC Cube executors (dav-c310-cube)"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" --cce-aicore-arch=dav-c310-cube -o "$AIC_OBJECT" "$KERNEL_SOURCE"

echo "[BUILD] CCEC U2 AIV0 SIMT UBUF builder and Vector executors (dav-c310-vec)"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" --cce-aicore-arch=dav-c310-vec -o "$AIV_OBJECT" "$KERNEL_SOURCE"

echo "[BUILD] CCEC U2 optimized bitcode inventory"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" --cce-aicore-arch=dav-c310-cube \
    -Xclang -emit-llvm-bc -o "$AIC_BITCODE" "$KERNEL_SOURCE"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" --cce-aicore-arch=dav-c310-vec \
    -Xclang -emit-llvm-bc -o "$AIV_BITCODE" "$KERNEL_SOURCE"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" --cce-aicore-arch=dav-c310-vec \
    -Xclang -emit-llvm-bc -o "$TRANSPORT_PROBE_BITCODE" "$TRANSPORT_PROBE_SOURCE"
"$LLVM_BCANALYZER" -dump "$AIC_BITCODE" > "$AIC_BITCODE_DUMP"
"$LLVM_BCANALYZER" -dump "$AIV_BITCODE" > "$AIV_BITCODE_DUMP"
"$LLVM_BCANALYZER" -dump "$TRANSPORT_PROBE_BITCODE" > "$TRANSPORT_PROBE_DUMP"

required_aiv_symbols=(
    "U2SimtBuildTasks"
    "PU3AS6V"
    "llvm.hivm.store.vfsimt.info"
    "llvm.hivm.get.TID.X"
    "llvm.hivm.atom.CAS.G.u64"
    "llvm.hivm.atom.ADD.G.u64"
    "llvm.hivm.atom.MAX.G.u64"
    "llvm.hivm.fence.workitems"
    "llvm.hivm.DCCI.DST"
    "llvm.hivm.DSB"
    "llvm.hivm.vldsx1.v64f32"
    "llvm.hivm.vadd.s.x.v64f32"
    "llvm.hivm.vmul.s.x.v64f32"
    "llvm.hivm.vstsx1.v64f32"
    "llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV"
    "llvm.hivm.SET.FLAG.IMM"
    "llvm.hivm.WAIT.FLAG.IMM"
)
for symbol in "${required_aiv_symbols[@]}"; do
    if ! grep -Fq "$symbol" "$AIV_BITCODE_DUMP"; then
        echo "optimized U2 AIV bitcode is missing: $symbol" >&2
        exit 1
    fi
done
volatile_loads="$(grep -c '<INST_LOAD.*op3=1' "$AIV_BITCODE_DUMP" || true)"
volatile_stores="$(grep -c '<INST_STORE.*op3=1' "$AIV_BITCODE_DUMP" || true)"
if (( volatile_loads < 1 || volatile_stores < 2 )); then
    echo "optimized U2 bitcode lost the volatile AS6 UBUF roundtrip: loads=$volatile_loads stores=$volatile_stores" >&2
    exit 1
fi

required_transport_probe_symbols=(
    "U2PayloadTransportIrProbe"
    "PU3AS6V"
    "llvm.hivm.store.vfsimt.info"
    "llvm.hivm.get.TID.X"
    "llvm.hivm.fence.workitems"
)
for symbol in "${required_transport_probe_symbols[@]}"; do
    if ! grep -Fq "$symbol" "$TRANSPORT_PROBE_DUMP"; then
        echo "optimized U2 transport-only bitcode is missing: $symbol" >&2
        exit 1
    fi
done
transport_volatile_loads="$(grep -c '<INST_LOAD.*op3=1' "$TRANSPORT_PROBE_DUMP" || true)"
transport_volatile_stores="$(grep -c '<INST_STORE.*op3=1' "$TRANSPORT_PROBE_DUMP" || true)"
transport_gm_loads="$(grep -c '<INST_LOAD.*op3=0' "$TRANSPORT_PROBE_DUMP" || true)"
transport_gm_stores="$(grep -c '<INST_STORE.*op3=0' "$TRANSPORT_PROBE_DUMP" || true)"
if (( transport_volatile_loads != 1 || transport_volatile_stores != 1 ||
      transport_gm_loads < 1 || transport_gm_stores < 1 )); then
    echo "optimized U2 dynamic transport changed: volatile-load/store=$transport_volatile_loads/$transport_volatile_stores ordinary-load/store=$transport_gm_loads/$transport_gm_stores" >&2
    exit 1
fi
if rg -n 'llvm\.hivm\.(MOV\.UB\.TO\.OUT|SET\.LOOP[^ ]*UBTOOUT|[^ ]*MTE3)' "$TRANSPORT_PROBE_DUMP"; then
    echo "optimized U2 payload helper unexpectedly contains an MTE3/UBTOOUT intrinsic." >&2
    exit 1
fi

required_aic_symbols=(
    "RunG0CubeMatmul"
    "llvm.hivm.atom.CAS.G.s64"
    "llvm.hivm.atom.ADD.G.s64"
    "llvm.hivm.DCCI.DST"
    "llvm.hivm.MOV.OUT.TO.L1.MULTI.ND2NZ.U32.V310"
    "llvm.hivm.LOAD.L1.TO.L0A.2Dv2.f32"
    "llvm.hivm.LOAD.L1.TO.L0B.2Dv2.f32"
    "llvm.hivm.MAD.f322f32.c310"
    "llvm.hivm.FIX.L0C.TO.OUT.f32.EXT"
)
for symbol in "${required_aic_symbols[@]}"; do
    if ! grep -Fq "$symbol" "$AIC_BITCODE_DUMP"; then
        echo "optimized U2 AIC bitcode is missing: $symbol" >&2
        exit 1
    fi
done
echo "[CHECK] transport-only bitcode retains volatile AS6 UBUF with no MTE3/UBTOOUT"
echo "[CHECK] full bitcode retains SIMT atomics/fence, DCCI, Vector/Cube workloads and legal workload MTE3"

echo "[BUILD] Static U2 1:2 mixed AICore ELF"
"$LD_LLD" -m aicorelinux -Ttext=0 -static -o "$KERNEL_ELF" "$AIC_OBJECT" "$AIV_OBJECT"

SYMBOLS="$("$READELF_BIN" --symbols --wide --sym-base=10 "$KERNEL_ELF")"
SECTIONS="$("$READELF_BIN" --sections --wide "$KERNEL_ELF")"
RELOCATIONS="$("$READELF_BIN" --relocs --wide "$KERNEL_ELF")"
for entry in "$AIC_ENTRY" "$AIV_ENTRY"; do
    if ! awk -v name="$entry" \
        '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
         END {exit count != 1}' <<<"$SYMBOLS"; then
        echo "final U2 ELF must define one non-empty GLOBAL entry: $entry" >&2
        exit 1
    fi
    if [[ "$SECTIONS" != *".ascend.meta.$entry"* ]]; then
        echo "final U2 ELF is missing metadata: .ascend.meta.$entry" >&2
        exit 1
    fi
done
global_functions="$(awk '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" {print $NF}' <<<"$SYMBOLS")"
expected_global_functions="$(printf '%s\n%s\n' "$AIC_ENTRY" "$AIV_ENTRY")"
if [[ "$global_functions" != "$expected_global_functions" ]]; then
    echo "final U2 ELF exports unexpected GLOBAL functions:" >&2
    printf '%s\n' "$global_functions" >&2
    exit 1
fi
for local_pattern in 'U2SimtBuildTasks.*_simt_entry$' 'SimtPrepareTask' 'RunG0VectorAdd' 'RunG0VectorMultiply' 'RunG0CubeMatmul'; do
    if ! awk -v pattern="$local_pattern" \
        '$4 == "FUNC" && $5 == "LOCAL" && $3 + 0 > 0 && $NF ~ pattern {count++}
         END {exit count != 1}' <<<"$SYMBOLS"; then
        echo "final U2 ELF must retain one non-empty LOCAL function matching: $local_pattern" >&2
        exit 1
    fi
done
undefined_globals="$(awk '$5 == "GLOBAL" && $7 == "UND" {print $NF}' <<<"$SYMBOLS")"
if [[ -n "$undefined_globals" ]]; then
    echo "final U2 ELF contains undefined GLOBAL symbols:" >&2
    printf '%s\n' "$undefined_globals" >&2
    exit 1
fi
if [[ "$RELOCATIONS" != *"There are no relocations"* ]]; then
    echo "final U2 ELF unexpectedly retains relocations." >&2
    exit 1
fi

METADATA_OUTPUT="$(python3 -m msobjdump -d "$KERNEL_ELF")"
if [[ "$(grep -Fc 'KERNEL_TYPE: MIX_AIC_MAIN' <<<"$METADATA_OUTPUT")" -ne 2 ||
      "$(grep -Fc 'MIX_TASK_RATION: [1:2]' <<<"$METADATA_OUTPUT")" -ne 2 ]]; then
    echo "U2 mixed metadata is not an exact pair of MIX_AIC_MAIN [1:2] entries:" >&2
    printf '%s\n' "$METADATA_OUTPUT" >&2
    exit 1
fi
AIV_META_HEX="$("$READELF_BIN" -x ".ascend.meta.$AIV_ENTRY" "$KERNEL_ELF")"
read -r COMPILER_ALLOC_UB_BYTES SU_STACK_BYTES SIMT_WARP_STACK_BYTES SIMT_DVG_STACK_BYTES < <(
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
        raise SystemExit("truncated AIV metadata TLV")
    values[tag] = int.from_bytes(blob[offset : offset + size], "little")
    offset += size

print(*(values.get(tag, -1) for tag in (7, 8, 9, 10)))
PY
)
if [[ "$AIV_META_HEX" != *"0c000400 04000000"* ]] ||
   (( COMPILER_ALLOC_UB_BYTES != 0x4000 || SU_STACK_BYTES <= 0 || SU_STACK_BYTES > 0x1000 ||
      SIMT_WARP_STACK_BYTES <= 0 || SIMT_WARP_STACK_BYTES > 0x1000 ||
      SIMT_DVG_STACK_BYTES <= 0 || SIMT_DVG_STACK_BYTES >= 0x1000 ||
      SIMT_WARP_STACK_BYTES > 1536 || SIMT_DVG_STACK_BYTES > 1536 )); then
    echo "U2 AIV metadata must encode SIMD_SIMT_MIX_VF=4, fit its 16 KiB compiler UB and fit the 1536/1536 B ACL-init SIMT/DVG stack config." >&2
    printf '%s\n' "$AIV_META_HEX" >&2
    exit 1
fi
UBUF_REGION_BYTES=$((4 * 18 * 64))
VECTOR_UB_BYTES=$((0x20000 + 128 * 128 * 4))
MAX_LOCAL_BYTES=$((224 * 1024))
if (( UBUF_REGION_BYTES != 4608 ||
      UBUF_REGION_BYTES + SU_STACK_BYTES + SIMT_WARP_STACK_BYTES > COMPILER_ALLOC_UB_BYTES ||
      VECTOR_UB_BYTES != 192 * 1024 || VECTOR_UB_BYTES + COMPILER_ALLOC_UB_BYTES > MAX_LOCAL_BYTES )); then
    echo "U2 must fit staging + SU/SIMT stacks in compiler UB and preserve the 192+16/224 KiB AIV budget." >&2
    exit 1
fi
echo "[CHECK] stack metadata: SU=$SU_STACK_BYTES SIMT=$SIMT_WARP_STACK_BYTES DVG=$SIMT_DVG_STACK_BYTES bytes"
echo "[CHECK] ACL-init stack config: SIMT=1536 DVG=1536 bytes"
echo "[CHECK] ELF has exact entries/functions/metadata and 4608 B staging in 16 KiB compiler UB, 192+16/224 KiB local budget"

echo "[BUILD] GCC 15 U2 ACL host ($("$GXX15" -dumpfullversion))"
"$GXX15" -O2 -std=c++17 -Wall -Wextra -Werror \
    -DSIMT_CROSS_CORE_U2 \
    -Wno-deprecated-declarations \
    -I"$GM_ROOT/common" \
    -I"$SIMT_ROOT/common" \
    -I"$UBUF_ROOT/common" \
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
    echo "U2 build did not produce the required kernel and host artifacts." >&2
    exit 1
fi

manifest_tmp="$(mktemp "$BUILD_DIR/u2_build_manifest.XXXXXX")"
trap 'rm -f -- "$manifest_tmp"' EXIT
(
    cd "$SIMT_ROOT"
    sha256sum \
        "${U2_BUILD_INPUTS[@]}" \
        ubuf/build/ccec/simt_cross_core_u2_kernel.o \
        ubuf/build/ccec/simt_cross_core_u2_host
) > "$manifest_tmp"
mv -f -- "$manifest_tmp" "$BUILD_MANIFEST"
trap - EXIT

echo "[BUILD] U2 CCEC complete"
echo "[BUILD] kernel: $KERNEL_ELF"
echo "[BUILD] host:   $HOST_BINARY"
echo "[BUILD] manifest: $BUILD_MANIFEST"
