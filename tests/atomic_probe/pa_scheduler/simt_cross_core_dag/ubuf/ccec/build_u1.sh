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
BUILD_DIR="$UBUF_ROOT/build/ccec"
KERNEL_SOURCE="$SCRIPT_DIR/u1_multi_slot_kernel.cpp"
HOST_SOURCE="$SCRIPT_DIR/u1_multi_slot_host.cpp"
COMMON_HEADER="$UBUF_ROOT/common/u1_multi_slot.h"
AIC_OBJECT="$BUILD_DIR/simt_cross_core_u1_aic.o"
AIV_OBJECT="$BUILD_DIR/simt_cross_core_u1_aiv.o"
AIC_BITCODE="$BUILD_DIR/simt_cross_core_u1_aic.bc"
AIV_BITCODE="$BUILD_DIR/simt_cross_core_u1_aiv.bc"
AIC_BITCODE_DUMP="$BUILD_DIR/simt_cross_core_u1_aic.bc.dump"
AIV_BITCODE_DUMP="$BUILD_DIR/simt_cross_core_u1_aiv.bc.dump"
KERNEL_ELF="$BUILD_DIR/simt_cross_core_u1_kernel.o"
HOST_BINARY="$BUILD_DIR/simt_cross_core_u1_host"
BUILD_MANIFEST="$BUILD_DIR/u1_build_manifest.sha256"
AIC_ENTRY="simt_cross_core_u1_0_mix_aic"
AIV_ENTRY="simt_cross_core_u1_0_mix_aiv"
U1_BUILD_INPUTS=(
    run.sh
    common/full_pa_exec_protocol.h
    ubuf/common/ubuf_staging_protocol.h
    ubuf/common/u1_multi_slot.h
    ubuf/ccec/u1_multi_slot_kernel.cpp
    ubuf/ccec/u1_multi_slot_host.cpp
    ubuf/ccec/build_u1.sh
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
    -I"$UBUF_ROOT/common"
    -I"$SIMT_ROOT/common"
)

echo "[CHECK] U1 pure-SIMT builder, four volatile UBUF slots and direct-GM diagnostic transport"
if rg -n '#include.*(cross_core_ordinary|ops-nn|kernel_operator)' "$UBUF_ROOT" -g '*.h' -g '*.cpp'; then
    echo "U1 must not include cross_core_ordinary, ops-nn or AscendC implementation headers." >&2
    exit 1
fi
if ! grep -Fq 'LAUNCH_BOUND(kThreadCount) void U1SimtBuildMultiSlot(' "$KERNEL_SOURCE" ||
   ! grep -Fq 'const uint32_t warp = thread / kWarpSize;' "$KERNEL_SOURCE" ||
   ! grep -Fq 'const uint32_t lane = thread % kWarpSize;' "$KERNEL_SOURCE" ||
   ! grep -Fq 'lane != 0U' "$KERNEL_SOURCE" ||
   ! grep -Fq 'task_id = first_task; task_id < kTaskCount; task_id += kWarpCount' "$KERNEL_SOURCE" ||
   ! grep -Fq 'const uint32_t last_task = warp + kWarpCount;' "$KERNEL_SOURCE" ||
   ! grep -Fq 'cce::dim3{kThreadCount, 1U, 1U}' "$KERNEL_SOURCE" ||
   ! grep -Fq 'constexpr uint32_t kThreadCount = kWarpSize * kWarpCount;' "$COMMON_HEADER" ||
   ! grep -Fq 'constexpr uint32_t kTaskCount = 128U;' "$COMMON_HEADER" ||
   ! grep -Fq 'constexpr uint32_t kTasksPerWarp = kTaskCount / kWarpCount;' "$COMMON_HEADER"; then
    echo "U1 must launch 2048 threads and let each of 64 lane0 leaders build warp and warp+64." >&2
    exit 1
fi
if ! grep -Fq '__ubuf__ volatile uint64_t *staging_region' "$KERNEL_SOURCE" ||
   ! grep -Fq 'staging_payload[word] = SimtPayloadWord(nonce, task_id, word);' "$KERNEL_SOURCE" ||
   ! grep -Fq 'const uint64_t value = staging_payload[word];' "$KERNEL_SOURCE" ||
   ! grep -Fq 'payload[word] = value;' "$KERNEL_SOURCE" ||
   ! grep -Fq 'TransportKind::SimtUbufReadToGmWordStore' "$HOST_SOURCE"; then
    echo "U1 must retain an observable volatile UBUF store/load followed by direct GM word stores." >&2
    exit 1
fi
if grep -Fq 'for (uint32_t word = 0U; word < kMaxPayloadWords; ++word)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'kUbufSlotStrideBytes == 1152U && kUbufRegionBytes == 4608U' "$COMMON_HEADER" ||
   ! grep -Fq 'slot_region = staging_region + slot_id * kUbufSlotStrideWords' "$KERNEL_SOURCE" ||
   ! grep -Fq 'state->control.ubuf_alignment_bytes == kUbufAlignmentBytes' "$KERNEL_SOURCE" ||
   ! grep -Fq 'U1FatalReason::InvalidTaskState' "$KERNEL_SOURCE" ||
   ! grep -Fq 'ubuf_guard_check_count' "$KERNEL_SOURCE" ||
   ! grep -Fq 'DecodeU1Fatal' "$HOST_SOURCE"; then
    echo "U1 must expose its guarded UBUF ABI, write only valid payload words and fail closed with one fatal ABI." >&2
    exit 1
fi
if rg -n 'copy_ubuf_to_gm|PIPE_MTE3|TSTORE|SET\.LOOP.*UBTOOUT|MOV\.UB\.TO\.OUT' \
        "$KERNEL_SOURCE" "$COMMON_HEADER"; then
    echo "U1 diagnostic transport must not contain an MTE3/UBTOOUT implementation." >&2
    exit 1
fi
if ! grep -Fq 'constexpr uint64_t kExpectedMte3Count = 0U;' "$HOST_SOURCE" ||
   [[ "$(grep -Fc 'kExpectedMte3Count' "$HOST_SOURCE")" -lt 2 ]]; then
    echo "U1 host must independently retain an expected MTE3 count of zero." >&2
    exit 1
fi
if ! grep -Fq 'kBuilderOwner) << g0::kStateBuildOwnerShift' "$KERNEL_SOURCE"; then
    echo "all U1 task states must use physical AIV0 owner32; warp ids belong only in reports/tickets." >&2
    exit 1
fi
if ! grep -Fq '__aicore__ void RunExecutor(__gm__ U1ProbeState *state)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'AtomicCas(&state->tasks[task_id].control.state, built, claimed)' "$KERNEL_SOURCE" ||
   [[ "$(grep -Fc 'AtomicLoad(&state->fatal.value) != 0U' "$KERNEL_SOURCE")" -lt 4 ]] ||
   ! grep -Fq 'AtomicCas(&state->tasks[task_id].control.state, claimed, built)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'dcci(static_cast<__gm__ void *>(payload_bytes + line * kCacheLineBytes)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'DoneState(task_id)' "$KERNEL_SOURCE"; then
    echo "AIV1 Main Scalar must be the unique owner33 serial Claim/DCCI/validate/DONE executor." >&2
    exit 1
fi
builder_scalar_body="$(sed -n '/^__aicore__ void RunBuilder(/,/^__aicore__ void RunExecutor(/p' "$KERNEL_SOURCE" | sed '$d')"
if ! grep -Fq 'main_scalar_build_action_count == 0U' "$HOST_SOURCE" ||
   ! grep -Fq 'subblock, 0U, 0U, timeout_count' <<<"$builder_scalar_body" ||
   rg -n 'AtomicCas\(|BuildingState\(|BuiltState\(|DoneState\(|payload\.words|build_report|exec_report' \
       <<<"$builder_scalar_body"; then
    echo "U1 builder Main Scalar telemetry must stay zero and its scoped source must contain no task-build action." >&2
    exit 1
fi
if ! grep -Fq 'SimtSlotBusyState(generation, task_id)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'SimtSlotFreeState(generation + 1U)' "$KERNEL_SOURCE" ||
   grep -Fq 'SlotGenerationForTask' "$KERNEL_SOURCE" ||
   ! grep -Fq 'build_report[5] = static_cast<uint64_t>(slot_id)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'duplicate-slot-generation' "$HOST_SOURCE" ||
   ! grep -Fq 'missing-slot-generation' "$HOST_SOURCE"; then
    echo "U1 slot generations must come from runtime FREE(g) acquisition and be reported as a complete per-slot set." >&2
    exit 1
fi
if ! grep -Fq 'SimtStageAnchorIdentity(anchor_mask, task_id, &anchor_failure)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'staged == kAnchorTaskCount && mask == kAnchorStagedMask' "$KERNEL_SOURCE" ||
   ! grep -Fq '&state->anchor_staged_mask.value' "$KERNEL_SOURCE" ||
   ! grep -Fq '&state.anchor_staged_mask' "$HOST_SOURCE" ||
   ! grep -Fq 'kAnchorStagedMask' "$HOST_SOURCE"; then
    echo "U1 anchors must publish distinct identity bits and gate on exact count=4/mask=0xf." >&2
    exit 1
fi
if ! grep -Fq 'SimtResetUnpublishedTask' "$KERNEL_SOURCE" ||
   ! grep -Fq 'SimtDecrementBusy' "$KERNEL_SOURCE" ||
   ! grep -Fq 'kAtomicCasAttemptLimit' "$KERNEL_SOURCE" ||
   ! grep -Fq 'asc_atomic_add(global_busy, static_cast<uint64_t>(1U));' "$KERNEL_SOURCE"; then
    echo "U1 failure cleanup must reset unpublished tasks and preserve exact slot/global-busy accounting." >&2
    exit 1
fi

source_line() {
    local needle="$1"
    local line
    line="$(grep -nF "$needle" "$KERNEL_SOURCE" | head -n 1 | cut -d: -f1)"
    if [[ -z "$line" ]]; then
        echo "U1 source-order check cannot find: $needle" >&2
        exit 1
    fi
    printf '%s' "$line"
}

ubuf_store_line="$(source_line 'staging_payload[word] = SimtPayloadWord(nonce, task_id, word);')"
ubuf_load_line="$(source_line 'const uint64_t value = staging_payload[word];')"
gm_store_line="$(source_line 'payload[word] = value;')"
fence_line="$(source_line 'asc_threadfence();')"
built_line="$(source_line 'asc_atomic_cas(task, building, built)')"
release_line="$(grep -nF 'if (!SimtReleaseSlot(slot_state, slot_releases, global_busy, fatal, generation, task_id))' \
    "$KERNEL_SOURCE" | tail -n 1 | cut -d: -f1)"
claim_line="$(source_line 'AtomicCas(&state->tasks[task_id].control.state, built, claimed)')"
dcci_line="$(source_line 'dcci(static_cast<__gm__ void *>(payload_bytes + line * kCacheLineBytes)')"
payload_read_line="$(source_line 'const uint64_t value = state->tasks[task_id].payload.words[word];')"
done_line="$(source_line 'AtomicCas(&state->tasks[task_id].control.state, claimed, DoneState(task_id))')"
if ! (( ubuf_store_line < ubuf_load_line && ubuf_load_line < gm_store_line && gm_store_line < fence_line &&
        fence_line < built_line && built_line < release_line &&
        claim_line < dcci_line && dcci_line < payload_read_line && payload_read_line < done_line )); then
    echo "U1 source order must remain UBUF-store < UBUF-load < GM-store < fence < BUILT < release and Claim < DCCI < read < DONE." >&2
    exit 1
fi
if ! grep -Fq 'uint32_t runs = 100U;' "$HOST_SOURCE" ||
   ! grep -Fq 'inactive-lane-mutated' "$HOST_SOURCE" ||
   ! grep -Fq 'payload-tail' "$HOST_SOURCE" ||
   ! grep -Fq 'duplicate-slot-generation' "$HOST_SOURCE" ||
   ! grep -Fq 'missing-slot-generation' "$HOST_SOURCE" ||
   ! grep -Fq 'state.global_max_busy_depth' "$HOST_SOURCE"; then
    echo "U1 host must cover 100-round reuse, 1984 inactive lanes, payload tail, generations and maxbusy=4." >&2
    exit 1
fi

echo "[BUILD] CCEC U1 AIC observer (dav-c310-cube)"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" --cce-aicore-arch=dav-c310-cube \
    -o "$AIC_OBJECT" "$KERNEL_SOURCE"

echo "[BUILD] CCEC U1 AIV0 SIMT builder + AIV1 Scalar executor (dav-c310-vec)"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" --cce-aicore-arch=dav-c310-vec \
    -o "$AIV_OBJECT" "$KERNEL_SOURCE"

echo "[BUILD] CCEC U1 optimized bitcode inventory"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" --cce-aicore-arch=dav-c310-cube \
    -Xclang -emit-llvm-bc -o "$AIC_BITCODE" "$KERNEL_SOURCE"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" --cce-aicore-arch=dav-c310-vec \
    -Xclang -emit-llvm-bc -o "$AIV_BITCODE" "$KERNEL_SOURCE"
"$LLVM_BCANALYZER" -dump "$AIC_BITCODE" > "$AIC_BITCODE_DUMP"
"$LLVM_BCANALYZER" -dump "$AIV_BITCODE" > "$AIV_BITCODE_DUMP"

required_aiv_symbols=(
    "U1SimtBuildMultiSlot"
    "PU3AS6V"
    "llvm.hivm.store.vfsimt.info"
    "llvm.hivm.get.TID.X"
    "llvm.hivm.atom.CAS.G.u64"
    "llvm.hivm.atom.ADD.G.u64"
    "llvm.hivm.fence.workitems"
    "llvm.hivm.DCCI.DST"
    "llvm.hivm.DSB"
    "llvm.hivm.ST.DEV.u64"
    "llvm.hivm.SET.FLAG.IMM"
    "llvm.hivm.WAIT.FLAG.IMM"
)
for symbol in "${required_aiv_symbols[@]}"; do
    if ! grep -Fq "$symbol" "$AIV_BITCODE_DUMP"; then
        echo "optimized U1 AIV bitcode is missing: $symbol" >&2
        exit 1
    fi
done
volatile_loads="$(grep -c '<INST_LOAD.*op3=1' "$AIV_BITCODE_DUMP" || true)"
volatile_stores="$(grep -c '<INST_STORE.*op3=1' "$AIV_BITCODE_DUMP" || true)"
if (( volatile_loads < 1 || volatile_stores < 2 )); then
    echo "optimized U1 bitcode lost the volatile UBUF roundtrip: loads=$volatile_loads stores=$volatile_stores" >&2
    exit 1
fi
if rg -n 'llvm\.hivm\.(MOV\.UB\.TO\.OUT|SET\.LOOP[^ ]*UBTOOUT|[^ ]*MTE3)' "$AIV_BITCODE_DUMP"; then
    echo "optimized U1 AIV bitcode unexpectedly contains MTE3/UBTOOUT transport." >&2
    exit 1
fi
for symbol in "llvm.hivm.atom.CAS.G.s64" "llvm.hivm.atom.ADD.G.s64" "llvm.hivm.DCCI.DST" \
              "llvm.hivm.DSB" "llvm.hivm.ST.DEV.u64"; do
    if ! grep -Fq "$symbol" "$AIV_BITCODE_DUMP"; then
        echo "optimized U1 AIV Scalar executor bitcode is missing: $symbol" >&2
        exit 1
    fi
done
echo "[CHECK] bitcode retains AS6 volatile UBUF load/store and contains no MTE3/UBTOOUT intrinsic"

echo "[BUILD] Static U1 1:2 mixed AICore ELF"
"$LD_LLD" -m aicorelinux -Ttext=0 -static -o "$KERNEL_ELF" "$AIC_OBJECT" "$AIV_OBJECT"

SYMBOLS="$("$READELF_BIN" --symbols --wide --sym-base=10 "$KERNEL_ELF")"
SECTIONS="$("$READELF_BIN" --sections --wide "$KERNEL_ELF")"
RELOCATIONS="$("$READELF_BIN" --relocs --wide "$KERNEL_ELF")"
for entry in "$AIC_ENTRY" "$AIV_ENTRY"; do
    if ! awk -v name="$entry" \
        '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
         END {exit count != 1}' <<<"$SYMBOLS"; then
        echo "final U1 ELF must define one non-empty GLOBAL entry: $entry" >&2
        exit 1
    fi
    if [[ "$SECTIONS" != *".ascend.meta.$entry"* ]]; then
        echo "final U1 ELF is missing metadata: .ascend.meta.$entry" >&2
        exit 1
    fi
done
global_functions="$(awk '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" {print $NF}' <<<"$SYMBOLS")"
expected_global_functions="$(printf '%s\n%s\n' "$AIC_ENTRY" "$AIV_ENTRY")"
if [[ "$global_functions" != "$expected_global_functions" ]]; then
    echo "final U1 ELF exports unexpected GLOBAL functions:" >&2
    printf '%s\n' "$global_functions" >&2
    exit 1
fi
if ! awk '$4 == "FUNC" && $5 == "LOCAL" && $3 + 0 > 0 && $NF ~ /U1SimtBuildMultiSlot.*_simt_entry$/ {count++}
         END {exit count != 1}' <<<"$SYMBOLS"; then
    echo "final U1 ELF must retain one non-empty local SIMT builder with an AS6 UBUF argument." >&2
    exit 1
fi
undefined_globals="$(awk '$5 == "GLOBAL" && $7 == "UND" {print $NF}' <<<"$SYMBOLS")"
if [[ -n "$undefined_globals" ]]; then
    echo "final U1 ELF contains undefined GLOBAL symbols:" >&2
    printf '%s\n' "$undefined_globals" >&2
    exit 1
fi
if [[ "$RELOCATIONS" != *"There are no relocations"* ]]; then
    echo "final U1 ELF unexpectedly retains relocations." >&2
    exit 1
fi

METADATA_OUTPUT="$(python3 -m msobjdump -d "$KERNEL_ELF")"
if [[ "$(grep -Fc 'KERNEL_TYPE: MIX_AIC_MAIN' <<<"$METADATA_OUTPUT")" -ne 2 ||
      "$(grep -Fc 'MIX_TASK_RATION: [1:2]' <<<"$METADATA_OUTPUT")" -ne 2 ]]; then
    echo "U1 mixed metadata is not an exact pair of MIX_AIC_MAIN [1:2] entries:" >&2
    printf '%s\n' "$METADATA_OUTPUT" >&2
    exit 1
fi
AIV_META_HEX="$("$READELF_BIN" -x ".ascend.meta.$AIV_ENTRY" "$KERNEL_ELF")"
if [[ "$AIV_META_HEX" != *"0c000400 04000000"* || "$AIV_META_HEX" != *"07000400 00200000"* ]]; then
    echo "U1 AIV metadata must encode SIMD_SIMT_MIX_VF=4 and 8 KiB SIMT share memory." >&2
    printf '%s\n' "$AIV_META_HEX" >&2
    exit 1
fi
UBUF_REGION_BYTES=$((4 * 18 * 64))
SIMT_SHARE_BYTES=$((8 * 1024))
MAX_SHARE_BYTES=$((224 * 1024))
if (( UBUF_REGION_BYTES != 4608 || UBUF_REGION_BYTES > SIMT_SHARE_BYTES ||
      SIMT_SHARE_BYTES > MAX_SHARE_BYTES )); then
    echo "U1 guarded UBUF region must fit the compiler-emitted share memory and preserve the 32 KiB DCache floor." >&2
    exit 1
fi
echo "[CHECK] ELF has exact mixed entries; the guarded four-slot 4608 B region fits TLV7=8 KiB share memory"

echo "[BUILD] GCC 15 U1 ACL host ($("$GXX15" -dumpfullversion))"
"$GXX15" -O2 -std=c++17 -Wall -Wextra -Werror \
    -Wno-deprecated-declarations \
    -I"$UBUF_ROOT/common" \
    -I"$SIMT_ROOT/common" \
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
    echo "U1 build did not produce the required kernel and host artifacts." >&2
    exit 1
fi

manifest_tmp="$(mktemp "$BUILD_DIR/u1_build_manifest.XXXXXX")"
trap 'rm -f -- "$manifest_tmp"' EXIT
(
    cd "$SIMT_ROOT"
    sha256sum \
        "${U1_BUILD_INPUTS[@]}" \
        ubuf/build/ccec/simt_cross_core_u1_kernel.o \
        ubuf/build/ccec/simt_cross_core_u1_host
) > "$manifest_tmp"
mv -f -- "$manifest_tmp" "$BUILD_MANIFEST"
trap - EXIT

echo "[BUILD] U1 CCEC complete"
echo "[BUILD] kernel: $KERNEL_ELF"
echo "[BUILD] host:   $HOST_BINARY"
echo "[BUILD] manifest: $BUILD_MANIFEST"
