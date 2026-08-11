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
BUILD_DIR="$SCRIPT_DIR/build"
SOURCE="$SCRIPT_DIR/kernel.cpp"
DIRECT_SOURCE="$SCRIPT_DIR/direct_macro_only_failure.cpp"
FULL_BITCODE="$BUILD_DIR/full_build_probe.bc"
FULL_BITCODE_DUMP="$BUILD_DIR/full_build_probe.bc.dump"
SHELL_OBJECT="$BUILD_DIR/full_build_probe_shell.o"
SHELL_ELF="$BUILD_DIR/full_build_probe_shell.elf"
ENTRY="aicpu_plan_simt_full_build_probe_0_mix_aiv"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set." >&2
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
for tool in "$CCEC" "$LD_LLD" "$LLVM_BCANALYZER"; do
    if [[ ! -x "$tool" ]]; then
        echo "required executable is missing: $tool" >&2
        exit 1
    fi
done
if ! command -v "$READELF_BIN" >/dev/null 2>&1; then
    echo "readelf is required." >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

FLAGS=(
    -c -O3 -g -x cce -Wall -Werror -std=c++17
    --cce-aicore-only
    --cce-aicore-arch=dav-c310-vec
    -DPTO_FDWIC_SHARED_MAP=1
    -DPTO_FDWIC_TENSORMAP_RING_CAP=128
    -DPTO_FDWIC_SHARED_INSERT_TURN_GROUPS=1
    -DPA_BUILD_SWIMLANE=0
    -DPA_BUILD_ATOMIC_SWIMLANE=0
    -DPA_BUILD_COMPACT_GENERIC_TRACE=0
    -DPA_BUILD_SUBMIT_PMU=0
    -DPA_BUILD_PERF_CLOCK=1
    -DPA_SUBMIT_PMU_PHASE_ID=0
    -Wno-logical-op-parentheses
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
)

expect_ccec_failure() {
    local label="$1"
    local source="$2"
    local object="$3"
    local log="$4"
    local expected="$5"
    shift 5
    set +e
    "$CCEC" "${FLAGS[@]}" "$@" -o "$object" "$source" >"$log" 2>&1
    local status=$?
    set -e
    if [[ $status -eq 0 ]]; then
        echo "$label unexpectedly compiled; update the probe conclusion." >&2
        exit 1
    fi
    if ! grep -Fq "$expected" "$log"; then
        echo "$label failed for an unexpected reason:" >&2
        tail -n 40 "$log" >&2
        exit 1
    fi
    echo "[EXPECTED BLOCKER] $label: $expected"
}

compile_control() {
    local label="$1"
    local object="$2"
    shift 2
    "$CCEC" "${FLAGS[@]}" "$@" -o "$object" "$SOURCE"
    echo "[CONTROL PASS] $label"
}

echo "[CHECK] probe retains the production full-Build and ordinary writer calls"
required_source=(
    '#include "../../scalar_build/common/pa_scheduler_core.h"'
    '#define PA_DEVICE \'
    '#define PA_DEVICE_NOINLINE \'
    'static __simt_vf__ __aicore__ LAUNCH_BOUND(kBuilderThreads) void'
    'pa_scheduler::BuildRuntimePlanTask<SimtOps, false>('
    'pa_scheduler::PublishSharedTaskWriterMetadata<SimtOps>('
    'delta.ordinary_count = 1U;'
    'cce::async_invoke<BuildCanonicalPlanTaskWithScalarImplementation>('
    'wait_flag(PIPE_V, PIPE_S, EVENT_ID0);'
)
for pattern in "${required_source[@]}"; do
    if ! grep -Fq "$pattern" "$SOURCE"; then
        echo "full-build probe source is missing: $pattern" >&2
        exit 1
    fi
done
for forbidden in 'task_id % 5' 'FullPaTaskPlan' 'SimtPrepareTask'; do
    if grep -Fq "$forbidden" "$SOURCE"; then
        echo "full-build probe contains a forbidden PA shortcut: $forbidden" >&2
        exit 1
    fi
done

expect_ccec_failure \
    "only PA_DEVICE/PA_DEVICE_NOINLINE are changed" \
    "$DIRECT_SOURCE" "$BUILD_DIR/direct_macro_only_failure.o" \
    "$BUILD_DIR/direct_macro_only_failure.log" \
    "simt_callee function can only be called by simt_vf/simt_callee function"
for callback in MakeCallbackQueryView InitCreateInfo MakeCallbackOutputView; do
    if ! grep -Fq "$callback" "$BUILD_DIR/direct_macro_only_failure.log"; then
        echo "direct macro failure no longer names callback $callback." >&2
        exit 1
    fi
done

echo "[BUILD] optimized full Build LLVM bitcode after probe-only header overlay"
"$CCEC" "${FLAGS[@]}" -Xclang -emit-llvm-bc \
    -o "$FULL_BITCODE" "$SOURCE"
"$LLVM_BCANALYZER" -dump "$FULL_BITCODE" > "$FULL_BITCODE_DUMP"
required_ir=(
    'BuildCanonicalPlanTaskWithScalarImplementation'
    'llvm.hivm.store.vfsimt.info'
    'llvm.hivm.get.TID.X'
    'llvm.hivm.atom.ADD.G.s64'
    'llvm.hivm.atom.CAS.G.s64'
    'llvm.hivm.atom.MAX.G.s64'
    'llvm.hivm.DCCI.DST'
    'llvm.hivm.fence.workitems'
    'llvm.hivm.stg.uncache.b64'
    'llvm.hivm.SET.FLAG.IMM'
    'llvm.hivm.WAIT.FLAG.IMM'
)
for symbol in "${required_ir[@]}"; do
    if ! grep -Fq "$symbol" "$FULL_BITCODE_DUMP"; then
        echo "optimized full-build bitcode is missing: $symbol" >&2
        exit 1
    fi
done
echo "[IR PASS] full Build reaches optimized VF IR with atomic/DCCI/mixed wait"

expect_ccec_failure \
    "full Build VF machine code" \
    "$SOURCE" "$BUILD_DIR/full_build_probe.o" \
    "$BUILD_DIR/full_build_probe.log" \
    "ERROR: error pointer address space cast"
expect_ccec_failure \
    "full Build with real simt_callee noinline helpers" \
    "$SOURCE" "$BUILD_DIR/full_build_real_noinline.o" \
    "$BUILD_DIR/full_build_real_noinline.log" \
    "ERROR: error pointer address space cast" \
    -DPA_FULL_BUILD_PROBE_REAL_NOINLINE=1

compile_control \
    "Runtime Plan acquire/decode" \
    "$BUILD_DIR/plan_decode_probe.o" \
    -DPA_FULL_BUILD_PROBE_PLAN_DECODE_ONLY=1

expect_ccec_failure \
    "TaskArgs writer delta with runtime local/GM TensorDesc" \
    "$SOURCE" "$BUILD_DIR/writer_delta_dynamic_address_space.o" \
    "$BUILD_DIR/writer_delta_dynamic_address_space.log" \
    "ERROR: error pointer address space cast" \
    -DPA_FULL_BUILD_PROBE_WRITER_DELTA_DYNAMIC_ADDRESS_SPACE_ONLY=1
expect_ccec_failure \
    "TaskArgs writer delta forced to GM TensorDesc" \
    "$SOURCE" "$BUILD_DIR/writer_delta_gm_only.o" \
    "$BUILD_DIR/writer_delta_gm_only.log" \
    "Copy one register into another with a different width" \
    -DPA_FULL_BUILD_PROBE_WRITER_DELTA_GM_ONLY=1

expect_ccec_failure \
    "Exec payload with runtime local/GM TensorDesc" \
    "$SOURCE" "$BUILD_DIR/exec_dynamic_address_space.o" \
    "$BUILD_DIR/exec_dynamic_address_space.log" \
    "ERROR: error pointer address space cast" \
    -DPA_FULL_BUILD_PROBE_EXEC_DYNAMIC_ADDRESS_SPACE_ONLY=1
compile_control \
    "Exec payload with a fixed GM TensorDesc reference" \
    "$BUILD_DIR/exec_reference_probe.o" \
    -DPA_FULL_BUILD_PROBE_EXEC_REFERENCE_ONLY=1

compile_control \
    "ordinary_count=1 PublishSharedTaskWriterMetadata chain" \
    "$BUILD_DIR/writer_metadata_probe.o" \
    -DPA_FULL_BUILD_PROBE_WRITER_METADATA_ONLY=1

compile_control \
    "static VF plus mixed AIV async/wait shell" \
    "$SHELL_OBJECT" \
    -DPA_FULL_BUILD_PROBE_SHELL_ONLY=1
"$LD_LLD" -m aicorelinux -Ttext=0 -static \
    -o "$SHELL_ELF" "$SHELL_OBJECT"

SYMBOLS="$("$READELF_BIN" --symbols --wide --sym-base=10 "$SHELL_ELF")"
if ! awk -v name="$ENTRY" \
    '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
     END {exit count != 1}' <<<"$SYMBOLS"; then
    echo "shell ELF must export exactly one non-empty GLOBAL $ENTRY." >&2
    exit 1
fi
if ! awk \
    '$4 == "FUNC" && $5 == "LOCAL" && $3 + 0 > 0 &&
     $NF ~ /BuildCanonicalPlanTaskWithScalarImplementation.*_simt_entry$/ {count++}
     END {exit count != 1}' <<<"$SYMBOLS"; then
    echo "shell ELF must retain exactly one non-empty LOCAL SIMT entry." >&2
    exit 1
fi
global_functions="$(awk '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" {print $NF}' <<<"$SYMBOLS")"
if [[ "$global_functions" != "$ENTRY" ]]; then
    echo "shell ELF exports unexpected GLOBAL functions:" >&2
    printf '%s\n' "$global_functions" >&2
    exit 1
fi

META_HEX="$("$READELF_BIN" -x ".ascend.meta.$ENTRY" "$SHELL_ELF")"
if [[ "$META_HEX" != *'0c000400 04000000'* ]]; then
    echo "AIV metadata must encode SIMD_SIMT_MIX_VF=4." >&2
    printf '%s\n' "$META_HEX" >&2
    exit 1
fi

echo "[STATIC PASS] one LOCAL VF, one GLOBAL entry, MIX_VF=4"
echo "[RESULT] direct full Build reuse is BLOCKED before a runnable object"
echo "[RESULT] ordinary_count=1 metadata publication itself is VF-codegen capable"
echo "[RESULT] see README.md and $BUILD_DIR/*.log for isolated blockers"
