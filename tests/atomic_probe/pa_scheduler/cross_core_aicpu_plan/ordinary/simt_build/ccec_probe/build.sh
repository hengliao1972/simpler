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
KERNEL_SOURCE="$SCRIPT_DIR/kernel.cpp"
DEVICE_OBJECT="$BUILD_DIR/aicpu_plan_simt_v2_probe_aiv.o"
DEVICE_BITCODE="$BUILD_DIR/aicpu_plan_simt_v2_probe_aiv.bc"
BITCODE_DUMP="$BUILD_DIR/aicpu_plan_simt_v2_probe_aiv.bc.dump"
KERNEL_ELF="$BUILD_DIR/aicpu_plan_simt_v2_probe_kernel.o"
ENTRY="aicpu_plan_simt_v2_probe_0_mix_aiv"

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

for tool in "$CCEC" "$LD_LLD" "$LLVM_BCANALYZER"; do
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

DEVICE_FLAGS=(
    -c -O3 -g -x cce -Wall -Werror -std=c++17
    --cce-aicore-only
    --cce-aicore-arch=dav-c310-vec
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

echo "[CHECK] canonical Plan-v2 source contract"
required_source_patterns=(
    '#include "../common/simt_plan_build_protocol.h"'
    'static __simt_vf__ __aicore__ LAUNCH_BOUND(128) void ConsumeCanonicalPlanV2('
    'simt_contract::kBuilderThreads'
    'simt_contract::kWarpSize'
    'simt_contract::kBuilderLeaders'
    'lane != 0U'
    'const uint64_t closed_task_count = SimtAtomicObserve('
    'first_control = SimtAtomicObserve(&cell->control.value);'
    'for (uint32_t line = 0U; line < published_lines; ++line)'
    'asc_dcci_single(static_cast<__gm__ void *>('
    'second_control = SimtAtomicObserve(&cell->control.value);'
    'abi_version == plan::kRuntimePlanAbiVersion'
    'expected_lines == published_lines'
    'cce::async_invoke<ConsumeCanonicalPlanV2>('
    'set_flag(PIPE_V, PIPE_S, EVENT_ID0);'
    'wait_flag(PIPE_V, PIPE_S, EVENT_ID0);'
    'PublishLeaderReport(leader_reports, warp, 0U, kReportMagic);'
)
for pattern in "${required_source_patterns[@]}"; do
    if ! grep -Fq "$pattern" "$KERNEL_SOURCE"; then
        echo "canonical SIMT Plan source is missing: $pattern" >&2
        exit 1
    fi
done

forbidden_source_patterns=(
    'task_id % 5'
    'task_id%5'
    'FullPaTaskPlan'
    'TaskKind'
    'SimtPrepareTask'
    'SimtBuildRequest'
    'RequestCell'
    'HostTaskPlan'
    'host_plan'
    'TakeBuildTicket('
    'TakeClosedPlanBuildTicket('
    'build_next'
)
for pattern in "${forbidden_source_patterns[@]}"; do
    if grep -Fq "$pattern" "$KERNEL_SOURCE"; then
        echo "canonical SIMT Plan source contains forbidden shortcut/second ABI: $pattern" >&2
        exit 1
    fi
done

echo "[SCOPE] compile gate only: dynamic Build ticket acquisition is intentionally outside this probe"
echo "[SCOPE] asc_dcci_single + asc_threadfence lowering is not A5 memory-model evidence"

echo "[BUILD] CCEC AIV object"
"$CCEC" "${DEVICE_FLAGS[@]}" -o "$DEVICE_OBJECT" "$KERNEL_SOURCE"

echo "[BUILD] CCEC optimized SIMT bitcode"
"$CCEC" "${DEVICE_FLAGS[@]}" -Xclang -emit-llvm-bc \
    -o "$DEVICE_BITCODE" "$KERNEL_SOURCE"
"$LLVM_BCANALYZER" -dump "$DEVICE_BITCODE" > "$BITCODE_DUMP"

required_bitcode_symbols=(
    'ConsumeCanonicalPlanV2'
    'SimdMetadataAnchor'
    'llvm.hivm.store.vfsimt.info'
    'llvm.hivm.get.TID.X'
    'llvm.hivm.atom.ADD.G.u64'
    'llvm.hivm.DCCI.DST'
    'llvm.hivm.fence.workitems'
    'llvm.hivm.SET.FLAG.IMM'
    'llvm.hivm.WAIT.FLAG.IMM'
    'llvm.hivm.ldg.uncache.s64'
    'llvm.hivm.stg.uncache.b64'
)
for symbol in "${required_bitcode_symbols[@]}"; do
    if ! grep -Fq "$symbol" "$BITCODE_DUMP"; then
        echo "optimized Plan-v2 SIMT bitcode is missing: $symbol" >&2
        exit 1
    fi
done
echo "[CHECK] bitcode contains SIMT launch, atomic observe, per-line DCCI, leader reports and V/S completion"

echo "[BUILD] AIV-only mixed ELF"
"$LD_LLD" -m aicorelinux -Ttext=0 -static \
    -o "$KERNEL_ELF" "$DEVICE_OBJECT"

SYMBOLS="$("$READELF_BIN" --symbols --wide --sym-base=10 "$KERNEL_ELF")"
SECTIONS="$("$READELF_BIN" --sections --wide "$KERNEL_ELF")"
RELOCATIONS="$("$READELF_BIN" --relocs --wide "$KERNEL_ELF")"

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
if ! awk '$4 == "FUNC" && $5 == "LOCAL" && $3 + 0 > 0 && $NF ~ /ConsumeCanonicalPlanV2.*_simt_entry$/ {count++}
          END {exit count != 1}' <<<"$SYMBOLS"; then
    echo "final ELF must retain exactly one non-empty LOCAL canonical Plan SIMT entry." >&2
    exit 1
fi
undefined_globals="$(awk '$5 == "GLOBAL" && $7 == "UND" {print $NF}' <<<"$SYMBOLS")"
if [[ -n "$undefined_globals" ]]; then
    echo "final ELF contains undefined GLOBAL symbols:" >&2
    printf '%s\n' "$undefined_globals" >&2
    exit 1
fi
if [[ "$RELOCATIONS" != *"There are no relocations"* ]]; then
    echo "final ELF unexpectedly retains relocations." >&2
    exit 1
fi
metadata_sections="$(awk '{for (i=1; i<=NF; ++i) if ($i ~ /^\.ascend\.meta\./) print $i}' <<<"$SECTIONS")"
if [[ "$metadata_sections" != ".ascend.meta.$ENTRY" ]]; then
    echo "final ELF metadata sections are not exact: $metadata_sections" >&2
    exit 1
fi

METADATA_OUTPUT="$(python3 -m msobjdump -d "$KERNEL_ELF")"
if [[ "$METADATA_OUTPUT" != *'KERNEL_TYPE: MIX_AIC_MAIN'* ||
      "$METADATA_OUTPUT" != *'MIX_TASK_RATION: [1:2]'* ]]; then
    echo "Plan-v2 probe metadata is not MIX_AIC_MAIN [1:2]:" >&2
    printf '%s\n' "$METADATA_OUTPUT" >&2
    exit 1
fi
AIV_META_HEX="$("$READELF_BIN" -x ".ascend.meta.$ENTRY" "$KERNEL_ELF")"
if [[ "$AIV_META_HEX" != *'0c000400 04000000'* ||
      "$AIV_META_HEX" != *'07000400 00200000'* ]]; then
    echo "AIV metadata must encode SIMD_SIMT_MIX_VF=4 and an 8 KiB VF stack reserve." >&2
    printf '%s\n' "$AIV_META_HEX" >&2
    exit 1
fi

if [[ ! -s "$DEVICE_BITCODE" || ! -s "$KERNEL_ELF" ]]; then
    echo "canonical Plan-v2 SIMT compile probe did not produce required artifacts." >&2
    exit 1
fi

echo "[CHECK] ELF exports one AIV entry, retains one local SIMT entry and encodes SIMD_SIMT_MIX_VF=4"
echo "[BUILD] canonical Plan-v2 SIMT CCEC compile gate complete"
echo "[BUILD] bitcode: $DEVICE_BITCODE"
echo "[BUILD] kernel:  $KERNEL_ELF"
