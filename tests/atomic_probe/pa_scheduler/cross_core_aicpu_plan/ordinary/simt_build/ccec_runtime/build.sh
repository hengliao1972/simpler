#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
KERNEL_SOURCE="$SCRIPT_DIR/kernel.cpp"
SCALAR_SOURCE="$SCRIPT_DIR/scalar_continuation.cpp"
REPORT_HEADER="$SCRIPT_DIR/runtime_report.h"
REAL_RUNTIME_HEADER="$SCRIPT_DIR/../common/simt_real_state_runtime.h"
VERSION_SCRIPT="$SCRIPT_DIR/exports.map"
KERNEL_OBJECT="$BUILD_DIR/runtime_kernel.o"
SCALAR_OBJECT="$BUILD_DIR/scalar_continuation.o"
KERNEL_BITCODE="$BUILD_DIR/runtime_kernel.bc"
SCALAR_BITCODE="$BUILD_DIR/scalar_continuation.bc"
KERNEL_DUMP="$BUILD_DIR/runtime_kernel.bc.dump"
SCALAR_DUMP="$BUILD_DIR/scalar_continuation.bc.dump"
ELF="$BUILD_DIR/aicpu_plan_simt_runtime.o"
ENTRY="aicpu_plan_simt_runtime_0_mix_aiv"
CONTINUATION="aicpu_plan_simt_scalar_continuation"

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
if ! python3 -c 'import msobjdump' >/dev/null 2>&1; then
    echo "the CANN msobjdump module is required." >&2
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
    -DPA_RUNTIME_PLAN_BUILD_WORKERS=4
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

echo "[CHECK] frozen four-leader dynamic Build control flow"
required_kernel_patterns=(
    'static __simt_vf__ __aicore__ LAUNCH_BOUND(128) void'
    'simt::AttachClosedPlan<SimtOps>('
    'simt::TakeAttachedBuildTicket<SimtOps>('
    'runtime.BindTask(reservation.task_id)'
    'simt::BuildCanonicalPlanTask<SimtOps>('
    'simt::SimtBuildOwner(leader)'
    'simt::ArriveBuilderLeaderOnce<SimtOps>('
    'simt::PublishBuildRelease<SimtOps>('
    'cce::async_invoke<BuildClosedCanonicalPlanVf>('
    'set_flag(PIPE_V, PIPE_S, EVENT_ID0);'
    'wait_flag(PIPE_V, PIPE_S, EVENT_ID0);'
    'aicpu_plan_simt_scalar_continuation('
    'LaunchBeginClock, begin_clock'
    'reinterpret_cast<__gm__ uint8_t *>(line),'
)
for pattern in "${required_kernel_patterns[@]}"; do
    if ! grep -Fq "$pattern" "$KERNEL_SOURCE"; then
        echo "runtime kernel is missing: $pattern" >&2
        exit 1
    fi
done

required_scalar_patterns=(
    '#define PA_DEVICE __aicore__ inline'
    'runtime_plan_control.build_release.value'
    'runtime_plan_control.fatal.value'
    '&state->fatal.value'
    'ContinuationMagic'
    '__builtin_cce_st_dev('
)
for pattern in "${required_scalar_patterns[@]}"; do
    if ! grep -Fq "$pattern" "$SCALAR_SOURCE"; then
        echo "Scalar continuation is missing: $pattern" >&2
        exit 1
    fi
done

required_runtime_adapter_patterns=(
    'static_assert(SimtBuildOwner(0U) == 0U);'
    'static_assert(SimtBuildOwner(3U) == 3U);'
    'struct SimtRealStateRuntime {'
    'bool BindTask(uint32_t task_id)'
)
for pattern in "${required_runtime_adapter_patterns[@]}"; do
    if ! grep -Fq "$pattern" "$REAL_RUNTIME_HEADER"; then
        echo "real-state Runtime adapter is missing: $pattern" >&2
        exit 1
    fi
done

python3 - "$KERNEL_SOURCE" "$REPORT_HEADER" <<'PY'
import pathlib
import sys

kernel = pathlib.Path(sys.argv[1]).read_text()
report = pathlib.Path(sys.argv[2]).read_text()

ordered = [
    "const uint64_t begin_clock = static_cast<uint64_t>(get_sys_cnt());",
    "cce::async_invoke<BuildClosedCanonicalPlanVf>(",
    "wait_flag(PIPE_V, PIPE_S, EVENT_ID0);",
    "reinterpret_cast<__gm__ uint8_t *>(line),",
]
positions = [kernel.index(token) for token in ordered]
positions.append(kernel.rindex("aicpu_plan_simt_scalar_continuation("))
if positions != sorted(positions):
    raise SystemExit(
        "runtime order must be clock -> VF -> V/S join -> report DCCI -> Scalar continuation"
    )

publisher_begin = kernel.index("PA_DEVICE void PublishLeaderReport(")
publisher_end = kernel.index(
    "static __simt_vf__ __aicore__ LAUNCH_BOUND(128) void",
    publisher_begin,
)
publisher = kernel[publisher_begin:publisher_end]
for token in (
    "SimtOps::StorePayloadWord",
    "SimtOps::StoreBarrier();",
    "gate::kLeaderMagic",
):
    if token not in publisher:
        raise SystemExit("leader report publication is missing " + token)
if publisher.index("SimtOps::StorePayloadWord") > publisher.index(
    "SimtOps::StoreBarrier();"
) or publisher.rindex("gate::kLeaderMagic") < publisher.index(
    "SimtOps::StoreBarrier();"
):
    raise SystemExit("leader report must publish payload -> fence -> magic")

for token in (
    "constexpr uint32_t kReportLineBytes = 128U;",
    "struct alignas(kReportLineBytes) ReportLine",
    "ReportLine leaders[kLeaderCount];",
):
    if token not in report:
        raise SystemExit("isolated leader report ABI is missing " + token)
PY

for source in "$KERNEL_SOURCE" "$SCALAR_SOURCE" "$REPORT_HEADER"; do
    for forbidden in \
        'task_id % 5' \
        'task_id%5' \
        'FullPaTaskPlan' \
        'DecodePaRuntimeTaskPlan' \
        'TaskKind' \
        'HostTaskPlan' \
        'first_task_id' \
        'first_task'; do
        if grep -Fq "$forbidden" "$source"; then
            echo "runtime gate contains forbidden PA/Host shortcut: $forbidden ($source)" >&2
            exit 1
        fi
    done
done

echo "[BUILD] dav-c310-vec runtime and independent Scalar TU"
"$CCEC" "${FLAGS[@]}" -o "$KERNEL_OBJECT" "$KERNEL_SOURCE"
"$CCEC" "${FLAGS[@]}" -o "$SCALAR_OBJECT" "$SCALAR_SOURCE"

echo "[BUILD] optimized LLVM bitcode for both identities"
"$CCEC" "${FLAGS[@]}" -Xclang -emit-llvm-bc \
    -o "$KERNEL_BITCODE" "$KERNEL_SOURCE"
"$CCEC" "${FLAGS[@]}" -Xclang -emit-llvm-bc \
    -o "$SCALAR_BITCODE" "$SCALAR_SOURCE"
"$LLVM_BCANALYZER" -dump "$KERNEL_BITCODE" > "$KERNEL_DUMP"
"$LLVM_BCANALYZER" -dump "$SCALAR_BITCODE" > "$SCALAR_DUMP"

required_kernel_ir=(
    'BuildClosedCanonicalPlanVf'
    'AttachClosedPlan'
    'TakeAttachedBuildTicket'
    'BindTask'
    'BuildCanonicalPlanTask'
    'ArriveBuilderLeaderOnce'
    'PublishBuildRelease'
    'build_next'
    'build_workers_done'
    'build_release'
    'llvm.hivm.store.vfsimt.info'
    'llvm.hivm.get.TID.X'
    'llvm.hivm.atom.ADD.G.s64'
    'llvm.hivm.atom.CAS.G.s64'
    'llvm.hivm.atom.EXCH.G.s64'
    'llvm.hivm.DCCI.DST'
    'llvm.hivm.fence.workitems'
    'llvm.hivm.stg.uncache.b64'
    'llvm.hivm.get.CLOCK64'
    'llvm.hivm.SET.FLAG.IMM'
    'llvm.hivm.WAIT.FLAG.IMM'
)
for symbol in "${required_kernel_ir[@]}"; do
    if ! grep -Fq "$symbol" "$KERNEL_DUMP"; then
        echo "runtime kernel IR is missing: $symbol" >&2
        exit 1
    fi
done
for symbol in \
    "$CONTINUATION" \
    'llvm.hivm.atom.ADD.G.s64' \
    'llvm.hivm.atom.ADD.G.s32' \
    'llvm.hivm.ST.DEV.u64' \
    'llvm.hivm.GET.SYS.CNT'; do
    if ! grep -Fq "$symbol" "$SCALAR_DUMP"; then
        echo "Scalar continuation IR is missing: $symbol" >&2
        exit 1
    fi
done

echo "[LINK] static double-TU ELF with only the launch entry exported"
"$LD_LLD" -m aicorelinux -Ttext=0 -static \
    --version-script="$VERSION_SCRIPT" \
    -o "$ELF" "$KERNEL_OBJECT" "$SCALAR_OBJECT"

SYMBOLS="$("$READELF_BIN" --symbols --wide --sym-base=10 "$ELF")"
if ! awk -v name="$ENTRY" \
    '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" &&
     $NF == name && $3 + 0 > 0 {count++}
     END {exit count != 1}' <<<"$SYMBOLS"; then
    echo "final ELF must export one non-empty $ENTRY." >&2
    exit 1
fi
global_functions="$(awk \
    '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" {print $NF}' \
    <<<"$SYMBOLS")"
if [[ "$global_functions" != "$ENTRY" ]]; then
    echo "final ELF exports unexpected GLOBAL functions:" >&2
    printf '%s\n' "$global_functions" >&2
    exit 1
fi
if ! awk \
    '$4 == "FUNC" && $5 == "LOCAL" && $3 + 0 > 0 &&
     $NF ~ /BuildClosedCanonicalPlanVf.*_simt_entry$/ {count++}
     END {exit count != 1}' <<<"$SYMBOLS"; then
    echo "final ELF must retain one non-empty LOCAL 128-thread VF." >&2
    exit 1
fi
if ! awk -v name="$CONTINUATION" \
    '$4 == "FUNC" && $5 == "LOCAL" && $7 != "UND" &&
     $NF == name && $3 + 0 > 32 {count++}
     END {exit count != 1}' <<<"$SYMBOLS"; then
    echo "Scalar continuation must resolve as a non-trivial LOCAL function." >&2
    exit 1
fi
undefined_globals="$(awk \
    '$5 == "GLOBAL" && $7 == "UND" {print $NF}' <<<"$SYMBOLS")"
if [[ -n "$undefined_globals" ]]; then
    echo "final ELF contains undefined GLOBAL symbols:" >&2
    printf '%s\n' "$undefined_globals" >&2
    exit 1
fi
RELOCATIONS="$("$READELF_BIN" --relocs --wide "$ELF")"
if [[ "$RELOCATIONS" != *"There are no relocations"* ]]; then
    echo "final ELF unexpectedly retains relocations." >&2
    exit 1
fi

SECTIONS="$("$READELF_BIN" --sections --wide "$ELF")"
metadata_sections="$(awk \
    '{for (i=1; i<=NF; ++i) if ($i ~ /^\.ascend\.meta\./) print $i}' \
    <<<"$SECTIONS")"
if [[ "$metadata_sections" != ".ascend.meta.$ENTRY" ]]; then
    echo "runtime ELF metadata sections are not exact: $metadata_sections" >&2
    exit 1
fi
METADATA_OUTPUT="$(python3 -m msobjdump -d "$ELF")"
if [[ "$METADATA_OUTPUT" != *'KERNEL_TYPE: MIX_AIC_MAIN'* ||
      "$METADATA_OUTPUT" != *'MIX_TASK_RATION: [1:2]'* ]]; then
    echo "runtime metadata is not MIX_AIC_MAIN [1:2]." >&2
    printf '%s\n' "$METADATA_OUTPUT" >&2
    exit 1
fi
META_HEX="$("$READELF_BIN" -x ".ascend.meta.$ENTRY" "$ELF")"
if [[ "$META_HEX" != *'0c000400 04000000'* ]]; then
    echo "runtime AIV metadata must encode SIMD_SIMT_MIX_VF=4." >&2
    printf '%s\n' "$META_HEX" >&2
    exit 1
fi

echo "[PASS] four warp leaders retain dynamic Attach/Take/Bind/Build/Arrive/Release IR"
echo "[PASS] AIV0 records time before VF, joins V->S, DCCI-validates four isolated reports"
echo "[PASS] fatal and success both reach the non-trivial independent Scalar continuation"
echo "[PASS] one GLOBAL entry, one LOCAL VF, MIX_VF=4, no undefined symbol"
echo "[BUILD] kernel: $ELF"
