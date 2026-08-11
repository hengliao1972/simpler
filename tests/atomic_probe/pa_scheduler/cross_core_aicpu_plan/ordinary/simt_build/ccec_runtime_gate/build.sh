#!/usr/bin/env bash
# CCEC O3/-Werror machine-code gate for the real SchedulerState adapter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE="$SCRIPT_DIR/kernel.cpp"
RUNTIME_HEADER="$MODE_DIR/common/simt_real_state_runtime.h"
BUILDER_HEADER="$MODE_DIR/common/simt_plan_task_builder.h"
REAL_MAP_HEADER="$MODE_DIR/../scalar_build/common/pa_shared_tensormap.h"
BUILD_DIR="$SCRIPT_DIR/build"
OBJECT="$BUILD_DIR/simt_real_state_runtime_gate.o"
BITCODE="$BUILD_DIR/simt_real_state_runtime_gate.bc"
BITCODE_DUMP="$BUILD_DIR/simt_real_state_runtime_gate.bc.dump"
ENTRY="aicpu_plan_simt_real_state_runtime_gate_0_mix_aiv"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set." >&2
    exit 1
fi

CCEC="$ASCEND_HOME_PATH/bin/ccec"
READELF_BIN="${READELF:-readelf}"
if command -v llvm-bcanalyzer >/dev/null 2>&1; then
    LLVM_BCANALYZER="$(command -v llvm-bcanalyzer)"
else
    LLVM_BCANALYZER="/opt/mlir-debug/bin/llvm-bcanalyzer"
fi
for tool in "$CCEC" "$LLVM_BCANALYZER"; do
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
    -DPA_RUNTIME_PLAN_BUILD_BACKEND=1
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

echo "[CHECK] adapter aliases the real Scalar state"
required_runtime=(
    'PA_GM SchedulerState *state;'
    'kSimtBuildOwnerLimit = kBuilderLeaders'
    'kSimtBuildOwnerLimit == 4U'
    'kSimtBuildOwnerLimit <= kWorkers'
    'state->shared_map.shared_outputs[task_id]'
    'state->claim_tournament[task_id]'
    'state->shared_map.writer_history[task_id]'
    'state->tasks[task_id]'
    'state->exec_cells[task_id]'
    'state->exec_fatal'
    'ReserveSharedOutputHeap<Ops, false>'
    'SharedCheckPreparedTaskAppend<Ops, false>'
    'SharedLookupRegion<Ops, false, true>'
    'runtime_plan_control.fatal.value'
    'FaninLowerBound('
)
for pattern in "${required_runtime[@]}"; do
    if ! grep -Fq "$pattern" "$RUNTIME_HEADER"; then
        echo "real-state adapter is missing: $pattern" >&2
        exit 1
    fi
done
if grep -Eq 'struct[[:space:]]+(Narrow|Mirror).*(State|Sidecar)' \
        "$RUNTIME_HEADER"; then
    echo "real-state adapter must not define a mirror GM state." >&2
    exit 1
fi

python3 - "$RUNTIME_HEADER" "$SOURCE" "$REAL_MAP_HEADER" <<'PY'
import pathlib
import re
import sys

runtime = pathlib.Path(sys.argv[1]).read_text()
source = pathlib.Path(sys.argv[2]).read_text()
real_map = pathlib.Path(sys.argv[3]).read_text()

if re.search(
    r"reinterpret_cast<[^>]*(?:SimtWriterHistoryCell|"
    r"SharedWriterHistoryCell|SimtWriterRegion|SharedRegionValue)[^>]*>",
    runtime,
):
    raise SystemExit(
        "canonical/production structured objects must be converted, not aliased"
    )
if "PA_GM SharedWriterHistoryCell *WriterHistory(" not in runtime or \
        "SharedRegionValue real_entries[kMaxTaskTensors]" not in runtime:
    raise SystemExit("real history/ordinary typed mapping is missing")

heap_start = runtime.index("PA_DEVICE bool ReserveOutputHeap(")
heap_end = runtime.index("PA_DEVICE uint32_t OrdinaryBucket", heap_start)
heap = runtime[heap_start:heap_end]
null_base_check = heap.index("total != 0U && state->heap_base == 0U")
reservation = heap.index("ReserveSharedOutputHeap<Ops, false>")
if null_base_check > reservation:
    raise SystemExit("null heap base must fail before the first reservation atomic")

append_start = runtime.index("PA_DEVICE bool AppendPreparedEntryStcg(")
append_body = runtime[append_start:runtime.index("};\n\n}  // namespace", append_start)]
if append_body.count("Ops::StorePayloadWord(") != 4:
    raise SystemExit("ordinary payload must be exactly four canonical stcg words")
if "Ops::InvalidateRegion" in append_body or "Ops::FlushRegion" in append_body:
    raise SystemExit("SIMT ordinary writer must not execute reader-side DCCI")
fence = append_body.index("Ops::StoreBarrier();")
seq_publish = append_body.index(
    "&slot.seq.value, static_cast<int64_t>(cursor)", fence
)
tail_publish = append_body.index("&controls.tail.value, tail + 1", seq_publish)
if not fence < seq_publish < tail_publish:
    raise SystemExit("ordinary publication must be stcg -> fence -> seq -> tail")

metadata_start = runtime.index("PA_DEVICE bool PublishMetadataCompletion(")
metadata_end = runtime.index("PA_DEVICE PA_GM exec::SharedExecCell", metadata_start)
metadata = runtime[metadata_start:metadata_end]
if metadata.count("Ops::Exchange(") != 2 or "&cell.vend" not in metadata or "&cell.flag" not in metadata:
    raise SystemExit("metadata completion must use real TaskCell atomic vend+flag")
if not metadata.index("&cell.vend") < metadata.index("Ops::StoreBarrier();") < metadata.index("&cell.flag"):
    raise SystemExit("metadata completion order must remain vend -> fence -> flag")

flush_start = source.index("PA_DEVICE static void FlushRegion(\n        __gm__ void *")
flush_end = source.index(
    "PA_DEVICE static void FlushRegion(\n        __gm__ volatile uint64_t *",
    flush_start + 1,
)
flush = source[flush_start:flush_end]
if "asc_dcci_single" in flush or "asc_threadfence" not in flush:
    raise SystemExit("writer FlushRegion must be fence-only")
invalidate_start = source.index("PA_DEVICE static void InvalidateRegion(\n        __gm__ const void *")
invalidate_end = source.index(
    "PA_DEVICE static void InvalidateRegion(\n        __gm__ void *",
    invalidate_start + 1,
)
invalidate = source[invalidate_start:invalidate_end]
if "asc_dcci_single" not in invalidate or "asc_threadfence" not in invalidate:
    raise SystemExit("reader InvalidateRegion must keep DCCI plus fence")

if "SharedReadRegionSlot<Ops" not in real_map or "TraceConfiguredDcciInvalidate" not in real_map:
    raise SystemExit("real reader helper lost atomic/DCCI acquire sequence")
PY

for forbidden in \
    'task_id % 5' \
    'task_id%5' \
    'FullPaTaskPlan' \
    'HostTaskPlan' \
    'DecodePaRuntimeTaskPlan'; do
    if grep -Fq "$forbidden" "$SOURCE" "$RUNTIME_HEADER"; then
        echo "real-state gate contains forbidden PA reconstruction: $forbidden" >&2
        exit 1
    fi
done

echo "[BUILD] dav-c310-vec O3/-Werror machine object"
"$CCEC" "${FLAGS[@]}" -o "$OBJECT" "$SOURCE"

echo "[BUILD] optimized LLVM bitcode"
"$CCEC" "${FLAGS[@]}" -Xclang -emit-llvm-bc \
    -o "$BITCODE" "$SOURCE"
"$LLVM_BCANALYZER" -dump "$BITCODE" > "$BITCODE_DUMP"

required_ir=(
    'RealStateBuildVf'
    'llvm.hivm.store.vfsimt.info'
    'llvm.hivm.get.TID.X'
    'llvm.hivm.atom.ADD.G.s32'
    'llvm.hivm.atom.ADD.G.s64'
    'llvm.hivm.atom.EXCH.G.s32'
    'llvm.hivm.atom.EXCH.G.s64'
    'llvm.hivm.atom.EXCH.G.u64'
    'llvm.hivm.atom.CAS.G.s64'
    'llvm.hivm.stg.uncache.b64'
    'llvm.hivm.DCCI.DST'
    'llvm.hivm.fence.workitems'
    'llvm.hivm.get.CLOCK64'
)
for symbol in "${required_ir[@]}"; do
    if ! grep -Fq "$symbol" "$BITCODE_DUMP"; then
        echo "optimized real-state bitcode is missing: $symbol" >&2
        exit 1
    fi
done

symbols="$("$READELF_BIN" --symbols --wide --sym-base=10 "$OBJECT")"
if ! awk -v name="$ENTRY" \
    '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" &&
     $NF == name && $3 + 0 > 0 {count++}
     END {exit count != 1}' <<<"$symbols"; then
    echo "machine object must export one non-empty GLOBAL $ENTRY." >&2
    exit 1
fi
if ! awk \
    '$4 == "FUNC" && $5 == "LOCAL" && $3 + 0 > 0 &&
     $NF ~ /RealStateBuildVf.*_simt_entry$/ {count++}
     END {exit count != 1}' <<<"$symbols"; then
    echo "machine object must retain one real-state SIMT entry." >&2
    exit 1
fi
undefined_globals="$(
    awk '$5 == "GLOBAL" && $7 == "UND" {print $NF}' <<<"$symbols"
)"
if [[ -n "$undefined_globals" ]]; then
    echo "machine object contains undefined GLOBAL symbols:" >&2
    printf '%s\n' "$undefined_globals" >&2
    exit 1
fi

echo "[PASS] scalar type include is machine-code reachable under simt_callee"
echo "[PASS] real SchedulerState/sidecar/TaskCell/ExecCell mapping is instantiated"
echo "[PASS] writer stcg+fence and reader atomic+DCCI publication directions are locked"
echo "[BUILD] object:  $OBJECT"
echo "[BUILD] bitcode: $BITCODE"
