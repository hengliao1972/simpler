#!/usr/bin/env bash
# Complete template-instantiation and machine-object gate for the narrow SIMT builder.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/narrow"
SOURCE="$SCRIPT_DIR/narrow_builder_kernel.cpp"
SCALAR_SOURCE="$SCRIPT_DIR/narrow_scalar_continuation.cpp"
HEADER="$SCRIPT_DIR/../common/simt_plan_task_builder.h"
CPU_TEST="$SCRIPT_DIR/../test/test_narrow_simt_plan_task_builder.cpp"
OBJECT="$BUILD_DIR/narrow_builder.o"
SCALAR_OBJECT="$BUILD_DIR/narrow_scalar_continuation.o"
BITCODE="$BUILD_DIR/narrow_builder.bc"
BITCODE_DUMP="$BUILD_DIR/narrow_builder.bc.dump"
ELF="$BUILD_DIR/narrow_builder_kernel.o"
ENTRY="aicpu_plan_narrow_simt_builder_compile_0_mix_aiv"

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

echo "[CHECK] complete canonical Plan-v2 Build chain is instantiated"
required_source=(
    '#include "../common/simt_plan_task_builder.h"'
    'static __simt_vf__ __aicore__ LAUNCH_BOUND(128) void'
    'simt::BuildCanonicalPlanTask<SimtOps>('
    'runtime, view, task_id, leader, scratch'
    'CheckOrdinaryAppend('
    'AppendOrdinary('
    'LookupOrdinary('
    'PublishMetadataCompletion('
    'exec::SharedExecCell'
    'FaninLowerBound('
)
for pattern in "${required_source[@]}"; do
    if ! grep -Fq "$pattern" "$SOURCE"; then
        echo "narrow full-Build source is missing: $pattern" >&2
        exit 1
    fi
done

required_header=(
    'AcquireCanonicalPlanTask<Ops>(view, task_id, scratch)'
    'MaterializeAndPublishOutputs<Ops>('
    'WaitForInsertPredecessor<Ops>(runtime, task_id)'
    'PublishWriterMetadataAndCompletion<Ops>('
    'CollectFaninAndResolveTensors<Ops>('
    'PublishTerminalBuildResult<Ops>('
    'exec::BuildAndPublishExecPayload<Ops>('
    'runtime.InsertCompletion(task_id), initial,'
    'scratch.ordinary_count, task_id'
    'SimtWriterHistoryCell'
    'SimtEncodeSharedSymbolHistoryKey('
    'SimtDecodeSharedSymbolHistoryKey('
    'const uint32_t zero_based_key = key - 1U;'
    'PA_GM volatile uint64_t *history_words ='
    '&history_words[2U + symbol]'
    'Ops::StoreBarrier();'
    'RollbackOutputsBeforeInsertCompletion<Ops>('
    'scratch.published_output_count'
    'runtime.OutputPublished(task_id, slot)'
    'runtime.OutputLastWriter(task_id, slot)'
    '!runtime.FaninLowerBound(task_id, reader_lower_bound)'
    'producer < reader_lower_bound'
    'tag != plan::TensorTag::OutputExisting'
)
for pattern in "${required_header[@]}"; do
    if ! grep -Fq "$pattern" "$HEADER"; then
        echo "narrow full-Build header is missing: $pattern" >&2
        exit 1
    fi
done

required_cpu_gate=(
    'TestSecondOutputPublishConflictRollsBackOwnedPrefix'
    'TestSecondOutputDescriptorFailureRollsBackReservations'
    'TestNoDependencySharedOutputReferenceFailsBeforeSideEffects'
    'TestAllFaninSourcesHonorHeapWindow'
    'TestSharedSymbolHistoryKeyAbi'
    'key == 1U'
    'plan::kBuildReleasePending'
    'offsetof(simt::SimtWriterRegion, buffer_addr)'
    'offsetof(simt::SimtWriterRegion, reserved)'
    'offsetof(simt::SimtWriterHistoryRecord, previous_writer)'
    'offsetof(simt::SimtWriterHistoryCell, magic)'
    'offsetof(simt::SimtWriterHistoryCell, reserved)'
    'offsetof(simt::SimtWriterHistoryCell, entries)'
)
for pattern in "${required_cpu_gate[@]}"; do
    if ! grep -Fq "$pattern" "$CPU_TEST"; then
        echo "narrow CPU semantic gate is missing: $pattern" >&2
        exit 1
    fi
done

# Writer 与 reader 的 cache 操作必须保持方向性：canonical payload 由
# asc_stcg bypass 写，fence 后才发 atomic；只有 reader acquire 执行 DCCI。
python3 - "$SOURCE" <<'PY'
import pathlib
import sys

text = pathlib.Path(sys.argv[1]).read_text()
flush_start = text.index(
    "PA_DEVICE static void FlushRegion(\n        __gm__ void *address"
)
flush_end = text.index(
    "PA_DEVICE static void FlushRegion(\n        __gm__ volatile uint64_t *address",
    flush_start + 1,
)
flush_body = text[flush_start:flush_end]
if "asc_dcci_single" in flush_body or "asc_threadfence" not in flush_body:
    raise SystemExit("writer FlushRegion must be fence-only and must not execute DCCI")
invalidate_start = text.index(
    "PA_DEVICE static void InvalidateRegion(\n        __gm__ const void *address"
)
invalidate_end = text.index(
    "PA_DEVICE static void InvalidateRegion(\n        __gm__ void *address",
    invalidate_start + 1,
)
invalidate_body = text[invalidate_start:invalidate_end]
if "asc_dcci_single" not in invalidate_body or "asc_threadfence" not in invalidate_body:
    raise SystemExit("reader InvalidateRegion must retain per-line DCCI and fence")
PY

python3 - "$HEADER" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
ordinary_history_store = re.search(
    r"history->(?:magic|writer_task|count|reserved)\s*=(?!=)|"
    r"history->entries\[[^]]+\]\.(?:symbol_key|previous_writer)\s*=(?!=)",
    text,
)
if ordinary_history_store:
    raise SystemExit(
        "history writer contains ordinary GM store: " +
        ordinary_history_store.group(0)
    )
PY

for forbidden in \
    'DecodePaRuntimeTaskPlan' \
    'TaskArgs args' \
    'task_id %' \
    'task_id%' \
    'FullPaTaskPlan' \
    'SimtPrepareTask' \
    'HostTaskPlan'; do
    if grep -Fq "$forbidden" "$SOURCE" "$HEADER"; then
        echo "narrow full-Build contains forbidden reconstruction: $forbidden" >&2
        exit 1
    fi
done

echo "[BUILD] dav-c310-vec machine object under -Werror"
"$CCEC" "${FLAGS[@]}" -o "$OBJECT" "$SOURCE"
echo "[BUILD] separate Scalar scheduler-continuation object"
"$CCEC" "${FLAGS[@]}" -o "$SCALAR_OBJECT" "$SCALAR_SOURCE"

echo "[BUILD] optimized LLVM bitcode"
"$CCEC" "${FLAGS[@]}" -Xclang -emit-llvm-bc \
    -o "$BITCODE" "$SOURCE"
"$LLVM_BCANALYZER" -dump "$BITCODE" > "$BITCODE_DUMP"

required_ir=(
    'BuildCanonicalPlanTaskNarrowVf'
    'llvm.hivm.store.vfsimt.info'
    'llvm.hivm.get.TID.X'
    'llvm.hivm.atom.ADD.G.s64'
    'llvm.hivm.atom.CAS.G.s64'
    'llvm.hivm.DCCI.DST'
    'llvm.hivm.fence.workitems'
    'llvm.hivm.stg.uncache.b64'
    'llvm.hivm.get.CLOCK64'
    'llvm.hivm.SET.FLAG.IMM'
    'llvm.hivm.WAIT.FLAG.IMM'
)
for symbol in "${required_ir[@]}"; do
    if ! grep -Fq "$symbol" "$BITCODE_DUMP"; then
        echo "optimized narrow full-Build bitcode is missing: $symbol" >&2
        exit 1
    fi
done

echo "[BUILD] linked AIV machine object"
"$LD_LLD" -m aicorelinux -Ttext=0 -static \
    -o "$ELF" "$OBJECT" "$SCALAR_OBJECT"

SYMBOLS="$("$READELF_BIN" --symbols --wide --sym-base=10 "$ELF")"
if ! awk -v name="$ENTRY" \
    '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
     END {exit count != 1}' <<<"$SYMBOLS"; then
    echo "final object must export exactly one non-empty GLOBAL $ENTRY." >&2
    exit 1
fi
if ! awk \
    '$4 == "FUNC" && $5 == "LOCAL" && $3 + 0 > 0 &&
     $NF ~ /BuildCanonicalPlanTaskNarrowVf.*_simt_entry$/ {count++}
     END {exit count != 1}' <<<"$SYMBOLS"; then
    echo "final object must retain one non-empty narrow full-Build SIMT entry." >&2
    exit 1
fi
if ! awk \
    '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" &&
     $NF == "aicpu_plan_narrow_scalar_continuation" && $3 + 0 > 0 {count++}
     END {exit count != 1}' <<<"$SYMBOLS"; then
    echo "final object must resolve one non-empty Scalar continuation." >&2
    exit 1
fi
undefined_globals="$(awk '$5 == "GLOBAL" && $7 == "UND" {print $NF}' <<<"$SYMBOLS")"
if [[ -n "$undefined_globals" ]]; then
    echo "final object contains undefined GLOBAL symbols:" >&2
    printf '%s\n' "$undefined_globals" >&2
    exit 1
fi

echo "[PASS] full BuildCanonicalPlanTask template generated machine code"
echo "[PASS] separate Scalar-core TU continuation resolved after VF join"
echo "[PASS] writer publication is stcg -> fence -> atomic; DCCI remains reader-only"
echo "[PASS] pre-completion output rollback and [N-H,N) fanin contracts remain source-gated"
echo "[PASS] Plan acquire, materialize, ordinary/symbol Register, fanin and BUILT publication remain in optimized IR"
echo "[BUILD] object:  $OBJECT"
echo "[BUILD] bitcode: $BITCODE"
echo "[BUILD] kernel:  $ELF"
