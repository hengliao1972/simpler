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
KERNEL_SOURCE="$SCRIPT_DIR/g0_full_pa_kernel.cpp"
WORKLOAD_SOURCE="$SCRIPT_DIR/full_pa_workloads.h"
MODEL_SOURCE="$SIMT_ROOT/common/full_pa_model.h"
HOST_SOURCE="$SCRIPT_DIR/g0_full_pa_host.cpp"
SWIMLANE_ACL_CONFIG="$SCRIPT_DIR/g0_swimlane_acl.json"
G0_VARIANT="${SIMT_CROSS_CORE_G0_VARIANT:-base}"
BUILDER_WARP_COUNT="${SIMT_CROSS_CORE_GM_BUILDER_WARPS:-16}"
if [[ ! "$BUILDER_WARP_COUNT" =~ ^[0-9]+$ ]] ||
   (( BUILDER_WARP_COUNT < 1 || BUILDER_WARP_COUNT > 64 )); then
    echo "SIMT_CROSS_CORE_GM_BUILDER_WARPS must be an integer in 1..64." >&2
    exit 1
fi
case "$G0_VARIANT" in
    base)
        OUTPUT_TAG="g0"
        DEVICE_VARIANT_FLAGS=()
        AIV_VARIANT_FLAGS=()
        HOST_VARIANT_FLAGS=()
        ;;
    swimlane)
        OUTPUT_TAG="g0_swimlane"
        DEVICE_VARIANT_FLAGS=(-DSIMT_CROSS_CORE_G0_SWIMLANE)
        # cce-vf-stack-size 会叠加到默认 2 KiB compiler UB；0x3800
        # 最终形成 tag7=0x4000，即 16 KiB 的 AIV VF 总保留。
        AIV_VARIANT_FLAGS=(-mllvm -cce-vf-stack-size=0x3800)
        HOST_VARIANT_FLAGS=(-DSIMT_CROSS_CORE_G0_SWIMLANE)
        ;;
    *)
        echo "unknown SIMT_CROSS_CORE_G0_VARIANT: $G0_VARIANT (expected base or swimlane)" >&2
        exit 1
        ;;
esac
AIC_OBJECT="$BUILD_DIR/simt_cross_core_${OUTPUT_TAG}_aic.o"
AIV_OBJECT="$BUILD_DIR/simt_cross_core_${OUTPUT_TAG}_aiv.o"
AIC_BITCODE="$BUILD_DIR/simt_cross_core_${OUTPUT_TAG}_aic.bc"
AIV_BITCODE="$BUILD_DIR/simt_cross_core_${OUTPUT_TAG}_aiv.bc"
AIC_BITCODE_DUMP="$BUILD_DIR/simt_cross_core_${OUTPUT_TAG}_aic.bc.dump"
AIV_BITCODE_DUMP="$BUILD_DIR/simt_cross_core_${OUTPUT_TAG}_aiv.bc.dump"
KERNEL_ELF="$BUILD_DIR/simt_cross_core_${OUTPUT_TAG}_kernel.o"
HOST_BINARY="$BUILD_DIR/simt_cross_core_${OUTPUT_TAG}_host"
BUILD_MANIFEST="$BUILD_DIR/${OUTPUT_TAG}_build_manifest.sha256"
AIC_ENTRY="simt_cross_core_g0_0_mix_aic"
AIV_ENTRY="simt_cross_core_g0_0_mix_aiv"
G0_BUILD_INPUTS=(
    run.sh
    common/full_pa_exec_protocol.h
    common/full_pa_model.h
    gm/common/g0_full_pa.h
    gm/ccec/full_pa_workloads.h
    gm/ccec/g0_full_pa_kernel.cpp
    gm/ccec/g0_full_pa_host.cpp
    gm/ccec/build_g0.sh
)
if [[ "$G0_VARIANT" == "swimlane" ]]; then
    G0_BUILD_INPUTS+=(gm/common/g0_swimlane.h gm/ccec/g0_swimlane_acl.json)
fi

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
if ! command -v rg >/dev/null 2>&1; then
    echo "rg is required for G0 source validation." >&2
    exit 1
fi
if ! python3 -c 'import msobjdump' >/dev/null 2>&1; then
    echo "the CANN msobjdump module is required for metadata validation." >&2
    exit 1
fi
for source in "$KERNEL_SOURCE" "$WORKLOAD_SOURCE" "$MODEL_SOURCE" "$HOST_SOURCE"; do
    if [[ ! -s "$source" ]]; then
        echo "required G0 source is missing: $source" >&2
        exit 1
    fi
done
if [[ "$G0_VARIANT" == "swimlane" ]]; then
    if [[ ! -s "$SWIMLANE_ACL_CONFIG" ]] ||
       ! python3 - "$SWIMLANE_ACL_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    config = json.load(source)
stack = config.get("StackSize", {})
if stack != {"simt_stack_size": 1536, "simt_divergence_stack_size": 4608}:
    raise SystemExit(1)
PY
    then
        echo "G0 swimlane ACL config must encode the 512 B stride-aligned 1536 B SIMT and 4608 B DVG stack limits." >&2
        exit 1
    fi
fi

mkdir -p "$BUILD_DIR"

COMMON_DEVICE_FLAGS=(
    -c -O3 -g -x cce -Wall -std=c++17
    "-DSIMT_CROSS_CORE_G0_BUILDER_WARP_COUNT=$BUILDER_WARP_COUNT"
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

echo "[CHECK] GM source closure, 1..16 independent builders, warps/builder=$BUILDER_WARP_COUNT and publication order (variant=$G0_VARIANT)"
if rg -n '#include.*(cross_core|ops-nn)' "$SIMT_ROOT" -g '*.h' -g '*.cpp'; then
    echo "G0 must not include cross_core or ops-nn source files." >&2
    exit 1
fi
if ! grep -Fq 'constexpr uint32_t kBuilderWarpCount = SIMT_CROSS_CORE_G0_BUILDER_WARP_COUNT;' "$MODEL_SOURCE" ||
   ! grep -Fq 'constexpr uint32_t kBuilderThreadCount = kBuilderWarpCount * kWarpSize;' "$MODEL_SOURCE" ||
   ! grep -Fq 'constexpr uint32_t kMaxBuilderCount = 16U;' "$MODEL_SOURCE" ||
   ! grep -Fq 'constexpr uint32_t kMaxBuilderThreadCount = kBuilderThreadCount * kMaxBuilderCount;' "$MODEL_SOURCE" ||
   ! grep -Fq 'constexpr uint32_t kBuilderTaskStride = kBuilderWarpCount;' "$MODEL_SOURCE" ||
   ! rg -q -U 'static_assert\(\s*kBuilderWarpCount >= 1U && kBuilderWarpCount <= 64U' "$MODEL_SOURCE" ||
   ! grep -Fq 'const bool active = lane == 0U && warp < kBuilderWarpCount;' "$KERNEL_SOURCE" ||
   ! grep -Fq 'const uint32_t first_assigned_task = builder_instance * kBuilderWarpCount + warp;' "$KERNEL_SOURCE" ||
   ! grep -Fq 'const uint32_t task_stride = builder_count * kBuilderWarpCount;' "$KERNEL_SOURCE" ||
   ! grep -Fq 'task_id = first_assigned_task; task_id < task_count; task_id += task_stride' "$KERNEL_SOURCE" ||
   ! grep -Fq 'cce::dim3{kBuilderThreadCount, 1U, 1U}' "$KERNEL_SOURCE"; then
    echo "GM must launch the configured SIMT warp count per builder and statically shard tasks across lane 0." >&2
    exit 1
fi
if ! grep -Fq 'BuilderCountValid(state->control.builder_count)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'state->control.builder_thread_count == kBuilderThreadCount' "$KERNEL_SOURCE" ||
   ! grep -Fq 'const uint32_t global_thread = builder_instance * kBuilderThreadCount + thread;' "$KERNEL_SOURCE" ||
   ! grep -Fq 'const uint32_t global_warp = builder_instance * kBuilderWarpCount + warp;' "$KERNEL_SOURCE" ||
   ! grep -Fq 'if (aiv_id >= state->control.builder_count)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'state->control.builder_count, owner' "$KERNEL_SOURCE"; then
    echo "GM must keep the configured threads per VF while selecting 1..16 fixed builder AIVs at runtime." >&2
    exit 1
fi
if ! grep -Fq 'if (thread == 0U)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'g0_swimlane::AtomicSite::SimtBuilderStartedIncrement' "$KERNEL_SOURCE" ||
   ! grep -Fq 'asc_atomic_add(builder_started, static_cast<uint64_t>(0U))' "$KERNEL_SOURCE" ||
   ! grep -Fq 'if (observed == builder_count)' "$KERNEL_SOURCE"; then
    echo "G0/G1 each VF must contribute exactly one builder-start arrival before active leaders leave the gate." >&2
    exit 1
fi
if ! grep -Fq 'const uint32_t aiv_id = block * subblock_dim + subblock;' "$KERNEL_SOURCE" ||
   ! grep -Fq 'subblock_dim != 2U' "$KERNEL_SOURCE" ||
   ! grep -Fq 'subblock >= subblock_dim' "$KERNEL_SOURCE"; then
    echo "G0/G1 must derive dense AIV0/AIV1 owner ids from the checked 1:2 mixed topology." >&2
    exit 1
fi
if [[ "$G0_VARIANT" == "swimlane" ]]; then
    if rg -q 'ScalarPollEpisode[[:space:]]+(built|fanin)_episodes\[' "$KERNEL_SOURCE" ||
       ! grep -Fq 'ScalarPollEpisode built_episode0;' "$KERNEL_SOURCE" ||
       ! grep -Fq 'ScalarPollEpisode fanin_episode3;' "$KERNEL_SOURCE" ||
       ! grep -Fq 'InitializeScalarPollEpisode(&built_episode0);' "$KERNEL_SOURCE" ||
       ! grep -Fq 'InitializeScalarPollEpisode(&fanin_episode3);' "$KERNEL_SOURCE" ||
       ! grep -Fq 'refusing to overwrite existing swimlane output' "$HOST_SOURCE"; then
        echo "G0 swimlane must keep four poll slots explicitly scalar-addressed and refuse output overwrite." >&2
        exit 1
    fi
fi

vf_start_line="$(grep -nF 'void G0SimtBuildTasks(' "$KERNEL_SOURCE" | cut -d: -f1)"
vf_end_line="$(grep -nF '#endif  // defined(__DAV_VEC__)' "$KERNEL_SOURCE" | head -1 | cut -d: -f1)"
builder_publish_line="$(grep -nF 'g0_swimlane::AtomicSite::SimtBuilderFinishedPublish' "$KERNEL_SOURCE" | cut -d: -f1)"
builder_gate_line="$(awk '/void G0SimtBuildTasks\(/ {inside=1} inside && /const bool start_ready/ {print NR; exit}' "$KERNEL_SOURCE")"
claim_line="$(awk '/void G0SimtBuildTasks\(/ {inside=1} inside && /const uint32_t claim = SimtTryClaimTask\(/ {print NR; exit}' "$KERNEL_SOURCE")"
prepare_call_line="$(awk '/void G0SimtBuildTasks\(/ {inside=1} inside && /if \(!SimtPrepareTask\(/ {print NR; exit}' "$KERNEL_SOURCE")"
commit_call_line="$(awk '/void G0SimtBuildTasks\(/ {inside=1} inside && /if \(!SimtCommitTask\(/ {print NR; exit}' "$KERNEL_SOURCE")"
report_direct_line="$(grep -nF 'asc_stcg(report + 6U' "$KERNEL_SOURCE" | cut -d: -f1)"
async_line="$(grep -nF 'cce::async_invoke<G0SimtBuildTasks>' "$KERNEL_SOURCE" | cut -d: -f1)"
builder_wait_line="$(awk '/cce::async_invoke<G0SimtBuildTasks>/ {inside=1} inside && /wait_flag\(PIPE_V, PIPE_S, EVENT_ID0\)/ {print NR; exit}' "$KERNEL_SOURCE")"
leader_report_guard_line="$(awk '/void G0SimtBuildTasks\(/ {inside=1} inside && /^[[:space:]]*if \(active\) \{$/ {print NR; exit}' "$KERNEL_SOURCE")"
thread_report_line="$(grep -nF 'thread_report_words + global_thread * kThreadReportStrideWords' "$KERNEL_SOURCE" | cut -d: -f1)"
if [[ -z "$vf_start_line" || -z "$vf_end_line" || -z "$builder_publish_line" || -z "$builder_gate_line" ||
      -z "$claim_line" || -z "$prepare_call_line" || -z "$commit_call_line" || -z "$report_direct_line" ||
      -z "$async_line" || -z "$builder_wait_line" ||
      -z "$leader_report_guard_line" || -z "$thread_report_line" ]] ||
   ! (( vf_start_line < builder_gate_line && builder_gate_line < claim_line && claim_line < prepare_call_line &&
         prepare_call_line < commit_call_line && commit_call_line < report_direct_line &&
         report_direct_line < builder_publish_line && builder_publish_line < leader_report_guard_line &&
         leader_report_guard_line < thread_report_line && thread_report_line < vf_end_line &&
         vf_end_line < async_line && async_line < builder_wait_line )); then
    echo "G0/G1 must gate both VFs, claim before task writes, publish direct diagnostic evidence, and finish in SIMT." >&2
    exit 1
fi
if grep -Eq '^[[:space:]]*report\[[0-7](U)?\][[:space:]]*=' "$KERNEL_SOURCE" ||
   ! grep -Fq 'SimtPublishBuildReportWord(' "$KERNEL_SOURCE" ||
   ! grep -Fq 'g0_swimlane::AtomicSite::SimtBuildReportPublish' "$KERNEL_SOURCE" ||
   ! grep -Fq 'kReportPoisonWord, value, true' "$KERNEL_SOURCE" ||
   ! grep -Fq 'asc_stcg(report + 6U, static_cast<uint64_t>(1U) | (static_cast<uint64_t>(1U) << 32U));' \
       "$KERNEL_SOURCE"; then
    echo "G0/G1 statically owned build reports must use non-cacheable stores; U2 keeps its separate atomic evidence path." >&2
    exit 1
fi
if ! grep -Fq 'plan[4] = static_cast<uint64_t>(encoded_meta)' "$KERNEL_SOURCE" ||
   ! grep -Fq '(static_cast<uint64_t>(build_owner) << 16U)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'thread_report_words + global_thread * kThreadReportStrideWords' "$KERNEL_SOURCE" ||
   ! grep -Fq '(static_cast<uint64_t>(claim_losses) << 32U)' "$KERNEL_SOURCE"; then
    echo "G0/G1 must preserve per-task builder owner and disjoint per-instance global thread/warp reports." >&2
    exit 1
fi
if ! grep -Fq 'inactive-builder-lane-mutated' "$HOST_SOURCE"; then
    echo "G0/G1 host must prove inactive SIMT lanes leave their diagnostic reports poisoned." >&2
    exit 1
fi
if grep -Fq 'ScalarCas(&state->drain.builder_finished' "$KERNEL_SOURCE"; then
    echo "G0 Main Scalar must not publish builder_finished." >&2
    exit 1
fi
if grep -Eq 'result(->|\.)build_count[[:space:]]*=[[:space:]]*state->control.task_count|result(->|\.)commit_count[[:space:]]*=[[:space:]]*state->control.task_count' \
    "$KERNEL_SOURCE"; then
    echo "G0/G1 Main Scalar roles must not receive SIMT task build or commit attribution." >&2
    exit 1
fi
if ! grep -Fq 'SimtAllocBuildingState(nonce, task_id, build_owner)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'g0_swimlane::AtomicSite::SimtTaskBuildClaim' "$KERNEL_SOURCE" ||
   ! grep -Fq 'g0_swimlane::AtomicSite::SimtAllocCompletionFlagPublish' "$KERNEL_SOURCE" ||
   ! grep -Fq 'SimtCompetingExecStateValid(observed, task_id, build_owner, builder_count)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'return kSimtClaimLost;' "$KERNEL_SOURCE"; then
    echo "G0/G1 must give Alloc a special flag claim and treat only legal competing task states as a claim loss." >&2
    exit 1
fi
if ! grep -Fq '(static_cast<uint64_t>(build_owner) << kStateBuildOwnerShift)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'token->control.build_owner = decoded.build_owner;' "$KERNEL_SOURCE" ||
   ! grep -Fq 'ClaimedState(task_id, token->control.build_owner, owner)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'DoneState(task_id, token->control.build_owner, owner)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'OwnerEngine(owner, state->control.builder_count)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'OwnerRoleAt(owner, builder_count)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'AivExecutorCount(state->control.builder_count)' "$KERNEL_SOURCE"; then
    echo "G0/G1 must preserve the actual build owner through claim, token, DONE, roles and runtime drain counts." >&2
    exit 1
fi

prepare_start_line="$(grep -nF 'inline bool SimtPrepareTask(' "$KERNEL_SOURCE" | cut -d: -f1)"
descriptor_line="$(awk '/inline bool SimtPrepareTask\(/ {inside=1} inside && /SimtStoreOutputDescriptor\(/ {print NR; exit}' "$KERNEL_SOURCE")"
descriptor_fence_line="$(awk '/inline bool SimtPrepareTask\(/ {inside=1} inside && /SimtStoreOutputDescriptor\(/ {descriptor=1; next} descriptor && /asc_threadfence\(\);/ {print NR; exit}' "$KERNEL_SOURCE")"
fresh_publish_line="$(grep -nF 'g0_swimlane::AtomicSite::SimtOutputPublishedPublish' "$KERNEL_SOURCE" | cut -d: -f1)"
commit_start_line="$(grep -nF 'inline bool SimtCommitTask(' "$KERNEL_SOURCE" | cut -d: -f1)"
metadata_output_wait_line="$(awk '/inline bool SimtCommitTask\(/ {inside=1} inside && /if \(!SimtWaitOutputPublished\(/ {print NR; exit}' "$KERNEL_SOURCE")"
predecessor_wait_line="$(awk '/inline bool SimtCommitTask\(/ {inside=1} inside && /if \(!SimtWaitAtomicValue\(/ {print NR; exit}' "$KERNEL_SOURCE")"
if [[ -z "$prepare_start_line" || -z "$descriptor_line" || -z "$descriptor_fence_line" || -z "$fresh_publish_line" ||
      -z "$commit_start_line" || -z "$metadata_output_wait_line" || -z "$predecessor_wait_line" ]] ||
   ! (( prepare_start_line < descriptor_line && descriptor_line < descriptor_fence_line &&
         descriptor_fence_line < fresh_publish_line &&
         fresh_publish_line < commit_start_line && commit_start_line < metadata_output_wait_line &&
         metadata_output_wait_line < predecessor_wait_line )); then
    echo "G0 fresh outputs must publish after descriptor construction; sparse metadata writers must acquire their target before the strict writer-predecessor wait." >&2
    exit 1
fi
if ! grep -Fq 'const uint64_t metadata_insert_contract = task[kPlanOffsetWords + 7U];' "$KERNEL_SOURCE" ||
   ! grep -Fq 'for (uint32_t writer = 0U; writer < writer_count; ++writer)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'SimtDecodeSharedSymbolKey(' "$KERNEL_SOURCE" ||
   awk '
       /inline bool SimtCommitTask\(/ {inside=1}
       inside && /if \(kind == static_cast<uint32_t>\(TaskKind::Alloc\)\)/ {inside=0}
       inside && /TaskKind::Up/ {found=1}
       END {exit found ? 0 : 1}
   ' "$KERNEL_SOURCE"; then
    echo "G0 metadata commit must consume the generic writer-intent contract; operator-specific TaskKind branches are forbidden." >&2
    exit 1
fi
if ! grep -Fq 'SimtPreviousMetadataWriterForSymbol(task_id, producer, output_slot)' "$KERNEL_SOURCE" ||
   ! grep -Fq '(static_cast<uint64_t>(previous) << 32U)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'if (predecessor_task == producer)' "$KERNEL_SOURCE" ||
   ! grep -Fq '++*predecessor_wait_count;' "$KERNEL_SOURCE" ||
   grep -Fq 'AtomicSite::SimtMetadataLastWriterLoad' "$KERNEL_SOURCE"; then
    echo "G0 metadata writers must prepare exact per-symbol predecessors once, wait only real predecessors and CAS without a last-writer atomic load." >&2
    exit 1
fi

invalidate_start_line="$(grep -nF 'InvalidatePayloadLines(' "$KERNEL_SOURCE" | head -1 | cut -d: -f1)"
invalidate_dcci_line="$(awk '/InvalidatePayloadLines\(/ {inside=1} inside && /dcci\(/ {print NR; exit}' "$KERNEL_SOURCE")"
invalidate_dsb_line="$(awk '/InvalidatePayloadLines\(/ {inside=1} inside && /dsb\(DSB_ALL\);/ {print NR; exit}' "$KERNEL_SOURCE")"
invalidate_call_line="$(grep -nF 'InvalidatePayloadLines(task, task_id, layout.payload_lines' "$KERNEL_SOURCE" | cut -d: -f1)"
first_payload_read_line="$(awk '/InvalidatePayloadLines\(task, task_id, layout.payload_lines/ {inside=1; next} inside && /PayloadWord\(task,/ {print NR; exit}' "$KERNEL_SOURCE")"
if [[ -z "$invalidate_start_line" || -z "$invalidate_dcci_line" || -z "$invalidate_dsb_line" ||
      -z "$invalidate_call_line" || -z "$first_payload_read_line" ]] ||
   ! (( invalidate_start_line < invalidate_dcci_line && invalidate_dcci_line < invalidate_dsb_line &&
         invalidate_call_line < first_payload_read_line )) ||
   ! grep -Fq 'for (uint32_t line = 0U; line < payload_lines; ++line)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'payload + line * kCacheLineBytes' "$KERNEL_SOURCE"; then
    echo "G0 executors must invalidate every published payload cache line before ordinary reads." >&2
    exit 1
fi

reset_dcci_count="$(awk '
    /ResetToken\(__gm__/ {inside=1}
    /PublishTerminalTokenState\(/ {inside=0}
    inside && /dcci\(/ {count++}
    END {print count + 0}
' "$KERNEL_SOURCE")"
if [[ "$reset_dcci_count" -ne 0 ]] ||
   ! grep -Fq 'PublishTerminalTokenState(state, owner, result->ticket_count G0_SCALAR_TRACE_ARGUMENT);' "$KERNEL_SOURCE"; then
    echo "G0 token reset must retain dispatch and publish cache lines only once at executor drain." >&2
    exit 1
fi
if ! grep -Fq 'dcci(static_cast<__gm__ void *>(dispatch), kSingleCacheLine);' "$KERNEL_SOURCE" ||
   ! grep -Fq 'dispatch + kCacheLineBytes' "$KERNEL_SOURCE"; then
    echo "G0 terminal retained-dispatch publication is missing line 0 or 1." >&2
    exit 1
fi
for dispatch_line in 6U 7U; do
    if ! grep -Fq "dispatch + $dispatch_line * kCacheLineBytes" "$KERNEL_SOURCE"; then
        echo "G0 terminal retained-dispatch publication is missing line $dispatch_line." >&2
        exit 1
    fi
done

poison_store_line="$(grep -nF 'StoreDev64(reinterpret_cast<__gm__ uint64_t *>(output), output_poison);' "$KERNEL_SOURCE" | cut -d: -f1)"
poison_dsb_line="$(awk '/StoreDev64\(reinterpret_cast<__gm__ uint64_t \*>\(output\), output_poison\);/ {inside=1; next} inside && /dsb\(DSB_ALL\);/ {print NR; exit}' "$KERNEL_SOURCE")"
first_workload_line="$(grep -nE 'RunG0(VectorAdd|VectorMultiply|CubeMatmul)\(' "$KERNEL_SOURCE" | head -1 | cut -d: -f1)"
last_workload_line="$(grep -nE 'RunG0(VectorAdd|VectorMultiply|CubeMatmul)\(' "$KERNEL_SOURCE" | tail -1 | cut -d: -f1)"
witness_line="$(grep -nF 'if (!PublishExecutionWitness(' "$KERNEL_SOURCE" | cut -d: -f1)"
vend_line="$(grep -nF 'token->control.completion_vend' "$KERNEL_SOURCE" | tail -1 | cut -d: -f1)"
flag_line="$(grep -nF 'g0_swimlane::AtomicSite::ScalarCompletionFlagPublish' "$KERNEL_SOURCE" | cut -d: -f1)"
done_line="$(grep -nF 'DoneState(task_id, token->control.build_owner, owner)' "$KERNEL_SOURCE" | cut -d: -f1)"
if [[ -z "$poison_store_line" || -z "$poison_dsb_line" || -z "$first_workload_line" ||
      -z "$last_workload_line" || -z "$witness_line" || -z "$vend_line" || -z "$flag_line" ||
      -z "$done_line" ]] ||
   ! (( poison_store_line < poison_dsb_line && poison_dsb_line < first_workload_line )) ||
   ! (( last_workload_line < witness_line && witness_line < vend_line && vend_line < flag_line &&
         flag_line < done_line )); then
    echo "G0 completion must poison output before workload, then follow workload -> witness -> vend -> flag -> DONE." >&2
    exit 1
fi
if ! grep -Fq 'StoreDev64(words + 7U, fanin_ready_prefix);' "$KERNEL_SOURCE" ||
   ! grep -Fq 'state, owner, task_id, kind, checksum,' "$KERNEL_SOURCE" ||
   ! grep -Fq 'token->control.fanin_ready_prefix G0_SCALAR_TRACE_ARGUMENT' "$KERNEL_SOURCE" ||
   ! grep -Fq 'witness.fanin_ready_prefix == expected.fanin_count' "$HOST_SOURCE"; then
    echo "G0 execution witness must record the runtime token fanin-ready prefix and host must validate it." >&2
    exit 1
fi
if ! grep -Fq 'constexpr uint32_t kDrainExpectedArrivals = 6U;' "$KERNEL_SOURCE" ||
   ! grep -Fq 'for (uint32_t group = 0U; group < kDrainGroupCount; ++group)' "$KERNEL_SOURCE" ||
   ! grep -Fq 'g0_swimlane::AtomicSite::ScalarDrainVerifyLoad' "$KERNEL_SOURCE" ||
   ! grep -Fq '&state->drain.builder_started.value, true' "$KERNEL_SOURCE" ||
   ! grep -Fq 'constexpr uint32_t kDrainGroupCount = 16U;' "$SIMT_ROOT/common/full_pa_exec_protocol.h"; then
    echo "G0/G1 final drain must preserve 16 groups with 6 arrivals and verify every configured builder started." >&2
    exit 1
fi
for workload in RunG0VectorAdd RunG0VectorMultiply RunG0CubeMatmul; do
    if ! grep -Fq "$workload(" "$KERNEL_SOURCE" || ! grep -Fq "void $workload(" "$WORKLOAD_SOURCE"; then
        echo "G0 is missing real workload definition/call: $workload" >&2
        exit 1
    fi
done

echo "[BUILD] CCEC G0 AIC Cube executors (dav-c310-cube)"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" "${DEVICE_VARIANT_FLAGS[@]}" --cce-aicore-arch=dav-c310-cube \
    -o "$AIC_OBJECT" "$KERNEL_SOURCE"

echo "[BUILD] CCEC G0 AIV SIMT builder/Vector executors (dav-c310-vec)"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" "${DEVICE_VARIANT_FLAGS[@]}" "${AIV_VARIANT_FLAGS[@]}" \
    --cce-aicore-arch=dav-c310-vec \
    -o "$AIV_OBJECT" "$KERNEL_SOURCE"

echo "[BUILD] CCEC G0 optimized bitcode inventory"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" "${DEVICE_VARIANT_FLAGS[@]}" --cce-aicore-arch=dav-c310-cube \
    -Xclang -emit-llvm-bc -o "$AIC_BITCODE" "$KERNEL_SOURCE"
"$CCEC" "${COMMON_DEVICE_FLAGS[@]}" "${DEVICE_VARIANT_FLAGS[@]}" "${AIV_VARIANT_FLAGS[@]}" \
    --cce-aicore-arch=dav-c310-vec \
    -Xclang -emit-llvm-bc -o "$AIV_BITCODE" "$KERNEL_SOURCE"
"$LLVM_BCANALYZER" -dump "$AIC_BITCODE" > "$AIC_BITCODE_DUMP"
"$LLVM_BCANALYZER" -dump "$AIV_BITCODE" > "$AIV_BITCODE_DUMP"

required_aiv_symbols=(
    "G0SimtBuildTasks"
    "llvm.hivm.store.vfsimt.info"
    "llvm.hivm.get.TID.X"
    "llvm.hivm.atom.CAS.G.u64"
    "llvm.hivm.atom.ADD.G.u64"
    "llvm.hivm.fence.workitems"
    "llvm.hivm.DCCI.DST"
    "llvm.hivm.vldsx1.v64f32"
    "llvm.hivm.vadd.s.x.v64f32"
    "llvm.hivm.vmul.s.x.v64f32"
    "llvm.hivm.vstsx1.v64f32"
    "llvm.hivm.SET.FLAG.IMM"
    "llvm.hivm.WAIT.FLAG.IMM"
)
for symbol in "${required_aiv_symbols[@]}"; do
    if ! grep -Fq "$symbol" "$AIV_BITCODE_DUMP"; then
        echo "optimized G0 AIV bitcode is missing: $symbol" >&2
        exit 1
    fi
done
required_aic_symbols=(
    "RunG0CubeMatmul"
    "llvm.hivm.atom.CAS.G.s64"
    "llvm.hivm.atom.ADD.G.s64"
    "llvm.hivm.DCCI.DST"
    "llvm.hivm.SET.MTE2.NZ.PARA"
    "llvm.hivm.MOV.OUT.TO.L1.MULTI.ND2NZ.U32.V310"
    "llvm.hivm.LOAD.L1.TO.L0A.2Dv2.f32"
    "llvm.hivm.LOAD.L1.TO.L0B.2Dv2.f32"
    "llvm.hivm.MAD.f322f32.c310"
    "llvm.hivm.FIX.L0C.TO.OUT.f32.EXT"
    "llvm.hivm.SET.FLAG.IMM"
    "llvm.hivm.WAIT.FLAG.IMM"
)
for symbol in "${required_aic_symbols[@]}"; do
    if ! grep -Fq "$symbol" "$AIC_BITCODE_DUMP"; then
        echo "optimized G0 AIC bitcode is missing: $symbol" >&2
        exit 1
    fi
done
echo "[CHECK] bitcode contains SIMT atomics/fence, line DCCI, Vector add/multiply, Cube matmul and drain atomics"

echo "[BUILD] Static G0 1:2 mixed AICore ELF"
"$LD_LLD" -m aicorelinux -Ttext=0 -static -o "$KERNEL_ELF" "$AIC_OBJECT" "$AIV_OBJECT"

SYMBOLS="$("$READELF_BIN" --symbols --wide --sym-base=10 "$KERNEL_ELF")"
SECTIONS="$("$READELF_BIN" --sections --wide "$KERNEL_ELF")"
RELOCATIONS="$("$READELF_BIN" --relocs --wide "$KERNEL_ELF")"
for entry in "$AIC_ENTRY" "$AIV_ENTRY"; do
    if ! awk -v name="$entry" \
        '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
         END {exit count != 1}' <<<"$SYMBOLS"; then
        echo "final G0 ELF must define one non-empty GLOBAL entry: $entry" >&2
        exit 1
    fi
    if [[ "$SECTIONS" != *".ascend.meta.$entry"* ]]; then
        echo "final G0 ELF is missing metadata: .ascend.meta.$entry" >&2
        exit 1
    fi
done
global_functions="$(awk '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" {print $NF}' <<<"$SYMBOLS")"
expected_global_functions="$(printf '%s\n%s\n' "$AIC_ENTRY" "$AIV_ENTRY")"
if [[ "$global_functions" != "$expected_global_functions" ]]; then
    echo "final G0 ELF exports unexpected GLOBAL functions:" >&2
    printf '%s\n' "$global_functions" >&2
    exit 1
fi
for local_pattern in 'G0SimtBuildTasks.*_simt_entry$' 'RunG0VectorAdd' 'RunG0VectorMultiply' 'RunG0CubeMatmul'; do
    if ! awk -v pattern="$local_pattern" \
        '$4 == "FUNC" && $5 == "LOCAL" && $3 + 0 > 0 && $NF ~ pattern {count++}
         END {exit count != 1}' <<<"$SYMBOLS"; then
        echo "final G0 ELF must retain one non-empty LOCAL function matching: $local_pattern" >&2
        exit 1
    fi
done
undefined_globals="$(awk '$5 == "GLOBAL" && $7 == "UND" {print $NF}' <<<"$SYMBOLS")"
if [[ -n "$undefined_globals" ]]; then
    echo "final G0 ELF contains undefined GLOBAL symbols:" >&2
    printf '%s\n' "$undefined_globals" >&2
    exit 1
fi
if [[ "$RELOCATIONS" != *"There are no relocations"* ]]; then
    echo "final G0 ELF unexpectedly retains relocations." >&2
    exit 1
fi

METADATA_OUTPUT="$(python3 -m msobjdump -d "$KERNEL_ELF")"
if [[ "$(grep -Fc 'KERNEL_TYPE: MIX_AIC_MAIN' <<<"$METADATA_OUTPUT")" -ne 2 ||
      "$(grep -Fc 'MIX_TASK_RATION: [1:2]' <<<"$METADATA_OUTPUT")" -ne 2 ]]; then
    echo "G0 mixed metadata is not an exact pair of MIX_AIC_MAIN [1:2] entries:" >&2
    printf '%s\n' "$METADATA_OUTPUT" >&2
    exit 1
fi
AIV_META_HEX="$("$READELF_BIN" -x ".ascend.meta.$AIV_ENTRY" "$KERNEL_ELF")"
EXPECTED_VF_STACK_META="00200000"
SIMT_SHARE_BYTES=$((8 * 1024))
if [[ "$G0_VARIANT" == "swimlane" ]]; then
    EXPECTED_VF_STACK_META="00400000"
    SIMT_SHARE_BYTES=$((16 * 1024))
fi
if [[ "$AIV_META_HEX" != *"0c000400 04000000"* ||
      "$AIV_META_HEX" != *"07000400 $EXPECTED_VF_STACK_META"* ]]; then
    echo "G0 AIV metadata must encode SIMD_SIMT_MIX_VF=4 and the variant-specific VF stack reserve." >&2
    printf '%s\n' "$AIV_META_HEX" >&2
    exit 1
fi
if ! grep -Fq 'constexpr int kG0WorkloadTile = 128;' "$WORKLOAD_SOURCE" ||
   ! grep -Fq 'TASSIGN(input_a_tile, 0x0);' "$WORKLOAD_SOURCE" ||
   ! grep -Fq 'TASSIGN(input_b_tile, 0x10000);' "$WORKLOAD_SOURCE" ||
   ! grep -Fq 'TASSIGN(output_tile, 0x20000);' "$WORKLOAD_SOURCE"; then
    echo "G0 Vector workload must retain three non-overlapping 128x128 FP32 UB tiles." >&2
    exit 1
fi
VECTOR_TILE_BYTES=$((128 * 128 * 4))
VECTOR_UB_BYTES=$((0x20000 + VECTOR_TILE_BYTES))
MAX_LOCAL_BYTES=$((224 * 1024))
if (( VECTOR_TILE_BYTES != 64 * 1024 || VECTOR_UB_BYTES != 192 * 1024 ||
      VECTOR_UB_BYTES + SIMT_SHARE_BYTES > MAX_LOCAL_BYTES )); then
    echo "G0 Vector UB plus SIMT share memory exceeds the 224 KiB A5 budget." >&2
    exit 1
fi
echo "[CHECK] ELF has exact mixed entries/functions/metadata and a $((VECTOR_UB_BYTES / 1024))+$((SIMT_SHARE_BYTES / 1024))/$((MAX_LOCAL_BYTES / 1024)) KiB AIV local-memory budget"

echo "[BUILD] GCC 15 G0 ACL host ($("$GXX15" -dumpfullversion))"
"$GXX15" -O2 -std=c++17 -Wall -Wextra -Werror \
    -Wno-deprecated-declarations \
    "-DSIMT_CROSS_CORE_G0_BUILDER_WARP_COUNT=$BUILDER_WARP_COUNT" \
    "${HOST_VARIANT_FLAGS[@]}" \
    -I"$GM_ROOT/common" \
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
    echo "G0 build did not produce the required kernel and host artifacts." >&2
    exit 1
fi

manifest_tmp="$(mktemp "$BUILD_DIR/${OUTPUT_TAG}_build_manifest.XXXXXX")"
trap 'rm -f -- "$manifest_tmp"' EXIT
(
    cd "$SIMT_ROOT"
    sha256sum \
        "${G0_BUILD_INPUTS[@]}" \
        "gm/build/ccec/simt_cross_core_${OUTPUT_TAG}_kernel.o" \
        "gm/build/ccec/simt_cross_core_${OUTPUT_TAG}_host"
) > "$manifest_tmp"
mv -f -- "$manifest_tmp" "$BUILD_MANIFEST"
trap - EXIT

echo "[BUILD] G0 CCEC complete (variant=$G0_VARIANT)"
echo "[BUILD] kernel: $KERNEL_ELF"
echo "[BUILD] host:   $HOST_BINARY"
echo "[BUILD] manifest: $BUILD_MANIFEST"
