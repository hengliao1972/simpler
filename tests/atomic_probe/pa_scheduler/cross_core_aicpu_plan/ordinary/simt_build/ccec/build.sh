#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 任一工具失败、未定义变量或管道中间失败都立即终止，避免继续使用半成品 device ELF。
set -euo pipefail

# 所有输入和产物都从脚本自身位置解析，调用者无需位于仓库根目录。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
SCALAR_ROOT="$(cd "$ROOT_DIR/../scalar_build" && pwd)"
SCALAR_COMMON_DIR="$SCALAR_ROOT/common"
SCALAR_CCEC_DIR="$SCALAR_ROOT/ccec"
AICPU_DIR="$SCALAR_ROOT/aicpu"
PA_ORCHESTRATION_SOURCE="$REPO_ROOT/examples/a5/fully_distributed_within_core/paged_attention_unroll/kernels/orchestration/paged_attention_orch.cpp"
PATH_A_DISPATCHER_SOURCE="$REPO_ROOT/tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/common/protocol_probe/plan_aicpu_dispatcher.cpp"
ARTIFACT_MANIFEST_NAME="pa_scheduler_artifacts.manifest"
RUNTIME_PLAN_BUILD_BACKEND="simt"
RUNTIME_PLAN_BUILD_BACKEND_ID=1
RUNTIME_PLAN_BUILD_WORKERS=4
RUNTIME_PLAN_BUILD_LEADERS=4
RUNTIME_PLAN_EXECUTE_WORKERS=96
ACL_STACK_ALIGNMENT_BYTES=512
ACL_SIMT_STACK_BYTES=8704
ACL_SIMT_DIVERGENCE_STACK_BYTES=8704
AIV_COMPILER_UB_BYTES=$((16 * 1024))
AIV_VECTOR_UB_BYTES=$((192 * 1024))
AIV_LOCAL_MEMORY_BYTES=$((224 * 1024))

# 目录身份固定为 AICPU Plan + ordinary TensorMap + SIMT Build。这里不
# 接受 private/shared 或任意 W 参数；Build 固定为一个 128-thread VF 的
# 四个 warp leader，Execute/FinalDrain 固定由 32 AIC + 64 AIV Scalar。
TENSORMAP_MODE="shared"
TENSORMAP_MODE_ID=1
TENSORMAP_RING_CAP=128
SHARED_INSERT_TURN_GROUPS="${PA_SHARED_INSERT_TURN_GROUPS:-1}"
case "$SHARED_INSERT_TURN_GROUPS" in
    1|2|4|8|16|32|64|128) ;;
    *)
        echo "PA_SHARED_INSERT_TURN_GROUPS must be a power of two from 1 through 128." >&2
        exit 1
        ;;
esac
# CCEC 不生成跨证据链的统一 ELF。swimlane 与 perf-clock 分别拥有
# 独立目录和编译身份；首阶段不提供 submit-PMU。
BUILD_VARIANT="${1:-swimlane}"
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [swimlane|perf-clock]" >&2
    exit 1
fi
COMPACT_GENERIC_TRACE=0
case "$BUILD_VARIANT" in
    swimlane)
        PHASE_NAME="none"
        PHASE_ID=0
        BUILD_DIR="$ROOT_DIR/build/ccec/$TENSORMAP_MODE/swimlane"
        COMPACT_GENERIC_TRACE="$TENSORMAP_MODE_ID"
        VARIANT_DEFINES=(
            "-DPTO_FDWIC_SHARED_MAP=$TENSORMAP_MODE_ID"
            -DPA_BUILD_SWIMLANE=1
            -DPA_BUILD_ATOMIC_SWIMLANE=1
            "-DPA_BUILD_COMPACT_GENERIC_TRACE=$COMPACT_GENERIC_TRACE"
            -DPA_BUILD_SUBMIT_PMU=0
            -DPA_BUILD_PERF_CLOCK=0
            -DPA_SUBMIT_PMU_PHASE_ID=0
        )
        ;;
    perf-clock)
        PHASE_NAME="none"
        PHASE_ID=0
        BUILD_DIR="$ROOT_DIR/build/ccec/$TENSORMAP_MODE/perf-clock"
        VARIANT_DEFINES=(
            "-DPTO_FDWIC_SHARED_MAP=$TENSORMAP_MODE_ID"
            -DPA_BUILD_SWIMLANE=0
            -DPA_BUILD_COMPACT_GENERIC_TRACE=0
            -DPA_BUILD_SUBMIT_PMU=0
            -DPA_BUILD_PERF_CLOCK=1
            -DPA_SUBMIT_PMU_PHASE_ID=0
        )
        ;;
    *)
        echo "Usage: $0 [swimlane|perf-clock]" >&2
        exit 1
        ;;
esac

# 物理泳道布局与传给 Host、AICore、AICPU owner/dispatcher 四产物的
# compact 编译身份来自同一组 build-side
# 变量。run.sh 会按 mode/variant 独立推导并逐字段核对，避免 producer
# 和 consumer 共用同一处错误。trace-free 变体虽然不分配泳道缓冲，仍
# 固化其编译 ABI，防止 host/kernel 交叉复用。
TRACE_SUBMIT_CLAIM_RECORD_BYTES=32
TRACE_RECORDS_PER_CORE=28416
if [[ "$COMPACT_GENERIC_TRACE" -eq 1 ]]; then
    TRACE_GENERIC_RECORD_BYTES=16
    TRACE_WORKER_STRIDE_BYTES=593920
else
    TRACE_GENERIC_RECORD_BYTES=32
    TRACE_WORKER_STRIDE_BYTES=1048576
fi

echo "[CHECK] cross-core atomic/DCCI source coverage"
PA_ATOMIC_DCCI_COVERAGE_ROOT="$SCALAR_ROOT" \
    "${PYTHON:-python3}" \
    "$ROOT_DIR/../../common/test_atomic_dcci_source_coverage.py"

# 正式 ELF 继续以真实 CPU 状态映射和 Host oracle 为最低可行门槛；
# 这些测试不模拟 A5 性能，只拒绝 mirror state、协议字段漂移和 backend
# 身份交叉消费。A5 上板由统一审计后单独执行。
echo "[CHECK] CPU SIMT protocol/task/runtime/Host feasibility gates"
"$ROOT_DIR/cpu/build_protocol.sh"
"$ROOT_DIR/cpu/build_task_builder.sh"
"$ROOT_DIR/cpu/build_real_state_runtime.sh"
"$ROOT_DIR/cpu/build_backend_host_contract.sh"
"$ROOT_DIR/cpu/build_route_policy.sh"
"$ROOT_DIR/cpu/build_host_acl_init.sh"
"$ROOT_DIR/cpu/build_writer_projection.sh"

# 正式 standalone CCEC 产物先固定使用已验证的 128×128 布局；CAP 仍
# 显式进入四产物编译身份和 manifest，避免默认值漂移后静默混件。
VARIANT_DEFINES+=("-DPTO_FDWIC_TENSORMAP_RING_CAP=$TENSORMAP_RING_CAP")
VARIANT_DEFINES+=(
    "-DPTO_FDWIC_SHARED_INSERT_TURN_GROUPS=$SHARED_INSERT_TURN_GROUPS"
    "-DPA_RUNTIME_PLAN_BUILD_BACKEND=$RUNTIME_PLAN_BUILD_BACKEND_ID"
    "-DPA_RUNTIME_PLAN_BUILD_WORKERS=$RUNTIME_PLAN_BUILD_WORKERS"
    -DPA_CCEC_BLOCK_LOCAL_STATS=1
)
echo "[BUILD] ordinary SIMT backend W=$RUNTIME_PLAN_BUILD_WORKERS execute=$RUNTIME_PLAN_EXECUTE_WORKERS, shared insert-turn groups=$SHARED_INSERT_TURN_GROUPS"

# 编译只依赖本目录源码与用户安装的 CANN/PTO 头，不引用 pa_scheduler 目录外的 simpler 构建产物。
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set; source the CANN 9.1 set_env.sh first." >&2
    exit 1
fi

CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD="$ASCEND_HOME_PATH/bin/ld.lld"
HCC="$ASCEND_HOME_PATH/tools/hcc/bin/aarch64-target-linux-gnu-g++"
CXX_BIN="${CXX:-g++}"
READELF_BIN="${READELF:-readelf}"
PTO_INCLUDE_ROOT="${PTO_ISA_ROOT:-$ASCEND_HOME_PATH/x86_64-linux}"

# ccec/ld.lld 必须来自当前已 source 的 CANN；host 编译器和 readelf 允许用户通过环境变量替换。
if [[ ! -x "$CCEC" || ! -x "$LD" || ! -x "$HCC" ]]; then
    echo "CCEC, ld.lld, or AArch64 HCC is missing under ASCEND_HOME_PATH=$ASCEND_HOME_PATH" >&2
    exit 1
fi
if ! command -v "$READELF_BIN" >/dev/null 2>&1; then
    echo "readelf is required to verify the mixed AICore ELF." >&2
    exit 1
fi
if ! command -v sha256sum >/dev/null 2>&1; then
    echo "sha256sum is required to publish the CCEC artifact manifest." >&2
    exit 1
fi
if [[ ! -f "$PTO_INCLUDE_ROOT/include/pto/common/kernel_meta.hpp" ]]; then
    echo "PTO kernel metadata header is missing under $PTO_INCLUDE_ROOT/include" >&2
    exit 1
fi
for header in pto/pto-inst.hpp pto/common/constants.hpp pto/common/pto_tile.hpp; do
    if [[ ! -f "$PTO_INCLUDE_ROOT/include/$header" ]]; then
        echo "PTO real-compute header is missing: $PTO_INCLUDE_ROOT/include/$header" >&2
        exit 1
    fi
done

mkdir -p "$BUILD_DIR"
rm -f -- "$BUILD_DIR/$ARTIFACT_MANIFEST_NAME"
# 旧复制目录可能残留 PMU owner；当前两类构建主动清除，避免运行时误加载。
rm -f \
    "$BUILD_DIR/libpa_scheduler_pmu_owner_dispatcher.so" \
    "$BUILD_DIR/libpa_scheduler_pmu_owner_aicpu.so" \
    "$BUILD_DIR/libpa_scheduler_plan_dispatcher.so" \
    "$BUILD_DIR/libpa_scheduler_plan_aicpu.so" \
    "$BUILD_DIR"/pa_scheduler_compete_first_callback_*.o

# 关闭编译器自动插入的 scalar DCCI，由 kernel.cpp 中与 PA 对齐的显式失效/回写协议负责 cache 可见性。
# 两种架构共用这些 ABI、栈和优化参数，避免 AIC/AIV 对共享 SchedulerState 产生不同解释。
COMMON_FLAGS=(
    -c -O3 -g -x cce -Wall -Werror -std=c++17
    --cce-aicore-only
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false
    -Wno-logical-op-parentheses
    -Wno-bitwise-op-parentheses
    -Wno-unused-local-typedef
    -Wno-missing-braces
    -Wno-unused-variable
    -Wno-unused-function
    -Wno-unneeded-internal-declaration
    -Wno-unused-but-set-variable
    -I"$SCALAR_COMMON_DIR"
    -I"$ROOT_DIR/common"
    -I"$ROOT_DIR/adapter"
    -I"$PTO_INCLUDE_ROOT/include"
    -I"$ASCEND_HOME_PATH/x86_64-linux/asc"
    -I"$ASCEND_HOME_PATH/x86_64-linux/asc/include"
    "${VARIANT_DEFINES[@]}"
)

# tag7 的 compiler UB 必须同时容纳 formal VF 的 SU 与 SIMT stack。
# 0x3800 叠加 A5 CCEC 默认 2 KiB 后形成 16 KiB；swimlane/perf-clock
# 共用该正式布局，避免 trace 变体碰巧有栈、trace-off 仍溢出。
AIV_STACK_FLAGS=(
    -mllvm -cce-vf-stack-size=0x3800
)

# direct Plan entry 不恢复旧 callback split，但 LocalStats 仍会跨 noinline
# Build/Execute helper 传引用。CCEC 对这种栈引用没有可靠合同，因此按已验证
# 的 A5 block-local 机制为 AIC/AIV 各保留一份精确 1152B 状态。relocate
# 让最终 mixed ELF 把两份状态排成互不重叠的连续区域。
BLOCK_LOCAL_STATS_BYTES=1152
COMMON_FLAGS+=(
    -mllvm -cce-block-local-relocate=true
    -mllvm "-cce-block-local-reserve-size=$BLOCK_LOCAL_STATS_BYTES"
)

echo "[CHECK] formal SIMT entry/adapter source contracts"
for pattern in \
    'RuntimePlanBuildIdentityPreflight(state)' \
    'RuntimePlanExternalBuildWindow window{' \
    'pa_scheduler_simt_runtime_plan_continuation_aiv(';
do
    if ! grep -Fq "$pattern" "$SCALAR_CCEC_DIR/kernel.cpp"; then
        echo "Scalar AIV continuation hook is missing: $pattern" >&2
        exit 1
    fi
done
for pattern in \
    'InitAclForCompiledRuntimePlanBuildBackend()' \
    'kCompiledRuntimePlanBuildBackend ==' \
    'RuntimePlanBuildBackend::Simt' \
    'aclInit(path)' \
    'return aclInit(nullptr);';
do
    if ! grep -Fq "$pattern" "$SCALAR_CCEC_DIR/host.cpp"; then
        echo "formal SIMT Host ACL initialization is missing: $pattern" >&2
        exit 1
    fi
done
for pattern in \
    'static __simt_vf__ __aicore__ LAUNCH_BOUND(128) void' \
    'simt::AttachClosedPlan<SimtOps>(' \
    'simt::TakeAttachedBuildTicket<SimtOps>(' \
    'runtime.BindTask(reservation.task_id)' \
    'simt::BuildCanonicalPlanTask<SimtOps>(' \
    'simt::ArriveBuilderLeaderOnce<SimtOps>(' \
    'LastInsertCompletionPublished(' \
    'simt::PublishBuildRelease<SimtOps>(' \
    'cce::async_invoke<BuildClosedCanonicalPlanVf>(' \
    'wait_flag(PIPE_V, PIPE_S, EVENT_ID0);' \
    'pa_scheduler_simt_runtime_plan_preflight_aiv(state)' \
    'pa_scheduler_simt_runtime_plan_continuation_aiv(';
do
    if ! grep -Fq "$pattern" "$SCRIPT_DIR/kernel.cpp"; then
        echo "formal SIMT entry is missing: $pattern" >&2
        exit 1
    fi
done
for pattern in \
    'struct PaSimtRoutePolicy {' \
    'aicpu_plan_adapter::ValidatePaAdapterMetadata(' \
    'PaSimtFunctionIdMatchesAdapterKind(' \
    'case TaskKind::Qk:' \
    'return function_id == 0U;' \
    'case TaskKind::Up:' \
    'return function_id == 3U;';
do
    if ! grep -Fq "$pattern" "$ROOT_DIR/adapter/pa_simt_route_policy.h"; then
        echo "PA SIMT route adapter is missing: $pattern" >&2
        exit 1
    fi
done

python3 - \
    "$SCRIPT_DIR/kernel.cpp" \
    "$ROOT_DIR/adapter/pa_simt_route_policy.h" \
    "$SCALAR_CCEC_DIR/host.cpp" \
    "$ACL_SIMT_STACK_BYTES" \
    "$ACL_SIMT_DIVERGENCE_STACK_BYTES" \
    "$ACL_STACK_ALIGNMENT_BYTES" <<'PY'
import json
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1]).read_text()
route = pathlib.Path(sys.argv[2]).read_text()
host = pathlib.Path(sys.argv[3]).read_text()
acl_simt_stack = int(sys.argv[4])
acl_dvg_stack = int(sys.argv[5])
acl_alignment = int(sys.argv[6])
ordered = [
    "const uint64_t external_build_begin =",
    "AcquireBuildIdentity(state);",
    "pa_scheduler_simt_runtime_plan_preflight_aiv(state);",
    "cce::async_invoke<BuildClosedCanonicalPlanVf>(",
    "wait_flag(PIPE_V, PIPE_S, EVENT_ID0);",
    "const uint64_t external_build_end =",
]
positions = [source.index(token) for token in ordered]
positions.append(source.rindex("pa_scheduler_simt_runtime_plan_continuation_aiv("))
if positions != sorted(positions):
    raise SystemExit(
        "formal order must be lifecycle begin -> acquire -> read-only preflight -> VF -> V/S join -> Scalar continuation"
    )
if source.count("identity_preflight_ok == 0U") != 1:
    raise SystemExit(
        "formal VF must contain exactly one complete identity-preflight guard"
    )

for forbidden in (
    "task_id % 5",
    "task_id%5",
    "FullPaTaskPlan",
    "DecodePaRuntimeTaskPlan",
    "SharedPaTaskOffset",
    "TaskKind",
    "HostTaskPlan",
    "first_task_id",
):
    if forbidden in source:
        raise SystemExit(
            "formal generic Build entry contains forbidden PA/Host reconstruction: "
            + forbidden
        )

for forbidden in (
    "SharedPaFunctionIdMatches(",
    "FunctionId(",
    "task_id %",
    "task_id%",
    "SharedPaTaskOffset(",
):
    if forbidden in route:
        raise SystemExit(
            "SIMT route function-id validation reconstructs PA identity: "
            + forbidden
        )

config_match = re.search(
    r'constexpr char kConfig\[\] = R"acl\((.*?)\)acl";', host, re.DOTALL
)
if config_match is None:
    raise SystemExit("formal SIMT Host must keep one auditable raw ACL config")
config = json.loads(config_match.group(1))
if config != {
    "StackSize": {
        "simt_stack_size": acl_simt_stack,
        "simt_divergence_stack_size": acl_dvg_stack,
    }
}:
    raise SystemExit("formal SIMT Host ACL stack config does not match build identity")
if (
    acl_simt_stack % acl_alignment != 0
    or acl_dvg_stack % acl_alignment != 0
):
    raise SystemExit("formal SIMT ACL capacities must be 512-byte aligned")

ordered_host = [
    "kCompiledRuntimePlanBuildBackend ==",
    "RuntimePlanBuildBackend::Simt",
    "mkstemp(path)",
    "write(fd, kConfig + written",
    "const aclError init_error = aclInit(path);",
    "if (unlink(path) != 0)",
    "return init_error;",
    "return aclInit(nullptr);",
    'CheckAcl(InitAclForCompiledRuntimePlanBuildBackend(), "aclInit")',
]
host_positions = [host.index(token) for token in ordered_host]
if host_positions != sorted(host_positions):
    raise SystemExit(
        "formal Host must select backend, publish/close temp config, initialize, "
        "unlink, preserve Scalar aclInit(nullptr), then call the helper"
    )
if host.count("aclInit(path)") != 1 or host.count("aclInit(nullptr)") != 1:
    raise SystemExit("formal Host must retain exactly one SIMT and one Scalar aclInit call")
if host.count("(void)unlink(path);") != 2:
    raise SystemExit("formal Host must unlink the temp config on write/close failure")
PY

# 正式 shared PA entry 已实例化 ordered writer-delta 路径；reader
# progress/reclaim 与旧 WriterIntentSet 仍只保留为隔离协议原语。构建
# 额外对这些模板做真实后端代码生成和静态链接，分别锁定 cube/vector
# 的 CcecOps、GM 地址空间、atomicCAS、DCCI 以及“无未解析 compiler
# builtin”契约。probe 不加入 DEVICE_OBJECTS，
# 检查后立即删除，因此不会改变正式 mixed ELF、I-cache 布局或运行性能。
(
    SHARED_PROTOCOL_PROBE_AIC_OBJECT="$BUILD_DIR/.shared_protocol_probe_aic.o"
    SHARED_PROTOCOL_PROBE_AIV_OBJECT="$BUILD_DIR/.shared_protocol_probe_aiv.o"
    SHARED_PROTOCOL_PROBE_AIC_ELF="$BUILD_DIR/.shared_protocol_probe_aic.elf"
    SHARED_PROTOCOL_PROBE_AIV_ELF="$BUILD_DIR/.shared_protocol_probe_aiv.elf"
    cleanup_shared_protocol_probe() {
        rm -f \
            "$SHARED_PROTOCOL_PROBE_AIC_OBJECT" \
            "$SHARED_PROTOCOL_PROBE_AIV_OBJECT" \
            "$SHARED_PROTOCOL_PROBE_AIC_ELF" \
            "$SHARED_PROTOCOL_PROBE_AIV_ELF"
    }
    # 独立子 shell 的 EXIT trap 不会覆盖正式 manifest 的原子发布 trap；
    # 编译或链接任一步失败也会清掉隐藏 probe，不留下半成品混淆现场。
    trap cleanup_shared_protocol_probe EXIT
    cleanup_shared_protocol_probe

    echo "[CHECK] CCEC AIC generic shared protocol instantiation"
    "$CCEC" "${COMMON_FLAGS[@]}" \
        --cce-aicore-arch=dav-c310-cube \
        -DPA_BUILD_AIC \
        -o "$SHARED_PROTOCOL_PROBE_AIC_OBJECT" \
        "$SCALAR_CCEC_DIR/shared_protocol_compile_probe.cpp"
    "$LD" -m aicorelinux -Ttext=0 -static \
        -o "$SHARED_PROTOCOL_PROBE_AIC_ELF" \
        "$SHARED_PROTOCOL_PROBE_AIC_OBJECT"

    echo "[CHECK] CCEC AIV generic shared protocol instantiation"
    "$CCEC" "${COMMON_FLAGS[@]}" \
        "${AIV_STACK_FLAGS[@]}" \
        --cce-aicore-arch=dav-c310-vec \
        -DPA_BUILD_AIV \
        -o "$SHARED_PROTOCOL_PROBE_AIV_OBJECT" \
        "$SCALAR_CCEC_DIR/shared_protocol_compile_probe.cpp"
    "$LD" -m aicorelinux -Ttext=0 -static \
        -o "$SHARED_PROTOCOL_PROBE_AIV_ELF" \
        "$SHARED_PROTOCOL_PROBE_AIV_OBJECT"
)

# AIC 复用 Scalar 正式入口；AIV 拆成一个只含 hidden continuation 的
# Scalar identity object，以及唯一拥有 global AIV entry/SIMT VF 的 object。
# 三者必须使用完全相同的 backend/W/trace ABI 定义。
echo "[BUILD] CCEC Scalar AIC entry (dav-c310-cube)"
"$CCEC" "${COMMON_FLAGS[@]}" \
    --cce-aicore-arch=dav-c310-cube \
    -DPA_BUILD_AIC \
    -o "$BUILD_DIR/pa_scheduler_aic.o" \
    "$SCALAR_CCEC_DIR/kernel.cpp"

echo "[BUILD] CCEC hidden Scalar AIV continuation (dav-c310-vec)"
"$CCEC" "${COMMON_FLAGS[@]}" \
    "${AIV_STACK_FLAGS[@]}" \
    --cce-aicore-arch=dav-c310-vec \
    -DPA_BUILD_AIV \
    -DPA_SIMT_EXTERNAL_BUILD_AIV_HELPERS \
    -o "$BUILD_DIR/pa_scheduler_aiv_continuation.o" \
    "$SCALAR_CCEC_DIR/kernel.cpp"

echo "[BUILD] CCEC unique SIMT AIV entry (dav-c310-vec, 128 threads/4 leaders)"
"$CCEC" "${COMMON_FLAGS[@]}" \
    "${AIV_STACK_FLAGS[@]}" \
    --cce-aicore-arch=dav-c310-vec \
    -o "$BUILD_DIR/pa_scheduler_aiv_entry.o" \
    "$SCRIPT_DIR/kernel.cpp"

if command -v llvm-bcanalyzer >/dev/null 2>&1; then
    LLVM_BCANALYZER="$(command -v llvm-bcanalyzer)"
else
    LLVM_BCANALYZER="/opt/mlir-debug/bin/llvm-bcanalyzer"
fi
if [[ ! -x "$LLVM_BCANALYZER" ]]; then
    echo "llvm-bcanalyzer is required for the formal SIMT IR gate." >&2
    exit 1
fi
echo "[BUILD] optimized formal AIV entry/continuation LLVM bitcode"
"$CCEC" "${COMMON_FLAGS[@]}" \
    "${AIV_STACK_FLAGS[@]}" \
    --cce-aicore-arch=dav-c310-vec \
    -Xclang -emit-llvm-bc \
    -o "$BUILD_DIR/pa_scheduler_aiv_entry.bc" \
    "$SCRIPT_DIR/kernel.cpp"
"$CCEC" "${COMMON_FLAGS[@]}" \
    "${AIV_STACK_FLAGS[@]}" \
    --cce-aicore-arch=dav-c310-vec \
    -DPA_BUILD_AIV \
    -DPA_SIMT_EXTERNAL_BUILD_AIV_HELPERS \
    -Xclang -emit-llvm-bc \
    -o "$BUILD_DIR/pa_scheduler_aiv_continuation.bc" \
    "$SCALAR_CCEC_DIR/kernel.cpp"
"$LLVM_BCANALYZER" -dump \
    "$BUILD_DIR/pa_scheduler_aiv_entry.bc" > \
    "$BUILD_DIR/pa_scheduler_aiv_entry.bc.dump"
"$LLVM_BCANALYZER" -dump \
    "$BUILD_DIR/pa_scheduler_aiv_continuation.bc" > \
    "$BUILD_DIR/pa_scheduler_aiv_continuation.bc.dump"

for symbol in \
    'BuildClosedCanonicalPlanVf' \
    'AttachClosedPlan' \
    'TakeAttachedBuildTicket' \
    'BindTask' \
    'BuildCanonicalPlanTask' \
    'ArriveBuilderLeaderOnce' \
    'PublishBuildRelease' \
    'llvm.hivm.store.vfsimt.info' \
    'llvm.hivm.get.TID.X' \
    'llvm.hivm.atom.ADD.G.s64' \
    'llvm.hivm.atom.CAS.G.s64' \
    'llvm.hivm.atom.EXCH.G.s64' \
    'llvm.hivm.DCCI.DST' \
    'llvm.hivm.fence.workitems' \
    'llvm.hivm.stg.uncache.b64' \
    'llvm.hivm.SET.FLAG.IMM' \
    'llvm.hivm.WAIT.FLAG.IMM';
do
    if ! grep -Fq "$symbol" "$BUILD_DIR/pa_scheduler_aiv_entry.bc.dump"; then
        echo "formal SIMT AIV entry IR is missing: $symbol" >&2
        exit 1
    fi
done
for symbol in \
    'pa_scheduler_simt_runtime_plan_preflight_aiv' \
    'pa_scheduler_simt_runtime_plan_continuation_aiv' \
    'RuntimePlanExternalBuildWindow' \
    'RunScheduler';
do
    if ! grep -Fq "$symbol" "$BUILD_DIR/pa_scheduler_aiv_continuation.bc.dump"; then
        echo "formal Scalar continuation IR is missing: $symbol" >&2
        exit 1
    fi
done
echo "[CHECK] optimized IR retains full SIMT Build and Scalar continuation"

check_workload_dispatcher_object() {
    local object_path="$1"
    local expected_symbol="$2"
    local wrong_role_symbol="$3"
    local object_symbols
    object_symbols="$("$READELF_BIN" --symbols --wide --sym-base=10 "$object_path")"
    if ! awk -v name="$expected_symbol" \
        '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
         END {exit count != 1}' <<<"$object_symbols"; then
        echo "Expected exactly one non-empty strong workload dispatcher in $object_path: $expected_symbol" >&2
        exit 1
    fi
    if awk -v name="$wrong_role_symbol" \
        '$NF == name {found = 1} END {exit !found}' <<<"$object_symbols"; then
        echo "Wrong-role workload dispatcher leaked into $object_path: $wrong_role_symbol" >&2
        exit 1
    fi
}
check_workload_dispatcher_object \
    "$BUILD_DIR/pa_scheduler_aic.o" \
    pa_execute_real_winner_workload_aic \
    pa_execute_real_winner_workload_aiv
check_workload_dispatcher_object \
    "$BUILD_DIR/pa_scheduler_aiv_continuation.o" \
    pa_execute_real_winner_workload_aiv \
    pa_execute_real_winner_workload_aic
echo "[CHECK] role-specific real-compute dispatchers are strong and do not cross roles"

check_block_local_stats_object() {
    local object_path="$1"
    local expected_symbol="$2"
    local object_symbols section_layout section_size_hex section_alignment
    object_symbols="$("$READELF_BIN" --symbols --wide --sym-base=10 "$object_path")"
    if ! awk -v name="$expected_symbol" -v bytes="$BLOCK_LOCAL_STATS_BYTES" \
        '$4 == "OBJECT" && $5 == "GLOBAL" && $7 != "UND" &&
         $NF == name && $3 + 0 == bytes {count++}
         END {exit count != 1}' <<<"$object_symbols"; then
        echo "Expected one exact-size role-local LocalStats object: $object_path ($expected_symbol)" >&2
        exit 1
    fi
    section_layout="$(
        "$READELF_BIN" --sections --wide "$object_path" |
            awk '{for (column = 1; column <= NF; ++column) {
                if ($column == ".bl.uninit") {
                    print $(column + 4), $NF
                    exit
                }
            }}'
    )"
    read -r section_size_hex section_alignment <<<"$section_layout"
    if [[ -z "$section_size_hex" || -z "$section_alignment" ]]; then
        echo "Missing role-local LocalStats section: $object_path" >&2
        exit 1
    fi
    if ((16#$section_size_hex != BLOCK_LOCAL_STATS_BYTES)) ||
       ((section_alignment != 64)); then
        echo "Role-local LocalStats section must be exactly ${BLOCK_LOCAL_STATS_BYTES}B and 64B aligned: $object_path" >&2
        exit 1
    fi
}
check_block_local_stats_object \
    "$BUILD_DIR/pa_scheduler_aic.o" \
    pa_scheduler_plan_local_stats_aic
check_block_local_stats_object \
    "$BUILD_DIR/pa_scheduler_aiv_continuation.o" \
    pa_scheduler_plan_local_stats_aiv
echo "[CHECK] AIC/AIV direct entries each own one exact block-local LocalStats"

DEVICE_OBJECTS=(
    "$BUILD_DIR/pa_scheduler_aic.o"
    "$BUILD_DIR/pa_scheduler_aiv_continuation.o"
    "$BUILD_DIR/pa_scheduler_aiv_entry.o"
)
echo "[CHECK] closed-Plan build uses AIC + hidden AIV continuation + unique SIMT AIV entry"

# 静态链接把两个 device object 合成一个可由 runtime 按 1:2 比例启动的 mixed AICore ELF。
echo "[BUILD] Static 1:2 mixed AICore ELF"
"$LD" -m aicorelinux -Ttext=0 -static \
    --version-script="$SCRIPT_DIR/pa_scheduler_device_exports.map" \
    -o "$BUILD_DIR/pa_scheduler_kernel.o" \
    "${DEVICE_OBJECTS[@]}"

SYMBOL_TABLE="$("$READELF_BIN" --symbols --wide --sym-base=10 "$BUILD_DIR/pa_scheduler_kernel.o")"
SECTION_TABLE="$("$READELF_BIN" --sections --wide "$BUILD_DIR/pa_scheduler_kernel.o")"
# 构建成功不等于 mixed launch 可用：同时检查两个入口符号及其 metadata section，缺一即拒绝产物。
# `set -e` 同时保证 readelf 自身失败时不会拿空字符串继续做伪检查。
for entry in pa_scheduler_0_mix_aic pa_scheduler_0_mix_aiv; do
    if ! awk -v name="$entry" \
        '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 != "0" && $3 != "0x0" {found = 1} END {exit !found}' \
        <<<"$SYMBOL_TABLE"; then
        echo "Missing non-empty defined GLOBAL mixed-kernel entry: $entry" >&2
        exit 1
    fi
    if [[ "$SECTION_TABLE" != *".ascend.meta.$entry"* ]]; then
        echo "Missing mixed-kernel metadata section: .ascend.meta.$entry" >&2
        exit 1
    fi
done
echo "[CHECK] both 1:2 mixed entries and metadata sections are present"

# perf-clock 最终 ELF 必须证明最重的泳道写记录慢体已经在编译期消失。
# 正向身份由 manifest SHA 和运行时 build_variant 双重闭合；不额外向
# `.text` 塞 marker，避免仅用于取证的代码改变后续热函数 I-cache 对齐。
if [[ "$BUILD_VARIANT" == "perf-clock" ]]; then
    if awk \
        '$7 != "UND" && index($NF, "WritePollBatchRecordRaw") != 0 {found = 1}
         END {exit !found}' <<<"$SYMBOL_TABLE"; then
        echo "Swimlane record writer leaked into perf-clock AICore ELF." >&2
        exit 1
    fi
    echo "[CHECK] perf-clock swimlane record writer is absent; identity uses manifest/runtime handshake"
fi

# A5 runtime 会把已定义的 GLOBAL FUNC 当作可启动候选；最终 device ELF 只允许
# 两个带 metadata 的 mixed 入口暴露为全局函数。任何新增 helper 都必须保持 LOCAL。
while IFS= read -r global_func; do
    case "$global_func" in
        pa_scheduler_0_mix_aic|pa_scheduler_0_mix_aiv) ;;
        *)
            echo "Unexpected GLOBAL device function (possible kernel-entry pollution): $global_func" >&2
            exit 1
            ;;
    esac
done < <(awk '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" {print $NF}' <<<"$SYMBOL_TABLE")
echo "[CHECK] only the two mixed entries are exported as GLOBAL device functions"

for helper in \
    pa_scheduler_simt_runtime_plan_preflight_aiv \
    pa_scheduler_simt_runtime_plan_continuation_aiv;
do
    if ! awk -v name="$helper" \
        '$4 == "FUNC" && $5 == "LOCAL" && $7 != "UND" &&
         $NF == name && $3 + 0 > 0 {count++}
         END {exit count != 1}' <<<"$SYMBOL_TABLE"; then
        echo "formal SIMT helper must resolve as one non-empty LOCAL function: $helper" >&2
        exit 1
    fi
done
if ! awk \
    '$4 == "FUNC" && $5 == "LOCAL" && $7 != "UND" &&
     $NF ~ /BuildClosedCanonicalPlanVf.*_simt_entry$/ &&
     $3 + 0 > 0 {count++}
     END {exit count != 1}' <<<"$SYMBOL_TABLE"; then
    echo "final ELF must retain one non-empty LOCAL 128-thread Build VF." >&2
    exit 1
fi
if awk '$5 == "GLOBAL" && $7 == "UND" {found = 1} END {exit !found}' \
    <<<"$SYMBOL_TABLE"; then
    echo "Final ordinary SIMT ELF contains undefined GLOBAL symbols." >&2
    exit 1
fi

AIV_META_HEX="$($READELF_BIN -x .ascend.meta.pa_scheduler_0_mix_aiv "$BUILD_DIR/pa_scheduler_kernel.o")"
read -r \
    COMPILER_ALLOC_UB_BYTES \
    SU_STACK_BYTES \
    SIMT_WARP_STACK_BYTES \
    SIMT_DVG_STACK_BYTES < <(
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
        raise SystemExit("truncated formal AIV metadata TLV")
    values[tag] = int.from_bytes(blob[offset : offset + size], "little")
    offset += size
print(*(values.get(tag, -1) for tag in (7, 8, 9, 10)))
PY
)
if [[ "$AIV_META_HEX" != *'0c000400 04000000'* ]] ||
   (( COMPILER_ALLOC_UB_BYTES != AIV_COMPILER_UB_BYTES ||
      SU_STACK_BYTES <= 0 || SIMT_WARP_STACK_BYTES <= 0 ||
      SIMT_DVG_STACK_BYTES <= 0 ||
      SU_STACK_BYTES + SIMT_WARP_STACK_BYTES > COMPILER_ALLOC_UB_BYTES ||
      SIMT_WARP_STACK_BYTES > ACL_SIMT_STACK_BYTES ||
      SIMT_DVG_STACK_BYTES > ACL_SIMT_DIVERGENCE_STACK_BYTES ||
      ACL_SIMT_STACK_BYTES % ACL_STACK_ALIGNMENT_BYTES != 0 ||
      ACL_SIMT_DIVERGENCE_STACK_BYTES % ACL_STACK_ALIGNMENT_BYTES != 0 ||
      AIV_VECTOR_UB_BYTES + COMPILER_ALLOC_UB_BYTES > AIV_LOCAL_MEMORY_BYTES )); then
    echo "formal AIV metadata must encode MIX_VF=4, fit SU+SIMT in 16 KiB compiler UB, fit the 8704/8704 B ACL config, and preserve the 192+16/224 KiB A5 local budget." >&2
    printf '%s\n' "$AIV_META_HEX" >&2
    exit 1
fi
echo "[CHECK] formal AIV stacks: compiler_ub=$COMPILER_ALLOC_UB_BYTES SU=$SU_STACK_BYTES SIMT=$SIMT_WARP_STACK_BYTES DVG=$SIMT_DVG_STACK_BYTES ACL=$ACL_SIMT_STACK_BYTES/$ACL_SIMT_DIVERGENCE_STACK_BYTES bytes"
echo "[CHECK] one LOCAL VF, hidden continuation/preflight, MIX_VF=4, and $((AIV_VECTOR_UB_BYTES / 1024))+$((COMPILER_ALLOC_UB_BYTES / 1024))/$((AIV_LOCAL_MEMORY_BYTES / 1024)) KiB A5 local budget"

# direct mixed entry 调用按核型区分的真计算 dispatcher；version script
# 将其与底层 Cube/Vector 实体全部局部化，最终只保留两个 mixed 入口。
for workload_symbol in \
    pa_execute_real_winner_workload_aic \
    pa_execute_real_winner_workload_aiv \
    pa_real_cube_workload_aic \
    pa_real_vector_add_workload_aiv \
    pa_real_vector_mul_workload_aiv; do
    workload_size="$(
        awk -v name="$workload_symbol" \
            '$4 == "FUNC" && $5 == "LOCAL" && $7 != "UND" && index($NF, name) != 0 && $3 + 0 > 0 {print $3; exit}' \
            <<<"$SYMBOL_TABLE"
    )"
    if [[ -z "$workload_size" ]]; then
        echo "Missing non-empty LOCAL CCEC real-compute workload function: $workload_symbol" >&2
        exit 1
    fi
    if awk -v name="$workload_symbol" \
        '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && index($NF, name) != 0 {found = 1} END {exit !found}' \
        <<<"$SYMBOL_TABLE"; then
        echo "CCEC real-compute helper must not be a GLOBAL kernel candidate: $workload_symbol" >&2
        exit 1
    fi
done
echo "[CHECK] CCEC cube/vector real-compute helpers are non-empty LOCAL functions"

aic_stats_offset="$(
    awk '$4 == "OBJECT" && $5 == "LOCAL" &&
         $NF == "pa_scheduler_plan_local_stats_aic" {print $2; exit}' \
        <<<"$SYMBOL_TABLE"
)"
aiv_stats_offset="$(
    awk '$4 == "OBJECT" && $5 == "LOCAL" &&
         $NF == "pa_scheduler_plan_local_stats_aiv" {print $2; exit}' \
        <<<"$SYMBOL_TABLE"
)"
final_block_local_layout="$(
    awk '{for (column = 1; column <= NF; ++column) {
        if ($column == ".bl_uninit") {
            print $(column + 4), $NF
            exit
        }
    }}' <<<"$SECTION_TABLE"
)"
read -r final_block_local_size_hex final_block_local_alignment \
    <<<"$final_block_local_layout"
if [[ -z "$aic_stats_offset" || -z "$aiv_stats_offset" ||
      -z "$final_block_local_size_hex" ||
      -z "$final_block_local_alignment" ]]; then
    echo "Final mixed ELF is missing its role-local LocalStats layout." >&2
    exit 1
fi
if ((16#$aic_stats_offset != 0)) ||
   ((16#$aiv_stats_offset != BLOCK_LOCAL_STATS_BYTES)) ||
   ((16#$final_block_local_size_hex != 2 * BLOCK_LOCAL_STATS_BYTES)) ||
   ((final_block_local_alignment != 64)); then
    echo "Final mixed ELF must contain two non-overlapping 1152B role-local LocalStats objects." >&2
    exit 1
fi
echo "[CHECK] final mixed ELF keeps two non-overlapping block-local LocalStats objects"

if [[ -n "$("$READELF_BIN" --relocs --wide "$BUILD_DIR/pa_scheduler_kernel.o" |
      sed -n '/Relocation section/p')" ]]; then
    echo "Final closed-Plan mixed ELF must not retain relocations." >&2
    exit 1
fi
echo "[CHECK] final closed-Plan ELF has no relocations"

# AICPU owner 独立编译真实 PA orchestration。Host 只把通用输入元数据和
# 空 Plan 存储地址交给它；task identity 只能由 callback backend 产生。
AICPU_COMMON_INCLUDES=(
    -I"$REPO_ROOT/src/a5/platform/include"
    -I"$REPO_ROOT/src/common/platform/include"
    -I"$REPO_ROOT/src/common/task_interface"
    -I"$REPO_ROOT/src/common/log/include"
    -I"$REPO_ROOT/src/common"
    -I"$REPO_ROOT/src/a5/runtime/fully_distributed_within_core/runtime"
    -I"$REPO_ROOT/src/a5/runtime/fully_distributed_within_core/common"
    -I"$REPO_ROOT/src/a5/runtime"
    -I"$REPO_ROOT/src/a5/runtime/fully_distributed_within_core/orchestration"
    -I"$AICPU_DIR"
)
PLAN_OWNER_SO="$BUILD_DIR/libpa_scheduler_plan_aicpu.so"
PLAN_DISPATCHER_SO="$BUILD_DIR/libpa_scheduler_plan_dispatcher.so"
echo "[BUILD] AICPU real-PA Plan owner"
"$HCC" -std=c++17 -O3 -g -fPIC -fno-gnu-unique \
    -Wall -Wextra -Werror -Wno-unused-but-set-parameter \
    -DPTO_FDWIC_SHARED_MAP=1 -DPTO_FDWIC_SCHEDULER_MODE=1 \
    "-DPA_RUNTIME_PLAN_BUILD_BACKEND=$RUNTIME_PLAN_BUILD_BACKEND_ID" \
    "-DPA_RUNTIME_PLAN_BUILD_WORKERS=$RUNTIME_PLAN_BUILD_WORKERS" \
    "${AICPU_COMMON_INCLUDES[@]}" \
    -shared -Wl,-z,defs -Wl,--build-id \
    "$PA_ORCHESTRATION_SOURCE" \
    "$AICPU_DIR/aicpu_plan_backend.cpp" \
    "$AICPU_DIR/aicpu_plan_adapter_bridge.cpp" \
    "$AICPU_DIR/aicpu_plan_owner.cpp" \
    -o "$PLAN_OWNER_SO"

echo "[BUILD] verified Path-A bootstrap dispatcher"
"$HCC" -shared -fPIC -O3 -g -std=gnu++17 \
    -Wall -Wextra -Werror -Wl,--build-id \
    "$PATH_A_DISPATCHER_SOURCE" \
    -o "$PLAN_DISPATCHER_SO"

for artifact in "$PLAN_OWNER_SO" "$PLAN_DISPATCHER_SO"; do
    "$READELF_BIN" -h "$artifact" | awk -F: \
        '/Machine:/ { if ($2 !~ /AArch64/) exit 1; found = 1 }
         END { exit(found ? 0 : 1) }'
done
for symbol in \
    plan_protocol_aicpu_exec \
    aicpu_orchestration_entry \
    aicpu_orchestration_config \
    aicpu_plan_backend_bind \
    aicpu_plan_backend_close \
    aicpu_plan_backend_result; do
    "$READELF_BIN" -Ws "$PLAN_OWNER_SO" | awk -v symbol="$symbol" \
        '$7 != "UND" && $8 == symbol && $3 + 0 > 0 {found = 1}
         END {exit(found ? 0 : 1)}'
done
if "$READELF_BIN" -Ws "$PLAN_OWNER_SO" | awk \
    '$7 == "UND" && ($8 ~ /^dist_/ || $8 ~ /^aicpu_plan_adapter_/) {found = 1}
     END {exit(found ? 0 : 1)}'; then
    echo "AICPU Plan owner retains an unresolved callback/backend symbol." >&2
    exit 1
fi
for symbol in \
    StaticTileFwkBackendKernelServer \
    DynTileFwkBackendKernelServer \
    DynTileFwkBackendKernelServerInit; do
    "$READELF_BIN" -Ws "$PLAN_DISPATCHER_SO" | awk -v symbol="$symbol" \
        '$7 != "UND" && $8 == symbol && $3 + 0 > 0 {found = 1}
         END {exit(found ? 0 : 1)}'
done
echo "[CHECK] AICPU owner/dispatcher machine, entry, and closure gates passed"

# host runner 只链接用户 CANN 9.1 的 ACL/runtime，并写入同一安装目录的 rpath，运行时不需要 simpler 动态库。
# `-Werror` 让 host API 签名或尺寸类型变化在构建期暴露，避免到上板阶段才出现参数截断。
echo "[BUILD] CCEC host runner"
"$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror -Wno-deprecated-declarations \
    "${VARIANT_DEFINES[@]}" \
    -I"$SCALAR_COMMON_DIR" \
    -I"$ASCEND_HOME_PATH/include" \
    -I"$ASCEND_HOME_PATH/pkg_inc" \
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
    "$SCALAR_CCEC_DIR/host.cpp" \
    -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -lascendcl -lruntime \
    -ldl \
    -o "$BUILD_DIR/pa_scheduler_host"

# host、kernel、AICPU owner 与 dispatcher 全部成功后才发布统一
# manifest。v6 同时固化 closed-Plan ABI、producer 入口、SIMT backend、
# 四个 Build leader 与最终 96 Scalar Execute population；
# run.sh 只消费带完整 manifest 的目录，因此中断重编不会混用新旧镜像。
ARTIFACTS=(
    pa_scheduler_host
    pa_scheduler_kernel.o
    libpa_scheduler_plan_aicpu.so
    libpa_scheduler_plan_dispatcher.so
)
for artifact in "${ARTIFACTS[@]}"; do
    if [[ ! -s "$BUILD_DIR/$artifact" ]]; then
        echo "Cannot publish CCEC manifest; artifact is missing or empty: $artifact" >&2
        exit 1
    fi
done
if [[ ! -x "$BUILD_DIR/pa_scheduler_host" ]]; then
    echo "Cannot publish CCEC manifest; host runner is not executable." >&2
    exit 1
fi

MANIFEST_PATH="$BUILD_DIR/$ARTIFACT_MANIFEST_NAME"
MANIFEST_TMP="$(mktemp "$BUILD_DIR/.${ARTIFACT_MANIFEST_NAME}.tmp.XXXXXX")"
cleanup_manifest_tmp() {
    if [[ -n "${MANIFEST_TMP:-}" ]]; then
        rm -f -- "$MANIFEST_TMP"
    fi
}
trap cleanup_manifest_tmp EXIT
{
    printf '# schema=pa_scheduler_artifacts/v6\n'
    printf '# tensormap_mode=%s\n' "$TENSORMAP_MODE"
    printf '# tensormap_mode_id=%u\n' "$TENSORMAP_MODE_ID"
    printf '# tensormap_ring_cap=%u\n' "$TENSORMAP_RING_CAP"
    printf '# shared_insert_turn_groups=%u\n' \
        "$SHARED_INSERT_TURN_GROUPS"
    printf '# generic_record_bytes=%u\n' \
        "$TRACE_GENERIC_RECORD_BYTES"
    printf '# submit_claim_record_bytes=%u\n' \
        "$TRACE_SUBMIT_CLAIM_RECORD_BYTES"
    printf '# records_per_core=%u\n' \
        "$TRACE_RECORDS_PER_CORE"
    printf '# worker_stride_bytes=%u\n' \
        "$TRACE_WORKER_STRIDE_BYTES"
    printf '# variant=%s\n' "$BUILD_VARIANT"
    printf '# phase=%s\n' "$PHASE_NAME"
    printf '# phase_id=%u\n' "$PHASE_ID"
    printf '# runtime_plan_abi=%u\n' 2
    printf '# runtime_plan_cell_bytes=%u\n' 4608
    printf '# runtime_plan_capacity=%u\n' 4352
    printf '# plan_owner_entry=%s\n' plan_protocol_aicpu_exec
    printf '# scheduler_input=%s\n' aicpu_closed_runtime_plan
    printf '# runtime_plan_build_backend=%s\n' \
        "$RUNTIME_PLAN_BUILD_BACKEND"
    printf '# runtime_plan_build_backend_id=%u\n' \
        "$RUNTIME_PLAN_BUILD_BACKEND_ID"
    printf '# runtime_plan_build_workers=%u\n' \
        "$RUNTIME_PLAN_BUILD_WORKERS"
    printf '# runtime_plan_build_leaders=%u\n' \
        "$RUNTIME_PLAN_BUILD_LEADERS"
    printf '# runtime_plan_execute_workers=%u\n' \
        "$RUNTIME_PLAN_EXECUTE_WORKERS"
    (cd "$BUILD_DIR" && sha256sum "${ARTIFACTS[@]}")
} > "$MANIFEST_TMP"
mv -f -- "$MANIFEST_TMP" "$MANIFEST_PATH"
MANIFEST_TMP=""
trap - EXIT
echo "[CHECK] CCEC artifact manifest published: $MANIFEST_PATH"

echo "[BUILD] complete: $BUILD_DIR"
