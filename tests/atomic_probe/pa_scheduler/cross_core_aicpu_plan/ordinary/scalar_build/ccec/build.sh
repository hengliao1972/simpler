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
AICPU_DIR="$ROOT_DIR/aicpu"
PA_ORCHESTRATION_SOURCE="$REPO_ROOT/examples/a5/fully_distributed_within_core/paged_attention_unroll/kernels/orchestration/paged_attention_orch.cpp"
PATH_A_DISPATCHER_SOURCE="$REPO_ROOT/tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/common/protocol_probe/plan_aicpu_dispatcher.cpp"
ARTIFACT_MANIFEST_NAME="pa_scheduler_artifacts.manifest"

# 目录身份固定为 cross-core shared PA。这里不再接受 private/shared 参数，
# 防止将 same-core 或 private 产物误放进本目录。
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
# Pipeline policy is part of every compiled image and its output path.  Accept
# the stable human names at the shell boundary, but always pass the common
# header its numeric 0/1 ABI.
PIPELINE_POLICY_INPUT="${PA_RUNTIME_PLAN_PIPELINE_POLICY:-0}"
case "$PIPELINE_POLICY_INPUT" in
    0|plan-ahead-closed)
        PIPELINE_POLICY_ID=0
        PIPELINE_NAME="plan-ahead-closed"
        PIPELINE_KEY="$PIPELINE_NAME"
        LAUNCH_ORDER="plan-sync-before-aicore"
        PRODUCER_READY="closed"
        CONSUMER_ADMISSION="closed-only"
        READY_PREFILL_TASKS=0
        SCHEDULER_INPUT="aicpu_closed_runtime_plan"
        if [[ -n "${PA_AICPU_PLAN_READY_PREFILL_TASKS+x}" ]]; then
            echo "plan-ahead-closed does not accept PA_AICPU_PLAN_READY_PREFILL_TASKS." >&2
            exit 1
        fi
        ;;
    1|streaming-future)
        PIPELINE_POLICY_ID=1
        PIPELINE_NAME="streaming-future"
        LAUNCH_ORDER="dual-stream-overlap"
        PRODUCER_READY="prefill"
        CONSUMER_ADMISSION="ready-future-ticket"
        READY_PREFILL_TASKS="${PA_AICPU_PLAN_READY_PREFILL_TASKS:-128}"
        if [[ ! "$READY_PREFILL_TASKS" =~ ^[0-9]+$ ]] ||
           ((READY_PREFILL_TASKS == 0 || READY_PREFILL_TASKS > 32768)); then
            echo "PA_AICPU_PLAN_READY_PREFILL_TASKS must be an integer in [1, 32768]." >&2
            exit 1
        fi
        PIPELINE_KEY="streaming-future-p${READY_PREFILL_TASKS}"
        SCHEDULER_INPUT="aicpu_streaming_runtime_plan"
        ;;
    *)
        echo "PA_RUNTIME_PLAN_PIPELINE_POLICY must be 0|plan-ahead-closed or 1|streaming-future." >&2
        exit 1
        ;;
esac
PIPELINE_DEFINES=(
    "-DPA_RUNTIME_PLAN_PIPELINE_POLICY=$PIPELINE_POLICY_ID"
)
if [[ "$PIPELINE_POLICY_ID" -eq 1 ]]; then
    PIPELINE_DEFINES+=(
        "-DPA_AICPU_PLAN_READY_PREFILL_TASKS=$READY_PREFILL_TASKS"
    )
fi
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
        AICPU_TASK_TRACE=1
        PHASE_NAME="none"
        PHASE_ID=0
        BUILD_DIR="$ROOT_DIR/build/ccec/$TENSORMAP_MODE/$PIPELINE_KEY/swimlane"
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
        AICPU_TASK_TRACE=0
        PHASE_NAME="none"
        PHASE_ID=0
        BUILD_DIR="$ROOT_DIR/build/ccec/$TENSORMAP_MODE/$PIPELINE_KEY/perf-clock"
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
PA_ATOMIC_DCCI_COVERAGE_ROOT="$ROOT_DIR" \
    "${PYTHON:-python3}" \
    "$ROOT_DIR/../../common/test_atomic_dcci_source_coverage.py"

# 正式 standalone CCEC 产物先固定使用已验证的 128×128 布局；CAP 仍
# 显式进入四产物编译身份和 manifest，避免默认值漂移后静默混件。
VARIANT_DEFINES+=("-DPTO_FDWIC_TENSORMAP_RING_CAP=$TENSORMAP_RING_CAP")
VARIANT_DEFINES+=(
    "-DPTO_FDWIC_SHARED_INSERT_TURN_GROUPS=$SHARED_INSERT_TURN_GROUPS"
    -DPA_CCEC_BLOCK_LOCAL_STATS=1
    "${PIPELINE_DEFINES[@]}"
)
echo "[BUILD] shared insert-turn groups=$SHARED_INSERT_TURN_GROUPS"
echo "[BUILD] pipeline=$PIPELINE_NAME launch_order=$LAUNCH_ORDER producer_ready=$PRODUCER_READY consumer_admission=$CONSUMER_ADMISSION prefill=$READY_PREFILL_TASKS"

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

CLOCK_CORRELATION_HOST_TEST="$BUILD_DIR/.test_aicpu_clock_correlation"
"$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
    "$AICPU_DIR/test_aicpu_clock_correlation.cpp" \
    -o "$CLOCK_CORRELATION_HOST_TEST"
"$CLOCK_CORRELATION_HOST_TEST"
rm -f -- "$CLOCK_CORRELATION_HOST_TEST"
echo "[CHECK] clock-correlation interval/intersection fail-closed tests passed"
if grep -Eq \
        'HostMonotonicRawNanoseconds|CLOCK_MONOTONIC_RAW|pipeline_(begin|end)_raw_ns|ValidateOwnerClockBracket' \
        "$SCRIPT_DIR/host.cpp" ||
   grep -Eq 'pipeline_(begin|end)_raw_ns' \
        "$AICPU_DIR/aicpu_clock_correlation_abi.h"; then
    echo "CCEC Host/evidence must not treat Host RAW as the AICPU clock domain." >&2
    exit 1
fi
echo "[CHECK] Host absolute clock is excluded from device correlation evidence"
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
    -c -O3 -g -x cce -Wall -std=c++17
    --cce-aicore-only
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false
    -I"$ROOT_DIR/common"
    -I"$PTO_INCLUDE_ROOT/include"
    "${VARIANT_DEFINES[@]}"
)

# direct Plan entry 不恢复旧 callback split，但 LocalStats 仍会跨 noinline
# Build/Execute helper 传引用。CCEC 对这种栈引用没有可靠合同，因此按已验证
# 的 A5 block-local 机制为 AIC/AIV 各保留一份精确 1216B 状态。relocate
# 让最终 mixed ELF 把两份状态排成互不重叠的连续区域。
BLOCK_LOCAL_STATS_BYTES=1216
COMMON_FLAGS+=(
    -mllvm -cce-block-local-relocate=true
    -mllvm "-cce-block-local-reserve-size=$BLOCK_LOCAL_STATS_BYTES"
)

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
        "$SCRIPT_DIR/shared_protocol_compile_probe.cpp"
    "$LD" -m aicorelinux -Ttext=0 -static \
        -o "$SHARED_PROTOCOL_PROBE_AIC_ELF" \
        "$SHARED_PROTOCOL_PROBE_AIC_OBJECT"

    echo "[CHECK] CCEC AIV generic shared protocol instantiation"
    "$CCEC" "${COMMON_FLAGS[@]}" \
        --cce-aicore-arch=dav-c310-vec \
        -DPA_BUILD_AIV \
        -o "$SHARED_PROTOCOL_PROBE_AIV_OBJECT" \
        "$SCRIPT_DIR/shared_protocol_compile_probe.cpp"
    "$LD" -m aicorelinux -Ttext=0 -static \
        -o "$SHARED_PROTOCOL_PROBE_AIV_ELF" \
        "$SHARED_PROTOCOL_PROBE_AIV_OBJECT"
)

# 泳道中的 AICPU producer 必须通过独立的四时间戳握手映射到 AICore
# SYS_CNT。该 AIV ELF 不链接进被测 mixed kernel，Host 也只在 JSON capture
# 前后启动它，因此不会改变 Plan/Build 指令布局或性能窗口。
CLOCK_CORRELATION_OBJECT="$BUILD_DIR/pa_scheduler_clock_correlation_aiv.o"
CLOCK_CORRELATION_ELF="$BUILD_DIR/pa_scheduler_clock_correlation_kernel.o"
CLOCK_CORRELATION_ENTRY="pa_clock_correlation_0_mix_aiv"
echo "[BUILD] standalone AICPU/AICore clock-correlation AIV"
"$CCEC" \
    -c -O3 -g -x cce -Wall -Werror -std=c++17 \
    --cce-aicore-only \
    --cce-aicore-arch=dav-c310-vec \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=false \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -mllvm -cce-aicore-dcci-before-kernel-end=false \
    -I"$PTO_INCLUDE_ROOT/include" \
    -o "$CLOCK_CORRELATION_OBJECT" \
    "$SCRIPT_DIR/clock_correlation_kernel.cpp"
"$LD" -m aicorelinux -Ttext=0 -static \
    -o "$CLOCK_CORRELATION_ELF" "$CLOCK_CORRELATION_OBJECT"
CLOCK_CORRELATION_SYMBOLS="$(
    "$READELF_BIN" --symbols --wide --sym-base=10 \
        "$CLOCK_CORRELATION_ELF"
)"
CLOCK_CORRELATION_SECTIONS="$(
    "$READELF_BIN" --sections --wide "$CLOCK_CORRELATION_ELF"
)"
if ! awk -v name="$CLOCK_CORRELATION_ENTRY" \
    '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" &&
     $NF == name && $3 + 0 > 0 {count++}
     END {exit count != 1}' <<<"$CLOCK_CORRELATION_SYMBOLS" ||
   [[ "$CLOCK_CORRELATION_SECTIONS" != *".ascend.meta.$CLOCK_CORRELATION_ENTRY"* ]]; then
    echo "Clock-correlation ELF is missing its sole non-empty AIV entry/metadata." >&2
    exit 1
fi
if [[ -n "$(
    "$READELF_BIN" --relocs --wide "$CLOCK_CORRELATION_ELF" |
        sed -n '/Relocation section/p'
)" ]]; then
    echo "Clock-correlation ELF must not retain relocations." >&2
    exit 1
fi
while IFS= read -r global_func; do
    if [[ "$global_func" != "$CLOCK_CORRELATION_ENTRY" ]]; then
        echo "Unexpected GLOBAL clock-correlation device function: $global_func" >&2
        exit 1
    fi
done < <(
    awk '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" {print $NF}' \
        <<<"$CLOCK_CORRELATION_SYMBOLS"
)
echo "[CHECK] standalone clock-correlation entry/metadata/closure gates passed"

# 同一入口源码分别面向 cube 与 vector ISA 编译，宏只选择各自的全局入口和 mixed metadata。
echo "[BUILD] CCEC AIC entry (dav-c310-cube)"
"$CCEC" "${COMMON_FLAGS[@]}" \
    --cce-aicore-arch=dav-c310-cube \
    -DPA_BUILD_AIC \
    -o "$BUILD_DIR/pa_scheduler_aic.o" \
    "$SCRIPT_DIR/kernel.cpp"

echo "[BUILD] CCEC AIV entry (dav-c310-vec)"
"$CCEC" "${COMMON_FLAGS[@]}" \
    --cce-aicore-arch=dav-c310-vec \
    -DPA_BUILD_AIV \
    -o "$BUILD_DIR/pa_scheduler_aiv.o" \
    "$SCRIPT_DIR/kernel.cpp"

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
    "$BUILD_DIR/pa_scheduler_aiv.o" \
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
    "$BUILD_DIR/pa_scheduler_aiv.o" \
    pa_scheduler_plan_local_stats_aiv
echo "[CHECK] AIC/AIV direct entries each own one exact block-local LocalStats"

DEVICE_OBJECTS=(
    "$BUILD_DIR/pa_scheduler_aic.o"
    "$BUILD_DIR/pa_scheduler_aiv.o"
)
echo "[CHECK] closed-Plan build uses the two direct mixed-role objects only"

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
    echo "Final mixed ELF must contain two non-overlapping 1216B role-local LocalStats objects." >&2
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
    "-DPA_BUILD_SWIMLANE=$AICPU_TASK_TRACE" \
    "${PIPELINE_DEFINES[@]}" \
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
    -I"$ROOT_DIR/common" \
    -I"$ASCEND_HOME_PATH/include" \
    -I"$ASCEND_HOME_PATH/pkg_inc" \
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
    "$SCRIPT_DIR/host.cpp" \
    -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
    -lascendcl -lruntime \
    -ldl \
    -o "$BUILD_DIR/pa_scheduler_host"

if [[ "$BUILD_VARIANT" == "swimlane" ]]; then
    JOINT_ANALYZE_REJECTION_LOG="$BUILD_DIR/.joint_analyze_rejection.log"
    if "$BUILD_DIR/pa_scheduler_host" \
            --kernel "$BUILD_DIR/pa_scheduler_kernel.o" \
            --analyze-swimlane \
            >"$JOINT_ANALYZE_REJECTION_LOG" 2>&1 ||
       ! grep -Fq \
            "use --swimlane-json and the ordinary offline joint AICPU/AICore analyzer" \
            "$JOINT_ANALYZE_REJECTION_LOG"; then
        echo "CCEC swimlane Host must reject the legacy AICore-only analyzer." >&2
        rm -f -- "$JOINT_ANALYZE_REJECTION_LOG"
        exit 1
    fi
    rm -f -- "$JOINT_ANALYZE_REJECTION_LOG"
    echo "[EXPECTED REJECTION] CCEC Host forbids legacy AICore-only analysis"
fi

# host、kernel、AICPU owner 与 dispatcher 全部成功后才发布统一
# manifest。v10 同时固化 Runtime Plan ABI、容量、producer 入口、独立
# clock-correlation 四时间戳协议与
# Plan/Build pipeline policy；
# run.sh 只消费带完整 manifest 的目录，因此中断重编不会混用新旧镜像。
ARTIFACTS=(
    pa_scheduler_host
    pa_scheduler_kernel.o
    pa_scheduler_clock_correlation_kernel.o
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
    printf '# schema=pa_scheduler_artifacts/v10\n'
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
    printf '# runtime_plan_abi=%u\n' 3
    printf '# runtime_plan_cell_bytes=%u\n' 4608
    printf '# runtime_plan_capacity=%u\n' 4352
    printf '# plan_owner_entry=%s\n' plan_protocol_aicpu_exec
    printf '# scheduler_input=%s\n' "$SCHEDULER_INPUT"
    printf '# pipeline=%s\n' "$PIPELINE_NAME"
    printf '# launch_order=%s\n' "$LAUNCH_ORDER"
    printf '# producer_ready=%s\n' "$PRODUCER_READY"
    printf '# consumer_admission=%s\n' "$CONSUMER_ADMISSION"
    printf '# prefill=%u\n' "$READY_PREFILL_TASKS"
    printf '# clock_correlation_abi=%u\n' 2
    printf '# clock_correlation_samples=%u\n' 8
    printf '# clock_correlation_max_alignment_error_ns=%u\n' 50000
    printf '# aicpu_task_trace_enabled=%u\n' "$AICPU_TASK_TRACE"
    printf '# aicpu_task_trace_record_bytes=%u\n' 64
    printf '# aicpu_operation_trace_enabled=%u\n' "$AICPU_TASK_TRACE"
    printf '# aicpu_operation_trace_record_bytes=%u\n' 64
    printf '# aicpu_operation_trace_fixed_records=%u\n' 64
    printf '# aicpu_operation_trace_records_per_plan_cell=%u\n' 32
    (cd "$BUILD_DIR" && sha256sum "${ARTIFACTS[@]}")
} > "$MANIFEST_TMP"
mv -f -- "$MANIFEST_TMP" "$MANIFEST_PATH"
MANIFEST_TMP=""
trap - EXIT
echo "[CHECK] CCEC artifact manifest published: $MANIFEST_PATH"

echo "[BUILD] complete: $BUILD_DIR"
