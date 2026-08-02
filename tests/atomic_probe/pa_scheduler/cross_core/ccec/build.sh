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
        # 性能基线必须保持与正式 swimlane/none 相同的 split-finish
        # 调用形状，不能为减少编译工作偷偷换成 inline finish。
        ;;
    *)
        echo "Usage: $0 [swimlane|perf-clock]" >&2
        exit 1
        ;;
esac
VARIANT_DEFINES+=(-DPA_COMPETE_FIRST_SPLIT_FINISH=1)

# 物理泳道布局与传给三镜像的 compact 编译宏来自同一组 build-side
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

# 正式 standalone CCEC 产物先固定使用已验证的 128×128 布局；CAP 仍
# 显式进入三镜像编译身份和 manifest，避免默认值漂移后静默混件。
VARIANT_DEFINES+=("-DPTO_FDWIC_TENSORMAP_RING_CAP=$TENSORMAP_RING_CAP")
VARIANT_DEFINES+=(
    "-DPTO_FDWIC_SHARED_INSERT_TURN_GROUPS=$SHARED_INSERT_TURN_GROUPS"
)
echo "[BUILD] shared insert-turn groups=$SHARED_INSERT_TURN_GROUPS"

# 编译只依赖本目录源码与用户安装的 CANN/PTO 头，不引用 pa_scheduler 目录外的 simpler 构建产物。
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set; source the CANN 9.1 set_env.sh first." >&2
    exit 1
fi

CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD="$ASCEND_HOME_PATH/bin/ld.lld"
CXX_BIN="${CXX:-g++}"
READELF_BIN="${READELF:-readelf}"
PTO_INCLUDE_ROOT="${PTO_ISA_ROOT:-$ASCEND_HOME_PATH/x86_64-linux}"

# ccec/ld.lld 必须来自当前已 source 的 CANN；host 编译器和 readelf 允许用户通过环境变量替换。
if [[ ! -x "$CCEC" || ! -x "$LD" ]]; then
    echo "CCEC or ld.lld is missing under ASCEND_HOME_PATH=$ASCEND_HOME_PATH" >&2
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
    "$BUILD_DIR/libpa_scheduler_pmu_owner_aicpu.so"

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

# 跨核执行扫描游标使 CompeteFirstSplitRuntimeState 的当前 ABI 为 1728B。
# 只给 split 产物开启 block-local relocation，并按头文件静态断言锁定的
# 精确尺寸预留；runtime object、单 role section 与最终双 role 布局都必须
# 使用同一数值，不能靠多留一条未说明的 cache line 掩盖 ABI 漂移。
SPLIT_STATE_STORAGE_BYTES=1728
# shared nonwinner 在 caller 内直接收尾。S5b 的任意 Scalar Build 允许
# AIC/AIV 都成为 Alloc/QK/SF/PV/UP winner，因此两个 role caller 都必须
# 保留五条跨 TU winner finish relocation。split-finish 是两种构建共同的
# 固定调用形状，不能为 perf-clock 改成另一条 inline 路径。
SPLIT_FINISH_CALL_SITES=5
COMMON_FLAGS+=(
    -mllvm -cce-block-local-relocate=true
    -mllvm "-cce-block-local-reserve-size=$SPLIT_STATE_STORAGE_BYTES"
)

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

text_relocation_count_for_symbol() {
    local object_path="$1"
    local symbol_name="$2"
    "$READELF_BIN" --relocs --wide "$object_path" | awk -v name="$symbol_name" '
        /^Relocation section '\''\.rela\.text'\''/ {in_text = 1; next}
        /^Relocation section / {in_text = 0}
        in_text {
            for (column = 1; column <= NF; ++column) {
                if ($column == name) {
                    count++
                    next
                }
            }
        }
        END {print count + 0}
    '
}

check_split_role_objects() {
    local role="$1"
    local wrong_role="$2"
    local caller="$BUILD_DIR/pa_scheduler_${role}.o"
    local runtime="$BUILD_DIR/pa_scheduler_compete_first_callback_runtime_${role}.o"
    local finish="$BUILD_DIR/pa_scheduler_compete_first_callback_finish_${role}.o"
    local state_symbol="pa_scheduler_compete_first_callback_state_${role}"
    local finish_symbol="pa_scheduler_compete_first_callback_finish_${role}"
    local orchestration_symbol="pa_scheduler_compete_first_callback_orchestration_${role}"
    local entry_symbol="pa_scheduler_0_mix_${role}"
    local dispatcher_symbol="pa_execute_real_winner_workload_${role}"
    local caller_symbols runtime_symbols finish_symbols
    caller_symbols="$("$READELF_BIN" --symbols --wide --sym-base=10 "$caller")"
    runtime_symbols="$("$READELF_BIN" --symbols --wide --sym-base=10 "$runtime")"
    finish_symbols="$("$READELF_BIN" --symbols --wide --sym-base=10 "$finish")"

    if ! awk -v name="$orchestration_symbol" \
        '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
         END {exit count != 1}' <<<"$caller_symbols"; then
        echo "Missing unique strong compete-first orchestration in caller: $caller ($orchestration_symbol)" >&2
        exit 1
    fi
    for imported in "$state_symbol" "$finish_symbol"; do
        if ! awk -v name="$imported" \
            '$5 == "GLOBAL" && $7 == "UND" && $NF == name {count++} END {exit count != 1}' \
            <<<"$caller_symbols"; then
            echo "Caller must import exactly one matching split symbol: $caller ($imported)" >&2
            exit 1
        fi
    done
    if [[ "$(text_relocation_count_for_symbol "$caller" "$finish_symbol")" -ne \
          "$SPLIT_FINISH_CALL_SITES" ]]; then
        echo "Caller must contain exactly $SPLIT_FINISH_CALL_SITES role-compatible finish .rela.text relocations: $caller" >&2
        exit 1
    fi
    if [[ "$(text_relocation_count_for_symbol "$caller" "$state_symbol")" -eq 0 ]]; then
        echo "Caller must access its matching external block-local state: $caller" >&2
        exit 1
    fi
    if "$READELF_BIN" --sections --wide "$caller" | awk \
        'index($0, ".ascend.meta.") != 0 {found = 1} END {exit !found}'; then
        echo "Compete-first caller object must not define launch metadata: $caller" >&2
        exit 1
    fi

    if ! awk -v name="$state_symbol" -v bytes="$SPLIT_STATE_STORAGE_BYTES" \
        '$4 == "OBJECT" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 == bytes {count++}
         END {exit count != 1}' <<<"$runtime_symbols"; then
        echo "Runtime must own one exact-size block-local state: $runtime ($state_symbol)" >&2
        exit 1
    fi
    if ! awk -v name="$entry_symbol" \
        '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
         END {exit count != 1}' <<<"$runtime_symbols"; then
        echo "Runtime must own one non-empty mixed entry: $runtime ($entry_symbol)" >&2
        exit 1
    fi
    if ! awk -v name="$orchestration_symbol" \
        '$5 == "GLOBAL" && $7 == "UND" && $NF == name {count++} END {exit count != 1}' \
        <<<"$runtime_symbols"; then
        echo "Runtime must import one role-specific orchestration: $runtime ($orchestration_symbol)" >&2
        exit 1
    fi
    if [[ "$(text_relocation_count_for_symbol "$runtime" "$orchestration_symbol")" -ne 1 ]]; then
        echo "Runtime entry must contain exactly one orchestration call relocation: $runtime" >&2
        exit 1
    fi
    local block_local_record block_local_section_index block_local_size_hex block_local_alignment
    block_local_record="$(
        "$READELF_BIN" --sections --wide "$runtime" | awk '
            {for (column = 1; column <= NF; ++column) {
                if ($column == ".bl.uninit") {
                    section_index = $(column - 1)
                    gsub(/\[/, "", section_index)
                    gsub(/\]/, "", section_index)
                    print section_index, $(column + 4), $NF
                    exit
                }
            }}
        '
    )"
    read -r block_local_section_index block_local_size_hex block_local_alignment \
        <<<"$block_local_record"
    if [[ -z "$block_local_section_index" || -z "$block_local_size_hex" ||
          $((16#$block_local_size_hex)) -ne "$SPLIT_STATE_STORAGE_BYTES" ||
          "$block_local_alignment" -ne 64 ]]; then
        echo "Runtime .bl.uninit must be exactly ${SPLIT_STATE_STORAGE_BYTES}B and 64B aligned: $runtime" >&2
        exit 1
    fi
    if ! awk -v name="$state_symbol" -v section="$block_local_section_index" \
        '$4 == "OBJECT" && $7 == section && $NF == name {count++}
         END {exit count != 1}' <<<"$runtime_symbols"; then
        echo "Runtime state must be defined in its exact .bl.uninit section: $runtime" >&2
        exit 1
    fi
    local runtime_sections
    runtime_sections="$("$READELF_BIN" --sections --wide "$runtime")"
    if ! awk -v name=".ascend.meta.$entry_symbol" '
        {for (column = 1; column <= NF; ++column) {
            if ($column == name) found = 1
        }}
        END {exit !found}
    ' <<<"$runtime_sections"; then
        echo "Runtime object is missing matching mixed-entry metadata: $runtime" >&2
        exit 1
    fi
    if awk -v name=".ascend.meta.pa_scheduler_0_mix_${wrong_role}" '
        {for (column = 1; column <= NF; ++column) {
            if ($column == name) found = 1
        }}
        END {exit !found}
    ' <<<"$runtime_sections"; then
        echo "Wrong-role mixed-entry metadata leaked into runtime object: $runtime" >&2
        exit 1
    fi

    if ! awk -v name="$finish_symbol" \
        '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
         END {exit count != 1}' <<<"$finish_symbols"; then
        echo "Finish object must define one non-empty strong finish: $finish ($finish_symbol)" >&2
        exit 1
    fi
    # cross_core 的 split finish 只负责 winner Materialize/Register/Build
    # 和 execution payload 发布；执行进度由 orchestration caller 在每次
    # Submit 收口及 FinalDrain 推进。因此 finish 只应导入 block-local
    # runtime state，真实 AIC/AIV dispatcher 必须留在 caller 一侧。
    for imported in "$state_symbol"; do
        if ! awk -v name="$imported" \
            '$5 == "GLOBAL" && $7 == "UND" && $NF == name {count++} END {exit count != 1}' \
            <<<"$finish_symbols"; then
            echo "Finish object must import exactly one matching symbol: $finish ($imported)" >&2
            exit 1
        fi
        if [[ "$(text_relocation_count_for_symbol "$finish" "$imported")" -eq 0 ]]; then
            echo "Finish object must reference its matching imported symbol: $finish ($imported)" >&2
            exit 1
        fi
    done
    if ! awk -v name="$dispatcher_symbol" \
        '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
         END {exit count != 1}' <<<"$caller_symbols"; then
        echo "Cross-core caller must own one non-empty matching dispatcher: $caller ($dispatcher_symbol)" >&2
        exit 1
    fi
    if awk -v name="$dispatcher_symbol" \
        '$NF == name {found = 1} END {exit !found}' <<<"$finish_symbols"; then
        echo "Cross-core finish must not execute a kernel dispatcher: $finish ($dispatcher_symbol)" >&2
        exit 1
    fi
    if "$READELF_BIN" --sections --wide "$finish" | awk \
        'index($0, ".ascend.meta.") != 0 {found = 1} END {exit !found}'; then
        echo "Compete-first finish object must not define launch metadata: $finish" >&2
        exit 1
    fi

    local forbidden symbol_table object_path
    for object_path in "$caller" "$runtime" "$finish"; do
        case "$object_path" in
            "$caller") symbol_table="$caller_symbols" ;;
            "$runtime") symbol_table="$runtime_symbols" ;;
            *) symbol_table="$finish_symbols" ;;
        esac
        for forbidden in \
            "pa_scheduler_compete_first_callback_state_${wrong_role}" \
            "pa_scheduler_compete_first_callback_finish_${wrong_role}" \
            "pa_scheduler_compete_first_callback_orchestration_${wrong_role}" \
            "pa_execute_real_winner_workload_${wrong_role}" \
            "pa_scheduler_0_mix_${wrong_role}"; do
            if awk -v name="$forbidden" \
                '$NF == name {found = 1} END {exit !found}' <<<"$symbol_table"; then
                echo "Wrong-role compete-first symbol leaked into $object_path: $forbidden" >&2
                exit 1
            fi
        done
    done

    if awk -v name="$entry_symbol" '$NF == name {found = 1} END {exit !found}' \
        <<<"$caller_symbols"; then
        echo "Compete-first caller must not own a launch entry: $caller ($entry_symbol)" >&2
        exit 1
    fi
    if awk -v name="$entry_symbol" '$NF == name {found = 1} END {exit !found}' \
        <<<"$finish_symbols"; then
        echo "Compete-first finish must not own a launch entry: $finish ($entry_symbol)" >&2
        exit 1
    fi
    for forbidden in "$finish_symbol" "$dispatcher_symbol"; do
        if awk -v name="$forbidden" '$NF == name {found = 1} END {exit !found}' \
            <<<"$runtime_symbols"; then
            echo "Runtime entry/state owner contains an unexpected helper: $runtime ($forbidden)" >&2
            exit 1
        fi
    done
    if awk -v name="$orchestration_symbol" '$NF == name {found = 1} END {exit !found}' \
        <<<"$finish_symbols"; then
        echo "Compete-first finish must not contain orchestration: $finish ($orchestration_symbol)" >&2
        exit 1
    fi
}

echo "[BUILD] CCEC AIC compete-first runtime/state owner"
"$CCEC" "${COMMON_FLAGS[@]}" --cce-aicore-arch=dav-c310-cube -DPA_BUILD_AIC \
    -o "$BUILD_DIR/pa_scheduler_compete_first_callback_runtime_aic.o" \
    "$SCRIPT_DIR/callback_runtime_entry.cpp"
echo "[BUILD] CCEC AIC compete-first noinline finish"
"$CCEC" "${COMMON_FLAGS[@]}" --cce-aicore-arch=dav-c310-cube -DPA_BUILD_AIC \
    -o "$BUILD_DIR/pa_scheduler_compete_first_callback_finish_aic.o" \
    "$SCRIPT_DIR/callback_finish.cpp"
echo "[BUILD] CCEC AIV compete-first runtime/state owner"
"$CCEC" "${COMMON_FLAGS[@]}" --cce-aicore-arch=dav-c310-vec -DPA_BUILD_AIV \
    -o "$BUILD_DIR/pa_scheduler_compete_first_callback_runtime_aiv.o" \
    "$SCRIPT_DIR/callback_runtime_entry.cpp"
echo "[BUILD] CCEC AIV compete-first noinline finish"
"$CCEC" "${COMMON_FLAGS[@]}" --cce-aicore-arch=dav-c310-vec -DPA_BUILD_AIV \
    -o "$BUILD_DIR/pa_scheduler_compete_first_callback_finish_aiv.o" \
    "$SCRIPT_DIR/callback_finish.cpp"
check_split_role_objects aic aiv
check_split_role_objects aiv aic
DEVICE_OBJECTS=(
    "$BUILD_DIR/pa_scheduler_compete_first_callback_runtime_aic.o"
    "$BUILD_DIR/pa_scheduler_aic.o"
    "$BUILD_DIR/pa_scheduler_compete_first_callback_finish_aic.o"
    "$BUILD_DIR/pa_scheduler_compete_first_callback_runtime_aiv.o"
    "$BUILD_DIR/pa_scheduler_aiv.o"
    "$BUILD_DIR/pa_scheduler_compete_first_callback_finish_aiv.o"
)
echo "[CHECK] compete-first caller/runtime/finish role and state symbols are complete"

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

# finish TU 需要调用按核型区分的真计算 dispatcher；version script 将其与
# 底层 Cube/Vector 实体全部局部化，最终只保留两个 mixed kernel 入口。
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

for role in aic aiv; do
        finish_symbol="pa_scheduler_compete_first_callback_finish_${role}"
        orchestration_symbol="pa_scheduler_compete_first_callback_orchestration_${role}"
        state_symbol="pa_scheduler_compete_first_callback_state_${role}"
        for local_function in "$finish_symbol" "$orchestration_symbol"; do
            if ! awk -v name="$local_function" \
                '$4 == "FUNC" && $5 == "LOCAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
                 END {exit count != 1}' <<<"$SYMBOL_TABLE"; then
                echo "Missing unique LOCAL compete-first function: $local_function" >&2
                exit 1
            fi
        done
        if ! awk -v name="$state_symbol" -v bytes="$SPLIT_STATE_STORAGE_BYTES" \
            '$4 == "OBJECT" && $5 == "LOCAL" && $7 != "UND" && $NF == name && $3 + 0 == bytes {count++}
             END {exit count != 1}' <<<"$SYMBOL_TABLE"; then
            echo "Missing exact-size LOCAL compete-first state: $state_symbol" >&2
            exit 1
        fi
    done
    aic_state_hex="$(awk '$NF == "pa_scheduler_compete_first_callback_state_aic" {print $2; exit}' \
        <<<"$SYMBOL_TABLE")"
    aiv_state_hex="$(awk '$NF == "pa_scheduler_compete_first_callback_state_aiv" {print $2; exit}' \
        <<<"$SYMBOL_TABLE")"
    final_block_local_record="$(
        awk '{for (column = 1; column <= NF; ++column) {
            if ($column == ".bl_uninit") {
                section_index = $(column - 1)
                gsub(/\[/, "", section_index)
                gsub(/\]/, "", section_index)
                print section_index, $(column + 4), $NF
                exit
            }
        }}' <<<"$SECTION_TABLE"
    )"
    read -r final_block_local_section_index final_block_local_size_hex \
        final_block_local_alignment <<<"$final_block_local_record"
    if [[ -z "$aic_state_hex" || -z "$aiv_state_hex" ||
          $((16#$aic_state_hex)) -ne 0 ||
          $((16#$aiv_state_hex)) -ne "$SPLIT_STATE_STORAGE_BYTES" ||
          -z "$final_block_local_section_index" || -z "$final_block_local_size_hex" ||
          $((16#$final_block_local_size_hex)) -ne $((2 * SPLIT_STATE_STORAGE_BYTES)) ||
          "$final_block_local_alignment" -ne 64 ]]; then
        echo "Final block-local layout must be two exact, non-overlapping 64B-aligned compete-first states." >&2
        exit 1
    fi
    for state_symbol in \
        pa_scheduler_compete_first_callback_state_aic \
        pa_scheduler_compete_first_callback_state_aiv; do
        if ! awk -v name="$state_symbol" -v section="$final_block_local_section_index" \
            '$4 == "OBJECT" && $7 == section && $NF == name {count++}
             END {exit count != 1}' <<<"$SYMBOL_TABLE"; then
            echo "Final compete-first state must be bound to the exact .bl_uninit section: $state_symbol" >&2
            exit 1
        fi
    done
    if [[ -n "$("$READELF_BIN" --relocs --wide "$BUILD_DIR/pa_scheduler_kernel.o" |
          sed -n '/Relocation section/p')" ]]; then
        echo "Final compete-first mixed ELF must not retain relocations." >&2
        exit 1
    fi
echo "[CHECK] final ELF keeps helpers LOCAL, binds two exact states, and has no relocations"

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

# host 和 kernel 全部成功后才发布统一 manifest。两件套由 v4 十二行
# 身份头固化 shared mode、variant、物理泳道布局和 SHA256；
# run.sh 只消费带完整 manifest 的目录，因此中断重编不会混用新旧镜像。
ARTIFACTS=(
    pa_scheduler_host
    pa_scheduler_kernel.o
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
    printf '# schema=pa_scheduler_artifacts/v4\n'
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
    (cd "$BUILD_DIR" && sha256sum "${ARTIFACTS[@]}")
} > "$MANIFEST_TMP"
mv -f -- "$MANIFEST_TMP" "$MANIFEST_PATH"
MANIFEST_TMP=""
trap - EXIT
echo "[CHECK] CCEC artifact manifest published: $MANIFEST_PATH"

echo "[BUILD] complete: $BUILD_DIR"
