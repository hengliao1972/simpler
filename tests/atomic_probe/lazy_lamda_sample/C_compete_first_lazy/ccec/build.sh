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
LAZY_SAMPLE_MANIFEST_NAME="artifacts.manifest"
LAZY_SAMPLE_TEXT_LAYOUT_NAME="device_text_layout.manifest"
LAZY_SAMPLE_SPLIT=0
LAZY_SAMPLE_VARIANT=1
PMU_VARIANT=0

# All three artifacts compile the same swimlane-capable source with submit PMU
# disabled.  Performance runs pass --no-swimlane at runtime.
if [[ $# -ne 1 || "$1" != "compete-first-lazy" ]]; then
    echo "This frozen package accepts only: $0 compete-first-lazy" >&2
    exit 1
fi
BUILD_VARIANT="$1"
case "$BUILD_VARIANT" in
    original)
        LAZY_SAMPLE_SHAPE="original"
        LAZY_SAMPLE_SHAPE_ID=0
        LAZY_SAMPLE_OBSERVATION="legacy-straight-line-eager"
        LAZY_SAMPLE_FINISH_SHAPE="legacy-inline-submit"
        BUILD_DIR="$ROOT_DIR/build/original"
        VARIANT_DEFINES=(-DPA_BUILD_SWIMLANE=1 -DPA_BUILD_SUBMIT_PMU=0 -DPA_SUBMIT_PMU_PHASE_ID=0)
        ;;
    compete-first|compete-first-lazy)
        LAZY_SAMPLE_SPLIT=1
        LAZY_SAMPLE_OBSERVATION="split-combination-semantic"
        LAZY_SAMPLE_FINISH_SHAPE="noinline-cross-tu"
        if [[ "$BUILD_VARIANT" == "compete-first" ]]; then
            LAZY_SAMPLE_SHAPE="compete-first"
            LAZY_SAMPLE_SHAPE_ID=1
        else
            LAZY_SAMPLE_SHAPE="compete-first-lazy"
            LAZY_SAMPLE_SHAPE_ID=2
        fi
        BUILD_DIR="$ROOT_DIR/build/$LAZY_SAMPLE_SHAPE"
        VARIANT_DEFINES=(
            -DPA_BUILD_SWIMLANE=1
            -DPA_BUILD_SUBMIT_PMU=0
            -DPA_SUBMIT_PMU_PHASE_ID=0
            "-DPA_LAZY_SAMPLE_SHAPE_ID=$LAZY_SAMPLE_SHAPE_ID"
            -DPA_LAZY_SAMPLE_SPLIT_FINISH=1
        )
        ;;
    *)
        echo "Unknown variant: $BUILD_VARIANT (expected original|compete-first|compete-first-lazy)" >&2
        exit 1
        ;;
esac

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
if [[ ! -x "$CCEC" || ! -x "$LD" ]]; then
    echo "CCEC or ld.lld is missing under ASCEND_HOME_PATH=$ASCEND_HOME_PATH" >&2
    exit 1
fi
if [[ "$PMU_VARIANT" -eq 1 && ! -x "$HCC" ]]; then
    echo "The AICPU HCC compiler is missing under ASCEND_HOME_PATH=$ASCEND_HOME_PATH" >&2
    exit 1
fi
if ! command -v "$READELF_BIN" >/dev/null 2>&1; then
    echo "readelf is required to verify the mixed AICore ELF." >&2
    exit 1
fi
if [[ "$BUILD_VARIANT" != "swimlane" ]] && ! command -v sha256sum >/dev/null 2>&1; then
    echo "sha256sum is required to publish variant artifact manifests." >&2
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
if [[ "$PMU_VARIANT" -eq 0 ]]; then
    # 旧统一构建可能在根目录残留 PMU owner；swimlane 构建主动移除这两个
    # 不属于本变体的产物，避免 direct host 调用误加载上一版诊断 SO。
    rm -f \
        "$BUILD_DIR/libpa_scheduler_pmu_owner_dispatcher.so" \
        "$BUILD_DIR/libpa_scheduler_pmu_owner_aicpu.so"
else
    # manifest 是同一 phase 四件套唯一的“可运行”标记。重建一开始先使旧
    # manifest 失效；即使后续编译中断，run.sh 也不会消费目录里的半成品。
    rm -f -- "$BUILD_DIR/$SUBMIT_PMU_MANIFEST_NAME"
fi
if [[ "$LAZY_SAMPLE_VARIANT" -eq 1 ]]; then
    rm -f -- \
        "$BUILD_DIR/$LAZY_SAMPLE_MANIFEST_NAME" \
        "$BUILD_DIR/$LAZY_SAMPLE_TEXT_LAYOUT_NAME"
fi

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

# split finish 的完整 runtime state 为 1600B，超过 CCEC 默认保留的
# block-local 栈空间。编译器 hidden help 明确该参数以 byte 为单位、上限
# 4KiB。实测 1600B 与 2048B 虽生成相同大小的 .text，内容 SHA 却不同，
# 因此使用当前 ABI 的精确尺寸而不增加无依据余量，并严格限于 split 变体。
LAZY_SAMPLE_SPLIT_STATE_BYTES=1600
LAZY_SAMPLE_FINISH_CALL_SITES=5
LAZY_SAMPLE_BLOCK_LOCAL_RESERVE_BYTES=0
if [[ "$LAZY_SAMPLE_SPLIT" -eq 1 ]]; then
    LAZY_SAMPLE_BLOCK_LOCAL_RESERVE_BYTES=$LAZY_SAMPLE_SPLIT_STATE_BYTES
    COMMON_FLAGS+=(
        -mllvm -cce-block-local-relocate=true
        -mllvm "-cce-block-local-reserve-size=$LAZY_SAMPLE_BLOCK_LOCAL_RESERVE_BYTES"
    )
fi

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
    local runtime="$BUILD_DIR/pa_scheduler_lazy_sample_callback_runtime_${role}.o"
    local finish="$BUILD_DIR/pa_scheduler_lazy_sample_callback_finish_${role}.o"
    local state_symbol="pa_scheduler_lazy_sample_callback_state_${role}"
    local finish_symbol="pa_scheduler_lazy_sample_callback_finish_${role}"
    local orchestration_symbol="pa_scheduler_lazy_sample_callback_orchestration_${role}"
    local dispatcher_symbol="pa_execute_real_winner_workload_${role}"
    local entry_symbol="pa_scheduler_0_mix_${role}"
    local caller_symbols runtime_symbols finish_symbols
    caller_symbols="$("$READELF_BIN" --symbols --wide --sym-base=10 "$caller")"
    runtime_symbols="$("$READELF_BIN" --symbols --wide --sym-base=10 "$runtime")"
    finish_symbols="$("$READELF_BIN" --symbols --wide --sym-base=10 "$finish")"

    if ! awk -v name="$orchestration_symbol" \
        '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
         END {exit count != 1}' <<<"$caller_symbols"; then
        echo "Missing unique strong split orchestration in caller object: $orchestration_symbol" >&2
        exit 1
    fi
    for imported in "$state_symbol" "$finish_symbol"; do
        if ! awk -v name="$imported" \
            '$5 == "GLOBAL" && $7 == "UND" && $NF == name {count++}
             END {exit count != 1}' <<<"$caller_symbols"; then
            echo "Caller must import exactly one matching split symbol: $caller ($imported)" >&2
            exit 1
        fi
    done
    if [[ "$(text_relocation_count_for_symbol "$caller" "$finish_symbol")" -ne "$LAZY_SAMPLE_FINISH_CALL_SITES" ]]; then
        echo "Caller must contain exactly $LAZY_SAMPLE_FINISH_CALL_SITES all-task split-finish .rela.text relocations: $caller" >&2
        exit 1
    fi
    if [[ "$(text_relocation_count_for_symbol "$caller" "$state_symbol")" -eq 0 ]]; then
        echo "Caller must access its matching external block-local state: $caller" >&2
        exit 1
    fi
    if "$READELF_BIN" --sections --wide "$caller" | awk \
        'index($0, ".ascend.meta.") != 0 {found = 1} END {exit !found}'; then
        echo "Split caller object must not define launch metadata: $caller" >&2
        exit 1
    fi

    if ! awk -v name="$state_symbol" -v bytes="$LAZY_SAMPLE_SPLIT_STATE_BYTES" \
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
        '$5 == "GLOBAL" && $7 == "UND" && $NF == name {count++}
         END {exit count != 1}' <<<"$runtime_symbols"; then
        echo "Runtime must strongly import one role-specific orchestration: $runtime ($orchestration_symbol)" >&2
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
          $((16#$block_local_size_hex)) -ne "$LAZY_SAMPLE_SPLIT_STATE_BYTES" ||
          "$block_local_alignment" -ne 64 ]]; then
        echo "Runtime block-local section must be exact-size and 64B aligned: $runtime" >&2
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
    for imported in "$state_symbol" "$dispatcher_symbol"; do
        if ! awk -v name="$imported" \
            '$5 == "GLOBAL" && $7 == "UND" && $NF == name {count++}
             END {exit count != 1}' <<<"$finish_symbols"; then
            echo "Finish object must import exactly one matching symbol: $finish ($imported)" >&2
            exit 1
        fi
        if [[ "$(text_relocation_count_for_symbol "$finish" "$imported")" -eq 0 ]]; then
            echo "Finish object must reference its matching imported symbol: $finish ($imported)" >&2
            exit 1
        fi
    done
    if "$READELF_BIN" --sections --wide "$finish" | awk \
        'index($0, ".ascend.meta.") != 0 {found = 1} END {exit !found}'; then
        echo "Split finish object must not define launch metadata: $finish" >&2
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
            "pa_scheduler_lazy_sample_callback_state_${wrong_role}" \
            "pa_scheduler_lazy_sample_callback_finish_${wrong_role}" \
            "pa_scheduler_lazy_sample_callback_orchestration_${wrong_role}" \
            "pa_execute_real_winner_workload_${wrong_role}" \
            "pa_scheduler_0_mix_${wrong_role}"; do
            if awk -v name="$forbidden" \
                '$NF == name {found = 1} END {exit !found}' <<<"$symbol_table"; then
                echo "Wrong-role split symbol leaked into $object_path: $forbidden" >&2
                exit 1
            fi
        done
    done

    for forbidden in "$entry_symbol"; do
        if awk -v name="$forbidden" '$NF == name {found = 1} END {exit !found}' \
            <<<"$caller_symbols"; then
            echo "Split caller must not own a launch entry: $caller ($forbidden)" >&2
            exit 1
        fi
        if awk -v name="$forbidden" '$NF == name {found = 1} END {exit !found}' \
            <<<"$finish_symbols"; then
            echo "Split finish must not own a launch entry: $finish ($forbidden)" >&2
            exit 1
        fi
    done
    for forbidden in "$finish_symbol" "$dispatcher_symbol"; do
        if awk -v name="$forbidden" '$NF == name {found = 1} END {exit !found}' \
            <<<"$runtime_symbols"; then
            echo "Runtime entry/state owner contains an unexpected split helper: $runtime ($forbidden)" >&2
            exit 1
        fi
    done
    if awk -v name="$orchestration_symbol" '$NF == name {found = 1} END {exit !found}' \
        <<<"$finish_symbols"; then
        echo "Split finish must not contain orchestration: $finish ($orchestration_symbol)" >&2
        exit 1
    fi
}

if [[ "$LAZY_SAMPLE_SPLIT" -eq 1 ]]; then
    echo "[BUILD] CCEC AIC split runtime entry/state owner (dav-c310-cube)"
    "$CCEC" "${COMMON_FLAGS[@]}" \
        --cce-aicore-arch=dav-c310-cube \
        -DPA_BUILD_AIC \
        -o "$BUILD_DIR/pa_scheduler_lazy_sample_callback_runtime_aic.o" \
        "$SCRIPT_DIR/callback_runtime_entry.cpp"
    echo "[BUILD] CCEC AIC all-task split finish (dav-c310-cube)"
    "$CCEC" "${COMMON_FLAGS[@]}" \
        --cce-aicore-arch=dav-c310-cube \
        -DPA_BUILD_AIC \
        -o "$BUILD_DIR/pa_scheduler_lazy_sample_callback_finish_aic.o" \
        "$SCRIPT_DIR/callback_finish.cpp"
    echo "[BUILD] CCEC AIV split runtime entry/state owner (dav-c310-vec)"
    "$CCEC" "${COMMON_FLAGS[@]}" \
        --cce-aicore-arch=dav-c310-vec \
        -DPA_BUILD_AIV \
        -o "$BUILD_DIR/pa_scheduler_lazy_sample_callback_runtime_aiv.o" \
        "$SCRIPT_DIR/callback_runtime_entry.cpp"
    echo "[BUILD] CCEC AIV all-task split finish (dav-c310-vec)"
    "$CCEC" "${COMMON_FLAGS[@]}" \
        --cce-aicore-arch=dav-c310-vec \
        -DPA_BUILD_AIV \
        -o "$BUILD_DIR/pa_scheduler_lazy_sample_callback_finish_aiv.o" \
        "$SCRIPT_DIR/callback_finish.cpp"
    check_split_role_objects aic aiv
    check_split_role_objects aiv aic
    echo "[CHECK] split caller/runtime/finish objects satisfy role, state, metadata, and call-boundary gates"
    DEVICE_OBJECTS=(
        "$BUILD_DIR/pa_scheduler_lazy_sample_callback_runtime_aic.o"
        "$BUILD_DIR/pa_scheduler_aic.o"
        "$BUILD_DIR/pa_scheduler_lazy_sample_callback_finish_aic.o"
        "$BUILD_DIR/pa_scheduler_lazy_sample_callback_runtime_aiv.o"
        "$BUILD_DIR/pa_scheduler_aiv.o"
        "$BUILD_DIR/pa_scheduler_lazy_sample_callback_finish_aiv.o"
    )
else
    DEVICE_OBJECTS=(
        "$BUILD_DIR/pa_scheduler_aic.o"
        "$BUILD_DIR/pa_scheduler_aiv.o"
    )
fi

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
    if ! awk -v name=".ascend.meta.$entry" '
        {for (column = 1; column <= NF; ++column) {
            if ($column == name) found = 1
        }}
        END {exit !found}
    ' <<<"$SECTION_TABLE"; then
        echo "Missing mixed-kernel metadata section: .ascend.meta.$entry" >&2
        exit 1
    fi
done
echo "[CHECK] both 1:2 mixed entries and metadata sections are present"

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

# runtime-finish TU 后续需要复用同一个真计算 dispatcher，因此 caller object
# 按核型提供 strong 定义；version script 必须把它们在最终 mixed ELF 中重新
# 局部化。这里同时检查两个 role-specific dispatcher 与底层 Cube/Vector 实体，
# 禁止因抽取 adapter 漏掉任一真实负载路径。
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

if [[ "$LAZY_SAMPLE_VARIANT" -eq 1 ]]; then
    # callback/builder/front must remain inline in both families.  Only split
    # shapes may retain the two fixed, role-specific runtime finish functions.
    if awk \
        '$4 == "FUNC" && $7 != "UND" &&
         (index($NF, "LazySampleCallback") != 0 || index($NF, "SubmitQk") != 0 ||
          index($NF, "LazySampleSplitState") != 0) {found = 1}
         END {exit !found}' <<<"$SYMBOL_TABLE"; then
        echo "lazy sample callback builder/front unexpectedly survived as an out-of-line device function." >&2
        exit 1
    fi
    if [[ "$LAZY_SAMPLE_SPLIT" -eq 1 ]]; then
        for role in aic aiv; do
            finish_symbol="pa_scheduler_lazy_sample_callback_finish_${role}"
            if ! awk -v name="$finish_symbol" \
                '$4 == "FUNC" && $5 == "LOCAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
                 END {exit count != 1}' <<<"$SYMBOL_TABLE"; then
                echo "Missing unique non-empty LOCAL split finish in final ELF: $finish_symbol" >&2
                exit 1
            fi
            orchestration_symbol="pa_scheduler_lazy_sample_callback_orchestration_${role}"
            if ! awk -v name="$orchestration_symbol" \
                '$4 == "FUNC" && $5 == "LOCAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
                 END {exit count != 1}' <<<"$SYMBOL_TABLE"; then
                echo "Missing unique non-empty LOCAL split orchestration in final ELF: $orchestration_symbol" >&2
                exit 1
            fi
            state_symbol="pa_scheduler_lazy_sample_callback_state_${role}"
            if ! awk -v name="$state_symbol" -v bytes="$LAZY_SAMPLE_SPLIT_STATE_BYTES" \
                '$4 == "OBJECT" && $5 == "LOCAL" && $7 != "UND" && $NF == name && $3 + 0 == bytes {count++}
                 END {exit count != 1}' <<<"$SYMBOL_TABLE"; then
                echo "Missing unique exact-size LOCAL split state in final ELF: $state_symbol" >&2
                exit 1
            fi
        done
        aic_state_hex="$(awk '$NF == "pa_scheduler_lazy_sample_callback_state_aic" {print $2; exit}' <<<"$SYMBOL_TABLE")"
        aiv_state_hex="$(awk '$NF == "pa_scheduler_lazy_sample_callback_state_aiv" {print $2; exit}' <<<"$SYMBOL_TABLE")"
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
            final_block_local_alignment \
            <<<"$final_block_local_record"
        if [[ -z "$aic_state_hex" || -z "$aiv_state_hex" ||
              $((16#$aic_state_hex)) -ne 0 ||
              $((16#$aiv_state_hex)) -ne "$LAZY_SAMPLE_SPLIT_STATE_BYTES" ||
              -z "$final_block_local_section_index" || -z "$final_block_local_size_hex" ||
              $((16#$final_block_local_size_hex)) -ne $((2 * LAZY_SAMPLE_SPLIT_STATE_BYTES)) ||
              "$final_block_local_alignment" -ne 64 ]]; then
            echo "Final split block-local layout must be two exact, non-overlapping 64B-aligned states." >&2
            exit 1
        fi
        for state_symbol in \
            pa_scheduler_lazy_sample_callback_state_aic \
            pa_scheduler_lazy_sample_callback_state_aiv; do
            if ! awk -v name="$state_symbol" -v section="$final_block_local_section_index" \
                '$4 == "OBJECT" && $7 == section && $NF == name {count++}
                 END {exit count != 1}' <<<"$SYMBOL_TABLE"; then
                echo "Final split state must be bound to the exact .bl_uninit section: $state_symbol" >&2
                exit 1
            fi
        done
        echo "[CHECK] split finishes/orchestrations/states are LOCAL and final block-local layout is exact"
    fi
    if [[ -n "$("$READELF_BIN" --relocs --wide "$BUILD_DIR/pa_scheduler_kernel.o" | sed -n '/Relocation section/p')" ]]; then
        echo "Final lazy sample callback mixed ELF must not retain relocations." >&2
        exit 1
    fi
    echo "[CHECK] lazy sample callback builder/front is inline and final ELF has no relocations"
fi

check_icache_probe_layout() {
    local role="$1"
    local target="pa_icache_target_${role}"
    local harness="pa_icache_measure_${role}"
    local thrash="pa_icache_thrash_${role}"
    local target_record
    local harness_record
    local thrash_record
    target_record="$(awk -v name="$target" '$4 == "FUNC" && index($NF, name) != 0 {print $2, $3; exit}' <<<"$SYMBOL_TABLE")"
    harness_record="$(awk -v name="$harness" '$4 == "FUNC" && index($NF, name) != 0 {print $2, $3; exit}' <<<"$SYMBOL_TABLE")"
    thrash_record="$(awk -v name="$thrash" '$4 == "FUNC" && index($NF, name) != 0 {print $2, $3; exit}' <<<"$SYMBOL_TABLE")"
    if [[ -z "$target_record" || -z "$harness_record" || -z "$thrash_record" ]]; then
        echo "Missing I-cache probe symbols for $role" >&2
        exit 1
    fi

    local target_hex target_size harness_hex harness_size thrash_hex thrash_size
    read -r target_hex target_size <<<"$target_record"
    read -r harness_hex harness_size <<<"$harness_record"
    read -r thrash_hex thrash_size <<<"$thrash_record"
    local target_address=$((16#$target_hex))
    local harness_address=$((16#$harness_hex))
    local thrash_address=$((16#$thrash_hex))
    if (( target_address % 128 != 0 || target_size == 0 || target_size > 16 )); then
        echo "Invalid single-fetch-block I-cache target for $role: address=0x$target_hex size=$target_size" >&2
        exit 1
    fi
    if (( thrash_size < 65536 )); then
        echo "I-cache thrash body is smaller than 64 KiB for $role: size=$thrash_size" >&2
        exit 1
    fi
    if (( harness_address % 128 != 0 || target_address + 128 > harness_address ||
          harness_address + harness_size > thrash_address )); then
        echo "I-cache layout must be target -> harness -> thrash for $role" >&2
        exit 1
    fi
    echo "[CHECK] $role I-cache target=0x$target_hex/$target_size harness=0x$harness_hex/$harness_size "\
         "thrash=0x$thrash_hex/$thrash_size"
}

emit_text_section_fingerprint() {
    local object_path="$1"
    local artifact_name
    artifact_name="$(basename "$object_path")"
    local text_record text_address_hex text_offset_hex text_size_hex
    text_record="$(
        "$READELF_BIN" --sections --wide "$object_path" | awk '
            {for (column = 1; column <= NF; ++column) {
                if ($column == ".text") {
                    print $(column + 2), $(column + 3), $(column + 4)
                    exit
                }
            }}
        '
    )"
    read -r text_address_hex text_offset_hex text_size_hex <<<"$text_record"
    if [[ -z "$text_address_hex" || -z "$text_offset_hex" || -z "$text_size_hex" ]]; then
        echo "Cannot fingerprint missing .text section: $object_path" >&2
        return 1
    fi
    local text_size=$((16#$text_size_hex))
    local text_sha
    text_sha="$(
        dd if="$object_path" bs=1 skip=$((16#$text_offset_hex)) count="$text_size" status=none |
            sha256sum | awk '{print $1}'
    )"
    printf 'text %s %u %s\n' "$artifact_name" "$text_size" "$text_sha"
}

emit_symbol_body_fingerprint() {
    local object_path="$1"
    local symbol_name="$2"
    local artifact_name
    artifact_name="$(basename "$object_path")"
    local symbol_record symbol_address_hex symbol_size
    symbol_record="$(
        "$READELF_BIN" --symbols --wide --sym-base=10 "$object_path" | awk -v name="$symbol_name" '
            $4 == "FUNC" && $7 != "UND" && $NF == name && $3 + 0 > 0 {
                count++
                address = $2
                size = $3
            }
            END {
                if (count != 1) exit 1
                print address, size
            }
        '
    )" || {
        echo "Cannot fingerprint non-unique or empty function: $object_path ($symbol_name)" >&2
        return 1
    }
    read -r symbol_address_hex symbol_size <<<"$symbol_record"

    local text_record text_address_hex text_offset_hex text_size_hex
    text_record="$(
        "$READELF_BIN" --sections --wide "$object_path" | awk '
            {for (column = 1; column <= NF; ++column) {
                if ($column == ".text") {
                    print $(column + 2), $(column + 3), $(column + 4)
                    exit
                }
            }}
        '
    )"
    read -r text_address_hex text_offset_hex text_size_hex <<<"$text_record"
    if [[ -z "$text_address_hex" || -z "$text_offset_hex" || -z "$text_size_hex" ]]; then
        echo "Cannot locate .text for function fingerprint: $object_path ($symbol_name)" >&2
        return 1
    fi
    local symbol_address=$((16#$symbol_address_hex))
    local text_address=$((16#$text_address_hex))
    local text_size=$((16#$text_size_hex))
    local relative_offset=$((symbol_address - text_address))
    if (( relative_offset < 0 || symbol_size <= 0 || relative_offset + symbol_size > text_size )); then
        echo "Function body lies outside .text: $object_path ($symbol_name)" >&2
        return 1
    fi
    local symbol_sha
    symbol_sha="$(
        dd if="$object_path" bs=1 \
            skip=$((16#$text_offset_hex + relative_offset)) \
            count="$symbol_size" status=none |
            sha256sum | awk '{print $1}'
    )"
    printf 'symbol %s %s %u %s\n' \
        "$artifact_name" "$symbol_name" "$symbol_size" "$symbol_sha"
}

# 两个正式 ELF 都不携带旧 cold/warm 校准冲刷体；submit-pmu 只观察真实
# Submit。保留上面的检查函数供历史布局取证时复核，但正式构建不调用它。

# PMU selector/CTRL 的所有权必须由主 aicpu_scheduler 配置并在退出前恢复。
# standalone 目录内自带 Path-A dispatcher 与 owner：前者负责把 owner SO
# 落到设备预安装目录，后者由 mode=0 JSON 注册并通过统一入口执行命令。
# swimlane 构建不生成 PMU owner/dispatcher；submit-pmu 则把 kernel、host、
# owner 与 dispatcher 全部放在同一个 phase 目录，禁止跨 phase 复用。
if [[ "$PMU_VARIANT" -eq 1 ]]; then
    echo "[BUILD] self-contained AICPU PMU dispatcher"
    "$HCC" -shared -fPIC -O3 -g -std=gnu++17 -Wall -Wextra -Werror \
        -Wl,--build-id \
        "$SCRIPT_DIR/pmu_owner_dispatcher.cpp" \
        -o "$BUILD_DIR/libpa_scheduler_pmu_owner_dispatcher.so"

    echo "[BUILD] self-contained AICPU PMU owner"
    "$HCC" -shared -fPIC -O3 -g -std=gnu++17 -Wall -Wextra -Werror \
        -Wl,--build-id \
        "${VARIANT_DEFINES[@]}" \
        -I"$SCRIPT_DIR" \
        "$SCRIPT_DIR/pmu_owner_aicpu.cpp" \
        -o "$BUILD_DIR/libpa_scheduler_pmu_owner_aicpu.so"

    OWNER_HEADER="$("$READELF_BIN" --file-header "$BUILD_DIR/libpa_scheduler_pmu_owner_aicpu.so")"
    OWNER_SYMBOLS="$("$READELF_BIN" --dyn-syms --wide "$BUILD_DIR/libpa_scheduler_pmu_owner_aicpu.so")"
    DISPATCHER_HEADER="$("$READELF_BIN" --file-header "$BUILD_DIR/libpa_scheduler_pmu_owner_dispatcher.so")"
    DISPATCHER_SYMBOLS="$("$READELF_BIN" --dyn-syms --wide "$BUILD_DIR/libpa_scheduler_pmu_owner_dispatcher.so")"
    if [[ "$OWNER_HEADER" != *"Type:                              DYN"* ||
          "$OWNER_HEADER" != *"Machine:                           AArch64"* ||
          "$DISPATCHER_HEADER" != *"Type:                              DYN"* ||
          "$DISPATCHER_HEADER" != *"Machine:                           AArch64"* ]]; then
        echo "PMU dispatcher and owner must both be AArch64 shared objects." >&2
        exit 1
    fi
    if [[ "$OWNER_SYMBOLS" != *" simpler_aicpu_exec"* ]]; then
        echo "Missing main AICPU PMU owner entry: simpler_aicpu_exec" >&2
        exit 1
    fi
    for entry in StaticTileFwkBackendKernelServer DynTileFwkBackendKernelServerInit DynTileFwkBackendKernelServer; do
        if [[ "$DISPATCHER_SYMBOLS" != *" $entry"* ]]; then
            echo "Missing AICPU PMU dispatcher entry: $entry" >&2
            exit 1
        fi
    done
    echo "[CHECK] Path-A dispatcher and main AICPU PMU owner exports are present"
fi

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

if [[ "$PMU_VARIANT" -eq 1 ]]; then
    # host、kernel、owner、dispatcher 全部成功后才生成 manifest；校验和使用
    # 相对文件名，目录复制后仍可在 run 前原样复核。临时文件与最终文件位于
    # 同一目录，mv 只承担单文件原子发布，不会暴露半写 manifest。
    SUBMIT_PMU_ARTIFACTS=(
        pa_scheduler_host
        pa_scheduler_kernel.o
        libpa_scheduler_pmu_owner_aicpu.so
        libpa_scheduler_pmu_owner_dispatcher.so
    )
    for artifact in "${SUBMIT_PMU_ARTIFACTS[@]}"; do
        if [[ ! -s "$BUILD_DIR/$artifact" ]]; then
            echo "Cannot publish submit-pmu manifest; artifact is missing or empty: $artifact" >&2
            exit 1
        fi
    done
    if [[ ! -x "$BUILD_DIR/pa_scheduler_host" ]]; then
        echo "Cannot publish submit-pmu manifest; host runner is not executable." >&2
        exit 1
    fi

    MANIFEST_PATH="$BUILD_DIR/$SUBMIT_PMU_MANIFEST_NAME"
    MANIFEST_TMP="$(mktemp "$BUILD_DIR/.${SUBMIT_PMU_MANIFEST_NAME}.tmp.XXXXXX")"
    cleanup_manifest_tmp() {
        if [[ -n "${MANIFEST_TMP:-}" ]]; then
            rm -f -- "$MANIFEST_TMP"
        fi
    }
    trap cleanup_manifest_tmp EXIT
    {
        printf '# schema=pa_scheduler_submit_pmu_artifacts/v1\n'
        printf '# variant=submit-pmu\n'
        printf '# phase=%s\n' "$PHASE_NAME"
        printf '# phase_id=%u\n' "$PHASE_ID"
        (cd "$BUILD_DIR" && sha256sum "${SUBMIT_PMU_ARTIFACTS[@]}")
    } > "$MANIFEST_TMP"
    mv -f -- "$MANIFEST_TMP" "$MANIFEST_PATH"
    MANIFEST_TMP=""
    trap - EXIT
    echo "[CHECK] submit-pmu artifact manifest published: $MANIFEST_PATH"
fi
if [[ "$LAZY_SAMPLE_VARIANT" -eq 1 ]]; then
    TEXT_LAYOUT_PATH="$BUILD_DIR/$LAZY_SAMPLE_TEXT_LAYOUT_NAME"
    TEXT_LAYOUT_TMP="$(mktemp "$BUILD_DIR/.${LAZY_SAMPLE_TEXT_LAYOUT_NAME}.tmp.XXXXXX")"
    cleanup_text_layout_tmp() {
        if [[ -n "${TEXT_LAYOUT_TMP:-}" ]]; then
            rm -f -- "$TEXT_LAYOUT_TMP"
        fi
    }
    trap cleanup_text_layout_tmp EXIT
    {
        printf '# schema=pa_scheduler_device_text_layout/v1\n'
        printf '# backend=ccec\n'
        printf '# shape=%s\n' "$LAZY_SAMPLE_SHAPE"
        printf '# block_local_reserve_bytes=%u\n' "$LAZY_SAMPLE_BLOCK_LOCAL_RESERVE_BYTES"
        printf '# compiler=%s\n' '$ASCEND_HOME_PATH/bin/ccec'
        printf '# compiler_sha256=%s\n' "$(sha256sum "$CCEC" | awk '{print $1}')"
        printf '# linker=%s\n' '$ASCEND_HOME_PATH/bin/ld.lld'
        printf '# linker_sha256=%s\n' "$(sha256sum "$LD" | awk '{print $1}')"
        emit_text_section_fingerprint "$BUILD_DIR/pa_scheduler_kernel.o"
        for object_path in "${DEVICE_OBJECTS[@]}"; do
            emit_text_section_fingerprint "$object_path"
        done
        for role in aic aiv; do
            emit_symbol_body_fingerprint \
                "$BUILD_DIR/pa_scheduler_kernel.o" "pa_scheduler_0_mix_${role}"
            if [[ "$LAZY_SAMPLE_SPLIT" -eq 1 ]]; then
                emit_symbol_body_fingerprint \
                    "$BUILD_DIR/pa_scheduler_lazy_sample_callback_runtime_${role}.o" \
                    "pa_scheduler_0_mix_${role}"
                emit_symbol_body_fingerprint \
                    "$BUILD_DIR/pa_scheduler_${role}.o" \
                    "pa_scheduler_lazy_sample_callback_orchestration_${role}"
                emit_symbol_body_fingerprint \
                    "$BUILD_DIR/pa_scheduler_lazy_sample_callback_finish_${role}.o" \
                    "pa_scheduler_lazy_sample_callback_finish_${role}"
                emit_symbol_body_fingerprint \
                    "$BUILD_DIR/pa_scheduler_kernel.o" \
                    "pa_scheduler_lazy_sample_callback_orchestration_${role}"
                emit_symbol_body_fingerprint \
                    "$BUILD_DIR/pa_scheduler_kernel.o" \
                    "pa_scheduler_lazy_sample_callback_finish_${role}"
            else
                emit_symbol_body_fingerprint \
                    "$BUILD_DIR/pa_scheduler_${role}.o" "pa_scheduler_0_mix_${role}"
            fi
        done
    } > "$TEXT_LAYOUT_TMP"
    mv -f -- "$TEXT_LAYOUT_TMP" "$TEXT_LAYOUT_PATH"
    TEXT_LAYOUT_TMP=""
    trap - EXIT
    awk '$1 == "text" {
        printf "[TEXT] %s size=%s sha256=%s\n", $2, $3, $4
    }' "$TEXT_LAYOUT_PATH"
    echo "[CHECK] lazy sample callback device .text layout manifest published: $TEXT_LAYOUT_PATH"

    LAZY_SAMPLE_ARTIFACTS=(
        pa_scheduler_host
        pa_scheduler_kernel.o
    )
    if [[ "$LAZY_SAMPLE_SPLIT" -eq 1 ]]; then
        LAZY_SAMPLE_ARTIFACTS+=(pa_scheduler_lazy_sample_callback_runtime_aic.o)
    fi
    LAZY_SAMPLE_ARTIFACTS+=(pa_scheduler_aic.o)
    if [[ "$LAZY_SAMPLE_SPLIT" -eq 1 ]]; then
        LAZY_SAMPLE_ARTIFACTS+=(pa_scheduler_lazy_sample_callback_finish_aic.o)
    fi
    if [[ "$LAZY_SAMPLE_SPLIT" -eq 1 ]]; then
        LAZY_SAMPLE_ARTIFACTS+=(pa_scheduler_lazy_sample_callback_runtime_aiv.o)
    fi
    LAZY_SAMPLE_ARTIFACTS+=(pa_scheduler_aiv.o)
    if [[ "$LAZY_SAMPLE_SPLIT" -eq 1 ]]; then
        LAZY_SAMPLE_ARTIFACTS+=(pa_scheduler_lazy_sample_callback_finish_aiv.o)
    fi
    LAZY_SAMPLE_ARTIFACTS+=("$LAZY_SAMPLE_TEXT_LAYOUT_NAME")
    for artifact in "${LAZY_SAMPLE_ARTIFACTS[@]}"; do
        if [[ ! -s "$BUILD_DIR/$artifact" ]]; then
            echo "Cannot publish lazy sample callback manifest; artifact is missing or empty: $artifact" >&2
            exit 1
        fi
    done
    if [[ ! -x "$BUILD_DIR/pa_scheduler_host" ]]; then
        echo "Cannot publish lazy sample callback manifest; host runner is not executable." >&2
        exit 1
    fi
    MANIFEST_PATH="$BUILD_DIR/$LAZY_SAMPLE_MANIFEST_NAME"
    MANIFEST_TMP="$(mktemp "$BUILD_DIR/.${LAZY_SAMPLE_MANIFEST_NAME}.tmp.XXXXXX")"
    cleanup_callback_manifest_tmp() {
        if [[ -n "${MANIFEST_TMP:-}" ]]; then
            rm -f -- "$MANIFEST_TMP"
        fi
    }
    trap cleanup_callback_manifest_tmp EXIT
    {
        printf '# schema=pa_scheduler_lazy_sample_callback_artifacts/v1\n'
        printf '# backend=ccec\n'
        printf '# shape=%s\n' "$LAZY_SAMPLE_SHAPE"
        printf '# shape_id=%u\n' "$LAZY_SAMPLE_SHAPE_ID"
        printf '# observation=%s\n' "$LAZY_SAMPLE_OBSERVATION"
        (cd "$BUILD_DIR" && sha256sum "${LAZY_SAMPLE_ARTIFACTS[@]}")
    } > "$MANIFEST_TMP"
    mv -f -- "$MANIFEST_TMP" "$MANIFEST_PATH"
    MANIFEST_TMP=""
    trap - EXIT
    echo "[CHECK] lazy sample callback artifact manifest published: $MANIFEST_PATH"
fi

echo "[BUILD] complete: $BUILD_DIR"
