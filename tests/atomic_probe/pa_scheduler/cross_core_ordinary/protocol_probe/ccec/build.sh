#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 任一编译或静态验证失败都立即终止，避免留下看似可用、实际协议已漂移的产物。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROBE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CROSS_CORE_ROOT="$(cd "$PROBE_ROOT/.." && pwd)"
BUILD_DIR="$PROBE_ROOT/build/ccec"

KERNEL_SOURCE="$SCRIPT_DIR/kernel.cpp"
HOST_SOURCE="$SCRIPT_DIR/host.cpp"
EXPORTS_FILE="$SCRIPT_DIR/cross_core_device_exports.map"
AIC_OBJECT="$BUILD_DIR/cross_core_payload_probe_aic.o"
AIV_OBJECT="$BUILD_DIR/cross_core_payload_probe_aiv.o"
AIC_BITCODE="$BUILD_DIR/cross_core_payload_probe_aic.bc"
AIV_BITCODE="$BUILD_DIR/cross_core_payload_probe_aiv.bc"
AIC_IR="$BUILD_DIR/cross_core_payload_probe_aic.ll"
AIV_IR="$BUILD_DIR/cross_core_payload_probe_aiv.ll"
KERNEL_ELF="$BUILD_DIR/cross_core_payload_probe_kernel.o"
HOST_BINARY="$BUILD_DIR/cross_core_payload_probe_host"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set; source the CANN 9.1 set_env.sh first." >&2
    exit 1
fi

CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD_LLD="$ASCEND_HOME_PATH/bin/ld.lld"
READELF_BIN="${READELF:-readelf}"
PTO_INCLUDE_ROOT="${PTO_ISA_ROOT:-$ASCEND_HOME_PATH/x86_64-linux}"

# CCEC 生成的 bitcode 需要由兼容的 llvm-dis 展开，随后才能逐入口验证
# atomic 返回值数据依赖及 CAS/DCCI/DSB 的真实后端顺序。
if [[ -n "${LLVM_DIS:-}" ]]; then
    LLVM_DIS_BIN="$LLVM_DIS"
elif command -v llvm-dis >/dev/null 2>&1; then
    LLVM_DIS_BIN="$(command -v llvm-dis)"
else
    LLVM_DIS_BIN="/opt/mlir-debug/bin/llvm-dis"
fi

# host 固定使用 GCC 15；不接受 PATH 中无版本后缀的 g++ 静默替代。
if command -v g++-15 >/dev/null 2>&1; then
    GXX15="$(command -v g++-15)"
elif [[ -n "${GCC15_ROOT:-}" && -x "$GCC15_ROOT/usr/bin/g++-15" ]]; then
    GXX15="$GCC15_ROOT/usr/bin/g++-15"
else
    echo "g++-15 is required; expose it through PATH or GCC15_ROOT." >&2
    exit 1
fi

for tool in "$CCEC" "$LD_LLD" "$LLVM_DIS_BIN" "$GXX15"; do
    if [[ ! -x "$tool" ]]; then
        echo "Required executable is missing: $tool" >&2
        exit 1
    fi
done
if ! command -v "$READELF_BIN" >/dev/null 2>&1; then
    echo "readelf is required to verify device objects and the mixed ELF." >&2
    exit 1
fi
if [[ ! -f "$PTO_INCLUDE_ROOT/include/pto/common/kernel_meta.hpp" ]]; then
    echo "PTO kernel metadata header is missing under $PTO_INCLUDE_ROOT/include" >&2
    exit 1
fi

GXX15_MAJOR="$($GXX15 -dumpversion | cut -d. -f1)"
if [[ "$GXX15_MAJOR" != "15" ]]; then
    echo "The selected host compiler is not GCC 15: $GXX15" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

# 与 PA CCEC 路径保持一致：关闭编译器自动 scalar DCCI 和 kernel-end
# DCCI，只保留 kernel.cpp 中可审计的显式 cache 可见性协议。
COMMON_DEVICE_FLAGS=(
    -c -O3 -g -x cce -Wall -std=c++17
    --cce-aicore-only
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false
    -I"$CROSS_CORE_ROOT/common"
    -I"$SCRIPT_DIR"
    -I"$PTO_INCLUDE_ROOT/include"
)

compile_role() {
    local role="$1"
    local arch="$2"
    local object_path="$3"
    local bitcode_path="$4"
    local ir_path="$5"

    echo "[BUILD] CCEC $role device object ($arch)"
    "$CCEC" "${COMMON_DEVICE_FLAGS[@]}" \
        --cce-aicore-arch="$arch" \
        -o "$object_path" \
        "$KERNEL_SOURCE"

    echo "[BUILD] CCEC $role optimized LLVM IR"
    "$CCEC" "${COMMON_DEVICE_FLAGS[@]}" \
        --cce-aicore-arch="$arch" \
        -Xclang -emit-llvm-bc \
        -o "$bitcode_path" \
        "$KERNEL_SOURCE"
    "$LLVM_DIS_BIN" "$bitcode_path" -o "$ir_path"
}

extract_ir_function() {
    local ir_path="$1"
    local function_name="$2"
    awk -v needle="$function_name" '
        !inside && /^define / && index($0, needle) != 0 {
            inside = 1
        }
        inside {
            print
        }
        inside && /^}/ {
            exit
        }
    ' "$ir_path"
}

single_match() {
    local matches="$1"
    [[ -n "$matches" ]] &&
        [[ "$(printf '%s\n' "$matches" | wc -l)" -eq 1 ]]
}

verify_return_dependency() {
    local block="$1"
    local role="$2"
    local variant="$3"
    local dcci_line="$4"

    local cas_matches cas_line cas_instruction cas_ssa
    # fatal 首错发布同样使用 CAS(expected=0)。本检查只选择
    # BUILT->CLAIMED 的 returning CAS，它以动态 observed state 为
    # expected；错误分支上的 fatal CAS 不能冒充 acquire 依赖。
    cas_matches="$({
        grep -nF '@llvm.hivm.atom.CAS.G.s64' <<<"$block" |
            grep -vE 'i64 0,[[:space:]]+i64' || true
    })"
    if ! single_match "$cas_matches"; then
        echo "$role $variant must contain exactly one returning claim CAS." >&2
        exit 1
    fi
    cas_line="${cas_matches%%:*}"
    cas_instruction="${cas_matches#*:}"
    if [[ "$cas_instruction" =~ ^[[:space:]]*(%[[:alnum:]_.-]+)[[:space:]]*= ]]; then
        cas_ssa="${BASH_REMATCH[1]}"
    else
        echo "$role $variant CAS result is not bound to an SSA value." >&2
        exit 1
    fi

    # 返回型 CAS 必须直接参与一次 eq/ne 比较，不能只检查调用存在后就把
    # 返回值丢弃；比较结果随后必须直接控制条件分支。
    local compare_matches compare_line compare_instruction compare_ssa
    compare_matches="$({
        grep -nE "^[[:space:]]*%[[:alnum:]_.-]+[[:space:]]*=[[:space:]]*icmp[[:space:]]+(eq|ne)[[:space:]]+i64.*${cas_ssa}([,[:space:]]|$)" \
            <<<"$block" || true
    })"
    if ! single_match "$compare_matches"; then
        echo "$role $variant CAS SSA $cas_ssa must feed exactly one i64 comparison." >&2
        exit 1
    fi
    compare_line="${compare_matches%%:*}"
    compare_instruction="${compare_matches#*:}"
    if [[ "$compare_instruction" =~ ^[[:space:]]*(%[[:alnum:]_.-]+)[[:space:]]*=[[:space:]]*icmp ]]; then
        compare_ssa="${BASH_REMATCH[1]}"
    else
        echo "$role $variant cannot identify the CAS comparison SSA." >&2
        exit 1
    fi

    local branch_matches branch_line
    branch_matches="$(
        grep -nE "^[[:space:]]*br[[:space:]]+i1[[:space:]]+${compare_ssa},[[:space:]]+label" \
            <<<"$block" || true
    )"
    if ! single_match "$branch_matches"; then
        echo "$role $variant comparison SSA $compare_ssa must control exactly one branch." >&2
        exit 1
    fi
    branch_line="${branch_matches%%:*}"
    if ! (( cas_line < compare_line && compare_line < branch_line && branch_line < dcci_line )); then
        echo "$role $variant must preserve CAS -> compare -> branch -> DCCI ordering." >&2
        exit 1
    fi
}

verify_ir_variant() {
    local ir_path="$1"
    local role="$2"
    local variant="$3"
    local function_name="$4"

    local definition_count
    definition_count="$(awk -v needle="$function_name" \
        '/^define / && index($0, needle) != 0 {count++} END {print count + 0}' \
        "$ir_path")"
    if [[ "$definition_count" -ne 1 ]]; then
        echo "$role must define exactly one $function_name; found $definition_count." >&2
        exit 1
    fi

    local block
    block="$(extract_ir_function "$ir_path" "$function_name")"
    if [[ -z "$block" ]]; then
        echo "Cannot extract $role $function_name from $ir_path" >&2
        exit 1
    fi

    local cas_count dcci_count dsb_count cas_line
    local dcci_matches first_dcci_line reference_dcci_line
    cas_count="$({
        grep -F '@llvm.hivm.atom.CAS.G.s64' <<<"$block" |
            grep -vcE 'i64 0,[[:space:]]+i64' || true
    })"
    dcci_count="$(grep -Fc '@llvm.hivm.DCCI.DST' <<<"$block" || true)"
    dsb_count="$(grep -Fc '@llvm.hivm.DSB' <<<"$block" || true)"
    if [[ "$cas_count" -ne 1 || "$dcci_count" -ne 2 ]]; then
        echo "$role $variant requires one claim CAS, one payload DCCI and one conditional descriptor-reference DCCI; got claim-CAS=$cas_count DCCI=$dcci_count." >&2
        exit 1
    fi
    cas_line="$({
        grep -nF '@llvm.hivm.atom.CAS.G.s64' <<<"$block" |
            grep -vE 'i64 0,[[:space:]]+i64' |
            cut -d: -f1
    })"
    dcci_matches="$(grep -nF '@llvm.hivm.DCCI.DST' <<<"$block")"
    first_dcci_line="${dcci_matches%%:*}"
    reference_dcci_line="${dcci_matches##*$'\n'}"
    reference_dcci_line="${reference_dcci_line%%:*}"

    verify_return_dependency \
        "$block" "$role" "$variant" "$first_dcci_line"

    local dsb_matches first_dsb_line payload_dsb_line tail_dsb_line
    dsb_matches="$(grep -nF '@llvm.hivm.DSB' <<<"$block" || true)"
    if [[ "$variant" == "minimal" ]]; then
        if [[ "$dsb_count" -ne 2 ]]; then
            echo "$role minimal requires payload/reference invalidate tail DSBs; found $dsb_count." >&2
            exit 1
        fi
        payload_dsb_line="${dsb_matches%%:*}"
        tail_dsb_line="${dsb_matches%%:*}"
        tail_dsb_line="${dsb_matches##*$'\n'}"
        tail_dsb_line="${tail_dsb_line%%:*}"
        if ! (( cas_line < first_dcci_line &&
                first_dcci_line < payload_dsb_line &&
                payload_dsb_line < reference_dcci_line &&
                reference_dcci_line < tail_dsb_line )); then
            echo "$role minimal must preserve CAS < payload DCCI/DSB < reference DCCI/DSB." >&2
            exit 1
        fi
    else
        if [[ "$dsb_count" -ne 3 ]]; then
            echo "$role pre_dsb requires pre-acquire plus payload/reference tail DSBs; found $dsb_count." >&2
            exit 1
        fi
        first_dsb_line="${dsb_matches%%:*}"
        payload_dsb_line="$(sed -n '2p' <<<"$dsb_matches")"
        payload_dsb_line="${payload_dsb_line%%:*}"
        tail_dsb_line="${dsb_matches##*$'\n'}"
        tail_dsb_line="${tail_dsb_line%%:*}"
        if ! (( cas_line < first_dsb_line &&
                first_dsb_line < first_dcci_line &&
                first_dcci_line < payload_dsb_line &&
                payload_dsb_line < reference_dcci_line &&
                reference_dcci_line < tail_dsb_line )); then
            echo "$role pre_dsb must preserve CAS < pre-DSB < payload DCCI/DSB < reference DCCI/DSB." >&2
            exit 1
        fi
    fi

    echo "[CHECK] $role $variant IR: returning CAS dependency and payload/reference invalidate order are exact"
}

verify_device_object() {
    local object_path="$1"
    local role="$2"
    local symbols relocations undefined_globals
    symbols="$($READELF_BIN --symbols --wide --sym-base=10 "$object_path")"
    relocations="$($READELF_BIN --relocs --wide "$object_path")"
    undefined_globals="$(awk '$5 == "GLOBAL" && $7 == "UND" {print $NF}' <<<"$symbols")"
    if [[ -n "$undefined_globals" ]]; then
        echo "$role device object contains undefined GLOBAL symbols:" >&2
        printf '%s\n' "$undefined_globals" >&2
        exit 1
    fi
    if [[ "$symbols" == *"__multi3"* || "$relocations" == *"__multi3"* ]]; then
        echo "$role device object unexpectedly depends on __multi3." >&2
        exit 1
    fi
    echo "[CHECK] $role object has no undefined GLOBAL/runtime helper or __multi3"
}

compile_role AIC dav-c310-cube "$AIC_OBJECT" "$AIC_BITCODE" "$AIC_IR"
compile_role AIV dav-c310-vec "$AIV_OBJECT" "$AIV_BITCODE" "$AIV_IR"

for role_ir in "AIC:$AIC_IR" "AIV:$AIV_IR"; do
    role="${role_ir%%:*}"
    ir_path="${role_ir#*:}"
    verify_ir_variant "$ir_path" "$role" minimal CrossCoreClaimMinimal
    verify_ir_variant "$ir_path" "$role" pre_dsb CrossCoreClaimPreDsb
done

verify_device_object "$AIC_OBJECT" AIC
verify_device_object "$AIV_OBJECT" AIV

echo "[BUILD] Static 1:1 mixed AICore ELF"
"$LD_LLD" -m aicorelinux -Ttext=0 -static \
    --version-script="$EXPORTS_FILE" \
    -o "$KERNEL_ELF" \
    "$AIC_OBJECT" "$AIV_OBJECT"

SYMBOL_TABLE="$($READELF_BIN --symbols --wide --sym-base=10 "$KERNEL_ELF")"
SECTION_TABLE="$($READELF_BIN --sections --wide "$KERNEL_ELF")"
RELOCATION_TABLE="$($READELF_BIN --relocs --wide "$KERNEL_ELF")"
EXPECTED_ENTRIES=(
    cross_core_payload_probe_0_mix_aic
    cross_core_payload_probe_0_mix_aiv
)

for entry in "${EXPECTED_ENTRIES[@]}"; do
    if ! awk -v name="$entry" \
        '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" && $NF == name && $3 + 0 > 0 {count++}
         END {exit count != 1}' <<<"$SYMBOL_TABLE"; then
        echo "Missing unique non-empty GLOBAL device entry: $entry" >&2
        exit 1
    fi
    if [[ "$SECTION_TABLE" != *".ascend.meta.$entry"* ]]; then
        echo "Missing metadata section for device entry: .ascend.meta.$entry" >&2
        exit 1
    fi
done

metadata_count=0
while IFS= read -r metadata_section; do
    ((metadata_count += 1))
    case "$metadata_section" in
        .ascend.meta.cross_core_payload_probe_0_mix_aic|.ascend.meta.cross_core_payload_probe_0_mix_aiv) ;;
        *)
            echo "Unexpected device-entry metadata section: $metadata_section" >&2
            exit 1
            ;;
    esac
done < <(
    awk '{
        for (column = 1; column <= NF; ++column) {
            if ($column ~ /^\.ascend\.meta\./) print $column
        }
    }' <<<"$SECTION_TABLE"
)
if [[ "$metadata_count" -ne 2 ]]; then
    echo "Final mixed ELF must contain exactly two device-entry metadata sections." >&2
    exit 1
fi

while IFS= read -r global_function; do
    case "$global_function" in
        cross_core_payload_probe_0_mix_aic|cross_core_payload_probe_0_mix_aiv) ;;
        *)
            echo "Unexpected GLOBAL device function: $global_function" >&2
            exit 1
            ;;
    esac
done < <(awk '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" {print $NF}' <<<"$SYMBOL_TABLE")

UNDEFINED_GLOBALS="$(awk '$5 == "GLOBAL" && $7 == "UND" {print $NF}' <<<"$SYMBOL_TABLE")"
if [[ -n "$UNDEFINED_GLOBALS" ]]; then
    echo "Final mixed ELF contains undefined GLOBAL/runtime helper symbols:" >&2
    printf '%s\n' "$UNDEFINED_GLOBALS" >&2
    exit 1
fi
if [[ "$SYMBOL_TABLE" == *"__multi3"* || "$RELOCATION_TABLE" == *"__multi3"* ]]; then
    echo "Final mixed ELF unexpectedly contains __multi3." >&2
    exit 1
fi
echo "[CHECK] final ELF exports only the two expected entries and has no undefined helper"

echo "[BUILD] GCC 15 host runner ($($GXX15 -dumpfullversion))"
"$GXX15" -O2 -std=c++17 -Wall -Wextra -Werror -Wno-deprecated-declarations \
    -I"$CROSS_CORE_ROOT/common" \
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
    echo "Build completed without the required non-empty kernel/host artifacts." >&2
    exit 1
fi

echo "[BUILD] complete: $BUILD_DIR"
echo "[BUILD] kernel:  $KERNEL_ELF"
echo "[BUILD] host:    $HOST_BINARY"
echo "[BUILD] AIC IR:  $AIC_IR"
echo "[BUILD] AIV IR:  $AIV_IR"
