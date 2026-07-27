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
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/ccec/shared/shared-protocol-litmus"
MANIFEST="$BUILD_DIR/shared_protocol_litmus_artifacts.manifest"
SHARED_HEADER="$SCRIPT_DIR/shared_protocol_litmus_shared.h"
LITMUS_PROCESS_TIMEOUT_SECONDS=60
SHARED_ABI_GENERATION="$(
    sed -n \
        's/^constexpr uint32_t kSharedAbiGeneration = \([0-9][0-9]*\);$/\1/p' \
        "$SHARED_HEADER"
)"
if [[ ! "$SHARED_ABI_GENERATION" =~ ^[0-9]+$ ]]; then
    echo "Cannot read one shared ABI generation from $SHARED_HEADER." >&2
    exit 1
fi

usage() {
    cat <<'EOF'
Usage:
  ./ccec/shared_protocol_litmus.sh build
  ./ccec/shared_protocol_litmus.sh run \
      --scenario history|reader-reclaim|all \
      [--ordering compiler-clobber|payload-dependency|dsb-all|all] \
      [--device N] [--runs N]

Each selected scenario/direction/ordering tuple runs in a fresh host process.
History runs both directions with ordering "na". Reader-reclaim runs both
directions for all three orderings by default. Scenario "all" runs history
first, then all reader-reclaim orderings.

--runs defaults to 20 and repeats every selected tuple that many times.
EOF
}

require_toolchain() {
    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        echo "ASCEND_HOME_PATH is not set; source the user CANN 9.1 set_env.sh first." >&2
        exit 1
    fi
    CCEC="$ASCEND_HOME_PATH/bin/ccec"
    LD="$ASCEND_HOME_PATH/bin/ld.lld"
    CXX_BIN="${CXX:-g++}"
    READELF_BIN="${READELF:-readelf}"
    LLVM_DIS_BIN="${LLVM_DIS:-/opt/mlir-debug/bin/llvm-dis}"
    PTO_INCLUDE_ROOT="${PTO_ISA_ROOT:-$ASCEND_HOME_PATH/x86_64-linux}"
    for tool in "$CCEC" "$LD"; do
        if [[ ! -x "$tool" ]]; then
            echo "Missing CANN tool: $tool" >&2
            exit 1
        fi
    done
    if [[ "$LLVM_DIS_BIN" == */* ]]; then
        if [[ ! -x "$LLVM_DIS_BIN" ]]; then
            echo "Missing LLVM bitcode disassembler: $LLVM_DIS_BIN" >&2
            echo "Set LLVM_DIS to an executable llvm-dis path." >&2
            exit 1
        fi
    elif ! command -v "$LLVM_DIS_BIN" >/dev/null 2>&1; then
        echo "Missing LLVM bitcode disassembler in PATH: $LLVM_DIS_BIN" >&2
        echo "Set LLVM_DIS to an executable llvm-dis path." >&2
        exit 1
    fi
    if ! command -v "$CXX_BIN" >/dev/null 2>&1 ||
       ! command -v "$READELF_BIN" >/dev/null 2>&1 ||
       ! command -v sha256sum >/dev/null 2>&1; then
        echo "shared protocol litmus requires C++, readelf, and sha256sum." >&2
        exit 1
    fi
}

extract_ir_function() {
    local ir_file="$1"
    local function_name="$2"
    awk -v name="$function_name" '
        !inside && /^define / && index($0, name) {
            inside = 1
        }
        inside {
            print
        }
        inside && /^}/ {
            exit
        }
    ' "$ir_file"
}

require_ir_text() {
    local block="$1"
    local needle="$2"
    local label="$3"
    if ! grep -Fq -- "$needle" <<<"$block"; then
        echo "Missing $label in optimized CCEC IR." >&2
        exit 1
    fi
}

reject_ir_text() {
    local block="$1"
    local needle="$2"
    local label="$3"
    if grep -Fq -- "$needle" <<<"$block"; then
        echo "Unexpected $label in optimized CCEC IR." >&2
        exit 1
    fi
}

single_ir_line() {
    local block="$1"
    local needle="$2"
    local label="$3"
    local matches
    matches="$(
        grep -nF -- "$needle" <<<"$block" || true
    )"
    if [[ "$(wc -l <<<"$matches")" -ne 1 ||
          -z "$matches" ]]; then
        echo "Expected exactly one $label in optimized CCEC IR." >&2
        exit 1
    fi
    printf '%s\n' "${matches%%:*}"
}

ssa_definition_in_block() {
    local block="$1"
    local ssa="$2"
    local label="$3"
    local matches
    matches="$(
        grep -F -- "  $ssa = " <<<"$block" || true
    )"
    if [[ "$(wc -l <<<"$matches")" -ne 1 ||
          -z "$matches" ]]; then
        echo "Expected exactly one $label SSA definition: $ssa" >&2
        exit 1
    fi
    printf '%s\n' "$matches"
}

extract_ir_basic_block() {
    local function_block="$1"
    local block_label="$2"
    awk -v target="$block_label" '
        /^[[:alnum:]_.-]+:/ {
            current = $0
            sub(/:.*/, "", current)
            if (inside && current != target) {
                exit
            }
            if (current == target) {
                inside = 1
            }
        }
        inside {
            print
        }
    ' <<<"$function_block"
}

ir_instruction_block_label() {
    local function_block="$1"
    local instruction_needle="$2"
    local label="$3"
    local matches
    matches="$(
        awk -v needle="$instruction_needle" '
            /^[[:alnum:]_.-]+:/ {
                current = $0
                sub(/:.*/, "", current)
            }
            index($0, needle) {
                print current
            }
        ' <<<"$function_block"
    )"
    if [[ "$(wc -l <<<"$matches")" -ne 1 ||
          -z "$matches" ]]; then
        echo "Expected exactly one $label instruction block in optimized CCEC IR." >&2
        exit 1
    fi
    printf '%s\n' "$matches"
}

unique_ir_result() {
    local block="$1"
    local instruction_pattern="$2"
    local label="$3"
    local matches
    matches="$(
        grep -E -- "$instruction_pattern" \
            <<<"$block" || true
    )"
    if [[ "$(wc -l <<<"$matches")" -ne 1 ||
          -z "$matches" ]]; then
        echo "Expected exactly one $label instruction in optimized CCEC IR." >&2
        exit 1
    fi
    local result_pattern='^[[:space:]]*(%[[:alnum:]_.-]+)[[:space:]]*='
    if [[ ! "$matches" =~ $result_pattern ]]; then
        echo "Cannot identify $label SSA result in optimized CCEC IR." >&2
        exit 1
    fi
    printf '%s\n' "${BASH_REMATCH[1]}"
}

validate_reader_ordering_ir() {
    local ir_file="$1"
    local core_label="$2"
    local entry_name
    local entry_block
    local compiler_block
    local dependency_block
    local dsb_block
    local validate_block
    if [[ "$core_label" == AIC ]]; then
        entry_name=pa_scheduler_0_mix_aic
    elif [[ "$core_label" == AIV ]]; then
        entry_name=pa_scheduler_0_mix_aiv
    else
        echo "Unsupported shared protocol IR core label: $core_label" >&2
        exit 1
    fi
    entry_block="$(
        extract_ir_function "$ir_file" "$entry_name"
    )"
    compiler_block="$(
        extract_ir_function \
            "$ir_file" "CloseReaderCompilerClobber"
    )"
    dependency_block="$(
        extract_ir_function \
            "$ir_file" "CloseReaderPayloadDependency"
    )"
    dsb_block="$(
        extract_ir_function "$ir_file" "CloseReaderDsbAll"
    )"
    validate_block="$(
        extract_ir_function \
            "$ir_file" "ValidateReaderSnapshotAfterReuse"
    )"
    if [[ -z "$entry_block" ]]; then
        echo "Missing $core_label O3 IR entry function: $entry_name" >&2
        exit 1
    fi

    local function_block
    local function_name
    for function_name in \
        CloseReaderCompilerClobber \
        CloseReaderPayloadDependency \
        CloseReaderDsbAll \
        ValidateReaderSnapshotAfterReuse; do
        function_block="$(
            extract_ir_function "$ir_file" "$function_name"
        )"
        if [[ -z "$function_block" ]]; then
            echo "Missing $core_label O3 IR function: $function_name" >&2
            exit 1
        fi
    done

    require_ir_text \
        "$compiler_block" \
        'asm sideeffect "", "~{memory}"' \
        "$core_label compiler-only memory clobber"
    require_ir_text \
        "$compiler_block" \
        '@llvm.hivm.atom.CAS.G.s64' \
        "$core_label compiler-only reader CAS"
    require_ir_text \
        "$compiler_block" \
        'i64 1, i64 2, i32 0' \
        "$core_label compiler-only constant CAS transition"
    reject_ir_text \
        "$compiler_block" \
        '@llvm.hivm.DSB' \
        "$core_label compiler-only device barrier"
    local compiler_clobber_line
    local compiler_cas_line
    compiler_clobber_line="$(
        single_ir_line \
            "$compiler_block" \
            'asm sideeffect "", "~{memory}"' \
            "$core_label compiler-only clobber"
    )"
    compiler_cas_line="$(
        single_ir_line \
            "$compiler_block" \
            '@llvm.hivm.atom.CAS.G.s64' \
            "$core_label compiler-only CAS"
    )"
    if ((compiler_clobber_line >= compiler_cas_line)); then
        echo "$core_label compiler clobber no longer precedes reader CAS." >&2
        exit 1
    fi

    require_ir_text \
        "$dependency_block" \
        'asm sideeffect "MOV $0, $0", "=l,0,~{memory}"' \
        "$core_label payload tied MOV"
    require_ir_text \
        "$dependency_block" \
        'sub i32' \
        "$core_label payload dependency delta"
    require_ir_text \
        "$dependency_block" \
        '@llvm.hivm.atom.CAS.G.s64' \
        "$core_label payload-dependent reader CAS"
    reject_ir_text \
        "$dependency_block" \
        'i64 1, i64 2, i32 0' \
        "$core_label payload path collapsed to constant CAS"
    reject_ir_text \
        "$dependency_block" \
        '@llvm.hivm.DSB' \
        "$core_label payload path device barrier"
    if ! grep -Eq \
            '@llvm\.hivm\.atom\.CAS\.G\.s64.*i64 %[[:alnum:]_.-]+, i64 %[[:alnum:]_.-]+, i32 0' \
            <<<"$dependency_block"; then
        echo "$core_label payload CAS operands are not both dynamic in optimized CCEC IR." >&2
        exit 1
    fi
    local dependency_header="${dependency_block%%$'\n'*}"
    local ssa_pattern='(%[[:alnum:]_.-]+)'
    local dependency_signature_pattern
    dependency_signature_pattern="i32 noundef $ssa_pattern, i64 noundef $ssa_pattern, i64 noundef $ssa_pattern, i64 noundef $ssa_pattern, i32 noundef $ssa_pattern, i32 noundef $ssa_pattern"
    if [[ ! "$dependency_header" =~ $dependency_signature_pattern ]]; then
        echo "$core_label payload helper lost its worker + five scalar leaf ABI." >&2
        exit 1
    fi
    local helper_buffer="${BASH_REMATCH[2]}"
    local helper_lo="${BASH_REMATCH[3]}"
    local helper_hi="${BASH_REMATCH[4]}"
    local helper_producer="${BASH_REMATCH[5]}"
    local helper_reserved="${BASH_REMATCH[6]}"
    local checksum_leaves=()
    local helper_value
    for helper_value in \
        "$helper_buffer" "$helper_lo" "$helper_hi"; do
        local low_half
        local shifted_half
        local high_half
        low_half="$(
            unique_ir_result \
                "$dependency_block" \
                " = trunc i64 $helper_value to i32" \
                "$core_label payload low half"
        )"
        shifted_half="$(
            unique_ir_result \
                "$dependency_block" \
                " = lshr i64 $helper_value, 32" \
                "$core_label payload high-half shift"
        )"
        high_half="$(
            unique_ir_result \
                "$dependency_block" \
                " = trunc i64 $shifted_half to i32" \
                "$core_label payload high half"
        )"
        checksum_leaves+=("$low_half" "$high_half")
    done
    checksum_leaves+=("$helper_producer" "$helper_reserved")
    local checksum_steps
    checksum_steps="$(
        grep -Fc '16777619' <<<"$dependency_block" || true
    )"
    if [[ "$checksum_steps" -ne 8 ]]; then
        echo "$core_label payload helper must preserve exactly eight checksum steps." >&2
        exit 1
    fi
    local checksum_value=""
    local checksum_leaf
    local checksum_xor_pattern
    local checksum_xor
    for checksum_leaf in "${checksum_leaves[@]}"; do
        if [[ -z "$checksum_value" ]]; then
            checksum_xor_pattern=" = xor i32 ($checksum_leaf, -2128831035|-2128831035, $checksum_leaf)"
        else
            checksum_xor_pattern=" = xor i32 ($checksum_value, $checksum_leaf|$checksum_leaf, $checksum_value)"
        fi
        checksum_xor="$(
            unique_ir_result \
                "$dependency_block" \
                "$checksum_xor_pattern" \
                "$core_label payload checksum XOR"
        )"
        checksum_value="$(
            unique_ir_result \
                "$dependency_block" \
                " = mul i32 $checksum_xor, 16777619" \
                "$core_label payload checksum multiply"
        )"
    done
    local final_checksum="$checksum_value"
    local dependency_mov_instruction
    dependency_mov_instruction="$(
        grep -F \
            'asm sideeffect "MOV $0, $0", "=l,0,~{memory}"' \
            <<<"$dependency_block"
    )"
    local dependency_mov_pattern
    dependency_mov_pattern="^[[:space:]]*$ssa_pattern[[:space:]]*=.*\\(i32[[:space:]]$final_checksum\\)"
    if [[ ! "$dependency_mov_instruction" =~ $dependency_mov_pattern ]]; then
        echo "$core_label tied MOV no longer consumes the full payload checksum." >&2
        exit 1
    fi
    local dependency_mov="${BASH_REMATCH[1]}"
    local dependency_sub_instruction
    dependency_sub_instruction="$(
        grep -E ' = sub i32 ' <<<"$dependency_block"
    )"
    local dependency_sub_pattern
    dependency_sub_pattern="^[[:space:]]*$ssa_pattern[[:space:]]*=[[:space:]]sub[[:space:]]i32[[:space:]]$dependency_mov,[[:space:]]$final_checksum"
    if [[ ! "$dependency_sub_instruction" =~ $dependency_sub_pattern ]]; then
        echo "$core_label payload dependency delta lost MOV/checksum use-def." >&2
        exit 1
    fi
    local dependency_delta="${BASH_REMATCH[1]}"
    local dependency_zext_instruction
    dependency_zext_instruction="$(
        grep -E \
            " = zext i32 $dependency_delta to i64" \
            <<<"$dependency_block"
    )"
    if [[ ! "$dependency_zext_instruction" =~ ^[[:space:]]*$ssa_pattern[[:space:]]*= ]]; then
        echo "$core_label payload dependency delta no longer reaches i64 CAS operands." >&2
        exit 1
    fi
    local dependency_delta64="${BASH_REMATCH[1]}"
    local expected_instruction
    local desired_instruction
    expected_instruction="$(
        grep -E \
            " = add .*i64 $dependency_delta64, 1" \
            <<<"$dependency_block"
    )"
    desired_instruction="$(
        grep -E \
            " = add .*i64 $dependency_delta64, 2" \
            <<<"$dependency_block"
    )"
    if [[ ! "$expected_instruction" =~ ^[[:space:]]*$ssa_pattern[[:space:]]*= ||
          ! "$desired_instruction" =~ ^[[:space:]]*$ssa_pattern[[:space:]]*= ]]; then
        echo "$core_label payload dependency no longer forms both CAS operands." >&2
        exit 1
    fi
    [[ "$expected_instruction" =~ ^[[:space:]]*$ssa_pattern[[:space:]]*= ]]
    local dependency_expected="${BASH_REMATCH[1]}"
    [[ "$desired_instruction" =~ ^[[:space:]]*$ssa_pattern[[:space:]]*= ]]
    local dependency_desired="${BASH_REMATCH[1]}"
    local dependency_cas_instruction
    dependency_cas_instruction="$(
        grep -F '@llvm.hivm.atom.CAS.G.s64' \
            <<<"$dependency_block"
    )"
    if [[ "$dependency_cas_instruction" != *"i64 $dependency_expected, i64 $dependency_desired, i32 0"* ]]; then
        echo "$core_label payload CAS no longer consumes both delta-derived operands." >&2
        exit 1
    fi
    local dependency_mov_line
    local dependency_sub_line
    local dependency_cas_line
    dependency_mov_line="$(
        single_ir_line \
            "$dependency_block" \
            'asm sideeffect "MOV $0, $0", "=l,0,~{memory}"' \
            "$core_label payload MOV"
    )"
    dependency_sub_line="$(
        single_ir_line \
            "$dependency_block" \
            ' = sub i32 ' \
            "$core_label payload delta"
    )"
    dependency_cas_line="$(
        single_ir_line \
            "$dependency_block" \
            '@llvm.hivm.atom.CAS.G.s64' \
            "$core_label payload CAS"
    )"
    if ((dependency_mov_line >= dependency_sub_line ||
         dependency_sub_line >= dependency_cas_line)); then
        echo "$core_label payload MOV/delta/CAS instruction order changed." >&2
        exit 1
    fi

    require_ir_text \
        "$dsb_block" \
        '@llvm.hivm.DSB(i64 0)' \
        "$core_label DSB_ALL"
    require_ir_text \
        "$dsb_block" \
        '@llvm.hivm.atom.CAS.G.s64' \
        "$core_label DSB reader CAS"
    require_ir_text \
        "$dsb_block" \
        'i64 1, i64 2, i32 0' \
        "$core_label DSB constant CAS transition"
    local dsb_clobber_lines
    mapfile -t dsb_clobber_lines < <(
        grep -nF 'asm sideeffect "", "~{memory}"' \
            <<<"$dsb_block" |
            cut -d: -f1
    )
    if [[ "${#dsb_clobber_lines[@]}" -ne 2 ]]; then
        echo "$core_label DSB path must retain exactly two compiler clobbers." >&2
        exit 1
    fi
    local dsb_line
    local dsb_cas_line
    dsb_line="$(
        single_ir_line \
            "$dsb_block" '@llvm.hivm.DSB(i64 0)' \
            "$core_label DSB_ALL"
    )"
    dsb_cas_line="$(
        single_ir_line \
            "$dsb_block" '@llvm.hivm.atom.CAS.G.s64' \
            "$core_label DSB CAS"
    )"
    if ((dsb_clobber_lines[0] >= dsb_line ||
         dsb_line >= dsb_clobber_lines[1] ||
         dsb_clobber_lines[1] >= dsb_cas_line)); then
        echo "$core_label clobber/DSB/clobber/CAS instruction order changed." >&2
        exit 1
    fi

    require_ir_text \
        "$validate_block" \
        '@llvm.hivm.atom.ADD.G.s64' \
        "$core_label post-reuse gate reload"
    local validation_calls
    validation_calls="$(
        grep -c \
            'call.*ValidateReaderSnapshotAfterReuse' \
            <<<"$entry_block" || true
    )"
    if [[ "$validation_calls" -ne 1 ]]; then
        echo "$core_label O3 IR must retain exactly one post-reuse snapshot validation call." >&2
        exit 1
    fi
    local validate_header="${validate_block%%$'\n'*}"
    local validate_signature_pattern
    validate_signature_pattern="\\(ptr addrspace\\(1\\)[^,]*[[:space:]]$ssa_pattern, i32[^,]*[[:space:]]$ssa_pattern, ptr[^,]*[[:space:]]$ssa_pattern, i64[^,]*[[:space:]]$ssa_pattern, i64[^,]*[[:space:]]$ssa_pattern\\)"
    if [[ ! "$validate_header" =~ $validate_signature_pattern ]]; then
        echo "$core_label snapshot validator lost state/signal/snapshot/seq ABI." >&2
        exit 1
    fi
    local validate_state="${BASH_REMATCH[1]}"
    local validate_signal="${BASH_REMATCH[2]}"
    local validate_snapshot="${BASH_REMATCH[3]}"
    local validate_seq_before="${BASH_REMATCH[4]}"
    local validate_seq_after="${BASH_REMATCH[5]}"
    local validate_signal64
    validate_signal64="$(
        unique_ir_result \
            "$validate_block" \
            " = zext i32 $validate_signal to i64" \
            "$core_label reuse signal extension"
    )"
    local validate_tasks_base
    validate_tasks_base="$(
        unique_ir_result \
            "$validate_block" \
            " = getelementptr inbounds %\"struct\\.pa_scheduler::SchedulerState\", ptr addrspace\\(1\\) $validate_state, i(32|64) 0, i32 6" \
            "$core_label scheduler task-array base"
    )"
    local validate_annotated_signal
    validate_annotated_signal="$(
        unique_ir_result \
            "$validate_block" \
            " = (tail )?call i64 @llvm\\.annotation\\.i64\\.p0\\(i64 $validate_signal64" \
            "$core_label annotated reuse signal"
    )"
    local validate_task_cell
    validate_task_cell="$(
        unique_ir_result \
            "$validate_block" \
            " = getelementptr inbounds \\[[0-9]+ x %\"struct\\.pa_scheduler::TaskCell\"\\], ptr addrspace\\(1\\) $validate_tasks_base, i(32|64) 0, i64 $validate_annotated_signal" \
            "$core_label signal-selected task cell"
    )"
    local validate_gate_pointer
    validate_gate_pointer="$(
        unique_ir_result \
            "$validate_block" \
            " = getelementptr inbounds %\"struct\\.pa_scheduler::TaskCell\", ptr addrspace\\(1\\) $validate_task_cell, i(32|64) 0, i32 2" \
            "$core_label TaskCell::deps_prepared pointer"
    )"
    local validate_token
    validate_token="$(
        unique_ir_result \
            "$validate_block" \
            " = (tail )?call i64 @llvm\\.hivm\\.atom\\.ADD\\.G\\.s64\\(ptr addrspace\\(1\\) $validate_gate_pointer, i64 0, i32 0\\)" \
            "$core_label same-signal gate atomic reload"
    )"
    local validate_compare
    validate_compare="$(
        unique_ir_result \
            "$validate_block" \
            " = icmp eq i64 ($validate_token, $validate_signal64|$validate_signal64, $validate_token)" \
            "$core_label gate-token comparison"
    )"
    local validate_branch
    validate_branch="$(
        grep -E \
            "br i1 $validate_compare, label %[[:alnum:]_.-]+, label %[[:alnum:]_.-]+" \
            <<<"$validate_block" || true
    )"
    if [[ "$(wc -l <<<"$validate_branch")" -ne 1 ||
          -z "$validate_branch" ]]; then
        echo "$core_label snapshot validator lost its unique gate-success branch." >&2
        exit 1
    fi
    local validate_branch_pattern
    validate_branch_pattern="br i1 $validate_compare, label %([[:alnum:]_.-]+), label %([[:alnum:]_.-]+)"
    if [[ ! "$validate_branch" =~ $validate_branch_pattern ]]; then
        echo "$core_label cannot identify the gate-success basic block." >&2
        exit 1
    fi
    local validate_success_label="${BASH_REMATCH[1]}"
    local validate_failure_label="${BASH_REMATCH[2]}"
    if [[ "$validate_success_label" == "$validate_failure_label" ]]; then
        echo "$core_label gate-success and failure blocks unexpectedly alias." >&2
        exit 1
    fi
    local validate_compare_block_label
    validate_compare_block_label="$(
        ir_instruction_block_label \
            "$validate_block" \
            "br i1 $validate_compare" \
            "$core_label gate-token branch"
    )"
    local validate_success_block
    validate_success_block="$(
        extract_ir_basic_block \
            "$validate_block" "$validate_success_label"
    )"
    if [[ -z "$validate_success_block" ]]; then
        echo "$core_label cannot extract the gate-success basic block." >&2
        exit 1
    fi
    local validate_success_header="${validate_success_block%%$'\n'*}"
    local validate_success_predecessor_pattern
    validate_success_predecessor_pattern="^$validate_success_label:[[:space:]]*; preds = %$validate_compare_block_label$"
    if [[ ! "$validate_success_header" =~ $validate_success_predecessor_pattern ]]; then
        echo "$core_label snapshot success block has an entry other than the gate-token true edge." >&2
        exit 1
    fi

    local validate_seq_or
    validate_seq_or="$(
        unique_ir_result \
            "$validate_success_block" \
            " = or i64 ($validate_seq_before, $validate_seq_after|$validate_seq_after, $validate_seq_before)" \
            "$core_label guarded seq pair"
    )"
    local validate_seq_check
    validate_seq_check="$(
        unique_ir_result \
            "$validate_success_block" \
            " = icmp eq i64 ($validate_seq_or, 0|0, $validate_seq_or)" \
            "$core_label guarded seq check"
    )"

    local validate_payload_values=()
    local validate_payload_types=(i64 i64 i64 i32 i32)
    local validate_payload_expected=(30064771072 0 8 0 0)
    local validate_field_pointer
    local validate_field_value
    validate_field_value="$(
        unique_ir_result \
            "$validate_success_block" \
            " = load i64, ptr $validate_snapshot," \
            "$core_label guarded snapshot buffer load"
    )"
    validate_payload_values+=("$validate_field_value")
    local validate_field
    for validate_field in 1 2 3 4; do
        validate_field_pointer="$(
            unique_ir_result \
                "$validate_success_block" \
                " = getelementptr inbounds %\"struct\\.pa_scheduler::SharedRegionValue\", ptr $validate_snapshot, i(32|64) 0, i32 $validate_field" \
                "$core_label guarded snapshot field-$validate_field pointer"
        )"
        validate_field_value="$(
            unique_ir_result \
                "$validate_success_block" \
                " = load ${validate_payload_types[validate_field]}, ptr $validate_field_pointer," \
                "$core_label guarded snapshot field-$validate_field load"
        )"
        validate_payload_values+=("$validate_field_value")
    done
    local validate_checks=("$validate_seq_check")
    local validate_index
    local validate_field_check
    for ((validate_index = 0;
          validate_index < 3;
          ++validate_index)); do
        validate_field_check="$(
            unique_ir_result \
                "$validate_success_block" \
                " = icmp eq ${validate_payload_types[validate_index]} (${validate_payload_values[validate_index]}, ${validate_payload_expected[validate_index]}|${validate_payload_expected[validate_index]}, ${validate_payload_values[validate_index]})" \
                "$core_label guarded snapshot field-$validate_index check"
        )"
        validate_checks+=("$validate_field_check")
    done
    local validate_small_fields_or
    validate_small_fields_or="$(
        unique_ir_result \
            "$validate_success_block" \
            " = or i32 (${validate_payload_values[3]}, ${validate_payload_values[4]}|${validate_payload_values[4]}, ${validate_payload_values[3]})" \
            "$core_label guarded producer/reserved pair"
    )"
    validate_field_check="$(
        unique_ir_result \
            "$validate_success_block" \
            " = icmp eq i32 ($validate_small_fields_or, 0|0, $validate_small_fields_or)" \
            "$core_label guarded producer/reserved check"
    )"
    validate_checks+=("$validate_field_check")
    local validate_plain_load_count
    validate_plain_load_count="$(
        grep -Ec \
            ' = load (i64|i32), ptr ' \
            <<<"$validate_block" || true
    )"
    if [[ "$validate_plain_load_count" -ne 5 ]]; then
        echo "$core_label snapshot validator must contain exactly five ordinary payload loads." >&2
        exit 1
    fi

    # 只允许五个叶子检查经 i1 AND 汇成成功值。这样既排除仅被 debug
    # metadata 引用的“假消费”，也证明 seq/五字段确实共同决定最终返回。
    declare -A validate_masks=()
    local validate_check
    local validate_leaf_index
    for ((validate_leaf_index = 0;
          validate_leaf_index < ${#validate_checks[@]};
          ++validate_leaf_index)); do
        validate_masks["${validate_checks[validate_leaf_index]}"]=$((
            1 << validate_leaf_index
        ))
    done
    local validate_and_lines
    validate_and_lines="$(
        grep -E \
            '^[[:space:]]*%[[:alnum:]_.-]+ = and i1 %[[:alnum:]_.-]+, %[[:alnum:]_.-]+' \
            <<<"$validate_success_block" || true
    )"
    local validate_changed=1
    local validate_and_line
    local validate_and_pattern
    local validate_and_result
    local validate_and_left
    local validate_and_right
    validate_and_pattern='^[[:space:]]*(%[[:alnum:]_.-]+) = and i1 (%[[:alnum:]_.-]+), (%[[:alnum:]_.-]+)'
    while [[ "$validate_changed" -eq 1 ]]; do
        validate_changed=0
        while IFS= read -r validate_and_line; do
            [[ -z "$validate_and_line" ]] && continue
            if [[ ! "$validate_and_line" =~ $validate_and_pattern ]]; then
                continue
            fi
            validate_and_result="${BASH_REMATCH[1]}"
            validate_and_left="${BASH_REMATCH[2]}"
            validate_and_right="${BASH_REMATCH[3]}"
            if [[ -v "validate_masks[$validate_and_left]" &&
                  -v "validate_masks[$validate_and_right]" ]]; then
                local validate_new_mask=$((
                    validate_masks["$validate_and_left"] |
                    validate_masks["$validate_and_right"]
                ))
                if [[ ! -v "validate_masks[$validate_and_result]" ||
                      "${validate_masks[$validate_and_result]}" -ne "$validate_new_mask" ]]; then
                    validate_masks["$validate_and_result"]="$validate_new_mask"
                    validate_changed=1
                fi
            fi
        done <<<"$validate_and_lines"
    done
    local validate_full_mask=$((
        (1 << ${#validate_checks[@]}) - 1
    ))
    local validate_success_value=""
    local validate_mask_value
    for validate_check in "${!validate_masks[@]}"; do
        validate_mask_value="${validate_masks[$validate_check]}"
        if [[ "$validate_mask_value" -eq "$validate_full_mask" ]]; then
            if [[ -n "$validate_success_value" ]]; then
                echo "$core_label snapshot validator has multiple full-check success values." >&2
                exit 1
            fi
            validate_success_value="$validate_check"
        fi
    done
    if [[ -z "$validate_success_value" ]]; then
        echo "$core_label seq and payload checks no longer form one complete success value." >&2
        exit 1
    fi
    local validate_merge_branch
    validate_merge_branch="$(
        grep -E \
            '^[[:space:]]*br label %[[:alnum:]_.-]+' \
            <<<"$validate_success_block" || true
    )"
    if [[ "$(wc -l <<<"$validate_merge_branch")" -ne 1 ||
          -z "$validate_merge_branch" ]]; then
        echo "$core_label snapshot success block lost its unique merge edge." >&2
        exit 1
    fi
    local validate_merge_pattern
    validate_merge_pattern='br label %([[:alnum:]_.-]+)'
    [[ "$validate_merge_branch" =~ $validate_merge_pattern ]]
    local validate_merge_label="${BASH_REMATCH[1]}"
    local validate_merge_block
    validate_merge_block="$(
        extract_ir_basic_block \
            "$validate_block" "$validate_merge_label"
    )"
    if [[ -z "$validate_merge_block" ]]; then
        echo "$core_label cannot extract snapshot validation merge block." >&2
        exit 1
    fi
    local validate_phi_line
    validate_phi_line="$(
        grep -E \
            '^[[:space:]]*%[[:alnum:]_.-]+ = phi i1 ' \
            <<<"$validate_merge_block" || true
    )"
    if [[ "$(wc -l <<<"$validate_phi_line")" -ne 1 ||
          -z "$validate_phi_line" ]]; then
        echo "$core_label snapshot validator lost its unique result phi." >&2
        exit 1
    fi
    local validate_phi_result_pattern
    validate_phi_result_pattern='^[[:space:]]*(%[[:alnum:]_.-]+) = phi i1 '
    [[ "$validate_phi_line" =~ $validate_phi_result_pattern ]]
    local validate_phi_result="${BASH_REMATCH[1]}"
    local validate_phi_inputs
    validate_phi_inputs="$(
        grep -oE \
            '\[ [^]]+ \]' \
            <<<"$validate_phi_line" || true
    )"
    local validate_phi_input
    local validate_phi_input_pattern
    local validate_phi_value
    local validate_phi_predecessor
    local validate_success_input_count=0
    validate_phi_input_pattern='^\[ (false|%[[:alnum:]_.-]+), %([[:alnum:]_.-]+) \]$'
    while IFS= read -r validate_phi_input; do
        [[ -z "$validate_phi_input" ]] && continue
        if [[ ! "$validate_phi_input" =~ $validate_phi_input_pattern ]]; then
            echo "$core_label cannot parse snapshot validator phi input." >&2
            exit 1
        fi
        validate_phi_value="${BASH_REMATCH[1]}"
        validate_phi_predecessor="${BASH_REMATCH[2]}"
        if [[ "$validate_phi_predecessor" == "$validate_success_label" ]]; then
            if [[ "$validate_phi_value" != "$validate_success_value" ]]; then
                echo "$core_label success edge no longer returns the complete snapshot check." >&2
                exit 1
            fi
            validate_success_input_count=$((
                validate_success_input_count + 1
            ))
        elif [[ "$validate_phi_value" != false ]]; then
            echo "$core_label snapshot validator has a non-failure bypass edge." >&2
            exit 1
        fi
    done <<<"$validate_phi_inputs"
    if [[ "$validate_success_input_count" -ne 1 ]]; then
        echo "$core_label snapshot validator must have exactly one success phi edge." >&2
        exit 1
    fi
    local validate_return_count
    validate_return_count="$(
        grep -Ec \
            '^[[:space:]]*ret i1 ' \
            <<<"$validate_block" || true
    )"
    if [[ "$validate_return_count" -ne 1 ]]; then
        echo "$core_label snapshot validator must retain exactly one boolean return." >&2
        exit 1
    fi
    require_ir_text \
        "$validate_merge_block" \
        "ret i1 $validate_phi_result" \
        "$core_label complete snapshot-validation return"

    local dependency_call
    dependency_call="$(
        grep 'call.*CloseReaderPayloadDependency' \
            <<<"$entry_block"
    )"
    local dependency_call_pattern
    dependency_call_pattern="CloseReaderPayloadDependency.*i32 noundef $ssa_pattern, i64 noundef $ssa_pattern, i64 noundef $ssa_pattern, i64 noundef $ssa_pattern, i32 noundef $ssa_pattern, i32 noundef $ssa_pattern"
    if [[ ! "$dependency_call" =~ $dependency_call_pattern ]]; then
        echo "$core_label payload close call lost five scalar arguments." >&2
        exit 1
    fi
    local call_payload_values=(
        "${BASH_REMATCH[2]}"
        "${BASH_REMATCH[3]}"
        "${BASH_REMATCH[4]}"
        "${BASH_REMATCH[5]}"
        "${BASH_REMATCH[6]}"
    )
    local call_payload_types=(i64 i64 i64 i32 i32)
    local payload_index
    local payload_definition
    for ((payload_index = 0;
          payload_index < ${#call_payload_values[@]};
          ++payload_index)); do
        payload_definition="$(
            ssa_definition_in_block \
                "$entry_block" \
                "${call_payload_values[payload_index]}" \
                "$core_label captured payload"
        )"
        if [[ "$payload_definition" != *"load ${call_payload_types[payload_index]}, ptr addrspace(1)"* ]]; then
            echo "$core_label payload close argument is not a direct GM load." >&2
            exit 1
        fi
    done

    local validation_call
    validation_call="$(
        grep 'call.*ValidateReaderSnapshotAfterReuse' \
            <<<"$entry_block"
    )"
    local validation_call_pattern
    validation_call_pattern="ValidateReaderSnapshotAfterReuse.*\\(ptr addrspace\\(1\\)[^,]*[[:space:]]$ssa_pattern, i32[^,]*[[:space:]]$ssa_pattern, ptr[^,]*[[:space:]]$ssa_pattern, i64[^,]*[[:space:]]$ssa_pattern, i64[^,]*[[:space:]]$ssa_pattern\\)"
    if [[ ! "$validation_call" =~ $validation_call_pattern ]]; then
        echo "$core_label snapshot validation call lost state/signal/snapshot/seq arguments." >&2
        exit 1
    fi
    local ordering_function
    local ordering_calls
    for ordering_function in \
        CloseReaderCompilerClobber \
        CloseReaderPayloadDependency \
        CloseReaderDsbAll; do
        ordering_calls="$(
            grep -c "call.*$ordering_function" \
                <<<"$entry_block" || true
        )"
        if [[ "$ordering_calls" -ne 1 ]]; then
            echo "$core_label O3 IR must call $ordering_function exactly once." >&2
            exit 1
        fi
    done
    echo "[CHECK] $core_label O3 IR preserves three reader-close paths and same-gate guarded snapshot validation"
}

build_litmus() {
    require_toolchain
    mkdir -p "$BUILD_DIR"
    rm -f -- \
        "$MANIFEST" \
        "$BUILD_DIR/shared_protocol_litmus_aic.bc" \
        "$BUILD_DIR/shared_protocol_litmus_aic.ll" \
        "$BUILD_DIR/shared_protocol_litmus_aiv.bc" \
        "$BUILD_DIR/shared_protocol_litmus_aiv.ll"

    local common_flags=(
        -c -O3 -g -x cce -Wall -std=c++17
        --cce-aicore-only
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=false
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-dcci-before-kernel-end=false
        -DPTO_FDWIC_SHARED_MAP=1
        -DPTO_FDWIC_TENSORMAP_RING_CAP=128
        -DPA_BUILD_SWIMLANE=0
        -DPA_BUILD_SUBMIT_PMU=0
        -DPA_BUILD_PERF_CLOCK=0
        -DPA_SUBMIT_PMU_PHASE_ID=0
        -I"$ROOT_DIR/common"
        -I"$PTO_INCLUDE_ROOT/include"
    )

    echo "[BUILD] shared protocol litmus AIC entry"
    "$CCEC" "${common_flags[@]}" \
        --cce-aicore-arch=dav-c310-cube \
        -DPA_BUILD_AIC \
        -o "$BUILD_DIR/shared_protocol_litmus_aic.o" \
        "$SCRIPT_DIR/shared_protocol_litmus_kernel.cpp"

    echo "[BUILD] shared protocol litmus AIV entry"
    "$CCEC" "${common_flags[@]}" \
        --cce-aicore-arch=dav-c310-vec \
        -DPA_BUILD_AIV \
        -o "$BUILD_DIR/shared_protocol_litmus_aiv.o" \
        "$SCRIPT_DIR/shared_protocol_litmus_kernel.cpp"

    echo "[BUILD] shared protocol litmus AIC optimized LLVM bitcode"
    "$CCEC" "${common_flags[@]}" \
        --cce-aicore-arch=dav-c310-cube \
        -DPA_BUILD_AIC \
        -Xclang -emit-llvm-bc \
        -o "$BUILD_DIR/shared_protocol_litmus_aic.bc" \
        "$SCRIPT_DIR/shared_protocol_litmus_kernel.cpp"
    "$LLVM_DIS_BIN" \
        "$BUILD_DIR/shared_protocol_litmus_aic.bc" \
        -o "$BUILD_DIR/shared_protocol_litmus_aic.ll"

    echo "[BUILD] shared protocol litmus AIV optimized LLVM bitcode"
    "$CCEC" "${common_flags[@]}" \
        --cce-aicore-arch=dav-c310-vec \
        -DPA_BUILD_AIV \
        -Xclang -emit-llvm-bc \
        -o "$BUILD_DIR/shared_protocol_litmus_aiv.bc" \
        "$SCRIPT_DIR/shared_protocol_litmus_kernel.cpp"
    "$LLVM_DIS_BIN" \
        "$BUILD_DIR/shared_protocol_litmus_aiv.bc" \
        -o "$BUILD_DIR/shared_protocol_litmus_aiv.ll"

    local ir_artifact
    for ir_artifact in \
        "$BUILD_DIR/shared_protocol_litmus_aic.bc" \
        "$BUILD_DIR/shared_protocol_litmus_aic.ll" \
        "$BUILD_DIR/shared_protocol_litmus_aiv.bc" \
        "$BUILD_DIR/shared_protocol_litmus_aiv.ll"; do
        if [[ ! -s "$ir_artifact" ]]; then
            echo "CCEC shared protocol IR artifact is empty: $ir_artifact" >&2
            exit 1
        fi
    done
    echo "[CHECK] AIC/AIV optimized LLVM bitcode and textual IR are non-empty"
    validate_reader_ordering_ir \
        "$BUILD_DIR/shared_protocol_litmus_aic.ll" AIC
    validate_reader_ordering_ir \
        "$BUILD_DIR/shared_protocol_litmus_aiv.ll" AIV

    local object
    for object in \
        "$BUILD_DIR/shared_protocol_litmus_aic.o" \
        "$BUILD_DIR/shared_protocol_litmus_aiv.o"; do
        if "$READELF_BIN" --relocs --wide "$object" |
           grep -q '__multi3'; then
            echo "CCEC shared protocol path generated unsupported __multi3: $object" >&2
            exit 1
        fi
        if "$READELF_BIN" --symbols --wide "$object" |
           awk '$5 == "GLOBAL" && $7 == "UND" {found = 1} END {exit !found}'; then
            echo "CCEC shared protocol object retains an undefined global symbol: $object" >&2
            "$READELF_BIN" --symbols --wide "$object" |
                awk '$5 == "GLOBAL" && $7 == "UND" {print}' >&2
            exit 1
        fi
    done
    echo "[CHECK] AIC/AIV shared protocol objects need no device runtime helper"

    "$LD" -m aicorelinux -Ttext=0 -static \
        --version-script="$SCRIPT_DIR/pa_scheduler_device_exports.map" \
        -o "$BUILD_DIR/shared_protocol_litmus_kernel.o" \
        "$BUILD_DIR/shared_protocol_litmus_aic.o" \
        "$BUILD_DIR/shared_protocol_litmus_aiv.o"

    local symbols
    local sections
    symbols="$(
        "$READELF_BIN" --symbols --wide --sym-base=10 \
            "$BUILD_DIR/shared_protocol_litmus_kernel.o"
    )"
    sections="$(
        "$READELF_BIN" --sections --wide \
            "$BUILD_DIR/shared_protocol_litmus_kernel.o"
    )"
    local entry
    for entry in pa_scheduler_0_mix_aic pa_scheduler_0_mix_aiv; do
        if ! awk -v name="$entry" \
            '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" &&
             $NF == name && $3 + 0 > 0 {found = 1}
             END {exit !found}' <<<"$symbols"; then
            echo "Missing non-empty shared protocol mixed entry: $entry" >&2
            exit 1
        fi
        if [[ "$sections" != *".ascend.meta.$entry"* ]]; then
            echo "Missing shared protocol mixed metadata: .ascend.meta.$entry" >&2
            exit 1
        fi
    done
    if awk \
        '$4 == "FUNC" && $5 == "GLOBAL" && $7 != "UND" &&
         $NF != "pa_scheduler_0_mix_aic" &&
         $NF != "pa_scheduler_0_mix_aiv" {found = 1}
         END {exit !found}' <<<"$symbols"; then
        echo "Shared protocol ELF exports an unexpected GLOBAL function." >&2
        exit 1
    fi
    if "$READELF_BIN" --relocs --wide \
           "$BUILD_DIR/shared_protocol_litmus_kernel.o" |
       grep -q '^Relocation section'; then
        echo "Shared protocol mixed ELF retains relocations." >&2
        exit 1
    fi
    echo "[CHECK] shared protocol mixed ELF has two entries, metadata, and no relocations"

    echo "[BUILD] shared protocol litmus host"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
        -Wno-deprecated-declarations \
        -DPTO_FDWIC_SHARED_MAP=1 \
        -DPTO_FDWIC_TENSORMAP_RING_CAP=128 \
        -DPA_BUILD_SWIMLANE=0 \
        -DPA_BUILD_SUBMIT_PMU=0 \
        -DPA_BUILD_PERF_CLOCK=0 \
        -DPA_SUBMIT_PMU_PHASE_ID=0 \
        -I"$ROOT_DIR/common" \
        -I"$ASCEND_HOME_PATH/include" \
        -I"$ASCEND_HOME_PATH/pkg_inc" \
        -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
        -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
        "$SCRIPT_DIR/shared_protocol_litmus_host.cpp" \
        -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -lascendcl -lruntime \
        -o "$BUILD_DIR/shared_protocol_litmus_host"

    local manifest_tmp
    manifest_tmp="$(mktemp "$BUILD_DIR/.shared_protocol_litmus_manifest.XXXXXX")"
    trap 'rm -f -- "${manifest_tmp:-}"' EXIT
    {
        printf '# schema=pa_scheduler_shared_protocol_litmus/v2\n'
        printf '# tensormap_mode=shared\n'
        printf '# shared_abi_generation=%s\n' "$SHARED_ABI_GENERATION"
        printf '# scenarios=history,reader-reclaim\n'
        printf '# history_directions=aic-to-aiv,aiv-to-aic\n'
        printf '# reader_reclaim_directions=aic-to-aiv,aiv-to-aic\n'
        printf '# reader_reclaim_orderings=compiler-clobber,payload-dependency,dsb-all\n'
        (
            cd "$BUILD_DIR"
            sha256sum \
                shared_protocol_litmus_host \
                shared_protocol_litmus_kernel.o \
                shared_protocol_litmus_aic.ll \
                shared_protocol_litmus_aiv.ll
        )
    } > "$manifest_tmp"
    mv -f -- "$manifest_tmp" "$MANIFEST"
    manifest_tmp=""
    trap - EXIT
    echo "[BUILD] complete: $BUILD_DIR"
}

validate_artifacts() {
    if [[ ! -x "$BUILD_DIR/shared_protocol_litmus_host" ||
          ! -s "$BUILD_DIR/shared_protocol_litmus_kernel.o" ||
          ! -s "$BUILD_DIR/shared_protocol_litmus_aic.ll" ||
          ! -s "$BUILD_DIR/shared_protocol_litmus_aiv.ll" ||
          ! -s "$MANIFEST" ]]; then
        echo "Missing shared protocol litmus artifacts; run '$0 build' first." >&2
        exit 1
    fi
    if ! command -v sha256sum >/dev/null 2>&1; then
        echo "Shared protocol litmus validation requires sha256sum." >&2
        exit 1
    fi

    local expected_headers=(
        "# schema=pa_scheduler_shared_protocol_litmus/v2"
        "# tensormap_mode=shared"
        "# shared_abi_generation=$SHARED_ABI_GENERATION"
        "# scenarios=history,reader-reclaim"
        "# history_directions=aic-to-aiv,aiv-to-aic"
        "# reader_reclaim_directions=aic-to-aiv,aiv-to-aic"
        "# reader_reclaim_orderings=compiler-clobber,payload-dependency,dsb-all"
    )
    local expected_artifacts=(
        shared_protocol_litmus_host
        shared_protocol_litmus_kernel.o
        shared_protocol_litmus_aic.ll
        shared_protocol_litmus_aiv.ll
    )
    local expected_lines=$((
        ${#expected_headers[@]} + ${#expected_artifacts[@]}
    ))
    local manifest_valid=true
    if [[ "$(wc -l < "$MANIFEST")" -ne "$expected_lines" ]]; then
        manifest_valid=false
    fi

    local index
    for ((index = 0; index < ${#expected_headers[@]}; ++index)); do
        if [[ "$(sed -n "$((index + 1))p" "$MANIFEST")" != \
              "${expected_headers[index]}" ]]; then
            manifest_valid=false
        fi
    done

    local digest
    local artifact
    local extra
    local line_number
    for ((index = 0; index < ${#expected_artifacts[@]}; ++index)); do
        line_number=$((${#expected_headers[@]} + index + 1))
        digest=""
        artifact=""
        extra=""
        if ! read -r digest artifact extra < <(
                sed -n "${line_number}p" "$MANIFEST"
            ); then
            manifest_valid=false
        fi
        if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ||
              "$artifact" != "${expected_artifacts[index]}" ||
              -n "$extra" ]]; then
            manifest_valid=false
        fi
    done

    if [[ "$manifest_valid" != true ]]; then
        echo "Shared protocol litmus manifest identity is invalid." >&2
        exit 1
    fi
    (
        cd "$BUILD_DIR"
        tail -n "${#expected_artifacts[@]}" "$MANIFEST" |
            sha256sum --strict -c -
    )
}

run_one_litmus_process() {
    local scenario="$1"
    local direction="$2"
    local ordering="$3"
    local device="$4"
    local run="$5"
    local runs="$6"
    echo "[RUN] scenario=$scenario direction=$direction ordering=$ordering process=$run/$runs"
    # 单进程正常包含约 1 GiB state 的 H2D/D2H，实测通常在 8 秒左右。
    # 60 秒只负责把 ACL stream 异常停滞变成明确失败，不做自动重试，
    # 避免跳过一个可能属于协议本身的偶发等待。
    timeout --signal=INT --kill-after=5s \
        "${LITMUS_PROCESS_TIMEOUT_SECONDS}s" \
        "$BUILD_DIR/shared_protocol_litmus_host" \
        "$BUILD_DIR/shared_protocol_litmus_kernel.o" \
        "$scenario" "$direction" "$ordering" "$device" || {
            local status=$?
            echo "[FAIL] shared protocol process did not complete: " \
                 "scenario=$scenario direction=$direction " \
                 "ordering=$ordering process=$run/$runs status=$status" >&2
            return "$status"
        }
}

run_litmus() {
    if ! command -v timeout >/dev/null 2>&1; then
        echo "shared protocol litmus run requires the coreutils timeout command." >&2
        exit 1
    fi
    local device=0
    local runs=20
    local scenario=""
    local ordering=all
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --scenario)
                if [[ $# -lt 2 ||
                      ( "$2" != "history" &&
                        "$2" != "reader-reclaim" &&
                        "$2" != "all" ) ]]; then
                    echo "--scenario requires history, reader-reclaim, or all." >&2
                    exit 1
                fi
                scenario="$2"
                shift 2
                ;;
            --ordering)
                if [[ $# -lt 2 ||
                      ( "$2" != "compiler-clobber" &&
                        "$2" != "payload-dependency" &&
                        "$2" != "dsb-all" &&
                        "$2" != "all" ) ]]; then
                    echo "--ordering requires compiler-clobber, payload-dependency, dsb-all, or all." >&2
                    exit 1
                fi
                ordering="$2"
                shift 2
                ;;
            --device)
                if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
                    echo "--device requires a non-negative integer." >&2
                    exit 1
                fi
                device="$2"
                shift 2
                ;;
            --runs)
                if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ||
                      "$2" -lt 1 || "$2" -gt 100 ]]; then
                    echo "--runs must be in [1, 100]." >&2
                    exit 1
                fi
                runs="$2"
                shift 2
                ;;
            *)
                echo "Unknown shared protocol litmus option: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done
    if [[ -z "$scenario" ]]; then
        echo "--scenario history, reader-reclaim, or all is required." >&2
        usage >&2
        exit 1
    fi
    if [[ "$scenario" != "reader-reclaim" && "$ordering" != "all" ]]; then
        echo "--ordering may be non-all only with --scenario reader-reclaim." >&2
        exit 1
    fi

    validate_artifacts
    local directions=(aic-to-aiv aiv-to-aic)
    local reader_orderings=(
        compiler-clobber
        payload-dependency
        dsb-all
    )
    if [[ "$scenario" == "reader-reclaim" && "$ordering" != "all" ]]; then
        reader_orderings=("$ordering")
    fi

    local run
    local direction
    local reader_ordering
    local process_count=0
    for ((run = 1; run <= runs; ++run)); do
        if [[ "$scenario" == "history" || "$scenario" == "all" ]]; then
            for direction in "${directions[@]}"; do
                run_one_litmus_process \
                    history "$direction" na "$device" "$run" "$runs"
                ((process_count += 1))
            done
        fi
        if [[ "$scenario" == "reader-reclaim" ||
              "$scenario" == "all" ]]; then
            for reader_ordering in "${reader_orderings[@]}"; do
                for direction in "${directions[@]}"; do
                    run_one_litmus_process \
                        reader-reclaim "$direction" "$reader_ordering" \
                        "$device" "$run" "$runs"
                    ((process_count += 1))
                done
            done
        fi
    done
    echo "[PASS] scenario=$scenario ordering=$ordering runs=$runs fresh_processes=$process_count"
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
fi

action="$1"
shift
case "$action" in
    build)
        if [[ $# -ne 0 ]]; then
            usage >&2
            exit 1
        fi
        build_litmus
        ;;
    run)
        run_litmus "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
