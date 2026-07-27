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
  ./ccec/shared_protocol_litmus.sh run --scenario history [--device N] [--runs N]

The run action executes one direction per host process:
  AIC writers -> AIV reader
  AIV writers -> AIC reader

--runs defaults to 20 and means 20 fresh processes for each direction.
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
    PTO_INCLUDE_ROOT="${PTO_ISA_ROOT:-$ASCEND_HOME_PATH/x86_64-linux}"
    for tool in "$CCEC" "$LD"; do
        if [[ ! -x "$tool" ]]; then
            echo "Missing CANN tool: $tool" >&2
            exit 1
        fi
    done
    if ! command -v "$CXX_BIN" >/dev/null 2>&1 ||
       ! command -v "$READELF_BIN" >/dev/null 2>&1 ||
       ! command -v sha256sum >/dev/null 2>&1; then
        echo "shared protocol litmus requires C++, readelf, and sha256sum." >&2
        exit 1
    fi
}

build_litmus() {
    require_toolchain
    mkdir -p "$BUILD_DIR"
    rm -f -- "$MANIFEST"

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
        printf '# schema=pa_scheduler_shared_protocol_litmus/v1\n'
        printf '# tensormap_mode=shared\n'
        printf '# shared_abi_generation=%s\n' "$SHARED_ABI_GENERATION"
        printf '# scenarios=history\n'
        printf '# history_directions=aic-to-aiv,aiv-to-aic\n'
        (
            cd "$BUILD_DIR"
            sha256sum \
                shared_protocol_litmus_host \
                shared_protocol_litmus_kernel.o
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
          ! -s "$MANIFEST" ]]; then
        echo "Missing shared protocol litmus artifacts; run '$0 build' first." >&2
        exit 1
    fi
    if [[ "$(wc -l < "$MANIFEST")" -ne 7 ||
          "$(sed -n '1p' "$MANIFEST")" != \
          "# schema=pa_scheduler_shared_protocol_litmus/v1" ||
          "$(sed -n '2p' "$MANIFEST")" != \
          "# tensormap_mode=shared" ||
          "$(sed -n '3p' "$MANIFEST")" != \
          "# shared_abi_generation=$SHARED_ABI_GENERATION" ||
          "$(sed -n '4p' "$MANIFEST")" != \
          "# scenarios=history" ||
          "$(sed -n '5p' "$MANIFEST")" != \
          "# history_directions=aic-to-aiv,aiv-to-aic" ||
          "$(awk 'NR == 6 {print $2}' "$MANIFEST")" != \
          "shared_protocol_litmus_host" ||
          "$(awk 'NR == 7 {print $2}' "$MANIFEST")" != \
          "shared_protocol_litmus_kernel.o" ]]; then
        echo "Shared protocol litmus manifest identity is invalid." >&2
        exit 1
    fi
    (
        cd "$BUILD_DIR"
        tail -n 2 "$MANIFEST" | sha256sum -c -
    )
}

run_litmus() {
    local device=0
    local runs=20
    local scenario=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --scenario)
                if [[ $# -lt 2 || "$2" != "history" ]]; then
                    echo "--scenario currently requires history." >&2
                    exit 1
                fi
                scenario="$2"
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
        echo "--scenario history is required." >&2
        usage >&2
        exit 1
    fi
    validate_artifacts
    local run
    local direction
    for ((run = 1; run <= runs; ++run)); do
        for direction in aic-to-aiv aiv-to-aic; do
            echo "[RUN] scenario=$scenario direction=$direction process=$run/$runs"
            "$BUILD_DIR/shared_protocol_litmus_host" \
                "$BUILD_DIR/shared_protocol_litmus_kernel.o" \
                "$scenario" "$direction" "$device"
        done
    done
    echo "[PASS] scenario=$scenario directions=2 fresh_processes_per_direction=$runs"
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
