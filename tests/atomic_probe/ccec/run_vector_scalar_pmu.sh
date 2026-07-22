#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Build and run the single-AIV PA vector-loop scalar-busy PMU probe without
# touching the aggregate run_all.sh. The probe reuses the pa_scheduler ten-slot
# PMU owner, compiled with PA_BUILD_SUBMIT_PMU=0 so CNT5 remains MTE3 busy.
#
# Usage:
#   ./run_vector_scalar_pmu.sh          # build + run
#   ./run_vector_scalar_pmu.sh build    # build only
#   ./run_vector_scalar_pmu.sh run      # run existing artifacts
#
# The probe intentionally fixes rounds={16,128}, five measured pairs and a
# 180-second process timeout. ATOMIC_PROBE_DEVICE / TASK_DEVICE reuse the
# existing atomic-probe device-selection contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PA_CCEC_DIR="$SCRIPT_DIR/../pa_scheduler/ccec"
BUILD_DIR="$SCRIPT_DIR/build/vector_scalar_pmu"
ACTION="${1:-all}"
if [[ "$ACTION" != "all" && "$ACTION" != "build" && "$ACTION" != "run" ]]; then
    echo "Usage: $0 [all|build|run]" >&2
    exit 1
fi
if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [all|build|run]" >&2
    exit 1
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set; source the local CANN environment first." >&2
    exit 1
fi

CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD="$ASCEND_HOME_PATH/bin/ld.lld"
HCC="$ASCEND_HOME_PATH/tools/hcc/bin/aarch64-target-linux-gnu-g++"
CXX_BIN="${CXX:-g++}"
READELF_BIN="${READELF:-readelf}"
PTO_INCLUDE_ROOT="${PTO_ISA_ROOT:-$ASCEND_HOME_PATH/x86_64-linux}"

if [[ ! -x "$CCEC" || ! -x "$LD" || ! -x "$HCC" ]]; then
    echo "CCEC, ld.lld, or AICPU cross compiler is missing under $ASCEND_HOME_PATH" >&2
    exit 1
fi
for header in pto/common/kernel_meta.hpp pto/common/constants.hpp pto/common/pto_tile.hpp pto/pto-inst.hpp; do
    if [[ ! -f "$PTO_INCLUDE_ROOT/include/$header" ]]; then
        echo "Local PTO header is missing: $PTO_INCLUDE_ROOT/include/$header" >&2
        exit 1
    fi
done
if ! command -v "$CXX_BIN" >/dev/null 2>&1 || ! command -v "$READELF_BIN" >/dev/null 2>&1; then
    echo "Host C++ compiler and readelf are required." >&2
    exit 1
fi

KERNEL="$BUILD_DIR/vector_scalar_pmu_kernel.o"
HOST="$BUILD_DIR/vector_scalar_pmu_host"
OWNER="$BUILD_DIR/libvector_scalar_pmu_owner_aicpu.so"
DISPATCHER="$BUILD_DIR/libvector_scalar_pmu_owner_dispatcher.so"

check_artifacts() {
    for artifact in "$KERNEL" "$HOST" "$OWNER" "$DISPATCHER"; do
        if [[ ! -s "$artifact" ]]; then
            echo "Missing or empty artifact: $artifact" >&2
            exit 1
        fi
    done

    local entry="vector_scalar_pmu_0_mix_aiv"
    local symbols sections owner_header owner_symbols dispatcher_header dispatcher_symbols
    symbols="$("$READELF_BIN" --symbols --wide "$KERNEL")"
    sections="$("$READELF_BIN" --sections --wide "$KERNEL")"
    if [[ "$symbols" != *" $entry"* || "$sections" != *".ascend.meta.$entry"* ]]; then
        echo "Missing non-empty AIV entry or metadata for $entry" >&2
        exit 1
    fi

    owner_header="$("$READELF_BIN" --file-header "$OWNER")"
    owner_symbols="$("$READELF_BIN" --dyn-syms --wide "$OWNER")"
    dispatcher_header="$("$READELF_BIN" --file-header "$DISPATCHER")"
    dispatcher_symbols="$("$READELF_BIN" --dyn-syms --wide "$DISPATCHER")"
    if [[ "$owner_header" != *"Machine:                           AArch64"* ||
          "$dispatcher_header" != *"Machine:                           AArch64"* ||
          "$owner_symbols" != *" simpler_aicpu_exec"* ]]; then
        echo "PMU owner/dispatcher architecture or owner entry check failed." >&2
        exit 1
    fi
    for entry in StaticTileFwkBackendKernelServer DynTileFwkBackendKernelServerInit DynTileFwkBackendKernelServer; do
        if [[ "$dispatcher_symbols" != *" $entry"* ]]; then
            echo "Missing PMU bootstrap dispatcher entry: $entry" >&2
            exit 1
        fi
    done
    echo "[CHECK] AIV entry/meta and self-contained ten-slot PMU owner artifacts PASS"
}

build_probe() {
    mkdir -p "$BUILD_DIR"
    local common_flags=(
        -c -O3 -g -x cce -Wall -std=c++17
        --cce-aicore-only
        --cce-aicore-arch=dav-c310-vec
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=false
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
        -mllvm -cce-aicore-dcci-before-kernel-end=false
        -DCCEC_SYNC_AIV_ONLY
        -I"$SCRIPT_DIR"
        -I"$PTO_INCLUDE_ROOT/include"
    )

    echo "[BUILD] single-AIV PA vector-loop PMU kernel"
    "$CCEC" "${common_flags[@]}" \
        -o "$BUILD_DIR/vector_scalar_pmu_vec.o" \
        "$SCRIPT_DIR/vector_scalar_pmu.cpp"
    "$LD" -m aicorelinux -Ttext=0 -static \
        -o "$KERNEL" "$BUILD_DIR/vector_scalar_pmu_vec.o"

    echo "[BUILD] self-contained AICPU ten-slot PMU dispatcher"
    "$HCC" -shared -fPIC -O3 -g -std=gnu++17 -Wall -Wextra -Werror \
        -Wl,--build-id \
        "$PA_CCEC_DIR/pmu_owner_dispatcher.cpp" \
        -o "$DISPATCHER"

    echo "[BUILD] self-contained AICPU ten-slot PMU owner (CNT5=MTE3)"
    "$HCC" -shared -fPIC -O3 -g -std=gnu++17 -Wall -Wextra -Werror \
        -Wl,--build-id \
        -DPA_BUILD_SUBMIT_PMU=0 \
        -I"$PA_CCEC_DIR" \
        "$PA_CCEC_DIR/pmu_owner_aicpu.cpp" \
        -o "$OWNER"

    echo "[BUILD] host runner"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror -Wno-deprecated-declarations \
        -I"$SCRIPT_DIR" \
        -I"$PA_CCEC_DIR" \
        -I"$REPO_ROOT/src/common" \
        -I"$REPO_ROOT/src/a5/platform/include" \
        -I"$ASCEND_HOME_PATH/include" \
        -I"$ASCEND_HOME_PATH/pkg_inc" \
        -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
        -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
        "$SCRIPT_DIR/vector_scalar_pmu_host.cpp" \
        -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -lascendcl -lruntime -ldl -pthread \
        -o "$HOST"
    check_artifacts
    echo "[BUILD] complete: $BUILD_DIR"
}

run_probe() {
    check_artifacts
    echo "[RUN] device=${ATOMIC_PROBE_DEVICE:-${TASK_DEVICE:-0}} rounds=16,128 repeats=5"
    timeout 180 "$HOST" "$KERNEL"
}

if [[ "$ACTION" == "all" || "$ACTION" == "build" ]]; then
    build_probe
fi
if [[ "$ACTION" == "all" || "$ACTION" == "run" ]]; then
    run_probe
fi
