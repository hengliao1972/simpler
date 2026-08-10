#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# SPDX-License-Identifier: CANN-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
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
    echo "ASCEND_HOME_PATH is not set; source the CANN environment first." >&2
    exit 1
fi

CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD="$ASCEND_HOME_PATH/bin/ld.lld"
HCC="$ASCEND_HOME_PATH/tools/hcc/bin/aarch64-target-linux-gnu-g++"
CXX_BIN="${CXX:-g++}"
READELF_BIN="${READELF:-readelf}"
PTO_INCLUDE_ROOT="${PTO_ISA_ROOT:-$ASCEND_HOME_PATH/x86_64-linux}"
for tool in "$CCEC" "$LD" "$HCC"; do
    if [[ ! -x "$tool" ]]; then
        echo "Missing required tool: $tool" >&2
        exit 1
    fi
done
if [[ ! -f "$PTO_INCLUDE_ROOT/include/pto/common/kernel_meta.hpp" ]]; then
    echo "Missing PTO metadata header under $PTO_INCLUDE_ROOT/include" >&2
    exit 1
fi

KERNEL="$BUILD_DIR/plan_protocol_probe_kernel.o"
DISPATCHER="$BUILD_DIR/libplan_protocol_dispatcher.so"
OWNER="$BUILD_DIR/libplan_protocol_aicpu.so"
HOST="$BUILD_DIR/plan_protocol_probe_host"

build_probe() {
    mkdir -p "$BUILD_DIR"
    echo "[BUILD] AIV return-ready/DCCI consumer"
    "$CCEC" \
        -c -O3 -g -x cce -Wall -std=c++17 \
        --cce-aicore-only \
        --cce-aicore-arch=dav-c310-vec \
        -mllvm -cce-aicore-stack-size=0x8000 \
        -mllvm -cce-aicore-function-stack-size=0x8000 \
        -mllvm -cce-aicore-record-overflow=false \
        -mllvm -cce-aicore-addr-transform \
        -mllvm -cce-aicore-dcci-insert-for-scalar=false \
        -mllvm -cce-aicore-dcci-before-kernel-end=false \
        -I"$SCRIPT_DIR" \
        -I"$PTO_INCLUDE_ROOT/include" \
        -o "$BUILD_DIR/aiv_consumer.o" \
        "$SCRIPT_DIR/aiv_consumer.cpp"
    "$LD" -m aicorelinux -Ttext=0 -static \
        -o "$KERNEL" "$BUILD_DIR/aiv_consumer.o"

    local symbols sections entry="plan_protocol_probe_0_mix_aiv"
    symbols="$("$READELF_BIN" --symbols --wide "$KERNEL")"
    sections="$("$READELF_BIN" --sections --wide "$KERNEL")"
    if [[ "$symbols" != *" $entry"* || "$sections" != *".ascend.meta.$entry"* ]]; then
        echo "Missing AIV entry or metadata for $entry" >&2
        exit 1
    fi

    echo "[BUILD] main-scheduler AICPU producer"
    "$HCC" -shared -fPIC -O3 -g -std=gnu++17 -Wall -Wextra -Werror \
        -Wl,--build-id \
        -I"$SCRIPT_DIR" \
        "$SCRIPT_DIR/aicpu_producer.cpp" \
        -o "$OWNER"

    echo "[BUILD] copied/renamed Path-A bootstrap dispatcher"
    "$HCC" -shared -fPIC -O3 -g -std=gnu++17 -Wall -Wextra -Werror \
        -Wl,--build-id \
        "$SCRIPT_DIR/plan_aicpu_dispatcher.cpp" \
        -o "$DISPATCHER"

    echo "[BUILD] self-contained host runner"
    "$CXX_BIN" -O2 -g -std=c++17 -Wall -Wextra -Werror -Wno-deprecated-declarations \
        -I"$SCRIPT_DIR" \
        -I"$ASCEND_HOME_PATH/include" \
        -I"$ASCEND_HOME_PATH/pkg_inc" \
        -I"$ASCEND_HOME_PATH/pkg_inc/runtime" \
        -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
        "$SCRIPT_DIR/host.cpp" \
        -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -lascendcl -lruntime -ldl -pthread \
        -o "$HOST"
    echo "[BUILD] complete: $BUILD_DIR"
}

run_probe() {
    for artifact in "$KERNEL" "$DISPATCHER" "$OWNER" "$HOST"; do
        if [[ ! -e "$artifact" ]]; then
            echo "Missing build artifact: $artifact; run '$0 build' first." >&2
            exit 1
        fi
    done
    echo "[RUN] device=${ATOMIC_PROBE_DEVICE:-${TASK_DEVICE:-0}}"
    timeout "${PLAN_PROTOCOL_TIMEOUT:-180}" \
        "$HOST" "$KERNEL" "$DISPATCHER" "$OWNER"
}

if [[ "$ACTION" == "all" || "$ACTION" == "build" ]]; then
    build_probe
fi
if [[ "$ACTION" == "all" || "$ACTION" == "run" ]]; then
    run_probe
fi
