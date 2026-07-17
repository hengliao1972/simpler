#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Run the minimal nested-lambda/capture/template compiler probe on one or all
# of the CPU, AscendC, and pure-CCEC paths.
#
# Usage: ./run_nested_lambda.sh [cpu|ascendc|ccec|ccec-caller-capture|all]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
MODE="${1:-all}"
BUILD_DIR="$(mktemp -d /tmp/nested_lambda_probe.XXXXXX)"
trap 'rm -rf "$BUILD_DIR"' EXIT

run_cpu() {
    local compiler="${CXX:-g++}"
    echo "=== CPU compiler path ==="
    "$compiler" -O2 -std=c++17 -Wall -Wextra -Werror \
        "$SCRIPT_DIR/cpu/nested_lambda.cpp" \
        -o "$BUILD_DIR/nested_lambda_cpu"
    "$BUILD_DIR/nested_lambda_cpu"

    echo "=== CPU L0TaskArgs runtime-read path ==="
    "$compiler" -O2 -std=c++17 -Wall -Wextra -Werror -ffunction-sections \
        -I"$SCRIPT_DIR/ccec" \
        -I"$REPO_ROOT/src/a5/platform/onboard/aicore" \
        -I"$REPO_ROOT/src/a5/platform/include" \
        -I"$REPO_ROOT/src/common/platform/include" \
        -I"$REPO_ROOT/src/common/task_interface" \
        -I"$REPO_ROOT/src/common/log/include" \
        -I"$REPO_ROOT/src/common" \
        -I"$REPO_ROOT/src/a5/runtime/fully_distributed_within_core/runtime" \
        -I"$REPO_ROOT/src/a5/runtime/fully_distributed_within_core/common" \
        -I"$REPO_ROOT/src/a5/runtime/fully_distributed_within_core/orchestration" \
        -I"$REPO_ROOT/src/a5/runtime" \
        "$SCRIPT_DIR/cpu/nested_lambda_args_runtime_read.cpp" \
        -Wl,--gc-sections \
        -o "$BUILD_DIR/nested_lambda_args_runtime_read_cpu"
    "$BUILD_DIR/nested_lambda_args_runtime_read_cpu"
}

require_cann() {
    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        echo "ASCEND_HOME_PATH is not set; source CANN bin/setenv.bash first." >&2
        exit 1
    fi
}

run_ascendc() {
    require_cann
    echo "=== AscendC compiler/A5 path ==="
    "$SCRIPT_DIR/ascendc/_run_asc_probe.sh" nested_lambda
}

run_ccec() {
    require_cann
    if [[ -z "${PTO_ISA_ROOT:-}" ]] && \
       [[ -f "$ASCEND_HOME_PATH/x86_64-linux/include/pto/common/kernel_meta.hpp" ]]; then
        export PTO_ISA_ROOT="$ASCEND_HOME_PATH/x86_64-linux"
    fi
    echo "=== Pure CCEC compiler/A5 path ==="
    "$SCRIPT_DIR/ccec/run_all.sh" nested_lambda
}

run_ccec_caller_capture() {
    require_cann
    if [[ -z "${PTO_ISA_ROOT:-}" ]] && \
       [[ -f "$ASCEND_HOME_PATH/x86_64-linux/include/pto/common/kernel_meta.hpp" ]]; then
        export PTO_ISA_ROOT="$ASCEND_HOME_PATH/x86_64-linux"
    fi
    echo "=== Pure CCEC AIC inline caller-capture path ==="
    "$SCRIPT_DIR/ccec/run_all.sh" nested_lambda_cross_tu
}

case "$MODE" in
cpu)
    run_cpu
    ;;
ascendc)
    run_ascendc
    ;;
ccec)
    run_ccec
    ;;
ccec-caller-capture | ccec-cross-tu)
    run_ccec_caller_capture
    ;;
all)
    run_cpu
    run_ascendc
    run_ccec
    run_ccec_caller_capture
    ;;
*)
    echo "Usage: $0 [cpu|ascendc|ccec|ccec-caller-capture|all]" >&2
    exit 2
    ;;
esac
