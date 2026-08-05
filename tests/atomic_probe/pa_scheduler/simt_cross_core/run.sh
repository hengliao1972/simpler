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
S0_ROOT="$SCRIPT_DIR/protocol_probe"
S0_BUILD="$S0_ROOT/build/ccec"

usage() {
    cat <<'EOF'
Usage:
  ./run.sh build-s0
  ./run.sh run-s0 --device N [--runs N]

build-s0 运行 CPU optimized/ASan/UBSan/TSan，并构建、静态检查 CCEC/ELF。
run-s0 只运行已构建的真实 A5 探针；调用前仍需按仓库规则执行 A5 precheck。
EOF
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
fi

ACTION="$1"
shift
case "$ACTION" in
    build-s0)
        if [[ $# -ne 0 ]]; then
            echo "build-s0 does not accept additional arguments." >&2
            exit 1
        fi
        "$S0_ROOT/cpu/build.sh"
        "$S0_ROOT/ccec/build.sh"
        ;;
    run-s0)
        if [[ ! -x "$S0_BUILD/simt_cross_core_s0_host" ||
              ! -s "$S0_BUILD/simt_cross_core_s0_kernel.o" ]]; then
            echo "S0 artifacts are missing; run: $0 build-s0" >&2
            exit 1
        fi
        "$S0_BUILD/simt_cross_core_s0_host" \
            --kernel "$S0_BUILD/simt_cross_core_s0_kernel.o" "$@"
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
