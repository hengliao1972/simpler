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
SIMT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ORDINARY_ROOT="$(cd "$SIMT_ROOT/.." && pwd)"
SCALAR_ROOT="$ORDINARY_ROOT/scalar_build"
HOST_SOURCE="$SCALAR_ROOT/ccec/host.cpp"
BUILD_DIR="$SIMT_ROOT/build/cpu/host_acl_init"
CXX_BIN="${CXX:-g++}"
NM_BIN="${NM:-nm}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set; source the CANN environment first." >&2
    exit 1
fi
if ! command -v "$CXX_BIN" >/dev/null 2>&1 ||
   ! command -v "$NM_BIN" >/dev/null 2>&1;
then
    echo "A C++ compiler and nm are required for the Host ACL-init gate." >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

python3 - "$HOST_SOURCE" <<'PY'
import json
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1]).read_text()
match = re.search(
    r'constexpr char kConfig\[\] = R"acl\((.*?)\)acl";', source, re.DOTALL
)
if match is None:
    raise SystemExit("ordinary SIMT Host lacks its auditable raw ACL config")
if json.loads(match.group(1)) != {
    "StackSize": {
        "simt_stack_size": 8704,
        "simt_divergence_stack_size": 8704,
    }
}:
    raise SystemExit("ordinary SIMT Host ACL config is not exact 8704/8704 B")
for token in (
    "kCompiledRuntimePlanBuildBackend ==",
    "RuntimePlanBuildBackend::Simt",
    "aclInit(path)",
    "return aclInit(nullptr);",
    'CheckAcl(InitAclForCompiledRuntimePlanBuildBackend(), "aclInit")',
):
    if token not in source:
        raise SystemExit("ordinary Host ACL-init source contract is missing: " + token)
PY

COMMON_FLAGS=(
    -c
    -O2
    -std=c++17
    -Wall
    -Wextra
    -Werror
    -Wno-deprecated-declarations
    -DPTO_FDWIC_SHARED_MAP=1
    -DPTO_FDWIC_TENSORMAP_RING_CAP=128
    -DPTO_FDWIC_SHARED_INSERT_TURN_GROUPS=1
    -DPA_BUILD_SWIMLANE=1
    -DPA_BUILD_ATOMIC_SWIMLANE=1
    -DPA_BUILD_COMPACT_GENERIC_TRACE=1
    -DPA_BUILD_SUBMIT_PMU=0
    -DPA_BUILD_PERF_CLOCK=0
    -DPA_SUBMIT_PMU_PHASE_ID=0
    -DPA_CCEC_BLOCK_LOCAL_STATS=1
    -I"$SCALAR_ROOT/common"
    -I"$ASCEND_HOME_PATH/include"
    -I"$ASCEND_HOME_PATH/pkg_inc"
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime"
    -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime"
)

SCALAR_OBJECT="$BUILD_DIR/host_scalar.o"
SIMT_OBJECT="$BUILD_DIR/host_simt.o"

echo "[BUILD] ordinary Scalar Host ACL initialization"
"$CXX_BIN" "${COMMON_FLAGS[@]}" \
    -DPA_RUNTIME_PLAN_BUILD_BACKEND=0 \
    -DPA_RUNTIME_PLAN_BUILD_WORKERS=96 \
    "$HOST_SOURCE" -o "$SCALAR_OBJECT"

echo "[BUILD] ordinary SIMT Host ACL initialization"
"$CXX_BIN" "${COMMON_FLAGS[@]}" \
    -DPA_RUNTIME_PLAN_BUILD_BACKEND=1 \
    -DPA_RUNTIME_PLAN_BUILD_WORKERS=4 \
    "$HOST_SOURCE" -o "$SIMT_OBJECT"

scalar_undefined="$($NM_BIN -u "$SCALAR_OBJECT")"
simt_undefined="$($NM_BIN -u "$SIMT_OBJECT")"
if grep -Eq '(^|[[:space:]])(mkstemp|write)(@.*)?$' <<<"$scalar_undefined"; then
    echo "Scalar Host must compile to the original aclInit(nullptr) path without SIMT temp-config I/O." >&2
    exit 1
fi
for symbol in mkstemp write; do
    if ! grep -Eq "(^|[[:space:]])${symbol}(@.*)?$" <<<"$simt_undefined"; then
        echo "SIMT Host object is missing temporary ACL-config I/O: $symbol" >&2
        exit 1
    fi
done
if strings "$SCALAR_OBJECT" | grep -Fq 'pa_scheduler_simt_acl_'; then
    echo "Scalar Host object leaked the SIMT ACL temporary-file path." >&2
    exit 1
fi
simt_strings="$(strings "$SIMT_OBJECT")"
for token in \
    'pa_scheduler_simt_acl_' \
    'simt_stack_size' \
    'simt_divergence_stack_size';
do
    if ! grep -Fq "$token" <<<"$simt_strings"; then
        echo "SIMT Host object is missing its exact ACL-init contract: $token" >&2
        exit 1
    fi
done

echo "[PASS] Scalar keeps aclInit(nullptr); SIMT embeds secure 8704/8704 B temporary ACL config."
