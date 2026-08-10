#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
BUILD_DIR="$REPO_ROOT/tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/build/ordinary/scalar_build/aicpu"
PA_SOURCE="$REPO_ROOT/examples/a5/fully_distributed_within_core/paged_attention_unroll/kernels/orchestration/paged_attention_orch.cpp"

mkdir -p "$BUILD_DIR/host" "$BUILD_DIR/aarch64"

COMMON_DEFINES=(
    -DPTO_FDWIC_SHARED_MAP=1
    -DPTO_FDWIC_SCHEDULER_MODE=1
)
COMMON_INCLUDES=(
    -I"$REPO_ROOT/src/a5/platform/include"
    -I"$REPO_ROOT/src/common/platform/include"
    -I"$REPO_ROOT/src/common/task_interface"
    -I"$REPO_ROOT/src/common/log/include"
    -I"$REPO_ROOT/src/common"
    -I"$REPO_ROOT/src/a5/runtime/fully_distributed_within_core/runtime"
    -I"$REPO_ROOT/src/a5/runtime/fully_distributed_within_core/common"
    -I"$REPO_ROOT/src/a5/runtime"
    -I"$REPO_ROOT/src/a5/runtime/fully_distributed_within_core/orchestration"
    -I"$SCRIPT_DIR"
)

build_so() {
    local cxx="$1"
    local output="$2"
    "$cxx" -std=c++17 -O3 -g -fPIC -fno-gnu-unique \
        -Wall -Wextra -Werror -Wno-unused-but-set-parameter \
        "${COMMON_DEFINES[@]}" "${COMMON_INCLUDES[@]}" \
        -shared -Wl,-z,defs \
        "$PA_SOURCE" \
        "$SCRIPT_DIR/aicpu_plan_backend.cpp" \
        "$SCRIPT_DIR/aicpu_plan_adapter_bridge.cpp" \
        -o "$output"
}

HOST_SO="$BUILD_DIR/host/libpaged_attention_aicpu_plan.so"
build_so "${CXX:-g++}" "$HOST_SO"
"${CXX:-g++}" -std=c++17 -O2 -g -Wall -Wextra -Werror \
    "${COMMON_DEFINES[@]}" "${COMMON_INCLUDES[@]}" \
    "$SCRIPT_DIR/test_pa_orchestration_so.cpp" -ldl \
    -o "$BUILD_DIR/host/test_pa_orchestration_so"

required_symbols=(
    aicpu_orchestration_entry
    aicpu_orchestration_config
    aicpu_plan_backend_bind
    aicpu_plan_backend_close
    aicpu_plan_backend_result
)
for symbol in "${required_symbols[@]}"; do
    readelf -Ws "$HOST_SO" | awk -v symbol="$symbol" \
        '$7 != "UND" && $8 == symbol { found = 1 } END { exit(found ? 0 : 1) }'
done
if readelf -Ws "$HOST_SO" | awk \
    '$7 == "UND" && ($8 ~ /^dist_/ || $8 ~ /^aicpu_plan_adapter_/) { found = 1 } END { exit(found ? 0 : 1) }'; then
    echo "unexpected unresolved backend symbol" >&2
    exit 1
fi

"$BUILD_DIR/host/test_pa_orchestration_so" "$HOST_SO"

HCC="${ASCEND_HOME_PATH:?ASCEND_HOME_PATH is required}/tools/hcc/bin/aarch64-target-linux-gnu-g++"
AARCH64_SO="$BUILD_DIR/aarch64/libpaged_attention_aicpu_plan.so"
build_so "$HCC" "$AARCH64_SO"
readelf -h "$AARCH64_SO" | awk -F: '/Machine:/ { if ($2 !~ /AArch64/) exit 1; found = 1 } END { exit(found ? 0 : 1) }'
for symbol in "${required_symbols[@]}"; do
    readelf -Ws "$AARCH64_SO" | awk -v symbol="$symbol" \
        '$7 != "UND" && $8 == symbol { found = 1 } END { exit(found ? 0 : 1) }'
done
if readelf -Ws "$AARCH64_SO" | awk \
    '$7 == "UND" && ($8 ~ /^dist_/ || $8 ~ /^aicpu_plan_adapter_/) { found = 1 } END { exit(found ? 0 : 1) }'; then
    echo "unexpected unresolved AArch64 backend symbol" >&2
    exit 1
fi

echo "PASS Host dlopen + AArch64 readelf gates"
