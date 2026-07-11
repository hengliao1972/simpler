#!/bin/bash
# Build & run the ccec ByPassDCache probe on A5 onboard hardware.
#
# Compiles bypass_dcache_ccec.cpp with ccec -x cce (AIC + AIV), links into
# an AICore binary, then compiles the host launcher with g++ and runs it.
#
# Key difference from bisheng -xasc: this uses
#   -mllvm -cce-aicore-dcci-insert-for-scalar=false
# which prevents the compiler from auto-inserting dcci after scalar stores.
# This gives precise control over L1 cache state — critical for mode 5
# (L1 pollution test) where we need the scalar store to stay in L1 only.
#
# Usage:
#   ./run_bypass_dcache.sh          # build + run
#   ./run_bypass_dcache.sh run      # run only (skip build)
#
# Requires: ASCEND_HOME_PATH set (source $ASCEND_HOME_PATH/bin/setenv.bash)
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "Error: ASCEND_HOME_PATH not set. Run: source \$ASCEND_HOME_PATH/bin/setenv.bash"
    exit 1
fi

CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD="$ASCEND_HOME_PATH/bin/ld.lld"

TIKCFW="$ASCEND_HOME_PATH/x86_64-linux/tikcpp/tikcfw"
ASC_INCLUDE="$ASCEND_HOME_PATH/x86_64-linux/asc/include"
ASC_IMPL="$ASCEND_HOME_PATH/x86_64-linux/asc/impl"
ASCENDC_INCLUDE="$ASCEND_HOME_PATH/x86_64-linux/ascendc/include"

INC_FLAGS=(
    -I"$TIKCFW"
    -I"$ASC_INCLUDE" -I"$ASC_INCLUDE/basic_api"
    -I"$ASC_IMPL" -I"$ASC_IMPL/basic_api"
    -I"$ASCENDC_INCLUDE" -I"$ASCENDC_INCLUDE/basic_api"
    -I"$ASCENDC_INCLUDE/basic_api/impl"
)

CCEC_FLAGS=(
    -c -O3 -g -x cce -Wall -std=c++17
    --cce-aicore-only
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
)

BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

KERNEL_SRC="$SCRIPT_DIR/bypass_dcache_ccec.cpp"
KERNEL_OBJ="$BUILD_DIR/bypass_dcache_kernel.o"
HOST_SRC="$SCRIPT_DIR/bypass_dcache_ccec_host.cpp"
HOST_BIN="$BUILD_DIR/bypass_dcache_ccec_host"

ACTION="${1:-all}"

if [ "$ACTION" != "run" ]; then
    echo "=== Compiling AIC (dav-c310-cube) ==="
    "$CCEC" "${CCEC_FLAGS[@]}" --cce-aicore-arch=dav-c310-cube \
        "${INC_FLAGS[@]}" -o "$BUILD_DIR/bypass_aic.o" "$KERNEL_SRC"

    echo "=== Compiling AIV (dav-c310-vec) ==="
    "$CCEC" "${CCEC_FLAGS[@]}" --cce-aicore-arch=dav-c310-vec \
        "${INC_FLAGS[@]}" -o "$BUILD_DIR/bypass_vec.o" "$KERNEL_SRC"

    echo "=== Linking AICore binary ==="
    "$LD" -m aicorelinux -Ttext=0 -static --allow-multiple-definition \
        -o "$KERNEL_OBJ" "$BUILD_DIR/bypass_aic.o" "$BUILD_DIR/bypass_vec.o"

    echo "=== Compiling host ==="
    g++ -O2 -std=c++17 \
        -I"$ASCEND_HOME_PATH/include" \
        "$HOST_SRC" \
        -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -lascendcl \
        -o "$HOST_BIN"

    echo "Build complete: $KERNEL_OBJ, $HOST_BIN"
fi

export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"
echo "=== Running ccec ByPassDCache probe ==="
"$HOST_BIN" "$KERNEL_OBJ"
