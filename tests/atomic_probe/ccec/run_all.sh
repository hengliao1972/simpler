#!/bin/bash
# Build & run ALL ccec atomic probes on A5 onboard hardware.
#
# For each probe: compiles the kernel .cpp with ccec -x cce (AIC + AIV),
# links into an AICore binary with ld.lld, then compiles the host launcher
# with g++ and runs it.
#
# All kernels are pure-CCEC (ccec_utils.h + lowercase builtins); no
# kernel_operator.h, no AscendC APIs.
#
# Usage:
#   ./run_all.sh                 # build + run all
#   ./run_all.sh build           # build only (skip run)
#   ./run_all.sh run             # run only (skip build)
#   ./run_all.sh <probe_name>    # build + run a single probe, e.g. atomic_blast
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
    -I"$SCRIPT_DIR"
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

# Probe table: kernel_src : kernel_obj : host_src : host_bin
PROBES=(
    "atomic_cas_probe.cpp:atomic_cas_kernel.o:atomic_cas_host.cpp:atomic_cas_host"
    "entire_flush_clobber.cpp:entire_flush_clobber_kernel.o:entire_flush_clobber_host.cpp:entire_flush_clobber_host"
    "bypass_dcache_ccec.cpp:bypass_dcache_kernel.o:bypass_dcache_ccec_host.cpp:bypass_dcache_ccec_host"
    "dcci_clean_clobber.cpp:dcci_clean_kernel.o:dcci_clean_clobber_host.cpp:dcci_clean_host"
    "atomic_blast.cpp:atomic_blast_kernel.o:atomic_blast_host.cpp:atomic_blast_host"
    "dcci_seam.cpp:dcci_seam_kernel.o:dcci_seam_host.cpp:dcci_seam_host"
    "concurrent_stress.cpp:concurrent_stress_kernel.o:concurrent_stress_host.cpp:concurrent_stress_host"
)

ACTION="${1:-all}"

build_one() {
    local ks="$1" ko="$2" hs="$3" hb="$4"
    local tag
    tag="$(basename "$ko" .o)"

    echo "=== [$tag] Compiling AIC (dav-c310-cube) ==="
    "$CCEC" "${CCEC_FLAGS[@]}" --cce-aicore-arch=dav-c310-cube \
        "${INC_FLAGS[@]}" -o "$BUILD_DIR/${tag}_aic.o" "$SCRIPT_DIR/$ks"

    echo "=== [$tag] Compiling AIV (dav-c310-vec) ==="
    "$CCEC" "${CCEC_FLAGS[@]}" --cce-aicore-arch=dav-c310-vec \
        "${INC_FLAGS[@]}" -o "$BUILD_DIR/${tag}_vec.o" "$SCRIPT_DIR/$ks"

    echo "=== [$tag] Linking AICore binary ==="
    "$LD" -m aicorelinux -Ttext=0 -static --allow-multiple-definition \
        -o "$BUILD_DIR/$ko" "$BUILD_DIR/${tag}_aic.o" "$BUILD_DIR/${tag}_vec.o"

    echo "=== [$tag] Compiling host ==="
    g++ -O2 -std=c++17 \
        -I"$ASCEND_HOME_PATH/include" \
        "$SCRIPT_DIR/$hs" \
        -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" \
        -lascendcl \
        -o "$BUILD_DIR/$hb"

    echo "[$tag] build complete: $ko, $hb"
}

run_one() {
    local ko="$1" hb="$2"
    local tag
    tag="$(basename "$ko" .o)"
    echo "=== Running [$tag] ==="
    "$BUILD_DIR/$hb" "$BUILD_DIR/$ko"
    echo
}

# Single-probe selection: ./run_all.sh atomic_blast
if [[ "$ACTION" != "all" && "$ACTION" != "build" && "$ACTION" != "run" ]]; then
    SELECT="$ACTION"
    ACTION="all"
else
    SELECT=""
fi

export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"

for entry in "${PROBES[@]}"; do
    IFS=':' read -r ks ko hs hb <<< "$entry"
    if [[ -n "$SELECT" && "$(basename "$ks" .cpp)" != "$SELECT" ]]; then
        continue
    fi

    if [[ "$ACTION" == "all" || "$ACTION" == "build" ]]; then
        build_one "$ks" "$ko" "$hs" "$hb"
    fi
    if [[ "$ACTION" == "all" || "$ACTION" == "run" ]]; then
        run_one "$ko" "$hb"
    fi
done

echo "=== Done. ==="
