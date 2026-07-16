#!/bin/bash
# Build & run ALL ccec atomic probes on A5 onboard hardware.
#
# For each probe: compiles the kernel .cpp as AIV-only with ccec -x cce,
# links it into an AICore binary with ld.lld, then compiles the host launcher
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
#   ./run_all.sh <probe_name> build|run
#
# Requires: ASCEND_HOME_PATH set (source $ASCEND_HOME_PATH/bin/setenv.bash)
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "Error: ASCEND_HOME_PATH not set. Run: source \$ASCEND_HOME_PATH/bin/setenv.bash"
    exit 1
fi

if [ -z "${PTO_ISA_ROOT:-}" ] || [ ! -f "$PTO_ISA_ROOT/include/pto/common/kernel_meta.hpp" ]; then
    echo "Error: PTO_ISA_ROOT must point to a PTO-ISA checkout containing include/pto/common/kernel_meta.hpp" >&2
    exit 1
fi

CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD="$ASCEND_HOME_PATH/bin/ld.lld"

# CCEC kernels may use compiler intrinsic headers and the PTO metadata header,
# but must not resolve any AscendC/kernel_operator header through this runner.
INC_FLAGS=(
    -I"$SCRIPT_DIR"
    -I"$PTO_ISA_ROOT/include"
)

CCEC_FLAGS=(
    -c -O3 -g -x cce -Wall -std=c++17
    --cce-aicore-only
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -mllvm -cce-aicore-dcci-before-kernel-end=false
)

BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"
RUN_TIMEOUT="${ATOMIC_PROBE_TIMEOUT:-120}"

# Probe table: kernel_src : kernel_obj : host_src : host_bin
PROBES=(
    "atomic_cas_probe.cpp:atomic_cas_kernel.o:atomic_cas_host.cpp:atomic_cas_host"
    "entire_flush_clobber.cpp:entire_flush_clobber_kernel.o:entire_flush_clobber_host.cpp:entire_flush_clobber_host"
    "bypass_dcache_ccec.cpp:bypass_dcache_kernel.o:bypass_dcache_ccec_host.cpp:bypass_dcache_ccec_host"
    "dcci_clean_clobber.cpp:dcci_clean_kernel.o:dcci_clean_clobber_host.cpp:dcci_clean_host"
    "atomic_blast.cpp:atomic_blast_kernel.o:atomic_blast_host.cpp:atomic_blast_host"
    "dcci_seam.cpp:dcci_seam_kernel.o:dcci_seam_host.cpp:dcci_seam_host"
    "dcci_atomic_clobber.cpp:dcci_atomic_clobber_kernel.o:dcci_atomic_clobber_host.cpp:dcci_atomic_clobber_host"
    "concurrent_stress.cpp:concurrent_stress_kernel.o:concurrent_stress_host.cpp:concurrent_stress_host"
    "atomic_exch_same_line.cpp:atomic_exch_same_line_kernel.o:st_dev_same_line_host.cpp:atomic_exch_same_line_host"
    "st_dev_same_line.cpp:st_dev_same_line_kernel.o:st_dev_same_line_host.cpp:st_dev_same_line_host"
    "st_dev_separate_line_stress.cpp:st_dev_separate_line_stress_kernel.o:st_dev_separate_line_stress_host.cpp:st_dev_separate_line_stress_host"
    "st_dev_single_core_stress.cpp:st_dev_single_core_stress_kernel.o:st_dev_single_core_stress_host.cpp:st_dev_single_core_stress_host"
    "ld_dev_fanout_publish.cpp:ld_dev_fanout_publish_kernel.o:ld_dev_fanout_publish_host.cpp:ld_dev_fanout_publish_host"
    "cacheline_matrix.cpp:cacheline_matrix_kernel.o:cacheline_matrix_host.cpp:cacheline_matrix_host"
)

REQUESTED="${1:-all}"
SELECT=""
ACTION="all"
if [[ "$REQUESTED" == "all" || "$REQUESTED" == "build" || "$REQUESTED" == "run" ]]; then
    ACTION="$REQUESTED"
    if [[ $# -gt 1 ]]; then
        echo "Unexpected argument: $2" >&2
        exit 1
    fi
else
    SELECT="$REQUESTED"
    ACTION="${2:-all}"
    if [[ "$ACTION" != "all" && "$ACTION" != "build" && "$ACTION" != "run" ]]; then
        echo "Unknown action: $ACTION" >&2
        exit 1
    fi
    if [[ $# -gt 2 ]]; then
        echo "Unexpected argument: $3" >&2
        exit 1
    fi
fi

echo "=== CCEC Atomic Probe Metadata ==="
echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git_head=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "ascend_home=$ASCEND_HOME_PATH"
echo "pto_isa_root=$PTO_ISA_ROOT"
echo "pto_isa_head=$(git -C "$PTO_ISA_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "ccec=$($CCEC --version 2>&1 | head -n 1)"
echo "device=${ATOMIC_PROBE_DEVICE:-${TASK_DEVICE:-0}}"
echo "action=$ACTION"
echo "timeout_seconds=$RUN_TIMEOUT"
printf '\n'

build_one() {
    local ks="$1" ko="$2" hs="$3" hb="$4"
    local tag
    tag="$(basename "$ko" .o)"

    local probe_flags=(-DCCEC_SYNC_AIV_ONLY)
    if [[ "$ks" == "cacheline_matrix.cpp" ]]; then
        probe_flags+=(-DCCEC_MATRIX_AIV_ONLY)
    fi

    echo "=== [$tag] Compiling AIV-only (dav-c310-vec) ==="
    "$CCEC" "${CCEC_FLAGS[@]}" --cce-aicore-arch=dav-c310-vec \
        "${probe_flags[@]}" "${INC_FLAGS[@]}" \
        -o "$BUILD_DIR/${tag}_vec.o" "$SCRIPT_DIR/$ks"

    echo "=== [$tag] Linking AICore binary ==="
    "$LD" -m aicorelinux -Ttext=0 -static --allow-multiple-definition \
        -o "$BUILD_DIR/$ko" "$BUILD_DIR/${tag}_vec.o"

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
    local probe_failures=0
    tag="$(basename "$ko" .o)"
    echo "=== Running [$tag] ==="
    if [[ "$tag" == "cacheline_matrix_kernel" ]]; then
        if [[ -n "${ATOMIC_PROBE_MATRIX_MODE:-}" ]]; then
            matrix_modes=("$ATOMIC_PROBE_MATRIX_MODE")
        else
            matrix_modes=(0 1 2 3 4 5 6 7)
        fi
        for matrix_mode in "${matrix_modes[@]}"; do
            echo "--- CCEC matrix variant=aiv mode=$matrix_mode ---"
            if ! ATOMIC_PROBE_MATRIX_MODE="$matrix_mode" \
                timeout "$RUN_TIMEOUT" "$BUILD_DIR/$hb" "$BUILD_DIR/$ko" aiv; then
                probe_failures=$((probe_failures + 1))
            fi
        done
    elif [[ "$tag" == "bypass_dcache_kernel" ]]; then
        if [[ -n "${ATOMIC_PROBE_MODE:-}" ]]; then
            probe_modes=("$ATOMIC_PROBE_MODE")
        else
            probe_modes=(0 1 2 3 4 5 6 7)
        fi
        for probe_mode in "${probe_modes[@]}"; do
            echo "--- CCEC bypass mode=$probe_mode ---"
            if ! ATOMIC_PROBE_MODE="$probe_mode" \
                timeout "$RUN_TIMEOUT" "$BUILD_DIR/$hb" "$BUILD_DIR/$ko"; then
                probe_failures=$((probe_failures + 1))
            fi
        done
    elif [[ "$tag" == "dcci_clean_kernel" ]]; then
        if [[ -n "${ATOMIC_PROBE_MODE:-}" ]]; then
            probe_modes=("$ATOMIC_PROBE_MODE")
        else
            probe_modes=(0 1 2 3 4)
        fi
        for probe_mode in "${probe_modes[@]}"; do
            echo "--- CCEC dcci-clean mode=$probe_mode ---"
            if ! ATOMIC_PROBE_MODE="$probe_mode" \
                timeout "$RUN_TIMEOUT" "$BUILD_DIR/$hb" "$BUILD_DIR/$ko"; then
                probe_failures=$((probe_failures + 1))
            fi
        done
    elif [[ "$tag" == "atomic_blast_kernel" ]]; then
        if [[ -n "${ATOMIC_PROBE_MODE:-}" ]]; then
            probe_modes=("$ATOMIC_PROBE_MODE")
        else
            probe_modes=(0 1 2)
        fi
        for probe_mode in "${probe_modes[@]}"; do
            echo "--- CCEC atomic-blast mode=$probe_mode ---"
            if ! ATOMIC_PROBE_MODE="$probe_mode" \
                timeout "$RUN_TIMEOUT" "$BUILD_DIR/$hb" "$BUILD_DIR/$ko"; then
                probe_failures=$((probe_failures + 1))
            fi
        done
    elif [[ "$tag" == "concurrent_stress_kernel" ]]; then
        if [[ -n "${ATOMIC_PROBE_MODE:-}" ]]; then
            probe_modes=("$ATOMIC_PROBE_MODE")
        else
            probe_modes=(0 1 2)
        fi
        for probe_mode in "${probe_modes[@]}"; do
            echo "--- CCEC concurrent-stress mode=$probe_mode ---"
            if ! ATOMIC_PROBE_MODE="$probe_mode" \
                timeout "$RUN_TIMEOUT" "$BUILD_DIR/$hb" "$BUILD_DIR/$ko"; then
                probe_failures=$((probe_failures + 1))
            fi
        done
    elif [[ "$tag" == "dcci_seam_kernel" ]]; then
        if [[ -n "${ATOMIC_PROBE_MODE:-}" ]]; then
            probe_modes=("$ATOMIC_PROBE_MODE")
        else
            probe_modes=(0 1 2 3 4)
        fi
        for probe_mode in "${probe_modes[@]}"; do
            echo "--- CCEC dcci-seam mode=$probe_mode ---"
            if ! ATOMIC_PROBE_MODE="$probe_mode" \
                timeout "$RUN_TIMEOUT" "$BUILD_DIR/$hb" "$BUILD_DIR/$ko"; then
                probe_failures=$((probe_failures + 1))
            fi
        done
    elif [[ "$tag" == "dcci_atomic_clobber_kernel" ]]; then
        if [[ -n "${ATOMIC_PROBE_MODE:-}" ]]; then
            probe_modes=("$ATOMIC_PROBE_MODE")
        else
            probe_modes=(0 1 2 3 4 5 6 7)
        fi
        for probe_mode in "${probe_modes[@]}"; do
            echo "--- CCEC dcci-atomic-clobber mode=$probe_mode ---"
            if ! ATOMIC_PROBE_MODE="$probe_mode" \
                timeout "$RUN_TIMEOUT" "$BUILD_DIR/$hb" "$BUILD_DIR/$ko"; then
                probe_failures=$((probe_failures + 1))
            fi
        done
    elif [[ "$tag" == "st_dev_same_line_kernel" ]]; then
        if [[ -n "${ATOMIC_PROBE_MODE:-}" ]]; then
            probe_modes=("$ATOMIC_PROBE_MODE")
        else
            probe_modes=(0 1 2)
        fi
        for probe_mode in "${probe_modes[@]}"; do
            echo "--- CCEC st-dev same-line mode=$probe_mode ---"
            if ! ATOMIC_PROBE_MODE="$probe_mode" \
                timeout "$RUN_TIMEOUT" "$BUILD_DIR/$hb" "$BUILD_DIR/$ko"; then
                probe_failures=$((probe_failures + 1))
            fi
        done
    elif [[ "$tag" == "st_dev_separate_line_stress_kernel" ]]; then
        if [[ -n "${ATOMIC_PROBE_MODE:-}" ]]; then
            probe_modes=("$ATOMIC_PROBE_MODE")
        else
            probe_modes=(0 1 2 3)
        fi
        for probe_mode in "${probe_modes[@]}"; do
            echo "--- CCEC st-dev separate-line-only stress mode=$probe_mode ---"
            if ! ATOMIC_PROBE_MODE="$probe_mode" \
                timeout "$RUN_TIMEOUT" "$BUILD_DIR/$hb" "$BUILD_DIR/$ko"; then
                probe_failures=$((probe_failures + 1))
            fi
        done
    elif [[ "$tag" == "st_dev_single_core_stress_kernel" ]]; then
        if [[ -n "${ATOMIC_PROBE_MODE:-}" ]]; then
            probe_modes=("$ATOMIC_PROBE_MODE")
        else
            # High-signal loop-end cases plus their per-write DSB controls.
            # Modes 0/2/3 remain available through ATOMIC_PROBE_MODE.
            probe_modes=(1 4 5 6)
        fi
        for probe_mode in "${probe_modes[@]}"; do
            echo "--- CCEC st-dev single-core stress mode=$probe_mode ---"
            if ! ATOMIC_PROBE_MODE="$probe_mode" \
                timeout "$RUN_TIMEOUT" "$BUILD_DIR/$hb" "$BUILD_DIR/$ko"; then
                probe_failures=$((probe_failures + 1))
            fi
        done
    elif [[ "$tag" == "ld_dev_fanout_publish_kernel" ]]; then
        if [[ -n "${ATOMIC_PROBE_MODE:-}" ]]; then
            probe_modes=("$ATOMIC_PROBE_MODE")
        else
            probe_modes=(0 1 2)
        fi
        for probe_mode in "${probe_modes[@]}"; do
            echo "--- CCEC ld-dev fanout publish mode=$probe_mode ---"
            if ! ATOMIC_PROBE_MODE="$probe_mode" \
                timeout "$RUN_TIMEOUT" "$BUILD_DIR/$hb" "$BUILD_DIR/$ko"; then
                probe_failures=$((probe_failures + 1))
            fi
        done
    else
        if ! timeout "$RUN_TIMEOUT" "$BUILD_DIR/$hb" "$BUILD_DIR/$ko"; then
            probe_failures=$((probe_failures + 1))
        fi
    fi
    if [[ "$probe_failures" -ne 0 ]]; then
        echo "CCEC [$tag] failed runs: $probe_failures" >&2
        return 1
    fi
    echo
}

export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"

selected=0
suite_run_failures=0
for entry in "${PROBES[@]}"; do
    IFS=':' read -r ks ko hs hb <<< "$entry"
    if [[ -n "$SELECT" && "$(basename "$ks" .cpp)" != "$SELECT" ]]; then
        continue
    fi
    selected=$((selected + 1))

    if [[ "$ACTION" == "all" || "$ACTION" == "build" ]]; then
        build_one "$ks" "$ko" "$hs" "$hb"
    fi
    if [[ "$ACTION" == "all" || "$ACTION" == "run" ]]; then
        if ! run_one "$ko" "$hb"; then
            suite_run_failures=$((suite_run_failures + 1))
        fi
    fi
done

if [[ "$selected" -eq 0 ]]; then
    echo "Unknown CCEC probe: $SELECT" >&2
    exit 1
fi

echo "=== Done. run_failures=$suite_run_failures ==="
if [[ "$suite_run_failures" -ne 0 ]]; then
    exit 1
fi
