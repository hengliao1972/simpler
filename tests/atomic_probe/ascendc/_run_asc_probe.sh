#!/usr/bin/env bash
# Onboard .asc probe runner for the atomic minibench suite.
#
# Compiles and runs each .asc probe on A5 onboard hardware. Each probe is a
# standalone CANN kernel + host program (like tests/atomic.asc) that directly
# tests the A5 hardware atomic/dcci seam.
#
# Usage:
#   ./_run_asc_probe.sh [probe_name]
#   ./_run_asc_probe.sh mb1_claim_probe
#   ./_run_asc_probe.sh                  # run all probes
#
# Requires: ASCEND_HOME_PATH set (source $ASCEND_HOME_PATH/bin/setenv.bash)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROBE="${1:-}"
RUN_TIMEOUT="${ATOMIC_PROBE_TIMEOUT:-60}"
BUILD_DIR="${ATOMIC_PROBE_BUILD_DIR:-}"
KEEP_BUILD="${ATOMIC_PROBE_KEEP_BUILD:-0}"

if ! command -v bisheng >/dev/null 2>&1; then
    echo "bisheng is not available; source the CANN setenv.bash first." >&2
    exit 1
fi

if [ -z "$BUILD_DIR" ]; then
    BUILD_DIR="$(mktemp -d /tmp/atomic_probe.XXXXXX)"
    if [ "$KEEP_BUILD" != "1" ]; then
        trap 'rm -rf "$BUILD_DIR"' EXIT
    fi
else
    mkdir -p "$BUILD_DIR"
fi

# Discover all .asc probes
ASC_FILES=()
if [ -n "$PROBE" ]; then
    PROBE="${PROBE%.asc}"
    if [ ! -f "$SCRIPT_DIR/$PROBE.asc" ]; then
        echo "Probe not found: $SCRIPT_DIR/$PROBE.asc" >&2
        exit 1
    fi
    ASC_FILES=("$SCRIPT_DIR/$PROBE.asc")
else
    for f in "$SCRIPT_DIR"/*.asc; do
        [ -f "$f" ] && ASC_FILES+=("$f")
    done
fi

if [ ${#ASC_FILES[@]} -eq 0 ]; then
    echo "No .asc probes found."
    exit 1
fi

REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
GIT_HEAD="$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
BISHENG_VERSION="$(bisheng --version 2>&1 | head -n 1)"

echo "=== Atomic Minibench Onboard Probes ==="
echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo=${REPO_ROOT:-unknown}"
echo "git_head=$GIT_HEAD"
echo "ascend_home=${ASCEND_HOME_PATH:-unset}"
echo "bisheng=$BISHENG_VERSION"
echo "npu_arch=dav-3510"
echo "device=${ATOMIC_PROBE_DEVICE:-${TASK_DEVICE:-0}}"
echo "timeout_seconds=$RUN_TIMEOUT"
echo "build_dir=$BUILD_DIR"
printf 'probes='
printf '%s ' "${ASC_FILES[@]##*/}"
printf '\n\n'

failures=0
executables=0

compile_and_run() {
    local asc="$1"
    local display_name="$2"
    local out_bin="$3"
    shift 3

    executables=$((executables + 1))
    echo "--- $display_name ---"
    if bisheng -xasc "$asc" --npu-arch=dav-3510 "$@" -o "$out_bin" 2>&1; then
        echo "  compiled -> $out_bin"
        if timeout "$RUN_TIMEOUT" "$out_bin"; then
            echo "  PASS"
        else
            status=$?
            echo "  RUN FAILED (exit $status)" >&2
            failures=$((failures + 1))
        fi
    else
        echo "  COMPILE FAILED" >&2
        failures=$((failures + 1))
    fi
    printf '\n'
}

for asc in "${ASC_FILES[@]}"; do
    name=$(basename "$asc" .asc)
    if [ "$name" = "cacheline_matrix" ]; then
        compile_and_run "$asc" "${name}[AIV]" "$BUILD_DIR/${name}_aiv_out" -DPROBE_CORE_VARIANT=0
        if [ "${ATOMIC_PROBE_RUN_MIX:-0}" = "1" ]; then
            compile_and_run "$asc" "${name}[AIC+AIV optional]" "$BUILD_DIR/${name}_mix_out" \
                -DPROBE_CORE_VARIANT=2 -D__MIX_CORE_AIC_RATION__=1
        fi
    else
        compile_and_run "$asc" "$name" "$BUILD_DIR/${name}_out"
    fi
done

echo "=== Atomic Minibench Summary: failures=$failures executables=$executables sources=${#ASC_FILES[@]} ==="
[ "$failures" -eq 0 ]
