#!/bin/bash
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
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROBE="${1:-}"

# Discover all .asc probes
ASC_FILES=()
if [ -n "$PROBE" ]; then
    ASC_FILES=("$SCRIPT_DIR/$PROBE.asc")
else
    for f in "$SCRIPT_DIR"/mb*_*.asc; do
        [ -f "$f" ] && ASC_FILES+=("$f")
    done
fi

if [ ${#ASC_FILES[@]} -eq 0 ]; then
    echo "No .asc probes found."
    exit 1
fi

echo "=== Atomic Minibench Onboard Probes ==="
echo "Probes: ${ASC_FILES[*]}"
echo ""

for asc in "${ASC_FILES[@]}"; do
    name=$(basename "$asc" .asc)
    echo "--- $name ---"
    out_bin="/tmp/${name}_out"
    if bisheng -xasc "$asc" --npu-arch=dav-3510 -o "$out_bin" 2>&1; then
        echo "  compiled -> $out_bin"
        timeout 60 "$out_bin" || echo "  RUN FAILED (exit $?)"
    else
        echo "  COMPILE FAILED"
    fi
    echo ""
done
