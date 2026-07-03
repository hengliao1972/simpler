#!/usr/bin/env bash
# CCEC probe for orch cpp — C.4.b work.
#
# Standalone compile of a fdwic example's orchestration source with CCEC to
# discover which device-space decorations / structural changes it still
# needs before it can live inside aicore_kernel.o. Do NOT wire the output
# into the build — the .o is discarded.
#
# Usage:
#   bash scripts/probe_bgemm_orch_ccec.sh [orch_cpp_path]
#
# Defaults to benchmark_bgemm's orch. Exits non-zero if CCEC rejects.

set -u
set -o pipefail

CCEC="${CCEC:-/usr/local/Ascend/cann-9.1.T500/bin/ccec}"
SIMPLER_ROOT="${SIMPLER_ROOT:-/home/pyptouser/chenpeng/simpler}"

ORCH="${1:-${SIMPLER_ROOT}/examples/a5/fully_distributed_within_core/benchmark_bgemm/kernels/orchestration/bgemm_orch.cpp}"
OUT_DIR="${OUT_DIR:-/tmp/orch_ccec_probe}"
mkdir -p "${OUT_DIR}"

INCS=(
    -I"${SIMPLER_ROOT}/src/a5/platform/onboard/aicore"
    -I"${SIMPLER_ROOT}/src/a5/platform/include"
    -I"${SIMPLER_ROOT}/src/common/platform/include"
    -I"${SIMPLER_ROOT}/src/common/task_interface"
    -I"${SIMPLER_ROOT}/src/common/log/include"
    -I"${SIMPLER_ROOT}/src/common"
    -I"${SIMPLER_ROOT}/src/a5/runtime/fully_distributed_within_core/runtime"
    -I"${SIMPLER_ROOT}/src/a5/runtime/fully_distributed_within_core/common"
    -I"${SIMPLER_ROOT}/src/a5/runtime/fully_distributed_within_core/orchestration"
    -I"${SIMPLER_ROOT}/src/a5/runtime"
)

FLAGS=(
    -c -O3 -g -x cce -Wall -std=c++17
    --cce-aicore-only
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    --cce-aicore-arch=dav-c310-cube
)

OUT_O="${OUT_DIR}/$(basename "${ORCH}" .cpp).o"

echo "[probe] ccec compiling ${ORCH##${SIMPLER_ROOT}/} -> ${OUT_O}"
"${CCEC}" "${FLAGS[@]}" "${INCS[@]}" -o "${OUT_O}" "${ORCH}"
rc=$?
if [[ $rc -eq 0 ]]; then
    echo "[probe] OK"
else
    echo "[probe] FAILED (rc=${rc})"
fi
exit $rc
