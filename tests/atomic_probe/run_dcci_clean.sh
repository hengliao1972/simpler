#!/bin/bash
# Build & run the dcci clean-clobber probe.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    echo "Error: ASCEND_HOME_PATH not set"; exit 1
fi
CCEC="$ASCEND_HOME_PATH/bin/ccec"
LD="$ASCEND_HOME_PATH/bin/ld.lld"
TIKCFW="$ASCEND_HOME_PATH/x86_64-linux/tikcpp/tikcfw"
ASC_INCLUDE="$ASCEND_HOME_PATH/x86_64-linux/asc/include"
ASC_IMPL="$ASCEND_HOME_PATH/x86_64-linux/asc/impl"
ASCENDC_INCLUDE="$ASCEND_HOME_PATH/x86_64-linux/ascendc/include"
INC=(-I"$TIKCFW" -I"$ASC_INCLUDE" -I"$ASC_INCLUDE/basic_api" -I"$ASC_IMPL" -I"$ASC_IMPL/basic_api" -I"$ASCENDC_INCLUDE" -I"$ASCENDC_INCLUDE/basic_api" -I"$ASCENDC_INCLUDE/basic_api/impl")
CFLAGS=(-c -O3 -g -x cce -Wall -std=c++17 --cce-aicore-only
    -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false)
BD="$SCRIPT_DIR/build"; mkdir -p "$BD"
KS="$SCRIPT_DIR/dcci_clean_clobber.cpp"
KO="$BD/dcci_clean_kernel.o"
HS="$SCRIPT_DIR/dcci_clean_clobber_host.cpp"
HB="$BD/dcci_clean_host"
ACTION="${1:-all}"
if [ "$ACTION" != "run" ]; then
    echo "=== Compiling AIC ==="
    "$CCEC" "${CFLAGS[@]}" --cce-aicore-arch=dav-c310-cube "${INC[@]}" -o "$BD/dc_aic.o" "$KS"
    echo "=== Compiling AIV ==="
    "$CCEC" "${CFLAGS[@]}" --cce-aicore-arch=dav-c310-vec "${INC[@]}" -o "$BD/dc_vec.o" "$KS"
    echo "=== Linking ==="
    "$LD" -m aicorelinux -Ttext=0 -static --allow-multiple-definition -o "$KO" "$BD/dc_aic.o" "$BD/dc_vec.o"
    echo "=== Host ==="
    g++ -O2 -std=c++17 -I"$ASCEND_HOME_PATH/include" "$HS" \
        -L"$ASCEND_HOME_PATH/x86_64-linux/lib64" -Wl,-rpath,"$ASCEND_HOME_PATH/x86_64-linux/lib64" -lascendcl -o "$HB"
fi
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"
"$HB" "$KO"
