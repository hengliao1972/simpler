#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"
BUILD_DIR="$REPO_ROOT/tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/build/ordinary/scalar_build/aicpu"
PA_SOURCE="$REPO_ROOT/examples/a5/fully_distributed_within_core/paged_attention_unroll/kernels/orchestration/paged_attention_orch.cpp"

READY_PREFILL_TASKS="${PA_AICPU_PLAN_READY_PREFILL_TASKS:-128}"
if [[ ! "$READY_PREFILL_TASKS" =~ ^[0-9]+$ ]] ||
   ((READY_PREFILL_TASKS == 0 || READY_PREFILL_TASKS > 32768)); then
    echo "PA_AICPU_PLAN_READY_PREFILL_TASKS must be an integer in [1, 32768]." >&2
    exit 1
fi
SANITIZE="${PA_AICPU_PLAN_SANITIZE:-0}"
if [[ "$SANITIZE" != 0 && "$SANITIZE" != 1 ]]; then
    echo "PA_AICPU_PLAN_SANITIZE must be 0 or 1." >&2
    exit 1
fi
HOST_SANITIZE_FLAGS=()
if [[ "$SANITIZE" == 1 ]]; then
    HOST_SANITIZE_FLAGS=(
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
fi

mkdir -p "$BUILD_DIR/host" "$BUILD_DIR/aarch64"

COMMON_DEFINES=(
    -DPTO_FDWIC_SHARED_MAP=1
    -DPTO_FDWIC_SCHEDULER_MODE=1
    "-DPA_AICPU_PLAN_READY_PREFILL_TASKS=$READY_PREFILL_TASKS"
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
    local policy="$3"
    shift 3
    "$cxx" -std=c++17 -O3 -g -fPIC -fno-gnu-unique \
        -Wall -Wextra -Werror -Wno-unused-but-set-parameter \
        "${COMMON_DEFINES[@]}" \
        "-DPA_RUNTIME_PLAN_PIPELINE_POLICY=$policy" \
        "${COMMON_INCLUDES[@]}" \
        -shared -Wl,-z,defs \
        "$PA_SOURCE" \
        "$SCRIPT_DIR/aicpu_plan_backend.cpp" \
        "$SCRIPT_DIR/aicpu_plan_adapter_bridge.cpp" \
        "$@" \
        -o "$output"
}

required_symbols=(
    aicpu_orchestration_entry
    aicpu_orchestration_config
    aicpu_plan_backend_bind
    aicpu_plan_backend_close
    aicpu_plan_backend_result
)
HCC="${ASCEND_HOME_PATH:?ASCEND_HOME_PATH is required}/tools/hcc/bin/aarch64-target-linux-gnu-g++"
DISPATCHER_SO="$BUILD_DIR/aarch64/libpaged_attention_aicpu_plan_dispatcher.so"
rm -f -- "$DISPATCHER_SO"
"$HCC" -shared -fPIC -O3 -g -std=gnu++17 -Wall -Wextra -Werror \
    -Wl,--build-id \
    "$REPO_ROOT/tests/atomic_probe/pa_scheduler/cross_core_aicpu_plan/common/protocol_probe/plan_aicpu_dispatcher.cpp" \
    -o "$DISPATCHER_SO"
readelf -h "$DISPATCHER_SO" | awk -F: '/Machine:/ { if ($2 !~ /AArch64/) exit 1; found = 1 } END { exit(found ? 0 : 1) }'
for symbol in StaticTileFwkBackendKernelServer DynTileFwkBackendKernelServer DynTileFwkBackendKernelServerInit; do
    readelf -Ws "$DISPATCHER_SO" | awk -v symbol="$symbol" \
        '$7 != "UND" && $8 == symbol && $3 + 0 > 0 { found = 1 } END { exit(found ? 0 : 1) }'
done

POLICY_IDS=(0 1)
POLICY_NAMES=(plan-ahead-closed "streaming-future-p${READY_PREFILL_TASKS}")
AARCH64_SOS=()
for policy_index in "${!POLICY_IDS[@]}"; do
    policy="${POLICY_IDS[$policy_index]}"
    policy_name="${POLICY_NAMES[$policy_index]}"
    host_policy_dir="$BUILD_DIR/host/$policy_name"
    aarch64_policy_dir="$BUILD_DIR/aarch64/$policy_name"
    mkdir -p "$host_policy_dir" "$aarch64_policy_dir"

    HOST_SO="$host_policy_dir/libpaged_attention_aicpu_plan.so"
    HOST_TEST="$host_policy_dir/test_pa_orchestration_so"
    HOST_TRACE_SO="$host_policy_dir/libpaged_attention_aicpu_plan_trace.so"
    HOST_TRACE_TEST="$host_policy_dir/test_pa_orchestration_so_trace"
    AARCH64_SO="$aarch64_policy_dir/libpaged_attention_aicpu_plan.so"
    rm -f -- \
        "$HOST_SO" "$HOST_TEST" "$HOST_TRACE_SO" "$HOST_TRACE_TEST" \
        "$AARCH64_SO"

    echo "[BUILD] AICPU policy=$policy_name host smoke"
    build_so \
        "${CXX:-g++}" "$HOST_SO" "$policy" \
        "${HOST_SANITIZE_FLAGS[@]}"
    "${CXX:-g++}" -std=c++17 -O2 -g -Wall -Wextra -Werror \
        "${COMMON_DEFINES[@]}" \
        "-DPA_RUNTIME_PLAN_PIPELINE_POLICY=$policy" \
        "${COMMON_INCLUDES[@]}" \
        "${HOST_SANITIZE_FLAGS[@]}" \
        "$SCRIPT_DIR/test_pa_orchestration_so.cpp" -ldl \
        -o "$HOST_TEST"
    for symbol in "${required_symbols[@]}"; do
        readelf -Ws "$HOST_SO" | awk -v symbol="$symbol" \
            '$7 != "UND" && $8 == symbol { found = 1 } END { exit(found ? 0 : 1) }'
    done
    if readelf -Ws "$HOST_SO" | awk \
        '$7 == "UND" && ($8 ~ /^dist_/ || $8 ~ /^aicpu_plan_adapter_/) { found = 1 } END { exit(found ? 0 : 1) }'; then
        echo "unexpected unresolved backend symbol for policy=$policy_name" >&2
        exit 1
    fi
    "$HOST_TEST" "$HOST_SO"

    echo "[BUILD] AICPU policy=$policy_name trace-on host smoke"
    build_so \
        "${CXX:-g++}" "$HOST_TRACE_SO" "$policy" \
        -DPA_BUILD_SWIMLANE=1 \
        "${HOST_SANITIZE_FLAGS[@]}"
    "${CXX:-g++}" -std=c++17 -O2 -g -Wall -Wextra -Werror \
        "${COMMON_DEFINES[@]}" \
        -DPA_BUILD_SWIMLANE=1 \
        "-DPA_RUNTIME_PLAN_PIPELINE_POLICY=$policy" \
        "${COMMON_INCLUDES[@]}" \
        "${HOST_SANITIZE_FLAGS[@]}" \
        "$SCRIPT_DIR/test_pa_orchestration_so.cpp" -ldl \
        -o "$HOST_TRACE_TEST"
    "$HOST_TRACE_TEST" "$HOST_TRACE_SO"

    echo "[BUILD] AICPU policy=$policy_name AArch64 SO"
    build_so \
        "$HCC" "$AARCH64_SO" "$policy" \
        "$SCRIPT_DIR/aicpu_plan_owner.cpp"
    readelf -h "$AARCH64_SO" | awk -F: \
        '/Machine:/ { if ($2 !~ /AArch64/) exit 1; found = 1 } END { exit(found ? 0 : 1) }'
    for symbol in "${required_symbols[@]}"; do
        readelf -Ws "$AARCH64_SO" | awk -v symbol="$symbol" \
            '$7 != "UND" && $8 == symbol { found = 1 } END { exit(found ? 0 : 1) }'
    done
    readelf -Ws "$AARCH64_SO" | awk \
        '$7 != "UND" && $8 == "plan_protocol_aicpu_exec" && $3 + 0 > 0 { found = 1 } END { exit(found ? 0 : 1) }'
    if readelf -Ws "$AARCH64_SO" | awk \
        '$7 == "UND" && ($8 ~ /^dist_/ || $8 ~ /^aicpu_plan_adapter_/) { found = 1 } END { exit(found ? 0 : 1) }'; then
        echo "unexpected unresolved AArch64 backend symbol for policy=$policy_name" >&2
        exit 1
    fi
    AARCH64_SOS+=("$AARCH64_SO")
done

# Host smoke 只能锁死“同一地址清零后连续生产两轮”的协议顺序，不能模拟
# AICPU 对 Host DMA 的 stale clean cache。对最终 AArch64 产物再加一道指令
# 门槛：initialize 必须先执行 cell-control civac，完成 dsb+isb 后才出现
# 第一次 acquire load。这样至少不会因编译重排或后续重构静默删掉复用修复。
LLVM_OBJDUMP_BIN="${LLVM_OBJDUMP:-$ASCEND_HOME_PATH/bin/llvm-objdump}"
if [[ ! -x "$LLVM_OBJDUMP_BIN" ]]; then
    echo "Missing llvm-objdump for AICPU cache-order gate: $LLVM_OBJDUMP_BIN" >&2
    exit 1
fi
for policy_index in "${!AARCH64_SOS[@]}"; do
    AARCH64_SO="${AARCH64_SOS[$policy_index]}"
    policy_name="${POLICY_NAMES[$policy_index]}"
INITIALIZE_DISASSEMBLY="$($LLVM_OBJDUMP_BIN -d --demangle \
    --disassemble-symbols=aicpu_plan_adapter_initialize "$AARCH64_SO")"
if ! awk '
    /dc[[:space:]]+civac/ && !civac { civac = NR }
    /dsb[[:space:]]+sy/ && civac && !dsb { dsb = NR }
    /isb/ && dsb && !isb { isb = NR }
    /ldar/ && !load { load = NR }
    END {
        exit(civac && dsb && isb && load &&
             civac < dsb && dsb < isb && isb < load ? 0 : 1)
    }
' <<< "$INITIALIZE_DISASSEMBLY"; then
    echo "AICPU Plan reuse cache-order gate failed for policy=$policy_name: expected civac -> dsb sy -> isb -> ldar" >&2
    exit 1
fi

# 两条 policy 的 initialize 都必须先完成 cell Empty 检查和 reset
# control clean，之后才允许进入 Ready/ReadyFailed 的隔离 closed-line
# 发布路径。正常 Ready 的具体时机由上面的双 policy Host smoke 直接断言。
if ! awk '
    /dc[[:space:]]+civac/ && !cell_civac { cell_civac = NR }
    /ldar/ && cell_civac && !empty_load { empty_load = NR }
    /dc[[:space:]]+cvac/ && empty_load && !reset_clean {
        reset_clean = NR; next
    }
    /dsb[[:space:]]+sy/ && reset_clean && !reset_barrier {
        reset_barrier = NR; next
    }
    /isb/ && reset_barrier && !reset_isb { reset_isb = NR; next }
    /ldar/ && reset_isb && !ready_barrier { ready_load = NR }
    /dsb[[:space:]]+sy/ && ready_load && !ready_barrier {
        ready_barrier = NR; next
    }
    /str[[:space:]]+x[0-9]+,[[:space:]]*\[x[0-9]+,[[:space:]]*#128\]/ &&
        ready_barrier && !open_store { open_store = NR; next }
    /dc[[:space:]]+cvac/ && open_store && !open_clean {
        open_clean = NR; next
    }
    /dsb[[:space:]]+sy/ && open_clean && !open_barrier {
        open_barrier = NR; next
    }
    /isb/ && open_barrier && !open_isb { open_isb = NR }
    END {
        exit(cell_civac && empty_load && reset_clean && reset_barrier &&
             reset_isb && ready_load && ready_barrier && open_store &&
             open_clean && open_barrier && open_isb &&
             cell_civac < empty_load && empty_load < reset_clean &&
             reset_clean < reset_barrier && reset_barrier < reset_isb &&
             reset_isb < ready_load && ready_load < ready_barrier &&
             ready_barrier < open_store && open_store < open_clean &&
             open_clean < open_barrier && open_barrier < open_isb ? 0 : 1)
    }
' <<< "$INITIALIZE_DISASSEMBLY"; then
    echo "AICPU Plan ready-order gate failed for policy=$policy_name: isolated closed publish must follow cell/reset completion" >&2
    exit 1
fi

# owner 在 backend bind 之前也可能因 request/tensor metadata 失败。
# 该路径必须先使 fatal 可见，再在 Ready 同一条 closed line
# 发布 -3；否则只轮询 -2 的 AICore 永远不会看到 fatal。
OWNER_DISASSEMBLY="$($LLVM_OBJDUMP_BIN -d --demangle \
    --disassemble-symbols=plan_protocol_aicpu_exec "$AARCH64_SO")"
if ! awk '
    /str[[:space:]]+x[0-9]+,[[:space:]]*\[x[0-9]+,[[:space:]]*#640\]/ &&
        !fatal_store { fatal_store = NR; next }
    /dc[[:space:]]+cvac/ && fatal_store && !fatal_clean {
        fatal_clean = NR; next
    }
    /dsb[[:space:]]+sy/ && fatal_clean && !fatal_barrier {
        fatal_barrier = NR; next
    }
    /mov[[:space:]]+x[0-9]+,[[:space:]]*#-3/ &&
        fatal_barrier && !failed_value { failed_value = NR; next }
    /str[[:space:]]+x[0-9]+,[[:space:]]*\[x[0-9]+,[[:space:]]*#128\]/ &&
        failed_value && !closed_store { closed_store = NR; next }
    /dc[[:space:]]+cvac/ && closed_store && !closed_clean {
        closed_clean = NR; next
    }
    /dsb[[:space:]]+sy/ && closed_clean && !closed_barrier {
        closed_barrier = NR; next
    }
    /isb/ && closed_barrier && !closed_isb { closed_isb = NR }
    END {
        exit(fatal_store && fatal_clean && fatal_barrier && failed_value &&
             closed_store && closed_clean && closed_barrier && closed_isb &&
             fatal_store < fatal_clean && fatal_clean < fatal_barrier &&
             fatal_barrier < failed_value && failed_value < closed_store &&
             closed_store < closed_clean && closed_clean < closed_barrier &&
             closed_barrier < closed_isb ? 0 : 1)
    }
' <<< "$OWNER_DISASSEMBLY"; then
    echo "AICPU owner failure wake gate failed for policy=$policy_name: expected fatal clean -> closed=-3 clean" >&2
    exit 1
fi

# Finish callback 只预写 control=Empty 的 GM payload。PlanAheadClosed
# 可在 pack 前按需执行 cell-control civac+barrier+Empty 检查，
# 但最终 flags 尚未知道时不得对 payload 执行 cvac，更不得
# 发布 control；真正的 payload 可见性边界只在 publish_staged。
STAGE_DISASSEMBLY="$($LLVM_OBJDUMP_BIN -d --demangle \
    --disassemble-symbols=aicpu_plan_adapter_stage "$AARCH64_SO")"
if [[ "$policy_name" == "plan-ahead-closed" ]]; then
    if grep -Eq 'dc[[:space:]]+cvac' <<< "$STAGE_DISASSEMBLY"; then
        echo "AICPU Plan stage unexpectedly cleaned Empty-cell payload for policy=$policy_name" >&2
        exit 1
    fi
else
    if grep -Eq 'dc[[:space:]]+(cvac|civac)|dsb[[:space:]]+sy|isb' \
        <<< "$STAGE_DISASSEMBLY"; then
        echo "AICPU Plan stage unexpectedly made Empty-cell payload globally visible for policy=$policy_name" >&2
        exit 1
    fi
fi
if ! grep -Eq 'str[[:space:]]+x[0-9]+,' <<< "$STAGE_DISASSEMBLY"; then
    echo "AICPU Plan stage no longer contains the direct GM payload pack for policy=$policy_name" >&2
    exit 1
fi

# StreamingFuture 必须逐 task 完成 payload -> Published 发布。
# PlanAheadClosed 没有并发 consumer，只发出 exact payload/control
# clean，由 close 的最终 frontier DSB 统一等待完成。
PUBLISH_DISASSEMBLY="$($LLVM_OBJDUMP_BIN -d --demangle \
    --disassemble-symbols=aicpu_plan_adapter_publish_staged "$AARCH64_SO")"
if [[ "$policy_name" == "plan-ahead-closed" ]]; then
if ! awk '
    /dc[[:space:]]+cvac/ && !payload_clean { payload_clean = NR; next }
    /str[[:space:]]+x[0-9]+,/ && payload_clean && !control_store {
        control_store = NR; next
    }
    /dc[[:space:]]+cvac/ && control_store && !control_clean {
        control_clean = NR; next
    }
    END {
        exit(payload_clean && control_store && control_clean &&
             payload_clean < control_store &&
             control_store < control_clean ? 0 : 1)
    }
' <<< "$PUBLISH_DISASSEMBLY" ||
   grep -Eq 'dsb[[:space:]]+sy|isb' <<< "$PUBLISH_DISASSEMBLY"; then
    echo "AICPU Plan-ahead publish gate failed for policy=$policy_name: expected deferred payload cvac -> control store -> cvac without a per-task barrier" >&2
    exit 1
fi

CLOSE_DISASSEMBLY="$($LLVM_OBJDUMP_BIN -d --demangle \
    --disassemble-symbols=aicpu_plan_adapter_close "$AARCH64_SO")"
if ! awk '
    /ldar/ && !validated_load { validated_load = NR }
    /dsb[[:space:]]+sy/ && validated_load && !frontier_barrier {
        frontier_barrier = NR; next
    }
    /str[[:space:]]+x[0-9]+,[[:space:]]*\[x[0-9]+\]/ &&
        frontier_barrier && !frontier_store { frontier_store = NR; next }
    /dc[[:space:]]+cvac/ && frontier_store && !frontier_clean {
        frontier_clean = NR; next
    }
    /dsb[[:space:]]+sy/ && frontier_clean && !frontier_publish_barrier {
        frontier_publish_barrier = NR; next
    }
    /isb/ && frontier_publish_barrier && !frontier_isb {
        frontier_isb = NR
    }
    END {
        exit(validated_load && frontier_barrier && frontier_store &&
             frontier_clean && frontier_publish_barrier && frontier_isb &&
             validated_load < frontier_barrier &&
             frontier_barrier < frontier_store &&
             frontier_store < frontier_clean &&
             frontier_clean < frontier_publish_barrier &&
             frontier_publish_barrier < frontier_isb ? 0 : 1)
    }
' <<< "$CLOSE_DISASSEMBLY"; then
    echo "AICPU Plan-ahead close gate failed for policy=$policy_name: expected validation -> DSB -> final frontier store/clean -> DSB -> ISB" >&2
    exit 1
fi
else
if ! awk '
    /dc[[:space:]]+cvac/ && !payload_clean { payload_clean = NR; next }
    /dsb[[:space:]]+sy/ && payload_clean && !payload_barrier {
        payload_barrier = NR; next
    }
    /str[[:space:]]+x[0-9]+,/ && payload_barrier && !control_store {
        control_store = NR; next
    }
    /dc[[:space:]]+cvac/ && control_store && !control_clean {
        control_clean = NR; next
    }
    /dsb[[:space:]]+sy/ && control_clean && !control_barrier {
        control_barrier = NR; next
    }
    /isb/ && control_barrier && !control_isb { control_isb = NR }
    END {
        exit(payload_clean && payload_barrier && control_store &&
             control_clean && control_barrier && control_isb &&
             payload_clean < payload_barrier &&
             payload_barrier < control_store &&
             control_store < control_clean &&
             control_clean < control_barrier &&
             control_barrier < control_isb ? 0 : 1)
    }
' <<< "$PUBLISH_DISASSEMBLY"; then
    echo "AICPU Plan publish cache-order gate failed for policy=$policy_name: expected payload cvac -> dsb -> control store -> cvac -> dsb -> isb" >&2
    exit 1
fi
fi
done

echo "PASS both Runtime Plan policies: Host dlopen Ready/reuse + AArch64 owner/dispatcher/cache-order gates"
