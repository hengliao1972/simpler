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
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ $# -gt 2 ]]; then
    echo "Usage: $0 [private|shared] [swimlane|perf-clock]" >&2
    exit 1
fi
TENSORMAP_MODE="${1:-private}"
case "$TENSORMAP_MODE" in
    private) TENSORMAP_MODE_ID=0 ;;
    shared) TENSORMAP_MODE_ID=1 ;;
    *)
        echo "Unknown TensorMap mode: $TENSORMAP_MODE (expected private|shared)" >&2
        exit 1
        ;;
esac
BUILD_VARIANT="${2:-swimlane}"
case "$BUILD_VARIANT" in
    swimlane)
        VARIANT_DEFINES=(
            -DPA_BUILD_SWIMLANE=0
            -DPA_BUILD_SUBMIT_PMU=0
            -DPA_BUILD_PERF_CLOCK=0
        )
        ;;
    perf-clock)
        VARIANT_DEFINES=(
            -DPA_BUILD_SWIMLANE=0
            -DPA_BUILD_SUBMIT_PMU=0
            -DPA_BUILD_PERF_CLOCK=1
        )
        ;;
    *)
        echo "Unknown CPU build variant: $BUILD_VARIANT (expected swimlane|perf-clock)" >&2
        exit 1
        ;;
esac
BUILD_DIR="$ROOT_DIR/build/cpu/$TENSORMAP_MODE/$BUILD_VARIANT"
CXX_BIN="${CXX:-g++}"
TENSORMAP_RING_CAP=128
SHARED_INSERT_TURN_GROUPS="${PA_SHARED_INSERT_TURN_GROUPS:-1}"
case "$SHARED_INSERT_TURN_GROUPS" in
    1|2|4|8|16|32|64|128) ;;
    *)
        echo "PA_SHARED_INSERT_TURN_GROUPS must be a power of two from 1 through 128." >&2
        exit 1
        ;;
esac
if [[ "$TENSORMAP_MODE" != "shared" &&
      "$SHARED_INSERT_TURN_GROUPS" != "1" ]]; then
    echo "PA_SHARED_INSERT_TURN_GROUPS only applies to shared TensorMap builds." >&2
    exit 1
fi
VARIANT_DEFINES+=(
    "-DPTO_FDWIC_SHARED_INSERT_TURN_GROUPS=$SHARED_INSERT_TURN_GROUPS"
)
SCHEDULER_BINARY="pa_scheduler_cpu"
if [[ "$TENSORMAP_MODE" == "shared" &&
      "$SHARED_INSERT_TURN_GROUPS" != "1" ]]; then
    SCHEDULER_BINARY="pa_scheduler_cpu_turn_g${SHARED_INSERT_TURN_GROUPS}"
fi

# CPU 后端只依赖 C++17、pthread 和本目录 common/，不需要 CANN。
# CXX 可显式指向用户目录下的 g++-15，未设置时沿用当前 PATH 中的 g++。

# CPU build 与设备 build 使用平行目录，便于 run.sh 根据 backend 做严格选择，
# 也避免把 host 回归二进制误当成 A5 产物。
mkdir -p "$BUILD_DIR"

echo "[BUILD] CPU scheduler executable"
echo "[BUILD] shared insert-turn groups=$SHARED_INSERT_TURN_GROUPS"
# -pthread 同时提供编译期线程宏和链接期 pthread 支持；严格告警用于防止
# CPU 等价层因类型或原子接口变化而静默偏离设备端公共协议。private/shared
# 都实例化同一 scheduler，模式宏只选择各自已经接线的 TensorMap backend。
"$CXX_BIN" -O3 -std=c++17 -pthread -Wall -Wextra -Werror \
    "-DPTO_FDWIC_SHARED_MAP=$TENSORMAP_MODE_ID" \
    "-DPTO_FDWIC_TENSORMAP_RING_CAP=$TENSORMAP_RING_CAP" \
    "${VARIANT_DEFINES[@]}" \
    -I"$ROOT_DIR/common" \
    "$SCRIPT_DIR/main.cpp" \
    -o "$BUILD_DIR/$SCHEDULER_BINARY"

# PollBatch 是 common/ 中的设备/CPU 共用模板。这里用普通 C++17 编译器
# 直接实例化并执行边界自测；任一断言失败都会借助 set -e 阻止构建成功。
echo "[BUILD] atomic PollBatch boundary self-test"
"$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
    "-DPTO_FDWIC_SHARED_MAP=$TENSORMAP_MODE_ID" \
    -DPA_BUILD_SWIMLANE=1 \
    -I"$ROOT_DIR/common" \
    "$ROOT_DIR/test/test_atomic_poll_batch.cpp" \
    -o "$BUILD_DIR/test_atomic_poll_batch"

echo "[TEST] atomic PollBatch boundary self-test"
"$BUILD_DIR/test_atomic_poll_batch"

# TensorMap 独立回归不启动 96 个 worker，也不运行模拟 kernel：private
# 覆盖单线程 ring 的回收回绕与 reference 差分；shared ring 是当前
# ordered writer-delta 的 ordinary-region 原语，隔离覆盖 seq/ABA、回收
# 与容量预检。PA Case1 当前 ordinary entry 为零，因此这些门槛仍不能
# 代替后面的完整 96-worker Submit 测试。
if [[ "$TENSORMAP_MODE" == "private" ]]; then
    # 同一生产 helper 在固定 16K 总槽下覆盖多组 CAP×bucket 形态；
    # 正式 scheduler 仍只编默认 128，隔离门槛不冒充运行期 auto。
    for cap in 32 64 128 256 16384; do
        binary="$BUILD_DIR/test_private_tensor_map_ring_cap${cap}"
        echo "[BUILD] private TensorMap ring self-test CAP=$cap"
        "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
            -DPTO_FDWIC_SHARED_MAP=0 \
            "-DPTO_FDWIC_TENSORMAP_RING_CAP=$cap" \
            -DPA_BUILD_SWIMLANE=1 \
            -I"$ROOT_DIR/common" \
            "$ROOT_DIR/test/test_private_tensor_map_ring.cpp" \
            -o "$binary"

        echo "[TEST] private TensorMap ring self-test CAP=$cap"
        "$binary"
    done
else
    # shared writer 插入完成链位于 TaskCell::deps_prepared：task 0 无前驱，
    # task N 只等 N-1，发布时只 CAS 自己的完成字。旧 sidecar turn 全部
    # 写入 canary 并要求零触碰；另覆盖空写、损坏值与重复发布 fatal。
    echo "[BUILD] shared per-task insert-completion self-test"
    "$CXX_BIN" -O2 -std=c++17 -pthread -Wall -Wextra -Werror \
        -DPTO_FDWIC_SHARED_MAP=1 \
        "-DPTO_FDWIC_SHARED_INSERT_TURN_GROUPS=$SHARED_INSERT_TURN_GROUPS" \
        -DPA_BUILD_SWIMLANE=1 \
        -I"$ROOT_DIR/common" \
        "$ROOT_DIR/test/test_shared_insert_turn.cpp" \
        -o "$BUILD_DIR/test_shared_insert_completion"

    echo "[TEST] shared per-task insert-completion self-test"
    "$BUILD_DIR/test_shared_insert_completion"

    # host 必须从最终 SchedulerState.context_lens 独立重建 shared task
    # plan，不能复用 device helper 形成同错 oracle。该测试覆盖 G0/G1/G2/G4、
    # mixed 累计 batch_start、TaskAt 元数据、partial group 输出字节、writer
    # dependency chain，以及测试专用 CLI 的广播/逐 batch 形式。
    echo "[BUILD] shared authoritative host task-plan self-test"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
        -DPTO_FDWIC_SHARED_MAP=1 \
        -DPA_BUILD_SWIMLANE=1 \
        -I"$ROOT_DIR/common" \
        "$ROOT_DIR/test/test_shared_host_task_plan.cpp" \
        -o "$BUILD_DIR/test_shared_host_task_plan"

    echo "[TEST] shared authoritative host task-plan self-test"
    "$BUILD_DIR/test_shared_host_task_plan"

    for cap in 32 64 128 256 16384; do
        binary="$BUILD_DIR/test_shared_tensor_map_ring_cap${cap}"
        echo "[BUILD] isolated shared ordinary-region ring self-test CAP=$cap"
        "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
            -DPTO_FDWIC_SHARED_MAP=1 \
            "-DPTO_FDWIC_TENSORMAP_RING_CAP=$cap" \
            -DPA_BUILD_SWIMLANE=1 \
            -I"$ROOT_DIR/common" \
            "$ROOT_DIR/test/test_shared_tensor_map_ring.cpp" \
            -o "$binary"

        echo "[TEST] isolated shared ordinary-region ring self-test CAP=$cap"
        "$binary"
    done

    # shared raw 只保留真实稀疏边界：所有 task 有连续 EfDrain+Claim
    # 和 Submit 父区间，loser 没有业务子区间；PrepareMap 必须彻底缺席。
    echo "[BUILD] shared sparse raw-trace self-test"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
        -DPTO_FDWIC_SHARED_MAP=1 \
        -I"$ROOT_DIR/common" \
        "$ROOT_DIR/test/test_shared_sparse_trace.cpp" \
        -o "$BUILD_DIR/test_shared_sparse_trace"

    echo "[TEST] shared sparse raw-trace self-test"
    "$BUILD_DIR/test_shared_sparse_trace"

    # fresh-output symbol 与 region ring 是两条独立协议。该用例单独锁定
    # descriptor 最终封口、只读 fanin、ready descriptor 直写 slot、
    # 构建后 INOUT writer commit、失败 slot 撤销及非法引用 fail-closed，
    # 避免只靠完整 96 线程回放偶然覆盖。
    echo "[BUILD] shared-output symbol self-test"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror -pthread \
        -DPTO_FDWIC_SHARED_MAP=1 \
        -DPA_BUILD_SWIMLANE=1 \
        -I"$ROOT_DIR/common" \
        "$ROOT_DIR/test/test_shared_output_symbols.cpp" \
        -o "$BUILD_DIR/test_shared_output_symbols"

    echo "[TEST] shared-output symbol self-test"
    "$BUILD_DIR/test_shared_output_symbols"

    # 通用 writer-intent 门槛不使用 PA TaskKind/ticket：symbol 锁定
    # 多跳、跨 cache-line history、乱序和 partial-CAS 终止语义；
    # ownerless ordinary region 锁定 A->B->C，并验证空 transaction 也
    # 推进 per-task completion，旧 sidecar turn 保持 canary。
    echo "[BUILD] generic shared writer-intent self-test"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror -pthread \
        -DPTO_FDWIC_SHARED_MAP=1 \
        "-DPTO_FDWIC_SHARED_INSERT_TURN_GROUPS=$SHARED_INSERT_TURN_GROUPS" \
        -DPA_BUILD_SWIMLANE=1 \
        -I"$ROOT_DIR/common" \
        "$ROOT_DIR/test/test_shared_writer_intent.cpp" \
        -o "$BUILD_DIR/test_shared_writer_intent"

    echo "[TEST] generic shared writer-intent self-test"
    timeout --foreground 15s "$BUILD_DIR/test_shared_writer_intent"

    # shared heap 与 region/symbol 协议分开验证：锁定 8 shard、1 KiB
    # 对齐、首版禁止 wrap、并发唯一分配及 terminal 容量竞争不回滚。
    echo "[BUILD] shared heap no-wrap reserve self-test"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror -pthread \
        -DPTO_FDWIC_SHARED_MAP=1 \
        -DPA_BUILD_SWIMLANE=1 \
        -I"$ROOT_DIR/common" \
        "$ROOT_DIR/test/test_shared_heap_reserve.cpp" \
        -o "$BUILD_DIR/test_shared_heap_reserve"

    echo "[TEST] shared heap no-wrap reserve self-test"
    "$BUILD_DIR/test_shared_heap_reserve"

    # Claim 保持原 cursor 协议：Alloc/Cube 使用 prefix 四分片，Vector
    # 使用 shared sidecar 八分片；96 worker 锁定 role 候选数、唯一
    # winner、重复 loser，并要求 Claim 不触碰 deps_prepared。
    echo "[BUILD] shared cursor Claim self-test"
    "$CXX_BIN" -O2 -std=c++17 -pthread -Wall -Wextra -Werror \
        -DPTO_FDWIC_SHARED_MAP=1 \
        -DPA_BUILD_SWIMLANE=1 \
        -I"$ROOT_DIR/common" \
        "$ROOT_DIR/test/test_shared_vector_claim_cursor.cpp" \
        -o "$BUILD_DIR/test_shared_vector_claim_cursor"

    echo "[TEST] shared cursor Claim self-test"
    "$BUILD_DIR/test_shared_vector_claim_cursor"

    # Materialize 在触碰 shared cursor 前必须完成数量、引用、shape/stride
    # 和地址区间预检；这些 reserve 前拒绝路径不能推进 heap。FetchAdd 后
    # 才暴露的容量竞争则按 terminal 契约保留 overrun 现场。
    echo "[BUILD] shared winner materialize self-test"
    "$CXX_BIN" -O2 -std=c++17 -Wall -Wextra -Werror \
        -DPTO_FDWIC_SHARED_MAP=1 \
        -DPA_BUILD_SWIMLANE=1 \
        -I"$ROOT_DIR/common" \
        "$ROOT_DIR/test/test_shared_materialize.cpp" \
        -o "$BUILD_DIR/test_shared_materialize"

    echo "[TEST] shared winner materialize self-test"
    "$BUILD_DIR/test_shared_materialize"

    # 完整 96-worker Submit 逐 task 计数 cursor Claim、前驱 completion
    # load 和本 task completion CAS；同时锁定 loser 零 map 访问、旧
    # sidecar turn 零触碰，以及 lookup/Build/执行仍可跨前任 Build。
    echo "[BUILD] shared ordered-insert Submit self-test"
    "$CXX_BIN" -O2 -std=c++17 -pthread -Wall -Wextra -Werror \
        -DPTO_FDWIC_SHARED_MAP=1 \
        "-DPTO_FDWIC_SHARED_INSERT_TURN_GROUPS=$SHARED_INSERT_TURN_GROUPS" \
        -DPA_BUILD_SWIMLANE=1 \
        -I"$ROOT_DIR/common" \
        "$ROOT_DIR/test/test_shared_ordered_submit.cpp" \
        -o "$BUILD_DIR/test_shared_ordered_submit"

    echo "[TEST] shared ordered-insert Submit self-test"
    timeout --foreground 15s "$BUILD_DIR/test_shared_ordered_submit"
fi

# set -e 保证编译或链接失败时不会打印 complete，也不会在组合构建中继续
# 后续步骤；只有成功退出的构建才被本脚本声明为可运行产物。
echo "[BUILD] complete: $BUILD_DIR/$SCHEDULER_BINARY"
