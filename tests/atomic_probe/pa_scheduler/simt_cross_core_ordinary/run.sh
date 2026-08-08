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
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
GM_ROOT="$SCRIPT_DIR/gm"
GM_BUILD="$GM_ROOT/build/ccec"
G0_BUILD_MANIFEST="$GM_BUILD/g0_build_manifest.sha256"
G0_SWIMLANE_BUILD_MANIFEST="$GM_BUILD/g0_swimlane_build_manifest.sha256"
G0_SWIMLANE_ACL_CONFIG="$GM_ROOT/ccec/g0_swimlane_acl.json"

usage() {
    cat <<'EOF'
Usage:
  ./run.sh build-gm
  ./run.sh run-gm --builders 1..32 [--device N] [--batches 1|256] [--runs N]
  ./run.sh build-gm-swimlane
  ./run.sh run-gm-swimlane --builders 1..32 [--device N] [--batches 1|256] --runs 1 --swimlane-json FILE

本目录只实现 GM 路径。SIMT builder 与 Scalar executor 沿用 simt_cross_core，
但真实 metadata writer 按 task-id 使用一条全局稀疏 TensorMap 插入链。
真实 PA workload 固定为 QK/SF/PV/UP=6/28/4/1。

构建时可沿用现有 SIMT_CROSS_CORE_GM_BUILDER_WARPS=1..64 选择每个 builder
的 warp 数，并用 SIMT_CROSS_CORE_GM_TOKENS_PER_OWNER=1..4 选择每个执行器
可同时跟踪的 task 数；SIMT_CROSS_CORE_GM_DISPATCH_WINDOW_BATCHES=1..256
选择上游优先派发窗口。环境提供 task-submit 时，真实 A5 运行必须放入 task-submit；
没有调度器的显式授权环境只允许 unlocked 运行，并提示结果可能受其他进程干扰。
EOF
}

require_a5_access() {
    local stage_label="$1"
    local precheck_rc=0
    if "$REPO_ROOT/.claude/skills/onboard-arch-precheck/check.sh" a5; then
        precheck_rc=0
    else
        precheck_rc=$?
    fi
    if (( precheck_rc != 0 )); then
        if (( precheck_rc != 1 )); then
            echo "$stage_label A5 arch precheck rejected this silicon (exit=$precheck_rc); refusing hardware access." >&2
            exit "$precheck_rc"
        fi
        if [[ -n "${TASK_DEVICE:-}" ]] || command -v task-submit >/dev/null 2>&1; then
            echo "$stage_label A5 arch precheck failed; refusing the submitted/managed hardware run." >&2
            exit 1
        fi
        printf '%s\n' \
            '[WARN] A5 arch precheck unavailable; proceeding only because this is an explicitly authorized unlocked run' \
            >&2
    fi
    if [[ -z "${TASK_DEVICE:-}" ]] && command -v task-submit >/dev/null 2>&1; then
        echo "$stage_label detected task-submit but is not inside a submitted task." >&2
        exit 1
    fi
    if [[ -z "${TASK_DEVICE:-}" ]]; then
        echo '[WARN] task-submit not found; running unlocked — results may be noisy if another process uses this NPU' >&2
    fi
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
fi

action="$1"
shift

case "$action" in
    build-gm)
        if [[ $# -ne 0 ]]; then
            echo "build-gm does not accept arguments; set SIMT_CROSS_CORE_GM_BUILDER_WARPS in the environment." >&2
            exit 1
        fi
        "$GM_ROOT/cpu/build_g0.sh"
        "$GM_ROOT/ccec/build_g0.sh"
        ;;
    build-gm-swimlane)
        if [[ $# -ne 0 ]]; then
            echo "build-gm-swimlane does not accept arguments; set SIMT_CROSS_CORE_GM_BUILDER_WARPS in the environment." >&2
            exit 1
        fi
        "$GM_ROOT/cpu/build_g0.sh"
        SIMT_CROSS_CORE_G0_VARIANT=swimlane "$GM_ROOT/ccec/build_g0.sh"
        ;;
    run-gm|run-gm-swimlane)
        swimlane=0
        host="$GM_BUILD/simt_cross_core_ordinary_g0_host"
        kernel="$GM_BUILD/simt_cross_core_ordinary_g0_kernel.o"
        manifest="$G0_BUILD_MANIFEST"
        fixed_args=()
        if [[ "$action" == "run-gm-swimlane" ]]; then
            swimlane=1
            host="$GM_BUILD/simt_cross_core_ordinary_g0_swimlane_host"
            kernel="$GM_BUILD/simt_cross_core_ordinary_g0_swimlane_kernel.o"
            manifest="$G0_SWIMLANE_BUILD_MANIFEST"
            fixed_args=(--acl-config "$G0_SWIMLANE_ACL_CONFIG")
        fi
        if [[ ! -x "$host" || ! -s "$kernel" || ! -s "$manifest" ]]; then
            echo "$action artifacts are missing or incomplete; run the matching build action first." >&2
            exit 1
        fi
        if ! (cd "$SCRIPT_DIR" && sha256sum --check --status "$manifest"); then
            echo "$action sources and artifacts do not match the successful-build manifest; rebuild first." >&2
            exit 1
        fi
        require_a5_access "$action"
        builder_arg_seen=0
        device_args=()
        for argument in "$@"; do
            case "$argument" in
                --kernel|--kernel=*|--acl-config|--acl-config=*)
                    echo "$action fixes its kernel and ACL configuration; do not override them." >&2
                    exit 1
                    ;;
                --builders|--builders=*)
                    builder_arg_seen=1
                    ;;
                --device|--device=*)
                    if [[ -n "${TASK_DEVICE:-}" ]]; then
                        echo "$action injects --device from TASK_DEVICE; do not pass it inside task-submit." >&2
                        exit 1
                    fi
                    ;;
            esac
        done
        if [[ "$builder_arg_seen" -ne 1 ]]; then
            echo "$action requires an explicit --builders N value in 1..32." >&2
            exit 1
        fi
        if [[ -n "${TASK_DEVICE:-}" ]]; then
            device_args=(--device "$TASK_DEVICE")
        fi
        timeout --foreground 600s "$host" --kernel "$kernel" "${fixed_args[@]}" "${device_args[@]}" "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "unknown action: $action" >&2
        usage >&2
        exit 1
        ;;
esac
