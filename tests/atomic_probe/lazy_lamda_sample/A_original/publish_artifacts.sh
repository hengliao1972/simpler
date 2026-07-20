#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/build/original"
TARGET_DIR="$SCRIPT_DIR/artifacts/rebuilt"
mkdir -p "$TARGET_DIR"

for name in pa_scheduler_host pa_scheduler_kernel.o device_text_layout.manifest artifacts.manifest; do
    if [[ ! -f "$SOURCE_DIR/$name" ]]; then
        echo "Missing clean-build result: $SOURCE_DIR/$name" >&2
        exit 1
    fi
    cp -f -- "$SOURCE_DIR/$name" "$TARGET_DIR/$name"
done
(cd "$TARGET_DIR" && sha256sum artifacts.manifest device_text_layout.manifest pa_scheduler_host pa_scheduler_kernel.o) \
    > "$TARGET_DIR/published.sha256"
