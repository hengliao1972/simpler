#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VARIANT="compete-first-lazy"
BUILD_DIR="$SCRIPT_DIR/build/$VARIANT"

# This package deliberately owns exactly one build directory.  Remove the
# complete old directory so stale objects can never survive a rebuild.
if [[ "$BUILD_DIR" != "$SCRIPT_DIR/build/compete-first-lazy" ]]; then
    echo "Refusing to clean an unexpected build path: $BUILD_DIR" >&2
    exit 1
fi
if [[ -e "$BUILD_DIR" ]]; then
    rm -rf -- "$BUILD_DIR"
fi

"$SCRIPT_DIR/ccec/build.sh" "$VARIANT"
