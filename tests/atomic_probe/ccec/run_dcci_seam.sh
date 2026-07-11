#!/bin/bash
# Compatibility entry point for the CCEC publish/observe seam probe.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/run_all.sh" dcci_seam "${1:-all}"
