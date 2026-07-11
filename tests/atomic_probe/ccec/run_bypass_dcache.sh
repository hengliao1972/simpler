#!/bin/bash
# Compatibility entry point for the CCEC bypass-DCache probe.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/run_all.sh" bypass_dcache_ccec "${1:-all}"
