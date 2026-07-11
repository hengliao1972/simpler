#!/bin/bash
# Compatibility entry point for the CCEC dcci clean/clobber probe.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/run_all.sh" dcci_clean_clobber "${1:-all}"
