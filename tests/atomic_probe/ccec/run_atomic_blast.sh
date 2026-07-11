#!/bin/bash
# Compatibility entry point for the CCEC atomic blast-radius probe.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/run_all.sh" atomic_blast "${1:-all}"
