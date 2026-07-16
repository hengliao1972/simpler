#!/bin/bash
# Compatibility entry point for the CCEC AtomicCAS probe.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/run_all.sh" atomic_cas_probe "${1:-all}"
