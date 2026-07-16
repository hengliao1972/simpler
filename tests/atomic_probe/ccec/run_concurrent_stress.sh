#!/bin/bash
# Compatibility entry point for the CCEC concurrent stress probe.
set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/run_all.sh" concurrent_stress "${1:-all}"
