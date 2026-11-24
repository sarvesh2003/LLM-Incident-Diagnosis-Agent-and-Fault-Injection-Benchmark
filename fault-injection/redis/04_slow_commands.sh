#!/bin/bash
# Redis fault 4: DEBUG SLEEP blocks the event loop, standing in for a slow command (e.g. KEYS).
# Symptom: latency histogram shift with no error logs.
# Restore: none needed; DEBUG SLEEP ends on its own.
# Usage: 04_slow_commands.sh <inject|restore>  [TARGET=redis1 DURATION=10]
set -euo pipefail
source "$(dirname "$0")/../common.sh"

TARGET="${TARGET:-redis1}"
DURATION="${DURATION:-10}"

inject() {
    echo "Blocking $TARGET's event loop for ${DURATION}s via DEBUG SLEEP..."
    redis_cli "$TARGET" DEBUG SLEEP "$DURATION"
    echo "Done."
}

restore() {
    echo "No restore needed — DEBUG SLEEP is self-terminating."
}

case "${1:-}" in
    inject) log_change inject; inject ;;
    restore) log_change restore; restore ;;
    *) echo "Usage: $0 <inject|restore>  (optional: TARGET=redis1 DURATION=10)"; exit 1 ;;
esac
