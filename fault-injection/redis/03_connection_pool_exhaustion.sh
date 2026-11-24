#!/bin/bash
# Redis fault 3: server-side maxclients lowered so far that client pools get refused.
# Symptom: redis_connected_clients at the cap; "ERR max number of clients reached" in app logs.
# Restore: CONFIG SET maxclients back to the default, or restart the node if no slot is free.
# Usage: 03_connection_pool_exhaustion.sh <inject|restore>  [TARGET=redis1 LOW_MAXCLIENTS=5]
set -euo pipefail
source "$(dirname "$0")/../common.sh"

TARGET="${TARGET:-redis1}"
LOW_MAXCLIENTS="${LOW_MAXCLIENTS:-5}"
DEFAULT_MAXCLIENTS=10000

inject() {
    echo "Setting maxclients=$LOW_MAXCLIENTS on $TARGET..."
    redis_cli "$TARGET" CONFIG SET maxclients "$LOW_MAXCLIENTS"
    echo "Done. Check redis_connected_clients + app logs for connection refusals."
}

restore() {
    echo "Restoring maxclients=$DEFAULT_MAXCLIENTS on $TARGET..."
    # With every slot taken, CONFIG SET itself may not get a connection.
    # Restarting resets maxclients to the built-in default, since the startup
    # command never sets it. redis-cli exits 0 on error replies, so check the
    # reply text rather than the exit code.
    RESULT="$(redis_cli "$TARGET" CONFIG SET maxclients "$DEFAULT_MAXCLIENTS" 2>&1 || true)"
    echo "$RESULT"
    if [[ "$RESULT" != *OK* ]]; then
        echo "CONFIG SET was refused (no free connection slot) — restarting $TARGET instead..."
        compose restart "$TARGET"
    fi
}

case "${1:-}" in
    inject) log_change inject; inject ;;
    restore) log_change restore; restore ;;
    *) echo "Usage: $0 <inject|restore>  (optional: TARGET=redis1 LOW_MAXCLIENTS=5)"; exit 1 ;;
esac
