#!/bin/bash
# Redis fault 1: one Redis cluster node is stopped, like a process or host failure.
# Symptom: redis_up drops to 0 for that instance; cluster_state may degrade.
# Restore: start the node; it rejoins the cluster.
# Usage: 01_node_stopped.sh <inject|restore>  [TARGET=redis2]
set -euo pipefail
source "$(dirname "$0")/../common.sh"

TARGET="${TARGET:-redis2}"

inject() {
    echo "Stopping $TARGET..."
    compose stop "$TARGET"
}

restore() {
    echo "Starting $TARGET (rejoins cluster automatically, container filesystem persisted)..."
    compose start "$TARGET"
}

case "${1:-}" in
    inject) log_change inject; inject ;;
    restore) log_change restore; restore ;;
    *) echo "Usage: $0 <inject|restore>  (optional: TARGET=redis2 env var)"; exit 1 ;;
esac
