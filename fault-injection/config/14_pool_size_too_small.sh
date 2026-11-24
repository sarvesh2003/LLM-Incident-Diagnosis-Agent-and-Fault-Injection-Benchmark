#!/bin/bash
# Config fault 14: Redis client connection pool capped far below concurrent load.
# Symptom: requests queue for a free connection; pool-wait timeouts under load.
# Restore: unset the override and recreate $TARGET (library default).
# Usage: 14_pool_size_too_small.sh <inject|restore>  [TARGET=go-server LOW_POOL_SIZE=2]
set -euo pipefail
source "$(dirname "$0")/../common.sh"

TARGET="${TARGET:-go-server}"
LOW_POOL_SIZE="${LOW_POOL_SIZE:-2}"

inject() {
    echo "Setting Redis pool size to $LOW_POOL_SIZE on $TARGET..."
    export REDIS_POOL_SIZE="$LOW_POOL_SIZE"
    compose up -d --force-recreate "$TARGET"
    echo "Done. Send concurrent load to see pool-wait timeouts."
}

restore() {
    echo "Clearing pool size override on $TARGET (back to library default)..."
    unset REDIS_POOL_SIZE
    compose up -d --force-recreate "$TARGET"
}

case "${1:-}" in
    inject) log_change inject; inject ;;
    restore) log_change restore; restore ;;
    *) echo "Usage: $0 <inject|restore>  (optional: TARGET=go-server LOW_POOL_SIZE=2)"; exit 1 ;;
esac
