#!/bin/bash
# Config fault 15: Redis client retries set high with zero backoff.
# Symptom: none while Redis is healthy (visible only in config); any transient
# Redis issue turns into pile-on retry load.
# Restore: unset the overrides and recreate $TARGET (library defaults).
# Usage: 15_bad_retry_backoff.sh <inject|restore>  [TARGET=go-server HIGH_RETRIES=20]
set -euo pipefail
source "$(dirname "$0")/../common.sh"

TARGET="${TARGET:-go-server}"
HIGH_RETRIES="${HIGH_RETRIES:-20}"

inject() {
    echo "Setting MaxRetries=$HIGH_RETRIES with zero backoff on $TARGET..."
    export REDIS_MAX_RETRIES="$HIGH_RETRIES"
    export REDIS_MIN_RETRY_BACKOFF_MS="0"
    export REDIS_MAX_RETRY_BACKOFF_MS="0"
    compose up -d --force-recreate "$TARGET"
    echo "Done. Static misconfig — pair with a transient Redis fault to see the pile-on effect."
}

restore() {
    echo "Clearing retry/backoff overrides on $TARGET (back to library defaults: 3 retries, 8ms-512ms backoff)..."
    unset REDIS_MAX_RETRIES REDIS_MIN_RETRY_BACKOFF_MS REDIS_MAX_RETRY_BACKOFF_MS
    compose up -d --force-recreate "$TARGET"
}

case "${1:-}" in
    inject) log_change inject; inject ;;
    restore) log_change restore; restore ;;
    *) echo "Usage: $0 <inject|restore>  (optional: TARGET=go-server HIGH_RETRIES=20)"; exit 1 ;;
esac
