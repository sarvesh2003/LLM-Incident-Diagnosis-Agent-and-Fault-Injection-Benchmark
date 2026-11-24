#!/bin/bash
# Resource fault 11: the target container is paused (frozen, not killed).
# Symptom: the container still shows "Up" in `docker ps` but does no work.
# Restore: unpause it.
# Usage: 11_container_paused.sh <inject|restore>  [TARGET=go-server]
set -euo pipefail
source "$(dirname "$0")/../common.sh"

TARGET="${TARGET:-go-server}"

inject() {
    echo "Pausing $TARGET..."
    compose pause "$TARGET"
}

restore() {
    echo "Unpausing $TARGET..."
    compose unpause "$TARGET"
}

case "${1:-}" in
    inject) log_change inject; inject ;;
    restore) log_change restore; restore ;;
    *) echo "Usage: $0 <inject|restore>  (optional: TARGET=go-server)"; exit 1 ;;
esac
