#!/bin/bash
# Resource fault 12: the target container's CPU is capped with docker update --cpus.
# Symptom: the CPU-starved service cannot keep pace with incoming load.
# Restore: recreate the container from the compose file.
# Usage: 12_cpu_throttle.sh <inject|restore>  [TARGET=go-server LOW_CPUS=0.1]
set -euo pipefail
source "$(dirname "$0")/../common.sh"

TARGET="${TARGET:-go-server}"
LOW_CPUS="${LOW_CPUS:-0.1}"

inject() {
    CID="$(container_id "$TARGET")"
    echo "Throttling $TARGET to ${LOW_CPUS} CPUs..."
    docker update --cpus="$LOW_CPUS" "$CID"
}

restore() {
    echo "Recreating $TARGET to fully clear the CPU cap..."
    # docker update ignores zero values, so neither --cpus=0 nor
    # --cpu-quota=-1 clears HostConfig.NanoCpus; the cap would survive.
    # Recreating from the compose file is the reliable way to clear it.
    compose up -d --force-recreate -- "$TARGET"
}

case "${1:-}" in
    inject) log_change inject; inject ;;
    restore) log_change restore; restore ;;
    *) echo "Usage: $0 <inject|restore>  (optional: TARGET=go-server LOW_CPUS=0.1)"; exit 1 ;;
esac
