#!/bin/bash
# Redis fault 2: low maxmemory with allkeys-lru, then a write flood that forces eviction.
# Symptom: redis_evicted_keys_total climbs; redis_memory_used_bytes pinned at the cap.
# Restore: maxmemory 0, policy noeviction, FLUSHALL on all three nodes.
# Usage: 02_eviction_storm.sh <inject|restore>  [TARGET=redis1 FLOOD_KEYS=200000]
set -euo pipefail
source "$(dirname "$0")/../common.sh"

TARGET="${TARGET:-redis1}"
LOW_MAXMEMORY="5mb"
FLOOD_KEYS="${FLOOD_KEYS:-200000}"

inject() {
    echo "Capping maxmemory=$LOW_MAXMEMORY, policy=allkeys-lru on $TARGET..."
    redis_cli "$TARGET" CONFIG SET maxmemory "$LOW_MAXMEMORY"
    redis_cli "$TARGET" CONFIG SET maxmemory-policy allkeys-lru

    echo "Flooding $TARGET with $FLOOD_KEYS keys to force eviction..."
    # --cluster spreads writes across all 3 nodes by hash slot, so restore()
    # flushes all 3.
    compose exec -T "$TARGET" redis-benchmark --cluster -q -t set -n "$FLOOD_KEYS" -d 200 -r 1000000000

    echo "Done. Check redis_evicted_keys_total in Prometheus."
}

restore() {
    echo "Restoring maxmemory=0, policy=noeviction on $TARGET..."
    redis_cli "$TARGET" CONFIG SET maxmemory 0
    redis_cli "$TARGET" CONFIG SET maxmemory-policy noeviction

    echo "Clearing fault-injected keys on redis1, redis2, redis3 (the flood landed on all 3, not just $TARGET)..."
    redis_cli redis1 FLUSHALL
    redis_cli redis2 FLUSHALL
    redis_cli redis3 FLUSHALL
}

case "${1:-}" in
    inject) log_change inject; inject ;;
    restore) log_change restore; restore ;;
    *) echo "Usage: $0 <inject|restore>  (optional: TARGET=redis1 FLOOD_KEYS=200000)"; exit 1 ;;
esac
