#!/bin/bash
# Kafka fault 17: the Kafka broker is stopped (sustained outage).
# Symptom: ingestion requests fail (no Kafka acknowledgment) and enrichment stops.
# Restore: start the broker again.
# Usage: 17_broker_stopped.sh <inject|restore>
set -euo pipefail
source "$(dirname "$0")/../common.sh"

case "${1:-}" in
    inject) log_change inject; compose stop kafka ;;
    restore) log_change restore; compose start kafka ;;
    *) echo "Usage: $0 <inject|restore>"; exit 1 ;;
esac
