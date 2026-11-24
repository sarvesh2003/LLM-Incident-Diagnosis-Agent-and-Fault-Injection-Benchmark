#!/bin/bash
# Kafka fault 8: a non-protobuf string published straight to raw_transactions, bypassing gRPC.
# Symptom: go-enricher fails to unmarshal it (fraud_enricher_errors_total{reason="unmarshal"}).
# Restore: none needed; the bad message is consumed and dropped.
# Usage: 08_malformed_event.sh <inject|restore>
set -euo pipefail
source "$(dirname "$0")/../common.sh"

inject() {
    echo "Publishing a malformed (non-protobuf) message to raw_transactions..."
    echo "this-is-not-a-valid-protobuf-payload-$(date +%s)" | compose exec -T kafka kafka-console-producer --broker-list localhost:9092 --topic raw_transactions
    echo "Done. Check fraud_enricher_errors_total{reason=\"unmarshal\"}."
}

restore() {
    echo "No restore needed — the bad message is consumed and dropped, no lasting state."
}

case "${1:-}" in
    inject) log_change inject; inject ;;
    restore) log_change restore; restore ;;
    *) echo "Usage: $0 <inject|restore>"; exit 1 ;;
esac
