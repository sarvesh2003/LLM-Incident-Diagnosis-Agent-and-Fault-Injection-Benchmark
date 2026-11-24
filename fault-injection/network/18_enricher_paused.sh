#!/bin/bash
# Resource fault 18: the go-enricher consumer is paused (sustained).
# Symptom: consumer lag grows and enriched output stops while the container shows "Up".
# Restore: unpause go-enricher.
# Usage: 18_enricher_paused.sh <inject|restore>
set -euo pipefail
source "$(dirname "$0")/../common.sh"

case "${1:-}" in
    inject) log_change inject; compose pause go-enricher ;;
    restore) log_change restore; compose unpause go-enricher ;;
    *) echo "Usage: $0 <inject|restore>"; exit 1 ;;
esac
