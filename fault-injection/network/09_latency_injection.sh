#!/bin/bash
# Network fault 9: tc netem delay on the target container's interface.
# A throwaway netshoot sidecar joins the container's network namespace, so the
# app image needs no changes and no permanent NET_ADMIN.
# Symptom: no error lines anywhere; only the latency histogram shifts.
# Restore: delete the netem qdisc.
# Usage: 09_latency_injection.sh <inject|restore>  [TARGET=go-server DELAY=200ms IFACE=eth0]
set -euo pipefail
source "$(dirname "$0")/../common.sh"

TARGET="${TARGET:-go-server}"
DELAY="${DELAY:-200ms}"
IFACE="${IFACE:-eth0}"

inject() {
    CID="$(container_id "$TARGET")"
    echo "Adding ${DELAY} delay to $TARGET's $IFACE..."
    docker run --rm --net=container:"$CID" --cap-add=NET_ADMIN nicolaka/netshoot \
        tc qdisc add dev "$IFACE" root netem delay "$DELAY"
    echo "Done. Check the app's own latency histogram — no error logs expected."
}

restore() {
    CID="$(container_id "$TARGET")"
    echo "Removing netem delay from $TARGET's $IFACE..."
    # Only tc's "Invalid qdisc name" (no netem qdisc present) means already
    # clean. Any other failure must surface: this fault leaves no error
    # signs, so a silently failed restore would corrupt every later trial.
    OUTPUT="$(docker run --rm --net=container:"$CID" --cap-add=NET_ADMIN nicolaka/netshoot \
        tc qdisc del dev "$IFACE" root netem 2>&1)" && echo "Removed." || {
        if [[ "$OUTPUT" == *"Invalid qdisc name"* ]]; then
            echo "(already clean, nothing to remove)"
        else
            echo "ERROR: restore failed unexpectedly: $OUTPUT" >&2
            exit 1
        fi
    }
}

case "${1:-}" in
    inject) log_change inject; inject ;;
    restore) log_change restore; restore ;;
    *) echo "Usage: $0 <inject|restore>  (optional: TARGET=go-server DELAY=200ms IFACE=eth0)"; exit 1 ;;
esac
