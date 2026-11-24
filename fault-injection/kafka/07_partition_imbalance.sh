#!/bin/bash
# Kafka fault 7: a burst of transactions for one user_id; messages are keyed by user_id,
# so all of them land on a single partition.
# Symptom: offset and lag growth skewed to one partition while the others stay flat.
# Restore: none needed; only extra traffic is sent.
# Usage: 07_partition_imbalance.sh <inject|restore>  [BURST_USER=... BURST_COUNT=500]
set -euo pipefail
source "$(dirname "$0")/../common.sh"

# A self-describing user id would appear in service logs and give the agent
# the answer, so the burst uses a UUID like the real user ids.
BURST_USER="${BURST_USER:-a47ac10b-58cc-4372-a567-0e02b2c3d900}"
BURST_COUNT="${BURST_COUNT:-500}"

inject() {
    echo "Sending $BURST_COUNT transactions for a single user ($BURST_USER)..."
    compose exec -T python-producer python3 -c "
import sys, time, uuid, grpc
from proto.fraud.v1 import fraud_pb2, fraud_pb2_grpc

channel = grpc.insecure_channel('go-server:50051')
stub = fraud_pb2_grpc.FraudIngestionStub(channel)

sent, failed = 0, 0
for i in range($BURST_COUNT):
    req = fraud_pb2.TransactionRequest(
        transaction_id=str(uuid.uuid4()),
        user_id='$BURST_USER',
        amount=100.0,
        timestamp=int(time.time() * 1000),
        is_fraud=False,
        type='PAYMENT',
        old_balance_orig=1000.0,
        new_balance_orig=900.0,
        old_balance_dest=0.0,
        new_balance_dest=0.0,
        is_unauthorized_overdraft=0.0,
        ip_address='8.8.8.8',
    )
    try:
        stub.SendTransaction(req)
        sent += 1
    except grpc.RpcError:
        failed += 1

print(f'Burst done: {sent} sent, {failed} failed.')
# A mostly rejected burst creates no skew; fail so it is not mistaken for a working injection.
if sent < $BURST_COUNT * 0.9:
    print(f'ERROR: only {sent}/{$BURST_COUNT} transactions actually sent - the skew this fault relies on did not happen.', file=sys.stderr)
    sys.exit(1)
"
    echo "Done. Check per-partition offsets/lag for skew."
}

restore() {
    echo "No restore needed — this only sends extra traffic, no state to undo."
}

case "${1:-}" in
    inject) log_change inject; inject ;;
    restore) log_change restore; restore ;;
    *) echo "Usage: $0 <inject|restore>  (optional: BURST_USER=... BURST_COUNT=500)"; exit 1 ;;
esac
