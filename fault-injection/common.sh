#!/bin/bash
# Shared helpers sourced by every fault script.

COMPOSE_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/docker-compose.yaml"

compose() {
    docker compose -f "$COMPOSE_FILE" "$@"
}

redis_cli() {
    local node="$1"
    shift
    compose exec -T "$node" redis-cli "$@"
}

container_id() {
    # A scaled service returns one ID per line, but docker update and
    # --net=container: take exactly one. Faults applied through this helper
    # therefore hit only the first replica of a scaled service.
    compose ps -q "$1" | head -n1
}

# Local audit log, one JSON line per inject/restore. Git-ignored and never
# exposed to the agent through any tool.
CHANGELOG_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/changelog.jsonl"

log_change() {
    local action="$1"
    local fault_name family
    fault_name="$(basename "$0" .sh)"
    family="$(basename "$(dirname "$0")")"
    printf '{"timestamp":"%s","family":"%s","fault":"%s","action":"%s"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$family" "$fault_name" "$action" >> "$CHANGELOG_FILE"
}
