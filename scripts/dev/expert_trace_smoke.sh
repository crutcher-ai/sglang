#!/usr/bin/env bash
# Runs a lightweight decode request and pulls expert trace artefacts.
set -euo pipefail

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
TRACE_DIR=${TRACE_DIR:-$HOME/sglang-expert-trace}
PHASE=${PHASE:-decode}
PROMPT=${PROMPT:-"List three prime numbers."}

export SGLANG_MOE_TRACE_DIR="$TRACE_DIR"
export SGLANG_MOE_TRACE_PHASE="$PHASE"

mkdir -p "$TRACE_DIR"

printf 'Trace dir: %s\nPhase: %s\n' "$TRACE_DIR" "$PHASE"

curl -sf "http://$HOST:$PORT/start_expert_distribution_record" >/dev/null || true

curl -sf -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"temperature\":0.2,\"max_tokens\":4}" >/dev/null

curl -sf "http://$HOST:$PORT/dump_expert_distribution_record" >/dev/null || true

printf '\nExpert trace files:\n'
find "$TRACE_DIR" -maxdepth 1 -name 'expert_trace_*' -print | sort || true
