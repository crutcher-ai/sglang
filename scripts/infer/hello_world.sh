#!/usr/bin/env bash
set -euo pipefail

# Minimal one-shot inference to tick Prometheus counters.
# Usage: hello_world.sh [host] [port]

HOST=${1:-127.0.0.1}
PORT=${2:-30000}

payload='{"text":"The capital of France is","sampling_params":{"temperature":0.0,"max_new_tokens":1}}'
curl -fsS -m 5 -H 'Content-Type: application/json' \
  --data "$payload" "http://${HOST}:${PORT}/generate" >/dev/null || true
