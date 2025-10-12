#!/usr/bin/env bash
set -euo pipefail

# Simple readiness probe for the local SGLang server (host network)
# Prints: ready|starting|down

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
URL="http://${HOST}:${PORT}/get_model_info"

status() {
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS --max-time 1 "$URL" >/dev/null; then
      echo ready; return 0
    fi
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 1 "$URL" || echo "")"
    if printf '%s' "$code" | grep -Eq '^[1-5][0-9]{2}$'; then
      echo starting; return 0
    fi
  else
    # Fallback to bash+timeout+cat /dev/tcp (best effort)
    if (exec 3<>/dev/tcp/${HOST}/${PORT}) 2>/dev/null; then
      echo starting; return 0
    fi
  fi
  echo down
}

status
