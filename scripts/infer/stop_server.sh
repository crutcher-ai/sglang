#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-sglang-dev}"
PORT="${PORT:-30000}"

# Best-effort stop
docker exec -u devuser "$CONTAINER_NAME" bash -lc "pkill -f 'sglang.launch_server' || true" >/dev/null 2>&1 || true

# Wait for port to be free (host network)
deadline=$((SECONDS+20))
while [ $SECONDS -lt $deadline ]; do
  if ! (exec 3<>/dev/tcp/127.0.0.1/${PORT}) 2>/dev/null; then
    echo "stopped"
    exit 0
  fi
  sleep 1
done

echo "WARN: server may still be running on port $PORT" >&2
exit 1

