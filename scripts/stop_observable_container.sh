#!/usr/bin/env bash
set -euo pipefail

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'EOF'
Usage: stop_observable_container.sh

Stops the sglang observability helper container (if running).
No additional arguments are supported.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Pass --help for usage." >&2
      exit 2
      ;;
  esac
done

CONTAINER_NAME="sglang-dev"

if docker ps -aq -f name="${CONTAINER_NAME}" >/dev/null 2>&1; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  echo "${CONTAINER_NAME}" stopped.
fi
