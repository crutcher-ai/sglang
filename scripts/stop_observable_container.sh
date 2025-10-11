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

CONTAINER_NAME="${CONTAINER_NAME:-sglang-dev}"

existing_cid=$(docker ps -aq -f name="^${CONTAINER_NAME}$" || true)
if [[ -n "${existing_cid}" ]]; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  echo "${CONTAINER_NAME}" stopped.
fi

HOST_OBS_ROOT="${HOST_OBS_ROOT:-$HOME/sglang-observability}"
RUN_META_FILE="${HOST_OBS_ROOT}/telemetry/container_run_meta.env"
rm -f "${RUN_META_FILE}" >/dev/null 2>&1 || true
