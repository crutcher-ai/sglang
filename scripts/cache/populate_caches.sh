#!/usr/bin/env bash
set -euo pipefail

# Host wrapper that invokes the in-container sgl-admin CLI to run provider prep
# and emits RESULT_* lines.

CONTAINER_NAME="${CONTAINER_NAME:-sglang-dev}"

args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--tp|--deep-gemm|--moe|--flashinfer|--inductor|--all)
      args+=("$1")
      if [[ "$1" != "--all" ]]; then args+=("$2"); shift; fi
      shift ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/cache/populate_caches.sh [flags]

This host wrapper forwards flags to the in-container admin CLI:
  docker exec -u devuser sglang-dev bash -lc \
    "python /workspaces/sglang/.devcontainer/tools/sgl_admin.py caches ensure ..."
EOF
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

container_id=$(docker ps -q -f name="$CONTAINER_NAME" | head -1)
if [[ -z "$container_id" ]]; then
  echo "ERROR: container $CONTAINER_NAME not running" >&2
  exit 1
fi

docker exec -u devuser "$container_id" bash -lc "python /workspaces/sglang/.devcontainer/tools/sgl_admin.py caches ensure ${args[*]@Q}" | sed -u 's/^/CONTAINER: /'
