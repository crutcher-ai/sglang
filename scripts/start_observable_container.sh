#!/usr/bin/env bash
set -euo pipefail

# Location of this script and repo root
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'EOF'
Usage: start_observable_container.sh

Starts the sglang observability helper container, waits for the per-run manifest
to become available, and prints both host-side and container-side manifest
paths. The script exits non-zero if the container fails to launch.

Environment variables:
  HOST_MANIFEST_ROOT  Override the host path used when composing manifest
                      pointers (defaults to /.devcontainer/storage/container_runs).

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOST_STORAGE_ROOT="$(realpath "${ROOT_DIR}/.devcontainer/storage")"
HOST_MANIFEST_ROOT="${HOST_STORAGE_ROOT}/container_runs"
RUN_META_FILE="${HOST_STORAGE_ROOT}/container_run_meta.env"

CONTAINER_NAME="sglang-dev"
IMAGE_NAME="sglang-dev:gh200"

docker ps -aq -f name="${CONTAINER_NAME}" >/dev/null 2>&1 && {
  echo "Removing existing container ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}

# Run storage prep to ensure bind mounts exist
"${ROOT_DIR}/.devcontainer/setup-storage.sh"

: > "${RUN_META_FILE}"

container_id=$(docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --shm-size 32g \
  --ipc host \
  --network host \
  --add-host host.docker.internal:host-gateway \
  --cap-add SYS_ADMIN \
  -e HOST_MANIFEST_ROOT="${HOST_MANIFEST_ROOT}" \
  -v "${HOST_STORAGE_ROOT}/models:/models" \
  -v "${HOST_STORAGE_ROOT}/huggingface:/home/devuser/.cache/huggingface" \
  -v "${HOST_STORAGE_ROOT}/profiles:/profiles" \
  -v "${HOST_STORAGE_ROOT}/logs:/telemetry/logs" \
  -v "${HOST_STORAGE_ROOT}/prometheus:/telemetry/prometheus" \
  -v "${HOST_STORAGE_ROOT}/jaeger:/telemetry/jaeger" \
  -v "${HOST_STORAGE_ROOT}/container_run_meta.env:/telemetry/container_run_meta.env" \
  -v "${HOST_MANIFEST_ROOT}:/telemetry/container_runs" \
  -w /workspaces/sglang \
  "${IMAGE_NAME}" \
  sleep infinity)

echo "${CONTAINER_NAME} is starting (container id ${container_id})."

echo "Waiting for manifest pointer at ${RUN_META_FILE}..."
manifest_container=""
manifest_host=""
start_time=$(date +%s)
last_heartbeat=-5

while true; do
  now=$(date +%s)
  elapsed=$((now - start_time))

  status=$(docker inspect -f '{{.State.Status}}' "${container_id}" 2>/dev/null || true)
  if [ -z "${status}" ]; then
    echo "ERROR: container ${CONTAINER_NAME} (${container_id}) disappeared during startup." >&2
    exit 1
  fi

  if [ "${status}" != "running" ]; then
    exit_code=$(docker inspect -f '{{.State.ExitCode}}' "${container_id}" 2>/dev/null || echo "unknown")
    reason=$(docker inspect -f '{{.State.Error}}' "${container_id}" 2>/dev/null || true)
    echo "ERROR: container ${CONTAINER_NAME} (${container_id}) exited while starting (status=${status}, exit_code=${exit_code})." >&2
    if [ -n "${reason}" ]; then
      echo "Container error: ${reason}" >&2
    fi
    echo "Last 50 container log lines:" >&2
    docker logs --tail 50 "${container_id}" >&2 || true
    exit 1
  fi

  if [ -s "${RUN_META_FILE}" ]; then
    manifest_container=$(awk -F= '/^CONTAINER_RUN_META_JSON=/{print $2}' "${RUN_META_FILE}" || true)
    manifest_host=$(awk -F= '/^CONTAINER_RUN_META_JSON_HOST=/{print $2}' "${RUN_META_FILE}" || true)

    if [ -z "${manifest_host}" ] && [ -n "${manifest_container}" ]; then
      manifest_host="${HOST_MANIFEST_ROOT}/$(basename "${manifest_container}")"
    fi

    if [ -n "${manifest_host}" ] && [ -f "${manifest_host}" ]; then
      warnings_output=""
      if command -v python3 >/dev/null 2>&1; then
        warnings_output=$(MANIFEST_PATH="${manifest_host}" HOST_STORAGE_ROOT="${HOST_STORAGE_ROOT}" REPO_ROOT="${ROOT_DIR}" python3 - <<'PY'
import json
import os
from pathlib import Path, PurePosixPath

manifest_path = Path(os.environ["MANIFEST_PATH"])
host_storage_root = Path(os.environ["HOST_STORAGE_ROOT"]).resolve()
repo_root = Path(os.environ["REPO_ROOT"]).resolve()

with manifest_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

storage = data.get("storage", {})
exporter_state = data.get("exporters_state", {})

telemetry_surfaces = data.setdefault("telemetry_surfaces", {})
telemetry_surfaces.setdefault(
    "sglang_metrics",
    {
        "status": "expected",
        "details": "Prometheus scrape at localhost:30000 once SGLang server launches with --enable-metrics",
    },
)
telemetry_surfaces.setdefault(
    "tracing",
    {
        "status": "expected",
        "details": "OTLP gRPC :4317, HTTP :4318 (emitted when --enable-trace is set)",
    },
)
telemetry_surfaces["node_metrics"] = {
    "status": exporter_state.get("node_exporter", "unknown"),
    "details": "node_exporter on localhost:9100",
}
telemetry_surfaces["dcgm_metrics"] = {
    "status": exporter_state.get("dcgm_exporter", "unknown"),
    "details": "dcgm-exporter on localhost:9400 (requires SYS_ADMIN capability)",
}

telemetry_root = PurePosixPath("/telemetry")
container_paths = {}
for key, value in storage.items():
    p = PurePosixPath(value)
    try:
        container_paths[key] = str(p.relative_to(telemetry_root))
    except ValueError:
        container_paths[key] = value
container_paths["storage_root"] = "telemetry"

storage_root_rel = PurePosixPath(os.path.relpath(host_storage_root, repo_root))

host_paths = {"storage_root": str(storage_root_rel)}

if "log_file" in storage:
    host_paths["log_file"] = str(storage_root_rel / "logs" / PurePosixPath(storage["log_file"]).name)
if "prometheus_dir" in storage:
    host_paths["prometheus_dir"] = str(storage_root_rel / "prometheus" / PurePosixPath(storage["prometheus_dir"]).name)
if "jaeger_dir" in storage:
    host_paths["jaeger_dir"] = str(storage_root_rel / "jaeger" / PurePosixPath(storage["jaeger_dir"]).name)
host_paths["profiles_root"] = str(storage_root_rel / "profiles")
host_paths["models_root"] = str(storage_root_rel / "models")
host_paths["huggingface_root"] = str(storage_root_rel / "huggingface")

data["paths"] = {
    "host": host_paths,
    "container": container_paths,
}

def directory_has_files(directory: Path) -> bool:
    for child in directory.rglob("*"):
        if child.is_file():
            return True
    return False


profiles_info = {}
profiles_root = host_storage_root / "profiles"
if profiles_root.exists():
    for entry in sorted(profiles_root.glob("*")):
        if not entry.is_dir():
            continue
        rel_repo = PurePosixPath(os.path.relpath(entry, repo_root))
        has_data = directory_has_files(entry)
        profiles_info[entry.name] = {
            "path": str(rel_repo),
            "has_cached_data": has_data,
        }

if profiles_info:
    data["profile_storage"] = profiles_info

storage_roots = {}
for name in ["models", "huggingface", "profiles"]:
    path = host_storage_root / name
    rel_repo = PurePosixPath(os.path.relpath(path, repo_root))
    storage_roots[name] = str(rel_repo)

data["storage_roots"] = storage_roots

with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

warnings = data.get("warnings") or []
for line in warnings:
    print(line)
PY
        ) || true
      fi

      if [ -n "${warnings_output}" ]; then
        echo "Observability warnings recorded in manifest:"
        while IFS= read -r warning_line; do
          [ -z "${warning_line}" ] && continue
          echo " - ${warning_line}"
        done <<EOF
${warnings_output}
EOF
      fi

      echo "Observability container ready after ${elapsed}s."

      if [ -n "${manifest_host}" ]; then
        echo "CONTAINER_RUN_META_JSON_HOST=${manifest_host}"
      fi
      if [ -n "${manifest_container}" ]; then
        echo "CONTAINER_RUN_META_JSON=${manifest_container}"
      fi
      break
    fi
  fi

  if [ $elapsed -ge 0 ] && [ $((elapsed / 5)) -gt $((last_heartbeat / 5)) ]; then
    echo "Container starting... (${elapsed}s elapsed)"
    last_heartbeat=$elapsed
  fi

  sleep 1
done
