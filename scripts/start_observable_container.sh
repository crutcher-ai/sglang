#!/usr/bin/env bash
set -euo pipefail

# Location of this script and repo root
# Simple help and arg handling
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'EOF'
Usage: start_observable_container.sh

Starts the sglang observability helper container, waits for the per-run manifest
to become available, enriches it (paths, cache snapshot), and prints both host-
side and container-side manifest paths. The script exits non-zero if the
container fails to launch.

Environment variables:
  HOST_OBS_ROOT        Parent root for all observability data (default: $HOME/sglang-observability).
  HOST_TELEMETRY_ROOT  Override path for telemetry (default: $HOST_OBS_ROOT/telemetry).
  HOST_PROFILES_ROOT   Override path for kernel caches (default: $HOST_OBS_ROOT/profiles).
  HOST_MODELS_ROOT     Override path for models (default: $HOST_OBS_ROOT/models).
  HOST_HF_ROOT         Override path for HuggingFace cache (default: $HOST_OBS_ROOT/huggingface).
  CONTAINER_NAME       Container name (default: sglang-dev)
  IMAGE_NAME           Image name (default: sglang-dev:gh200)

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

# Determine host roots under a single parent in HOME by default
HOST_OBS_ROOT="${HOST_OBS_ROOT:-$HOME/sglang-observability}"
HOST_TELEMETRY_ROOT="${HOST_TELEMETRY_ROOT:-${HOST_OBS_ROOT}/telemetry}"
HOST_PROFILES_ROOT="${HOST_PROFILES_ROOT:-${HOST_OBS_ROOT}/profiles}"
HOST_MODELS_ROOT="${HOST_MODELS_ROOT:-${HOST_OBS_ROOT}/models}"

HOST_MANIFEST_ROOT="${HOST_TELEMETRY_ROOT}/container_runs"
RUN_META_FILE="${HOST_TELEMETRY_ROOT}/container_run_meta.env"

CONTAINER_NAME="${CONTAINER_NAME:-sglang-dev}"
IMAGE_NAME="${IMAGE_NAME:-sglang-dev:gh200}"

# Testing hooks:
#  - VALIDATE_ONLY=1: run preflight + directory setup and exit 0 before any Docker calls
#  - EXPECT_OWNER_UID/EXPECT_OWNER_GID: override expected ownership in preflight checks
EXPECT_UID="${EXPECT_OWNER_UID:-$(id -u)}"
EXPECT_GID="${EXPECT_OWNER_GID:-$(id -g)}"
# Allow different expected owner for specific files (used by tests)
EXPECT_POINTER_UID="${EXPECT_POINTER_OWNER_UID:-${EXPECT_UID}}"
EXPECT_POINTER_GID="${EXPECT_POINTER_OWNER_GID:-${EXPECT_GID}}"

# In validate-only mode, we skip all Docker interactions entirely.
if [[ "${VALIDATE_ONLY:-0}" != "1" ]]; then
  existing_cid=$(docker ps -aq -f name="^${CONTAINER_NAME}$" || true)
  if [[ -n "${existing_cid}" ]]; then
    echo "Removing existing container ${CONTAINER_NAME} (${existing_cid})"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
fi

# Prepare host directories
mkdir -p \
  "${HOST_TELEMETRY_ROOT}/logs" \
  "${HOST_TELEMETRY_ROOT}/prometheus" \
  "${HOST_TELEMETRY_ROOT}/jaeger" \
  "${HOST_MANIFEST_ROOT}" \
  "${HOST_PROFILES_ROOT}" \
  "${HOST_PROFILES_ROOT}/triton" \
  "${HOST_PROFILES_ROOT}/torchinductor" \
  "${HOST_PROFILES_ROOT}/flashinfer" \
  "${HOST_PROFILES_ROOT}/deep_gemm" \
  "${HOST_PROFILES_ROOT}/moe_configs/configs" \
  "${HOST_PROFILES_ROOT}/.locks" \
  "${HOST_PROFILES_ROOT}/.in_progress" \
  "${HOST_MODELS_ROOT}"

# Optional preflight: ensure host dirs (and key files, if present) are owned by the expected user/group
if [[ "${HOST_DIR_OWNERSHIP_IGNORE:-}" != "1" ]]; then
  check_own() {
    local p="$1"; [[ -e "$p" ]] || return 0
    local ug
    ug=$(stat -c '%u:%g' "$p" 2>/dev/null || echo '')
    if [[ "$ug" != "${EXPECT_UID}:${EXPECT_GID}" ]]; then
      echo "ERROR: $p is owned by $ug, not ${EXPECT_UID}:${EXPECT_GID}. Fix with chown -R ${EXPECT_UID}:${EXPECT_GID} '$p' or set HOST_DIR_OWNERSHIP_IGNORE=1 to proceed." >&2
      exit 3
    fi
  }
  check_own_file() {
    local p="$1"; [[ -e "$p" ]] || return 0
    local ug
    ug=$(stat -c '%u:%g' "$p" 2>/dev/null || echo '')
    if [[ "$ug" != "${EXPECT_POINTER_UID}:${EXPECT_POINTER_GID}" ]]; then
      echo "ERROR: $p is owned by $ug, not ${EXPECT_POINTER_UID}:${EXPECT_POINTER_GID}. Fix with chown ${EXPECT_POINTER_UID}:${EXPECT_POINTER_GID} '$p' or remove it." >&2
      exit 3
    fi
  }
  check_own "${HOST_TELEMETRY_ROOT}"; check_own "${HOST_PROFILES_ROOT}"; check_own "${HOST_MODELS_ROOT}"
  # If a stale pointer file exists, validate its ownership too.
  if [[ -e "${RUN_META_FILE}" ]]; then
    check_own_file "${RUN_META_FILE}"
  fi
fi

# Remove stale pointer to avoid reading an old manifest
if [[ -f "${RUN_META_FILE}" ]]; then
  rm -f "${RUN_META_FILE}" || true
fi

# In validate-only mode, stop here after preflight and directory initialization.
if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "VALIDATE_ONLY: preflight passed (no container launched)"
  exit 0
fi

INIT_HOOK="/workspaces/sglang/.devcontainer/post-create.sh"

container_id=$(docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --shm-size 32g \
  --ipc host \
  --network host \
  --add-host host.docker.internal:host-gateway \
  --cap-add SYS_ADMIN \
  -e INIT_RUN_HOOK="${INIT_HOOK}" \
  -e HOST_TELEMETRY_ROOT="${HOST_TELEMETRY_ROOT}" \
  -e HOST_PROFILES_ROOT="${HOST_PROFILES_ROOT}" \
  -e HOST_MODELS_ROOT="${HOST_MODELS_ROOT}" \
  -e HOST_MANIFEST_ROOT="${HOST_MANIFEST_ROOT}" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -e RUN_FILE_UID="$(id -u)" -e RUN_FILE_GID="$(id -g)" \
  --entrypoint /bin/bash \
  -v "${HOST_TELEMETRY_ROOT}:/telemetry" \
  -v "${HOST_PROFILES_ROOT}:/profiles" \
  -v "${HOST_MODELS_ROOT}:/models" \
  -v "${ROOT_DIR}:/workspaces/sglang" \
  -w /workspaces/sglang \
  "${IMAGE_NAME}" \
  -lc 'set -e; \
        if id devuser >/dev/null 2>&1; then \
          if [ -n "${HOST_GID:-}" ]; then groupmod -g "${HOST_GID}" devuser 2>/dev/null || true; fi; \
          if [ -n "${HOST_UID:-}" ]; then usermod -u "${HOST_UID}" devuser 2>/dev/null || true; fi; \
          chown -R devuser:devuser /home/devuser 2>/dev/null || true; \
        fi; \
        exec /workspaces/sglang/.devcontainer/observability/init-run.sh sleep infinity')

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

  if [ -e "${RUN_META_FILE}" ]; then
    manifest_container=$(awk -F= '/^CONTAINER_RUN_META_JSON=/{print $2}' "${RUN_META_FILE}" || true)
    manifest_host=$(awk -F= '/^CONTAINER_RUN_META_JSON_HOST=/{print $2}' "${RUN_META_FILE}" || true)

    if [ -z "${manifest_host}" ] && [ -n "${manifest_container}" ]; then
      manifest_host="${HOST_MANIFEST_ROOT}/$(basename "${manifest_container}")"
    fi

    if [ -n "${manifest_host}" ] && [ -f "${manifest_host}" ]; then
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
