#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/telemetry/logs}"
RUN_META_FILE="${RUN_META_FILE:-/telemetry/container_run_meta.env}"

mkdir -p "${LOG_ROOT}"

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
short_id=$(cat /proc/sys/kernel/random/uuid | cut -d- -f1)
container_run_id="container-run-${timestamp}-${short_id}"
log_file="${LOG_ROOT}/${container_run_id}.log"

touch "${log_file}"
chmod 600 "${log_file}"

exec > >(tee -a "${log_file}") 2>&1

{
  echo "CONTAINER_RUN_ID=${container_run_id}"
  echo "CONTAINER_STARTED_AT=${timestamp}"
  echo "CONTAINER_LOG_FILE=${log_file}"
} > "${RUN_META_FILE}"

chmod 600 "${RUN_META_FILE}"

echo "Initialized container run: ${container_run_id} (metadata written to ${RUN_META_FILE})"

export CONTAINER_RUN_ID="${container_run_id}"
exec "$@"
