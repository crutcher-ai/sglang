#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/telemetry/logs}"
RUN_META_FILE="${RUN_META_FILE:-/telemetry/container_run_meta.env}"
PROM_STORAGE_ROOT="${PROM_STORAGE_ROOT:-/telemetry/prometheus}"
JAEGER_STORAGE_ROOT="${JAEGER_STORAGE_ROOT:-/telemetry/jaeger}"
PROM_CONFIG_TEMPLATE="${PROM_CONFIG_TEMPLATE:-/opt/observability/prometheus.yml.tmpl}"
PROM_CONFIG_FILE="${PROM_CONFIG_FILE:-/etc/prometheus/prometheus.yml}"

mkdir -p "${LOG_ROOT}" "${PROM_STORAGE_ROOT}" "${JAEGER_STORAGE_ROOT}"

ensure_dcgm_capability() {
  local exporter_bin="${DCGM_EXPORTER_BIN:-/usr/local/bin/dcgm-exporter}"

  if [ ! -x "${exporter_bin}" ]; then
    echo "dcgm-exporter binary not found at ${exporter_bin}; skipping capability grant"
    return
  fi

  if ! command -v setcap >/dev/null 2>&1; then
    echo "Warning: setcap not available inside container; ensure dcgm-exporter can access CAP_SYS_ADMIN"
    return
  fi

  if setcap 'cap_sys_admin=+ep' "${exporter_bin}" 2>/dev/null; then
    if "${exporter_bin}" -v >/dev/null 2>&1; then
      echo "dcgm-exporter granted cap_sys_admin for profiling metrics"
    else
      echo "Warning: dcgm-exporter capability self-check failed; profiling metrics may be unavailable"
      setcap 'cap_sys_admin=-ep' "${exporter_bin}" 2>/dev/null || true
    fi
  else
    echo "Warning: unable to set cap_sys_admin on ${exporter_bin}; start container with --cap-add SYS_ADMIN"
  fi
}

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
short_id=$(cat /proc/sys/kernel/random/uuid | cut -d- -f1)
container_run_id="container-run-${timestamp}-${short_id}"
log_file="${LOG_ROOT}/${container_run_id}.log"

prom_storage_dir="${PROM_STORAGE_ROOT}/${container_run_id}"
jaeger_run_root="${JAEGER_STORAGE_ROOT}/${container_run_id}"
jaeger_key_dir="${jaeger_run_root}/badger/key"
jaeger_value_dir="${jaeger_run_root}/badger/data"

mkdir -p "${prom_storage_dir}" "${jaeger_key_dir}" "${jaeger_value_dir}" "$(dirname "${PROM_CONFIG_FILE}")"

touch "${log_file}"
chmod 600 "${log_file}"

run_uid="${RUN_FILE_UID:-}"
run_gid="${RUN_FILE_GID:-}"
if [ -z "${run_uid}" ] || [ -z "${run_gid}" ]; then
  if id devuser >/dev/null 2>&1; then
    run_uid="$(id -u devuser)"
    run_gid="$(id -g devuser)"
  else
    run_uid="0"
    run_gid="0"
  fi
fi

chown "${run_uid}:${run_gid}" "${log_file}"
chown -R "${run_uid}:${run_gid}" "${prom_storage_dir}" "${jaeger_run_root}"
meta_dir="$(dirname "${RUN_META_FILE}")"
install -d -m 770 "${meta_dir}"
chown "${run_uid}:${run_gid}" "${meta_dir}"

{
  echo "CONTAINER_RUN_ID=${container_run_id}"
  echo "CONTAINER_STARTED_AT=${timestamp}"
  echo "CONTAINER_LOG_FILE=${log_file}"
  echo "PROMETHEUS_TSDB=${prom_storage_dir}"
  echo "JAEGER_STORAGE_ROOT=${jaeger_run_root}"
} > "${RUN_META_FILE}"
chmod 600 "${RUN_META_FILE}"
chown "${run_uid}:${run_gid}" "${RUN_META_FILE}"

export CONTAINER_RUN_ID="${container_run_id}"

exec > >(tee -a "${log_file}") 2>&1

echo "Initialized container run: ${container_run_id}"

eval "${PROMETHEUS_EXTRA_ENV:-true}" >/dev/null 2>&1 || true

ensure_dcgm_capability

export SPAN_STORAGE_TYPE=badger
export BADGER_EPHEMERAL=false
export BADGER_DIRECTORY_KEY="${jaeger_key_dir}"
export BADGER_DIRECTORY_VALUE="${jaeger_value_dir}"

if [ -f "${PROM_CONFIG_TEMPLATE}" ]; then
  CONTAINER_RUN_ID="${container_run_id}" envsubst < "${PROM_CONFIG_TEMPLATE}" > "${PROM_CONFIG_FILE}"
fi

prometheus \
  --config.file="${PROM_CONFIG_FILE}" \
  --storage.tsdb.path="${prom_storage_dir}" \
  --storage.tsdb.retention.time="${PROMETHEUS_RETENTION:-30d}" \
  --web.listen-address="0.0.0.0:9090" &

jaeger-all-in-one \
  --collector.otlp.enabled=true \
  --collector.otlp.grpc.host-port=":4317" \
  --collector.otlp.http.host-port=":4318" \
  --admin.http.host-port=":14269" &

# Start DCGM hostengine and rely on dcgm-exporter to load profiling modules
nv-hostengine --pid /tmp/nv-hostengine.pid --log-level ERROR -f /tmp/nv-hostengine.log || true

# Exporters
node_exporter --web.listen-address=":9100" &
dcgm_collectors="${DCGM_EXPORTER_COLLECTORS:-/etc/dcgm-exporter/metrics.csv}"
dcgm-exporter --collectors "${dcgm_collectors}" --address :9400 &

echo "Background services started: Prometheus, Jaeger, node_exporter, dcgm-exporter"

eval "${INIT_RUN_HOOK:-true}"

# Ensure hostengine is stopped when container exits
trap 'pkill -TERM -P $$; if [ -f /tmp/nv-hostengine.pid ]; then pkill -F /tmp/nv-hostengine.pid 2>/dev/null || true; rm -f /tmp/nv-hostengine.pid; fi' EXIT

exec "$@"
