#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/telemetry/logs}"
RUN_META_FILE="${RUN_META_FILE:-/telemetry/container_run_meta.env}"
PROM_STORAGE_ROOT="${PROM_STORAGE_ROOT:-/telemetry/prometheus}"
JAEGER_STORAGE_ROOT="${JAEGER_STORAGE_ROOT:-/telemetry/jaeger}"
RUN_MANIFEST_ROOT="${RUN_MANIFEST_ROOT:-/telemetry/container_runs}"
PROM_CONFIG_TEMPLATE="${PROM_CONFIG_TEMPLATE:-/opt/observability/prometheus.yml.tmpl}"
PROM_CONFIG_FILE="${PROM_CONFIG_FILE:-/etc/prometheus/prometheus.yml}"

mkdir -p "${LOG_ROOT}" "${PROM_STORAGE_ROOT}" "${JAEGER_STORAGE_ROOT}"

warnings=()

warn() {
  local message="$1"
  warnings+=("${message}")
  echo "WARNING: ${message}" >&2
}

ensure_dcgm_capability() {
  local exporter_bin="${DCGM_EXPORTER_BIN:-/usr/local/bin/dcgm-exporter}"

  if [ ! -x "${exporter_bin}" ]; then
    warn "dcgm-exporter binary not found at ${exporter_bin}; skipping capability grant"
    return
  fi

  if ! command -v setcap >/dev/null 2>&1; then
    warn "setcap not available; ensure dcgm-exporter has cap_sys_admin"
    return
  fi

  if setcap 'cap_sys_admin=+ep' "${exporter_bin}" 2>/dev/null; then
    if "${exporter_bin}" -v >/dev/null 2>&1; then
      echo "dcgm-exporter granted cap_sys_admin for profiling metrics"
    else
      warn "dcgm-exporter capability self-check failed; profiling metrics may be unavailable"
      setcap 'cap_sys_admin=-ep' "${exporter_bin}" 2>/dev/null || true
    fi
  else
    warn "unable to set cap_sys_admin on ${exporter_bin}; start container with --cap-add SYS_ADMIN"
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

mkdir -p "${RUN_MANIFEST_ROOT}"
chown "${run_uid}:${run_gid}" "${RUN_MANIFEST_ROOT}"
manifest_json="${RUN_MANIFEST_ROOT}/${container_run_id}.json"

host_manifest_root="${HOST_MANIFEST_ROOT:-}"
host_manifest_path=""
if [ -n "${host_manifest_root}" ]; then
  host_manifest_path="${host_manifest_root}/${container_run_id}.json"
fi

{
  echo "CONTAINER_RUN_META_JSON=${manifest_json}"
  if [ -n "${host_manifest_path}" ]; then
    echo "CONTAINER_RUN_META_JSON_HOST=${host_manifest_path}"
  fi
} > "${RUN_META_FILE}"
chmod 600 "${RUN_META_FILE}"
chown "${run_uid}:${run_gid}" "${RUN_META_FILE}"

export CONTAINER_RUN_ID="${container_run_id}"
export CONTAINER_RUN_META_JSON="${manifest_json}"
if [ -n "${host_manifest_path}" ]; then
  export CONTAINER_RUN_META_JSON_HOST="${host_manifest_path}"
fi

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
prometheus_pid=$!

jaeger-all-in-one \
  --collector.otlp.enabled=true \
  --collector.otlp.grpc.host-port=":4317" \
  --collector.otlp.http.host-port=":4318" \
  --admin.http.host-port=":14269" &
jaeger_pid=$!

# Start DCGM hostengine and rely on dcgm-exporter to load profiling modules
existing_hostengine=$(pgrep -x nv-hostengine || true)
if [ -n "${existing_hostengine}" ]; then
  warn "nv-hostengine already running (pids: ${existing_hostengine})"
fi

nv-hostengine --pid /tmp/nv-hostengine.pid --log-level ERROR -f /tmp/nv-hostengine.log || warn "nv-hostengine failed to start; dcgm metrics may be degraded"

# Exporters
node_exporter --web.listen-address=":9100" &
node_exporter_pid=$!
dcgm_collectors="${DCGM_EXPORTER_COLLECTORS:-/etc/dcgm-exporter/metrics.csv}"
dcgm-exporter --collectors "${dcgm_collectors}" --address :9400 &
dcgm_exporter_pid=$!

sleep 1

node_exporter_state="running"
dcgm_exporter_state="running"

if ! ps -p "${node_exporter_pid}" >/dev/null 2>&1; then
  node_exporter_state="failed"
  warn "node_exporter failed to start"
fi

if ! ps -p "${dcgm_exporter_pid}" >/dev/null 2>&1; then
  dcgm_exporter_state="failed"
  warn "dcgm-exporter failed to start"
fi

warnings_joined=""
if [ ${#warnings[@]} -gt 0 ]; then
  warnings_joined=$(printf '%s\x1e' "${warnings[@]}")
  warnings_joined=${warnings_joined%$'\x1e'}
fi

warnings_json=$(WARNINGS="${warnings_joined}" python3 - <<'PY'
import json, os
joined = os.environ.get("WARNINGS", "")
if joined:
    warnings = joined.split("\x1e")
else:
    warnings = []
print(json.dumps(warnings))
PY
)

git_revision=$(git -C /workspaces/sglang rev-parse HEAD 2>/dev/null || echo "unknown")
container_image="${CONTAINER_IMAGE:-unknown}"
start_iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cat > "${manifest_json}" <<EOF
{
  "container_run_id": "${container_run_id}",
  "started_at": "${start_iso}",
  "image": "${container_image}",
  "git_revision": "${git_revision}",
  "services": {
    "prometheus": {"port": 9090},
    "jaeger": {"ui": 16686, "otlp_grpc": 4317, "otlp_http": 4318},
    "node_exporter": {"port": 9100},
    "dcgm_exporter": {"port": 9400}
  },
  "telemetry_surfaces": {
    "sglang_metrics": {
      "status": "expected",
      "details": "Prometheus scrape at localhost:30000 once SGLang server launches with --enable-metrics"
    },
    "tracing": {
      "status": "expected",
      "details": "OTLP gRPC :4317, HTTP :4318 (emitted when --enable-trace is set)"
    },
    "node_metrics": {
      "status": "${node_exporter_state}",
      "details": "node_exporter on localhost:9100"
    },
    "dcgm_metrics": {
      "status": "${dcgm_exporter_state}",
      "details": "dcgm-exporter on localhost:9400 (requires SYS_ADMIN capability)"
    }
  },
  "storage": {
    "log_file": "${log_file}",
    "prometheus_dir": "${prom_storage_dir}",
    "jaeger_dir": "${jaeger_run_root}"
  },
  "exporters_state": {
    "node_exporter": "${node_exporter_state}",
    "dcgm_exporter": "${dcgm_exporter_state}"
  },
  "collectors": {
    "sglang_metrics": "all",
    "tracing": "all",
    "node_metrics": "all",
    "dcgm_metrics": "all"
  },
  "warnings": ${warnings_json}
}
EOF

chown "${run_uid}:${run_gid}" "${manifest_json}"

echo "CONTAINER_RUN_META_JSON=${manifest_json}"
if [ -n "${host_manifest_path}" ]; then
  echo "CONTAINER_RUN_META_JSON_HOST=${host_manifest_path}"
fi

echo "Background services started: Prometheus, Jaeger, node_exporter, dcgm-exporter"

eval "${INIT_RUN_HOOK:-true}"

# Ensure hostengine is stopped when container exits
trap 'pkill -TERM -P $$; if [ -f /tmp/nv-hostengine.pid ]; then pkill -F /tmp/nv-hostengine.pid 2>/dev/null || true; rm -f /tmp/nv-hostengine.pid; fi' EXIT

exec "$@"
