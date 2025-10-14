#!/usr/bin/env bash
set -euo pipefail

RUN_META_FILE="${RUN_META_FILE:-/telemetry/container_run_meta.env}"
PROM_CONFIG_TEMPLATE="${PROM_CONFIG_TEMPLATE:-/opt/observability/prometheus.yml.tmpl}"

as_devuser() {
  sudo -u devuser -E "$@"
}


container_run_id="${CONTAINER_RUN_ID_OVERRIDE:-}"
if [ -z "${container_run_id}" ]; then
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  short_id=$(cat /proc/sys/kernel/random/uuid | cut -d- -f1)
  container_run_id="container-run-${timestamp}-${short_id}"
fi

RUN_ROOT_DEFAULT="/telemetry/container_runs/${container_run_id}"
RUN_ROOT="${RUN_ROOT_OVERRIDE:-${RUN_ROOT:-${RUN_ROOT_DEFAULT}}}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs}"
PROM_STORAGE_ROOT="${PROM_STORAGE_ROOT:-${RUN_ROOT}/prometheus}"
PROM_CONFIG_FILE="${PROM_CONFIG_FILE:-${RUN_ROOT}/configs/prometheus.yml}"
CONFIGS_DIR="${CONFIGS_DIR:-${PROM_CONFIG_FILE%/*}}"
JAEGER_STORAGE_ROOT="${JAEGER_STORAGE_ROOT:-${RUN_ROOT}/jaeger}"
RUN_MANIFEST_ROOT="${RUN_MANIFEST_ROOT:-${RUN_ROOT}}"
METRICS_DIR="${METRICS_DIR:-${RUN_ROOT}/metrics}"
EXPORTER_LOG_ROOT="${EXPORTER_LOG_ROOT:-${LOG_ROOT}/exporters}"

meta_dir="$(dirname "${RUN_META_FILE}")"
as_devuser install -d -m 770 "${meta_dir}"

as_devuser install -d -m 770 "${RUN_ROOT}"
as_devuser mkdir -p \
  "${LOG_ROOT}" \
  "${EXPORTER_LOG_ROOT}" \
  "${PROM_STORAGE_ROOT}" \
  "${RUN_MANIFEST_ROOT}" \
  "${METRICS_DIR}" \
  "${JAEGER_STORAGE_ROOT}" \
  "${CONFIGS_DIR}"

log_file="${LOG_ROOT}/observability.log"
as_devuser touch "${log_file}"
as_devuser chmod 600 "${log_file}"

prom_storage_dir="${PROM_STORAGE_ROOT}"
jaeger_run_root="${JAEGER_STORAGE_ROOT}"

manifest_json="${RUN_MANIFEST_ROOT}/manifest.json"

host_run_dir="${HOST_RUN_DIR:-}"
host_manifest_root="${HOST_MANIFEST_ROOT:-}"
if [ -n "${host_run_dir}" ]; then
  host_manifest_path="${host_run_dir}/manifest.json"
else
  host_manifest_path="${host_manifest_root}/${container_run_id}.json"
fi

host_telemetry_root="${HOST_TELEMETRY_ROOT:-}"
host_profiles_root="${HOST_PROFILES_ROOT:-}"
if [ -z "${host_run_dir}" ] && ([ -z "${host_telemetry_root}" ] || [ -z "${host_profiles_root}" ]); then
  warn "HOST telem/profiles roots not set; host paths in manifest may be unusable"
fi

if [ -n "${host_run_dir}" ]; then
  host_prometheus_path="${host_run_dir}/prometheus"
  host_jaeger_path="${host_run_dir}/jaeger"
  host_log_path="${host_run_dir}/logs/observability.log"
  host_metrics_path="${host_run_dir}/metrics"
  host_configs_path="${host_run_dir}/configs"
else
  host_prometheus_path="${host_telemetry_root:+${host_telemetry_root}/prometheus/${container_run_id}}"
  host_jaeger_path="${host_telemetry_root:+${host_telemetry_root}/jaeger/${container_run_id}}"
  host_log_path="${host_telemetry_root:+${host_telemetry_root}/logs/${container_run_id}.log}"
  host_metrics_path="${host_telemetry_root:+${host_telemetry_root}/metrics}"
  host_configs_path=""
fi

if [ -n "${host_run_dir}" ]; then
  host_exporter_logs_path="${host_run_dir}/logs/exporters"
elif [ -n "${host_telemetry_root}" ]; then
  host_exporter_logs_path="${host_telemetry_root}/logs/exporters"
else
  host_exporter_logs_path=""
fi

host_profiles_path="${host_profiles_root}"

if [ -n "${host_run_dir}" ]; then
  host_run_dir_effective="${host_run_dir}"
else
  host_run_dir_effective="${host_telemetry_root}"
fi

{
  echo "CONTAINER_RUN_META_JSON=${manifest_json}"
  if [ -n "${host_manifest_path}" ]; then
    echo "CONTAINER_RUN_META_JSON_HOST=${host_manifest_path}"
  fi
} | as_devuser tee "${RUN_META_FILE}" >/dev/null
as_devuser chmod 600 "${RUN_META_FILE}"

export CONTAINER_RUN_ID="${container_run_id}"
export CONTAINER_RUN_META_JSON="${manifest_json}"
export RUN_ROOT
export METRICS_DIR
if [ -n "${host_manifest_path}" ]; then
  export CONTAINER_RUN_META_JSON_HOST="${host_manifest_path}"
fi

exec > >(tee -a "${log_file}") 2>&1

echo "Initialized container run: ${container_run_id}"

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


# Emit container_started event (best-effort)
bash /workspaces/sglang/.devcontainer/observability/eventlog.sh event container_started run_id="${container_run_id}" || true
trap 'bash /workspaces/sglang/.devcontainer/observability/eventlog.sh event container_stopped run_id="'"${container_run_id}"'" || true' EXIT

eval "${PROMETHEUS_EXTRA_ENV:-true}" >/dev/null 2>&1 || true

ensure_dcgm_capability

if [ -f "${PROM_CONFIG_TEMPLATE}" ]; then
  CONTAINER_RUN_ID="${container_run_id}" envsubst < "${PROM_CONFIG_TEMPLATE}" > "${PROM_CONFIG_FILE}"
fi

prometheus \
  --config.file="${PROM_CONFIG_FILE}" \
  --storage.tsdb.path="${prom_storage_dir}" \
  --storage.tsdb.retention.time="${PROMETHEUS_RETENTION:-30d}" \
  --web.listen-address="0.0.0.0:9090" &
prometheus_pid=$!

# Start DCGM hostengine and rely on dcgm-exporter to load profiling modules
existing_hostengine=$(pgrep -x nv-hostengine || true)
if [ -n "${existing_hostengine}" ]; then
  warn "nv-hostengine already running (pids: ${existing_hostengine})"
fi

# Ensure exporters logs directory exists under the run telemetry directory
mkdir -p "${EXPORTER_LOG_ROOT}" || true

# Persist hostengine logs under the run telemetry directory
nv-hostengine --pid /tmp/nv-hostengine.pid --log-level ERROR -f "${EXPORTER_LOG_ROOT}/nv-hostengine.log" || warn "nv-hostengine failed to start; dcgm metrics may be degraded"

# Exporters
node_exporter --web.listen-address=":9100" --collector.textfile.directory="${METRICS_DIR}" >>"${EXPORTER_LOG_ROOT}/node-exporter.log" 2>&1 &
node_exporter_pid=$!
dcgm_collectors="${DCGM_EXPORTER_COLLECTORS:-/etc/dcgm-exporter/metrics.csv}"
dcgm-exporter --collectors "${dcgm_collectors}" --address :9400 >>"${EXPORTER_LOG_ROOT}/dcgm-exporter.log" 2>&1 &
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

cat <<EOF | as_devuser tee "${manifest_json}" >/dev/null
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
  "paths": {
    "container": {
      "run_dir": "${RUN_ROOT}",
      "telemetry_root": "${RUN_ROOT}",
      "prometheus_dir": "${prom_storage_dir}",
      "jaeger_dir": "${jaeger_run_root}",
      "metrics_dir": "${METRICS_DIR}",
      "exporter_logs_dir": "${EXPORTER_LOG_ROOT}",
      "configs_dir": "${CONFIGS_DIR}",
      "log_file": "${log_file}",
      "profiles_root": "/profiles"
    },
    "host": {
      "run_dir": "${host_run_dir_effective}",
      "telemetry_root": "${host_run_dir_effective}",
      "prometheus_dir": "${host_prometheus_path}",
      "jaeger_dir": "${host_jaeger_path}",
      "metrics_dir": "${host_metrics_path}",
      "exporter_logs_dir": "${host_exporter_logs_path}",
      "configs_dir": "${host_configs_path}",
      "log_file": "${host_log_path}",
      "profiles_root": "${host_profiles_path}"
    }
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
    "prometheus_config": "${PROM_CONFIG_FILE}",
    "jaeger_dir": "${jaeger_run_root}",
    "jaeger_config": "${jaeger_run_root}/config.yaml",
    "metrics_dir": "${METRICS_DIR}",
    "exporter_logs_dir": "${EXPORTER_LOG_ROOT}"
  },
  "configs": {
    "prometheus": "${PROM_CONFIG_FILE}",
    "jaeger": "${jaeger_run_root}/config.yaml"
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

echo "CONTAINER_RUN_META_JSON=${manifest_json}"
if [ -n "${host_manifest_path}" ]; then
  echo "CONTAINER_RUN_META_JSON_HOST=${host_manifest_path}"
fi

echo "Background services started: Prometheus, node_exporter, dcgm-exporter (Jaeger v2 runs on host ports)"

export PYTHONPATH="/workspaces/sglang/python${PYTHONPATH:+:${PYTHONPATH}}"

if [ -n "${INIT_RUN_HOOK:-}" ]; then
  as_devuser bash -lc "${INIT_RUN_HOOK}"
fi

# Ensure hostengine is stopped when container exits and clear pointer
cleanup() {
  pkill -TERM -P $$ 2>/dev/null || true
  if [ -f /tmp/nv-hostengine.pid ]; then
    pkill -F /tmp/nv-hostengine.pid 2>/dev/null || true
    rm -f /tmp/nv-hostengine.pid
  fi
  as_devuser rm -f "${RUN_META_FILE}"
}
trap cleanup EXIT

as_devuser "$@" &
child_pid=$!
wait "$child_pid"
