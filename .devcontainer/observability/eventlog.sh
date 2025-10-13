#!/usr/bin/env bash
set -euo pipefail

# Emit run-scoped Prometheus textfile metrics for events.
# Usage:
#   eventlog.sh event <name> key=value [key=value ...]

METRICS_DIR=${METRICS_DIR:-/telemetry/metrics}
mkdir -p "${METRICS_DIR}" || true

escape_label() {
  # Escape backslashes and double quotes per Prometheus textfile exposition.
  local s="$1"
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  printf '%s' "$s"
}

now_epoch() { date -u +%s; }

emit_event() {
  local name="$1"; shift || true
  local ts
  ts=$(now_epoch)

  local labels=("event=\"$(escape_label "$name")\"")
  local kv
  for kv in "$@"; do
    # accept key=value, ignore malformed
    if [[ "$kv" == *"="* ]]; then
      local k="${kv%%=*}"; local v="${kv#*=}"
      labels+=("$(escape_label "$k")=\"$(escape_label "$v")\"")
    fi
  done
  local old_ifs="$IFS"; IFS=,; local lbl="{${labels[*]}}"; IFS="$old_ifs"

  # Write atomically
  local fname="${METRICS_DIR}/event-${name}-$(date -u +%Y%m%dT%H%M%SZ)-$$.prom"
  local tmp="${fname}.tmp"
  {
    echo "sglang_event_time_seconds${lbl} ${ts}"
    echo "sglang_event_info${lbl} 1"
  } >"${tmp}"
  mv "${tmp}" "${fname}"
}

cmd=${1:-}
case "$cmd" in
  event)
    shift || true
    if [[ $# -lt 1 ]]; then
      echo "Usage: $0 event <name> [key=value ...]" >&2
      exit 2
    fi
    emit_event "$@"
    ;;
  *)
    echo "Unknown command: ${cmd:-<none>}" >&2
    echo "Usage: $0 event <name> [key=value ...]" >&2
    exit 2
    ;;
esac
