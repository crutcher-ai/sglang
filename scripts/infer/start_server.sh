#!/usr/bin/env bash
set -euo pipefail

# Starts sglang.launch_server inside the running container as devuser.
# Emits a small JSON with run_id, health, port, manifest_host_path, log_file, started_at_iso.

CONTAINER_NAME="${CONTAINER_NAME:-sglang-dev}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_OBS_ROOT="${HOST_OBS_ROOT:-$HOME/sglang-observability}"
RUN_META_FILE="${HOST_OBS_ROOT}/telemetry/container_run_meta.env"
HOST_PORT="${PORT:-30000}"
TP_SIZE="${TP:-1}"

# Optional overrides (otherwise read from sglang-config.json in container)
MODEL_PATH="${MODEL_PATH:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-}"
# Optional tracing toggles
ENABLE_TRACE="${ENABLE_TRACE:-}"
OLTP_TRACES_ENDPOINT="${OLTP_TRACES_ENDPOINT:-}"

now_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }
die() { echo "ERROR: $*" >&2; exit 1; }

[ -f "$RUN_META_FILE" ] || die "manifest pointer not found: $RUN_META_FILE (start the container first)"

MANIFEST_HOST=$(awk -F= '/^CONTAINER_RUN_META_JSON_HOST=/{print $2}' "$RUN_META_FILE" || true)
[ -n "$MANIFEST_HOST" ] || die "host manifest path missing in pointer"
[ -f "$MANIFEST_HOST" ] || die "host manifest not found: $MANIFEST_HOST"

RUN_ID=$(python3 - "$MANIFEST_HOST" <<'PY'
import json,sys
p=sys.argv[1]
d=json.load(open(p))
print((d.get('run') or {}).get('container_run_id') or d.get('container_run_id') or '')
PY)
[ -n "$RUN_ID" ] || die "container_run_id missing in manifest"

LOG_FILE=$(python3 - "$MANIFEST_HOST" <<'PY'
import json,sys
p=sys.argv[1]
d=json.load(open(p))
print(((d.get('storage') or {}).get('log_file')) or '')
PY)
[ -n "$LOG_FILE" ] || die "log file missing in manifest"

# If healthy already, print status and exit
if "$SCRIPT_DIR/status.sh" | grep -qx ready; then
  cat <<EOF
{
  "schema_version": 1,
  "run_id": "$RUN_ID",
  "health": "ready",
  "port": $HOST_PORT,
  "manifest_host_path": "$MANIFEST_HOST",
  "log_file": "$LOG_FILE",
  "started_at_iso": "$(now_iso)"
}
EOF
  exit 0
fi

# Build server flags by reading the config inside the container unless overrides are provided
read_cfg_py='python - <<PY
import json, os
cfg = json.load(open("/workspaces/sglang/.devcontainer/tools/sglang-config.json"))
model = os.environ.get("MODEL_PATH") or (cfg.get("model") or {}).get("default_model_path")
server_cfg = cfg.get("server") or {}
print(json.dumps({
  "model": model,
  "kv": os.environ.get("KV_CACHE_DTYPE") or (cfg.get("model") or {}).get("kv_cache_dtype"),
  "mem": os.environ.get("MEM_FRACTION_STATIC") or server_cfg.get("mem_fraction_static"),
  "chunk": os.environ.get("CHUNKED_PREFILL_SIZE") or server_cfg.get("chunked_prefill_size"),
  "ctx": os.environ.get("CONTEXT_LENGTH") or server_cfg.get("context_length"),
  "maxp": os.environ.get("MAX_PREFILL_TOKENS") or server_cfg.get("max_prefill_tokens"),
  "maxt": os.environ.get("MAX_TOTAL_TOKENS") or server_cfg.get("max_total_tokens"),
  "mamba": os.environ.get("MAX_MAMBA_CACHE_SIZE") or server_cfg.get("max_mamba_cache_size"),
  "trace": os.environ.get("ENABLE_TRACE") or server_cfg.get("enable_trace"),
  "otlp": os.environ.get("OLTP_TRACES_ENDPOINT") or server_cfg.get("oltp_traces_endpoint"),
}))
PY'

read_cfg() {
  docker exec -u devuser \
    -e MODEL_PATH="$MODEL_PATH" \
    -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE" \
    -e MEM_FRACTION_STATIC="$MEM_FRACTION_STATIC" \
    -e CHUNKED_PREFILL_SIZE="$CHUNKED_PREFILL_SIZE" \
    -e CONTEXT_LENGTH="$CONTEXT_LENGTH" \
    -e MAX_PREFILL_TOKENS="$MAX_PREFILL_TOKENS" \
    -e MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS" \
    -e MAX_MAMBA_CACHE_SIZE="$MAX_MAMBA_CACHE_SIZE" \
    -e ENABLE_TRACE="$ENABLE_TRACE" \
    -e OLTP_TRACES_ENDPOINT="$OLTP_TRACES_ENDPOINT" \
    "$CONTAINER_NAME" bash -lc "$read_cfg_py" 2>/dev/null || echo '{}'
}

cfg_json=$(MODEL_PATH="$MODEL_PATH" KV_CACHE_DTYPE="$KV_CACHE_DTYPE" MEM_FRACTION_STATIC="$MEM_FRACTION_STATIC" CHUNKED_PREFILL_SIZE="$CHUNKED_PREFILL_SIZE" CONTEXT_LENGTH="$CONTEXT_LENGTH" MAX_PREFILL_TOKENS="$MAX_PREFILL_TOKENS" MAX_TOTAL_TOKENS="$MAX_TOTAL_TOKENS" MAX_MAMBA_CACHE_SIZE="$MAX_MAMBA_CACHE_SIZE" ENABLE_TRACE="$ENABLE_TRACE" OLTP_TRACES_ENDPOINT="$OLTP_TRACES_ENDPOINT" read_cfg)

val() { python3 - "$@" << 'PY'
import json,sys
print((json.loads(sys.argv[1]).get(sys.argv[2]) or ""))
PY
}

MODEL=$(val "$cfg_json" model)
[ -n "$MODEL" ] || die "MODEL_PATH not provided and default not found"
KV=$(val "$cfg_json" kv)
MEM=$(val "$cfg_json" mem)
CHUNK=$(val "$cfg_json" chunk)
CTX=$(val "$cfg_json" ctx)
MAXP=$(val "$cfg_json" maxp)
MAXT=$(val "$cfg_json" maxt)
MAMBA=$(val "$cfg_json" mamba)
TRACE=$(val "$cfg_json" trace)
OTLP=$(val "$cfg_json" otlp)

# Start server in the container (avoid Bash @Q quoting; pass env vars instead)
docker exec -u devuser \
  -e MODEL="$MODEL" \
  -e MEM="$MEM" \
  -e KV="$KV" \
  -e CHUNK="$CHUNK" \
  -e CTX="$CTX" \
  -e MAXP="$MAXP" \
  -e MAXT="$MAXT" \
  -e MAMBA="$MAMBA" \
  -e TRACE="$TRACE" \
  -e OTLP="$OTLP" \
  -e LOG_FILE="$LOG_FILE" \
  "$CONTAINER_NAME" bash -lc "\
  nohup python -m sglang.launch_server \\
    --model-path \"\$MODEL\" \\
    --host 0.0.0.0 --port $HOST_PORT \\
    --tp-size $TP_SIZE \\
    \${MEM:+--mem-fraction-static \"\$MEM\"} \\
    \${KV:+--kv-cache-dtype \"\$KV\"} \\
    \${CHUNK:+--chunked-prefill-size \"\$CHUNK\"} \\
    \${CTX:+--context-length \"\$CTX\"} \\
    \${MAXP:+--max-prefill-tokens \"\$MAXP\"} \\
    \${MAXT:+--max-total-tokens \"\$MAXT\"} \\
    \${MAMBA:+--max-mamba-cache-size \"\$MAMBA\"} \\
    \${TRACE:+--enable-trace} \\
    \${OTLP:+--oltp-traces-endpoint \"\$OTLP\"} \\
    --enable-metrics --trust-remote-code \\
    >> \"\$LOG_FILE\" 2>&1 & disown" >/dev/null

# Poll readiness (configurable, default 180s)
READY_TIMEOUT=${READY_TIMEOUT:-180}
deadline=$((SECONDS+READY_TIMEOUT))
while [ $SECONDS -lt $deadline ]; do
  if "$SCRIPT_DIR/status.sh" | grep -qx ready; then
    cat <<EOF
{
  "schema_version": 1,
  "run_id": "$RUN_ID",
  "health": "ready",
  "port": $HOST_PORT,
  "manifest_host_path": "$MANIFEST_HOST",
  "log_file": "$LOG_FILE",
  "started_at_iso": "$(now_iso)"
}
EOF
    exit 0
  fi
  sleep 1
done

die "server did not become ready on port $HOST_PORT"
