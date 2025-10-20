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
# Optional OTEL sampler (propagated to container if provided)
OTEL_TRACES_SAMPLER="${OTEL_TRACES_SAMPLER:-}"
# Optional debug toggle passed into containerized server
SGL_DEBUG="${SGL_DEBUG:-}"
PYTHON_BIN="${PYTHON_BIN:-/sgl-workspace/sglang/.venv/bin/python}"

now_iso() { date -u +%Y-%m-%dT%H:%M:%SZ; }
die() { echo "ERROR: $*" >&2; exit 1; }

emit_ready_json() {
  python3 - "$@" <<'PY'
import json, sys
import os
import tempfile
import io
import errno
(
    started_at,
    run_id,
    session_id,
    port,
    manifest_path,
    log_file,
    session_record_path,
    tp_size,
    mem,
    chunk,
    ctx,
    maxp,
    maxt,
    mamba,
) = sys.argv[1:]

def maybe_number(value: str):
    if not value:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value

out = {
    "schema_version": 1,
    "run_id": run_id,
    "server_session_id": session_id,
    "health": "ready",
    "port": int(port),
    "manifest_host_path": manifest_path,
    "log_file": log_file,
    "started_at_iso": started_at,
    "session_record_host_path": session_record_path,
    "sizing": {
        "tp_size": int(tp_size),
        "mem_fraction_static": maybe_number(mem),
        "chunked_prefill_size": maybe_number(chunk),
        "context_length": maybe_number(ctx),
        "max_prefill_tokens": maybe_number(maxp),
        "max_total_tokens": maybe_number(maxt),
        "max_mamba_cache_size": maybe_number(mamba),
    },
}
json_str = json.dumps(out)
print(json_str)
if session_record_path:
    from pathlib import Path
    path = Path(session_record_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: tmp file in same directory, fsync file then rename, fsync dir
    tmp = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix='.start.', suffix='.json.tmp', dir=str(path.parent))
        tmp = tmp_name
        with io.open(fd, 'w', encoding='utf-8', closefd=False) as fh:
            fh.write(json_str + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, str(path))
        # Fsync the directory to ensure rename is durable
        dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
PY
}

[ -f "$RUN_META_FILE" ] || die "manifest pointer not found: $RUN_META_FILE (start the container first)"

MANIFEST_HOST=$(awk -F= '/^CONTAINER_RUN_META_JSON_HOST=/{print $2}' "$RUN_META_FILE" || true)
[ -n "$MANIFEST_HOST" ] || die "host manifest path missing in pointer"
[ -f "$MANIFEST_HOST" ] || die "host manifest not found: $MANIFEST_HOST"

RUN_ID=$(python3 - "$MANIFEST_HOST" <<'PY'
import json,sys
p=sys.argv[1]
d=json.load(open(p))
print((d.get('run') or {}).get('container_run_id') or d.get('container_run_id') or '')
PY
)
[ -n "$RUN_ID" ] || die "container_run_id missing in manifest"

LOG_FILE=$(python3 - "$MANIFEST_HOST" <<'PY'
import json,sys
p=sys.argv[1]
d=json.load(open(p))
print(((d.get('storage') or {}).get('log_file')) or '')
PY
)
[ -n "$LOG_FILE" ] || die "log file missing in manifest"

HOST_LOG_FILE=$(python3 - "$MANIFEST_HOST" <<'PY'
import json,sys
p=sys.argv[1]
d=json.load(open(p))
paths = d.get('paths') or {}
host_paths = paths.get('host') if isinstance(paths.get('host'), dict) else {}
print(host_paths.get('log_file', ''))
PY
)

RUN_DIR_HOST=$(python3 - "$MANIFEST_HOST" <<'PY'
import json,sys
p=sys.argv[1]
d=json.load(open(p))
paths = d.get('paths') or {}
host_paths = paths.get('host') if isinstance(paths.get('host'), dict) else {}
print(host_paths.get('run_dir', ''))
PY
)

if [ -z "$RUN_DIR_HOST" ] && [ -n "$HOST_LOG_FILE" ]; then
  RUN_DIR_HOST="$(cd "$(dirname "$HOST_LOG_FILE")"/.. && pwd)"
fi

SERVER_SESSION_ID=$(python3 - <<'PY'
import uuid
print("srv-" + uuid.uuid4().hex)
PY
)

SESSION_TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
if [ -n "$RUN_DIR_HOST" ]; then
  SESSION_RECORD_HOST_PATH="${RUN_DIR_HOST%/}/logs/provider_sessions/${SESSION_TIMESTAMP}_${SERVER_SESSION_ID}/start.json"
else
  SESSION_RECORD_HOST_PATH=""
fi

if [ -n "$HOST_LOG_FILE" ]; then
  printf '[%s] server session start: id=%s tp=%s mem=%s chunk=%s ctx=%s maxp=%s maxt=%s max_mamba=%s session_record=%s\n' "$(now_iso)" "$SERVER_SESSION_ID" "$TP_SIZE" "${MEM:-}" "${CHUNK:-}" "${CTX:-}" "${MAXP:-}" "${MAXT:-}" "${MAMBA:-}" "$SESSION_RECORD_HOST_PATH" >> "$HOST_LOG_FILE" || true
fi

# If healthy already, print status and exit
if "$SCRIPT_DIR/status.sh" | grep -qx ready; then
  existing_session="unknown"
  if [ -n "$RUN_DIR_HOST" ]; then
    latest_start_json=$(ls -1 "$RUN_DIR_HOST"/logs/provider_sessions/*/start.json 2>/dev/null | sort | tail -n1 || true)
    if [ -n "$latest_start_json" ]; then
      candidate=$(python3 - "$latest_start_json" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    print(data.get("server_session_id", ""))
except Exception:
    pass
PY
)
      if [ -n "$candidate" ]; then existing_session="$candidate"; fi
    fi
  fi
  if [ "$existing_session" = "unknown" ] && [ -n "$HOST_LOG_FILE" ] && [ -f "$HOST_LOG_FILE" ]; then
    candidate=$(grep 'server session start:' "$HOST_LOG_FILE" | tail -n1 | sed -n 's/.*id=\([^ ]*\).*/\1/p')
    if [ -n "$candidate" ]; then existing_session="$candidate"; fi
  fi
  die "server already running (session=$existing_session). stop it first before starting a new session"
fi

# Build server flags by reading the config inside the container unless overrides are provided
read_cfg_py='"'"${PYTHON_BIN}"'" - <<PY
import json, os
cfg = json.load(open("/sgl-workspace/sglang/.devcontainer/tools/sglang-config.json"))
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
PY
'

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
# Fallback: read model path from host-side config if container read failed
if [ -z "$MODEL" ]; then
  HOST_CFG_PATH="${SCRIPT_DIR%/}/../../.devcontainer/tools/sglang-config.json"
  if [ -f "$HOST_CFG_PATH" ]; then
    MODEL=$(python3 - "$HOST_CFG_PATH" <<'PY'
import json, os, sys
p=sys.argv[1]
try:
    d=json.load(open(p))
    print(os.environ.get("MODEL_PATH") or (d.get("model") or {}).get("default_model_path") or "")
except Exception:
    print("")
PY
)
    if [ -n "$MODEL" ] && [ -n "$HOST_LOG_FILE" ]; then
      printf '[%s] fallback: using host config model path: %s\n' "$(now_iso)" "$MODEL" >> "$HOST_LOG_FILE" || true
    fi
  fi
fi
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
MODEL_SLUG="$(basename "$MODEL")"

if [ -z "$OTLP" ]; then
  OTLP="127.0.0.1:4317"
fi

# Resource attributes for tracing:
# - container_run identifies the helper container run (low cardinality)
# - service.instance.id is the authoritative session identifier for filtering
#   in Jaeger; do not also add a duplicate server_session_id resource attribute.
BASE_OTEL_ATTRS="container_run=$RUN_ID,service.instance.id=$SERVER_SESSION_ID"
if [ -n "${OTEL_RESOURCE_ATTRIBUTES:-}" ]; then
  OTEL_ATTRS="${OTEL_RESOURCE_ATTRIBUTES},${BASE_OTEL_ATTRS}"
else
  OTEL_ATTRS="$BASE_OTEL_ATTRS"
fi
# NOTE: upstream uses OLTP_TRACES_ENDPOINT spelling; keep matching their env contract
# even though local variable is OTLP (OpenTelemetry Protocol).

TRACE_STATE="disabled"
if [ -n "$TRACE" ]; then
  TRACE_STATE="enabled"
fi

# GPU preflight via venv interpreter (fail fast)
docker exec -u devuser "$CONTAINER_NAME" bash -lc "\
  set -e; "$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
    ok = bool(torch.cuda.is_available()) and (getattr(torch.version, 'cuda', None) == '12.9')
    if not ok:
        raise SystemExit(2)
except Exception as e:
    print('ERROR: CUDA preflight failed:', e, file=sys.stderr)
    raise SystemExit(1)
PY" || die "venv CUDA preflight failed"


docker exec -u devuser \
  -e SGL_DEBUG="$SGL_DEBUG" \
  -e OTEL_TRACES_SAMPLER="$OTEL_TRACES_SAMPLER" \
  "$CONTAINER_NAME" bash -lc "bash /sgl-workspace/sglang/.devcontainer/observability/eventlog.sh event sglang_trace_config run_id=\"$RUN_ID\" server_session_id=\"$SERVER_SESSION_ID\" trace=$TRACE_STATE otlp=\"$OTLP\" otel_attrs=\"$OTEL_ATTRS\" || true" >/dev/null

if [ -n "$HOST_LOG_FILE" ]; then
  printf '[%s] tracing config: trace=%s otlp=%s server_session_id=%s otel_attrs="%s"\n' "$(now_iso)" "$TRACE_STATE" "$OTLP" "$SERVER_SESSION_ID" "$OTEL_ATTRS" >> "$HOST_LOG_FILE" || true
fi

# Start server in the container (avoid Bash @Q quoting; pass env vars instead)
# Persist caches into /profiles mounts for the main server as well
# Fail fast: ensure DeepGEMM cache dir is writable inside the container
docker exec -u devuser "$CONTAINER_NAME" bash -lc 'test -d /profiles/deep_gemm -a -w /profiles/deep_gemm' \
  || die "SGL_DG_CACHE_DIR (/profiles/deep_gemm) missing or not writable in container"

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
  -e OTEL_TRACES_SAMPLER="$OTEL_TRACES_SAMPLER" \
  -e SGL_DEBUG="$SGL_DEBUG" \
  -e SGLANG_MOE_TRACE_DIR="$SGLANG_MOE_TRACE_DIR" \
  -e SGLANG_MOE_TRACE_PHASE="$SGLANG_MOE_TRACE_PHASE" \
  -e SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC="$SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC" \
  -e EXPERT_DISTRIBUTION_RECORDER_MODE="$EXPERT_DISTRIBUTION_RECORDER_MODE" \
  -e OTEL_RESOURCE_ATTRIBUTES="$OTEL_ATTRS" \
  -e SGL_CONTAINER_RUN_ID="$RUN_ID" \
  -e SGL_SERVER_SESSION_ID="$SERVER_SESSION_ID" \
  -e TRITON_CACHE_DIR="/profiles/triton" \
  -e TORCHINDUCTOR_CACHE_DIR="/profiles/torchinductor" \
  -e FLASHINFER_WORKSPACE_DIR="/profiles/flashinfer" \
  -e FLASHINFER_JIT_LOG_DIR="/profiles/flashinfer/90a" \
  -e SGL_DG_CACHE_DIR="/profiles/deep_gemm" \
  -e SGLANG_MOE_CONFIG_DIR="/profiles/moe_configs" \
  -e LOG_FILE="$LOG_FILE" \
  -e PYTHON_BIN="$PYTHON_BIN" \
  "$CONTAINER_NAME" bash -lc "\
  bash /sgl-workspace/sglang/.devcontainer/observability/eventlog.sh event sglang_started run_id=\"$RUN_ID\" server_session_id=\"$SERVER_SESSION_ID\" model_slug=\"$MODEL_SLUG\" tp=\"$TP_SIZE\" kv_cache_dtype=\"$KV\" || true; \
  echo '[trace_debug] SGLANG_MOE_TRACE_DIR='\"$SGLANG_MOE_TRACE_DIR\"' SGLANG_MOE_TRACE_PHASE='\"$SGLANG_MOE_TRACE_PHASE\"' SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC='\"$SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC\" >> \"$LOG_FILE\"; \
  PY_BIN=\"${PYTHON_BIN}\"; \
  nohup \"\$PY_BIN\" -m sglang.launch_server \\
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
    \${EXPERT_DISTRIBUTION_RECORDER_MODE:+--expert-distribution-recorder-mode \"$EXPERT_DISTRIBUTION_RECORDER_MODE\"} \\
    \${SGLANG_EXTRA_ARGS:+$SGLANG_EXTRA_ARGS} \\
    >> \"\$LOG_FILE\" 2>&1 & disown" >/dev/null

# Poll readiness (configurable, default 180s)
READY_TIMEOUT=${READY_TIMEOUT:-180}
deadline=$((SECONDS+READY_TIMEOUT))
while [ $SECONDS -lt $deadline ]; do
  if "$SCRIPT_DIR/status.sh" | grep -qx ready; then
    probe_json=$(curl -fsS --max-time 10 "http://127.0.0.1:${HOST_PORT}/trace_probe" 2>/dev/null || true)
    if [ -z "$probe_json" ]; then
      die "trace probe endpoint unavailable on port ${HOST_PORT}"
    fi
    if ! python3 - "$probe_json" <<'PY'; then
import json
import sys

data = json.loads(sys.argv[1])
enabled = data.get("tracing_enabled")
threads = data.get("threads_registered")
if enabled not in (True, 1, "true") or not threads:
    raise SystemExit(1)
PY
      die "trace probe endpoint reported tracing disabled"
    fi
    # Emit sglang_ready event and optional hello-world tick
    docker exec -u devuser -e OTEL_TRACES_SAMPLER="$OTEL_TRACES_SAMPLER" -e SGL_DEBUG="$SGL_DEBUG" "$CONTAINER_NAME" bash -lc \
      "bash /sgl-workspace/sglang/.devcontainer/observability/eventlog.sh event sglang_ready run_id=\"$RUN_ID\" server_session_id=\"$SERVER_SESSION_ID\" model_slug=\"$MODEL_SLUG\" tp=\"$TP_SIZE\" kv_cache_dtype=\"$KV\" || true"
    if [ "${HELLO_AFTER_READY:-}" = "1" ]; then
      docker exec -u devuser "$CONTAINER_NAME" bash -lc \
        'bash /sgl-workspace/sglang/scripts/infer/hello_world.sh 127.0.0.1 '"$HOST_PORT"' || true'
    fi
    STARTED_AT_ISO="$(now_iso)"
    emit_ready_json "$STARTED_AT_ISO" "$RUN_ID" "$SERVER_SESSION_ID" "$HOST_PORT" "$MANIFEST_HOST" "$LOG_FILE" "$SESSION_RECORD_HOST_PATH" "$TP_SIZE" "$MEM" "$CHUNK" "$CTX" "$MAXP" "$MAXT" "$MAMBA"
    if [ -n "$HOST_LOG_FILE" ]; then
      printf '[%s] sglang server ready: session=%s port=%s manifest=%s session_record=%s\n' "$STARTED_AT_ISO" "$SERVER_SESSION_ID" "$HOST_PORT" "$MANIFEST_HOST" "$SESSION_RECORD_HOST_PATH" >> "$HOST_LOG_FILE" || true
    fi
    exit 0
  fi
  sleep 1
done

echo "ERROR: server did not become ready on port $HOST_PORT within $READY_TIMEOUT seconds" >&2
if [ -n "$LOG_FILE" ] && docker exec -u devuser "$CONTAINER_NAME" bash -lc "test -f \"$LOG_FILE\""; then
  echo "--- tail of server log ($LOG_FILE) ---" >&2
  docker exec -u devuser "$CONTAINER_NAME" bash -lc "tail -n 200 \"$LOG_FILE\"" >&2 || true
fi
die "server did not become ready on port $HOST_PORT"
