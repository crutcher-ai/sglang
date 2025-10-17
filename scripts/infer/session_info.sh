#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: session_info.sh [--manifest PATH] [--pointer PATH] [--validate] [--help]

Print the authoritative server session JSON (start.json) from the most-recent
provider session under the current container run directory. This script is
strictly read-only and performs no heuristics or fallbacks.

Options:
  --manifest PATH   Absolute path to a container-run manifest JSON.
  --pointer PATH    Absolute path to container_run_meta.env (default:
                    $HOST_OBS_ROOT/telemetry/container_run_meta.env).
  --validate        Validate presence only; do not print JSON.
  -h, --help        Show this message.

Environment overrides:
  CONTAINER_RUN_META_JSON_HOST   If set, used as manifest path (overrides --pointer).
  HOST_OBS_ROOT                  Base for default pointer path
                                 (default: $HOME/sglang-observability).

Exit codes:
  0  success (JSON printed unless --validate)
  2  inconsistent/not ready (sessions directory or start.json missing)
  3  bad pointer/manifest (missing or invalid)
  4  dependency/runtime error (e.g., Python failure)
  1  unknown/usage error
EOF
}

# Defaults
HOST_OBS_ROOT="${HOST_OBS_ROOT:-$HOME/sglang-observability}"
POINTER_DEFAULT="${HOST_OBS_ROOT%/}/telemetry/container_run_meta.env"
MANIFEST_PATH=""
POINTER_PATH="$POINTER_DEFAULT"
VALIDATE_ONLY=0

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      [[ $# -ge 2 ]] || { echo "ERROR: --manifest requires a path" >&2; usage; exit 1; }
      MANIFEST_PATH="$2"; shift 2;;
    --pointer)
      [[ $# -ge 2 ]] || { echo "ERROR: --pointer requires a path" >&2; usage; exit 1; }
      POINTER_PATH="$2"; shift 2;;
    --validate)
      VALIDATE_ONLY=1; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "ERROR: Unknown argument: $1" >&2; usage; exit 1;;
  esac
done

# Resolve manifest path (NO FALLBACKS beyond pointer -> host JSON)
if [[ -z "$MANIFEST_PATH" ]]; then
  if [[ -n "${CONTAINER_RUN_META_JSON_HOST:-}" ]]; then
    MANIFEST_PATH="$CONTAINER_RUN_META_JSON_HOST"
  else
    if [[ ! -f "$POINTER_PATH" ]]; then
      echo "ERROR: pointer not found: $POINTER_PATH" >&2
      exit 3
    fi
    # Extract CONTAINER_RUN_META_JSON_HOST=... (tolerate whitespace and quotes)
    raw_host_path=$(awk -F= '/^[[:space:]]*CONTAINER_RUN_META_JSON_HOST[[:space:]]*=/{print $2; exit}' "$POINTER_PATH" || true)
    raw_host_path="${raw_host_path//\r/}"
    raw_host_path="${raw_host_path//\t/}"
    raw_host_path="${raw_host_path//\n/}"
    raw_host_path="${raw_host_path## }"  # trim leading space
    raw_host_path="${raw_host_path%% }"  # trim trailing space
    # Strip surrounding single/double quotes if present
    if [[ "${raw_host_path}" == '"'*'"' ]]; then
      raw_host_path="${raw_host_path#\"}"; raw_host_path="${raw_host_path%\"}"
    elif [[ "${raw_host_path}" == "'*'" ]]; then
      raw_host_path="${raw_host_path#\'}"; raw_host_path="${raw_host_path%\'}"
    fi
    if [[ -z "$raw_host_path" ]]; then
      echo "ERROR: CONTAINER_RUN_META_JSON_HOST not defined in pointer: $POINTER_PATH" >&2
      exit 3
    fi
    MANIFEST_PATH="$raw_host_path"
  fi
fi

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "ERROR: manifest not found: $MANIFEST_PATH" >&2
  exit 3
fi

PY_SCRIPT='''
import json, sys, os, glob

def die(msg, code):
    sys.stderr.write(msg + "\n"); sys.exit(code)

manifest_path = sys.argv[1]

try:
    with open(manifest_path, 'r', encoding='utf-8') as f:
        m = json.load(f)
except Exception as e:
    die(f"ERROR: failed to parse manifest JSON: {e}", 3)

paths = (m.get('paths') or {})
host_paths = paths.get('host') if isinstance(paths.get('host'), dict) else {}
run_dir = host_paths.get('run_dir') or host_paths.get('telemetry_root')
if not run_dir or not isinstance(run_dir, str):
    die("ERROR: manifest missing paths.host.run_dir (or telemetry_root)", 3)

sessions_dir = os.path.join(run_dir, 'logs', 'provider_sessions')
if not os.path.isdir(sessions_dir):
    die(f"ERROR: sessions directory missing: {sessions_dir}", 2)

candidates = []
for d in glob.glob(os.path.join(sessions_dir, '*')):
    if os.path.isdir(d):
        p = os.path.join(d, 'start.json')
        if os.path.isfile(p):
            try:
                mt = os.stat(p).st_mtime
            except Exception:
                continue
            candidates.append((mt, p))

if not candidates:
    die("ERROR: no session start.json found", 2)

candidates.sort(key=lambda x: x[0], reverse=True)
start_path = candidates[0][1]

# Validate JSON readability
try:
    with open(start_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    # Re-open to stream original bytes
    with open(start_path, 'r', encoding='utf-8') as f:
        content = f.read()
except Exception as e:
    die(f"ERROR: failed to parse session JSON: {e}", 2)

sys.stdout.write(content)
'''

if [[ "$VALIDATE_ONLY" -eq 1 ]]; then
  if python3 - "$MANIFEST_PATH" <<<"$PY_SCRIPT" >/dev/null 2>&1; then
    exit 0
  else
    # Try to extract exit code from Python; if unavailable map to 2
    code=$?
    if [[ $code -eq 0 ]]; then code=2; fi
    exit $code
  fi
fi

# Normal mode: print JSON to stdout on success; propagate Python exit code
python3 - "$MANIFEST_PATH" <<<"$PY_SCRIPT" || {
  code=$?
  # Normalize unexpected failures to 4 per contract
  if [[ $code -ne 2 && $code -ne 3 ]]; then code=4; fi
  exit $code
}
