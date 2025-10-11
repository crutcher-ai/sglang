#!/usr/bin/env bash
set -euo pipefail

# Prints a live snapshot of kernel caches using canonical host paths from the run manifest.
# Fails fast if the run pointer or manifest is missing.

sel_triton=false
sel_inductor=false
sel_flashinfer=false
sel_deepgemm=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --triton) sel_triton=true; shift ;;
    --inductor) sel_inductor=true; shift ;;
    --flashinfer) sel_flashinfer=true; shift ;;
    --deep-gemm|--deep_gemm) sel_deepgemm=true; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/cache/inspect_caches.sh [--json] [--triton|--inductor|--flashinfer|--deep-gemm ...]

Print snapshot for kernel caches under HOST_PROFILES_ROOT.
Defaults to all caches when none are specified.
EOF
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Resolve canonical host roots from the pointer → manifest
HOST_OBS_ROOT="${HOST_OBS_ROOT:-$HOME/sglang-observability}"
RUN_META_FILE="${HOST_TELEMETRY_ROOT:-${HOST_OBS_ROOT}/telemetry}/container_run_meta.env"
if [[ ! -f "$RUN_META_FILE" ]]; then
  echo "ERROR: pointer file not found at $RUN_META_FILE" >&2
  exit 3
fi
MANIFEST_HOST=$(awk -F= '/^CONTAINER_RUN_META_JSON_HOST=/{print $2}' "$RUN_META_FILE" || true)
if [[ -z "$MANIFEST_HOST" || ! -f "$MANIFEST_HOST" ]]; then
  echo "ERROR: manifest not found via pointer ($MANIFEST_HOST)" >&2
  exit 3
fi
HOST_PROFILES_ROOT=$(python3 - "$MANIFEST_HOST" <<'PY'
import json,sys
mf=sys.argv[1]
d=json.load(open(mf))
paths=(d.get('paths') or {}).get('host') or {}
root=paths.get('profiles_root') or None
if not root:
    # Fallback to storage_root/../profiles only if present in manifest
    sr=paths.get('storage_root')
    root = (sr.rstrip('/')+'/../profiles') if sr else None
if not root:
    print('')
else:
    print(root)
PY
)
if [[ -z "$HOST_PROFILES_ROOT" || ! -d "$HOST_PROFILES_ROOT" ]]; then
  echo "ERROR: could not resolve profiles root from manifest" >&2
  exit 3
fi

if ! $sel_triton && ! $sel_inductor && ! $sel_flashinfer && ! $sel_deepgemm; then
  sel_triton=true; sel_inductor=true; sel_flashinfer=true; sel_deepgemm=true
fi

container_id=$(docker ps -q -f name=sglang-dev | head -1 || true)
get_py_info='python3 - <<PY
import json
def get_ver(mod):
  try:
    m = __import__(mod)
    return getattr(m, "__version__", "unknown")
  except Exception:
    return "unknown"
info = {"torch_version": get_ver("torch"), "triton_version": get_ver("triton")}
try:
  info["flashinfer_version"] = get_ver("flashinfer")
except Exception:
  info["flashinfer_version"] = "unknown"
try:
  import torch
  cc = torch.cuda.get_device_capability(0)
  info["compute_capability"] = f"sm_{cc[0]}{cc[1]}"
  info["device_name"] = torch.cuda.get_device_name(0)
except Exception:
  info.update({"compute_capability":"unknown","device_name":"unknown"})
print(json.dumps(info))
PY'

torch_version=unknown; triton_version=unknown; flashinfer_version=unknown; compute_capability=unknown; device_name=unknown
if [[ -n "$container_id" ]]; then
  py_json=$(docker exec "$container_id" bash -lc "$get_py_info" 2>/dev/null || echo '{}')
  torch_version=$(echo "$py_json" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("torch_version","unknown"))')
  triton_version=$(echo "$py_json" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("triton_version","unknown"))')
  flashinfer_version=$(echo "$py_json" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("flashinfer_version","unknown"))')
  compute_capability=$(echo "$py_json" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("compute_capability","unknown"))')
  device_name=$(echo "$py_json" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("device_name","unknown"))')
fi

driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1 || echo unknown)

torch_version="$torch_version" triton_version="$triton_version" flashinfer_version="$flashinfer_version" \
compute_capability="$compute_capability" device_name="$device_name" driver_version="$driver_version" cuda="$cuda" \
python3 - "$HOST_PROFILES_ROOT" "$sel_triton" "$sel_inductor" "$sel_flashinfer" "$sel_deepgemm" <<'PY'
import json, os, sys
from pathlib import Path
from datetime import datetime, timezone

root = Path(sys.argv[1])
sel = {
  'triton': sys.argv[2].lower() == 'true',
  'torchinductor': sys.argv[3].lower() == 'true',
  'flashinfer': sys.argv[4].lower() == 'true',
  'deep_gemm': sys.argv[5].lower() == 'true',
}

env = {
  'torch_version': os.environ.get('torch_version','unknown'),
  'triton_version': os.environ.get('triton_version','unknown'),
  'flashinfer_version': os.environ.get('flashinfer_version','unknown'),
  'compute_capability': os.environ.get('compute_capability','unknown'),
  'device_name': os.environ.get('device_name','unknown'),
  'driver_version': os.environ.get('driver_version','unknown'),
  'cuda': os.environ.get('cuda','unknown'),
  'sglang_commit': 'unknown',
  'tp_size': None,
  'model_slug': None,
}

def stats(p: Path):
    if not p.exists():
        return {"exists": False, "size_bytes": 0, "file_count": 0, "latest_mtime_iso": None}
    size = 0; count = 0; latest = 0.0
    for r,_,files in os.walk(p):
        for fn in files:
            fp = Path(r)/fn
            try:
                st = fp.stat()
            except FileNotFoundError:
                continue
            size += st.st_size; count += 1; latest = max(latest, st.st_mtime)
    iso = None if latest==0 else datetime.fromtimestamp(latest, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    return {"exists": True, "size_bytes": size, "file_count": count, "latest_mtime_iso": iso}

def marker_info(p: Path):
    mr = root/'.in_progress'
    if not mr.exists():
        return False, None
    for m in mr.rglob('*'):
        if m.is_file():
            try:
                return True, json.loads(m.read_text())
            except Exception:
                return True, {"owner_pid": None, "started_at": None}
    return False, None

snapshot = {"schema_version":"1"}
for key, sub in (('triton','triton'), ('torchinductor','torchinductor'), ('flashinfer','flashinfer'), ('deep_gemm','deep_gemm')):
    if not sel[key]:
        continue
    s = stats(root/sub)
    s.update({"valid": bool(s["exists"]), "path": str(root/sub), "signature": env})
    partial, pinfo = marker_info(root/sub)
    s["partial"] = partial
    if partial:
        s["valid"] = False
        s["reason"] = "in_progress_or_aborted"
        s["partial_info"] = pinfo
    snapshot[key] = s

print(json.dumps(snapshot, indent=2))
PY
