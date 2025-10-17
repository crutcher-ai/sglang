#!/usr/bin/env bash
set -euo pipefail

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
TOTAL=${TOTAL:-80}
CONC=${CONCURRENCY:-8}
TIMEOUT=${TIMEOUT:-120}
PROM_SAMPLE_SECS=${PROM_SAMPLE_SECS:-1}

payload=$(mktemp)
cat > "$payload" <<'JSON'
{
  "text": "Summarize the CAP theorem and its implications for distributed databases. Include concrete examples and trade-offs.",
  "sampling_params": {"temperature": 0.2, "max_new_tokens": 256}
}
JSON

RUN_META_FILE="$HOME/sglang-observability/telemetry/container_run_meta.env"
if [[ ! -f "$RUN_META_FILE" ]]; then
  echo "ERROR: pointer not found: $RUN_META_FILE" >&2; exit 2
fi
MANIFEST_HOST=$(awk -F= '/^CONTAINER_RUN_META_JSON_HOST=/{print $2}' "$RUN_META_FILE")
if [[ -z "$MANIFEST_HOST" || ! -f "$MANIFEST_HOST" ]]; then echo "ERROR: manifest missing" >&2; exit 2; fi
RUN_DIR=$(python3 - "$MANIFEST_HOST" <<'PY'
import json,sys
mf=sys.argv[1]
d=json.load(open(mf))
print((d.get('paths',{}) .get('host') or {}).get('run_dir',''))
PY
)
if [[ -z "$RUN_DIR" ]]; then echo "ERROR: run_dir not in manifest" >&2; exit 2; fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="$RUN_DIR/benchmarks/$STAMP"
mkdir -p "$OUT_DIR"
cp "$payload" "$OUT_DIR/payload.json"

echo '--- capturing pre metrics ---'
curl -s "http://${HOST}:${PORT}/metrics" > "$OUT_DIR/sglang_before.prom"

echo '--- start sampling node/dcgm ---'
(
  echo "ts,gpu_util_pct,fb_used_MiB,power_W,node_load1,node_load5,node_load15,mem_used_GB"
  while :; do
    TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    DCGM=$(curl -s http://${HOST}:9400/metrics || true)
    NODE=$(curl -s http://${HOST}:9100/metrics || true)
    gpu_util=$(echo "$DCGM" | grep -E '^DCGM_FI_DEV_GPU_UTIL\{[^}]*gpu="0"' | tail -n1 | awk '{print $NF}' || echo 0)
    fb_used=$(echo "$DCGM" | grep -E '^DCGM_FI_DEV_FB_USED\{[^}]*gpu="0"' | tail -n1 | awk '{print $NF}' || echo 0)
    power=$(echo "$DCGM" | grep -E '^DCGM_FI_DEV_POWER_USAGE\{[^}]*gpu="0"' | tail -n1 | awk '{print $NF}' || echo 0)
    l1=$(echo "$NODE" | grep -E '^node_load1 ' | awk '{print $NF}' || echo 0)
    l5=$(echo "$NODE" | grep -E '^node_load5 ' | awk '{print $NF}' || echo 0)
    l15=$(echo "$NODE" | grep -E '^node_load15 ' | awk '{print $NF}' || echo 0)
    mem_tot=$(echo "$NODE" | grep -E '^node_memory_MemTotal_bytes ' | awk '{print $NF}' || echo 1)
    mem_avl=$(echo "$NODE" | grep -E '^node_memory_MemAvailable_bytes ' | awk '{print $NF}' || echo 0)
    mem_used_gb=$(python3 - <<PY 2>/dev/null
import os
try:
    tot=float(os.environ['MT']); av=float(os.environ['MA']); print((tot-av)/1e9)
except Exception:
    print(0)
PY
MT="$mem_tot" MA="$mem_avl")
    echo "$TS,$gpu_util,$fb_used,$power,$l1,$l5,$l15,$mem_used_gb"
    sleep "$PROM_SAMPLE_SECS"
  done
) > "$OUT_DIR/samples.csv" 2>/dev/null &
SAMPLE_PID=$!

echo '--- run workload ---'
python3 scripts/bench/http_bench.py \
  --host "$HOST" --port "$PORT" --path /generate \
  --body "$OUT_DIR/payload.json" \
  --total "$TOTAL" --concurrency "$CONC" --timeout "$TIMEOUT" \
  --outdir "$OUT_DIR" | tee "$OUT_DIR/http_summary.json"

echo '--- stop sampling ---'
kill "$SAMPLE_PID" >/dev/null 2>&1 || true

echo '--- capturing post metrics ---'
curl -s "http://${HOST}:${PORT}/metrics" > "$OUT_DIR/sglang_after.prom"

echo '--- analyze window ---'
python3 - <<'PY' "$OUT_DIR"
import sys, re, json, statistics as st, glob
outdir=sys.argv[1]

def parse_hist(text, prefix):
    sumv=count=0.0; buckets={}
    for line in text.splitlines():
        if line.startswith(prefix+"_sum"):
            sumv=float(line.rsplit(' ',1)[-1])
        elif line.startswith(prefix+"_count"):
            count=float(line.rsplit(' ',1)[-1])
        elif line.startswith(prefix+"_bucket"):
            m=re.search(r'le="([^"]+)"', line)
            if m:
                le=m.group(1); val=float(line.rsplit(' ',1)[-1]); buckets[le]=val
    return sumv, count, buckets

bef=open(outdir+"/sglang_before.prom").read(); aft=open(outdir+"/sglang_after.prom").read()
_, ttft_cb, _ = parse_hist(bef,'sglang:time_to_first_token_seconds')
_, ttft_ca, _ = parse_hist(aft,'sglang:time_to_first_token_seconds')
_, e2e_cb, _  = parse_hist(bef,'sglang:e2e_request_latency_seconds')
_, e2e_ca, _  = parse_hist(aft,'sglang:e2e_request_latency_seconds')

lat=[]; codes=[]
for m in sorted(glob.glob(outdir+"/meta_*.txt")):
    txt=open(m).read().strip()
    mc=re.search(r'code:(\d+)', txt); mt=re.search(r'time:([0-9\.]+)', txt)
    if mc: codes.append(int(mc.group(1)))
    if mt:
        try: lat.append(float(mt.group(1)))
        except: pass
rep={
  'workload': {'total': len(lat), 'concurrency': 8},
  'http_ok': sum(1 for c in codes if c==200),
  'curl_p50_s': (st.median(lat) if lat else None),
  'curl_p95_est_s': (st.quantiles(lat, n=20)[18] if len(lat)>=20 else (max(lat) if lat else None)),
  'sglang_e2e_count_delta': (e2e_ca-e2e_cb),
  'sglang_ttft_count_delta': (ttft_ca-ttft_cb)
}
open(outdir+"/report.json","w").write(json.dumps(rep, indent=2))
print(json.dumps(rep, indent=2))
PY

echo "Report written to: $OUT_DIR/report.json"
