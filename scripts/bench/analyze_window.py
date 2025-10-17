#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


def parse_hist(text: str, prefix: str):
    sumv = count = 0.0
    for line in text.splitlines():
        if line.startswith(prefix + "_sum"):
            sumv = float(line.rsplit(" ", 1)[-1])
        elif line.startswith(prefix + "_count"):
            count = float(line.rsplit(" ", 1)[-1])
    return sumv, count


def parse_counter(text: str, name: str):
    # returns first numeric value for the named counter
    for line in text.splitlines():
        if line.startswith(name + "{"):
            try:
                return float(line.rsplit(" ", 1)[-1])
            except:
                pass
    return 0.0


def analyze(bench_dir: Path):
    before = bench_dir / "sglang_before.prom"
    after = bench_dir / "sglang_after.prom"
    http_summary = bench_dir / "http_summary.json"
    if not (before.exists() and after.exists() and http_summary.exists()):
        raise SystemExit("missing inputs in " + str(bench_dir))
    b = before.read_text()
    a = after.read_text()
    # Token counters (prefill/decode)
    prefill_b = parse_counter(b, "sglang:prompt_tokens_total")
    prefill_a = parse_counter(a, "sglang:prompt_tokens_total")
    decode_b = parse_counter(b, "sglang:generation_tokens_total")
    decode_a = parse_counter(a, "sglang:generation_tokens_total")
    prefill_delta = prefill_a - prefill_b
    decode_delta = decode_a - decode_b
    # Hist sums for TTFT and E2E
    ttft_b, ttft_cb = parse_hist(b, "sglang:time_to_first_token_seconds")
    ttft_a, ttft_ca = parse_hist(a, "sglang:time_to_first_token_seconds")
    e2e_b, e2e_cb = parse_hist(b, "sglang:e2e_request_latency_seconds")
    e2e_a, e2e_ca = parse_hist(a, "sglang:e2e_request_latency_seconds")
    ttft_sum = ttft_a - ttft_b
    e2e_sum = e2e_a - e2e_b
    decode_sum = max(0.0, e2e_sum - ttft_sum)
    # Wall window seconds from HTTP summary
    raw = http_summary.read_text()
    window_s = 0.0
    try:
        h = json.loads(raw)
        window_s = float(h.get("window_seconds", 0.0))
    except Exception:
        # Fallback: regex extract window_seconds
        m = re.search(r'"window_seconds"\s*:\s*([0-9\.]+)', raw)
        if m:
            window_s = float(m.group(1))
        if not window_s:
            # Final fallback: derive from samples.csv timestamp range or row count (1 Hz)
            samples = bench_dir / "samples.csv"
            try:
                lines = samples.read_text().splitlines()
                if len(lines) > 2:
                    # parse first and last timestamp
                    import datetime as dt

                    def parse(ts):
                        return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")

                    first = parse(lines[1].split(",")[0])
                    last = parse(lines[-1].split(",")[0])
                    window_s = max(0.0, (last - first).total_seconds())
                else:
                    window_s = max(0.0, float(len(lines) - 1))
            except Exception:
                window_s = 0.0
    # Throughputs
    prefill_tps_wall = (prefill_delta / window_s) if window_s > 0 else None
    decode_tps_wall = (decode_delta / window_s) if window_s > 0 else None
    prefill_tps_stage = (prefill_delta / ttft_sum) if ttft_sum > 0 else None
    decode_tps_stage = (decode_delta / decode_sum) if decode_sum > 0 else None
    out = {
        "bench_dir": str(bench_dir),
        "window_seconds": window_s,
        "tokens": {"prefill_delta": prefill_delta, "decode_delta": decode_delta},
        "sums": {"ttft_sum": ttft_sum, "e2e_sum": e2e_sum, "decode_sum": decode_sum},
        "tps_wall": {"prefill": prefill_tps_wall, "decode": decode_tps_wall},
        "tps_stage": {"prefill": prefill_tps_stage, "decode": decode_tps_stage},
        "counts": {
            "ttft_count_delta": ttft_ca - ttft_cb,
            "e2e_count_delta": e2e_ca - e2e_cb,
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_dir", help="path to benchmark directory under the run")
    ap.add_argument("--out", help="optional output JSON path")
    args = ap.parse_args()
    res = analyze(Path(args.bench_dir))
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
