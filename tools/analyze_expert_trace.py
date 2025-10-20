import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Analyze SGLang expert-trace JSONL files")
    ap.add_argument(
        "--dir", required=True, help="Directory containing expert_trace_*.jsonl"
    )
    ap.add_argument(
        "--top", type=int, default=10, help="Top-N experts per layer to show"
    )
    args = ap.parse_args()

    trace_dir = Path(args.dir)
    files = sorted(trace_dir.glob("expert_trace_*.jsonl"))
    if not files:
        print("No expert_trace_*.jsonl files found in", trace_dir)
        return 2

    per_layer_counts = defaultdict(Counter)
    total_records = 0
    total_token_events = 0

    for f in files:
        with f.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                layer = int(rec.get("layer", -1))
                tokens = rec.get("tokens", [])
                for tok in tokens:
                    for ex in tok.get("experts", []):
                        per_layer_counts[layer][int(ex)] += 1
                        total_token_events += 1
                total_records += 1

    print(
        f"files={len(files)} records={total_records} token_expert_events={total_token_events}"
    )
    for layer in sorted(k for k in per_layer_counts.keys() if k >= 0):
        c = per_layer_counts[layer]
        layer_total = sum(c.values()) or 1
        uniques = len(c)
        print(f"layer {layer:>3}: total={layer_total} unique_experts={uniques}")
        for ex, cnt in c.most_common(args.top):
            pct = 100.0 * cnt / layer_total
            print(f"  expert {ex:>4}: {cnt:>7} ({pct:5.1f}%)")


if __name__ == "__main__":
    raise SystemExit(main())
