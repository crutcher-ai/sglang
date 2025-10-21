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
    # Optional filters so we can focus on a particular run/session.
    ap.add_argument("--run-id", help="Filter records by run_id")
    ap.add_argument("--session-id", help="Filter records by server_session_id")
    ap.add_argument(
        "--phase",
        default=None,
        help="Filter by phase (e.g., decode, prefill, mixed). Comma-separated allowed.",
    )
    ap.add_argument(
        "--dedupe",
        action="store_true",
        help="De-duplicate token events by (phase, step, layer, slot, position, experts)",
    )
    ap.add_argument(
        "--step-window",
        type=int,
        default=0,
        help="Show per-step token counts for the last N steps (0=disable)",
    )
    args = ap.parse_args()

    trace_dir = Path(args.dir)
    files = sorted(trace_dir.glob("expert_trace_*.jsonl"))
    if not files:
        print("No expert_trace_*.jsonl files found in", trace_dir)
        return 2

    per_layer_counts = defaultdict(Counter)
    per_phase_layer_counts = defaultdict(lambda: defaultdict(Counter))
    total_records = 0
    total_token_events = 0
    records_by_phase = Counter()
    token_events_by_phase = Counter()
    step_layer_events = Counter()  # (phase, step, layer) -> token count
    all_steps = set()

    phase_filter = None
    if args.phase:
        phase_filter = {p.strip().lower() for p in args.phase.split(",") if p.strip()}

    seen = set()  # dedupe across files if requested

    for f in files:
        with f.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                run_id = rec.get("run_id")
                session_id = rec.get("session_id")

                if args.run_id and run_id != args.run_id:
                    continue
                if args.session_id and session_id != args.session_id:
                    continue

                layer = int(rec.get("layer", -1))
                phase = (rec.get("phase", "unknown") or "unknown").lower()
                if (
                    phase_filter
                    and phase not in phase_filter
                    and "all" not in phase_filter
                ):
                    continue
                tokens = rec.get("tokens", [])
                step = int(rec.get("step", -1))
                for tok in tokens:
                    slot = tok.get("slot")
                    pos = tok.get("position")
                    experts = tuple(int(ex) for ex in tok.get("experts", []))
                    if args.dedupe:
                        key = (phase, step, layer, slot, pos, experts)
                        if key in seen:
                            continue
                        seen.add(key)
                    for ex in experts:
                        per_layer_counts[layer][int(ex)] += 1
                        per_phase_layer_counts[phase][layer][int(ex)] += 1
                        total_token_events += 1
                        token_events_by_phase[phase] += 1
                    if experts:
                        step_layer_events[(phase, step, layer)] += 1
                        all_steps.add((phase, step))
                total_records += 1
                records_by_phase[phase] += 1

    print(
        f"files={len(files)} records={total_records} token_expert_events={total_token_events}"
    )

    if records_by_phase:
        print("phase summary:")
        for phase, rec_cnt in sorted(records_by_phase.items()):
            tok_cnt = token_events_by_phase.get(phase, 0)
            print(f"  {phase:>8}: records={rec_cnt} token_events={tok_cnt}")

    # Step coverage summary (helps confirm decode visibility)
    if all_steps:
        by_phase = defaultdict(set)
        for phase, step in all_steps:
            by_phase[phase].add(step)
        print("\nstep coverage:")
        for phase in sorted(by_phase.keys()):
            steps = sorted(by_phase[phase])
            print(
                f"  {phase:>8}: unique_steps={len(steps)} first={steps[0] if steps else 'n/a'} last={steps[-1] if steps else 'n/a'}"
            )

    if args.step_window and step_layer_events:
        max_step = max(s for (_, s, _) in step_layer_events.keys())
        min_step = max_step - args.step_window + 1
        print(
            f"\nper-step token counts (last {args.step_window} steps, phase filtered):"
        )
        for step in range(min_step, max_step + 1):
            # Sum across layers for readability
            total = sum(
                cnt for (ph, st, _), cnt in step_layer_events.items() if st == step
            )
            if total:
                print(f"  step {step:>6}: tokens={total}")

    for layer in sorted(k for k in per_layer_counts.keys() if k >= 0):
        c = per_layer_counts[layer]
        layer_total = sum(c.values()) or 1
        uniques = len(c)
        print(f"layer {layer:>3}: total={layer_total} unique_experts={uniques}")
        for ex, cnt in c.most_common(args.top):
            pct = 100.0 * cnt / layer_total
            print(f"  expert {ex:>4}: {cnt:>7} ({pct:5.1f}%)")

    if per_phase_layer_counts:
        print("\nper-phase detail:")
        for phase in sorted(per_phase_layer_counts.keys()):
            print(f"phase={phase}")
            for layer in sorted(
                k for k in per_phase_layer_counts[phase].keys() if k >= 0
            ):
                c = per_phase_layer_counts[phase][layer]
                layer_total = sum(c.values()) or 1
                uniques = len(c)
                print(
                    f"  layer {layer:>3}: total={layer_total} unique_experts={uniques}"
                )
                for ex, cnt in c.most_common(args.top):
                    pct = 100.0 * cnt / layer_total
                    print(f"    expert {ex:>4}: {cnt:>7} ({pct:5.1f}%)")


if __name__ == "__main__":
    raise SystemExit(main())
