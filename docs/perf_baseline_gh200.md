# GH200 Hopper (96 GB HBM3) — Perf + Memory Baseline, Observability, and Bench Harness

This document is a complete, end‑to‑end handoff of how the repo is set up to:

- launch the helper + server in a single, upstream‑style path/venv,
- collect rich telemetry (Prometheus + Jaeger v2),
- run a windowed HTTP benchmark with metrics sampling, and
- reason about memory headroom and parallelism (slots) with a repeatable method.

It reflects the exact changes/scripts now in the tree and the empirical results we measured on a GH200 with 96 GB HBM3 and 480 GB host DDR (DCGM FB_* refers to HBM only).

## Environment and Topology

- Hardware: NVIDIA GH200 (SM90), 96 GB HBM3, plus 480 GB host DDR (DCGM FB_* ≡ HBM3).
- Driver/CUDA: 12.9.
- Base image inside helper: `lmsysorg/sglang:dev` (arm64) with CUDA torch 2.8.0+cu129 in system site‑packages.
- Helper container runs Prometheus (9090), node_exporter (9100), dcgm‑exporter (9400), and Jaeger v2 (UI 16686, OTLP gRPC 4317).
- Canonical repo mount inside container: `/sgl-workspace/sglang` (upstream‑aligned).

## One Mount / One Venv

- Venv lives at `/sgl-workspace/sglang/.venv`.
- Venv created with `--system-site-packages` so it can use the container’s CUDA torch.
- sglang installed editable `--no-deps` to avoid pulling CPU torch into the venv.
- No PYTHONPATH hacks; only the venv interpreter is used.

## Observability Helper

- Start helper + exporters:
  - `./scripts/start_observable_container.sh`
  - Writes pointer: `~/sglang-observability/telemetry/container_run_meta.env`.
  - Prints the per‑run manifest JSON paths (host + container).
- Inside the helper, Jaeger v2 is OTLP‑native and served on host: 16686 (UI), 4317/4318 (OTLP).
- Prometheus scrapes node_exporter + dcgm‑exporter. SGLang `/metrics` is scraped by the bench harness on demand (we don’t auto‑add that scrape target).

## Server Launcher (inside helper)

- Script: `scripts/infer/start_server.sh`
- Reads `.devcontainer/tools/sglang-config.json` inside container and applies env overrides.
- CUDA preflight checks torch.cuda and CUDA 12.9.
- Exports structured trace resource attrs: `container_run=<run_id>`, `service.instance.id=<server_session_id>`.
- Caches go under `/profiles/{triton,torchinductor,flashinfer,deep_gemm,moe_configs}`.
- NEW: You can pass arbitrary extra CLI args to `sglang.launch_server` via `SGLANG_EXTRA_ARGS`. Example:
  - `SGLANG_EXTRA_ARGS="--mamba-ssm-dtype bfloat16"` (see capacity notes below).

### Common env overrides

- `MEM_FRACTION_STATIC` (float)
- `CONTEXT_LENGTH`, `MAX_TOTAL_TOKENS`, `MAX_PREFILL_TOKENS` (ints)
- `MAX_MAMBA_CACHE_SIZE` (int ∼ slot cap)
- `KV_CACHE_DTYPE` (`fp8_e4m3` for FP8 KV)
- `ENABLE_TRACE=1` `OTEL_TRACES_SAMPLER=always_on`

## Windowed Bench Harness

- Runner: `scripts/bench/run_window_bench.sh`
  - Input vars: `CONCURRENCY` (default 8), `TOTAL` (default 80)
  - Captures SGLang `/metrics` before/after window.
  - Samples dcgm (9400) and node (9100) once/sec to `samples.csv` during the window.
  - Emits a bench directory under the current run: `~/sglang-observability/telemetry/container_runs/<run_id>/benchmarks/<timestamp>/`.
- Analyzer: `scripts/bench/analyze_window.py <bench_dir> --out <bench_dir>/tokens_report.json`
  - Computes wall TPS and stage TPS (prefill vs decode) from Prom metrics deltas and histogram sums.

### CSV fix (important)

- `scripts/bench/run_window_bench.sh` now writes numeric values for DCGM and node metrics using `grep -E ... | awk '{print $NF}'`. This avoids the earlier issue where model strings (e.g., `GH200`) polluted numeric fields.

### Quick smoke

- `CONCURRENCY=96 TOTAL=640 scripts/bench/run_window_bench.sh`
- Then analyze: `python3 scripts/bench/analyze_window.py <bench_dir> --out <bench_dir>/tokens_report.json`

## Memory / Parallelism: What We Measured

We report everything as observed in DCGM (HBM3) and from server log pool sizes.

- KV pool:
  - FP8 KV (`fp8_e4m3`) at 16k ctx: ≈ 0.18 GB (K+V).
  - FP8 KV at 32k ctx: ≈ 0.38 GB.
  - FP8 KV at 262k ctx: ≈ 3.0 GB.
  - KV is small vs. per‑slot costs at lower contexts.

- Mamba pool (hybrid GDN/Mamba):
  - Log line: `Mamba Cache is allocated. conv_state size: X GB, ssm_state size: Y GB`.
  - Empirical per‑slot increment (16k ctx): ≈ 73.6 MiB/slot (delta of Mamba between size 16 and 64 divided by 48).
  - Switching Mamba SSM dtype to `bfloat16` reduces the large temporal state; per‑slot decreases materially (in practice the log deltas still aggregated ~3.45 GB from size 16→64 on our runs; we recommend validating this on each model).

- Total per‑slot slope (decode‑heavy; mean over 20–60s windows):
  - 262k ctx: ≈ 153.5 MiB/slot
  - 32k ctx:  ≈ 96.2 MiB/slot
  - 16k ctx:  ≈ 75–85 MiB/slot (mean window), with peak spikes 10–40 MiB/slot higher for ~tens of seconds post‑startup.

### Startup spikes

Using 60s windows immediately after readiness, peak‑based slopes ((max64−min16)/48) were consistently above mean‑based slopes by ~10–40 MiB/slot, indicating a brief allocator/graph‑capture “spike then settle”. For throughput windows, use the mean series during steady state.

## Decode‑Heavy Throughput (ctx=16k, mfs=0.98, Mamba SSM bf16, MAX_MAMBA=160)

- Concurrency 96
  - Window: 69.15 s
  - Tokens Δ: prefill 12,800; decode 163,840
  - Wall TPS: prefill 185.10; decode 2,369.29
  - Stage TPS: prefill 4.18; decode 49.59
  - HBM (fb_used MiB) during window: min/mean/max = 94,330 / 94,330 / 94,330

- Concurrency 112
  - Window: 66.38 s
  - Tokens Δ: prefill 12,800; decode 163,840
  - Wall TPS: prefill 192.83; decode 2,468.18
  - Stage TPS: prefill 3.37; decode 50.12
  - HBM: min/mean/max = 94,330 / 94,330 / 94,330

- Concurrency 128
  - Window: 57.22 s
  - Wall TPS (decode): 2,863.55

- Concurrency 160
  - Window: 58.89 s
  - Wall TPS (decode): 2,782.03

The decode‑heavy knee sits near ~128 for this model/config; aggregate throughput slightly regresses at 160.

## Operational Runbook

### Start helper

```
./scripts/start_observable_container.sh
```

Inspect pointer at `~/sglang-observability/telemetry/container_run_meta.env`.

### Launch server (examples)

Baseline decode‑heavy (bf16 Mamba SSM, ctx 16k):

```
SGLANG_EXTRA_ARGS="--mamba-ssm-dtype bfloat16" \
MEM_FRACTION_STATIC=0.98 \
ENABLE_TRACE=1 OTEL_TRACES_SAMPLER=always_on \
MAX_MAMBA_CACHE_SIZE=160 CONTEXT_LENGTH=16384 MAX_TOTAL_TOKENS=16384 MAX_PREFILL_TOKENS=16384 \
READY_TIMEOUT=600 ./scripts/infer/start_server.sh
```

If startup fails with “Not enough memory” at high caps:

- Try a slightly smaller `MAX_MAMBA_CACHE_SIZE` (e.g., 160 → 144/152/160).
- Or reduce `CONTEXT_LENGTH` (e.g., 16k → 12k) and retry.

### Run windowed bench

```
CONCURRENCY=96  TOTAL=640 scripts/bench/run_window_bench.sh
CONCURRENCY=112 TOTAL=640 scripts/bench/run_window_bench.sh
CONCURRENCY=128 TOTAL=640 scripts/bench/run_window_bench.sh
CONCURRENCY=160 TOTAL=640 scripts/bench/run_window_bench.sh
```

Analyze:

```
python3 scripts/bench/analyze_window.py <bench_dir> --out <bench_dir>/tokens_report.json
```

### MoE Expert Trace (per‑token) quick start

Enable recorder + JSONL writer at launch. Files are written to `/telemetry/expert-trace` in the helper and mapped to the host under `$HOME/sglang-observability/telemetry/expert-trace`.

```
EXPERT_DISTRIBUTION_RECORDER_MODE=per_token \
SGLANG_MOE_TRACE_DIR=/telemetry/expert-trace \
SGLANG_MOE_TRACE_PHASE=all \
SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC=5 \
ENABLE_TRACE=1 OTEL_TRACES_SAMPLER=always_on \
MEM_FRACTION_STATIC=0.98 CONTEXT_LENGTH=16384 MAX_TOTAL_TOKENS=16384 MAX_PREFILL_TOKENS=16384 \
MAX_MAMBA_CACHE_SIZE=160 READY_TIMEOUT=600 \
./scripts/infer/start_server.sh

# Drive a short decode, then flush and list
curl -sf -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"List three prime numbers."}],"temperature":0.2,"max_tokens":8}' >/dev/null
curl -sf http://127.0.0.1:30000/dump_expert_distribution_record >/dev/null
ls -1t $HOME/sglang-observability/telemetry/expert-trace/expert_trace_* | head -1
```

Troubleshooting (empty JSONL):

- Force STANDARD Top‑K: `SGLANG_FORCE_STANDARD_TOPK=1` and run with `SGLANG_EXTRA_ARGS="--moe-runner-backend standard"`.
- Keep Triton and enable decode‑time taps:
  - `SGLANG_MOE_TRACE_DECODE_FROM_RUNNER=1` (runner tap), `SGLANG_MOE_TRACE_FROM_DISPATCH=1` (EP=1 backstop).
  - Optional: `SGLANG_MOE_TRACE_TOPK=11` to reflect 10+1 shared in summaries.
- Ensure `/telemetry/expert-trace` is writable by the `devuser` inside the container.
- Verify CLI flags reached the server: look for `[trace_debug] argv=...` in `observability.log`.

### HBM min/mean/max during windows

`<bench_dir>/samples.csv` has `fb_used_MiB` for quick min/mean/max (now always numeric).

## Practical Guidance

- Keep context low (16k) to maximize parallelism; per‑slot slope rises with context.
- Use `--mamba-ssm-dtype bfloat16` to reduce Mamba per‑slot footprint materially.
- Use `mem_fraction_static≈0.98` to robustly admit large slot caps; it doesn’t reduce per‑slot, it avoids negative budgets at startup.
- Expect a short post‑startup HBM spike; size headroom accordingly.
- Keep telemetry on while characterizing; disable after the baseline is locked to quantify overhead.

## Files of Interest

- scripts/start_observable_container.sh
- scripts/infer/start_server.sh (supports `SGLANG_EXTRA_ARGS`)
- .devcontainer/tools/sglang-config.json (defaults)
- scripts/bench/run_window_bench.sh (CSV fixed for numeric FB_USED etc.)
- scripts/bench/analyze_window.py (tokens_report.json)

## Known Rough Edges

- Very high `MAX_MAMBA_CACHE_SIZE` can fail at startup even with low context if the profiler computes a negative token budget. Reduce `MAX_MAMBA_CACHE_SIZE` or context a notch; or temporarily increase `mem_fraction_static` during admission testing.
- cuda‑graph capture can transiently lift HBM; use a short window warmup before critical measurements.

## Prefill‑Heavy + Mixed Workloads (New Findings)

This section captures everything we learned while extending the baseline beyond the original decode‑heavy experiments.

### Additions at a glance

- Prefill‑heavy windows: larger prefill chunks at C=16 markedly improve throughput.
- Mixed windows (prefill ~15.9k + decode ~256): aggregate TPS is dominated by prefill; overlap/“aggressiveness” knobs have minimal impact at this mix.
- Chunk sweeps: documented wall TPS and stage TPS vs. chunk size.
- Admission/warmup guidance for very large contexts (64k/128k/256k).
- Runner ergonomics: multi‑payload mixes via `BENCH_PAYLOAD_PATHS` and `http_bench.py --bodies`.

### Prefill‑only (C=16, ctx=16k): chunked prefill sweep

- Setup: LongBench v2 context (~15,875 tokens; Qwen tokenizer). Caps: ctx=16k; `MAX_*_TOKENS=16k`; KV fp8; Mamba SSM bf16.
- Prefill wall TPS (single windows):
  - chunk 2k → ≈ 14,327
  - chunk 8k → ≈ 23,482
  - chunk 12k → ≈ 25,047
  - chunk 16k → ≈ 32,830
- Stage TPS rises with chunk size; larger chunks reduce scheduler/launch overhead.
- With ctx raised to 32k for the same 16k prompt (chunk 16k), throughput stayed similar/slightly higher (~33,321). Compute is driven by actual tokens and chunk shape; the cap mainly affects admission.

### Mixed (50/50 requests: prefill ~15.9k; decode ~256), ctx=16k

- Aggregate wall TPS ≈ 11.9–12.1k across C=64..128; decode contributes ~196–198 TPS.
- Marginal gains with concurrency are <5%; practical knee ~96–112 under a p95 e2e SLO.
- Overlap (`--enable-two-batch-overlap`) and `--schedule-conservativeness ∈ {0.1,0.6,0.9}` did not move aggregate TPS at this mix.
- Tiny chunks (1k) halve aggregate TPS (~6.3–6.5k) and lengthen windows.

#### Additional observations (offline vs. online; practical knobs)

- “Prefill” in our metrics is input tokens (delta of `sglang:prompt_tokens_total`).
- Offline harnesses differ from the online window in scheduling and runtime overhead:
  - `bench_offline_throughput --backend runtime` re-tokenizes prompts and may apply chat defaults; this can inflate token counts and trigger context guardrails if inputs are too close to the cap. Use datasets that explicitly control length (e.g., generated-shared-prefix) or add a margin below the cap.
  - To mimic the online mixed window offline, set `--chunked-prefill-size 16384`, `--max-running-requests 96`, and use long-prefix datasets; expect decode rates (e.g., ~170 tok/s internal) to align in ballpark with online stage decode TPS, while aggregate prefill tok/s may differ due to the different loops.
- Sizing: set `MAX_MAMBA_CACHE_SIZE` near your target concurrency (e.g., 96 for C≈96) to avoid wasting memory.
- Readiness gates: enable tracing and `HELLO_AFTER_READY=1` (or a stricter `VERIFY_INFERENCE` if adopted) to ensure first-inference succeeds post‑startup.

### Big prompts (assets + procedure)

Payloads (raw context, Qwen tokenizer) live under `benchmarks-local/prompts/lb2_qwen3next` with a `manifest.json` detailing ids and token counts:

- 32k (31,754): `lb2_qwen3next_32k_66ee8bab....json`
- 64k (64,605): `lb2_qwen3next_64k_671b170c....json`
- 128k (130,055): `lb2_qwen3next_128k_66f568dc....json`
- 256k (255,047): `lb2_qwen3next_256k_66f2b546....json`

Recommended runs (prefill‑only, C=16, one pass each):

- For each target N ∈ {64k, 128k, 256k}:
  - Set `CONTEXT_LENGTH=MAX_*_TOKENS=N` and `CHUNKED_PREFILL_SIZE=N` (full‑chunk prefill).
  - If startup OOMs during CUDA graph capture/deep_gemm warmups, first lower `mem_fraction_static` (e.g., 0.98→0.94/0.92), and/or add `--cuda-graph-max-bs` (e.g., 16/8). If still failing, reduce `MAX_MAMBA_CACHE_SIZE` (160→96→64).
  - Run a single window with `TOTAL=16`.
  - Analyze with `scripts/bench/analyze_window.py` and record `tps_wall.prefill`, `tps_stage.prefill`, and `window_seconds`.

These headroom tweaks are admission‑only; they do not change steady‑state per‑token slope.

### Runner enhancements (multi‑payload and BENCH_ROOT‑only)

- `scripts/bench/http_bench.py` supports `--bodies path1,path2,...` (round‑robin) for mixed workloads.
- `scripts/bench/run_window_bench.sh`:
  - `BENCH_PAYLOAD_PATHS` (comma‑separated) and `BENCH_PAYLOAD_PATH` (single) select payloads.
  - `BENCH_ROOT` lets you run fully in‑container without a manifest (writes directly to the provided path).

### Practical guidance (updated)

- Prefill‑heavy at fixed C (e.g., 16): use the largest `chunked_prefill_size` admission allows; for single long prompts, full‑chunk prefill yields the best efficiency.
- Mixed at 16k: if you want overlap/aggressiveness to matter, increase decode’s share (more tokens or higher mix weight). Otherwise prefill dominates and concurrency looks flat.
- Keep bench artifacts organized via `BENCH_ROOT`; collect `sglang_before/after.prom`, `samples.csv`, `http_summary.json`, `tokens_report.json` per window.

## Code changes to the bench harness (context for upstreaming)

We made small, targeted improvements for repeatability and multi‑payload mixes:

- `scripts/bench/run_window_bench.sh`
  - Added `BENCH_PAYLOAD_PATH` (single) and `BENCH_PAYLOAD_PATHS` (comma‑separated) to select request bodies and copy them into the bench directory.
  - Added `BENCH_ROOT` to write benchmarks directly under a provided path, bypassing manifest resolution (useful inside the helper container).
  - CSV numeric extraction fix for DCGM/node metrics: pipe metric lines through `awk '{print $NF}'` to guarantee numeric fields in `samples.csv`.

- `scripts/bench/http_bench.py`
  - Added mutually exclusive `--body` (single) and `--bodies` (multi) flags; in multi mode, requests round‑robin the bodies list.
  - Added `--outdir` (explicit), writes `meta_*.txt`, `out_*.json`, and a compact `http_summary.json` with `window_seconds`.
  - Tightened aiohttp client behavior (timeouts, larger read buffer) for high‑concurrency windows.

Upstream suitability:

- The `http_bench.py --bodies` support and `--outdir` improvements are broadly useful and low‑risk to upstream as a focused PR.
- The CSV numeric fix in the runner and the BENCH_ROOT/payload‑copy ergonomics are specific to this repo’s helper workflow but can be proposed as optional quality‑of‑life improvements.
- A follow‑up improvement we’d propose upstream: in `bench_offline_throughput`, treat non‑JSON error responses as failures in aggregation instead of raising, and offer an explicit length guard for runtime backend to prevent context‑overflow requests.
