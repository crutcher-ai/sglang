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
