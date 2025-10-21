# Curriculum: Preparing to Instrument MoE Router Decisions (Qwen3‑Next)

This study plan reflects the current code and launcher behavior. It replaces older notes and removes prior cruft. Follow it in order.

Important ground rules (read‑only phase)
- Do not modify files or commit changes while studying.
- Do not execute tests.
- Your goal is to understand how the helper, launcher, observability surfaces, and harnesses work; how to discover and analyze artifacts; and where/what to instrument for MoE router decisions (top‑k experts).
- After this study, you’ll be quizzed. Only after passing the follow‑up questions will you be allowed to make code changes or run tests.

## 0) Orientation (10–15 min)
Read:
- `python/sglang/README.md`
- `docs/perf_baseline_gh200.md`
- `docs/inference_harness.md`
- `docs/observability_tracing_ops.md` (see “Expert Trace (MoE)”).

## 1) Observability helper & topology (20–30 min)
Files: `.devcontainer/README.md`, `scripts/start_observable_container.sh`, `.devcontainer/observability/init-run.sh`.
Learn:
- Components/ports: Jaeger v2 (UI 16686, OTLP 4317/4318), Prometheus (9090), node_exporter (9100), dcgm‑exporter (9400).
- Pointer → manifest: `$HOME/sglang-observability/telemetry/container_run_meta.env` → `.../container_runs/<RUN_ID>/manifest.json`.
- Session records on server ready: `logs/provider_sessions/<ISO>_<SESSION>/start.json`.
- Trace dir mapping: container `/telemetry/expert-trace` → host `$HOME/sglang-observability/telemetry/expert-trace`.
Self‑check: list the ports, and where Prom/Jaeger TSDBs land.

## 2) Server lifecycle & readiness (20–30 min)
Files: `scripts/infer/start_server.sh`, `scripts/infer/status.sh|stop_server.sh|session_info.sh`, `python/sglang/srt/entrypoints/http_server.py` (`/trace_probe`).
Learn:
- Env→CLI basics: `MEM_FRACTION_STATIC`, `CONTEXT_LENGTH`, `MAX_{PREFILL,TOTAL}_TOKENS`, `MAX_MAMBA_CACHE_SIZE`, `KV_CACHE_DTYPE`.
- Tracing: `ENABLE_TRACE=1`; `OTEL_TRACES_SAMPLER`; resource attrs `container_run`, `service.instance.id`.
- MoE recorder + writer (forwarded by the launcher):
  - `EXPERT_DISTRIBUTION_RECORDER_MODE` → `--expert-distribution-recorder-mode`.
  - `SGLANG_MOE_TRACE_DIR`, `SGLANG_MOE_TRACE_PHASE`, `SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC`.
  - Diagnostic: `SGLANG_FORCE_STANDARD_TOPK=1` to force STANDARD top‑k and guarantee emission.
- Readiness gates: `/get_model_info` returns 200; `/trace_probe` shows `{"tracing_enabled": true, "threads_registered": >0}`.
Self‑check: identify `port`, `run_id`, `session_record_host_path`, and the `sizing` block from a ready JSON.

## 3) Online window bench + analyzer (25–35 min)
Files: `scripts/bench/run_window_bench.sh`, `scripts/bench/http_bench.py`, `scripts/bench/analyze_window.py`.
Learn:
- Flow: capture `/metrics` before/after, sample node/DCGM 1 Hz to `samples.csv`, run workload, produce `http_summary.json` and `tokens_report.json`.
- CSV numeric fix (`awk '{print $NF}'`).
- `BENCH_ROOT` / `BENCH_PAYLOAD_PATH{,S}` and `http_bench.py --body|--bodies`.
- Analyzer math: counter deltas; hist sums; wall vs. stage TPS.
Self‑check: explain wall vs. stage TPS and exact metric names used.

## 4) Offline benches & datasets (20–30 min)
Files/docs: `python/sglang/bench_offline_throughput.py`, docs (Offline Benchmarks).
Learn:
- `engine` vs `runtime`; runtime may inflate token counts.
- random / ShareGPT / GSP datasets and sizing pitfalls; keep GSP system prompt len a few hundred under cap.
Self‑check: why 15,900 can overflow 16k, and which flag avoids it.

## 5) Tracing & metrics surfaces (15–20 min)
Files: `docs/observability_tracing_ops.md`, `test/srt/test_trace_thread_registration.py`.
Learn:
- Jaeger v2: UI 16686; API v3; smoke-span proof.
- `/trace_probe` reflects tracer state; auto thread registration matters.
- Prom metrics for token counts and latencies.
Self‑check: filter a session in Jaeger by resource attrs.

## 6) Memory model & admission (30–40 min)
Files: `python/sglang/srt/model_executor/model_runner.py`, `python/sglang/srt/configs/mamba_utils.py`, `python/sglang/srt/configs/qwen3_next.py`.
Learn: `profile_max_num_token` and KV cell sizes; pool init guardrails; why `MAX_MAMBA_CACHE_SIZE≈concurrency` avoids waste.
Self‑check: run through a 16k ctx example and confirm clamping.

## 7) MoE routing: instrumentation points (30–45 min)
Files: `python/sglang/srt/layers/moe/*`, `python/sglang/srt/eplb/*`, `python/sglang/srt/managers/*`.
Learn:
- Router → logits/probabilities → top‑k selection per token.
- Hooks: `on_select_experts` (STANDARD TopK) and DeepEP dispatch hooks.
- Emission: ExpertTraceWriter JSONLs (per‑step), recorder accumulators (.pt).
- Config: recorder mode, writer envs, optional `SGLANG_FORCE_STANDARD_TOPK=1` to ensure hooks fire; careful with cardinality.
Self‑check: identify where to insert lightweight attributes or metrics with minimal overhead.

## 8) Tests (read‑only)
Files: `test/`.
Learn: where trace and bench tests live and what they assert.

## 9) Expert Trace (prototype, updated) (20–30 min)
Files: `python/sglang/srt/trace/expert_trace.py`, `python/sglang/srt/eplb/expert_distribution.py`, `tools/analyze_expert_trace.py`.
Learn:
- Writer activation via `SGLANG_MOE_TRACE_DIR`.
- Hooks: `on_forward_pass_start`, `handle_on_select_experts`, `flush`.
- File naming: `expert_trace_<run>_<session>_rankXXXXX.jsonl`.
- Analyzer: per‑layer totals and top‑N experts; new flags `--phase`, `--dedupe`, `--step-window` help verify decode coverage.
Self‑check: where slots/positions/seq_lens come from (ForwardBatch).

## 10) Qwen3‑Next decode nuances (5–10 min)
File: `python/sglang/srt/models/qwen3_next.py`.
Learn:
- `get_model_config_for_expert_location(...)` is present; per_token recorder aligned to top‑10.
- Some decode backends (e.g., Triton fused, EP=1) may bypass Python Top‑K hooks. Use env‑gated taps:
  - `SGLANG_MOE_TRACE_DECODE_FROM_RUNNER=1` to emit from the decode runner, and
  - `SGLANG_MOE_TRACE_FROM_DISPATCH=1` to emit from the EP=1 standard dispatcher.
- Verify CLI backend flags with the server’s argv log.
Self‑check: explain the assertion if a model lacks this hook.

---

Reminder: During this read‑only phase, do not modify files or run tests. Focus on understanding and be ready to answer the follow‑up mastery questions before making any changes.
