# Inference Harness (Chat Completions Only)

This document describes the helpers and result schema for running one‑shot and
multi‑turn inference against a single SGLang server (one container → one
server). The harness uses OpenAI‑compatible Chat Completions exclusively.

## Scope

- Transport: `/v1/chat/completions` only.
- Models: Qwen3‑Next‑80B‑A3B‑Thinking‑FP8 (thinking always on) and Instruct
  counterpart (never thinking). No server‑side toggle is attempted.
- Thinking handling: Split `<think>…</think>` in the returned assistant text
  (Qwen Next may emit closing tag `</think>` only). For continuation, include
  only the final assistant content in history; never re‑send thinking.
- Metrics: Return only server‑reported usage (prompt_tokens, completion_tokens)
  and client wall‑time latencies. Deeper accounting and charts should be
  queried from Prometheus using the time window we return.

## Helpers

- `scripts/infer/start_server.sh`
  - Starts `sglang.launch_server` inside the helper container.
  - Probes `http://127.0.0.1:30000/get_model_info` until ready.
  - On ready, prints one JSON line to stdout and writes the exact JSON atomically to
    `$HOME/sglang-observability/telemetry/container_runs/<RUN_ID>/logs/provider_sessions/<ISO>_<SESSION>/start.json`.
  - Recorder + Expert Trace (MoE) envs forwarded to the model process:
    - `EXPERT_DISTRIBUTION_RECORDER_MODE` → `--expert-distribution-recorder-mode <per_token|stat|per_pass|stat_approx>`
    - `SGLANG_MOE_TRACE_DIR` (enables JSONL writer when set)
    - `SGLANG_MOE_TRACE_PHASE` (`decode|prefill|both|all`)
    - `SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC` (seconds)
    - Diagnostic: `SGLANG_FORCE_STANDARD_TOPK=1` forces STANDARD top‑k routing path (ensures per‑token on_select_experts hooks fire even when TRITON_KERNEL/BYPASSED fast paths are available).

- `scripts/infer/session_info.sh`
  - Read‑only attach helper. Usage: `session_info.sh --manifest <ABS_MANIFEST_PATH>`.
  - Prints the authoritative session start JSON (newest provider_sessions/*/start.json). No fallbacks.

- `scripts/infer/status.sh`
  - Prints `ready|starting|down` based on the same health endpoint.

- `scripts/infer/stop_server.sh`
  - Stops the server process and waits until the port is free.

### Runner Enhancements (Mixed Payloads + In-Repo Bench Root)

The windowed bench harness supports single or multiple payloads and can write artifacts in-repo without manifest resolution.

- `scripts/bench/http_bench.py`
  - `--body <path>`: single JSON payload.
  - `--bodies <p1,p2,...>`: multiple JSON payloads; requests round-robin across them.

- `scripts/bench/run_window_bench.sh`
  - `BENCH_PAYLOAD_PATH=<path>`: single JSON payload (copied into the bench dir).
  - `BENCH_PAYLOAD_PATHS=<p1,p2,...>`: multiple JSON payloads (copied into the bench dir and passed to `http_bench.py --bodies`).
  - `BENCH_ROOT=<path>`: write the bench directory under this path and skip manifest resolution entirely (useful when running fully inside the helper container or avoiding host path permissions).

Examples

Mixed (round-robin two payloads) at C=64 with a ~60s window:

```
CONCURRENCY=64 TOTAL=80 TIMEOUT=600 BENCH_ROOT=benchmarks-local \
BENCH_PAYLOAD_PATHS=benchmarks-local/prompts/longbench_v2_66f52c6d821e116aacb32cb0_context_only.json,\
benchmarks-local/prompts/decode_heavy_payload.json \
scripts/bench/run_window_bench.sh
python3 scripts/bench/analyze_window.py <bench_dir> --out <bench_dir>/tokens_report.json
```

Prefill-only (single large prompt) at C=16, full-chunk prefill:

```
SGLANG_EXTRA_ARGS="--mamba-ssm-dtype bfloat16" \
CONTEXT_LENGTH=16384 MAX_TOTAL_TOKENS=16384 MAX_PREFILL_TOKENS=16384 \
CHUNKED_PREFILL_SIZE=16384 READY_TIMEOUT=600 PORT=30000 ./scripts/infer/start_server.sh
CONCURRENCY=16 TOTAL=80 TIMEOUT=1200 BENCH_ROOT=benchmarks-local \
BENCH_PAYLOAD_PATH=benchmarks-local/prompts/lb2_qwen3next/lb2_qwen3next_32k_66ee8bab821e116aacb21e44.json \
scripts/bench/run_window_bench.sh
python3 scripts/bench/analyze_window.py <bench_dir> --out <bench_dir>/tokens_report.json
```

Very large contexts (64k/128k/256k): admission and capture tuning

- Add `--cuda-graph-max-bs` via `SGLANG_EXTRA_ARGS` (e.g., 16 for 64k/128k, 8 for 256k) and lower `MEM_FRACTION_STATIC` slightly (e.g., 0.94/0.92) to clear CUDA graph warmups without changing steady-state slope. If admission still fails, reduce `MAX_MAMBA_CACHE_SIZE` in small steps (160→96→64) before shrinking chunk size.

### Quick MoE Trace (per‑token) example

Start the server with a real recorder and the ExpertTraceWriter enabled. The trace directory lives inside the helper at `/telemetry/expert-trace` and maps to the host path `$HOME/sglang-observability/telemetry/expert-trace`.

```
EXPERT_DISTRIBUTION_RECORDER_MODE=per_token \
SGLANG_MOE_TRACE_DIR=/telemetry/expert-trace \
SGLANG_MOE_TRACE_PHASE=decode \
SGLANG_MOE_TRACE_FLUSH_INTERVAL_SEC=5 \
ENABLE_TRACE=1 OTEL_TRACES_SAMPLER=always_on \
MEM_FRACTION_STATIC=0.98 CONTEXT_LENGTH=16384 MAX_TOTAL_TOKENS=16384 MAX_PREFILL_TOKENS=16384 \
MAX_MAMBA_CACHE_SIZE=160 READY_TIMEOUT=600 \
./scripts/infer/start_server.sh
```

Drive a small decode (completion tokens > 0) and flush:

```
curl -sf -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"List three prime numbers."}],"temperature":0.2,"max_tokens":8}' >/dev/null
curl -sf http://127.0.0.1:30000/dump_expert_distribution_record >/dev/null
```

List and inspect JSONL on the host:

```
ls -1t $HOME/sglang-observability/telemetry/expert-trace/expert_trace_* | head -1
head -n 2 $(ls -1t $HOME/sglang-observability/telemetry/expert-trace/expert_trace_* | head -1)
```

Analyzer (already in-tree):

```
python3 tools/analyze_expert_trace.py --dir $HOME/sglang-observability/telemetry/expert-trace --top 10
```

If the JSONL is empty (0‑byte), force the STANDARD top‑k path to ensure hooks fire:

```
SGLANG_FORCE_STANDARD_TOPK=1 SGLANG_EXTRA_ARGS="--moe-runner-backend triton" \
EXPERT_DISTRIBUTION_RECORDER_MODE=per_token SGLANG_MOE_TRACE_DIR=/telemetry/expert-trace \
./scripts/infer/start_server.sh
```

## Server Ready JSON (start_server.sh)

On readiness, the server launcher prints a single JSON line and writes the same JSON atomically to the session directory:

```
{
  "schema_version": 1,
  "run_id": "container-run-…",
  "server_session_id": "srv-…",
  "health": "ready",
  "port": 30000,
  "manifest_host_path": "/abs/path/to/manifest.json",
  "log_file": "/telemetry/logs/…/observability.log",
  "started_at_iso": "…Z",
  "session_record_host_path": "/abs/path/to/logs/provider_sessions/<ISO>_<SESSION>/start.json",
  "sizing": {
    "tp_size": 1,
    "mem_fraction_static": 0.94,
    "chunked_prefill_size": 4096,
    "context_length": 262144,
    "max_prefill_tokens": 262144,
    "max_total_tokens": 262144,
    "max_mamba_cache_size": 1
  }
}
```

Attach flows must obtain this JSON via `scripts/infer/session_info.sh`.

- `tools/infer_client.py`
  - One‑shot request runner (Chat Completions). Builds messages from
    `--system/--context/--prompt` and sampling params.
  - Splits reasoning/content by the last `</think>` if present; otherwise treats
    the full text as content.
  - Prints a compact JSON result and writes artifacts to the current run folder.

- `tools/infer_runner.py`
  - Executes a scenario of many tests (JSON or YAML). Multi‑turn conversations
    apply “content‑only continuation”.

## Return Schema (One‑Shot)

```
{
  "test_id": "os1",
  "status": "ok|http_error|timeout|schema_error",
  "http_status": 200,
  "stop_reason": "stop|length|…",  # from OpenAI finish_reason
  "usage": {"prompt_tokens": 312, "completion_tokens": 420},
  "timings": {"start_ts": "…Z", "end_ts": "…Z", "total_latency_ms": 1842},
  "request_snapshot": {
    "base_url": "http://127.0.0.1:30000/v1",
    "model_id": "local",
    "system": "…", "context": "…", "prompt": "…",
    "sampling": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 1024},
    "thinking_hint": "qwen-thinking|null"
  },
  "response_snapshot": {
    "assistant_text_raw": "... </think> Final ...",
    "assistant_reasoning_text": "… or null …",
    "assistant_content_text": "Final …"
  },
  "prom_bookmark": {
    "container_run_id": "container-run-…",
    "window": {"start_ts": "…Z", "end_ts": "…Z"}
  },
  "container_log_anchor": {"path": "…/logs/…log", "window": {"start_ts": "…Z", "end_ts": "…Z"}}
}
```

## Defaults

- Thinking model: `temperature=0.6`, `top_p=0.95`, generous `max_tokens`.
- Instruct model: `temperature=0.7`, `top_p=0.8`.
- No “reasoning budget” knob is set; none is available in open‑source stacks.

## Notes

- When the server provides a split field (e.g., `message.reasoning_content`),
  the client prefers that over text splitting by `</think>`.
- The health/networking model assumes the container is reachable on
  `127.0.0.1:<PORT>` (host networking or an equivalent mapping).

## Filesystem Layout (artifacts)

Artifacts for each test are written under the active run directory reported by
the manifest pointer:

```
$HOME/sglang-observability/telemetry/container_runs/
  <CONTAINER_RUN_ID>/
    inference/<test_id>/
      transcript.json  # raw + split
      metrics.json     # usage + timings + status

## Notes on Tokenization and Prompt Sizing

 - This repository’s Qwen3-Next model should be counted with the Hugging Face AutoTokenizer
   (`Qwen/Qwen3-Next-80B-A3B-Thinking-FP8`, `trust_remote_code=True`). Do not use `cl100k_base` unless the server is explicitly configured with a tiktoken JSON tokenizer.
 - When selecting large prompts (e.g., LongBench v2 contexts), count the exact text you will send to `/generate` (raw context only for prefill tests; no chat template). Curated payloads and a manifest of Qwen-token counts are available under `benchmarks-local/prompts/lb2_qwen3next/`.
```

## Offline Benchmarks (engine/runtime) — Practical Notes

SGLang includes an offline throughput harness (`python -m sglang.bench_offline_throughput`) that can mimic many online workloads without HTTP.

- Backends:
  - `--backend engine` drives the engine directly (no HTTP runtime loop).
  - `--backend runtime` spins up a lightweight HTTP runtime internally and drives requests against it (re-tokenization and chat-template defaults can inflate token counts).

- Fixed lengths with the "random" dataset: set `--random-range-ratio 1.0` to make `--random-input-len N` and `--random-output-len M` exact. With `--random-range-ratio 0.0`, the harness samples uniformly from `[1..N]` and `[0..M]`.

- ShareGPT dataset: `--sharegpt-output-len` is an enforced cap; `--sharegpt-context-len` prunes prompts that would overflow the cap, but it does not enforce a minimum prompt length.

- Generated Shared Prefix (GSP): `--dataset-name generated-shared-prefix` with `--gsp-system-prompt-len` (long prefix), `--gsp-question-len` (short), and `--gsp-output-len` (short decode) produces prefill-heavy synthetic requests. For 16k caps, set `gsp-system-prompt-len` a few hundred below 16k (e.g., 15000) to avoid post-template inflation.

- Mirroring an online mixed window (e.g., ~15.9k prefill + 256 decode, high concurrency) offline:
  - Use GSP or curated long prompts; set `--chunked-prefill-size 16384` and `--max-running-requests 96` with the same 16k caps and FP8 KV + bf16 SSM settings. Expect differences vs. online due to scheduling/runtime overheads.

Example commands (inside the helper container):

```
# Fixed 2k/2k, runtime backend
/sgl-workspace/sglang/.venv/bin/python -m sglang.bench_offline_throughput \
  --backend runtime \
  --model-path /models/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 \
  --dataset-name random --num-prompts 64 \
  --random-input-len 2048 --random-output-len 2048 --random-range-ratio 1.0 \
  --context-length 16384 --max-total-tokens 16384 --max-prefill-tokens 16384 \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 \
  --mem-fraction-static 0.98 --max-mamba-cache-size 160 \
  --result-filename /sgl-workspace/sglang/results/offline_2k2k_runtime.jsonl

# Prefill-heavy (GSP), 16k caps, chunk=16k, concurrency≈96
/sgl-workspace/sglang/.venv/bin/python -m sglang.bench_offline_throughput \
  --backend runtime \
  --model-path /models/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 \
  --dataset-name generated-shared-prefix \
  --gsp-num-groups 64 --gsp-prompts-per-group 10 \
  --gsp-system-prompt-len 15000 --gsp-question-len 64 --gsp-output-len 256 \
  --context-length 16384 --max-total-tokens 16384 --max-prefill-tokens 16384 \
  --chunked-prefill-size 16384 --max-running-requests 96 \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 \
  --mem-fraction-static 0.98 --max-mamba-cache-size 96 \
  --result-filename /sgl-workspace/sglang/results/offline_mixedlike_gsp_runtime.jsonl
```

Troubleshooting: if the runtime backend emits HTTP 400 “longer than context length”, reduce the target input length (or enable `--sharegpt-context-len`), then re-run. If any requests fail, fix lengths first; otherwise the summarizer may error while aggregating results.
