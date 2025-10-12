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
  - Starts `sglang.launch_server` inside the running container as `devuser`.
  - Health checks `http://127.0.0.1:30000/get_model_info` until ready.
  - Prints a small JSON status with `run_id`, `health`, `port`, `manifest_host_path`,
    `log_file`, and `started_at_iso`.

- `scripts/infer/status.sh`
  - Prints `ready|starting|down` based on the same health endpoint.

- `scripts/infer/stop_server.sh`
  - Stops the server process in the container and waits until the port is free.

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
  "stop_reason": "stop|length|abort",
  "usage": {"prompt_tokens": 312, "completion_tokens": 420},
  "timings": {"start_ts": "…Z", "end_ts": "…Z", "total_latency_ms": 1842},
  "request_snapshot": {
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

## Filesystem Layout (artifacts)

Artifacts for each test are written under the active run directory reported by
the manifest pointer:

```
$HOME/sglang-observability/telemetry/container_runs/
  <CONTAINER_RUN_ID>/
    inference/<test_id>/
      transcript.json  # raw + split
      metrics.json     # usage + timings + status
```

