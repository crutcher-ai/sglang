# Tracing Debug Session Report — 2025-10-16

## Conversation Timeline (Abbreviated)

### 1. Context Review
- Read `.devcontainer/README.md`, `scripts/start_observable_container.sh`, and `scripts/infer/start_server.sh` to understand the helper stack (Prometheus, Jaeger v2, storage layout, session manifests) and the tracing CLI/environment contract.

### 2. Baseline Reproduction (No Instrumentation)
- Ran `./scripts/start_observable_container.sh` (defaults; host networking, Jaeger on 16686/4317).
- Started the server: `ENABLE_TRACE=1 OTEL_TRACES_SAMPLER=always_on SGL_DEBUG=1 ./scripts/infer/start_server.sh`.
- Nudged the server with `./scripts/infer/hello_world.sh 127.0.0.1 30000`.
- Observed:
  - `curl -s http://localhost:16686/api/v3/services -> {"services":[]}` and legacy endpoints empty.
  - `tcpdump -n -i lo port 4317` captured zero packets during the request.
  - `/home/ubuntu/sglang-observability/.../logs/observability.log` showed tracing config emitted, but no `[trace_debug]` breadcrumbs from `trace_req_start`.

### 3. Instrumentation Pass #1
- Added `_debug_log` scaffolding in `python/sglang/srt/tracing/trace.py` and lightweight logging in HTTP/tokenizer/scheduler entrypoints to emit `[trace_debug] …` lines when tracing hooks fire.
- Restarted helper + server; behaviour unchanged (Jaeger empty) but logs now confirmed only the scheduler process ran `process_tracing_init`.

### 4. Instrumentation Pass #2 (File-Based Probe)
- Introduced `_trace_probe` helper writing to `/telemetry/trace_probe.log` in:
  - `python/sglang/srt/tracing/trace.py`
  - `python/sglang/srt/entrypoints/http_server.py`
  - `python/sglang/srt/managers/tokenizer_manager.py`
  - `python/sglang/srt/managers/scheduler.py`
- Modified `scripts/infer/start_server.sh` to propagate `SGL_DEBUG`, carry `server_session_id`, and extend readiness JSON logging.
- Extended READY_TIMEOUT to 600 in some runs to avoid premature abort.
- Initial probe runs still showed only scheduler initialising tracing; HTTP branch silent.

### 5. Successful Trace Emission (Run ID `container-run-20251016T205746Z-c584b0d9`)
- Command sequence:
  1. `./scripts/start_observable_container.sh`
  2. `READY_TIMEOUT=600 ENABLE_TRACE=1 OTEL_TRACES_SAMPLER=always_on SGL_DEBUG=1 ./scripts/infer/start_server.sh`
  3. After readiness JSON printed, `./scripts/infer/hello_world.sh 127.0.0.1 30000`
- Evidence captured:
  - `/home/ubuntu/sglang-observability/telemetry/trace_probe.log` lines showing `http_single_init`, `process_tracing_init success`, `trace_set_thread_info`, `trace_req_start span_created`.
  - `curl -s http://localhost:16686/api/v3/services -> {"services":["sglang"]}`.
  - `curl …/api/v3/traces` returned two `result.resourceSpans` bundles containing spans `Req …`, `tokenize`, `dispatch`, `Scheduler [TP 0] …`, etc.
  - `/trace_probe` endpoint returned `{ "pid":433, "tracing_enabled":true, "threads_registered":1, ... }`.

### 6. Follow-up Attempts & Timeouts
- Additional runs with fresh helper (READY_TIMEOUT=600) occasionally failed to reach readiness (curl refusing connections) because the model load exceeded the timeout. When the helper killed the server early, no trace was emitted—highlighting the need to let warm-up finish (~60–70 s with caches present).

## Tests & Diagnostics (Chronological Highlights)

| Stage | Command / Check | Result |
|-------|-----------------|--------|
| Baseline | `curl -s http://127.0.0.1:16686/api/v3/services` | `{"services":[]}` |
| Baseline | `tcpdump -n -i lo port 4317` during hello_world | No packets |
| Baseline | `docker exec sglang-dev python -c "import sglang.srt.tracing.trace as t; print(t.tracing_enabled)"` | `False` |
| Instrumented | `grep trace_debug observability.log` | Scheduler-only breadcrumbs |
| Probe Run | Inspection of `/telemetry/trace_probe.log` | Scheduler-only entries |
| Successful Run | `/telemetry/trace_probe.log` | HTTP branch + span creation lines |
| Successful Run | `curl … /api/v3/services` | `{"services":["sglang"]}` |
| Successful Run | `curl … /api/v3/traces` | 2 `resourceSpans` with full tree |
| Successful Run | `curl http://127.0.0.1:30000/trace_probe` | `tracing_enabled=true`, `threads_registered=1` |

## Code Modifications (Summary)

The only behavioural additions are instrumentation and metadata logging:

1. **`python/sglang/srt/tracing/trace.py`**
   - Added `_trace_probe` helper writing to `/telemetry/trace_probe.log` (configurable via `TRACE_PROBE_FILE`).
   - `_debug_log` now records each message through both the logger and `_trace_probe`.

2. **`python/sglang/srt/entrypoints/http_server.py`**
   - In both multi-tokenizer and single-tokenizer branches, log entry/exit of tracing init plus try/except around `process_tracing_init` to capture failures.
   - Emit probe lines after `trace_set_thread_info` and when `trace_req_start` is expected to run.
   - Added `/trace_probe` endpoint returning current tracer state (`pid`, `tracing_enabled`, thread count, active probe file).

3. **`python/sglang/srt/managers/tokenizer_manager.py`**
   - Logged constructor state (`enable_trace`, `SGL_DEBUG`).
   - During `generate_request`, log diagnostics captured from `sglang.srt.tracing.trace` (`tracing_enabled`, registered threads) via `_trace_probe`.

4. **`python/sglang/srt/managers/scheduler.py`**
   - Logged when scheduler workers call `process_tracing_init` (endpoint, ranks).

5. **`scripts/infer/start_server.sh`**
   - Switched inline JSON reader to `python3` and added host-fallback for model path.
   - Annotated OTEL resource attributes with `service.instance.id=<SERVER_SESSION_ID>` instead of the run ID, and exported `SGL_SERVER_SESSION_ID` into the container env.
   - Passed `OTEL_TRACES_SAMPLER` / `SGL_DEBUG` into downstream commands.
   - Readiness JSON now includes session ID, run ID, manifest path, etc., and logs a “server ready” breadcrumb to the helper log.

6. **`python/sglang/srt/entrypoints/engine.py` (transient)**
   - A probe variant briefly introduced an indentation error; reverted to upstream formatting. No functional change retained in final state (the only addition kept was the `_trace_probe` helper definition used by other instrumentation).

No business logic or tracer configuration was altered; the probes gather evidence only.

## Successful Run — Reproduction Steps

**Environment**: helper stack (`start_observable_container.sh`) defaults; Jaeger 2.11.0 on host; model `Qwen3-Next-80B-A3B-Thinking-FP8` (from `/models`).

1. **Start helper**
   ```bash
   ./scripts/start_observable_container.sh
   ```
   - Outputs `CONTAINER_RUN_META_JSON_HOST=/home/ubuntu/sglang-observability/telemetry/container_runs/container-run-20251016T205746Z-c584b0d9/manifest.json`.

2. **Launch server**
   ```bash
   READY_TIMEOUT=600 ENABLE_TRACE=1 OTEL_TRACES_SAMPLER=always_on SGL_DEBUG=1 ./scripts/infer/start_server.sh
   ```
   - Load log excerpt (timestamps UTC):
     - `20:58:51` Load weight begin.
     - `20:59:46` Load weight end.
     - `20:59:47` DeepGEMM compile (cached) + CUDA graph capture.
     - `20:59:53` Uvicorn ready; readiness JSON:
       ```json
       {
         "run_id": "container-run-20251016T205746Z-c584b0d9",
         "server_session_id": "srv-779c410e2eeb4232a384f8a8a0e57e3d",
         ...
       }
       ```
   - `/telemetry/trace_probe.log` now contains both scheduler and HTTP tracing entries.

3. **Send request**
   ```bash
   ./scripts/infer/hello_world.sh 127.0.0.1 30000
   ```
   - Logged spans: `trace_req_start span_created rid=529a312b54a14259b7ba35a80775bc17`, etc.

4. **Verify tracing outputs**
   ```bash
   curl -s http://127.0.0.1:16686/api/v3/services
   # -> {"services":["sglang"]}

   START=$(date -u -d '5 minutes ago' +%FT%TZ)
   END=$(date -u +%FT%TZ)
   curl -s --get 'http://127.0.0.1:16686/api/v3/traces' \
     --data-urlencode "query.service_name=sglang" \
     --data-urlencode "query.start_time_min=${START}" \
     --data-urlencode "query.start_time_max=${END}" \
     --data-urlencode "query.search_depth=20" | jq '.result.resourceSpans | length'
   # -> 2
   ```

5. **Optional introspection**
   ```bash
   curl -s http://127.0.0.1:30000/trace_probe
   # {"pid":433,"tracing_enabled":true,"threads_registered":1,...}
   ```

6. **Artifacts**
   - Probe file: `/home/ubuntu/sglang-observability/telemetry/trace_probe.log`.
   - Server log: `/home/ubuntu/sglang-observability/telemetry/container_runs/container-run-20251016T205746Z-c584b0d9/logs/observability.log`.
   - Jaeger Badger storage: `/home/ubuntu/sglang-observability/telemetry/container_runs/container-run-20251016T205746Z-c584b0d9/jaeger/badger/` (contains SSTables after spans arrive).

## Proposed Next Steps

1. **Old Behaviour Reproduction (Control)**
   - Revert instrumentation temporarily (or guard `_trace_probe` behind an env flag) and repeat the successful run procedure. Confirm traces still appear; if they disappear, diff runtime settings to isolate the dependency.

2. **Long Prompt Validation**
   - Issue a longer `/generate` (e.g., `max_new_tokens: 512`) to ensure spans persist for extended workflows, not just 1-token prompts.

3. **Router / Multi-Tokenizer Scenario**
   - Launch with `tokenizer_worker_num > 1` to confirm the router code path still initialises tracing (multi-worker branch now emits probe lines).

4. **Start-Up Determinism**
   - Track warm-up durations by inspecting `Load weight begin/end` & `Capture cuda graph` timestamps. If caches are present, ready time should stay ~60–70 seconds; log anomalies as separate issues.

5. **Remove/Toggle Instrumentation**
   - Once root cause is fully understood and reproducible without probes, strip or behind-feature-flag the `_trace_probe` helpers to avoid log noise in production.

6. **Document Repro Flow**
   - Incorporate the step-by-step procedure (including `/trace_probe` endpoint) into the repo docs so future debugging doesn’t start from scratch.
