# Tracing Migration Plan — Upstream SGLang + Jaeger v2 (Dev/CI)

Goal
- Replace legacy tracing with upstream SGLang tracing and Jaeger v2 (OTLP‑native) so traces persist and are queryable by service + time window.
- Do not modify SGLang code. Adapt our helpers and SLICE‑Bench only.

Scope (this change)
- Start/ensure Jaeger v2 on host (OTLP gRPC 4317, HTTP 4318, UI 16686) with Badger persistence.
- Launch SGLang with upstream flags: `--enable-trace` and `--oltp-traces-endpoint <host:port>`.
- SLICE‑Bench queries Jaeger API v3 using RFC‑3339 time bounds; authoritative export is service + window (no run‑scoped filter initially).
- Defer run‑scoped attributes (e.g., container_run) until ingestion is consistently green.

Out of scope
- No Jaeger v1 usage or fallback.
- No per‑run span/resource tagging in this pass (we’ll add later once pipeline is green).

Repository moves
1) sglang/scripts/start_observable_container.sh
   - Ensure a `jaeger-v2` container (host) is running with ports 16686/4317/4318 and run-scoped Badger bind mounts under `$HOST_TELEMETRY_ROOT/container_runs/<RUN_ID>/jaeger/badger`.
   - Pre-create the per-run host directory tree (`logs/`, `prometheus/`, `metrics/`, `configs/`, `jaeger/…`) before launching the helper container; copy the Jaeger config into `configs/jaeger.yaml` for later audit.
   - Keep Prometheus/node_exporter/dcgm-exporter behavior unchanged.
   - In `VALIDATE_ONLY=1` mode, do not launch Jaeger or create run-specific directories.

2) sglang/.devcontainer/observability/init-run.sh
   - Stop launching `jaeger-all-in-one` v1 inside the helper container.
   - Stage all telemetry under `/telemetry/container_runs/<RUN_ID>/…` and export the same structure to host `/home/<user>/sglang-observability/telemetry/container_runs/<RUN_ID>/…` (manifest, logs, metrics, configs, Prometheus TSDB, Jaeger Badger store).
   - Manifest JSON points to the per-run paths (container + host) so SLICE-Bench collectors can ingest without additional lookup.

3) sglang/scripts/infer/start_server.sh
   - Map env → upstream flags only:
     - If `ENABLE_TRACE=1` (or `SGLANG_ENABLE_TRACE=1`): add `--enable-trace`.
     - Derive endpoint for `--oltp-traces-endpoint`:
       - If `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` present, strip `http(s)://` and trailing paths to get `host:port`.
       - Else default `127.0.0.1:4317` (host‑network helper hitting Jaeger v2 published port).
   - Export advisory standard OTel envs:
     - `OTEL_SERVICE_NAME=sglang`, `OTEL_TRACES_SAMPLER=${OTEL_TRACES_SAMPLER:-always_on}`; optional `OTEL_LOG_LEVEL`.
   - Remove custom run attrs (`OTEL_RESOURCE_ATTRIBUTES` with container_run, `SGL_CONTAINER_RUN_ID`).

4) SLICE‑Bench/src/slice_bench/integration/telemetry.py
   - Switch to Jaeger API v3 endpoints:
     - Services: `GET /api/v3/services`.
     - Traces: `GET /api/v3/traces` with params on the nested `query.*` fields:
       - `query.service_name=<service>`
       - `query.start_time_min=<RFC‑3339 UTC>`
       - `query.start_time_max=<RFC‑3339 UTC>`
       - `query.search_depth=<max_traces or sane default>`
   - Write artefacts:
     - `telemetry/jaeger_services.json` (probe)
     - `telemetry/jaeger_probe.json` (service‑only, time‑bounded probe)
     - `telemetry/jaeger_traces.json` (authoritative, service + window)
   - Do not add run attribute filters initially (revisit once resource tagging added).

5) SLICE‑Bench/tests/test_telemetry_reporting.py
   - Adjust mocked endpoints to `/api/v3/services` and `/api/v3/traces`.
   - Keep trace ID extraction minimal for tests (fake payloads should mirror OTLP JSON: `{"result": {"resourceSpans": [...]}}`).

Operator flow (post‑change)
1. Start observability helper (ensures Jaeger v2):
   - `./scripts/start_observable_container.sh`
   - Output includes: `Jaeger v2 ready: UI http://localhost:16686 OTLP gRPC 127.0.0.1:4317`
2. Start SGLang server with tracing:
   - `ENABLE_TRACE=1 OTEL_TRACES_SAMPLER=always_on ./scripts/infer/start_server.sh`
3. Run SLICE‑Bench pack; collector saves:
   - `telemetry/jaeger_services.json`, `telemetry/jaeger_probe.json`, `telemetry/jaeger_traces.json`.
4. Verify Jaeger UI at http://localhost:16686 (pick service `sglang`).

Verification gates (green)
- Jaeger services include `sglang`.
- Probe and authoritative JSON have non‑empty `result.resourceSpans` arrays.
- SGLang log has no OTel import/connection errors.

Deferred (phase 2)
- Add `service.instance.id=<RUN_ID>` via `OTEL_RESOURCE_ATTRIBUTES` and (optionally) `run_id` span attr on root spans.
- Re‑enable strict run filter in SLICE‑Bench (attributes filter) once tagging is in.

Risks & mitigations
- Port conflicts on 4317/4318/16686 → make ports configurable via env; document defaults.
- Badger dir permissions → create host dirs and verify writability before launching Jaeger.
- API shape differences in v3 → we preserve full payloads to JSON; tests use minimal fakes.
