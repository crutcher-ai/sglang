# Tracing Ops — Jaeger v2 + Upstream SGLang

## Start Order
- Start the observability helper + Jaeger v2 on host ports:
  - `./scripts/start_observable_container.sh`
  - The helper recreates the Jaeger v2 container (data persists via bind mounts). Health: `curl -fsS http://localhost:13133/status` → HTTP 200.
- Start the SGLang server with upstream flags only:
  - `ENABLE_TRACE=1 OTEL_TRACES_SAMPLER=always_on ./scripts/infer/start_server.sh`
  - On ready, the script prints one JSON line and writes the exact JSON atomically to `logs/provider_sessions/<ISO>_<SESSION>/start.json`.
- Run SLICE‑Bench or your workload; Jaeger queries use API v3.

## One‑liner to start Jaeger v2 manually (optional)
```
cat <<'YAML' > jaeger-config.yaml
service:
  extensions: [jaeger_storage, jaeger_query, healthcheckv2]
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [jaeger_storage_exporter]

receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

exporters:
  jaeger_storage_exporter:
    trace_storage: badger_store

processors:
  batch: {}

extensions:
  healthcheckv2:
    use_v2: true
    http:
      endpoint: 0.0.0.0:13133
      status:
        enabled: true
  jaeger_query:
    storage:
      traces: badger_store
    grpc:
      endpoint: 0.0.0.0:16685
    http:
      endpoint: 0.0.0.0:16686
  jaeger_storage:
    backends:
      badger_store:
        badger:
          directories:
            keys: /badger/keys
            values: /badger/values
          ephemeral: false
YAML

mkdir -p $HOME/sglang-observability/jaeger-v2/keys $HOME/sglang-observability/jaeger-v2/values
docker run -d --name jaeger-v2 \
  -p 16686:16686 -p 4317:4317 -p 4318:4318 -p 13133:13133 \
  -v $HOME/sglang-observability/jaeger-v2:/badger \
  -v $PWD/jaeger-config.yaml:/etc/jaeger/config.yaml:ro \
  jaegertracing/jaeger:2.11.0 \
  --config=/etc/jaeger/config.yaml
```

## Tiny OTLP probe (Python)
```
# pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider(resource=Resource.create({"service.name": "probe"}))
provider.add_span_processor(BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://127.0.0.1:4317")
))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("probe")
with tracer.start_as_current_span("smoke-span"):
    pass
provider.shutdown()
```

## Per-Run Artefacts
- `$HOME/sglang-observability/telemetry/container_runs/<RUN_ID>/`
  - `manifest.json` (authoritative per‑run manifest)
  - `logs/observability.log` (helper + server breadcrumbs)
  - `logs/provider_sessions/<ISO>_<SESSION>/start.json` (authoritative session metadata; atomic write)
  - `prometheus/` (Prometheus TSDB)
  - `jaeger/badger/{keys,values}/` (Jaeger v2 store)
  - `configs/` (Prometheus + Jaeger YAML snapshots)

## Verify (Evidence‑First)
- UI: http://localhost:16686 (service `sglang`).
- Files under your run dir:
  - `telemetry/jaeger_services.json` includes `sglang`.
  - `telemetry/jaeger_probe.json` has `result.resourceSpans` populated.
  - `telemetry/jaeger_traces.json` has `result.resourceSpans` populated after filtering.
  - All Jaeger API calls use `/api/v3/*` with nested `query.*` parameters.

## Attach to an Existing Session
- Use the read‑only helper: `scripts/infer/session_info.sh --manifest <ABS_MANIFEST_PATH>`.
- Prints the newest `logs/provider_sessions/*/start.json` verbatim. Exit codes:
  - 0: success; 2: inconsistent/missing session start.json; 3: bad pointer/manifest; 4: runtime error.
- There are no fallbacks (no log scanning, no synthesized metadata).

## Notes
- No Jaeger v1 fallback.
- Upstream SGLang flags only: `--enable-trace` and `--oltp-traces-endpoint <host:port>`.
- Jaeger v3 responses are OTLP JSON envelopes (`result.resourceSpans[...]`); the legacy `data[]` payload is gone by design.
- Tracing resource attributes: `container_run=<RUN_ID>`, `service.instance.id=<SERVER_SESSION_ID>`.
