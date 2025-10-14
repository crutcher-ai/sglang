# Tracing Ops — Jaeger v2 + Upstream SGLang

## Start order
- Ensure Jaeger v2 (host ports):
  - `./scripts/start_observable_container.sh` (starts Prometheus/node/dcgm as before, and ensures Jaeger v2 on host ports 4317/4318/16686 with Badger persistence)
  - The helper always recreates the Jaeger container with the YAML config (data persists via bind mounts), so you don't need to clean up manually.
  - Readiness probe: `curl -fsS http://localhost:13133/status` (HTTP 200 = ready).
- Start SGLang with upstream flags only:
  - `ENABLE_TRACE=1 OTEL_TRACES_SAMPLER=always_on ./scripts/infer/start_server.sh`
- Run SLICE‑Bench; it queries Jaeger v2 API v3 and writes evidence JSONs.

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

## Per-run artefacts
- The helper now emits one directory per container run: `$HOST/sglang-observability/telemetry/container_runs/<RUN_ID>/`.
- Each run folder contains the manifest, logs, Prometheus TSDB, Jaeger Badger store, textfile metrics, and a `configs/` snapshot (Prometheus + Jaeger YAML).
- Inside the helper container the same tree is mounted at `/telemetry/container_runs/<RUN_ID>/`, so SLICE-Bench can copy artefacts without chasing multiple roots.

## Verify (evidence‑first)
- UI: http://localhost:16686 (service `sglang`)
- Files under run_dir/telemetry:
  - `jaeger_services.json` includes `sglang`
  - `jaeger_probe.json` has `result.resourceSpans` populated
  - `jaeger_traces.json` has `result.resourceSpans` populated
  - All Jaeger API calls use `/api/v3/*` with nested `query.*` parameters

## Notes
- No Jaeger v1 fallback. No custom run tags in this pass.
- Upstream SGLang flags only: `--enable-trace` and `--oltp-traces-endpoint <host:port>`.
- Jaeger v3 responses are OTLP JSON envelopes (`result.resourceSpans[...]`); the legacy `data[]` payload is gone by design.
